import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Order.Sqrt
import Mathlib.Algebra.QuadraticEquation
import Mathlib.Algebra.Ratio
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.Continuous
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.SpecificDefs
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.SimpleGraph.Coloring
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Real
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Sequence
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.LinearAlgebra.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Notation
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.Theory
import Mathlib.Tactic
import Mathlib.Tactic.Polyrith
import Mathlib.Topology.ConvexHull
import Mathlib.Topology.Euclidean.Geometry.InscribedAngle

namespace impossible_to_arrange_distinct_integers_in_grid_l57_57116

theorem impossible_to_arrange_distinct_integers_in_grid :
  ¬ ∃ (f : Fin 25 × Fin 41 → ℤ),
    (∀ i j, abs (f i - f j) ≤ 16 → (i ≠ j) → (i.1 = j.1 ∨ i.2 = j.2)) ∧
    (∃ i j, i ≠ j ∧ f i = f j) := 
sorry

end impossible_to_arrange_distinct_integers_in_grid_l57_57116


namespace larger_sphere_radius_l57_57863

theorem larger_sphere_radius :
    let r := 2
    let n := 12
    let V_s := (4 / 3) * Real.pi * r^3
    let V_total := n * V_s
    let R := (V_total / ((4 / 3) * Real.pi))^(1/3)
    R = Real.cbrt (96) := 
by 
    sorry

end larger_sphere_radius_l57_57863


namespace marksmen_consistency_excellent_shots_distribution_expected_value_of_excellent_shots_l57_57646

noncomputable def variance (l : List ℝ) : ℝ := (l.map (λ x, (x - l.average) ^ 2)).sum / (l.length - 1)

def excellent_shots_probability (scores: List ℝ) (threshold: ℝ) : ℝ :=
  (scores.filter (λ x, x >= threshold)).length / scores.length

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ := (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem marksmen_consistency (A_scores B_scores: List ℝ)
  (hA : A_scores = [7, 8, 7, 9, 5, 4, 9, 10, 7, 4])
  (hB : B_scores = [9, 5, 7, 8, 7, 6, 8, 6, 7, 7]) :
  variance A_scores = 4 ∧ variance B_scores = 1.2 := by sorry

theorem excellent_shots_distribution (A_scores : List ℝ)
  (hA : A_scores = [7, 8, 7, 9, 5, 4, 9, 10, 7, 4])
  (threshold : ℝ)
  (h_threshold : threshold = 8) :
  let p := excellent_shots_probability A_scores threshold in
  binomial_probability 3 0 p = 27/125 ∧
  binomial_probability 3 1 p = 54/125 ∧
  binomial_probability 3 2 p = 36/125 ∧
  binomial_probability 3 3 p = 8/125 := by sorry

theorem expected_value_of_excellent_shots (A_scores : List ℝ)
  (hA : A_scores = [7, 8, 7, 9, 5, 4, 9, 10, 7, 4])
  (threshold : ℝ)
  (h_threshold : threshold = 8) :
  let p := excellent_shots_probability A_scores threshold in
  3 * p = 1.2 := by sorry

end marksmen_consistency_excellent_shots_distribution_expected_value_of_excellent_shots_l57_57646


namespace interval_between_segments_l57_57100

def population_size : ℕ := 800
def sample_size : ℕ := 40

theorem interval_between_segments : population_size / sample_size = 20 :=
by
  -- Insert proof here
  sorry

end interval_between_segments_l57_57100


namespace sqrt6_eq_l57_57286

theorem sqrt6_eq (r : Real) (h : r = Real.sqrt 2 + Real.sqrt 3) : Real.sqrt 6 = (r ^ 2 - 5) / 2 :=
by
  sorry

end sqrt6_eq_l57_57286


namespace apples_left_to_eat_raw_l57_57888

variable (n : ℕ) (picked : n = 85) (wormy : n / 5) (bruised : wormy + 9)

theorem apples_left_to_eat_raw (h_picked : n = 85) (h_wormy : wormy = n / 5) (h_bruised : bruised = wormy + 9) : n - (wormy + bruised) = 42 := 
sorry

end apples_left_to_eat_raw_l57_57888


namespace sum_of_other_three_numbers_l57_57199

-- Definitions for the vertices and their values
variables (a b c d e : ℤ)
variables (x y z : ℤ)

-- Hypotheses based on given conditions
hypothesis h1 : a = 1
hypothesis h2 : b = 5
hypothesis h3 : (a + b + c) = (a + c + d) ∧ (a + c + d) = (d + b + e)

-- Theorem statement to prove the sum of the other three numbers is 11
theorem sum_of_other_three_numbers : c + d + e = 11 :=
by
  sorry

end sum_of_other_three_numbers_l57_57199


namespace angle_B_measure_func_monotonic_interval_l57_57477

open Real

variables (a b c A B C : ℝ)
variables (AB BC AC : ℝ)

-- Conditions and definitions
def triangle_side_opposite (angle : ℝ) (side : ℝ) : Prop := ∀ a b c, angle = ∠a b c ∧ side = |b c|
def triangle_sine_rule (a b c : ℝ) (A B C : ℝ) : Prop := (a / sin A = b / sin B) ∧ (b / sin B = c / sin C)
def vector_dot_product_relation (a b c : ℝ) (A B : ℝ) : Prop := (c - 2 * a) * (AB * BC * cos(π - B)) = c * (BC * AC * cos C)
def func (x : ℝ) : ℝ := cos x * (a * sin x - 2 * cos x) + 1
def func_f_B_le_x (B x : ℝ) : Prop := func x ≤ func B

-- Proving the measure of angle B
theorem angle_B_measure (h1 : triangle_side_opposite A a) (h2 : triangle_side_opposite B b) (h3 : triangle_side_opposite C c)
  (h4 : vector_dot_product_relation a b c A B) (h5 : triangle_sine_rule a b c A B C) : B = π / 3 :=
sorry

-- Finding the monotonically decreasing interval
theorem func_monotonic_interval (h : ∀ x : ℝ, func_f_B_le_x (π / 3) x) (k : ℤ) :
  ∃ I : Set ℝ, I = {x | (π / 3 + k * π) ≤ x ∧ x ≤ (5 * π / 6 + k * π)} :=
sorry

end angle_B_measure_func_monotonic_interval_l57_57477


namespace circle_radius_proof_l57_57823

theorem circle_radius_proof (K M L N : Point) (R : ℝ) (O : Point) (a b : ℝ) 
  (h1 : KM ⊥ LN)
  (h2 : KL ∥ MN)
  (h3 : distance K L = 2)
  (h4 : distance M N = 2)
  (h5 : K = (-a, 0))
  (h6 : M = (a, 0))
  (h7 : L = (0, -b))
  (h8 : N = (0, b)) : R = sqrt 2 :=
sorry

end circle_radius_proof_l57_57823


namespace g_of_f_three_l57_57527

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3*x^2 + 3*x + 2

theorem g_of_f_three : g (f 3) = 1952 := by
  sorry

end g_of_f_three_l57_57527


namespace roots_square_sum_eq_l57_57343

theorem roots_square_sum_eq (p q r : ℚ) (h : 3 * Polynomial.X^3 - 4 * Polynomial.X^2 + 7 * Polynomial.X - 9 = 0) 
  (hpq : p + q + r = 4/3) (hpqr : p * q + q * r + r * p = 7/3) : p^2 + q^2 + r^2 = -26 / 9 := by
  sorry

end roots_square_sum_eq_l57_57343


namespace sum_of_odd_divisors_180_l57_57619

open Nat

theorem sum_of_odd_divisors_180 : 
  ∑ d in (finset.filter (λ n, odd n) (finset.divisors 180)), d = 78 :=
  sorry

end sum_of_odd_divisors_180_l57_57619


namespace magic_triangle_max_sum_l57_57484

open Set

def permutation_of (lst1 lst2 : List ℕ) : Prop :=
  lst1.toFinset = lst2.toFinset

theorem magic_triangle_max_sum :
  ∀ (a b c d e f : ℕ),
    permutation_of [a, b, c, d, e, f] [7, 8, 9, 10, 11, 12] →
    ∃ S, S = a + b + c ∧ S = c + d + e ∧ S = e + f + a ∧
    ∀ S', (S' = a + b + c ∧ S' = c + d + e ∧ S' = e + f + a) → S' ≤ 30 :=
by
  sorry

end magic_triangle_max_sum_l57_57484


namespace calculate_power_l57_57687

theorem calculate_power (x : ℝ) : (256:ℝ) ^ (4/5:ℝ) = 64 := by
    have h256 : (256:ℝ) = (2:ℝ) ^ 8 := by norm_num
    have h_exponent_mul : (2:ℝ) ^ (8 * (4 / 5)) = (2:ℝ) ^ 6.4 := by norm_num
    have simplified : (2:ℝ) ^ 6 = 64 := by norm_num
    sorry

end calculate_power_l57_57687


namespace equal_length_in_triangle_l57_57128

structure IsoscelesTriangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
  (AB_AC_equal : dist A B = dist A C)

variable {A B C P Q : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q]

theorem equal_length_in_triangle
  (hABC : IsoscelesTriangle A B C)
  (hP_in_ABC : P ∈ interior (convex_hull ℝ (finset.image coe (finset.of A B C))))
  (hQ_in_ABC : Q ∈ interior (convex_hull ℝ (finset.image coe (finset.of A B C))))
  (h_angle_BPC : ∠ B P C = (3 / 2) * ∠ B A C)
  (hBP_EQ_AQ : dist B P = dist A Q)
  (hAP_EQ_CQ : dist A P = dist C Q) :
  dist A P = dist P Q :=
by
  sorry

end equal_length_in_triangle_l57_57128


namespace no_odd_numbers_in_sequence_l57_57437

-- Define the sequence
def sequence (n : ℕ) : ℝ := 3.5 + 4 * n

-- Define a predicate to check if a number is odd
def is_odd (x : ℝ) : Prop := ∃ (n : ℕ), x = 2 * n + 1

-- The main theorem statement
theorem no_odd_numbers_in_sequence : ∀ (n : ℕ), ¬ is_odd (sequence n) := by
  sorry

end no_odd_numbers_in_sequence_l57_57437


namespace solve_for_x_l57_57990

theorem solve_for_x : 
  ∀ x : ℝ, (4^5 + 4^5 + 4^5 = 2^x) → x = Real.log 2 3 + 10 := 
by 
  intro x h,
  sorry

end solve_for_x_l57_57990


namespace vertex_coordinates_x_axis_intercept_length_passes_fixed_points_slope_inequality_correct_statements_l57_57746

variable {m x : ℝ}

def quadratic_function (m x : ℝ) : ℝ := 2 * m * x^2 + (1 - m) * x - 1 - m

theorem vertex_coordinates (h : m = -1) : 
    vertex (quadratic_function m x) = (1/2, 1/2) := sorry

theorem x_axis_intercept_length (h : m > 0) : 
    segment_length (intersections_x_axis (quadratic_function m x)) > 3/2 := sorry

theorem passes_fixed_points (h : m ≠ 0) :
    passes_through (quadratic_function m x) [(1, 0), (-1/2, -3/2)] := sorry

theorem slope_inequality (h : m < 0) (x1 x2 : ℝ) (hx : x1 > 1/4 ∧ x2 > 1/4 ∧ x1 ≠ x2) :
    ¬ (slope (quadratic_function m x) (x1, quadratic_function m x1) (x2, quadratic_function m x2) < 0) := sorry

theorem correct_statements : set {1, 2, 3} := 
{ n | n = 1 ∨ n = 2 ∨ n = 3 } := sorry

end vertex_coordinates_x_axis_intercept_length_passes_fixed_points_slope_inequality_correct_statements_l57_57746


namespace rectangular_coords_of_neg_theta_l57_57669

theorem rectangular_coords_of_neg_theta 
  (x y z : ℝ) 
  (rho theta phi : ℝ)
  (hx : x = 8)
  (hy : y = 6)
  (hz : z = -3)
  (h_rho : rho = Real.sqrt (x^2 + y^2 + z^2))
  (h_cos_phi : Real.cos phi = z / rho)
  (h_sin_phi : Real.sin phi = Real.sqrt (1 - (Real.cos phi)^2))
  (h_tan_theta : Real.tan theta = y / x) :
  (rho * Real.sin phi * Real.cos (-theta), rho * Real.sin phi * Real.sin (-theta), rho * Real.cos phi) = (8, -6, -3) := 
  sorry

end rectangular_coords_of_neg_theta_l57_57669


namespace angle_GYH_35_l57_57102

theorem angle_GYH_35 
  (parallel_AB_CD_EF : ∀ (a b : Line), Parallel a b)
  (angle_AXG_145 : ∠ AXG = 145) :
  ∠ GYH = 35 :=
by 
  sorry

end angle_GYH_35_l57_57102


namespace knocks_to_knicks_l57_57799

variable (knicks knacks knocks : Type)

variable (k_to_kn : 8 * knicks = 3 * knacks)
variable (kn_to_kn: 4 * knacks = 5 * knocks)

theorem knocks_to_knicks (k : knocks) : 
  30 * k = 64 * (knicks) :=
sorry

end knocks_to_knicks_l57_57799


namespace sum_of_variables_is_233_l57_57171

-- Define A, B, C, D, E, F with their corresponding values.
def A : ℤ := 13
def B : ℤ := 9
def C : ℤ := -3
def D : ℤ := -2
def E : ℕ := 165
def F : ℕ := 51

-- Define the main theorem to prove the sum of A, B, C, D, E, F equals 233.
theorem sum_of_variables_is_233 : A + B + C + D + E + F = 233 := 
by {
  -- Proof is not required according to problem statement, hence using sorry.
  sorry
}

end sum_of_variables_is_233_l57_57171


namespace range_of_a_l57_57048

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) :=
by 
sorry

end range_of_a_l57_57048


namespace find_n_lcm_l57_57214

theorem find_n_lcm (m n : ℕ) (h1 : Nat.lcm m n = 690) (h2 : n ≥ 100) (h3 : n < 1000) (h4 : ¬ (3 ∣ n)) (h5 : ¬ (2 ∣ m)) : n = 230 :=
sorry

end find_n_lcm_l57_57214


namespace monomial_sum_exponents_l57_57474

theorem monomial_sum_exponents (m n : ℕ) (h1: 2 = n) (h2: m = 3) : m - 2 * n = -1 := by
  sorry

end monomial_sum_exponents_l57_57474


namespace minimum_combined_horses_ponies_l57_57304

noncomputable def ranch_min_total (P H : ℕ) : ℕ :=
  P + H

theorem minimum_combined_horses_ponies (P H : ℕ) 
  (h1 : ∃ k : ℕ, P = 16 * k ∧ k ≥ 1)
  (h2 : H = P + 3) 
  (h3 : P = 80) 
  (h4 : H = 83) :
  ranch_min_total P H = 163 :=
by
  sorry

end minimum_combined_horses_ponies_l57_57304


namespace increasing_function_range_l57_57414

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then
  -x^2 - a*x - 5
else
  a / x

theorem increasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-3 ≤ a ∧ a ≤ -2) :=
by
  sorry

end increasing_function_range_l57_57414


namespace number_of_articles_l57_57665

-- Define main conditions
variable (N : ℕ) -- Number of articles
variable (CP SP : ℝ) -- Cost price and Selling price per article

-- Condition 1: Cost price of N articles equals the selling price of 15 articles
def condition1 : Prop := N * CP = 15 * SP

-- Condition 2: Selling price includes a 33.33% profit on cost price
def condition2 : Prop := SP = CP * 1.3333

-- Prove that the number of articles N equals 20
theorem number_of_articles (h1 : condition1 N CP SP) (h2 : condition2 CP SP) : N = 20 :=
by sorry

end number_of_articles_l57_57665


namespace number_of_valid_arrangements_l57_57557

def acts := ["Singing", "Dancing", "Acrobatics", "Skits"]
def positions := ["1", "2", "3", "4"]

def is_valid_arrangement (arr : List String) : Prop :=
  arr.length = acts.length ∧
  arr.head ≠ "Skits" ∧ -- Skits is not in 1st position
  arr.get! 1 ≠ "Acrobatics" ∧ -- Acrobatics is not in 2nd position
  arr.get! 2 ≠ "Dancing" ∧ -- Dancing is not in 3rd position
  arr.getLast! ≠ "Singing" -- Singing is not in 4th position

def valid_permutations : List (List String) :=
  List.filter is_valid_arrangement (List.permutations acts)

theorem number_of_valid_arrangements : valid_permutations.length = 9 := by
  sorry

end number_of_valid_arrangements_l57_57557


namespace ab_root_of_polynomial_l57_57177

theorem ab_root_of_polynomial 
  (a b : ℂ) (ha : a^4 + a^3 - 1 = 0) (hb : b^4 + b^3 - 1 = 0) : 
  (ab : ℂ) :
  (ab^6 + ab^4 + ab^3 - ab^2 - 1 = 0) :=
by sorry

end ab_root_of_polynomial_l57_57177


namespace hyperbola_eccentricity_l57_57781

theorem hyperbola_eccentricity (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : 4 * c^2 = 25) (h₃ : a = 1/2) : c/a = 5 :=
by
  sorry

end hyperbola_eccentricity_l57_57781


namespace circle_equation_l57_57081

theorem circle_equation 
  (a b r : ℂ)
  (C : set (ℝ × ℝ))
  (h1 : (0,0) ∈ C)
  (h2 : (4,0) ∈ C)
  (h3 : ∀ x : ℝ, (x, 1) ∈ C → false) : 
  C = {p | (p.1 - 2)^2 + (p.2 + 3/2)^2 = 25/4} :=
by 
  sorry

end circle_equation_l57_57081


namespace max_happy_monkeys_l57_57667

theorem max_happy_monkeys (pears bananas peaches mandarins : ℕ) (H : pears = 20 ∧ bananas = 30 ∧ peaches = 40 ∧ mandarins = 50) : 
  ∃ max_monkeys : ℕ, max_monkeys = 45 := 
by 
  use 45
  sorry

end max_happy_monkeys_l57_57667


namespace dan_picked_9_limes_l57_57697

variable (total_limes given_by_sara: ℕ)
-- Define the total number of limes Dan has now and the number given by Sara
axiom H1 : total_limes = 13
axiom H2 : given_by_sara = 4

-- Define the number of limes Dan picked
def limes_picked : ℕ := total_limes - given_by_sara

-- Prove that the number of limes Dan picked is 9
theorem dan_picked_9_limes : limes_picked 13 4 = 9 :=
by
  rw [H1, H2]
  sorry

end dan_picked_9_limes_l57_57697


namespace root_poly_evaluation_ne_zero_l57_57188

open Polynomial

noncomputable def has_integer_roots (f : Polynomial ℤ) : Prop :=
  ∃ α : ℂ, IsRoot f α ∧ |α| > 3 / 2

theorem root_poly_evaluation_ne_zero
  (f : Polynomial ℤ) (hf : Irreducible f) (hα : has_integer_roots f) :
  ∀ α : ℂ, IsRoot f α → |α| > 3 / 2 → ¬IsRoot f (α^3 + 1) := by
  sorry

end root_poly_evaluation_ne_zero_l57_57188


namespace total_questions_solved_l57_57921

-- Define the number of questions Taeyeon solved in a day and the number of days
def Taeyeon_questions_per_day : ℕ := 16
def Taeyeon_days : ℕ := 7

-- Define the number of questions Yura solved in a day and the number of days
def Yura_questions_per_day : ℕ := 25
def Yura_days : ℕ := 6

-- Define the total number of questions Taeyeon and Yura solved
def Total_questions_Taeyeon : ℕ := Taeyeon_questions_per_day * Taeyeon_days
def Total_questions_Yura : ℕ := Yura_questions_per_day * Yura_days
def Total_questions : ℕ := Total_questions_Taeyeon + Total_questions_Yura

-- Prove that the total number of questions solved by Taeyeon and Yura is 262
theorem total_questions_solved : Total_questions = 262 := by
  sorry

end total_questions_solved_l57_57921


namespace modified_cube_edges_l57_57312

theorem modified_cube_edges (side_length_cube : ℝ) (side_length_small_cube : ℝ) (num_corners : ℕ) :
  side_length_cube = 4 → side_length_small_cube = 1 → num_corners = 8 → 
  let original_edges := 12 in
  let new_edges_per_corner := 3 in
  let total_new_edges := original_edges in
  let total_edges := original_edges + total_new_edges in
  total_edges = 24 :=
by
  intros side_length_cube_eq_4 side_length_small_cube_eq_1 num_corners_eq_8
  let original_edges := 12
  let new_edges_per_corner := 3
  let total_new_edges := original_edges
  let total_edges := original_edges + total_new_edges
  sorry

end modified_cube_edges_l57_57312


namespace sin_inequality_l57_57032

theorem sin_inequality (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < π)
                      (h3 : 0 < x2) (h4 : x2 < π) :
  (sin x1 + sin x2) / 2 < sin ((x1 + x2) / 2) :=
sorry

end sin_inequality_l57_57032


namespace Geli_pushups_and_runs_l57_57726

def initial_pushups : ℕ := 10
def increment_pushups : ℕ := 5
def workouts_per_week : ℕ := 3
def weeks_in_a_month : ℕ := 4
def pushups_per_mile_run : ℕ := 30

def workout_days_in_month : ℕ := workouts_per_week * weeks_in_a_month

def pushups_on_day (day : ℕ) : ℕ := initial_pushups + (day - 1) * increment_pushups

def total_pushups : ℕ := (workout_days_in_month / 2) * (initial_pushups + pushups_on_day workout_days_in_month)

def one_mile_runs (total_pushups : ℕ) : ℕ := total_pushups / pushups_per_mile_run

theorem Geli_pushups_and_runs :
  total_pushups = 450 ∧ one_mile_runs total_pushups = 15 :=
by
  -- Here, we should prove total_pushups = 450 and one_mile_runs total_pushups = 15.
  sorry

end Geli_pushups_and_runs_l57_57726


namespace case_a_sticks_case_b_square_l57_57599
open Nat 

premise n12 : Nat := 12
premise sticks12_sum : Nat := (n12 * (n12 + 1)) / 2  -- Sum of first 12 natural numbers
premise length_divisibility_4 : ¬ (sticks12_sum % 4 = 0)  -- Check if sum is divisible by 4

-- Need to break at least 2 sticks to form a square
theorem case_a_sticks (h : sticks12_sum = 78) (h2 : length_divisibility_4 = true) : 
  ∃ (k : Nat), k >= 2 := sorry

premise n15 : Nat := 15
premise sticks15_sum : Nat := (n15 * (n15 + 1)) / 2  -- Sum of first 15 natural numbers
premise length_divisibility4_b : sticks15_sum % 4 = 0  -- Check if sum is divisible by 4

-- Possible to form a square without breaking any sticks
theorem case_b_square (h : sticks15_sum = 120) (h2 : length_divisibility4_b = true) : 
  ∃ (k : Nat), k = 0 := sorry

end case_a_sticks_case_b_square_l57_57599


namespace dot_product_a_b_l57_57430

-- Given unit vectors a, b, c
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (h_a : ∥a∥ = 1) (h_b : ∥b∥ = 1) (h_c : ∥c∥ = 1)
variables (h_eq : a + b + 2 • c = 0)

-- Prove that the dot product of a and b is equal to 1
theorem dot_product_a_b : ⟪a, b⟫ = 1 :=
by
  sorry

end dot_product_a_b_l57_57430


namespace students_per_van_correct_l57_57857

-- Define the conditions.
def num_vans : Nat := 6
def num_minibuses : Nat := 4
def students_per_minibus : Nat := 24
def total_students : Nat := 156

-- Define the number of students on each van is 'V'
def V : Nat := sorry 

-- State the final question/proof.
theorem students_per_van_correct : V = 10 :=
  sorry


end students_per_van_correct_l57_57857


namespace problem_statement_l57_57405

-- Definition of the conditions given in part (a)
variables (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_strict_dec : ∀ x y ∈ Icc (-1:ℝ) 0, x < y → f x > f y)
variables (α β : ℝ) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2)

-- Statement to prove the conclusion based on conditions
theorem problem_statement :
  f (sin α) < f (cos β) :=
sorry

end problem_statement_l57_57405


namespace chairs_problem_l57_57481

theorem chairs_problem (B G W : ℕ) 
  (h1 : G = 3 * B) 
  (h2 : W = B + G - 13) 
  (h3 : B + G + W = 67) : 
  B = 10 :=
by
  sorry

end chairs_problem_l57_57481


namespace expected_area_of_ant_movement_l57_57324

open BigOperators

def expected_area_of_quadrilateral (width height time : ℕ) : ℕ :=
  let horizontal_moves := time / 2
  let vertical_moves := time / 2
  let new_width := width - 2 * horizontal_moves
  let new_height := height - 2 * vertical_moves
  new_width * new_height

theorem expected_area_of_ant_movement :
  expected_area_of_quadrilateral 20 23 10 = 130 :=
by
  unfold expected_area_of_quadrilateral
  simp
  norm_num
  sorry

end expected_area_of_ant_movement_l57_57324


namespace popsicle_sticks_difference_l57_57923

def popsicle_sticks_boys (boys : ℕ) (sticks_per_boy : ℕ) : ℕ :=
  boys * sticks_per_boy

def popsicle_sticks_girls (girls : ℕ) (sticks_per_girl : ℕ) : ℕ :=
  girls * sticks_per_girl

theorem popsicle_sticks_difference : 
    popsicle_sticks_boys 10 15 - popsicle_sticks_girls 12 12 = 6 := by
  sorry

end popsicle_sticks_difference_l57_57923


namespace proof_problem_l57_57851

noncomputable theory
open_locale real

variables {a b c : ℝ} {A B C : ℝ}

def condition1 (a b c A B C : ℝ) := 2 * real.cos C * (a * real.cos B + b * real.cos A) = c
def condition2 (a : ℝ) (area : ℝ) := a = 2 ∧ area = (3 * real.sqrt 3) / 2

theorem proof_problem (h1 : condition1 a b c A B C) (h2 : condition2 a ((3 * real.sqrt 3) / 2)) :
  C = π / 3 ∧ (a + b + real.sqrt (a^2 + b^2 - 2 * a * b * real.cos C) = 5 + real.sqrt 7) :=
sorry

end proof_problem_l57_57851


namespace apples_left_to_eat_l57_57885

theorem apples_left_to_eat (total_apples : ℕ) (one_fifth_has_worms : total_apples / 5) (nine_more_bruised : one_fifth_has_worms + 9):
  let wormy_apples := total_apples / 5 in
  let bruised_apples := wormy_apples + 9 in
  let apples_left_raw := total_apples - wormy_apples - bruised_apples in
  total_apples = 85 →
  apples_left_raw = 42 :=
by
  intros
  sorry

end apples_left_to_eat_l57_57885


namespace find_temperature_l57_57928

theorem find_temperature 
  (temps : List ℤ)
  (h_len : temps.length = 8)
  (h_mean : (temps.sum / 8 : ℝ) = -0.5)
  (h_temps : temps = [-6, -3, x, -6, 2, 4, 3, 0]) : 
  x = 2 :=
by 
  sorry

end find_temperature_l57_57928


namespace degree_of_angle_C_l57_57112

theorem degree_of_angle_C 
  (A B C : ℝ) 
  (h1 : A = 4 * x) 
  (h2 : B = 4 * x) 
  (h3 : C = 7 * x) 
  (h_sum : A + B + C = 180) : 
  C = 84 := 
by 
  sorry

end degree_of_angle_C_l57_57112


namespace general_eqn_E_area_AMN_l57_57500

noncomputable def curve_parametric (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ, 2 * Real.sin θ)

noncomputable def moving_point (x₀ y₀ : ℝ) : ℝ × ℝ :=
  (x₀, (3 * y₀) / 2)

def point_on_curve_E (θ : ℝ) : Prop :=
  let (x₀, y₀) := curve_parametric θ in
  let (x, y) := moving_point x₀ y₀ in
  x^2 + y^2 = 9

def area_of_triangle_AMN (A : ℝ × ℝ) (M N : ℝ × ℝ) : ℝ :=
  abs ((fst M - fst A) * (snd N - snd A) - (fst N - fst A) * (snd M - snd A)) / 2

theorem general_eqn_E : 
  ∀ (θ : ℝ), point_on_curve_E θ :=
by
  sorry

theorem area_AMN :
  ∀ (M N : ℝ × ℝ), M ∈ set_of (λ P, fst P = snd P) → 
                   N ∈ set_of (λ P, fst P = snd P) → 
                   dist M N = 6 → 
                   let A := (5, 0) in 
                   area_of_triangle_AMN A M N = 15 * Real.sqrt 2 / 2 :=
by
  sorry

end general_eqn_E_area_AMN_l57_57500


namespace intersection_point_l57_57371

theorem intersection_point (k : ℚ) :
  (∃ x y : ℚ, x + k * y = 0 ∧ 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0) ↔ (k = -1/2) :=
by sorry

end intersection_point_l57_57371


namespace first_player_wins_l57_57570

def number := 328
def divisors := {1, 2, 4, 8, 41, 82, 164, 328}

def condition1 : number = 328 := rfl
def condition2 : ∀ x ∈ divisors, x ∣ number := by sorry
def condition3 : ∀ x y ∈ divisors, (x ∣ y → x = y) := by sorry
def condition4 : ∀ x ∈ divisors, x = 328 ↔ x = number := by sorry

theorem first_player_wins (number := 328) (divisors := {1, 2, 4, 8, 41, 82, 164, 328})
  (h1 : number = 328) (h2 : ∀ x ∈ divisors, x ∣ number) 
  (h3 : ∀ x y ∈ divisors, (x ∣ y → x = y)) 
  (h4 : ∀ x ∈ divisors, x = 328 ↔ x = number) : 
  "first player has a winning strategy" := by sorry

end first_player_wins_l57_57570


namespace girls_from_clay_is_30_l57_57961

-- Definitions for the given conditions
def total_students : ℕ := 150
def total_boys : ℕ := 90
def total_girls : ℕ := 60
def students_jonas : ℕ := 50
def students_clay : ℕ := 70
def students_hart : ℕ := 30
def boys_jonas : ℕ := 25

-- Theorem to prove that the number of girls from Clay Middle School is 30
theorem girls_from_clay_is_30 
  (h1 : total_students = 150)
  (h2 : total_boys = 90)
  (h3 : total_girls = 60)
  (h4 : students_jonas = 50)
  (h5 : students_clay = 70)
  (h6 : students_hart = 30)
  (h7 : boys_jonas = 25) : 
  ∃ girls_clay : ℕ, girls_clay = 30 :=
by 
  sorry

end girls_from_clay_is_30_l57_57961


namespace hamburgers_left_over_l57_57671

theorem hamburgers_left_over (total_hamburgers served_hamburgers : ℕ) (h1 : total_hamburgers = 9) (h2 : served_hamburgers = 3) :
    total_hamburgers - served_hamburgers = 6 := by
  sorry

end hamburgers_left_over_l57_57671


namespace dans_age_present_l57_57355

theorem dans_age_present : ∃ x : ℕ, (x + 16 = 4 * (x - 8)) ∧ x = 16 :=
by
  existsi 16
  split
  · simp
  · refl

end dans_age_present_l57_57355


namespace paper_holes_after_unfolding_l57_57345

noncomputable def rectangular_paper_folding
    (initial_width : ℝ) (initial_height : ℝ)
    (fold1 : ℝ × ℝ → ℝ × ℝ) (fold2 : ℝ × ℝ → ℝ × ℝ)
    (fold3 : ℝ × ℝ → ℝ × ℝ) (punch_hole : ℝ × ℝ → ℝ × ℝ) : Prop :=
  ∃ unfolded_holes : ℕ, unfolded_holes = 8 ∧
    ∀ (w h : ℝ), (initial_width = 12 ∧ initial_height = 8) →
    fold1 (initial_width, initial_height) = (12, 4) →
    fold2 (12, 4) = (6, 4) →
    fold3 (6, 4) = (6, 2) →
    punch_hole (6, 2) = (6, 2) →
    unfolded_holes = 8

theorem paper_holes_after_unfolding :
  rectangular_paper_folding 12 8 
    (λ p, (p.1, p.2 / 2)) -- Fold bottom to top
    (λ p, (p.1 / 2, p.2)) -- Fold right to left
    (λ p, (p.1, p.2 / 2)) -- Fold top to bottom
    (λ p, p) := -- Punch hole near center
begin
  sorry
end

end paper_holes_after_unfolding_l57_57345


namespace sum_x_y_eq_2_l57_57528

open Real

theorem sum_x_y_eq_2 (x y : ℝ) (h : x - 1 = 1 - y) : x + y = 2 :=
by
  sorry

end sum_x_y_eq_2_l57_57528


namespace champion_certificate_awarded_to_class_b_l57_57089

noncomputable def class_a_scores : List ℕ := [87, 100, 96, 120, 97]
noncomputable def class_b_scores : List ℕ := [100, 95, 110, 91, 104]

noncomputable def median (l : List ℕ) : ℕ := l.nth (l.length / 2).get_or_else 0

def variance (l : List ℕ) : ℚ :=
  let mean := (l.sum : ℚ) / l.length
  (l.map (λ x => (x - mean) ^ 2)).sum / l.length

theorem champion_certificate_awarded_to_class_b :
  median (class_a_scores.sort) = 97 ∧ 
  median (class_b_scores.sort) = 100 ∧ 
  variance class_a_scores = 118.8 ∧ 
  variance class_b_scores = 44.4 ∧ 
  variance class_a_scores > variance class_b_scores ∧ 
  ∃ l: List ℕ, l = class_b_scores := sorry

end champion_certificate_awarded_to_class_b_l57_57089


namespace nth_smallest_d0_perfect_square_l57_57664

theorem nth_smallest_d0_perfect_square (n : ℕ) : 
  ∃ (d_0 : ℕ), (∃ v : ℕ, ∀ t : ℝ, (2 * t * t + d_0 = v * t) ∧ (∃ k : ℕ, v = k ∧ k * k = v * v)) 
               ∧ d_0 = 4^(n - 1) := 
by sorry

end nth_smallest_d0_perfect_square_l57_57664


namespace complete_the_square_l57_57254

theorem complete_the_square (x : ℝ) (h : x^2 - 8 * x - 1 = 0) : (x - 4)^2 = 17 :=
by
  -- proof steps would go here, but we use sorry for now
  sorry

end complete_the_square_l57_57254


namespace cube_vertex6_edge_ends_l57_57503

theorem cube_vertex6_edge_ends :
  ∃ (a b c : ℕ), {a, b, c} = {2, 3, 5} ∨ {a, b, c} = {3, 5, 7} ∨ {a, b, c} = {2, 3, 7} :=
by
  -- Proof is omitted
  sorry

end cube_vertex6_edge_ends_l57_57503


namespace negation_if_positive_then_square_positive_l57_57216

theorem negation_if_positive_then_square_positive :
  (¬ (∀ x : ℝ, x > 0 → x^2 > 0)) ↔ (∀ x : ℝ, x ≤ 0 → x^2 ≤ 0) :=
by
  sorry

end negation_if_positive_then_square_positive_l57_57216


namespace impossible_coins_l57_57158

theorem impossible_coins (p_1 p_2 : ℝ) 
  (h1 : (1 - p_1) * (1 - p_2) = p_1 * p_2)
  (h2 : p_1 * (1 - p_2) + p_2 * (1 - p_1) = p_1 * p_2) : False := 
sorry

end impossible_coins_l57_57158


namespace sum_powers_odd_mod_l57_57167

noncomputable theory

-- Lean 4 statement
theorem sum_powers_odd_mod (n : ℕ) :
  (∑ k in finset.range n, ((2^k - 1) ^ (2^k - 1))) % (2 ^ n) = 0 ∧ 
  (∑ k in finset.range n, ((2^k - 1) ^ (2^k - 1))) % (2 ^ (n + 1)) ≠ 0 := 
sorry

end sum_powers_odd_mod_l57_57167


namespace sequence_a_2005_eq_1_l57_57502

theorem sequence_a_2005_eq_1 :
  ∀ (a : ℕ → ℤ),
  a 1 = 1 →
  a 2 = 2 →
  (∀ n, n ≥ 3 → a n = a (n - 1) - a (n - 2)) →
  a 2005 = 1 :=
by {
  intros a h1 h2 h_rec,
  sorry
}

end sequence_a_2005_eq_1_l57_57502


namespace min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l57_57580

-- Part (a): For n = 12:
theorem min_sticks_to_break_for_square_12 : ∀ (n : ℕ), n = 12 → 
  (∃ (sticks : Finset ℕ), sticks.card = 12 ∧ sticks.sum id = 78 ∧ (¬ (78 % 4 = 0) → 
  ∃ (b : ℕ), b = 2)) := 
by sorry

-- Part (b): For n = 15:
theorem can_form_square_without_breaking_15 : ∀ (n : ℕ), n = 15 → 
  (∃ (sticks : Finset ℕ), sticks.card = 15 ∧ sticks.sum id = 120 ∧ (120 % 4 = 0)) :=
by sorry

end min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l57_57580


namespace quadrilateral_is_isosceles_trapezoid_l57_57908

theorem quadrilateral_is_isosceles_trapezoid
    (ABCD : Type*)
    [quadrilateral ABCD]
    {ω : circle}
    {I : point}
    (inscribed : inscribed_in_quadrilateral ω ABCD)
    (center : center_of ω = I) :
    ∃ (AB CD : line), parallel AB CD ∧ length_side AD = length_side BC := 
sorry

end quadrilateral_is_isosceles_trapezoid_l57_57908


namespace sum_c_d_l57_57076

def g (x : ℝ) (c d : ℝ) : ℝ := (x - 5) / (x^2 + c*x + d)

theorem sum_c_d (c d : ℝ) (h1 : ∀ (x : ℝ), x = 2 → x^2 + c * x + d = 0)
                        (h2 : ∀ (x : ℝ), x = -3 → x^2 + c * x + d = 0) :
  c + d = -5 :=
sorry

end sum_c_d_l57_57076


namespace average_return_speed_l57_57270

theorem average_return_speed :
  let d := 120
  let r := 50
  let t_total := 5.5
  let t_to_s := d / r
  let t_back := t_total - t_to_s
  let avg_rate := d / t_back
  avg_rate ≈ 38.71 :=
by
  let d := 120
  let r := 50
  let t_total := 5.5
  let t_to_s := d / r
  let t_back := t_total - t_to_s
  let avg_rate := d / t_back
  exact sorry

end average_return_speed_l57_57270


namespace square_with_12_sticks_square_with_15_sticks_l57_57587

-- Definitions for problem conditions
def sum_of_first_n_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def can_form_square (total_length : ℕ) : Prop :=
  total_length % 4 = 0

-- Given n = 12, check if breaking 2 sticks is required to form a square
theorem square_with_12_sticks : (n = 12) → ¬ can_form_square (sum_of_first_n_natural_numbers 12) → true :=
by
  intros
  sorry

-- Given n = 15, check if it is possible to form a square without breaking any sticks
theorem square_with_15_sticks : (n = 15) → can_form_square (sum_of_first_n_natural_numbers 15) → true :=
by
  intros
  sorry

end square_with_12_sticks_square_with_15_sticks_l57_57587


namespace three_digit_numbers_divisible_by_9_l57_57445

theorem three_digit_numbers_divisible_by_9 : 
  let smallest := 108
  let largest := 999
  let common_diff := 9
  -- Using the nth-term formula for an arithmetic sequence
  -- nth term: l = a + (n-1) * d
  -- For l = 999, a = 108, d = 9
  -- (999 = 108 + (n-1) * 9) -> (n-1) = 99 -> n = 100
  -- Hence, the number of such terms (3-digit numbers) in the sequence is 100.
  ∃ n, n = 100 ∧ (largest = smallest + (n-1) * common_diff)
by {
  let smallest := 108
  let largest := 999
  let common_diff := 9
  use 100
  sorry
}

end three_digit_numbers_divisible_by_9_l57_57445


namespace ellipse_equation_a_slopes_sum_l57_57734

-- Definitions and conditions
def ellipse (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
def is_focus (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
def is_point_on_line (x y : ℝ) (k : ℝ) : Prop := (y + 1 = k * (x - 2))
def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)
def point_P (x y : ℝ) : Prop := (x = 2 ∧ y = -1)
def point_A (x y : ℝ) (b : ℝ) : Prop := (x = 0 ∧ y = b)
def line_not_passing_point_A (k : ℝ) (b : ℝ) : Prop := 
  ∀ (x y : ℝ), (is_point_on_line x y k) → (point_A x y b) → False

theorem ellipse_equation_a (x y a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a = 2) (h4 : b = 1) :
  ellipse x y a b ↔ (x^2 / 4 + y^2 = 1) := by
  sorry

theorem slopes_sum (a b : ℝ) (k : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : point_P 2 (-1)) 
  (h4 : point_A 0 b b) 
  (h5 : line_not_passing_point_A k b) 
  (h6 : ellipse 2 (-1) a b) 
  (h7 : ellipse 0 b a b)
  (h8 : slope 0 b 2 (-1) = k) :
  slope 0 b 2 (-1) + slope 0 b 2 (-1) = -1 := by
  sorry

end ellipse_equation_a_slopes_sum_l57_57734


namespace miniature_tower_height_calc_l57_57890

noncomputable def actual_tower_height : ℝ := 60
noncomputable def actual_dome_volume : ℝ := 150000
noncomputable def miniature_dome_volume : ℝ := 0.15

theorem miniature_tower_height_calc : 
  let ratio := actual_dome_volume / miniature_dome_volume in
  let scale_factor := ratio ^ (1 / 3 : ℝ) in
  let miniature_tower_height := actual_tower_height / scale_factor in
  miniature_tower_height = 0.6 :=
by
  let ratio := actual_dome_volume / miniature_dome_volume 
  let scale_factor := ratio ^ (1 / 3 : ℝ)
  let miniature_tower_height := actual_tower_height / scale_factor
  have h1 : ratio = 1000000 := by sorry
  have h2 : scale_factor = 100 := by sorry
  have h3 : miniature_tower_height = 0.6 := by sorry
  exact h3

end miniature_tower_height_calc_l57_57890


namespace range_of_a_l57_57021

variable (a x : ℝ)

def p := sqrt (2 * x - 1) ≤ 1
def q := (x - a) * (x - (a + 1)) ≤ 0 

theorem range_of_a (h : ∀ {x}, q x → p x) (h' : ¬ ∀ {x}, p x → q x) : 0 ≤ a ∧ a ≤ 1 / 2 :=
sorry

end range_of_a_l57_57021


namespace min_eccentricity_sum_l57_57788

def circle_O1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 16
def circle_O2 (x y r : ℝ) : Prop := x^2 + y^2 = r^2 ∧ 0 < r ∧ r < 2

def moving_circle_tangent (e1 e2 : ℝ) (r : ℝ) : Prop :=
  e1 = 2 / (4 - r) ∧ e2 = 2 / (4 + r)

theorem min_eccentricity_sum : ∃ (e1 e2 : ℝ) (r : ℝ), 
  circle_O1 x y ∧ circle_O2 x y r ∧ moving_circle_tangent e1 e2 r ∧
    e1 > e2 ∧ (e1 + 2 * e2) = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end min_eccentricity_sum_l57_57788


namespace cos_b_plus_sqrt_two_cos_c_range_l57_57519

theorem cos_b_plus_sqrt_two_cos_c_range (A B C : ℝ)
  (h1 : sin A = √2 / 2)
  (h2 : 0 < C) (h3 : C < π) 
  : (∀ B C : ℝ, 0 < B ∧ B + C = π - A) →
    (cos B + √2 * cos C) ∈ set.Icc 0 1 ∪ set.Ioo 2 (√5) :=
begin
  sorry
end

end cos_b_plus_sqrt_two_cos_c_range_l57_57519


namespace f_nonneg_3p_l57_57721

def t (m : ℤ) : ℤ :=
  if (m + 1) % 3 = 0 then 1 else if (m + 2) % 3 = 0 then 2 else 3

def f : ℤ → ℤ
| -1 := 0
|  0 := 1
|  1 := -1
| (2^n + m) :=
  if h : 2^n > m then
    f (2^n - t m) - f m
  else
    0 -- definition partiality, noncomputable part is handled below

theorem f_nonneg_3p (p : ℤ) (h : p ≥ 0) : f (3 * p) ≥ 0 :=
by
  sorry

end f_nonneg_3p_l57_57721


namespace distance_D_D_l57_57610

def point := ℝ × ℝ

def reflect_y (p : point) : point := (-p.1, p.2)

def distance (p1 p2 : point) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2))

theorem distance_D_D' : 
  let D : point := (3, 2)
  let D' := reflect_y D
  distance D D' = 6 :=
by
  let D : point := (3, 2)
  let D' := reflect_y D
  show distance D D' = 6
  sorry

end distance_D_D_l57_57610


namespace reflections_of_orthocenter_concyclic_l57_57169

theorem reflections_of_orthocenter_concyclic 
  {A B C H H' H'' H''' : Type*} 
  (h_orthocenter : orthocenter A B C H)
  (h_reflection_bc : reflection H BC H')
  (h_reflection_ca : reflection H CA H'')
  (h_reflection_ab : reflection H AB H''')
  (h_circumcircle : circumcircle A B C) : 
  lies_on_circumcircle H' A B C ∧ 
  lies_on_circumcircle H'' A B C ∧ 
  lies_on_circumcircle H''' A B C := 
sorry

end reflections_of_orthocenter_concyclic_l57_57169


namespace total_flea_count_l57_57378

theorem total_flea_count : 
  let gertrude_fleas := 10 in
  let olive_fleas := gertrude_fleas / 2 in
  let maud_fleas := olive_fleas * 5 in
  gertrude_fleas + olive_fleas + maud_fleas = 40 :=
by
  sorry

end total_flea_count_l57_57378


namespace perfect_square_trinomial_l57_57406

-- Define the conditions
theorem perfect_square_trinomial (k : ℤ) : 
  ∃ (a b : ℤ), (a^2 = 1 ∧ b^2 = 16 ∧ (x^2 + k * x * y + 16 * y^2 = (a * x + b * y)^2)) ↔ (k = 8 ∨ k = -8) :=
by
  sorry

end perfect_square_trinomial_l57_57406


namespace cash_discount_percentage_correct_l57_57293

def cost_price := 1.0
def listed_price := cost_price * 1.5
def selling_price := cost_price * 40
def total_cost_price := cost_price * 45
def expected_selling_price := total_cost_price * 1.2
def cash_discount := expected_selling_price - selling_price
def cash_discount_percentage := (cash_discount / expected_selling_price) * 100

theorem cash_discount_percentage_correct :
  abs (cash_discount_percentage - 25.93) < 0.01 := sorry

end cash_discount_percentage_correct_l57_57293


namespace find_a_value_l57_57808

-- Define the problem conditions
def line_eq_condition (a : ℝ) := ∃ (k : ℝ), k = 1 ∧ k = -a / (2 * a - 3)

-- Define the proof goal
theorem find_a_value (a : ℝ) (h : line_eq_condition a) : a = 1 :=
by sorry

end find_a_value_l57_57808


namespace find_ordered_pair_l57_57695

variables {x y : ℝ}
def u : ℝ × ℝ × ℝ := (3, x, -9)
def v : ℝ × ℝ × ℝ := (6, 5, y)

theorem find_ordered_pair (h : u.1 × v.2 - u.2 × v.1 = 0 ∧ u.2 × v.3 - u.3 × v.2 = 0 ∧ u.3 × v.1 - u.1 × v.3 = 0) :
  (x, y) = (5/2, -18) :=
sorry

end find_ordered_pair_l57_57695


namespace impossible_coins_l57_57161

theorem impossible_coins (p1 p2 : ℝ) : 
  ¬ ((1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end impossible_coins_l57_57161


namespace smallest_three_digit_number_l57_57985

/-- 
  Prove that the smallest three-digit number
  that can be formed by using the five different
  numbers 3, 0, 2, 5, and 7 only once, is 203.
-/
theorem smallest_three_digit_number : 
  ∃ (n : ℕ), 
  n = 203 ∧ 
  (∀ (x y z : ℕ), 
    x ∈ {3, 0, 2, 5, 7} ∧ 
    y ∈ {3, 0, 2, 5, 7} ∧ 
    z ∈ {3, 0, 2, 5, 7} ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z →
    (∀ m, m = x * 100 + y * 10 + z → m ≥ 203)) ∧
  n < 300 :=
by
  -- Place the proof here
  sorry

end smallest_three_digit_number_l57_57985


namespace girls_collected_more_mushrooms_l57_57113

variables (N I A V : ℝ)

theorem girls_collected_more_mushrooms 
    (h1 : N > I) 
    (h2 : N > A) 
    (h3 : N > V) 
    (h4 : I ≤ N) 
    (h5 : I ≤ A) 
    (h6 : I ≤ V) 
    (h7 : A > V) : 
    N + I > A + V := 
by {
    sorry
}

end girls_collected_more_mushrooms_l57_57113


namespace iron_wire_left_l57_57300

-- Given conditions as variables
variable (initial_usage : ℚ) (additional_usage : ℚ)

-- Conditions as hypotheses
def conditions := initial_usage = 2 / 9 ∧ additional_usage = 3 / 9

-- The goal to prove
theorem iron_wire_left (h : conditions initial_usage additional_usage):
  1 - initial_usage - additional_usage = 4 / 9 :=
by
  -- Insert proof here
  sorry

end iron_wire_left_l57_57300


namespace exists_2018_relatively_consistent_numbers_l57_57971

def relatively_consistent (a b : ℕ) : Prop :=
  let (x, y) := if a < b then (a, b) else (b, a)
  in ∃ S : Finset ℕ, S ⊆ (Finset.divisors x) ∧ S.sum = y

theorem exists_2018_relatively_consistent_numbers :
  ∃ (seq : Finset ℕ), seq.card = 2018 ∧ (∀ a b ∈ seq, a ≠ b → relatively_consistent a b) :=
sorry

end exists_2018_relatively_consistent_numbers_l57_57971


namespace elaine_spent_20_percent_on_rent_last_year_l57_57862

variable (E P : ℝ)

theorem elaine_spent_20_percent_on_rent_last_year
  (h1 : 1.25 * E)
  (h2 : 0.30 * 1.25 * E)
  (h3 : 0.375 * E = 1.875 * (P / 100) * E) :
  P = 20 := 
by
  sorry

end elaine_spent_20_percent_on_rent_last_year_l57_57862


namespace find_p_minus_q_l57_57044

theorem find_p_minus_q (p q : ℝ) (h : ∀ x, x^2 - 6 * x + q = 0 ↔ (x - p)^2 = 7) : p - q = 1 :=
sorry

end find_p_minus_q_l57_57044


namespace correct_word_is_tradition_l57_57237

-- Definitions of the words according to the problem conditions
def tradition : String := "custom, traditional practice"
def balance : String := "equilibrium"
def concern : String := "worry, care about"
def relationship : String := "relation"

-- The sentence to be filled
def sentence (word : String) : String :=
"There’s a " ++ word ++ " in our office that when it’s somebody’s birthday, they bring in a cake for us all to share."

-- The proof problem statement
theorem correct_word_is_tradition :
  ∀ word, (word ≠ tradition) → (sentence word ≠ "There’s a tradition in our office that when it’s somebody’s birthday, they bring in a cake for us all to share.") :=
by sorry

end correct_word_is_tradition_l57_57237


namespace count_three_digit_numbers_divisible_by_9_l57_57459

theorem count_three_digit_numbers_divisible_by_9 : 
  (finset.filter (λ n, (n % 9 = 0)) (finset.range 1000).filter (λ n, 100 ≤ n)).card = 100 :=
by
  sorry

end count_three_digit_numbers_divisible_by_9_l57_57459


namespace Joe_first_lift_weight_l57_57832

variable (F S : ℝ)

theorem Joe_first_lift_weight (h1 : F + S = 600) (h2 : 2 * F = S + 300) : F = 300 := 
sorry

end Joe_first_lift_weight_l57_57832


namespace area_of_triangle_value_of_angle_C_l57_57514

-- Define the problem constraints and prove the statements
noncomputable def triangle_problem_part1 (a b c : ℝ) (A B C : ℝ) : Prop :=
  B = 150 ∧
  a = sqrt 3 * c ∧
  b = 2 * sqrt 7 ∧
  (1/2) * a * c * (Float.sin (Float.toRadians B)) = sqrt 3

noncomputable def triangle_problem_part2 (A B C : ℝ) : Prop :=
  B = 150 ∧
  (∀ A C : ℝ, sin A + sqrt 3 * sin C = sqrt 2 / 2) ∧
  C = 15

theorem area_of_triangle {a b c A B C : ℝ} :
  triangle_problem_part1 a b c A B C :=
by {
  sorry
}

theorem value_of_angle_C {A B C : ℝ} :
  triangle_problem_part2 A B C :=
by {
  sorry
}

end area_of_triangle_value_of_angle_C_l57_57514


namespace inequality_solution_l57_57723

theorem inequality_solution (x : ℝ) (h : 0 < x) : x^3 - 9*x^2 + 52*x > 0 := 
sorry

end inequality_solution_l57_57723


namespace range_of_m_l57_57051

noncomputable def f (a x : ℝ) : ℝ :=
  ((1 - a) / 2) * x^2 + a * x - Real.log x

theorem range_of_m (a : ℝ) (x₁ x₂ m : ℝ) :
  a ∈ Ioo (4 : ℝ) 5 →
  x₁ ∈ Icc (1 : ℝ) 2 →
  x₂ ∈ Icc (1 : ℝ) 2 →
  ((a - 1) / 2) * m + Real.log 2 > |f a x₁ - f a x₂| →
  m ≥ 1 / 2 :=
sorry

end range_of_m_l57_57051


namespace no_arithmetic_mean_l57_57894

def eight_thirteen : ℚ := 8 / 13
def eleven_seventeen : ℚ := 11 / 17
def five_eight : ℚ := 5 / 8

-- Define the function to calculate the arithmetic mean of two rational numbers
def arithmetic_mean (a b : ℚ) : ℚ :=
(a + b) / 2

-- The theorem statement
theorem no_arithmetic_mean :
  eight_thirteen ≠ arithmetic_mean eleven_seventeen five_eight ∧
  eleven_seventeen ≠ arithmetic_mean eight_thirteen five_eight ∧
  five_eight ≠ arithmetic_mean eight_thirteen eleven_seventeen :=
sorry

end no_arithmetic_mean_l57_57894


namespace perimeter_of_equilateral_triangle_l57_57197

-- Defining the conditions
def area_eq_twice_side (s : ℝ) : Prop :=
  (s^2 * Real.sqrt 3) / 4 = 2 * s

-- Defining the proof problem
theorem perimeter_of_equilateral_triangle (s : ℝ) (h : area_eq_twice_side s) : 
  3 * s = 8 * Real.sqrt 3 :=
sorry

end perimeter_of_equilateral_triangle_l57_57197


namespace find_angle_A_determine_triangle_shape_l57_57478

noncomputable def angle_A (A : ℝ) (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 7 / 2 ∧ m = (Real.cos (A / 2)^2, Real.cos (2 * A)) ∧ 
  n = (4, -1)

theorem find_angle_A : 
  ∃ A : ℝ,  (0 < A ∧ A < Real.pi) ∧ angle_A A (Real.cos (A / 2)^2, Real.cos (2 * A)) (4, -1) 
  := sorry

noncomputable def triangle_shape (a b c : ℝ) (A : ℝ) : Prop :=
  a = Real.sqrt 3 ∧ a^2 = b^2 + c^2 - b * c * Real.cos (A)

theorem determine_triangle_shape :
  ∀ (b c : ℝ), (b * c ≤ 3) → triangle_shape (Real.sqrt 3) b c (Real.pi / 3) →
  (b = Real.sqrt 3 ∧ c = Real.sqrt 3)
  := sorry


end find_angle_A_determine_triangle_shape_l57_57478


namespace number_of_mismatching_socks_l57_57184

def SteveTotalSocks := 48
def StevePairsMatchingSocks := 11

theorem number_of_mismatching_socks :
  SteveTotalSocks - (StevePairsMatchingSocks * 2) = 26 := by
  sorry

end number_of_mismatching_socks_l57_57184


namespace largest_equal_cost_l57_57929

-- Define the decimal cost calculation function.
def decimal_cost (n : ℕ) : ℕ :=
  (n.digits 10).sum (λ d, d + 2)

-- Define the binary cost calculation function.
def binary_cost (n : ℕ) : ℕ :=
  (n.digits 2).sum (λ d, d + 2)

-- Prove the largest integer less than 2000 with equal cost in both options.
theorem largest_equal_cost : ∃ (n : ℕ), n < 2000 ∧ decimal_cost n = binary_cost n ∧ ∀ (m : ℕ), m < 2000 → decimal_cost m = binary_cost m → m <= n := 
  sorry

end largest_equal_cost_l57_57929


namespace part_I_part_II_l57_57055

noncomputable theory

open Real

section
variable (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (a + 1 / a) * log x - x + 1 / x

theorem part_I (h : 0 < a) :
  (f' x = (a + 1 / a) / x - 1 - 1 / x^2) = 0 → x = (a + 1 / a + sqrt ((a + 1 / a)^2 - 4)) / 2 → 
  (a ≥ 1) :=
sorry

theorem part_II (a : ℝ) 
  (h1 : 1 < a) (h2 : a ≤ exp 1) (x₁ x₂ : ℝ) 
  (h₃ : 0 < x₁ ∧ x₁ < 1) (h₄ : 1 < x₂) :
  ∃ Mₐ, Mₐ = f a x₂ - f a x₁ ∧ (Mₐ = f e x₂ - f e x₁) :=
sorry
end

end part_I_part_II_l57_57055


namespace binomial_constant_term_l57_57496

theorem binomial_constant_term : 
  let x : ℝ := 1 in
  (∑ k in finset.range (6 + 1), (nat.choose 6 k) * 6^k * x^(6 - 2*k)) = 4320 :=
by 
  sorry

end binomial_constant_term_l57_57496


namespace first_ellipse_standard_eq_second_ellipse_standard_eq_l57_57362

theorem first_ellipse_standard_eq (a b c : ℝ) (h1 : 2 * a = 12) (h2 : c / a = 2 / 3)
  (h3 : c ^ 2 = a ^ 2 - b ^ 2) :
  (a = 6) ∧ (b = sqrt 20) → 
  ∃ (a2 b2 : ℝ), 
  a2 = 36 ∧ b2 = 20 ∧ 
  ∀ (x y : ℝ), x^2 / a2 + y^2 / b2 = 1 :=
begin
  sorry
end

theorem second_ellipse_standard_eq (A B : ℝ × ℝ) (m n : ℝ)
  (h : A = (sqrt 6 / 3, sqrt 3) ∧ B = (2 * sqrt 2 / 3, 1))
  (h1 : m * (sqrt 6 / 3) ^ 2 + n * (sqrt 3) ^ 2 = 1)
  (h2 : m * (2 * sqrt 2 / 3) ^ 2 + n * 1 ^ 2 = 1) :
  (m = 1) ∧ (n = 1 / 9) →
  ∃ (a2 b2 : ℝ), 
  a2 = 1 ∧ b2 = 9 ∧ 
  ∀ (x y : ℝ), x^2 / a2 + y^2 / b2 = 1 :=
begin
  sorry
end

end first_ellipse_standard_eq_second_ellipse_standard_eq_l57_57362


namespace figure_cut_successfully_l57_57354

-- Defining the problem statement based on the given conditions and required proof
def rectangular_sheet (sheet : Type) := ∃ (length width : ℝ), length > 0 ∧ width > 0
def cutout_possible (sheet : Type) (fig : Type) := 
  rectangular_sheet sheet ∧  -- Condition that the sheet is rectangular
  -- Conditions that instructions are followed
  (∃ (angle : ℝ), angle = 180 ∧ 
   -- Abstracting specific cutting procedure into a placeholder function cut_instructions
   ∃ (cut_instructions : sheet → sheet → Prop), cut_instructions = sorry)

theorem figure_cut_successfully :
  ∀ (sheet fig : Type), 
  cutout_possible sheet fig → -- Given conditions
  fig -- Answer to be proved
:= by 
sory -- Proof to be filled

end figure_cut_successfully_l57_57354


namespace unique_painted_cube_l57_57291

theorem unique_painted_cube : 
  ∀ (cube : Cube), 
  (paint cube yellow 1) ∧ (paint cube red 3) ∧ (paint cube blue 2) 
  ∧ (rotationally_equivalent cube)
  → unique_solution cube := 
by
  sorry

end unique_painted_cube_l57_57291


namespace sum_solutions_less_equal_30_l57_57986

def satisfies_congruence (x : ℕ) : Prop :=
  7 * (5 * x - 3) % 10 = 35 % 10

noncomputable def sum_of_solutions : ℕ :=
  (∑ x in finset.range 31, if satisfies_congruence x then x else 0)

theorem sum_solutions_less_equal_30 :
  sum_of_solutions = 225 := by
  sorry

end sum_solutions_less_equal_30_l57_57986


namespace meaningful_fraction_l57_57609

theorem meaningful_fraction (x : ℝ) : (∃ y, y = (1 / (x - 2))) ↔ x ≠ 2 :=
by
  sorry

end meaningful_fraction_l57_57609


namespace point_O_on_line_DK_l57_57290

variables {A B C O K M N B1 D K : Type}
variables [inc_circle : inscribed_circle A B C O]
variables [tangent_points : tangency_points A B C K M N]
variables [median : is_median B B1 A C]
variables [intersection : intersects_median_and_mn BB1 D MN]

theorem point_O_on_line_DK
  (h1 : circle_center O)
  (h2 : tangent_to AC K A B C A1 C1)
  (h3 : line_through D)
  (h4 : D_on_MN MN B1 D)
  (h5 : right_angles)
  (h6 : parallel_to A B C L AC)
  (h7 : congruent_triangles):
  on_line DK O :=
sorry

end point_O_on_line_DK_l57_57290


namespace circle_C_l57_57043

noncomputable def circle_equation : Type :=
  {a : ℝ // a > 0} → {r : ℝ // r > 0} → 
  (M : ℝ × ℝ) → 
  (D : ℝ) → Prop

axiom point_M : (0, Real.sqrt 3)
axiom line_2x_y_plus_1_zero : 2 * x - y + 1 = 0
axiom distance_from_center_to_line : Real.sqrt(((a,0).dist (line_2x_y_plus_1_zero)) = 3 * Real.sqrt 5 / 5

-- We need to state the problem with provided conditions that prove the goal
def find_equation_circle (a : ℝ) (r : ℝ) : Prop :=
  (a > 0) ∧ 
  ((0 - a)^2 + (Real.sqrt 3)^2 = r^2) ∧
  (Real.abs(2 * a + 1) / Real.sqrt 5 = 3 * Real.sqrt 5 / 5) →
  ((x - 1)^2 + y^2 = 4)

-- The target problem statement
theorem circle_C (a : ℝ) (r : ℝ) (p : find_equation_circle a r) : 
  ((x - a) ^ 2 + y ^ 2 = r ^ 2) :=
sorry

end circle_C_l57_57043


namespace sin_750_eq_half_l57_57342

theorem sin_750_eq_half : sin (750 * real.pi / 180) = 1 / 2 := by
  -- Use trigonometric identity to simplify angle
  -- 750 degrees is equivalent to (750 - 720) degrees because 720 = 360 * 2
  have h1 : 750 * real.pi / 180 = 30 * real.pi / 180, by
    norm_num,
    -- 750 - 720 = 30
    rw [mul_div_cancel_left 750 (ne_of_gt (by norm_num : (0 : ℝ) < 180))],
    norm_num,
  -- Use known value of sin 30 degrees which is 1/2
  rw [h1, sin_pi_div_six],
  -- Sin of 30 degrees is 1/2
  exact rfl

end sin_750_eq_half_l57_57342


namespace count_positive_3_digit_numbers_divisible_by_9_l57_57450

-- Conditions
def is_divisible_by_9 (n : ℕ) : Prop := 9 ∣ n

def is_positive_3_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Theorem to be proved
theorem count_positive_3_digit_numbers_divisible_by_9 : 
  {n : ℕ | is_positive_3_digit_number n ∧ is_divisible_by_9 n}.card = 100 :=
sorry

end count_positive_3_digit_numbers_divisible_by_9_l57_57450


namespace sum_of_reciprocals_is_constant_l57_57130

/--
Let O be the centroid of the base triangle ABC of a regular triangular pyramid P-ABC. 
A moving plane passing through O intersects the lateral edges PA, PB, and PC at points Q, R, S respectively.
Then the sum 1/PQ + 1/PR + 1/PS = 3 is a constant independent of the plane QRS position.
-/
theorem sum_of_reciprocals_is_constant 
  {A B C P O Q R S : Type} 
  [RegularTriangularPyramid P A B C O]
  (H: PlaneThrough O Q R S ∧ IntersectsEdges P A B C Q R S):
  (1 / dist P Q + 1 / dist P R + 1 / dist P S) = 3 :=
sorry

end sum_of_reciprocals_is_constant_l57_57130


namespace number_of_intersection_points_l57_57472

-- Define the standard parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Define what it means for a line to be tangent to the parabola
def is_tangent (m : ℝ) (c : ℝ) : Prop :=
  ∃ x0 : ℝ, parabola x0 = m * x0 + c ∧ 2 * x0 = m

-- Define what it means for a line to intersect the parabola
def line_intersects_parabola (m : ℝ) (c : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, parabola x1 = m * x1 + c ∧ parabola x2 = m * x2 + c

-- Main theorem statement
theorem number_of_intersection_points :
  (∃ m c : ℝ, is_tangent m c) → (∃ m' c' : ℝ, line_intersects_parabola m' c') →
  ∃ n : ℕ, n = 1 ∨ n = 2 ∨ n = 3 :=
sorry

end number_of_intersection_points_l57_57472


namespace total_distance_traveled_l57_57999

variable (vm vr t d_up d_down : ℝ)
variable (H_river_speed : vr = 3)
variable (H_row_speed : vm = 6)
variable (H_time : t = 1)

theorem total_distance_traveled (H_upstream : d_up = vm - vr) 
                                (H_downstream : d_down = vm + vr) 
                                (total_time : d_up / (vm - vr) + d_down / (vm + vr) = t) : 
                                2 * (d_up + d_down) = 4.5 := 
                                by
  sorry

end total_distance_traveled_l57_57999


namespace kolia_lowest_rank_l57_57678

def unique_scores_in_round (round: ℕ) (students: ℕ) (scores: Fin students → ℕ) : Prop :=
  ∀ i j, i ≠ j → scores i ≠ scores j

noncomputable def kolia_round_ranks (students: ℕ) (rounds: ℕ) : ℕ → ℕ → Prop := 
  λ i round, (round = 0 → i = 3) ∧ (round = 1 → i = 4) ∧ (round = 2 → i = 5)

theorem kolia_lowest_rank (students: ℕ) (rounds: ℕ) 
  (h_students: students = 25)
  (h_rounds: rounds = 3)
  (unique_scores: ∀ round < rounds, unique_scores_in_round round students (λ i, i + round)) :
  ∃ lowest_rank, lowest_rank = 10 := 
    by
      sorry

#reduce kolia_lowest_rank

end kolia_lowest_rank_l57_57678


namespace pages_read_on_Monday_l57_57147

variable (P : Nat) (W : Nat)
def TotalPages : Nat := P + 12 + W

theorem pages_read_on_Monday :
  (TotalPages P W = 51) → (P = 39) :=
by
  sorry

end pages_read_on_Monday_l57_57147


namespace area_of_triangle_value_of_angle_C_l57_57513

-- Define the problem constraints and prove the statements
noncomputable def triangle_problem_part1 (a b c : ℝ) (A B C : ℝ) : Prop :=
  B = 150 ∧
  a = sqrt 3 * c ∧
  b = 2 * sqrt 7 ∧
  (1/2) * a * c * (Float.sin (Float.toRadians B)) = sqrt 3

noncomputable def triangle_problem_part2 (A B C : ℝ) : Prop :=
  B = 150 ∧
  (∀ A C : ℝ, sin A + sqrt 3 * sin C = sqrt 2 / 2) ∧
  C = 15

theorem area_of_triangle {a b c A B C : ℝ} :
  triangle_problem_part1 a b c A B C :=
by {
  sorry
}

theorem value_of_angle_C {A B C : ℝ} :
  triangle_problem_part2 A B C :=
by {
  sorry
}

end area_of_triangle_value_of_angle_C_l57_57513


namespace triangle_third_side_l57_57251

theorem triangle_third_side (a b : ℝ) (h₁ : a = 7) (h₂ : b = 11) : ∃ k : ℕ, 4 < k ∧ k < 18 ∧ k = 17 := 
by {
  let s := a + b,
  let d := b - a,
  have h₃ : s > 17 := by linarith,
  have h₄ : 4 < d := by linarith,
  have h₅ : d < 18 := by linarith,
  have h₆ : 17 < 18 := by linarith,
  use 17,
  linarith,
  sorry
}

end triangle_third_side_l57_57251


namespace decreasing_f_l57_57166

def f (x : ℝ) : ℝ := 3^(x^2 - 2*x)

theorem decreasing_f : ∀ x1 x2 : ℝ, x1 ≤ 1 → x2 ≤ 1 → x1 < x2 → f x1 > f x2 :=
by
  -- Sorry placeholder for the missing proof
  sorry

end decreasing_f_l57_57166


namespace tangents_parallel_common_tangent_iff_P_on_QR_l57_57560

-- Definitions of circles and tangency
variables (γ γ₁ γ₂ : ℝ → ℝ → Prop) -- Representing circles as sets of points
variables (Q R : ℝ × ℝ) -- Intersection points
variables (A₁ A₂ : ℝ × ℝ) -- Tangency points
variables (P : ℝ × ℝ) -- Arbitrary point on γ
variables (B₁ B₂ : ℝ × ℝ) -- Intersection points of PA₁, PA₂ with γ₁, γ₂

-- Prove tangent parallelism
theorem tangents_parallel
  (H₁ : ∀ (x y : ℝ × ℝ), γ₁ x y → γ₂ x y → (x, y) = Q ∨ (x, y) = R) -- γ₁ and γ₂ intersect at Q and R
  (H₂ : ∀ (x : ℝ × ℝ), γ x (A₁) ∧ γ₁ x (A₁)) -- γ₁ touches γ at A₁
  (H₃ : ∀ (x : ℝ × ℝ), γ x (A₂) ∧ γ₂ x (A₂)) -- γ₂ touches γ at A₂
  (H₄ : ∀ (x : ℝ × ℝ) (y : ℝ × ℝ), γ x y → (x, y) = P) -- P is on γ
  (H₅ : ∀ (x : ℝ × ℝ) (y : ℝ × ℝ), γ x y → (y, B₁) = (y, γ₁) ∨ (y, B₂) = (y, γ₂)): -- Definition of B₁ and B₂
  Prop :=
by
  sorry

-- Prove common tangent equivalence
theorem common_tangent_iff_P_on_QR 
  (H₁ : ∀ (x y : ℝ × ℝ), γ₁ x y → γ₂ x y → (x, y) = Q ∨ (x, y) = R) -- γ₁ and γ₂ intersect at Q and R
  (H₂ : ∀ (x : ℝ × ℝ), γ x (A₁) ∧ γ₁ x (A₁)) -- γ₁ touches γ at A₁
  (H₃ : ∀ (x : ℝ × ℝ), γ x (A₂) ∧ γ₂ x (A₂)) -- γ₂ touches γ at A₂
  (H₄ : ∀ (x : ℝ × ℝ), γ x y → (x, y) = P) -- P is on γ
  (H₅ : ∀ (x : ℝ × ℝ) (y : ℝ × ℝ), γ x y → (y, B₁) = (y, γ₁) ∨ (y, B₂) = (y, γ₂)) -- Definition of B₁ and B₂
  : (∃ p, p ∈ segment Q R → P = p) ↔ (is_tangent γ₁ B₁ B₂ ∧ is_tangent γ₂ B₂ B₁) :=
by
  sorry

-- Definitions and helper functions (if needed) could be added here
def is_tangent (c₁ c₂ : ℝ) (b1 b2 : ℝ × ℝ) : Prop := sorry
def segment (a b : ℝ × ℝ) : set (ℝ × ℝ) := sorry

end tangents_parallel_common_tangent_iff_P_on_QR_l57_57560


namespace count_3_digit_numbers_divisible_by_9_l57_57438

theorem count_3_digit_numbers_divisible_by_9 : 
  let count := (range (integer_divisible_in_range 9 100 999)).length
  count = 100 := 
by
  sorry

noncomputable def integer_divisible_in_range (k m n : ℕ) : List ℕ :=
  let start := m / k + (if (m % k = 0) then 0 else 1)
  let end_ := n / k
  List.range (end_ - start + 1) |>.map (λ i => (start + i) * k)

noncomputable def range (xs : List ℕ) := xs

end count_3_digit_numbers_divisible_by_9_l57_57438


namespace maximum_value_sqrt22_l57_57530

noncomputable def max_value_expression (x y z : ℝ) : ℝ := 2 * x * y * (real.sqrt 6) + 8 * y * z

theorem maximum_value_sqrt22 (x y z : ℝ)
  (h1 : x ≥ 0)
  (h2 : y ≥ 0)
  (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 1) :
  max_value_expression x y z ≤ real.sqrt 22 := sorry

end maximum_value_sqrt22_l57_57530


namespace length_approx_24_802_l57_57809

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length ^ 2

noncomputable def length_of_rectangle_eq_area_square (width length : ℝ) (side_length : ℝ) : Prop :=
  (width * length = area_of_square side_length)

theorem length_approx_24_802
  (side_length : ℝ) (h1 : side_length = 12.5)
  (width : ℝ) (h2 : width = 6.3) :
  ∃ length : ℝ, length_of_rectangle_eq_area_square width length side_length ∧ length ≈ 24.802 :=
sorry

end length_approx_24_802_l57_57809


namespace find_x_from_equation_l57_57988

theorem find_x_from_equation :
  ∃ x : ℝ, 4^5 + 4^5 + 4^5 = 2^x ∧ x = 10 + Real.log2 3 :=
by
  sorry

end find_x_from_equation_l57_57988


namespace female_in_coach_class_l57_57892

theorem female_in_coach_class (total_passengers : ℕ) (pct_female : ℚ) (pct_first_class : ℚ) (frac_male_first_class : ℚ) :
  total_passengers = 120 →
  pct_female = 0.30 →
  pct_first_class = 0.10 →
  frac_male_first_class = 1 / 3 →
  let total_females := total_passengers * pct_female in
  let first_class_passengers := total_passengers * pct_first_class in
  let males_in_first_class := first_class_passengers * frac_male_first_class in
  let females_in_first_class := first_class_passengers - males_in_first_class in
  let females_in_coach_class := total_females - females_in_first_class in
  females_in_coach_class = 28 :=
by 
  intros
  sorry

end female_in_coach_class_l57_57892


namespace petya_password_l57_57900

def ways_to_create_password : Nat :=
  let total_digits := 9  -- Digits 0-6, 8-9
  let total_passwords := total_digits ^ 4  -- Total possible 4-digit passwords
  let distinct_passwords := (Nat.factorial 9) / (Nat.factorial (9 - 4))  -- \binom{9}{4} * 4!
  total_passwords - distinct_passwords

theorem petya_password (ways_to_create_password = 3537) : ways_to_create_password = 3537 := 
sorry

end petya_password_l57_57900


namespace num_lines_intersecting_skew_lines_l57_57085

-- Definitions for the conditions
def is_skew (a b : Line) : Prop :=
  ¬ ∃ p : Point, p ∈ a ∧ p ∈ b ∧ ¬ is_parallel a b

-- Main theorem statement
theorem num_lines_intersecting_skew_lines (a b c : Line) 
  (h_skew_ab : is_skew a b)
  (h_skew_bc : is_skew b c)
  (h_skew_ca : is_skew c a) : 
  ∃! d : Line, ∀ p : Point, (p ∈ d) → (p ∈ a) ∧ (p ∈ b) ∧ (p ∈ c) :=
sorry

end num_lines_intersecting_skew_lines_l57_57085


namespace necessary_and_sufficient_condition_l57_57036

variable (a : ℝ)

theorem necessary_and_sufficient_condition :
  (a^2 + 4 * a - 5 > 0) ↔ (|a + 2| > 3) := sorry

end necessary_and_sufficient_condition_l57_57036


namespace hyperbola_eccentricity_b_value_l57_57403

theorem hyperbola_eccentricity_b_value (b : ℝ) (a : ℝ) (e : ℝ) 
  (h1 : a^2 = 1) (h2 : e = 2) 
  (h3 : b > 0) (h4 : b^2 = 4 - 1) : 
  b = Real.sqrt 3 := 
by 
  sorry

end hyperbola_eccentricity_b_value_l57_57403


namespace min_sticks_to_be_broken_form_square_without_breaks_l57_57595

noncomputable def total_length (n : ℕ) : ℕ := n * (n + 1) / 2

def divisible_by_4 (x : ℕ) : Prop := x % 4 = 0

theorem min_sticks_to_be_broken (n : ℕ) : n = 12 → (¬ divisible_by_4 (total_length n)) ∧ (minimal_breaks n = 2) :=
by
  intro h1
  rw h1
  have h2 : total_length 12 = 78 := by decide
  have h3 : ¬ divisible_by_4 78 := by decide
  exact ⟨h3, sorry⟩

theorem form_square_without_breaks (n : ℕ) : n = 15 → divisible_by_4 (total_length n) ∧ (minimal_breaks n = 0) :=
by
  intro h1
  rw h1
  have h2 : total_length 15 = 120 := by decide
  have h3 : divisible_by_4 120 := by decide
  exact ⟨h3, sorry⟩

end min_sticks_to_be_broken_form_square_without_breaks_l57_57595


namespace distance_from_m_to_v_in_tetrahedron_l57_57110

theorem distance_from_m_to_v_in_tetrahedron
  (VA_perpendicular_VB : true)
  (VB_perpendicular_VC : true)
  (VC_perpendicular_VA : true)
  (d_M_to_VAB : ℝ)
  (d_M_to_VAC : ℝ)
  (d_M_to_VBC : ℝ)
  (d_M_to_VAB_eq : d_M_to_VAB = 2)
  (d_M_to_VAC_eq : d_M_to_VAC = 3)
  (d_M_to_VBC_eq : d_M_to_VBC = 6) :
  distance_from_point_to_origin (6, 3, 2) = 7 := by
  sorry

end distance_from_m_to_v_in_tetrahedron_l57_57110


namespace impossible_grid_arrangement_l57_57114

theorem impossible_grid_arrangement :
  ¬ ∃ (f : Fin 25 → Fin 41 → ℤ),
    (∀ i j, abs (f i j - f (i + 1) j) ≤ 16 ∧ abs (f i j - f i (j + 1)) ≤ 16 ∧
            f i j ≠ f (i + 1) j ∧ f i j ≠ f i (j + 1)) := 
sorry

end impossible_grid_arrangement_l57_57114


namespace problem_1_problem_2_problem_3_l57_57710

-- Define basic elements for boys and girls
inductive Person
| boy (n : ℕ) : Person
| girl (n : ℕ) : Person

open Person

-- Definitions for the number of different arrangements
def arrangements_with_girls_at_ends : ℕ :=
  2 * (4 * 3 * 2 * 1)

def arrangements_with_girls_not_adjacent : ℕ :=
  6 * (5 * 4 * 3 * 2 * 1 * 2)

def arrangements_girl_A_right_of_B : ℕ :=
  6! // divFactorial (4!)

theorem problem_1 : arrangements_with_girls_at_ends = 48 := by sorry

theorem problem_2 : arrangements_with_girls_not_adjacent = 480 := by sorry

theorem problem_3 : arrangements_girl_A_right_of_B = 360 := by sorry

end problem_1_problem_2_problem_3_l57_57710


namespace range_of_alpha_l57_57041

theorem range_of_alpha (x α : ℝ) (h_curve : ∃ (P : ℝ×ℝ), P.2 = P.1^3 - P.1 + 2/3) 
  (h_tangent : ∀ x, tan α = 3 * x^2 - 1) : 
  α ∈ (Set.Icc 0 (Real.pi / 2)) ∪ (Set.Icc (3 * Real.pi / 4) Real.pi) :=
sorry

end range_of_alpha_l57_57041


namespace group_of_12_people_friends_condition_l57_57156

theorem group_of_12_people_friends_condition :
  ∃ (p1 p2 : ℕ) (s : Finset ℕ), s.card = 5 ∧ (∀ x ∈ s, (friends x p1 ∧ friends x p2) ∨ (¬ friends x p1 ∧ ¬ friends x p2)) :=
by
  sorry

end group_of_12_people_friends_condition_l57_57156


namespace range_of_f_l57_57411

-- Definition of the function f with given conditions
def f (x b : ℝ) := 2^(x - b)

-- Conditions: 2 ≤ x ≤ 4 and f(3) = 1
variable (x : ℝ)
variable (b : ℝ)
variable (h1 : 2 ≤ x ∧ x ≤ 4)
variable (h2 : f 3 b = 1)

-- Statement that range of f(x) is [1/2, 2]
theorem range_of_f : ∀ x : ℝ, (2 ≤ x ∧ x ≤ 4) →  1 / 2 ≤ f x 3 ∧ f x 3 ≤ 2 :=
by 
  -- Here we will insert the proof
  sorry

end range_of_f_l57_57411


namespace parabola_equation_standard_l57_57028

theorem parabola_equation_standard (m : ℝ) (h1 : ∃ p > 0, ∀ m, 6 = (p / 2) + 5) : ∃ p,  y^2 = -4 * x := by
  sorry

end parabola_equation_standard_l57_57028


namespace initial_carrots_l57_57957

theorem initial_carrots (n : ℕ) 
    (h1: 3640 = 180 * (n - 4) + 760) 
    (h2: 180 * (n - 4) < 3640) 
    (h3: 4 * 190 = 760) : 
    n = 20 :=
by
  sorry

end initial_carrots_l57_57957


namespace valid_assignment_count_l57_57494

-- Define tasks and volunteers
inductive Task : Type
| translation | tour_guiding | etiquette

def num_tasks : ℕ := 3
def num_volunteers : ℕ := 5

-- Definition of a valid assignment where each task is assigned at least one volunteer
def isValidAssignment (assignment : Fin num_volunteers → Task) : Prop :=
  (∀ t : Task, ∃ i : Fin num_volunteers, assignment i = t)

-- The count of all valid assignments is 150
theorem valid_assignment_count : 
  (Fin num_volunteers → Task) → 
  (isValidAssignment → ℕ) := 
  sorry
-- You'll need to define this theorem appropriately

end valid_assignment_count_l57_57494


namespace even_numbers_between_150_and_350_l57_57071

theorem even_numbers_between_150_and_350 : 
  let smallest_even := 152
  let largest_even := 348
  (∃ n, (2 * n > 150) ∧ (2 * n < 350) ∧ (n <= 174)) →
  (∑ n in (finset.range 100).filter (λ n, (2 * (75 + n) > 150) ∧ (2 * (75 + n) < 350)), n) = 99 :=
by
  sorry

end even_numbers_between_150_and_350_l57_57071


namespace cos_a5_l57_57420

noncomputable def geom_seq (r : ℝ) (a : ℝ) (n : ℕ) := a * r^n

theorem cos_a5 {a r : ℝ} 
  (h₄ : geom_seq r a 3 * geom_seq r a 7 = π^2 / 9) : 
  cos (geom_seq r a 5) = ±1/2 := 
sorry

end cos_a5_l57_57420


namespace abs_nested_expression_l57_57468

theorem abs_nested_expression (x : ℝ) (h : x > 3) : |1 - |2 - x|| = x - 3 :=
by sorry

end abs_nested_expression_l57_57468


namespace minimize_surface_area_of_prism_l57_57733

theorem minimize_surface_area_of_prism 
  (V : ℝ) 
  (a h : ℝ) 
  (h_volume : V = (sqrt 3 / 4) * a^2 * h) 
  (h_surface_area : ∀ a h, (3 * a * h + sqrt 3 / 2 * a^2) = minimize_surface_area (a, h)) : 
  a = sqrt (3) * (4/3) * (V / a^3) := 
sorry

end minimize_surface_area_of_prism_l57_57733


namespace length_DJ_l57_57849

-- Definitions from the problem
variables {D E F J G H I : Type}
variables (DE DF EF : ℝ)
variables (u v w : ℝ)
variables (DJ r : ℝ)

-- Given conditions
def conditions (D E F J G H I : Type) (DE DF EF : ℝ) : Prop :=
  DE = 17 ∧ DF = 19 ∧ EF = 16

-- The mathematical proof problem
theorem length_DJ (D E F J G H I : Type) (DE DF EF : ℝ)
  (conditions : conditions D E F J G H I DE DF EF) : DJ = √(405.307) :=
by
  sorry

end length_DJ_l57_57849


namespace sum_of_three_squares_l57_57136

theorem sum_of_three_squares (n : ℕ) (h_pos : 0 < n) (h_square : ∃ m : ℕ, 3 * n + 1 = m^2) : ∃ x y z : ℕ, n + 1 = x^2 + y^2 + z^2 :=
by
  sorry

end sum_of_three_squares_l57_57136


namespace decreasing_even_function_condition_l57_57751

theorem decreasing_even_function_condition (f : ℝ → ℝ) 
    (h1 : ∀ x y : ℝ, x < y → y < 0 → f y < f x) 
    (h2 : ∀ x : ℝ, f (-x) = f x) : f 13 < f 9 ∧ f 9 < f 1 := 
by
  sorry

end decreasing_even_function_condition_l57_57751


namespace count_3_digit_numbers_divisible_by_9_l57_57455

theorem count_3_digit_numbers_divisible_by_9 : 
  (finset.filter (λ x : ℕ, x % 9 = 0) (finset.Icc 100 999)).card = 100 := 
sorry

end count_3_digit_numbers_divisible_by_9_l57_57455


namespace hyperbola_eccentricity_eq_sqrt_10_l57_57716

theorem hyperbola_eccentricity_eq_sqrt_10 (b : ℝ) 
  (h_line_slope : ∃ k, k = 1) 
  (h_vertex_A : ∃ A : ℝ × ℝ, A = (-1, 0))
  (h_asymptotes : ∃ f : ℝ → ℝ, ∀ x, f x = b * x ∨ f x = -(b * x))
  (h_midpoint_B : ∃ B C : ℝ × ℝ, B = (-(1 / (b + 1)), -(1 / (b + 1)) * b) ∧ C = (1 / (b - 1), (1 / (b - 1)) * b) ∧ 2 * B.1 = A.1 + C.1) :
  sqrt (1 + b^2) = sqrt 10 :=
  sorry

end hyperbola_eccentricity_eq_sqrt_10_l57_57716


namespace charcoal_drawings_count_l57_57963

-- Defining the conditions
def total_drawings : Nat := 25
def colored_pencil_drawings : Nat := 14
def blending_marker_drawings : Nat := 7

-- Defining the target value for charcoal drawings
def charcoal_drawings : Nat := total_drawings - (colored_pencil_drawings + blending_marker_drawings)

-- The theorem we need to prove
theorem charcoal_drawings_count : charcoal_drawings = 4 :=
by
  -- Lean proof goes here, but since we skip the proof, we'll just use 'sorry'
  sorry

end charcoal_drawings_count_l57_57963


namespace no_diameter_diagonal_needed_l57_57388

variable {A B C D : Point} {R : ℝ}

-- Define the property of being a circumscribed circle
def circumscribed_circle (A B C D : Point) (R : ℝ) : Prop :=
  ∃ O : Point, (dist O A = R) ∧ (dist O B = R) ∧ (dist O C = R) ∧ (dist O D = R)

-- Given condition
def quad_eq (A B C D : Point) (R : ℝ) : Prop :=
  (dist A B)^2 + (dist B C)^2 + (dist C D)^2 + (dist A D)^2 = 8 * R^2

-- Theorem statement
theorem no_diameter_diagonal_needed (A B C D : Point) (R : ℝ) :
  circumscribed_circle A B C D R →
  quad_eq A B C D R →
  ¬ (let diag1 := dist A C in diag1 = 2 * R ∨ let diag2 := dist B D in diag2 = 2 * R) := sorry

end no_diameter_diagonal_needed_l57_57388


namespace quadrilateral_area_is_127_5_l57_57108

structure Quadrilateral (A B C D : Type) :=
(angle_BCD_right : ∠ B C D = 90)
(AB_length : AB = 15)
(BC_length : BC = 5)
(CD_length : CD = 12)
(AD_length : AD = 13)
(diagonals_intersect_right : ∠ A C ∠ B D = 90)

def area_of_quadrilateral {A B C D : Type} [Quadrilateral A B C D] : ℝ :=
let area_triangle_BCD := (1 / 2) * (5 * 12) in
let area_triangle_ABD := (1 / 2) * (15 * 13) in
area_triangle_BCD + area_triangle_ABD

theorem quadrilateral_area_is_127_5
  {A B C D : Type}
  [h : Quadrilateral A B C D] :
  area_of_quadrilateral = 127.5 :=
sorry

end quadrilateral_area_is_127_5_l57_57108


namespace intersection_points_l57_57556

theorem intersection_points (f : ℝ → ℝ) (h_inv : Function.Injective f) : 
  set.count {x : ℝ | f (x^2) = f (x^4)} = 3 :=
by
  -- Define the set of points where f(x^2) = f(x^4)
  let S := {x : ℝ | f (x^2) = f (x^4)}
  have h1 : S = {x | x = -1} ∪ {x | x = 0} ∪ {x | x = 1}, from sorry
  
  -- These are the only solutions to f(x^2) = f(x^4)
  have h2 : ∀ x ∈ S, x = -1 ∨ x = 0 ∨ x = 1, from sorry
    
  -- Therefore, there are exactly 3 points where f(x^2) = f(x^4)
  exact Finite.count_set_of_mem_eq _ (λ _, h2)

end intersection_points_l57_57556


namespace polynomial_system_solution_l57_57334

variable {x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ}

theorem polynomial_system_solution (
  h1 : x₁ + 3 * x₂ + 5 * x₃ + 7 * x₄ + 9 * x₅ + 11 * x₆ + 13 * x₇ = 3)
  (h2 : 3 * x₁ + 5 * x₂ + 7 * x₃ + 9 * x₄ + 11 * x₅ + 13 * x₆ + 15 * x₇ = 15)
  (h3 : 5 * x₁ + 7 * x₂ + 9 * x₃ + 11 * x₄ + 13 * x₅ + 15 * x₆ + 17 * x₇ = 85) :
  7 * x₁ + 9 * x₂ + 11 * x₃ + 13 * x₄ + 15 * x₅ + 17 * x₆ + 19 * x₇ = 213 :=
sorry

end polynomial_system_solution_l57_57334


namespace promotion_cost_l57_57968

theorem promotion_cost :
  ∀ (CostOfFlour CostOfSalt Revenue Profit TotalExpenses PromotionCost: ℕ),
    (CostOfFlour = 200) →
    (CostOfSalt = 2) →
    (Revenue = 10000) →
    (Profit = 8798) →
    (TotalExpenses = CostOfFlour + CostOfSalt + PromotionCost) →
    (Profit = Revenue - TotalExpenses) →
    PromotionCost = 1000 := by
  intros CostOfFlour CostOfSalt Revenue Profit TotalExpenses PromotionCost
  assume h1 : CostOfFlour = 200
  assume h2 : CostOfSalt = 2
  assume h3 : Revenue = 10000
  assume h4 : Profit = 8798
  assume h5 : TotalExpenses = CostOfFlour + CostOfSalt + PromotionCost
  assume h6 : Profit = Revenue - TotalExpenses
  sorry

end promotion_cost_l57_57968


namespace servings_required_l57_57670

/-- Each serving of cereal is 2.0 cups, and 36 cups are needed. Prove that the number of servings required is 18. -/
theorem servings_required (cups_per_serving : ℝ) (total_cups : ℝ) (h1 : cups_per_serving = 2.0) (h2 : total_cups = 36.0) :
  total_cups / cups_per_serving = 18 :=
by
  sorry

end servings_required_l57_57670


namespace area_of_triangle_length_and_sin_l57_57509

variable (a b c : ℝ) (cosC : ℝ) (sinC : ℝ) (sinA : ℝ)
variable (S : ℝ)

-- Given conditions
def conditions : Prop := 
  (a = 4) ∧ 
  (b = 5) ∧ 
  (cosC = 1 / 8) ∧ 
  (sinC = sqrt(1 - (cosC ^ 2))) ∧ 
  (S = (1 / 2) * a * b * sinC) ∧ 
  (c = sqrt(a^2 + b^2 - 2 * a * b * cosC)) ∧ 
  (sinA = a * sinC / c)

-- Prove that the area of triangle ABC is 15 * sqrt 7 / 4
theorem area_of_triangle (h : conditions a b c cosC sinC sinA S) :
  S = 15 * sqrt 7 / 4 := 
by sorry

-- Prove the length of side c is 6 and the value of sin A is sqrt 7 / 4
theorem length_and_sin (h : conditions a b c cosC sinC sinA S) :
  c = 6 ∧ sinA = sqrt 7 / 4 := 
by sorry

end area_of_triangle_length_and_sin_l57_57509


namespace g_g_g_3_l57_57536

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 2*n + 1 else 4*n - 3

theorem g_g_g_3 : g (g (g 3)) = 241 := by
  sorry

end g_g_g_3_l57_57536


namespace evlyn_can_buy_grapes_l57_57079

theorem evlyn_can_buy_grapes 
  (price_pears price_oranges price_lemons price_grapes : ℕ)
  (h1 : 10 * price_pears = 5 * price_oranges)
  (h2 : 4 * price_oranges = 6 * price_lemons)
  (h3 : 3 * price_lemons = 2 * price_grapes) :
  (20 * price_pears = 10 * price_grapes) :=
by
  -- The proof is omitted using sorry
  sorry

end evlyn_can_buy_grapes_l57_57079


namespace num_real_roots_eq_4_l57_57217

theorem num_real_roots_eq_4 : 
  let f (x : ℝ) : ℝ := x^4 - 2^|x| 
  in ∃ (roots : ℕ), roots = 4 ∧ ∀ x, f x = 0 ↔ ∃ y, x = y :=
by 
  sorry

end num_real_roots_eq_4_l57_57217


namespace transformed_sin_eq_l57_57914

noncomputable def transformed_function (x : ℝ) : ℝ :=
  sin (2 * x + π / 3)

theorem transformed_sin_eq : ∀ x : ℝ, transformed_function x = sin (2 * x + π / 3) := by
  intro x
  simp [transformed_function]
  sorry

end transformed_sin_eq_l57_57914


namespace numerator_in_second_fraction_l57_57083

theorem numerator_in_second_fraction (p q x: ℚ) (h1 : p / q = 4 / 5) (h2 : 11 / 7 + x / (2 * q + p) = 2) : x = 6 :=
sorry

end numerator_in_second_fraction_l57_57083


namespace cube_white_odd_impossible_cube_black_odd_possible_l57_57638

def is_possible_odd_white (cube : list (list (list bool))) : Prop :=
  ∀ i j, odd ((cube[i] |>.map (λ row, row[j]) |>.count true))

def is_possible_odd_black (cube : list (list (list bool))) : Prop :=
  ∀ i j, odd ((cube[i] |>.map (λ row, row[j]) |>.count false))

theorem cube_white_odd_impossible : 
  ¬∃ cube : list (list (list bool)), 
  (cube.flatten |>.count true = 14) ∧ (cube.flatten |>.count false = 13) ∧ is_possible_odd_white cube :=
sorry

theorem cube_black_odd_possible : 
  ∃ cube : list (list (list bool)), 
  (cube.flatten |>.count true = 14) ∧ (cube.flatten |>.count false = 13) ∧ is_possible_odd_black cube :=
sorry

end cube_white_odd_impossible_cube_black_odd_possible_l57_57638


namespace square_with_12_sticks_square_with_15_sticks_l57_57583

-- Definitions for problem conditions
def sum_of_first_n_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def can_form_square (total_length : ℕ) : Prop :=
  total_length % 4 = 0

-- Given n = 12, check if breaking 2 sticks is required to form a square
theorem square_with_12_sticks : (n = 12) → ¬ can_form_square (sum_of_first_n_natural_numbers 12) → true :=
by
  intros
  sorry

-- Given n = 15, check if it is possible to form a square without breaking any sticks
theorem square_with_15_sticks : (n = 15) → can_form_square (sum_of_first_n_natural_numbers 15) → true :=
by
  intros
  sorry

end square_with_12_sticks_square_with_15_sticks_l57_57583


namespace value_expr1_value_expr2_l57_57728

variable {α : ℝ}

def condition (α : ℝ) : Prop := tan α / (tan α - 1) = -1

theorem value_expr1 (h : condition α) : (sin α - 3 * cos α) / (sin α + cos α) = -5 / 3 := by
  sorry

theorem value_expr2 (h : condition α) : sin α ^ 2 + sin α * cos α + 2 = 13 / 5 := by
  sorry

end value_expr1_value_expr2_l57_57728


namespace impossible_division_l57_57841

noncomputable def total_matches := 1230

theorem impossible_division :
  ∀ (x y z : ℕ), 
  (x + y + z = total_matches) → 
  (z = (1 / 2) * (x + y + z)) → 
  false :=
by
  sorry

end impossible_division_l57_57841


namespace part_one_part_two_l57_57737

-- Define circle O centered at origin with equation x^2 + y^2 = 1
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define parabola E with equation y = x^2 - 2
def parabola_E (x y : ℝ) : Prop := y = x^2 - 2

-- Define the perpendicular condition between points (x1, y1) and (x2, y2)
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Prove that under given conditions, the equation of the line l is y = -1
theorem part_one (k b x1 y1 x2 y2 : ℝ) (h1 : circle_O x1 y1) (h2 : circle_O x2 y2)
(h3 : parabola_E x1 y1) (h4 : parabola_E x2 y2) (h5 : perpendicular x1 y1 x2 y2)
(h6 : ∃ x, parabola_E x (k*x + b) ∧ circle_O x (k*x + b)) :
  b = -1 ∧ k = 0 := sorry

-- Given conditions for part two
def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y1 - y2) / (x1 - x2)

-- Define P point on the parabola E
def point_on_parabola (x0 y0 : ℝ) : Prop := y0 = x0^2 - 2

-- Define the tangent condition for the circle O
def tangent_condition (x0 y0 k : ℝ) : Prop := abs (y0 - k * x0) = real.sqrt(k^2 + 1)

-- Prove that the coordinates of point P are either (-sqrt(3)/3, -5/3) or (sqrt(3), 1)
theorem part_two (x0 y0 k1 k2 x1 y1 x2 y2 : ℝ) 
(h1 : point_on_parabola x0 y0)
(h2 : tangent_condition x0 y0 k1)
(h3 : tangent_condition x0 y0 k2)
(h4 : parabola_E x1 y1)
(h5 : parabola_E x2 y2)
(h6 : k1 + k2 = (2 * x0 * y0) / (x0^2 - 1))
(h7 : slope x1 y1 x2 y2 = -real.sqrt(3)) :
  (x0 = -real.sqrt(3)/3 ∧ y0 = -5/3) ∨ (x0 = real.sqrt(3) ∧ y0 = 1) := sorry

end part_one_part_two_l57_57737


namespace irrational_count_l57_57318

def is_irrational (x : ℝ) : Prop := ¬∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_count :
  let numbers := [Real.sqrt 9, 3.14159265, -Real.sqrt 3, 0, Real.pi, 5 / 6, 0.101001001001...] in
  (numbers.filter is_irrational).length = 3 :=
by sorry

end irrational_count_l57_57318


namespace min_sticks_to_be_broken_form_square_without_breaks_l57_57597

noncomputable def total_length (n : ℕ) : ℕ := n * (n + 1) / 2

def divisible_by_4 (x : ℕ) : Prop := x % 4 = 0

theorem min_sticks_to_be_broken (n : ℕ) : n = 12 → (¬ divisible_by_4 (total_length n)) ∧ (minimal_breaks n = 2) :=
by
  intro h1
  rw h1
  have h2 : total_length 12 = 78 := by decide
  have h3 : ¬ divisible_by_4 78 := by decide
  exact ⟨h3, sorry⟩

theorem form_square_without_breaks (n : ℕ) : n = 15 → divisible_by_4 (total_length n) ∧ (minimal_breaks n = 0) :=
by
  intro h1
  rw h1
  have h2 : total_length 15 = 120 := by decide
  have h3 : divisible_by_4 120 := by decide
  exact ⟨h3, sorry⟩

end min_sticks_to_be_broken_form_square_without_breaks_l57_57597


namespace cover_obtuse_triangle_with_isosceles_right_angled_triangle_l57_57127

theorem cover_obtuse_triangle_with_isosceles_right_angled_triangle : 
  ∀ (A B C : Type) (circle : set Type) (radius : ℝ) (hypotenuse_length : ℝ), 
  (radius = 1) →
  (hypotenuse_length = sqrt 2 + 1) →
  (∃ O : Type, circle = set.sep O (λ x, dist O x = radius)) →
  (angle B C A > 90) →
  (angle A C B ≤ 45) →
  ∃ (isosceles_right_triangle : Type) (A D E : Type), 
  (isosceles_right_triangle ∈ (set.sep (isosceles_right_triangle) (λ x, hypotenuse x = hypotenuse_length))) ∧
  (set.finite ({A, B, C, D, E} : set Type)) ∧
  (inside_triangle C isosceles_right_triangle) := 
by 
  sorry

end cover_obtuse_triangle_with_isosceles_right_angled_triangle_l57_57127


namespace last_three_nonzero_digits_100_fact_l57_57359

theorem last_three_nonzero_digits_100_fact :
  let N := (100.factorial / 10^24)
  in (∃ M, M = (100.factorial / 5^24) ∧
      M % 125 = 1 ∧
      (∃ K : ℕ, 2^24 ≡ 16 [MOD 125])) → 
  (N % 1000 = 376) :=
begin
  sorry
end

end last_three_nonzero_digits_100_fact_l57_57359


namespace sum_of_exterior_angles_of_pentagon_is_360_l57_57229

theorem sum_of_exterior_angles_of_pentagon_is_360 
  (h : ∀ (n : ℕ), n > 2 → (∑ (i : ℕ) in finset.range n, exterior_angle i n) = 360) :
  (∑ (i : ℕ) in finset.range 5, exterior_angle i 5) = 360 :=
by
  sorry

end sum_of_exterior_angles_of_pentagon_is_360_l57_57229


namespace log_local_odd_poly_local_odd_l57_57376

-- Definition of local odd function
def local_odd (f : ℝ → ℝ) (domain : Set ℝ) := ∃ x ∈ domain, f (-x) = - f x

-- Part (1)
theorem log_local_odd (m : ℝ) : local_odd (fun x => log x * log 2 (x + m)) (Set.Icc (-1) 1) ↔ 1 < m ∧ m < real.sqrt 2 :=
sorry

-- Part (2)
theorem poly_local_odd (m : ℝ) : local_odd (fun x => 9 ^ x - m * 3 ^ (x + 1) - 3) Set.univ ↔ -2 / 3 ≤ m :=
sorry

end log_local_odd_poly_local_odd_l57_57376


namespace expected_value_is_one_l57_57956

-- Definitions of the conditions
def num_black : ℕ := 3
def num_red : ℕ := 1

def score (ball : ℕ) : ℕ :=
  if ball = 0 then 2 else 0

def prob_black (num_drawn : ℕ) (remaining : ℕ) : ℚ :=
  (num_black - num_drawn) / remaining.to_rat

def prob_red (num_drawn : ℕ) (remaining : ℕ) : ℚ :=
  (num_red - num_drawn) / remaining.to_rat

def expected_score : ℚ :=
  (prob_black 0 (num_black + num_red) * prob_black 1 (num_black + num_red - 1) * score 1 +
   prob_black 0 (num_black + num_red) * prob_red 1 (num_black + num_red - 1) * score 0 +
   prob_red 0 (num_black + num_red) * prob_black 1 (num_black + num_red - 1) * score 0)

-- The goal
theorem expected_value_is_one : expected_score = 1 := by
  sorry

end expected_value_is_one_l57_57956


namespace find_a_b_and_max_value_l57_57774

def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2
def f' (a b x : ℝ) : ℝ := (a / x) - 2 * b * x

theorem find_a_b_and_max_value :
  (∀ a b : ℝ, f' a b 1 = 0 ∧ f a b 1 = -1/2 →
    a = 1 ∧ b = 1/2) ∧
  (∀ a b : ℝ, a = 1 → b = 1/2 → ∀ x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1, f a b x ≤ f 1 1/2 1) :=
by
  sorry

end find_a_b_and_max_value_l57_57774


namespace elmer_fuel_savings_l57_57711

theorem elmer_fuel_savings
  (y : ℝ)  -- old car fuel efficiency in km per liter
  (d : ℝ)  -- old car fuel cost per liter
  (trip_distance : ℝ)  -- trip distance in km
  (h1 : trip_distance = 600)
  (h2 : ∃ new_car_fuel_efficiency, new_car_fuel_efficiency = 1.4 * y)
  (h3 : ∃ diesel_cost_per_liter, diesel_cost_per_liter = 1.3 * d)
  : 
  let old_car_cost := (trip_distance / y) * d in
  let new_car_cost := (trip_distance / (1.4 * y)) * (1.3 * d) in
  let savings := (old_car_cost - new_car_cost) / old_car_cost * 100 in
  savings = 7.143 :=
by
  exist -- Insert any placeholder for proof script as this is not required.
  sorry

end elmer_fuel_savings_l57_57711


namespace area_relation_l57_57475

variables (x y z u p q r : Real) (ABC_area : Real)

-- Given conditions
def area_external_triangles : Real := x + y + z
def area_internal_triangle : Real := u
def area_ABC : Real := ABC_area
def decomposition_ABC : Prop := ABC_area = u + p + q + r

-- Theorem statement
theorem area_relation (hdecomp : decomposition_ABC) :
  x + y + z - u = 2 * ABC_area :=
by
  have h1 : area_external_triangles x y z - area_internal_triangle u = 2 * area_ABC ABC_area := sorry
  exact h1

end area_relation_l57_57475


namespace zoo_gorilla_percentage_l57_57232

theorem zoo_gorilla_percentage :
  ∀ (visitors_per_hour : ℕ) (open_hours : ℕ) (gorilla_visitors : ℕ) (total_visitors : ℕ)
    (percentage : ℕ),
  visitors_per_hour = 50 → open_hours = 8 → gorilla_visitors = 320 →
  total_visitors = visitors_per_hour * open_hours →
  percentage = (gorilla_visitors * 100) / total_visitors →
  percentage = 80 :=
by
  intros visitors_per_hour open_hours gorilla_visitors total_visitors percentage
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h3, h4] at h5
  exact h5

end zoo_gorilla_percentage_l57_57232


namespace find_side_length_of_triangle_l57_57508

theorem find_side_length_of_triangle (a b : ℝ) (B : ℝ) (h_a : a = 2)
    (h_b : b = Real.sqrt 7) (h_B : B = Real.pi / 3) : ∃ c : ℝ, c = 3 :=
by
  -- Conditions
  have h_cosB: Real.cos B = 1 / 2,
    from Real.cos_pi_div_three -- Uses the known trigonometric value for cos(π/3)
  
  -- From the cosine rule and given conditions
  have h_cosine_rule: b * b = a * a + c * c - 2 * a * c * Real.cos B,
    calc b * b = 7 : by rw [h_b, Real.sqrt_mul_self (show (0 : ℝ) ≤ 7, by norm_num)]
           ... = 4 + c * c - 2 * 2 * c * 1 / 2 : by rw [h_a, ←h_cosB]; sorry -- Fill in the steps
  
  have h_quad_eq: c * c - 2 * c + 3 = 0,
    sorry -- Simplification step
  
  -- Solutions to quadratic equation c * c - 2 * c + 3 = 0 are c = 3 and c = -1
  -- Discarding the negative solution as it's not valid for the length of a side
  
  exact ⟨3, rfl⟩

end find_side_length_of_triangle_l57_57508


namespace vectors_perpendicular_l57_57432

def vec (a b : ℝ) := (a, b)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

@[simp]
def a := vec (-1) 2
@[simp]
def b := vec 1 3

theorem vectors_perpendicular :
  dot_product a (vector_sub a b) = 0 := by
  sorry

end vectors_perpendicular_l57_57432


namespace monotone_decreasing_and_difference_l57_57416
open Function

noncomputable def f (x : ℝ) := x / (x^2 - 1)

theorem monotone_decreasing_and_difference (h1 : 2 ≤ x ∧ x ≤ 3) (h2 : 2 ≤ y ∧ y ≤ 3) :
  (∀ x y, x < y → f(y) < f(x)) ∧ (f 2 - f 3 = 7 / 24) := sorry

end monotone_decreasing_and_difference_l57_57416


namespace sum_in_Q_l57_57740

open Set

def is_set_P (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k
def is_set_Q (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k - 1
def is_set_M (x : ℤ) : Prop := ∃ k : ℤ, x = 4 * k + 1

variables (a b : ℤ)

theorem sum_in_Q (ha : is_set_P a) (hb : is_set_Q b) : is_set_Q (a + b) := 
sorry

end sum_in_Q_l57_57740


namespace collinear_projection_l57_57065

def vector_a : ℝ × ℝ × ℝ := (2, 2, -1)
def vector_b : ℝ × ℝ × ℝ := (-1, 4, 3)
def vector_p : ℝ × ℝ × ℝ := (40/29, 64/29, 17/29)

theorem collinear_projection :
  ∃ (p : ℝ × ℝ × ℝ),
    p = vector_p ∧
    (∃ (t : ℝ), p = (2 + t * (-3), 2 + t * 2, -1 + t * 4)) ∧
    (∃ (t1 t2 : ℝ), p = (2 + t1 * (-3), 2 + t1 * 2, -1 + t1 * 4) ∧ p = (-1 + t2 * 2, 4 + t2 * 1, 3 + t2 * 4)) := 
by
  sorry

end collinear_projection_l57_57065


namespace a_b_product_l57_57035

theorem a_b_product (a b : ℝ) (h1 : 2 * a - b = 1) (h2 : 2 * b - a = 7) : (a + b) * (a - b) = -16 :=
by
  -- The proof would be provided here.
  sorry

end a_b_product_l57_57035


namespace triangle_median_equiv_l57_57854

-- Assuming necessary non-computable definitions (e.g., α for angles, R for real numbers) and non-computable nature of some geometric properties.

noncomputable def triangle (A B C : ℝ) := 
A + B + C = Real.pi

noncomputable def length_a (R A : ℝ) : ℝ := 2 * R * Real.sin A
noncomputable def length_b (R B : ℝ) : ℝ := 2 * R * Real.sin B
noncomputable def length_c (R C : ℝ) : ℝ := 2 * R * Real.sin C

noncomputable def median_a (b c A : ℝ) : ℝ := (2 * b * c) / (b + c) * Real.cos (A / 2)

theorem triangle_median_equiv (A B C R : ℝ) (hA : triangle A B C) :
  (1 / (length_a R A) + 1 / (length_b R B) = 1 / (median_a (length_b R B) (length_c R C) A)) ↔ (C = 2 * Real.pi / 3) := 
by sorry

end triangle_median_equiv_l57_57854


namespace count_three_digit_numbers_divisible_by_9_l57_57461

theorem count_three_digit_numbers_divisible_by_9 : 
  (finset.filter (λ n, (n % 9 = 0)) (finset.range 1000).filter (λ n, 100 ≤ n)).card = 100 :=
by
  sorry

end count_three_digit_numbers_divisible_by_9_l57_57461


namespace g_is_odd_l57_57703

def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intros x
  -- proof goes here
  sorry

end g_is_odd_l57_57703


namespace imaginary_part_of_z_l57_57409

theorem imaginary_part_of_z : 
  let z := (1 + 3 * complex.i) / (3 - complex.i) in 
  complex.im z = 1 := 
by 
  sorry

end imaginary_part_of_z_l57_57409


namespace clock_stopped_time_l57_57748

noncomputable def alpha_hor (t : ℝ) : ℝ := t / 120
noncomputable def alpha_min (t : ℝ) : ℝ := t / 10
noncomputable def superimposed_condition (t : ℝ) (k : ℕ) : Prop := alpha_min t = alpha_hor t + 360 * k

theorem clock_stopped_time :
  ∃ t : ℝ, (32400 ≤ t ∧ t ≤ 46800) ∧ -- Within the timeframe 9:00 to 13:00 in seconds
  (∃ k : ℕ, 8 ≤ k ∧ k ≤ 12 ∧ superimposed_condition t k ∧ -- Valid k for superimposed condition
  (∃ b : ℝ, 0 < b ∧ b < 120 ∧ b = 6 * t - alpha_hor t)) ∧ -- Angle condition
  t = 35345 :=
begin
  sorry
end

end clock_stopped_time_l57_57748


namespace smallest_n_is_9_l57_57370

def smallest_n_satisfying_monochromatic_triangles (n : ℕ) : Prop :=
  ∀ (G : SimpleGraph (Fin n)), G.is_complete ∧ G.edge_coloring 2
  → ∃ (v : Fin n), ∃ (t1 t2 : SimpleGraph.Triangle G),
    t1.color = t2.color ∧
    (t1.vertex_indices ∪ t2.vertex_indices).card ≤ 5

theorem smallest_n_is_9 : smallest_n_satisfying_monochromatic_triangles 9 :=
sorry

end smallest_n_is_9_l57_57370


namespace three_digit_solution_count_l57_57796

open Nat

theorem three_digit_solution_count :
  let three_digit_range := (100:ℕ) to (999:ℕ)
  -- Statement of conditions
  ∃ n : ℕ,
    n = (three_digit_range.filter (λ x => (4897 * x + 603) % 29 = 1427 % 29)).length ∧
    n = 28 :=
by
  sorry

end three_digit_solution_count_l57_57796


namespace euler_line_isosceles_l57_57155

theorem euler_line_isosceles
  (ABC : Triangle)
  (O : Point) -- Circumcenter of triangle ABC
  (I : Point) -- Incenter of triangle ABC
  (H : Point) -- Orthocenter of triangle ABC
  (Euler_passes_through_I : EulerLine ABC O H ∋ I)
  : IsIsosceles ABC :=
by
  -- Definitions and sorry for now
  sorry

end euler_line_isosceles_l57_57155


namespace coordinate_difference_l57_57109

theorem coordinate_difference (m n : ℝ) (h : m = 4 * n + 5) :
  (4 * (n + 0.5) + 5) - m = 2 :=
by
  -- proof skipped
  sorry

end coordinate_difference_l57_57109


namespace ordered_sum_ways_l57_57098

theorem ordered_sum_ways (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 2) : 
  ∃ (ways : ℕ), ways = 70 :=
by
  sorry

end ordered_sum_ways_l57_57098


namespace solution_of_trig_equation_l57_57631

theorem solution_of_trig_equation (x : ℝ) :
  (sin x * sin (3 * x) + sin (4 * x) * sin (8 * x) = 0) ↔
  (∃ n : ℤ, x = n * π / 7) ∨ (∃ k : ℤ, x = k * π / 5) :=
begin
  sorry
end

end solution_of_trig_equation_l57_57631


namespace domain_of_function_l57_57205

noncomputable def is_domain_of_function (x : ℝ) : Prop :=
  (4 - x^2 ≥ 0) ∧ (x ≠ 1)

theorem domain_of_function :
  {x : ℝ | is_domain_of_function x} = {x : ℝ | -2 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_function_l57_57205


namespace integer_part_not_perfect_square_l57_57552

noncomputable def expr (n : ℕ) : ℝ :=
  2 * Real.sqrt (n + 1) / (Real.sqrt (n + 1) - Real.sqrt n)

theorem integer_part_not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = ⌊expr n⌋ :=
  sorry

end integer_part_not_perfect_square_l57_57552


namespace sum_squares_mod_divisor_l57_57982

-- Define the sum of the squares from 1 to 10
def sum_squares := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2)

-- Define the divisor
def divisor := 11

-- Prove that the remainder of sum_squares when divided by divisor is 0
theorem sum_squares_mod_divisor : sum_squares % divisor = 0 :=
by
  sorry

end sum_squares_mod_divisor_l57_57982


namespace increase_dimension_by_2_feet_l57_57119

-- Defining the initial dimensions
def initial_length := 13
def initial_width := 18

-- Defining the increase in dimension
variable (x : ℝ)

-- Defining the dimensions after increasing
def new_length := initial_length + x
def new_width := initial_width + x

-- Defining the areas
def area_each_room := new_length * new_width
def total_area := 4 * area_each_room + 2 * area_each_room

-- Stating the problem's total area
axiom total_area_axiom : total_area = 1800

-- The problem to prove
theorem increase_dimension_by_2_feet : x = 2 :=
begin
  sorry
end

end increase_dimension_by_2_feet_l57_57119


namespace no_valid_triples_exist_l57_57073

theorem no_valid_triples_exist :
  ∀ (a b c : ℤ), 3 ≤ a ∧ 2 ≤ b ∧ 1 ≤ c ∧ real.log b / real.log a = c^2 ∧ a + b + c = 2010 → false :=
by
  intros a b c
  intro h
  sorry

end no_valid_triples_exist_l57_57073


namespace part1_part2_l57_57020

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
noncomputable def g (a : ℝ) : ℝ := a^2 - a - 2

theorem part1 (x : ℝ) : (f x 3 > 6) ↔ (x < -4 ∨ x > 2) := by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ∈ set.Icc (-a) 1 → f x a ≤ g a) → a ≥ 3 := by
  sorry

end part1_part2_l57_57020


namespace tetrahedron_volume_and_height_l57_57639

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def volume_tetrahedron (A1 A2 A3 A4 : Point3D) : ℝ :=
  let v1 := (A2.x - A1.x, A2.y - A1.y, A2.z - A1.z)
  let v2 := (A3.x - A1.x, A3.y - A1.y, A3.z - A1.z)
  let v3 := (A4.x - A1.x, A4.y - A1.y, A4.z - A1.z)
  let tprod := v1.1 * ((v2.2 * v3.3) - (v2.3 * v3.2)) -
               v1.2 * ((v2.1 * v3.3) - (v2.3 * v3.1)) +
               v1.3 * ((v2.1 * v3.2) - (v2.2 * v3.1))
  (1 / 6) * |tprod|

def height_tetrahedron (A1 A2 A3 A4 : Point3D) (vol : ℝ) : ℝ :=
  let v1 := (A2.x - A1.x, A2.y - A1.y, A2.z - A1.z)
  let v2 := (A3.x - A1.x, A3.y - A1.y, A3.z - A1.z)
  let cross := (v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)
  let area := (1 / 2) * real.sqrt (cross.1^2 + cross.2^2 + cross.3^2)
  (3 * vol) / area

theorem tetrahedron_volume_and_height :
  let A1 := Point3D.mk 1 1 (-1)
  let A2 := Point3D.mk 2 3 1
  let A3 := Point3D.mk 3 2 1
  let A4 := Point3D.mk 5 9 (-8)
  volume_tetrahedron A1 A2 A3 A4 = 7.5 ∧
  height_tetrahedron A1 A2 A3 A4 (volume_tetrahedron A1 A2 A3 A4) = 45 / real.sqrt 17 :=
by
  -- Proof goes here
  sorry

end tetrahedron_volume_and_height_l57_57639


namespace number_of_digits_of_product_3_power_6_6_power_3_l57_57792

/-- Given the product 3^6 * 6^3, prove that the number of digits is equal to 6 -/
theorem number_of_digits_of_product_3_power_6_6_power_3 :
  (Nat.log10 (3^6 * 6^3)).natAbs + 1 = 6 := sorry

end number_of_digits_of_product_3_power_6_6_power_3_l57_57792


namespace tangent_line_parabola_l57_57013

theorem tangent_line_parabola (k : ℝ) 
  (h : ∀ (x y : ℝ), 4 * x + 6 * y + k = 0 → y^2 = 32 * x) : k = 72 := 
sorry

end tangent_line_parabola_l57_57013


namespace volunteer_A_not_in_community_A_probability_l57_57725

-- Define the setup for the problem
def volunteers := ["A", "B", "C", "D"]
def communities := ["A", "B", "C"]

-- State the conditions
-- 1. Each volunteer can only choose one community
-- 2. Each community must have at least one volunteer

theorem volunteer_A_not_in_community_A_probability :
  (probability (A ∉ community_A | allocation volunteers communities)) = 2 / 3 :=
sorry

end volunteer_A_not_in_community_A_probability_l57_57725


namespace geom_series_sum_correct_l57_57686

noncomputable def geometric_series_sum (b1 r : ℚ) (n : ℕ) : ℚ :=
b1 * (1 - r ^ n) / (1 - r)

theorem geom_series_sum_correct :
  geometric_series_sum (3/4) (3/4) 15 = 3177905751 / 1073741824 := by
sorry

end geom_series_sum_correct_l57_57686


namespace matrix_power_eq_l57_57465

theorem matrix_power_eq {a n : ℕ} (h : (matrix.stdBasisMatrix 3 3 ![1, 3, a] ![0, 1, 5] ![0, 0, 1]) ^ n = matrix.stdBasisMatrix 3 3 ![1, 15, 1010] ![0, 1, 25] ![0, 0, 1]) : a + n = 172 :=
by sorry

end matrix_power_eq_l57_57465


namespace count_3_digit_numbers_divisible_by_9_l57_57457

theorem count_3_digit_numbers_divisible_by_9 : 
  (finset.filter (λ x : ℕ, x % 9 = 0) (finset.Icc 100 999)).card = 100 := 
sorry

end count_3_digit_numbers_divisible_by_9_l57_57457


namespace more_than_16_segments_exist_l57_57954

theorem more_than_16_segments_exist :
  ∃ (points : Fin 13 → ℝ × ℝ) (segments : Finset (Fin 13 × Fin 13)),
    (∀ i j k : Fin 13, i ≠ j → i ≠ k → j ≠ k → ¬(collin points i j k)) ∧
    (∀ (s : Finset (Fin 13)), (4 ≤ s.card → ¬is_quadrilateral points s)) ∧
    16 < segments.card :=
sorry

def collin (points : Fin 13 → ℝ × ℝ) (i j k : Fin 13) : Prop := 
  let (x1, y1) := points i
  let (x2, y2) := points j
  let (x3, y3) := points k
  (x2 - x1) * (y3 - y1) = (y2 - y1) * (x3 - x1)

def is_quadrilateral (points : Fin 13 → ℝ × ℝ) (s : Finset (Fin 13)) : Prop :=
  ∃ (a b c d : Fin 13), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  ¬(collin points a b c) ∧ ¬(collin points b c d) ∧ 
  ¬(collin points c d a) ∧ ¬(collin points d a b)

end more_than_16_segments_exist_l57_57954


namespace who_broke_the_glass_l57_57724

def BaoBao_statement (K: Prop) := K
def KeKe_statement (M: Prop) := M
def MaoMao_statement (K: Prop) := ¬K
def DuoDuo_statement (D: Prop) := ¬D

theorem who_broke_the_glass (B K M D : Prop)
  (BaoBao : BaoBao_statement K)
  (KeKe : KeKe_statement M)
  (MaoMao : MaoMao_statement K)
  (DuoDuo : DuoDuo_statement D)
  (one_truth : (B ↔ (BaoBao)) ∨ (K ↔ (KeKe)) ∨ (M ↔ (MaoMao)) ∨ (D ↔ (DuoDuo))
  ∧ (B ↔ not (KeKe ∧ MaoMao ∧ DuoDuo))) :
  M :=
sorry

end who_broke_the_glass_l57_57724


namespace intervals_of_monotonicity_f_minus1_ge_half_l57_57766

-- Define the function f(x) given the parameter 'a'
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (x^2) / 2 - (a + 1) * x

-- Define the conditions for the intervals of monotonicity
theorem intervals_of_monotonicity (a : ℝ) (h_a : a > 0) :
  ((0 < a ∧ a < 1) → (∀ x, (0 < x ∧ x < a) → ∂(f a x) / ∂x > 0) ∧ 
                    (∀ x, (a < x ∧ x < 1) → ∂(f a x) / ∂x < 0) ∧ 
                    (∀ x, (1 < x) → ∂(f a x) / ∂x > 0)) ∧
  (a = 1 → ∀ x, (0 < x) → ∂(f a x) / ∂x ≥ 0) ∧
  ((a > 1) → (∀ x, (0 < x ∧ x < 1) → ∂(f a x) / ∂x > 0) ∧ 
            (∀ x, (1 < x ∧ x < a) → ∂(f a x) / ∂x < 0) ∧ 
            (∀ x, (a < x) → ∂(f a x) / ∂x > 0)) :=
sorry

-- Define the function f(x) for the specific case when a = -1
noncomputable def f_minus1 (x : ℝ) : ℝ := -Real.log x + (x^2) / 2

-- State the theorem that f(x) ≥ 1/2 when a = -1
theorem f_minus1_ge_half (x : ℝ) (hx : 0 < x) : f_minus1 x ≥ 1 / 2 :=
sorry

end intervals_of_monotonicity_f_minus1_ge_half_l57_57766


namespace rhombus_diagonals_perpendicular_certain_l57_57207

-- Define what a rhombus is in the context of the problem.
def is_rhombus (Q : Type) [quadrilateral Q] : Prop :=
  ∃ a b c d : Q, (diagonals_perpendicular a b c d)

-- Define what it means for the event of drawing a rhombus with perpendicular diagonals.
def drawing_rhombus_with_perpendicular_diagonals : Prop := 
  ∀ (Q : Type) [quadrilateral Q], is_rhombus Q → (∃ a b c d : Q, diagonals_perpendicular a b c d)

-- The theorem stating that drawing a rhombus with its diagonals perpendicular is certain.
theorem rhombus_diagonals_perpendicular_certain :
  drawing_rhombus_with_perpendicular_diagonals :=
by
  sorry

end rhombus_diagonals_perpendicular_certain_l57_57207


namespace range_m_l57_57700

noncomputable def even_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f x = f (-x)

noncomputable def decreasing_on_non_neg (f : ℝ → ℝ) : Prop := 
  ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_m (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_dec : decreasing_on_non_neg f) :
  ∀ m, f (1 - m) < f m → m < 1 / 2 :=
by
  sorry

end range_m_l57_57700


namespace series_sum_formula_l57_57744

theorem series_sum_formula (n : ℕ) (hn: n > 0) : 
  (∑ k in Finset.range n, (k + 1) * (1 / ((k + 2)! : ℚ))) = 1 - 1 / ((n + 1)! : ℚ) :=
sorry

end series_sum_formula_l57_57744


namespace firecracker_confiscation_l57_57124

variables
  (F : ℕ)   -- Total number of firecrackers bought
  (R : ℕ)   -- Number of firecrackers remaining after confiscation
  (D : ℕ)   -- Number of defective firecrackers
  (G : ℕ)   -- Number of good firecrackers before setting off half
  (C : ℕ)   -- Number of firecrackers confiscated

-- Define the conditions:
def conditions := 
  F = 48 ∧
  D = R / 6 ∧
  G = 2 * 15 ∧
  R - D = G ∧
  F - R = C

-- The theorem to prove:
theorem firecracker_confiscation (h : conditions F R D G C) : C = 12 := 
  sorry

end firecracker_confiscation_l57_57124


namespace company_b_profit_l57_57692

-- Definitions as per problem conditions
def A_profit : ℝ := 90000
def A_share : ℝ := 0.60
def B_share : ℝ := 0.40

-- Theorem statement to be proved
theorem company_b_profit : B_share * (A_profit / A_share) = 60000 :=
by
  sorry

end company_b_profit_l57_57692


namespace problem1_problem2_l57_57052

open Real

-- Defining the function f
def f (x : ℝ) : ℝ := abs (2 * x - 1)

-- Problem 1 Statement: Proving m = 3/2 given the solution set of the transformed inequality
theorem problem1 (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, (f (x + 1/2) ≥ 2 * m + 1) ↔ x ∈ Iic (-2) ∨ x ∈ Ici 2) → m = 3/2 :=
sorry

-- Problem 2 Statement: Proving minimum value of a is 4 given the inequality
theorem problem2 (a : ℝ) :
  (∀ x y : ℝ, f(x) ≤ 2^y + a / 2^y + abs (2 * x + 3)) → a ≥ 4 :=
sorry

end problem1_problem2_l57_57052


namespace range_of_f_in_domain_l57_57948

-- Define the function and the domain
def f (x : ℝ) : ℝ := 1 - 2 * x

-- Define the starting and ending points of the domain
def x_start : ℝ := 1
def x_end : ℝ := 2

-- Define the range that we need to prove
def range_f : Set ℝ := {y | ∃ (x : ℝ), x_start ≤ x ∧ x ≤ x_end ∧ y = f(x)}

-- State the problem
theorem range_of_f_in_domain : range_f = Set.Icc (-3) (-1) :=
by {
  sorry
}

end range_of_f_in_domain_l57_57948


namespace range_of_a_l57_57056

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 4 * x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ 3 :=
sorry

end range_of_a_l57_57056


namespace product_increases_exactly_13_times_by_subtracting_3_l57_57107

theorem product_increases_exactly_13_times_by_subtracting_3 :
  ∃ (n1 n2 n3 n4 n5 n6 n7 : ℕ),
    13 * (n1 * n2 * n3 * n4 * n5 * n6 * n7) =
      ((n1 - 3) * (n2 - 3) * (n3 - 3) * (n4 - 3) * (n5 - 3) * (n6 - 3) * (n7 - 3)) :=
sorry

end product_increases_exactly_13_times_by_subtracting_3_l57_57107


namespace range_of_a_l57_57419

noncomputable def f (a x : ℝ) : ℝ := a^2 * x - 2 * a + 1

theorem range_of_a (a : ℝ) : (¬ ∀ x : ℝ, x ∈ set.Icc 0 1 → f a x > 0) → a ≥ 1 / 2 :=
by
  sorry

end range_of_a_l57_57419


namespace case_a_sticks_case_b_square_l57_57601
open Nat 

premise n12 : Nat := 12
premise sticks12_sum : Nat := (n12 * (n12 + 1)) / 2  -- Sum of first 12 natural numbers
premise length_divisibility_4 : ¬ (sticks12_sum % 4 = 0)  -- Check if sum is divisible by 4

-- Need to break at least 2 sticks to form a square
theorem case_a_sticks (h : sticks12_sum = 78) (h2 : length_divisibility_4 = true) : 
  ∃ (k : Nat), k >= 2 := sorry

premise n15 : Nat := 15
premise sticks15_sum : Nat := (n15 * (n15 + 1)) / 2  -- Sum of first 15 natural numbers
premise length_divisibility4_b : sticks15_sum % 4 = 0  -- Check if sum is divisible by 4

-- Possible to form a square without breaking any sticks
theorem case_b_square (h : sticks15_sum = 120) (h2 : length_divisibility4_b = true) : 
  ∃ (k : Nat), k = 0 := sorry

end case_a_sticks_case_b_square_l57_57601


namespace tom_ate_one_pound_of_carrots_l57_57967

noncomputable def calories_from_carrots (C : ℝ) : ℝ := 51 * C
noncomputable def calories_from_broccoli (C : ℝ) : ℝ := (51 / 3) * (2 * C)
noncomputable def total_calories (C : ℝ) : ℝ :=
  calories_from_carrots C + calories_from_broccoli C

theorem tom_ate_one_pound_of_carrots :
  ∃ C : ℝ, total_calories C = 85 ∧ C = 1 :=
by
  use 1
  simp [total_calories, calories_from_carrots, calories_from_broccoli]
  sorry

end tom_ate_one_pound_of_carrots_l57_57967


namespace rational_multiples_of_pi_l57_57533

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
def A (n : ℕ) : ℝ := (real.cos (n * x) + real.cos (n * y))

theorem rational_multiples_of_pi 
  (hx : x ∈ ℝ)
  (hy : y ∈ ℝ)
  (finite_set : set.finite {A n | n > 0}) : 
  ∃ r₁ r₂ : ℚ, x = r₁ * real.pi ∧ y = r₂ * real.pi := 
sorry

end rational_multiples_of_pi_l57_57533


namespace find_side_b_l57_57852

noncomputable def angle_sum_property (A B C : ℝ) [fact (0 < A)] [fact (A < π)]
                          [fact (0 < B)] [fact (B < π)]
                          [fact (0 < C)] [fact (C < π)] : Prop :=
A + B + C = π

noncomputable def law_of_sines (a b c : ℝ) (A B C : ℝ) : Prop :=
(a / Real.sin A) = (b / Real.sin B) ∧ (b / Real.sin B) = (c / Real.sin C) ∧ (a / Real.sin A) = (c / Real.sin C)

theorem find_side_b 
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : A = π / 4)
  (h2 : C = π / 6)
  (h3 : c = 10)
  (h4 : angle_sum_property A B C)
  (h5 : law_of_sines a b c A B C)
  : b = 5 * (Real.sqrt 6 + Real.sqrt 2) := 
sorry

end find_side_b_l57_57852


namespace proportion_of_overfilled_cars_is_40_minimum_proportion_of_passengers_in_overfilled_cars_is_53_proportion_passengers_in_overfilled_not_less_than_cars_l57_57213

structure CarDistribution :=
  (range : string) (percentage : ℝ)

def data : List CarDistribution := [
  ⟨"10 to 19", 4⟩,
  ⟨"20 to 29", 6⟩,
  ⟨"30 to 39", 12⟩,
  ⟨"40 to 49", 18⟩,
  ⟨"50 to 59", 20⟩,
  ⟨"60 to 69", 20⟩,
  ⟨"70 to 79", 14⟩,
  ⟨"80 or more", 6⟩
]

noncomputable def overfilledProportion : ℝ :=
  data.filter (fun d => d.range ≥ "60").sumBy (λ d => d.percentage)

theorem proportion_of_overfilled_cars_is_40 :
  overfilledProportion = 40 := by
  sorry

noncomputable def minProportionPassengersInOverfilled : ℝ :=
  (60 * (data.filter (fun d => d.range = "60 to 69")).sumBy (λ d => d.percentage)
  + 70 * (data.filter (fun d => d.range = "70 to 79")).sumBy (λ d => d.percentage)
  + 80 * (data.filter (fun d => d.range = "80 or more")).sumBy (λ d => d.percentage))
  / (10 * (data.filter (fun d => d.range = "10 to 19")).sumBy (λ d => d.percentage)
  + 20 * (data.filter (fun d => d.range = "20 to 29")).sumBy (λ d => d.percentage)
  + 30 * (data.filter (fun d => d.range = "30 to 39")).sumBy (λ d => d.percentage)
  + 40 * (data.filter (fun d => d.range = "40 to 49")).sumBy (λ d => d.percentage)
  + 50 * (data.filter (fun d => d.range = "50 to 59")).sumBy (λ d => d.percentage)
  + 60 * (data.filter (fun d => d.range = "60 to 69")).sumBy (λ d => d.percentage)
  + 70 * (data.filter (fun d => d.range = "70 to 79")).sumBy (λ d => d.percentage)
  + 80 * (data.filter (fun d => d.range = "80 or more")).sumBy (λ d => d.percentage))

theorem minimum_proportion_of_passengers_in_overfilled_cars_is_53 :
  minProportionPassengersInOverfilled = 0.53 := by
  sorry

theorem proportion_passengers_in_overfilled_not_less_than_cars :
  ∀ (totalCars passInOverfilled totalPassengers proportionOverfilledCars proportionOverfilledPassengers : ℝ),
    totalCars > 0 → totalPassengers > 0 →
    proportionOverfilledCars = (data.filter (fun d => d.range ≥ "60")).sumBy (λ d => d.percentage) →
    proportionOverfilledCars = 0.40 →
    proportionOverfilledPassengers = minProportionPassengersInOverfilled →
    proportionOverfilledPassengers >= proportionOverfilledCars := by
  sorry

end proportion_of_overfilled_cars_is_40_minimum_proportion_of_passengers_in_overfilled_cars_is_53_proportion_passengers_in_overfilled_not_less_than_cars_l57_57213


namespace union_complement_set_l57_57426

theorem union_complement_set (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5}) 
  (hA : A = {1, 2, 3, 5}) (hB : B = {2, 4}) :
  (U \ A) ∪ B = {0, 2, 4} :=
by
  rw [Set.diff_eq, hU, hA, hB]
  simp
  sorry

end union_complement_set_l57_57426


namespace solution1_solution2_l57_57761

noncomputable def problem1 (x y : ℝ) (p : ℝ) : Prop :=
  x - 2 * y + 1 = 0 ∧ y^2 = 2 * p * x ∧ 0 < p ∧ (abs (sqrt (1 + 4) * (y - y))) = 4 * sqrt 15

theorem solution1 (p: ℝ) : p = 2 :=
  sorry

noncomputable def problem2 (x y m n : ℝ) : Prop :=
  y^2 = 4 * x ∧ ∃ (F : ℝ × ℝ), F = (1, 0) ∧
  (∀ (M N : ℝ × ℝ), M ∈ y^2 = 4 * x ∧ N ∈ y^2 = 4 * x ∧ (F.1 - M.1) * (F.2 - N.1) + (F.2 - M.2) * (F.2 - N.2) = 0 →
  let area := (1/2) * abs ((N.1 - M.1) * (F.2 - M.2) - (N.2 - M.2) * (F.1 - M.1)) in
  ∃ min_area : ℝ, min_area = 12 - 8 * sqrt 2)

theorem solution2 (x y m n : ℝ) : ∃ min_area : ℝ, min_area = 12 - 8 * sqrt 2 :=
  sorry

end solution1_solution2_l57_57761


namespace probability_origin_not_in_convex_hull_l57_57554

open ProbabilityTheory
open Set

noncomputable def S1 : Set ℂ := {z : ℂ | complex.abs z = 1}

theorem probability_origin_not_in_convex_hull :
  ∀ (points : Fin 7 → ℂ) (h_points : ∀ i, points i ∈ S1),
  ∃ p : ℝ, p = 57/64 := by
    sorry

end probability_origin_not_in_convex_hull_l57_57554


namespace distinct_sums_proof_7_l57_57958

def num_of_distinct_sums (coins1 : ℕ) (coins2 : ℕ) (n : ℕ) : ℕ :=
  ((list.range (n+1)).map (λ x, 1 * (n - x) + 0.5 * x)).erase_dup.length

theorem distinct_sums_proof_7 (h1 : coins1 = 5) (h2 : coins2 = 6) (h3 : n = 6) :
  num_of_distinct_sums coins1 coins2 n = 7 :=
by
  sorry

end distinct_sums_proof_7_l57_57958


namespace rectangular_garden_length_l57_57634

theorem rectangular_garden_length (L P B : ℕ) (h1 : P = 600) (h2 : B = 150) (h3 : P = 2 * (L + B)) : L = 150 :=
by
  sorry

end rectangular_garden_length_l57_57634


namespace f_second_derivative_at_zero_l57_57843

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 2 else 2 * (2^(1/7))^(n - 1)

noncomputable def f (x : ℝ) : ℝ :=
  x * (x - a_n 1) * (x - a_n 2) * (x - a_n 3) * (x - a_n 4) * (x - a_n 5) * (x - a_n 6) * (x - a_n 7) * (x - a_n 8)

theorem f_second_derivative_at_zero : f'' 0 = 2^12 := sorry

end f_second_derivative_at_zero_l57_57843


namespace range_of_a_l57_57773

theorem range_of_a (a : ℝ) :
    (∀ x1 x2 : ℝ, x1 < x2 → 
    (ite (x1 < 1) ((3 * a - 1) * x1 + 4 * a) (log a x1) - 
    ite (x2 < 1) ((3 * a - 1) * x2 + 4 * a) (log a x2) > 0)) 
    → (1 / 7 ≤ a ∧ a < 1 / 3) :=
sorry

end range_of_a_l57_57773


namespace circle_radius_l57_57572

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance formula for two points
def dist (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

-- Points given in the problem
def point1 : Point := ⟨2, 3⟩
def point2 : Point := ⟨3, 1⟩

-- Center of the circle is on the x-axis; hence its y-coordinate is 0
def center : Point := ⟨3 / 2, 0⟩

theorem circle_radius :
  dist center point1 = dist center point2 ∧ dist center point1 = Real.sqrt 9.25 :=
by
  sorry

end circle_radius_l57_57572


namespace sequence_formula_l57_57425

theorem sequence_formula (a : ℕ+ → ℕ) (h₁ : a 1 = 2) (h₂ : a 2 = 3) (h₃ : ∀ n : ℕ+, n ≥ 3 → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  ∀ n : ℕ+, a n = 2 ^ (n - 1) + 1 :=
by sorry

end sequence_formula_l57_57425


namespace parallel_medians_half_angle_l57_57975

theorem parallel_medians_half_angle {ABC A'B'C' : Triangle}
  (h1 : ABC.is_right_triangle)
  (h2 : A'B'C'.is_right_triangle)
  (h3 : ABC.median_parallel_to_hypotenuse = A'B'C'.median_parallel_to_hypotenuse) :
  ∃ leg1ABC leg2A'B'C' hypotenuse_ABC hypotenuse_A'B'C',
  ∠(leg1ABC, leg2A'B'C') = 1/2 * ∠(hypotenuse_ABC, hypotenuse_A'B'C') := 
sorry

end parallel_medians_half_angle_l57_57975


namespace cos_D_zero_l57_57473

noncomputable def area_of_triangle (a b: ℝ) (sinD: ℝ) : ℝ := 1 / 2 * a * b * sinD

theorem cos_D_zero (DE DF : ℝ) (D : ℝ) (h1 : area_of_triangle DE DF (Real.sin D) = 98) (h2 : Real.sqrt (DE * DF) = 14) : Real.cos D = 0 :=
  by
  sorry

end cos_D_zero_l57_57473


namespace proof_expression1_proof_expression2_l57_57337

-- Definitions for exponential properties
def expression1_part1 := (2 + 3/5) ^ 0
def expression1_part2 := 2 ^ (-2) * abs (-0.064) ^ (1/3)
def expression1_part3 := (9/4) ^ (1/2)

-- Definitions for logarithmic properties
def lg(x: ℝ) := log x / log 10

def expression2_part1 := (lg 2) ^ 2
def expression2_part2 := lg 2 * lg 5
def expression2_part3 := lg 5
def expression2_part4 := 3 -- since 2 ^ (log 2 3) = 3
def expression2_part5 := log 2 (1/8) -- since log 2 (1/8) = -3

-- Composing expressions
def expression1 := expression1_part1 + expression1_part2 - expression1_part3
def expression2 := expression2_part1 + expression2_part2 + expression2_part3 - expression2_part4 * expression2_part5

theorem proof_expression1 : expression1 = -0.4 := by sorry
theorem proof_expression2 : expression2 = 11 := by sorry

end proof_expression1_proof_expression2_l57_57337


namespace calculate_expression_l57_57732

theorem calculate_expression (x : ℝ) (hx : x = Real.log 3 / Real.log 4) : (2^x - 2^(-x))^2 = 4 / 3 :=
by
  sorry

end calculate_expression_l57_57732


namespace gift_equation_l57_57995

theorem gift_equation (x : ℝ) : 15 * (x + 40) = 900 := 
by
  sorry

end gift_equation_l57_57995


namespace max_value_of_f_maximum_achieved_l57_57568

noncomputable def f : ℝ → ℝ := λ x, -2 * x^2 + 6 * x

theorem max_value_of_f :
  (x : ℝ) → (h : -2 < x ∧ x ≤ 2) → f x ≤ 9 / 2 :=
by
  sorry

theorem maximum_achieved :
  (h : -2 < (3 / 2 : ℝ) ∧ (3 / 2 : ℝ) ≤ 2) → f (3 / 2) = 9 / 2 :=
by
  sorry

end max_value_of_f_maximum_achieved_l57_57568


namespace convert_and_sum_correct_l57_57365

theorem convert_and_sum_correct:
  let a := 2 * 8^2 + 5 * 8^1 + 4 * 8^0,
      b := 1 * 4^1 + 3 * 4^0,
      c := 1 * 5^2 + 3 * 5^1 + 2 * 5^0,
      d := 2 * 3^1 + 2 * 3^0
  in ⌊(a / b) + (c / d)⌋ = 29 :=
by
  sorry

end convert_and_sum_correct_l57_57365


namespace hyperbola_line_eq_l57_57027

/-- Statement: 
Given a hyperbola \( \frac{x^2}{4} - y^2 = 1 \), prove that the line passing through one 
of its foci and parallel to one of its asymptotes has the equation \( y = -\frac{1}{2}x + \frac{\sqrt{5}}{2} \).
-/
theorem hyperbola_line_eq :
  ∀ l : ℝ → ℝ, (∃ c : ℝ, l c = 0 ∧ 
  (∀ x, l x = - (1 / 2) * x + sqrt 5 / 2) ∧ 
  (∀ x y, ((x^2 / 4) - y^2 = 1) ∧ 
  (l = λ x, - (1 / 2) * x + sqrt 5 / 2))) :=
begin
  sorry
end

end hyperbola_line_eq_l57_57027


namespace range_of_a_l57_57058

noncomputable def f (x : ℝ) : ℝ := x - (1 / x)

theorem range_of_a :
  (∀ x ∈ set.Icc (1 : ℝ) (3 / 2), f (a * x - 1) > f 2) ↔
  (a ∈ set.Ioo (1/2 : ℝ) (2/3) ∪ set.Ioi (3 : ℝ)) :=
sorry

end range_of_a_l57_57058


namespace question_1_question_2_question_3_l57_57739

def A := {x : ℕ | 0 < x ∧ 1 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 3}
def B := {x : ℕ | 0 < x ∧ -3 < x - 6 ∧ x - 6 < 3}

theorem question_1 : (A.card × B.card) - (A ∩ B).card = 21 := 
by sorry

theorem question_2 : (A ∪ B).card.choose 3 = 20 := 
by sorry

theorem question_3 : ∑ x in A, (choose 3 (B.erase x)).card * (10 * (dec_formulae.erase [x])).card = 300 :=
by sorry

end question_1_question_2_question_3_l57_57739


namespace domain_of_function_l57_57204

def log_domain {b : ℝ} (f : ℝ → ℝ) (g : ℝ → ℝ) (lb ub : ℝ) :=
  ∀ x, lb < x ∧ x < ub ↔ f (g x) > 0

theorem domain_of_function :
  log_domain (λ y, 1 / (Real.sqrt y)) (λ x, Real.logb 0.5 (4 * x - 3)) (3 / 4) 1 :=
by
  sorry

end domain_of_function_l57_57204


namespace triangle_similar_AEF_DEF_l57_57154

-- Definitions for the conditions
variables (A B C D E F : Type) [Square ABCD] [angle BEF = 90]

-- The theorem statement
theorem triangle_similar_AEF_DEF : (triangle ABE ∼ triangle DEF) :=
sorry

end triangle_similar_AEF_DEF_l57_57154


namespace fishing_problem_solution_l57_57964

-- Definitions for the conditions
variables (A B C : Type*) (catch_most catch_least : A)
variables (statements : A → Prop)
variables [linear_order ℕ] -- Order structure for the number of fish caught

-- Condition that none of them caught the same number of fish
def distinct_catches (a_catch b_catch c_catch : ℕ) : Prop :=
  a_catch ≠ b_catch ∧ b_catch ≠ c_catch ∧ a_catch ≠ c_catch

-- Relevant statements by A, B, and C
def statement_A_1 (catch_A catch_B catch_C : ℕ) : Prop := catch_A > catch_B ∧ catch_A > catch_C
def statement_A_2 (catch_C catch_B : ℕ) : Prop := catch_C < catch_B

def statement_B_1 (catch_B : ℕ) : Prop := catch_B > 0
def statement_B_2 (catch_A catch_B catch_C : ℕ) : Prop := catch_B > catch_A + catch_C

def statement_C_1 (catch_C : ℕ) : Prop := catch_C > 0
def statement_C_2 (catch_C catch_B : ℕ) : Prop := catch_B = catch_C / 2

-- Statements are true if exactly 3 out of the 6 statements are true
def three_true_statements (catch_A catch_B catch_C : ℕ) : Prop :=
  (statement_A_1 catch_A catch_B catch_C) + (statement_A_2 catch_C catch_B)
  + (statement_B_1 catch_B) + (statement_B_2 catch_A catch_B catch_C)
  + (statement_C_1 catch_C) + (statement_C_2 catch_C catch_B) = 3

-- Proof statement
theorem fishing_problem_solution (catch_A catch_B catch_C : ℕ)
  (distinct : distinct_catches catch_A catch_B catch_C)
  (three_true : three_true_statements catch_A catch_B catch_C) :
  (catch_B > catch_A ∧ catch_B > catch_C) ∧ (catch_C < catch_A ∧ catch_C < catch_B) :=
sorry

end fishing_problem_solution_l57_57964


namespace geom_seq_sum_first_five_l57_57826

noncomputable def geometric_sequence_sum (q : ℕ) : Prop :=
  let a : ℕ → ℕ := λ n, 3 * q ^ (n - 1)
  a 1 + a 2 + a 3 = 21 ∧ 
  a 1 = 3 ∧ 
  a 3 = 3 * q ^ 2 ∧ 
  a 4 = 3 * q ^ 3 ∧ 
  a 5 = 3 * q ^ 4

theorem geom_seq_sum_first_five (q : ℕ) (h : geometric_sequence_sum q) :  (3 * q^2) + (3 * q^3) + (3 * q^4) = 84 :=
sorry

end geom_seq_sum_first_five_l57_57826


namespace find_sixth_number_l57_57636

theorem find_sixth_number (avg_all : ℝ) (avg_first6 : ℝ) (avg_last6 : ℝ) (total_avg : avg_all = 10.7) (first6_avg: avg_first6 = 10.5) (last6_avg: avg_last6 = 11.4) : 
  let S1 := 6 * avg_first6
  let S2 := 6 * avg_last6
  let total_sum := 11 * avg_all
  let X := total_sum - (S1 - X + S2 - X)
  X = 13.7 :=
by 
  sorry

end find_sixth_number_l57_57636


namespace sqrt_sum_equality_l57_57265

theorem sqrt_sum_equality :
  (Real.sqrt (9 - 6 * Real.sqrt 2) + Real.sqrt (9 + 6 * Real.sqrt 2) = 2 * Real.sqrt 6) :=
by
  sorry

end sqrt_sum_equality_l57_57265


namespace bounds_of_T_l57_57133

def f (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ f x = y}

theorem bounds_of_T :
  let N := 3
  let n := 4 / 3 in
  n ∈ T ∧ N ∉ T := by
  sorry

end bounds_of_T_l57_57133


namespace radicals_like_simplest_forms_l57_57047

theorem radicals_like_simplest_forms (a b : ℝ) (h1 : 2 * a + b = 7) (h2 : a = b + 2) :
  a = 3 ∧ b = 1 :=
by
  sorry

end radicals_like_simplest_forms_l57_57047


namespace compute_relationship_l57_57105

noncomputable def relationship_x_y (r x y : ℝ) : Prop :=
  ∃ (AB AD AC AF FC : ℝ),
  AB = 2 * r ∧
  AD = 2 * r / 3 ∧
  AC = AD + 8 * r / 9 ∧
  AF = AC / 2 ∧
  FC = AF ∧
  x = AF ∧
  y = r ∧
  y = 9 * x / 7

theorem compute_relationship (r x y : ℝ) (h : relationship_x_y r x y) : y = 9 * x / 7 :=
by
  cases h with AB h,
  cases h with AD h,
  cases h with AC h,
  cases h with AF h,
  cases h with FC h,
  cases h with hx hy,
  cases hy,
  exact hy_right

end compute_relationship_l57_57105


namespace coeff_x3_expansion_eq_coef_l57_57285

theorem coeff_x3_expansion (x : ℝ) :
  (∑ k in Finset.range 10, (Nat.choose 9 k) * (-1)^k * x^(9 - 2 * k)) =∑ m in Finset.range 1, (if m = 3 then -84 else 0) :=
  sorry

theorem eq_coef (n : ℕ) :
  (Nat.choose n 2 = Nat.choose n 5) → n = 7 :=
  sorry

end coeff_x3_expansion_eq_coef_l57_57285


namespace possible_shapes_opqr_l57_57787

theorem possible_shapes_opqr (x1 y1 x2 y2 : ℝ)
  (h_distinct: ¬(x1, y1) = (x2, y2) ∧ ¬(x2, y2) = (x1 + x2, y1 + y2) ∧ ¬(x1, y1) = (x1 + x2, y1 + y2)):
  let P := (x1, y1)
  let Q := (x2, y2)
  let R := (x1 + x2, y1 + y2)
  in  -- These points form the quadrilateral OPQR
  (exists k1 k2 : ℝ, y1 = k1 * x1 ∧ y2 = k2 * x2 ∧
    ((k1 = k2 ∧ R = (x1 + x2, k1 * (x1 + x2))) ∨ (k1 ≠ k2 ∧ (x1 * y2 = x2 * y1)))) :=
sorry

end possible_shapes_opqr_l57_57787


namespace rhombus_area_from_intersecting_equilateral_triangles_l57_57972

theorem rhombus_area_from_intersecting_equilateral_triangles :
  ∀ (s : ℕ), 
  (∀ (t₁ t₂ : finset (fin (s * s))),
    finset.card t₁ = 3 ∧ finset.card t₂ = 3 ∧ ∀ x ∈ t₁ ∩ t₂, ∀ y, 
    ∃ h : fin (s * s), t₁ x = h ∧ t₂ y = h ∧
    (∀ a : ℕ, a ∈ t₁ → a.1.1 = s) ∧ 
    (∀ b : ℕ, b ∈ t₂ → b.1.1 = -s)) →
  (∀ s = 4,
    (4 - s) * (s*ℝ.sqrt 3 - 4) = (8*ℝ.sqrt 3 - 8))
  ∧ s = 4 → 
 sorry

end rhombus_area_from_intersecting_equilateral_triangles_l57_57972


namespace modulus_of_fraction_l57_57384

noncomputable def z1 : ℂ := 2 + complex.i
noncomputable def z2 : ℂ := 2 + 2 * complex.i

theorem modulus_of_fraction (z2_eq : z2 = 2 + 2 * complex.i) :
  (complex.abs (z1 / z2) = real.sqrt 10 / 4) :=
by
  sorry

end modulus_of_fraction_l57_57384


namespace not_set_of_difficult_problems_l57_57260

-- Define the context and entities
inductive Exercise
| ex (n : Nat) : Exercise  -- Example definition for exercises, assumed to be numbered

def is_difficult (ex : Exercise) : Prop := sorry  -- Placeholder for the subjective predicate

-- Define the main problem statement
theorem not_set_of_difficult_problems
  (Difficult : Exercise → Prop) -- Subjective predicate defining difficult problems
  (H_subj : ∀ (e : Exercise), (Difficult e ↔ is_difficult e)) :
  ¬(∃ (S : Set Exercise), ∀ e, e ∈ S ↔ Difficult e) :=
sorry

end not_set_of_difficult_problems_l57_57260


namespace even_numbers_between_150_and_350_l57_57072

theorem even_numbers_between_150_and_350 : 
  let smallest_even := 152
  let largest_even := 348
  (∃ n, (2 * n > 150) ∧ (2 * n < 350) ∧ (n <= 174)) →
  (∑ n in (finset.range 100).filter (λ n, (2 * (75 + n) > 150) ∧ (2 * (75 + n) < 350)), n) = 99 :=
by
  sorry

end even_numbers_between_150_and_350_l57_57072


namespace determinant_of_matrix_A_l57_57344

def A : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    [-2, 4, -1],
    [5, -3, 2],
    [0, 6, -3]
  ]

theorem determinant_of_matrix_A : A.det = 72 := by
  sorry

end determinant_of_matrix_A_l57_57344


namespace min_huge_count_l57_57504

-- Define the vertices and the numbering condition
def vertices := Finset.range 300

-- Define the numbering condition: Bijective function from vertices to 1..300 respecting the problem constraints
def numbering (v : Fin 300) : Fin 300 → ℕ :=
  λ i, (i + v.val) % 300 + 1

-- Define the condition that for each number, equal number of smaller numbers are among the 15 closest clockwise and counterclockwise
def balanced (numbering : Fin 300 → ℕ) : Prop :=
  ∀ v, let cw := Finset.Ico (v + 1) (v + 16) in
       let ccw := Finset.Ico (v + 1 - 15) (v + 1) in
       (Finset.filter (λ i, numbering i < numbering v) cw).card =
       (Finset.filter (λ i, numbering i < numbering v) ccw).card

-- Define a "huge" number
def is_huge (numbering : Fin 300 → ℕ) (v : Fin 300) : Prop :=
  ∀ i, (i ∈ Finset.Ico (v + 1 - 15) (v + 1 + 15)) → numbering i < numbering v

-- Define the minimum number of huge numbers condition
def min_huge_numbers (numbering : Fin 300 → ℕ) : Prop :=
  (∃ n, n = 10 ∧ ∃ (S : Finset (Fin 300)), S.card = n ∧ ∀ v ∈ S, is_huge numbering v)

-- The final theorem
theorem min_huge_count (numbering_condition : ∀ (v : Fin 300), balanced (numbering v)) :
  min_huge_numbers (numbering) := by
  sorry

end min_huge_count_l57_57504


namespace contrapositive_of_real_roots_l57_57876

theorem contrapositive_of_real_roots (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0 :=
sorry

end contrapositive_of_real_roots_l57_57876


namespace find_prime_pairs_l57_57000

theorem find_prime_pairs (p q : ℕ) (p_prime : Nat.Prime p) (q_prime : Nat.Prime q) 
  (h1 : ∃ a : ℤ, a^2 = p - q)
  (h2 : ∃ b : ℤ, b^2 = p * q - q) : 
  (p, q) = (3, 2) :=
by {
    sorry
}

end find_prime_pairs_l57_57000


namespace total_cost_of_plates_and_cups_l57_57274

theorem total_cost_of_plates_and_cups 
  (P C : ℝ)
  (h : 100 * P + 200 * C = 7.50) :
  20 * P + 40 * C = 1.50 :=
by
  sorry

end total_cost_of_plates_and_cups_l57_57274


namespace equation_has_no_real_solutions_l57_57704

/-- Prove that the graph of the given equation is empty.

Given the equation:
x^2 + 3y^2 - 4x - 6y + 10 = 0,
we need to show that there are no real (x, y) solutions that satisfy this equation.
-/
theorem equation_has_no_real_solutions :
  ∀ (x y : ℝ), x^2 + 3 * y^2 - 4 * x - 6 * y + 10 ≠ 0 :=
by {
  assume x y,
  /- The proof steps would show that the transformed equation cannot be satisfied by any real (x, y) -/
  sorry
}

end equation_has_no_real_solutions_l57_57704


namespace arithmetic_sequence_properties_l57_57526

open Real

noncomputable def arithmetic_sequence (a d : ℤ) : ℕ → ℤ
| 0       := a
| (n + 1) := arithmetic_sequence n + d

theorem arithmetic_sequence_properties (a d : ℤ)
  (h1 : 1 + a + a^5 = 1)
  (h2 : a^2 + a^4 + a = 99) :
  d = -2 ∧ a = 35 - 2 * 20 ∧ ∃ n, n = 20 := 
sorry

end arithmetic_sequence_properties_l57_57526


namespace find_a_l57_57423

variable (a : ℝ)

def point : ℝ × ℝ := (a, 2)

def line : ℝ → ℝ → ℝ := λ x y, x - y + 3

def distance_from_point_to_line (p : ℝ × ℝ) (l : ℝ → ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / Math.sqrt 2

theorem find_a (h1 : a > 0) (h2 : distance_from_point_to_line (a, 2) line = 1) :
  a = Math.sqrt 2 - 1 :=
sorry

end find_a_l57_57423


namespace transform_sin_l57_57242

open Real

theorem transform_sin :
  ∀ x : ℝ, (let f := λ x, sin x in
  let g := λ x, sin (x + π / 6) in
  let h := λ x, sin (1 / 2 * x + π / 6) in
  h x = sin (1 / 2 * x + π / 6)) :=
begin
  intros x,
  let f := λ x, sin x,
  let g := λ x, sin (x + π / 6),
  let h := λ x, sin (1 / 2 * x + π / 6),
  sorry,
end

end transform_sin_l57_57242


namespace number_of_subcommittees_l57_57482

theorem number_of_subcommittees :
  ∃ (k : ℕ), ∀ (num_people num_sub_subcommittees subcommittee_size : ℕ), 
  num_people = 360 → 
  num_sub_subcommittees = 3 → 
  subcommittee_size = 6 → 
  k = (num_people * num_sub_subcommittees) / subcommittee_size :=
sorry

end number_of_subcommittees_l57_57482


namespace michael_max_correct_answers_l57_57298

theorem michael_max_correct_answers (c w b : ℕ) 
  (h1 : c + w + b = 30) 
  (h2 : 4 * c - 3 * w = 72) : 
  c ≤ 21 := 
sorry

end michael_max_correct_answers_l57_57298


namespace number_of_tiles_l57_57308

noncomputable def tile_count (room_length : ℝ) (room_width : ℝ) (tile_length : ℝ) (tile_width : ℝ) :=
  let room_area := room_length * room_width
  let tile_area := tile_length * tile_width
  room_area / tile_area

theorem number_of_tiles :
  tile_count 10 15 (1 / 4) (5 / 12) = 1440 := by
  sorry

end number_of_tiles_l57_57308


namespace angle_AFC_right_angle_l57_57392

-- Define acute triangle ABC
variables {A B C : Type} [Nonempty A] [Nonempty B] [Nonempty C]
variables (triangle_ABC : Triangle A B C) (acute_triangle : AcuteTriangle triangle_ABC)

-- Define point D as the foot of the perpendicular from A to BC
variables (D : Point)
variables (foot_perpendicular_ADBC : FootOfPerpendicular D A (B, C))

-- Define point E on segment AD with AE/ED = CD/DB
variables (E : Point)
variables (E_on_AD : LiesOnSegment E (A, D)) (ratio_condition : (AE / ED) = (CD / DB))

-- Define point F as the foot of the perpendicular from D to BE
variables (F : Point)
variables (foot_perpendicular_DEBE : FootOfPerpendicular F D (B, E))

-- Statement to prove
theorem angle_AFC_right_angle : angle_β_FC  = π / 2 :=
by sorry

end angle_AFC_right_angle_l57_57392


namespace number_halfway_l57_57571

theorem number_halfway (a b : ℚ) (h1 : a = 1/12) (h2 : b = 1/10) : (a + b) / 2 = 11 / 120 := by
  sorry

end number_halfway_l57_57571


namespace problem1_problem2_l57_57433

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

theorem problem1 (x : ℝ) (h : ∃ λ : ℝ, a x = λ • b x) : Real.tan x = -3/2 :=
by sorry

noncomputable def f (x : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 + (b x).1 ^ 2 + (b x).2 ^ 2

theorem problem2 : (∃ x : ℝ, ∀ k : ℤ, x = π / 8 + k * π) ∧ (∀ x : ℝ, f x ≤ sqrt 2 / 2) :=
by sorry

end problem1_problem2_l57_57433


namespace min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l57_57581

-- Part (a): For n = 12:
theorem min_sticks_to_break_for_square_12 : ∀ (n : ℕ), n = 12 → 
  (∃ (sticks : Finset ℕ), sticks.card = 12 ∧ sticks.sum id = 78 ∧ (¬ (78 % 4 = 0) → 
  ∃ (b : ℕ), b = 2)) := 
by sorry

-- Part (b): For n = 15:
theorem can_form_square_without_breaking_15 : ∀ (n : ℕ), n = 15 → 
  (∃ (sticks : Finset ℕ), sticks.card = 15 ∧ sticks.sum id = 120 ∧ (120 % 4 = 0)) :=
by sorry

end min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l57_57581


namespace area_triangle_MNC_less_half_area_ACEF_l57_57897
-- Import necessary libraries for geometric definitions and reasoning

-- Define the geometry setup and required properties
theorem area_triangle_MNC_less_half_area_ACEF
  (O A B C D E F : Point)
  (circle : Circle O)
  (hA : A ∈ circle)
  (hB : B ∈ circle)
  (hC : C ∈ circle)
  (hD : D ∈ circle)
  (hE : E ∈ circle)
  (hF : F ∈ circle)
  (hBD_diameter : B ≠ D ∧ Circle.diameter circle B D)
  (hBD_perp_CF : Line B D ⊥ Line C F)
  (h_concurrent : Concurrent (Line C F) (Line B E) (Line A D))
  (M N : Point)
  (hM_alt : FootAltitude B (Line A C) M)
  (hN_alt : FootAltitude D (Line C E) N) :
  Area (Triangle M N C) < 0.5 * Area (Quadrilateral A C E F) := by
  sorry

end area_triangle_MNC_less_half_area_ACEF_l57_57897


namespace unit_vector_in_direction_a_plus_b_range_of_t_l57_57883

-- Definitions for given vectors
def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (1/2, real.sqrt(3)/2)

-- Conditions for Part 2
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)
def a_dot_b : ℝ := 1
def a_magnitude : ℝ := 2
def b_magnitude : ℝ := 1

-- Proof Statement for Part 1
theorem unit_vector_in_direction_a_plus_b :
  let sum := (a.1 + b.1, a.2 + b.2) in
  let mag := magnitude sum in
  (sum.1 / mag, sum.2 / mag) = (5*real.sqrt(7)/14, real.sqrt(21)/14) :=
sorry

-- Proof Statement for Part 2
theorem range_of_t :
  ∀ t : ℝ,
  dot_product (2 * t * a.1, 2 * t * a.2 + 7 * b.2) (a.1 + t * b.1, a.2 + t * b.2) < 0 ↔
  t ∈ set.union (set.Ioo (-7) (-real.sqrt(14)/2)) (set.Ioo (-real.sqrt(14)/2) (-1/2)) :=
sorry

end unit_vector_in_direction_a_plus_b_range_of_t_l57_57883


namespace boys_oak_eighth_grade_l57_57487

def students := 120
def boys := 70
def girls := 50
def pine := 70
def oak := 50
def seventh := 60
def eighth := 60
def pine_girls := 30
def oak_girls := girls - pine_girls -- 20
def pine_half_seventh := pine / 2 -- 35
def pine_half_eighth := pine / 2 -- 35

/-- Prove the number of boys from Oak Middle School in 8th grade is 15 -/
theorem boys_oak_eighth_grade : 
  let pine_boys := boys - pine_girls in
  let oak_boys := boys - pine_boys in
  let oak_boys_eighth := oak_boys / 2 in
  oak_boys_eighth = 15 :=
by 
  let pine_boys := boys - pine_girls
  let oak_boys := boys - pine_boys
  let oak_boys_eighth := oak_boys / 2
  show oak_boys_eighth = 15 from sorry

end boys_oak_eighth_grade_l57_57487


namespace collinear_points_l57_57391

/-- Given a triangle ABC with points C1 on side AB, A1 on side BC, and B1 on side CA,
such that (AC1/BC1) * (BA1/CA1) * (CB1/AB1) = 1, prove that points A1, B1, and C1 are collinear. --/
theorem collinear_points
  (A B C C1 A1 B1 : Point)
  (hC1 : OnLine C1 A B)
  (hA1 : OnLine A1 B C)
  (hB1 : OnLine B1 C A)
  (hRatios : (dist A C1 / dist B C1) * (dist B A1 / dist C A1) * (dist C B1 / dist A B1) = 1) :
  Collinear A1 B1 C1 := 
sorry

end collinear_points_l57_57391


namespace min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l57_57582

-- Part (a): For n = 12:
theorem min_sticks_to_break_for_square_12 : ∀ (n : ℕ), n = 12 → 
  (∃ (sticks : Finset ℕ), sticks.card = 12 ∧ sticks.sum id = 78 ∧ (¬ (78 % 4 = 0) → 
  ∃ (b : ℕ), b = 2)) := 
by sorry

-- Part (b): For n = 15:
theorem can_form_square_without_breaking_15 : ∀ (n : ℕ), n = 15 → 
  (∃ (sticks : Finset ℕ), sticks.card = 15 ∧ sticks.sum id = 120 ∧ (120 % 4 = 0)) :=
by sorry

end min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l57_57582


namespace tangent_line_at_origin_l57_57380

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^3 + a*x^2 + (a-3)*x

def is_even (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

theorem tangent_line_at_origin (h_even : is_even (λ x, (f a)' x)) : 
  (3 : ℝ) * (0 : ℝ) + (0 : ℝ) = (0 : ℝ)  :=
by
  sorry

end tangent_line_at_origin_l57_57380


namespace volume_of_rotated_solid_l57_57812

noncomputable def volume_solid_generated (x y : ℝ) : ℝ :=
if |x/3| + |y/3| = 2 then 144 * π else 0

theorem volume_of_rotated_solid {x y : ℝ} (hx : |x/3| + |y/3| = 2) :
  volume_solid_generated x y = 144 * π :=
begin
  unfold volume_solid_generated,
  simp [hx],
end

end volume_of_rotated_solid_l57_57812


namespace sum_of_vectors_from_center_is_zero_l57_57531

-- Define a function that takes in n, the number of vertices (sides) of the polygon, and returns the sum of vectors from the center to each vertex.

noncomputable def sum_vectors_to_center (n : ℕ) (vertices : Fin n → ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  ∑ i, (vertices i - center)

-- The main theorem stating that this sum is zero for a regular polygon
theorem sum_of_vectors_from_center_is_zero (n : ℕ) (regular_polygon : ∀ i : Fin n, ℝ × ℝ) (O : ℝ × ℝ)
  (h : ∀ i : Fin n, ‖regular_polygon i - O‖ = ‖regular_polygon 0 - O‖) : 
  sum_vectors_to_center n regular_polygon O = (0, 0) := by
  sorry

end sum_of_vectors_from_center_is_zero_l57_57531


namespace unique_integer_triple_l57_57268

theorem unique_integer_triple (a b c : ℤ) (h1 : a + b = c) (h2 : b * c = a) :
  (a = -4 ∧ b = 2 ∧ c = -2) :=
begin
  sorry
end

end unique_integer_triple_l57_57268


namespace no_solution_exists_l57_57566

theorem no_solution_exists :
  ¬ ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 150 ∧ n % 8 = 0 ∧ n % 10 = 6 ∧ n % 7 = 6 := 
by
  sorry

end no_solution_exists_l57_57566


namespace find_k_l57_57747

variable (a b : ℝ^3) (k : ℝ)

theorem find_k
  (ha : ‖a‖ = 3)
  (hb : ‖b‖ = 4)
  (h : (a + k • b) ⬝ (a - k • b) = 0) :
  k = 3 / 4 ∨ k = -3 / 4 :=
sorry

end find_k_l57_57747


namespace females_in_orchestra_not_in_band_l57_57193

/-- The number of females in the orchestra who are NOT in the band is 10, given the following -/
/-- The Pythagoras High School band has 120 female and 110 male members. -/
/-- The Pythagoras High School orchestra has 100 female and 130 male members. -/
/-- There are 90 females and 80 males who are members of both the band and orchestra. -/
/-- Altogether, there are 280 students who are in either band or orchestra or both. -/
theorem females_in_orchestra_not_in_band :
  let females_band := 120 in
  let males_band := 110 in
  let females_orchestra := 100 in
  let males_orchestra := 130 in
  let females_both := 90 in
  let males_both := 80 in
  let total_students := 280 in
  let total_females := females_band + females_orchestra - females_both in
  let total_males := total_students - total_females in
  100 - 90 = 10 :=
by
  -- Problem and conditions are stated. Proof is omitted.
  sorry

end females_in_orchestra_not_in_band_l57_57193


namespace sequence_periodic_2018_l57_57574

noncomputable def a : ℕ → ℚ 
| 0       := 2
| (n + 1) := 1 / (1 - a n)

theorem sequence_periodic_2018 :
  a 2018 = -1 :=
sorry

end sequence_periodic_2018_l57_57574


namespace f_neg_ln_neg_x_f_ln_x_sub_4k_l57_57039

noncomputable def f : ℝ → ℝ := sorry

theorem f_neg_ln_neg_x (x : ℝ) (h1 : f x = -f (-x))
  (h2 : ∀ x, f (2 - x) = f x) (h3 : ∀ x, x ∈ Ioo 0 1 → f x = Real.log x) :
  x ∈ Ico (-1 : ℝ) 0 → f x = -Real.log (-x) :=
sorry

theorem f_ln_x_sub_4k (x : ℝ) (k : ℤ) (h1 : f x = -f (-x))
  (h2 : ∀ x, f (2 - x) = f x) (h3 : ∀ x, x ∈ Ioo 0 1 → f x = Real.log x) :
  x ∈ Ioc (4 * k : ℝ) (4 * k + 1) → f x = Real.log (x - 4 * k) :=
sorry

end f_neg_ln_neg_x_f_ln_x_sub_4k_l57_57039


namespace square_with_12_sticks_square_with_15_sticks_l57_57586

-- Definitions for problem conditions
def sum_of_first_n_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def can_form_square (total_length : ℕ) : Prop :=
  total_length % 4 = 0

-- Given n = 12, check if breaking 2 sticks is required to form a square
theorem square_with_12_sticks : (n = 12) → ¬ can_form_square (sum_of_first_n_natural_numbers 12) → true :=
by
  intros
  sorry

-- Given n = 15, check if it is possible to form a square without breaking any sticks
theorem square_with_15_sticks : (n = 15) → can_form_square (sum_of_first_n_natural_numbers 15) → true :=
by
  intros
  sorry

end square_with_12_sticks_square_with_15_sticks_l57_57586


namespace range_of_a_l57_57778

noncomputable def f (x : ℝ) := (Real.log x) / x
noncomputable def g (x a : ℝ) := -Real.exp 1 * x^2 + a * x

theorem range_of_a (a : ℝ) : (∀ x1 : ℝ, ∃ x2 ∈ Set.Icc (1/3) 2, f x1 ≤ g x2 a) → 2 ≤ a :=
sorry

end range_of_a_l57_57778


namespace convert_cylindrical_to_rectangular_l57_57352

-- Define cylindrical coordinates
def cylindrical_point : ℝ × ℝ × ℝ := (6, Real.pi / 3, 2)

-- Convert cylindrical to rectangular given the conversion formulas
def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

-- Statement to prove
theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 6 (Real.pi / 3) 2 = (3, 3 * Real.sqrt 3, 2) :=
by
  -- Use this with sorry for proof placeholder
  sorry

end convert_cylindrical_to_rectangular_l57_57352


namespace smallest_positive_period_tan_l57_57943

noncomputable def max_value (a b x : ℝ) := b + a * Real.sin x = -1
noncomputable def min_value (a b x : ℝ) := b - a * Real.sin x = -5
noncomputable def a_negative (a : ℝ) := a < 0

theorem smallest_positive_period_tan :
  ∃ (a b : ℝ), (max_value a b 0) ∧ (min_value a b 0) ∧ (a_negative a) →
  (1 / |3 * a + b|) * Real.pi = Real.pi / 9 :=
by
  sorry

end smallest_positive_period_tan_l57_57943


namespace square_with_12_sticks_square_with_15_sticks_l57_57585

-- Definitions for problem conditions
def sum_of_first_n_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def can_form_square (total_length : ℕ) : Prop :=
  total_length % 4 = 0

-- Given n = 12, check if breaking 2 sticks is required to form a square
theorem square_with_12_sticks : (n = 12) → ¬ can_form_square (sum_of_first_n_natural_numbers 12) → true :=
by
  intros
  sorry

-- Given n = 15, check if it is possible to form a square without breaking any sticks
theorem square_with_15_sticks : (n = 15) → can_form_square (sum_of_first_n_natural_numbers 15) → true :=
by
  intros
  sorry

end square_with_12_sticks_square_with_15_sticks_l57_57585


namespace min_sticks_12_to_break_can_form_square_15_l57_57589

-- Problem definition for n = 12
def sticks_12 : List Nat := List.range' 1 12

theorem min_sticks_12_to_break : 
  ... (I realize I need to translate a step better) ..............
  sorry

-- Problem definition for n = 15
def sticks_15 : List Nat := List.range' 1 15

theorem can_form_square_15 : 
  ... (implementing a nice explanation)
  sorry

end min_sticks_12_to_break_can_form_square_15_l57_57589


namespace arithmetic_sequence_geometric_sequence_sum_l57_57393

variables {n : Nat} {a S b T : Nat → Nat}
variable (d : Nat)

def a_sequence := (2 * n - 1)
def b_sequence (a : Nat → Nat) := λ n, a n * 3^n
def sum_sequence (a : Nat → Nat) := λ n, (n * (2 * (a 1) + (n - 1) * d)) / 2

theorem arithmetic_sequence :
  a 1 + a 3 + a 5 = 15 →
  sum_sequence a 7 = 49 →
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

theorem geometric_sequence_sum :
  (∀ n, a n = 2 * n - 1) →
  ∀ n, b n = a n * 3^n →
  T n = (n - 1) * 3^(n + 1) + 3 :=
by
  sorry

end arithmetic_sequence_geometric_sequence_sum_l57_57393


namespace largest_integer_dividing_sum_of_5_consecutive_integers_l57_57942

theorem largest_integer_dividing_sum_of_5_consecutive_integers :
  ∀ (a : ℤ), ∃ (n : ℤ), n = 5 ∧ 5 ∣ ((a - 2) + (a - 1) + a + (a + 1) + (a + 2)) := by
  sorry

end largest_integer_dividing_sum_of_5_consecutive_integers_l57_57942


namespace urn_prob_correct_l57_57327

-- Definition of the scenario
def urn_prob_final_distribution : ℕ :=
  -- We assume that there are initially two red and two blue balls.
  let urn_initial := (2, 2) -- (number of red balls, number of blue balls)

  -- Operations performed 5 times drawing from urn, adding same color ball back from box
  -- We need to prove that after these operations, correct probability is calculated
  let operations := 5
  let final_urn := (6, 6) -- Final desired number of red and blue balls
  7 -- The denominator for probability fraction, numerator found to be 3
  -- Return the probability of final distribution as fraction of total possibilities
  sorry

-- The theorem statement capturing the problem
theorem urn_prob_correct :
  urn_prob_final_distribution = 7 :=
sorry

end urn_prob_correct_l57_57327


namespace length_of_de_equals_eight_l57_57997

theorem length_of_de_equals_eight
  (a b c d e : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (bc : c - b = 3 * (d - c))
  (ab : b - a = 5)
  (ac : c - a = 11)
  (ae : e - a = 21) :
  e - d = 8 := by
  sorry

end length_of_de_equals_eight_l57_57997


namespace tangent_line_at_one_range_of_a_l57_57767

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.log x
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - (1 / x)

theorem tangent_line_at_one (a : ℝ) (h : a = 1) :
  let f_val := f 1 1
  let tangent_slope := f_deriv 1 1
  tangent_slope = 1 ∧ f_val = 1 ∧ (∀ x, (y = 1 * x): Prop) :=
by
  sorry

theorem range_of_a
(h : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → |f a x| ≥ 1) :
  a ∈ Set.Ici (Real.exp(1) / 2) :=
by
  sorry

end tangent_line_at_one_range_of_a_l57_57767


namespace number_of_roots_of_equation_l57_57360

theorem number_of_roots_of_equation :
  (∃ (s : set ℝ), (∀ x ∈ s, sqrt (10 - 2 * x) = x^2 * sqrt (10 - 2 * x)) ∧ s.card = 3) :=
sorry

end number_of_roots_of_equation_l57_57360


namespace equation_of_line_l57_57575

-- Define the vector v
def v : Type := ℝ × ℝ

-- Define the projection of v onto a
def proj (a : v) (v : v) : v :=
  let num := (v.1 * a.1 + v.2 * a.2)
  let denom := (a.1 * a.1 + a.2 * a.2)
  (num / denom * a.1, num / denom * a.2)

-- Define the given vectors a and the resulting projected vector b
def a : v := (3, 4)
def b : v := (-3/2, -2)

theorem equation_of_line :
  ∃ m b, ∀ x y, proj a (x, y) = b → y = m * x + b :=
by
  exists -3/4, -25/8
  intros x y h
  sorry

end equation_of_line_l57_57575


namespace min_sticks_12_to_break_can_form_square_15_l57_57592

-- Problem definition for n = 12
def sticks_12 : List Nat := List.range' 1 12

theorem min_sticks_12_to_break : 
  ... (I realize I need to translate a step better) ..............
  sorry

-- Problem definition for n = 15
def sticks_15 : List Nat := List.range' 1 15

theorem can_form_square_15 : 
  ... (implementing a nice explanation)
  sorry

end min_sticks_12_to_break_can_form_square_15_l57_57592


namespace largest_integral_x_l57_57008

theorem largest_integral_x (x : ℤ) (h1 : 1/4 < (x:ℝ)/6) (h2 : (x:ℝ)/6 < 7/9) : x ≤ 4 :=
by
  -- This is where the proof would go
  sorry

end largest_integral_x_l57_57008


namespace count_three_digit_numbers_divisible_by_9_l57_57462

theorem count_three_digit_numbers_divisible_by_9 : 
  (finset.filter (λ n, (n % 9 = 0)) (finset.range 1000).filter (λ n, 100 ≤ n)).card = 100 :=
by
  sorry

end count_three_digit_numbers_divisible_by_9_l57_57462


namespace perpendicular_lines_a_value_l57_57063

theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ l1 l2, l1 = ax - y + 2a ∧ l2 = (2a - 1)x + ay + a ∧ ∀ θ1 θ2, θ1 ∘ l1 = 0 ∧ θ2 ∘ l2 = 0 ∧ cos θ1 - θ2 = 0) → (a = 0 ∨ a = 1) :=
sorry

end perpendicular_lines_a_value_l57_57063


namespace men_per_table_correct_l57_57316

def tables := 6
def women_per_table := 3
def total_customers := 48
def total_women := women_per_table * tables
def total_men := total_customers - total_women
def men_per_table := total_men / tables

theorem men_per_table_correct : men_per_table = 5 := by
  sorry

end men_per_table_correct_l57_57316


namespace edward_good_games_l57_57282

theorem edward_good_games (games_from_friend games_from_garage_sale games_not_working : ℕ) (h1 : games_from_friend = 41) (h2 : games_from_garage_sale = 14) (h3 : games_not_working = 31) : (games_from_friend + games_from_garage_sale - games_not_working) = 24 :=
by
  rw [h1, h2, h3]
  simp
  sorry

end edward_good_games_l57_57282


namespace power_of_thousand_l57_57269

-- Define the notion of googol
def googol := 10^100

-- Prove that 1000^100 is equal to googol^3
theorem power_of_thousand : (1000 ^ 100) = googol^3 := by
  -- proof step to be filled here
  sorry

end power_of_thousand_l57_57269


namespace integral_inequality_nonnegative_random_variable_l57_57870

noncomputable theory
open MeasureTheory

variables {Ω : Type*} [MeasureSpace Ω] (ξ : Ω → ℝ) (p : ℝ) 
  [IsMeasurableFunction ξ] [NonnegativeFunction ξ] (μ : Measure Ω)

theorem integral_inequality_nonnegative_random_variable (hp : p > 1) :
  (∫ x in 0..∞, (μ.expectation (λ ω, min (ξ ω ^ p) (x ^ p)) / x ^ p)) = (p / (p - 1)) * μ.expectation ξ :=
sorry

end integral_inequality_nonnegative_random_variable_l57_57870


namespace number_of_friends_l57_57118

-- Define the conditions
def initial_apples := 55
def apples_given_to_father := 10
def apples_per_person := 9

-- Define the formula to calculate the number of friends
def friends (initial_apples apples_given_to_father apples_per_person : ℕ) : ℕ :=
  (initial_apples - apples_given_to_father - apples_per_person) / apples_per_person

-- State the Lean theorem
theorem number_of_friends :
  friends initial_apples apples_given_to_father apples_per_person = 4 :=
by
  sorry

end number_of_friends_l57_57118


namespace probability_sequence_l57_57611

def total_cards := 52
def first_card_is_six_of_diamonds := 1 / total_cards
def remaining_cards := total_cards - 1
def second_card_is_queen_of_hearts (first_card_was_six_of_diamonds : Prop) := 1 / remaining_cards
def probability_six_of_diamonds_and_queen_of_hearts : ℝ :=
  first_card_is_six_of_diamonds * second_card_is_queen_of_hearts sorry

theorem probability_sequence : 
  probability_six_of_diamonds_and_queen_of_hearts = 1 / 2652 := sorry

end probability_sequence_l57_57611


namespace jerome_contact_list_count_l57_57121

theorem jerome_contact_list_count :
  (let classmates := 20
   let out_of_school_friends := classmates / 2
   let family := 3 -- two parents and one sister
   let total_contacts := classmates + out_of_school_friends + family
   total_contacts = 33) :=
by
  let classmates := 20
  let out_of_school_friends := classmates / 2
  let family := 3
  let total_contacts := classmates + out_of_school_friends + family
  show total_contacts = 33
  sorry

end jerome_contact_list_count_l57_57121


namespace probability_A_mc_and_B_tf_probability_at_least_one_mc_l57_57828

-- Define the total number of questions
def total_questions : ℕ := 5

-- Define the number of multiple choice questions and true or false questions
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2

-- First proof problem: Probability that A draws a multiple-choice question and B draws a true or false question
theorem probability_A_mc_and_B_tf :
  (multiple_choice_questions * true_false_questions : ℚ) / (total_questions * (total_questions - 1)) = 3 / 10 :=
by
  sorry

-- Second proof problem: Probability that at least one of A and B draws a multiple-choice question
theorem probability_at_least_one_mc :
  1 - (true_false_questions * (true_false_questions - 1) : ℚ) / (total_questions * (total_questions - 1)) = 9 / 10 :=
by
  sorry

end probability_A_mc_and_B_tf_probability_at_least_one_mc_l57_57828


namespace min_value_of_x_plus_2y_l57_57762

theorem min_value_of_x_plus_2y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 18 :=
sorry

end min_value_of_x_plus_2y_l57_57762


namespace range_of_k_l57_57811

-- Define the condition and the main theorem statement.
theorem range_of_k (k : ℝ) : 
  (∀ x ∈ set.Icc 0 (real.pi / 2), cos (2 * x) - 2 * real.sqrt 3 * sin x * cos x ≤ k + 1) ↔ k ≥ 0 := 
sorry

end range_of_k_l57_57811


namespace correct_statements_l57_57680

theorem correct_statements :
  (∀ x, Rational x ↔ x in ℝ) = false ∧
  (∃ x, x^2 = 25 ∧ x = -5) ∧
  (¬(∃ x, x^2 = 25 ∧ x = -5)) = false ∧
  (∃ x, 3 * x - 5 ≤ -2 ∧ x = 1) ∧
  (∀ x y, Irrational x → Irrational y → Irrational (x + y) = false) ∧
  (∀ x, Irrational x → ¬ is_terminating (decimal_rep x)) :=
begin
  sorry
end

end correct_statements_l57_57680


namespace case_a_sticks_case_b_square_l57_57598
open Nat 

premise n12 : Nat := 12
premise sticks12_sum : Nat := (n12 * (n12 + 1)) / 2  -- Sum of first 12 natural numbers
premise length_divisibility_4 : ¬ (sticks12_sum % 4 = 0)  -- Check if sum is divisible by 4

-- Need to break at least 2 sticks to form a square
theorem case_a_sticks (h : sticks12_sum = 78) (h2 : length_divisibility_4 = true) : 
  ∃ (k : Nat), k >= 2 := sorry

premise n15 : Nat := 15
premise sticks15_sum : Nat := (n15 * (n15 + 1)) / 2  -- Sum of first 15 natural numbers
premise length_divisibility4_b : sticks15_sum % 4 = 0  -- Check if sum is divisible by 4

-- Possible to form a square without breaking any sticks
theorem case_b_square (h : sticks15_sum = 120) (h2 : length_divisibility4_b = true) : 
  ∃ (k : Nat), k = 0 := sorry

end case_a_sticks_case_b_square_l57_57598


namespace total_weight_of_dumbbells_l57_57549

theorem total_weight_of_dumbbells : 
  let initial_dumbbells := 4
  let additional_dumbbells := 2
  let weight_per_dumbbell := 20 in
  (initial_dumbbells + additional_dumbbells) * weight_per_dumbbell = 120 := 
by
  -- conditions and definitions
  let initial_dumbbells := 4
  let additional_dumbbells := 2
  let weight_per_dumbbell := 20
  -- calculation
  calc
  (initial_dumbbells + additional_dumbbells) * weight_per_dumbbell 
  = (4 + 2) * 20 : by rw [initial_dumbbells, additional_dumbbells, weight_per_dumbbell]
  ... = 6 * 20 : by norm_num
  ... = 120 : by norm_num

end total_weight_of_dumbbells_l57_57549


namespace slower_rider_speed_l57_57245

theorem slower_rider_speed (d : ℝ) (t1 t2 : ℝ) (h : ℝ) :
  (d = 20) → (t1 = 4) → (t2 = 10) →
  (2 * h) = (2*h + h) * (t1/t2) →
  h = 5/3 :=
by
  intros h0 h1 h2 h3,
  sorry

end slower_rider_speed_l57_57245


namespace min_val_neg_infty_to_0_l57_57399

variables {X : Type*} {a b : ℝ} {f g : ℝ → ℝ} {h : ℝ → ℝ} 

-- Given conditions as definitions
def odd_fn (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def h_def : ℝ → ℝ := λ x, a * (f x)^3 - b * g x - 2
def max_val (h : ℝ → ℝ) (x : ℝ) := ∀ x > 0, h x ≤ 5

-- Fact to prove
theorem min_val_neg_infty_to_0 (hf : odd_fn f) (hg : odd_fn g) (h_max : max_val h_def) :
  ∀ x < 0, h_def x ≥ -9 := sorry

end min_val_neg_infty_to_0_l57_57399


namespace ratio_of_side_lengths_l57_57659

theorem ratio_of_side_lengths (w1 w2 : ℝ) (s1 s2 : ℝ)
  (h1 : w1 = 8) (h2 : w2 = 64)
  (v1 : w1 = s1 ^ 3)
  (v2 : w2 = s2 ^ 3) : 
  s2 / s1 = 2 := by
  sorry

end ratio_of_side_lengths_l57_57659


namespace ratio_G_to_C_is_1_1_l57_57068

variable (R C G : ℕ)

-- Given conditions
def Rover_has_46_spots : Prop := R = 46
def Cisco_has_half_R_minus_5 : Prop := C = R / 2 - 5
def Granger_Cisco_combined_108 : Prop := G + C = 108
def Granger_Cisco_equal : Prop := G = C

-- Theorem stating the final answer to the problem
theorem ratio_G_to_C_is_1_1 (h1 : Rover_has_46_spots R) 
                            (h2 : Cisco_has_half_R_minus_5 C R) 
                            (h3 : Granger_Cisco_combined_108 G C) 
                            (h4 : Granger_Cisco_equal G C) : 
                            G / C = 1 := by
  sorry

end ratio_G_to_C_is_1_1_l57_57068


namespace spending_spring_months_l57_57212

variable (s_feb s_may : ℝ)

theorem spending_spring_months (h1 : s_feb = 2.8) (h2 : s_may = 5.6) : s_may - s_feb = 2.8 := 
by
  sorry

end spending_spring_months_l57_57212


namespace jerome_contacts_total_l57_57122

def jerome_classmates : Nat := 20
def jerome_out_of_school_friends : Nat := jerome_classmates / 2
def jerome_family_members : Nat := 2 + 1
def jerome_total_contacts : Nat := jerome_classmates + jerome_out_of_school_friends + jerome_family_members

theorem jerome_contacts_total : jerome_total_contacts = 33 := by
  sorry

end jerome_contacts_total_l57_57122


namespace line_dividing_circle_maximizes_area_l57_57396

noncomputable theory

-- Define the circular region and the point P
def circular_region (x y : ℝ) : Prop := x^2 + y^2 ≤ 4
def point_P : ℝ × ℝ := (1, 1)

-- Define the desired result
def optimal_line_eq (x y : ℝ) : Prop := x + y - 2 = 0

-- Formal statement of the problem
theorem line_dividing_circle_maximizes_area : 
  ∃ (x y : ℝ), (circular_region x y ∧ (x, y) = point_P) → optimal_line_eq x y :=
by sorry

end line_dividing_circle_maximizes_area_l57_57396


namespace leading_coefficient_of_polynomial_l57_57367

theorem leading_coefficient_of_polynomial :
  let p := -3 * (x^4 - x^3 + x) + 7 * (x^4 + 2) - 4 * (2 * x^4 + 2 * x^2 + 1)
  in polynomial.leading_coeff p = -4 :=
by
  sorry

end leading_coefficient_of_polynomial_l57_57367


namespace five_Mondays_in_May_l57_57186

-- Define April and May parameters
def days_in_April : ℕ := 30
def days_in_May : ℕ := 31

-- Define a condition for April having five Sundays
def has_five_Sundays_in_April (first_Sunday_April : ℕ) : Prop :=
  (first_Sunday_April = 1 ∨ first_Sunday_April = 2) ∧
  days_in_April = 30

-- The main theorem to prove
theorem five_Mondays_in_May (first_Sunday_April : ℕ) :
  has_five_Sundays_in_April first_Sunday_April →
  ∃ first_May_day : ℕ, 
    (first_Sunday_April = 2 ∧ first_May_day = 1) →
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ days_in_May → 
        (k - first_May_day) % 7 = 0 → 
        k = 1 ∨ k = 8 ∨ k = 15 ∨ k = 22 ∨ k = 29) :=
begin
  sorry,
end

end five_Mondays_in_May_l57_57186


namespace inverse_proportionality_decrease_l57_57185

variable (x y k p : ℝ)
variable (hp : 0 < p) (hx : 0 < x) (hy : 0 < y)
variable (hxy : x * y = k)

theorem inverse_proportionality_decrease {x y p : ℝ}
  (hx : 0 < x) (hy : 0 < y) (hp : 0 < p) (hxy : x * y = (k : ℝ)) :
  let x' := x * (1 + p / 100) in
  let y' := y * 100 / (100 + p) in
  (y - y') / y * 100 = 100 * p / (100 + p) :=
by
  sorry

end inverse_proportionality_decrease_l57_57185


namespace Alex_donut_holes_covered_l57_57685

noncomputable def Alex_radius : ℝ := 5
noncomputable def Bella_radius : ℝ := 7
noncomputable def Carlos_radius : ℝ := 9

def Alex_surface_area : ℝ := 4 * Real.pi * Alex_radius^2
def Bella_surface_area: ℝ := 4 * Real.pi * Bella_radius^2
def Carlos_surface_area : ℝ := 4 * Real.pi * Carlos_radius^2

theorem Alex_donut_holes_covered :
    ∀ (coating_rate : ℝ), 
    ∀ (start_time: ℝ), Alex_surface_area ≠ 0  → Bella_surface_area ≠ 0 → Carlos_surface_area ≠ 0 →
    ∀ t, t = 63504 * Real.pi →  
    t / Alex_surface_area = 635 :=
by
    intros; sorry

end Alex_donut_holes_covered_l57_57685


namespace count_three_digit_numbers_divisible_by_9_l57_57458

theorem count_three_digit_numbers_divisible_by_9 : 
  (finset.filter (λ n, (n % 9 = 0)) (finset.range 1000).filter (λ n, 100 ≤ n)).card = 100 :=
by
  sorry

end count_three_digit_numbers_divisible_by_9_l57_57458


namespace no_nonzero_real_solution_l57_57074

theorem no_nonzero_real_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬ (1 / a + 1 / b = 2 / (a + b)) :=
begin
  sorry,
end

end no_nonzero_real_solution_l57_57074


namespace area_ofTriangle_PRD_l57_57702

open Real

def P := (0 : ℝ, 12 : ℝ)
def R := (4 : ℝ, 12 : ℝ)
def D := (12 : ℝ, 0 : ℝ)
def C (q : ℝ) := (12 : ℝ, q)

noncomputable def area_PRD (q : ℝ) : ℝ :=
  1/2 * abs ((R.1 - D.1) * (P.2 - C q.2))

theorem area_ofTriangle_PRD (q : ℝ): area_PRD q = 48 - 4*q :=
by
  -- Proof omitted
  sorry

end area_ofTriangle_PRD_l57_57702


namespace dealer_gross_profit_l57_57296

theorem dealer_gross_profit
  (purchase_price : ℝ)
  (markup_rate : ℝ)
  (discount_rate : ℝ)
  (initial_selling_price : ℝ)
  (final_selling_price : ℝ)
  (gross_profit : ℝ)
  (h0 : purchase_price = 150)
  (h1 : markup_rate = 0.5)
  (h2 : discount_rate = 0.2)
  (h3 : initial_selling_price = purchase_price + markup_rate * initial_selling_price)
  (h4 : final_selling_price = initial_selling_price - discount_rate * initial_selling_price)
  (h5 : gross_profit = final_selling_price - purchase_price) :
  gross_profit = 90 :=
sorry

end dealer_gross_profit_l57_57296


namespace median_of_set_is_82_l57_57215

noncomputable def numbers : List ℕ := [75, 77, 79, 81, 83, 85, 87]
noncomputable def x : ℕ := 89 -- Derived from 656 - 567
noncomputable def fullSet : List ℕ := numbers ++ [x]

theorem median_of_set_is_82 
  (h1 : (∑ i in fullSet, i) / fullSet.length = 82)
  (h2 : fullSet = [75, 77, 79, 81, 83, 85, 87, 89]) :
  (List.median fullSet) = 82 := sorry

end median_of_set_is_82_l57_57215


namespace isosceles_triangle_angle_l57_57322

theorem isosceles_triangle_angle (α x : ℝ) 
(acute_isosceles : ∀ {A B C : ℝ}, ∠ABC = α ∧ ∠ACB = α)
(inscribed_circle : ∀ {A B C : ℝ}, circle_inscribed A B C)
(tangents_meet_D : ∀ {B C D : ℝ}, tangents_BC_meet_D B C D)
(angle_relation: ∠ABC = (3 / 2) * ∠D)
(angler_bac : ∠BAC = x) :
x = π :=
by
  sorry

end isosceles_triangle_angle_l57_57322


namespace resulting_solution_percentage_l57_57303

theorem resulting_solution_percentage (w_original: ℝ) (w_replaced: ℝ) (c_original: ℝ) (c_new: ℝ) :
  c_original = 0.9 → w_replaced = 0.7142857142857143 → c_new = 0.2 →
  (0.2571428571428571 + 0.14285714285714285) / (0.2857142857142857 + 0.7142857142857143) * 100 = 40 := 
by
  intros h1 h2 h3
  sorry

end resulting_solution_percentage_l57_57303


namespace election_total_votes_l57_57960

theorem election_total_votes
  (V : ℕ)
  (winner_votes : ℕ)
  (runner_up_votes : ℕ)
  (remaining_votes : ℕ)
  (h1 : winner_votes = 0.45 * V)
  (h2 : runner_up_votes = 0.28 * V)
  (h3 : remaining_votes = 0.27 * V)
  (h4 : winner_votes - runner_up_votes = 550) :
  V ≈ 3235 :=
by 
  sorry

end election_total_votes_l57_57960


namespace solve_for_a_and_monotonicity_l57_57882

noncomputable def f (a x : ℝ) := (x + a) * Real.exp x

theorem solve_for_a_and_monotonicity : 
  ∃ a : ℝ, 
    (∀ x, f a x = (x + a) * Real.exp x) ∧ 
    (f a 1 = (1 + a) * Real.exp 1) ∧
    let f' x := (1 + x + a) * Real.exp x in
      (f' 1 = Real.exp 1) ∧ 
      (a = -1) ∧
      (∀ x, x > 0 → (f' x > 0)) ∧ 
      (∀ x, x < 0 → (f' x < 0))
:= 
begin
  sorry
end

end solve_for_a_and_monotonicity_l57_57882


namespace option_A_is_correct_l57_57264

theorem option_A_is_correct (a b : ℝ) (h : a ≠ 0) : (a^2 / (a * b)) = (a / b) :=
by
  -- Proof will be filled in here
  sorry

end option_A_is_correct_l57_57264


namespace socks_same_color_combinations_l57_57825

open Nat

theorem socks_same_color_combinations :
  let white_socks := 5
  let brown_socks := 6
  let blue_socks := 3
  let red_socks := 2
  (choose white_socks 2) + (choose brown_socks 2) + (choose blue_socks 2) + (choose red_socks 2) = 29 :=
by
  sorry

end socks_same_color_combinations_l57_57825


namespace petya_password_l57_57901

def ways_to_create_password : Nat :=
  let total_digits := 9  -- Digits 0-6, 8-9
  let total_passwords := total_digits ^ 4  -- Total possible 4-digit passwords
  let distinct_passwords := (Nat.factorial 9) / (Nat.factorial (9 - 4))  -- \binom{9}{4} * 4!
  total_passwords - distinct_passwords

theorem petya_password (ways_to_create_password = 3537) : ways_to_create_password = 3537 := 
sorry

end petya_password_l57_57901


namespace infinite_subsequence_with_same_gcd_l57_57907

open Nat

theorem infinite_subsequence_with_same_gcd (seq : ℕ → ℕ) (h1 : ∀ n, 0 < seq n) (h2 : ∀ n m, seq n = seq m → seq n = seq m) :
  ∃ (subseq : ℕ → ℕ), (∀ n m, gcd (subseq n) (subseq m) = gcd (subseq 0) (subseq 0)) :=
sorry

end infinite_subsequence_with_same_gcd_l57_57907


namespace circle_tangent_to_directrix_l57_57023

-- Definitions and conditions
def parabola (p : ℝ) : set (ℝ × ℝ) := {xy | xy.2^2 = 2 * p * xy.1}

def directrix (p : ℝ) : set (ℝ × ℝ) := {xy | xy.1 = -p / 2}

noncomputable def distance (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- The proof statement
theorem circle_tangent_to_directrix
  {p : ℝ} (h : 0 < p)
  {P Q F : ℝ × ℝ}
  (hP : P ∈ parabola p) (hQ : Q ∈ parabola p) (hF : F = (p/2, 0))
  (hPQ_focus : distance F P + distance F Q = distance P Q) :
  let M := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) in
  distance M (p/2, 0) = distance M (-(p / 2), 0) :=
sorry

end circle_tangent_to_directrix_l57_57023


namespace toms_profit_l57_57241

noncomputable def cost_of_flour : Int :=
  let flour_needed := 500
  let bag_size := 50
  let bag_cost := 20
  (flour_needed / bag_size) * bag_cost

noncomputable def cost_of_salt : Int :=
  let salt_needed := 10
  let salt_cost_per_pound := (2 / 10)  -- Represent $0.2 as a fraction to maintain precision with integers in Lean
  salt_needed * salt_cost_per_pound

noncomputable def total_expenses : Int :=
  let flour_cost := cost_of_flour
  let salt_cost := cost_of_salt
  let promotion_cost := 1000
  flour_cost + salt_cost + promotion_cost

noncomputable def revenue_from_tickets : Int :=
  let ticket_price := 20
  let tickets_sold := 500
  tickets_sold * ticket_price

noncomputable def profit : Int :=
  revenue_from_tickets - total_expenses

theorem toms_profit : profit = 8798 :=
  by
    sorry

end toms_profit_l57_57241


namespace concyclic_points_l57_57640

theorem concyclic_points
  (ABC : Type)
  [triangle ABC]
  (I_a : point)
  (X : point)
  (A' : point)
  (Y Z T : point)
  (r : ℝ)
  (h1 : excenter I_a ABC)
  (h2 : excircle_touches I_a BC X)
  (h3 : diametrically_opposite A A' (circumcircle ABC))
  (h4 : on_segment Y I_a X)
  (h5 : on_segment Z B A')
  (h6 : on_segment T C A')
  (h7 : dist I_a Y = r)
  (h8 : dist B Z = r)
  (h9 : dist C T = r)
  : concyclic X Y Z T := sorry

end concyclic_points_l57_57640


namespace a_n_general_term_b_n_general_term_sum_c_2016_l57_57407

noncomputable def a : ℕ → ℕ
| 1     := 2
| 5     := 32
| (n+1) := 2 * a n

def b (n : ℕ) : ℕ := n

def S (n : ℕ) : ℕ := (n - 1) * 2^(n + 1) + 2

theorem a_n_general_term (n : ℕ) : a n = 2^n := by
  induction n with
  | zero => rw a; sorry -- need to complete the proof
  | succ n ih => rw [a, ih]; sorry -- need to complete the proof

theorem b_n_general_term (n : ℕ) : b n = n := by 
  rw b; -- need to complete the proof
  sorry

theorem sum_c_2016 : 
  let c : ℕ → ℤ := sorry, -- definition for the sequence c can be very complex
  ∑ k in finset.range 2016, c k = 2^64 + 1951 := by 
    sorry -- length proof needs to be provided

end a_n_general_term_b_n_general_term_sum_c_2016_l57_57407


namespace max_product_of_sine_l57_57469

theorem max_product_of_sine (x : Fin 2002 → ℝ) (h : (∏ i, Real.tan (x i)) = 1) : 
  ∃ s : ℝ, 
    (s = ∏ i in (Finset.range 2012), Real.sin (x i)) 
    ∧ s ≤ 1 / 2 ^ 1006 := by
  sorry

end max_product_of_sine_l57_57469


namespace concyclic_M1_N_P1_Q_l57_57134

axiom circle (Γ : Type) : Type 
axiom center (Γ : circle) : Type
axiom point (center : Type) : Type
axiom radius (center : Type) : Type
axiom Parallel (r1 r2 : radius) : Prop
axiom direction  (r1 r2 : radius) : Prop
axiom intersects (l : Type) (Γ : circle) : point (center Γ)

variables {Γ1 Γ2 : circle}
variables {O1 : center Γ1}
variables {O2 : center Γ2}
variables {M1 M2 : radius O1}
variables {P1 P2 : radius O2}
variables {N Q : point O2}

axiom O1M1_parallel_O2M2 : Parallel M1 M2
axiom O1M1_same_direction_O2M2 : direction M1 M2
axiom O1P1_parallel_O2P2 : Parallel P1 P2
axiom O1P1_same_direction_O2P2 : direction P1 P2
axiom M1M2_intersects_Γ2_at_N : intersects M1M2 Γ2 = N
axiom P1P2_intersects_Γ2_at_Q : intersects P1P2 Γ2 = Q

theorem concyclic_M1_N_P1_Q : -- Provide the type of theorem
  concyclic_points Γ2 O2 N Q P1 M1 :=
sorry

end concyclic_M1_N_P1_Q_l57_57134


namespace close_point_distance_is_correct_l57_57612

noncomputable def distanceClosestPoints : ℝ :=
  let center1 := (3, 5)
  let center2 := (20, 15)
  let radius1 := 5
  let radius2 := 15
  let distance_between_centers := Real.sqrt ((20 - 3)^2 + (15 - 5)^2)
  distance_between_centers - (radius1 + radius2)

theorem close_point_distance_is_correct :
  let center1 := (3, 5)
  let center2 := (20, 15)
  let radius1 := 5
  let radius2 := 15
  Real.sqrt ((20 - 3)^2 + (15 - 5)^2) - (radius1 + radius2) = Real.sqrt 389 - 20 :=
by {
  dunfold distanceClosestPoints,
  sorry
}

end close_point_distance_is_correct_l57_57612


namespace parity_of_F_is_even_l57_57939

noncomputable def F (f : ℝ → ℝ) := f

theorem parity_of_F_is_even (f g : ℝ → ℝ) 
  (h1 : ∀ x, f(x) + f(-x) = 0) 
  (h2 : ∀ x, g(x) * g(-x) = 1) 
  (h3 : ∀ x, x ≠ 0 → g(x) ≠ 1) : 
  ∀ x, F(f)(-x) = F(f)(x) :=
by
  sorry

end parity_of_F_is_even_l57_57939


namespace problem_statement_l57_57847

-- Define the sequence recursively.
def a : ℕ → ℚ
| 0       := 0   -- Define a_0 as 0 by convention.
| 1       := 1   -- Given condition a_1 = 1.
| (n + 2) := let an1 := a (n + 1); an := a n
             in (an1 + (-1 : ℤ) ^ (n + 2)).toRat / an

-- The main theorem to prove: the value of a_3 / a_4.
theorem problem_statement : a 3 / a 4 = 1 / 6 := by
  sorry

end problem_statement_l57_57847


namespace initial_profit_percentage_l57_57649

theorem initial_profit_percentage (CP : ℕ) (extra : ℕ) (profit_after_extra : ℕ) (initial_profit_percent : ℕ) :
  CP = 300 → extra = 18 → profit_after_extra = 18 →
  (∀ P : ℕ, SP = CP + (P * CP / 100) → (SP + extra = CP + (profit_after_extra * CP / 100) → P = 12)) :=
by
  assume CP_eq : CP = 300,
  assume extra_eq : extra = 18,
  assume profit_after_extra_eq : profit_after_extra = 18,
  sorry

end initial_profit_percentage_l57_57649


namespace triangle_third_side_l57_57250

theorem triangle_third_side (a b : ℝ) (h₁ : a = 7) (h₂ : b = 11) : ∃ k : ℕ, 4 < k ∧ k < 18 ∧ k = 17 := 
by {
  let s := a + b,
  let d := b - a,
  have h₃ : s > 17 := by linarith,
  have h₄ : 4 < d := by linarith,
  have h₅ : d < 18 := by linarith,
  have h₆ : 17 < 18 := by linarith,
  use 17,
  linarith,
  sorry
}

end triangle_third_side_l57_57250


namespace prime_dates_in_2015_l57_57694

def is_prime (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def prime_months := { 2, 3, 5, 7, 11 }

noncomputable def prime_days := { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31 }

noncomputable def days_in_month (month : ℕ) : ℕ :=
  if month = 2 then 28 else
  if month = 4 ∨ month = 6 ∨ month = 9 ∨ month = 11 then 30 else 31

theorem prime_dates_in_2015 : 
  let prime_dates_count : ℕ := 
    ∑ m in prime_months, (∑ d in prime_days, if d ≤ days_in_month m then 1 else 0)
  in prime_dates_count = 52 := 
by
  sorry

end prime_dates_in_2015_l57_57694


namespace Darius_scored_10_points_l57_57356

theorem Darius_scored_10_points
  (D Marius Matt : ℕ)
  (h1 : Marius = D + 3)
  (h2 : Matt = D + 5)
  (h3 : D + Marius + Matt = 38) : 
  D = 10 :=
by
  sorry

end Darius_scored_10_points_l57_57356


namespace imaginary_part_of_z_eq_neg_2_l57_57385

theorem imaginary_part_of_z_eq_neg_2 (z : ℂ) 
    (h : z = (2 + complex.I) / complex.I) 
    : z.im = -2 := 
by 
  sorry

end imaginary_part_of_z_eq_neg_2_l57_57385


namespace apples_left_to_eat_raw_l57_57887

variable (n : ℕ) (picked : n = 85) (wormy : n / 5) (bruised : wormy + 9)

theorem apples_left_to_eat_raw (h_picked : n = 85) (h_wormy : wormy = n / 5) (h_bruised : bruised = wormy + 9) : n - (wormy + bruised) = 42 := 
sorry

end apples_left_to_eat_raw_l57_57887


namespace simplify_expression_evaluate_at_1_l57_57916

theorem simplify_expression (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) (h3 : x ≠ -1) :
  ( ( (x^2 - 4) / (x^2 - x - 6) + (x + 2) / (x - 3) ) / ( (x + 1) / (x - 3) ) ) = (2 * x) / (x + 1) :=
sorry

theorem evaluate_at_1 (h1 : 1 ≠ 3) (h2 : 1 ≠ -2) (h3 : 1 ≠ -1) : 
  ( ( (1^2 - 4) / (1^2 - 1 - 6) + (1 + 2) / (1 - 3) ) / ( (1 + 1) / (1 - 3) ) ) = 1 :=
by
  have h_simplified : ( ( (1^2 - 4) / (1^2 - 1 - 6) + (1 + 2) / (1 - 3) ) / ( (1 + 1) / (1 - 3) ) ) = (2 * 1) / (1 + 1) :=
    by rw simplify_expression; assumption
  rw h_simplified
  norm_num

end simplify_expression_evaluate_at_1_l57_57916


namespace cannot_be_written_as_square_l57_57168

theorem cannot_be_written_as_square (A B : ℤ) : 
  99999 + 111111 * Real.sqrt 3 ≠ (A + B * Real.sqrt 3) ^ 2 :=
by
  -- Here we would provide the actual mathematical proof
  sorry

end cannot_be_written_as_square_l57_57168


namespace point_on_graph_iff_sqrt_10_graph_parallel_to_minus_x_iff_four_y_decreases_as_x_increases_iff_greater_than_three_l57_57421

def linear_function (k : ℝ) (x : ℝ) : ℝ :=
  (3 - k) * x - 2 * k^2 + 18

theorem point_on_graph_iff_sqrt_10 (k : ℝ) :
  linear_function k 0 = -2 ↔ k = real.sqrt 10 ∨ k = -real.sqrt 10 := sorry

theorem graph_parallel_to_minus_x_iff_four (k : ℝ) :
  (3 - k) = -1 ↔ k = 4 := sorry

theorem y_decreases_as_x_increases_iff_greater_than_three (k : ℝ) :
  (3 - k < 0) ↔ k > 3 := sorry

end point_on_graph_iff_sqrt_10_graph_parallel_to_minus_x_iff_four_y_decreases_as_x_increases_iff_greater_than_three_l57_57421


namespace no_such_coins_l57_57163

theorem no_such_coins (p1 p2 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1)
  (cond1 : (1 - p1) * (1 - p2) = p1 * p2)
  (cond2 : p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :
  false :=
  sorry

end no_such_coins_l57_57163


namespace like_terms_mn_eq_neg1_l57_57797

variable (m n : ℤ)

theorem like_terms_mn_eq_neg1
  (hx : m + 3 = 4)
  (hy : n + 3 = 1) :
  m + n = -1 :=
sorry

end like_terms_mn_eq_neg1_l57_57797


namespace OXYZ_parallelogram_num_sides_of_polygon_perimeter_equivalence_l57_57064

variables {A B C D E F O X Y Z : Type}

-- Condition: defining the triangles and arbitrary points
def is_point_in_triangle (P : Type) (triangle : Type) : Prop := sorry
def is_parallelogram (O X Y Z : Type) : Prop := sorry

-- Condition: any points X and Y in the specified triangles
variable (O : Type)
variable (triangle_ABC triangle_DEF : Type)
variable (X Y : Type)
variable (H_X_in_ABC : is_point_in_triangle X triangle_ABC)
variable (H_Y_in_DEF : is_point_in_triangle Y triangle_DEF)

-- Step 1: Prove that quadrilateral OXYZ is always a parallelogram
theorem OXYZ_parallelogram : ∀ X Y,
  is_point_in_triangle X triangle_ABC →
  is_point_in_triangle Y triangle_DEF →
  ∃ Z, is_parallelogram O X Y Z := sorry

-- Step 2: Determine the number of sides of the polygon
def num_sides_polygon (n1 n2 : ℕ) := max n1 n2 ≤ max n1 n2 ∧ max n1 n2 ≤ n1 + n2

theorem num_sides_of_polygon (n1 n2 : ℕ) :
  n1 > 0 → n2 > 0 → num_sides_polygon n1 n2 := sorry

-- Step 3: Prove the perimeter equivalence
def perimeter_of_triangle (triangle : Type) : ℝ := sorry
def perimeter_of_polygon (points : list Type) : ℝ := sorry

theorem perimeter_equivalence :
  ∀ (X Y : Type), is_point_in_triangle X triangle_ABC → is_point_in_triangle Y triangle_DEF →
  let Z := some (OXYZ_parallelogram O X Y H_X_in_ABC H_Y_in_DEF) in
  perimeter_of_polygon [O, X, Y, Z] = perimeter_of_triangle triangle_ABC + perimeter_of_triangle triangle_DEF := sorry

end OXYZ_parallelogram_num_sides_of_polygon_perimeter_equivalence_l57_57064


namespace max_third_side_of_triangle_l57_57249

theorem max_third_side_of_triangle (a b : ℕ) (h₁ : a = 7) (h₂ : b = 11) : 
  ∃ c : ℕ, c < a + b ∧ c = 17 :=
by 
  sorry

end max_third_side_of_triangle_l57_57249


namespace max_days_to_be_cost_effective_is_8_l57_57277

-- Definitions
def cost_of_hiring (₥ : ℕ) := 50000
def cost_of_materials (₥ : ℕ) := 20000
def husbands_daily_wage (₥ : ℕ) := 2000
def wifes_daily_wage (₥ : ℕ) := 1500

-- Total daily wage
def total_daily_wage := husbands_daily_wage 0 + wifes_daily_wage 0

-- Cost difference
def cost_difference := cost_of_hiring 0 - cost_of_materials 0

-- Maximum number of days
def max_days_to_be_cost_effective := cost_difference / total_daily_wage

-- Prove that the maximum number of days is 8
theorem max_days_to_be_cost_effective_is_8 : max_days_to_be_cost_effective = 8 := by
  sorry

end max_days_to_be_cost_effective_is_8_l57_57277


namespace square_perimeter_lemma_l57_57220

theorem square_perimeter_lemma
  (t1 t2 t3 t4 k1 k2 k3 k4 : ℝ)
  (h1 : t1 + t2 + t3 = t4)
  (h2 : k1 = 4 * real.sqrt t1)
  (h3 : k2 = 4 * real.sqrt t2)
  (h4 : k3 = 4 * real.sqrt t3)
  (h5 : k4 = 4 * real.sqrt t4) :
  k1 + k2 + k3 ≤ k4 * real.sqrt 3 :=
by { sorry }

end square_perimeter_lemma_l57_57220


namespace count_positive_3_digit_numbers_divisible_by_9_l57_57448

-- Conditions
def is_divisible_by_9 (n : ℕ) : Prop := 9 ∣ n

def is_positive_3_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Theorem to be proved
theorem count_positive_3_digit_numbers_divisible_by_9 : 
  {n : ℕ | is_positive_3_digit_number n ∧ is_divisible_by_9 n}.card = 100 :=
sorry

end count_positive_3_digit_numbers_divisible_by_9_l57_57448


namespace coefficient_x2_in_expansion_l57_57709

theorem coefficient_x2_in_expansion :
  ∀ x : ℝ, x ≠ 0 →
  (let poly := (x + 2 + 1 / x) ^ 5 in
  (∃ c : ℝ, c * x^2 ∈ polynomial.coeff poly) ∧ c = 120) :=
begin
  intros,
  sorry 
end

end coefficient_x2_in_expansion_l57_57709


namespace sin_sub_alpha_l57_57040

theorem sin_sub_alpha (α : ℝ) (h1 : (π / 2) < α ∧ α < π) (h2 : sin (π / 4 + α) = 3 / 4) : 
  sin (π / 4 - α) = - (sqrt 7) / 4 :=
sorry

end sin_sub_alpha_l57_57040


namespace perpendicular_lines_l57_57427

theorem perpendicular_lines (m : ℝ) :
  let l1 := (m + 2) * x - y + 5 = 0
  let l2 := (m + 3) * x + (m + 18) * y + 2 = 0
  l1.perpendicular l2 → m = -6 ∨ m = 2 := 
by
  sorry

end perpendicular_lines_l57_57427


namespace slips_with_number_3_l57_57919

def total_slips : ℕ := 20
def expected_value : ℝ := 6

-- Using ℕ for discrete counts and ℝ for the expected value and probabilities
variables (x y z : ℕ)

-- Conditions
def total_slips_condition := x + y + z = total_slips
def expected_value_condition := (3*x + 9*y + 15*z : ℝ) / total_slips = expected_value

theorem slips_with_number_3 : total_slips_condition → expected_value_condition → x = 15 :=
by
  intros h_total h_expected
  -- Additional steps to reach conclusion will be here
  sorry

end slips_with_number_3_l57_57919


namespace case_a_sticks_case_b_square_l57_57600
open Nat 

premise n12 : Nat := 12
premise sticks12_sum : Nat := (n12 * (n12 + 1)) / 2  -- Sum of first 12 natural numbers
premise length_divisibility_4 : ¬ (sticks12_sum % 4 = 0)  -- Check if sum is divisible by 4

-- Need to break at least 2 sticks to form a square
theorem case_a_sticks (h : sticks12_sum = 78) (h2 : length_divisibility_4 = true) : 
  ∃ (k : Nat), k >= 2 := sorry

premise n15 : Nat := 15
premise sticks15_sum : Nat := (n15 * (n15 + 1)) / 2  -- Sum of first 15 natural numbers
premise length_divisibility4_b : sticks15_sum % 4 = 0  -- Check if sum is divisible by 4

-- Possible to form a square without breaking any sticks
theorem case_b_square (h : sticks15_sum = 120) (h2 : length_divisibility4_b = true) : 
  ∃ (k : Nat), k = 0 := sorry

end case_a_sticks_case_b_square_l57_57600


namespace rearrangement_proof_l57_57333

-- Define vertices as an inductive type
inductive Vertex
| A | B | C | D | E | F

-- Define adjacency relation
def adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.B => True
| Vertex.B, Vertex.C => True
| Vertex.C, Vertex.D => True
| Vertex.D, Vertex.E => True
| Vertex.E, Vertex.F => True
| Vertex.F, Vertex.A => True
| Vertex.B, Vertex.A => True
| Vertex.C, Vertex.B => True
| Vertex.D, Vertex.C => True
| Vertex.E, Vertex.D => True
| Vertex.F, Vertex.E => True
| Vertex.A, Vertex.F => True
| _, _ => False

-- Initial positions of characters at vertices
def initial_positions : List (Vertex × Vertex) :=
  [(Vertex.A, Vertex.A), (Vertex.B, Vertex.B), (Vertex.C, Vertex.C), 
   (Vertex.D, Vertex.D), (Vertex.E, Vertex.E), (Vertex.F, Vertex.F)]

-- Number of rearrangements such that each character is adjacent to its original position
def number_of_rearrangements : Nat :=
  4

theorem rearrangement_proof : 
  ∃ (f : List (Vertex × Vertex) → List (Vertex × Vertex)), 
  (∀ x ∈ initial_positions, adjacent (x.1) (f initial_positions).find!.2) ∧ 
  (f initial_positions).length = 4 := 
sorry

end rearrangement_proof_l57_57333


namespace b_n_formula_sum_S_formula_min_m_value_l57_57389

namespace ProofProblems

-- Given conditions for sequences a_n and b_n
def a_seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 / 2 ∧ ∀ n, 2 * a (n + 1) = a n - 3 / 2

def b_seq (b : ℕ → ℚ) (a : ℕ → ℚ) : Prop :=
  b 1 = 2 * a 1 ∧ b 3 = 4 * a 1 - a 3 ∧ ∀ n, b n + b (n + 2) = 2 * b (n + 1)

-- Problem 1: Prove the formula for b_n
theorem b_n_formula (a : ℕ → ℚ) (b : ℕ → ℚ) (h_a : a_seq a) (h_b : b_seq b a) :
  ∀ n, b n = n :=
sorry

-- Given definition for c_n
def c_seq (a : ℕ → ℚ) (b : ℕ → ℚ) (c : ℕ → ℚ) : Prop :=
  ∀ n, c n = (a n + 3 / 2) * b n

-- Problem 2: Prove the sum S_n
def sum_S (c : ℕ → ℚ) : ℕ → ℚ
| 0     := 0
| (n+1) := c (n+1) + sum_S n

theorem sum_S_formula (a : ℕ → ℚ) (b : ℕ → ℚ) (c : ℕ → ℚ) (h_a : a_seq a) (h_b : b_seq b a) (h_c : c_seq a b c) :
  ∀ n, sum_S c n = 8 - (n + 2) * (1 / 2)^(n-2) :=
sorry

-- Problem 3: Find the minimum value of m
theorem min_m_value (a : ℕ → ℚ) (b : ℕ → ℚ) (c : ℕ → ℚ) (h_a : a_seq a) (h_b : b_seq b a) (h_c : c_seq a b c) :
  ∃ (n : ℕ), sum_S c n ≤ 2 ∧ ∀ m < 2, ¬ (∃ n, sum_S c n ≤ m) :=
sorry

end ProofProblems

end b_n_formula_sum_S_formula_min_m_value_l57_57389


namespace triangle_cos_range_l57_57517

theorem triangle_cos_range 
  {A B C : ℝ} 
  (hA : A + B + C = π)
  (hSinA : Real.sin A = Real.sqrt 2 / 2) :
  (∀ (x : ℝ), ∃ (B C : ℝ), x = Real.cos B + Real.sqrt 2 * Real.cos C ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  → ((Real.cos B + Real.sqrt 2 * Real.cos C) ∈ (0, 1] ∪ (2, Real.sqrt 5]) :=
  sorry

end triangle_cos_range_l57_57517


namespace find_p_find_p_eta_ge_2_l57_57142

noncomputable def binom_cdf_ge (n : ℕ) (p : ℚ) : ℚ := 
  1 - (Finset.range n).sum (λ k, Nat.choose n k * p^k * (1 - p)^(n - k))

theorem find_p (p : ℚ) (h1 : binom_cdf_ge 2 p = 5/9) : 
  p = 1/3 :=
sorry

theorem find_p_eta_ge_2 (p : ℚ) (h1 : binom_cdf_ge 2 p = 5/9)
  (h2 : p = 1/3) : 
  binom_cdf_ge 3 p - (1 - p)^3 - 3 * p * (1 - p)^2 = 7/27 :=
sorry

end find_p_find_p_eta_ge_2_l57_57142


namespace slope_angle_0_l57_57060

theorem slope_angle_0 (y : ℝ) : y = sqrt 3 → ∃ θ : ℝ, θ = 0 :=
by
  intro hy_eq
  use 0
  sorry

end slope_angle_0_l57_57060


namespace max_area_equilateral_triangle_in_rectangle_l57_57949

-- Define the problem parameters
def rect_width : ℝ := 12
def rect_height : ℝ := 15

-- State the theorem to be proved
theorem max_area_equilateral_triangle_in_rectangle 
  (width height : ℝ) (h_width : width = rect_width) (h_height : height = rect_height) :
  ∃ area : ℝ, area = 369 * Real.sqrt 3 - 540 := 
sorry

end max_area_equilateral_triangle_in_rectangle_l57_57949


namespace total_fault_line_movement_l57_57331

-- Define the movements in specific years.
def movement_past_year : ℝ := 1.25
def movement_year_before : ℝ := 5.25

-- Theorem stating the total movement of the fault line over the two years.
theorem total_fault_line_movement : movement_past_year + movement_year_before = 6.50 :=
by
  -- Proof is omitted.
  sorry

end total_fault_line_movement_l57_57331


namespace polynomial_no_integer_root_l57_57869

variables {P : Polynomial ℤ} {a b c d n : ℤ}

-- Conditions
def conditions :=
  P.eval a = 5 ∧ P.eval b = 5 ∧ P.eval c = 5 ∧ P.eval d = 5 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Statement to prove
theorem polynomial_no_integer_root (h : conditions) : 
  ∀ n : ℤ, P.eval n ≠ 0 :=
sorry

end polynomial_no_integer_root_l57_57869


namespace distance_probability_l57_57524

noncomputable def T := {p | (p = (0, 0)) ∨ (p = (1, 0)) ∨ (p = (0, 1))}

def probability_distance_ge_one_third
  (T : Set (ℝ × ℝ)) : ℝ :=
  let p1 := -- probability calculation for points on specific segments
  let p2 := -- probability calculation for points on specific segments
  let p3 := -- probability calculation for points on specific segments
  (p1 + p2 + p3) / 3

theorem distance_probability 
  (h : ∃ A B ∈ T, dist A B ≥ 1 / 3) :
  probability_distance_ge_one_third T = (p1 + p2 + p3) / 3 :=
sorry

end distance_probability_l57_57524


namespace collinear_Tangency_Points_l57_57534

/-- Given a trapezoid ABCD with AB parallel to CD and AB > CD. This trapezoid is circumscribed around a circle with center I.
Points M and N are the points of tangency of the incircle of triangle ABC with sides AC and AB, respectively. -/
variables {I A B C D M N : Point}

/-- Conditions for the geometry setup: AB parallel to CD, AB > CD, and the trapezoid is inscribed in a circle with center I. 
Note: Tangency conditions for M and N would be included in the proof. -/
axiom trapezoid_circumscribed : parallelogram A B C D → circle (inscribed_in trapezoid A B C D)
axiom parallel_AB_CD : is_parallel (line_through A B) (line_through C D)
axiom AB_greater_CD : length (line_through A B) > length (line_through C D)
axiom tangency_points_M_N : tangency M (circle I) (line_through A C) ∧ tangency N (circle I) (line_through A B)

/-- The theorem stating that points M, N, and I are collinear. -/
theorem collinear_Tangency_Points : collinear {M, N, I} :=
by
  sorry

end collinear_Tangency_Points_l57_57534


namespace minimum_positive_period_of_f_f_at_alpha_l57_57776

def f (x : Real) : Real :=
  2 * sin x * sin (x + π / 3) - (1 / 2)

theorem minimum_positive_period_of_f :
  (∀ x : Real, f (x + π) = f x) :=
by
  sorry

theorem f_at_alpha (α : Real) (hα : tan α = sqrt 3 / 2) :
  f α = 11 / 14 :=
by
  sorry

end minimum_positive_period_of_f_f_at_alpha_l57_57776


namespace area_ABC_find_AC_l57_57323

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
abs ((B.fst - A.fst) * (C.snd - A.snd) - (C.fst - A.fst) * (B.snd - A.snd)) / 2

noncomputable def length (A B : ℝ × ℝ) : ℝ :=
real.sqrt ((B.fst - A.fst) ^ 2 + (B.snd - A.snd) ^ 2)

axiom conditions :
  ∃ (A B C O P T K : ℝ × ℝ),
  (∃ ω : ℝ × ℝ → Prop, ω A ∧ ω O ∧ ω C ∧ ω B) ∧
  (∃ Ω : ℝ × ℝ → Prop, Ω A ∧ Ω O ∧ Ω C ∧ Ω P ∧ Ω B) ∧
  (A ≠ C) ∧ (A ≠ B) ∧ (B ≠ C) ∧ (T ≠ P) ∧
  circle ω O →
  tangent ω A T →
  tangent ω C T →
  triangle_area A P K = 10 ∧
  triangle_area C P K = 6 ∧
  ∃ (equ1 equ2 : ℝ × ℝ),
  length equ1 equ2 = 1 ∧
  line (A P O K C T).

theorem area_ABC (A B C O P T K : ℝ × ℝ) (h : conditions A B C O P T K) :
  triangle_area A B C = 128 / 3 :=
sorry

theorem find_AC (A B C O P T K : ℝ × ℝ) (h : conditions A B C O P T K)
  (h_angle_ABC : ∠(B, A, C) = real.arctan 2) :
  length A C = 4 * real.sqrt 26 / real.sqrt 3 :=
sorry

end area_ABC_find_AC_l57_57323


namespace problem_part1_problem_part2_l57_57381

noncomputable def f (x : ℝ) (a : ℝ) := (1 / 2) * x^2 + a * Real.exp x - Real.log x

theorem problem_part1 (a : ℝ) :
  (∀ x, deriv (λ x, (1 / 2) * x^2 + a * Real.exp x - Real.log x) x = x + a * Real.exp x - 1 / x) →
  deriv (f (1 / 2) a) = 0 →
  a = (3 * Real.sqrt Real.exp 1) / (2 * Real.exp 1) ∧ 
  (∀ x, x > 1 / 2 → deriv (f x a) > 0) ∧ 
  (∀ x, x < 1 / 2 → deriv (f x a) < 0) :=
sorry

theorem problem_part2 (a : ℝ) (h : a > 0) :
  ∀ x > 0, f x a > 1 / 2 :=
sorry

end problem_part1_problem_part2_l57_57381


namespace pressure_force_and_center_of_pressure_l57_57292

-- Definitions based on given conditions
def density_water := 1000 -- in kg/m^3
def gravity := 9.8 -- in m/s^2
def depth := 5.4 -- in meters
def width := 1.5 -- in meters

-- Total force exerted by the water on the sluice gate
def force_exerted := 214926 -- in Newtons

-- Point of application of the force (center of pressure)
def depth_center_of_pressure := 3.6 -- in meters
def horizontal_center_of_pressure := 0.75 -- in meters

-- Proof problem statement
theorem pressure_force_and_center_of_pressure :
  ∃ (F : ℝ) (h_cp x_cp : ℝ),
    F = force_exerted ∧
    h_cp = depth_center_of_pressure ∧
    x_cp = horizontal_center_of_pressure ∧
    F = density_water * gravity * width * depth^2 / 2 ∧
    h_cp = (2 / 3) * depth ∧
    x_cp = width / 2 :=
sorry

end pressure_force_and_center_of_pressure_l57_57292


namespace triangle_angle_sum_l57_57783

theorem triangle_angle_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_ratio : a / 5 = b / 7 ∧ b / 7 = c / 8) : 
  ∃ θ : ℝ, θ = 60 ∧ (180 - θ = 120) :=
by
  have hθ := sorry,
  use hθ,
  split;
  sorry

end triangle_angle_sum_l57_57783


namespace range_of_a_monotonic_increasing_f_x_greater_than_2x_l57_57383

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := real.exp x - 1 + real.log ((x / a) + 1)

-- Problem 1: Prove the range of a for which f(x) is monotonically increasing on (-1,0)
theorem range_of_a_monotonic_increasing (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 0 → (real.exp x + 1 / (x + a) > 0)) ↔ a ∈ set.Iic (1 - real.exp (-1)) ∨ a ∈ set.Ici 1 :=
sorry

-- Problem 2: Prove that f(x) > 2x given the specified conditions
theorem f_x_greater_than_2x (a x : ℝ) (ha : 0 < a ∧ a ≤ 1) (hx : 0 < x) :
  f x a > 2 * x :=
sorry

end range_of_a_monotonic_increasing_f_x_greater_than_2x_l57_57383


namespace shaded_fraction_of_grid_area_l57_57148

-- Definitions of the grid and conditions
def grid_side_length := 6
def shaded_square_area := 1 / 2
def total_grid_area := grid_side_length * grid_side_length

-- Statement of the problem to prove
theorem shaded_fraction_of_grid_area :
  let fraction := shaded_square_area / total_grid_area in
  fraction = 1 / 72 := by
  sorry

end shaded_fraction_of_grid_area_l57_57148


namespace at_least_500_beautiful_l57_57082

noncomputable def is_beautiful (a : ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → ¬ (Nat.prime (a^(n + 2) + 3 * a^n + 1))

theorem at_least_500_beautiful :
  ∃ S : Finset ℕ, S.card ≥ 500 ∧ ∀ a ∈ S, a ∈ { i | i ≤ 2018 } ∧ is_beautiful a :=
by
  sorry

end at_least_500_beautiful_l57_57082


namespace right_triangle_incenter_circumcenter_distance_l57_57306

-- Define a structure for a right triangle with given side lengths.
structure RightTriangle :=
  (a b c : ℕ)
  (h : a * a + b * b = c * c)

-- Define the right triangle with the given side lengths.
def myRightTriangle : RightTriangle := { a := 8, b := 15, c := 17, h := by norm_num }

-- Define the inradius and circumradius for the right triangle.
noncomputable def inradius (T : RightTriangle) : ℝ :=
  let a := T.a in
  let b := T.b in
  let s := (a + b + T.c) / 2 in
  let area := (a * b) / 2 in
  area / s

noncomputable def circumradius (T : RightTriangle) : ℝ :=
  T.c / 2

-- Calculate the distance between the incenter and circumcenter.
noncomputable def distance_incenter_circumcenter (T : RightTriangle) : ℝ :=
  let r := inradius T in
  let R := circumradius T in
  let y := (T.b - T.a) / 2 in -- x, y, z values can be calculated based on triangle solutions
  let DO := (T.c / 2) - y in
  let IO := Real.sqrt (r * r + DO * DO) in
  IO

-- Define the main proof problem statement.
theorem right_triangle_incenter_circumcenter_distance : distance_incenter_circumcenter myRightTriangle = Real.sqrt 85 / 2 :=
by sorry

end right_triangle_incenter_circumcenter_distance_l57_57306


namespace range_of_quadratic_function_l57_57231

theorem range_of_quadratic_function : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 0 ≤ x^2 - 4 * x + 3 ∧ x^2 - 4 * x + 3 ≤ 8 :=
by
  intro x hx
  sorry

end range_of_quadratic_function_l57_57231


namespace sequence_sum_l57_57848

theorem sequence_sum
  (a : ℕ → ℤ)
  (n : ℕ) (h_n : 0 < n)
  (h_a1 : a 1 = 2)
  (h_recur : ∀ k : ℕ, 1 < k → a k = 4 * a (k - 1) + 3) :
  let S_n := ∑ i in Finset.range n, a (i + 1) in
  S_n = 4^n - n - 1 :=
by
  sorry

end sequence_sum_l57_57848


namespace current_price_of_soda_l57_57317

theorem current_price_of_soda (C S : ℝ) (h1 : 1.25 * C = 15) (h2 : C + S = 16) : 1.5 * S = 6 :=
by
  sorry

end current_price_of_soda_l57_57317


namespace find_divisor_l57_57272

-- Define the problem statement in Lean 4
theorem find_divisor :
  ∃ d : ℕ, ∃ q : ℕ, nat.prime q ∧ nat.is_square (9453 - d * q) ∧ 9453 = d * q + (9453 - d * q) ∧ d = 61 :=
sorry

end find_divisor_l57_57272


namespace find_f_23π_over_6_l57_57404

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 (x : ℝ) : f (x + π) = f x + Real.sin x
axiom cond2 (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) : f x = 0

theorem find_f_23π_over_6 : f (23 * π / 6) = 1 / 2 :=
by 
  -- Here the proof would go, but it's marked as sorry for this problem statement.
  sorry

end find_f_23π_over_6_l57_57404


namespace problem1_proof_problem2_proof_l57_57339

noncomputable def problem1 : Real :=
  Real.sqrt 2 * Real.sqrt 3 + Real.sqrt 24

theorem problem1_proof : problem1 = 3 * Real.sqrt 6 :=
  sorry

noncomputable def problem2 : Real :=
  (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3)

theorem problem2_proof : problem2 = 6 :=
  sorry

end problem1_proof_problem2_proof_l57_57339


namespace expected_coffee_more_than_tea_l57_57677

def prime_numbers : Set ℕ := {2, 3, 5, 7}
def composite_numbers : Set ℕ := {4, 6, 8}
def total_days : ℕ := 365

noncomputable def probability (A : Set ℕ) : ℚ := (A.card : ℚ) / 7

noncomputable def expected_days (p : ℚ) : ℚ := p * total_days

theorem expected_coffee_more_than_tea :
  expected_days (probability prime_numbers) - expected_days (probability composite_numbers) = 53 :=
by
  sorry

end expected_coffee_more_than_tea_l57_57677


namespace tetrahedron_inequalities_l57_57865

-- Define the conditions
variable {V : Type*} [inner_product_space ℝ V] (tetrahedron : fin 4 → V) (centroid : V)
variable {R : ℝ} (circumsphere : V → ℝ) -- Circumsphere given by radius R

-- Intersect points A'_i on GA_i
variable (GA_i GA'_i : fin 4 → ℝ)

-- Hypotheses required to state the theorem
hypothesis centroid_def : centroid = (1/4) • (tetrahedron 0 + tetrahedron 1 + tetrahedron 2 + tetrahedron 3)
hypothesis intersection_def : ∀ i, GA'_i i = circumsphere (tetrahedron i) -- Intersection with circumsphere

-- Define the Lean theorem
theorem tetrahedron_inequalities 
  (hGA : ∀ i, GA_i i = dist centroid (tetrahedron i))
  (hGA' : ∀ i, GA'_i i = dist centroid (circumsphere (tetrahedron i))) :
  ( (GA_i 0) * (GA_i 1) * (GA_i 2) * (GA_i 3) <= (GA'_i 0) * (GA'_i 1) * (GA'_i 2) * (GA'_i 3) ) ∧
  ( (1 / (GA'_i 0) + 1 / (GA'_i 1) + 1 / (GA'_i 2) + 1 / (GA'_i 3)) <= (1 / (GA_i 0) + 1 / (GA_i 1) + 1 / (GA_i 2) + 1 / (GA_i 3)) ) := 
sorry

end tetrahedron_inequalities_l57_57865


namespace rationalize_denominator_sum_equals_49_l57_57174

open Real

noncomputable def A : ℚ := -1
noncomputable def B : ℚ := -3
noncomputable def C : ℚ := 1
noncomputable def D : ℚ := 2
noncomputable def E : ℚ := 33
noncomputable def F : ℚ := 17

theorem rationalize_denominator_sum_equals_49 :
  let expr := (A * sqrt 3 + B * sqrt 5 + C * sqrt 11 + D * sqrt E) / F
  49 = A + B + C + D + E + F :=
by {
  -- The proof will go here.
  exact sorry
}

end rationalize_denominator_sum_equals_49_l57_57174


namespace line_through_ellipse_midpoint_l57_57410

theorem line_through_ellipse_midpoint (x y x₁ y₁ x₂ y₂ : ℝ)
  (h_ellipse : x₁^2 / 25 + y₁^2 / 16 = 1)
  (h_ellipse2 : x₂^2 / 25 + y₂^2 / 16 = 1)
  (h_midpoint : (x₁ + x₂) / 2 = 2 ∧ (y₁ + y₂) / 2 = 1)
  (h_point_eq : (64 * (x₁ - x₂) = 50 * (y₁ - y₂))) :
  32 * x - 25 * y - 89 = 0 :=
begin
  sorry
end

end line_through_ellipse_midpoint_l57_57410


namespace no_such_coins_l57_57165

theorem no_such_coins (p1 p2 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1)
  (cond1 : (1 - p1) * (1 - p2) = p1 * p2)
  (cond2 : p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :
  false :=
  sorry

end no_such_coins_l57_57165


namespace correct_option_is_d_l57_57624

theorem correct_option_is_d :
  let a := 2*x + y
  let b := (1 : \R) / 3 * a^2
  let c := (x - y) / \pi
  let m := 0
  let e1 := 1 / x
  let e2 := (5 : \R) * y / (4 * x)
  (is_polynomial a) ∧ (is_polynomial b) ∧ (is_polynomial c) ∧ (is_polynomial m) ∧ 
  ((¬ is_polynomial e1) ∧ (¬ is_polynomial e2)) ∧ 
  -- Verify conditions for other options are false
  ¬ ((coeff (\pi * x^2 * y / 4) = 1 / 4) ∧ (degree (\pi * x^2 * y / 4) = 4)) ∧
  ¬ ((coeff (monomial m) = 0) ∧ (degree (monomial m) = 1)) ∧
  ¬ ((is_cubic (2 * x^2 + x * y^2 + 3)) ∧ (degree (2 * x^2 + x * y^2 + 3) = 2)) :=
sorry

end correct_option_is_d_l57_57624


namespace ten_thousand_times_ten_thousand_l57_57926

theorem ten_thousand_times_ten_thousand :
  (10000 : ℕ) * (10000 : ℕ) = 100000000 := by
  have h₁ : (10000 : ℕ) = 1 * 10^4 := by sorry
  calc
    (10000 : ℕ) * (10000 : ℕ)
        = (1 * 10^4) * (1 * 10^4) : by rw [h₁, h₁]
    ... = 1 * 10^8 : by sorry
    ... = 100000000 : by sorry

end ten_thousand_times_ten_thousand_l57_57926


namespace percent_employed_females_l57_57505

theorem percent_employed_females (percent_employed : ℝ) (percent_employed_males : ℝ) :
  percent_employed = 0.64 →
  percent_employed_males = 0.55 →
  (percent_employed - percent_employed_males) / percent_employed * 100 = 14.0625 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end percent_employed_females_l57_57505


namespace cube_root_integer_l57_57910

theorem cube_root_integer :
  let x := 2^3 * 5^6 * 7^3 in
  real.sqrt3 x = 350 :=
by
  sorry

end cube_root_integer_l57_57910


namespace inequality_solution_l57_57183

theorem inequality_solution 
  (x : ℝ) 
  (h : 2*x^4 + x^2 - 4*x - 3*x^2 * |x - 2| + 4 ≥ 0) : 
  x ∈ Set.Iic (-2) ∪ Set.Icc ((-1 - Real.sqrt 17) / 4) ((-1 + Real.sqrt 17) / 4) ∪ Set.Ici 1 :=
sorry

end inequality_solution_l57_57183


namespace integers_sum_eighteen_l57_57223

theorem integers_sum_eighteen (a b : ℕ) (h₀ : a ≠ b) (h₁ : a < 20) (h₂ : b < 20) (h₃ : Nat.gcd a b = 1) 
(h₄ : a * b + a + b = 95) : a + b = 18 :=
by
  sorry

end integers_sum_eighteen_l57_57223


namespace correct_answer_l57_57993

-- Conditions in Lean 4
def propositionA : Prop := (4, 3) = (3, 4)

def propositionB : Prop := ∀ x : ℝ, (x^2 + 1 > 0 ∧ -4 < 0)

def perpendicular (l1 l2 : ℝ → ℝ) : Prop := ∀ m, l1 m = -1/(l2 m)

def propositionC (l1 l2 l3 : ℝ → ℝ) : Prop := perpendicular l1 l3 ∧ perpendicular l2 l3 → ∀ x : ℝ, l1 x = l2 x

def propositionD : Prop := ∀ α β : ℝ, α = β → vertical α β

-- Correct Answer
theorem correct_answer : propositionB := sorry

end correct_answer_l57_57993


namespace part1_part2_part3_l57_57019

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2
noncomputable def g (x : ℝ) : ℝ := 10 * (x - 1)
def a (n : ℕ) : ℝ := sorry  -- define the sequence {a_n} with the given recurrence relation
def b (n : ℕ) : ℝ := (9 / 10) * (n + 2) * (a n - 1)

theorem part1 : ∀ n : ℕ, ∃ r : ℝ, (a n - 1) = r ^ n :=
sorry

theorem part2 : ∃ n : ℕ, ∀ m : ℕ, b m ≤ b n ∧ (n = 7 ∨ n = 8) := 
sorry

theorem part3 (t : ℝ) : (∀ m : ℕ, m > 0 → (t ^ m) / (b m) < (t ^ (m + 1)) / (b (m + 1))) → t > 6 / 5 :=
sorry

end part1_part2_part3_l57_57019


namespace point_quadrant_l57_57904

def Point (x : ℝ) (y : ℝ) := Prod ℝ ℝ

def First_Quadrant (p : Point) : Prop := p.fst > 0 ∧ p.snd > 0

def lies_in_quadrant (p : Point) : String :=
  if p.fst > 0 ∧ p.snd > 0 then "First quadrant"
  else if p.fst < 0 ∧ p.snd > 0 then "Second quadrant"
  else if p.fst < 0 ∧ p.snd < 0 then "Third quadrant"
  else if p.fst > 0 ∧ p.snd < 0 then "Fourth quadrant"
  else "On Axis"

theorem point_quadrant : lies_in_quadrant (Point.mk 3 4) = "First quadrant" :=
by
  sorry

end point_quadrant_l57_57904


namespace inequality_bi_l57_57137

variable {α : Type*} [LinearOrderedField α]

-- Sequence of positive real numbers
variable (a : ℕ → α)
-- Conditions for a_i
variable (ha : ∀ i, i > 0 → i * (a i)^2 ≥ (i + 1) * a (i - 1) * a (i + 1))
-- Positive real numbers x and y
variables (x y : α) (hx : x > 0) (hy : y > 0)
-- Definition of b_i
def b (i : ℕ) : α := x * a i + y * a (i - 1)

theorem inequality_bi (i : ℕ) (hi : i ≥ 2) : i * (b a x y i)^2 > (i + 1) * (b a x y (i - 1)) * (b a x y (i + 1)) := 
sorry

end inequality_bi_l57_57137


namespace largest_k_such_that_S_k_eq_0_l57_57878

noncomputable def S : ℕ → ℤ
| 0       := 0
| (k + 1) := (finset.range (k + 1)).sum (λ i, (i.succ : ℤ) * (if S i < i.succ then 1 else -1))

theorem largest_k_such_that_S_k_eq_0 : ∃ (k : ℕ), k ≤ 2010 ∧ S k = 0 ∧ ∀ (k' : ℕ), k' ≤ 2010 ∧ S k' = 0 → k' ≤ 1092 :=
by
  sorry

end largest_k_such_that_S_k_eq_0_l57_57878


namespace correct_relation_l57_57626

-- Definitions of sets and conditions
def empty_set : Set ℕ := ∅
def single_set : Set ℕ := {0}
def zero : ℕ := 0

-- Theorem to prove given the conditions
theorem correct_relation : ¬(empty_set ∈ single_set) ∧ ¬(zero ⊆ single_set) ∧ (zero ∈ single_set) ∧ ¬(empty_set = single_set) :=
by
  sorry

end correct_relation_l57_57626


namespace sum_of_variables_is_233_l57_57172

-- Define A, B, C, D, E, F with their corresponding values.
def A : ℤ := 13
def B : ℤ := 9
def C : ℤ := -3
def D : ℤ := -2
def E : ℕ := 165
def F : ℕ := 51

-- Define the main theorem to prove the sum of A, B, C, D, E, F equals 233.
theorem sum_of_variables_is_233 : A + B + C + D + E + F = 233 := 
by {
  -- Proof is not required according to problem statement, hence using sorry.
  sorry
}

end sum_of_variables_is_233_l57_57172


namespace min_travel_time_l57_57247

/-- Two people, who have one bicycle, need to travel from point A to point B, which is 40 km away from point A. 
The first person walks at a speed of 4 km/h and rides the bicycle at 30 km/h, 
while the second person walks at a speed of 6 km/h and rides the bicycle at 20 km/h. 
Prove that the minimum time in which they can both get to point B is 25/9 hours. -/
theorem min_travel_time (d : ℕ) (v_w1 v_c1 v_w2 v_c2 : ℕ) (min_time : ℚ) 
  (h_d : d = 40)
  (h_v1_w : v_w1 = 4)
  (h_v1_c : v_c1 = 30)
  (h_v2_w : v_w2 = 6)
  (h_v2_c : v_c2 = 20)
  (h_min_time : min_time = 25 / 9) :
  ∃ y x : ℚ, 4*y + (2/3)*y*30 = 40 ∧ min_time = y + (2/3)*y :=
sorry

end min_travel_time_l57_57247


namespace angle_A_is_pi_over_6_maximum_area_of_triangle_l57_57031

open Real

variables {a b c : ℝ} {A B C : ℝ}

-- Problem (I)
theorem angle_A_is_pi_over_6
(h1 : ∀ (t : ℝ), t = (1 / 2) → b = 2 * a * sin A * cos C + c * sin (2 * A))
(h2 : 0 < A) (h3 : A < π / 2) :
A = π / 6 :=
by
  sorry

-- Problem (II)
noncomputable def triangle_area (a b c : ℝ) (A : ℝ) : ℝ :=
1 / 2 * b * c * sin A

theorem maximum_area_of_triangle
  (h1 : a = 2)
  (h2 : ∀ (b c : ℝ), a^2 = b^2 + c^2 - 2 * b * c * cos A → bc ≤ 4 / (2 - sqrt 3) ∧ (bc = 4 / (2 - sqrt 3) ↔ b = c = sqrt (6) + 2))
  :
  ∃ b c : ℝ, triangle_area 2 b c (π / 6) = 2 + sqrt 3 :=
by
  sorry

end angle_A_is_pi_over_6_maximum_area_of_triangle_l57_57031


namespace four_dim_measure_correct_l57_57095

variables (r : ℝ)

/- Conditions -/
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def area2D (r : ℝ) : ℝ := Real.pi * r ^ 2
def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r ^ 2
def volume3D (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3
def volume4D (r : ℝ) : ℝ := 8 * Real.pi * r ^ 3

/- Hypothesis based on observed pattern -/
def four_dim_measure (r : ℝ) : ℝ := 2 * Real.pi * r ^ 4

/- Theorem -/
theorem four_dim_measure_correct :
  ∀ r : ℝ, (∃ V, volume4D r = V) → (∃ W, W = four_dim_measure r) :=
by
  intros
  exists volume4D r
  exists four_dim_measure r
  sorry

end four_dim_measure_correct_l57_57095


namespace equal_after_operations_l57_57605

theorem equal_after_operations :
  let initial_first_number := 365
  let initial_second_number := 24
  let first_number_after_n_operations := initial_first_number - 19 * 11
  let second_number_after_n_operations := initial_second_number + 12 * 11
  first_number_after_n_operations = second_number_after_n_operations := sorry

end equal_after_operations_l57_57605


namespace solve_for_x_l57_57713

theorem solve_for_x (x : ℝ) (h : log 8 (3 * x - 5) = 2) : x = 23 := 
  sorry

end solve_for_x_l57_57713


namespace johnson_family_seating_l57_57192

/-- The total number of ways to seat 6 boys and 4 girls in 10 chairs with at least 3 boys next to each other is 3627792. -/
theorem johnson_family_seating :
  ∃ (arrangements : ℕ), arrangements = 10.factorial - ((9.factorial / (4.factorial * 5.factorial)) * 2^3) ∧ arrangements = 3627792 :=
by
  sorry

end johnson_family_seating_l57_57192


namespace even_numbers_count_l57_57069

theorem even_numbers_count (a b : ℕ) (h1 : 150 < a) (h2 : a % 2 = 0) (h3 : b < 350) (h4 : b % 2 = 0) (h5 : 150 < b) (h6 : a < 350) (h7 : 154 ≤ b) (h8 : a ≤ 152) :
  ∃ n : ℕ, ∀ k : ℕ, k = 99 ↔ 2 * k + 150 = b - a + 2 :=
by
  sorry

end even_numbers_count_l57_57069


namespace smallest_int_neither_prime_nor_square_no_prime_lt_70_l57_57984

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬ p ∣ n

theorem smallest_int_neither_prime_nor_square_no_prime_lt_70
  (n : ℕ) : 
  n = 5183 ∧ ¬ is_prime n ∧ ¬ is_square n ∧ has_no_prime_factor_less_than n 70 ∧
  (∀ m : ℕ, 0 < m → m < 5183 →
    ¬ (¬ is_prime m ∧ ¬ is_square m ∧ has_no_prime_factor_less_than m 70)) :=
by sorry

end smallest_int_neither_prime_nor_square_no_prime_lt_70_l57_57984


namespace kyro_percentage_paid_l57_57330

theorem kyro_percentage_paid
    (aryan_debt : ℕ) -- Aryan owes Fernanda $1200
    (kyro_debt : ℕ) -- Kyro owes Fernanda
    (aryan_debt_twice_kyro_debt : aryan_debt = 2 * kyro_debt) -- Aryan's debt is twice what Kyro owes
    (aryan_payment : ℕ) -- Aryan's payment
    (aryan_payment_percentage : aryan_payment = 60 * aryan_debt / 100) -- Aryan pays 60% of her debt
    (initial_savings : ℕ) -- Initial savings in Fernanda's account
    (final_savings : ℕ) -- Final savings in Fernanda's account
    (initial_savings_cond : initial_savings = 300) -- Fernanda's initial savings is $300
    (final_savings_cond : final_savings = 1500) -- Fernanda's final savings is $1500
    : kyro_payment = 80 * kyro_debt / 100 := -- Kyro paid 80% of her debt
by {
    sorry
}

end kyro_percentage_paid_l57_57330


namespace avg_rate_of_change_l57_57558

def f (x : ℝ) := 2 * x + 1

theorem avg_rate_of_change : (f 5 - f 1) / (5 - 1) = 2 := by
  sorry

end avg_rate_of_change_l57_57558


namespace white_mice_count_l57_57480

variable (T W B : ℕ) -- Declare variables T (total), W (white), B (brown)

def W_condition := W = (2 / 3) * T  -- White mice condition
def B_condition := B = 7           -- Brown mice condition
def T_condition := T = W + B       -- Total mice condition

theorem white_mice_count : W = 14 :=
by
  sorry  -- Proof to be filled in

end white_mice_count_l57_57480


namespace simplify_expression_l57_57996

theorem simplify_expression 
  (a b : ℝ) 
  (h1 : a ≠ b) 
  (h2 : 0 < a) 
  (h3 : 0 < b) 
  (h4 : -1 < a ∧ a < 1) :
  ((a ^ (1/2) - b ^ (1/2))⁻¹ * (a ^ (3/2) - b ^ (3/2)) - (sqrt(a) + sqrt(b)) ^ 2)
  / ((ab.sqrt * ab).sqrt ^ 3 + (1 + (a * (1 - a ^ 2) ^ (-1/2)) ^ 2)) = -a ^ 2 := 
begin
  sorry
end

end simplify_expression_l57_57996


namespace apples_to_pears_l57_57807

theorem apples_to_pears (a o p : ℕ) 
  (h1 : 10 * a = 5 * o) 
  (h2 : 3 * o = 4 * p) : 
  (20 * a) = 40 / 3 * p :=
sorry

end apples_to_pears_l57_57807


namespace handrail_length_correct_l57_57673

noncomputable def handrail_length_of_spiral_staircase 
  (deg_turn : ℝ) (height : ℝ) (radius : ℝ) : ℝ :=
  let arc_length := (deg_turn / 360) * 2 * Real.pi * radius
  in Real.sqrt (height^2 + arc_length^2)

theorem handrail_length_correct :
  handrail_length_of_spiral_staircase 225 12 4 ≈ 19.8 := by
  sorry

end handrail_length_correct_l57_57673


namespace integral_value_l57_57230

noncomputable def integral_func (x : ℝ) : ℝ := real.sqrt (x * (2 - x))

theorem integral_value :
    ∫ x in 0..1, integral_func x = real.pi / 4 :=
by
  sorry

end integral_value_l57_57230


namespace find_area_ABCD_l57_57844

-- Define all points A, B, C, D, E, F, G, H as variables in R²
variables {A B C D E F G H : Type}

-- Define condition of parallelogram
def isParallelogram (A B C D : Type) : Prop := 
  (A, B, C, D are vertices of a parallelogram)

-- Define E as the midpoint of CD
def isMidpoint (E C D : Type) : Prop :=
  (coordinates E = ((coordinates C + coordinates D) / 2))

-- Define intersections
def intersection (X1 Y1 X2 Y2 : Type) : Type :=
  { P : Type | P lies on both line X1Y1 and line X2Y2 }

-- Define conditions on intersections
def F_intersection : Prop :=
  F = intersection A E B D

def H_intersection : Prop :=
  H = intersection A C B E

def G_intersection : Prop :=
  G = intersection A C B D

-- Define area of EHGF
def area_EHGF : ℝ := 15

-- Define target: area of parallelogram ABCD
def area_ABCD : ℝ :=
  180

-- The final statement to prove
theorem find_area_ABCD (h_parallelogram : isParallelogram A B C D)
  (h_midpoint : isMidpoint E C D)
  (h_F : F_intersection)
  (h_H : H_intersection)
  (h_G : G_intersection)
  (area_EQ : 15 = area_EHGF) :
  area_ABCD = 180 :=
begin
  sorry
end

end find_area_ABCD_l57_57844


namespace other_asymptote_of_hyperbola_l57_57544

theorem other_asymptote_of_hyperbola :
  ∀ (x y : ℝ),
  (∀ x, y = 2 * x) →
  (∀ (a b : ℝ), a = 4 → y = 2 * 4 → b = 8) →
  (∀ y, y ≠ 2 * x → y = - (1 / 2) * x + 10) :=
by assumption -- we will assume these conditions are given

end other_asymptote_of_hyperbola_l57_57544


namespace solve_for_x0_l57_57765

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 3^(x + 1) else log 2 x

theorem solve_for_x0 (x0 : ℝ) (h : f x0 = 3) : x0 = 0 ∨ x0 = 8 :=
sorry

end solve_for_x0_l57_57765


namespace number_of_positive_integer_solutions_l57_57010

def is_integer_solution (k : ℕ) := 
  ∃ x : ℤ, k * x - 10 = 2 * k

def number_of_solutions (n : ℕ) :=
  ∑ k in { k ∈ finset.range n | is_integer_solution k }, 1

theorem number_of_positive_integer_solutions :
  number_of_solutions 11 = 4 :=
by
  sorry

end number_of_positive_integer_solutions_l57_57010


namespace john_spent_fraction_l57_57860

theorem john_spent_fraction (initial_money snacks_left necessities_left snacks_fraction : ℝ)
  (h1 : initial_money = 20)
  (h2 : snacks_fraction = 1/5)
  (h3 : snacks_left = initial_money * snacks_fraction)
  (h4 : necessities_left = 4)
  (remaining_money : ℝ) (h5 : remaining_money = initial_money - snacks_left)
  (spent_on_necessities : ℝ) (h6 : spent_on_necessities = remaining_money - necessities_left) 
  (fraction_spent : ℝ) (h7 : fraction_spent = spent_on_necessities / remaining_money) : 
  fraction_spent = 3/4 := 
sorry

end john_spent_fraction_l57_57860


namespace like_terms_sum_l57_57034

theorem like_terms_sum (m n : ℕ) (h1 : 2 * m = 2) (h2 : n = 3) : m + n = 4 :=
sorry

end like_terms_sum_l57_57034


namespace weight_of_new_girl_l57_57273

theorem weight_of_new_girl (W N : ℝ) (h_weight_replacement: (20 * W / 20 + 40 - 40 + 40) / 20 = W / 20 + 2) :
  N = 80 :=
by
  sorry

end weight_of_new_girl_l57_57273


namespace graph_empty_l57_57706

theorem graph_empty {x y : ℝ} : 
  x^2 + 3 * y^2 - 4 * x - 6 * y + 10 = 0 → false := 
by 
  sorry

end graph_empty_l57_57706


namespace arithmetic_sequence_general_term_arithmetic_sequence_max_sum_l57_57029

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) : 
  (∀ n : ℕ, a n = a 1 + (n - 1) * (-2)) → 
  a 2 = 1 → 
  a 5 = -5 → 
  ∀ n : ℕ, a n = -2 * n + 5 :=
by
  intros h₁ h₂ h₅
  sorry

theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * (-2)) →
  a 2 = 1 → 
  a 5 = -5 → 
  ∃ n : ℕ, n = 2 ∧ S n = 4 :=
by
  intros hSn h₂ h₅
  sorry

end arithmetic_sequence_general_term_arithmetic_sequence_max_sum_l57_57029


namespace remaining_shape_perimeter_l57_57305

def rectangle_perimeter (L W : ℕ) : ℕ := 2 * (L + W)

theorem remaining_shape_perimeter (L W S : ℕ) (hL : L = 12) (hW : W = 5) (hS : S = 2) :
  rectangle_perimeter L W = 34 :=
by
  rw [hL, hW]
  rfl

end remaining_shape_perimeter_l57_57305


namespace interval_of_y_l57_57299

theorem interval_of_y (y : ℝ) (h : y = (1 / y) * (-y) - 5) : -6 ≤ y ∧ y ≤ -4 :=
by sorry

end interval_of_y_l57_57299


namespace number_of_correct_propositions_l57_57875

-- Define basic types for lines and planes
variable (Line : Type) (Plane : Type)

-- Define basic relationships
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contain_line : Plane → Plane → Line)
variable (parallel_plane : Line → Plane → Prop)

noncomputable def proposition1 (m l : Line) (α : Plane) : Prop :=
  parallel m l → perpendicular m α → perpendicular l α

noncomputable def proposition2 (m l : Line) (α : Plane) : Prop :=
  parallel m l → parallel_plane m α → parallel_plane l α

noncomputable def proposition3 (α β γ : Plane) (l m n : Line) : Prop :=
  contain_line α β = l ∧ contain_line β γ = m ∧ contain_line γ α = n →
  parallel l m ∧ parallel m n

noncomputable def proposition4 (α β γ : Plane) (m l n : Line) : Prop :=
  contain_line α β = m ∧ contain_line β γ = l ∧ contain_line γ α = n ∧ parallel n β →
  parallel l m

theorem number_of_correct_propositions :
  ∀ (Line Plane : Type) (parallel : Line → Line → Prop) (perpendicular : Line → Plane → Prop)
    (contain_line : Plane → Plane → Line) (parallel_plane : Line → Plane → Prop),
  let l1 := proposition1 Line Plane parallel perpendicular,
      l2 := proposition2 Line Plane parallel parallel_plane,
      l3 := proposition3 Line Plane contain_line parallel,
      l4 := proposition4 Line Plane contain_line parallel in
  (if l1 then 1 else 0) + (if l2 then 1 else 0) + (if l3 then 1 else 0) + (if l4 then 1 else 0) = 2 :=
by sorry

end number_of_correct_propositions_l57_57875


namespace propositions_truth_l57_57320

theorem propositions_truth :
  let P1 := (∀ x > 0, x^2 - x ≤ 0) ↔ (∃ x > 0, x^2 - x > 0),
      P2 := ∀ (A B : ℝ), (A > B → sin A > sin B),
      P3 := ∀ (a : ℕ → ℝ) (n : ℕ), (a n * a (n + 2) = (a (n + 1))^2 ↔ (∃ r, ∃ (k m : ℤ), (a n = r^k ∧ a (n + 1) = r^(k + 1) ∧ a (n + 2) = r^(k + 2))),
      f := (λ x : ℝ, log x + 1 / log x),
      P4 := ∀ x > 0, f x ≥ 2
  in (P1 ∧ ¬P2 ∧ ¬P3 ∧ ¬P4) = true :=
by
  sorry

end propositions_truth_l57_57320


namespace length_major_axis_of_ellipse_l57_57325

theorem length_major_axis_of_ellipse 
  (F1 F2 : ℝ × ℝ)
  (hx : F1 = (9, 20))
  (hy : F2 = (49, 55))
  (tangent_x_axis : True) :
  let F2' := (49, -55) in
  dist F1 F2' = 85 :=
sorry

end length_major_axis_of_ellipse_l57_57325


namespace triangle_area_angle_C_l57_57512

noncomputable def area_of_triangle (a b c B : ℝ) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem triangle_area :
    (B = Real.pi * 5 / 6) →
    (a = Real.sqrt 3 * c) →
    (b = 2 * Real.sqrt 7) →
    c = 2 →
    area_of_triangle a b c B = Real.sqrt 3 :=
  by
  sorry

theorem angle_C :
    (A B C : ℝ) →
    (B = 150 * Real.pi / 180) →
    (sin A + Real.sqrt 3 * sin C = Real.sqrt 2 / 2) →
    (A + B + C = Real.pi) →
    C = 15 * Real.pi / 180 :=
  by
  sorry

end triangle_area_angle_C_l57_57512


namespace angle_between_vectors_l57_57397

noncomputable def e1 : ℝˣ := sorry  -- Representing e1 as a non-zero real unit vector
noncomputable def e2 : ℝˣ := sorry  -- Representing e2 as a non-zero real unit vector

axiom unit_vectors : ∥e1∥ = 1 ∧ ∥e2∥ = 1
axiom perpendicular_condition : e1 • (e1 + 2 * e2) = 0

theorem angle_between_vectors : real.angle e1 e2 = 120 :=
by sorry

end angle_between_vectors_l57_57397


namespace vertex_and_value_l57_57777

def parabola (x : ℝ) : ℝ := x^2 - 8*x + 12

theorem vertex_and_value :
  (∃ x y : ℝ, y = parabola x ∧ x = - (Parabola.coeff_b / (2 * Parabola.coeff_a)) ∧ 
    y = parabola (- (Parabola.coeff_b / (2 * Parabola.coeff_a))) ∧ (x, y) = (4, -4)) ∧
  (parabola 3 = -3) :=
begin
  sorry
end

end vertex_and_value_l57_57777


namespace equilateral_triangle_perimeter_l57_57195

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 3 * s = 8 * Real.sqrt 3 := 
by 
  sorry

end equilateral_triangle_perimeter_l57_57195


namespace area_enclosed_by_abs_eq_6_l57_57708

theorem area_enclosed_by_abs_eq_6 : 
  let S := {p : ℝ × ℝ | abs p.1 + abs p.2 = 6} in
  (measure_theory.measure.hm (measure_theory.outer_measure.caratheodory.measure S)) = 72 :=
by
  sorry

end area_enclosed_by_abs_eq_6_l57_57708


namespace square_product_third_sides_of_right_triangles_l57_57911

theorem square_product_third_sides_of_right_triangles (T₁ T₂ : Type)
  [Triangle T₁] [RightTriangle T₁] [HasArea T₁ (3 : ℝ)]
  [Triangle T₂] [RightTriangle T₂] [HasArea T₂ (2 : ℝ)]
  (s₁ s₂ : T₁)
  (s₃ s₄ : T₂)
  (h₁ : CongruentSide s₁ s₂)
  (h₂ : CongruentSide s₃ s₄) :
  product_square_third_sides T₁ T₂ = 216 := 
sorry

end square_product_third_sides_of_right_triangles_l57_57911


namespace calc_f_ln_log5_2_l57_57412

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  ((x + 1) ^ 2 + a * sin x) / (x ^ 2 + 1) + 3

theorem calc_f_ln_log5_2 (a : ℝ) (h : f a (Real.log (Real.log 5 / Real.log 2)) = 5) :
  f a (Real.log (Real.log 2 / Real.log 5)) = 3 :=
  sorry

end calc_f_ln_log5_2_l57_57412


namespace distance_from_right_focus_to_line_correct_l57_57202

def distance_from_right_focus_to_line (a b : ℝ) (right_focus distance : ℝ) : Prop :=
  (1 < a) ∧ (0 < b) ∧ (right_focus = sqrt (a^2 - b^2)) ∧
  (distance = abs (right_focus * sqrt(2) / 2))

theorem distance_from_right_focus_to_line_correct (a b : ℝ) (H : 1 < a) (H1 : 0 < b) :
  ∃ (right_focus distance : ℝ), distance_from_right_focus_to_line a b right_focus distance ∧
    distance = sorry := -- price 'B' computed distance to be used instead of 'sorry'
by
  sorry

end distance_from_right_focus_to_line_correct_l57_57202


namespace apples_left_to_eat_l57_57884

theorem apples_left_to_eat (total_apples : ℕ) (one_fifth_has_worms : total_apples / 5) (nine_more_bruised : one_fifth_has_worms + 9):
  let wormy_apples := total_apples / 5 in
  let bruised_apples := wormy_apples + 9 in
  let apples_left_raw := total_apples - wormy_apples - bruised_apples in
  total_apples = 85 →
  apples_left_raw = 42 :=
by
  intros
  sorry

end apples_left_to_eat_l57_57884


namespace true_statements_count_l57_57840

/-- Define "polyline distance" between two points. -/
def polyline_distance (P Q : ℝ × ℝ) : ℝ := |P.1 - Q.1| + |P.2 - Q.2|

/-- Points A, B, C, M, N in a 2D Cartesian coordinate system -/
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (1, 0)

/-- Statement ① -/
def statement_1 : Prop := polyline_distance A B = 5

/-- Statement ② -/
def statement_2 : Prop := 
  ∀ (P : ℝ × ℝ), polyline_distance P (0, 0) = 1 → (|P.1| + |P.2| = 1)

/-- Statement ③ -/
def statement_3 : Prop :=
  ∀ (C : ℝ × ℝ), 
    (C.1 ≥ -1 ∧ C.1 ≤ 1) ∧ (C.2 ≤ 3 ∧ C.2 ≥ 0) →
    polyline_distance A C + polyline_distance C B = polyline_distance A B

/-- Statement ④ -/
def statement_4 : Prop :=
  ∀ (P : ℝ × ℝ), 
    (polyline_distance P M = polyline_distance P N) → 
    (P.1 = 0)

/-- Total number of true statements -/
def number_of_true_statements : ℕ :=
  (if statement_1 then 1 else 0) + 
  (if statement_2 then 1 else 0) + 
  (if statement_3 then 1 else 0) + 
  (if statement_4 then 1 else 0)

/-- Proof that the number of true statements is 3 -/
theorem true_statements_count : number_of_true_statements = 3 := by 
  sorry

end true_statements_count_l57_57840


namespace union_sets_l57_57132

def S : Set ℕ := {0, 1}
def T : Set ℕ := {0, 3}

theorem union_sets : S ∪ T = {0, 1, 3} :=
by
  sorry

end union_sets_l57_57132


namespace distance_between_lights_l57_57555

/-- Small lights are hung on a string 8 inches apart in the order blue, blue, red, red, red, 
and this pattern continues. 1 foot is equal to 12 inches. We want to show that the distance between the 
4th red light and the 28th red light is 26 feet. -/
theorem distance_between_lights 
  (inches_per_light_spacing : ℕ)
  (pattern_length : ℕ)
  (blue_count : ℕ)
  (red_count : ℕ)
  (feet_per_inch : ℕ)
  (fourth_red_pos : ℕ)
  (twenty_eighth_red_pos : ℕ)
  (total_inches_between_lights : ℕ)
  (total_feet_between_lights : ℕ)
  : total_feet_between_lights = 26 := 
by 
  -- Conditions
  have inches_per_light_spacing := 8,
  have pattern_length := 5,
  have blue_count := 2,
  have red_count := 3,
  have feet_per_inch := 12,
  have fourth_red_pos := 8,
  have twenty_eighth_red_pos := 48,
  have total_inches_between_lights := (twenty_eighth_red_pos - fourth_red_pos) * inches_per_light_spacing,
  have total_feet_between_lights := total_inches_between_lights / feet_per_inch,
  
  -- Concluding the theorem
  sorry

end distance_between_lights_l57_57555


namespace incorrect_simplification_l57_57627

theorem incorrect_simplification :
  (-(1 + 1/2) ≠ 1 + 1/2) := 
by sorry

end incorrect_simplification_l57_57627


namespace matrix_equation_l57_57341

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 0],
  ![4, -2]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![5, -1],
  ![2, 3]
]

def I : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![1, 0],
  ![0, 1]
]

theorem matrix_equation : 
  2 • (A ⬝ B) + 3 • I = ![
  ![33, -6],
  ![32, -17]] := by
  sorry

end matrix_equation_l57_57341


namespace tetrahedron_equal_edges_if_equal_areas_l57_57831

variables (A B C D : ℝ^3)
def face_area (P Q R : ℝ^3) : ℝ := 
  (1 / 2) * ((Q - P) × (R - P)).norm

theorem tetrahedron_equal_edges_if_equal_areas
  (h1 : face_area A B C = face_area A B D)
  (h2 : face_area A C D = face_area B C D)
  (h3 : face_area A B C = face_area A C D)
  (h4 : face_area A B C = face_area B C D) :
  dist A B = dist A C ∧ dist A C = dist A D ∧ dist A D = dist B C ∧ dist B C = dist B D ∧ dist B D = dist C D := sorry

end tetrahedron_equal_edges_if_equal_areas_l57_57831


namespace kevin_total_distance_l57_57861

noncomputable def kevin_hop_total_distance_after_seven_leaps : ℚ :=
  let a := (1 / 4 : ℚ)
  let r := (3 / 4 : ℚ)
  let n := 7
  a * (1 - r^n) / (1 - r)

theorem kevin_total_distance (total_distance : ℚ) :
  total_distance = kevin_hop_total_distance_after_seven_leaps → 
  total_distance = 14197 / 16384 := by
  intro h
  sorry

end kevin_total_distance_l57_57861


namespace cubic_function_sum_l57_57017

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 + 3 * x - (5 / 12)

theorem cubic_function_sum :
  ∑ k in Finset.range 2016, f ((k + 1) / 2017) = 2016 :=
by
  sorry

end cubic_function_sum_l57_57017


namespace problem_statement_l57_57813

noncomputable theory

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def eight_times_inverse_domain_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ set.Icc a b, x^2 - 4*x + 8/x = 0 → x ∈ set.Icc 2 (real.sqrt 5 + 1)

theorem problem_statement (g : ℝ → ℝ) (m : ℝ) (h_odd: is_odd_function g)
  (h_neg : ∀ x ∈ set.Iic 0, g x = x^2 + (m+2)*x + m - 2) :
  m = 2 → eight_times_inverse_domain_interval (λ x, -x^2 + 4*x) 2 (real.sqrt 5 + 1) :=
begin
  sorry
end

end problem_statement_l57_57813


namespace total_plate_combinations_l57_57088

open Nat

def valid_letters := 24
def letter_positions := (choose 4 2)
def valid_digits := 10
def total_combinations := letter_positions * (valid_letters * valid_letters) * (valid_digits ^ 3)

theorem total_plate_combinations : total_combinations = 3456000 :=
  by
    -- Replace this sorry with steps to prove the theorem
    sorry

end total_plate_combinations_l57_57088


namespace collinear_points_l57_57525

noncomputable def circles_intersection {P Q X Y Z A B C A' B' C' : Type} 
  (ω1 ω2 ω3 : circle) (Q : point) (A : point) 
  (A' : point) (B' : point) (C' : point) (B : point) (C : point) (concur : Q ∈ ω1 ∧ Q ∈ ω2 ∧ Q ∈ ω3) 
  (interA' : A' ∈ ω2 ∧ A' ∈ ω3) (interB' : B' ∈ ω1 ∧ B' ∈ ω3) (interC' : C' ∈ ω1 ∧ C' ∈ ω2) 
  (A_on_ω1 : A ∈ ω1) (B_on_ω2 : B ∈ ω2) (C_on_ω3 : C ∈ ω3) 
  (AB' : B ∈ line A B') (AC' : C ∈ line A C') : Prop := 
  collinear (B, C, A')

theorem collinear_points 
  {P Q X Y Z A B C A' B' C' : Type}
  (ω1 ω2 ω3 : circle) (Q : point) (A : point) 
  (A' : point) (B' : point) (C' : point) (B : point) (C : point) 
  (concur : Q ∈ ω1 ∧ Q ∈ ω2 ∧ Q ∈ ω3)
  (interA' : A' ∈ ω2 ∧ A' ∈ ω3)
  (interB' : B' ∈ ω1 ∧ B' ∈ ω3)
  (interC' : C' ∈ ω1 ∧ C' ∈ ω2)
  (A_on_ω1 : A ∈ ω1)
  (B_on_ω2 : B ∈ ω2)
  (C_on_ω3 : C ∈ ω3)
  (AB' : B ∈ line A B')
  (AC' : C ∈ line A C') :
    collinear (B, C, A') := 
by { sorry }

end collinear_points_l57_57525


namespace arithmetic_geom_seq_l57_57498

noncomputable def geom_seq (a q : ℝ) : ℕ → ℝ 
| 0     => a
| (n+1) => q * (geom_seq a q n)

theorem arithmetic_geom_seq
  (a q : ℝ)
  (h_arith : 2 * geom_seq a q 1 = 1 + (geom_seq a q 2 - 1))
  (h_q : q = 2) :
  (geom_seq a q 2 + geom_seq a q 3) / (geom_seq a q 4 + geom_seq a q 5) = 1 / 4 :=
by
  sorry

end arithmetic_geom_seq_l57_57498


namespace binomial_coefficient_sum_l57_57576

-- Define the polynomial
def polynomial (x : ℝ) : ℝ := (1 + 2 * x) ^ 5

-- Define the problem statement
theorem binomial_coefficient_sum : 
  (∑ i in Finset.range (5 + 1), Nat.choose 5 i * (2 : ℕ) ^ i) = 32 :=
by
  sorry

end binomial_coefficient_sum_l57_57576


namespace circles_share_chord_ratio_constant_l57_57553

theorem circles_share_chord_ratio_constant
  (A B : Point)
  (circle1 circle2 circle3 : Circle)
  (h1 : chord_in_circle A B circle1)
  (h2 : chord_in_circle A B circle2)
  (h3 : chord_in_circle A B circle3) :
  ∀ (l : Line), (l.contains A) ∧ (¬ l.is_chord AB) →
  ∀ (X Y Z : Point), (X ≠ B) ∧ (on_circle X circle1) ∧ 
  (l.contains X) ∧ (AX_intersects_two_circles X l circle2 circle3 Y Z) → 
  (Y_between_X_and_Z Y X Z) →
  ∃ k : ℝ, XY / YZ = k := 
begin
  sorry
end

end circles_share_chord_ratio_constant_l57_57553


namespace percent_increase_twice_eq_44_percent_l57_57219

variable (P : ℝ) (x : ℝ)

theorem percent_increase_twice_eq_44_percent (h : P * (1 + x)^2 = P * 1.44) : x = 0.2 :=
by sorry

end percent_increase_twice_eq_44_percent_l57_57219


namespace vertical_asymptote_values_l57_57377

theorem vertical_asymptote_values (c : ℝ) :
  (∀ x : ℝ, (g(x) = (x^2 - 2 * x + c) / (x^2 - x - 6))) ∧
  ((∃ x1 : ℝ, (x1 = 3 ∧ (x^2 - 2 * x + c) = (x - 3) * h1(x)) ∧ 
  ¬(∃ x2 : ℝ, (x2 = -2 ∧ (x^2 - 2 * x + c) = (x + 2) * h2(x)))) ∨
  ((∃ x1 : ℝ, (x1 = -2 ∧ (x^2 - 2 * x + c) = (x + 2) * h1(x)) ∧ 
  ¬(∃ x2 : ℝ, (x2 = 3 ∧ (x^2 - 2 * x + c) = (x - 3) * h2(x)))))) ↔
  c = -3 ∨ c = -8 :=
sorry

end vertical_asymptote_values_l57_57377


namespace tiffany_found_bags_l57_57239

variable (bags_monday total_bags : ℕ)

-- Given Conditions:
-- Tiffany had 4 bags on Monday 
axiom bags_monday_eq : bags_monday = 4

-- She had a total of 6 bags after the next day
axiom total_bags_eq : total_bags = 6

-- Prove that Tiffany found 2 bags on the next day
theorem tiffany_found_bags : (total_bags - bags_monday) = 2 := by
  rw [bags_monday_eq, total_bags_eq]
  sorry

end tiffany_found_bags_l57_57239


namespace dana_jellybeans_l57_57016

noncomputable def jellybeans_in_dana_box (alex_capacity : ℝ) (mul_factor : ℝ) : ℝ :=
  let alex_volume := 1 * 1 * 1.5
  let dana_volume := mul_factor * mul_factor * (mul_factor * 1.5)
  let volume_ratio := dana_volume / alex_volume
  volume_ratio * alex_capacity

theorem dana_jellybeans
  (alex_capacity : ℝ := 150)
  (mul_factor : ℝ := 3) :
  jellybeans_in_dana_box alex_capacity mul_factor = 4050 :=
by
  rw [jellybeans_in_dana_box]
  simp
  sorry

end dana_jellybeans_l57_57016


namespace find_P_in_50th_group_l57_57717

theorem find_P_in_50th_group : 
  let last_number_50 := 2 * (∑ k in Finset.range 51, k)
  let first_number_50 := last_number_50 - 2 * 49
  let sum_50th_group := 25 * (first_number_50 + last_number_50)
  50 * 2501 = sum_50th_group :=
by
  let last_number_50 := 2 * (∑ k in Finset.range 51, k)
  let first_number_50 := last_number_50 - 2 * 49
  let sum_50th_group := 25 * (first_number_50 + last_number_50)
  show 50 * 2501 = sum_50th_group
  sorry

end find_P_in_50th_group_l57_57717


namespace three_digit_numbers_divisible_by_9_l57_57447

theorem three_digit_numbers_divisible_by_9 : 
  let smallest := 108
  let largest := 999
  let common_diff := 9
  -- Using the nth-term formula for an arithmetic sequence
  -- nth term: l = a + (n-1) * d
  -- For l = 999, a = 108, d = 9
  -- (999 = 108 + (n-1) * 9) -> (n-1) = 99 -> n = 100
  -- Hence, the number of such terms (3-digit numbers) in the sequence is 100.
  ∃ n, n = 100 ∧ (largest = smallest + (n-1) * common_diff)
by {
  let smallest := 108
  let largest := 999
  let common_diff := 9
  use 100
  sorry
}

end three_digit_numbers_divisible_by_9_l57_57447


namespace taller_tree_height_l57_57577

theorem taller_tree_height (H : ℝ) (h_difference : 24) (height_ratio : 2 = 3*(H - 24)/H) : H = 72 := 
by 
  sorry

end taller_tree_height_l57_57577


namespace pears_for_apples_l57_57804

-- Define the costs of apples, oranges, and pears.
variables {cost_apples cost_oranges cost_pears : ℕ}

-- Condition 1: Ten apples cost the same as five oranges
axiom apples_equiv_oranges : 10 * cost_apples = 5 * cost_oranges

-- Condition 2: Three oranges cost the same as four pears
axiom oranges_equiv_pears : 3 * cost_oranges = 4 * cost_pears

-- Theorem: Tyler can buy 13 pears for the price of 20 apples
theorem pears_for_apples : 20 * cost_apples = 13 * cost_pears :=
sorry

end pears_for_apples_l57_57804


namespace ellipse_equation_and_max_chord_length_l57_57394

-- Definition of the ellipse C based on given conditions
def ellipse_C := { p : ℝ × ℝ | (p.1^2 / 4) + p.2^2 = 1 }

-- Given points and slope for the line
def left_focus : ℝ × ℝ := (-Real.sqrt 3, 0)
def right_vertex : ℝ × ℝ := (2, 0)
def line_slope : ℝ := 1 / 2

-- Definition of the line l based on the slope
def line_l (b : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = (line_slope * p.1 + b)

theorem ellipse_equation_and_max_chord_length :
  (∀ p, p ∈ ellipse_C ↔ (p.1^2 / 4 + p.2^2 = 1)) ∧
  (∃ b, (b = 0) ∧ ∀ A B, (A ∈ ellipse_C) ∧ (B ∈ ellipse_C) ∧
  (line_l b A) ∧ (line_l b B) → (Real.sqrt (10 - 5 * b^2) = Real.sqrt 10) ∧
  (A.2 = line_slope * A.1) ∧ (B.2 = line_slope * B.1)) :=
by
  sorry

end ellipse_equation_and_max_chord_length_l57_57394


namespace total_number_of_legs_of_spiders_l57_57622

theorem total_number_of_legs_of_spiders:
  let human_legs := 2
  let first_spider_legs := 2 * (2 * human_legs)
  let second_spider_legs := 3 * first_spider_legs
  let third_spider_legs := second_spider_legs - 5
  in first_spider_legs + second_spider_legs + third_spider_legs = 51 :=
by
  let human_legs := 2
  let first_spider_legs := 2 * (2 * human_legs)
  let second_spider_legs := 3 * first_spider_legs
  let third_spider_legs := second_spider_legs - 5
  have h1 : first_spider_legs = 8 := by sorry
  have h2 : second_spider_legs = 24 := by sorry
  have h3 : third_spider_legs = 19 := by sorry
  calc
    first_spider_legs + second_spider_legs + third_spider_legs
        = 8 + 24 + 19 : by sorry
    ... = 51 : by sorry

end total_number_of_legs_of_spiders_l57_57622


namespace smallest_odd_number_divisible_by_3_l57_57227

theorem smallest_odd_number_divisible_by_3 : ∃ n : ℕ, n = 3 ∧ ∀ m : ℕ, (m % 2 = 1 ∧ m % 3 = 0) → m ≥ n := 
by
  sorry

end smallest_odd_number_divisible_by_3_l57_57227


namespace four_points_determine_four_planes_l57_57321

/-- Given four non-coplanar points, there exist 4 distinct planes each determined by any three of these points. -/
theorem four_points_determine_four_planes (a b c d : Point)
  (h_non_coplanar : ¬ Collinear {a, b, c, d}) : 
  ∃ p1 p2 p3 p4 : Plane, 
    (Plane.DeterminedBy p1 {a, b, c}) ∧ (Plane.DeterminedBy p2 {a, b, d}) ∧
    (Plane.DeterminedBy p3 {a, c, d}) ∧ (Plane.DeterminedBy p4 {b, c, d}) ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧
    p2 ≠ p3 ∧ p2 ≠ p4 ∧
    p3 ≠ p4 := sorry

end four_points_determine_four_planes_l57_57321


namespace color_of_217th_marble_l57_57309

-- Definitions of conditions
def total_marbles := 240
def pattern_length := 15
def red_marbles := 6
def blue_marbles := 5
def green_marbles := 4
def position := 217

-- Lean 4 statement
theorem color_of_217th_marble :
  (position % pattern_length ≤ red_marbles) :=
by sorry

end color_of_217th_marble_l57_57309


namespace rabbits_and_raccoons_l57_57683

variable (b_r t_r x : ℕ)

theorem rabbits_and_raccoons : 
  2 * b_r = x ∧ 3 * t_r = x ∧ b_r = t_r + 3 → x = 18 := 
by
  sorry

end rabbits_and_raccoons_l57_57683


namespace find_length_AE_l57_57837

theorem find_length_AE 
  (ABCD_rect : Rectangle ABCD)
  (perpendicular : AE ⊥ BD at E)
  (area_rect : area ABCD = 40)
  (tri_ratio : area (Triangle ABE) / area (Triangle DBA) = 1 / 5) :
  length AE = 8 := by
sorry

end find_length_AE_l57_57837


namespace not_every_real_covered_by_rational_intervals_l57_57856

theorem not_every_real_covered_by_rational_intervals:
  (∀ n : ℕ, ∃ r_n : ℚ, (n > 0) → r_n ∈ set.Ioo (r_n - 1/n) (r_n + 1/n))
    → (∃ x : ℝ, ∀ n : ℕ, x ∉ set.Ioo (r_n - 1/n) (r_n + 1/n)) :=
sorry

end not_every_real_covered_by_rational_intervals_l57_57856


namespace ellipse_equation_l57_57764

theorem ellipse_equation (b : Real) (c : Real)
  (h₁ : 0 < b ∧ b < 5) 
  (h₂ : 25 - b^2 = c^2)
  (h₃ : 5 + c = 2 * b) :
  ∃ (b : Real), (b^2 = 16) ∧ (∀ x y : Real, (x^2 / 25 + y^2 / b^2 = 1 ↔ x^2 / 25 + y^2 / 16 = 1)) := 
sorry

end ellipse_equation_l57_57764


namespace find_cos_minus_sin_l57_57379

theorem find_cos_minus_sin (α : ℝ) 
  (h1 : sin α * cos α = -12 / 25) 
  (h2 : α ∈ Ioo (-π / 2) 0) :
  cos α - sin α = 7 / 5 := 
sorry

end find_cos_minus_sin_l57_57379


namespace river_depth_in_mid_may_l57_57491

theorem river_depth_in_mid_may (D : ℝ) 
  (mid_june_depth : D + 10) 
  (mid_july_depth : 3 * (D + 10)) 
  (mid_july_depth_val : mid_july_depth = 45) :
  D = 5 := 
by
  sorry

end river_depth_in_mid_may_l57_57491


namespace petya_password_l57_57903

/-- Petya is creating a password for his smartphone. The password consists of 4 decimal digits.
Petya wants the password to not contain the digit 7, and it must have at least two identical digits.
We are to prove that the number of ways Petya can create such a password is 3537. -/
theorem petya_password : 
  let total_passwords := 9^4,
      all_distinct_passwords := (Finset.card (Finset.powersetLen 4 (Finset.range 9))).choose 4 * factorial 4 in
  total_passwords - all_distinct_passwords = 3537 :=
by
  sorry

end petya_password_l57_57903


namespace perimeter_ratio_l57_57933

theorem perimeter_ratio 
  (area_ratio : 49 / 64)
  (length_ratio : 7 / 8) : 91 / 72 :=
by
  sorry

end perimeter_ratio_l57_57933


namespace probability_interval_l57_57350

variable (P_A P_B q : ℚ)

axiom prob_A : P_A = 5/6
axiom prob_B : P_B = 3/4
axiom prob_A_and_B : q = P_A + P_B - 1

theorem probability_interval :
  7/12 ≤ q ∧ q ≤ 3/4 :=
by
  sorry

end probability_interval_l57_57350


namespace jungkook_initial_money_l57_57436

theorem jungkook_initial_money (notebook half_spent pencil_remaining : ℤ) (H1 : notebook = half_spent / 2)
(H2 : pencil_remaining = half_spent / 2 / 2) (H3 : pencil_remaining = 750) : half_spent = 3000 := 
by 
  -- Definitions and conditions
  have H4 : half_spent / 2 = 1500, from H3 ▸ rfl; sorry
  -- To be completed in proof
  
cancel sorry

end jungkook_initial_money_l57_57436


namespace proof_inequalities_l57_57152

theorem proof_inequalities (A B C D E : ℝ) (p q r s t : ℝ)
  (h1 : A < B) (h2 : B < C) (h3 : C < D) (h4 : D < E)
  (h5 : p = B - A) (h6 : q = C - A) (h7 : r = D - A)
  (h8 : s = E - B) (h9 : t = E - D)
  (ineq1 : p + 2 * s > r + t)
  (ineq2 : r + t > p)
  (ineq3 : r + t > s) :
  (p < r / 2) ∧ (s < t + p / 2) :=
by 
  sorry

end proof_inequalities_l57_57152


namespace sum_of_good_integers_eq_172_l57_57014

def is_good (n : ℕ) : Prop := 
  ∃ (phi : ℕ) (tau : ℕ), phi = n.totient ∧ tau = n.divisor_count ∧ (phi + 4 * tau = n)

def find_sum_of_good_integers (N : ℕ) : ℕ := 
  finset.sum (finset.filter is_good (finset.range (N + 1))) id

theorem sum_of_good_integers_eq_172 : find_sum_of_good_integers 256 = 172 :=
by sorry

end sum_of_good_integers_eq_172_l57_57014


namespace Lilly_fish_count_l57_57538

theorem Lilly_fish_count (Rosy_fish : ℕ) (Total_fish : ℕ) (h1 : Rosy_fish = 12) (h2 : Total_fish = 22) :
  ∃ (Lilly_fish : ℕ), Lilly_fish = 10 :=
by
  use 10
  rw [h1, h2]
  sorry

end Lilly_fish_count_l57_57538


namespace total_weight_of_dumbbells_l57_57550

theorem total_weight_of_dumbbells : 
  let initial_dumbbells := 4
  let additional_dumbbells := 2
  let weight_per_dumbbell := 20 in
  (initial_dumbbells + additional_dumbbells) * weight_per_dumbbell = 120 := 
by
  -- conditions and definitions
  let initial_dumbbells := 4
  let additional_dumbbells := 2
  let weight_per_dumbbell := 20
  -- calculation
  calc
  (initial_dumbbells + additional_dumbbells) * weight_per_dumbbell 
  = (4 + 2) * 20 : by rw [initial_dumbbells, additional_dumbbells, weight_per_dumbbell]
  ... = 6 * 20 : by norm_num
  ... = 120 : by norm_num

end total_weight_of_dumbbells_l57_57550


namespace triangle_perimeter_l57_57507

theorem triangle_perimeter (A B C X Y Z W : Point)
  (h_angle_C : angle A C B = 90)
  (h_AB : dist A B = 10)
  (h_squares : is_square A B X Y ∧ is_square B C W Z)
  (h_cyclic : cyclic_quadrilateral X Y Z W)
  : perimeter A B C = 10 + 10 * sqrt 2 := 
sorry

end triangle_perimeter_l57_57507


namespace sum_equality_a_b_c_sum_result_l57_57980

def T (n : ℕ) : ℚ :=
  ∑ k in finset.range n, (-1)^(k+1) * (k^3 + k^2 + k + 1) / k.factorial

theorem sum_equality :
  T 50 = 5303 / 50! - 1 :=
sorry

theorem a_b_c_sum_result :
  let a := 5303
  let b := 50
  let c := 1
  a + b + c = 5354 :=
by
  have h : T 50 = 5303 / 50! - 1 := sorry
  sorry

end sum_equality_a_b_c_sum_result_l57_57980


namespace edges_of_remaining_solid_after_removal_l57_57311

theorem edges_of_remaining_solid_after_removal (a b c : ℕ) (h₁ : a = 4) (h₂ : b = 2) (h₃ : c = 8) :
  let initial_edges := 12 in
  let new_edges := initial_edges * (b + 1) in
  new_edges = 36 :=
by
  sorry

end edges_of_remaining_solid_after_removal_l57_57311


namespace original_average_age_l57_57934

-- Definitions based on conditions
def original_strength : ℕ := 12
def new_student_count : ℕ := 12
def new_student_average_age : ℕ := 32
def age_decrease : ℕ := 4
def total_student_count : ℕ := original_strength + new_student_count
def combined_total_age (A : ℕ) : ℕ := original_strength * A + new_student_count * new_student_average_age
def new_average_age (A : ℕ) : ℕ := A - age_decrease

-- Statement of the problem
theorem original_average_age (A : ℕ) (h : combined_total_age A / total_student_count = new_average_age A) : A = 40 := 
by 
  sorry

end original_average_age_l57_57934


namespace combined_savings_zero_l57_57674

theorem combined_savings_zero
  (price : ℕ)
  (offer : ℕ → ℕ → ℕ → ℕ)
  (dave_needs : ℕ)
  (doug_needs : ℕ) :
  price = 100 →
  offer 3 1 0 = 0 →
  dave_needs = 9 →
  doug_needs = 6 →
  (let dave_separate_cost := dave_needs * price,
       dave_discounted_cost := ((dave_needs // 3) * 3 + (dave_needs % 3)) * price,
       dave_savings := dave_separate_cost - dave_discounted_cost,
       doug_separate_cost := doug_needs * price,
       doug_discounted_cost := ((doug_needs // 3) * 3 + (doug_needs % 3)) * price,
       doug_savings := doug_separate_cost - doug_discounted_cost,
       combined_separate_cost := (dave_needs + doug_needs) * price,
       combined_discounted_cost := (((dave_needs + doug_needs) // 3) * 3 + ((dave_needs + doug_needs) % 3)) * price,
       combined_savings := combined_separate_cost - combined_discounted_cost)
    → combined_savings - (dave_savings + doug_savings) = 0 :=
by
  intros
  sorry

end combined_savings_zero_l57_57674


namespace suitable_for_census_l57_57266

-- Define the conditions as propositions
def OptionA : Prop := "Surveying the lifespan of a batch of light bulbs"
def OptionB : Prop := "Surveying the water quality in the Gan River Basin"
def OptionC : Prop := "Surveying the viewership of the 'Legendary Stories' program on Jiangxi TV"
def OptionD : Prop := "Surveying the heights of all classmates"

-- The statement to be proved
theorem suitable_for_census : OptionD = "Surveying the heights of all classmates" := 
by 
  sorry

end suitable_for_census_l57_57266


namespace find_number_l57_57643

theorem find_number : ∃ n : ℝ, 50 + (5 * n) / (180 / 3) = 51 ∧ n = 12 := 
by
  use 12
  sorry

end find_number_l57_57643


namespace find_range_of_m_l57_57210

theorem find_range_of_m (ω : ℝ) (φ : ℝ) (m : ℝ) 
  (h1 : 0 < φ ∧ φ < π)
  (h2 : ω = 3)
  (h3 : φ = π / 3)
  (h4 : ∀ x, (π / 6) ≤ x ∧ x ≤ m → -1 ≤ cos (ω * x + φ) ∧ cos (ω * x + φ) ≤ -sqrt(3)/2) :
  (2 * π / 9) ≤ m ∧ m ≤ (5 * π / 18) :=
begin
  sorry,
end

end find_range_of_m_l57_57210


namespace part_a_part_b_l57_57307
noncomputable theory

-- Definitions given in the problem
structure Ring (α : Type*) [metric_space α] :=
  (center : α)
  (inner_radius outer_radius : ℝ)
  (width_eq_one : outer_radius - inner_radius = 1)

-- Proving the impossibility of uncountable disjoint rings with width 1 that cannot be separated.
theorem part_a : ¬ ∃ (S : set (Ring ℝ)), S.uncountable ∧ ∀ (A B ∈ S), A ≠ B → ¬disjoint A B := sorry

-- Definitions for rings with zero width
structure RingZeroWidth (α : Type*) [metric_space α] :=
  (center : α)
  (radius : ℝ)

-- Proving the possibility of uncountable disjoint rings with width 0 that cannot be separated.
theorem part_b : ∃ (S : set (RingZeroWidth ℝ)), S.uncountable ∧ ∀ (A B ∈ S), A ≠ B → ¬disjoint A B := sorry

end part_a_part_b_l57_57307


namespace hyperbola_problem_l57_57780

-- Given the conditions of the hyperbola
def hyperbola (x y: ℝ) (b: ℝ) : Prop := (x^2) / 4 - (y^2) / (b^2) = 1 ∧ b > 0

-- Asymptote condition
def asymptote (b: ℝ) : Prop := (b / 2) = (Real.sqrt 6 / 2)

-- Foci, point P condition
def foci_and_point (PF1 PF2: ℝ) : Prop := PF1 / PF2 = 3 / 1 ∧ PF1 - PF2 = 4

-- Math proof problem
theorem hyperbola_problem (b PF1 PF2: ℝ) (P: ℝ × ℝ) :
  hyperbola P.1 P.2 b ∧ asymptote b ∧ foci_and_point PF1 PF2 →
  |PF1 + PF2| = 2 * Real.sqrt 10 :=
by
  sorry

end hyperbola_problem_l57_57780


namespace cube_root_neg_003375_l57_57937

theorem cube_root_neg_003375 :
  ∃ x : ℝ, x ^ 3 = -0.003375 ∧ x = -0.15 :=
begin
  use -0.15,
  split,
  { norm_num, },
  { refl, },
end

end cube_root_neg_003375_l57_57937


namespace percentage_calculation_l57_57652

theorem percentage_calculation :
  ∀ (P : ℝ),
  (0.3 * 0.5 * 4400 = 99) →
  (P * 4400 = 99) →
  P = 0.0225 :=
by
  intros P condition1 condition2
  -- From the given conditions, it follows directly
  sorry

end percentage_calculation_l57_57652


namespace paul_tips_l57_57898

theorem paul_tips (P : ℕ) (h1 : P + 16 = 30) : P = 14 :=
by
  sorry

end paul_tips_l57_57898


namespace simplify_and_find_value_l57_57179

noncomputable def sqrt3 : ℝ := Real.sqrt 3
noncomputable def sqrt2 : ℝ := Real.sqrt 2

def a : ℝ := sqrt3 - sqrt2
def b : ℝ := sqrt3 + sqrt2

theorem simplify_and_find_value :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) + 2 * a * (b - a) = 6 :=
by
  sorry

end simplify_and_find_value_l57_57179


namespace impossible_coins_l57_57160

theorem impossible_coins (p1 p2 : ℝ) : 
  ¬ ((1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end impossible_coins_l57_57160


namespace no_such_coins_l57_57164

theorem no_such_coins (p1 p2 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1)
  (cond1 : (1 - p1) * (1 - p2) = p1 * p2)
  (cond2 : p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :
  false :=
  sorry

end no_such_coins_l57_57164


namespace right_triangle_area_l57_57618

theorem right_triangle_area (a c : ℝ) (h₁ : a = 18) (h₂ : c = 30) (h₃ : ∃ b : ℝ, b ^ 2 = c ^ 2 - a ^ 2) :
  ∃ area : ℝ, area = 0.5 * a * sqrt (c^2 - a^2) ∧ area = 216 :=
by
  sorry

end right_triangle_area_l57_57618


namespace math_problem_l57_57757

noncomputable def parabola (p : ℝ) := {x : ℝ × ℝ // x.2 ^ 2 = 2 * p * x.1}
def line (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def focus (p : ℝ) : ℝ × ℝ := (p, 0)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem math_problem (p : ℝ)
  (h_line_parabola_intersect : ∃ A B : ℝ × ℝ, line A.1 A.2 ∧ line B.1 B.2 ∧ parabola p A ∧ parabola p B ∧ dist A B = 4 * real.sqrt 15)
  (h_focus : ∃ F : ℝ × ℝ, F = focus p)
  (h_points_MN : ∃ M N : ℝ × ℝ, parabola p M ∧ parabola p N ∧ dot_product (M - (p, 0)) (N - (p, 0)) = 0) :
  p = 2 ∧ (∃ M N : ℝ × ℝ, parabola 2 M ∧ parabola 2 N ∧ dot_product (M - (2, 0)) (N - (2, 0)) = 0) ∧
  (∀ M N : ℝ × ℝ, parabola 2 M ∧ parabola 2 N ∧ dot_product (M - (2, 0)) (N - (2, 0)) = 0 → 
  (1/2) * |M.1 * N.2 - M.2 * N.1| = 12 - 8 * real.sqrt 2) := sorry

end math_problem_l57_57757


namespace larissa_points_product_l57_57836

def total_points_first_10 := 5 + 6 + 4 + 7 + 5 + 6 + 2 + 3 + 4 + 9
def points_less_than_15 (x : ℕ) : Prop := x < 15
def avg_points_is_integer (total_points : ℕ) (games : ℕ) : ℕ := total_points / games

theorem larissa_points_product :
  ∃ x11 x12 : ℕ,
    points_less_than_15 x11 ∧
    points_less_than_15 x12 ∧
    avg_points_is_integer (total_points_first_10 + x11) 11 ∧
    avg_points_is_integer (total_points_first_10 + x11 + x12) 12 ∧
    (x11 * x12 = 20) :=
sorry

end larissa_points_product_l57_57836


namespace inequality_base_case_l57_57253

theorem inequality_base_case :
  ∃ n₀ ∈ {1, 2, 3, 4}, 
  (1 + (1 : ℝ) / real.sqrt 2 + (1 : ℝ) / real.sqrt 3 + ... + (1 : ℝ) / real.sqrt n₀ > real.sqrt n₀) ∧
  ∀ n < n₀, ¬(1 + (1 : ℝ) / real.sqrt 2 + (1 : ℝ) / real.sqrt 3 + ... + (1 : ℝ) / real.sqrt n > real.sqrt n) := 
begin
  sorry
end

end inequality_base_case_l57_57253


namespace subsets_with_5_and_6_l57_57795

-- Define the main problem
theorem subsets_with_5_and_6 (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6}) :
  (s.filter (λ x, x = 5 ∨ x = 6)).card = 16 :=
sorry

end subsets_with_5_and_6_l57_57795


namespace pizza_topping_cost_l57_57176

/- 
   Given:
   1. Ruby ordered 3 pizzas.
   2. Each pizza costs $10.00.
   3. The total number of toppings were 4.
   4. Ruby added a $5.00 tip to the order.
   5. The total cost of the order, including tip, was $39.00.

   Prove: The cost per topping is $1.00.
-/
theorem pizza_topping_cost (cost_per_pizza : ℝ) (total_pizzas : ℕ) (tip : ℝ) (total_cost : ℝ) 
    (total_toppings : ℕ) (x : ℝ) : 
    cost_per_pizza = 10 → total_pizzas = 3 → tip = 5 → total_cost = 39 → total_toppings = 4 → 
    total_cost = cost_per_pizza * total_pizzas + x * total_toppings + tip →
    x = 1 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end pizza_topping_cost_l57_57176


namespace number_of_pairs_sum_68_l57_57011

theorem number_of_pairs_sum_68 :
  {ab : ℕ × ℕ // 1 ≤ ab.1 ∧ ab.2 ≤ 64 ∧ ab.1 < ab.2 ∧ ab.1 + ab.2 = 68}.card = 30 := 
by
  sorry

end number_of_pairs_sum_68_l57_57011


namespace discriminants_equal_l57_57283

-- Definitions for monic quadratic polynomials with distinct roots
variables {α : Type*} [Field α]

-- Polynomial P has distinct roots a1 and a2 and is monic
def P (x : α) (a1 a2 : α) : α := (x - a1) * (x - a2)

-- Polynomial Q has distinct roots b1 and b2 and is monic
def Q (x : α) (b1 b2 : α) : α := (x - b1) * (x - b2)

-- Given condition on the sum of values
axiom sum_condition {a1 a2 b1 b2 : α} :
  P b1 a1 a2 + P b2 a1 a2 = Q a1 b1 b2 + Q a2 b1 b2

-- Discriminant of a monic quadratic polynomial
def discriminant (a b : α) : α := (a - b) ^ 2

-- The goal of the proof: discriminants are equal
theorem discriminants_equal {a1 a2 b1 b2 : α} (h1 : a1 ≠ a2) (h2 : b1 ≠ b2) : 
  discriminant a1 a2 = discriminant b1 b2 :=
begin
  -- Proof goes here
  sorry
end

end discriminants_equal_l57_57283


namespace right_triangle_hypotenuse_length_l57_57838

theorem right_triangle_hypotenuse_length
  (a b : ℝ)
  (ha : a = 12)
  (hb : b = 16) :
  c = 20 :=
by
  -- Placeholder for the proof
  sorry

end right_triangle_hypotenuse_length_l57_57838


namespace arithmetic_mean_of_fractions_l57_57893

theorem arithmetic_mean_of_fractions :
  let a := 7 / 9
  let b := 5 / 6
  let c := 8 / 9
  2 * b = a + c :=
by
  sorry

end arithmetic_mean_of_fractions_l57_57893


namespace triangle_area_angle_C_l57_57510

noncomputable def area_of_triangle (a b c B : ℝ) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem triangle_area :
    (B = Real.pi * 5 / 6) →
    (a = Real.sqrt 3 * c) →
    (b = 2 * Real.sqrt 7) →
    c = 2 →
    area_of_triangle a b c B = Real.sqrt 3 :=
  by
  sorry

theorem angle_C :
    (A B C : ℝ) →
    (B = 150 * Real.pi / 180) →
    (sin A + Real.sqrt 3 * sin C = Real.sqrt 2 / 2) →
    (A + B + C = Real.pi) →
    C = 15 * Real.pi / 180 :=
  by
  sorry

end triangle_area_angle_C_l57_57510


namespace part_a_part_b_l57_57281

def neighboring (cell1 cell2 : ℕ × ℕ) : Prop :=
  (abs (cell1.1 - cell2.1) = 1 ∧ cell1.2 = cell2.2) ∨ (abs (cell1.2 - cell2.2) = 1 ∧ cell1.1 = cell2.1)

def valid_arrangement (marbles : Finset (ℕ × ℕ)) : Prop :=
  ∀ (m1 m2 ∈ marbles), m1 ≠ m2 → ¬ neighboring m1 m2

theorem part_a : ∃ (marbles : Finset (ℕ × ℕ)), marbles.card = 2024 ∧ valid_arrangement marbles ∧
  (∀ marble ∈ marbles, ∀ cell, neighboring marble cell → cell ∈ marbles) := 
  sorry

theorem part_b : ∀ (marbles : Finset (ℕ × ℕ)), marbles.card = 2023 → valid_arrangement marbles → 
  ∃ marble ∈ marbles, ∃ cell, neighboring marble cell ∧ cell ∉ marbles ∧ 
  valid_arrangement ((marbles.erase marble).insert cell) := 
  sorry

end part_a_part_b_l57_57281


namespace flour_more_than_sugar_l57_57145

theorem flour_more_than_sugar (f_s f_t f_i : ℕ) (hf_s : f_s = 3) (hf_t : f_t = 10) (hf_i : f_i = 2) : (f_t - f_i) - f_s = 5 :=
by
  rw [hf_t, hf_i, hf_s]
  apply rfl

end flour_more_than_sugar_l57_57145


namespace sum_of_min_equals_sum_of_powers_l57_57818

theorem sum_of_min_equals_sum_of_powers (n k : ℕ) :
  ∑ s in (Finset.pi (Finset.range k).to_finset (λ _, Finset.range n)), s.to_list.min = ∑ m in Finset.range (n+1), m^k := 
sorry

end sum_of_min_equals_sum_of_powers_l57_57818


namespace area_of_circle_l57_57366

-- Define the given conditions
def pi_approx : ℝ := 3
def radius : ℝ := 0.6

-- Prove that the area is 1.08 given the conditions
theorem area_of_circle : π = pi_approx → radius = 0.6 → 
  (pi_approx * radius^2 = 1.08) :=
by
  intros hπ hr
  sorry

end area_of_circle_l57_57366


namespace even_numbers_count_l57_57070

theorem even_numbers_count (a b : ℕ) (h1 : 150 < a) (h2 : a % 2 = 0) (h3 : b < 350) (h4 : b % 2 = 0) (h5 : 150 < b) (h6 : a < 350) (h7 : 154 ≤ b) (h8 : a ≤ 152) :
  ∃ n : ℕ, ∀ k : ℕ, k = 99 ↔ 2 * k + 150 = b - a + 2 :=
by
  sorry

end even_numbers_count_l57_57070


namespace find_value_4a_2b_c_l57_57224

-- Given conditions
noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 - 3*x + 4

def A := (-4 : ℝ, 0 : ℝ)
def B := (1 : ℝ, 0 : ℝ)
def C := (0 : ℝ, 4 : ℝ)
def OA := 4 : ℝ
def OB := 1 : ℝ
def parabola (x : ℝ) := -x^2 - 3*x + 4

-- Statement to prove
theorem find_value_4a_2b_c :
  (4 * (-1) - 2 * (-3) + 4 = 6) :=
by
  sorry

end find_value_4a_2b_c_l57_57224


namespace find_x_for_h_eq_x_l57_57874

theorem find_x_for_h_eq_x :
  (∀ x : ℝ, h (5 * x - 2) = 3 * x + 5) →
  h (31 / 2) = 31 / 2 :=
by
  sorry

end find_x_for_h_eq_x_l57_57874


namespace sqrt_sum_inequality_l57_57535

variable {n : ℕ}
variable (a : Fin n → ℝ)

-- Ensure positive numbers
def all_positive : Prop := ∀ i, 0 < a i

-- Lean statement for the problem
theorem sqrt_sum_inequality (hpos : all_positive a) :
    (∑ i in Finset.range n, Real.sqrt (∑ j in Finset.Icc i (n-1), a j)) ≥ Real.sqrt (∑ k in Finset.range n, k^2 * a k) :=
sorry

end sqrt_sum_inequality_l57_57535


namespace dot_product_of_unit_vectors_l57_57428

variables {V : Type*} [inner_product_space ℝ V]

-- Define the vectors a, b, and c
variables (a b c : V)

-- Define the conditions: a, b, and c are unit vectors
def is_unit_vector (v : V) := ‖v‖ = 1

-- Define the condition that a + b + 2c = 0
def vector_sum_zero (a b c : V) := a + b + 2 • c = 0

-- The theorem to prove
theorem dot_product_of_unit_vectors
  (ha : is_unit_vector a)
  (hb : is_unit_vector b)
  (hc : is_unit_vector c)
  (h_sum : vector_sum_zero a b c) :
  ⟪a, b⟫ = 1 := 
sorry

end dot_product_of_unit_vectors_l57_57428


namespace problem_l57_57719

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin (π/4 + x))^2 + Real.sqrt 3 * (Real.sin x * Real.cos (2 * x))

theorem problem (x : ℝ) (h₁ : x ∈ Set.Icc (π/4) (π/2)) :
  (f (5 * π / 12) = 3) ∧ (∀ m, (∀ x ∈ Set.Icc (π/4) (π/2), f x = m) → 1 < m ∧ m < 4) :=
sorry

end problem_l57_57719


namespace unique_solutions_angles_l57_57001

noncomputable def findAngles : Set ℝ :=
  {α | 0 ≤ α ∧ α ≤ 360 ∧ 
       (∀ x y, 
        (y ^ 3 ≥ 3 * (x ^ 2) * y) ∧ 
        ((x - Real.cos α) ^ 2 + (y - Real.sin α) ^ 2 = (2 - Real.sqrt 3) / 4))}

theorem unique_solutions_angles :
  findAngles = {15, 45, 135, 165, 255, 285} :=
begin
  sorry
end

end unique_solutions_angles_l57_57001


namespace positive_integers_satisfying_condition_l57_57561

theorem positive_integers_satisfying_condition :
  ∃! n : ℕ, 0 < n ∧ 24 - 6 * n > 12 :=
by
  sorry

end positive_integers_satisfying_condition_l57_57561


namespace impossible_coins_l57_57159

theorem impossible_coins (p_1 p_2 : ℝ) 
  (h1 : (1 - p_1) * (1 - p_2) = p_1 * p_2)
  (h2 : p_1 * (1 - p_2) + p_2 * (1 - p_1) = p_1 * p_2) : False := 
sorry

end impossible_coins_l57_57159


namespace count_3_digit_numbers_divisible_by_9_l57_57456

theorem count_3_digit_numbers_divisible_by_9 : 
  (finset.filter (λ x : ℕ, x % 9 = 0) (finset.Icc 100 999)).card = 100 := 
sorry

end count_3_digit_numbers_divisible_by_9_l57_57456


namespace problem1_problem2_l57_57775

-- Definition of the function f
def f (x a : ℝ) : ℝ := x^2 - a * x

-- First proof subproblem statement
theorem problem1 (a : ℝ) (h : a = 2) : 
  {x : ℝ | f x a ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 3} := sorry

-- Second proof subproblem statement
theorem problem2 (a : ℝ) :
  (∀ x ∈ Ici (1 : ℝ), f x a ≥ -x^2 - 2) ↔ a ≤ 4 := sorry

end problem1_problem2_l57_57775


namespace count_3_digit_numbers_divisible_by_9_l57_57439

theorem count_3_digit_numbers_divisible_by_9 : 
  let count := (range (integer_divisible_in_range 9 100 999)).length
  count = 100 := 
by
  sorry

noncomputable def integer_divisible_in_range (k m n : ℕ) : List ℕ :=
  let start := m / k + (if (m % k = 0) then 0 else 1)
  let end_ := n / k
  List.range (end_ - start + 1) |>.map (λ i => (start + i) * k)

noncomputable def range (xs : List ℕ) := xs

end count_3_digit_numbers_divisible_by_9_l57_57439


namespace three_digit_numbers_divisible_by_9_l57_57446

theorem three_digit_numbers_divisible_by_9 : 
  let smallest := 108
  let largest := 999
  let common_diff := 9
  -- Using the nth-term formula for an arithmetic sequence
  -- nth term: l = a + (n-1) * d
  -- For l = 999, a = 108, d = 9
  -- (999 = 108 + (n-1) * 9) -> (n-1) = 99 -> n = 100
  -- Hence, the number of such terms (3-digit numbers) in the sequence is 100.
  ∃ n, n = 100 ∧ (largest = smallest + (n-1) * common_diff)
by {
  let smallest := 108
  let largest := 999
  let common_diff := 9
  use 100
  sorry
}

end three_digit_numbers_divisible_by_9_l57_57446


namespace solution_set_for_xf_l57_57203

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x ∈ set.univ
axiom f_at_neg2 : f (-2) = 2
axiom f_deriv_ineq : ∀ x : ℝ, x * (deriv f x) > - (f x)

theorem solution_set_for_xf : {x : ℝ | x * f x < -4} = set.Iio (-2) :=
by
  sorry

end solution_set_for_xf_l57_57203


namespace max_third_side_of_triangle_l57_57248

theorem max_third_side_of_triangle (a b : ℕ) (h₁ : a = 7) (h₂ : b = 11) : 
  ∃ c : ℕ, c < a + b ∧ c = 17 :=
by 
  sorry

end max_third_side_of_triangle_l57_57248


namespace choose_two_out_of_three_l57_57267

theorem choose_two_out_of_three : Nat.choose 3 2 = 3 := by
  sorry

end choose_two_out_of_three_l57_57267


namespace length_of_GH_l57_57842

theorem length_of_GH (AB CD GH : ℤ) (h_parallel : AB = 240 ∧ CD = 160 ∧ (AB + CD) = GH*2) : GH = 320 / 3 :=
by sorry

end length_of_GH_l57_57842


namespace arrange_prob_correct_l57_57945

theorem arrange_prob_correct :
  let chars : List Char := ['上', '医', '医', '国']
  in let total_permutations := Nat.factorial 4 / Nat.factorial 2
  in 1 / total_permutations = 1 / 12 :=
by
  sorry

end arrange_prob_correct_l57_57945


namespace hyperbola_equation_line_through_hyperbola_intersections_l57_57779

-- Define the hyperbola parameters and prove the given equation.
theorem hyperbola_equation :
  ∃ a b : ℝ, a = 1 ∧ b = sqrt (2^2 - a^2) ∧ a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2 - (y^2 / b^2) = 1 :=
sorry

-- Define the line passing through M(1,3) and intersecting the hyperbola at A and B
theorem line_through_hyperbola_intersections (M : (ℝ × ℝ)) (hM : M = (1, 3)) :
  ∃ l : ℝ → ℝ, (∀ x y, (x, y) ∈ (λ x, x + 2) →
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁^2 - (y₁^2 / 3) = 1 ∧ x₂^2 - (y₂^2 / 3) = 1 ∧
  x₁ + x₂ = 2 ∧ y₁ + y₂ = 6 ∧ (x₁ - x₂) / (y₁ - y₂) = 1) :=
sorry

end hyperbola_equation_line_through_hyperbola_intersections_l57_57779


namespace find_y_l57_57871

def oslash (a b : ℝ) : ℝ := (sqrt (3 * a + b))^3

theorem find_y (y : ℝ) (h : oslash 9 y = 64) : y = -11 :=
by
  sorry

end find_y_l57_57871


namespace ratio_of_ages_l57_57913

theorem ratio_of_ages
  (Sandy_age : ℕ)
  (Molly_age : ℕ)
  (h1 : Sandy_age = 49)
  (h2 : Molly_age = Sandy_age + 14) : (Sandy_age : ℚ) / Molly_age = 7 / 9 :=
by
  -- To complete the proof.
  sorry

end ratio_of_ages_l57_57913


namespace pears_for_apples_l57_57805

-- Define the costs of apples, oranges, and pears.
variables {cost_apples cost_oranges cost_pears : ℕ}

-- Condition 1: Ten apples cost the same as five oranges
axiom apples_equiv_oranges : 10 * cost_apples = 5 * cost_oranges

-- Condition 2: Three oranges cost the same as four pears
axiom oranges_equiv_pears : 3 * cost_oranges = 4 * cost_pears

-- Theorem: Tyler can buy 13 pears for the price of 20 apples
theorem pears_for_apples : 20 * cost_apples = 13 * cost_pears :=
sorry

end pears_for_apples_l57_57805


namespace rectangle_hexagons_square_l57_57644

theorem rectangle_hexagons_square (length short_side : ℕ) (y : ℕ) (h1 : length = 15) (h2 : short_side = 10) (hexagons_formed_square : 2 * (length * short_side) = (5 * real.sqrt (6))^2) :
  y = 3 * short_side → y = 30 :=
begin
  intros,
  rw h2,
  exact rfl,
end

end rectangle_hexagons_square_l57_57644


namespace find_larger_number_l57_57953

variables (x y : ℝ)

def sum_cond : Prop := x + y = 17
def diff_cond : Prop := x - y = 7

theorem find_larger_number (h1 : sum_cond x y) (h2 : diff_cond x y) : x = 12 :=
sorry

end find_larger_number_l57_57953


namespace sum_of_digits_l57_57620

theorem sum_of_digits (a b c : ℕ) (h1 : a = 2) (h2 : b = 2008) (h3 : c = 2009) :
    let n := (a^b * (a * 5)^(c - b) * 7);
    let d := (10^b * 5);
    let dec_sum := 5;
    ∑ (digit in (d.digits 10)), digit = 5 :=
by
  -- Definitions and initial simplifications
  sorry

end sum_of_digits_l57_57620


namespace angle_is_60_degrees_l57_57045

variables (a b : EuclideanSpace ℝ (Fin 3))

def magnitude (v : EuclideanSpace ℝ (Fin 3)) : ℝ := 
  Real.sqrt (v.inner v)

axiom magnitude_a : magnitude a = 2
axiom magnitude_b : magnitude b = 1
axiom inner_equation : InnerProductSpace.IsROrC.inner a (a - 2 • b) = (2:ℝ)

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ := 
  Real.acos ((InnerProductSpace.IsROrC.inner a b) / ((magnitude a) * (magnitude b)))

theorem angle_is_60_degrees : 
  angle_between_vectors a b = Real.pi / 3 :=
by
  sorry

end angle_is_60_degrees_l57_57045


namespace impossible_to_arrange_distinct_integers_in_grid_l57_57117

theorem impossible_to_arrange_distinct_integers_in_grid :
  ¬ ∃ (f : Fin 25 × Fin 41 → ℤ),
    (∀ i j, abs (f i - f j) ≤ 16 → (i ≠ j) → (i.1 = j.1 ∨ i.2 = j.2)) ∧
    (∃ i j, i ≠ j ∧ f i = f j) := 
sorry

end impossible_to_arrange_distinct_integers_in_grid_l57_57117


namespace average_speed_l57_57668

theorem average_speed (x y : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y)
  (total_time : x / 4 + y / 3 + y / 6 + x / 4 = 5) :
  (2 * (x + y)) / 5 = 4 :=
by
  sorry

end average_speed_l57_57668


namespace solve_for_x_l57_57989

theorem solve_for_x : 
  ∀ x : ℝ, (4^5 + 4^5 + 4^5 = 2^x) → x = Real.log 2 3 + 10 := 
by 
  intro x h,
  sorry

end solve_for_x_l57_57989


namespace find_highest_score_l57_57198

theorem find_highest_score (H L : ℚ)
    (sum_of_5_scores : 5 * 81.6 = 408)
    (sum_of_middle_3_scores : 88 + 84 + 76 = 248)
    (average_relation : (248 + H) / 4 - (248 + L) / 4 = 6)
    (total_sum_relation : 248 + H + L = 408) :
    H = 92 :=
begin
    sorry
end

end find_highest_score_l57_57198


namespace find_angle_B_and_max_area_l57_57488

/-- In an acute triangle ABC, with sides opposite angles A, B, C denoted as a, b, c respectively,
    and given vectors m and n such that m ⊥ n, and with b = √3:

    (I) Proves that B == π/3 --
    (II) Proves that the maximum area of the triangle is 3√3/4 and a, b, c = √3 -/

variable (A B C a b c : ℝ)
variable (m n : ℝ × ℝ)
variable (A B C : ℝ)
variable (h₁ : b = Real.sqrt 3)
variable (h₂ : m = (2, c))
variable (h₃ : n = (b/2 * Real.cos C - Real.sin A, Real.cos B))
variable (h₄ : m.1 * n.1 + m.2 * n.2 = 0)

theorem find_angle_B_and_max_area :
  B = π / 3 ∧ 
  ((A = π / 3) ∧ (a = Real.sqrt 3) ∧ (c = Real.sqrt 3) ∧
  (a * c * Real.sin B = 3 * Real.sqrt 3 / 4)) :=
by
  sorry

end find_angle_B_and_max_area_l57_57488


namespace square_area_l57_57830

theorem square_area (A B C D P Q R : ℝ) (s : ℝ)
    (h1 : A ≠ B) (h2 : A ≠ D) (h3 : s > 0)
    (h4 : dist A B = s) (h5 : dist A D = s) 
    (h6 : dist B R = 3) (h7 : dist P R = 4)
    (h8 : A ≠ D) (h9 : A ≠ P) (h10 : B ≠ P)
    (h11 : C ≠ Q) (h12 : C ≠ R) 
    (h13 : dist B P + dist P R = dist B R)
    (h14 : ∠BPR = 90) : s^2 = 49 :=
  sorry

end square_area_l57_57830


namespace planar_figure_partitions_l57_57906

theorem planar_figure_partitions (F : Type) (diam : F → ℝ) (h : ∀ f, diam f ≤ 1) :
  (∃ P1 P2 P3 : Set F, ∀ i ∈ {P1, P2, P3}, diameter i ≤ (sqrt 3) / 2) ∧
  (∃ P1 P2 P3 P4 : Set F, ∀ i ∈ {P1, P2, P3, P4}, diameter i ≤ (sqrt 2) / 2) ∧
  (∃ P1 P2 P3 P4 P5 P6 P7 : Set F, ∀ i ∈ {P1, P2, P3, P4, P5, P6, P7}, diameter i ≤ 1 / 2) :=
by
  sorry

end planar_figure_partitions_l57_57906


namespace maximum_automobiles_on_ferry_l57_57998

-- Define the conditions
def ferry_capacity_tons : ℕ := 50
def automobile_min_weight : ℕ := 1600
def automobile_max_weight : ℕ := 3200

-- Define the conversion factor from tons to pounds
def ton_to_pound : ℕ := 2000

-- Define the converted ferry capacity in pounds
def ferry_capacity_pounds := ferry_capacity_tons * ton_to_pound

-- Proof statement
theorem maximum_automobiles_on_ferry : 
  ferry_capacity_pounds / automobile_min_weight = 62 :=
by
  -- Given: ferry capacity is 50 tons and 1 ton = 2000 pounds
  -- Therefore, ferry capacity in pounds is 50 * 2000 = 100000 pounds
  -- The weight of the lightest automobile is 1600 pounds
  -- Maximum number of automobiles = 100000 / 1600 = 62.5
  -- Rounding down to the nearest whole number gives 62
  sorry  -- Proof steps would be filled here

end maximum_automobiles_on_ferry_l57_57998


namespace rationalize_denominator_sum_equals_49_l57_57173

open Real

noncomputable def A : ℚ := -1
noncomputable def B : ℚ := -3
noncomputable def C : ℚ := 1
noncomputable def D : ℚ := 2
noncomputable def E : ℚ := 33
noncomputable def F : ℚ := 17

theorem rationalize_denominator_sum_equals_49 :
  let expr := (A * sqrt 3 + B * sqrt 5 + C * sqrt 11 + D * sqrt E) / F
  49 = A + B + C + D + E + F :=
by {
  -- The proof will go here.
  exact sorry
}

end rationalize_denominator_sum_equals_49_l57_57173


namespace base8_to_base10_4532_l57_57351

theorem base8_to_base10_4532 : 
    (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := 
by sorry

end base8_to_base10_4532_l57_57351


namespace find_lambda_l57_57401

variables {e1 e2 : ℝ^3} -- Defining the vectors e1 and e2
variable (a : ℝ^3) -- Defining the vector a
variables (λ : ℝ) -- Defining the variable λ

noncomputable def is_unit_vector (v : ℝ^3) : Prop :=
  ∥v∥ = 1
  
noncomputable def angle_between_vectors (v1 v2 : ℝ^3) (θ : ℝ) : Prop :=
  real.angle v1 v2 = θ

noncomputable def vector_a (e1 e2 : ℝ^3) (λ : ℝ) : ℝ^3 :=
  2 • e1 + (1 - λ) • e2

noncomputable def is_perpendicular (v1 v2 : ℝ^3) : Prop :=
  dot_product v1 v2 = 0

theorem find_lambda
  (h_unit_e1 : is_unit_vector e1)
  (h_unit_e2 : is_unit_vector e2)
  (h_angle_e1_e2 : angle_between_vectors e1 e2 (π / 3))
  (h_perpendicular : is_perpendicular (vector_a e1 e2 λ) e2) :
  λ = 2 :=
sorry

end find_lambda_l57_57401


namespace limit_log_alpha_n_neg_log_2_l57_57348

noncomputable def alpha_n (n : ℕ) : ℝ :=
  Inf {I : ℝ | ∃ f : polynomial ℤ, degree f = n ∧ ∫ x in -1..1, x^n * (f.eval x : ℝ) = I}

theorem limit_log_alpha_n_neg_log_2 :
  tendsto (λ n : ℕ, (log (alpha_n n) / n)) at_top (𝓝 (-log 2)) :=
sorry

end limit_log_alpha_n_neg_log_2_l57_57348


namespace couch_cost_l57_57699

theorem couch_cost
  (C : ℕ)  -- Cost of the couch
  (table_cost : ℕ := 100)
  (lamp_cost : ℕ := 50)
  (amount_paid : ℕ := 500)
  (amount_owed : ℕ := 400)
  (total_furniture_cost : ℕ := C + table_cost + lamp_cost)
  (remaining_amount_owed : total_furniture_cost - amount_paid = amount_owed) :
   C = 750 := 
sorry

end couch_cost_l57_57699


namespace time_to_pass_faster_train_l57_57978

-- Define the given conditions
def speed_slower_train : ℝ := 36
def speed_faster_train : ℝ := 45
def length_faster_train : ℝ := 90.0072
def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

-- Define the time it takes for the man in the slower train to pass the faster train
theorem time_to_pass_faster_train (h1 : speed_slower_train = 36) (h2 : speed_faster_train = 45) (h3 : length_faster_train = 90.0072) :
  (length_faster_train / (kmph_to_mps (speed_slower_train + speed_faster_train))) = 4 := 
sorry

end time_to_pass_faster_train_l57_57978


namespace sum_sequence_f_l57_57532

def f (x : ℚ) : ℚ := (1 + 10 * x) / (10 - 100 * x)

theorem sum_sequence_f :
  (finset.range 6000).sum (λ n, f^[n.succ] (1/2)) = 510 := sorry

end sum_sequence_f_l57_57532


namespace solve_for_x_l57_57180

theorem solve_for_x (x : ℝ) (h : (x - 5)^3 = (1 / 27)⁻¹) : x = 8 :=
sorry

end solve_for_x_l57_57180


namespace volume_to_surface_area_ratio_l57_57310

-- Definitions based on the conditions
def unit_cube_volume : ℕ := 1
def num_unit_cubes : ℕ := 7
def unit_cube_total_volume : ℕ := num_unit_cubes * unit_cube_volume

def surface_area_of_central_cube : ℕ := 0
def exposed_faces_per_surrounding_cube : ℕ := 5
def num_surrounding_cubes : ℕ := 6
def total_surface_area : ℕ := num_surrounding_cubes * exposed_faces_per_surrounding_cube

-- Mathematical proof statement
theorem volume_to_surface_area_ratio : 
  (unit_cube_total_volume : ℚ) / (total_surface_area : ℚ) = 7 / 30 :=
by sorry

end volume_to_surface_area_ratio_l57_57310


namespace parabola_shift_l57_57816

theorem parabola_shift (x : ℝ) :
  let y := -x^2
  let y_shifted := - (x - 2)^2 - 3
  in y_shifted = - (x - 2)^2 - 3 := by
  sorry

end parabola_shift_l57_57816


namespace hexagon_side_length_l57_57545

theorem hexagon_side_length (d : ℝ) (h : d = 15) : 
  let side := 10 * Real.sqrt 3 in side = d / (Real.sqrt 3 / 2) := 
by
  unfold let
  simp only [Real.sqrt, ←mul_assoc, Real.mul_div_cancel_left, ne.def, not_false_iff]
  rw [h, div_eq_iff (show Real.sqrt 3 ≠ (0 : ℝ), by linarith [Real.sqrt_ne_zero])]
  norm_num
  sorry

end hexagon_side_length_l57_57545


namespace argentina_matches_played_in_final_stage_total_matches_played_in_final_stage_l57_57839

def group_stage_matches (teams : List String) : List (String × String) :=
  teams.product teams |>.filter (λ (a, b) => a ≠ b)

theorem argentina_matches_played_in_final_stage (group_count : ℕ) (teams_per_group : ℕ) :
  let group_stage_matches := group_count * (teams_per_group * (teams_per_group - 1)) / 2
  let knockout_stage_matches := 8 + 4 + 2 + 1 + 1
  3 + 4 = 7 :=
by {
  -- Argentina played 3 matches in the group stage
  have group_stage_matches_argentina : 3 = teams_per_group - 1,
  -- Argentina played 4 matches in the knockout stage
  have knockout_stage_matches_argentina : 4 = knockout_stage_matches - (8 - 1) - (4 - 1) - (2 - 1) - 1,
  calc
  3 + 4 = 7 : by rwa [group_stage_matches_argentina, knockout_stage_matches_argentina]
}

theorem total_matches_played_in_final_stage (group_count : ℕ) (teams_per_group : ℕ) :
  let group_stage_matches := group_count * (teams_per_group * (teams_per_group - 1)) / 2
  let knockout_stage_matches := 8 + 4 + 2 + 1 + 1
  group_stage_matches + knockout_stage_matches = 64 :=
by {
  have calc_group_stage_matches : group_stage_matches = 48,
  have calc_knockout_stage_matches : knockout_stage_matches = 16,
  calc
  48 + 16 = 64 : by rwa [calc_group_stage_matches, calc_knockout_stage_matches]
}

end argentina_matches_played_in_final_stage_total_matches_played_in_final_stage_l57_57839


namespace football_hits_ground_l57_57206

theorem football_hits_ground :
  ∃ t : ℚ, -16 * t^2 + 18 * t + 60 = 0 ∧ 0 < t ∧ t = 41 / 16 :=
by
  sorry

end football_hits_ground_l57_57206


namespace count_positive_3_digit_numbers_divisible_by_9_l57_57451

-- Conditions
def is_divisible_by_9 (n : ℕ) : Prop := 9 ∣ n

def is_positive_3_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Theorem to be proved
theorem count_positive_3_digit_numbers_divisible_by_9 : 
  {n : ℕ | is_positive_3_digit_number n ∧ is_divisible_by_9 n}.card = 100 :=
sorry

end count_positive_3_digit_numbers_divisible_by_9_l57_57451


namespace christel_initial_dolls_l57_57357

theorem christel_initial_dolls :
  (∀ C : ℕ, ∀ D : ℕ, ∀ A : ℕ,
    D = 18 → A = 7 →
    A = (C - 5) + 2 → 
    A = D + 3 →
    C = 10) :=
begin
  intros C D A hD hA hA_C_2 hA_D_3,
  -- Skip the proof
  sorry
end

end christel_initial_dolls_l57_57357


namespace find_m_l57_57750

noncomputable theory
open Real

def f (x m : ℝ) : ℝ := x + exp (2 * x) - m

theorem find_m (m : ℝ) (h : ∀ x, f x m = x + exp (2 * x) - m)
  (h0 : f' = 1 + 2 * exp (2 * x))
  (tangent_at_0 : ∃ (y : ℝ), y = 1 - m)
  (area_triangle : (1/2) * |1 - m| * |(m - 1) / 3| = 1/6) :
  m = 2 ∨ m = 0 := 
sorry

end find_m_l57_57750


namespace preferred_point_condition_l57_57785

theorem preferred_point_condition (x y : ℝ) (h₁ : x^2 + y^2 ≤ 2008)
  (cond : ∀ x' y', (x'^2 + y'^2 ≤ 2008) → (x' ≤ x → y' ≥ y) → (x = x' ∧ y = y')) :
  x^2 + y^2 = 2008 ∧ x ≤ 0 ∧ y ≥ 0 :=
by
  sorry

end preferred_point_condition_l57_57785


namespace dot_product_a_b_l57_57431

-- Given unit vectors a, b, c
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (h_a : ∥a∥ = 1) (h_b : ∥b∥ = 1) (h_c : ∥c∥ = 1)
variables (h_eq : a + b + 2 • c = 0)

-- Prove that the dot product of a and b is equal to 1
theorem dot_product_a_b : ⟪a, b⟫ = 1 :=
by
  sorry

end dot_product_a_b_l57_57431


namespace find_integer_solutions_l57_57714

theorem find_integer_solutions :
  (a b : ℤ) →
  3 * a^2 * b^2 + b^2 = 517 + 30 * a^2 →
  (a = 2 ∧ b = 7) ∨ (a = -2 ∧ b = 7) ∨ (a = 2 ∧ b = -7) ∨ (a = -2 ∧ b = -7) :=
sorry

end find_integer_solutions_l57_57714


namespace find_mark_age_l57_57608

-- Define Mark and Aaron's ages
variables (M A : ℕ)

-- The conditions
def condition1 : Prop := M - 3 = 3 * (A - 3) + 1
def condition2 : Prop := M + 4 = 2 * (A + 4) + 2

-- The proof statement
theorem find_mark_age (h1 : condition1 M A) (h2 : condition2 M A) : M = 28 :=
by sorry

end find_mark_age_l57_57608


namespace book_arrangements_l57_57603

theorem book_arrangements (unique_books : Finset ℕ) (identical_books : ℕ) 
  (h1 : unique_books.card = 4) (h2 : identical_books = 3) : 
  (Finset.card unique_books + identical_books)! / identical_books.factorial = 840 :=
by
  sorry

end book_arrangements_l57_57603


namespace angle_is_60_degrees_l57_57062

def vec_a : ℝ × ℝ × ℝ := (0, 1, 1)
def vec_b : ℝ × ℝ × ℝ := (1, 1, 0)

noncomputable def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2 + a.3^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2 + b.3^2)
  Real.acos (dot_product / (magnitude_a * magnitude_b))

theorem angle_is_60_degrees : angle_between_vectors vec_a vec_b = Real.pi / 3 :=
by
  sorry

end angle_is_60_degrees_l57_57062


namespace count_3_digit_numbers_divisible_by_9_l57_57441

theorem count_3_digit_numbers_divisible_by_9 : 
  let count := (range (integer_divisible_in_range 9 100 999)).length
  count = 100 := 
by
  sorry

noncomputable def integer_divisible_in_range (k m n : ℕ) : List ℕ :=
  let start := m / k + (if (m % k = 0) then 0 else 1)
  let end_ := n / k
  List.range (end_ - start + 1) |>.map (λ i => (start + i) * k)

noncomputable def range (xs : List ℕ) := xs

end count_3_digit_numbers_divisible_by_9_l57_57441


namespace set_representation_l57_57329

theorem set_representation : 
  { x : ℕ | x < 5 } = {0, 1, 2, 3, 4} :=
sorry

end set_representation_l57_57329


namespace prove_q_ge_bd_and_p_eq_ac_l57_57573

-- Definitions for the problem
variables (a b c d p q : ℕ)

-- Conditions given in the problem
axiom h1: a * d - b * c = 1
axiom h2: (a : ℚ) / b > (p : ℚ) / q
axiom h3: (p : ℚ) / q > (c : ℚ) / d

-- The theorem to be proved
theorem prove_q_ge_bd_and_p_eq_ac (a b c d p q : ℕ) (h1 : a * d - b * c = 1) 
  (h2 : (a : ℚ) / b > (p : ℚ) / q) (h3 : (p : ℚ) / q > (c : ℚ) / d) :
  q ≥ b + d ∧ (q = b + d → p = a + c) :=
by
  sorry

end prove_q_ge_bd_and_p_eq_ac_l57_57573


namespace sin_alpha_plus_pi_div_two_eq_neg_four_fifths_l57_57730

open Real

theorem sin_alpha_plus_pi_div_two_eq_neg_four_fifths
  (α : ℝ)
  (h1 : tan (α - π) = 3 / 4)
  (h2 : α ∈ Ioo (π / 2) (3 * π / 2)) :
  sin (α + π / 2) = -4 / 5 :=
sorry

end sin_alpha_plus_pi_div_two_eq_neg_four_fifths_l57_57730


namespace impossible_coins_l57_57162

theorem impossible_coins (p1 p2 : ℝ) : 
  ¬ ((1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end impossible_coins_l57_57162


namespace fewer_popsicle_sticks_l57_57925

theorem fewer_popsicle_sticks :
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  boys_total - girls_total = 6 := 
by
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  show boys_total - girls_total = 6
  sorry

end fewer_popsicle_sticks_l57_57925


namespace find_value_of_function_l57_57769

theorem find_value_of_function (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x + f (-x) = 3 * x + 2) : 
  f 2 = 20 / 3 :=
sorry

end find_value_of_function_l57_57769


namespace farmers_income_2010_l57_57712

def avg_income_2010 (initial_wage initial_other wage_growth_rate other_income_increase years : ℝ) : ℝ :=
  initial_wage * (1 + wage_growth_rate) ^ years + initial_other + years * other_income_increase

theorem farmers_income_2010 :
  let initial_wage := 3600
  let initial_other := 2700
  let wage_growth_rate := 0.06
  let other_income_increase := 320
  let years := 5 
  8800 ≤ avg_income_2010 initial_wage initial_other wage_growth_rate other_income_increase years ∧ 
  avg_income_2010 initial_wage initial_other wage_growth_rate other_income_increase years < 9200 := 
  sorry

end farmers_income_2010_l57_57712


namespace min_value_of_polynomial_l57_57569

noncomputable def f (x : ℝ) : ℝ := x * (x + 4) * (x + 8) * (x + 12)

theorem min_value_of_polynomial : ∃ x : ℝ, f x = -256 := by
  let f (x : ℝ) : ℝ := x * (x + 4) * (x + 8) * (x + 12)
  have h : True := sorry  -- Here we would prove the function attains a value of -256
  exact ⟨-(2 + 2*real.sqrt 5), h⟩ -- Plugging in the value solving to -256

end min_value_of_polynomial_l57_57569


namespace collinear_orthocenters_l57_57932

-- Define the acute-angled triangle ABC
variables {A B C : Type} [is_triangle A B C]

-- Define the points L1 and P1 on the internal bisector of angle B and the points L2 and P2 on the external bisector
variables {H L1 P1 L2 P2 : Type}
  [meets_internal_bisector B A H B C]
  [meets_internal_bisector B C H B A]
  [meets_external_bisector B A H B C]
  [meets_external_bisector B C H B A]

-- Define the orthocenters of specified triangles
variables {H1 H2 : Type}
  [is_orthocenter H1 HL1P1]
  [is_orthocenter H2 HL2P2]

theorem collinear_orthocenters :
  collinear {B, H1, H2} :=
sorry

end collinear_orthocenters_l57_57932


namespace count_positive_3_digit_numbers_divisible_by_9_l57_57452

-- Conditions
def is_divisible_by_9 (n : ℕ) : Prop := 9 ∣ n

def is_positive_3_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Theorem to be proved
theorem count_positive_3_digit_numbers_divisible_by_9 : 
  {n : ℕ | is_positive_3_digit_number n ∧ is_divisible_by_9 n}.card = 100 :=
sorry

end count_positive_3_digit_numbers_divisible_by_9_l57_57452


namespace prob_not_hearing_fav_song_in_first_6_minutes_l57_57690

theorem prob_not_hearing_fav_song_in_first_6_minutes : 
  let total_songs := 12
  let favorite_song_length := 270 -- 4 minutes 30 seconds in seconds
  let total_time := 6 * 60 -- 6 minutes in seconds
  let arrangements := total_songs.factorial
  let ways_hear_full_song := (total_songs - 1).factorial + (total_songs - 2).factorial + (total_songs - 2).factorial
  let prob_hear_full_song := (ways_hear_full_song : ℤ) / (arrangements : ℤ)
  in 1 - prob_hear_full_song = (119 : ℤ) / (132 : ℤ) :=
by
  sorry

end prob_not_hearing_fav_song_in_first_6_minutes_l57_57690


namespace line_angle_l57_57226

theorem line_angle (x y : ℝ) (h : x + √3 * y - 1 = 0) : ∃ θ : ℝ, θ = 5 * Real.pi / 6 :=
by
  sorry

end line_angle_l57_57226


namespace Daniel_spent_2721_l57_57698

-- Define the conditions
def total_games := 346
def games_at_12 := 80
def discount_games := 40
def discount_percentage := 0.20
def price_at_12 := 12.00
def price_at_8 := 8.00
def price_at_7 := 7.00
def price_at_3 := 3.00
def percentage_8 := 0.40
def percentage_7 := 0.50

-- Define the calculation of total cost
def total_cost : ℝ :=
  let cost_80_games := (games_at_12 - discount_games) * price_at_12 + discount_games * (price_at_12 * (1 - discount_percentage))
  let remaining_games := total_games - games_at_12
  let games_at_8 := Int.floor (remaining_games * percentage_8)
  let games_at_7 := Int.floor (remaining_games * percentage_7)
  let games_at_3 := remaining_games - games_at_8 - games_at_7
  cost_80_games
    + games_at_8 * price_at_8
    + games_at_7 * price_at_7
    + games_at_3 * price_at_3

-- Prove that Daniel's total expenditure is $2721
theorem Daniel_spent_2721 :
  total_cost = 2721 := 
by
  sorry

end Daniel_spent_2721_l57_57698


namespace find_a_and_monotonic_intervals_find_range_of_a_for_two_zeros_l57_57057

noncomputable def f (a x : ℝ) := a * real.exp x - real.log (x + 2) + real.log a - 2

theorem find_a_and_monotonic_intervals (a : ℝ) : 
  (∃ x : ℝ, 2023 = x ∧ deriv (λ x, f a x) x = 0) → 
  a = 1 / (2025 * real.exp 2023) ∧
  (∀ x ∈ Ioo (-2 : ℝ) 2023, deriv (λ x, f a x) x < 0) ∧ 
  (∀ x ∈ Ioo 2023 ∞, deriv (λ x, f a x) x > 0) 
:= by
  sorry

theorem find_range_of_a_for_two_zeros (a : ℝ) : 
  a ≠ 0 → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔
  a > 0 ∧ a < real.exp 1 
:= by
  sorry

end find_a_and_monotonic_intervals_find_range_of_a_for_two_zeros_l57_57057


namespace problem_statement_l57_57810

def complex_number (m : ℂ) : ℂ :=
  (m^2 - 3*m - 4) + (m^2 - 5*m - 6) * Complex.I

theorem problem_statement (m : ℂ) :
  (complex_number m).im = m^2 - 5*m - 6 →
  (complex_number m).re = 0 →
  m ≠ -1 ∧ m ≠ 6 :=
by
  sorry

end problem_statement_l57_57810


namespace max_truck_height_l57_57315

theorem max_truck_height :
  ∀ (radius width h : ℝ), 
  radius = 4.5 ∧ width = 2.7 → 
  h ≤ Real.sqrt (radius ^ 2 - width ^ 2) ↔ h ≤ 3.6 := 
begin
  sorry
end

end max_truck_height_l57_57315


namespace speechGroupProblem_l57_57827

def countSelectionWays (male_total female_total males_selected females_selected : ℕ) : ℕ :=
  Nat.choose male_total males_selected * Nat.choose female_total females_selected

def countSpeechWays (non_consecutive : ℕ) (remaining_people non_consecutive_selected : ℕ) : ℕ :=
  Nat.perm remaining_people remaining_people * Nat.perm (remaining_people + non_consecutive - non_consecutive_selected) non_consecutive_selected

def totalWays (male_total female_total males_selected females_selected non_consecutive : ℕ) : ℕ :=
  (countSelectionWays male_total female_total males_selected females_selected) * (countSpeechWays non_consecutive males_selected females_selected)

theorem speechGroupProblem :
  totalWays 4 3 3 2 2 = 864 :=
by
  sorry

end speechGroupProblem_l57_57827


namespace number_minus_29_l57_57238

theorem number_minus_29 (x : ℕ) (h : x - 46 = 15) : x - 29 = 32 :=
sorry

end number_minus_29_l57_57238


namespace fibonacci_recurrence_l57_57275

def F : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => F (n + 1) + F n

theorem fibonacci_recurrence (n : ℕ) (h: n ≥ 2) : 
  F n = F (n-1) + F (n-2) := by
 {
 sorry
 }

end fibonacci_recurrence_l57_57275


namespace fly_in_box_maximum_path_length_l57_57660

theorem fly_in_box_maximum_path_length :
  let side1 := 1
  let side2 := Real.sqrt 2
  let side3 := Real.sqrt 3
  let space_diagonal := Real.sqrt (side1^2 + side2^2 + side3^2)
  let face_diagonal1 := Real.sqrt (side1^2 + side2^2)
  let face_diagonal2 := Real.sqrt (side1^2 + side3^2)
  let face_diagonal3 := Real.sqrt (side2^2 + side3^2)
  (4 * space_diagonal + 2 * face_diagonal3) = 4 * Real.sqrt 6 + 2 * Real.sqrt 5 :=
by
  sorry

end fly_in_box_maximum_path_length_l57_57660


namespace correct_answers_l57_57490

-- Definitions
variable (C W : ℕ)
variable (h1 : C + W = 120)
variable (h2 : 3 * C - W = 180)

-- Goal statement
theorem correct_answers : C = 75 :=
by
  sorry

end correct_answers_l57_57490


namespace new_students_l57_57835

theorem new_students (S_i : ℕ) (L : ℕ) (S_f : ℕ) (N : ℕ) 
  (h₁ : S_i = 11) 
  (h₂ : L = 6) 
  (h₃ : S_f = 47) 
  (h₄ : S_f = S_i - L + N) : 
  N = 42 :=
by 
  rw [h₁, h₂, h₃] at h₄
  sorry

end new_students_l57_57835


namespace matrix_power_example_l57_57864

open Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![(Real.cos (π / 4)), 0, -(Real.sin (π / 4)),
    0, 1, 0,
    (Real.sin (π / 4)), 0, (Real.cos (π / 4))]

theorem matrix_power_example : B ^ 2024 = 1 := by
  sorry

end matrix_power_example_l57_57864


namespace coloring_scheme_formula_l57_57289

-- Define the recurrence relation for the coloring problem
def coloring_schemes : ℕ → ℕ → ℕ 
| 0 _ := 0
| 1 k := k
| 2 k := k * (k - 1)
| n k := (k - 2) * (coloring_schemes (n - 1) k) + (k - 1) * (coloring_schemes (n - 2) k)

-- The main theorem to prove
theorem coloring_scheme_formula (n k : ℕ) (h : 2 ≤ n) : 
  coloring_schemes n k = (k - 1)^n + (k - 1) * (-1)^n :=
sorry

end coloring_scheme_formula_l57_57289


namespace total_crayons_lost_or_given_away_l57_57150

def crayons_given_away : ℕ := 52
def crayons_lost : ℕ := 535

theorem total_crayons_lost_or_given_away :
  crayons_given_away + crayons_lost = 587 :=
by
  sorry

end total_crayons_lost_or_given_away_l57_57150


namespace general_and_rectangular_eqs_max_area_triangle_PAB_l57_57499

noncomputable def line_parametric_eqs (t : ℝ) : ℝ × ℝ :=
(6 - (sqrt 3 / 2) * t, (1 / 2) * t)

noncomputable def curve_polar_eq (theta : ℝ) : ℝ :=
6 * cos theta

theorem general_and_rectangular_eqs :
  (∀ t : ℝ, let (x, y) := line_parametric_eqs t in x + sqrt 3 * y - 6 = 0) ∧
  (∀ theta : ℝ, let rho := curve_polar_eq theta in rho * cos theta = 6 * cos theta ∧ rho^2 = x^2 + y^2 →
   (x - 3)^2 + y^2 = 9) :=
by
  sorry

theorem max_area_triangle_PAB (P A B : ℝ × ℝ) :
  (∃ t : ℝ, A = line_parametric_eqs t ∧ B = line_parametric_eqs (-t)) ∧
  (∃ theta : ℝ, P = (curve_polar_eq theta * cos theta, curve_polar_eq theta * sin theta)) ∧
  (sqrt ((3 - 0)^2 + (0 - 0)^2) = 3) ∧
  (sqrt (1 + 3^2) = 2) →
  abs (3 * 3 * sqrt 3 / 4) = (27 * sqrt 3 / 4) :=
by
  sorry

end general_and_rectangular_eqs_max_area_triangle_PAB_l57_57499


namespace avg_weight_remaining_boys_l57_57559

-- Definitions of given conditions
def avgWeight24 := 50.25
def totalBoys := 32
def avgWeightClass := 48.975

-- Definition for weights
def totalWeight (n : Nat) (avg : Float) := n * avg

def remainingBoys := totalBoys - 24

theorem avg_weight_remaining_boys :
  let totalWeight24 := totalWeight 24 avgWeight24
  let totalWeightClass := totalWeight totalBoys avgWeightClass
  let W := (totalWeightClass - totalWeight24) / remainingBoys
  W = 45.15 :=
by
  sorry

end avg_weight_remaining_boys_l57_57559


namespace coeff_x_expansion_l57_57104

theorem coeff_x_expansion (x : ℝ) : 
  (coeff (expand (x^2 + 3*x + 2)^5 1)) = 240 := 
by
  sorry

end coeff_x_expansion_l57_57104


namespace find_interval_of_monotonic_increase_find_cos_2theta_l57_57066

-- Definitions from conditions
def a (x : ℝ) : ℝ × ℝ := (1, Real.cos (2 * x))
def b (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), -Real.sqrt 3)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Hypothesis for question 1
def interval_of_monotonic_increase (k : ℤ) : set ℝ := {x | k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12}

-- Proposition for question 1
theorem find_interval_of_monotonic_increase (k : ℤ) : 
  ∃ I, ∀ x ∈ I, f x = 2 * Real.sin (2 * x - Real.pi / 3) :=
sorry

-- Proposition for question 2
theorem find_cos_2theta (θ : ℝ) (h : f (θ / 2 + 2 * Real.pi / 3) = 6 / 5) : 
  Real.cos (2 * θ) = 7 / 25 :=
sorry

end find_interval_of_monotonic_increase_find_cos_2theta_l57_57066


namespace three_digit_numbers_divisible_by_9_l57_57443

theorem three_digit_numbers_divisible_by_9 : 
  let smallest := 108
  let largest := 999
  let common_diff := 9
  -- Using the nth-term formula for an arithmetic sequence
  -- nth term: l = a + (n-1) * d
  -- For l = 999, a = 108, d = 9
  -- (999 = 108 + (n-1) * 9) -> (n-1) = 99 -> n = 100
  -- Hence, the number of such terms (3-digit numbers) in the sequence is 100.
  ∃ n, n = 100 ∧ (largest = smallest + (n-1) * common_diff)
by {
  let smallest := 108
  let largest := 999
  let common_diff := 9
  use 100
  sorry
}

end three_digit_numbers_divisible_by_9_l57_57443


namespace value_of_a1_l57_57030

def seq (a : ℕ → ℚ) (a_8 : ℚ) : Prop :=
  ∀ n : ℕ, (a (n + 1) = 1 / (1 - a n)) ∧ a 8 = 2

theorem value_of_a1 (a : ℕ → ℚ) (h : seq a 2) : a 1 = 1 / 2 :=
  sorry

end value_of_a1_l57_57030


namespace polikarp_make_first_box_empty_l57_57153

theorem polikarp_make_first_box_empty (n : ℕ) (h : n ≤ 30) : ∃ (x y : ℕ), x + y ≤ 10 ∧ ∀ k : ℕ, k ≤ x → k + k * y = n :=
by
  sorry

end polikarp_make_first_box_empty_l57_57153


namespace magnitude_sum_vector_magnitude_l57_57434

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (4, -2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Prove that the vectors are parallel implies x = -2
lemma parallel_vectors (x : ℝ) (h : (4, -2) ∥ (x, 1)) : x = -2 :=
sorry

-- Prove the magnitude of the sum of vectors
theorem magnitude_sum (h : (4, -2) ∥ (-2, 1)) : ‖(4 + (-2), -2 + 1)‖ = sqrt 5 :=
by 
  rw [real.norm_eq_abs, abs_eq_sqrt_sq_iff]
  have : (4 + (-2))^2 + (-2 + 1)^2 = 5, by ring
  rw this
  ring
  sorry

-- Main theorem that combines both steps
theorem vector_magnitude (h : (vector_a ∥ vector_b (-2))) : ‖(4 + (-2), -2 + 1)‖ = sqrt 5 :=
by 
  apply magnitude_sum h
  sorry

end magnitude_sum_vector_magnitude_l57_57434


namespace geometric_sequence_seventh_term_l57_57369

theorem geometric_sequence_seventh_term :
  ∀ (a₁ a₂ : ℤ) (n : ℕ), a₁ = 3 → a₂ = -6 → n = 7 → 
  ((∃ r, r = a₂ / a₁ ∧ (aₙ = a₁ * r^(n-1))) →  aₙ = 192) :=
begin
  sorry
end

end geometric_sequence_seventh_term_l57_57369


namespace cookie_radius_and_area_l57_57353

def boundary_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 = 2 * x + 4 * y

theorem cookie_radius_and_area :
  (∃ r : ℝ, r = Real.sqrt 13) ∧ (∃ A : ℝ, A = 13 * Real.pi) :=
by
  sorry

end cookie_radius_and_area_l57_57353


namespace find_correct_speed_l57_57540

variables (d t : ℝ) -- Defining distance and time as real numbers

theorem find_correct_speed
  (h1 : d = 30 * (t + 5 / 60))
  (h2 : d = 50 * (t - 5 / 60)) :
  ∃ r : ℝ, r = 37.5 ∧ d = r * t :=
by 
  -- Skip the proof for now
  sorry

end find_correct_speed_l57_57540


namespace kite_condition_rectangle_condition_l57_57506

-- Definitions for the problem conditions
def Triangle (A B C : Type) := ∃ a b c, a ≠ b ∧ b ≠ c ∧ c ≠ a

-- Definitions for centers based on the problem description
def IsCircumcenter (O : Type) (△ : Triangle A B C) := sorry
def IsIncenter (K : Type) (△ : Triangle A B C) := sorry

-- The mathematically equivalent proof problem
theorem kite_condition (A B C O K : Type) 
  (h_triangle : Triangle A B C)
  (h_O : IsCircumcenter O h_triangle)
  (h_K : IsIncenter K h_triangle) :
  (∃ (h_kite : Quadrilateral B K C O), h_kite) ↔ (AB = AC) := sorry

theorem rectangle_condition (A B C O K : Type) 
  (h_triangle : Triangle A B C)
  (h_O : IsCircumcenter O h_triangle)
  (h_K : IsIncenter K h_triangle) :
  ¬ ∃ (h_rectangle : Quadrilateral B K C O), h_rectangle := sorry

end kite_condition_rectangle_condition_l57_57506


namespace triangle_area_l57_57819

-- Define the conditions
variables {A B C : ℝ} {a b c : ℝ} 
hypothesis (h1 : real.cos A = 3 / 5)
hypothesis (h2 : real.sin C = 2 * real.cos B)
hypothesis (h3 : a = 4)

-- Prove the area of triangle ABC
theorem triangle_area {A B C a b c : ℝ} 
  (h1 : real.cos A = 3 / 5)
  (h2 : real.sin C = 2 * real.cos B)
  (h3 : a = 4) : 
  ∃ (area : ℝ), area = 8 :=
by
  -- The exact proof is omitted and replaced by sorry
  sorry

end triangle_area_l57_57819


namespace conjugate_complex_point_in_fourth_quadrant_l57_57049

noncomputable def sum_powers_of_i (n : ℕ) : ℂ :=
  ∑ k in finset.range n, complex.I ^ (k + 1)

def z (n : ℕ) : ℂ := 
  (sum_powers_of_i n) / (2 + complex.I)

def z_conjugate (n : ℕ) : ℂ :=
  complex.conj (z n)

-- The Lean statement to prove the problem
theorem conjugate_complex_point_in_fourth_quadrant :
  let n := 2017,
      z_value := z n,
      bar_z := z_conjugate n,
      coord_bar_z := (bar_z.re, bar_z.im)
  in
    coord_bar_z = (1/5, -2/5) ∧ bar_z.im < 0 ∧ bar_z.re > 0 :=
begin
  sorry
end

end conjugate_complex_point_in_fourth_quadrant_l57_57049


namespace sum_a_b_l57_57798

theorem sum_a_b (a b : ℚ) (h1 : 3 * a + 5 * b = 47) (h2 : 4 * a + 2 * b = 38) : a + b = 85 / 7 :=
by
  sorry

end sum_a_b_l57_57798


namespace replace_asterisks_divisible_by_6_l57_57845

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0
def is_divisible_by_6 (n : ℕ) : Prop := is_even n ∧ is_divisible_by_3 n

def count_valid_numbers : ℕ := 1728 -- From the solution steps

theorem replace_asterisks_divisible_by_6 :
  let fixed_sum := 2 + 0 + 1 + 6 + 0 + 2 in
  let digit_choices := [0, 2, 4, 7, 8, 9] in
  (Σ' (d1 d2 d3 d4 d5 : ℕ) 
      (H1 : d1 ∈ digit_choices) 
      (H2 : d2 ∈ digit_choices) 
      (H3 : d3 ∈ digit_choices) 
      (H4 : d4 ∈ digit_choices) 
      (H5 : d5 ∈ digit_choices) 
      (Hlast : d5 ∈ [0, 2, 4, 8])
      (Hsum : is_divisible_by_3 (fixed_sum + d1 + d2 + d3 + d4 + d5)),
    true).card = count_valid_numbers := 
by {
  sorry
}

end replace_asterisks_divisible_by_6_l57_57845


namespace probability_three_girls_l57_57093

open BigOperators
open Finset

noncomputable def choose (n k : ℕ) : ℕ := if h : k ≤ n then nat.choose n k else 0

def probability_all_girls (num_girls num_boys num_select : ℕ) : ℚ :=
  (choose num_girls num_select : ℚ) / choose (num_girls + num_boys) num_select

theorem probability_three_girls (total_contestants : ℕ) (num_girls : ℕ) (num_boys : ℕ) (num_select : ℕ) :
  total_contestants = 8 → num_girls = 5 → num_boys = 3 → num_select = 3 →
  probability_all_girls num_girls num_boys num_select = 5 / 28 :=
by
  intros h_total h_girls h_boys h_select
  rw [h_total, h_girls, h_boys, h_select]
  sorry

end probability_three_girls_l57_57093


namespace fraction_condition_l57_57745

theorem fraction_condition (x : ℝ) (h₁ : x > 1) (h₂ : 1 / x < 1) : false :=
sorry

end fraction_condition_l57_57745


namespace distance_between_trees_l57_57479

theorem distance_between_trees (L : ℕ) (n : ℕ) (hL : L = 150) (hn : n = 11) (h_end_trees : n > 1) : 
  (L / (n - 1)) = 15 :=
by
  -- Replace with the appropriate proof
  sorry

end distance_between_trees_l57_57479


namespace jen_visits_exactly_two_countries_l57_57234

noncomputable def probability_of_visiting_exactly_two_countries (p_chile p_madagascar p_japan p_egypt : ℝ) : ℝ :=
  let p_chile_madagascar := (p_chile * p_madagascar) * (1 - p_japan) * (1 - p_egypt)
  let p_chile_japan := (p_chile * p_japan) * (1 - p_madagascar) * (1 - p_egypt)
  let p_chile_egypt := (p_chile * p_egypt) * (1 - p_madagascar) * (1 - p_japan)
  let p_madagascar_japan := (p_madagascar * p_japan) * (1 - p_chile) * (1 - p_egypt)
  let p_madagascar_egypt := (p_madagascar * p_egypt) * (1 - p_chile) * (1 - p_japan)
  let p_japan_egypt := (p_japan * p_egypt) * (1 - p_chile) * (1 - p_madagascar)
  p_chile_madagascar + p_chile_japan + p_chile_egypt + p_madagascar_japan + p_madagascar_egypt + p_japan_egypt

theorem jen_visits_exactly_two_countries :
  probability_of_visiting_exactly_two_countries 0.4 0.35 0.2 0.15 = 0.2432 :=
by
  sorry

end jen_visits_exactly_two_countries_l57_57234


namespace f_sum_neg_l57_57564

theorem f_sum_neg (a b c : ℝ) (h1 : a + b > 0) (h2 : b + c > 0) (h3 : c + a > 0) :
  let f := λ x : ℝ, -x^3 - x in
  f(a) + f(b) + f(c) < 0 :=
by
  let f := λ x : ℝ, -x^3 - x
  let hodd := ∀ x, f(-x) = -f(x)
  let hdec := ∀ x, deriv f x < 0
  sorry

end f_sum_neg_l57_57564


namespace coefficient_of_x_squared_in_expansion_l57_57358

theorem coefficient_of_x_squared_in_expansion :
  let general_term (r : ℕ) := (-2)^r * (Nat.choose 7 r) * x^((7 - r) / 2)
  (∃ k, k = -280 ∧ (∃ r, r = 3 ∧ general_term r = k * x^2) ) :=
by
  let general_term := λ (r : ℕ), (-2 : ℤ)^r * (Nat.choose 7 r : ℤ) * (x^((7 - r) / 2) : ℤ)
  exists (k : ℤ),
  have h1 : k = -280,
  exists 3,
  have h2 : 3 = 3,
  general_term 3 = k * (x^2 : ℤ),
  sorry

end coefficient_of_x_squared_in_expansion_l57_57358


namespace find_value_of_expression_l57_57077

theorem find_value_of_expression (m n : ℕ) (h : m - n = -2) : 2 - 5 * m + 5 * n = 12 := 
by 
  sorry

end find_value_of_expression_l57_57077


namespace angle_is_pi_over_3_l57_57790

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Given conditions
def condition1 : Prop := inner a (a + b) = 5
def condition2 : Prop := ∥a∥ = 2
def condition3 : Prop := ∥b∥ = 1

-- Define the angle between vectors a and b
noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  real.arccos (inner a b / (∥a∥ * ∥b∥))

-- Statement to be proved
theorem angle_is_pi_over_3 (h1 : condition1 a b) (h2 : condition2 a) (h3 : condition3 b) :
  angle_between_vectors a b = π / 3 :=
sorry

end angle_is_pi_over_3_l57_57790


namespace score_standard_deviation_l57_57372

theorem score_standard_deviation (mean std_dev : ℝ)
  (h1 : mean = 76)
  (h2 : mean - 2 * std_dev = 60) :
  100 = mean + 3 * std_dev :=
by
  -- Insert proof here
  sorry

end score_standard_deviation_l57_57372


namespace range_of_a_l57_57408

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a * x + 4 < 0 → false) → (-4 ≤ a ∧ a ≤ 4) :=
by 
  sorry

end range_of_a_l57_57408


namespace ordered_pairs_count_l57_57793

def num_solutions : Prop :=
  ∃(n : ℕ), 
  n = 26 ∧
  ((λ pairs, pairs = ∑ x in Finset.Icc (-2 : ℤ) 3, ∑ y in Finset.Icc (x^2) (x + 6), 1).card = n)

theorem ordered_pairs_count : num_solutions :=
begin
  sorry
end

end ordered_pairs_count_l57_57793


namespace solution1_solution2_l57_57759

noncomputable def problem1 (x y : ℝ) (p : ℝ) : Prop :=
  x - 2 * y + 1 = 0 ∧ y^2 = 2 * p * x ∧ 0 < p ∧ (abs (sqrt (1 + 4) * (y - y))) = 4 * sqrt 15

theorem solution1 (p: ℝ) : p = 2 :=
  sorry

noncomputable def problem2 (x y m n : ℝ) : Prop :=
  y^2 = 4 * x ∧ ∃ (F : ℝ × ℝ), F = (1, 0) ∧
  (∀ (M N : ℝ × ℝ), M ∈ y^2 = 4 * x ∧ N ∈ y^2 = 4 * x ∧ (F.1 - M.1) * (F.2 - N.1) + (F.2 - M.2) * (F.2 - N.2) = 0 →
  let area := (1/2) * abs ((N.1 - M.1) * (F.2 - M.2) - (N.2 - M.2) * (F.1 - M.1)) in
  ∃ min_area : ℝ, min_area = 12 - 8 * sqrt 2)

theorem solution2 (x y m n : ℝ) : ∃ min_area : ℝ, min_area = 12 - 8 * sqrt 2 :=
  sorry

end solution1_solution2_l57_57759


namespace max_poly_factors_real_coeff_x8_minus_1_l57_57221

theorem max_poly_factors_real_coeff_x8_minus_1 : ∃ k, (∀ p : Polynomial ℝ, p.factorization x^8 - 1 ∧ (∀ i, p i ≠ 0) → p.degree > 0) ∧ (k ≤ 5) :=
sorry

end max_poly_factors_real_coeff_x8_minus_1_l57_57221


namespace cube_root_of_sqrt_l57_57336

theorem cube_root_of_sqrt (x : ℝ) (h : x = 0.000001) : Real.cbrt (Real.sqrt x) = 0.1 :=
by
  rw [h]
  sorry

end cube_root_of_sqrt_l57_57336


namespace non_neg_reals_inequality_l57_57395

theorem non_neg_reals_inequality (a b c : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h₃ : a + b + c ≤ 3) :
  (a / (1 + a^2) + b / (1 + b^2) + c / (1 + c^2) ≤ 3/2) ∧
  (3/2 ≤ (1 / (1 + a) + 1 / (1 + b) + 1 / (1 + c))) :=
by
  sorry

end non_neg_reals_inequality_l57_57395


namespace exists_infinitely_many_perfect_functions_l57_57523

-- Define the set of points with integer coordinates
def Lambda : Set (ℤ × ℤ) := {P | True}

-- Define the collection of functions from Lambda to {1, -1}
def F := Lambda → ℤ

-- Define the Euclidean distance between two points P and Q
def d (P Q : ℤ × ℤ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define what it means for a function to be perfect
def is_perfect (f : F) : Prop :=
  ∀ g : F, set.finite {P : Λ | f P ≠ g P} →
  ∑ P Q in {P Q : Λ | 0 < d P Q ∧ d P Q < 2010}, (f P * f Q - g P * g Q) / d P Q ≥ 0

-- The statement to be proven
theorem exists_infinitely_many_perfect_functions :
  ∃ (S : Set F), set.infinite S ∧ ∀ f₁ f₂ ∈ S, f₁ ≠ f₂ → ¬ ∃ t : ℤ × ℤ, ∀ P, f₁ P = f₂ (t + P) :=
sorry

end exists_infinitely_many_perfect_functions_l57_57523


namespace max_t_value_l57_57135

open Real

def f (m x : ℝ) := (1/2) * m * x^2 - 2 * x + log (x + 1)

lemma local_extremum_at_one (m : ℝ) : 
  (∃ m, m = 3/2 → ∃ ε > 0, ∀ x, abs (x - 1) < ε → f m x ≥ f m 1) :=
begin
  sorry
end

noncomputable def g (m x : ℝ) := x^3 + (1/2) * m * x^2 - 2 * x

theorem max_t_value (m : ℝ) (hm : m ∈ set.Ico (-4 : ℝ) (-1)) : 
  ∃ t, 1 < t ∧ t ≤ (1 + sqrt 13) / 2 ∧ (∀ x ∈ set.Icc (1 : ℝ) t, g m x ≤ g m 1) :=
begin
  sorry
end

end max_t_value_l57_57135


namespace find_x_squared_plus_y_squared_l57_57800

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 - y^2 + x + y = 44) : x^2 + y^2 = 109 :=
sorry

end find_x_squared_plus_y_squared_l57_57800


namespace smallest_positive_period_of_f_max_and_min_of_f_in_interval_l57_57418

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 3) + sin (2 * x - π / 3) + 2 * (cos x)^2 - 1

theorem smallest_positive_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T' ≥ T) :=
sorry

theorem max_and_min_of_f_in_interval :
  ∃ (max min : ℝ), (∀ x ∈ Icc (-π/4) (π/4), f x ≤ max) ∧ (∀ x ∈ Icc (-π/4) (π/4), min ≤ f x) ∧ max = sqrt 2 ∧ min = -1  :=
sorry

end smallest_positive_period_of_f_max_and_min_of_f_in_interval_l57_57418


namespace three_digit_palindromes_l57_57464

theorem three_digit_palindromes (n : ℕ) (a b : ℕ) : 
  (n = a * 100 + b * 10 + a) → 
  (1 ≤ a ∧ a ≤ 4) → 
  (b ∈ {0, 2, 4, 6, 8}) → 
  (100 ≤ n ∧ n < 500) → 
  ∃ (c : ℕ), c = 20 := sorry

end three_digit_palindromes_l57_57464


namespace value_of_m_product_of_PA_PB_l57_57422

-- Definition of parameterized line l
def line_l (t : ℝ) (m : ℝ) := (2 + (1/2) * t, m + (sqrt 3 / 2) * t)

-- Given point P
def point_P := (1, 2)

-- Proof that m = 2 + sqrt 3
theorem value_of_m : ∃ t : ℝ, let x_eq := 2 + (1/2) * t in
  let y_eq := (2 + sqrt 3) + (sqrt 3 / 2) * t in
  point_P.1 = x_eq ∧ point_P.2 = y_eq :=
sorry

-- Definitions related to curve C1 and line intersection points
def circle_eq (x y : ℝ) := x^2 + y^2 = 16
def curve_C1 (ρ : ℝ) := ρ = 4

-- Roots from intersections t1 and t2
axiom roots_intersection : ∃ (t1 t2: ℝ), (t1 * t2 = 4*sqrt 3 - 5) ∧ (t1 + t2 = -(5 + 2*sqrt 3))

-- Find |PA| * |PB|
theorem product_of_PA_PB : 
  ∃ (t1 t2: ℝ), 
  t1 * t2 = 4 * sqrt 3 - 5 ∧ 
  t1 + t2 = -(5 + 2 * sqrt 3) ∧ 
  abs (t1 + 2) * abs (t2 + 2) = 4 * sqrt 3 - 13 :=
sorry

end value_of_m_product_of_PA_PB_l57_57422


namespace min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l57_57579

-- Part (a): For n = 12:
theorem min_sticks_to_break_for_square_12 : ∀ (n : ℕ), n = 12 → 
  (∃ (sticks : Finset ℕ), sticks.card = 12 ∧ sticks.sum id = 78 ∧ (¬ (78 % 4 = 0) → 
  ∃ (b : ℕ), b = 2)) := 
by sorry

-- Part (b): For n = 15:
theorem can_form_square_without_breaking_15 : ∀ (n : ℕ), n = 15 → 
  (∃ (sticks : Finset ℕ), sticks.card = 15 ∧ sticks.sum id = 120 ∧ (120 % 4 = 0)) :=
by sorry

end min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l57_57579


namespace exists_t_s_l57_57280

-- Define the rad function
def rad (n : ℕ) : ℕ :=
  if n = 0 then 0
  else n.factors.sorted.eraseDup.prod

-- Define the sequence a_n
noncomputable def a_seq : ℕ → ℕ
| 0     := 0
| (n+1) := a_seq n + rad (a_seq n)

-- Define the problem statement
theorem exists_t_s :
  ∃ t s : ℕ, ∃ primes : List ℕ, primes.sorted = true ∧ primes.nodup = true ∧ primes.length = s ∧ a_seq t = primes.prod :=
by
  sorry

end exists_t_s_l57_57280


namespace correct_transformation_l57_57262

theorem correct_transformation (a b : ℝ) (h : a ≠ 0) : 
  (a^2 / (a * b) = a / b) :=
by sorry

end correct_transformation_l57_57262


namespace count_solutions_l57_57794

def is_solution (a b c : ℕ) : Prop :=
  Nat.lcm a b = 180 ∧ Nat.lcm a c = 450 ∧ Nat.lcm b c = 675

theorem count_solutions :
  (finset.univ.filter (λ (t : ℕ × ℕ × ℕ), is_solution t.1 t.2.1 t.2.2)).card = 3 :=
by
  sorry

end count_solutions_l57_57794


namespace max_points_tangent_no_collinear_l57_57257

-- Definitions
def is_tangent (sphere : set ℝ) (p1 p2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, ∃ point_on_sphere : ℝ × ℝ × ℝ, 
  (point_on_sphere ∈ sphere) ∧ 
  (tangent_point ℝ p1 p2 point_on_sphere)

def no_collinear (points : set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ p1 p2 p3 ∈ points, 
  ¬ collinear p1 p2 p3

-- Proving the maximum number of points
theorem max_points_tangent_no_collinear (sphere : set ℝ) (points : set (ℝ × ℝ × ℝ)) 
(h : ∀ p1 p2 ∈ points, is_tangent sphere p1 p2) 
(h' : no_collinear points) : 
  cardinality points ≤ 4 :=
sorry

end max_points_tangent_no_collinear_l57_57257


namespace count_books_in_row_on_tuesday_l57_57240

-- Define the given conditions
def tiles_count_monday : ℕ := 38
def books_count_monday : ℕ := 75
def total_count_tuesday : ℕ := 301
def tiles_count_tuesday := tiles_count_monday * 2

-- The Lean statement we need to prove
theorem count_books_in_row_on_tuesday (hcbooks : books_count_monday = 75) 
(hc1 : total_count_tuesday = 301) 
(hc2 : tiles_count_tuesday = tiles_count_monday * 2):
  (total_count_tuesday - tiles_count_tuesday) / books_count_monday = 3 :=
by
  sorry

end count_books_in_row_on_tuesday_l57_57240


namespace sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age_l57_57521

-- Definitions of the conditions
def L := 2
def C := 2 * L^2  -- Chloe's age today based on Liam's age
def J := C + 3    -- Joey's age today

-- The future time when Joey's age is twice Liam's age
def future_time : ℕ := (sorry : ℕ) -- Placeholder for computation of 'n'
lemma compute_n : 2 * (L + future_time) = J + future_time := sorry

-- Joey's age at future time when it is twice Liam's age
def age_at_future_time : ℕ := J + future_time

-- Sum of the two digits of Joey's age at that future time
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Final statement: sum of the digits of Joey's age at the specified future time
theorem sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age :
  digit_sum age_at_future_time = 9 :=
by
  exact sorry

end sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age_l57_57521


namespace triangle_area_angle_C_l57_57511

noncomputable def area_of_triangle (a b c B : ℝ) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem triangle_area :
    (B = Real.pi * 5 / 6) →
    (a = Real.sqrt 3 * c) →
    (b = 2 * Real.sqrt 7) →
    c = 2 →
    area_of_triangle a b c B = Real.sqrt 3 :=
  by
  sorry

theorem angle_C :
    (A B C : ℝ) →
    (B = 150 * Real.pi / 180) →
    (sin A + Real.sqrt 3 * sin C = Real.sqrt 2 / 2) →
    (A + B + C = Real.pi) →
    C = 15 * Real.pi / 180 :=
  by
  sorry

end triangle_area_angle_C_l57_57511


namespace largest_positive_integer_n_l57_57009

def is_perfect_square (x : ℕ) : Prop :=
  ∃ m : ℕ, m * m = x

def euler_totient (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ k, nat.coprime k n).card

theorem largest_positive_integer_n (n : ℕ) :
  (∀ i, 1 ≤ i ∧ (n = i) → is_perfect_square (i * euler_totient i)) → (n = 1) :=
by
  sorry

end largest_positive_integer_n_l57_57009


namespace mask_donation_equation_l57_57651

theorem mask_donation_equation (x : ℝ) : 
  1 + (1 + x) + (1 + x)^2 = 4.75 :=
sorry

end mask_donation_equation_l57_57651


namespace back_wheel_revolutions_l57_57541

theorem back_wheel_revolutions (r_front r_back : ℝ) (revolutions_front : ℕ) (h : r_front = 3) (h2 : r_back = 0.5) (h3 : revolutions_front = 150) :
  let circumference_front := 2 * Real.pi * r_front,
      distance_front := circumference_front * revolutions_front,
      circumference_back := 2 * Real.pi * r_back,
      revolutions_back := distance_front / circumference_back
  in revolutions_back = 900 :=
by
  sorry

end back_wheel_revolutions_l57_57541


namespace math_problem_l57_57756

noncomputable def parabola (p : ℝ) := {x : ℝ × ℝ // x.2 ^ 2 = 2 * p * x.1}
def line (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def focus (p : ℝ) : ℝ × ℝ := (p, 0)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem math_problem (p : ℝ)
  (h_line_parabola_intersect : ∃ A B : ℝ × ℝ, line A.1 A.2 ∧ line B.1 B.2 ∧ parabola p A ∧ parabola p B ∧ dist A B = 4 * real.sqrt 15)
  (h_focus : ∃ F : ℝ × ℝ, F = focus p)
  (h_points_MN : ∃ M N : ℝ × ℝ, parabola p M ∧ parabola p N ∧ dot_product (M - (p, 0)) (N - (p, 0)) = 0) :
  p = 2 ∧ (∃ M N : ℝ × ℝ, parabola 2 M ∧ parabola 2 N ∧ dot_product (M - (2, 0)) (N - (2, 0)) = 0) ∧
  (∀ M N : ℝ × ℝ, parabola 2 M ∧ parabola 2 N ∧ dot_product (M - (2, 0)) (N - (2, 0)) = 0 → 
  (1/2) * |M.1 * N.2 - M.2 * N.1| = 12 - 8 * real.sqrt 2) := sorry

end math_problem_l57_57756


namespace distance_from_S_to_plane_PQR_l57_57905

noncomputable def distance_to_plane {α : Type*} [normed_group α] [normed_space ℝ α] 
  (S P Q R : α) (plane : affine_subspace ℝ α) : ℝ :=
dist (orthogonal_projection plane S : α) S

variables {P Q R S : ℝ × ℝ × ℝ}
variables {SP SQ SR : ℝ}

-- The conditions
def conditions (P Q R S : ℝ × ℝ × ℝ) (SP SQ SR : ℝ) : Prop :=
  SP = 15 ∧ SQ = 15 ∧ SR = 8 ∧ 
  dist P S = SP ∧ dist Q S = SQ ∧ dist R S = SR ∧ 
  P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧ 
  euclidean_space.e_dist_sq ℝ P Q = (dist P S^2 + dist Q S^2) ∧
  euclidean_space.e_dist_sq ℝ P R = (dist P S^2 + dist R S^2) ∧
  euclidean_space.e_dist_sq ℝ R Q = (dist R S^2 + dist Q S^2)

-- The proof statement
theorem distance_from_S_to_plane_PQR :
  ∀ (P Q R S : ℝ × ℝ × ℝ) (SP SQ SR : ℝ),
  conditions P Q R S SP SQ SR → 
  distance_to_plane S P Q R (affine_span ℝ {P, Q, R}) = 8 :=
by
  intros P Q R S SP SQ SR h_cond
  sorry

end distance_from_S_to_plane_PQR_l57_57905


namespace max_min_area_sum_l57_57866

variables {a b c A : ℝ}
def non_degenerate_triangle (a b c : ℝ) := (a + b + c = 4) ∧ (a = b * c * (real.sin A)^2)

theorem max_min_area_sum (M m : ℝ) (h1 : non_degenerate_triangle a b c)
  (hM : M = (1/2) * b * c * real.sqrt (a / (b * c)))
  (hm : m = (1/2)) :
  M^2 + m^2 = 91 / 108 :=
by {
  -- Proof steps would go here
  sorry
}

end max_min_area_sum_l57_57866


namespace find_quadruple_l57_57398

/-- Problem Statement:
Given distinct positive integers a, b, c, and d such that a + b = c * d and a * b = c + d,
find the quadruple (a, b, c, d) that meets these conditions.
-/

theorem find_quadruple :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
            0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
            (a + b = c * d) ∧ (a * b = c + d) ∧
            ((a, b, c, d) = (1, 5, 3, 2) ∨ (a, b, c, d) = (1, 5, 2, 3) ∨
             (a, b, c, d) = (5, 1, 3, 2) ∨ (a, b, c, d) = (5, 1, 2, 3) ∨
             (a, b, c, d) = (2, 3, 1, 5) ∨ (a, b, c, d) = (3, 2, 1, 5) ∨
             (a, b, c, d) = (2, 3, 5, 1) ∨ (a, b, c, d) = (3, 2, 5, 1)) :=
sorry

end find_quadruple_l57_57398


namespace johns_income_l57_57859

theorem johns_income 
  (J : ℝ)
  (john_tax_rate : ℝ := 0.30)
  (ingrid_income : ℝ := 72000)
  (ingrid_tax_rate : ℝ := 0.40)
  (combined_tax_rate : ℝ := 0.35581395348837205) :
  (J := 57000 : ℝ) := 
sorry

end johns_income_l57_57859


namespace camp_cedar_boys_count_l57_57688

-- Define the number of boys, girls, and counselors.
variables (B G K : ℕ)

-- Define the conditions
def girls_eq_3_times_boys (B G : ℕ) : Prop :=
  G = 3 * B

def total_children_eq_counselors_times_8 (K : ℕ) : Prop :=
  K * 8 = B + 3 * B

def required_counselors (K :ℕ) : Prop :=
  K = 20

-- Proposition to prove that number of boys is 40 given the conditions
theorem camp_cedar_boys_count
  (h1 : girls_eq_3_times_boys B G)
  (h2 : total_children_eq_counselors_times_8 K)
  (h3 : required_counselors K) : B = 40 :=
by sorry

end camp_cedar_boys_count_l57_57688


namespace smallest_n_interesting_meeting_l57_57736

theorem smallest_n_interesting_meeting (m : ℕ) (hm : 2 ≤ m) :
  ∀ (n : ℕ), (n ≤ 3 * m - 1) ∧ (∀ (rep : Finset (Fin (3 * m))), rep.card = n →
  ∃ subrep : Finset (Fin (3 * m)), subrep.card = 3 ∧ ∀ (x y : Fin (3 * m)), x ∈ subrep → y ∈ subrep → x ≠ y → ∃ z : Fin (3 * m), z ∈ subrep ∧ z = x + y) → n = 2 * m + 1 := by
  sorry

end smallest_n_interesting_meeting_l57_57736


namespace union_sets_l57_57803

open Set

variable {α : Type*}

def setA : Set ℝ := { x | -2 < x ∧ x < 0 }
def setB : Set ℝ := { x | -1 < x ∧ x < 1 }
def setC : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem union_sets : setA ∪ setB = setC := 
by {
  sorry
}

end union_sets_l57_57803


namespace triangle_area_l57_57715

theorem triangle_area :
  let y1 := 0;
      y2 := 1;
      y3 := 5;
      x1 := (2, 0); -- (x, y) intersection point from y - 2x = 1 
      x2 := (10, 0); -- (x, y) intersection point from 2y + x = 10 
      p1 := (0, 0); -- Intersection on y-axis 
      p2 := (1, 0); -- Intersection on y-axis 
  let base := y3 - y2; -- Length of base on y-axis 
  let height := 8 / 5; -- Length of height from intersection point of the lines 
  let area := (1 / 2) * base * height;
  area = 3.2 :=
by
  sorry

end triangle_area_l57_57715


namespace root_conditions_l57_57182

theorem root_conditions (f g : ℝ → ℝ)
  (h1 : ∀ x, f x = x^3 - 7*x^2 + 12*x - 10)
  (h2 : ∀ x, g x = x^3 - 10*x^2 - 2*x + 20) 
  (h3 : ∃ x0, f x0 = 0 ∧ g (2*x0) = 0) : 
  (∃ x0, (x0 = 5 ∧ f x0 = 0) → g 10 = 0) ∧ 
  (∃ x0, (x0 = -√2 ∧ f x0 = 0)) ∧ 
  (∃ x0, (x0 = √2 ∧ f x0 = 0)) :=
by 
  sorry

end root_conditions_l57_57182


namespace total_area_of_rhombuses_in_hexagon_l57_57332

theorem total_area_of_rhombuses_in_hexagon (A B C D E F : Point)
  (hexagon_regular : RegularHexagon A B C D E F)
  (area_hexagon : area (Hexagon A B C D E F) = 80) :
  ∃ (rhombus r : Rhombus) (S : set Rhombus), (r ∈ S) ∧ (area (⋃₀ S) = 45) :=
sorry

end total_area_of_rhombuses_in_hexagon_l57_57332


namespace square_BDEF_area_is_100_l57_57501

noncomputable def prove_square_area (AB BC EH : ℝ) (hAB : AB = 15) (hBC : BC = 20) (hEH : EH = 2) : Prop :=
  let AC := Real.sqrt (AB^2 + BC^2) in
  let x := 10 in
  -- Verify the conditions (implicitly through the proof construction)
  hAB ∧ hBC ∧ hEH ∧ (15 - x) / 2 = (x - (2 * 25 / 15)) / (20 * 2 / 15) → -- geometric similarity
  (x^2 = 100)  -- Conclude the area of the square BDEF

theorem square_BDEF_area_is_100 {AB BC EH : ℝ} (hAB : AB = 15) (hBC : BC = 20) (hEH : EH = 2) : prove_square_area AB BC EH hAB hBC hEH := sorry

end square_BDEF_area_is_100_l57_57501


namespace area_of_triangle_DEF_l57_57243

-- Define points D, E, and F
def D : ℝ × ℝ := (4, 0)
def E : ℝ × ℝ := (0, 4)
def is_on_line (point : ℝ × ℝ) : Prop := point.1 + point.2 = 10

-- Define function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 |> Real.sqrt

-- Define the length of the base DF
def base_DF (F : ℝ × ℝ) (h : is_on_line F) : ℝ :=
  distance D F

-- Define the height from E to the line DF
def height_E : ℝ := 4

-- Define the area of the triangle DEF
def area_triang_DEF (F : ℝ × ℝ) (h : is_on_line F) : ℝ :=
  1 / 2 * base_DF F h * height_E

-- Final theorem statement
theorem area_of_triangle_DEF (F : ℝ × ℝ) (h : is_on_line F) : area_triang_DEF F h = 12 := 
sorry

end area_of_triangle_DEF_l57_57243


namespace first_player_always_wins_l57_57614

theorem first_player_always_wins :
  ∃ A B : ℤ, A ≠ 0 ∧ B ≠ 0 ∧
  (A = 1998 ∧ B = -2 * 1998) ∧
  (∀ a b c : ℤ, (a = A ∨ a = B ∨ a = 1998) ∧ 
                (b = A ∨ b = B ∨ b = 1998) ∧ 
                (c = A ∨ c = B ∨ c = 1998) ∧ 
                a ≠ b ∧ b ≠ c ∧ a ≠ c → 
                ∃ x1 x2 : ℚ, x1 ≠ x2 ∧ 
                (a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0)) :=
by
  sorry

end first_player_always_wins_l57_57614


namespace number_of_possible_sums_of_subset_l57_57880

theorem number_of_possible_sums_of_subset :
  ∀ (C : Finset ℕ), C.card = 70 → (∀ x ∈ C, x ∈ Finset.range 121) → 
  let U := C.sum id in 
  ∃ (S : Finset ℕ), S.card = 3501 ∧ (∀ u ∈ S, ∃ (C' : Finset ℕ), C'.card = 70 
  ∧ (∀ x ∈ C', x ∈ Finset.range 121) ∧ C'.sum id = u)
:= 
begin
  -- Proof goes here
  sorry
end

end number_of_possible_sums_of_subset_l57_57880


namespace coefficient_x3_in_expansion_of_1_minus_x_power_6_l57_57103

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ :=
if k ≤ n then nat.choose n k else 0

-- Define the general term of the binomial expansion
noncomputable def binomial_term (n k : ℕ) (x : ℝ) : ℝ :=
binom n k * (-x)^k

-- Define the problem and prove the coefficient of x^3 in the expansion of (1-x)^6
theorem coefficient_x3_in_expansion_of_1_minus_x_power_6 :
  binomial_term 6 3 1 = -20 :=
sorry

end coefficient_x3_in_expansion_of_1_minus_x_power_6_l57_57103


namespace probability_multiple_of_4_5_or_7_l57_57676

-- Define the problem conditions
def num_cards : ℕ := 150
def cards : Finset ℕ := Finset.range (num_cards + 1) \ {0}
def is_multiple_of (m : ℕ) (n : ℕ) : Prop := n % m = 0

-- Define the count of multiples
def count_multiples (m : ℕ) : ℕ := (Finset.filter (is_multiple_of m) cards).card

-- Define the Probability Theorem
theorem probability_multiple_of_4_5_or_7 : (count_multiples 4 + count_multiples 5 + count_multiples 7
  - count_multiples 20 - count_multiples 28 - count_multiples 35 + count_multiples 140) / num_cards = 73 / 150 :=
by sorry

end probability_multiple_of_4_5_or_7_l57_57676


namespace circle_properties_l57_57654

-- Given definition specifying the statement condition.
def area_of_circle (r : ℝ) : ℝ := r^2 * Real.pi

-- Proof problem statement
theorem circle_properties (r d C : ℝ) (h : area_of_circle r = 4 * Real.pi) : 
  (d = 2 * r) ∧ (C = 2 * Real.pi * r) ∧ (d = 4) ∧ (C = 4 * Real.pi) :=
by
  -- Given the area condition, we aim to prove the diameter and circumference properties
  sorry

end circle_properties_l57_57654


namespace graph_empty_l57_57707

theorem graph_empty {x y : ℝ} : 
  x^2 + 3 * y^2 - 4 * x - 6 * y + 10 = 0 → false := 
by 
  sorry

end graph_empty_l57_57707


namespace find_a1_l57_57026

noncomputable def a_n : ℕ → ℝ := sorry
noncomputable def S_n : ℕ → ℝ := sorry

theorem find_a1
  (h1 : ∀ n : ℕ, a_n 2 * a_n 8 = 2 * a_n 3 * a_n 6)
  (h2 : S_n 5 = -62) :
  a_n 1 = -2 :=
sorry

end find_a1_l57_57026


namespace find_t_l57_57302

theorem find_t (interval : Set ℝ) (a b : ℝ) (prob : ℝ) (t : ℝ) :
  interval = Set.Icc (-2 : ℝ) 4 →
  a = -2 → b = 4 → prob = 1 / 4 →
  (∀ x, x ∈ interval → x*x ≤ t → prob) →
  t = 9 / 16 :=
by
  sorry

end find_t_l57_57302


namespace history_book_cost_l57_57979

def total_books : ℕ := 90
def cost_math_book : ℕ := 4
def total_price : ℕ := 397
def math_books_bought : ℕ := 53

theorem history_book_cost :
  ∃ (H : ℕ), H = (total_price - (math_books_bought * cost_math_book)) / (total_books - math_books_bought) ∧ H = 5 :=
by
  sorry

end history_book_cost_l57_57979


namespace vector_dot_product_inequality_l57_57067

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_dot_product_inequality (a b : V) : 
  |inner a b| ≤ ∥a∥ * ∥b∥ := sorry

end vector_dot_product_inequality_l57_57067


namespace greatest_prime_factor_of_expression_l57_57256

theorem greatest_prime_factor_of_expression : 
  ∃ p, prime p ∧ (∀ q, prime q ∧ q ∣ (3^7 + 6^6) → q ≤ p) ∧ p = 67 :=
begin
  sorry
end

end greatest_prime_factor_of_expression_l57_57256


namespace pumpkins_total_l57_57912

theorem pumpkins_total (s m : ℕ) (h₁ : s = 51) (h₂ : m = 23) : s + m = 74 := by
  rw [h₁, h₂]
  exact Nat.add_comm 51 23

end pumpkins_total_l57_57912


namespace triangle_cos_range_l57_57516

theorem triangle_cos_range 
  {A B C : ℝ} 
  (hA : A + B + C = π)
  (hSinA : Real.sin A = Real.sqrt 2 / 2) :
  (∀ (x : ℝ), ∃ (B C : ℝ), x = Real.cos B + Real.sqrt 2 * Real.cos C ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  → ((Real.cos B + Real.sqrt 2 * Real.cos C) ∈ (0, 1] ∪ (2, Real.sqrt 5]) :=
  sorry

end triangle_cos_range_l57_57516


namespace prob_union_l57_57749

noncomputable theory
open_locale classical

variables {Ω : Type} {P : set Ω → ℝ} {A B : set Ω}

-- Conditions of the problem
axiom mutually_exclusive : disjoint A B
axiom prob_A : P A = 0.3
axiom prob_B : P B = 0.4

-- Theorem statement
theorem prob_union : P (A ∪ B) = 0.7 :=
by sorry

end prob_union_l57_57749


namespace sum_of_real_roots_of_even_function_l57_57752

noncomputable def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem sum_of_real_roots_of_even_function (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_intersects : ∃ a b c d, f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d) :
  a + b + c + d = 0 :=
sorry

end sum_of_real_roots_of_even_function_l57_57752


namespace proof_characters_with_initial_D_l57_57146

def total_characters : ℕ := 360

def initial_A : ℕ := total_characters / 3
def initial_B : ℕ := (total_characters - initial_A) / 4
def remaining_after_B : ℕ := total_characters - initial_A - initial_B
def initial_C : ℕ := remaining_after_B / 5
def remaining_after_C : ℕ := remaining_after_B - initial_C
def initial_G : ℕ := remaining_after_C / 6
def remaining_after_G : ℕ := remaining_after_C - initial_G

def remaining_characters := total_characters - (initial_A + initial_B + initial_C + initial_G)

variables (D E F H : ℕ)

def condition_D : Prop := D = 3 * E
def condition_F : Prop := F = 2 * E
def condition_H : Prop := H = F
def condition_sum : Prop := D + E + F + H = remaining_characters

theorem proof_characters_with_initial_D : condition_D ∧ condition_F ∧ condition_H ∧ condition_sum → D = 45 :=
by
  sorry

end proof_characters_with_initial_D_l57_57146


namespace find_angle_ABC_l57_57546

-- Definitions for points A, B, C, D, E, F, the circumcircle ω, and the angles involved
variable (A B C D E F : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
noncomputable def circumcircle : Type := sorry
noncomputable def ∠ (P Q R : Type) : Type := sorry

-- Given conditions
variable (parallelogram_ABCD : A -> B -> C -> D -> Prop)
variable (angle_B_lt_90 : ∠ A B C < 90)
variable (AB_lt_BC : A -> B -> C -> D -> Prop)
variable (on_circumcircle : E -> circumcircle A B C -> Prop)
variable (tangents_to_circumcircle : (E -> circumcircle A B C -> Prop) -> D -> Prop)
variable (angle_EDA_eq_angle_FDC : ∠ E D A = ∠ F D C)

-- To prove that angle ABC is 60 degrees
theorem find_angle_ABC :
  ∀ (parallelogram_ABCD A B C D) (angle_B_lt_90 : ∠ A B C < 90) (AB_lt_BC A B C D) 
  (on_circumcircle E (circumcircle A B C)) (on_circumcircle F (circumcircle A B C)) 
  (tangents_to_circumcircle (on_circumcircle E (circumcircle A B C)) D) 
  (tangents_to_circumcircle (on_circumcircle F (circumcircle A B C)) D) 
  (angle_EDA_eq_angle_FDC : ∠ E D A = ∠ F D C),
  ∠ A B C = 60 :=
by sorry -- Proof not provided

end find_angle_ABC_l57_57546


namespace clear_board_possible_l57_57149

def operation (board : Array (Array Nat)) (op_type : String) (index : Fin 8) : Array (Array Nat) :=
  match op_type with
  | "column" => board.map (λ row => row.modify index fun x => x - 1)
  | "row" => board.modify index fun row => row.map (λ x => 2 * x)
  | _ => board

def isZeroBoard (board : Array (Array Nat)) : Prop :=
  board.all (λ row => row.all (λ x => x = 0))

theorem clear_board_possible (initial_board : Array (Array Nat)) : 
  ∃ (ops : List (String × Fin 8)), 
    isZeroBoard (ops.foldl (λ b ⟨t, i⟩ => operation b t i) initial_board) :=
sorry

end clear_board_possible_l57_57149


namespace square_with_12_sticks_square_with_15_sticks_l57_57584

-- Definitions for problem conditions
def sum_of_first_n_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def can_form_square (total_length : ℕ) : Prop :=
  total_length % 4 = 0

-- Given n = 12, check if breaking 2 sticks is required to form a square
theorem square_with_12_sticks : (n = 12) → ¬ can_form_square (sum_of_first_n_natural_numbers 12) → true :=
by
  intros
  sorry

-- Given n = 15, check if it is possible to form a square without breaking any sticks
theorem square_with_15_sticks : (n = 15) → can_form_square (sum_of_first_n_natural_numbers 15) → true :=
by
  intros
  sorry

end square_with_12_sticks_square_with_15_sticks_l57_57584


namespace avg_speed_of_car_l57_57288

theorem avg_speed_of_car 
    (uphill_speed : ℝ) (downhill_speed : ℝ)
    (uphill_distance : ℝ) (downhill_distance : ℝ)
    (h1 : uphill_speed = 30)
    (h2 : downhill_speed = 60)
    (h3 : uphill_distance = 100)
    (h4 : downhill_distance = 50) :
  let total_distance := uphill_distance + downhill_distance,
      time_uphill := uphill_distance / uphill_speed,
      time_downhill := downhill_distance / downhill_speed,
      total_time := time_uphill + time_downhill,
      avg_speed := total_distance / total_time
  in avg_speed = 36 :=
by
  sorry

end avg_speed_of_car_l57_57288


namespace midpoint_of_intersection_l57_57815

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y = 2

-- Define the parabola equation
def parabola_eq (y x : ℝ) : Prop := y^2 = 4 * x

-- Define the intersection points and their properties
def intersection_points (A B : ℝ × ℝ) (lines : A.1 - A.2 = 2 ∧ B.1 - B.2 = 2) (parabolas : A.2^2 = 4 * A.1 ∧ B.2^2 = 4 * B.1) : Prop

-- Define the midpoint of segment AB
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ( (A.1 + B.1) / 2, (A.2 + B.2) / 2 )

-- State the theorem to be proved
theorem midpoint_of_intersection :
  ∀ A B : ℝ × ℝ, 
    (line_eq A.1 A.2 ∧ line_eq B.1 B.2) →
    (parabola_eq A.2 A.1 ∧ parabola_eq B.2 B.1) →
    midpoint A B = (4, 2) := by
  sorry

end midpoint_of_intersection_l57_57815


namespace collinear_B_K1_K2_l57_57485

-- Definitions of the scalene triangle and required points and lines
variables {A B C : Point} -- Points defining the scalene triangle
variables (B1 B2 K1 K2 I D : Point) -- Additional points on lines and circles
variables (ω : Circle) -- Incircle of the triangle
variables (h_triangle : scalene_triangle A B C) -- Given triangle is scalene
variables (h_B1 : is_angle_bisector B B1) -- B1 lies as the angle bisector on AC
variables (h_B2 : is_adjacent_angle_bisector B B2) -- B2 lies as the angle bisector of the adjacent angle on AC
variables (h_tangent1 : tangent B1 ω K1) -- Tangent from B1 to the incircle touching at K1
variables (h_tangent2 : tangent B2 ω K2) -- Tangent from B2 to the incircle touching at K2

-- Main theorem
theorem collinear_B_K1_K2 : collinear B K1 K2 :=
sorry -- Proof omitted

end collinear_B_K1_K2_l57_57485


namespace find_y_coordinate_of_P_l57_57868

noncomputable def A : ℝ × ℝ := (-4, 0)
noncomputable def B : ℝ × ℝ := (-3, 2)
noncomputable def C : ℝ × ℝ := (3, 2)
noncomputable def D : ℝ × ℝ := (4, 0)
noncomputable def ell1 (P : ℝ × ℝ) : Prop := (P.1 + 4) ^ 2 / 25 + (P.2) ^ 2 / 9 = 1
noncomputable def ell2 (P : ℝ × ℝ) : Prop := (P.1 + 3) ^ 2 / 25 + ((P.2 - 2) ^ 2) / 16 = 1

theorem find_y_coordinate_of_P :
  ∃ y : ℝ,
    ell1 (0, y) ∧ ell2 (0, y) ∧
    y = 6 / 7 ∧
    6 + 7 = 13 :=
by
  sorry

end find_y_coordinate_of_P_l57_57868


namespace cyclist_speed_l57_57244

/-- 
  Two cyclists A and B start at the same time from Newton to Kingston, a distance of 50 miles. 
  Cyclist A travels 5 mph slower than cyclist B. After reaching Kingston, B immediately turns 
  back and meets A 10 miles from Kingston. --/
theorem cyclist_speed (a b : ℕ) (h1 : b = a + 5) (h2 : 40 / a = 60 / b) : a = 10 :=
by
  sorry

end cyclist_speed_l57_57244


namespace find_x_intercept_l57_57789

variables (a x y : ℝ)
def l1 (a x y : ℝ) : Prop := (a + 2) * x + 3 * y = 5
def l2 (a x y : ℝ) : Prop := (a - 1) * x + 2 * y = 6
def are_parallel (a : ℝ) : Prop := (- (a + 2) / 3) = (- (a - 1) / 2)
def x_intercept_of_l1 (a x : ℝ) : Prop := l1 a x 0

theorem find_x_intercept (h : are_parallel a) : x_intercept_of_l1 7 (5 / 9) := 
sorry

end find_x_intercept_l57_57789


namespace honey_production_l57_57080

theorem honey_production (honey_bees : ℕ) (honey : ℕ) (days : ℕ) 
  (h1 : honey_bees = 20) 
  (h2 : honey = 1) 
  (h3 : days = 20) 
  (honey_per_bee : ℕ) 
  (h_per_bee : honey_per_bee * honey_bees * days / days = honey * honey_bees) :
  honey_per_bee * honey_bees = 20 :=
by
  have h4 : honey_per_bee = honey
  { sorry },
  simp [h2, h4, h3] at h_per_bee,
  assumption

end honey_production_l57_57080


namespace discount_difference_is_correct_l57_57682

def original_amount : ℝ := 12000

def single_discount_amount (p : ℝ) (x : ℝ) : ℝ :=
  x * (1 - p / 100)

def successive_discount_amount
  (p1 p2 p3 : ℝ) (x : ℝ) : ℝ :=
  let after_first := x * (1 - p1 / 100)
  let after_second := after_first * (1 - p2 / 100)
  after_second * (1 - p3 / 100)

def difference_between_discounts : ℝ :=
  let amount_after_single := single_discount_amount 30 original_amount
  let amount_after_successive := successive_discount_amount 20 6 4 original_amount
  amount_after_successive - amount_after_single

theorem discount_difference_is_correct :
  difference_between_discounts = 263.04 := 
  sorry

end discount_difference_is_correct_l57_57682


namespace range_of_a_l57_57037

variables {α : Type*} [linear_ordered_field α]

theorem range_of_a {f : α → α} (h_inc : ∀ x y, -1 ≤ x ∧ x < 2 → -1 ≤ y ∧ y < 2 → x ≤ y → f x ≤ f y) 
                   (h_dom : ∀ x, -1 ≤ x ∧ x < 2) 
                   (h_ineq: ∀ a, f (a - 1) > f (1 - 3 * a)) : 
  ∀ a, 1 / 2 < a ∧ a ≤ 2 / 3 :=
begin
  intro a,
  sorry
end

end range_of_a_l57_57037


namespace option_A_is_correct_l57_57263

theorem option_A_is_correct (a b : ℝ) (h : a ≠ 0) : (a^2 / (a * b)) = (a / b) :=
by
  -- Proof will be filled in here
  sorry

end option_A_is_correct_l57_57263


namespace apples_to_pears_l57_57806

theorem apples_to_pears (a o p : ℕ) 
  (h1 : 10 * a = 5 * o) 
  (h2 : 3 * o = 4 * p) : 
  (20 * a) = 40 / 3 * p :=
sorry

end apples_to_pears_l57_57806


namespace perimeter_of_equilateral_triangle_l57_57196

-- Defining the conditions
def area_eq_twice_side (s : ℝ) : Prop :=
  (s^2 * Real.sqrt 3) / 4 = 2 * s

-- Defining the proof problem
theorem perimeter_of_equilateral_triangle (s : ℝ) (h : area_eq_twice_side s) : 
  3 * s = 8 * Real.sqrt 3 :=
sorry

end perimeter_of_equilateral_triangle_l57_57196


namespace expected_winnings_l57_57666

def probability_heads : ℚ := 1 / 3
def probability_tails : ℚ := 1 / 2
def probability_edge : ℚ := 1 / 6

def winning_heads : ℚ := 2
def winning_tails : ℚ := 2
def losing_edge : ℚ := -4

def expected_value : ℚ := probability_heads * winning_heads + probability_tails * winning_tails + probability_edge * losing_edge

theorem expected_winnings : expected_value = 1 := by
  sorry

end expected_winnings_l57_57666


namespace min_x1_x2_l57_57417

def f (x : ℝ) (t : ℝ) : ℝ :=
  if x >= 0 then sqrt x - t else 2 * (x + 1) - t

theorem min_x1_x2 :
  (∀ t : ℝ, 
    (t >= 0 ∧ t < 2) → 
    let x1 := t^2 in 
    let x2 := (1 / 2) * t - 1 in 
    x1 > x2 → x1 - x2 ≥ 15 / 16) :=
sorry

end min_x1_x2_l57_57417


namespace find_theta_l57_57042

noncomputable def P := (Real.sin (3 * Real.pi / 4), Real.cos (3 * Real.pi / 4))

theorem find_theta
  (theta : ℝ)
  (h_theta_range : 0 ≤ theta ∧ theta < 2 * Real.pi)
  (h_P_theta : P = (Real.sin theta, Real.cos theta)) :
  theta = 7 * Real.pi / 4 :=
sorry

end find_theta_l57_57042


namespace remainder_when_divided_by_63_l57_57623

theorem remainder_when_divided_by_63 (x : ℤ) (h1 : ∃ q : ℤ, x = 63 * q + r ∧ 0 ≤ r ∧ r < 63) (h2 : ∃ k : ℤ, x = 9 * k + 2) :
  ∃ r : ℤ, 0 ≤ r ∧ r < 63 ∧ r = 7 :=
by
  sorry

end remainder_when_divided_by_63_l57_57623


namespace trig_intersection_identity_l57_57187

theorem trig_intersection_identity (x0 : ℝ) (hx0 : x0 ≠ 0) (htan : -x0 = Real.tan x0) :
  (x0^2 + 1) * (1 + Real.cos (2 * x0)) = 2 := 
sorry

end trig_intersection_identity_l57_57187


namespace multiply_large_numbers_l57_57633

theorem multiply_large_numbers :
  72519 * 9999 = 724817481 :=
by
  sorry

end multiply_large_numbers_l57_57633


namespace number_at_2019th_vertex_l57_57111

variables {α : Type*} [AddCommGroup α]

def vertices_numbers (n : ℕ) : ℕ → α

axiom sum_nine_consecutive_vertices (n : ℕ) (f : ℕ → α) (h₁ : n = 2019) :
  ∀ k : ℕ, (f k) + (f (k + 1)) + (f (k + 2)) + (f (k + 3)) + (f (k + 4)) + (f (k + 5)) + (f (k + 6)) + (f (k + 7)) + (f (k + 8)) = 300

axiom vertex_19 (f : ℕ → α) : f 19 = 19
axiom vertex_20 (f : ℕ → α) : f 20 = 20

theorem number_at_2019th_vertex (f : ℕ → ℕ) (h₁ : ∀ k, (f k) + (f (k + 1)) + (f (k + 2)) + (f (k + 3)) + (f (k + 4)) + (f (k + 5)) + (f (k + 6)) + (f (k + 7)) + (f (k + 8)) = 300)
  (h₂ : f 19 = 19) (h₃ : f 20 = 20) : f 2019 = 61 :=
sorry

end number_at_2019th_vertex_l57_57111


namespace find_first_sequence_l57_57696

open Nat

def strictly_decreasing (s : List ℕ) : Prop :=
  ∀ i j, i < j → i < s.length ∧ j < s.length → s.get i > s.get j

def no_term_divides (s : List ℕ) : Prop :=
  ∀ i j, i < j → i < s.length ∧ j < s.length → ¬ (s.get j ∣ s.get i)

def minimal_under_ordering (s : List ℕ) : Prop :=
  ∀ t, strictly_decreasing t → no_term_divides t →
  (t.length = s.length → (∃ k, k < s.length ∧ s.get k < t.get k ∧ ∀ i, i < k → s.get i = t.get i) → False)

def first_sequence (n : ℕ) : List ℕ :=
  if h : n > 0 then List.range' n n | []

theorem find_first_sequence (n : ℕ) (h : n > 0) :
  let s := first_sequence (2 * n - 1)
  in strictly_decreasing s ∧ no_term_divides s ∧ minimal_under_ordering s := by
  sorry

end find_first_sequence_l57_57696


namespace tangent_to_parabola_l57_57012

open Real

-- Let's assume x0 and y0 to be points on parabola satisfying conditions
def parabola (x : ℝ) : ℝ := x^2

def focus : ℝ × ℝ := (0, 1/4)

def point_on_parabola (x0 : ℝ) : ℝ × ℝ := (x0, parabola x0)

def tangent_line (x0 y0 : ℝ) (x : ℝ) : ℝ := 2 * x0 * x - x0^2

def line_from_focus_to_point (x0 y0 : ℝ) : ℝ → ℝ := 
  let slope := (y0 - 1/4) / x0
  (λ x, slope * x + 1/4)

def angle_between_lines (m1 m2 : ℝ) : ℝ :=
  abs ((m1 - m2) / (1 + m1 * m2))

theorem tangent_to_parabola :
  ∃ (x0 : ℝ), 
    (angle_between_lines (2 * x0) ((x0 - 1/(4 * x0))) = 1 ∧
    (tangent_line (1/2) (1/4) = (λ x, x - 1/4)) ∧ 
    (tangent_line (-1/2) (1/4) = (λ x, -x - 1/4))) :=
sorry

end tangent_to_parabola_l57_57012


namespace simplify_expression_l57_57879

theorem simplify_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^4 + b^4 = a + b) (h2 : a^2 + b^2 = 2) :
  (a^2 / b^2 + b^2 / a^2 - 1 / (a^2 * b^2)) = 1 := 
sorry

end simplify_expression_l57_57879


namespace discount_rate_l57_57632

theorem discount_rate (marked_price selling_price discount_rate: ℝ) 
  (h₁: marked_price = 80)
  (h₂: selling_price = 68)
  (h₃: discount_rate = ((marked_price - selling_price) / marked_price) * 100) : 
  discount_rate = 15 :=
by
  sorry

end discount_rate_l57_57632


namespace determine_operator_positive_integer_equality_impossible_l57_57991

-- Define the fractions involved
def frac1 : ℚ := 169 / 30
def frac2 : ℚ := 13 / 15
def target : ℚ := 13 / 2

-- Prove the equality with the operations addition and division
theorem determine_operator (op : ℚ → ℚ → ℚ) : 
  (op frac1 frac2 = target) ↔ (op = (+) ∨ op = (/)) :=
by sorry

-- Additional theorem to prove the impossibility with positive integers
theorem positive_integer_equality_impossible {A B : ℕ} (h1 : 0 < A) (h2 : 0 < B) :
  ¬ (A + B = A / B ∨ A / B = B * B) :=
by sorry

end determine_operator_positive_integer_equality_impossible_l57_57991


namespace roots_opposite_signs_and_value_range_l57_57375

theorem roots_opposite_signs_and_value_range (m : ℝ) :
  let a := m + 3
  let b := -4m
  let c := 2m - 1
  let Δ := b^2 - 4 * a * c
  Δ > 0 →
  (b / a < 0) →
  ((2m - 1) / (m + 3) < 0) →
  (-3 < m ∧ m < 0) :=
by
  sorry

end roots_opposite_signs_and_value_range_l57_57375


namespace total_players_l57_57287

-- Definitions of the given conditions
def K : ℕ := 10
def Kho_only : ℕ := 40
def Both : ℕ := 5

-- The lean statement that captures the problem of proving the total number of players equals 50
theorem total_players : (K - Both) + Kho_only + Both = 50 :=
by
  -- Placeholder for the proof
  sorry

end total_players_l57_57287


namespace adjusted_distance_buoy_fourth_l57_57675

theorem adjusted_distance_buoy_fourth :
  let a1 := 20  -- distance to the first buoy
  let d := 4    -- common difference (distance between consecutive buoys)
  let ocean_current_effect := 3  -- effect of ocean current
  
  -- distances from the beach to buoys based on their sequence
  let a2 := a1 + d 
  let a3 := a2 + d
  let a4 := a3 + d
  
  -- distance to the fourth buoy without external factors
  let distance_to_fourth_buoy := a1 + 3 * d
  
  -- adjusted distance considering the ocean current
  let adjusted_distance := distance_to_fourth_buoy - ocean_current_effect
  adjusted_distance = 29 := 
by
  let a1 := 20
  let d := 4
  let ocean_current_effect := 3
  let a2 := a1 + d
  let a3 := a2 + d
  let a4 := a3 + d
  let distance_to_fourth_buoy := a1 + 3 * d
  let adjusted_distance := distance_to_fourth_buoy - ocean_current_effect
  sorry

end adjusted_distance_buoy_fourth_l57_57675


namespace train_crossing_time_l57_57463

-- Define the given conditions
def length_of_train : ℝ := 70 -- meters
def length_of_bridge : ℝ := 80 -- meters
def speed_of_train : ℝ := 36 * (1000 / 3600) -- kmph converted to m/s

-- Define the problem statement
theorem train_crossing_time :
  (length_of_train + length_of_bridge) / speed_of_train = 15 :=
by
  sorry

end train_crossing_time_l57_57463


namespace count_3_digit_numbers_divisible_by_9_l57_57440

theorem count_3_digit_numbers_divisible_by_9 : 
  let count := (range (integer_divisible_in_range 9 100 999)).length
  count = 100 := 
by
  sorry

noncomputable def integer_divisible_in_range (k m n : ℕ) : List ℕ :=
  let start := m / k + (if (m % k = 0) then 0 else 1)
  let end_ := n / k
  List.range (end_ - start + 1) |>.map (λ i => (start + i) * k)

noncomputable def range (xs : List ℕ) := xs

end count_3_digit_numbers_divisible_by_9_l57_57440


namespace Fermat_little_theorem_euler_totient_multiplicative_property1_euler_totient_multiplicative_property2_l57_57615

/-- Part I: Fermat's Little Theorem -/
theorem Fermat_little_theorem (a : ℕ) (p : ℕ) (hp : Nat.Prime p) : a^p ≡ a [MOD p] :=
sorry

/-- Part II(a): Multiplicative property of Euler's Totient function -/
theorem euler_totient_multiplicative_property1 (m n : ℤ) :
  let gcd := Int.gcd m n in let lcm := Int.lcm m n in 
  m.nat_abs.totient * n.nat_abs.totient = gcd.nat_abs.totient * lcm.nat_abs.totient :=
sorry

/-- Part II(b): Totient function and products related equality -/
theorem euler_totient_multiplicative_property2 (m n : ℤ) :
  let gcd := Int.gcd m n in
  m.nat_abs.totient * n.nat_abs.totient * gcd.nat_abs.totient = (m * n).nat_abs.totient * gcd.nat_abs.totient :=
sorry

end Fermat_little_theorem_euler_totient_multiplicative_property1_euler_totient_multiplicative_property2_l57_57615


namespace lcm_210_297_l57_57981

theorem lcm_210_297 : Nat.lcm 210 297 = 20790 := 
by sorry

end lcm_210_297_l57_57981


namespace binomial_combination_is_integer_l57_57373

theorem binomial_combination_is_integer (n k : ℕ) (h₁ : 1 ≤ k) (h₂ : k < n) :
  ∃ (m : ℤ), ((n - 2 * k - 1) / (k + 1)) * (nat.factorial n / (nat.factorial k * nat.factorial (n - k))) = m :=
by
  sorry

end binomial_combination_is_integer_l57_57373


namespace lines_perpendicular_l57_57361

theorem lines_perpendicular 
  (a b : ℝ) (θ : ℝ)
  (L1 : ∀ x y : ℝ, x * Real.cos θ + y * Real.sin θ + a = 0)
  (L2 : ∀ x y : ℝ, x * Real.sin θ - y * Real.cos θ + b = 0)
  : ∀ m1 m2 : ℝ, m1 = -(Real.cos θ) / (Real.sin θ) → m2 = (Real.sin θ) / (Real.cos θ) → m1 * m2 = -1 :=
by 
  intros m1 m2 h1 h2
  sorry

end lines_perpendicular_l57_57361


namespace problem_statement_l57_57022

theorem problem_statement
  (a : ℕ → ℝ) (a_sorted : ∀ i j : ℕ, i < j → a i ≤ a j)
  (sum_a_eq : (∑ i in Finset.range 8, a i) = 8 * x)
  (sum_a_squared_eq : (∑ i in Finset.range 8, (a i)^2) = 8 * y) :
  2 * Real.sqrt (y - x^2) ≤ a 7 - a 0 ∧ a 7 - a 0 ≤ 4 * Real.sqrt (y - x^2) := sorry

end problem_statement_l57_57022


namespace mrs_young_bonnets_monday_l57_57539

theorem mrs_young_bonnets_monday :
  ∃ M : ℕ, 
    (∃ T : ℕ, T = 2 * M) ∧
    (∃ Th : ℕ, Th = M + 5) ∧
    (∃ F : ℕ, F = (M + 5) - 5) ∧
    5 * M + 5 = 55 ∧
    M = 10 := 
begin
  -- the proof goes here
  sorry
end

end mrs_young_bonnets_monday_l57_57539


namespace price_of_red_car_l57_57891

noncomputable def car_price (total_amount loan_amount interest_rate : ℝ) : ℝ :=
  loan_amount + (total_amount - loan_amount) / (1 + interest_rate)

theorem price_of_red_car :
  car_price 38000 20000 0.15 = 35000 :=
by sorry

end price_of_red_car_l57_57891


namespace find_special_pairs_l57_57930

def is_special_pair (n k : ℕ) : Prop :=
  n ≤ k ∧ k ≤ Int.ceil (3 * n / 2 : ℝ)

theorem find_special_pairs (n k : ℕ) 
  (h : 1 ≤ k ∧ k ≤ 2 * n) : (∀ coins : list char, length coins = 2 * n ∧ list.countp (= 'A') coins = n ∧ list.countp (= 'B') coins = n →
  ∃ steps : list (list char) × list char, steps.from_left n ∨ steps.from_right n) ↔ is_special_pair n k :=
sorry

end find_special_pairs_l57_57930


namespace polynomial_coeff_sum_in_interval_l57_57141

/-- Given the polynomial expression (1 + x + x^2)^m, prove that the sum 
    ∑_{i=0}^{floor (2k/3)} (-1)^i * a_(k-i), i lies in the interval [0, 1] 
    for all k ≥ 0 where a_(m, n) are the coefficients. -/
theorem polynomial_coeff_sum_in_interval (a : ℕ → ℕ → ℤ)
  (h : ∀ m, (1 + X + X^2)^m = ∑ n in range (2 * m + 1), a m n * X^n) :
  ∀ k ≥ 0, (∑ i in range (Nat.floor (2 * k / 3) + 1), (-1)^i * a (k - i) i) ∈ Icc 0 1 :=
by
  intros k hk
  sorry

end polynomial_coeff_sum_in_interval_l57_57141


namespace group_C_contains_only_polyhedra_l57_57679

def is_polyhedron (s : Type) : Prop :=
  (∃ (faces : ℕ) (edges : ℕ) (vertices : ℕ), True) -- rudimentary definition, to be expanded.

inductive Solid
| TriangularPrism : Solid
| SquareFrustum : Solid
| Cube : Solid
| HexagonalPyramid : Solid
| Sphere : Solid
| Cone : Solid
| TruncatedCone : Solid
| Hemisphere : Solid

open Solid

-- Define the groups
def GroupA : List Solid := [TriangularPrism, SquareFrustum, Sphere, Cone]
def GroupB : List Solid := [TriangularPrism, SquareFrustum, Cube, TruncatedCone]
def GroupC : List Solid := [TriangularPrism, SquareFrustum, Cube, HexagonalPyramid]
def GroupD : List Solid := [Cone, TruncatedCone, Sphere, Hemisphere]

-- Checking if a group contains only polyhedra
def all_polyhedra (group : List Solid) : Prop :=
  ∀ s, s ∈ group → is_polyhedron s

-- Statement of the problem
theorem group_C_contains_only_polyhedra :
  all_polyhedra GroupC :=
sorry

end group_C_contains_only_polyhedra_l57_57679


namespace coplanar_and_touches_l57_57918

-- Given Definitions
variables (S : Type) [metric_space S] (plane : affine_subspace ℝ S)
variables (A B C D : S) (h_no_collinear : ¬ collinear ℝ ({A, B, C}))

-- A', B', C', D' defined such that S is tangent to the faces of tetrahedrons A'BCD, B'CDA, C'DAB, D'ABC
variables (A' B' C' D' : S)
variables (h_A' : tangent_to_faces_of (A' :: B :: C :: D :: []) S)
variables (h_B' : tangent_to_faces_of (B' :: C :: D :: A :: []) S)
variables (h_C' : tangent_to_faces_of (C' :: D :: A :: B :: []) S)
variables (h_D' : tangent_to_faces_of (D' :: A :: B :: C :: []) S)

-- Goal
theorem coplanar_and_touches :
  coplanar ℝ ({A', B', C', D'}) ∧ plane_touches S ({A', B', C', D'}) :=
sorry

end coplanar_and_touches_l57_57918


namespace apples_left_to_eat_l57_57886

theorem apples_left_to_eat (total_apples : ℕ) (one_fifth_has_worms : total_apples / 5) (nine_more_bruised : one_fifth_has_worms + 9):
  let wormy_apples := total_apples / 5 in
  let bruised_apples := wormy_apples + 9 in
  let apples_left_raw := total_apples - wormy_apples - bruised_apples in
  total_apples = 85 →
  apples_left_raw = 42 :=
by
  intros
  sorry

end apples_left_to_eat_l57_57886


namespace distance_covered_at_40_kmph_l57_57650

-- We define our conditions
variable (x : ℝ) -- distance covered at 40 kmph

axiom distance_total : ∀ (x : ℝ), (x + (250 - x) = 250)
axiom time_total : ∀ (x : ℝ), (x / 40 + (250 - x) / 60 = 5)

-- Define the theorem we aim to prove with Lean
theorem distance_covered_at_40_kmph : ∀ x, (distance_total x) ∧ (time_total x) → x = 100 :=
  by
  sorry

end distance_covered_at_40_kmph_l57_57650


namespace solve_x_l57_57648

noncomputable def diamond (a b : ℝ) : ℝ := a / b

axiom diamond_assoc (a b c : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0) : 
  diamond a (diamond b c) = a / (b / c)

axiom diamond_id (a : ℝ) (a_nonzero : a ≠ 0) : diamond a a = 1

theorem solve_x (x : ℝ) (h₁ : 1008 ≠ 0) (h₂ : 12 ≠ 0) (h₃ : x ≠ 0) : diamond 1008 (diamond 12 x) = 50 → x = 25 / 42 :=
by
  sorry

end solve_x_l57_57648


namespace equilateral_triangle_perimeter_l57_57194

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 3 * s = 8 * Real.sqrt 3 := 
by 
  sorry

end equilateral_triangle_perimeter_l57_57194


namespace number_of_different_duty_schedules_l57_57965

-- Define a structure for students
inductive Student
| A | B | C

-- Define days of the week excluding Sunday as all duties are from Monday to Saturday
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

-- Define the conditions in Lean
def condition_A_does_not_take_Monday (schedules : Day → Student) : Prop :=
  schedules Day.Monday ≠ Student.A

def condition_B_does_not_take_Saturday (schedules : Day → Student) : Prop :=
  schedules Day.Saturday ≠ Student.B

-- Define the function to count valid schedules
noncomputable def count_valid_schedules : ℕ :=
  sorry  -- This would be the computation considering combinatorics

-- Theorem statement to prove the correct answer
theorem number_of_different_duty_schedules 
    (schedules : Day → Student)
    (h1 : condition_A_does_not_take_Monday schedules)
    (h2 : condition_B_does_not_take_Saturday schedules)
    : count_valid_schedules = 42 :=
sorry

end number_of_different_duty_schedules_l57_57965


namespace alpha_sufficient_but_not_necessary_condition_of_beta_l57_57529
open Classical

variable (x : ℝ)
def α := x = -1
def β := x ≤ 0

theorem alpha_sufficient_but_not_necessary_condition_of_beta :
  (α x → β x) ∧ ¬(β x → α x) :=
by
  sorry

end alpha_sufficient_but_not_necessary_condition_of_beta_l57_57529


namespace dot_product_of_unit_vectors_l57_57429

variables {V : Type*} [inner_product_space ℝ V]

-- Define the vectors a, b, and c
variables (a b c : V)

-- Define the conditions: a, b, and c are unit vectors
def is_unit_vector (v : V) := ‖v‖ = 1

-- Define the condition that a + b + 2c = 0
def vector_sum_zero (a b c : V) := a + b + 2 • c = 0

-- The theorem to prove
theorem dot_product_of_unit_vectors
  (ha : is_unit_vector a)
  (hb : is_unit_vector b)
  (hc : is_unit_vector c)
  (h_sum : vector_sum_zero a b c) :
  ⟪a, b⟫ = 1 := 
sorry

end dot_product_of_unit_vectors_l57_57429


namespace count_valid_arrangements_l57_57822

theorem count_valid_arrangements :
  let grid_size := (4, 4) -- 4x4 grid
  let num_as := 2 -- 2 'a's
  let num_bs := 2 -- 2 'b's
  (num_arrangements grid_size num_as num_bs = 3960) :=
by
  -- Declaration of the number of arrangements under the given constraints
  sorry

end count_valid_arrangements_l57_57822


namespace area_of_rectangle_l57_57909

-- Definitions from problem conditions
variable (AB CD x : ℝ)
variable (h1 : AB = 24)
variable (h2 : CD = 60)
variable (h3 : BC = x)
variable (h4 : BF = 2 * x)
variable (h5 : similar (triangle AEB) (triangle FDC))

-- Goal: Prove the area of rectangle BCFE
theorem area_of_rectangle (h1 : AB = 24) (h2 : CD = 60) (x y : ℝ) 
  (h3 : BC = x) (h4 : BF = 2 * x) (h5 : BC * BF = y) : y = 1440 :=
sorry -- proof will be provided here

end area_of_rectangle_l57_57909


namespace tangent_slope_at_point_552_32_l57_57983

noncomputable def slope_of_tangent_at_point (cx cy px py : ℚ) : ℚ :=
if py - cy = 0 then 
  0 
else 
  (px - cx) / (py - cy)

theorem tangent_slope_at_point_552_32 : slope_of_tangent_at_point 3 2 5 5 = -2 / 3 :=
by
  -- Conditions from problem
  have h1 : slope_of_tangent_at_point 3 2 5 5 = -2 / 3 := 
    sorry
  
  exact h1

end tangent_slope_at_point_552_32_l57_57983


namespace volume_spatial_geometric_body_l57_57101

theorem volume_spatial_geometric_body : 
  let S := set_of (λ P : ℝ × ℝ × ℝ, P.1^2 + P.2^2 + P.3^2 ≤ 1 ∧ P.1 ≥ 0 ∧ P.2 ≥ 0 ∧ P.3 ≥ 0) in
  measure_theory.measure.volume S = π / 6 :=
sorry

end volume_spatial_geometric_body_l57_57101


namespace partition_condition_l57_57228

-- Defining the problem conditions in Lean 4

def integers_set (S : ℕ) : Prop :=
  ∃ (A : multiset ℕ), (∀ x ∈ A, 0 < x ∧ x ≤ 10) ∧ (A.sum = S)

def can_partition (A : multiset ℕ) : Prop :=
  ∃ (B C : multiset ℕ), A = B + C ∧ B.sum ≤ 80 ∧ C.sum ≤ 80

theorem partition_condition (S : ℕ) :
  (integers_set S → (can_partition (A : multiset ℕ) → A.sum = S)) ↔ S ≤ 152 := 
sorry

end partition_condition_l57_57228


namespace max_9_elements_in_set_example_set_satisfies_l57_57877

noncomputable def max_elements (A : set ℕ) : Prop :=
∀ m n ∈ A, abs (m - n) ≥ (m * n) / 25

theorem max_9_elements_in_set (A : set ℕ) (h : max_elements A) : A.toFinset.card ≤ 9 := sorry

def example_set : set ℕ := {1, 2, 3, 4, 6, 8, 12, 24, 600}

theorem example_set_satisfies : max_elements example_set := sorry

end max_9_elements_in_set_example_set_satisfies_l57_57877


namespace johnny_fishes_l57_57917

theorem johnny_fishes (total_fishes sony_multiple j : ℕ) (h1 : total_fishes = 120) (h2 : sony_multiple = 7) (h3 : total_fishes = j + sony_multiple * j) : j = 15 :=
by sorry

end johnny_fishes_l57_57917


namespace age_solution_l57_57075

theorem age_solution :
  ∃ me you : ℕ, me + you = 63 ∧ 
  ∃ x : ℕ, me = 2 * x ∧ you = x ∧ me = 36 ∧ you = 27 :=
by
  sorry

end age_solution_l57_57075


namespace largest_difference_of_primes_sum_126_l57_57952

open Nat

theorem largest_difference_of_primes_sum_126 (p q : ℕ) (hp : Prime p) (hq : Prime q) (h_sum : p + q = 126) (h_diff : p ≠ q):
  abs (p - q) ≤ 100 := sorry

end largest_difference_of_primes_sum_126_l57_57952


namespace second_runner_stop_time_l57_57976

def pace1 : ℝ := 8  -- The pace of the first runner in minutes per mile
def pace2 : ℝ := 7  -- The pace of the second runner in minutes per mile
def race_distance : ℝ := 10  -- The total distance of the race in miles
def stop_time : ℝ := 8  -- The time that the second runner stops for a drink in minutes

theorem second_runner_stop_time (t : ℝ) 
  (h1 : t = (race_distance - 1) * pace2 - stop_time)
  (h2 : t = 56) : t = 56 :=
by
  rw [h1, h2]
  exact h2

end second_runner_stop_time_l57_57976


namespace highest_visitors_at_4pm_yellow_warning_time_at_12_30pm_l57_57642

-- Definitions for cumulative visitors entering and leaving
def y (x : ℕ) : ℕ := 850 * x + 100
def z (x : ℕ) : ℕ := 200 * x - 200

-- Definition for total number of visitors at time x
def w (x : ℕ) : ℕ := y x - z x

-- Proof problem statements
theorem highest_visitors_at_4pm :
  ∀x, x ≤ 9 → w 9 ≥ w x :=
sorry

theorem yellow_warning_time_at_12_30pm :
  ∃x, w x = 2600 :=
sorry

end highest_visitors_at_4pm_yellow_warning_time_at_12_30pm_l57_57642


namespace range_of_a_l57_57814

-- Defining the function f : ℝ → ℝ
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x + a * Real.log x

-- Main theorem statement
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ (f a x1 = 0 ∧ f a x2 = 0)) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l57_57814


namespace games_bought_l57_57722

/-- 
Given:
1. Geoffrey received €20 from his grandmother.
2. Geoffrey received €25 from his aunt.
3. Geoffrey received €30 from his uncle.
4. Geoffrey now has €125 in his wallet.
5. Geoffrey has €20 left after buying games.
6. Each game costs €35.

Prove that Geoffrey bought 3 games.
-/
theorem games_bought 
  (grandmother_money aunt_money uncle_money total_money left_money game_cost spent_money games_bought : ℤ)
  (h1 : grandmother_money = 20)
  (h2 : aunt_money = 25)
  (h3 : uncle_money = 30)
  (h4 : total_money = 125)
  (h5 : left_money = 20)
  (h6 : game_cost = 35)
  (h7 : spent_money = total_money - left_money)
  (h8 : games_bought = spent_money / game_cost) :
  games_bought = 3 := 
sorry

end games_bought_l57_57722


namespace gcd_factorial_8_12_l57_57015

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def gcd (a b : ℕ) : ℕ := (Nat.gcd a b)

theorem gcd_factorial_8_12 :
  gcd (factorial 8) (factorial 12) = factorial 8 :=
by sorry

end gcd_factorial_8_12_l57_57015


namespace function_value_at_2017_l57_57053

noncomputable def f (x : ℝ) : ℝ := 
if x ≤ 2 ∧ x ≥ -1 then x + 1 else f (x % 3)

theorem function_value_at_2017 :
  (∀ x, f(x + 3) = f(x)) →
  (∀ x, -1 ≤ x ∧ x ≤ 2 → f(x) = x + 1) →
  f 2017 = 2 := 
by {
  intros h_periodicity h_initial_value,
  sorry
}

end function_value_at_2017_l57_57053


namespace ellipse_equation_and_sum_abs_coeffs_l57_57693

-- Define the parametric equations
def x (t : ℝ) : ℝ := (3 * (Real.sin t + 2)) / (3 + Real.cos t)
def y (t : ℝ) : ℝ := (4 * (Real.cos t - 2)) / (3 + Real.cos t)

-- Provide the main theorem reflecting the problem
theorem ellipse_equation_and_sum_abs_coeffs :
  (∃ A B C D E F : ℤ, 
    (∀ t : ℝ, 
      64 * (x t)^2 + 9 * (x t) * (y t) + 81 * (y t)^2 - 192 * (x t) - 243 * (y t) + 784 = 0)
    ∧ |A| + |B| + |C| + |D| + |E| + |F| = 1373) :=
sorry

end ellipse_equation_and_sum_abs_coeffs_l57_57693


namespace slope_of_vertical_line_l57_57950

theorem slope_of_vertical_line (x : ℝ) : x = 3 → ¬ ∃ m : ℝ, slope (line (pt (3, 0)) (pt (3, 1))) = m :=
by
  sorry

end slope_of_vertical_line_l57_57950


namespace artist_hair_color_black_l57_57607

structure Person :=
  (name : String)
  (job : String)
  (hair_color : String)

theorem artist_hair_color_black (sculptor violinist artist : Person)
  (sculptor.name = "Belov")
  (violinist.name = "Chernov")
  (artist.name = "Ryzhov")
  (sculptor.hair_color ≠ "white" ∧ sculptor.hair_color ≠ "Belov")
  (violinist.hair_color ≠ "black" ∧ violinist.hair_color ≠ "Chernov")
  (artist.hair_color ≠ "red" ∧ artist.hair_color ≠ "Ryzhov")
  (black_haired_person : ∃ p, p.hair_color = "black" ∧ p.name = "Ryzhov")
  (belov_affirmed : sculptor.name = "Belov")
  (chernov_hair_color_black : violinist.hair_color = "black") :
  artist.hair_color = "black" := 
by
  sorry

end artist_hair_color_black_l57_57607


namespace popsicle_sticks_difference_l57_57922

def popsicle_sticks_boys (boys : ℕ) (sticks_per_boy : ℕ) : ℕ :=
  boys * sticks_per_boy

def popsicle_sticks_girls (girls : ℕ) (sticks_per_girl : ℕ) : ℕ :=
  girls * sticks_per_girl

theorem popsicle_sticks_difference : 
    popsicle_sticks_boys 10 15 - popsicle_sticks_girls 12 12 = 6 := by
  sorry

end popsicle_sticks_difference_l57_57922


namespace rangeOfMagnitude_l57_57966

noncomputable def ellipseC : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1^2)/2 + p.2^2 = 1}
def rightFocus : ℝ × ℝ := (Real.sqrt 2, 0)
def lineThruFocus (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = k * p.2 + Real.sqrt 2}

variable (A B : ℝ × ℝ)
variable (F : ℝ × ℝ := rightFocus)
variable (T : ℝ × ℝ := (2,0))

def validIntersection (A B : ℝ × ℝ) : Prop := 
  (A ∈ ellipseC) ∧ (B ∈ ellipseC) ∧ (A ∈ lineThruFocus k) ∧ (B ∈ lineThruFocus k) ∧ (A ≠ B)

def vectorFA (A : ℝ × ℝ) (F : ℝ × ℝ) := (A.1 - F.1, A.2 - F.2)
def vectorFB (B : ℝ × ℝ) (F : ℝ × ℝ) := (B.1 - F.1, B.2 - F.2)
def lambdaCond (λ : ℝ) : Prop := λ ∈ Set.Icc (-2 : ℝ) (-1 : ℝ)

theorem rangeOfMagnitude (λ : ℝ) (hλ : lambdaCond λ) (k : ℝ)
    (hA_B_valid : validIntersection A B):
    (vectorFA A F) = λ • (vectorFB B F) → 
    let TA := (A.1 - T.1, A.2 - T.2)
    let TB := (B.1 - T.1, B.2 - T.2)
    Set.range TA + TB ⊆ Icc (2 : ℝ) (13 * Real.sqrt 2 / 8) := by
  sorry

end rangeOfMagnitude_l57_57966


namespace fraction_of_boys_reading_l57_57657

-- Initial conditions
variables (total_girls total_boys not_reading_total : ℕ)
variable (fraction_girls_reading : ℚ)
variables (girls_reading boys_not_reading total_students boys_reading : ℕ)

theorem fraction_of_boys_reading :
  total_girls = 12 →
  total_boys = 10 →
  fraction_girls_reading = 5 / 6 →
  not_reading_total = 4 →
  girls_reading = 10 →
  (fraction_girls_reading * total_girls).toInt = girls_reading →
  total_students = total_girls + total_boys →
  girls_not_reading = total_girls - girls_reading →
  boys_not_reading = not_reading_total - girls_not_reading →
  boys_reading = total_boys - boys_not_reading →
  (boys_reading / total_boys : ℚ) = 4 / 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end fraction_of_boys_reading_l57_57657


namespace length_AB_calculation_l57_57252

-- Define lengths of the sides of triangle ABC
def length_AB : ℝ := 15
def length_BC : ℝ := 25

-- Define lengths of the sides of triangle DEF
def length_DE : ℝ := 9
def length_EF : ℝ := 21
def length_DF : ℝ := 15

-- Define similarity of the triangles
def triangles_similar (AB BC AC DE EF DF : ℝ) : Prop :=
  (AB / DE = BC / EF) ∧ (AB / DE = AC / DF)

-- Given that triangles are similar
lemma similar_triangles_condition :
  triangles_similar length_AB length_BC (length_AB * length_EF / length_BC) length_DE length_EF length_DF :=
by
  unfold triangles_similar
  rw [div_eq_div_iff (ne_of_gt (by linarith)) (ne_of_gt (by linarith))]
  exact And.intro rfl rfl

-- Prove that AB = 12.6 cm given the similarity and side lengths
theorem length_AB_calculation :
  length_AB * length_EF / length_DE = 12.6 :=
by sorry

end length_AB_calculation_l57_57252


namespace area_of_triangle_value_of_angle_C_l57_57515

-- Define the problem constraints and prove the statements
noncomputable def triangle_problem_part1 (a b c : ℝ) (A B C : ℝ) : Prop :=
  B = 150 ∧
  a = sqrt 3 * c ∧
  b = 2 * sqrt 7 ∧
  (1/2) * a * c * (Float.sin (Float.toRadians B)) = sqrt 3

noncomputable def triangle_problem_part2 (A B C : ℝ) : Prop :=
  B = 150 ∧
  (∀ A C : ℝ, sin A + sqrt 3 * sin C = sqrt 2 / 2) ∧
  C = 15

theorem area_of_triangle {a b c A B C : ℝ} :
  triangle_problem_part1 a b c A B C :=
by {
  sorry
}

theorem value_of_angle_C {A B C : ℝ} :
  triangle_problem_part2 A B C :=
by {
  sorry
}

end area_of_triangle_value_of_angle_C_l57_57515


namespace count_three_digit_numbers_divisible_by_9_l57_57460

theorem count_three_digit_numbers_divisible_by_9 : 
  (finset.filter (λ n, (n % 9 = 0)) (finset.range 1000).filter (λ n, 100 ≤ n)).card = 100 :=
by
  sorry

end count_three_digit_numbers_divisible_by_9_l57_57460


namespace sequence_product_eq_l57_57222

theorem sequence_product_eq :
  (∏ n in Finset.range 9, (1 - 1 / (n + 2)^2)) = (11 / 20) :=
by
  sorry

end sequence_product_eq_l57_57222


namespace tire_repair_l57_57689

namespace CarTravel

variable (A B MidPoint : ℝ)
variable (initial_speed_A initial_speed_B : ℝ)
variable (time_to_midpoint : ℝ)
variable (total_time : ℝ)
variable (new_speed_A_ratio initial_speed_ratio : ℝ)
variable (tire_repair_duration : ℝ)
variable (distance_AB : ℝ)
variable (fraction_distance_repaired : ℝ)
variable (new_speed_A : ℝ)
variable (distance_traveled_to_burst : ℝ)
variable (time_traveled_to_burst : ℝ)
variable (time_after_repair : ℝ)
variable (total_travel_time_A : ℝ)

# Define conditions
protected def conditions : Prop :=
  initial_speed_ratio = 5 / 4 ∧
  total_time = 3 ∧
  time_to_midpoint = 3 ∧
  new_speed_A = initial_speed_A * 1.2 ∧
  MidPoint = distance_AB / 2 ∧
  total_travel_time_A = (distance_AB / 3) / initial_speed_A +
                       (MidPoint - distance_AB / 3) / new_speed_A ∧
  repair_duration = 3 - ((1 / initial_speed_ratio) * (distance_AB / 2) / initial_speed_A) -
                    ((1 / new_speed_A_ratio) * (distance_AB / 3) / new_speed_A)

# Define the theorem to prove
theorem tire_repair : conditions ->
  tire_repair_duration = 52 / 60 :=
  begin
    intros cond,
    sorry
  end

end CarTravel

end tire_repair_l57_57689


namespace percentage_defective_units_shipped_l57_57106

noncomputable def defective_percent : ℝ := 0.07
noncomputable def shipped_percent : ℝ := 0.05

theorem percentage_defective_units_shipped :
  defective_percent * shipped_percent * 100 = 0.35 :=
by
  -- Proof body here
  sorry

end percentage_defective_units_shipped_l57_57106


namespace people_to_right_of_taehyung_l57_57190

-- Given conditions
def total_people : Nat := 11
def people_to_left_of_taehyung : Nat := 5

-- Question and proof: How many people are standing to Taehyung's right?
theorem people_to_right_of_taehyung : total_people - people_to_left_of_taehyung - 1 = 4 :=
by
  sorry

end people_to_right_of_taehyung_l57_57190


namespace sum_of_interior_angles_nth_term_arithmetic_progression_ratio_parallel_lines_trigonometric_expression_l57_57821

-- Problem 55.1
theorem sum_of_interior_angles (n : ℕ) (a : ℝ) (h1 : n = 8) : a = 1080 :=
by
  sorry

-- Problem I5.2
theorem nth_term_arithmetic_progression (a1 : ℝ) (d : ℝ) (an : ℝ) (n : ℕ) (h1 : a1 = 80) (h2 : d = 50) (h3 : an = 1080) : n = 21 :=
by
  sorry

-- Problem I5.3
theorem ratio_parallel_lines (AP PB AC PQ : ℝ) (h1 : AP / PB = 2) (h2 : AC = 33) : PQ = 25 :=
by
  sorry

-- Problem I5.4
theorem trigonometric_expression (K : ℝ) (h1 : K = (Real.sin (65 * Real.pi / 180)) * (Real.tan (60 * Real.pi / 180))^2 / (Real.tan (30 * Real.pi / 180) * (Real.cos (30 * Real.pi / 180)) * (Real.cos (x * Real.pi / 180))) : ℝ) : K = 6 :=
by
  sorry

end sum_of_interior_angles_nth_term_arithmetic_progression_ratio_parallel_lines_trigonometric_expression_l57_57821


namespace length_of_row_of_small_cubes_l57_57658

/-!
# Problem: Calculate the length of a row of smaller cubes

A cube with an edge length of 0.5 m is cut into smaller cubes, each with an edge length of 2 mm.
Prove that the length of the row formed by arranging the smaller cubes in a continuous line 
is 31 km and 250 m.
-/

noncomputable def large_cube_edge_length_m : ℝ := 0.5
noncomputable def small_cube_edge_length_mm : ℝ := 2

theorem length_of_row_of_small_cubes :
  let length_mm := 31250000
  (31 : ℝ) * 1000 + (250 : ℝ) = length_mm / 1000 + 250 := 
sorry

end length_of_row_of_small_cubes_l57_57658


namespace index_eq_card_group_div_card_subgroup_l57_57867

theorem index_eq_card_group_div_card_subgroup
  (G : Type) [group G] [fintype G] (H : subgroup G) [finite H] [decidable_pred (∈ H)] :
  fintype.card G = fintype.card H * fintype.card (quotient_group.quotient H) :=
sorry

end index_eq_card_group_div_card_subgroup_l57_57867


namespace radius_of_inscribed_circle_l57_57655

-- Define the parameters of the problem
def R : ℝ := 6
def θ : ℝ := π / 3

-- Define what it means for the circle to be inscribed and tangent to the sector.
def inscribed_and_tangent (r : ℝ) : Prop :=
  ∃ (T : Point), distance T O = R - r ∧ (∀ (A : Point), is_tangent A T)

-- The final statement which we need to prove
theorem radius_of_inscribed_circle : ∃ r : ℝ, inscribed_and_tangent r ∧ r = 2 :=
sorry

end radius_of_inscribed_circle_l57_57655


namespace large_cube_painted_blue_l57_57297

theorem large_cube_painted_blue (n : ℕ) (hp : 1 ≤ n) 
  (hc : (6 * n^2) = (1 / 3) * 6 * n^3) : n = 3 := by
  have hh := hc
  sorry

end large_cube_painted_blue_l57_57297


namespace tan_phi_eq_neg_sqrt3_l57_57727

theorem tan_phi_eq_neg_sqrt3 (φ : ℝ) (h1 : cos (π / 2 + φ) = sqrt 3 / 2) (h2 : abs φ < π / 2) : tan φ = -sqrt 3 :=
by
  sorry

end tan_phi_eq_neg_sqrt3_l57_57727


namespace inequality_holds_l57_57140

theorem inequality_holds (x y : ℝ) (hx₀ : 0 < x) (hy₀ : 0 < y) (hxy : x + y = 1) :
  (1 / x^2 - 1) * (1 / y^2 - 1) ≥ 9 :=
sorry

end inequality_holds_l57_57140


namespace find_second_year_interest_rate_l57_57326

noncomputable def annual_interest_rate_after_second_year 
  (initial_investment : ℝ) (first_year_interest_rate : ℝ) 
  (end_of_first_year_amount : ℝ) (end_of_second_year_amount : ℝ) : ℝ :=
let s := ((end_of_second_year_amount / end_of_first_year_amount) - 1) * 100 in
s

theorem find_second_year_interest_rate 
  (initial_investment : ℝ) (first_year_interest_rate : ℝ) 
  (end_of_first_year_amount : ℝ) (end_of_second_year_amount : ℝ) 
  (interest_rate : ℝ) :
  initial_investment = 15000 →
  first_year_interest_rate = 0.10 →
  end_of_first_year_amount = 16500 →
  end_of_second_year_amount = 17850 →
  interest_rate = annual_interest_rate_after_second_year initial_investment first_year_interest_rate end_of_first_year_amount end_of_second_year_amount →
  interest_rate = 8.2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_second_year_interest_rate_l57_57326


namespace union_of_A_is_interval_l57_57129

noncomputable def A (p q : ℝ) := {x : ℝ | x^2 + p * x + q = 0}

theorem union_of_A_is_interval :
  (⋃ (p q : ℝ) (hp : -1 ≤ p ∧ p ≤ 1) (hq : -1 ≤ q ∧ q ≤ 1), A p q) = 
  Icc (- (1 + Real.sqrt 5) / 2) ((1 + Real.sqrt 5) / 2) :=
by
  sorry

end union_of_A_is_interval_l57_57129


namespace part1_part2_l57_57754

noncomputable def condition1 : Prop :=
∀ (x y : ℝ), x - 2 * y + 1 = 0

noncomputable def condition2 (p : ℝ) : Prop :=
∀ (x y : ℝ), y^2 = 2 * p * x ∧ p > 0

noncomputable def condition3 (A B : (ℝ × ℝ)) (p : ℝ) : Prop :=
dist A B = 4 * real.sqrt 15 ∧
(∀ (x y : ℝ), (x - 2 * y + 1 = 0) ∧ (y^2 = 2 * p * x))

noncomputable def condition4 (C : ℝ) : Prop :=
F = (C, 0)

noncomputable def condition5 (M N F : (ℝ × ℝ)) : Prop :=
∃ (x1 y1 x2 y2 : ℝ), F = (1, 0) ∧
M = (x1, y1) ∧ N = (x2, y2) ∧ (x1 + x2) / 2 = C ∧ (y1 + y2) / 2 = 0 ∧ (F.1 * x1 + F.2 * y1) * (F.1 * x2 + F.2 * y2) = 0

theorem part1 :
  (∃ (p : ℝ), condition1 ∧ condition2 p ∧ condition3 (1, 0) (2, 0) p) →
  (p = 2) :=
  sorry

theorem part2 :
  (∃ (M N F : (ℝ × ℝ)), condition4 4 ∧ condition5 M N F) →
  (∀ (F : ℝ × ℝ), F = (1,0) →
  F + ℝ∧
  M=(1, C) N=(2, 409)∧
  ((M-product length eqad 0)MFN) =
  (12-8 * (2))) :=
 sorry

end part1_part2_l57_57754


namespace ice_cream_cone_cost_is_5_l57_57858

noncomputable def cost_of_ice_cream_cone (x : ℝ) : Prop := 
  let total_cost_of_cones := 15 * x
  let total_cost_of_puddings := 5 * 2
  let extra_spent_on_cones := total_cost_of_cones - total_cost_of_puddings
  extra_spent_on_cones = 65

theorem ice_cream_cone_cost_is_5 : ∃ x : ℝ, cost_of_ice_cream_cone x ∧ x = 5 :=
by 
  use 5
  unfold cost_of_ice_cream_cone
  simp
  sorry

end ice_cream_cone_cost_is_5_l57_57858


namespace count_3_digit_numbers_divisible_by_9_l57_57442

theorem count_3_digit_numbers_divisible_by_9 : 
  let count := (range (integer_divisible_in_range 9 100 999)).length
  count = 100 := 
by
  sorry

noncomputable def integer_divisible_in_range (k m n : ℕ) : List ℕ :=
  let start := m / k + (if (m % k = 0) then 0 else 1)
  let end_ := n / k
  List.range (end_ - start + 1) |>.map (λ i => (start + i) * k)

noncomputable def range (xs : List ℕ) := xs

end count_3_digit_numbers_divisible_by_9_l57_57442


namespace point_G_trajectory_correctness_no_positive_real_numbers_mnγ_l57_57024

-- Definitions based on conditions in the problem
def circle (m n r : ℝ) (x y : ℝ) : Prop := (x - m)^2 + (y - n)^2 = r^2

def point_N : ℝ × ℝ := (1, 0)

def point_on_circle (m n r x y : ℝ) : Prop := circle m n r x y

def point_Q (P N Q : ℝ × ℝ) : Prop := 
  (2 * (fst N - fst P) = fst Q - fst P) ∧
  (2 * (snd N - snd P) = snd Q - snd P)

def orthogonal_vectors (G Q N P : ℝ × ℝ) : Prop :=
  ((fst G - fst Q) * (fst N - fst P) + (snd G - snd Q) * (snd N - snd P)) = 0

def equation_ellipse (x y : ℝ) : Prop := 
  (x^2) / 4 + (y^2) / 3 = 1

noncomputable def point_G_trajectory (m n r : ℝ) : Prop :=
  ∀ (x y : ℝ), point_on_circle m n r x y → equation_ellipse x y

theorem point_G_trajectory_correctness :
  point_G_trajectory (-1) 0 4 := 
  sorry

theorem no_positive_real_numbers_mnγ (m n r : ℝ) :
  (0 < m ∧ 0 < n ∧ 0 < r) → ∀ (A B : ℝ × ℝ) (M N : ℝ × ℝ), 
  (circle m n r (fst A) (snd A)) ∧ 
  (circle m n r (fst B) (snd B)) ∧ 
  (M = (m,n)) ∧ 
  (N = (1,0)) ∧ 
  (M ≠ N) ∧ 
  (∃ D, (fst D = (fst A + fst B)/2) ∧ (snd D = (snd A + snd B)/2)) → 
  ¬ (fst (1,(0 : ℝ)) = D) := 
  sorry

end point_G_trajectory_correctness_no_positive_real_numbers_mnγ_l57_57024


namespace find_k_l57_57091

variables (k : ℝ)
def vector_oa := (-3 : ℝ, 1 : ℝ)
def vector_ob := (-2 : ℝ, k)
def vector_ab := (vector_ob.1 - vector_oa.1, vector_ob.2 - vector_oa.2)

theorem find_k (h : vector_oa.1 * vector_ab.1 + vector_oa.2 * vector_ab.2 = 0) : k = 4 :=
by {
    sorry
}

end find_k_l57_57091


namespace trig_simplification_l57_57178

variable (x : ℝ)

theorem trig_simplification
  (h1 : sin x = 2 * sin (x / 2) * cos (x / 2))
  (h2 : cos x = 1 - 2 * sin (x / 2) ^ 2) :
  (1 + sin x - cos x) / (1 + sin x + cos x) = tan (x / 2) := by
  sorry

end trig_simplification_l57_57178


namespace pizza_fraction_covered_l57_57364

theorem pizza_fraction_covered :
  (∀ (d: ℝ), d = 16 → 
    (∀ (n: ℝ), n = 8 → 
      (∀ (total_pepperoni: ℕ), total_pepperoni = 32 → 
        let pepperoni_diameter := d / n in
        let pepperoni_radius := pepperoni_diameter / 2 in
        let pepperoni_area := Real.pi * (pepperoni_radius ^ 2) in
        let total_pepperoni_area := total_pepperoni * pepperoni_area in
        let pizza_radius := d / 2 in
        let pizza_area := Real.pi * (pizza_radius ^ 2) in
        total_pepperoni_area / pizza_area = 1 / 2)
    ))
  := by
    intros d d_eq n n_eq total_pepperoni total_pepperoni_eq
    let pepperoni_diameter := d / n
    let pepperoni_radius := pepperoni_diameter / 2
    let pepperoni_area := Real.pi * (pepperoni_radius ^ 2)
    let total_pepperoni_area := total_pepperoni * pepperoni_area
    let pizza_radius := d / 2
    let pizza_area := Real.pi * (pizza_radius ^ 2)
    sorry

end pizza_fraction_covered_l57_57364


namespace problem_statement_l57_57802

theorem problem_statement
  (a b c : ℝ)
  (h1 : a + 2 * b + 3 * c = 12)
  (h2 : a^2 + b^2 + c^2 = a * b + a * c + b * c) :
  a + b^2 + c^3 = 14 := 
sorry

end problem_statement_l57_57802


namespace solve_for_x_l57_57181

theorem solve_for_x (x : ℝ) : (∃ x = 168 / 5, (∛(15 * x + ∛(15 * x + 8)) = 8)) :=
by
  sorry

end solve_for_x_l57_57181


namespace count_permutations_between_23145_and_43521_l57_57319

open List

def num_of_valid_permutations : ℕ :=
  length (filter (λ x, 23145 < x ∧ x < 43521) (map (λ l, foldl (λ acc d, 10 * acc + d) 0 l) (perm {1, 2, 3, 4, 5})))

theorem count_permutations_between_23145_and_43521 : num_of_valid_permutations = 58 :=
by sorry

end count_permutations_between_23145_and_43521_l57_57319


namespace solve_for_x_l57_57635

theorem solve_for_x (x y z : ℤ) (h1 : x + y + z = 14) (h2 : x - y - z = 60) (h3 : x + z = 2 * y) : x = 37 := by
  sorry

end solve_for_x_l57_57635


namespace min_value_of_angle_function_l57_57817

theorem min_value_of_angle_function (α β γ : ℝ) (h1 : α + β + γ = Real.pi) (h2 : 0 < α) (h3 : α < Real.pi) :
  ∃ α, α = (2 * Real.pi / 3) ∧ (4 / α + 1 / (Real.pi - α)) = (9 / Real.pi) := by
  sorry

end min_value_of_angle_function_l57_57817


namespace count_3_digit_numbers_divisible_by_9_l57_57453

theorem count_3_digit_numbers_divisible_by_9 : 
  (finset.filter (λ x : ℕ, x % 9 = 0) (finset.Icc 100 999)).card = 100 := 
sorry

end count_3_digit_numbers_divisible_by_9_l57_57453


namespace one_fourth_more_than_x_equals_twenty_percent_less_than_80_l57_57606

theorem one_fourth_more_than_x_equals_twenty_percent_less_than_80 :
  ∃ n : ℝ, (80 - 0.30 * 80 = 56) ∧ (5 / 4 * n = 56) ∧ (n = 45) :=
by
  sorry

end one_fourth_more_than_x_equals_twenty_percent_less_than_80_l57_57606


namespace difference_in_edge_lengths_of_equilateral_triangles_l57_57200

noncomputable def eq_triangle_edge_diff : ℝ × ℝ × ℝ := (a, b, √3)

theorem difference_in_edge_lengths_of_equilateral_triangles (a b : ℝ) (h1 : ∀ x y, x ∈ S → y ∈ T → dist x y = √3) :
  abs (a - b) = 6 := 
sorry

end difference_in_edge_lengths_of_equilateral_triangles_l57_57200


namespace angle_DBQ_eq_angle_PAC_l57_57018

-- Definitions for the geometric setup
variables (P A B C D Q : Point)
variable (circle : Circle)

-- Assumptions based on the problem
variables (is_outside : P ∉ circle)
variables (is_tangent_PA : is_tangent P A circle)
variables (is_tangent_PB : is_tangent P B circle)
variables (is_secant : is_secant_through P C D circle)
variable (C_between_P_D : between P C D)
variable (on_chord_Q : on_chord Q C D circle)
variables (angle_DAQ_eq_angle_PBC : angle (D, A, Q) = angle (P, B, C))

-- Theorem to be proved
theorem angle_DBQ_eq_angle_PAC : angle (D, B, Q) = angle (P, A, C) :=
sorry

end angle_DBQ_eq_angle_PAC_l57_57018


namespace election_total_votes_l57_57959

theorem election_total_votes
  (V : ℕ)
  (winner_votes : ℕ)
  (runner_up_votes : ℕ)
  (remaining_votes : ℕ)
  (h1 : winner_votes = 0.45 * V)
  (h2 : runner_up_votes = 0.28 * V)
  (h3 : remaining_votes = 0.27 * V)
  (h4 : winner_votes - runner_up_votes = 550) :
  V ≈ 3235 :=
by 
  sorry

end election_total_votes_l57_57959


namespace abs_iff_sq_gt_l57_57078

theorem abs_iff_sq_gt (x y : ℝ) : (|x| > |y|) ↔ (x^2 > y^2) :=
by sorry

end abs_iff_sq_gt_l57_57078


namespace parameter_a_range_l57_57002

variable (a b x y : ℝ)

def system_of_equations := 
  arcsin ((a - y) / 3) = arcsin ((4 - x) / 4) ∧ 
  x^2 + y^2 - 8*x - 8*y = b
  
theorem parameter_a_range : 
  (∃ b, ∃ x y, system_of_equations a b x y) ↔ 
  (a > -13/3 ∧ a < 37/3) :=
by 
  sorry

end parameter_a_range_l57_57002


namespace fisherman_gets_14_tunas_every_day_l57_57563

-- Define the conditions
def red_snappers_per_day := 8
def cost_per_red_snapper := 3
def cost_per_tuna := 2
def total_earnings_per_day := 52

-- Define the hypothesis
def total_earnings_from_red_snappers := red_snappers_per_day * cost_per_red_snapper  -- $24
def total_earnings_from_tunas := total_earnings_per_day - total_earnings_from_red_snappers -- $28
def number_of_tunas := total_earnings_from_tunas / cost_per_tuna -- 14

-- Lean statement to verify
theorem fisherman_gets_14_tunas_every_day : number_of_tunas = 14 :=
by 
  sorry

end fisherman_gets_14_tunas_every_day_l57_57563


namespace triangle_ratio_l57_57476

theorem triangle_ratio (A B C G H P : Type)
  [inst: Nonempty A] [inst: Nonempty B] [inst: Nonempty C]
  [inst: Nonempty G] [inst: Nonempty H] [inst: Nonempty P]
  (on_line_AB : G ∈ [A, B])
  (on_line_BC : H ∈ [B, C])
  (intersect : (AG ∩ CH).P)
  (AP_PG_eq : ratio (length A P) (length P G) = 5 : ℝ)
  (CP_PH_eq : ratio (length C P) (length P H) = 3 : ℝ) :
  ratio (length B H) (length H C) = 3 / 7 := 
sorry

end triangle_ratio_l57_57476


namespace find_x_l57_57741

theorem find_x (x : ℝ) (h1: 2 * sin x * tan x = 3) (h2: -π < x) (h3: x < 0) : 
  x = -π / 3 :=
  sorry

end find_x_l57_57741


namespace min_value_l57_57033

theorem min_value (a b : ℝ) (h₁ : a + b = 2) (h₂ : 0 < a) (h₃ : 0 < b) : 
  ∃ a : ℝ, ∃ b : ℝ, a + b = 2 ∧ 0 < a ∧ 0 < b ∧ (a = 4/3 ∧ (1 / a + a / (8 * b) = 1)) :=
by
  use (4 / 3)
  split
  { sorry }
  use (2 - 4 / 3)
  repeat { split, sorry }

end min_value_l57_57033


namespace max_area_external_tangency_max_area_internal_tangency_l57_57970

-- Definitions
variables (A B C D E : Type) 
variables (d1 d2 : ℝ)
variables (m1 : ℝ) 

-- External tangency theorem statement
theorem max_area_external_tangency (h_tangent : tangent A B C D E) 
  (h_diameters : diameters d1 d2)
  (h_angle : angle A 45) :
  max_area (quadrilateral B C D E) = (d1 + d2)^2 / 4 :=
sorry

-- Internal tangency theorem statement
theorem max_area_internal_tangency (h_tangent : tangent A B C D E) 
  (h_diameters : diameters d1 d2)
  (h_angle : angle A 45) :
  max_area (quadrilateral B C D E) = (d2^2 - d1^2) / 4 :=
sorry

end max_area_external_tangency_max_area_internal_tangency_l57_57970


namespace episode_length_l57_57144

/-- Subject to the conditions provided, we prove the length of each episode watched by Maddie. -/
theorem episode_length
  (total_episodes : ℕ)
  (monday_minutes : ℕ)
  (thursday_minutes : ℕ)
  (weekend_minutes : ℕ)
  (episodes_length : ℕ)
  (monday_watch : monday_minutes = 138)
  (thursday_watch : thursday_minutes = 21)
  (weekend_watch : weekend_minutes = 105)
  (total_episodes_watch : total_episodes = 8)
  (total_minutes : monday_minutes + thursday_minutes + weekend_minutes = total_episodes * episodes_length) :
  episodes_length = 33 := 
by 
  sorry

end episode_length_l57_57144


namespace sum_f_log_a_eq_99_over_2_l57_57415

noncomputable def f (x : ℝ) : ℝ := 3^x / (1 + 3^x)

variable (a : ℕ → ℝ)
variable [IsPositiveSequence a] -- Custom predicate ensuring sequence is positive geometric
variable (h_a50 : a 50 = 1)
variable (h_geom : ∀ n : ℕ, 49 ≤ n → a n * a (100 - n) = 1)

theorem sum_f_log_a_eq_99_over_2 
  (h_pos_seq : IsPositiveSequence a) -- Ensuring a is a positive sequence
  (h_a50 : a 50 = 1) 
  (h_geom : ∀ n : ℕ, 1 ≤ n → n ≤ 48 → a n * a (100 - n) = 1) : 
  (∑ i in finset.range 99, f (real.log (a (i + 1)))) = 99 / 2 :=
by
  sorry

end sum_f_log_a_eq_99_over_2_l57_57415


namespace fewer_popsicle_sticks_l57_57924

theorem fewer_popsicle_sticks :
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  boys_total - girls_total = 6 := 
by
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  show boys_total - girls_total = 6
  sorry

end fewer_popsicle_sticks_l57_57924


namespace inequality_sqrt3_abc_l57_57731

theorem inequality_sqrt3_abc (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_ineq : a * b * c ≤ a + b + c) : 
  a^2 + b^2 + c^2 ≥ real.sqrt 3 * (a * b * c) :=
by {
  sorry
}

end inequality_sqrt3_abc_l57_57731


namespace not_necessarily_periodic_l57_57735

-- Define the conditions of the problem
noncomputable def a : ℕ → ℕ := sorry
noncomputable def t : ℕ → ℕ := sorry
axiom h_t : ∀ k : ℕ, ∃ t_k : ℕ, ∀ n : ℕ, a (k + n * t_k) = a k

-- The theorem stating that the sequence is not necessarily periodic
theorem not_necessarily_periodic : ¬ ∃ T : ℕ, ∀ k : ℕ, a (k + T) = a k := sorry

end not_necessarily_periodic_l57_57735


namespace integral_fK_eq_l57_57402

noncomputable def f (x : ℝ) : ℝ := 1 / x

def fK (x : ℝ) : ℝ :=
  if f x ≤ 1 then 1 else f x

theorem integral_fK_eq :
  ∫ x in (1 / 4)..2, fK x = 2 * Real.log 2 + 1 := by
  sorry

end integral_fK_eq_l57_57402


namespace largest_angle_in_scalene_triangle_l57_57486

namespace TriangleProof

variables (x : ℝ)

-- Assumptions
def is_scalene (a b c : ℝ) := a ≠ b ∧ b ≠ c ∧ a ≠ c
axiom sum_of_angles (a b c : ℝ) : a + b + c = 180
axiom angle1 : ℝ := 50
axiom angle2 (x : ℝ) : ℝ := x
axiom angle3 (x : ℝ) : ℝ := x + 10

-- Theorem statement
theorem largest_angle_in_scalene_triangle :
  ∃ x, is_scalene (angle1 :ℝ) (angle2 x) (angle3 x) ∧ sum_of_angles (angle1 :ℝ) (angle2 x) (angle3 x) 
  → max (angle1 :ℝ) (max (angle2 x) (angle3 x)) = 70 :=
by
  sorry

end TriangleProof

end largest_angle_in_scalene_triangle_l57_57486


namespace cos_a_sub_pi_l57_57742

theorem cos_a_sub_pi (a : ℝ) (h1 : π / 2 < a) (h2 : a < π) (h3 : 3 * sin (2 * a) = 2 * cos a) : 
  cos (a - π) = 2 * sqrt 2 / 3 := 
sorry

end cos_a_sub_pi_l57_57742


namespace ant_walk_distance_l57_57613

noncomputable def radius_large : ℝ := 15
noncomputable def radius_small : ℝ := 5

def arc_distance_large : ℝ := (1 / 4) * (2 * Real.pi * radius_large)
def radial_distance_inward : ℝ := radius_large - radius_small
def arc_distance_small : ℝ := (1 / 2) * (2 * Real.pi * radius_small)
def radial_distance_outward : ℝ := radius_large - radius_small

def total_walk_distance : ℝ :=
  arc_distance_large + radial_distance_inward + arc_distance_small + radial_distance_outward

theorem ant_walk_distance :
  total_walk_distance = (12.5 * Real.pi + 20) :=
by
  sorry

end ant_walk_distance_l57_57613


namespace geoms_seq_find_a1_d_l57_57743

noncomputable def an (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ := a1 + n * d
noncomputable def bn (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ := 1 / (an a1 d (2^n))

theorem geoms_seq (a1 : ℝ) (d : ℝ) (h_log : log (an a1 d 2) = (log (an a1 d 0) + log (an a1 d 4)) / 2)
  (h_pos : ∀ n : ℕ, an a1 d n > 0) :
  geometric_seq (λ n : ℕ, bn a1 d n) :=
sorry

theorem find_a1_d (h_sum : ∀ (a1 : ℝ) (d : ℝ), (bn a1 d 1 + bn a1 d 2 + bn a1 d 3 = 7 / 24)) :
  ∃ (a1 : ℝ) (d : ℝ), (a1 = 72 / 7 ∧ d = 0) ∨ (a1 = 3 ∧ d = 3) :=
sorry

end geoms_seq_find_a1_d_l57_57743


namespace min_sticks_12_to_break_can_form_square_15_l57_57591

-- Problem definition for n = 12
def sticks_12 : List Nat := List.range' 1 12

theorem min_sticks_12_to_break : 
  ... (I realize I need to translate a step better) ..............
  sorry

-- Problem definition for n = 15
def sticks_15 : List Nat := List.range' 1 15

theorem can_form_square_15 : 
  ... (implementing a nice explanation)
  sorry

end min_sticks_12_to_break_can_form_square_15_l57_57591


namespace area_ratio_of_quad_to_triangle_l57_57131

noncomputable def ratio_area_quad_to_triangle
  (A B C D P : Vect) [InnerProductSpace ℝ Vect] 
  (h : P - A + 3 * (P - B) + 4 * (P - C) + 5 * (P - D) = 0) :
  real :=
13 / 8

theorem area_ratio_of_quad_to_triangle
  (A B C D P : Vect) [InnerProductSpace ℝ Vect]
  (h : P - A + 3 * (P - B) + 4 * (P - C) + 5 * (P - D) = 0) :
  ratio_area_quad_to_triangle A B C D P h = 13 / 8 :=
sorry

end area_ratio_of_quad_to_triangle_l57_57131


namespace range_of_a_l57_57947

variable (a : ℝ)

theorem range_of_a (ha : a ≥ 1/4) : ¬ ∃ x : ℝ, a * x^2 + x + 1 < 0 := sorry

end range_of_a_l57_57947


namespace find_vec_at_neg3_l57_57663

-- Define conditions
def vec_t (t : ℝ) (a d : ℝ × ℝ) : ℝ × ℝ := (a.1 + t * d.1, a.2 + t * d.2)

-- Given conditions
def t1_vec := (4, 5) : ℝ × ℝ
def t5_vec := (12, -11) : ℝ × ℝ

-- Target vector on the line at t = -3
def target_vec := (-4, 21) : ℝ × ℝ

-- Proof goal
theorem find_vec_at_neg3 (a d : ℝ × ℝ) 
  (h1 : vec_t 1 a d = t1_vec) 
  (h5 : vec_t 5 a d = t5_vec) : 
  vec_t (-3) a d = target_vec := 
sorry

end find_vec_at_neg3_l57_57663


namespace proof_problem_l57_57054

noncomputable def f (x : ℝ) : ℝ := 2 * sin (ω * x + φ - π / 6) 

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem proof_problem (ω φ : ℝ) (f : ℝ → ℝ) (Hω : ω > 0) (Hφ : 0 < φ ∧ φ < π) 
(H_even : is_even f) (H_sym : ∀ x, f(x) = f(2k * π - x)) :

  (f = λ x, 2 * cos (2 * x)) ∧ (∀ x ∈ Icc (-π / 8) (3 * π / 8), f x ∈ Icc (-sqrt 2 + 1) 3) := 
sorry

end proof_problem_l57_57054


namespace weighing_error_probability_l57_57255

theorem weighing_error_probability :
  ∀ (X : ℝ → ℝ) (σ : ℝ),
  (∀ x, X x = pdf_normal 0 σ x) →
  σ = 20 →
  ∫ x in -10..10, X x = 0.383 :=
begin
  intros X σ X_norm σ_value,
  sorry,
end

end weighing_error_probability_l57_57255


namespace fraction_numerator_greater_than_denominator_l57_57218

theorem fraction_numerator_greater_than_denominator {x : ℝ} : 
  -1 ≤ x ∧ x ≤ 3 ∧ 5 * x + 2 > 8 - 3 * x ↔ (3 / 4) < x ∧ x ≤ 3 :=
by 
  sorry

end fraction_numerator_greater_than_denominator_l57_57218


namespace solution_set_of_inequality_group_l57_57951

theorem solution_set_of_inequality_group (x : ℝ) : (x > -3 ∧ x < 5) ↔ (-3 < x ∧ x < 5) :=
by
  sorry

end solution_set_of_inequality_group_l57_57951


namespace inverse_function_solution_l57_57138

noncomputable def f (a b x : ℝ) := 2 / (a * x + b)

theorem inverse_function_solution (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : f a b 2 = 1 / 2) : b = 1 - 2 * a :=
by
  -- Assuming the inverse function condition means f(2) should be evaluated.
  sorry

end inverse_function_solution_l57_57138


namespace petya_password_l57_57902

/-- Petya is creating a password for his smartphone. The password consists of 4 decimal digits.
Petya wants the password to not contain the digit 7, and it must have at least two identical digits.
We are to prove that the number of ways Petya can create such a password is 3537. -/
theorem petya_password : 
  let total_passwords := 9^4,
      all_distinct_passwords := (Finset.card (Finset.powersetLen 4 (Finset.range 9))).choose 4 * factorial 4 in
  total_passwords - all_distinct_passwords = 3537 :=
by
  sorry

end petya_password_l57_57902


namespace value_of_f_neg_5_over_2_l57_57873

def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
else if -1 ≤ x ∧ x < 0 then 2 * (-x) * (1 + x)
else f (x % 2) -- uses periodicity of 2

theorem value_of_f_neg_5_over_2 : f (-5/2) = 1/2 :=
by sorry

end value_of_f_neg_5_over_2_l57_57873


namespace decreasing_interval_maximum_on_interval_l57_57413

open Real

-- Definition of the function f: ℝ → ℝ
noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- 1. Prove that f(x) is decreasing on (0, 2)
theorem decreasing_interval :
  ∀ x ∈ Ioo 0 2, deriv f x < 0 :=
by
  sorry

-- 2. Prove that the maximum value of f(x) on [-4, 3] is 0
theorem maximum_on_interval :
  ∃ y ∈ Icc (-4 : ℝ) 3, ∀ z ∈ Icc (-4 : ℝ) 3, f z ≤ f y ∧ f y = 0 :=
by
  sorry

end decreasing_interval_maximum_on_interval_l57_57413


namespace subway_max_lines_l57_57094

noncomputable def max_lines_subway (lines : Type) (stations : Type) 
  (has_station : lines → set stations) (is_transfer_station : stations → Prop) : ℕ :=
    sorry

theorem subway_max_lines (lines : Type) (stations : Type) 
  (has_station : lines → set stations) (is_transfer_station : stations → Prop)
  (h1 : ∀ l : lines, 4 ≤ (has_station l).card) 
  (h2 : ∃ t1 t2 t3 : stations, is_transfer_station t1 ∧ is_transfer_station t2 ∧ is_transfer_station t3 ∧
      ∀ t : stations, is_transfer_station t → t = t1 ∨ t = t2 ∨ t = t3)
  (h3 : ∀ t : stations, is_transfer_station t → ∃ l1 l2 : lines, l1 ≠ l2 ∧ t ∈ has_station l1 ∧ t ∈ has_station l2) 
  (h4 : ∀ s1 s2 : stations, ∃ l1 l2 : lines, (s1 ∈ has_station l1) ∧ (s2 ∈ has_station l1) ∨ 
      ((s1 ∈ has_station l1) ∧ (s2 ∈ has_station l2) ∧ (∃ t : stations, is_transfer_station t ∧ t ∈ has_station l1 ∧ t ∈ has_station l2)) ∨
      ((∃ l3 : lines, (s1 ∈ has_station l1) ∧ (s2 ∈ has_station l3) ∧ 
        (∃ t : stations, is_transfer_station t ∧ t ∈ has_station l1 ∧ t ∈ has_station l2 ∧ t ∈ has_station l3))) : 
    ∃ n : ℕ, n = 10 ∧ max_lines_subway lines stations has_station is_transfer_station = n :=
    sorry

end subway_max_lines_l57_57094


namespace muffin_banana_ratio_l57_57189

variables (m b : ℝ)

theorem muffin_banana_ratio (h1 : 4 * m + 3 * b = x) 
                            (h2 : 2 * (4 * m + 3 * b) = 2 * m + 16 * b) : 
                            m / b = 5 / 3 :=
by sorry

end muffin_banana_ratio_l57_57189


namespace area_of_DKM_l57_57833

namespace Geometry

-- Define the equilateral triangle
variables (ΔABC : Triangle) (a : ℝ)
(h_equilateral : IsEquilateralTriangle ΔABC a)

-- Define the circles and tangents
variables (R r : ℝ) (O Q : Point) (D K M : Point)
(h_incircle : IsIncircle ΔABC R O D)
(h_inner_circle : IsInnerCircle ΔABC r Q K M)

-- Define the areas of the segments
def segment_area (R : ℝ) : ℝ := (π * R^2) / 6 - (R^2 * (sqrt 3)) / 4

-- Define the target area to prove
def target_area (a : ℝ) : ℝ := (a^2 * (24 * sqrt 3 - 11 * π)) / 648

-- State the theorem
theorem area_of_DKM (a : ℝ) (R r : ℝ) (O Q D K M : Point)
  (h_equilateral : IsEquilateralTriangle ΔABC a)
  (h_incircle : IsIncircle ΔABC R O D)
  (h_inner_circle : IsInnerCircle ΔABC r Q K M) :
  area_DKM (ΔABC a R r O Q D K M) = target_area a :=
by sorry

end Geometry

end area_of_DKM_l57_57833


namespace find_x_from_equation_l57_57987

theorem find_x_from_equation :
  ∃ x : ℝ, 4^5 + 4^5 + 4^5 = 2^x ∧ x = 10 + Real.log2 3 :=
by
  sorry

end find_x_from_equation_l57_57987


namespace start_time_is_10_am_l57_57938

-- Definitions related to the problem statements
def distance_AB : ℝ := 600
def speed_A_to_B : ℝ := 70
def speed_B_to_A : ℝ := 80
def meeting_time : ℝ := 14  -- using 24-hour format, 2 pm as 14

-- Prove that the starting time is 10 am given the conditions
theorem start_time_is_10_am (t : ℝ) :
  (speed_A_to_B * t + speed_B_to_A * t = distance_AB) →
  (meeting_time - t = 10) :=
sorry

end start_time_is_10_am_l57_57938


namespace communication_system_even_n_l57_57090

theorem communication_system_even_n (n : ℕ) (subscribers : ℕ := 2001) 
  (connections : ∀ s, s < subscribers → ℕ := λ s, n) :
  2001 * n % 2 = 0 :=
sorry

end communication_system_even_n_l57_57090


namespace sum_of_digits_at_positions_l57_57328

theorem sum_of_digits_at_positions :
  let initial_sequence := (List.repeat [1, 2, 3, 4, 5, 6] (12000 / 6)).join,
      erase_every_nth (l : List ℕ) (n : ℕ) : List ℕ := List.filter_map_with_index 
        (λ i x => if (i + 1) % n = 0 then none else some x) l,
      after_first_erasure := erase_every_nth initial_sequence 2,
      after_second_erasure := erase_every_nth after_first_erasure 3,
      final_sequence := erase_every_nth after_second_erasure 4,
      pos_2999 := final_sequence.nth! (2999 - 1),
      pos_3000 := final_sequence.nth! (3000 - 1),
      pos_3001 := final_sequence.nth! (3001 - 1)
  in pos_2999 + pos_3000 + pos_3001 = 5 := by sorry

end sum_of_digits_at_positions_l57_57328


namespace players_started_first_half_l57_57672

variable (total_players : Nat)
variable (first_half_substitutions : Nat)
variable (second_half_substitutions : Nat)
variable (players_not_playing : Nat)

theorem players_started_first_half :
  total_players = 24 →
  first_half_substitutions = 2 →
  second_half_substitutions = 2 * first_half_substitutions →
  players_not_playing = 7 →
  let total_substitutions := first_half_substitutions + second_half_substitutions 
  let players_played := total_players - players_not_playing
  ∃ S, S + total_substitutions = players_played ∧ S = 11 := 
by
  sorry

end players_started_first_half_l57_57672


namespace option_C_represents_a_option_A_not_representing_a_option_B_not_representing_a_option_D_not_representing_a_l57_57497

def vector (α : Type _) := EuclideanSpace (Fin 2) α

def a : vector ℝ := ![-3, 7]

def e1_A : vector ℝ := ![0, 1]
def e2_A : vector ℝ := ![0, -2]

def e1_B : vector ℝ := ![1, 5]
def e2_B : vector ℝ := ![-2, -10]

def e1_C : vector ℝ := ![-5, 3]
def e2_C : vector ℝ := ![-2, 1]

def e1_D : vector ℝ := ![7, 8]
def e2_D : vector ℝ := ![-7, -8]

theorem option_C_represents_a :
  ∃ (λ μ : ℝ), a = λ • e1_C + μ • e2_C :=
by { 
  use [11, -26], 
  sorry 
}

theorem option_A_not_representing_a :
  ¬∃ (λ μ : ℝ), a = λ • e1_A + μ • e2_A :=
by {
  sorry 
}

theorem option_B_not_representing_a :
  ¬∃ (λ μ : ℝ), a = λ • e1_B + μ • e2_B :=
by {
  sorry
}

theorem option_D_not_representing_a :
  ¬∃ (λ μ : ℝ), a = λ • e1_D + μ • e2_D :=
by {
  sorry
}

end option_C_represents_a_option_A_not_representing_a_option_B_not_representing_a_option_D_not_representing_a_l57_57497


namespace inequality_positive_reals_l57_57374

theorem inequality_positive_reals (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (x / (1 + x^2)) + (y / (1 + y^2)) + (z / (1 + z^2)) ≤ (3 * Real.sqrt 3) / 4 := by
  sorry

end inequality_positive_reals_l57_57374


namespace ellipse_equation_and_lambda_mu_constant_l57_57881

theorem ellipse_equation_and_lambda_mu_constant :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (eccentricity : ℝ) = 2 * real.sqrt 2 / 3 ∧
  (circle_radius : ℝ) = 3 ∧ -- radius obtained from the given circle equation x^2 + y^2 = 9
  (equation : (a : ℝ) (b : ℝ) -> Prop)
    (equation 9 1) ∧ -- (a^2 = 9 and b^2 = 1 which formats as the equation of ellipse)
  (∃ (Q : ℝ × ℝ) (RM MQ NQ : ℝ × ℝ) (λ μ : ℝ), Q = (1,0) ∧ 
     λ + μ = -9 / 4) :=
sorry

end ellipse_equation_and_lambda_mu_constant_l57_57881


namespace ceil_floor_expression_l57_57335

theorem ceil_floor_expression : 
  let a := (18 : ℚ) / 11
  let b := (-33 : ℚ) / 4 
  let term1 := a * b
  let term2 := a * (floor b)
  ⌈term1⌉ - ⌊term2⌋ = 2 := 
begin
  sorry
end

end ceil_floor_expression_l57_57335


namespace min_sticks_to_be_broken_form_square_without_breaks_l57_57593

noncomputable def total_length (n : ℕ) : ℕ := n * (n + 1) / 2

def divisible_by_4 (x : ℕ) : Prop := x % 4 = 0

theorem min_sticks_to_be_broken (n : ℕ) : n = 12 → (¬ divisible_by_4 (total_length n)) ∧ (minimal_breaks n = 2) :=
by
  intro h1
  rw h1
  have h2 : total_length 12 = 78 := by decide
  have h3 : ¬ divisible_by_4 78 := by decide
  exact ⟨h3, sorry⟩

theorem form_square_without_breaks (n : ℕ) : n = 15 → divisible_by_4 (total_length n) ∧ (minimal_breaks n = 0) :=
by
  intro h1
  rw h1
  have h2 : total_length 15 = 120 := by decide
  have h3 : divisible_by_4 120 := by decide
  exact ⟨h3, sorry⟩

end min_sticks_to_be_broken_form_square_without_breaks_l57_57593


namespace line_passing_through_point_parallel_to_another_line_l57_57004

def line_equation_passing_through_and_parallel (p : ℝ × ℝ) (c₁ c₂ : ℝ) 
    (h₁ : p = (2, -4)) (h₂ : c₁ = 1) (h₃ : c₂ = -6) : Prop :=
  ∃ a b c : ℝ, a * p.1 + b * p.2 + c = 0 ∧ a * 2 - b = 6 ∧ c₂ = -6

theorem line_passing_through_point_parallel_to_another_line :
  line_equation_passing_through_and_parallel (2, -4) 1 (-6)
    (by refl) (by rfl) (by rfl) :=
sorry

end line_passing_through_point_parallel_to_another_line_l57_57004


namespace parker_total_weight_l57_57548

-- Define the number of initial dumbbells and their weight
def initial_dumbbells := 4
def weight_per_dumbbell := 20

-- Define the number of additional dumbbells
def additional_dumbbells := 2

-- Define the total weight calculation
def total_weight := initial_dumbbells * weight_per_dumbbell + additional_dumbbells * weight_per_dumbbell

-- Prove that the total weight is 120 pounds
theorem parker_total_weight : total_weight = 120 :=
by
  -- proof skipped
  sorry

end parker_total_weight_l57_57548


namespace number_of_correct_statements_l57_57209

noncomputable def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

axiom random_variable_X_distribution {σ : ℝ} : 
  ∀(X : ℝ → ℝ), X = normal_distribution 2 σ

axiom probability_X_lt_5 {σ : ℝ} :
  ∀(P : ℝ → ℝ), P (λ X, X < 5) = 0.8

axiom regression_equation :
  ∀(x y : ℝ), y = 0.85 * x - 82

axiom bags_setup :
  ∀ (whites_A blacks_A : ℕ) (whites_B blacks_B : ℕ),
    whites_A = 3 ∧ blacks_A = 2 ∧ whites_B = 4 ∧ blacks_B = 4

axiom probability_white_ball :
  ∀ (P : ℕ → ℝ), P (λ _, true) to_finite = 13/25

theorem number_of_correct_statements : 
  (statement_1_correct → statement_2_correct → statement_3_correct → ∃ n = 2) := 
sorry

end number_of_correct_statements_l57_57209


namespace exp_comparison_l57_57340

noncomputable def exp1 : ℝ := 0.2 ^ 3
noncomputable def exp2 : ℝ := 2 ^ 0.3

theorem exp_comparison : exp1 < exp2 := by
  sorry

end exp_comparison_l57_57340


namespace sum_of_number_and_reverse_l57_57201

theorem sum_of_number_and_reverse (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) 
(h : (10 * a + b) - (10 * b + a) = 3 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 33 := 
sorry

end sum_of_number_and_reverse_l57_57201


namespace intersection_length_l57_57628

noncomputable def pointA_polar := (2, Real.pi / 6)

noncomputable def lineL_polar_angle := Real.pi / 3

noncomputable def circleC_polar_eq (θ : ℝ) : ℝ :=
  √2 * Real.cos (θ - Real.pi / 4)

theorem intersection_length :
  let A := (Real.sqrt 3, 0)
  ∃ t1 t2 : ℝ, 
  let l_params := λ t, (Real.sqrt 3 + t / 2, 1 + t * (Real.sqrt 3 / 2)) in
  ∀ (B C : ℝ × ℝ), 
  (B = l_params t1 ∧ B ≠ A ∧ C = l_params t2 ∧ C ≠ A ∧
  (B.1 ^ 2 + B.2 ^ 2 - B.1 - B.2 = 0) ∧ 
  (C.1 ^ 2 + C.2 ^ 2 - C.1 - C.2 = 0)) → 
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = t1 ^ 2 ∧
  (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2 = t2 ^ 2 ∧
  (|B.1 - A.1| * |C.1 - A.1| + |B.2 - A.2| * |C.2 - A.2| = 3 - Real.sqrt 3) :=
begin
  sorry
end

end intersection_length_l57_57628


namespace range_of_a_l57_57038

def f (x : ℝ) : ℝ :=
  if x >= 0 then 1 - 3^x else -1 + 3^(-x)

theorem range_of_a (a : ℝ) : (∀ x ∈ set.Icc 2 8, f (Real.log x / Real.log 2)^2 + f (5 - a * Real.log x / Real.log 2) ≥ 0) ↔ a ≥ 6 :=
  sorry

end range_of_a_l57_57038


namespace part_a_part_b_l57_57899

noncomputable def probability_Peter_satisfied : ℚ :=
  let total_people := 100
  let men := 50
  let women := 50
  let P_both_men := (men - 1 : ℚ)/ (total_people - 1 : ℚ) * (men - 2 : ℚ)/ (total_people - 2 : ℚ)
  1 - P_both_men

theorem part_a : probability_Peter_satisfied = 25 / 33 := 
  sorry

noncomputable def expected_satisfied_men : ℚ :=
  let men := 50
  probability_Peter_satisfied * men

theorem part_b : expected_satisfied_men = 1250 / 33 := 
  sorry

end part_a_part_b_l57_57899


namespace susan_arrives_before_sam_by_14_minutes_l57_57920

theorem susan_arrives_before_sam_by_14_minutes (d : ℝ) (susan_speed sam_speed : ℝ) (h1 : d = 2) (h2 : susan_speed = 12) (h3 : sam_speed = 5) : 
  let susan_time := d / susan_speed
  let sam_time := d / sam_speed
  let susan_minutes := susan_time * 60
  let sam_minutes := sam_time * 60
  sam_minutes - susan_minutes = 14 := 
by
  sorry

end susan_arrives_before_sam_by_14_minutes_l57_57920


namespace length_NM_constant_l57_57656

-- Define a circle with chord AB
variable {circle : Type}
variable [metric_space circle] [normed_space ℝ circle]

-- Define points A, B, W, C, X, Y, N, M as elements of the circle
variable (A B W C X Y N M : circle)

-- Assumptions about the segments and points
variable (AB_length : ℝ)
variable (mid_W : midpoint A B W)
variable (point_on_major_arc : ∀ C : circle, major_arc A B C)
variable (tangent_A : tangent A X)
variable (tangent_B : tangent B Y)
variable (tangent_C : tangent C X)
variable (intersect_WX_AB : intersect WX AB N)
variable (intersect_WY_AB : intersect WY AB M)

theorem length_NM_constant (C : circle) :
  dist N M = AB_length / 2 :=
sorry

end length_NM_constant_l57_57656


namespace ramu_selling_price_l57_57170

theorem ramu_selling_price :
  let purchase_price := 48000 : ℝ
  let repair_cost := 14000 : ℝ
  let total_cost := purchase_price + repair_cost
  let profit_percent := 17.580645161290324 : ℝ
  let profit := (profit_percent / 100) * total_cost
  let selling_price := total_cost + profit
  selling_price = 72900 :=
by
  let purchase_price := 48000 : ℝ
  let repair_cost := 14000 : ℝ
  let total_cost := purchase_price + repair_cost
  let profit_percent := 17.580645161290324 : ℝ
  let profit := (profit_percent / 100) * total_cost
  let selling_price := total_cost + profit
  sorry

end ramu_selling_price_l57_57170


namespace renovation_days_l57_57279

/-
Conditions:
1. Cost to hire a company: 50000 rubles
2. Cost of buying materials: 20000 rubles
3. Husband's daily wage: 2000 rubles
4. Wife's daily wage: 1500 rubles
Question:
How many workdays can they spend on the renovation to make it more cost-effective?
-/

theorem renovation_days (cost_hire_company cost_materials : ℕ) 
  (husband_daily_wage wife_daily_wage : ℕ) 
  (more_cost_effective_days : ℕ) :
  cost_hire_company = 50000 → 
  cost_materials = 20000 → 
  husband_daily_wage = 2000 → 
  wife_daily_wage = 1500 → 
  more_cost_effective_days = 8 :=
by
  intros
  sorry

end renovation_days_l57_57279


namespace apples_left_to_eat_raw_l57_57889

variable (n : ℕ) (picked : n = 85) (wormy : n / 5) (bruised : wormy + 9)

theorem apples_left_to_eat_raw (h_picked : n = 85) (h_wormy : wormy = n / 5) (h_bruised : bruised = wormy + 9) : n - (wormy + bruised) = 42 := 
sorry

end apples_left_to_eat_raw_l57_57889


namespace impossible_grid_arrangement_l57_57115

theorem impossible_grid_arrangement :
  ¬ ∃ (f : Fin 25 → Fin 41 → ℤ),
    (∀ i j, abs (f i j - f (i + 1) j) ≤ 16 ∧ abs (f i j - f i (j + 1)) ≤ 16 ∧
            f i j ≠ f (i + 1) j ∧ f i j ≠ f i (j + 1)) := 
sorry

end impossible_grid_arrangement_l57_57115


namespace common_difference_of_arithmetic_sequence_l57_57046

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 0 = 2) 
  (h2 : ∀ n, a (n+1) = a n + d)
  (h3 : a 9 = 20): 
  d = 2 := 
by
  sorry

end common_difference_of_arithmetic_sequence_l57_57046


namespace jerome_contact_list_count_l57_57120

theorem jerome_contact_list_count :
  (let classmates := 20
   let out_of_school_friends := classmates / 2
   let family := 3 -- two parents and one sister
   let total_contacts := classmates + out_of_school_friends + family
   total_contacts = 33) :=
by
  let classmates := 20
  let out_of_school_friends := classmates / 2
  let family := 3
  let total_contacts := classmates + out_of_school_friends + family
  show total_contacts = 33
  sorry

end jerome_contact_list_count_l57_57120


namespace range_of_omega_l57_57771

noncomputable def function_with_highest_points (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x + Real.pi / 4)

theorem range_of_omega (ω : ℝ) (hω : ω > 0)
  (h : ∀ x ∈ Set.Icc 0 1, 2 * Real.sin (ω * x + Real.pi / 4) = 2) :
  Set.Icc (17 * Real.pi / 4) (25 * Real.pi / 4) :=
by
  sorry

end range_of_omega_l57_57771


namespace circle_equation_l57_57084

theorem circle_equation (x y : ℝ) : 
  (∀ (a b : ℝ), (a - 1)^2 + (b - 1)^2 = 2 → (a, b) = (0, 0)) ∧
  ((0 - 1)^2 + (0 - 1)^2 = 2) → 
  (x - 1)^2 + (y - 1)^2 = 2 := 
by 
  sorry

end circle_equation_l57_57084


namespace smaller_tetrahedron_volume_ratio_l57_57829

theorem smaller_tetrahedron_volume_ratio {m n : ℕ} (m_prime: Nat.coprime m n) (h : 1/27 = m/n) : m + n = 28 :=
sorry

end smaller_tetrahedron_volume_ratio_l57_57829


namespace lisa_pizza_l57_57143

theorem lisa_pizza (P H S : ℕ) 
  (h1 : H = 2 * P) 
  (h2 : S = P + 12) 
  (h3 : P + H + S = 132) : 
  P = 30 := 
by
  sorry

end lisa_pizza_l57_57143


namespace division_proof_l57_57617

theorem division_proof :
  ((2 * 4 * 6) / (1 + 3 + 5 + 7) - (1 * 3 * 5) / (2 + 4 + 6)) / (1 / 2) = 3.5 :=
by
  -- definitions based on conditions
  let numerator1 := 2 * 4 * 6
  let denominator1 := 1 + 3 + 5 + 7
  let numerator2 := 1 * 3 * 5
  let denominator2 := 2 + 4 + 6
  -- the statement of the theorem
  sorry

end division_proof_l57_57617


namespace num_balls_picked_l57_57647

-- Define the variables involved: the numbers of red, blue, and green balls,
-- and the probability of picking two red balls.

def num_red : ℕ := 3
def num_blue : ℕ := 2
def num_green : ℕ := 3
def total_balls : ℕ := num_red + num_blue + num_green
def prob_both_red : ℝ := 0.10714285714285714

-- The combination function defined in Lean's math library
def comb (n k : ℕ) := Nat.choose n k

-- Statement of the problem
theorem num_balls_picked (n : ℕ) (h : comb total_balls n = 28 ∧ (3 : ℝ) / comb total_balls n = prob_both_red) : n = 2 :=
sorry

end num_balls_picked_l57_57647


namespace smallest_positive_period_and_monotonic_intervals_max_value_in_interval_find_m_l57_57770

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (real.sqrt 3 / 2) * real.sin (2 * x) - real.cos x ^ 2 - m

theorem smallest_positive_period_and_monotonic_intervals :
  ∀ m : ℝ, (∀ x : ℝ, f x m = real.sin (2 * x - real.pi / 6) - m - 0.5) →
  ∃ T > 0, (T = real.pi ∧
    (∀ k : ℤ, ∃ a b : ℝ, a = -real.pi / 6 + k * real.pi ∧ b = real.pi / 3 + k * real.pi ∧
    ( ∀ x : ℝ, a ≤ x ∧ x ≤ b → ∀ t : ℝ, f (x + t) m = f x m ∧ (real.cos (2 * (x + t) - real.pi / 6) > 0 ↔ t = 0) )
  )) :=
sorry

theorem max_value_in_interval_find_m :
  ∃ m : ℝ, (x ∈ Icc (5 * real.pi / 24) (3 * real.pi / 4)) →
  (∀ x : ℝ, f x m ≤ 0) →
  (f (real.pi / 3) m = 0 ∧ m = 0.5) :=
sorry

end smallest_positive_period_and_monotonic_intervals_max_value_in_interval_find_m_l57_57770


namespace average_weight_of_rock_l57_57125

-- Define all the conditions
def price_per_pound : ℝ := 4
def total_amount : ℝ := 60
def number_of_rocks : ℕ := 10

-- The statement we need to prove
theorem average_weight_of_rock :
  (total_amount / price_per_pound) / number_of_rocks = 1.5 :=
sorry

end average_weight_of_rock_l57_57125


namespace smallest_value_3a_plus_1_l57_57466

theorem smallest_value_3a_plus_1 (a : ℚ) (h : 8 * a^2 + 6 * a + 5 = 2) : 3 * a + 1 = -5 / 4 :=
sorry

end smallest_value_3a_plus_1_l57_57466


namespace f_2019_equals_12_max_f_n_le_2019_max_g_n_le_100_l57_57940

-- Definitions for functions f and g
def f : ℕ → ℕ
| 1       := 1
| (n + 1) := if n % 10 = 0 then f ((n + 1) / 10) else f n + 1

def g : ℕ → ℕ
| 1       := 1
| (n + 1) := if (n + 1) % 3 = 0 then g ((n + 1) / 3) else g n + 1

-- Statements for proof problems
theorem f_2019_equals_12 : f 2019 = 12 := sorry

theorem max_f_n_le_2019 : ∀ n, n ≤ 2019 → f n ≤ 28 ∧ ∃ n, n ≤ 2019 ∧ f n = 28 := sorry

theorem max_g_n_le_100 : ∀ n, n ≤ 100 → g n ≤ 8 ∧ ∃ n, n ≤ 100 ∧ g n = 8 := sorry

end f_2019_equals_12_max_f_n_le_2019_max_g_n_le_100_l57_57940


namespace line_through_C_equidistant_from_A_and_B_l57_57271

variables {α : Type*} [MetricSpace α] [NormedAddCommGroup α] [NormedSpace ℝ α]

-- Definitions of lines and points
def parallel_lines (a b : set α) : Prop := ∀ x ∈ a, ∀ y ∈ b, (x - y) ∈ ℝ ∙ (a - b)
def is_midpoint (M A B : α) : Prop := dist M A = dist M B ∧ 2 • M = A + B

-- Given conditions
variables (a b : set α) (A B C : α)
variable ha : parallel_lines a b
variable hA : A ∈ a
variable hB : B ∈ b

-- Required proof goal
theorem line_through_C_equidistant_from_A_and_B :
  ∃ l : set α, (∃ A₁ ∈ a, ∃ B₁ ∈ b, l = line_through C ⟨A₁, B₁⟩) ∧
    (parallel_lines l (line_through A B) ∨ ∃ M, is_midpoint M A B ∧ C ∈ line_through M) :=
sorry

end line_through_C_equidistant_from_A_and_B_l57_57271


namespace circle_equation_hyperbola_tangent_l57_57005

theorem circle_equation_hyperbola_tangent :
  ∃ (m : ℝ), m > 0 ∧ (∀ x y, (x - m)^2 + y^2 = 9) ∧ 
             m = 5 ∧ 
             (∀ x, abs (3 * m / (sqrt (3^2 + 4^2))) = 3) := 
begin
  sorry
end

end circle_equation_hyperbola_tangent_l57_57005


namespace math_problem_l57_57758

noncomputable def parabola (p : ℝ) := {x : ℝ × ℝ // x.2 ^ 2 = 2 * p * x.1}
def line (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def focus (p : ℝ) : ℝ × ℝ := (p, 0)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem math_problem (p : ℝ)
  (h_line_parabola_intersect : ∃ A B : ℝ × ℝ, line A.1 A.2 ∧ line B.1 B.2 ∧ parabola p A ∧ parabola p B ∧ dist A B = 4 * real.sqrt 15)
  (h_focus : ∃ F : ℝ × ℝ, F = focus p)
  (h_points_MN : ∃ M N : ℝ × ℝ, parabola p M ∧ parabola p N ∧ dot_product (M - (p, 0)) (N - (p, 0)) = 0) :
  p = 2 ∧ (∃ M N : ℝ × ℝ, parabola 2 M ∧ parabola 2 N ∧ dot_product (M - (2, 0)) (N - (2, 0)) = 0) ∧
  (∀ M N : ℝ × ℝ, parabola 2 M ∧ parabola 2 N ∧ dot_product (M - (2, 0)) (N - (2, 0)) = 0 → 
  (1/2) * |M.1 * N.2 - M.2 * N.1| = 12 - 8 * real.sqrt 2) := sorry

end math_problem_l57_57758


namespace part1_part2_l57_57853

variable (A B C a b c : ℝ)
variable (triangle : Prop -- Definition of triangle 

)-- Define angle B
def find_angle_B (eq1 : a = b * Real.cos C + sqrt 3 * c * Real.sin B) (b_eq : b = 1) : Prop :=
  B = Real.pi / 6

-- Define maximum area of triangle 
def max_area_triangle (eq1 : a = b * Real.cos C + sqrt 3 * c * Real.sin B) (b_eq : b = 1) : Prop :=
  let area := (1 / 4) * a * c 
  let ac_max := 2 + sqrt 3 
  area = ac_max / 4
  
-- The actual conjectures
theorem part1 (eq1 : a = b * Real.cos C + sqrt 3 * c * Real.sin B) (b_eq : b = 1) : find_angle_B eq1 b_eq := by
  sorry

theorem part2 (eq1 : a = b * Real.cos C + sqrt 3 * c * Real.sin B) (b_eq : b = 1) : max_area_triangle eq1 b_eq := by
  sorry

end part1_part2_l57_57853


namespace part1_part2_l57_57791

-- Definition of the conditions
def Point := ℝ × ℝ

def O : Point := (0, 0)
def N : Point := (1, 0)
def Q : Point := (-1/2, -1/2)

-- Noncomputable definition due to sqrt
noncomputable def E (P : Point) : Prop := ∃ x y : ℝ, P = (x, y) ∧ 0 ≤ x ∧ x < 1 ∧ y^2 = x

-- Part (1) statement: Prove the locus of point P satisfies the equation y^2 = x
theorem part1 (P : Point) : E P → P.2^2 = P.1 :=
by sorry

-- Part (2) statement: Prove the range of k such that the line through Q with slope k intersects E at exactly one point
theorem part2 (k : ℝ) : (∃ (x y : ℝ), (y + 1/2 = k * (x + 1/2)) ∧ E (x, y) ∧ ∀ (x' y' : ℝ), (y' + 1/2 = k * (x' + 1/2)) ∧ E (x', y') → (x', y') = (x, y))
                   ↔ k ∈ (Ioo (-1/3 : ℝ) 1 ∪ { (1 + Real.sqrt 3) / 2 }) :=
by sorry

end part1_part2_l57_57791


namespace angle_of_inclination_l57_57225

theorem angle_of_inclination (x y : ℝ) (α : ℝ) (h : sqrt 3 * x + y - 1 = 0) 
  (h_slope : ∀ x y, y = -sqrt 3 * x + 1 → slope_of_line y (-sqrt 3 * x + 1) = -sqrt 3) : 
  tan α = -sqrt 3 → α = 2 * π / 3 :=
by
  sorry

end angle_of_inclination_l57_57225


namespace distance_from_start_l57_57520

/--
Given conditions:
1. Jane moves 15 meters due north, 
2. then turns and moves 40 meters due east,
3. then turns and moves 3 meters due south,
prove that the distance from her starting point is 41.76 meters.
--/
theorem distance_from_start : 
  let y := 15 - 3 in
  let x := 40 in
  real.sqrt (y^2 + x^2) = sqrt 1744 :=
by
  let jane_distance := real.sqrt ((15 - 3)^2 + 40^2)
  have h : jane_distance = sqrt 1744 := sorry
  show sqrt ( (15 - 3)^2 + 40^2 ) = sqrt 1744 from h

end distance_from_start_l57_57520


namespace ratio_NF_AB_l57_57050

noncomputable theory

-- Conditions
def ellipse (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 5 = 1
def is_right_focus (F : ℝ × ℝ) : Prop := F = (2, 0)
def chord_through_focus (A B F : ℝ × ℝ) : Prop := 
  (F.1 - A.1) * (F.1 - B.1) + (F.2 - A.2) * (F.2 - B.2) < 0
def perpendicular_bisector_intersects_x_axis (A B N : ℝ × ℝ) : Prop := 
  (N.1, N.2) = (N.1, 0) ∧ N.2 = 0

-- Proof statement
theorem ratio_NF_AB (A B F N : ℝ × ℝ) 
  (h_ellipse_A : ellipse A.1 A.2)
  (h_ellipse_B : ellipse B.1 B.2) 
  (h_F : is_right_focus F)
  (h_chord_AB : chord_through_focus A B F)
  (h_perpendicular : perpendicular_bisector_intersects_x_axis A B N) :
  |N.1 - F.1| / Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1 / 3 := 
sorry

end ratio_NF_AB_l57_57050


namespace fraction_boys_is_correct_l57_57824

-- Define the initial conditions
def total_students : ℕ := 30
def girls_no_pet : ℕ := 8
def percent_girls_no_pet : ℝ := 0.60

-- Define the number of girls
noncomputable def girls_total : ℕ :=
  girls_no_pet / percent_girls_no_pet

-- Define the number of boys
noncomputable def boys_total : ℕ :=
  total_students - girls_total

-- Define the fraction of boys
noncomputable def fraction_boys : ℝ :=
  boys_total / total_students

-- The goal is to prove this statement
theorem fraction_boys_is_correct :
  fraction_boys = 17 / 30 := by
  sorry

end fraction_boys_is_correct_l57_57824


namespace correct_weight_misread_l57_57935

theorem correct_weight_misread (initial_avg correct_avg : ℝ) (num_boys : ℕ) (misread_weight : ℝ)
  (h_initial : initial_avg = 58.4) (h_correct : correct_avg = 58.85) (h_num_boys : num_boys = 20)
  (h_misread_weight : misread_weight = 56) :
  ∃ x : ℝ, x = 65 :=
by
  sorry

end correct_weight_misread_l57_57935


namespace middle_number_is_mixed2_1_5_l57_57258

noncomputable def sqrt5 : Real := Real.sqrt 5
noncomputable def frac7_3 : Real := 7 / 3
noncomputable def repeating2_0_5 : Real := 2 + 0.0 + 0.0555555 -- repeating addition understood as 2.05...
noncomputable def mixed2_1_5 : Real := 2 + 1/5

theorem middle_number_is_mixed2_1_5 : 
    let numbers := [sqrt5, 2.1, frac7_3, repeating2_0_5, mixed2_1_5]
    in 
    let sorted_numbers := List.sort numbers
    in 
    sorted_numbers.get! 2 = mixed2_1_5 :=
by
  sorry

end middle_number_is_mixed2_1_5_l57_57258


namespace right_triangle_area_l57_57092

theorem right_triangle_area (a b c h : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) (h2 : a = b) (h3 : h = 4) (h4 : sqrt(2) * a = c) :
  1 / 2 * a * h = 8 * sqrt(2) := 
sorry

end right_triangle_area_l57_57092


namespace trisha_total_distance_l57_57542

theorem trisha_total_distance :
  let d1 := 0.1111111111111111
  let d2 := 0.1111111111111111
  let d3 := 0.6666666666666666
  d1 + d2 + d3 = 0.8888888888888888 := 
by
  sorry

end trisha_total_distance_l57_57542


namespace part1_part2_l57_57755

noncomputable def condition1 : Prop :=
∀ (x y : ℝ), x - 2 * y + 1 = 0

noncomputable def condition2 (p : ℝ) : Prop :=
∀ (x y : ℝ), y^2 = 2 * p * x ∧ p > 0

noncomputable def condition3 (A B : (ℝ × ℝ)) (p : ℝ) : Prop :=
dist A B = 4 * real.sqrt 15 ∧
(∀ (x y : ℝ), (x - 2 * y + 1 = 0) ∧ (y^2 = 2 * p * x))

noncomputable def condition4 (C : ℝ) : Prop :=
F = (C, 0)

noncomputable def condition5 (M N F : (ℝ × ℝ)) : Prop :=
∃ (x1 y1 x2 y2 : ℝ), F = (1, 0) ∧
M = (x1, y1) ∧ N = (x2, y2) ∧ (x1 + x2) / 2 = C ∧ (y1 + y2) / 2 = 0 ∧ (F.1 * x1 + F.2 * y1) * (F.1 * x2 + F.2 * y2) = 0

theorem part1 :
  (∃ (p : ℝ), condition1 ∧ condition2 p ∧ condition3 (1, 0) (2, 0) p) →
  (p = 2) :=
  sorry

theorem part2 :
  (∃ (M N F : (ℝ × ℝ)), condition4 4 ∧ condition5 M N F) →
  (∀ (F : ℝ × ℝ), F = (1,0) →
  F + ℝ∧
  M=(1, C) N=(2, 409)∧
  ((M-product length eqad 0)MFN) =
  (12-8 * (2))) :=
 sorry

end part1_part2_l57_57755


namespace length_of_curve_l57_57390

variables (a : ℝ)

def length_of_curve_traced (a : ℝ) : ℝ := (Real.pi * a / 2) * (1 + Real.sqrt 5)

theorem length_of_curve (a : ℝ) (h1 : 0 < a) :
  let S := midpoint (A B) in -- Assume A, B, C, D, K and L are appropriately defined points along segment KL.
  -- additional conditions as needed for rolling motion of square
  length_of_curve_traced a = (Real.pi * a / 2) * (1 + Real.sqrt 5) :=
sorry

end length_of_curve_l57_57390


namespace range_of_a_if_solution_non_empty_l57_57718

variable (f : ℝ → ℝ) (a : ℝ)

/-- Given that the solution set of f(x) < | -1 | is non-empty,
    we need to prove that |a| ≥ 4. -/
theorem range_of_a_if_solution_non_empty (h : ∃ x, f x < 1) : |a| ≥ 4 :=
sorry

end range_of_a_if_solution_non_empty_l57_57718


namespace imo2010_q6_l57_57006

theorem imo2010_q6 (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (⟨x.to_int⟩ * y) = f x * ⟨f y⟩) →
  (f = 0) ∨ (∃ v ∈ Ioo 1 2, ∀ x, f x = v) :=
by sorry

end imo2010_q6_l57_57006


namespace range_of_a_l57_57738

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) : (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) ↔ (a < 0 ∨ (1 / 4 < a ∧ a < 4)) :=
by
  sorry

end range_of_a_l57_57738


namespace find_f_and_monotonicity_intervals_of_monotonicity_max_min_values_l57_57768

def f (x a b) := x^3 + a * x^2 + b

theorem find_f_and_monotonicity (a b : ℝ) :
  f 1 a b = 0 ∧ (deriv (λ x, f x a b)) 1 = -3 → a = -3 ∧ b = 2 :=
by
  assume h1 h2 : f 1 a b = 0 ∧ (deriv (λ x, f x a b)) 1 = -3
  sorry

theorem intervals_of_monotonicity : 
  (∀ x, f x -3 2 = x^3 - 3*x^2 + 2) →
  (∀ x, deriv (λ x, f x -3 2) x = 3*x^2 - 6*x) →
  ((∀ x, (x < 0 ∨ x > 2) → deriv (λ x, f x -3 2) x > 0) ∧ (∀ x, (0 < x ∧ x < 2) → deriv (λ x, f x -3 2) x < 0)) :=
by
  assume h1 h2
  sorry

theorem max_min_values (t : ℝ) (h : t > 0):
  (t ≤ 2 → ∀ x ∈ Icc 0 t, f x -3 2 ≤ f 0 -3 2 ∧ f x -3 2 > f t -3 2) ∧
  (2 < t ∧ t ≤ 3 → ∀ x ∈ Icc 0 t, f x -3 2 = f 0 -3 2 ∧ f x -3 2 > f 2 -3 2) ∧
  (t > 3 → ∀ x ∈ Icc 0 t, f x -3 2 < f t -3 2 ∧ f x -3 2 > f 2 -3 2) :=
by
  assume h
  sorry

end find_f_and_monotonicity_intervals_of_monotonicity_max_min_values_l57_57768


namespace case_a_sticks_case_b_square_l57_57602
open Nat 

premise n12 : Nat := 12
premise sticks12_sum : Nat := (n12 * (n12 + 1)) / 2  -- Sum of first 12 natural numbers
premise length_divisibility_4 : ¬ (sticks12_sum % 4 = 0)  -- Check if sum is divisible by 4

-- Need to break at least 2 sticks to form a square
theorem case_a_sticks (h : sticks12_sum = 78) (h2 : length_divisibility_4 = true) : 
  ∃ (k : Nat), k >= 2 := sorry

premise n15 : Nat := 15
premise sticks15_sum : Nat := (n15 * (n15 + 1)) / 2  -- Sum of first 15 natural numbers
premise length_divisibility4_b : sticks15_sum % 4 = 0  -- Check if sum is divisible by 4

-- Possible to form a square without breaking any sticks
theorem case_b_square (h : sticks15_sum = 120) (h2 : length_divisibility4_b = true) : 
  ∃ (k : Nat), k = 0 := sorry

end case_a_sticks_case_b_square_l57_57602


namespace trigonometric_identity_proof_l57_57729

theorem trigonometric_identity_proof :
  (∀ α : ℝ, sqrt 2 * sin (α + π / 4) = 4 * cos α → 2 * sin α ^ 2 - sin α * cos α + cos α ^ 2 = 8 / 5) :=
by sorry

end trigonometric_identity_proof_l57_57729


namespace archery_scores_l57_57834

theorem archery_scores (A_scores B_scores : Fin 5 → ℕ)
  (A_scores_range : ∀ i, 1 ≤ A_scores i ∧ A_scores i ≤ 10)
  (B_scores_range : ∀ i, 1 ≤ B_scores i ∧ B_scores i ≤ 10)
  (A_product : (Finset.univ.product (fun i => A_scores i) = 1764))
  (B_product : (Finset.univ.product (fun i => B_scores i) = 1764))
  (A_sum_eq_B_sum_minus_4 : Finset.univ.sum (fun i => A_scores i) + 4 = Finset.univ.sum (fun i => B_scores i)) :
  Finset.univ.sum (fun i => A_scores i) = 24 ∧
  Finset.univ.sum (fun i => B_scores i) = 28 := sorry

end archery_scores_l57_57834


namespace total_students_correct_l57_57684

-- Definitions based on the conditions
def students_germain : Nat := 13
def students_newton : Nat := 10
def students_young : Nat := 12
def overlap_germain_newton : Nat := 2
def overlap_germain_young : Nat := 1

-- Total distinct students (using inclusion-exclusion principle)
def total_distinct_students : Nat :=
  students_germain + students_newton + students_young - overlap_germain_newton - overlap_germain_young

-- The theorem we want to prove
theorem total_students_correct : total_distinct_students = 32 :=
  by
    -- We state the computation directly; proof is omitted
    sorry

end total_students_correct_l57_57684


namespace windmere_zoo_two_legged_birds_l57_57896

theorem windmere_zoo_two_legged_birds (b m u : ℕ) (head_count : b + m + u = 300) (leg_count : 2 * b + 4 * m + 3 * u = 710) : b = 230 :=
sorry

end windmere_zoo_two_legged_birds_l57_57896


namespace xiao_wang_exam_grades_l57_57994

theorem xiao_wang_exam_grades 
  (x y : ℕ) 
  (h1 : (x * y + 98) / (x + 1) = y + 1)
  (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) : 
  x + 2 = 10 ∧ y - 1 = 88 := 
by
  sorry

end xiao_wang_exam_grades_l57_57994


namespace jelly_bean_probabilities_l57_57661

theorem jelly_bean_probabilities :
  let p_red := 0.15
  let p_orange := 0.35
  let p_yellow := 0.2
  let p_green := 0.3
  p_red + p_orange + p_yellow + p_green = 1 :=
by
  sorry

end jelly_bean_probabilities_l57_57661


namespace trajectory_of_Q_l57_57400

noncomputable def point := ℝ × ℝ

def line_eq (a b c : ℝ) (p : point) : Prop :=
  let (x, y) := p in a * x + b * y + c = 0

def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def distance (p1 p2 : point) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)^0.5

def is_on_extension (p1 p2 p3 : point) : Prop :=
  distance p1 p2 = distance p2 p3

theorem trajectory_of_Q :
  ∀ (P Q M : point),
    line_eq 2 (-1) 3 P →
    M = (-1, 2) →
    is_on_extension P M Q →
    line_eq 2 (-1) 5 Q := by
  intros P Q M hP hM hQ
  sorry

end trajectory_of_Q_l57_57400


namespace part_a_l57_57382

-- Definition of the function f
def f (x : ℝ) := 2 * (Real.sqrt 3) * (Real.cos x) ^ 2 + 2 * (Real.sin x) * (Real.cos x) - Real.sqrt 3

theorem part_a (x : ℝ) : f x = 2 * Real.cos (2 * x - Real.pi / 6) := 
by
  sorry

end part_a_l57_57382


namespace solution1_solution2_l57_57760

noncomputable def problem1 (x y : ℝ) (p : ℝ) : Prop :=
  x - 2 * y + 1 = 0 ∧ y^2 = 2 * p * x ∧ 0 < p ∧ (abs (sqrt (1 + 4) * (y - y))) = 4 * sqrt 15

theorem solution1 (p: ℝ) : p = 2 :=
  sorry

noncomputable def problem2 (x y m n : ℝ) : Prop :=
  y^2 = 4 * x ∧ ∃ (F : ℝ × ℝ), F = (1, 0) ∧
  (∀ (M N : ℝ × ℝ), M ∈ y^2 = 4 * x ∧ N ∈ y^2 = 4 * x ∧ (F.1 - M.1) * (F.2 - N.1) + (F.2 - M.2) * (F.2 - N.2) = 0 →
  let area := (1/2) * abs ((N.1 - M.1) * (F.2 - M.2) - (N.2 - M.2) * (F.1 - M.1)) in
  ∃ min_area : ℝ, min_area = 12 - 8 * sqrt 2)

theorem solution2 (x y m n : ℝ) : ∃ min_area : ℝ, min_area = 12 - 8 * sqrt 2 :=
  sorry

end solution1_solution2_l57_57760


namespace largest_integral_x_l57_57007

theorem largest_integral_x (x : ℤ) (h1 : 1/4 < (x:ℝ)/6) (h2 : (x:ℝ)/6 < 7/9) : x ≤ 4 :=
by
  -- This is where the proof would go
  sorry

end largest_integral_x_l57_57007


namespace equation_has_no_real_solutions_l57_57705

/-- Prove that the graph of the given equation is empty.

Given the equation:
x^2 + 3y^2 - 4x - 6y + 10 = 0,
we need to show that there are no real (x, y) solutions that satisfy this equation.
-/
theorem equation_has_no_real_solutions :
  ∀ (x y : ℝ), x^2 + 3 * y^2 - 4 * x - 6 * y + 10 ≠ 0 :=
by {
  assume x y,
  /- The proof steps would show that the transformed equation cannot be satisfied by any real (x, y) -/
  sorry
}

end equation_has_no_real_solutions_l57_57705


namespace trisha_dogs_food_expense_l57_57969

theorem trisha_dogs_food_expense :
  ∀ (meat chicken veggies eggs initial remaining final: ℤ),
    meat = 17 → 
    chicken = 22 → 
    veggies = 43 → 
    eggs = 5 → 
    remaining = 35 → 
    initial = 167 →
    final = initial - (meat + chicken + veggies + eggs) - remaining →
    final = 45 := 
by
  intros meat chicken veggies eggs initial remaining final h_meat h_chicken h_veggies h_eggs h_remaining h_initial h_final
  sorry

end trisha_dogs_food_expense_l57_57969


namespace store_inequalities_l57_57653

variable (x : ℤ) -- define x as an integer variable

-- Conditions as given
def prod_cost_A := 8
def prod_cost_B := 2
def items_B := 2 * x - 4
def total_items_condition := (x + (2 * x - 4)) ≥ 32
def total_cost_condition := (8 * x + 2 * (2 * x - 4)) ≤ 148

-- The proof problem stating the question is == answer given the conditions
theorem store_inequalities (x : ℤ) (h1 : total_items_condition x) (h2 : total_cost_condition x) : 
  (x + (2 * x - 4) ≥ 32) ∧ (8 * x + 2 * (2 * x - 4) ≤ 148) := by
  split
  assumption
  assumption

end store_inequalities_l57_57653


namespace unique_sequence_l57_57701

theorem unique_sequence (a : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 1) ^ 2 = 1 + (n + 2021) * a n) →
  (∀ n : ℕ, a n = n + 2019) :=
by
  sorry

end unique_sequence_l57_57701


namespace rational_coefficients_of_rational_values_l57_57855

def polynomial (R : Type*) [comm_ring R] := R → R
def is_rational (R : Type*) [field R] (x : R) := ∃ a b : R, b ≠ 0 ∧ x = a / b
def takes_rational_values (P : polynomial ℚ) (pts : list ℚ) := ∀ x ∈ pts, is_rational ℚ (P x)

theorem rational_coefficients_of_rational_values
    (P : polynomial ℚ)
    (n : ℕ)
    (pts : fin (n + 1) → ℚ)
    (h_pts_distinct : function.injective pts)
    (h_rational_values : takes_rational_values P (finset.univ.map pts))
  : ∀ k, k ≤ n → ∃ a b : ℚ, b ≠ 0 ∧ ∃ l : ℕ, k = l / b :=
sorry

end rational_coefficients_of_rational_values_l57_57855


namespace committee_probability_at_least_one_boy_one_girl_l57_57931

-- Definitions from conditions
def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 5

-- Theorem statement
theorem committee_probability_at_least_one_boy_one_girl :
  -- Probability that the committee has at least one boy and one girl is 1705/1771
  (probability_at_least_one_boy_and_one_girl total_members boys girls committee_size) = (1705 / 1771 : ℚ) :=
sorry

-- Helper function to calculate the probability
noncomputable def probability_at_least_one_boy_and_one_girl 
  (total_members boys girls committee_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_members committee_size
  let ways_all_boys := Nat.choose boys committee_size
  let ways_all_girls := Nat.choose girls committee_size
  let ways_all_one_gender := ways_all_boys + ways_all_girls
  let probability_all_one_gender := (ways_all_one_gender : ℚ) / (total_ways : ℚ)
  1 - probability_all_one_gender

-- Example usage
#eval probability_at_least_one_boy_and_one_girl total_members boys girls committee_size -- Should give 1705/1771

end committee_probability_at_least_one_boy_one_girl_l57_57931


namespace square_area_y_coords_l57_57313

theorem square_area_y_coords (a b c d : ℝ) (h0 : a = 0) (h1 : b = 3) (h2 : c = 6) (h3 : d = 9) :
  let side := b - a in
  (side * side = 9) :=
by
  sorry

end square_area_y_coords_l57_57313


namespace circle_rolls_left_l57_57471

/-- Let a circle with a diameter of 1 unit roll one round to the left on a number line.
    If a point P on the circle starts at position 3 on the number line, the final position
    of point P on the number line is 3 - π. -/
theorem circle_rolls_left (d : ℝ) (start_pos : ℝ) (circumference : ℝ) (end_pos : ℝ) :
  d = 1 →
  start_pos = 3 →
  circumference = π * d →
  end_pos = start_pos - circumference →
  end_pos = 3 - π :=
by { intros, sorry }

end circle_rolls_left_l57_57471


namespace vector_AB_equals_4_vector_AM_l57_57346

-- Given definitions to set up the problem conditions
variables (A B C D E M : Type)
variables (vector : A → A → Type)
variables (AB AD DE CD CE CM AM : A → A → Type)
variables (angle_A : Prop) (angle_ACD_DCE_ECB : Prop)
variables (condition1 : ∀ (x y : A), vector x y = 3 * vector x E + 2 * vector E y)
variables (condition2 : ∀ (x y : A), vector x y = vector x D + 2 * vector D y)

-- The theorem statement
theorem vector_AB_equals_4_vector_AM
  (h1 : angle_A)
  (h2 : angle_ACD_DCE_ECB)
  (h3 : condition1 (vector A D) (vector D E))
  (h4 : condition2 (vector C D) (vector E M)) :
  vector A B = 4 * vector A M :=
by
  sorry

end vector_AB_equals_4_vector_AM_l57_57346


namespace geometric_sequences_l57_57208

theorem geometric_sequences :
  ∃ (a q : ℝ) (a1 a2 a3 : ℕ → ℝ), 
    (∀ n, a1 n = a * (q - 2) ^ n) ∧ 
    (∀ n, a2 n = 2 * a * (q - 1) ^ n) ∧ 
    (∀ n, a3 n = 4 * a * q ^ n) ∧
    a = 1 ∧ q = 4 ∨ a = 192 / 31 ∧ q = 9 / 8 ∧
    (a + 2 * a + 4 * a = 84) ∧
    (a * (q - 2) + 2 * a * (q - 1) + 4 * a * q = 24) :=
sorry

end geometric_sequences_l57_57208


namespace set_intersection_complement_l57_57537

theorem set_intersection_complement :
  let U := Set.univ : Set ℝ
  let A := { x : ℝ | x^2 - 2 * x < 0 }
  let B := { x : ℝ | 1 < x }
  let complement_B := { x : ℝ | x ≤ 1 }
  A ∩ complement_B = { x : ℝ | 0 < x ∧ x ≤ 1 } :=
by {
  let U := Set.univ : Set ℝ,
  let A := { x : ℝ | x^2 - 2 * x < 0 },
  let B := { x : ℝ | 1 < x },
  let complement_B := { x : ℝ | x ≤ 1 },
  sorry
}

end set_intersection_complement_l57_57537


namespace pq_identity_l57_57720

theorem pq_identity
  (p q : ℚ)
  (h : ∀ (x : ℚ) (hx : 0 < x),
    p / (2^x - 1) + q / (2^x + 4) = (10 * 2^x + 7) / ((2^x - 1) * (2^x + 4))) :
  p - q = -16 / 5 :=
by
  sorry

end pq_identity_l57_57720


namespace coffee_maker_capacity_l57_57211

theorem coffee_maker_capacity (x : ℝ) (h : 0.36 * x = 45) : x = 125 :=
sorry

end coffee_maker_capacity_l57_57211


namespace num_values_of_z_l57_57295

noncomputable def f (z : ℂ) : ℂ := -complex.I * conj z

theorem num_values_of_z : ∃ (z : ℂ), abs z = 7 ∧ f z = z ∧ is_disjoint (finset.range 3) = 2 :=
by
  sorry

end num_values_of_z_l57_57295


namespace christine_wander_time_l57_57691

noncomputable def distance : ℝ := 80
noncomputable def speed : ℝ := 20
noncomputable def time : ℝ := distance / speed

theorem christine_wander_time : time = 4 := 
by
  sorry

end christine_wander_time_l57_57691


namespace negation_of_P_equiv_correct_choice_C_l57_57061

open Real

def log_base2_three : ℝ := log 3 / log 2

theorem negation_of_P_equiv_correct_choice_C :
  (¬ ∃ x₀ ∈ set.Ici (1 : ℝ), (log_base2_three) ^ x₀ > 1) ↔ 
  (∀ x ∈ set.Ici (1 : ℝ), (log_base2_three) ^ x ≤ 1) :=
by
  sorry

end negation_of_P_equiv_correct_choice_C_l57_57061


namespace average_goal_l57_57175

-- Define the list of initial rolls
def initial_rolls : List ℕ := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

-- Define the next roll
def next_roll : ℕ := 2

-- Define the goal for the average
def goal_average : ℕ := 3

-- The theorem to prove that Ronald's goal for the average of all his rolls is 3
theorem average_goal : (List.sum (initial_rolls ++ [next_roll]) / (List.length (initial_rolls ++ [next_roll]))) = goal_average :=
by
  -- The proof will be provided later
  sorry

end average_goal_l57_57175


namespace limit_f2n_over_2n_fac_l57_57992

open Real

noncomputable def f (x : ℝ) : ℝ := exp x / x

theorem limit_f2n_over_2n_fac:
  (tendsto (λ n : ℕ, (f^[2 * n] 1) / (2 * n)!) at_top (𝓝 1)) :=
sorry

end limit_f2n_over_2n_fac_l57_57992


namespace number_of_ordered_pairs_l57_57368

theorem number_of_ordered_pairs :
  {n : ℕ | ∃ a b : ℤ, 1 < a ∧ a < b + 2 ∧ b + 2 < 10} = 28 :=
by sorry

end number_of_ordered_pairs_l57_57368


namespace sum_same_probability_l57_57483

noncomputable def same_probability_sum (k n m : ℕ) : Prop :=
  let min_sum := n
  let max_sum := n * m
  let midpoint := (min_sum + max_sum) / 2
  (2 * midpoint - k)

theorem sum_same_probability (k n m : ℕ) (hk : k = 12) (hn : n = 8) (hm : m = 6) :
  same_probability_sum k n m = 44 :=
by
  have min_sum := n
  have max_sum := n * m
  have midpoint := (min_sum + max_sum) / 2
  have counterpart_sum := 2 * midpoint - k
  rw [hk, hn, hm]
  simp only [min_sum, max_sum, midpoint, counterpart_sum]
  sorry

end sum_same_probability_l57_57483


namespace length_of_bridge_l57_57973

theorem length_of_bridge (v : ℝ) : 
  let time := 1 / 6
  let distance_A := (v + 2) * time
  let distance_B := (v - 1) * time
  distance_A + distance_B = (2 * v + 1) / 6 :=
by
  let time : ℝ := 1 / 6
  let distance_A := (v + 2) * time
  let distance_B := (v - 1) * time
  calc
    distance_A + distance_B = (v + 2) * time + (v - 1) * time : by rfl
    ... = ((v + 2) + (v - 1)) * time : by rw [right_distrib]
    ... = (2 * v + 1) / 6 : by sorry

end length_of_bridge_l57_57973


namespace renovation_days_l57_57278

/-
Conditions:
1. Cost to hire a company: 50000 rubles
2. Cost of buying materials: 20000 rubles
3. Husband's daily wage: 2000 rubles
4. Wife's daily wage: 1500 rubles
Question:
How many workdays can they spend on the renovation to make it more cost-effective?
-/

theorem renovation_days (cost_hire_company cost_materials : ℕ) 
  (husband_daily_wage wife_daily_wage : ℕ) 
  (more_cost_effective_days : ℕ) :
  cost_hire_company = 50000 → 
  cost_materials = 20000 → 
  husband_daily_wage = 2000 → 
  wife_daily_wage = 1500 → 
  more_cost_effective_days = 8 :=
by
  intros
  sorry

end renovation_days_l57_57278


namespace part1_part2_l57_57763

def geometric_seq (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  a 1 = 1 ∧ ∀ n, S n = (1 - (2 ^ n)) / (1 - 2) 

def arithmetic_seq (S : ℕ → ℕ) :=
  ∀ n, 6 * S (n + 1) = 4 * S n + 2 * S (n + 2)

theorem part1 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  geometric_seq a S ∧ arithmetic_seq S → ∀ n, a n = 2 ^ (n - 1) :=
by
  intros h
  sorry

def b_seq (a : ℕ → ℕ) (b : ℕ → ℕ) :=
  ∀ n, (n % 2 = 1 → b n = a n) ∧ (n % 2 = 0 → b n = n - 1)

theorem part2 (a b : ℕ → ℕ) (S : ℕ → ℕ) :
  geometric_seq a S ∧ arithmetic_seq S ∧ b_seq a b → ∀ n, (∑ i in range (2 * n), b (i + 1)) = (4^n - 1) / 3 + n^2 :=
by
  intros h
  sorry

end part1_part2_l57_57763


namespace impossible_coins_l57_57157

theorem impossible_coins (p_1 p_2 : ℝ) 
  (h1 : (1 - p_1) * (1 - p_2) = p_1 * p_2)
  (h2 : p_1 * (1 - p_2) + p_2 * (1 - p_1) = p_1 * p_2) : False := 
sorry

end impossible_coins_l57_57157


namespace elberta_amount_l57_57435

noncomputable def solve_elberta_amount (granny_smith : ℝ) (elberta : ℝ) (anjou : ℝ) : Prop :=
  granny_smith = 75 ∧ anjou = (1/4) * granny_smith ∧ elberta = anjou + 3 ∧ elberta = 21.75

theorem elberta_amount :
  solve_elberta_amount 75 21.75 (1/4 * 75) :=
by {
  have h1 : (1 / 4) * 75 = 18.75 := by norm_num,
  have h2 : 18.75 + 3 = 21.75 := by norm_num,
  exact ⟨rfl, h1, h2, rfl⟩,
}

end elberta_amount_l57_57435


namespace area_of_park_l57_57637

theorem area_of_park (L B : ℝ) (h1 : L / B = 1 / 3) (h2 : 12 * 1000 / 60 * 4 = 2 * (L + B)) : 
  L * B = 30000 :=
by
  sorry

end area_of_park_l57_57637


namespace system_of_equations_correct_l57_57246

-- Define the problem conditions
variable (x y : ℝ) -- Define the productivity of large and small harvesters

-- Define the correct system of equations as per the problem
def system_correct : Prop := (2 * (2 * x + 5 * y) = 3.6) ∧ (5 * (3 * x + 2 * y) = 8)

-- State the theorem to prove the correctness of the system of equations under given conditions
theorem system_of_equations_correct (x y : ℝ) : (2 * (2 * x + 5 * y) = 3.6) ∧ (5 * (3 * x + 2 * y) = 8) :=
by
  sorry

end system_of_equations_correct_l57_57246


namespace min_value_sequence_l57_57784

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 2, a (n + 1) + a n = (n + 1) * (Real.cos (n * Real.pi / 2))

def sum_sequence (a : ℕ → ℝ) (n : ℕ) :=
  ∑ i in Finset.range (n + 1), a i

variables (a : ℕ → ℝ) (m : ℝ) (S_n : ℕ → ℝ)

theorem min_value_sequence 
  (H1 : sequence a) 
  (H2 : S_n 2017 + m = 1010) 
  (H3 : S_n = sum_sequence a) 
  (H4 : a 1 * m > 0) : 
  ∃ a1 : ℝ, ∃ m1 : ℝ, (H5 : a1 = a 1) (H6 : m1 = m), 
  (1 / a1 + 1 / m1 = 2) ∧ (∀ x y : ℝ, (x = a1 ∧ y = m1) → (1 / x + 1 / y ≥ 2)) := 
sorry

end min_value_sequence_l57_57784


namespace cuboid_parallel_pairs_l57_57470

-- Definitions based on problem conditions
universe u

structure Line :=
  (p1 : Point u)
  (p2 : Point u)
  (is_line : p1 ≠ p2)

structure Plane :=
  (v1 : Point u)
  (v2 : Point u)
  (v3 : Point u)
  (v4 : Point u)
  (is_plane : v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4)

structure Cuboid :=
  (faces : Fin 6 → Plane) -- 6 faces

def parallel_line_plane_pair (l : Line) (p : Plane) : Prop :=
  -- Assuming the definition directly from the problem statement (line parallel to plane if...)
  sorry

-- The theorem to prove
theorem cuboid_parallel_pairs (c : Cuboid) :
  ∃ n, n = 48 ∧ ∀ l p, parallel_line_plane_pair l p → (l, p) ∈ pairs c.n := sorry

end cuboid_parallel_pairs_l57_57470


namespace set_intersection_complement_l57_57786

open Set

def I := {n : ℕ | True}
def A := {x ∈ I | 2 ≤ x ∧ x ≤ 10}
def B := {x | Nat.Prime x}

theorem set_intersection_complement :
  A ∩ (I \ B) = {4, 6, 8, 9, 10} := by
  sorry

end set_intersection_complement_l57_57786


namespace collinear_vectors_magnitude_l57_57782

def a : ℝ × ℝ := (1, 2)
def b (k : ℝ) : ℝ × ℝ := (-2, k)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v1 = (t * v2.1, t * v2.2)

def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vec_scale (v : ℝ × ℝ) (s : ℝ) : ℝ × ℝ :=
  (s * v.1, s * v.2)

def vec_mag (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem collinear_vectors_magnitude :
  collinear (1, 2) (-2, -4) → vec_mag (vec_add (vec_scale (1, 2) 3) (-2, -4)) = Real.sqrt 5 :=
by 
  intro col
  sorry

end collinear_vectors_magnitude_l57_57782


namespace quadrilateral_reflection_area_l57_57551

-- Assuming that S and point O are given.
variables (S : ℝ) (O : Point) (ABCD : Quadrilateral)
variables (E F G H : Point) -- Midpoints of sides of quadrilateral ABCD

-- Definition of reflections of O with respect to midpoints E, F, G, H
variables (E₁ F₁ G₁ H₁ : Point)

-- Conditions
def is_convex (ABCD : Quadrilateral) : Prop := -- Definition of convex quadrilateral
  sorry

def area (Q : Quadrilateral) : ℝ := -- Definition of area of quadrilateral
  sorry

def midpoint (A B : Point) : Point := -- Definition of midpoint of two points
  sorry

def reflection (O M : Point) : Point := -- Definition of reflection of O with respect to M
  sorry

-- Given conditions
axiom h1 : is_convex ABCD
axiom h2 : area ABCD = S
axiom h3 : E = midpoint A B
axiom h4 : F = midpoint B C
axiom h5 : G = midpoint C D
axiom h6 : H = midpoint D A
axiom h7 : E₁ = reflection O E
axiom h8 : F₁ = reflection O F
axiom h9 : G₁ = reflection O G
axiom h10 : H₁ = reflection O H

-- Target theorem statement
theorem quadrilateral_reflection_area : area (Quadrilateral_of_points E₁ F₁ G₁ H₁) = 2 * S :=
by
  sorry

end quadrilateral_reflection_area_l57_57551


namespace count_positive_3_digit_numbers_divisible_by_9_l57_57449

-- Conditions
def is_divisible_by_9 (n : ℕ) : Prop := 9 ∣ n

def is_positive_3_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Theorem to be proved
theorem count_positive_3_digit_numbers_divisible_by_9 : 
  {n : ℕ | is_positive_3_digit_number n ∧ is_divisible_by_9 n}.card = 100 :=
sorry

end count_positive_3_digit_numbers_divisible_by_9_l57_57449


namespace quadrilateral_circumcircle_properties_l57_57424

-- Given:
variables {R : ℝ} -- R is the radius of the circumcircle

-- Definitions that fit the proof problem
def sum_of_squares_vertices := ∀ (A B C D P : ℝ × ℝ), AP^2 + BP^2 + CP^2 + DP^2 = 8 * R^2

def sum_of_squares_sides := ∀ (A B C D : ℝ × ℝ), (A - B).norm_sq + (B - C).norm_sq + (C - D).norm_sq + (D - A).norm_sq = 8 * R^2

-- The theorem combining both parts of the problem
theorem quadrilateral_circumcircle_properties (R : ℝ) :
  (sum_of_squares_vertices R) ∧ (sum_of_squares_sides R) :=
by
  sorry

end quadrilateral_circumcircle_properties_l57_57424


namespace differential_equation_solution_l57_57641

theorem differential_equation_solution (C y : ℝ) :
  ∃ (x : ℝ), x = C * exp (sin y) - 2 * (1 + sin y) ∧
  (∀ (f : ℝ → ℝ), f y = (differential_s f y (C * exp (sin y) - 2 * (1 + sin y))) =
     f y = 1 / ((C * exp (sin y) - 2 * (1 + sin y)) * cos y + sin (2 * y))) := sorry

end differential_equation_solution_l57_57641


namespace zero_points_of_f_l57_57025

def f (x : ℝ) : ℝ := if x >= -2 then 2^x - 3 else f (-2 - x)

theorem zero_points_of_f :
  (∃ x ∈ Ioo 1 2, f x = 0) ∧ (∃ x ∈ Ioo (-6) (-5), f x = 0) :=
by
  sorry

end zero_points_of_f_l57_57025


namespace sum_of_obtuse_angles_l57_57467

theorem sum_of_obtuse_angles (A B : ℝ) (hA1 : A > π / 2) (hA2 : A < π)
  (hB1 : B > π / 2) (hB2 : B < π)
  (hSinA : Real.sin A = Real.sqrt 5 / 5)
  (hSinB : Real.sin B = Real.sqrt 10 / 10) :
  A + B = 7 * π / 4 := 
sorry

end sum_of_obtuse_angles_l57_57467


namespace females_next_to_each_other_l57_57233

theorem females_next_to_each_other : 
  let students := ["M1", "M2", "F1", "F2"]
  in  (count_arrangements_with_females_adjacent students = 12) :=
by
  -- Definitions and calculations required to prove the statement
  sorry

end females_next_to_each_other_l57_57233


namespace max_days_to_be_cost_effective_is_8_l57_57276

-- Definitions
def cost_of_hiring (₥ : ℕ) := 50000
def cost_of_materials (₥ : ℕ) := 20000
def husbands_daily_wage (₥ : ℕ) := 2000
def wifes_daily_wage (₥ : ℕ) := 1500

-- Total daily wage
def total_daily_wage := husbands_daily_wage 0 + wifes_daily_wage 0

-- Cost difference
def cost_difference := cost_of_hiring 0 - cost_of_materials 0

-- Maximum number of days
def max_days_to_be_cost_effective := cost_difference / total_daily_wage

-- Prove that the maximum number of days is 8
theorem max_days_to_be_cost_effective_is_8 : max_days_to_be_cost_effective = 8 := by
  sorry

end max_days_to_be_cost_effective_is_8_l57_57276


namespace at_least_one_speaks_japanese_independent_events_boy_korean_l57_57191

theorem at_least_one_speaks_japanese (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (boys_japanese : ℕ) (girls_japanese : ℕ) : 
  total = 36 ∧ boys = 12 ∧ girls = 24 ∧ boys_japanese = 8 ∧ girls_japanese = 12 →
  (20.choose 1 * 16.choose 1 + 20.choose 2) / 36.choose 2 = 17 / 21 :=
by sorry

theorem independent_events_boy_korean {m n : ℕ} (total : ℕ) (boys : ℕ) 
  (girls : ℕ) : 
  total = 36 ∧ boys = 12 ∧ girls = 24 ∧ 6 ≤ m ∧ m ≤ 8 →
  (n = 2 * m) ↔ ((m = 6 ∧ n = 12) ∨ (m = 7 ∧ n = 14) ∨ (m = 8 ∧ n = 16)) :=
by sorry

end at_least_one_speaks_japanese_independent_events_boy_korean_l57_57191


namespace part1a_part1b_part2_part3_l57_57629

-- Part (1) Equivalent Proof Problems
theorem part1a : (1 / (Real.sqrt 3 + Real.sqrt 2)) = (Real.sqrt 3 - Real.sqrt 2) :=
sorry

theorem part1b : (1 / (Real.sqrt 5 + Real.sqrt 3)) = (1 / 2) * (Real.sqrt 5 - Real.sqrt 3) :=
sorry

-- Part (2) Equivalent Proof Problem
theorem part2 : 
  (list.range (121 - 11 + 1) / 2).map (λn, (1 / (Real.sqrt ((11 + 2 * n).toNat) + Real.sqrt ((11 + 2 * n - 2).toNat)))).sum = 4 :=
sorry

-- Part (3) Equivalent Proof Problem
theorem part3 (a : ℝ) (h : a = 1 / (Real.sqrt 2 - 1)) : 4 * a ^ 2 - 8 * a + 1 = 5 :=
sorry

end part1a_part1b_part2_part3_l57_57629


namespace winning_margin_is_500_l57_57236

noncomputable def election_problem (total_votes winning_votes losing_votes won_by : ℝ) : Prop :=
  winning_votes = 0.75 * total_votes ∧
  winning_votes = 750 ∧
  losing_votes = 0.25 * total_votes ∧
  won_by = winning_votes - losing_votes

theorem winning_margin_is_500 : ∃ total_votes winning_votes losing_votes won_by,
  election_problem total_votes winning_votes losing_votes won_by ∧
  won_by = 500 :=
by
  let total_votes := 1000
  let winning_votes := 750
  let losing_votes := 250
  let won_by := 500
  use [total_votes, winning_votes, losing_votes, won_by]
  split
  { split
    { exact rfl }
    { split
      { exact rfl }
      { split
        { exact rfl }
        { exact rfl } } } }
  { exact rfl }
  sorry

end winning_margin_is_500_l57_57236


namespace find_scalar_m_l57_57087

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

-- Define points A, B, C, and M in the vector space V
variables (A B C M : V)

-- Define the conditions as hypotheses
hypothesis h1 : (M - A) + (M - B) + (M - C) = 0
hypothesis h2 : (B - A) + (C - A) + m • (M - A) = 0

-- Define and state the problem
theorem find_scalar_m (A B C M : V) (h1 : (M - A) + (M - B) + (M - C) = 0)
                      (h2 : (B - A) + (C - A) + m • (M - A) = 0) : m = -3 :=
sorry

end find_scalar_m_l57_57087


namespace min_sticks_12_to_break_can_form_square_15_l57_57588

-- Problem definition for n = 12
def sticks_12 : List Nat := List.range' 1 12

theorem min_sticks_12_to_break : 
  ... (I realize I need to translate a step better) ..............
  sorry

-- Problem definition for n = 15
def sticks_15 : List Nat := List.range' 1 15

theorem can_form_square_15 : 
  ... (implementing a nice explanation)
  sorry

end min_sticks_12_to_break_can_form_square_15_l57_57588


namespace max_perpendicular_diagonals_l57_57387

theorem max_perpendicular_diagonals (n : ℕ) (h : n ≥ 3) : 
  (∃ m : ℕ, (∃ k, n = 2 * k + 1) -> m = n - 3) ∧ 
  (∃ l : ℕ, (∃ k, n = 2 * k) -> l = n - 2) :=
begin
  -- Proof required here
  sorry
end

end max_perpendicular_diagonals_l57_57387


namespace square_of_twice_magnitude_l57_57139

variable (w : ℂ) (h : complex.abs w = 11)

theorem square_of_twice_magnitude : (2 * complex.abs w) ^ 2 = 484 := by
  sorry

end square_of_twice_magnitude_l57_57139


namespace true_propositions_l57_57625

-- Proposition A
def propositionA (x y : ℝ) : Prop :=
  (x > 2 ∧ y > 3) ↔ (x + y > 5)

-- Proposition B
def propositionB (x : ℝ) : Prop :=
  (x > 1) → (|x| > 0)

-- Proposition C
def propositionC (a b c : ℝ) (ha : a ≠ 0) : Prop :=
  (b^2 - 4 * a * c = 0) ↔ (∃ x : ℝ, a * x^2 + b * x + c = 0)

-- Proposition D
def propositionD (a b c : ℝ) : Prop :=
  (a^2 + b^2 = c^2) ↔ (a > 0 ∧ b > 0 ∧ c > 0 ∧ (∃ α β γ : ℝ, α^2 + β^2 = γ^2))

theorem true_propositions : Prop :=
  (propositionB) ∧ (propositionD)

end true_propositions_l57_57625


namespace parker_total_weight_l57_57547

-- Define the number of initial dumbbells and their weight
def initial_dumbbells := 4
def weight_per_dumbbell := 20

-- Define the number of additional dumbbells
def additional_dumbbells := 2

-- Define the total weight calculation
def total_weight := initial_dumbbells * weight_per_dumbbell + additional_dumbbells * weight_per_dumbbell

-- Prove that the total weight is 120 pounds
theorem parker_total_weight : total_weight = 120 :=
by
  -- proof skipped
  sorry

end parker_total_weight_l57_57547


namespace cos_x_is_necessary_but_not_sufficient_for_sin_x_zero_l57_57562

-- Defining the conditions
def cos_x_eq_one (x : ℝ) : Prop := Real.cos x = 1
def sin_x_eq_zero (x : ℝ) : Prop := Real.sin x = 0

-- Main theorem statement
theorem cos_x_is_necessary_but_not_sufficient_for_sin_x_zero (x : ℝ) : 
  (∀ x, cos_x_eq_one x → sin_x_eq_zero x) ∧ (∃ x, sin_x_eq_zero x ∧ ¬ cos_x_eq_one x) :=
by 
  sorry

end cos_x_is_necessary_but_not_sufficient_for_sin_x_zero_l57_57562


namespace minimum_value_of_3m_plus_n_l57_57941

theorem minimum_value_of_3m_plus_n
  {a m n : ℝ}
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : (∀ x, y = a^(x+3) - 2))
  (h4 : (3m + n ≥ 16))
  (h5 : m > 0)
  (h6 : n > 0)
  : 3m + n = 16 := 
begin
  sorry
end

end minimum_value_of_3m_plus_n_l57_57941


namespace find_angle_AMC_l57_57086

def angle_AMC (A B C M : Type) (α β γ δ : ℝ) : Prop :=
  α = 70 ∧ 
  β = 10 ∧ 
  γ = 30 ∧
  δ = 80 ∧ 
  AC = BC ∧ 
  ∃ (τ: ℝ), τ = 70

theorem find_angle_AMC (A B C M : Type) (α β γ δ : ℝ) (AC BC : ℝ) :
  γ = 30 → 
  β = 10 → 
  δ = 80 → 
  AC = BC → 
  τ = 70 →
  angle_AMC A B C M α β γ δ :=
by 
  intro h₁ h₂ h₃ h₄ h₅
  exact ⟨h₅, h₂, h₁, h₃, h₄, ⟨h₅⟩⟩

end find_angle_AMC_l57_57086


namespace chromium_percentage_new_alloy_l57_57099

variable (w1 w2 : ℝ) (cr1 cr2 : ℝ)

theorem chromium_percentage_new_alloy (h_w1 : w1 = 15) (h_w2 : w2 = 30) (h_cr1 : cr1 = 0.12) (h_cr2 : cr2 = 0.08) :
  (cr1 * w1 + cr2 * w2) / (w1 + w2) * 100 = 9.33 := by
  sorry

end chromium_percentage_new_alloy_l57_57099


namespace find_n_in_arithmetic_sequence_l57_57489

theorem find_n_in_arithmetic_sequence :
  ∀ (a_3 a_7 : ℕ -> ℕ) (S_n : ℕ -> ℕ),
  (a_3 8) ∧ (a_7 20) ∧ (S_n = λn, ∑ k in range n, 1 / ((3 * k + 1 - 1) * (3 * (k + 1) + 1 - 1))) ∧ S_n 4 / 25 → n = 16 :=
by
  intro a_3 a_7 S_n h
  sorry

end find_n_in_arithmetic_sequence_l57_57489


namespace pentagon_side_length_is_10cm_l57_57662

noncomputable def construct_pentagon_radius (AB BC CD: ℝ) : ℝ :=
let BD := Real.sqrt ((AB / 2) ^ 2 + BC ^ 2) in
BD

theorem pentagon_side_length_is_10cm {AB BC CD R: ℝ} 
  (h1: AB = 10)
  (h2: BC = 5)
  (h3: CD = 5)
  (h4: R = construct_pentagon_radius AB BC CD) :
  true := 
sorry

end pentagon_side_length_is_10cm_l57_57662


namespace two_triangles_with_two_good_sides_l57_57495

theorem two_triangles_with_two_good_sides (m n : ℕ) (hmo : Odd m) (hno : Odd n) :
  ∀ partition : Finset (Finset (Fin 2 × Fin 2)), 
    (∀ t ∈ partition, ∃ s ∈ t, is_good_side s) →
    (∀ s, is_bad_side s → ∃ t1 t2 ∈ partition, s ∈ t1 ∧ s ∈ t2) →
    ∃ t1 t2 ∈ partition, (count_good_sides t1 ≥ 2) ∧ (count_good_sides t2 ≥ 2) :=
by
  sorry

-- Definitions for sides
def is_good_side (s : Fin 2 × Fin 2) : Prop := 
∃ j k : ℤ, (s.1 = j ∧ s.2 = k ∧ (altitude s = 1))

def is_bad_side (s : Fin 2 × Fin 2) : Prop := 
¬ is_good_side s

-- Counting function
def count_good_sides (t : Finset (Fin 2 × Fin 2)) : ℕ := 
t.countp is_good_side

end two_triangles_with_two_good_sides_l57_57495


namespace tims_annual_interest_rate_l57_57895

theorem tims_annual_interest_rate :
  ∃ r : ℝ,
  let P_T : ℝ := 500
      r_L : ℝ := 0.05
      P_L : ℝ := 1000
      A_L := P_L * (1 + r_L)^2
      I_L  := A_L - P_L
      I_T := I_L + 2.50 in
  1 + r = sqrt ((I_T + P_T) / P_T) :=
  by
  let r := 0.1
  have h₀ : P_T = 500 := by rfl
  have h_L1 : P_L = 1000 := by rfl
  have h_L2 : r_L = 0.05 := by rfl
  -- Lana's amount after 2 years:
  have h_L3 : A_L = 1000 * (1 + 0.05)^2 := by rfl
  have h_L4 : A_L = 1000 * 1.1025 := by rfl
  have h_L5 : A_L = 1102.5 := by rfl
  -- Lana's interest after 2 years:
  have h_L6 : I_L = 1102.5 - 1000 := by rfl
  have h_L7 : I_L = 102.5 := by rfl
  -- Tim's interest after 2 years:
  have h_T1 : I_T = I_L + 2.5 := by rfl
  have h_T2 : I_T = 105 := by rfl
  -- Tim's amount after 2 years:
  have h_T3 : 605 = 500 * (1 + r)^2 := by
  have h_T4 : sqrt (605 / 500) = 1 + r := by
  have h_T5 : 1.1 = 1 + r := by
  show 1 + r = sqrt ((I_T + 500) / 500) from eq.symm h_T5
  sorry

end tims_annual_interest_rate_l57_57895


namespace infinite_product_eq_l57_57338

theorem infinite_product_eq : 
  (∏ (k : ℕ) in (finset.range (nat.succ nat.card_void)), 3^(k.succ * 1/(3^k.succ))) = 3^(3/4) := 
  sorry

end infinite_product_eq_l57_57338


namespace parallelogram_angle_bisectors_form_rectangle_l57_57946

theorem parallelogram_angle_bisectors_form_rectangle (A B C D P Q R S : Type) 
  [parallelogram A B C D]
  (h1 : ∠ABC = ∠CDA)
  (h2 : ∠BCD = ∠DAB)
  (h3 : ∠ABC + ∠BCD = 180)
  (h4 : ∠BCD + ∠CDA = 180)
  (h5 : ∠CDA + ∠DAB = 180)
  (h6 : ∠DAB + ∠ABC = 180)
  (h7 : is_intersection_point_of_angle_bisectors P (A B C D))
  (h8 : is_intersection_point_of_angle_bisectors Q (A B C D))
  (h9 : is_intersection_point_of_angle_bisectors R (A B C D))
  (h10 : is_intersection_point_of_angle_bisectors S (A B C D)) :
  is_rectangle P Q R S :=
sorry

end parallelogram_angle_bisectors_form_rectangle_l57_57946


namespace find_x_of_expression_l57_57284

theorem find_x_of_expression (x : ℝ) (h : (sqrt 27 + sqrt 243) / sqrt x = 3.0000000000000004) : x = 48 :=
sorry

end find_x_of_expression_l57_57284


namespace x_intercept_of_line_through_points_l57_57846

noncomputable def point := ℝ × ℝ

def line (p1 p2 : point) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let m := (y2 - y1) / (x2 - x1)
  (m, y1 - m * x1)

def x_intercept (m b : ℝ) : ℝ :=
  -b / m

theorem x_intercept_of_line_through_points :
  let p1 : point := (10, 3)
  let p2 : point := (−10, −7)
  let p3 : point := (5, 1)
  let (m, b) := line p1 p2
  x_intercept m b = 4 :=
by
  let p1 : point := (10, 3)
  let p2 : point := (−10, −7)
  let p3 : point := (5, 1)
  let (m, b) := line p1 p2
  sorry

end x_intercept_of_line_through_points_l57_57846


namespace x_range_l57_57349

-- Each bᵢ ∈ {1, 3}
def valid_b (b : ℕ → ℕ) := ∀ i, b i = 1 ∨ b i = 3

-- x defined by the series: x = b₁/2 + b₂/2² + ... + b₂₀/2²⁰
def x (b : ℕ → ℕ) := (∑ i in Finset.range 20, b i / 2^(i+1))

-- Proof statement: final range of x is 1 ≤ x < 4.
theorem x_range (b : ℕ → ℕ) (h : valid_b b) :
  1 ≤ x b ∧ x b < 4 :=
sorry

end x_range_l57_57349


namespace f_expression_g_properties_l57_57772

noncomputable def f (x : ℝ) (A : ℝ) : ℝ := A * Real.sin (1 / 3 * x + Real.pi / 6)

def A_pos (A : ℝ) : Prop := 0 < A

def period (T : ℝ) : Prop := T = 6 * Real.pi

def f_constraint (f : ℝ → ℝ) : Prop := f (2 * Real.pi) = 2

theorem f_expression (A : ℝ) (hA : A_pos A) (hT : period 6 * Real.pi) (hC : f_constraint (f A)) :
  f = λ x, 4 * Real.sin (1 / 3 * x + Real.pi / 6) :=
sorry

noncomputable def g (x : ℝ) : ℝ := f 4 x + 2

theorem g_properties :
  (∀ k : ℤ, ∀ x ∈ Icc (6 * k * Real.pi - 2 * Real.pi) (Real.pi + 6 * k * Real.pi), ∀ y ∈ Icc (6 * k * Real.pi - 2 * Real.pi) (Real.pi + 6 * k * Real.pi), x < y → g x < g y) ∧
  (∀ k : ℤ, ∀ x ∈ Icc (Real.pi + 6 * k * Real.pi) (4 * Real.pi + 6 * k * Real.pi), ∀ y ∈ Icc (Real.pi + 6 * k * Real.pi) (4 * Real.pi + 6 * k * Real.pi), x < y → g y < g x) ∧
  (∃ x : ℝ, g x = 6) :=
sorry

end f_expression_g_properties_l57_57772


namespace probability_fractional_product_l57_57977

def first_spinner : Set ℚ := {3, 1 / 2}
def second_spinner : Set ℚ := {2 / 3, 5, 7}

def favorable_outcome (a b : ℚ) : Prop := (a * b).den ≠ 1

theorem probability_fractional_product :
  let total_outcomes : ℕ := 6
  let favorable_outcomes := 4 in
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 3 :=
by
  sorry

end probability_fractional_product_l57_57977


namespace pumps_time_ratio_l57_57974

/-
  Given:
    - Pump X can fill a tank in 40 minutes.
    - Pump Y can empty the same tank in 48 minutes.
  Prove:
    The ratio of the time taken by both pumps together to fill the tank to the time taken by pump X alone to fill the tank is 6.
-/
theorem pumps_time_ratio
  (R_X R_Y : ℝ)
  (h_R_X : R_X = 1 / 40)
  (h_R_Y : R_Y = 1 / 48) :
  let net_rate := R_X - R_Y in
  let time_together := 1 / net_rate in
  (time_together / 40) = 6 := by
  sorry

end pumps_time_ratio_l57_57974


namespace min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l57_57578

-- Part (a): For n = 12:
theorem min_sticks_to_break_for_square_12 : ∀ (n : ℕ), n = 12 → 
  (∃ (sticks : Finset ℕ), sticks.card = 12 ∧ sticks.sum id = 78 ∧ (¬ (78 % 4 = 0) → 
  ∃ (b : ℕ), b = 2)) := 
by sorry

-- Part (b): For n = 15:
theorem can_form_square_without_breaking_15 : ∀ (n : ℕ), n = 15 → 
  (∃ (sticks : Finset ℕ), sticks.card = 15 ∧ sticks.sum id = 120 ∧ (120 % 4 = 0)) :=
by sorry

end min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l57_57578


namespace maximum_k_l57_57301

-- Define the set M
def M : Set (ℤ × ℤ) := { p | let x := p.1; let y := p.2 in ((abs x - 1)^2 + (abs y - 1)^2 < 4) }

-- Given conditions
def is_in_M (p : ℤ × ℤ) : Prop := p ∈ M

-- Define probability for the condition xy >= k
def prob_xy_ge_k (k : ℤ) : ℚ := 6 / 25

-- Define the main theorem
theorem maximum_k (k : ℤ) (h1 : k > 0) (h2 : prob_xy_ge_k k = 6 / 25) : k = 2 := sorry

end maximum_k_l57_57301


namespace divide_into_two_equal_parts_l57_57363

-- Define a geometric figure and its properties
structure GeometricFigure where
  symmetry : Bool
  area : ℝ
  divided_area : Π (cut: ℝ -> Bool), (cut 0.5 ∧ cut 1.0 → Bool) -> Prop

-- The problem now is to show that there exists a cut on the figure that divides it equally
theorem divide_into_two_equal_parts (fig : GeometricFigure) : 
  fig.symmetry = true →
  ∃ (cut: ℝ -> Bool), (cut 0.5 ∧ cut 1.0 → Bool) ∧ 
  fig.divided_area cut = (fig.area / 2) :=
by
  sorry

end divide_into_two_equal_parts_l57_57363


namespace regression_analysis_issues_l57_57492

theorem regression_analysis_issues
  (population_applicability : Prop)
  (time_validity : Prop)
  (sample_range_influence : Prop)
  (forecast_average : Prop) :
  (population_applicability ∧ time_validity ∧ sample_range_influence ∧ forecast_average) ↔
  ((∀ P, population_applicability) ∧
   (∀ T, time_validity) ∧
   (∀ S, sample_range_influence) ∧
   (∀ F, forecast_average)) :=
by
  split
  { intros h
   { split
     }
  }
  sorry

end regression_analysis_issues_l57_57492


namespace max_partitioned_test_plots_is_78_l57_57294

def field_length : ℕ := 52
def field_width : ℕ := 24
def total_fence : ℕ := 1994
def gcd_field_dimensions : ℕ := Nat.gcd field_length field_width

-- Since gcd_field_dimensions divides both 52 and 24 and gcd_field_dimensions = 4
def possible_side_lengths : List ℕ := [1, 2, 4]

noncomputable def max_square_plots : ℕ :=
  let max_plots (a : ℕ) : ℕ := (field_length / a) * (field_width / a)
  let valid_fence (a : ℕ) : Bool :=
    let vertical_fence := (field_length / a - 1) * field_width
    let horizontal_fence := (field_width / a - 1) * field_length
    vertical_fence + horizontal_fence ≤ total_fence
  let valid_lengths := possible_side_lengths.filter valid_fence
  valid_lengths.map max_plots |>.maximum? |>.getD 0

theorem max_partitioned_test_plots_is_78 : max_square_plots = 78 := by
  sorry

end max_partitioned_test_plots_is_78_l57_57294


namespace min_sticks_to_be_broken_form_square_without_breaks_l57_57594

noncomputable def total_length (n : ℕ) : ℕ := n * (n + 1) / 2

def divisible_by_4 (x : ℕ) : Prop := x % 4 = 0

theorem min_sticks_to_be_broken (n : ℕ) : n = 12 → (¬ divisible_by_4 (total_length n)) ∧ (minimal_breaks n = 2) :=
by
  intro h1
  rw h1
  have h2 : total_length 12 = 78 := by decide
  have h3 : ¬ divisible_by_4 78 := by decide
  exact ⟨h3, sorry⟩

theorem form_square_without_breaks (n : ℕ) : n = 15 → divisible_by_4 (total_length n) ∧ (minimal_breaks n = 0) :=
by
  intro h1
  rw h1
  have h2 : total_length 15 = 120 := by decide
  have h3 : divisible_by_4 120 := by decide
  exact ⟨h3, sorry⟩

end min_sticks_to_be_broken_form_square_without_breaks_l57_57594


namespace charcoal_drawings_count_l57_57962

-- Defining the conditions
def total_drawings : Nat := 25
def colored_pencil_drawings : Nat := 14
def blending_marker_drawings : Nat := 7

-- Defining the target value for charcoal drawings
def charcoal_drawings : Nat := total_drawings - (colored_pencil_drawings + blending_marker_drawings)

-- The theorem we need to prove
theorem charcoal_drawings_count : charcoal_drawings = 4 :=
by
  -- Lean proof goes here, but since we skip the proof, we'll just use 'sorry'
  sorry

end charcoal_drawings_count_l57_57962


namespace jerome_contacts_total_l57_57123

def jerome_classmates : Nat := 20
def jerome_out_of_school_friends : Nat := jerome_classmates / 2
def jerome_family_members : Nat := 2 + 1
def jerome_total_contacts : Nat := jerome_classmates + jerome_out_of_school_friends + jerome_family_members

theorem jerome_contacts_total : jerome_total_contacts = 33 := by
  sorry

end jerome_contacts_total_l57_57123


namespace part1_part2_l57_57753

noncomputable def condition1 : Prop :=
∀ (x y : ℝ), x - 2 * y + 1 = 0

noncomputable def condition2 (p : ℝ) : Prop :=
∀ (x y : ℝ), y^2 = 2 * p * x ∧ p > 0

noncomputable def condition3 (A B : (ℝ × ℝ)) (p : ℝ) : Prop :=
dist A B = 4 * real.sqrt 15 ∧
(∀ (x y : ℝ), (x - 2 * y + 1 = 0) ∧ (y^2 = 2 * p * x))

noncomputable def condition4 (C : ℝ) : Prop :=
F = (C, 0)

noncomputable def condition5 (M N F : (ℝ × ℝ)) : Prop :=
∃ (x1 y1 x2 y2 : ℝ), F = (1, 0) ∧
M = (x1, y1) ∧ N = (x2, y2) ∧ (x1 + x2) / 2 = C ∧ (y1 + y2) / 2 = 0 ∧ (F.1 * x1 + F.2 * y1) * (F.1 * x2 + F.2 * y2) = 0

theorem part1 :
  (∃ (p : ℝ), condition1 ∧ condition2 p ∧ condition3 (1, 0) (2, 0) p) →
  (p = 2) :=
  sorry

theorem part2 :
  (∃ (M N F : (ℝ × ℝ)), condition4 4 ∧ condition5 M N F) →
  (∀ (F : ℝ × ℝ), F = (1,0) →
  F + ℝ∧
  M=(1, C) N=(2, 409)∧
  ((M-product length eqad 0)MFN) =
  (12-8 * (2))) :=
 sorry

end part1_part2_l57_57753


namespace Paul_sold_350_pencils_l57_57151

-- Variables representing conditions
def pencils_per_day : ℕ := 100
def days_in_week : ℕ := 5
def starting_stock : ℕ := 80
def ending_stock : ℕ := 230

-- The total pencils Paul made in a week
def total_pencils_made : ℕ := pencils_per_day * days_in_week

-- The total pencils before selling any
def total_pencils_before_selling : ℕ := total_pencils_made + starting_stock

-- The number of pencils sold is the difference between total pencils before selling and ending stock
def pencils_sold : ℕ := total_pencils_before_selling - ending_stock

theorem Paul_sold_350_pencils :
  pencils_sold = 350 :=
by {
  -- The proof body is replaced with sorry to indicate a placeholder for the proof.
  sorry
}

end Paul_sold_350_pencils_l57_57151


namespace min_sticks_to_be_broken_form_square_without_breaks_l57_57596

noncomputable def total_length (n : ℕ) : ℕ := n * (n + 1) / 2

def divisible_by_4 (x : ℕ) : Prop := x % 4 = 0

theorem min_sticks_to_be_broken (n : ℕ) : n = 12 → (¬ divisible_by_4 (total_length n)) ∧ (minimal_breaks n = 2) :=
by
  intro h1
  rw h1
  have h2 : total_length 12 = 78 := by decide
  have h3 : ¬ divisible_by_4 78 := by decide
  exact ⟨h3, sorry⟩

theorem form_square_without_breaks (n : ℕ) : n = 15 → divisible_by_4 (total_length n) ∧ (minimal_breaks n = 0) :=
by
  intro h1
  rw h1
  have h2 : total_length 15 = 120 := by decide
  have h3 : divisible_by_4 120 := by decide
  exact ⟨h3, sorry⟩

end min_sticks_to_be_broken_form_square_without_breaks_l57_57596


namespace product_of_ratios_eq_l57_57604

theorem product_of_ratios_eq :
  (∃ x_1 y_1 x_2 y_2 x_3 y_3 : ℝ,
    (x_1^3 - 3 * x_1 * y_1^2 = 2006) ∧
    (y_1^3 - 3 * x_1^2 * y_1 = 2007) ∧
    (x_2^3 - 3 * x_2 * y_2^2 = 2006) ∧
    (y_2^3 - 3 * x_2^2 * y_2 = 2007) ∧
    (x_3^3 - 3 * x_3 * y_3^2 = 2006) ∧
    (y_3^3 - 3 * x_3^2 * y_3 = 2007)) →
    (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = 1 / 1003 :=
by
  sorry

end product_of_ratios_eq_l57_57604


namespace yellow_number_is_136_l57_57616

theorem yellow_number_is_136:
  ∃ (x : ℕ), (∃ a b c : ℕ,
    0 < a ∧ a < b ∧ b < c ∧ 
    x = 100*a + 10*b + c ∧ 
    let permutations := [100*a + 10*b + c, 100*a + 10*c + b, 100*b + 10*a + c, 
                         100*b + 10*c + a, 100*c + 10*a + b, 100*c + 10*b + a] in
    (1 / 6 : ℚ) * (permutations.sum) = 370 ∧ 
    let underlined := [100*a + 10*b + c, 100*a + 10*c + b, 100*b + 10*a + c].filter (λ n, n < x) in
    underlined.length = 3 ∧ 
    (1 / 3 : ℚ) * (underlined.sum) = 205) ∧
  x = 136 := sorry

end yellow_number_is_136_l57_57616


namespace find_QY_length_l57_57820

theorem find_QY_length (XY YZ ZX : ℝ) (h₁ : XY = 9) (h₂ : YZ = 8) (h₃ : ZX = 7) 
(h₄ : ∃ QX QY, ∀ (X Y Z Q : Type), ∆ QXY ∼ ∆ QYZ) :
  QY = 448 / 17 :=
begin
  sorry
end

end find_QY_length_l57_57820


namespace gravel_cost_calculation_l57_57801

def cubicYardToCubicFoot : ℕ := 27
def costPerCubicFoot : ℕ := 8
def volumeInCubicYards : ℕ := 8

theorem gravel_cost_calculation : 
  (volumeInCubicYards * cubicYardToCubicFoot * costPerCubicFoot) = 1728 := 
by
  -- This is just a placeholder to ensure the statement is syntactically correct.
  sorry

end gravel_cost_calculation_l57_57801


namespace simplify_and_evaluate_l57_57915

theorem simplify_and_evaluate
  (m : ℝ) (hm : m = 2 + Real.sqrt 2) :
  (1 - (m / (m + 2))) / ((m^2 - 4*m + 4) / (m^2 - 4)) = Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_l57_57915


namespace problem_solution_l57_57522

variable A : Set ℕ
variable h1 : ∀ n, (n ∈ A ↔ (¬(2*n ∈ A) ∧ ¬(3*n ∈ A))) ∧ ((2*n ∈ A ↔ (¬(n ∈ A) ∧ ¬(3*n ∈ A)))) ∧ (3*n ∈ A ↔ (¬(n ∈ A) ∧ ¬(2*n ∈ A)))
variable h2 : 2 ∈ A

theorem problem_solution : 13824 ∉ A := sorry

end problem_solution_l57_57522


namespace shaded_region_area_eq_4_l57_57493

-- Define the conditions
def square_side_length : ℝ := 4
def AE_AF_length : ℝ := 1
def FG_EF_ratio : ℝ := 2

-- Define the points
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (square_side_length, 0)
def point_C : ℝ × ℝ := (square_side_length, square_side_length)
def point_D : ℝ × ℝ := (0, square_side_length)
def point_E : ℝ × ℝ := (AE_AF_length, AE_AF_length)
def point_F : ℝ × ℝ := (AE_AF_length, 0)
def point_G : ℝ × ℝ := (2 * AE_AF_length, AE_AF_length)
def point_H : ℝ × ℝ := (AE_AF_length, 2 * AE_AF_length)

-- Define the area calculation
def area_shaded_region : ℝ := 
  let area_large_trapezoid := 4 * 4 / 2 -- Calculation to be derived from intersection logic
  let area_rectangle_EFGH := (2 * AE_AF_length) * AE_AF_length
  area_large_trapezoid - area_rectangle_EFGH

-- Declare the theorem
theorem shaded_region_area_eq_4 :
  area_shaded_region = 4 := sorry

end shaded_region_area_eq_4_l57_57493


namespace min_sum_of_vertex_face_l57_57944

noncomputable def is_valid_cube_vertex (vertices: Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ vertices → b ∈ vertices → c ∈ vertices → a ≠ b → b ≠ c → a ≠ c → 
  a + b + c ≥ 10

noncomputable def min_sum_cube_face : ℕ :=
  16

theorem min_sum_of_vertex_face : ∃ (vertices: Finset ℕ), vertices.card = 4 ∧ is_valid_cube_vertex vertices ∧ vertices.sum = min_sum_cube_face
:= sorry

end min_sum_of_vertex_face_l57_57944


namespace z_amount_per_rupee_l57_57314

theorem z_amount_per_rupee (x y z : ℝ) 
  (h1 : ∀ rupees_x, y = 0.45 * rupees_x)
  (h2 : y = 36)
  (h3 : x + y + z = 156)
  (h4 : ∀ rupees_x, x = rupees_x) :
  ∃ a : ℝ, z = a * x ∧ a = 0.5 := 
by
  -- Placeholder for the actual proof
  sorry

end z_amount_per_rupee_l57_57314


namespace domain_of_f_l57_57003

noncomputable def f (x : ℝ) : ℝ := real.sqrt (15 * x ^ 2 - 13 * x - 8)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, y = f x} = {x : ℝ | x ≤ -2/5 ∨ x ≥ 4/3} :=
by
  sorry

end domain_of_f_l57_57003


namespace sum_numerator_denominator_of_repeating_decimal_l57_57621

theorem sum_numerator_denominator_of_repeating_decimal :
  (let a := 5 in let b := 14 in a + b = 19) ↔
  (0.\overline{35} = (5 : ℚ) / 14) :=
sorry

end sum_numerator_denominator_of_repeating_decimal_l57_57621


namespace min_value_of_expression_is_6_l57_57872

noncomputable def min_value_of_expression (a b c : ℝ) : ℝ :=
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a

theorem min_value_of_expression_is_6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : min_value_of_expression a b c = 6 :=
by
  sorry

end min_value_of_expression_is_6_l57_57872


namespace red_balls_removed_to_certain_event_l57_57097

theorem red_balls_removed_to_certain_event (total_balls red_balls yellow_balls : ℕ) (m : ℕ)
  (total_balls_eq : total_balls = 8)
  (red_balls_eq : red_balls = 3)
  (yellow_balls_eq : yellow_balls = 5)
  (certain_event_A : ∀ remaining_red_balls remaining_yellow_balls,
    remaining_red_balls = red_balls - m → remaining_yellow_balls = yellow_balls →
    remaining_red_balls = 0) : m = 3 :=
by
  sorry

end red_balls_removed_to_certain_event_l57_57097


namespace number_leaves_remainder_3_l57_57235

theorem number_leaves_remainder_3 (n : ℕ) (h1 : 1680 % 9 = 0) (h2 : 1680 = n * 9) : 1680 % 1677 = 3 := by
  sorry

end number_leaves_remainder_3_l57_57235


namespace circle_tangent_to_ellipse_radius_l57_57681

def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def focus (a b : ℝ) : ℝ :=
  real.sqrt (a^2 - b^2)

def circle (c : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

theorem circle_tangent_to_ellipse_radius :
  let a := 6 -- semi-major axis
  let b := 5 -- semi-minor axis
  let c := (real.sqrt 11, 0) -- focus of the ellipse
  let e := ellipse a b -- equation of the ellipse
  ∃ r, circle c r ⊆ e ∧ ∀ q ∈ circle c r, q ∈ e :=
begin
  use 5,
  -- proof omitted
  sorry
end

end circle_tangent_to_ellipse_radius_l57_57681


namespace votes_stabilize_l57_57955

theorem votes_stabilize (votes : ℕ → ℕ → bool) (n : ℕ) (N : ℕ := 25):
  (∀ k, ∀ i, 0 ≤ i ∧ i < N → votes (k + 1) i = (if (votes k i = votes k ((i + N - 1) % N) ∨ votes k i = votes k ((i + 1) % N)) then votes k i else ¬ votes k i)) →
  ∃ t, ∀ i, 0 ≤ i ∧ i < N → votes t i = votes (t + 1) i :=
begin
  sorry,
end

end votes_stabilize_l57_57955


namespace math_problem_l57_57259

variable (x : ℕ)
variable (h : x + 7 = 27)

theorem math_problem : (x = 20) ∧ (((x / 5) + 5) * 7 = 63) :=
by
  have h1 : x = 20 := by {
    -- x can be solved here using the condition, but we use sorry to skip computation.
    sorry
  }
  have h2 : (((x / 5) + 5) * 7 = 63) := by {
    -- The second part result can be computed using the derived x value, but we use sorry to skip computation.
    sorry
  }
  exact ⟨h1, h2⟩

end math_problem_l57_57259


namespace find_measure_of_angle_A_l57_57850

namespace TriangleProblem

noncomputable def triangle_angle_measure (B C A : ℝ) (BC AC : ℝ) (angleB : ℝ) : Prop :=
  BC = real.sqrt 3 ∧
  AC = 1 ∧
  angleB = real.pi / 6 ∧
  (A = real.pi / 3 ∨ A = 2 * real.pi / 3)

theorem find_measure_of_angle_A :
  ∃ A, triangle_angle_measure A (real.sqrt 3) 1 (real.pi / 6) := sorry

end TriangleProblem

end find_measure_of_angle_A_l57_57850


namespace max_min_values_l57_57567

noncomputable def f (x : ℝ) : ℝ := |x + 5 / 2|

def interval : set ℝ := {x | -5 ≤ x ∧ x ≤ -2}

theorem max_min_values (x : ℝ) (h : x ∈ interval) :
  (∀ y ∈ interval, f y ≤ f (-5)) ∧ (∀ y ∈ interval, f (-5 / 2) ≤ f y) :=
by
  sorry

end max_min_values_l57_57567


namespace min_sticks_12_to_break_can_form_square_15_l57_57590

-- Problem definition for n = 12
def sticks_12 : List Nat := List.range' 1 12

theorem min_sticks_12_to_break : 
  ... (I realize I need to translate a step better) ..............
  sorry

-- Problem definition for n = 15
def sticks_15 : List Nat := List.range' 1 15

theorem can_form_square_15 : 
  ... (implementing a nice explanation)
  sorry

end min_sticks_12_to_break_can_form_square_15_l57_57590


namespace correct_transformation_l57_57261

theorem correct_transformation (a b : ℝ) (h : a ≠ 0) : 
  (a^2 / (a * b) = a / b) :=
by sorry

end correct_transformation_l57_57261


namespace cannot_equal_by_same_steps_l57_57630

theorem cannot_equal_by_same_steps (a b : ℕ) (h₁ : a = 19) (h₂ : b = 98) : ¬ ∃ s : ℕ, ∃ f : fin s → (ℕ → ℕ), (∀ i, f i = (λ x, x * x) ∨ f i = (λ x, x + 1)) ∧ (iterate (f fin.last s) s 19 = iterate (f fin.last s) s 98) :=
by
  intro ⟨s, f, hf, heq⟩,
  sorry

end cannot_equal_by_same_steps_l57_57630


namespace AE_ED_equals_AF_FB_l57_57543

noncomputable theory

def is_isosceles_trapezoid (A B C D : Point) : Prop :=
  -- Assuming that it’s defined elsewhere
  sorry

def AE_ED_eq_AF_FB
  (A B C D E F : Point) 
  (h_isosceles_abcd : is_isosceles_trapezoid A B C D)
  (h_on_ad : E ∈ line_segment A D)
  (h_on_ab : F ∈ line_segment A B)
  (h_isosceles_cdef : is_isosceles_trapezoid C D E F) : Prop :=
  AE E * ED E D = AF A F * FB F B

theorem AE_ED_equals_AF_FB
  (A B C D E F : Point)
  (h_isosceles_abcd : is_isosceles_trapezoid A B C D)
  (h_on_ad : E ∈ line_segment A D)
  (h_on_ab : F ∈ line_segment A B)
  (h_isosceles_cdef : is_isosceles_trapezoid C D E F) :
  AE_ED_eq_AF_FB A B C D E F h_isosceles_abcd h_on_ad h_on_ab h_isosceles_cdef :=
sorry

end AE_ED_equals_AF_FB_l57_57543


namespace triangle_area_calculation_l57_57059

-- Define the variables and conditions given in the problem
variables (x y : ℝ)
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

def F1 := (-5, 0) : ℝ × ℝ
def F2 := (5, 0) : ℝ × ℝ
variables (P : ℝ × ℝ)
def dist (A B : ℝ × ℝ) : ℝ := (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

-- Conditions
axiom h1 : hyperbola P.1 P.2
axiom h2 : dist P F2 = 8 / 15 * dist F1 F2

-- Goal statement
theorem triangle_area_calculation : (1/2) * dist P F1 * dist F1 F2 * (real.sin (real.arcsin 8 / 17)) = 80 / 3 :=
sorry

end triangle_area_calculation_l57_57059


namespace negation_of_at_most_one_solution_is_at_least_two_solutions_l57_57936

-- Define the initial proposition
def at_most_one_solution : Prop := ∀ x y, solution x ∧ solution y → x = y

-- Define the negation of the initial proposition
def at_least_two_solutions : Prop := ∃ x y, solution x ∧ solution y ∧ x ≠ y

-- The main theorem to prove the negation
theorem negation_of_at_most_one_solution_is_at_least_two_solutions :
  ¬ at_most_one_solution ↔ at_least_two_solutions :=
by
  sorry

end negation_of_at_most_one_solution_is_at_least_two_solutions_l57_57936


namespace g_of_58_l57_57565

def g : ℝ → ℝ := sorry

axiom g_fun_eq (x y : ℝ) : g (x * y) = x * g y
axiom g_one : g 1 = 40

theorem g_of_58 : g 58 = 2320 :=
by
  -- We can use the functional equation and the given value of g(1) to prove g(58) = 2320.
  have h : g (58 * 1) = 58 * g 1, from g_fun_eq 58 1,
  rw [←mul_one 58] at h,
  rw g_one at h,
  exact h

end g_of_58_l57_57565


namespace three_digit_numbers_divisible_by_9_l57_57444

theorem three_digit_numbers_divisible_by_9 : 
  let smallest := 108
  let largest := 999
  let common_diff := 9
  -- Using the nth-term formula for an arithmetic sequence
  -- nth term: l = a + (n-1) * d
  -- For l = 999, a = 108, d = 9
  -- (999 = 108 + (n-1) * 9) -> (n-1) = 99 -> n = 100
  -- Hence, the number of such terms (3-digit numbers) in the sequence is 100.
  ∃ n, n = 100 ∧ (largest = smallest + (n-1) * common_diff)
by {
  let smallest := 108
  let largest := 999
  let common_diff := 9
  use 100
  sorry
}

end three_digit_numbers_divisible_by_9_l57_57444


namespace f_at_2_f_pos_solution_set_l57_57386

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - (3 - a) * x + 2 * (1 - a)

-- Question (I)
theorem f_at_2 : f a 2 = 0 := by sorry

-- Question (II)
theorem f_pos_solution_set :
  (∀ x, (a < -1 → (f a x > 0 ↔ (x < 2 ∨ 1 - a < x))) ∧
       (a = -1 → ¬(f a x > 0)) ∧
       (a > -1 → (f a x > 0 ↔ (1 - a < x ∧ x < 2)))) := 
by sorry

end f_at_2_f_pos_solution_set_l57_57386


namespace cubic_polynomial_root_l57_57347

theorem cubic_polynomial_root (x p q r : ℝ) (hp : p = 729) (hq : q = 27) (hr : r = 26) :
  27 * x^3 - 11 * x^2 - 11 * x - 3 = 0 → x = (real.cbrt p + real.cbrt q + 1) / r → p + q + r = 782 :=
by 
sorry

end cubic_polynomial_root_l57_57347


namespace cos_b_plus_sqrt_two_cos_c_range_l57_57518

theorem cos_b_plus_sqrt_two_cos_c_range (A B C : ℝ)
  (h1 : sin A = √2 / 2)
  (h2 : 0 < C) (h3 : C < π) 
  : (∀ B C : ℝ, 0 < B ∧ B + C = π - A) →
    (cos B + √2 * cos C) ∈ set.Icc 0 1 ∪ set.Ioo 2 (√5) :=
begin
  sorry
end

end cos_b_plus_sqrt_two_cos_c_range_l57_57518


namespace tian_ji_winning_probability_l57_57126

-- Definitions for the conditions
def is_better (x y : Horse) : Prop := sorry
noncomputable def random_horse : Horse := sorry

-- King Qi's horses
def a : Horse := sorry
def b : Horse := sorry
def c : Horse := sorry

-- Tian Ji's horses
def A : Horse := sorry
def B : Horse := sorry
def C : Horse := sorry

-- Conditions
axiom top_horse_condition : is_better A b ∧ is_better a A
axiom middle_horse_condition : is_better B c ∧ is_better b B
axiom lower_horse_condition : is_better c C

-- Question: Prove the probability of Tian Ji winning is 1/3
theorem tian_ji_winning_probability : 
  (probability (win (random_horse : A | B | C) (random_horse : a | b | c)) : ℚ) = 1/3 :=
sorry

end tian_ji_winning_probability_l57_57126


namespace national_flag_length_l57_57927

-- Definitions from the conditions specified in the problem
def width : ℕ := 128
def ratio_length_to_width (L W : ℕ) : Prop := L / W = 3 / 2

-- The main theorem to prove
theorem national_flag_length (L : ℕ) (H : ratio_length_to_width L width) : L = 192 :=
by
  sorry

end national_flag_length_l57_57927


namespace sum_x_coords_Q3_l57_57645

theorem sum_x_coords_Q3 (x_coords_Q1 : Fin 50 → ℝ) 
  (h_sum_coords_Q1 : (∑ i, x_coords_Q1 i) = 1000) : 
  let x_coords_Q2 := fun i => (x_coords_Q1 i + x_coords_Q1 ((i + 1) % 50)) / 2 in
  let x_coords_Q3 := fun i => (x_coords_Q2 i + x_coords_Q2 ((i + 1) % 50)) / 2 in
  (∑ i, x_coords_Q3 i) = 1000 := 
  sorry

end sum_x_coords_Q3_l57_57645


namespace min_tan_prod_l57_57096

variable {A B C : Real}

-- Conditions: ∆ABC is an acute triangle and sin A = 2 sin B sin C
variable (h1 : sin A = 2 * sin B * sin C)
variable (h2 : 0 < A ∧ A < π / 2)
variable (h3 : 0 < B ∧ B < π / 2)
variable (h4 : 0 < C ∧ C < π / 2)
variable (h5 : A + B + C = π)

-- Proof target: minimum value of tan A * tan B * tan C
theorem min_tan_prod : ∃ t : Real, t = tan A * tan B * tan C ∧ ∀ x : Real, x = tan A * tan B * tan C → x ≥ 8 :=
by
  sorry

end min_tan_prod_l57_57096


namespace count_3_digit_numbers_divisible_by_9_l57_57454

theorem count_3_digit_numbers_divisible_by_9 : 
  (finset.filter (λ x : ℕ, x % 9 = 0) (finset.Icc 100 999)).card = 100 := 
sorry

end count_3_digit_numbers_divisible_by_9_l57_57454
