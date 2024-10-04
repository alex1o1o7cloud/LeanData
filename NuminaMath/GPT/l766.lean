import Mathlib
import Mathlib.Algebra.Combinatorics.Compositions
import Mathlib.Algebra.CubicEquation
import Mathlib.Algebra.Fact
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.Powers
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Module.LinearMap
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometric.Basic
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Derangements.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Probability
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Probability.Theory
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.NormNum

namespace sum_a3_a4_a5_l766_766189

-- Definitions based on given conditions
def sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, 2 * S n = a (n + 1) - 1)

-- The proof problem to be translated into Lean
theorem sum_a3_a4_a5 : 
  ∃ (a S : ℕ → ℕ), sequence a S ∧ (a 3 + a 4 + a 5 = 117) :=
by
  -- Definitions (Based on the conditions stated in the problem, should be inferred within proof)
  sorry

end sum_a3_a4_a5_l766_766189


namespace cylinder_volume_is_correct_l766_766481

-- define the given conditions
def cylinder_diameter := 3
def cylinder_height := 3

-- calculate the radius
def cylinder_radius := cylinder_diameter / 2

-- define the volume using the given formula
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

-- state the theorem, the volume for the given cylinder radius and height
theorem cylinder_volume_is_correct :
  cylinder_volume cylinder_radius cylinder_height = (27 / 4) * π :=
by
  -- proof is skipped
  sorry

end cylinder_volume_is_correct_l766_766481


namespace gecko_sales_ratio_l766_766888

theorem gecko_sales_ratio (x : ℕ) (h1 : 86 + x = 258) : 86 / Nat.gcd 172 86 = 1 ∧ 172 / Nat.gcd 172 86 = 2 := by
  sorry

end gecko_sales_ratio_l766_766888


namespace add_zero_eq_self_l766_766815

theorem add_zero_eq_self (n x : ℤ) (h : n + x = n) : x = 0 := 
sorry

end add_zero_eq_self_l766_766815


namespace area_of_square_l766_766554

theorem area_of_square (side : ℝ) (h : side = 6) : (side * side) = 36 := by
  have h1 : side = 6 := h
  rw h1
  norm_num
  done

end area_of_square_l766_766554


namespace roman_coins_left_l766_766318

theorem roman_coins_left (X Y : ℕ) (h1 : X * Y = 50) (h2 : (X - 7) * Y = 28) : X - 7 = 8 :=
by
  sorry

end roman_coins_left_l766_766318


namespace sum_of_digits_l766_766790

variable (a b c d e f : ℕ)

theorem sum_of_digits :
  ∀ (a b c d e f : ℕ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧
    100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 →
    a + b + c + d + e + f = 28 := 
by
  intros a b c d e f h
  sorry

end sum_of_digits_l766_766790


namespace min_distance_from_A_to_circle_l766_766485

noncomputable def minimum_distance (A : Point) (O : Point) (M : Point) (B : Point) : ℝ :=
  distance A M

theorem min_distance_from_A_to_circle (
    A B M : Point, α t1 t2 : ℝ,
    hcirc : ∀ x y, (x + 1)^2 + y^2 = 7/9,
    hpara : C y = y^2 = 4 * x,
    hcollinear : ∀ α, ∀ t, B = M + 3 * (A - M),
    hdist : ∀ t1 t2, t1 + t2 = 4 * cos α / sin α^2 ∧ t1 * t2 = 4 / sin α^2
) : minimum_distance (x : ℝ) (α : ℝ) = sqrt(7) / 3 :=
begin
  sorry
end

end min_distance_from_A_to_circle_l766_766485


namespace min_tangent_length_l766_766107

open Real

theorem min_tangent_length : 
  let line : ℝ → ℝ := λ x, x + 1
  let circle_center := (3 : ℝ, 0 : ℝ)
  let radius := 1
  let distance_line := abs (4 / Real.sqrt 2)
  let tangent_length := Real.sqrt ((distance_line)^2 - radius^2)
  tangent_length = Real.sqrt 7 :=
by
  sorry

end min_tangent_length_l766_766107


namespace sequence_tenth_term_l766_766308

open Nat

/-- A sequence S such that 
  S(1) = 2, 
  S(2) = 3, 
  S(3) = 5, 
  S(4) = 8, 
  S(5) = 13, 
  and for n > 2, S(n) = S(n-1) + S(n-2).
    
  Prove that S(10) = 144 --/
theorem sequence_tenth_term :
  let S : ℕ → ℕ := λ n, if n = 1 then 2 else if n = 2 then 3 else S (n - 1) + S (n - 2)
  in S 10 = 144 :=
by
  sorry

end sequence_tenth_term_l766_766308


namespace overlap_difference_l766_766657

namespace GeometryBiology

noncomputable def total_students : ℕ := 350
noncomputable def geometry_students : ℕ := 210
noncomputable def biology_students : ℕ := 175

theorem overlap_difference : 
    let max_overlap := min geometry_students biology_students;
    let min_overlap := geometry_students + biology_students - total_students;
    max_overlap - min_overlap = 140 := 
by
  sorry

end GeometryBiology

end overlap_difference_l766_766657


namespace distance_A_B_l766_766421

theorem distance_A_B (d : ℝ)
  (speed_A : ℝ := 100) (speed_B : ℝ := 90) (speed_C : ℝ := 75)
  (location_A location_B : point) (is_at_A : location_A = point_A) (is_at_B : location_B = point_B)
  (t_meet_AB : ℝ := d / (speed_A + speed_B))
  (t_meet_AC : ℝ := t_meet_AB + 3)
  (distance_AC : ℝ := speed_A * 3)
  (distance_C : ℝ := speed_C * t_meet_AC) :
  d = 650 :=
by {
  sorry
}

end distance_A_B_l766_766421


namespace part1_part2_l766_766215

def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

theorem part1 (a : ℝ) :
  (Set.Icc (a - 7) (a - 3)) = (Set.Icc (-5 : ℝ) (-1 : ℝ)) -> a = 2 :=
by
  intro h
  sorry

theorem part2 (m : ℝ) :
  (∃ x_0 : ℝ, f x_0 2 < 4 * m + m^2) -> (m < -5 ∨ m > 1) :=
by
  intro h
  sorry

end part1_part2_l766_766215


namespace problem1_problem2_l766_766175

variable {α : Type} [Field α]
variable {tan : α → α}

-- Condition
def condition (α : ℝ) : Prop := (1 + tan α) / (1 - tan α) = 2

-- Target expressions
def expr1 (α : ℝ) : ℝ := (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α - Real.cos α)
def expr2 (α : ℝ) : ℝ := Real.sin α * Real.cos α + 2

-- Proof problem
theorem problem1 (α : ℝ) (h : condition α) : expr1 α = 5 := by
  sorry

theorem problem2 (α : ℝ) (h : condition α) : expr2 α = 23 / 10 := by
  sorry

end problem1_problem2_l766_766175


namespace remaining_families_own_cats_each_l766_766243

theorem remaining_families_own_cats_each :
  ∀ (total_families families_with_2_dogs families_with_1_dog total_pets : ℕ),
    total_families = 50 →
    families_with_2_dogs = 15 →
    families_with_1_dog = 20 →
    total_pets = 80 →
    let families_with_dogs := families_with_2_dogs + families_with_1_dog in
    let remaining_families := total_families - families_with_dogs in
    let total_dogs := families_with_2_dogs * 2 + families_with_1_dog in
    let total_cats := total_pets - total_dogs in
    let cats_per_remaining_family := total_cats / remaining_families in
    cats_per_remaining_family = 2 :=
by
  intros total_families families_with_2_dogs families_with_1_dog total_pets
         h1 h2 h3 h4
  simp only [h1, h2, h3, h4]
  sorry

end remaining_families_own_cats_each_l766_766243


namespace solve_system_l766_766142

-- Definitions of constants
def x1 := (35 + Real.sqrt 1321) / 24
def x2 := (35 - Real.sqrt 1321) / 24
def y1 := (-125 - 7 * Real.sqrt 1321) / 72
def y2 := (-125 + 7 * Real.sqrt 1321) / 72

-- Main theorem
theorem solve_system :
  (7 * x1 + 3 * y1 = 5 ∧ 4 * x1^2 + 5 * y1 = 9) ∧
  (7 * x2 + 3 * y2 = 5 ∧ 4 * x2^2 + 5 * y2 = 9) :=
  sorry

end solve_system_l766_766142


namespace graph_translation_l766_766789

theorem graph_translation (x : ℝ) : 
  cos (2 * x) = sin (2 * (x - π / 12) + π / 2) ↔ sin (2 * x + π / 3) :=
sorry

end graph_translation_l766_766789


namespace sum_primes_between_1_50_special_conditions_l766_766947

theorem sum_primes_between_1_50_special_conditions :
  let primes_between_1_50 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let special_primes := primes_between_1_50.filter (λ p, p % 4 = 1 ∧ p % 3 = 2)
  special_primes.sum = 92 :=
by sorry

end sum_primes_between_1_50_special_conditions_l766_766947


namespace polynomial_solution_l766_766001

def h (x : ℝ) : ℝ := x^3 - 2 * x^2 + 4 * x - 1

def j (x : ℝ) (p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem polynomial_solution : ∃ (p q r : ℝ), (j x p q r = x^3 + 4 * x^2 + 8 * x + 7) ∧ 
                                     (-p = -4) ∧
                                     (-q = -8) ∧
                                     (-r = -7) ∧
                                     (∀ t : ℝ, h(t+2) = 0 → j(t, p, q, r) = 0) :=
by
  sorry

end polynomial_solution_l766_766001


namespace range_of_expression_l766_766174

variable {a b : ℝ}

theorem range_of_expression 
  (h₁ : -1 < a + b) (h₂ : a + b < 3)
  (h₃ : 2 < a - b) (h₄ : a - b < 4) :
  -9 / 2 < 2 * a + 3 * b ∧ 2 * a + 3 * b < 13 / 2 := 
sorry

end range_of_expression_l766_766174


namespace solution_irrational_sum_l766_766233

noncomputable def irrational_pair_satisfy_sum : Prop :=
∃ (a b : ℝ), irrational a ∧ irrational b ∧ a + b = 2

theorem solution_irrational_sum :
  irrational_pair_satisfy_sum :=
begin
  use [Real.pi, 2 - Real.pi],
  split,
  { apply irrational_pi },
  split,
  { apply irrational.sub, apply irrational_two, apply irrational_pi },
  { norm_num, rw sub_self }
end

end solution_irrational_sum_l766_766233


namespace reduce_to_single_digit_l766_766181

theorem reduce_to_single_digit (n : ℕ) : ∃ k ≤ 10, ∃ m, m < 10 ∧ iterate (λ x, sum_of_digits x) k n = m :=
sorry

end reduce_to_single_digit_l766_766181


namespace compound_statement_false_l766_766997

theorem compound_statement_false (p q : Prop) (h : ¬ (p ∧ q)) : ¬ p ∨ ¬ q :=
sorry

end compound_statement_false_l766_766997


namespace factor_expression_l766_766526

theorem factor_expression (x : ℝ) :
  (3*x^3 + 48*x^2 - 14) - (-9*x^3 + 2*x^2 - 14) =
  2*x^2 * (6*x + 23) :=
by
  sorry

end factor_expression_l766_766526


namespace exponent_of_7_eq_1_l766_766471

theorem exponent_of_7_eq_1 : ∃ (x : ℤ), 7 ^ x = 1 := by
  exists 0
  sorry

end exponent_of_7_eq_1_l766_766471


namespace multiple_of_weight_lifted_l766_766140

variable (F : ℝ) (M : ℝ)

theorem multiple_of_weight_lifted 
  (H1: ∀ (B : ℝ), B = 2 * F) 
  (H2: ∀ (B : ℝ), ∀ (W : ℝ), W = 3 * B) 
  (H3: ∃ (B : ℝ), (3 * B = 600)) 
  (H4: M * F = 150) : 
  M = 1.5 :=
by
  sorry

end multiple_of_weight_lifted_l766_766140


namespace altitude_harmonic_mean_l766_766314

-- Define a structure for a Triangle
structure Triangle :=
(a b c : ℝ) -- side lengths
(t : ℝ) -- area
(s : ℝ) -- semiperimeter

-- Altitude from vertex A
def h_a (T : Triangle) : ℝ :=
  2 * T.t / T.a

-- Radii of the excircles opposite vertices B and C
def r_b (T : Triangle) : ℝ :=
  T.t / (T.s - T.b)

def r_c (T : Triangle) : ℝ :=
  T.t / (T.s - T.c)

-- Harmonic mean of r_b and r_c
def H (T : Triangle) : ℝ :=
  2 / (1 / r_b T + 1 / r_c T)

-- The main theorem statement
theorem altitude_harmonic_mean (T : Triangle) : h_a T = H T :=
  sorry

end altitude_harmonic_mean_l766_766314


namespace sin_equals_cos_630_l766_766146

open Real

theorem sin_equals_cos_630 (n : ℤ) (h_range : -180 ≤ n ∧ n ≤ 180) (h_eq : sin (n * (π / 180)) = cos (630 * (π / 180))): 
  n = 0 ∨ n = 180 ∨ n = -180 :=
by
  sorry

end sin_equals_cos_630_l766_766146


namespace inequality_problem_l766_766991

theorem inequality_problem
  (n : ℕ)
  (x : Fin n → ℝ)
  (hx : ∀ i, 0 < x i)
  (hsum : ∑ i, x i = 1)
  : (∑ i, x i / Real.sqrt (1 - x i)) ≥ (1 / Real.sqrt (n - 1)) * (∑ i, Real.sqrt (x i)) :=
sorry

end inequality_problem_l766_766991


namespace sum_of_distances_not_equal_l766_766178

theorem sum_of_distances_not_equal {P : Type} [metric_space P] (A B : P) 
  (points : fin 45 → P) (hA_ne_B : A ≠ B) 
  (h : ∀ i, ¬(metric_segment A B (points i))) :
  ∑ i, dist (points i) A ≠ ∑ i, dist (points i) B :=
sorry

end sum_of_distances_not_equal_l766_766178


namespace no_daily_coverage_l766_766248

theorem no_daily_coverage (ranks : Nat → Nat)
  (h_ranks_ordered : ∀ i, ranks (i+1) ≥ 3 * ranks i)
  (h_cycle : ∀ i, ∃ N : Nat, ranks i = N ∧ ∃ k : Nat, k = N ∧ ∀ m, m % (2 * N) < N → (¬ ∃ j, ranks j ≤ N))
  : ¬ (∀ d : Nat, ∃ j : Nat, (∃ k : Nat, d % (2 * (ranks j)) < ranks j))
  := sorry

end no_daily_coverage_l766_766248


namespace sqrt_simplification_l766_766373

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l766_766373


namespace required_average_for_tickets_l766_766894

theorem required_average_for_tickets 
  (june_score : ℝ) (patty_score : ℝ) (josh_score : ℝ) (henry_score : ℝ)
  (num_children : ℝ) (total_score : ℝ) (average_score : ℝ) (S : ℝ)
  (h1 : june_score = 97) (h2 : patty_score = 85) (h3 : josh_score = 100) 
  (h4 : henry_score = 94) (h5 : num_children = 4) 
  (h6 : total_score = june_score + patty_score + josh_score + henry_score)
  (h7 : average_score = total_score / num_children) 
  (h8 : average_score = 94)
  : S ≤ 94 :=
sorry

end required_average_for_tickets_l766_766894


namespace angle_between_vectors_magnitude_expression_l766_766202

open Real

def vec {n : Type} [NormedAddCommGroup n] [NormedSpace ℝ n] := n

variables (a b : vec ℝ) (θ : ℝ)

-- Given Conditions
axiom norm_a : norm a = 4
axiom norm_b : norm b = 3
axiom dot_ab : inner a b = 6

-- First Part: Angle between vectors
theorem angle_between_vectors :
  θ = real.arccos (6 / (4 * 3)) :=
sorry

-- Second Part: Magnitude of vector expression
theorem magnitude_expression :
  norm (3 • a - 4 • b) = 12 :=
sorry

end angle_between_vectors_magnitude_expression_l766_766202


namespace parabola_equation_and_cos_angle_OAF_l766_766262

-- Define the parabola C with given conditions
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

-- Define the condition that p is greater than 0
axiom p_pos (p : ℝ) : p > 0

-- Define the focus of the parabola 
def parabola_focus (p : ℝ) := (real.sqrt p, 0)

-- Define the points P and F symmetric about the y-axis
def symmetric_points (P F : ℝ × ℝ) := P = (-F.1, F.2)

-- Define a circle with diameter PF
def circle_diameter (P F : ℝ × ℝ) (x y : ℝ) := (x - (P.1 + F.1) / 2)^2 + (y - (P.2 + F.2) / 2)^2 = ((P.1 - F.1)^2 + (P.2 - F.2)^2) / 4

-- Define the intersection of the circle and parabola at point A
def intersection (x₀ y₀ : ℝ) (p : ℝ) (P F : ℝ × ℝ) (hp : parabola p x₀ y₀) (hc : circle_diameter P F x₀ y₀) : Prop := true

-- Prove the equivalent proof problems
theorem parabola_equation_and_cos_angle_OAF :
  ∀ (p : ℝ) (P F : ℝ × ℝ) (x₀ y₀ : ℝ),
    p_pos p →
    parabola_focus p = F →
    symmetric_points P F →
    ∃ x₀ y₀, intersection x₀ y₀ p P F (by sorry) (by sorry) ∧
    y₀^2 = 8 * x₀ ∧
    let O := (0, 0) in
    let OA := (x₀, y₀) in
    let FA := (x₀ - 2, y₀) in
    cos_angle O OA FA = (real.sqrt 5 - 1) / 2 :=
begin
  sorry
end

-- Define the cosine of the angle between two vectors
def cos_angle (O A F : ℝ × ℝ) := 
  let OA := (A.1 - O.1, A.2 - O.2) in
  let FA := (A.1 - F.1, A.2 - F.2) in
  ((OA.1 * FA.1 + OA.2 * FA.2) / (real.sqrt (OA.1^2 + OA.2^2) * real.sqrt (FA.1^2 + FA.2^2)))

end parabola_equation_and_cos_angle_OAF_l766_766262


namespace proposition_p_false_range_a_l766_766313

theorem proposition_p_false_range_a :
  (∀ a : ℝ, (¬ ∃ x_0 ∈ set.Icc (0 : ℝ) (Real.pi / 4), sin (2 * x_0) + cos (2 * x_0) > a)
           ↔ a ≥ Real.sqrt 2) :=
by
  sorry

end proposition_p_false_range_a_l766_766313


namespace rectangle_ratio_proof_l766_766566

theorem rectangle_ratio_proof
  (s : ℝ)
  (h₁: 9 * s^2 = (3 * s)^2):
  (let x := (3 * s - s) in let y := s / 3 in x / y = 6) :=
by
  sorry

end rectangle_ratio_proof_l766_766566


namespace find_R_l766_766992

theorem find_R (Q : ℝ) (R : ℝ) (h : Q = 2) :
  (∑ n in finset.range 1998, Q / (real.sqrt (n * Q) + real.sqrt ((n + 1) * Q))) = 
  R / (real.sqrt Q + real.sqrt (1999 * Q)) → 
  R = 3996 :=
by 
  intro hSum,
  rw h at hSum,
  sorry  -- Skip proof, as instructed

end find_R_l766_766992


namespace sqrt_sqrt_4_is_sqrt_2_l766_766012

theorem sqrt_sqrt_4_is_sqrt_2 :
  ∀ x ≥ 0, sqrt (sqrt 4) = ±sqrt 2 :=
begin
  -- Given condition
  let x := sqrt 4,
  have h1 : sqrt 4 = 2 := sqrt_eq_iff_sqr_eq.mpr (by norm_num), -- sqrt 4 = 2
  -- Prove
  have h2 : sqrt x = ±sqrt 2 := sorry
end

end sqrt_sqrt_4_is_sqrt_2_l766_766012


namespace alternating_operations_l766_766864

theorem alternating_operations (x : ℝ) (n : ℕ) (h : x ≠ 0) : 
  ∃ y : ℝ, y = x ^ ((-2) ^ n) :=
by
  sorry

end alternating_operations_l766_766864


namespace parallelogram_opposite_angles_equal_l766_766664

theorem parallelogram_opposite_angles_equal (A B C D : Type) [Parallelogram A B C D] (hA : angle A = 50) : angle C = 50 := 
by
  sorry

end parallelogram_opposite_angles_equal_l766_766664


namespace simplify_expression_l766_766382

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l766_766382


namespace max_profit_price_l766_766816

-- Define the initial conditions
def purchase_price : ℝ := 80
def initial_selling_price : ℝ := 90
def initial_sales_volume : ℝ := 400
def price_increase_effect : ℝ := 1
def sales_volume_decrease : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ :=
  let selling_price := initial_selling_price + x
  let sales_volume := initial_sales_volume - x * sales_volume_decrease
  let profit_per_item := selling_price - purchase_price
  profit_per_item * sales_volume

-- The statement that needs to be proved
theorem max_profit_price : ∃ x : ℝ, x = 10 ∧ (initial_selling_price + x = 100) := by
  sorry

end max_profit_price_l766_766816


namespace collinear_O_P_Q_slopes_sum_l766_766460

variables {a b : ℝ} (h_ab : 0 < b ∧ b < a)
variables (A B P Q O : Type*) [metric_space A] [metric_space B] [metric_space P] [metric_space Q] [metric_space O]
variables (ellipse_eq : ∀ p : O, (p.x^2 / a^2 + p.y^2 / b^2 = 1))
variables (hyperbola_eq : ∀ p : O, (p.x^2 / a^2 - p.y^2 / b^2 = 1))
variables (λ : ℝ) (h_λ : |λ| > 1)

axiom vertices : ∃ A B : O, true -- vertices A and B
axiom points : ∃ P Q : O, (P ≠ A ∧ P ≠ B ∧ Q ≠ A ∧ Q ≠ B)
axiom vector_condition : (∀ v, (\overrightarrow "AP" + \overrightarrow "BP" = λ * (\overrightarrow "AQ" + \overrightarrow "BQ")))

theorem collinear_O_P_Q : collinear ℝ {O, P, Q} :=
sorry

theorem slopes_sum : 
  ∀ k1 k2 k3 k4 : ℝ, slopes A B P Q k1 k2 k3 k4 → k1 + k2 + k3 + k4 = 0 :=
sorry

end collinear_O_P_Q_slopes_sum_l766_766460


namespace solution_set_of_inequality_l766_766689

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h1 : ∀ x : ℝ, deriv f x = 2 * f x)
                                    (h2 : f 0 = 1) :
  { x : ℝ | f (Real.log (x^2 - x)) < 4 } = { x | -1 < x ∧ x < 0 ∨ 1 < x ∧ x < 2 } :=
by {
  sorry
}

end solution_set_of_inequality_l766_766689


namespace sqrt_mixed_number_eq_l766_766546

noncomputable def mixed_number : ℝ := 8 + 1 / 9

theorem sqrt_mixed_number_eq : Real.sqrt (8 + 1 / 9) = Real.sqrt 73 / 3 := by
  sorry

end sqrt_mixed_number_eq_l766_766546


namespace athenian_is_orthocentric_l766_766105

noncomputable def is_athenian_set (A1 A2 A3 A4 : Point) : Prop :=
  ∃ P : Point,
    (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 → ¬ Collinear P (line A_i A_j)) ∧
    (∀ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧
                j ≠ k ∧ j ≠ l ∧ k ≠ l →
      let M_ij := midpoint (A_i A_j),
          M_kl := midpoint (A_k A_l) in
      Perpendicular (line P M_ij) (line P M_kl))

theorem athenian_is_orthocentric (A1 A2 A3 A4 : Point) :
  is_athenian_set A1 A2 A3 A4 →
  is_orthocentric_set A1 A2 A3 A4 ∧
  is_nine_point_circle (nine_point_circle (triangle A1 A2 A3)) = nine_point_circle (triangle A2 A3 A4) := by
  sorry

end athenian_is_orthocentric_l766_766105


namespace base_prime_representation_1260_l766_766434

theorem base_prime_representation_1260 : 
  let base_prime_representation (n : ℕ) : ℕ :=
    if n = 0 then 0 else
    let exps := 
      ((Nat.primeFactors 2).find ?_).getD 0 * 1000 +
      ((Nat.primeFactors 3).find ?_).getD 0 * 100 +
      ((Nat.primeFactors 5).find ?_).getD 0 * 10 +
      ((Nat.primeFactors 7).find ?_).getD 0
    in exps in
  base_prime_representation 1260 = 2212 :=
by
  sorry

end base_prime_representation_1260_l766_766434


namespace collinearity_of_A_I_A1_l766_766510

open Triangle

theorem collinearity_of_A_I_A1 {A B C I H B1 C1 B2 C2 K A1 : Point} (h1 : incenter I (Triangle.mk A B C)) 
  (h2 : orthocenter H (Triangle.mk A B C))
  (h3 : midpoint B1 A C)
  (h4 : midpoint C1 A B)
  (h5 : on_ray I B1 B2)
  (h6 : on_ray I C1 C2)
  (h7 : intersects_at (line_through B2 C2) (line_through B C) K)
  (h8 : circumcenter A1 (Triangle.mk B H C)) :
  (area (Triangle.mk B K B2) = area (Triangle.mk C K C2)) ↔ collinear {A, I, A1} :=
sorry

end collinearity_of_A_I_A1_l766_766510


namespace ellipse_range_l766_766688

noncomputable def ellipse_eccentricity_range (k : ℝ) : Prop :=
  (0 < k ∧ k < 3) ∨ (k > 16 / 3)

theorem ellipse_range (k : ℝ) (e : ℝ)
  (h1 : 1 / 2 < e) (h2 : e < 1)
  (ellipse_eq : ∀ x y : ℝ, x^2 / 4 + y^2 / k = 1 ↔ 
    ((4 > k > 0) ∧ (e^2 = (4 - k) / 4)) ∨ 
    ((k > 4) ∧ (e^2 = (k - 4) / k))):
  ellipse_eccentricity_range k :=
begin
  sorry
end

end ellipse_range_l766_766688


namespace eval_expression_l766_766810

theorem eval_expression : 4 * (8 - 3) - 7 = 13 :=
by
  sorry

end eval_expression_l766_766810


namespace num_integers_with_three_consecutive_odd_sums_l766_766627

/-- There are exactly 9 integers N less than 2000 that can be written as the sum
of j consecutive positive odd integers from exactly 3 values of j ≥ 1. -/
theorem num_integers_with_three_consecutive_odd_sums :
  ∃ (N : Finset ℕ), 
  (∀ n ∈ N, 
      ∃ j (n m : ℕ), 
        (j ≥ 1 ∧ n < 2000 ∧ n = j * (2 * m + j))) ∧ 
  |N| = 9 :=
by
  sorry

end num_integers_with_three_consecutive_odd_sums_l766_766627


namespace distance_from_R_to_midpoint_PQ_l766_766661

theorem distance_from_R_to_midpoint_PQ {P Q R : Type*} [metric_space P] [metric_space Q] [metric_space R]
  (PQ PR QR : ℕ)
  (h1 : PQ = 15) (h2 : PR = 9) (h3 : QR = 12) :
  distance R (midpoint P Q) = 7.5 := 
sorry

end distance_from_R_to_midpoint_PQ_l766_766661


namespace three_buses_interval_l766_766038

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end three_buses_interval_l766_766038


namespace angle_CAD_eq_30_l766_766919

variable (A B C D E : Point)
variable [hConvex : ConvexQuad A B C D]
variable (hIntersect : IntersectingDiagonals A B C D E)
variable (hEqualSides : AB = BC)
variable (hBisector : AngleBisector DB (∠ADC))
variable (hAngleABC : ∠ABC = 100)
variable (hAngleBEA : ∠BEA = 70)

theorem angle_CAD_eq_30 :
  ∠CAD = 30 :=
sorry

end angle_CAD_eq_30_l766_766919


namespace shares_sum_4000_l766_766593

variables (w x y z : ℝ)

def relation_z_w : Prop := z = 1.20 * w
def relation_y_z : Prop := y = 1.25 * z
def relation_x_y : Prop := x = 1.35 * y
def w_after_3_years : ℝ := 8 * w
def z_after_3_years : ℝ := 8 * z
def y_after_3_years : ℝ := 8 * y
def x_after_3_years : ℝ := 8 * x

theorem shares_sum_4000 (w : ℝ) :
  relation_z_w w z →
  relation_y_z z y →
  relation_x_y y x →
  x_after_3_years x + y_after_3_years y + z_after_3_years z + w_after_3_years w = 4000 :=
by
  intros h_z_w h_y_z h_x_y
  rw [relation_z_w, relation_y_z, relation_x_y] at *
  sorry

end shares_sum_4000_l766_766593


namespace equation_of_motion_l766_766774

section MotionLaw

variable (t s : ℝ)
variable (v : ℝ → ℝ)
variable (C : ℝ)

-- Velocity function
def velocity (t : ℝ) : ℝ := 6 * t^2 + 1

-- Displacement function (indefinite integral of velocity)
def displacement (t : ℝ) (C : ℝ) : ℝ := 2 * t^3 + t + C

-- Given condition: displacement at t = 3 is 60
axiom displacement_at_3 : displacement 3 C = 60

-- Prove that the equation of motion is s = 2t^3 + t + 3
theorem equation_of_motion :
  ∃ C, displacement t C = 2 * t^3 + t + 3 :=
by
  use 3
  sorry

end MotionLaw

end equation_of_motion_l766_766774


namespace roots_harmonic_iff_det_zero_l766_766412

theorem roots_harmonic_iff_det_zero
  {a b c d e x1 x2 x3 x4 : ℝ} (h_a : a ≠ 0)
  (h_roots : a * x1^4 + 4 * b * x1^3 + 6 * c * x1^2 + 4 * d * x1 + e = 0 ∧
             a * x2^4 + 4 * b * x2^3 + 6 * c * x2^2 + 4 * d * x2 + e = 0 ∧
             a * x3^4 + 4 * b * x3^3 + 6 * c * x3^2 + 4 * d * x3 + e = 0 ∧
             a * x4^4 + 4 * b * x4^3 + 6 * c * x4^2 + 4 * d * x4 + e = 0) :
  ((x1 - x3) * (x2 - x4) + (x2 - x3) * (x1 - x4) = 0) ↔
  (matrix.det ![
    ![a, b, c],
    ![b, c, d],
    ![c, d, e]
  ] = 0) := sorry

end roots_harmonic_iff_det_zero_l766_766412


namespace PB_is_symmedian_of_KPL_l766_766431

variable (w1 w2 : Circle)
variable (A B C D K L P : Point)
variable (CD : Line)
variable (KC LD : Line)

-- Assume the conditions.
axiom circles_intersect : Intersects w1 w2 A B
axiom common_tangent : Tangent CD w1 C ∧ Tangent CD w2 D
axiom B_closer_to_CD : CloserTo B CD A
axiom line_through_A : LineThrough A (IntersectAt w1 K) (IntersectAt w2 L)
axiom K_between_A_L : Between K A L
axiom KC_LD_intersect_at_P : Intersects KC LD P

-- Prove that PB is a symmedian of triangle KPL.
theorem PB_is_symmedian_of_KPL : Symmedian P B K P L :=
  sorry

end PB_is_symmedian_of_KPL_l766_766431


namespace correct_option_C_l766_766908

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem correct_option_C :
  ∀ x1 x2 : ℝ, (x1 ∈ Set.Icc (Real.pi / 2) (2 * Real.pi / 3)) → (x2 ∈ Set.Icc (Real.pi / 2) (2 * Real.pi / 3)) → x1 < x2 → f x1 > f x2 :=
by
  sorry

end correct_option_C_l766_766908


namespace largest_perfect_square_factor_of_3402_l766_766054

theorem largest_perfect_square_factor_of_3402 : 
  ∃ (n : ℕ), (∃ (k : ℕ), 3402 = k * k * n) ∧ 81 = k * k :=
begin
  sorry
end

end largest_perfect_square_factor_of_3402_l766_766054


namespace tetrahedron_inscribed_angles_equal_l766_766114

theorem tetrahedron_inscribed_angles_equal
  (A B C D S R Q P O : Type)
  (r : ℝ)
  (tetrahedron : A → B → C → D → Prop)
  (sphere : S → R → Q → P → Prop)
  (center : S → O → r → Prop)
  (touch_points_equal : ∀ {X Y Z : Type}, center X O r → center Y O r → center Z O r → touches_face X Y Z tetrahedron → touches_face X Y Z sphere)
  :
  {∠ ASB, ∠ BSC, ∠ CSA} = {∠ ARB, ∠ BRD, ∠ DRA} ∧
  {∠ ARB, ∠ BRD, ∠ DRA} = {∠ AQC, ∠ CQD, ∠ DQA} ∧
  {∠ AQC, ∠ CQD, ∠ DQA} = {∠ BPC, ∠ CPD, ∠ DPB} := 
begin
  sorry
end

end tetrahedron_inscribed_angles_equal_l766_766114


namespace triangle_equilateral_l766_766224

noncomputable def u := -1/2 + ℂ.I * real.sqrt 3 / 2 

def rotated (P : ℂ) (A : ℂ) : ℂ := (1 + u) * A - u * P

def sequence (A0 A1 A2 : ℂ) (P0 : ℂ) : ℕ → ℂ
| 0     := P0
| (k+1) := rotated (sequence A0 A1 A2 k) (if k % 3 = 0 then A0 else if k % 3 = 1 then A1 else A2)

def is_equilateral (A1 A2 A3 : ℂ) : Prop :=
  ∃ l : ℝ, (A2 - A1).abs = l ∧ (A3 - A2).abs = l ∧ (A1 - A3).abs = l

theorem triangle_equilateral (A1 A2 A3 : ℂ) (P0 : ℂ) 
  (h : sequence A1 A2 A3 P0 1986 = P0) : is_equilateral A1 A2 A3 :=
sorry

end triangle_equilateral_l766_766224


namespace probability_of_exactly_one_shortening_l766_766856

-- Define the conditions and problem
def shorten_sequence_exactly_one_probability (n : ℕ) (p : ℝ) : ℝ :=
  nat.choose n 1 * p^1 * (1 - p)^(n - 1)

-- Given conditions
def problem_8a := shorten_sequence_exactly_one_probability 2014 0.1 ≈ 1.564e-90

-- Lean statement
theorem probability_of_exactly_one_shortening :
  problem_8a := sorry

end probability_of_exactly_one_shortening_l766_766856


namespace find_mn_l766_766570

theorem find_mn (m n : ℕ) (h : m > 0 ∧ n > 0) (eq1 : m^2 + n^2 + 4 * m - 46 = 0) :
  mn = 5 ∨ mn = 15 := by
  sorry

end find_mn_l766_766570


namespace area_of_triangle_l766_766277

noncomputable def segment_length_AB : ℝ := 10
noncomputable def point_AP : ℝ := 2
noncomputable def point_PB : ℝ := segment_length_AB - point_AP -- PB = AB - AP 
noncomputable def radius_omega1 : ℝ := point_AP / 2 -- radius of ω1
noncomputable def radius_omega2 : ℝ := point_PB / 2 -- radius of ω2
noncomputable def distance_centers : ℝ := 5 -- given directly
noncomputable def length_XY : ℝ := 4 -- given directly
noncomputable def altitude_PZ : ℝ := 8 / 5 -- given directly
noncomputable def area_triangle_XPY : ℝ := (1 / 2) * length_XY * altitude_PZ

theorem area_of_triangle : area_triangle_XPY = 16 / 5 := by
  sorry

end area_of_triangle_l766_766277


namespace product_of_p_r_s_l766_766632

theorem product_of_p_r_s :
  ∃ p r s : ℕ, 3^p + 3^5 = 252 ∧ 2^r + 58 = 122 ∧ 5^3 * 6^s = 117000 ∧ p * r * s = 36 :=
by
  sorry

end product_of_p_r_s_l766_766632


namespace three_buses_interval_l766_766041

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end three_buses_interval_l766_766041


namespace greatest_integer_x_l766_766048

theorem greatest_integer_x :
  ∃ (x : ℤ), (∀ (y : ℤ), (8 : ℝ) / 11 > (x : ℝ) / 15) ∧
    ¬ (8 / 11 > (x + 1 : ℝ) / 15) ∧
    x = 10 :=
by
  sorry

end greatest_integer_x_l766_766048


namespace find_certain_number_l766_766469

noncomputable def certain_number : ℝ :=
  let x := 36 in x

theorem find_certain_number (x : ℝ) :
  x + 8 = 44 ↔ x = 36 :=
by 
  sorry

end find_certain_number_l766_766469


namespace K_travel_time_40_miles_l766_766465

noncomputable def K_time (x : ℝ) : ℝ := 40 / x

theorem K_travel_time_40_miles (x : ℝ) (d : ℝ) (Δt : ℝ)
  (h1 : d = 40)
  (h2 : Δt = 1 / 3)
  (h3 : ∃ (Kmiles_r : ℝ) (Mmiles_r : ℝ), Kmiles_r = x ∧ Mmiles_r = x - 0.5)
  (h4 : ∃ (Ktime : ℝ) (Mtime : ℝ), Ktime = d / x ∧ Mtime = d / (x - 0.5) ∧ Mtime - Ktime = Δt) :
  K_time x = 5 := sorry

end K_travel_time_40_miles_l766_766465


namespace part_a_l766_766833

theorem part_a (k : ℕ) (h : ∀ a b : ℕ, ¬ (a > 0 ∧ b > 0 ∧ ab + (a + 1) * (b + 1) = 2^k)) : nat.prime (k + 1) :=
sorry

end part_a_l766_766833


namespace positive_int_satisfy_count_satisfying_positive_int_l766_766630

open Int

theorem positive_int_satisfy (n : ℕ) :
  (∃ k : ℕ, (50 * k - 1100 = n) ∧ (k = Nat.floor ((50 * k - 1100 : ℤ)^(1/3)))) ↔ n > 0 := 
by
  sorry

theorem count_satisfying_positive_int :
  (finset.univ.filter (λ n, (∃ k : ℕ, (50 * k - 1100 = n) ∧ (k = Nat.floor ((50 * k - 1100 : ℤ)^(1/3))))) .card = 10 := 
by
  sorry

end positive_int_satisfy_count_satisfying_positive_int_l766_766630


namespace simplify_expr_l766_766350

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l766_766350


namespace simplify_expression_l766_766322

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l766_766322


namespace eval_expression_l766_766808

theorem eval_expression : 4 * (8 - 3) - 7 = 13 :=
by
  sorry

end eval_expression_l766_766808


namespace minimum_area_triangle_l766_766245

variables
  (V A B D C : Type)
  [MetricSpace V]
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace D]
  [MetricSpace C]
  (d_VC : ℝ) (d_AB : ℝ)
  (h1 : d_VC = 3)
  (h2 : d_AB = 2)

theorem minimum_area_triangle :
  ∃ (area : ℝ), area = (sqrt 23) / 3 :=
by
  sorry

end minimum_area_triangle_l766_766245


namespace complex_quadrant_l766_766000

theorem complex_quadrant (z : ℂ) (h : z = (↑0 + 1*I) / (1 + 1*I)) : z.re > 0 ∧ z.im > 0 := 
by
  sorry

end complex_quadrant_l766_766000


namespace product_not_zero_l766_766525

theorem product_not_zero (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 5) : (x - 2) * (x - 5) ≠ 0 := 
by 
  sorry

end product_not_zero_l766_766525


namespace bryan_push_ups_l766_766521

theorem bryan_push_ups (sets : ℕ) (push_ups_per_set : ℕ) (fewer_in_last_set : ℕ) 
  (h1 : sets = 3) (h2 : push_ups_per_set = 15) (h3 : fewer_in_last_set = 5) :
  (sets - 1) * push_ups_per_set + (push_ups_per_set - fewer_in_last_set) = 40 := by 
  -- We are setting sorry here to skip the proof.
  sorry

end bryan_push_ups_l766_766521


namespace find_smallest_z_l766_766392

noncomputable def smallest_possible_positive_z (x z : ℝ) : ℝ :=
if h1 : sin x = 1 ∧ sin (x + z) = real.sqrt 3 / 2 then
  let k := (2 * Math.pi) in
  if z = Math.pi / 6 + 2 * k then z else sorry
else sorry

theorem find_smallest_z :
  ∀ (x z : ℝ),
  sin x = 1 →
  sin (x + z) = real.sqrt 3 / 2 →
  smallest_possible_positive_z x z = Math.pi / 6 :=
by
  sorry

end find_smallest_z_l766_766392


namespace arithmetic_sequence_sum_l766_766909

theorem arithmetic_sequence_sum :
  let S1 := (∑ i in range (2093 - 2001 + 1), 2001 + i)
  let S2 := (∑ i in range (313 - 221 + 1), 221 + i)
  let S3 := (∑ i in range (493 - 401 + 1), 401 + i)
  S1 - S2 + S3 = 207141 :=
by
  -- Define the sequences
  let S1 := (∑ i in range (2093 - 2001 + 1), 2001 + i)
  let S2 := (∑ i in range (313 - 221 + 1), 221 + i)
  let S3 := (∑ i in range (493 - 401 + 1), 401 + i)
  -- Use the above definitions to prove the theorem
  have h1 : S1 - S2 + S3 = 207141 := sorry
  exact h1

end arithmetic_sequence_sum_l766_766909


namespace complement_of_A_l766_766223

open Set

-- Define the universal set U
def U : Set ℝ := univ

-- Define the set A based on the given condition
def A : Set ℝ := {x | x + 2 > 4}

-- State the theorem and the goal to prove the complement of A in U
theorem complement_of_A :
  compl U A = {x : ℝ | x ≤ 2} :=
sorry

end complement_of_A_l766_766223


namespace tangent_line_at_neg_ln_2_range_of_a_inequality_range_of_a_zero_point_l766_766603

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

/-- Problem 1 -/
theorem tangent_line_at_neg_ln_2 :
  let x := -Real.log 2
  let y := f x
  ∃ k b : ℝ, (y - b) = k * (x - (-Real.log 2)) ∧ k = (Real.exp x - 1) ∧ b = Real.log 2 + 1/2 :=
sorry

/-- Problem 2 -/
theorem range_of_a_inequality :
  ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x > a * x) ↔ a ∈ Set.Iio (Real.exp 1 - 1) :=
sorry

/-- Problem 3 -/
theorem range_of_a_zero_point :
  ∀ a : ℝ, (∃! x : ℝ, f x - a * x = 0) ↔ a ∈ (Set.Iio (-1) ∪ Set.Ioi (Real.exp 1 - 1)) :=
sorry

end tangent_line_at_neg_ln_2_range_of_a_inequality_range_of_a_zero_point_l766_766603


namespace math_problem_l766_766441

theorem math_problem :
  ((2^2 - 1 + 3^3)⁻¹ - 4) / 2 = -119 / 60 := 
sorry

end math_problem_l766_766441


namespace min_value_expr_l766_766940

open Real

theorem min_value_expr : ∃ x y : ℝ, 
  let expr := (sqrt (2 * (1 + cos (2 * x))) - sqrt (3 - sqrt 2) * sin x + 1) *
               (3 + 2 * sqrt (7 - sqrt 2) * cos y - cos (2 * y))
  in expr = -9 :=
by sorry

end min_value_expr_l766_766940


namespace condition_for_b_parallel_alpha_l766_766981

variables (a b : Line) (α : Plane)

-- Definitions used in Lean 4 statement based on conditions identified
def a_in_alpha : Prop := a ⊆ α
def b_parallel_a : Prop := b ∥ a
def b_parallel_alpha : Prop := b ∥ α
def b_in_alpha : Prop := b ⊆ α

theorem condition_for_b_parallel_alpha (ha_in_alpha : a_in_alpha a α) :
  b_parallel_a b a ∧ ¬ b_in_alpha b α ↔ b_parallel_alpha b α := 
sorry

end condition_for_b_parallel_alpha_l766_766981


namespace sum_of_areas_of_geometric_series_circles_l766_766753

theorem sum_of_areas_of_geometric_series_circles :
  let radii := λ n : ℕ, 1 / 2 ^ n,
      areas := λ n : ℕ, Real.pi * (radii n) ^ 2,
      total_area := ∑' n, areas n
  in total_area = (4 / 3) * Real.pi := by
  sorry

end sum_of_areas_of_geometric_series_circles_l766_766753


namespace angle_BDC_is_55_l766_766087

def right_triangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] : Prop :=
  ∃ (angle_A angle_B angle_C : ℝ), angle_A + angle_B + angle_C = 180 ∧
  angle_A = 20 ∧ angle_C = 90

def bisector (B D : Type) [Inhabited B] [Inhabited D] (angle_ABC : ℝ) : Prop :=
  ∃ (angle_DBC : ℝ), angle_DBC = angle_ABC / 2

theorem angle_BDC_is_55 (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] :
  right_triangle A B C →
  bisector B D 70 →
  ∃ angle_BDC : ℝ, angle_BDC = 55 :=
by sorry

end angle_BDC_is_55_l766_766087


namespace α_plus_β_eq_3π_over_4_l766_766199

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

axiom h1 : 0 < α ∧ α < real.pi
axiom h2 : 0 < β ∧ β < real.pi
axiom h3 : real.cos α = real.sqrt 10 / 10
axiom h4 : real.cos β = real.sqrt 5 / 5

theorem α_plus_β_eq_3π_over_4 : α + β = 3 * real.pi / 4 :=
by
  sorry

end α_plus_β_eq_3π_over_4_l766_766199


namespace equilateral_triangle_l766_766279

noncomputable def find_k : ℝ := 1 / Real.sqrt 3

theorem equilateral_triangle (A B C A' B' C' : ℝ × ℝ) 
  (k : ℝ) 
  (h1 : k = find_k) 
  (h2 : k > 0) 
  (h3 : dist (A, A') = k * dist (B, C)) 
  (h4 : dist (B, B') = k * dist (A, C)) 
  (h5 : dist (C, C') = k * dist (A, B)) 
  (h6 : A' = (A.1 + k * (C.2 - B.2), A.2 + k * (B.1 - C.1)))
  (h7 : B' = (B.1 + k * (A.2 - C.2), B.2 + k * (C.1 - A.1)))
  (h8 : C' = (C.1 + k * (B.2 - A.2), C.2 + k * (A.1 - B.1))) :
  equilateral A' B' C' := 
sorry

end equilateral_triangle_l766_766279


namespace square_reciprocal_sum_integer_l766_766461

theorem square_reciprocal_sum_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) : ∃ m : ℤ, a^2 + 1/a^2 = m := by
  sorry

end square_reciprocal_sum_integer_l766_766461


namespace sqrt_simplification_l766_766371

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l766_766371


namespace correct_statement_D_l766_766819

def point (x y : ℝ) : Type := (ℝ × ℝ)

def distance (a b : ℝ) : ℝ := abs (a - b)

theorem correct_statement_D :
  ∀ (p1 p2 : point ℝ ℝ), p1 = (3, -2) ∧ p2 = (3, 1) → distance p1.2 p2.2 = 3 :=
begin
  intros p1 p2 h,
  cases h,
  simp [point, distance],
  exact abs_eq_self.mpr (by norm_num),
end

end correct_statement_D_l766_766819


namespace number_of_zeros_of_f_l766_766075

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry
def e (x : ℝ) : ℝ := Real.exp x
def sin (x : ℝ) : ℝ := Real.sin x
def cos (x : ℝ) : ℝ := Real.cos x

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom equation : ∀ x : ℝ, f x + g x + f x * g x = (e x + sin x) * cos x

theorem number_of_zeros_of_f : ∃ n : ℕ, n = 5 ∧ ∀ (count : ℕ), (count = 5) ↔ (set.subset {x ∈ set.Icc 0 (4 * Real.pi) | f x = 0} (set.range (λ k, k * Real.pi / 2)) ∧ {x ∈ set.Icc 0 (4 * Real.pi) | f x = 0}.card = count) :=
sorry

end number_of_zeros_of_f_l766_766075


namespace tan_x_value_l766_766197

theorem tan_x_value 
  (x : ℝ) 
  (h1 : cos (π + x) = 3 / 5)
  (h2 : π < x ∧ x < 2 * π) : 
  tan x = 4 / 3 := 
by
  sorry

end tan_x_value_l766_766197


namespace f_is_odd_f_max_min_on_interval_l766_766179

variables {f : ℝ → ℝ}

-- Conditions from the problem
axiom f_add : ∀ (a b : ℝ), f(a + b) = f(a) + f(b)
axiom f_positive : ∀ (x : ℝ), (x > 0) → f(x) < 0
axiom f_one : f(1) = -2

-- Part 1: Prove that f is an odd function
theorem f_is_odd : ∀ x : ℝ, f(-x) = -f(x) :=
by
  sorry

-- Part 2: Prove the maximum and minimum values of f on the interval [-3, 3]
theorem f_max_min_on_interval : 
  (∀ x ∈ [-3, 3], f(x) ≤ 6) ∧ (∀ x ∈ [-3, 3], f(x) ≥ -6) :=
by
  sorry

end f_is_odd_f_max_min_on_interval_l766_766179


namespace twelfth_term_arithmetic_sequence_l766_766440

-- Given conditions
def first_term : ℚ := 1 / 4
def common_difference : ℚ := 1 / 2

-- Statement to prove
theorem twelfth_term_arithmetic_sequence :
  (first_term + 11 * common_difference) = 23 / 4 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l766_766440


namespace sum_valid_k_117_l766_766560

noncomputable def sum_valid_k : ℕ :=
  (Finset.filter (λ k : ℕ, (∃ a b : ℕ, x^{100} - a * x^k + b = (x^2 - 2 * x + 1) * P (x)) 
  ∧ (1 ≤ k) ∧ (k ≤ 99)) (Finset.range 100)).sum id

theorem sum_valid_k_117 : sum_valid_k = 117 := 
  sorry

end sum_valid_k_117_l766_766560


namespace find_f2_l766_766214

noncomputable def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 :=
by
  sorry

end find_f2_l766_766214


namespace accurate_time_when_car_clock_shows_11_l766_766922

theorem accurate_time_when_car_clock_shows_11
  (phone_start_time car_start_time : ℕ)
  (phone_stop_time car_stop_time : ℕ)
  (car_clock_final_time : ℕ)
  (phone_start_time = 1500)
  (car_start_time = 1500)
  (phone_stop_time = 1545)
  (car_stop_time = 1600)
  (car_clock_final_time = 2300) :
  phone_time_when_car_clock_is_2300 = 2315 :=
by
  sorry

end accurate_time_when_car_clock_shows_11_l766_766922


namespace general_formula_a_n_l766_766188

noncomputable def S (a : ℕ → ℕ) : ℕ → ℕ
| 0     => 0
| (n+1) => a (n+1) + S a n

theorem general_formula_a_n (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 1 ∧ 
  (∀ n, a (n+1) = 2 * S n) →
  (∀ n, a n = if n = 1 then 1 else 2 * 3^(n-2)) :=
begin
  sorry
end

end general_formula_a_n_l766_766188


namespace correct_statements_l766_766212

-- Statements definitions:

-- Statement 1: Arc length of the sector
def sector_arc_length (α : ℝ) (r : ℝ) : ℝ :=
  α * r

-- Statement 2: Systematic sampling description
def systematic_sampling (rows : ℕ) (seats_per_row : ℕ) (selected_seat : ℕ) : Bool :=
  selected_seat ≤ seats_per_row

-- Statement 3: Complementary Events
def complementary_events (E : Prop) (F : Prop) : Prop :=
  E ↔ ¬F

-- Statement 4: Monotonicity and Inequality within an interval
def tan_gt_x_gt_sin (x : ℝ) : Prop :=
  0 < x ∧ x < Real.pi / 2 ∧ tan x > x ∧ x > sin x

-- Statement 5: Variance transformation rule
def variance_transformation (σ² : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  a^2 * σ²

-- Proof problem statement
theorem correct_statements :
  (sector_arc_length (2/3 * Real.pi) 2 = 4 * Real.pi / 3) ∧
  (systematic_sampling 25 20 15) ∧
  (complementary_events (∃ x, x ≥ 0 ∧ x < 1) (∀ x, x < 0 ∨ x ≥ 1)) ∧
  (∀ x, tan_gt_x_gt_sin x) ∧
  (variance_transformation 8 2 1 ≠ 16) :=
sorry

end correct_statements_l766_766212


namespace train_bridge_crossing_time_l766_766876

noncomputable def train_length : ℝ := 700
noncomputable def bridge_length : ℝ := 130
noncomputable def train_speed_kmh : ℝ := 21
noncomputable def total_distance : ℝ := train_length + bridge_length
noncomputable def speed_conversion_factor : ℝ := 1000 / 3600
noncomputable def train_speed_ms : ℝ := train_speed_kmh * speed_conversion_factor
noncomputable def travel_time : ℝ := total_distance / train_speed_ms

theorem train_bridge_crossing_time :
  travel_time ≈ 142.29 := sorry

end train_bridge_crossing_time_l766_766876


namespace incenter_coordinates_l766_766211

/-- Given the coordinates of the three vertices of triangle ABC as A(x1, y1), B(x2, y2), and C(x3, y3), and the lengths of the three sides are a, b, c. Prove that the coordinates of the incenter I of this triangle are ((a * x1 + b * x2 + c * x3) / (a + b + c), (a * y1 + b * y2 + c * y3) / (a + b + c)). --/
theorem incenter_coordinates {x1 y1 x2 y2 x3 y3 a b c : ℝ} 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  let I_x := (a * x1 + b * x2 + c * x3) / (a + b + c) in
  let I_y := (a * y1 + b * y2 + c * y3) / (a + b + c) in
  ∃ (I : ℝ × ℝ), I = (I_x, I_y) := 
sorry

end incenter_coordinates_l766_766211


namespace sum_of_sin6_l766_766891

theorem sum_of_sin6 :
  let angles := (List.range 10).map (λ k, 10 * k)
  let sin6_sum := angles.sum (λ k, (Real.sin (k * Real.pi / 180)) ^ 6)
  sin6_sum = 3.125 :=
by
  let angles := (List.range 10).map (λ k, 10 * k)
  let sin6_sum := angles.sum (λ k, (Real.sin (k * Real.pi / 180)) ^ 6)
  exact sorry

end sum_of_sin6_l766_766891


namespace pascals_triangle_contains_15_l766_766631

theorem pascals_triangle_contains_15 :
  ∃! n : ℕ, ∃ k : ℕ, nat.choose n k = 15 :=
sorry

end pascals_triangle_contains_15_l766_766631


namespace line_intersects_circle_slope_angle_l766_766571

-- Define the circle and the line
def circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

def line (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the distance between two intersection points
def distance (A B : ℝ × ℝ) := sqrt ((fst B - fst A)^2 + (snd B - snd A)^2)

-- The problem statements
theorem line_intersects_circle
  (m : ℝ) :
  ∃ (x1 y1 x2 y2 : ℝ), circle x1 y1 ∧ circle x2 y2 ∧ line m x1 y1 ∧ line m x2 y2 ∧ (x1, y1) ≠ (x2, y2) :=
sorry

theorem slope_angle 
  (m : ℝ)
  (A B : ℝ × ℝ)
  (hA : circle A.fst A.snd)
  (hB : circle B.fst B.snd)
  (hlA : line m A.fst A.snd)
  (hlB : line m B.fst B.snd)
  (hAB : distance A B = sqrt 17) :
  ∠(m * A.fst - A.snd) = 60 ∨ ∠(m * A.fst - A.snd) = 120 :=
sorry

end line_intersects_circle_slope_angle_l766_766571


namespace evaluate_expression_l766_766170

theorem evaluate_expression (x : ℝ) : x * (x * (x * (x - 3) - 5) + 12) + 2 = x^4 - 3 * x^3 - 5 * x^2 + 12 * x + 2 :=
by
  sorry

end evaluate_expression_l766_766170


namespace simplify_radical_expression_l766_766360

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l766_766360


namespace probability_of_shortening_exactly_one_digit_l766_766838

theorem probability_of_shortening_exactly_one_digit :
  let n := 2014 in
  let p := 0.1 in
  let q := 0.9 in
  (n.choose 1) * p ^ 1 * q ^ (n - 1) = 201.4 * 1.564 * 10^(-90) :=
by sorry

end probability_of_shortening_exactly_one_digit_l766_766838


namespace number_of_subsets_with_one_isolated_element_l766_766278

def is_isolated_element (A : Finset ℤ) (k : ℤ) : Prop :=
  k ∈ A ∧ k - 1 ∉ A ∧ k + 1 ∉ A

def has_exactly_one_isolated_element (A : Finset ℤ) (S : Finset ℤ) : Prop :=
  ∃ k ∈ S, is_isolated_element S k ∧ ∀ j ∈ S, j ≠ k → ¬ is_isolated_element S j

def example_set : Finset ℤ := {1, 2, 3, 4, 5}

theorem number_of_subsets_with_one_isolated_element :
  (example_set.powerset.filter (has_exactly_one_isolated_element example_set)).card = 13 := by
  sorry

end number_of_subsets_with_one_isolated_element_l766_766278


namespace total_supermarkets_FGH_chain_l766_766020

def supermarkets_us : ℕ := 47
def supermarkets_difference : ℕ := 10
def supermarkets_canada : ℕ := supermarkets_us - supermarkets_difference
def total_supermarkets : ℕ := supermarkets_us + supermarkets_canada

theorem total_supermarkets_FGH_chain : total_supermarkets = 84 :=
by 
  sorry

end total_supermarkets_FGH_chain_l766_766020


namespace red_triangles_count_l766_766310

theorem red_triangles_count (n : ℕ) (h1 : n > 1) (points : Fin 2n → Prop) (h2 : ∀ p : Fin 2n, ∀ q : Fin 2n, p ≠ q) (segments : set (Fin 2n × Fin 2n)) (h3 : ∀ p q, (p, q) ∈ segments ∨ (q, p) ∈ segments) (red_segments : set (Fin 2n × Fin 2n)) (h4 : red_segments.card = n^2 + 1) :
  ∃ (red_triangles : set (Fin 2n × Fin 2n × Fin 2n)), red_triangles.card ≥ n ∧ ∀ t ∈ red_triangles, (t.1, t.2) ∈ red_segments ∧ (t.2, t.3) ∈ red_segments ∧ (t.3, t.1) ∈ red_segments 
  := sorry

end red_triangles_count_l766_766310


namespace no_rotation_of_11_gears_l766_766660

theorem no_rotation_of_11_gears :
  ∀ (gears : Fin 11 → ℕ → Prop), 
    (∀ i, gears i 0 ∧ gears (i + 1) 1 → gears i 0 = ¬gears (i + 1) 1) →
    gears 10 0 = gears 0 0 →
    False :=
by
  sorry

end no_rotation_of_11_gears_l766_766660


namespace largestPerfectSquareFactorOf3402_l766_766061

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem largestPerfectSquareFactorOf3402 :
  ∃ k : ℕ, isPerfectSquare k ∧ k ∣ 3402 ∧ ∀ m : ℕ, isPerfectSquare m ∧ m ∣ 3402 → m ≤ k := 
begin
  use 81,
  split,
  { use 9, exact rfl },
  split,
  { norm_num },
  { intros m h,
    cases h with hm hm',
    cases hm with x hx,
    rw hx at hm',
    by_cases h0 : x = 0,
    { subst h0, norm_num at hm', exact hm' },
    by_cases h1 : x = 9,
    { subst h1, norm_num at hm', exact hm' },
    norm_num at hm',
    sorry
  }
end

end largestPerfectSquareFactorOf3402_l766_766061


namespace problem_a_problem_b_l766_766843

open Probability

noncomputable def sequenceShortenedByExactlyOneDigitProbability : ℝ :=
  1.564 * 10^(-90)

theorem problem_a (n : ℕ) (p : ℝ) (k : ℕ) (initialLength : ℕ)
  (h_n : n = 2014) (h_p : p = 0.1) (h_k : k = 1) (h_initialLength : initialLength = 2015) :
  (binomialₓ n k p * (1 - p)^(n - k)) = sequenceShortenedByExactlyOneDigitProbability := 
sorry

noncomputable def expectedLengthNewSequence : ℝ :=
  1813.6

theorem problem_b (n : ℕ) (p : ℝ) (initialLength : ℕ) 
  (h_n : n = 2014) (h_p : p = 0.1) (h_initialLength : initialLength = 2015) :
  (initialLength - n * p) = expectedLengthNewSequence :=
sorry

end problem_a_problem_b_l766_766843


namespace find_a_l766_766968

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logb 2 (x^2 + a)

theorem find_a (a : ℝ) : f 3 a = 1 → a = -7 :=
by
  intro h
  unfold f at h
  sorry

end find_a_l766_766968


namespace percentage_increase_of_return_trip_l766_766489

noncomputable def speed_increase_percentage (initial_speed avg_speed : ℝ) : ℝ :=
  ((2 * avg_speed * initial_speed) / avg_speed - initial_speed) * 100 / initial_speed

theorem percentage_increase_of_return_trip :
  let initial_speed := 30
  let avg_speed := 34.5
  speed_increase_percentage initial_speed avg_speed = 35.294 :=
  sorry

end percentage_increase_of_return_trip_l766_766489


namespace quadratic_decreasing_a_ge_1_l766_766650

theorem quadratic_decreasing_a_ge_1 (a : ℝ) (f : ℝ → ℝ) (h : f = λ x, x^2 - 2*a*x + 3) 
  (decreasing_on : ∀ (x y : ℝ), x ≤ y ∧ y ∈ set.Iic (1 : ℝ) → f y ≤ f x) : 
  a ≥ 1 := 
sorry

end quadratic_decreasing_a_ge_1_l766_766650


namespace simplify_radical_expression_l766_766354

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l766_766354


namespace find_alpha_l766_766633

theorem find_alpha (α : ℝ) (h : Real.sin α * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) = 1) :
  α = 13 * Real.pi / 18 :=
sorry

end find_alpha_l766_766633


namespace speed_ratio_A_to_B_l766_766044

variables {u v : ℝ}

axiom perp_lines_intersect_at_o : true
axiom points_move_along_lines_at_constant_speed : true
axiom point_A_at_O_B_500_yards_away_at_t_0 : true
axiom after_2_minutes_A_and_B_equidistant : 2 * u = 500 - 2 * v
axiom after_10_minutes_A_and_B_equidistant : 10 * u = 10 * v - 500

theorem speed_ratio_A_to_B : u / v = 2 / 3 :=
by 
  sorry

end speed_ratio_A_to_B_l766_766044


namespace cost_of_asian_stamps_80s_l766_766468

variable (stamps_80s : String → Nat)
variable (price_per_stamp : String → Float)

axiom prices : price_per_stamp "China" = 0.10 ∧
                price_per_stamp "India" = 0.10 ∧
                price_per_stamp "Japan" = 0.12 ∧
                price_per_stamp "Thailand" = 0.07

axiom counts_80s : stamps_80s "China" = 10 ∧
                  stamps_80s "India" = 5 ∧
                  stamps_80s "Japan" = 8 ∧
                  stamps_80s "Thailand" = 12

theorem cost_of_asian_stamps_80s : 
  (stamps_80s "China") * (price_per_stamp "China") + 
  (stamps_80s "India") * (price_per_stamp "India") + 
  (stamps_80s "Japan") * (price_per_stamp "Japan") + 
  (stamps_80s "Thailand") * (price_per_stamp "Thailand") = 3.30 :=
by
  have h_prices := prices
  have h_counts_80s := counts_80s
  sorry

end cost_of_asian_stamps_80s_l766_766468


namespace evaluate_expression_l766_766811

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 :=
by sorry

end evaluate_expression_l766_766811


namespace polynomial_roots_l766_766931

-- Problem statement: prove that the roots of the given polynomial are {-1, 3, 3}
theorem polynomial_roots : 
  (λ x => x^3 - 5 * x^2 + 3 * x + 9) = (λ x => (x + 1) * (x - 3) ^ 2) :=
by
  sorry

end polynomial_roots_l766_766931


namespace complex_modulus_l766_766744

noncomputable def z : ℂ := (1 + 3 * Complex.I) / (1 + Complex.I)

theorem complex_modulus 
  (h : (1 + Complex.I) * z = 1 + 3 * Complex.I) : 
  Complex.abs (z^2) = 5 := 
by
  sorry

end complex_modulus_l766_766744


namespace log_4_of_128_sqrt_2_l766_766925

theorem log_4_of_128_sqrt_2 : log 4 (128 * sqrt 2) = 15 / 4 := 
by
  sorry

end log_4_of_128_sqrt_2_l766_766925


namespace magnitude_difference_l766_766995

variables (e1 e2 : ℝ → ℝ → ℝ)
variables (u1 u2: ℝ)
def unit_vector (e : ℝ → ℝ → ℝ) : Prop :=
  |e u1 u2| = 1

def angle_between (e1 e2 : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ :=
  (e1 e2) = (real.cos (60: ℝ) = 0.5)

theorem magnitude_difference {e1 e2 : ℝ → ℝ → ℝ} : 
  (unit_vector e1) → (unit_vector e2) → (angle_between e1 e2)  → 
  (|e1 - 2 * e2| = real.sqrt 3) :=
by
  sorry

end magnitude_difference_l766_766995


namespace polygon_perimeter_l766_766102

theorem polygon_perimeter (n : ℕ) (side_length : ℝ) (ext_angle : ℝ) 
  (regular : side_length = 8) (angle : ext_angle = 45) 
  (sides : n = 360 / ext_angle) : 
  n * side_length = 64 :=
by 
  rw [regular, angle, sides]
  norm_num
  sorry  

end polygon_perimeter_l766_766102


namespace cat_food_cans_count_l766_766129

-- Conditions from the problem.
variables (c : ℕ)
def packages_cat_food := 6
def packages_dog_food := 2
def cans_per_dog_food_package := 3
def extra_cans_of_cat_food := 48

-- Translate to Lean: Proof that the number of cans in each package of cat food is 9.
theorem cat_food_cans_count : 
  6 * c = 2 * 3 + 48 → 
  c = 9 :=
by 
  intro h
  have h1 : 6 * c = 6 + 48 := h
  have h2 : 6 * c = 54 := by rw [h1]
  have h3 : c = 54 / 6 := eq.symm (nat.mul_equiv_div h2)
  have h4 : c = 9 := by norm_num [h3]
  exact h4

end cat_food_cans_count_l766_766129


namespace range_of_a_l766_766598

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * |x|

theorem range_of_a (a : ℝ) : 
  (f (-a) + f a ≤ 2 * f 2) → (a ∈ set.Icc (-2 : ℝ) 2) :=
by
  sorry

end range_of_a_l766_766598


namespace pentagon_area_l766_766487

-- Define the conditions
def side_lengths : list ℕ := [14, 21, 22, 28, 37]

-- The Pythagorean theorem relation
def pythagorean_theorem (u v e : ℕ) : Prop := u^2 + v^2 = e^2

-- Define u and v in terms of b, d, c, and a
def uv_relation (a b c d u v : ℕ) : Prop := u = b - d ∧ v = c - a

-- Given the lengths are from the side_lengths list
def side_lengths_condition (a b c d e : ℕ) : Prop :=
  a ∈ side_lengths ∧ b ∈ side_lengths ∧ c ∈ side_lengths ∧ d ∈ side_lengths ∧ e ∈ side_lengths ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

-- Prove the area of the pentagon
theorem pentagon_area (a b c d e u v : ℕ) 
  (h_side_lengths : side_lengths_condition a b c d e)
  (h_py : pythagorean_theorem u v e)
  (h_uv : uv_relation a b c d u v) :
  (b * c - (1 / 2) * u * v) = 826 := 
begin
  sorry,
end

end pentagon_area_l766_766487


namespace price_increase_after_percentage_cut_l766_766411

theorem price_increase_after_percentage_cut (P : ℝ) (cut_percentage : ℝ) (increase_percentage : ℝ) :
  cut_percentage = 20 → 
  increase_percentage = 25 → 
  (P * (1 - cut_percentage / 100) * (1 + increase_percentage / 100) = P) :=
begin
  intros h_cut h_increase,
  have h_cut_formula : 1 - 20 / 100 = 0.8,
  { norm_num },
  have h_increase_formula : 1 + 25 / 100 = 1.25,
  { norm_num },
  rw [h_cut, h_increase, h_cut_formula, h_increase_formula],
  norm_num,
end

end price_increase_after_percentage_cut_l766_766411


namespace line_slope_angle_l766_766762

theorem line_slope_angle : 
  (∃ α : ℝ, (∃ k : ℝ, ( l : ℝ → ℝ → Prop := λ x y, sqrt 3 * x + y + 3 = 0) ∧ k = -sqrt 3 ∧ tan α = k) 
             ∧ α * (180 / Real.pi) = 120) :=
begin
  sorry
end

end line_slope_angle_l766_766762


namespace find_x_l766_766229

theorem find_x (x : ℝ) (h : 2 * x - 3 * x + 5 * x = 80) : x = 20 :=
by 
  -- placeholder for proof
  sorry 

end find_x_l766_766229


namespace extra_men_needed_approx_is_60_l766_766078

noncomputable def extra_men_needed : ℝ :=
  let total_distance := 15.0   -- km
  let total_days := 300.0      -- days
  let initial_workforce := 40.0 -- men
  let completed_distance := 2.5 -- km
  let elapsed_days := 100.0    -- days

  let remaining_distance := total_distance - completed_distance -- km
  let remaining_days := total_days - elapsed_days               -- days

  let current_rate := completed_distance / elapsed_days -- km/day
  let required_rate := remaining_distance / remaining_days -- km/day

  let required_factor := required_rate / current_rate
  let new_workforce := initial_workforce * required_factor
  let extra_men := new_workforce - initial_workforce

  extra_men

theorem extra_men_needed_approx_is_60 :
  abs (extra_men_needed - 60) < 1 :=
sorry

end extra_men_needed_approx_is_60_l766_766078


namespace smallest_n_for_disjoint_monochromatic_triangles_l766_766945

theorem smallest_n_for_disjoint_monochromatic_triangles :
  ∃ (n : ℕ), n = 9 ∧ (∀ (K : simple_graph (fin n)),
    (∀ (c : K.edge -> bool),
      ∃ T1 T2 T3 : K.triangle,
        T1.monochromatic c ∧ T2.monochromatic c ∧ T3.monochromatic c ∧
        ¬disjoint T1.edges T2.edges ∧ ¬disjoint T2.edges T3.edges ∧ ¬disjoint T1.edges T3.edges)) :=
begin
  use 9,
  sorry,
end

end smallest_n_for_disjoint_monochromatic_triangles_l766_766945


namespace z_real_z_pure_imaginary_z_third_quadrant_l766_766594

def z (m : ℝ) : ℂ := ⟨m^2 - 3 * m, m^2 - m - 6⟩

theorem z_real (m : ℝ) : (im (z m) = 0) ↔ (m = 3 ∨ m = -2) := by
  sorry

theorem z_pure_imaginary (m : ℝ) : (re (z m) = 0 ∧ im (z m) ≠ 0) ↔ (m = 0) := by
  sorry

theorem z_third_quadrant (m : ℝ) : (re (z m) < 0 ∧ im (z m) < 0) ↔ (0 < m ∧ m < 3) := by
  sorry

end z_real_z_pure_imaginary_z_third_quadrant_l766_766594


namespace john_heavier_than_roy_l766_766273

theorem john_heavier_than_roy (john_weight : ℕ) (roy_weight : ℕ) (h1 : john_weight = 81) (h2 : roy_weight = 4) : john_weight - roy_weight = 77 := 
by
  rw [h1, h2]
  norm_num

end john_heavier_than_roy_l766_766273


namespace find_all_polynomials_l766_766929

noncomputable def polynomial (R : Type*) [CommutativeRing R] := R[X]

variables {R : Type*} [CommRing R]

def condition (f : polynomial ℝ) (x y z : ℝ) (h : x + y + z = 0) : 
  f.eval (x * y) + f.eval (y * z) + f.eval (z * x) = f.eval (x * y + y * z + z * x) :=
sorry

theorem find_all_polynomials (f : polynomial ℝ) :
  (∀ x y z : ℝ, x + y + z = 0 → condition f x y z) → 
  ∃ c : ℝ, f = polynomial.C c * polynomial.X :=
sorry

end find_all_polynomials_l766_766929


namespace added_area_equation_solve_for_x_l766_766873

variable (x y : ℝ)

-- Conditions
def original_length : ℝ := 8
def original_width : ℝ := 6
def added_area : ℝ := y

-- Problem Statement
theorem added_area_equation (h : added_area = (original_length + x) * (original_width + x) - (original_length * original_width)) :
  y = x^2 + 14 * x := by
  sorry

theorem solve_for_x (h : x^2 + 14 * x = 32) : x = 2 := by
  sorry

end added_area_equation_solve_for_x_l766_766873


namespace min_value_expr_l766_766939

open Real

theorem min_value_expr : ∃ x y : ℝ, 
  let expr := (sqrt (2 * (1 + cos (2 * x))) - sqrt (3 - sqrt 2) * sin x + 1) *
               (3 + 2 * sqrt (7 - sqrt 2) * cos y - cos (2 * y))
  in expr = -9 :=
by sorry

end min_value_expr_l766_766939


namespace bike_ride_hours_l766_766885

theorem bike_ride_hours (x : ℚ) (total_time : ℚ := 8) (rest_time : ℚ := 0.5) (total_distance : ℚ := 132) (speed_energetic : ℚ := 25) (speed_tired : ℚ := 15) :
  (total_time - rest_time - x) = (7.5 - x) → 25 * x + 15 * (7.5 - x) = 132 → x = 39 / 20 :=
by {
  intro h1 h2,
  sorry
}

end bike_ride_hours_l766_766885


namespace intersecting_lines_l766_766436

theorem intersecting_lines (p q r s t : ℝ) : (∃ u v : ℝ, p * u^2 + q * v^2 + r * u + s * v + t = 0) →
  ( ∃ p q : ℝ, p * q < 0 ∧ 4 * t = r^2 / p + s^2 / q ) :=
sorry

end intersecting_lines_l766_766436


namespace calc_factorial_sum_l766_766127

theorem calc_factorial_sum : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 2 * Nat.factorial 6 = 40320 := 
sorry

end calc_factorial_sum_l766_766127


namespace cyclic_quadrilateral_equality_l766_766705

theorem cyclic_quadrilateral_equality
  (A B C D L M E F P Q : Point)
  (h_cyclic : CyclicQuadrilateral A B C D)
  (h_midpoint_L : Midpoint L A B)
  (h_midpoint_M : Midpoint M C D)
  (h_inter_E : intersection_point E A C B D)
  (h_ray_meet_F : rays_meet_at F A B D C)
  (h_inter_P : intersection_point P L M D E)
  (h_foot_Q : perpendicular_foot Q P E M)
  (h_orthocenter : orthocenter E F L M) :
  EP^2 / EQ = (1 / 2) * (BD^2 / DF - BC^2 / CF) :=
by
  sorry

end cyclic_quadrilateral_equality_l766_766705


namespace cricket_runs_product_l766_766652

theorem cricket_runs_product :
  let runs_first_10 := [11, 6, 7, 5, 12, 8, 3, 10, 9, 4]
  let total_runs_first_10 := runs_first_10.sum
  let total_runs := total_runs_first_10 + 2 + 7
  2 < 15 ∧ 7 < 15 ∧ (total_runs_first_10 + 2) % 11 = 0 ∧ (total_runs_first_10 + 2 + 7) % 12 = 0 →
  (2 * 7) = 14 :=
by
  intros h
  sorry

end cricket_runs_product_l766_766652


namespace cannot_reach_full_plus_l766_766213

def flip_sign (s : char) : char :=
  if s = '+' then '-' else '+'

def flip_row (table : list (list char)) (r : ℕ) : list (list char) :=
  table.take r ++ [table[r].map flip_sign] ++ table.drop (r + 1)

def flip_col (table : list (list char)) (c : ℕ) : list (list char) :=
  table.map (λ row, row.take c ++ [flip_sign row[c]] ++ row.drop (c + 1))

def initial_table : list (list char) :=
  [['+', '+', '-', '+'],
   ['-', '-', '+', '+'],
   ['+', '+', '+', '+'],
   ['+', '-', '+', '-']]

def is_full_of_plus (table : list (list char)) : Prop :=
  table.all (λ row, row.all (λ ch, ch = '+'))

theorem cannot_reach_full_plus :
  ¬ ∃ (moves : list (ℕ ⊕ ℕ)), 
    is_full_of_plus (moves.foldl (λ t move, 
      match move with
      | Sum.inl r => flip_row t r
      | Sum.inr c => flip_col t c
      end) initial_table) :=
by
  sorry

end cannot_reach_full_plus_l766_766213


namespace investment_rate_l766_766878

theorem investment_rate (total : ℝ) (invested_at_3_percent : ℝ) (rate_3_percent : ℝ) 
                        (invested_at_5_percent : ℝ) (rate_5_percent : ℝ) 
                        (desired_income : ℝ) (remaining : ℝ) (additional_income : ℝ) (r : ℝ) : 
  total = 12000 ∧ 
  invested_at_3_percent = 5000 ∧ 
  rate_3_percent = 0.03 ∧ 
  invested_at_5_percent = 4000 ∧ 
  rate_5_percent = 0.05 ∧ 
  desired_income = 600 ∧ 
  remaining = total - invested_at_3_percent - invested_at_5_percent ∧ 
  additional_income = desired_income - (invested_at_3_percent * rate_3_percent + invested_at_5_percent * rate_5_percent) ∧ 
  r = (additional_income / remaining) * 100 → 
  r = 8.33 := 
by
  sorry

end investment_rate_l766_766878


namespace sin_value_of_angle_l766_766993

noncomputable def r (x y : ℝ) := Real.sqrt (x^2 + y^2)

theorem sin_value_of_angle (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  Real.sin (Real.arctan (y / x)) = 2 * Real.sqrt 5 / 5 :=
by
  rw [hx, hy]
  have hr : r (-1) 2 = Real.sqrt 5 := 
  by
    sorry
  rw [←hr]
  apply Real.sin_arctan
  exact div_nonneg (by norm_num1) (by norm_num1)
  sorry

end sin_value_of_angle_l766_766993


namespace math_problem_l766_766081

-- Definition of percentages and fractions for clarity
def percentage (p : ℝ) (x : ℝ) := (p / 100) * x
def fraction (p : ℝ) (x : ℝ) := p * x

-- Conditions
def x := percentage 80 40
def y := fraction (2/5) 25

-- Theorem statement
theorem math_problem : x - y = 22 := 
by 
  -- skipped steps 
  sorry

end math_problem_l766_766081


namespace sqrt_mixed_number_eq_l766_766545

noncomputable def mixed_number : ℝ := 8 + 1 / 9

theorem sqrt_mixed_number_eq : Real.sqrt (8 + 1 / 9) = Real.sqrt 73 / 3 := by
  sorry

end sqrt_mixed_number_eq_l766_766545


namespace sum_of_roots_cubic_l766_766797

theorem sum_of_roots_cubic :
  ∑ x in (Polynomial.roots (Polynomial.of_fn [6, -3, -18, 9])), x = 0.5 :=
by
  sorry

end sum_of_roots_cubic_l766_766797


namespace tetrahedron_projection_area_l766_766791

noncomputable def maximum_projection_area (hypotenuse : ℝ) (dihedral_angle : ℝ) : ℝ :=
  if hypotenuse = 2 ∧ dihedral_angle = real.pi / 3 then 1 else 0

theorem tetrahedron_projection_area :
  maximum_projection_area 2 (real.pi / 3) = 1 :=
sorry

end tetrahedron_projection_area_l766_766791


namespace upper_limit_of_prime_range_l766_766778

theorem upper_limit_of_prime_range : 
  ∃ x : ℝ, (26 / 3 < 11) ∧ (11 < x) ∧ (x < 17) :=
by
  sorry

end upper_limit_of_prime_range_l766_766778


namespace polar_to_rectangular_l766_766912

theorem polar_to_rectangular  (r θ : ℝ) (h_r : r = 10) (h_θ : θ = 2 * real.pi / 3) :
  (r * real.cos θ, r * real.sin θ) = (-5, 5 * real.sqrt 3) :=
by
  rw [h_r, h_θ]
  have h_cos : real.cos (2 * real.pi / 3) = -1 / 2 := sorry
  have h_sin : real.sin (2 * real.pi / 3) = real.sqrt 3 / 2 := sorry
  simp only [h_cos, h_sin, mul_assoc, mul_neg, mul_one, mul_div_cancel' (5) (ne_of_gt (2 : ℝ).zero_lt_two)]

end polar_to_rectangular_l766_766912


namespace max_value_of_f_l766_766148

def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x

theorem max_value_of_f : ∃ m, (∀ x, f(x) ≤ m) ∧ m = 5 :=
by
  sorry

end max_value_of_f_l766_766148


namespace largest_whole_x_l766_766934

theorem largest_whole_x (x : ℕ) (h : 11 * x < 150) : x ≤ 13 :=
sorry

end largest_whole_x_l766_766934


namespace average_minutes_proof_l766_766098

noncomputable def average_minutes_heard (total_minutes : ℕ) (total_attendees : ℕ) (full_listened_fraction : ℚ) (none_listened_fraction : ℚ) (half_remainder_fraction : ℚ) : ℚ := 
  let full_listeners := full_listened_fraction * total_attendees
  let none_listeners := none_listened_fraction * total_attendees
  let remaining_listeners := total_attendees - full_listeners - none_listeners
  let half_listeners := half_remainder_fraction * remaining_listeners
  let quarter_listeners := remaining_listeners - half_listeners
  let total_heard := (full_listeners * total_minutes) + (none_listeners * 0) + (half_listeners * (total_minutes / 2)) + (quarter_listeners * (total_minutes / 4))
  total_heard / total_attendees

theorem average_minutes_proof : 
  average_minutes_heard 120 100 (30/100) (15/100) (40/100) = 59.1 := 
by
  sorry

end average_minutes_proof_l766_766098


namespace distance_AC_100_l766_766043

theorem distance_AC_100 (d_AB : ℝ) (t1 : ℝ) (t2 : ℝ) (AC : ℝ) (CB : ℝ) :
  d_AB = 150 ∧ t1 = 3 ∧ t2 = 12 ∧ d_AB = AC + CB ∧ AC / 3 = CB / 12 → AC = 100 := 
by
  sorry

end distance_AC_100_l766_766043


namespace three_buses_interval_l766_766039

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end three_buses_interval_l766_766039


namespace num_of_neg_x_sqrt_pos_integer_l766_766957

theorem num_of_neg_x_sqrt_pos_integer : 
  ( ∃ n : ℕ, 
      (1 ≤ n ∧ n ≤ 9) 
      ∧ (∀ x, x = ↑n^2 - 100 → x < 0) ) -> 
  (nat.card (set_of (λ x, ∃ n : ℕ, 1 ≤ n ∧ n ≤ 9 ∧ x = n^2 - 100)) = 9) :=
begin
  sorry
end

end num_of_neg_x_sqrt_pos_integer_l766_766957


namespace question_1_question_2_l766_766612

theorem question_1 (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n : ℕ, 1 ≤ n → S n = 2 * a n - n) →
  (∀ n : ℕ, 1 ≤ n → (a (n + 1) + 1 = 2 * (a n + 1))) := by
  sorry

theorem question_2 (a : ℕ → ℤ) (n : ℕ) :
  (∀ k : ℕ, 1 ≤ k → a k = 2^k - 1) →
  a 1 + a 3 + a 5 + ... + a (2*n+1) = (2^(2*n+3) - 3*n - 5) / 3 := by
  sorry

end question_1_question_2_l766_766612


namespace cities_with_fewer_than_500000_residents_l766_766738

theorem cities_with_fewer_than_500000_residents (P Q R : ℕ) 
  (h1 : P + Q + R = 100) 
  (h2 : P = 40) 
  (h3 : Q = 35) 
  (h4 : R = 25) : P + Q = 75 :=
by 
  sorry

end cities_with_fewer_than_500000_residents_l766_766738


namespace C5_properties_l766_766927


def C5 : SimpleGraph (Fin 5) :=
{ adj := λ i j, 
    i = j + 1 ∨ i + 1 = j ∨ (i = 0 ∧ j = 4) ∨ (i = 4 ∧ j = 0),
  symm := by
    finish,
  loopless := by
    finish
}

noncomputable def C5_chromatic_number : ℕ :=
  chromatic_number C5

def C5_no_3_cliques : Prop :=
  ∀ (v1 v2 v3 : Fin 5), (C5.adj v1 v2 ∧ C5.adj v2 v3 ∧ C5.adj v3 v1) → false

theorem C5_properties :
  C5_chromatic_number = 3 ∧ C5_no_3_cliques :=
begin
  sorry, -- Proof goes here
end

end C5_properties_l766_766927


namespace simplify_expression_l766_766071

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (b * c + a * c + a * b) :=
by
  sorry

end simplify_expression_l766_766071


namespace added_number_is_five_l766_766444

variable (n x : ℤ)

theorem added_number_is_five (h1 : n % 25 = 4) (h2 : (n + x) % 5 = 4) : x = 5 :=
by
  sorry

end added_number_is_five_l766_766444


namespace matrix_multiplication_correct_l766_766157

def A : Matrix (Fin 4) (Fin 3) ℤ :=
  ![
    ![0, -1, 2],
    ![2, 1, 1],
    ![3, 0, 1],
    ![3, 7, 1]
  ]

def B : Matrix (Fin 3) (Fin 2) ℤ :=
  ![
    ![3, 1],
    ![2, 1],
    ![1, 0]
  ]

def C : Matrix (Fin 4) (Fin 2) ℤ :=
  ![
    ![0, -1],
    ![9, 3],
    ![10, 3],
    ![24, 10]
  ]

theorem matrix_multiplication_correct : A ⬝ B = C :=
  by
  sorry

end matrix_multiplication_correct_l766_766157


namespace sum_of_cubes_mod_4_l766_766292

theorem sum_of_cubes_mod_4 :
  let b := 2
  let n := 2010
  ( (n * (n + 1) / 2) ^ 2 ) % (b ^ 2) = 1 :=
by
  let b := 2
  let n := 2010
  sorry

end sum_of_cubes_mod_4_l766_766292


namespace most_likely_dissatisfied_passengers_expected_dissatisfied_passengers_variance_dissatisfied_passengers_l766_766829

noncomputable def prob_most_likely_dissatisfied (n : ℕ) : ℕ :=
  1

noncomputable def expected_dissatisfied (n : ℕ) : ℝ :=
  real.sqrt (n / real.pi)

noncomputable def variance_dissatisfied (n : ℕ) : ℝ :=
  ((real.pi - 2) / (2 * real.pi)) * n

theorem most_likely_dissatisfied_passengers (n : ℕ) :
  prob_most_likely_dissatisfied n = 1 := 
sorry

theorem expected_dissatisfied_passengers (n : ℕ) :
  expected_dissatisfied n = real.sqrt (n / real.pi) :=
sorry

theorem variance_dissatisfied_passengers (n : ℕ) :
  variance_dissatisfied n = ((real.pi - 2) / (2 * real.pi)) * n :=
sorry

end most_likely_dissatisfied_passengers_expected_dissatisfied_passengers_variance_dissatisfied_passengers_l766_766829


namespace no_common_points_implies_parallel_l766_766484

variable (a : Type) (P : Type) [LinearOrder P] [AddGroupWithOne P]
variable (has_no_common_point : a → P → Prop)
variable (is_parallel : a → P → Prop)

theorem no_common_points_implies_parallel (a_line : a) (a_plane : P) :
  has_no_common_point a_line a_plane ↔ is_parallel a_line a_plane :=
sorry

end no_common_points_implies_parallel_l766_766484


namespace shortest_distance_to_line_l766_766574

-- Definitions of given elements
variables (R : Type*) [linear_ordered_field R]
variables (l : set (euclidean_space R 2)) (A M : euclidean_space R 2)
variables (hA : A ∉ l) (hM : M ∈ l) (h_perpendicular : is_perpendicular A M l)

-- Main theorem statement
theorem shortest_distance_to_line 
  (N : euclidean_space R 2) (hN : N ∈ l) (hne : N ≠ M) : 
  dist A M < dist A N :=
sorry

end shortest_distance_to_line_l766_766574


namespace count_correct_propositions_l766_766086

variables {α β γ : Plane} {l m n : Line}

-- Original Propositions
def prop1 : Prop := (α ⊥ γ) ∧ (β ⊥ γ) → (α ∥ β)
def prop2 : Prop := (m ∥ β) ∧ (n ∥ β) → (α ∥ β)
def prop3 : Prop := (l ⊂ α) ∧ (α ∥ β) → (l ∥ β)
def prop4 : Prop := (α ∩ β = γ) ∧ (β ∩ γ = m) ∧ (γ ∩ α = n) ∧ (l ∥ m) → (m ∥ n)

-- Main statement to prove
theorem count_correct_propositions :
  (list.filter (λ p, p) [prop1, prop2, prop3, prop4]).length = 2 := sorry

end count_correct_propositions_l766_766086


namespace exists_prime_between_n_and_factorial_l766_766696

theorem exists_prime_between_n_and_factorial (n : ℕ) (h : n > 2) : ∃ p : ℕ, prime p ∧ n < p ∧ p < n! :=
by
  sorry

end exists_prime_between_n_and_factorial_l766_766696


namespace simplify_expression_l766_766327

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l766_766327


namespace point_not_on_graph_l766_766820

noncomputable def is_on_graph (x y : ℝ) : Prop := 
  y = (x - 1) / (x + 2)

theorem point_not_on_graph : ¬ (is_on_graph (-2) 1) := 
by
  unfold is_on_graph
  simp
  sorry

end point_not_on_graph_l766_766820


namespace simon_project_score_l766_766240

-- Define the initial conditions
def num_students_before : Nat := 20
def num_students_total : Nat := 21
def avg_before : ℕ := 86
def avg_after : ℕ := 88

-- Calculate total score before Simon's addition
def total_score_before : ℕ := num_students_before * avg_before

-- Calculate total score after Simon's addition
def total_score_after : ℕ := num_students_total * avg_after

-- Definition to represent Simon's score
def simon_score : ℕ := total_score_after - total_score_before

-- Theorem that we need to prove
theorem simon_project_score : simon_score = 128 :=
by
  sorry

end simon_project_score_l766_766240


namespace minimum_ratio_of_areas_l766_766208

noncomputable def S_triangle (vertices: list (ℝ × ℝ)) : ℝ :=
  let area := ((vertices[0].1 * (vertices[1].2 - vertices[2].2) +
                vertices[1].1 * (vertices[2].2 - vertices[0].2) +
                vertices[2].1 * (vertices[0].2 - vertices[1].2)) / 2).abs
  area

theorem minimum_ratio_of_areas :
  ∀ (A B C D E F : ℝ × ℝ), 
  (DEF_right: A ≠ B ∧ B ≠ C ∧ C ≠ A) → 
  vertices_on_sides: (∃ D_on_AB : A ≤ B, ∃ E_on_BC : B ≤ C, ∃ F_on_CA : C ≤ A),
  angle_DEF : ∠ DEF = π / 2 ∧ ∠ EDF = π / 6 →
  (S_triangle [D, E, F] / S_triangle [A, B, C]) = 3 / 14 :=
sorry

end minimum_ratio_of_areas_l766_766208


namespace coincide_green_square_pairs_l766_766924

structure Figure :=
  (green_squares : ℕ)
  (red_triangles : ℕ)
  (blue_triangles : ℕ)

theorem coincide_green_square_pairs (f : Figure) (hs : f.green_squares = 4)
  (rt : f.red_triangles = 3) (bt : f.blue_triangles = 6)
  (gs_coincide : ∀ n, n ≤ f.green_squares ⟶ n = f.green_squares) 
  (rt_coincide : ∃ n, n = 2) (bt_coincide : ∃ n, n = 2) 
  (red_blue_pairs : ∃ n, n = 3) : 
  ∃ pairs, pairs = 4 :=
by 
  sorry

end coincide_green_square_pairs_l766_766924


namespace smallest_a_l766_766063

theorem smallest_a (a : ℕ) (h₁ : Nat.gcd a 70 > 1) (h₂ : Nat.gcd a 84 > 1) : a = 14 :=
sorry

end smallest_a_l766_766063


namespace evaluate_expression_l766_766813

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 :=
by sorry

end evaluate_expression_l766_766813


namespace miles_collection_height_l766_766304

-- Definitions based on conditions
def pages_per_inch_miles : ℕ := 5
def pages_per_inch_daphne : ℕ := 50
def daphne_height_inches : ℕ := 25
def longest_collection_pages : ℕ := 1250

-- Theorem to prove the height of Miles's book collection.
theorem miles_collection_height :
  (longest_collection_pages / pages_per_inch_miles) = 250 := by sorry

end miles_collection_height_l766_766304


namespace packs_sold_per_hour_in_low_season_l766_766122

theorem packs_sold_per_hour_in_low_season
  (h1 : 6) -- 6 packs are sold per hour during peak season
  (h2 : 60) -- Each tuna pack is sold at $60
  (h3 : 15) -- Fish sold for 15 hours a day
  (h4 : 5400 = 900 * x + 1800) -- Peak season revenue = low season revenue + $1800
  : x = 4 := 
by
  sorry

end packs_sold_per_hour_in_low_season_l766_766122


namespace probability_shortening_exactly_one_digit_l766_766845
open Real

theorem probability_shortening_exactly_one_digit :
  ∀ (sequence : List ℕ),
    (sequence.length = 2015) ∧
    (∀ (x ∈ sequence), x = 0 ∨ x = 9) →
      (P (λ seq, (length (shrink seq) = 2014)) sequence = 1.564e-90) := sorry

end probability_shortening_exactly_one_digit_l766_766845


namespace problem_a_problem_b_l766_766841

open Probability

noncomputable def sequenceShortenedByExactlyOneDigitProbability : ℝ :=
  1.564 * 10^(-90)

theorem problem_a (n : ℕ) (p : ℝ) (k : ℕ) (initialLength : ℕ)
  (h_n : n = 2014) (h_p : p = 0.1) (h_k : k = 1) (h_initialLength : initialLength = 2015) :
  (binomialₓ n k p * (1 - p)^(n - k)) = sequenceShortenedByExactlyOneDigitProbability := 
sorry

noncomputable def expectedLengthNewSequence : ℝ :=
  1813.6

theorem problem_b (n : ℕ) (p : ℝ) (initialLength : ℕ) 
  (h_n : n = 2014) (h_p : p = 0.1) (h_initialLength : initialLength = 2015) :
  (initialLength - n * p) = expectedLengthNewSequence :=
sorry

end problem_a_problem_b_l766_766841


namespace total_distance_traveled_is_80_l766_766863

/-- 
  We are given that a ball is dropped from a height of 20 meters. On each subsequent bounce, 
  it reaches 2/3 of the height from which it fell. The ball is caught at its high point 
  after hitting the ground for the fourth time.

  We want to prove that the total distance traveled by the ball when it is caught, 
  to the nearest meter, is 80 meters.
-/
theorem total_distance_traveled_is_80 :
  let initial_height := 20
  let r := 2 / 3
  let heights := [initial_height, initial_height * r, initial_height * r^2, initial_height * r^3, initial_height * r^4]
  let total_distance := heights.head + (heights.tail.map (· * 2)).sum
  total_distance ≈ 80 :=
by
  sorry

end total_distance_traveled_is_80_l766_766863


namespace f_100_l766_766690

noncomputable def f : ℕ → ℕ
| 0       := 0
| (n+1)   := f n + n + 1

lemma f_add (a b : ℕ) : f (a + b) = f a + f b + a * b := by
  induction b with b' ih
  · rw [Nat.add_zero, mul_zero, add_zero]
  · rw [Nat.add_succ, f, ih, add_assoc, add_assoc, add_comm b']
    exact congr_arg (fun x => x + a) (add_assoc _ _ _)

theorem f_100 : f 100 = 5050 := by
  sorry

end f_100_l766_766690


namespace inverse_as_linear_combination_l766_766687

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![4, 0], ![2, -6]]
def I : Matrix (Fin 2) (Fin 2) ℚ := 1

theorem inverse_as_linear_combination :
  ∃ c d : ℚ, Matrix.inv N = c • N + d • I := by
  use (1/24 : ℚ)
  use (1/12 : ℚ)
  sorry

end inverse_as_linear_combination_l766_766687


namespace find_H_coordinates_l766_766788

noncomputable def point := ℝ × ℝ × ℝ

def E : point := (-2, 3, 1)
def F : point := (4, 0, -5)
def G : point := (2, -4, 3)

def is_midpoint (M P Q : point) : Prop :=
  (M.1 = (P.1 + Q.1) / 2) ∧ (M.2 = (P.2 + Q.2) / 2) ∧ (M.3 = (P.3 + Q.3) / 2)

def is_parallelogram (E F G H : point) : Prop :=
  let M := ((E.1 + G.1) / 2, (E.2 + G.2) / 2, (E.3 + G.3) / 2) in
  is_midpoint M F H

theorem find_H_coordinates :
  ∃ H : point, is_parallelogram E F G H ∧ H = (-4, -1, 9) :=
by
  use (-4, -1, 9)
  split
  · -- To show it's a parallelogram, we need corresponding midpoint equality proofs
    let M : point := (0, -0.5, 2)
    have hM : is_midpoint M E G := by
      unfold is_midpoint
      split
      · simp [E, G]
      split
      · simp [E, G]
      · simp [E, G]
    have hFH : is_midpoint M F (-4, -1, 9) := by
      unfold is_midpoint
      split
      · simp [F, (-4, -1, 9)]
      split
      · simp [F, (-4, -1, 9)]
      · simp [F, (-4, -1, 9)]
    exact ⟨hM, hFH⟩
  · rfl

end find_H_coordinates_l766_766788


namespace ordered_pairs_unique_solution_l766_766942

noncomputable def ordered_pairs_count : ℕ :=
  Nat.card {p : ℝ × ℝ // 4^(p.1^2 + p.2) + 4^(p.1 + p.2^2) = 2}

theorem ordered_pairs_unique_solution : ordered_pairs_count = 1 :=
  sorry

end ordered_pairs_unique_solution_l766_766942


namespace smallest_oneic_divisible_by_63_l766_766491

-- Given conditions: Definition of a oneic number and divisibility.
def oneic_number (n : ℕ) : ℕ := (10^n - 1) / 9

theorem smallest_oneic_divisible_by_63 :
  (∃ n : ℕ, oneic_number n % 63 = 0 ∧ ∀ m < n, oneic_number m % 63 ≠ 0) → n = 18 :=
begin
  sorry
end

end smallest_oneic_divisible_by_63_l766_766491


namespace TA_TB_TC_eq_2AM_l766_766287

-- Definitions of points and triangle
variables {Point : Type} [EuclideanGeometry Point]

-- Definitions according to the given problem
variables {A B C T M : Point}
variables {triangle_ABC : Triangle A B C}
variables {triangle_prop : ∠A = 60°}
variables {T_inside_ABC : T ∈ interior triangle_ABC}
variables {T_angles : ∠ATB = 120° ∧ ∠BTC = 120° ∧ ∠CTA = 120°}
variables {M_midpoint : midpoint B C M}

-- Statement to prove
theorem TA_TB_TC_eq_2AM :
  TA + TB + TC = 2 * AM :=
sorry

end TA_TB_TC_eq_2AM_l766_766287


namespace intervals_of_monotonicity_ln_plus_a_over_x_minus_1_gt_1_l766_766300

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a / (x - 1)

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := ((x - 5 / 6) * (x - 6 / 5)) / (x * (x - 1) ^ 2)

-- (I)Prove intervals of monotonicity when a = 1/30
theorem intervals_of_monotonicity (a : ℝ) (h : a = 1 / 30) :
  ∀ x : ℝ, (0 < x ∧ x < 5 / 6) ∨ (6 / 5 < x ∧ x < +∞) → f' x a > 0 ∧ ((5 / 6 < x ∧ x < 1) ∨ (1 < x ∧ x < 6 / 5)) → f' x a < 0 :=
sorry

-- (II)Prove ln x + a / (x-1) > 1 for a ≥ 1/2 and x ∈ (1, +∞)
theorem ln_plus_a_over_x_minus_1_gt_1 (a : ℝ) (h : a ≥ 1 / 2) (x : ℝ) (hx : 1 < x) :
  Real.log x + a / (x - 1) > 1 :=
sorry

end intervals_of_monotonicity_ln_plus_a_over_x_minus_1_gt_1_l766_766300


namespace number_of_chords_l766_766963

theorem number_of_chords : (Nat.choose 10 3 + Nat.choose 10 4 + Nat.choose 10 5 + Nat.choose 10 6 + Nat.choose 10 7 + Nat.choose 10 8 + Nat.choose 10 9 + Nat.choose 10 10) = 968 :=
by
  sorry

end number_of_chords_l766_766963


namespace motorcycles_in_anytown_l766_766453

theorem motorcycles_in_anytown 
  (trucks sedans motorcycles : ℕ)
  (h : trucks : sedans : motorcycles = 3 : 7 : 2)
  (h_sedans : sedans = 9100) :
  motorcycles = 2600 :=
sorry

end motorcycles_in_anytown_l766_766453


namespace part1_solution_set_part2_range_of_m_l766_766959

noncomputable def f (x : ℝ) : ℝ := abs (x + 2) * abs (x - 3)

theorem part1_solution_set :
  {x : ℝ | f x > 7 - x} = {x : ℝ | x < -6 ∨ x > 2} :=
sorry

theorem part2_range_of_m (m : ℝ) :
  (∃ x : ℝ, f x ≤ abs (3 * m - 2)) → m ∈ Set.Iic (-1) ∪ Set.Ici (7 / 3) :=
sorry

end part1_solution_set_part2_range_of_m_l766_766959


namespace point_on_circle_l766_766254

theorem point_on_circle (x : ℝ) :
  let C := (7, 0)
      r := 14
      cir_eq := (x - 7)^2 + 10^2 = 196 in
  cir_eq → (x = 7 + 4 * Real.sqrt 6) ∨ (x = 7 - 4 * Real.sqrt 6) :=
by
  intros
  unfold cir_eq at *
  sorry

end point_on_circle_l766_766254


namespace fraction_of_phone_numbers_l766_766906

theorem fraction_of_phone_numbers (a b : ℕ) (h₁ : b = 7 * 10^7) 
  (h₂ : a = 10^6) : a / b = 1 / 70 := 
by
  rw [h₁, h₂]
  norm_num
  sorry

end fraction_of_phone_numbers_l766_766906


namespace max_value_f_max_value_g_l766_766088

-- Problem (1)
theorem max_value_f : 
  ∃ θ ∈ set.Ioc 0 (π / 2), (∀ θ' ∈ set.Ioc 0 (π / 2), (cos (θ' / 2) * sin θ') ≤ (4 * real.sqrt 3 / 9)) ∧ 
  (cos (θ / 2) * sin θ) = (4 * real.sqrt 3 / 9) := 
sorry

-- Problem (2)
theorem max_value_g : 
  ∃ θ ∈ set.Ioc 0 (π / 2), (∀ θ' ∈ set.Ioc 0 (π / 2), (sin (θ' / 2) * cos θ') ≤ (real.sqrt 6 / 9)) ∧ 
  (sin (θ / 2) * cos θ) = (real.sqrt 6 / 9) := 
sorry

end max_value_f_max_value_g_l766_766088


namespace angle_between_olivia_and_nathan_l766_766910

-- Define the points on Earth
structure PointOnEarth where
  latitude : ℝ    -- Latitude in degrees
  longitude : ℝ   -- Longitude in degrees

-- Define Olivia's and Nathan's positions.
def Olivia : PointOnEarth := {
  latitude := 0,
  longitude := 10
}

def Nathan : PointOnEarth := {
  latitude := 30,
  longitude := -40
}

-- Define the center of the Earth.
def C : PointOnEarth := {
  latitude := 0,
  longitude := 0
}

-- Non-exact equivalent proof statement since the Earth is a perfect sphere.
def angle_CON_approx : ℝ :=
  43.3   -- The given approximate result

-- The theorem to prove the angle ∠CON approximately equals 43.3°
theorem angle_between_olivia_and_nathan :
  let O := Olivia
  let N := Nathan
  let C := C
  ∠ O C N ≈ angle_CON_approx :=
sorry

end angle_between_olivia_and_nathan_l766_766910


namespace exponentiation_problem_l766_766639

variable (x : ℝ) (m n : ℝ)

theorem exponentiation_problem (h1 : x ^ m = 5) (h2 : x ^ n = 1 / 4) :
  x ^ (2 * m - n) = 100 :=
sorry

end exponentiation_problem_l766_766639


namespace comparison_of_probabilities_l766_766709

noncomputable def fair_die (n : ℕ) : ℕ → ℝ := λ i, if i ∈ {1, 2, 3, 4, 5, 6} then 1 / 6 else 0

noncomputable def random_variable_sum (num_dice : ℕ) (die_result : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range num_dice, die_result (i + 1)

noncomputable def probability_sum_at_least (num_dice : ℕ) (threshold : ℝ) : Prop :=
  (random_variable_sum num_dice fair_die) ≥ threshold

noncomputable def probability_sum_less_than (num_dice : ℕ) (threshold : ℝ) : Prop :=
  (random_variable_sum num_dice fair_die) < threshold

theorem comparison_of_probabilities : probability_sum_at_least 90 500 > probability_sum_less_than 90 130 :=
sorry

end comparison_of_probabilities_l766_766709


namespace sum_last_two_digits_modified_fibonacci_factorial_series_l766_766064

theorem sum_last_two_digits_modified_fibonacci_factorial_series :
  let fibs := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
  let facts := List.map (λ n, n!) fibs
  let last_two_digits := List.map (λ n : Nat, n % 100) facts
  let sum_last_two_digits := last_two_digits.take 6 ++ last_two_digits.drop 6.map (λ _, 0)
  List.sum sum_last_two_digits % 10 = 5 := by
  sorry

end sum_last_two_digits_modified_fibonacci_factorial_series_l766_766064


namespace arithmetic_seq_sum_l766_766466

theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 3 + a 7 = 37) :
  a 2 + a 4 + a 6 + a 8 = 74 := 
sorry

end arithmetic_seq_sum_l766_766466


namespace x_is_two_or_three_l766_766984

open Finset

variable {α : Type*} [DecidableEq α]

def A : Finset ℕ := {1, 2, 3}
def B (x : ℕ) : Finset ℕ := {1, x, 4}
def U : Finset ℕ := {1, 2, 3, 4}

theorem x_is_two_or_three {x : ℕ} (h : A ∪ B x = U) : x = 2 ∨ x = 3 :=
by
  have hAB : ∀ y, y ∈ A ∪ B x → y ∈ U := by rw [h]
  -- Further proof steps would go here
  sorry

end x_is_two_or_three_l766_766984


namespace identify_correct_statements_l766_766506

-- Definitions of terms and conditions
def triangle := Type*

def obtuse (a : ℝ) : Prop := a > 90

def angle_bisectors_intersect_inside (T : triangle) : Prop :=
  ∃ P, ∀ A B C, A ≠ B ∧ B ≠ C ∧ C ≠ A → P ∈ (bisector_segment A B C A)

def altitudes_intersect_point (T : triangle) : Prop :=
  ∃ P, ∀ A B C, A ≠ B ∧ B ≠ C ∧ C ≠ A →
    is_acute_triangle A B C ∧ P ∈ altitudes_intersection_point_in (A B C) ∨
    (right_angle_edge A B C P) ∨
    (obtuse_triangle A B C ∧ P ∈ altitudes_intersection_point_out (A B C))

def median_equal_area (T : triangle) : Prop :=
  ∀ A B C M, median_segment A B C M → (area_triangle A M C = area_triangle B M C)

-- Proof problem statement
theorem identify_correct_statements (T : triangle) :
    (at_most_one_angle_obtuse T) ∧
    (angle_bisectors_intersect_inside T) ∧
    ¬(altitudes_intersect_point T) ∧
    ¬(isosceles_triangle_always_acute_or_right T) ∧
    (median_equal_area T) :=
begin
  -- these statements match the problem conditions and the correct answers
  sorry
end

end identify_correct_statements_l766_766506


namespace sequence_formula_l766_766186

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 2 * ∑ i in finset.range n, sequence i

theorem sequence_formula (n : ℕ) : sequence n = 
  match n with
  | 0   => 1
  | k + 1 => 2 * 3 ^ k
  end :=
sorry

end sequence_formula_l766_766186


namespace band_member_earnings_l766_766164

theorem band_member_earnings :
  let attendees := 500
  let ticket_price := 30
  let band_share_percentage := 70 / 100
  let band_members := 4
  let total_earnings := attendees * ticket_price
  let band_earnings := total_earnings * band_share_percentage
  let earnings_per_member := band_earnings / band_members
  earnings_per_member = 2625 := 
by {
  sorry
}

end band_member_earnings_l766_766164


namespace walking_west_is_negative_l766_766238

-- Definitions based on conditions
def east (m : Int) : Int := m
def west (m : Int) : Int := -m

-- Proof statement (no proof required, so use "sorry")
theorem walking_west_is_negative (m : Int) (h : east 8 = 8) : west 10 = -10 :=
by
  sorry

end walking_west_is_negative_l766_766238


namespace no_sequence_ak_eq_bk_even_n_sum_of_descending_sequence_l766_766578

-- Problem 1
def sequences_n3_satisfying_condition : list (list ℕ × list ℕ) :=
[( [1, 3, 2], [3, 1, 2] ),
 ( [2, 3, 1], [2, 1, 3] )]

-- Problem 2
theorem no_sequence_ak_eq_bk_even_n (n : ℕ) (h : n % 2 = 0) : 
  ¬ ∃ (a : ℕ → ℕ) b, 
    (∀ k, k ∈ (finset.range n).map (nat.succ) → a k ∈ (finset.range n).map (nat.succ)
     ∧ b k = n + 1 - a k
     ∧ a k = b k) :=
begin
  sorry
end

-- Problem 3
theorem sum_of_descending_sequence (n : ℕ) :
  ∑ k in finset.range (n+1), (k+1) * (n+1 - k) = (1/6 : ℚ) * n * (n+1) * (n+2) :=
sorry

end no_sequence_ak_eq_bk_even_n_sum_of_descending_sequence_l766_766578


namespace convex_quadrilateral_geometric_progression_exists_l766_766920

theorem convex_quadrilateral_geometric_progression_exists :
  ∃ (a b c d e f : ℝ) (q : Quadrilateral a b c d e f),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
    q.convex ∧
    (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4 ∧ f = a * r^5) :=
sorry

end convex_quadrilateral_geometric_progression_exists_l766_766920


namespace shift_cos_to_sin_l766_766024

theorem shift_cos_to_sin (x : ℝ) : 
  ∀ (f g : ℝ → ℝ), 
  (f x = cos (2 * x)) → 
  (g x = sin (2 * x)) → 
  (g x = f (x - (π / 4))) := 
by 
  intros f g hf hg 
  rw [hf, hg]
  sorry

end shift_cos_to_sin_l766_766024


namespace ratio_of_salad_to_fry_pack_l766_766518

-- Define the conditions
def total_cost : ℝ := 15
def cost_per_fry_pack : ℝ := 2
def cost_burger : ℝ := 5

-- Define the price of the salad
def price_salad : ℝ := total_cost - cost_burger - 2 * cost_per_fry_pack

-- Prove the required ratio
theorem ratio_of_salad_to_fry_pack : price_salad / cost_per_fry_pack = 3 := by
  have h_total_cost : total_cost = 15 := by rfl
  have h_cost_burger : cost_burger = 5 := by rfl
  have h_cost_per_fry_pack : cost_per_fry_pack = 2 := by rfl
  have h_price_salad : price_salad = 15 - 5 - 2 * 2 := by
    rw [h_total_cost, h_cost_burger, h_cost_per_fry_pack]
  simp [price_salad, h_price_salad]
  sorry

end ratio_of_salad_to_fry_pack_l766_766518


namespace sarees_original_price_l766_766759

theorem sarees_original_price (P : ℝ) (h : 0.75 * 0.85 * P = 248.625) : P = 390 :=
by
  sorry

end sarees_original_price_l766_766759


namespace train_length_l766_766452

theorem train_length (v_kmh : ℝ) (p_len : ℝ) (t_sec : ℝ) (l_train : ℝ) 
  (h_v : v_kmh = 72) (h_p : p_len = 250) (h_t : t_sec = 26) :
  l_train = 270 :=
by
  sorry

end train_length_l766_766452


namespace volume_of_regular_square_pyramid_l766_766009

noncomputable def volume_of_pyramid {a : ℝ} (h_pos : 0 < a) (h_angle : ∃ θ : ℝ, θ = 45) : ℝ :=
  let height := a / 2 in
  (1 / 3) * (a^2) * height

theorem volume_of_regular_square_pyramid (a : ℝ) (h_pos : 0 < a) (h_angle : ∃ θ : ℝ, θ = 45) :
  volume_of_pyramid h_pos h_angle = a^3 / 6 :=
sorry

end volume_of_regular_square_pyramid_l766_766009


namespace third_intermission_served_l766_766923

def total_served : ℚ :=  0.9166666666666666
def first_intermission : ℚ := 0.25
def second_intermission : ℚ := 0.4166666666666667

theorem third_intermission_served : first_intermission + second_intermission ≤ total_served →
  (total_served - (first_intermission + second_intermission)) = 0.25 :=
by
  sorry

end third_intermission_served_l766_766923


namespace cube_root_of_8_is_2_l766_766733

theorem cube_root_of_8_is_2 : (∛8 : ℝ) = 2 :=
by
  sorry

end cube_root_of_8_is_2_l766_766733


namespace max_value_of_3sinx_4cosx_is_5_l766_766153

def max_value_of_function (a b : ℝ) : ℝ :=
  (sqrt (a^2 + b^2))

theorem max_value_of_3sinx_4cosx_is_5 :
  max_value_of_function 3 4 = 5 :=
by
  sorry

end max_value_of_3sinx_4cosx_is_5_l766_766153


namespace matching_pair_in_6_plates_l766_766017

theorem matching_pair_in_6_plates :
    ∃ (white green red pink purple : ℕ), 
     0 < white → 0 < green → 0 < red → 0 < pink → 0 < purple → 
     ∀ (pulls : Finset (Σ c : fin 5, fin (match c with
                                              | 0 => white
                                              | 1 => green
                                              | 2 => red
                                              | 3 => pink
                                              | _ => purple))),
        pulls.card = 6 → 
        ∃ c1 c2 : pulls, c1.1 = c2.1 :=
by sorry

end matching_pair_in_6_plates_l766_766017


namespace Gloria_pine_tree_price_l766_766620

theorem Gloria_pine_tree_price :
  ∀ (cabin_cost cash cypress_count pine_count maple_count cypress_price maple_price left_over_price : ℕ)
  (cypress_total maple_total total_required total_from_cypress_and_maple total_needed amount_per_pine : ℕ),
    cabin_cost = 129000 →
    cash = 150 →
    cypress_count = 20 →
    pine_count = 600 →
    maple_count = 24 →
    cypress_price = 100 →
    maple_price = 300 →
    left_over_price = 350 →
    cypress_total = cypress_count * cypress_price →
    maple_total = maple_count * maple_price →
    total_required = cabin_cost - cash + left_over_price →
    total_from_cypress_and_maple = cypress_total + maple_total →
    total_needed = total_required - total_from_cypress_and_maple →
    amount_per_pine = total_needed / pine_count →
    amount_per_pine = 200 :=
by
  intros
  sorry

end Gloria_pine_tree_price_l766_766620


namespace problem_a_problem_b_l766_766844

open Probability

noncomputable def sequenceShortenedByExactlyOneDigitProbability : ℝ :=
  1.564 * 10^(-90)

theorem problem_a (n : ℕ) (p : ℝ) (k : ℕ) (initialLength : ℕ)
  (h_n : n = 2014) (h_p : p = 0.1) (h_k : k = 1) (h_initialLength : initialLength = 2015) :
  (binomialₓ n k p * (1 - p)^(n - k)) = sequenceShortenedByExactlyOneDigitProbability := 
sorry

noncomputable def expectedLengthNewSequence : ℝ :=
  1813.6

theorem problem_b (n : ℕ) (p : ℝ) (initialLength : ℕ) 
  (h_n : n = 2014) (h_p : p = 0.1) (h_initialLength : initialLength = 2015) :
  (initialLength - n * p) = expectedLengthNewSequence :=
sorry

end problem_a_problem_b_l766_766844


namespace admissible_coloring_transition_l766_766395

/-- In a 100x100 board where each cell is either black or white, a coloring is admissible if
    for any row or column, the number of black colored cells is between 50 and 60 inclusive.
    We want to prove that one can transition from any admissible coloring A of the board
    to any other admissible coloring B using a sequence of valid recoloring operations. -/
theorem admissible_coloring_transition :
  ∀ (A B : (fin 100) → (fin 100) → bool),
    (∀ i, 50 ≤ (finset.univ.filter (λ j, A i j = tt)).card ∧
          (finset.univ.filter (λ j, A i j = tt)).card ≤ 60) →
    (∀ j, 50 ≤ (finset.univ.filter (λ i, A i j = tt)).card ∧
          (finset.univ.filter (λ i, A i j = tt)).card ≤ 60) →
    (∀ i, 50 ≤ (finset.univ.filter (λ j, B i j = tt)).card ∧
          (finset.univ.filter (λ j, B i j = tt)).card ≤ 60) →
    (∀ j, 50 ≤ (finset.univ.filter (λ i, B i j = tt)).card ∧
          (finset.univ.filter (λ i, B i j = tt)).card ≤ 60) →
    (∃ (f : list ((fin 100) × (fin 100))), 
       ∀ k, k < f.length → 
          let C := (A : (fin 100 → fin 100 → bool)) 
            (f.k.1 := B (f.k.1) ; 
             f.k.2 := B (f.k.2)) 
          in (∀ i, 50 ≤ (finset.univ.filter (λ j, C i j = tt)).card ∧
                  (finset.univ.filter (λ j, C i j = tt)).card ≤ 60 ∧
              (∀ j, 50 ≤ (finset.univ.filter (λ i, C i j = tt)).card ∧
                  (finset.univ.filter (λ i, C i j = tt)).card ≤ 60)
) :=
sorry

end admissible_coloring_transition_l766_766395


namespace count_k_with_eight_as_first_digit_l766_766693

noncomputable def has_eight_as_first_digit (k : ℕ) : Prop :=
  let a := Real.log10 8
  let f := k * a - Real.floor (k * a)
  0.90309 ≤ f ∧ f < 0.95424

theorem count_k_with_eight_as_first_digit : 
  (S : Finset ℕ := Finset.filter (λ k, has_eight_as_first_digit k) (Finset.range 3001)) :
  S.card = 153 :=
by 
  let a := Real.log10 8
  let interval_length := 0.95424 - 0.90309
  let expected_count := Real.floor (interval_length * 3000)
  sorry

end count_k_with_eight_as_first_digit_l766_766693


namespace arithmetic_sequence_sum_l766_766666

theorem arithmetic_sequence_sum :
  (∀ (a : ℕ → ℕ), (a 2 = 4) ∧ (a 4 + a 7 = 15) → (∀ n, a n = n + 2))
  ∧ (let a (n : ℕ) := n + 2 in 
    (∑ k in Finset.range 10, 1 / (a k * a (k + 1)) = 10 / 39)) :=
by
  sorry

end arithmetic_sequence_sum_l766_766666


namespace solution_set_of_inequality_l766_766415

theorem solution_set_of_inequality :
  {x : ℝ | (3 / (5 - 3 * x) > 1)} = set.Ioo (2 / 3) (5 / 3) := sorry

end solution_set_of_inequality_l766_766415


namespace ticket_cost_difference_l766_766428

theorem ticket_cost_difference (num_prebuy : ℕ) (price_prebuy : ℕ) (num_gate : ℕ) (price_gate : ℕ)
  (h_prebuy : num_prebuy = 20) (h_price_prebuy : price_prebuy = 155)
  (h_gate : num_gate = 30) (h_price_gate : price_gate = 200) :
  num_gate * price_gate - num_prebuy * price_prebuy = 2900 :=
by
  sorry

end ticket_cost_difference_l766_766428


namespace seed_germination_probability_l766_766783

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem seed_germination_probability :
  ∀ (n : ℕ) (p : ℝ) (k : ℕ), n = 15 → p = 0.9 → k = 14 →
  (binomial n k) * (p ^ k) * ((1 - p) ^ (n - k)) = (binomial 15 14) * (0.9 ^ 14) * ((1 - 0.9) ^ (15 - 14)) :=
by
  intro n p k hn hp hk
  rw [hn, hp, hk]
  sorry

end seed_germination_probability_l766_766783


namespace smallest_positive_x_for_palindrome_addition_l766_766439

def is_palindrome (n : ℕ) : Prop :=
  n.to_digits 10 = n.to_digits 10.reverse

theorem smallest_positive_x_for_palindrome_addition :
  ∃ (x : ℕ), x > 0 ∧ is_palindrome (x + 4321) ∧ x = 13 :=
by
  sorry

end smallest_positive_x_for_palindrome_addition_l766_766439


namespace find_a_values_l766_766614

def setA (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.snd - 3) / (p.fst - 2) = a + 1}

def setB (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (a^2 - 1) * p.fst + (a - 1) * p.snd = 15}

def sets_disjoint (A B : Set (ℝ × ℝ)) : Prop := ∀ p : ℝ × ℝ, p ∉ A ∪ B

theorem find_a_values (a : ℝ) :
  sets_disjoint (setA a) (setB a) ↔ a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -4 :=
sorry

end find_a_values_l766_766614


namespace positive_integer_in_base_proof_l766_766070

noncomputable def base_conversion_problem (A B : ℕ) (n : ℕ) : Prop :=
  n = 9 * A + B ∧ n = 8 * B + A ∧ A < 9 ∧ B < 8 ∧ A ≠ 0 ∧ B ≠ 0

theorem positive_integer_in_base_proof (A B n : ℕ) (h : base_conversion_problem A B n) : n = 0 :=
sorry

end positive_integer_in_base_proof_l766_766070


namespace volume_of_cube_l766_766575

noncomputable def distance (P L : ℝ) : ℝ := 1  -- assuming P and L are the distance apart

-- Given definitions
variables (P L : Point)
variables (d : ℝ)
variables (a : ℝ) (V : ℝ)

-- Given condition: distance from P to L is 1 cm
axiom dist_P_L : distance P L = 1

-- Prove the volume V of the cube
theorem volume_of_cube (P L : Point) (d = 1 : ℝ) : 
  ∃ (a : ℝ), (a = (Real.sqrt 6) / 2) ∧ V = a ^ 3 = (3 * Real.sqrt 6) / 4 :=
sorry

end volume_of_cube_l766_766575


namespace sum_of_first_10_terms_l766_766414

-- Define the sequence aₙ
def a : ℕ → ℕ
| 0     := 2
| (n+1) := a n + 2^n

-- Define the inverse of the sequence aₙ
def inv_sequence (n : ℕ) : ℝ :=
  1 / (a n)

-- Define the sum of the first 10 terms of the inverse sequence
def sum_first_10_inv_sequence : ℝ :=
  ∑ i in range 10, inv_sequence i

-- The theorem we want to prove
theorem sum_of_first_10_terms :
  sum_first_10_inv_sequence = 1023 / 1024 :=
begin
  sorry
end

end sum_of_first_10_terms_l766_766414


namespace different_tea_packets_or_miscalculation_l766_766915

theorem different_tea_packets_or_miscalculation : 
  ∀ (n_1 n_2 : ℕ), 3 ≤ t_1 ∧ t_1 ≤ 4 ∧ 3 ≤ t_2 ∧ t_2 ≤ 4 ∧
  (74 = t_1 * x ∧ 105 = t_2 * y → x ≠ y) ∨ 
  (∃ (e_1 e_2 : ℕ), (e_1 + e_2 = 74) ∧ (e_1 + e_2 = 105) → false) :=
by
  -- Construction based on the provided mathematical problem
  sorry

end different_tea_packets_or_miscalculation_l766_766915


namespace events_mutually_exclusive_not_complementary_l766_766540

-- Define the set of balls and people
inductive Ball : Type
| b1 | b2 | b3 | b4

inductive Person : Type
| A | B | C | D

-- Define the event types
structure Event :=
  (p : Person)
  (b : Ball)

-- Define specific events as follows
def EventA : Event := { p := Person.A, b := Ball.b1 }
def EventB : Event := { p := Person.B, b := Ball.b1 }

-- We want to prove the relationship between two specific events:
-- "Person A gets ball number 1" and "Person B gets ball number 1"
-- Namely, that they are mutually exclusive but not complementary.

theorem events_mutually_exclusive_not_complementary :
  (∀ e : Event, (e = EventA → ¬ (e = EventB)) ∧ ¬ (e = EventA ∨ e = EventB)) :=
sorry

end events_mutually_exclusive_not_complementary_l766_766540


namespace volume_of_cube_is_1000_l766_766272

-- Each die has an edge of 2 cm
def edge_length_of_die : ℝ := 2

-- Total number of dice
def number_of_dice : ℕ := 125

-- The number of dice along each edge of the cube
def dice_per_edge : ℕ := Int.to_nat (real.cbrt number_of_dice)

-- The edge length of the whole cube
def edge_length_of_cube : ℝ := dice_per_edge * edge_length_of_die

-- The volume of the cube
def volume_of_cube : ℝ := edge_length_of_cube ^ 3

-- Theorem to prove
theorem volume_of_cube_is_1000 
  (h1 : edge_length_of_die = 2)
  (h2 : number_of_dice = 125)
  (h3 : dice_per_edge = Int.to_nat (real.cbrt number_of_dice))
  (h4 : edge_length_of_cube = dice_per_edge * edge_length_of_die)
  (h5 : volume_of_cube = edge_length_of_cube ^ 3) : 
  volume_of_cube = 1000 := 
  sorry

end volume_of_cube_is_1000_l766_766272


namespace triangle_sine_ratio_l766_766509

variables {a b c a' b' c' : ℝ}
variables {β1 γ1 α2 γ2 α3 β3 : ℝ}

theorem triangle_sine_ratio (h1: 0 < a) (h2: 0 < b) (h3: 0 < c)
    (h4: 0 < a') (h5: 0 < b') (h6: 0 < c')
    (β1_angle: 0 < β1) (γ1_angle: 0 < γ1) (α2_angle: 0 < α2) 
    (γ2_angle: 0 < γ2) (α3_angle: 0 < α3) (β3_angle: 0 < β3) :
    (sin (β1 + γ1)) / (a * a') = (sin (α2 + γ2)) / (b * b') ∧
    (sin (β1 + γ1)) / (a * a') = (sin (α3 + β3)) / (c * c') :=
by sorry

end triangle_sine_ratio_l766_766509


namespace ratio_of_Carla_to_Cosima_l766_766306

variables (C M : ℝ)

-- Natasha has 3 times as much money as Carla
axiom h1 : 3 * C = 60

-- Carla has the same amount of money as Cosima
axiom h2 : C = M

-- Prove: the ratio of Carla's money to Cosima's money is 1:1
theorem ratio_of_Carla_to_Cosima : C / M = 1 :=
by sorry

end ratio_of_Carla_to_Cosima_l766_766306


namespace at_most_n_diameters_l766_766971

theorem at_most_n_diameters {n : ℕ} (h : n ≥ 3) (points : Fin n → ℝ × ℝ) (d : ℝ) 
  (hd : ∀ i j, dist (points i) (points j) ≤ d) :
  ∃ (diameters : Fin n → Fin n), 
    (∀ i, dist (points i) (points (diameters i)) = d) ∧
    (∀ i j, (dist (points i) (points j) = d) → 
      (∃ k, k = i ∨ k = j → diameters k = if k = i then j else i)) :=
sorry

end at_most_n_diameters_l766_766971


namespace find_value_of_a_l766_766998

theorem find_value_of_a (a : ℝ) :
  let line := ∀ x, -x + 1 = -a + 1 in
  let center_of_circle := (a, -1) in
  line a → a = 2 :=
by
  intro line_passes_through_center
  have center := (a, -1)
  have equation_at_center := line_passes_through_center a
  show a = 2
  sorry

end find_value_of_a_l766_766998


namespace plant_height_after_year_l766_766477

theorem plant_height_after_year (current_height : ℝ) (monthly_growth : ℝ) (months_in_year : ℕ) (total_growth : ℝ)
  (h1 : current_height = 20)
  (h2 : monthly_growth = 5)
  (h3 : months_in_year = 12)
  (h4 : total_growth = monthly_growth * months_in_year) :
  current_height + total_growth = 80 :=
sorry

end plant_height_after_year_l766_766477


namespace election_required_percentage_l766_766244

def votes_cast : ℕ := 10000

def geoff_percentage : ℕ := 5
def geoff_received_votes := (geoff_percentage * votes_cast) / 1000

def extra_votes_needed : ℕ := 5000
def total_votes_needed := geoff_received_votes + extra_votes_needed

def required_percentage := (total_votes_needed * 100) / votes_cast

theorem election_required_percentage : required_percentage = 505 / 10 :=
by
  sorry

end election_required_percentage_l766_766244


namespace percentage_of_green_ducks_l766_766659

def total_ducks := 100
def green_ducks_smaller_pond := 9
def green_ducks_larger_pond := 22
def total_green_ducks := green_ducks_smaller_pond + green_ducks_larger_pond

theorem percentage_of_green_ducks :
  (total_green_ducks / total_ducks) * 100 = 31 :=
by
  sorry

end percentage_of_green_ducks_l766_766659


namespace larger_circle_radius_l766_766754

theorem larger_circle_radius (r R : ℝ) 
  (h : (π * R^2) / (π * r^2) = 5 / 2) : 
  R = r * Real.sqrt 2.5 :=
sorry

end larger_circle_radius_l766_766754


namespace simplify_expr_l766_766348

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l766_766348


namespace calculate_expression_l766_766126

theorem calculate_expression : 56.8 * 35.7 + 56.8 * 28.5 + 64.2 * 43.2 = 6420 := 
by sorry

end calculate_expression_l766_766126


namespace bus_interval_three_buses_l766_766032

theorem bus_interval_three_buses (T : ℕ) (h : T = 21) : (T * 2) / 3 = 14 :=
by
  sorry

end bus_interval_three_buses_l766_766032


namespace train_speed_l766_766501

theorem train_speed (length : ℕ) (time : ℕ) (h_length : length = 120) (h_time : time = 4) : 
  (length / time = 30) :=
by
  rw [h_length, h_time]
  norm_num
  sorry -- Add the proof steps here if required, but for now, we just state the theorem.

end train_speed_l766_766501


namespace range_of_m_l766_766565

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), (x > 2 * m ∧ x ≥ m - 3) ∧ x = 1) ↔ 0 ≤ m ∧ m < 0.5 :=
by
  sorry

end range_of_m_l766_766565


namespace parabola_properties_l766_766605

-- Define the parabola C with its given properties
def parabola (p : ℝ) (p_pos : p > 0) : ℝ × ℝ → Prop := λ Q, Q.snd ^ 2 = 2 * p * Q.fst

-- Given point Q(2,2)
def Q : ℝ × ℝ := (2, 2)

-- Define the line passing through point M(2,0)
def line (m : ℝ) : ℝ → ℝ := λ y, m * y + 2

-- Define slope calculations for lines through the origin
def slope (x y : ℝ) : ℝ := y / x

-- Main theorem statement
theorem parabola_properties :
  (∀ p : ℝ, p > 0 → parabola p Q) →
  (∃ p : ℝ, p > 0 ∧ parabola p Q ∧ (parabola 1 Q) ∧ (Q.snd ^ 2 = 2 * Q.fst)) ∧
  (∀ m : ℝ, ∃ y1 y2 : ℝ, y1 + y2 = 2 * m ∧ y1 * y2 = -4 ∧ (k1 k2 = -1) 
  ∧ (let k1 := slope (line m y1) y1 in
      let k2 := slope (line m y2) y2 in
      k1 * k2 = -1)) :=
by
  sorry

end parabola_properties_l766_766605


namespace valid_subset_count_l766_766629

theorem valid_subset_count : 
  (∑ k in Finset.range 10, Nat.choose (18 - 2 * k + 1) k) = 871 :=
by sorry

end valid_subset_count_l766_766629


namespace sqrt_simplification_l766_766367

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l766_766367


namespace maximum_value_of_expression_l766_766299

theorem maximum_value_of_expression (a c : ℝ) 
  (h_mono : ∀ x, f'(x) = ax^2 - 4x ≥ 0)
  (h_a_le : a ≤ 4)
  (h_ac : a * c = 4) :
  max_value := (frac a (c^2 + 4)) + (frac c (a^2 + 4)) = 1/2 :=
sorry

end maximum_value_of_expression_l766_766299


namespace total_discount_l766_766681

-- Definitions for prices and discounts
def tshirt_price_euro : ℝ := 30
def backpack_price_euro : ℝ := 10
def cap_price_euro : ℝ := 5
def jeans_price_euro : ℝ := 50
def sneakers_price_dollar : ℝ := 60

def tshirt_discount : ℝ := 0.10
def backpack_discount : ℝ := 0.20
def cap_discount : ℝ := 0.15
def jeans_discount : ℝ := 0.25
def sneakers_discount : ℝ := 0.30

def euro_to_dollar : ℝ := 1.2

-- Lean 4 statement for the proof
theorem total_discount (tshirt_price_euro backpack_price_euro cap_price_euro jeans_price_euro sneakers_price_dollar : ℝ)
                       (tshirt_discount backpack_discount cap_discount jeans_discount sneakers_discount euro_to_dollar : ℝ) :
  let tshirt_discount_euro := tshirt_discount * tshirt_price_euro
      backpack_discount_euro := backpack_discount * backpack_price_euro
      cap_discount_euro := cap_discount * cap_price_euro
      jeans_discount_euro := jeans_discount * jeans_price_euro
      sneakers_discount_euro := (sneakers_discount * sneakers_price_dollar) / euro_to_dollar
      total_discount_euro := tshirt_discount_euro + backpack_discount_euro + cap_discount_euro + jeans_discount_euro + sneakers_discount_euro
      total_discount_dollar := total_discount_euro * euro_to_dollar
  in
  total_discount_euro = 33.25 ∧ total_discount_dollar = 39.90 := sorry

end total_discount_l766_766681


namespace no_matrix_adds_three_to_second_column_l766_766936

theorem no_matrix_adds_three_to_second_column :
  ¬ ∃ (M : Matrix (Fin 2) (Fin 2) ℚ), 
    ∀ (X : Matrix (Fin 2) (Fin 2) ℚ), 
    M.mul X = X + (Matrix.vecCons (Matrix.vecCons 0 3 : Fin 2 → ℚ) (Matrix.vecCons 0 3 : Fin 2 → ℚ)) :=
begin
  sorry
end

end no_matrix_adds_three_to_second_column_l766_766936


namespace sequence_general_formula_l766_766583

section
variable {f : ℝ → ℝ}
variable {a : ℝ}
variable {xₙ : ℕ → ℝ}

-- Given conditions
def f_def (x : ℝ) := x / (a * x + 3)
def a_value := a = -1 / 2
def x_initial := xₙ 1 = 1
def x_recur (n : ℕ) := xₙ (n + 1) = f (xₙ n)

-- Define the sequence as per the given problem
noncomputable def x_n_general (n : ℕ) := 2 / (n + 1)

-- Theorem to prove the general form of the sequence
theorem sequence_general_formula (a_val : a_value) (initial : x_initial) (recur : ∀ n, x_recur n) :
  ∀ n, xₙ n = x_n_general n := sorry
          
end

end sequence_general_formula_l766_766583


namespace cells_at_day_10_l766_766508

theorem cells_at_day_10 :
  ∃ n : ℕ, n = 240 ∧
  let initial_cells := 5 in
  let cells_day_2 := initial_cells * 3 in
  let cells_day_4 := cells_day_2 * 2 in
  let cells_day_6 := cells_day_4 * 2 in
  let cells_day_8 := cells_day_6 * 2 in
  let cells_day_10 := cells_day_8 * 2 in
  n = cells_day_10 :=
by {
  existsi 240,
  split,
  { refl },
  { simp only [*, mul_assoc, Nat.mul_eq_zero] }
}

end cells_at_day_10_l766_766508


namespace largest_perfect_square_factor_of_3402_l766_766053

theorem largest_perfect_square_factor_of_3402 : 
  ∃ (n : ℕ), (∃ (k : ℕ), 3402 = k * k * n) ∧ 81 = k * k :=
begin
  sorry
end

end largest_perfect_square_factor_of_3402_l766_766053


namespace distance_between_home_and_retreat_l766_766562

theorem distance_between_home_and_retreat (D : ℝ) 
  (h1 : D / 50 + D / 75 = 10) : D = 300 :=
sorry

end distance_between_home_and_retreat_l766_766562


namespace students_in_a_class_l766_766880

theorem students_in_a_class
  (art_students : ℕ)
  (music_students : ℕ)
  (both_students : ℕ)
  (H_art : art_students = 35)
  (H_music : music_students = 32)
  (H_both : both_students = 19)
  (H_total : (art_students + music_students - both_students) = 48) :
  ∃ total_students : ℕ, total_students = 48 :=
by
  use (art_students + music_students - both_students)
  rw [H_art, H_music, H_both]
  exact H_total

end students_in_a_class_l766_766880


namespace bryden_total_amount_l766_766866

-- Define the conditions
def collector_multiplier : ℝ := 25 -- collector offers 2500% of face value, which is 25 times face value
def face_value_per_quarter : ℝ := 0.25 -- face value of each quarter
def num_quarters : ℝ := 7 -- Bryden has seven state quarters
def bonus_per_five_quarters : ℝ := 2 -- $2 bonus for every set of five quarters

-- Calculate the total expected amount Bryden will get
theorem bryden_total_amount : 
  let total_face_value := num_quarters * face_value_per_quarter in
  let initial_payment := collector_multiplier * total_face_value in
  let total_amount := initial_payment + bonus_per_five_quarters in
  total_amount = 45.75 :=
by
  -- Use sorry to skip the proof
  sorry

end bryden_total_amount_l766_766866


namespace find_last_three_digits_l766_766879

def is_digit_list (n : ℕ) : Prop :=
  n ≥ 2000 ∧ (n.nat_abs.digits 10).head = 2

def last_three_digits_in_order (n1 n2 n3 : ℕ) : Prop :=
  (∀ n, is_digit_list n) → (n1, n2, n3 = 4, 2, 7)

theorem find_last_three_digits :
  ∃ n1 n2 n3 : ℕ, last_three_digits_in_order n1 n2 n3 ∧ n1 = 4 ∧ n2 = 2 ∧ n3 = 7 :=
by
  sorry

end find_last_three_digits_l766_766879


namespace chessboard_distance_sum_equals_l766_766047

def is_chessboard_8x8 (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8

def is_black_square (i j : ℕ) : Prop :=
  (i + j) % 2 = 1

def is_white_square (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

noncomputable def sum_of_squares (x y : ℕ) (f : ℕ → ℕ → Prop) : ℕ :=
  ∑ i in (finset.range 8).map (λ i, i + 1), 
    ∑ j in (finset.range 8).map (λ j, j + 1), 
    if f i j then (x - i) ^ 2 + (y - j) ^ 2 else 0

theorem chessboard_distance_sum_equals (x y : ℕ) :
  is_chessboard_8x8 x y →
  let a := sum_of_squares x y is_black_square
  let b := sum_of_squares x y is_white_square
  a = b :=
by
  intros hxy a b
  sorry

end chessboard_distance_sum_equals_l766_766047


namespace simplify_radical_expression_l766_766361

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l766_766361


namespace group_scores_analysis_l766_766089

def group1_scores : List ℕ := [92, 90, 91, 96, 96]
def group2_scores : List ℕ := [92, 96, 90, 95, 92]

def median (l : List ℕ) : ℕ := sorry
def mode (l : List ℕ) : ℕ := sorry
def mean (l : List ℕ) : ℕ := sorry
def variance (l : List ℕ) : ℕ := sorry

theorem group_scores_analysis :
  median group2_scores = 92 ∧
  mode group1_scores = 96 ∧
  mean group2_scores = 93 ∧
  variance group1_scores = 64 / 10 ∧
  variance group2_scores = 48 / 10 ∧
  variance group2_scores < variance group1_scores :=
by
  sorry

end group_scores_analysis_l766_766089


namespace cost_of_meal_l766_766121

noncomputable def total_cost (hamburger_cost fry_cost drink_cost : ℕ) (num_hamburgers num_fries num_drinks : ℕ) (discount_rate : ℕ) : ℕ :=
  let initial_cost := (hamburger_cost * num_hamburgers) + (fry_cost * num_fries) + (drink_cost * num_drinks)
  let discount := initial_cost * discount_rate / 100
  initial_cost - discount

theorem cost_of_meal :
  total_cost 5 3 2 3 4 6 10 = 35 := by
  sorry

end cost_of_meal_l766_766121


namespace divisors_comparison_l766_766719

noncomputable def divisors_ending_with_1_or_9 (n : ℕ) : Finset ℕ :=
  (Finset.filter (λ d : ℕ, d ∣ n ∧ (d % 10 = 1 ∨ d % 10 = 9)) (Finset.range (n + 1)))

noncomputable def divisors_ending_with_3_or_7 (n : ℕ) : Finset ℕ :=
  (Finset.filter (λ d : ℕ, d ∣ n ∧ (d % 10 = 3 ∨ d % 10 = 7)) (Finset.range (n + 1)))

theorem divisors_comparison (n : ℕ) (hn : 0 < n) :
  (divisors_ending_with_1_or_9 n).card ≥ (divisors_ending_with_3_or_7 n).card := 
sorry

end divisors_comparison_l766_766719


namespace expression_equivalence_l766_766067

-- Define the initial expression
def expr (w : ℝ) : ℝ := 3 * w + 4 - 2 * w^2 - 5 * w - 6 + w^2 + 7 * w + 8 - 3 * w^2

-- Define the simplified expression
def simplified_expr (w : ℝ) : ℝ := 5 * w - 4 * w^2 + 6

-- Theorem stating the equivalence
theorem expression_equivalence (w : ℝ) : expr w = simplified_expr w :=
by
  -- we would normally simplify and prove here, but we state the theorem and skip the proof for now.
  sorry

end expression_equivalence_l766_766067


namespace speed_of_man_l766_766502

theorem speed_of_man (L : ℝ) (V_train_kmh : ℝ) (t : ℝ) (V_man_kmh : ℝ) :
  L = 550 → V_train_kmh = 60 → t = 29.997600191984645 → V_man_kmh ≈ 6.00228 :=
by
  intros hL hV_train ht
  -- Definition conversions
  let V_train_ms := V_train_kmh * 1000 / 3600
  let Vr := L / t
  let V_man_ms := Vr - V_train_ms
  let V_man_kmh := V_man_ms * 3600 / 1000

  -- Proof of the desired speed
  have hVr : Vr = 18.334 := by sorry
  have hV_man_ms : V_man_ms = 1.6673 := by sorry
  have hV_man_kmh : V_man_kmh = 6.00228 := by sorry

  -- Conclusion with approximation
  exact (by rwa [hV_man_kmh at V_man_kmh]).trans (by norm_num)

end speed_of_man_l766_766502


namespace simplify_expression_l766_766378

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l766_766378


namespace question_1_question_2_l766_766597

theorem question_1 (a : ℝ) : 
  (∀ (x : ℝ), f x = x ^ 2 + 2 * a * x - 3) →
  f(a + 1) - f(a) = 9 → a = 2 :=
sorry

theorem question_2 (a : ℝ) : 
  (∀ (x : ℝ), f x = x ^ 2 + 2 * a * x - 3) →
  ∃ (a : ℝ), f (a) = -4 → (a = 1 ∨ a = -1) :=
sorry

end question_1_question_2_l766_766597


namespace correct_vector_statement_l766_766447
open Set

-- Definitions and conditions
def CollinearVectors (a b : ℝ^3) : Prop := ∃ k : ℝ, b = k • a

def EqualVectors (a b : ℝ^3) : Prop := a = b

def ParallelVectors (a b : ℝ^3) : Prop := ∃ k : ℝ, b = k • a ∧ k ≠ 0

def VectorComparison (a b : ℝ^3) : Prop := (∀ k : ℝ, (a = k • b ∧ k > 1)) → False

-- Main theorem
theorem correct_vector_statement :
  let A := ∀ (a b : ℝ^3), CollinearVectors a b ↔ ParallelVectors a b
  let B := ∀ (a b : ℝ^3), ∥a∥ = ∥b∥ → ∃ c d : ℝ^3, EqualVectors c d
  let C := ∀ (a b c : ℝ^3), ParallelVectors a b ∧ ParallelVectors b c → ParallelVectors a c
  let D := ∀ (a b : ℝ^3), (ParallelVectors a b ∧ ∥a∥ > ∥b∥) → VectorComparison a b
  B ∧ ¬A ∧ ¬C ∧ ¬D := 
begin
  sorry
end

end correct_vector_statement_l766_766447


namespace max_value_of_f_l766_766147

def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x

theorem max_value_of_f : ∃ m, (∀ x, f(x) ≤ m) ∧ m = 5 :=
by
  sorry

end max_value_of_f_l766_766147


namespace find_width_of_lawn_l766_766101

noncomputable def width_of_lawn
    (length : ℕ)
    (cost : ℕ)
    (cost_per_sq_m : ℕ)
    (road_width : ℕ) : ℕ :=
  let total_area := cost / cost_per_sq_m
  let road_area_length := road_width * length
  let eq_area := total_area - road_area_length
  eq_area / road_width

theorem find_width_of_lawn :
  width_of_lawn 110 4800 3 10 = 50 :=
by
  sorry

end find_width_of_lawn_l766_766101


namespace profit_calculation_more_profitable_method_l766_766865

def profit_end_of_month (x : ℝ) : ℝ :=
  0.3 * x - 900

def profit_beginning_of_month (x : ℝ) : ℝ :=
  0.26 * x

theorem profit_calculation (x : ℝ) (h₁ : profit_end_of_month x = 0.3 * x - 900)
  (h₂ : profit_beginning_of_month x = 0.26 * x) :
  profit_end_of_month x = 0.3 * x - 900 ∧ profit_beginning_of_month x = 0.26 * x :=
by 
  sorry

theorem more_profitable_method (x : ℝ) (hx : x = 20000)
  (h_beg : profit_beginning_of_month x = 0.26 * x)
  (h_end : profit_end_of_month x = 0.3 * x - 900) :
  profit_beginning_of_month x > profit_end_of_month x ∧ profit_beginning_of_month x = 5200 :=
by 
  sorry

end profit_calculation_more_profitable_method_l766_766865


namespace range_of_sum_l766_766228

theorem range_of_sum {x y : ℝ} (h : 2^x + 2^y = 1) : x + y ∈ set.Iic (-2) :=
by 
  sorry

end range_of_sum_l766_766228


namespace odd_prime_exists_nat_l766_766930

-- Define the conditions
def nat_sqrt (x : ℕ) : ℕ := (Int.rsqrt x).natAbs

-- Define the main theorem
theorem odd_prime_exists_nat (p : ℕ) (hp : Prime p) :
  (∃ m : ℕ, nat_sqrt m + nat_sqrt (m + p) ∈ ℕ) ↔ (p % 2 = 1) :=
begin
  sorry
end

end odd_prime_exists_nat_l766_766930


namespace car_distance_problem_l766_766042

theorem car_distance_problem
  (d y z r : ℝ)
  (initial_distance : d = 113)
  (right_turn_distance : y = 15)
  (second_car_distance : z = 35)
  (remaining_distance : r = 28)
  (x : ℝ) :
  2 * x + z + y + r = d → 
  x = 17.5 :=
by
  intros h
  sorry  

end car_distance_problem_l766_766042


namespace large_lemonhead_doll_cost_l766_766079

variable (L : ℝ)

def cost_of_large_lemonhead_doll (L : ℝ) : Prop :=
  ∃ (L : ℝ), 
    (L > 0) ∧ 
    (350 / (L - 2) = 350 / L + 20) ∧ 
    L = 7

theorem large_lemonhead_doll_cost : cost_of_large_lemonhead_doll 7 :=
by
  unfold cost_of_large_lemonhead_doll
  use 7
  split
  { linarith }
  split
  { apply sorry } 
  { refl }

end large_lemonhead_doll_cost_l766_766079


namespace greatest_difference_units_digit_l766_766770

/-- 
The three-digit integer of the form 72X is a multiple of 4.
Prove that the greatest possible difference between two of the possibilities for the units digit is 8
-/
theorem greatest_difference_units_digit : 
  ∃ (n m : ℕ), (720 ≤ n) ∧ (n ≤ 729) ∧ ((n % 4 = 0) ∧ m = n % 10) → (max m 0 - 0 = 8) := 
begin
  sorry,
end

end greatest_difference_units_digit_l766_766770


namespace range_of_m_l766_766225

variable {x y m : ℝ}

theorem range_of_m (hx : 0 < x) (hy : 0 < y) (h_eq : 1/x + 4/y = 1) (h_ineq : ∃ x y, x + y/4 < m^2 - 3*m) : m < -1 ∨ m > 4 :=
sorry

end range_of_m_l766_766225


namespace marion_score_correct_l766_766757

-- Definitions based on conditions
def total_items : ℕ := 40
def ella_incorrect : ℕ := 4
def ella_correct : ℕ := total_items - ella_incorrect
def marion_score : ℕ := (ella_correct / 2) + 6

-- Statement of the theorem
theorem marion_score_correct : marion_score = 24 :=
by
  -- proof goes here
  sorry

end marion_score_correct_l766_766757


namespace number_of_outfits_l766_766727

def red_shirts : ℕ := 6
def green_shirts : ℕ := 7
def number_pants : ℕ := 9
def blue_hats : ℕ := 10
def red_hats : ℕ := 10

theorem number_of_outfits :
  (red_shirts * number_pants * blue_hats) + (green_shirts * number_pants * red_hats) = 1170 :=
by
  sorry

end number_of_outfits_l766_766727


namespace least_natural_addend_to_palindrome_l766_766090

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

theorem least_natural_addend_to_palindrome :
  ∃ (n : ℕ), n = 100 ∧ is_palindrome (52425 + n) :=
by
  use 100
  split
  · rfl
  · sorry

end least_natural_addend_to_palindrome_l766_766090


namespace largest_perfect_square_factor_of_3402_l766_766058

theorem largest_perfect_square_factor_of_3402 :
  ∃ (n : ℕ), n^2 ∣ 3402 ∧ (∀ m : ℕ, m^2 ∣ 3402 → m^2 ≤ n^2) :=
begin
  use 3, -- n is 3
  split,
  {
    norm_num,
    rw [mul_comm, nat.dvd_prime_pow (dec_trivial : prime 3)] ; [norm_num, dec_trivial],
  },
  {
    intros m h,
    have h_dvds, from nat.prime_dvd_prime_pow (dec_trivial : prime 3) h,
    cases h_dvds,
    {
      exact h_dvds.symm ▸ nat.le_refl _,
    },
    {
      suffices : 1 ≤ 3, from h_dvds.symm ▸ nat.pow_le_pow_of_le_left this 2,
      norm_num,
    }
  }
end

end largest_perfect_square_factor_of_3402_l766_766058


namespace problem_equivalent_l766_766230

theorem problem_equivalent :
  500 * 2019 * 0.0505 * 20 = 2019^2 :=
by
  sorry

end problem_equivalent_l766_766230


namespace inverse_function_log_base_two_l766_766205

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_log_base_two (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : f a (a^2) = a) : f a = fun x => Real.log x / Real.log 2 := 
by
  sorry

end inverse_function_log_base_two_l766_766205


namespace cube_root_of_8_is_2_l766_766734

theorem cube_root_of_8_is_2 : (∛8 : ℝ) = 2 :=
by
  sorry

end cube_root_of_8_is_2_l766_766734


namespace ten_pow_necessary_but_not_sufficient_l766_766462

theorem ten_pow_necessary_but_not_sufficient (a b : ℝ) (h1 : 10^a > 10^b) (h2 : log 10 (10^a) > log 10 (10^b)) : 
  (a > b) ∧ (¬ (∀ (x : ℝ), (10 ^ x > 10 ^ b) → (log 10 (10 ^ x) > log 10 (10 ^ b)))) :=
by
  sorry

end ten_pow_necessary_but_not_sufficient_l766_766462


namespace limit_v_n_l766_766004

-- Define the sequence (u_n)
def u_n : ℕ → ℝ 
| 0       := 1
| 1       := 2
| (n + 2) := 3 * u_n (n+1) - u_n n

-- Define the sequence (v_n) as the sum of arccot of (u_n)
def v_n (n : ℕ) := ∑ k in Finset.range (n + 1), Real.arccot (u_n k)

-- Define the limit of (v_n) as n approaches infinity to prove the answer
theorem limit_v_n : tendsto (λ n, v_n n) at_top (𝓝 (- (Real.pi / 2))) :=
sorry

end limit_v_n_l766_766004


namespace shorten_by_one_expected_length_l766_766853

-- Definition of the random sequence and binomial distribution properties
def digit_sequence {n : ℕ} (p : ℝ) (len : ℕ) : ℝ := 
  let q := 1 - p in
  (len.choose 1) * (p^1 * q^(len - 1))

-- Part (a) statement
theorem shorten_by_one 
  (len : ℕ) 
  (p : ℝ := 0.1) 
  (q := 1 - p) 
  (x := len - 1) : 
  digit_sequence p x = 1.564e-90 :=
by sorry

-- Part (b) statement
theorem expected_length 
  (initial_len : ℕ) 
  (p : ℝ := 0.1) (removals := initial_len - 1) 
  (expected_removed : ℝ := removals * p) : 
  (initial_len : ℝ) - expected_removed = 1813.6 :=
by sorry

end shorten_by_one_expected_length_l766_766853


namespace locus_of_midpoint_l766_766972

theorem locus_of_midpoint {P Q M : ℝ × ℝ} (hP_on_circle : P.1^2 + P.2^2 = 13)
  (hQ_perpendicular_to_y_axis : Q.1 = P.1) (h_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1^2 / (13 / 4)) + (M.2^2 / 13) = 1 := 
sorry

end locus_of_midpoint_l766_766972


namespace simplify_expr_l766_766351

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l766_766351


namespace min_ratio_sum_four_l766_766875

def least_ratio_sum (areas_red : List (ℝ × ℝ)) (areas_blue : List (ℝ × ℝ)) : ℝ :=
  (areas_red.map (λ r, r.2 / r.1)).sum + (areas_blue.map (λ b, b.1 / b.2)).sum

theorem min_ratio_sum_four
  (areas_red areas_blue : List (ℝ × ℝ))
  (sum_areas_red : (List.map (λ r, r.1 * r.2) areas_red).sum = 1 / 2)
  (sum_areas_blue : (List.map (λ b, b.1 * b.2) areas_blue).sum = 1 / 2) :
  least_ratio_sum areas_red areas_blue ≥ 4 := 
sorry

end min_ratio_sum_four_l766_766875


namespace parabola_equation_l766_766643

theorem parabola_equation (h k a : ℝ) (same_shape : ∀ x, -2 * x^2 + 2 = a * x^2 + k) (vertex : h = 4 ∧ k = -2) :
  ∀ x, -2 * (x - 4)^2 - 2 = a * (x - h)^2 + k :=
by
  -- This is where the actual proof would go
  simp
  sorry

end parabola_equation_l766_766643


namespace find_all_complex_numbers_l766_766928

noncomputable def complex_solution (z : ℂ) : Prop :=
  z^2 = -45 + 28 * complex.I

theorem find_all_complex_numbers (z : ℂ) : 
  complex_solution z ↔ z = 2 + 7 * complex.I ∨ z = -2 - 7 * complex.I := 
by
  sorry

end find_all_complex_numbers_l766_766928


namespace correct_mean_of_values_l766_766749

theorem correct_mean_of_values :
  ∀ (incorrect_mean : ℝ) (n: ℕ) (wrong_vals correct_vals : list ℝ),
    incorrect_mean = 170 →
    n = 30 →
    wrong_vals = [150, 195, 160] →
    correct_vals = [190, 200, 175] →
    (let incorrect_sum := incorrect_mean * n,
         sum_wrong_vals := list.sum wrong_vals,
         sum_correct_vals := list.sum correct_vals,
         correct_sum := incorrect_sum - sum_wrong_vals + sum_correct_vals,
         correct_mean := correct_sum / n in correct_mean = 172) :=
begin
  intros,
  sorry
end

end correct_mean_of_values_l766_766749


namespace labor_productivity_increase_l766_766750

noncomputable def regression_equation (x : ℝ) : ℝ := 50 + 60 * x

theorem labor_productivity_increase (Δx : ℝ) (hx : Δx = 1) :
  regression_equation (x + Δx) - regression_equation x = 60 :=
by
  sorry

end labor_productivity_increase_l766_766750


namespace sequence_general_formula_function_range_l766_766184
open Real

noncomputable def seq (n : ℕ) : ℝ :=
  if (n % 2 = 0) then 2 ^ (n / 2) else 2 ^ ((n - 1) / 2)

theorem sequence_general_formula :
  ∀ n : ℕ, seq n = if (n % 2 = 0) then 2 ^ (n / 2) else 2 ^ ((n - 1) / 2) :=
by sorry

theorem function_range (A φ : ℝ) (hA : A = seq 4 + 1) (h_max : A * sin (2 * (π / 6) + φ) = A)
  : 0 < φ ∧ φ < π → (∀ x ∈ Icc (-π / 12) (π / 2), A * sin (2 * x + φ) ∈ Icc (-5 / 2) 5) :=
by sorry

end sequence_general_formula_function_range_l766_766184


namespace simplify_expr_l766_766349

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l766_766349


namespace roots_of_quadratic_l766_766010

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic (m : ℝ) :
  let a := 1
  let b := (3 * m - 1)
  let c := (2 * m^2 - m)
  discriminant a b c ≥ 0 :=
by
  sorry

end roots_of_quadratic_l766_766010


namespace percentage_books_not_sold_l766_766826

theorem percentage_books_not_sold :
  let S := 900
  let M := 75
  let T := 50
  let W := 64
  let Th := 78
  let F := 135
  let TS := M + T + W + Th + F
  let BNS := S - TS
  let PNS := (BNS / S) * 100
  PNS = 55.33 := by
{
  let S := 900
  let M := 75
  let T := 50
  let W := 64
  let Th := 78
  let F := 135
  let TS := M + T + W + Th + F
  let BNS := S - TS
  let PNS := (BNS / S) * 100
  have TS_calc : TS = 402 := by norm_num
  have BNS_calc : BNS = 498 := by norm_num
  have PNS_calc : PNS = 55.33 := by norm_num
  exact PNS_calc
}

end percentage_books_not_sold_l766_766826


namespace f_at_neg_one_l766_766953

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + 3 * x + 16

noncomputable def f_with_r (x : ℝ) (a r : ℝ) : ℝ := (x^3 + a * x^2 + 3 * x + 16) * (x - r)

theorem f_at_neg_one (a b c r : ℝ) (h1 : ∀ x, g x a = 0 → f_with_r x a r = 0)
  (h2 : a - r = 5) (h3 : 16 - 3 * r = 150) (h4 : -16 * r = c) :
  f_with_r (-1) a r = -1347 :=
by
  sorry

end f_at_neg_one_l766_766953


namespace simplify_expression_l766_766321

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l766_766321


namespace area_F1PF2_eq_4sqrt3_l766_766580

instance : IsEllipsoid ℝ (ellipse_equation : x^2 / 5 + y^2 = 1)

def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)
def point_C : ℝ × ℝ := (2, 3)
def angle_F1PF2 := 60 * Real.pi / 180 -- converting degrees to radians

noncomputable def area_triangle_F1_P_F2 : ℝ :=
  1/2 * 16 * Real.sin (angle_F1PF2)

theorem area_F1PF2_eq_4sqrt3 : 
  ∀ (C P : ℝ × ℝ) 
  (H1 : ∃ curve C, OnEllipse C (y = x^2 / 5 + 1)) 
  (H2 : point_C ∈ C) 
  (H3 : P ∈ C) 
  (H4 : anglereal F1 P F2 = angle_F1PF2), 
  area_triangle_F1_P_F2 = 4 * Real.sqrt 3 :=
sorry

end area_F1PF2_eq_4sqrt3_l766_766580


namespace compute_expression_l766_766897

-- Defining notation for the problem expression and answer simplification
theorem compute_expression : 
    9 * (2/3)^4 = 16/9 := by 
  sorry

end compute_expression_l766_766897


namespace minimal_rooms_l766_766312

-- Definitions
def numTourists := 100

def roomsAvailable (n k : Nat) : Prop :=
  ∀ k_even : k % 2 = 0, 
    ∃ m : Nat, k = 2 * m ∧ n = 100 * (m + 1) ∨
    ∀ k_odd : k % 2 = 1, k = 2 * m + 1 ∧ n = 100 * (m + 1) + 1

-- Proof statement
theorem minimal_rooms (k n : Nat) : roomsAvailable n k :=
by 
  -- The proof is provided in the solution steps
  sorry

end minimal_rooms_l766_766312


namespace cycling_speed_l766_766003

-- Definitions for the conditions given
def length_to_breadth_ratio (L B : ℝ) : Prop := L / B = 1 / 3
def area_of_park (L B : ℝ) : Prop := L * B = 120000
def cycling_time (perimeter : ℝ) : Prop := perimeter / 1600 * 8 = perimeter

-- Derived Definitions
def perimeter_of_park (L B : ℝ) : ℝ := 2 * L + 2 * B
def speed (perimeter : ℝ) (time : ℝ) : ℝ := (perimeter / time) * (60 / 1000)

-- Proof statement
theorem cycling_speed (L B : ℝ) (h1 : length_to_breadth_ratio L B) (h2 : area_of_park L B) (h3 : cycling_time (perimeter_of_park L B)) : speed (perimeter_of_park L B) 8 = 10 / 3 :=
by sorry

end cycling_speed_l766_766003


namespace determine_function_l766_766218

noncomputable def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem determine_function :
  (ω > 0) →
  (-π / 2 ≤ φ ∧ φ ≤ π / 2) →
  ∀ (x1 x0 : ℝ), (f x1 - f x0 = 2√2) →
  ∀ (x y : ℝ), ((x, y) = (2, -1/2)) →
  f x = Real.sin (π / 2 * x + π / 6) :=
by
  intros h_omega h_phi h_peak_distance h_point
  sorry

end determine_function_l766_766218


namespace probability_of_shortening_exactly_one_digit_l766_766839

theorem probability_of_shortening_exactly_one_digit :
  let n := 2014 in
  let p := 0.1 in
  let q := 0.9 in
  (n.choose 1) * p ^ 1 * q ^ (n - 1) = 201.4 * 1.564 * 10^(-90) :=
by sorry

end probability_of_shortening_exactly_one_digit_l766_766839


namespace volume_increase_152_l766_766406

/--
If the length, width, and height of a rectangular prism are increased by 20%, 40%, and 50% respectively, then the percentage increase in its volume is 152%.
-/
theorem volume_increase_152
  (L W H : ℝ)
  (hL : 0 < L)
  (hW : 0 < W)
  (hH : 0 < H) :
  let V_original := L * W * H,
      L_new := 1.20 * L,
      W_new := 1.40 * W,
      H_new := 1.50 * H,
      V_new := L_new * W_new * H_new,
      percentage_increase := (V_new - V_original) / V_original * 100 :=
  percentage_increase = 152 :=
by {
  let V_original := L * W * H,
  let L_new := 1.20 * L,
  let W_new := 1.40 * W,
  let H_new := 1.50 * H,
  let V_new := L_new * W_new * H_new,
  let percentage_increase := (V_new - V_original) / V_original * 100,
  have : V_original > 0, from mul_pos (mul_pos hL hW) hH,
  calc
    percentage_increase
        = ((1.20 * 1.40 * 1.50 * V_original) - V_original) / V_original * 100 : by sorry
    ... = (2.52 * V_original - V_original) / V_original * 100 : by sorry
    ... = (1.52 * V_original) / V_original *100 : by sorry
    ... = 1.52 * 100 : by sorry
    ... = 152 : by sorry
}

end volume_increase_152_l766_766406


namespace compute_fraction_power_l766_766900

theorem compute_fraction_power :
  9 * (2/3)^4 = 16/9 := 
by
  sorry

end compute_fraction_power_l766_766900


namespace vector_a_perp_vector_b_implies_magnitude_l766_766617

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (3, m)
def vector_b : ℝ × ℝ := (1, -3)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

theorem vector_a_perp_vector_b_implies_magnitude :
  ∀ (m : ℝ), dot_product (vector_a m) vector_b = 0 → magnitude (vector_a m) = Real.sqrt 10 :=
by
  intros m h
  sorry

end vector_a_perp_vector_b_implies_magnitude_l766_766617


namespace number_of_methods_to_finish_testing_in_3_tries_l766_766776

theorem number_of_methods_to_finish_testing_in_3_tries :
  let total_products := 10
  let unstable_products := 2
  let stable_products := total_products - unstable_products
  -- Number of ways the testing finishes exactly in 3 tries
  nat.choose(unstable_products, 1) * stable_products * nat.choose(stable_products - 1, 1) * nat.factorial(2) = 32 :=
by
  sorry

end number_of_methods_to_finish_testing_in_3_tries_l766_766776


namespace vector_dot_product_l766_766792

variables {V : Type} [inner_product_space ℝ V] (a b : V)

theorem vector_dot_product
  (h1 : ∥a - 2 • b∥ = 1)
  (h2 : ∥2 • a + 3 • b∥ = 2) :
  (5 • a - 3 • b) ⬝ (a - 9 • b) = 5 :=
sorry

end vector_dot_product_l766_766792


namespace decision_making_system_reliability_l766_766394

theorem decision_making_system_reliability (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  (10 * p^3 - 15 * p^4 + 6 * p^5 > 3 * p^2 - 2 * p^3) -> (1 / 2 < p) ∧ (p < 1) :=
by
  sorry

end decision_making_system_reliability_l766_766394


namespace max_y_value_l766_766937

theorem max_y_value : 
  ∃ x, x ∈ Icc (-2 * Real.pi / 3) (-Real.pi / 2) ∧ 
  y (x) = tan (x + 3 * Real.pi / 4) - tan (x + Real.pi / 4) + sin (x + Real.pi / 4) ∧ 
  ∀ x ∈ Icc (-2 * Real.pi / 3) (-Real.pi / 2), 
    tan (x + 3 * Real.pi / 4) - tan (x + Real.pi / 4) + sin (x + Real.pi / 4) ≤ y (x) := 
exists_intro (-Real.pi / 2)
(sorry)

end max_y_value_l766_766937


namespace find_total_area_l766_766667

noncomputable def side_length_of_square (area : ℝ) := real.sqrt area

def is_midpoint (H E F : ℝ → ℝ → Prop) : Prop :=
∃ x y, H x y ∧ ∃ x1 y1 x2 y2, E x1 y1 ∧ F x2 y2 ∧ (x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2)

def is_one_third_along (H B C : ℝ → ℝ → Prop) : Prop :=
∃ x y, H x y ∧ ∃ x1 y1 x2 y2, B x1 y1 ∧ C x2 y2 ∧ (x = x1 + (x2 - x1) / 3 ∧ y = y1 + (y2 - y1) / 3)

theorem find_total_area (A B C D E F G H : ℝ → ℝ → Prop)
  (h_square1 : ∃ p1 p2 p3 p4, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p4 ≠ p1 ∧ side_length_of_square 25)
  (h_square2 : ∃ p1 p2 p3 p4, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p4 ≠ p1 ∧ side_length_of_square 25)
  (h_midpoint : is_midpoint H E F)
  (h_one_third : is_one_third_along H B C) :
  polygon_area A B H F G D = 27.09 :=
sorry

end find_total_area_l766_766667


namespace pieces_per_block_is_32_l766_766871

-- Define the number of pieces of junk mail given to each house
def pieces_per_house : ℕ := 8

-- Define the number of houses in each block
def houses_per_block : ℕ := 4

-- Calculate the total number of pieces of junk mail given to each block
def total_pieces_per_block : ℕ := pieces_per_house * houses_per_block

-- Prove that the total number of pieces of junk mail given to each block is 32
theorem pieces_per_block_is_32 : total_pieces_per_block = 32 := 
by sorry

end pieces_per_block_is_32_l766_766871


namespace parabola_properties_and_slope_product_l766_766608

-- Define the given conditions
def parabola_passing_through (p : ℝ) (P : ℝ × ℝ) : Prop :=
  P.snd ^ 2 = 2 * p * P.fst

def line_intersects_parabola (m : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A.snd^2 = 2 * A.fst ∧ B.snd^2 = 2 * B.fst ∧ (A ≠ B ∧ m * A.snd + 2 = A.fst ∧ m * B.snd + 2 = B.fst)

-- Define the proof problem
theorem parabola_properties_and_slope_product :
  ∀ (p : ℝ) (Q M O A B : ℝ × ℝ),
    parabola_passing_through p Q ∧ Q = (2, 2) ∧ p > 0 ∧
    M = (2, 0) ∧ O = (0, 0) ∧ 
    line_intersects_parabola ((A.snd - 0) / (A.fst - 0)) M ∧ 
    line_intersects_parabola ((B.snd - 0) / (B.fst - 0)) M →
    (∃ (std_eq directrix : ℝ → ℝ), 
       std_eq = (λ y, 2 * y^2) ∧ 
       directrix = (λ x, - (1/2) * x) ∧ 
       ((A.snd / A.fst) * (B.snd / B.fst)) = -1) :=
by
  sorry

end parabola_properties_and_slope_product_l766_766608


namespace max_wickets_in_innings_l766_766451

-- Define the max wickets a bowler can take per over
def max_wickets_per_over : ℕ := 3

-- Define the number of overs bowled by the bowler
def overs_bowled : ℕ := 6

-- Assume the total players in a cricket team
def total_players : ℕ := 11

-- Lean statement that proves the maximum number of wickets the bowler can take in an innings
theorem max_wickets_in_innings :
  3 * 6 ≥ total_players - 1 →
  max_wickets_per_over * overs_bowled ≥ total_players - 1 :=
by
  sorry

end max_wickets_in_innings_l766_766451


namespace find_angle_ACB_l766_766264

-- Definitions corresponding to the conditions
def angleABD : ℝ := 145
def angleBAC : ℝ := 105
def supplementary (a b : ℝ) : Prop := a + b = 180
def triangleAngleSum (a b c : ℝ) : Prop := a + b + c = 180

theorem find_angle_ACB :
  ∃ (angleACB : ℝ), 
    supplementary angleABD angleABC ∧
    triangleAngleSum angleBAC angleABC angleACB ∧
    angleACB = 40 := 
sorry

end find_angle_ACB_l766_766264


namespace area_ADP_product_of_areas_is_perfect_square_l766_766459

variables {S_ABP S_BCP S_CDP S_ADP : ℝ}

/-- Given a convex quadrilateral ABCD with diagonals intersecting at point P and the areas of triangles
    ABP, BCP, and CDP, find the area of triangle ADP. -/
def find_area_ADP (S_ABP S_BCP S_CDP : ℝ) : ℝ :=
  S_ABP * S_CDP / S_BCP

-- To state the equivalence as a theorem (proof not included):
theorem area_ADP (S_ABP S_BCP S_CDP : ℝ) :
  S_ADP = S_ABP * S_CDP / S_BCP :=
sorry


/-- Show that if the areas of the four triangles formed by the diagonals of a convex quadrilateral 
    are integers, the product of these areas is a perfect square. -/
theorem product_of_areas_is_perfect_square 
  (S_ABP S_BCP S_CDP S_ADP : ℕ) (h : S_ADP * S_BCP = S_ABP * S_CDP) :
  ∃ k : ℕ, S_ABP * S_BCP * S_CDP * S_ADP = k^2 :=
sorry

end area_ADP_product_of_areas_is_perfect_square_l766_766459


namespace slope_of_tangent_line_at_point_l766_766219

noncomputable def f (x : ℝ) := g x + x^2

-- Assume g is differentiable at x = 1
variable (g : ℝ → ℝ) (dg1 : deriv g 1 = 2)

theorem slope_of_tangent_line_at_point :
  deriv f 1 = 4 :=
by
  -- Use the fact that f(x) = g(x) + x^2
  have hf : ∀ x, f x = g x + x^2 := λ x, rfl
  
  -- Differentiate f(x)
  rw [hf]
  rw deriv_add
  rw deriv_id'
  rw deriv_const

  -- Given that g'(1) = 2
  have dg1 : deriv g 1 = 2 := by assumption

  -- Compute the slope of the tangent line
  rw dg1
  norm_num
  sorry

end slope_of_tangent_line_at_point_l766_766219


namespace g_monotonically_increasing_interval_l766_766512

-- Definitions as per the problem conditions
def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + π / 3)

-- Translation of f(x) to obtain g(x)
def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - π / 3)

-- Statement to prove the interval where g(x) is monotonically increasing
theorem g_monotonically_increasing_interval (k : ℤ) : 
  ∀ x : ℝ, 
  (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) ↔ 
  (g' x > 0) := 
sorry

end g_monotonically_increasing_interval_l766_766512


namespace number_of_ensembles_sold_l766_766307

-- Define the prices
def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earring_price : ℕ := 10
def ensemble_price : ℕ := 45

-- Define the quantities sold
def necklaces_sold : ℕ := 5
def bracelets_sold : ℕ := 10
def earrings_sold : ℕ := 20

-- Define the total income
def total_income : ℕ := 565

-- Define the function or theorem that determines the number of ensembles sold
theorem number_of_ensembles_sold : 
  (total_income = (necklaces_sold * necklace_price) + (bracelets_sold * bracelet_price) + (earrings_sold * earring_price) + (2 * ensemble_price)) :=
sorry

end number_of_ensembles_sold_l766_766307


namespace triangular_weight_60_l766_766008

def round_weight := ℝ
def triangular_weight := ℝ
def rectangular_weight := 90

variables (c t : ℝ)

-- Conditions
axiom condition1 : c + t = 3 * c
axiom condition2 : 4 * c + t = t + c + rectangular_weight

theorem triangular_weight_60 : t = 60 :=
  sorry

end triangular_weight_60_l766_766008


namespace pitch_required_to_finish_road_l766_766874

theorem pitch_required_to_finish_road :
  ∀ (total_road_length first_day_miles second_day_miles third_day_miles: ℕ)
    (first_day_truckloads_per_mile second_day_truckloads_per_mile 
     third_day_truckloads_per_mile fourth_day_truckload : ℕ)
    (bags_of_gravel_per_truckload : ℕ) (bags_to_pitch_ratio : ℝ),
  total_road_length = 20 →
  first_day_miles = 4 →
  first_day_truckloads_per_mile = 3 →
  second_day_truckloads_per_mile = 4 →
  second_day_miles = 7 →
  third_day_truckloads_per_mile = 2 →
  third_day_miles = 5 →
  fourth_day_truckload = 1 →
  bags_of_gravel_per_truckload = 2 →
  bags_to_pitch_ratio = 5 →
  let remaining_miles := total_road_length - (first_day_miles + second_day_miles + third_day_miles) in
  let total_truckloads := remaining_miles * fourth_day_truckload in
  let barrels_per_truckload := (bags_of_gravel_per_truckload : ℝ) / (bags_to_pitch_ratio) in
  ⌈total_truckloads * barrels_per_truckload⌉ = 2 :=
by
  sorry

end pitch_required_to_finish_road_l766_766874


namespace simplify_radical_expression_l766_766364

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l766_766364


namespace find_value_mn_l766_766638

-- Defining the conditions
variables (m n : ℝ)
hypothesis (hm : m^2 - 2 * m = 1)
hypothesis (hn : n^2 - 2 * n = 1)
hypothesis (h_neq : m ≠ n)

-- Statement to be proved
theorem find_value_mn : (m + n) - (m * n) = 3 :=
by
  sorry

end find_value_mn_l766_766638


namespace sum_of_squares_eight_l766_766015

theorem sum_of_squares_eight (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 28) : x^2 + y^2 = 8 := 
  sorry

end sum_of_squares_eight_l766_766015


namespace sin_of_tan_l766_766994

variable (a θ : ℝ)

theorem sin_of_tan (h1 : a ≠ 0) (h2 : tan θ = -a) : sin θ = -√2 / 2 :=
sorry

end sin_of_tan_l766_766994


namespace parabola_focus_distance_l766_766590

theorem parabola_focus_distance (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1) (h_dist_y_axis : |P.1| = 4) : 
  dist P (4, 0) = 5 :=
sorry

end parabola_focus_distance_l766_766590


namespace second_largest_div_smallest_l766_766782

noncomputable def second_largest (a b c : ℕ) : ℕ :=
  if a ≤ b ∧ b ≤ c then b
  else if a ≤ c ∧ c ≤ b then c
  else a

noncomputable def smallest (a b c : ℕ) : ℕ :=
  if a ≤ b ∧ a ≤ c then a
  else if b ≤ a ∧ b ≤ c then b
  else c

theorem second_largest_div_smallest (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) : 
  (second_largest a b c) / (smallest a b c) = 1.1 :=
by {
  sorry
}

end second_largest_div_smallest_l766_766782


namespace last_integer_in_sequence_l766_766761

theorem last_integer_in_sequence : ∀ (n : ℕ), n = 1000000 → (∀ k : ℕ, n = k * 3 → k * 3 < n) → n = 1000000 :=
by
  intro n hn hseq
  have h := hseq 333333 sorry
  exact hn

end last_integer_in_sequence_l766_766761


namespace area_of_triangle_right_l766_766670

variable {A B C D : Type}
variable [LinearOrderedField A]

theorem area_of_triangle_right (hRT : 0 < A) (h_right_triangle : is_right_triangle ABC) (h_AB : B = 13) (h_BD : D = 12) :
    area_of_triangle ABC = 202.8 :=
sorry

end area_of_triangle_right_l766_766670


namespace orthogonal_k_l766_766176

open BigOperators

noncomputable def a : ℝ × ℝ × ℝ := (1, 1, 1)
noncomputable def b : ℝ × ℝ × ℝ := (1, 2, 3)
def k := -2

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem orthogonal_k :
  (dot_product (k • a + b) a = 0) :=
by
  sorry

end orthogonal_k_l766_766176


namespace prime_triple_is_237_l766_766932

theorem prime_triple_is_237 :
  ∃ (p q r : ℕ), prime p ∧ prime q ∧ prime r ∧ (p, q, r) = (2, 3, 7) ∧ 
    ∀ (p q r : ℕ), prime p ∧ prime q ∧ prime r →
      (let n₁ := p^2 + 2 * q,
           d₁ := q + r,
           n₂ := q^2 + 9 * r,
           d₂ := r + p,
           n₃ := r^2 + 3 * p,
           d₃ := p + q in
         d₁ ∣ n₁ ∧ d₂ ∣ n₂ ∧ d₃ ∣ n₃ → (p, q, r) = (2, 3, 7)) :=
sorry

end prime_triple_is_237_l766_766932


namespace greatest_difference_units_digit_l766_766773

theorem greatest_difference_units_digit (d : Nat) 
  (h : d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h_mod4 : (72 % 100 + d) % 4 = 0) :
  ∃ a b, a ∈ {0, 4, 8} ∧ b ∈ {0, 4, 8} ∧ a ≠ b ∧ (abs (a - b) = 8) := 
begin
  -- Proof is omitted, only statement is required
  sorry
end

end greatest_difference_units_digit_l766_766773


namespace symmetry_in_mathematics_l766_766025

-- Define the options
def optionA := "summation of harmonic series from 1 to 100"
def optionB := "general quadratic equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0"
def optionC := "Law of Sines: a / sin A = b / sin B = c / sin C"
def optionD := "arithmetic operation: 123456789 * 9 + 10 = 1111111111"

-- Define the symmetry property
def exhibits_symmetry (option: String) : Prop :=
  option = optionC

-- The theorem to prove
theorem symmetry_in_mathematics : ∃ option, exhibits_symmetry option := by
  use optionC
  sorry

end symmetry_in_mathematics_l766_766025


namespace coefficient_x3_in_binomial_expansion_l766_766732

theorem coefficient_x3_in_binomial_expansion :
  (fin 3).choose 3 = 4 :=
by
  sorry

end coefficient_x3_in_binomial_expansion_l766_766732


namespace parabola_properties_l766_766977

-- Definition of the parabola and conditions
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {P | P.2^2 = 2 * p * P.1}

-- Point A inside the parabola and the focus F
def point_A : ℝ × ℝ := (3, 2)
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- The point P when |PA| + |PF| is minimized
def point_P : ℝ × ℝ := (2, 2)

-- The fixed point through which lines CD always passes
def fixed_point : ℝ × ℝ := (4, -2)

-- The proof problem
theorem parabola_properties (p : ℝ) (h : p > 0)
  (A : ℝ × ℝ) (hA : A = point_A)
  (F : ℝ × ℝ) (hF : F = focus p):
  p = 1 ∧
  parabola p h = {P | P.2^2 = 2 * P.1} ∧
  point_P = (2, 2) ∧
  (∀ y1 y2, let C := (y1^2 / 2, y1), D := (y2^2 / 2, y2) in
    C.1 = P.1 ∧ D.1 = P.1 → (P.2 - y1) * (P.2 - y2) + 4 = 0 → 
    ∃ k, (∀ x y, y = k * x + fixed_point.2 ↔ x = fixed_point.1))
    :=
  sorry

end parabola_properties_l766_766977


namespace simplify_sqrt_expression_l766_766341

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l766_766341


namespace locus_eq_is_ellipse_max_distance_to_C2_l766_766263

noncomputable def locus_eq (x y : ℝ) (k : ℝ) : Prop :=
  y = k * x ∧ y = -((x - 2) / k)

noncomputable def on_curve_C1 (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1 ∧ y ≠ 0

noncomputable def distance_to_line_C2 (x y : ℝ) : ℝ :=
  abs (x + y - 6) / real.sqrt 2

theorem locus_eq_is_ellipse :
  ∀ x y, 
  (∃ k, locus_eq x y k) ↔ on_curve_C1 x y := sorry

theorem max_distance_to_C2 :
  ∀ (α : ℝ), 
  (on_curve_C1 (real.cos α + 1) (real.sin α)) →
  ∃ d, d = 1 + 5 * real.sqrt 2 / 2 := sorry

end locus_eq_is_ellipse_max_distance_to_C2_l766_766263


namespace domain_of_f_l766_766903

def f (x : ℝ) : ℝ := 1 / (⌊x^2 - 7 * x + 13⌋)

theorem domain_of_f :
  (∀ x : ℝ, ⌊x^2 - 7 * x + 13⌋ > 0) →
  (set_of (λ x, 1 / (⌊x^2 - 7 * x + 13⌋) ≠ 0) = {x : ℝ | x ≤ 3 ∨ x ≥ 4}) :=
by
  sorry

end domain_of_f_l766_766903


namespace euler_formula_connected_planar_graph_l766_766701

theorem euler_formula_connected_planar_graph 
(V E F : ℕ) 
(G : Type*) [graph G] 
(h_connected : Connected G) 
(h_planar : Planar G)
(h_V : vertices G = V)
(h_E : edges G = E)
(h_F : faces G = F) : 
V - E + F = 2 := 
sorry

end euler_formula_connected_planar_graph_l766_766701


namespace simplify_expression_l766_766328

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l766_766328


namespace bus_interval_three_buses_l766_766034

theorem bus_interval_three_buses (T : ℕ) (h : T = 21) : (T * 2) / 3 = 14 :=
by
  sorry

end bus_interval_three_buses_l766_766034


namespace marble_ratio_l766_766302

theorem marble_ratio
  (L_b : ℕ) (J_y : ℕ) (A : ℕ)
  (A_b : ℕ) (A_y : ℕ) (R : ℕ)
  (h1 : L_b = 4)
  (h2 : J_y = 22)
  (h3 : A = 19)
  (h4 : A_y = J_y / 2)
  (h5 : A = A_b + A_y)
  (h6 : A_b = L_b * R) :
  R = 2 := by
  sorry

end marble_ratio_l766_766302


namespace dan_licks_l766_766914

/-- 
Given that Michael takes 63 licks, Sam takes 70 licks, David takes 70 licks, 
Lance takes 39 licks, and the average number of licks for all five people is 60, 
prove that Dan takes 58 licks to get to the center of a lollipop.
-/
theorem dan_licks (D : ℕ) 
  (M : ℕ := 63) 
  (S : ℕ := 70) 
  (Da : ℕ := 70) 
  (L : ℕ := 39)
  (avg : ℕ := 60) :
  ((M + S + Da + L + D) / 5 = avg) → D = 58 :=
by sorry

end dan_licks_l766_766914


namespace cosine_product_inequality_l766_766697

theorem cosine_product_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  8 * Real.cos A * Real.cos B * Real.cos C ≤ 1 := 
sorry

end cosine_product_inequality_l766_766697


namespace not_satisfy_rec_relation_l766_766402

def geom_seq (n : ℕ) : ℝ := 1 / (3^(n-1))
def partial_sum (n : ℕ) : ℝ := ∑ k in finset.range n, geom_seq (k + 1)

theorem not_satisfy_rec_relation (k : ℕ) :
  ¬ (partial_sum (k + 1) = 1 + (1 / 3) * partial_sum k) := 
sorry

end not_satisfy_rec_relation_l766_766402


namespace smallest_positive_z_l766_766391

theorem smallest_positive_z
  (x z : ℝ)
  (hx : cos x = 0)
  (hz : cos (x + z) = -1/2) :
  z = π / 6 :=
sorry

end smallest_positive_z_l766_766391


namespace tag_sum_is_large_l766_766516

noncomputable def tag_sum : ℝ :=
    let W : ℝ := 200
    let X : ℝ := (2/3) * W
    let Y : ℝ := W + X
    let Z : ℝ := real.sqrt Y
    let P : ℝ := X^3
    let Q : ℝ := nat.factorial W.to_nat / 100000
    W + X + Y + Z + P + Q

theorem tag_sum_is_large :
  let Q : ℝ := nat.factorial 200 / 100000 in
  tag_sum ≈ Q := by
    sorry

end tag_sum_is_large_l766_766516


namespace problem_part1_problem_part2_l766_766182

noncomputable def f_seq (a : ℝ) : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| (n + 1) := λ x, x * f_seq a n x + f_seq a n (a * x)

theorem problem_part1 (a : ℝ) (n : ℕ) : ∀ x, f_seq a n x = x^n * f_seq a n (1 / x) :=
sorry

theorem problem_part2 (a : ℝ) (n : ℕ) : 
  f_seq a n = 
    λ x, 1 + ∑ j in finset.range (n + 1), 
           ((finset.range j).prod (λ k, (a^n - a^k)) / (finset.range j).prod (λ k, a - a^k)) * x^j :=
sorry

end problem_part1_problem_part2_l766_766182


namespace continuity_of_f_at_1_l766_766717

theorem continuity_of_f_at_1 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |(-3 * x^2 - 6) - (-3 * 1^2 - 6)| < ε :=
by
  assume ε hε,
  let δ := ε / 6,
  use δ,
  split,
  linarith,
  assume x hx,
  have : |(-3 * x^2 - 6) - (-9)| = 3 * |1 - x| * |1 + x|,
  sorry

end continuity_of_f_at_1_l766_766717


namespace polynomial_has_one_positive_real_solution_in_0_2_l766_766531

noncomputable def P (x : ℝ) : ℝ := x^11 + 9 * x^10 + 20 * x^9 + 2000 * x^8 - 1500 * x^7

theorem polynomial_has_one_positive_real_solution_in_0_2 :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ P(x) = 0 :=
sorry

end polynomial_has_one_positive_real_solution_in_0_2_l766_766531


namespace band_member_earnings_l766_766165

theorem band_member_earnings :
  let attendees := 500
  let ticket_price := 30
  let band_share_percentage := 70 / 100
  let band_members := 4
  let total_earnings := attendees * ticket_price
  let band_earnings := total_earnings * band_share_percentage
  let earnings_per_member := band_earnings / band_members
  earnings_per_member = 2625 := 
by {
  sorry
}

end band_member_earnings_l766_766165


namespace simplify_expr_l766_766352

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l766_766352


namespace max_value_of_trig_function_l766_766152

theorem max_value_of_trig_function : 
  ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 5 := sorry


end max_value_of_trig_function_l766_766152


namespace trader_profit_percentage_l766_766822

theorem trader_profit_percentage (P : ℝ) (h₀ : 0 ≤ P) : 
  let discount := 0.40
  let increase := 0.80
  let purchase_price := P * (1 - discount)
  let selling_price := purchase_price * (1 + increase)
  let profit := selling_price - P
  (profit / P) * 100 = 8 := 
by
  sorry

end trader_profit_percentage_l766_766822


namespace minimum_value_of_expression_l766_766068

theorem minimum_value_of_expression (x : ℝ) (hx : x ≠ 0) : 
  (x^2 + 1 / x^2) ≥ 2 ∧ (x^2 + 1 / x^2 = 2 ↔ x = 1 ∨ x = -1) := 
by
  sorry

end minimum_value_of_expression_l766_766068


namespace units_cost_l766_766021

theorem units_cost (x y z : ℝ) 
  (h1 : 3 * x + 7 * y + z = 3.15)
  (h2 : 4 * x + 10 * y + z = 4.20) : 
  x + y + z = 1.05 :=
by 
  sorry

end units_cost_l766_766021


namespace distance_DF_eq_abs_b_c_l766_766404

variable {α : Type*} [LinearOrder α] [LinearOrderedAddCommGroup α]

namespace TriangleIncircle

-- Definitions for points, distances, and the triangle structure
structure Triangle (α : Type*) [LinearOrderedAddCommGroup α] :=
(A B C : α)
(BC AC AB : α)
(BC_eq_a : BC = a)
(AC_eq_b : AC = b)
(AB_eq_c : AB = c)

variables {ABC : Triangle α} (a b c : α) [ABC.BC_eq_a a] [ABC.AC_eq_b b] [ABC.AB_eq_c c]

-- Main theorem statement
theorem distance_DF_eq_abs_b_c : 
  ∀ (D E F : α), touches_inc (ABC.inc BC) D → diameter (ABC.inc) D E → 
  intersects_extended (AE) BC F → DF = |b - c| := 
by
  sorry

end TriangleIncircle

end distance_DF_eq_abs_b_c_l766_766404


namespace trapezium_area_l766_766824

-- Definitions based on the problem conditions
def length_side_a : ℝ := 20
def length_side_b : ℝ := 18
def distance_between_sides : ℝ := 15

-- Statement of the proof problem
theorem trapezium_area :
  (1 / 2 * (length_side_a + length_side_b) * distance_between_sides) = 285 := by
  sorry

end trapezium_area_l766_766824


namespace kabob_cubes_calculation_l766_766106

-- Define the properties of a slab of beef
def cubes_per_slab := 80
def cost_per_slab := 25

-- Define Simon's usage and expenditure
def simons_budget := 50
def number_of_kabob_sticks := 40

-- Auxiliary calculations for proofs (making noncomputable if necessary)
noncomputable def cost_per_cube := cost_per_slab / cubes_per_slab
noncomputable def cubes_per_kabob_stick := (2 * cubes_per_slab) / number_of_kabob_sticks

-- The theorem we want to prove
theorem kabob_cubes_calculation :
  cubes_per_kabob_stick = 4 := by
  sorry

end kabob_cubes_calculation_l766_766106


namespace probability_shortening_exactly_one_digit_l766_766849
open Real

theorem probability_shortening_exactly_one_digit :
  ∀ (sequence : List ℕ),
    (sequence.length = 2015) ∧
    (∀ (x ∈ sequence), x = 0 ∨ x = 9) →
      (P (λ seq, (length (shrink seq) = 2014)) sequence = 1.564e-90) := sorry

end probability_shortening_exactly_one_digit_l766_766849


namespace acme_vowel_soup_l766_766505

-- Define the set of vowels
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

-- Define the number of each vowel
def num_vowels (v : Char) : ℕ := 5

-- Define a function to count the number of five-letter words
def count_five_letter_words : ℕ :=
  (vowels.card) ^ 5

-- Theorem to be proven
theorem acme_vowel_soup :
  count_five_letter_words = 3125 :=
by
  -- Proof omitted
  sorry

end acme_vowel_soup_l766_766505


namespace intersection_complement_M_N_l766_766237

noncomputable def U : set ℕ := {n : ℕ | 0 < n}

noncomputable def M : set ℕ := {x ∈ U | ∃ k ∈ U, x ≠ 2*k - 1 }

noncomputable def N : set ℝ := {y : ℝ | ∃ x : ℝ, y = x + 4 / x ∧ (1/2 : ℝ) ≤ x ∧ x ≤ 4 }

noncomputable def complement_M_in_U : set ℕ := {x ∈ U | ∀ k ∈ U, x = 2*k - 1 }

theorem intersection_complement_M_N :
  ((complement_M_in_U : set ℝ) ∩ N) = {5, 7} :=
sorry

end intersection_complement_M_N_l766_766237


namespace line_through_vertex_parabola_count_l766_766173

theorem line_through_vertex_parabola_count :
  let vertex_x := a / 2,
      vertex_y := a^2 - a * vertex_x + (vertex_x)^2
  in ∀ a : ℝ, 
     (vertex_y == 2 * vertex_x + a) →
     a = 0 ∨ a = 8 / 3 :=
by 
  sorry

end line_through_vertex_parabola_count_l766_766173


namespace overall_loss_amount_l766_766496

noncomputable def total_worth := 19999.99
noncomputable def percent_part1 := 0.20
noncomputable def profit_percent := 0.10
noncomputable def percent_part2 := 0.80
noncomputable def loss_percent := 0.05

theorem overall_loss_amount :
  let worth_part1 := percent_part1 * total_worth,
      profit_part1 := profit_percent * worth_part1,
      worth_part2 := percent_part2 * total_worth,
      loss_part2 := loss_percent * worth_part2,
      overall_loss := loss_part2 - profit_part1
  in overall_loss = 400 :=
by
  sorry

end overall_loss_amount_l766_766496


namespace fill_pool_in_40_hours_l766_766786

noncomputable def total_minutes (hoses: Nat) (hose_rate: Nat) (partially_blocked_rate: Rat): Rat :=
  ((hoses * hose_rate) + partially_blocked_rate) * 60

noncomputable def time_to_fill (pool_volume: Nat) (total_flow_rate: Rat): Rat :=
  pool_volume / total_flow_rate

theorem fill_pool_in_40_hours :
  let pool_volume := 32000
  let hose_count := 4
  let hose_rate := 3
  let partially_blocked_rate := 1.5
  time_to_fill pool_volume (total_minutes hose_count hose_rate partially_blocked_rate) ≈ 40 := 
by
  sorry

end fill_pool_in_40_hours_l766_766786


namespace shape_2016_l766_766760

def sequence_period := ["△", "O", "口", "O"]

def nth_shape (n : ℕ) : String :=
  sequence_period[n % sequence_period.length]

theorem shape_2016 :
  nth_shape 2016 = "O" :=
by
  sorry

end shape_2016_l766_766760


namespace zero_in_interval_l766_766712

noncomputable def f : ℝ → ℝ := λ x, Real.exp x + 2 * x - 3

theorem zero_in_interval : ∃ c ∈ Ioo (1/2 : ℝ) 1, f c = 0 :=
by 
  sorry

end zero_in_interval_l766_766712


namespace orthocenter_is_circumcenter_of_APQ_l766_766297

variables (A B C H P Q : Type)
variables [acute_triangle : acute_angled_triangle ABC] [orthocenter H ABC]
variables [circumcircle_intersect_1 : circumcircle_of_triangle_intersects BHC AC P]
variables [circumcircle_intersect_2 : circumcircle_of_triangle_intersects BHC AB Q]

theorem orthocenter_is_circumcenter_of_APQ :
  circumcenter_of_triangle H APQ :=
sorry

end orthocenter_is_circumcenter_of_APQ_l766_766297


namespace compare_sqrt_difference_minimize_materials_plan_compare_a_inv_l766_766317

-- Problem 1
theorem compare_sqrt_difference : 3 - Real.sqrt 2 > 4 - 2 * Real.sqrt 2 := 
  sorry

-- Problem 2
theorem minimize_materials_plan (x y : ℝ) (h : x > y) : 
  4 * x + 6 * y > 3 * x + 7 * y := 
  sorry

-- Problem 3
theorem compare_a_inv (a : ℝ) (h : a > 0) : 
  (0 < a ∧ a < 1) → a < 1 / a ∧ (a = 1 → a = 1 / a) ∧ (a > 1 → a > 1 / a) :=
  sorry

end compare_sqrt_difference_minimize_materials_plan_compare_a_inv_l766_766317


namespace band_member_earnings_l766_766162

-- Define conditions
def n_people : ℕ := 500
def p_ticket : ℚ := 30
def r_earnings : ℚ := 0.7
def n_members : ℕ := 4

-- Definition of total earnings and share per band member
def total_earnings : ℚ := n_people * p_ticket
def band_share : ℚ := total_earnings * r_earnings
def amount_per_member : ℚ := band_share / n_members

-- Statement to be proved
theorem band_member_earnings : amount_per_member = 2625 := 
by
  -- Proof goes here
  sorry

end band_member_earnings_l766_766162


namespace algebra_ineq_example_l766_766683

theorem algebra_ineq_example (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 = x + y + z) :
  x + y + z + 3 ≥ 6 * ( ( (xy + yz + zx) / 3 ) ^ (1/3) ) :=
by
  sorry

end algebra_ineq_example_l766_766683


namespace triangular_weight_is_60_l766_766006

/-- Suppose there are weights: 5 identical round, 2 identical triangular, and 1 rectangular weight of 90 grams.
    The conditions are: 
    1. One round weight and one triangular weight balance three round weights.
    2. Four round weights and one triangular weight balance one triangular weight, one round weight, and one rectangular weight.
    Prove that the weight of the triangular weight is 60 grams. -/
theorem triangular_weight_is_60 
  (R T : ℕ)  -- We declare weights of round and triangular weights as natural numbers
  (h1 : R + T = 3 * R)  -- The first balance condition
  (h2 : 4 * R + T = T + R + 90)  -- The second balance condition
  : T = 60 := 
by
  sorry  -- Proof omitted

end triangular_weight_is_60_l766_766006


namespace consecutive_integer_sums_l766_766558

def sum_of_consecutive_integer_representations (n : ℕ) : ℕ :=
  (List.factors (2 * n)).count (λ k => k % 2 = 1)

theorem consecutive_integer_sums (n : ℕ) : ∃ (f : ℕ), 
  f = sum_of_consecutive_integer_representations n := 
sorry

end consecutive_integer_sums_l766_766558


namespace centroid_of_triangle_l766_766418

-- Definitions and conditions
def is_lattice_point (p : ℤ × ℤ) : Prop := 
  true -- Placeholder for a more specific definition if necessary

def triangle (A B C : ℤ × ℤ) : Prop := 
  true -- Placeholder for defining a triangle with vertices at integer grid points

def no_other_nodes_on_sides (A B C : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert no other integer grid points on the sides

def exactly_one_node_inside (A B C O : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert exactly one integer grid point inside the triangle

def medians_intersection_is_point_O (A B C O : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert \(O\) is the intersection point of the medians

-- Theorem statement
theorem centroid_of_triangle 
  (A B C O : ℤ × ℤ)
  (h1 : is_lattice_point A)
  (h2 : is_lattice_point B)
  (h3 : is_lattice_point C)
  (h4 : triangle A B C)
  (h5 : no_other_nodes_on_sides A B C)
  (h6 : exactly_one_node_inside A B C O) : 
  medians_intersection_is_point_O A B C O :=
sorry

end centroid_of_triangle_l766_766418


namespace range_of_a_l766_766986

theorem range_of_a (a : ℝ) : 
  (M = {x : ℝ | 2 * x + 1 < 3}) → 
  (N = {x : ℝ | x < a}) → 
  (M ∩ N = N) ↔ a ≤ 1 :=
by
  let M := {x : ℝ | 2 * x + 1 < 3}
  let N := {x : ℝ | x < a}
  simp [Set.subset_def]
  sorry

end range_of_a_l766_766986


namespace trip_duration_is_6_hours_l766_766475

def distance_1 := 55 * 4
def total_distance (A : ℕ) := distance_1 + 70 * A
def total_time (A : ℕ) := 4 + A
def average_speed (A : ℕ) := total_distance A / total_time A

theorem trip_duration_is_6_hours (A : ℕ) (h : 60 = average_speed A) : total_time A = 6 :=
by
  sorry

end trip_duration_is_6_hours_l766_766475


namespace band_member_earnings_l766_766161

-- Define conditions
def n_people : ℕ := 500
def p_ticket : ℚ := 30
def r_earnings : ℚ := 0.7
def n_members : ℕ := 4

-- Definition of total earnings and share per band member
def total_earnings : ℚ := n_people * p_ticket
def band_share : ℚ := total_earnings * r_earnings
def amount_per_member : ℚ := band_share / n_members

-- Statement to be proved
theorem band_member_earnings : amount_per_member = 2625 := 
by
  -- Proof goes here
  sorry

end band_member_earnings_l766_766161


namespace sufficient_but_not_necessary_l766_766582

variable {x : ℝ}

def p := x ≤ 1
def q := 1 / x < 1
def not_p := ¬ p

theorem sufficient_but_not_necessary :
  (not_p → q) ∧ (¬ (q → not_p)) :=
by
  sorry

end sufficient_but_not_necessary_l766_766582


namespace digits_divisible_by_eleven_l766_766532

theorem digits_divisible_by_eleven (A B : ℕ) (hA : 0 ≤ A ∧ A < 10) (hB : 0 ≤ B ∧ B < 10) :
  (4 + A * 1000 + B * 100 + 2 * 10 + B) % 11 = 0 ↔ A = 6 ∧ 0 ≤ B ∧ B < 10 :=
begin
  sorry
end

end digits_divisible_by_eleven_l766_766532


namespace usual_time_catch_bus_l766_766794

-- Define the problem context
variable (S T : ℝ)

-- Hypotheses for the conditions given
def condition1 : Prop := S * T = (4 / 5) * S * (T + 4)
def condition2 : Prop := S ≠ 0

-- Theorem that states the fact we need to prove
theorem usual_time_catch_bus (h1 : condition1 S T) (h2 : condition2 S) : T = 16 :=
by
  -- proof omitted
  sorry

end usual_time_catch_bus_l766_766794


namespace min_value_expression_l766_766938

theorem min_value_expression : ∀ (x y : ℝ), ∃ z : ℝ, z ≥ 3*x^2 + 2*x*y + 3*y^2 + 5 ∧ z = 5 :=
by
  sorry

end min_value_expression_l766_766938


namespace coefficient_x3_expansion_l766_766996

theorem coefficient_x3_expansion (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → ((∀ coeff : ℝ, 
  (x - a/x)^5 = ∑ r in finset.range 6, (((-a)^r) * (nat.choose 5 r)) * x^(5-2*r) → coeff = 30 → 
  (5 - 2 * (nat.choose 5 r) = 3)) )) : a = -6 :=
by sorry

end coefficient_x3_expansion_l766_766996


namespace interval_with_three_buses_l766_766029

theorem interval_with_three_buses (interval_two_buses : ℕ) (total_route_time : ℕ) (bus_count : ℕ) : 
  interval_two_buses = 21 → total_route_time = 2 * interval_two_buses → bus_count = 3 → 
  (total_route_time / bus_count) = 14 :=
by
  intros h1 h2 h3
  rw [h1, h3, ← h2]
  simp
  sorry

end interval_with_three_buses_l766_766029


namespace total_amount_after_5_months_l766_766478

-- Definitions from the conditions
def initial_deposit : ℝ := 100
def monthly_interest_rate : ℝ := 0.0036  -- 0.36% expressed as a decimal

-- Definition of the function relationship y with respect to x
def total_amount (x : ℕ) : ℝ := initial_deposit + initial_deposit * monthly_interest_rate * x

-- Prove the total amount after 5 months is 101.8
theorem total_amount_after_5_months : total_amount 5 = 101.8 :=
by
  sorry

end total_amount_after_5_months_l766_766478


namespace simplify_expression_l766_766385

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l766_766385


namespace quad_intersection_distance_l766_766737

theorem quad_intersection_distance :
  let y := λ x : ℝ, x^2 - 2*x - 3 in
  (y 1 = 0) ∧ (y 2 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ y x₁ = 0 ∧ y x₂ = 0 ∧ |x₁ - x₂| = 4 := 
by
  sorry

end quad_intersection_distance_l766_766737


namespace shaded_area_correct_l766_766542

-- Given definitions
def square_side_length : ℝ := 1
def grid_rows : ℕ := 3
def grid_columns : ℕ := 9

def triangle1_area : ℝ := 3
def triangle2_area : ℝ := 1
def triangle3_area : ℝ := 3
def triangle4_area : ℝ := 3

def total_grid_area := (grid_rows * grid_columns : ℕ) * square_side_length^2
def total_unshaded_area := triangle1_area + triangle2_area + triangle3_area + triangle4_area

-- Problem statement
theorem shaded_area_correct :
  total_grid_area - total_unshaded_area = 17 := 
by
  sorry

end shaded_area_correct_l766_766542


namespace simplify_sqrt_expression_l766_766337

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l766_766337


namespace solve_for_x_l766_766642

theorem solve_for_x (x : ℝ) (h : 1 / 4 - 1 / 6 = 4 / x) : x = 48 := 
sorry

end solve_for_x_l766_766642


namespace total_words_story_l766_766275

def words_per_line : ℕ := 10
def lines_per_page : ℕ := 20
def pages_filled : ℚ := 1.5
def words_left : ℕ := 100

theorem total_words_story : 
    words_per_line * lines_per_page * pages_filled + words_left = 400 := 
by
sorry

end total_words_story_l766_766275


namespace three_digit_perfect_squares_with_reversed_perfect_squares_l766_766022

/-- There exist exactly three three-digit numbers that are perfect squares and 
    whose reversals are also three-digit perfect squares. -/
theorem three_digit_perfect_squares_with_reversed_perfect_squares : 
    ∃ (n₁ n₂ n₃ : ℕ), 
    (100 ≤ n₁ ∧ n₁ ≤ 999 ∧ ∃ a₁, n₁ = a₁ * a₁ ∧ reverse_digits n₁ < 1000 ∧ ∃ b₁, reverse_digits n₁ = b₁ * b₁) ∧
    (100 ≤ n₂ ∧ n₂ ≤ 999 ∧ ∃ a₂, n₂ = a₂ * a₂ ∧ reverse_digits n₂ < 1000 ∧ ∃ b₂, reverse_digits n₂ = b₂ * b₂) ∧
    (100 ≤ n₃ ∧ n₃ ≤ 999 ∧ ∃ a₃, n₃ = a₃ * a₃ ∧ reverse_digits n₃ < 1000 ∧ ∃ b₃, reverse_digits n₃ = b₃ * b₃) ∧
    n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₂ ≠ n₃ :=
sorry

/-- Helper function to reverse the digits of a number. -/
def reverse_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.reverse |> List.foldl (λ a d, a * 10 + d) 0

end three_digit_perfect_squares_with_reversed_perfect_squares_l766_766022


namespace total_amount_spent_on_lunch_l766_766073

-- Define the amounts spent
def your_spending : ℝ
def friend_spending : ℝ
def total_spending : ℝ

-- Conditions:
-- 1. Your friend spent $5 more than you.
def condition1 := friend_spending = your_spending + 5

-- 2. Your friend spent $10 on their lunch.
def condition2 := friend_spending = 10

-- The question:
theorem total_amount_spent_on_lunch : condition1 ∧ condition2 → total_spending = 15 :=
by
  sorry

end total_amount_spent_on_lunch_l766_766073


namespace percentage_profit_correct_l766_766679

noncomputable def sellingPrice : Float := 670.0
noncomputable def originalCost : Float := 496.30
noncomputable def profit : Float := sellingPrice - originalCost
noncomputable def percentageProfit : Float := (profit / originalCost) * 100

theorem percentage_profit_correct :
  abs (percentageProfit - 34.99) < 0.01 := by
  sorry

end percentage_profit_correct_l766_766679


namespace geometric_series_sum_l766_766935

noncomputable def a : ℤ := 5
noncomputable def r : ℚ := -1 / 2

theorem geometric_series_sum :
  let s := a / (1 - r) in
  s = 10 / 3 := 
by {
  let a : ℚ := 5,
  let r : ℚ := -1 / 2,
  let s : ℚ := a / (1 - r),
  exact Eq.refl (10 / 3)
}

end geometric_series_sum_l766_766935


namespace average_lifespan_of_bulbs_l766_766099

noncomputable def midpoint (a b : ℝ) : ℝ := (a + b) / 2

theorem average_lifespan_of_bulbs :
  let count1 := 30
  let count2 := 30
  let count3 := 40
  let total_count := count1 + count2 + count3
  let midpoint1 := midpoint 60 100
  let midpoint2 := midpoint 100 140
  let midpoint3 := midpoint 140 180
  let total_lifespan := (midpoint1 * count1) + (midpoint2 * count2) + (midpoint3 * count3)
  let average_lifespan := total_lifespan / total_count
  average_lifespan = 124 := by
  sorry

end average_lifespan_of_bulbs_l766_766099


namespace problem_statement_l766_766637

def h (x : ℝ) : ℝ := Real.cos x

def f (y : ℝ) : ℝ := 
  if y = Real.cos x then 
    Real.cos x / (x ^ 2 + 1) 
  else
    0  -- This part will not be used in the proof

theorem problem_statement :
  f (h (Real.pi / 4)) = (Real.sqrt 2) / (2 * (Real.pi ^ 2 / 16 + 1)) := by
  sorry

end problem_statement_l766_766637


namespace max_closed_list_size_l766_766515

-- Define the initial list as a set of 10 distinct natural numbers
def initial_list := {a : ℕ | a ∈ finset.range 10 ∧ (∃ i: ℕ, a = i + 1)}

-- Define what it means for the list to be closed
def is_closed (s : finset ℕ) : Prop :=
  ∀ x y ∈ s, (nat.lcm x y ∈ s) ∨ (nat.lcm x y ∈ finset.range 10 → false)

-- Define the theorem we need to prove
theorem max_closed_list_size : 
  ∃ s : finset ℕ, initial_list ⊆ s ∧ is_closed s ∧ s.card = 1023 :=
sorry

end max_closed_list_size_l766_766515


namespace platform_length_correct_l766_766472

noncomputable def length_of_platform : ℝ :=
let T := 140 in
let S_kmph := 55 in
let t := 43.196544276457885 in
let S_mps := (S_kmph * 1000) / 3600 in
let D := S_mps * t in
D - T

theorem platform_length_correct :
  length_of_platform = 519.4444444444443 :=
by
  have T := 140 : ℝ
  have S_kmph := 55 : ℝ
  have t := 43.196544276457885 : ℝ
  let S_mps := (S_kmph * 1000) / 3600
  let D := S_mps * t
  have L := D - T
  show length_of_platform = 519.4444444444443,
  from sorry

end platform_length_correct_l766_766472


namespace isosceles_triangle_median_length_l766_766268

theorem isosceles_triangle_median_length (A B C M : Type) 
  (triangle : triangle A B C) 
  (isosceles : A B = A C) 
  (AB_AC : A B = 13) 
  (BC : B C = 15) 
  (M_midpoint : midpoint M B C ∈ segment B C) :
  length (A M) = Real.sqrt 112.75 :=
by
  sorry

end isosceles_triangle_median_length_l766_766268


namespace cards_total_eq_292_l766_766780

noncomputable def total_cards (n : ℕ) : ℕ :=
  (n^2 + 36)

theorem cards_total_eq_292 : ∃ (n : ℕ), (n^2 + 36 = (n + 1)^2 + 3) ∧ total_cards n = 292 := 
by
  use 16
  -- proving that n = 16 satisfies the given equation
  have h : 16^2 + 36 = (16 + 1)^2 + 3 := by
    calc 16^2 + 36 = 256 + 36 : by norm_num
                 ... = 292 : by norm_num
                 ... = (17^2 + 3) : by norm_num
  split
  { exact h, }
  { refl, }

end cards_total_eq_292_l766_766780


namespace possible_remainder_degrees_l766_766443

theorem possible_remainder_degrees (p : Polynomial ℝ) : 
  ∃ r : Polynomial ℝ, degree (3 * X^2 - 7 * X + 5) = 2 ∧ degree r < degree (3 * X^2 - 7 * X + 5) := by
  -- According to the division algorithm for polynomials, the degree of r should be less than the degree of the divisor
  have h_divisor : degree (3 * X^2 - 7 * X + 5) = 2 := by
    -- Simplify and compute degree of 3*x^2 - 7*x + 5
    sorry
  
  -- The degrees of the remainder r must be less than 2, thus could be 0 or 1.
  exact ⟨r, h_divisor, by sorry⟩

end possible_remainder_degrees_l766_766443


namespace integral_solution_l766_766905

noncomputable def integral_problem : ℝ :=
  ∫ x in 0..2, (real.sqrt (4 - x^2) + x^2)

theorem integral_solution : integral_problem = real.pi + 8 / 3 :=
  sorry

end integral_solution_l766_766905


namespace general_term_sum_first_n_terms_l766_766611

-- Definitions based on conditions
def arith_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 2) = a (n + 1) + d
def geo_seq (b : ℕ → ℤ) := ∀ (i j k : ℕ), i < j ∧ j < k → b (j - i) * b (k - j) = b (k - i)

-- Known conditions
def a (n : ℕ) : ℤ := if n = 0 then 0 else n
def d := 1

-- Prove the general term formula for {a_n} is a_n = n
theorem general_term (n : ℕ) : a (n + 1) = n + 1 := by
  sorry

-- Definitions of sequence {b_n}
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Prove the sum of the first n terms for {b_n} is n / (n + 1)
theorem sum_first_n_terms (n : ℕ) : (Finset.range n).sum (λ i, b i) = (n : ℚ) / (n + 1) := by
  sorry

end general_term_sum_first_n_terms_l766_766611


namespace necessary_condition_necessary_but_not_sufficient_l766_766231

variable (X : Type) [DiscreteRandomVariable X]
variable (a b : ℝ)

def variance (X : Type) [DiscreteRandomVariable X] : ℝ := sorry

axiom variance_transform (a : ℝ) (b : ℝ) (X : Type) [DiscreteRandomVariable X] :
  variance (a • X + b) = a^2 * variance X

theorem necessary_condition (h : variance (a • X + b) = 4 * variance X) : 
  a = 2 ∨ a = -2 :=
by
  sorry

theorem necessary_but_not_sufficient (h : variance (a • X + b) = 4 * variance X) : 
  ∃ a, a = 2 ∨ a = -2 :=
by
  sorry

end necessary_condition_necessary_but_not_sufficient_l766_766231


namespace greatest_difference_units_digit_l766_766768

/-- 
The three-digit integer of the form 72X is a multiple of 4.
Prove that the greatest possible difference between two of the possibilities for the units digit is 8
-/
theorem greatest_difference_units_digit : 
  ∃ (n m : ℕ), (720 ≤ n) ∧ (n ≤ 729) ∧ ((n % 4 = 0) ∧ m = n % 10) → (max m 0 - 0 = 8) := 
begin
  sorry,
end

end greatest_difference_units_digit_l766_766768


namespace number_of_correct_statements_l766_766686

variables {a d S : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n+1) = a n + d

def sum_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

-- Given conditions
def S6_gt_S7_gt_S5 (S : ℕ → ℤ) : Prop :=
  S 6 > S 7 ∧ S 7 > S 5

-- Statements
def statement_1 (d : ℤ) : Prop := d > 0
def statement_2 (S : ℕ → ℤ) : Prop := S 11 > 0
def statement_3 (S : ℕ → ℤ) : Prop := S 12 < 0
def statement_4 (S : ℕ → ℤ) : Prop := ∀ n, S n ≤ S 11
def statement_5 (a : ℕ → ℤ) : Prop := abs (a 5) > abs (a 7)

-- The number of correct statements is 2
theorem number_of_correct_statements 
  (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (arithmetic_seq : is_arithmetic_sequence a d)
  (sum_seq : sum_of_arithmetic_sequence S a)
  (cond : S6_gt_S7_gt_S5 S) :
  (¬statement_1 d) ∧
  statement_2 S ∧
  ¬statement_3 S ∧
  ¬statement_4 S ∧
  statement_5 a → true :=
by sorry

end number_of_correct_statements_l766_766686


namespace problem_distance_l766_766533

open Real

def parabola (x : ℝ) := x^2 + 3 * x + 2

def line (x : ℝ) := 8 - x

noncomputable def distance_between_points (p1 p2 : ℝ × ℝ) :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem_distance : 
  let points := [
    (x, line x) | x in {-6, 1}
  ] in
  distance_between_points (points.head!) (points.tail!.head!) = 7 * sqrt 2 := by
  sorry

end problem_distance_l766_766533


namespace find_m_l766_766399

-- Define the conditions
def ellipse_eq (x y : ℝ) (m : ℝ) : Prop := (x^2) / m + (y^2) / 4 = 1
def eccentricity (e : ℝ) : Prop := e = 2

-- Statement of the problem
theorem find_m (m : ℝ) (e : ℝ) (h1 : eccentricity e) (h2 : ∀ x y : ℝ, ellipse_eq x y m) :
  m = 3 ∨ m = 5 :=
sorry

end find_m_l766_766399


namespace number_of_different_towers_l766_766479

theorem number_of_different_towers
  (red blue yellow : ℕ)
  (total_height : ℕ)
  (total_cubes : ℕ)
  (discarded_cubes : ℕ)
  (ways_to_leave_out : ℕ)
  (multinomial_coefficient : ℕ) : 
  red = 3 → blue = 4 → yellow = 5 → total_height = 10 → total_cubes = 12 → discarded_cubes = 2 →
  ways_to_leave_out = 66 → multinomial_coefficient = 4200 →
  (ways_to_leave_out * multinomial_coefficient) = 277200 :=
by
  -- proof skipped
  sorry

end number_of_different_towers_l766_766479


namespace simplify_cubic_root_l766_766723

-- Define the given integers
def a := 40
def b := 50
def c := 60

-- Define the expression under the cubic root and its simplified form
def expression := a^3 + b^3 + c^3
def simplified := 10 * ∛ (4^3 + 5^3 + 6^3)

-- Theorem statement
theorem simplify_cubic_root : ∛ expression = simplified :=
sorry

end simplify_cubic_root_l766_766723


namespace question1_question2_question3_l766_766493

def f : Nat → Nat → Nat := sorry

axiom condition1 : f 1 1 = 1
axiom condition2 : ∀ m n, f m (n + 1) = f m n + 2
axiom condition3 : ∀ m, f (m + 1) 1 = 2 * f m 1

theorem question1 (n : Nat) : f 1 n = 2 * n - 1 :=
sorry

theorem question2 (m : Nat) : f m 1 = 2 ^ (m - 1) :=
sorry

theorem question3 : f 2002 9 = 2 ^ 2001 + 16 :=
sorry

end question1_question2_question3_l766_766493


namespace simplify_sqrt_expression_l766_766342

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l766_766342


namespace product_of_possible_values_of_N_l766_766514

theorem product_of_possible_values_of_N (N : ℤ) : 
  (∃ M L : ℤ, 
    (M = L + N) ∧ 
    ((M - 6) - (L + 4) = 3 ∨ (M - 6) - (L + 4) = -3)) → 
  (N = 13 ∨ N = 7) → 
  13 * 7 = 91 :=
begin
  sorry
end

end product_of_possible_values_of_N_l766_766514


namespace value_of_f_at_9_l766_766568

def f (n : ℕ) : ℕ := n^3 + n^2 + n + 17

theorem value_of_f_at_9 : f 9 = 836 := sorry

end value_of_f_at_9_l766_766568


namespace three_buses_interval_l766_766040

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end three_buses_interval_l766_766040


namespace simplify_sqrt_expression_l766_766338

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l766_766338


namespace cashier_can_satisfy_request_l766_766423

theorem cashier_can_satisfy_request (k : ℕ) (h : k > 8) : ∃ m n : ℕ, k = 3 * m + 5 * n :=
sorry

end cashier_can_satisfy_request_l766_766423


namespace evaluate_expression_l766_766799

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end evaluate_expression_l766_766799


namespace largest_perfect_square_factor_of_3402_l766_766055

theorem largest_perfect_square_factor_of_3402 : 
  ∃ (n : ℕ), (∃ (k : ℕ), 3402 = k * k * n) ∧ 81 = k * k :=
begin
  sorry
end

end largest_perfect_square_factor_of_3402_l766_766055


namespace range_of_data_set_l766_766579

/-- Define the set of data -/
def dataSet : Set ℤ := {2, -2, 6, 4, -1}

/-- The proof that the range of the set is 8 -/
theorem range_of_data_set : (Set.max dataSet - Set.min dataSet) = 8 := by
  admit -- using 'sorry' since Lean does not have built-in max/min computation for sets

end range_of_data_set_l766_766579


namespace jessica_watermelons_l766_766677

theorem jessica_watermelons (original : ℕ) (eaten : ℕ) (remaining : ℕ) 
    (h1 : original = 35) 
    (h2 : eaten = 27) 
    (h3 : remaining = original - eaten) : 
  remaining = 8 := 
by {
    -- This is where the proof would go
    sorry
}

end jessica_watermelons_l766_766677


namespace remaining_pool_area_l766_766093

-- Define the area of the circle given its diameter
def circle_area (diameter : ℝ) : ℝ := 
  let radius := diameter / 2
  π * radius^2

-- Define the area of the rectangle
def rectangle_area (length width : ℝ) : ℝ := 
  length * width

-- Define the remaining area after subtracting the rectangle's area from the circle's area
def remaining_area (diameter rect_length rect_width : ℝ) : ℝ :=
  circle_area diameter - rectangle_area rect_length rect_width

-- Conditions:
-- Diameter of the circular pool = 13 meters
-- Length and width of the rectangular obstacle = 2.5 meters and 4 meters

-- Statement: Prove that the area of the remaining pool space is 132.7325π - 10
theorem remaining_pool_area :
  remaining_area 13 2.5 4 = 132.7325 * π - 10 :=
by 
  sorry

end remaining_pool_area_l766_766093


namespace ticket_price_difference_l766_766430

def pre_bought_payment (number_pre : ℕ) (price_pre : ℕ) : ℕ :=
  number_pre * price_pre

def gate_payment (number_gate : ℕ) (price_gate : ℕ) : ℕ :=
  number_gate * price_gate

theorem ticket_price_difference :
  ∀ (number_pre number_gate price_pre price_gate : ℕ),
  number_pre = 20 →
  price_pre = 155 →
  number_gate = 30 →
  price_gate = 200 →
  gate_payment number_gate price_gate - pre_bought_payment number_pre price_pre = 2900 :=
by {
  intros,
  sorry
}

end ticket_price_difference_l766_766430


namespace v3_correct_l766_766793

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

-- Define the values of v_0, v_1, v_2, and v_3 according to Qin Jiushao's algorithm
def v0 : ℝ := 1
def v1 := 2 - 12
def v2 := v1 * 2 + 60
def v3 := v2 * 2 - 160

-- The theorem we want to prove
theorem v3_correct : v3 = -80 := by
  -- Definitions according to Qin Jiushao's algorithm
  have v1_def : v1 = 2 - 12 := rfl
  have v2_def : v2 = v1 * 2 + 60 := rfl
  have v3_def : v3 = v2 * 2 - 160 := rfl
  rwa [v1_def, v2_def, v3_def]
  sorry

end v3_correct_l766_766793


namespace probability_shortening_exactly_one_digit_l766_766848
open Real

theorem probability_shortening_exactly_one_digit :
  ∀ (sequence : List ℕ),
    (sequence.length = 2015) ∧
    (∀ (x ∈ sequence), x = 0 ∨ x = 9) →
      (P (λ seq, (length (shrink seq) = 2014)) sequence = 1.564e-90) := sorry

end probability_shortening_exactly_one_digit_l766_766848


namespace milk_production_l766_766726

theorem milk_production (a b c d e : ℕ) (ha : a ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) :
  let average_rate_per_cow_per_day := b / (a * c),
      x := (2 * b) / (3 * a * c)
  in (average_rate_per_cow_per_day = b / (a * c) ∧ x = (2 * b) / (3 * a * c)) → (d * e * b) / (a * c) = (d * e * b) / (a * c) := sorry

end milk_production_l766_766726


namespace total_students_in_both_classrooms_l766_766674

theorem total_students_in_both_classrooms
  (x y : ℕ)
  (hx1 : 80 * x - 250 = 90 * (x - 5))
  (hy1 : 85 * y - 480 = 95 * (y - 8)) :
  x + y = 48 := 
sorry

end total_students_in_both_classrooms_l766_766674


namespace length_of_AR_l766_766280

theorem length_of_AR {k r : ℝ} (h : r = k / 2) :
  let AB := 2 * k
      BC := k
      OP := r
      P := (k, 0)
      O := (k, k / 2)
      A := (2 * k, 0)
  in
  sqrt (3 * k^2 / 4) = sqrt 3 * k / 2 :=
by
  sorry

end length_of_AR_l766_766280


namespace max_value_of_3sinx_4cosx_is_5_l766_766155

def max_value_of_function (a b : ℝ) : ℝ :=
  (sqrt (a^2 + b^2))

theorem max_value_of_3sinx_4cosx_is_5 :
  max_value_of_function 3 4 = 5 :=
by
  sorry

end max_value_of_3sinx_4cosx_is_5_l766_766155


namespace coaching_fee_l766_766319

theorem coaching_fee :
  let joined_on_new_years_day := true,
      discontinued_on_nov_4 := true,
      daily_charge := 23,
      discount_percent := 10 / 100,
      days_in_full_months := 304,  -- days from Jan 1 to end of October
      extra_days_in_nov := 4,
      public_holidays := 5,
      total_days := days_in_full_months + extra_days_in_nov - public_holidays,
      periods_of_days := total_days / 30,
      full_periods := 10,  -- total_days / 30 is approximately 10 with integer division
      days_in_full_periods := full_periods * 30,
      discount_per_period := 30 * daily_charge * discount_percent,
      total_discount := full_periods * discount_per_period,
      total_fee_without_discount := total_days * daily_charge,
      total_fee_with_discount := total_fee_without_discount - total_discount
  in total_fee_with_discount = 6256 := 
  by 
    -- The proof details would go here.
    sorry

end coaching_fee_l766_766319


namespace simplify_radical_expression_l766_766362

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l766_766362


namespace sum_of_possible_candies_l766_766069

theorem sum_of_possible_candies :
  let C_values := {C | C < 100 ∧ C % 6 = 2 ∧ C % 8 = 6}
  sum C_values = 200 :=
by
  let C_values := {C | C < 100 ∧ C % 6 = 2 ∧ C % 8 = 6}
  sorry

end sum_of_possible_candies_l766_766069


namespace max_value_of_trig_function_l766_766150

theorem max_value_of_trig_function : 
  ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 5 := sorry


end max_value_of_trig_function_l766_766150


namespace greatest_integer_solution_l766_766051

theorem greatest_integer_solution :
  ∃ x : ℤ, (∃ (k : ℤ), (8 : ℚ) / 11 > k / 15 ∧ k = 10) ∧ x = 10 :=
by {
  sorry
}

end greatest_integer_solution_l766_766051


namespace karen_cases_picked_up_l766_766274

theorem karen_cases_picked_up (total_boxes : ℤ) (boxes_per_case : ℤ) (h1 : total_boxes = 36) (h2 : boxes_per_case = 12) : (total_boxes / boxes_per_case) = 3 := by
  sorry

end karen_cases_picked_up_l766_766274


namespace holey_triangle_tiling_condition_l766_766482

-- Define the conditions of the holey triangle
structure HoleyTriangle where
  n : ℕ -- side length of the triangle
  holes : Set (ℕ × ℕ) -- set of coordinates (i, j) representing the holes
  
-- Define the condition for tiling with diamonds
def canTileWithDiamonds (T : HoleyTriangle) : Prop :=
  ∀ (k : ℕ) (1 ≤ k ∧ k ≤ T.n),
    (∃S, (∀ (i j : ℕ), (i, j) ∈ S → 1 ≤ i ∧ i ≤ k ∧ 1 ≤ j ∧ j ≤ k) ∧ (|S ∩ T.holes| ≤ k))

-- Statement of the theorem
theorem holey_triangle_tiling_condition (T : HoleyTriangle) :
  canTileWithDiamonds T ↔
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ T.n →
    (∃S, (∀ (i j : ℕ), (i, j) ∈ S → 1 ≤ i ∧ i ≤ k ∧ 1 ≤ j ∧ j ≤ k) ∧ (|S ∩ T.holes| ≤ k)) := 
sorry

end holey_triangle_tiling_condition_l766_766482


namespace handshakes_4_handshakes_n_l766_766644

-- Defining the number of handshakes for n people
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

-- Proving that the number of handshakes for 4 people is 6
theorem handshakes_4 : handshakes 4 = 6 := by
  sorry

-- Proving that the number of handshakes for n people is (n * (n - 1)) / 2
theorem handshakes_n (n : ℕ) : handshakes n = (n * (n - 1)) / 2 := by 
  sorry

end handshakes_4_handshakes_n_l766_766644


namespace chord_equation_through_point_bisected_at_P_l766_766534

theorem chord_equation_through_point_bisected_at_P :
  ∀ (A1 A2 P : ℝ × ℝ),
    (∃ x1 y1 x2 y2,
      A1 = (x1, y1) ∧ A2 = (x2, y2) ∧ P = (2, -1) ∧
      (x1^2 / 6 + y1^2 / 5 = 1) ∧ (x2^2 / 6 + y2^2 / 5 = 1) ∧
      (x1 + x2 = 4) ∧ (y1 + y2 = -2)) →
    (∀ k b,
      (∀ x y, (x, y) = A1 ∨ (x, y) = A2 ↔ (y = k * x + b)) →
      (b = -13 ∧ k = 5 / 3 ∧ ∀ x y, y = k * x + b ↔ 5 * x - 3 * y - 13 = 0)) :=
begin
  sorry
end

end chord_equation_through_point_bisected_at_P_l766_766534


namespace degree_resultant_l766_766133

def polynomial_1 (x : ℝ) := 2 * x^5 - 3 * x^3 + x^2 - 14
def polynomial_2 (x : ℝ) := 3 * x^11 - 9 * x^8 + 9 * x^5 + 30
def polynomial_3 (x : ℝ) := x^3 + 5

def prod_poly (x : ℝ) := polynomial_1 x * polynomial_2 x
def power_poly (x : ℝ) := polynomial_3 x ^ 7
def resultant_poly (x : ℝ) := prod_poly x - power_poly x

theorem degree_resultant : degree (resultant_poly x) = 21 :=
  sorry

end degree_resultant_l766_766133


namespace sufficient_but_not_necessary_l766_766536

theorem sufficient_but_not_necessary (a : ℝ) : 
  (∀ a : ℝ, a = 2 → (∃ x : ℝ, x^2 - 3*x + a = 0)) ∧ ¬(∀ a : ℝ, (∃ x : ℝ, x^2 - 3*x + a = 0) → a = 2) :=
begin
  sorry
end

end sufficient_but_not_necessary_l766_766536


namespace solve_inverse_function_problem_l766_766649

def f (x : ℝ) : ℝ := log x / log 3 + 1

noncomputable def f_inv (x : ℝ) : ℝ := 3 ^ x - 1

theorem solve_inverse_function_problem : f_inv 8 = 2 := by
  sorry

end solve_inverse_function_problem_l766_766649


namespace correct_options_l766_766818

noncomputable theory

def direction_vector_a := (2 : ℝ, 3, -1) -- Direction vector for line l1
def direction_vector_b := (-1 : ℝ, -3 / 2, 1 / 2) -- Direction vector for line l2

def direction_vector_c := (1 : ℝ, -1, 2) -- Direction vector for another line l
def normal_vector_alpha_1 := (6 : ℝ, 4, -1) -- Normal vector for plane alpha

def direction_vector_d := (0 : ℝ, 3, 0) -- Direction vector for another line l
def normal_vector_alpha_2 := (0 : ℝ, -1, 0) -- Normal vector for plane alpha

def normal_vector_u := (2 : ℝ, 2, -1) -- Normal vector for plane alpha
def normal_vector_v := (-3 : ℝ, 4, 2) -- Normal vector for plane beta

axiom option_A_correct : 
  ∃ k : ℝ, direction_vector_b = k • (2 : ℝ, 3, -1)

axiom option_B_incorrect : 
  ¬(direction_vector_c.1 * normal_vector_alpha_1.1 + direction_vector_c.2 * normal_vector_alpha_1.2 + direction_vector_c.3 * normal_vector_alpha_1.3 = 0)

axiom option_C_incorrect : 
  ¬(normal_vector_alpha_2 = (-1/3) • (0 : ℝ, 3, 0))

axiom option_D_correct : 
  normal_vector_u.1 * normal_vector_v.1 + normal_vector_u.2 * normal_vector_v.2 + normal_vector_u.3 * normal_vector_v.3 = 0

theorem correct_options : option_A_correct ∧ option_D_correct := by
  sorry

end correct_options_l766_766818


namespace problem_solution_l766_766634

variable {α : ℝ}

def problem_condition : Prop := sin (π / 4 + α) = 1 / 2

theorem problem_solution (h : problem_condition) :
  (sin (5 * π / 4 + α) / cos (9 * π / 4 + α)) * cos (7 * π / 4 - α) = -1 / 2 :=
by
  sorry

end problem_solution_l766_766634


namespace probability_point_closer_to_7_than_0_l766_766490

noncomputable def segment_length (a b : ℝ) : ℝ := b - a
noncomputable def closer_segment (a c b : ℝ) : ℝ := segment_length c b

theorem probability_point_closer_to_7_than_0 :
  let a := 0
  let b := 10
  let c := 7
  let midpoint := (a + c) / 2
  let total_length := b - a
  let closer_length := segment_length midpoint b
  (closer_length / total_length) = 0.7 :=
by
  sorry

end probability_point_closer_to_7_than_0_l766_766490


namespace probability_of_4_rainy_days_out_of_6_l766_766082

noncomputable def probability_of_rain_on_given_day : ℝ := 0.5

noncomputable def probability_of_rain_on_exactly_k_days (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem probability_of_4_rainy_days_out_of_6 :
  probability_of_rain_on_exactly_k_days 6 4 probability_of_rain_on_given_day = 0.234375 :=
by
  sorry

end probability_of_4_rainy_days_out_of_6_l766_766082


namespace t_minus_s_correct_l766_766104

def classEnrollments := [80, 40, 40, 20, 10, 10]  -- Condition for class enrollments

-- Definition for t (average number of students per class chosen by a teacher)
def avgStudentsPerTeacher : Float :=
  classEnrollments.sum.toFloat / classEnrollments.length.toFloat

-- Definition for s (average number of students encountered by a randomly picked student)
def avgStudentsPerStudent : Float :=
  classEnrollments.map (λ x => (x.toFloat * (x.toFloat / 200.0))).sum

-- Definition for t - s
def diffTMinusS : Float :=
  avgStudentsPerTeacher - avgStudentsPerStudent

-- Lean statement to assert the required proof
theorem t_minus_s_correct :
  diffTMinusS = -17.67 :=
by
  sorry

end t_minus_s_correct_l766_766104


namespace probability_exactly_two_sunny_days_l766_766564

-- Define the conditions
def rain_probability : ℝ := 0.8
def sun_probability : ℝ := 1 - rain_probability
def days : ℕ := 5
def sunny_days : ℕ := 2
def rainy_days : ℕ := days - sunny_days

-- Define the combinatorial and probability calculations
def comb (n k : ℕ) : ℕ := Nat.choose n k
def probability_sunny_days : ℝ := comb days sunny_days * (sun_probability ^ sunny_days) * (rain_probability ^ rainy_days)

theorem probability_exactly_two_sunny_days : probability_sunny_days = 51 / 250 := by
  sorry

end probability_exactly_two_sunny_days_l766_766564


namespace problems_left_to_grade_l766_766830

-- Defining all the conditions
def worksheets_total : ℕ := 14
def worksheets_graded : ℕ := 7
def problems_per_worksheet : ℕ := 2

-- Stating the proof problem
theorem problems_left_to_grade : 
  (worksheets_total - worksheets_graded) * problems_per_worksheet = 14 := 
by
  sorry

end problems_left_to_grade_l766_766830


namespace arithmetic_sequence_general_formula_l766_766739

theorem arithmetic_sequence_general_formula (x : ℤ) (a_n : ℕ → ℤ) 
  (h1 : a_n 1 = x - 1) (h2 : a_n 2 = x + 1) (h3 : a_n 3 = 2x + 3) : 
  ∀ n, a_n n = 2 * n - 3 :=
by
  sorry

end arithmetic_sequence_general_formula_l766_766739


namespace simplify_expression_l766_766383

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l766_766383


namespace optimal_diagonal_length_is_66_69_l766_766494

noncomputable def max_pentagon_area_diagonal_length (perimeter : ℝ) : ℝ :=
  let a := (1 + 2 * Real.sqrt 2) / 4
  let b := perimeter / 2
  in b / (2 * a)

theorem optimal_diagonal_length_is_66_69 :
  max_pentagon_area_diagonal_length 255.3 ≈ 66.69 := by
  -- Prove that the optimal length for the diagonal x is approximately 66.69 meters
  sorry

end optimal_diagonal_length_is_66_69_l766_766494


namespace log_two_plus_log_five_eq_one_l766_766892

-- Lean 4 statement
theorem log_two_plus_log_five_eq_one : log 2 + log 5 = 1 := by sorry

end log_two_plus_log_five_eq_one_l766_766892


namespace skilled_picker_capacity_minimize_costs_l766_766074

theorem skilled_picker_capacity (x : ℕ) (h1 : ∀ x : ℕ, ∀ s : ℕ, s = 3 * x) (h2 : 450 * 25 = 3 * x * 25 + 600) :
  s = 30 :=
by
  sorry

theorem minimize_costs (s n m : ℕ)
(h1 : s ≤ 20)
(h2 : n ≤ 15)
(h3 : 600 = s * 30 + n * 10)
(h4 : ∀ y, y = s * 300 + n * 80) :
  m = 15 ∧ s = 15 :=
by
  sorry

end skilled_picker_capacity_minimize_costs_l766_766074


namespace elongation_isosceles_triangle_problem_l766_766115

/-- Given an isosceles triangle M₀N₀P with M₀P = N₀P and ∠M₀PN₀ = α ° and the condition 
    that 10α is an integer, prove that after 5 elongations, MₖN₅₋ₖP being an isosceles 
    triangle leads to 10α = 264. -/
theorem elongation_isosceles_triangle_problem 
  (α : ℝ)
  (h1 : M₀P = N₀P)
  (h2 : ∠ M₀PN₀ = α)
  (h3 : (10 * α).denom = 1) :
  after_elongations_isosceles Mₖ N₅₋ₖ P := sorry

end elongation_isosceles_triangle_problem_l766_766115


namespace compute_fraction_power_l766_766901

theorem compute_fraction_power :
  9 * (2/3)^4 = 16/9 := 
by
  sorry

end compute_fraction_power_l766_766901


namespace sequence_difference_l766_766764

noncomputable def S (n : ℕ+) : ℤ :=
  2 * n ^ 2 - 3 * n

def a (n : ℕ+) : ℤ :=
  if n = 1 then S n else S n - S (n - 1)

theorem sequence_difference (p q : ℕ+) (h : p - q = 5) : a p - a q = 20 := 
by
  sorry

end sequence_difference_l766_766764


namespace num_of_neg_x_sqrt_pos_integer_l766_766958

theorem num_of_neg_x_sqrt_pos_integer : 
  ( ∃ n : ℕ, 
      (1 ≤ n ∧ n ≤ 9) 
      ∧ (∀ x, x = ↑n^2 - 100 → x < 0) ) -> 
  (nat.card (set_of (λ x, ∃ n : ℕ, 1 ≤ n ∧ n ≤ 9 ∧ x = n^2 - 100)) = 9) :=
begin
  sorry
end

end num_of_neg_x_sqrt_pos_integer_l766_766958


namespace smallest_period_of_f_min_value_of_f_range_of_g_l766_766600

noncomputable def f (x : ℝ) : ℝ := 
  (1 / 2) * Real.sin (2 * x) - sqrt 3 * (Real.cos x) ^ 2

noncomputable def g (x : ℝ) : ℝ := 
  f (x / 2)

theorem smallest_period_of_f : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T' ≥ π) := 
  sorry

theorem min_value_of_f : (∀ x, f x ≥ -((2 + sqrt 3) / 2)) ∧ (∃ x, f x = -((2 + sqrt 3) / 2)) :=
  sorry

theorem range_of_g : (∀ x, x ∈ Icc (π / 2) π → g x ∈ Icc ((1 - sqrt 3) / 2) ((2 - sqrt 3) / 2)) :=
  sorry

end smallest_period_of_f_min_value_of_f_range_of_g_l766_766600


namespace multiplication_example_l766_766464

-- Definition of factors and product
def factors (a b : ℕ) (c : ℕ) : Prop := a * b = c

-- Theorem stating that 12 and 5 are factors, and 60 is the product
theorem multiplication_example : factors 12 5 60 :=
by 
  have h : 12 * 5 = 60 := rfl
  exact h

#check multiplication_example

end multiplication_example_l766_766464


namespace max_value_of_trig_function_l766_766151

theorem max_value_of_trig_function : 
  ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 5 := sorry


end max_value_of_trig_function_l766_766151


namespace repeating_decimal_sum_l766_766552

theorem repeating_decimal_sum :
  let x : ℚ := 0.2 -- 0.\overline{2}
  let y : ℚ := 0.02 -- 0.\overline{02}
  x + y = 80 / 333 :=
by {
  -- We use library functions to convert repeating decimals to fractions
  let x : ℚ := 2 / 9
  let y : ℚ := 2 / 99
  -- Show the expected sum is the fraction in lowest terms:
  x + y = (80 / 333)
  sorry
}

end repeating_decimal_sum_l766_766552


namespace proof_problem_l766_766201

variable {a b c d e f : ℝ}

theorem proof_problem :
  (a * b * c = 130) →
  (b * c * d = 65) →
  (d * e * f = 250) →
  (a * f / (c * d) = 0.5) →
  (c * d * e = 1000) :=
by
  intros h1 h2 h3 h4
  sorry

end proof_problem_l766_766201


namespace complement_of_intersection_l766_766301

noncomputable def real_set_complement (A : set ℝ) : set ℝ := 
  {x | x ∉ A}

theorem complement_of_intersection :
  let M := {x : ℝ | abs x < 1},
      N := {y : ℝ | ∃ x, y = 3^x ∧ abs x < 1} in
  real_set_complement (M ∩ N) = {y : ℝ | y ≤ 1/3} ∪ {y : ℝ | y ≥ 1} :=
by
  sorry

end complement_of_intersection_l766_766301


namespace pipe_b_fills_tank_7_times_faster_l766_766715

theorem pipe_b_fills_tank_7_times_faster 
  (time_A : ℝ) 
  (time_B : ℝ)
  (combined_time : ℝ) 
  (hA : time_A = 30)
  (h_combined : combined_time = 3.75) 
  (hB : time_B = time_A / 7) :
  time_B =  30 / 7 :=
by
  sorry

end pipe_b_fills_tank_7_times_faster_l766_766715


namespace natalie_sister_diaries_l766_766305

theorem natalie_sister_diaries :
  let initial_diaries := 23
  let bought_diaries := 5 * initial_diaries
  let total_diaries := initial_diaries + bought_diaries
  let lost_diaries := (7 / 9) * total_diaries
  let diaries_now := total_diaries - lost_diaries
  diaries_now = 31 :=
by
  let initial_diaries := 23
  let bought_diaries := 5 * initial_diaries
  let total_diaries := initial_diaries + bought_diaries
  let lost_diaries := (7 / 9) * total_diaries
  let diaries_now := total_diaries - lost_diaries
  exact eq.refl 31
  sorry

end natalie_sister_diaries_l766_766305


namespace largest_prime_mersenne_below_500_l766_766861

def is_mersenne (m : ℕ) (n : ℕ) := m = 2^n - 1
def is_power_of_2 (n : ℕ) := ∃ (k : ℕ), n = 2^k

theorem largest_prime_mersenne_below_500 : ∀ (m : ℕ), 
  m < 500 →
  (∃ n, is_power_of_2 n ∧ is_mersenne m n ∧ Nat.Prime m) →
  m ≤ 3 := 
by
  sorry

end largest_prime_mersenne_below_500_l766_766861


namespace BH_eq_DE_l766_766572

-- Variables and Assumptions
variables {A B C H O D E : Point}
variable (circle_O : Circumcircle A B C O)
variable (altitudes_intersect : AltitudesIntersectAt A B C H)
variable (D_on_extension_BO : OnExtension BO D)
variable (angle_ADC_eq_angle_ABC : ∠ADC = ∠ABC)
variable (line_H_parallel_BO : ∥ H BO E)
variable (E_on_minor_arc_AC : OnMinorArcACcircleO E A C)

-- Theorem to prove
theorem BH_eq_DE (h1 : AB < BC) 
  (h2 : AltitudesIntersectAt A B C H) 
  (h3 : OnExtension BO D) 
  (h4 : ∠ADC = ∠ABC) 
  (h5 : ParallelThroughPoint H BO) 
  (h6 : OnMinorArcACcircleO E A C) : 
  BH = DE :=
sorry

end BH_eq_DE_l766_766572


namespace simplify_expr_l766_766343

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l766_766343


namespace number_of_integer_coordinate_points_on_line_l766_766869

noncomputable def count_integer_points_on_line : ℕ :=
  let A := (2 : ℤ, 3 : ℤ)
  let B := (50 : ℤ, 303 : ℤ)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  let line_eq := λ x : ℤ, (slope * x - slope * A.1 + A.2)
  let is_between_A_B := λ x : ℤ, A.1 < x ∧ x < B.1
  let is_integer_coordinate := λ x : ℤ, (line_eq x).denom = 1
  finset.filter (λ x, is_between_A_B x ∧ is_integer_coordinate x) (finset.range 51).filter(λ x, x > 2).val

theorem number_of_integer_coordinate_points_on_line :
  count_integer_points_on_line = 11 := by
  sorry

end number_of_integer_coordinate_points_on_line_l766_766869


namespace time_after_hours_l766_766396

def current_time := 9
def total_hours := 2023
def clock_cycle := 12

theorem time_after_hours : (current_time + total_hours) % clock_cycle = 8 := by
  sorry

end time_after_hours_l766_766396


namespace eval_expression_l766_766807

theorem eval_expression : 4 * (8 - 3) - 7 = 13 :=
by
  sorry

end eval_expression_l766_766807


namespace four_distinct_sum_equal_l766_766728

theorem four_distinct_sum_equal (S : Finset ℕ) (hS : S.card = 10) (hS_subset : S ⊆ Finset.range 38) :
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d :=
by
  sorry

end four_distinct_sum_equal_l766_766728


namespace optimal_first_change_l766_766538

def initial_number : ℝ := 0.1234567

def change_digit_to_nine (n : ℕ) (d : ℕ) := 
if n ≤ 6 then 
  let digits := (0.1234567 * 10^d).toNat % 10 in -- toNat to extract digit and % to get the specific digit
  if digits < 9 then 
    9 * 10^(6 - d) + initial_number * 10^6 % 10^(6-d)  
  else initial_number
else initial_number

theorem optimal_first_change : change_digit_to_nine 1 1  = 0.9234567 :=
by
  -- proof to be written
  sorry

end optimal_first_change_l766_766538


namespace ellipse_foci_distance_sum_l766_766591

theorem ellipse_foci_distance_sum
    (x y : ℝ)
    (PF1 PF2 : ℝ)
    (a : ℝ)
    (h_ellipse : (x^2 / 36) + (y^2 / 16) = 1)
    (h_foci : ∀F1 F2, ∃e > 0, F1 = (e, 0) ∧ F2 = (-e, 0))
    (h_point_on_ellipse : ∀x y, (x^2 / 36) + (y^2 / 16) = 1 → (x, y) = (PF1, PF2))
    (h_semi_major_axis : a = 6):
    |PF1| + |PF2| = 12 := 
by
  sorry

end ellipse_foci_distance_sum_l766_766591


namespace simplify_expr_l766_766353

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l766_766353


namespace incorrect_statement_b_l766_766907

theorem incorrect_statement_b (a b : ℝ) (k s : ℝ) :
  (a > b → a + c > b + c) ∧
  ((∀ (a b : ℝ), (a ≤ b) → a + c ≤ b + c) → 
  (∀ (a b : ℝ), (a = b) ↔ √(a*b) ≤ (a + b) / 2)) ∧
  (∀ (x y : ℝ), (x + y = s) → x * y ≤ ((x + y) / 2) * ((x + y) / 2)) ∧
  ¬ (∀ (x y : ℝ), x * y = k → x^2 + y^2 ≥ 2 * k) := sorry

end incorrect_statement_b_l766_766907


namespace sum_of_acute_angles_l766_766298

open Real

theorem sum_of_acute_angles (θ₁ θ₂ : ℝ)
  (h1 : 0 < θ₁ ∧ θ₁ < π / 2)
  (h2 : 0 < θ₂ ∧ θ₂ < π / 2)
  (h_eq : (sin θ₁) ^ 2020 / (cos θ₂) ^ 2018 + (cos θ₁) ^ 2020 / (sin θ₂) ^ 2018 = 1) :
  θ₁ + θ₂ = π / 2 := sorry

end sum_of_acute_angles_l766_766298


namespace simplify_expr_l766_766345

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l766_766345


namespace arithmetic_expression_value_l766_766806

theorem arithmetic_expression_value : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_expression_value_l766_766806


namespace ticket_cost_difference_l766_766427

theorem ticket_cost_difference (num_prebuy : ℕ) (price_prebuy : ℕ) (num_gate : ℕ) (price_gate : ℕ)
  (h_prebuy : num_prebuy = 20) (h_price_prebuy : price_prebuy = 155)
  (h_gate : num_gate = 30) (h_price_gate : price_gate = 200) :
  num_gate * price_gate - num_prebuy * price_prebuy = 2900 :=
by
  sorry

end ticket_cost_difference_l766_766427


namespace can_find_34_real_coins_l766_766777

noncomputable def find_real_coins (coins : Fin 100 → ℝ) (coins_real : ∀ i j, 0 ≤ i - j → 50 ≤ i - j → coins i = coins j → coins_real_weight (coins i)) : Prop :=
  ∃ (i j : Fin 100), i < j ∧ (coins i = coins j) →
  (∀ i, 0 ≤ i → i < 17 ∨ 84 ≤ i < 100 → coins i = coins_real_weight (coins i)) ∨
  (∀ i, 0 ≤ i < 34 → coins i = coins_real_weight (coins i)) ∨
  (∀ i, 67 ≤ i < 100 → coins i = coins_real_weight (coins i))
  
theorem can_find_34_real_coins : 
  (∀ coins (coins_real : ∀ i j, 0 ≤ i - j → 50 ≤ i - j → coins i = coins j -> coins_real_weight (coins i)),
   find_real_coins coins coins_real :=
sorry

end can_find_34_real_coins_l766_766777


namespace log_base_2_of_prop_l766_766889

noncomputable def prop : ℝ := 16 * real.rpow 8 (1/3) * real.rpow 32 (1/5)

theorem log_base_2_of_prop : real.logb 2 prop = 6 :=
by {
    have h1 : 16 = 2^4, from sorry,
    have h2 : real.rpow 8 (1 / 3) = 2^1, from sorry,
    have h3 : real.rpow 32 (1 / 5) = 2^1, from sorry,
    sorry
}

end log_base_2_of_prop_l766_766889


namespace combined_height_difference_is_correct_l766_766676

-- Define the initial conditions
def uncle_height : ℕ := 72
def james_initial_height : ℕ := (2 * uncle_height) / 3
def sarah_initial_height : ℕ := (3 * james_initial_height) / 4

-- Define the growth spurts
def james_growth_spurt : ℕ := 10
def sarah_growth_spurt : ℕ := 12

-- Define their heights after growth spurts
def james_final_height : ℕ := james_initial_height + james_growth_spurt
def sarah_final_height : ℕ := sarah_initial_height + sarah_growth_spurt

-- Define the combined height of James and Sarah after growth spurts
def combined_height : ℕ := james_final_height + sarah_final_height

-- Define the combined height difference between uncle and both James and Sarah now
def combined_height_difference : ℕ := combined_height - uncle_height

-- Lean statement to prove the combined height difference
theorem combined_height_difference_is_correct : combined_height_difference = 34 := by
  -- proof omitted
  sorry

end combined_height_difference_is_correct_l766_766676


namespace sequence_formula_l766_766185

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 2 * ∑ i in finset.range n, sequence i

theorem sequence_formula (n : ℕ) : sequence n = 
  match n with
  | 0   => 1
  | k + 1 => 2 * 3 ^ k
  end :=
sorry

end sequence_formula_l766_766185


namespace greatest_diff_units_digit_of_multiple_of_4_l766_766765

theorem greatest_diff_units_digit_of_multiple_of_4 : 
  let valid_units_digits := { d : ℕ // 0 ≤ d ∧ d ≤ 9 ∧ (720 + d) % 4 = 0 } in
  ∃ d1 d2 ∈ valid_units_digits, d1 ≠ d2 ∧ d1 - d2 = 8 :=
by
  sorry

end greatest_diff_units_digit_of_multiple_of_4_l766_766765


namespace projectile_time_to_meet_l766_766825

theorem projectile_time_to_meet
  (d v1 v2 : ℝ)
  (hd : d = 1455)
  (hv1 : v1 = 470)
  (hv2 : v2 = 500) :
  (d / (v1 + v2)) * 60 = 90 := by
  sorry

end projectile_time_to_meet_l766_766825


namespace periodic_decimal_sum_l766_766548

theorem periodic_decimal_sum : (0.2).periodic + (0.02).periodic = 8 / 33 :=
by
  -- Defining the periodic decimal representations in terms of their fractional forms:
  let a : ℚ := 2 / 9 -- equivalent to 0.\overline{2}
  let b : ℚ := 2 / 99 -- equivalent to 0.\overline{02}
  -- Asserting the sum of these fractions:
  have h : a + b = 24 / 99, by sorry
  -- Reducing the fraction to its lowest terms:
  have h' : 24 / 99 = 8 / 33, by sorry
  -- Concluding the equality:
  exact Eq.trans h h'

end periodic_decimal_sum_l766_766548


namespace sum_of_roots_tan_equation_l766_766948

theorem sum_of_roots_tan_equation :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π ∧ (tan x)^2 - 5 * tan x + 6 = 0 →
  x = arctan 3 ∨ x = arctan 2 →
  x + x = arctan 3 + arctan 2 :=
begin
  sorry
end

end sum_of_roots_tan_equation_l766_766948


namespace arithmetic_expression_value_l766_766803

theorem arithmetic_expression_value : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_expression_value_l766_766803


namespace fibonacci_sequence_x_l766_766309

theorem fibonacci_sequence_x {a : ℕ → ℕ} 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2) 
  (h3 : a 3 = 3) 
  (h_fib : ∀ n, n ≥ 3 → a (n + 1) = a n + a (n - 1)) : 
  a 5 = 8 := 
sorry

end fibonacci_sequence_x_l766_766309


namespace right_triangle_area_l766_766877

theorem right_triangle_area :
  ∃ (a b c : ℕ), a + b + c = 12 ∧ a * a + b * b = c * c ∧ (1/2 : ℝ) * a * b = 6 := 
sorry

end right_triangle_area_l766_766877


namespace marble_probability_l766_766437

theorem marble_probability (green purple white total favorable: ℕ) (h_green: green = 4) (h_purple: purple = 3) (h_white: white = 6) (h_total: total = green + purple + white) (h_favorable: favorable = green + purple) :
  (favorable : ℚ) / total = 7 / 13 :=
by
  rw [h_green, h_purple, h_white] at h_total h_favorable
  norm_num at h_total h_favorable
  rw [h_favorable, h_total]
  norm_num

end marble_probability_l766_766437


namespace payback_period_is_165_days_l766_766623

namespace CryptoMining

-- Initial conditions
def cost_system_unit : ℕ := 9499
def cost_graphics_card : ℕ := 31431
def num_graphics_cards : ℕ := 2
def power_system_unit : ℕ := 120
def power_graphics_card : ℕ := 125
def daily_earnings_per_card : ℚ := 0.00877
def value_of_ethereum : ℚ := 27790.37
def electricity_cost : ℚ := 5.38

-- Derived calculations
def total_cost : ℕ := cost_system_unit + num_graphics_cards * cost_graphics_card
def daily_earnings_in_rubles : ℚ := num_graphics_cards * daily_earnings_per_card * value_of_ethereum
def total_power_consumption_kw : ℚ := (power_system_unit + num_graphics_cards * power_graphics_card) / 1000
def daily_energy_consumption_kwh : ℚ := total_power_consumption_kw * 24
def daily_electricity_cost : ℚ := daily_energy_consumption_kwh * electricity_cost
def daily_profit : ℚ := daily_earnings_in_rubles - daily_electricity_cost

-- Proof that the payback period is 165 days
theorem payback_period_is_165_days : (total_cost : ℚ) / daily_profit ≈ 165 := by
  sorry

end CryptoMining

end payback_period_is_165_days_l766_766623


namespace simplify_radical_expression_l766_766359

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l766_766359


namespace intersection_point_l766_766052

theorem intersection_point :
  ∃ (x y : ℚ), (3 * y = -2 * x + 6) ∧ (-2 * y = 6 * x + 4) ∧ (x = -12 / 7) ∧ (y = 22 / 7) :=
by
  use -12 / 7, 22 / 7
  split
  sorry

end intersection_point_l766_766052


namespace distance_from_remaining_vertex_l766_766417

theorem distance_from_remaining_vertex (a : ℝ) (d1 d2 d3 d4 : ℝ) (h1 : d1 = 2) (h2 : d2 = 3) 
                                      (h3 : d3 = 7) (h : {d4 | d4 + 2 = 10 ∨ d4 + 3 = 9 ∨ d4 + 7 = 5}):
  d4 = 8 ∨ d4 = 6 :=
by
  intro d4 h4
  cases h4 with h4a h4a
  case h4a => sorry
  case h4a => sorry
  case h4a => sorry

end distance_from_remaining_vertex_l766_766417


namespace fraction_of_odd_products_is_025_l766_766663

-- Define the condition of the table
def in_table (a b : ℕ) : Prop := a < 16 ∧ b < 16

-- Define the condition for being odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Problem: What fraction of the products in a 16x16 table are odd
theorem fraction_of_odd_products_is_025 :
  let num_products := 16 * 16 in
  let odd_count := (finset.filter (λ ab : ℕ × ℕ, is_odd (ab.1 * ab.2)) 
                     (finset.product (finset.range 16) (finset.range 16))).card in
  (odd_count : ℚ) / num_products = 0.25 :=
by
  sorry

end fraction_of_odd_products_is_025_l766_766663


namespace problem_solution_l766_766966

theorem problem_solution 
  (θ : ℝ)
  (h : sin (π / 2 + θ) = -1 / 2):
  2 * sin (θ / 2) ^ 2 - 1 = 1 / 2 :=
sorry

end problem_solution_l766_766966


namespace sqrt_simplification_l766_766368

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l766_766368


namespace inversely_proportional_ratio_l766_766393

theorem inversely_proportional_ratio (x y x1 x2 y1 y2 : ℝ) 
  (h_inv_prop : x * y = x1 * y2) 
  (h_ratio : x1 / x2 = 3 / 5) 
  (x1_nonzero : x1 ≠ 0) 
  (x2_nonzero : x2 ≠ 0) 
  (y1_nonzero : y1 ≠ 0) 
  (y2_nonzero : y2 ≠ 0) : 
  y1 / y2 = 5 / 3 := 
sorry

end inversely_proportional_ratio_l766_766393


namespace marion_score_correct_l766_766758

-- Definitions based on conditions
def total_items : ℕ := 40
def ella_incorrect : ℕ := 4
def ella_correct : ℕ := total_items - ella_incorrect
def marion_score : ℕ := (ella_correct / 2) + 6

-- Statement of the theorem
theorem marion_score_correct : marion_score = 24 :=
by
  -- proof goes here
  sorry

end marion_score_correct_l766_766758


namespace units_digit_of_product_of_first_three_positive_composite_numbers_l766_766066

theorem units_digit_of_product_of_first_three_positive_composite_numbers :
  (4 * 6 * 8) % 10 = 2 :=
by sorry

end units_digit_of_product_of_first_three_positive_composite_numbers_l766_766066


namespace distinct_and_divisibility_l766_766294

open Nat

theorem distinct_and_divisibility (p : ℕ) (a : ℕ) 
  (h1 : Prime p) 
  (h2 : p % 2 = 1)
  (h3 : a < p) 
  (h4 : (a^2 + 1) % p = 0) : 
  a ≠ p - a ∧ P(a) = p ∧ P(p - a) = p := 
by {
  sorry
}

end distinct_and_divisibility_l766_766294


namespace largest_perfect_square_factor_of_3402_l766_766057

theorem largest_perfect_square_factor_of_3402 :
  ∃ (n : ℕ), n^2 ∣ 3402 ∧ (∀ m : ℕ, m^2 ∣ 3402 → m^2 ≤ n^2) :=
begin
  use 3, -- n is 3
  split,
  {
    norm_num,
    rw [mul_comm, nat.dvd_prime_pow (dec_trivial : prime 3)] ; [norm_num, dec_trivial],
  },
  {
    intros m h,
    have h_dvds, from nat.prime_dvd_prime_pow (dec_trivial : prime 3) h,
    cases h_dvds,
    {
      exact h_dvds.symm ▸ nat.le_refl _,
    },
    {
      suffices : 1 ≤ 3, from h_dvds.symm ▸ nat.pow_le_pow_of_le_left this 2,
      norm_num,
    }
  }
end

end largest_perfect_square_factor_of_3402_l766_766057


namespace BillCookingTime_l766_766519

-- Definitions corresponding to the conditions
def chopTimePepper : Nat := 3  -- minutes to chop one pepper
def chopTimeOnion : Nat := 4   -- minutes to chop one onion
def grateTimeCheese : Nat := 1 -- minutes to grate cheese for one omelet
def cookTimeOmelet : Nat := 5  -- minutes to assemble and cook one omelet

def numberOfPeppers : Nat := 4  -- number of peppers Bill needs to chop
def numberOfOnions : Nat := 2   -- number of onions Bill needs to chop
def numberOfOmelets : Nat := 5  -- number of omelets Bill prepares

-- Calculations based on conditions
def totalChopTimePepper : Nat := numberOfPeppers * chopTimePepper
def totalChopTimeOnion : Nat := numberOfOnions * chopTimeOnion
def totalGrateTimeCheese : Nat := numberOfOmelets * grateTimeCheese
def totalCookTimeOmelet : Nat := numberOfOmelets * cookTimeOmelet

-- Total preparation and cooking time
def totalTime : Nat := totalChopTimePepper + totalChopTimeOnion + totalGrateTimeCheese + totalCookTimeOmelet

-- Theorem statement
theorem BillCookingTime :
  totalTime = 50 := by
  sorry

end BillCookingTime_l766_766519


namespace compute_expression_l766_766896

-- Defining notation for the problem expression and answer simplification
theorem compute_expression : 
    9 * (2/3)^4 = 16/9 := by 
  sorry

end compute_expression_l766_766896


namespace max_odd_sum_lemma_l766_766763

noncomputable def max_odd_sum_300 (a : Fin 10 → ℕ) : ℕ :=
  a 0 + a 2 + a 4 + a 6 + a 8

theorem max_odd_sum_lemma (a : Fin 10 → ℕ) (h : ∀ i j, i < j → a i < a j) (sum_eq_300: ∑ i, a i = 300) :
  max_odd_sum_300 a = 147 :=
sorry

end max_odd_sum_lemma_l766_766763


namespace perimeter_new_figure_l766_766390

theorem perimeter_new_figure 
  (ABCD_square : ∀ {x y : Type}, square x y → side_length x = 20)
  (BFC_triangle : ∀ {x : Type}, equilateral_triangle x → side_length x = 20)
  (rotation : ∀ {x y : Type}, rotate x y 60) :
  perimeter new_figure = 120 :=
by
  sorry

end perimeter_new_figure_l766_766390


namespace inequality_x_solution_l766_766645

theorem inequality_x_solution (a b c d x : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  ( (a^3 / (a^3 + 15 * b * c * d))^(1/2) = a^x / (a^x + b^x + c^x + d^x) ) ↔ x = 15 / 8 := 
sorry

end inequality_x_solution_l766_766645


namespace find_even_digits_solution_l766_766454

theorem find_even_digits_solution (a b : ℕ) (h₁ : a ≥ 2) (h₂ : b ≥ 2) (h₃ : a % 2 = 0) :
    (∀ n : ℕ, a^b + 1 = n * (10^((a^b).digits.length) - 1) / 9) →
    (a = 2 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 5) ∨ (a = 6 ∧ b = 5) :=
by
    intro h_digits
    sorry

end find_even_digits_solution_l766_766454


namespace probability_of_exactly_one_shortening_l766_766859

-- Define the conditions and problem
def shorten_sequence_exactly_one_probability (n : ℕ) (p : ℝ) : ℝ :=
  nat.choose n 1 * p^1 * (1 - p)^(n - 1)

-- Given conditions
def problem_8a := shorten_sequence_exactly_one_probability 2014 0.1 ≈ 1.564e-90

-- Lean statement
theorem probability_of_exactly_one_shortening :
  problem_8a := sorry

end probability_of_exactly_one_shortening_l766_766859


namespace min_jumps_to_point_l766_766097

-- Define the jump condition: a jump is valid if it has length 4
def valid_jump (a b : ℤ) : Prop := a^2 + b^2 = 16

-- Define a valid point on the plane that the frog can land based on jumps
def valid_point (origin target : ℤ × ℤ) (steps : list (ℤ × ℤ)) : Prop :=
  steps ≠ [] ∧
  (steps.head = target → list.foldl (λ (p : ℤ × ℤ) (d : ℤ × ℤ), (p.1 + d.1, p.2 + d.2)) origin steps = target)

-- The minimum number of jumps required to reach (6,2) from (0,0)
theorem min_jumps_to_point : ∃ steps : list (ℤ × ℤ), valid_point (0,0) (6,2) steps ∧ steps.length = 2 :=
sorry

end min_jumps_to_point_l766_766097


namespace find_F_l766_766588

theorem find_F (F : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → (x = 0) ∧ (y = 1 ∨ y = -1)) ∧ 
  (∀ x y : ℝ, x^2 + y^2 - 6*x - 8*y + F = 0 → (x = 3) ∧ (y = 4 ∨ y = -4)) ∧ 
  (abs (sqrt ((3 - 0)^2 + (4 - 0)^2) - 1) = abs (sqrt (25 - F) - 1)) 
  → F = -11 :=
by 
  sorry

end find_F_l766_766588


namespace problem_l766_766284

def f (x : ℝ) (a b c d : ℝ) : ℝ := a * x^7 + b * x^5 - c * x^3 + d * x + 3

theorem problem (a b c d : ℝ) (h : f 92 a b c d = 2) : f 92 a b c d + f (-92) a b c d = 6 :=
by
  sorry

end problem_l766_766284


namespace rectangular_hyperbola_of_ellipse_l766_766946

theorem rectangular_hyperbola_of_ellipse (a b c : ℝ) (h1 : a = 5) (h2 : b = 3)
    (h3 : c = Real.sqrt (a^2 - b^2)) (h4 : c = 4)
    (h5 : (∀ x y : ℝ, ∀ ε : ℝ > 0, ∃ δ : ℝ > 0,
           (Real.sqrt (x^2 + y^2) < δ) → (Real.sqrt ((ε * x)^2 + (ε * y)^2) < 1))) :
    ∃ k : ℝ, (k = 8) ∧ (∀ x y : ℝ, x^2 / k - y^2 / k = 1) :=
by
  sorry

end rectangular_hyperbola_of_ellipse_l766_766946


namespace evaluate_expression_l766_766800

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end evaluate_expression_l766_766800


namespace find_j_tangent_ellipse_l766_766949

  theorem find_j_tangent_ellipse (j : ℝ) :
    (∃ (x y : ℝ), 4 * x - 7 * y + j = 0 ∧ x^2 + 4 * y^2 = 16 ∧ (7 * y - j)^2 / 16 + 4 * y^2 = 16) ↔ 
    (j = real.sqrt 450.5 ∨ j = -real.sqrt 450.5) := 
  sorry
  
end find_j_tangent_ellipse_l766_766949


namespace grapes_original_weight_l766_766646

-- Define the conditions
def grapes_water_percentage := 0.91
def raisins_water_percentage := 0.19
def raisins_weight_kg := 9
def non_water_content_weight_grapes (G : ℝ) : ℝ := (1 - grapes_water_percentage) * G
def non_water_content_weight_raisins (R : ℝ) : ℝ := (1 - raisins_water_percentage) * R

-- Main theorem: the original weight of the grapes, given that the current weight of raisins is 9 kilograms.
theorem grapes_original_weight (G : ℝ) (R : ℝ) :
    non_water_content_weight_grapes G = non_water_content_weight_raisins R →
    R = raisins_weight_kg →
    G = 81 :=
by sorry

end grapes_original_weight_l766_766646


namespace sum_50th_row_l766_766131

-- Define the sequence f(n)
def f : ℕ → ℕ
| 0     := 0
| 1     := 18
| (n+2) := 4 * f (n+1)

-- The goal is to prove that for the 50th row, f(50) equals 18 * 4^49
theorem sum_50th_row : f 50 = 18 * 4^49 :=
by
  sorry

end sum_50th_row_l766_766131


namespace transform_circle_to_ellipse_l766_766426

theorem transform_circle_to_ellipse (x y x'' y'' : ℝ) (h_circle : x^2 + y^2 = 1)
  (hx_trans : x = x'' / 2) (hy_trans : y = y'' / 3) :
  (x''^2 / 4) + (y''^2 / 9) = 1 :=
by {
  sorry
}

end transform_circle_to_ellipse_l766_766426


namespace real_solution_count_l766_766944

theorem real_solution_count : 
  ∃ (n : ℕ), n = 1 ∧
    ∀ x : ℝ, 
      (3 * x / (x ^ 2 + 2 * x + 4) + 4 * x / (x ^ 2 - 4 * x + 4) = 1) ↔ (x = 2) :=
by
  sorry

end real_solution_count_l766_766944


namespace square_side_length_l766_766438

theorem square_side_length (A : ℕ) (h : A = 400) : ∃ s : ℕ, s * s = A ∧ s = 20 :=
by {
  use 20,
  split,
  {
    rw h,
    norm_num,
  },
  {
    norm_num,
  },
  sorry,
}

end square_side_length_l766_766438


namespace bricks_required_l766_766480

   -- Definitions from the conditions
   def courtyard_length_meters : ℝ := 42
   def courtyard_width_meters : ℝ := 22
   def brick_length_cm : ℝ := 16
   def brick_width_cm : ℝ := 10

   -- The Lean statement to prove
   theorem bricks_required : (courtyard_length_meters * courtyard_width_meters * 10000) / (brick_length_cm * brick_width_cm) = 57750 :=
   by 
       sorry
   
end bricks_required_l766_766480


namespace find_fourth_root_l766_766787

-- Define the polynomial P(x)
def P (x : ℝ) (a b : ℝ) : ℝ := b * x^3 + (3 * b + a) * x^2 + (a - 2 * b) * x + (5 - b)

-- Define the known roots
def known_roots (x : ℝ) := x = -1 ∨ x = 2 ∨ x = 4 ∨ x = -8

-- Prove the fourth root is -8
theorem find_fourth_root (a b : ℝ) (hx₁ : P (-1) a b = 0) (hx₂ : P 2 a b = 0) 
  (hx₃ : P 4 a b = 0) (hx₄ : P (-8) a b = 0) : known_roots (-8) :=
by {
  -- Proof would go here, but we include sorry since it is not required to solve explicitly.
  sorry
}

end find_fourth_root_l766_766787


namespace incorrect_option_C_l766_766881

-- Definitions of increasing and decreasing functions
def increasing (f : ℝ → ℝ) := ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≤ f x₂
def decreasing (f : ℝ → ℝ) := ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≥ f x₂

-- The incorrectness of option C
theorem incorrect_option_C (f g : ℝ → ℝ) 
  (h₁ : increasing f) 
  (h₂ : decreasing g) : ¬ increasing (fun x => f x + g x) := 
sorry

end incorrect_option_C_l766_766881


namespace watch_selling_prices_l766_766495

-- Given conditions
variable (CP1 CP2 CP3 : ℕ)
variable (P1 P2 P3 : ℕ)

-- Given values
axiom cp1_val : CP1 = 1400
axiom p1_val : P1 = 5
axiom cp2_val : CP2 = 1800
axiom p2_val : P2 = 8
axiom cp3_val : CP3 = 2500
axiom p3_val : P3 = 12

-- Definition of Selling Price
def selling_price (CP : ℕ) (P : ℕ) : ℕ := CP + (CP * P / 100)

-- Correct answers to prove
theorem watch_selling_prices :
  selling_price CP1 P1 = 1470 ∧
  selling_price CP2 P2 = 1944 ∧
  selling_price CP3 P3 = 2800 :=
by
  simp [selling_price, cp1_val, p1_val, cp2_val, p2_val, cp3_val, p3_val]
  sorry

end watch_selling_prices_l766_766495


namespace complete_the_square_l766_766072

theorem complete_the_square (x : ℝ) : x^2 + 6 * x + 3 = 0 ↔ (x + 3)^2 = 6 := 
by
  sorry

end complete_the_square_l766_766072


namespace complex_quadrant_l766_766286

noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := (2 * i) / (1 + i)

theorem complex_quadrant : ∃ q : ℚ, q = 1 := by
  have h : z = 1 + i := sorry
  sorry

end complex_quadrant_l766_766286


namespace f_neg_m_equals_neg_8_l766_766599

def f (x : ℝ) : ℝ := x^5 + x^3 + 1

theorem f_neg_m_equals_neg_8 (m : ℝ) (h : f m = 10) : f (-m) = -8 :=
by
  sorry

end f_neg_m_equals_neg_8_l766_766599


namespace PR_length_l766_766258

-- Define the conditions under which the theorem should hold
variables {P Q R S : Type} [linear_order P] [linear_order Q] [linear_order R] [linear_order S]

-- Given values from the problem
def PQ : ℝ := 8
def QR : ℝ := 14
def RS : ℝ := 12

-- Set up the problem in Lean 4
theorem PR_length (M: Type) (PM: ℝ) : 
  (QR / 2 = 7) ∧ (PM = QR / 2) ∧ 
  (PQ' = PQ) ∧ (QR' = QR) ∧ 
  (RS / 2 = 6) ∧ (PR = 2 * (RS / 2)) → 
  PR = 12 :=
sorry

end PR_length_l766_766258


namespace exists_element_in_union_l766_766694

open Set

variables {α : Type} [Fintype α]

theorem exists_element_in_union 
  (S : Finset (Finset α))
  (n : ℕ)
  (A : Fin n → Finset α)
  (hS : ∀ (i j : Fin n), A i ∪ A j ∈ S)
  (h_min_card : ∀ i, (A i).card ≥ 2)
  (hS_eq : S = Finset.univ.image A) :
  ∃ x ∈ ⋃ i, A i, ∃ (m ≥ n / Inf (Finset.image (λ i, (A i).card) Finset.univ)), 
  ∀ k < m, x ∈ A k := sorry

end exists_element_in_union_l766_766694


namespace determine_guilty_resident_l766_766785

structure IslandResident where
  name : String
  is_guilty : Bool
  is_knight : Bool
  is_liar : Bool
  is_normal : Bool -- derived condition: ¬is_knight ∧ ¬is_liar

def A : IslandResident := { name := "A", is_guilty := false, is_knight := false, is_liar := false, is_normal := true }
def B : IslandResident := { name := "B", is_guilty := true, is_knight := true, is_liar := false, is_normal := false }
def C : IslandResident := { name := "C", is_guilty := false, is_knight := false, is_liar := true, is_normal := false }

-- Condition: Only one of them is guilty.
def one_guilty (A B C : IslandResident) : Prop :=
  A.is_guilty ≠ B.is_guilty ∧ A.is_guilty ≠ C.is_guilty ∧ B.is_guilty ≠ C.is_guilty ∧ (A.is_guilty ∨ B.is_guilty ∨ C.is_guilty)

-- Condition: The guilty one is a knight.
def guilty_is_knight (A B C : IslandResident) : Prop :=
  (A.is_guilty → A.is_knight) ∧ (B.is_guilty → B.is_knight) ∧ (C.is_guilty → C.is_knight)

-- Statements made by each resident.
def statements_made (A B C : IslandResident) : Prop :=
  (A.is_guilty = false) ∧ (B.is_guilty = false) ∧ (B.is_normal = false)

theorem determine_guilty_resident (A B C : IslandResident) :
  one_guilty A B C →
  guilty_is_knight A B C →
  statements_made A B C →
  B.is_guilty ∧ B.is_knight :=
by
  sorry

end determine_guilty_resident_l766_766785


namespace simplify_expression_l766_766325

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l766_766325


namespace max_ab_externally_tangent_circles_l766_766731

open Real

noncomputable def C1 (x y a: ℝ) : Prop := (x - a)^2 + (y + 2)^2 = 4
noncomputable def C2 (x y b: ℝ) : Prop := (x + b)^2 + (y + 2)^2 = 1
noncomputable def externally_tangent (a b: ℝ) : Prop := a + b = 3

theorem max_ab_externally_tangent_circles (a b: ℝ) (x y: ℝ):
  (C1 x y a) → 
  (C2 x y b) → 
  externally_tangent a b →
  ∃ k : ℝ, k = 9/4 ∧ ab ≤ k :=
by
  intros _ _ _
  existsi 9/4 
  split
  . reflexivity
  . sorry

end max_ab_externally_tangent_circles_l766_766731


namespace boxes_in_pantry_l766_766682

theorem boxes_in_pantry (b p c: ℕ) (h: p = 100) (hc: c = 50) (g: b = 225) (weeks: ℕ) (consumption: ℕ)
    (total_birdseed: ℕ) (new_boxes: ℕ) (initial_boxes: ℕ) : 
    weeks = 12 → consumption = (100 + 50) * weeks → total_birdseed = 1800 →
    new_boxes = 3 → total_birdseed = b * 8 → initial_boxes = 5 :=
by
  sorry

end boxes_in_pantry_l766_766682


namespace number_is_nine_l766_766076

theorem number_is_nine (x : ℤ) (h : 3 * (2 * x + 9) = 81) : x = 9 :=
by
  sorry

end number_is_nine_l766_766076


namespace election_votes_l766_766249

theorem election_votes (V : ℝ) 
  (h1 : 0.15 * V = 0.15 * V)
  (h2 : 0.85 * V = 309400 / 0.65)
  (h3 : 0.65 * (0.85 * V) = 309400) : 
  V = 560000 :=
by {
  sorry
}

end election_votes_l766_766249


namespace greatest_difference_units_digit_l766_766769

/-- 
The three-digit integer of the form 72X is a multiple of 4.
Prove that the greatest possible difference between two of the possibilities for the units digit is 8
-/
theorem greatest_difference_units_digit : 
  ∃ (n m : ℕ), (720 ≤ n) ∧ (n ≤ 729) ∧ ((n % 4 = 0) ∧ m = n % 10) → (max m 0 - 0 = 8) := 
begin
  sorry,
end

end greatest_difference_units_digit_l766_766769


namespace unit_cubes_intersected_by_plane_l766_766483

theorem unit_cubes_intersected_by_plane :
  ∀ (c : cube) (length : ℝ),
    c.he == 4 →
    let s := 1 in
    let unit := unit_cube s in
    let large_cube := cube (4 * s) c.origin in
    let plane := plane.through_diag_and_bi large_cube in
    length = 4 →
    number_of_unit_cubes_intersected plane large_cube = 32 :=
by {
  intros,
  sorry
}

end unit_cubes_intersected_by_plane_l766_766483


namespace payback_period_correct_l766_766625

noncomputable def payback_period (cost_sys_unit : ℤ) (cost_gpu : ℤ) (num_gpus : ℤ)
  (power_sys_unit : ℕ) (power_gpu : ℕ) (mining_speed : ℚ)
  (earnings_per_gpu_per_day : ℚ) (eth_to_rub : ℚ) (electricity_cost_per_kwh : ℚ) : ℚ :=
let total_cost := cost_sys_unit + num_gpus * cost_gpu in
let total_daily_earnings := num_gpus * earnings_per_gpu_per_day * eth_to_rub in
let total_power_w := power_sys_unit + num_gpus * power_gpu in
let total_power_kw := (total_power_w : ℚ) / 1000 in
let daily_energy_consumption_kwh := total_power_kw * 24 in
let daily_electricity_cost := daily_energy_consumption_kwh * electricity_cost_per_kwh in
let daily_profit := total_daily_earnings - daily_electricity_cost in
total_cost / daily_profit

theorem payback_period_correct :
  payback_period 9499 31431 2 120 125 32 0.00877 27790.37 5.38 ≈ 165 := sorry

end payback_period_correct_l766_766625


namespace kilos_of_bananas_l766_766714

-- Define the conditions
def initial_money := 500
def remaining_money := 426
def cost_per_kilo_potato := 2
def cost_per_kilo_tomato := 3
def cost_per_kilo_cucumber := 4
def cost_per_kilo_banana := 5
def kilos_potato := 6
def kilos_tomato := 9
def kilos_cucumber := 5

-- Total cost of potatoes, tomatoes, and cucumbers
def total_cost_vegetables : ℕ := 
  (kilos_potato * cost_per_kilo_potato) +
  (kilos_tomato * cost_per_kilo_tomato) +
  (kilos_cucumber * cost_per_kilo_cucumber)

-- Money spent on bananas
def money_spent_on_bananas : ℕ := initial_money - remaining_money - total_cost_vegetables

-- The proof problem statement
theorem kilos_of_bananas : money_spent_on_bananas / cost_per_kilo_banana = 14 :=
by
  -- The sorry is a placeholder for the proof
  sorry

end kilos_of_bananas_l766_766714


namespace interval_with_three_buses_l766_766028

theorem interval_with_three_buses (interval_two_buses : ℕ) (total_route_time : ℕ) (bus_count : ℕ) : 
  interval_two_buses = 21 → total_route_time = 2 * interval_two_buses → bus_count = 3 → 
  (total_route_time / bus_count) = 14 :=
by
  intros h1 h2 h3
  rw [h1, h3, ← h2]
  simp
  sorry

end interval_with_three_buses_l766_766028


namespace lines_intersect_or_parallel_l766_766252

-- Define the conditions for the lines a, b, and c
variables {Point : Type} [Inhabited Point] (Line : Type) [Nonempty Line]
variables (a b c : Line)

-- Conditions
axiom not_in_same_plane : ¬ ∃ (P : Set Point), (∀ l ∈ {a, b, c}, ∃ (pts : Set Point), is_plane pts ∧ ∀ l', l' ∈ Line pts → l' = l)
axiom not_skew_a_b : ∃ (P : Set Point), is_plane P ∧ ∀ l ∈ {a, b}, ∃ (l' : Point → Point → Prop), l' ∈ P
axiom not_skew_a_c : ∃ (P : Set Point), is_plane P ∧ ∀ l ∈ {a, c}, ∃ (l' : Point → Point → Prop), l' ∈ P
axiom not_skew_b_c : ∃ (P : Set Point), is_plane P ∧ ∀ l ∈ {b, c}, ∃ (l' : Point → Point → Prop), l' ∈ P

-- Goal to prove
theorem lines_intersect_or_parallel : 
  (∃ p : Point, ∀ l ∈ {a, b, c}, l.contains p) ∨ (∀ l₁ l₂ ∈ {a, b, c}, l₁ ≠ l₂ → parallel l₁ l₂) :=
sorry

end lines_intersect_or_parallel_l766_766252


namespace xavier_travel_time_increase_l766_766449

theorem xavier_travel_time_increase (
  (speed_initial : ℝ) (speed_final : ℝ) (time_total_minutes : ℝ) (distance_pq : ℝ)
  (initial_time_minutes : ℝ) :
  speed_initial = 50 ∧
  speed_final = 60 ∧
  time_total_minutes = 48 ∧
  distance_pq = 52 ∧
  initial_time_minutes = 24) :
  true :=
begin
  sorry
end

end xavier_travel_time_increase_l766_766449


namespace independence_test_problems_l766_766595

def condition1 : Prop := "The cure rate of a drug for a certain disease"
def condition2 : Prop := "Whether there is a difference in the treatment of the same disease with two different drugs"
def condition3 : Prop := "The probability of smokers contracting lung disease"
def condition4 : Prop := "Whether the smoking population is related to gender"
def condition5 : Prop := "Whether internet cafes are related to juvenile crime"

def can_solve_using_independence_tests (c : Prop) : Prop :=
  c = condition2 ∨ c = condition4 ∨ c = condition5

theorem independence_test_problems :
  {c | can_solve_using_independence_tests c} = {condition2, condition4, condition5} :=
by
  -- proof goes here
  sorry

end independence_test_problems_l766_766595


namespace simplify_expression_l766_766331

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l766_766331


namespace B_arrives_first_l766_766862

-- Definitions of walking speed, running speed, and total distance
variables (x y S : ℝ) (hx : 0 < x) (hy : 0 < y) (hS : 0 < S)

theorem B_arrives_first (hx_ne_y : x ≠ y) : 
  let tA := (S * (x + y)) / (2 * x * y) in
  let tB := 2 * S / (x + y) in
  tA > tB :=
by
  let tA := (S * (x + y)) / (2 * x * y)
  let tB := 2 * S / (x + y)
  have H : tA - tB = (S * (x - y)^2) / (2 * x * y * (x + y))
  sorry

end B_arrives_first_l766_766862


namespace remainder_div_10007_1279_l766_766046

theorem remainder_div_10007_1279 :
  let a := 10007
  let b := 1279 in
  a % b = 1054 := sorry

end remainder_div_10007_1279_l766_766046


namespace area_of_triangle_PQD_l766_766665

theorem area_of_triangle_PQD (A B C D P Q : ℝ)
  (h_square : ∀ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    ∀ s : ℝ, s ^ 2 = 4 → dist a b = dist b c ∧ dist b c = dist c d ∧ dist c d = dist d a ∧ dist d a = dist a b)
  (h_equilateral : ∀ p q d : ℝ, dist p q = dist q d ∧ dist q d = dist d p ∧ dist d p = dist p q)
  (h_halves : ∀ a p b : ℝ, dist a p = dist p b)
  (h_bounds : ∃ x y : ℝ, dist a b = dist c d = x ∧ x = 2) :
  ∃ A_PQD, A_PQD = (5 * Real.sqrt 3) / 4 :=
by
  sorry

end area_of_triangle_PQD_l766_766665


namespace B_capital_investment_l766_766112

-- Define the conditions
variables (P x : ℝ)

def profit_share_B (P x : ℝ) : ℝ := (x / (20000 + x)) * P
def profit_share_A (P x : ℝ) : ℝ := (8000 / (20000 + x)) * P
def profit_share_C (P x : ℝ) : ℝ := (12000 / (20000 + x)) * P

-- Given values
axiom B_profit : profit_share_B P x = 1900
axiom diff_profit_A_C : profit_share_C P x - profit_share_A P x = 760

-- Prove that x is 10000
theorem B_capital_investment : x = 10000 :=
sorry

end B_capital_investment_l766_766112


namespace total_guitars_l766_766124

theorem total_guitars (Barbeck_guitars Steve_guitars Davey_guitars : ℕ) (h1 : Barbeck_guitars = 2 * Steve_guitars) (h2 : Davey_guitars = 3 * Barbeck_guitars) (h3 : Davey_guitars = 18) : Barbeck_guitars + Steve_guitars + Davey_guitars = 27 :=
by sorry

end total_guitars_l766_766124


namespace AB_is_diameter_of_Γ_l766_766589

-- Definitions necessary to structure the problem statement in Lean
variables (A B C D E F : Type)
variables [circumcircle : circle Γ A B C D] (AB CD AC BD : line)
variables (m n : ℝ) (λ μ : ℝ)

-- Assumptions based on the conditions above
axiom AB_not_parallel_CD : ¬ parallel AB CD
axiom AB_GT_CD : length AB > length CD
axiom intersect_AC_BD_at_E : intersects AC BD E
axiom project_E_onto_AB : projection E AB = F
axiom EF_bisects_CFD : bisects (line_through E F) (angle_at F C D)

-- Main theorem
theorem AB_is_diameter_of_Γ : diameter AB Γ :=
sorry

end AB_is_diameter_of_Γ_l766_766589


namespace integer_solution_probability_probability_is_one_third_l766_766221

open Real

theorem integer_solution_probability :
  ∃ (k : ℤ), -5 / 2 < (k : ℝ) ∧ (k : ℝ) ≤ 3 ∧ (2 * k + 5 : ℤ) > 0 ∧ k ≤ -1 :=
by {
  sorry
}

theorem probability_is_one_third :
  (∑ k in { -2, -1, 0, 1, 2, 3 | -5 / 2 < (k : ℝ) ∧ (k : ℝ) ≤ 3 ∧ (2 * k + 5 : ℤ) > 0 ∧ k ≤ -1 }, 1)
  / (∑ k in { -2, -1, 0, 1, 2, 3 | -5 / 2 < (k : ℝ) ∧ (k : ℝ) ≤ 3 ∧ (2 * k + 5 : ℤ) > 0 }, 1) = (1 / 3) :=
by {
  sorry
}

end integer_solution_probability_probability_is_one_third_l766_766221


namespace derivative_at_one_l766_766569

noncomputable def f (x : ℝ) : ℝ := (2^x) / (2 * (Real.log 2 - 1) * x)

theorem derivative_at_one : (derivative f 1) = 1 := by
  sorry

end derivative_at_one_l766_766569


namespace find_a_value_l766_766596

variable {a : ℝ}

def f (x : ℝ) : ℝ := if x ≤ 0 then 1 - x else a^x

theorem find_a_value (h : f 1 = f (-1)) : a = 2 :=
by
  -- Proof to be completed
  sorry

end find_a_value_l766_766596


namespace students_per_van_l766_766951

def number_of_boys : ℕ := 60
def number_of_girls : ℕ := 80
def number_of_vans : ℕ := 5

theorem students_per_van : (number_of_boys + number_of_girls) / number_of_vans = 28 := by
  sorry

end students_per_van_l766_766951


namespace total_water_capacity_l766_766867

-- Define the given conditions as constants
def numTrucks : ℕ := 5
def tanksPerTruck : ℕ := 4
def capacityPerTank : ℕ := 200

-- Define the claim as a theorem
theorem total_water_capacity :
  numTrucks * (tanksPerTruck * capacityPerTank) = 4000 :=
by
  sorry

end total_water_capacity_l766_766867


namespace find_a_plus_b_l766_766751

theorem find_a_plus_b (a b : ℝ) (h1 : 2 * a = -6) (h2 : a^2 - b = 4) : a + b = 2 :=
by
  sorry

end find_a_plus_b_l766_766751


namespace trigonometric_value_of_x_l766_766232

theorem trigonometric_value_of_x (x : ℝ) : 
  (sin (4 * real.pi * x / 180) * sin (5 * real.pi * x / 180) = - cos (4 * real.pi * x / 180) * cos (5 * real.pi * x / 180)) → 
  x = 10 :=
by
  sorry

end trigonometric_value_of_x_l766_766232


namespace division_rounded_l766_766720

theorem division_rounded (h : Real := 8 / 125) (r : Real := 0.064) : 
  Real.round (1000 * h) / 1000 = r := 
sorry

end division_rounded_l766_766720


namespace csc_diff_l766_766635

variable {x : ℝ}

theorem csc_diff (h : ∃ (a : ℝ), sin x = a ∧ cos x = a * (√(a * (1 / cos x)))) :
  csc x ^ 6 - csc x ^ 2 = (1 - cos x ^ 12) / (cos x ^ 18) :=
begin
  -- Proof to be added
  sorry
end

end csc_diff_l766_766635


namespace greatest_integer_x_l766_766049

theorem greatest_integer_x :
  ∃ (x : ℤ), (∀ (y : ℤ), (8 : ℝ) / 11 > (x : ℝ) / 15) ∧
    ¬ (8 / 11 > (x + 1 : ℝ) / 15) ∧
    x = 10 :=
by
  sorry

end greatest_integer_x_l766_766049


namespace Brittany_second_test_grade_is_83_l766_766125

theorem Brittany_second_test_grade_is_83
  (first_test_score : ℝ) (first_test_weight : ℝ) 
  (second_test_weight : ℝ) (final_weighted_average : ℝ) : 
  first_test_score = 78 → 
  first_test_weight = 0.40 →
  second_test_weight = 0.60 →
  final_weighted_average = 81 →
  ∃ G : ℝ, 0.40 * first_test_score + 0.60 * G = final_weighted_average ∧ G = 83 :=
by
  sorry

end Brittany_second_test_grade_is_83_l766_766125


namespace negative_values_count_l766_766956

theorem negative_values_count :
  {n : ℕ | ∃ x : ℤ, n = Int.natAbs ((x + 100).natAbs) ∧ x + 100 = n^2 ∧ x < 0}.finite
    → ∃ x_set : Finset ℤ, ∀ x ∈ x_set, (∃ n : ℕ, n > 0 ∧ x + 100 = n^2) ∧ x < 0 ∧ x_set.card = 9 :=
by
  sorry

end negative_values_count_l766_766956


namespace rectangle_property_l766_766577

theorem rectangle_property (x : ℝ) (h : 2 * ((x + 6) * (3 * x)) = 2 * (x + 6) + 2 * (3 * x) + 12):
  x ≈ 0.74 :=
sorry

end rectangle_property_l766_766577


namespace amy_homework_time_l766_766172

def total_problems (math_problems spelling_problems : Nat) : Nat := math_problems + spelling_problems

def time_to_finish_problems (total_problems problems_per_hour : Nat) : Nat := total_problems / problems_per_hour

theorem amy_homework_time 
  (math_problems : Nat) 
  (spelling_problems : Nat) 
  (problems_per_hour : Nat)
  (h_math : math_problems = 18)
  (h_spelling : spelling_problems = 6)
  (h_hourly_rate : problems_per_hour = 4) :
  time_to_finish_problems (total_problems math_problems spelling_problems) problems_per_hour = 6 :=
by
  rw [h_math, h_spelling, h_hourly_rate]
  simp [total_problems, time_to_finish_problems]
  sorry

end amy_homework_time_l766_766172


namespace faye_coloring_books_l766_766139

theorem faye_coloring_books (initial_books : ℕ) (gave_away : ℕ) (bought_more : ℕ) (h1 : initial_books = 34) (h2 : gave_away = 3) (h3 : bought_more = 48) : 
  initial_books - gave_away + bought_more = 79 :=
by
  sorry

end faye_coloring_books_l766_766139


namespace tangent_and_parallel_condition_l766_766296

variable {α : Type} [LinearOrder α]

-- Definitions of points A, B, C, D
variable (A B C D : α)

-- Definition for a convex quadrilateral
def is_convex_quadrilateral : Prop := -- (Given that a quadrilateral is convex, with specific properties omitted)
  sorry

-- Definition for a line being tangent to a circle
def is_tangent (line : α) (circle_diameter : α) : Prop := -- (Line is tangent if meets at exactly one point)
  sorry

-- Definition for lines being parallel
def parallel (line1 line2 : α) : Prop := -- (Two lines are parallel if they do not intersect)
  sorry

theorem tangent_and_parallel_condition :
  is_convex_quadrilateral A B C D →
  is_tangent CD (diameter (A, B)) →
  (is_tangent AB (diameter (C, D)) ↔ parallel BC AD) :=
by
  sorry

end tangent_and_parallel_condition_l766_766296


namespace cost_price_of_watch_l766_766503

variable (CP : ℝ)
variable (SP_loss SP_gain : ℝ)
variable (h1 : SP_loss = CP * 0.725)
variable (h2 : SP_gain = CP * 1.125)
variable (h3 : SP_gain - SP_loss = 275)

theorem cost_price_of_watch : CP = 687.50 :=
by
  sorry

end cost_price_of_watch_l766_766503


namespace probability_of_exactly_one_shortening_l766_766855

-- Define the conditions and problem
def shorten_sequence_exactly_one_probability (n : ℕ) (p : ℝ) : ℝ :=
  nat.choose n 1 * p^1 * (1 - p)^(n - 1)

-- Given conditions
def problem_8a := shorten_sequence_exactly_one_probability 2014 0.1 ≈ 1.564e-90

-- Lean statement
theorem probability_of_exactly_one_shortening :
  problem_8a := sorry

end probability_of_exactly_one_shortening_l766_766855


namespace two_connected_graph_squared_hamiltonian_l766_766640

open GraphTheory

variable (G : SimpleGraph V) [Fintype V]

def is_hamiltonian_cycle (H : SimpleGraph V) :=
  ∃ C : Set (Sym2 V), (C ⊆ H.edgeSet) ∧ (∀ v ∈ V, ∃ e ∈ C, v ∈ e) ∧ (∀ v ∈ V, v.adjDegrees.filter (λ d, d ∈ C) = 2)

theorem two_connected_graph_squared_hamiltonian (h2conn : G.isTwoConnected) :
  is_hamiltonian_cycle (G^2) :=
sorry

end two_connected_graph_squared_hamiltonian_l766_766640


namespace cryptarithmetic_puzzle_sol_l766_766265

theorem cryptarithmetic_puzzle_sol (A B C D : ℕ) 
  (h1 : A + B + C = D) 
  (h2 : B + C = 7) 
  (h3 : A - B = 1) : D = 9 := 
by 
  sorry

end cryptarithmetic_puzzle_sol_l766_766265


namespace alpha_not_rational_l766_766271

theorem alpha_not_rational (alpha : ℝ) (h : real.cos (real.pi * alpha / 180) = 1 / 3) : ¬ ∃ (m n : ℤ), n ≠ 0 ∧ alpha = m / n :=
sorry

end alpha_not_rational_l766_766271


namespace percentage_reduction_l766_766497

theorem percentage_reduction 
  (original_employees : ℝ)
  (new_employees : ℝ)
  (h1 : original_employees = 208.04597701149424)
  (h2 : new_employees = 181) :
  ((original_employees - new_employees) / original_employees) * 100 = 13.00 :=
by
  sorry

end percentage_reduction_l766_766497


namespace find_b_l766_766407

theorem find_b (b : ℝ) (tangent_condition : ∀ x y : ℝ, y = -2 * x + b → y^2 = 8 * x) : b = -1 :=
sorry

end find_b_l766_766407


namespace angle_in_second_quadrant_l766_766965

theorem angle_in_second_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.sin α > 0) : 
    α ∈ Set.Ioo (π / 2) π := 
    sorry

end angle_in_second_quadrant_l766_766965


namespace train_length_is_correct_l766_766108

noncomputable def length_of_train (train_speed : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) : ℝ :=
  let speed_m_s := train_speed * (1000 / 3600)
  let total_distance := speed_m_s * time_to_cross
  total_distance - bridge_length

theorem train_length_is_correct :
  length_of_train 36 24.198064154867613 132 = 109.98064154867613 :=
by
  sorry

end train_length_is_correct_l766_766108


namespace probability_sqrt_2_plus_sqrt_2_le_abs_v_plus_w_l766_766691

noncomputable def is_root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

def roots_of_unity (n : ℕ) : set ℂ :=
  {z | is_root_of_unity z n}

theorem probability_sqrt_2_plus_sqrt_2_le_abs_v_plus_w :
  ∀ (v w : ℂ), distinct v w
  ∧ v ∈ roots_of_unity 2017 ∧ w ∈ roots_of_unity 2017 →
  (√(2 + √2) ≤ abs (v + w)) → (1 / 4) :=
by sorry

end probability_sqrt_2_plus_sqrt_2_le_abs_v_plus_w_l766_766691


namespace scientific_notation_example_l766_766253

theorem scientific_notation_example : (8485000 : ℝ) = 8.485 * 10 ^ 6 := 
by 
  sorry

end scientific_notation_example_l766_766253


namespace altitudes_concurrent_l766_766467

-- Define the vertices of the triangle
variables (A B C H : ℝ^3)

-- Define that the point H is the intersection of altitudes from B to AC and C to AB in the triangle ABC
def intersection_of_altitudes (A B C H : ℝ^3) : Prop :=
  (H - B) ⬝ (A - C) = 0 ∧ (H - C) ⬝ (A - B) = 0 

-- Theorem: The three altitudes of triangle ABC are concurrent
theorem altitudes_concurrent (A B C : ℝ^3) : ∃ H : ℝ^3, intersection_of_altitudes A B C H :=
sorry

end altitudes_concurrent_l766_766467


namespace n_plus_5_divisible_by_6_l766_766100

theorem n_plus_5_divisible_by_6 (n : ℕ) (h1 : (n + 2) % 3 = 0) (h2 : (n + 3) % 4 = 0) : (n + 5) % 6 = 0 := 
sorry

end n_plus_5_divisible_by_6_l766_766100


namespace no_solutions_for_a3_plus_5b3_eq_2016_l766_766289

theorem no_solutions_for_a3_plus_5b3_eq_2016 (a b : ℤ) : a^3 + 5 * b^3 ≠ 2016 :=
by sorry

end no_solutions_for_a3_plus_5b3_eq_2016_l766_766289


namespace max_value_of_f_l766_766149

def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x

theorem max_value_of_f : ∃ m, (∀ x, f(x) ≤ m) ∧ m = 5 :=
by
  sorry

end max_value_of_f_l766_766149


namespace marbles_solution_l766_766425

def marbles_problem : Prop :=
  ∃ (marbles : Finset (Finset ℕ)), 
    marbles.card = 28 ∧ ∀ m ∈ marbles,
    (∃ (c ∈ {0, 1, 2, 3}), m = {c, c} ∨
     ∃ (c1 c2 ∈ {0, 1, 2, 3}), c1 ≠ c2 ∧ m = {c1, c2})

theorem marbles_solution : marbles_problem :=
sorry

end marbles_solution_l766_766425


namespace find_B_find_sin_A_find_sin_2A_minus_B_l766_766267

open Real

noncomputable def triangle_conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a * cos C + c * cos A = 2 * b * cos B) ∧ (7 * a = 5 * b)

theorem find_B (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) :
  B = π / 3 :=
sorry

theorem find_sin_A (a b c A B C : ℝ) (h : triangle_conditions a b c A B C)
  (hB : B = π / 3) :
  sin A = 3 * sqrt 3 / 14 :=
sorry

theorem find_sin_2A_minus_B (a b c A B C : ℝ) (h : triangle_conditions a b c A B C)
  (hB : B = π / 3) (hA : sin A = 3 * sqrt 3 / 14) :
  sin (2 * A - B) = 8 * sqrt 3 / 49 :=
sorry

end find_B_find_sin_A_find_sin_2A_minus_B_l766_766267


namespace angle_DEF_eq_60_l766_766699

-- Definitions for the equilateral triangle and the given segments.
variable (A B C D E F : Point)
variable (triangleABC : EquilateralTriangle A B C)
variable (onSegmentBC : PointOnSegment B C D)
variable (onSegmentCA : PointOnSegment C A E)
variable (onSegmentAB : PointOnSegment A B F)
variable (FA_eq_9 : FA = 9)
variable (AE_eq_6 : AE = 6)
variable (EC_eq_6 : EC = 6)
variable (CD_eq_4 : CD = 4)

-- The goal is to prove that the angle DEF is 60 degrees.
theorem angle_DEF_eq_60 : ∠DEF = 60 := by
  sorry

end angle_DEF_eq_60_l766_766699


namespace inequality_solution_l766_766141

theorem inequality_solution (x : ℝ) : 
  (1 / (x^2 + 4) > 5 / x + 21 / 10) ↔ x ∈ set.Ioo (-2) 0 :=
by sorry

end inequality_solution_l766_766141


namespace band_member_share_l766_766166

def num_people : ℕ := 500
def ticket_price : ℝ := 30
def band_share_percent : ℝ := 0.70
def num_band_members : ℕ := 4

theorem band_member_share : 
  (num_people * ticket_price * band_share_percent) / num_band_members = 2625 := by
  sorry

end band_member_share_l766_766166


namespace probability_sum_eq_4_l766_766613

-- Defining sets A and B
def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 2, 3}

-- Defining the function that checks if the sum of two elements equals 4
def sum_eq_4 (a b : ℕ) : Prop := a + b = 4

-- Counting the number of desirable outcomes
noncomputable def desired_outcomes : ℕ := (A.product B).filter (λ pair => sum_eq_4 pair.1 pair.2).card

-- Counting the total number of possible outcomes
noncomputable def total_outcomes : ℕ := (A.product B).card

-- Calculating the probability
noncomputable def probability : ℚ := desired_outcomes / total_outcomes

-- Stating the theorem
theorem probability_sum_eq_4 : probability = 1/3 := by
  sorry

end probability_sum_eq_4_l766_766613


namespace Sara_has_73_percent_of_dollar_l766_766721

def value_of_pennies (pennies : ℕ) : ℕ := pennies * 1
def value_of_nickels (nickels : ℕ) : ℕ := nickels * 5
def value_of_dime (dimes : ℕ) : ℕ := dimes * 10
def value_of_quarters (quarters : ℕ) : ℕ := quarters * 25

theorem Sara_has_73_percent_of_dollar 
    (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (quarters : ℕ)
    (h_pennies : pennies = 3) (h_nickels : nickels = 2) 
    (h_dimes : dimes = 1) (h_quarters : quarters = 2) :
    (value_of_pennies pennies + value_of_nickels nickels + value_of_dime dimes + value_of_quarters quarters) = 73 := 
by
  rw [h_pennies, h_nickels, h_dimes, h_quarters]
  simp [value_of_pennies, value_of_nickels, value_of_dime, value_of_quarters]
  sorry

end Sara_has_73_percent_of_dollar_l766_766721


namespace rabbits_eventually_stop_final_rooms_occupied_correctly_l766_766882

def has_rabbits (rooms : ℤ → ℕ) (n : ℤ) : Prop := rooms n > 0

def initial_state (rooms : ℤ → ℕ) : Prop :=
  (∀ k < 0, rooms k = 0) ∧
  (∀ k > 10, rooms k = 0) ∧
  (∀ k, 0 ≤ k ∧ k ≤ 10 → rooms k = k + 1)

def step (rooms : ℤ → ℕ) (new_rooms : ℤ → ℕ) : Prop :=
  ∀ n, new_rooms n = 
    (if has_rabbits rooms (n+1) then (rooms (n+1)) / 2 else 0) +
    (if has_rabbits rooms (n-1) then (rooms (n-1)) / 2 else 0)

def eventually_stopped (rooms : ℤ → ℕ) : Prop :=
  ∃ t : ℕ, ∀ t' ≥ t, rooms = step rooms rooms

def final_configuration (rooms : ℤ → ℕ) : Prop :=
  rooms = (λ n, if ((n ≥ -26 ∧ n ≤ 28) ∨ (n ≥ 30 ∧ n ≤ 40)) 
                 then 1 else 0)

theorem rabbits_eventually_stop : ∀ (rooms : ℤ → ℕ),
  initial_state rooms →
  (∃ new_rooms : ℤ → ℕ, step rooms new_rooms) →
  eventually_stopped rooms :=
sorry

theorem final_rooms_occupied_correctly : ∀ (rooms : ℤ → ℕ),
  initial_state rooms →
  eventually_stopped rooms →
  final_configuration rooms :=
sorry

end rabbits_eventually_stop_final_rooms_occupied_correctly_l766_766882


namespace find_f_2015_l766_766180

variable {R : Type} [Real R]

def f : R → R := sorry

axiom symm_about_two : ∀ x : R, f x = f (2 - x)
axiom f_neg_five : f (-5) = -2

theorem find_f_2015 : f 2015 = -2 :=
by
  sorry

end find_f_2015_l766_766180


namespace determine_value_of_c_l766_766604

theorem determine_value_of_c (b : ℝ) (h₁ : ∀ x : ℝ, 0 ≤ x^2 + x + b) (h₂ : ∃ m : ℝ, ∀ x : ℝ, x^2 + x + b < c ↔ x = m + 8) : 
    c = 16 :=
sorry

end determine_value_of_c_l766_766604


namespace intersection_points_l766_766045

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 15 * x - 15
noncomputable def parabola2 (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 12

theorem intersection_points :
  { p : ℝ × ℝ // parabola1 p.1 = p.2 ∧ parabola2 p.1 = p.2 } = 
    { ((-3 : ℝ), (57 : ℝ)), ((12 : ℝ), (237 : ℝ)) } :=
by
  sorry

end intersection_points_l766_766045


namespace pentagon_concurrent_lines_l766_766276

structure Point (α : Type) := (x : α) (y : α)

structure Pentagon (α : Type) := 
  (A B C D E : Point α)

structure IntersectionPoints (α : Type) :=
  (A' B' C' D' E' : Point α)

structure FurtherIntersectionPoints (α : Type) :=
  (A'' B'' C'' D'' E'' : Point α)

theorem pentagon_concurrent_lines {α : Type} [field α]
  (P : Pentagon α)
  (IntPts : IntersectionPoints α)
  (FurtherIntPts : FurtherIntersectionPoints α)
  (H1 : ∃ A' , (P.B = A' ∧ P.D = A') ∧ P.C = A')
  (H2 : ∃ B' , (P.C = B' ∧ P.E = B') ∧ P.A = B')
  (H3 : ∃ C' , (P.D = C' ∧ P.A = C') ∧ P.B = C')
  (H4 : ∃ D' , (P.E = D' ∧ P.B = D') ∧ P.C = D')
  (H5 : ∃ E' , (P.A = E' ∧ P.C = E') ∧ P.D = E')
  (H6 : ∃ A'' , (circle_intersection (P.A) (P.B) (D') = A'') ∧ circle_intersection (P.A) (P.C) (E') = A'')
  (H7 : ∃ B'' , (circle_intersection (P.B) (P.C) (E') = B'') ∧ circle_intersection (P.B) (P.D) (A') = B'')
  (H8 : ∃ C'' , (circle_intersection (P.C) (P.D) (A') = C'') ∧ circle_intersection (P.C) (P.E) (B') = C'')
  (H9 : ∃ D'' , (circle_intersection (P.D) (P.E) (B') = D'') ∧ circle_intersection (P.D) (P.A) (C') = D'')
  (H10: ∃ E'' , (circle_intersection (P.E) (P.A) (C') = E'') ∧ circle_intersection (P.E) (P.B) (D') = E'')
  : are_concurrent (P.A, A'') (P.B, B'') (P.C, C'') (P.D, D'') (P.E, E'') := sorry

def circle_intersection (A B D' : Point α) (C E A' : Point α) :=
  sorry

def are_concurrent (A : Point α, A'' : Point α) (B : Point α, B'' : Point α) (C : Point α, C'' : Point α) (D : Point α, D'' : Point α) (E : Point α, E'' : Point α) := sorry


end pentagon_concurrent_lines_l766_766276


namespace general_term_a_n_sum_T_n_l766_766207

theorem general_term_a_n (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = 2^(n+1) - 2)
  (ha : ∀ n, a n = if n = 1 then S 1 else S n - S (n-1)) :
  ∀ n, a n = 2^n :=
begin
  sorry
end

theorem sum_T_n (a b : ℕ → ℕ) (c : ℕ → ℚ) (T : ℕ → ℚ)
  (ha : ∀ n, a n = 2^n)
  (hb : ∀ n, b n = n)
  (hc : ∀ n, c n = b n / (a n : ℚ))
  (hT : ∀ n, T n = ∑ i in (range n), c (i + 1)) :
  ∀ n, T n = 2 - (n + 2) / (2^n) :=
begin
  sorry
end

end general_term_a_n_sum_T_n_l766_766207


namespace equation_of_line_l766_766916

-- Points A and B
def A := (-1, 1 : ℝ)
def B := (3, 9 : ℝ)

-- Definition of a line through two points
def line_through (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  let k := (p2.2 - p1.2) / (p2.1 - p1.1)
  y = k * (x - p1.1) + p1.2

-- Theorem stating that the line passing through A and B has the equation 2x - y + 3 = 0
theorem equation_of_line : ∀ x y : ℝ, line_through A B x y ↔ 2 * x - y + 3 = 0 :=
by
  sorry

end equation_of_line_l766_766916


namespace symmetric_complex_numbers_l766_766832

theorem symmetric_complex_numbers (z1 z2 : ℂ) (h1 : z1 = 2 - 3 * complex.I) (h2 : z2 = -z1) : z2 = -2 + 3 * complex.I :=
by {
  sorry
}

end symmetric_complex_numbers_l766_766832


namespace geom_seq_monotonic_decreasing_l766_766975

variable {α : Type*} [LinearOrderedField α]

def geom_sequence (a : ℕ → α) (q : α) := ∀ n, a (n + 1) = a n * q

def geom_sequence_sum (S : ℕ → α) (a : ℕ → α) := ∀ n, S n = (finset.range (n + 1)).sum a

theorem geom_seq_monotonic_decreasing {a : ℕ → α} {q : α} (h_geom : geom_sequence a q)
  (h_sum : geom_sequence_sum (λ n, a n) q) (h_a2 : a 1 = 12)
  (h_a3_a5 : a 2 * a 4 = 4) :
  ∀ n, a (2 * n) > a (2 * (n + 1)) :=
by 
  sorry

end geom_seq_monotonic_decreasing_l766_766975


namespace maximilian_wealth_greater_than_national_wealth_l766_766433

theorem maximilian_wealth_greater_than_national_wealth (x y z : ℝ) (h1 : 2 * x > z) (h2 : y < z) :
    x > (2 * x + y) - (x + z) :=
by
  sorry

end maximilian_wealth_greater_than_national_wealth_l766_766433


namespace gcd_1994_1995_l766_766085

def pow (a : ℕ) (b : ℕ) : ℕ := Nat.mulNatPow a b

theorem gcd_1994_1995 : Nat.gcd (1994 * pow 1994 1994 + pow 1994 (1994 + 1)) (1994 * 1995) = 1994 * 1995 := by
  sorry

end gcd_1994_1995_l766_766085


namespace triangular_weight_is_60_l766_766005

/-- Suppose there are weights: 5 identical round, 2 identical triangular, and 1 rectangular weight of 90 grams.
    The conditions are: 
    1. One round weight and one triangular weight balance three round weights.
    2. Four round weights and one triangular weight balance one triangular weight, one round weight, and one rectangular weight.
    Prove that the weight of the triangular weight is 60 grams. -/
theorem triangular_weight_is_60 
  (R T : ℕ)  -- We declare weights of round and triangular weights as natural numbers
  (h1 : R + T = 3 * R)  -- The first balance condition
  (h2 : 4 * R + T = T + R + 90)  -- The second balance condition
  : T = 60 := 
by
  sorry  -- Proof omitted

end triangular_weight_is_60_l766_766005


namespace max_value_of_3sinx_4cosx_is_5_l766_766154

def max_value_of_function (a b : ℝ) : ℝ :=
  (sqrt (a^2 + b^2))

theorem max_value_of_3sinx_4cosx_is_5 :
  max_value_of_function 3 4 = 5 :=
by
  sorry

end max_value_of_3sinx_4cosx_is_5_l766_766154


namespace regular_pyramid_sufficient_condition_l766_766416

-- Define the basic structure of a pyramid
structure Pyramid :=
  (lateral_face_is_equilateral_triangle : Prop)  
  (base_is_square : Prop)  
  (apex_angles_of_lateral_face_are_45_deg : Prop)  
  (projection_of_vertex_at_intersection_of_base_diagonals : Prop)
  (is_regular : Prop)

-- Define the hypothesis conditions
variables 
  (P : Pyramid)
  (h1 : P.lateral_face_is_equilateral_triangle)
  (h2 : P.base_is_square)
  (h3 : P.apex_angles_of_lateral_face_are_45_deg)
  (h4 : P.projection_of_vertex_at_intersection_of_base_diagonals)

-- Define the statement of the proof
theorem regular_pyramid_sufficient_condition :
  (P.lateral_face_is_equilateral_triangle → P.is_regular) ∧ 
  (¬(P.lateral_face_is_equilateral_triangle) → ¬P.is_regular) ↔
  (P.lateral_face_is_equilateral_triangle ∧ ¬P.base_is_square ∧ ¬P.apex_angles_of_lateral_face_are_45_deg ∧ ¬P.projection_of_vertex_at_intersection_of_base_diagonals) := 
by { sorry }


end regular_pyramid_sufficient_condition_l766_766416


namespace quadratic_real_roots_l766_766618

theorem quadratic_real_roots (k : ℝ) : 
  let Δ := (k + 4) ^ 2 - 4 * 1 * 4 * k 
  in Δ ≥ 0 :=
by
  let Δ := (k + 4) ^ 2 - 4 * 1 * 4 * k
  calc Δ = (k + 4) ^ 2 - 16 * k : by rfl
       ... = k^2 + 8 * k + 16 - 16 * k : by ring
       ... = k^2 - 8 * k + 16 : by ring
       ... = (k - 4)^2 : by ring
       ... ≥ 0 : by apply sq_nonneg

end quadratic_real_roots_l766_766618


namespace range_of_a_l766_766740

theorem range_of_a (a : ℝ) (h : a > 0) (f : ℝ → ℝ) (hf : ∀ x, f x = (1/3) * a * x^3 - x^2 + 5) :
  ¬ (monotone_on f (set.Ioo 0 2)) → a > 1 :=
sorry

end range_of_a_l766_766740


namespace tenth_term_arithmetic_sequence_l766_766065

theorem tenth_term_arithmetic_sequence :
  ∀ (a1 d : ℚ) (n : ℕ), a1 = 1/2 ∧ d = 1/6 ∧ n = 10 → a1 + (n-1) * d = 2 := 
by
  intros a1 d n h,
  rcases h with ⟨ha1, hd, hn⟩,
  rw [ha1, hd, hn],
  norm_num,
  exact rfl,

end tenth_term_arithmetic_sequence_l766_766065


namespace interval_with_three_buses_l766_766030

theorem interval_with_three_buses (interval_two_buses : ℕ) (total_route_time : ℕ) (bus_count : ℕ) : 
  interval_two_buses = 21 → total_route_time = 2 * interval_two_buses → bus_count = 3 → 
  (total_route_time / bus_count) = 14 :=
by
  intros h1 h2 h3
  rw [h1, h3, ← h2]
  simp
  sorry

end interval_with_three_buses_l766_766030


namespace largestPerfectSquareFactorOf3402_l766_766060

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem largestPerfectSquareFactorOf3402 :
  ∃ k : ℕ, isPerfectSquare k ∧ k ∣ 3402 ∧ ∀ m : ℕ, isPerfectSquare m ∧ m ∣ 3402 → m ≤ k := 
begin
  use 81,
  split,
  { use 9, exact rfl },
  split,
  { norm_num },
  { intros m h,
    cases h with hm hm',
    cases hm with x hx,
    rw hx at hm',
    by_cases h0 : x = 0,
    { subst h0, norm_num at hm', exact hm' },
    by_cases h1 : x = 9,
    { subst h1, norm_num at hm', exact hm' },
    norm_num at hm',
    sorry
  }
end

end largestPerfectSquareFactorOf3402_l766_766060


namespace shorten_by_one_expected_length_l766_766850

-- Definition of the random sequence and binomial distribution properties
def digit_sequence {n : ℕ} (p : ℝ) (len : ℕ) : ℝ := 
  let q := 1 - p in
  (len.choose 1) * (p^1 * q^(len - 1))

-- Part (a) statement
theorem shorten_by_one 
  (len : ℕ) 
  (p : ℝ := 0.1) 
  (q := 1 - p) 
  (x := len - 1) : 
  digit_sequence p x = 1.564e-90 :=
by sorry

-- Part (b) statement
theorem expected_length 
  (initial_len : ℕ) 
  (p : ℝ := 0.1) (removals := initial_len - 1) 
  (expected_removed : ℝ := removals * p) : 
  (initial_len : ℝ) - expected_removed = 1813.6 :=
by sorry

end shorten_by_one_expected_length_l766_766850


namespace proposition_correct_l766_766118

theorem proposition_correct :
  let p1 := (∀ (A B : ℝ), A > B ↔ sin A > sin B)
  let p2 := (∀ (f : ℝ → ℝ), (∃ x, 1 < x ∧ x < 2 ∧ f x = 0) ↔ f 1 * f 2 < 0)
  let p3 := (∀ (a_n : ℕ → ℝ), (a_n 1 = 1 ∧ a_n 5 = 16) → (a_n 3 = 4) ∨ (a_n 3 = -4))
  let p4 := (∀ (x : ℝ), sin (2 - 2 * x + 4) = sin (4 - 2 * x))
  (if p1 then true else false) &&
  (if p2 then true else false) = false &&
  (if p3 then true else false) = false &&
  (if p4 then true else false) = false := by
  sorry

end proposition_correct_l766_766118


namespace graph_properties_l766_766742

theorem graph_properties (x : ℝ) :
  (∃ p : ℝ × ℝ, p = (1, -7) ∧ y = -7 * x) ∧
  (x ≠ 0 → y * x < 0) ∧
  (x > 0 → y < 0) :=
by
  sorry

end graph_properties_l766_766742


namespace third_median_length_l766_766109

-- Proposition stating the problem with conditions and the conclusion
theorem third_median_length (m1 m2 : ℝ) (area : ℝ) (h1 : m1 = 4) (h2 : m2 = 5) (h_area : area = 10 * Real.sqrt 3) : 
  ∃ m3 : ℝ, m3 = 3 * Real.sqrt 10 :=
by
  sorry  -- proof is not included

end third_median_length_l766_766109


namespace parabola_properties_l766_766209

theorem parabola_properties :
  (∃ p : ℝ, 0 < p ∧ (∀ (x y : ℝ), (x, y) = (3, -2 * real.sqrt 6) → y^2 = 2 * p * x)) →
  let eq := ∀ (x y : ℝ), y^2 = 8 * x in
  let chord_length := ∃ (A B : ℝ × ℝ), A ≠ B ∧
                            2 * A.1 - A.2 - 3 = 0 ∧
                            2 * B.1 - B.2 - 3 = 0 ∧
                            4 * A.1^2 - 20 * A.1 + 9 = 0 ∧
                            4 * B.1^2 - 20 * B.1 + 9 = 0 ∧
                            ∀ x₁ x₂ x₁' x₂' y₁ y₂ y₁' y₂', A.1 = x₁ → B.1 = x₂ → A.2 = y₁ → B.2 = y₂ →
                            (x₁ + x₂ = 5 ∧ x₁ * x₂ = 9 / 4) →
                            abs (x₁ - x₂) * real.sqrt ((2^2 + 1) * abs (x₁ + x₂)^2 - 4 * x₁ * x₂) = 4 * real.sqrt 5
  in eq ∧ chord_length := sorry

end parabola_properties_l766_766209


namespace part1_part2_l766_766985

-- Definition of sets A, B, and Proposition p for Part 1
def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a = 0}
def p (a : ℝ) : Prop := ∀ x ∈ B a, x ∈ A

-- Part 1: Prove the range of a
theorem part1 (a : ℝ) : (p a) → 0 < a ∧ a ≤ 1 :=
  by sorry

-- Definition of sets A and C for Part 2
def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 3 > 0}
def necessary_condition (m : ℝ) : Prop := ∀ x ∈ A, x ∈ C m

-- Part 2: Prove the range of m
theorem part2 (m : ℝ) : necessary_condition m → m ≤ 7 / 2 :=
  by sorry

end part1_part2_l766_766985


namespace vinegar_used_is_15_l766_766656

noncomputable def vinegar_used (T : ℝ) : ℝ :=
  let water := (3 / 5) * 20
  let total_volume := 27
  let vinegar := total_volume - water
  vinegar

theorem vinegar_used_is_15 (T : ℝ) (h1 : (3 / 5) * 20 = 12) (h2 : 27 - 12 = 15) (h3 : (5 / 6) * T = 15) : vinegar_used T = 15 :=
by
  sorry

end vinegar_used_is_15_l766_766656


namespace quadrilateral_angle_sum_l766_766251

theorem quadrilateral_angle_sum (E F G H : ℝ) 
  (h1 : E = 3 * F)
  (h2 : E = 4 * G)
  (h3 : E = 6 * H)
  (h4 : E + F + G + H = 360) : 
  E ≈ 206 :=
begin
  sorry
end

end quadrilateral_angle_sum_l766_766251


namespace sum_of_face_values_l766_766499

-- Definitions
variables {H S D C : ℕ}
-- Conditions
def condition1 : Prop := S = 11 * H
def condition2 : Prop := C = D + 45

-- Proof Statement
theorem sum_of_face_values 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : H = 3)
  (h4 : D = 10)
  : H + S + D + C = 101 :=
by
  -- Derive values
  have hS : S = 33, by rw [condition1, h3]; norm_num,
  have hC : C = 55, by rw [condition2, h4]; norm_num,
  -- Calculate total sum
  rw [h3, hS, h4, hC]; norm_num

end sum_of_face_values_l766_766499


namespace fraction_to_decimal_l766_766911

theorem fraction_to_decimal : (7 / 32 : ℚ) = 0.21875 := 
by {
  sorry
}

end fraction_to_decimal_l766_766911


namespace man_swim_upstream_distance_l766_766486

def swimming_problem (c : ℝ) (d : ℝ) : Prop :=
  (11.5 + c) * 3 = 51 ∧ (11.5 - c) * 3 = d

theorem man_swim_upstream_distance (c d : ℝ) (h : swimming_problem c d) : d = 18 := by
  obtain ⟨downstream_eqn, upstream_eqn⟩ := h
  have current_speed : c = 5.5 := by
    linarith
  have upstream_swim : d = 18 := by
    rw [current_speed] at upstream_eqn
    linarith
  exact upstream_swim

end man_swim_upstream_distance_l766_766486


namespace sum_integer_coefficients_l766_766553

theorem sum_integer_coefficients (x y : ℂ) :
  let a := 256 * x^8
  let b := 2401 * y^8
  let c := 16 * (x^4) 
  let d := 49 * (y^4)
  a - b = 
    (4 * x^2 - 7 * y^2) * 
    (4 * x^2 + 7 * y^2) *
    (4 * x^2 + 7 * complex.I * y^2) * 
    (4 * x^2 - 7 * complex.I * y^2) →
  (4 + -7 + 4 + 7 + 4 + 4 = 16) := 
by
  sorry

end sum_integer_coefficients_l766_766553


namespace greatest_difference_units_digit_l766_766772

theorem greatest_difference_units_digit (d : Nat) 
  (h : d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h_mod4 : (72 % 100 + d) % 4 = 0) :
  ∃ a b, a ∈ {0, 4, 8} ∧ b ∈ {0, 4, 8} ∧ a ≠ b ∧ (abs (a - b) = 8) := 
begin
  -- Proof is omitted, only statement is required
  sorry
end

end greatest_difference_units_digit_l766_766772


namespace calculate_expression_value_l766_766798

theorem calculate_expression_value : 
  3 - ((-3 : ℚ) ^ (-3 : ℤ) * 2) = 83 / 27 := 
by
  sorry

end calculate_expression_value_l766_766798


namespace tangent_line_equation_l766_766921

open Real

noncomputable def circle_center : ℝ × ℝ := (2, 1)
noncomputable def tangent_point : ℝ × ℝ := (4, 3)

def circle_equation (x y : ℝ) : Prop := (x - circle_center.1)^2 + (y - circle_center.2)^2 = 1

theorem tangent_line_equation :
  ∀ (x y : ℝ), ( (x = 4 ∧ y = 3) ∨ circle_equation x y ) → 2 * x + 2 * y - 7 = 0 :=
sorry

end tangent_line_equation_l766_766921


namespace interval_with_three_buses_l766_766027

theorem interval_with_three_buses (interval_two_buses : ℕ) (total_route_time : ℕ) (bus_count : ℕ) : 
  interval_two_buses = 21 → total_route_time = 2 * interval_two_buses → bus_count = 3 → 
  (total_route_time / bus_count) = 14 :=
by
  intros h1 h2 h3
  rw [h1, h3, ← h2]
  simp
  sorry

end interval_with_three_buses_l766_766027


namespace final_segment_distance_l766_766913

theorem final_segment_distance :
  let north_distance := 2
  let east_distance := 1
  let south_distance := 1
  let net_north := north_distance - south_distance
  let net_east := east_distance
  let final_distance := Real.sqrt (net_north ^ 2 + net_east ^ 2)
  final_distance = Real.sqrt 2 :=
by
  sorry

end final_segment_distance_l766_766913


namespace smallest_number_3333377733_l766_766795

-- Define the required conditions and prove the given problem
def is_composed_of_3_and_7 (n : ℕ) : Prop :=
  ∀ d ∈ (Int.digits 10 (Int.ofNat n)), d = 3 ∨ d = 7

def divisible_by_3_and_7 (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 7 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (Int.digits 10 (Int.ofNat n)).sum

def smallest_number (n : ℕ) : Prop :=
  is_composed_of_3_and_7 n ∧ divisible_by_3_and_7 n ∧ divisible_by_3_and_7 (sum_of_digits n)

theorem smallest_number_3333377733 : ∃ n, smallest_number n ∧ n = 3333377733 :=
by
  have h : smallest_number 3333377733 := sorry -- proof goes here
  exact ⟨3333377733, ⟨h, rfl⟩⟩

end smallest_number_3333377733_l766_766795


namespace construct_line_segment_through_D_P_l766_766311

-- Assume existence of line segments s1 and s2
variables {s1 s2 : Set Point}

-- Assume existence of points P and D, where P is the intersection point of extensions of s1 and s2
-- and does not lie on the piece of paper, while D is an arbitrary point on the piece of paper
variable (P : Point)
variable (D : Point)
variable (s1_ext : extends s1)
variable (s2_ext : extends s2)

-- Definition of intersection point
def intersection (line1 line2 : Set Point) : Point := sorry

-- Assume conditions for intersection of extensions
axiom hyp1 : ∃ P, s1_ext ∩ s2_ext = {P} ∧ P ∉ piece_of_paper

-- The main theorem statement
theorem construct_line_segment_through_D_P : ∃ (E : Point), line_through D E ∧ E ∈ extension (line_through D P) := 
sorry

end construct_line_segment_through_D_P_l766_766311


namespace primitive_root_mod_p2_l766_766295

variable (p : ℕ) [hp : Fact (Nat.Prime p)] (x : ℕ)

-- Assume x is a primitive root modulo p, which implies gcd(x, p) = 1 and x^(p-1) ≡ 1 (mod p).
def is_primitive_root_mod_p (x p : ℕ) [Fact (Nat.Prime p)] : Prop :=
  Nat.gcd x p = 1 ∧ Nat.pow_mod x (p-1) p = 1

theorem primitive_root_mod_p2 (hp : Fact (Nat.Prime p)) (odd_p : p % 2 = 1) (hx : is_primitive_root_mod_p x p) :
  is_primitive_root_mod_p x (p^2) ∨ is_primitive_root_mod_p (x + p) (p^2) := sorry

end primitive_root_mod_p2_l766_766295


namespace total_expense_l766_766320

noncomputable def sandys_current_age : ℕ := 36 - 2
noncomputable def sandys_monthly_expense : ℕ := 10 * sandys_current_age
noncomputable def alexs_current_age : ℕ := sandys_current_age / 2
noncomputable def alexs_next_month_expense : ℕ := 2 * sandys_monthly_expense

theorem total_expense : 
  sandys_monthly_expense + alexs_next_month_expense = 1020 := 
by 
  sorry

end total_expense_l766_766320


namespace marble_203_is_green_l766_766500

-- Define the conditions
def total_marbles : ℕ := 240
def cycle_length : ℕ := 15
def red_count : ℕ := 6
def green_count : ℕ := 5
def blue_count : ℕ := 4
def marble_pattern (n : ℕ) : String :=
  if n % cycle_length < red_count then "red"
  else if n % cycle_length < red_count + green_count then "green"
  else "blue"

-- Define the color of the 203rd marble
def marble_203 : String := marble_pattern 202

-- State the theorem
theorem marble_203_is_green : marble_203 = "green" :=
by
  sorry

end marble_203_is_green_l766_766500


namespace dim_solution_space_l766_766555

noncomputable def linear_equation_solution_space : Type := sorry

theorem dim_solution_space :
  let eqs : list (Fin 5 → ℝ) :=
    [λ ⟨i, _⟩, match i with
              | 1 => 1
              | 2 => 2
              | 3 => -3
              | 4 => 0
              | 5 => 0
              end,
     λ ⟨i, _⟩, match i with
              | 1 => 2
              | 2 => -1
              | 3 => 3
              | 4 => 0
              | 5 => 4
              end,
     λ ⟨i, _⟩, match i with
              | 1 => 2
              | 2 => 0
              | 3 => 5
              | 4 => -3
              | 5 => 4
              end] in
  let basis_vectors : list (Fin 5 → ℝ) :=
    [λ ⟨i, _⟩, match i with
              | 1 => -5 / 2
              | 2 => -2
              | 3 => 1
              | 4 => 0
              | 5 => 0
              end,
     λ ⟨i, _⟩, match i with
              | 1 => 3 / 2
              | 2 => 3
              | 3 => 0
              | 4 => 1
              | 5 => 0
              end,
     λ ⟨i, _⟩, match i with
              | 1 => -2
              | 2 => 0
              | 3 => 0
              | 4 => 0
              | 5 => 1
              end] in
  let general_solution (C1 C2 C3 : ℝ) : Fin 5 → ℝ := 
    λ ⟨i, _⟩, C1 * (basis_vectors.head ⟨i, sorry⟩) 
               + C2 * (basis_vectors.tail.head ⟨i, sorry⟩) 
               + C3 * (basis_vectors.tail.tail.head ⟨i, sorry⟩) in
  linear_algebra.dimension eqs = 3 ∧
  ∀ (v : Fin 5 → ℝ), v ∈ vector_space_span basis_vectors ↔ v ∈ solution_space eqs :=
  sorry

end dim_solution_space_l766_766555


namespace triangle_problem_l766_766239

-- Definitions from conditions
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Sides opposite to angles A, B, C respectively

-- Given conditions
def condition1 : Prop := b * (Real.sin A) = a * (Real.sin (2 * B))
def condition2 : Prop := b = Real.sqrt 10
def condition3 : Prop := a + c = a * c

-- Proof of the main theorem
theorem triangle_problem : condition1 → condition2 → condition3 → 
                           B = Real.pi / 3 ∧ 
                           (1 / 2) * a * c * (Real.sin B) = (5 * Real.sqrt 3) / 4 :=
by 
  intros h1 h2 h3
  sorry

end triangle_problem_l766_766239


namespace simplify_expression_l766_766380

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l766_766380


namespace molecular_weight_N2O5_l766_766143

theorem molecular_weight_N2O5 :
  let atomic_weight_N := 14.01
  let atomic_weight_O := 16.00
  let molecular_weight_N2O5 := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  molecular_weight_N2O5 = 108.02 := 
by
  sorry

end molecular_weight_N2O5_l766_766143


namespace largest_divisor_m_squared_minus_n_squared_l766_766285

theorem largest_divisor_m_squared_minus_n_squared (n : ℤ) (h1 : ∃ b : ℤ, n = 2 * b) :
  let m := n + 1 in ∀ k : ℤ, (∀ b : ℤ, k ∣ (m^2 - n^2)) → k = 1 := by
  sorry

end largest_divisor_m_squared_minus_n_squared_l766_766285


namespace elegant_interval_solution_l766_766651

noncomputable def elegant_interval : ℝ → ℝ × ℝ := sorry

theorem elegant_interval_solution (m : ℝ) (a b : ℕ) (s : ℝ) (p : ℕ) :
  a < m ∧ m < b ∧ a + 1 = b ∧ 3 < s + b ∧ s + b ≤ 13 ∧ s = Real.sqrt a ∧ b * b + a * s = p → p = 33 ∨ p = 127 := 
by sorry

end elegant_interval_solution_l766_766651


namespace radius_of_circle_roots_l766_766537

theorem radius_of_circle_roots (z : ℂ) (h : (z - 2)^6 = 64 * z^6) : 
  ∀ z, |z - 2| = 2 * |z| → by sorry

end radius_of_circle_roots_l766_766537


namespace ticket_price_difference_l766_766429

def pre_bought_payment (number_pre : ℕ) (price_pre : ℕ) : ℕ :=
  number_pre * price_pre

def gate_payment (number_gate : ℕ) (price_gate : ℕ) : ℕ :=
  number_gate * price_gate

theorem ticket_price_difference :
  ∀ (number_pre number_gate price_pre price_gate : ℕ),
  number_pre = 20 →
  price_pre = 155 →
  number_gate = 30 →
  price_gate = 200 →
  gate_payment number_gate price_gate - pre_bought_payment number_pre price_pre = 2900 :=
by {
  intros,
  sorry
}

end ticket_price_difference_l766_766429


namespace arithmetic_expression_value_l766_766805

theorem arithmetic_expression_value : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_expression_value_l766_766805


namespace real_numbers_inequality_l766_766290

theorem real_numbers_inequality (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1 / 3) * (a + b + c)^2 :=
by
  sorry

end real_numbers_inequality_l766_766290


namespace probability_of_shortening_exactly_one_digit_l766_766837

theorem probability_of_shortening_exactly_one_digit :
  let n := 2014 in
  let p := 0.1 in
  let q := 0.9 in
  (n.choose 1) * p ^ 1 * q ^ (n - 1) = 201.4 * 1.564 * 10^(-90) :=
by sorry

end probability_of_shortening_exactly_one_digit_l766_766837


namespace distance_AC_l766_766567

theorem distance_AC (south_dist : ℕ) (west_dist : ℕ) (north_dist : ℕ) (east_dist : ℕ) :
  south_dist = 50 → west_dist = 70 → north_dist = 30 → east_dist = 40 →
  Real.sqrt ((south_dist - north_dist)^2 + (west_dist - east_dist)^2) = 36.06 :=
by
  intros h_south h_west h_north h_east
  rw [h_south, h_west, h_north, h_east]
  simp
  norm_num
  sorry

end distance_AC_l766_766567


namespace shorten_by_one_expected_length_l766_766852

-- Definition of the random sequence and binomial distribution properties
def digit_sequence {n : ℕ} (p : ℝ) (len : ℕ) : ℝ := 
  let q := 1 - p in
  (len.choose 1) * (p^1 * q^(len - 1))

-- Part (a) statement
theorem shorten_by_one 
  (len : ℕ) 
  (p : ℝ := 0.1) 
  (q := 1 - p) 
  (x := len - 1) : 
  digit_sequence p x = 1.564e-90 :=
by sorry

-- Part (b) statement
theorem expected_length 
  (initial_len : ℕ) 
  (p : ℝ := 0.1) (removals := initial_len - 1) 
  (expected_removed : ℝ := removals * p) : 
  (initial_len : ℝ) - expected_removed = 1813.6 :=
by sorry

end shorten_by_one_expected_length_l766_766852


namespace pascal_triangle_difference_l766_766584

open Nat

def binom : ℕ → ℕ → ℕ
| n, k => if h : k ≤ n then Nat.choose n k else 0

theorem pascal_triangle_difference :
  (∑ i in Finset.range 102, (binom 101 i : ℚ) / (binom 102 i)) - 
  (∑ i in Finset.range 101, (binom 100 i : ℚ) / (binom 101 i)) = 0.5 :=
by
  sorry

end pascal_triangle_difference_l766_766584


namespace probability_of_at_least_two_white_balls_l766_766654

open Nat

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 := 1
| 0, k + 1 := 0
| n + 1, k + 1 := binom n k + binom n (k + 1)

theorem probability_of_at_least_two_white_balls (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) (draws : ℕ) :
  total_balls = 10 ∧ red_balls = 6 ∧ white_balls = 4 ∧ draws = 4 →
  let total_ways := binom total_balls draws in
  let ways_at_least_2_white := (binom white_balls 2 * binom red_balls 2) +
                               (binom white_balls 3 * binom red_balls 1) +
                               (binom white_balls 4) in
  (ways_at_least_2_white / total_ways : ℚ) = 23 / 42 :=
begin
  intros,
  sorry
end

end probability_of_at_least_two_white_balls_l766_766654


namespace s_is_group_l766_766288

variable {S : Type*} [Nonempty S]
variable (mul : S → S → S)
variable (pow : S → ℕ → S)

-- Assumptions
axiom associative : ∀ a b c : S, mul (mul a b) c = mul a (mul b c)
axiom left_cancellation : ∀ a b c : S, mul a b = mul a c → b = c
axiom right_cancellation : ∀ a b c : S, mul b a = mul c a → b = c
axiom finite_powers : ∀ a : S, Finite {n : ℕ | pow a n = a}

-- Goal: S is a group
def is_group (S : Type*) [Nonempty S] (mul : S → S → S) (pow : S → ℕ → S) : Prop :=
  ∀ a b c : S, 
    mul (mul a b) c = mul a (mul b c) ∧
    ∀ a b c : S, mul a b = mul a c → b = c ∧
    ∀ a b c : S, mul b a = mul c a → b = c ∧
    (∀ a : S, Finite {n : ℕ | pow a n = a}) → 
    Nonempty (Group S)

theorem s_is_group : is_group S mul pow :=
  sorry

end s_is_group_l766_766288


namespace sqrt_simplification_l766_766365

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l766_766365


namespace smallest_number_of_eggs_over_150_l766_766821

theorem smallest_number_of_eggs_over_150 
  (d : ℕ) 
  (h1: 12 * d - 3 > 150) 
  (h2: ∀ k < d, 12 * k - 3 ≤ 150) :
  12 * d - 3 = 153 :=
by
  sorry

end smallest_number_of_eggs_over_150_l766_766821


namespace simplify_expression_l766_766384

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l766_766384


namespace remainder_divided_by_82_l766_766234

theorem remainder_divided_by_82 (x : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) ↔ (∃ m : ℤ, x + 13 = 41 * m + 18) :=
by
  sorry

end remainder_divided_by_82_l766_766234


namespace steve_travel_time_l766_766398

theorem steve_travel_time :
  ∀ (d : ℕ) (v_back : ℕ) (v_to : ℕ),
  d = 20 →
  v_back = 10 →
  v_to = v_back / 2 →
  d / v_to + d / v_back = 6 := 
by
  intros d v_back v_to h1 h2 h3
  sorry

end steve_travel_time_l766_766398


namespace walking_rate_ratio_l766_766473

theorem walking_rate_ratio (R R' : ℚ) (D : ℚ) (h1: D = R * 14) (h2: D = R' * 12) : R' / R = 7 / 6 :=
by 
  sorry

end walking_rate_ratio_l766_766473


namespace simplify_expression_calculate_expression_l766_766387

-- Problem 1
theorem simplify_expression (x : ℝ) : 
  (x + 1) * (x + 1) - x * (x + 1) = x + 1 := by
  sorry

-- Problem 2
theorem calculate_expression : 
  (-1 : ℝ) ^ 2023 + 2 ^ (-2 : ℝ) + 4 * (Real.cos (Real.pi / 6))^2 = 9 / 4 := by
  sorry

end simplify_expression_calculate_expression_l766_766387


namespace range_of_f_2m_l766_766609

def a (x: ℝ) : ℝ × ℝ := (Real.sin x, 1)
def b (x: ℝ) : ℝ × ℝ := (Real.sqrt 3, Real.cos x)
def f (x: ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem range_of_f_2m:
  ∀ m: ℝ, 0 < m ∧ m ≤ π / 6 → (1 ≤ f(2 * m) ∧ f(2 * m) ≤ 2) :=
by
  intros
  sorry

end range_of_f_2m_l766_766609


namespace alice_bob_coffee_shop_spending_l766_766954

theorem alice_bob_coffee_shop_spending (A B : ℝ) (h1 : B = 0.5 * A) (h2 : A = B + 15) : A + B = 45 :=
by
  sorry

end alice_bob_coffee_shop_spending_l766_766954


namespace max_elements_of_S_l766_766270

-- Define the relation on set S and the conditions given
variable {S : Type} (R : S → S → Prop)

-- Lean translation of the conditions
def condition_1 (a b : S) : Prop :=
  (R a b ∨ R b a) ∧ ¬ (R a b ∧ R b a)

def condition_2 (a b c : S) : Prop :=
  R a b ∧ R b c → R c a

-- Define the problem statement:
theorem max_elements_of_S (h1 : ∀ a b : S, condition_1 R a b)
                          (h2 : ∀ a b c : S, condition_2 R a b c) :
  ∃ (n : ℕ), (∀ T : Finset S, T.card ≤ n) ∧ (∃ T : Finset S, T.card = 3) :=
sorry

end max_elements_of_S_l766_766270


namespace g_2015_eq_1_l766_766586

def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := f(x) + 1 - x

axiom f_cond1 : f(1) = 1
axiom f_cond2 : ∀ x : ℝ, f(x + 5) ≥ f(x) + 5
axiom f_cond3 : ∀ x : ℝ, f(x + 1) ≤ f(x) + 1

theorem g_2015_eq_1 : g(2015) = 1 := 
by 
  sorry

end g_2015_eq_1_l766_766586


namespace donation_percentage_l766_766488

noncomputable def income : ℝ := 266666.67
noncomputable def remaining_income : ℝ := 0.25 * income
noncomputable def final_amount : ℝ := 40000

theorem donation_percentage :
  ∃ D : ℝ, D = 40 /\ (1 - D / 100) * remaining_income = final_amount :=
by
  sorry

end donation_percentage_l766_766488


namespace simplify_radical_expression_l766_766363

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l766_766363


namespace compute_expression_l766_766895

-- Defining notation for the problem expression and answer simplification
theorem compute_expression : 
    9 * (2/3)^4 = 16/9 := by 
  sorry

end compute_expression_l766_766895


namespace unit_digit_14_pow_100_l766_766458

theorem unit_digit_14_pow_100 : (14 ^ 100) % 10 = 6 :=
by
  sorry

end unit_digit_14_pow_100_l766_766458


namespace sqrt_simplification_l766_766374

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l766_766374


namespace problem_condition_l766_766210

open Real

theorem problem_condition (x θ : ℝ) (k : ℤ) (hx : 0 ≤ x ∧ x ≤ 1)
    (h : x^2 * cos θ - x * (1 - x) + (1 - x)^2 * sin θ > 0) :
    ∃ k : ℤ, (θ ∈ (2 * k * π + π / 12, 2 * k * π + 5 * π / 12)) :=
sorry

end problem_condition_l766_766210


namespace john_cannot_afford_lower_grades_l766_766886

-- Conditions as definitions
def total_assignments : ℕ := 60
def target_percentage : ℝ := 0.9
def target_assignments : ℕ := (target_percentage * total_assignments).to_nat
def assignments_completed : ℕ := 40
def assignments_with_B_grade : ℕ := 32

-- Equivalent proof problem statement
theorem john_cannot_afford_lower_grades :
  total_assignments = 60 ∧
  target_assignments = 54 ∧
  assignments_completed = 40 ∧
  assignments_with_B_grade = 32 →
  ∀ remaining, remaining = (total_assignments - assignments_completed) → remaining < (target_assignments - assignments_with_B_grade) →
  false :=
by
  intros
  sorry

end john_cannot_afford_lower_grades_l766_766886


namespace part_one_part_two_l766_766216

-- Part (1) the given function and conditions
def f (x : ℝ) : ℝ := (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - (Real.sin x)^2

-- Statement for Part (1)
theorem part_one (x₀ : ℝ) (hx₀ : 0 ≤ x₀ ∧ x₀ ≤ Real.pi / 3) (hf : f (x₀ / 2) = 1 / 5) :
  Real.cos (2 * x₀) = (49 - 3 * Real.sqrt 33) / 100 := sorry

-- Part (2) the given function and conditions
def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

-- Statement for Part (2)
theorem part_two (B : ℝ) (theta : ℝ) (hBf : g B = 2) (hB : B = Real.pi/6) (hAD : AD = 3) (hDC : DC = 1) (hθ : θ = angle A B D) :
  Real.sin θ = Real.sqrt 13 / 13 := sorry

end part_one_part_two_l766_766216


namespace cube_root_of_8_l766_766736

theorem cube_root_of_8 : (∃ x : ℝ, x * x * x = 8) ∧ (∃ y : ℝ, y * y * y = 8 → y = 2) :=
by
  sorry

end cube_root_of_8_l766_766736


namespace probability_shortening_exactly_one_digit_l766_766846
open Real

theorem probability_shortening_exactly_one_digit :
  ∀ (sequence : List ℕ),
    (sequence.length = 2015) ∧
    (∀ (x ∈ sequence), x = 0 ∨ x = 9) →
      (P (λ seq, (length (shrink seq) = 2014)) sequence = 1.564e-90) := sorry

end probability_shortening_exactly_one_digit_l766_766846


namespace find_pplusq_l766_766193

-- Given conditions
def is_root (a b c : ℂ) (x : ℂ) : Prop := 2 * x^2 + a * x + b = 0

theorem find_pplusq (p q : ℝ) (hp : is_root p q (3 * complex.I - 2))
  (conjugate_root : is_root p q (-3 * complex.I - 2)) :
  p + q = 34 :=
sorry

end find_pplusq_l766_766193


namespace sqrt_simplification_l766_766372

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l766_766372


namespace find_v_l766_766530

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1;
    3, 0]

noncomputable def v : Matrix (Fin 2) (Fin 1) ℝ :=
  !![0;
    1 / 30.333]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem find_v : 
  (A ^ 10 + A ^ 8 + A ^ 6 + A ^ 4 + A ^ 2 + I) * v = !![0; 12] :=
  sorry

end find_v_l766_766530


namespace interval_with_three_buses_l766_766031

theorem interval_with_three_buses (interval_two_buses : ℕ) (total_route_time : ℕ) (bus_count : ℕ) : 
  interval_two_buses = 21 → total_route_time = 2 * interval_two_buses → bus_count = 3 → 
  (total_route_time / bus_count) = 14 :=
by
  intros h1 h2 h3
  rw [h1, h3, ← h2]
  simp
  sorry

end interval_with_three_buses_l766_766031


namespace inclination_angle_of_line_l766_766405

theorem inclination_angle_of_line (α : ℝ) (h_eq : ∀ x y, x - y + 1 = 0 ↔ y = x + 1) (h_range : 0 < α ∧ α < 180) :
  α = 45 :=
by
  -- α is the inclination angle satisfying tan α = 1 and 0 < α < 180
  sorry

end inclination_angle_of_line_l766_766405


namespace pure_imaginary_solutions_l766_766136

noncomputable def polynomial_equation (x : ℂ) : ℂ := x^4 - 4*x^3 + 10*x^2 - 64*x - 100

theorem pure_imaginary_solutions :
  ∃ (x : ℂ), polynomial_equation x = 0 ∧ (∃ k : ℝ, x = k * complex.I) →
  ((x = 4 * complex.I) ∨ (x = -4 * complex.I)) :=
by
  sorry

end pure_imaginary_solutions_l766_766136


namespace range_of_a_l766_766648

noncomputable def is_monotonically_decreasing (f : ℝ → ℝ) (I : set ℝ) :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y < f x

theorem range_of_a (a : ℝ) :
  is_monotonically_decreasing (λ x, (Real.log (1 - a * x) / log 10)) {x | 0 < x ∧ x < 1} ↔ 0 < a ∧ a ≤ 1 :=
sorry

end range_of_a_l766_766648


namespace sqrt_simplification_l766_766369

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l766_766369


namespace sum_of_special_sequence_third_terms_l766_766893

theorem sum_of_special_sequence_third_terms :
  ∃ d1 d2 (x1 x2 : ℤ),
  (d1 ∈ [5, 7]) ∧ (d2 ∈ [5, 7]) ∧ 
  (x1 ≠ x2 ∧ x1 = d1 - 5 ∧ x2 = d2 - 7) ∧
  let t1 := x1 + d1 in
  let t2 := x2 + d2 in
  t1 + t2 = 31 :=
sorry

end sum_of_special_sequence_third_terms_l766_766893


namespace multiples_of_six_l766_766628

theorem multiples_of_six (a b : ℕ) (h₁ : a = 5) (h₂ : b = 127) :
  ∃ n : ℕ, n = 21 ∧ ∀ x : ℕ, (a < 6 * x ∧ 6 * x < b) ↔ (1 ≤ x ∧ x ≤ 21) :=
by
  sorry

end multiples_of_six_l766_766628


namespace fourth_place_l766_766563

/-- Definitions for the names of the students -/
def Ryan : Type := ℕ
def Henry : Type := ℕ
def Faiz : Type := ℕ
def Toma : Type := ℕ
def Omar : Type := ℕ

/-- Speed hierarchy as conditions -/
variable (R H F T O : ℕ)
variable (h1 : R > H)
variable (h2 : R > F)
variable (h3 : F > H)
variable (h4 : T > R)
variable (h5 : O > T)

/-- Proving that Faiz finishes fourth -/
theorem fourth_place (h1 : R > H) (h2 : R > F) (h3 : F > H) (h4 : T > R) (h5 : O > T) : 
  fourth [O, T, R, F, H] = F :=
by 
  sorry

end fourth_place_l766_766563


namespace simplify_expression_l766_766326

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l766_766326


namespace minimum_value_of_omega_l766_766204

open Real

noncomputable def min_omega (ω : ℝ) : Prop :=
  ∃ (ϕ : ℝ), 
    (∀ x y, (π / 18 ≤ x ∧ x ≤ 5 * π / 36 ∧ π / 18 ≤ y ∧ y ≤ 5 * π / 36 ∧ x < y) → 
    sin (ω * x + ϕ) > sin (ω * y + ϕ))
    ∧ (sin (ω * (-π / 36) + ϕ) = sin (ω * (-π / 36) + ϕ))
    ∧ (sin (ω * (7 * π / 72) + ϕ) = 0)

theorem minimum_value_of_omega : ∃ ω : ℝ, ω = 4 ∧ min_omega ω :=
by
  use 4
  split
  · rfl
  sorry


end minimum_value_of_omega_l766_766204


namespace range_of_sum_coords_on_ellipse_l766_766576

theorem range_of_sum_coords_on_ellipse (x y : ℝ) 
  (h : x^2 / 144 + y^2 / 25 = 1) : 
  -13 ≤ x + y ∧ x + y ≤ 13 := 
sorry

end range_of_sum_coords_on_ellipse_l766_766576


namespace foci_distance_eq_2sqrt2_l766_766741

noncomputable def hyperbola_foci_distance (f : ℝ × ℝ → Prop) := 
  ∀ (x y : ℝ), f (x, y) ↔ x * y = 2

theorem foci_distance_eq_2sqrt2 :
  ∃ d : ℝ, (∀ (foci1 foci2 : ℝ × ℝ), foci1 = (real.sqrt 2, real.sqrt 2) ∧ foci2 = (-real.sqrt 2, -real.sqrt 2) →
    d = dist foci1 foci2) ∧ d = 2 * real.sqrt 2 :=
  sorry

end foci_distance_eq_2sqrt2_l766_766741


namespace simplify_expression_l766_766376

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l766_766376


namespace find_number_l766_766817

theorem find_number (x : ℝ) (h : (2 * x - 37 + 25) / 8 = 5) : x = 26 :=
sorry

end find_number_l766_766817


namespace Carol_mother_carrots_l766_766128

theorem Carol_mother_carrots (carol_picked : ℕ) (total_good : ℕ) (total_bad : ℕ) (total_carrots : ℕ) (mother_picked : ℕ) :
  carol_picked = 29 → total_good = 38 → total_bad = 7 → total_carrots = total_good + total_bad → mother_picked = total_carrots - carol_picked → mother_picked = 16 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  sorry

end Carol_mother_carrots_l766_766128


namespace radius_of_circle_l766_766662

theorem radius_of_circle
  (A B C D M : Point)
  (r : ℝ)
  (h_isosceles_trapezoid : is_isosceles_trapezoid A B C D)
  (h_parallel : is_parallel B C D A)
  (h_circle_tangent_BC : is_tangent (circle_centred_radius M r) B C)
  (h_circle_tangent_AB : is_tangent (circle_centred_radius M r) A B)
  (h_circle_tangent_CD : is_tangent (circle_centred_radius M r) C D)
  (h_intersection_point : lines_intersect_at A C B D M)
  (h_ratio : ratio AD BC = 9/7)
  (h_area : area A B C D = 8) :
  r = 7 * root 7 18 / 16 := 
sorry

end radius_of_circle_l766_766662


namespace basketball_game_score_l766_766658

theorem basketball_game_score
  (a d b r : ℕ)
  (h1 : d > 0)
  (h2 : r > 1)
  (eagles_quarters : ℕ → ℕ := λ n, a + n * d)
  (lions_quarters : ℕ → ℕ := λ n, b * r ^ n)
  (eagles_first_half_score : ℕ := eagles_quarters 0 + eagles_quarters 1)
  (lions_first_half_score : ℕ := lions_quarters 0 + lions_quarters 1)
  (eagles_total_score : ℕ := ∑ i in Finset.range 4, eagles_quarters i)
  (lions_total_score : ℕ := ∑ i in Finset.range 4, lions_quarters i)
  (h3 : eagles_total_score = lions_total_score + 2)
  (h4 : ∀ n < 4, eagles_quarters n ≤ 100)
  (h5 : ∀ n < 4, lions_quarters n ≤ 100) :
  eagles_first_half_score + lions_first_half_score = 25 :=
sorry

end basketball_game_score_l766_766658


namespace greatest_integer_solution_l766_766050

theorem greatest_integer_solution :
  ∃ x : ℤ, (∃ (k : ℤ), (8 : ℚ) / 11 > k / 15 ∧ k = 10) ∧ x = 10 :=
by {
  sorry
}

end greatest_integer_solution_l766_766050


namespace quadratic_inequality_solution_l766_766559

theorem quadratic_inequality_solution {x : ℝ} :
  (x^2 + x - 6 ≤ 0) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l766_766559


namespace book_price_increase_percentage_l766_766410

theorem book_price_increase_percentage :
  ∀ (initial_price new_price : ℝ), initial_price = 300 ∧ new_price = 450 →
  ((new_price - initial_price) / initial_price * 100) = 50 :=
by
  intros initial_price new_price h
  cases h with h1 h2
  rw [h1, h2]
  norm_num
  done

end book_price_increase_percentage_l766_766410


namespace greatest_difference_units_digit_l766_766771

theorem greatest_difference_units_digit (d : Nat) 
  (h : d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h_mod4 : (72 % 100 + d) % 4 = 0) :
  ∃ a b, a ∈ {0, 4, 8} ∧ b ∈ {0, 4, 8} ∧ a ≠ b ∧ (abs (a - b) = 8) := 
begin
  -- Proof is omitted, only statement is required
  sorry
end

end greatest_difference_units_digit_l766_766771


namespace Jack_has_68_cheese_crackers_l766_766706

-- Defining the given conditions
def Jack_has_2_5_times_more_crackers_than_Marcus (jack crackers: ℕ) (marcus crackers: ℕ) :=
  jack = 5 * marcus / 2

def Marcus_has_3_times_more_crackers_than_Mona (marcus crackers: ℕ) (mona crackers: ℕ) :=
  marcus = 3 * mona

def Nicholas_has_6_more_crackers_than_Mona (nicholas crackers: ℕ) (mona crackers: ℕ) :=
  nicholas = mona + 6

def Tamara_has_twice_as_many_crackers_as_Nicholas (tamara crackers: ℕ) (nicholas crackers: ℕ) :=
  tamara = 2 * nicholas

def Marcus_has (marcus crackers: ℕ) :=
  marcus = 27

-- Lean 4 statement to prove Jack has 68 cheese crackers
theorem Jack_has_68_cheese_crackers:
  ∀ (jack marcus mona nicholas tamara: ℕ),
  Jack_has_2_5_times_more_crackers_than_Marcus jack marcus →
  Marcus_has_3_times_more_crackers_than_Mona marcus mona →
  Nicholas_has_6_more_crackers_than_Mona nicholas mona →
  Tamara_has_twice_as_many_crackers_as_Nicholas tamara nicholas →
  Marcus_has marcus →
  jack = 68 :=
by
  intros jack marcus mona nicholas tamara h1 h2 h3 h4 h5
  sorry

end Jack_has_68_cheese_crackers_l766_766706


namespace find_k_for_line_l766_766960

-- Define the point and the line equation
def point := (1 / 3, -5 : ℝ)
def line_equation (k : ℝ) (x y : ℝ) := -2 / 3 - 3 * k * x = 7 * y

-- State the theorem
theorem find_k_for_line : ∃ k : ℝ, line_equation k (fst point) (snd point) ∧ k = 103 / 3 :=
by {
  sorry
}

end find_k_for_line_l766_766960


namespace rows_have_same_good_cells_iff_n_odd_l766_766190

def good_cell (n : ℕ) (A : ℕ → ℕ → ℕ) (i j : ℕ) : Prop :=
  A i j > j

noncomputable def valid_arrangement (n : ℕ) (A : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j, A i j ∈ finset.range n ∧ ∀ i, (finset.univ.image (λ j, A i j)).card = n ∧ (finset.univ.image (λ i, A i j)).card = n)

theorem rows_have_same_good_cells_iff_n_odd (n : ℕ) :
  (∃ A : ℕ → ℕ → ℕ, valid_arrangement n A ∧ (∀ i j, good_cell n A i j = true) → (finset.card (finset.filter (good_cell n A) (finset.univ.product finset.univ)) / n) = (n - 1) / 2) ↔ odd n :=
sorry

end rows_have_same_good_cells_iff_n_odd_l766_766190


namespace find_radii_of_spheres_l766_766422

def externally_tangent_spheres_touching_plane (x y z : ℝ) : Prop := 
  (2 * real.sqrt (y * z) = 1) ∧
  (2 * real.sqrt (x * y) = 2) ∧
  (2 * real.sqrt (x * z) = real.sqrt 3)

theorem find_radii_of_spheres
  (x y z : ℝ)
  (h1 : externally_tangent_spheres_touching_plane x y z)
  (h2 : y * z = 1 / 4)
  (h3 : x * y = 1)
  (h4 : x * z = 3 / 4) :
  (x = real.sqrt 3) ∧ (y = 1 / real.sqrt 3) ∧ (z = real.sqrt 3 / 4) :=
by {
  sorry
} 

end find_radii_of_spheres_l766_766422


namespace skew_lines_implies_parallel_planes_l766_766636

variables {a b : Set Point}
variable {alpha beta : Set Plane}

def are_skew_lines (a b : Set Point) : Prop :=
  ∃p q r s : Point, p ∈ a ∧ q ∈ a ∧ r ∈ b ∧ s ∈ b ∧ collinear p q ∧ ¬collinear r s ∧
  (∀ p', p' ∈ a → ∃ p'', p'' ∈ b ∧ ¬ collinear p' p'')

def planes_parallel (alpha beta : Set Plane) : Prop :=
  ∀ p q ∈ alpha, ∀ r s ∈ beta, parallel (line_through_points p q) (line_through_points r s)

def line_in_plane (a : Set Point) (alpha : Set Plane) : Prop :=
  ∀ p ∈ a, p ∈ alpha

theorem skew_lines_implies_parallel_planes
  (h_skew : are_skew_lines a b)
  : ∃ alpha beta, planes_parallel alpha beta ∧ line_in_plane a alpha ∧ line_in_plane b beta := by
  sorry

end skew_lines_implies_parallel_planes_l766_766636


namespace percentage_increase_after_decrease_and_increase_l766_766235

theorem percentage_increase_after_decrease_and_increase 
  (P : ℝ) 
  (h : 0.8 * P + (x / 100) * (0.8 * P) = 1.16 * P) : 
  x = 45 :=
by
  sorry

end percentage_increase_after_decrease_and_increase_l766_766235


namespace right_triangle_median_squared_sum_l766_766266

open Real

theorem right_triangle_median_squared_sum (A B C D : Point)
    (BC AD : ℝ)
    (right_triangle : ∠ ABC = π/2)
    (BC_eq_10 : BC = 10)
    (AD_eq_6 : AD = 6)
    (median_AD : (A + C) / 2 = D)
    (D_on_BC : D ∈ Line(B, C)) :
    AB^2 + AC^2 = 122 :=
by
  sorry

end right_triangle_median_squared_sum_l766_766266


namespace problem_a_problem_b_l766_766840

open Probability

noncomputable def sequenceShortenedByExactlyOneDigitProbability : ℝ :=
  1.564 * 10^(-90)

theorem problem_a (n : ℕ) (p : ℝ) (k : ℕ) (initialLength : ℕ)
  (h_n : n = 2014) (h_p : p = 0.1) (h_k : k = 1) (h_initialLength : initialLength = 2015) :
  (binomialₓ n k p * (1 - p)^(n - k)) = sequenceShortenedByExactlyOneDigitProbability := 
sorry

noncomputable def expectedLengthNewSequence : ℝ :=
  1813.6

theorem problem_b (n : ℕ) (p : ℝ) (initialLength : ℕ) 
  (h_n : n = 2014) (h_p : p = 0.1) (h_initialLength : initialLength = 2015) :
  (initialLength - n * p) = expectedLengthNewSequence :=
sorry

end problem_a_problem_b_l766_766840


namespace sum_logs_equals_l766_766527

theorem sum_logs_equals :
  (∑ k in finset.range 62 \ {0, 1}, log 2 (1 + 1 / (k + 3)) * log (k + 3) 2 * log (k + 4) 2) = 0.46470 :=
by
  sorry

end sum_logs_equals_l766_766527


namespace isosceles_triangle_angles_l766_766695

theorem isosceles_triangle_angles (A B C D : Type)
  [h1 : IsoscelesAt A B C]
  [h2 : AngleBisectorAt B C D]
  [h3 : DistanceEqual B D A D] :
  angleBAC = 36 ∧ angleABC = 72 ∧ angleACB = 72 :=
by
  sorry

end isosceles_triangle_angles_l766_766695


namespace sqrt_simplification_l766_766375

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l766_766375


namespace compute_fraction_power_l766_766902

theorem compute_fraction_power :
  9 * (2/3)^4 = 16/9 := 
by
  sorry

end compute_fraction_power_l766_766902


namespace band_member_earnings_l766_766163

theorem band_member_earnings :
  let attendees := 500
  let ticket_price := 30
  let band_share_percentage := 70 / 100
  let band_members := 4
  let total_earnings := attendees * ticket_price
  let band_earnings := total_earnings * band_share_percentage
  let earnings_per_member := band_earnings / band_members
  earnings_per_member = 2625 := 
by {
  sorry
}

end band_member_earnings_l766_766163


namespace euler_formula_connected_planar_graph_l766_766700

theorem euler_formula_connected_planar_graph 
(V E F : ℕ) 
(G : Type*) [graph G] 
(h_connected : Connected G) 
(h_planar : Planar G)
(h_V : vertices G = V)
(h_E : edges G = E)
(h_F : faces G = F) : 
V - E + F = 2 := 
sorry

end euler_formula_connected_planar_graph_l766_766700


namespace kyle_speed_l766_766680

theorem kyle_speed (S : ℝ) (joseph_speed : ℝ) (joseph_time : ℝ) (kyle_time : ℝ) (H1 : joseph_speed = 50) (H2 : joseph_time = 2.5) (H3 : kyle_time = 2) (H4 : joseph_speed * joseph_time = kyle_time * S + 1) : S = 62 :=
by
  sorry

end kyle_speed_l766_766680


namespace identify_A_B_l766_766711

variable {Person : Type}
variable (isTruthful isLiar : Person → Prop)
variable (isBoy isGirl : Person → Prop)

variables (A B : Person)

-- Conditions
axiom truthful_or_liar : ∀ x : Person, isTruthful x ∨ isLiar x
axiom boy_or_girl : ∀ x : Person, isBoy x ∨ isGirl x
axiom not_both_truthful_and_liar : ∀ x : Person, ¬(isTruthful x ∧ isLiar x)
axiom not_both_boy_and_girl : ∀ x : Person, ¬(isBoy x ∧ isGirl x)

-- Statements made by A and B
axiom A_statement : isTruthful A → isLiar B 
axiom B_statement : isBoy B → isGirl A 

-- Goal: prove the identities of A and B
theorem identify_A_B : isTruthful A ∧ isBoy A ∧ isLiar B ∧ isBoy B :=
by {
  sorry
}

end identify_A_B_l766_766711


namespace construct_quadrilateral_l766_766979

variable {A B C D : Type}   -- Define quadrilateral's vertices A, B, C, D
variable {a b c d l m : ℝ}  -- Define lengths of sides and distances

theorem construct_quadrilateral (h1 : |b - d| < 2 * l < b + d) (h2 : |a - c| < 2 * m < a + c) :
  ∃ ABCD : A × B × C × D, 
    (dist A B = a ∧ dist B C = b ∧ dist C D = c ∧ dist D A = d) ∧
    (dist (midpoint A B) (midpoint C D) = l) :=
sorry

end construct_quadrilateral_l766_766979


namespace simplify_expression_l766_766381

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l766_766381


namespace bus_interval_three_buses_l766_766036

theorem bus_interval_three_buses (T : ℕ) (h : T = 21) : (T * 2) / 3 = 14 :=
by
  sorry

end bus_interval_three_buses_l766_766036


namespace point_relationship_on_parabola_neg_x_plus_1_sq_5_l766_766585

theorem point_relationship_on_parabola_neg_x_plus_1_sq_5
  (y_1 y_2 y_3 : ℝ) :
  (A : ℝ × ℝ) = (-2, y_1) →
  (B : ℝ × ℝ) = (1, y_2) →
  (C : ℝ × ℝ) = (2, y_3) →
  (A.2 = -(A.1 + 1)^2 + 5) →
  (B.2 = -(B.1 + 1)^2 + 5) →
  (C.2 = -(C.1 + 1)^2 + 5) →
  y_1 > y_2 ∧ y_2 > y_3 :=
by
  sorry

end point_relationship_on_parabola_neg_x_plus_1_sq_5_l766_766585


namespace find_number_in_3rd_row_51st_column_find_number_in_3rd_row_51st_column_l766_766511

theorem find_number_in_3rd_row_51st_column : 
  (seq_starting_from (start := 1) (cols_period := 4) (nums_per_cycle := 9) (row := 3) (col := 51)) = 113 := 
sorry

-- Definitions from conditions
def seq_starting_from (start : ℕ) (cols_period : ℕ) (nums_per_cycle : ℕ) (row : ℕ) (col : ℕ) : ℤ :=
  let col_period := (col % cols_period)
  let complete_cycles := (col / cols_period)
  let base_value_start_period := start + complete_cycles * nums_per_cycle
  base_value_start_period + (row - 1) * cols_period + col_period

theorem find_number_in_3rd_row_51st_column : 
  (seq_starting_from (start := 1) (cols_period := 4) (nums_per_cycle := 9) (row := 3) (col := 51)) = 113 := 
sorry

end find_number_in_3rd_row_51st_column_find_number_in_3rd_row_51st_column_l766_766511


namespace simplify_radical_expression_l766_766357

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l766_766357


namespace shortest_distance_l766_766281

-- Define the vectors and points involved in the problem
def P (t : ℝ) : ℝ × ℝ × ℝ :=
  (3 * t + 4, -t + 1, 2 * t + 3)

def Q (s : ℝ) : ℝ × ℝ × ℝ :=
  (2 * s + 1, s - 1, -3 * s + 5)

-- Define the distance squared function between two points
def distance_squared (P Q : ℝ × ℝ × ℝ) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

-- The theorem to prove
theorem shortest_distance : ∃ (s t : ℝ), sqrt (distance_squared (P t) (Q s)) = √5 :=
by sorry

end shortest_distance_l766_766281


namespace simplify_sqrt_expression_l766_766340

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l766_766340


namespace evaluate_expression_l766_766812

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 :=
by sorry

end evaluate_expression_l766_766812


namespace diagonal_difference_zero_l766_766860

theorem diagonal_difference_zero : 
  let orig_matrix := [[10, 11, 12], [19, 20, 21], [28, 29, 30]]
  let transformed_matrix := [[12, 11, 10], [19, 20, 21], [30, 29, 28]]
  let main_diagonal_sum := transformed_matrix[0][0] + transformed_matrix[1][1] + transformed_matrix[2][2]
  let anti_diagonal_sum := transformed_matrix[0][2] + transformed_matrix[1][1] + transformed_matrix[2][0]
  in |main_diagonal_sum - anti_diagonal_sum| = 0 :=
by
  let orig_matrix := [[10, 11, 12], [19, 20, 21], [28, 29, 30]]
  let transformed_matrix := [[12, 11, 10], [19, 20, 21], [30, 29, 28]]
  let main_diagonal_sum := transformed_matrix[0][0] + transformed_matrix[1][1] + transformed_matrix[2][2]
  let anti_diagonal_sum := transformed_matrix[0][2] + transformed_matrix[1][1] + transformed_matrix[2][0]
  have main_diagonal_sum_is_60 : main_diagonal_sum = 12 + 20 + 28 := by rfl
  have anti_diagonal_sum_is_60 : anti_diagonal_sum = 10 + 20 + 30 := by rfl
  rw [main_diagonal_sum_is_60, anti_diagonal_sum_is_60]
  exact abs_eq_zero.mpr rfl

end diagonal_difference_zero_l766_766860


namespace min_distance_sum_l766_766194

-- Definitions for points P and Q on their respective curves
variables {P Q : ℝ × ℝ}
variables (hP : P.snd ^ 2 = 4 * P.fst) (hQ : Q.fst ^ 2 + (Q.snd - 4) ^ 2 = 1)

-- Declaration of the theorem about the minimum value of the specified sum
theorem min_distance_sum :
  let F := (1, 0)
  let C := (0, 4)
  let dist (A B : ℝ × ℝ) := real.sqrt ((A.fst - B.fst) ^ 2 + (A.snd - B.snd) ^ 2)
  let dist_to_axis (A : ℝ × ℝ) := abs (A.fst - 1)
  (∃ P Q, hP ∧ hQ ∧ (P = (x_P, y_P) ∧ Q = (x_Q, y_Q))) →
    ∃ P Q, dist P Q + dist_to_axis P = real.sqrt 17 - 1 :=
by
  sorry

end min_distance_sum_l766_766194


namespace probability_of_shortening_exactly_one_digit_l766_766836

theorem probability_of_shortening_exactly_one_digit :
  let n := 2014 in
  let p := 0.1 in
  let q := 0.9 in
  (n.choose 1) * p ^ 1 * q ^ (n - 1) = 201.4 * 1.564 * 10^(-90) :=
by sorry

end probability_of_shortening_exactly_one_digit_l766_766836


namespace r_limit_as_m_approaches_zero_l766_766685

noncomputable def Q (m : ℝ) : ℝ :=
  -Real.sqrt (m + 4)

noncomputable def r (m : ℝ) : ℝ :=
  (Q (-m) - Q m) / m

theorem r_limit_as_m_approaches_zero : 
  filter.tendsto (r) (nhds 0) (nhds (-1 / 2)) :=
sorry

end r_limit_as_m_approaches_zero_l766_766685


namespace band_member_share_l766_766167

def num_people : ℕ := 500
def ticket_price : ℝ := 30
def band_share_percent : ℝ := 0.70
def num_band_members : ℕ := 4

theorem band_member_share : 
  (num_people * ticket_price * band_share_percent) / num_band_members = 2625 := by
  sorry

end band_member_share_l766_766167


namespace central_square_flip_l766_766653

def is_plus_sign (x : ℕ × ℕ) : bool :=
  if x = (2, 2) then false else true

def flip_sign (grid : array (array bool)) (top_left : ℕ × ℕ) (k : ℕ) : array (array bool) :=
  let rows := [0 : ℕ(5), 1 : ℕ(5), 2 : ℕ(5), 3 : ℕ(5), 4 : ℕ(5)],
      cols := [0 : ℕ(5), 1 : ℕ(5), 2 : ℕ(5), 3 : ℕ(5), 4 : ℕ(5)] in
  rows.map (λ r, cols.map (λ c,
    if r >= (fst top_left) ∧ r < (fst top_left) + k ∧
       c >= (snd top_left) ∧ c < (snd top_left) + k then
          bnot (grid[r][c])
    else grid[r][c]))

def all_plus_sign (grid : array (array bool)) : bool :=
  grid.all (λ row, row.all id)

theorem central_square_flip : 
  ∃ sequence_of_flips : list (ℕ × ℕ × ℕ),
    let final_grid := sequence_of_flips.foldl (λ grid (flip_data : ℕ × ℕ × ℕ),
        let (x, y, k) := flip_data in
        flip_sign grid (x, y) k)
      ![(is_plus_sign 0 0), (is_plus_sign 0 1), (is_plus_sign 0 2), (is_plus_sign 0 3), (is_plus_sign 0 4),
        (is_plus_sign 1 0), (is_plus_sign 1 1), (is_plus_sign 1 2), (is_plus_sign 1 3), (is_plus_sign 1 4),
        (is_plus_sign 2 0), (is_plus_sign 2 1), (is_plus_sign 2 2), (is_plus_sign 2 3), (is_plus_sign 2 4),
        (is_plus_sign 3 0), (is_plus_sign 3 1), (is_plus_sign 3 2), (is_plus_sign 3 3), (is_plus_sign 3 4),
        (is_plus_sign 4 0), (is_plus_sign 4 1), (is_plus_sign 4 2), (is_plus_sign 4 3), (is_plus_sign 4 4)] 
    in all_plus_sign final_grid :=
sorry

end central_square_flip_l766_766653


namespace greatest_diff_units_digit_of_multiple_of_4_l766_766767

theorem greatest_diff_units_digit_of_multiple_of_4 : 
  let valid_units_digits := { d : ℕ // 0 ≤ d ∧ d ≤ 9 ∧ (720 + d) % 4 = 0 } in
  ∃ d1 d2 ∈ valid_units_digits, d1 ≠ d2 ∧ d1 - d2 = 8 :=
by
  sorry

end greatest_diff_units_digit_of_multiple_of_4_l766_766767


namespace aliyah_more_phones_l766_766116

theorem aliyah_more_phones (vivi_phones : ℕ) (phone_price : ℕ) (total_money : ℕ) (aliyah_more : ℕ) : 
  vivi_phones = 40 → 
  phone_price = 400 → 
  total_money = 36000 → 
  40 + 40 + aliyah_more = total_money / phone_price → 
  aliyah_more = 10 :=
sorry

end aliyah_more_phones_l766_766116


namespace largest_a_pow_b_l766_766933

theorem largest_a_pow_b (a b : ℕ) (h_pos_a : 1 < a) (h_pos_b : 1 < b) (h_eq : a^b * b^a + a^b + b^a = 5329) : 
  a^b = 64 :=
by
  sorry

end largest_a_pow_b_l766_766933


namespace how_many_rocks_l766_766675

section see_saw_problem

-- Conditions
def Jack_weight : ℝ := 60
def Anna_weight : ℝ := 40
def rock_weight : ℝ := 4

-- Theorem statement
theorem how_many_rocks : (Jack_weight - Anna_weight) / rock_weight = 5 :=
by
  -- Proof is omitted, just ensuring the theorem statement
  sorry

end see_saw_problem

end how_many_rocks_l766_766675


namespace clock_first_ring_at_midnight_l766_766520

theorem clock_first_ring_at_midnight (rings_every_n_hours : ℕ) (rings_per_day : ℕ) (hours_in_day : ℕ) :
  rings_every_n_hours = 3 ∧ rings_per_day = 8 ∧ hours_in_day = 24 →
  ∃ first_ring_time : Nat, first_ring_time = 0 :=
by
  sorry

end clock_first_ring_at_midnight_l766_766520


namespace alcohol_solution_proof_l766_766450

def pure_alcohol (volume : ℝ) (percentage : ℝ) : ℝ :=
  volume * (percentage / 100)

def final_solution_volume (initial_volume : ℝ) (added_volume : ℝ) : ℝ :=
  initial_volume + added_volume

def final_alcohol_percentage (initial_alcohol : ℝ) (added_volume : ℝ) (final_volume : ℝ) : ℝ :=
  (initial_alcohol + added_volume) / final_volume * 100

theorem alcohol_solution_proof :
  let initial_solution_volume := 6 in
  let initial_alcohol_percentage := 35 in
  let volume_to_add := 1.8 in
  pure_alcohol initial_solution_volume initial_alcohol_percentage + volume_to_add / final_solution_volume initial_solution_volume volume_to_add = 50 :=
by
  sorry -- Proof goes here

end alcohol_solution_proof_l766_766450


namespace negative_values_count_l766_766955

theorem negative_values_count :
  {n : ℕ | ∃ x : ℤ, n = Int.natAbs ((x + 100).natAbs) ∧ x + 100 = n^2 ∧ x < 0}.finite
    → ∃ x_set : Finset ℤ, ∀ x ∈ x_set, (∃ n : ℕ, n > 0 ∧ x + 100 = n^2) ∧ x < 0 ∧ x_set.card = 9 :=
by
  sorry

end negative_values_count_l766_766955


namespace initial_eggs_count_l766_766019

theorem initial_eggs_count (harry_adds : ℕ) (total_eggs : ℕ) (initial_eggs : ℕ) :
  harry_adds = 5 → total_eggs = 52 → initial_eggs = total_eggs - harry_adds → initial_eggs = 47 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end initial_eggs_count_l766_766019


namespace inequality_proof_equality_conditions_l766_766200

theorem inequality_proof
  (x y : ℝ)
  (h1 : x ≥ y)
  (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y) + y / Real.sqrt (y + 1) + 1 / Real.sqrt (x + 1) ≥
   y / Real.sqrt (x + y) + x / Real.sqrt (x + 1) + 1 / Real.sqrt (y + 1)) :=
by
  sorry

theorem equality_conditions
  (x y : ℝ) :
  (x = y ∨ x = 1 ∨ y = 1) ↔
  (x / Real.sqrt (x + y) + y / Real.sqrt (y + 1) + 1 / Real.sqrt (x + 1) =
   y / Real.sqrt (x + y) + x / Real.sqrt (x + 1) + 1 / Real.sqrt (y + 1)) :=
by
  sorry

end inequality_proof_equality_conditions_l766_766200


namespace simplify_expr_l766_766346

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l766_766346


namespace simplify_sqrt_expression_l766_766332

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l766_766332


namespace diplomats_speak_french_l766_766884

theorem diplomats_speak_french (
    T : ℕ,
    hT : T = 100,
    N : ℕ,
    hN : N = 0.2 * T,
    B : ℕ,
    hB : B = 0.1 * T,
    dR : ℕ,
    hdR : dR = 32
  ) : (T - N) - (T - dR) + B = 22 := by
  sorry

end diplomats_speak_french_l766_766884


namespace evaluate_expression_l766_766801

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end evaluate_expression_l766_766801


namespace product_equals_one_l766_766904

noncomputable def product_expression : ℂ :=
  ∏ (k : ℕ) in Finset.range 10, ∏ (j : ℕ) in Finset.range 12,
    (Complex.exp (2 * Real.pi * Complex.I * j / 13) - Complex.exp (2 * Real.pi * Complex.I * k / 11))

theorem product_equals_one : product_expression = 1 := 
by {
  sorry
}

end product_equals_one_l766_766904


namespace shortest_distance_parabola_line_l766_766834

theorem shortest_distance_parabola_line :
  ∃ (P Q : ℝ × ℝ), P.2 = P.1^2 - 6 * P.1 + 15 ∧ Q.2 = 2 * Q.1 - 7 ∧
  ∀ (p q : ℝ × ℝ), p.2 = p.1^2 - 6 * p.1 + 15 → q.2 = 2 * q.1 - 7 → 
  dist p q ≥ dist P Q :=
sorry

end shortest_distance_parabola_line_l766_766834


namespace trailing_zeros_2014_l766_766890

def legendre_p (n p : ℕ) : ℕ :=
  (List.range (n.log p).succ).sum (λ k => n / p^k.succ)

def trailing_zeros (n : ℕ) : ℕ :=
  legendre_p n 5

theorem trailing_zeros_2014 : trailing_zeros 2014 = 501 :=
by sorry

end trailing_zeros_2014_l766_766890


namespace number_of_sides_l766_766096

def side_length : ℕ := 16
def perimeter : ℕ := 80

theorem number_of_sides (h1: side_length = 16) (h2: perimeter = 80) : (perimeter / side_length = 5) :=
by
  -- Proof should be inserted here.
  sorry

end number_of_sides_l766_766096


namespace correct_time_fraction_l766_766091

theorem correct_time_fraction :
  let hours := 24
  let correct_hours := 10
  let minutes_per_hour := 60
  let correct_minutes_per_hour := 20
  (correct_hours * correct_minutes_per_hour : ℝ) / (hours * minutes_per_hour) = (5 / 36 : ℝ) :=
by
  let hours := 24
  let correct_hours := 10
  let minutes_per_hour := 60
  let correct_minutes_per_hour := 20
  sorry

end correct_time_fraction_l766_766091


namespace proof_u_plus_v_l766_766026

-- Definitions of all given conditions
def PQ : ℝ := 5
def QR : ℝ := 12
def PR : ℝ := Real.sqrt (PQ ^ 2 + QR ^ 2) -- Since PR = √(5^2 + 12^2) = 13

def PS : ℝ := 20
def RS : ℝ := Real.sqrt (PR ^ 2 + PS ^ 2) -- Since RS = √(13^2 + 20^2) = √(569)

def ST : ℝ := (PS / PR) * PQ -- Similar triangles give us this ratio
def SR : ℝ := RS -- This is directly RS

def ratST_SR : ℚ := ((ST : ℚ) / (SR : ℚ)

def u : ℚ := 100
def v : ℚ := 13

theorem proof_u_plus_v :
  u + v = 113 := by
  sorry

end proof_u_plus_v_l766_766026


namespace sqrt_sqrt_4_is_sqrt_2_l766_766011

theorem sqrt_sqrt_4_is_sqrt_2 :
  ∀ x ≥ 0, sqrt (sqrt 4) = ±sqrt 2 :=
begin
  -- Given condition
  let x := sqrt 4,
  have h1 : sqrt 4 = 2 := sqrt_eq_iff_sqr_eq.mpr (by norm_num), -- sqrt 4 = 2
  -- Prove
  have h2 : sqrt x = ±sqrt 2 := sorry
end

end sqrt_sqrt_4_is_sqrt_2_l766_766011


namespace binomial_sum_pattern_l766_766710

theorem binomial_sum_pattern (n : ℕ) (h : 0 < n) : 
  (∑ i in Finset.range n, Nat.choose (2*n - 1) i) = 4^(n-1) := 
by sorry

end binomial_sum_pattern_l766_766710


namespace line_always_passes_through_fixed_point_l766_766442

theorem line_always_passes_through_fixed_point (a : ℝ) : 
  ∀ (x y : ℝ), (ax - y + 1 - 3a = 0) → (x = 3 ∧ y = 1) :=
by {
  intro x y,
  intro h,
  sorry,
}

end line_always_passes_through_fixed_point_l766_766442


namespace cyclic_quadrilateral_perpendicular_l766_766775

theorem cyclic_quadrilateral_perpendicular {A B C D : Point}
  (h_cyclic_quad : CyclicQuadrilateral A B C D)
  (F1 F2 F3 F4 : Point)
  (h_midpoints : 
    MidpointArc A B = F1 ∧ 
    MidpointArc B C = F2 ∧ 
    MidpointArc C D = F3 ∧
    MidpointArc D A = F4):
  Perpendicular (Segment F1 F3) (Segment F2 F4) := 
sorry

end cyclic_quadrilateral_perpendicular_l766_766775


namespace percentage_to_decimal_l766_766080

theorem percentage_to_decimal : (5 / 100 : ℚ) = 0.05 := by
  sorry

end percentage_to_decimal_l766_766080


namespace differentiable_composite_l766_766704

variables {α β : Type*} [NormedField α] [NormedField β]
variables {f : α → α} {g : α → β} {x0 : α} {y0 : β}

def differentiable_at (f : α → β) (x : α) : Prop :=
∃ f', has_deriv_at f f' x

def differentiable (f : α → β) (U : Set α) : Prop :=
∀ x ∈ U, differentiable_at f x

theorem differentiable_composite 
  (hf : differentiable_at f x0)
  (hg : differentiable_at g (f x0))
  (hneigh : ∃ U : Set α, (x0 ∈ U) ∧ (∀ x ∈ U, x ≠ x0 → f x ≠ f x0)) :
  differentiable_at (g ∘ f) x0 ∧ deriv (g ∘ f) x0 = (deriv g (f x0)) * (deriv f x0) :=
sorry

end differentiable_composite_l766_766704


namespace part_I_part_II_part_III_l766_766217

noncomputable def f (x : ℝ) := x / (x^2 - 1)

-- (I) Prove that f(2) = 2/3.
theorem part_I : f 2 = 2 / 3 :=
by sorry

-- (II) Prove that f(x) is decreasing on the interval (-1, 1).
theorem part_II : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 > f x2 :=
by sorry

-- (III) Prove that f(x) is an odd function.
theorem part_III : ∀ x : ℝ, f (-x) = -f x :=
by sorry

end part_I_part_II_part_III_l766_766217


namespace wise_men_task_l766_766747

theorem wise_men_task (n : ℕ) (h : n = 100) (a b c d e f g : ℕ) (Hdistinct : list.nodup [a, b, c, d, e, f, g]) (Hsum : a + b + c + d + e + f + g = n) (Hsorted : [a, b, c, d, e, f, g].sorted (≤)) :
    ∃ A B C D E F G : ℕ, list.nodup [A, B, C, D, E, F, G] ∧ A + B + C + d + E + F + G = 100 ∧ ([A, B, C, d, E, F, G].sorted (≤) ∧ d = d) :=
by
  sorry

end wise_men_task_l766_766747


namespace keystone_arch_larger_angle_l766_766746

def isosceles_trapezoid_larger_angle (n : ℕ) : Prop :=
  n = 10 → ∃ (x : ℝ), x = 99

theorem keystone_arch_larger_angle :
  isosceles_trapezoid_larger_angle 10 :=
by
  sorry

end keystone_arch_larger_angle_l766_766746


namespace simplify_expression_l766_766377

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l766_766377


namespace increasing_odd_function_inequality_l766_766592

noncomputable def f : ℝ → ℝ := sorry  -- Define the function f, the specifics are not required

theorem increasing_odd_function_inequality (f_odd : ∀ x, f (-x) = -f x)
  (f_increasing : ∀ {x y : ℝ}, 0 ≤ x → x ≤ 5 → 0 ≤ y → y ≤ 5 → x < y → f x < f y) :
  f(-3) > f(-π) ∧ f(-π) > f(-4) :=
by
  sorry

end increasing_odd_function_inequality_l766_766592


namespace greeting_cards_problem_l766_766962

-- The Lean statement only, no proof required
theorem greeting_cards_problem :
  let people := Fin 4
  in card (derangements people) = 9 :=
by sorry

end greeting_cards_problem_l766_766962


namespace simplify_expression_l766_766379

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l766_766379


namespace infinite_composite_in_sequence_l766_766692

theorem infinite_composite_in_sequence (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) (h₃ : 1 ≤ b) (h₄ : b ≤ 9) : 
  ∃ᶠ n in at_top, ¬ (nat.prime (nat.of_digits 10 (list.repeat a n ++ [b]))) :=
sorry

end infinite_composite_in_sequence_l766_766692


namespace set_complement_subset_l766_766196

universe u
variable {U : Type u} (M N : Set U)

theorem set_complement_subset
  (h1 : M ⊆ U)
  (h2 : N ⊆ U)
  (h3 : M ∩ N = N) : (U \ M) ⊆ (U \ N) :=
by
  sorry

end set_complement_subset_l766_766196


namespace payback_period_is_165_days_l766_766622

namespace CryptoMining

-- Initial conditions
def cost_system_unit : ℕ := 9499
def cost_graphics_card : ℕ := 31431
def num_graphics_cards : ℕ := 2
def power_system_unit : ℕ := 120
def power_graphics_card : ℕ := 125
def daily_earnings_per_card : ℚ := 0.00877
def value_of_ethereum : ℚ := 27790.37
def electricity_cost : ℚ := 5.38

-- Derived calculations
def total_cost : ℕ := cost_system_unit + num_graphics_cards * cost_graphics_card
def daily_earnings_in_rubles : ℚ := num_graphics_cards * daily_earnings_per_card * value_of_ethereum
def total_power_consumption_kw : ℚ := (power_system_unit + num_graphics_cards * power_graphics_card) / 1000
def daily_energy_consumption_kwh : ℚ := total_power_consumption_kw * 24
def daily_electricity_cost : ℚ := daily_energy_consumption_kwh * electricity_cost
def daily_profit : ℚ := daily_earnings_in_rubles - daily_electricity_cost

-- Proof that the payback period is 165 days
theorem payback_period_is_165_days : (total_cost : ℚ) / daily_profit ≈ 165 := by
  sorry

end CryptoMining

end payback_period_is_165_days_l766_766622


namespace simplify_sqrt_expression_l766_766333

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l766_766333


namespace outer_perimeter_l766_766397

theorem outer_perimeter (F G H I J K L M N : ℕ) 
  (h_outer : F + G + H + I + J = 42) 
  (h_inner : K + L + M = 20) 
  (h_adjustment : N = 4) : 
  F + G + H + I + J - K - L - M + N = 26 := 
by 
  sorry

end outer_perimeter_l766_766397


namespace find_n_find_m_l766_766206

theorem find_n (n : ℕ) (m : ℚ) (h1 : (∑ k in finset.range (n + 1), nat.choose n k) = 64) :
  n = 6 :=
by
  sorry

theorem find_m (m : ℚ) (h2 : ∃ m' : ℚ, m' = (m ^ 3) * (nat.choose 6 3) ∧ (m' = 35 / 16)) :
  m = (7^(1/3)) / 4 :=
by
  sorry

end find_n_find_m_l766_766206


namespace circle_center_radius_sum_l766_766698

theorem circle_center_radius_sum :
  let D := {p : ℝ × ℝ | (p.1^2 - 8 * p.1 + p.2^2 + 14 * p.2 = -28)} in
  ∃ c d s : ℝ, D = {p : ℝ × ℝ | (p.1 - c)^2 + (p.2 + d)^2 = s^2} ∧
  c = 4 ∧
  d = -7 ∧
  s = sqrt 37 ∧
  c + d + s = -3 + sqrt 37 :=
by
  sorry

end circle_center_radius_sum_l766_766698


namespace minimize_intercepts_line_eqn_l766_766870

theorem minimize_intercepts_line_eqn (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : (1:ℝ)/a + (1:ℝ)/b = 1)
  (h2 : ∃ a b, a + b = 4 ∧ a = 2 ∧ b = 2) :
  ∀ (x y : ℝ), x + y - 2 = 0 :=
by 
  sorry

end minimize_intercepts_line_eqn_l766_766870


namespace probability_different_colors_l766_766241

-- Define the total number of chips
def total_chips : ℕ := 6 + 5 + 4 + 3

-- Define the probabilities of drawing each color and then another color
def prob_blue_not_blue := (6 / total_chips : ℚ) * (12 / total_chips)
def prob_red_not_red := (5 / total_chips : ℚ) * (13 / total_chips)
def prob_yellow_not_yellow := (4 / total_chips : ℚ) * (14 / total_chips)
def prob_green_not_green := (3 / total_chips : ℚ) * (15 / total_chips)

-- Sum of probabilities for drawing two chips of different colors
def prob_different_colors := prob_blue_not_blue + prob_red_not_red + prob_yellow_not_yellow + prob_green_not_green

-- The theorem to be proved
theorem probability_different_colors :
  prob_different_colors = 137 / 162 :=
by
  sorry

end probability_different_colors_l766_766241


namespace four_r_eq_sum_abcd_l766_766291

theorem four_r_eq_sum_abcd (a b c d r : ℤ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : (r - a) * (r - b) * (r - c) * (r - d) = 4) : 
  4 * r = a + b + c + d :=
by 
  sorry

end four_r_eq_sum_abcd_l766_766291


namespace simplify_expression_l766_766330

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l766_766330


namespace num_elements_union_set_l766_766191

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}
def union_set : Set ℕ := A ∪ B

theorem num_elements_union_set : union_set.card = 4 := by
  sorry

end num_elements_union_set_l766_766191


namespace parabola_properties_and_slope_product_l766_766607

-- Define the given conditions
def parabola_passing_through (p : ℝ) (P : ℝ × ℝ) : Prop :=
  P.snd ^ 2 = 2 * p * P.fst

def line_intersects_parabola (m : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A.snd^2 = 2 * A.fst ∧ B.snd^2 = 2 * B.fst ∧ (A ≠ B ∧ m * A.snd + 2 = A.fst ∧ m * B.snd + 2 = B.fst)

-- Define the proof problem
theorem parabola_properties_and_slope_product :
  ∀ (p : ℝ) (Q M O A B : ℝ × ℝ),
    parabola_passing_through p Q ∧ Q = (2, 2) ∧ p > 0 ∧
    M = (2, 0) ∧ O = (0, 0) ∧ 
    line_intersects_parabola ((A.snd - 0) / (A.fst - 0)) M ∧ 
    line_intersects_parabola ((B.snd - 0) / (B.fst - 0)) M →
    (∃ (std_eq directrix : ℝ → ℝ), 
       std_eq = (λ y, 2 * y^2) ∧ 
       directrix = (λ x, - (1/2) * x) ∧ 
       ((A.snd / A.fst) * (B.snd / B.fst)) = -1) :=
by
  sorry

end parabola_properties_and_slope_product_l766_766607


namespace city_H_has_highest_increase_l766_766561

-- Conditions
def population_1990_F := 90
def population_2000_F := 120
def adjustment_F := 1.10

def population_1990_G := 80
def population_2000_G := 110
def adjustment_G := 0.95

def population_1990_H := 70
def population_2000_H := 115
def adjustment_H := 1.10

def population_1990_I := 65
def population_2000_I := 100
def adjustment_I := 0.98

def population_1990_J := 95
def population_2000_J := 145
def adjustment_J := 1.0  -- No adjustment

noncomputable def percentage_increase (p1990 p2000 adj: ℝ) : ℝ :=
  ((p2000 * adj) - p1990) / p1990

theorem city_H_has_highest_increase :
  percentage_increase population_1990_H population_2000_H adjustment_H >
  percentage_increase population_1990_F population_2000_F adjustment_F ∧
  percentage_increase population_1990_H population_2000_H adjustment_H >
  percentage_increase population_1990_G population_2000_G adjustment_G ∧
  percentage_increase population_1990_H population_2000_H adjustment_H >
  percentage_increase population_1990_I population_2000_I adjustment_I ∧
  percentage_increase population_1990_H population_2000_H adjustment_H >
  percentage_increase population_1990_J population_2000_J adjustment_J :=
  sorry

end city_H_has_highest_increase_l766_766561


namespace units_digit_of_result_l766_766743

theorem units_digit_of_result (a b c : ℕ) (h1 : a = c + 3) : 
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  let result := original - reversed
  result % 10 = 7 :=
by
  sorry

end units_digit_of_result_l766_766743


namespace function_is_odd_and_periodic_l766_766401

noncomputable def f (x : ℝ) : ℝ := cos^2 (x + π / 4) - sin^2 (x + π / 4)

theorem function_is_odd_and_periodic : 
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∃ T : ℝ, T = π ∧ ∀ x : ℝ, f (x + T) = f x) :=
by
  sorry

end function_is_odd_and_periodic_l766_766401


namespace approx_average_sqft_per_person_l766_766120

noncomputable def average_sqft_per_person 
  (population : ℕ) 
  (land_area_sqmi : ℕ) 
  (sqft_per_sqmi : ℕ) : ℕ :=
(sqft_per_sqmi * land_area_sqmi) / population

theorem approx_average_sqft_per_person :
  average_sqft_per_person 331000000 3796742 (5280 ^ 2) = 319697 := 
sorry

end approx_average_sqft_per_person_l766_766120


namespace distance_orthocenter_incenter_l766_766084

-- Define the conditions of the problem
variables (A B C D E F H I : Type) [LinearOrderedField A]

-- Define the points and distances as given
def AB : A := 20
def BC : A := 15
def CA : A := 7

-- Define the distance between orthocenter and incenter
def dist_orthocenter_incenter : A := 15

-- The theorem to be proved
theorem distance_orthocenter_incenter
  (h1 : A)
  (h2 : B)
  (h3 : C)
  (h4 : D)
  (h5 : E)
  (h6 : F)
  (h_orthocenter : H = orthocenter_of_triangle h1 h2 h3)
  (h_incenter : I = incenter_of_triangle h4 h5 h6) :
  distance H I = dist_orthocenter_incenter :=
sorry

end distance_orthocenter_incenter_l766_766084


namespace gloria_initial_dimes_l766_766619

variable (Q D : ℕ)

theorem gloria_initial_dimes (h1 : D = 5 * Q) 
                             (h2 : (3 * Q) / 5 + D = 392) : 
                             D = 350 := 
by {
  sorry
}

end gloria_initial_dimes_l766_766619


namespace dave_files_left_l766_766135

theorem dave_files_left 
  (initial_apps : ℕ) 
  (initial_files : ℕ) 
  (apps_left : ℕ)
  (files_more_than_apps : ℕ) 
  (h1 : initial_apps = 11) 
  (h2 : initial_files = 3) 
  (h3 : apps_left = 2)
  (h4 : files_more_than_apps = 22) 
  : ∃ (files_left : ℕ), files_left = apps_left + files_more_than_apps :=
by
  use 24
  sorry

end dave_files_left_l766_766135


namespace area_of_square_KLMN_l766_766668

theorem area_of_square_KLMN :
  let side_length_ABCD := Real.sqrt 36 in
  let area_ABCD := side_length_ABCD ^ 2 in
  let side_length_ABCD := 6 in
  let area_triangle := (1 / 2) * 2 * 4 in
  let total_area_removed := 4 * area_triangle in
  let area_KLMN := area_ABCD - total_area_removed in
  area_KLMN = 20 := 
by
  have h1 : side_length_ABCD = Real.sqrt 36 := rfl
  have h2 : area_ABCD = side_length_ABCD ^ 2 := rfl
  have h3 : side_length_ABCD = 6 := rfl
  have h4 : area_triangle = (1 / 2) * 2 * 4 := rfl
  have h5 : total_area_removed = 4 * area_triangle := rfl
  have h6 : area_KLMN = area_ABCD - total_area_removed := rfl
  let area_ABCD := 6^2 in
  let area_KLMN := 36 - 16 in
  show area_KLMN = 20 from rfl

end area_of_square_KLMN_l766_766668


namespace bus_interval_three_buses_l766_766033

theorem bus_interval_three_buses (T : ℕ) (h : T = 21) : (T * 2) / 3 = 14 :=
by
  sorry

end bus_interval_three_buses_l766_766033


namespace construct_intersection_point_l766_766976

/-- Given a line l and a segment OA, parallel to l, using only a double-sided ruler, 
    construct the points of intersection of the line l with a circle of radius OA centered at O. -/
theorem construct_intersection_point (l : Line) (O A : Point) (h1 : parallel l OA) :
  ∃ X, is_intersection X l (circle O OA) :=
sorry

end construct_intersection_point_l766_766976


namespace identity_proof_geometric_proof_l766_766718

variables {E : Type*} [inner_product_space ℝ E]
variables (a b : E) (R m p : ℝ)
variables (C A B M : E)

-- Identity proof
theorem identity_proof :
  (inner_product_space ℝ E) → 
  (a.inner b) = ((1/(2:ℝ)) * a + (1/(2:ℝ)) * b) ⬝ ((1/(2:ℝ)) * a +  (1/(2:ℝ)) * b) - ((1/(2:ℝ)) * a - (1/(2:ℝ)) * b) ⬝ ((1/(2:ℝ)) * a - (1/(2:ℝ)) * b) :=
sorry

-- Geometric problem proof
theorem geometric_proof :
  (dist C M = m) → 
  ((M - A).inner (M - B) = m^2 - R^2) :=
sorry

end identity_proof_geometric_proof_l766_766718


namespace smallest_possible_stamps_l766_766713

theorem smallest_possible_stamps (M : ℕ) : 
  ((M % 5 = 2) ∧ (M % 7 = 2) ∧ (M % 9 = 2) ∧ (M > 2)) → M = 317 := 
by 
  sorry

end smallest_possible_stamps_l766_766713


namespace compute_expression_l766_766898

-- Defining notation for the problem expression and answer simplification
theorem compute_expression : 
    9 * (2/3)^4 = 16/9 := by 
  sorry

end compute_expression_l766_766898


namespace greatest_diff_units_digit_of_multiple_of_4_l766_766766

theorem greatest_diff_units_digit_of_multiple_of_4 : 
  let valid_units_digits := { d : ℕ // 0 ≤ d ∧ d ≤ 9 ∧ (720 + d) % 4 = 0 } in
  ∃ d1 d2 ∈ valid_units_digits, d1 ≠ d2 ∧ d1 - d2 = 8 :=
by
  sorry

end greatest_diff_units_digit_of_multiple_of_4_l766_766766


namespace probability_divisible_triplet_l766_766420

theorem probability_divisible_triplet :
  let S := {1, 2, 3, 4, 5, 6}
  let triples := { (a, b, c) | a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a < b ∧ b < c }
  let valid_triples := { (a, b, c) ∈ triples | a ∣ b ∧ a ∣ c }
  let num_triples := (triples.toFinset.card : ℚ)
  let num_valid_triples := (valid_triples.toFinset.card : ℚ)
  (num_valid_triples / num_triples) = 11 / 20 := 
  sorry

end probability_divisible_triplet_l766_766420


namespace intersection_conditions_length_MN_l766_766573

noncomputable def line (k : ℝ) : (ℝ × ℝ) → Prop := λ p, p.2 = k * p.1 + 1
def circle (p : ℝ × ℝ) := (p.1 - 2)^2 + (p.2 - 3)^2 = 1

theorem intersection_conditions 
    (k : ℝ)
    (h1 : ∀ p, line k p → circle p) :
  (∃ M N, line k M ∧ line k N ∧ circle M ∧ circle N ∧ M ≠ N) 
    ↔ (4 - real.sqrt 7) / 3 < k ∧ k < (4 + real.sqrt 7) / 3 :=
sorry

theorem length_MN 
    (k : ℝ)
    (h1 : ∀ p, line k p → circle p)
    (h2 : ∃ M N, line k M ∧ line k N ∧ circle M ∧ circle N ∧ M ≠ N ∧ (M.1 * N.1 + M.2 * N.2 = 12)) :
  |(h2.some).1.1 - (h2.some).2.1| + |(h2.some).1.2 - (h2.some).2.2| = 2 :=
sorry

end intersection_conditions_length_MN_l766_766573


namespace baby_hippos_per_female_hippo_l766_766781

theorem baby_hippos_per_female_hippo 
  (newborns_diff : ℕ)
  (start_elephants : ℕ)
  (start_hippos : ℕ)
  (total_female_hippos : ℕ)
  (total_animals_post_birth : ℕ) 
  (h1 : newborns_diff = 10)
  (h2 : start_elephants = 20)
  (h3 : start_hippos = 35)
  (h4 : total_female_hippos = 5 / 7 * start_hippos)
  (h5 : total_animals_post_birth = 315) :
  let baby_hippos_per_female := (total_animals_post_birth - start_elephants - start_hippos - h1) / (2 * total_female_hippos) in
  baby_hippos_per_female = 5 := 
by
  sorry

end baby_hippos_per_female_hippo_l766_766781


namespace three_buses_interval_l766_766037

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end three_buses_interval_l766_766037


namespace simplify_expression_l766_766386

theorem simplify_expression 
  (a b c : ℝ) 
  (h1 : a = sqrt 7)
  (h2 : b = 2 * sqrt 7)
  (h3 : c = 3 * sqrt 7) :
  a - b + c = 2 * sqrt 7 :=
by
  sorry

end simplify_expression_l766_766386


namespace x_value_satisfies_l766_766917

noncomputable def x_solution (c d : ℂ) (x : ℝ) :=
  ∃ c d : ℂ,
    |c| = 3 ∧
    |d| = 5 ∧
    x = 6 * real.sqrt 6 ∧
    cd = x - 3 * complex.I

theorem x_value_satisfies (c d : ℂ) (x : ℝ) (h1 : |c| = 3) (h2 : |d| = 5) :
  x_solution c d x → x = 6 * real.sqrt 6 :=
by
  sorry

end x_value_satisfies_l766_766917


namespace stack_height_l766_766961

theorem stack_height (d : ℝ) (r : ℝ) (h_eq : ℝ) (h_stack : ℝ) : 
  d = 12 → 
  r = d / 2 → 
  h_eq = (real.sqrt 3) * (d / 2) → 
  h_stack = r + h_eq + r → 
  h_stack = 12 + 6 * real.sqrt 3 := 
by 
  intros
  subst_vars

-- To automatically end the proof and check the structure
sorry

end stack_height_l766_766961


namespace handshake_count_l766_766513

theorem handshake_count 
  (married_couples : ℕ)
  (only_women_shake : ∀ (men : ℕ), ∀ (women : ℕ), women = 15 → men = 15 → true)
  (no_spouse_handshake : ∀ (x : ℕ), ∀ (y : ℕ), x ≠ y → true)
  (no_men_handshake : ∀ (x : ℕ), ∀ (y : ℕ), x ≠ y → false)
  : ∑ i in (finset.range 15).p, ∑ j in (finset.range 15).p, i ≠ j = 105 :=
by
  sorry

end handshake_count_l766_766513


namespace terminal_side_point_sin_eq_3_over_5_l766_766999

theorem terminal_side_point_sin_eq_3_over_5 (m : ℝ) (θ : ℝ) (h1 : ∃ (P : ℝ × ℝ), P = (4, m) ∧ P ∈ { P | ∃ θ : ℝ, sin θ = 3/5 })
  (h2 : 4^2 + m^2 = (5 : ℝ)^2) : m = 3 :=
sorry

end terminal_side_point_sin_eq_3_over_5_l766_766999


namespace sin_alpha_minus_pi_div_4_tan_2alpha_l766_766988

variable (α : ℝ)

-- Conditions
def sin_alpha := sin α = 4 / 5
def alpha_in_interval := α ∈ Ioo (π / 2) π

-- Problems
theorem sin_alpha_minus_pi_div_4 
  (h1 : sin_alpha α) 
  (h2 : alpha_in_interval α) : 
  sin (α - π / 4) = (7 * Real.sqrt 2) / 10 := 
sorry

theorem tan_2alpha 
  (h1 : sin_alpha α) 
  (h2 : alpha_in_interval α) : 
  tan (2 * α) = 24 / 7 := 
sorry

end sin_alpha_minus_pi_div_4_tan_2alpha_l766_766988


namespace factorization_option_D_l766_766446

-- Define variables
variables (x y : ℝ)

-- Define the expressions
def left_side_D := -4 * x^2 + 12 * x * y - 9 * y^2
def right_side_D := -(2 * x - 3 * y)^2

-- Theorem statement
theorem factorization_option_D : left_side_D x y = right_side_D x y :=
sorry

end factorization_option_D_l766_766446


namespace max_range_of_temps_l766_766456

-- Define the conditions
def average_temp (temps : List ℝ) : ℝ := (temps.sum) / (temps.length)
noncomputable def min_temp (temps : List ℝ) : ℝ := temps.foldl min (temps.head?.getOrElse 0)

-- Given conditions
variable (temps : List ℝ)
variable (h_length : temps.length = 5)
variable (h_avg : average_temp temps = 50)
variable (h_min : min_temp temps = 40)

-- Statement: Prove that the possible maximum range of the temperatures is 50
theorem max_range_of_temps : (temps.maximumD 0 - temps.minimumD 0) = 50 :=
by sorry

end max_range_of_temps_l766_766456


namespace find_min_f1_f2_l766_766293

theorem find_min_f1_f2
  (f : ℤ → ℤ)
  (h1 : ∀ x y : ℤ, f (x^2 - 3 * y^2) + f (x^2 + y^2) = 2 * (x + y) * f (x - y))
  (h2 : ∀ n : ℤ, 0 < n → 0 < f n)
  (h3 : nat.gcd (f 2015) (f 2016) ^ 2 = f 2015 * f 2016) :
  f 1 + f 2 = 246 :=
sorry

end find_min_f1_f2_l766_766293


namespace no_valid_sequences_l766_766978

noncomputable def sequence_problem (m : ℕ) (n : ℕ) : Prop :=
  ¬ ∃ (a : Fin n → ℝ) (x : Fin n → ℝ),
    (∀ i, 0 ≤ a i) ∧
    (∀ i, x i ≠ 0) ∧
    x (Fin.last n) = Finset.max' (Finset.univ.image x) (by simp) ∧
    (∀ i : Fin (n - 1), a i > m + 1 - (x ⟨i + 1, by simp [Nat.succ_lt_succ_iff]⟩ / x i)) ∧
    a (Fin.last n) > m + 1 - (∑ i in Finset.univ.erase (Fin.last n), x i / x (Fin.last n)) ∧
    (∑ i in Finset.univ, a i) ≤ m + 1 - 1 / m

theorem no_valid_sequences (m : ℕ) (n : ℕ) (hm : 0 < m) (hn : 0 < n) : sequence_problem m n :=
begin
  sorry
end

end no_valid_sequences_l766_766978


namespace proposition_relationship_l766_766203

variables (A B C : Prop)

theorem proposition_relationship :
  (A → B) ∧ ¬(B → A) ∧ ((B ↔ C)) → (C → A) ∧ ¬(A → C) :=
begin
  sorry
end

end proposition_relationship_l766_766203


namespace baba_yaga_students_see_l766_766887

variable (B G : ℕ)

theorem baba_yaga_students_see 
(h1 : B + G = 33)
(h2 : ∀ b, b ∈ finset.range B → closed_right_eye b)
(h3 : ∀ g, g ∈ finset.range (G / 3) → closed_right_eye g)
(h4 : ∀ g, g ∈ finset.range G → closed_left_eye g)
(h5 : ∀ b, b ∈ finset.range (B / 3) → closed_left_eye b) :
  ∃ n, n = 22 := by
  sorry

end baba_yaga_students_see_l766_766887


namespace BD_eq_2XY_l766_766257

open EuclideanGeometry

variables {A B C D P X Y : Point}
variables {BD AP AXB ADB AYD ABD APB CPD : Angle}

-- Given conditions
axiom angle_ABC_eq_90 : ∠ ABC = 90
axiom angle_ADC_eq_90 : ∠ ADC = 90
axiom angle_APB_eq_twice_CPD : ∠ APB = 2 * ∠ CPD
axiom angle_AXB_eq_twice_ADB : ∠ AXB = 2 * ∠ ADB
axiom angle_AYD_eq_twice_ABD : ∠ AYD = 2 * ∠ ABD

-- The theorem to be proved
theorem BD_eq_2XY : BD = 2 * XY := by sorry

end BD_eq_2XY_l766_766257


namespace length_FL_4_5_l766_766260

-- Definitions based on conditions
def RightAngledTriangle (A B C : Point) : Prop :=
  right_angle A B C ∨ right_angle B C A ∨ right_angle C A B

noncomputable def length (A B : Point) : Real := sorry

def isMidpoint (M A B : Point) : Prop :=
  length A M = length M B

def isAltitude (D A B C : Point) : Prop :=
  right_angle A D C

-- Define points and conditions
variables (A B C D E F L : Point)
variables (hABC : RightAngledTriangle A B C)
variables (hAB_eq_6 : length A B = 6)
variables (hAC_eq_6 : length A C = 6)
variables (hAltitude : isAltitude D A B C)
variables (hBE_median : isMidpoint E B C)
variables (hF_int : ∃ F, liesOnLine F A D ∧ liesOnLine F B E)
variables (hL_on_FD : liesOnLine L F D)

-- Main theorem to prove
theorem length_FL_4_5 : length F L = 4.5 :=
sorry

end length_FL_4_5_l766_766260


namespace find_value_of_xy_l766_766989

-- Define the given conditions and declaration of the proof statement
theorem find_value_of_xy (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h_distinct : x ≠ y) (h_eq : x^2 + 2 / x = y + 2 / y) : x * y = 2 :=
sorry

end find_value_of_xy_l766_766989


namespace angle_MTN_range_l766_766132

-- Definitions and conditions
def IsoscelesTriangle (P Q R : Point) : Prop :=
  dist P Q = 6 ∧ dist P R = 10 ∧ dist Q R = 10

def RollingCircle (P Q R T M N : Point) (r : ℝ) : Prop :=
  r = 3 ∧ Circle r R T ∧ Intersection PQ M ∧ Intersection QR N

-- Theorem statement
theorem angle_MTN_range (P Q R T M N : Point)
  (h_triangle : IsoscelesTriangle P Q R)
  (h_circle : RollingCircle P Q R T M N 3) :
  0 ≤ angle M T N ∧ angle M T N ≤ 90 :=
by sorry

end angle_MTN_range_l766_766132


namespace find_a_l766_766610

-- Define the conditions and the theorem
def is_power_function (f : ℝ → ℝ) : Prop :=
∃ a : ℝ, ∃ b : ℝ, f = λ x, b * x^a

theorem find_a (a : ℝ) :
    let f := (λ x : ℝ, (a^2 - 9a + 19) * x^(2*a - 9)) in
    is_power_function f ∧ ∀ x : ℝ, f x ≠ 0 → x ≠ 0 → f 0 = 0 → x ≠ 0 → a = 3 := 
by 
    sorry

end find_a_l766_766610


namespace knit_time_for_mitten_l766_766707

theorem knit_time_for_mitten :
  ∃ x : ℝ, (3 * (14 + 2 * x) = 48) ∧ x = 1 :=
by
  use 1
  split
  · calc
      3 * (14 + 2 * 1) = 3 * 16 := by rfl
      _                = 48       := by norm_num
  · rfl

end knit_time_for_mitten_l766_766707


namespace isosceles_triangle_perimeter_l766_766980

/-
Problem:
Given an isosceles triangle with side lengths 5 and 6, prove that the perimeter of the triangle is either 16 or 17.
-/

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 5 ∨ a = 6) (h₂ : b = 5 ∨ b = 6) (h₃ : a ≠ b) : 
  (a + a + b = 16 ∨ a + a + b = 17) ∧ (b + b + a = 16 ∨ b + b + a = 17) :=
by
  sorry

end isosceles_triangle_perimeter_l766_766980


namespace vector_angle_sin_eq_l766_766282

variable {V : Type*} [inner_product_space ℝ V]

variables (a b c : V) (θ : ℝ)

theorem vector_angle_sin_eq (ha : ∥a∥ = 2) (hb : ∥b∥ = 7) (hc : ∥c∥ = 6)
  (h_ax_a_cross_b : a × (a × b) = 2 • c) :
  real.sin (inner_product_space.angle a b) = 6 / 7 :=
sorry

end vector_angle_sin_eq_l766_766282


namespace intersection_distance_of_C1_C2_l766_766669

noncomputable theory

open Real

def parametric_C1 (α : ℝ) : ℝ × ℝ :=
  let x := 1 + sqrt 3 * cos α
  let y := sqrt 3 * sin α
  (x, y)

def polar_C2 (θ : ℝ) : ℝ × ℝ :=
  let ρ := 2 * sin θ
  let x := ρ * cos θ
  let y := ρ * sin θ
  (x, y)

theorem intersection_distance_of_C1_C2 :
  (∃ α θ : ℝ, (parametric_C1 α).fst = (polar_C2 θ).fst ∧ (parametric_C1 α).snd = (polar_C2 θ).snd) →
  let d := 2 in d = 2 :=
  sorry

end intersection_distance_of_C1_C2_l766_766669


namespace problem_statement_l766_766725

-- Define the geometric configuration
variables {A B C O A' A1 A2 HA : Point}
variable [scalene_ABC : scalene_triangle A B C]
variable [circumcenter : is_circumcenter O A B C]
variable [on_extension : is_point_on_extension A' A O]
variable [angle_condition : angle_BAA' = angle_CAA']
variable [perpendicular_A1 : is_foot_of_perpendicular A' A1 A B]
variable [perpendicular_A2 : is_foot_of_perpendicular A' A2 A C]
variable [foot_HA : is_foot_of_perpendicular HA A B C]
variable {RA RB RC R : ℝ}

-- Definitions of radii of circumcircles
variable [circumradius_RA : is_circumradius RA triangle H_A A1 A2]
variable [circumradius_RB : is_circumradius RB triangle HB B1 B2]
variable [circumradius_RC : is_circumradius RC triangle HC C1 C2]
variable [circumradius_R : is_circumradius R triangle A B C]

-- The final proof statement
theorem problem_statement :
  (1 / RA) + (1 / RB) + (1 / RC) = (2 / R) :=
sorry

end problem_statement_l766_766725


namespace age_of_15th_person_l766_766828

theorem age_of_15th_person (avg_16 : ℝ) (avg_5 : ℝ) (avg_9 : ℝ) (total_16 : ℝ) (total_5 : ℝ) (total_9 : ℝ) :
  avg_16 = 15 ∧ avg_5 = 14 ∧ avg_9 = 16 ∧
  total_16 = 16 * avg_16 ∧ total_5 = 5 * avg_5 ∧ total_9 = 9 * avg_9 →
  (total_16 - total_5 - total_9) = 26 :=
by
  sorry

end age_of_15th_person_l766_766828


namespace max_is_a_plus_b_l766_766987

theorem max_is_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) (h5 : a ≠ b) :
  max (max (max (a + b) (2 * real.sqrt (a * b))) (a^2 + b^2)) (2 * a * b) = a + b :=
sorry

end max_is_a_plus_b_l766_766987


namespace shorten_by_one_expected_length_l766_766851

-- Definition of the random sequence and binomial distribution properties
def digit_sequence {n : ℕ} (p : ℝ) (len : ℕ) : ℝ := 
  let q := 1 - p in
  (len.choose 1) * (p^1 * q^(len - 1))

-- Part (a) statement
theorem shorten_by_one 
  (len : ℕ) 
  (p : ℝ := 0.1) 
  (q := 1 - p) 
  (x := len - 1) : 
  digit_sequence p x = 1.564e-90 :=
by sorry

-- Part (b) statement
theorem expected_length 
  (initial_len : ℕ) 
  (p : ℝ := 0.1) (removals := initial_len - 1) 
  (expected_removed : ℝ := removals * p) : 
  (initial_len : ℝ) - expected_removed = 1813.6 :=
by sorry

end shorten_by_one_expected_length_l766_766851


namespace pressure_force_calc_l766_766524

/-!
 # Pressure Force Calculation

 We are proving that the pressure force P exerted by water on a vertically submerged plate 
 in the shape of a triangle is given by \( P = \frac{\gamma b h^3}{3} \) given certain conditions.
-/

noncomputable def pressure_force (b h γ : ℝ) : ℝ :=
  let integral := ∫ x in 0..h, (x^2 * (b / h))
  (γ * b * integral / h)

theorem pressure_force_calc (b h γ : ℝ) : 
  pressure_force b h γ = γ * b * h^2 / 3 :=
by
  sorry

end pressure_force_calc_l766_766524


namespace simplify_sqrt_expression_l766_766335

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l766_766335


namespace simplify_expr_l766_766344

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l766_766344


namespace complex_fraction_plus_conjugate_l766_766990

-- Define the complex number z
def z : ℂ := 1 + complex.i

-- State the theorem to prove the required equation
theorem complex_fraction_plus_conjugate (z := 1 + complex.i) : (1 / z + conj z = 3 / 2 - (3 / 2) * complex.i) :=
by
  sorry

end complex_fraction_plus_conjugate_l766_766990


namespace question_proof_l766_766950

theorem question_proof:
  ∀ (n m : ℕ),
  ∃ nums : list (vector ℕ n),
  (∀ a b : vector ℕ n, a ∈ nums ∧ b ∈ nums ∧ a ≠ b → ∃ m_positions : finset (fin n), m_positions.card = m ∧ ∀ i ∈ m_positions, a.nth i = b.nth i) ∧
  (∀ i : fin n, ∃ (not_all_same : finset ℕ), not_all_same.card = 2 ∧ ∀ num ∈ nums, num.nth i ∈ not_all_same) →
  (2 / 5 : ℚ) ≤ (m / n : ℚ) ∧ (m / n : ℚ) ≤ (8 / 5 : ℚ) := by
  sorry

end question_proof_l766_766950


namespace main_theorem_l766_766236

noncomputable def proof_P_X_le_neg_1 : Prop :=
  ∀ (X : ℝ → ℝ) (σ : ℝ),
    (∃ (μ : ℝ) (σ2 : ℝ), X = λ x, (μ + σ2 * x) ∧ μ = 2 ∧ σ2 = σ) ∧ 
    (probability (X ≤ 5) = 0.8) →
    (probability (X ≤ -1) = 0.2)

theorem main_theorem : proof_P_X_le_neg_1 :=
  sorry

end main_theorem_l766_766236


namespace machines_job_completion_time_l766_766094

theorem machines_job_completion_time (t : ℕ) 
  (hR_rate : ∀ t, 1 / t = 1 / 216) 
  (hS_rate : ∀ t, 1 / t = 1 / 216) 
  (same_num_machines : ∀ R S, R = 9 ∧ S = 9) 
  (total_time : 12 = 12) 
  (jobs_completed : 1 = (18 / t) * 12) : 
  t = 216 := 
sorry

end machines_job_completion_time_l766_766094


namespace find_dividend_l766_766457

theorem find_dividend : 
  ∃ d : ℕ, let divisor := 14 in 
           let quotient := 12 in 
           let remainder := 8 in 
           d = (divisor * quotient) + remainder :=
begin
  use 176,
  sorry,
end

end find_dividend_l766_766457


namespace problem_statement_l766_766192

open Real

def floor_sum_cubed_roots : ℕ := 
  ∑ k in Finset.range 8000, ⌊(k : ℝ)^(1/3)⌋

theorem problem_statement : 
  ⌊(floor_sum_cubed_roots : ℝ) / 100⌋ = 1159 := 
sorry

end problem_statement_l766_766192


namespace a2017_is_1_over_65_l766_766119

noncomputable def fraction_sequence (n : ℕ) : ℚ :=
  let seq := List.range (n + 1)
            |> List.filterMap (fun d => if d > 1 then some (List.range d |> List.tail!) else none)
            |> List.join
  seq.get? n

theorem a2017_is_1_over_65 : fraction_sequence 2017 = 1 / 65 := by
  sorry

end a2017_is_1_over_65_l766_766119


namespace hyperbola_asymptotes_m_value_l766_766159

theorem hyperbola_asymptotes_m_value : 
    (∀ x y : ℝ, (x^2 / 144 - y^2 / 81 = 1) → (y = (3/4) * x ∨ y = -(3/4) * x)) := 
by sorry

end hyperbola_asymptotes_m_value_l766_766159


namespace arithmetic_expression_value_l766_766804

theorem arithmetic_expression_value : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_expression_value_l766_766804


namespace simplify_radical_expression_l766_766356

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l766_766356


namespace area_change_900_l766_766492

variables (l w : ℕ)
hypothesis hw : 2 * l + 2 * w = 160

theorem area_change_900 (hlw : l + w = 80) : 
  let A_original := l * w,
      A_new := (l + 10) * (w + 10),
      Delta_A := A_new - A_original
  in Delta_A = 900 :=
by 
  sorry

end area_change_900_l766_766492


namespace repeating_decimal_divisible_by_2_or_5_l766_766413

theorem repeating_decimal_divisible_by_2_or_5 
  (a b k pq u m n : ℕ) 
  (repeating_decimal : (∃ m n : ℕ, nat.gcd m n = 1 ∧ 0.ab ⋯ k⟦pq…u⟧ = m / n) 
  (has_non_repeating_part : ∃ k > 0)
  : n % 2 = 0 ∨ n % 5 = 0 :=
by
  sorry

end repeating_decimal_divisible_by_2_or_5_l766_766413


namespace find_general_solution_l766_766556

-- Defining vectors 
abbreviation Vector4 := Fin 4 → ℝ

def system (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 + 7 * x2 - 8 * x3 + 9 * x4 = 0 ∧
  2 * x1 - 3 * x2 + 3 * x3 - 2 * x4 = 0 ∧
  4 * x1 + 11 * x2 - 13 * x3 + 16 * x4 = 0 ∧
  7 * x1 - 2 * x2 + x3 + 3 * x4 = 0

def solution1 : Vector4 := ![3/17, 19/17, 1, 0]
def solution2 : Vector4 := ![-13/17, -20/17, 0, 1]

theorem find_general_solution : 
  ∃ α β : ℝ, ∀ x1 x2 x3 x4 : ℝ, (system x1 x2 x3 x4) →
  (∃ a b : ℝ, @Fin.vecFromVec α β solution1 + @Fin.vecFromVec a b solution2 = ![x1, x2, x3, x4]) :=
sorry

end find_general_solution_l766_766556


namespace rounding_incorrect_option_B_l766_766435

theorem rounding_incorrect_option_B : 
  (round_nearest (0.05049) (0.1) = 0.1) ∧
  (round_nearest (0.05049) (0.001) ≠ 0.051) ∧
  (round_nearest (0.05049) (0.01) = 0.05) ∧
  (round_nearest (0.05049) (0.0001) = 0.0505) →
  true :=
by
  -- proof will be provided here
  sorry

end rounding_incorrect_option_B_l766_766435


namespace equal_areas_triangle_height_l766_766745

theorem equal_areas_triangle_height (l b h : ℝ) (hlb : l > b) 
  (H1 : l * b = (1/2) * l * h) : h = 2 * b :=
by 
  -- skipping proof
  sorry

end equal_areas_triangle_height_l766_766745


namespace eulers_formula_l766_766702

open Set

theorem eulers_formula (V E F : ℕ) (h_connectivity : ConnectedGraph G)
  (h_planarity : PlanarGraph G)
  (h_vertices : card (vertices G) = V)
  (h_edges : card (edges G) = E)
  (h_faces : card (faces G) = F) :
  V - E + F = 2 :=
sorry

end eulers_formula_l766_766702


namespace eval_expression_l766_766809

theorem eval_expression : 4 * (8 - 3) - 7 = 13 :=
by
  sorry

end eval_expression_l766_766809


namespace rational_solution_counts_l766_766943

theorem rational_solution_counts :
  (∃ (x y : ℚ), x^2 + y^2 = 2) ∧ 
  (¬ ∃ (x y : ℚ), x^2 + y^2 = 3) := 
by 
  sorry

end rational_solution_counts_l766_766943


namespace compute_fraction_power_l766_766899

theorem compute_fraction_power :
  9 * (2/3)^4 = 16/9 := 
by
  sorry

end compute_fraction_power_l766_766899


namespace simplify_fractions_l766_766724

theorem simplify_fractions : 
  (150 / 225) + (90 / 135) = 4 / 3 := by 
  sorry

end simplify_fractions_l766_766724


namespace customer_total_payment_l766_766095

def Riqing_Beef_Noodles_quantity : ℕ := 24
def Riqing_Beef_Noodles_price_per_bag : ℝ := 1.80
def Riqing_Beef_Noodles_discount : ℝ := 0.8

def Kang_Shifu_Ice_Red_Tea_quantity : ℕ := 6
def Kang_Shifu_Ice_Red_Tea_price_per_box : ℝ := 1.70
def Kang_Shifu_Ice_Red_Tea_discount : ℝ := 0.8

def Shanlin_Purple_Cabbage_Soup_quantity : ℕ := 5
def Shanlin_Purple_Cabbage_Soup_price_per_bag : ℝ := 3.40

def Shuanghui_Ham_Sausage_quantity : ℕ := 3
def Shuanghui_Ham_Sausage_price_per_bag : ℝ := 11.20
def Shuanghui_Ham_Sausage_discount : ℝ := 0.9

def total_price : ℝ :=
  (Riqing_Beef_Noodles_quantity * Riqing_Beef_Noodles_price_per_bag * Riqing_Beef_Noodles_discount) +
  (Kang_Shifu_Ice_Red_Tea_quantity * Kang_Shifu_Ice_Red_Tea_price_per_box * Kang_Shifu_Ice_Red_Tea_discount) +
  (Shanlin_Purple_Cabbage_Soup_quantity * Shanlin_Purple_Cabbage_Soup_price_per_bag) +
  (Shuanghui_Ham_Sausage_quantity * Shuanghui_Ham_Sausage_price_per_bag * Shuanghui_Ham_Sausage_discount)

theorem customer_total_payment :
  total_price = 89.96 :=
by
  unfold total_price
  sorry

end customer_total_payment_l766_766095


namespace simplify_sqrt_expression_l766_766336

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l766_766336


namespace not_transform_1_to_811_l766_766872

theorem not_transform_1_to_811 :
  ∀ (n : ℕ), ¬ (n = 1 → (∃ m, (m = 811 ∧ can_transform_by_multiplication_and_permutation n m)))
  :=
  sorry

/-- Defines the allowed operations: multiplying by 2 and rearranging digits. -/
def can_transform_by_multiplication_and_permutation (a b : ℕ) : Prop :=
  (a = 2 * b ∨ a = b ∨ b = 2 * a) ∧ valid_permutation a b

/-- Defines what constitutes a valid permutation of the digits. -/
def valid_permutation (a b : ℕ) : Prop :=
  let digits_a := (a.digits 10).erase 0
  let digits_b := (b.digits 10).erase 0
  digits_a ~ digits_b  -- permutation relation, ~ denotes they are permutations of each other

end not_transform_1_to_811_l766_766872


namespace vector_parallel_x_l766_766964

theorem vector_parallel_x (x : ℝ) :
  let a := (x, 2)
  let b := (1, 6)
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → x = 1 / 3 :=
by { intros h, sorry }

end vector_parallel_x_l766_766964


namespace simplify_expr_l766_766347

theorem simplify_expr : sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  sorry

end simplify_expr_l766_766347


namespace hyperbola_eccentricity_l766_766748

variable {a b : ℝ}
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (l : line)
variable (hyperbola : conic_section)
variable (h_intersection : line_intersects_conic l hyperbola)
variable (angle_equal : ∀ A O C, ∠AOC = ∠BOC → ∠AOC = 60)

-- Define the line and hyperbola based on the description provided
def line := {y : ℝ | y = 2 * b}

def hyperbola := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- The goal is to prove the eccentricity of the hyperbola
theorem hyperbola_eccentricity : ∃ e : ℝ, e = sqrt 19 / 2 :=
by
  -- Given conditions and known result
  have hAOC : ∠AOC = 60 := by sorry
  have hC : C = (2 * sqrt 3 / 3 * b, 2 * b) := by sorry
  have hyp_eq : ((2 * sqrt 3 / 3 * b)^2 / a^2) - (4 * b^2 / b^2) = 1 := by sorry
  have b_eq : b = (sqrt 15 / 2) * a := by sorry
  have c_eq : c = sqrt (a^2 + b^2) := by sorry
  have final_eccentricity : e = (sqrt 19) / 2 := by sorry
  exact ⟨final_eccentricity⟩

end hyperbola_eccentricity_l766_766748


namespace periodic_decimal_sum_l766_766549

theorem periodic_decimal_sum : (0.2).periodic + (0.02).periodic = 8 / 33 :=
by
  -- Defining the periodic decimal representations in terms of their fractional forms:
  let a : ℚ := 2 / 9 -- equivalent to 0.\overline{2}
  let b : ℚ := 2 / 99 -- equivalent to 0.\overline{02}
  -- Asserting the sum of these fractions:
  have h : a + b = 24 / 99, by sorry
  -- Reducing the fraction to its lowest terms:
  have h' : 24 / 99 = 8 / 33, by sorry
  -- Concluding the equality:
  exact Eq.trans h h'

end periodic_decimal_sum_l766_766549


namespace problem_equiv_l766_766602

noncomputable def f (x : ℝ) : ℝ := real.log (real.sqrt (1 + x^2) - x)

theorem problem_equiv (x : ℝ) :
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x y, 0 < x → x < y → f y < f x) ∧
  (set.range f = set.univ) :=
by
  sorry

end problem_equiv_l766_766602


namespace complex_division_l766_766831

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the problem statement
theorem complex_division : (1 - 2 * i) / (2 + i) = -i := 
  sorry

end complex_division_l766_766831


namespace odds_against_C_l766_766626

variable (P_A P_B P_C : ℚ)
variable (odds_A odds_B odds_C : ℚ)

-- conditions
def prob_A (odds_A : ℚ) : ℚ := odds_A / (odds_A + 1)  -- odds_A here represents 5/2
def prob_B (odds_B : ℚ) : ℚ := odds_B / (odds_B + 1)  -- odds_B here represents 4/5

-- actual values from problem statement
def odds_A_val : ℚ := 5/2
def odds_B_val : ℚ := 4/5

-- probabilities calculated from the odds
def P_A := prob_A odds_A_val
def P_B := prob_B odds_B_val

-- probability of C winning
def P_C : ℚ := 1 - (P_A + P_B)

-- odds against C winning
def odds_C : ℚ := (1 - P_C) / P_C

-- The theorem to prove
theorem odds_against_C : odds_C = 53 / 10 :=
by
  sorry

end odds_against_C_l766_766626


namespace band_member_earnings_l766_766160

-- Define conditions
def n_people : ℕ := 500
def p_ticket : ℚ := 30
def r_earnings : ℚ := 0.7
def n_members : ℕ := 4

-- Definition of total earnings and share per band member
def total_earnings : ℚ := n_people * p_ticket
def band_share : ℚ := total_earnings * r_earnings
def amount_per_member : ℚ := band_share / n_members

-- Statement to be proved
theorem band_member_earnings : amount_per_member = 2625 := 
by
  -- Proof goes here
  sorry

end band_member_earnings_l766_766160


namespace minimum_passing_rate_l766_766250

-- Define the conditions as hypotheses
variable (total_students : ℕ)
variable (correct_q1 : ℕ)
variable (correct_q2 : ℕ)
variable (correct_q3 : ℕ)
variable (correct_q4 : ℕ)
variable (correct_q5 : ℕ)
variable (pass_threshold : ℕ)

-- Assume all percentages are converted to actual student counts based on total_students
axiom students_answered_q1_correctly : correct_q1 = total_students * 81 / 100
axiom students_answered_q2_correctly : correct_q2 = total_students * 91 / 100
axiom students_answered_q3_correctly : correct_q3 = total_students * 85 / 100
axiom students_answered_q4_correctly : correct_q4 = total_students * 79 / 100
axiom students_answered_q5_correctly : correct_q5 = total_students * 74 / 100
axiom passing_criteria : pass_threshold = 3

-- Define the main theorem statement to be proven
theorem minimum_passing_rate (total_students : ℕ) :
  (total_students - (total_students * 19 / 100 + total_students * 9 / 100 + 
  total_students * 15 / 100 + total_students * 21 / 100 + 
  total_students * 26 / 100) / pass_threshold) / total_students * 100 ≥ 70 :=
  by sorry

end minimum_passing_rate_l766_766250


namespace books_added_is_10_l766_766018

-- Define initial number of books on the shelf
def initial_books : ℕ := 38

-- Define the final number of books on the shelf
def final_books : ℕ := 48

-- Define the number of books that Marta added
def books_added : ℕ := final_books - initial_books

-- Theorem stating that Marta added 10 books
theorem books_added_is_10 : books_added = 10 :=
by
  sorry

end books_added_is_10_l766_766018


namespace largest_equal_containers_l766_766110

theorem largest_equal_containers :
  ∀ containers : ℕ, 150 containers →
  ∀ oranges_per_container : ℕ → ℕ, (∀ c, 130 ≤ oranges_per_container c ∧ oranges_per_container c ≤ 160) → 
  ∃ n, n = 5 ∧ (∑ c in finset.range 150, oranges_per_container c) / n ≥ 1 := 
sorry

end largest_equal_containers_l766_766110


namespace x_squared_minus_y_squared_l766_766177

theorem x_squared_minus_y_squared :
  ∀ (x y : ℝ), (x + y = 4 ∧ x - y = 6) → (x^2 - y^2 = 24) := by
  intro x y
  intro h
  cases h with h1 h2
  sorry

end x_squared_minus_y_squared_l766_766177


namespace arithmetic_sequence_15th_term_l766_766400

theorem arithmetic_sequence_15th_term :
  ∀ (a d n : ℕ), a = 3 → d = 13 - a → n = 15 → 
  a + (n - 1) * d = 143 :=
by
  intros a d n ha hd hn
  rw [ha, hd, hn]
  sorry

end arithmetic_sequence_15th_term_l766_766400


namespace numberOfBoysInClass_l766_766117

-- Define the problem condition: students sit in a circle and boy at 5th position is opposite to boy at 20th position
def studentsInCircle (n : ℕ) : Prop :=
  (n > 5) ∧ (n > 20) ∧ ((20 - 5) * 2 + 2 = n)

-- The main theorem: Given the conditions, prove the total number of boys equals 32
theorem numberOfBoysInClass : ∀ n : ℕ, studentsInCircle n → n = 32 :=
by
  intros n hn
  sorry

end numberOfBoysInClass_l766_766117


namespace simplify_sqrt_expression_l766_766334

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l766_766334


namespace randy_wants_to_become_expert_by_age_l766_766316

noncomputable def piano_expert_age (current_age : ℕ) 
                                   (weekly_practice_hours : ℕ) 
                                   (weeks_per_year : ℕ) 
                                   (total_required_hours : ℕ) : ℕ :=
  let yearly_practice_hours := weekly_practice_hours * weeks_per_year
  let required_years := total_required_hours / yearly_practice_hours
  current_age + required_years

theorem randy_wants_to_become_expert_by_age (h_current_age : ℕ)
                                            (h_weekly_practice_hours : ℕ)
                                            (h_weeks_per_year : ℕ)
                                            (h_total_required_hours : ℕ) :
  piano_expert_age h_current_age h_weekly_practice_hours h_weeks_per_year h_total_required_hours = 20 :=
by
  -- Given conditions specific to Randy
  let current_age := 12
  let daily_hours := 5
  let days_per_week := 5
  let weeks_per_year := 50
  let total_required_hours := 10000
  let weekly_practice_hours := daily_hours * days_per_week

  -- Calculate the required age
  have h := piano_expert_age current_age weekly_practice_hours weeks_per_year total_required_hours
  -- Prove that the resulting age is 20
  show h = 20
  sorry

end randy_wants_to_become_expert_by_age_l766_766316


namespace simplify_radical_expression_l766_766358

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l766_766358


namespace percentage_first_correctly_l766_766092

-- Define the known conditions as constants
def percentage_answered_second_correctly : ℝ := 70
def percentage_answered_neither : ℝ := 20
def percentage_answered_both_correctly : ℝ := 65

-- State the theorem using the given constants to prove the percentage of students who answered the first question correctly
theorem percentage_first_correctly :
  let percentage_total := 100
  let percentage_answered_one_or_both := percentage_total - percentage_answered_neither
  let percentage_first := percentage_answered_one_or_both - (percentage_answered_second_correctly - percentage_answered_both_correctly) 
  in percentage_first = 75 := by
  sorry

end percentage_first_correctly_l766_766092


namespace conic_section_is_ellipse_l766_766539

noncomputable def sqrt_distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def is_ellipse (x y : ℝ) : Prop :=
  sqrt_distance x y 0 2 + sqrt_distance x y 6 (-4) = 14

theorem conic_section_is_ellipse : 
  ∀ (x y : ℝ), is_ellipse x y → sqrt_distance 0 2 6 (-4) < 14 :=
by
  intros x y h
  -- the distance between (0, 2) and (6, -4)
  have d := sqrt_distance 0 2 6 (-4)
  -- proof that d = 6 * real.sqrt 2
  have d_proof : d = 6 * real.sqrt 2 := sorry
  rw [d_proof]
  -- 6 * sqrt 2 < 14
  norm_num
  sorry

end conic_section_is_ellipse_l766_766539


namespace basketball_classes_l766_766242

theorem basketball_classes (x : ℕ) : (x * (x - 1)) / 2 = 10 :=
sorry

end basketball_classes_l766_766242


namespace arithmetic_geometric_progression_l766_766023

theorem arithmetic_geometric_progression (a d : ℝ)
    (h1 : 2 * (a - d) * a * (a + d + 7) = 1000)
    (h2 : a^2 = 2 * (a - d) * (a + d + 7)) :
    d = 8 ∨ d = -8 := 
    sorry

end arithmetic_geometric_progression_l766_766023


namespace distinct_values_of_S_l766_766641

-- Definitions for conditions
def S (n : ℤ) : ℂ := complex.I ^ n + complex.I ^ (-n)

-- Statement of the theorem to be proved
theorem distinct_values_of_S : (finset.image S (finset.range 4)).card = 3 :=
sorry

end distinct_values_of_S_l766_766641


namespace circumscribed_equilateral_exists_inscribed_equilateral_exists_l766_766823

-- Definitions of points and triangle
structure Point := (x : ℝ) (y : ℝ)
structure Triangle := (A B C : Point)

-- Conditions: triangle ABC
variables (A B C : Point)
def triangleABC : Triangle := ⟨A, B, C⟩

-- Theorem Statement for circumscribed equilateral triangle
theorem circumscribed_equilateral_exists (ABC : Triangle) :
  ∃ (DEF : Triangle) (M : Point),
  is_equilateral DEF ∧
  perp_intersects_at_single_point ABC DEF M := 
sorry

-- Theorem Statement for inscribed equilateral triangle
theorem inscribed_equilateral_exists (ABC : Triangle) :
  ∃ (DEF : Triangle) (M : Point),
  is_equilateral DEF ∧
  perp_drops_intersect_at_single_point ABC DEF M := 
sorry

-- Auxiliary Definitions required for the theorems

def is_equilateral (T : Triangle) : Prop :=
  distance T.A T.B = distance T.B T.C ∧ 
  distance T.B T.C = distance T.C T.A

def perp_intersects_at_single_point 
  (ABC : Triangle) 
  (DEF : Triangle) 
  (M : Point) : Prop := 
  properties_based_on_conditions -- to be defined

def perp_drops_intersect_at_single_point 
  (ABC : Triangle) 
  (DEF : Triangle) 
  (M : Point) : Prop := 
  properties_based_on_conditions -- to be defined

-- This is a placeholder for the actual properties and conditions definitions
def properties_based_on_conditions : Prop := 
  sorry

end circumscribed_equilateral_exists_inscribed_equilateral_exists_l766_766823


namespace vegetable_plot_area_l766_766784

variable (V W : ℝ)

theorem vegetable_plot_area (h1 : (1/2) * V + (1/3) * W = 13) (h2 : (1/2) * W + (1/3) * V = 12) : V = 18 :=
by
  sorry

end vegetable_plot_area_l766_766784


namespace duration_of_period_l766_766868

/-- The duration of the period at which B gains Rs. 1125 by lending 
Rs. 25000 at rate of 11.5% per annum and borrowing the same 
amount at 10% per annum -/
theorem duration_of_period (principal : ℝ) (rate_borrow : ℝ) (rate_lend : ℝ) (gain : ℝ) : 
  ∃ (t : ℝ), principal = 25000 ∧ rate_borrow = 0.10 ∧ rate_lend = 0.115 ∧ gain = 1125 → 
  t = 3 :=
by
  sorry

end duration_of_period_l766_766868


namespace lele_dongdong_meet_probability_l766_766259

-- Define the conditions: distances and speeds
def segment_length : ℕ := 500
def n : ℕ := sorry
def d : ℕ := segment_length * n
def lele_speed : ℕ := 18
def dongdong_speed : ℕ := 24

-- Define times to traverse distance d
def t_L : ℚ := d / lele_speed
def t_D : ℚ := d / dongdong_speed

-- Define the time t when they meet
def t : ℚ := d / (lele_speed + dongdong_speed)

-- Define the maximum of t_L and t_D
def max_t_L_t_D : ℚ := max t_L t_D

-- Define the probability they meet on their way
def P_meet : ℚ := t / max_t_L_t_D

-- The theorem to prove the probability of meeting is 97/245
theorem lele_dongdong_meet_probability : P_meet = 97 / 245 :=
sorry

end lele_dongdong_meet_probability_l766_766259


namespace initial_students_count_l766_766729

variable (n T : ℕ)
variables (initial_average remaining_average dropped_score : ℚ)
variables (initial_students remaining_students : ℕ)

theorem initial_students_count :
  initial_average = 62.5 →
  remaining_average = 63 →
  dropped_score = 55 →
  T = initial_average * n →
  T - dropped_score = remaining_average * (n - 1) →
  n = 16 :=
by
  intros h_avg_initial h_avg_remaining h_dropped_score h_total h_total_remaining
  sorry

end initial_students_count_l766_766729


namespace problem_a_problem_b_l766_766842

open Probability

noncomputable def sequenceShortenedByExactlyOneDigitProbability : ℝ :=
  1.564 * 10^(-90)

theorem problem_a (n : ℕ) (p : ℝ) (k : ℕ) (initialLength : ℕ)
  (h_n : n = 2014) (h_p : p = 0.1) (h_k : k = 1) (h_initialLength : initialLength = 2015) :
  (binomialₓ n k p * (1 - p)^(n - k)) = sequenceShortenedByExactlyOneDigitProbability := 
sorry

noncomputable def expectedLengthNewSequence : ℝ :=
  1813.6

theorem problem_b (n : ℕ) (p : ℝ) (initialLength : ℕ) 
  (h_n : n = 2014) (h_p : p = 0.1) (h_initialLength : initialLength = 2015) :
  (initialLength - n * p) = expectedLengthNewSequence :=
sorry

end problem_a_problem_b_l766_766842


namespace sqrt_simplification_l766_766366

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l766_766366


namespace fourth_permutation_is_593_l766_766134

def digits : List Nat := [3, 5, 9]

noncomputable def permutations_list := digits.permutations.map 
  (fun l => l.foldl (fun acc d => acc * 10 + d) 0)

noncomputable def sorted_permutations := (permutations_list.data).erase_dup.qsort Nat.lt

theorem fourth_permutation_is_593 : 
  List.get sorted_permutations 3 = 593 := by
  sorry

end fourth_permutation_is_593_l766_766134


namespace membership_change_l766_766077

theorem membership_change (N : ℕ) : 
  let fall_members := N * (1 + 0.05)
  let spring_members := fall_members * (1 - 0.19)
  let percentage_change := (fall_members - spring_members) / fall_members * 100
  percentage_change = 19.05 :=
by
  sorry

end membership_change_l766_766077


namespace repeating_decimal_sum_l766_766551

theorem repeating_decimal_sum :
  let x : ℚ := 0.2 -- 0.\overline{2}
  let y : ℚ := 0.02 -- 0.\overline{02}
  x + y = 80 / 333 :=
by {
  -- We use library functions to convert repeating decimals to fractions
  let x : ℚ := 2 / 9
  let y : ℚ := 2 / 99
  -- Show the expected sum is the fraction in lowest terms:
  x + y = (80 / 333)
  sorry
}

end repeating_decimal_sum_l766_766551


namespace arithmetic_sequence_sum_l766_766014

variable {α : Type*} [linear_ordered_field α]

def geometric_mean (a b c : α) : Prop :=
  c = (a * b) ^ (1 / 2)

theorem arithmetic_sequence_sum (a_1 d S_8 : α) (h : d ≠ 0) (h_geometric_mean : geometric_mean (a_3 d a_1) (a_7 d a_1) (a_4 d a_1)) (h_sum_8 : S_8 = 32) :
  let a_3 := a_1 + 2 * d
  let a_4 := a_1 + 3 * d
  let a_7 := a_1 + 6 * d
  let S_n (n : ℕ) := n * a_1 + (n * (n - 1) / 2) * d
  S_n 10 = 60 :=
by
  sorry

end arithmetic_sequence_sum_l766_766014


namespace multiple_remainder_l766_766445

-- Let x be a positive integer such that x ≡ 5 (mod 9)
variable (x : ℕ)
variable (hx : x % 9 = 5)

-- We want to prove that 7x ≡ 8 (mod 9)
theorem multiple_remainder (k : ℕ) (hk : k = 7) : (7 * x) % 9 = 8 :=
by
  -- Hypothesize values based on the problem statement
  have h : x ≡ 5 [MOD 9] := hx
  rw [← Nat.mod_eq_of_lt hk]
  have h2 : (7 * x) % 9 = (7 * 5) % 9 := by sorry
  rw [h2]
  exact sorry

end multiple_remainder_l766_766445


namespace distance_between_stations_l766_766083

-- Define the speeds of the trains
variables (v1 v2 : ℝ)

-- Define the time period
def t : ℝ := 3

-- Define the condition that they are 70 km apart both 3 hours and 6 hours after starting
def distance_apart (t : ℝ) (v1 v2 : ℝ) : ℝ := 2 * t * v1 + 2 * t * v2

-- Define the distance condition
def condition_distance_70_km := (distance_apart t v1 v2 = 70)

-- Prove the distance between the stations is 70 km
theorem distance_between_stations (v1 v2 : ℝ) (h : condition_distance_70_km) : 
  let D := t * v1 + t * v2 in D = 70 := by
    sorry

end distance_between_stations_l766_766083


namespace evaluate_expression_l766_766802

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end evaluate_expression_l766_766802


namespace payback_period_correct_l766_766624

noncomputable def payback_period (cost_sys_unit : ℤ) (cost_gpu : ℤ) (num_gpus : ℤ)
  (power_sys_unit : ℕ) (power_gpu : ℕ) (mining_speed : ℚ)
  (earnings_per_gpu_per_day : ℚ) (eth_to_rub : ℚ) (electricity_cost_per_kwh : ℚ) : ℚ :=
let total_cost := cost_sys_unit + num_gpus * cost_gpu in
let total_daily_earnings := num_gpus * earnings_per_gpu_per_day * eth_to_rub in
let total_power_w := power_sys_unit + num_gpus * power_gpu in
let total_power_kw := (total_power_w : ℚ) / 1000 in
let daily_energy_consumption_kwh := total_power_kw * 24 in
let daily_electricity_cost := daily_energy_consumption_kwh * electricity_cost_per_kwh in
let daily_profit := total_daily_earnings - daily_electricity_cost in
total_cost / daily_profit

theorem payback_period_correct :
  payback_period 9499 31431 2 120 125 32 0.00877 27790.37 5.38 ≈ 165 := sorry

end payback_period_correct_l766_766624


namespace angle_C_in_triangle_l766_766615

theorem angle_C_in_triangle (A B C : Type) [triangle ABC] (h_c : ℝ)
  (AB : ℝ) (angleA : ℝ) (H1 : 2 * h_c = AB) (H2 : angleA = 75) : 
  let C_angle := 180 - angleA - 30 in
  C_angle = 75 :=
by
  sorry

end angle_C_in_triangle_l766_766615


namespace find_divisor_l766_766827

-- Define the initial conditions
def dividend := 23
def quotient := 5
def remainder := 3

-- Define the divisor and the equation corresponding to the division algorithm
variable (d : ℕ)
def div_eqn := dividend = (d * quotient) + remainder

-- State the theorem that needs to be proven
theorem find_divisor (h : div_eqn) : d = 4 :=
by
  sorry

end find_divisor_l766_766827


namespace marion_score_is_correct_l766_766755

-- Definition of the problem conditions
def exam_total_items := 40
def ella_incorrect_answers := 4

-- Calculate Ella's score
def ella_score := exam_total_items - ella_incorrect_answers

-- Calculate half of Ella's score
def half_ella_score := ella_score / 2

-- Marion's score is 6 more than half of Ella's score
def marion_score := half_ella_score + 6

-- The theorem we need to prove
theorem marion_score_is_correct : marion_score = 24 := by
  sorry

end marion_score_is_correct_l766_766755


namespace max_n_factoring_polynomial_l766_766557

theorem max_n_factoring_polynomial :
  ∃ n A B : ℤ, (3 * n + A = 217) ∧ (A * B = 72) ∧ (3 * B + A = n) :=
sorry

end max_n_factoring_polynomial_l766_766557


namespace simplify_expression_l766_766329

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l766_766329


namespace evaluate_expression_l766_766814

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 :=
by sorry

end evaluate_expression_l766_766814


namespace longest_side_of_triangle_l766_766169

theorem longest_side_of_triangle (y : ℝ) 
  (side1 : ℝ := 8) (side2 : ℝ := y + 5) (side3 : ℝ := 3 * y + 2)
  (h_perimeter : side1 + side2 + side3 = 47) :
  max side1 (max side2 side3) = 26 :=
sorry

end longest_side_of_triangle_l766_766169


namespace repaint_houses_possible_l766_766256

theorem repaint_houses_possible (families : Type) [fintype families] :
  ∃ (f : families → {c // c = red ∨ c = blue ∨ c = green}),
  ∀ (σ : families → families), bijective σ → ∀ (x : families), f (σ x) ≠ f x :=
  sorry

end repaint_houses_possible_l766_766256


namespace min_value_f_max_value_y_l766_766941

-- Define the function f and the conditions
def f (x : ℝ) : ℝ := 4 / x + x
def positive_domain (x : ℝ) : Prop := x > 0

-- State that the minimum value of f(x) is 4 for x > 0
theorem min_value_f : ∀ x, positive_domain x → f x ≥ 4 := by
  intros x hx
  sorry

-- Define the function y and the conditions
def y (x : ℝ) : ℝ := x * (1 - 3 * x)
def domain_y (x : ℝ) : Prop := 0 < x ∧ x < 1 / 3

-- State that the maximum value of y(x) is 1/12 for 0 < x < 1/3
theorem max_value_y : ∀ x, domain_y x → y x ≤ 1 / 12 := by
  intros x hx
  sorry

end min_value_f_max_value_y_l766_766941


namespace passing_marks_l766_766474

variable (T P : ℝ)

theorem passing_marks :
  (0.35 * T = P - 40) →
  (0.60 * T = P + 25) →
  P = 131 :=
by
  intro h1 h2
  -- Proof steps should follow here.
  sorry

end passing_marks_l766_766474


namespace periodic_decimal_sum_l766_766547

theorem periodic_decimal_sum : (0.2).periodic + (0.02).periodic = 8 / 33 :=
by
  -- Defining the periodic decimal representations in terms of their fractional forms:
  let a : ℚ := 2 / 9 -- equivalent to 0.\overline{2}
  let b : ℚ := 2 / 99 -- equivalent to 0.\overline{02}
  -- Asserting the sum of these fractions:
  have h : a + b = 24 / 99, by sorry
  -- Reducing the fraction to its lowest terms:
  have h' : 24 / 99 = 8 / 33, by sorry
  -- Concluding the equality:
  exact Eq.trans h h'

end periodic_decimal_sum_l766_766547


namespace decimal_digits_count_l766_766926

theorem decimal_digits_count (a b c : ℕ) (h1 : a = 5^8) (h2 : b = 10^6) (h3 : c = 125) :
  let n := a / (b * c) in
  ∃ d : ℕ, n = d / 10 ∧ d % 10 ≠ 0 ∧ d / 10 = 32 :=
sorry

end decimal_digits_count_l766_766926


namespace largest_perfect_square_factor_of_3402_l766_766056

theorem largest_perfect_square_factor_of_3402 :
  ∃ (n : ℕ), n^2 ∣ 3402 ∧ (∀ m : ℕ, m^2 ∣ 3402 → m^2 ≤ n^2) :=
begin
  use 3, -- n is 3
  split,
  {
    norm_num,
    rw [mul_comm, nat.dvd_prime_pow (dec_trivial : prime 3)] ; [norm_num, dec_trivial],
  },
  {
    intros m h,
    have h_dvds, from nat.prime_dvd_prime_pow (dec_trivial : prime 3) h,
    cases h_dvds,
    {
      exact h_dvds.symm ▸ nat.le_refl _,
    },
    {
      suffices : 1 ≤ 3, from h_dvds.symm ▸ nat.pow_le_pow_of_le_left this 2,
      norm_num,
    }
  }
end

end largest_perfect_square_factor_of_3402_l766_766056


namespace C_investment_is_20000_l766_766113

-- Definitions of investments and profits
def A_investment : ℕ := 12000
def B_investment : ℕ := 16000
def total_profit : ℕ := 86400
def C_share_of_profit : ℕ := 36000

-- The proof problem statement
theorem C_investment_is_20000 (X : ℕ) (hA : A_investment = 12000) (hB : B_investment = 16000)
  (h_total_profit : total_profit = 86400) (h_C_share_of_profit : C_share_of_profit = 36000) :
  X = 20000 :=
sorry

end C_investment_is_20000_l766_766113


namespace minimum_value_op_dot_fp_l766_766647

theorem minimum_value_op_dot_fp (x y : ℝ) (h_ellipse : x^2 / 2 + y^2 = 1) :
  let OP := (x, y)
  let FP := (x - 1, y)
  let dot_product := x * (x - 1) + y^2
  dot_product ≥ 1 / 2 :=
by
  sorry

end minimum_value_op_dot_fp_l766_766647


namespace selling_price_for_given_profit_selling_price_to_maximize_profit_l766_766476

-- Define the parameters
def cost_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_monthly_sales : ℝ := 500
def sales_decrement_per_unit_increase : ℝ := 10

-- Define the function for monthly sales based on price increment
def monthly_sales (x : ℝ) : ℝ := initial_monthly_sales - sales_decrement_per_unit_increase * x

-- Define the function for selling price based on price increment
def selling_price (x : ℝ) : ℝ := initial_selling_price + x

-- Define the function for monthly profit
def monthly_profit (x : ℝ) : ℝ :=
  let total_revenue := monthly_sales x * selling_price x 
  let total_cost := monthly_sales x * cost_price
  total_revenue - total_cost

-- Problem 1: Prove the selling price when monthly profit is 8750 yuan
theorem selling_price_for_given_profit : 
  ∃ x : ℝ, monthly_profit x = 8750 ∧ (selling_price x = 75 ∨ selling_price x = 65) :=
sorry

-- Problem 2: Prove the selling price that maximizes the monthly profit
theorem selling_price_to_maximize_profit : 
  ∀ x : ℝ, monthly_profit x ≤ monthly_profit 20 ∧ selling_price 20 = 70 :=
sorry

end selling_price_for_given_profit_selling_price_to_maximize_profit_l766_766476


namespace clara_weight_l766_766013

theorem clara_weight (a c : ℝ) (h1 : a + c = 220) (h2 : c - a = c / 3) : c = 88 :=
by
  sorry

end clara_weight_l766_766013


namespace concentrate_candies_l766_766779

theorem concentrate_candies
  (n : ℕ) (h_n : n ≥ 4) (candies : ℕ)
  (candies_per_plate : ℕ → ℕ)
  (h_total : candies_per_plate.sum (finset.range n) ≥ 4) :
  ∃ i : ℕ, candies_per_plate.sum (finset.range n) = candies_per_plate i :=
sorry

end concentrate_candies_l766_766779


namespace positive_iff_sum_and_product_positive_l766_766198

theorem positive_iff_sum_and_product_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) :=
by
  sorry

end positive_iff_sum_and_product_positive_l766_766198


namespace nancy_balloons_l766_766708

variable (MaryBalloons : ℝ) (NancyBalloons : ℝ)

theorem nancy_balloons (h1 : NancyBalloons = 4 * MaryBalloons) (h2 : MaryBalloons = 1.75) : 
  NancyBalloons = 7 := 
by 
  sorry

end nancy_balloons_l766_766708


namespace eggs_distribution_l766_766504

theorem eggs_distribution (A B C crates : ℕ) (hA : A = 58) (hB : B = 76) (hC : C = 27) (hCrates : crates = 18) :
  let total_eggs := A + B + C in
  let remaining_eggs := total_eggs % crates in
  let per_person := remaining_eggs / 3 in
  let leftover := remaining_eggs % 3 in
  (per_person = 5 ∧ leftover = 2) →
  (Abigail_eggs Beatrice_eggs Carson_eggs : ℕ)
  (hAbigail : Abigail_eggs = per_person + 1) (hBeatrice : Beatrice_eggs = per_person + 1) (hCarson : Carson_eggs = per_person)
  : Abigail_eggs = 6 ∧ Beatrice_eggs = 6 ∧ Carson_eggs = 5 :=
by {
  sorry
}

end eggs_distribution_l766_766504


namespace part1_part2_l766_766983

def A (a : ℝ) : set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 1}
def B : set ℝ := {x | 0 ≤ (4 - x) / (x + 2)}

theorem part1 (a : ℝ) (h : a = 2) : (compl (A a)) ∩ B = {x | -2 < x ∧ x < 1} := by
  sorry

theorem part2 (a : ℝ) : (A a ∪ B = B) ↔ (a ∈ Iio (-2) ∪ Ioo (-1) (3 / 2)) := by
  sorry

end part1_part2_l766_766983


namespace paths_E_to_G_through_F_and_H_l766_766227

-- Define positions of E, F, H, and G on the grid.
structure Point where
  x : ℕ
  y : ℕ

def E : Point := { x := 0, y := 0 }
def F : Point := { x := 3, y := 2 }
def H : Point := { x := 5, y := 4 }
def G : Point := { x := 8, y := 4 }

-- Function to calculate number of paths from one point to another given the number of right and down steps
def paths (start goal : Point) : ℕ :=
  let right_steps := goal.x - start.x
  let down_steps := goal.y - start.y
  Nat.choose (right_steps + down_steps) right_steps

theorem paths_E_to_G_through_F_and_H : paths E F * paths F H * paths H G = 60 := by
  sorry

end paths_E_to_G_through_F_and_H_l766_766227


namespace parabola_properties_l766_766606

-- Define the parabola C with its given properties
def parabola (p : ℝ) (p_pos : p > 0) : ℝ × ℝ → Prop := λ Q, Q.snd ^ 2 = 2 * p * Q.fst

-- Given point Q(2,2)
def Q : ℝ × ℝ := (2, 2)

-- Define the line passing through point M(2,0)
def line (m : ℝ) : ℝ → ℝ := λ y, m * y + 2

-- Define slope calculations for lines through the origin
def slope (x y : ℝ) : ℝ := y / x

-- Main theorem statement
theorem parabola_properties :
  (∀ p : ℝ, p > 0 → parabola p Q) →
  (∃ p : ℝ, p > 0 ∧ parabola p Q ∧ (parabola 1 Q) ∧ (Q.snd ^ 2 = 2 * Q.fst)) ∧
  (∀ m : ℝ, ∃ y1 y2 : ℝ, y1 + y2 = 2 * m ∧ y1 * y2 = -4 ∧ (k1 k2 = -1) 
  ∧ (let k1 := slope (line m y1) y1 in
      let k2 := slope (line m y2) y2 in
      k1 * k2 = -1)) :=
by
  sorry

end parabola_properties_l766_766606


namespace band_member_share_l766_766168

def num_people : ℕ := 500
def ticket_price : ℝ := 30
def band_share_percent : ℝ := 0.70
def num_band_members : ℕ := 4

theorem band_member_share : 
  (num_people * ticket_price * band_share_percent) / num_band_members = 2625 := by
  sorry

end band_member_share_l766_766168


namespace triangular_weight_60_l766_766007

def round_weight := ℝ
def triangular_weight := ℝ
def rectangular_weight := 90

variables (c t : ℝ)

-- Conditions
axiom condition1 : c + t = 3 * c
axiom condition2 : 4 * c + t = t + c + rectangular_weight

theorem triangular_weight_60 : t = 60 :=
  sorry

end triangular_weight_60_l766_766007


namespace bus_interval_three_buses_l766_766035

theorem bus_interval_three_buses (T : ℕ) (h : T = 21) : (T * 2) / 3 = 14 :=
by
  sorry

end bus_interval_three_buses_l766_766035


namespace quadrilateral_diagonal_length_l766_766535

-- Definitions based on the conditions
variables (AB BD DC CA DB : ℝ)
variables (θ : ℝ)
variables (cos_theta : ℝ)

-- Conditions
def pythagorean_cos_theta_condition : Prop :=
  cos_theta = (6^2 + 5^2 - 8^2) / (2 * 6 * 5)

def law_of_cosines_condition : Prop :=
  108.2 = 6^2 + 11^2 + 2 * 6 * 11 * cos_theta

-- The main problem statement
theorem quadrilateral_diagonal_length :
  pythagorean_cos_theta_condition ∧ law_of_cosines_condition →
  sqrt 108.2 = sqrt 108.2 := 
sorry

end quadrilateral_diagonal_length_l766_766535


namespace find_k_value_l766_766507

theorem find_k_value (
  (ABC : Triangle)
) (circumcircle : ∀ (P : Point), P ∈ pts ABC ↔ P ∈ circ (pts ABC)) -- triangle inscribed in a circle
  (isosceles_ABC : angle ABC.A ABC.B = angle ABC.A ABC.C)  -- isosceles triangle
  (BAC_eq_sixty: angle ABC.B ABC.A = 60) : ∃ k : ℝ, k = 2 :=
begin
  sorry
end

end find_k_value_l766_766507


namespace complex_number_problem_l766_766974

theorem complex_number_problem (z : ℂ) (h : (2 + complex.i) * z = 1 + 3 * complex.i) :
  |z| = Real.sqrt 2 ∧ (z ^ 2 - 2 * z + 2 = 0) :=
by
  sorry

end complex_number_problem_l766_766974


namespace probability_of_draw_l766_766655

-- Define the probabilities
def P_w : ℝ := 0.40        -- Probability of player A winning
def P_nl : ℝ := 0.90       -- Probability of player A not losing

-- Define the statement to prove
theorem probability_of_draw : ∃ P : ℝ, P_nl = P_w + P ∧ P = 0.50 :=
by
  use 0.50
  split
  . norm_num1
  . refl
  sorry

end probability_of_draw_l766_766655


namespace largest_x_value_l766_766062

theorem largest_x_value (x : ℝ) (h : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
by sorry

end largest_x_value_l766_766062


namespace probability_five_common_correct_l766_766672

-- Define the conditions
def compulsory_subjects : ℕ := 3  -- Chinese, Mathematics, and English
def elective_from_physics_history : ℕ := 1  -- Physics and History
def elective_from_four : ℕ := 4  -- Politics, Geography, Chemistry, Biology

def chosen_subjects_by_xiaoming_xiaofang : ℕ := 2  -- two subjects from the four electives

-- Calculate total combinations
noncomputable def total_combinations : ℕ := Nat.choose 4 2 * Nat.choose 4 2

-- Calculate combinations to have exactly five subjects in common
noncomputable def combinations_five_common : ℕ := Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 2 1

-- Calculate the probability
noncomputable def probability_five_common : ℚ := combinations_five_common / total_combinations

-- The theorem to be proved
theorem probability_five_common_correct : probability_five_common = 2 / 3 := by
  sorry

end probability_five_common_correct_l766_766672


namespace combined_cost_is_correct_l766_766111

-- Definitions based on the conditions
def dryer_cost : ℕ := 150
def washer_cost : ℕ := 3 * dryer_cost
def combined_cost : ℕ := dryer_cost + washer_cost

-- Statement to be proved
theorem combined_cost_is_correct : combined_cost = 600 :=
by
  sorry

end combined_cost_is_correct_l766_766111


namespace correct_event_statement_l766_766432

-- Define a fair six-sided die
def Die := { n // 1 ≤ n ∧ n ≤ 6 }

-- Define the event of rolling two dice
def roll_two_dice : List (Die × Die) :=
  List.product (List.range' 1 6) (List.range' 1 6)

-- Define the conditions of possible events
def event_rolling_two_ones := (1, 1)
def event_sum_six := List.filter (λ (p : Die × Die), p.1 + p.2 = 6) roll_two_dice
def event_rolling_two_sixes := (6, 6)
def event_sum_fourteen := List.filter (λ (p : Die × Die), p.1 + p.2 = 14) roll_two_dice

-- Define what makes an event certain, impossible, or random
def is_impossible_event (event : List (Die × Die)) : Prop := event = []
def is_certain_event (event : List (Die × Die)) : Prop := event = roll_two_dice
def is_random_event (event : List (Die × Die)) : Prop := ¬is_impossible_event event ∧ ¬is_certain_event event

-- Define the statement we need to prove
theorem correct_event_statement :
  (is_impossible_event [event_rolling_two_ones] = false)
  ∧ (is_certain_event event_sum_six = false)
  ∧ (is_random_event [event_rolling_two_sixes] = true)
  ∧ (is_random_event event_sum_fourteen = false) := by
  sorry

end correct_event_statement_l766_766432


namespace pencil_cost_l766_766543

-- Definitions based on the conditions
def total_money : ℝ := 20
def cost_per_pen : ℝ := 2
def pencils_bought : ℝ := 5
def pens_bought : ℝ := 6

-- Theorem stating the question and its equivalently correct answer
theorem pencil_cost :
  let remaining_money := total_money - (cost_per_pen * pens_bought) in
  remaining_money / pencils_bought = 1.60 :=
by
  sorry

end pencil_cost_l766_766543


namespace seq_a_eval_a4_l766_766222

theorem seq_a_eval_a4 (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 2, a n = 2 * a (n - 1) + 1) : a 4 = 15 :=
sorry

end seq_a_eval_a4_l766_766222


namespace jerry_paid_more_than_tom_l766_766424

-- Definitions and conditions

def num_slices : ℕ := 12
def cost_plain_pizza : ℝ := 12
def cost_anchovies : ℝ := 3
def cost_onions : ℝ := 2
def slices_per_third : ℕ := num_slices / 3
def total_cost : ℝ := cost_plain_pizza + cost_anchovies + cost_onions

def is_valid_slices : Prop := num_slices % slices_per_third = 0 -- Ensure slices are evenly distributed

def cost_per_slice : ℝ := total_cost / num_slices

def jerry_slices : ℕ := 2 * slices_per_third + 2
def tom_slices : ℕ := num_slices - jerry_slices

def jerry_cost : ℝ := jerry_slices * cost_per_slice
def tom_cost : ℝ := tom_slices * cost_per_slice

def payment_difference : ℝ := jerry_cost - tom_cost

-- Statement to prove
theorem jerry_paid_more_than_tom
  (valid_slices : is_valid_slices) :
  payment_difference = 11.36 := by
  sorry

end jerry_paid_more_than_tom_l766_766424


namespace sqrt_simplification_l766_766370

noncomputable def simplify_expression (x y z : ℝ) : ℝ := 
  x - y + z
  
theorem sqrt_simplification :
  simplify_expression (Real.sqrt 7) (Real.sqrt (28)) (Real.sqrt 63) = 2 * Real.sqrt 7 := 
by 
  have h1 : Real.sqrt 28 = Real.sqrt (4 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 4) (Real.square_nonneg 7)),
  have h2 : Real.sqrt (4 * 7) = Real.sqrt 4 * Real.sqrt 7, from Real.sqrt_mul (4) (7),
  have h3 : Real.sqrt 63 = Real.sqrt (9 * 7), from Real.sqrt_eq (Real.nonneg_of_mul_nonneg_right (by norm_num : 0 ≤ 9) (Real.square_nonneg 7)),
  have h4 : Real.sqrt (9 * 7) = Real.sqrt 9 * Real.sqrt 7, from Real.sqrt_mul (9) (7),
  rw [h1, h2, h3, h4],
  sorry

end sqrt_simplification_l766_766370


namespace negation_forall_pos_l766_766408

theorem negation_forall_pos (h : ∀ x : ℝ, 0 < x → x^2 + x + 1 > 0) :
  ∃ x : ℝ, 0 < x ∧ x^2 + x + 1 ≤ 0 :=
sorry

end negation_forall_pos_l766_766408


namespace infinite_solutions_for_equation_l766_766315

theorem infinite_solutions_for_equation :
  ∃ (x y z : ℤ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ ∀ (k : ℤ), (x^2 + y^5 = z^3) :=
sorry

end infinite_solutions_for_equation_l766_766315


namespace simplify_sqrt_expression_l766_766339

theorem simplify_sqrt_expression :
  (\sqrt{7} - \sqrt{28} + \sqrt{63} : Real) = 2 * \sqrt{7} :=
by
  sorry

end simplify_sqrt_expression_l766_766339


namespace opposite_of_neg_neg_five_l766_766752

-- Define the conditions in Lean 4
axiom neg_neg (a : ℤ) : -(-a) = a
axiom opp_pos (a : ℤ) : a > 0 → -(a) = -a

-- Formalize the problem statement in Lean 4
theorem opposite_of_neg_neg_five : 
  let x := -(-5) in -x = -5 := 
by 
  intros 
  rw [neg_neg 5]
  exact rfl


end opposite_of_neg_neg_five_l766_766752


namespace axis_of_symmetry_l766_766220

def f (x : ℝ) : ℝ := sin (x + π / 6)

def g (x : ℝ) : ℝ := sin (2 * x - π / 6)

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, g (x + k * π / 2) = g x → x = π / 3 :=
sorry

end axis_of_symmetry_l766_766220


namespace probability_of_exactly_one_shortening_l766_766858

-- Define the conditions and problem
def shorten_sequence_exactly_one_probability (n : ℕ) (p : ℝ) : ℝ :=
  nat.choose n 1 * p^1 * (1 - p)^(n - 1)

-- Given conditions
def problem_8a := shorten_sequence_exactly_one_probability 2014 0.1 ≈ 1.564e-90

-- Lean statement
theorem probability_of_exactly_one_shortening :
  problem_8a := sorry

end probability_of_exactly_one_shortening_l766_766858


namespace probability_top_card_is_king_or_queen_l766_766498

-- Defining the basic entities of the problem
def standard_deck_size := 52
def ranks := 13
def suits := 4
def number_of_kings := 4
def number_of_queens := 4
def number_of_kings_and_queens := number_of_kings + number_of_queens

-- Statement: Calculating the probability that the top card is either a King or a Queen
theorem probability_top_card_is_king_or_queen :
  (number_of_kings_and_queens : ℚ) / standard_deck_size = 2 / 13 := by
  -- Skipping the proof for now
  sorry

end probability_top_card_is_king_or_queen_l766_766498


namespace quarters_in_school_year_l766_766016

variable (students : ℕ) (artworks_per_student_per_quarter : ℕ) (total_artworks : ℕ) (school_years : ℕ)

def number_of_quarters (students : ℕ) (artworks_per_student_per_quarter : ℕ) (total_artworks : ℕ) (school_years : ℕ) : ℕ :=
  (total_artworks / (students * artworks_per_student_per_quarter * school_years))

theorem quarters_in_school_year :
  number_of_quarters 15 2 240 2 = 4 :=
by sorry

end quarters_in_school_year_l766_766016


namespace percentage_error_divide_instead_of_multiply_l766_766455

theorem percentage_error_divide_instead_of_multiply (x : ℝ) :
  let correct_result := 5 * x,
      incorrect_result := x / 5,
      error := correct_result - incorrect_result,
      percentage_error := (error / correct_result) * 100
  in percentage_error = 96 := 
by
  intros,
  sorry

end percentage_error_divide_instead_of_multiply_l766_766455


namespace complex_number_problem_l766_766973

theorem complex_number_problem (z : ℂ) (h : (2 + complex.i) * z = 1 + 3 * complex.i) :
  |z| = Real.sqrt 2 ∧ (z ^ 2 - 2 * z + 2 = 0) :=
by
  sorry

end complex_number_problem_l766_766973


namespace river_width_l766_766103

theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) 
  (h1 : depth = 2) 
  (h2 : flow_rate = 4000 / 60)  -- Flow rate in meters per minute
  (h3 : volume_per_minute = 6000) :
  volume_per_minute / (flow_rate * depth) = 45 :=
by
  sorry

end river_width_l766_766103


namespace right_triangle_possible_third_side_l766_766246

theorem right_triangle_possible_third_side (a b : ℕ) (h : a = 5 ∧ b = 12 ∨ a = 12 ∧ b = 5) :
  ∃ c : ℝ, (c = sqrt (a^2 + b^2) ∨ c = sqrt (b^2 - a^2)) :=
by {
  sorry
}

end right_triangle_possible_third_side_l766_766246


namespace simplify_expression_l766_766323

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l766_766323


namespace trapezium_area_l766_766144

theorem trapezium_area :
  ∀ (a b h : ℝ), a = 20 → b = 18 → h = 5 → (1 / 2 * (a + b) * h = 95) :=
by 
  intros a b h ha hb hh
  rw [ha, hb, hh]
  norm_num
  sorry

end trapezium_area_l766_766144


namespace total_pages_read_l766_766303

-- Define the average pages read by Lucas for the first four days.
def day1_4_avg : ℕ := 42

-- Define the average pages read by Lucas for the next two days.
def day5_6_avg : ℕ := 50

-- Define the pages read on the last day.
def day7 : ℕ := 30

-- Define the total number of days for which measurement is provided.
def total_days : ℕ := 7

-- Prove that the total number of pages Lucas read is 298.
theorem total_pages_read : 
  4 * day1_4_avg + 2 * day5_6_avg + day7 = 298 := 
by 
  sorry

end total_pages_read_l766_766303


namespace marion_score_is_correct_l766_766756

-- Definition of the problem conditions
def exam_total_items := 40
def ella_incorrect_answers := 4

-- Calculate Ella's score
def ella_score := exam_total_items - ella_incorrect_answers

-- Calculate half of Ella's score
def half_ella_score := ella_score / 2

-- Marion's score is 6 more than half of Ella's score
def marion_score := half_ella_score + 6

-- The theorem we need to prove
theorem marion_score_is_correct : marion_score = 24 := by
  sorry

end marion_score_is_correct_l766_766756


namespace simplify_expression_l766_766324

theorem simplify_expression :
  (\sqrt(7) - \sqrt(28) + \sqrt(63) = 2 * \sqrt(7)) :=
by
  sorry

end simplify_expression_l766_766324


namespace eulers_formula_l766_766703

open Set

theorem eulers_formula (V E F : ℕ) (h_connectivity : ConnectedGraph G)
  (h_planarity : PlanarGraph G)
  (h_vertices : card (vertices G) = V)
  (h_edges : card (edges G) = E)
  (h_faces : card (faces G) = F) :
  V - E + F = 2 :=
sorry

end eulers_formula_l766_766703


namespace evaluate_expression_l766_766544

theorem evaluate_expression (x : ℝ) (h1 : x^5 + 1 ≠ 0) (h2 : x^5 - 1 ≠ 0) :
  ( ((x^2 - 2*x + 2)^2 * (x^3 - x^2 + 1)^2 / (x^5 + 1)^2)^2 *
    ((x^2 + 2*x + 2)^2 * (x^3 + x^2 + 1)^2 / (x^5 - 1)^2)^2 )
  = 1 := 
by 
  sorry

end evaluate_expression_l766_766544


namespace probability_shortening_exactly_one_digit_l766_766847
open Real

theorem probability_shortening_exactly_one_digit :
  ∀ (sequence : List ℕ),
    (sequence.length = 2015) ∧
    (∀ (x ∈ sequence), x = 0 ∨ x = 9) →
      (P (λ seq, (length (shrink seq) = 2014)) sequence = 1.564e-90) := sorry

end probability_shortening_exactly_one_digit_l766_766847


namespace parabola_sinusoid_do_not_intersect_l766_766722

-- Defining the parabola equation
def y1 (x : ℝ) : ℝ := x^2 - x + 5.35

-- Defining the sinusoidal function equation
def y2 (x : ℝ) : ℝ := 2 * Real.sin x + 3

-- Main theorem stating that the parabola and the sinusoidal function do not intersect
theorem parabola_sinusoid_do_not_intersect : ∀ x : ℝ, (y2 x ≤ 5) → (y1 x ≥ 5.1) → false :=
sorry

end parabola_sinusoid_do_not_intersect_l766_766722


namespace math_problem_l766_766982

theorem math_problem (p : Prop) (q : Prop) (hp : p = (2 + 2 = 5)) (hq : q = (3 > 2)) :
  ((p ∨ q) ∧ ¬q) = false :=
by {
  have h₁ : p = false,
  { rw hp, exact nat.bit0_le_one, }, -- prove 2 + 2 ≠ 5
  have h₂ : q = true,
  { rw hq, exact nat.one_sub_le_one, }, -- prove 3 > 2
  have h₃ : p ∨ q,
  { rw [h₁, h₂], exact or.inr true.intro, }, -- prove p ∨ q
  have h₄ : ¬q = false,
  { rw hq, exact not_false, },
  exact not_and_not_of_not_eq_inr h₃ h₄,
  sorry
}

end math_problem_l766_766982


namespace solvable_range_of_k_l766_766970
open Real

theorem solvable_range_of_k {a : ℝ} (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  ∀ k : ℝ, (∃ x : ℝ, log a (x - a * k) = (log a (x^2 - a^2))^2) ↔ k ∈ (Iio (-1) ∪ Ioo 0 1) :=
by
  sorry

end solvable_range_of_k_l766_766970


namespace second_candidate_votes_l766_766419

-- Definitions and conditions
def total_votes (W : ℝ) : ℝ := W + 3000 + 20000

-- Given condition
@[simp] def winning_percentage := (71.42857142857143 / 100 : ℝ)
def winning_votes (V : ℝ) := winning_percentage * V

-- The main theorem to prove: the second candidate received 20000 votes
theorem second_candidate_votes :
  ∃ (V : ℝ), (V - winning_votes V = 3000 + 20000) → (20000 = 20000) :=
by
  exists (80500.0)
  intro h
  trivial

end second_candidate_votes_l766_766419


namespace solve_equation1_solve_equation2_l766_766389

open Real

theorem solve_equation1 (x : ℝ) : (x - 2)^2 = 9 → (x = 5 ∨ x = -1) :=
by
  intro h
  sorry -- Proof would go here

theorem solve_equation2 (x : ℝ) : (2 * x^2 - 3 * x - 1 = 0) → (x = (3 + sqrt 17) / 4 ∨ x = (3 - sqrt 17) / 4) :=
by
  intro h
  sorry -- Proof would go here

end solve_equation1_solve_equation2_l766_766389


namespace geometric_seq_increasing_condition_not_sufficient_nor_necessary_l766_766261

-- Definitions based on conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = q * a n
def monotonically_increasing (a : ℕ → ℝ) := ∀ n : ℕ, a n ≤ a (n + 1)
def common_ratio_gt_one (q : ℝ) := q > 1

-- Proof statement of the problem
theorem geometric_seq_increasing_condition_not_sufficient_nor_necessary 
    (a : ℕ → ℝ) (q : ℝ) 
    (h1 : geometric_sequence a q) : 
    ¬(common_ratio_gt_one q ↔ monotonically_increasing a) :=
sorry

end geometric_seq_increasing_condition_not_sufficient_nor_necessary_l766_766261


namespace batsman_average_after_12th_l766_766541

theorem batsman_average_after_12th (runs_12th : ℕ) (average_increase : ℕ) (initial_innings : ℕ)
   (initial_average : ℝ) (runs_before_12th : ℕ → ℕ) 
   (h1 : runs_12th = 48)
   (h2 : average_increase = 2)
   (h3 : initial_innings = 11)
   (h4 : initial_average = 24)
   (h5 : ∀ i, i < initial_innings → runs_before_12th i ≥ 20)
   (h6 : ∃ i, runs_before_12th i = 25 ∧ runs_before_12th (i + 1) = 25) :
   (11 * initial_average + runs_12th) / 12 = 26 :=
by
  sorry

end batsman_average_after_12th_l766_766541


namespace box_box_13_eq_24_l766_766171

def box (n : ℕ) : ℕ := (Nat.divisors n).sum

theorem box_box_13_eq_24 : box (box 13) = 24 := 
by
  sorry

end box_box_13_eq_24_l766_766171


namespace find_x_collinear_l766_766616

-- Given vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (1, -3)
def vec_c (x : ℝ) : ℝ × ℝ := (-2, x)

-- Definition of vectors being collinear
def collinear (v₁ v₂ : ℝ × ℝ) : Prop :=
∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

-- Question: What is the value of x such that vec_a + vec_b is collinear with vec_c(x)?
theorem find_x_collinear : ∃ x : ℝ, collinear (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) (vec_c x) ∧ x = 1 :=
by
  sorry

end find_x_collinear_l766_766616


namespace region_area_l766_766522

theorem region_area :
  let region (x y : ℝ) := abs (x - 2 * y^2) + x + 2 * y^2 ≤ 8 - 4 * y
  ∑ (x y : ℝ) in region, 1 = 30 :=
sorry

end region_area_l766_766522


namespace tennis_players_l766_766247

theorem tennis_players (total_members badminton_players neither_players both_players : ℕ)
  (h1 : total_members = 80)
  (h2 : badminton_players = 48)
  (h3 : neither_players = 7)
  (h4 : both_players = 21) :
  total_members - neither_players = badminton_players - both_players + (total_members - neither_players - badminton_players + both_players) + both_players →
  ((total_members - neither_players) - (badminton_players - both_players) - both_players) + both_players = 46 :=
by
  intros h
  sorry

end tennis_players_l766_766247


namespace sum_of_poss_vals_eq_minus_one_l766_766138

theorem sum_of_poss_vals_eq_minus_one :
  (∀ x : ℝ, median {3, 5, 10, 20, x} = (3 + 5 + 10 + 20 + x) / 5 → 
  x = -13 ∨ x = 12) →
  (∀ x : ℝ, median {3, 5, 10, 20, x} = (3 + 5 + 10 + 20 + x) / 5 → 
  ∑ x in {-13, 12}, x = -1) :=
by
  sorry

end sum_of_poss_vals_eq_minus_one_l766_766138


namespace Mencius_misunderstanding_correct_option_l766_766463

/--
Mencius's philosophy: Mencius advocated that human nature is inherently good.

Options:
A: "Fish is what I desire" comes from the Confucian classic *Mencius*, which records Mencius's words, political views, philosophical stances, and personal cultivation. "Fish is what I desire" discusses how to treat life and death, righteousness, and profit, and expounds Mencius's view that human nature is evil.
B: The article begins with a metaphorical argument, using fish and bear's paws as metaphors to discuss the choices in life should prioritize "righteousness," and when necessary, "forsake life for righteousness," while also criticizing those who forget righteousness for profit.
C: The "this sentiment" in "It is not only the wise who feel this way" refers to the compassionate heart, the heart of shame and dislike, the heart of modesty and yielding, and the heart of distinguishing right from wrong, among other good hearts.
D: When facing the choice between life and death, those who "forsake life for righteousness" demonstrate that "there are things they desire more than life, hence they do not seek to live at any cost" and "there are things they detest more than death, hence there are dangers they will not avoid."

The correct option identifying the misunderstanding regarding Mencius's stance is option A.
-/
theorem Mencius_misunderstanding_correct_option
  (mencius_human_nature : ∀ (p : String), p = "Human nature is inherently good")
  (optionA_misunderstanding : String = "Fish is what I desire discusses ... expounds Mencius's view that human nature is evil")
  (answer_is_A : String = "A") :
  (optionA_misunderstanding = "A") := sorry

end Mencius_misunderstanding_correct_option_l766_766463


namespace quadratic_sum_solutions_l766_766002

noncomputable def sum_of_solutions (a b c : ℝ) : ℝ := 
  (-b/a)

theorem quadratic_sum_solutions : 
  ∀ x : ℝ, sum_of_solutions 1 (-9) (-45) = 9 := 
by
  intro x
  sorry

end quadratic_sum_solutions_l766_766002


namespace vector_condition_l766_766226

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

variables (x : ℝ)
def vector_a : ℝ × ℝ := (1, x)
def vector_b : ℝ × ℝ := (x - 2, x)

theorem vector_condition :
  vector_magnitude (vector_a x + vector_b x) = vector_magnitude (vector_a x - vector_b x) ↔
  x = -2 ∨ x = 1 :=
by sorry

end vector_condition_l766_766226


namespace arithmetic_mean_15_23_37_45_l766_766523

def arithmetic_mean (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

theorem arithmetic_mean_15_23_37_45 :
  arithmetic_mean 15 23 37 45 = 30 :=
by {
  sorry
}

end arithmetic_mean_15_23_37_45_l766_766523


namespace minimum_sum_of_labels_l766_766529

theorem minimum_sum_of_labels :
  ∀ (r : ℕ → ℕ), 
  (∀ i, 1 ≤ r i ∧ r i ≤ 10) →
  (∀ (i1 i2 : ℕ), i1 ≠ i2 → r i1 ≠ r i2) →
  ∑ i in Finset.range 10, (1 / (2 * r i + 3 * (i + 1))) ≥ 4 / 11 :=
by
  intros r hr1 hrp
  sorry

end minimum_sum_of_labels_l766_766529


namespace smallest_n_l766_766587

theorem smallest_n (n : ℕ) (h1 : n > 1) (h2 : 2016 ∣ (3 * n^3 + 2013)) : n = 193 := 
sorry

end smallest_n_l766_766587


namespace simplify_and_evaluate_l766_766388

theorem simplify_and_evaluate (x y : ℤ) (h1 : x = -1) (h2 : y = -2) :
  ((x + y) ^ 2 - (3 * x - y) * (3 * x + y) - 2 * y ^ 2) / (-2 * x) = -2 :=
by 
  sorry

end simplify_and_evaluate_l766_766388


namespace ellipse_equation_intersection_line_range_l766_766730

-- Definitions of conditions
def center_at_origin : Prop := (0, 0) = origin

def foci_on_x_axis (F1 F2 : Point) : Prop := F1.y = 0 ∧ F2.y = 0

def focal_distance (F1 F2 : Point) : Prop := dist F1 F2 = 4

def internal_angle_condition (P F1 F2 : Point) : Prop := 
  angle F1 P F2 ≤ π / 2

-- Statements for parts of the problem
  
-- Part (1): Proving the equation of the ellipse
theorem ellipse_equation (F1 F2 : Point) (P : Point) 
  (h1 : center_at_origin) (h2 : foci_on_x_axis F1 F2)
  (h3 : focal_distance F1 F2) (h4 : internal_angle_condition P F1 F2) :
  ellipse_eq P 8 4 := 
sorry

-- Part (2): Proving the range of m for the intersecting line and ellipse
theorem intersection_line_range (F1 F2 A B : Point) (k m : ℝ) 
  (h1 : center_at_origin) (h2 : foci_on_x_axis F1 F2)
  (h3 : focal_distance F1 F2) (h4 : internal_angle_condition A F1 F2)
  (h5 : intersects_line_with_ellipse k m A B)
  (h6 : |vec OA + vec OB| = |vec OA - vec OB|) :
  m > sqrt 2 ∨ m < -sqrt 2 :=
sorry

end ellipse_equation_intersection_line_range_l766_766730


namespace blue_markers_count_l766_766528

-- Definitions based on the problem's conditions
def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

-- Statement to prove
theorem blue_markers_count :
  total_markers - red_markers = 1028 := by
  sorry

end blue_markers_count_l766_766528


namespace largestPerfectSquareFactorOf3402_l766_766059

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem largestPerfectSquareFactorOf3402 :
  ∃ k : ℕ, isPerfectSquare k ∧ k ∣ 3402 ∧ ∀ m : ℕ, isPerfectSquare m ∧ m ∣ 3402 → m ≤ k := 
begin
  use 81,
  split,
  { use 9, exact rfl },
  split,
  { norm_num },
  { intros m h,
    cases h with hm hm',
    cases hm with x hx,
    rw hx at hm',
    by_cases h0 : x = 0,
    { subst h0, norm_num at hm', exact hm' },
    by_cases h1 : x = 9,
    { subst h1, norm_num at hm', exact hm' },
    norm_num at hm',
    sorry
  }
end

end largestPerfectSquareFactorOf3402_l766_766059


namespace simplify_radical_expression_l766_766355

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l766_766355


namespace num_pairs_a_b_l766_766952

theorem num_pairs_a_b (a b : ℝ) :
  (∃ (x y : ℤ), a * (x : ℝ) + b * (y : ℝ) = 1 ∧ (x^2 : ℝ) + (y^2 : ℝ) = 50) ↔ (a, b) ∈ {pair : ℝ × ℝ | pair ∈ 72} :=
sorry

end num_pairs_a_b_l766_766952


namespace RightTriangleAcuteAngles_l766_766183

theorem RightTriangleAcuteAngles (A B C M O : Type) [metric_space Type]
  (h1 : CM = AB / 4)
  (h2 : ∠ C = 90) :
  ∠ A = 15 ∧ ∠ B = 75 :=
  sorry

end RightTriangleAcuteAngles_l766_766183


namespace sum_valid_a_values_l766_766158

theorem sum_valid_a_values :
  (∑ a in {a | ∃ b : ℕ, 0 < a ∧ 0 < b ∧ (a - b) * real.sqrt (a * b) = 2016}, a) = 209 :=
sorry

end sum_valid_a_values_l766_766158


namespace sum_of_squares_of_areas_l766_766409

-- Defining the tetrahedron and the areas involved
variables {A B C D : Type} [HasInner A] [HasInner B] [HasInner C] [HasInner D]
variables (S : ℝ)
variables (S_DBC S_DAC S_DAB : ℝ)

-- Given conditions
axiom angles_at_D_are_right_angles : ∀ (α β γ : ℝ), α = 90 → β = 90 → γ = 90
axiom area_ABC : S

-- The statement to be proved
theorem sum_of_squares_of_areas (S_DBC S_DAC S_DAB : ℝ) :
  S_DBC ^ 2 + S_DAC ^ 2 + S_DAB ^ 2 = S ^ 2 :=
sorry

end sum_of_squares_of_areas_l766_766409


namespace alpha_bound_l766_766137

theorem alpha_bound (α : ℝ) (x : ℕ → ℝ) (h_x_inc : ∀ n, x n < x (n + 1))
    (x0_one : x 0 = 1) (h_alpha : α = ∑' n, x (n + 1) / (x n)^3) :
    α ≥ 3 * Real.sqrt 3 / 2 := 
sorry

end alpha_bound_l766_766137


namespace slope_PQ_l766_766969

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (d : ℕ)

-- Conditions given in the problem
def arith_seq : Prop := ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms : Prop := ∀ n : ℕ, S n = n * (a 1 + a n) / 2

def condition_a5 : Prop := a 5 = 19

def condition_S5 : Prop := S 5 = 55

-- Target to prove the slope between P(3, a_3) and Q(4, a_4) is 4
theorem slope_PQ : arith_seq a d ∧ sum_of_first_n_terms S a ∧ condition_a5 a ∧ condition_S5 S a → (a 4 - a 3 = 4) := 
by 
    sorry

end slope_PQ_l766_766969


namespace probability_of_exactly_one_shortening_l766_766857

-- Define the conditions and problem
def shorten_sequence_exactly_one_probability (n : ℕ) (p : ℝ) : ℝ :=
  nat.choose n 1 * p^1 * (1 - p)^(n - 1)

-- Given conditions
def problem_8a := shorten_sequence_exactly_one_probability 2014 0.1 ≈ 1.564e-90

-- Lean statement
theorem probability_of_exactly_one_shortening :
  problem_8a := sorry

end probability_of_exactly_one_shortening_l766_766857


namespace repeating_decimal_sum_l766_766550

theorem repeating_decimal_sum :
  let x : ℚ := 0.2 -- 0.\overline{2}
  let y : ℚ := 0.02 -- 0.\overline{02}
  x + y = 80 / 333 :=
by {
  -- We use library functions to convert repeating decimals to fractions
  let x : ℚ := 2 / 9
  let y : ℚ := 2 / 99
  -- Show the expected sum is the fraction in lowest terms:
  x + y = (80 / 333)
  sorry
}

end repeating_decimal_sum_l766_766550


namespace shorten_by_one_expected_length_l766_766854

-- Definition of the random sequence and binomial distribution properties
def digit_sequence {n : ℕ} (p : ℝ) (len : ℕ) : ℝ := 
  let q := 1 - p in
  (len.choose 1) * (p^1 * q^(len - 1))

-- Part (a) statement
theorem shorten_by_one 
  (len : ℕ) 
  (p : ℝ := 0.1) 
  (q := 1 - p) 
  (x := len - 1) : 
  digit_sequence p x = 1.564e-90 :=
by sorry

-- Part (b) statement
theorem expected_length 
  (initial_len : ℕ) 
  (p : ℝ := 0.1) (removals := initial_len - 1) 
  (expected_removed : ℝ := removals * p) : 
  (initial_len : ℝ) - expected_removed = 1813.6 :=
by sorry

end shorten_by_one_expected_length_l766_766854


namespace cube_root_of_8_l766_766735

theorem cube_root_of_8 : (∃ x : ℝ, x * x * x = 8) ∧ (∃ y : ℝ, y * y * y = 8 → y = 2) :=
by
  sorry

end cube_root_of_8_l766_766735


namespace area_triangle_PQR_l766_766269

open_locale classical

-- Define the triangle ABC with given side lengths
def triangle_ABC (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] :=
  dist A B = 20 ∧ dist B C = 21 ∧ dist A C = 29

-- Define points M and N with given ratios
def points_M_N (A B C M N : Type*) [metric_space A] [metric_space B] [metric_space C] 
              [metric_space M] [metric_space N] :=
  dist A B = 20 ∧ ∃ (AM MB : ℝ), AM = 12 ∧ MB = 8 ∧ dist A M = AM ∧ dist M B = MB ∧ 
  dist B C = 21 ∧ ∃ (CN NB : ℝ), CN = 14 ∧ NB = 7 ∧ dist C N = CN ∧ dist N B = NB

-- Define lines being parallel 
def lines_parallel (A B C M N P Q : Type*) [metric_space A] [metric_space B] [metric_space C] 
                   [metric_space M] [metric_space N] [metric_space P] [metric_space Q] :=
  line_through M P ∥ line_through B C ∧ line_through N Q ∥ line_through A B

-- Define points P and Q on side AC
def points_P_Q (A C P Q : Type*) [metric_space A] [metric_space C] [metric_space P] [metric_space Q] :=
  ∃ (P Q : Type*), P ∈ segment A C ∧ Q ∈ segment A C

-- Define the intersection point R of lines MP and NQ
def intersection_point (M P N Q R : Type*) [metric_space M] [metric_space P] [metric_space N] 
                       [metric_space Q] [metric_space R] :=
  ∃ (R : Type*), R ∈ line_through M P ∧ R ∈ line_through N Q

-- Define the area of triangle PQR
def area_of_triangle_PQR (A B C M N P Q R : Type*) [metric_space A] [metric_space B] [metric_space C] 
                         [metric_space M] [metric_space N] [metric_space P] [metric_space Q] 
                         [metric_space R] : ℝ :=
  let area_ABC := 1 / 2 * 20 * 21 in
  (area_ABC * (4/15) * (4/15))

-- Lean Theorem Statement
theorem area_triangle_PQR (A B C M N P Q R : Type*) [metric_space A] [metric_space B] [metric_space C] 
                          [metric_space M] [metric_space N] [metric_space P] [metric_space Q] 
                          [metric_space R] :
  triangle_ABC A B C →
  points_M_N A B C M N →
  lines_parallel A B C M N P Q →
  points_P_Q A C P Q →
  intersection_point M P N Q R →
  area_of_triangle_PQR A B C M N P Q R = 224 / 15 :=
sorry

end area_triangle_PQR_l766_766269


namespace slope_of_line_l_is_neg_4_div_3_l766_766255

noncomputable def vector_dot (u v : ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2

noncomputable def norm (u : ℝ × ℝ) : ℝ :=
real.sqrt (u.1 * u.1 + u.2 * u.2)

noncomputable def projection_length (v u : ℝ × ℝ) : ℝ :=
(vector_dot v u) / (norm u)

theorem slope_of_line_l_is_neg_4_div_3 :
  ∀ (u : ℝ × ℝ),
    let oa := (1, 4)
    let ob := (-3, 1)
    (projection_length oa u) = (projection_length ob u) ∨ (projection_length oa u) = -(projection_length ob u)
    ∧ (real.atan2 u.2 u.1 > real.pi / 2 ∧ real.atan2 u.2 u.1 < real.pi) 
    → (u.2 / u.1) = -4 / 3 :=
by
  sorry

end slope_of_line_l_is_neg_4_div_3_l766_766255


namespace geom_series_sum_l766_766796

theorem geom_series_sum : 
  let a := 1 in
  let r := 3 in
  let n := 8 in
  (a * (r^n - 1)) / (r - 1) = 3280 :=
by
  sorry

end geom_series_sum_l766_766796


namespace f_odd_f_monotone_solve_inequality_l766_766601

def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
by sorry

theorem f_monotone : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
by sorry

theorem solve_inequality : ∀ m : ℝ, f (1 - m) + f (1 - m^2) < 0 → (m < -2 ∨ 1 < m) :=
by sorry

end f_odd_f_monotone_solve_inequality_l766_766601


namespace isosceles_right_triangle_cover_obtuse_l766_766581

theorem isosceles_right_triangle_cover_obtuse {A B C : Type} [triangle A B C] 
    (h_obtuse : obtuse_triangle A B C) 
    (h_circumradius : circumradius A B C = 1) : 
    ∃ D E F : Type, isosceles_right_triangle D E F ∧ hypotenuse D E F = sqrt 2 + 1 ∧ triangle_on A B C D E F :=
by
  sorry

end isosceles_right_triangle_cover_obtuse_l766_766581


namespace tangent_sums_zero_l766_766470

theorem tangent_sums_zero (n : ℕ) (hodd : n % 2 = 1) (hn : n ≥ 3) (P : ℂ)
  (points : Fin n → ℂ) (in_circle : ∀ (k : Fin n), IsTangent ((circle_radius P) (points k))) :
  ∑ k in Finset.range n, (-1)^k * complex.abs (points k - P) = 0 :=
sorry

end tangent_sums_zero_l766_766470


namespace graphene_thickness_correct_notation_l766_766621

-- Defining the thickness value
def grapheneThickness : ℝ := 0.00000000034

-- The scientific notation
def scientificNotation : ℝ := 3.4 * 10^(-10)

-- Statement to prove the conversion to scientific notation is correct
theorem graphene_thickness_correct_notation : grapheneThickness = scientificNotation := by
  sorry

end graphene_thickness_correct_notation_l766_766621


namespace volume_ratio_of_regular_tetrahedron_and_cube_midpoints_l766_766195

noncomputable def regular_tetrahedron_volume (s : ℝ) : ℝ :=
  (s^3 * real.sqrt 6) / 24

noncomputable def cube_volume (s : ℝ) : ℝ :=
  (s^3) / 8

theorem volume_ratio_of_regular_tetrahedron_and_cube_midpoints (s : ℝ) (m n : ℕ) (h_rel_prime : nat.coprime m n) :
  let VT := regular_tetrahedron_volume s in
  let VC := cube_volume s in
  VT / VC = (m : ℝ) / (n : ℝ) →
  m + n = 11 :=
by
  intros
  sorry

end volume_ratio_of_regular_tetrahedron_and_cube_midpoints_l766_766195


namespace number_of_triples_l766_766156

noncomputable def count_valid_triples : ℕ :=
  ∑ a in Finset.range 2017, 2016 - a + 1

theorem number_of_triples : count_valid_triples = 2031120 := by
  sorry

end number_of_triples_l766_766156


namespace general_formula_a_n_l766_766187

noncomputable def S (a : ℕ → ℕ) : ℕ → ℕ
| 0     => 0
| (n+1) => a (n+1) + S a n

theorem general_formula_a_n (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 1 ∧ 
  (∀ n, a (n+1) = 2 * S n) →
  (∀ n, a n = if n = 1 then 1 else 2 * 3^(n-2)) :=
begin
  sorry
end

end general_formula_a_n_l766_766187


namespace range_for_p_range_for_q_range_for_p_and_q_false_or_q_true_l766_766283

-- Define propositions p and q
def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, (a - 3 / 2) > 0 ∧ (a - 3 / 2) < 1
def proposition_q (a : ℝ) : Prop := ∀ x : ℝ, ¬ (x : ℝ), (1 / 2)^(|x - 1|) ≥ a

-- The ranges for each case
theorem range_for_p (a : ℝ) : proposition_p a → (3 / 2 < a ∧ a < 5 / 2) :=
by sorry

theorem range_for_q (a : ℝ) : proposition_q a → (a > 1) :=
by sorry

theorem range_for_p_and_q_false_or_q_true (a : ℝ) : ¬ (proposition_p a ∧ proposition_q a) ∧ (proposition_p a ∨ proposition_q a) → ((1 < a ∧ a ≤ 3 / 2) ∨ (a ≥ 5 / 2)) :=
by sorry

end range_for_p_range_for_q_range_for_p_and_q_false_or_q_true_l766_766283


namespace more_ios_employees_l766_766673

theorem more_ios_employees (n m : ℕ) 
  (h1 : ∀ dA : ℕ, dA = 7n) 
  (h2 : ∀ dB : ℕ, dB = 15m) 
  (h3 : ∀ rA : ℕ, rA = 15n) 
  (h4 : ∀ rB : ℕ, rB = 9m)
  (h5 : 7 * n + 15 * m = 15 * n + 9 * m) : m > n :=
begin
  sorry
end

end more_ios_employees_l766_766673


namespace max_value_of_y_l766_766684

noncomputable def maxY (x y : ℝ) : ℝ :=
  if x^2 + y^2 = 10 * x + 60 * y then y else 0

theorem max_value_of_y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 60 * y) : 
  y ≤ 30 + 5 * Real.sqrt 37 :=
sorry

end max_value_of_y_l766_766684


namespace fourth_term_geometric_progression_l766_766130

theorem fourth_term_geometric_progression (a b c : ℝ) (h1 : a = real.cbrt 3) (h2 : b = real.root 4 3) (h3 : c = real.root 12 3) : 
  let d := c * (b / a) in d = 1 := 
by
  sorry

end fourth_term_geometric_progression_l766_766130


namespace no_such_primes_l766_766883

theorem no_such_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_three : p > 3) (hq_gt_three : q > 3) (hq_div_p2_minus_1 : q ∣ (p^2 - 1)) 
  (hp_div_q2_minus_1 : p ∣ (q^2 - 1)) : false := 
sorry

end no_such_primes_l766_766883


namespace officers_selection_count_l766_766123

theorem officers_selection_count :
  (nat.choose 20 6) - (nat.choose 12 6 + (nat.choose 8 1 * nat.choose 12 5)) = 31500 :=
by
  sorry

end officers_selection_count_l766_766123


namespace find_circle_equation_l766_766145

noncomputable def circle_equation (A B : ℝ × ℝ) (L : ℝ → ℝ → Prop) (C : ℝ × ℝ) (r : ℝ) :=
  ∀ (x y : ℝ), (A = (1, -1)) → (B = (-1, 1)) → (L = λ x y, x + y - 2 = 0) →
  (L C.1 C.2) → (dist (C.1, C.2) A = r) → ((x - C.1)^2 + (y - C.2)^2 = r^2)

theorem find_circle_equation :
  circle_equation (1, -1) (-1, 1) (λ x y, x + y - 2 = 0) (1, 1) 2 :=
by
  intros x y HA HB HL HOnLine HRadius
  rw [HA, HB, HL] at *
  sorry

end find_circle_equation_l766_766145


namespace hyperbola_parabola_intersection_l766_766403

open Real

theorem hyperbola_parabola_intersection :
  let A := (4, 4)
  let B := (4, -4)
  |dist A B| = 8 :=
by
  let hyperbola_asymptote (x y: ℝ) := x^2 - y^2 = 1
  let parabola_equation (x y : ℝ) := y^2 = 4 * x
  sorry

end hyperbola_parabola_intersection_l766_766403


namespace count_last_digit_1_is_202_l766_766671

def is_last_digit_1 (n : ℕ) : Prop := n % 10 = 1

noncomputable def A : set ℕ := {n | 1 ≤ n ∧ n ≤ 2011}

theorem count_last_digit_1_is_202 :
  (A.filter is_last_digit_1).card = 202 := sorry

end count_last_digit_1_is_202_l766_766671


namespace probability_of_shortening_exactly_one_digit_l766_766835

theorem probability_of_shortening_exactly_one_digit :
  let n := 2014 in
  let p := 0.1 in
  let q := 0.9 in
  (n.choose 1) * p ^ 1 * q ^ (n - 1) = 201.4 * 1.564 * 10^(-90) :=
by sorry

end probability_of_shortening_exactly_one_digit_l766_766835


namespace range_zero_of_roots_l766_766918

theorem range_zero_of_roots (x y z w : ℝ) (h1 : x + y + z + w = 0) 
                            (h2 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 :=
  sorry

end range_zero_of_roots_l766_766918


namespace jim_total_out_of_pocket_is_25100_l766_766678

-- Definition of the initial cost of the first ring
def initial_ring_cost : ℝ := 10000

-- Definition of the cost of the second ring (twice the initial ring cost)
def second_ring_cost : ℝ := 2 * initial_ring_cost

-- Sale price of the first ring (half of its value)
def sale_price_first_ring : ℝ := initial_ring_cost / 2

-- Payment in euros for the first ring
def payment_in_euros : ℝ := 4000

-- Exchange rate in dollars per euro
def exchange_rate : ℝ := 0.8

-- Conversion from euros to dollars
def euros_to_dollars (amount_euros : ℝ) : ℝ := amount_euros / exchange_rate

-- Definition of the exchange fee percentage
def exchange_fee_percentage : ℝ := 0.02

-- Calculate the exchange fee in dollars
def exchange_fee (amount_dollars : ℝ) : ℝ := exchange_fee_percentage * amount_dollars

-- Amount received after exchange fee
def amount_received_dollars : ℝ := euros_to_dollars payment_in_euros - exchange_fee (euros_to_dollars payment_in_euros)

-- Total out of pocket amount
def total_out_of_pocket : ℝ := initial_ring_cost + second_ring_cost - amount_received_dollars

-- Theorem to be proved
theorem jim_total_out_of_pocket_is_25100 : total_out_of_pocket = 25100 :=
by
  sorry

end jim_total_out_of_pocket_is_25100_l766_766678


namespace relationship_between_a_b_c_l766_766967

noncomputable def a : ℝ := 0.9 ^ 10
noncomputable def b : ℝ := 10 ^ 0.9
noncomputable def c : ℝ := Real.log 10 / Real.log 0.9

theorem relationship_between_a_b_c : b > a ∧ a > c := by
  sorry

end relationship_between_a_b_c_l766_766967


namespace ratio_circumference_height_max_volume_l766_766448

-- Define the problem conditions
def rectangle_perimeter := 12
def height := 2

-- Proven that the ratio of the circumference of the cylinder's base to its height is 2:1 when the volume is maximized
theorem ratio_circumference_height_max_volume :
  let base_circumference := 6 - height in
  2 * height = base_circumference :=
by
  let R := (6 - height:int) / (2 * Real.pi)
  have volume : Real := (Real.pi * (R^2) * height)
  let d_volume := (3 * (height^2) - 24 * height + 36) / (4 * Real.pi)
  have h1 : height = 2 := sorry

  simp [h1]

  sorry

end ratio_circumference_height_max_volume_l766_766448


namespace total_cost_of_trip_is_192_l766_766517

def calculate_total_cost_of_trip
  (distance_A : ℕ) (distance_B : ℕ) (distance_C : ℕ)
  (mileage : ℕ) (tank_capacity : ℕ)
  (price_A : ℝ) (price_B : ℝ) (price_C : ℝ) : ℝ :=
let total_distance := distance_A + distance_B + distance_C in
let gallons_needed := total_distance / mileage in
let fillups := 3 in -- since he fills up once in each city
let total_gallons := fillups * tank_capacity in
let cost_A := tank_capacity * price_A in
let cost_B := tank_capacity * price_B in
let cost_C := tank_capacity * price_C in
cost_A + cost_B + cost_C

theorem total_cost_of_trip_is_192 :
  calculate_total_cost_of_trip 290 450 620 30 20 3.10 3.30 3.20 = 192 := by
  sorry

end total_cost_of_trip_is_192_l766_766517


namespace symmetric_sum_l766_766716

theorem symmetric_sum (m n : ℤ) (hA : n = 3) (hB : m = -2) : m + n = 1 :=
by
  rw [hA, hB]
  exact rfl

end symmetric_sum_l766_766716
