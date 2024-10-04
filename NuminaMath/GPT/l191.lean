import Mathlib
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Field
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.QuadraticEquations
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.ParametricCurves
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.Integral.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.GroupTheory.Group.Defs
import Mathlib.LinearAlgebra.Determinant
import Mathlib.NumberTheory.Primorial
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.Basic
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Topology.Instances.Real

namespace factorial_sum_division_l191_191279

theorem factorial_sum_division : (7.factorial + 8.factorial + 9.factorial) / 6.factorial = 567 := by
  sorry

end factorial_sum_division_l191_191279


namespace three_person_subcommittees_from_eight_l191_191799

theorem three_person_subcommittees_from_eight :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end three_person_subcommittees_from_eight_l191_191799


namespace highest_value_of_a_divisible_by_8_l191_191308

theorem highest_value_of_a_divisible_by_8 :
  ∃ (a : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (8 ∣ (100 * a + 16)) ∧ 
  (∀ (b : ℕ), (0 ≤ b ∧ b ≤ 9) → 8 ∣ (100 * b + 16) → b ≤ a) :=
sorry

end highest_value_of_a_divisible_by_8_l191_191308


namespace three_person_subcommittees_from_eight_l191_191801

theorem three_person_subcommittees_from_eight :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end three_person_subcommittees_from_eight_l191_191801


namespace convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191876

theorem convert_polar_to_rectangular (ρ θ x y : ℝ) (hρ : ρ = 2 * √2 * cos θ) :
  (x - √2) ^ 2 + y ^ 2 = 2 :=
sorry

theorem parametric_equations_and_no_common_points (x y θ : ℝ) :
  let A := (1 : ℝ, 0 : ℝ)
  let P (θ : ℝ) := (3 - √2 + 2 * cos θ, 2 * sin θ)
  let C1 := (P 0).1 ^ 2 + (P 0).2 ^ 2 = 4
  (x - √2) ^ 2 + y ^ 2 = 2 ∧ (C1 = [((x, y) = P θ)]) → False :=
sorry

end convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191876


namespace excenter_on_line_LM_l191_191072

open_locale classical

-- Let \(ABCD\) be a cyclic convex quadrilateral.
variables (A B C D E L M : Type*) [cyclic_convex_quadrilateral A B C D]

-- Let \(\Gamma\) be its circumcircle.
variable Γ : circumcircle A B C D

-- Let \(E\) be the intersection of the diagonals \(AC\) and \(BD\).
variable hE : is_intersection (A, C) (B, D) E

-- Let \(L\) be the center of the circle tangent to sides \(AB, BC\), and \(CD\).
variable hL : is_circle_center_tangent_to_sides L A B C D

-- Let \(M\) be the midpoint of the arc \(BC\) of \(\Gamma\) not containing \(A\) and \(D\).
variable hM : is_arc_midpoint Γ B C (λ p, ¬((A = p) ∨ (D = p))) M

-- Prove that the excenter of triangle \(BCE\) opposite \(E\) lies on the line \(LM\).
theorem excenter_on_line_LM (N : Type*) [excenter_triangle_opposite E B C N] : 
  lies_on_line L M N :=
sorry

end excenter_on_line_LM_l191_191072


namespace derived_triangles_similar_iff_right_triangle_derived_triangle_1_similar_iff_eq_triangles_derived_triangle_2_similar_iff_height_eq_base_l191_191346

-- Define the basic structure of the given problem
structure IsoscelesTriangle (a b : ℝ) :=
  (height_a : ℝ)
  (height_b : ℝ)

-- Derived triangles on given heights with legs equal to 'a'
structure DerivedTriangle (triangle : IsoscelesTriangle) :=
  (height : ℝ)
  (base : ℝ := triangle.a)

-- Create an instance of IsoscelesTriangle
def original_triangle (a b : ℝ) (ma mb : ℝ) : IsoscelesTriangle a b := {
  height_a := ma,
  height_b := mb
}

-- Construct derived triangles
def derived_triangle_1 (triangle : IsoscelesTriangle) : DerivedTriangle triangle := {
  height := triangle.height_a
}

def derived_triangle_2 (triangle : IsoscelesTriangle) : DerivedTriangle triangle := {
  height := triangle.height_b
}

-- Problem statements as Lean theorems
theorem derived_triangles_similar_iff_right_triangle (a b ma mb : ℝ) :
  is_right_triangle (original_triangle a b ma mb) ↔
  triangle_similarity (derived_triangle_1 (original_triangle a b ma mb))
                      (derived_triangle_2 (original_triangle a b ma mb)) :=
sorry

theorem derived_triangle_1_similar_iff_eq_triangles (a b ma mb : ℝ) :
  a = b ↔ triangle_similarity (original_triangle a b ma mb)
                               (derived_triangle_1 (original_triangle a b ma mb)) :=
sorry

theorem derived_triangle_2_similar_iff_height_eq_base (a b ma mb : ℝ) :
  ma = a ↔ triangle_similarity (original_triangle a b ma mb)
                                (derived_triangle_2 (original_triangle a b ma mb)) :=
sorry

end derived_triangles_similar_iff_right_triangle_derived_triangle_1_similar_iff_eq_triangles_derived_triangle_2_similar_iff_height_eq_base_l191_191346


namespace find_y_l191_191517

noncomputable def inverse_proportion_y_value (x y k : ℝ) : Prop :=
  (x * y = k) ∧ (x + y = 52) ∧ (x = 3 * y) ∧ (x = -10) → (y = -50.7)

theorem find_y (x y k : ℝ) (h : inverse_proportion_y_value x y k) : y = -50.7 :=
  sorry

end find_y_l191_191517


namespace solve_exponent_equation_l191_191696

theorem solve_exponent_equation (x y z : ℕ) :
  7^x + 1 = 3^y + 5^z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  sorry

end solve_exponent_equation_l191_191696


namespace sum_geom_series_i_l191_191764

noncomputable def geometric_sum (n : ℕ) (r : ℂ) : ℂ :=
(1 - r^(n+1)) / (1 - r)

theorem sum_geom_series_i : 
  geometric_sum 2018 (complex.I) = complex.I :=
sorry

end sum_geom_series_i_l191_191764


namespace cheese_left_after_10_customers_l191_191164

theorem cheese_left_after_10_customers :
  ∀ (S : ℕ → ℚ), (∀ n, S n = (20 * n) / (n + 10)) →
  20 - S 10 = 10 := by
  sorry

end cheese_left_after_10_customers_l191_191164


namespace num_integers_in_abs_inequality_l191_191387

theorem num_integers_in_abs_inequality (x : ℤ) : 
  (|x - 3| ≤ 7.4) → real.count (λ x, (↑(-4) : ℝ) ≤ x ∧ x ≤ 10.4) = 15 :=
by
  sorry

end num_integers_in_abs_inequality_l191_191387


namespace conditional_probability_of_B_given_A_l191_191162

noncomputable def num_points_facing_up_first_die : Type := { x : ℕ // x > 0 ∧ x <= 6 }

noncomputable def num_points_facing_up_second_die : Type := { y : ℕ // y > 0 ∧ y <= 6 }

noncomputable def event_A (x y : ℕ) : Prop := (x + y) % 2 = 1

noncomputable def event_B (x y : ℕ) : Prop := x + y < 6

theorem conditional_probability_of_B_given_A :
  let outcomes := (finset.product (finset.range 1 6) (finset.range 1 6))
  let event_A_outcomes := outcomes.filter (λ (p : ℕ × ℕ), event_A p.1 p.2)
  let event_B_outcomes := outcomes.filter (λ (p : ℕ × ℕ), event_B p.1 p.2)
  let event_A_and_B_outcomes := event_A_outcomes.filter (λ (p : ℕ × ℕ), event_B p.1 p.2)
  P_B_given_A : ℚ :=
  P_B_given_A = (event_A_and_B_outcomes.card / outcomes.card) / (event_A_outcomes.card / outcomes.card) :=
by
  sorry

end conditional_probability_of_B_given_A_l191_191162


namespace min_abs_difference_l191_191814

theorem min_abs_difference (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a * b - 4 * a + 3 * b = 221) : |a - b| = 1 :=
sorry

end min_abs_difference_l191_191814


namespace C_and_C1_no_common_points_l191_191914

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  let x := rho * cos theta
  let y := rho * sin theta
  (x, y)

def curve_C_eq (theta : ℝ) : ℝ :=
  2 * sqrt 2 * cos theta

def A := (1, 0 : ℝ)

-- Predicate indicating point M is on curve C given in rectangular coordinates.
def on_curve_C (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  (x - sqrt 2) ^ 2 + y ^ 2 = 2

-- Definition for point P satisfying the given vector equation.
def point_P (A M : ℝ × ℝ) : ℝ × ℝ :=
  let x_A := A.1
  let y_A := A.2
  let x_M := M.1
  let y_M := M.2
  (√2 * (x_M - x_A) + x_A, √2 * (y_M - y_A) + y_A)

-- Prove that the rectangular coordinate equation is as specified and parametrized locus C_1 exists and has no common points with C.
theorem C_and_C1_no_common_points (M P : ℝ × ℝ) (theta : ℝ)
  (h_M_on_C : on_curve_C M)
  (h_P_eq : P = point_P A M) :
  let C1_center := (3 - sqrt 2, 0)
  let C1_radius := 2
  let C_center   := (sqrt 2, 0)
  let C_radius   := sqrt 2
  let C1_eq := (x - (3 - sqrt 2))^2 + y^2 = 4
  (M, P) ∉ C ∩ C1 :=
sorry

end C_and_C1_no_common_points_l191_191914


namespace systematic_sampling_prizes_l191_191829

theorem systematic_sampling_prizes (n : ℕ) (p : ℕ -> ℕ) 
  (h₀ : ∀ i, 1 ≤ i → i ≤ 300 → p i = i) 
  (h₁ : ∀ k, 1 ≤ k → k ≤ 100 → ∃ m, 1 ≤ m ∧ m ≤ 100 ∧ p m = 6) 
  (h₂ : ∀ j, 1 ≤ j → j ≤ 100 → p (j + 100) = p j + 100):
  (p 1 = 6 ∧ p 101 = 106 ∧ p 201 = 206) := 
begin
  sorry
end

end systematic_sampling_prizes_l191_191829


namespace symmetric_function_expression_l191_191364

variable (f : ℝ → ℝ)
variable (h_sym : ∀ x y, f (-2 - x) = - f x)
variable (h_def : ∀ x, 0 < x → f x = 1 / x)

theorem symmetric_function_expression : ∀ x, x < -2 → f x = 1 / (2 + x) :=
by
  intro x
  intro hx
  sorry

end symmetric_function_expression_l191_191364


namespace shortest_distance_ln_curve_to_line_l191_191145

noncomputable def distance_from_curve_to_line : ℝ :=
  let f (x : ℝ) := Real.log (2 * x - 1)
  let L (x y : ℝ) := 2 * x - y + 3
  let point_on_curve := (1, f 1)
  Real.abs (2 * point_on_curve.1 - point_on_curve.2 + 3) / Real.sqrt (2^2 + (-1)^2)

theorem shortest_distance_ln_curve_to_line : distance_from_curve_to_line = Real.sqrt 5 := sorry

end shortest_distance_ln_curve_to_line_l191_191145


namespace decode_last_e_occurrence_l191_191218

-- Define the cryptographic shift function
noncomputable def letter_shift (ch : Char) (shift : ℕ) : Char :=
  let base := 'a'.val.to_nat
  let target := (ch.val.to_nat - base + shift) % 26 + base
  Char.of_nat target

-- Define the problem's conditions
def cryptographic_shift (msg : String) (char : Char) (nth_occurrence : ℕ) : Char :=
  if nth_occurrence = 1 then
    letter_shift char 2
  else
    let sum_first_n_even := nth_occurrence * (nth_occurrence + 1)
    letter_shift char sum_first_n_even

-- Define the function to count occurrences of a character in a string
def count_occurrences (msg : String) (char : Char) : ℕ :=
  msg.to_list.filter (λ c => c = char).length

-- The main theorem to prove
theorem decode_last_e_occurrence :
  let msg := "We see severe weather events, even severe extremities emerge."
  let last_e := 'e'
  let occurrences := count_occurrences msg last_e
  cryptographic_shift msg last_e occurrences = 'u' :=
  by
  sorry

end decode_last_e_occurrence_l191_191218


namespace sum_of_fractions_l191_191312

theorem sum_of_fractions:
  (2 / 5) + (3 / 8) + (1 / 4) = 1 + (1 / 40) :=
by
  sorry

end sum_of_fractions_l191_191312


namespace find_f_2_solve_inequality_l191_191444

noncomputable def f : ℝ → ℝ :=
  sorry -- definition of f cannot be constructed without further info

axiom f_decreasing : ∀ x y : ℝ, 0 < x → 0 < y → (x ≤ y → f x ≥ f y)

axiom f_additive : ∀ x y : ℝ, 0 < x → 0 < y → f (x + y) = f x + f y - 1

axiom f_4 : f 4 = 5

theorem find_f_2 : f 2 = 3 :=
  sorry

theorem solve_inequality (m : ℝ) (h : f (m - 2) ≤ 3) : m ≥ 4 :=
  sorry

end find_f_2_solve_inequality_l191_191444


namespace remainder_of_b97_is_52_l191_191953

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem remainder_of_b97_is_52 : (b 97) % 81 = 52 := 
sorry

end remainder_of_b97_is_52_l191_191953


namespace solve_equation_l191_191301

theorem solve_equation (x : ℝ) :
  (√(√(√(√x)))) = 15 / (8 - √(√(√(√x)))) ↔ x = 625 ∨ x = 81 :=
by sorry

end solve_equation_l191_191301


namespace min_match_7_min_match_general_l191_191987

-- Assuming the natural number type is used for the number of teams
variable (n : ℕ) (hn : 3 ≤ n)

-- Define the properties required by Turán's theorem
def t_in_turan (n r : ℕ) : ℕ := ⌊(n^2 : ℝ) / (r^2 : ℝ)⌋ -- Turán's theorem approximation

-- Define the complete graph K_n
def complete_graph_edges (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define the minimum number of matches needed
def min_matches (n : ℕ) : ℕ :=
  (complete_graph_edges n) - (t_in_turan n 2)

-- Write the theorem to assert the calculated values
theorem min_match_7 : min_matches 7 = 9 := by {
  sorry
}

theorem min_match_general (n : ℕ) (hn : 3 ≤ n) : min_matches n = 
  (n * (n - 1) / 2) - ⌊(n^2 : ℝ) / 4⌋ := by {
  sorry
}

end min_match_7_min_match_general_l191_191987


namespace mabel_tomatoes_l191_191973

theorem mabel_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 := by
  sorry

end mabel_tomatoes_l191_191973


namespace largest_integer_l191_191729

theorem largest_integer (a b c d : ℤ) 
  (h1 : a + b + c = 210) 
  (h2 : a + b + d = 230) 
  (h3 : a + c + d = 245) 
  (h4 : b + c + d = 260) : 
  d = 105 :=
by 
  sorry

end largest_integer_l191_191729


namespace lattice_point_distance_l191_191229

theorem lattice_point_distance (d : ℝ) : 
  (∃ (vertices : set (ℝ × ℝ)), vertices = {(0, 0), (100, 0), (100, 100), (0, 100)} ⊆ set.univ) ∧
  (∃ (p : ℝ), p = 3 / 4 ∧ ∀ (x y : ℝ), x ∈ set.Icc 0 100 → y ∈ set.Icc 0 100 → 
    (let distance := λ (x y : ℝ × ℝ), real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)
     in (distance (x, y) < d) -> (x, y) ∈ {(i, j) | i ∈ (set.Icc 0 100), j ∈ (set.Icc 0 100)})) → 
  d = 0.5 :=
by
  sorry

end lattice_point_distance_l191_191229


namespace bianca_deleted_songs_l191_191579

theorem bianca_deleted_songs :
  ∃ S : ℕ, 2 + S + 7 = 17 ∧ S = 8 :=
by
  use 8
  split
  -- First part is ensuring the equation holds
  { simp }
  -- Second part is showing the correct number of songs
  { refl }

end bianca_deleted_songs_l191_191579


namespace cosine_BHD_l191_191407

-- Definitions for angles and cosines in a rectangular solid
def angle_DHG := 30
def angle_FHB := 45

theorem cosine_BHD (angle_DHG angle_FHB : ℝ) (h1 : angle_DHG = 30) (h2 : angle_FHB = 45) :
  real.cos (real.to_radians (angle_BHD angle_DHG angle_FHB)) = 2 / 3 :=
sorry

end cosine_BHD_l191_191407


namespace find_k_plus_p_plus_q_l191_191765

variables (A B C D E : Type)
variables [metric_space ℝ] [HasAngle A B C] [LineSegment ℝ BC BD CE]
variables (m : ℝ → ℝ) -- a function to measure angles

-- Conditions
def conditions (m : ℝ → ℝ) : Prop :=
  m (A.angle B C) = 45 ∧
  Segment.length BC 15 ∧
  LineSegment.perpendicular BD AC :=
  LineSegment.perpendicular CE AB ∧
  m (DBC.angle) = 2 * m (ECB.angle)

-- Prove / Theorem statement
theorem find_k_plus_p_plus_q (h: conditions m) : ∃ (k p q : ℝ),
  EC.length = k * (sqrt p + sqrt q) ∧
  k + p + q = 11 :=
sorry

end find_k_plus_p_plus_q_l191_191765


namespace three_person_subcommittees_from_eight_l191_191800

theorem three_person_subcommittees_from_eight :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end three_person_subcommittees_from_eight_l191_191800


namespace handheld_fan_sampling_with_replacement_handheld_fan_sampling_without_replacement_handheld_fan_proportion_with_replacement_handheld_fan_proportion_without_replacement_l191_191630

theorem handheld_fan_sampling_with_replacement :
  (∃ x : ℕ → ℕ → ℕ, x 2 3 = 2) →
  (∃ μ : ℕ → ℝ, μ 2 = 6 / 5) :=
by
  intros
  sorry

theorem handheld_fan_sampling_without_replacement :
  (∃ p0 p1 p2 : ℝ, p0 = 158 / 995 ∧ p1 = 480 / 995 ∧ p2 = 357 / 995) →
  (∃ μ : ℕ → ℝ, μ 2 = 6 / 5) :=
by
  intros
  sorry

theorem handheld_fan_proportion_with_replacement :
  (∃ p : ℝ, p = 0.66647) →
  (∃ f10 : ℝ → ℝ, | f10(10) - 3 / 5 | ≤ 0.1) :=
by
  intros
  sorry

theorem handheld_fan_proportion_without_replacement :
  (∃ p : ℝ, p = 0.67908) →
  (∃ f10 : ℝ → ℝ, | f10(10) - 3 / 5 | ≤ 0.1) :=
by
  intros
  sorry

end handheld_fan_sampling_with_replacement_handheld_fan_sampling_without_replacement_handheld_fan_proportion_with_replacement_handheld_fan_proportion_without_replacement_l191_191630


namespace laborers_count_l191_191159

theorem laborers_count (L : ℝ) (h : 10 = 0.385 * L) : L ≈ 26 :=
by
  sorry

end laborers_count_l191_191159


namespace eval_expression_l191_191689

theorem eval_expression :
  2 * log 3 2 - log 3 (32 / 9) + log 3 8 - 5^(2 * log 5 3) = -7 :=
by
  sorry

end eval_expression_l191_191689


namespace convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191878

theorem convert_polar_to_rectangular (ρ θ x y : ℝ) (hρ : ρ = 2 * √2 * cos θ) :
  (x - √2) ^ 2 + y ^ 2 = 2 :=
sorry

theorem parametric_equations_and_no_common_points (x y θ : ℝ) :
  let A := (1 : ℝ, 0 : ℝ)
  let P (θ : ℝ) := (3 - √2 + 2 * cos θ, 2 * sin θ)
  let C1 := (P 0).1 ^ 2 + (P 0).2 ^ 2 = 4
  (x - √2) ^ 2 + y ^ 2 = 2 ∧ (C1 = [((x, y) = P θ)]) → False :=
sorry

end convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191878


namespace part_one_part_two_l191_191781

noncomputable def f (x : ℝ) : ℝ :=
  (x + 1) * Real.log x - x + 1

def f_prime (x : ℝ) : ℝ :=
  Real.log x + 1 / x

def g (x : ℝ) : ℝ :=
  Real.log x - x

theorem part_one (a : ℝ) :
  (∀ x > 0, x * f_prime x ≤ x^2 + a * x + 1) → a ∈ Ici (-1) :=
sorry

theorem part_two :
  ∀ x > 0, (x - 1) * f x ≥ 0 :=
sorry

end part_one_part_two_l191_191781


namespace num_p_on_line_satisfy_condition_l191_191378

-- Define the given points M and N
structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := { x := -1, y := 0 }
def N : Point := { x := 1, y := 0 }

-- Define the line l
def line_l (P : Point) : Prop := P.y = -2 * P.x + 3

-- Define the condition |PM| + |PN| = 4
def dist (P1 P2 : Point) : ℝ := real.sqrt ((P1.x - P2.x) ^ 2 + (P1.y - P2.y) ^ 2)

def dist_condition (P : Point) : Prop := dist P M + dist P N = 4

-- Lean Theorem: Prove there are exactly 2 points on the line satisfying the condition
theorem num_p_on_line_satisfy_condition : 
  ∃ P1 P2 : Point, line_l P1 ∧ line_l P2 ∧ dist_condition P1 ∧ dist_condition P2 ∧ P1 ≠ P2 ∧ ∀ P : Point, line_l P ∧ dist_condition P → (P = P1 ∨ P = P2) :=
by sorry

end num_p_on_line_satisfy_condition_l191_191378


namespace fencing_cost_l191_191304

noncomputable def total_cost_fencing (diameter : ℝ) (rate : ℝ) : ℝ :=
  let circumference := Real.pi * diameter
  circumference * rate

theorem fencing_cost :
  total_cost_fencing 20 1.50 ≈ 94.25 :=
by 
  sorry

end fencing_cost_l191_191304


namespace value_of_y_l191_191140

noncomputable def k : ℝ := 168.75

theorem value_of_y (x y : ℝ) (h1 : x * y = k) (h2 : x + y = 30) (h3 : x = 3 * y) : y = -16.875 :=
by 
  sorry

end value_of_y_l191_191140


namespace seed_mixture_ryegrass_percent_l191_191998

theorem seed_mixture_ryegrass_percent (R : ℝ) :
  let X := 0.40
  let percentage_X_in_mixture := 1 / 3
  let percentage_Y_in_mixture := 2 / 3
  let final_ryegrass := 0.30
  (final_ryegrass = percentage_X_in_mixture * X + percentage_Y_in_mixture * R) → 
  R = 0.25 :=
by
  intros X percentage_X_in_mixture percentage_Y_in_mixture final_ryegrass H
  sorry

end seed_mixture_ryegrass_percent_l191_191998


namespace trigonometric_identity_l191_191355

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / 
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := 
by
  -- proof steps are omitted, using sorry to skip the proof.
  sorry

end trigonometric_identity_l191_191355


namespace negation_of_exactly_one_even_l191_191170

variable (a b c : ℕ)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ ¬ is_even b ∧ is_even c)

theorem negation_of_exactly_one_even :
  ¬ exactly_one_even a b c ↔ (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
                                 (is_even a ∧ is_even b) ∨
                                 (is_even a ∧ is_even c) ∨
                                 (is_even b ∧ is_even c) :=
by sorry

end negation_of_exactly_one_even_l191_191170


namespace max_ratio_three_digit_sum_l191_191177

theorem max_ratio_three_digit_sum (N a b c : ℕ) (hN : N = 100 * a + 10 * b + c) (ha : 1 ≤ a) (hb : b ≤ 9) (hc : c ≤ 9) :
  (∀ (N' a' b' c' : ℕ), N' = 100 * a' + 10 * b' + c' → 1 ≤ a' → b' ≤ 9 → c' ≤ 9 → (N' : ℚ) / (a' + b' + c') ≤ 100) :=
sorry

end max_ratio_three_digit_sum_l191_191177


namespace polarToRectangular_noCommonPoints_l191_191886

section PolarToRectangular
variables {θ : ℝ} {ρ : ℝ}

-- Condition: polar coordinate equation ρ = 2√2 cos θ
def polarEq (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Correct answer: rectangular coordinate equation (x - √2)² + y² = 2
theorem polarToRectangular 
  (h : ρ = polarEq θ)
  (x y : ℝ)
  (hx : x = ρ * cos θ)
  (hy : y = ρ * sin θ) :
  (x - sqrt 2) ^ 2 + y ^ 2 = 2 :=
sorry

end PolarToRectangular

section LocusAndCommonPoints
variables {x y x₁ y₁ : ℝ} {C C₁ : set (ℝ × ℝ)}

-- Condition: (1, 0) is point A, M is a moving point on circle C, P satisfies AP = √2AM
def pointA := (1, 0 : ℝ) 
def isOnCircleC (M : ℝ × ℝ) : Prop := (M.1 - sqrt 2) ^ 2 + M.2 ^ 2 = 2
def scaledVector (A P M : ℝ × ℝ) : Prop :=
  P.1 - A.1 = sqrt 2 * (M.1 - A.1) ∧ P.2 - A.2 = sqrt 2 * (M.2 - A.2)

-- Correct answer: Circles C and C₁ do not have any common points
theorem noCommonPoints
  (A : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : isOnCircleC M)
  (P : ℝ × ℝ)
  (hP : scaledVector A P M) :
  ∀ P : ℝ × ℝ, ¬ (isOnCircleC P ∧ (P.1 - (3 - sqrt 2)) ^ 2 + P.2 ^ 2 = 4) :=
sorry

end LocusAndCommonPoints

end polarToRectangular_noCommonPoints_l191_191886


namespace C_and_C1_no_common_points_l191_191917

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  let x := rho * cos theta
  let y := rho * sin theta
  (x, y)

def curve_C_eq (theta : ℝ) : ℝ :=
  2 * sqrt 2 * cos theta

def A := (1, 0 : ℝ)

-- Predicate indicating point M is on curve C given in rectangular coordinates.
def on_curve_C (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  (x - sqrt 2) ^ 2 + y ^ 2 = 2

-- Definition for point P satisfying the given vector equation.
def point_P (A M : ℝ × ℝ) : ℝ × ℝ :=
  let x_A := A.1
  let y_A := A.2
  let x_M := M.1
  let y_M := M.2
  (√2 * (x_M - x_A) + x_A, √2 * (y_M - y_A) + y_A)

-- Prove that the rectangular coordinate equation is as specified and parametrized locus C_1 exists and has no common points with C.
theorem C_and_C1_no_common_points (M P : ℝ × ℝ) (theta : ℝ)
  (h_M_on_C : on_curve_C M)
  (h_P_eq : P = point_P A M) :
  let C1_center := (3 - sqrt 2, 0)
  let C1_radius := 2
  let C_center   := (sqrt 2, 0)
  let C_radius   := sqrt 2
  let C1_eq := (x - (3 - sqrt 2))^2 + y^2 = 4
  (M, P) ∉ C ∩ C1 :=
sorry

end C_and_C1_no_common_points_l191_191917


namespace num_three_person_subcommittees_from_eight_l191_191805

def num_committees (n k : ℕ) : ℕ := (Nat.fact n) / ((Nat.fact k) * (Nat.fact (n - k)))

theorem num_three_person_subcommittees_from_eight (n : ℕ) (h : n = 8) : num_committees n 3 = 56 :=
by
  rw [h]
  sorry

end num_three_person_subcommittees_from_eight_l191_191805


namespace probability_equation_solution_l191_191260

theorem probability_equation_solution :
  (∃ x : ℝ, 3 * x ^ 2 - 8 * x + 5 = 0 ∧ 0 ≤ x ∧ x ≤ 1) → ∃ x = 1 :=
by
  sorry

end probability_equation_solution_l191_191260


namespace find_m_l191_191794

theorem find_m (m : ℝ) : 
  (∃ m, 2 * (abs (7 - 2) / real.sqrt (4^2 + (-2)^2)) = abs m / real.sqrt (1^2 + (-2)^2) ∧ m > 0) → m = 5 := 
begin
  intro h,
  sorry
end

end find_m_l191_191794


namespace C_and_C1_no_common_points_l191_191913

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  let x := rho * cos theta
  let y := rho * sin theta
  (x, y)

def curve_C_eq (theta : ℝ) : ℝ :=
  2 * sqrt 2 * cos theta

def A := (1, 0 : ℝ)

-- Predicate indicating point M is on curve C given in rectangular coordinates.
def on_curve_C (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  (x - sqrt 2) ^ 2 + y ^ 2 = 2

-- Definition for point P satisfying the given vector equation.
def point_P (A M : ℝ × ℝ) : ℝ × ℝ :=
  let x_A := A.1
  let y_A := A.2
  let x_M := M.1
  let y_M := M.2
  (√2 * (x_M - x_A) + x_A, √2 * (y_M - y_A) + y_A)

-- Prove that the rectangular coordinate equation is as specified and parametrized locus C_1 exists and has no common points with C.
theorem C_and_C1_no_common_points (M P : ℝ × ℝ) (theta : ℝ)
  (h_M_on_C : on_curve_C M)
  (h_P_eq : P = point_P A M) :
  let C1_center := (3 - sqrt 2, 0)
  let C1_radius := 2
  let C_center   := (sqrt 2, 0)
  let C_radius   := sqrt 2
  let C1_eq := (x - (3 - sqrt 2))^2 + y^2 = 4
  (M, P) ∉ C ∩ C1 :=
sorry

end C_and_C1_no_common_points_l191_191913


namespace domain_of_f_l191_191124

-- Define the function f(x)
noncomputable def f : ℝ → ℝ := λ x, (sqrt (1 - x)) / x

-- Define the domain of the function f(x)
def domain_f (x : ℝ) : Prop := (x ≤ 1 ∧ x ≠ 0)

-- State the theorem about the domain of the function f(x)
theorem domain_of_f : {x : ℝ | x ≤ 1 ∧ x ≠ 0} = {x : ℝ | x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 1} :=
by 
  sorry

end domain_of_f_l191_191124


namespace number_of_special_permutations_l191_191946

theorem number_of_special_permutations : 
  (Finset.card {p : Finset (Fin 12) // 
     ∃ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} a_{12} : Fin 12),
     Set.univ = {a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_{10}, a_{11}, a_{12}} ∧
     a_1 > a_2 ∧ a_2 > a_3 ∧ a_3 > a_4 ∧ a_4 > a_5 ∧ a_5 > a_6 ∧
     a_6 < a_7 ∧ a_7 < a_8 ∧ a_8 < a_9 ∧ a_9 < a_{10} ∧ a_{10} < a_{11} ∧ a_{11} < a_{12}} = 462 :=
begin
  sorry
end

end number_of_special_permutations_l191_191946


namespace must_be_divisor_of_p_l191_191958

theorem must_be_divisor_of_p (p q r s : ℕ) (hpq : Nat.gcd p q = 40) (hqr : Nat.gcd q r = 50) 
  (hrs : Nat.gcd r s = 75) (hsp : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) : 17 ∣ p :=
sorry

end must_be_divisor_of_p_l191_191958


namespace a9_value_l191_191327

theorem a9_value (a : ℕ → ℝ) (x : ℝ) (h : (1 + x) ^ 10 = 
  (a 0) + (a 1) * (1 - x) + (a 2) * (1 - x)^2 + 
  (a 3) * (1 - x)^3 + (a 4) * (1 - x)^4 + 
  (a 5) * (1 - x)^5 + (a 6) * (1 - x)^6 + 
  (a 7) * (1 - x)^7 + (a 8) * (1 - x)^8 + 
  (a 9) * (1 - x)^9 + (a 10) * (1 - x)^10) : 
  a 9 = -20 :=
sorry

end a9_value_l191_191327


namespace complex_square_eq_l191_191352

open Complex

theorem complex_square_eq {a b : ℝ} (h : (a + b * Complex.I)^2 = Complex.mk 3 4) : a^2 + b^2 = 5 :=
by {
  sorry
}

end complex_square_eq_l191_191352


namespace increasing_interval_function_l191_191508

theorem increasing_interval_function :
  ∀ (k : ℤ), increasing_interval (λ x : ℝ, real.sqrt (2 * real.sin (2 * x - real.pi / 3) - 1))
  (set.Icc (k * real.pi + real.pi / 4) (k * real.pi + real.pi / 12 * 5)) :=
by sorry

end increasing_interval_function_l191_191508


namespace quartic_polynomial_with_roots_3_plus_sqrt_5_and_2_plus_sqrt_7_l191_191695

theorem quartic_polynomial_with_roots_3_plus_sqrt_5_and_2_plus_sqrt_7 :
  ∃ (p : Polynomial ℚ), p.monic ∧ (Polynomial.root p (3 + Real.sqrt 5)) ∧ (Polynomial.root p (2 + Real.sqrt 7)) ∧
  (p = Polynomial.X^4 - 10 * Polynomial.X^3 + 21 * Polynomial.X^2 + 6 * Polynomial.X - 12) :=
sorry

end quartic_polynomial_with_roots_3_plus_sqrt_5_and_2_plus_sqrt_7_l191_191695


namespace C_and_C1_no_common_points_l191_191911

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  let x := rho * cos theta
  let y := rho * sin theta
  (x, y)

def curve_C_eq (theta : ℝ) : ℝ :=
  2 * sqrt 2 * cos theta

def A := (1, 0 : ℝ)

-- Predicate indicating point M is on curve C given in rectangular coordinates.
def on_curve_C (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  (x - sqrt 2) ^ 2 + y ^ 2 = 2

-- Definition for point P satisfying the given vector equation.
def point_P (A M : ℝ × ℝ) : ℝ × ℝ :=
  let x_A := A.1
  let y_A := A.2
  let x_M := M.1
  let y_M := M.2
  (√2 * (x_M - x_A) + x_A, √2 * (y_M - y_A) + y_A)

-- Prove that the rectangular coordinate equation is as specified and parametrized locus C_1 exists and has no common points with C.
theorem C_and_C1_no_common_points (M P : ℝ × ℝ) (theta : ℝ)
  (h_M_on_C : on_curve_C M)
  (h_P_eq : P = point_P A M) :
  let C1_center := (3 - sqrt 2, 0)
  let C1_radius := 2
  let C_center   := (sqrt 2, 0)
  let C_radius   := sqrt 2
  let C1_eq := (x - (3 - sqrt 2))^2 + y^2 = 4
  (M, P) ∉ C ∩ C1 :=
sorry

end C_and_C1_no_common_points_l191_191911


namespace e_is_dq_sequence_l191_191728

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d a₀, ∀ n, a n = a₀ + n * d

def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∃ q b₀, q > 0 ∧ ∀ n, b n = b₀ * q^n

def is_dq_sequence (c : ℕ → ℕ) : Prop :=
  ∃ a b, is_arithmetic_sequence a ∧ is_geometric_sequence b ∧ ∀ n, c n = a n + b n

def e (n : ℕ) : ℕ :=
  n + 2^n

theorem e_is_dq_sequence : is_dq_sequence e :=
  sorry

end e_is_dq_sequence_l191_191728


namespace part_a_part_b_l191_191585

-- Part (a)
def F (x : ℝ) : ℝ := 2 * (Int.floor x) - Real.cos (3 * Real.pi * (x - Int.floor x))

theorem part_a (y : ℝ) : 
  Continuous F ∧ ∀ y : ℝ, ∃! x₁ x₂ x₃ : ℝ, F x₁ = y ∧ F x₂ = y ∧ F x₃ = y := 
sorry

-- Part (b)
theorem part_b (k : ℕ) (hk : Even k) (hk_pos : 0 < k) :
  ¬ ∃ f : ℝ → ℝ, Continuous f ∧ ∀ y ∈ (Set.range f), ∃ (l : Finset ℝ), l.card = k ∧ ∀ x ∈ l, f x = y := 
sorry

end part_a_part_b_l191_191585


namespace bill_original_selling_price_l191_191586

theorem bill_original_selling_price 
  (P : ℝ) 
  (S : ℝ)
  (h1 : S = 1.10 * P)
  (h2 : 1.17 * P - S = 49) :
  S = 770 :=
by
  have h3 : P = 700, from calc
    P = 49 / 0.07 : by
      field_simp [h2]
      ring
    _ = 700 : by norm_num,
  calc
    S = 1.10 * P : h1
    _ = 1.10 * 700 : by rw h3
    _ = 770 : by norm_num

end bill_original_selling_price_l191_191586


namespace convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191893

-- Definitions from the problem
def polar_eq_C (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Conversion to rectangular coordinates
def rectangular_eq_C (x y : ℝ) : Prop := (x - sqrt 2)^2 + y^2 = 2

-- Definitions of points and vectors
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨1, 0⟩

def M (θ : ℝ) : Point :=
  let ρ := polar_eq_C θ
  ⟨ρ * cos θ, ρ * sin θ⟩

def P (x y : ℝ) : Point := ⟨x, y⟩

-- Conditions and transformations
def vector_eq (p1 p2 : Point) (a : ℝ) : Prop :=
  P p1.x p1.y = ⟨a * p2.x + (1 - a) * A.x, a * p2.y + (1 - a) * A.y⟩

def parametric_eq_P (θ : ℝ) : Point :=
  ⟨3 - sqrt 2 + 2 * cos θ, 2 * sin θ⟩

-- Proof problems
theorem convert_polar_to_rectangular (θ : ℝ) :
  rectangular_eq_C (polar_eq_C θ * cos θ) (polar_eq_C θ * sin θ) :=
sorry

theorem parametric_eq_valid (θ : ℝ) :
  vector_eq (parametric_eq_P θ) (M θ) (sqrt 2) :=
sorry

theorem no_common_points :
  ∀ θ₁ θ₂, parametric_eq_P θ₁ ≠ M θ₂ :=
sorry

end convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191893


namespace sally_total_score_l191_191830

theorem sally_total_score :
  ∀ (correct incorrect unanswered : ℕ) (score_correct score_incorrect : ℝ),
    correct = 17 →
    incorrect = 8 →
    unanswered = 5 →
    score_correct = 1 →
    score_incorrect = -0.25 →
    (correct * score_correct +
     incorrect * score_incorrect +
     unanswered * 0) = 15 :=
by
  intros correct incorrect unanswered score_correct score_incorrect
  intros h_corr h_incorr h_unan h_sc h_si
  sorry

end sally_total_score_l191_191830


namespace solve_fraction_problem_l191_191555

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l191_191555


namespace ice_cream_orders_l191_191986

variables (V C S M O T : ℕ)

theorem ice_cream_orders :
  (V = 56) ∧ (C = 28) ∧ (S = 70) ∧ (M = 42) ∧ (O = 84) ↔
  (V = 2 * C) ∧
  (S = 25 * T / 100) ∧
  (M = 15 * T / 100) ∧
  (T = 280) ∧
  (V = 20 * T / 100) ∧
  (V + C + S + M + O = T) :=
by
  sorry

end ice_cream_orders_l191_191986


namespace uncle_dave_nieces_l191_191167

theorem uncle_dave_nieces : ∃ n : ℕ, 1573 = n * 143 ∧ n = 11 := 
begin
  use 11,
  split,
  {
    norm_num,
  },
  {
    refl,
  },
end

end uncle_dave_nieces_l191_191167


namespace miranda_monthly_savings_l191_191458

noncomputable def total_cost := 260
noncomputable def sister_contribution := 50
noncomputable def months := 3

theorem miranda_monthly_savings : 
  (total_cost - sister_contribution) / months = 70 := 
by
  sorry

end miranda_monthly_savings_l191_191458


namespace C_and_C1_no_common_points_l191_191844

noncomputable def C_equation : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - real.sqrt 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (1, 0)

noncomputable def M_on_C (θ : ℝ) : ℝ × ℝ :=
  let x := 2 * real.sqrt 2 * real.cos θ * real.cos θ in
  let y := 2 * real.sqrt 2 * real.sin θ * real.sin θ in
  (x, y)

noncomputable def P_on_locus (x_M y_M : ℝ) : ℝ × ℝ :=
  let x := real.sqrt 2 * (x_M - point_A.1) + point_A.1 in
  let y := real.sqrt 2 * y_M in
  (x, y)

noncomputable def Locus_P_equation (θ : ℝ) : ℝ × ℝ :=
  let x := 3 - real.sqrt 2 + 2 * real.cos θ in
  let y := 2 * real.sin θ in
  (x, y)

def has_common_points (C1 C2 : set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C1 ∧ p ∈ C2

theorem C_and_C1_no_common_points :
  ¬ has_common_points
    {p | C_equation p}
    {p | ∃ θ, P_on_locus (M_on_C θ).1 (M_on_C θ).2 = p} :=
sorry

end C_and_C1_no_common_points_l191_191844


namespace rational_units_digit_sum_squares_as_decimal_units_digit_sum_squares_is_rational_l191_191961

-- Define the units digit of the sum of squares up to n
def units_digit_sum_squares (n : ℕ) : ℕ := (List.range (n + 1)).map (λ x => (x ^ 2) % 10).sum % 10

-- The periodicity of the units digit sequence
theorem rational_units_digit_sum_squares_as_decimal : ∀ n : ℕ, units_digit_sum_squares n = units_digit_sum_squares (n + 20) :=
sorry

-- The main theorem that 0.a_1a_2a_3... is rational
theorem units_digit_sum_squares_is_rational : ℚ :=
0 + Rat.from_decimal 20 (List.range 20).map units_digit_sum_squares sorry

end rational_units_digit_sum_squares_as_decimal_units_digit_sum_squares_is_rational_l191_191961


namespace math_problem_solution_l191_191142

noncomputable def a_range : Set ℝ := {a : ℝ | (0 < a ∧ a ≤ 1) ∨ (5 ≤ a ∧ a < 6)}

theorem math_problem_solution (a : ℝ) :
  (1 - 4 * (a^2 - 6 * a) > 0 ∧ a^2 - 6 * a < 0) ∨ ((a - 3)^2 - 4 < 0)
  ∧ ¬((1 - 4 * (a^2 - 6 * a) > 0 ∧ a^2 - 6 * a < 0) ∧ ((a - 3)^2 - 4 < 0)) →
  a ∈ a_range :=
sorry

end math_problem_solution_l191_191142


namespace cyclic_iff_perpendicular_diagonals_l191_191086

section ProofProblem

variables {A B C D P Q R S : Point}
variables {incircle : Circle}

-- Conditions
def quadrilateral_with_incircle (A B C D P Q R S : Point) (incircle : Circle) :=
  tangent incircle A B P ∧ tangent incircle B C Q ∧
  tangent incircle C D R ∧ tangent incircle D A S

-- Statement
theorem cyclic_iff_perpendicular_diagonals
  (h : quadrilateral_with_incircle A B C D P Q R S incircle) :
  cyclic A B C D ↔ perpendicular (line_through P R) (line_through Q S) :=
sorry

end ProofProblem

end cyclic_iff_perpendicular_diagonals_l191_191086


namespace num_dislikers_tv_books_games_is_correct_l191_191989

-- Definitions of the conditions as given in step A
def total_people : ℕ := 1500
def pct_dislike_tv : ℝ := 0.4
def pct_dislike_tv_books : ℝ := 0.15
def pct_dislike_tv_books_games : ℝ := 0.5

-- Calculate intermediate values
def num_tv_dislikers := pct_dislike_tv * total_people
def num_tv_books_dislikers := pct_dislike_tv_books * num_tv_dislikers
def num_tv_books_games_dislikers := pct_dislike_tv_books_games * num_tv_books_dislikers

-- Final proof statement ensuring the correctness of the solution
theorem num_dislikers_tv_books_games_is_correct :
  num_tv_books_games_dislikers = 45 := by
  -- Sorry placeholder for the proof. In actual Lean usage, this would require fulfilling the proof obligations.
  sorry

end num_dislikers_tv_books_games_is_correct_l191_191989


namespace count_solutions_eq_4_l191_191501

theorem count_solutions_eq_4 :
  ∀ x : ℝ, (x^2 - 5)^2 = 16 → x = 3 ∨ x = -3 ∨ x = 1 ∨ x = -1  := sorry

end count_solutions_eq_4_l191_191501


namespace value_of_m_l191_191819

theorem value_of_m (m : ℝ) (h : ∃ x : ℝ, x^2 - 3 * x + m = 0 ∧ x = 1) : m = 2 :=
by 
  obtain ⟨x, hx1, hx2⟩ := h
  rw [hx2, pow_two, mul_one, one_mul] at hx1
  linarith

end value_of_m_l191_191819


namespace no_real_roots_of_quadratic_l191_191154

theorem no_real_roots_of_quadratic 
(a b c : ℝ) 
(h1 : b + c > a)
(h2 : b + a > c)
(h3 : c + a > b) :
(b^2 + c^2 - a^2)^2 - 4 * b^2 * c^2 < 0 :=
by
  sorry

end no_real_roots_of_quadratic_l191_191154


namespace min_diff_f_l191_191738

def f (x : ℝ) := 2017 * x ^ 2 - 2018 * x + 2019 * 2020

theorem min_diff_f (t : ℝ) : 
  let f_max := max (f t) (f (t + 2))
  let f_min := min (f t) (f (t + 2))
  (f_max - f_min) ≥ 2017 :=
sorry

end min_diff_f_l191_191738


namespace probability_equation_solution_l191_191262

theorem probability_equation_solution :
  (∃ x : ℝ, 3 * x ^ 2 - 8 * x + 5 = 0 ∧ 0 ≤ x ∧ x ≤ 1) → ∃ x = 1 :=
by
  sorry

end probability_equation_solution_l191_191262


namespace trihedral_angle_plane_angles_acute_l191_191240

open Real

-- Define what it means for an angle to be acute
def is_acute (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π / 2

-- Define the given conditions
variable {A B C α β γ : ℝ}
variable (hA : is_acute A)
variable (hB : is_acute B)
variable (hC : is_acute C)

-- State the problem: if dihedral angles are acute, then plane angles are also acute
theorem trihedral_angle_plane_angles_acute :
  is_acute A → is_acute B → is_acute C → is_acute α ∧ is_acute β ∧ is_acute γ :=
sorry

end trihedral_angle_plane_angles_acute_l191_191240


namespace total_drums_l191_191461

theorem total_drums (x y : ℕ) (hx : 30 * x + 20 * y = 160) : x + y = 7 :=
sorry

end total_drums_l191_191461


namespace angle_bisector_coincide_l191_191412

theorem angle_bisector_coincide
  (A B C S : Type)
  [EuclideanGeometry.AffinePlane A B C S]
  (H : Affine.Point A → Affine.Point B → Affine.Point C → Affine.Point S)
  (A' : Affine.Point S)
  (H_foot : Affine.Point A → Affine.Point B → Affine.Point C) :
  let angle_bisector (x y z : Affine.Point S) := EuclideanGeometry.angle_bisector x y z in
  angle_bisector A H A' = angle_bisector A B C :=
by
  sorry

end angle_bisector_coincide_l191_191412


namespace taxi_cost_per_mile_l191_191092

variable (x : ℝ)

-- Mike's total cost
def Mike_total_cost := 2.50 + 36 * x

-- Annie's total cost
def Annie_total_cost := 2.50 + 5.00 + 16 * x

-- The primary theorem to prove
theorem taxi_cost_per_mile : Mike_total_cost x = Annie_total_cost x → x = 0.25 := by
  sorry

end taxi_cost_per_mile_l191_191092


namespace Ivan_walk_time_l191_191691

variables (u v : ℝ) (T t : ℝ)

-- Define the conditions
def condition1 : Prop := T = 10 * v / u
def condition2 : Prop := T + 70 = t
def condition3 : Prop := v * t = u * T + v * (t - T + 70)

-- Problem statement: Given the conditions, prove T = 80
theorem Ivan_walk_time (h1 : condition1 u v T) (h2 : condition2 T t) (h3 : condition3 u v T t) : 
  T = 80 := by
  sorry

end Ivan_walk_time_l191_191691


namespace find_coordinates_l191_191122

theorem find_coordinates
  (x y : ℝ)
  (h1 : -π ≤ x ∧ x ≤ 2 * π)
  (h2 : 0 ≤ y ∧ y ≤ 3 * π)
  (h3 : sin x + sin y = sin 3)
  (h4 : cos x + cos y = cos 3) :
  (x = 3 - π / 3 ∧ y = 3 + π / 3) ∨
  (x = 3 - 5 * π / 3 ∧ y = 3 + 5 * π / 3) ∨
  (x = 3 + π / 3 ∧ y = 3 + 5 * π / 3) ∨
  (x = 3 + π / 3 ∧ y = 3 - π / 3) :=
sorry

end find_coordinates_l191_191122


namespace find_root_l191_191248

theorem find_root :
  ∃ x : ℝ, (3 * x^2 - 8 * x + 5 = 0) ∧ (x = 1) :=
begin
  sorry
end

end find_root_l191_191248


namespace find_n_l191_191172

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n < 103 ∧ 100 * n % 103 = 65 % 103 ∧ n = 68 :=
by
  sorry

end find_n_l191_191172


namespace FG_square_l191_191044

def trapezoid_EFGH (EF FG GH EH : ℝ) : Prop :=
  ∃ x y : ℝ, 
  EF = 4 ∧
  EH = 31 ∧
  FG = x ∧
  GH = y ∧
  x^2 + (y - 4)^2 = 961 ∧
  x^2 = 4 * y

theorem FG_square (EF EH FG GH x y : ℝ) (h : trapezoid_EFGH EF FG GH EH) :
  FG^2 = 132 :=
by
  obtain ⟨x, y, h1, h2, h3, h4, h5, h6⟩ := h
  exact sorry

end FG_square_l191_191044


namespace cost_of_camel_is_6000_l191_191597

noncomputable def cost_of_camel : ℕ := 6000

variables (C H O E : ℕ)
variables (cost_of_camel_rs cost_of_horses cost_of_oxen cost_of_elephants : ℕ)

-- Conditions
axiom cond1 : 10 * C = 24 * H
axiom cond2 : 16 * H = 4 * O
axiom cond3 : 6 * O = 4 * E
axiom cond4 : 10 * E = 150000

theorem cost_of_camel_is_6000
    (cond1 : 10 * C = 24 * H)
    (cond2 : 16 * H = 4 * O)
    (cond3 : 6 * O = 4 * E)
    (cond4 : 10 * E = 150000) :
  cost_of_camel = 6000 := 
sorry

end cost_of_camel_is_6000_l191_191597


namespace convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191897

-- Definitions from the problem
def polar_eq_C (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Conversion to rectangular coordinates
def rectangular_eq_C (x y : ℝ) : Prop := (x - sqrt 2)^2 + y^2 = 2

-- Definitions of points and vectors
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨1, 0⟩

def M (θ : ℝ) : Point :=
  let ρ := polar_eq_C θ
  ⟨ρ * cos θ, ρ * sin θ⟩

def P (x y : ℝ) : Point := ⟨x, y⟩

-- Conditions and transformations
def vector_eq (p1 p2 : Point) (a : ℝ) : Prop :=
  P p1.x p1.y = ⟨a * p2.x + (1 - a) * A.x, a * p2.y + (1 - a) * A.y⟩

def parametric_eq_P (θ : ℝ) : Point :=
  ⟨3 - sqrt 2 + 2 * cos θ, 2 * sin θ⟩

-- Proof problems
theorem convert_polar_to_rectangular (θ : ℝ) :
  rectangular_eq_C (polar_eq_C θ * cos θ) (polar_eq_C θ * sin θ) :=
sorry

theorem parametric_eq_valid (θ : ℝ) :
  vector_eq (parametric_eq_P θ) (M θ) (sqrt 2) :=
sorry

theorem no_common_points :
  ∀ θ₁ θ₂, parametric_eq_P θ₁ ≠ M θ₂ :=
sorry

end convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191897


namespace length_FD_of_folded_square_l191_191423

theorem length_FD_of_folded_square :
  let A := (0, 0)
  let B := (8, 0)
  let D := (0, 8)
  let C := (8, 8)
  let E := (6, 0)
  let F := (8, 8 - (FD : ℝ))
  (ABCD_square : ∀ {x y : ℝ}, (x = 0 ∨ x = 8) ∧ (y = 0 ∨ y = 8)) →  
  let DE := (6 - 0 : ℝ)
  let Pythagorean_statement := (8 - FD) ^ 2 = FD ^ 2 + 6 ^ 2
  ∃ FD : ℝ, FD = 7 / 4 :=
sorry

end length_FD_of_folded_square_l191_191423


namespace sin_arithmetic_sequence_l191_191697

noncomputable def sin_value (a : ℝ) := Real.sin (a * (Real.pi / 180))

theorem sin_arithmetic_sequence (a : ℝ) : 
  (0 < a) ∧ (a < 360) ∧ (sin_value a + sin_value (3 * a) = 2 * sin_value (2 * a)) ↔ a = 90 ∨ a = 270 :=
by 
  sorry

end sin_arithmetic_sequence_l191_191697


namespace total_legs_of_camden_dogs_l191_191659

-- Defining the number of dogs Justin has
def justin_dogs : ℕ := 14

-- Defining the number of dogs Rico has
def rico_dogs : ℕ := justin_dogs + 10

-- Defining the number of dogs Camden has
def camden_dogs : ℕ := 3 * rico_dogs / 4

-- Defining the total number of legs Camden's dogs have
def camden_dogs_legs : ℕ := camden_dogs * 4

-- The proof statement
theorem total_legs_of_camden_dogs : camden_dogs_legs = 72 :=
by
  -- skip proof
  sorry

end total_legs_of_camden_dogs_l191_191659


namespace find_f_when_extremum_monotone_condition_l191_191784

-- Define the function f(x) and its derivative
def f (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c
def f_deriv (a b x : ℝ) : ℝ := -3*x^2 + 2*a*x + b

-- Given conditions
variable (a b c : ℝ)

axiom slope_tangent : f_deriv a b 1 = -3
axiom function_value_at_1 : f a b c 1 = -2
axiom extremum_condition : f_deriv a b (-2) = 0

-- Statement for question 1
theorem find_f_when_extremum :
  ∃ (a b c : ℝ), f a b c x = -x^3 - 2*x^2 + 4*x - 3 := sorry

-- Statement for question 2
theorem monotone_condition (b : ℝ) :
  (∀ x ∈ set.Icc (-2 : ℝ) (0 : ℝ), f_deriv (-2) b x ≥ 0) → b ∈ set.Ici 4 := sorry

end find_f_when_extremum_monotone_condition_l191_191784


namespace gcd_198_286_l191_191176

theorem gcd_198_286 : Nat.gcd 198 286 = 22 :=
by
  sorry

end gcd_198_286_l191_191176


namespace convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191881

theorem convert_polar_to_rectangular (ρ θ x y : ℝ) (hρ : ρ = 2 * √2 * cos θ) :
  (x - √2) ^ 2 + y ^ 2 = 2 :=
sorry

theorem parametric_equations_and_no_common_points (x y θ : ℝ) :
  let A := (1 : ℝ, 0 : ℝ)
  let P (θ : ℝ) := (3 - √2 + 2 * cos θ, 2 * sin θ)
  let C1 := (P 0).1 ^ 2 + (P 0).2 ^ 2 = 4
  (x - √2) ^ 2 + y ^ 2 = 2 ∧ (C1 = [((x, y) = P θ)]) → False :=
sorry

end convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191881


namespace probability_solution_l191_191256

theorem probability_solution : 
  let N := (3/8 : ℝ)
  let M := (5/8 : ℝ)
  let P_D_given_N := (x : ℝ) => x^2
  (3 : ℝ) * x^2 - (8 : ℝ) * x + (5 : ℝ) = 0 → x = 1 := 
by
  sorry

end probability_solution_l191_191256


namespace liters_to_pints_l191_191771

theorem liters_to_pints (LP : 0.5 = 1.08 / 2) : (3 : ℝ) * (1.08 / 0.5) = 6.5 := 
by 
  have H1 : 1.08 / 0.5 = 2.16 := by 
    sorry 
  have H2 : 3 * 2.16 = 6.48 := by 
    sorry 
  have H3 : 6.48 ≈ 6.5 := by
    sorry
  exact H3

end liters_to_pints_l191_191771


namespace distance_after_2_seconds_is_8_units_l191_191473

theorem distance_after_2_seconds_is_8_units :
  ∀ (B C : ℚ) (vB vC : ℚ),
    B = -8 →
    C = 16 →
    vB = 6 →
    vC = -2 →
    ∀ t : ℚ, 
      8 = abs (B + vB * t - (C + vC * t)) → 
      t = 2
    := 
begin
  intros B C vB vC hB hC hvB hvC t ht,
  rw [hB, hC, hvB, hvC] at ht,
  -- This results in:
  -- 8 = abs (-8 + 6 * t - (16 - 2 * t)) = abs (-24 + 8 * t)
  sorry
end

end distance_after_2_seconds_is_8_units_l191_191473


namespace remainder_b_div_11_l191_191063

theorem remainder_b_div_11 (n : ℕ) (h_pos : 0 < n) (b : ℕ) (h_b : b ≡ (5^(2*n) + 6)⁻¹ [ZMOD 11]) : b % 11 = 8 :=
by
  sorry

end remainder_b_div_11_l191_191063


namespace gcd_of_198_and_286_l191_191174

theorem gcd_of_198_and_286:
  let a := 198 
  let b := 286 
  let pf1 : a = 2 * 3^2 * 11 := by rfl
  let pf2 : b = 2 * 11 * 13 := by rfl
  gcd a b = 22 := by sorry

end gcd_of_198_and_286_l191_191174


namespace exists_equal_white_black_segment_l191_191546

variable (n : ℕ)
variable (chain : List Bool)  -- False represents a black ball, True represents a white ball

def white_ball_count (l : List Bool) : ℕ :=
  l.countp id

def black_ball_count (l : List Bool) : ℕ :=
  l.length - (white_ball_count l)

def segment_with_equal_balls (chain: List Bool) (n: ℕ) : Prop :=
  ∃ m, m + n ≤ chain.length ∧ white_ball_count (chain.slice m (m + n)) = n ∧ black_ball_count (chain.slice m (m + n)) = n

theorem exists_equal_white_black_segment 
  (h_length : chain.length = 4 * n) 
  (h_white : white_ball_count chain = 2 * n) 
  (h_black : black_ball_count chain = 2 * n) : 
  segment_with_equal_balls chain n :=
sorry

end exists_equal_white_black_segment_l191_191546


namespace arccos_lt_arctan_solution_l191_191303

theorem arccos_lt_arctan_solution (x : ℝ) : 
  (∀ x, \arccos x < \arctan x) ↔ (0.5 < x ∧ x ≤ 1) :=
sorry

end arccos_lt_arctan_solution_l191_191303


namespace george_initial_socks_l191_191734

theorem george_initial_socks (total_socks now_socks bought_socks given_socks : ℕ) : 
  total_socks = now_socks - bought_socks - given_socks → 
  total_socks = 68 → 
  bought_socks = 36 → 
  given_socks = 4 → 
  now_socks = 28 := 
by
  intros h1 h2 h3 h4
  rw [h2, h3, h4] at h1
  exact h1

end george_initial_socks_l191_191734


namespace primes_in_range_50_70_count_l191_191809

theorem primes_in_range_50_70_count :
  (finset.filter (λ p, prime p ∧ (∃ k, p = 6 * k + 1 ∨ p = 6 * k - 1)) (finset.range 71)).count ≥ 50 ≤ 4 :=
by sorry

end primes_in_range_50_70_count_l191_191809


namespace extend_string_equal_l191_191188

theorem extend_string_equal (r R : ℝ) : 
  let red_extension := 2 * Real.pi * (r + 1) - 2 * Real.pi * r,
      blue_extension := 2 * Real.pi * (R + 1) - 2 * Real.pi * R
  in red_extension = blue_extension :=
by
  sorry

end extend_string_equal_l191_191188


namespace sequence_nonzero_not_multiple_of_4_l191_191335

def sequence (a : ℕ → ℤ) : Prop :=
  (a 1 = 1) ∧ (a 2 = 2) ∧ 
  (∀ n : ℕ, 
    if (a n) * (a (n + 1)) % 2 = 0 then a (n + 2) = 5 * a (n + 1) - 3 * a n
    else a (n + 2) = a (n + 1) - a n)

theorem sequence_nonzero_not_multiple_of_4 (a : ℕ → ℤ) 
  (h : sequence a) : 
  ∀ n : ℕ, a n ≠ 0 ∧ ¬(4 ∣ a n) :=
by
  sorry

end sequence_nonzero_not_multiple_of_4_l191_191335


namespace doubling_segment_proof_l191_191379

variable {Point : Type}
variables (A B C D M P E : Point)
variables (line1 line2 : set Point)
variables (segment_ab : set Point) (segment_cd : set Point)
variable (midpoint : Point → Point → Point → Prop)
variable (intersection : Point → Point → Point → Prop)
variable (parallel : set Point → set Point → Prop)
variable (segment_length : Point → Point → ℝ)

noncomputable def doubling_segment :=
  parallel line1 line2 ∧
  segment_ab = { A, B } ∧ segment_cd = { C, D } ∧
  A ∈ line1 ∧ B ∈ line1 ∧ C ∈ line2 ∧ D ∈ line2 ∧ 
  midpoint M C D ∧
  intersection P A M ∧ intersection P B D ∧ 
  intersection E P C ∧ E ∈ segment_ab ∧
  segment_length E B = 2 * segment_length A B

theorem doubling_segment_proof : doubling_segment → 
  segment_length E B = 2 * segment_length A B :=
sorry

end doubling_segment_proof_l191_191379


namespace real_condition_implies_a_is_one_half_l191_191761

theorem real_condition_implies_a_is_one_half (a : ℝ) (h : (1 + a * Complex.i) / (2 + Complex.i) ∈ ℝ) : a = 1 / 2 :=
sorry

end real_condition_implies_a_is_one_half_l191_191761


namespace color_regions_l191_191928

theorem color_regions (n : ℕ) (h : n > 0) :
  ∃ f : ℝ × ℝ → bool, ∀ (R₁ R₂ : ℝ × ℝ) (L : set (ℝ × ℝ)),
  is_line L → (R₁ ∈ L) → (R₂ ∈ L) → f R₁ ≠ f R₂ := sorry

end color_regions_l191_191928


namespace greater_number_l191_191527

theorem greater_number (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 2) (h3 : a > b) : a = 21 := by
  sorry

end greater_number_l191_191527


namespace tangent_line_circle_midpoint_locus_l191_191368

/-- 
Let O be the circle x^2 + y^2 = 1,
M be the point (-1, -4), and
N be the point (2, 0).
-/
structure CircleTangentMidpointProblem where
  (x y : ℝ)
  (O_eq : x^2 + y^2 = 1)
  (M_eq : x = -1 ∧ y = -4)
  (N_eq : x = 2 ∧ y = 0)

/- Part (1) -/
theorem tangent_line_circle (x y : ℝ) (O_eq : x^2 + y^2 = 1) 
                            (Mx My : ℝ) : ((Mx = -1 ∧ My = -4) → 
                          
                            (x = -1 ∨ 15 * x - 8 * y - 17 = 0)) := by
  sorry

/- Part (2) -/
theorem midpoint_locus (x y : ℝ) (O_eq : x^2 + y^2 = 1) 
                       (Nx Ny : ℝ) : ((Nx = 2 ∧ Ny = 0) → 
                       
                       ((x-1)^2 + y^2 = 1 ∧ (0 ≤ x ∧ x < 1 / 2))) := by
  sorry

end tangent_line_circle_midpoint_locus_l191_191368


namespace ella_dice_roll_probability_l191_191395

theorem ella_dice_roll_probability :
  let p := (5 / 6) * (4 / 5) ^ 9 * (1 / 6) ^ 2 in
  p = 0.0008 :=
by
  let p := (5 / 6) * (4 / 5) ^ 9 * (1 / 6) ^ 2
  have h : p = (5 / 6) * (4 / 5) ^ 9 * (1 / 6) ^ 2 := rfl
  sorry

end ella_dice_roll_probability_l191_191395


namespace valid_12_letter_words_mod_1000_l191_191406

noncomputable def a : ℕ → ℕ
| 3       := 8
| (n + 4) := 2 * (a (n + 3) + c (n + 3))
| _       := 0 -- base case should cover nonnegatives only

noncomputable def b : ℕ → ℕ
| 3       := 0
| (n + 4) := a (n + 3)
| _       := 0 -- base case should cover nonnegatives only

noncomputable def c : ℕ → ℕ
| 3       := 0
| (n + 4) := 2 * b (n + 3)
| _       := 0 -- base case should cover nonnegatives only

def number_of_valid_words(n : ℕ) : ℕ :=
  a n + b n + c n

theorem valid_12_letter_words_mod_1000 :
  number_of_valid_words 12 % 1000 = X := 
sorry

end valid_12_letter_words_mod_1000_l191_191406


namespace corridor_perimeter_l191_191115

theorem corridor_perimeter
  (P1 P2 : ℕ)
  (h₁ : P1 = 16)
  (h₂ : P2 = 24) : 
  2 * ((P2 / 4 + (P1 + P2) / 4) + (P2 / 4) - (P1 / 4)) = 40 :=
by {
  -- The proof can be filled here
  sorry
}

end corridor_perimeter_l191_191115


namespace min_value_sqrt_expression_l191_191960

open Real

theorem min_value_sqrt_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    ∃ c, c = sqrt 6 ∧ (∀ (x y : ℝ), 0 < x → 0 < y → 
    sqrt ((x^2 + y^2) * (4 * x^2 + y^2)) / (x * y) ≥ c) :=
begin
    use sqrt 6,
    split,
    { refl },
    intros x y hx hy,
    sorry
end

end min_value_sqrt_expression_l191_191960


namespace circle_equation_l191_191397

theorem circle_equation 
  (C : ℝ → ℝ → Prop)
  (h1 : C 1 3)
  (h2 : C 3 5)
  (h_center : ∃ a b, (∀ x y, C x y ↔ (x - a)^2 + (y - b)^2 = 4) ∧ 2 * a - b + 3 = 0):
  ∃ a b (r : ℝ), (∀ x y, C x y ↔ (x - a)^2 + (y - b)^2 = r^2) ∧ a = 1 ∧ b = 5 ∧ r = 2 :=
by
  sorry

end circle_equation_l191_191397


namespace smaller_pack_size_l191_191686

theorem smaller_pack_size {x : ℕ} (total_eggs large_pack_size large_packs : ℕ) (eggs_in_smaller_packs : ℕ) :
  total_eggs = 79 → large_pack_size = 11 → large_packs = 5 → eggs_in_smaller_packs = total_eggs - large_pack_size * large_packs →
  x * 1 = eggs_in_smaller_packs → x = 24 :=
by sorry

end smaller_pack_size_l191_191686


namespace trig_identity_l191_191328

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) : 
  (1 / (Real.cos α ^ 2 + Real.sin (2 * α))) = 10 / 3 := 
by 
  sorry

end trig_identity_l191_191328


namespace combined_selling_price_correct_l191_191627

def cost_A : ℕ := 500
def cost_B : ℕ := 800
def cost_C : ℕ := 1200
def profit_A : ℕ := 25
def profit_B : ℕ := 30
def profit_C : ℕ := 20

def selling_price (cost profit_percentage : ℕ) : ℕ :=
  cost + (profit_percentage * cost / 100)

def combined_selling_price : ℕ :=
  selling_price cost_A profit_A + selling_price cost_B profit_B + selling_price cost_C profit_C

theorem combined_selling_price_correct : combined_selling_price = 3105 := by
  sorry

end combined_selling_price_correct_l191_191627


namespace find_x_from_expression_l191_191314

theorem find_x_from_expression :
  (sqrt x / sqrt 0.81 + sqrt 0.81 / sqrt 0.49 = 2.507936507936508) -> x = 1.21 :=
by
  sorry

end find_x_from_expression_l191_191314


namespace any_triangle_can_be_divided_into_four_isosceles_l191_191429

theorem any_triangle_can_be_divided_into_four_isosceles (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) 
(h_c_largest: c ≥ a ∧ c ≥ b): 
  ∃ d m1 m2, 
    is_perpendicular (a, b, c) (d, a) ∧ 
    is_median (d, b) (m1, c) ∧ 
    is_median (d, c) (m2, b) ∧ 
    is_isosceles_triangle (a, m1, d) ∧ 
    is_isosceles_triangle (m1, b, d) ∧ 
    is_isosceles_triangle (a, m2, d) ∧ 
    is_isosceles_triangle (m2, c, d) :=
sorry

end any_triangle_can_be_divided_into_four_isosceles_l191_191429


namespace investment_triples_in_4_years_l191_191245

noncomputable def smallest_investment_period_to_triple (r : ℝ) : ℕ :=
  let A (P : ℝ) (t : ℕ) := P * (1 + r) ^ t in
  Nat.find (λ t, 3 < (1 + r) ^ t)

theorem investment_triples_in_4_years :
  smallest_investment_period_to_triple 0.3334 = 4 :=
by
  sorry

end investment_triples_in_4_years_l191_191245


namespace a_2023_eq_26_l191_191450

/-- 
Define the sum of the digits of a natural number.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/--
Define the sequence (a_k) and (n_k) based on given conditions.
The initial value is n_1 = 5.
-/
noncomputable def a (k : ℕ) : ℕ :=
  if k = 0 then 0
  else if k = 1 then 5^2 + 1
  else let nk := sum_of_digits (a (k - 1))
       in nk^2 + 1

/--
Prove a_{2023} = 26 given defined sequence conditions.
-/
theorem a_2023_eq_26 : a 2023 = 26 :=
  sorry

end a_2023_eq_26_l191_191450


namespace add_to_fraction_l191_191560

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l191_191560


namespace C_and_C1_no_common_points_l191_191840

noncomputable def C_equation : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - real.sqrt 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (1, 0)

noncomputable def M_on_C (θ : ℝ) : ℝ × ℝ :=
  let x := 2 * real.sqrt 2 * real.cos θ * real.cos θ in
  let y := 2 * real.sqrt 2 * real.sin θ * real.sin θ in
  (x, y)

noncomputable def P_on_locus (x_M y_M : ℝ) : ℝ × ℝ :=
  let x := real.sqrt 2 * (x_M - point_A.1) + point_A.1 in
  let y := real.sqrt 2 * y_M in
  (x, y)

noncomputable def Locus_P_equation (θ : ℝ) : ℝ × ℝ :=
  let x := 3 - real.sqrt 2 + 2 * real.cos θ in
  let y := 2 * real.sin θ in
  (x, y)

def has_common_points (C1 C2 : set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C1 ∧ p ∈ C2

theorem C_and_C1_no_common_points :
  ¬ has_common_points
    {p | C_equation p}
    {p | ∃ θ, P_on_locus (M_on_C θ).1 (M_on_C θ).2 = p} :=
sorry

end C_and_C1_no_common_points_l191_191840


namespace symmetrical_distance_l191_191834

noncomputable def distance_between_points (P P' : EuclideanSpace ℝ (Fin 3)) : ℝ := 
  (Finset.univ.sum (λ i, (P i - P' i) ^ 2)).sqrt

theorem symmetrical_distance (P P' : EuclideanSpace ℝ (Fin 3)) 
  (hP : P = ![2, 3, 5]) (hP' : P' = ![-2, 3, 5]): distance_between_points P P' = 4 := by
  sorry

end symmetrical_distance_l191_191834


namespace find_lambda_l191_191796

variables (a b : ℝ × ℝ)
def vec_add (v1 v2: ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vec_parallel (v1 v2: ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_lambda (λ : ℝ) (h₀ : a = (1, 3)) (h₁ : b = (2, 1))
  (h₂ : vec_parallel (vec_add a (vec_scalar_mul 2 b)) 
                     (vec_add (vec_scalar_mul 3 a) (vec_scalar_mul λ b))) :
  λ = 6 :=
by simp [vec_add, vec_scalar_mul, vec_parallel] at h₂; sorry

end find_lambda_l191_191796


namespace complex_number_in_first_quadrant_l191_191036

-- Define the complex number in the given problem
def complex_number : ℂ := 2 / (1 - complex.i)

-- Define the condition: the point corresponding to the complex number
def point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

-- Define the first quadrant
def in_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

-- The statement we want to prove
theorem complex_number_in_first_quadrant : in_first_quadrant (point complex_number) :=
sorry

end complex_number_in_first_quadrant_l191_191036


namespace collinear_A1_B1_C1_l191_191434

variables {A B C K A1 B1 C1 : Point}
variables (ABC : Triangle A B C) (K_sym : is_symmedian_point K ABC)
variables (hA1_def : A1 ∈ line_through B C ∧ cyclic_quad (quadrilateral A B A1 K))
variables (hB1_def : B1 ∈ line_through C A ∧ cyclic_quad (quadrilateral B C B1 K))
variables (hC1_def : C1 ∈ line_through A B ∧ cyclic_quad (quadrilateral C A C1 K))

theorem collinear_A1_B1_C1 : collinear A1 B1 C1 :=
by
  sorry

end collinear_A1_B1_C1_l191_191434


namespace johnny_words_l191_191431

def words_johnny (J : ℕ) :=
  let words_madeline := 2 * J
  let words_timothy := 2 * J + 30
  let total_words := J + words_madeline + words_timothy
  total_words = 3 * 260 → J = 150

-- Statement of the main theorem (no proof provided, hence sorry is used)
theorem johnny_words (J : ℕ) : words_johnny J :=
by sorry

end johnny_words_l191_191431


namespace geometric_sum_3030_l191_191526

theorem geometric_sum_3030 {a r : ℝ}
  (h1 : a * (1 - r ^ 1010) / (1 - r) = 300)
  (h2 : a * (1 - r ^ 2020) / (1 - r) = 540) :
  a * (1 - r ^ 3030) / (1 - r) = 732 :=
sorry

end geometric_sum_3030_l191_191526


namespace white_paint_amount_is_correct_l191_191455

noncomputable def totalAmountOfPaint (bluePaint: ℝ) (bluePercentage: ℝ): ℝ :=
  bluePaint / bluePercentage

noncomputable def whitePaintAmount (totalPaint: ℝ) (whitePercentage: ℝ): ℝ :=
  totalPaint * whitePercentage

theorem white_paint_amount_is_correct (bluePaint: ℝ) (bluePercentage: ℝ) (whitePercentage: ℝ) (totalPaint: ℝ) :
  bluePaint = 140 → bluePercentage = 0.7 → whitePercentage = 0.1 → totalPaint = totalAmountOfPaint 140 0.7 →
  whitePaintAmount totalPaint 0.1 = 20 :=
by
  intros
  sorry

end white_paint_amount_is_correct_l191_191455


namespace last_two_digits_of_7_pow_2015_l191_191464

theorem last_two_digits_of_7_pow_2015 : ((7 ^ 2015) % 100) = 43 := 
by
  sorry

end last_two_digits_of_7_pow_2015_l191_191464


namespace polar_to_rectangular_eq_no_common_points_l191_191850

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_rectangular_eq (ρ θ x y: ℝ) (h1: ρ = 2 * Real.sqrt 2 * Real.cos θ) 
(h2: polar_to_rectangular ρ θ = (x, y)) : (x - Real.sqrt 2) ^ 2 + y ^ 2 = 2 :=
by
  sorry

theorem no_common_points (x y θ : ℝ) (ρ : ℝ := 2 * Real.sqrt 2 * Real.cos θ)
(A := (1, 0) : ℝ × ℝ)
(M := polar_to_rectangular ρ θ)
(P := (sqrt 2 * (M.1 - 1) + 1, sqrt 2 * M.2)) : 
((P.1 - (3 - Real.sqrt 2)) ^ 2 + P.2 ^ 2 = 4) → 
¬ ((P.1 - Real.sqrt 2) ^ 2 + P.2 ^ 2 = 2) :=
by
  intro H1 H2
  sorry

end polar_to_rectangular_eq_no_common_points_l191_191850


namespace max_n_for_positive_sum_l191_191366

-- Define the arithmetic sequence \(a_n\)
def arithmetic_sequence (a d : ℤ) (n : ℕ) := a + n * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (a d : ℤ) (n : ℕ) := n * (2 * a + (n-1) * d) / 2

theorem max_n_for_positive_sum 
  (a : ℤ) 
  (d : ℤ) 
  (h_max_sum : ∃ m : ℕ, S_n a d m = S_n a d (m+1))
  (h_ratio : (arithmetic_sequence a d 15) / (arithmetic_sequence a d 14) < -1) :
  27 = 27 :=
sorry

end max_n_for_positive_sum_l191_191366


namespace probability_not_all_same_l191_191160

-- Definitions for the given conditions
inductive Color
| red | yellow | green

def draw (n : ℕ) : list Color := replicate n Color.red ++ replicate n Color.yellow ++ replicate n Color.green

-- Problem statement
theorem probability_not_all_same : 
  let total_ways := 3^3 in
  let same_color_ways := 3 in
  (1 - (same_color_ways / total_ways) = 8 / 9) :=
by
  sorry

end probability_not_all_same_l191_191160


namespace circumcircle_centers_congruent_l191_191409

variable {A B C D E F X Y Z : Type}
variable (h_belongs : ∀ {X Y Z : Type}, D = midpoint B X ∧ E = midpoint C Y ∧ F = midpoint A Z)
variable (h_triangle : ∀ ⦃A B C : Type⦄, geometry.is_triangle A B C ∧ geometry.is_acute A B C ∧ geometry.is_scalene A B C)
variable (h_altitudes : ∀ ⦃A B C D E F : Type⦄, altitude A D B ∧ altitude B E C ∧ altitude C F A)

theorem circumcircle_centers_congruent :
  let O1 := circumcenter A C X
  let O2 := circumcenter A B Y
  let O3 := circumcenter B C Z
  triangle.is_congruent (triangle O1 O2 O3) (triangle A B C) :=
sorry

end circumcircle_centers_congruent_l191_191409


namespace proof_problem_l191_191903

noncomputable def polar_to_rectangular : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x - sqrt 2) ^ 2 + y ^ 2 = 2) ↔ (∀ θ, rho = 2 * sqrt 2 * Real.cos θ → (x ^ 2 + y ^ 2 = rho ^ 2) ∧ (x = rho * Real.cos θ)) sorry

noncomputable def check_common_points : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x = 3 - sqrt 2 + 2 * Real.cos θ) ∧ (y = 2 * Real.sin θ)) ∧
              ∀ (M_P C_P : ℝ), C_P = 2 ∧ |(3 - sqrt 2) - sqrt 2| = 3 - 2 * sqrt 2 < 2 - sqrt 2 :=
              ¬ ∃ p, (p ∈ C) ∧ (p ∈ C_1) sorry

theorem proof_problem : polar_to_rectangular ∧ check_common_points := by
  sorry

end proof_problem_l191_191903


namespace correct_equation_for_t_l191_191683

-- Define the rates for Doug and Dave
def dougRate : ℝ := 1 / 5
def daveRate : ℝ := 1 / 7

-- Combined rate
def combinedRate : ℝ := dougRate + daveRate

-- Theorem to prove the correct equation for time t
theorem correct_equation_for_t (t : ℝ) : combinedRate * (t - 1) = 1 :=
by
  -- Proof omitted, as only the statement is required
  sorry

end correct_equation_for_t_l191_191683


namespace tan_alpha_plus_pi_over_4_l191_191367

theorem tan_alpha_plus_pi_over_4 (x y : ℝ) (h1 : 3 * x + 4 * y = 0) : 
  Real.tan ((Real.arctan (- 3 / 4)) + π / 4) = 1 / 7 := 
by
  sorry

end tan_alpha_plus_pi_over_4_l191_191367


namespace midpoints_collinear_l191_191539

theorem midpoints_collinear 
  (A B C A' B' C' : Point)
  (h₁: Triangle ABC)
  (h₂: Triangle A'B'C')
  (identical: ∃ (f : Point → Point), Isometry f ∧ (f A', f B', f C') = (A, B, C))
  (flipped: ∃ (g : Point → Point), Isometry g ∧ IsReflection g ∧ (g A', g B', g C') = (A, B, C))
  : Collinear [midpoint A A', midpoint B B', midpoint C C'] := 
begin
  sorry
end

end midpoints_collinear_l191_191539


namespace sin_gt_cos_cond_l191_191813

-- Definitions and conditions
variable (A B C : ℝ)
variable [OrderedField ℝ]
-- A, B, and C are interior angles of a triangle, hence 0 < A, B, C < π 
-- and A + B + C = π.
variable (h_triangle : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ A + B + C = Real.pi)

-- The main theorem to prove
theorem sin_gt_cos_cond (h_sine: Real.sin A > Real.sin B) : Real.cos A + Real.cos (A + C) < 0 ↔ Real.sin A > Real.sin B :=
by
  sorry

end sin_gt_cos_cond_l191_191813


namespace thirteen_numbers_sum_996_l191_191150

theorem thirteen_numbers_sum_996 :
  ∃ (S : Finset ℕ), S.card = 13 ∧ S.sum id = 996 ∧ ∀ n ∈ S, digit_sum n = 6 :=
begin
  let S := {6, 15, 24, 33, 42, 51, 60, 105, 114, 123, 132, 141, 150},
  use S,
  split,
  { -- S.card = 13
    simp [S] },
  split,
  { -- S.sum id = 996
    simp [S, Finset.sum_insert, Finset.sum_singleton],
    norm_num },
  { -- ∀ n ∈ S, digit_sum n = 6
    intros n hn,
    finset_cases n hn;
    norm_num }
end

/-- Helper function to compute the sum of the digits of a number. --/
def digit_sum (n : ℕ) : ℕ :=
  string.toNat (string.join (string.map (λ c, if '0' ≤ c ∧ c ≤ '9' then to_string c else "")) (to_string n))

end thirteen_numbers_sum_996_l191_191150


namespace probability_solution_l191_191257

theorem probability_solution : 
  let N := (3/8 : ℝ)
  let M := (5/8 : ℝ)
  let P_D_given_N := (x : ℝ) => x^2
  (3 : ℝ) * x^2 - (8 : ℝ) * x + (5 : ℝ) = 0 → x = 1 := 
by
  sorry

end probability_solution_l191_191257


namespace card_distribution_ways_l191_191294

theorem card_distribution_ways : ∀ (cards : Finset ℕ), (cards = {1, 2, ..., 20}) →
  (∃ elbert yaiza : Finset ℕ, elbert.card = 10 ∧ yaiza.card = 10 ∧
  ∀ elbert_hand yaiza_hand : List ℕ,
  elbert_hand = elbert.to_finset.val ∧ yaiza_hand = yaiza.to_finset.val →
  turns_played elbert_hand yaiza_hand = ({1, a, b, c, d} : Finset ℕ) →
  yaiza_loses (to_finset elbert_hand) (to_finset yaiza_hand) →
  num_distributions elbert_hand yaiza_hand = 240) := sorry

end card_distribution_ways_l191_191294


namespace gcd_of_198_and_286_l191_191173

theorem gcd_of_198_and_286:
  let a := 198 
  let b := 286 
  let pf1 : a = 2 * 3^2 * 11 := by rfl
  let pf2 : b = 2 * 11 * 13 := by rfl
  gcd a b = 22 := by sorry

end gcd_of_198_and_286_l191_191173


namespace cos_B_value_max_angle_B_value_l191_191360

-- Definition of the conditions in Lean
variables {A B C: ℝ} {a b c : ℝ}

-- Conditions: sides opposite to angles A, B, C are a, b, c respectively
def sides_opposite (a b c : ℝ) (A B C : ℝ) : Prop :=
  true  -- This is a placeholder for the actual matching of sides to angles which Lean geometry might provide

-- Condition: sides a, b, c form a geometric progression
def geometric_progression (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Condition: sin C = 2 * sin A
def sin_condition (A C : ℝ) : Prop :=
  sin C = 2 * sin A

-- Definitions combining all conditions
def triangle_conditions (a b c A B C : ℝ) : Prop :=
  sides_opposite a b c A B C ∧ geometric_progression a b c ∧ sin_condition A C

noncomputable def cos_B (a b c A B C : ℝ) [triangle_conditions a b c A B C] : ℝ :=
  (a^2 + c^2 - b^2) / (2 * a * c)

theorem cos_B_value (a b c A B C : ℝ) [triangle_conditions a b c A B C] :
  cos_B a b c A B C = 3 / 4 :=
sorry

noncomputable def max_angle_B (a b c A B C : ℝ) [triangle_conditions a b c A B C] : ℝ :=
B

theorem max_angle_B_value (a b c A B C : ℝ) [triangle_conditions a b c A B C] :
  max_angle_B a b c A B C = π / 3 ∧ a = b ∧ b = c :=
sorry

end cos_B_value_max_angle_B_value_l191_191360


namespace prod_mod_11_remainder_zero_l191_191550

theorem prod_mod_11_remainder_zero : (108 * 110) % 11 = 0 := 
by sorry

end prod_mod_11_remainder_zero_l191_191550


namespace farmer_harvest_correct_l191_191617

def estimated_harvest : ℕ := 48097
def additional_harvest : ℕ := 684
def total_harvest : ℕ := 48781

theorem farmer_harvest_correct : estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end farmer_harvest_correct_l191_191617


namespace convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191898

-- Definitions from the problem
def polar_eq_C (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Conversion to rectangular coordinates
def rectangular_eq_C (x y : ℝ) : Prop := (x - sqrt 2)^2 + y^2 = 2

-- Definitions of points and vectors
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨1, 0⟩

def M (θ : ℝ) : Point :=
  let ρ := polar_eq_C θ
  ⟨ρ * cos θ, ρ * sin θ⟩

def P (x y : ℝ) : Point := ⟨x, y⟩

-- Conditions and transformations
def vector_eq (p1 p2 : Point) (a : ℝ) : Prop :=
  P p1.x p1.y = ⟨a * p2.x + (1 - a) * A.x, a * p2.y + (1 - a) * A.y⟩

def parametric_eq_P (θ : ℝ) : Point :=
  ⟨3 - sqrt 2 + 2 * cos θ, 2 * sin θ⟩

-- Proof problems
theorem convert_polar_to_rectangular (θ : ℝ) :
  rectangular_eq_C (polar_eq_C θ * cos θ) (polar_eq_C θ * sin θ) :=
sorry

theorem parametric_eq_valid (θ : ℝ) :
  vector_eq (parametric_eq_P θ) (M θ) (sqrt 2) :=
sorry

theorem no_common_points :
  ∀ θ₁ θ₂, parametric_eq_P θ₁ ≠ M θ₂ :=
sorry

end convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191898


namespace alvin_benny_time_difference_l191_191643

-- Define the conditions
def Alvin_speed_flat := 25 -- in kph
def Alvin_speed_downhill := 35 -- in kph
def Alvin_speed_uphill := 10 -- in kph
def Benny_speed_flat := 35 -- in kph
def Benny_speed_downhill := 45 -- in kph
def Benny_speed_uphill := 15 -- in kph

def dist_XY := 15 -- km, distance from X to Y (uphill)
def dist_YZ := 25 -- km, distance from Y to Z (downhill)
def dist_ZX := 30 -- km, distance from Z to X (flat)

-- Define the calculations
noncomputable def Alvin_time : ℝ := (dist_XY / Alvin_speed_uphill) + (dist_YZ / Alvin_speed_downhill) + (dist_ZX / Alvin_speed_flat)
noncomputable def Benny_time : ℝ := (dist_ZX / Benny_speed_flat) + (dist_YZ / Benny_speed_uphill) + (dist_XY / Benny_speed_downhill)

noncomputable def time_difference := (Alvin_time - Benny_time) * 60 -- converting from hours to minutes

-- Prove the time difference
theorem alvin_benny_time_difference : time_difference ≈ 33.42 :=
by {
  sorry
}

end alvin_benny_time_difference_l191_191643


namespace probability_equation_solution_l191_191263

theorem probability_equation_solution :
  (∃ x : ℝ, 3 * x ^ 2 - 8 * x + 5 = 0 ∧ 0 ≤ x ∧ x ≤ 1) → ∃ x = 1 :=
by
  sorry

end probability_equation_solution_l191_191263


namespace convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191895

-- Definitions from the problem
def polar_eq_C (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Conversion to rectangular coordinates
def rectangular_eq_C (x y : ℝ) : Prop := (x - sqrt 2)^2 + y^2 = 2

-- Definitions of points and vectors
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨1, 0⟩

def M (θ : ℝ) : Point :=
  let ρ := polar_eq_C θ
  ⟨ρ * cos θ, ρ * sin θ⟩

def P (x y : ℝ) : Point := ⟨x, y⟩

-- Conditions and transformations
def vector_eq (p1 p2 : Point) (a : ℝ) : Prop :=
  P p1.x p1.y = ⟨a * p2.x + (1 - a) * A.x, a * p2.y + (1 - a) * A.y⟩

def parametric_eq_P (θ : ℝ) : Point :=
  ⟨3 - sqrt 2 + 2 * cos θ, 2 * sin θ⟩

-- Proof problems
theorem convert_polar_to_rectangular (θ : ℝ) :
  rectangular_eq_C (polar_eq_C θ * cos θ) (polar_eq_C θ * sin θ) :=
sorry

theorem parametric_eq_valid (θ : ℝ) :
  vector_eq (parametric_eq_P θ) (M θ) (sqrt 2) :=
sorry

theorem no_common_points :
  ∀ θ₁ θ₂, parametric_eq_P θ₁ ≠ M θ₂ :=
sorry

end convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191895


namespace proof_problem_l191_191867

section Exercise

variable (θ : ℝ)

def polarCoordEq (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos θ

def toRectCoord (ρ θ : ℝ) : Prop := ρ^2 = (2 * Real.sqrt 2 * Real.cos θ) * ρ

def circleEq (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 2

def parametricLocusC1 (θ : ℝ) : (ℝ × ℝ) := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

theorem proof_problem :
  (∃ ρ θ, polarCoordEq ρ θ → circleEq (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  ∀ θ, ¬ ∃ x y, (circleEq x y) ∧ (parametricLocusC1 θ = (x, y)) :=
by 
  sorry

end Exercise

end proof_problem_l191_191867


namespace probability_of_product_multiple_of_80_l191_191015

namespace ProbabilityProblem

def S : Finset ℕ := {5, 10, 16, 20, 24, 40, 80}

def isMultipleOf80 (n : ℕ) : Prop := 80 ∣ n

def selectTwoDistinct (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.sup (λ a, (s.erase a).map (λ b, (a, b)))

theorem probability_of_product_multiple_of_80 :
  let pairs := selectTwoDistinct S in
  let valid_pairs := pairs.filter (λ (a, b), isMultipleOf80 (a * b)) in
  (valid_pairs.card : ℚ) / pairs.card = 5 / 21 := 
by 
  -- proof goes here
  sorry

end ProbabilityProblem

end probability_of_product_multiple_of_80_l191_191015


namespace marla_adds_white_paint_l191_191453

-- Define the conditions as hypotheses.
variables (total_percent blue_percent red_percent white_percent proportion_of_blue x : ℕ)
variable (total_ounces : ℕ)
hypothesis (H1 : total_percent = 100)
hypothesis (H2 : blue_percent = 70)
hypothesis (H3 : red_percent = 20)
hypothesis (H4 : white_percent = total_percent - blue_percent - red_percent)
hypothesis (H5 : total_ounces = 140)
hypothesis (H6 : blue_percent * x = white_percent * total_ounces)

-- The problem statement
theorem marla_adds_white_paint : 
  blue_percent * x = white_percent * total_ounces → 
  (x = 20)
:= sorry

end marla_adds_white_paint_l191_191453


namespace percentage_less_than_a_plus_d_l191_191609

def symmetric_distribution (a d : ℝ) (p : ℝ) : Prop :=
  p = (68 / 100 : ℝ) ∧ 
  (p / 2) = (34 / 100 : ℝ)

theorem percentage_less_than_a_plus_d (a d : ℝ) 
  (symmetry : symmetric_distribution a d (68 / 100)) : 
  (0.5 + (34 / 100) : ℝ) = (84 / 100 : ℝ) :=
by
  sorry

end percentage_less_than_a_plus_d_l191_191609


namespace distribution_plans_count_l191_191675

theorem distribution_plans_count (students units : ℕ) (at_least_one : ∀ (i : ℕ), i ≤ units) :
  students = 4 → units = 3 → at_least_one 4 → (∃ plans : ℕ, plans = 12) :=
by
  sorry

end distribution_plans_count_l191_191675


namespace wallpaper_expenditure_l191_191026

structure Room :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

def cost_per_square_meter : ℕ := 75

def total_expenditure (room : Room) : ℕ :=
  let perimeter := 2 * (room.length + room.width)
  let area_of_walls := perimeter * room.height
  let area_of_ceiling := room.length * room.width
  let total_area := area_of_walls + area_of_ceiling
  total_area * cost_per_square_meter

theorem wallpaper_expenditure (room : Room) : 
  room = Room.mk 30 25 10 →
  total_expenditure room = 138750 :=
by 
  intros h
  rw [h]
  sorry

end wallpaper_expenditure_l191_191026


namespace sum_of_squares_divisibility_l191_191668

theorem sum_of_squares_divisibility :
  let T := {t : ℤ | ∃ n : ℤ, t = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2} in
  (∀ t ∈ T, ¬(5 ∣ t)) ∧ (∃ t ∈ T, 7 ∣ t) :=
by
  sorry

end sum_of_squares_divisibility_l191_191668


namespace longest_frog_shortest_grasshopper_difference_l191_191027

-- Define the distances jumped by frogs and grasshoppers
def frog_distances : List ℕ := [39, 45, 50]
def grasshopper_distances : List ℕ := [17, 22, 28, 31]

-- Define the conclusion to be proven:
theorem longest_frog_shortest_grasshopper_difference :
  (List.maximum frog_distances) - (List.minimum grasshopper_distances) = 33 := by
  sorry

end longest_frog_shortest_grasshopper_difference_l191_191027


namespace polar_to_rectangular_eq_no_common_points_l191_191853

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_rectangular_eq (ρ θ x y: ℝ) (h1: ρ = 2 * Real.sqrt 2 * Real.cos θ) 
(h2: polar_to_rectangular ρ θ = (x, y)) : (x - Real.sqrt 2) ^ 2 + y ^ 2 = 2 :=
by
  sorry

theorem no_common_points (x y θ : ℝ) (ρ : ℝ := 2 * Real.sqrt 2 * Real.cos θ)
(A := (1, 0) : ℝ × ℝ)
(M := polar_to_rectangular ρ θ)
(P := (sqrt 2 * (M.1 - 1) + 1, sqrt 2 * M.2)) : 
((P.1 - (3 - Real.sqrt 2)) ^ 2 + P.2 ^ 2 = 4) → 
¬ ((P.1 - Real.sqrt 2) ^ 2 + P.2 ^ 2 = 2) :=
by
  intro H1 H2
  sorry

end polar_to_rectangular_eq_no_common_points_l191_191853


namespace max_value_sqrt_expr_max_reaches_at_zero_l191_191722

theorem max_value_sqrt_expr (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) : sqrt (36 + x) + sqrt (36 - x) ≤ 12 :=
by sorry

theorem max_reaches_at_zero : sqrt (36 + 0) + sqrt (36 - 0) = 12 :=
by sorry

end max_value_sqrt_expr_max_reaches_at_zero_l191_191722


namespace polynomial_root_l191_191945

-- Define polynomial type
variable {R : Type*} [CommRing R]

def seq (f : R → R) (n : ℕ) : ℕ → R
| 0       := 0
| (nat.succ 0) := f 0
| (nat.succ (nat.succ n)) := f (seq (nat.succ n))

theorem polynomial_root (f : R → R) (k : ℕ) 
  (hf : ∀ i : ℕ, f i ∈ set.univ)
  (hk : k ≥ 3)
  (h_sequence : seq f k = 0) :
  seq f 1 = 0 ∨ seq f 2 = 0 := 
sorry

end polynomial_root_l191_191945


namespace max_sum_of_a_seq_l191_191752

open Nat

noncomputable def a_seq : ℕ → ℝ
| 0     => 81
| (2 * k + 1) => 3^(a_seq (2 * k))
| (2 * k + 2) => -1 + log 3 (a_seq (2 * k + 1))

noncomputable def S_n (n : ℕ) : ℝ :=
(sum (Finset.range n) (λ i, a_seq i))

theorem max_sum_of_a_seq (n : ℕ) : S_n n ≤ 127 := 
sorry

end max_sum_of_a_seq_l191_191752


namespace proof_problem_l191_191907

noncomputable def polar_to_rectangular : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x - sqrt 2) ^ 2 + y ^ 2 = 2) ↔ (∀ θ, rho = 2 * sqrt 2 * Real.cos θ → (x ^ 2 + y ^ 2 = rho ^ 2) ∧ (x = rho * Real.cos θ)) sorry

noncomputable def check_common_points : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x = 3 - sqrt 2 + 2 * Real.cos θ) ∧ (y = 2 * Real.sin θ)) ∧
              ∀ (M_P C_P : ℝ), C_P = 2 ∧ |(3 - sqrt 2) - sqrt 2| = 3 - 2 * sqrt 2 < 2 - sqrt 2 :=
              ¬ ∃ p, (p ∈ C) ∧ (p ∈ C_1) sorry

theorem proof_problem : polar_to_rectangular ∧ check_common_points := by
  sorry

end proof_problem_l191_191907


namespace staff_discount_price_l191_191219

theorem staff_discount_price (d : ℝ) : (d - 0.15*d) * 0.90 = 0.765 * d :=
by
  have discount1 : d - 0.15 * d = d * 0.85 :=
    by ring
  have discount2 : (d * 0.85) * 0.90 = d * (0.85 * 0.90) :=
    by ring
  have final_price : d * (0.85 * 0.90) = d * 0.765 :=
    by norm_num
  rw [discount1, discount2, final_price]
  sorry

end staff_discount_price_l191_191219


namespace carrie_sends_80_texts_l191_191272

-- Definition of conditions
def texts_per_saturday := 5
def texts_per_sunday := 5
def texts_per_weekday := 2
def weekends_per_week := 2
def weekdays_per_week := 5
def weeks := 4

-- Statement of the theorem
theorem carrie_sends_80_texts :
  (weeks * ((texts_per_saturday + texts_per_sunday) * weekends_per_week / 2 + texts_per_weekday * weekdays_per_week)) = 80 := 
by simp [texts_per_saturday, texts_per_sunday, texts_per_weekday, weekends_per_week, weekdays_per_week, weeks]; sorry

end carrie_sends_80_texts_l191_191272


namespace initial_number_of_angelfish_l191_191670

/-- The initial number of fish in the tank. -/
def initial_total_fish (A : ℕ) := 94 + A + 89 + 58

/-- The remaining number of fish for each species after sale. -/
def remaining_fish (A : ℕ) := 64 + (A - 48) + 72 + 34

/-- Given: 
1. The total number of remaining fish in the tank is 198.
2. The initial number of fish for each species: 94 guppies, A angelfish, 89 tiger sharks, 58 Oscar fish.
3. The number of fish sold: 30 guppies, 48 angelfish, 17 tiger sharks, 24 Oscar fish.
Prove: The initial number of angelfish is 76. -/
theorem initial_number_of_angelfish (A : ℕ) (h : remaining_fish A = 198) : A = 76 :=
sorry

end initial_number_of_angelfish_l191_191670


namespace max_table_rows_l191_191316

open Function

-- Definitions
def is_permutation {α : Type*} [DecidableEq α] (s : Finset α) (f : Fin (s.card) → α) : Prop :=
  ∀ {a}, a ∈ s ↔ ∃ i, f i = a

def table_conditions (n : ℕ) (table : Fin n → Fin 9 → ℕ) : Prop :=
  ∀ row : Fin n,
    (Finset.univ.image (table row)).card = 9 ∧
    (∀ row2 : Fin n, row ≠ row2 → ∃ col, table row col = table row2 col)

-- Main theorem
theorem max_table_rows : ∃ n, table_conditions n (λ i j, ...) ∧ n = 8! := sorry

end max_table_rows_l191_191316


namespace shortest_distance_ad_l191_191837

-- Given conditions
variables {a b : ℝ} (h1 : a > b) (h2 : b > 0)

-- Angles in degrees (for understanding)
def angle_abc : ℝ := 120
def angle_bcd_min : ℝ := 0
def angle_bcd_max : ℝ := 60

-- Distances
def distance_ab : ℝ := a
def distance_bc : ℝ := a
def distance_cd : ℝ := b

-- Prove the shortest distance between points A and D
theorem shortest_distance_ad :
  ∃ d, d = real.sqrt (3*a^2 + b^2 - 2*real.sqrt(3)*a*b) :=
sorry

end shortest_distance_ad_l191_191837


namespace find_sum_placed_on_SI_l191_191313

noncomputable def simple_interest_sum (SI CI : ℝ) : ℝ := SI = CI / 2

noncomputable def compound_interest (P : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (λ acc rate → acc * (1 + rate)) P - P

-- Conditions extracted and defined
def principal : ℝ := 4000
def rate_changes : List ℝ := [0.05, 0.06, 0.07, 0.08] -- Equivalent 5%, 6%, 7%, 8%

def SI_rate_first_year : ℝ := 0.08
def SI_rate_second_year : ℝ := 0.12
def SI_rate_third_year : ℝ := 0.17

-- Calculated CI from conditions
def CI : ℝ := compound_interest principal rate_changes
def SI_factor : ℝ := 0.08 + 0.12 + 0.17

-- Statement to prove S = simple_interest_sum given the conditions and calculated results
theorem find_sum_placed_on_SI (S : ℝ) : 
  simple_interest_sum (S * SI_factor) CI → S = 1546.93 :=
sorry

end find_sum_placed_on_SI_l191_191313


namespace standard_deviation_of_applicants_l191_191118

theorem standard_deviation_of_applicants 
    (average_age : ℝ) (std_dev : ℝ) (max_diff_ages : ℕ) 
    (h_average_age : average_age = 31)
    (h_max_diff_ages : max_diff_ages = 17) 
    (h_within_std_dev : ∀ (age : ℝ), 
        (|age - average_age| ≤ std_dev) ↔ 
        (age = average_age - std_dev ∨ age = average_age + std_dev ∨ 
         average_age - std_dev < age ∧ age < average_age + std_dev)) :
    std_dev = 8 :=
begin
  sorry
end

end standard_deviation_of_applicants_l191_191118


namespace total_gold_cost_l191_191322

-- Given conditions
def gary_grams : ℕ := 30
def gary_cost_per_gram : ℕ := 15
def anna_grams : ℕ := 50
def anna_cost_per_gram : ℕ := 20

-- Theorem statement to prove
theorem total_gold_cost :
  (gary_grams * gary_cost_per_gram + anna_grams * anna_cost_per_gram) = 1450 := 
by
  sorry

end total_gold_cost_l191_191322


namespace variance_difference_l191_191992

noncomputable def D (X : ℝ) : ℝ := sorry
def independent (X Y : ℝ) : Prop := sorry

theorem variance_difference (X Y : ℝ) (h_ind : independent X Y) : D(X - Y) = D(X) + D(Y) := 
by 
  sorry

end variance_difference_l191_191992


namespace sum_of_first_15_terms_l191_191356

noncomputable theory

variable {a : ℕ → ℝ} -- geometric sequence
variable {b : ℕ → ℝ} -- arithmetic sequence

-- Conditions:
-- 1. a_n is a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

-- 2. b_n is an arithmetic sequence
def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

-- 3. a_2 * a_14 = 4 * a_8
def condition1 (a : ℕ → ℝ) : Prop :=
  a 2 * a 14 = 4 * a 8

-- 4. b_8 = a_8
def condition2 (a b : ℕ → ℝ) : Prop :=
  b 8 = a 8

-- Proving the sum of the first 15 terms of sequence b_n is 60
theorem sum_of_first_15_terms (a b : ℕ → ℝ) [is_geometric a] [is_arithmetic b]
  (h1 : condition1 a) (h2 : condition2 a b) :
  ∑ i in finset.range 15, b i = 60 :=
sorry

end sum_of_first_15_terms_l191_191356


namespace number_of_students_l191_191149

theorem number_of_students (x : ℕ) (total_cards : ℕ) (h : x * (x - 1) = total_cards) (h_total : total_cards = 182) : x = 14 :=
by
  sorry

end number_of_students_l191_191149


namespace min_value_of_Box_l191_191811

theorem min_value_of_Box (c d : ℤ) (hcd : c * d = 42) (distinct_values : c ≠ d ∧ c ≠ 85 ∧ d ≠ 85) :
  ∃ (Box : ℤ), (c^2 + d^2 = Box) ∧ (Box = 85) :=
by
  sorry

end min_value_of_Box_l191_191811


namespace midpoints_collinear_l191_191474

noncomputable def midpoint (P Q : ℝ^3) : ℝ^3 := (P + Q) / 2

theorem midpoints_collinear (A B C D M K : ℝ^3) (k : ℝ)
  (hM : M = (1 - k) • A + k • B)
  (hK : K = (1 - k) • D + k • C) :
  let P := midpoint A D,
      Q := midpoint B C,
      R := midpoint M K in
  ∃ λ : ℝ, R = (1 - λ) • P + λ • Q := 
sorry

end midpoints_collinear_l191_191474


namespace binomial_cos_sqrt_l191_191655

theorem binomial_cos_sqrt (z : ℝ) (hz : z ≥ 3) :
  (3 - cos (sqrt (z^2 - 9)))^2 = 9 - 6 * cos (sqrt (z^2 - 9)) + cos (sqrt (z^2 - 9))^2 :=
by
  sorry

end binomial_cos_sqrt_l191_191655


namespace sequence_property_l191_191339

variable (a : ℕ → ℝ)

theorem sequence_property (h : ∀ n : ℕ, 0 < a n) 
  (h_property : ∀ n : ℕ, (a n)^2 ≤ a n - a (n + 1)) :
  ∀ n : ℕ, a n < 1 / n :=
by
  sorry

end sequence_property_l191_191339


namespace ratio_c_div_d_l191_191599

theorem ratio_c_div_d (a b d : ℝ) (h1 : 8 = 0.02 * a) (h2 : 2 = 0.08 * b) (h3 : d = 0.05 * a) (c : ℝ) (h4 : c = b / a) : c / d = 1 / 320 := 
sorry

end ratio_c_div_d_l191_191599


namespace numNickels_proof_l191_191029

-- Definitions of the conditions
def numNickels := n
def numDimes := 2 * n
def numQuarters := numDimes / 2
def totalValue := (5 * numNickels + 10 * numDimes + 25 * numQuarters)

-- The main statement to prove
theorem numNickels_proof (n : ℕ) (h1 : numDimes = 2 * numNickels) 
  (h2 : numQuarters = numDimes / 2) (h3 : totalValue = 1950) : numNickels = 39 := by
  sorry

end numNickels_proof_l191_191029


namespace smallest_integer_with_conditions_l191_191512

theorem smallest_integer_with_conditions (x : ℕ) : 
  (∃ x, x.factors.count = 18 ∧ 18 ∣ x ∧ 24 ∣ x) → x = 972 :=
by
  sorry

end smallest_integer_with_conditions_l191_191512


namespace find_point_P_l191_191358

noncomputable def rotate_clockwise (θ : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 * Real.cos θ + v.2 * Real.sin θ, - v.1 * Real.sin θ + v.2 * Real.cos θ)

theorem find_point_P : 
  let A := (1 : ℝ, 2 : ℝ)
  let B := (1 + Real.sqrt 2, 2 - 2 * Real.sqrt 2)
  let θ := -(Real.pi / 4)
  let AB := (B.1 - A.1, B.2 - A.2)
  let AP := rotate_clockwise θ AB
  let P := (A.1 + AP.1, A.2 + AP.2)
  P = (0, -1) :=
by
  sorry

end find_point_P_l191_191358


namespace johny_total_travel_distance_l191_191050

def TravelDistanceSouth : ℕ := 40
def TravelDistanceEast : ℕ := TravelDistanceSouth + 20
def TravelDistanceNorth : ℕ := 2 * TravelDistanceEast
def TravelDistanceWest : ℕ := TravelDistanceNorth / 2

theorem johny_total_travel_distance
    (hSouth : TravelDistanceSouth = 40)
    (hEast  : TravelDistanceEast = 60)
    (hNorth : TravelDistanceNorth = 120)
    (hWest  : TravelDistanceWest = 60)
    (totalDistance : ℕ := TravelDistanceSouth + TravelDistanceEast + TravelDistanceNorth + TravelDistanceWest) :
    totalDistance = 280 := by
  sorry

end johny_total_travel_distance_l191_191050


namespace mr_william_land_percentage_l191_191195

theorem mr_william_land_percentage (total_tax : ℝ) (mr_william_tax : ℝ) (tax_rate : ℝ) (village_land : ℝ) : 
  total_tax = 5000 ∧ mr_william_tax = 480 ∧ tax_rate = 0.60 →
  (mr_william_tax / total_tax) * 100 = 5.76 :=
by
  intros h,
  rcases h with ⟨htotal, hwilliam, htax⟩,
  calc
    (mr_william_tax / total_tax) * 100 = (480 / 5000) * 100 : by rw [hwilliam, htotal]
    ... = 0.096 * 100 : by norm_num
    ... = 9.6 : by norm_num
    ... = 5.76 : by sorry

end mr_william_land_percentage_l191_191195


namespace units_digit_power_l191_191710

theorem units_digit_power : ∀ n : ℕ, (n % 2 = 0) → (n ≠ 0) → (nat.digits 10 (4^n)).head! = 6 :=
by
  intros n h1 h2
  sorry

end units_digit_power_l191_191710


namespace minPositivePeriod_eq_pi_and_maxValue_eq_1_l191_191513

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6) + cos (2 * x + π / 3)

theorem minPositivePeriod_eq_pi_and_maxValue_eq_1 :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π) ∧ (∀ x, f x ≤ 1) ∧ (∃ c, f c = 1) :=
by
  sorry

end minPositivePeriod_eq_pi_and_maxValue_eq_1_l191_191513


namespace event_d_is_certain_l191_191255

theorem event_d_is_certain : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ (3 * x^2 - 8 * x + 5 = 0) ∧ (x = 1) := 
by
  use 1
  split
  -- 0 ≤ x ∧ x ≤ 1 step
  split
  -- proof for 0 ≤ 1
  norm_num
  -- proof for 1 ≤ 1
  norm_num
  -- 3 * x^2 - 8 * x + 5 = 0 step
  split
  norm_num
  ring_nf
  -- x = 1 step
  rfl

end event_d_is_certain_l191_191255


namespace contradiction_proof_l191_191479

theorem contradiction_proof :
  ∀ (a b c d : ℝ),
    a + b = 1 →
    c + d = 1 →
    ac + bd > 1 →
    (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) →
    false := 
by
  intros a b c d h1 h2 h3 h4
  sorry

end contradiction_proof_l191_191479


namespace ones_prime_l191_191660

theorem ones_prime (k : ℕ) : 
  ∃ N : ℕ, (N = (10^(6 * k - 1) - 1) / 9) ∧ Nat.Prime N :=
begin
  sorry
end

end ones_prime_l191_191660


namespace loom_cloth_weaving_rate_l191_191244

-- Define the constants
def total_cloth : ℝ := 25
def total_time : ℝ := 195.3125
def cloth_per_second : ℝ := total_cloth / total_time

-- State the theorem to be proven
theorem loom_cloth_weaving_rate : cloth_per_second = 0.128 := by
  -- We will skip the proof with sorry
  sorry

end loom_cloth_weaving_rate_l191_191244


namespace number_of_ways_to_select_book_l191_191135

-- Definitions directly from the problem's conditions
def numMathBooks : Nat := 3
def numChineseBooks : Nat := 5
def numEnglishBooks : Nat := 8

-- The proof problem statement in Lean 4
theorem number_of_ways_to_select_book : numMathBooks + numChineseBooks + numEnglishBooks = 16 := 
by
  show 3 + 5 + 8 = 16
  sorry

end number_of_ways_to_select_book_l191_191135


namespace total_maximum_marks_combined_l191_191978

variables (marksMike marksSarah marksJohn shortfallMike shortfallSarah shortfallJohn : ℕ)
variables (passingPercentMike passingPercentSarah passingPercentJohn : ℝ)

def mike_total_marks (m s : ℕ) (p : ℝ) : ℝ := (m + s : ℝ) / p
def sarah_total_marks (m s : ℕ) (p : ℝ) : ℝ := (m + s : ℝ) / p
def john_total_marks (m s : ℕ) (p : ℝ) : ℝ := (m + s : ℝ) / p

theorem total_maximum_marks_combined :
  (mike_total_marks 212 16 0.30) + (sarah_total_marks 276 24 0.35) + (john_total_marks 345 30 0.40) = 2554.64 :=
by
  sorry

end total_maximum_marks_combined_l191_191978


namespace polar_to_rect_eq_and_locus_no_common_points_l191_191860

theorem polar_to_rect_eq_and_locus_no_common_points :
  let C := { (x, y) | (x - Real.sqrt 2)^2 + y^2 = 2 } in
  let C1 := { (x, y) | ∃ θ, x = 3 - Real.sqrt 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ } in
  (∀ (ρ θ : Real), ρ = 2 * Real.sqrt 2 * Real.cos θ → (Real.sqrt (ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos θ) - Real.sqrt 2)^2 + (Real.sin θ)^2 = 2 )
  ∧ ¬∃ (p : Real × Real), p ∈ C ∧ p ∈ C1 :=
by
  sorry

end polar_to_rect_eq_and_locus_no_common_points_l191_191860


namespace nested_radical_inequality_l191_191080

noncomputable def nested_radical_sequence (n : ℕ) : ℝ :=
match n with
| 0     => 0  -- Define x_0 for the base case, although it's not directly used in the problem.
| (n+1) => real.sqrt (2 + nested_radical_sequence n)

theorem nested_radical_inequality (n : ℕ) (hn : n > 0) :
  2 - nested_radical_sequence n > (2 - nested_radical_sequence (n - 1)) * (1 / 4) :=
sorry

end nested_radical_inequality_l191_191080


namespace proper_subset_f_l191_191437

noncomputable def h (x : ℚ) : ℕ := sorry  -- bijection from ℚ to ℕ 

def f (a : ℝ) : Set ℕ := 
  {n : ℕ | ∃ (x : ℚ), x < a ∧ h x = n}

theorem proper_subset_f {a b : ℝ} (h₀ : a < b) : f a ⊂ f b :=
by {
  -- Proof would go here
  sorry
}

end proper_subset_f_l191_191437


namespace find_f1_l191_191762

def f (x : ℝ) : ℝ := x^2 + 3 * x * f'(2)

theorem find_f1 : 1 + f'(1) = -3 := sorry

end find_f1_l191_191762


namespace oranges_required_for_profit_l191_191606

theorem oranges_required_for_profit :
  (let cost_price_per_orange := 15 / 4 in
   let selling_price_per_orange := 25 / 6 in
   let profit_per_orange := selling_price_per_orange - cost_price_per_orange in
   let oranges_count := 200 / profit_per_orange in
   ceil oranges_count = 477) :=
begin
  let cost_price_per_orange := 15 / 4,
  let selling_price_per_orange := 25 / 6,
  let profit_per_orange := selling_price_per_orange - cost_price_per_orange,
  let oranges_count := 200 / profit_per_orange,
  have h: ceil oranges_count = 477,
  sorry
end

end oranges_required_for_profit_l191_191606


namespace g_value_at_2023_l191_191956

noncomputable def g (x : ℝ) : ℝ :=
  sorry

theorem g_value_at_2023 (g : ℝ → ℝ) 
  (h1 : ∀ x > 0, g x > 0)
  (h2 : ∀ x y, x > y → y > 0 → g (x - y) = sqrt (g (x * y) + 3)) :
  g 2023 = 3 :=
sorry

end g_value_at_2023_l191_191956


namespace remainder_polynomial_division_l191_191180

theorem remainder_polynomial_division :
  ∀ (x : ℝ), (2 * x^2 - 21 * x + 55) % (x + 3) = 136 := 
sorry

end remainder_polynomial_division_l191_191180


namespace parabola_properties_l191_191528

-- Given conditions
variables (a b c : ℝ)
variable (h_vertex : ∃ a b c : ℝ, (∀ x, a * (x+1)^2 + 4 = ax^2 + b * x + c))
variable (h_intersection : ∃ A : ℝ, 2 < A ∧ A < 3 ∧ a * A^2 + b * A + c = 0)

-- Define the proof problem
theorem parabola_properties (h_vertex : (b = 2 * a)) (h_a : a < 0) (h_c : c = 4 + a) : 
  ∃ x : ℕ, x = 2 ∧ 
  (∀ a b c : ℝ, a * b * c < 0 → false) ∧ 
  (-4 < a ∧ a < -1 → false) ∧
  (a * c + 2 * b > 1 → false) :=
sorry

end parabola_properties_l191_191528


namespace median_of_set_is_88_l191_191136

noncomputable def mean (s : List ℝ) : ℝ :=
  s.sum / s.length

def isMedian (l : List ℝ) (m : ℝ) : Prop :=
  (l.sorted (≤)).nth (l.length / 2) = m

theorem median_of_set_is_88 (y : ℝ) (l : List ℝ)
  (hl : l = [91, 89, 85, 88, 90, 87, y])
  (mean_cond : mean l = 88) :
  isMedian l 88 :=
sorry

end median_of_set_is_88_l191_191136


namespace angle_BCA_eq_35_l191_191828

variables {A B C D M O : Type} [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry M] [EuclideanGeometry O]

-- Conditions
axiom angle_ABM_eq_55 : ∀ {A B M : Type}, ∠ A B M = 55
axiom angle_AMB_eq_70 : ∀ {A M B : Type}, ∠ A M B = 70
axiom angle_BOC_eq_80 : ∀ {B O C : Type}, ∠ B O C = 80
axiom angle_ADC_eq_60 : ∀ {A D C : Type}, ∠ A D C = 60
axiom midpoint_M : ∀ {A D M : Type}, midpoint A D M
axiom convex_quadrilateral_ABCD : ∀ {A B C D : Type}, convex_quadrilateral A B C D

-- Statement to prove
theorem angle_BCA_eq_35 : ∀ {A B C D M O : Type} [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry M] [EuclideanGeometry O],
  convex_quadrilateral A B C D →
  midpoint A D M →
  (∠ A B M = 55) →
  (∠ A M B = 70) →
  (∠ B O C = 80) →
  (∠ A D C = 60) →
  ∠ B C A = 35 :=
by
  intros
  sorry

end angle_BCA_eq_35_l191_191828


namespace polynomial_Q_exists_l191_191949

def P (x : ℝ) := (x - 1) * (x - 2) * (x - 3)

theorem polynomial_Q_exists (Q : ℝ → ℝ):
  (∃ R : ℝ → ℝ, degree (P(Q(x))) = degree (P(x) * R(x)) ∧ degree Q = 2) →
  (∃ n : ℕ, n = 22) :=
by 
  sorry

end polynomial_Q_exists_l191_191949


namespace geometric_sequence_problem_l191_191038

theorem geometric_sequence_problem (a : ℕ → ℤ)
  (q : ℤ)
  (h1 : a 2 * a 5 = -32)
  (h2 : a 3 + a 4 = 4)
  (hq : ∃ (k : ℤ), q = k) :
  a 9 = -256 := 
sorry

end geometric_sequence_problem_l191_191038


namespace polar_to_rectangular_eq_no_common_points_l191_191847

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_rectangular_eq (ρ θ x y: ℝ) (h1: ρ = 2 * Real.sqrt 2 * Real.cos θ) 
(h2: polar_to_rectangular ρ θ = (x, y)) : (x - Real.sqrt 2) ^ 2 + y ^ 2 = 2 :=
by
  sorry

theorem no_common_points (x y θ : ℝ) (ρ : ℝ := 2 * Real.sqrt 2 * Real.cos θ)
(A := (1, 0) : ℝ × ℝ)
(M := polar_to_rectangular ρ θ)
(P := (sqrt 2 * (M.1 - 1) + 1, sqrt 2 * M.2)) : 
((P.1 - (3 - Real.sqrt 2)) ^ 2 + P.2 ^ 2 = 4) → 
¬ ((P.1 - Real.sqrt 2) ^ 2 + P.2 ^ 2 = 2) :=
by
  intro H1 H2
  sorry

end polar_to_rectangular_eq_no_common_points_l191_191847


namespace power_function_through_point_l191_191014

theorem power_function_through_point (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^a) (h_point : f 2 = 8) :
  f = λ x, x^3 :=
by
  sorry

end power_function_through_point_l191_191014


namespace extra_coverage_calculation_l191_191684

/-- Define the conditions -/
def bag_coverage : ℕ := 500
def lawn_length : ℕ := 35
def lawn_width : ℕ := 48
def number_of_bags : ℕ := 6

/-- Define the main theorem to prove -/
theorem extra_coverage_calculation :
  number_of_bags * bag_coverage - (lawn_length * lawn_width) = 1320 := 
by
  sorry

end extra_coverage_calculation_l191_191684


namespace slices_left_per_person_is_2_l191_191097

variables (phil_slices andre_slices small_pizza_slices large_pizza_slices : ℕ)
variables (total_slices_eaten total_slices_left slices_per_person : ℕ)

-- Conditions
def conditions : Prop :=
  phil_slices = 9 ∧
  andre_slices = 9 ∧
  small_pizza_slices = 8 ∧
  large_pizza_slices = 14 ∧
  total_slices_eaten = phil_slices + andre_slices ∧
  total_slices_left = (small_pizza_slices + large_pizza_slices) - total_slices_eaten ∧
  slices_per_person = total_slices_left / 2

theorem slices_left_per_person_is_2 (h : conditions phil_slices andre_slices small_pizza_slices large_pizza_slices total_slices_eaten total_slices_left slices_per_person) :
  slices_per_person = 2 :=
sorry

end slices_left_per_person_is_2_l191_191097


namespace solve_fraction_problem_l191_191557

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l191_191557


namespace find_n_l191_191109

open Set

theorem find_n
  (A : Fin 6 → Set ℕ) (B : Fin n → Set ℕ)
  (hA1 : ∀ i, Fintype.card (A i) = 4)
  (hB1 : ∀ i, ∃ j, Fintype.card (B j) = 2)
  (S : Set ℕ) (hS1 : S = ⋃ i, A i) (hS2 : S = ⋃ j, B j)
  (hA2 : ∀ x ∈ S, {i : Fin 6 | x ∈ A i}.card = 4)
  (hB2 : ∀ x ∈ S, {j : Fin n | x ∈ B j}.card = 3): 
  n = 9 :=
by
  sorry

end find_n_l191_191109


namespace conjecture_sum_result_fixed_sum_result_general_sum_result_alternating_l191_191985

/-- Conjecture: proving the identity for 1 / (n * (n + 1)) -/
theorem conjecture (n : ℕ) (hn : n ≠ 0) : 
  1 / (n * (n + 1) : ℚ) = (1 / n : ℚ) - (1 / (n + 1) : ℚ) :=
sorry

/-- Sum result: proving the sum for ∑_{k=1}^{2023} 1 / (k * (k + 1)) -/
theorem sum_result_fixed : 
  (∑ k in Finset.range 2023 | λ k => 1 / ((k + 1) * (k + 2) : ℚ)) = 2023 / 2024 :=
sorry

/-- Sum result: proving the sum for ∑_{k=1}^{n} 1 / (k * (k + 1)) -/
theorem sum_result_general (n : ℕ) : 
  (∑ k in Finset.range n | λ k => 1 / ((k + 1) * (k + 2) : ℚ)) = n / (n + 1) :=
sorry

/-- Sum result: proving the sum for ∑_{k=1}^{1012} 1 / ((2k) * (2k + 2)) -/
theorem sum_result_alternating : 
  (∑ k in Finset.range 1012 | λ k => 1 / ((2 * (k + 1)) * (2 * (k + 1) + 2) : ℚ)) = 253 / 1013 :=
sorry

end conjecture_sum_result_fixed_sum_result_general_sum_result_alternating_l191_191985


namespace k_range_l191_191754

noncomputable def range_of_k (k : ℝ): Prop :=
  ∀ x : ℤ, (x - 2) * (x + 1) > 0 ∧ (2 * x + 5) * (x + k) < 0 → x = -2

theorem k_range:
  (∃ k : ℝ, range_of_k k) ↔ -3 ≤ k ∧ k < 2 :=
by
  sorry

end k_range_l191_191754


namespace franks_age_l191_191732

variable (F G : ℕ)

def gabriel_younger_than_frank : Prop := G = F - 3
def total_age_is_seventeen : Prop := F + G = 17

theorem franks_age (h1 : gabriel_younger_than_frank F G) (h2 : total_age_is_seventeen F G) : F = 10 :=
by
  sorry

end franks_age_l191_191732


namespace total_trees_correct_l191_191417

def apricot_trees : ℕ := 58
def peach_trees : ℕ := 3 * apricot_trees
def total_trees : ℕ := apricot_trees + peach_trees

theorem total_trees_correct : total_trees = 232 :=
by
  sorry

end total_trees_correct_l191_191417


namespace snail_distance_44_days_l191_191631

theorem snail_distance_44_days :
  (∑ n in Finset.range (44 + 1), (1 / (n + 1)) - (1 / (n + 2))) = 44 / 45 :=
by
  sorry

end snail_distance_44_days_l191_191631


namespace total_books_read_l191_191457

theorem total_books_read : 
  let Megan_books := 45 in
  let Kelcie_books := (1 / 3) * Megan_books in
  let John_books := Kelcie_books + 7 in
  let Greg_books := 2 * John_books + 11 in
  let Alice_books := 2.5 * Greg_books - 10 in
  Megan_books + Kelcie_books + John_books + Greg_books + Alice_books = 264 :=
by
  sorry

end total_books_read_l191_191457


namespace distance_between_trees_l191_191825

theorem distance_between_trees (l : ℕ) (n : ℕ) (d : ℕ) (h_length : l = 225) (h_trees : n = 26) (h_segments : n - 1 = 25) : d = 9 :=
sorry

end distance_between_trees_l191_191825


namespace texts_about_grocery_shopping_l191_191937

theorem texts_about_grocery_shopping (G : ℕ) (h1 : 5 * G = y) (h2 : 0.1 * (G + y) = z) (h3 : G + y + z = 33) : G = 5 :=
by
  sorry

end texts_about_grocery_shopping_l191_191937


namespace probability_top_card_is_star_l191_191625

-- Define the ranks and suits
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

inductive Suit
| Spades | Hearts | Diamonds | Clubs | Star | Moon

-- Define the card structure
structure Card :=
(rank : Rank) (suit : Suit)

-- Define the deck
def deck : List Card :=
(Suit.recOn (fun r => Rank.recOn r.list) []).bind (fun suit => Rank.recOn (fun r => [{rank := r, suit := suit}]) [])

-- The number of cards in the Star suit
def number_of_star_cards : ℕ := 13

-- The total number of cards
def total_number_of_cards : ℕ := 78

-- Define the probability
def probability_of_star : ℝ := (number_of_star_cards : ℝ) / (total_number_of_cards : ℝ)

-- The theorem to prove
theorem probability_top_card_is_star : probability_of_star = (1 / 6 : ℝ) := by
  sorry

end probability_top_card_is_star_l191_191625


namespace intersection_A_B_l191_191084

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end intersection_A_B_l191_191084


namespace real_part_of_z_squared_eq_one_l191_191399

noncomputable def real_part_of_z_squared (z : ℂ) : ℝ :=
  (z * z).re

theorem real_part_of_z_squared_eq_one (z : ℂ)
  (h1 : abs (z - conj z) = 2)
  (h2 : abs z * abs (conj z) = 3) :
  real_part_of_z_squared z = 1 :=
by
  sorry

end real_part_of_z_squared_eq_one_l191_191399


namespace donald_total_payment_l191_191679

theorem donald_total_payment :
  let original_price_laptop := 800
      discount_laptop := 0.15
      original_price_accessories := 200
      discount_accessories := 0.10
      reduced_price_laptop := original_price_laptop * (1 - discount_laptop)
      reduced_price_accessories := original_price_accessories * (1 - discount_accessories)
      total_payment := reduced_price_laptop + reduced_price_accessories
  in total_payment = 860 :=
by
  sorry

end donald_total_payment_l191_191679


namespace intersection_points_count_l191_191515

noncomputable def f (x : ℝ) : ℝ := 2 * log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem intersection_points_count : 
  ∃ (x1 x2 : ℝ), f x1 = g x1 ∧ f x2 = g x2 ∧ x1 ≠ x2 :=
sorry

end intersection_points_count_l191_191515


namespace integral_solutions_count_l191_191013

theorem integral_solutions_count (c : ℝ) (d : ℕ) 
  (h1 : c = 3 / 2) 
  (h2 : d = ∑ x in (Finset.range 7), if ∥(x : ℝ) / 2 - Real.sqrt 2∥ < c then 1 else 0) : d = 6 :=
sorry

end integral_solutions_count_l191_191013


namespace peters_workday_end_l191_191096

def hours_before_lunch : ℕ := 5
def lunch_duration : ℕ := 1
def total_daily_hours : ℕ := 9
def start_time_hour : ℕ := 8
def lunch_start_hour : ℕ := 13
def resume_time_hour := lunch_start_hour + lunch_duration
def worked_before_lunch := lunch_start_hour - start_time_hour
def remaining_work_hours := total_daily_hours - worked_before_lunch
def end_time_hour := resume_time_hour + remaining_work_hours

theorem peters_workday_end : end_time_hour = 18 := by
  unfold hours_before_lunch lunch_duration total_daily_hours start_time_hour lunch_start_hour resume_time_hour worked_before_lunch remaining_work_hours end_time_hour
  simp
  sorry

end peters_workday_end_l191_191096


namespace equivalent_single_discount_l191_191161

variable (x : ℝ)
variable (original_price : ℝ := x)
variable (discount1 : ℝ := 0.15)
variable (discount2 : ℝ := 0.10)
variable (discount3 : ℝ := 0.05)

theorem equivalent_single_discount :
  let final_price := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let equivalent_discount := (1 - final_price / original_price)
  equivalent_discount = 0.27 := 
by 
  sorry

end equivalent_single_discount_l191_191161


namespace add_to_frac_eq_l191_191567

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l191_191567


namespace largest_y_coordinate_l191_191667

theorem largest_y_coordinate (x y : ℝ) : 
  (x^2 / 49 + (y - 3)^2 / 25 = 0) → y = 3 :=
by
  intro h
  -- This is where the proofs steps would go if required.
  sorry

end largest_y_coordinate_l191_191667


namespace correct_equation_for_t_l191_191682

-- Define the rates for Doug and Dave
def dougRate : ℝ := 1 / 5
def daveRate : ℝ := 1 / 7

-- Combined rate
def combinedRate : ℝ := dougRate + daveRate

-- Theorem to prove the correct equation for time t
theorem correct_equation_for_t (t : ℝ) : combinedRate * (t - 1) = 1 :=
by
  -- Proof omitted, as only the statement is required
  sorry

end correct_equation_for_t_l191_191682


namespace f_monotonic_intervals_g_has_exactly_two_zeros_l191_191785

-- Part 1: Monotonic Intervals of f(x)
def f (x : ℝ) : ℝ := 4^(x + 1) - 2^x

theorem f_monotonic_intervals :
  (∀ x, x ≤ -3 → monotone_decreasing_on f set.Iic x) ∧
  (∀ x, x ≥ -3 → monotone_increasing_on f set.Ici x) :=
sorry

-- Part 2: Range of Real Number a
def g (x a : ℝ) : ℝ := f(x) - (3/16) * a^2 + (1/4) * a

theorem g_has_exactly_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ a = 0 ∧ g x₂ a = 0) ↔ (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 4/3) :=
sorry

end f_monotonic_intervals_g_has_exactly_two_zeros_l191_191785


namespace possible_remainders_of_b_l191_191066

-- Define the conditions
def congruent_mod (a b n : ℤ) : Prop := (a - b) % n = 0

variables {n : ℤ} (hn : n > 0)

theorem possible_remainders_of_b (n : ℤ) (hn : n > 0) :
  ∃ b : ℤ, b ∈ {8, 5, 3, 10} ∧ 
           (congruent_mod b (5^(2*n) + 6)⁻¹ 11 ∧
           ¬ congruent_mod (5^(2*n) + 6) 0 11) :=
by sorry

end possible_remainders_of_b_l191_191066


namespace sin_of_A_length_of_c_l191_191377

theorem sin_of_A (a b : ℝ) (B : ℝ) (h_a : a = 3) (h_b : b = 5) (h_B : B = real.pi * (2 / 3)) : 
  real.sin (real.arcsin (3 * real.sin (real.arcsin (B / 2)) / 5)) = (3 * real.sqrt 3) / 10 :=
sorry

theorem length_of_c (a b : ℝ) (C : ℝ) (h_a : a = 3) (h_b : b = 5) (h_C : C = real.pi * (2 / 3)) :
  real.sqrt (a^2 + b^2 - 2 * a * b * real.cos C) = 7 :=
sorry

end sin_of_A_length_of_c_l191_191377


namespace ellipse_equation_length_MN_l191_191083

-- Definitions based on given conditions
def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | let ⟨x, y⟩ := p in (x^2 / a^2 + y^2 / b^2 = 1)}

def circle (x y : ℝ) : set (ℝ × ℝ) :=
  {p | let ⟨x', y'⟩ := p in (x' - x)^2 + (y' - y)^2 = 2}

-- Prove the given problem
theorem ellipse_equation (a b c : ℝ)
  (h1 : a > b > 0)
  (h2 : b = 2)
  (h3 : c = 2)
  (h4 : a^2 = b^2 + c^2) :
  ellipse a b = {p | let ⟨x, y⟩ := p in (x^2 / 8 + y^2 / 4 = 1)} :=
sorry

theorem length_MN (M N C : ℝ × ℝ)
  (h1 : C = (1, 1))
  (h2 : C = ((M.1 + N.1) / 2, (M.2 + N.2) / 2))
  (h3 : M ∈ ellipse (2 * sqrt 2) 2)
  (h4 : N ∈ ellipse (2 * sqrt 2) 2) :
  dist M N = 5 * sqrt 6 / 3 :=
sorry

end ellipse_equation_length_MN_l191_191083


namespace domain_of_f_f_is_odd_l191_191370

-- Define the function f(x)
def f (x : ℝ) : ℝ := log (3, (1 - x) / (1 + x))

-- State the required properties
theorem domain_of_f : ∀ x : ℝ, (1 - x) / (1 + x) > 0 ↔ -1 < x ∧ x < 1 := sorry

theorem f_is_odd : ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x := sorry

end domain_of_f_f_is_odd_l191_191370


namespace max_sqrt_expr_eq_12_l191_191712

noncomputable def max_value_sqrt_expr : ℝ :=
  real.sup (set.image (λ x : ℝ, real.sqrt (36 + x) + real.sqrt (36 - x)) (set.Icc (-36) 36))

theorem max_sqrt_expr_eq_12 : max_value_sqrt_expr = 12 := by
  sorry

end max_sqrt_expr_eq_12_l191_191712


namespace culture_medium_preparation_l191_191021

theorem culture_medium_preparation :
  ∀ (V : ℝ), 0 < V → 
  ∃ (nutrient_broth pure_water saline_water : ℝ),
    nutrient_broth = V / 3 ∧
    pure_water = V * 0.3 ∧
    saline_water = V - (nutrient_broth + pure_water) :=
by
  sorry

end culture_medium_preparation_l191_191021


namespace kiril_konstantinovich_age_is_full_years_l191_191943

theorem kiril_konstantinovich_age_is_full_years
  (years : ℕ)
  (months : ℕ)
  (weeks : ℕ)
  (days : ℕ)
  (hours : ℕ) :
  (years = 48) →
  (months = 48) →
  (weeks = 48) →
  (days = 48) →
  (hours = 48) →
  Int.floor (
    years + 
    (months / 12 : ℝ) + 
    (weeks * 7 / 365 : ℝ) + 
    (days / 365 : ℝ) + 
    (hours / (24 * 365) : ℝ)
  ) = 53 :=
by
  intro hyears hmonths hweeks hdays hhours
  rw [hyears, hmonths, hweeks, hdays, hhours]
  sorry

end kiril_konstantinovich_age_is_full_years_l191_191943


namespace option_B_not_acceptable_weight_l191_191220

def labeled_weight := 10
def tolerance := 0.5
def acceptable_range := (labeled_weight - tolerance, labeled_weight + tolerance)
def weight_option_B := 10.7

theorem option_B_not_acceptable_weight :
  ¬ (acceptable_range.1 ≤ weight_option_B ∧ weight_option_B ≤ acceptable_range.2) :=
by
  sorry

end option_B_not_acceptable_weight_l191_191220


namespace polar_to_rectangular_eq_no_common_points_l191_191852

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_rectangular_eq (ρ θ x y: ℝ) (h1: ρ = 2 * Real.sqrt 2 * Real.cos θ) 
(h2: polar_to_rectangular ρ θ = (x, y)) : (x - Real.sqrt 2) ^ 2 + y ^ 2 = 2 :=
by
  sorry

theorem no_common_points (x y θ : ℝ) (ρ : ℝ := 2 * Real.sqrt 2 * Real.cos θ)
(A := (1, 0) : ℝ × ℝ)
(M := polar_to_rectangular ρ θ)
(P := (sqrt 2 * (M.1 - 1) + 1, sqrt 2 * M.2)) : 
((P.1 - (3 - Real.sqrt 2)) ^ 2 + P.2 ^ 2 = 4) → 
¬ ((P.1 - Real.sqrt 2) ^ 2 + P.2 ^ 2 = 2) :=
by
  intro H1 H2
  sorry

end polar_to_rectangular_eq_no_common_points_l191_191852


namespace pyramid_volume_l191_191055

noncomputable def volume_of_pyramid 
  (ABCD : Type) 
  (rectangle : ABCD) 
  (DM_perpendicular : Prop) 
  (MA MC MB : ℕ) 
  (lengths : MA = 11 ∧ MC = 13 ∧ MB = 15 ∧ DM = 5) : ℝ :=
  80 * Real.sqrt 6

theorem pyramid_volume (ABCD : Type) 
    (rectangle : ABCD) 
    (DM_perpendicular : Prop) 
    (MA MC MB DM : ℕ)
    (lengths : MA = 11 ∧ MC = 13 ∧ MB = 15 ∧ DM = 5) 
  : volume_of_pyramid ABCD rectangle DM_perpendicular MA MC MB lengths = 80 * Real.sqrt 6 :=
  by {
    sorry
  }

end pyramid_volume_l191_191055


namespace proof_tan_alpha_l191_191081

open Real

-- Definitions for the conditions in the problem
def square_pyramid (AB CD : ℝ) (height : ℝ) := 
  ∃ (A₀ B₀ C₀ D₀ E₀ : ℝ×ℝ×ℝ),
    ∃ (s : ℝ), 
      A₀ = (0, 0, 0) ∧ 
      B₀ = (s, 0, 0) ∧
      C₀ = (s, s, 0) ∧
      D₀ = (0, s, 0) ∧
      E₀ = (s/2, s/2, height) ∧
      s = AB ∧ AB = CD

def find_tan_alpha (pyramid : Type) :=
  ∃ (α : ℝ), 0 < α ∧ α < π/2 ∧ tan α = 17 / 144

-- The conditions defined as Lean definitions
def conditions := 
  square_pyramid 12 12 (1/2)

-- The theorem to be proven given the conditions
theorem proof_tan_alpha : conditions → find_tan_alpha (square_pyramid 12 12 (1/2)) :=
by { sorry }

end proof_tan_alpha_l191_191081


namespace polar_to_rect_eq_and_locus_no_common_points_l191_191858

theorem polar_to_rect_eq_and_locus_no_common_points :
  let C := { (x, y) | (x - Real.sqrt 2)^2 + y^2 = 2 } in
  let C1 := { (x, y) | ∃ θ, x = 3 - Real.sqrt 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ } in
  (∀ (ρ θ : Real), ρ = 2 * Real.sqrt 2 * Real.cos θ → (Real.sqrt (ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos θ) - Real.sqrt 2)^2 + (Real.sin θ)^2 = 2 )
  ∧ ¬∃ (p : Real × Real), p ∈ C ∧ p ∈ C1 :=
by
  sorry

end polar_to_rect_eq_and_locus_no_common_points_l191_191858


namespace symmetry_proof_l191_191964

structure Point := (x y : ℝ)
structure Symmetric_point (A O : Point) : Point := (x := 2 * O.x - A.x) (y := 2 * O.y - A.y)

theorem symmetry_proof (A O1 O2 O3 : Point) :
  let A1 := Symmetric_point A O1
  let A2 := Symmetric_point A1 O2
  let A4 := Symmetric_point A O1
  let A5 := Symmetric_point A4 O2
  let A6 := Symmetric_point A5 O3
  A6 = A :=
by
  sorry

end symmetry_proof_l191_191964


namespace sum_of_cubes_pattern_l191_191463

theorem sum_of_cubes_pattern :
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2) :=
by
  sorry

end sum_of_cubes_pattern_l191_191463


namespace franks_age_l191_191731

variable (F G : ℕ)

def gabriel_younger_than_frank : Prop := G = F - 3
def total_age_is_seventeen : Prop := F + G = 17

theorem franks_age (h1 : gabriel_younger_than_frank F G) (h2 : total_age_is_seventeen F G) : F = 10 :=
by
  sorry

end franks_age_l191_191731


namespace weight_of_bowling_ball_l191_191983

-- Define weights of bowling ball and canoe
variable (b c : ℚ)

-- Problem conditions
def cond1 : Prop := (9 * b = 5 * c)
def cond2 : Prop := (4 * c = 120)

-- The statement to prove
theorem weight_of_bowling_ball (h1 : cond1 b c) (h2 : cond2 c) : b = 50 / 3 := sorry

end weight_of_bowling_ball_l191_191983


namespace two_digit_product_digits_l191_191139

theorem two_digit_product_digits : 
  ∃ n, (n = 3 ∨ n = 4) ∧ ∀ a b, (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) → nat.digits 10 (a * b) = n := 
by
  sorry

end two_digit_product_digits_l191_191139


namespace circle_intersection_range_l191_191008

theorem circle_intersection_range (r : ℝ) (H : r > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ (x+3)^2 + (y-4)^2 = 36) → (1 < r ∧ r < 11) := 
by
  sorry

end circle_intersection_range_l191_191008


namespace find_k_l191_191445

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem find_k (x_0 : ℝ) (k : ℤ) (h_eq : f x_0 = 0) (h_int : x_0 ∈ Ioo (k:ℝ) (k+1)) : k = 2 :=
by
  sorry

end find_k_l191_191445


namespace F_expression_g_monotonic_l191_191749

-- Definitions corresponding to conditions
def f (a b x : ℝ) := a * x^2 + b * x + 1
def F (a b x : ℝ) := if x > 0 then f a b x else - (f a b x)
def g (a b k x : ℝ) := f a b x - k * x

-- Condition that a > 0
variable (a : ℝ) (b : ℝ) (k : ℝ) (x : ℝ)
variable (h_a_pos : 0 < a)
variable (h_f_neg1_zero : f a b (-1) = 0)
variable (h_f_nonneg : ∀ x : ℝ, 0 ≤ f a b x)

-- Theorem corresponding to question 1
theorem F_expression : F a b x = (if x > 0 then (x + 1)^2 else -(x + 1)^2) :=
sorry

-- Definition for monotonicity
def monotonic_on (f : ℝ → ℝ) (I : Set ℝ) := ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

-- Theorem corresponding to question 2
theorem g_monotonic : monotonic_on (g a b k) (Set.Icc (-2) 2) ↔ k ≥ 6 ∨ k ≤ -2 :=
sorry

end F_expression_g_monotonic_l191_191749


namespace no_prime_satisfies_condition_l191_191459

theorem no_prime_satisfies_condition :
  ¬ ∃ p : ℕ, p > 1 ∧ 10 * (p : ℝ) = (p : ℝ) + 5.4 := by {
  sorry
}

end no_prime_satisfies_condition_l191_191459


namespace conic_section_eccentricity_l191_191365

theorem conic_section_eccentricity (x y : ℝ) (h : 10 * x - 2 * x * y - 2 * y + 1 = 0) : 
  eccentricity x y = real.sqrt 2 :=
sorry

end conic_section_eccentricity_l191_191365


namespace max_sides_polygon_is_7_l191_191402

noncomputable def max_sides_convex_polygon_with_obtuse_and_right_angles (n : ℕ) : Prop :=
  ∃ (o1 o2 o3 o4 r1 r2 : ℝ) (a : fin (n - 6) → ℝ),
    (0 < o1) ∧ (o1 < 180) ∧ 
    (0 < o2) ∧ (o2 < 180) ∧ 
    (0 < o3) ∧ (o3 < 180) ∧ 
    (0 < o4) ∧ (o4 < 180) ∧ 
    (o1 > 90) ∧ (o2 > 90) ∧ (o3 > 90) ∧ (o4 > 90) ∧
    (r1 = 90) ∧ (r2 = 90) ∧
    (∀ i, (0 < a i) ∧ (a i < 90)) ∧ 
    (270 < o1 + o2 + o3 + o4) ∧  (o1 + o2 + o3 + o4 < 720) ∧
    (o1 + o2 + o3 + o4 + r1 + r2 + (∑ i, a i) = 180 * (n - 2))

theorem max_sides_polygon_is_7 : ∀ n, max_sides_convex_polygon_with_obtuse_and_right_angles n → n = 7 := sorry

end max_sides_polygon_is_7_l191_191402


namespace min_internal_fence_length_l191_191618

-- Setup the given conditions in Lean 4
def total_land_area (length width : ℕ) : ℕ := length * width

def sotkas_to_m2 (sotkas : ℕ) : ℕ := sotkas * 100

-- Assume a father had three sons and left them an inheritance of land
def land_inheritance := 9 -- in sotkas

-- The dimensions of the land
def length := 25 
def width := 36

-- Prove that:
theorem min_internal_fence_length :
  ∃ (ways : ℕ) (min_length : ℕ),
    total_land_area length width = sotkas_to_m2 land_inheritance ∧
    (∀ (l1 l2 l3 w1 w2 w3 : ℕ),
      l1 * w1 = sotkas_to_m2 3 ∧ l2 * w2 = sotkas_to_m2 3 ∧ l3 * w3 = sotkas_to_m2 3 →
      ways = 4 ∧ min_length = 49) :=
by
  sorry

end min_internal_fence_length_l191_191618


namespace n_eq_b_eq_sixteen_l191_191589

theorem n_eq_b_eq_sixteen (n b : ℝ) (h₁ : n = 2^0.25) (h₂ : n^b = 16) : b = 16 := by
  sorry

end n_eq_b_eq_sixteen_l191_191589


namespace train_passing_time_approx_l191_191384

def speed_first_train : ℝ := 78 -- kmph
def speed_second_train : ℝ := 65 -- kmph
def length_first_train : ℝ := 110 -- meters
def length_second_train : ℝ := 85 -- meters

def relative_speed : ℝ := (speed_first_train + speed_second_train) * (1000 / 3600) -- convert kmph to m/s

def total_length : ℝ := length_first_train + length_second_train

def passing_time : ℝ := total_length / relative_speed

theorem train_passing_time_approx :
  abs (passing_time - 4.91) < 0.01 :=
by
  trivial
  -- Given that the precise computations are performed as described in the solution steps,
  -- the result passing_time would be ≈ 4.91 seconds.

end train_passing_time_approx_l191_191384


namespace apogee_reach_second_stage_model_engine_off_time_l191_191603

-- Given conditions
def altitudes := [(0, 0), (1, 24), (2, 96), (4, 386), (5, 514), (6, 616), (9, 850), (13, 994), (14, 1000), (16, 976), (19, 850), (24, 400)]
def second_stage_curve (x : ℝ) : ℝ := -6 * x^2 + 168 * x - 176

-- Proof problems
theorem apogee_reach : (14, 1000) ∈ altitudes :=
sorry  -- Need to prove the inclusion of the apogee point in the table

theorem second_stage_model : 
    second_stage_curve 14 = 1000 ∧ 
    second_stage_curve 16 = 976 ∧ 
    second_stage_curve 19 = 850 ∧ 
    ∃ n, n = 4 :=
sorry  -- Need to prove the analytical expression is correct and n = 4

theorem engine_off_time : 
    ∃ t : ℝ, t = 14 + 5 * Real.sqrt 6 ∧ second_stage_curve t = 100 :=
sorry  -- Need to prove the engine off time calculation

end apogee_reach_second_stage_model_engine_off_time_l191_191603


namespace total_trees_l191_191418

theorem total_trees (apricot_trees : ℕ) (peach_mult : ℕ) (peach_trees : ℕ) (total_trees : ℕ) :
  apricot_trees = 58 → peach_mult = 3 → peach_trees = peach_mult * apricot_trees → total_trees = apricot_trees + peach_trees → total_trees = 232 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw h3 at h4
  rw h4
  exact rfl

end total_trees_l191_191418


namespace white_paint_amount_is_correct_l191_191454

noncomputable def totalAmountOfPaint (bluePaint: ℝ) (bluePercentage: ℝ): ℝ :=
  bluePaint / bluePercentage

noncomputable def whitePaintAmount (totalPaint: ℝ) (whitePercentage: ℝ): ℝ :=
  totalPaint * whitePercentage

theorem white_paint_amount_is_correct (bluePaint: ℝ) (bluePercentage: ℝ) (whitePercentage: ℝ) (totalPaint: ℝ) :
  bluePaint = 140 → bluePercentage = 0.7 → whitePercentage = 0.1 → totalPaint = totalAmountOfPaint 140 0.7 →
  whitePaintAmount totalPaint 0.1 = 20 :=
by
  intros
  sorry

end white_paint_amount_is_correct_l191_191454


namespace max_value_sqrt_expression_l191_191725

theorem max_value_sqrt_expression : 
  ∀ (x : ℝ), -36 ≤ x ∧ x ≤ 36 → 
  sqrt (36 + x) + sqrt (36 - x) ≤ 12 :=
by
  -- Proof goes here
  sorry

end max_value_sqrt_expression_l191_191725


namespace area_of_tangent_segments_annulus_l191_191742

noncomputable def circle := {center : ℝ × ℝ // center = (0, 0)}
def radius := 3
def segment_length := 3

def internalSegmentArea {R r : ℝ} (annulus_inner_radius : ℝ) (annulus_outer_radius : ℝ) : ℝ :=
  π * (annulus_outer_radius ^ 2 - annulus_inner_radius ^ 2)

theorem area_of_tangent_segments_annulus 
  (c : circle) 
  (r : ℝ) 
  (l : ℝ) 
  (h_radius : r = radius)
  (h_length : l = segment_length) : 
  internalSegmentArea 3 (3 * real.sqrt 5 / 2) = 9 * π / 4 :=
by
  sorry

end area_of_tangent_segments_annulus_l191_191742


namespace max_cosA_sin2B_cosC_l191_191241

theorem max_cosA_sin2B_cosC :
  ∀ (A B C : ℝ), A + B + C = π → 
  (cos A + (1 - cos B ^ 2) * -(cos A)) ≤ 1 := 
by
  sorry

end max_cosA_sin2B_cosC_l191_191241


namespace baker_bought_131_new_cakes_l191_191267

def number_of_new_cakes_bought (initial_cakes: ℕ) (cakes_sold: ℕ) (excess_sold: ℕ): ℕ :=
    cakes_sold - excess_sold - initial_cakes

theorem baker_bought_131_new_cakes :
    number_of_new_cakes_bought 8 145 6 = 131 :=
by
  -- This is where the proof would normally go
  sorry

end baker_bought_131_new_cakes_l191_191267


namespace train_cross_bridge_time_30_seconds_l191_191636

noncomputable def train_crossing_time
  (length_of_train : ℕ) -- 110 meters
  (speed_of_train_kmph : ℕ) -- 45 km/hr
  (length_of_bridge : ℕ) -- 265 meters
  : ℕ :=
  let total_distance := length_of_train + length_of_bridge in
  let speed_of_train_mps := (speed_of_train_kmph * 1000) / 3600 in
  total_distance / speed_of_train_mps

theorem train_cross_bridge_time_30_seconds :
  train_crossing_time 110 45 265 = 30 :=
  sorry

end train_cross_bridge_time_30_seconds_l191_191636


namespace polynomial_zero_or_one_root_l191_191593

theorem polynomial_zero_or_one_root (p : Polynomial ℤ) 
  (h : ∀ m n : ℕ, ∃ a : ℤ, n ∣ p.eval (a ^ m)) : 
  p.eval 0 = 0 ∨ p.eval 1 = 0 := 
begin
  sorry
end

end polynomial_zero_or_one_root_l191_191593


namespace A_partitionable_iff_l191_191054

def is_partitionable (n k : ℕ) (A : Finset ℕ) : Prop :=
  ∃ parts : Finset (Finset ℕ), parts.card = k ∧ 
    (∀ part ∈ parts, part.sum id = A.sum id / k) ∧
    (∀ part1 part2 ∈ parts, part1 ≠ part2 → part1 ∩ part2 = ∅)

theorem A_partitionable_iff (n k : ℕ) (h_n : 1 ≤ n) : 
  let A := Finset.range (n + 1) in
  is_partitionable n k A ↔ (2 * k ∣ n * (n + 1) ∧ 2 * k ≤ n + 1) :=
by {
  sorry
}

end A_partitionable_iff_l191_191054


namespace number_of_nozzles_approx_6_l191_191497

theorem number_of_nozzles_approx_6 :
  let P := 0.35
  let K := 24.96
  let S := 14
  let W := 20
  let q := K * Real.sqrt (10 * P)
  let N := (S * W) / q
  N ≈ 6 := by
  intros
  let P := 0.35
  let K := 24.96
  let S := 14
  let W := 20
  let q := K * Real.sqrt (10 * P)
  let N := (S * W) / q
  sorry

end number_of_nozzles_approx_6_l191_191497


namespace mary_classes_l191_191090

theorem mary_classes (c : ℕ) (hf : ℕ) (hp : ℕ) (he : ℕ) (hpaints : ℕ) (Hfolders : ℕ) (Hpencils : ℕ) (Herasers : ℕ) (Hspent : ℕ) (Hpaints : ℕ) (hf_cost : hf = 6 * c) (hp_cost : hp = 2 * 3 * c) (he_needed : he = c / 2) (he_cost : he = he_needed) (cost_eq : 6 * c + 2 * 3 * c + he = 75)
: c = 6 := 
by
  sorry

end mary_classes_l191_191090


namespace sum_lcm_not_pow_two_l191_191990

-- Define the problem conditions
variable (l : List ℕ) [decidable_eq ℕ]
variable (red blue : List ℕ)

-- Condition 1: l is a list of consecutive natural numbers
def consecutive (l : List ℕ) : Prop :=
  ∀ (i : ℕ), i < l.length - 1 → l.nth i + 1 = l.nth (i + 1)

-- Condition 2: red and blue are subsets of l
def is_sublist (sub l : List ℕ) : Prop :=
  ∀ (x : ℕ), x ∈ sub → x ∈ l

-- Condition 3: both red and blue are present
def nonempty (l : List ℕ) : Prop := l ≠ []

-- Least common multiple
def lcm_list (l : List ℕ) : ℕ :=
  list.foldr lcm 1 l

-- Theorem to prove the given question.
theorem sum_lcm_not_pow_two (hn : nonempty red) (hb : nonempty blue) 
  (hc : consecutive l) (hr : is_sublist red l) (hb : is_sublist blue l) : 
  ¬ ∃ k : ℕ, 2^k = lcm_list red + lcm_list blue := 
sorry

end sum_lcm_not_pow_two_l191_191990


namespace tissues_per_pack_calculation_l191_191647

-- Conditions
def number_of_boxes := 10
def cost_of_boxes := 1000
def packs_per_box := 20
def cost_per_tissue := 0.05
def cost_per_pack := 5

-- Question and proof problem
theorem tissues_per_pack_calculation : 
  (cost_of_boxes / number_of_boxes / packs_per_box / cost_per_tissue) = 100 := 
by sorry

end tissues_per_pack_calculation_l191_191647


namespace find_root_l191_191250

theorem find_root :
  ∃ x : ℝ, (3 * x^2 - 8 * x + 5 = 0) ∧ (x = 1) :=
begin
  sorry
end

end find_root_l191_191250


namespace expected_value_of_coins_heads_l191_191226

noncomputable def penny_value := 1
noncomputable def nickel_value := 5
noncomputable def quarter_value := 25
noncomputable def halfdollar_value := 50

noncomputable def expected_value_heads :=
  (1 / 2.0) * penny_value + 
  (1 / 2.0) * nickel_value + 
  (1 / 2.0) * quarter_value + 
  (1 / 2.0) * halfdollar_value

theorem expected_value_of_coins_heads : expected_value_heads = 40.5 := 
by
  sorry

end expected_value_of_coins_heads_l191_191226


namespace milk_poured_l191_191678

theorem milk_poured (dons_milk : ℚ) (poured_fraction : ℚ) : 
  dons_milk = 3/7 → poured_fraction = 5/8 → poured_fraction * dons_milk = 15/56 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end milk_poured_l191_191678


namespace find_number_to_add_l191_191573

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l191_191573


namespace max_distance_ellipse_to_fixed_point_l191_191779

theorem max_distance_ellipse_to_fixed_point :
  ∃ (a : ℝ) (e : ℝ) (P B : ℝ × ℝ),
    a = sqrt 5 ∧ e = 2 * sqrt 5 / 5 ∧ 
    (∀ θ, P = (cos θ, sqrt 5 * sin θ)) ∧ B = (-1, 0) ∧
    ∀ d, d = sqrt ((cos θ + 1)^2 + 5 * sin θ^2) → 
    ∃ d_max, d_max = 5 / 2 := sorry

end max_distance_ellipse_to_fixed_point_l191_191779


namespace circumcenter_locus_circle_centroid_locus_circle_orthocenter_locus_circle_l191_191071

variable {α : Type*}
variable [MetricSpace α]

/- Conditions -/
variable (circle1 circle2 : Circle α)
variable (A : α)
variable (lineThroughOtherIntersection : Line α)
variable (B C : α)
variable (HA : IsIntersectionPoint circle1 circle2 A)
variable (HB : lineThroughOtherIntersection ∈ CirclePointsExcept circle1 A B)
variable (HC : lineThroughOtherIntersection ∉ CirclePointsExcept circle2 A C)

/- Definitions -/

noncomputable def circumcenter_locus (A B C : α) : Set α :=
{ center | ∃ Δ : Triangle, Δ.A = A ∧ Δ.B = B ∧ Δ.C = C ∧ center = CenterOfCircumcircle Δ }

noncomputable def centroid_locus (A B C : α) : Set α :=
{ centroid | ∃ Δ : Triangle, Δ.A = A ∧ Δ.B = B ∧ Δ.C = C ∧ centroid = Centroid Δ }

noncomputable def orthocenter_locus (A B C : α) : Set α :=
{ orthocenter | ∃ Δ : Triangle, Δ.A = A ∧ Δ.B = B ∧ Δ.C = C ∧ orthocenter = Orthocenter Δ }

/- Lean 4 Statements -/

theorem circumcenter_locus_circle (A B C : α)
  (HA : IsIntersectionPoint circle1 circle2 A)
  (HB : lineThroughOtherIntersection ∈ CirclePointsExcept circle1 A B)
  (HC : lineThroughOtherIntersection ∉ CirclePointsExcept circle2 A C) :
  ∃ r : ℝ, is_circle A r (circumcenter_locus A B C) :=
sorry

theorem centroid_locus_circle (A B C : α)
  (HA : IsIntersectionPoint circle1 circle2 A)
  (HB : lineThroughOtherIntersection ∈ CirclePointsExcept circle1 A B)
  (HC : lineThroughOtherIntersection ∉ CirclePointsExcept circle2 A C) :
  ∃ r : ℝ, is_circle A r (centroid_locus A B C) :=
sorry

theorem orthocenter_locus_circle (A B C : α)
  (HA : IsIntersectionPoint circle1 circle2 A)
  (HB : lineThroughOtherIntersection ∈ CirclePointsExcept circle1 A B)
  (HC : lineThroughOtherIntersection ∉ CirclePointsExcept circle2 A C) :
  ∃ r : ℝ, is_circle A r (orthocenter_locus A B C) :=
sorry

end circumcenter_locus_circle_centroid_locus_circle_orthocenter_locus_circle_l191_191071


namespace convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191877

theorem convert_polar_to_rectangular (ρ θ x y : ℝ) (hρ : ρ = 2 * √2 * cos θ) :
  (x - √2) ^ 2 + y ^ 2 = 2 :=
sorry

theorem parametric_equations_and_no_common_points (x y θ : ℝ) :
  let A := (1 : ℝ, 0 : ℝ)
  let P (θ : ℝ) := (3 - √2 + 2 * cos θ, 2 * sin θ)
  let C1 := (P 0).1 ^ 2 + (P 0).2 ^ 2 = 4
  (x - √2) ^ 2 + y ^ 2 = 2 ∧ (C1 = [((x, y) = P θ)]) → False :=
sorry

end convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191877


namespace convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191894

-- Definitions from the problem
def polar_eq_C (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Conversion to rectangular coordinates
def rectangular_eq_C (x y : ℝ) : Prop := (x - sqrt 2)^2 + y^2 = 2

-- Definitions of points and vectors
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨1, 0⟩

def M (θ : ℝ) : Point :=
  let ρ := polar_eq_C θ
  ⟨ρ * cos θ, ρ * sin θ⟩

def P (x y : ℝ) : Point := ⟨x, y⟩

-- Conditions and transformations
def vector_eq (p1 p2 : Point) (a : ℝ) : Prop :=
  P p1.x p1.y = ⟨a * p2.x + (1 - a) * A.x, a * p2.y + (1 - a) * A.y⟩

def parametric_eq_P (θ : ℝ) : Point :=
  ⟨3 - sqrt 2 + 2 * cos θ, 2 * sin θ⟩

-- Proof problems
theorem convert_polar_to_rectangular (θ : ℝ) :
  rectangular_eq_C (polar_eq_C θ * cos θ) (polar_eq_C θ * sin θ) :=
sorry

theorem parametric_eq_valid (θ : ℝ) :
  vector_eq (parametric_eq_P θ) (M θ) (sqrt 2) :=
sorry

theorem no_common_points :
  ∀ θ₁ θ₂, parametric_eq_P θ₁ ≠ M θ₂ :=
sorry

end convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191894


namespace proof_problem_l191_191870

section Exercise

variable (θ : ℝ)

def polarCoordEq (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos θ

def toRectCoord (ρ θ : ℝ) : Prop := ρ^2 = (2 * Real.sqrt 2 * Real.cos θ) * ρ

def circleEq (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 2

def parametricLocusC1 (θ : ℝ) : (ℝ × ℝ) := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

theorem proof_problem :
  (∃ ρ θ, polarCoordEq ρ θ → circleEq (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  ∀ θ, ¬ ∃ x y, (circleEq x y) ∧ (parametricLocusC1 θ = (x, y)) :=
by 
  sorry

end Exercise

end proof_problem_l191_191870


namespace income_ratio_l191_191521

-- Define the conditions
variables (I_A I_B E_A E_B : ℝ)
variables (Savings_A Savings_B : ℝ)

-- Given conditions
def expenditure_ratio : E_A / E_B = 3 / 2 := sorry
def savings_A : Savings_A = 1600 := sorry
def savings_B : Savings_B = 1600 := sorry
def income_A : I_A = 4000 := sorry
def expenditure_A : E_A = I_A - Savings_A := sorry
def expenditure_B : E_B = I_B - Savings_B := sorry

-- Prove it's implied that the ratio of incomes is 5:4
theorem income_ratio : I_A / I_B = 5 / 4 :=
by
  sorry

end income_ratio_l191_191521


namespace add_to_fraction_l191_191561

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l191_191561


namespace ABC_collinear_find_m_l191_191919

-- Define the coordinates and required condition for collinearity
def vector_OA (x : ℝ) := (1 : ℝ, Real.cos x : ℝ)
def vector_OB (x : ℝ) := (1 + Real.sin x : ℝ, Real.cos x : ℝ)
def vector_OC (x : ℝ) := (1 + (2 / 3) * Real.sin x : ℝ, Real.cos x : ℝ)

-- Condition: OC vector equation
axiom OC_vector_eq (x : ℝ) : vector_OC x = ((1 / 3) * ᴜ (1 : ℝ, Real.cos x) + (2 / 3) * ᴜ (1 + Real.sin x : ℝ, Real.cos x : ℝ))

-- Prove that A, B, C are collinear based on given vectors
theorem ABC_collinear : ∀ (x : ℝ) (H : x ∈ Set.Icc 0 (Real.pi / 2)), 
  collinear ℝ {vector_OA x, vector_OB x, vector_OC x} := 
  sorry

-- Define function f(x)
def f(x : ℝ) (m : ℝ) : ℝ := 
  let OA := 1 + Real.cos x
  let OC := 1 + (2 / 3) * Real.sin x + Real.cos x ^ 2
  let AB := Real.sin x
  OA * OC - (2 * m ^ 2 + (2 / 3)) * AB

-- Prove minimal value of f(x) equals 1/2 yields m = ±1/2
theorem find_m (x : ℝ) (H : x ∈ Set.Icc 0 (Real.pi / 2)) : 
  ∃ m : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x m ≥ 1 / 2) → m = 1 / 2 ∨ m = -1 / 2 :=
  sorry

end ABC_collinear_find_m_l191_191919


namespace hyperbola_fixed_point_proof_l191_191376

open Real

def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def is_eccentricity (a c e : ℝ) : Prop :=
  e = c / a ∧ e = sqrt 2

def on_hyperbola {a b x y : ℝ} (p : a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)) (T : ℝ × ℝ) : Prop :=
  T.fst = 3 ∧ T.snd = sqrt 5 ∧ (T.fst)^2 / a^2 - (T.snd)^2 / b^2 = 1

def in_hyperbola (a b x y : ℝ) (H : a > 0 ∧ b > 0 ∧ c^2 = a^2 + b^2) (h : on_hyperbola H) : Prop :=
  a = 2 ∧ b = 2 ∧ h = true

def fixed_point (M : ℝ × ℝ) (P Q : ℝ × ℝ) (x : ℝ) : Prop :=
  M = (1, x) ∧ P ≠ A ∧ Q ≠ B ∧ line_pq_through_fixed P Q

theorem hyperbola_fixed_point_proof :
  ∀ (A B M P Q : ℝ × ℝ),
  ∀ (a b : ℝ) (h : {T : ℝ × ℝ | on_hyperbola(T=3, sqrt 5)}) 
  (ma : line (1, sqrt 5) A) (mb : line (1, sqrt 5) B), 
  P ≠ A ∧ Q ≠ B → fixed_point M P Q (4, 0) :=
sorry


end hyperbola_fixed_point_proof_l191_191376


namespace harry_total_cost_in_silver_l191_191380

def cost_of_spellbooks_in_gold := 5 * 5
def cost_of_potion_kits_in_silver := 3 * 20
def cost_of_owl_in_gold := 28
def gold_to_silver := 9

def cost_in_silver :=
  (cost_of_spellbooks_in_gold + cost_of_owl_in_gold) * gold_to_silver + cost_of_potion_kits_in_silver

theorem harry_total_cost_in_silver : cost_in_silver = 537 := by
  sorry

end harry_total_cost_in_silver_l191_191380


namespace sum_alpha_F_k_l191_191545

-- Definitions of alpha and beta
def alpha := (1 + Real.sqrt 5) / 2
def beta := (1 - Real.sqrt 5) / 2

-- Definitions of Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

-- Given formula for F_n
def F (n : ℕ) : ℝ := (alpha^n - beta^n) / Real.sqrt 5

-- The proposition to be proved
theorem sum_alpha_F_k (n : ℕ) (h : n > 1) :
    ∑ k in Finset.range n + 1, (alpha^k * F k + 1/2) = F (2 * n + 1) :=
by
  sorry

end sum_alpha_F_k_l191_191545


namespace equal_medians_l191_191988

open EuclideanGeometry

theorem equal_medians (A B C D M N : Point)
  (hD : ∃ (α β : ℝ), α = 1 ∧ β = 2 ∧ α • (C - A) = D - A)
  (hM : midpoint D B M)
  (hN : midpoint B C N) :
  dist A M = dist D N := 
  sorry

end equal_medians_l191_191988


namespace probability_at_most_3_heads_l191_191656

open Real

theorem probability_at_most_3_heads (n : ℕ) (hn : n = 10) : 
  let p : ℝ := (∑ k in finset.range 4, (nat.choose n k)) / 2^n 
in p = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_l191_191656


namespace investment_in_scientific_notation_l191_191238

-- Define the problem conditions
def total_investment : ℕ := 82 * 10^9

-- Define the goal statement using theorem
theorem investment_in_scientific_notation :
  total_investment = 8.2 * 10^10 := sorry

end investment_in_scientific_notation_l191_191238


namespace flag_arrangements_l191_191531

/-- There are 21 flags, of which 12 are identical red flags and 9 are identical white flags.
    There are two distinguishable flagpoles. Each flagpole has at least one flag,
    and no two white flags on either pole are adjacent.
    Let M be the number of distinguishable arrangements using all of the flags.
    The theorem states that M mod 500 is equal to 355. -/
theorem flag_arrangements (r w : ℕ) (flagpoles : ℕ) : 
  (∃ M, 
    (r = 12 ∧ 
     w = 9 ∧
     flagpoles = 2 ∧ 
     M = (number_of_distinguishable_arrangements r w flagpoles) ∧
     M % 500 = 355)) :=
sorry

end flag_arrangements_l191_191531


namespace f_no_extrema_l191_191085

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_satisfies_differential_eq (x : ℝ) (h : x > 0) : x * (deriv f x) - f x = x * real.log x

axiom f_initial_condition : f (1 / real.exp 1) = 1 / real.exp 1

-- Statement to be proven
theorem f_no_extrema (x : ℝ) (h : x > 0) : 
  ¬(∃ a : ℝ, (a > 0 ∧ (∀ y : ℝ, y > 0 → f y ≥ f a)) ∨ (a > 0 ∧ (∀ y : ℝ, y > 0 → f y ≤ f a))) :=
sorry

end f_no_extrema_l191_191085


namespace least_n_ge_100_divides_sum_of_powers_l191_191178

theorem least_n_ge_100_divides_sum_of_powers (n : ℕ) (h₁ : n ≥ 100) :
    77 ∣ (Finset.sum (Finset.range (n + 1)) (λ k => 2^k) - 1) ↔ n = 119 :=
by
  sorry

end least_n_ge_100_divides_sum_of_powers_l191_191178


namespace problem1_problem2_l191_191375

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 1) - 1

theorem problem1 :
  (∀ x, -1 < x ∧ x < 0 → f' x < 0) ∧ (∀ x, 0 < x → f' x > 0) :=
sorry

theorem problem2 (x t : ℝ) (h : x > t ∧ t ≥ 0) :
  Real.exp (x - t) + Real.log (t + 1) > Real.log (x + 1) + 1 :=
sorry

end problem1_problem2_l191_191375


namespace C_and_C1_no_common_points_l191_191918

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  let x := rho * cos theta
  let y := rho * sin theta
  (x, y)

def curve_C_eq (theta : ℝ) : ℝ :=
  2 * sqrt 2 * cos theta

def A := (1, 0 : ℝ)

-- Predicate indicating point M is on curve C given in rectangular coordinates.
def on_curve_C (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  (x - sqrt 2) ^ 2 + y ^ 2 = 2

-- Definition for point P satisfying the given vector equation.
def point_P (A M : ℝ × ℝ) : ℝ × ℝ :=
  let x_A := A.1
  let y_A := A.2
  let x_M := M.1
  let y_M := M.2
  (√2 * (x_M - x_A) + x_A, √2 * (y_M - y_A) + y_A)

-- Prove that the rectangular coordinate equation is as specified and parametrized locus C_1 exists and has no common points with C.
theorem C_and_C1_no_common_points (M P : ℝ × ℝ) (theta : ℝ)
  (h_M_on_C : on_curve_C M)
  (h_P_eq : P = point_P A M) :
  let C1_center := (3 - sqrt 2, 0)
  let C1_radius := 2
  let C_center   := (sqrt 2, 0)
  let C_radius   := sqrt 2
  let C1_eq := (x - (3 - sqrt 2))^2 + y^2 = 4
  (M, P) ∉ C ∩ C1 :=
sorry

end C_and_C1_no_common_points_l191_191918


namespace sum_first_110_terms_l191_191343

noncomputable def sum_arithmetic (a1 d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem sum_first_110_terms (a1 d : ℚ) (h1 : sum_arithmetic a1 d 10 = 100)
  (h2 : sum_arithmetic a1 d 100 = 10) : sum_arithmetic a1 d 110 = -110 := by
  sorry

end sum_first_110_terms_l191_191343


namespace power_computation_efficiency_l191_191746

theorem power_computation_efficiency (a : ℝ) (n : ℕ) (h : n > 0) : 
    ∃ (f : ℕ → ℝ), (∀ (x : ℕ), f (x + 1) = f x * a) ∧
    (∃ (m ≤ 2 * (Nat.log2 n + 1)), ∀ c, (c ≤ m) ∧ (f n = a ^ n)) :=
sorry

end power_computation_efficiency_l191_191746


namespace C_and_C1_no_common_points_l191_191916

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  let x := rho * cos theta
  let y := rho * sin theta
  (x, y)

def curve_C_eq (theta : ℝ) : ℝ :=
  2 * sqrt 2 * cos theta

def A := (1, 0 : ℝ)

-- Predicate indicating point M is on curve C given in rectangular coordinates.
def on_curve_C (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  (x - sqrt 2) ^ 2 + y ^ 2 = 2

-- Definition for point P satisfying the given vector equation.
def point_P (A M : ℝ × ℝ) : ℝ × ℝ :=
  let x_A := A.1
  let y_A := A.2
  let x_M := M.1
  let y_M := M.2
  (√2 * (x_M - x_A) + x_A, √2 * (y_M - y_A) + y_A)

-- Prove that the rectangular coordinate equation is as specified and parametrized locus C_1 exists and has no common points with C.
theorem C_and_C1_no_common_points (M P : ℝ × ℝ) (theta : ℝ)
  (h_M_on_C : on_curve_C M)
  (h_P_eq : P = point_P A M) :
  let C1_center := (3 - sqrt 2, 0)
  let C1_radius := 2
  let C_center   := (sqrt 2, 0)
  let C_radius   := sqrt 2
  let C1_eq := (x - (3 - sqrt 2))^2 + y^2 = 4
  (M, P) ∉ C ∩ C1 :=
sorry

end C_and_C1_no_common_points_l191_191916


namespace average_age_union_l191_191206

variables (A B C : ℕ) (a b c : ℕ)
variables (hA : A = 40 * a) (hB : B = 25 * b) (hC : C = 45 * c)
variables (hAB : (A + B) = 31 * (a + b)) (hAC : (A + C) = 42 * (a + c)) (hBC : (B + C) = 36 * (b + c))

theorem average_age_union (A B C : ℕ) (a b c : ℕ)
    (hA : A = 40 * a)
    (hB : B = 25 * b)
    (hC : C = 45 * c)
    (hAB : (A + B) = 31 * (a + b))
    (hAC : (A + C) = 42 * (a + c))
    (hBC : (B + C) = 36 * (b + c)) :
    (A + B + C) / (a + b + c) = 72.35 := sorry

end average_age_union_l191_191206


namespace mary_number_l191_191091

-- Definitions of the properties and conditions
def is_two_digit_number (x : ℕ) : Prop :=
  10 ≤ x ∧ x < 100

def switch_digits (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a

def conditions_met (x : ℕ) : Prop :=
  is_two_digit_number x ∧ 91 ≤ switch_digits (4 * x - 7) ∧ switch_digits (4 * x - 7) ≤ 95

-- The statement to prove
theorem mary_number : ∃ x : ℕ, conditions_met x ∧ x = 14 :=
by {
  sorry
}

end mary_number_l191_191091


namespace triangle_area_l191_191638

noncomputable def a := 5
noncomputable def b := 4
noncomputable def s := (13 : ℝ) / 2 -- semi-perimeter
noncomputable def area := Real.sqrt (s * (s - a) * (s - b) * (s - b))

theorem triangle_area :
  a + 2 * b = 13 →
  (a > 0) → (b > 0) →
  (a < 2 * b) →
  (a + b > b) → 
  (a + b > b) →
  area = Real.sqrt 61.09375 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- We assume validity of these conditions and skip the proof for brevity.
  sorry

end triangle_area_l191_191638


namespace oldest_child_age_l191_191496

theorem oldest_child_age
  (ages : Fin 7 → ℕ)
  (avg_age : (∑ i, ages i) / 7 = 10)
  (distinct_ages : ∀ i ≠ j, ages i ≠ ages j)
  (consecutive_diff : ∀ i j, i < j → ages j = ages i + 3 * (j.1 - i.1)) :
  ages 6 = 19 := by
  sorry

end oldest_child_age_l191_191496


namespace proof_problem_l191_191865

section Exercise

variable (θ : ℝ)

def polarCoordEq (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos θ

def toRectCoord (ρ θ : ℝ) : Prop := ρ^2 = (2 * Real.sqrt 2 * Real.cos θ) * ρ

def circleEq (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 2

def parametricLocusC1 (θ : ℝ) : (ℝ × ℝ) := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

theorem proof_problem :
  (∃ ρ θ, polarCoordEq ρ θ → circleEq (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  ∀ θ, ¬ ∃ x y, (circleEq x y) ∧ (parametricLocusC1 θ = (x, y)) :=
by 
  sorry

end Exercise

end proof_problem_l191_191865


namespace projection_value_of_parabola_l191_191398

theorem projection_value_of_parabola
  (x y : ℝ) (h1 : x^2 = 4 * y)
  (h2 : abs (y + 1) = 5) :
  let A1 := (-sqrt (25 - x^2), 0),
      A2 := (sqrt (25 - x^2), 0),
      P := (x, y),
      AB := (A2.1 - A1.1, A2.2 - A1.2),
      AP := (x - A1.1, y - A1.2) in
  (AP.1 * AB.1 + AP.2 * AB.2) = 18 :=
by
  -- proof goes here
  sorry

end projection_value_of_parabola_l191_191398


namespace probability_both_red_buttons_l191_191047

noncomputable def jar_a_initial_red : ℕ := 10
noncomputable def jar_a_initial_blue : ℕ := 15
noncomputable def jar_a_initial_total := jar_a_initial_red + jar_a_initial_blue
noncomputable def removed_buttons : ℕ := (jar_a_initial_total - (jar_a_initial_total * 3 / 5).natAbs) / 2
noncomputable def jar_a_remaining_red := jar_a_initial_red - removed_buttons
noncomputable def jar_b_red := removed_buttons
noncomputable def jar_a_remaining_total := jar_a_initial_total * 3 / 5
noncomputable def jar_a_red_prob := jar_a_remaining_red / jar_a_remaining_total
noncomputable def jar_b_red_prob := jar_b_red / (2 * removed_buttons)
noncomputable def combined_prob := jar_a_red_prob * jar_b_red_prob

theorem probability_both_red_buttons : combined_prob = 1 / 6 :=
by
  sorry

end probability_both_red_buttons_l191_191047


namespace polar_to_rectangular_eq_no_common_points_l191_191848

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_rectangular_eq (ρ θ x y: ℝ) (h1: ρ = 2 * Real.sqrt 2 * Real.cos θ) 
(h2: polar_to_rectangular ρ θ = (x, y)) : (x - Real.sqrt 2) ^ 2 + y ^ 2 = 2 :=
by
  sorry

theorem no_common_points (x y θ : ℝ) (ρ : ℝ := 2 * Real.sqrt 2 * Real.cos θ)
(A := (1, 0) : ℝ × ℝ)
(M := polar_to_rectangular ρ θ)
(P := (sqrt 2 * (M.1 - 1) + 1, sqrt 2 * M.2)) : 
((P.1 - (3 - Real.sqrt 2)) ^ 2 + P.2 ^ 2 = 4) → 
¬ ((P.1 - Real.sqrt 2) ^ 2 + P.2 ^ 2 = 2) :=
by
  intro H1 H2
  sorry

end polar_to_rectangular_eq_no_common_points_l191_191848


namespace cannot_fit_all_pictures_l191_191493

theorem cannot_fit_all_pictures 
  (typeA_capacity : Nat) (typeB_capacity : Nat) (typeC_capacity : Nat)
  (typeA_count : Nat) (typeB_count : Nat) (typeC_count : Nat)
  (total_pictures : Nat)
  (h1 : typeA_capacity = 12)
  (h2 : typeB_capacity = 18)
  (h3 : typeC_capacity = 24)
  (h4 : typeA_count = 6)
  (h5 : typeB_count = 4)
  (h6 : typeC_count = 3)
  (h7 : total_pictures = 480) :
  (typeA_capacity * typeA_count + typeB_capacity * typeB_count + typeC_capacity * typeC_count < total_pictures) :=
  by sorry

end cannot_fit_all_pictures_l191_191493


namespace maximize_area_OMAN_proof_l191_191155

noncomputable def maximize_area_OMAN
  (O : Point) (A : Point)
  (angle_alpha : ℝ) 
  (angle_beta : ℝ)
  (h_alpha_beta : angle_alpha + angle_beta < Real.pi)
  (h_inside : (is_inside_angle O A angle_alpha))
  (M N : Point)
  (h_angle_MAN : ∠ M A N = angle_beta)
  (h_distance_eq : dist A M = dist A N) :
  Prop :=
  sorry

# We need a formal statement asserting the maximization of the area of OMAN.
theorem maximize_area_OMAN_proof 
  (O A : Point)
  (angle_alpha angle_beta : ℝ)
  (M N : Point)
  (h_alpha_beta : angle_alpha + angle_beta < Real.pi)
  (h_inside : (is_inside_angle O A angle_alpha))
  (h_angle_MAN : ∠ M A N = angle_beta)
  (h_distance_eq : dist A M = dist A N) :
  maximize_area_OMAN O A angle_alpha angle_beta h_alpha_beta h_inside M N h_angle_MAN h_distance_eq :=
sorry

end maximize_area_OMAN_proof_l191_191155


namespace sequence_proof_l191_191046

noncomputable def sequence_exists : Prop :=
  ∃ a : ℕ → ℕ, 
    (∀ r s : ℕ, 1 ≤ r → r ≤ s → s ≤ 2016 → Nat.is_composite (∑ i in Finset.range (s - r + 1), a (i + r))) ∧
    (∀ i : ℕ, i < 2016 → Nat.gcd (a i) (a (i + 1)) = 1) ∧
    (∀ i : ℕ, i < 2015 → Nat.gcd (a i) (a (i + 2)) = 1)

theorem sequence_proof : sequence_exists :=
  sorry

end sequence_proof_l191_191046


namespace student_tickets_second_day_l191_191489

variable (S T x: ℕ)

theorem student_tickets_second_day (hT : T = 9) (h_eq1 : 4 * S + 3 * T = 79) (h_eq2 : 12 * S + x * T = 246) : x = 10 :=
by
  sorry

end student_tickets_second_day_l191_191489


namespace circle_center_l191_191614

theorem circle_center (a b : ℝ) :
  (a ≠ 1 ∧ b ≠ 1 ∧ (a - 1) * (-1/2) = b - 1) ∧
  (2 * b + a = 3) ∧
  (b - (2) * (a - 1)) = (0 - 2) →  -- Conditions from the problem.
  (∃ a b : ℝ, a ≠ 1 ∧ b ≠ 1 ∧ a = 1/3 ∧ b = 4/3) :=
begin
  intros,
  -- Proof skipped, just the statement is here.
  sorry
end

end circle_center_l191_191614


namespace domain_tan_function_l191_191702

theorem domain_tan_function (x : ℝ) :
  ¬ ∃ k : ℤ, x = 2 * k + 1 / 3 ↔
  y = tan (π / 2 * x + π / 3) :=
sorry

end domain_tan_function_l191_191702


namespace minimum_highways_exists_l191_191979

theorem minimum_highways_exists (cities : Finset City) (hwys : Finset (City × City)) :
  cities.card = 10 ∧ 
  (∀ (c1 c2 c3 : City), {c1, c2, c3} ⊆ cities → 
    (hwys ⊆ (Finset.product cities cities) ∧ c1 ∈ cities ∧ c2 ∈ cities ∧ c3 ∈ cities →
      (hwys.contains (c1, c2) ∧ hwys.contains (c2, c3) ∧ hwys.contains (c3, c1)) ∨     
      ((∃ c1 c2 c3 : City, {c1, c2, c3} ⊆ cities) →
        (((hwys.contains (c1, c2) ∧ ¬ hwys.contains (c2, c3) ∧ ¬ hwys.contains (c3, c1)) ∨
          (¬ hwys.contains (c1, c2) ∧ hwys.contains (c2, c3) ∧ ¬ hwys.contains (c3, c1)) ∨
          (¬ hwys.contains (c1, c2) ∧ ¬ hwys.contains (c2, c3) ∧ hwys.contains (c3, c1))) ∧
          ¬ (hwys.contains (c1, c2) ∧ hwys.contains (c2, c3) ∧ hwys.contains (c3, c1))))) → 
  hwys.card ≥ 40 :=
by
  sorry

end minimum_highways_exists_l191_191979


namespace range_of_a_l191_191332

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then 2^x + 1 else -x^2 + a * x

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a < 3) ↔ (2 ≤ a ∧ a < 2 * Real.sqrt 3) := by
  sorry

end range_of_a_l191_191332


namespace mountain_climbing_time_proof_l191_191185

noncomputable def mountain_climbing_time (x : ℝ) : ℝ := (x + 2) / 4

theorem mountain_climbing_time_proof (x : ℝ) (h1 : (x / 3 + (x + 2) / 4 = 4)) : mountain_climbing_time x = 2 := by
  -- assume the given conditions and proof steps explicitly
  sorry

end mountain_climbing_time_proof_l191_191185


namespace max_elements_in_subset_l191_191446

def is_valid_subset (M : set ℕ) (A : set ℕ) : Prop :=
  ∀ x ∈ A, (15 * x ∈ M → 15 * x ∈ A)

theorem max_elements_in_subset (M : set ℕ) (A : set ℕ) 
  (hM : M = {x : ℕ | 1 ≤ x ∧ x ≤ 2005}) 
  (hA : A ⊆ M) 
  (h_valid : is_valid_subset M A) :
  ∃ (n : ℕ), n = finset.card (↑A : finset ℕ) ∧ n ≤ 8 :=
begin
  sorry,
end

end max_elements_in_subset_l191_191446


namespace cos_alpha_second_quadrant_l191_191773

theorem cos_alpha_second_quadrant (α : ℝ) (h₁ : (π / 2) < α ∧ α < π) (h₂ : Real.sin α = 5 / 13) :
  Real.cos α = -12 / 13 :=
by
  sorry

end cos_alpha_second_quadrant_l191_191773


namespace goose_survived_first_year_l191_191093

theorem goose_survived_first_year (total_eggs : ℕ) (eggs_hatched_ratio : ℚ) (first_month_survival_ratio : ℚ) 
  (first_year_no_survival_ratio : ℚ) 
  (eggs_hatched_ratio_eq : eggs_hatched_ratio = 2/3) 
  (first_month_survival_ratio_eq : first_month_survival_ratio = 3/4)
  (first_year_no_survival_ratio_eq : first_year_no_survival_ratio = 3/5)
  (total_eggs_eq : total_eggs = 500) :
  ∃ (survived_first_year : ℕ), survived_first_year = 100 :=
by
  sorry

end goose_survived_first_year_l191_191093


namespace concentrated_kola_percentage_l191_191210

theorem concentrated_kola_percentage 
  (initial_volume new_volume : ℝ) 
  (percentage_water percentage_sugar new_percentage_sugar : ℝ) 
  (added_sugar added_water added_kola : ℝ) 
  (c s : ℝ) :
  initial_volume = 340 ∧
  new_volume = 360 ∧
  percentage_water = 0.88 ∧
  added_sugar = 3.2 ∧
  added_water = 10 ∧
  added_kola = 6.8 ∧
  percentage_sugar = 7.5 / 100 ∧
  percentage_water + c + s = 1 ∧
  (s * initial_volume + added_sugar) / new_volume = percentage_sugar := 
  c = 0.05 :=
sorry

end concentrated_kola_percentage_l191_191210


namespace snail_distance_44_days_l191_191632

theorem snail_distance_44_days :
  (∑ n in Finset.range (44 + 1), (1 / (n + 1)) - (1 / (n + 2))) = 44 / 45 :=
by
  sorry

end snail_distance_44_days_l191_191632


namespace victor_weight_is_correct_l191_191211

-- Define the given conditions
def bear_daily_food : ℕ := 90
def victors_food_in_3_weeks : ℕ := 15
def days_in_3_weeks : ℕ := 21

-- Define the equivalent weight of Victor based on the given conditions
def victor_weight : ℕ := bear_daily_food * days_in_3_weeks / victors_food_in_3_weeks

-- Prove that the weight of Victor is 126 pounds
theorem victor_weight_is_correct : victor_weight = 126 := by
  sorry

end victor_weight_is_correct_l191_191211


namespace f_neg_one_l191_191955

-- Define f as a function from ℝ to ℝ
noncomputable def f : ℝ → ℝ := sorry

-- Assume f is an odd function
axiom odd_function (x : ℝ) : f (-x) = -f x

-- Define f for x ≥ 0
axiom f_pos (x : ℝ) (hx : x ≥ 0) : f x = 2 * x + 2 * x + b

noncomputable def b : ℝ := sorry

-- Define f(0) and solve for b
axiom f_zero : f 0 = 0

-- Show that f(-1) = -3
theorem f_neg_one : f (-1) = -3 :=
by
  -- Loop in the necessary definitions and axioms
  sorry

end f_neg_one_l191_191955


namespace planes_relationships_l191_191767

variables {m n l : Line}
variables {α β : Plane}

def skew_lines (m n : Line) : Prop := 
  ¬(m ∥ n) ∧ ¬(m = n) ∧ ¬(∃ p : Point, p ∈ m ∧ p ∈ n)

def perp_to_plane (l : Line) (α : Plane) : Prop :=
  ∀ p q : Point, p ∈ l → p ∈ α → q ∈ l → q ∈ α → p = q

def parallel (l : Line) (α : Plane) : Prop :=
  ∀ p ∈ l, ∀ q : Point, q ∈ α ∧ q ∉ l → ∃ r : Line, r ∥ l ∧ ∀ s ∈ α, ∃ t : Point, t ∈ r ∧ t = s

theorem planes_relationships (m n l : Line) (α β : Plane)
  (h_skew : skew_lines m n)
  (h1 : perp_to_plane m α)
  (h2 : perp_to_plane n β)
  (h3 : perp_to_plane l m)
  (h4 : perp_to_plane l n)
  (h5 : ¬(l ⊆ α))
  (h6 : ¬(l ⊆ β)) :
  ∃ p ∈ α, p ∈ β ∧ (∀ q ∈ (α ∩ β), l ∥ (Line.mk q l)) := sorry

end planes_relationships_l191_191767


namespace length_AB_of_trapezoid_ABCD_l191_191535

noncomputable def trapezoid_ABCD (A B C D O P : Type)
  [Condition1 : Trapezoid A B C D]
  [Condition2 : Parallel AB CD]
  [Condition3 : Segment BC = 39]
  [Condition4 : Segment CD = 39]
  [Condition5 : Perpendicular AD BD]
  [Condition6 : Intersection AC BD O]
  [Condition7 : Midpoint P BD]
  [Condition8 : Length OP = 13] : Prop :=
  length AB = 78

theorem length_AB_of_trapezoid_ABCD
  (A B C D O P : Type)
  [trapezoid_ABCD A B C D O P] :
  length AB = 78 := 
sorry

end length_AB_of_trapezoid_ABCD_l191_191535


namespace pq_implies_q_l191_191392

theorem pq_implies_q (p q : Prop) (h₁ : p ∨ q) (h₂ : ¬p) : q :=
by
  sorry

end pq_implies_q_l191_191392


namespace inverse_of_73_mod_74_l191_191297

theorem inverse_of_73_mod_74 :
  73 * 73 ≡ 1 [MOD 74] :=
by
  sorry

end inverse_of_73_mod_74_l191_191297


namespace area_ratio_l191_191146

variables (A B C A' B' C' : Type)
variables (AB BC CA : ℝ)
variables (AA' : ℝ) (BB' : ℝ) (CC' : ℝ)

-- Given conditions
axiom h1 : AA' = 3 * AB
axiom h2 : BB' = 5 * BC
axiom h3 : CC' = 8 * CA

theorem area_ratio {A B C A' B' C' : Type} {AB BC CA : ℝ} (h1 : AA' = 3 * AB) (h2 : BB' = 5 * BC) (h3 : CC' = 8 * CA) :
  let k := 8 in (k^2 = 64) :=
by
  let k := 8
  calc
    k^2 = 64 : by sorry

#eval "The code compiles successfully."

end area_ratio_l191_191146


namespace max_value_sqrt_expression_l191_191723

theorem max_value_sqrt_expression : 
  ∀ (x : ℝ), -36 ≤ x ∧ x ≤ 36 → 
  sqrt (36 + x) + sqrt (36 - x) ≤ 12 :=
by
  -- Proof goes here
  sorry

end max_value_sqrt_expression_l191_191723


namespace polar_to_rect_eq_and_locus_no_common_points_l191_191859

theorem polar_to_rect_eq_and_locus_no_common_points :
  let C := { (x, y) | (x - Real.sqrt 2)^2 + y^2 = 2 } in
  let C1 := { (x, y) | ∃ θ, x = 3 - Real.sqrt 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ } in
  (∀ (ρ θ : Real), ρ = 2 * Real.sqrt 2 * Real.cos θ → (Real.sqrt (ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos θ) - Real.sqrt 2)^2 + (Real.sin θ)^2 = 2 )
  ∧ ¬∃ (p : Real × Real), p ∈ C ∧ p ∈ C1 :=
by
  sorry

end polar_to_rect_eq_and_locus_no_common_points_l191_191859


namespace problem_solution_l191_191787

-- Define the main functions f and g with their respective parameters
def f (a b x : ℝ) := a * Real.sin (2 * x - Real.pi / 3) + b
def g (a b x : ℝ) := b * Real.cos (a * x + Real.pi / 6)

-- Make the conditions of the problem explicit in Lean
def conditions (a b : ℝ) : Prop :=
  (f a b (0 : ℝ) = 1) ∧ (f a b (Real.pi / 2) = -5) ∧ (a > 0)

-- Define the values to be proven for a and b
def values_a_b := (a = 3) ∧ (b = -2)

-- Define the set of x values for the maximum of g(x)
def x_set (k : ℤ) : ℝ := 5 * Real.pi / 18 + 2 * k * Real.pi / 3

-- Define the maximum value of g
def max_value_g := 2

-- The final theorem stating the original problem and its solution
theorem problem_solution (a b : ℝ) (k : ℤ) :
  conditions a b →
  (values_a_b ∧ (g a b (x_set k) = max_value_g) ∧ (∀ x, g a b x ≤ max_value_g)) :=
by
  sorry

end problem_solution_l191_191787


namespace range_of_k_l191_191126

theorem range_of_k (a b k : ℝ) (x y : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (3 : ℝ) = 3) (h4 : (-1 : ℝ) = -1) 
  (h5 : x^2 / a^2 + y^2 / b^2 = 1) :
  (k > 1/4) → (k ∈ set.Ioi (1/4)) :=
by 
  intro hk
  exact hk

end range_of_k_l191_191126


namespace convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191880

theorem convert_polar_to_rectangular (ρ θ x y : ℝ) (hρ : ρ = 2 * √2 * cos θ) :
  (x - √2) ^ 2 + y ^ 2 = 2 :=
sorry

theorem parametric_equations_and_no_common_points (x y θ : ℝ) :
  let A := (1 : ℝ, 0 : ℝ)
  let P (θ : ℝ) := (3 - √2 + 2 * cos θ, 2 * sin θ)
  let C1 := (P 0).1 ^ 2 + (P 0).2 ^ 2 = 4
  (x - √2) ^ 2 + y ^ 2 = 2 ∧ (C1 = [((x, y) = P θ)]) → False :=
sorry

end convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191880


namespace length_B1_C1_l191_191338

-- Definitions for the given conditions
def is_right_triangle (A B C : Type) [right_triangle A B C] (AC BC : ℝ) :=
  AC = 3 ∧ BC = 4

def is_translation (p1 p2 : Type) (dir : Type) :=
  ∃ (d : dir), p2 = p1 + d

-- Proposition to prove the length of B1 C1
theorem length_B1_C1 (A B C A1 B1 C1 : Type)
  [right_triangle A B C]
  [is_translation A A1 (line BC)]
  [is_translation B B1 (line A1C)]
  [is_translation C C1 (line A1B1)]
  (angle_A1B1C1 : ∠A1 B1 C1 = 90)
  (A1B1 : ℝ) (B1C1 : ℝ) :
  A1B1 = 1 → B1C1 = 12 :=
begin
  sorry
end

end length_B1_C1_l191_191338


namespace concyclicity_l191_191034

   -- Definitions and conditions
   variables (A B C D P Q : Point)
   variables (triangleABC : IsoscelesTriangle A B C)
   variables (H1 : side_length ABC B = side_length ABC C)

   -- Isosceles condition
   variable (isosceles : AB = AC > BC)
   variable (condition1 : DA = DB + DC)

   -- Perpendicular bisector conditions
   variable (pb1 : PerpendicularBisector A B P)
   variable (pb2 : PerpendicularBisector A C Q)

   -- Angle bisector conditions
   variable (ab1 : ExternalAngleBisector D A B P)
   variable (ab2 : ExternalAngleBisector D A C Q)

   -- Concyclic condition to be proved
   theorem concyclicity : Concyclic B C P Q :=
   by
     sorry
   
end concyclicity_l191_191034


namespace slope_tangent_line_at_zero_l191_191147

noncomputable def f (x : ℝ) : ℝ := (2 * x - 5) / (x^2 + 1)

theorem slope_tangent_line_at_zero : 
  (deriv f 0) = 2 :=
sorry

end slope_tangent_line_at_zero_l191_191147


namespace double_integral_value_l191_191654

noncomputable def region_D (x y : ℝ) : Prop :=
  (4 ≤ (y-2)^2 + x^2) ∧ ((y-4)^2 + x^2 ≤ 16) ∧ (y ≥ x / Real.sqrt 3) ∧ (x ≥ 0)

theorem double_integral_value :
  (∫∫ x in {p | region_D p.1 p.2}, x ∂p) = 35 :=
by
  sorry

end double_integral_value_l191_191654


namespace angle_ACB_length_MK_length_AB_area_CMN_l191_191500

-- Defining the given conditions
variables {A B C M N K L : Point} 
variable {O : Circle}
variable (h_circle_radius : O.radius = sqrt 3)
variable (h_tangent_BC : O.tangent B C K)
variable (h_tangent_AC : O.tangent A C L)
variable (h_intersect_AB : O.intersects_AB A B M N)
variable (h_M_between_A_N : M.between A N)
variable (h_MK_parallel_AC : MK.parallel AC)
variable (h_KC : KC = 1)
variable (h_AL : AL = 6)

-- Proof problems to be defined
theorem angle_ACB (h_circle_radius h_tangent_BC h_tangent_AC h_intersect_AB h_M_between_A_N h_MK_parallel_AC h_KC h_AL) :
  angle A C B = 2 * pi / 3 := sorry

theorem length_MK (h_circle_radius h_tangent_BC h_tangent_AC h_intersect_AB h_M_between_A_N h_MK_parallel_AC h_KC h_AL) :
  MK.length = 3 := sorry

theorem length_AB (h_circle_radius h_tangent_BC h_tangent_AC h_intersect_AB h_M_between_A_N h_MK_parallel_AC h_KC h_AL) :
  AB.length = 7 * sqrt 21 / 4 := sorry

theorem area_CMN (h_circle_radius h_tangent_BC h_tangent_AC h_intersect_AB h_M_between_A_N h_MK_parallel_AC h_KC h_AL) :
  triangle_area C M N = 5 * sqrt 3 / 4 := sorry

end angle_ACB_length_MK_length_AB_area_CMN_l191_191500


namespace g_675_l191_191079

noncomputable def g : ℕ → ℤ :=
  sorry -- This definition is abstracted as we only need the properties for the proof

axiom g_mul (x y : ℕ) : x > 0 → y > 0 → g(x * y) = g(x) + g(y)
axiom g_15 : g(15) = 17
axiom g_45 : g(45) = 23

theorem g_675 : g(675) = 40 :=
  sorry

end g_675_l191_191079


namespace quadratic_solution_difference_square_l191_191001

theorem quadratic_solution_difference_square :
  ∀ (Φ ϕ : ℝ), (Φ ≠ ϕ) ∧ (Φ^2 - 3*Φ + 1 = 0) ∧ (ϕ^2 - 3*ϕ + 1 = 0) → (Φ - ϕ)^2 = 5 :=
by {
  intros Φ ϕ h,
  cases h with h1 h2,
  cases h2 with hΦ hϕ,
  -- We will fill the proof here
  sorry
}

end quadratic_solution_difference_square_l191_191001


namespace ellipse_foci_and_eccentricity_l191_191305

theorem ellipse_foci_and_eccentricity :
  (∃ (a b c : ℝ), a = 3 ∧ b = 1.5 ∧ c = (sqrt 6.75) ∧
  (2 * c = 5.196) ∧ (c / a = 0.866)) :=
by
  let a := 3
  let b := 1.5
  let c := sqrt 6.75
  exact ⟨a, b, c, rfl, rfl, rfl, by norm_num, by norm_num⟩

end ellipse_foci_and_eccentricity_l191_191305


namespace remaining_soup_can_feed_adults_l191_191607

-- Definitions for the problem conditions
def cans_per_children : ℕ := 6
def cans_per_adults : ℕ := 4
def initial_cans : ℕ := 6
def children_fed : ℕ := 18

-- Statement to express the problem in Lean 4
theorem remaining_soup_can_feed_adults :
  (children_fed / cans_per_children * cans_per_children = children_fed) →
  initial_cans - (children_fed / cans_per_children) = 3 →
  (initial_cans - (children_fed / cans_per_children)) * cans_per_adults = 12 :=
begin
  intros h1 h2,
  sorry
end

end remaining_soup_can_feed_adults_l191_191607


namespace simplify_fraction_product_l191_191484

theorem simplify_fraction_product :
  8 * (15 / 14) * (-49 / 45) = - (28 / 3) :=
by
  sorry

end simplify_fraction_product_l191_191484


namespace sum_of_powers_l191_191002

theorem sum_of_powers (n : ℕ) (h : 8 ∣ n) : 
  (∑ k in Finset.range (n + 1), (k + 1) * (-Complex.i)^k) = 1.5 * n + 1 - 0.25 * n * Complex.i :=
by
  sorry

end sum_of_powers_l191_191002


namespace given_conditions_then_inequality_holds_l191_191478

theorem given_conditions_then_inequality_holds 
  {α : ℝ} {n : ℕ} {x : ℕ → ℝ}
  (hα : α ≤ 1)
  (hx_inequality : ∀ i, i ≤ n → 1 ≥ x i ∧ x i > 0)
  (hx_decreasing : ∀ i j, i ≤ j ∧ i ≤ n ∧ j ≤ n → x i ≥ x j) :
  (1 + ∑ i in Finset.range (n+1), x i) ^ α ≤
  1 + ∑ i in Finset.range (n+1), (i+1) ^ (α-1) * (x i)^α :=
sorry

end given_conditions_then_inequality_holds_l191_191478


namespace dice_sum_18_l191_191574

theorem dice_sum_18 (n : ℕ) : 
  (∃ k : ℕ, k = (5.choose k) ∧ 
    sum (λ i : fin 5, (nat.bounded 1 8) i ≥ n) = 18) →
  n = 2380 :=
sorry

end dice_sum_18_l191_191574


namespace num_integers_in_abs_inequality_l191_191386

theorem num_integers_in_abs_inequality (x : ℤ) : 
  (|x - 3| ≤ 7.4) → real.count (λ x, (↑(-4) : ℝ) ≤ x ∧ x ≤ 10.4) = 15 :=
by
  sorry

end num_integers_in_abs_inequality_l191_191386


namespace smallest_n_for_area_gt_5000_l191_191280

def complex_vertex (n : ℝ) : ℂ := n + complex.I

def area_of_triangle (a b c : ℂ) : ℝ :=
  (complex.abs ((b - a) * complex.I * (c - a))) / 2

theorem smallest_n_for_area_gt_5000 :
  ∃ (n : ℝ), ((0 < n) ∧ (4 * n ^ 5 - 6 * n ^ 4 - 2 * n ^ 3 + 6 * n ^ 2 + 2 * n + 2 > 10000)) ∧
             ∀ (m : ℝ), (0 < m ∧ (4 * m ^ 5 - 6 * m ^ 4 - 2 * m ^ 3 + 6 * m ^ 2 + 2 * m + 2 > 10000) → n ≤ m) :=
begin
  use 9,
  sorry
end

end smallest_n_for_area_gt_5000_l191_191280


namespace female_salmon_count_l191_191532

theorem female_salmon_count (total_salmon : ℕ) (male_salmon : ℕ) (h1 : total_salmon = 971639) (h2 : male_salmon = 712261) : ∃ female_salmon : ℕ, female_salmon = total_salmon - male_salmon ∧ female_salmon = 259378 :=
by
  have female_salmon := total_salmon - male_salmon
  use female_salmon
  split
  . refl
  . rw [h1, h2]
    exact calc
      971639 - 712261 = 259378 : by norm_num

end female_salmon_count_l191_191532


namespace convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191875

theorem convert_polar_to_rectangular (ρ θ x y : ℝ) (hρ : ρ = 2 * √2 * cos θ) :
  (x - √2) ^ 2 + y ^ 2 = 2 :=
sorry

theorem parametric_equations_and_no_common_points (x y θ : ℝ) :
  let A := (1 : ℝ, 0 : ℝ)
  let P (θ : ℝ) := (3 - √2 + 2 * cos θ, 2 * sin θ)
  let C1 := (P 0).1 ^ 2 + (P 0).2 ^ 2 = 4
  (x - √2) ^ 2 + y ^ 2 = 2 ∧ (C1 = [((x, y) = P θ)]) → False :=
sorry

end convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191875


namespace area_quadrilateral_ABCD_l191_191827

-- Define the quadrilateral and its properties
def quadrilateral_ABCD (A B C D : Type) [has_dist A B : ℝ] [has_dist B C : ℝ] [has_dist C D : ℝ] [has_angle B C D : ℝ] :=
  has_dist A B = 4 ∧ has_dist B C = 5 ∧ has_dist C D = 4 ∧ has_angle B C D = 60

-- Prove the area of quadrilateral ABCD
theorem area_quadrilateral_ABCD (A B C D : Type) 
  [has_dist A B : ℝ] [has_dist B C : ℝ] [has_dist C D : ℝ] [has_angle B C D : ℝ] 
  (h : quadrilateral_ABCD A B C D) : 
  let area_triangle := (1/2) * (has_dist A B) * (has_dist C D) * Math.sqrt(3) / 2 in
  2 * area_triangle = 8 * Math.sqrt(3) :=
  sorry

end area_quadrilateral_ABCD_l191_191827


namespace count_rational_numbers_in_sequence_l191_191504

noncomputable theory

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_rational_numbers_in_sequence :
  let seq := λ k : ℕ, 7.301 + k * 0.001 in
  let sqrt_seq := λ k : ℕ, Real.sqrt (seq k) in
  -- Ensure k covers sequence from 0 up to the point where seq k = 16.003
  (∃ k0 k1 : ℕ, seq k0 = 7.301 ∧ seq k1 = 16.003 ∧ k0 ≤ k1) →
  -- Count how many elements in sqrt_seq are rational numbers 
  (finset.range (k1 - k0 + 1)).filter (λ k, is_perfect_square (Real.floor (seq k * 1000))) = 13 :=
by
  sorry

end count_rational_numbers_in_sequence_l191_191504


namespace eighth_flip_last_l191_191396

def fair_coin_probability (p : ℕ → ℝ) (flip_count : ℕ) : ℝ :=
  if flip_count = 8 then (1/2)^7 else 0

theorem eighth_flip_last (p : ℕ → ℝ) (flip_count : ℕ) :
  (∀ n, p n = 1/2) →
  flip_count = 8 →
  fair_coin_probability p flip_count = 1/128 :=
by
  intros h1 h2
  rw h2
  simp [fair_coin_probability, h1]
  sorry

end eighth_flip_last_l191_191396


namespace inequality_holds_l191_191700

theorem inequality_holds (a b c d : ℝ) (h : a ∈ Icc (-1 : ℝ) (∞) ∧ b ∈ Icc (-1 : ℝ) (∞) ∧ c ∈ Icc (-1 : ℝ) (∞) ∧ d ∈ Icc (-1 : ℝ) (∞)) : 
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3 / 4) * (a + b + c + d) := 
sorry

end inequality_holds_l191_191700


namespace arithmetic_sequence_a3_l191_191923

theorem arithmetic_sequence_a3 (a1 d : ℤ) (h : a1 + (a1 + d) + (a1 + 2 * d) + (a1 + 3 * d) + (a1 + 4 * d) = 20) : 
  a1 + 2 * d = 4 := by
  sorry

end arithmetic_sequence_a3_l191_191923


namespace line_passes_through_fixed_point_l191_191359

-- Statement to prove that the line always passes through the point (2, 2)
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, ∃ x y : ℝ, 
  (1 + 4 * k) * x - (2 - 3 * k) * y + (2 - 14 * k) = 0 ∧ x = 2 ∧ y = 2 :=
sorry

end line_passes_through_fixed_point_l191_191359


namespace pentagon_fifth_angle_pentagon_fifth_angle_45_l191_191831

theorem pentagon_fifth_angle (x : ℝ) : 
  let sum := 104 + 97 + x + 2 * x
  let R := 540 - sum
  R = 339 - 3 * x := 
by
  sorry

theorem pentagon_fifth_angle_45 : 
  pentagon_fifth_angle 45 :=
by
  exact pentagon_fifth_angle 45

end pentagon_fifth_angle_pentagon_fifth_angle_45_l191_191831


namespace number_of_valid_votes_candidates_percentage_of_total_votes_candidates_overall_voter_turnout_coalition_valid_votes_percentage_l191_191033

-- Define valid vote percentages and total votes cast
def percent_valid_votes_X : ℝ := 0.47
def percent_valid_votes_Y : ℝ := 0.32
def percent_valid_votes_Z : ℝ := 0.21

def percent_invalid_votes : ℝ := 0.18
def total_votes : ℕ := 750000
def registered_voters : ℕ := 900000

-- Calculated values
def valid_vote_percentage := 1 - percent_invalid_votes
def valid_votes := (valid_vote_percentage * total_votes.toFloat).toNat
def votes_X := (percent_valid_votes_X * valid_votes.toFloat).toNat
def votes_Y := (percent_valid_votes_Y * valid_votes.toFloat).toNat
def votes_Z := (percent_valid_votes_Z * valid_votes.toFloat).toNat

-- Percentages of total votes including invalid votes
def percent_total_votes_X := (votes_X.toFloat / total_votes.toFloat) * 100
def percent_total_votes_Y := (votes_Y.toFloat / total_votes.toFloat) * 100
def percent_total_votes_Z := (votes_Z.toFloat / total_votes.toFloat) * 100

-- Voter turnout percentage
def voter_turnout := (total_votes.toFloat / registered_voters.toFloat) * 100

-- Percentage of valid votes for coalition (X and Y)
def coalition_percent_valid_votes := percent_valid_votes_X + percent_valid_votes_Y

-- Proof Statements
theorem number_of_valid_votes_candidates :
    votes_X = 289050 ∧ votes_Y = 196800 ∧ votes_Z = 129150 := by
    sorry

theorem percentage_of_total_votes_candidates :
    percent_total_votes_X = 38.54 ∧ percent_total_votes_Y = 26.24 ∧ percent_total_votes_Z = 17.22 := by
    sorry

theorem overall_voter_turnout :
    voter_turnout = 83.33 := by
    sorry

theorem coalition_valid_votes_percentage :
    coalition_percent_valid_votes = 0.79 := by
    sorry

end number_of_valid_votes_candidates_percentage_of_total_votes_candidates_overall_voter_turnout_coalition_valid_votes_percentage_l191_191033


namespace total_amount_l191_191194

variable (x y z : ℝ)

def condition1 : Prop := y = 0.45 * x
def condition2 : Prop := z = 0.30 * x
def condition3 : Prop := y = 36

theorem total_amount (h1 : condition1 x y)
                     (h2 : condition2 x z)
                     (h3 : condition3 y) :
  x + y + z = 140 :=
by
  sorry

end total_amount_l191_191194


namespace distance_sum_eq_distance_l191_191947

variables (A B C D E P : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space P]
variables (α : A → B → C → Prop) (β : A → B → D → Prop) (γ : A → C → E → Prop) (δ : D → E → P → Prop)
variables (BA : P → Prop) (BC : P → Prop) (opposite_side : P → Prop)
variables (d : P → (P → P → Prop) → ℝ)

-- Conditions
variable (h1 : α A B C)
variable (h2 : β A B D)
variable (h3 : γ A C E)
variable (h4 : δ D E P)
variable (h5 : BA P)
variable (h6 : BC P)
variable (h7 : opposite_side P)

-- Proof goal
theorem distance_sum_eq_distance :
  ∀ P, δ D E P → BA P → BC P → opposite_side P → d P (BC P) + d P (A C) = d P (A B) :=
sorry

end distance_sum_eq_distance_l191_191947


namespace right_triangle_leg_sum_range_l191_191750

theorem right_triangle_leg_sum_range (x y : ℝ) (hypotenuse : x^2 + y^2 = 5) : 
  sqrt 5 < x + y ∧ x + y ≤ sqrt 10 :=
by
  sorry

end right_triangle_leg_sum_range_l191_191750


namespace abs_sum_neq_3_nor_1_l191_191004

theorem abs_sum_neq_3_nor_1 (a b : ℤ) (h₁ : |a| = 3) (h₂ : |b| = 1) : (|a + b| ≠ 3) ∧ (|a + b| ≠ 1) := sorry

end abs_sum_neq_3_nor_1_l191_191004


namespace max_height_projectile_l191_191628

-- Define the quadratic function for height above the ground
def height (t : ℝ) : ℝ :=
  -20 * t^2 + 100 * t + 36

-- Prove that the maximum height of the projectile is 161 feet
theorem max_height_projectile : ∃ t : ℝ, height t = 161 :=
sorry

end max_height_projectile_l191_191628


namespace hyperbola_eqn_solution_l191_191507

noncomputable def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem hyperbola_eqn_solution (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_eccentricity : a * 2 * 2 = b^2 + a^2)
  (h_tangent : a * b / (√(a^2 + b^2)) = √3) :
  hyperbola_equation 2 (2 * √3) x y :=
by
  sorry

end hyperbola_eqn_solution_l191_191507


namespace point_A_coordinates_l191_191130

noncomputable def f (a x : ℝ) : ℝ := a * x - 1

theorem point_A_coordinates (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 1 :=
sorry

end point_A_coordinates_l191_191130


namespace max_value_xy_on_segment_l191_191836

-- Definitions from the conditions
def is_on_segment_AB (P A B : ℝ × ℝ) : Prop := P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

def collinear_condition (x y : ℝ) : Prop := (x / 3) + (y / 4) = 1

open_locale big_operators
open_locale classical

theorem max_value_xy_on_segment {P A B C : ℝ × ℝ} (h₀ : ∠CAB = 90) (h₁ : A = (0, 3)) (h₂ : B = (4, 0)) (h₃ : C = (0, 0))
  (h₄ : is_on_segment_AB P A B) (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_collinear : collinear_condition x y) :
  xy ≤ 3
  := by
  sorry

end max_value_xy_on_segment_l191_191836


namespace basketball_weight_l191_191485

theorem basketball_weight (b k : ℝ) (h1 : 6 * b = 4 * k) (h2 : 3 * k = 72) : b = 16 :=
by
  sorry

end basketball_weight_l191_191485


namespace painting_time_equation_l191_191681

theorem painting_time_equation (t : ℝ) :
  let Doug_rate := (1 : ℝ) / 5
  let Dave_rate := (1 : ℝ) / 7
  let combined_rate := Doug_rate + Dave_rate
  (combined_rate * (t - 1) = 1) :=
sorry

end painting_time_equation_l191_191681


namespace max_k_condition_l191_191347

theorem max_k_condition (x₀ x₁ x₂ x₃ : ℝ) (h₀ : x₀ > x₁) (h₁ : x₁ > x₂) (h₂ : x₂ > x₃) (h₃ : x₃ > 0) 
: ∃ k : ℝ, (∀  (x₀ x₁ x₂ x₃ : ℝ) (h₀ : x₀ > x₁) (h₁ : x₁ > x₂) (h₂ : x₂ > x₃) (h₃ : x₃ > 0), 
  log 1993 (x₀ / x₁) + log 1993 (x₁ / x₂) + log 1993 (x₂ / x₃) ≥ k * log 1993 (x₀ / x₃)) ∧ k = 9 :=
by
  use 9
  sorry

end max_k_condition_l191_191347


namespace Carrie_hourly_wage_l191_191273

theorem Carrie_hourly_wage (hours_per_week : ℕ) (weeks_per_month : ℕ) (cost_bike : ℕ) (remaining_money : ℕ)
  (total_hours : ℕ) (total_savings : ℕ) (x : ℕ) :
  hours_per_week = 35 → 
  weeks_per_month = 4 → 
  cost_bike = 400 → 
  remaining_money = 720 → 
  total_hours = hours_per_week * weeks_per_month → 
  total_savings = cost_bike + remaining_money → 
  total_savings = total_hours * x → 
  x = 8 :=
by 
  intros h_hw h_wm h_cb h_rm h_th h_ts h_tx
  sorry

end Carrie_hourly_wage_l191_191273


namespace relationship_log2_2_pow_03_l191_191144

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem relationship_log2_2_pow_03 : 
  log_base_2 0.3 < (0.3)^2 ∧ (0.3)^2 < 2^(0.3) :=
by
  sorry

end relationship_log2_2_pow_03_l191_191144


namespace triangle_ABC_area_l191_191823

variables (A B C D E F : Type) [HasArea A B C D E F]

noncomputable def is_midpoint (D B C : Type) : Prop :=
  -- Definition for midpoint
  D = midpoint B C

noncomputable def points_on_line_segment (A C : Type) (E : Type) : Prop :=
  -- Definition for points on line such that AE:EC = 1:2
  E ∈ segment A C ∧ ratio A E C = 1:2

noncomputable def points_on_line_segment (A D : Type) (F : Type) : Prop :=
  -- Definition for points on line such that AF:FD = 2:1
  F ∈ segment A D ∧ ratio A F D = 2:1

variables (area : A B C D E F → ℝ)

theorem triangle_ABC_area :
  is_midpoint D B C →
  points_on_line_segment A C E →
  points_on_line_segment A D F →
  area ⟨D, E, F⟩ = 24 →
  area ⟨A, B, C⟩ = 432 :=
begin
  sorry
end

end triangle_ABC_area_l191_191823


namespace what_is_A_score_l191_191320

def scores (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
  A ≤ 100 ∧ B ≤ 100 ∧ C ≤ 100 ∧ D ≤ 100

def statements (A B C D : ℕ) (rankA rankB rankC rankD : ℕ) : Prop :=
  (rankA < rankB ∧ rankA < rankC) ∨ (¬(rankA < rankB ∧ rankA < rankC)) ∧  -- A's statement half true half false
  (B = 90 ∧ B = D + 2) ∨ (¬(B = 90 ∧ B = D + 2)) ∧  -- B's statement half true half false
  (rankC < rankD ∧ rankC < rankA) ∨ (¬(rankC < rankD ∧ rankC < rankA)) ∧  -- C's statement half true half false
  (D = 91 ∧ D = B + 3) ∨ (¬(D = 91 ∧ D = B + 3))  -- D's statement half true half false 

def uniqueStatements (A B C D : ℕ) : Prop := 
  scores A B C D ∧ statements A B C D _ _ _ _  -- Note "_" indicates rankings, which do not need specification as conditions only deal directly with scores.

theorem what_is_A_score (A B C D : ℕ) (h : uniqueStatements A B C D) : A = 90 :=
sorry

end what_is_A_score_l191_191320


namespace candidate_percentage_l191_191608

theorem candidate_percentage (P : ℚ) (votes_cast : ℚ) (loss : ℚ)
  (h1 : votes_cast = 2000) 
  (h2 : loss = 640) 
  (h3 : (P / 100) * votes_cast + (P / 100) * votes_cast + loss = votes_cast) :
  P = 34 :=
by 
  sorry

end candidate_percentage_l191_191608


namespace polar_to_rect_eq_and_locus_no_common_points_l191_191856

theorem polar_to_rect_eq_and_locus_no_common_points :
  let C := { (x, y) | (x - Real.sqrt 2)^2 + y^2 = 2 } in
  let C1 := { (x, y) | ∃ θ, x = 3 - Real.sqrt 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ } in
  (∀ (ρ θ : Real), ρ = 2 * Real.sqrt 2 * Real.cos θ → (Real.sqrt (ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos θ) - Real.sqrt 2)^2 + (Real.sin θ)^2 = 2 )
  ∧ ¬∃ (p : Real × Real), p ∈ C ∧ p ∈ C1 :=
by
  sorry

end polar_to_rect_eq_and_locus_no_common_points_l191_191856


namespace fewest_students_l191_191575

theorem fewest_students :
  ∃ n : ℕ, (n % 4 = 1 ∧ n % 5 = 2 ∧ n % 7 = 3) ∧
  ∀ m : ℕ, (m % 4 = 1 ∧ m % 5 = 2 ∧ m % 7 = 3) → n ≤ m :=
begin
  sorry
end

end fewest_students_l191_191575


namespace polar_to_rect_eq_and_locus_no_common_points_l191_191862

theorem polar_to_rect_eq_and_locus_no_common_points :
  let C := { (x, y) | (x - Real.sqrt 2)^2 + y^2 = 2 } in
  let C1 := { (x, y) | ∃ θ, x = 3 - Real.sqrt 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ } in
  (∀ (ρ θ : Real), ρ = 2 * Real.sqrt 2 * Real.cos θ → (Real.sqrt (ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos θ) - Real.sqrt 2)^2 + (Real.sin θ)^2 = 2 )
  ∧ ¬∃ (p : Real × Real), p ∈ C ∧ p ∈ C1 :=
by
  sorry

end polar_to_rect_eq_and_locus_no_common_points_l191_191862


namespace simplify_fraction_l191_191107

noncomputable def sin_15 := Real.sin (15 * Real.pi / 180)
noncomputable def cos_15 := Real.cos (15 * Real.pi / 180)
noncomputable def angle_15 := 15 * Real.pi / 180

theorem simplify_fraction : (1 / sin_15 - 1 / cos_15 = 2 * Real.sqrt 2) :=
by
  sorry

end simplify_fraction_l191_191107


namespace digit_sum_2021_number_l191_191215

theorem digit_sum_2021_number (N : ℕ) (k : ℕ) (h_sum_digits : (nat.digits 10 N).sum = 2021) (h_div : N / 7 = list.length (list.replicate k 7)):
  k = 503 :=
sorry

end digit_sum_2021_number_l191_191215


namespace polynomial_roots_l191_191707

theorem polynomial_roots (k r : ℝ) (hk_pos : k > 0) 
(h_sum : r + 1 = 2 * k) (h_prod : r * 1 = k) : 
  r = 1 ∧ (∀ x, (x - 1) * (x - 1) = x^2 - 2 * x + 1) := 
by 
  sorry

end polynomial_roots_l191_191707


namespace parabola_equation_passing_through_point_l191_191148

theorem parabola_equation_passing_through_point (p : ℝ) (h : p > 0) :
  (∀ (x y : ℝ), (y = 2) ∧ (x = 1) → y^2 = 2 * p * x) → y^2 = 4 * x :=
by
  intro h1
  have h2 : 4 = 2 * p := by sorry   -- substitution of (1, 2)
  have h3 : p = 2 := by sorry       -- solving for p
  show y^2 = 4 * x by
    rw [h3]
  sorry  -- the actual proof that y^2 = 4 * x

end parabola_equation_passing_through_point_l191_191148


namespace tonya_stamps_after_trade_l191_191938

noncomputable def stamps_left_after_trade 
  (matches_per_stamp : ℕ) (matches_per_matchbook : ℕ) (tonya_stamps : ℕ) 
  (jimmy_matchbooks : ℕ) : ℕ :=
  let jimmy_stamps := (matches_per_matchbook * jimmy_matchbooks) / matches_per_stamp in
  tonya_stamps - jimmy_stamps

theorem tonya_stamps_after_trade : stamps_left_after_trade 12 24 13 5 = 3 := by
  sorry

end tonya_stamps_after_trade_l191_191938


namespace complement_angle_l191_191329

theorem complement_angle (A : ℝ) (hA : A = 35) : 90 - A = 55 := by
  sorry

end complement_angle_l191_191329


namespace pork_price_increase_l191_191824

variable (x : ℝ)
variable (P_aug P_oct : ℝ)
variable (P_aug := 32)
variable (P_oct := 64)

theorem pork_price_increase :
  P_aug * (1 + x) ^ 2 = P_oct :=
sorry

end pork_price_increase_l191_191824


namespace total_trees_l191_191419

theorem total_trees (apricot_trees : ℕ) (peach_mult : ℕ) (peach_trees : ℕ) (total_trees : ℕ) :
  apricot_trees = 58 → peach_mult = 3 → peach_trees = peach_mult * apricot_trees → total_trees = apricot_trees + peach_trees → total_trees = 232 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw h3 at h4
  rw h4
  exact rfl

end total_trees_l191_191419


namespace diver_time_to_ship_l191_191583

theorem diver_time_to_ship 
  (rate_of_descent : ℕ)
  (total_depth : ℕ) 
  (rate_assumption : rate_of_descent = 60)
  (depth_assumption : total_depth = 3600) : 
  total_depth / rate_of_descent = 60 := 
by
  rw [rate_assumption, depth_assumption]
  simp
  exact rfl

end diver_time_to_ship_l191_191583


namespace roots_of_polynomial_l191_191133

-- Define the polynomial
def polynomial (x : ℝ) (b c : ℝ) : ℝ := (2 / Real.sqrt 3) * x^2 + b * x + c

-- Define the conditions for the problem
def conditions (b c : ℝ) : Prop := 
  ∃ K L M : ℝ, (K = 0) ∧ (polynomial L b c = 0) ∧ (polynomial M b c = 0) ∧ 
  (L ≠ M) ∧ (L = 3 * M) ∧ (KM = 2 * L) ∧ (angle LKM = π / 3)

-- Define the statement we want to prove
theorem roots_of_polynomial (b c : ℝ) (h : conditions b c) : 
  ∃ p : ℝ, p = 1.5 ∧ 3 * p = 4.5 :=
begin
  sorry
end

end roots_of_polynomial_l191_191133


namespace probability_equation_solution_l191_191261

theorem probability_equation_solution :
  (∃ x : ℝ, 3 * x ^ 2 - 8 * x + 5 = 0 ∧ 0 ≤ x ∧ x ≤ 1) → ∃ x = 1 :=
by
  sorry

end probability_equation_solution_l191_191261


namespace pieces_in_figure_20_l191_191981

def num_small_pieces (n : ℕ) : ℕ := 4 * n
def num_large_pieces (n : ℕ) : ℕ := n^2 - n

theorem pieces_in_figure_20 : num_small_pieces(20) + num_large_pieces(20) = 460 :=
by
  have small_pieces := num_small_pieces 20
  have large_pieces := num_large_pieces 20
  have sum_pieces := small_pieces + large_pieces
  rw [num_small_pieces, num_large_pieces]
  norm_num
  sorry

end pieces_in_figure_20_l191_191981


namespace students_in_all_three_l191_191833

-- Definitions of the given problem conditions
def total_students : ℕ := 100
def chinese_competition : ℕ := 39
def math_competition : ℕ := 49
def english_competition : ℕ := 41
def both_chinese_and_math : ℕ := 14
def both_math_and_english : ℕ := 13
def both_chinese_and_english : ℕ := 9
def no_competition : ℕ := 1

-- The number of students who participated in all three competitions
def all_three_competitions : ℕ := 6

-- Lean Statement
theorem students_in_all_three :
  let total_three := chinese_competition + math_competition + english_competition
  let common_two := both_chinese_and_math + both_math_and_english + both_chinese_and_english
  in total_students - no_competition = total_three - common_two + all_three_competitions :=
by {
  let total_three := 39 + 49 + 41,
  let common_two := 14 + 13 + 9,
  have : 100 - 1 = total_three - common_two + 6,
  exact this,
  sorry
}

end students_in_all_three_l191_191833


namespace chain_of_tangent_circles_exists_iff_integer_angle_multiple_l191_191103

noncomputable def angle_between_tangent_circles (R₁ R₂ : Circle) (line : Line) : ℝ :=
-- the definition should specify how we get the angle between the tangent circles
sorry

def n_tangent_circles_exist (R₁ R₂ : Circle) (n : ℕ) : Prop :=
-- the definition should specify the existence of a chain of n tangent circles
sorry

theorem chain_of_tangent_circles_exists_iff_integer_angle_multiple 
  (R₁ R₂ : Circle) (n : ℕ) (line : Line) : 
  n_tangent_circles_exist R₁ R₂ n ↔ ∃ k : ℤ, angle_between_tangent_circles R₁ R₂ line = k * (360 / n) :=
sorry

end chain_of_tangent_circles_exists_iff_integer_angle_multiple_l191_191103


namespace shaded_area_after_3_iterations_l191_191637

def equilateral_triangle (a : ℝ) : Type :=
{ side_length := a }

def midpoint (a b: ℝ × ℝ) : ℝ × ℝ :=
((a.1 + b.1) / 2, (a.2 + b.2) / 2)

def divided_triangle_area (area : ℝ) (iterations : ℕ) : ℝ :=
area / (4 ^ iterations)

theorem shaded_area_after_3_iterations 
  (a : ℝ) (area_initial : ℝ)
  (h_eq_area : area_initial = (8 * 8 * Real.sqrt 3 / 4))
  (P Q R : ℝ × ℝ)
  (hP : P = midpoint (0, 0) (a, 0))
  (hQ : Q = midpoint (a, 0) (a/2, Real.sqrt(3) * a))
  (hR : R = midpoint (a/2, Real.sqrt(3) * a) (0, 0)) :
  divided_triangle_area area_initial 1 +
  divided_triangle_area area_initial 2 +
  divided_triangle_area area_initial 3 = (21 / 4) * Real.sqrt 3 :=
by 
  sorry

end shaded_area_after_3_iterations_l191_191637


namespace max_possible_weight_of_pair_l191_191157

-- Definitions
def total_weight : ℕ := 1000  -- Total weight in grams
def red_apples : ℕ := 10  -- Number of red apples
def yellow_apples : ℕ := 10  -- Number of yellow apples
def max_weight_diff : ℕ := 40  -- Maximum weight difference between any two apples of the same color

-- Statement to be proved
theorem max_possible_weight_of_pair 
  (r w : ℕ → ℕ) (wt_r : ℕ) (wt_y : ℕ): 
  (Σ i, r i + Σ i, w i) = total_weight → 
  (∀ (i j : ℕ), r i - r j ≤ max_weight_diff ∧ w i - w j ≤ max_weight_diff) →
  (∀ (k : ℕ), k < red_apples → r k ≤ r (k + 1)) ∧ (∀ (k : ℕ), k < yellow_apples → w k ≥ w (k + 1)) →
  ∃ p : ℕ × ℕ, 
    (p ∈ list.zip (list.range red_apples) (list.reverse (list.range yellow_apples))) ∧ 
    (r p.1 + w p.2 = 136) := sorry

end max_possible_weight_of_pair_l191_191157


namespace combinatorial_identity_l191_191739

theorem combinatorial_identity (n : ℕ) (h1 : n ≥ 2) (h2 : nat.choose n 2 = 15) : n * (n - 1) = 30 :=
sorry

end combinatorial_identity_l191_191739


namespace balance_proof_l191_191315

variable (a b c : ℕ)

theorem balance_proof (h1 : 5 * a + 2 * b = 15 * c) (h2 : 2 * a = b + 3 * c) : 4 * b = 7 * c :=
sorry

end balance_proof_l191_191315


namespace number_of_polynomials_bounded_by_n_l191_191436

noncomputable def degree (p : Polynomial ℚ) : ℕ := Polynomial.degree p.toNat

theorem number_of_polynomials_bounded_by_n 
  (P : Polynomial ℚ) 
  (h_nonconstant : ∀ x : ℚ, P.derivative.eval x ≠ 0) 
  (h_irreducible : irreducible P) :
  ∃ n : ℕ, (degree P = n) ∧ (∀ Q : Polynomial ℚ, degree Q < n → ∃ m ≤ n, P ∣ (P.compose Q)) :=
sorry

end number_of_polynomials_bounded_by_n_l191_191436


namespace triangle_ABC_reflection_of_incenter_l191_191426

theorem triangle_ABC_reflection_of_incenter 
  (A B C I X Y : Point)
  (hAB : distance A B = 8)
  (hAC : distance A C = 10)
  (hI_incenter : is_incenter I A B C)
  (hX_reflection : reflection I (line_through A B) X)
  (hY_reflection : reflection I (line_through A C) Y)
  (hXY_bisects_AI : bisects XY (segment A I)) :
  distance B C ^ 2 = 84 :=
sorry

end triangle_ABC_reflection_of_incenter_l191_191426


namespace true_statement_count_l191_191281

def reciprocal (n : ℕ) : ℚ := 1 / n

def statement_i := (reciprocal 4 + reciprocal 8 = reciprocal 12)
def statement_ii := (reciprocal 9 - reciprocal 3 = reciprocal 6)
def statement_iii := (reciprocal 3 * reciprocal 9 = reciprocal 27)
def statement_iv := (reciprocal 15 / reciprocal 3 = reciprocal 5)

theorem true_statement_count :
  (¬statement_i ∧ ¬statement_ii ∧ statement_iii ∧ statement_iv) ↔ (2 = 2) :=
by sorry

end true_statement_count_l191_191281


namespace slices_left_per_person_is_2_l191_191098

variables (phil_slices andre_slices small_pizza_slices large_pizza_slices : ℕ)
variables (total_slices_eaten total_slices_left slices_per_person : ℕ)

-- Conditions
def conditions : Prop :=
  phil_slices = 9 ∧
  andre_slices = 9 ∧
  small_pizza_slices = 8 ∧
  large_pizza_slices = 14 ∧
  total_slices_eaten = phil_slices + andre_slices ∧
  total_slices_left = (small_pizza_slices + large_pizza_slices) - total_slices_eaten ∧
  slices_per_person = total_slices_left / 2

theorem slices_left_per_person_is_2 (h : conditions phil_slices andre_slices small_pizza_slices large_pizza_slices total_slices_eaten total_slices_left slices_per_person) :
  slices_per_person = 2 :=
sorry

end slices_left_per_person_is_2_l191_191098


namespace combined_length_of_straight_parts_l191_191232

noncomputable def length_of_straight_parts (R : ℝ) (p : ℝ) : ℝ := p * R

theorem combined_length_of_straight_parts :
  ∀ (R : ℝ) (p : ℝ), R = 80 ∧ p = 0.25 → length_of_straight_parts R p = 20 :=
by
  intros R p h
  cases' h with hR hp
  rw [hR, hp]
  simp [length_of_straight_parts]
  sorry

end combined_length_of_straight_parts_l191_191232


namespace sequence_593_to_598_l191_191336

def arrows : ℕ → char :=
  λ n, ['A', 'B', 'C', 'D', 'E'] ! (n % 5)

theorem sequence_593_to_598 :
  (arrows 593, arrows 594, arrows 595, arrows 596, arrows 597, arrows 598) = 
  ('C', 'D', 'E', 'A', 'B', 'C') :=
by 
  -- we'll use the definition of arrows and properties of modulo arithmetic to prove this statement
  sorry

end sequence_593_to_598_l191_191336


namespace trapezoid_tangent_condition_l191_191031

variable (A B C D E : Point)
variable (a b c d : ℝ)
variable (trapezoid ABCD : ∀ {AB : Line} {CD : Line}, Parallels AB CD)
variable (midpoint_E : E = midpoint B C)
variable (tangent_ABED : is_tangent_quad (ABED : Quadrilateral))
variable (tangent_AECD : is_tangent_quad (AECD : Quadrilateral))

theorem trapezoid_tangent_condition (trapezoid ABCD) (midpoint_E) (tangent_ABED) (tangent_AECD):
  a + c = b / 3 + d ∧ (1 / a) + (1 / c) = 3 / b :=
sorry

end trapezoid_tangent_condition_l191_191031


namespace george_remaining_eggs_l191_191326

theorem george_remaining_eggs (cases boxes_per_case eggs_per_box boxes_sold : ℕ) 
    (h_cases: cases = 7) 
    (h_boxes_per_case: boxes_per_case = 12) 
    (h_eggs_per_box: eggs_per_box = 8) 
    (h_boxes_sold: boxes_sold = 3) : 
  (cases * boxes_per_case * eggs_per_box) - (boxes_sold * eggs_per_box) = 648 := 
by
  rw [h_cases, h_boxes_per_case, h_eggs_per_box, h_boxes_sold]
  norm_num
  sorry

end george_remaining_eggs_l191_191326


namespace arc_length_sector_area_trig_values_l191_191207

-- Conditions
def radius := 10
def alpha_deg := 60
def P := (-4, 3)

-- Unit conversion from degrees to radians (π radians = 180 degrees)
def alpha := (60: ℝ) * (Real.pi / 180)

-- 1. Proving the length of the arc
theorem arc_length : 
  (radius * alpha = (10:ℝ) * ((60:ℝ) * (Real.pi / 180))) :=
  by sorry

-- 2. Proving the area of the sector
theorem sector_area :
  (0.5 * radius^2 * alpha = 0.5 * (10:ℝ)^2 * ((60:ℝ) * (Real.pi / 180))) :=
  by sorry

-- 3. Proving the trigonometric functions of the angle
theorem trig_values :
  (Real.sin alpha = 3 / 5) ∧
  (Real.cos alpha = -4 / 5) ∧
  (Real.tan alpha = -3 / 4) :=
  by sorry

end arc_length_sector_area_trig_values_l191_191207


namespace equilateral_triangle_of_rhombus_l191_191922

theorem equilateral_triangle_of_rhombus {A B C D E F : Type*} 
  [triangle ABC] (h1 : acute_triangle ABC) (h2 : altitude AD ∧ altitude BE ∧ altitude CF)
  (h3 : rhombus FBDE) : equilateral ABC :=
by
  sorry

end equilateral_triangle_of_rhombus_l191_191922


namespace solution_set_of_inequality_l191_191524

theorem solution_set_of_inequality :
  { x : ℝ | x^2 - 5 * x + 6 ≤ 0 } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l191_191524


namespace sequence_property_l191_191340

variable (a : ℕ → ℝ)

theorem sequence_property (h : ∀ n : ℕ, 0 < a n) 
  (h_property : ∀ n : ℕ, (a n)^2 ≤ a n - a (n + 1)) :
  ∀ n : ℕ, a n < 1 / n :=
by
  sorry

end sequence_property_l191_191340


namespace find_z_find_w_modulus_l191_191743

noncomputable def z (b : ℝ) : ℂ := 3 + b * complex.I

theorem find_z (b : ℝ) (hb : 0 < b) (h : (complex.re ((z b - 2)^2)) = 0) : 
  z b = 3 + complex.I :=
by
  sorry

noncomputable def w (b : ℝ) : ℂ := z b / (2 + complex.I)

theorem find_w_modulus (b : ℝ) (hb : 0 < b) (hz : h : (complex.re ((z b - 2)^2)) = 0) : 
  complex.abs (w b) = real.sqrt 2 :=
by
  sorry

end find_z_find_w_modulus_l191_191743


namespace remainder_b_div_11_l191_191065

theorem remainder_b_div_11 (n : ℕ) (h_pos : 0 < n) (b : ℕ) (h_b : b ≡ (5^(2*n) + 6)⁻¹ [ZMOD 11]) : b % 11 = 8 :=
by
  sorry

end remainder_b_div_11_l191_191065


namespace sum_three_numbers_l191_191591

theorem sum_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 23 := by
  sorry

end sum_three_numbers_l191_191591


namespace polar_of_C2_rectangular_of_l_max_distance_to_line_l191_191929

open Complex Real Topology

-- Conditions
def C1_param_x (α : ℝ) : ℝ := 1 + 2 * cos α
def C1_param_y (α : ℝ) : ℝ := sqrt 3 * sin α

def C2_param_x (α : ℝ) : ℝ := (1 + 2 * cos α) / 2
def C2_param_y (α : ℝ) : ℝ := (sqrt 3 * sin α) / 3

-- Problem 1: Proving polar and rectangular equations
theorem polar_of_C2 (α : ℝ) : ∃ ρ θ, (ρ^2 - ρ*cos θ - (3/4) = 0) ∧ 
  ∀ x y, x = (1/2) + cos α ∧ y = sin α → false := 
by
  sorry

theorem rectangular_of_l : ∀ ρ θ, 4*ρ*sin(θ + π/3) + 1 = 0 → 
  ∃ x y, 2*sqrt 3* x + 2* y + 1 = 0 := 
by
  sorry

-- Conditions and structure for Problem 2
def C3 (x y : ℝ) : Prop := (y^2 / 3) + x^2 = 1

theorem max_distance_to_line (P : ℝ × ℝ) (hP : C3 P.1 P.2) :
  ∃ d : ℝ, d = (1 + 2 * sqrt 6) / 4 := 
by
  sorry

end polar_of_C2_rectangular_of_l_max_distance_to_line_l191_191929


namespace anticipated_sedans_l191_191264

theorem anticipated_sedans (sales_sports_cars sedans_ratio sports_ratio sports_forecast : ℕ) 
  (h_ratio : sports_ratio = 5) (h_sedans_ratio : sedans_ratio = 8) (h_sports_forecast : sports_forecast = 35)
  (h_eq : sales_sports_cars = sports_ratio * sports_forecast) :
  sales_sports_cars * 8 / 5 = 56 :=
by
  sorry

end anticipated_sedans_l191_191264


namespace milk_production_days_l191_191816

theorem milk_production_days (x : ℕ) (h : x > 0) :
  let daily_production_per_cow := (x + 1) / (x * (x + 2))
  let total_daily_production := (x + 4) * daily_production_per_cow
  ((x + 7) / total_daily_production) = (x * (x + 2) * (x + 7)) / ((x + 1) * (x + 4)) := 
by
  sorry

end milk_production_days_l191_191816


namespace abs_monotonic_increasing_even_l191_191646

theorem abs_monotonic_increasing_even :
  (∀ x : ℝ, |x| = |(-x)|) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → |x1| ≤ |x2|) :=
by
  sorry

end abs_monotonic_increasing_even_l191_191646


namespace circumscribed_pyramid_volume_circumscribed_pyramid_surface_area_l191_191246

theorem circumscribed_pyramid_volume (r : ℝ) : 
    ∃ V, V = (8 * r^3 * real.pi^2) / (3 * (real.pi - 1)) :=
sorry

theorem circumscribed_pyramid_surface_area (r : ℝ) : 
    ∃ F, F = (8 * r^2 * real.pi^2) / (real.pi - 1) :=
sorry

end circumscribed_pyramid_volume_circumscribed_pyramid_surface_area_l191_191246


namespace find_number_to_add_l191_191571

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l191_191571


namespace measured_diagonal_length_l191_191247

theorem measured_diagonal_length (a b c d diag : Real)
  (h1 : a = 1) (h2 : b = 2) (h3 : c = 2.8) (h4 : d = 5) (hd : diag = 7.5) :
  diag = 2.8 :=
sorry

end measured_diagonal_length_l191_191247


namespace sum_a_coeffs_eq_neg2_sum_abs_a_coeffs_eq_3_exp_n_l191_191595

theorem sum_a_coeffs_eq_neg2 (a : ℕ) (a_coeffs : Fin (a + 1) → ℝ) (n : ℕ) :
  (∑ i in Finset.range (n + 1), a_coeffs ⟨i, by sorry⟩) = -2 :=
  sorry

theorem sum_abs_a_coeffs_eq_3_exp_n (a : ℕ) (a_coeffs : Fin (a + 1) → ℝ) (n : ℕ) :
  (∑ i in Finset.range (n + 1), |a_coeffs ⟨i, by sorry⟩|) = (3 : ℝ) ^ n :=
  sorry

end sum_a_coeffs_eq_neg2_sum_abs_a_coeffs_eq_3_exp_n_l191_191595


namespace proportional_segments_l191_191104

variables {Tetrahedron : Type}
variables (A B C D P : Tetrahedron)
variables (S_ABC S_ABD : ℝ)  
variables (has_edge : (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D))
variables (intersects_CD : ∃ P, P ∈ line_segment C D)
variables (plane_halves_dihedral : bisects_dihedral_plane AB (plane_thru A B P))

theorem proportional_segments : (S_ABC * dist C P) = (S_ABD * dist P D) :=
sorry

end proportional_segments_l191_191104


namespace cathy_time_saving_l191_191274

theorem cathy_time_saving :
  let monday_speed := 6
      tuesday_speed := 5
      thursday_speed := 4
      saturday_speed := 4.5
      distance := 3
      consistent_speed := 5
      monday_time := distance / monday_speed
      tuesday_time := distance / tuesday_speed
      thursday_time := distance / thursday_speed
      saturday_time := distance / saturday_speed
      actual_total_time := monday_time + tuesday_time + thursday_time + saturday_time
      consistent_total_time := distance / consistent_speed * 4
  in (actual_total_time - consistent_total_time) * 60 = 7 :=
by
  let monday_speed := 6
  let tuesday_speed := 5
  let thursday_speed := 4
  let saturday_speed := 4.5
  let distance := 3
  let consistent_speed := 5
  let monday_time := distance / monday_speed
  let tuesday_time := distance / tuesday_speed
  let thursday_time := distance / thursday_speed
  let saturday_time := distance / saturday_speed
  let actual_total_time := monday_time + tuesday_time + thursday_time + saturday_time
  let consistent_total_time := distance / consistent_speed * 4
  have : (actual_total_time - consistent_total_time) * 60 = 7 := sorry
  exact this

end cathy_time_saving_l191_191274


namespace music_commercials_ratio_l191_191685

theorem music_commercials_ratio (T C: ℕ) (hT: T = 112) (hC: C = 40) : (T - C) / C = 9 / 5 := by
  sorry

end music_commercials_ratio_l191_191685


namespace max_sqrt_expr_eq_12_l191_191711

noncomputable def max_value_sqrt_expr : ℝ :=
  real.sup (set.image (λ x : ℝ, real.sqrt (36 + x) + real.sqrt (36 - x)) (set.Icc (-36) 36))

theorem max_sqrt_expr_eq_12 : max_value_sqrt_expr = 12 := by
  sorry

end max_sqrt_expr_eq_12_l191_191711


namespace find_p_l191_191298

theorem find_p (p : ℝ) : 16^4 = (8^3 / 2) * 2^(16 * p) → p = 1/2 :=
by
  sorry

end find_p_l191_191298


namespace proof_problem_l191_191866

section Exercise

variable (θ : ℝ)

def polarCoordEq (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos θ

def toRectCoord (ρ θ : ℝ) : Prop := ρ^2 = (2 * Real.sqrt 2 * Real.cos θ) * ρ

def circleEq (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 2

def parametricLocusC1 (θ : ℝ) : (ℝ × ℝ) := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

theorem proof_problem :
  (∃ ρ θ, polarCoordEq ρ θ → circleEq (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  ∀ θ, ¬ ∃ x y, (circleEq x y) ∧ (parametricLocusC1 θ = (x, y)) :=
by 
  sorry

end Exercise

end proof_problem_l191_191866


namespace age_proof_l191_191494

def current_age (x: ℕ) (a: ℕ) (b: ℕ) (c: ℕ) : Prop :=
  let age_you        := x
  let age_brother    := a
  let age_sister     := b
  let age_youngbro   := c in

  -- Conditions from 10 years ago
  let condition1     := age_brother = 2 * (age_you - 10)
  let condition2     := age_sister = (age_you - 10) / 2
  let condition3     := age_youngbro = age_sister in

  -- Combined age in fifteen years
  let combined_age   := age_you + 15 + (age_brother + 15) + (age_sister + 15) + (age_youngbro + 15)
  let condition4     := combined_age = 110 in

  condition1 ∧ condition2 ∧ condition3 ∧ condition4

theorem age_proof : ∃ (x: ℕ) (a: ℕ) (b: ℕ) (c: ℕ), current_age x a b c ∧ x = 16 :=
by
  -- Proof is omitted
  sorry

end age_proof_l191_191494


namespace _l191_191931

noncomputable def median_theorem (P Q R M : Type) [AddCommGroup P] [Module ℝ P]
  (p q r m n : ℝ) (h1 : QR = p) (h2 : m = n = p / 2) : n / m = r / q :=
by sorry

end _l191_191931


namespace point_outside_circle_l191_191011

variable (A O : Type) [metric_space O] [has_dist A O]

noncomputable def diameter : ℝ := 10
noncomputable def distance_A_to_O : ℝ := 6
noncomputable def radius : ℝ := diameter / 2

theorem point_outside_circle (diameter : ℝ) (distance_A_to_O : ℝ) (radius : ℝ) (h_d : diameter = 10) (h_dA : distance_A_to_O = 6) (h_r : radius = diameter / 2) :
  distance_A_to_O > radius := 
by 
  rw [h_d, h_dA, h_r]
  sorry

end point_outside_circle_l191_191011


namespace max_value_sqrt_expr_max_reaches_at_zero_l191_191720

theorem max_value_sqrt_expr (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) : sqrt (36 + x) + sqrt (36 - x) ≤ 12 :=
by sorry

theorem max_reaches_at_zero : sqrt (36 + 0) + sqrt (36 - 0) = 12 :=
by sorry

end max_value_sqrt_expr_max_reaches_at_zero_l191_191720


namespace ratio_black_bears_to_white_bears_l191_191661

theorem ratio_black_bears_to_white_bears
  (B W Br : ℕ)
  (hB : B = 60)
  (hBr : Br = B + 40)
  (h_total : B + W + Br = 190) :
  B / W = 2 :=
by
  sorry

end ratio_black_bears_to_white_bears_l191_191661


namespace root_in_interval_l191_191337

-- Definitions for conditions
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := (x^2 - 5 * x + 6) * g(x) + x^3 + x - 25

-- Statement of the problem
theorem root_in_interval (g : ℝ → ℝ) (h_cont : Continuous g) :
  ∃ c ∈ set.Ioo 2 3, f g c = 0 :=
sorry

end root_in_interval_l191_191337


namespace surface_area_of_sphere_l191_191756

noncomputable def sphere_surface_area : ℝ :=
  let AB := 2
  let SA := 2
  let SB := 2
  let SC := 2
  let ABC_is_isosceles_right := true -- denotes the property
  let SABC_on_sphere := true -- denotes the property
  let R := (2 * Real.sqrt 3) / 3
  let surface_area := 4 * Real.pi * R^2
  surface_area

theorem surface_area_of_sphere : sphere_surface_area = (16 * Real.pi) / 3 := 
sorry

end surface_area_of_sphere_l191_191756


namespace circumscribed_quadrilateral_l191_191518

theorem circumscribed_quadrilateral (m : ℝ) :
  let l1 := fun x y => mx + y = 4
  let l2 := fun x y => (m + 2) * x - 3 * y = -7
  (∃ x y, l1 x y ∧ y = 4) ∧ (∃ x y, l2 x y ∧ y = 7 / 3) →
  (∃ k : ℝ, k * (-m) * (m + 2) / 3 = -1) → m = 1 ∨ m = -3 :=
by
  sorry

end circumscribed_quadrilateral_l191_191518


namespace least_number_to_add_l191_191183

def required_number (a b : ℕ) : ℕ :=
  let r := a % b
  b - r

theorem least_number_to_add (a b : ℕ) (h : a = 1019) (hb : b = 25) : required_number a b = 6 := by
  simp only [required_number, h, hb]
  rw [Nat.mod_eq_of_lt]
  simp
  exact 6


end least_number_to_add_l191_191183


namespace min_composite_diff_sum_to_105_l191_191601

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

theorem min_composite_diff_sum_to_105 :
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ a + b = 105 ∧ abs (a - b) = 3 :=
by
  sorry

end min_composite_diff_sum_to_105_l191_191601


namespace pages_assigned_l191_191048

theorem pages_assigned (P : ℕ) 
  (h1 : 0.30 * P = 9) : P = 30 :=
by sorry

end pages_assigned_l191_191048


namespace max_value_sqrt_expr_max_reaches_at_zero_l191_191721

theorem max_value_sqrt_expr (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) : sqrt (36 + x) + sqrt (36 - x) ≤ 12 :=
by sorry

theorem max_reaches_at_zero : sqrt (36 + 0) + sqrt (36 - 0) = 12 :=
by sorry

end max_value_sqrt_expr_max_reaches_at_zero_l191_191721


namespace initial_bottles_count_l191_191530

theorem initial_bottles_count : 
  ∀ (jason_buys harry_buys bottles_left initial_bottles : ℕ), 
  jason_buys = 5 → 
  harry_buys = 6 → 
  bottles_left = 24 → 
  initial_bottles = bottles_left + jason_buys + harry_buys → 
  initial_bottles = 35 :=
by
  intros jason_buys harry_buys bottles_left initial_bottles
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end initial_bottles_count_l191_191530


namespace combined_gold_cost_l191_191325

def gary_gold_weight : ℕ := 30
def gary_gold_cost_per_gram : ℕ := 15
def anna_gold_weight : ℕ := 50
def anna_gold_cost_per_gram : ℕ := 20

theorem combined_gold_cost : (gary_gold_weight * gary_gold_cost_per_gram) + (anna_gold_weight * anna_gold_cost_per_gram) = 1450 :=
by {
  sorry -- Proof goes here
}

end combined_gold_cost_l191_191325


namespace third_place_money_l191_191051

theorem third_place_money :
  let people := 8
  let contribution_per_person := 5
  let total_pot := people * contribution_per_person
  let first_place_percentage := 0.80
  let remaining_percentage := 0.20
  let first_place_money := first_place_percentage * total_pot
  let remaining_money := remaining_percentage * total_pot
  let split_remaining_money := remaining_money / 2 in
  split_remaining_money = 4
:= by
  sorry

end third_place_money_l191_191051


namespace median_room_number_of_remaining_players_l191_191028

theorem median_room_number_of_remaining_players (rooms : List ℕ) (h25 : rooms = List.range 25) (h15 : 15 ∈ rooms) (h16 : 16 ∈ rooms) : 
  let remaining_rooms := rooms.erase 15 |>.erase 16
  List.length remaining_rooms = 23 ∧ List.nth remaining_rooms 11 = some 14 :=
by
  sorry

end median_room_number_of_remaining_players_l191_191028


namespace max_correct_answers_l191_191626

theorem max_correct_answers (c w b : ℕ) (h1 : c + w + b = 25) (h2 : 6 * c - 3 * w = 60) : c ≤ 15 :=
begin
  sorry
end


end max_correct_answers_l191_191626


namespace greatest_prime_factor_of_expression_l191_191548

theorem greatest_prime_factor_of_expression :
  let expr := 2^8 + 4^7 in
  let factors := [5, 13] in
  expr.gcd 13 = 13 := by
  sorry

end greatest_prime_factor_of_expression_l191_191548


namespace eight_b_value_l191_191760

theorem eight_b_value (a b : ℝ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 8 * b = 16 :=
by
  sorry

end eight_b_value_l191_191760


namespace C_and_C1_no_common_points_l191_191839

noncomputable def C_equation : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - real.sqrt 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (1, 0)

noncomputable def M_on_C (θ : ℝ) : ℝ × ℝ :=
  let x := 2 * real.sqrt 2 * real.cos θ * real.cos θ in
  let y := 2 * real.sqrt 2 * real.sin θ * real.sin θ in
  (x, y)

noncomputable def P_on_locus (x_M y_M : ℝ) : ℝ × ℝ :=
  let x := real.sqrt 2 * (x_M - point_A.1) + point_A.1 in
  let y := real.sqrt 2 * y_M in
  (x, y)

noncomputable def Locus_P_equation (θ : ℝ) : ℝ × ℝ :=
  let x := 3 - real.sqrt 2 + 2 * real.cos θ in
  let y := 2 * real.sin θ in
  (x, y)

def has_common_points (C1 C2 : set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C1 ∧ p ∈ C2

theorem C_and_C1_no_common_points :
  ¬ has_common_points
    {p | C_equation p}
    {p | ∃ θ, P_on_locus (M_on_C θ).1 (M_on_C θ).2 = p} :=
sorry

end C_and_C1_no_common_points_l191_191839


namespace new_average_of_numbers_l191_191120

theorem new_average_of_numbers 
  (initial_average : ℝ) (initial_count : ℕ) (specified_number : ℝ) (multiplier : ℝ) (new_average : ℝ) 
  (h1 : initial_average = 6.8) 
  (h2 : initial_count = 5) 
  (h3 : specified_number = 6) 
  (h4 : multiplier = 3)
  (h5 : new_average = 9.2) :
  let initial_sum := initial_average * initial_count,
      new_value  := specified_number * multiplier,
      increase_in_sum := new_value - specified_number,
      new_sum := initial_sum + increase_in_sum,
      computed_new_average := new_sum / initial_count in
  computed_new_average = new_average :=
by
  sorry

end new_average_of_numbers_l191_191120


namespace cube_volume_given_surface_area_l191_191156

theorem cube_volume_given_surface_area (SA : ℝ) (a V : ℝ) (h : SA = 864) (h1 : 6 * a^2 = SA) (h2 : V = a^3) : 
  V = 1728 := 
by 
  sorry

end cube_volume_given_surface_area_l191_191156


namespace coat_price_reduction_l191_191024

theorem coat_price_reduction :
  let orig_price := 500
  let first_discount := 0.15 * orig_price
  let price_after_first := orig_price - first_discount
  let second_discount := 0.10 * price_after_first
  let price_after_second := price_after_first - second_discount
  let tax := 0.07 * price_after_second
  let price_with_tax := price_after_second + tax
  let final_price := price_with_tax - 200
  let reduction_amount := orig_price - final_price
  let percent_reduction := (reduction_amount / orig_price) * 100
  percent_reduction = 58.145 :=
by
  sorry

end coat_price_reduction_l191_191024


namespace max_value_after_80_operations_l191_191600

-- Initial state is a 10x10 matrix filled with zeros
def initial_matrix : Matrix (Fin 10) (Fin 10) ℕ :=
  λ _ _, 0

-- Definition of an operation: updating the minimum value and its neighbors
def perform_operation (m : Matrix (Fin 10) (Fin 10) ℕ) : Matrix (Fin 10) (Fin 10) ℕ := sorry

-- Performing a sequence of operations
def perform_operations (m : Matrix (Fin 10) (Fin 10) ℕ) (n : ℕ) : Matrix (Fin 10) (Fin 10) ℕ :=
  nat.rec_on n m (λ _ m_n, perform_operation m_n)

-- The theorem to prove
theorem max_value_after_80_operations : ∀ (m : Matrix (Fin 10) (Fin 10) ℕ),
  ∃ (cell_value: ℕ), cell_value = 20 ∧ 
  (exists i j, (perform_operations initial_matrix 80) i j = cell_value) := sorry

end max_value_after_80_operations_l191_191600


namespace population_relation_l191_191119

-- Conditions: average life expectancies
def life_expectancy_gondor : ℝ := 64
def life_expectancy_numenor : ℝ := 92
def combined_life_expectancy (g n : ℕ) : ℝ := 85

-- Proof Problem: Given the conditions, prove the population relation
theorem population_relation (g n : ℕ) (h1 : life_expectancy_gondor * g + life_expectancy_numenor * n = combined_life_expectancy g n * (g + n)) : n = 3 * g :=
by
  sorry

end population_relation_l191_191119


namespace triangle_statements_correct_l191_191932

theorem triangle_statements_correct (A B C a b c : ℝ)
  (hA_gt_B_iff_sinA_gt_sinB : A > B ↔ sin A > sin B)
  (hB60 : B = 60 ∧ b^2 = a * c → A = 60 ∧ b = a ∧ b = c)
  (hb_eq_a_cosC_plus_c_sinA : b = a * cos C + c * sin A → A = 45) :
  (A > B ↔ sin A > sin B) ∧ (B = 60 ∧ b^2 = a * c → A = 60 ∧ b = a ∧ b = c) ∧ (b = a * cos C + c * sin A → A = 45) :=
by
  sorry

end triangle_statements_correct_l191_191932


namespace count_valid_integers_l191_191806

-- The property of interest
def has_permutation_as_multiple_of_7 (n : ℕ) : Prop :=
  ∃ (m : ℕ), m ≤ 998 ∧ 101 ≤ m ∧ (m % 7 = 0) ∧ (m ∈ (n.digits 10).permutations.map (λ ds, ds.foldl (λ (acc : ℕ) d, 10 * acc + d) 0))

-- The set of all integers between 101 and 998
def valid_integers := {n : ℕ | 101 ≤ n ∧ n ≤ 998}

-- The main theorem stating the result
theorem count_valid_integers : (valid_integers.filter has_permutation_as_multiple_of_7).card = 100 := 
by
  sorry

end count_valid_integers_l191_191806


namespace proof_problem_l191_191906

noncomputable def polar_to_rectangular : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x - sqrt 2) ^ 2 + y ^ 2 = 2) ↔ (∀ θ, rho = 2 * sqrt 2 * Real.cos θ → (x ^ 2 + y ^ 2 = rho ^ 2) ∧ (x = rho * Real.cos θ)) sorry

noncomputable def check_common_points : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x = 3 - sqrt 2 + 2 * Real.cos θ) ∧ (y = 2 * Real.sin θ)) ∧
              ∀ (M_P C_P : ℝ), C_P = 2 ∧ |(3 - sqrt 2) - sqrt 2| = 3 - 2 * sqrt 2 < 2 - sqrt 2 :=
              ¬ ∃ p, (p ∈ C) ∧ (p ∈ C_1) sorry

theorem proof_problem : polar_to_rectangular ∧ check_common_points := by
  sorry

end proof_problem_l191_191906


namespace max_sqrt_expr_eq_12_l191_191713

noncomputable def max_value_sqrt_expr : ℝ :=
  real.sup (set.image (λ x : ℝ, real.sqrt (36 + x) + real.sqrt (36 - x)) (set.Icc (-36) 36))

theorem max_sqrt_expr_eq_12 : max_value_sqrt_expr = 12 := by
  sorry

end max_sqrt_expr_eq_12_l191_191713


namespace number_of_integer_pairs_l191_191391

theorem number_of_integer_pairs (m n : ℕ) (h_pos : m > 0 ∧ n > 0) (h_ineq : m^2 + m * n < 30) :
  ∃ k : ℕ, k = 48 :=
sorry

end number_of_integer_pairs_l191_191391


namespace adult_ticket_cost_is_16_l191_191621

-- Define the problem
def group_size := 6 + 10 -- Total number of people
def child_tickets := 6 -- Number of children
def adult_tickets := 10 -- Number of adults
def child_ticket_cost := 10 -- Cost per child ticket
def total_ticket_cost := 220 -- Total cost for all tickets

-- Prove the cost of an adult ticket
theorem adult_ticket_cost_is_16 : 
  (total_ticket_cost - (child_tickets * child_ticket_cost)) / adult_tickets = 16 := by
  sorry

end adult_ticket_cost_is_16_l191_191621


namespace function_monotonically_increasing_intervals_l191_191374

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 5 * x + 2 * Real.log x

-- State the theorem
theorem function_monotonically_increasing_intervals :
  ∀ x : ℝ, x > 0 → (0 < x ∧ x < 1 / 2 ∨ x > 2 → deriv f x > 0) :=
by
  sorry

end function_monotonically_increasing_intervals_l191_191374


namespace janet_dresses_l191_191936

theorem janet_dresses : 
  ∃ D : ℕ, 
    (D / 2) * (2 / 3) + (D / 2) * (6 / 3) = 32 → D = 24 := 
by {
  sorry
}

end janet_dresses_l191_191936


namespace fill_half_cistern_in_15_minutes_l191_191192

-- Define the condition that a fill pipe can fill 1/2 of the cistern in 15 minutes
def fill_time (half_cistern: ℚ) (minutes: ℕ) : Prop :=
  half_cistern = 1 / 2 ∧ minutes = 15

-- Proof statement to show that the time to fill 1/2 of the cistern is 15 minutes
theorem fill_half_cistern_in_15_minutes : ∀ (half_cistern: ℚ) (minutes: ℕ),
  fill_time half_cistern minutes → minutes = 15 := 
by 
  intro half_cistern minutes
  intro h
  cases h with h1 h2
  exact h2

end fill_half_cistern_in_15_minutes_l191_191192


namespace adam_total_spending_l191_191239

def first_laptop_cost : ℤ := 500
def second_laptop_cost : ℤ := 3 * first_laptop_cost
def total_cost : ℤ := first_laptop_cost + second_laptop_cost

theorem adam_total_spending : total_cost = 2000 := by
  sorry

end adam_total_spending_l191_191239


namespace max_area_of_rectangular_pen_with_fence_l191_191741

-- Definitions
def total_fencing : ℝ := 50
def gate_length : ℝ := 5
def usable_fencing : ℝ := total_fencing - gate_length

-- Area of rectangle given by sides x and (usable_fencing / 2 - x), simplified
def area (x : ℝ) : ℝ := x * (usable_fencing / 2 - x)

-- Proving the maximum area value
theorem max_area_of_rectangular_pen_with_fence : 
  ∃ x : ℝ, ∃ max_area : ℝ, 
  max_area = 126.5625 ∧ 
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ usable_fencing -> area y ≤ max_area) :=
sorry

end max_area_of_rectangular_pen_with_fence_l191_191741


namespace sin_alpha_plus_beta_l191_191331

theorem sin_alpha_plus_beta :
  ∀ (α β : ℝ), (sin α = 2 / 3) ∧ (α ∈ set.Ioo (π / 2) π) ∧ 
              (cos β = -3 / 5) ∧ (β ∈ set.Ioo π (3 * π / 2)) →
              sin (α + β) = (4 * real.sqrt 5 - 6) / 15 :=
by
  intros α β h,
  have h1 : sin α = 2 / 3 := h.1,
  have h2 : α ∈ set.Ioo (π / 2) π := h.2.1,
  have h3 : cos β = -3 / 5 := h.2.2.1,
  have h4 : β ∈ set.Ioo π (3 * π / 2) := h.2.2.2,
  sorry

end sin_alpha_plus_beta_l191_191331


namespace area_triangle_BEF_l191_191996

variable (A B C D E F : Type) [MetricSpace A] 
variable (parallelogram : IsParallelogram A B C D)
variable (area_ABCD : Area A B C D = 48)
variable (midpoint_E : IsMidpoint E A B)
variable (angle_bisector_F : AngleBisector F C)

theorem area_triangle_BEF :
  Area B E F = 6 := sorry

end area_triangle_BEF_l191_191996


namespace rational_eq_reciprocal_l191_191230

theorem rational_eq_reciprocal (x : ℚ) (h : x = 1 / x) : x = 1 ∨ x = -1 :=
by {
  sorry
}

end rational_eq_reciprocal_l191_191230


namespace OPQ_is_isosceles_l191_191757

variables {A B C O H I M D P Q : Type*}

-- Given conditions
axiom acute_triangle (A B C : Type*) : Type*
axiom circumcircle (O : Type*) (A B C : Type*) : Type*
axiom orthocenter (H : Type*) (A B C : Type*) : Type*
axiom incenter (I : Type*) (A B C : Type*) : Type*
axiom midpoint (M : Type*) (A H : Type*) : Type*
axiom ao_parallel_mi : ∀ (A O M I : Type*), (A → O) ∥ (M → I)
axiom ah_intersects_circle (A H D : Type*) (O : Type*) : Type*
axiom line_intersections (A O D : Type*) (B C P Q : Type*) : Type*

-- Theorem stating the question
theorem OPQ_is_isosceles (A B C O H I M D P Q : Type*)
  (h1 : acute_triangle A B C)
  (h2 : circumcircle O A B C)
  (h3 : orthocenter H A B C)
  (h4 : incenter I A B C)
  (h5 : midpoint M A H)
  (h6 : ao_parallel_mi A O M I)
  (h7 : ah_intersects_circle A H D O)
  (h8 : line_intersections A O D B C P Q) : OP = OQ := 
sorry

end OPQ_is_isosceles_l191_191757


namespace polarToRectangular_noCommonPoints_l191_191884

section PolarToRectangular
variables {θ : ℝ} {ρ : ℝ}

-- Condition: polar coordinate equation ρ = 2√2 cos θ
def polarEq (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Correct answer: rectangular coordinate equation (x - √2)² + y² = 2
theorem polarToRectangular 
  (h : ρ = polarEq θ)
  (x y : ℝ)
  (hx : x = ρ * cos θ)
  (hy : y = ρ * sin θ) :
  (x - sqrt 2) ^ 2 + y ^ 2 = 2 :=
sorry

end PolarToRectangular

section LocusAndCommonPoints
variables {x y x₁ y₁ : ℝ} {C C₁ : set (ℝ × ℝ)}

-- Condition: (1, 0) is point A, M is a moving point on circle C, P satisfies AP = √2AM
def pointA := (1, 0 : ℝ) 
def isOnCircleC (M : ℝ × ℝ) : Prop := (M.1 - sqrt 2) ^ 2 + M.2 ^ 2 = 2
def scaledVector (A P M : ℝ × ℝ) : Prop :=
  P.1 - A.1 = sqrt 2 * (M.1 - A.1) ∧ P.2 - A.2 = sqrt 2 * (M.2 - A.2)

-- Correct answer: Circles C and C₁ do not have any common points
theorem noCommonPoints
  (A : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : isOnCircleC M)
  (P : ℝ × ℝ)
  (hP : scaledVector A P M) :
  ∀ P : ℝ × ℝ, ¬ (isOnCircleC P ∧ (P.1 - (3 - sqrt 2)) ^ 2 + P.2 ^ 2 = 4) :=
sorry

end LocusAndCommonPoints

end polarToRectangular_noCommonPoints_l191_191884


namespace convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191896

-- Definitions from the problem
def polar_eq_C (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Conversion to rectangular coordinates
def rectangular_eq_C (x y : ℝ) : Prop := (x - sqrt 2)^2 + y^2 = 2

-- Definitions of points and vectors
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨1, 0⟩

def M (θ : ℝ) : Point :=
  let ρ := polar_eq_C θ
  ⟨ρ * cos θ, ρ * sin θ⟩

def P (x y : ℝ) : Point := ⟨x, y⟩

-- Conditions and transformations
def vector_eq (p1 p2 : Point) (a : ℝ) : Prop :=
  P p1.x p1.y = ⟨a * p2.x + (1 - a) * A.x, a * p2.y + (1 - a) * A.y⟩

def parametric_eq_P (θ : ℝ) : Point :=
  ⟨3 - sqrt 2 + 2 * cos θ, 2 * sin θ⟩

-- Proof problems
theorem convert_polar_to_rectangular (θ : ℝ) :
  rectangular_eq_C (polar_eq_C θ * cos θ) (polar_eq_C θ * sin θ) :=
sorry

theorem parametric_eq_valid (θ : ℝ) :
  vector_eq (parametric_eq_P θ) (M θ) (sqrt 2) :=
sorry

theorem no_common_points :
  ∀ θ₁ θ₂, parametric_eq_P θ₁ ≠ M θ₂ :=
sorry

end convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191896


namespace max_lattice_points_no_centroid_lattice_point_l191_191209

-- Definitions based on conditions
def is_lattice_point (p : ℤ × ℤ) : Prop := true

def is_centroid_lattice_point (p1 p2 p3 p4 : ℤ × ℤ) : Prop :=
  let x_centroid := (p1.1 + p2.1 + p3.1 + p4.1) / 4
  let y_centroid := (p1.2 + p2.2 + p3.2 + p4.2) / 4
  x_centroid % 1 = 0 ∧ y_centroid % 1 = 0

noncomputable def largest_lattice_point_set_not_centroid_lattice_point (n : ℕ) : ℕ :=
  if ∃ S : finset (ℤ × ℤ), S.card = n ∧
    (∀ p1 p2 p3 p4 ∈ S, ¬ is_centroid_lattice_point p1 p2 p3 p4) then n else 0

-- Statement of the theorem/problem
theorem max_lattice_points_no_centroid_lattice_point : ∀ n : ℕ,
  largest_lattice_point_set_not_centroid_lattice_point n ≤ 12 :=
sorry

end max_lattice_points_no_centroid_lattice_point_l191_191209


namespace money_put_in_by_A_l191_191193

theorem money_put_in_by_A 
  (B_capital : ℕ := 25000)
  (total_profit : ℕ := 9600)
  (A_management_fee : ℕ := 10)
  (A_total_received : ℕ := 4200) 
  (A_puts_in : ℕ) :
  (A_management_fee * total_profit / 100 
    + (A_puts_in / (A_puts_in + B_capital)) * (total_profit - A_management_fee * total_profit / 100) = A_total_received)
  → A_puts_in = 15000 :=
  by
    sorry

end money_put_in_by_A_l191_191193


namespace fireflies_remaining_l191_191470

theorem fireflies_remaining
  (initial_fireflies : ℕ)
  (fireflies_joined : ℕ)
  (fireflies_flew_away : ℕ)
  (h_initial : initial_fireflies = 3)
  (h_joined : fireflies_joined = 12 - 4)
  (h_flew_away : fireflies_flew_away = 2)
  : initial_fireflies + fireflies_joined - fireflies_flew_away = 9 := by
  sorry

end fireflies_remaining_l191_191470


namespace boat_speed_ratio_l191_191212

theorem boat_speed_ratio (v_b v_c : ℕ) (hb : v_b = 15) (hc : v_c = 5) : 
  let v_down := v_b + v_c in
  let v_up := v_b - v_c in
  let t_down := 1 / v_down in
  let t_up := 1 / v_up in
  let t := t_down + t_up in
  let avg_speed := 2 / t in
  avg_speed / v_b = 8 / 9 :=
by sorry

end boat_speed_ratio_l191_191212


namespace Barons_theorem_correct_l191_191268

theorem Barons_theorem_correct (a b : ℕ) (ha: 0 < a) (hb: 0 < b) : 
  ∃ n : ℕ, 0 < n ∧ ∃ k1 k2 : ℕ, an = k1 ^ 2 ∧ bn = k2 ^ 3 := 
sorry

end Barons_theorem_correct_l191_191268


namespace pie_count_correct_l191_191275

structure Berries :=
  (strawberries : ℕ)
  (blueberries : ℕ)
  (raspberries : ℕ)

def christine_picking : Berries := {strawberries := 10, blueberries := 8, raspberries := 20}

def rachel_picking : Berries :=
  let c := christine_picking
  {strawberries := 2 * c.strawberries,
   blueberries := 2 * c.blueberries,
   raspberries := c.raspberries / 2}

def total_berries (b1 b2 : Berries) : Berries :=
  {strawberries := b1.strawberries + b2.strawberries,
   blueberries := b1.blueberries + b2.blueberries,
   raspberries := b1.raspberries + b2.raspberries}

def pie_requirements : Berries := {strawberries := 3, blueberries := 2, raspberries := 4}

def max_pies (total : Berries) (requirements : Berries) : Berries :=
  {strawberries := total.strawberries / requirements.strawberries,
   blueberries := total.blueberries / requirements.blueberries,
   raspberries := total.raspberries / requirements.raspberries}

def correct_pies : Berries := {strawberries := 10, blueberries := 12, raspberries := 7}

theorem pie_count_correct :
  let total := total_berries christine_picking rachel_picking;
  max_pies total pie_requirements = correct_pies :=
by {
  sorry
}

end pie_count_correct_l191_191275


namespace sixth_root_of_594823321_l191_191288

theorem sixth_root_of_594823321 :
  \sqrt[6]{(594823321 : ℕ)} = 51 := by
  sorry

end sixth_root_of_594823321_l191_191288


namespace same_number_of_acquaintances_l191_191826

universe u

-- Let G be a Type representing people in the society.
variable (G : Type u)

-- Define the acquaintance relation as a binary relation on G.
variable (acquainted : G → G → Prop)

-- Axiom 1: Any two acquaintances do not have mutual acquaintances.
axiom no_common_acquaintances :
  ∀ {a b : G}, acquainted a b → ∀ c, acquainted a c → acquainted b c → False

-- Axiom 2: Any two strangers have exactly two mutual acquaintances.
axiom two_common_acquaintances :
  ∀ {a b : G},
    ¬ acquainted a b →
    ∃ c d, c ≠ d ∧ acquainted a c ∧ acquainted b c ∧ acquainted a d ∧ acquainted b d

-- The theorem to prove: everyone in the society has the same number of acquaintances.
theorem same_number_of_acquaintances :
  ∀ x y : G, (∃ n : ℕ, (∀ z, acquainted x z ↔ (λ k, with_num_of_friends k = n) z)) :=
by
  intros x y
  -- Fill in the proof steps here
  sorry

end same_number_of_acquaintances_l191_191826


namespace pair_opposite_numbers_l191_191243

theorem pair_opposite_numbers (a : ℝ) : ∃ x y, x = real.cbrt a ∧ y = real.cbrt (-a) ∧ x = -y :=
by
  sorry

end pair_opposite_numbers_l191_191243


namespace inverse_function_d_value_l191_191492

theorem inverse_function_d_value (d : ℝ) 
(h1 : ∀ x, g x = 1 / (3 * x + d)) 
(h2 : ∀ x, g⁻¹ x = (1 - 3 * x) / (3 * x)) :
  d = (3 + Real.sqrt 13) / 2 ∨ d = (3 - Real.sqrt 13) / 2 := 
by
  sorry

end inverse_function_d_value_l191_191492


namespace inheritance_amount_l191_191980

theorem inheritance_amount (x : ℝ) (total_taxes_paid : ℝ) (federal_tax_rate : ℝ) (state_tax_rate : ℝ) 
  (federal_tax_paid : ℝ) (state_tax_base : ℝ) (state_tax_paid : ℝ) 
  (federal_tax_eq : federal_tax_paid = federal_tax_rate * x)
  (state_tax_base_eq : state_tax_base = x - federal_tax_paid)
  (state_tax_eq : state_tax_paid = state_tax_rate * state_tax_base)
  (total_taxes_eq : total_taxes_paid = federal_tax_paid + state_tax_paid) 
  (total_taxes_val : total_taxes_paid = 18000)
  (federal_tax_rate_val : federal_tax_rate = 0.25)
  (state_tax_rate_val : state_tax_rate = 0.15)
  : x = 50000 :=
sorry

end inheritance_amount_l191_191980


namespace B_P_R_Q_concyclic_l191_191774

-- Let the given quadrilateral and the circles be defined
variables {A B C D E P Q R : Type*} 

-- Definitions of the conditions
variables [InCircle Γ (A, B, C, D)] 
          [InThrough AC BD E]
          
variables [LineThrough E (A, B) P]
          [LineThrough E (B, C) Q]
          [Tangent PQ Γ']
          [PassThrough Γ' E D]
          [SecIntersection Γ Γ' R]

-- The proof goal
theorem B_P_R_Q_concyclic 
    (H1 : InCircle Γ (A, B, C, D))
    (H2 : InThrough (AC, BD, E))
    (H3 : LineThrough E (A, B) P)
    (H4 : LineThrough E (B, C) Q)
    (H5 : Tangent PQ Γ')
    (H6 : PassThrough Γ' E D)
    (H7 : SecIntersection Γ Γ' R) : 
    Concyclic (B :: P :: R :: Q :: []) :=
sorry

end B_P_R_Q_concyclic_l191_191774


namespace probability_solution_l191_191258

theorem probability_solution : 
  let N := (3/8 : ℝ)
  let M := (5/8 : ℝ)
  let P_D_given_N := (x : ℝ) => x^2
  (3 : ℝ) * x^2 - (8 : ℝ) * x + (5 : ℝ) = 0 → x = 1 := 
by
  sorry

end probability_solution_l191_191258


namespace union_A_B_intersection_complement_A_B_l191_191793

open Set

variable (A B R : Set ℝ)
variable (x : ℝ)

-- Conditions
def A := {x | 3 ≤ x ∧ x < 7}
def B := {x | x^2 - 12*x + 20 < 0}

-- Questions
theorem union_A_B (A B : Set ℝ) :
  A ∪ B = {x | 2 < x ∧ x < 10} :=
sorry

theorem intersection_complement_A_B (A B : Set ℝ) :
  (compl A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
sorry

-- Proof skipped with sorry

end union_A_B_intersection_complement_A_B_l191_191793


namespace parabola_standard_equation_hyperbola_standard_equation_l191_191128

/-
  Given:
  1. The focus of the parabola is on the x-axis.
  2. The distance from the focus to the directrix is 10.

  Prove:
  1. The standard equation of the parabola is y^2 = 20x.
  2. Given the standard equation of the parabola, the hyperbola with one of its foci at this focus and the conjugate axis length of 8 has the equation (x^2)/41 - (y^2)/16 = 1, and its asymptotes' equations are approximately 7x = ±3√41y
-/

noncomputable def parabola_equation (p : ℝ) : Prop :=
  y^2 = 4 * p * x

theorem parabola_standard_equation :
  ∀ p : ℝ, p = 10 → parabola_equation (1 / 2 * 10) := 
begin
  intro p,
  assume h : p = 10,
  have hp : 2 * (1 / 2 * 10) = 10 := by linarith,
  rw <-h at hp,
  sorry -- proof required
end


theorem hyperbola_standard_equation (p : ℝ) (b : ℝ) :
  ∀ p b : ℝ, p = 10 → b = 4 →
  ((x^2)/(p^2 + b^2) - (y^2)/b^2 = 1) := 
begin
  intros p b hp hb,
  have c := (p / 2) * (p / 2),
  rw [<-hp, <-hb] at *,
  have h_first := (p*p/4 + b*b/4) : (p^2 / 4) = 25,
  sorry -- proof required
end_update

end parabola_standard_equation_hyperbola_standard_equation_l191_191128


namespace planes_intersection_parallel_to_line_l191_191768

structure Line (P : Type) [AffineSpace P] :=
(perp : ∀ (α : Set P), ¬ α ∩ (Set.univ : Set P))

structure Plane (P : Type) [AffineSpace P] :=
(perp_to_line : ∀ (m : Line P), m.perp)

variables (P : Type) [h : AffineSpace P]

def are_skew_lines (m n : Line P) : Prop :=
¬ (∃ p : P, p ∈ m ∧ p ∈ n) ∧ ¬ (∃ (v₁ : Submodule P), v₁ ⊆ m ∧ v₁ ⊆ n)

def is_perp_to_plane {P : Type} [AffineSpace P] (m : Line P) (α : Plane P) : Prop :=
α.perp_to_line m

def plane_intersects (α β : Plane P) : Prop :=
∃ p : P, p ∈ α ∧ p ∈ β

def intersection_line_parallel {P : Type} [AffineSpace P] (α β : Plane P) (l : Line P) : Prop :=
∀ p ∈ (α ∩ β), ∃ direction : Submodule P, p + direction ∈ l

theorem planes_intersection_parallel_to_line 
  (m n l : Line P) 
  (α β : Plane P) 
  (h_skew : are_skew_lines m n)
  (h_perpm_alpha : is_perp_to_plane m α)
  (h_perpn_beta : is_perp_to_plane n β)
  (h_perpl_m : l.perp m)
  (h_perpl_n : l.perp n)
  (h_not_in_alpha : ∀ p ∈ l, ¬ p ∈ α)
  (h_not_in_beta : ∀ p ∈ l, ¬ p ∈ β) : 
  plane_intersects α β ∧ intersection_line_parallel α β l := 
sorry

end planes_intersection_parallel_to_line_l191_191768


namespace harry_total_payment_in_silvers_l191_191382

-- Definitions for the conditions
def spellbook_gold_cost : ℕ := 5
def spellbook_count : ℕ := 5
def potion_kit_silver_cost : ℕ := 20
def potion_kit_count : ℕ := 3
def owl_gold_cost : ℕ := 28
def silver_per_gold : ℕ := 9

-- Translate the total cost to silver
noncomputable def total_cost_in_silvers : ℕ :=
  spellbook_count * spellbook_gold_cost * silver_per_gold + 
  potion_kit_count * potion_kit_silver_cost + 
  owl_gold_cost * silver_per_gold

-- State the theorem
theorem harry_total_payment_in_silvers : total_cost_in_silvers = 537 :=
by
  unfold total_cost_in_silvers
  sorry

end harry_total_payment_in_silvers_l191_191382


namespace robert_books_completed_l191_191106

-- Define the parameters
def reading_speed := 120 -- pages per hour
def book_length := 360 -- pages
def total_reading_time := 8 -- hours
def break_time := 0.25 -- hours per break (15 minutes)

-- Define the problem statement
theorem robert_books_completed :
  ∃ books_completed : ℕ, books_completed = 2 ∧ 
  (let hours_per_book := book_length / reading_speed in
   let num_breaks := total_reading_time - 1 in
   let total_break_time := num_breaks * break_time in
   let effective_reading_time := total_reading_time - total_break_time in
   effective_reading_time / hours_per_book = 2.0833333333333335) :=
by
  -- here the proof would go
  sorry

end robert_books_completed_l191_191106


namespace max_value_sqrt_expression_l191_191724

theorem max_value_sqrt_expression : 
  ∀ (x : ℝ), -36 ≤ x ∧ x ≤ 36 → 
  sqrt (36 + x) + sqrt (36 - x) ≤ 12 :=
by
  -- Proof goes here
  sorry

end max_value_sqrt_expression_l191_191724


namespace monotonic_interval_range_of_m_bounds_on_a_l191_191789

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a*x - 1
noncomputable def g (x : ℝ) : ℝ := (2 - Real.exp 1) * x

theorem monotonic_interval (a : ℝ) (h : a = Real.exp 1) :
  ∀ x : ℝ,
    let h := λ x, f x a - g x in
    ((differentiable_at ℝ h x) →
      ((∀ y : ℝ, y > Real.log 2 → deriv h y > 0) ∧ (∀ y : ℝ, y < Real.log 2 → deriv h y < 0))) :=
sorry

theorem range_of_m (m : ℝ) : 
  let F := λ x, if x ≤ m then f x (Real.exp 1) else g x in
  set.range F = set.univ → 0 ≤ m ∧ m ≤ 1 / (Real.exp 1 - 2) :=
sorry

theorem bounds_on_a (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 ∧ f x₁ a = f x₂ a ∧ abs (x₁ - x₂) ≥ 1) → 
  Real.exp 1 - 1 ≤ a ∧ a ≤ (Real.exp 2 - Real.exp 1) :=
sorry

end monotonic_interval_range_of_m_bounds_on_a_l191_191789


namespace number_of_people_l191_191121

def average_weight_increase (n : ℕ) : Prop :=
  2.5 * ↑n = 25

theorem number_of_people (n : ℕ) (h1 : average_weight_increase n) : n = 10 :=
by sorry

end number_of_people_l191_191121


namespace ellipse_equation_y_coordinate_range_exists_point_on_x_axis_l191_191344

-- Problem (I)
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 1/2 = 1/a) : 
  ∃ (b : ℝ), a = 2 ∧ c = 1 ∧ b = sqrt (a^2 - c^2) ∧ (a = 2 ∧ (frac x^2 / 4 + frac y^2 / 3 = 1) ) := sorry

-- Problem (II)
theorem y_coordinate_range (Q : ℝ) (ell : ∀ x y ∈ ℝ, frac x^2 / 4 + frac y^2 / 3 = 1) (h_slope : 0) :
  Q ∈ set.Icc (- sqrt(3)/12) (sqrt(3)/12) := sorry

-- Problem (III)
theorem exists_point_on_x_axis (m : ℝ) (h_bisect : ∃ M : ℝ×ℝ, (M.1 = m ∧ M.2 = 0) ∧ ∀ x¹ x² ∈ ℝ, x₁ + x₂ = 8k² / 3 + 4k² ∧ x₁ ⋅ x₂ = frac 4k² - 12 / (3 + 4k²) → x₁ ⋅ x₂ - (m + 1) ⋅ (x₁ + x₂) + 2 ⋅ m = 0 ) :
  m = 4 := sorry

end ellipse_equation_y_coordinate_range_exists_point_on_x_axis_l191_191344


namespace conic_section_parabola_l191_191674

theorem conic_section_parabola (x y : ℝ) :
  abs (y - 3) = sqrt ((x + 1)^2 + y^2) ↔ (∃ a b c : ℝ, y = a * x^2 + b * x + c) :=
by {
  sorry
}

end conic_section_parabola_l191_191674


namespace value_of_a_l191_191003

theorem value_of_a (a : ℝ) (h : (1 : ℝ)^2 - 2 * (1 : ℝ) + a = 0) : a = 1 := 
by 
  sorry

end value_of_a_l191_191003


namespace sum_odd_numbers_l191_191543

theorem sum_odd_numbers (n : ℕ) (h : n > 0) : (Finset.range n).sum (λ i, 2 * i + 1) = n * n :=
by sorry

end sum_odd_numbers_l191_191543


namespace incorrect_quotient_l191_191404

theorem incorrect_quotient
    (correct_quotient : ℕ)
    (correct_divisor : ℕ)
    (incorrect_divisor : ℕ)
    (h1 : correct_quotient = 28)
    (h2 : correct_divisor = 21)
    (h3 : incorrect_divisor = 12) :
  correct_divisor * correct_quotient / incorrect_divisor = 49 :=
by
  sorry

end incorrect_quotient_l191_191404


namespace C_and_C1_no_common_points_l191_191841

noncomputable def C_equation : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - real.sqrt 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (1, 0)

noncomputable def M_on_C (θ : ℝ) : ℝ × ℝ :=
  let x := 2 * real.sqrt 2 * real.cos θ * real.cos θ in
  let y := 2 * real.sqrt 2 * real.sin θ * real.sin θ in
  (x, y)

noncomputable def P_on_locus (x_M y_M : ℝ) : ℝ × ℝ :=
  let x := real.sqrt 2 * (x_M - point_A.1) + point_A.1 in
  let y := real.sqrt 2 * y_M in
  (x, y)

noncomputable def Locus_P_equation (θ : ℝ) : ℝ × ℝ :=
  let x := 3 - real.sqrt 2 + 2 * real.cos θ in
  let y := 2 * real.sin θ in
  (x, y)

def has_common_points (C1 C2 : set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C1 ∧ p ∈ C2

theorem C_and_C1_no_common_points :
  ¬ has_common_points
    {p | C_equation p}
    {p | ∃ θ, P_on_locus (M_on_C θ).1 (M_on_C θ).2 = p} :=
sorry

end C_and_C1_no_common_points_l191_191841


namespace PA_perpendicular_to_BC_l191_191649

open Classical

noncomputable def problem_statement (A B C O1 O2 E F G H P : Point) 
  [Incircle O1 (Triangle A B C)]
  [Incircle O2 (Triangle A B C)]
  (TangencyA : TangentPointToTriangleSide O1 A B E)
  (TangencyB : TangentPointToTriangleSide O1 B C F)
  (TangencyC : TangentPointToTriangleSide O1 C A G)
  (TangencyD : TangentPointToTriangleSide O2 A B H)
  (TangencyE : TangentPointToTriangleSide O2 B C E)
  (TangencyF : TangentPointToTriangleSide O2 C A F)
  (Hcollinearity : Collinear [E, G, P])
  (Fcollinearity : Collinear [F, H, P])
  : Prop := Perpendicular (LineThrough P A) (LineThrough B C)

-- To assert the proposition as true:
theorem PA_perpendicular_to_BC : 
  ∀ (A B C O1 O2 E F G H P : Point),
  Incircle O1 (Triangle A B C) →
  Incircle O2 (Triangle A B C) →
  TangentPointToTriangleSide O1 A B E →
  TangentPointToTriangleSide O1 B C F →
  TangentPointToTriangleSide O1 C A G →
  TangentPointToTriangleSide O2 A B H →
  TangentPointToTriangleSide O2 B C E →
  TangentPointToTriangleSide O2 C A F →
  Collinear [E, G, P] →
  Collinear [F, H, P] →
  Perpendicular (LineThrough P A) (LineThrough B C) := 
by 
  sorry

end PA_perpendicular_to_BC_l191_191649


namespace find_number_to_add_l191_191570

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l191_191570


namespace point_in_second_quadrant_l191_191920

theorem point_in_second_quadrant (a : ℝ) : Quadrant (-3, a^2 + 2) = Quadrant.second := 
sorry

end point_in_second_quadrant_l191_191920


namespace balls_into_boxes_l191_191101

theorem balls_into_boxes (balls boxes : ℕ) (h_balls : balls = 4) (h_boxes : boxes = 3) :
  ∑ d in ({(1, 1, 2), (1, 3)} : finset (ℕ × ℕ × ℕ)), 
  (nat.choose balls 2) * (nat.factorial boxes) = 36 :=
by sorry

end balls_into_boxes_l191_191101


namespace smallest_lambda_l191_191231

noncomputable def lambda (α : ℝ) : ℝ := max α 1

theorem smallest_lambda (α : ℝ) (z1 z2 : ℂ) (x : ℝ) 
  (hα : 0 ≤ α) 
  (hz : |z1| ≤ α * |z1 - z2|)
  (hx : 0 ≤ x ∧ x ≤ 1) : 
  |z1 - x * z2| ≤ lambda α * |z1 - z2| := 
begin
  sorry
end

end smallest_lambda_l191_191231


namespace max_value_sqrt_sum_l191_191716

theorem max_value_sqrt_sum (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) : 
  ∃ M, (∀ y, -36 ≤ y ∧ y ≤ 36 → sqrt (36 + y) + sqrt (36 - y) ≤ M) ∧ M = 12 :=
by
  sorry

end max_value_sqrt_sum_l191_191716


namespace complex_flowchart_has_three_structures_l191_191733

def complex_flowchart_decomposable (f : Flowchart) : Prop :=
  ∃ (s1 s2 s3 : Structure), f = ⟦s1, s2, s3⟧

def is_sequence (s : Structure) : Prop := sorry
def is_condition (s : Structure) : Prop := sorry
def is_loop (s : Structure) : Prop := sorry

theorem complex_flowchart_has_three_structures (f : Flowchart) :
  complex_flowchart_decomposable f →
  ∃ s1 s2 s3, is_sequence s1 ∧ is_condition s2 ∧ is_loop s3 :=
sorry

end complex_flowchart_has_three_structures_l191_191733


namespace least_blue_eyed_with_lunch_box_l191_191694

noncomputable def min_students_with_both (students blue_eyes lunch_boxes : ℕ) : ℕ :=
  let without_lunch_boxes := students - lunch_boxes
  blue_eyes - without_lunch_boxes

theorem least_blue_eyed_with_lunch_box
  (students blue_eyes lunch_boxes : ℕ)
  (h_students : students = 35)
  (h_blue_eyes : blue_eyes = 15)
  (h_lunch_boxes : lunch_boxes = 23) :
  min_students_with_both students blue_eyes lunch_boxes = 3 :=
by
  rw [h_students, h_blue_eyes, h_lunch_boxes]
  unfold min_students_with_both
  norm_num
  sorry

end least_blue_eyed_with_lunch_box_l191_191694


namespace prime_ge_5_div_24_l191_191962

theorem prime_ge_5_div_24 (p : ℕ) (hp : Prime p) (hp_ge_5 : p ≥ 5) : 24 ∣ p^2 - 1 := 
sorry

end prime_ge_5_div_24_l191_191962


namespace mabel_total_tomatoes_l191_191969

theorem mabel_total_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 :=
by
  sorry

end mabel_total_tomatoes_l191_191969


namespace equal_sum_seq_value_at_18_l191_191235

-- Define what it means for a sequence to be an equal-sum sequence with a common sum
def equal_sum_seq (a : ℕ → ℤ) (c : ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = c

theorem equal_sum_seq_value_at_18
  (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : equal_sum_seq a 5) :
  a 18 = 3 :=
sorry

end equal_sum_seq_value_at_18_l191_191235


namespace hexagon_area_is_correct_l191_191223

noncomputable def hexagon_area : Real := 
  let r1 : Real := 3 -- radius of the smaller circle
  let r2 : Real := 5 -- radius of the larger circle
  let area_equilateral_triangle (s : Real) : Real := (Math.sqrt 3 / 4) * s ^ 2
  let area_small_triangles := 3 * area_equilateral_triangle r1
  let area_large_triangles := 3 * area_equilateral_triangle r2
  area_small_triangles + area_large_triangles

theorem hexagon_area_is_correct :
  hexagon_area = (51 * Math.sqrt 3) / 2 :=
by
  sorry

end hexagon_area_is_correct_l191_191223


namespace minimum_translation_value_l191_191505

noncomputable def min_translation {m : ℝ} (h : m > 0) : ℝ :=
  let y := λ x : ℝ, sin (2 * x + π / 3)
  ∃ (m : ℝ), (∀ x : ℝ, y x = sin (2 * (x - m) + π / 3)) ∧ 
             ∀ (x : ℝ), sin (2 * (x - m) + π / 3) = sin (2 * (-x - m) + π / 3) ∧ m = 5 * π / 12

theorem minimum_translation_value : min_translation := 
by
  sorry

end minimum_translation_value_l191_191505


namespace pizza_slices_left_per_person_l191_191099

def total_slices (small: Nat) (large: Nat) : Nat := small + large

def total_eaten (phil: Nat) (andre: Nat) : Nat := phil + andre

def slices_left (total: Nat) (eaten: Nat) : Nat := total - eaten

def pieces_per_person (left: Nat) (people: Nat) : Nat := left / people

theorem pizza_slices_left_per_person :
  ∀ (small large phil andre people: Nat),
  small = 8 → large = 14 → phil = 9 → andre = 9 → people = 2 →
  pieces_per_person (slices_left (total_slices small large) (total_eaten phil andre)) people = 2 :=
by
  intros small large phil andre people h_small h_large h_phil h_andre h_people
  rw [h_small, h_large, h_phil, h_andre, h_people]
  /-
  Here we conclude the proof.
  -/
  sorry

end pizza_slices_left_per_person_l191_191099


namespace triangle_PQL_angles_l191_191596

noncomputable def circumcenter (P Q L : Point) : Point := sorry -- Placeholder for circumcenter definition

theorem triangle_PQL_angles 
  {P Q R L M : Point}
  (h1 : angle_bisector QL (triangle PQR))
  (h2 : circumcenter P Q L = M)
  (h3 : symmetric_about M L PQ) :
  angle PLQ = 120 ∧ angle LPQ = 30 ∧ angle PQL = 30 := 
sorry

end triangle_PQL_angles_l191_191596


namespace toy_robot_shipment_l191_191635

-- Define the conditions provided in the problem
def thirty_percent_displayed (total: ℕ) : ℕ := (3 * total) / 10
def seventy_percent_stored (total: ℕ) : ℕ := (7 * total) / 10

-- The main statement to prove: if 70% of the toy robots equal 140, then the total number of toy robots is 200
theorem toy_robot_shipment (total : ℕ) (h : seventy_percent_stored total = 140) : total = 200 :=
by
  -- We will fill in the proof here
  sorry

end toy_robot_shipment_l191_191635


namespace proof_problem_l191_191871

section Exercise

variable (θ : ℝ)

def polarCoordEq (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos θ

def toRectCoord (ρ θ : ℝ) : Prop := ρ^2 = (2 * Real.sqrt 2 * Real.cos θ) * ρ

def circleEq (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 2

def parametricLocusC1 (θ : ℝ) : (ℝ × ℝ) := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

theorem proof_problem :
  (∃ ρ θ, polarCoordEq ρ θ → circleEq (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  ∀ θ, ¬ ∃ x y, (circleEq x y) ∧ (parametricLocusC1 θ = (x, y)) :=
by 
  sorry

end Exercise

end proof_problem_l191_191871


namespace distance_M_to_NF_l191_191622

noncomputable def problem_statement : Prop :=
  let C := {p : ℝ × ℝ | p.2^2 = 4 * p.1} in
  let F := (1, 0 : ℝ) in
  let directrix := {p : ℝ × ℝ | p.1 = -1} in
  let l := {p : ℝ × ℝ | p.2 = √3 * (p.1 - 1)} in
  let M := (3, 2 * √3 : ℝ) in
  let N := (-1, 2 * √3 : ℝ) in
  let NF := {p : ℝ × ℝ | p.2 - 2 * √3 = -(1 / √3) * (p.1 - 1)} in
  let d := abs ((-(1 / √3) * M.1 + M.2 - 2 * √3) / √((1 / √3)^2 + 1)) in
  d = 3 * √3

theorem distance_M_to_NF : problem_statement := sorry

end distance_M_to_NF_l191_191622


namespace solve_ordered_pair_l191_191290

theorem solve_ordered_pair :
    ∃ (x y : ℚ), (3 * x - 4 * y = -6) ∧ (6 * x - 7 * y = 5) ∧
                 (x = 62 / 3) ∧ (y = 17) :=
by
  use 62/3, 17
  split
  { calc
      3 * (62/3) - 4 * 17 = 62 - 4 * 17 : by norm_num
      ... = 62 - 68 : by norm_num
      ... = -6 : by norm_num }
  split
  { calc
      6 * (62/3) - 7 * 17 = 2 * 62 - 7 * 17 : by norm_num
      ... = 124 - 119 : by norm_num
      ... = 5 : by norm_num }
  split 
  rfl
  rfl

end solve_ordered_pair_l191_191290


namespace find_max_value_l191_191775

theorem find_max_value (f : ℝ → ℝ) (h₀ : f 0 = -5) (h₁ : ∀ x, deriv f x = 4 * x^3 - 4 * x) :
  ∃ x, f x = -5 ∧ (∀ y, f y ≤ f x) ∧ x = 0 :=
sorry

end find_max_value_l191_191775


namespace inverse_sqrt_correct_l191_191704

def f (x : ℝ) : ℝ := √x

def f_inv (y : ℝ) := y ^ 2

theorem inverse_sqrt_correct (h : ∀ x, x ≥ 0 → f_inv (f x) = x) : f_inv 2 = 4 :=
by
  have h1 : f (f_inv 2) = sqrt (2^2) := by rfl
  have h2 : sqrt 4 = 2 := by sorry
  have h3 : 2^2 = 4 := by rfl
  show f_inv 2 = 4, from h3

end inverse_sqrt_correct_l191_191704


namespace correspondence1_is_function_l191_191644

/-- Definitions of sets and correspondence functions -/

def A1 := {-1, 0, 1}
def B1 := {0, 1}
def f1 (x : ℤ) := x * x

def A2 := {0, 1 : ℕ}
def B2 := {-1, 0, 1 : ℤ}
noncomputable def f2 (x : ℕ) := if x = 0 then 0 else if x = 1 then 1 else -1

def A3 := ℤ
def B3 := ℚ
def f3 (x : ℤ) := if x = 0 then 0 else (1 : ℚ) / (x : ℚ)

def A4 := ℝ
def B4 := {x : ℝ | x > 0}
def f4 (x : ℝ) := |x|

/-- Proof that correspondence 1 is a function -/
theorem correspondence1_is_function : 
  ∃ f : ℤ → ℤ, ( ∀ x ∈ A1, f x ∈ B1 ) ∧ ( ∀ x1 x2 ∈ A1, f x1 = f x2 → x1 = x2 ) := by
  use f1
  sorry

end correspondence1_is_function_l191_191644


namespace total_oranges_l191_191939

theorem total_oranges (joan_oranges : ℕ) (sara_oranges : ℕ) 
                      (h1 : joan_oranges = 37) 
                      (h2 : sara_oranges = 10) :
  joan_oranges + sara_oranges = 47 := by
  sorry

end total_oranges_l191_191939


namespace no_integer_solutions_system_l191_191995

theorem no_integer_solutions_system :
  ¬(∃ x y z : ℤ, 
    x^6 + x^3 + x^3 * y + y = 147^157 ∧ 
    x^3 + x^3 * y + y^2 + y + z^9 = 157^147) :=
  sorry

end no_integer_solutions_system_l191_191995


namespace event_d_is_certain_l191_191254

theorem event_d_is_certain : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ (3 * x^2 - 8 * x + 5 = 0) ∧ (x = 1) := 
by
  use 1
  split
  -- 0 ≤ x ∧ x ≤ 1 step
  split
  -- proof for 0 ≤ 1
  norm_num
  -- proof for 1 ≤ 1
  norm_num
  -- 3 * x^2 - 8 * x + 5 = 0 step
  split
  norm_num
  ring_nf
  -- x = 1 step
  rfl

end event_d_is_certain_l191_191254


namespace exists_decreasing_lcm_sequence_l191_191293

theorem exists_decreasing_lcm_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j, i < j → a i < a j) ∧ 
    (∀ i : Fin 99, Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))) :=
sorry

end exists_decreasing_lcm_sequence_l191_191293


namespace gcd_permutations_202120222023_l191_191307

def digit_sum (n : Nat) : Nat :=
  n.digits.sum

theorem gcd_permutations_202120222023 :
  ∃ (d : Nat), d = 9 ∧ (∀ p : Nat, p ∈ (Nat.digits_permutations 202120222023) → d ∣ p) :=
begin
  sorry
end

end gcd_permutations_202120222023_l191_191307


namespace average_percent_increase_in_profit_per_car_l191_191266

theorem average_percent_increase_in_profit_per_car
  (N P : ℝ) -- N: Number of cars sold last year, P: Profit per car last year
  (HP1 : N > 0) -- Non-zero number of cars
  (HP2 : P > 0) -- Non-zero profit
  (HProfitIncrease : 1.3 * (N * P) = 1.3 * N * P) -- Total profit increased by 30%
  (HCarDecrease : 0.7 * N = 0.7 * N) -- Number of cars decreased by 30%
  : ((1.3 / 0.7) - 1) * 100 = 85.7 := sorry

end average_percent_increase_in_profit_per_car_l191_191266


namespace smallest_prime_dividing_7pow15_plus_9pow17_l191_191551

theorem smallest_prime_dividing_7pow15_plus_9pow17 :
  Nat.Prime 2 ∧ (∀ p : ℕ, Nat.Prime p → p ∣ (7^15 + 9^17) → 2 ≤ p) :=
by
  sorry

end smallest_prime_dividing_7pow15_plus_9pow17_l191_191551


namespace tan_alpha_div_tan_beta_cos_beta_value_l191_191330

variables (α β : ℝ)

-- Conditions
axiom sin_alpha_plus_beta : sin (α + β) = 4 / 5
axiom sin_alpha_minus_beta : sin (α - β) = 3 / 5

-- Problem statement
theorem tan_alpha_div_tan_beta :
  (tan α / tan β) = 7 :=
sorry

theorem cos_beta_value (h1 : 0 < β) (h2 : β < α) (h3 : α ≤ π / 4) :
  cos β = 7 * sqrt 2 / 10 :=
sorry

end tan_alpha_div_tan_beta_cos_beta_value_l191_191330


namespace seeds_sum_l191_191189

def Bom_seeds : ℕ := 300

def Gwi_seeds : ℕ := Bom_seeds + 40

def Yeon_seeds : ℕ := 3 * Gwi_seeds

def total_seeds : ℕ := Bom_seeds + Gwi_seeds + Yeon_seeds

theorem seeds_sum : total_seeds = 1660 := by
  sorry

end seeds_sum_l191_191189


namespace length_of_shortest_side_l191_191821

-- Define the conditions
def angle_B := Real.pi / 4
def angle_C := Real.pi / 3
def side_c : ℝ := 1

-- Define the problem: length of the shortest side b
theorem length_of_shortest_side : angle_B < angle_C → 
  side_c / Real.sin angle_C = side_c / Real.sin angle_B → 
  ∃ (b : ℝ), b = side_c * Real.sin angle_B / Real.sin angle_C := sorry

end length_of_shortest_side_l191_191821


namespace angle_ADB_eq_angle_FDC_l191_191822

open_locale classical

noncomputable def triangle (A B C : Type) := triangle (A B C)

variables {A B C D E F : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables [ordered_field A] [ordered_field B] [ordered_field C] [ordered_field D] [ordered_field E] [ordered_field F]

variables (A B C D E F : Type) [noncomputable] [fintype]

variables (triangle_ABC : triangle A B C) 
variables (angle_A_eq_90 : ∠A = 90)
variables (D_on_AC : D ∈ AC)
variables (E_on_BD : E ∈ BD)
variables (ext_AE_inter_BC_at_F : ∃ F, (AE).extended ∩ BC = {F})
variables (BE_ratio_ED_eq_2AC_DC : BE/ED = 2*AC/DC)

theorem angle_ADB_eq_angle_FDC : ∠ADB = ∠FDC := 
sorry

end angle_ADB_eq_angle_FDC_l191_191822


namespace car_speed_15_seconds_less_l191_191213

theorem car_speed_15_seconds_less (v : ℝ) : 
  (∀ v, 75 = 3600 / v + 15) → v = 60 :=
by
  intro H
  -- Proof goes here
  sorry

end car_speed_15_seconds_less_l191_191213


namespace total_trees_correct_l191_191416

def apricot_trees : ℕ := 58
def peach_trees : ℕ := 3 * apricot_trees
def total_trees : ℕ := apricot_trees + peach_trees

theorem total_trees_correct : total_trees = 232 :=
by
  sorry

end total_trees_correct_l191_191416


namespace find_root_l191_191249

theorem find_root :
  ∃ x : ℝ, (3 * x^2 - 8 * x + 5 = 0) ∧ (x = 1) :=
begin
  sorry
end

end find_root_l191_191249


namespace angle_alpha_is_90_l191_191933

-- Define the conditions
variables (a : ℝ) (h_a : ℝ)
variables (ratio_a1 : ℝ) (ratio_a2 : ℝ)
hypothesis (h_are_1 : ratio_a1 = 4)
hypothesis (h_are_2 : ratio_a2 = 1)
hypothesis (h_side_length: a = 15)
hypothesis (h_height: h_a = 6)

noncomputable def alpha_angle (a1 a2 h_a : ℝ) : ℝ :=
  let tan_alpha := h_a / a1 in
  let tan_beta := h_a / a2 in
  if tan_alpha = 1/2 ∧ tan_beta = 2 then 90 else 0

-- The main theorem
theorem angle_alpha_is_90 :
  ∀ a1 a2 : ℝ, 
  (4 * a2 = a) →
  (a1 + a2 = a) →
  (h_a / a1 = 1 / 2) →
  (a1 = 4 * a2) →
  (a2 = 3) →
  (alpha_angle a1 a2 h_a = 90)
:=
begin
  intros a1 a2,
  sorry -- Proof not required
end

end angle_alpha_is_90_l191_191933


namespace num_three_person_subcommittees_from_eight_l191_191803

def num_committees (n k : ℕ) : ℕ := (Nat.fact n) / ((Nat.fact k) * (Nat.fact (n - k)))

theorem num_three_person_subcommittees_from_eight (n : ℕ) (h : n = 8) : num_committees n 3 = 56 :=
by
  rw [h]
  sorry

end num_three_person_subcommittees_from_eight_l191_191803


namespace solve_fraction_problem_l191_191556

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l191_191556


namespace perimeter_of_rectangle_l191_191116

theorem perimeter_of_rectangle (L W : ℕ) (A P : ℕ) (hL : L = 15) (hW : W = 20) (hA : A = (L * W)) : P = 2 * (L + W) := 
  by
    have : L * W = 15 * 20 := by rw [hL, hW]
    have : P = 2 * (L + W) := by rw [hL, hW]
    rw [nat.mul_comm] at hA
    sorry

end perimeter_of_rectangle_l191_191116


namespace n_in_terms_of_m_increasing_intervals_l191_191782

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := m * x^3 + n * x^2

theorem n_in_terms_of_m (m n : ℝ) (h : m ≠ 0)
  (tangent_parallel_to_x_axis : (3 * m * 4 + 2 * n * 2) = 0) : n = -3 * m :=
by
  sorry

theorem increasing_intervals (m : ℝ) (h : m ≠ 0) (n_eq_neg3m : n = -3 * m) :
  (∀ x, f m n x = m * x^3 - 3 * m * x^2) →
  if m > 0 then
     ((∀ x, x < 0 ∨ x > 2 → (3 * m * x^2 - 6 * m * x > 0)) ∧
      (∀ x, -(x < 0 ∨ x > 2) → (3 * m * x^2 - 6 * m * x ≤ 0)))
  else
     ((∀ x, (0 < x ∧ x < 2) → (3 * m * x^2 - 6 * m * x > 0)) ∧
      (∀ x, -(0 < x ∧ x < 2)  → (3 * m * x^2 - 6 * m * x ≤ 0))) :=
by
  sorry

end n_in_terms_of_m_increasing_intervals_l191_191782


namespace terminal_side_in_third_quadrant_l191_191394

theorem terminal_side_in_third_quadrant (a : Real) (h1 : sin a + cos a < 0) (h2 : tan a > 0) :
  (π < a ∧ a < 3 * π / 2) :=
by
  sorry

end terminal_side_in_third_quadrant_l191_191394


namespace num_integers_between_800_and_1000_with_sum_22_divisible_by_5_l191_191807

theorem num_integers_between_800_and_1000_with_sum_22_divisible_by_5 : 
  (∃ n1 n2 : ℕ, 800 ≤ n1 ∧ n1 ≤ 1000 ∧ (n1 % 5 = 0) ∧ (n1.digits.sum = 22) ∧
                    800 ≤ n2 ∧ n2 ≤ 1000 ∧ (n2 % 5 = 0) ∧ (n2.digits.sum = 22) ∧
                    (n1 ≠ n2)) ∧ 
  (∀ n : ℕ, 800 ≤ n ∧ n ≤ 1000 ∧ (n % 5 = 0) ∧ (n.digits.sum = 22) → 
             n = 840 ∨ n = 895) :=
by
  sorry

end num_integers_between_800_and_1000_with_sum_22_divisible_by_5_l191_191807


namespace log_function_range_l191_191520

open Real

-- Definition of the function
def f (x : ℝ) : ℝ := log x / log 2 + 3

-- The theorem to be proved
theorem log_function_range : (∀ y : ℝ, ∃ x : ℝ, f x = y) :=
by {
  sorry
}

end log_function_range_l191_191520


namespace roots_product_eq_l191_191077

theorem roots_product_eq
  (a b m p r : ℚ)
  (h₀ : a * b = 3)
  (h₁ : ∀ x, x^2 - m * x + 3 = 0 → (x = a ∨ x = b))
  (h₂ : ∀ x, x^2 - p * x + r = 0 → (x = a + 1 / b ∨ x = b + 1 / a)) : 
  r = 16 / 3 :=
by
  sorry

end roots_product_eq_l191_191077


namespace polynomial_no_extremum_points_l191_191062

variables {R : Type*} [OrderedRing R] {a b x : R}

theorem polynomial_no_extremum_points (f : R → R) [polynomial R] 
  (interv : a ≤ b) : 
  (∃ (f : R → R) [polynomial R], (∀ x ∈ set.Icc a b, x ∉ set_of (λ c, is_extremum f c))) :=
sorry

end polynomial_no_extremum_points_l191_191062


namespace mabel_tomatoes_l191_191977

theorem mabel_tomatoes :
  ∃ (t1 t2 t3 t4 : ℕ), 
    t1 = 8 ∧ 
    t2 = t1 + 4 ∧ 
    t3 = 3 * (t1 + t2) ∧ 
    t4 = 3 * (t1 + t2) ∧ 
    (t1 + t2 + t3 + t4) = 140 :=
by
  sorry

end mabel_tomatoes_l191_191977


namespace select_best_athlete_l191_191153

structure AthletePerformance :=
  (average : ℝ)
  (variance : ℝ)

def Athlete_A : AthletePerformance := ⟨185, 3.6⟩
def Athlete_B : AthletePerformance := ⟨180, 3.6⟩
def Athlete_C : AthletePerformance := ⟨185, 7.4⟩
def Athlete_D : AthletePerformance := ⟨180, 8.1⟩

theorem select_best_athlete (A B C D : AthletePerformance) :
  A = Athlete_A ∧ B = Athlete_B ∧ C = Athlete_C ∧ D = Athlete_D →
  (A.average = max (max A.average B.average) (max C.average D.average) ∧ 
  A.variance = min (min (if A.average = C.average then A.variance else real.infinity) 
                         (if B.average = C.average then B.variance else real.infinity))
                    (min (if C.average = A.average then C.variance else real.infinity) 
                         (if D.average = A.average then D.variance else real.infinity)))
  :=
sorry

end select_best_athlete_l191_191153


namespace sum_of_two_digit_numbers_l191_191576

/-- Given two conditions regarding multiplication mistakes, we prove the sum of the numbers. -/
theorem sum_of_two_digit_numbers
  (A B C D : ℕ)
  (h1 : (10 * A + B) * (60 + D) = 2496)
  (h2 : (10 * A + B) * (20 + D) = 936) :
  (10 * A + B) + (10 * C + D) = 63 :=
by
  -- Conditions and necessary steps for solving the problem would go here.
  -- We're focusing on stating the problem, not the solution.
  sorry

end sum_of_two_digit_numbers_l191_191576


namespace cannot_be_sum_of_two_or_more_consecutive_integers_l191_191187

def is_power_of_two (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

theorem cannot_be_sum_of_two_or_more_consecutive_integers (n : ℕ) :
  (¬∃ k m : ℕ, k ≥ 2 ∧ n = (k * (2 * m + k + 1)) / 2) ↔ is_power_of_two n :=
by
  sorry

end cannot_be_sum_of_two_or_more_consecutive_integers_l191_191187


namespace triangle_inequality_l191_191441

variable {A B C : Type}
variable [euclidean_geometry A]
variable [euclidean_geometry B]
variable [euclidean_geometry C]

-- Definitions of points and properties for A, B, C, and incenter I
def is_acute_triangle (A B C : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] : Prop := sorry
def incenter (A B C : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] : Type := sorry
noncomputable def AI_squared (A B C I : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] [euclidean_geometry I] : real := sorry
noncomputable def BI_squared (A B C I : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] [euclidean_geometry I] : real := sorry
noncomputable def CI_squared (A B C I : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] [euclidean_geometry I] : real := sorry
noncomputable def AB_squared (A B C : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] : real := sorry
noncomputable def BC_squared (A B C : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] : real := sorry
noncomputable def CA_squared (A B C : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] : real := sorry

theorem triangle_inequality (A B C : Type) [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] (I : incenter A B C)
  (h : is_acute_triangle A B C) :
  3 * (AI_squared A B C I + BI_squared A B C I + CI_squared A B C I) >= AB_squared A B C + BC_squared A B C + CA_squared A B C :=
sorry

end triangle_inequality_l191_191441


namespace jellybean_selection_probability_is_correct_l191_191605

noncomputable def jellybeans_probability : ℚ := sorry

theorem jellybean_selection_probability_is_correct :
  let total_jellybeans := 15
  let red_jellybeans := 5
  let blue_jellybeans := 3
  let green_jellybeans := 2
  let white_jellybeans := 5
  let total_pick := 4
  let red_pick := 2
  let green_pick := 1
  let non_red_green_pick := 1
  let possible_outcomes := (Matrix.binom total_jellybeans total_pick)
  let success_red := (Matrix.binom red_jellybeans red_pick)
  let success_green := (Matrix.binom green_jellybeans green_pick)
  let remaining_jellybeans := total_jellybeans - red_jellybeans - green_jellybeans
  let success_non_red_green := (Matrix.binom remaining_jellybeans non_red_green_pick)
  let successful_outcomes := success_red * success_green * success_non_red_green
  (successful_outcomes : ℚ) / (possible_outcomes : ℚ) = 32 / 273 := 
begin
  sorry,
end

end jellybean_selection_probability_is_correct_l191_191605


namespace range_of_k_no_valid_k_l191_191420

noncomputable theory
open_locale classical

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 12*x + 32 = 0

def line_eq (k x : ℝ) : ℝ :=
  k*x + 2

def line_circle_intersect (k : ℝ) : Prop :=
  let discriminant := (4 * (k - 3)^2 - 144 * (1 + k^2)) in
  discriminant > 0

def vector_collinear (x1 x2 y1 y2 : ℝ) : Prop :=
  (x1 + x2) = -3 * (y1 + y2)

theorem range_of_k :
  ∀ k : ℝ,
  line_circle_intersect k → (-3/4 : ℝ) < k ∧ k < (0 : ℝ) :=
begin
  sorry
end

theorem no_valid_k :
  ¬ ∃ k : ℝ, (line_circle_intersect k ∧ ∀ (x1 x2 y1 y2 : ℝ), vector_collinear x1 x2 y1 y2) :=
begin
  sorry
end

end range_of_k_no_valid_k_l191_191420


namespace exists_three_similar_1995_digit_numbers_l191_191462

theorem exists_three_similar_1995_digit_numbers :
  ∃ (N1 N2 N3 : ℕ), 
  (∀ n, N1.digits n = 459459 * (10 ^ (1995 - 1))) ∧
  (∀ n, N2.digits n = 495495 * (10 ^ (1995 - 1))) ∧
  (∀ n, N3.digits n = 954954 * (10 ^ (1995 - 1))) ∧
  (∀ n, N1.digits n ≠ 0) ∧ (∀ n, N2.digits n ≠ 0) ∧ (∀ n, N3.digits n ≠ 0) ∧
  (N1 + N2 = N3) ∧
  (∃ k, ∀ n, N2 = rotate_digits n k N1) ∧
  (∃ k, ∀ n, N3 = rotate_digits n k N2)
:= sorry

end exists_three_similar_1995_digit_numbers_l191_191462


namespace percentage_of_amount_l191_191701

theorem percentage_of_amount :
  (0.25 * 300) = 75 :=
by
  sorry

end percentage_of_amount_l191_191701


namespace jonathan_additional_payment_zero_l191_191941

theorem jonathan_additional_payment_zero :
  ∀ (cost_of_pens : ℝ) (hiro_usd : ℝ) (conversion_rate : ℝ) (jonathan_spent : ℝ),
    cost_of_pens = 15 →
    hiro_usd = 20 →
    conversion_rate = 1.35 →
    jonathan_spent = 3 →
    let hiro_gbp := hiro_usd / conversion_rate in
    let remaining_cost := cost_of_pens - jonathan_spent in
    let additional_payment := remaining_cost - hiro_gbp in
    additional_payment = 0 :=
by
  intros cost_of_pens hiro_usd conversion_rate jonathan_spent
  intros h1 h2 h3 h4
  dsimp [hiro_gbp, remaining_cost, additional_payment]
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jonathan_additional_payment_zero_l191_191941


namespace tetrahedron_triangle_area_l191_191102

theorem tetrahedron_triangle_area {a b c a' b' c' V R : ℝ} :
  ∃ S : ℝ,
  (∀ (AD BD CD BC CA AB : ℝ), AD = a ∧ BD = b ∧ CD = c ∧ BC = a' ∧ CA = b' ∧ AB = c' ∧
   (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → (AD * BC = S) ∧ (BD * CA = S) ∧ (CD * AB = S)) ∧
  (∀ V R : ℝ, V = V ∧ R = R ∧ S = 6 * V * R) :=
begin
  sorry
end

end tetrahedron_triangle_area_l191_191102


namespace number_of_ways_to_form_valid_number_l191_191041

-- Define the available digits
def available_digits : List ℕ := [0, 2, 4, 7, 8, 9]

-- Define the fixed part of the number
def fixed_part : List ℕ := [2, 0, 1, 6, 0, 0, 2]

-- Define a function to check if a number is divisible by 6
def divisible_by_6 (n : ℕ) : Prop :=
  n % 6 = 0

-- Define a function to create the 11-digit number from the fixed part and a list of 5 digits
def form_number (ds : List ℕ) : ℕ :=
  let digits := fixed_part ++ ds
  digits.foldl (λ acc d => acc * 10 + d) 0

-- Define the final problem statement: prove there are 1728 ways to form such a number
theorem number_of_ways_to_form_valid_number : 
  { ds : List ℕ // ds.length = 5 ∧ (∀ d ∈ ds, d ∈ available_digits) ∧ divisible_by_6 (form_number ds) }.card = 1728 :=
sorry

end number_of_ways_to_form_valid_number_l191_191041


namespace polarToRectangular_noCommonPoints_l191_191889

section PolarToRectangular
variables {θ : ℝ} {ρ : ℝ}

-- Condition: polar coordinate equation ρ = 2√2 cos θ
def polarEq (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Correct answer: rectangular coordinate equation (x - √2)² + y² = 2
theorem polarToRectangular 
  (h : ρ = polarEq θ)
  (x y : ℝ)
  (hx : x = ρ * cos θ)
  (hy : y = ρ * sin θ) :
  (x - sqrt 2) ^ 2 + y ^ 2 = 2 :=
sorry

end PolarToRectangular

section LocusAndCommonPoints
variables {x y x₁ y₁ : ℝ} {C C₁ : set (ℝ × ℝ)}

-- Condition: (1, 0) is point A, M is a moving point on circle C, P satisfies AP = √2AM
def pointA := (1, 0 : ℝ) 
def isOnCircleC (M : ℝ × ℝ) : Prop := (M.1 - sqrt 2) ^ 2 + M.2 ^ 2 = 2
def scaledVector (A P M : ℝ × ℝ) : Prop :=
  P.1 - A.1 = sqrt 2 * (M.1 - A.1) ∧ P.2 - A.2 = sqrt 2 * (M.2 - A.2)

-- Correct answer: Circles C and C₁ do not have any common points
theorem noCommonPoints
  (A : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : isOnCircleC M)
  (P : ℝ × ℝ)
  (hP : scaledVector A P M) :
  ∀ P : ℝ × ℝ, ¬ (isOnCircleC P ∧ (P.1 - (3 - sqrt 2)) ^ 2 + P.2 ^ 2 = 4) :=
sorry

end LocusAndCommonPoints

end polarToRectangular_noCommonPoints_l191_191889


namespace calculate_expression_l191_191658

theorem calculate_expression :
  -1 ^ 2023 + (Real.pi - 3.14) ^ 0 + |(-2 : ℝ)| = 2 :=
by
  sorry

end calculate_expression_l191_191658


namespace find_root_l191_191251

theorem find_root :
  ∃ x : ℝ, (3 * x^2 - 8 * x + 5 = 0) ∧ (x = 1) :=
begin
  sorry
end

end find_root_l191_191251


namespace sufficient_conditions_for_perpendicular_l191_191950

noncomputable def planes_and_lines (α β γ : Plane) (m n : Line) : Prop :=
(α ∩ β = n ∧ m ⊥ n) ∨
(α ∩ γ = m ∧ β ⊥ α ∧ β ⊥ γ) ∨
(α ∥ γ ∧ m ∥ γ) ∨
(n ⊥ α ∧ n ⊥ β ∧ m ⊥ α)

theorem sufficient_conditions_for_perpendicular (α β γ : Plane) (m n : Line)
  (h : planes_and_lines α β γ m n) : (m ⊥ β) :=
by
  sorry

end sufficient_conditions_for_perpendicular_l191_191950


namespace add_to_frac_eq_l191_191565

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l191_191565


namespace compare_f_values_l191_191373

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2 * Real.cos x

theorem compare_f_values :
  f 0 < f (-1 / 3) ∧ f (-1 / 3) < f (2 / 5) :=
by
  sorry

end compare_f_values_l191_191373


namespace kai_walking_speed_l191_191942

def trackCircumferenceDifference(radius_inner radius_outer : ℝ) :=
  2 * π * radius_outer - 2 * π * radius_inner

theorem kai_walking_speed (r_inner r_outer : ℝ) (r_outer_eq : r_outer = r_inner + 6)
                          (time_diff : ℝ) (time_diff_eq : time_diff = 180) :
  let circ_diff := trackCircumferenceDifference r_inner r_outer in
  circ_diff = 12 * π →
  let speed := circ_diff / time_diff in
  speed = π / 15 :=
by
  sorry

end kai_walking_speed_l191_191942


namespace find_line_equation_l191_191007

def is_isosceles_right_triangle (A B C : Point) : Prop :=
  -- Assuming A, B, and C form an isosceles right triangle condition (defined adequately if needed)
  sorry

structure Point :=
  (x : ℝ)
  (y : ℝ)

def passes_through (l : ℝ → ℝ → Prop) (P : Point) : Prop :=
  l P.x P.y

def line_equation (a b c : ℝ) : ℝ → ℝ → Prop := 
  λ x y, a * x + b * y + c = 0

theorem find_line_equation (P : Point)
  (hP : P = ⟨2, 3⟩)
  (H1 : ∃ l, passes_through l P ∧ (is_isosceles_right_triangle (Point.mk 0 0) (Point.mk (l 0 0) (l 1 0)) P)) :
  (∃ a b c, (line_equation a b c) = λ x y, x + y - 5) ∨
  (∃ a b c, (line_equation a b c) = λ x y, x - y + 1) :=
begin
  sorry
end

end find_line_equation_l191_191007


namespace proof_problem_l191_191902

noncomputable def polar_to_rectangular : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x - sqrt 2) ^ 2 + y ^ 2 = 2) ↔ (∀ θ, rho = 2 * sqrt 2 * Real.cos θ → (x ^ 2 + y ^ 2 = rho ^ 2) ∧ (x = rho * Real.cos θ)) sorry

noncomputable def check_common_points : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x = 3 - sqrt 2 + 2 * Real.cos θ) ∧ (y = 2 * Real.sin θ)) ∧
              ∀ (M_P C_P : ℝ), C_P = 2 ∧ |(3 - sqrt 2) - sqrt 2| = 3 - 2 * sqrt 2 < 2 - sqrt 2 :=
              ¬ ∃ p, (p ∈ C) ∧ (p ∈ C_1) sorry

theorem proof_problem : polar_to_rectangular ∧ check_common_points := by
  sorry

end proof_problem_l191_191902


namespace cookies_per_batch_l191_191087

theorem cookies_per_batch (students : ℕ) (cookies_per_student : ℕ) (chocolate_batches : ℕ) (oatmeal_batches : ℕ) (additional_batches : ℕ) (cookies_needed : ℕ) (dozens_per_batch : ℕ) :
  (students = 24) →
  (cookies_per_student = 10) →
  (chocolate_batches = 2) →
  (oatmeal_batches = 1) →
  (additional_batches = 2) →
  (cookies_needed = students * cookies_per_student) →
  dozens_per_batch * (12 * (chocolate_batches + oatmeal_batches + additional_batches)) = cookies_needed →
  dozens_per_batch = 4 :=
by
  intros
  sorry

end cookies_per_batch_l191_191087


namespace discount_rate_on_pony_jeans_l191_191196

theorem discount_rate_on_pony_jeans 
  (F P : ℝ) 
  (H1 : F + P = 22) 
  (H2 : 45 * F + 36 * P = 882) : 
  P = 12 :=
by
  sorry

end discount_rate_on_pony_jeans_l191_191196


namespace purchasing_methods_count_l191_191217

theorem purchasing_methods_count :
  ∃ n, n = 6 ∧
    ∃ (x y : ℕ), 
      60 * x + 70 * y ≤ 500 ∧
      x ≥ 3 ∧
      y ≥ 2 :=
sorry

end purchasing_methods_count_l191_191217


namespace polar_to_rectangular_eq_no_common_points_l191_191849

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_rectangular_eq (ρ θ x y: ℝ) (h1: ρ = 2 * Real.sqrt 2 * Real.cos θ) 
(h2: polar_to_rectangular ρ θ = (x, y)) : (x - Real.sqrt 2) ^ 2 + y ^ 2 = 2 :=
by
  sorry

theorem no_common_points (x y θ : ℝ) (ρ : ℝ := 2 * Real.sqrt 2 * Real.cos θ)
(A := (1, 0) : ℝ × ℝ)
(M := polar_to_rectangular ρ θ)
(P := (sqrt 2 * (M.1 - 1) + 1, sqrt 2 * M.2)) : 
((P.1 - (3 - Real.sqrt 2)) ^ 2 + P.2 ^ 2 = 4) → 
¬ ((P.1 - Real.sqrt 2) ^ 2 + P.2 ^ 2 = 2) :=
by
  intro H1 H2
  sorry

end polar_to_rectangular_eq_no_common_points_l191_191849


namespace repeating_decimal_denominators_l191_191110

theorem repeating_decimal_denominators (a b c : ℕ) (h_digits : a < 10 ∧ b < 10 ∧ c < 10)
  (h_not_all_nine : a ≠ 9 ∨ b ≠ 9 ∨ c ≠ 9) 
  (h_not_all_zero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  (0.abc (h_digits : a < 10 ∧ b < 10 ∧ c < 10) := by sorry) where
  quotient_denom_count := 7 :=
by sorry

end repeating_decimal_denominators_l191_191110


namespace marla_adds_white_paint_l191_191452

-- Define the conditions as hypotheses.
variables (total_percent blue_percent red_percent white_percent proportion_of_blue x : ℕ)
variable (total_ounces : ℕ)
hypothesis (H1 : total_percent = 100)
hypothesis (H2 : blue_percent = 70)
hypothesis (H3 : red_percent = 20)
hypothesis (H4 : white_percent = total_percent - blue_percent - red_percent)
hypothesis (H5 : total_ounces = 140)
hypothesis (H6 : blue_percent * x = white_percent * total_ounces)

-- The problem statement
theorem marla_adds_white_paint : 
  blue_percent * x = white_percent * total_ounces → 
  (x = 20)
:= sorry

end marla_adds_white_paint_l191_191452


namespace cyclist_final_speed_l191_191616

def u : ℝ := 16
def a : ℝ := 0.5
def t : ℕ := 7200

theorem cyclist_final_speed : 
  (u + a * t) * 3.6 = 13017.6 := by
  sorry

end cyclist_final_speed_l191_191616


namespace probability_A_join_street_dance_and_B_join_calligraphy_or_photography_l191_191234

/-- There are 4 clubs (drama, calligraphy, photography, street dance) each with 1 available spot.
    There are 4 students (named A and B among them) who want to join one of the clubs. 
    Each student can only join one club. 
    The probability that A will join the street dance club and B will join either the calligraphy
    club or the photography club is 1/6. -/
theorem probability_A_join_street_dance_and_B_join_calligraphy_or_photography :
    ∃ (A B : ℕ) (clubs : finset (ℕ × ℕ)), 
    clubs.card = 24 ∧ 
    ∃ favorable_outcomes : finset (ℕ × ℕ), 
    favorable_outcomes.card = 4 ∧ 
    ∃ total_outcomes : finset (ℕ × ℕ),
    total_outcomes.card = 24 ∧ 
    (favorable_outcomes.card : ℚ) / (total_outcomes.card : ℚ) = 1 / 6 := 
sorry

end probability_A_join_street_dance_and_B_join_calligraphy_or_photography_l191_191234


namespace quadratic_root_interval_l191_191357

theorem quadratic_root_interval 
(a b c : ℝ) 
(h1 : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ax₁ ^ 2 + bx₁ + c = 0 ∧ ax₂ ^ 2 + bx₂ + c = 0) 
(h2 : abs (a * b - a * c) > abs (b ^ 2 - a * c) + abs (a * b - c ^ 2)) 
: ∃ x ∈ Ioc 0 2, a * x ^ 2 + b * x + c = 0 ∧ ∀ y ∈ Ioc 0 2, a * y ^ 2 + b * y + c = 0 → y = x := 
sorry

end quadratic_root_interval_l191_191357


namespace factorize_expression_l191_191692

variable {a x y : ℝ}

theorem factorize_expression : ax^2 - 16ay^2 = a * (x + 4 * y) * (x - 4 * y) :=
by
  -- Placeholder for the actual proof
  sorry

end factorize_expression_l191_191692


namespace arc_length_of_curve_is_sqrt_2_l191_191652

noncomputable def f (x : ℝ) : ℝ := 1 + Real.arcsin x - Real.sqrt (1 - x^2)

noncomputable def f' (x : ℝ) : ℝ := (1 + x) / Real.sqrt (1 - x^2)

noncomputable def arc_length (a b : ℝ) (f' : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt(1 + (f' x)^2)

theorem arc_length_of_curve_is_sqrt_2 : arc_length 0 (3 / 4) f' = Real.sqrt 2 := 
by 
  sorry

end arc_length_of_curve_is_sqrt_2_l191_191652


namespace locus_center_C_l191_191662
-- We need to import the necessary library for Lean

-- Definitions for the given circle equations
def C1_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 + 4 * y + 3 = 0
def C2_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 4 * y - 77 = 0

-- Noncomputable definition for the locus equation
noncomputable def locus_eq : ℝ → ℝ → Prop := 
  λ x y, y^2 / 25 + x^2 / 21 = 1

-- The theorem stating the equivalence
theorem locus_center_C (x y : ℝ) :
  (C1_eq 0 (-2) ∧ C2_eq 0 2) → (locus_eq x y) :=
begin
  sorry
end

end locus_center_C_l191_191662


namespace number_of_true_propositions_l191_191791

theorem number_of_true_propositions (p q : Prop) (h1 : 3 > 1) (h2 : 4 ∈ {2, 3}) :
  (if p ∧ q then 1 else 0) + (if p ∨ q then 1 else 0) + (if ¬p then 1 else 0) = 1 :=
by
  sorry

end number_of_true_propositions_l191_191791


namespace log_exponent_inequality_l191_191393

theorem log_exponent_inequality {x y : ℝ} 
  (h : (real.log 3 / real.log 2) ^ x - (real.log 3 / real.log 5) ^ x 
       ≥ (real.log 3 / real.log 2) ^ (-y) - (real.log 3 / real.log 5) ^ (-y)) : 
  x + y ≥ 0 := 
sorry

end log_exponent_inequality_l191_191393


namespace FourPointsConcyclic_l191_191058

-- Define the conditions of the problem
variables (a b : ℝ) (P : ℝ × ℝ)
let x0 := P.1
let y0 := P.2

variables (l1 l2 : ℝ → ℝ × ℝ)
let alpha beta : ℝ
let A B C D : ℝ × ℝ

-- Two lines through P with given angles
def l1 (t : ℝ) := (x0 + t * Real.cos alpha, y0 + t * Real.sin alpha)
def l2 (p : ℝ) := (x0 + p * Real.cos beta, y0 + p * Real.sin beta)

-- Condition on angles
axiom angle_cond : alpha + beta = Real.pi

-- Lines intersect ellipse at points A, B, C, D respectively
axiom inter1 : ∃ t : ℝ, l1 t = A ∧ ∃ t' : ℝ, l1 t' = B
axiom inter2 : ∃ p : ℝ, l2 p = C ∧ ∃ p' : ℝ, l2 p' = D

-- Definition of concyclic points
def are_concyclic (A B C D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ) (R : ℝ), 
  ∀ (P : ℝ × ℝ), P = A ∨ P = B ∨ P = C ∨ P = D → ((P.1 - O.1)^2 + (P.2 - O.2)^2 = R^2)

-- The main theorem
theorem FourPointsConcyclic (a b : ℝ) (P : ℝ × ℝ) (l1 l2 : ℝ → ℝ × ℝ) (alpha beta : ℝ) (A B C D : ℝ × ℝ)
    (angle_cond : alpha + beta = Real.pi)
    (inter1 : ∃ t : ℝ, l1 t = A ∧ ∃ t' : ℝ, l1 t' = B)
    (inter2 : ∃ p : ℝ, l2 p = C ∧ ∃ p' : ℝ, l2 p' = D) :
  are_concyclic A B C D :=
  sorry

end FourPointsConcyclic_l191_191058


namespace math_problem_statements_are_correct_l191_191737

theorem math_problem_statements_are_correct (a b : ℝ) (h : a > b ∧ b > 0) :
  (¬ (b / a > (b + 3) / (a + 3))) ∧ ((3 * a + 2 * b) / (2 * a + 3 * b) < a / b) ∧
  (¬ (2 * Real.sqrt a < Real.sqrt (a - b) + Real.sqrt b)) ∧ 
  (Real.log ((a + b) / 2) > (Real.log a + Real.log b) / 2) :=
by
  sorry

end math_problem_statements_are_correct_l191_191737


namespace num_positive_integer_solutions_l191_191516

theorem num_positive_integer_solutions : 
  ∃ n : ℕ, (∀ x : ℕ, x ≤ n → x - 1 < Real.sqrt 5) ∧ n = 3 :=
by
  sorry

end num_positive_integer_solutions_l191_191516


namespace add_to_fraction_l191_191559

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l191_191559


namespace eval_sixty_four_cubed_root_l191_191296

theorem eval_sixty_four_cubed_root : (64 : ℝ)^(1/3) = 4 :=
by
  have h : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [h, ←rpow_mul (show (0 : ℝ) < 2 from by norm_num)]
  norm_num
  sorry

end eval_sixty_four_cubed_root_l191_191296


namespace proof_problem_l191_191872

section Exercise

variable (θ : ℝ)

def polarCoordEq (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos θ

def toRectCoord (ρ θ : ℝ) : Prop := ρ^2 = (2 * Real.sqrt 2 * Real.cos θ) * ρ

def circleEq (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 2

def parametricLocusC1 (θ : ℝ) : (ℝ × ℝ) := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

theorem proof_problem :
  (∃ ρ θ, polarCoordEq ρ θ → circleEq (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  ∀ θ, ¬ ∃ x y, (circleEq x y) ∧ (parametricLocusC1 θ = (x, y)) :=
by 
  sorry

end Exercise

end proof_problem_l191_191872


namespace triangle_right_if_sum_of_radii_eq_BC_l191_191967

-- Define the structure of the triangle and its properties
structure Triangle :=
  (a b c : ℝ)
  (semi_perimeter: ℝ := (a + b + c) / 2)
  (P : ℝ := Math.sqrt(semi_perimeter * (semi_perimeter - a) * (semi_perimeter - b) * (semi_perimeter - c)))
  (r : ℝ := P / semi_perimeter)
  (r_a : ℝ := P / (semi_perimeter - a))

-- Define the triangle being a right triangle
def Triangle.is_right (T : Triangle) : Prop :=
  T.a^2 + T.b^2 = T.c^2 ∨ T.a^2 + T.c^2 = T.b^2 ∨ T.b^2 + T.c^2 = T.a^2

-- Define the required proof statement
theorem triangle_right_if_sum_of_radii_eq_BC (a b c : ℝ) :
  let T : Triangle := { a := a, b := b, c := c } in
  (T.r + T.r_a = T.a) → T.is_right :=
by
  -- Pending proof
  sorry

end triangle_right_if_sum_of_radii_eq_BC_l191_191967


namespace range_of_function_l191_191519

noncomputable def function_range : set ℝ :=
  {y | ∃ x, y = cos x ^ 2 - 4 * sin x}

theorem range_of_function : function_range = set.Icc (-4 : ℝ) (4 : ℝ) :=
by
  sorry

end range_of_function_l191_191519


namespace convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191892

-- Definitions from the problem
def polar_eq_C (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Conversion to rectangular coordinates
def rectangular_eq_C (x y : ℝ) : Prop := (x - sqrt 2)^2 + y^2 = 2

-- Definitions of points and vectors
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨1, 0⟩

def M (θ : ℝ) : Point :=
  let ρ := polar_eq_C θ
  ⟨ρ * cos θ, ρ * sin θ⟩

def P (x y : ℝ) : Point := ⟨x, y⟩

-- Conditions and transformations
def vector_eq (p1 p2 : Point) (a : ℝ) : Prop :=
  P p1.x p1.y = ⟨a * p2.x + (1 - a) * A.x, a * p2.y + (1 - a) * A.y⟩

def parametric_eq_P (θ : ℝ) : Point :=
  ⟨3 - sqrt 2 + 2 * cos θ, 2 * sin θ⟩

-- Proof problems
theorem convert_polar_to_rectangular (θ : ℝ) :
  rectangular_eq_C (polar_eq_C θ * cos θ) (polar_eq_C θ * sin θ) :=
sorry

theorem parametric_eq_valid (θ : ℝ) :
  vector_eq (parametric_eq_P θ) (M θ) (sqrt 2) :=
sorry

theorem no_common_points :
  ∀ θ₁ θ₂, parametric_eq_P θ₁ ≠ M θ₂ :=
sorry

end convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191892


namespace fifth_rollercoaster_speed_l191_191534

theorem fifth_rollercoaster_speed (s1 s2 s3 s4 avg_speed s5 : ℕ)
  (h1 : s1 = 50)
  (h2 : s2 = 62)
  (h3 : s3 = 73)
  (h4 : s4 = 70)
  (h_avg : avg_speed = 59) :
  s5 = 40 :=
by {
  -- Calculate the total expected speed from the average speed
  have h_total := avg_speed * 5,
  -- Substitute the given average speed
  rw h_avg at h_total,
  have h_total : h_total = 295 := by simp,

  -- Sum the four known speeds and compare to the total expected speed
  have h_s := s1 + s2 + s3 + s4,
  rw [h1, h2, h3, h4] at h_s,
  have h_s : h_s = 255 := by simp,

  -- Calculate the speed of the fifth rollercoaster
  have h5 := h_total - h_s,
  have h5 : h5 = 40 := by simp,

  -- Conclude that the fifth rollercoaster speed is 40
  exact h5,
}

end fifth_rollercoaster_speed_l191_191534


namespace find_m_l191_191353

theorem find_m (x m : ℝ) (h1 : log 10 (tan x) + log 10 (cos x) = -1)
  (h2 : log 10 (tan x + cos x) = 1/2 * (log 10 m - 1)) : m = 1.2 :=
sorry

end find_m_l191_191353


namespace k_correct_l191_191415

-- Definitions for the problem
def right_triangle (A B C : Type) [metric_space A] (AB BC AC : ℝ) (h : AB^2 + BC^2 = AC^2) : Prop := 
h = (6^2 + 8^2 = 10^2)

def angle_bisector_length_k (BD : ℝ) (k : ℝ) (h : BD = k * real.sqrt 2) : Prop :=
h = (BD = k * real.sqrt 2)

noncomputable def k_value : ℝ := 26 / 7

theorem k_correct {A B C D : Type} [metric_space A] :
  right_triangle A B C 6 8 10 (by norm_num) ∧
  angle_bisector_length_k (52/7) k_value (by norm_num) →
  k_value = 26 / 7 :=
by intros; sorry

end k_correct_l191_191415


namespace max_value_sqrt_sum_l191_191718

theorem max_value_sqrt_sum (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) : 
  ∃ M, (∀ y, -36 ≤ y ∧ y ≤ 36 → sqrt (36 + y) + sqrt (36 - y) ≤ M) ∧ M = 12 :=
by
  sorry

end max_value_sqrt_sum_l191_191718


namespace proof_problem_l191_191905

noncomputable def polar_to_rectangular : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x - sqrt 2) ^ 2 + y ^ 2 = 2) ↔ (∀ θ, rho = 2 * sqrt 2 * Real.cos θ → (x ^ 2 + y ^ 2 = rho ^ 2) ∧ (x = rho * Real.cos θ)) sorry

noncomputable def check_common_points : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x = 3 - sqrt 2 + 2 * Real.cos θ) ∧ (y = 2 * Real.sin θ)) ∧
              ∀ (M_P C_P : ℝ), C_P = 2 ∧ |(3 - sqrt 2) - sqrt 2| = 3 - 2 * sqrt 2 < 2 - sqrt 2 :=
              ¬ ∃ p, (p ∈ C) ∧ (p ∈ C_1) sorry

theorem proof_problem : polar_to_rectangular ∧ check_common_points := by
  sorry

end proof_problem_l191_191905


namespace harry_total_payment_in_silvers_l191_191383

-- Definitions for the conditions
def spellbook_gold_cost : ℕ := 5
def spellbook_count : ℕ := 5
def potion_kit_silver_cost : ℕ := 20
def potion_kit_count : ℕ := 3
def owl_gold_cost : ℕ := 28
def silver_per_gold : ℕ := 9

-- Translate the total cost to silver
noncomputable def total_cost_in_silvers : ℕ :=
  spellbook_count * spellbook_gold_cost * silver_per_gold + 
  potion_kit_count * potion_kit_silver_cost + 
  owl_gold_cost * silver_per_gold

-- State the theorem
theorem harry_total_payment_in_silvers : total_cost_in_silvers = 537 :=
by
  unfold total_cost_in_silvers
  sorry

end harry_total_payment_in_silvers_l191_191383


namespace area_of_closed_figure_l191_191132

noncomputable def f : ℝ → ℝ := λ x, sin (2 * x) - sqrt 3 * cos (2 * x)

noncomputable def g : ℝ → ℝ := λ x, 2 * sin (2 * x)

theorem area_of_closed_figure : 
  ∫ x in 0..(π / 3), g x = 3 / 2 := by
  sorry

end area_of_closed_figure_l191_191132


namespace sequence_sum_a5_a6_l191_191753

-- Given sequence partial sum definition
def partial_sum (n : ℕ) : ℕ := n^3

-- Definition of sequence term a_n
def a (n : ℕ) : ℕ := partial_sum n - partial_sum (n - 1)

-- Main theorem to prove a_5 + a_6 = 152
theorem sequence_sum_a5_a6 : a 5 + a 6 = 152 :=
by
  sorry

end sequence_sum_a5_a6_l191_191753


namespace sum_of_abs_eq_8_eq_10_over_3_l191_191182

theorem sum_of_abs_eq_8_eq_10_over_3 :
  (∑ x in {x : ℝ | |3 * x - 5| = 8}, x) = 10 / 3 :=
sorry

end sum_of_abs_eq_8_eq_10_over_3_l191_191182


namespace problem_statement_l191_191075

noncomputable def r (a b : ℚ) : ℚ := 
  let ab := a * b
  let a_b_recip := a + (1/b)
  let b_a_recip := b + (1/a)
  a_b_recip * b_a_recip

theorem problem_statement (a b : ℚ) (m : ℚ) (h1 : a * b = 3) (h2 : ∃ p, (a + 1 / b) * (b + 1 / a) = (ab + 1 / ab + 2)) :
  r a b = 16 / 3 := by
  sorry

end problem_statement_l191_191075


namespace inequality_proof_l191_191082

variable (a b c d : ℝ)
variable (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)

-- Define conditions
def positive (x : ℝ) := x > 0
def unit_circle (x y : ℝ) := x^2 + y^2 = 1

-- Define the main theorem
theorem inequality_proof
  (ha : positive a)
  (hb : positive b)
  (hc : positive c)
  (hd : positive d)
  (habcd : a * b + c * d = 1)
  (hP1 : unit_circle x1 y1)
  (hP2 : unit_circle x2 y2)
  (hP3 : unit_circle x3 y3)
  (hP4 : unit_circle x4 y4)
  : 
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2
  ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := sorry

end inequality_proof_l191_191082


namespace range_of_m_l191_191333

theorem range_of_m (m : ℝ) :
  ¬(1^2 + 2*1 - m > 0) ∧ (2^2 + 2*2 - m > 0) ↔ (3 ≤ m ∧ m < 8) :=
by
  sorry

end range_of_m_l191_191333


namespace analogical_reasoning_complex_numbers_l191_191502

theorem analogical_reasoning_complex_numbers : 
  (let z : ℂ := 0 in 
    ((∀ a b : ℂ, (a + b = b + a) ∧ (a - b = a + (-b))) ∧ 
    ¬(∃ z : ℂ, |z|^2 = z^2) ∧ 
    ¬(∀ z1 z2 : ℂ, z1 - z2 > 0 → z1 > z2) ∧ 
    (∀ u v : ℂ, (u + v).re = u.re + v.re ∧ (u + v).im = u.im + v.im))) = true :=
by
  sorry

end analogical_reasoning_complex_numbers_l191_191502


namespace proof_problem_l191_191901

noncomputable def polar_to_rectangular : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x - sqrt 2) ^ 2 + y ^ 2 = 2) ↔ (∀ θ, rho = 2 * sqrt 2 * Real.cos θ → (x ^ 2 + y ^ 2 = rho ^ 2) ∧ (x = rho * Real.cos θ)) sorry

noncomputable def check_common_points : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x = 3 - sqrt 2 + 2 * Real.cos θ) ∧ (y = 2 * Real.sin θ)) ∧
              ∀ (M_P C_P : ℝ), C_P = 2 ∧ |(3 - sqrt 2) - sqrt 2| = 3 - 2 * sqrt 2 < 2 - sqrt 2 :=
              ¬ ∃ p, (p ∈ C) ∧ (p ∈ C_1) sorry

theorem proof_problem : polar_to_rectangular ∧ check_common_points := by
  sorry

end proof_problem_l191_191901


namespace find_x_l191_191991

theorem find_x (P Q R S : Type) (collinear : P Q R) (angle_PQS : ℝ) (angle_SQR : ℝ) (h1 : angle_PQS = 42) (h2 : angle_SQR = 2 * x) : 
  42 + 2 * x = 180 → x = 69 := 
by 
  sorry

end find_x_l191_191991


namespace x0_in_interval_l191_191317

theorem x0_in_interval {x x0 : ℝ} (h₁ : f x = x) (h₂ : g x0 = Real.log x0)
  (h₃ : x * x0 + x * Real.log x0 = 0) : x0 ∈ Set.Ioo (1 / Real.exp 1) (1 / Real.sqrt (Real.exp 1)) :=
begin
  sorry
end

def f : ℝ → ℝ := id

def g : ℝ → ℝ := λ x, Real.log x

end x0_in_interval_l191_191317


namespace eddy_wins_probability_l191_191432

-- Define the initial conditions
def lennart_initial : ℕ := 7
def eddy_initial : ℕ := 3

-- The question is to find the probability that Eddy ends with 10 dollars
def probability_eddy_wins : ℚ := 3 / 10

-- Define the game conditions
axiom fair_game_condition (n : ℕ) : Prop := ∃ (coin : ℕ), coin == 0 ∨ coin == 1

theorem eddy_wins_probability :
  ∀ (bet_amount : ℕ),
  bet_amount = min lennart_initial eddy_initial →
  fair_game_condition bet_amount →
  (∃ p : ℚ, p = probability_eddy_wins) :=
by
  assume bet_amount bet_condition fair_game,
  sorry

end eddy_wins_probability_l191_191432


namespace markup_is_correct_l191_191642

noncomputable def profit (S : ℝ) : ℝ := 0.12 * S
noncomputable def expenses (S : ℝ) : ℝ := 0.10 * S
noncomputable def cost (S : ℝ) : ℝ := S - (profit S + expenses S)
noncomputable def markup (S : ℝ) : ℝ :=
  ((S - cost S) / (cost S)) * 100

theorem markup_is_correct:
  markup 10 = 28.21 :=
by
  sorry

end markup_is_correct_l191_191642


namespace range_of_a_l191_191820

theorem range_of_a (a : ℝ) : (∀ x : ℤ, 1 - 2 * (x : ℝ) > -3 ∧ (x : ℝ) - a ≥ 0 ↔ x ∈ {-3, -2, -1, 0, 1}) ↔ (-4 < a ∧ a ≤ -3) :=
by
    sorry

end range_of_a_l191_191820


namespace prove_parabolas_meeting_result_l191_191944

noncomputable def parabolas_meeting_result : Prop :=
  ∃ (P1 P2 : Set (ℝ × ℝ)) 
    (F1 F2 : ℝ × ℝ) 
    (l1 l2 : Set (ℝ × ℝ)), 
    (P1 ≠ P2) ∧ 
    (l1 ≠ l2) ∧ 
    (F1 ≠ F2) ∧ 
    ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2 = 1) ∧ 
    ( ∀ x, x ∈ P1 ↔ dist x F1 = dist x l1) ∧ 
    ( ∀ x, x ∈ P2 ↔ dist x F2 = dist x l2) ∧ 
    ( ∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ P1 ∧ A ∈ P2 ∧ B ∈ P1 ∧ B ∈ P2 ∧ 
        let AB2 := (A.1 - B.1)^2 + (A.2 - B.2)^2 in 
        ∃ m n : ℕ, nat.coprime m n ∧ AB2 = m / n ∧ 100 * m + n = 1504 )

theorem prove_parabolas_meeting_result : parabolas_meeting_result :=
sorry

end prove_parabolas_meeting_result_l191_191944


namespace min_ac_plus_bd_l191_191747

noncomputable def minimum_value_ac_plus_bd : ℝ :=
  let y := sorry -- represents y^2 = 4x
  let F := sorry -- represents the focus of the parabola
  let A := sorry -- point A on the parabola
  let B := sorry -- point B on the parabola
  let C := sorry -- foot of perpendicular from A
  let D := sorry -- foot of perpendicular from B
  let AC := abs (A - C) -- Euclidean distance definition
  let BD := abs (B - D) -- Euclidean distance definition
  let min_value := 2
  in min_value

theorem min_ac_plus_bd : minimum_value_ac_plus_bd = 2 := by
  sorry

end min_ac_plus_bd_l191_191747


namespace add_to_frac_eq_l191_191564

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l191_191564


namespace series_converges_to_one_l191_191666

noncomputable def series_sum : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * (n : ℝ)^2 - 2 * (n : ℝ) + 1) / ((n : ℝ)^4 - (n : ℝ)^3 + (n : ℝ)^2 - (n : ℝ) + 1) else 0

theorem series_converges_to_one : series_sum = 1 := 
  sorry

end series_converges_to_one_l191_191666


namespace proof_problem_l191_191904

noncomputable def polar_to_rectangular : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x - sqrt 2) ^ 2 + y ^ 2 = 2) ↔ (∀ θ, rho = 2 * sqrt 2 * Real.cos θ → (x ^ 2 + y ^ 2 = rho ^ 2) ∧ (x = rho * Real.cos θ)) sorry

noncomputable def check_common_points : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x = 3 - sqrt 2 + 2 * Real.cos θ) ∧ (y = 2 * Real.sin θ)) ∧
              ∀ (M_P C_P : ℝ), C_P = 2 ∧ |(3 - sqrt 2) - sqrt 2| = 3 - 2 * sqrt 2 < 2 - sqrt 2 :=
              ¬ ∃ p, (p ∈ C) ∧ (p ∈ C_1) sorry

theorem proof_problem : polar_to_rectangular ∧ check_common_points := by
  sorry

end proof_problem_l191_191904


namespace test_probability_l191_191490

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def probability_at_least_k_correct (n k : ℕ) (p : ℚ) : ℚ :=
  ∑ i in Finset.range (n + 1) \ Finset.range k, binomial_probability n i p

theorem test_probability :
  probability_at_least_k_correct 20 12 (1/2) = 127475 / 524288 :=
by
  sorry

end test_probability_l191_191490


namespace four_color_theorem_l191_191114

theorem four_color_theorem (G : Type) [graph G] [planar G] :
  ∃ φ : G → fin 4, ∀ v₁ v₂, adjacent v₁ v₂ → φ v₁ ≠ φ v₂ :=
sorry

end four_color_theorem_l191_191114


namespace correct_81st_in_set_s_l191_191965

def is_in_set_s (x : ℕ) : Prop :=
  ∃ n : ℕ, x = 8 * n + 5

noncomputable def find_81st_in_set_s : ℕ :=
  8 * 80 + 5

theorem correct_81st_in_set_s : find_81st_in_set_s = 645 := by
  sorry

end correct_81st_in_set_s_l191_191965


namespace girls_percentage_less_than_boys_l191_191138

theorem girls_percentage_less_than_boys (g b : ℝ) (h : b = 1.25 * g) : (1 - g / b) * 100 = 20 :=
by
  have hb : b = (5 / 4) * g := by rw [h]
  have ratio : g / b = 4 / 5 := by rw [hb]; exact (div_mul_cancel' 4 5).symm
  have percentage_less : (1 - g / b) = 1 - (4 / 5) := by rw [ratio]
  rw [sub_eq_add_neg, add_comm, add_sub_assoc, sub_self, zero_add, mul_comm, ← sub_mul, mul_assoc, mul_div_cancel_left (1 - 4 / 5) (ne_of_gt (by norm_num : (5:ℝ) > 0))]
  norm_num at *
  exact sub_self 0
  sorry

end girls_percentage_less_than_boys_l191_191138


namespace complex_math_problem_l191_191758

theorem complex_math_problem (a : ℝ) (h₀ : a > 0) :
  let z₁ := complex.ofReal (3 / a) + complex.I * (10 - a^2)
  let z₂ := complex.ofReal (2 / (1 - a)) + complex.I * (2 * a - 5)
  let conj_z₁ := complex.conj z₁
  (conj_z₁ + z₂).im = 0 →
  a = 3 ∧ complex.abs ((z₁ / z₂).re + (z₁ / z₂).im * complex.I) = 1 :=
by
  sorry

end complex_math_problem_l191_191758


namespace AB_length_l191_191477

noncomputable def length_of_AB (x y : ℝ) (P_ratio Q_ratio : ℝ × ℝ) (PQ_distance : ℝ) : ℝ :=
    x + y

theorem AB_length (x y : ℝ) (P_ratio : ℝ × ℝ := (3, 5)) (Q_ratio : ℝ × ℝ := (4, 5)) (PQ_distance : ℝ := 3) 
    (h1 : 5 * x = 3 * y) -- P divides AB in the ratio 3:5
    (h2 : 5 * (x + 3) = 4 * (y - 3)) -- Q divides AB in the ratio 4:5 and PQ = 3 units
    : length_of_AB x y P_ratio Q_ratio PQ_distance = 43.2 := 
by sorry

end AB_length_l191_191477


namespace average_minutes_run_l191_191265

theorem average_minutes_run
  (f : ℕ) -- number of fifth graders
  (third_minutes : ℕ := 14)
  (fourth_minutes : ℕ := 18)
  (fifth_minutes : ℕ := 11)
  (third_count : ℕ := 3 * f)
  (fourth_count : ℕ := 1.5 * f)
  (fifth_count : ℕ := f)
  (total_minutes := third_minutes * third_count + fourth_minutes * fourth_count + fifth_minutes * fifth_count)
  (total_students := third_count + fourth_count + fifth_count) :
  (total_minutes / total_students = 160 / 11) :=
by
  sorry

end average_minutes_run_l191_191265


namespace greatest_price_drop_is_april_l191_191216

-- Define the price changes for each month
def price_change (month : ℕ) : ℝ :=
  match month with
  | 1 => 1.00
  | 2 => -1.50
  | 3 => -0.50
  | 4 => -3.75 -- including the -1.25 adjustment
  | 5 => 0.50
  | 6 => -2.25
  | _ => 0 -- default case, although we only deal with months 1-6

-- Define a predicate for the month with the greatest drop
def greatest_drop_month (m : ℕ) : Prop :=
  m = 4

-- Main theorem: Prove that the month with the greatest price drop is April
theorem greatest_price_drop_is_april : greatest_drop_month 4 :=
by
  -- Use Lean tactics to prove the statement
  sorry

end greatest_price_drop_is_april_l191_191216


namespace combinatorial_identity_l191_191740

theorem combinatorial_identity (n : ℕ) (h1 : n ≥ 2) (h2 : nat.choose n 2 = 15) : n * (n - 1) = 30 :=
sorry

end combinatorial_identity_l191_191740


namespace hydrogen_mass_percentage_l191_191705

theorem hydrogen_mass_percentage :
  ∀ (P_N P_H P_total : ℝ), P_N = 77.78 → P_total = 100 → P_H = P_total - P_N → P_H = 22.22 := by
  intros P_N P_H P_total hP_N hP_total hP_eq
  rw [hP_N, hP_total, hP_eq]
  sorry

end hydrogen_mass_percentage_l191_191705


namespace polarToRectangular_noCommonPoints_l191_191883

section PolarToRectangular
variables {θ : ℝ} {ρ : ℝ}

-- Condition: polar coordinate equation ρ = 2√2 cos θ
def polarEq (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Correct answer: rectangular coordinate equation (x - √2)² + y² = 2
theorem polarToRectangular 
  (h : ρ = polarEq θ)
  (x y : ℝ)
  (hx : x = ρ * cos θ)
  (hy : y = ρ * sin θ) :
  (x - sqrt 2) ^ 2 + y ^ 2 = 2 :=
sorry

end PolarToRectangular

section LocusAndCommonPoints
variables {x y x₁ y₁ : ℝ} {C C₁ : set (ℝ × ℝ)}

-- Condition: (1, 0) is point A, M is a moving point on circle C, P satisfies AP = √2AM
def pointA := (1, 0 : ℝ) 
def isOnCircleC (M : ℝ × ℝ) : Prop := (M.1 - sqrt 2) ^ 2 + M.2 ^ 2 = 2
def scaledVector (A P M : ℝ × ℝ) : Prop :=
  P.1 - A.1 = sqrt 2 * (M.1 - A.1) ∧ P.2 - A.2 = sqrt 2 * (M.2 - A.2)

-- Correct answer: Circles C and C₁ do not have any common points
theorem noCommonPoints
  (A : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : isOnCircleC M)
  (P : ℝ × ℝ)
  (hP : scaledVector A P M) :
  ∀ P : ℝ × ℝ, ¬ (isOnCircleC P ∧ (P.1 - (3 - sqrt 2)) ^ 2 + P.2 ^ 2 = 4) :=
sorry

end LocusAndCommonPoints

end polarToRectangular_noCommonPoints_l191_191883


namespace radius_of_omega3_l191_191276

theorem radius_of_omega3 (r : ℝ) 
  (radius_omega1 : ℝ := 3)
  (radius_omega2 : ℝ := 2)
  (distance_between_lines : ℝ := 2 * radius_omega1)
  (distance_between_centers_12 : ℝ := radius_omega1 + radius_omega2)
  (distance_O2_line_a : ℝ := radius_omega2)
  (distance_O3_line_b : ℝ := r)
  (distance_between_centers_13 : ℝ := radius_omega1 + r)
  (distance_between_centers_23 : ℝ := radius_omega2 + r) :
  (distance_between_centers_23 ^ 2 = (distance_between_lines - distance_O2_line_a - r) ^ 2 + (2 * sqrt 6 - 2 * sqrt (3 * r)) ^ 2) →
  r = 9 / 2 :=
by {
  -- variables
  sorry
}

end radius_of_omega3_l191_191276


namespace circle_diameter_8_sqrt_3_l191_191228

theorem circle_diameter_8_sqrt_3
  (A B M : Point)
  (C D : Point)
  (h_AB_diameter : diameter A B)
  (h_M_on_AB : M ∈ segment A B)
  (h_C_on_circum : is_on_circumference C (circle_with_diameter A B))
  (h_D_on_circum : is_on_circumference D (circle_with_diameter A B))
  (h_angle_AMC_eq_30 : ∠AM_C = 30°)
  (h_angle_BMD_eq_30 : ∠BM_D = 30°)
  (h_CD_eq_12 : distance C D = 12) :
  diameter_of_circle (circle_with_diameter A B) = 8 * real.sqrt 3 := by
  sorry

end circle_diameter_8_sqrt_3_l191_191228


namespace area_of_parallelogram_l191_191547

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem area_of_parallelogram : 
  ∀ (A B C D : ℝ × ℝ), 
  A = (0, 0) → B = (8, 0) → C = (5, 10) → D = (13, 10) →
  let base := distance A B in
  let height := abs (C.2 - A.2) in
  base * height = 80 := 
sorry

end area_of_parallelogram_l191_191547


namespace stratified_not_systematic_scenarios_l191_191225

def students_in_first_grade : ℕ := 108
def students_in_second_grade : ℕ := 81
def students_in_third_grade : ℕ := 81
def total_students : ℕ := students_in_first_grade + students_in_second_grade + students_in_third_grade

def scenario_1 := [5, 9, 100, 107, 111, 121, 180, 195, 200, 265]
def scenario_2 := [7, 34, 61, 88, 115, 142, 169, 196, 223, 250]
def scenario_3 := [30, 57, 84, 111, 138, 165, 192, 219, 246, 270]
def scenario_4 := [11, 38, 60, 90, 119, 146, 173, 200, 227, 254]

theorem stratified_not_systematic_scenarios :
  (scenario_1 ⊆ list.range (students_in_first_grade + 1) ∧ ∃ t, scenario_1 = (list.stra t 4 students_in_first_grade ∪ list.stra t 3 (students_in_first_grade + 1) (students_in_first_grade + students_in_second_grade) ∪ list.stra t 3 (students_in_first_grade + students_in_second_grade + 1) total_students) ∧ ¬systematic scenario_1) 
  ∧
  (scenario_2 ⊆ list.range (students_in_first_grade + 1) ∧ ∃ t, scenario_2 = (list.stra t 4 students_in_first_grade ∪ list.stra t 3 (students_in_first_grade + 1) (students_in_first_grade + students_in_second_grade) ∪ list.stra t 3 (students_in_first_grade + students_in_second_grade + 1) total_students) ∧ ¬systematic scenario_2)
  ∧
  (scenario_4 ⊆ list.range (students_in_first_grade + 1) ∧ ∃ t, scenario_4 = (list.stra t 4 students_in_first_grade ∪ list.stra t 3 (students_in_first_grade + 1) (students_in_first_grade + students_in_second_grade) ∪ list.stra t 3 (students_in_first_grade + students_in_second_grade + 1) total_students) ∧ ¬systematic scenario_4) :=
sorry

end stratified_not_systematic_scenarios_l191_191225


namespace minimum_value_of_t_l191_191751

noncomputable theory

-- Definitions based on the conditions:
def a (n : ℕ) : ℚ := if n = 1 then 1/5 else a (n) = a (1) * a (n - 1)  -- Note: In practice, more proper definitions would be needed to implement this recursion

def S (n : ℕ) : ℚ := Σ i in finset.range (n+1), a i

-- The statement we need to prove
theorem minimum_value_of_t (t : ℚ) : (∀ n : ℕ, S n < t) → (1/4 <= t) :=
sorry

end minimum_value_of_t_l191_191751


namespace reasoning_incorrect_l191_191503

def line : Type := sorry
def plane : Type := sorry
def subset (l : line) (p : plane) : Prop := sorry
def parallel_line (l1 l2 : line) : Prop := sorry
def parallel_plane (l : line) (p : plane) : Prop := sorry

variables (a b: line) (α : plane)

-- Given conditions
axiom b_not_subset_alpha : ¬ subset b α
axiom a_subset_alpha : subset a α
axiom b_parallel_alpha : parallel_plane b α

-- Prove the reasoning that b_parallel_plane b α implies parallel b a is incorrect
theorem reasoning_incorrect : ¬ (b_parallel_alpha → parallel_line b a) :=
sorry

end reasoning_incorrect_l191_191503


namespace circle_line_intersect_property_l191_191035
open Real

theorem circle_line_intersect_property :
  let ρ := fun θ : ℝ => 4 * sqrt 2 * sin (3 * π / 4 - θ)
  let cartesian_eq := fun x y : ℝ => (x - 2) ^ 2 + (y - 2) ^ 2 = 8
  let slope := sqrt 3
  let line_param := fun t : ℝ => (1/2 * t, 2 + sqrt 3 / 2 * t)
  let t_roots := {t | ∃ t1 t2 : ℝ, t1 + t2 = 2 ∧ t1 * t2 = -4 ∧ (t = t1 ∨ t = t2)}
  
  (∀ t ∈ t_roots, 
    let (x, y) := line_param t
    cartesian_eq x y)
  → abs ((1 : ℝ) / abs 1 - (1 : ℝ) / abs 2) = 1 / 2 :=
by
  intro ρ cartesian_eq slope line_param t_roots h
  sorry

end circle_line_intersect_property_l191_191035


namespace area_ADFC_is_10_sqrt_119_minus_60_l191_191037

open Real

noncomputable def area_of_quadrilateral (A B C D F : Point) (AB AC AD DB BC AF : ℝ)
  (h1 : ∠C = 90)
  (h2 : AD = 2 * DB)
  (h3 : DF ⊥ AB ∧ midpoint A B F)
  (h4 : AB = 24)
  (h5 : AC = 10) : ℝ :=
  let DB := AB / 3 in
  let AD := 2 * DB in
  let BC := sqrt (AB^2 - AC^2) in
  let area_triangle_ABC := (1 / 2) * AC * BC in
  let area_triangle_ADF := (1 / 2) * (AB / 2) * AC in
  area_triangle_ABC - area_triangle_ADF

theorem area_ADFC_is_10_sqrt_119_minus_60
  (A B C D F : Point)
  (h1 : ∠C = 90)
  (h2 : AD = 2 * DB)
  (h3 : DF ⊥ AB ∧ midpoint A B F)
  (h4 : AB = 24)
  (h5 : AC = 10) :
  area_of_quadrilateral A B C D F AB AC AD DB BC AF h1 h2 h3 h4 h5 = 10 * sqrt 119 - 60 :=
sorry

end area_ADFC_is_10_sqrt_119_minus_60_l191_191037


namespace solution_set_of_xf_x_positive_l191_191363

/-- 
Given:
* f : ℝ → ℝ is an even function.
* f is monotonically increasing on (-∞, 0).
* f(-1) = 0.

Prove that the solution set of xf(x) > 0 is (-∞, -1) ∪ (0, 1).
--/
theorem solution_set_of_xf_x_positive 
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_mono_inc : ∀ x y, x < y → x < 0 → y < 0 → f x ≤ f y)
  (h_f_neg1_zero : f (-1) = 0) :
  {x : ℝ | x * f x > 0} = set.Iic (-1) ∪ set.Ioo 0 1 :=
  sorry

end solution_set_of_xf_x_positive_l191_191363


namespace resized_poster_height_l191_191688

theorem resized_poster_height (original_width original_height new_width : ℕ) 
  (H1 : original_width = 3) 
  (H2 : original_height = 4) 
  (H3 : new_width = 12) : 
  new_width * original_height / original_width = 16 := 
by
  rw [H1, H2, H3]
  norm_num

end resized_poster_height_l191_191688


namespace common_point_of_circumcircles_l191_191640

variables {A B C D X Y : Type}

-- Definitions of points and geometry setup
def convex_quadrilateral (A B C D : Type) : Prop := sorry -- a placeholder for a definition of convex quadrilateral
def on_segment (X A B : Type) : Prop := sorry -- a placeholder for "X is on segment AB"
def intersection (P A B C D : Type) : Prop := sorry -- a placeholder for "P is the intersection of lines AC and DX"

-- Definitions of circumcircles
def circumcircle (A B C : Type) : Type := sorry -- a placeholder for the circumcircle of triangle ABC

-- Main theorem to prove
theorem common_point_of_circumcircles
  (H1 : convex_quadrilateral A B C D)
  (H2 : on_segment X A B)
  (H3 : intersection Y A C D X):
  ∃ P : Type, P ∈ circumcircle A B C ∧ P ∈ circumcircle C D Y ∧ P ∈ circumcircle B D X :=
sorry

end common_point_of_circumcircles_l191_191640


namespace number_of_ways_to_choose_museums_l191_191233

-- Define the conditions
def number_of_grades : Nat := 6
def number_of_museums : Nat := 6
def number_of_grades_Museum_A : Nat := 2

-- Prove the number of ways to choose museums such that exactly two grades visit Museum A
theorem number_of_ways_to_choose_museums :
  (Nat.choose number_of_grades number_of_grades_Museum_A) * (5 ^ (number_of_grades - number_of_grades_Museum_A)) = Nat.choose 6 2 * 5 ^ 4 :=
by
  sorry

end number_of_ways_to_choose_museums_l191_191233


namespace polar_equation_of_circle_length_of_segment_PQ_l191_191043

noncomputable def parametric_circle (φ : ℝ) : ℝ × ℝ :=
(1 + Real.cos φ, Real.sin φ)

def cartesian_to_polar (xy : ℝ × ℝ) : ℝ × ℝ :=
let (x, y) := xy in (Real.sqrt (x^2 + y^2), Real.arctan2 y x)

def polar_equation_circle (ρ θ : ℝ) : Prop :=
ρ = 2 * Real.cos θ

theorem polar_equation_of_circle :
  ∀ φ : ℝ, let xy := parametric_circle φ
  in let (ρ, θ) := cartesian_to_polar xy
  in polar_equation_circle ρ θ :=
by {
  intros φ _,
  sorry
}

def line_l (ρ θ : ℝ) : Prop :=
2 * ρ * Real.sin (θ + π / 3) = 3 * Real.sqrt 3

def ray_eq (xy : ℝ × ℝ) : Prop :=
let (x, y) := xy in Real.sqrt 3 * x - y = 0 ∧ x ≥ 0

def intersection_points (point : ℝ × ℝ) (p : ℝ × ℝ) (q : ℝ × ℝ) : Prop :=
polar_equation_circle (fst (cartesian_to_polar p)) (snd (cartesian_to_polar p)) ∧
line_l (fst (cartesian_to_polar q)) (snd (cartesian_to_polar q)) ∧
ray_eq point ∧ point = p ∧ point = q

theorem length_of_segment_PQ :
 ∀ φ : ℝ, ∀ point p q : ℝ × ℝ, intersection_points point p q
 → abs (fst (cartesian_to_polar p) - fst (cartesian_to_polar q)) = 2 :=
by {
  intros φ point p q h,
  sorry
}

end polar_equation_of_circle_length_of_segment_PQ_l191_191043


namespace none_of_these_l191_191930

-- Conditions
variables (O A B D E P : Point)
variables (r : ℝ)

-- Given conditions
-- 1. O is the center of the circle (implicitly defined)
-- 2. AB ⊥ AD, where D is on the circle
axiom AB_perp_AD : orthogonal (A - B) (A - D)

-- 3. AOEB is a straight line
axiom AOEB_straight : collinear ℝ[O A E B]

-- 4. AP = AB
axiom AP_eq_AB : dist A P = dist A B

-- 5. AD = 2 * r (r is the circle's radius)
axiom AD_eq_2r : dist A D = 2 * r

-- Define Points should lie on a circle with center O
axiom D_on_circle : dist O D = r

-- Define O as the center
def is_center (O : Point) := ∀ (X : Point), dist O X = r → X = D

-- Proof statement
theorem none_of_these : ∀ (AP PB AB AD OB AO : ℝ),
  AP * AP ≠ PB * AB ∧
  AP * dist D O ≠ PB * AD ∧
  AB * AB ≠ AD * dist D E ∧
  AB * AD ≠ OB * AO :=
by { sorry }

end none_of_these_l191_191930


namespace amount_of_naoh_combined_l191_191706

example : 
  (∀ x y : ℕ, (x = 1 ∧ y = 1 ∧ (x + y = 2))) → 
  (∀ x y : ℕ, (x = 1 ∧ y = 1 ∧ (x * y = 1))) → 
  (1 + 1 = 2) → true :=
begin 
  intros, 
  admit
end

-- the rest code follows
open Classical

theorem amount_of_naoh_combined (NH4Cl NaOH NH4OH NaCl : ℕ) 
    (h1 : NH4Cl = 1) 
    (h2 : NH4OH = 1) 
    (h3 : NH4Cl + NaOH = NH4OH + NaCl) 
    (h4 : NH4Cl + NaOH = NH4OH + NaCl) 
    (h5 : NH4Cl + NaOH = 2 ∧ NH4Cl + NH4OH = 2 ∧ NaCl + NaOH = 2) → NaOH = 1 := sorry

end amount_of_naoh_combined_l191_191706


namespace length_of_AC1_l191_191927

-- Define the conditions of the problem
def AB : ℝ := 4
def AD : ℝ := 3
def AA1 : ℝ := 3
def angle_BAD : ℝ := 90
def angle_BAA1 : ℝ := 60
def angle_DAA1 : ℝ := 60

-- State the main theorem to prove
theorem length_of_AC1 : sqrt (AB^2 + AD^2 + AA1^2 + 
  2 * AB * AD * cos (angle_BAD.toRealRad) + 
  2 * AB * AA1 * cos (angle_BAA1.toRealRad) + 
  2 * AD * AA1 * cos (angle_DAA1.toRealRad)) = sqrt 55 :=
  sorry

end length_of_AC1_l191_191927


namespace special_even_diff_regular_l191_191653

def first_n_even_sum (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2)

def special_even_sum (n : ℕ) : ℕ :=
  let sum_cubes := (n * (n + 1) / 2) ^ 2
  let sum_squares := n * (n + 1) * (2 * n + 1) / 6
  2 * (sum_cubes + sum_squares)

theorem special_even_diff_regular : 
  let n := 100
  special_even_sum n - first_n_even_sum n = 51403900 :=
by
  sorry

end special_even_diff_regular_l191_191653


namespace difference_set_B_set_A_l191_191590

noncomputable def sum_even_numbers (start end_: ℕ) : ℕ :=
  let n := (end_ - start) / 2 + 1 in
  (n / 2) * (start + end_)

def set_A_sum : ℕ := sum_even_numbers 32 80

def set_B_sum : ℕ := sum_even_numbers 62 110

theorem difference_set_B_set_A : set_B_sum - set_A_sum = 750 := by
  sorry

end difference_set_B_set_A_l191_191590


namespace parents_without_fulltime_jobs_l191_191199

theorem parents_without_fulltime_jobs (total : ℕ) (mothers fathers full_time_mothers full_time_fathers : ℕ) 
(h1 : mothers = 2 * fathers / 3)
(h2 : full_time_mothers = 9 * mothers / 10)
(h3 : full_time_fathers = 3 * fathers / 4)
(h4 : mothers + fathers = total) :
(100 * (total - (full_time_mothers + full_time_fathers))) / total = 19 :=
by
  sorry

end parents_without_fulltime_jobs_l191_191199


namespace reflect_across_y_axis_l191_191421

-- Definition of the original point A
def pointA : ℝ × ℝ := (2, 3)

-- Definition of the reflected point across the y-axis
def reflectedPoint (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- The theorem stating the reflection result
theorem reflect_across_y_axis : reflectedPoint pointA = (-2, 3) :=
by
  -- Proof (skipped)
  sorry

end reflect_across_y_axis_l191_191421


namespace fractions_between_simplified_l191_191088

theorem fractions_between_simplified :
  ∃ n d : ℕ, (73 < (22*d)/3 ∧ (22*d)/3 < 76) ∧
                  (73 < (15*d)/2 ∧ (15*d)/2 < 76 ) ∧
                  Nat.gcd n d = 1 ∧ 10 ≤ n ∧ n < 100 ∧ 10 ≤ d ∧ d < 100 ∧ 
                  (
                    (n = 81 ∧ d = 11) ∨ (n = 82 ∧ d = 11) ∨ 
                    (n = 89 ∧ d = 12) ∨ (n = 96 ∧ d = 13) ∨ 
                    (n = 97 ∧ d = 13)
                  ) :=
begin
  sorry
end

end fractions_between_simplified_l191_191088


namespace g_a_eq_g_inv_a_l191_191951

noncomputable def f (a x : ℝ) : ℝ :=
  a * real.sqrt (1 - x^2) + real.sqrt (1 + x) + real.sqrt (1 - x)

-- Define t in terms of x 
def t (x : ℝ) : ℝ := real.sqrt (1 + x) + real.sqrt (1 - x)

-- m(t) function derived from f(x)
noncomputable def m (a t : ℝ) : ℝ := (1/2 : ℝ) * a * t^2 + t - a

-- Define the maximum value g(a) based on derived m(t)
noncomputable def g (a : ℝ) : ℝ :=
  if a > -1/2 then a + 2
  else if -real.sqrt 2 / 2 < a ∧ a <= -1/2 then -a - 1/(2*a)
  else real.sqrt 2

-- Proof statement that g(a) = g(1/a) for specific conditions on a
theorem g_a_eq_g_inv_a (a : ℝ) : 
  g a = g (1/a) ↔ (a = 1 ∨ (-real.sqrt 2 ≤ a ∧ a ≤ -(real.sqrt 2)/2)) :=
sorry

end g_a_eq_g_inv_a_l191_191951


namespace product_seq_induction_l191_191169

theorem product_seq_induction (n : ℕ) (h : 0 < n) : 
  (∏ i in finset.range (n+1), (1 - 1/(i+3))) = 2/(n+2) :=
sorry

end product_seq_induction_l191_191169


namespace find_tanC_l191_191427

-- Define the conditions
variables {A B C : Real} (cotA cotB cotC tanC : Real)
axiom cotA_cotC : cotA * cotC = 1
axiom cotB_cotC : cotB * cotC = 1 / 18

-- Define the target statement to prove
theorem find_tanC : (cotA * cotC = 1) → (cotB * cotC = 1 / 18) → tanC = 9 + 2 * Real.sqrt 62 :=
begin
  intros h1 h2,
  sorry,
end

end find_tanC_l191_191427


namespace combined_degrees_of_summer_jolly_winter_autumn_l191_191108

theorem combined_degrees_of_summer_jolly_winter_autumn
  (ratio_summer : ℕ) (ratio_jolly : ℕ) (ratio_winter : ℕ) (ratio_autumn : ℕ)
  (summer_degrees : ℕ) (total_parts : ℕ) (each_part : ℕ)
  (h1 : ratio_summer = 6)
  (h2 : ratio_jolly = 4)
  (h3 : ratio_winter = 5)
  (h4 : ratio_autumn = 3)
  (h5 : summer_degrees = 150)
  (h6 : total_parts = ratio_summer + ratio_jolly + ratio_winter + ratio_autumn)
  (h7 : each_part = summer_degrees / ratio_summer) :
  (total_parts * each_part = 450) :=
begin
  sorry
end

end combined_degrees_of_summer_jolly_winter_autumn_l191_191108


namespace point_outside_circle_l191_191012

variable (A O : Type) [metric_space O] [has_dist A O]

noncomputable def diameter : ℝ := 10
noncomputable def distance_A_to_O : ℝ := 6
noncomputable def radius : ℝ := diameter / 2

theorem point_outside_circle (diameter : ℝ) (distance_A_to_O : ℝ) (radius : ℝ) (h_d : diameter = 10) (h_dA : distance_A_to_O = 6) (h_r : radius = diameter / 2) :
  distance_A_to_O > radius := 
by 
  rw [h_d, h_dA, h_r]
  sorry

end point_outside_circle_l191_191012


namespace C_and_C1_no_common_points_l191_191910

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  let x := rho * cos theta
  let y := rho * sin theta
  (x, y)

def curve_C_eq (theta : ℝ) : ℝ :=
  2 * sqrt 2 * cos theta

def A := (1, 0 : ℝ)

-- Predicate indicating point M is on curve C given in rectangular coordinates.
def on_curve_C (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  (x - sqrt 2) ^ 2 + y ^ 2 = 2

-- Definition for point P satisfying the given vector equation.
def point_P (A M : ℝ × ℝ) : ℝ × ℝ :=
  let x_A := A.1
  let y_A := A.2
  let x_M := M.1
  let y_M := M.2
  (√2 * (x_M - x_A) + x_A, √2 * (y_M - y_A) + y_A)

-- Prove that the rectangular coordinate equation is as specified and parametrized locus C_1 exists and has no common points with C.
theorem C_and_C1_no_common_points (M P : ℝ × ℝ) (theta : ℝ)
  (h_M_on_C : on_curve_C M)
  (h_P_eq : P = point_P A M) :
  let C1_center := (3 - sqrt 2, 0)
  let C1_radius := 2
  let C_center   := (sqrt 2, 0)
  let C_radius   := sqrt 2
  let C1_eq := (x - (3 - sqrt 2))^2 + y^2 = 4
  (M, P) ∉ C ∩ C1 :=
sorry

end C_and_C1_no_common_points_l191_191910


namespace projection_of_2_3_on_neg4_7_l191_191736

variables (a b : ℝ × ℝ)

def projection_of_a_on_b (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1 * b.1 + b.2 * b.2)
  dot_product / magnitude_b

theorem projection_of_2_3_on_neg4_7 :
  projection_of_a_on_b (2, 3) (-4, 7) = Real.sqrt 65 / 5 :=
by
  sorry

end projection_of_2_3_on_neg4_7_l191_191736


namespace cant_generate_AC_l191_191190

-- Let's define the rules and sequences as an inductive type
inductive Sequence where
  | AB : Sequence
  | AC : Sequence
  | others : (String) → Sequence

def endsWithB : Sequence → Bool
  | Sequence.others s => s.reverse.head? = some 'B'
  | _ => false

def startsWithA : Sequence → Bool
  | Sequence.others s => s.head? = some 'A'
  | _ => false

def countB : Sequence → Nat
  | Sequence.others s => s.filter (fun c => c = 'B').length
  | _ => 0

def isDivisibleBy3 (n : Nat) : Bool :=
  n % 3 = 0

def canGenerateAC (seq : Sequence) : Bool :=
  match seq with
  | Sequence.AC => true
  | _ => false

theorem cant_generate_AC :
  ∀ (s : Sequence), s = Sequence.AB → ¬canGenerateAC s
| _ => by
  sorry

end cant_generate_AC_l191_191190


namespace C_and_C1_no_common_points_l191_191838

noncomputable def C_equation : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - real.sqrt 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (1, 0)

noncomputable def M_on_C (θ : ℝ) : ℝ × ℝ :=
  let x := 2 * real.sqrt 2 * real.cos θ * real.cos θ in
  let y := 2 * real.sqrt 2 * real.sin θ * real.sin θ in
  (x, y)

noncomputable def P_on_locus (x_M y_M : ℝ) : ℝ × ℝ :=
  let x := real.sqrt 2 * (x_M - point_A.1) + point_A.1 in
  let y := real.sqrt 2 * y_M in
  (x, y)

noncomputable def Locus_P_equation (θ : ℝ) : ℝ × ℝ :=
  let x := 3 - real.sqrt 2 + 2 * real.cos θ in
  let y := 2 * real.sin θ in
  (x, y)

def has_common_points (C1 C2 : set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C1 ∧ p ∈ C2

theorem C_and_C1_no_common_points :
  ¬ has_common_points
    {p | C_equation p}
    {p | ∃ θ, P_on_locus (M_on_C θ).1 (M_on_C θ).2 = p} :=
sorry

end C_and_C1_no_common_points_l191_191838


namespace C_and_C1_no_common_points_l191_191912

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  let x := rho * cos theta
  let y := rho * sin theta
  (x, y)

def curve_C_eq (theta : ℝ) : ℝ :=
  2 * sqrt 2 * cos theta

def A := (1, 0 : ℝ)

-- Predicate indicating point M is on curve C given in rectangular coordinates.
def on_curve_C (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  (x - sqrt 2) ^ 2 + y ^ 2 = 2

-- Definition for point P satisfying the given vector equation.
def point_P (A M : ℝ × ℝ) : ℝ × ℝ :=
  let x_A := A.1
  let y_A := A.2
  let x_M := M.1
  let y_M := M.2
  (√2 * (x_M - x_A) + x_A, √2 * (y_M - y_A) + y_A)

-- Prove that the rectangular coordinate equation is as specified and parametrized locus C_1 exists and has no common points with C.
theorem C_and_C1_no_common_points (M P : ℝ × ℝ) (theta : ℝ)
  (h_M_on_C : on_curve_C M)
  (h_P_eq : P = point_P A M) :
  let C1_center := (3 - sqrt 2, 0)
  let C1_radius := 2
  let C_center   := (sqrt 2, 0)
  let C_radius   := sqrt 2
  let C1_eq := (x - (3 - sqrt 2))^2 + y^2 = 4
  (M, P) ∉ C ∩ C1 :=
sorry

end C_and_C1_no_common_points_l191_191912


namespace planes_relationships_l191_191766

variables {m n l : Line}
variables {α β : Plane}

def skew_lines (m n : Line) : Prop := 
  ¬(m ∥ n) ∧ ¬(m = n) ∧ ¬(∃ p : Point, p ∈ m ∧ p ∈ n)

def perp_to_plane (l : Line) (α : Plane) : Prop :=
  ∀ p q : Point, p ∈ l → p ∈ α → q ∈ l → q ∈ α → p = q

def parallel (l : Line) (α : Plane) : Prop :=
  ∀ p ∈ l, ∀ q : Point, q ∈ α ∧ q ∉ l → ∃ r : Line, r ∥ l ∧ ∀ s ∈ α, ∃ t : Point, t ∈ r ∧ t = s

theorem planes_relationships (m n l : Line) (α β : Plane)
  (h_skew : skew_lines m n)
  (h1 : perp_to_plane m α)
  (h2 : perp_to_plane n β)
  (h3 : perp_to_plane l m)
  (h4 : perp_to_plane l n)
  (h5 : ¬(l ⊆ α))
  (h6 : ¬(l ⊆ β)) :
  ∃ p ∈ α, p ∈ β ∧ (∀ q ∈ (α ∩ β), l ∥ (Line.mk q l)) := sorry

end planes_relationships_l191_191766


namespace inradius_isosceles_triangle_l191_191411

theorem inradius_isosceles_triangle (a : ℝ) (h_pos : 0 < a) (h_area : 1/2 * a * (24/a) = 12) :
  let r := (8/a) in r = 3/2 :=
by
  sorry

end inradius_isosceles_triangle_l191_191411


namespace num_three_person_subcommittees_from_eight_l191_191804

def num_committees (n k : ℕ) : ℕ := (Nat.fact n) / ((Nat.fact k) * (Nat.fact (n - k)))

theorem num_three_person_subcommittees_from_eight (n : ℕ) (h : n = 8) : num_committees n 3 = 56 :=
by
  rw [h]
  sorry

end num_three_person_subcommittees_from_eight_l191_191804


namespace solve_inequality_l191_191954

variable {f : ℝ → ℝ}
variable {x : ℝ}
variable increasing_f : ∀ {a b : ℝ}, a < b → f(a) < f(b)
variable functional_eq_f : ∀ {a b : ℝ}, f(a * b) = f(a) + f(b)
variable f_value_3 : f(3) = 1

theorem solve_inequality (h: f(x) + f(x-2) > 1) : x > 3 ∨ x < -1 :=
by
  sorry

end solve_inequality_l191_191954


namespace painting_time_equation_l191_191680

theorem painting_time_equation (t : ℝ) :
  let Doug_rate := (1 : ℝ) / 5
  let Dave_rate := (1 : ℝ) / 7
  let combined_rate := Doug_rate + Dave_rate
  (combined_rate * (t - 1) = 1) :=
sorry

end painting_time_equation_l191_191680


namespace polarToRectangular_noCommonPoints_l191_191888

section PolarToRectangular
variables {θ : ℝ} {ρ : ℝ}

-- Condition: polar coordinate equation ρ = 2√2 cos θ
def polarEq (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Correct answer: rectangular coordinate equation (x - √2)² + y² = 2
theorem polarToRectangular 
  (h : ρ = polarEq θ)
  (x y : ℝ)
  (hx : x = ρ * cos θ)
  (hy : y = ρ * sin θ) :
  (x - sqrt 2) ^ 2 + y ^ 2 = 2 :=
sorry

end PolarToRectangular

section LocusAndCommonPoints
variables {x y x₁ y₁ : ℝ} {C C₁ : set (ℝ × ℝ)}

-- Condition: (1, 0) is point A, M is a moving point on circle C, P satisfies AP = √2AM
def pointA := (1, 0 : ℝ) 
def isOnCircleC (M : ℝ × ℝ) : Prop := (M.1 - sqrt 2) ^ 2 + M.2 ^ 2 = 2
def scaledVector (A P M : ℝ × ℝ) : Prop :=
  P.1 - A.1 = sqrt 2 * (M.1 - A.1) ∧ P.2 - A.2 = sqrt 2 * (M.2 - A.2)

-- Correct answer: Circles C and C₁ do not have any common points
theorem noCommonPoints
  (A : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : isOnCircleC M)
  (P : ℝ × ℝ)
  (hP : scaledVector A P M) :
  ∀ P : ℝ × ℝ, ¬ (isOnCircleC P ∧ (P.1 - (3 - sqrt 2)) ^ 2 + P.2 ^ 2 = 4) :=
sorry

end LocusAndCommonPoints

end polarToRectangular_noCommonPoints_l191_191888


namespace domain_of_f_range_of_f_l191_191783

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.sqrt (4 - 8^x)

-- Domain and Range of f(x)
theorem domain_of_f : ∀ x : ℝ, f x = Real.sqrt (4 - 8^x) → x ≤ 2 / 3 → f x ∈ Set.interval 0 2 :=
by sorry

theorem range_of_f (x : ℝ) : f x ≤ 1 → (log 3 / log 2 / 3) ≤ x ∧ x ≤ 2 / 3 :=
by sorry

end domain_of_f_range_of_f_l191_191783


namespace value_of_x_minus_y_l191_191334

theorem value_of_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := 
by
  sorry

end value_of_x_minus_y_l191_191334


namespace electric_charge_of_oxygen_electrons_l191_191482

theorem electric_charge_of_oxygen_electrons :
  (number_of_electrons : ℕ) (charge_per_electron : ℤ)
  (h₀ : charge_per_electron = -1)
  (h₁ : number_of_electrons = 8) :
  (total_charge : ℤ) := 
  begin
    let total_charge := number_of_electrons * charge_per_electron,
    sorry
  end

end electric_charge_of_oxygen_electrons_l191_191482


namespace add_to_frac_eq_l191_191566

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l191_191566


namespace roots_product_eq_l191_191078

theorem roots_product_eq
  (a b m p r : ℚ)
  (h₀ : a * b = 3)
  (h₁ : ∀ x, x^2 - m * x + 3 = 0 → (x = a ∨ x = b))
  (h₂ : ∀ x, x^2 - p * x + r = 0 → (x = a + 1 / b ∨ x = b + 1 / a)) : 
  r = 16 / 3 :=
by
  sorry

end roots_product_eq_l191_191078


namespace ratio_of_saramago_readers_to_total_l191_191020

def PalabrasBookstore := 
  let W := 150 -- total number of workers 
  let K := W / 6 -- number of workers who have read the latest Kureishi book
  let B := 12 -- number of workers who have read both books
  let N := λ S => S - B - 1 -- condition for workers who read neither book
  let total_workers_eq := λ S => (S - B) + (K - B) + B + N S = W

  theorem ratio_of_saramago_readers_to_total (S : ℕ) : 
    total_workers_eq S →
    2 * S = W →
    S / W = 1 / 2 := 
  by
    sorry

end ratio_of_saramago_readers_to_total_l191_191020


namespace congruence_example_l191_191000

theorem congruence_example (x : ℤ) (h : 5 * x + 3 ≡ 1 [ZMOD 18]) : 3 * x + 8 ≡ 14 [ZMOD 18] :=
sorry

end congruence_example_l191_191000


namespace ab_cd_eq_neg_37_over_9_l191_191815

theorem ab_cd_eq_neg_37_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a + b + d = 2)
  (h3 : a + c + d = 3)
  (h4 : b + c + d = -3) :
  a * b + c * d = -37 / 9 := by
  sorry

end ab_cd_eq_neg_37_over_9_l191_191815


namespace domain_of_f_l191_191125

def f (x : ℝ) : ℝ := sqrt (1 - 3^x) + 1/x^2

theorem domain_of_f : ∀ x : ℝ, (1 - 3^x ≥ 0) ∧ (x ≠ 0) ↔ x < 0 :=
by
  intros x,
  sorry

end domain_of_f_l191_191125


namespace conjugate_subgroups_l191_191052

open Group

variables (G : Type*) [Group G] (n : ℕ) (H1 H2 : Subgroup G)
variables (h1 : H1.index = n) (h2 : H2.index = n)
variables (intersectIndex : (H1 ⊓ H2).index = n * (n - 1))

theorem conjugate_subgroups :
  ∃ g : G, g⁻¹ * H1 * g = H2 :=
sorry

end conjugate_subgroups_l191_191052


namespace convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191900

-- Definitions from the problem
def polar_eq_C (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Conversion to rectangular coordinates
def rectangular_eq_C (x y : ℝ) : Prop := (x - sqrt 2)^2 + y^2 = 2

-- Definitions of points and vectors
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨1, 0⟩

def M (θ : ℝ) : Point :=
  let ρ := polar_eq_C θ
  ⟨ρ * cos θ, ρ * sin θ⟩

def P (x y : ℝ) : Point := ⟨x, y⟩

-- Conditions and transformations
def vector_eq (p1 p2 : Point) (a : ℝ) : Prop :=
  P p1.x p1.y = ⟨a * p2.x + (1 - a) * A.x, a * p2.y + (1 - a) * A.y⟩

def parametric_eq_P (θ : ℝ) : Point :=
  ⟨3 - sqrt 2 + 2 * cos θ, 2 * sin θ⟩

-- Proof problems
theorem convert_polar_to_rectangular (θ : ℝ) :
  rectangular_eq_C (polar_eq_C θ * cos θ) (polar_eq_C θ * sin θ) :=
sorry

theorem parametric_eq_valid (θ : ℝ) :
  vector_eq (parametric_eq_P θ) (M θ) (sqrt 2) :=
sorry

theorem no_common_points :
  ∀ θ₁ θ₂, parametric_eq_P θ₁ ≠ M θ₂ :=
sorry

end convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191900


namespace sum_x_coordinates_intersect_x_axis_l191_191673

theorem sum_x_coordinates_intersect_x_axis (c : ℝ) (h : -2 = -c / 4) :
  let x1 := -2;
      x2 := -c / 4;
  x1 + x2 = -2 :=
by
  sorry

end sum_x_coordinates_intersect_x_axis_l191_191673


namespace length_of_b_l191_191018

noncomputable def side_b (a c B : ℝ) : ℝ :=
  sqrt (a^2 + c^2 - 2 * a * c * real.cos B)

theorem length_of_b (a c B : ℝ) (ha : a = 9) (hc : c = 2 * real.sqrt 3) (hB : B = 150 * real.pi / 180) :
  side_b a c B = 7 * real.sqrt 3 := by
  sorry

end length_of_b_l191_191018


namespace find_value_of_S_l191_191475

theorem find_value_of_S (S : ℝ)
  (h1 : (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 180) :
  S = 180 :=
sorry

end find_value_of_S_l191_191475


namespace term_2500_mod_7_l191_191282

def sequence_term (k: ℕ): ℕ :=
  let n := if k % 2 = 0 then k / 2 else (k + 1) / 2 
  in if k % 2 = 0 then n else n - 1

theorem term_2500_mod_7 : (sequence_term 2500 % 7) = 1 := 
sorry

end term_2500_mod_7_l191_191282


namespace event_d_is_certain_l191_191252

theorem event_d_is_certain : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ (3 * x^2 - 8 * x + 5 = 0) ∧ (x = 1) := 
by
  use 1
  split
  -- 0 ≤ x ∧ x ≤ 1 step
  split
  -- proof for 0 ≤ 1
  norm_num
  -- proof for 1 ≤ 1
  norm_num
  -- 3 * x^2 - 8 * x + 5 = 0 step
  split
  norm_num
  ring_nf
  -- x = 1 step
  rfl

end event_d_is_certain_l191_191252


namespace solve_equation_l191_191302

theorem solve_equation (x : ℝ) :
  (√(√(√(√x)))) = 15 / (8 - √(√(√(√x)))) ↔ x = 625 ∨ x = 81 :=
by sorry

end solve_equation_l191_191302


namespace mabel_total_tomatoes_l191_191971

theorem mabel_total_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 :=
by
  sorry

end mabel_total_tomatoes_l191_191971


namespace graphs_intersect_at_two_points_range_of_projection_length_l191_191748

section quadratic_linear_intersection

variables (a b c x1 x2 : ℝ)
variable (hx1x2 : x1 > x2)
variables (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0)

-- Define the quadratic and linear functions
def f (x : ℝ) := a * x^2 + b * x + c
def g (x : ℝ) := -b * x

-- Statement 1: The graphs intersect at two distinct points
theorem graphs_intersect_at_two_points (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b c x1 = g b x1 ∧ f a b c x2 = g b x2 := 
sorry

-- Statement 2: The range of the length of projection segment A1B1
theorem range_of_projection_length (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  sqrt 3 < abs (x1 - x2) ∧ abs (x1 - x2) < 2 * sqrt 3 := 
sorry

end quadratic_linear_intersection

end graphs_intersect_at_two_points_range_of_projection_length_l191_191748


namespace distance_from_apex_to_larger_section_l191_191166

theorem distance_from_apex_to_larger_section (a1 a2 D : ℝ) 
  (h₁ : a1 = 360) (h₂ : a2 = 810) (h₃ : D = 10) :
  ∃ h : ℝ, (h - (2 / 3) * h = D) ∧ (h = 30) :=
by
  -- Define the ratio of the areas
  have ratio_area : a1 / a2 = 4 / 9 := by
    rw [h₁, h₂]
    norm_num

  -- Ratio of the linear dimensions
  have ratio_dim : sqrt (a1 / a2) = 2 / 3 := by
    rw [ratio_area]
    norm_num

  -- Solve for h
  use 30
  constructor
  · -- Prove the equation: h - (2 / 3) * h = D
    norm_num
    sorry
  · -- Prove that h actually equals 30
    norm_num

end distance_from_apex_to_larger_section_l191_191166


namespace eliana_additional_steps_first_day_l191_191687

variables (x : ℝ)

def eliana_first_day_steps := 200 + x
def eliana_second_day_steps := 2 * eliana_first_day_steps
def eliana_third_day_steps := eliana_second_day_steps + 100
def eliana_total_steps := eliana_first_day_steps + eliana_second_day_steps + eliana_third_day_steps

theorem eliana_additional_steps_first_day : eliana_total_steps = 1600 → x = 100 :=
by {
  sorry
}

end eliana_additional_steps_first_day_l191_191687


namespace runway_show_total_time_l191_191498

-- Define the conditions
def time_per_trip : Nat := 2
def num_models : Nat := 6
def trips_bathing_suits_per_model : Nat := 2
def trips_evening_wear_per_model : Nat := 3
def trips_per_model : Nat := trips_bathing_suits_per_model + trips_evening_wear_per_model
def total_trips : Nat := trips_per_model * num_models

-- State the theorem
theorem runway_show_total_time : total_trips * time_per_trip = 60 := by
  -- fill in the proof here
  sorry

end runway_show_total_time_l191_191498


namespace polar_to_rect_eq_and_locus_no_common_points_l191_191863

theorem polar_to_rect_eq_and_locus_no_common_points :
  let C := { (x, y) | (x - Real.sqrt 2)^2 + y^2 = 2 } in
  let C1 := { (x, y) | ∃ θ, x = 3 - Real.sqrt 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ } in
  (∀ (ρ θ : Real), ρ = 2 * Real.sqrt 2 * Real.cos θ → (Real.sqrt (ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos θ) - Real.sqrt 2)^2 + (Real.sin θ)^2 = 2 )
  ∧ ¬∃ (p : Real × Real), p ∈ C ∧ p ∈ C1 :=
by
  sorry

end polar_to_rect_eq_and_locus_no_common_points_l191_191863


namespace population_ratios_l191_191663

variable (P_X P_Y P_Z : Nat)

theorem population_ratios
  (h1 : P_Y = 2 * P_Z)
  (h2 : P_X = 10 * P_Z) : P_X / P_Y = 5 := by
  sorry

end population_ratios_l191_191663


namespace no_perfect_square_for_nnplus1_l191_191430

theorem no_perfect_square_for_nnplus1 :
  ¬ ∃ (n : ℕ), 0 < n ∧ ∃ (k : ℕ), n * (n + 1) = k * k :=
sorry

end no_perfect_square_for_nnplus1_l191_191430


namespace smallest_possible_value_l191_191049

theorem smallest_possible_value (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) (h2 : n ≡ 2 [MOD 9]) (h3 : n ≡ 6 [MOD 7]) :
  n = 116 :=
by
  -- Proof omitted
  sorry

end smallest_possible_value_l191_191049


namespace general_term_seq1_general_term_seq2_l191_191703

-- Definition of the sequence aₙ satisfying a₁ = 0 and a_{n+1} = aₙ + n
def seq1 : ℕ → ℕ 
| 1     := 0
| (n+2) := seq1 (n+1) + n + 1

-- Proof Problem 1: General term formula for seq1 is aₙ = n(n-1)/2
theorem general_term_seq1 (n : ℕ) (h : n > 0) : seq1 n = n * (n - 1) / 2 :=
by sorry

-- Definition of the sequence aₙ satisfying a₁ = 1 and (a_{n+1}/aₙ) = (n+2)/n
def seq2 : ℕ → ℕ 
| 1     := 1
| (n+2) := (seq2 (n+1) * (n + 2)) / n

-- Proof Problem 2: General term formula for seq2 is aₙ = n(n+1)/2
theorem general_term_seq2 (n : ℕ) (h : n > 0) : seq2 n = n * (n + 1) / 2 :=
by sorry

end general_term_seq1_general_term_seq2_l191_191703


namespace tim_score_l191_191184

def first8Primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Function to compute the product of a list of numbers
def product_of_list (l : List ℕ) : ℕ :=
  l.foldl (*) 1

-- Function to compute the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Function to compute the sum of the first N even numbers
def sum_of_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (n + 1)

theorem tim_score :
  sum_of_first_n_even_numbers (sum_of_digits (product_of_list first8Primes)) = 2352 :=
by
  sorry

end tim_score_l191_191184


namespace total_pears_picked_is_correct_l191_191997

-- Define the number of pears picked by Sara and Sally
def pears_picked_by_Sara : ℕ := 45
def pears_picked_by_Sally : ℕ := 11

-- The total number of pears picked
def total_pears_picked := pears_picked_by_Sara + pears_picked_by_Sally

-- The theorem statement: prove that the total number of pears picked is 56
theorem total_pears_picked_is_correct : total_pears_picked = 56 := by
  sorry

end total_pears_picked_is_correct_l191_191997


namespace dinosaur_count_l191_191214

theorem dinosaur_count (h : ℕ) (l : ℕ) (H1 : h = 1) (H2 : l = 3) (total_hl : ℕ) (H3 : total_hl = 20) :
  ∃ D : ℕ, 4 * D = total_hl := 
by
  use 5
  sorry

end dinosaur_count_l191_191214


namespace solve_equation_l191_191299

theorem solve_equation (x : ℝ) : (sqrt (sqrt x) = 15 / (8 - sqrt (sqrt x))) ↔ (x = 625 ∨ x = 81) := 
by
  sorry

end solve_equation_l191_191299


namespace trajectory_and_dot_product_min_value_l191_191744

theorem trajectory_and_dot_product_min_value :
  (∀ (C : ℝ × ℝ), (dist C (0, 1) = dist C (0, -1)) ↔ (C.1 ^ 2 = 4 * C.2)) ∧
  (∀ (k : ℝ), k ≠ 0 →
    let P := (λ (x₁ : ℝ), (x₁, k * x₁ + 1)) in
    let Q := (λ (x₂ : ℝ), (x₂, k * x₂ + 1)) in
    let R := (-2 / k, -1) in
    ∃ x₁ x₂ : ℝ, x₁ + x₂ = 4 * k ∧ x₁ * x₂ = -4 ∧
    (P x₁ - R) ⬝ (Q x₂ - R) ≥ 4 * (k ^ 2 + 1 / k ^ 2) + 8 ∧
    (P x₁ - R) ⬝ (Q x₂ - R) = 16) :=
by sorry

end trajectory_and_dot_product_min_value_l191_191744


namespace max_value_expression_l191_191925

theorem max_value_expression (x y k z : ℕ) (h1 : x ≠ y) (h2 : x ≠ k) (h3 : x ≠ z)
    (h4 : y ≠ k) (h5 : y ≠ z) (h6 : k ≠ z) (h7 : {x, y, k, z} = {1, 2, 3, 4}) : 
    k * x^y - z = 127 := 
by
  sorry

end max_value_expression_l191_191925


namespace polar_to_rect_eq_and_locus_no_common_points_l191_191864

theorem polar_to_rect_eq_and_locus_no_common_points :
  let C := { (x, y) | (x - Real.sqrt 2)^2 + y^2 = 2 } in
  let C1 := { (x, y) | ∃ θ, x = 3 - Real.sqrt 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ } in
  (∀ (ρ θ : Real), ρ = 2 * Real.sqrt 2 * Real.cos θ → (Real.sqrt (ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos θ) - Real.sqrt 2)^2 + (Real.sin θ)^2 = 2 )
  ∧ ¬∃ (p : Real × Real), p ∈ C ∧ p ∈ C1 :=
by
  sorry

end polar_to_rect_eq_and_locus_no_common_points_l191_191864


namespace convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191899

-- Definitions from the problem
def polar_eq_C (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Conversion to rectangular coordinates
def rectangular_eq_C (x y : ℝ) : Prop := (x - sqrt 2)^2 + y^2 = 2

-- Definitions of points and vectors
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨1, 0⟩

def M (θ : ℝ) : Point :=
  let ρ := polar_eq_C θ
  ⟨ρ * cos θ, ρ * sin θ⟩

def P (x y : ℝ) : Point := ⟨x, y⟩

-- Conditions and transformations
def vector_eq (p1 p2 : Point) (a : ℝ) : Prop :=
  P p1.x p1.y = ⟨a * p2.x + (1 - a) * A.x, a * p2.y + (1 - a) * A.y⟩

def parametric_eq_P (θ : ℝ) : Point :=
  ⟨3 - sqrt 2 + 2 * cos θ, 2 * sin θ⟩

-- Proof problems
theorem convert_polar_to_rectangular (θ : ℝ) :
  rectangular_eq_C (polar_eq_C θ * cos θ) (polar_eq_C θ * sin θ) :=
sorry

theorem parametric_eq_valid (θ : ℝ) :
  vector_eq (parametric_eq_P θ) (M θ) (sqrt 2) :=
sorry

theorem no_common_points :
  ∀ θ₁ θ₂, parametric_eq_P θ₁ ≠ M θ₂ :=
sorry

end convert_polar_to_rectangular_parametric_eq_valid_no_common_points_l191_191899


namespace find_number_to_add_l191_191569

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l191_191569


namespace number_of_cards_in_deck_l191_191403

theorem number_of_cards_in_deck (N : ℕ) (h_spades : N > 13) (h_probability : (N - 13) / N = 0.75) : N = 52 :=
sorry

end number_of_cards_in_deck_l191_191403


namespace line_circle_intersection_l191_191291

theorem line_circle_intersection :
  let O : ℝ × ℝ := (0, 0)
  let r : ℝ := 2
  let distance_from_center_to_line (cx cy a b c: ℝ) : ℝ := (|a * cx + b * cy + c|) / (real.sqrt (a^2 + b^2))
  ∀ x y : ℝ, x ^ 2 + y ^ 2 = r ^ 2 ->
  sqrt 3 * x + sqrt 3 * y = 4 ->
  distance_from_center_to_line 0 0 (sqrt 3) (sqrt 3) (-4) < r :=
by
  -- Using fact: distance d from line to center of circle with given parameters:
  -- distance_from_center_to_line 0 0 (sqrt 3) (sqrt 3) (-4) = 4 / (sqrt 6) = 2sqrt 6 / 3
  -- and we know 2sqrt 6 / 3 < 2 given in provided proof
  sorry

end line_circle_intersection_l191_191291


namespace max_value_sqrt_sum_l191_191715

theorem max_value_sqrt_sum (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) : 
  ∃ M, (∀ y, -36 ≤ y ∧ y ≤ 36 → sqrt (36 + y) + sqrt (36 - y) ≤ M) ∧ M = 12 :=
by
  sorry

end max_value_sqrt_sum_l191_191715


namespace integral_evaluation_l191_191690

noncomputable def integral_example : ℝ :=
  ∫ x in 0..(1/2), exp (2 * x)

theorem integral_evaluation:
  integral_example = (1 / 2) * (Real.exp 1 - 1) :=
by
  sorry

end integral_evaluation_l191_191690


namespace real_part_of_z_l191_191143

/-- Define the complex number z as 5 / (1 - 2i) -/
def z : ℂ := 5 / (1 - 2 * Complex.i)

theorem real_part_of_z : Complex.re z = 1 :=
  sorry

end real_part_of_z_l191_191143


namespace downstream_time_l191_191604

variable (B C D : ℝ)

-- Conditions
def ratio_condition : Prop := B = 4 * C
def upstream_time_condition : Prop := ∀D, D = (B - C) * 6

-- Conclusion
theorem downstream_time : ratio_condition B C → upstream_time_condition B C D → ∃ Td, Td = 3.6 :=
by
  intro ratio_condition
  intro upstream_time_condition
  use 3.6
  sorry

end downstream_time_l191_191604


namespace inequality_solution_l191_191487

theorem inequality_solution :
  ∀ x : ℝ, ( (x - 3) / ( (x - 2) ^ 2 ) < 0 ) ↔ ( x < 2 ∨ (2 < x ∧ x < 3) ) :=
by
  sorry

end inequality_solution_l191_191487


namespace polarToRectangular_noCommonPoints_l191_191891

section PolarToRectangular
variables {θ : ℝ} {ρ : ℝ}

-- Condition: polar coordinate equation ρ = 2√2 cos θ
def polarEq (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Correct answer: rectangular coordinate equation (x - √2)² + y² = 2
theorem polarToRectangular 
  (h : ρ = polarEq θ)
  (x y : ℝ)
  (hx : x = ρ * cos θ)
  (hy : y = ρ * sin θ) :
  (x - sqrt 2) ^ 2 + y ^ 2 = 2 :=
sorry

end PolarToRectangular

section LocusAndCommonPoints
variables {x y x₁ y₁ : ℝ} {C C₁ : set (ℝ × ℝ)}

-- Condition: (1, 0) is point A, M is a moving point on circle C, P satisfies AP = √2AM
def pointA := (1, 0 : ℝ) 
def isOnCircleC (M : ℝ × ℝ) : Prop := (M.1 - sqrt 2) ^ 2 + M.2 ^ 2 = 2
def scaledVector (A P M : ℝ × ℝ) : Prop :=
  P.1 - A.1 = sqrt 2 * (M.1 - A.1) ∧ P.2 - A.2 = sqrt 2 * (M.2 - A.2)

-- Correct answer: Circles C and C₁ do not have any common points
theorem noCommonPoints
  (A : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : isOnCircleC M)
  (P : ℝ × ℝ)
  (hP : scaledVector A P M) :
  ∀ P : ℝ × ℝ, ¬ (isOnCircleC P ∧ (P.1 - (3 - sqrt 2)) ^ 2 + P.2 ^ 2 = 4) :=
sorry

end LocusAndCommonPoints

end polarToRectangular_noCommonPoints_l191_191891


namespace train_length_l191_191236

noncomputable def initial_speed_train : ℝ := 90 * 1000 / 3600 -- m/s
noncomputable def acceleration_train : ℝ := 0.5 -- m/s²
noncomputable def speed_motorbike : ℝ := 72 * 1000 / 3600 -- m/s
noncomputable def overtaking_time : ℝ := 50 -- seconds
noncomputable def length_motorbike : ℝ := 2 -- meters

theorem train_length :
    let final_velocity_train := initial_speed_train + acceleration_train * overtaking_time in
    let distance_train := initial_speed_train * overtaking_time + (1/2) * acceleration_train * overtaking_time ^ 2 in
    let distance_motorbike := speed_motorbike * overtaking_time in
    let length_train := distance_train - distance_motorbike + length_motorbike in
    length_train = 877 :=
by
    sorry

end train_length_l191_191236


namespace find_c_find_A_l191_191032

open Real

noncomputable def acute_triangle_sides (A B C a b c : ℝ) : Prop :=
  a = b * cos C + (sqrt 3 / 3) * c * sin B

theorem find_c (A B C a b c : ℝ) (ha : a = 2) (hb : b = sqrt 7) 
  (hab : acute_triangle_sides A B C a b c) : c = 3 := 
sorry

theorem find_A (A B C : ℝ) (h : sqrt 3 * sin (2 * A - π / 6) - 2 * (sin (C - π / 12))^2 = 0)
  (h_range : π / 6 < A ∧ A < π / 2) : A = π / 4 :=
sorry

end find_c_find_A_l191_191032


namespace max_sqrt_expr_eq_12_l191_191714

noncomputable def max_value_sqrt_expr : ℝ :=
  real.sup (set.image (λ x : ℝ, real.sqrt (36 + x) + real.sqrt (36 - x)) (set.Icc (-36) 36))

theorem max_sqrt_expr_eq_12 : max_value_sqrt_expr = 12 := by
  sorry

end max_sqrt_expr_eq_12_l191_191714


namespace standard_equation_of_circle_l191_191311

-- Define the problem and conditions
def point := ℝ × ℝ

def line (a b c : ℝ) := {p : point // a * p.1 + b * p.2 = c}

def on_line (p : point) (l : line a b c) : Prop :=
  a * p.1 + b * p.2 = c

def distance_squared (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

def circle (center : point) (radius : ℝ) := {p : point // distance_squared p center = radius ^ 2}

-- Given conditions
def A : point := (1, -1)
def B : point := (-1, 1)
def lineC : line 1 1 2 := ⟨(1, 1), rfl⟩

-- Prove the standard equation of the circle
theorem standard_equation_of_circle :
  ∃ (C : point) r, on_line C lineC ∧ circle C r :=
sorry

end standard_equation_of_circle_l191_191311


namespace log_exp_inequality_l191_191952

noncomputable def a : ℝ := Real.log 3
noncomputable def b : ℝ := Real.log 0.5
noncomputable def c : ℝ := 2 ^ (-0.3)

theorem log_exp_inequality : b < c ∧ c < a := by
  have h1 : a = Real.log 3 := rfl
  have h2 : b = Real.log 0.5 := rfl
  have h3 : c = 2 ^ (-0.3) := rfl
  have log_properties : Real.log 3 > 1 ∧ 1 > 0 ∧ 0 > Real.log 0.5 :=
    by sorry -- properties of the logarithmic function
  have exp_properties : 0 < 2 ^ (-0.3) ∧ 2 ^ (-0.3) < 1 := 
    by sorry -- properties of the exponential function
  exact sorry -- combine and prove the inequalities

end log_exp_inequality_l191_191952


namespace forest_leaves_count_correct_l191_191221

def number_of_trees : ℕ := 20
def number_of_main_branches_per_tree : ℕ := 15
def number_of_sub_branches_per_main_branch : ℕ := 25
def number_of_tertiary_branches_per_sub_branch : ℕ := 30
def number_of_leaves_per_sub_branch : ℕ := 75
def number_of_leaves_per_tertiary_branch : ℕ := 45

def total_leaves_on_sub_branches_per_tree :=
  number_of_main_branches_per_tree * number_of_sub_branches_per_main_branch * number_of_leaves_per_sub_branch

def total_sub_branches_per_tree :=
  number_of_main_branches_per_tree * number_of_sub_branches_per_main_branch

def total_leaves_on_tertiary_branches_per_tree :=
  total_sub_branches_per_tree * number_of_tertiary_branches_per_sub_branch * number_of_leaves_per_tertiary_branch

def total_leaves_per_tree :=
  total_leaves_on_sub_branches_per_tree + total_leaves_on_tertiary_branches_per_tree

def total_leaves_in_forest :=
  total_leaves_per_tree * number_of_trees

theorem forest_leaves_count_correct :
  total_leaves_in_forest = 10687500 := 
by sorry

end forest_leaves_count_correct_l191_191221


namespace exists_circle_radius_s_contains_all_l191_191435

variable (A B C : Point)
variable (a b c : ℝ)
variable (x y z s : ℝ)
variable (perimeter_condition : a + b + c = 2 * s)
variable (acute_triangle : is_acute_triangle A B C)
variable (pairwise_disjoint_circles : disjoint (circle A x) (circle B y) ∧ disjoint (circle B y) (circle C z) ∧ disjoint (circle A x) (circle C z))

theorem exists_circle_radius_s_contains_all :
  ∃ (O : Point) (R : ℝ), R = s ∧ contains (circle O R) (circle A x) ∧ contains (circle O R) (circle B y) ∧ contains (circle O R) (circle C z) := 
sorry

end exists_circle_radius_s_contains_all_l191_191435


namespace determinant_matrix_A_l191_191665

open Matrix

def matrix_A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, 0, -2], ![1, 3, 4], ![0, -1, 1]]

theorem determinant_matrix_A :
  det matrix_A = 33 :=
by
  sorry

end determinant_matrix_A_l191_191665


namespace ajhsme_1989_reappears_at_12_l191_191134

def cycle_length_letters : ℕ := 6
def cycle_length_digits  : ℕ := 4
def target_position : ℕ := Nat.lcm cycle_length_letters cycle_length_digits

theorem ajhsme_1989_reappears_at_12 :
  target_position = 12 :=
by
  -- Proof steps can be filled in here
  sorry

end ajhsme_1989_reappears_at_12_l191_191134


namespace problem_statement_l191_191076

noncomputable def r (a b : ℚ) : ℚ := 
  let ab := a * b
  let a_b_recip := a + (1/b)
  let b_a_recip := b + (1/a)
  a_b_recip * b_a_recip

theorem problem_statement (a b : ℚ) (m : ℚ) (h1 : a * b = 3) (h2 : ∃ p, (a + 1 / b) * (b + 1 / a) = (ab + 1 / ab + 2)) :
  r a b = 16 / 3 := by
  sorry

end problem_statement_l191_191076


namespace ratio_lcm_gcf_210_462_l191_191549

theorem ratio_lcm_gcf_210_462 : 
  let a := 210
  let b := 462
  let gcf_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  in lcm_ab / gcf_ab = 55 := 
by 
  let a := 210
  let b := 462
  let gcf_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  have h_gcf : gcf_ab = 42 := by sorry
  have h_lcm : lcm_ab = 2310 := by sorry
  calc
    lcm_ab / gcf_ab = 2310 / 42 : by rw [h_gcf, h_lcm]
    ... = 55 : by norm_num

end ratio_lcm_gcf_210_462_l191_191549


namespace find_x_l191_191443

def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + · · ·)))

theorem find_x (x : ℝ) (h : bowtie 5 x = 12) : x = 42 :=
sorry

end find_x_l191_191443


namespace combined_gold_cost_l191_191324

def gary_gold_weight : ℕ := 30
def gary_gold_cost_per_gram : ℕ := 15
def anna_gold_weight : ℕ := 50
def anna_gold_cost_per_gram : ℕ := 20

theorem combined_gold_cost : (gary_gold_weight * gary_gold_cost_per_gram) + (anna_gold_weight * anna_gold_cost_per_gram) = 1450 :=
by {
  sorry -- Proof goes here
}

end combined_gold_cost_l191_191324


namespace toby_breakfast_calories_l191_191163

noncomputable def calories_bread := 100
noncomputable def calories_peanut_butter_per_serving := 200
noncomputable def servings_peanut_butter := 2

theorem toby_breakfast_calories :
  1 * calories_bread + servings_peanut_butter * calories_peanut_butter_per_serving = 500 :=
by
  sorry

end toby_breakfast_calories_l191_191163


namespace range_of_f_l191_191371

noncomputable def f (x : ℝ) : ℝ := x^(-1/2)

theorem range_of_f : SetOf (y : ℝ) (∃ x > 0, f x = y) = Set.Ioi 0 :=
by
  sorry

end range_of_f_l191_191371


namespace convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191879

theorem convert_polar_to_rectangular (ρ θ x y : ℝ) (hρ : ρ = 2 * √2 * cos θ) :
  (x - √2) ^ 2 + y ^ 2 = 2 :=
sorry

theorem parametric_equations_and_no_common_points (x y θ : ℝ) :
  let A := (1 : ℝ, 0 : ℝ)
  let P (θ : ℝ) := (3 - √2 + 2 * cos θ, 2 * sin θ)
  let C1 := (P 0).1 ^ 2 + (P 0).2 ^ 2 = 4
  (x - √2) ^ 2 + y ^ 2 = 2 ∧ (C1 = [((x, y) = P θ)]) → False :=
sorry

end convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191879


namespace product_of_sequence_l191_191984

noncomputable def sequence (n : ℕ) : ℚ := 1 + (2 / n)

theorem product_of_sequence : (∏ n in Finset.range 100, sequence (n + 1)) = 5151 := by
  sorry

end product_of_sequence_l191_191984


namespace polarToRectangular_noCommonPoints_l191_191887

section PolarToRectangular
variables {θ : ℝ} {ρ : ℝ}

-- Condition: polar coordinate equation ρ = 2√2 cos θ
def polarEq (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Correct answer: rectangular coordinate equation (x - √2)² + y² = 2
theorem polarToRectangular 
  (h : ρ = polarEq θ)
  (x y : ℝ)
  (hx : x = ρ * cos θ)
  (hy : y = ρ * sin θ) :
  (x - sqrt 2) ^ 2 + y ^ 2 = 2 :=
sorry

end PolarToRectangular

section LocusAndCommonPoints
variables {x y x₁ y₁ : ℝ} {C C₁ : set (ℝ × ℝ)}

-- Condition: (1, 0) is point A, M is a moving point on circle C, P satisfies AP = √2AM
def pointA := (1, 0 : ℝ) 
def isOnCircleC (M : ℝ × ℝ) : Prop := (M.1 - sqrt 2) ^ 2 + M.2 ^ 2 = 2
def scaledVector (A P M : ℝ × ℝ) : Prop :=
  P.1 - A.1 = sqrt 2 * (M.1 - A.1) ∧ P.2 - A.2 = sqrt 2 * (M.2 - A.2)

-- Correct answer: Circles C and C₁ do not have any common points
theorem noCommonPoints
  (A : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : isOnCircleC M)
  (P : ℝ × ℝ)
  (hP : scaledVector A P M) :
  ∀ P : ℝ × ℝ, ¬ (isOnCircleC P ∧ (P.1 - (3 - sqrt 2)) ^ 2 + P.2 ^ 2 = 4) :=
sorry

end LocusAndCommonPoints

end polarToRectangular_noCommonPoints_l191_191887


namespace group_sizes_correct_l191_191158

-- Define the number of fruits and groups
def num_bananas : Nat := 527
def num_oranges : Nat := 386
def num_apples : Nat := 319

def groups_bananas : Nat := 11
def groups_oranges : Nat := 103
def groups_apples : Nat := 17

-- Define the expected sizes of each group
def bananas_per_group : Nat := 47
def oranges_per_group : Nat := 3
def apples_per_group : Nat := 18

-- Prove the sizes of the groups are as expected
theorem group_sizes_correct :
  (num_bananas / groups_bananas = bananas_per_group) ∧
  (num_oranges / groups_oranges = oranges_per_group) ∧
  (num_apples / groups_apples = apples_per_group) :=
by
  -- Division in Nat rounds down
  have h1 : num_bananas / groups_bananas = 47 := by sorry
  have h2 : num_oranges / groups_oranges = 3 := by sorry
  have h3 : num_apples / groups_apples = 18 := by sorry
  exact ⟨h1, h2, h3⟩

end group_sizes_correct_l191_191158


namespace bridge_length_correct_l191_191542

-- Define the conditions as given in the original problem
def speed_train_a := 42 * 1000 / 3600 -- in meters/sec
def length_train_a := 500 -- in meters
def crossing_time_a := 60 -- in seconds
def speed_train_b := 70 * 1000 / 3600 -- in meters/sec
def length_train_b := 800 -- in meters

-- Calculate distances and check conditions
def distance_train_a := speed_train_a * crossing_time_a -- train a's distance in 60 seconds
def bridge_length := distance_train_a - length_train_a -- length of the bridge

def total_length_to_cover := length_train_a + length_train_b + bridge_length
def relative_speed := speed_train_a + speed_train_b
def time_to_meet := total_length_to_cover / relative_speed

-- Prove that the length of the bridge is 200.2 meters
theorem bridge_length_correct : bridge_length = 200.2 :=
by
  sorry

end bridge_length_correct_l191_191542


namespace correct_number_of_statements_l191_191354

variables (v n1 n2 : Vector3) (l α β : Plane)

-- Definitions for the problem conditions
def is_parallell (v1 v2 : Vector3) : Prop := ... -- Definition of parallel vectors
def is_perpendicular (v1 v2 : Vector3) : Prop := ... -- Definition of perpendicular vectors
def is_parallel_plane (p1 p2 : Plane) : Prop := ... -- Definition of parallel planes
def is_perpendicular_plane (p1 p2 : Plane) : Prop := ... -- Definition of perpendicular planes

-- Statements to be proved
def statement_1 : Prop := is_parallel n1 n2 ↔ is_parallel_plane α β
def statement_2 : Prop := is_perpendicular n1 n2 ↔ is_perpendicular_plane α β
def statement_3 : Prop := is_parallel v n1 ↔ is_parallel_plane l α
def statement_4 : Prop := is_perpendicular v n1 ↔ is_perpendicular_plane l α

def correct_statements_count : ℕ := 
  [statement_1, statement_2, statement_3, statement_4].count(λ x, x)

theorem correct_number_of_statements : correct_statements_count v n1 n2 l α β = 2 :=
sorry

end correct_number_of_statements_l191_191354


namespace num_integers_in_abs_inequality_l191_191388

theorem num_integers_in_abs_inequality : 
  (∃! n : ℕ, ∀ x : ℤ, abs (x - 3) ≤ 74/10 → x ∈ set.range (λ k : ℤ, -4 + k) ∧ x ≤ 10) :=
sorry

end num_integers_in_abs_inequality_l191_191388


namespace infinite_sum_l191_191664

theorem infinite_sum : 
  (∑' j : ℕ, ∑' k : ℕ, 2 ^ (-(4 * k + j + (k + j) ^ 2))) = 4 / 3 :=
by
  -- filling in the actual proof is not required here
  sorry

end infinite_sum_l191_191664


namespace word_limit_correct_l191_191171

-- Definition for the conditions
def saturday_words : ℕ := 450
def sunday_words : ℕ := 650
def exceeded_amount : ℕ := 100

-- The total words written
def total_words : ℕ := saturday_words + sunday_words

-- The word limit which we need to prove
def word_limit : ℕ := total_words - exceeded_amount

theorem word_limit_correct : word_limit = 1000 := by
  unfold word_limit total_words saturday_words sunday_words exceeded_amount
  sorry

end word_limit_correct_l191_191171


namespace parabola_passes_through_point_l191_191141

theorem parabola_passes_through_point {x y : ℝ} (h_eq : y = (1/2) * x^2 - 2) :
  (x = 2 ∧ y = 0) :=
by
  sorry

end parabola_passes_through_point_l191_191141


namespace chessboard_not_partitionable_into_rook_pairs_l191_191449

def rook_pair (x y : ℕ × ℕ) : Prop :=
  ((x.1 = y.1 ∧ abs (x.2 - y.2) = 3) ∨ (x.2 = y.2 ∧ abs (x.1 - y.1) = 3))

def is_rook_board_partition_possible : Prop :=
  ∀ partition : list (ℕ × ℕ) → Prop,
    (∀ pair ∈ partition, rook_pair pair.fst pair.snd) →
    ∃ (unpaired_cell : ℕ × ℕ),
    unpaired_cell ∉ union partition (λ p, [p.fst, p.snd])

theorem chessboard_not_partitionable_into_rook_pairs :
  ¬ is_rook_board_partition_possible :=
  sorry

end chessboard_not_partitionable_into_rook_pairs_l191_191449


namespace weather_repeats_2047_l191_191924

noncomputable def weather_repeat_year (P : Polynomial ℤ) (Q : Polynomial ℤ) (n : ℕ) : ℕ :=
  sorry -- placeholder for actual function definition

theorem weather_repeats_2047 (P_2015 : Polynomial ℤ) :
  (P_2015.eval 1 ≠ 0) ∧ (∃ Q, Q = (1 + Polynomial.x) ^ 32 % Polynomial.x ^ 31 ∧ (weather_repeat_year P_2015 Q 32 = 2047)) :=
by
  sorry

end weather_repeats_2047_l191_191924


namespace monotonic_increasing_interval_of_sin_2x_minus_pi_over_6_l191_191514

theorem monotonic_increasing_interval_of_sin_2x_minus_pi_over_6 :
  ∀ k : ℤ, monotonic_increasing (λ x : ℝ, sin (2 * x - (real.pi / 6)))
    (set.Icc (k * real.pi - real.pi / 6) (k * real.pi + real.pi / 3)) :=
by
  sorry

end monotonic_increasing_interval_of_sin_2x_minus_pi_over_6_l191_191514


namespace most_axes_of_symmetry_l191_191645

theorem most_axes_of_symmetry
  (equilateral_triangle_axes : ℕ)
  (rectangle_axes : ℕ)
  (square_axes : ℕ)
  (circle_axes : ℕ → Prop) :
  (equilateral_triangle_axes = 3) →
  (rectangle_axes = 2) →
  (square_axes = 4) →
  circle_axes (ℕ) → 
  ∀ n, n ≥ 4 → ¬ (circle_axes n) ∧ (∃ k, k > 4 → circle_axes k) :=
begin
  intro H,
  sorry,
end

end most_axes_of_symmetry_l191_191645


namespace host_not_start_l191_191112

theorem host_not_start (k n : ℕ) (h : ℕ) (H : k = 1 ∧ n = 10 * h - 1) :
  ∀ (p q : ℝ), p = (k : ℝ) / (k + n) ∧ q = ((n : ℝ) / (k + n)) * (k / (k + n - 1)) →
  p - q = k / (k + n) * (k - 1) / (k + n - 1) :=
by
  intro p q
  intro Hpq
  cases Hpq with Hp Hq
  rw [Hp, Hq]
  sorry

end host_not_start_l191_191112


namespace zero_approx_interval_l191_191495

def f (x : ℝ) : ℝ := 2^x + 2 * x - 3

theorem zero_approx_interval :
  ∃ x : ℝ, (1/2 < x) ∧ (x < 1) ∧ (f x = 0) :=
by {
  have h₁ : f (1/2) < 0 := by { sorry },
  have h₂ : f 1 > 0 := by { sorry },
  have h₃ : ∀ a b : ℝ, (a < b) → (f a < 0) → (f b > 0) → ∃ c : ℝ, a < c ∧ c < b ∧ f c = 0 :=
    by { sorry },
  exact h₃ (1/2) 1 (by norm_num) h₁ h₂,
}

end zero_approx_interval_l191_191495


namespace centroid_bisects_area_l191_191577

theorem centroid_bisects_area 
  {A B C : Type} [EuclideanSpace ℝ (Fin₃ → ℝ)]
  (P Q R : A) (G : EuclideanSpace ℝ (Fin₃ → ℝ)) 
  (hG : G = centroid ℝ ![P, Q, R])
  (l : EuclideanSpace ℝ (Fin₃ → ℝ) → EuclideanSpace ℝ (Fin₃ → ℝ))
  (hl : ∀ x, x = G → l x = x) :
  (∀ Δ1 Δ2: EuclideanSpace ℝ (Fin₃ → ℝ), 
    (T : Triangle. ℝ P Q R) ∧ l G = G →
     area T = area (l Δ1) + area (l Δ2) → 
     area (l Δ1) = area (l Δ2)) :=
sorry

end centroid_bisects_area_l191_191577


namespace log_graph_passes_through_point_l191_191131

variable a : ℝ
variable h_a_pos : a > 0
variable h_a_neq_one : a ≠ 1

theorem log_graph_passes_through_point :
  ∃ x y, (x = 3) ∧ (y = 1) ∧ (y = Real.log a (4 - x) + 1) :=
by sorry

end log_graph_passes_through_point_l191_191131


namespace tangent_line_eq_normal_line_eq_tangent_line_at_t₀_normal_line_at_t₀_l191_191277

-- Define the parametric equations for x and y
def x (t : ℝ) : ℝ := t - t^4
def y (t : ℝ) : ℝ := t^2 - t^3

-- Define the point t₀
def t₀ : ℝ := 1

-- Define the corresponding point on the curve (x₀, y₀)
def x₀ : ℝ := x t₀
def y₀ : ℝ := y t₀

-- Compute the derivatives
def dx_dt (t : ℝ) : ℝ := 1 - 4 * t^3
def dy_dt (t : ℝ) : ℝ := 2 * t - 3 * t^2

-- Evaluate the slope of the tangent line at t₀
def slope_tangent_line : ℝ := (dy_dt t₀) / (dx_dt t₀)

-- Prove tangent line equation
theorem tangent_line_eq : ∀ x : ℝ, y₀ + (y - y₀) = (slope_tangent_line) * (x - x₀) :=
by sorry

-- Prove normal line equation
theorem normal_line_eq : ∀ x : ℝ, y₀ + (y - y₀) = (-(1 / slope_tangent_line)) * (x - x₀) :=
by sorry

-- Prove the specific equations of tangent and normal lines
theorem tangent_line_at_t₀ : ∀ x : ℝ, y₀ + (y - y₀) = (1/3) * (x - x₀) :=
by sorry

theorem normal_line_at_t₀ : ∀ x : ℝ, y₀ + (y - y₀) = -3 * (x - x₀) :=
by sorry

end tangent_line_eq_normal_line_eq_tangent_line_at_t₀_normal_line_at_t₀_l191_191277


namespace emma_missing_coins_l191_191295

theorem emma_missing_coins (x : ℝ) (hx : x > 0) :
  let lost_coins := (2 / 3) * x in
  let found_coins := (3 / 4) * lost_coins in
  let remaining_lost_coins := lost_coins - found_coins in
  remaining_lost_coins / x = (1 / 6) :=
by
  sorry

end emma_missing_coins_l191_191295


namespace mabel_tomatoes_l191_191976

theorem mabel_tomatoes :
  ∃ (t1 t2 t3 t4 : ℕ), 
    t1 = 8 ∧ 
    t2 = t1 + 4 ∧ 
    t3 = 3 * (t1 + t2) ∧ 
    t4 = 3 * (t1 + t2) ∧ 
    (t1 + t2 + t3 + t4) = 140 :=
by
  sorry

end mabel_tomatoes_l191_191976


namespace geometric_sequence_a3_is_15_l191_191405

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
a1 * q^(n - 1)

theorem geometric_sequence_a3_is_15 (q : ℝ) (a1 : ℝ) (a5 : ℝ) 
  (h1 : a1 = 3) (h2 : a5 = 75) (h_seq : ∀ n, a5 = geometric_sequence a1 q n) :
  geometric_sequence a1 q 3 = 15 :=
by 
  sorry

end geometric_sequence_a3_is_15_l191_191405


namespace weight_of_new_person_l191_191201

def total_weight_increase (num_people : ℕ) (weight_increase_per_person : ℝ) : ℝ :=
  num_people * weight_increase_per_person

def new_person_weight (old_person_weight : ℝ) (total_weight_increase : ℝ) : ℝ :=
  old_person_weight + total_weight_increase

theorem weight_of_new_person :
  let old_person_weight := 50
  let num_people := 8
  let weight_increase_per_person := 2.5
  new_person_weight old_person_weight (total_weight_increase num_people weight_increase_per_person) = 70 := 
by
  sorry

end weight_of_new_person_l191_191201


namespace angle_PMN_45_degrees_l191_191926

variables {P Q R M N : Type*} [EuclideanPlane P Q R M N]

-- Angle definitions
variables {angle_PQR : ℝ} {angle_PM : ℝ} {angle_PN : ℝ} {angle_PMN : ℝ}

-- Conditions
def given_conditions (angle_PQR = 60) (PR ≠ RQ) (PM = PN) (PQ_perpendicular_MN : PQ ⊥ MN) : Prop :=
  angle_PQR = 60 ∧ PR ≠ RQ ∧ PM = PN ∧ PQ ∠ 90

-- Statement to prove
theorem angle_PMN_45_degrees : 
  given_conditions angle_PQR PR RQ PM PN PQ ⊥ MN → angle_PMN = 45 :=
begin
  sorry
end

end angle_PMN_45_degrees_l191_191926


namespace mabel_total_tomatoes_l191_191970

theorem mabel_total_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 :=
by
  sorry

end mabel_total_tomatoes_l191_191970


namespace g_242_l191_191522

noncomputable def g : ℕ → ℝ := sorry

axiom g_property : ∀ (x y m : ℕ), x > 0 → y > 0 → m > 0 → x + y = 3^m → g(x) + g(y) = (m + 1)^2

theorem g_242 : g 242 = 24.5 :=
by
  sorry

end g_242_l191_191522


namespace factorize_n_squared_minus_nine_l191_191693

theorem factorize_n_squared_minus_nine (n : ℝ) : n^2 - 9 = (n + 3) * (n - 3) := 
sorry

end factorize_n_squared_minus_nine_l191_191693


namespace max_points_nxngrid_l191_191053

theorem max_points_nxngrid (n : ℕ) (h : n ≥ 3) (h_odd : odd n) : 
    ∃ max_points : ℕ, max_points = n * (n + 1) :=
begin
  let max_points := n * (n + 1),
  use max_points,
  sorry
end

end max_points_nxngrid_l191_191053


namespace remainder_b_div_11_l191_191064

theorem remainder_b_div_11 (n : ℕ) (h_pos : 0 < n) (b : ℕ) (h_b : b ≡ (5^(2*n) + 6)⁻¹ [ZMOD 11]) : b % 11 = 8 :=
by
  sorry

end remainder_b_div_11_l191_191064


namespace polarToRectangular_noCommonPoints_l191_191890

section PolarToRectangular
variables {θ : ℝ} {ρ : ℝ}

-- Condition: polar coordinate equation ρ = 2√2 cos θ
def polarEq (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Correct answer: rectangular coordinate equation (x - √2)² + y² = 2
theorem polarToRectangular 
  (h : ρ = polarEq θ)
  (x y : ℝ)
  (hx : x = ρ * cos θ)
  (hy : y = ρ * sin θ) :
  (x - sqrt 2) ^ 2 + y ^ 2 = 2 :=
sorry

end PolarToRectangular

section LocusAndCommonPoints
variables {x y x₁ y₁ : ℝ} {C C₁ : set (ℝ × ℝ)}

-- Condition: (1, 0) is point A, M is a moving point on circle C, P satisfies AP = √2AM
def pointA := (1, 0 : ℝ) 
def isOnCircleC (M : ℝ × ℝ) : Prop := (M.1 - sqrt 2) ^ 2 + M.2 ^ 2 = 2
def scaledVector (A P M : ℝ × ℝ) : Prop :=
  P.1 - A.1 = sqrt 2 * (M.1 - A.1) ∧ P.2 - A.2 = sqrt 2 * (M.2 - A.2)

-- Correct answer: Circles C and C₁ do not have any common points
theorem noCommonPoints
  (A : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : isOnCircleC M)
  (P : ℝ × ℝ)
  (hP : scaledVector A P M) :
  ∀ P : ℝ × ℝ, ¬ (isOnCircleC P ∧ (P.1 - (3 - sqrt 2)) ^ 2 + P.2 ^ 2 = 4) :=
sorry

end LocusAndCommonPoints

end polarToRectangular_noCommonPoints_l191_191890


namespace circle_center_sum_l191_191499

theorem circle_center_sum (h k : ℝ) :
  (∃ h k : ℝ, ∀ x y : ℝ, (x^2 + y^2 = 6 * x + 8 * y - 15) → (h, k) = (3, 4)) →
  h + k = 7 :=
by
  sorry

end circle_center_sum_l191_191499


namespace part1_part2_l191_191786

-- The function definition
def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x^2 + 1

-- Derivative of the function
def f' (x : ℝ) (a : ℝ) : ℝ := 1/x - 2 * a * x

-- First part: Extreme value condition
theorem part1 (a : ℝ) : 
  f' 4 a = 0 → a = 1 / 32 :=
by
  sorry

-- Second part: Monotonic decreasing condition
theorem part2 (a : ℝ) : 
  (∀ x : ℝ, 3 < x → f' x a ≤ 0) → a ≥ 1 / 18 :=
by
  sorry

end part1_part2_l191_191786


namespace matching_seat_row_exists_minimum_matching_seat_row_l191_191204

theorem matching_seat_row_exists 
  (m n : ℕ) 
  (tickets : List (ℕ × ℕ)) 
  (h1 : tickets.length = m * n) 
  (h2 : ∀ (ticket : ℕ × ℕ), ticket ∈ tickets → ticket.fst < m ∧ ticket.snd < n)
  (h3 : ∀ (person : ℕ) (row seat : ℕ), 
        ∃ ticket ∈ tickets, (ticket.fst = row) ∨ (ticket.snd = seat)) : 
  ∃ (person : ℕ) (row seat : ℕ), 
    row < m ∧ seat < n ∧ (row, seat) ∈ tickets ∧ row = person ∧ seat = person := sorry

theorem minimum_matching_seat_row 
  (m n : ℕ) 
  (tickets : List (ℕ × ℕ)) 
  (h1 : tickets.length = m * n) 
  (h2 : ∀ (ticket : ℕ × ℕ), ticket ∈ tickets → ticket.fst < m ∧ ticket.snd < n) 
  (h3 : ∀ (person : ℕ) (row seat : ℕ), 
        ∃ ticket ∈ tickets, (ticket.fst = row) ∨ (ticket.snd = seat))
  (h4 : ∀ (row seat : ℕ), row < m ∧ seat < n → (row, seat) ∉ tickets) : 
  ∃ (row seat : ℕ),
    (∃ (suspected_tickets : List (ℕ × ℕ)), 
      ∀ (ticket ∈ suspected_tickets), ticket ∈ tickets ∧ suspected_tickets.length = 1 ∧ 
      ∃ person, person < m ∧ row < m ∧ seat < n ∧ ticket = (row, seat) ∧ ticket.fst = person ∧ ticket.snd = person) := sorry

end matching_seat_row_exists_minimum_matching_seat_row_l191_191204


namespace number_is_composite_l191_191993

-- Definition: A number is composite if it has a divisor other than 1 and itself.
def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem number_is_composite (n : ℕ) (h : n > 1) :
  is_composite (1 / 3 * (2^(2^(n + 1)) + 2^(2^n) + 1)) :=
begin
  sorry
end

end number_is_composite_l191_191993


namespace proof_problem_l191_191869

section Exercise

variable (θ : ℝ)

def polarCoordEq (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos θ

def toRectCoord (ρ θ : ℝ) : Prop := ρ^2 = (2 * Real.sqrt 2 * Real.cos θ) * ρ

def circleEq (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 2

def parametricLocusC1 (θ : ℝ) : (ℝ × ℝ) := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

theorem proof_problem :
  (∃ ρ θ, polarCoordEq ρ θ → circleEq (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  ∀ θ, ¬ ∃ x y, (circleEq x y) ∧ (parametricLocusC1 θ = (x, y)) :=
by 
  sorry

end Exercise

end proof_problem_l191_191869


namespace polar_to_rectangular_eq_no_common_points_l191_191855

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_rectangular_eq (ρ θ x y: ℝ) (h1: ρ = 2 * Real.sqrt 2 * Real.cos θ) 
(h2: polar_to_rectangular ρ θ = (x, y)) : (x - Real.sqrt 2) ^ 2 + y ^ 2 = 2 :=
by
  sorry

theorem no_common_points (x y θ : ℝ) (ρ : ℝ := 2 * Real.sqrt 2 * Real.cos θ)
(A := (1, 0) : ℝ × ℝ)
(M := polar_to_rectangular ρ θ)
(P := (sqrt 2 * (M.1 - 1) + 1, sqrt 2 * M.2)) : 
((P.1 - (3 - Real.sqrt 2)) ^ 2 + P.2 ^ 2 = 4) → 
¬ ((P.1 - Real.sqrt 2) ^ 2 + P.2 ^ 2 = 2) :=
by
  intro H1 H2
  sorry

end polar_to_rectangular_eq_no_common_points_l191_191855


namespace area_triangle_ABC_l191_191428

open Real

-- Assuming the coordinates of points and other required definitions
structure Triangle :=
  (A B C : ℝ × ℝ)

structure Midpoint (p1 p2 : ℝ × ℝ) :=
  (M : ℝ × ℝ)
  (mid_property : M = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2))

structure Trisection (p1 p2 : ℝ × ℝ) :=
  (P Q : ℝ × ℝ)
  (trisect_property : P = ((2 * p1.1 + p2.1) / 3, (2 * p1.2 + p2.2) / 3) ∧ Q = ((p1.1 + 2 * p2.1) / 3, (p1.2 + 2 * p2.2) / 3))

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

theorem area_triangle_ABC {A B C M N P Q : ℝ × ℝ}
  (midpoints : Midpoint A B)
  (midpointn : Midpoint A C)
  (trisection : Trisection B C)
  (circum_pts : ∃ (O : ℝ × ℝ) (r : ℝ), dist O A = r ∧ dist O M = r ∧ dist O N = r ∧ dist O P = r ∧ dist O Q = r)
  (BC_eq_1 : dist B C = 1) :
  area_of_triangle A B C = sqrt 7 / 12 := by sorry

end area_triangle_ABC_l191_191428


namespace fraction_of_girls_is_221_over_440_l191_191650

-- Define the conditions as given data points
def rawlings_total_students : ℕ := 240
def waverly_total_students : ℕ := 200

def rawlings_boys_to_girls_ratio := (3 : ℕ) / (2 : ℕ)
def waverly_boys_to_girls_ratio := (3 : ℕ) / (5 : ℕ)

-- Calculate the number of girls in each school
noncomputable def rawlings_girls : ℕ :=
  let boys := 3 * 48 in
  let girls := 2 * 48 in
  girls

noncomputable def waverly_girls : ℕ :=
  let boys := 3 * 25 in
  let girls := 5 * 25 in
  girls

-- Total number of girls at the event
def total_girls : ℕ := rawlings_girls + waverly_girls
def total_students : ℕ := rawlings_total_students + waverly_total_students

-- The fraction we need to prove
def fraction_girls : ℚ := (total_girls : ℚ) / (total_students : ℚ)

theorem fraction_of_girls_is_221_over_440 :
  fraction_girls = (221 : ℚ) / (440 : ℚ) :=
  by sorry

end fraction_of_girls_is_221_over_440_l191_191650


namespace even_integers_units_digit_l191_191553

theorem even_integers_units_digit :
  let evens_with_5 := [50, 52, 54, 56, 58]
  ∃ p : ℕ, p = List.prod evens_with_5 ∧ (p % 10) = 0 :=
begin
  sorry
end

end even_integers_units_digit_l191_191553


namespace quadratic_form_proof_l191_191480

theorem quadratic_form_proof (k : ℝ) (a b c : ℝ) (h1 : 8*k^2 - 16*k + 28 = a * (k + b)^2 + c) (h2 : a = 8) (h3 : b = -1) (h4 : c = 20) : c / b = -20 :=
by {
  sorry
}

end quadratic_form_proof_l191_191480


namespace share_of_e_l191_191587

variable (E F : ℝ)
variable (D : ℝ := (5/3) * E)
variable (D_alt : ℝ := (1/2) * F)
variable (E_alt : ℝ := (3/2) * F)
variable (profit : ℝ := 25000)

theorem share_of_e (h1 : D = (5/3) * E) (h2 : D = (1/2) * F) (h3 : E = (3/2) * F) :
  (E / ((5/2) * F + (3/2) * F + F)) * profit = 7500 :=
by
  sorry

end share_of_e_l191_191587


namespace mabel_tomatoes_l191_191972

theorem mabel_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 := by
  sorry

end mabel_tomatoes_l191_191972


namespace minimum_A_cars_required_maximum_profit_achieved_l191_191835
noncomputable theory

def total_cars := 20
def cost_price_A := 16
def selling_price_A := 16.8
def cost_price_B := 28
def selling_price_B := 29.4
def profit_A := selling_price_A - cost_price_A
def profit_B := selling_price_B - cost_price_B

def minimum_A_cars := 15
def maximum_profit := 19

-- Part 1: At least 15 A type cars should be purchased
theorem minimum_A_cars_required : ∀ (x : ℕ), 3 * (total_cars - x) ≤ x → x ≥ minimum_A_cars := sorry

-- Part 2: Maximum profit occurs with 15 A type cars
theorem maximum_profit_achieved : ∀ (x : ℕ), x ≥ minimum_A_cars → 
                                    profit_A * (x : ℝ) + profit_B * (total_cars - x) ≤ maximum_profit :=
sorry

end minimum_A_cars_required_maximum_profit_achieved_l191_191835


namespace find_f_neg_3_l191_191788

theorem find_f_neg_3
    (a : ℝ)
    (f : ℝ → ℝ)
    (h : ∀ x, f x = a^2 * x^3 + a * Real.sin x + abs x + 1)
    (h_f3 : f 3 = 5) :
    f (-3) = 3 :=
by
    sorry

end find_f_neg_3_l191_191788


namespace fraction_identity_l191_191812

theorem fraction_identity (a b : ℝ) (h : 2 * a = 5 * b) : a / b = 5 / 2 := by
  sorry

end fraction_identity_l191_191812


namespace solve_laplace_dirichlet_bvp_in_cylinder_l191_191486

noncomputable def dirichlet_bvp_solution (r z : ℝ) (mu J0 J1 J2 sinh : ℝ → ℝ) : ℝ :=
  ∑' n : ℕ, (4 * J2 (mu n)) / ((mu n)^2 * (J1 (mu n))^2 * sinh (mu n)) * J0 (mu n * r) * sinh (mu n * (1 - z))

theorem solve_laplace_dirichlet_bvp_in_cylinder
  (u : ℝ → ℝ → ℝ)
  (laplace : (ℝ → ℝ → ℝ) → ℝ → ℝ → ℝ)
  (J0 J1 J2 : ℝ → ℝ)
  (mu : ℕ → ℝ)
  (sinh : ℝ → ℝ) :
  (∀ r z, 0 ≤ r → r < 1 → 0 < z → z < 1 → laplace u r z = 0) →
  (∀ r, 0 ≤ r → r < 1 → u r 0 = 1 - r^2) →
  (∀ r, 0 ≤ r → r < 1 → u r 1 = 0) →
  (∀ z, 0 < z → z < 1 → u 1 z = 0) →
  (∀ r z, u r z = dirichlet_bvp_solution r z mu J0 J1 J2 sinh) :=
begin
  sorry
end

end solve_laplace_dirichlet_bvp_in_cylinder_l191_191486


namespace teorema_dos_bicos_white_gray_eq_angle_x_l191_191584

-- Define the problem statement
theorem teorema_dos_bicos_white_gray_eq
    (n : ℕ)
    (AB CD : ℝ)
    (peaks : Fin n → ℝ)
    (white_angles gray_angles : Fin n → ℝ)
    (h_parallel : AB = CD)
    (h_white_angles : ∀ i, white_angles i = peaks i)
    (h_gray_angles : ∀ i, gray_angles i = peaks i):
    (Finset.univ.sum white_angles) = (Finset.univ.sum gray_angles) := sorry

theorem angle_x
    (AB CD : ℝ)
    (x : ℝ)
    (h_parallel : AB = CD):
    x = 32 := sorry

end teorema_dos_bicos_white_gray_eq_angle_x_l191_191584


namespace probability_of_A_l191_191202

theorem probability_of_A (P : set (set ℝ) → ℝ) (A B : set ℝ)
  (h_probB : P B = 0.4) 
  (h_probAB : P (A ∩ B) = 0.25) 
  (h_probAorB : P (A ∪ B) = 0.6) : 
  P A = 0.45 :=
by
  sorry

end probability_of_A_l191_191202


namespace total_gold_cost_l191_191323

-- Given conditions
def gary_grams : ℕ := 30
def gary_cost_per_gram : ℕ := 15
def anna_grams : ℕ := 50
def anna_cost_per_gram : ℕ := 20

-- Theorem statement to prove
theorem total_gold_cost :
  (gary_grams * gary_cost_per_gram + anna_grams * anna_cost_per_gram) = 1450 := 
by
  sorry

end total_gold_cost_l191_191323


namespace parts_of_a_number_l191_191620

theorem parts_of_a_number 
  (a p q : ℝ) 
  (x y z : ℝ)
  (h1 : y + z = p * x)
  (h2 : x + y = q * z)
  (h3 : x + y + z = a) :
  x = a / (1 + p) ∧ y = a * (p * q - 1) / ((p + 1) * (q + 1)) ∧ z = a / (1 + q) := 
by 
  sorry

end parts_of_a_number_l191_191620


namespace monotonically_increasing_intervals_l191_191137

theorem monotonically_increasing_intervals :
  let f := λ x : ℝ, (1/3) * x^3 + (1/2) * x^2 in
  ∀ x : ℝ, f x > 0 ∨ x ∈ (-∞, -1) ∪ (0, ∞) :=
by
  sorry

end monotonically_increasing_intervals_l191_191137


namespace problem_statement_l191_191772

noncomputable def cos2x_over_cos_pi4_plus_x (x : ℝ) (h1 : sin (π/4 - x) = 5/13) (h2 : 0 < x ∧ x < π/4) : ℝ :=
  cos (2 * x) / cos (π/4 + x)

theorem problem_statement (x : ℝ) (h1 : sin (π/4 - x) = 5/13) (h2 : 0 < x ∧ x < π/4) :
  cos2x_over_cos_pi4_plus_x x h1 h2 = 24/13 :=
sorry

end problem_statement_l191_191772


namespace soda_recipes_needed_l191_191651

theorem soda_recipes_needed (students : ℕ) (absence_rate : ℝ) (sodas_per_student : ℕ) (sodas_per_recipe : ℕ) :
  students = 144 →
  absence_rate = 0.35 →
  sodas_per_student = 3 →
  sodas_per_recipe = 18 →
  let attending_students := (students * (1 - absence_rate).floor).to_nat,
      total_sodas_needed := attending_students * sodas_per_student,
      recipes_needed := (total_sodas_needed / sodas_per_recipe).ceil in
  recipes_needed = 16 := by
  intros
  let attending_students := (students * (1 - absence_rate).floor).to_nat
  let total_sodas_needed := attending_students * sodas_per_student
  let recipes_needed := (total_sodas_needed / sodas_per_recipe).ceil
  have : recipes_needed = 16 := sorry
  exact this

end soda_recipes_needed_l191_191651


namespace Nell_gave_26_cards_to_Jeff_l191_191982

-- Defining the conditions
def initial_cards : ℕ := 573
def bought_cards : ℕ := 127
def gave_Jack : ℕ := 195
def gave_Jimmy : ℕ := 75
def gave_Jeff_percentage : ℚ := 0.06
def remaining_cards_after_all : ℕ := 210

-- They represent the total card count calculated here
def total_cards : ℕ := initial_cards + bought_cards

-- Cards given away before Jeff
def cards_given_before_Jeff : ℕ := gave_Jack + gave_Jimmy

-- Remaining cards calculated before giving to Jeff
def remaining_cards_before_Jeff : ℕ := total_cards - cards_given_before_Jeff

-- Amount given to Jeff calculated
def gave_Jeff : ℕ := (gave_Jeff_percentage * remaining_cards_before_Jeff).nat_ceil

-- Lean 4 proof statement
theorem Nell_gave_26_cards_to_Jeff :
  gave_Jeff = 26 :=
by
  sorry

end Nell_gave_26_cards_to_Jeff_l191_191982


namespace monotonically_increasing_intervals_inequality_solution_set_l191_191745

-- Given conditions for f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a*x^3 + b*x^2 + c*x + d

-- Ⅰ) Prove the intervals of monotonic increase
theorem monotonically_increasing_intervals (a c : ℝ) (x : ℝ) (h_f : ∀ x, f a 0 c 0 x = a*x^3 + c*x)
  (h_a : a = 1) (h_c : c = -3) :
  (∀ x < -1, f a 0 c 0 x < 0) ∧ (∀ x > 1, f a 0 c 0 x > 0) := 
sorry

-- Ⅱ) Prove the solution sets for the inequality given m
theorem inequality_solution_set (m x : ℝ) :
  (m = 0 → x > 0) ∧
  (m > 0 → (x > 4*m ∨ 0 < x ∧ x < m)) ∧
  (m < 0 → (x > 0 ∨ 4*m < x ∧ x < m)) :=
sorry

end monotonically_increasing_intervals_inequality_solution_set_l191_191745


namespace num_three_person_subcommittees_from_eight_l191_191802

def num_committees (n k : ℕ) : ℕ := (Nat.fact n) / ((Nat.fact k) * (Nat.fact (n - k)))

theorem num_three_person_subcommittees_from_eight (n : ℕ) (h : n = 8) : num_committees n 3 = 56 :=
by
  rw [h]
  sorry

end num_three_person_subcommittees_from_eight_l191_191802


namespace point_outside_circle_l191_191009

theorem point_outside_circle (diam : ℝ) (dist : ℝ) (r : ℝ) (diam_eq : diam = 10) (dist_eq : dist = 6) (r_eq : r = diam / 2) :
  dist > r :=
by {
  rw [diam_eq, dist_eq, r_eq],
  norm_num,
  exact lt_trans zero_lt_five six_gt_five,
}

end point_outside_circle_l191_191009


namespace triangle_area_correct_l191_191237

noncomputable def vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := ((4, -1), (12, 7), (4, 7))

theorem triangle_area_correct : 
  let (v1, v2, v3) := vertices in
  ∃ A : ℝ, A = 32.0 ∧ 
    (let x1 := v1.1, y1 := v1.2 
         x2 := v2.1, y2 := v2.2 
         x3 := v3.1, y3 := v3.2
     in (x1 = 4 ∧ y1 = -1) ∧ 
        (x2 = 12 ∧ y2 = 7) ∧ 
        (x3 = 4 ∧ y3 = 7) ∧ 
        A = (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))) :=
sorry

end triangle_area_correct_l191_191237


namespace problem1_l191_191005

theorem problem1 (x y : ℝ) (h : |x + 1| + (2 * x - y)^2 = 0) : x^2 - y = 3 :=
sorry

end problem1_l191_191005


namespace solve_equation_l191_191300

theorem solve_equation (x : ℝ) : (sqrt (sqrt x) = 15 / (8 - sqrt (sqrt x))) ↔ (x = 625 ∨ x = 81) := 
by
  sorry

end solve_equation_l191_191300


namespace identify_S_l191_191073

-- Define the polynomials and the conditions
def P (z : ℂ) : ℂ := sorry -- P(z) is some polynomial
def S (z : ℂ) : ℂ := z + 2 -- S(z) is the polynomial to be identified
def f (z : ℂ) : ℂ := z^2023 + 2 -- The given polynomial
def g (z : ℂ) : ℂ := z^3 + z^2 + 1 -- The divisor polynomial

theorem identify_S :
  ∃ (P S : ℂ → ℂ), f = (g * P) + S ∧ degree S < 3 ∧ (∀ z, S z = z + 2) :=
by {
  use P,
  use S,
  split,
  { sorry }, -- Proof that z^2023 + 2 = (z^3 + z^2 + 1)P(z) + S(z)
  split,
  { sorry }, -- Proof that degree S < 3
  { intros z, sorry } -- Proof that S(z) = z + 2
}

end identify_S_l191_191073


namespace possible_remainders_of_b_l191_191067

-- Define the conditions
def congruent_mod (a b n : ℤ) : Prop := (a - b) % n = 0

variables {n : ℤ} (hn : n > 0)

theorem possible_remainders_of_b (n : ℤ) (hn : n > 0) :
  ∃ b : ℤ, b ∈ {8, 5, 3, 10} ∧ 
           (congruent_mod b (5^(2*n) + 6)⁻¹ 11 ∧
           ¬ congruent_mod (5^(2*n) + 6) 0 11) :=
by sorry

end possible_remainders_of_b_l191_191067


namespace correct_conclusion_l191_191523

noncomputable def proof_problem (a x : ℝ) (x1 x2 : ℝ) :=
  (a * (x - 1) * (x - 3) + 2 > 0 ∧ x1 < x2 ∧ 
   (∀ x, a * (x - 1) * (x - 3) + 2 > 0 ↔ x < x1 ∨ x > x2)) →
  (x1 + x2 = 4 ∧ 3 < x1 * x2 ∧ x1 * x2 < 4 ∧ 
   (∀ x, ((3 * a + 2) * x^2 - 4 * a * x + a < 0) ↔ (1 / x2 < x ∧ x < 1 / x1)))

theorem correct_conclusion (a x x1 x2 : ℝ) : 
proof_problem a x x1 x2 :=
by 
  unfold proof_problem 
  sorry

end correct_conclusion_l191_191523


namespace incircle_centers_on_line_l191_191165

noncomputable theory
open set

variables
  (w₁ w₂ : Circle)
  (l : Line)
  (m : Line)
  (X : Line)
  (Y Z : Line)
  (XY : X)
  (XZ : X)
  (containsXY : XY ∈ w₁)
  (containsXZ : XZ ∈ w₂)

def triangle_contains (t : Triangle) (c : Circle) := sorry --definition that tracks containment

def incircle_center_on_line (t : Triangle) (c₁ c₂ : Circle) : Prop :=
  ∃ l' : Line, ∀ (p : point), is_incenter p t → p ∈ l'

theorem incircle_centers_on_line : 
  ∀ (w₁ w₂ : Circle) (l m : Line) (X Y Z : Point), 
    X ∈ m ∧ Y ∈ l ∧ Z ∈ l ∧ tangent_to_circle X Y w₁ ∧ tangent_to_circle X Z w₂ ∧
    contains w₁ (Triangle.mk X Y Z) ∧ contains w₂ (Triangle.mk X Y Z) → 
    incircle_center_on_line (Triangle.mk X Y Z) w₁ w₂ :=
begin
  sorry
end

end incircle_centers_on_line_l191_191165


namespace bigger_number_l191_191580

theorem bigger_number (yoongi : ℕ) (jungkook : ℕ) (h1 : yoongi = 4) (h2 : jungkook = 6 + 3) : jungkook > yoongi :=
by
  sorry

end bigger_number_l191_191580


namespace proof_problem_l191_191868

section Exercise

variable (θ : ℝ)

def polarCoordEq (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos θ

def toRectCoord (ρ θ : ℝ) : Prop := ρ^2 = (2 * Real.sqrt 2 * Real.cos θ) * ρ

def circleEq (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 2

def parametricLocusC1 (θ : ℝ) : (ℝ × ℝ) := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

theorem proof_problem :
  (∃ ρ θ, polarCoordEq ρ θ → circleEq (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  ∀ θ, ¬ ∃ x y, (circleEq x y) ∧ (parametricLocusC1 θ = (x, y)) :=
by 
  sorry

end Exercise

end proof_problem_l191_191868


namespace point_outside_circle_l191_191010

theorem point_outside_circle (diam : ℝ) (dist : ℝ) (r : ℝ) (diam_eq : diam = 10) (dist_eq : dist = 6) (r_eq : r = diam / 2) :
  dist > r :=
by {
  rw [diam_eq, dist_eq, r_eq],
  norm_num,
  exact lt_trans zero_lt_five six_gt_five,
}

end point_outside_circle_l191_191010


namespace angle_between_vectors_l191_191797

variables (a b : EuclideanSpace ℝ (Fin 3))
noncomputable def theta : ℝ := arccos (-sqrt 3 / 2)

theorem angle_between_vectors (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (hab : a ⬝ b = -sqrt 3) : 
  angle a b = theta := by
  sorry

end angle_between_vectors_l191_191797


namespace distance_between_lights_l191_191451

theorem distance_between_lights : 
  let pattern := [red, green, red, green, green] in
  let distance_between : Nat := 4 in
  let red_positions : List Nat := 
    (List.range 130).filter (λ n => pattern.get! (n % pattern.length) == red) in
  let pos_4th_red := red_positions.get! (4 - 1) in
  let pos_26th_red := red_positions.get! (26 - 1) in
  let num_gaps := pos_26th_red - pos_4th_red in
  let total_distance_in_inches := num_gaps * distance_between in
  let total_distance_in_feet := total_distance_in_inches / 12 in
  total_distance_in_feet = 18.33 :=
by
  sorry

end distance_between_lights_l191_191451


namespace event_d_is_certain_l191_191253

theorem event_d_is_certain : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ (3 * x^2 - 8 * x + 5 = 0) ∧ (x = 1) := 
by
  use 1
  split
  -- 0 ≤ x ∧ x ≤ 1 step
  split
  -- proof for 0 ≤ 1
  norm_num
  -- proof for 1 ≤ 1
  norm_num
  -- 3 * x^2 - 8 * x + 5 = 0 step
  split
  norm_num
  ring_nf
  -- x = 1 step
  rfl

end event_d_is_certain_l191_191253


namespace problem1_problem2_l191_191205

theorem problem1 : (sqrt 18 - sqrt 32 + sqrt 2) = 0 :=
sorry

theorem problem2 : (sqrt 12 * (sqrt 3 / 2) / sqrt 2) = (3 * sqrt 2 / 2) :=
sorry

end problem1_problem2_l191_191205


namespace jade_handled_84_transactions_l191_191200

def Mabel_transactions : ℕ := 90

def Anthony_transactions (mabel : ℕ) : ℕ := mabel + mabel / 10

def Cal_transactions (anthony : ℕ) : ℕ := (2 * anthony) / 3

def Jade_transactions (cal : ℕ) : ℕ := cal + 18

theorem jade_handled_84_transactions :
  Jade_transactions (Cal_transactions (Anthony_transactions Mabel_transactions)) = 84 := 
sorry

end jade_handled_84_transactions_l191_191200


namespace trisect_pi_over_n_l191_191994

-- Define the condition that n is not a multiple of 3
def not_multiple_of_three (n : ℕ) : Prop :=
  ¬ (∃ k : ℕ, n = 3 * k)

-- Define the trisection constructibility of an angle
def trisectible_with_ruler_and_compass (α : ℝ) : Prop :=
  sorry -- Definition of trisectibility would require more domain-specific formalization

-- The main theorem statement
theorem trisect_pi_over_n (n : ℕ) (h : not_multiple_of_three n) : trisectible_with_ruler_and_compass (real.pi / n) :=
  sorry -- Here, we put the proof of the statement

end trisect_pi_over_n_l191_191994


namespace base7_of_2345_l191_191286

def decimal_to_base7 (n : ℕ) : ℕ :=
  6 * 7^3 + 5 * 7^2 + 6 * 7^1 + 0 * 7^0

theorem base7_of_2345 : decimal_to_base7 2345 = 6560 := by
  sorry

end base7_of_2345_l191_191286


namespace solve_fraction_problem_l191_191554

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l191_191554


namespace circle_radius_l191_191615

theorem circle_radius (k r : ℝ) (h : k > 8) 
  (h1 : r = |k - 8|)
  (h2 : r = k / Real.sqrt 5) : 
  r = 8 * Real.sqrt 5 + 8 := 
sorry

end circle_radius_l191_191615


namespace magnitude_of_z_l191_191963

open Complex Real

theorem magnitude_of_z (r : ℝ) (z : ℂ) (h1 : abs r < sqrt 8) (h2 : z + 1 / z = r) : abs z = 1 := by
  sorry

end magnitude_of_z_l191_191963


namespace find_start_number_l191_191023

def count_even_not_divisible_by_3 (start end_ : ℕ) : ℕ :=
  (end_ / 2 + 1) - (end_ / 6 + 1) - (if start = 0 then start / 2 else start / 2 + 1 - (start - 1) / 6 - 1)

theorem find_start_number (start end_ : ℕ) (h1 : end_ = 170) (h2 : count_even_not_divisible_by_3 start end_ = 54) : start = 8 :=
by 
  rw [h1] at h2
  sorry

end find_start_number_l191_191023


namespace coefficient_x_squared_in_expansion_l191_191770

theorem coefficient_x_squared_in_expansion :
  let n := ∫ x in 1..(Real.exp 6), 1/x
  n = 6 →
  let expr := (fun x : ℂ => (x - (3 / x))^n)
  (expr.expand).coeff(Complex.) (= 135)    

end coefficient_x_squared_in_expansion_l191_191770


namespace harry_total_cost_in_silver_l191_191381

def cost_of_spellbooks_in_gold := 5 * 5
def cost_of_potion_kits_in_silver := 3 * 20
def cost_of_owl_in_gold := 28
def gold_to_silver := 9

def cost_in_silver :=
  (cost_of_spellbooks_in_gold + cost_of_owl_in_gold) * gold_to_silver + cost_of_potion_kits_in_silver

theorem harry_total_cost_in_silver : cost_in_silver = 537 := by
  sorry

end harry_total_cost_in_silver_l191_191381


namespace ellipse_k_values_l191_191780

theorem ellipse_k_values (k : ℝ) :
  (∃ a b : ℝ, a = (k + 8) ∧ b = 9 ∧ 
  (b > a → (a * (1 - (1 / 2) ^ 2) = b - a) ∧ k = 4) ∧ 
  (a > b → (b * (1 - (1 / 2) ^ 2) = a - b) ∧ k = -5/4)) :=
sorry

end ellipse_k_values_l191_191780


namespace problem_1_problem_2_l191_191372

def f (x : ℝ) : ℝ := 
  sqrt 3 * sin x^2 + cos (π / 4 - x)^2 - (1 + sqrt 3) / 2

theorem problem_1 : 
  ∃ x ∈ Icc 0 (π / 2), f x = 1 :=
sorry

theorem problem_2 (A B : ℝ) (hA_lt_B : A < B) 
  (hA_in : A ∈ Icc 0 (π / 2)) (hB_in: B ∈ Icc 0 (π / 2)) :
  f A = 1 / 2 ∧ f B = 1 / 2 → 
  ∃ (C : ℝ), (C = π - A - B) ∧ (sin A / sin C = sqrt 2) :=
sorry

end problem_1_problem_2_l191_191372


namespace center_of_circle_l191_191042

theorem center_of_circle (ρ θ : ℝ) (h : ρ = 2 * Real.cos (θ - π / 4)) : (ρ, θ) = (1, π / 4) :=
sorry

end center_of_circle_l191_191042


namespace geometric_sequence_problem_l191_191778

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
variable (a1 a2 a3 a4 a5 : ℝ)

-- Assume the sequence is geometric
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The sum of the first n terms of the sequence
def sum_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = a 0 * ((1 - q^(n + 1)) / (1 - q))

theorem geometric_sequence_problem (h1 : 8 * a 1 + a 4 = 0)
  (h2 : geometric_sequence a q)
  (h3 : sum_of_sequence a S) :
  S 2 / a 2 = 3 / 4 :=
sorry

end geometric_sequence_problem_l191_191778


namespace positive_difference_of_prime_factors_l191_191179

theorem positive_difference_of_prime_factors :
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p * q = 172081 ∧ abs (p - q) = 13224 :=
by {
  let p := 13,
  let q := 13237,
  have p_prime : nat.prime p := by norm_num,
  have q_prime : nat.prime q := by norm_num,
  have factorization : p * q = 172081 := by norm_num,
  have difference : abs (p - q) = 13224 := by norm_num,
  exact ⟨p, q, p_prime, q_prime, factorization, difference⟩,
}

end positive_difference_of_prime_factors_l191_191179


namespace sum_q_p_values_l191_191669

noncomputable def p (x : ℤ) : ℤ := Int.natAbs x + 1
noncomputable def q (x : ℤ) : ℤ := -x^2

theorem sum_q_p_values :
  ( ∑ x in [-3, -2, -1, 0, 1, 2, 3].toFinset, q (p x) ) = -59 :=
by
  sorry

end sum_q_p_values_l191_191669


namespace table_height_l191_191544

theorem table_height
  (l d h : ℤ)
  (h_eq1 : l + h - d = 36)
  (h_eq2 : 2 * l + h = 46)
  (l_eq_d : l = d) :
  h = 36 :=
by
  sorry

end table_height_l191_191544


namespace exists_equal_cost_route_l191_191022

theorem exists_equal_cost_route (
    num_cities : ℕ,
    num_airlines : ℕ,
    flights : list (ℕ × ℕ × ℕ),
    flight_cost : ℕ → ℚ,
    no_cheaper_layover : ∀ x y z : ℕ, ∃ k1 k2 : ℕ, (x, y, k1) ∈ flights ∧ (y, z, k2) ∈ flights → flight_cost (k1 * k2) ≥ flight_cost k2
) :
  ∃ (A B C : ℕ), (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C) ∧ 
  ∃ k1 k2 : ℕ, (A, C, k1) ∈ flights ∧ (C, B, k2) ∈ flights ∧ flight_cost k1 = flight_cost k2 :=
by {
  sorry
}

end exists_equal_cost_route_l191_191022


namespace compute_sum_eq_one_fourth_l191_191278

theorem compute_sum_eq_one_fourth :
  (∑ n in (finset.range (3 + 1)).filter (λ n, n ≥ 3), ∑ k in (finset.range (n-2 + 1)).filter (λ k, k ≤ n-2), (k+1: ℝ) / 3^(n+k)) = (1: ℝ) / 4 :=
sorry

end compute_sum_eq_one_fourth_l191_191278


namespace fraction_zero_solution_l191_191127

theorem fraction_zero_solution (x : ℝ) (h1 : x - 5 = 0) (h2 : 4 * x^2 - 1 ≠ 0) : x = 5 :=
by {
  sorry -- The proof
}

end fraction_zero_solution_l191_191127


namespace solve_fraction_problem_l191_191558

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l191_191558


namespace population_increase_rate_l191_191400

theorem population_increase_rate (persons : ℕ) (minutes : ℕ) (seconds_per_minute : ℕ) (total_seconds : ℕ) (rate : ℕ)
  (h1 : persons = 220)
  (h2 : minutes = 55)
  (h3 : seconds_per_minute = 60)
  (h4 : total_seconds = minutes * seconds_per_minute)
  (h5 : total_seconds = 3300)
  (h6 : rate = total_seconds / persons) :
  rate = 15 :=
by
  rw [h1, h2, h3, h4] at h5
  exact (nat.div_eq_of_eq_mul_right (by norm_num) h5).symm

end population_increase_rate_l191_191400


namespace find_scalar_k_l191_191957

variables (R : Type*) [Field R]
variables (u v w : R^3)

theorem find_scalar_k (k : R) 
  (h₀ : u + v + w = 0)
  (h₁ : ∀ u v w : R^3, k * (v × u) + 2 * (v × w) + (w × u) = 0) :
  k = 3 :=
by
  sorry

end find_scalar_k_l191_191957


namespace number_of_teams_in_league_l191_191921

theorem number_of_teams_in_league : 
  ∃ n : ℕ, (∑ i in finset.range n, i) = 36 ∧ n = 9 :=
begin
  sorry,
end

end number_of_teams_in_league_l191_191921


namespace agent_takes_19_percent_l191_191491

def agentPercentage (copies_sold : ℕ) (advance_copies : ℕ) (price_per_copy : ℕ) (steve_earnings : ℕ) : ℕ :=
  let total_earnings := copies_sold * price_per_copy
  let agent_earnings := total_earnings - steve_earnings
  let percentage_agent := 100 * agent_earnings / total_earnings
  percentage_agent

theorem agent_takes_19_percent :
  agentPercentage 1000000 100000 2 1620000 = 19 :=
by 
  sorry

end agent_takes_19_percent_l191_191491


namespace clairaut_equation_solution_l191_191045

open Real

noncomputable def clairaut_solution (f : ℝ → ℝ) (C : ℝ) : Prop :=
  (∀ x, f x = C * x + 1/(2 * C)) ∨ (∀ x, (f x)^2 = 2 * x)

theorem clairaut_equation_solution (y : ℝ → ℝ) :
  (∀ x, y x = x * (deriv y x) + 1/(2 * (deriv y x))) →
  ∃ C, clairaut_solution y C :=
sorry

end clairaut_equation_solution_l191_191045


namespace eleven_pow_2048_mod_17_l191_191181

theorem eleven_pow_2048_mod_17 : 11^2048 % 17 = 1 := by
  sorry

end eleven_pow_2048_mod_17_l191_191181


namespace statement2_statement3_l191_191727

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Conditions for the statements
axiom cond1 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c p = q ∧ f a b c q = p
axiom cond2 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c p = f a b c q
axiom cond3 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c (p + q) = c

-- Statement 2 correctness
theorem statement2 (a b c p q : ℝ) (hpq : p ≠ q) (h : f a b c p = f a b c q) : 
  f a b c (p + q) = c :=
sorry

-- Statement 3 correctness
theorem statement3 (a b c p q : ℝ) (hpq : p ≠ q) (h : f a b c (p + q) = c) : 
  p + q = 0 ∨ f a b c p = f a b c q :=
sorry

end statement2_statement3_l191_191727


namespace proof_problem_l191_191908

noncomputable def polar_to_rectangular : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x - sqrt 2) ^ 2 + y ^ 2 = 2) ↔ (∀ θ, rho = 2 * sqrt 2 * Real.cos θ → (x ^ 2 + y ^ 2 = rho ^ 2) ∧ (x = rho * Real.cos θ)) sorry

noncomputable def check_common_points : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x = 3 - sqrt 2 + 2 * Real.cos θ) ∧ (y = 2 * Real.sin θ)) ∧
              ∀ (M_P C_P : ℝ), C_P = 2 ∧ |(3 - sqrt 2) - sqrt 2| = 3 - 2 * sqrt 2 < 2 - sqrt 2 :=
              ¬ ∃ p, (p ∈ C) ∧ (p ∈ C_1) sorry

theorem proof_problem : polar_to_rectangular ∧ check_common_points := by
  sorry

end proof_problem_l191_191908


namespace number_of_boys_l191_191408

-- Define the conditions as seen in part a)
def number_of_girls : ℕ := 635
def additional_boys : ℕ := 510

-- Statement to prove the number of boys given the conditions
theorem number_of_boys : ∃ (b : ℕ), b = number_of_girls + additional_boys :=
by
  use 1145
  rw [number_of_girls, additional_boys]
  norm_num
  sorry

end number_of_boys_l191_191408


namespace proof_problem_l191_191873

section Exercise

variable (θ : ℝ)

def polarCoordEq (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos θ

def toRectCoord (ρ θ : ℝ) : Prop := ρ^2 = (2 * Real.sqrt 2 * Real.cos θ) * ρ

def circleEq (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 2

def parametricLocusC1 (θ : ℝ) : (ℝ × ℝ) := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

theorem proof_problem :
  (∃ ρ θ, polarCoordEq ρ θ → circleEq (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  ∀ θ, ¬ ∃ x y, (circleEq x y) ∧ (parametricLocusC1 θ = (x, y)) :=
by 
  sorry

end Exercise

end proof_problem_l191_191873


namespace franks_age_l191_191730

variable (F G : ℕ)

def gabriel_younger_than_frank : Prop := G = F - 3
def total_age_is_seventeen : Prop := F + G = 17

theorem franks_age (h1 : gabriel_younger_than_frank F G) (h2 : total_age_is_seventeen F G) : F = 10 :=
by
  sorry

end franks_age_l191_191730


namespace gcd_198_286_l191_191175

theorem gcd_198_286 : Nat.gcd 198 286 = 22 :=
by
  sorry

end gcd_198_286_l191_191175


namespace integer_solutions_of_3_pow_a_minus_5_pow_b_eq_2_l191_191698

theorem integer_solutions_of_3_pow_a_minus_5_pow_b_eq_2 :
  ∀ (a b : ℤ), 3 ^ a - 5 ^ b = 2 ↔ (a = 1 ∧ b = 0) ∨ (a = 3 ∧ b = 2) :=
by
  sorry

end integer_solutions_of_3_pow_a_minus_5_pow_b_eq_2_l191_191698


namespace problem_solution_l191_191413

open Real

def area_of_convex_quadrilateral (AB BC CD DA : ℝ) (angle_CDA : ℝ) : ℝ := 
  let area_ACD := 1 / 2 * CD * DA
  let AC := sqrt (AD^2 + DC^2 - 2 * AD * DC * cos angle_CDA)
  let s := (AB + BC + AC) / 2
  let area_ABC := sqrt (s * (s - AB) * (s - BC) * (s - AC))
  area_ACD + area_ABC

theorem problem_solution 
  (AB BC CD DA : ℝ) (H_AB : AB = 6) (H_BC : BC = 3) (H_CD : CD = 10) (H_DA : DA = 10) 
  (angle_CDA : ℝ) (H_angle_CDA : angle_CDA = π / 2)
  (a b c : ℕ) (H_area : area_of_convex_quadrilateral AB BC CD DA angle_CDA = 50 + 30 * sqrt 2)
  (H_repr : sqrt a + b * sqrt c = 50 + 30 * sqrt 2) :
  a + b + c = 82 := 
sorry

end problem_solution_l191_191413


namespace x_zero_sufficient_not_necessary_for_sin_zero_l191_191594

theorem x_zero_sufficient_not_necessary_for_sin_zero :
  (∀ x : ℝ, x = 0 → Real.sin x = 0) ∧ (∃ y : ℝ, Real.sin y = 0 ∧ y ≠ 0) :=
by
  sorry

end x_zero_sufficient_not_necessary_for_sin_zero_l191_191594


namespace square_of_TS_length_l191_191025

-- Definition of given parameters
variables (r1 r2 d : ℝ) -- radii and distance variables
variables (r1_eq r2_eq d_eq : ℝ) (r1_pos r2_pos d_pos: r1 > 0) (r2 > 0) (d > 0)

-- Distance and radii conditions
def circle_radii_and_distance_conditions := (r1 = 7) ∧ (r2 = 9) ∧ (d = 14)

-- Equal chords TS and SU
variables (TS SU : ℝ) (TS_eq_SU : TS = SU)

-- Prove square of length of TS
theorem square_of_TS_length : 
  circle_radii_and_distance_conditions ∧ TS_eq_SU → TS^2 = 130 :=
by sorry

end square_of_TS_length_l191_191025


namespace david_marks_physics_l191_191287

def marks_english := 96
def marks_math := 95
def marks_chemistry := 97
def marks_biology := 95
def average_marks := 93
def number_of_subjects := 5

theorem david_marks_physics : 
  let total_marks := average_marks * number_of_subjects 
  let total_known_marks := marks_english + marks_math + marks_chemistry + marks_biology
  let marks_physics := total_marks - total_known_marks
  marks_physics = 82 :=
by
  sorry

end david_marks_physics_l191_191287


namespace planes_intersection_parallel_to_line_l191_191769

structure Line (P : Type) [AffineSpace P] :=
(perp : ∀ (α : Set P), ¬ α ∩ (Set.univ : Set P))

structure Plane (P : Type) [AffineSpace P] :=
(perp_to_line : ∀ (m : Line P), m.perp)

variables (P : Type) [h : AffineSpace P]

def are_skew_lines (m n : Line P) : Prop :=
¬ (∃ p : P, p ∈ m ∧ p ∈ n) ∧ ¬ (∃ (v₁ : Submodule P), v₁ ⊆ m ∧ v₁ ⊆ n)

def is_perp_to_plane {P : Type} [AffineSpace P] (m : Line P) (α : Plane P) : Prop :=
α.perp_to_line m

def plane_intersects (α β : Plane P) : Prop :=
∃ p : P, p ∈ α ∧ p ∈ β

def intersection_line_parallel {P : Type} [AffineSpace P] (α β : Plane P) (l : Line P) : Prop :=
∀ p ∈ (α ∩ β), ∃ direction : Submodule P, p + direction ∈ l

theorem planes_intersection_parallel_to_line 
  (m n l : Line P) 
  (α β : Plane P) 
  (h_skew : are_skew_lines m n)
  (h_perpm_alpha : is_perp_to_plane m α)
  (h_perpn_beta : is_perp_to_plane n β)
  (h_perpl_m : l.perp m)
  (h_perpl_n : l.perp n)
  (h_not_in_alpha : ∀ p ∈ l, ¬ p ∈ α)
  (h_not_in_beta : ∀ p ∈ l, ¬ p ∈ β) : 
  plane_intersects α β ∧ intersection_line_parallel α β l := 
sorry

end planes_intersection_parallel_to_line_l191_191769


namespace triangle_angle_contradiction_l191_191186

theorem triangle_angle_contradiction (α β γ : ℝ) (h_sum : α + β + γ = 180) :
  (α > 60 ∧ β > 60 ∧ γ > 60) → false := 
by 
suffices h : α + β + γ > 180,
  from (h_sum.trans h).elim,
suffices hα : α > 0, from add_lt_add_of_lt_add_of_le hα,
suffices hβ : β > 0, from add_lt_add_of_lt_add_of_le hβ,
suffices hγ : γ > 0, from h_le_succ_of_le _,
repeat { linarith, }
theta := sorry  -- This suffices part is intentional for the placeholder logic

end triangle_angle_contradiction_l191_191186


namespace inequality_proof_l191_191438

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_condition : (a + b + c + d) * (1/a + 1/b + 1/c + 1/d) = 20) :
  (a^2 + b^2 + c^2 + d^2) * (1/(a^2) + 1/(b^2) + 1/(c^2) + 1/(d^2)) ≥ 36 :=
by
  sorry

end inequality_proof_l191_191438


namespace production_average_lemma_l191_191588

theorem production_average_lemma (n : ℕ) (h1 : 50 * n + 60 = 55 * (n + 1)) : n = 1 :=
by
  sorry

end production_average_lemma_l191_191588


namespace cookies_left_over_l191_191641

def abigail_cookies : Nat := 53
def beatrice_cookies : Nat := 65
def carson_cookies : Nat := 26
def pack_size : Nat := 10

theorem cookies_left_over : (abigail_cookies + beatrice_cookies + carson_cookies) % pack_size = 4 := 
by
  sorry

end cookies_left_over_l191_191641


namespace probability_more_than_6_grandchildren_l191_191460

theorem probability_more_than_6_grandchildren 
    (n : ℕ) (p : ℚ) (h_n : n = 12) (h_p : p = 1/2) :
  let P := (1 : ℚ) - (Nat.choose 12 6) / (2^12) in
  P = 793 / 1024 :=
by
  sorry

end probability_more_than_6_grandchildren_l191_191460


namespace number_of_pairs_l191_191289

theorem number_of_pairs :
  (∃ (pairs : Finset (ℕ × ℕ)), (∀ (pair : ℕ × ℕ), pair ∈ pairs → 1 ≤ pair.1 ∧ pair.1 ≤ 30 ∧ 3 ≤ pair.2 ∧ pair.2 ≤ 30 ∧ (pair.1 % pair.2 = 0) ∧ (pair.1 % (pair.2 - 2) = 0)) ∧ pairs.card = 22) := by
  sorry

end number_of_pairs_l191_191289


namespace add_to_fraction_l191_191563

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l191_191563


namespace calculate_fraction_l191_191270

theorem calculate_fraction :
  (18 - 6) / ((3 + 3) * 2) = 1 := by
  sorry

end calculate_fraction_l191_191270


namespace exist_k_plus_one_balls_l191_191483

variables (n k : ℕ) (balls : Type) [fintype balls] (a b : balls → ℕ)
  (h : ∀ x, a x ≥ 1 ∧ b x ≥ 1)
  (H1 : ∑ x, a x = n)
  (H2 : ∑ x, b x = n + k)

theorem exist_k_plus_one_balls (n k : ℕ) (balls : Type) [fintype balls]
  (a b : balls → ℕ) (h : ∀ x, a x ≥ 1 ∧ b x ≥ 1) 
  (H1 : ∑ x, a x = n) (H2 : ∑ x, b x = n + k) :
  ∃ S : finset balls, S.card = k + 1 ∧ ∀ x ∈ S, a x > b x :=
by sorry

end exist_k_plus_one_balls_l191_191483


namespace count_4_digit_divisible_by_45_l191_191222

theorem count_4_digit_divisible_by_45 : 
  ∃ n, n = 11 ∧ (∀ a b : ℕ, a + b = 2 ∨ a + b = 11 → (20 + b * 10 + 5) % 45 = 0) :=
sorry

end count_4_digit_divisible_by_45_l191_191222


namespace sum_of_largest_and_smallest_prime_factors_of_1560_l191_191552

theorem sum_of_largest_and_smallest_prime_factors_of_1560 : 
  ∃ (factors : List ℕ), factors = [2, 3, 5, 13] ∧ List.sum [List.minimum factors, List.maximum factors] = 15 :=
by
  sorry

end sum_of_largest_and_smallest_prime_factors_of_1560_l191_191552


namespace mabel_tomatoes_l191_191974

theorem mabel_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 := by
  sorry

end mabel_tomatoes_l191_191974


namespace find_n_l191_191699

theorem find_n : ∃ (n : ℕ), n = 2^(2*6-1) - 5*6 - 3 ∧ n = (2^(6-1) - 1)*(2^6 + 1) ∧ n = 2015 := 
by 
  -- We assume we have an n
  use 2015,
  -- Prove that 2015 equals to both forms of the equation from conditions
  have h1 : 2015 = 2^(2*6-1) - 5*6 - 3,
  { sorry },
  -- Assuming h1, prove that 2015 equals the second form given in the conditions
  have h2 : 2015 = (2^(6-1) - 1)*(2^6 + 1),
  { sorry },
  -- Combine h1 and h2 to meet the required conditions
  exact ⟨ h1, h2, rfl ⟩

end find_n_l191_191699


namespace internet_bill_proof_l191_191089

variable (current_bill : ℕ)
variable (internet_bill_30Mbps : ℕ)
variable (annual_savings : ℕ)
variable (additional_amount_20Mbps : ℕ)

theorem internet_bill_proof
  (h1 : current_bill = 20)
  (h2 : internet_bill_30Mbps = 40)
  (h3 : annual_savings = 120)
  (monthly_savings : ℕ := annual_savings / 12)
  (h4 : monthly_savings = 10)
  (h5 : internet_bill_30Mbps - (current_bill + additional_amount_20Mbps) = 10) :
  additional_amount_20Mbps = 10 :=
by
  sorry

end internet_bill_proof_l191_191089


namespace max_value_sqrt_sum_l191_191061

theorem max_value_sqrt_sum (a b c : ℝ) (h_cond : a + b + c = 8) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) : 
  (sqrt (3 * a + 2) + sqrt (3 * b + 2) + sqrt (3 * c + 2)) ≤ 3 * sqrt 10 := 
sorry

end max_value_sqrt_sum_l191_191061


namespace polarToRectangular_noCommonPoints_l191_191885

section PolarToRectangular
variables {θ : ℝ} {ρ : ℝ}

-- Condition: polar coordinate equation ρ = 2√2 cos θ
def polarEq (θ : ℝ) : ℝ := 2 * sqrt 2 * cos θ

-- Correct answer: rectangular coordinate equation (x - √2)² + y² = 2
theorem polarToRectangular 
  (h : ρ = polarEq θ)
  (x y : ℝ)
  (hx : x = ρ * cos θ)
  (hy : y = ρ * sin θ) :
  (x - sqrt 2) ^ 2 + y ^ 2 = 2 :=
sorry

end PolarToRectangular

section LocusAndCommonPoints
variables {x y x₁ y₁ : ℝ} {C C₁ : set (ℝ × ℝ)}

-- Condition: (1, 0) is point A, M is a moving point on circle C, P satisfies AP = √2AM
def pointA := (1, 0 : ℝ) 
def isOnCircleC (M : ℝ × ℝ) : Prop := (M.1 - sqrt 2) ^ 2 + M.2 ^ 2 = 2
def scaledVector (A P M : ℝ × ℝ) : Prop :=
  P.1 - A.1 = sqrt 2 * (M.1 - A.1) ∧ P.2 - A.2 = sqrt 2 * (M.2 - A.2)

-- Correct answer: Circles C and C₁ do not have any common points
theorem noCommonPoints
  (A : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : isOnCircleC M)
  (P : ℝ × ℝ)
  (hP : scaledVector A P M) :
  ∀ P : ℝ × ℝ, ¬ (isOnCircleC P ∧ (P.1 - (3 - sqrt 2)) ^ 2 + P.2 ^ 2 = 4) :=
sorry

end LocusAndCommonPoints

end polarToRectangular_noCommonPoints_l191_191885


namespace max_value_sqrt_sum_l191_191717

theorem max_value_sqrt_sum (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) : 
  ∃ M, (∀ y, -36 ≤ y ∧ y ≤ 36 → sqrt (36 + y) + sqrt (36 - y) ≤ M) ∧ M = 12 :=
by
  sorry

end max_value_sqrt_sum_l191_191717


namespace fraction_identity_l191_191198

theorem fraction_identity (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 := 
by 
  sorry

end fraction_identity_l191_191198


namespace white_ducks_count_l191_191318

theorem white_ducks_count (W : ℕ) : 
  (5 * W + 10 * 7 + 12 * 6 = 157) → W = 3 :=
by
  sorry

end white_ducks_count_l191_191318


namespace three_person_subcommittees_from_eight_l191_191798

theorem three_person_subcommittees_from_eight :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end three_person_subcommittees_from_eight_l191_191798


namespace snail_distance_at_44th_day_l191_191633

theorem snail_distance_at_44th_day :
  (∑ n in Finset.range 44, 1 / (n + 1) - 1 / (n + 2)) = 44 / 45 := by
  sorry

end snail_distance_at_44th_day_l191_191633


namespace C_and_C1_no_common_points_l191_191842

noncomputable def C_equation : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - real.sqrt 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (1, 0)

noncomputable def M_on_C (θ : ℝ) : ℝ × ℝ :=
  let x := 2 * real.sqrt 2 * real.cos θ * real.cos θ in
  let y := 2 * real.sqrt 2 * real.sin θ * real.sin θ in
  (x, y)

noncomputable def P_on_locus (x_M y_M : ℝ) : ℝ × ℝ :=
  let x := real.sqrt 2 * (x_M - point_A.1) + point_A.1 in
  let y := real.sqrt 2 * y_M in
  (x, y)

noncomputable def Locus_P_equation (θ : ℝ) : ℝ × ℝ :=
  let x := 3 - real.sqrt 2 + 2 * real.cos θ in
  let y := 2 * real.sin θ in
  (x, y)

def has_common_points (C1 C2 : set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C1 ∧ p ∈ C2

theorem C_and_C1_no_common_points :
  ¬ has_common_points
    {p | C_equation p}
    {p | ∃ θ, P_on_locus (M_on_C θ).1 (M_on_C θ).2 = p} :=
sorry

end C_and_C1_no_common_points_l191_191842


namespace cos_sum_angles_l191_191476

noncomputable def angle_APD : ℝ := 60 * Real.pi / 180
noncomputable def angle_BPE : ℝ := 75.52 * Real.pi / 180
noncomputable def cos_BPC_add_CPD : ℝ := -Real.sqrt 2 / 2

theorem cos_sum_angles 
  (h1 : ∀ (A B C D E : ℝ) (AB BC CD DE : ℝ), AB = BC ∧ BC = CD ∧ CD = DE)
  (h2 : ∀ (P : ℝ), Real.cos(60 * Real.pi / 180) = 1 / 2 ∧ Real.cos(75.52 * Real.pi / 180) = 1 / 4) :
  Real.cos (angle_APD + angle_BPE) = cos_BPC_add_CPD :=
  sorry

end cos_sum_angles_l191_191476


namespace time_to_fill_tank_l191_191203

-- Define the rates of the pipes
def rate_first_fill : ℚ := 1 / 15
def rate_second_fill : ℚ := 1 / 15
def rate_outlet_empty : ℚ := -1 / 45

-- Define the combined rate
def combined_rate : ℚ := rate_first_fill + rate_second_fill + rate_outlet_empty

-- Define the time to fill the tank
def fill_time (rate : ℚ) : ℚ := 1 / rate

theorem time_to_fill_tank : fill_time combined_rate = 9 := 
by 
  -- Proof omitted
  sorry

end time_to_fill_tank_l191_191203


namespace polynomial_sum_zero_l191_191351

-- Define the given conditions
def polynomial (a b : ℝ) (c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Assume the given polynomials f_1 to f_100
def f (a b : ℝ) (c : Fin 100 → ℝ) (i : Fin 100) : ℝ → ℝ := polynomial a b (c i)

-- Assume each polynomial form and the specific root conditions
variables (a b : ℝ) (c : Fin 100 → ℝ) (roots : Fin 100 → ℝ)

-- Assume roots condition for each polynomial
axiom roots_def : ∀ i : Fin 100, f a b c i (roots i) = 0

-- The main theorem statement to prove the sum
theorem polynomial_sum_zero :
  let sum := (Finset.range 100).sum (λ i => f a b c (Fin.ofNat' ((i + 1) % 100)) (roots i))
  in sum = 0 :=
sorry

end polynomial_sum_zero_l191_191351


namespace decreasing_on_interval_l191_191448

noncomputable def f (x b : ℝ) : ℝ := x^2 - 12 * x + b

theorem decreasing_on_interval (b : ℝ) :
  ∀ x1 x2 : ℝ, x1 ∈ Set.Ioo (-∞) (-1) → x2 ∈ Set.Ioo (-∞) (-1) → x1 < x2 → f x1 b > f x2 b :=
by
  sorry

end decreasing_on_interval_l191_191448


namespace sum_alternating_powers_of_neg1_l191_191709

theorem sum_alternating_powers_of_neg1 :
  let a := λ n : ℤ, (-1) ^ n in
  (∑ n in finset.Icc (-15) (15), a n) = -1 :=
by
  let a := λ n : ℤ, (-1) ^ n
  have h : ∀ n ∈ finset.Icc (-15) (15), a n = (-1) ^ n := by
    intro n hn
    rfl
  sorry

end sum_alternating_powers_of_neg1_l191_191709


namespace volume_of_pyramid_l191_191369

theorem volume_of_pyramid (S A B C : Point) (d : ℝ) (SC : d = 6) (A_on_sphere : dist S A = 3)
  (B_on_sphere : dist S B = 3) (AB_equal : dist A B = 3) :
  volume (tetrahedron S A B C) = (9 * Real.sqrt 2) / 2 :=
sorry

end volume_of_pyramid_l191_191369


namespace product_of_real_solutions_l191_191708

theorem product_of_real_solutions :
  (∏ s in {s : ℝ | ∀ x : ℝ, x ≠ 0 → (3 / (5 * x) = (s - 2 * x) / 4)}.to_finset) = -19.2 :=
by sorry

end product_of_real_solutions_l191_191708


namespace triangle_side_length_l191_191755

theorem triangle_side_length 
  (a b c : ℝ) 
  (cosA : ℝ) 
  (h1: a = Real.sqrt 5) 
  (h2: c = 2) 
  (h3: cosA = 2 / 3) 
  (h4: a^2 = b^2 + c^2 - 2 * b * c * cosA) : 
  b = 3 := 
by 
  sorry

end triangle_side_length_l191_191755


namespace equilateral_triangle_side_length_l191_191410

-- Define the equilateral triangle and given conditions
variables {A B C F : Type*} [linear_ordered_comm_ring A]
variables (AB AC BC BF CF : A)

-- Given conditions translated into equations
-- Assume specific values for easier handling, if necessary (though not always a must)
axiom h1 : BC = AB   -- condition for the equilateral triangle sides being equal
axiom h2 : BF + CF = BC  -- Point F lying on BC
axiom h3 : BF = 3 * CF  -- Area of ΔABF is 3 times area of ΔACF
axiom h4 : BF - CF = 5  -- Difference in their perimeters is 5 cm

-- Prove that the side length of the equilateral triangle is 10 cm
theorem equilateral_triangle_side_length : BC = 10 := 
by {
  sorry
}

end equilateral_triangle_side_length_l191_191410


namespace mass_after_5730_years_liangzhu_period_l191_191019

-- Conditions
variables (N0 : ℝ) (t : ℝ)
def decay_law (N0 : ℝ) (t : ℝ) := N0 * 2^(-t / 5730)

-- Prove the mass after 5730 years is half of the original mass
theorem mass_after_5730_years (N0 : ℝ) : decay_law N0 5730 = N0 / 2 := 
by sorry

-- Given mass fraction condition and the time range
def mass_in_range (N0 : ℝ) (t : ℝ) := 
  (3 / 7 : ℝ) ≤ decay_law N0 t ∧ decay_law N0 t ≤ 1 / 2

-- Prove the period is between 5730 and 6876 years ago
theorem liangzhu_period (N0 : ℝ) : 
  (3 / 7 : ℝ) ≤ decay_law N0 t ∧ decay_law N0 t ≤ 1 / 2 → 5730 ≤ t ∧ t ≤ 6876 := 
by sorry

end mass_after_5730_years_liangzhu_period_l191_191019


namespace C_and_C1_no_common_points_l191_191915

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  let x := rho * cos theta
  let y := rho * sin theta
  (x, y)

def curve_C_eq (theta : ℝ) : ℝ :=
  2 * sqrt 2 * cos theta

def A := (1, 0 : ℝ)

-- Predicate indicating point M is on curve C given in rectangular coordinates.
def on_curve_C (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  (x - sqrt 2) ^ 2 + y ^ 2 = 2

-- Definition for point P satisfying the given vector equation.
def point_P (A M : ℝ × ℝ) : ℝ × ℝ :=
  let x_A := A.1
  let y_A := A.2
  let x_M := M.1
  let y_M := M.2
  (√2 * (x_M - x_A) + x_A, √2 * (y_M - y_A) + y_A)

-- Prove that the rectangular coordinate equation is as specified and parametrized locus C_1 exists and has no common points with C.
theorem C_and_C1_no_common_points (M P : ℝ × ℝ) (theta : ℝ)
  (h_M_on_C : on_curve_C M)
  (h_P_eq : P = point_P A M) :
  let C1_center := (3 - sqrt 2, 0)
  let C1_radius := 2
  let C_center   := (sqrt 2, 0)
  let C_radius   := sqrt 2
  let C1_eq := (x - (3 - sqrt 2))^2 + y^2 = 4
  (M, P) ∉ C ∩ C1 :=
sorry

end C_and_C1_no_common_points_l191_191915


namespace damage_in_gbp_is_correct_l191_191224

-- Definitions based on conditions
def damage_AUD : ℕ := 45000000
def initial_exchange_rate : ℚ := 1.8
def final_exchange_rate : ℚ := 1.7

-- Average exchange rate calculation
def average_exchange_rate : ℚ := (initial_exchange_rate + final_exchange_rate) / 2

-- Calculation of damage in British pounds
def damage_GBP : ℚ := damage_AUD / average_exchange_rate

-- Proof goal
theorem damage_in_gbp_is_correct :
  round (damage_AUD / average_exchange_rate) = 25714286 :=
by
  have h1 : average_exchange_rate = 1.75 := by norm_num
  have h2 : damage_AUD / average_exchange_rate = 25714285.714285714 := by norm_num
  rw h2
  norm_num
  sorry

end damage_in_gbp_is_correct_l191_191224


namespace train_crosses_signal_pole_in_18_seconds_l191_191602

-- Define the given conditions
def train_length := 300  -- meters
def platform_length := 450  -- meters
def time_to_cross_platform := 45  -- seconds

-- Define the question and the correct answer
def time_to_cross_signal_pole := 18  -- seconds (this is what we need to prove)

-- Define the total distance the train covers when crossing the platform
def total_distance_crossing_platform := train_length + platform_length  -- meters

-- Define the speed of the train
def train_speed := total_distance_crossing_platform / time_to_cross_platform  -- meters per second

theorem train_crosses_signal_pole_in_18_seconds :
  300 / train_speed = time_to_cross_signal_pole :=
by
  -- train_speed is defined directly in terms of the given conditions
  unfold train_speed total_distance_crossing_platform train_length platform_length time_to_cross_platform
  sorry

end train_crosses_signal_pole_in_18_seconds_l191_191602


namespace multiple_of_eight_l191_191016

theorem multiple_of_eight (x y : ℤ) (h : ∀ (k : ℤ), 24 + 16 * k = 8) : ∃ (k : ℤ), x + 16 * y = 8 * k := 
by
  sorry

end multiple_of_eight_l191_191016


namespace sequence_formula_l191_191624

theorem sequence_formula (n : ℕ) : 
  (λ a_n, a_n = 1/(n+1)) = (λ a_n, a_n = (1 / n)) :=
sorry

end sequence_formula_l191_191624


namespace remaining_time_for_each_l191_191271

-- Define the conditions
def movie_duration : Int := 120 -- movie is 2 hours long in minutes
def camila_start_before_maverick : Int := 30 -- Camila starts 30 minutes before Maverick
def maverick_start_before_daniella : Int := 45 -- Maverick starts 45 minutes before Daniella
def daniella_remaining_time : Int := 30 -- Daniella has 30 minutes left to watch

-- Define the theorem to prove
theorem remaining_time_for_each :
  (remaining_time (daniella_remaining_time, maverick_start_before_daniella, camila_start_before_maverick)) =
  (0, 0, 30) :=
sorry

-- Define the auxiliary function to compute the remaining time for each
def remaining_time (daniella_remaining_time maverick_start_before_daniella camila_start_before_maverick : Int) :
    Int × Int × Int :=
  let maverick_remaining_time := 0 -- Since Maverick finishes when Daniella has 30 minutes left
  let camila_remaining_time := 0 -- Since Camila finishes when Maverick finishes
  (camila_remaining_time, maverick_remaining_time, daniella_remaining_time)

end remaining_time_for_each_l191_191271


namespace initial_pennies_l191_191611

theorem initial_pennies (P : ℕ)
  (h1 : P - (P / 2 + 1) = P / 2 - 1)
  (h2 : (P / 2 - 1) - (P / 4 + 1 / 2) = P / 4 - 3 / 2)
  (h3 : (P / 4 - 3 / 2) - (P / 8 + 3 / 4) = P / 8 - 9 / 4)
  (h4 : P / 8 - 9 / 4 = 1)
  : P = 26 := 
by
  sorry

end initial_pennies_l191_191611


namespace no_equilateral_triangle_on_integer_lattice_l191_191677

theorem no_equilateral_triangle_on_integer_lattice :
  ∀ (A B C : ℤ × ℤ), 
  A ≠ B → B ≠ C → C ≠ A →
  (dist A B = dist B C ∧ dist B C = dist C A) → 
  false :=
by sorry

end no_equilateral_triangle_on_integer_lattice_l191_191677


namespace S9_is_36_l191_191059

noncomputable def quadratic_min : ℝ :=
  real.inf {x | x^2 - 2 * x + 3}

noncomputable def a_sequence (n : ℕ) : ℝ :=
  if n = 1 then quadratic_min
  else a_sequence (n - 1) + 1/2

noncomputable def S (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (a_sequence 1 + a_sequence n)

theorem S9_is_36:
  S 9 = 36 :=
sorry

end S9_is_36_l191_191059


namespace convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191882

theorem convert_polar_to_rectangular (ρ θ x y : ℝ) (hρ : ρ = 2 * √2 * cos θ) :
  (x - √2) ^ 2 + y ^ 2 = 2 :=
sorry

theorem parametric_equations_and_no_common_points (x y θ : ℝ) :
  let A := (1 : ℝ, 0 : ℝ)
  let P (θ : ℝ) := (3 - √2 + 2 * cos θ, 2 * sin θ)
  let C1 := (P 0).1 ^ 2 + (P 0).2 ^ 2 = 4
  (x - √2) ^ 2 + y ^ 2 = 2 ∧ (C1 = [((x, y) = P θ)]) → False :=
sorry

end convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191882


namespace question1_question2_l191_191529

-- Define the conditions
def numTraditionalChinesePaintings : Nat := 5
def numOilPaintings : Nat := 2
def numWatercolorPaintings : Nat := 7

-- Define the number of ways to choose one painting from each category
def numWaysToChooseOnePaintingFromEachCategory : Nat :=
  numTraditionalChinesePaintings * numOilPaintings * numWatercolorPaintings

-- Define the number of ways to choose two paintings of different types
def numWaysToChooseTwoPaintingsOfDifferentTypes : Nat :=
  (numTraditionalChinesePaintings * numOilPaintings) +
  (numTraditionalChinesePaintings * numWatercolorPaintings) +
  (numOilPaintings * numWatercolorPaintings)

-- Theorems to prove the required results
theorem question1 : numWaysToChooseOnePaintingFromEachCategory = 70 := by
  sorry

theorem question2 : numWaysToChooseTwoPaintingsOfDifferentTypes = 59 := by
  sorry

end question1_question2_l191_191529


namespace remainder_equality_l191_191349

variable (P P' Q D : ℕ) 
variable (h1 : P > P')
variable (h2 : Q > 0)
variable (h3 : P < D)
variable (h4 : P' < D)
variable (h5 : Q < D)

def remainder (a b : ℕ) : ℕ := a % b

noncomputable def R : ℕ := remainder P D
noncomputable def R' : ℕ := remainder P' D
noncomputable def s : ℕ := remainder (P + P') D
noncomputable def s' : ℕ := remainder (R + R') D

theorem remainder_equality : s = s' :=
sorry

end remainder_equality_l191_191349


namespace must_be_divisor_of_p_l191_191959

theorem must_be_divisor_of_p (p q r s : ℕ) (hpq : Nat.gcd p q = 40) (hqr : Nat.gcd q r = 50) 
  (hrs : Nat.gcd r s = 75) (hsp : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) : 17 ∣ p :=
sorry

end must_be_divisor_of_p_l191_191959


namespace add_to_frac_eq_l191_191568

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l191_191568


namespace possible_remainders_of_b_l191_191068

-- Define the conditions
def congruent_mod (a b n : ℤ) : Prop := (a - b) % n = 0

variables {n : ℤ} (hn : n > 0)

theorem possible_remainders_of_b (n : ℤ) (hn : n > 0) :
  ∃ b : ℤ, b ∈ {8, 5, 3, 10} ∧ 
           (congruent_mod b (5^(2*n) + 6)⁻¹ 11 ∧
           ¬ congruent_mod (5^(2*n) + 6) 0 11) :=
by sorry

end possible_remainders_of_b_l191_191068


namespace snail_distance_at_44th_day_l191_191634

theorem snail_distance_at_44th_day :
  (∑ n in Finset.range 44, 1 / (n + 1) - 1 / (n + 2)) = 44 / 45 := by
  sorry

end snail_distance_at_44th_day_l191_191634


namespace c_work_time_l191_191191

theorem c_work_time (A B C : ℝ) 
  (h1 : A + B = 1/10) 
  (h2 : B + C = 1/5) 
  (h3 : C + A = 1/15) : 
  C = 1/12 :=
by
  -- Proof will go here
  sorry

end c_work_time_l191_191191


namespace range_of_EM_l191_191776

variables {x_0 : ℝ}

-- Definitions of the given conditions
def parabola (x : ℝ) : ℝ := (1 / 2) * x^2
def circle (x y : ℝ) : Prop := x^2 + y^2 = 1
def line_tangent (m x x0 : ℝ) (y : ℝ) : Prop := y = m * (x - x0) + parabola x_0
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Point E definition
def E : ℝ × ℝ := (0, -1 / 2)

-- The main statement to prove
theorem range_of_EM {A B M : ℝ × ℝ}
  (hA : circle A.1 A.2) (hB : circle B.1 B.2)
  (hM : M = midpoint A B)
  (hl : ∃ x0, line_tangent x0 A.1 x0 A.2 ∧ line_tangent x0 B.1 x0 B.2)
  : ∀ E = (0, -1 / 2), (distance E M ∈ Icc (Real.sqrt (2 * Real.sqrt 3 - 3) / 2) ((2 * Real.sqrt 2 - 1) / 2)) :=
sorry

end range_of_EM_l191_191776


namespace problem_1_problem_2_l191_191795

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), Real.cos (2 * x))
noncomputable def vec_b : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).fst * vec_b.fst + (vec_a x).snd * vec_b.snd + 2

theorem problem_1 (x : ℝ) : x ∈ Set.Icc (k * Real.pi - (5 / 12) * Real.pi) (k * Real.pi + (1 / 12) * Real.pi) → ∃ k : ℤ, ∀ x : ℝ, f (x) = Real.sin (2 * x + (1 / 3) * Real.pi) + 2 :=
sorry

theorem problem_2 (x : ℝ) : x ∈ Set.Icc (π / 6) (2 * π / 3) → f (π / 6) = (Real.sqrt 3 / 2) + 2 ∧ f (7 * π / 12) = 1 :=
sorry

end problem_1_problem_2_l191_191795


namespace select_president_and_vp_of_opposite_genders_l191_191095

theorem select_president_and_vp_of_opposite_genders
  (total_members : ℕ)
  (num_males : ℕ)
  (num_females : ℕ)
  (no_dual_offices : Prop) :
  total_members = 24 →
  num_males = 14 →
  num_females = 10 →
  (no_dual_offices = true) →
  (num_males * num_females + num_females * num_males) = 280 :=
by
  intros h_total h_males h_females h_no_dual
  rw [h_total, h_males, h_females]
  sorry

end select_president_and_vp_of_opposite_genders_l191_191095


namespace smallest_integer_with_conditions_l191_191511

theorem smallest_integer_with_conditions (x : ℕ) : 
  (∃ x, x.factors.count = 18 ∧ 18 ∣ x ∧ 24 ∣ x) → x = 972 :=
by
  sorry

end smallest_integer_with_conditions_l191_191511


namespace log_expression_defined_l191_191672

theorem log_expression_defined (x : ℝ) : ∃ c : ℝ, (∀ x > c, (x > 7^8)) :=
by
  existsi 7^8
  intro x hx
  sorry

end log_expression_defined_l191_191672


namespace center_of_circle_is_correct_l191_191612

-- Define the conditions as Lean functions and statements
def is_tangent (x y : ℝ) : Prop :=
  (3 * x + 4 * y = 48) ∨ (3 * x + 4 * y = -12)

def is_on_line (x y : ℝ) : Prop := x = y

-- Define the proof statement
theorem center_of_circle_is_correct (x y : ℝ) (h1 : is_tangent x y) (h2 : is_on_line x y) :
  (x, y) = (18 / 7, 18 / 7) :=
sorry

end center_of_circle_is_correct_l191_191612


namespace value_of_c7_l191_191466

theorem value_of_c7 
  (a : ℕ → ℕ)
  (b : ℕ → ℕ)
  (c : ℕ → ℕ)
  (h1 : ∀ n, a n = n)
  (h2 : ∀ n, b n = 2^(n-1))
  (h3 : ∀ n, c n = a n * b n) :
  c 7 = 448 :=
by
  sorry

end value_of_c7_l191_191466


namespace find_x_l191_191197

theorem find_x (x : ℝ) (h : 0.25 * x = 0.10 * 500 - 5) : x = 180 :=
by
  sorry

end find_x_l191_191197


namespace two_integer_solutions_iff_range_a_l191_191152

theorem two_integer_solutions_iff_range_a (a : ℝ) :
  (∃ int_solutions : set ℤ, (∀ x ∈ int_solutions, (x : ℝ)^2 - x + a - a^2 < 0 ∧ (x : ℝ) + 2 * a > 1) ∧ int_solutions.size = 2) ↔ (1 < a ∧ a ≤ 2) := 
sorry

end two_integer_solutions_iff_range_a_l191_191152


namespace area_region_z1_z2_l191_191422

-- Definitions based on given conditions
def z1 (t : ℝ) : ℂ := t + complex.I * (1 - t) -- Represents the line segment
def z2 (θ : ℝ) : ℂ := complex.exp (complex.I * θ) -- Represents the unit circle

-- The condition range
axiom h_t : ∀ t, 0 ≤ t ∧ t ≤ 1
axiom h_θ : ∀ θ, 0 ≤ θ ∧ θ < 2 * real.pi

-- The statement to prove
theorem area_region_z1_z2 : (∃ r, r = 2 * real.sqrt 2 + real.pi) :=
sorry

end area_region_z1_z2_l191_191422


namespace smallest_x_with_18_factors_and_factors_18_24_l191_191509

theorem smallest_x_with_18_factors_and_factors_18_24 :
  ∃ (x : ℕ), (∃ (a b : ℕ), x = 2^a * 3^b ∧ 18 ∣ x ∧ 24 ∣ x ∧ (a + 1) * (b + 1) = 18) ∧
    (∀ y, (∃ (c d : ℕ), y = 2^c * 3^d ∧ 18 ∣ y ∧ 24 ∣ y ∧ (c + 1) * (d + 1) = 18) → x ≤ y) :=
by
  sorry

end smallest_x_with_18_factors_and_factors_18_24_l191_191509


namespace odd_function_parallicity_l191_191818

def parallicity (f : ℝ → ℝ) :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = f x2 ∧ f' x1 = f' x2

theorem odd_function_parallicity
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_deriv_odd : ∀ x, deriv f (-x) = - deriv f x) :
  parallicity f :=
sorry

end odd_function_parallicity_l191_191818


namespace convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191874

theorem convert_polar_to_rectangular (ρ θ x y : ℝ) (hρ : ρ = 2 * √2 * cos θ) :
  (x - √2) ^ 2 + y ^ 2 = 2 :=
sorry

theorem parametric_equations_and_no_common_points (x y θ : ℝ) :
  let A := (1 : ℝ, 0 : ℝ)
  let P (θ : ℝ) := (3 - √2 + 2 * cos θ, 2 * sin θ)
  let C1 := (P 0).1 ^ 2 + (P 0).2 ^ 2 = 4
  (x - √2) ^ 2 + y ^ 2 = 2 ∧ (C1 = [((x, y) = P θ)]) → False :=
sorry

end convert_polar_to_rectangular_parametric_equations_and_no_common_points_l191_191874


namespace problem_statement_l191_191832

open Set

variables {Point : Type} [MetricSpace Point]

-- Define circle, tangent line, midpoint, and inscribed triangle conditions
noncomputable def circle (c : Set Point) : Prop := sorry
noncomputable def tangent (l : Set Point) (c : Set Point) : Prop := sorry
noncomputable def midpoint (M Q R : Point) : Prop := sorry
noncomputable def inscribed (c : Set Point) (P Q R : Point) : Prop := sorry
noncomputable def extended_line (N T P : Point) : Prop := sorry

theorem problem_statement {c : Set Point} {l : Set Point} {M Q R P N T : Point}
  (hc : circle c)
  (hl : tangent l c)
  (hM : M ∈ l)
  (hmid : midpoint M Q R)
  (hins : inscribed c P Q R) :
  extended_line N T P := sorry

end problem_statement_l191_191832


namespace add_to_fraction_l191_191562

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l191_191562


namespace sum_abs_arith_prog_l191_191525

-- Given the conditions:
-- 1. The sum of the absolute values of the terms of a finite arithmetic progression is 100.
-- 2. The sum remains 100 when all terms are increased by 1 or 2.
theorem sum_abs_arith_prog (n d : ℕ) (a : ℕ → ℤ) 
  (h_sum : (∑ i in finset.range n, |a i|) = 100)
  (h_sum_inc1 : (∑ i in finset.range n, |a i + 1|) = 100)
  (h_sum_inc2 : (∑ i in finset.range n, |a i + 2|) = 100) :
  n^2 * d = 400 :=
sorry

end sum_abs_arith_prog_l191_191525


namespace sequence_bound_l191_191341

theorem sequence_bound (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_condition : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
sorry

end sequence_bound_l191_191341


namespace carpet_cost_576_l191_191629

noncomputable def rectangular_floor_cost : ℕ :=
let floor_length := 24 in
let floor_width := 64 in
let carpet_side := 8 in
let carpet_cost := 24 in
let floor_area := floor_length * floor_width in
let carpet_area := carpet_side * carpet_side in
let num_carpet_squares := floor_area / carpet_area in
let total_cost := num_carpet_squares * carpet_cost in
total_cost

theorem carpet_cost_576 : rectangular_floor_cost = 576 := by
  unfold rectangular_floor_cost
  rfl

end carpet_cost_576_l191_191629


namespace number_of_games_played_l191_191401

-- Define our conditions
def teams : ℕ := 14
def games_per_pair : ℕ := 5

-- Define the function to calculate the number of combinations
def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the expected total games
def total_games : ℕ := 455

-- Statement asserting that given the conditions, the number of games played in the season is total_games
theorem number_of_games_played : (combinations teams 2) * games_per_pair = total_games := 
by 
  sorry

end number_of_games_played_l191_191401


namespace triangle_sides_geometric_progression_l191_191472

noncomputable theory
open Classical

variables {A B C L : Point ( EuclideanSpace 3 ℝ )}

-- Definitions of the conditions 
def angle_bisector (A B C L : Point ( EuclideanSpace 3 ℝ )) : Prop := 
  ∃ α : ℝ, α < 90 ∧ angle A B L = α ∧ angle B C L = α ∧ angle C A L = α

-- Definition of the geometric progression property
def geometric_progression (a b c : ℝ) : Prop := 
  a / b = b / c

theorem triangle_sides_geometric_progression 
  (h1 : angle_bisector A B C L) :
  geometric_progression (dist A B) (dist B C) (dist C A) :=
sorry

end triangle_sides_geometric_progression_l191_191472


namespace collinear_points_l191_191536

open EuclideanGeometry

variables {A B C O S P : Point}

-- *Import necessary axioms related to circles, triangles, tangents, etc.*
axiom inscribed_triangle : ∀ (A B C O : Point),
  Circle.inscribed O A B C →
  ∃ Ω : Circle, Circle.inside Ω A ∧ Circle.inside Ω B ∧ Circle.inside Ω C

axiom circle_diameter : ∀ (A O : Point) (ω : Circle),
  ω.radius = dist A O / 2 → Circle.main_diameter ω A O

axiom circumcircle_intersect : ∀ (ω1 ω2 : Circle) (A O B C : Point),
  ω1.diameter A O → Circle.circumcircle ω2 O B C → ∃ S, S ≠ O ∧ Circle.intersect ω1 ω2 S

axiom tangents_intersect : ∀ (Ω : Circle) (B C : Point),
  Circle.ontangents Ω B C → ∃ P, Line.tangent_intersect Ω B C P

axiom perpendicular_diameter : ∀ (A O P S : Point) (ω : Circle),
  ω.diameter A O → ω.center = O → Circle.on ω S →
  Line.perpendicular (Line.through A P) (Line.through O S)

theorem collinear_points : ∀ (A B C O S P : Point) (Ω : Circle) (ω : Circle),
  Circle.inscribed O A B C → Circle.diameter ω A O →
  Circle.circumcircle Ω O B C → Circle.intersect Ω ω S →
  Tangents_intersect Ω B C P → Line.collinear {A, S, P} :=
by
  intros A B C O S P Ω ω H1 H2 H3 H4 H5
  sorry

end collinear_points_l191_191536


namespace maximum_pairs_l191_191321

-- Let's define our conditions
variables {a b : ℤ} (set : set ℤ)
def conditions (a b : ℤ) (set : set ℤ) :=
  a < b ∧
  a ∈ set ∧ b ∈ set ∧
  (∀ i j, i ≠ j → ∀ x y, x ≠ y → (x ∈ set ∧ y ∈ set → i + j ≠ x + y)) ∧
  a + b ≤ 2009

-- Proving the maximum number of pairs k
theorem maximum_pairs : 
  ∃ (k : ℕ), k = 803 ∧ ∀ (pairs : list (ℤ × ℤ)),
  (∀ pair ∈ pairs, conditions pair.1 pair.2 set)
  → list.length pairs ≤ k :=
sorry

end maximum_pairs_l191_191321


namespace largest_k_for_inequality_l191_191309

theorem largest_k_for_inequality :
  ∃ k : ℝ, (k = 27 / 4) ∧ ∀ a b : ℝ, 0 < a → 0 < b →
  (a + b) * (a * b + 1) * (b + 1) ≥ k * a * b^2 :=
by
  use 27 / 4
  split
  { norm_num }
  { intros a b a_pos b_pos
    sorry
  }

end largest_k_for_inequality_l191_191309


namespace line_OP_eq_line_l_intersects_circle_C_l191_191424

/-- Define the polar coordinates for points M and N. --/
noncomputable def M := (2 : ℝ, 0 : ℝ)
noncomputable def N := (2 * Real.sqrt 3 / 3 : ℝ, Real.pi / 2 : ℝ)

/-- Define the parametric equations for the circle C. --/
noncomputable def circle_x (θ : ℝ) := 2 + 2 * Real.cos θ
noncomputable def circle_y (θ : ℝ) := Real.sqrt 3 + 2 * Real.sin θ

/-- Midpoint P of segment MN. --/
noncomputable def midpoint_P := (1 : ℝ, Real.sqrt 3 / 3 : ℝ)

/-- Prove the rectangular equation of the line OP. --/
theorem line_OP_eq : ∃ (a b : ℝ), a = 0 ∧ b = Real.sqrt 3 / 3 ∧ 
  (∀ x y, y = midpoint_P.2 / midpoint_P.1 * x) :=
sorry

/-- Define the rectangular coordinates of points M and N before finding the equation of line l. --/
noncomputable def M_rect := (2 : ℝ, 0 : ℝ)
noncomputable def N_rect := (0 : ℝ, 2 * Real.sqrt 3 / 3 : ℝ)

/-- Prove that line l intersects circle C. --/
theorem line_l_intersects_circle_C : 
  let line_l := (x y : ℝ) → x + Real.sqrt 3 * y - 2 = 0 in
  let center_C := (2 : ℝ, Real.sqrt 3 : ℝ) in
  let radius_C := 2 in
  let distance_to_line (x₀ y₀ : ℝ) (A B C : ℝ) := 
    abs ((A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)) in
  distance_to_line center_C.1 center_C.2 1 (Real.sqrt 3) (-2) < radius_C :=
sorry

end line_OP_eq_line_l_intersects_circle_C_l191_191424


namespace rita_daily_minimum_payment_l191_191481

theorem rita_daily_minimum_payment (total_cost down_payment balance daily_payment : ℝ) 
    (h1 : total_cost = 120)
    (h2 : down_payment = total_cost / 2)
    (h3 : balance = total_cost - down_payment)
    (h4 : daily_payment = balance / 10) : daily_payment = 6 :=
by
  sorry

end rita_daily_minimum_payment_l191_191481


namespace subset_intersection_exists_l191_191433

theorem subset_intersection_exists (α : ℝ) (hα : α < (3 - Real.sqrt 5) / 2) :
  ∃ (n p : ℕ), p > α * 2^n ∧ ∃ (S T : Finset (Finset (Fin n))),
  S.card = p ∧ T.card = p ∧ 
  (∀ i j, i ∈ S → j ∈ T → (i ∩ j).nonempty) :=
by
  sorry

end subset_intersection_exists_l191_191433


namespace smallest_number_l191_191348

def binary_101010 : ℕ := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0
def base5_111 : ℕ := 1 * 5^2 + 1 * 5^1 + 1 * 5^0
def octal_32 : ℕ := 3 * 8^1 + 2 * 8^0
def base6_54 : ℕ := 5 * 6^1 + 4 * 6^0

theorem smallest_number : octal_32 < binary_101010 ∧ octal_32 < base5_111 ∧ octal_32 < base6_54 :=
by
  sorry

end smallest_number_l191_191348


namespace abs_diff_gt_cube_root_mul_l191_191810

theorem abs_diff_gt_cube_root_mul (a b : ℤ) (h_diff : a ≠ b) (h_div : (a^2 + a * b + b^2) ∣ (a * b * (a + b))) : 
  abs (a - b) > (↑(real.sqrt (real.cbrt (a * b)))) := 
sorry

end abs_diff_gt_cube_root_mul_l191_191810


namespace passenger_difference_l191_191469

theorem passenger_difference {x : ℕ} :
  (30 + x = 3 * x + 14) →
  6 = 3 * x - x - 16 :=
by
  sorry

end passenger_difference_l191_191469


namespace value_of_nested_fraction_l191_191292

theorem value_of_nested_fraction :
  10 + 5 + (1 / 2) * (9 + 5 + (1 / 2) * (8 + 5 + (1 / 2) * (7 + 5 + (1 / 2) * (6 + 5 + (1 / 2) * (5 + 5 + (1 / 2) * (4 + 5 + (1 / 2) * (3 + 5 ))))))) = 28 + (1 / 128) :=
sorry

end value_of_nested_fraction_l191_191292


namespace congruent_triangles_l191_191242

theorem congruent_triangles (A B C D E F : ℝ)
  (h1 : dist A B = dist D E)
  (h2 : dist B C = dist E F)
  (h3 : dist C A = dist F D) :
  (triangle.congruent A B C D E F) :=
by
  sorry

end congruent_triangles_l191_191242


namespace more_than_10_numbers_with_sum_20_l191_191999

theorem more_than_10_numbers_with_sum_20
    (a : ℕ → ℕ)
    (len : ℕ)
    (sum_eq_20 : ∑ i in finset.range len, a i = 20)
    (no_elem_eq_3 : ∀ i < len, a i ≠ 3)
    (no_consec_sum_eq_3 : ∀ i j, 0 ≤ i → i < j → j ≤ len → (∑ k in finset.range (j - i), a (i + k)) ≠ 3) :
  len > 10 :=
begin
  sorry
end

end more_than_10_numbers_with_sum_20_l191_191999


namespace circle_center_l191_191613

theorem circle_center (a b : ℝ) :
  (a ≠ 1 ∧ b ≠ 1 ∧ (a - 1) * (-1/2) = b - 1) ∧
  (2 * b + a = 3) ∧
  (b - (2) * (a - 1)) = (0 - 2) →  -- Conditions from the problem.
  (∃ a b : ℝ, a ≠ 1 ∧ b ≠ 1 ∧ a = 1/3 ∧ b = 4/3) :=
begin
  intros,
  -- Proof skipped, just the statement is here.
  sorry
end

end circle_center_l191_191613


namespace polar_to_rect_eq_and_locus_no_common_points_l191_191861

theorem polar_to_rect_eq_and_locus_no_common_points :
  let C := { (x, y) | (x - Real.sqrt 2)^2 + y^2 = 2 } in
  let C1 := { (x, y) | ∃ θ, x = 3 - Real.sqrt 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ } in
  (∀ (ρ θ : Real), ρ = 2 * Real.sqrt 2 * Real.cos θ → (Real.sqrt (ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos θ) - Real.sqrt 2)^2 + (Real.sin θ)^2 = 2 )
  ∧ ¬∃ (p : Real × Real), p ∈ C ∧ p ∈ C1 :=
by
  sorry

end polar_to_rect_eq_and_locus_no_common_points_l191_191861


namespace busy_squirrels_count_l191_191456

variable (B : ℕ)
variable (busy_squirrel_nuts_per_day : ℕ := 30)
variable (sleepy_squirrel_nuts_per_day : ℕ := 20)
variable (days : ℕ := 40)
variable (total_nuts : ℕ := 3200)

theorem busy_squirrels_count : busy_squirrel_nuts_per_day * days * B + sleepy_squirrel_nuts_per_day * days = total_nuts → B = 2 := by
  sorry

end busy_squirrels_count_l191_191456


namespace jake_peaches_l191_191935

theorem jake_peaches (steven jake : ℕ) (h1 : steven = 16) (h2 : jake = steven - 7) : jake = 9 := by
  rw [h1, h2]
  norm_num

end jake_peaches_l191_191935


namespace find_hyperbola_parameter_l191_191790

theorem find_hyperbola_parameter (m : ℝ) : (∀ x y : ℝ, (y = (sqrt 2 / 2) * x ∨ y = -(sqrt 2 / 2) * x) 
    ↔ (x^2 / 4 - y^2 / m = 1)) → 
    m = 2 :=
by
  assume hyp : ∀ x y : ℝ, (y = (sqrt 2 / 2) * x ∨ y = -(sqrt 2 / 2) * x) ↔ (x^2 / 4 - y^2 / m = 1)
  have : sqrt m / 2 = sqrt 2 / 2 := sorry
  exact sorry

end find_hyperbola_parameter_l191_191790


namespace ladybugs_without_spots_l191_191488

-- Defining the conditions given in the problem
def total_ladybugs : ℕ := 67082
def ladybugs_with_spots : ℕ := 12170

-- Proving the number of ladybugs without spots
theorem ladybugs_without_spots : total_ladybugs - ladybugs_with_spots = 54912 := by
  sorry

end ladybugs_without_spots_l191_191488


namespace line_equation_l191_191777

-- Define the point and slope
def point := (2, -1 : ℝ × ℝ)
def slope : ℝ := 2

-- The equation of the line passing through the given point with the given slope
theorem line_equation (h : slope = 2 ∧ point = (2, -1)) : 
    ∃ (a b c : ℝ), a * 2 + b * (-1) + c = 0 ∧  a = 2 ∧ b = -1 ∧ c = -5 := 
by 
    sorry

end line_equation_l191_191777


namespace probability_solution_l191_191259

theorem probability_solution : 
  let N := (3/8 : ℝ)
  let M := (5/8 : ℝ)
  let P_D_given_N := (x : ℝ) => x^2
  (3 : ℝ) * x^2 - (8 : ℝ) * x + (5 : ℝ) = 0 → x = 1 := 
by
  sorry

end probability_solution_l191_191259


namespace common_area_shaded_l191_191283

open Real

-- Define the original conditions
def side_length := 2.0
def cos_beta := 3.0 / 5.0

-- Noncomputable due to square root
noncomputable def sin_beta := sqrt (1 - cos_beta^2)

-- Define the range for beta angle
def beta_range (β : ℝ) : Prop := 0 < β ∧ β < π / 2

-- Define the problem of finding the common area
theorem common_area_shaded (β : ℝ) (hβ : beta_range β) (hcos : cos β = cos_beta) :
  -- area of the shaded region is 2/3
  let area_common := 2.0 / 3.0 
  in true :=
sorry

end common_area_shaded_l191_191283


namespace angle_BXY_l191_191039

theorem angle_BXY (
  AB_CD_parallel: parallel (line AB) (line CD),
  angle_condition: ∀ (x y : ℝ), angle AXE = 4 * angle CYX - 120,
  corresponding_angles: angle AXE = angle CYX
  ) : angle BXY = 40 := by
  sorry

end angle_BXY_l191_191039


namespace construct_segment_through_point_l191_191447

theorem construct_segment_through_point {α β γ : ℝ} (A O B P C D : EuclideanGeometry.Point)
  (h_angle : EuclideanGeometry.angle O A B < π)
  (h_interior : EuclideanGeometry.is_interior P (EuclideanGeometry.angle O A B))
  (h_on_ray_C : EuclideanGeometry.is_on_ray C O A)
  (h_on_ray_D : EuclideanGeometry.is_on_ray D O B)
  (h_pass_through_P : EuclideanGeometry.is_on_line P C D)
  : EuclideanGeometry.ratio_segments C P D = 1 / 2 :=
sorry

end construct_segment_through_point_l191_191447


namespace number_thought_of_eq_95_l191_191592

theorem number_thought_of_eq_95 (x : ℝ) (h : (x / 5) + 23 = 42) : x = 95 := 
by
  sorry

end number_thought_of_eq_95_l191_191592


namespace rhombus_diagonal_length_l191_191123

-- Let d1 and d2 be the lengths of the diagonals of the rhombus and A be its area.
variables (d1 d2 A : ℝ)

-- Given conditions
axiom h1 : d2 = 20
axiom h2 : A = 300
axiom area_formula : A = (d1 * d2) / 2

-- Prove that the length of the other diagonal (d1) is 30 cm
theorem rhombus_diagonal_length : d1 = 30 :=
by
  -- Import the necessary module for real numbers and algebra
  sorry

end rhombus_diagonal_length_l191_191123


namespace max_value_sqrt_expression_l191_191726

theorem max_value_sqrt_expression : 
  ∀ (x : ℝ), -36 ≤ x ∧ x ≤ 36 → 
  sqrt (36 + x) + sqrt (36 - x) ≤ 12 :=
by
  -- Proof goes here
  sorry

end max_value_sqrt_expression_l191_191726


namespace emily_finishes_first_l191_191671

variable (z r : ℝ)
variable (hz_pos : 0 < z)
variable (hr_pos : 0 < r)

-- Define lawn sizes
def area_david : ℝ := z
def area_emily : ℝ := z / 3
def area_frank : ℝ := 2 * z

-- Define mowing rates
def rate_emily : ℝ := r
def rate_david : ℝ := 2 * r / 3
def rate_frank : ℝ := 2 * r

-- Define mowing times
def time_david : ℝ := area_david / rate_david
def time_emily : ℝ := area_emily / rate_emily
def time_frank : ℝ := area_frank / rate_frank

theorem emily_finishes_first (hz_pos : 0 < z) (hr_pos : 0 < r) :
    time_emily z r < time_david z r ∧ time_emily z r < time_frank z r :=
by
  sorry

end emily_finishes_first_l191_191671


namespace value_of_c7_l191_191467

def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

theorem value_of_c7 : c 7 = 448 := by
  sorry

end value_of_c7_l191_191467


namespace determine_parity_of_f_l191_191619

def parity_of_f (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = 0

theorem determine_parity_of_f (f : ℝ → ℝ) :
  (∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)) →
  parity_of_f f :=
sorry

end determine_parity_of_f_l191_191619


namespace max_value_sqrt_expr_max_reaches_at_zero_l191_191719

theorem max_value_sqrt_expr (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) : sqrt (36 + x) + sqrt (36 - x) ≤ 12 :=
by sorry

theorem max_reaches_at_zero : sqrt (36 + 0) + sqrt (36 - 0) = 12 :=
by sorry

end max_value_sqrt_expr_max_reaches_at_zero_l191_191719


namespace remainder_of_S_is_one_l191_191735

theorem remainder_of_S_is_one :
  let S := (∑ k in Finset.range 50, (2 * k + 1)^2 - (2 * k + 2)^2) + 101^2
  in S % 103 = 1 :=
by
  sorry

end remainder_of_S_is_one_l191_191735


namespace number_of_real_roots_l191_191060

noncomputable def num_real_roots_det (a b c d : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0) : ℕ :=
  let p := λ (x : ℝ), x * (x^2 + b^2 + c^2 + d^2) in
  if h₄ : b^2 + c^2 + d^2 = 0 then 0 else 1

theorem number_of_real_roots (a b c d : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0) :
  num_real_roots_det a b c d h₀ h₁ h₂ h₃ = 1 :=
sorry

end number_of_real_roots_l191_191060


namespace candle_lighting_time_l191_191537

noncomputable def burn_time (l : ℝ) : ℝ :=
  (1 - l / 300) * 1 + (1 - l / 180) * 3

theorem candle_lighting_time (ell t : ℝ) :
  (∀ t, (ell * (1 - t / 180)) = 3 * (ell * (1 - t / 300))) → t = 180 :=
by
  assume h: (ell * (1 - t / 180)) = 3 * (ell * (1 - t / 300)),
  sorry

end candle_lighting_time_l191_191537


namespace sushi_father_lollipops_l191_191111

-- Define the conditions
def lollipops_eaten : ℕ := 5
def lollipops_left : ℕ := 7

-- Define the total number of lollipops brought
def total_lollipops := lollipops_eaten + lollipops_left

-- Proof statement
theorem sushi_father_lollipops : total_lollipops = 12 := sorry

end sushi_father_lollipops_l191_191111


namespace amount_of_tin_in_new_alloy_l191_191598

theorem amount_of_tin_in_new_alloy
    (weight_alloy_A : ℝ) (weight_alloy_B : ℝ)
    (ratio_lead_tin_A : (ℝ × ℝ)) (ratio_tin_copper_B : (ℝ × ℝ))
    (h1 : weight_alloy_A = 135)
    (h2 : weight_alloy_B = 145)
    (h3 : ratio_lead_tin_A = (3, 5))
    (h4 : ratio_tin_copper_B = (2, 3))
    : let weight_tin_A := (ratio_lead_tin_A.2 / (ratio_lead_tin_A.1 + ratio_lead_tin_A.2)) * weight_alloy_A in
      let weight_tin_B := (ratio_tin_copper_B.1 / (ratio_tin_copper_B.1 + ratio_tin_copper_B.2)) * weight_alloy_B in
      weight_tin_A + weight_tin_B = 142.375 := by 
{
  sorry
}

end amount_of_tin_in_new_alloy_l191_191598


namespace find_number_to_add_l191_191572

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l191_191572


namespace ratio_of_two_numbers_l191_191151

variable {a b : ℝ}

theorem ratio_of_two_numbers
  (h1 : a + b = 7 * (a - b))
  (h2 : 0 < b)
  (h3 : a > b) :
  a / b = 4 / 3 := by
  sorry

end ratio_of_two_numbers_l191_191151


namespace domain_of_f_l191_191306

-- Define the function
def f (x : ℝ) : ℝ := sqrt (2 - x) + 1 / x

-- Define the conditions
def condition1 (x : ℝ) : Prop := 2 - x ≥ 0
def condition2 (x : ℝ) : Prop := x ≠ 0

-- Define the domain
def domain (x : ℝ) : Prop := (x ∈ Set.Iic 2 ∧ x ≠ 0)

-- Define the goal statement
theorem domain_of_f : {x : ℝ | domain x} = {x : ℝ | x ∈ (Set.Iio 0) ∪ Set.Ioc 0 2} :=
by
  sorry

end domain_of_f_l191_191306


namespace total_earnings_l191_191934

theorem total_earnings (hourly_wage : ℕ) (men : ℕ) (jobs : ℕ) (hours_per_job : ℕ) :
  men = 3 → hourly_wage = 10 → jobs = 5 → hours_per_job = 1 → 
  men * hourly_wage * hours_per_job * jobs = 150 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end total_earnings_l191_191934


namespace houston_bound_passes_dallas_bound_l191_191269

theorem houston_bound_passes_dallas_bound :
  ∀ (departure_time interval travel_time : ℕ), departure_time = 1230 ∧ interval = 60 ∧ travel_time = 5 →
  ∃ n : ℕ, n = 10 :=
by
  intros departure_time interval travel_time h
  rcases h with ⟨dep_time_eq, interval_eq, travel_time_eq⟩
  have n : ℕ := 10
  use n
  sorry

end houston_bound_passes_dallas_bound_l191_191269


namespace part_a_part_b_l191_191442

noncomputable theory

-- Define the basic elements: points, triangle, and segments
variables {A B C D E F I P G : Type} [EuclideanGeometry A B C D E F I P G]

-- Define the given conditions
variables (triangleABC : Triangle A B C)
variables (D_on_AB : On D (Segment A B))
variables (CD_eq_AC : CD = AC)
variables (incircle_tangentE : On E (Incircle triangleABC))
variables (incircle_tangentF : On F (Incircle triangleABC))
variables (I_eq_incenterBCD : I = incenter (triangle B C D))
variables (P_eq_intersectionAI_EF : P = intersection (Segment A I) (Segment E F))
variables (G_on_AB : On G (Segment A B))
variables (IG_parallel_EF : Parallel (Segment I G) (Segment E F))

-- Part (a): Prove DI = IG
theorem part_a : segment_length (Segment D I) = segment_length (Segment I G) := sorry

-- Part (b): Prove AP = PI
theorem part_b : segment_length (Segment A P) = segment_length (Segment P I) := sorry

end part_a_part_b_l191_191442


namespace least_n_for_distance_l191_191056

theorem least_n_for_distance (n : ℕ) : n = 17 ↔ (100 ≤ n * (n + 1) / 3) := sorry

end least_n_for_distance_l191_191056


namespace valid_number_of_odd_integers_equals_two_l191_191390

noncomputable def numValidOddIntegers : ℕ :=
  {n : ℕ | odd n ∧ n ≥ 2 ∧ (∀ (z_i : ℕ → ℂ), 
    (∀ i, |z_i i| = 1) ∧ (∑ i in finset.range n, z_i i = 0) ∧ 
    (∀ i, ∃ k : ℤ, angle (z_i ((i + 1) % n)) (z_i i) = k * (Real.pi / 4))) →
      (∃ k : ℤ, ∀ i j, (z_i j = exp (k * (Real.pi * Complex.I * of_nat j / n))) ∧
                     (i ≠ j → z_i i ≠ z_i j))
  }.to_finset.card

theorem valid_number_of_odd_integers_equals_two :
    numValidOddIntegers = 2 := sorry

end valid_number_of_odd_integers_equals_two_l191_191390


namespace right_triangle_longer_leg_l191_191030

theorem right_triangle_longer_leg (a b c : ℕ) (h₀ : a^2 + b^2 = c^2) (h₁ : c = 65) (h₂ : a < b) : b = 60 :=
sorry

end right_triangle_longer_leg_l191_191030


namespace no_2023_integers_product_sum_l191_191676

theorem no_2023_integers_product_sum (a : Fin 2023 → Int) 
  (h1 : (∏ i, a i) = 2023) 
  (h2 : (∑ i, a i) = 0) : False := 
sorry

end no_2023_integers_product_sum_l191_191676


namespace num_integers_in_abs_inequality_l191_191389

theorem num_integers_in_abs_inequality : 
  (∃! n : ℕ, ∀ x : ℤ, abs (x - 3) ≤ 74/10 → x ∈ set.range (λ k : ℤ, -4 + k) ∧ x ≤ 10) :=
sorry

end num_integers_in_abs_inequality_l191_191389


namespace unique_a0_l191_191439

noncomputable def a_n (a_0 : ℚ) (n : ℕ) : ℚ :=
a_0 + (3^n - 1)

theorem unique_a0 (a_0 : ℚ) :
(∀ (j k : ℕ), 0 < j → j < k → ((a_n a_0 k)^j / (a_n a_0 j)^k).denominator = 1) →
a_0 = 1 :=
sorry

end unique_a0_l191_191439


namespace hyperbola_equation_l191_191361

open Real

-- Define the conditions in Lean
def is_hyperbola_form (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def is_positive (x : ℝ) : Prop := x > 0

def parabola_focus : (ℝ × ℝ) := (1, 0)

def hyperbola_vertex_eq_focus (a : ℝ) : Prop := a = parabola_focus.1

def hyperbola_eccentricity (e a c : ℝ) : Prop := e = c / a

-- Our proof statement
theorem hyperbola_equation :
  ∃ (a b : ℝ), is_positive a ∧ is_positive b ∧
  hyperbola_vertex_eq_focus a ∧
  hyperbola_eccentricity (sqrt 5) a (sqrt 5) ∧
  is_hyperbola_form a b 1 0 :=
by sorry

end hyperbola_equation_l191_191361


namespace no_non_zero_integers_doubled_by_digit_rearrangement_l191_191105

theorem no_non_zero_integers_doubled_by_digit_rearrangement :
  ∀ (d : ℕ) (m : ℕ), d ∈ {2, 4} → 
  ∃ (X : ℤ), X = (d * (2 * 10^m - 1) / 8) → 
  X < 0 ∨ X * 8 ≠ d * (2 * 10^m - 1) :=
by
  sorry

end no_non_zero_integers_doubled_by_digit_rearrangement_l191_191105


namespace imaginary_part_of_z_l191_191817

noncomputable def z : ℂ := sorry

theorem imaginary_part_of_z : (z - complex.I = (4 - 2 * complex.I) / (1 + 2 * complex.I)) → im z = -1 :=
by
  intros h
  sorry

end imaginary_part_of_z_l191_191817


namespace correlations_are_1_3_4_l191_191129

def relation1 : Prop := ∃ (age wealth : ℝ), true
def relation2 : Prop := ∀ (point : ℝ × ℝ), ∃ (coords : ℝ × ℝ), coords = point
def relation3 : Prop := ∃ (yield : ℝ) (climate : ℝ), true
def relation4 : Prop := ∃ (diameter height : ℝ), true
def relation5 : Prop := ∃ (student : Type) (school : Type), true

theorem correlations_are_1_3_4 :
  (relation1 ∨ relation3 ∨ relation4) ∧ ¬ (relation2 ∨ relation5) :=
sorry

end correlations_are_1_3_4_l191_191129


namespace value_of_c7_l191_191468

def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

theorem value_of_c7 : c 7 = 448 := by
  sorry

end value_of_c7_l191_191468


namespace chord_length_l191_191310

theorem chord_length (x y : ℝ) :
    (x^2 + y^2 - 2*y = 0) ∧ (y = x) → length_of_chord x y = sqrt 2 :=
by 
  sorry

end chord_length_l191_191310


namespace perimeter_of_equilateral_triangle_with_inscribed_circles_l191_191533

noncomputable def side_length_of_equilateral_triangle_with_inscribed_circles
    (r : ℝ) : ℝ :=
  let a := 2 * sqrt 3 * r + 2 * r in a

theorem perimeter_of_equilateral_triangle_with_inscribed_circles
    (r : ℝ) (h : r = 4) :
  3 * side_length_of_equilateral_triangle_with_inscribed_circles r = 12 * sqrt 3 + 48 :=
by
  rw [side_length_of_equilateral_triangle_with_inscribed_circles, h]
  sorry

end perimeter_of_equilateral_triangle_with_inscribed_circles_l191_191533


namespace pipe_fill_time_without_leakage_l191_191227

theorem pipe_fill_time_without_leakage (t : ℕ) (h1 : 7 * t * (1/t - 1/70) = 1) : t = 60 :=
by
  sorry

end pipe_fill_time_without_leakage_l191_191227


namespace volume_of_extended_set_correct_l191_191285

noncomputable def rectangular_parallelepiped : ℝ³ := ⟨2, 3, 4⟩

noncomputable def volume_of_extended_set (length width height : ℝ) : ℝ :=
  let V_box := length * width * height
  let V_z := 2 * (length * width * 2)
  let V_y := 2 * (length * height * 2)
  let V_x := 2 * (width * height * 2)
  let prisms_2 := 4 * (2 * 2)
  let prisms_3 := 4 * (2 * 3)
  let prisms_4 := 4 * (2 * 4)
  V_box + V_z + V_y + V_x + prisms_2 + prisms_3 + prisms_4

theorem volume_of_extended_set_correct :
  volume_of_extended_set 2 3 4 = 304 := by
  sorry

end volume_of_extended_set_correct_l191_191285


namespace intersection_of_sets_l191_191792

theorem intersection_of_sets :
  let A := {1, 2}
  let B := {x : ℝ | x^2 - 3 * x + 2 = 0}
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_sets_l191_191792


namespace problem_statement_l191_191070

variable (a : ℝ)

def P : Prop := -1 < a ∧ a < 1

def Q : Prop := ∃ (x y : ℝ), x * y < 0 ∧ x^2 + (a - 2) * x + (2 * a - 8) = 0 ∧ y^2 + (a - 2) * y + (2 * a - 8) = 0

theorem problem_statement : ¬ (P ∧ Q) ∧ ¬ (Q → P) :=
by
  sorry

end problem_statement_l191_191070


namespace odd_function_property_l191_191763

-- Define the property of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the proof problem
theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) : ∀ x : ℝ, f x + f (-x) = 0 :=
by
  intros x
  have hx := h x
  rw [hx]
  simp
  sorry

end odd_function_property_l191_191763


namespace polar_to_rect_eq_and_locus_no_common_points_l191_191857

theorem polar_to_rect_eq_and_locus_no_common_points :
  let C := { (x, y) | (x - Real.sqrt 2)^2 + y^2 = 2 } in
  let C1 := { (x, y) | ∃ θ, x = 3 - Real.sqrt 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ } in
  (∀ (ρ θ : Real), ρ = 2 * Real.sqrt 2 * Real.cos θ → (Real.sqrt (ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos θ) - Real.sqrt 2)^2 + (Real.sin θ)^2 = 2 )
  ∧ ¬∃ (p : Real × Real), p ∈ C ∧ p ∈ C1 :=
by
  sorry

end polar_to_rect_eq_and_locus_no_common_points_l191_191857


namespace emus_per_pen_l191_191940

noncomputable def number_of_pens : ℕ := 4
noncomputable def eggs_per_day_per_female_emu : ℕ := 1
noncomputable def fraction_of_female_emus : ℚ := 1 / 2
noncomputable def eggs_per_week : ℕ := 84
noncomputable def days_per_week : ℕ := 7

theorem emus_per_pen :
  let female_emus := eggs_per_week / days_per_week in
  let total_emus := 2 * female_emus in
  let emus_per_pen := total_emus / number_of_pens in
  emus_per_pen = 6 :=
by
  sorry

end emus_per_pen_l191_191940


namespace distinct_factors_count_l191_191385

theorem distinct_factors_count (a b c : ℕ) (h1 : a = 4) (h2 : b = 3) (h3 : c = 2) :
  (finset.card ((finset.range (a + 1)).product 
                ((finset.range (b + 1)).product 
                (finset.range (c + 1))) : nat)) = 60 :=
sorry

end distinct_factors_count_l191_191385


namespace fourth_square_area_l191_191541

theorem fourth_square_area (PQ QR RS PS : ℝ) 
  (hPQ : PQ^2 = 25) 
  (hQR : QR^2 = 64)
  (hRS : RS^2 = 49) 
  (h_right_ΔPQR : PQ^2 + QR^2 = PR^2)
  (h_right_ΔPRS : PR^2 + RS^2 = PS^2) :
  PS^2 = 138 :=
by
  -- translate the steps involved in solution using pythagorean theorem
  have hPR : PR^2 = 25 + 64, from hPQ.symm ▸ hQR.symm ▸ h_right_ΔPQR,
  rw [←hPQ, ←hQR] at hPR,
  have hPReq : PR^2 = 89, by linarith [hPQ, hQR],
  have hPS : PS^2 = 49 + PR^2, from hRS.symm ▸ hPR.symm ▸ h_right_ΔPRS,
  rw [←hRS] at hPS,
  rw [hPReq] at hPS,
  exact hPS.symm

-- note: we are including sorry for missing proof parts

end fourth_square_area_l191_191541


namespace all_terms_perfect_squares_l191_191345

noncomputable def sequence := ℕ → ℤ

axiom rec_condition (a : sequence) (n : ℕ) (h : n ≥ 2) : a (n + 1) = 3 * a n - 3 * a (n - 1) + a (n - 2)

axiom init_condition (a : sequence) : 2 * a 1 = a 0 + a 2 - 2

axiom perfect_square_condition (a : sequence) : 
  ∀ m : ℕ, ∃ k : ℕ, ∀ i : ℕ, i < m → ∃ x : ℕ, a (k + i) = x * x

theorem all_terms_perfect_squares (a : sequence) : ∀ n : ℕ, ∃ x : ℕ, a n = x * x := 
by
  sorry

end all_terms_perfect_squares_l191_191345


namespace incircle_in_tangent_l191_191040

noncomputable theory
open_locale classical

variables {A B C K L M N P Q : Type*} [incircle_of_eq_isosceles_triangle : Type*]
variables [isosceles_triangle : Π {x y z : Type*}, incircle_of_eq_isosceles_triangle → (x = y) → (x = z) → Prop]
variables [tangent_points : Π {x y z k l m : Type*}, incircle_of_eq_isosceles_triangle → (x = y) → (y = z) → (z = k) → (k = l) → (l = m) → Prop]

theorem incircle_in_tangent {incircle_of_eq_isosceles_triangle}
(iso : isosceles_triangle incircle_of_eq_isosceles_triangle)
(tang : tangent_points incircle_of_eq_isosceles_triangle)
(N_intersection : ∃ (N : Type*), is_intersection (line OL) (line KM) N)
(Q_intersection : ∃ (Q : Type*), is_intersection (line BN) (line CA) Q)
(P_foot_perpendicular : ∃ (P : Type*), is_foot_perpendicular (line BQ) A P)
(eq_ratio_BP_AP_2PQ : BP = AP + 2 * PQ) :
  (ratio_AB_BC = real.sqrt 2 / 2 ∨ ratio_AB_BC = real.sqrt 10 / 2) :=
sorry

end incircle_in_tangent_l191_191040


namespace polar_to_rectangular_eq_no_common_points_l191_191851

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_rectangular_eq (ρ θ x y: ℝ) (h1: ρ = 2 * Real.sqrt 2 * Real.cos θ) 
(h2: polar_to_rectangular ρ θ = (x, y)) : (x - Real.sqrt 2) ^ 2 + y ^ 2 = 2 :=
by
  sorry

theorem no_common_points (x y θ : ℝ) (ρ : ℝ := 2 * Real.sqrt 2 * Real.cos θ)
(A := (1, 0) : ℝ × ℝ)
(M := polar_to_rectangular ρ θ)
(P := (sqrt 2 * (M.1 - 1) + 1, sqrt 2 * M.2)) : 
((P.1 - (3 - Real.sqrt 2)) ^ 2 + P.2 ^ 2 = 4) → 
¬ ((P.1 - Real.sqrt 2) ^ 2 + P.2 ^ 2 = 2) :=
by
  intro H1 H2
  sorry

end polar_to_rectangular_eq_no_common_points_l191_191851


namespace pq_elements_example_l191_191948

open Set

def pq_elements_count (P Q : Set ℝ) : ℕ :=
  (P * Q).toFinset.card

theorem pq_elements_example : 
  let P := {-1, 0, 1}
  let Q := {-2, 2}
  pq_elements_count P Q = 3 := 
by
  let P : Set ℝ := {-1, 0, 1}
  let Q : Set ℝ := {-2, 2}
  -- Here we define "P * Q"
  have PQ_def : P * Q = {z | ∃ a b, a ∈ P ∧ b ∈ Q ∧ z = a * b} := rfl
  -- Converting the set to a finite set and counting elements
  let PQ_finset : Finset ℝ := (P * Q).toFinset
  have PQ_finset_card : PQ_finset.card = 3 := by
    sorry

end pq_elements_example_l191_191948


namespace year2018_is_Wu_Xu_l191_191113

def heavenlyStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
def earthlyBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

def sexagenaryCycle : List (String × String) := List.zip heavenlyStems.cycle12 earthlyBranches.cycle10

def year2016 : (String × String) := ("Bing", "Shen")

def yearInSexagenaryCycle : ℕ → (String × String)
| n => sexagenaryCycle.get! (n % 60)

theorem year2018_is_Wu_Xu : yearInSexagenaryCycle (2018 - 1984 + 60) = ("Wu", "Xu") := sorry

end year2018_is_Wu_Xu_l191_191113


namespace line_through_P_with_equal_intercepts_circle_tangent_to_l_axis_l191_191759

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the point P
def P : (ℝ × ℝ) := (1, 2)

-- Define the criteria for equal intercepts
def equal_intercepts (a : ℝ) : Prop := (0, a) = (a, 0)

-- Define the first theorem
theorem line_through_P_with_equal_intercepts :
  (∃ (a : ℝ), equal_intercepts a ∧ a ≠ 0 ∧ ∀ (x y : ℝ), (x + y = 3 → P = (x, y)) ∨ (a = 0 → 3 * x - 2 * y = 0)) :=
sorry

-- Define the point A (intersection with x-axis) and point B (intersection with y-axis)
def A : (ℝ × ℝ) := (4, 0)
def B : (ℝ × ℝ) := (0, 3)

-- Define the criterion for the circle tangent to a line and axes
def tangent_circle (r : ℝ) (center : ℝ × ℝ) : Prop :=
  let (h, k) := center in
  (r = 1) ∧ ((h - 1)^2 + (k - 1)^2 = 1) ∧
    (∀ (x y : ℝ), line_l x y → (x-h)^2 + (y-k)^2 = r^2) ∧
    center = (1, 1)

-- Define the second theorem
theorem circle_tangent_to_l_axis :
  (∃ r c, tangent_circle r c) :=
sorry

end line_through_P_with_equal_intercepts_circle_tangent_to_l_axis_l191_191759


namespace angle_PQR_30_l191_191425

-- Define the geometrical setup and conditions for the trapezium
variables {P Q R S : Type} [line P Q] [line Q R] [line R S] [line S P]
variables {PS PQ SR : ℝ} (h_parallel : is_parallel PQ SR)
variables (h_RSP_120 : ∠ R S P = 120) (h_PS_SR : PS = SR) (h_ratio : PS = (1 / 3) * PQ)

theorem angle_PQR_30 (h_parallel : is_parallel PQ SR)
  (h_RSP_120 : ∠ R S P = 120)
  (h_PS_SR : PS = SR)
  (h_ratio : PS = (1 / 3) * PQ) :
  ∠ P Q R = 30 :=
  sorry

end angle_PQR_30_l191_191425


namespace sum_floor_expression_l191_191440

theorem sum_floor_expression (p : ℕ) (h_prime : Nat.Prime p) (h_form : ∃ k : ℕ, p = 4 * k + 1) :
  ∑ i in Finset.range p \ {0}, (Int.floor ((2 * i ^ 2 : ℤ) / p) - 2 * Int.floor ((i ^ 2 : ℤ) / p)) = (p - 1) / 2 := 
sorry

end sum_floor_expression_l191_191440


namespace count_sums_of_cubes_divisible_by_five_lt_1000_l191_191808

theorem count_sums_of_cubes_divisible_by_five_lt_1000 :
  let cubes := { k | ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ k = a^3 + b^3 }
  in (cubes.filter (λ x, x < 1000 ∧ x % 5 = 0)).card = 17 := by 
  sorry

end count_sums_of_cubes_divisible_by_five_lt_1000_l191_191808


namespace pizza_slices_left_per_person_l191_191100

def total_slices (small: Nat) (large: Nat) : Nat := small + large

def total_eaten (phil: Nat) (andre: Nat) : Nat := phil + andre

def slices_left (total: Nat) (eaten: Nat) : Nat := total - eaten

def pieces_per_person (left: Nat) (people: Nat) : Nat := left / people

theorem pizza_slices_left_per_person :
  ∀ (small large phil andre people: Nat),
  small = 8 → large = 14 → phil = 9 → andre = 9 → people = 2 →
  pieces_per_person (slices_left (total_slices small large) (total_eaten phil andre)) people = 2 :=
by
  intros small large phil andre people h_small h_large h_phil h_andre h_people
  rw [h_small, h_large, h_phil, h_andre, h_people]
  /-
  Here we conclude the proof.
  -/
  sorry

end pizza_slices_left_per_person_l191_191100


namespace number_of_valid_schedules_l191_191610

-- Definitions from conditions
def members : Finset (Fin 10) := Finset.univ
def days : Finset (Fin 5) := Finset.univ

def isValidSchedule (sched : Array (Finset (Fin 10)) 5) : Prop :=
  (sched 0).card = 2 ∧ (sched 1).card = 2 ∧ (sched 2).card = 2 ∧ (sched 3).card = 2 ∧ (sched 4).card = 2 ∧
  (∃ i, {0, 1, 2, 3, 4}.mem i ∧ A ∈ sched i ∧ B ∈ sched i) ∧
  (¬ (∃ i, {0, 1, 2, 3, 4}.mem i ∧ C ∈ sched i ∧ D ∈ sched i))

-- The proof statement
theorem number_of_valid_schedules : 
  ∑ sched in {sched : Array (Finset (Fin 10)) 5 | isValidSchedule sched}.toFinset, 
  1 = 5400 :=
sorry

end number_of_valid_schedules_l191_191610


namespace age_of_other_replaced_man_l191_191117

variable (A B C : ℕ)
variable (B_new1 B_new2 : ℕ)
variable (avg_old avg_new : ℕ)

theorem age_of_other_replaced_man (hB : B = 23) 
    (h_avg_new : (B_new1 + B_new2) / 2 = 25)
    (h_avg_inc : (A + B_new1 + B_new2) / 3 > (A + B + C) / 3) : 
    C = 26 := 
  sorry

end age_of_other_replaced_man_l191_191117


namespace max_min_difference_l191_191966

noncomputable def f (x : ℝ) (n : ℕ+) : ℝ := (x^2 + n) / (x^2 + x + 1)

theorem max_min_difference (n : ℕ+) :
  let an := sup {f x n | x : ℝ}
  let bn := inf {f x n | x : ℝ}
  an - bn = (4 / 3) * real.sqrt (n^2 - n + 1) := by
  sorry

end max_min_difference_l191_191966


namespace quadrilateral_area_is_33_l191_191623

-- Definitions for the points and their coordinates
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 4, y := 0}
def B : Point := {x := 0, y := 12}
def C : Point := {x := 10, y := 0}
def E : Point := {x := 3, y := 3}

-- Define the quadrilateral area computation
noncomputable def areaQuadrilateral (O B E C : Point) : ℝ :=
  let triangle_area (p1 p2 p3 : Point) :=
    abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2
  triangle_area O B E + triangle_area O E C

-- Statement to prove
theorem quadrilateral_area_is_33 : areaQuadrilateral {x := 0, y := 0} B E C = 33 := by
  sorry

end quadrilateral_area_is_33_l191_191623


namespace probability_letter_in_mathematics_l191_191006

theorem probability_letter_in_mathematics :
  let unique_letters := 8
  let total_letters := 26
  (unique_letters.to_rat / total_letters.to_rat).num = 4 ∧
  (unique_letters.to_rat / total_letters.to_rat).denom = 13 :=
by
  -- Here we are letting Lean know about the definition of unique_letters and total_letters,
  -- and we assert that the simplified fraction is 4/13.
  sorry

end probability_letter_in_mathematics_l191_191006


namespace vertex_partition_half_neighbors_l191_191057

-- Let G be a graph with vertex set V and edge set E.
variables {V : Type*} [fintype V] (G : simple_graph V)

-- Proof (no proof steps implemented, just the statement)
theorem vertex_partition_half_neighbors (G : simple_graph V) :
  ∃ (A B : set V), 
    (A ∩ B = ∅) ∧ (A ∪ B = set.univ) ∧ 
    (∀ v ∈ V, (1 / 2 : ℝ) ≤ (G.neighbor_finset v ∩ B).card / (G.neighbor_finset v).card) :=
  sorry

end vertex_partition_half_neighbors_l191_191057


namespace number_of_girl_students_l191_191168

theorem number_of_girl_students (total_third_graders : ℕ) (boy_students : ℕ) (girl_students : ℕ) 
  (h1 : total_third_graders = 123) (h2 : boy_students = 66) (h3 : total_third_graders = boy_students + girl_students) :
  girl_students = 57 :=
by
  sorry

end number_of_girl_students_l191_191168


namespace boats_arrival_interval_l191_191968

-- Set up the givens
def speed_losyash := 4 -- km/h
def launch_interval := 0.5 -- hours
def speed_boat := 10 -- km/h

-- Define the distance Losyash walks every launch interval
def distance_losyash := speed_losyash * launch_interval -- 2 km in this case

-- Define the relative speed of the boats with respect to Losyash
def relative_speed_boat := speed_boat - speed_losyash -- 6 km/h in this case

-- Define the distance between successive boats
def distance_between_boats := relative_speed_boat * launch_interval -- 3 km in this case

-- Define the time interval needed for each boat to travel the distance between successive launches
def time_interval_boats := distance_between_boats / speed_boat -- 0.3 hours or 18 minutes in this case

-- Convert hours to minutes to match the answer provided in the solution
def time_interval_boats_minutes := time_interval_boats * 60 -- 18 minutes

-- Statement to prove
theorem boats_arrival_interval : time_interval_boats_minutes = 18 :=
by
  sorry

end boats_arrival_interval_l191_191968


namespace principal_amount_l191_191582

theorem principal_amount (SI P R T : ℝ) 
  (h1 : R = 12) (h2 : T = 3) (h3 : SI = 3600) : 
  SI = P * R * T / 100 → P = 10000 :=
by
  intros h
  sorry

end principal_amount_l191_191582


namespace does_not_pass_first_quadrant_l191_191506

def linear_function (x : ℝ) : ℝ := -3 * x - 2

def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem does_not_pass_first_quadrant : ∀ (x : ℝ), ¬ in_first_quadrant x (linear_function x) := 
sorry

end does_not_pass_first_quadrant_l191_191506


namespace friends_for_picnic_only_l191_191208

theorem friends_for_picnic_only (M MP MG G PG A P : ℕ) 
(h1 : M + MP + MG + A = 10)
(h2 : G + MG + A = 5)
(h3 : MP = 4)
(h4 : MG = 2)
(h5 : PG = 0)
(h6 : A = 2)
(h7 : M + P + G + MP + MG + PG + A = 31) : 
    P = 20 := by {
  sorry
}

end friends_for_picnic_only_l191_191208


namespace sequence_bound_l191_191342

theorem sequence_bound (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_condition : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
sorry

end sequence_bound_l191_191342


namespace correct_calculation_A_l191_191578

theorem correct_calculation_A :
  (√15 / √5) = √3 :=
by
  -- proof to be provided
  sorry
 
end correct_calculation_A_l191_191578


namespace C_and_C1_no_common_points_l191_191846

noncomputable def C_equation : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - real.sqrt 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (1, 0)

noncomputable def M_on_C (θ : ℝ) : ℝ × ℝ :=
  let x := 2 * real.sqrt 2 * real.cos θ * real.cos θ in
  let y := 2 * real.sqrt 2 * real.sin θ * real.sin θ in
  (x, y)

noncomputable def P_on_locus (x_M y_M : ℝ) : ℝ × ℝ :=
  let x := real.sqrt 2 * (x_M - point_A.1) + point_A.1 in
  let y := real.sqrt 2 * y_M in
  (x, y)

noncomputable def Locus_P_equation (θ : ℝ) : ℝ × ℝ :=
  let x := 3 - real.sqrt 2 + 2 * real.cos θ in
  let y := 2 * real.sin θ in
  (x, y)

def has_common_points (C1 C2 : set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C1 ∧ p ∈ C2

theorem C_and_C1_no_common_points :
  ¬ has_common_points
    {p | C_equation p}
    {p | ∃ θ, P_on_locus (M_on_C θ).1 (M_on_C θ).2 = p} :=
sorry

end C_and_C1_no_common_points_l191_191846


namespace concyclic_points_l191_191648

noncomputable section

-- Define the necessary elements and assumptions based on the conditions.
variables {A B C E F T P : Type*} [Preorder A] [Preorder B] [Preorder C] [Preorder E] [Preorder F] [Preorder T] [Preorder P]
variables (AB AC BF CE BC : ℝ)
variables (I_B I_C K : Type*) [Preorder I_B] [Preorder I_C] [Preorder K]

-- Assume all conditions
variables (ABC_triangle : ∀ A B C : Type*, IsTriangle A B C)
  (acute_triangle : AB > AC)
  (points_on_sides : IsPointOnSide E AC ∧ IsPointOnSide F AB)
  (length_condition : BF + CE = BC)
  (excenters : Excenter I_B B ∧ Excenter I_C C)
  (intersection_T : IntersectAt EI_C FI_B T)
  (midpoint_K : MidPoint K (Arc (CircumCircle A B C) BAC))
  (second_intersection_P : SecondPointIntersection KT (CircumCircle A B C) P)

-- The goal is to prove that the four points are concyclic
theorem concyclic_points : ConcyclicPoints T F P E := sorry

end concyclic_points_l191_191648


namespace eccentricity_of_ellipse_l191_191362

noncomputable def ellipse_eccentricity (a b : ℝ) (h_a_b : a > b > 0) (min_PQ max_PQ : ℝ) 
(h_min : min_PQ = 1) (h_max : max_PQ = 9) : ℝ :=
let c := Real.sqrt (a^2 - b^2) in c / a

theorem eccentricity_of_ellipse : 
  ∀ (a b : ℝ) (h_a_b : a > b > 0) (min_PQ max_PQ : ℝ), 
    min_PQ = 1 → max_PQ = 9 → 
    ellipse_eccentricity a b h_a_b min_PQ max_PQ = 4 / 5 :=
begin 
  intros a b h_a_b min_PQ max_PQ h_min h_max,
  have h1 : a - Real.sqrt (a^2 - b^2) = 1, from sorry,
  have h2 : a + Real.sqrt (a^2 - b^2) = 9, from sorry,
  sorry
end

end eccentricity_of_ellipse_l191_191362


namespace proof_problem_l191_191909

noncomputable def polar_to_rectangular : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x - sqrt 2) ^ 2 + y ^ 2 = 2) ↔ (∀ θ, rho = 2 * sqrt 2 * Real.cos θ → (x ^ 2 + y ^ 2 = rho ^ 2) ∧ (x = rho * Real.cos θ)) sorry

noncomputable def check_common_points : Prop :=
  ∃ (x y : ℝ), (∀ θ, (x = 3 - sqrt 2 + 2 * Real.cos θ) ∧ (y = 2 * Real.sin θ)) ∧
              ∀ (M_P C_P : ℝ), C_P = 2 ∧ |(3 - sqrt 2) - sqrt 2| = 3 - 2 * sqrt 2 < 2 - sqrt 2 :=
              ¬ ∃ p, (p ∈ C) ∧ (p ∈ C_1) sorry

theorem proof_problem : polar_to_rectangular ∧ check_common_points := by
  sorry

end proof_problem_l191_191909


namespace fireflies_remaining_l191_191471

theorem fireflies_remaining
  (initial_fireflies : ℕ)
  (fireflies_joined : ℕ)
  (fireflies_flew_away : ℕ)
  (h_initial : initial_fireflies = 3)
  (h_joined : fireflies_joined = 12 - 4)
  (h_flew_away : fireflies_flew_away = 2)
  : initial_fireflies + fireflies_joined - fireflies_flew_away = 9 := by
  sorry

end fireflies_remaining_l191_191471


namespace smallest_x_with_18_factors_and_factors_18_24_l191_191510

theorem smallest_x_with_18_factors_and_factors_18_24 :
  ∃ (x : ℕ), (∃ (a b : ℕ), x = 2^a * 3^b ∧ 18 ∣ x ∧ 24 ∣ x ∧ (a + 1) * (b + 1) = 18) ∧
    (∀ y, (∃ (c d : ℕ), y = 2^c * 3^d ∧ 18 ∣ y ∧ 24 ∣ y ∧ (c + 1) * (d + 1) = 18) → x ≤ y) :=
by
  sorry

end smallest_x_with_18_factors_and_factors_18_24_l191_191510


namespace gcf_of_32_and_12_l191_191540

theorem gcf_of_32_and_12 : ∀ n : ℕ, LCM n 12 = 48 → n = 32 → GCD n 12 = 8 :=
by
  intros n h_lcm h_n
  sorry

end gcf_of_32_and_12_l191_191540


namespace sum_numerator_perfect_square_l191_191284

theorem sum_numerator_perfect_square :
  ∀ (x : ℕ → ℚ), 
  x 1 = (4 / 3 : ℚ) ∧ (∀ n ≥ 1, x (n + 1) = (x n ^ 2) / (x n ^ 2 - x n + 1)) →
  ∀ n : ℕ, ∃ a : ℤ, S n = ∑ k in finset.range n, x k ∧ numerator (S n) = a^2 :=
by sorry

end sum_numerator_perfect_square_l191_191284


namespace value_of_c7_l191_191465

theorem value_of_c7 
  (a : ℕ → ℕ)
  (b : ℕ → ℕ)
  (c : ℕ → ℕ)
  (h1 : ∀ n, a n = n)
  (h2 : ∀ n, b n = 2^(n-1))
  (h3 : ∀ n, c n = a n * b n) :
  c 7 = 448 :=
by
  sorry

end value_of_c7_l191_191465


namespace sufficient_not_necessary_not_necessary_example_1_not_necessary_example_2_sufficient_but_not_necessary_l191_191069

variable {x y : ℝ}

theorem sufficient_not_necessary (h₁ : x ≥ 1) (h₂ : y ≥ 2) : x + y ≥ 3 :=
by linarith

theorem not_necessary_example_1 : 0 + 4 ≥ 3 :=
by linarith

theorem not_necessary_example_2 : ¬(0 ≥ 1) ∧ (4 ≥ 2) :=
by simp
# 1 is not greater than or equal to 0 but 4 is greater than or equal to 2

theorem sufficient_but_not_necessary : (∀ x y, (x ≥ 1 ∧ y ≥ 2) → (x + y ≥ 3)) ∧ ¬(∀ x y, (x + y ≥ 3) → (x ≥ 1 ∧ y ≥ 2)) :=
begin
  split,
  { intros x y h,
    cases h with hx hy,
    linarith },
  { intro h,
    have ex := h 0 4,
    simp at ex,
    contradiction }
end

end sufficient_not_necessary_not_necessary_example_1_not_necessary_example_2_sufficient_but_not_necessary_l191_191069


namespace C_and_C1_no_common_points_l191_191845

noncomputable def C_equation : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - real.sqrt 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (1, 0)

noncomputable def M_on_C (θ : ℝ) : ℝ × ℝ :=
  let x := 2 * real.sqrt 2 * real.cos θ * real.cos θ in
  let y := 2 * real.sqrt 2 * real.sin θ * real.sin θ in
  (x, y)

noncomputable def P_on_locus (x_M y_M : ℝ) : ℝ × ℝ :=
  let x := real.sqrt 2 * (x_M - point_A.1) + point_A.1 in
  let y := real.sqrt 2 * y_M in
  (x, y)

noncomputable def Locus_P_equation (θ : ℝ) : ℝ × ℝ :=
  let x := 3 - real.sqrt 2 + 2 * real.cos θ in
  let y := 2 * real.sin θ in
  (x, y)

def has_common_points (C1 C2 : set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C1 ∧ p ∈ C2

theorem C_and_C1_no_common_points :
  ¬ has_common_points
    {p | C_equation p}
    {p | ∃ θ, P_on_locus (M_on_C θ).1 (M_on_C θ).2 = p} :=
sorry

end C_and_C1_no_common_points_l191_191845


namespace final_quantity_of_pure_milk_l191_191639

noncomputable def initial_volume := 75
noncomputable def removed_volume := 9
noncomputable def initial_milk := initial_volume
noncomputable def first_dilution_milk := initial_milk - removed_volume
noncomputable def final_milk_fraction := (first_dilution_milk / initial_volume) * removed_volume
noncomputable def final_milk := first_dilution_milk - final_milk_fraction

theorem final_quantity_of_pure_milk :
    final_milk = 58.08 :=
by
    sorry

end final_quantity_of_pure_milk_l191_191639


namespace C_and_C1_no_common_points_l191_191843

noncomputable def C_equation : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - real.sqrt 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (1, 0)

noncomputable def M_on_C (θ : ℝ) : ℝ × ℝ :=
  let x := 2 * real.sqrt 2 * real.cos θ * real.cos θ in
  let y := 2 * real.sqrt 2 * real.sin θ * real.sin θ in
  (x, y)

noncomputable def P_on_locus (x_M y_M : ℝ) : ℝ × ℝ :=
  let x := real.sqrt 2 * (x_M - point_A.1) + point_A.1 in
  let y := real.sqrt 2 * y_M in
  (x, y)

noncomputable def Locus_P_equation (θ : ℝ) : ℝ × ℝ :=
  let x := 3 - real.sqrt 2 + 2 * real.cos θ in
  let y := 2 * real.sin θ in
  (x, y)

def has_common_points (C1 C2 : set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C1 ∧ p ∈ C2

theorem C_and_C1_no_common_points :
  ¬ has_common_points
    {p | C_equation p}
    {p | ∃ θ, P_on_locus (M_on_C θ).1 (M_on_C θ).2 = p} :=
sorry

end C_and_C1_no_common_points_l191_191843


namespace find_cos_A_l191_191414

variables (α x y : ℝ)
variables (α_eq : α = α) (perimeterEq : 150 + 210 + x + y = 640) (AD_ne_BC : x ≠ y)
variables (law_cosine_eq : (x^2 + 22500 - 300 * x * real.cos α = y^2 + 44100 - 420 * y * real.cos α))

theorem find_cos_A (h1: α_eq) (h2: perimeterEq) (h3: AD_ne_BC) (h4: law_cosine_eq) : real.cos α = (some_expression) / (some_denominator) :=
by 
  sorry

end find_cos_A_l191_191414


namespace mabel_tomatoes_l191_191975

theorem mabel_tomatoes :
  ∃ (t1 t2 t3 t4 : ℕ), 
    t1 = 8 ∧ 
    t2 = t1 + 4 ∧ 
    t3 = 3 * (t1 + t2) ∧ 
    t4 = 3 * (t1 + t2) ∧ 
    (t1 + t2 + t3 + t4) = 140 :=
by
  sorry

end mabel_tomatoes_l191_191975


namespace telescoping_product_l191_191657

theorem telescoping_product :
  (∏ k in Finset.range (100 - 2) + 3, (1 - (1 / k))) = (1 / 50) := 
by
  sorry

end telescoping_product_l191_191657


namespace unique_function_satisfying_condition_l191_191074

-- Define the function space and the set X of real numbers greater than 1.
def real_greater_than_one (x : ℝ) : Prop := x > 1
def fun_space (f : ℝ → ℝ) : Prop := ∀ {x y : ℝ}, real_greater_than_one x → real_greater_than_one y → ∀ {a b : ℝ}, 0 < a → 0 < b → f(mul_pow x a y b) ≤ (f x)^(1/(4 * a)) * (f y)^(1/(4 * b))

-- Define the existence of a constant k and the function f as specified.
noncomputable def k : ℝ := sorry  -- k is some constant
noncomputable def f (x : ℝ) : ℝ := k^(1/real.log x)

-- Ensure f is in the specified function space
def valid_function_in_space : Prop := 
  ∀ (x y : ℝ) (hx : real_greater_than_one x) (hy : real_greater_than_one y) (a b : ℝ) (ha : 0 < a) (hb : 0 < b), 
     f(x^a * y^b) ≤ (f x)^(1/(4*a)) * (f y)^(1/(4*b))

theorem unique_function_satisfying_condition :
  valid_function_in_space ∧ (∀ x, real_greater_than_one x → f x = k^(1/real.log x)) :=
begin
  split,
  {
    intros x y hx hy a b ha hb,
    sorry,  -- Proof placeholder
  },
  intros x hx,
  exact rfl,
end

end unique_function_satisfying_condition_l191_191074


namespace super_knight_tour_impossible_l191_191094

-- Define the 12x12 chessboard size
def board_size : ℕ := 12

-- Define the super knight move rule from one corner of a 3x4 rectangle to the opposite
def super_knight_move (x y : ℕ) : (ℕ × ℕ) → (ℕ × ℕ) :=
  λ ⟨a, b⟩, if x = 3 ∧ y = 4 then (a + 2, b + 1) else (a, b)  -- Simplification for alternating move

-- Prove that the super knight can't traverse the board as specified
theorem super_knight_tour_impossible 
  (board_size = 12) 
  (super_knight_rule : ∀ (x y : ℕ), (x % 2 = 0 ∧ y % 2 = 1) → ¬ (x, y) ∈ (finset.univ : finset (ℕ × ℕ))) 
  : ∃ (start : ℕ × ℕ), ¬ ∀ (visited : finset (ℕ × ℕ)), visited.card = board_size * board_size → 
  (∃ (move : (ℕ × ℕ) → (ℕ × ℕ)), start ∉ visited ∧ ∀ p, move p ∉ visited) := 
sorry

end super_knight_tour_impossible_l191_191094


namespace polar_to_rectangular_eq_no_common_points_l191_191854

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_rectangular_eq (ρ θ x y: ℝ) (h1: ρ = 2 * Real.sqrt 2 * Real.cos θ) 
(h2: polar_to_rectangular ρ θ = (x, y)) : (x - Real.sqrt 2) ^ 2 + y ^ 2 = 2 :=
by
  sorry

theorem no_common_points (x y θ : ℝ) (ρ : ℝ := 2 * Real.sqrt 2 * Real.cos θ)
(A := (1, 0) : ℝ × ℝ)
(M := polar_to_rectangular ρ θ)
(P := (sqrt 2 * (M.1 - 1) + 1, sqrt 2 * M.2)) : 
((P.1 - (3 - Real.sqrt 2)) ^ 2 + P.2 ^ 2 = 4) → 
¬ ((P.1 - Real.sqrt 2) ^ 2 + P.2 ^ 2 = 2) :=
by
  intro H1 H2
  sorry

end polar_to_rectangular_eq_no_common_points_l191_191854


namespace balls_in_boxes_l191_191581

open Nat

theorem balls_in_boxes (balls boxes : ℕ) (h1 : balls = 6) (h2 : boxes = 4) : 
  (∃ f : Fin 4 → Fin 7, (∀ i, 0 < f i) ∧ (∑ i, f i = 6)) :=
sorry

end balls_in_boxes_l191_191581


namespace abs_2_minus_5_abs_neg2_minus_5_abs_x_minus_3_solutions_for_abs_x_minus_1_eq_3_integer_solutions_sum_distances_minimum_value_sum_distances_l191_191350

-- Distance calculation for specific points.
theorem abs_2_minus_5 : abs (2 - 5) = 3 :=
by sorry

theorem abs_neg2_minus_5 : abs ((-2) - 5) = 7 :=
by sorry

-- Geometric interpretation of absolute value expression.
theorem abs_x_minus_3 (x : ℚ) : abs (x - 3) = abs (x - 3) :=
by sorry

-- Solutions for the equation |x-1| = 3.
theorem solutions_for_abs_x_minus_1_eq_3 (x : ℚ) : abs (x-1) = 3 ↔ x = 4 ∨ x = -2 :=
by sorry

-- Integer solutions for |x-1| + |x+2| = 3.
theorem integer_solutions_sum_distances (x : ℤ) : abs (x-1) + abs (x+2) = 3 ↔ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 :=
by sorry

-- Minimum value of |x+8| + |x-3| + |x-6|.
theorem minimum_value_sum_distances (x : ℚ) : 
  (∃ x, ∀ y, abs (y + 8) + abs (y - 3) + abs (y - 6) ≥ abs (x + 8) + abs (x - 3) + abs (x - 6)) ∧ 
  abs (3 + 8) + abs (3 - 3) + abs (3 - 6) = 14 :=
by sorry

end abs_2_minus_5_abs_neg2_minus_5_abs_x_minus_3_solutions_for_abs_x_minus_1_eq_3_integer_solutions_sum_distances_minimum_value_sum_distances_l191_191350


namespace factorial_divisible_if_composite_l191_191319

open Nat

theorem factorial_divisible_if_composite (m : ℕ) (h_m_gt_one: m > 1) :
  (factorial (m - 1)).mod m = 0 ↔ ¬ prime m := by
  sorry

end factorial_divisible_if_composite_l191_191319


namespace theorem_find_angle_ACB_l191_191017

-- Define the triangle and its respective angles and points
def triangle_and_angles (A B C E : Type) [IsTriangle A B C] : Prop :=
  ∠BAC = 30 ∧ 3 * BE = 2 * EC ∧ ∠EAB = 45

-- Define the angle to be found
def find_angle_ACB : Type :=
  ∀ (A B C E : Type) [IsTriangle A B C], triangle_and_angles A B C E → ∠ACB = 15

theorem theorem_find_angle_ACB : find_angle_ACB :=
by
  sorry

end theorem_find_angle_ACB_l191_191017


namespace intersecting_circles_l191_191538

theorem intersecting_circles (m n : ℝ) (h_intersect : ∃ c1 c2 : ℝ × ℝ, 
  (c1.1 - c1.2 - 2 = 0) ∧ (c2.1 - c2.2 - 2 = 0) ∧
  ∃ r1 r2 : ℝ, (c1.1 - 1)^2 + (c1.2 - 3)^2 = r1^2 ∧ (c2.1 - 1)^2 + (c2.2 - 3)^2 = r2^2 ∧
  (c1.1 - m)^2 + (c1.2 - n)^2 = r1^2 ∧ (c2.1 - m)^2 + (c2.2 - n)^2 = r2^2) :
  m + n = 4 :=
sorry

end intersecting_circles_l191_191538
