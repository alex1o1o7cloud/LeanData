import Data.Rat.Basic
import Mathlib
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Logarithm
import Mathlib.Algebra.Monomial
import Mathlib.Algebra.Parallel
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Polynomial.Degree
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finite
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Perm
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Int.Modular
import Mathlib.Data.List
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Polynomial
import Mathlib.Data.Prime
import Mathlib.Data.ProbabilityTheory.Bayes
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Integration
import Mathlib.LinearAlgebra.FinVec
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.Divisors
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.Independent
import Mathlib.SetTheory.Sets.Basic
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Sorry
import data.real.basic
import data.set.basic
import tactic

namespace rounding_down_2A3_l275_275086

theorem rounding_down_2A3 (A : ℕ) (h : (2 * 100 + A * 10 + 3) // 10 = 280 // 10) : A = 8 :=
by
  sorry

end rounding_down_2A3_l275_275086


namespace problem_statement_l275_275780

def A : ℕ := 9 * 10 * 10 * 5
def B : ℕ := 9 * 10 * 10 * 2 / 3

theorem problem_statement : A + B = 5100 := by
  sorry

end problem_statement_l275_275780


namespace pile_of_sand_cannot_be_removed_in_1000_trips_l275_275856

theorem pile_of_sand_cannot_be_removed_in_1000_trips : 
  ∀ (a : ℕ → ℕ), (∀ k, a k = 8 ^ (nat.log 8 (a k) : ℕ)) → 
  (∀ k, a k % 7 = 1) → 
  20160000 % 7 = 0 → 
  ∑ k in finset.range 1000, a k % 7 = 3 → 
  ∑ k in finset.range 1000, a k ≠ 20160000 :=
by
  sorry

end pile_of_sand_cannot_be_removed_in_1000_trips_l275_275856


namespace distance_Miami_to_NewYork_l275_275945

-- Defining the complex points
def NewYork : ℂ := 0
def SanFrancisco : ℂ := 0 + 3400 * complex.i
def Miami : ℂ := 1020 + 1360 * complex.i

-- The statement to be proved
theorem distance_Miami_to_NewYork : complex.abs (Miami - NewYork) = 1700 :=
by
  sorry

end distance_Miami_to_NewYork_l275_275945


namespace largest_angle_l275_275510

theorem largest_angle (x : ℝ) (h1 : x + (2 * x + 10) = 105) : 
  (let y := 105, z := 180 - (x + (2 * x + 10)) 
  in (y < z ∧ y < (2 * x + 10)) ∨ 
     ((2 * x + 10) < z ∧ (2 * x + 10) < y) → 
     z = 75) :=
sorry

end largest_angle_l275_275510


namespace geometric_bodies_in_cube_l275_275223

theorem geometric_bodies_in_cube (v1 v2 v3 v4 : Point) :
  (¬ Plane v1 v2 v3 ≠ Plane v1 v2 v3 v4 ∧ SpatialQuadrilateral v1 v2 v3 v4) ∧
  (EquilateralTriangleFaceTetrahedron v1 v2 v3 v4) ∧
  (IsoscelesRightEquilateralFaceTetrahedron v1 v2 v3 v4) :=
sorry

end geometric_bodies_in_cube_l275_275223


namespace Aunt_Lucy_gift_correct_l275_275016

def Jade_initial : ℕ := 38
def Julia_initial : ℕ := Jade_initial / 2
def Jack_initial : ℕ := 12
def John_initial : ℕ := 15
def Jane_initial : ℕ := 20

def Aunt_Mary_gift : ℕ := 65
def Aunt_Susan_gift : ℕ := 70

def total_initial : ℕ :=
  Jade_initial + Julia_initial + Jack_initial + John_initial + Jane_initial

def total_after_gifts : ℕ := 225
def total_gifts : ℕ := total_after_gifts - total_initial
def Aunt_Lucy_gift : ℕ := total_gifts - (Aunt_Mary_gift + Aunt_Susan_gift)

theorem Aunt_Lucy_gift_correct :
  Aunt_Lucy_gift = total_after_gifts - total_initial - (Aunt_Mary_gift + Aunt_Susan_gift) := by
  sorry

end Aunt_Lucy_gift_correct_l275_275016


namespace total_votes_is_correct_l275_275390

-- Definitions and theorem statement
theorem total_votes_is_correct (T : ℝ) 
  (votes_for_A : ℝ) 
  (candidate_A_share : ℝ) 
  (valid_vote_fraction : ℝ) 
  (invalid_vote_fraction : ℝ) 
  (votes_for_A_equals: votes_for_A = 380800) 
  (candidate_A_share_equals: candidate_A_share = 0.80) 
  (valid_vote_fraction_equals: valid_vote_fraction = 0.85) 
  (invalid_vote_fraction_equals: invalid_vote_fraction = 0.15) 
  (valid_vote_computed: votes_for_A = candidate_A_share * valid_vote_fraction * T): 
  T = 560000 := 
by 
  sorry

end total_votes_is_correct_l275_275390


namespace max_intersections_of_line_with_circles_l275_275992

noncomputable theory

open Real EuclideanGeometry

-- Definitions based on the conditions
def coplanar_four_circles (C1 C2 C3 C4 : Circle) : Prop :=
  ∃ (P : Plane), C1.center ∈ P ∧ C2.center ∈ P ∧ C3.center ∈ P ∧ C4.center ∈ P

def non_concentric (C1 C2 C3 C4 : Circle) : Prop :=
  ∀ (i j : ℕ), (i ≠ j ∧ i < 4 ∧ j < 4) → (C1.center ≠ C2.center ∧ C1.center ≠ C3.center ∧ 
                                            C1.center ≠ C4.center ∧ 
                                            C2.center ≠ C3.center ∧
                                            C2.center ≠ C4.center ∧
                                            C3.center ≠ C4.center)

-- The main theorem statement
theorem max_intersections_of_line_with_circles :
  ∀ (C1 C2 C3 C4 : Circle), coplanar_four_circles C1 C2 C3 C4 →
  non_concentric C1 C2 C3 C4 →
  ∃ (l : Line), l.intersects C1 ≤ 2 ∧ l.intersects C2 ≤ 2 ∧ 
                l.intersects C3 ≤ 2 ∧ l.intersects C4 ≤ 2 →
                (finsum (λ C, l.intersects C) {C1, C2, C3, C4}) ≤ 8 :=
by
  sorry

end max_intersections_of_line_with_circles_l275_275992


namespace matrix_inverse_pair_l275_275117

open Matrix

variable {α : Type*} [Fintype α] [DecidableEq α]

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -2], ![12, -5]]

theorem matrix_inverse_pair :
  A.mul A = 1 := sorry

end matrix_inverse_pair_l275_275117


namespace find_tangency_point_l275_275707

-- Definitions based on conditions
def is_even (f : ℝ → ℝ) := ∀ x, f(-x) = f(x)
def f (x a: ℝ) := e^x + a * e^(-x)
def f_prime (x a: ℝ) := e^x - a * e^(-x)

-- Theorem to be proven
theorem find_tangency_point
  (a : ℝ) 
  (h_even : is_even (λ x, f x a))
  (h_slope : ∃ x, f_prime x 1 = 3 / 2) : 
  ∃ x, x = ln 2 :=
begin
  sorry
end

end find_tangency_point_l275_275707


namespace correct_multiplier_l275_275911

theorem correct_multiplier
  (x : ℕ)
  (incorrect_multiplier : ℕ := 34)
  (difference : ℕ := 1215)
  (number_to_be_multiplied : ℕ := 135) :
  number_to_be_multiplied * x - number_to_be_multiplied * incorrect_multiplier = difference →
  x = 43 :=
  sorry

end correct_multiplier_l275_275911


namespace total_area_of_figure_with_side_length_3_l275_275576

noncomputable def equilateral_triangle_area (s : ℕ) := (√3 / 4) * s^2

noncomputable def hexagon_area (s : ℕ) := 6 * equilateral_triangle_area s

noncomputable def total_figure_area (s : ℕ) := hexagon_area s + 6 * equilateral_triangle_area s

theorem total_area_of_figure_with_side_length_3 : total_figure_area 3 = 27 * (√3) := 
by 
  sorry

end total_area_of_figure_with_side_length_3_l275_275576


namespace interval_notation_l275_275830

theorem interval_notation {x : ℝ} : { x | x ≤ -1 } = set.Iic (-1) :=
by {
  sorry
}

end interval_notation_l275_275830


namespace mark_spending_l275_275799

theorem mark_spending (initial_money : ℕ) (first_store_half : ℕ) (first_store_additional : ℕ) 
                      (second_store_third : ℕ) (remaining_money : ℕ) (total_spent : ℕ) : 
  initial_money = 180 ∧ 
  first_store_half = 90 ∧ 
  first_store_additional = 14 ∧ 
  total_spent = first_store_half + first_store_additional ∧
  remaining_money = initial_money - total_spent ∧
  second_store_third = 60 ∧ 
  remaining_money - second_store_third = 16 ∧ 
  initial_money - (total_spent + second_store_third + 16) = 0 → 
  remaining_money - second_store_third = 16 :=
by
  intro h
  sorry

end mark_spending_l275_275799


namespace total_children_on_playground_l275_275137

theorem total_children_on_playground (girls boys : ℕ) (hg : girls = 28) (hb : boys = 35) : girls + boys = 63 := by
  rw [hg, hb]
  norm_num

end total_children_on_playground_l275_275137


namespace percentage_problem_l275_275174

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 100) : 1.20 * x = 600 :=
sorry

end percentage_problem_l275_275174


namespace cyclic_quadrilateral_PMID_l275_275425

-- Definitions of the geometrical entities conforming to the conditions
variables {A B C D E F I M P : Type}
variable [incircle_center : has_incircle_center I ABC] -- I is the center of the inscribed circle in triangle ABC
variables [touches_incircle D B C] [touches_incircle E C A] [touches_incircle F A B] -- Incircle touches sides BC, CA, and AB at D, E, and F
variable [midpoint_of M E F] -- M is the midpoint of EF
variable (A D : line) -- The line AD
variables [intersects_incircle A D] (P : point) [P_on_incircle P A I D] -- AD intersects the inscribed circle at P ≠ D

-- Statement of the theorem to be proved
theorem cyclic_quadrilateral_PMID :
  cyclic_quadrilateral P M I D :=
sorry

end cyclic_quadrilateral_PMID_l275_275425


namespace cosine_symmetry_about_x_axis_l275_275396

-- Definitions and Conditions
def f (x : ℝ) : ℝ := Real.cos x
def g (x : ℝ) : ℝ := -Real.cos x

-- Theorem Statement
theorem cosine_symmetry_about_x_axis :
  ∀ x, f x = Real.cos x ∧ g x = -Real.cos x ∧ g x = - f x :=
by
  sorry

end cosine_symmetry_about_x_axis_l275_275396


namespace squares_in_square_l275_275301

theorem squares_in_square :
  ∃ (L : ℝ), (∀ (n : ℕ), n > 0 → ∃ (x y : ℝ), 0 ≤ x ∧ x + 1/n ≤ L ∧ 0 ≤ y ∧ y + 1/n ≤ L ∧ (∀ (m : ℕ), m ≠ n → disjoint (square x y (1/n)) (square (x m) (y m) (1/m)))) ∧ L = 1.5 :=
sorry

end squares_in_square_l275_275301


namespace necessary_but_not_sufficient_l275_275787

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a + b > 4) ↔ (¬ (a > 2 ∧ b > 2)) ∧ ((a > 2 ∧ b > 2) → (a + b > 4)) :=
by
  sorry

end necessary_but_not_sufficient_l275_275787


namespace find_AM_MC_ratio_l275_275796

noncomputable def AM_MC_ratio (k1 k2 : Circle) (A B C D E F M : Point) (MID : Circle -> Point -> Point -> Prop) (TAN : Point -> Point -> Circle -> Prop) (INT : Point -> Point -> Point -> PointRelation) (CONCYCLIC : List Point -> Prop) (bic : Point -> Point -> Segment -> Prop) : Prop :=
  ∀ (h1 : concentric k1 k2),
     (A ∈ k1) → 
     (B ∈ k2) →
     TAN A B k2 →
     INT A B C k1 →
     MID A B D →
     (E ∈ k2) →
     (F ∈ k2) →
     INT A E E k2 →
     INT A F F k2 →
     bic D E M →
     bic C F M →
     (M ∈ AB) →
  ∃ r : Real, r = 5 / 3

theorem find_AM_MC_ratio (k1 k2 : Circle) (A B C D E F M : Point) 
  (MID : Circle -> Point -> Point -> Prop) (TAN : Point -> Point -> Circle -> Prop) (INT : Point -> Point -> Point -> PointRelation) (CONCYCLIC : List Point -> Prop) (bic : Point -> Point -> Segment -> Prop) :
  AM_MC_ratio k1 k2 A B C D E F M MID TAN INT CONCYCLIC bic := by
  sorry

end find_AM_MC_ratio_l275_275796


namespace least_number_subtracted_divisible_l275_275164

theorem least_number_subtracted_divisible (n : ℕ) (m : ℕ) (k : ℕ) : (n = 42739) → (m = 15) → (k = 4) → n % m = k :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end least_number_subtracted_divisible_l275_275164


namespace problem1_problem2_l275_275319

noncomputable theory

-- Given functions
def f (x : ℝ) := x^3 - x
def g (x : ℝ) (a : ℝ) := x^2 + a

-- Problem (1): If x1 = -1, show that a = 3
theorem problem1 (a : ℝ) (h : tangent_line_at f (-1) = tangent_line_at (g (-1)) (f (-1))) :
  a = 3 := sorry

-- Lean theorem to find the range of values for a such that tangent line conditions hold
theorem problem2 (x₁ : ℝ) (a : ℝ) (h : tangent_line_at f x₁ = tangent_line_at g x₁ a) :
  a ≥ -1 := sorry

end problem1_problem2_l275_275319


namespace product_of_roots_l275_275150

theorem product_of_roots : 
  (Real.root 81 4) * (Real.root 27 3) * (Real.sqrt 9) = 27 :=
by
  sorry

end product_of_roots_l275_275150


namespace focus_of_parabola_l275_275978

theorem focus_of_parabola : (∃ p : ℝ × ℝ, p = (-1, 35/12)) :=
by
  sorry

end focus_of_parabola_l275_275978


namespace trapezoid_area_perimeter_l275_275211

structure Trapezoid :=
  (A B C D : Type)
  (AB CD : Type)

theorem trapezoid_area_perimeter
  (A B C D : Type)
  (AB CD : Prop)
  (right_trapezoid : ∃ (B : Type), ⟦ B ≠ 0 ⟧)
  (parallel_AB_CD : ∃ (A B : Type), ⟦ A = B ⟧)
  (perpendicular_diagonals : ∃ (AC BD : Type), ⟦ AC ⊥ BD ⟧)
  (AC_length : ℝ := 12)
  (BD_length : ℝ := 9) :
  (area : ℝ := 54) ∧ (perimeter : ℝ := 30.5) :=
by
  sorry

end trapezoid_area_perimeter_l275_275211


namespace min_max_A_odd_sum_A_odd_sum_A_even_l275_275430

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def A (n : ℕ) : Set ℕ := {x | (2^n < x) ∧ (x < 2^(n+1)) ∧ ∃ m : ℕ, x = 3 * m}

theorem min_max_A_odd {n : ℕ} (h : is_odd n) : 
  ∃ min max : ℕ, min ∈ A n ∧ max ∈ A n ∧ (∀ y ∈ A n, min ≤ y) ∧ (∀ z ∈ A n, z ≤ max) ∧ min = 2^n + 1 ∧ max = 2^(n+1) - 1 :=
sorry

theorem sum_A_odd {n : ℕ} (h : is_odd n) :
  ∑ x in (A n).to_finset, x = 2^(2 * n - 1) + 2^(n - 1) :=
sorry

theorem sum_A_even {n : ℕ} (h : is_even n) :
  ∑ x in (A n).to_finset, x = 2^(2 * n - 1) - 2^(n - 1) :=
sorry

end min_max_A_odd_sum_A_odd_sum_A_even_l275_275430


namespace scientific_notation_l275_275847

theorem scientific_notation (n : ℕ) (h : n = 27000000) : 
  ∃ (m : ℝ) (e : ℤ), n = m * (10 : ℝ) ^ e ∧ m = 2.7 ∧ e = 7 :=
by 
  use 2.7 
  use 7
  sorry

end scientific_notation_l275_275847


namespace find_position_2002_l275_275079

def T (n : ℕ) : ℕ := n * (n + 1) / 2
def a (n : ℕ) : ℕ := T n + 1

theorem find_position_2002 : ∃ row col : ℕ, 1 ≤ row ∧ 1 ≤ col ∧ (a (row - 1) + (col - 1) = 2002 ∧ row = 15 ∧ col = 49) := 
sorry

end find_position_2002_l275_275079


namespace no_positive_integer_solution_l275_275559

theorem no_positive_integer_solution (m n : ℕ) (h : 0 < m) (h1 : 0 < n) : ¬ (5 * m^2 - 6 * m * n + 7 * n^2 = 2006) :=
sorry

end no_positive_integer_solution_l275_275559


namespace total_alternating_sum_formula_l275_275663

-- Define the alternating sum function for non-empty subsets of ℕ
def alternating_sum (s : Finset ℕ) : ℤ :=
  s.val.to_list.enum.map (λ p, if (p.1 % 2 = 0) then (p.2 : ℤ) else -(p.2 : ℤ)).sum

-- Define the total alternating sum for a given n
def total_alternating_sum (n : ℕ) : ℤ :=
  (Finset.powerset (Finset.range (n + 1))).filter (λ s, ¬s.is_empty).sum (λ s, alternating_sum s)

-- Statement to be proven
theorem total_alternating_sum_formula (n : ℕ) : total_alternating_sum n = n * 2^(n - 1) :=
  sorry

end total_alternating_sum_formula_l275_275663


namespace max_cos_sin_volume_of_solid_l275_275778

noncomputable def f (x : ℝ) : ℝ := 
  Real.Analysis.Limit1 (cos x ^ n + sin x ^ n) ^ (1 / n)

theorem max_cos_sin {x : ℝ} (hx : 0 ≤ x ∧ x ≤ π / 2) : 
  f(x) = max (cos x) (sin x) := by
  sorry

theorem volume_of_solid {a b : ℝ} (ha : a = sqrt 2 / 2) (hb : b = 1) :
  2 * π * (∫ (y : ℝ) in ha..hb, Real.arc_cos y * y) + 
  2 * π * (∫ (y : ℝ) in 0..ha, Real.arc_sin y * y) := by
  sorry

end max_cos_sin_volume_of_solid_l275_275778


namespace victor_draw_order_count_l275_275866

-- Definitions based on the problem conditions
def num_piles : ℕ := 3
def num_cards_per_pile : ℕ := 3
def total_cards : ℕ := num_piles * num_cards_per_pile

-- The cardinality of the set of valid sequences where within each pile cards must be drawn in order
def valid_sequences_count : ℕ :=
  Nat.factorial total_cards / (Nat.factorial num_cards_per_pile ^ num_piles)

-- Now we state the problem: proving the valid sequences count is 1680
theorem victor_draw_order_count :
  valid_sequences_count = 1680 :=
by
  sorry

end victor_draw_order_count_l275_275866


namespace number_of_valid_digits_l275_275198

theorem number_of_valid_digits :
  let N := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] in
  let valid_digits := [n | n ∈ N ∧ (640 + n) % 4 = 0] in
  valid_digits.length = 5 :=
by sorry

end number_of_valid_digits_l275_275198


namespace gcd_correct_l275_275525

-- Declare the variables
variables (a b r1 r2 r3 : ℕ)

-- Define the conditions
def condition1 : Prop := (1995 % 228 = 171)
def condition2 : Prop := (228 % 171 = 57)
def condition3 : Prop := (171 % 57 = 0)

-- Define the gcd function
def gcd_property : Prop := Nat.gcd 1995 228 = 57

-- Our main theorem statement
theorem gcd_correct :
  condition1 ∧ condition2 ∧ condition3 → gcd_property :=
begin
  sorry
end

end gcd_correct_l275_275525


namespace single_elimination_tournament_l275_275928

theorem single_elimination_tournament (teams : ℕ) (prelim_games : ℕ) (post_prelim_teams : ℕ) :
  teams = 24 →
  prelim_games = 4 →
  post_prelim_teams = teams - prelim_games →
  post_prelim_teams - 1 + prelim_games = 23 :=
by
  intros
  sorry

end single_elimination_tournament_l275_275928


namespace arctan_sum_gt_pi_h2_l275_275369

theorem arctan_sum_gt_pi_h2 (a b : ℝ) (h1 : a = 2 / 3) (h2 : (a + 1) * (b + 1) = 3) : 
  arctan a + arctan b > π / 2 :=
by
  sorry

end arctan_sum_gt_pi_h2_l275_275369


namespace negation_proposition_l275_275497

theorem negation_proposition:
  (¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) :=
by sorry

end negation_proposition_l275_275497


namespace minimum_value_sum_l275_275791

theorem minimum_value_sum (x : Fin 50 → ℝ)
    (h1 : ∀ i, 0 < x i)
    (h2 : ∑ i, (x i) ^ 2 = 2) :
    ∑ i, (x i) / (1 - (x i) ^ 2) ≥ 3 * Real.sqrt 3 := 
sorry

end minimum_value_sum_l275_275791


namespace stream_speed_l275_275186

variable (B S : ℝ)

def downstream_eq : Prop := B + S = 13
def upstream_eq : Prop := B - S = 5

theorem stream_speed (h1 : downstream_eq B S) (h2 : upstream_eq B S) : S = 4 :=
by
  sorry

end stream_speed_l275_275186


namespace floor_sum_condition_l275_275417

/-- 
Given four real numbers a, b, c, d such that 
  for all positive integers n,
  ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋
we must show that at least one of 
  a + b, a - c, a - d 
is an integer.
-/
theorem floor_sum_condition (a b c d : ℝ)
    (h : ∀ n : ℕ, 0 < n → floor (n * a) + floor (n * b) = floor (n * c) + floor (n * d)) :
    (∃ k : ℤ, a + b = k) ∨ (∃ k : ℤ, a - c = k) ∨ (∃ k : ℤ, a - d = k) := 
  sorry

end floor_sum_condition_l275_275417


namespace collectively_behind_l275_275518

noncomputable def sleep_hours_behind (weeknights weekend nights_ideal: ℕ) : ℕ :=
  let total_sleep := (weeknights * 5) + (weekend * 2)
  let ideal_sleep := nights_ideal * 7
  ideal_sleep - total_sleep

def tom_weeknight := 5
def tom_weekend := 6

def jane_weeknight := 7
def jane_weekend := 9

def mark_weeknight := 6
def mark_weekend := 7

def ideal_night := 8

theorem collectively_behind :
  sleep_hours_behind tom_weeknight tom_weekend ideal_night +
  sleep_hours_behind jane_weeknight jane_weekend ideal_night +
  sleep_hours_behind mark_weeknight mark_weekend ideal_night = 34 :=
by
  sorry

end collectively_behind_l275_275518


namespace number_of_terms_l275_275962

theorem number_of_terms (x y z w : ℕ) :
  ((x + y + z + w) ^ 10 + (x - y - z - w) ^ 10).num_terms = 56 := 
sorry

end number_of_terms_l275_275962


namespace dot_product_eq_zero_l275_275321

-- Definitions in Lean
def point := (ℝ, ℝ)

def A : point := (3, -1)
def B : point := (6, 1)

def line_normal_vector := (2, -3)

def vector_sub (u v : point) : point := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : point) : ℝ := u.1 * v.1 + u.2 * v.2

-- The theorem we want to prove
theorem dot_product_eq_zero : 
  dot_product (vector_sub B A) line_normal_vector = 0 := 
sorry

end dot_product_eq_zero_l275_275321


namespace number_of_arrangements_l275_275748

theorem number_of_arrangements
  (traditional_chinese_paintings : Finset ℕ)
  (oil_paintings : Finset ℕ)
  (ink_painting : Finset ℕ) 
  (h1 : traditional_chinese_paintings.card = 3)
  (h2 : oil_paintings.card = 2) 
  (h3 : ink_painting.card = 1) 
  (h4 : Disjoint traditional_chinese_paintings oil_paintings ∧ Disjoint traditional_chinese_paintings ink_painting ∧ Disjoint oil_paintings ink_painting) :
  ∃ n, n = 24 := by
  sorry

end number_of_arrangements_l275_275748


namespace katya_sold_glasses_l275_275772

-- Definitions based on the conditions specified in the problem
def ricky_sales : ℕ := 9

def tina_sales (K : ℕ) : ℕ := 2 * (K + ricky_sales)

def katya_sales_eq (K : ℕ) : Prop := tina_sales K = K + 26

-- Lean statement to prove Katya sold 8 glasses of lemonade
theorem katya_sold_glasses : ∃ (K : ℕ), katya_sales_eq K ∧ K = 8 :=
by
  sorry

end katya_sold_glasses_l275_275772


namespace area_ratio_S2_S1_l275_275050

theorem area_ratio_S2_S1 :
  let S1 := {(x, y) | log 10 (2 + x^2 + y^2) ≤ 1 + log 10 (x + y)},
      S2 := {(x, y) | log 10 (3 + x^2 + y^2) ≤ 2 + log 10 (x + y)},
      area_S1 := π * (4 * sqrt 3)^2,
      area_S2 := π * sqrt 4997 ^ 2
  in area_S2 / area_S1 = 104 := 
sorry

end area_ratio_S2_S1_l275_275050


namespace domain_of_composite_function_l275_275334

variable {α : Type*}
variable (f : ℝ → α)

-- Given condition: dom(f) = [1, ∞)
def dom_f (x : ℝ) : Prop := x ≥ 1

-- Domain of y = f(x-1) + f(4-x) is [2,3]
theorem domain_of_composite_function :
  (∀ x, dom_f f x ↔ x ≥ 1) →
  (∀ x, (∀ y, y = f(x-1) + f(4-x) → 
    ((2 ≤ x) ∧ (x ≤ 3)))) :=
begin 
  intros h x,
  intros y h1,
  split; 
  linarith,
end

end domain_of_composite_function_l275_275334


namespace start_age_of_planting_new_row_l275_275402

theorem start_age_of_planting_new_row (initial_trees planting_frequency age_doubling total_trees : ℕ) 
(h_init : initial_trees = 2 * 4)
(h_planting_frequency : planting_frequency = 4)
(h_age_doubling : age_doubling = 15)
(h_total_doubling : total_trees = 56) :
  ∃ x : ℕ, 2 * (initial_trees + planting_frequency * (age_doubling - x)) = total_trees ∧ x = 10 :=
begin
  use 10,
  split,
  {
    rw [h_init, h_planting_frequency, h_age_doubling, h_total_doubling],
    linarith,
  },
  {
    exact rfl,
  }
end

end start_age_of_planting_new_row_l275_275402


namespace simplify_factorial_expression_l275_275091

theorem simplify_factorial_expression : (12! : ℚ) / (10! + 3 * 9!) = 1320 / 13 := 
by
  sorry

end simplify_factorial_expression_l275_275091


namespace expenses_categorization_l275_275633

-- Define the sets of expenses and their categories
inductive Expense
| home_internet
| travel
| camera_rental
| domain_payment
| coffee_shop
| loan
| tax
| qualification_courses

open Expense

-- Define the conditions under which expenses can or cannot be economized
def isEconomizable : Expense → Prop
| home_internet        := true
| travel               := true
| camera_rental        := true
| domain_payment       := true
| coffee_shop          := true
| loan                 := false
| tax                  := false
| qualification_courses:= false

-- The main theorem statement
theorem expenses_categorization
  (expenses : list Expense) :
  (∀ e ∈ [home_internet, travel, camera_rental, domain_payment, coffee_shop], isEconomizable e) ∧
  (∀ e ∈ [loan, tax, qualification_courses], ¬ isEconomizable e) :=
by
  intros,
  simp [isEconomizable],
  exact ⟨
    (λ e he, by cases he; exact trivial),
    (λ e he, by cases he; tauto)⟩

end expenses_categorization_l275_275633


namespace power_difference_eq_l275_275688

variable (x : ℝ)

theorem power_difference_eq :
  x - (1 / x) = real.sqrt 3 →
  x^243 - (1 / x^243) = 6 * real.sqrt 3 :=
sorry

end power_difference_eq_l275_275688


namespace number_of_valid_digits_l275_275200

theorem number_of_valid_digits :
  let N := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] in
  let valid_digits := [n | n ∈ N ∧ (640 + n) % 4 = 0] in
  valid_digits.length = 5 :=
by sorry

end number_of_valid_digits_l275_275200


namespace completing_the_square_correct_l275_275538

theorem completing_the_square_correct :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x h
  sorry

end completing_the_square_correct_l275_275538


namespace negative_fraction_is_C_l275_275166

-- conditions as definitions
def option_A := 1 / 2
def option_B := -Real.pi
def option_C := -0.7
def option_D := -3 / 3

-- question translated to a proof problem
theorem negative_fraction_is_C : option_C < 0 ∧ ∃ a b : ℤ, option_C = a / b ∧ b ≠ 0 :=
by sorry

end negative_fraction_is_C_l275_275166


namespace numbers_sum_and_difference_l275_275505

variables (a b : ℝ)

theorem numbers_sum_and_difference (h : a / b = -1) : a + b = 0 ∧ (a - b = 2 * b ∨ a - b = -2 * b) :=
by {
  sorry
}

end numbers_sum_and_difference_l275_275505


namespace ramon_current_age_l275_275771

variable (R : ℕ) (L : ℕ)

theorem ramon_current_age :
  (L = 23) → (R + 20 = 2 * L) → R = 26 :=
by
  intro hL hR
  rw [hL] at hR
  have : R + 20 = 46 := by linarith
  linarith

end ramon_current_age_l275_275771


namespace sequence_formula_l275_275253

theorem sequence_formula (a : ℕ → ℕ) (h₀ : a 1 = 2) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n ^ 2 - n * a n + 1) :
  ∀ n : ℕ, a n = n + 1 :=
by
  sorry

end sequence_formula_l275_275253


namespace trigonometric_identity_correct_l275_275542

theorem trigonometric_identity_correct :
  ∀ (α β : ℝ), (cos (α - β) = cos α * cos β + sin α * sin β) := 
by
  -- Detailed proof skipped
  sorry

end trigonometric_identity_correct_l275_275542


namespace elsa_eva_cannot_meet_l275_275640

-- Definitions
def starting_point_elsa : ℤ × ℤ := (0, 0)
def starting_point_eva : ℤ × ℤ := (0, 1)

def adjacent (p q : ℤ × ℤ) : Prop :=
  (p.1 = q.1 ∧ abs (p.2 - q.2) = 1) ∨ (p.2 = q.2 ∧ abs (p.1 - q.1) = 1)

-- Goal to prove: Elsa and Éva cannot meet
theorem elsa_eva_cannot_meet :
  ¬ ∃ (n : ℕ) (x y : ℤ),
    (starting_point_elsa = (x, y) ∨
     ∃ (m : ℕ), adjacent ((0, 0) + m • 1, (0, 0) + m • 1)) ∧
    (starting_point_eva = (x, y) ∨
     ∃ (m : ℕ), adjacent ((0, 1) + m • 1, (0, 1) + m • 1)) :=
sorry

end elsa_eva_cannot_meet_l275_275640


namespace tank_capacity_l275_275205

-- Definition: tank capacity
variable (x : ℝ)

-- Conditions
def initially_full : ℝ := (2 / 3) * x
def after_usage_full : ℝ := (1 / 3) * x
def usage : ℝ := 15

-- Proof Problem
theorem tank_capacity : initially_full x - usage = after_usage_full x → x = 45 :=
by
  sorry

end tank_capacity_l275_275205


namespace problem_1_problem_2_l275_275314

-- Given functions f(x) = x^3 - x and g(x) = x^2 + a
def f (x : ℝ) : ℝ := x ^ 3 - x
def g (x : ℝ) (a : ℝ) : ℝ := x ^ 2 + a

-- (1) If x1 = -1, prove the value of a is 3 given the tangent line condition.
theorem problem_1 (a : ℝ) : 
  let x1 := -1 in
  let f' (x : ℝ) : ℝ := 3 * x ^ 2 - 1 in
  let g' (x : ℝ) : ℝ := 2 * x in
  f' x1 = 2 →  -- Slope of the tangent line at x1 = -1 for f(x)
  g' 1 = 2 →  -- Slope of the tangent line at x2 = 1 for g(x)
  (f x1 + (2 * (1 + 1)) = g 1 a) → -- The tangent line condition
  a = 3 := 
sorry

-- (2) Find the range of values for a under the tangent line condition.
theorem problem_2 (a : ℝ) : 
  let h (x1 : ℝ) := 9 * x1 ^ 4 - 8 * x1 ^ 3 - 6 * x1 ^ 2 + 1 in
   -- We check that the minimum value is h(1) = -4, which gives a range for a
  (∃ x1, x1 ≠ 1 ∧ a ≥ (-1)) :=
sorry

end problem_1_problem_2_l275_275314


namespace min_value_M_a1_min_value_M_3_l275_275286

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) := x^2 + x + a^2 + a
def g (a : ℝ) (x : ℝ) := x^2 - x + a^2 - a

-- Define M(x) as the maximum of f(x) and g(x)
def M (a : ℝ) (x : ℝ) := max (f a x) (g a x)

-- Problem 1: Prove that the minimum value of M(x) for a = 1 is 7/4
theorem min_value_M_a1 : ∃ x : ℝ, x.f = 1 → M 1 x = 7 / 4 :=
by
  sorry

-- Problem 2: Prove that if the minimum value of M(x) is 3, then a = (sqrt 14 - 1) / 2
theorem min_value_M_3 (a : ℝ) : (∃ x : ℝ, M a x = 3) → a = (Real.sqrt 14 - 1) / 2 :=
by
  sorry

end min_value_M_a1_min_value_M_3_l275_275286


namespace river_depth_l275_275923

noncomputable def depth_of_river (width : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) : ℝ :=
  let flow_rate_m_per_min := flow_rate_kmph * 1000 / 60
  volume_per_minute / (width * flow_rate_m_per_min)

theorem river_depth
  (width : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ)
  (h_width : width = 45)
  (h_flow_rate_kmph : flow_rate_kmph = 4)
  (h_volume_per_minute : volume_per_minute = 6000) :
  depth_of_river width flow_rate_kmph volume_per_minute = 2 :=
by
  rw [h_width, h_flow_rate_kmph, h_volume_per_minute]
  unfold depth_of_river
  have h_flow_rate_m_per_min : 4 * 1000 / 60 = 66.67 := by norm_num
  rw h_flow_rate_m_per_min
  norm_num
  sorry

end river_depth_l275_275923


namespace distance_between_centers_l275_275506

def radius1 : ℝ := 3
def radius2 : ℝ := 2
def externally_tangent (R r P : ℝ) : Prop := P = R + r

theorem distance_between_centers : ∃ P : ℝ, externally_tangent radius1 radius2 P ∧ P = 5 := by
  use 5
  unfold externally_tangent
  apply And.intro
  · sorry

end distance_between_centers_l275_275506


namespace least_five_digit_congruent_to_11_mod_14_l275_275529

theorem least_five_digit_congruent_to_11_mod_14 : ∃ n, 10000 ≤ n ∧ n < 100000 ∧ n % 14 = 11 ∧ ∀ m, (10000 ≤ m ∧ m < n → m % 14 ≠ 11) := 
by
  have h : 10007 % 14 = 11 := by norm_num
  use 10007
  constructor
  { exact le_refl 10007 }
  constructor
  { exact lt_of_le_of_lt le_refl 10007 decimal.ligit 10007 }
  constructor
  { exact h }
  sorry

end least_five_digit_congruent_to_11_mod_14_l275_275529


namespace triangle_PNR_area_l275_275762

-- Define variables
variable (P Q R M N : Type)

-- Define the given conditions
variables [RightAngledTriangle P Q R]
variable [Midpoint M P Q]
variable (h1 : length PR = 12)
variable (h2 : length QR = 16)
variable (h3 : perpendicular MN PQ)

-- Define proof statement
theorem triangle_PNR_area : 
  area (triangle P N R) = 21 := 
by
  sorry

end triangle_PNR_area_l275_275762


namespace solve_logarithmic_equation_l275_275967

noncomputable def equation_holds (x : ℝ) : Prop :=
  log (x^2 + 5*x + 6) = log ((x + 1) * (x + 4)) + log (x - 2)

theorem solve_logarithmic_equation (x : ℝ) (h1 : x > 2) (h2 : x^2 + 5*x + 6 > 0) (h3 : (x + 1) * (x + 4) > 0) : 
  equation_holds x ↔ x^3 - 4*x - 14 = 0 :=
sorry

end solve_logarithmic_equation_l275_275967


namespace trapezoid_parallel_intersections_l275_275774

theorem trapezoid_parallel_intersections 
  {A B C D P : Type} [geometry P] 
  (trapezoid : Trapezoid A B C D) 
  (h_parallel : A B ∥ C D) 
  (P_on_BC : lies_on P (segment B C)) :
  ∃ E F : Point,
  parallel (line_through A P) (line_through E C) ∧
  parallel (line_through P D) (line_through B F) ∧
  lies_on E (line_through A D) ∧
  lies_on F (line_through D A) := by
  sorry

end trapezoid_parallel_intersections_l275_275774


namespace problem_1_problem_2_l275_275313

-- Given functions f(x) = x^3 - x and g(x) = x^2 + a
def f (x : ℝ) : ℝ := x ^ 3 - x
def g (x : ℝ) (a : ℝ) : ℝ := x ^ 2 + a

-- (1) If x1 = -1, prove the value of a is 3 given the tangent line condition.
theorem problem_1 (a : ℝ) : 
  let x1 := -1 in
  let f' (x : ℝ) : ℝ := 3 * x ^ 2 - 1 in
  let g' (x : ℝ) : ℝ := 2 * x in
  f' x1 = 2 →  -- Slope of the tangent line at x1 = -1 for f(x)
  g' 1 = 2 →  -- Slope of the tangent line at x2 = 1 for g(x)
  (f x1 + (2 * (1 + 1)) = g 1 a) → -- The tangent line condition
  a = 3 := 
sorry

-- (2) Find the range of values for a under the tangent line condition.
theorem problem_2 (a : ℝ) : 
  let h (x1 : ℝ) := 9 * x1 ^ 4 - 8 * x1 ^ 3 - 6 * x1 ^ 2 + 1 in
   -- We check that the minimum value is h(1) = -4, which gives a range for a
  (∃ x1, x1 ≠ 1 ∧ a ≥ (-1)) :=
sorry

end problem_1_problem_2_l275_275313


namespace dot_product_OA_OB_l275_275453

def cos_func (x : ℝ) : ℝ := Real.cos ((π / 2) * x - (π / 6))

noncomputable def point_A : ℝ × ℝ := (1 / 3, 1)
noncomputable def point_B : ℝ × ℝ := (7 / 3, -1)

theorem dot_product_OA_OB : (point_A.1 * point_B.1 + point_A.2 * point_B.2) = -2 / 9 := by
  sorry

end dot_product_OA_OB_l275_275453


namespace number_of_groups_l275_275128

noncomputable def original_students : ℕ := 22 + 2

def students_per_group : ℕ := 8

theorem number_of_groups : original_students / students_per_group = 3 :=
by
  sorry

end number_of_groups_l275_275128


namespace number_of_valid_digits_l275_275197

theorem number_of_valid_digits :
  let N := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] in
  let valid_digits := [n | n ∈ N ∧ (640 + n) % 4 = 0] in
  valid_digits.length = 5 :=
by sorry

end number_of_valid_digits_l275_275197


namespace clock_angle_315_l275_275158

/-- Each hour mark on a clock face represents 30°. -/
def hour_mark_angle : ℝ := 30

/-- At 3:00, the hour hand is at 90°. -/
def hour_hand_3oclock_angle : ℝ := 3 * hour_mark_angle

/-- At 3:15, the minute hand points at 90°. -/
def minute_hand_315_angle : ℝ := hour_hand_3oclock_angle

/-- The movement of the hour hand from the 3:00 position at 3:15 is 7.5°. -/
def hour_hand_movement_315_angle : ℝ := hour_mark_angle * (15 / 60)

/-- The angle of the hour hand at 3:15 is 97.5°. -/
def hour_hand_315_angle : ℝ := hour_hand_3oclock_angle + hour_hand_movement_315_angle

/-- The measure of the smaller angle between the hour-hand and minute-hand at 3:15 is 7.5°. -/
theorem clock_angle_315 : 
  |hour_hand_315_angle - minute_hand_315_angle| = 7.5 :=
by
  sorry

end clock_angle_315_l275_275158


namespace carla_sharpening_time_l275_275959

theorem carla_sharpening_time (x : ℕ) (h : x + 3 * x = 40) : x = 10 :=
by
  sorry

end carla_sharpening_time_l275_275959


namespace part1_part2_l275_275306

section
variable (f g : ℝ → ℝ) (a : ℝ)

-- Define the given functions.
noncomputable def f := λ x : ℝ, x^3 - x
noncomputable def g := λ x : ℝ, x^2 + a

-- Part 1: Prove that if x₁ = -1, then a = 3
theorem part1 (x₁ : ℝ) (h : x₁ = -1) (h_tangent : ∀ x₃ : ℝ, f'(x₃) = g'(x₃)) : 
    a = 3 :=
sorry

-- Part 2: Prove the range of values for a is [-1, +∞).
theorem part2 (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, 3 * x₁^2 - 1 = 2 * x₂ ∧ a = x₂^2 - 2 * x₁^3) : 
    a ∈ Set.Ici (-1) :=
sorry

end

end part1_part2_l275_275306


namespace larger_ball_radius_l275_275125

open Real

theorem larger_ball_radius :
  let radius_small := 2
  let radius_large: Real
  let volume_small (r: Real) := (4/3) * π * r^3
  let volume_large (r: Real) := (4/3) * π * r^3
  volume_large radius_large = 5 * volume_small radius_small → radius_large = (40)^(1/3) :=
by
  intro radius_small radius_large volume_small volume_large h
  sorry

end larger_ball_radius_l275_275125


namespace distance_home_to_school_l275_275549

theorem distance_home_to_school :
  ∃ T D : ℝ, 6 * (T + 7/60) = D ∧ 12 * (T - 8/60) = D ∧ 9 * T = D ∧ D = 2.1 :=
by
  sorry

end distance_home_to_school_l275_275549


namespace bacterium_descendants_in_range_l275_275902

theorem bacterium_descendants_in_range (total_bacteria : ℕ) (initial : ℕ) 
  (h_total : total_bacteria = 1000) (h_initial : initial = total_bacteria) 
  (descendants : ℕ → ℕ)
  (h_step : ∀ k, descendants (k+1) ≤ descendants k / 2) :
  ∃ k, 334 ≤ descendants k ∧ descendants k ≤ 667 :=
by
  sorry

end bacterium_descendants_in_range_l275_275902


namespace time_to_pass_pole_l275_275589

-- Define the conditions
def speed_kmh : ℝ := 108
def time_to_cross_stationary_train : ℝ := 30
def length_stationary_train : ℝ := 600

-- Convert speed to m/s
def speed_ms : ℝ := speed_kmh * 1000 / 3600

-- Define the proof problem statement
theorem time_to_pass_pole : 
  ∀ (length_moving_train time_to_pass_pole : ℝ),
  (length_moving_train + length_stationary_train = speed_ms * time_to_cross_stationary_train) →
  (time_to_pass_pole = length_moving_train / speed_ms) →
  time_to_pass_pole = 10 := 
by
  sorry

end time_to_pass_pole_l275_275589


namespace parity_of_E2021_E2022_E2023_l275_275255

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def seq (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 0
  else seq (n - 2) + seq (n - 3)

theorem parity_of_E2021_E2022_E2023 :
  is_odd (seq 2021) ∧ is_even (seq 2022) ∧ is_odd (seq 2023) :=
by
  sorry

end parity_of_E2021_E2022_E2023_l275_275255


namespace bucket_volumes_correct_l275_275627

def buckets_initial : List (String × Nat) := [
    ("A", 11), ("B", 13), ("C", 12), ("D", 16), ("E", 10),
    ("F", 8), ("G", 15), ("H", 5)
]

-- Calculate the final contents of bucket X, Y, and Z
def calculate_buckets_contents (buckets : List (String × Nat)) : List (String × Nat) :=
  let E := buckets.lookup "E".get_or_else 0
  let B := buckets.lookup "B".get_or_else 0
  let X := E + B

  let H := buckets.lookup "H".get_or_else 0
  let G := buckets.lookup "G".get_or_else 0
  let F := buckets.lookup "F".get_or_else 0
  let C := buckets.lookup "C".get_or_else 0
  let D := buckets.lookup "D".get_or_else 0
  let Y := H + G + F / 2 + C + D

  let A := buckets.lookup "A".get_or_else 0
  let Z := A + F / 2

  [("X", X), ("Y", Y), ("Z", Z)]

theorem bucket_volumes_correct :
  calculate_buckets_contents buckets_initial = [("X", 23), ("Y", 52), ("Z", 22)] :=
by
  sorry

end bucket_volumes_correct_l275_275627


namespace distribution_less_than_m_plus_h_l275_275187

noncomputable def symmetric_distribution (m h : ℝ) (P : ℝ → ℝ) : Prop :=
  ∀ x, P m = 1/2 ∧ P (m + h) = 34/100 ∧ P (m - h) = 34/100

theorem distribution_less_than_m_plus_h
  (m h : ℝ) (P : ℝ → ℝ)
  (H : symmetric_distribution m h P) :
  ∃ percentage : ℝ, percentage = 84/100 :=
by
  sorry

end distribution_less_than_m_plus_h_l275_275187


namespace probability_heads_and_multiple_of_five_l275_275593

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def coin_is_fair : Prop := true -- since given in conditions, it’s fair, no need to reprove; assume true

def die_is_fair : Prop := true -- since given in conditions, it’s fair, no need to reprove; assume true

theorem probability_heads_and_multiple_of_five :
  coin_is_fair ∧ die_is_fair →
  (1 / 2) * (1 / 6) = (1 / 12) :=
by
  intro h
  sorry

end probability_heads_and_multiple_of_five_l275_275593


namespace volume_of_prism_l275_275517

theorem volume_of_prism (a b c : ℝ) (h₁ : a * b = 48) (h₂ : b * c = 36) (h₃ : a * c = 50) : 
    (a * b * c = 170) :=
by
  sorry

end volume_of_prism_l275_275517


namespace DE_EF_proof_l275_275767

noncomputable def length_DE (BC : ℝ) (angle_C : ℝ) : ℝ :=
  if BC = 30 ∧ angle_C = 45 then 15 else 0
  
noncomputable def length_EF (BC : ℝ) (angle_C : ℝ) : ℝ :=
  if BC = 30 ∧ angle_C = 45 then 15 - 15 * Real.sqrt 2 else 0

theorem DE_EF_proof (BC : ℝ) (angle_C : ℝ)
  (hBC : BC = 30) (hAngleC : angle_C = 45) :
  length_DE BC angle_C = 15 ∧ length_EF BC angle_C = 15 - 15 * Real.sqrt 2 :=
by
  simp [length_DE, length_EF, hBC, hAngleC]
  sorry

end DE_EF_proof_l275_275767


namespace b_value_for_continuity_l275_275987

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 2 then x + 4 else 3 * x + b

theorem b_value_for_continuity (b : ℝ) : (∀ x, continuous_at (λ x : ℝ, f x b) 2) → b = 0 := by
  intro h
  have h1 : f 2 b = 6 := by sorry
  have h2 : f 2 b = 3 * 2 + b := by sorry
  rw [h2] at h1
  linarith
  sorry

end b_value_for_continuity_l275_275987


namespace root_properties_of_cubic_l275_275330

theorem root_properties_of_cubic (z1 z2 : ℂ) (h1 : z1^2 + z1 + 1 = 0) (h2 : z2^2 + z2 + 1 = 0) :
  z1 * z2 = 1 ∧ z1^3 = 1 ∧ z2^3 = 1 :=
by
  -- Proof omitted
  sorry

end root_properties_of_cubic_l275_275330


namespace geometric_sequence_a5_l275_275483

theorem geometric_sequence_a5 (α : Type) [LinearOrderedField α] (a : ℕ → α)
  (h1 : ∀ n, a (n + 1) = a n * 2)
  (h2 : ∀ n, a n > 0)
  (h3 : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end geometric_sequence_a5_l275_275483


namespace find_m_l275_275698

variable {ℝ : Type*}
variables (a b : ℝ → ℝ → ℝ)
variables (m : ℝ)
variable (n : ℝ)
variables (A B C D : ℝ → ℝ → ℝ)

-- Conditions
-- Vectors a and b are not collinear
def not_collinear (a b : ℝ → ℝ → ℝ) : Prop :=
  ∀ k : ℝ, a ≠ k • b

axiom h1 : not_collinear a b

-- Vectors AB, BC, CD
def vector_AB (a b : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := 3 • a + b
def vector_BC (a b : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := a + m • b
def vector_CD (a b : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := 2 • a - b

axiom h2 : vector_AB a b = (λ x y, 3 * a x y + b x y)
axiom h3 : vector_BC a b = (λ x y, a x y + m * b x y)
axiom h4 : vector_CD a b = (λ x y, 2 * a x y - b x y)

-- Collinearity
def collinear (A C D : ℝ → ℝ → ℝ) : Prop :=
  ∃ n : ℝ, (λ x y, C x y - A x y) = n • (λ x y, D x y - C x y)

axiom h5 : collinear A C D

-- Question: Find the value of m
theorem find_m : m = -3 :=
by sorry

end find_m_l275_275698


namespace count_negative_numbers_l275_275380

def evaluate_exprs (e: Real) :=
  match e with
  | -(-2) => 2
  | -3^2 => -9
  | -| -5 | => -5
  | (-1/2)^3 => -1/8

theorem count_negative_numbers : 
  let exprs := [-( -2 : ℝ), -(3:ℝ) ^ 2, -| -5 : ℝ |, (-1/2:ℝ) ^ 3] in
  count exprs.negatives = 3 := by
sorry

end count_negative_numbers_l275_275380


namespace part1_part2_l275_275312

-- Part (1): Given \( f(x) = x^3 - x \), \( g(x) = x^2 + a \), and \( x_1 = -1 \), prove that \( a = 3 \).
theorem part1 (a : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x)
  (hg : ∀ x, g x = x^2 + a)
  (x1 : ℝ) (hx1 : x1 = -1)
  (tangent_match : ∀ x, deriv f x = deriv g x) : a = 3 := 
sorry

-- Part (2): Given \( f(x) = x^3 - x \) and \( g(x) = x^2 + a \), prove that the range of values for \( a \) is \( [-1, +\infty) \).
theorem part2 (a : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x)
  (hg : ∀ x, g x = x^2 + a)
  (range_of_a : ∀ t : set ℝ, ∃ u : set ℝ, ∀ x ∈ t, a ∈ u) : a ∈ set.Ici (-1) :=
sorry

end part1_part2_l275_275312


namespace number_of_divisors_of_n_cubed_l275_275731

theorem number_of_divisors_of_n_cubed (p : ℕ) (hp : Nat.Prime p) (hdiv : ∃ k, k = p^3) : 
    Nat.divisor_count (p^3)^3 = 10 :=
sorry

end number_of_divisors_of_n_cubed_l275_275731


namespace part1_part2_l275_275311

-- Part (1): Given \( f(x) = x^3 - x \), \( g(x) = x^2 + a \), and \( x_1 = -1 \), prove that \( a = 3 \).
theorem part1 (a : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x)
  (hg : ∀ x, g x = x^2 + a)
  (x1 : ℝ) (hx1 : x1 = -1)
  (tangent_match : ∀ x, deriv f x = deriv g x) : a = 3 := 
sorry

-- Part (2): Given \( f(x) = x^3 - x \) and \( g(x) = x^2 + a \), prove that the range of values for \( a \) is \( [-1, +\infty) \).
theorem part2 (a : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x)
  (hg : ∀ x, g x = x^2 + a)
  (range_of_a : ∀ t : set ℝ, ∃ u : set ℝ, ∀ x ∈ t, a ∈ u) : a ∈ set.Ici (-1) :=
sorry

end part1_part2_l275_275311


namespace angle_B_is_acute_l275_275865

theorem angle_B_is_acute {A B C : Type} [EuclideanGeometry A B C] (h1 : AB = AC) :
  ∠B < 90 :=
by
  sorry -- Assume for contradiction and proceed with the proof as described

end angle_B_is_acute_l275_275865


namespace right_triangle_tan_l275_275754

theorem right_triangle_tan (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (angle_BAC : MeasureTheory.Angle B A C = MeasureTheory.Angle.right)
  (AB_len : dist A B = 40)
  (BC_len : dist B C = 41) :
  tangent_of_angle A B C = 9 / 40 :=
sorry

end right_triangle_tan_l275_275754


namespace ranch_horses_ponies_difference_l275_275587

theorem ranch_horses_ponies_difference :
  ∃ P H : ℕ, 
  (P + H = 163) ∧ 
  (P % 16 = 0) ∧ 
  (H - P = 3) :=
begin
  sorry
end

end ranch_horses_ponies_difference_l275_275587


namespace max_area_triangle_ABC_l275_275792

open Real

-- Define the points and the function given in the problem.
def pointA : ℝ × ℝ := (0, 3)
def pointB : ℝ × ℝ := (4, 3)
def parabola (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Define the area function for the triangle ABC using the Shoelace formula.
def triangle_area (p q : ℝ) : ℝ :=
  1 / 2 * abs ((0) * q + (4) * q + p * 3 - 3 * 4 - 3 * p)

-- Problem condition that point C is on the parabola.
def pointC (p : ℝ) : ℝ × ℝ := (p, parabola p)

-- Define the problem statement in Lean.
theorem max_area_triangle_ABC : 
  ∃ p q, 
  pointC p = (p, p^2 - 4*p + 3) ∧ 
  0 ≤ p ∧ p ≤ 4 ∧ 
  triangle_area p (p^2 - 4 * p + 3) = 2 :=
sorry

end max_area_triangle_ABC_l275_275792


namespace parabola_intersection_length_l275_275207

theorem parabola_intersection_length (x1 x2 : ℝ) (y1 y2 : ℝ) :
  (x1 + x2 = 5) ∧ (y1 ^ 2 = 8 * x1) ∧ (y2 ^ 2 = 8 * x2) → 
  (abs((y1 - y2)) + abs((y2 - y1)) = 9) :=
by
  sorry

end parabola_intersection_length_l275_275207


namespace no_such_rectangle_l275_275676

theorem no_such_rectangle (a b x y : ℝ) (ha : a < b)
  (hx : x < a / 2) (hy : y < a / 2)
  (h_perimeter : 2 * (x + y) = a + b)
  (h_area : x * y = (a * b) / 2) :
  false :=
sorry

end no_such_rectangle_l275_275676


namespace original_average_l275_275404

theorem original_average (A : ℝ)
  (h1 : (Σ i in (finset.range 15), A) / 15 = A)
  (h2 : (Σ i in (finset.range 15), (A + 10)) / 15 = 50) :
  A = 40 :=
sorry

end original_average_l275_275404


namespace triangle_area_and_right_angle_l275_275256

-- Define the vertices
def A : Fin 2 → ℝ := ![1, 2]
def B : Fin 2 → ℝ := ![-1, -1]
def C : Fin 2 → ℝ := ![4, -3]

-- Compute vectors
def vectorCA : Fin 2 → ℝ := C - A
def vectorCB : Fin 2 → ℝ := C - B

-- Define the area function for a triangle given two vectors
def triangle_area (v w : Fin 2 → ℝ) : ℝ :=
  0.5 * abs (v 0 * w 1 - v 1 * w 0)

-- Check if vectors are orthogonal
def is_right_angled (v w : Fin 2 → ℝ) : Bool :=
  (v 0 * w 0 + v 1 * w 1) = 0

-- The main theorem
theorem triangle_area_and_right_angle :
  triangle_area vectorCA vectorCB = 19 / 2 ∧ ¬is_right_angled vectorCA vectorCB :=
by
  sorry

end triangle_area_and_right_angle_l275_275256


namespace Coupon_A_greater_than_B_and_C_at_prices_l275_275574

def coupon_discount_A (x : ℝ) : ℝ := 
  if x >= 75 then 0.15 * x else 0

def coupon_discount_B (x : ℝ) : ℝ := 
  if x >= 150 then 30 else 0

def coupon_discount_C (x : ℝ) : ℝ := 
  if x > 150 then 0.22 * (x - 150) else 0

theorem Coupon_A_greater_than_B_and_C_at_prices :
  ∀ x, x = 179.95 ∨ x = 199.95 ∨ x = 249.95 ∨ x = 299.95 ∨ x = 349.95 →
       (coupon_discount_A x > coupon_discount_B x) ∧ (coupon_discount_A x > coupon_discount_C x) ↔ 
       (x = 249.95 ∨ x = 299.95 ∨ x = 349.95) :=
by
  intro x hx
  sorry

end Coupon_A_greater_than_B_and_C_at_prices_l275_275574


namespace option_C_incorrect_l275_275064

structure Line := (point1 point2 : ℝ × ℝ × ℝ)
structure Plane := (point : ℝ × ℝ × ℝ) (normal : ℝ × ℝ × ℝ)

variables (m n : Line) (α β : Plane)

def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular_to_plane (p1 p2 : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def lines_parallel (l1 l2 : Line) : Prop := sorry
def lines_perpendicular (l1 l2 : Line) : Prop := sorry
def planes_parallel (p1 p2 : Plane) : Prop := sorry

theorem option_C_incorrect 
  (h1 : line_in_plane m α)
  (h2 : line_parallel_to_plane n α)
  (h3 : lines_parallel m n) :
  false :=
sorry

end option_C_incorrect_l275_275064


namespace expression_evaluation_l275_275272

theorem expression_evaluation : 
  (∛((-4:ℝ)^3)) + ((-(1/8:ℝ))^(-4/3)) + (Real.log 2 2)^2 + (Real.log 5 20) * (Real.log 2 20) = 13 :=
by sorry

end expression_evaluation_l275_275272


namespace quadratic_equation_solution_l275_275165

variable (x : ℝ)

theorem quadratic_equation_solution :
  (3 * x^2 + 6 * x = abs (-21 + 5)) ↔ (x = -1 + sqrt 19 ∨ x = -1 - sqrt 19) :=
by
  sorry

end quadratic_equation_solution_l275_275165


namespace probability_of_selecting_product_at_least_4_l275_275459

theorem probability_of_selecting_product_at_least_4 : 
  let products := {1, 2, 3, 4, 5} in
  let favorable_products := {x ∈ products | x ≥ 4} in
  let probability := (favorable_products.card : ℝ) / (products.card : ℝ) in
  probability = 2 / 5 :=
by
  let products := {1, 2, 3, 4, 5}
  let favorable_products := {x ∈ products | x ≥ 4}
  have cardinality_products : products.card = 5 := sorry
  have cardinality_favorable_products : favorable_products.card = 2 := sorry
  have h : (2 : ℝ) / (5 : ℝ) = 2 / 5 := by norm_num
  have probability := (favorable_products.card : ℝ) / (products.card : ℝ)
  rw [cardinality_products, cardinality_favorable_products] at probability
  exact (Eq.trans probability h).symm

end probability_of_selecting_product_at_least_4_l275_275459


namespace probability_case_7_probability_case_n_l275_275028

-- Define the scenario for n = 7
def probability_empty_chair_7 : Prop :=
  ∃ p : ℚ, p = 1 / 35

-- Theorem for the case when there are exactly 7 chairs
theorem probability_case_7 : probability_empty_chair_7 :=
begin
  -- Proof skipped
  sorry
end

-- Define the general scenario for n chairs (n >= 6)
def probability_empty_chair_n (n : ℕ) (h : n ≥ 6) : Prop :=
  ∃ p : ℚ, p = (n - 4) * (n - 5) / ((n - 1) * (n - 2))

-- Theorem for the case when there are n chairs (n ≥ 6)
theorem probability_case_n (n : ℕ) (h : n ≥ 6) : probability_empty_chair_n n h :=
begin
  -- Proof skipped
  sorry
end

end probability_case_7_probability_case_n_l275_275028


namespace undefined_values_of_expression_l275_275664

theorem undefined_values_of_expression (a : ℝ) :
  a^2 - 9 = 0 ↔ a = -3 ∨ a = 3 := 
sorry

end undefined_values_of_expression_l275_275664


namespace number_not_2_nice_nor_3_nice_l275_275276

noncomputable def k_nice (N k : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ (a^k).divisors.card = N

def count_not_knice (k1 k2 n : ℕ) : ℕ :=
  let is_k1_nice m := k_nice m k1
  let is_k2_nice m := k_nice m k2
  (Finset.range n).filter (λ m, ¬ is_k1_nice m ∧ ¬ is_k2_nice m).card

theorem number_not_2_nice_nor_3_nice : count_not_knice 2 3 500 = 168 := sorry

end number_not_2_nice_nor_3_nice_l275_275276


namespace original_number_l275_275914

theorem original_number (x : ℝ) (h : 1.35 * x = 680) : x = 503.70 :=
sorry

end original_number_l275_275914


namespace no_integer_n_such_that_squares_l275_275085

theorem no_integer_n_such_that_squares :
  ¬ ∃ n : ℤ, (∃ k1 : ℤ, 10 * n - 1 = k1 ^ 2) ∧
             (∃ k2 : ℤ, 13 * n - 1 = k2 ^ 2) ∧
             (∃ k3 : ℤ, 85 * n - 1 = k3 ^ 2) := 
by sorry

end no_integer_n_such_that_squares_l275_275085


namespace small_cube_edge_length_l275_275523

theorem small_cube_edge_length 
  (m n : ℕ)
  (h1 : 12 % m = 0) 
  (h2 : n = 12 / m) 
  (h3 : 6 * (n - 2)^2 = 12 * (n - 2)) 
  : m = 3 :=
by 
  sorry

end small_cube_edge_length_l275_275523


namespace area_of_trapezoid_EFGH_l275_275531

noncomputable def length (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

noncomputable def height_FG : ℝ :=
  6 - 2

noncomputable def area_trapezoid (E F G H : ℝ × ℝ) : ℝ :=
  let base1 := length E F
  let base2 := length G H
  let height := height_FG
  1/2 * (base1 + base2) * height

theorem area_of_trapezoid_EFGH :
  area_trapezoid (0, 0) (2, -3) (6, 0) (6, 4) = 2 * (Real.sqrt 13 + 4) :=
by
  sorry

end area_of_trapezoid_EFGH_l275_275531


namespace find_length_of_backyard_l275_275231

theorem find_length_of_backyard (L : ℕ) (width_shed length_shed : ℕ) :
  width_shed = 3 → length_shed = 5 → (L * 13 - width_shed * length_shed = 245) → L = 20 :=
begin
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  linarith,
end

end find_length_of_backyard_l275_275231


namespace math_problem_l275_275712

open Set

variable (U A B : Set ℝ)

noncomputable def U := {x : ℝ | -5 < x ∧ x ≤ 4}
noncomputable def A := {x ∈ U | -5 < x ∧ x < 3}
noncomputable def B := {x ∈ U | -1 ≤ x ∧ x ≤ 3}

theorem math_problem : A ∩ (U \ B) = { x : ℝ | -5 < x ∧ x < -1 } :=
by sorry

end math_problem_l275_275712


namespace gcd_factorial_8_10_l275_275659

-- Define the concept of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n 

-- Statement of the problem 
theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := 
by
  sorry

end gcd_factorial_8_10_l275_275659


namespace find_slope_of_line_l275_275695

theorem find_slope_of_line (k m x0 : ℝ) (P Q : ℝ × ℝ) 
  (hP : P.2^2 = 4 * P.1) 
  (hQ : Q.2^2 = 4 * Q.1) 
  (hMid : (P.1 + Q.1) / 2 = x0 ∧ (P.2 + Q.2) / 2 = 2) 
  (hLineP : P.2 = k * P.1 + m) 
  (hLineQ : Q.2 = k * Q.1 + m) : k = 1 :=
by sorry

end find_slope_of_line_l275_275695


namespace find_number_l275_275504

theorem find_number (x : ℝ) (h : 10 * x = 2 * x - 36) : x = -4.5 :=
sorry

end find_number_l275_275504


namespace well_rate_correct_l275_275270

noncomputable def well_rate (depth : ℝ) (diameter : ℝ) (total_cost : ℝ) : ℝ :=
  let r := diameter / 2
  let volume := Real.pi * r^2 * depth
  total_cost / volume

theorem well_rate_correct :
  well_rate 14 3 1583.3626974092558 = 15.993 :=
by
  sorry

end well_rate_correct_l275_275270


namespace probability_seven_chairs_probability_n_chairs_l275_275033
-- Importing necessary library to ensure our Lean code can be built successfully

-- Definition for case where n = 7
theorem probability_seven_chairs : 
  let total_seating := 7 * 6 * 5 / 6 
  let favorable_seating := 1 
  let probability := favorable_seating / total_seating 
  probability = 1 / 35 := 
by 
  sorry

-- Definition for general case where n ≥ 6
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) : 
  let total_seating := (n - 1) * (n - 2) / 2 
  let favorable_seating := (n - 4) * (n - 5) / 2 
  let probability := favorable_seating / total_seating 
  probability = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := 
by 
  sorry

end probability_seven_chairs_probability_n_chairs_l275_275033


namespace color_no_arith_seq_18_l275_275083

-- Define the set of natural numbers from 1 to 1986
def natSet : Set ℕ := { n | 1 ≤ n ∧ n ≤ 1986 }

-- Define a predicate for an arithmetic sequence of length 18 within the set
def is_arith_seq (seq : Fin 18 → ℕ) : Prop :=
  ∃ a d, ∀ i, seq i = a + i * d ∧ seq i ∈ natSet

-- Define a predicate for a monochromatic sequence of length 18
def monochromatic (seq : Fin 18 → ℕ) (coloring : ℕ → bool) : Prop :=
  ∀ i j, i ≠ j → seq i ≠ seq j ∧ coloring (seq i) = coloring (seq j)

-- The theorem statement
theorem color_no_arith_seq_18 :
  ∃ coloring : ℕ → bool, ∀ seq : Fin 18 → ℕ, is_arith_seq seq → ¬ monochromatic seq coloring :=
sorry

end color_no_arith_seq_18_l275_275083


namespace basic_astrophysics_research_degrees_l275_275189

def budget_percentages : List ℕ := [12, 18, 15, 23, 7, 10, 4, 5, 3]

theorem basic_astrophysics_research_degrees :
  let total_percent := 100
  let used_percent := budget_percentages.sum
  let remaining_percent := total_percent - used_percent
  let degrees_in_circle := 360
  let astrophysics_degrees := (remaining_percent * degrees_in_circle) / total_percent
  astrophysics_degrees = 10.8 :=
by
  sorry

end basic_astrophysics_research_degrees_l275_275189


namespace geometric_seq_correct_factorial_seq_correct_l275_275181

def geometric_seq (a q : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := geometric_seq n * q

def factorial_seq : ℕ → ℕ
| 0     := 1
| (n+1) := factorial_seq n * (n + 1)

theorem geometric_seq_correct (a q : ℕ) (n : ℕ) :
  ∃ u : ℕ → ℕ, u 0 = a ∧ (∀ n, u (n + 1) = u n * q) :=
begin
  use geometric_seq a q,
  split,
  { refl, },
  { intro n,
    refl, },
end

theorem factorial_seq_correct (n : ℕ) :
  ∃ u : ℕ → ℕ, u 0 = 1 ∧ (∀ n, u (n + 1) = u n * (n + 1)) :=
begin
  use factorial_seq,
  split,
  { refl, },
  { intro n,
    refl, },
end

end geometric_seq_correct_factorial_seq_correct_l275_275181


namespace EF_eq_EG_eq_EH_eq_EI_l275_275053

theorem EF_eq_EG_eq_EH_eq_EI
  (A B C D E F G H I : Point)
  (AB DC EF EH ABCD ADE : ℝ)
  (P Q R S : Point)
  (H1 : convex_quadrilateral A B C D)
  (H2 : interior_point E ABCD)
  (H3 : ∠ABF ∼ ∠DCE)
  (H4 : ∠BCG ∼ ∠ADE)
  (H5 : ∠CDH ∼ ∠BAE)
  (H6 : ∠DAI ∼ ∠CBE)
  (H7 : projection E AB P)
  (H8 : projection E BC Q)
  (H9 : projection E CD R)
  (H10 : projection E DA S)
  (H11 : cyclic_quadrilateral P Q R S) :
  EF * CD = EG * DA ∧ EG * DA = EH * AB ∧ EH * AB = EI * BC := 
sorry

end EF_eq_EG_eq_EH_eq_EI_l275_275053


namespace circle_radius_l275_275906

/-
  Given:
  - The area of the circle x = π r^2
  - The circumference of the circle y = 2π r
  - The sum x + y = 72π

  Prove:
  The radius r = 6
-/
theorem circle_radius (r : ℝ) (x : ℝ) (y : ℝ) 
  (h₁ : x = π * r ^ 2) 
  (h₂ : y = 2 * π * r) 
  (h₃ : x + y = 72 * π) : 
  r = 6 := 
sorry

end circle_radius_l275_275906


namespace find_foreign_language_score_l275_275144

variable (c m f : ℝ)

theorem find_foreign_language_score
  (h1 : (c + m + f) / 3 = 94)
  (h2 : (c + m) / 2 = 92) :
  f = 98 := by
  sorry

end find_foreign_language_score_l275_275144


namespace decreasing_function_inequality_l275_275293

theorem decreasing_function_inequality {f : ℝ → ℝ} (h_decreasing : ∀ x1 x2 : ℝ, x1 > x2 → f(x1) < f(x2))
  (h_condition : f (3 * a) < f (-2 * a + 10)) : a > 2 :=
sorry

end decreasing_function_inequality_l275_275293


namespace ratio_sheep_horses_l275_275604

theorem ratio_sheep_horses
  (horse_food_per_day : ℕ)
  (total_horse_food : ℕ)
  (number_of_sheep : ℕ)
  (number_of_horses : ℕ)
  (gcd_sheep_horses : ℕ):
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  number_of_sheep = 40 →
  number_of_horses = total_horse_food / horse_food_per_day →
  gcd number_of_sheep number_of_horses = 8 →
  (number_of_sheep / gcd_sheep_horses = 5) ∧ (number_of_horses / gcd_sheep_horses = 7) :=
by
  intros
  sorry

end ratio_sheep_horses_l275_275604


namespace function_equation_solution_l275_275429

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_equation_solution (a : ℝ) : 
  (∀ x y : ℝ, f (f x + 2 * y) = 6 * x + f (f y - x)) → (∀ x : ℝ, f x = 2 * x + a) := 
by 
  simp [f, a]
  sorry

end function_equation_solution_l275_275429


namespace proof_problem_l275_275290

-- The conditions for the circle
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, 2)
def center_C (a : ℝ) : Prop := a = 0

-- The equation of the circle to be proven
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

-- The conditions for the line intersecting the circle
def line_intersects_circle (k : ℝ) : Prop :=
  ∀ (x y : ℝ), y = k * x + 1 → x^2 + y^2 = 4

-- The dot product condition
def dot_product_condition (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = -2

-- Perpendicular line and max area conditions
def perp_line (k : ℝ) : ℝ × ℝ → ℝ × ℝ → Prop :=
  λ P Q, P.1 * Q.1 + P.2 * Q.2 = 0

constant maximum_area : ℝ
axiom area_quad (P Q M N : ℝ × ℝ) : 
  maximum_area = 7

-- Lean 4 statement for the proof
theorem proof_problem :
  (∀ (a : ℝ), center_C a → ∃ (x y : ℝ), circle_equation x y) ∧ 
  (∀ (k : ℝ), (∃ P Q : ℝ × ℝ, line_intersects_circle k ∧ dot_product_condition P Q) → k = 0) ∧ 
  (∀ (k : ℝ) (P Q M N : ℝ × ℝ), perp_line k (P, Q) ∧ circle_equation P.1 P.2 ∧ circle_equation Q.1 Q.2 ∧ circle_equation M.1 M.2 ∧ circle_equation N.1 N.2 → maximum_area = 7) :=
sorry

end proof_problem_l275_275290


namespace probability_case_7_probability_case_n_l275_275031

-- Define the scenario for n = 7
def probability_empty_chair_7 : Prop :=
  ∃ p : ℚ, p = 1 / 35

-- Theorem for the case when there are exactly 7 chairs
theorem probability_case_7 : probability_empty_chair_7 :=
begin
  -- Proof skipped
  sorry
end

-- Define the general scenario for n chairs (n >= 6)
def probability_empty_chair_n (n : ℕ) (h : n ≥ 6) : Prop :=
  ∃ p : ℚ, p = (n - 4) * (n - 5) / ((n - 1) * (n - 2))

-- Theorem for the case when there are n chairs (n ≥ 6)
theorem probability_case_n (n : ℕ) (h : n ≥ 6) : probability_empty_chair_n n h :=
begin
  -- Proof skipped
  sorry
end

end probability_case_7_probability_case_n_l275_275031


namespace intersection_M_N_l275_275069

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N:
  M ∩ N = {1, 2} :=
sorry

end intersection_M_N_l275_275069


namespace categorization_proof_l275_275635

-- Definitions of expenses
inductive ExpenseType
| Fixed
| Variable

-- Expense data structure
structure Expense where
  name : String
  type : ExpenseType

-- List of expenses
def fixed_expenses : List Expense :=
  [ { name := "Utility payments", type := ExpenseType.Fixed },
    { name := "Loan payments", type := ExpenseType.Fixed },
    { name := "Taxes", type := ExpenseType.Fixed } ]

def variable_expenses : List Expense :=
  [ { name := "Entertainment", type := ExpenseType.Variable },
    { name := "Travel", type := ExpenseType.Variable },
    { name := "Purchasing non-essential items", type := ExpenseType.Variable },
    { name := "Renting professional video cameras", type := ExpenseType.Variable },
    { name := "Maintenance of the blog", type := ExpenseType.Variable } ]

-- Expenses that can be economized
def economizable_expenses : List String :=
  [ "Payment for home internet and internet traffic",
    "Travel expenses",
    "Renting professional video cameras for a year",
    "Domain payment for blog maintenance",
    "Visiting coffee shops (4 times a month)" ]

-- Expenses that cannot be economized
def non_economizable_expenses : List String :=
  [ "Loan payments",
    "Tax payments",
    "Courses for qualification improvement in blogger school (onsite training)" ]

-- Additional expenses and economizing suggestions
def additional_expenses : List String :=
  [ "Professional development workshops",
    "Marketing and advertising costs",
    "Office supplies",
    "Subscription services" ]

-- Lean statement for the problem
theorem categorization_proof :
  (∀ exp ∈ fixed_expenses, exp.name ∈ non_economizable_expenses) ∧
  (∀ exp ∈ variable_expenses, exp.name ∈ economizable_expenses) ∧
  (∃ exp ∈ additional_expenses, true) :=
by
  sorry

end categorization_proof_l275_275635


namespace radical_product_l275_275155

def fourth_root (x : ℝ) : ℝ := x ^ (1/4)
def third_root (x : ℝ) : ℝ := x ^ (1/3)
def square_root (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_product :
  fourth_root 81 * third_root 27 * square_root 9 = 27 := 
by
  sorry

end radical_product_l275_275155


namespace length_of_polar_curve_l275_275557

theorem length_of_polar_curve :
  (∫ (φ : ℝ) in 0..(3/4 : ℝ), sqrt((4 * φ) ^ 2 + (4) ^ 2)) = (15 / 8 + 2 * Real.log 2) :=
by
  sorry

end length_of_polar_curve_l275_275557


namespace min_period_is_pi_l275_275066

-- Define the function f(x) = sin(ωx)
def f (ω x : ℝ) : ℝ := Real.sin (ω * x)

-- Declare the conditions
variables (P : ℝ × ℝ) (ω : ℝ)
hypothesis (cond1 : ∀ x, P = (Real.geomAvg (0, π/ω), f ω (Real.geomAvg (0, π/ω))))
hypothesis (cond2 : P.1 = Real.geomAvg (0, π/ω) ∧ P.2 = 0)

-- The period of the sine function is 2π/ω
noncomputable def period (ω : ℝ) : ℝ := (2 * π) / ω

-- The condition that P is the symmetric center of the graph of f(x) implies that the minimum distance from P to the axis of symmetry is π/4
noncomputable def min_distance_period (P : ℝ × ℝ) (ω : ℝ) (h1 : ∀ x, P = (Real.geomAvg (0, π/ω), f ω (Real.geomAvg (0, π/ω)))) (h2 : P.1 = Real.geomAvg (0, π/ω) ∧ P.2 = 0) : Prop :=
  ∃ k : ℕ, period ω / 2 = π / 4

-- The theorem statement
theorem min_period_is_pi (ω : ℝ) (P : ℝ × ℝ)
    (h1 : ∀ x, P = (Real.geomAvg (0, π/ω), f ω (Real.geomAvg (0, π/ω))))
    (h2 : P.1 = Real.geomAvg (0, π/ω) ∧ P.2 = 0)
    : period ω = π :=
by sorry

end min_period_is_pi_l275_275066


namespace triangle_ABC_angles_l275_275010

-- Define the triangle ABC with angle bisectors and median
variables {A B C I K L O : Type}
variables [IsTriangle A B C] (angle_bisector_A : AngleBisector A) (angle_bisector_B : AngleBisector B) (median_C : Median C)
variables (right_isosceles_intersections : IsRightIsosceles (triangle_intersections angle_bisector_A angle_bisector_B median_C))

-- The required proof for the angles of triangle ABC
theorem triangle_ABC_angles :
  ∠ABC = 60 ∧ ∠BAC = 30 ∧ ∠ACB = 90 :=
begin
  sorry
end

end triangle_ABC_angles_l275_275010


namespace problem1_problem2_l275_275320

noncomputable theory

-- Given functions
def f (x : ℝ) := x^3 - x
def g (x : ℝ) (a : ℝ) := x^2 + a

-- Problem (1): If x1 = -1, show that a = 3
theorem problem1 (a : ℝ) (h : tangent_line_at f (-1) = tangent_line_at (g (-1)) (f (-1))) :
  a = 3 := sorry

-- Lean theorem to find the range of values for a such that tangent line conditions hold
theorem problem2 (x₁ : ℝ) (a : ℝ) (h : tangent_line_at f x₁ = tangent_line_at g x₁ a) :
  a ≥ -1 := sorry

end problem1_problem2_l275_275320


namespace find_x_correct_l275_275654

noncomputable def find_x : Real :=
  let x := 0.8632
  if (0 < x) ∧ (Real.sqrt (18 * x) * Real.sqrt (2 * x) * Real.sqrt (25 * x) * Real.sqrt (5 * x) = 50)
    then x
    else 0

theorem find_x_correct : ∃ x : Real, (0 < x) ∧ (Real.sqrt (18 * x) * Real.sqrt (2 * x) * Real.sqrt (25 * x) * Real.sqrt (5 * x) = 50) ∧ (x = 0.8632) := by
  let x := 0.8632
  have h1 : 0 < x := by
    sorry
  have h2 : Real.sqrt (18 * x) * Real.sqrt (2 * x) * Real.sqrt (25 * x) * Real.sqrt (5 * x) = 50 := by
    sorry
  exists x
  constructor
    exact h1
  constructor
    exact h2
  rfl

end find_x_correct_l275_275654


namespace tank_capacity_l275_275910

theorem tank_capacity (x : ℝ) (h : 0.24 * x = 120) : x = 500 := 
sorry

end tank_capacity_l275_275910


namespace parabola_focus_on_line_l275_275003

theorem parabola_focus_on_line (p : ℝ) (h₁ : 0 < p) (h₂ : (2 * (p / 2) + 0 - 2 = 0)) : p = 2 :=
sorry

end parabola_focus_on_line_l275_275003


namespace trajectory_of_P_max_min_QA_QC_lambda_range_l275_275105

noncomputable def point (x y : ℝ) := (x, y)

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

variable (P Q A B C : ℝ × ℝ) 
variable (H1 : distance P A = 2 * distance P B)
variable (H2 : P.1 = 4 - Q.1 ∧ P.2 = 2 - Q.2)
variable (A_eq : A = (-2, 0))
variable (B_eq : B = (1, 0))
variable (C_eq : C = (3, 0))
variable (E F : ℝ × ℝ)
variable (H3 : F.1 + 2 = λ * (E.1 + 2) ∧ F.2 = λ * E.2)
variable (H4 : λ > 1)

theorem trajectory_of_P : (x y : ℝ) → 
  (distance (x, y) A = 2 * distance (x, y) B) ↔ (x - 2)^2 + y^2 = 4 :=
sorry

theorem max_min_QA_QC : (x_0 y_0 : ℝ) → 
  (x_0 - 2)^2 + (y_0 - 2)^2 = 4 → ∃ (z : ℝ),  
  13 ≤ (x_0 + 2)^2 + y_0^2 + (x_0 - 3)^2 + y_0^2 ∧ 
  (x_0 + 2)^2 + y_0^2 + (x_0 - 3)^2 + y_0^2 ≤ 53 :=
sorry

theorem lambda_range : 1 < λ ∧ λ ≤ (3 + real.sqrt 5) / 2 :=
sorry

end trajectory_of_P_max_min_QA_QC_lambda_range_l275_275105


namespace num_possibilities_l275_275193

def last_digit_divisible_by_4 (n : Nat) : Prop := (60 + n) % 4 = 0

theorem num_possibilities : {n : Nat | n < 10 ∧ last_digit_divisible_by_4 n}.card = 3 := by
  sorry

end num_possibilities_l275_275193


namespace closest_integer_to_N_l275_275478

def radius1 : Real := 5
def radius2 : Real := 3

def area (r : Real) : Real := Real.pi * r^2

def percentage_increase (r1 r2 : Real) : Real :=
  ((area r1 - area r2) / area r2) * 100

theorem closest_integer_to_N :
  Int.floor (percentage_increase radius1 radius2 + 0.5) = 178 :=
by
  sorry

end closest_integer_to_N_l275_275478


namespace total_journey_distance_l275_275892

variable (D : ℚ) (lateTime : ℚ := 1/4)

theorem total_journey_distance :
  (∃ (T : ℚ), T = D / 40 ∧ T + lateTime = D / 35) →
  D = 70 :=
by
  intros h
  obtain ⟨T, h1, h2⟩ := h
  have h3 : T = D / 40 := h1
  have h4 : T + lateTime = D / 35 := h2
  sorry

end total_journey_distance_l275_275892


namespace beef_original_weight_l275_275171

noncomputable def original_weight (W : ℝ) : Prop :=
  0.65 * W = 546

theorem beef_original_weight : ∃ W, original_weight W ∧ W = 840 :=
by
  use 840
  dsimp [original_weight]
  have h : 0.65 * 840 = 546 :=
    calc
      0.65 * 840 = 546 : by norm_num
  exact ⟨h, rfl⟩

end beef_original_weight_l275_275171


namespace shaded_area_eq_1600pi_l275_275489

variable (r1 r2 : ℝ) (AB : ℝ) (pi : ℝ)
hypothesis h1 : AB = 80
hypothesis h2 : ∃ O P A B : Type, ∀ (r1 r2 : ℝ), r1 > r2 ∧ ∀ (AB : ℝ), AB = 80 ∧ (∀ cord AB, tangent AB)

-- Prove that the shaded area is 1600π
theorem shaded_area_eq_1600pi 
(h : (AB = 80) ∧ ( ∃ O P A B : Type, ∀ (r1 r2 : ℝ), r1 > r2 ∧ ∀ (AB : ℝ), AB = 80 ∧ (∀ cord AB, tangent AB))) : 
  let AO := λ AP OP, AP^2 = AO^2 - OP^2 in 
  (AO^2π - AP^2π) = 1600π :=
sorry

end shaded_area_eq_1600pi_l275_275489


namespace sum_of_T_is_correct_nine_hundred_twenty_nine_in_base_3_l275_275059

-- Define the set T of all three-digit numbers in base 3
def T : Finset ℕ := {n | ∃ (d2 d1 d0 : ℕ), d2 ∈ {1, 2} ∧ d1 ∈ {0, 1, 2} ∧ d0 ∈ {0, 1, 2} ∧ n = d2 * 9 + d1 * 3 + d0}.to_finset

-- State the theorem
theorem sum_of_T_is_correct : ∑ n in T, n = 729 :=
by sorry

-- Show that 729 in base 10 converts to 1000000 in base 3
theorem nine_hundred_twenty_nine_in_base_3 : nat.to_digits 3 729 = [1, 0, 0, 0, 0, 0, 0] :=
by sorry

end sum_of_T_is_correct_nine_hundred_twenty_nine_in_base_3_l275_275059


namespace valid_documents_count_l275_275616

-- Definitions based on the conditions
def total_papers : ℕ := 400
def invalid_percentage : ℝ := 0.40
def valid_percentage : ℝ := 1.0 - invalid_percentage

-- Question and answer formalized as a theorem
theorem valid_documents_count : total_papers * valid_percentage = 240 := by
  sorry

end valid_documents_count_l275_275616


namespace expression_value_l275_275611

def a : ℕ := 45
def b : ℕ := 18
def c : ℕ := 10

theorem expression_value :
  (a + b)^2 - (a^2 + b^2 + c) = 1610 := by
  sorry

end expression_value_l275_275611


namespace solve_inequality_l275_275469

theorem solve_inequality (x : ℝ) (h : x ≥ 10^(-3/2)) : 
  (log10 x + 3)^7 + (log10 x)^7 + log10 (x^2) + 3 ≥ 0 := 
sorry

end solve_inequality_l275_275469


namespace julian_younger_than_frederick_by_20_l275_275048

noncomputable def Kyle: ℕ := 25
noncomputable def Tyson: ℕ := 20
noncomputable def Julian : ℕ := Kyle - 5
noncomputable def Frederick : ℕ := 2 * Tyson

theorem julian_younger_than_frederick_by_20 : Frederick - Julian = 20 :=
by
  sorry

end julian_younger_than_frederick_by_20_l275_275048


namespace odometer_problem_l275_275626

theorem odometer_problem
  (a b c : ℕ) -- a, b, c are natural numbers
  (h1 : 1 ≤ a) -- condition (a ≥ 1)
  (h2 : a + b + c ≤ 7) -- condition (a + b + c ≤ 7)
  (h3 : 99 * (c - a) % 55 = 0) -- 99(c - a) must be divisible by 55
  (h4 : 100 * a + 10 * b + c < 1000) -- ensuring a, b, c keeps numbers within 3-digits
  (h5 : 100 * c + 10 * b + a < 1000) -- ensuring a, b, c keeps numbers within 3-digits
  : a^2 + b^2 + c^2 = 37 := sorry

end odometer_problem_l275_275626


namespace flour_baking_soda_ratio_l275_275009

theorem flour_baking_soda_ratio 
  (sugar flour baking_soda : ℕ)
  (h1 : sugar = 2000)
  (h2 : 5 * flour = 6 * sugar)
  (h3 : 8 * (baking_soda + 60) = flour) :
  flour / baking_soda = 10 := by
  sorry

end flour_baking_soda_ratio_l275_275009


namespace annulus_diagonal_l275_275225

-- Definitions from conditions
variables (b c a XY : ℝ) (hb : b > c)

-- Problem statement
theorem annulus_diagonal (h1 : ∀ XZ XW, XZ = XW = a) 
                         (h2 : ∀ OX OZ XY, OX = b ∧ OZ = c ∧ XY = sqrt(b^2 + c^2)) : 
                         XY = sqrt(b^2 + c^2) := 
by
  sorry

end annulus_diagonal_l275_275225


namespace projection_of_b_on_a_l275_275353

variables {a b : ℝ → ℝ} -- considering a and b to be functions representing vectors in R^2

-- Conditions
def non_zero_vectors : Prop := (a 0 ≠ 0 ∨ a 1 ≠ 0) ∧ (b 0 ≠ 0 ∨ b 1 ≠ 0)
def magnitude_a : Prop := a 0 ^ 2 + a 1 ^ 2 = 4 -- |a|^2 = 4
def orthogonal_condition : Prop := a 0 * (a 0 + 2 * b 0) + a 1 * (a 1 + 2 * b 1) = 0 -- a ⊥ (a + 2b)

theorem projection_of_b_on_a : non_zero_vectors → magnitude_a → orthogonal_condition → 
  (a 0 * b 0 + a 1 * b 1) / (real.sqrt (a 0 ^ 2 + a 1 ^ 2)) = -1 :=
by
  intros h_non_zero h_magnitude h_orthogonal
  sorry

end projection_of_b_on_a_l275_275353


namespace cube_surface_areas_of_100_unit_cubes_l275_275822

theorem cube_surface_areas_of_100_unit_cubes :
  ∃ (surface_areas : List ℕ), 
  (length surface_areas ≥ 6) ∧ 
  (surface_areas.take 6 = [130, 134, 136, 138, 140, 142]) :=
by
  sorry

end cube_surface_areas_of_100_unit_cubes_l275_275822


namespace quadratic_one_real_root_positive_n_l275_275376

theorem quadratic_one_real_root_positive_n (n : ℝ) (h : (n ≠ 0)) :
  (∃ x : ℝ, (x^2 - 6*n*x - 9*n) = 0) ∧
  (∀ x y : ℝ, (x^2 - 6*n*x - 9*n) = 0 → (y^2 - 6*n*y - 9*n) = 0 → x = y) ↔
  n = 0 := by
  sorry

end quadratic_one_real_root_positive_n_l275_275376


namespace min_brilliant_triple_product_l275_275933

theorem min_brilliant_triple_product :
  ∃ a b c : ℕ, a > b ∧ b > c ∧ Prime a ∧ Prime b ∧ Prime c ∧ (a = b + 2 * c) ∧ (∃ k : ℕ, (a + b + c) = k^2) ∧ (a * b * c = 35651) :=
by
  sorry

end min_brilliant_triple_product_l275_275933


namespace range_of_m_l275_275681

def p (m : ℝ) : Prop :=
  ∀ (x : ℝ), 2^x - m + 1 > 0

def q (m : ℝ) : Prop :=
  5 - 2 * m > 1

theorem range_of_m (m : ℝ) (hpq : p m ∧ q m) : m ≤ 1 := sorry

end range_of_m_l275_275681


namespace product_of_roots_l275_275151

theorem product_of_roots : 
  (Real.root 81 4) * (Real.root 27 3) * (Real.sqrt 9) = 27 :=
by
  sorry

end product_of_roots_l275_275151


namespace length_AB_eq_four_l275_275324

variables {m n : ℝ} (A B : ℝ × ℝ)
def is_on_circle (p : ℝ × ℝ) (m n : ℝ) : Prop :=
  (p.1 - m)^2 + (p.2 - n)^2 = 9

variable (C : ℝ × ℝ := (m, n))

theorem length_AB_eq_four
  (hA : is_on_circle A m n)
  (hB : is_on_circle B m n)
  (distinct_points : A ≠ B)
  (h_mag : |⟨A.1 - m, A.2 - n⟩ + ⟨B.1 - m, B.2 - n⟩| = 2 * real.sqrt 5 )
  : dist A B = 4 :=
sorry

end length_AB_eq_four_l275_275324


namespace determine_m_l275_275110

noncomputable theory

def f (x m : ℝ) : ℝ := x^2 - 3 * x + m
def g (x m : ℝ) : ℝ := x^2 - 3 * x + 5 * m

theorem determine_m :
  (3 * f 5 m = 2 * g 5 m) → m = (10 / 7) :=
by 
  -- Proof steps would go here.
  sorry

end determine_m_l275_275110


namespace least_subtracted_number_l275_275535

theorem least_subtracted_number (N d : ℕ) (hN : N = 50248) (hd : d = 20) : 
  ∃ m, m = N % d ∧ (N - m) % d = 0 :=
by
  use N % d
  rw [hN, hd]
  sorry

end least_subtracted_number_l275_275535


namespace area_of_triangle_by_sin_sides_l275_275746

-- Problem setup:
variables {A B C : ℝ} -- angles
variables {a b c : ℝ} -- sides opposite to angles A, B, and C
variables (R : ℝ) (hR : R = 1) -- circumradius condition; unit circle implies R = 1
variables (hLawSines : (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) ∧ (c / Real.sin C = 2 * R))

-- Main goal:
theorem area_of_triangle_by_sin_sides {A B C : ℝ} {a b c : ℝ} : 
  let sin_a := Real.sin A,
      sin_b := Real.sin B,
      sin_c := Real.sin C in
  (a / sin_a = 2) ∧ (b / sin_b = 2) ∧ (c / sin_c = 2) → 
  (1/2 * a * b * Real.sin C ≠ 0) → -- ensure non-degenerate triangle condition
  (1/2 * (sin_a * sin_b * Real.sin C / 2) = (Real.sin A * Real.sin B * Real.sin C)) < (1/2 * a * b * Real.sin C / 2) := sorry

end area_of_triangle_by_sin_sides_l275_275746


namespace particle_final_position_l275_275916

def initial_position := (3, 0)
def rotation_radians := Real.pi / 6
def translation := (6, 0)
def number_of_moves := 120

def move (pos : ℂ) :=
  let ω := Complex.exp (Complex.I * rotation_radians)
  ω * pos + 6

noncomputable def final_position (initial_pos : ℂ) (moves : ℕ) :=
  (0 : ℕ).iterate (λ pos, move pos) initial_pos moves

theorem particle_final_position :
  final_position 3 number_of_moves = 3 := 
sorry

end particle_final_position_l275_275916


namespace sum_from_one_to_twelve_l275_275534

-- Define the sum of an arithmetic series
def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  (n * (a + l)) / 2

-- Theorem stating the sum of numbers from 1 to 12
theorem sum_from_one_to_twelve : sum_arithmetic_series 12 1 12 = 78 := by
  sorry

end sum_from_one_to_twelve_l275_275534


namespace count_valid_subsets_l275_275070

open Set

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def valid_subset (A : Set ℕ) : Prop :=
  (A ∩ {1, 2, 3} ≠ ∅) ∧ (A ∪ {4, 5, 6} ≠ S)

theorem count_valid_subsets : 
  {A : Set ℕ // ∀ A, A ⊆ S ∧ valid_subset A}.card = 888 :=
  sorry

end count_valid_subsets_l275_275070


namespace proper_divisor_rel_prime_l275_275089

open Nat

noncomputable def euler_totient (n : ℕ) : ℕ :=
  (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)).card

theorem proper_divisor_rel_prime {n : ℕ} (h1 : 3 < n) :
  ∃ m : ℕ, m ∣ (2 ^ euler_totient n - 1) ∧ m ≠ 1 ∧ Nat.coprime m n :=
by
  sorry

end proper_divisor_rel_prime_l275_275089


namespace cos_sub_sin_eq_l275_275783

noncomputable def θ_in_interval := {θ : Real // θ > π / 4 ∧ θ < π / 2}

theorem cos_sub_sin_eq :
  ∀ (θ : θ_in_interval),  sin (2 * θ.val) = 1 / 16 →
  cos θ.val - sin θ.val = - Real.sqrt 15 / 4 :=
by
  intros θ h
  sorry

end cos_sub_sin_eq_l275_275783


namespace part1_part2_l275_275309

-- Part (1): Given \( f(x) = x^3 - x \), \( g(x) = x^2 + a \), and \( x_1 = -1 \), prove that \( a = 3 \).
theorem part1 (a : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x)
  (hg : ∀ x, g x = x^2 + a)
  (x1 : ℝ) (hx1 : x1 = -1)
  (tangent_match : ∀ x, deriv f x = deriv g x) : a = 3 := 
sorry

-- Part (2): Given \( f(x) = x^3 - x \) and \( g(x) = x^2 + a \), prove that the range of values for \( a \) is \( [-1, +\infty) \).
theorem part2 (a : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x)
  (hg : ∀ x, g x = x^2 + a)
  (range_of_a : ∀ t : set ℝ, ∃ u : set ℝ, ∀ x ∈ t, a ∈ u) : a ∈ set.Ici (-1) :=
sorry

end part1_part2_l275_275309


namespace sum_of_products_divisible_by_47_l275_275415

theorem sum_of_products_divisible_by_47 :
  let S := ∑ i in (Finset.range 47).product (Finset.range 47), if i.1 ≠ i.2 then i.1 * i.2 else 0
  in 47 ∣ S := by
	begin
	sorry
	end=

end sum_of_products_divisible_by_47_l275_275415


namespace probability_case_7_probability_case_n_l275_275030

-- Define the scenario for n = 7
def probability_empty_chair_7 : Prop :=
  ∃ p : ℚ, p = 1 / 35

-- Theorem for the case when there are exactly 7 chairs
theorem probability_case_7 : probability_empty_chair_7 :=
begin
  -- Proof skipped
  sorry
end

-- Define the general scenario for n chairs (n >= 6)
def probability_empty_chair_n (n : ℕ) (h : n ≥ 6) : Prop :=
  ∃ p : ℚ, p = (n - 4) * (n - 5) / ((n - 1) * (n - 2))

-- Theorem for the case when there are n chairs (n ≥ 6)
theorem probability_case_n (n : ℕ) (h : n ≥ 6) : probability_empty_chair_n n h :=
begin
  -- Proof skipped
  sorry
end

end probability_case_7_probability_case_n_l275_275030


namespace decrease_percent_revenue_l275_275177

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let original_revenue := T * C
  let new_tax := 0.80 * T
  let new_consumption := 1.05 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 16 := 
by
  sorry

end decrease_percent_revenue_l275_275177


namespace find_primes_satisfying_condition_l275_275268

theorem find_primes_satisfying_condition :
  { p : ℕ | p.prime ∧ ∀ q : ℕ, q.prime ∧ q < p →
    ∀ k r : ℕ, p = k * q + r ∧ 0 ≤ r ∧ r < q → 
    ∀ a : ℕ, a > 1 → ¬ (a^2 ∣ r)} = {2, 3, 5, 7, 13} :=
begin
  sorry
end

end find_primes_satisfying_condition_l275_275268


namespace sum_of_angles_l275_275238

theorem sum_of_angles (θ : ℕ → ℝ) (k : ℕ) (n : ℕ) (r : ℝ) :
  -- Conditions
  (∀ k, z k = r * (Complex.cos (θ k) + Complex.sin (θ k) * Complex.I)) ∧
  (r > 0) ∧
  (∀ k, 0 ≤ θ(k) ∧ θ(k) < 360) →
  -- The sum of the angles
  θ(0) + θ(1) + θ(2) + θ(3) = 630 :=
sorry

end sum_of_angles_l275_275238


namespace sum_of_group_is_cube_l275_275885

theorem sum_of_group_is_cube {n : ℕ} (h : n ∈ {1, 2, 3, 4, 5}) : 
  let groups := [[1], [3, 5], [7, 9, 11], [13, 15, 17, 19], [21, 23, 25, 27, 29]] in
  sum of (groups.getOrElse (n-1) []) = n^3 :=
by sorry

end sum_of_group_is_cube_l275_275885


namespace part1_part2_l275_275438

open Set

variable {U A B : Set ℝ}

def U := Real
def A := {x : ℝ | 0 ≤ x ∧ x ≤ 3 }
def B (m : ℝ) := {x : ℝ | m - 2 ≤ x ∧ x ≤ 2 * m }

theorem part1 (m : ℝ) (hm : m = 3) : 
  A ∩ (U \ B m) = {x : ℝ | 0 ≤ x ∧ x < 1 } := 
by 
  sorry

theorem part2 :
  {m : ℝ | A ∪ B m = B m} = {m : ℝ | 3/2 ≤ m ∧ m ≤ 2} := 
by 
  sorry

end part1_part2_l275_275438


namespace find_fraction_l275_275567

theorem find_fraction
  (N : ℝ)
  (hN : N = 30)
  (h : 0.5 * N = (x / y) * N + 10):
  x / y = 1 / 6 :=
by
  sorry

end find_fraction_l275_275567


namespace length_of_chord_c3_common_chord_c1_c2_l275_275304

def C1 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 + 4 * p.2 - 4 = 0}
def C2 := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 + 2 * p.2 - 2 = 0}
def C3 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 - 2 * p.2 - 14 / 5 = 0}
def commonChord := {p : ℝ × ℝ | 2 * p.1 - p.2 + 1 = 0}

theorem length_of_chord_c3_common_chord_c1_c2 :
  (length_of_chord C3 commonChord) = 4 :=
sorry

end length_of_chord_c3_common_chord_c1_c2_l275_275304


namespace extreme_points_of_even_function_l275_275685

theorem extreme_points_of_even_function :
  ∀ f : ℝ → ℝ, (∀ x, f (-x) = f x) → (∀ x, 0 ≤ x → f x = (x^2 - 2 * x) * Real.exp x) →
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, ∃ c, (∀ y, y < x ∧ 0 ≤ y → f y < f x) ∧ (∀ y, y > x ∧ 0 ≤ y → f y > f x) :=
begin
  sorry
end

end extreme_points_of_even_function_l275_275685


namespace integer_with_18_factors_and_factors_18_24_l275_275841

theorem integer_with_18_factors_and_factors_18_24 (y : ℕ) :
  (nat.num_divisors y = 18) ∧ (18 ∣ y) ∧ (24 ∣ y) ↔ (y = 288) :=
by {
  sorry
}

end integer_with_18_factors_and_factors_18_24_l275_275841


namespace harly_dogs_final_count_l275_275719

theorem harly_dogs_final_count (initial_dogs : ℕ) (adopted_percentage : ℕ) (returned_dogs : ℕ) (adoption_rate : adopted_percentage = 40) (initial_count : initial_dogs = 80) (returned_count : returned_dogs = 5) :
  initial_dogs - (initial_dogs * adopted_percentage / 100) + returned_dogs = 53 :=
by
  sorry

end harly_dogs_final_count_l275_275719


namespace r_has_money_l275_275176

-- Define the variables and the conditions in Lean
variable (p q r : ℝ)
variable (h1 : p + q + r = 4000)
variable (h2 : r = (2/3) * (p + q))

-- Define the proof statement
theorem r_has_money : r = 1600 := 
  by
    sorry

end r_has_money_l275_275176


namespace expr_a_b_find_specific_set_possible_values_of_a_l275_275461

-- Definition 1: Verification of expressions for a and b
theorem expr_a_b (a b m n : ℤ) (h : a + b * real.sqrt 3 = (m + n * real.sqrt 3)^2) :
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n :=
sorry

-- Definition 2: Specific instances of m, n, leading to specific a, b
theorem find_specific_set :
  ∃ (a b m n : ℤ), a + b * real.sqrt 3 = (m + n * real.sqrt 3)^2 ∧ a = 7 ∧ b = 4 ∧ m = 2 ∧ n = 1 :=
begin
  use [7, 4, 2, 1],
  split, 
  { norm_num,
    have : 7 + 4 * real.sqrt 3 = (2 + 1 * real.sqrt 3)^2, by norm_num; ring,
    exact this },
  split, refl,
  split, refl,
  split, refl,
  split, refl
end

-- Definition 3: Finding values of a given m, n satisfying certain conditions
theorem possible_values_of_a (a m n : ℤ) (h : a + 6 * real.sqrt 3 = (m + n * real.sqrt 3)^2) (h2: 2 * m * n = 6) :
  a = 12 ∨ a = 28 :=
sorry

end expr_a_b_find_specific_set_possible_values_of_a_l275_275461


namespace problem_equiv_remainder_l275_275651

theorem problem_equiv_remainder :
  let N := 123456789012 in
  (N % 5 = 2) ∧
  (N % 3 = 0) ∧
  (N % 16 = 12) →
  (N % 240 = 132) :=
by
  intros N h5 h3 h16
  have h240 : N % 240 = 132 := sorry
  exact h240

end problem_equiv_remainder_l275_275651


namespace triangle_solution_l275_275742

noncomputable def find_b_and_area_of_triangle (A B C : ℝ) (a b c : ℝ) (S: ℝ) : Prop :=
  let sinA := (Real.sqrt 3) / 3
  let sinB := (Real.sqrt 6) / 3
  let sinC := 1 / 3
  (b = 3 * Real.sqrt 2) ∧ (S = (3 / 2) * Real.sqrt 2)

theorem triangle_solution :
  ∀ (A B C : ℝ) (a b c : ℝ) (S: ℝ),
    a = 3 →
    Real.cos A = (Real.sqrt 6) / 3 →
    B = A + Real.pi / 2 →
    find_b_and_area_of_triangle A B C a b c S :=
by {
  intros A B C a b c S ha hcosA hB,
  sorry
}

end triangle_solution_l275_275742


namespace trader_gain_l275_275953

-- Conditions
def cost_price (pen : Type) : ℕ → ℝ := sorry -- Type to represent the cost price of a pen
def selling_price (pen : Type) : ℕ → ℝ := sorry -- Type to represent the selling price of a pen
def gain_percentage : ℝ := 0.40 -- 40% gain

-- Statement of the problem to prove
theorem trader_gain (C : ℝ) (N : ℕ) : 
  (100 : ℕ) * C * gain_percentage = N * C → 
  N = 40 :=
by
  sorry

end trader_gain_l275_275953


namespace eleanor_overall_score_l275_275639

theorem eleanor_overall_score :
  let score1 := 0.7 * 15
  let score2 := 0.85 * 25
  let score3 := 0.75 * 35
  let total_correct := round(score1) + round(score2) + round(score3)
  let percentage := (total_correct / 75) * 100
  round(percentage) = 77 :=
by
  sorry

end eleanor_overall_score_l275_275639


namespace single_elimination_tournament_games_l275_275215

theorem single_elimination_tournament_games (n : ℕ) :
  n > 1 → (∑ i in Finset.range n, 1) - 1 = n - 1 :=
by
  sorry

end single_elimination_tournament_games_l275_275215


namespace Bill_threw_more_sticks_l275_275236

-- Definitions based on the given conditions
def Ted_sticks : ℕ := 10
def Ted_rocks : ℕ := 10
def Ted_double_Bill_rocks (R : ℕ) : Prop := Ted_rocks = 2 * R
def Bill_total_objects (S R : ℕ) : Prop := S + R = 21

-- The theorem stating Bill throws 6 more sticks than Ted
theorem Bill_threw_more_sticks (S R : ℕ) (h1 : Ted_double_Bill_rocks R) (h2 : Bill_total_objects S R) : S - Ted_sticks = 6 :=
by
  -- Definitions and conditions are loaded here
  sorry

end Bill_threw_more_sticks_l275_275236


namespace women_currently_in_room_l275_275013

variable (M W : ℕ)
variable (h_ratio : 7 * W = 9 * M)
variable (h_final_men : M + 8 = 28)

theorem women_currently_in_room : 
    let M := 20 in  
    let W := (9 * M) / 7 in  
    let W''' := 2 * (W - 2) in 
    W''' = 48 := 
by 
  let M := 20  
  let W := (9 * M) / 7 
  let W' := W - 6  
  let W'' := W' + 4 
  let W''' := 2 * (W'' - 2) 
  calc W = 180 / 7 := rfl
  calc W'DBL = 48 := 
(sorry
)

end women_currently_in_room_l275_275013


namespace running_time_approximation_l275_275905

def side_length : ℝ := 50
def grass_percentage : ℝ := 0.60
def sand_percentage : ℝ := 0.30
def mud_percentage : ℝ := 0.10

def speed_grass_kmh : ℝ := 14
def speed_sand_kmh : ℝ := 8
def speed_mud_kmh : ℝ := 5

def kmh_to_ms (speed : ℝ) : ℝ := speed * 1000 / 3600

def speed_grass_ms : ℝ := kmh_to_ms speed_grass_kmh
def speed_sand_ms : ℝ := kmh_to_ms speed_sand_kmh
def speed_mud_ms : ℝ := kmh_to_ms speed_mud_kmh

def perimeter : ℝ := 4 * side_length

def length_grass : ℝ := grass_percentage * perimeter
def length_sand : ℝ := sand_percentage * perimeter
def length_mud : ℝ := mud_percentage * perimeter

def time_grass : ℝ := length_grass / speed_grass_ms
def time_sand : ℝ := length_sand / speed_sand_ms
def time_mud : ℝ := length_mud / speed_mud_ms

def total_time : ℝ := time_grass + time_sand + time_mud

theorem running_time_approximation : |total_time - 72.27| < 0.1 := by
  sorry

end running_time_approximation_l275_275905


namespace shaded_area_l275_275864

-- Define the radius and central angle
def radius : ℝ := 10
def theta : ℝ := 30 * (Real.pi / 180)  -- converting degrees to radians

-- Function for the area of a sector given radius and angle in radians
def area_of_sector (r : ℝ) (θ : ℝ) : ℝ := (θ / (2 * Real.pi)) * Real.pi * r^2

-- Proof goal: The area of the sector given the conditions
theorem shaded_area : area_of_sector radius theta = 50 * Real.pi / 3 :=
by
  sorry

end shaded_area_l275_275864


namespace exists_irrationals_floor_neq_l275_275182

-- Define irrationality of a number
def irrational (x : ℝ) : Prop :=
  ¬ ∃ (r : ℚ), x = r

theorem exists_irrationals_floor_neq :
  ∃ (a b : ℝ), irrational a ∧ irrational b ∧ 1 < a ∧ 1 < b ∧ 
  ∀ (m n : ℕ), ⌊a ^ m⌋ ≠ ⌊b ^ n⌋ :=
by
  sorry

end exists_irrationals_floor_neq_l275_275182


namespace speed_in_still_water_correct_l275_275580

noncomputable def speed_in_still_water (speed_current : ℝ) (time_downstream : ℝ) (distance_downstream : ℝ) : ℝ :=
  let speed_current_ms := speed_current * 1000 / 3600
  let speed_downstream := distance_downstream / time_downstream
  let speed_still_water_ms := speed_downstream - speed_current_ms
  speed_still_water_ms * 3600 / 1000

theorem speed_in_still_water_correct :
  speed_in_still_water 2.5 35.99712023038157 90 ≈ 6.501251 :=
by
  sorry

end speed_in_still_water_correct_l275_275580


namespace proof_a2_plus_b2_eq_1_l275_275667

variable (a b : ℝ)
hypothesis (cond : a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = 1)

theorem proof_a2_plus_b2_eq_1 : a^2 + b^2 = 1 :=
sorry

end proof_a2_plus_b2_eq_1_l275_275667


namespace parallel_lines_l275_275426

-- Definitions based on conditions
variables (A B C O H M X N Y : Point)
variables (ω1 ω2 : Circle) 

-- Assume the necessary conditions from the problem
axiom circumcenter_triangle : ∀ (A B C : Point), Circumcenter A B C O
axiom orthocenter_triangle : ∀ (A B C : Point), Orthocenter A B C H
axiom circumcircle_BOC : Circumcircle B O C ω1
axiom circumcircle_BHC : Circumcircle B H C ω2
axiom intersect_circle_diameter_AO : Intersects (CircleDiameter A O) ω1 M 
axiom intersect_AM_omega1 : IntersectsLine (Line A M) ω1 X
axiom intersect_circle_diameter_AH : Intersects (CircleDiameter A H) ω2 N
axiom intersect_AN_omega2 : IntersectsLine (Line A N) ω2 Y

-- Statement to prove
theorem parallel_lines (A B C O H M X N Y : Point) (ω1 ω2 : Circle)
    [circumcenter_triangle A B C]
    [orthocenter_triangle A B C]
    [circumcircle_BOC B O C ω1]
    [circumcircle_BHC B H C ω2]
    [intersect_circle_diameter_AO (CircleDiameter A O) ω1 M]
    [intersect_AM_omega1 (Line A M) ω1 X]
    [intersect_circle_diameter_AH (CircleDiameter A H) ω2 N]
    [intersect_AN_omega2 (Line A N) ω2 Y] :
    Parallel (Line M N) (Line X Y) := 
sorry

end parallel_lines_l275_275426


namespace quadratic_vertex_l275_275112

theorem quadratic_vertex (a b c : ℤ)
  (h_vertex : ∀ x, (a * (x - 2)^2 - 3 = y) ↔ y = ax^2 + bx + c)
  (h_point1 : (0, 1) ∈ set_of (λ x y, y = ax^2 + bx + c))
  (h_point2 : (5, 6) ∈ set_of (λ x y, y = ax^2 + bx + c)) :
  a = 1 := 
sorry

end quadratic_vertex_l275_275112


namespace ratio_is_23_17_l275_275373

variables (c x y : ℝ)
theorem ratio_is_23_17 (h1 : x = 0.85 * c) (h2 : y = 1.15 * c) : y / x = 23 / 17 :=
begin
  sorry,
end

end ratio_is_23_17_l275_275373


namespace analytical_expression_and_minimum_value_A1_minimum_value_3_implies_a_value_l275_275287

noncomputable def f (x a : ℝ) : ℝ := x^2 + x + a^2 + a
noncomputable def g (x a : ℝ) : ℝ := x^2 - x + a^2 - a

def M (x a : ℝ) : ℝ := max (f x a) (g x a)

theorem analytical_expression_and_minimum_value_A1 :
  (∀ x : ℝ, a = 1 → M x a = if x ≥ -1 then x^2 + x + 2 else x^2 - x) ∧
  (∀ x : ℝ, a = 1 → M x a ≥ x^2 + x + 2 ∧ M x a ≥ x^2 - x ∧ (∀ x : ℝ, M x a ≥ 7 / 4)) :=
sorry

theorem minimum_value_3_implies_a_value (a : ℝ) :
  (∀ x : ℝ, M x a ≥ 3) → (a = (real.sqrt 14 - 1) / 2 ∨ a = -(real.sqrt 14 - 1) / 2) :=
sorry

end analytical_expression_and_minimum_value_A1_minimum_value_3_implies_a_value_l275_275287


namespace z_real_iff_z_complex_iff_z_pure_imaginary_iff_l275_275990

-- Define the complex number z
def z (a : ℝ) : ℂ := complex.mk (a ^ 2 - 1) (a + 1)

-- Statement for the value of a such that z is a real number
theorem z_real_iff (a : ℝ) : z a ∈ ℝ ↔ a = -1 := by
  sorry

-- Statement for the value of a such that z is a complex number
theorem z_complex_iff (a : ℝ) : z a ∈ ℂ ↔ a ≠ -1 := by
  sorry

-- Statement for the value of a such that z is a pure imaginary number
theorem z_pure_imaginary_iff (a : ℝ) : z a ∈ (λ (x : ℂ), x.im ≠ 0 ∧ x.re = 0) ↔ a = 1 := by
  sorry

end z_real_iff_z_complex_iff_z_pure_imaginary_iff_l275_275990


namespace certain_number_is_134_l275_275451

-- Define the given condition
def condition := (1206 / 3 = 3 * x)

-- Define the statement to prove
theorem certain_number_is_134 (x: ℕ) (h: condition): x = 134 :=
sorry

end certain_number_is_134_l275_275451


namespace total_female_officers_l275_275449

-- Definitions as conditions from step a)
def police_officers_on_duty : ℕ := 144
def percentage_female_on_duty : ℝ := 0.18
def female_officers_on_duty := police_officers_on_duty / 2

-- Statement to prove
theorem total_female_officers (h1 : percentage_female_on_duty * total_female_officers = female_officers_on_duty) : total_female_officers = 400 :=
by
  sorry

end total_female_officers_l275_275449


namespace estimate1_estimate2_estimate3_estimate4_l275_275263

theorem estimate1 : 100 * 70 = 7000 → 99 * 71 ≈ 7000 := by
  sorry

theorem estimate2 : 25 * 40 = 1000 → 25 * 39 ≈ 1000 := by
  sorry

theorem estimate3 : 120 / 3 = 40 → 124 / 3 ≈ 40 := by
  sorry

theorem estimate4 : 400 / 5 = 80 → 398 / 5 ≈ 80 := by
  sorry

end estimate1_estimate2_estimate3_estimate4_l275_275263


namespace totalBasketballs_proof_l275_275442

-- Define the number of balls for Lucca and Lucien
axiom numBallsLucca : ℕ := 100
axiom percBasketballsLucca : ℕ := 10
axiom numBallsLucien : ℕ := 200
axiom percBasketballsLucien : ℕ := 20

-- Calculate the number of basketballs they each have
def basketballsLucca := (percBasketballsLucca * numBallsLucca) / 100
def basketballsLucien := (percBasketballsLucien * numBallsLucien) / 100

-- Total number of basketballs
def totalBasketballs := basketballsLucca + basketballsLucien

theorem totalBasketballs_proof : totalBasketballs = 50 := by
  sorry

end totalBasketballs_proof_l275_275442


namespace problem1_problem2_l275_275614

-- Define Problem 1 statement
theorem problem1 : 
  (\sqrt 75 + \sqrt 27 - \sqrt (1/2) * \sqrt 12 + \sqrt 24 = 8 * \sqrt 3 + \sqrt 6) :=
sorry

-- Define Problem 2 statement
theorem problem2 : 
  ((\sqrt 3 + \sqrt 2) * (\sqrt 3 - \sqrt 2) - (\sqrt 5 - 1)^2 = 2 * \sqrt 5 - 5) :=
sorry

end problem1_problem2_l275_275614


namespace tan_A_l275_275750

-- Define the sides of the triangle and the right angle condition
variables (A B C : Type) [MetricSpace A B C]
variable (angle_BAC : Real)
variable (AB BC : ℝ)

-- Assume the given conditions
axiom BAC90 : angle_BAC = Real.pi / 2
axiom AB40 : AB = 40
axiom BC41 : BC = 41

-- Prove the value of tan A
theorem tan_A (A B C : Type) [MetricSpace A B C] (AB BC : ℝ) (angle_BAC : Real) (BAC90 : angle_BAC = Real.pi / 2)
    (AB40 : AB = 40) (BC41 : BC = 41) : 
    let AC := Real.sqrt (BC^2 - AB^2) in
    AC = 9 → Real.tan angle_BAC = 9 / 40 :=
sorry

end tan_A_l275_275750


namespace find_a2_b2_c2_l275_275261

-- Definitions and conditions
def circle_diameter : ℝ := 1
def circle_packed_in_first_quadrant := sorry  -- A detailed geometric arrangement would be complex to formalize
def union_of_circles (R : set point) := sorry  -- Define region R as union of the eight circles
def line_slope := 3
def line_divides_R_in_half := sorry  -- Line l divides R into two regions of equal area
def equation_form (a b c : ℤ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ Int.gcd (Int.gcd a b) c = 1

theorem find_a2_b2_c2 :
  ∃ a b c : ℤ, equation_form a b c ∧ (a^2 + b^2 + c^2 = 65) :=
by {
  sorry
}

end find_a2_b2_c2_l275_275261


namespace cosine_diagonal_angle_l275_275584

open Real

def vector_a : ℝ × ℝ × ℝ := (3, 2, 1)
def vector_b : ℝ × ℝ × ℝ := (2, -1, -1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2)

def angle_cosine (u v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cosine_diagonal_angle :
  angle_cosine (3 + 2, 2 - 1, 1 - 1) (2 - 3, -1 - 2, -1 - 1) = -4 / sqrt 91 :=
by
  sorry

end cosine_diagonal_angle_l275_275584


namespace fixed_point_of_line_l275_275490

theorem fixed_point_of_line (m : ℝ) : ∃ (x y : ℝ), mx - y + 2m + 1 = 0 ∧ x = -2 ∧ y = 1 :=
by {
  use [-2, 1],
  sorry
}

end fixed_point_of_line_l275_275490


namespace smallest_sum_of_two_distinct_primes_greater_than_500_l275_275873

   /-- Proving the smallest integer that is the sum of two distinct prime integers,
       each greater than 500, is 1012 --/
   theorem smallest_sum_of_two_distinct_primes_greater_than_500 : 
     ∃ (p1 p2 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ p1 ≠ p2 ∧ 500 < p1 ∧ 500 < p2 ∧ p1 + p2 = 1012 :=
   by
     sorry
   
end smallest_sum_of_two_distinct_primes_greater_than_500_l275_275873


namespace find_pairs_l275_275965

theorem find_pairs (x y : ℕ) (h : x > 0 ∧ y > 0) (d : ℕ) (gcd_cond : d = Nat.gcd x y)
  (eqn_cond : x * y * d = x + y + d ^ 2) : (x, y) = (2, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (3, 2) :=
by {
  sorry
}

end find_pairs_l275_275965


namespace units_digit_of_sum_factorials_49_l275_275781

-- Define a function to compute the factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the sum of factorials from 1 to n
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, factorial (k + 1))

-- Define the units digit function
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- The main theorem: the units digit of the sum of factorials from 1 to 49 is 3
theorem units_digit_of_sum_factorials_49 : units_digit (sum_factorials 49) = 3 := sorry

end units_digit_of_sum_factorials_49_l275_275781


namespace tangent_length_l275_275245

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (2, 3)
noncomputable def B : ℝ × ℝ := (4, 6)
noncomputable def C : ℝ × ℝ := (3, 10)

-- We need to show that the tangent length from O to the circumcircle of triangle ABC is √26
theorem tangent_length :
  let OA := Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2),
      OB := Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2)
  in Real.sqrt (OA * OB) = Real.sqrt 26 :=
by
  sorry

end tangent_length_l275_275245


namespace subtraction_result_l275_275900

noncomputable def division_value : ℝ := 1002 / 20.04

theorem subtraction_result : 2500 - division_value = 2450.0499 :=
by
  have division_eq : division_value = 49.9501 := by sorry
  rw [division_eq]
  norm_num

end subtraction_result_l275_275900


namespace proof_of_statement_l275_275603

noncomputable def problem_statement (Γ₁ Γ₂ : Circle) (E F A B C D G : Point) (AD BC : ℝ) (angleAFB : ℝ) : Prop :=
∃ (lineAD lineBC lineBD : Line),
  (∃ (F_on_lineAD : PointOnLine F lineAD) (A_on_lineAD : PointOnCircle A Γ₁) (D_on_lineAD : PointOnCircle D Γ₂),
  ∃ (F_on_lineBC : PointOnLine F lineBC) (B_on_lineBC : PointOnCircle B Γ₁) (C_on_lineBC : PointOnCircle C Γ₂),
  AD = BC ∧ angleAFB = 60 ∧
  (∃ (lineCG : Line), Parallel lineCG (LineFromPoints A B) ∧
  ∃ (G_on_lineBD : PointOnLine G lineBD),
  (∃ (BD_intersects_CG : IntersectionExists (LineFromPoints B D) lineCG),
   ∀ (EG : Line), Parallel EG (LineFromPoints A D)))

theorem proof_of_statement (Γ₁ Γ₂ : Circle) (E F A B C D G : Point) (AD BC : ℝ) (angleAFB : ℝ)
  (h1 : CirclesIntersect Γ₁ Γ₂ E F)
  (h2 : LineSegmentPassesThrough F A D lineAD)
  (h3 : LineSegmentPassesThrough F B C lineBC)
  (h4 : PointOnCircle A Γ₁)
  (h5 : PointOnCircle B Γ₁)
  (h6 : PointOnCircle C Γ₂)
  (h7 : PointOnCircle D Γ₂)
  (h8 : AD = BC)
  (h9 : angleAFB = 60)
  (h10 : LineThroughParallel C A B G)
  (h11 : LineIntersect BD G C) :
  EG_parallel_AD : Proof(EG, LineFromPoints(E, G), LineFromPoints(A, D)) := by
  sorry

end proof_of_statement_l275_275603


namespace sum_of_digits_1_to_5000_l275_275246

def sum_of_digits (n : ℕ) : ℕ :=
nat.digits 10 n |>.sum

def sequence_sum_of_digits (a b : ℕ) : ℕ :=
(list.range' a (b - a + 1)).map sum_of_digits |>.sum

theorem sum_of_digits_1_to_5000 :
  sequence_sum_of_digits 1 5000 = 167450 := 
sorry

end sum_of_digits_1_to_5000_l275_275246


namespace determine_y_l275_275839

theorem determine_y (y : ℕ) (h1 : ∀ d : ℕ, d ∣ y → 0 < d → d ∈ [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 48, 72, 144, 288])
  (h2 : 18 ∣ y) (h3 : 24 ∣ y) (h4 : 18 = (factors y).length) : y = 288 :=
by sorry

end determine_y_l275_275839


namespace soccer_team_arrangements_l275_275725

theorem soccer_team_arrangements : 
  ∃ (n : ℕ), n = 2 * (Nat.factorial 11)^2 := 
sorry

end soccer_team_arrangements_l275_275725


namespace dimes_count_l275_275170

def num_dimes (total_in_cents : ℤ) (value_quarter value_dime value_nickel : ℤ) (num_each : ℤ) : Prop :=
  total_in_cents = num_each * (value_quarter + value_dime + value_nickel)

theorem dimes_count (num_each : ℤ) :
  num_dimes 440 25 10 5 num_each → num_each = 11 :=
by sorry

end dimes_count_l275_275170


namespace jane_cans_l275_275078

theorem jane_cans (total_seeds : ℕ) (seeds_per_can : ℕ) : total_seeds = 54 → seeds_per_can = 6 → total_seeds / seeds_per_can = 9 :=
by
  intros h1 h2
  rw [h1, h2]
  rfl

end jane_cans_l275_275078


namespace terry_spent_for_breakfast_on_monday_l275_275476

theorem terry_spent_for_breakfast_on_monday : 
  ∃ (x : ℕ), (x + 2 * x + 6 * x = 54) ∧ x = 6 :=
by
  use 6
  sorry

end terry_spent_for_breakfast_on_monday_l275_275476


namespace minimize_J_l275_275362

noncomputable def H (p q : ℝ) := -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) := max (9 * p - 5) (4 - 7 * p)

theorem minimize_J :
  ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ ∀ p' : ℝ, 0 ≤ p' ∧ p' ≤ 1 → J p ≤ J p' :=
begin
  use (9 / 16),
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros p' hp,
    sorry
  }

end minimize_J_l275_275362


namespace eight_digits_placement_l275_275391

theorem eight_digits_placement : ∃ (t : list (list ℕ)), 
  (∀ r ∈ t.tail, list.sum r = list.sum (t.head) + ((t.index_of r) - 1)) ∧ 
  (∀ r ∈ t, r.length = 2) ∧ 
  length t = 4 ∧ 
  ∀ d ∈ t.join, 1 ≤ d ∧ d ≤ 9 ∧ 
  length (t.join.erase d) = 8 → 
  (count_total_arrangements t = 64) :=
begin
  sorry
end

end eight_digits_placement_l275_275391


namespace find_naturals_l275_275974

theorem find_naturals (n : ℕ) (k : ℤ) (h : 2^8 + 2^{11} + 2^n = k^2) : n = 12 :=
sorry

end find_naturals_l275_275974


namespace total_charge_for_trip_l275_275018

noncomputable def calc_total_charge (initial_fee : ℝ) (additional_charge : ℝ) (miles : ℝ) (increment : ℝ) :=
  initial_fee + (additional_charge * (miles / increment))

theorem total_charge_for_trip :
  calc_total_charge 2.35 0.35 3.6 (2 / 5) = 8.65 :=
by
  sorry

end total_charge_for_trip_l275_275018


namespace minimize_J_l275_275364

def H (p q : ℝ) : ℝ := -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

def J (p : ℝ) : ℝ := max (H p 0) (H p 1)

theorem minimize_J : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ p = 9 / 16 ∧ (∀ q, 0 ≤ q ∧ q ≤ 1 → J p ≤ J q) :=
by
  sorry

end minimize_J_l275_275364


namespace N_properties_l275_275649

def N : ℕ := 3625

theorem N_properties :
  (N % 32 = 21) ∧ (N % 125 = 0) ∧ (N^2 % 8000 = N % 8000) :=
by
  sorry

end N_properties_l275_275649


namespace cycling_time_to_library_l275_275726

-- Define the conditions as constants
constant time_to_park : ℝ
constant distance_to_park : ℝ
constant distance_to_library : ℝ
constant constant_speed : Prop

-- Set the given constants
axiom time_to_park_axiom : time_to_park = 30
axiom distance_to_park_axiom : distance_to_park = 5
axiom distance_to_library_axiom : distance_to_library = 3
axiom constant_speed_axiom : constant_speed = true

-- Define the desired result as a proposition
noncomputable def time_to_library : ℝ := 18

-- Prove the proposition
theorem cycling_time_to_library : (constant_speed → (time_to_library = (time_to_park / distance_to_park) * distance_to_library)) :=
by {
  rw [time_to_park_axiom, distance_to_park_axiom, distance_to_library_axiom],
  sorry
}

end cycling_time_to_library_l275_275726


namespace bullet_train_pass_man_in_approximately_twleve_seconds_l275_275571

noncomputable def bullet_train_length : ℝ := 220 -- in meters

noncomputable def bullet_train_speed : ℝ := 59 -- in kmph

noncomputable def man_speed : ℝ := 7 -- in kmph

noncomputable def kmph_to_mps (speed : ℝ) : ℝ := speed * (5 / 18) -- Conversion from kmph to m/s

noncomputable def relative_speed : ℝ := kmph_to_mps (bullet_train_speed + man_speed) -- Relative speed in m/s

noncomputable def passing_time : ℝ := bullet_train_length / relative_speed -- Time in seconds

theorem bullet_train_pass_man_in_approximately_twleve_seconds : abs (passing_time - 12) < 0.1 :=
begin
  sorry
end

end bullet_train_pass_man_in_approximately_twleve_seconds_l275_275571


namespace problem1_problem2_l275_275318

noncomputable theory

-- Given functions
def f (x : ℝ) := x^3 - x
def g (x : ℝ) (a : ℝ) := x^2 + a

-- Problem (1): If x1 = -1, show that a = 3
theorem problem1 (a : ℝ) (h : tangent_line_at f (-1) = tangent_line_at (g (-1)) (f (-1))) :
  a = 3 := sorry

-- Lean theorem to find the range of values for a such that tangent line conditions hold
theorem problem2 (x₁ : ℝ) (a : ℝ) (h : tangent_line_at f x₁ = tangent_line_at g x₁ a) :
  a ≥ -1 := sorry

end problem1_problem2_l275_275318


namespace tranquility_understanding_l275_275521

-- Define the painter's depiction condition
def depiction_of_tranquility (p : Painter) : Prop :=
  p.painting = "waterfall rushing down, with a small tree beside it, a bird's nest on the tree, and a sleeping bird inside the nest"

-- Define the true understanding condition
def true_understanding (p : Painter) : Prop :=
  unity_of_absolute_motion_and_relative_stillness p

-- Define the statements ② and ④
def statement_2 : Prop :=
  ∀ (p : Painter), (true_understanding p → sides_of_contradiction_are_opposing_and_unified p)

def statement_4 : Prop :=
  ∀ (p : Painter), (true_understanding p → nature_of_struggle_embedded_within_unity p)

-- Define Lean 4 statement for the proof problem
theorem tranquility_understanding (p : Painter) (dep : depiction_of_tranquility p) 
  (tr : true_understanding p) : 
  statement_2 p ∧ statement_4 p := by
  sorry

end tranquility_understanding_l275_275521


namespace factorize_expression_l275_275265

theorem factorize_expression (R : Type*) [CommRing R] (m n : R) : 
  m^2 * n - n = n * (m + 1) * (m - 1) := 
sorry

end factorize_expression_l275_275265


namespace find_number_l275_275503

theorem find_number (x : ℝ) (h : 10 * x = 2 * x - 36) : x = -4.5 :=
sorry

end find_number_l275_275503


namespace total_weight_of_7_moles_CaO_l275_275530

/-- Definitions necessary for the problem --/
def atomic_weight_Ca : ℝ := 40.08 -- atomic weight of calcium in g/mol
def atomic_weight_O : ℝ := 16.00 -- atomic weight of oxygen in g/mol
def molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O -- molecular weight of CaO in g/mol
def number_of_moles_CaO : ℝ := 7 -- number of moles of CaO

/-- The main theorem statement --/
theorem total_weight_of_7_moles_CaO :
  molecular_weight_CaO * number_of_moles_CaO = 392.56 :=
by
  sorry

end total_weight_of_7_moles_CaO_l275_275530


namespace sum_of_cubes_eq_219_75_l275_275487

def cubic_roots_eq : Prop :=
  ∃ a b c : ℝ,
    (a - (23 : ℝ).cbrt) * (b - (73 : ℝ).cbrt) * (c - (123 : ℝ).cbrt) = 1 / 4 ∧
    (a + b + c = (23 : ℝ).cbrt + (73 : ℝ).cbrt + (123 : ℝ).cbrt) ∧
    (a * b + a * c + b * c = (23 : ℝ).cbrt * (73 : ℝ).cbrt + (23 : ℝ).cbrt * (123 : ℝ).cbrt + (73 : ℝ).cbrt * (123 : ℝ).cbrt) ∧
    (a * b * c = (23 : ℝ).cbrt * (73 : ℝ).cbrt * (123 : ℝ).cbrt + 1 / 4)

theorem sum_of_cubes_eq_219_75 (a b c : ℝ) (h : cubic_roots_eq) : a^3 + b^3 + c^3 = 219.75 :=
sorry

end sum_of_cubes_eq_219_75_l275_275487


namespace relationship_among_a_b_c_l275_275629

noncomputable def f : ℝ → ℝ := λ x, x^3 * real.sin x
def a : ℝ := f (real.sin (real.pi / 3))
def b : ℝ := f (real.sin 2)
def c : ℝ := f (real.sin 3)

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by {
  -- Proof not required
  sorry
}

end relationship_among_a_b_c_l275_275629


namespace probability_of_selecting_digit_2_l275_275448

def repeating_decimal_rep (n d : ℕ) := "27" -- Represent repeating block '27' for 3/11.

theorem probability_of_selecting_digit_2 :
  let block := repeating_decimal_rep 3 11 in
  block = "27" → 
  (1 / (String.length block) : ℚ) * (String.count block '2') = 1 / 2 :=
by 
  intro block_eq
  sorry

end probability_of_selecting_digit_2_l275_275448


namespace matrix_vector_computation_l275_275062

variable (N : Matrix (Fin 2) (Fin 2) ℝ)
variable (a b : Vector (Fin 2) ℝ)

theorem matrix_vector_computation (h1 : N.mul_vec a = ![2, -3])
    (h2 : N.mul_vec b = ![5, 4]) :
    N.mul_vec (3 • a - 2 • b) = ![-4, -17] := sorry

end matrix_vector_computation_l275_275062


namespace sequence_general_term_correctness_l275_275111

def sequenceGeneralTerm (n : ℕ) : ℤ :=
  if n % 2 = 1 then
    0
  else
    (-1) ^ (n / 2 + 1)

theorem sequence_general_term_correctness (n : ℕ) :
  (∀ m, sequenceGeneralTerm m = 0 ↔ m % 2 = 1) ∧
  (∀ k, sequenceGeneralTerm k = (-1) ^ (k / 2 + 1) ↔ k % 2 = 0) :=
by
  sorry

end sequence_general_term_correctness_l275_275111


namespace value_of_unknown_number_l275_275899

theorem value_of_unknown_number (x n : ℤ) 
  (h1 : x = 88320) 
  (h2 : x + n + 9211 - 1569 = 11901) : 
  n = -84061 :=
by
  sorry

end value_of_unknown_number_l275_275899


namespace s_l275_275452

def graham_cracker_cost (individual_cost pack_cost : ℝ) (pack_size needed : ℕ) : ℝ :=
  let packs := (needed : ℝ) / (pack_size : ℝ)
  in (packs.ceil : ℝ) * pack_cost

def total_cost_s'mores_night : ℝ :=
  let num_people := 8
  let s'mores_per_person := 3

  let graham_crackers_needed := num_people * s'mores_per_person * 1
  let marshmallows_needed := num_people * s'mores_per_person * 1
  let chocolate_pieces_needed := num_people * s'mores_per_person * 1
  let caramel_pieces_needed := num_people * s'mores_per_person * 2
  let toffee_pieces_needed := num_people * s'mores_per_person * 4

  let graham_crackers_pack_cost := 1.80
  let graham_crackers_pack_size := 20

  let marshmallows_pack_cost := 2.00
  let marshmallows_pack_size := 15

  let chocolate_pieces_pack_cost := 2.00
  let chocolate_pieces_pack_size := 10

  let caramel_pieces_pack_cost := 4.50
  let caramel_pieces_pack_size := 25

  let toffee_pieces_pack_cost := 2.00
  let toffee_pieces_pack_size := 50

  let graham_crackers_cost := graham_cracker_cost 0.10 graham_crackers_pack_cost graham_crackers_pack_size graham_crackers_needed
  let marshmallows_cost := graham_cracker_cost 0.15 marshmallows_pack_cost marshmallows_pack_size marshmallows_needed
  let chocolate_pieces_cost := graham_cracker_cost 0.25 chocolate_pieces_pack_cost chocolate_pieces_pack_size chocolate_pieces_needed
  let caramel_pieces_cost := graham_cracker_cost 0.20 caramel_pieces_pack_cost caramel_pieces_pack_size caramel_pieces_needed
  let toffee_pieces_cost := graham_cracker_cost 0.05 toffee_pieces_pack_cost toffee_pieces_pack_size toffee_pieces_needed

  graham_crackers_cost + marshmallows_cost + chocolate_pieces_cost + caramel_pieces_cost + toffee_pieces_cost

theorem s'mores_night_cost : total_cost_s'mores_night = 26.60 :=
  by
    sorry

end s_l275_275452


namespace quadratic_root_a_l275_275278

theorem quadratic_root_a {a : ℝ} (h : (2 : ℝ) ∈ {x : ℝ | x^2 + 3 * x + a = 0}) : a = -10 :=
by
  sorry

end quadratic_root_a_l275_275278


namespace speed_while_going_up_l275_275581

theorem speed_while_going_up (V_down V_avg : ℝ) (hV_down : V_down = 36) (hV_avg : V_avg = 28.8) :
  let V_up := (V_avg * V_down) / (2 * V_avg - V_down)
  in V_up = 24 :=
by sorry

end speed_while_going_up_l275_275581


namespace product_of_points_l275_275624

def g (n : ℕ) : ℕ := 
  if n % 6 = 0 then 8
  else if n % 3 = 0 then 4
  else if n % 2 = 0 then 3
  else 1

def corasRolls : List ℕ := [5, 4, 3, 6, 2, 1]
def danasRolls : List ℕ := [6, 3, 4, 3, 5, 3]

def totalPoints (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem product_of_points : totalPoints corasRolls * totalPoints danasRolls = 480 := by
  sorry

end product_of_points_l275_275624


namespace vert_asymptotes_count_l275_275723

def f (x : ℝ) : ℝ := (x^2 - 1) / (x^3 + 6*x^2 - 7*x)

theorem vert_asymptotes_count : ∃ (n : ℕ), n = 2 ∧ ∀ x : ℝ, (x = 0 ∨ x = -7) → (∃ ε > 0, ∀ δ > 0, 0 < |x - δ| < ε → abs (f (x + δ)) > 1) :=
sorry

end vert_asymptotes_count_l275_275723


namespace area_of_EPGQ_l275_275462

noncomputable def area_of_region (length_rect width_rect half_length_rect : ℝ) : ℝ :=
  half_length_rect * width_rect

theorem area_of_EPGQ :
  let length_rect := 10.0
  let width_rect := 6.0
  let P_half_length := length_rect / 2
  let Q_half_length := length_rect / 2
  (area_of_region length_rect width_rect P_half_length) = 30.0 :=
by
  sorry

end area_of_EPGQ_l275_275462


namespace roots_of_fx_l275_275302

theorem roots_of_fx (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (x) + f (x + 2) = a) ∧ (f 1 = 0) ∧ (∀ x, f (-x) = -f (x)) →
  (set.filter (λ x, f x = 0) (set.Ioo (-3:ℝ) 7)).card = 9 :=
by sorry

end roots_of_fx_l275_275302


namespace probability_of_selecting_product_at_least_4_l275_275458

theorem probability_of_selecting_product_at_least_4 : 
  let products := {1, 2, 3, 4, 5} in
  let favorable_products := {x ∈ products | x ≥ 4} in
  let probability := (favorable_products.card : ℝ) / (products.card : ℝ) in
  probability = 2 / 5 :=
by
  let products := {1, 2, 3, 4, 5}
  let favorable_products := {x ∈ products | x ≥ 4}
  have cardinality_products : products.card = 5 := sorry
  have cardinality_favorable_products : favorable_products.card = 2 := sorry
  have h : (2 : ℝ) / (5 : ℝ) = 2 / 5 := by norm_num
  have probability := (favorable_products.card : ℝ) / (products.card : ℝ)
  rw [cardinality_products, cardinality_favorable_products] at probability
  exact (Eq.trans probability h).symm

end probability_of_selecting_product_at_least_4_l275_275458


namespace union_of_intervals_l275_275436

open Set

variable {α : Type*}

theorem union_of_intervals : 
  let A := Ioc (-1 : ℝ) 1
  let B := Ioo (0 : ℝ) 2
  A ∪ B = Ioo (-1 : ℝ) 2 := 
by
  let A := Ioc (-1 : ℝ) 1
  let B := Ioo (0 : ℝ) 2
  sorry

end union_of_intervals_l275_275436


namespace periodic_minus_decimal_is_correct_l275_275237

-- Definitions based on conditions

def periodic_63_as_fraction : ℚ := 63 / 99
def decimal_63_as_fraction : ℚ := 63 / 100
def difference : ℚ := periodic_63_as_fraction - decimal_63_as_fraction

-- Lean 4 statement to prove the mathematically equivalent proof problem
theorem periodic_minus_decimal_is_correct :
  difference = 7 / 1100 :=
by
  sorry

end periodic_minus_decimal_is_correct_l275_275237


namespace probability_not_above_x_axis_l275_275808

-- Definition of points as required by the conditions given.
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨4, 4⟩
def B : Point := ⟨-2, -2⟩
def C : Point := ⟨-8, -2⟩
def D : Point := ⟨0, 4⟩

-- The main theorem to be proved
theorem probability_not_above_x_axis (A B C D : Point) :
  let total_area := (1/2 : ℝ) * ((real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2) + real.sqrt ((C.x - B.x)^2 + (C.y - B.y)^2)) * 4 + (real.sqrt ((D.x - C.x)^2 + (D.y - C.y)^2) + real.sqrt ((A.x - D.x)^2 + (A.y - D.y)^2)) * 2)
  in 
  let area_EBCF := (1/2 : ℝ) * (real.sqrt ((C.x - B.x)^2 + (C.y - B.y)^2) + real.sqrt ((E.x - F.x)^2 + (E.y - F.y)^2)) * 2
  in 
  area_EBCF / total_area = (1 / 3 : ℝ) :=
sorry

end probability_not_above_x_axis_l275_275808


namespace lcm_9_15_lcm_10_21_gcd_50_203_final_result_l275_275269

-- Define the LCM function
def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

-- LCM calculations
theorem lcm_9_15 : lcm 9 15 = 45 := sorry
theorem lcm_10_21 : lcm 10 21 = 210 := sorry

-- Given numbers after modification
def num1 := 45 + 5
def num2 := 210 - 7

-- GCD Calculation
theorem gcd_50_203 : Nat.gcd 50 203 = 1 := sorry

-- Final statement
theorem final_result : Nat.gcd (lcm 9 15 + 5) (lcm 10 21 - 7) = 1 := 
by 
  rw [lcm_9_15, lcm_10_21]
  exact gcd_50_203

end lcm_9_15_lcm_10_21_gcd_50_203_final_result_l275_275269


namespace area_ACE_proof_l275_275233

/-
Given a trapezoid ABCD, the areas of triangles ADE, ABF, and BCF are given.
We need to prove that the area of triangle ACE is 8.
-/

variable {A B C D E F : Type*}
variable [trapezoid : Trapezoid A B C D]
variable (area : ∀ {X Y Z : Type*} [Triangle X Y Z], ℕ)
variable (ADE : Triangle A D E) (ABF : Triangle A B F) (BCF : Triangle B C F) (ACE : Triangle A C E)

-- Given conditions
axiom h1 : area ADE = 1
axiom h2 : area ABF = 9
axiom h3 : area BCF = 27

-- The goal
theorem area_ACE_proof : area ACE = 8 := 
sorry

end area_ACE_proof_l275_275233


namespace operational_probability_invariant_l275_275619

-- Define the network and conditions.
structure Network (α : Type) :=
  (servers : set α)
  (channels : α → α → Prop)
  (channel_fails_prob : α → α → ℝ)
  (independent_failures : ∀ u v, u ≠ v → channel_fails_prob u v ∈ set.Icc 0 1)

-- Define operational network.
def operational {α : Type} (N : Network α) (r : α) : Prop :=
  ∀ (s ∈ N.servers), ∃ (path : list α), path.head = some s ∧ path.last = some r ∧ ∀ (u v : α), list.in_pairs path (u, v) → ¬N.channel_fails_prob u v = 1

-- Main theorem statement in Lean.
theorem operational_probability_invariant {α : Type} (N : Network α) (r1 r2 : α) 
  (hp : ∀ u v, u ≠ v → N.channel_fails_prob u v ∈ set.Icc 0 1) :
  ∀ p, (0 ≤ p ∧ p ≤ 1) →
  ((probability (operational N r1) = probability (operational N r2))) :=
sorry

end operational_probability_invariant_l275_275619


namespace eccentricity_of_hyperbola_l275_275673

theorem eccentricity_of_hyperbola
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : ∃ (x y : ℝ), x^2 + y^2 - 4 * x + 6 * y = 0 
        ∧ (y / x = b / a ∨ y / x = -b / a)){
  eccentricity : ℝ :=
  begin
    sorry
  end
}

end eccentricity_of_hyperbola_l275_275673


namespace ratio_of_areas_l275_275057

variables {P : Type*} [InnerProductSpace ℝ P]
variables (D E F Q : P)

-- Given condition
def Q_condition := (Q - D) + (3 : ℝ) • (Q - E) + (2 : ℝ) • (Q - F) = 0

-- Areas definition
def area (A B C : P) : ℝ := sorry  -- Assuming area as a function (further proof of this function is needed)

-- The proof statement
theorem ratio_of_areas (h : Q_condition D E F Q) :
  (area D E F) / (area D Q F) = 3 :=
sorry

end ratio_of_areas_l275_275057


namespace root_product_l275_275145

theorem root_product : (Real.sqrt (Real.sqrt 81) * Real.cbrt 27 * Real.sqrt 9 = 27) :=
by
  sorry

end root_product_l275_275145


namespace ratio_of_areas_l275_275418

-- Define the equilateral triangle and the extensions
structure Triangle (α : Type*) := 
(A B C : α)
(x : α) -- the side length of the equilateral triangle
(area_ABC : α) -- the area of the original triangle
(area_A_B_C_ : α) -- the area of the extended triangle

noncomputable def area (side_length : ℝ) : ℝ :=
  (sqrt 3 / 4) * side_length ^ 2

-- Conditions: extensions of the sides
noncomputable def extension_side_length (side_length : ℝ) := 3 * side_length

-- Define the original and extended triangles
def original_triangle (side_length : ℝ) : ℝ := area side_length
def extended_triangle (side_length : ℝ) : ℝ := area (extension_side_length side_length)

-- Prove that the ratio of the areas is 9
theorem ratio_of_areas (side_length : ℝ) (h : side_length ≠ 0) :
  extended_triangle side_length / original_triangle side_length = 9 :=
by
  sorry

end ratio_of_areas_l275_275418


namespace color_changes_intermediate_l275_275909

theorem color_changes_intermediate (initial_changes : ℕ) (final_changes : ℕ) :
  initial_changes = 46 → final_changes = 26 →
  (∃ (n : ℕ), n % 2 = 0 ∧ initial_changes > n ∧ n > final_changes ∧ n = 28) :=
begin
  intros h_initial h_final,
  -- Write the proof here. For now we use sorry to skip the proof.
  sorry
end

end color_changes_intermediate_l275_275909


namespace plane_ABC_through_X_max_area_triangle_ABC_l275_275805

-- Defining points on the sphere and related conditions
variables (S : Point) (P : Point) (A B C : Point)
variable {radius : ℝ := 1}
variable [IsSphere S radius]
variable [PointsOnSphere P S radius]
variable [PointsOnSphere A S radius]
variable [PointsOnSphere B S radius]
variable [PointsOnSphere C S radius]
variable hABC_perpendicular : (ArePerpendicular P A B) ∧ (ArePerpendicular P B C) ∧ (ArePerpendicular P C A)

-- Define the fixed point
noncomputable def X : Point := S + 1/3 * (P - S)

-- Statement of the problem
theorem plane_ABC_through_X :
  ∀ (plane_ABC : Plane), planeContains plane_ABC P A B C → planeContains plane_ABC X :=
by sorry

theorem max_area_triangle_ABC :
  ∃ (area : ℝ), area = 1 :=
by sorry

end plane_ABC_through_X_max_area_triangle_ABC_l275_275805


namespace cost_of_grapes_and_orange_l275_275403

variables (p g o ch : ℝ)
variables (h1 : p + g + o + ch = 25)
variables (h2 : 2 * p = ch)
variables (h3 : p - g = o)

theorem cost_of_grapes_and_orange :
  g + o = 6.25 :=
by
  have h4 : o = p - g, from h3,
  have h5 : ch = 2 * p, from h2,
  have eq1 : p + g + (p - g) + 2 * p = 25, 
  {
    rw [←h4, ←h5, h1],
  },
  have h6 : 4 * p = 25, 
  {
    rw eq1,
  },
  have h7 : p = 25 / 4, 
  {
    norm_num,
    rw h6,
  },
  rw [←h4, h7],
  linarith,
  sorry

end cost_of_grapes_and_orange_l275_275403


namespace completing_the_square_transformation_l275_275539

theorem completing_the_square_transformation (x : ℝ) : 
  (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro h
  -- Transformation steps will be shown in the proof
  sorry

end completing_the_square_transformation_l275_275539


namespace distribution_equiv_implies_constants_l275_275785

open ProbabilityTheory

noncomputable section

theorem distribution_equiv_implies_constants (ξ : ℝ → ℝ) 
  (hnd : ¬∀ x, ξ x = ξ 0) -- ξ is non-degenerate
  (a : ℝ) (ha : a > 0) (b : ℝ)
  (h : ∀ x, ξ x = ξ (a * x + b)) : a = 1 ∧ b = 0 :=
by
  sorry

end distribution_equiv_implies_constants_l275_275785


namespace range_of_x_for_positive_function_value_l275_275693

variable {R : Type*} [LinearOrderedField R]

def even_function (f : R → R) := ∀ x, f (-x) = f x

def monotonically_decreasing_on_nonnegatives (f : R → R) := ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem range_of_x_for_positive_function_value (f : R → R)
  (hf_even : even_function f)
  (hf_monotonic : monotonically_decreasing_on_nonnegatives f)
  (hf_at_2 : f 2 = 0)
  (hf_positive : ∀ x, f (x - 1) > 0) :
  ∀ x, -1 < x ∧ x < 3 := sorry

end range_of_x_for_positive_function_value_l275_275693


namespace find_inverse_proportion_function_l275_275221

-- Define the options as functions
def optionA (x : ℝ) : ℝ := x / 2
def optionB (x : ℝ) : ℝ := 2 / (x + 1)
def optionC (x : ℝ) : ℝ := 2 / x
def optionD (x : ℝ) : ℝ := 2 * x

-- Define what it means to be an inverse proportion function
def isInverseProportion (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ x, f x = k / x)

-- Statement of the proof problem
theorem find_inverse_proportion_function :
  isInverseProportion optionC ∧
  ¬ isInverseProportion optionA ∧
  ¬ isInverseProportion optionB ∧
  ¬ isInverseProportion optionD :=
by sorry

end find_inverse_proportion_function_l275_275221


namespace cyclic_quadrilateral_l275_275232

noncomputable theory

variables {A B C D E F G H : Type} [inner_product_space ℝ A B C D E F G H]

-- Given conditions
def triangle (A B C : Type) : Prop := true

def tangency_points (D E F : Type) (BC AB AC : B) (I : Type) : Prop :=
  true

-- Define G and H for intersections
def intersection (ED FD : Type) : (G H : Type) := (arbitrary G, arbitrary H)

-- Problem statement
theorem cyclic_quadrilateral (A B C D E F G H : Type)
  (h₁ : triangle A B C)
  (h₂ : tangency_points D E F B (A B C) C)
  (h₃ : intersection E D F D = (G, H)) :
  cyclic G N H M :=
sorry

end cyclic_quadrilateral_l275_275232


namespace fraction_of_married_men_l275_275949

/-- At a social gathering, there are only single women and married men with their wives.
     The probability that a randomly selected woman is single is 3/7.
     The fraction of the people in the gathering that are married men is 4/11. -/
theorem fraction_of_married_men (women : ℕ) (single_women : ℕ) (married_men : ℕ) (total_people : ℕ) 
  (h_women_total : women = 7)
  (h_single_women_probability : single_women = women * 3 / 7)
  (h_married_women : women - single_women = married_men)
  (h_total_people : total_people = women + married_men) :
  married_men / total_people = 4 / 11 := 
by sorry

end fraction_of_married_men_l275_275949


namespace cos_angle_ST_QR_l275_275861

-- Definitions based on given conditions
variables (P Q R S T : Type) [inner_product_space ℝ P] [inner_product_space ℝ R]
variables (PQR_PST : is_triangle P Q R) (PST_PQR : is_triangle P S T)
variables (Q_midpoint_ST : midpoint Q S T)
variables (PQ_length : ∥P - Q∥ = 2) (ST_length : ∥S - T∥ = 2)
variables (QR_length : ∥Q - R∥ = 8) (PR_length : ∥P - R∥ = Real.sqrt 52)
variables (dot_product_relation : inner (P - Q) (P - S) + inner (P - R) (P - T) = 6)

theorem cos_angle_ST_QR :
  cos (angle (S - T) (Q - R)) = 1 :=
sorry

end cos_angle_ST_QR_l275_275861


namespace cube_sum_l275_275420

-- Definitions
variable (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω^2 + ω + 1 = 0) -- nonreal root

-- Theorem statement
theorem cube_sum : (2 - ω + ω^2)^3 + (2 + ω - ω^2)^3 = -2 :=
by 
  sorry

end cube_sum_l275_275420


namespace number_of_strawberries_stolen_l275_275356

-- Define the conditions
def daily_harvest := 5
def days_in_april := 30
def strawberries_given_away := 20
def strawberries_left_by_end := 100

-- Calculate total harvested strawberries
def total_harvest := daily_harvest * days_in_april
-- Calculate strawberries after giving away
def remaining_after_giveaway := total_harvest - strawberries_given_away

-- Prove the number of strawberries stolen
theorem number_of_strawberries_stolen : remaining_after_giveaway - strawberries_left_by_end = 30 := by
  sorry

end number_of_strawberries_stolen_l275_275356


namespace number_of_girls_in_school_l275_275385

theorem number_of_girls_in_school
  (B G : ℕ)
  (h_total_students : B + G = 604)
  (h_avg_age_boys : 12 * B)
  (h_avg_age_girls : 11 * G)
  (h_avg_age_school : 12 * B + 11 * G = 11.75 * 604) :
  G = 151 := sorry

end number_of_girls_in_school_l275_275385


namespace minimum_sum_of_distances_l275_275558

-- Definitions based on the conditions
def five_digit_numbers := {n : ℕ | 10000 ≤ n ∧ n ≤ 99999}

def distance (a b : ℕ) : ℕ :=
  let a_digits := (List.range 5).map (λ i => (a / 10^i) % 10),
      b_digits := (List.range 5).map (λ i => (b / 10^i) % 10)
  in List.findIndex (λ i => a_digits.nth i ≠ b_digits.nth i) (List.range 5).getD 5

-- The given conditions translated to constraints for distances
axiom total_pairs : 89999 = ∑ i in {1, 2, 3, 4, 5}, (∑ (a, b) in five_digit_numbers ×ˢ five_digit_numbers, distance a b = i)

theorem minimum_sum_of_distances :
  ∑ i in {1, 2, 3, 4, 5}, i * (∑ (a, b) in five_digit_numbers ×ˢ five_digit_numbers, distance a b = i) ≥ 101105 :=
sorry

end minimum_sum_of_distances_l275_275558


namespace pool_capacity_l275_275715

def hose1_rate : ℝ := 50
def duration1 : ℝ := 3
def drain1_rate (x : ℝ) : ℝ := x
def hose2_rate : ℝ := 70
def duration2 : ℝ := 2
def drain2_rate (x : ℝ) : ℝ := 10 + x

theorem pool_capacity (x : ℝ) : 
  let C := (hose1_rate - drain1_rate x) * duration1 + (hose1_rate + hose2_rate - drain2_rate x) * duration2 in
  C = 390 - 5 * x :=
by
  calc
    let C := (hose1_rate - drain1_rate x) * duration1 + (hose1_rate + hose2_rate - drain2_rate x) * duration2
    show C = 390 - 5 * x from sorry

end pool_capacity_l275_275715


namespace line_intersects_x_axis_at_point_l275_275912

theorem line_intersects_x_axis_at_point :
  ∃ x, (4 * x - 2 * 0 = 6) ∧ (2 - 0 = 2 * (0 - x)) → x = 2 := 
by
  sorry

end line_intersects_x_axis_at_point_l275_275912


namespace no_equilateral_cross_section_l275_275400

-- Given conditions
variables (O A B C : Point)
variables (a b : Real)
variable hAOB : ∠ A O B = 90
variable hAOC : ∠ A O C = 90
variable hBOC : ∠ B O C = Real.arccos (1 / 3)
variable hEquilateral : AB = AC ∧ AC = BC

-- Prove the question
theorem no_equilateral_cross_section :
  ¬ ∃ (O A B C : Point) (a b : Real) (hAOB : ∠ A O B = 90) (hAOC : ∠ A O C = 90)
    (hBOC : ∠ B O C = Real.arccos (1 / 3))
    (hEquilateral : AB = AC ∧ AC = BC), True :=
by
  sorry

end no_equilateral_cross_section_l275_275400


namespace distribution_schemes_80_l275_275259

theorem distribution_schemes_80 :
  let students := {1, 2, 3, 4, 5}
  let groups := {A, B, C}
  ∃ (f : students → groups),
    (∀ s ∈ students, f s ∈ groups) ∧ 
    (∃ a : finset students, ∃ b : finset students, ∃ c : finset students,
    (a ∩ b = ∅) ∧ (a ∩ c = ∅) ∧ (b ∩ c = ∅) ∧ 
    (a ∪ b ∪ c = students) ∧ 
    (2 ≤ a.card) ∧ (1 ≤ b.card) ∧ (1 ≤ c.card) ∧
    (a.card + b.card + c.card = 5)) :=
    ∃ (distributions : finset (students → groups)),
    distributions.card = 80 :=
sorry

end distribution_schemes_80_l275_275259


namespace Yoongi_has_fewest_apples_l275_275025

def Jungkook_apples : Nat := 6 * 3
def Yoongi_apples : Nat := 4
def Yuna_apples : Nat := 5

theorem Yoongi_has_fewest_apples :
  Yoongi_apples < Jungkook_apples ∧ Yoongi_apples < Yuna_apples :=
by
  sorry

end Yoongi_has_fewest_apples_l275_275025


namespace find_k_l275_275836

-- Definitions of points and the parallel condition
def A : (ℝ × ℝ) := (-6, 2)
def B : (ℝ × ℝ) := (2, -2)
def X : (ℝ × ℝ) := (-2, 10)
def Y (k : ℝ) : (ℝ × ℝ) := (8, k)

-- Definition to calculate slope
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- The proof problem
theorem find_k (k : ℝ) :
  slope A B = slope X (Y k) → k = 5 :=
by
  simp [slope, A, B, X, Y]
  sorry

end find_k_l275_275836


namespace tan_A_value_l275_275757

-- Define the conditions provided
variables (A B C : Type) [EuclideanGeometry A] -- A, B, and C are points in Euclidean geometry
variable (ABC : Triangle A B C) -- Triangle exists with vertices A, B, C
variable (h_angle : Angle A B C = 90) -- Given angle BAC is a right angle
variable (AB AC BC : ℝ) -- Side lengths of the triangle
variable (h_AB : AB = 40) -- Given length of AB
variable (h_BC : BC = 41) -- Given length of BC

-- Define the Pythagorean theorem calculation for AC
def AC : ℝ := sqrt (BC^2 - AB^2)
-- Applying the Pythagorean theorem given the conditions
axiom h_pythagorean : AC = sqrt (41^2 - 40^2)
axiom h_AC : AC = 9

-- Definition for the tangent of angle A
def tan_A := AC / AB

-- Prove the required value of tangent
theorem tan_A_value : tan_A = 9/40 := by
  -- Introduction and calculations here
  sorry -- Proof is skipped as per instructions

end tan_A_value_l275_275757


namespace evaporation_period_length_l275_275904

def initial_water_amount : ℝ := 10
def daily_evaporation_rate : ℝ := 0.0008
def percentage_evaporated : ℝ := 0.004  -- 0.4% expressed as a decimal

theorem evaporation_period_length :
  (percentage_evaporated * initial_water_amount) / daily_evaporation_rate = 50 := by
  sorry

end evaporation_period_length_l275_275904


namespace simplest_proper_fraction_simplest_improper_fraction_l275_275884

theorem simplest_proper_fraction (d : ℕ) (h1 : d > 7) (h2 : Nat.coprime 7 d) : (7 : ℚ) / d = 7 / 8 :=
by
  sorry

theorem simplest_improper_fraction (n : ℕ) (h1 : n ≥ 7) (h2 : Nat.coprime 7 n) : (n : ℚ) / 7 = 6 / 7 :=
by
  sorry

end simplest_proper_fraction_simplest_improper_fraction_l275_275884


namespace find_x_l275_275714

-- Definitions for the vectors a and b
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, 1)

-- Definition for the condition of parallel vectors
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

-- Mathematical statement to prove
theorem find_x (x : ℝ) 
  (h_parallel : are_parallel (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2)) : 
  x = -1 :=
sorry

end find_x_l275_275714


namespace largest_whole_number_l275_275271

theorem largest_whole_number (x : ℕ) (h : 9 * x < 150) : x ≤ 16 :=
by {
  have h₀ : 9 * 17 = 153, by norm_num,
  have h₁ : 153 > 150, by norm_num,
  have h₂ : 9 * 16 = 144, by norm_num,
  have h₃ : 144 < 150, by norm_num,
  split,
  { intro h,
    exact h₃, },
  { intro h,
    exact h₁, },
  exact h₀ }

end largest_whole_number_l275_275271


namespace irrational_in_set_l275_275941

theorem irrational_in_set :
  ∃ (x : ℝ), x = sqrt 3 ∧ irrational x ∧ 
  (∀ y ∈ ({sqrt 3, (0 : ℝ), -0.33, (10 : ℝ)} : set ℝ), y ≠ x → ¬irrational y) :=
by
  sorry

end irrational_in_set_l275_275941


namespace locus_of_T_circle_l275_275671

open Real EuclideanGeometry

noncomputable def locus_of_T (a b r : ℝ) : set (ℝ × ℝ) :=
  { T : ℝ × ℝ | prod.fst T^2 + prod.snd T^2 = 2 * r^2 - (a^2 + b^2) }

theorem locus_of_T_circle (a b r : ℝ) (M : ℝ × ℝ) (M_inside_circle : a = M.1 ∧ b = M.2) :
  exists (O : ℝ × ℝ) (R : ℝ),
    O = (0, 0) ∧
    R = sqrt (2 * r^2 - (a^2 + b^2)) ∧
    locus_of_T a b r = metric.ball O R :=
sorry

end locus_of_T_circle_l275_275671


namespace bonus_is_correct_l275_275131

noncomputable def wages_for_quarter_bonus (November_wage : ℝ) : ℝ := 
  let October_wage := (9 / 8) * November_wage
  let December_wage := (4 / 3) * November_wage
  let total_earnings := October_wage + November_wage + December_wage
  0.2 * total_earnings

theorem bonus_is_correct : (wages_for_quarter_bonus 2160 = 1494) :=
by {
  let October_wage := (9 / 8) * 2160
  let December_wage := (4 / 3) * 2160
  have h1 : December_wage = October_wage + 450,
  {
    unfold December_wage October_wage,
    norm_num,
  },
  
  have h2 : wages_for_quarter_bonus 2160 = 0.2 * (October_wage + 2160 + December_wage),
  {
    unfold wages_for_quarter_bonus October_wage December_wage,
  },
  norm_num at *,
  rw [h1, h2],
  norm_num,
  rw [<- mul_assoc],
  norm_num,
  rw bdw_num = 1494 
  sorry
}

end bonus_is_correct_l275_275131


namespace angle_F_measure_l275_275011

theorem angle_F_measure (α β γ : ℝ) (hD : α = 84) (hAngleSum : α + β + γ = 180) (hBeta : β = 4 * γ + 18) :
  γ = 15.6 := by
  sorry

end angle_F_measure_l275_275011


namespace sum_of_possible_values_of_N_for_five_lines_l275_275986

theorem sum_of_possible_values_of_N_for_five_lines : 
  let N := (λ (l1 l2 l3 l4 l5 : set ℝ) (h1 : l1 ≠ l2) (h2 : l1 ≠ l3) (h3 : l1 ≠ l4) (h4 : l1 ≠ l5)
                (h5 : l2 ≠ l3) (h6 : l2 ≠ l4) (h7 : l2 ≠ l5) (h8 : l3 ≠ l4) (h9 : l3 ≠ l5)
                (h10 : l4 ≠ l5),
                (∀ i j k : nat, i ≠ j ∧ i ≠ k ∧ j ≠ k → ¬(∃ x, x ∈ l1 ∧ x ∈ l2 ∧ x ∈ l3)) 
                → nat)
  in ∀ N, N = (∑ k in finset.range 11, k) := 
by
  sorry

end sum_of_possible_values_of_N_for_five_lines_l275_275986


namespace sales_overlap_l275_275743

-- Define the conditions
def bookstore_sale_days : List ℕ := [2, 6, 10, 14, 18, 22, 26, 30]
def shoe_store_sale_days : List ℕ := [1, 8, 15, 22, 29]

-- Define the statement to prove
theorem sales_overlap : (bookstore_sale_days ∩ shoe_store_sale_days).length = 1 := 
by
  sorry

end sales_overlap_l275_275743


namespace probability_red_higher_than_green_l275_275918

theorem probability_red_higher_than_green :
  let P (k : ℕ) := 2^(-k)
  in (∑' (k : ℕ), P k * P k) = (1 : ℝ) / 3 :=
by
  sorry

end probability_red_higher_than_green_l275_275918


namespace fraction_denominator_l275_275379

theorem fraction_denominator (x y Z : ℚ) (h : x / y = 7 / 3) (h2 : (x + y) / Z = 2.5) :
    Z = (4 * y) / 3 :=
by sorry

end fraction_denominator_l275_275379


namespace triangle_ratio_l275_275012

theorem triangle_ratio {A B C D E : Type} [EuclideanGeometry D]
  (triangle : Triangle D A B C)
  (AC2BC : triangle.AC = 2 * triangle.BC)
  (C90 : triangle.angleC = 90)
  (footD : triangle.altitude D C A B)
  (circleE : Circle_with_diameter AD intersects AC at E) :
  AE / EC = 4 := 
sorry

end triangle_ratio_l275_275012


namespace moles_of_CH4_l275_275721

-- Conditions
def carbon_moles : ℕ := 3
def hydrogen_moles : ℕ := 6
axiom balanced_eq (c h ch : ℕ) : ch = c ∧ ch = h / 2

-- Question
theorem moles_of_CH4 :
  let c := carbon_moles,
  let h := hydrogen_moles,
  let ch := c in
  balanced_eq c h ch → ch = 3 :=
by
  sorry

end moles_of_CH4_l275_275721


namespace arrangement_count_l275_275931

-- Define the animals and their positions
def animals := {L1, L2, L3, T1, T2} -- Three lions (L1, L2, L3) and two tigers (T1, T2)

-- Condition: No two tigers are next to each other
def valid_positions (positions : list (list char)) : Prop :=
  ∀ (i : ℕ), i < positions.length - 1 →
  (positions.nth i = some 'T' → positions.nth (i + 1) ≠ some 'T')

-- Question: The number of valid arrangements
theorem arrangement_count : ∃ n, n = 72 :=
  ∃ n, 
  (∃ (positions : list (list char)), positions.perm animals ∧ valid_positions positions) →
  n = 72

end arrangement_count_l275_275931


namespace inequality_proof_l275_275944

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 0) :
  (33 * a^2 - a) / (33 * a^2 + 1) + (33 * b^2 - b) / (33 * b^2 + 1) + (33 * c^2 - c) / (33 * c^2 + 1) ≥ 0 :=
sorry

end inequality_proof_l275_275944


namespace correct_statement_is_B_l275_275881

-- Definitions to be used
def random_event (E : Type) (p : E → Prop) := ∃ x : E, p x
def certain_event (E : Type) (p : E → Prop) := ∀ x : E, p x
def impossible_event (E : Type) (p : E → Prop) := ∀ x : E, ¬ p x
def comprehensive_survey_needed (context : Type) (desc : context → Prop) := true -- Assuming further elaboration

-- Contexts as described in the problem
constant equipment_parts_of_artificial_satellites : Type
constant quality_of_parts : equipment_parts_of_artificial_satellites → Prop

constant grade_9_students : Type
constant vision_of_students : grade_9_students → Prop

constant sports_lottery_ticket : Type
constant winning_event : sports_lottery_ticket → Prop

constant coin_toss : Type
constant head_facing_up : coin_toss → Prop

-- The mathematical proof problem in Lean 4
theorem correct_statement_is_B :
  comprehensive_survey_needed grade_9_students vision_of_students :=
sorry

end correct_statement_is_B_l275_275881


namespace triangle_inequality_holds_l275_275798

theorem triangle_inequality_holds (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^3 + b^3 + c^3 + 4 * a * b * c ≤ (9 / 32) * (a + b + c)^3 :=
by {
  sorry
}

end triangle_inequality_holds_l275_275798


namespace sum_of_triangulars_15_to_20_l275_275956

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_triangulars_15_to_20 : 
  (triangular_number 15 + triangular_number 16 + triangular_number 17 + triangular_number 18 + triangular_number 19 + triangular_number 20) = 980 :=
by
  sorry

end sum_of_triangulars_15_to_20_l275_275956


namespace emily_can_see_emerson_for_14_minutes_l275_275262

def emily_speed := 15
def emerson_speed := 8
def speed_increase := 3
def initial_distance_ahead := 0.8
def distance_behind := 0.8

theorem emily_can_see_emerson_for_14_minutes :
  let emily_effective_speed := emily_speed + speed_increase
  let emerson_effective_speed := emerson_speed + speed_increase
  let relative_speed := emily_effective_speed - emerson_effective_speed
  let catch_up_distance := initial_distance_ahead
  let distance_until_behind := distance_behind
  let catch_up_time := catch_up_distance / relative_speed
  let behind_time := distance_until_behind / relative_speed
  let total_observation_time_hours := catch_up_time + behind_time
  let total_observation_time_minutes := total_observation_time_hours * 60
  total_observation_time_minutes ≈ 14 :=
by
  sorry

end emily_can_see_emerson_for_14_minutes_l275_275262


namespace solve_functional_equation_l275_275820

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 4

theorem solve_functional_equation :
  (∃ x : ℝ, f(f(x)) = x) ↔
  (x = ( -1 + Real.sqrt 17 ) / 2 ∨
   x = ( -1 - Real.sqrt 17 ) / 2 ∨
   x = ( -3 + Real.sqrt 13 ) / 2 ∨
   x = ( -3 - Real.sqrt 13 ) / 2) :=
by
  sorry

end solve_functional_equation_l275_275820


namespace added_water_is_18_l275_275569

def capacity : ℕ := 40

def initial_full_percent : ℚ := 0.30

def final_full_fraction : ℚ := 3/4

def initial_water (capacity : ℕ) (initial_full_percent : ℚ) : ℚ :=
  initial_full_percent * capacity

def final_water (capacity : ℕ) (final_full_fraction : ℚ) : ℚ :=
  final_full_fraction * capacity

def water_added (initial_water : ℚ) (final_water : ℚ) : ℚ :=
  final_water - initial_water

theorem added_water_is_18 :
  water_added (initial_water capacity initial_full_percent) (final_water capacity final_full_fraction) = 18 := by
  sorry

end added_water_is_18_l275_275569


namespace product_of_roots_l275_275149

theorem product_of_roots : 
  (Real.root 81 4) * (Real.root 27 3) * (Real.sqrt 9) = 27 :=
by
  sorry

end product_of_roots_l275_275149


namespace trip_total_charge_is_correct_l275_275021

-- Define the initial fee
def initial_fee : ℝ := 2.35

-- Define the charge per increment
def charge_per_increment : ℝ := 0.35

-- Define the increment size in miles
def increment_size : ℝ := 2 / 5

-- Define the total distance of the trip
def trip_distance : ℝ := 3.6

-- Define the total charge function
def total_charge (initial : ℝ) (increment_charge : ℝ) (increment : ℝ) (distance : ℝ) : ℝ :=
  initial + (distance / increment) * increment_charge

-- Prove the total charge for a trip of 3.6 miles is $5.50
theorem trip_total_charge_is_correct :
  total_charge initial_fee charge_per_increment increment_size trip_distance = 5.50 :=
by
  sorry

end trip_total_charge_is_correct_l275_275021


namespace room_assignment_count_l275_275800

theorem room_assignment_count :
  (number_of_ways : ℕ) :=
  let configurations := [
    (5, 0, 0, 0),   -- 1 way
    (4, 1, 0, 0),   -- 5 ways
    (3, 2, 0, 0),   -- 10 ways
    (3, 1, 1, 0),   -- 20 ways
    (2, 2, 1, 0),   -- 15 ways
    (2, 1, 1, 1),   -- 10 ways
    (1, 1, 1, 1, 1) -- 0 ways
  ]
  number_of_ways = (1 + 5 + 10 + 20 + 15 + 10 + 0) :=
by 
sorry

end room_assignment_count_l275_275800


namespace gcd_840_1785_gcd_612_468_l275_275979

-- Definition for GCD using the Euclidean algorithm
def gcd_euclidean (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd_euclidean b (a % b)

-- Definition for GCD using successive subtraction
def gcd_subtraction (a b : ℕ) : ℕ :=
  if a = b then a
  else if a > b then gcd_subtraction (a - b) b
  else gcd_subtraction a (b - a)

-- Statement for the proof problem
theorem gcd_840_1785 : gcd_euclidean 840 1785 = 105 := by sorry

theorem gcd_612_468 : gcd_subtraction 612 468 = 156 := by sorry

end gcd_840_1785_gcd_612_468_l275_275979


namespace math_test_total_questions_l275_275135

theorem math_test_total_questions (Q : ℕ) (h : Q - 38 = 7) : Q = 45 :=
by
  sorry

end math_test_total_questions_l275_275135


namespace pos_of_three_power_neg_l275_275367

theorem pos_of_three_power_neg (x : ℝ) (hx : x < 0) : 3^(-x) > 0 :=
sorry

end pos_of_three_power_neg_l275_275367


namespace seed_mixture_percent_l275_275465

theorem seed_mixture_percent (X Y : ℝ) (w : ℝ) 
  (h1 : w = 100) 
  (h2 : 0.40 * X + 0.25 * Y = 40) 
  (h3 : X + Y = w) 
  (h4 : w * 0.4 = 40) : 
  X / w = 1 :=
by 
suffices : X = w, sorry

end seed_mixture_percent_l275_275465


namespace solve_for_x_l275_275788

-- The main theorem statement
theorem solve_for_x : (∃ h : ℝ → ℝ, (∀ x, h (3 * x + 2) = 4 * x - 5) ∧ (∃ x0, h x0 = x0 ∧ x0 = 23)) :=
by
  -- Define the function h
  let h : ℝ → ℝ := λ x, (4 * (x - 2) / 3) - 5
  -- Provide the function to the 'exists' clause
  use h
  -- Split the conditions
  split
  -- First part: proving the condition for h
  . intros x
    sorry -- This is where the function's property would be formally shown
  -- Second part: proving the existence and value of x where h(x) = x
  . use 23
    split
    -- Prove h(23) = 23
    . sorry -- This is where the specific computation for h would be shown
    -- Prove that the value is indeed 23
    . refl

end solve_for_x_l275_275788


namespace identify_switches_once_identify_switches_twice_l275_275888

theorem identify_switches_once (n : ℕ) (lit_by_switches : List (List ℕ)) :
  ∀ i, 1 ≤ i ∧ i ≤ n → ∃ (method : List (List ℕ)), (method.length = 3 ∧
  (∃ (j : ℕ), method.nth 0 = some [i] ∧ method.nth 2 = none ∧ 
             lit_by_switches.nth i = some [j])
  → ∃ (k : ℕ), k = i + 1 ∧ method.nth 1 = some [k - 1] ∧ 
                lit_by_switches.nth k = some [j - 1]) :=
sorry

theorem identify_switches_twice : ∀ (k : ℕ), k = 2 →
  ∃ (m : ℕ), m = (3^k) ∧ 
  ∀ (lit_by_switches : List (List ℕ)),
    ∀ (i : ℕ), 1 ≤ i ∧ i ≤ m → 
    ∃ (j : ℕ), 
    lit_by_switches.nth i = some [j] ∧ j ≤ m :=
sorry

end identify_switches_once_identify_switches_twice_l275_275888


namespace investment_future_value_l275_275552

-- Definitions extracted from the problem's conditions
def initial_investment : ℕ := 1500
def annual_interest_rate : ℚ := 0.08
def years_triple (x : ℚ) : ℚ := 112 / x
def time_period : ℕ := 28
def factor_triple_in_time_period (years_triple : ℚ) (time_period : ℚ) : ℚ := time_period / years_triple
def final_amount (initial_investment : ℚ) (triples : ℚ) : ℚ := initial_investment * (3 ^ (triples.to_nat))

-- The theorem statement
theorem investment_future_value :
  final_amount initial_investment (factor_triple_in_time_period (years_triple annual_interest_rate) time_period) = 13500 := 
by sorry

end investment_future_value_l275_275552


namespace probability_sum_eight_rolling_two_dice_l275_275463

theorem probability_sum_eight_rolling_two_dice : 
  let outcomes := [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                   (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
                   (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
                   (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                   (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
                   (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)],
      favorable_outcomes := [(2, 6), (3, 5), (4, 4), (5, 3), (6, 2)],
      probability := (favorable_outcomes.length : ℚ) / (outcomes.length : ℚ)
  in probability = 5 / 36 := 
by sorry

end probability_sum_eight_rolling_two_dice_l275_275463


namespace largest_prime_divisor_of_360_is_5_l275_275294

theorem largest_prime_divisor_of_360_is_5 (p : ℕ) (hp₁ : Nat.Prime p) (hp₂ : p ∣ 360) : p ≤ 5 :=
by 
sorry

end largest_prime_divisor_of_360_is_5_l275_275294


namespace border_pieces_is_75_l275_275520

-- Definitions based on conditions
def total_pieces : Nat := 500
def trevor_pieces : Nat := 105
def joe_pieces : Nat := 3 * trevor_pieces
def missing_pieces : Nat := 5

-- Number of border pieces
def border_pieces : Nat := total_pieces - missing_pieces - (trevor_pieces + joe_pieces)

-- Theorem statement
theorem border_pieces_is_75 : border_pieces = 75 :=
by
  -- Proof goes here
  sorry

end border_pieces_is_75_l275_275520


namespace S_of_target_l275_275622

variable (S : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ)

-- Conditions
axiom linear_property (a b : ℝ) (u v : ℝ × ℝ × ℝ) : 
  S (a • u + b • v) = a • S u + b • S v

axiom cross_product_property (u v : ℝ × ℝ × ℝ) : 
  S (u × v) = S u × S v

axiom S_of_given1 : 
  S ⟨8, 4, 2⟩ = ⟨5, -2, 10⟩

axiom S_of_given2 : 
  S ⟨-8, 2, 4⟩ = ⟨5, 10, -2⟩

theorem S_of_target : 
  S ⟨4, 10, 16⟩ = ⟨-10, 26, 38⟩ :=
  sorry

end S_of_target_l275_275622


namespace exists_circle_tangent_BD_BF_CE_τ_l275_275656

-- Definitions based on conditions:
variables (A B C D E F : Point) (τ k : Circle)
axiom AB_parallel_CE : is_parallel AB CE
axiom angle_ABC_gt_90 : ∠ABC > 90
axiom k_tangent_AD : is_tangent k AD
axiom k_tangent_CE : is_tangent k CE
axiom k_tangent_τ : is_tangent k τ
axiom k_τ_tangent_on_arc_ED : touches_on_arc k τ E D (missing_points {A, B, C})
axiom F_on_τ_tangent : F ≠ A ∧ is_tangent (line_through A F) k ∧ (F ∈ τ) ∧ (A ∉ AD)

-- The goal to be proved:
theorem exists_circle_tangent_BD_BF_CE_τ :
  ∃ r, is_tangent r BD ∧ is_tangent r BF ∧ is_tangent r CE ∧ is_tangent r τ := 
sorry

end exists_circle_tangent_BD_BF_CE_τ_l275_275656


namespace blue_pill_cost_l275_275274

variable (cost_blue_pill : ℕ) (cost_red_pill : ℕ) (daily_cost : ℕ) 
variable (num_days : ℕ) (total_cost : ℕ)
variable (cost_diff : ℕ)

theorem blue_pill_cost :
  num_days = 21 ∧
  total_cost = 966 ∧
  cost_diff = 4 ∧
  daily_cost = total_cost / num_days ∧
  daily_cost = cost_blue_pill + cost_red_pill ∧
  cost_blue_pill = cost_red_pill + cost_diff ∧
  daily_cost = 46 →
  cost_blue_pill = 25 := by
  sorry

end blue_pill_cost_l275_275274


namespace correct_answer_l275_275067

def P : Set ℝ := {1, 2, 3}
def Q : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem correct_answer : P ∩ Q ⊆ P := by
  sorry

end correct_answer_l275_275067


namespace eel_species_count_l275_275260

theorem eel_species_count (sharks eels whales total : ℕ)
    (h_sharks : sharks = 35)
    (h_whales : whales = 5)
    (h_total : total = 55)
    (h_species_sum : sharks + eels + whales = total) : eels = 15 :=
by
  -- Proof goes here
  sorry

end eel_species_count_l275_275260


namespace range_f_l275_275161

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_f : Set.Ioo 0 1 ∪ {1} = {y : ℝ | ∃ x : ℝ, f x = y} :=
by 
  sorry

end range_f_l275_275161


namespace min_value_M_a1_min_value_M_3_l275_275285

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) := x^2 + x + a^2 + a
def g (a : ℝ) (x : ℝ) := x^2 - x + a^2 - a

-- Define M(x) as the maximum of f(x) and g(x)
def M (a : ℝ) (x : ℝ) := max (f a x) (g a x)

-- Problem 1: Prove that the minimum value of M(x) for a = 1 is 7/4
theorem min_value_M_a1 : ∃ x : ℝ, x.f = 1 → M 1 x = 7 / 4 :=
by
  sorry

-- Problem 2: Prove that if the minimum value of M(x) is 3, then a = (sqrt 14 - 1) / 2
theorem min_value_M_3 (a : ℝ) : (∃ x : ℝ, M a x = 3) → a = (Real.sqrt 14 - 1) / 2 :=
by
  sorry

end min_value_M_a1_min_value_M_3_l275_275285


namespace serving_ways_correct_l275_275826

open Finset

def meal := {b: ℕ // b < 3}

def orders : Finset (Fin 10 × meal) := 
  {(⟨0, _⟩, ⟨0, _⟩), (⟨1, _⟩, ⟨0, _⟩), (⟨2, _⟩, ⟨0, _⟩), (⟨3, _⟩, ⟨0, _⟩), 
   (⟨4, _⟩, ⟨1, _⟩), (⟨5, _⟩, ⟨1, _⟩), (⟨6, _⟩, ⟨1, _⟩), 
   (⟨7, _⟩, ⟨2, _⟩), (⟨8, _⟩, ⟨2, _⟩), (⟨9, _⟩, ⟨2, _⟩)}

def servers := univ.perm

noncomputable def valid_serving_ways : ℕ := 
  let ways := (servers.filter (λ f: perm (Fin 10), (orders.filter (λ (o: Fin 10 × meal), (f o.1).fst = o.1)).card = 2)).card
  in ways

theorem serving_ways_correct : valid_serving_ways = 288 := sorry

end serving_ways_correct_l275_275826


namespace problem_statement_l275_275102

def scientific_notation (n: ℝ) (mantissa: ℝ) (exponent: ℤ) : Prop :=
  n = mantissa * 10 ^ exponent

theorem problem_statement : scientific_notation 320000 3.2 5 :=
by {
  sorry
}

end problem_statement_l275_275102


namespace determine_y_l275_275840

theorem determine_y (y : ℕ) (h1 : ∀ d : ℕ, d ∣ y → 0 < d → d ∈ [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 48, 72, 144, 288])
  (h2 : 18 ∣ y) (h3 : 24 ∣ y) (h4 : 18 = (factors y).length) : y = 288 :=
by sorry

end determine_y_l275_275840


namespace arithmetic_geometric_sum_l275_275393

theorem arithmetic_geometric_sum (S : ℕ → ℕ) (n : ℕ) 
  (h1 : S n = 48) 
  (h2 : S (2 * n) = 60)
  (h3 : (S (2 * n) - S n) ^ 2 = S n * (S (3 * n) - S (2 * n))) : 
  S (3 * n) = 63 := by
  sorry

end arithmetic_geometric_sum_l275_275393


namespace will_can_buy_correct_amount_of_toys_l275_275168

-- Define the initial conditions as constants
def initial_amount : Int := 57
def amount_spent : Int := 27
def cost_per_toy : Int := 6

-- Lemma stating the problem to prove.
theorem will_can_buy_correct_amount_of_toys : (initial_amount - amount_spent) / cost_per_toy = 5 :=
by
  sorry

end will_can_buy_correct_amount_of_toys_l275_275168


namespace surface_area_of_sphere_O2_l275_275014

noncomputable def cube_edge_length : ℝ := 1
noncomputable def sphere_O1_radius : ℝ := 1 / 2
noncomputable def sphere_O2_radius : ℝ := 1 - (sqrt 3) / 2

theorem surface_area_of_sphere_O2 (cube_edge_length sphere_O1_radius sphere_O2_radius : ℝ) :
    cube_edge_length = 1 →
    sphere_O1_radius = 1 / 2 →
    sphere_O2_radius = 1 - (sqrt 3) / 2 →
    4 * Real.pi * sphere_O2_radius ^ 2 = (7 - 4 * sqrt 3) * Real.pi :=
by
  intros h_edge h_O1_radius h_O2_radius
  rw [h_edge, h_O1_radius, h_O2_radius]
  sorry

end surface_area_of_sphere_O2_l275_275014


namespace number_of_two_digit_integers_l275_275701

/--
The number of different positive two-digit integers that can be formed 
using the digits 3, 5, 8, and 9 with no repeated digits.
-/
theorem number_of_two_digit_integers : 
  let digits := {3, 5, 8, 9}
  in (∃ (nums : Finset (ℕ × ℕ)), 
        nums.card = 12 ∧ 
        ∀ d ∈ nums, (d.1 ∈ digits ∧ d.2 ∈ digits ∧ d.1 ≠ d.2)) := 
by
  sorry

end number_of_two_digit_integers_l275_275701


namespace ramon_current_age_l275_275770

variable (R : ℕ) (L : ℕ)

theorem ramon_current_age :
  (L = 23) → (R + 20 = 2 * L) → R = 26 :=
by
  intro hL hR
  rw [hL] at hR
  have : R + 20 = 46 := by linarith
  linarith

end ramon_current_age_l275_275770


namespace angle_equality_l275_275297

structure Trapezoid (A B C D : Type) :=
  (base1 : A × D)
  (base2 : B × C)
  (sides_eq : A = D ∧ B = C)

structure Point (P : Type) :=
  (midpoint : D = C)

def Midpoint (D C M : Type) := M = (D + C) / 2

theorem angle_equality 
  {A B C D M : Type} 
  (h_trapezoid : Trapezoid A B C D) 
  (h_sides_eq : A = D) 
  (h_midpoint : Midpoint D C M) : 
  ∠ M B C = ∠ B C A := by 
  sorry

end angle_equality_l275_275297


namespace number_of_outfits_l275_275101

theorem number_of_outfits (shirts pants ties hats : ℕ) 
  (h_shirts : shirts = 8) 
  (h_pants : pants = 5) 
  (h_ties : ties = 5) 
  (h_hats : hats = 3) : 
  shirts * pants * ties * hats = 600 :=
by
  rw [h_shirts, h_pants, h_ties, h_hats]
  norm_num
  sorry

end number_of_outfits_l275_275101


namespace machines_to_complete_work_in_40_days_equals_28_l275_275473

noncomputable def work_rate_sum (W : ℕ → ℝ) : ℝ :=
  ∑ i in finRange 28, W i

theorem machines_to_complete_work_in_40_days_equals_28 (W : ℕ → ℝ) (H : work_rate_sum W * 10 = W_total) : 
  ∃ M, M = 28 ∧ work_rate_sum W / 4 = (work_rate_sum W) :=
begin
  sorry
end

end machines_to_complete_work_in_40_days_equals_28_l275_275473


namespace quotient_max_whole_cars_div_10_l275_275804

noncomputable def max_whole_cars_div_10 {v : ℕ → ℕ} 
  (car_length : ℕ) (eye_count : ℕ → ℕ → ℕ) (cars_speed : ℕ → ℕ) : ℕ := 
let n := arbitrary ℕ in
let distance := λ n : ℕ, car_length + (n * car_length) in
let speed := λ n : ℕ, cars_speed n in
let max_units := λ n : ℕ, speed n / distance n in
let max_cars := λ n : ℕ, max_units n in
3750 / 10

theorem quotient_max_whole_cars_div_10 : max_whole_cars_div_10 4 
  (λ units cars, arbitrary ℕ)
  (λ speed, 15000)  = 375 := 
by 
  unfold max_whole_cars_div_10 
  sorry

end quotient_max_whole_cars_div_10_l275_275804


namespace equilateral_triangle_probability_l275_275943

theorem equilateral_triangle_probability :
  ∀ (t : Triangle) (is_equilateral : t.is_equilateral)
    (divided_into_six : (∀ pt : Point, pt ∈ t.vertices → 
      ∃ mid : Point, mid ∈ t.midpoints → true)) 
    (shaded_regions : ℕ)
    (three_shaded : shaded_regions = 3),
  t.probability_of_hitting_shaded_region = 1 / 2 := 
by
  sorry

end equilateral_triangle_probability_l275_275943


namespace sin_three_pi_over_two_l275_275971

theorem sin_three_pi_over_two : Real.sin (3 * Real.pi / 2) = -1 := 
by
  sorry

end sin_three_pi_over_two_l275_275971


namespace steve_adds_each_year_l275_275098

def steve_initial_balance : ℝ := 100
def yearly_interest_rate : ℝ := 0.10
def balance_after_two_years : ℝ := 142

theorem steve_adds_each_year (X : ℝ) :
  (steve_initial_balance * (1 + yearly_interest_rate) + X) * (1 + yearly_interest_rate) + X = balance_after_two_years →
  X = 19 :=
begin
  sorry
end

end steve_adds_each_year_l275_275098


namespace A_remaining_time_is_3_l275_275550

-- Define the given conditions
def A_work_rate : ℝ := 1 / 9
def B_work_rate : ℝ := 1 / 15
def B_work_time : ℝ := 10
def B_completed_work : ℝ := B_work_rate * B_work_time
def remaining_work : ℝ := 1 - B_completed_work

-- Define the remaining work to be done by A and the time A will take to finish it
def A_remaining_time : ℝ := remaining_work / A_work_rate

-- State and prove the theorem
theorem A_remaining_time_is_3 :
  A_remaining_time = 3 :=
by
  -- Calculate intermediate values to simplify the statement
  have h1 : B_work_rate = 1 / 15 := rfl
  have h2 : B_work_time = 10 := rfl
  have h3 : B_completed_work = 10 * (1 / 15) := rfl
  have h4 : B_completed_work = 2 / 3 := by linarith
  have h5 : remaining_work = 1 - 2 / 3 := rfl
  have h6 : remaining_work = 1 / 3 := by linarith
  have h7 : A_work_rate = 1 / 9 := rfl
  have h8 : A_remaining_time = (1 / 3) / (1 / 9) := rfl
  have h9 : A_remaining_time = 3 := by linarith
  -- Conclude the proof
  exact h9

#eval A_remaining_time -- This should return the remaining time for A
#eval A_remaining_time_is_3 -- This should return proof of the theorem

end A_remaining_time_is_3_l275_275550


namespace boys_play_theater_l275_275802

theorem boys_play_theater (total_friends : ℕ) (fraction_girls : ℚ) (fraction_boys_theater : ℚ) (h1 : total_friends = 12) (h2 : fraction_girls = 2/3) (h3 : fraction_boys_theater = 3/4) :
  let girls := fraction_girls * total_friends in
  let boys := total_friends - girls in
  let boys_theater := fraction_boys_theater * boys in
  boys_theater = 3 :=
by
  -- The definitions of girls, boys, and boys_theater will be automatically inferred as expressions
  sorry

end boys_play_theater_l275_275802


namespace correct_proposition_l275_275222

theorem correct_proposition :
  (∃ x₀ : ℤ, x₀^2 = 1) ∧ ¬(∃ x₀ : ℤ, x₀^2 < 0) ∧ ¬(∀ x : ℤ, x^2 ≤ 0) ∧ ¬(∀ x : ℤ, x^2 ≥ 1) :=
by
  sorry

end correct_proposition_l275_275222


namespace rounding_bounds_l275_275915

theorem rounding_bounds:
  ∃ (max min : ℕ), (∀ x : ℕ, (x >= 1305000) → (x < 1305000) -> false) ∧ 
  (max = 1304999) ∧ 
  (min = 1295000) :=
by
  -- Proof steps would go here
  sorry

end rounding_bounds_l275_275915


namespace knight_traversal_impossible_l275_275447

-- Definitions of the problem
def infinite_chessboard : Type := ℤ × ℤ

def knight_can_traverse (board : infinite_chessboard → Prop) : Prop :=
  ∀ sq1 sq2 : infinite_chessboard, board sq1 → knight_move sq1 sq2 → board sq2

def piece_every_three_squares (board : infinite_chessboard → Prop) :=
  ∀ (x y : ℤ), board (3 * x, 3 * y)

def knight_move (pos1 pos2 : infinite_chessboard) : Prop :=
  ∃ (dx dy : ℤ), 
    (abs dx = 2 ∧ abs dy = 1 ∨ abs dx = 1 ∧ abs dy = 2) ∧
    (fst pos1 + dx = fst pos2) ∧ 
    (snd pos1 + dy = snd pos2)

theorem knight_traversal_impossible :
  ¬ (∃ (board : infinite_chessboard → Prop),
      piece_every_three_squares board ∧
      knight_can_traverse (λ sq, ¬ board sq)) :=
sorry

end knight_traversal_impossible_l275_275447


namespace arithmetic_sqrt_of_9_l275_275480

theorem arithmetic_sqrt_of_9 : real.sqrt 9 = 3 := 
sorry

end arithmetic_sqrt_of_9_l275_275480


namespace total_wheels_correct_l275_275946

def total_wheels (bicycles cars motorcycles tricycles quads : ℕ) 
(missing_bicycle_wheels broken_car_wheels missing_motorcycle_wheels : ℕ) : ℕ :=
  let bicycles_wheels := (bicycles - missing_bicycle_wheels) * 2 + missing_bicycle_wheels
  let cars_wheels := (cars - broken_car_wheels) * 4 + broken_car_wheels * 3
  let motorcycles_wheels := (motorcycles - missing_motorcycle_wheels) * 2
  let tricycles_wheels := tricycles * 3
  let quads_wheels := quads * 4
  bicycles_wheels + cars_wheels + motorcycles_wheels + tricycles_wheels + quads_wheels

theorem total_wheels_correct : total_wheels 25 15 8 3 2 5 2 1 = 134 := 
  by sorry

end total_wheels_correct_l275_275946


namespace PropA_neither_sufficient_nor_necessary_for_PropB_l275_275289

variable (a b : ℤ)

-- Proposition A
def PropA : Prop := a + b ≠ 4

-- Proposition B
def PropB : Prop := a ≠ 1 ∧ b ≠ 3

-- The required statement
theorem PropA_neither_sufficient_nor_necessary_for_PropB : ¬(PropA a b → PropB a b) ∧ ¬(PropB a b → PropA a b) :=
by
  sorry

end PropA_neither_sufficient_nor_necessary_for_PropB_l275_275289


namespace average_of_all_digits_l275_275134

theorem average_of_all_digits {a b : ℕ} (n : ℕ) (x y : ℕ) (h1 : a = 6) (h2 : b = 4) (h3 : n = 10) (h4 : x = 58) (h5 : y = 113) :
  ((a * x + b * y) / n = 80) :=
  sorry

end average_of_all_digits_l275_275134


namespace first_tap_fill_time_l275_275191

-- Define the conditions as stated in step a)
def rate_first_tap (T : ℝ) : ℝ := 1 / T
def rate_second_tap : ℝ := 1 / 5
def net_rate : ℝ := 1 / 20

-- The proof statement that we need to prove
theorem first_tap_fill_time (T : ℝ) (h1 : rate_first_tap T - rate_second_tap = net_rate) : T = 4 := by
  sorry

end first_tap_fill_time_l275_275191


namespace friedas_probability_l275_275282
open Probability

-- Definitions
def grid_size := 4
def state := (ℕ × ℕ)
def top_row_states := [(0, grid_size-1), (1, grid_size-1), (2, grid_size-1), (3, grid_size-1)]
def start_position : state := (0, 0)

def transition_prob := (s1 : state) (s2 : state) : ℚ :=
  if (σ = (s1.1 + 1, s1.2)) ||
     (σ = (s1.1 - 1, s1.2)) ||
     (σ = (s1.1, s1.2 + 1)) ||
     (σ = (s1.1, s1.2 - 1)) then 1/4 else 0

-- Probabilities
def p_n : ℕ → state → ℚ
| 0 s :=
  if s ∈ top_row_states then 1 else 0
| (n+1) s :=
  1/4 * ((p_n n ((s.1 + 1) % grid_size, s.2)) +
         (p_n n ((s.1 - 1 + grid_size) % grid_size, s.2)) +
         (p_n n  (s.1, (s.2 + 1) % grid_size)) +
         (p_n n  (s.1, (s.2 - 1 + grid_size) % grid_size)))

-- Main Theorem
theorem friedas_probability :
  p_n 5 start_position = 31/64
:= sorry

end friedas_probability_l275_275282


namespace ratio_of_ages_l275_275860

def tom_age := 40.5
def total_age := 54
def antonette_age := total_age - tom_age
def ratio := tom_age / antonette_age

theorem ratio_of_ages : ratio = 3 := by
  unfold tom_age total_age antonette_age ratio
  calc
    40.5 / (54 - 40.5) = 40.5 / 13.5 : rfl
                     ...              = 3       : by norm_num

end ratio_of_ages_l275_275860


namespace probability_case_7_probability_case_n_l275_275029

-- Define the scenario for n = 7
def probability_empty_chair_7 : Prop :=
  ∃ p : ℚ, p = 1 / 35

-- Theorem for the case when there are exactly 7 chairs
theorem probability_case_7 : probability_empty_chair_7 :=
begin
  -- Proof skipped
  sorry
end

-- Define the general scenario for n chairs (n >= 6)
def probability_empty_chair_n (n : ℕ) (h : n ≥ 6) : Prop :=
  ∃ p : ℚ, p = (n - 4) * (n - 5) / ((n - 1) * (n - 2))

-- Theorem for the case when there are n chairs (n ≥ 6)
theorem probability_case_n (n : ℕ) (h : n ≥ 6) : probability_empty_chair_n n h :=
begin
  -- Proof skipped
  sorry
end

end probability_case_7_probability_case_n_l275_275029


namespace gcd_fact_8_10_l275_275661

-- Definitions based on the conditions in a)
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Question and conditions translated to a proof problem in Lean
theorem gcd_fact_8_10 : Nat.gcd (fact 8) (fact 10) = 40320 := by
  sorry

end gcd_fact_8_10_l275_275661


namespace range_of_m_l275_275328

theorem range_of_m
  (m : ℝ) (h1 : ∀ x : ℝ, -x^2 + 8x + 20 ≥ 0 → x^2 - 2x + 1 - m^2 ≤ 0)
  (h2 : ¬ ∀ x : ℝ, x^2 - 2x + 1 - m^2 ≤ 0 → -x^2 + 8x + 20 ≥ 0) :
  m ≥ 9 :=
by
  sorry

end range_of_m_l275_275328


namespace pond_to_field_ratio_l275_275844

theorem pond_to_field_ratio 
  (w l : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : l = 28)
  (side_pond : ℝ := 7) 
  (A_pond : ℝ := side_pond ^ 2) 
  (A_field : ℝ := l * w):
  (A_pond / A_field) = 1 / 8 :=
by
  sorry

end pond_to_field_ratio_l275_275844


namespace range_f_when_a_is_2_range_of_a_l275_275704

-- Definition of the function f for given 'a' and 'x'.
def f (a : ℝ) (x : ℝ) : ℝ := 
  2 * a * sin (2 * x) + (a - 1) * (sin x + cos x) + 2 * a - 8

-- Part (1): Proof statement for the range of f when a = 2.
theorem range_f_when_a_is_2 :
  set.range (f 2) = set.Icc (-129 / 16) (-3) :=
sorry

-- Part (2): Proof statement for the range of 'a' where the given condition holds.
theorem range_of_a (a : ℝ) (x1 x2 : ℝ)
  (h1 : x1 ∈ set.Icc (-π / 2) 0) 
  (h2 : x2 ∈ set.Icc (-π / 2) 0) :
  (| f a x1 - f a x2 | ≤ a^2 + 1) ↔ (a = 1 ∨ a >= (17 + real.sqrt 257) / 16) :=
sorry

end range_f_when_a_is_2_range_of_a_l275_275704


namespace exists_k_good_function_l275_275416

def positive_integers := { n : ℕ // n > 0 }

def k_good (k : ℕ) (f : positive_integers → positive_integers) : Prop :=
∀ (m n : positive_integers), m ≠ n → gcd (f m + n) (f n + m) ≤ k

theorem exists_k_good_function (k : ℕ) (hk : k ≥ 2) :
  ∃ f : positive_integers → positive_integers, k_good k f :=
sorry

end exists_k_good_function_l275_275416


namespace segment_area_l275_275597

noncomputable def area_segment_above_triangle (a b c : ℝ) (triangle_area : ℝ) (y : ℝ) :=
  let ellipse_area := Real.pi * a * b
  ellipse_area - triangle_area

theorem segment_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 1) :
  let y := (4 * Real.sqrt 2) / 3
  let triangle_area := (1 / 2) * (2 * (b - y))
  area_segment_above_triangle a b c triangle_area y = 6 * Real.pi - 2 + (4 * Real.sqrt 2) / 3 := by
  sorry

end segment_area_l275_275597


namespace exists_sums_of_squares_and_product_l275_275428

theorem exists_sums_of_squares_and_product
  (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_condition: a * b > c^2) : 
  ∃ (n : ℕ) (x y : fin n → ℤ), 
    (∑ i, x i ^ 2 = a) ∧ 
    (∑ i, y i ^ 2 = b) ∧ 
    (∑ i, x i * y i = c) := 
sorry

end exists_sums_of_squares_and_product_l275_275428


namespace harlys_dogs_left_l275_275717

-- Define the initial conditions
def initial_dogs : ℕ := 80
def adoption_percentage : ℝ := 0.40
def dogs_taken_back : ℕ := 5

-- Compute the number of dogs left
theorem harlys_dogs_left : (80 - (int((0.40 * 80).to_nat) - 5)) = 53 :=
by
  sorry

end harlys_dogs_left_l275_275717


namespace find_CE_length_l275_275784

-- Definitions and conditions
def right_triangle (A B C : Type) [MetricSpace A] [AffineSpace ℝ A] : Prop :=
  ∃ right_angle (at_right_angle : B) (hypotenuse : Segment A B),
    ∠ABC = 90 ∧ ∠BCA = 90 ∧ ∠CAB = 90

noncomputable def area_of_triangle {A B C : Type} [MetricSpace A] [AffineSpace ℝ A] : ℝ := sorry

noncomputable def intersects (circle : Set Point) (line : Line) (E : Point) : Prop := sorry

-- Stating the main problem
theorem find_CE_length {A B C E : Point} 
  (h_right : right_triangle A B C)
  (h_circle : Circle (A ∧ C))
  (h_intersect : intersects h_circle (line_through B C) E)
  (h_area : area_of_triangle A B C = 180)
  (BC : Length (segment B C) = 30) :
  Length (segment C E) = 12 := 
sorry

end find_CE_length_l275_275784


namespace red_higher_than_green_l275_275921

open ProbabilityTheory

noncomputable def prob_bin_k (k : ℕ) : ℝ :=
  (2:ℝ)^(-k)

noncomputable def prob_red_higher_than_green : ℝ :=
  ∑' (k : ℕ), (prob_bin_k k) * (prob_bin_k (k + 1))

theorem red_higher_than_green :
  (∑' (k : ℕ), (2:ℝ) ^ (-k) * (2:ℝ) ^(-(k + 1))) = 1/3 :=
  by
  sorry

end red_higher_than_green_l275_275921


namespace original_decimal_number_l275_275074

theorem original_decimal_number (x : ℝ) (h : x / 100 = x - 1.485) : x = 1.5 := 
by
  sorry

end original_decimal_number_l275_275074


namespace exists_set_with_neighbors_l275_275084

-- Define the statement of the problem in Lean.
theorem exists_set_with_neighbors :
  ∃ (S : Finset (ℝ × ℝ)), 
    S.card = 3 ^ 1000 ∧
    ∀ P ∈ S, (Finset.filter (λ Q, (dist P Q) = 1) S).card ≥ 2000 :=
sorry

end exists_set_with_neighbors_l275_275084


namespace no_bounded_a_exists_l275_275431

noncomputable def a (n : ℕ) : ℕ :=
  -- function a that counts the number of different representations of n as a sum of different divisors

theorem no_bounded_a_exists :
  ¬ ∃ M : ℕ, ∀ n : ℕ, a(n) ≤ M := by
  sorry

end no_bounded_a_exists_l275_275431


namespace polynomial_munificence_l275_275657

def polynomial (x : ℝ) : ℝ := x^3 + x^2 - 3*x + 1

def interval : set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

def munificence (p : ℝ → ℝ) (s : set ℝ) : ℝ :=
  Sup (set.image (λ x, |p x|) s)

theorem polynomial_munificence :
  munificence polynomial interval = 4 :=
sorry

end polynomial_munificence_l275_275657


namespace line_eq_of_midpoint_and_hyperbola_l275_275336

theorem line_eq_of_midpoint_and_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : 9 * (8 : ℝ)^2 - 16 * (3 : ℝ)^2 = 144)
    (h2 : x1 + x2 = 16) (h3 : y1 + y2 = 6) (h4 : 9 * x1^2 - 16 * y1^2 = 144) (h5 : 9 * x2^2 - 16 * y2^2 = 144) :
    3 * (8 : ℝ) - 2 * (3 : ℝ) - 18 = 0 :=
by
  -- The proof steps would go here
  sorry

end line_eq_of_midpoint_and_hyperbola_l275_275336


namespace exists_non_monochromatic_coloring_l275_275678

-- Defining the set of points and properties
def P := Finset (Fin 1994)
axiom noncollinear : ∀ₓ P₁ P₂ P₃ ∈ P, P₁ ≠ P₂ → P₂ ≠ P₃ → P₁ ≠ P₃ → ¬ collinear P₁ P₂ P₃

-- Partition into 83 groups
def partition : Finset (Finset (Fin 1994)) := sorry
axiom partition_props : partition.card = 83 ∧ ∀ₓ g ∈ partition, 3 ≤ g.card

-- Defining the graph G*
def G_star : SimpleGraph (Fin 1994) := sorry
axiom G_star_min_tris : (G_star.num_triangles P) = (min_tris G_star)

-- Coloring axiom
axiom coloring : ∃ (coloring : G_star.Edge → Fin 4), ∀ₓ (a b c : Fin 1994), a ≠ b → b ≠ c → a ≠ c → ¬ monochromatic_triangle coloring a b c

-- Main theorem statement
theorem exists_non_monochromatic_coloring : ∃ coloring, ∀ₓ (a b c ∈ P), G_star.Adj a b → G_star.Adj b c → G_star.Adj a c → coloring (G_star.EdgeSet a b) ≠ coloring (G_star.EdgeSet b c) ∨ coloring (G_star.EdgeSet b c) ≠ coloring (G_star.EdgeSet a c) ∨ coloring (G_star.EdgeSet a c) ≠ coloring (G_star.EdgeSet a b) :=
sorry

end exists_non_monochromatic_coloring_l275_275678


namespace smallest_possible_sector_angle_l275_275022

theorem smallest_possible_sector_angle :
  ∃ (a_1 d : ℕ), (∀ i : ℕ, i ∈ Finset.range 8 → Nat.Prime (a_1 + i * d)) ∧
    (Finset.range 8).sum (λ i, a_1 + i * d) = 360 ∧
    (a_1 + 7 * d).Prime ∧
    (2 * a_1 + 7 * d = 90) →
  a_1 = 5 :=
by
  sorry

end smallest_possible_sector_angle_l275_275022


namespace total_drivers_l275_275853

theorem total_drivers (N : ℕ) (A : ℕ) (sA sB sC sD : ℕ) (total_sampled : ℕ)
  (hA : A = 96) (hsA : sA = 12) (hsB : sB = 21) (hsC : sC = 25) (hsD : sD = 43) (htotal : total_sampled = sA + sB + sC + sD)
  (hsA_proportion : (sA : ℚ) / A = (total_sampled : ℚ) / N) : N = 808 := by
  sorry

end total_drivers_l275_275853


namespace root_product_l275_275147

theorem root_product : (Real.sqrt (Real.sqrt 81) * Real.cbrt 27 * Real.sqrt 9 = 27) :=
by
  sorry

end root_product_l275_275147


namespace right_triangle_if_orthocenter_lies_nine_point_circle_l275_275082

noncomputable def orthocenter (ABC: Triangle) : Point := sorry
def nine_point_circle (ABC: Triangle) : Circle := sorry
def lies_on (p: Point) (c: Circle) : Prop := sorry

theorem right_triangle_if_orthocenter_lies_nine_point_circle {ABC : Triangle} (H := orthocenter ABC) (N := nine_point_circle ABC)
  (hH : lies_on H N) :
  (is_right_triangle ABC) :=
sorry

end right_triangle_if_orthocenter_lies_nine_point_circle_l275_275082


namespace equal_areas_of_triangles_l275_275857

theorem equal_areas_of_triangles 
  (A B C P A1 B1 C1 A2 B2 C2 : Euclidean_A) 
  (h1 : line_through P A1 ∥ line_through B C) 
  (h2 : line_through P B1 ∥ line_through C A) 
  (h3 : line_through P C1 ∥ line_through A B) 
  (h4 : line_through P A2 ∥ line_through B C) 
  (h5 : line_through P B2 ∥ line_through C A) 
  (h6 : line_through P C2 ∥ line_through A B) : 
  area (triangle A1 B1 C1) = area (triangle A2 B2 C2) := 
sorry

end equal_areas_of_triangles_l275_275857


namespace ant_trip_ratio_l275_275862

theorem ant_trip_ratio (A B : ℕ) (x c : ℕ) (h1 : A * x = c) (h2 : B * (3 / 2 * x) = 3 * c) :
  B = 2 * A :=
by
  sorry

end ant_trip_ratio_l275_275862


namespace op_4_3_equals_23_l275_275365

def op (a b : ℕ) : ℕ := a ^ 2 + a * b + a - b ^ 2

theorem op_4_3_equals_23 : op 4 3 = 23 := by
  -- Proof steps would go here
  sorry

end op_4_3_equals_23_l275_275365


namespace parameter_b_solution_exists_l275_275647

theorem parameter_b_solution_exists (b : ℝ) :
  (∃ a x y : ℝ, (x - b) ^ 2 + (y + b) ^ 2 = 4 ∧ y = 9 / ((x + a) ^ 2 + 1)) ↔ b ∈ set.Ico (-(11:ℝ)) 2 :=
by sorry

end parameter_b_solution_exists_l275_275647


namespace imaginary_part_of_complex_l275_275665

theorem imaginary_part_of_complex :
  let z := (2 - complex.I) / complex.I in
  z.im = -2 :=
by
  let z := (2 - complex.I) / complex.I
  have z_value : z = -1 - 2 * complex.I :=
    by
      -- simplification steps
      have := complex.div (2 - complex.I) complex.I
      rw [complex.div_eq_mul_conj] at this
      simp only [complex.I_re, complex.I_im, complex.conj_I, 
                 complex.of_real_zero, complex.of_real_one] at this
      simp [complex.I_sq] at this
      exact this
  have z_imag : z.im = (-1 - 2 * complex.I).im := congr_arg complex.im z_value
  simp [complex.of_real_im] at z_imag
  exact z_imag

end imaginary_part_of_complex_l275_275665


namespace four_digit_divisors_l275_275973

theorem four_digit_divisors :
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 →
  (1000 * a + 100 * b + 10 * c + d ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * d + 100 * a + 10 * b + c) →
  ∃ (e f : ℕ), e = a ∧ f = b ∧ (e ≠ 0 ∧ f ≠ 0) ∧ (1000 * e + 100 * e + 10 * f + f = 1000 * a + 100 * b + 10 * a + b) ∧
  (1000 * e + 100 * e + 10 * f + f ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * d + 100 * a + 10 * b + c) := 
by
  sorry

end four_digit_divisors_l275_275973


namespace function_varies_between_bounds_l275_275090

theorem function_varies_between_bounds (k : ℝ) (hk : k > 1) :
  ∀ x : ℝ, 
    let f := (x^2 - 2*x + k^2) / (x^2 + 2*x + k^2) in
    f ≥ (k - 1) / (k + 1) ∧ f ≤ (k + 1) / (k - 1) :=
sorry

end function_varies_between_bounds_l275_275090


namespace find_valid_ns_l275_275777

-- Define the divisor function d
def d (n : ℕ) : ℕ :=
  if h : n > 0 then nat.divisors n h else 0

-- Define the main property
def valid_n (n : ℕ) : Prop :=
  d (n - 1) + d n + d (n + 1) ≤ 8

-- Prove the main result
theorem find_valid_ns (n : ℕ) (h : n ≥ 3) : valid_n n ↔ n = 3 ∨ n = 4 ∨ n = 6 :=
by {
  sorry
}

end find_valid_ns_l275_275777


namespace problem_statement_l275_275378

theorem problem_statement :
  (∀ x y : ℤ, x * y = x * y - 2 * (x + y)) → (let z := (1 * (-3)) in z = 1) := by
  intros h
  let z : ℤ := 1 * (-3)
  sorry

end problem_statement_l275_275378


namespace percentage_increase_is_50_l275_275444

-- Definition of the conditions
def first_pair_cost : ℝ := 22
def total_cost : ℝ := 55
def second_pair_cost : ℝ := total_cost - first_pair_cost

-- Condition that the second pair of shoes is more expensive
def second_pair_is_more_expensive : Prop := second_pair_cost > first_pair_cost

-- Calculation of percentage increase
def percentage_increase : ℝ := ((second_pair_cost - first_pair_cost) / first_pair_cost) * 100

-- Proof statement
theorem percentage_increase_is_50 :
  second_pair_is_more_expensive → percentage_increase = 50 := 
begin
  assume h,
  sorry
end

end percentage_increase_is_50_l275_275444


namespace shaded_region_area_l275_275004

noncomputable def area_of_shaded_region (a b c d : ℝ) (area_rect : ℝ) : ℝ :=
  let dg : ℝ := (a * d) / (c + d)
  let area_triangle : ℝ := 0.5 * dg * b
  area_rect - area_triangle

theorem shaded_region_area :
  area_of_shaded_region 12 5 12 4 (4 * 5) = 85 / 8 :=
by
  simp [area_of_shaded_region]
  sorry

end shaded_region_area_l275_275004


namespace find_b_l275_275471

theorem find_b (b : ℤ) (h0 : 0 ≤ b) (h1 : b ≤ 16) 
(h2 : ∃ k : ℤ, 352037216 = 67^0 * 6 + 67^1 * 1 + 67^2 * 2 + 67^3 * 7 + 
               67^4 * 3 + 67^5 * 0 + 67^6 * 2 + 67^7 * 3 + 67^8 * 5 - b = k * 17) : 
b = 7 := 
sorry

end find_b_l275_275471


namespace range_f_l275_275162

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_f : Set.Ioo 0 1 ∪ {1} = {y : ℝ | ∃ x : ℝ, f x = y} :=
by 
  sorry

end range_f_l275_275162


namespace complex_quadrant_l275_275761

-- Define the complex number z
def z : ℂ := 1 + complex.i

-- Define the complex expression
def complex_expr : ℂ := (2 / z) + z^2

-- Proof statement to show that the point corresponding to this complex number is in the first quadrant
theorem complex_quadrant (h : z = 1 + complex.i) : complex_expr.re > 0 ∧ complex_expr.im > 0 := by
  -- Proof is omitted
  sorry

end complex_quadrant_l275_275761


namespace proof_problem_l275_275690

noncomputable def f : ℝ → ℝ := sorry -- specification of f is not provided directly in the problem

-- conditions
axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom reciprocal_f : ∀ x : ℝ, f (x + 1) = 1 / f x
axiom decreasing_f : ∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f x > f y

def a := f (Real.log 2 / Real.log 0.5)
def b := f (Real.log 4 / Real.log 2)
def c := f (Real.sqrt 2 - 2)

theorem proof_problem : a > c ∧ c > b :=
by
  rw [eq_of_neg x_eq, eq_of_sqrt2 log_evidence_magic]
  sorry

end proof_problem_l275_275690


namespace expected_value_D4_l275_275027

noncomputable def kelvin_the_frog.D_fourth_moment_expected_value : ℝ :=
  ∑ i in finset.range 10, (Real.cos θs i)^2 + ∑ i in finset.range 10, (Real.sin θs i)^2

theorem expected_value_D4 (θs : Fin 10 → ℝ) (h_uniform : ∀ i, 0 ≤ θs i ∧ θs i ≤ 2 * Real.pi) :
  ∑ i in finset.range 10, (Real.cos θs i)^2 +
  ∑ i in finset.range 10, (Real.sin θs i)^2 = 200 := 
sorry

end expected_value_D4_l275_275027


namespace problem_l275_275795

noncomputable def f (a x : ℝ) : ℝ := x^2 + a

def f_iter (f : ℝ → ℝ → ℝ) (n : ℕ) (a x : ℝ) : ℝ :=
  nat.rec_on n (f a x) (λ k ih, f a ih)

def setM : Set ℝ := { a | ∀ n : ℕ, |f_iter f n a 0| ≤ 2 }

theorem problem : setM = { a | -2 ≤ a ∧ a ≤ 1/4 } :=
by
  sorry

end problem_l275_275795


namespace squares_with_equal_side_lengths_l275_275212

noncomputable def square_division (square_side : ℝ)
    (parallel_lines1 parallel_lines2 : Finset ℝ) : Prop :=
  ∃ (rectangles : Finset (Set (ℝ × ℝ))),
    parallel_lines1.card = 9 ∧
    parallel_lines2.card = 9 ∧
    rectangles.card = 100 ∧
    ∃ (squares : Finset (Set (ℝ × ℝ))),
      squares.card = 9 ∧
      ∀ s ∈ squares, ∃ (side : ℝ), Set.eq_on (fun p : ℝ × ℝ => p.1 = side ∧ p.2 = side) s s.id

theorem squares_with_equal_side_lengths :
  ∀ (square_side : ℝ)
    (parallel_lines1 parallel_lines2 : Finset ℝ),
     square_division square_side parallel_lines1 parallel_lines2 →
     ∃ (s : (Set (ℝ × ℝ))), ∃ (t : (Set (ℝ × ℝ))), s ≠ t ∧ (s ∈ squares ∧ t ∈ squares) ∧ 
     ∃ a : ℝ, Set.eq_on (fun p : ℝ × ℝ => p.1 = a ∧ p.2 = a) s side ∧ Set.eq_on (fun p : ℝ × ℝ => p.1 = a ∧ p.2 = a) t side :=
sorry

end squares_with_equal_side_lengths_l275_275212


namespace region_A_area_correct_l275_275252

noncomputable def region_A_area : ℝ := 1200 - 200 * Real.pi

theorem region_A_area_correct : 
  let A := {z : ℂ | let x := z.re, y := z.im in 
              0 ≤ x ∧ x ≤ 40 ∧ 0 ≤ y ∧ y ≤ 40 ∧ 
              40 * x ≤ x^2 + y^2 ∧ 40 * y ≤ x^2 + y^2} 
  in (∃! (a : ℝ), a = region_A_area) 
:= sorry

end region_A_area_correct_l275_275252


namespace share_difference_l275_275595

theorem share_difference (x : ℕ) (p q r : ℕ) 
  (h1 : 3 * x = p) 
  (h2 : 7 * x = q) 
  (h3 : 12 * x = r) 
  (h4 : q - p = 2800) : 
  r - q = 3500 := by {
  sorry
}

end share_difference_l275_275595


namespace domain_of_f_l275_275486

def f (x : ℝ) := Real.sqrt (x * (x - 1))

theorem domain_of_f :
  {x : ℝ | x * (x - 1) ≥ 0} = {x : ℝ | x ≤ 0 ∨ x ≥ 1} :=
by
  sorry

end domain_of_f_l275_275486


namespace fundamental_theorem_arithmetic_l275_275250

theorem fundamental_theorem_arithmetic (n : ℕ) (h : n > 1) :
  Prime n ∨ ∃ p : ℕ, Prime p ∧ p * ∃ pi : ℕ, Prime pi ∧ pi * n = p := {
sorry } -- Proof omitted

end fundamental_theorem_arithmetic_l275_275250


namespace number_of_distinct_outfits_l275_275824

theorem number_of_distinct_outfits :
  let shirts := 7 in
  let pants := 5 in
  let options_for_ties := 7 in
  let options_for_hats := 5 in
  (shirts * pants * options_for_ties * options_for_hats) = 1225 :=
by
  sorry

end number_of_distinct_outfits_l275_275824


namespace solve_polynomial_equation_l275_275093

theorem solve_polynomial_equation :
  ∃ z, (z^5 + 40 * z^3 + 80 * z - 32 = 0) →
  ∃ x, (x = z + 4) ∧ ((x - 2)^5 + (x - 6)^5 = 32) :=
by
  sorry

end solve_polynomial_equation_l275_275093


namespace team_B_eliminated_after_three_matches_team_A_wins_championship_in_four_matches_fifth_match_needed_l275_275464

-- Defining the probabilities for win scenarios
def P₁ : ℝ := 2 / 3  -- Probability Team A wins against Team B
def P₂ : ℝ := 2 / 3  -- Probability Team A wins against Team C
def P₃ : ℝ := 1 / 2  -- Probability Team B wins against Team C

-- Condition: After three matches, Team B is eliminated
def prob_team_B_eliminated_after_three_matches : ℝ := (1 / 2) * (1 / 3) * (1 / 2) + (1 / 2) * (2 / 3) * (2 / 3)

-- Problem (1): Prove the probability that Team B is eliminated after the first three matches is 7/18
theorem team_B_eliminated_after_three_matches : prob_team_B_eliminated_after_three_matches = 7 / 18 :=
  sorry

-- Problem (2): Prove the probability that Team A wins the championship in only four matches is 8/27
theorem team_A_wins_championship_in_four_matches : (P₁ * P₂ * P₁) = 8 / 27 :=
  sorry

-- Problem (3): Prove the probability that a fifth match is needed
theorem fifth_match_needed : (1 - (P₁ * P₂ * P₁)) > 0 := 
  sorry

end team_B_eliminated_after_three_matches_team_A_wins_championship_in_four_matches_fifth_match_needed_l275_275464


namespace incorrect_student_B_l275_275116

-- Definitions for the conditions
def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y : ℝ, x ≤ y → f x ≥ f y
def monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y : ℝ, x ≤ y → f x ≤ f y
def symmetric_about_x1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (1 - x) = f (1 + x)
def not_min_value (f : ℝ → ℝ) : Prop := ∃ y : ℝ, f 0 > f y

variables (f : ℝ → ℝ)

-- The main proof statement
theorem incorrect_student_B :
  (monotonically_decreasing f (-∞) 0)
  ∧ (monotonically_increasing f 0 ∞)
  ∧ (symmetric_about_x1 f)
  ∧ (not_min_value f)
  ∧ ((monotonically_decreasing f (-∞) 0 ∧ monotonically_increasing f 0 ∞ ∧ symmetric_about_x1 f)
     ∨ (monotonically_decreasing f (-∞) 0 ∧ monotonically_increasing f 0 ∞ ∧ not_min_value f)
     ∨ (monotonically_decreasing f (-∞) 0 ∧ symmetric_about_x1 f ∧ not_min_value f)
     ∨ (monotonically_increasing f 0 ∞ ∧ symmetric_about_x1 f ∧ not_min_value f)) → false :=
begin
  sorry
end

end incorrect_student_B_l275_275116


namespace right_triangle_tan_l275_275753

theorem right_triangle_tan (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (angle_BAC : MeasureTheory.Angle B A C = MeasureTheory.Angle.right)
  (AB_len : dist A B = 40)
  (BC_len : dist B C = 41) :
  tangent_of_angle A B C = 9 / 40 :=
sorry

end right_triangle_tan_l275_275753


namespace probability_seating_7_probability_seating_n_l275_275040

-- Definitions and theorem for case (a): n = 7
def num_ways_to_seat_7 (n : ℕ) (k : ℕ) : ℕ := n * (n - 1) * (n - 2) / k!

def valid_arrangements_7 : ℕ := 1

theorem probability_seating_7 : 
  let total_ways := num_ways_to_seat_7 7 3 in
  let valid_ways := valid_arrangements_7 in
  total_ways = 35 ∧ (valid_ways : ℚ) / total_ways = 0.2 := 
by
  sorry

-- Definitions and theorem for case (b): general n
def choose (n k : ℕ) : ℕ := n * (n - 1) / 2

def valid_arrangements_n (n : ℕ) : ℚ := (n - 4) * (n - 5) / 2

theorem probability_seating_n (n : ℕ) (hn : n ≥ 6) : 
  let total_ways := choose (n - 1) 2 in
  let valid_ways := valid_arrangements_n n in
  total_ways = (n - 1) * (n - 2) / 2 ∧ 
  (valid_ways : ℚ) / total_ways = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by
  sorry

end probability_seating_7_probability_seating_n_l275_275040


namespace R_lies_on_BD_l275_275797

theorem R_lies_on_BD
  (A B C D S P Q R : Type)
  (cyc_quad : convex_cyclic_quadrilateral A B C D)
  (inter_diags : diagonal_int_at S A B C D)
  (circumcenter_ABS : is_circumcenter P A B S)
  (circumcenter_BCS : is_circumcenter Q B C S)
  (parallel_AD_through_P : parallel_line P AD)
  (parallel_CD_through_Q : parallel_line Q CD)
  (int_parallel_lines_at_R : intersect_parallel_at R P Q) :
  lies_on BD R := sorry

end R_lies_on_BD_l275_275797


namespace degree_of_monomial_l275_275485

-- Given monomial of form -5a^2b^3
def monomial := (-5 : ℝ) * (a^2 : ℕ) * (b^3 : ℕ)

-- Prove that the degree of the monomial -5a^2b^3 is 5
theorem degree_of_monomial : 
  let degree_of (m : ℝ) := 5 in
  degree_of (monomial) = 5 :=
sorry

end degree_of_monomial_l275_275485


namespace find_magnitude_of_b_l275_275332

variables {a b : ℝ} -- where these real numbers will act as magnitudes of vectors a and b

-- Conditions
def angle_between_a_and_b := real.pi / 3
def magnitude_of_a := 1
def magnitude_of_a_plus_b := real.sqrt 7

theorem find_magnitude_of_b (h : a = 1) (h1 : real.sqrt (a^2 + b^2 + 2*a*b*(real.cos (real.pi / 3))) = real.sqrt 7) : b = 2 := by
  sorry

end find_magnitude_of_b_l275_275332


namespace num_possibilities_l275_275194

def last_digit_divisible_by_4 (n : Nat) : Prop := (60 + n) % 4 = 0

theorem num_possibilities : {n : Nat | n < 10 ∧ last_digit_divisible_by_4 n}.card = 3 := by
  sorry

end num_possibilities_l275_275194


namespace find_cd_l275_275970

theorem find_cd : 
  (∀ x : ℝ, (4 * x - 3) / (x^2 - 3 * x - 18) = ((7 / 3) / (x - 6)) + ((5 / 3) / (x + 3))) :=
by
  intro x
  have h : x^2 - 3 * x - 18 = (x - 6) * (x + 3) := by
    sorry
  rw [h]
  sorry

end find_cd_l275_275970


namespace number_of_sets_l275_275119

theorem number_of_sets (A : Set ℕ) : 
  (∃ A, ∀ x, x ∈ {1, 2} ∪ A ↔ x ∈ {1, 2, 3}) → 
    {A | ∀ x, x ∈ {1, 2} ∪ A ↔ x ∈ {1, 2, 3}}.toFinset.card = 4 := 
by 
  sorry

end number_of_sets_l275_275119


namespace not_divisible_into_hierarchical_subsets_l275_275621

theorem not_divisible_into_hierarchical_subsets (A : Finset ℕ) (hA : A = Finset.range 14 \ Finset.singleton 0) :
  ¬(∃ P : Finset (Finset ℕ), 
    (∀ S ∈ P, S.card > 1 ∧ (∃ x ∈ S, x = (S.erase x).sum)) ∧
    (P.val = A.val)) := 
by
  sorry

end not_divisible_into_hierarchical_subsets_l275_275621


namespace tan_A_value_l275_275755

-- Define the conditions provided
variables (A B C : Type) [EuclideanGeometry A] -- A, B, and C are points in Euclidean geometry
variable (ABC : Triangle A B C) -- Triangle exists with vertices A, B, C
variable (h_angle : Angle A B C = 90) -- Given angle BAC is a right angle
variable (AB AC BC : ℝ) -- Side lengths of the triangle
variable (h_AB : AB = 40) -- Given length of AB
variable (h_BC : BC = 41) -- Given length of BC

-- Define the Pythagorean theorem calculation for AC
def AC : ℝ := sqrt (BC^2 - AB^2)
-- Applying the Pythagorean theorem given the conditions
axiom h_pythagorean : AC = sqrt (41^2 - 40^2)
axiom h_AC : AC = 9

-- Definition for the tangent of angle A
def tan_A := AC / AB

-- Prove the required value of tangent
theorem tan_A_value : tan_A = 9/40 := by
  -- Introduction and calculations here
  sorry -- Proof is skipped as per instructions

end tan_A_value_l275_275755


namespace median_high_jump_heights_l275_275948

noncomputable def high_jump_heights : list ℝ :=
  [1.50, 1.50, 1.60, 1.60, 1.60, 1.65, 1.65, 1.65, 1.70, 1.70, 1.75, 1.75, 1.75, 1.75, 1.80]

theorem median_high_jump_heights :
  ∃ (median : ℝ), median = 1.65 ∧
  let sorted_heights := high_jump_heights in (
    sorted_heights.length = 15 ∧
    sorted_heights.nth (sorted_heights.length / 2) = some median
  ) :=
by
  sorry

end median_high_jump_heights_l275_275948


namespace exponential_only_function_satisfying_add_mult_l275_275220

theorem exponential_only_function_satisfying_add_mult :
  (∃ f : ℝ → ℝ, (∀ x y, x > 0 ∧ y > 0 → f(x + y) = f(x) * f(y)) ∧
                 (∀ z, ∃ a, f(z) = a^z)) ∧
  ¬ (∃ f : ℝ → ℝ, (∀ x y, x > 0 ∧ y > 0 → f(x + y) = f(x) * f(y)) ∧
                   (∀ z, ∃ a, f(z) = log a z)) ∧
  ¬ (∃ f : ℝ → ℝ, (∀ x y, x > 0 ∧ y > 0 → f(x + y) = f(x) * f(y)) ∧
                   (∀ z, ∃ a, f(z) = z ^ a)) ∧
  ¬ (∃ f : ℝ → ℝ, (∀ x y, x > 0 ∧ y > 0 → f(x + y) = f(x) * f(y)) ∧
                   (∀ z, f(z) = a * z + b)) :=
by
  sorry

end exponential_only_function_satisfying_add_mult_l275_275220


namespace perimeter_of_H_is_two_pi_l275_275713

-- Definitions based on conditions
def Circle (center : Point) (radius : ℝ) : Set Point := sorry
def intersects_at_two_points (c1 c2 : Set Point) : Prop := sorry
def separates (c3 : Set Point) (p1 p2 : Point) : Prop := sorry

-- Given conditions
variables {c1 c2 c3 : Set Point}
variable radius : ℝ

-- Conditions specific to this problem
def conditions (c1 c2 c3 : Set Point) (radius : ℝ) : Prop :=
  (radius = 1 ∧
   intersects_at_two_points c1 c2 ∧
   intersects_at_two_points c2 c3 ∧
   intersects_at_two_points c1 c3 ∧
   ∃ p1 p2 p3 p4 p5 p6 : Point,
     separates c3 p1 p2 ∧
     separates c1 p3 p4 ∧
     separates c2 p5 p6)

-- Proof statement
theorem perimeter_of_H_is_two_pi 
  (h_conditions : conditions c1 c2 c3 radius) :
  ∃ (H : Set Point), (∀ p, p ∈ H ↔ (p ∈ c1 ∧ p ∈ c2) ∨ (p ∈ c2 ∧ p ∈ c3) ∨ (p ∈ c3 ∧ p ∈ c1)) ∧ perimeter H = 2 * real.pi := sorry

end perimeter_of_H_is_two_pi_l275_275713


namespace amy_music_files_l275_275224

-- Define the number of total files on the flash drive
def files_on_flash_drive := 48.0

-- Define the number of video files on the flash drive
def video_files := 21.0

-- Define the number of picture files on the flash drive
def picture_files := 23.0

-- Define the number of music files, derived from the conditions
def music_files := files_on_flash_drive - (video_files + picture_files)

-- The theorem we need to prove
theorem amy_music_files : music_files = 4.0 := by
  sorry

end amy_music_files_l275_275224


namespace sin_cos_sum_l275_275247

theorem sin_cos_sum : 
  sin (11 / 6 * Real.pi) + cos (10 / 3 * Real.pi) = -1 :=
by
  have h1 : sin ((2 - 1 / 6) * Real.pi) = sin (2 * Real.pi - 1 / 6 * Real.pi),
    from sorry,
  have h2 : cos ((3 + 1 / 3) * Real.pi) = cos (3 * Real.pi + 1 / 3 * Real.pi),
    from sorry,
  have h3 : sin (2 * Real.pi - 1 / 6 * Real.pi) = - sin (1 / 6 * Real.pi),
    from sorry,
  have h4 : cos (3 * Real.pi + 1 / 3 * Real.pi) = - cos (1 / 3 * Real.pi),
    from sorry,
  have s1 : sin (1 / 6 * Real.pi) = 1 / 2,
    from sorry,
  have c1 : cos (1 / 3 * Real.pi) = 1 / 2,
    from sorry,
  calc
    sin (11 / 6 * Real.pi) + cos (10 / 3 * Real.pi)
        = sin (2 * Real.pi - 1 / 6 * Real.pi) + cos (3 * Real.pi + 1 / 3 * Real.pi) : by rw [h1, h2]
    ... = - sin (1 / 6 * Real.pi) - cos (1 / 3 * Real.pi) : by rw [h3, h4]
    ... = - (1 / 2) - (1 / 2) : by rw [s1, c1]
    ... = -1 : by norm_num

end sin_cos_sum_l275_275247


namespace susie_spent_on_skincare_fraction_l275_275825

-- Definitions based on given conditions
def daily_hours : ℕ := 3
def hourly_rate : ℕ := 10
def days_in_week : ℕ := 7
def spent_on_makeup_fraction : ℚ := 3 / 10
def money_left_after_skincare : ℕ := 63

-- Main proof structure
theorem susie_spent_on_skincare_fraction :
  let weekly_earnings := daily_hours * hourly_rate * days_in_week in
  let spent_on_makeup := spent_on_makeup_fraction * weekly_earnings in
  let money_after_makeup := weekly_earnings - spent_on_makeup in
  let spent_on_skincare := money_after_makeup - money_left_after_skincare in
  (spent_on_skincare / weekly_earnings : ℚ) = 2 / 5 :=
by
  sorry

end susie_spent_on_skincare_fraction_l275_275825


namespace symmetric_pairs_count_is_two_l275_275759

def points_are_symmetric_about_origin (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f a = b ∧ f (-a) = -b

noncomputable def g : ℝ → ℝ 
| x => if x ≤ 0 then cos (π / 2 * x) else log (4, x + 1)

theorem symmetric_pairs_count_is_two :
  set.countable (set_of (points_are_symmetric_about_origin g)) = 2 :=
by sorry

end symmetric_pairs_count_is_two_l275_275759


namespace train_length_approx_l275_275929

-- Define the speed of the train in km/hr
def speed_kmh : ℝ := 52

-- Define the time it takes to cross the pole in seconds
def time_seconds : ℝ := 18

-- Define the conversion factor from km/hr to m/s
def kmhr_to_ms : ℝ := 1000 / 3600

-- Define the speed of the train in m/s
def speed_ms : ℝ := speed_kmh * kmhr_to_ms

-- Prove that the length of the train is approximately 259.92 meters
theorem train_length_approx :
  let length := speed_ms * time_seconds in
  length ≈ 259.92 :=
by
  sorry

end train_length_approx_l275_275929


namespace area_of_ABDF_l275_275507

-- Defining the dimensions of the rectangle ACDE
def lengthAC : ℝ := 32
def widthAE : ℝ := 20

-- Defining the midpoints B and F
def midpointB_lengthAC : ℝ := lengthAC / 2
def midpointF_widthAE : ℝ := widthAE / 2

-- Defining the area of the rectangle ACDE
def areaACDE := lengthAC * widthAE

-- Defining the area of triangles BCD and EFD
def areaBCD := 0.5 * midpointB_lengthAC * widthAE
def areaEFD := 0.5 * midpointF_widthAE * lengthAC

-- Defining the area of quadrilateral ABDF
def areaABDF := areaACDE - areaBCD - areaEFD

-- Proving that the area of quadrilateral ABDF is 320
theorem area_of_ABDF :
  areaABDF = 320 := by sorry

end area_of_ABDF_l275_275507


namespace solve_system1_solve_system2_l275_275470

-- Define the conditions and the proof problem for System 1
theorem solve_system1 (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : 3 * x + 2 * y = 7) :
  x = 2 ∧ y = 1 / 2 := by
  sorry

-- Define the conditions and the proof problem for System 2
theorem solve_system2 (x y : ℝ) (h1 : x - y = 3) (h2 : (x - y - 3) / 2 - y / 3 = -1) :
  x = 6 ∧ y = 3 := by
  sorry

end solve_system1_solve_system2_l275_275470


namespace minimize_J_l275_275361

noncomputable def H (p q : ℝ) := -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) := max (9 * p - 5) (4 - 7 * p)

theorem minimize_J :
  ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ ∀ p' : ℝ, 0 ≤ p' ∧ p' ≤ 1 → J p ≤ J p' :=
begin
  use (9 / 16),
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros p' hp,
    sorry
  }

end minimize_J_l275_275361


namespace sequence_formulas_max_of_k_l275_275677

noncomputable def S (n : ℕ) := (1/2) * n^2 + (11/2) * n
noncomputable def a (n : ℕ) := n + 5
noncomputable def b (n : ℕ) := 3 * n + 2
noncomputable def c (n : ℕ) := 3 / ((2 * a n - 11) * (2 * b n - 1))
noncomputable def T (n : ℕ) := ∑ i in finset.range (n + 1), c i

theorem sequence_formulas :
  ∀ n : ℕ, (a n = n + 5) ∧ (b n = 3 * n + 2) := by
  sorry

theorem max_of_k :
  ∃ k : ℕ, (k < 19) ∧ (∀ n : ℕ, T n > k / 57) := by
  sorry

end sequence_formulas_max_of_k_l275_275677


namespace store_owed_creditors_amount_l275_275214

theorem store_owed_creditors_amount :
  let total_items := 2000
  let retail_price_per_item := 50
  let discount_rate := 0.80
  let sold_percentage := 0.90
  let remaining_amount_after_sale := 3000
  let discounted_price_per_item := retail_price_per_item * (1 - discount_rate)
  let items_sold := total_items * sold_percentage
  let total_revenue := items_sold * discounted_price_per_item
  in total_revenue - remaining_amount_after_sale = 15000 :=
by
  sorry

end store_owed_creditors_amount_l275_275214


namespace sum_values_l275_275729

noncomputable def abs_eq_4 (x : ℝ) : Prop := |x| = 4
noncomputable def abs_eq_5 (x : ℝ) : Prop := |x| = 5

theorem sum_values (a b : ℝ) (h₁ : abs_eq_4 a) (h₂ : abs_eq_5 b) :
  a + b = 9 ∨ a + b = -1 ∨ a + b = 1 ∨ a + b = -9 := 
by
  -- Proof is omitted
  sorry

end sum_values_l275_275729


namespace intersection_A_B_union_complementB_A_l275_275323

open Set

def set_A : Set ℝ := { x | 1 ≤ x ∧ x < 6 }
def set_B : Set ℝ := { x | 2 < x ∧ x < 9 }

theorem intersection_A_B :
  set_A ∩ set_B = { x | 2 < x ∧ x < 6 } :=
by {
  ext x,
  simp [set_A, set_B],
  split;
  intro h;
  linarith,
}

theorem union_complementB_A :
  (compl set_B) ∪ set_A = { x | x < 6 } ∪ { x | 9 ≤ x } :=
by {
  ext x,
  simp [set_A, set_B],
  split;
  intro h;
  linarith,
}

end intersection_A_B_union_complementB_A_l275_275323


namespace mean_median_difference_l275_275446

def scores : List (ℚ × ℚ) := 
  [(0.20, 60), (0.40, 75), (0.25, 85), (0.15, 95)]

def mean (scores: List (ℚ × ℚ)) : ℚ :=
  scores.map (fun (percent, score) => percent * score).sum

def median (scores: List (ℚ × ℚ)) : ℚ :=
  let sorted_scores := scores.sort_by (fun (_, score) => score)
  let acc_percent := 0.0
  let rec find_median (acc_percent : ℚ) (rem_scores: List (ℚ × ℚ)) : ℚ :=
    match rem_scores with
    | [] => 0 -- default case just for type-checking purposes
    | (percent, score) :: rest =>
      let new_acc := acc_percent + percent
      if new_acc >= 0.5 then score
      else find_median new_acc rest
  find_median acc_percent sorted_scores

theorem mean_median_difference : 
  mean scores - median scores = 3 :=
by
  sorry

end mean_median_difference_l275_275446


namespace letitias_order_l275_275410

noncomputable def total_tip := 3 * 4
noncomputable def total_cost (L : ℝ) := 10 + L + 30
noncomputable def tip_rate := 0.20
noncomputable def computed_tip (L : ℝ) := tip_rate * total_cost L

theorem letitias_order : ∃ L : ℝ, computed_tip L = total_tip ∧ L = 20 := by
  use 20
  unfold computed_tip
  unfold total_tip
  unfold tip_rate
  unfold total_cost
  norm_num
  sorry

end letitias_order_l275_275410


namespace radical_product_l275_275153

def fourth_root (x : ℝ) : ℝ := x ^ (1/4)
def third_root (x : ℝ) : ℝ := x ^ (1/3)
def square_root (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_product :
  fourth_root 81 * third_root 27 * square_root 9 = 27 := 
by
  sorry

end radical_product_l275_275153


namespace harly_dogs_final_count_l275_275718

theorem harly_dogs_final_count (initial_dogs : ℕ) (adopted_percentage : ℕ) (returned_dogs : ℕ) (adoption_rate : adopted_percentage = 40) (initial_count : initial_dogs = 80) (returned_count : returned_dogs = 5) :
  initial_dogs - (initial_dogs * adopted_percentage / 100) + returned_dogs = 53 :=
by
  sorry

end harly_dogs_final_count_l275_275718


namespace tower_lights_l275_275760

theorem tower_lights (a : ℕ) (h : (∑ i in finset.range 7, 2^i) * a = 381) : a = 3 :=
by
  sorry

end tower_lights_l275_275760


namespace train_cross_bridge_time_l275_275360

open Nat

-- Defining conditions as per the problem
def train_length : ℕ := 200
def bridge_length : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := speed_kmph * 5 / 18
def total_distance : ℕ := train_length + bridge_length
def time_to_cross : ℕ := total_distance / speed_mps

-- Stating the theorem
theorem train_cross_bridge_time : time_to_cross = 35 := by
  sorry

end train_cross_bridge_time_l275_275360


namespace rent_3600_yields_88_optimal_rent_is_4100_and_max_revenue_is_304200_l275_275210

/-
Via conditions:
1. The rental company owns 100 cars.
2. When the monthly rent for each car is set at 3000 yuan, all cars can be rented out.
3. For every 50 yuan increase in the monthly rent per car, there will be one more car that is not rented out.
4. The maintenance cost for each rented car is 200 yuan per month.
-/

noncomputable def num_rented_cars (rent_per_car : ℕ) : ℕ :=
  if rent_per_car < 3000 then 100 else max 0 (100 - (rent_per_car - 3000) / 50)

noncomputable def monthly_revenue (rent_per_car : ℕ) : ℕ :=
  let cars_rented := num_rented_cars rent_per_car
  let maintenance_cost := 200 * cars_rented
  (rent_per_car - maintenance_cost) * cars_rented

theorem rent_3600_yields_88 : num_rented_cars 3600 = 88 :=
  sorry

theorem optimal_rent_is_4100_and_max_revenue_is_304200 :
  ∃ rent_per_car, rent_per_car = 4100 ∧ monthly_revenue rent_per_car = 304200 :=
  sorry

end rent_3600_yields_88_optimal_rent_is_4100_and_max_revenue_is_304200_l275_275210


namespace sum_of_series_approx_l275_275610

theorem sum_of_series_approx :
  (∑ k in finset.range 4, (3^k : ℝ) / (9^k - 1)) ≈ 0.537 :=
by sorry

end sum_of_series_approx_l275_275610


namespace transformation_proof_l275_275758

theorem transformation_proof 
    (M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, a], ![b, 4]])
    (line1 line2 : ℝ × ℝ → Prop)
    (transformed : Matrix (Fin 2) (Fin 1) ℝ → Matrix (Fin 2) (Fin 1) ℝ) :
    line1 = (λ (p : ℝ × ℝ), p.1 + p.2 + 2 = 0) →
    line2 = (λ (p : ℝ × ℝ), p.1 - p.2 - 4 = 0) →
    transformed = (λ p, M.mulVec p) →
    (line2 (transformed ![-2, 0])) ∧ (line2 (transformed ![0, -2])) →
    a = 2 ∧ b = 3 :=
by
  intros hline1 hline2 htransformed hcondition
  sorry

end transformation_proof_l275_275758


namespace solve_for_y_in_terms_of_x_l275_275345

theorem solve_for_y_in_terms_of_x (x y : ℝ) (h : 2 * x - 7 * y = 5) : y = (2 * x - 5) / 7 :=
sorry

end solve_for_y_in_terms_of_x_l275_275345


namespace find_x_l275_275109

def f (x: ℝ) : ℝ :=
if x ≤ 0 then x^2 + 1 else -3*x

theorem find_x (x: ℝ) (h: f (f x) = 10) : x = 1 :=
by
  sorry

end find_x_l275_275109


namespace relationship_abc_l275_275303

variable {ℝ}

/- Definitions from the conditions -/
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

/- Define the derivative conditions -/
def derivative1 (f f' : ℝ → ℝ) := ∀ x, f' x = (deriv f x)
def derivative2_pos (f f' f'' : ℝ → ℝ) := ∀ x, x ≠ 0 → f'' x + f x / x > 0

/- The specific values of a, b, and c -/
def a (f : ℝ → ℝ) : ℝ :=
  (1 / 2) * f (1 / 2)

def b (f : ℝ → ℝ) : ℝ :=
  -2 * f (-2)

def c (f : ℝ → ℝ) : ℝ :=
  (Real.log (1 / 2)) * f (Real.log (1 / 2))

/- Theorem statement -/
theorem relationship_abc (f f' f'' : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_deriv1 : derivative1 f f') 
  (h_deriv2_pos : derivative2_pos f f' f'') : 
  a f < c f ∧ c f < b f := 
sorry

end relationship_abc_l275_275303


namespace donate_to_charity_equals_324_l275_275388

variable {TuesdayA ThursdayA FridayA MondayB WednesdayB FridayB Market Mall Bakery Pie Neighbor Charity : ℕ}

def eggs_collected_from_A : ℕ :=
  TuesdayA + ThursdayA + FridayA

def eggs_collected_from_B : ℕ :=
  MondayB + WednesdayB + FridayB

def total_eggs_collected : ℕ :=
  eggs_collected_from_A + eggs_collected_from_B

def eggs_distributed : ℕ :=
  Market + Mall + Bakery + Pie + Neighbor

def eggs_donated_to_charity (total_collected : ℕ) (total_distributed: ℕ) : ℕ :=
  total_collected - total_distributed

theorem donate_to_charity_equals_324 :
  TuesdayA = 8 ∧ ThursdayA = 12 ∧ FridayA = 4 ∧ 
  MondayB = 6 ∧ WednesdayB = 9 ∧ FridayB = 3 ∧
  Market = 3 ∧ Mall = 5 ∧ Bakery = 2 ∧
  Pie = 4 ∧ Neighbor = 1 →
  eggs_donated_to_charity total_eggs_collected eggs_distributed = 324 := 
by 
sorrry

end donate_to_charity_equals_324_l275_275388


namespace min_value_of_exponential_sum_l275_275283

/-- Prove the minimum value of 4^a + 8^b given the condition 2a + 3b = 4. -/
theorem min_value_of_exponential_sum (a b : ℝ) (h : 2 * a + 3 * b = 4) : 
  4^a + 8^b ≥ 8 :=
begin
  sorry
end

end min_value_of_exponential_sum_l275_275283


namespace probability_difference_multiple_of_6_l275_275467

open Finset

/-- The probability that some pair of seven distinct positive integers chosen between 1 and 2010
    inclusive has a difference that is a multiple of 6 is 1. -/
theorem probability_difference_multiple_of_6 : 
  ∀ (s : Finset (Fin 2011)), s.card = 7 → (∃ (a b ∈ s), a ≠ b ∧ (a.1 - b.1) % 6 = 0) :=
by sorry

end probability_difference_multiple_of_6_l275_275467


namespace proposition_B_correct_l275_275939

theorem proposition_B_correct : ∀ x : ℝ, x > 0 → x - 1 ≥ Real.log x :=
by
  sorry

end proposition_B_correct_l275_275939


namespace system_solve_l275_275727

theorem system_solve (x y : ℚ) (h1 : 2 * x + y = 3) (h2 : 3 * x - 2 * y = 12) : x + y = 3 / 7 :=
by
  -- The proof will go here, but we skip it for now.
  sorry

end system_solve_l275_275727


namespace total_cost_is_correct_l275_275831

-- Definitions based on the provided conditions
def room_length : ℝ := 18
def room_breadth : ℝ := 7.5
def carpet_width_cm : ℝ := 75
def carpet_cost_per_meter : ℝ := 4.5

-- Conversion of carpet width
def carpet_width : ℝ := carpet_width_cm / 100

-- Area of the room
def room_area : ℝ := room_length * room_breadth

-- Number of carpet strips needed
def number_of_strips : ℝ := room_breadth / carpet_width

-- Total length of carpet needed
def total_carpet_length : ℝ := number_of_strips * room_length

-- Total cost calculation
def total_cost : ℝ := total_carpet_length * carpet_cost_per_meter

-- Proof Statement
theorem total_cost_is_correct : total_cost = 810 :=
by
  -- Here we will have proof steps
  sorry

end total_cost_is_correct_l275_275831


namespace probability_seven_chairs_probability_n_chairs_l275_275037
-- Importing necessary library to ensure our Lean code can be built successfully

-- Definition for case where n = 7
theorem probability_seven_chairs : 
  let total_seating := 7 * 6 * 5 / 6 
  let favorable_seating := 1 
  let probability := favorable_seating / total_seating 
  probability = 1 / 35 := 
by 
  sorry

-- Definition for general case where n ≥ 6
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) : 
  let total_seating := (n - 1) * (n - 2) / 2 
  let favorable_seating := (n - 4) * (n - 5) / 2 
  let probability := favorable_seating / total_seating 
  probability = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := 
by 
  sorry

end probability_seven_chairs_probability_n_chairs_l275_275037


namespace solvable_congruence_system_finite_LCM_l275_275051

namespace ProofProblem

theorem solvable_congruence_system_finite_LCM (m : ℕ → ℕ) :
  (∀ i, m i > 0) →
  (∃ x, ∀ i, x ≡ 2 * (m i)^2 [MOD (2 * (m i) - 1)]) ↔ 
  ∃ k, (∃ n, k = list.lcm (list.map (λ i, 2 * (m i) - 1) (list.finRange n))) :=
by sorry

end ProofProblem

end solvable_congruence_system_finite_LCM_l275_275051


namespace solve_system_of_equations_l275_275821

theorem solve_system_of_equations :
  ∃ x y : ℝ, 
  (4 * x - 3 * y = -0.5) ∧ 
  (5 * x + 7 * y = 10.3) ∧ 
  (|x - 0.6372| < 1e-4) ∧ 
  (|y - 1.0163| < 1e-4) :=
by
  sorry

end solve_system_of_equations_l275_275821


namespace foci_and_directrices_of_ellipse_l275_275277

noncomputable def parametricEllipse
    (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ + 1, 4 * Real.sin θ)

theorem foci_and_directrices_of_ellipse :
  (∀ θ : ℝ, parametricEllipse θ = (x, y)) →
  (∃ (f1 f2 : ℝ × ℝ) (d1 d2 : ℝ → Prop),
    f1 = (1, Real.sqrt 7) ∧
    f2 = (1, -Real.sqrt 7) ∧
    d1 = fun x => x = 1 + 9 / Real.sqrt 7 ∧
    d2 = fun x => x = 1 - 9 / Real.sqrt 7) := sorry

end foci_and_directrices_of_ellipse_l275_275277


namespace probability_of_ellipse_l275_275460

open ProbabilityTheory MeasureTheory

noncomputable def is_ellipse_with_foci_on_y_axis (m : ℝ) : Prop := 
  m^2 < 4

theorem probability_of_ellipse (h : ∀ m, m ∈ set.Icc 1 5) : 
  ℙ (λ m, is_ellipse_with_foci_on_y_axis m) = 1 / 4 :=
sorry

end probability_of_ellipse_l275_275460


namespace periodic_sequence_implies_integer_l275_275434

open Real

theorem periodic_sequence_implies_integer (α : ℝ) (hα : α > 1) (T : ℕ) (hT : 0 < T) 
  (a : ℕ → ℝ) (h_seq : ∀ n : ℕ, a n = α * floor (α ^ n) - floor (α ^ (n + 1))) 
  (h_periodic : ∀ n : ℕ, a (n + T) = a n) : α ∈ ℤ := 
sorry

end periodic_sequence_implies_integer_l275_275434


namespace four_digit_divisors_l275_275972

theorem four_digit_divisors :
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 →
  (1000 * a + 100 * b + 10 * c + d ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * d + 100 * a + 10 * b + c) →
  ∃ (e f : ℕ), e = a ∧ f = b ∧ (e ≠ 0 ∧ f ≠ 0) ∧ (1000 * e + 100 * e + 10 * f + f = 1000 * a + 100 * b + 10 * a + b) ∧
  (1000 * e + 100 * e + 10 * f + f ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * d + 100 * a + 10 * b + c) := 
by
  sorry

end four_digit_divisors_l275_275972


namespace range_of_a_l275_275687

noncomputable def p (a : ℝ) : Prop :=
∀ (x : ℝ), x > -1 → (x^2) / (x + 1) ≥ a

noncomputable def q (a : ℝ) : Prop :=
∃ (x : ℝ), (a*x^2 - a*x + 1 = 0)

theorem range_of_a (a : ℝ) :
  ¬ p a ∧ ¬ q a ∧ (p a ∨ q a) ↔ (a = 0 ∨ a ≥ 4) :=
by sorry

end range_of_a_l275_275687


namespace cuberoot_sum_l275_275240

-- Prove that the sum c + d = 60 for the simplified form of the given expression.
theorem cuberoot_sum :
  let c := 15
  let d := 45
  c + d = 60 :=
by
  sorry

end cuberoot_sum_l275_275240


namespace divisibility_property_l275_275054

noncomputable def seq (k : ℤ) : ℕ → ℤ
| 0       => 0
| 1       => k
| (n + 2) => k^2 * seq (n + 1) - seq n

theorem divisibility_property (k : ℤ) (n : ℕ) :
  (seq k (n + 1) * seq k n + 1) ∣ (seq k (n + 1))^2 + (seq k n)^2 :=
sorry

end divisibility_property_l275_275054


namespace example_theorem_l275_275709

def not_a_term : Prop := ∀ n : ℕ, ¬ (24 - 2 * n = 3)

theorem example_theorem : not_a_term :=
  by sorry

end example_theorem_l275_275709


namespace sticks_at_20_l275_275248

-- Define the sequence of sticks used at each stage
def sticks (n : ℕ) : ℕ :=
  if n = 1 then 5
  else if n ≤ 10 then 5 + 3 * (n - 1)
  else 32 + 4 * (n - 11)

-- Prove that the number of sticks at the 20th stage is 68
theorem sticks_at_20 : sticks 20 = 68 := by
  sorry

end sticks_at_20_l275_275248


namespace joy_reading_rate_l275_275409

theorem joy_reading_rate
  (h1 : ∀ t: ℕ, t = 20 → ∀ p: ℕ, p = 8 → ∀ t': ℕ, t' = 60 → ∃ p': ℕ, p' = (p * t') / t)
  (h2 : ∀ t: ℕ, t = 5 * 60 → ∀ p: ℕ, p = 120):
  ∃ r: ℕ, r = 24 :=
by
  sorry

end joy_reading_rate_l275_275409


namespace product_min_max_l275_275790

theorem product_min_max :
  (∀ x y : ℝ, 3 * x^2 + 6 * x * y + 4 * y^2 = 1 → 
  (let k := 3 * x^2 + 4 * x * y + 3 * y^2 in
  (k = (3 - Real.sqrt 10) / 4 ∨ k = (3 + Real.sqrt 10) / 4))) → 
  ((3 - Real.sqrt 10) / 4) * ((3 + Real.sqrt 10) / 4) = 1 / 16 :=
by 
  sorry

end product_min_max_l275_275790


namespace range_f_l275_275630

def odot (a b : ℝ) : ℝ :=
if a ≤ b then a else b

def f (x : ℝ) : ℝ :=
odot (2^x) (2^(-x))

theorem range_f : set.range f = set.Ioc 0 1 :=
by
sorry

end range_f_l275_275630


namespace solve_inequality_l275_275094

-- Define the conditions
def condition_inequality (x : ℝ) : Prop := abs x + abs (2 * x - 3) ≥ 6

-- Define the solution set form
def solution_set (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 3

-- State the theorem
theorem solve_inequality (x : ℝ) : condition_inequality x → solution_set x := 
by 
  sorry

end solve_inequality_l275_275094


namespace f_of_10_is_20_l275_275341

theorem f_of_10_is_20 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (3 * x + 1) = x^2 + 3 * x + 2) : f 10 = 20 :=
  sorry

end f_of_10_is_20_l275_275341


namespace maximized_area_using_squares_l275_275266

theorem maximized_area_using_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
  by sorry

end maximized_area_using_squares_l275_275266


namespace mixed_number_evaluation_l275_275644

theorem mixed_number_evaluation :
  let a := (4 + 1 / 3 : ℚ)
  let b := (3 + 2 / 7 : ℚ)
  let c := (2 + 5 / 6 : ℚ)
  let d := (1 + 1 / 2 : ℚ)
  let e := (5 + 1 / 4 : ℚ)
  let f := (3 + 2 / 5 : ℚ)
  (a + b - c) * (d + e) / f = 9 + 198 / 317 :=
by {
  let a : ℚ := 4 + 1 / 3
  let b : ℚ := 3 + 2 / 7
  let c : ℚ := 2 + 5 / 6
  let d : ℚ := 1 + 1 / 2
  let e : ℚ := 5 + 1 / 4
  let f : ℚ := 3 + 2 / 5
  sorry
}

end mixed_number_evaluation_l275_275644


namespace jose_profit_l275_275519

def tom_investment : ℕ := 30000
def jose_investment : ℕ := 45000
def maria_investment : ℕ := 60000

def tom_time : ℕ := 12
def jose_time : ℕ := 10
def maria_time : ℕ := 8

def total_profit : ℕ := 68000

-- Calculations
def tom_capital_months := tom_investment * tom_time
def jose_capital_months := jose_investment * jose_time
def maria_capital_months := maria_investment * maria_time

def total_capital_months := tom_capital_months + jose_capital_months + maria_capital_months

def jose_share := (jose_capital_months.to_rat / total_capital_months.to_rat) * total_profit.to_rat

theorem jose_profit :
  jose_share ≈ 23721.31 :=
sorry

end jose_profit_l275_275519


namespace sum_a_le_one_l275_275347

noncomputable def a : ℕ → ℚ
| 0       := 1 / 2
| (n + 1) := if n = 0 then 1 / 2 else ((2 * (n + 1) - 3) / (2 * (n + 1))) * a n

theorem sum_a_le_one (n : ℕ) : (∑ k in Finset.range (n + 1), a k) < 1 :=
sorry

end sum_a_le_one_l275_275347


namespace inclination_angle_range_l275_275737

theorem inclination_angle_range (k α : ℝ) (hk : 0 < k ∧ k < real.sqrt 3) (h : k = real.tan α) :
  0 < α ∧ α < real.pi / 3 :=
sorry

end inclination_angle_range_l275_275737


namespace solve_for_x_l275_275818

theorem solve_for_x :
  ∃ x : ℝ, (4 * x + 9 * x = 430 - 10 * (x + 4)) ∧ (x = 17) :=
begin
  use 17,
  split,
  { 
    calc 4 * 17 + 9 * 17 
        = 68 + 153 : by norm_num
    ... = 221 : by norm_num
    ... = 430 - 10 * (17 + 4) : by norm_num
    ... = 430 - 210 : by norm_num
    ... = 220 : by norm_num, 
    sorry
  },
  { refl }
end

end solve_for_x_l275_275818


namespace domain_of_function_proof_l275_275258

noncomputable def domain_of_function (x : ℝ) : Prop :=
  1 + x > 0 ∧ 4 - 2^x ≥ 0

theorem domain_of_function_proof : 
  ∀ x : ℝ, domain_of_function x ↔ (-1 < x ∧ x ≤ 2) :=
  by
  sorry

end domain_of_function_proof_l275_275258


namespace max_solutions_le_2_l275_275157

noncomputable def max_linear_eq (a : ℝ →₀ (Fin 10) → ℝ) (b : ℝ →₀ (Fin 10) → ℝ) (x : ℝ) : ℝ :=
  Real.max (Set.Union (λ i : Fin 10, ({a.to_fun.to_fn i * x + b.to_fun.to_fn i} : Set ℝ)))

theorem max_solutions_le_2 
  (a b : ℝ →₀ (Fin 10) → ℝ) 
  (h : ∀ i : Fin 10, a.to_fun.to_fn i ≠ 0) :
  ∃ (x1 x2 : ℝ), 
    x1 ≠ x2 ∧ max_linear_eq a b x1 = 0 ∧ max_linear_eq a b x2 = 0 :=
sorry

end max_solutions_le_2_l275_275157


namespace number_of_intersected_cubes_l275_275206

def large_cube := cube 3
def small_cubes_count := 27
def intersected_cubes_count (n : ℕ) : ℕ := 
  -- Function definition that models the intersection
  sorry

theorem number_of_intersected_cubes :
  intersected_cubes_count small_cubes_count = 19 :=
by 
  -- Hint to divide the problem correctly
  sorry

end number_of_intersected_cubes_l275_275206


namespace division_remainder_correct_l275_275958

theorem division_remainder_correct :
  ∃ q r, 987670 = 128 * q + r ∧ 0 ≤ r ∧ r < 128 ∧ r = 22 :=
by
  sorry

end division_remainder_correct_l275_275958


namespace star_result_l275_275249

-- Define the operation star
def star (m n p q : ℚ) := (m * p) * (n / q)

-- Given values
def a := (5 : ℚ) / 9
def b := (10 : ℚ) / 6

-- Condition to check
theorem star_result : star 5 9 10 6 = 75 := by
  sorry

end star_result_l275_275249


namespace probability_empty_chair_on_sides_7_chairs_l275_275047

theorem probability_empty_chair_on_sides_7_chairs :
  (1 : ℚ) / (35 : ℚ) = 0.2 := by
  sorry

end probability_empty_chair_on_sides_7_chairs_l275_275047


namespace can_identify_odd_ball_in_three_weighings_l275_275527

def weighing_possible (balls : Fin 12 → ℚ) (balance_scales : ∀ (w1 w2 : List ℚ), Bool) : Prop :=
  (∃ (odd_ball_index : Fin 12),
    let known_weights := filter (≠ balls odd_ball_index) (List.ofFn balls)
    (∀ (i : Fin 12), (balls i = balls odd_ball_index ∨ balls i ∈ known_weights) ∧
     (balls odd_ball_index < balls i ∨ balls odd_ball_index > balls i) ∧
     balance_scales (List.ofFn balls) (List.ofFn balls)) ∧
    (∃ (measurements : {L : List ℚ // L.length = 3} → Bool), 
      ∀ (w1 w2 w3 : List ℚ) (h : List.all (w1 ++ w2 ++ w3) (λ b, b ∈ List.ofFn balls)),
        measurements ⟨w1, by sorry⟩ ∧ measurements ⟨w2, by sorry⟩ ∧
        measurements ⟨w3, by sorry⟩)) 

theorem can_identify_odd_ball_in_three_weighings (balls : Fin 12 → ℚ) (balance_scales : ∀ (w1 w2 : List ℚ), Bool) :
  ∃ (measurements : {L : List ℚ // L.length = 3} → Bool),
    weighing_possible balls balance_scales :=
sorry

end can_identify_odd_ball_in_three_weighings_l275_275527


namespace find_z_l275_275700

def complex_number_z (z : ℂ) : Prop :=
  z * (1 - complex.i) = 1 + complex.i

theorem find_z (z : ℂ) (h : complex_number_z z) : z = complex.i :=
by sorry

end find_z_l275_275700


namespace twentieth_prime_is_71_l275_275088

/-- 
    Primality definition (for completeness, even though Lean already has a built-in prime definition). 
    This can be skipped if using Lean's built-in prime definition.
-/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Seventh prime number definition
def seventh_prime := 17

-- Verify that the twentieth prime number is indeed 71
theorem twentieth_prime_is_71 : seventh_prime = 17 → Nat.prime (fin 20) = 71 :=
by
  intro h
  sorry

end twentieth_prime_is_71_l275_275088


namespace original_average_l275_275405

theorem original_average (A : ℝ)
  (h1 : (Σ i in (finset.range 15), A) / 15 = A)
  (h2 : (Σ i in (finset.range 15), (A + 10)) / 15 = 50) :
  A = 40 :=
sorry

end original_average_l275_275405


namespace find_number_l275_275502

theorem find_number (x : ℝ) : (10 * x = 2 * x - 36) → (x = -4.5) :=
begin
  intro h,
  sorry
end

end find_number_l275_275502


namespace option_e_is_perfect_square_l275_275878

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem option_e_is_perfect_square :
  is_perfect_square (4^10 * 5^5 * 6^10) :=
sorry

end option_e_is_perfect_square_l275_275878


namespace probability_case_7_probability_case_n_l275_275032

-- Define the scenario for n = 7
def probability_empty_chair_7 : Prop :=
  ∃ p : ℚ, p = 1 / 35

-- Theorem for the case when there are exactly 7 chairs
theorem probability_case_7 : probability_empty_chair_7 :=
begin
  -- Proof skipped
  sorry
end

-- Define the general scenario for n chairs (n >= 6)
def probability_empty_chair_n (n : ℕ) (h : n ≥ 6) : Prop :=
  ∃ p : ℚ, p = (n - 4) * (n - 5) / ((n - 1) * (n - 2))

-- Theorem for the case when there are n chairs (n ≥ 6)
theorem probability_case_n (n : ℕ) (h : n ≥ 6) : probability_empty_chair_n n h :=
begin
  -- Proof skipped
  sorry
end

end probability_case_7_probability_case_n_l275_275032


namespace expected_value_X_l275_275669

open Probability

-- Define the conditions of the problem
def bag : Set ℕ := {1, 2, 3}  -- Representing red = 1, yellow = 2, blue = 3
def equal_prob : Measure ℕ := Measure.ofFinset bag (λ _, 1/3)

-- Define the event of drawing two consecutive red balls
def twoConseReds (draws : List ℕ) : Prop :=
  ∃ i, i < draws.length - 1 ∧ draws.nth i = some 1 ∧ draws.nth (i + 1) = some 1

-- Define the random variable X as the number of draws needed to achieve two consecutive red balls
noncomputable def X : Measure ℕ := Measure.map length {draws | twoConseReds draws}

-- Define the expected value of X
noncomputable def E_X : ℝ := ∫⁻ x, x.toReal ∂X

-- The proof statement
theorem expected_value_X : E_X = 12 :=
  sorry

end expected_value_X_l275_275669


namespace system_solution_l275_275095

theorem system_solution :
  ∃ x y : ℝ, (16 * x^2 + 8 * x * y + 4 * y^2 + 20 * x + 2 * y = -7) ∧ 
            (8 * x^2 - 16 * x * y + 2 * y^2 + 20 * x - 14 * y = -11) ∧
            x = -3 / 4 ∧ y = 1 / 2 :=
by
  sorry

end system_solution_l275_275095


namespace solve_inequality_l275_275887

theorem solve_inequality (x : ℝ) (h : 9.244 * sqrt(1 - 9 * (log x / log (1/8))^2) > 1 - 4 * (log x / log (1/8))) : 
  (1/2 : ℝ) ≤ x ∧ x < 1 :=
by
  sorry

end solve_inequality_l275_275887


namespace arithmetic_sequence_general_term_geometric_sequence_inequality_l275_275296

-- Sequence {a_n} and its sum S_n
def S (a : ℕ → ℤ) (n : ℕ) : ℤ := (Finset.range n).sum a

-- Sequence {b_n}
def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  2 * (S a (n + 1) - S a n) * (S a n) - n * (S a (n + 1) + S a n)

-- Arithmetic sequence and related conditions
theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a n = a 0 + n * d) (h2 : ∀ n, b a n = 0) :
  (∀ n, a n = 0) ∨ (∀ n, a n = n) :=
sorry

-- Conditions for sequences and finding the set of positive integers n
theorem geometric_sequence_inequality (a : ℕ → ℤ)
  (h1 : a 1 = 1) (h2 : a 2 = 3)
  (h3 : ∀ n, a (2 * n - 1) = 2^(n-1))
  (h4 : ∀ n, a (2 * n) = 3 * 2^(n-1)) :
  {n : ℕ | b a (2 * n) < b a (2 * n - 1)} = {1, 2, 3, 4, 5, 6} :=
sorry

end arithmetic_sequence_general_term_geometric_sequence_inequality_l275_275296


namespace monotonic_intervals_and_symmetry_center_l275_275015

noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def g (x : ℝ) : ℝ := 1 / (x + 3)
noncomputable def h (x : ℝ) : ℝ := (x + 4) / (x + 3)

theorem monotonic_intervals_and_symmetry_center :
  (∀ x, x ∈ Ioo (-∞) (-3) ∨ x ∈ Ioo (-3) (+∞) → h x < h (-∞) ∨ h x < h (+∞)) ∧
  (∃ c : ℝ, c = (-3, 1) ∧ ∀ x, h (2 * c - x) = h x) := 
sorry

end monotonic_intervals_and_symmetry_center_l275_275015


namespace cubic_sum_identity_l275_275371

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 10) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 100 :=
by sorry

end cubic_sum_identity_l275_275371


namespace find_a_l275_275703

-- Define the function f given a parameter a
def f (x a : ℝ) : ℝ := x^3 - 3*x^2 + a

-- Condition: f(x+1) is an odd function
theorem find_a (a : ℝ) (h : ∀ x : ℝ, f (-(x+1)) a = -f (x+1) a) : a = 2 := 
sorry

end find_a_l275_275703


namespace check_point_transformation_l275_275586

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x ^ 2 + y ^ 2)
  let θ := real.atan (y / x)
  (r, θ)

noncomputable def polar_to_rectangular (r θ: ℝ) : ℝ × ℝ :=
  (r * real.cos θ, r * real.sin θ)

noncomputable def point_transformation (x y : ℝ) : ℝ × ℝ :=
  let (r, θ) := rectangular_to_polar x y
  let new_r := r^3
  let new_θ := 3 * θ
  polar_to_rectangular new_r new_θ

theorem check_point_transformation :
  point_transformation 8 6 = (70.7, 997.6) :=
by
  rw point_transformation
  sorry

end check_point_transformation_l275_275586


namespace completing_the_square_correct_l275_275537

theorem completing_the_square_correct :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x h
  sorry

end completing_the_square_correct_l275_275537


namespace problem1_problem2_l275_275612

theorem problem1 : (π - 3.14)^0 + abs (-2) - real.cbrt 8 = 1 := by
  sorry

theorem problem2 : (-3 : ℝ)^2024 * (1 / 3)^2023 = 3 := by
  sorry

end problem1_problem2_l275_275612


namespace incenter_midpoint_of_DE_l275_275130

variable {A B C I D E : Point}
variable {CA CB : Line}

-- Define the triangle and incenter
variable (triangle_ABC : Triangle A B C)
variable (I_incenter : IsIncenter I triangle_ABC)

-- Define a circle inside the circumcircle that touches CA at D and BC at E
variable (inner_circle : Circle)
variable (inner_circle_condition : InnerCircleTouches inner_circle (Circumcircle triangle_ABC) CA D BC E)

-- Prove that I is the midpoint of DE
theorem incenter_midpoint_of_DE (triangle_ABC : Triangle A B C) (I_incenter : IsIncenter I triangle_ABC)
    (inner_circle : Circle) (inner_circle_condition : InnerCircleTouches inner_circle (Circumcircle triangle_ABC) CA D BC E) : 
    IsMidpoint I D E :=
by
  sorry

end incenter_midpoint_of_DE_l275_275130


namespace bell_rings_together_at_04_00_PM_l275_275930

noncomputable def LCM_18_24_30 : ℕ := Nat.lcm (Nat.lcm 18 24) 30

theorem bell_rings_together_at_04_00_PM :
  LCM_18_24_30 = 360 ∧ ((Time.of_nat 10 * 60 + 360).to_24_hour_format = (4, 0)) :=
by
  -- insert the formal proof here
  sorry

end bell_rings_together_at_04_00_PM_l275_275930


namespace probability_more_than_three_l275_275368

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_more_than_three (n m : ℕ) :
  n = 12 → binom 12 6 = 924 →
  let total_outcomes := 2^12 in
  let prob_equal := (binom 12 6) / total_outcomes.toRat in
  let desired_prob := ((1 : ℚ) - prob_equal) / 2 in
  desired_prob = 793 / 2048 := 
by
  intros h1 h2,
  sorry

end probability_more_than_three_l275_275368


namespace four_digit_divisible_by_40_form_ab20_l275_275359

theorem four_digit_divisible_by_40_form_ab20 :
  ∃ n : ℕ, n = 12 ∧ ∀ (ab : ℕ), ab20 = ab * 100 + 20 →
  (divisible (ab * 100 + 20) 40 ↔ (∃ k, ab = 8 * k ∧ 1 ≤ k ∧ k ≤ 12)) :=
sorry

end four_digit_divisible_by_40_form_ab20_l275_275359


namespace sqrt_three_irrational_l275_275544

theorem sqrt_three_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a:ℝ) / b = Real.sqrt 3 :=
sorry

end sqrt_three_irrational_l275_275544


namespace other_divisor_is_four_l275_275652

-- Definitions from conditions
def smallest_number_satisfying_conditions : ℕ := 1010
def k : ℕ := 2
def check_divisibility (n : ℕ) : Prop := n % 12 = 0 ∧ n % 18 = 0 ∧ n % 21 = 0 ∧ n % 28 = 0
def lcm_12_18_21_28 : ℕ := Nat.lcm (Nat.lcm 12 18) (Nat.lcm 21 28)

-- The main proof statement
theorem other_divisor_is_four : 
  ∃ (other_divisor : ℕ), 
  let smallest_num := smallest_number_satisfying_conditions - k in 
  check_divisibility smallest_number_satisfying_conditions ∧ 
  smallest_num % lcm_12_18_21_28 = 0 ∧ 
  smallest_num % other_divisor = 0 ∧ 
  ¬ other_divisor ∣ lcm_12_18_21_28 ∧ 
  other_divisor = 4 :=
by
  sorry

end other_divisor_is_four_l275_275652


namespace sum_of_possible_k_l275_275874

theorem sum_of_possible_k : 
  (∀ (k : ℝ), (∃ x : ℝ, x^2 - 4*x + 3 = 0 ∧ x^2 - 5*x + k = 0) → 
    k ∈ {4, 6}) → 
  ∑ k in ({4, 6} : finset ℝ), k = 10 := by
sorry

end sum_of_possible_k_l275_275874


namespace find_m_l275_275735

variable (m : ℝ)

def f (x : ℝ) : ℝ := x^2 - 2 * x + m

theorem find_m : (∀ x ∈ Icc (2 : ℝ) ⊤, f x ≥ -3) ∧ f 2 = -3 → m = -3 := 
by
  sorry

end find_m_l275_275735


namespace mason_water_intake_l275_275133

theorem mason_water_intake
  (Theo_Daily : ℕ := 8)
  (Roxy_Daily : ℕ := 9)
  (Total_Weekly : ℕ := 168)
  (Days_Per_Week : ℕ := 7) :
  (∃ M : ℕ, M * Days_Per_Week = Total_Weekly - (Theo_Daily + Roxy_Daily) * Days_Per_Week ∧ M = 7) :=
  by
  sorry

end mason_water_intake_l275_275133


namespace least_positive_integer_l275_275871

open Nat

theorem least_positive_integer (n : ℕ) (h1 : n ≡ 2 [MOD 5]) (h2 : n ≡ 2 [MOD 4]) (h3 : n ≡ 0 [MOD 3]) : n = 42 :=
sorry

end least_positive_integer_l275_275871


namespace min_socks_to_guarantee_15_pairs_l275_275927

-- Defining the conditions
def socks := {white : ℕ // white = 150} ∧ 
             {red : ℕ // red = 120} ∧ 
             {blue : ℕ // blue = 90} ∧ 
             {green : ℕ // green = 60} ∧ 
             {yellow : ℕ // yellow = 30}

-- The theorem statement
theorem min_socks_to_guarantee_15_pairs (s : socks) : 
  ∃ n, n >= 146 ∧ ∀ sock_selection, 
    (sock_selection ≤ n) → 
    (∃ color, sock_selection / 2 ≥ 15) :=
sorry

end min_socks_to_guarantee_15_pairs_l275_275927


namespace largest_three_digit_number_l275_275648

def is_digit (n : ℕ) : Prop := n ≤ 9

noncomputable def largest_perfect_square_diff : ℕ :=
  let n := 100 * 9 + 10 * 1 + 9
  n

theorem largest_three_digit_number :
  ∃ (a b c : ℕ), is_digit a ∧ is_digit b ∧ is_digit c ∧ 0 < a ∧
                 (largest_perfect_square_diff = 100 * a + 10 * b + c) ∧
                 (100 * a + 10 * b + c) - (a + b + c) = 30^2 :=
by {
  existsi 9, existsi 1, existsi 9,
  split, exact le_refl 9,
  split, exact le_refl 9,
  split, exact le_refl 9,
  split, exact zero_lt_succ 8,
  split,
  {
    rw largest_perfect_square_diff,
    refl,
  },
  {
    sorry -- Verification here
  }
}

end largest_three_digit_number_l275_275648


namespace inclusion_relationships_l275_275711

-- Definitions from the conditions
def SetA : Set (ℝ × ℝ) := {p | |p.1| + |p.2| ≤ 1}

def SetB : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}

def SetC : Set (ℝ × ℝ) := {p | |p.1| ≤ 1 ∧ |p.2| ≤ 1}

-- Statement of the problem
theorem inclusion_relationships :
  SetA ⊆ SetB ∧ SetB ⊆ SetC ∧ (∃ x, x ∈ SetB ∧ x ∉ SetA) ∧ (∃ x, x ∈ SetC ∧ x ∉ SetB) :=
begin
  sorry
end

end inclusion_relationships_l275_275711


namespace length_of_garden_side_l275_275859

theorem length_of_garden_side (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 112) (h2 : perimeter = 4 * side_length) : 
  side_length = 28 :=
by
  sorry

end length_of_garden_side_l275_275859


namespace equilateral_triangle_if_equal_altitude_ratios_l275_275081

open Classical

variables {A B C A1 B1 C1 H : Type}
variables [GeometryType A B C Old A1 B1 C1 H]

theorem equilateral_triangle_if_equal_altitude_ratios
  (h_AA1 : altitude A A1)
  (h_BB1 : altitude B B1)
  (h_CC1 : altitude C C1)
  (h_orthocenter : orthocenter H)
  (h_ratios : ∃ k : ℝ, (AH / A1H) = k ∧ (BH / B1H) = k ∧ (CH / C1H) = k) :
  equilateral_triangle A B C := sorry

end equilateral_triangle_if_equal_altitude_ratios_l275_275081


namespace max_ab_upper_bound_l275_275670

noncomputable def circle_center_coords : ℝ × ℝ :=
  let center_x := -1
  let center_y := 2
  (center_x, center_y)

noncomputable def max_ab_value (a b : ℝ) : ℝ :=
  if a = 1 - 2 * b then a * b else 0

theorem max_ab_upper_bound :
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 + 1 = 0}
  let line_cond : ℝ × ℝ := (-1, 2)
  (circle_center_coords = line_cond) →
  (∀ a b : ℝ, max_ab_value a b ≤ 1 / 8) :=
by
  intro circle line_cond h
  -- Proof is omitted as per instruction
  sorry

end max_ab_upper_bound_l275_275670


namespace median_of_temperatures_is_29_l275_275383

theorem median_of_temperatures_is_29 : 
  let temps := [30, 33, 24, 29, 24] 
  by
    let sorted_temps := List.mergeSort compare temps 
    List.get! sorted_temps 2 = 29 :=
begin
  let temps := [30, 33, 24, 29, 24],
  let sorted_temps := List.mergeSort compare temps,
  exact List.get! sorted_temps 2 = 29,
end

end median_of_temperatures_is_29_l275_275383


namespace problem1_problem2_l275_275317

noncomputable theory

-- Given functions
def f (x : ℝ) := x^3 - x
def g (x : ℝ) (a : ℝ) := x^2 + a

-- Problem (1): If x1 = -1, show that a = 3
theorem problem1 (a : ℝ) (h : tangent_line_at f (-1) = tangent_line_at (g (-1)) (f (-1))) :
  a = 3 := sorry

-- Lean theorem to find the range of values for a such that tangent line conditions hold
theorem problem2 (x₁ : ℝ) (a : ℝ) (h : tangent_line_at f x₁ = tangent_line_at g x₁ a) :
  a ≥ -1 := sorry

end problem1_problem2_l275_275317


namespace log_equal_implies_frac_one_l275_275472

theorem log_equal_implies_frac_one
  (p q : ℝ)
  (h1 : log 8 p = log 20 (p^2 + q))
  (h2 : log 10 q = log 20 (p^2 + q))
  (hp : p > 0)
  (hq : q > 0)
  : p^2 / q = 1 := by {
    sorry
  }

end log_equal_implies_frac_one_l275_275472


namespace find_AC_l275_275190

theorem find_AC (A B C : ℝ) (r1 r2 : ℝ) (AB : ℝ) (AC : ℝ) 
  (h_rad1 : r1 = 1) (h_rad2 : r2 = 3) (h_AB : AB = 2 * Real.sqrt 5) 
  (h_AC : AC = AB / 4) :
  AC = Real.sqrt 5 / 2 :=
by
  sorry

end find_AC_l275_275190


namespace curve_intersects_midpoint_l275_275793

noncomputable def midpoint (z0 z2: ℂ) := (z0 + z2) / 2

def curve (a b c : ℝ) (t : ℝ) : ℂ :=
  (a * complex.I) * (real.cos t) ^ 4 +
  ((1/2 : ℝ) + b * complex.I) * 2 * (real.cos t) ^ 2 * (real.sin t) ^ 2 +
  (1 + c * complex.I) * (real.sin t) ^ 4

theorem curve_intersects_midpoint (a b c : ℝ) (z1 : ℂ) :
  ∃ t : ℝ, curve a b c t = midpoint (a * complex.I) (1 + c * complex.I) :=
sorry

end curve_intersects_midpoint_l275_275793


namespace find_number_l275_275501

theorem find_number (x : ℝ) : (10 * x = 2 * x - 36) → (x = -4.5) :=
begin
  intro h,
  sorry
end

end find_number_l275_275501


namespace intersection_complement_l275_275071

open Set

noncomputable def U := ℝ
noncomputable def A : Set ℝ := { y | ∃ x ∈ U, y = 2^x ∧ x < 1 }
noncomputable def B : Set ℝ := { x | ∃ y ∈ U, ln (x - 1) = y }
noncomputable def CU_B : Set ℝ := { x | x ≤ 1 }

theorem intersection_complement :
  A ∩ CU_B = { y : ℝ | y ≤ 1 } :=
by
  sorry

end intersection_complement_l275_275071


namespace radical_product_l275_275154

def fourth_root (x : ℝ) : ℝ := x ^ (1/4)
def third_root (x : ℝ) : ℝ := x ^ (1/3)
def square_root (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_product :
  fourth_root 81 * third_root 27 * square_root 9 = 27 := 
by
  sorry

end radical_product_l275_275154


namespace difference_of_extreme_valid_numbers_l275_275254

theorem difference_of_extreme_valid_numbers :
  ∃ (largest smallest : ℕ),
    (largest = 222210 ∧ smallest = 100002) ∧ 
    (largest % 3 = 0 ∧ smallest % 3 = 0) ∧ 
    (largest ≥ 100000 ∧ largest < 1000000) ∧
    (smallest ≥ 100000 ∧ smallest < 1000000) ∧
    (∀ d, d ∈ [0, 1, 2] → (d ∈ [largest / 100000 % 10, largest / 10000 % 10, largest / 1000 % 10, largest / 100 % 10, largest / 10 % 10, largest % 10])) ∧
    (∀ d, d ∈ [0, 1, 2] → (d ∈ [smallest / 100000 % 10, smallest / 10000 % 10, smallest / 1000 % 10, smallest / 100 % 10, smallest / 10 % 10, smallest % 10])) ∧ 
    (∀ d ∈ [largest / 100000 % 10, largest / 10000 % 10, largest / 1000 % 10, largest / 100 % 10, largest / 10 % 10, largest % 10], d ∈ [0, 1, 2]) ∧
    (∀ d ∈ [smallest / 100000 % 10, smallest / 10000 % 10, smallest / 1000 % 10, smallest / 100 % 10, smallest / 10 % 10, smallest % 10], d ∈ [0, 1, 2]) ∧
    (largest - smallest = 122208) :=
by
  sorry

end difference_of_extreme_valid_numbers_l275_275254


namespace find_a_plus_h_l275_275107

-- Definitions of the asymptotes and given conditions
def asymptote1 (x : ℝ) : ℝ := 3 * x + 2
def asymptote2 (x : ℝ) : ℝ := -3 * x + 8
def hyperbola_point : ℝ × ℝ := (2, 10)

-- Proving the value of a + h
theorem find_a_plus_h (a b h k : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (asymptotes : ∀ x, asymptote1 x = 3 * x + 2 ∧ asymptote2 x = -3 * x + 8)
  (passes_through : ∃ h k : ℝ, ∀ x y, (x, y) = hyperbola_point → 
    ( (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1 ))
  (std_form : ∀ x y, (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1 := std_form a b h k ) :
  a + h = 6 * Real.sqrt 2 + 1 := 
sorry

end find_a_plus_h_l275_275107


namespace count_negative_fractions_l275_275594

def is_negative_fraction (x : ℚ) : Prop :=
  x < 0

def list_of_numbers : list ℚ := [15, -3/8, 3/20, -30, -128/10, -22/7]

theorem count_negative_fractions : 
  (list_of_numbers.filter is_negative_fraction).length = 3 := 
by sorry

end count_negative_fractions_l275_275594


namespace repeated_root_and_m_value_l275_275989

theorem repeated_root_and_m_value :
  (∃ x m : ℝ, (x = 2 ∨ x = -2) ∧ 
              (m / (x ^ 2 - 4) + 2 / (x + 2) = 1 / (x - 2)) ∧ 
              (m = 4 ∨ m = 8)) :=
sorry

end repeated_root_and_m_value_l275_275989


namespace ratio_BD_FM_eq_two_l275_275427

variable {A B C D F M : Type} [PointType A] [PointType B] [PointType C] [PointType D] [PointType F] [PointType M]

-- Hypotheses
variable (triangle_ABC : Triangle A B C)
variable (D_on_AB : D ∈ seg A B)
variable (F_intersection_CD_AM : Collinear C D F ∧ Collinear A M F ∧ Midpoint M B C)
variable (AF_eq_AD : dist A F = dist A D)
       
-- \( \frac{BD}{FM} = 2 \)
theorem ratio_BD_FM_eq_two
  (h1 : ∃ D ∈ seg A B, true)
  (h2 : ∃ F, F_intersection_CD_AM triangle_ABC D F)
  (h3 : dist A F = dist A D) :
  dist B D / dist F M = 2 := sorry

end ratio_BD_FM_eq_two_l275_275427


namespace profit_conditions_l275_275121

/-- Define the total profit function y in terms of years of operation x -/
def profit (x : ℕ) : ℝ := -10 * (x : ℝ)^2 + 120 * (x : ℝ) - 250

/-- Prove that the profit function satisfies the given conditions and questions -/
theorem profit_conditions : 
  (profit 3 = 20) ∧
  (profit 6 = 110) ∧
  (∀ y, is_max (λ x, profit x / x) y = 20) :=
by
  sorry

end profit_conditions_l275_275121


namespace append_digit_to_10_yields_4104_l275_275600

theorem append_digit_to_10_yields_4104 :
  ∃ a : ℕ, (1001 * a + 110) % 12 = 0 ∧ a < 10 ∧ 1001 * a + 110 = 4104 :=
by
  existsi (4 : ℕ)
  -- Verifying the conditions
  have h1 : (1001 * 4 + 110) % 12 = 0 := by norm_num
  have h2 : 4 < 10 := by norm_num
  have h3 : 1001 * 4 + 110 = 4104 := by norm_num
  exact ⟨h1, h2, h3⟩

end append_digit_to_10_yields_4104_l275_275600


namespace trig_eq_solution_set_l275_275468

-- Theorem statement that the given equation implies the solutions set
theorem trig_eq_solution_set (x : ℝ) (n k : ℤ) :
  cos (9 * x) - cos (5 * x) - real.sqrt 2 * cos (4 * x) + sin (9 * x) + sin (5 * x) = 0 →
  (∃ n : ℤ, x = n * real.pi / 7) ∨ (∃ k : ℤ, x = real.pi / 4 * (2 * k + 1)) :=
sorry

end trig_eq_solution_set_l275_275468


namespace correct_calculation_l275_275541

theorem correct_calculation : (- (1 / 2 : ℝ)) ^ (-2) = 4 := 
by 
-- We'll fill this gap with the statement and reason proving that the calculation is correct
sorry

end correct_calculation_l275_275541


namespace num_possibilities_l275_275195

def last_digit_divisible_by_4 (n : Nat) : Prop := (60 + n) % 4 = 0

theorem num_possibilities : {n : Nat | n < 10 ∧ last_digit_divisible_by_4 n}.card = 3 := by
  sorry

end num_possibilities_l275_275195


namespace square_perimeter_is_44_8_l275_275213

noncomputable def perimeter_of_congruent_rectangles_division (s : ℝ) (P : ℝ) : ℝ :=
  let rectangle_perimeter := 2 * (s + s / 4)
  if rectangle_perimeter = P then 4 * s else 0

theorem square_perimeter_is_44_8 :
  ∀ (s : ℝ) (P : ℝ), P = 28 → 4 * s = 44.8 → perimeter_of_congruent_rectangles_division s P = 44.8 :=
by intros s P h1 h2
   sorry

end square_perimeter_is_44_8_l275_275213


namespace trapezoid_diagonals_l275_275387

-- Definitions based on conditions
variables (d_1 d_2 midline: ℝ)

-- Given conditions
axiom H1 : d_1 = 12
axiom H2 : midline = 6.5
axiom H3 : (d_1 * d_2)/2 = (d_1^2)/2 + (midline^2)

theorem trapezoid_diagonals : d_2 = 5 :=
by 
  have H : d_1 * d_1 + d_2 * d_2 = (sqrt ((d_1 * d_1) + (d_2 * d_2)-h1^(2))^2 := 
       sorry
  sorry

end trapezoid_diagonals_l275_275387


namespace inscribed_circle_theta_l275_275572

/-- Given that a circle inscribed in triangle ABC is tangent to sides BC, CA, and AB at points
    where the tangential angles are 120 degrees, 130 degrees, and theta degrees respectively,
    we need to prove that theta is 110 degrees. -/
theorem inscribed_circle_theta 
  (ABC : Type)
  (A B C : ABC)
  (theta : ℝ)
  (tangent_angle_BC : ℝ)
  (tangent_angle_CA : ℝ) 
  (tangent_angle_AB : ℝ) 
  (h1 : tangent_angle_BC = 120)
  (h2 : tangent_angle_CA = 130) 
  (h3 : tangent_angle_AB = theta) : 
  theta = 110 :=
by
  sorry

end inscribed_circle_theta_l275_275572


namespace megatech_budget_allocation_l275_275907

theorem megatech_budget_allocation :
  let microphotonics := 14
  let food_additives := 10
  let gmo := 24
  let industrial_lubricants := 8
  let basic_astrophysics := 25
  microphotonics + food_additives + gmo + industrial_lubricants + basic_astrophysics = 81 →
  100 - 81 = 19 :=
by
  intros
  -- We are given the sums already, so directly calculate the remaining percentage.
  sorry

end megatech_budget_allocation_l275_275907


namespace students_in_PE_class_l275_275017

/-- Definition of the number of cupcakes Jessa needs to make -/
noncomputable def total_cupcakes : ℕ := 140

/-- Definition of the number of fourth-grade classes -/
noncomputable def fourth_grade_classes : ℕ := 3

/-- Definition of the number of students per fourth-grade class -/
noncomputable def students_per_class : ℕ := 30

/-- Definition of the number of cupcakes Jessa needs to make for fourth-grade classes -/
noncomputable def cupcakes_for_fourth_grade : ℕ := fourth_grade_classes * students_per_class

/-- Prove the number of students in the P.E. class -/
theorem students_in_PE_class : ℕ :=
  total_cupcakes - cupcakes_for_fourth_grade

#eval students_in_PE_class -- Expected output: 50

end students_in_PE_class_l275_275017


namespace area_of_union_of_transformed_triangles_l275_275766

-- Define the Triangle and its Medians
structure Triangle :=
  (A B C : Point)  -- defining the points A, B, and C

-- Define Point_G as the intersection of medians of the triangle ABC
def Point_G (T : Triangle) : Point := sorry  -- assuming it exists for now

-- Definition of rotation about a point by 360 degrees
def rotate_360 (G : Point) (P : Point) : Point := P

-- Define the function to calculate area of the Triangle
def area (T : Triangle) : ℝ := 
  0.5 *  real.sqrt ((15 + 20 + 25) * ( - 15 + 16 ) * (15 - 16) * (20 - 16)) 

-- Lean Statement Equivalent to Proof Problem
theorem area_of_union_of_transformed_triangles (T : Triangle)
  (h1 : T.AB = 15)
  (h2 : T.BC = 20)
  (h3 : T.AC = 25)
  (G := Point_G T)
  (A' := rotate_360 G T.A)
  (B' := rotate_360 G T.B)
  (C' := rotate_360 G T.C) :
  area T = 150 :=
sorry

end area_of_union_of_transformed_triangles_l275_275766


namespace sum_of_angles_of_roots_l275_275851

theorem sum_of_angles_of_roots (φ : ℕ → ℝ) 
  (h : ∀ k : ℕ, k < 5 → φ k = 240 / 5 + (360 * k) / 5) :
  ∑ k in finset.range 5, φ k = 1320 :=
by
  sorry

end sum_of_angles_of_roots_l275_275851


namespace total_weight_of_ripe_apples_is_1200_l275_275641

def total_apples : Nat := 14
def weight_ripe_apple : Nat := 150
def weight_unripe_apple : Nat := 120
def unripe_apples : Nat := 6
def ripe_apples : Nat := total_apples - unripe_apples
def total_weight_ripe_apples : Nat := ripe_apples * weight_ripe_apple

theorem total_weight_of_ripe_apples_is_1200 :
  total_weight_ripe_apples = 1200 := by
  sorry

end total_weight_of_ripe_apples_is_1200_l275_275641


namespace hexagon_area_eq_240_l275_275129

theorem hexagon_area_eq_240
  (DEF : Triangle)
  (perimeter_DEF :  DEF.perimeter = 48)
  (circumcircle_radius_DEF :  DEF.circumcircle.radius = 10)
  (DE'F'D'E'F : Hexagon)
  (formed_by_bisectors : DE'F'D'E'F.is_formed_by_angle_bisectors_of DEF) :
  DE'F'D'E'F.area = 240 :=
by sorry

end hexagon_area_eq_240_l275_275129


namespace probability_seating_7_probability_seating_n_l275_275038

-- Definitions and theorem for case (a): n = 7
def num_ways_to_seat_7 (n : ℕ) (k : ℕ) : ℕ := n * (n - 1) * (n - 2) / k!

def valid_arrangements_7 : ℕ := 1

theorem probability_seating_7 : 
  let total_ways := num_ways_to_seat_7 7 3 in
  let valid_ways := valid_arrangements_7 in
  total_ways = 35 ∧ (valid_ways : ℚ) / total_ways = 0.2 := 
by
  sorry

-- Definitions and theorem for case (b): general n
def choose (n k : ℕ) : ℕ := n * (n - 1) / 2

def valid_arrangements_n (n : ℕ) : ℚ := (n - 4) * (n - 5) / 2

theorem probability_seating_n (n : ℕ) (hn : n ≥ 6) : 
  let total_ways := choose (n - 1) 2 in
  let valid_ways := valid_arrangements_n n in
  total_ways = (n - 1) * (n - 2) / 2 ∧ 
  (valid_ways : ℚ) / total_ways = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by
  sorry

end probability_seating_7_probability_seating_n_l275_275038


namespace Q_equals_sum_binom_l275_275662

-- Define Q(n, k) as the coefficient of x^k in (1 + x + x^2 + x^3)^n
def Q (n k : ℕ) : ℕ :=
  (finset.range (k + 1)).sum (λ j, nat.choose n j * nat.choose n (k - 2 * j))

-- State the theorem with the conditions given
theorem Q_equals_sum_binom (n k : ℕ) : 
  Q n k = (finset.range (k + 1)).sum (λ j, nat.choose n j * nat.choose n (k - 2 * j)) :=
sorry

end Q_equals_sum_binom_l275_275662


namespace common_tangents_between_circles_l275_275981

noncomputable theory

-- Define the first circle and its properties.
def circle1 := { p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4 }

-- Define the second circle and its properties.
def circle2 := { p : ℝ × ℝ | (p.1 + 2)^2 + p.2^2 = 1 }

-- Define the distance between the centers of the circles.
def distance_between_centers := real.sqrt ((2 + 2)^2 + (0 - 0)^2)

-- Define the sum of the radii.
def sum_of_radii := 2 + 1

-- Prove the number of common tangent lines.
theorem common_tangents_between_circles : 
  ∀ (C1 C2 : set (ℝ × ℝ)), 
    C1 = circle1 ∧ C2 = circle2 → distance_between_centers > sum_of_radii → 
    ∃ tangents : ℕ, tangents = 4 :=
by
  sorry

end common_tangents_between_circles_l275_275981


namespace imaginary_roots_if_and_only_if_l275_275850

-- Definition of the problem conditions
def quadratic_eq (λ : ℝ) (x : ℂ) : ℂ := (1 : ℂ) - (complex.i : ℂ) * x^2 + (λ + complex.i) * x + (1 + complex.i * λ)

-- The theorem statement: The quadratic equation has two imaginary roots if and only if λ ≠ 2
theorem imaginary_roots_if_and_only_if (λ : ℝ) : 
  (∀ x : ℂ, quadratic_eq λ x = 0 → ∀ x : ℝ, false) ↔ λ ≠ 2 :=
by
  sorry

end imaginary_roots_if_and_only_if_l275_275850


namespace solution_set_l275_275691

-- Condition definitions
variables {f : ℝ → ℝ}

-- The conditions in English
axiom domain_of_f : ∀ x : ℝ, f x ∈ ℝ
axiom point_on_graph : f 0 = 2
axiom secant_slope : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 1

-- The main theorem: (question + correct answer)
theorem solution_set :
  { x : ℝ | f (Real.log (Real.exp x - 2)) < 2 + Real.log (Real.exp x - 2) } = set.Ioo (Real.log 2) (Real.log 3) :=
sorry

end solution_set_l275_275691


namespace probability_seating_7_probability_seating_n_l275_275042

-- Definitions and theorem for case (a): n = 7
def num_ways_to_seat_7 (n : ℕ) (k : ℕ) : ℕ := n * (n - 1) * (n - 2) / k!

def valid_arrangements_7 : ℕ := 1

theorem probability_seating_7 : 
  let total_ways := num_ways_to_seat_7 7 3 in
  let valid_ways := valid_arrangements_7 in
  total_ways = 35 ∧ (valid_ways : ℚ) / total_ways = 0.2 := 
by
  sorry

-- Definitions and theorem for case (b): general n
def choose (n k : ℕ) : ℕ := n * (n - 1) / 2

def valid_arrangements_n (n : ℕ) : ℚ := (n - 4) * (n - 5) / 2

theorem probability_seating_n (n : ℕ) (hn : n ≥ 6) : 
  let total_ways := choose (n - 1) 2 in
  let valid_ways := valid_arrangements_n n in
  total_ways = (n - 1) * (n - 2) / 2 ∧ 
  (valid_ways : ℚ) / total_ways = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by
  sorry

end probability_seating_7_probability_seating_n_l275_275042


namespace prod_one_minus_nonneg_reals_ge_half_l275_275509

theorem prod_one_minus_nonneg_reals_ge_half (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : 0 ≤ x3)
  (h_sum : x1 + x2 + x3 ≤ 1/2) : 
  (1 - x1) * (1 - x2) * (1 - x3) ≥ 1 / 2 := 
by
  sorry

end prod_one_minus_nonneg_reals_ge_half_l275_275509


namespace trajectory_eq_no_point_N_l275_275354

-- Definitions based on conditions
def A : (ℝ × ℝ) := (0, -a)
def B : (ℝ × ℝ) := (0, a)
variables (a m : ℝ) (h_a_pos : 0 < a) (h_m_non_zero : m ≠ 0)

def k1 (x y : ℝ) := (y + a) / x
def k2 (x y : ℝ) := (y - a) / x

-- Statement for proof 1
theorem trajectory_eq (P : ℝ × ℝ) (h_k1k2_m : k1 a P.1 P.2 * k2 a P.1 P.2 = m) : P.2^2 - m * P.1^2 = a^2 :=
sorry

-- Additional foci definitions for proof 2
def F1 : (ℝ × ℝ) := (0, Real.sqrt (a^2 + a^2 / m))
def F2 : (ℝ × ℝ) := (0, -Real.sqrt (a^2 + a^2 / m))

-- Statement for proof 2
theorem no_point_N (N : ℝ × ℝ) (h_on_circle : N.1^2 + N.2^2 = a^2) (h_m_range : m ∈ Ioo (-1) 0) :
  ¬ (∃ N, (N.1^2 + N.2^2 = a^2) ∧ (1/2) * 2 * Real.sqrt (a^2 + a^2 / m) * |N.1| = Real.sqrt (-m) * a^2) :=
sorry

end trajectory_eq_no_point_N_l275_275354


namespace min_value_of_2a_b_c_l275_275995

-- Given conditions
variables (a b c : ℝ)
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (h : a * (a + b + c) + b * c = 4 + 2 * Real.sqrt 3)

-- Question to prove
theorem min_value_of_2a_b_c : 2 * a + b + c = 2 * Real.sqrt 3 + 2 :=
sorry

end min_value_of_2a_b_c_l275_275995


namespace distance_from_D_to_plane_parallel_AC1_l275_275747

noncomputable def distance_DP_AC1 : ℝ :=
  let B1 := (0 : ℝ, 0, 0)
  let A := (0 : ℝ, 1, 1)
  let C1 := (1 : ℝ, 0, 0)
  let D := (1 : ℝ, 1, 1)
  let M := ((1/2 : ℝ), 0, (1/2 : ℝ))
  let N := ((1/2 : ℝ), 0, (1/2 : ℝ))
  let P := ((1/4 : ℝ), 0, (1/4 : ℝ))
  let DP_distance := (fun (D P : ℝ × ℝ × ℝ) => 
    let (x1, y1, z1) := D
    let (x2, y2, z2) := P
    (Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)))
  (DP_distance (1, 1, 1) (1/4, 0, 1/4))

theorem distance_from_D_to_plane_parallel_AC1 : distance_DP_AC1 = sqrt(86) / 86 :=
  sorry

end distance_from_D_to_plane_parallel_AC1_l275_275747


namespace frequency_and_rate_third_group_l275_275295

/-- 
Given a sample containing 30 pieces of data, that are divided into 4 groups with the ratio 
of 2:4:3:1, we are to prove that the frequency and frequency rate of the third group are 9 
and 0.3 respectively.
-/
theorem frequency_and_rate_third_group (total_data : ℕ) (ratios : ℕ × ℕ × ℕ × ℕ)
  (h_total_data : total_data = 30) 
  (h_ratios : ratios = (2, 4, 3, 1)) :
  let third_group_frequency := total_data * 3 / (2 + 4 + 3 + 1)
  let third_group_rate := third_group_frequency / total_data in
  third_group_frequency = 9 ∧ third_group_rate = 0.3 :=
by 
  sorry

end frequency_and_rate_third_group_l275_275295


namespace bucket_capacity_l275_275136

theorem bucket_capacity :
  ∀ (A B C : ℕ), 
  (A = 16 * 55) → 
  (B = 10 * 85) →
  (A + B = 1730) →
  (C = 5) →
  (1730 / C = 346) :=
by
  intros A B C hA hB hAB hC
  rw [hA, hB, hAB, hC]
  norm_num
  sorry

end bucket_capacity_l275_275136


namespace sin_cos_identity_l275_275897

theorem sin_cos_identity : 
  sin (20 * Real.pi / 180) * cos (40 * Real.pi / 180) + cos (20 * Real.pi / 180) * sin (40 * Real.pi / 180) = (Real.sqrt 3) / 2 := 
by 
  sorry

end sin_cos_identity_l275_275897


namespace problem_statement_l275_275440

def line_m (x y : ℝ) : Prop := 2 * x - 3 * y + 30 = 0

def point_P : ℝ × ℝ := (10, 10)

def rotate_30_counterclockwise (x y : ℝ) : ℝ × ℝ :=
  let θ := real.pi / 6
  let cosθ := real.cos θ
  let sinθ := real.sin θ
  (cosθ * (x - 10) - sinθ * (y - 10) + 10, 
   sinθ * (x - 10) + cosθ * (y - 10) + 10)

noncomputable def line_n_x_intercept : ℝ := 
  (20 * real.sqrt 3 + 20) / (2 * real.sqrt 3 + 3)

theorem problem_statement : 
  ∃ x₀ : ℝ, (∀ y : ℝ, rotate_30_counterclockwise x₀ y → line_m x₀ y) ∧ 
            x₀ = line_n_x_intercept :=
  sorry

end problem_statement_l275_275440


namespace width_of_g_domain_l275_275366

noncomputable def h (x : ℝ) : ℝ := sorry

def domain (f : ℝ → ℝ) : Set ℝ := sorry

theorem width_of_g_domain (h_domain : domain h = Set.Icc (-9 : ℝ) 9) :
  let g (x : ℝ) := h (x / 3)
  domain g = Set.Icc (-27 : ℝ) 27 := sorry
   ∧ (27 - (-27) = 54 := sorry

end width_of_g_domain_l275_275366


namespace smallest_x_exists_l275_275984

theorem smallest_x_exists : ∃ x : ℕ, 
  (5 * x ≡ 25 [MOD 20]) ∧ 
  (3 * x + 1 ≡ 4 [MOD 7]) ∧ 
  (2 * x - 3 ≡ x [MOD 13]) ∧ 
  x > 0 ∧ 
  ∀ y : ℕ, ((5 * y ≡ 25 [MOD 20]) ∧ 
              (3 * y + 1 ≡ 4 [MOD 7]) ∧ 
              (2 * y - 3 ≡ y [MOD 13]) ∧ 
              y > 0) → x ≤ y :=
begin
  let x := 29,
  use x,
  split, norm_num, exact dec_trivial,
  split, norm_num, exact dec_trivial,
  split, norm_num, exact dec_trivial,
  split, norm_num,
  intros y hy,
  have h : y = 29 := by sorry,
  rw h,
  exact le_refl 29,
end

end smallest_x_exists_l275_275984


namespace PropA_PropB_PropC_PropD_l275_275545

-- Definitions corresponding to the given conditions in the problem

def P (A : Set Ω) (prob : Measure Ω) := prob A

def areIndependent (A B : Set Ω) (prob : Measure Ω) : Prop :=
  P (A ∩ B) prob = (P A prob) * (P B prob)

def arePairwiseIndependent (A B C : Set Ω) (prob : Measure Ω) : Prop :=
  areIndependent A B prob ∧ areIndependent A C prob ∧ areIndependent B C prob

def areComplementary (A B : Set Ω) (prob : Measure Ω) : Prop :=
  P A prob + P B prob = 1 ∧ A ∪ B = ⊤ ∧ A ∩ B = ⊥ 

-- Statements for each proposition
theorem PropA (A B : Set Ω) (prob : Measure Ω) 
  (cond1 : P A prob = 1/3)
  (cond2 : P B prob = 2/3) : 
  ¬ areComplementary A B prob := 
sorry

theorem PropB (A B : Set Ω) (prob : Measure Ω) 
  (cond1 : P A prob = 1/3)
  (cond2 : P B prob = 2/3)
  (cond3 : P (A ∩ B) prob = 2/9) : 
  areIndependent A B prob := 
sorry

theorem PropC (A B C : Set Ω) (prob : Measure Ω) 
  (cond1 : P A prob = 1/2)
  (cond2 : P B prob = 1/2)
  (cond3 : P C prob = 1/2)
  (cond4 : P (A ∩ B ∩ C) prob = 1/8) :
  ¬ arePairwiseIndependent A B C prob := 
sorry

theorem PropD (A B : Set Ω) (prob : Measure Ω) 
  (cond1 : P A prob = 0.7)
  (cond2 : P B prob = 0.6)
  (cond3 : P (A ∩ B) prob = 0.42) 
  (cond4 : P (A ∪ B) prob = 0.88) : 
  areIndependent A B prob ∧ 
  P (A ∪ B) prob = 0.88 := 
sorry

end PropA_PropB_PropC_PropD_l275_275545


namespace integer_with_18_factors_and_factors_18_24_l275_275842

theorem integer_with_18_factors_and_factors_18_24 (y : ℕ) :
  (nat.num_divisors y = 18) ∧ (18 ∣ y) ∧ (24 ∣ y) ↔ (y = 288) :=
by {
  sorry
}

end integer_with_18_factors_and_factors_18_24_l275_275842


namespace particle_speed_l275_275585

def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 4, 5 * t - 7)

theorem particle_speed :
  (∀ t, particle_position t = (3 * t + 4, 5 * t - 7)) →
  ∃ v, v = √34 ∧ ∀ t, speed particle_position t = v :=
by
  intros h
  sorry

end particle_speed_l275_275585


namespace probability_of_selecting_product_not_less_than_4_l275_275457

theorem probability_of_selecting_product_not_less_than_4 :
  let total_products := 5 
  let favorable_outcomes := 2 
  (favorable_outcomes : ℚ) / total_products = 2 / 5 := 
by 
  sorry

end probability_of_selecting_product_not_less_than_4_l275_275457


namespace probability_rain_at_least_four_out_of_five_l275_275122

open ProbabilityTheory

noncomputable def probability_rain_each_day : ℚ := 3 / 4

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_rain_at_least_four_out_of_five :
  let p := probability_rain_each_day in
  let q := 1 - p in
  let probability_exactly (k : ℕ) := (binomial 5 k : ℚ) * (p ^ k) * (q ^ (5 - k)) in
  probability_exactly 4 + probability_exactly 5 = 81 / 128 :=
by
  sorry

end probability_rain_at_least_four_out_of_five_l275_275122


namespace exists_sum_two_squares_pos_l275_275894
theorem exists_sum_two_squares_pos (n : ℕ) (hn : n > 2) :
     (∃ (a b : ℕ), n = a^2 + b^2) ∧ (∃ (x : ℕ), n^2 = (x + 1)^3 - x^3)
   
end exists_sum_two_squares_pos_l275_275894


namespace inequality_a5_b5_c5_l275_275668

theorem inequality_a5_b5_c5 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^5 + b^5 + c^5 ≥ a^3 * b * c + a * b^3 * c + a * b * c^3 :=
by
  sorry

end inequality_a5_b5_c5_l275_275668


namespace sample_center_on_regression_line_l275_275126

variables (n : ℕ)
variables (x y : Fin n → ℝ) -- Fin n represents a finite set of n elements.
variables (b a : ℝ)

-- Sample Center definition
def sampleCenter : ℝ × ℝ := 
  (1 / n * ∑ i in Finset.univ, x i, 1 / n * ∑ i in Finset.univ, y i)

-- Regression Line definition
noncomputable def regressionLine (x : ℝ) : ℝ := b * x + a

-- Statement: Prove that the sample center is on the regression line
theorem sample_center_on_regression_line (h : regressionLine (1 / n * ∑ i in Finset.univ, x i) = 1 / n * ∑ i in Finset.univ, y i) :
  sampleCenter n x y = (1 / n * ∑ i in Finset.univ, x i, regressionLine (1 / n * ∑ i in Finset.univ, x i)) :=
sorry

end sample_center_on_regression_line_l275_275126


namespace construct_triangle_with_incircle_l275_275964

theorem construct_triangle_with_incircle 
  (r : ℝ) 
  (d1 d2 : ℝ) 
  (h : d1 > r ∨ d2 > r) :
  ∃ (A B C : ℝ × ℝ), 
    is_incircle (A, B, C) ∧
    segment_lengths (A, B, C) = (d1, d2) ∧
    inradius (A, B, C) = r  := sorry

end construct_triangle_with_incircle_l275_275964


namespace tunnel_length_l275_275568

-- Definitions of the conditions
def train_length : ℝ := 2  -- The train is 2 miles long
def train_speed : ℝ := 120  -- The train is moving at 120 miles per hour
def time_in_minutes : ℝ := 2  -- The train exits the tunnel exactly 2 minutes after the front entered

-- Conversion factor
def minutes_per_hour : ℝ := 60

-- Calculation of the distance the train travels in 2 minutes
def distance_travelled := (train_speed / minutes_per_hour) * time_in_minutes

-- The proof problem
theorem tunnel_length (h : distance_travelled = 4) : ∃ l : ℝ, l = 2 :=
by
  -- the provided hint indicates a link between the length of the tunnel and the travelled distance
  sorry

end tunnel_length_l275_275568


namespace floor_b_eq_floor_sqrt_l275_275794

open Nat

-- Definition of b(n) based on the given conditions
def b (n : ℕ) : ℝ := Inf { k + (n : ℝ) / k | k : ℕ+ }

-- The main theorem to prove
theorem floor_b_eq_floor_sqrt (n : ℕ) : 
  ⌊b n⌋ = ⌊Real.sqrt (4 * n + 1)⌋ := 
by
  sorry

end floor_b_eq_floor_sqrt_l275_275794


namespace relationship_m_n_l275_275732

theorem relationship_m_n (m n : ℝ) (
  hB : (1, m) ∈ {p : ℝ × ℝ | ∃ k, p.snd = -2 / p.fst} 
  hC : (4, n) ∈ {p : ℝ × ℝ | ∃ k, p.snd = -2 / p.fst} 
) : m < n := 
  sorry

end relationship_m_n_l275_275732


namespace find_n_l275_275730

theorem find_n (n : ℕ) : 
  (1/5 : ℝ)^35 * (1/4 : ℝ)^n = (1 : ℝ) / (2 * 10^35) → n = 18 :=
by
  intro h
  sorry

end find_n_l275_275730


namespace quadratic_transformation_l275_275123

theorem quadratic_transformation (x : ℝ) :
  let b := 392
  let c := -153164
  let quadratic := x^2 + 784 * x + 500
  ∃ quadratic' = (x + b)^2 + c,
    (c / b) = -391 := by
  sorry

end quadratic_transformation_l275_275123


namespace sum_of_squares_50_l275_275653

theorem sum_of_squares_50 :
  (∑ i in Finset.range 51, i^2) = 42925 :=
by
  sorry

end sum_of_squares_50_l275_275653


namespace no_such_function_exists_l275_275776

def S := {n : ℕ // n ≥ 2}

theorem no_such_function_exists :
  ¬ ∃ f : S → S, ∀ a b : S, a ≠ b → f a * f b = f ⟨a.val^2 * b.val^2, sorry⟩ :=
by
  sorry

end no_such_function_exists_l275_275776


namespace number_of_ragas_l275_275942

def short_note := 1
def long_note := 2
def total_beats := 11

theorem number_of_ragas : (number_of_sequences short_note long_note total_beats) = 144 := 
sorry

end number_of_ragas_l275_275942


namespace sampling_interval_l275_275526

noncomputable def adjusted_population_size (pop_size : ℕ) (sample_size : ℕ) : ℕ :=
  pop_size - (pop_size % sample_size)

theorem sampling_interval (pop_size : ℕ) (sample_size : ℕ) (interval : ℕ)
  (h_pop_size : pop_size = 123) (h_sample_size : sample_size = 12)
  (h_interval : interval = (adjusted_population_size pop_size sample_size) / sample_size) :
  interval = 10 :=
by
  have h_adjusted_population_size : adjusted_population_size pop_size sample_size = 120 := by
    unfold adjusted_population_size
    rw [h_pop_size, h_sample_size]
    norm_num
  have h_interval_calc : interval = 120 / 12 := by
    rw [←h_interval, h_adjusted_population_size]
  norm_num at h_interval_calc
  exact h_interval_calc

end sampling_interval_l275_275526


namespace tan_subtraction_l275_275370

theorem tan_subtraction (α β : ℝ) (h₁ : Real.tan α = 11) (h₂ : Real.tan β = 5) : 
  Real.tan (α - β) = 3 / 28 := 
  sorry

end tan_subtraction_l275_275370


namespace common_difference_is_3_l275_275697

open Nat

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Given conditions
axiom a₄_eq_3 : ∀ a₁ d : ℝ, arithmetic_seq a₁ d 4 = 3
axiom a₃_a₁₁_eq_24 : ∀ a₁ d : ℝ, arithmetic_seq a₁ d 3 + arithmetic_seq a₁ d 11 = 24

-- Prove that the common difference d equals 3
theorem common_difference_is_3 (a₁ d : ℝ) : (∃ a₁ d : ℝ, a₄_eq_3 a₁ d ∧ a₃_a₁₁_eq_24 a₁ d) → d = 3 :=
by
  sorry

end common_difference_is_3_l275_275697


namespace range_of_a_l275_275491

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 + 3*a*x^2 + 3*(a + 2)*x + 1

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' a x1 = 0 ∧ f' a x2 = 0) ↔ (a < -1 ∨ a > 2) :=
by
  sorry

end range_of_a_l275_275491


namespace cost_of_four_dozen_l275_275407

/-- If the cost of three dozen oranges is 18 dollars,
then the cost of four dozen oranges is 24 dollars at the same rate. -/
theorem cost_of_four_dozen (cost_three_dozen : ℝ) (h : cost_three_dozen = 18) :
  let cost_per_dozen := cost_three_dozen / 3
  in 4 * cost_per_dozen = 24 :=
by
  sorry

end cost_of_four_dozen_l275_275407


namespace exists_strictly_increasing_sequence_l275_275455

open Nat

theorem exists_strictly_increasing_sequence (a1 : ℕ) (h : a1 > 1) :
  ∃ (a : ℕ → ℕ), (strict_mono a) ∧ (a 0 = a1) ∧ 
  ∀ k ≥ 1, (∑ i in range k, (a i)^2) % (∑ i in range k, a i) = 0 :=
sorry

end exists_strictly_increasing_sequence_l275_275455


namespace valid_documents_count_l275_275615

-- Definitions based on the conditions
def total_papers : ℕ := 400
def invalid_percentage : ℝ := 0.40
def valid_percentage : ℝ := 1.0 - invalid_percentage

-- Question and answer formalized as a theorem
theorem valid_documents_count : total_papers * valid_percentage = 240 := by
  sorry

end valid_documents_count_l275_275615


namespace find_abc_l275_275374

theorem find_abc 
  (a b c : ℕ) 
  (h_coprime : Nat.coprime a b ∧ Nat.coprime a c ∧ Nat.coprime b c)
  (h1 : a^2 ∣ b^3 + c^3)
  (h2 : b^2 ∣ a^3 + c^3)
  (h3 : c^2 ∣ a^3 + b^3)
  : a = 1 ∧ b = 1 ∧ c = 1 :=
by 
  sorry

end find_abc_l275_275374


namespace problem_part1_problem_part2_l275_275833

-- Define η function as a piecewise function
def eta (x : ℝ) : ℝ :=
  if x >= 0 then 1 else 0

-- Problem statement for the first part
theorem problem_part1 (x : ℝ) (k m : ℤ) :
  (x ≥ 0 ∧ x < 7 * Real.pi ∧ (eta x - eta (x - 7 * Real.pi) = 1)) ↔ 
  (∃ k, x = Real.pi / 2 + 2 * Real.pi * k ∧ k ∈ ({0, 1, 2, 3} : Set ℤ)) ∨ 
  (∃ m, x = Real.pi + 2 * Real.pi * m ∧ m ∈ ({0, 1, 2} : Set ℤ)) :=
sorry

-- Problem statement for the second part
theorem problem_part2 (x : ℝ) (m : ℤ) :
  ((x < 0 ∨ x ≥ 7 * Real.pi) ∧ (eta x - eta (x - 7 * Real.pi) = 0)) ↔ 
  (∃ m, x = Real.pi + 2 * Real.pi * m ∧ 
         (m ∈ Set.range (λ n : ℤ, n ≥ 3 - 1) ∨ m ∈ Set.range (λ n : ℤ, n ≤ -1))) :=
sorry

end problem_part1_problem_part2_l275_275833


namespace locus_proof_l275_275437

-- Define the distance formula function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the absolute value function in ℝ
def abs (x : ℝ) : ℝ := real.abs x

-- Define the locus equation as a predicate
def locus (x y : ℝ) : Prop := 3*x^2 - y^2 - 16*x + 20 = 0

-- Define the given condition
def condition (x y : ℝ) : Prop :=
  (distance (x, y) (4, 0)) / (abs (x - 3)) = 2

-- Prove that under the given condition, the point lies on the locus
theorem locus_proof (x y : ℝ) (h : condition x y) : locus x y :=
by
  sorry

end locus_proof_l275_275437


namespace expression_1_correct_expression_2_correct_l275_275957

noncomputable def expression_1 : ℝ :=
  0.64 ^ (-1 / 2) - (-1 / 8) ^ 0 + 8 ^ (2 / 3) + (9 / 16) ^ (1 / 2)

noncomputable def expression_2 : ℝ :=
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 5) + (Real.log 5)

theorem expression_1_correct : expression_1 = 6 := 
  by sorry

theorem expression_2_correct : expression_2 = 1 := 
  by sorry

end expression_1_correct_expression_2_correct_l275_275957


namespace nine_digit_integers_div_by_4950_result_l275_275358

noncomputable def nine_digit_integers_div_by_4950_count : ℕ :=
  576

theorem nine_digit_integers_div_by_4950_result :
  ∃ n : ℕ, n = nine_digit_integers_div_by_4950_count :=
by {
  use 576,
  sorry
}

end nine_digit_integers_div_by_4950_result_l275_275358


namespace number_of_quadruplets_l275_275381

variables (a b c : ℕ)

theorem number_of_quadruplets (h1 : 2 * a + 3 * b + 4 * c = 1200)
                             (h2 : b = 3 * c)
                             (h3 : a = 2 * b) :
  4 * c = 192 :=
by
  sorry

end number_of_quadruplets_l275_275381


namespace ellipse_equation_line_BC_fixed_point_l275_275333

theorem ellipse_equation (C : ellipse) (focus_C : point) (e : ℝ) (focus_parabola : point) (e_val : e = sqrt 2 / 2) 
  (focus_cond : focus_C = (1, 0)) (parabola_eq : parabola.focus = (1, 0)) :
  let a := sqrt 2
  let b_squared := a^2 - 1 
  C.equation = (λ (x y : ℝ), (x^2)/2 + y^2 = 1) 
 :=
begin
  sorry
end

theorem line_BC_fixed_point (C : ellipse) (A B : point) (slope_AB slope_AC : ℝ) 
  (C_eq : C.equation = (λ (x y : ℝ), (x^2)/2 + y^2 = 1))
  (A_coord : A = (0, 1))
  (slope_product : slope_AB * slope_AC = 1/4) :
  line_through A B = (0, 3)
  := 
begin
  sorry
end

end ellipse_equation_line_BC_fixed_point_l275_275333


namespace exists_r_l275_275477

variable {n : ℕ}
variables {a : Fin n → ℕ}

theorem exists_r (h_sum : (∑ i, a i) = 2 * n + 2) :
  ∃ r : Fin n,
    (∀ k : Fin n, k < r → a ((r + k + 1) % n) ≤ 2 * (k + 1)) ∨
    (∃ r₁ r₂ : Fin n, r₁ ≠ r₂ ∧ 
      (∀ k : Fin n, k < r₁ → a ((r₁ + k + 1) % n) ≤ 3) ∧ 
      (∀ k : Fin n, k < r₂ → a ((r₂ + k + 1) % n) ≤ 3)) ∧
    (∀ r : Fin n, 
      (∀ k : Fin n, k < r → a ((r + k + 1) % n) < 3) → 
      ∀ k : Fin n, k < r → a ((r + k + 1) % n) < 2 * (k + 1)) :=
by
  sorry

end exists_r_l275_275477


namespace find_1008th_number_l275_275925

theorem find_1008th_number (a : ℕ → ℚ) (h_len : ∀ n, 1 ≤ n ∧ n ≤ 2015)
  (h_prod_all : ∏ i in (Finset.range 2015), a i = 2015)
  (h_prod_three : ∀ i, 1 ≤ i ∧ i ≤ 2013 → a i * a (i + 1) * a (i + 2) = 1) :
  a 1007 = 1 / 2015 :=
sorry

end find_1008th_number_l275_275925


namespace xyz_ratio_l275_275273

theorem xyz_ratio (k x y z : ℝ) (h1 : x + k * y + 3 * z = 0)
                                (h2 : 3 * x + k * y - 2 * z = 0)
                                (h3 : 2 * x + 4 * y - 3 * z = 0)
                                (x_ne_zero : x ≠ 0)
                                (y_ne_zero : y ≠ 0)
                                (z_ne_zero : z ≠ 0) :
  (k = 11) → (x * z) / (y ^ 2) = 10 := by
  sorry

end xyz_ratio_l275_275273


namespace find_cos_angle_B_find_AC_length_l275_275741

noncomputable def cos_A : ℝ := 3 / 4
noncomputable def angle_C_eq_2A : Prop := ∀ A C : ℝ, C = 2 * A
noncomputable def dot_product_BA_BC : ℝ := 27 / 2

theorem find_cos_angle_B (A C B : ℝ) (angle_C_eq_2A : C = 2 * A) (cos_A : ℝ) (dot_product_BA_BC : ℝ) :
  ∃ (cos_B : ℝ), cos_B = 9 / 16 :=
begin
  -- Proof omitted
  sorry
end

theorem find_AC_length (A B C : ℝ) (angle_C_eq_2A : C = 2 * A) (cos_A : ℝ) (dot_product_BA_BC : ℝ) :
  ∃ (AC_length : ℝ), AC_length = 5 :=
begin
  -- Proof omitted
  sorry
end

end find_cos_angle_B_find_AC_length_l275_275741


namespace min_expected_value_iff_equal_probabilities_l275_275264

variable {n : ℕ} (p : Fin n → ℝ)

theorem min_expected_value_iff_equal_probabilities
  (h₁ : (∀ i j : Fin n, i ≠ j → (p i) * (p j) = 0))
  (h₂ : ∑ i, p i = 1) :
  ∑ i, (p i) * (p i) = 1 / n ↔ ∀ i : Fin n, p i = 1 / n := 
sorry

end min_expected_value_iff_equal_probabilities_l275_275264


namespace trip_total_charge_is_correct_l275_275020

-- Define the initial fee
def initial_fee : ℝ := 2.35

-- Define the charge per increment
def charge_per_increment : ℝ := 0.35

-- Define the increment size in miles
def increment_size : ℝ := 2 / 5

-- Define the total distance of the trip
def trip_distance : ℝ := 3.6

-- Define the total charge function
def total_charge (initial : ℝ) (increment_charge : ℝ) (increment : ℝ) (distance : ℝ) : ℝ :=
  initial + (distance / increment) * increment_charge

-- Prove the total charge for a trip of 3.6 miles is $5.50
theorem trip_total_charge_is_correct :
  total_charge initial_fee charge_per_increment increment_size trip_distance = 5.50 :=
by
  sorry

end trip_total_charge_is_correct_l275_275020


namespace piecewise_continuous_l275_275435

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x > 3 then ax + 4
  else if x >= -3 then 2x - 6
  else 3x - b

theorem piecewise_continuous (a b : ℝ) (h1 : 3 * a + 4 = 6) (h2 : -12 = -9 - b) :
  a + b = -13 / 3 :=
by
  sorry

end piecewise_continuous_l275_275435


namespace monthly_earnings_l275_275810

variable (e : ℕ) (s : ℕ) (p : ℕ) (t : ℕ)

-- conditions
def half_monthly_savings := s = e / 2
def car_price := p = 16000
def saving_months := t = 8
def total_saving := s * t = p

theorem monthly_earnings : ∀ (e s p t : ℕ), 
  half_monthly_savings e s → 
  car_price p → 
  saving_months t → 
  total_saving s t p → 
  e = 4000 :=
by
  intros e s p t h1 h2 h3 h4
  sorry

end monthly_earnings_l275_275810


namespace find_angle_EDC_l275_275298

-- Given conditions
variables (A B C D E : Type) [point A] [point B] [point C] [point D] [point E]
variable (triangleABC : triangle A B C)
variable (angleABC : angle B A C)
variable (angleBAC : angle B A C)
variable (angleDEC : angle D E C)

-- Additional conditions
axiom BC_on_extend_D : ∃ D, line.extended_from C B
axiom DE_perpendicular_to_BC : ∃ E, line.perpendicular_to DE (line B C)
axiom angle_ABC := 70
axiom angle_BAC := 50
axiom angle_DEC := 100

-- Proof problem
theorem find_angle_EDC : angle D E C = 80 := by
  sorry

end find_angle_EDC_l275_275298


namespace problem_statement_l275_275331

open Real

-- Defining the circle
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Definitions for intersection, midpoint and other given conditions
def midpoint (A B P : ℝ × ℝ) : Prop := P = (A.1 + B.1)/2, (A.2 + B.2)/2

def intersects (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  circle A.1 A.2 ∧ circle B.1 B.2 ∧ A ≠ B ∧ ∃ t : ℝ, A = (t, l t) ∧ ∃ t : ℝ, B = (t, l t)

-- Equation of line l
def line_l (x : ℝ) : ℝ := -x + 2

theorem problem_statement (A B : ℝ × ℝ) :
  intersects line_l A B →
  midpoint A B (1, 1) →
  (∀ x y, y = line_l x → x + y - 2 = 0) ∧
  (∃ L : ℝ, L = 2*sqrt(2)) :=
by
  sorry

end problem_statement_l275_275331


namespace magnitude_of_AB_find_z_z_real_z_pure_imag_l275_275184

section Problem1
variables {A B : ℂ}
def OA : ℂ := 7 + 1*I
def OB : ℂ := 3 - 2*I
def AB : ℂ := OB - OA

theorem magnitude_of_AB : abs AB = 5 := by 
  sorry
end Problem1

section Problem2
variables {z : ℂ}
theorem find_z : (1 + 2*I) * conj(z) = (4 + 3*I) → z = 2 + I := by
  sorry
end Problem2

section Problem3
variables  {m : ℝ}
def z (m : ℝ) : ℂ := (m*(m-2))/(m-1) + (m^2 + 2*m - 3)*I

theorem z_real : (z m).im = 0 → m = -3 := by 
  sorry

theorem z_pure_imag : (z m).re = 0 → m = 0 ∨ m = 2 := by
  sorry
end Problem3

end magnitude_of_AB_find_z_z_real_z_pure_imag_l275_275184


namespace number_of_kids_l275_275096

-- Define the conditions
def whiteboards_per_kid_per_min (whiteboards total_time : ℕ) : ℕ → ℝ := λ k, (whiteboards : ℝ) / (total_time : ℝ) * (k : ℝ)

def time_per_whiteboard (time : ℕ) (whiteboards : ℕ) : ℝ := (time : ℝ) / (whiteboards : ℝ)

-- Given conditions
def condition1 := whiteboards_per_kid_per_min 3 20
def condition2 := time_per_whiteboard 160 6

-- Proof statement
theorem number_of_kids (k : ℕ) : k * (time_per_whiteboard 160 6) = 20 → k * 3 / 20 = 4 :=
by
  -- ensure the correct time per whiteboard
  have h : (time_per_whiteboard 160 6) = 26.67 := sorry
  -- ensure the correct whiteboards per kid in 20 minutes
  have g : (whiteboards_per_kid_per_min 3 20 k) = (0.75 : ℝ) := sorry
  -- use both conditions to prove the number of kids
  linarith
  

end number_of_kids_l275_275096


namespace combined_average_score_clubs_l275_275188

theorem combined_average_score_clubs
  (nA nB : ℕ) -- Number of members in each club
  (avgA avgB : ℝ) -- Average score of each club
  (hA : nA = 40)
  (hB : nB = 50)
  (hAvgA : avgA = 90)
  (hAvgB : avgB = 81) :
  (nA * avgA + nB * avgB) / (nA + nB) = 85 :=
by
  sorry -- Proof omitted

end combined_average_score_clubs_l275_275188


namespace find_tanB_l275_275398

-- Define the main theorem
theorem find_tanB (a b c : ℝ) (A B C : ℝ) 
  (h1 : (b^2 + c^2 - a^2 = (8 / 5) * b * c))
  (h2 : (cos A / a + cos B / b = sin C / c)) :
  tan B = -3 := 
sorry

end find_tanB_l275_275398


namespace special_pair_sum_condition_l275_275583

def is_special (a b : ℕ) : Prop :=
  (a = b + 1) ∨ (b = a + 1)

def sum_of_special_pairs (n m : ℕ) : Prop :=
  ∃ (k : ℕ) (pairs : Fin k → (ℕ × ℕ)),
    k ≥ 2 ∧
    (∀ i, is_special (pairs i).fst (pairs i).snd) ∧
    ((Finset.univ.sum (λ i, (pairs i).fst), Finset.univ.sum (λ i, (pairs i).snd)) = (n, m))

theorem special_pair_sum_condition (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : ¬ is_special n m) :
  sum_of_special_pairs n m ↔ n + m ≥ (n - m) ^ 2 := 
sorry

end special_pair_sum_condition_l275_275583


namespace eric_rides_at_constant_rate_l275_275113

def constant_rate : ℕ := 9

theorem eric_rides_at_constant_rate (c : ℕ) (k : ℕ → ℕ) (h1 : c = 1.5) (h2 : k 60 = 9) : 
  let rate := (λ t : ℕ, c * (t / 10)) in
  rate 60 = constant_rate :=
by
  sorry

end eric_rides_at_constant_rate_l275_275113


namespace printer_fraction_l275_275511

noncomputable def basic_computer_price : ℝ := 2000
noncomputable def total_basic_price : ℝ := 2500
noncomputable def printer_price : ℝ := total_basic_price - basic_computer_price -- inferred as 500

noncomputable def enhanced_computer_price : ℝ := basic_computer_price + 500
noncomputable def total_enhanced_price : ℝ := enhanced_computer_price + printer_price -- inferred as 3000

theorem printer_fraction  (h1 : basic_computer_price + printer_price = total_basic_price)
                          (h2 : basic_computer_price = 2000)
                          (h3 : enhanced_computer_price = basic_computer_price + 500) :
  printer_price / total_enhanced_price = 1 / 6 :=
  sorry

end printer_fraction_l275_275511


namespace area_of_shaded_region_l275_275516

/-- In a configuration where a smaller circle of radius 2 is internally tangent to two larger overlapping circles, each of radius 5, at points A and B where AB is the diameter of the smaller circle, the area of the region outside the smaller circle and inside each of the two larger circles is (9 / 2) * π - 20. -/
theorem area_of_shaded_region :
  let r_small := 2 in
  let r_large := 5 in
  let area_sm_circle := π * r_small^2 in
  let area_sector_large := (1 / 4) * π * r_large^2 in
  let area_triangle := (1/2) * r_large * (2 * r_small) in
  2 * (area_sector_large - area_triangle - area_sm_circle) = (9 / 2) * π - 20 :=
by
  sorry

end area_of_shaded_region_l275_275516


namespace hexagon_chord_length_l275_275577

theorem hexagon_chord_length (circle : Type) (hexagon : Type) 
  (inscribed_in : hexagon → circle) 
  (A B C D E F : hexagon) 
  (AB DE CF : ℝ) (BC EF DA : ℝ)
  (h1 : AB = 5) (h2 : DE = 5) (h3 : CF = 5)
  (h4 : BC = 4) (h5 : EF = 4) (h6 : DA = 4) : 
  ∃ (m n : ℕ), nat.gcd m n = 1 ∧ (chord_length circle hexagon (A E) = m / n) ∧ 
  m + n = 229 :=
by 
  sorry

end hexagon_chord_length_l275_275577


namespace sum_all_integers_c_l275_275061

theorem sum_all_integers_c (T : ℤ) : 
  (∀ c : ℤ, (∃ a b : ℤ, (x^2 + c * x + 2010 * c = (x + a) * (x + b))) 
    → c ∈ T) → |T| = 361800 := sorry

end sum_all_integers_c_l275_275061


namespace cake_slices_left_l275_275934

-- Setup initial conditions as definitions
def initial_cakes := 2
def slices_per_cake := 8
def slices_given_friends (total_slices : Nat) := total_slices / 4
def slices_given_family (remaining_slices : Nat) := remaining_slices / 3
def slices_eaten := 3

-- Define a theorem to prove the final number of slices
theorem cake_slices_left : 
  ∀ (initial_cakes slices_per_cake slices_eaten : Nat)
    (slices_given_friends slices_given_family : Nat → Nat),
  let total_slices := initial_cakes * slices_per_cake in
  let remaining_after_friends := total_slices - slices_given_friends(total_slices) in
  let remaining_after_family := remaining_after_friends - slices_given_family(remaining_after_friends) in
  final_slices = remaining_after_family - slices_eaten :=
by 
  intros
  sorry

end cake_slices_left_l275_275934


namespace plot_length_is_64_l275_275843

def length_of_plot_is_64 (breadth : ℝ) : Prop :=
  let length := breadth + 28 in
  let cost_per_meter := 26.5 in
  let total_cost := 5300 in
  let perimeter := total_cost / cost_per_meter in
  2 * length + 2 * breadth = perimeter

theorem plot_length_is_64 : ∃ (b : ℝ), length_of_plot_is_64 b ∧ b + 28 = 64 :=
by
  sorry

end plot_length_is_64_l275_275843


namespace speed_of_train_is_20_l275_275548

def length_of_train := 120 -- in meters
def time_to_cross := 6 -- in seconds

def speed_of_train := length_of_train / time_to_cross -- Speed formula

theorem speed_of_train_is_20 :
  speed_of_train = 20 := by
  sorry

end speed_of_train_is_20_l275_275548


namespace distribute_books_l275_275969

theorem distribute_books (m n : ℕ) (h1 : m = 3*n + 8) (h2 : ∃k, m = 5*k + r ∧ r < 5 ∧ r > 0) : 
  n = 5 ∨ n = 6 :=
by sorry

end distribute_books_l275_275969


namespace sum_of_real_roots_eq_l275_275705

def f (x : ℝ) : ℝ :=
  cos x * (2 * sqrt 3 * sin x - cos x) - (1 / 2) * cos (2 * x) + (1 / 2)

theorem sum_of_real_roots_eq (a : ℝ) (h1 : -1 < a) (h2 : a < 0) :
  ∑ x in (set_of (λ x, f x = a)).to_finset.filter (λ x, 0 ≤ x ∧ x ≤ 2 * real.pi), x
    = 13 * real.pi / 3 :=
  sorry

end sum_of_real_roots_eq_l275_275705


namespace cos_neg_79_pi_over_6_l275_275512

theorem cos_neg_79_pi_over_6 : 
  Real.cos (-79 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_neg_79_pi_over_6_l275_275512


namespace concurrency_on_euler_line_l275_275775

structure Triangle :=
(A B C : Point)

structure Point := 
(x y : ℝ)

def feet_of_altitudes (ABC : Triangle) : Point × Point × Point := sorry

def incircle_touches_at (A1 B1 C1 : Point) : Point × Point × Point := sorry

def euler_line (ABC : Triangle) : Line := sorry

def concurrency_point (AA2 BB2 CC2 : Line) : Point := sorry

theorem concurrency_on_euler_line (ABC : Triangle) 
  (A1 B1 C1 : Point) (hA : feet_of_altitudes ABC = (A1, B1, C1))
  (A2 B2 C2 : Point) (hB : incircle_touches_at A1 B1 C1 = (A2, B2, C2)) :
  let AA2 := LineThrough ABC.A A2
  let BB2 := LineThrough ABC.B B2
  let CC2 := LineThrough ABC.C C2
  let concurrency_pt := concurrency_point AA2 BB2 CC2
  concurrency_pt ∈ euler_line ABC :=
sorry

end concurrency_on_euler_line_l275_275775


namespace no_perfect_squares_l275_275281

theorem no_perfect_squares (x y z t : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : t > 0)
  (h5 : x * y - z * t = x + y) (h6 : x + y = z + t) : ¬(∃ a b : ℕ, a^2 = x * y ∧ b^2 = z * t) := 
by
  sorry

end no_perfect_squares_l275_275281


namespace usual_time_to_school_l275_275143

variable (R T : ℚ)

def distance (r t : ℚ) : ℚ := r * t

theorem usual_time_to_school :
  (distance R T = distance (7/6 * R) (T - 2)) → T = 14 :=
by
  intros h
  have h1 : R * T = (7/6) * R * (T - 2) := h
  have h2 : T = 14 := 
    calc 
      T = (7/6 * T - 7/3) :
        by sorry
  exact h2

end usual_time_to_school_l275_275143


namespace function_properties_l275_275837

/-- Define the function f. -/
def f (x : ℝ) : ℝ := sin (x - π / 4) * cos (x + π / 4) + 1 / 2

/-- Prove that the function is odd and has a period of π. -/
theorem function_properties :
  (∀ x, f (-x) = -f x) ∧ (∃ T > 0, T = π ∧ ∀ x, f (x + T) = f x) :=
by
  sorry

end function_properties_l275_275837


namespace root_product_l275_275146

theorem root_product : (Real.sqrt (Real.sqrt 81) * Real.cbrt 27 * Real.sqrt 9 = 27) :=
by
  sorry

end root_product_l275_275146


namespace relationship_among_MNP_l275_275683

variables {a b c M N P : ℝ}

theorem relationship_among_MNP (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < 1) : 
  let M := 2^a,
      N := 5^(-b),
      P := Real.log c 
  in P < N ∧ N < M := 
by 
  sorry

end relationship_among_MNP_l275_275683


namespace equation_in_terms_of_y_l275_275811

theorem equation_in_terms_of_y (x y : ℝ) (h : 2 * x + y = 5) : y = 5 - 2 * x :=
sorry

end equation_in_terms_of_y_l275_275811


namespace shaded_area_correct_l275_275763

def left_rectangle_area : ℝ := 3 * 6
def right_rectangle_area : ℝ := 5 * 10
def left_semicircle_area : ℝ := (1 / 2) * Real.pi * (3 ^ 2)
def right_semicircle_area : ℝ := (1 / 2) * Real.pi * (5 ^ 2)
def isosceles_triangle_area : ℝ := (1 / 2) * 3 * 3

def left_shaded_area : ℝ := left_rectangle_area - left_semicircle_area - isosceles_triangle_area
def right_shaded_area : ℝ := right_rectangle_area - right_semicircle_area

def total_shaded_area : ℝ := left_shaded_area + right_shaded_area

theorem shaded_area_correct : total_shaded_area = 63.5 - 17 * Real.pi := by
  sorry

end shaded_area_correct_l275_275763


namespace max_sin_cos_l275_275118

theorem max_sin_cos : ∀ x : Real, (sin x + cos x) ≤ Real.sqrt 2 := by
  sorry

end max_sin_cos_l275_275118


namespace handshake_count_l275_275889

theorem handshake_count (n : ℕ) (hn : n = 8) : nat.choose n 2 = 28 := by
  rw hn
  -- nat.choose 8 2 calculation
  sorry

end handshake_count_l275_275889


namespace width_of_field_l275_275554

-- Definitions for the conditions
variables (W L : ℝ) (P : ℝ)
axiom length_condition : L = (7 / 5) * W
axiom perimeter_condition : P = 2 * L + 2 * W
axiom perimeter_value : P = 336

-- Theorem to be proved
theorem width_of_field : W = 70 :=
by
  -- Here will be the proof body
  sorry

end width_of_field_l275_275554


namespace chord_length_polar_coord_l275_275006

theorem chord_length_polar_coord :
  (∃ θ ρ : ℝ, θ = π / 4 ∧ ρ = 4 * cos θ ∧ 2 * ρ = 4 * sqrt 2) :=
by
  -- let θ = π / 4 and ρ = 2 * sqrt 2
  use π / 4, 2 * sqrt 2
  -- prove the conditions hold
  split
  -- condition to ensure that θ = π / 4
  {
    exact eq.refl (π / 4),
  },
  split
  -- condition to ensure that ρ = 4 * cos θ
  {
    calc 4 * cos (π / 4) = 4 * (sqrt 2 / 2) : by rw cos_pi_div_four
    ... = 2 * sqrt 2 : by ring,
  },
  -- prove the final condition
  {
    exact eq.refl (4 * sqrt 2),
  }

end chord_length_polar_coord_l275_275006


namespace compare_magnitude_l275_275284

theorem compare_magnitude (a b : ℝ) (ha : a > 0) (hb : b > 0) : (sqrt a + sqrt b) > (sqrt (a + b)) := by
  sorry

end compare_magnitude_l275_275284


namespace meaning_of_sum_of_squares_l275_275495

theorem meaning_of_sum_of_squares (a b : ℝ) : a ^ 2 + b ^ 2 ≠ 0 ↔ (a ≠ 0 ∨ b ≠ 0) :=
by
  sorry

end meaning_of_sum_of_squares_l275_275495


namespace butterfly_area_extremes_l275_275680

noncomputable def butterfly_shape_area (ep e p θ: ℝ) : ℝ :=
  let ρ1 := ep / (1 - e * real.cos θ)
  let ρ2 := ep / (1 + e * real.sin θ)
  let ρ3 := ep / (1 + e * real.cos θ)
  let ρ4 := ep / (1 - e * real.sin θ)
  (1/2) * (ρ1 * ρ2 + ρ3 * ρ4)

theorem butterfly_area_extremes (a b e p : ℝ) (h1 : e ≠ 0) (h2 : a = p * (1 - e^2)) :
  ∃ (S_max S_min : ℝ), 
    S_max = b^2 ∧ 
    S_min = (2 * b^4) / (a^2 + b^2) ∧ 
    ∀ θ ∈ set.Icc 0 (real.pi / 2), 
    let S := butterfly_shape_area (e * p) e p θ in 
    S ≤ S_max ∧ S ≥ S_min :=
begin
  sorry -- Proof not required as per the prompt
end

end butterfly_area_extremes_l275_275680


namespace right_triangle_tan_l275_275752

theorem right_triangle_tan (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (angle_BAC : MeasureTheory.Angle B A C = MeasureTheory.Angle.right)
  (AB_len : dist A B = 40)
  (BC_len : dist B C = 41) :
  tangent_of_angle A B C = 9 / 40 :=
sorry

end right_triangle_tan_l275_275752


namespace sqrt_monotonically_increasing_l275_275877

-- Define the given functions
def fA (x : ℝ) : ℝ := -x + 1
def fB (x : ℝ) : ℝ := (x - 1)^2
def fC (x : ℝ) : ℝ := Real.sin x
def fD (x : ℝ) := Real.sqrt x

-- State the main theorem to be proved
theorem sqrt_monotonically_increasing : 
  ∀ x y : ℝ, 0 < x → x < y → fD x < fD y := 
sorry

end sqrt_monotonically_increasing_l275_275877


namespace digits_making_number_divisible_by_4_l275_275201

theorem digits_making_number_divisible_by_4 (N : ℕ) (hN : N < 10) :
  (∃ n0 n4 n8, n0 = 0 ∧ n4 = 4 ∧ n8 = 8 ∧ N = n0 ∨ N = n4 ∨ N = n8) :=
by
  sorry

end digits_making_number_divisible_by_4_l275_275201


namespace small_bottle_price_l275_275406

noncomputable def price_per_small_bottle (P : ℝ) : Prop :=
  1300 * 1.89 + 750 * P = 1.7034 * (1300 + 750)

theorem small_bottle_price : ∃ P : ℝ, price_per_small_bottle P ∧ P ≈ 1.38 := by
  sorry

end small_bottle_price_l275_275406


namespace minimum_moves_l275_275389

theorem minimum_moves (n : ℕ) : 
  n > 0 → ∃ k l : ℕ, k + 2 * l ≥ ⌊ (n^2 : ℝ) / 2 ⌋₊ ∧ k + l ≥ ⌊ (n^2 : ℝ) / 3 ⌋₊ :=
by 
  intro hn
  sorry

end minimum_moves_l275_275389


namespace product_of_roots_l275_275152

theorem product_of_roots : 
  (Real.root 81 4) * (Real.root 27 3) * (Real.sqrt 9) = 27 :=
by
  sorry

end product_of_roots_l275_275152


namespace meeting_people_count_l275_275854

theorem meeting_people_count (k : ℕ) (h1 : ∀ (p q : ℕ), ∃ d : ℕ, ∀ r : ℕ, (r ≠ p ∧ r ≠ q) → (∃ s : ℕ, s = 1)) : ∃ n : ℕ, n = 36 :=
by
  have h2 : ∀ x, x = (12 * k) := sorry
  have h3 : ∀ (p : ℕ), ∃ y : ℕ, y = (3 * k + 6) := sorry
  exists n
  sorry

end meeting_people_count_l275_275854


namespace gcd_136_1275_l275_275114

theorem gcd_136_1275 : Nat.gcd 136 1275 = 17 := by
sorry

end gcd_136_1275_l275_275114


namespace solve_for_x_in_equation_l275_275092

theorem solve_for_x_in_equation (x : ℝ)
  (h : (2 / 7) * (1 / 4) * x = 12) : x = 168 :=
sorry

end solve_for_x_in_equation_l275_275092


namespace arithmetic_sequence_general_term_absolute_sum_first_19_terms_l275_275349

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (h1 : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = a n + a (n + 2))
  (h2 : a 1 + a 4 = 41) (h3 : a 3 + a 7 = 26) :
  ∀ n : ℕ, a n = 28 - 3 * n := 
sorry

theorem absolute_sum_first_19_terms (a : ℕ → ℤ) (h1 : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = a n + a (n + 2))
  (h2 : a 1 + a 4 = 41) (h3 : a 3 + a 7 = 26) (an_eq : ∀ n : ℕ, a n = 28 - 3 * n) :
  |a 1| + |a 3| + |a 5| + |a 7| + |a 9| + |a 11| + |a 13| + |a 15| + |a 17| + |a 19| = 150 := 
sorry

end arithmetic_sequence_general_term_absolute_sum_first_19_terms_l275_275349


namespace part1_part2_l275_275305

section
variable (f g : ℝ → ℝ) (a : ℝ)

-- Define the given functions.
noncomputable def f := λ x : ℝ, x^3 - x
noncomputable def g := λ x : ℝ, x^2 + a

-- Part 1: Prove that if x₁ = -1, then a = 3
theorem part1 (x₁ : ℝ) (h : x₁ = -1) (h_tangent : ∀ x₃ : ℝ, f'(x₃) = g'(x₃)) : 
    a = 3 :=
sorry

-- Part 2: Prove the range of values for a is [-1, +∞).
theorem part2 (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, 3 * x₁^2 - 1 = 2 * x₂ ∧ a = x₂^2 - 2 * x₁^3) : 
    a ∈ Set.Ici (-1) :=
sorry

end

end part1_part2_l275_275305


namespace longer_piece_length_is_20_l275_275475

-- Define the rope length
def ropeLength : ℕ := 35

-- Define the ratio of the two pieces
def ratioA : ℕ := 3
def ratioB : ℕ := 4
def totalRatio : ℕ := ratioA + ratioB

-- Define the length of each part
def partLength : ℕ := ropeLength / totalRatio

-- Define the length of the longer piece
def longerPieceLength : ℕ := ratioB * partLength

-- Theorem to prove that the length of the longer piece is 20 inches
theorem longer_piece_length_is_20 : longerPieceLength = 20 := by 
  sorry

end longer_piece_length_is_20_l275_275475


namespace digits_1150_multiple_of_5_l275_275392

theorem digits_1150_multiple_of_5 : 
  let digits := [1, 1, 5, 0] in 
  (∀ permutation, multiset.perm digits permutation → list.length permutation = 4 → permutation.ilast ∈ [0, 5] → permutation.iget = 5) :=
by
  sorry

end digits_1150_multiple_of_5_l275_275392


namespace jellybean_problem_l275_275138

theorem jellybean_problem 
    (T L A : ℕ) 
    (h1 : T = L + 24) 
    (h2 : A = L / 2) 
    (h3 : T = 34) : 
    A = 5 := 
by 
  sorry

end jellybean_problem_l275_275138


namespace eval_expression_l275_275642

-- Define the floor and ceiling functions based on their properties
def my_floor (x : ℝ) : ℤ := int.floor x
def my_ceil (x : ℝ) : ℤ := int.ceil x

-- Define the values in the problem
def a := -5.67
def b := 34.1

-- Define the overall problem
theorem eval_expression :
  3 * (my_floor a + my_ceil b) = 87 :=
by
  -- Skipping the proof
  sorry

end eval_expression_l275_275642


namespace triangle_equality_AC_CM_l275_275002

theorem triangle_equality_AC_CM
  (A B C P H K M : Type)
  [triangle ABC]
  (altitude_CH : IsAltitude C H AB)
  (reflection_P : IsReflect A P BC)
  (circumcircle_intersection_K : IsSecondIntersection CH (circumcircle_of_triangle ACP) K)
  (KP_intersects_AB_at_M : Intersects KP AB M) :
  length AC = length CM := 
sorry

end triangle_equality_AC_CM_l275_275002


namespace prime_multiple_of_11_probability_l275_275807

theorem prime_multiple_of_11_probability :
  let cards := {n | n ∈ finset.range 1 101} in
  let primes := {n | nat.prime n} in
  let multiples_of_11 := {n | n % 11 = 0} in
  let eligible_cards := cards ∩ primes ∩ multiples_of_11 in
  (finset.card eligible_cards : ℚ) / (finset.card cards : ℚ) = 1 / 100 :=
by
  sorry

end prime_multiple_of_11_probability_l275_275807


namespace quadrilateral_area_l275_275832

def polynomial (x : ℝ) : ℝ := x^4 - 5 * x^3 + 8 * x^2 - 5 * x + 1 / 2

theorem quadrilateral_area 
  (a b c d : ℝ) 
  (h_roots : polynomial a = 0 ∧ polynomial b = 0 ∧ polynomial c = 0 ∧ polynomial d = 0) : 
  let s := (a + b + c + d) / 2 in 
  sqrt ((s - a) * (s - b) * (s - c) * (s - d)) = sqrt (5 / 4) := 
sorry

end quadrilateral_area_l275_275832


namespace intersection_point_of_lines_l275_275980

theorem intersection_point_of_lines :
  ∃ x y : ℝ, (5 * x - 2 * y = 20) ∧ (3 * x + 2 * y = 8) ∧ x = 3.5 ∧ y = -1.25 :=
begin
  sorry
end

end intersection_point_of_lines_l275_275980


namespace inequality_proof_l275_275065

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) :=
by
  sorry

end inequality_proof_l275_275065


namespace problem_1_problem_2_l275_275315

-- Given functions f(x) = x^3 - x and g(x) = x^2 + a
def f (x : ℝ) : ℝ := x ^ 3 - x
def g (x : ℝ) (a : ℝ) : ℝ := x ^ 2 + a

-- (1) If x1 = -1, prove the value of a is 3 given the tangent line condition.
theorem problem_1 (a : ℝ) : 
  let x1 := -1 in
  let f' (x : ℝ) : ℝ := 3 * x ^ 2 - 1 in
  let g' (x : ℝ) : ℝ := 2 * x in
  f' x1 = 2 →  -- Slope of the tangent line at x1 = -1 for f(x)
  g' 1 = 2 →  -- Slope of the tangent line at x2 = 1 for g(x)
  (f x1 + (2 * (1 + 1)) = g 1 a) → -- The tangent line condition
  a = 3 := 
sorry

-- (2) Find the range of values for a under the tangent line condition.
theorem problem_2 (a : ℝ) : 
  let h (x1 : ℝ) := 9 * x1 ^ 4 - 8 * x1 ^ 3 - 6 * x1 ^ 2 + 1 in
   -- We check that the minimum value is h(1) = -4, which gives a range for a
  (∃ x1, x1 ≠ 1 ∧ a ≥ (-1)) :=
sorry

end problem_1_problem_2_l275_275315


namespace gcd_of_102_and_238_l275_275524

theorem gcd_of_102_and_238 : Nat.gcd 102 238 = 34 := 
by 
  sorry

end gcd_of_102_and_238_l275_275524


namespace price_of_first_doughnut_l275_275835

theorem price_of_first_doughnut 
  (P : ℕ)  -- Price of the first doughnut
  (total_doughnuts : ℕ := 48)  -- Total number of doughnuts
  (price_per_dozen : ℕ := 6)  -- Price per dozen of additional doughnuts
  (total_cost : ℕ := 24)  -- Total cost spent
  (doughnuts_left : ℕ := total_doughnuts - 1)  -- Doughnuts left after the first one
  (dozens : ℕ := doughnuts_left / 12)  -- Number of whole dozens
  (cost_of_dozens : ℕ := dozens * price_per_dozen)  -- Cost of the dozens of doughnuts
  (cost_after_first : ℕ := total_cost - cost_of_dozens)  -- Remaining cost after dozens
  : P = 6 := 
by
  -- Proof to be filled in
  sorry

end price_of_first_doughnut_l275_275835


namespace ratio_AP_PB_l275_275867

open EuclideanGeometry

-- Definitions of the points
variables {A B C D E F P : Point}
variables {AB BC BD AD AE AP PB : Real}

-- Conditions
axiom AB_length : 0 < AB
axiom BC_equal_half_AB : BC = AB / 2
axiom BC_perpendicular_AB : Perpendicular B C A B
axiom C_perpendicular_AC : Perpendicular C C A B
axiom DE_equal_parallel_AB : DE_parallel_AB (DE = AB)
axiom E_perpendicular_AB : Perpendicular E E A B
axiom P_parallel_DF : Parallel DF P C

-- To prove
theorem ratio_AP_PB : AP / PB = 5 / 4 :=
begin
  sorry
end

end ratio_AP_PB_l275_275867


namespace real_imag_product_l275_275733

def z : ℂ := 1 - 1 * complex.i

theorem real_imag_product : (z.re * z.im) = -1 :=
by
  -- Proof goes here
  sorry

end real_imag_product_l275_275733


namespace sum_G_equals_126_l275_275275

noncomputable def G (n : ℕ) : ℕ :=
  if 2 ≤ n ∧ n ≤ 10 then 2 * (n + 1) else 0

theorem sum_G_equals_126 : (Finset.sum (Finset.range 11).filter (λ n, 2 ≤ n) G) = 126 := by
  sorry

end sum_G_equals_126_l275_275275


namespace equal_probability_of_selection_l275_275466

-- Define the number of total students
def total_students : ℕ := 86

-- Define the number of students to be eliminated through simple random sampling
def eliminated_students : ℕ := 6

-- Define the number of students selected through systematic sampling
def selected_students : ℕ := 8

-- Define the probability calculation
def probability_not_eliminated : ℚ := 80 / 86
def probability_selected : ℚ := 8 / 80
def combined_probability : ℚ := probability_not_eliminated * probability_selected

theorem equal_probability_of_selection :
  combined_probability = 4 / 43 :=
by
  -- We do not need to complete the proof as per instruction
  sorry

end equal_probability_of_selection_l275_275466


namespace correct_statements_A_and_D_l275_275879

-- Define the necessary statements as conditions.
def statementA (x y : ℝ) : Prop := (x * y > 0) ↔ (x / y > 0)
def statementB (x : ℝ) : Prop := ∃ x : ℝ, sqrt (x ^ 2 + 9) + 1 / sqrt (x ^ 2 + 9) = 2
def statementC (x : ℝ) : Prop := ∃ x : ℝ, (x ^ 2 + 5) / sqrt (x ^ 2 + 4) = 2
def statementD (x : ℝ) : Prop := ((x + 1) * (2 - x) < 0) ↔ (x < -1 ∨ x > 2)

-- We need to prove that statements A and D are correct.
-- Translate these into theorem statements.
theorem correct_statements_A_and_D :
  (∀ x y : ℝ, statementA x y) ∧ (∀ x : ℝ, statementD x) :=
by {
  split,
  { intros x y,
    sorry, -- Proof for statement A
  },
  { intros x,
    sorry, -- Proof for statement D
  }
}

end correct_statements_A_and_D_l275_275879


namespace harlys_dogs_left_l275_275716

-- Define the initial conditions
def initial_dogs : ℕ := 80
def adoption_percentage : ℝ := 0.40
def dogs_taken_back : ℕ := 5

-- Compute the number of dogs left
theorem harlys_dogs_left : (80 - (int((0.40 * 80).to_nat) - 5)) = 53 :=
by
  sorry

end harlys_dogs_left_l275_275716


namespace find_larger_number_l275_275104

-- Define the conditions
variables (A B : ℝ)
variable h1 : A - B = 1660
variable h2 : 0.075 * A = 0.125 * B

-- Define the theorem to prove the larger number
theorem find_larger_number (h1 : A - B = 1660) (h2 : 0.075 * A = 0.125 * B) : A = 4150 :=
by
  -- Proof will go here
  sorry

end find_larger_number_l275_275104


namespace count_integer_triangles_with_perimeter_12_l275_275722

theorem count_integer_triangles_with_perimeter_12 : 
  ∃! (sides : ℕ × ℕ × ℕ), sides.1 + sides.2.1 + sides.2.2 = 12 ∧ sides.1 + sides.2.1 > sides.2.2 ∧ sides.1 + sides.2.2 > sides.2.1 ∧ sides.2.1 + sides.2.2 > sides.1 ∧
  (sides = (2, 5, 5) ∨ sides = (3, 4, 5) ∨ sides = (4, 4, 4)) :=
by 
  exists 3
  sorry

end count_integer_triangles_with_perimeter_12_l275_275722


namespace journey_speed_l275_275208

theorem journey_speed
  (v : ℝ) -- Speed during the first four hours
  (total_distance : ℝ) (total_time : ℝ) -- Total distance and time of the journey
  (distance_part1 : ℝ) (time_part1 : ℝ) -- Distance and time for the first part of journey
  (distance_part2 : ℝ) (time_part2 : ℝ) -- Distance and time for the second part of journey
  (speed_part2 : ℝ) : -- Speed during the second part of journey
  total_distance = 24 ∧ total_time = 8 ∧ speed_part2 = 2 ∧ 
  time_part1 = 4 ∧ time_part2 = 4 ∧ 
  distance_part1 = v * time_part1 ∧ distance_part2 = speed_part2 * time_part2 →
  v = 4 := 
by
  sorry

end journey_speed_l275_275208


namespace packs_per_box_l275_275599

theorem packs_per_box (total_cost : ℝ) (num_boxes : ℕ) (cost_per_pack : ℝ) 
  (num_tissues_per_pack : ℕ) (cost_per_tissue : ℝ) (total_packs : ℕ) :
  total_cost = 1000 ∧ num_boxes = 10 ∧ cost_per_pack = num_tissues_per_pack * cost_per_tissue ∧ 
  num_tissues_per_pack = 100 ∧ cost_per_tissue = 0.05 ∧ total_packs * cost_per_pack = total_cost / num_boxes →
  total_packs = 20 :=
by
  sorry

end packs_per_box_l275_275599


namespace mixture_percentage_x_l275_275553

-- Define the initial conditions and percentages
def volume_total := 100
def percent_a_in_x := 0.10
def percent_a_in_y := 0.20
def desired_percent_a := 0.12

-- New variables for x and y that satisfy the mixture conditions
variables (x y : ℝ)

-- The Lean 4 statement to prove the condition
theorem mixture_percentage_x (hx : x + y = volume_total)
                             (ha : percent_a_in_x * x + percent_a_in_y * y = desired_percent_a * volume_total) :
                             (x / volume_total * 100 ≈ 97.78) := 
by
  sorry

end mixture_percentage_x_l275_275553


namespace solve_trig_problem_l275_275646

open Real

theorem solve_trig_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 := sorry

end solve_trig_problem_l275_275646


namespace four_digit_numbers_count_l275_275280

theorem four_digit_numbers_count : 
  ∃ (n : ℕ), n = 18 ∧ 
    ∃ (f : Fin 4 → Fin 3),
    ∀ i : Fin 4,
    (∃ d ∈ {0, 1, 2}, f i = d) ∧  -- Using digits 1, 2, 3 represented as 0, 1, 2
    (∀ i : Fin 3, ∃ j₁ j₂ : Fin 4, f j₁ = i ∧ f j₂ = i ∧ j₁ ≠ j₂) ∧  -- Each digit must be used at least once
    (∀ i j : Fin 4, i ≠ j → f i ≠ f j + 1)  -- Same digit cannot be adjacent to itself
  := sorry

end four_digit_numbers_count_l275_275280


namespace totalBasketballs_proof_l275_275441

-- Define the number of balls for Lucca and Lucien
axiom numBallsLucca : ℕ := 100
axiom percBasketballsLucca : ℕ := 10
axiom numBallsLucien : ℕ := 200
axiom percBasketballsLucien : ℕ := 20

-- Calculate the number of basketballs they each have
def basketballsLucca := (percBasketballsLucca * numBallsLucca) / 100
def basketballsLucien := (percBasketballsLucien * numBallsLucien) / 100

-- Total number of basketballs
def totalBasketballs := basketballsLucca + basketballsLucien

theorem totalBasketballs_proof : totalBasketballs = 50 := by
  sorry

end totalBasketballs_proof_l275_275441


namespace num_fixed_points_Q_le_n_l275_275056

-- Define the polynomial P with integer coefficients of degree n > 1
variables {n k : ℕ} (P : Polynomial ℤ)

-- Conditions
-- Degree of polynomial P
hypothesis (hP : P.degree > 1)

-- Positive integer k
hypothesis (hk : k > 0)

-- Define the iterated polynomial Q, with P applied k times
noncomputable def Q : Polynomial ℤ := 
(iterate k (fun T => Polynomial.comp T P)) Polynomial.X

-- Statement of the theorem
theorem num_fixed_points_Q_le_n : 
  ∀ (t : ℤ), Q.eval t = t → ∃ m, 0 ≤ m ∧ m ≤ n := sorry

end num_fixed_points_Q_le_n_l275_275056


namespace power_function_through_point_l275_275335

theorem power_function_through_point :
  (∃ a : ℝ, (∀ x : ℝ, f x = x ^ a) ∧ f 2 = sqrt 2) → (∃ a : ℝ, a = 1 / 2) :=
begin
  intros h,
  obtain ⟨a, haf, ha2⟩ := h,
  use a,
  rw [haf 2] at ha2,
  exact eq_of_pow_eq (show a ≠ 0, by linarith) ha2,
  linarith,
end

end power_function_through_point_l275_275335


namespace problem_equivalence_l275_275764

noncomputable def M : ℝ × ℝ := (0, 3)

noncomputable def line_l : ℝ → ℝ := λ x, -x + 3

noncomputable def curve_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

noncomputable def area_triangle (P A B : ℝ × ℝ) : ℝ := 
  (0.5) * abs ((A.1 - P.1) * (B.2 - P.2) - (A.2 - P.2) * (B.1 - P.1))

theorem problem_equivalence :
  ∃ (A B : ℝ × ℝ), 
  A ≠ B ∧ 
  curve_C A.1 A.2 ∧ 
  curve_C B.1 B.2 ∧ 
  line_l A.1 = A.2 ∧ 
  line_l B.1 = B.2 ∧ 
  ∀ P, curve_C P.1 P.2 → area_triangle P A B = (3 * real.sqrt 3) / 2 :=
sorry

end problem_equivalence_l275_275764


namespace clock_angle_3_15_l275_275528

def degrees_per_hour : ℝ := 360 / 12
def minute_position (minutes : ℝ) : ℝ := (minutes / 60) * 360
def hour_position (hours : ℝ) (minutes : ℝ) : ℝ := hours * degrees_per_hour + (minutes / 60) * degrees_per_hour

theorem clock_angle_3_15 : 
  let minute_hand := minute_position 15
  let hour_hand := hour_position 3 15
  (hour_hand - minute_hand) % 360 = 7.5 :=
by
  let minute_hand := minute_position 15
  let hour_hand := hour_position 3 15
  have h1 : minute_hand = 90 := by sorry
  have h2 : hour_hand = 97.5 := by sorry
  have h3 : (hour_hand - minute_hand) % 360 = 7.5 := by sorry
  exact h3

end clock_angle_3_15_l275_275528


namespace white_balls_in_bag_l275_275514

theorem white_balls_in_bag : 
  ∀ x : ℕ, (3 + 2 + x ≠ 0) → (2 : ℚ) / (3 + 2 + x) = 1 / 4 → x = 3 :=
by
  intro x
  intro h1
  intro h2
  sorry

end white_balls_in_bag_l275_275514


namespace smallest_two_digit_number_l275_275216

def is_valid (ab ba : ℕ) : Prop :=
  let product := ab * ba
  (product % 100) = 0

theorem smallest_two_digit_number : ∃ a b : ℕ, (10 ≤ 10*a + b ∧ 10*a + b ≤ 99) ∧ is_valid (10*a + b) (10*b + a) ∧ 10*a + b = 25 := 
begin
  sorry -- proof goes here
end

end smallest_two_digit_number_l275_275216


namespace distance_Denver_LosAngeles_l275_275219

/-- Define the points on the complex plane --/
def LosAngeles : ℂ := 0
def Boston : ℂ := 3200 * complex.I
def Denver : ℂ := 1200 + 1600 * complex.I

/-- The distance from Denver to Los Angeles --/
def distance_from_Denver_to_LosAngeles : ℝ :=
  complex.abs (Denver - LosAngeles)

theorem distance_Denver_LosAngeles :
  distance_from_Denver_to_LosAngeles = 2000 :=
by
  -- Place holder for the actual proof.
  sorry

end distance_Denver_LosAngeles_l275_275219


namespace yellow_jelly_beans_percentage_l275_275991

theorem yellow_jelly_beans_percentage :
  let beans_A := 32
  let beans_B := 34
  let beans_C := 36
  let beans_D := 38
  let ratio_yellow_A := 0.40
  let ratio_yellow_B := 0.30
  let ratio_yellow_C := 0.25
  let ratio_yellow_D := 0.15
  let yellow_A := (beans_A * ratio_yellow_A).round
  let yellow_B := (beans_B * ratio_yellow_B).round
  let yellow_C := (beans_C * ratio_yellow_C)
  let yellow_D := (beans_D * ratio_yellow_D).round
  let total_yellow := yellow_A + yellow_B + yellow_C + yellow_D
  let total_beans := beans_A + beans_B + beans_C + beans_D
  let ratio := (total_yellow / total_beans) * 100
  ratio ≈ 27 :=
sorry

end yellow_jelly_beans_percentage_l275_275991


namespace value_of_g_neg_quarter_l275_275702

def f (x : ℝ) : ℝ :=
if x > 0 then log (x) / log (2) else sorry -- we do not realize g(x) here since its value will be deduced in the proof

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem value_of_g_neg_quarter (g : ℝ → ℝ) (h1 : ∀ x > 0, f x = log (x) / log (2))
(h2 : is_odd_function f) :
g (-1 / 4) = 2 :=
by
sorry

end value_of_g_neg_quarter_l275_275702


namespace series_eval_l275_275645

def series_term (n : ℕ) : ℚ :=
  1 / ((2 * n - 1) * (2 * n + 1))

def series_sum (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, series_term (i + 1)

theorem series_eval :
  series_sum 100 = 100 / 201 :=
by
  sorry

end series_eval_l275_275645


namespace two_fruits_probability_l275_275234

noncomputable def prob_exactly_two_fruits : ℚ := 10 / 9

theorem two_fruits_probability :
  (∀ (f : ℕ → ℝ), (f 0 = 1/3) ∧ (f 1 = 1/3) ∧ (f 2 = 1/3) ∧
   (∃ f1 f2 f3, f1 ≠ f2 ∧ f1 ≠ f3 ∧ f2 ≠ f3 ∧
    (f1 + f2 + f3 = prob_exactly_two_fruits))) :=
sorry

end two_fruits_probability_l275_275234


namespace positive_difference_even_odd_sums_l275_275872

theorem positive_difference_even_odd_sums :
  let sum_first_n_even := λ n : ℕ, 2 * (n * (n + 1) / 2)
  let sum_first_n_odd := λ n : ℕ, n * n
  let twice_sum_even_25 := 2 * sum_first_n_even 25
  let thrice_sum_odd_20 := 3 * sum_first_n_odd 20
  abs (twice_sum_even_25 - thrice_sum_odd_20) = 100 :=
by
  let sum_first_n_even := λ n : ℕ, 2 * (n * (n + 1) / 2)
  let sum_first_n_odd := λ n : ℕ, n * n
  let twice_sum_even_25 := 2 * sum_first_n_even 25
  let thrice_sum_odd_20 := 3 * sum_first_n_odd 20
  have : twice_sum_even_25 = 1300 := by sorry
  have : thrice_sum_odd_20 = 1200 := by sorry
  show abs (twice_sum_even_25 - thrice_sum_odd_20) = 100 from by sorry

end positive_difference_even_odd_sums_l275_275872


namespace total_coins_is_twenty_l275_275895

def piles_of_quarters := 2
def piles_of_dimes := 3
def coins_per_pile := 4

theorem total_coins_is_twenty : piles_of_quarters * coins_per_pile + piles_of_dimes * coins_per_pile = 20 :=
by sorry

end total_coins_is_twenty_l275_275895


namespace correct_parentheses_modification_l275_275876

variable (a b c : ℝ)

theorem correct_parentheses_modification : a + 2b - 3c = a + (2b - 3c) :=
by
  sorry

end correct_parentheses_modification_l275_275876


namespace polynomial_derivative_inequality_l275_275414

theorem polynomial_derivative_inequality
  (P : Polynomial ℝ)
  (n : ℕ)
  (hdeg : P.degree = n)
  (hroots : ∀ r : ℝ, (r ∈ P.roots) → P.is_root r)
  : ∀ x : ℝ, (n - 1) * (P.derivative.eval x) ^ 2 ≥ n * P.eval x * P.derivative.derivative.eval x :=
sorry

end polynomial_derivative_inequality_l275_275414


namespace problem_l275_275343

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos (x - Real.pi / 2)

theorem problem 
: (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ c, c = (Real.pi / 2) ∧ f c = 0) → (T = Real.pi ∧ c = (Real.pi / 2)) :=
sorry

end problem_l275_275343


namespace f_at_quarter_pi_f_at_three_halves_pi_f_is_odd_f_is_periodic_l275_275291

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn (x y : ℝ) : f(x + y) + f(x - y) = 2 * f(x) * Real.cos(y)
axiom f_at_zero : f 0 = 0
axiom f_at_half_pi : f (Real.pi / 2) = 1

theorem f_at_quarter_pi : f (Real.pi / 4) = Real.sqrt 2 / 2 :=
by sorry

theorem f_at_three_halves_pi : f (3 * Real.pi / 2) = -1 :=
by sorry

theorem f_is_odd : ∀ x, f (-x) = -f(x) :=
by sorry

theorem f_is_periodic : ∀ x, f (x + 2 * Real.pi) = f (x) :=
by sorry

end f_at_quarter_pi_f_at_three_halves_pi_f_is_odd_f_is_periodic_l275_275291


namespace room_width_l275_275115

theorem room_width
  (length : ℝ)
  (cost_rate : ℝ)
  (total_cost : ℝ)
  (length_given : length = 5.5)
  (cost_rate_given : cost_rate = 400)
  (total_cost_given : total_cost = 8250) :
  (total_cost / cost_rate / length = 3.75) :=
by
  rw [total_cost_given, cost_rate_given, length_given]
  norm_num
  sorry

end room_width_l275_275115


namespace investment_rate_l275_275443

theorem investment_rate (x : ℝ) : 
  (∀ (n : ℝ), n > 0 → (3^n * 1500) = 13500) → 
  (112 / x = 14) → 
  x = 8 :=
by
  intros h1 h2
  sorry

end investment_rate_l275_275443


namespace minimal_area_triangle_AEG_l275_275602

theorem minimal_area_triangle_AEG (CE : ℝ) (AB : ℝ) (h1 : CE = 14) (h2 : AB > 14) :
  ∃ x : ℝ, (x > 14) → (let AEG_area := 98 in AEG_area = 98) :=
by
  use AB
  intro h
  -- Given conditions
  have area_AEG := 98
  show area_AEG = 98, from rfl
  sorry

end minimal_area_triangle_AEG_l275_275602


namespace parallelogram_sides_l275_275450

theorem parallelogram_sides (a b : ℕ): 
  (a = 3 * b) ∧ (2 * a + 2 * b = 24) → (a = 9) ∧ (b = 3) :=
by
  sorry

end parallelogram_sides_l275_275450


namespace n_gon_composite_l275_275229

theorem n_gon_composite (n : ℕ) (α : ℝ) (h1 : n ≥ 3) (h2 : is_irregular_n_gon n) (h3 : α ≠ 2 * Real.pi) (h4 : ∀ θ, θ = α → (rotate_n_gon θ n).coincide_with_itself) : composite n :=
by
  sorry

-- Definitions for the conditions
def is_irregular_n_gon (n : ℕ) : Prop := sorry
def rotate_n_gon (θ : ℝ) (n : ℕ) : n_gon := sorry
def coincide_with_itself (p : n_gon) : Prop := sorry
def composite (n : ℕ) : Prop := ¬prime n

end n_gon_composite_l275_275229


namespace apples_sold_eq_200_l275_275575

-- Define the constants and variables used in our problem
constant C : ℝ  -- Cost price of one apple
constant S : ℝ  -- Selling price of one apple
constant N : ℝ  -- Number of apples sold
constant G : ℝ := 50 * S  -- Total gain from selling apples

-- Given conditions
axiom gain_percent : (S - C) / C = 1 / 3
axiom total_gain : G = (1 / 3) * N * C

-- The statement to prove
theorem apples_sold_eq_200 : N = 200 :=
by
  -- Insert proof here
  sorry

end apples_sold_eq_200_l275_275575


namespace hyperbola_equation_proof_l275_275346

noncomputable def hyperbola_equation (y x : ℝ) (a b : ℝ) : Prop :=
  (y^2 / a^2) - (x^2 / b^2) = 1

theorem hyperbola_equation_proof : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ hyperbola_equation 1 (-3) a b ∧ sqrt(10) = a ∧ sqrt(6) = b :=
begin
  use [sqrt(10), sqrt(6)],
  split,
  { sorry }, -- a > 0
  split,
  { sorry }, -- b > 0
  split,
  { sorry }, -- hyperbola_equation 1 (-3) a b
  split,
  { refl }, -- sqrt(10) = a
  { refl }, -- sqrt(6) = b
end

end hyperbola_equation_proof_l275_275346


namespace geometric_sequence_product_l275_275087

noncomputable def q : ℝ := 
  (12 + 14 * Real.sqrt 2) / (7 * Real.sqrt 2 + 6)

noncomputable def a1 : ℝ := Real.sqrt 2

noncomputable def a (n : ℕ) : ℝ := a1 * (q ^ n)

theorem geometric_sequence_product :
  let a := λ n, a1 * (q ^ n) in
  let s1 := ∑ i in (Finset.range 5), a i in
  let s2 := ∑ i in (Finset.range 5).map ((+) 2) in
  s1 = 62 / (7 * Real.sqrt 2 - 6) ∧ s2 = 12 + 14 * Real.sqrt 2 →
  ∏ i in (Finset.range 7), a i = 2 ^ 14 := 
by
  let a := λ n, a1 * (q ^ n)
  let s1 := ∑ i in (Finset.range 5), a i
  let s2 := ∑ i in (Finset.range 5).map ((+) 2)
  assume h : s1 = 62 / (7 * Real.sqrt 2 - 6) ∧ s2 = 12 + 14 * Real.sqrt 2 
  sorry

end geometric_sequence_product_l275_275087


namespace average_of_remaining_ten_numbers_l275_275267

theorem average_of_remaining_ten_numbers
  (avg_50 : ℝ)
  (n_50 : ℝ)
  (avg_40 : ℝ)
  (n_40 : ℝ)
  (sum_50 : n_50 * avg_50 = 3800)
  (sum_40 : n_40 * avg_40 = 3200)
  (n_10 : n_50 - n_40 = 10)
  : (3800 - 3200) / 10 = 60 :=
by
  sorry

end average_of_remaining_ten_numbers_l275_275267


namespace ramon_current_age_is_26_l275_275768

-- Definitions based on the problem conditions
def loui_age : Nat := 23
def ramon_age_in_20_years (ramon_current_age : Nat) : Nat := ramon_current_age + 20
def twice_loui_age : Nat := 2 * loui_age
def ramon_condition (ramon_current_age : Nat) : Prop := ramon_age_in_20_years(ramon_current_age) = twice_loui_age

-- The theorem stating the proof problem
theorem ramon_current_age_is_26 (r : Nat) 
  (h1 : loui_age = 23) 
  (h2 : ramon_condition r) : 
  r = 26 :=
sorry

end ramon_current_age_is_26_l275_275768


namespace standard_ellipse_equation_l275_275300

def ellipse_equation (a b : ℝ) : string :=
  "x^2 / " ++ toString(a^2) ++ " + y^2 / " ++ toString(b^2) ++ " = 1"

theorem standard_ellipse_equation (xc yc : ℝ) (r : ℝ) (e : ℝ) (h_eq : r = 4) (h_ecc : e = 1 / 2) (h_circ : (xc - 1)^2 + yc^2 = r^2) :
  ellipse_equation 2 (sqrt 3) = "x^2 / 4 + y^2 / 3 = 1" := by
  sorry

end standard_ellipse_equation_l275_275300


namespace adam_tickets_left_l275_275235

def tickets_left (total_tickets : ℕ) (ticket_cost : ℕ) (total_spent : ℕ) : ℕ :=
  total_tickets - total_spent / ticket_cost

theorem adam_tickets_left :
  tickets_left 13 9 81 = 4 := 
by
  sorry

end adam_tickets_left_l275_275235


namespace smallest_n_l275_275049

theorem smallest_n (n : ℕ) (h : (17 * n - 1) % 11 = 0) : n = 2 := 
by 
    sorry

end smallest_n_l275_275049


namespace bathroom_floor_area_l275_275401

theorem bathroom_floor_area (b : ℕ) (kitchen_floor_area : ℕ) (mopping_rate : ℕ) (mopping_time : ℕ) (total_area_mopped : ℕ) :
  kitchen_floor_area = 80 →
  mopping_rate = 8 →
  mopping_time = 13 →
  total_area_mopped = mopping_rate * mopping_time →
  b = total_area_mopped - kitchen_floor_area →
  b = 24 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end bathroom_floor_area_l275_275401


namespace projection_length_constant_l275_275679

theorem projection_length_constant
  (A B C E F O D : Point) (ABC : Triangle) (is_acute : IsAcute ABC)
  (circumcenter_O : (Circumcenter ABC = O))
  (intersection_AD_BC : Intersect (LineThrough A O) (SideOf BC) D)
  (on_AB : OnSide E (SideOf AB))
  (on_AC : OnSide F (SideOf AC))
  (concyclic : Concyclic A E D F) :
  ∃ const, ∀ (A B C E F O D : Point) (ABC : Triangle) (is_acute : IsAcute ABC)
           (circumcenter_O : (Circumcenter ABC = O))
           (intersection_AD_BC : Intersect (LineThrough A O) (SideOf BC) D)
           (on_AB : OnSide E (SideOf AB))
           (on_AC : OnSide F (SideOf AC))
           (concyclic : Concyclic A E D F),
           ProjectionLength EF BC = const := by
  sorry

end projection_length_constant_l275_275679


namespace total_charge_for_trip_l275_275019

noncomputable def calc_total_charge (initial_fee : ℝ) (additional_charge : ℝ) (miles : ℝ) (increment : ℝ) :=
  initial_fee + (additional_charge * (miles / increment))

theorem total_charge_for_trip :
  calc_total_charge 2.35 0.35 3.6 (2 / 5) = 8.65 :=
by
  sorry

end total_charge_for_trip_l275_275019


namespace lambda_is_half_l275_275355

   theorem lambda_is_half
     (a b c : ℝ × ℝ)
     (λ : ℝ)
     (h_a : a = (1, 2))
     (h_b : b = (1, 0))
     (h_c : c = (3, 4))
     (h_parallel : ∃ k : ℝ, a + λ • b = k • c) :
     λ = 1 / 2 := by
   sorry
   
end lambda_is_half_l275_275355


namespace ellipse_equation_l275_275339

-- Define the given ellipse and parabola conditions
def ellipse (a b : ℝ) (x y : ℝ) := (a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1)
def parabola (x y : ℝ) := (y = x^2 / 8)
def eccentricity (a b : ℝ) := (a > b ∧ b > 0 ∧ (1 - (b^2 / a^2)) = 3/4)

-- Define the fixed points and areas
def L_parallel_to_AB (P : ℝ × ℝ) (x y : ℝ) := (P.1 - 2 * P.2 - 1 = 0)
def area_triangle_FMN (F : ℝ × ℝ) (M N : ℝ × ℝ) := (F.2 = 2 ∧ (abs (M.1 - N.1) / 5) * sqrt (8 * (M.2)^2 - 1) / 2 = 5 * sqrt 31 / 4)

-- Statement to prove in Lean
theorem ellipse_equation (a b : ℝ) (x1 x2 y1 y2 : ℝ) :
  (a > b ∧ b > 0 ∧ (1 - (b^2 / a^2)) = 3/4) →
  (abs (y1 - y2) / 5) * sqrt (8 * (b^2) - 1) / 2 = 5 * sqrt 31 / 4 →
  a = 4 ∧ b = 2 →
  ellipse 4 2 x1 y1 :=
by
  sorry

end ellipse_equation_l275_275339


namespace problem1_problem2_l275_275337

-- Problem 1: Given the sum of the first n terms of the sequence {a_n},
-- prove the general formula for a_n.
theorem problem1 (S : ℕ → ℚ) (a : ℕ → ℚ) (hS : ∀ n, S n = n^2 / 2 + n / 2) (h_init : a 1 = S 1) (h_rec : ∀ n, 1 ≤ n → a (n + 1) = S (n + 1) - S n) :
  ∀ n, 1 ≤ n → a n = n := 
sorry

-- Problem 2: Given b_1 = a_1 and (b_{n+1} / a_{n+1}) = (2b_n / a_n), 
-- prove the sum of the first n terms of {b_n} is T_n = 1 + (n-1) * 2^n.
theorem problem2 (a b : ℕ → ℚ) (T : ℕ → ℚ) (h_a : ∀ n, 1 ≤ n → a n = n) (b1_eq_a1 : b 1 = a 1) 
  (rec_formula_b : ∀ n, b (n + 1) / a (n + 1) = 2 * b n / a n) :
  (T : ℕ → ℚ), 
    (∀ n, T n = ∑ i in range n, b i) ∧
    (∀ n, 1 ≤ n → T n = 1 + (n - 1) * 2^n) :=
sorry

end problem1_problem2_l275_275337


namespace value_of_z_sub_y_add_x_l275_275623

-- Represent 312 in base 3
def base3_representation : List ℕ := [1, 0, 1, 2, 1, 0] -- 312 in base 3 is 101210

-- Define x, y, z
def x : ℕ := (base3_representation.count 0)
def y : ℕ := (base3_representation.count 1)
def z : ℕ := (base3_representation.count 2)

-- Proposition to be proved
theorem value_of_z_sub_y_add_x : z - y + x = 2 := by
  sorry

end value_of_z_sub_y_add_x_l275_275623


namespace bandi_has_winning_strategy_l275_275230

-- Definitions based on the problem conditions
def turn_sequence (n : ℕ) : Type := Fin n → Bool
def to_binary (seq : turn_sequence 4014) : ℤ := 
  (Finset.univ.sum (λ i, if seq i then 2^i else 0))

def is_sum_of_two_squares (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

def winning_strategy_for_bandi (seq : turn_sequence 4014) : Prop :=
  ¬ is_sum_of_two_squares (to_binary seq)

theorem bandi_has_winning_strategy :
  ∃ seq : turn_sequence 4014, winning_strategy_for_bandi seq :=
-- Proof would be here
sorry

end bandi_has_winning_strategy_l275_275230


namespace ratio_sheep_horses_l275_275605

theorem ratio_sheep_horses
  (horse_food_per_day : ℕ)
  (total_horse_food : ℕ)
  (number_of_sheep : ℕ)
  (number_of_horses : ℕ)
  (gcd_sheep_horses : ℕ):
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  number_of_sheep = 40 →
  number_of_horses = total_horse_food / horse_food_per_day →
  gcd number_of_sheep number_of_horses = 8 →
  (number_of_sheep / gcd_sheep_horses = 5) ∧ (number_of_horses / gcd_sheep_horses = 7) :=
by
  intros
  sorry

end ratio_sheep_horses_l275_275605


namespace increasing_intervals_decreasing_interval_l275_275846

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem increasing_intervals : 
  (∀ x, x < -1/3 → deriv f x > 0) ∧ 
  (∀ x, x > 1 → deriv f x > 0) :=
sorry

theorem decreasing_interval : 
  ∀ x, -1/3 < x ∧ x < 1 → deriv f x < 0 :=
sorry

end increasing_intervals_decreasing_interval_l275_275846


namespace tangent_line_at_x_minus1_l275_275708

noncomputable def f(x : ℝ) (a : ℝ) := a * x ^ 3 + 3 * x ^ 2 + 2
noncomputable def f_prime(x : ℝ) (a : ℝ) := (3 : ℝ) * a * x ^ 2 + 6 * x
noncomputable def tangent_line_eqn(x : ℝ) (m b : ℝ) := m * x + b

theorem tangent_line_at_x_minus1 (a : ℝ) (h1 : f_prime (-1) a = 3) :
  tangent_line_eqn (3 : ℝ) 5 = (3 * x + 5 : ℝ) := 
by {
  have h2 : f (-1) a = 2, from sorry,
  have h3 : tangent_line_eqn (3 : ℝ) 5 ∈ (λ x y : ℝ, y = 3 * x + 5),
  from sorry,
  exact sorry,
}

end tangent_line_at_x_minus1_l275_275708


namespace cake_slices_left_l275_275935

-- Setup initial conditions as definitions
def initial_cakes := 2
def slices_per_cake := 8
def slices_given_friends (total_slices : Nat) := total_slices / 4
def slices_given_family (remaining_slices : Nat) := remaining_slices / 3
def slices_eaten := 3

-- Define a theorem to prove the final number of slices
theorem cake_slices_left : 
  ∀ (initial_cakes slices_per_cake slices_eaten : Nat)
    (slices_given_friends slices_given_family : Nat → Nat),
  let total_slices := initial_cakes * slices_per_cake in
  let remaining_after_friends := total_slices - slices_given_friends(total_slices) in
  let remaining_after_family := remaining_after_friends - slices_given_family(remaining_after_friends) in
  final_slices = remaining_after_family - slices_eaten :=
by 
  intros
  sorry

end cake_slices_left_l275_275935


namespace array_entries_multiple_of_3_impossible_l275_275394

def matrix_A : matrix (fin 6) (fin 6) ℤ := ![
  ![2, 0, 1, 0, 2, 0],
  ![0, 2, 0, 1, 2, 0],
  ![1, 0, 2, 0, 2, 0],
  ![0, 1, 0, 2, 2, 0],
  ![1, 1, 1, 1, 2, 0],
  ![0, 0, 0, 0, 0, 0]
]

theorem array_entries_multiple_of_3_impossible :
  ¬ ∃ (k : ℕ) (hk : 1 < k ∧ k ≤ 6) (n : ℕ), 
    ∀ i j : fin 6, (matrix_A i j + n) % 3 = 0 :=
sorry

end array_entries_multiple_of_3_impossible_l275_275394


namespace problem_statement_l275_275292

variable (f : ℝ → ℝ) 

def prop1 (f : ℝ → ℝ) : Prop := ∃T > 0, T ≠ 3 / 2 ∧ ∀ x, f (x + T) = f x
def prop2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 4) = f (-x + 3 / 4)
def prop3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def prop4 (f : ℝ → ℝ) : Prop := Monotone f

theorem problem_statement (h₁ : ∀ x, f (x + 3 / 2) = -f x)
                          (h₂ : ∀ x, f (x - 3 / 4) = -f (-x - 3 / 4)) : 
                          (¬prop1 f) ∧ (prop2 f) ∧ (prop3 f) ∧ (¬prop4 f) :=
by
  sorry

end problem_statement_l275_275292


namespace projection_eq_l275_275982

variables 
  (u v : ℝ × ℝ) 
  (u_eq : u = ⟨3, -4⟩)
  (v_eq : v = ⟨-1, 0⟩)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def norm_sq (a : ℝ × ℝ) : ℝ :=
  a.1 * a.1 + a.2 * a.2

theorem projection_eq : 
  let proj_uv := (dot_product u v / norm_sq v) * v in
  proj_uv = ⟨3, 0⟩ :=
by
  -- proof omitted
  sorry

end projection_eq_l275_275982


namespace binom_10_1_eq_10_l275_275244

theorem binom_10_1_eq_10 : Nat.choose 10 1 = 10 := by
  sorry

end binom_10_1_eq_10_l275_275244


namespace red_higher_than_green_l275_275920

open ProbabilityTheory

noncomputable def prob_bin_k (k : ℕ) : ℝ :=
  (2:ℝ)^(-k)

noncomputable def prob_red_higher_than_green : ℝ :=
  ∑' (k : ℕ), (prob_bin_k k) * (prob_bin_k (k + 1))

theorem red_higher_than_green :
  (∑' (k : ℕ), (2:ℝ) ^ (-k) * (2:ℝ) ^(-(k + 1))) = 1/3 :=
  by
  sorry

end red_higher_than_green_l275_275920


namespace inequality_proof_l275_275689

open Real

variables (n : ℕ) (x : ℕ → ℝ)
 
axiom h1 : ∀ i, x i > 0
axiom h2 : ∑ i in finset.range n, x i = 1
axiom h3 : n ≥ 2
 
theorem inequality_proof : ∑ i in finset.range n, x i / sqrt (1 - x i) ≥ (1 / sqrt (n - 1)) * ∑ i in finset.range n, sqrt (x i) :=
sorry

end inequality_proof_l275_275689


namespace largest_circle_radius_l275_275983

theorem largest_circle_radius (a b c : ℝ) (h : a > b ∧ b > c) :
  ∃ radius : ℝ, radius = b :=
by
  sorry

end largest_circle_radius_l275_275983


namespace radical_product_l275_275156

def fourth_root (x : ℝ) : ℝ := x ^ (1/4)
def third_root (x : ℝ) : ℝ := x ^ (1/3)
def square_root (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_product :
  fourth_root 81 * third_root 27 * square_root 9 = 27 := 
by
  sorry

end radical_product_l275_275156


namespace simplify_eval_l275_275816

theorem simplify_eval (a : ℝ) (h : a = Real.sqrt 3 / 3) : (a + 1) ^ 2 + a * (1 - a) = Real.sqrt 3 + 1 := 
by
  sorry

end simplify_eval_l275_275816


namespace hexagon_angle_Q_l275_275636

theorem hexagon_angle_Q
  (a1 a2 a3 a4 a5 : ℝ)
  (h1 : a1 = 134) 
  (h2 : a2 = 98) 
  (h3 : a3 = 120) 
  (h4 : a4 = 110) 
  (h5 : a5 = 96) 
  (sum_hexagon_angles : a1 + a2 + a3 + a4 + a5 + Q = 720) : 
  Q = 162 := by {
  sorry
}

end hexagon_angle_Q_l275_275636


namespace reflect_circle_center_over_line_l275_275481

theorem reflect_circle_center_over_line : 
    let center_original := (7, -3)
    let line := λ x y, y = -x
    let reflected_center := (3, -7)
    (reflect (center_original) (line) = reflected_center) :=
begin
    sorry,
end

end reflect_circle_center_over_line_l275_275481


namespace area_of_circumcircle_ABC_is_correct_l275_275397

noncomputable def area_of_circumcircle (A B C : Type*) [euclidean_space A] [metric_space B] [has_inner C] 
  (AB AC : ℝ) (angle_A : ℝ) (hAB : AB = 3) (hAC : AC = 2) (h_angle_A : angle_A = π / 3) : ℝ :=
  let BC := real.sqrt (AB * AB + AC * AC - 2 * AB * AC * real.cos angle_A) in
  let R := BC / (2 * real.sin angle_A) in
  π * R * R

theorem area_of_circumcircle_ABC_is_correct :
  area_of_circumcircle ℝ ℝ ℝ 3 2 (π / 3) rfl rfl rfl = (7 * π) / 3 :=
by sorry

end area_of_circumcircle_ABC_is_correct_l275_275397


namespace ramon_current_age_is_26_l275_275769

-- Definitions based on the problem conditions
def loui_age : Nat := 23
def ramon_age_in_20_years (ramon_current_age : Nat) : Nat := ramon_current_age + 20
def twice_loui_age : Nat := 2 * loui_age
def ramon_condition (ramon_current_age : Nat) : Prop := ramon_age_in_20_years(ramon_current_age) = twice_loui_age

-- The theorem stating the proof problem
theorem ramon_current_age_is_26 (r : Nat) 
  (h1 : loui_age = 23) 
  (h2 : ramon_condition r) : 
  r = 26 :=
sorry

end ramon_current_age_is_26_l275_275769


namespace problem1_problem2_l275_275666

-- Define the conditions given in the problem
variables {a b : EuclideanSpace ℝ (Fin 2)} -- We assume 2-dimensional space for simplicity
variable (theta : ℝ)
variable (abs_a : ℝ)
variable (abs_b : ℝ)
variable (dot_ab : ℝ)

-- Set the values of the conditions
def conditions :=
  abs_a = 2 ∧ abs_b = 1 ∧ theta = 2 * Real.pi / 3 ∧ dot_ab = -1

-- (1) Prove that a • b = -1
theorem problem1 (h : conditions): (a • b) = -1 := 
  sorry

-- (2) Prove that |a - 2b| = 2 * sqrt 3
theorem problem2 (h : conditions): ∥ a - (2:ℝ) • b ∥ = 2 * Real.sqrt 3 := 
  sorry

end problem1_problem2_l275_275666


namespace vector_subtraction_magnitude_l275_275699

variables (a b : ℝ^3) -- assuming vectors are in 3-dimensional space

-- Given conditions
def dot_product_condition : ℝ := a ⬝ b -- ⬝ denotes dot product in Lean
def length_a : ℝ := ‖a‖ -- ‖ ‖ denotes norm (magnitude) in Lean
def length_b : ℝ := ‖b‖

theorem vector_subtraction_magnitude :
  dot_product_condition a b = 1 →
  length_a a = 2 →
  length_b b = 3 →
  ‖a - b‖ = sqrt 13 :=
by sorry

end vector_subtraction_magnitude_l275_275699


namespace point_C_lies_within_region_l275_275940

def lies_within_region (x y : ℝ) : Prop :=
  (x + y - 1 < 0) ∧ (x - y + 1 > 0)

theorem point_C_lies_within_region : lies_within_region 0 (-2) :=
by {
  -- Proof is omitted as per the instructions
  sorry
}

end point_C_lies_within_region_l275_275940


namespace flowers_in_each_row_l275_275801

theorem flowers_in_each_row (rows : ℕ) (total_remaining_flowers : ℕ) 
  (percentage_remaining : ℚ) (correct_rows : rows = 50) 
  (correct_remaining : total_remaining_flowers = 8000) 
  (correct_percentage : percentage_remaining = 0.40) :
  (total_remaining_flowers : ℚ) / percentage_remaining / (rows : ℚ) = 400 := 
by {
 sorry
}

end flowers_in_each_row_l275_275801


namespace limit_of_sequence_l275_275562

open Filter Real

noncomputable def a_n (n : ℕ) : ℝ := (2 - 2 * n) / (3 + 4 * n)
noncomputable def a : ℝ := -1 / 2

theorem limit_of_sequence :
  Tendsto (λ n : ℕ, a_n n) atTop (𝓝 a) := by
  sorry

end limit_of_sequence_l275_275562


namespace domain_g_correct_l275_275257

-- Define the function g
def g (x : ℝ) : ℝ := (5 * x + 2) / (Real.sqrt (x^2 - x - 6))

-- Define the set representing the domain of g(x)
def domain_g : Set ℝ := { x | x < -2 ∨ x > 3 }

-- State the theorem
theorem domain_g_correct : ∀ x, (x ∈ domain_g ↔ g x = (5 * x + 2) / (Real.sqrt (x^2 - x - 6))) :=
by 
  sorry

end domain_g_correct_l275_275257


namespace num_envelopes_correct_l275_275226

-- The problem conditions
variable (e_w : ℝ) (t_w : ℝ)

-- The given numerics
def envelope_weight : ℝ := 8.5
def total_weight_kg : ℝ := 7.225

-- The number of envelopes to be determined
def number_of_envelopes (e_w t_w : ℝ) : ℝ := t_w / e_w

theorem num_envelopes_correct : 
  number_of_envelopes envelope_weight (total_weight_kg * 1000) = 850 :=
by
  sorry

end num_envelopes_correct_l275_275226


namespace dominant_pairs_2016_l275_275924

noncomputable def max_dominant_pairs (n : ℕ) : ℕ :=
  -- In a round robin tournament with n participants, find the maximum number of dominant pairs
  if 2016 ≤ n then 2015 else sorry

theorem dominant_pairs_2016 : max_dominant_pairs 2016 = 2015 :=
by
  unfold max_dominant_pairs
  simp [if_pos (Nat.le_refl 2016)]
  sorry

end dominant_pairs_2016_l275_275924


namespace probability_seven_chairs_probability_n_chairs_l275_275034
-- Importing necessary library to ensure our Lean code can be built successfully

-- Definition for case where n = 7
theorem probability_seven_chairs : 
  let total_seating := 7 * 6 * 5 / 6 
  let favorable_seating := 1 
  let probability := favorable_seating / total_seating 
  probability = 1 / 35 := 
by 
  sorry

-- Definition for general case where n ≥ 6
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) : 
  let total_seating := (n - 1) * (n - 2) / 2 
  let favorable_seating := (n - 4) * (n - 5) / 2 
  let probability := favorable_seating / total_seating 
  probability = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := 
by 
  sorry

end probability_seven_chairs_probability_n_chairs_l275_275034


namespace number_of_students_earning_B_l275_275744

variables (a b c : ℕ) -- since we assume we only deal with whole numbers

-- Given conditions:
-- 1. The probability of earning an A is twice the probability of earning a B.
axiom h1 : a = 2 * b
-- 2. The probability of earning a C is equal to the probability of earning a B.
axiom h2 : c = b
-- 3. The only grades are A, B, or C and there are 45 students in the class.
axiom h3 : a + b + c = 45

-- Prove that the number of students earning a B is 11.
theorem number_of_students_earning_B : b = 11 :=
by
    sorry

end number_of_students_earning_B_l275_275744


namespace central_angle_of_sector_l275_275908

theorem central_angle_of_sector (P : ℝ) (x : ℝ) (h : P = 1 / 8) : x = 45 :=
by
  sorry

end central_angle_of_sector_l275_275908


namespace expenses_categorization_l275_275632

-- Define the sets of expenses and their categories
inductive Expense
| home_internet
| travel
| camera_rental
| domain_payment
| coffee_shop
| loan
| tax
| qualification_courses

open Expense

-- Define the conditions under which expenses can or cannot be economized
def isEconomizable : Expense → Prop
| home_internet        := true
| travel               := true
| camera_rental        := true
| domain_payment       := true
| coffee_shop          := true
| loan                 := false
| tax                  := false
| qualification_courses:= false

-- The main theorem statement
theorem expenses_categorization
  (expenses : list Expense) :
  (∀ e ∈ [home_internet, travel, camera_rental, domain_payment, coffee_shop], isEconomizable e) ∧
  (∀ e ∈ [loan, tax, qualification_courses], ¬ isEconomizable e) :=
by
  intros,
  simp [isEconomizable],
  exact ⟨
    (λ e he, by cases he; exact trivial),
    (λ e he, by cases he; tauto)⟩

end expenses_categorization_l275_275632


namespace grid_square_count_l275_275961

theorem grid_square_count :
  let width := 6
  let height := 6
  let num_1x1 := (width - 1) * (height - 1)
  let num_2x2 := (width - 2) * (height - 2)
  let num_3x3 := (width - 3) * (height - 3)
  let num_4x4 := (width - 4) * (height - 4)
  num_1x1 + num_2x2 + num_3x3 + num_4x4 = 54 :=
by
  sorry

end grid_square_count_l275_275961


namespace geometric_series_sum_l275_275609

theorem geometric_series_sum (a r : ℕ) (n : ℕ) (h₁ : a = 1) (h₂ : r = 3) (h₃ : r ^ (n - 1) = 19683) :
  (∑ k in Finset.range n, a * r ^ k) = 29524 := by
  sorry

end geometric_series_sum_l275_275609


namespace range_of_f_l275_275159

def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_of_f : set.Ioc 0 1 = set_of (λ y, ∃ x : ℝ, f x = y) :=
by
  sorry

end range_of_f_l275_275159


namespace range_of_quadratic_fn_l275_275566

noncomputable def quadratic_fn (x : ℝ) : ℝ :=
  x^2 - 4 * x + 3

theorem range_of_quadratic_fn :
  set.range (quadratic_fn ∘ (λ x, x)) ⊆ set.Icc (-1 : ℝ) 3 :=
by sorry

end range_of_quadratic_fn_l275_275566


namespace max_black_cells_in_grid_l275_275386

theorem max_black_cells_in_grid (n : ℕ) (grid : list (list bool)) (h1 : ∀ i j, i < n → j < n → grid.length = n → ∀ k, grid.nth i = some ⟦k⟧ → k.length = n) :
  (∀ i j, i < n → j < n → grid[i][j] = tt → 
    (if i > 0 then grid[i-1][j] = tt else true) +
    (if i < n - 1 then grid[i+1][j] = tt else true) +
    (if j > 0 then grid[i][j-1] = tt else true) +
    (if j < n - 1 then grid[i][j+1] = tt else true) ≤ 1) →
  (∑ i in range n, ∑ j in range n, if grid[i][j] then 1 else 0) ≤ 8 :=
sorry

end max_black_cells_in_grid_l275_275386


namespace units_digit_quotient_4_l275_275968

theorem units_digit_quotient_4 (n : ℕ) (h₁ : n ≥ 1) :
  (5^1994 + 6^1994) % 10 = 1 ∧ (5^1994 + 6^1994) % 7 = 5 → 
  (5^1994 + 6^1994) / 7 % 10 = 4 := 
sorry

end units_digit_quotient_4_l275_275968


namespace maximum_pieces_initially_used_l275_275169

theorem maximum_pieces_initially_used
  (w h n m : ℕ)
  (h_board_size : w = 19 ∧ h = 19)
  (h_valid_dimensions : n * m = 45)
  (h_new_rectangle : ∃ w_fixed : ℕ, w_fixed * h_orig = n ∧ w_fixed * (h_orig + 45 / w_fixed) = m)
  : max_pieces_on_board w h = 285 :=
sorry

end maximum_pieces_initially_used_l275_275169


namespace probability_red_higher_than_green_l275_275919

theorem probability_red_higher_than_green :
  let P (k : ℕ) := 2^(-k)
  in (∑' (k : ℕ), P k * P k) = (1 : ℝ) / 3 :=
by
  sorry

end probability_red_higher_than_green_l275_275919


namespace probability_correct_digit_l275_275547

theorem probability_correct_digit :
  let digits := Fin 10
  let correct_digit := 1 / 10
  correct_digit = (1 : ℝ) / (digits.card : ℝ) :=
sorry

end probability_correct_digit_l275_275547


namespace probability_empty_chair_on_sides_7_chairs_l275_275045

theorem probability_empty_chair_on_sides_7_chairs :
  (1 : ℚ) / (35 : ℚ) = 0.2 := by
  sorry

end probability_empty_chair_on_sides_7_chairs_l275_275045


namespace Billy_has_10_fish_l275_275608

def Billy_has_fish (Bobby Sarah Tony Billy : ℕ) : Prop :=
  Bobby = 2 * Sarah ∧
  Sarah = Tony + 5 ∧
  Tony = 3 * Billy ∧
  Bobby + Sarah + Tony + Billy = 145

theorem Billy_has_10_fish : ∃ (Billy : ℕ), Billy_has_fish (2 * (3 * Billy + 5)) (3 * Billy + 5) (3 * Billy) Billy ∧ Billy = 10 :=
by
  sorry

end Billy_has_10_fish_l275_275608


namespace greatest_sum_consecutive_lt_400_l275_275869

noncomputable def greatest_sum_of_consecutive_integers (n : ℤ) : ℤ :=
if n * (n + 1) < 400 then n + (n + 1) else 0

theorem greatest_sum_consecutive_lt_400 : ∃ n : ℤ, n * (n + 1) < 400 ∧ greatest_sum_of_consecutive_integers n = 39 :=
by
  sorry

end greatest_sum_consecutive_lt_400_l275_275869


namespace power_function_at_nine_l275_275838

theorem power_function_at_nine :
  (∀ (a : ℝ) (log_fun : ℝ → ℝ), (∀ x, log_fun x = log a (2 * x - 3) + 4) → (log_fun 2 = 4)) →
  (∃ α, ∀ x, f x = x^α) →
  f 9 = 81 := sorry

end power_function_at_nine_l275_275838


namespace at_most_one_obtuse_angle_l275_275142

-- Definition of obtuse angles and the property of triangles.
def obtuse_angle (a : ℝ) : Prop := a > 90
def interior_angles (a b c : ℝ) : Prop := a + b + c = 180

-- Prove the main proposition using contradiction.
theorem at_most_one_obtuse_angle (a b c : ℝ) (h_angles : interior_angles a b c) :
  (obtuse_angle a ∧ obtuse_angle b) → false :=
begin
  sorry
end

end at_most_one_obtuse_angle_l275_275142


namespace lattice_points_count_l275_275578

/-- A lattice point is a point whose coordinates are both integers. -/
def is_lattice_point (p : ℤ × ℤ) : Prop := true

/-- Define the region bounded by the curves y = x^2 and y = 4 - x^2. -/
def in_region (p : ℤ × ℤ) : Prop :=
  let x := p.1 in
  let y := p.2 in
  (y ≥ x^2) ∧ (y ≤ 4 - x^2)

/-- The number of lattice points on the boundary or inside the given region is 11. -/
theorem lattice_points_count : 
  let lattice_points := { p : ℤ × ℤ | is_lattice_point p ∧ in_region p } in
  lattice_points.to_finset.card = 11 := sorry

end lattice_points_count_l275_275578


namespace largest_angle_is_85_l275_275496

-- Define the three angles of the triangle
def angle1 : ℝ := 40
def angle2 : ℝ := 85
def angle3 : ℝ := 55

-- Use the angle sum property of a triangle
axiom angle_sum_property {a b c : ℝ} (h : a + b + c = 180) : a + b + c = 180

-- Lean statement to prove the largest angle
theorem largest_angle_is_85 (h : angle_sum_property (angle1 + angle2 + angle3) 180) : 
  max angle1 (max angle2 angle3) = angle2 :=
by
  sorry

end largest_angle_is_85_l275_275496


namespace probability_of_A_l275_275650

variable (A B : Prop)
variable (P : Prop → ℝ)

-- Given conditions
variable (h1 : P (A ∧ B) = 0.72)
variable (h2 : P (A ∧ ¬B) = 0.18)

theorem probability_of_A: P A = 0.90 := sorry

end probability_of_A_l275_275650


namespace carlos_brother_birthday_and_age_l275_275806

-- Define the problem conditions as hypotheses
def carlos_birthday := "2007-03-13"  -- assuming this denotes March 13, 2007
def day_of_week (date : String) : String :=  -- assuming a predefined function to get the day
  if date = "2007-03-13" then "Tuesday" else
    if date = "2012-02-17" then "Sunday" else "Unknown"  -- 2000 days from Mar 13, 2007 as per solution

-- Define the proof goal
theorem carlos_brother_birthday_and_age :
  day_of_week (nat.add 2000 carlos_birthday) = "Sunday" ∧ carlos_age carlos_birthday + 5 = 12 :=
by sorry

end carlos_brother_birthday_and_age_l275_275806


namespace number_of_baskets_l275_275855

def apples_per_basket : ℕ := 17
def total_apples : ℕ := 629

theorem number_of_baskets : total_apples / apples_per_basket = 37 :=
  by sorry

end number_of_baskets_l275_275855


namespace alicia_tax_l275_275592

theorem alicia_tax (wage_dollars : ℝ) (tax_rate : ℝ) (wage_dollars = 25) (tax_rate = 0.02) :
  (wage_dollars * (100 : ℝ) * tax_rate) = 50 :=
by
  sorry

end alicia_tax_l275_275592


namespace restore_triangle_of_given_conditions_l275_275827

noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def angle_bisector (A B C : Point) : Line := sorry
noncomputable def meet (l : Line) (c : Circle) : Point := sorry
noncomputable def meet_second (c1 : Circle) (c2 : Circle) : Point := sorry
noncomputable def construct_triangle (A P W : Point) : Triangle := sorry

theorem restore_triangle_of_given_conditions (A P W : Point)
  (H : Point := orthocenter_of_ABC) -- the orthocenter of the triangle
  (omega : Circle := circumcircle_of_ABC) -- the circumcircle
  (s : Circle := circle_with_diameter_AH) -- circle with diameter AH
  (b1 : meet (angle_bisector A B C) omega = W) -- condition 1
  (b2 : meet_second s omega = P) -- condition 2
  (restored_triangle : Triangle := construct_triangle A P W) 
  : Triangle := 
begin
  -- Here we would restore the triangle ABC using the points A, P, and W.
  exact restored_triangle,
end

end restore_triangle_of_given_conditions_l275_275827


namespace tan_46_deg_approx_l275_275618

-- We define the radians conversion for degrees.
def deg_to_rad (deg : ℝ) : ℝ := deg * (Real.pi / 180)

-- Define given conditions.
def x_deg : ℝ := 45
def x_rad : ℝ := deg_to_rad x_deg

def delta_x_deg : ℝ := 1
def delta_x_rad : ℝ := deg_to_rad delta_x_deg

-- Define the function tan and its differential.
noncomputable def tan (x : ℝ) : ℝ := Real.tan x
noncomputable def sec (x : ℝ) : ℝ := 1 / (Real.cos x)
noncomputable def sec_squared (x : ℝ) : ℝ := sec x * sec x

-- Perform the differential approximation.
noncomputable def dy : ℝ := (sec_squared x_rad) * delta_x_rad

-- Define the approximate value of tan(46°).
noncomputable def tan_46_approx : ℝ := tan x_rad + dy

-- The statement we want to prove.
theorem tan_46_deg_approx : tan_46_approx ≈ 1.0350 := by
  sorry

end tan_46_deg_approx_l275_275618


namespace analytical_expression_and_minimum_value_A1_minimum_value_3_implies_a_value_l275_275288

noncomputable def f (x a : ℝ) : ℝ := x^2 + x + a^2 + a
noncomputable def g (x a : ℝ) : ℝ := x^2 - x + a^2 - a

def M (x a : ℝ) : ℝ := max (f x a) (g x a)

theorem analytical_expression_and_minimum_value_A1 :
  (∀ x : ℝ, a = 1 → M x a = if x ≥ -1 then x^2 + x + 2 else x^2 - x) ∧
  (∀ x : ℝ, a = 1 → M x a ≥ x^2 + x + 2 ∧ M x a ≥ x^2 - x ∧ (∀ x : ℝ, M x a ≥ 7 / 4)) :=
sorry

theorem minimum_value_3_implies_a_value (a : ℝ) :
  (∀ x : ℝ, M x a ≥ 3) → (a = (real.sqrt 14 - 1) / 2 ∨ a = -(real.sqrt 14 - 1) / 2) :=
sorry

end analytical_expression_and_minimum_value_A1_minimum_value_3_implies_a_value_l275_275288


namespace triangle_segments_l275_275546

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_segments (a : ℕ) (h : a > 0) :
  ¬ triangle_inequality 1 2 3 ∧
  ¬ triangle_inequality 4 5 10 ∧
  triangle_inequality 5 10 13 ∧
  ¬ triangle_inequality (2 * a) (3 * a) (6 * a) :=
by
  -- Proof goes here
  sorry

end triangle_segments_l275_275546


namespace water_depth_when_upright_l275_275192

-- Define the parameters of the cylindrical tank
variables (h : ℝ) (diameter : ℝ) (horizontal_depth : ℝ) (upright_depth : ℝ)

-- Given conditions
constant height_condition : h = 20
constant diameter_condition : diameter = 5
constant horizontal_depth_condition : horizontal_depth = 4

-- The expected result
constant upright_depth_result : upright_depth = 8.1

-- The statement to be proved
theorem water_depth_when_upright : h = 20 ∧ diameter = 5 ∧ horizontal_depth = 4 → upright_depth = 8.1
:= by
  intros hc dc hdc
  exact upright_depth_result

end water_depth_when_upright_l275_275192


namespace efficiency_ratio_l275_275883

variables {p_0 V_0 : ℝ} (h1 : A_{1234} = p_0 * V_0)
                         (h2 : Q_{1234} = 6.5 * p_0 * V_0)
                         (h3 : A_{134} = 0.5 * p_0 * V_0)
                         (h4 : Q_{134} = 4.5 * p_0 * V_0)

noncomputable def efficiency (A Q : ℝ) : ℝ := A / Q

theorem efficiency_ratio :
  efficiency A_{1234} Q_{1234} / efficiency A_{134} Q_{134} = 18 / 13 :=
by
  rw [efficiency, h1, h2, div_div_div_cancel_right (((p_0 : ℝ) * V_0) : ℝ), efficiency, h3, h4]
  assume (hpV0 : ((p_0 : ℝ) * V_0) ≠ 0)
  rw [div_div_eq_mul_div, div_eq_mul_inv 6.5, div_eq_mul_inv 4.5]
  field_simp
  ring
  sorry

end efficiency_ratio_l275_275883


namespace sophia_car_rental_cost_l275_275097

variable (daily_cost : ℕ) (mileage_cost : ℕ) (days : ℕ) (miles : ℕ)

theorem sophia_car_rental_cost :
  daily_cost = 30 →
  mileage_cost = 0.25 →
  days = 5 →
  miles = 500 →
  daily_cost * days + mileage_cost * miles = 275 := by
  intros
  sorry

end sophia_car_rental_cost_l275_275097


namespace abs_ineq_subs_ineq_l275_275898

-- Problem 1
theorem abs_ineq (x : ℝ) : -2 ≤ x ∧ x ≤ 2 ↔ |x - 1| + |x + 1| ≤ 4 := 
sorry

-- Problem 2
theorem subs_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a + b + c := 
sorry

end abs_ineq_subs_ineq_l275_275898


namespace factorial_sum_mod_30_l275_275637

theorem factorial_sum_mod_30 :
  (finset.range 101).sum (λ n, n.factorial) % 30 = 3 :=
sorry

end factorial_sum_mod_30_l275_275637


namespace find_a_find_solution_set_l275_275068

-- Given conditions and definitions
def f (x : ℝ) (a : ℝ) := a * Real.exp (x - 1)
def g (x : ℝ) (a : ℝ) := if x < 2 then f x a else Real.log (x - 1) / Real.log 3

-- Conditions
def cond_f_1 := f (-1) 2 = 2 / Real.exp 2

-- Questions rephrased as theorem statements
theorem find_a : ∃ a, f (-1) a = 2 / Real.exp 2 := 
by {
  use 2,
  exact cond_f_1
} 

theorem find_solution_set (a : ℝ) (h : ∃ a, f (-1) a = 2 / Real.exp 2) : 
  {x : ℝ | g x a < 2} = {x | x < 1} ∪ {x | 1 < x ∧ x < 10} := 
by {
  -- Given that a = 2 from the previous proof
  use a,
  sorry
}

end find_a_find_solution_set_l275_275068


namespace range_of_a_l275_275329

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x ∈ Ioo (-1 : ℝ) 1, f x = 3 * x + Real.sin x) 
  (h2 : f (1 - a^2) + f (1 - a) < 0) : 1 < a ∧ a < Real.sqrt 2 :=
sorry

end range_of_a_l275_275329


namespace find_hoodie_cost_l275_275960

variables (H : ℝ)

-- Given conditions
-- 1. Flashlight costs 20% of the hoodie's price.
-- 2. Boots cost 10% off $110.
-- 3. Total spent is $195.
def hoodie_cost (H : ℝ) : Prop :=
  let flashlight := 0.20 * H in
  let boots := 110 * 0.90 in
  boots + flashlight + H = 195

-- Proof statement
theorem find_hoodie_cost (H : ℝ) (h : hoodie_cost H) : H = 80 :=
by sorry

end find_hoodie_cost_l275_275960


namespace num_ways_choose_officers_8_l275_275000

def numWaysToChooseOfficers (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem num_ways_choose_officers_8 : numWaysToChooseOfficers 8 = 336 := by
  sorry

end num_ways_choose_officers_8_l275_275000


namespace linear_correlation_coefficient_chi_square_test_l275_275829

noncomputable def sales_data : List (ℕ × ℝ) :=
  [(2016, 1.00), (2017, 1.40), (2018, 1.70), (2019, 1.90), (2020, 2.00)]

noncomputable def survey_data :=
  { male_traditional := 36,
    male_new_energy := 12,
    female_traditional := 4,
    female_new_energy := 8,
    total := 60 }

noncomputable def sum_y_squared_diff := 0.66
noncomputable def sum_xy_diff := 2.5
noncomputable def sqrt_6_6 := 2.6

theorem linear_correlation_coefficient :
  let x_bar := 2018
  let y_bar := 1.4
  let sum_x_squared_diff := 10
  let r := sum_xy_diff / (sqrt 10 * sqrt sum_y_squared_diff)
  r = 0.96 :=
by
  let x_bar := 2018
  let y_bar := 1.4
  let sum_x_squared_diff := 10
  let r := 2.5 / (sqrt 10 * sqrt 0.66)
  have h : r = 2.5 / 2.6 := by sorry
  rw h
  simp
  exact rfl

theorem chi_square_test :
  let n := 60
  let a := 36
  let b := 4
  let c := 12
  let d := 8
  let chi_square := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  chi_square = 7.5 :=
by
  let n := 60
  let a := 36
  let b := 4
  let c := 12
  let d := 8
  let ad_bc := a * d - b * c
  let chi_square := (n * (ad_bc)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  have h : chi_square = (60 * (36 * 8 - 4 * 12)^2) / (40 * 20 * 48 * 12) := by sorry
  rw h
  norm_num
  exact rfl

end linear_correlation_coefficient_chi_square_test_l275_275829


namespace gcd_factorial_8_10_l275_275658

-- Define the concept of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n 

-- Statement of the problem 
theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := 
by
  sorry

end gcd_factorial_8_10_l275_275658


namespace categorization_proof_l275_275634

-- Definitions of expenses
inductive ExpenseType
| Fixed
| Variable

-- Expense data structure
structure Expense where
  name : String
  type : ExpenseType

-- List of expenses
def fixed_expenses : List Expense :=
  [ { name := "Utility payments", type := ExpenseType.Fixed },
    { name := "Loan payments", type := ExpenseType.Fixed },
    { name := "Taxes", type := ExpenseType.Fixed } ]

def variable_expenses : List Expense :=
  [ { name := "Entertainment", type := ExpenseType.Variable },
    { name := "Travel", type := ExpenseType.Variable },
    { name := "Purchasing non-essential items", type := ExpenseType.Variable },
    { name := "Renting professional video cameras", type := ExpenseType.Variable },
    { name := "Maintenance of the blog", type := ExpenseType.Variable } ]

-- Expenses that can be economized
def economizable_expenses : List String :=
  [ "Payment for home internet and internet traffic",
    "Travel expenses",
    "Renting professional video cameras for a year",
    "Domain payment for blog maintenance",
    "Visiting coffee shops (4 times a month)" ]

-- Expenses that cannot be economized
def non_economizable_expenses : List String :=
  [ "Loan payments",
    "Tax payments",
    "Courses for qualification improvement in blogger school (onsite training)" ]

-- Additional expenses and economizing suggestions
def additional_expenses : List String :=
  [ "Professional development workshops",
    "Marketing and advertising costs",
    "Office supplies",
    "Subscription services" ]

-- Lean statement for the problem
theorem categorization_proof :
  (∀ exp ∈ fixed_expenses, exp.name ∈ non_economizable_expenses) ∧
  (∀ exp ∈ variable_expenses, exp.name ∈ economizable_expenses) ∧
  (∃ exp ∈ additional_expenses, true) :=
by
  sorry

end categorization_proof_l275_275634


namespace lea_buttons_l275_275411

theorem lea_buttons (Mari_buttons : ℕ) (Kendra_buttons : ℕ) (Sue_buttons : ℕ) (Will_buttons : ℕ) :
  Mari_buttons = 8 →
  Kendra_buttons = 5 * Mari_buttons + 4 →
  Sue_buttons = Kendra_buttons / 2 →
  Will_buttons = 2.5 * (Kendra_buttons + Sue_buttons) → 
  (Mari_buttons = 8) →
  (Lea_buttons = Will_buttons - 0.2 * Will_buttons) →
  Lea_buttons = 132 := by
  sorry

end lea_buttons_l275_275411


namespace smallest_weights_to_measure_1000_l275_275533

theorem smallest_weights_to_measure_1000 : 
  ∃ k : ℕ, (∃ f : ℕ → ℤ, (∀ n : ℕ, n ≤ 1000 → f n ∈ {-1, 0, 1}) ∧ 3^k ≥ 1000) ∧ k = 7 :=
begin
  sorry
end

end smallest_weights_to_measure_1000_l275_275533


namespace number_of_jeans_l275_275745

theorem number_of_jeans :
  ∀ (shirt_cost hat_cost jeans_cost total_cost num_shirts num_hats : ℕ),
  shirt_cost = 5 →
  hat_cost = 4 →
  jeans_cost = 10 →
  num_shirts = 3 →
  num_hats = 4 →
  total_cost = 51 →
  let shirt_total := shirt_cost * num_shirts in
  let hat_total := hat_cost * num_hats in
  let remaining_cost := total_cost - (shirt_total + hat_total) in
  remaining_cost / jeans_cost = 2 := 
begin
  intros,
  sorry -- proof goes here
end

end number_of_jeans_l275_275745


namespace base7_to_base10_l275_275868

theorem base7_to_base10 (a b c d e : ℕ) (h : 45321 = a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0)
  (ha : a = 4) (hb : b = 5) (hc : c = 3) (hd : d = 2) (he : e = 1) : 
  a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0 = 11481 := 
by 
  sorry

end base7_to_base10_l275_275868


namespace units_digit_of_product_l275_275536

-- Definitions for units digit patterns for powers of 5 and 7
def units_digit (n : ℕ) : ℕ := n % 10

def power5_units_digit := 5
def power7_units_cycle := [7, 9, 3, 1]

-- Statement of the problem
theorem units_digit_of_product :
  units_digit ((5 ^ 3) * (7 ^ 52)) = 5 :=
by
  sorry

end units_digit_of_product_l275_275536


namespace prove_KT_perp_BL_l275_275412

noncomputable def triangle (A B C : Type) := {
  BL: A → A → A,
  AD: A → A → A,
  T : A,
  K : A,
  L : A,
  B : A,
  C : A
}

def is_midpoint (M K L : Type) := M = (K + L) / 2

noncomputable def problem_statement (A B C : Type) [triangle A B C] (BL AD K T L : A) :=
  is_midpoint (AD K L) ∧
  ∃ (T: triangle.BL), T = K ∧
  ∃ (AD ∘ K), AD ∘ K → ⊥

theorem prove_KT_perp_BL
  (A B C : Type) [triangle A B C]
  (T K L : A)
  (h : problem_statement A B C triangle.BL triangle.AD K T L) :
  KT ⊥ BL := 
sorry

end prove_KT_perp_BL_l275_275412


namespace range_of_ab_l275_275422

noncomputable def f (x : ℝ) : ℝ := abs (2 - x^2)

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) : 0 < a * b ∧ a * b < 2 :=
by
  sorry

end range_of_ab_l275_275422


namespace sequence_sum_after_6_steps_l275_275952

noncomputable def sequence_sum (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else if n = 1 then 3
  else if n = 2 then 15
  else if n = 3 then 1435 -- would define how numbers sequence works recursively.
  else sorry -- next steps up to 6
  

theorem sequence_sum_after_6_steps : sequence_sum 6 = 191 := 
by
  sorry

end sequence_sum_after_6_steps_l275_275952


namespace domain_of_log_l275_275106

-- Define the logarithmic function
def log_base3 (z : ℝ) : ℝ := Real.log z / Real.log 3

-- Define the domain condition for the function
def domain_condition (x : ℝ) : Prop := 3 * x - 2 > 0

-- State the property we need to prove
theorem domain_of_log : ∀ x : ℝ, domain_condition x ↔ x > 2 / 3 := by
  sorry

end domain_of_log_l275_275106


namespace evaluate_g_ggg_neg1_l275_275423

def g (y : ℤ) : ℤ := y^3 - 3*y + 1

theorem evaluate_g_ggg_neg1 : g (g (g (-1))) = 6803 := 
by
  sorry

end evaluate_g_ggg_neg1_l275_275423


namespace max_collection_l275_275886

theorem max_collection : 
  let Yoongi := 4 
  let Jungkook := 6 / 3 
  let Yuna := 5 
  max Yoongi (max Jungkook Yuna) = 5 :=
by 
  let Yoongi := 4
  let Jungkook := (6 / 3) 
  let Yuna := 5
  show max Yoongi (max Jungkook Yuna) = 5
  sorry

end max_collection_l275_275886


namespace trapezoid_base_ratio_l275_275279

-- Define the context of the problem
variables (AB CD : ℝ) (h : AB < CD)

-- Define the main theorem to be proved
theorem trapezoid_base_ratio (h : AB / CD = 1 / 2) :
  ∃ (E F G H I J : ℝ), 
    EJ - EI = FI - FH / 5 ∧ -- These points create segments that divide equally as per the conditions 
    FI - FH = GH / 5 ∧
    GH - GI = HI / 5 ∧
    HI - HJ = JI / 5 ∧
    JI - JE = EJ / 5 :=
sorry

end trapezoid_base_ratio_l275_275279


namespace simplify_complex_fraction_l275_275686

theorem simplify_complex_fraction :
  let i := Complex.I in
  (2 - i) / (1 + i) = (1 / 2) - (3 / 2) * i :=
by
  sorry

end simplify_complex_fraction_l275_275686


namespace problem1_problem2_l275_275613

-- Define Problem 1 statement
theorem problem1 : 
  (\sqrt 75 + \sqrt 27 - \sqrt (1/2) * \sqrt 12 + \sqrt 24 = 8 * \sqrt 3 + \sqrt 6) :=
sorry

-- Define Problem 2 statement
theorem problem2 : 
  ((\sqrt 3 + \sqrt 2) * (\sqrt 3 - \sqrt 2) - (\sqrt 5 - 1)^2 = 2 * \sqrt 5 - 5) :=
sorry

end problem1_problem2_l275_275613


namespace problem_1_problem_2_l275_275563

-- Proof Problem for Question 1
theorem problem_1 (a b c d : ℝ) (ha : a = 8^(2/3)) 
  (hb : b = (-7/8)^0) (hc : c = (3-π)^4^(1/4)) (hd : d = ((-2)^6)^(1/2)) :
  a - b + c + d = π + 8 :=
  sorry

-- Proof Problem for Question 2
theorem problem_2 (x : ℝ) :
  (∃ x, (2*x - 1)/(3 - 4*x) ≥ 1 ↔ x ∈ set.Icc (2/3 : ℝ) (3/4 : ℝ)) :=
  sorry

end problem_1_problem_2_l275_275563


namespace complex_point_in_second_quadrant_l275_275734

def complex_quadrant : Prop :=
  let z := complex.i * (1 + complex.i)
  in z.re = -1 ∧ z.im = 1 ∧ -1 ≤ 0 ∧ 1 ≥ 0 → (1 = 0)

theorem complex_point_in_second_quadrant : complex_quadrant :=
sorry

end complex_point_in_second_quadrant_l275_275734


namespace probability_of_selecting_product_not_less_than_4_l275_275456

theorem probability_of_selecting_product_not_less_than_4 :
  let total_products := 5 
  let favorable_outcomes := 2 
  (favorable_outcomes : ℚ) / total_products = 2 / 5 := 
by 
  sorry

end probability_of_selecting_product_not_less_than_4_l275_275456


namespace existence_of_E_l275_275299

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 6) + (y^2 / 2) = 1

def point_on_x_axis (E : ℝ × ℝ) : Prop := E.snd = 0

def ea_dot_eb_constant (E A B : ℝ × ℝ) : ℝ :=
  let ea := (A.fst - E.fst, A.snd)
  let eb := (B.fst - E.fst, B.snd)
  ea.fst * eb.fst + ea.snd * eb.snd

noncomputable def E : ℝ × ℝ := (7/3, 0)

noncomputable def const_value : ℝ := (-5/9)

theorem existence_of_E :
  (∃ E, point_on_x_axis E ∧
        (∀ A B, ellipse_eq A.fst A.snd ∧ ellipse_eq B.fst B.snd →
                  ea_dot_eb_constant E A B = const_value)) :=
  sorry

end existence_of_E_l275_275299


namespace values_of_symbols_l275_275180

-- Define the three variables
variables (☆ ○ ◎ : ℕ)

-- Given conditions
axiom h1 : ☆ + ◎ = 46
axiom h2 : ☆ + ○ = 91
axiom h3 : ○ + ◎ = 63

-- Theorem statement with the required values
theorem values_of_symbols : 
  ☆ = 37 ∧ ○ = 54 ∧ ◎ = 9 :=
sorry

end values_of_symbols_l275_275180


namespace pump_B_rate_l275_275951

noncomputable def rate_A := 1 / 2
noncomputable def rate_C := 1 / 6

theorem pump_B_rate :
  ∃ B : ℝ, (rate_A + B - rate_C = 4 / 3) ∧ (B = 1) := by
  sorry

end pump_B_rate_l275_275951


namespace domain_of_f_x_l275_275692

theorem domain_of_f_x (f : ℝ → ℝ) : (∀ x, x ∈ [0, 3] → ∃ y, y = f(x^2 - 1)) → (∀ x, x ∈ [-1, 8] → ∃ y, y = f(x)) :=
by 
  sorry

end domain_of_f_x_l275_275692


namespace John_reads_50_pages_per_hour_l275_275408

noncomputable def pages_per_hour (reads_daily hours : ℕ) (total_pages total_weeks : ℕ) : ℕ :=
  let days := total_weeks * 7
  let pages_per_day := total_pages / days
  pages_per_day / reads_daily

theorem John_reads_50_pages_per_hour :
  pages_per_hour 2 2800 4 = 50 := by
  sorry

end John_reads_50_pages_per_hour_l275_275408


namespace symmetric_line_passes_through_point_l275_275344

theorem symmetric_line_passes_through_point (k : ℝ) : 
  (∀ x y : ℝ, k * x + y - k + 1 = 0 → (x = 1 ∧ y = -1)) →
  (∀ a b : ℝ, (a + 1) / 2 + (b - 1) / 2 = 1 → b + 1 = a - 1 → (a, b) = (3, 1)) :=
begin
  intro h,
  sorry,
end

end symmetric_line_passes_through_point_l275_275344


namespace part1_part2_l275_275310

-- Part (1): Given \( f(x) = x^3 - x \), \( g(x) = x^2 + a \), and \( x_1 = -1 \), prove that \( a = 3 \).
theorem part1 (a : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x)
  (hg : ∀ x, g x = x^2 + a)
  (x1 : ℝ) (hx1 : x1 = -1)
  (tangent_match : ∀ x, deriv f x = deriv g x) : a = 3 := 
sorry

-- Part (2): Given \( f(x) = x^3 - x \) and \( g(x) = x^2 + a \), prove that the range of values for \( a \) is \( [-1, +\infty) \).
theorem part2 (a : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^3 - x)
  (hg : ∀ x, g x = x^2 + a)
  (range_of_a : ∀ t : set ℝ, ∃ u : set ℝ, ∀ x ∈ t, a ∈ u) : a ∈ set.Ici (-1) :=
sorry

end part1_part2_l275_275310


namespace statement_C_correct_l275_275167

theorem statement_C_correct (a b : ℝ) (h1 : a < b) (h2 : a * b ≠ 0) : (1 / a) > (1 / b) :=
sorry

end statement_C_correct_l275_275167


namespace arrange_digits_11250_multiple_of_5_l275_275724

theorem arrange_digits_11250_multiple_of_5 :
  let digits := [1, 1, 2, 5, 0]
  let is_multiple_of_5 (n : ℕ) := n % 5 = 0
  let arrangements := {perm | ∃ (p : List ℕ), digits.perm p ∧ digits_to_nat p = perm ∧ is_multiple_of_5 perm}
  arrangements.card = 21 := sorry

end arrange_digits_11250_multiple_of_5_l275_275724


namespace find_interest_rate_l275_275917

noncomputable def principal : ℝ := 6000
noncomputable def interest_earned : ℝ := 945.75
noncomputable def total_amount : ℝ := principal + interest_earned
noncomputable def compounding_frequency : ℕ := 2
noncomputable def time_period : ℝ := 1.5

theorem find_interest_rate (r : ℝ) (h : total_amount = principal * (1 + r / (compounding_frequency * 100))^(compounding_frequency * time_period)) :
  r ≈ 9.9 :=
sorry

end find_interest_rate_l275_275917


namespace probability_empty_chair_on_sides_7_chairs_l275_275044

theorem probability_empty_chair_on_sides_7_chairs :
  (1 : ℚ) / (35 : ℚ) = 0.2 := by
  sorry

end probability_empty_chair_on_sides_7_chairs_l275_275044


namespace tangent_line_eq_at_x2_l275_275977

open Real

-- Define the given function
def f (x : ℝ) : ℝ := x * log x + x

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 2 + log x

-- Define the value of x at which we are finding the tangent
def x_val : ℝ := 2

-- Define the point-slope form equation for the tangent line
def tangent_eq (x y : ℝ) : Prop :=
  (2 + log 2) * x - y - 2 = 0

-- Theorem stating the tangent line equation
theorem tangent_line_eq_at_x2 : tangent_eq x_val (f x_val) :=
by
  sorry


end tangent_line_eq_at_x2_l275_275977


namespace time_to_cover_length_l275_275227

/-- Define the conditions for the problem -/
def angle_deg : ℝ := 30
def escalator_speed : ℝ := 12
def length_along_incline : ℝ := 160
def person_speed : ℝ := 8

/-- Define the combined speed as the sum of the escalator speed and the person speed -/
def combined_speed : ℝ := escalator_speed + person_speed

/-- Theorem stating the time taken to cover the length of the escalator is 8 seconds -/
theorem time_to_cover_length : (length_along_incline / combined_speed) = 8 := by
  sorry

end time_to_cover_length_l275_275227


namespace find_floor_at_same_time_l275_275823

def timeTaya (n : ℕ) : ℕ := 15 * (n - 22)
def timeJenna (n : ℕ) : ℕ := 120 + 3 * (n - 22)

theorem find_floor_at_same_time (n : ℕ) : n = 32 :=
by
  -- The goal is to show that Taya and Jenna arrive at the same floor at the same time
  have ht : 15 * (n - 22) = timeTaya n := rfl
  have hj : 120 + 3 * (n - 22) = timeJenna n := rfl
  -- equate the times
  have h : timeTaya n = timeJenna n := by sorry
  -- solving the equation for n = 32
  sorry

end find_floor_at_same_time_l275_275823


namespace tan_A_value_l275_275756

-- Define the conditions provided
variables (A B C : Type) [EuclideanGeometry A] -- A, B, and C are points in Euclidean geometry
variable (ABC : Triangle A B C) -- Triangle exists with vertices A, B, C
variable (h_angle : Angle A B C = 90) -- Given angle BAC is a right angle
variable (AB AC BC : ℝ) -- Side lengths of the triangle
variable (h_AB : AB = 40) -- Given length of AB
variable (h_BC : BC = 41) -- Given length of BC

-- Define the Pythagorean theorem calculation for AC
def AC : ℝ := sqrt (BC^2 - AB^2)
-- Applying the Pythagorean theorem given the conditions
axiom h_pythagorean : AC = sqrt (41^2 - 40^2)
axiom h_AC : AC = 9

-- Definition for the tangent of angle A
def tan_A := AC / AB

-- Prove the required value of tangent
theorem tan_A_value : tan_A = 9/40 := by
  -- Introduction and calculations here
  sorry -- Proof is skipped as per instructions

end tan_A_value_l275_275756


namespace bowling_ball_weight_l275_275655

theorem bowling_ball_weight :
  ∃ b c : ℝ, (5 * b = 2 * c) ∧ (3 * c = 72) ∧ (b = 9.6) :=
by
  use 9.6, 24
  split
  { -- 5b = 2c
    norm_num }
  split
  { -- 3c = 72
    norm_num }
  -- b = 9.6
  norm_num
  sorry

end bowling_ball_weight_l275_275655


namespace calculate_expression_l275_275786

def inequality_holds (a b c d x : ℝ) : Prop :=
  (x - a) * (x - b) * (x - d) / (x - c) ≥ 0

theorem calculate_expression : 
  ∀ (a b c d : ℝ),
    a < b ∧ b < d ∧
    (∀ x : ℝ, 
      (inequality_holds a b c d x ↔ x ≤ -7 ∨ (30 ≤ x ∧ x ≤ 32))) →
    a + 2 * b + 3 * c + 4 * d = 160 :=
sorry

end calculate_expression_l275_275786


namespace newspaper_photos_count_l275_275384

theorem newspaper_photos_count :
  ∃ (total_photos : ℕ), total_photos = 208 :=
begin
  -- Define the conditions
  let pages_4_photos := 25,
  let photos_per_page_4 := 4,
  let pages_6_photos := 18,
  let photos_per_page_6 := 6,

  -- Calculate the total number of photos
  let total_photos := (pages_4_photos * photos_per_page_4) + (pages_6_photos * photos_per_page_6),
  
  -- Prove the total number of photos is 208
  use total_photos,
  sorry
end

end newspaper_photos_count_l275_275384


namespace problem_a_b_c_l275_275996

theorem problem_a_b_c (a b c : ℝ) (h1 : a < b) (h2 : b < c) (h3 : ab + bc + ac = 0) (h4 : abc = 1) : |a + b| > |c| := 
by sorry

end problem_a_b_c_l275_275996


namespace find_integer_solutions_l275_275779

-- Axiom stating that p is prime
axiom is_prime (p : ℕ) : Prop

theorem find_integer_solutions (n k p : ℕ) (hn : Int) (hk : Nat) (hp : Nat) (is_prime p) :
  (|6 * hn^2 - 17 * hn - 39| = p^k) ↔ ((hn, hp, hk) ∈ [(-1, 2, 4), (-2, 19, 1), (4, 11, 1), (2, 7, 2), (-4, 5, 3)]) :=
sorry

end find_integer_solutions_l275_275779


namespace abc_eq_4prR_ab_bc_ca_eq_r2_p2_4rR_l275_275433

variable (a b c p r R : ℝ)

def p := (a + b + c) / 2

theorem abc_eq_4prR 
  (h1 : a = BC) 
  (h2 : b = AC) 
  (h3 : c = AB) 
  (hp : p = (a + b + c) / 2) 
  (hr : r > 0) 
  (hR : R > 0) : 
  a * b * c = 4 * p * r * R := sorry

theorem ab_bc_ca_eq_r2_p2_4rR
  (h1 : a = BC) 
  (h2 : b = AC) 
  (h3 : c = AB) 
  (hp : p = (a + b + c) / 2) 
  (hr : r > 0) 
  (hR : R > 0) :
  a * b + b * c + c * a = r ^ 2 + p ^ 2 + 4 * r * R := sorry

end abc_eq_4prR_ab_bc_ca_eq_r2_p2_4rR_l275_275433


namespace lines_intersect_at_single_point_l275_275351

def line1 (a b x y: ℝ) := a * x + 2 * b * y + 3 * (a + b + 1) = 0
def line2 (a b x y: ℝ) := b * x + 2 * (a + b + 1) * y + 3 * a = 0
def line3 (a b x y: ℝ) := (a + b + 1) * x + 2 * a * y + 3 * b = 0

theorem lines_intersect_at_single_point (a b : ℝ) :
  (∃ x y : ℝ, line1 a b x y ∧ line2 a b x y ∧ line3 a b x y) ↔ a + b = -1/2 :=
by
  sorry

end lines_intersect_at_single_point_l275_275351


namespace incorrect_judgment_D_l275_275499

theorem incorrect_judgment_D (x : ℝ) (h : x = 1) :
  ¬ (2 * x / (x ^ 2 - 1) - 1 / (x + 1) = 0) :=
by {
  rw h,
  -- At x = 1, the denominator of the original fraction becomes 0, Hence it is undefined.
  have denom_eq_zero : (1 ^ 2 - 1) = 0,
    by ring,
  have zero_div_denom : ¬((2 * 1 / 0 - 1 / 2) = 0),
    by norm_num,
  exact zero_div_denom
}

end incorrect_judgment_D_l275_275499


namespace unique_intersections_l275_275492

-- Define the logarithmic functions
def f1 (x: ℝ) : ℝ := log x / log 4         -- y = log_4 x
def f2 (x: ℝ) : ℝ := log 4 / log x         -- y = log_x 4
def f3 (x: ℝ) : ℝ := log x / log (1/4)     -- y = log_{1/4} x
def f4 (x: ℝ) : ℝ := log (1/4) / log x     -- y = log_x (1/4)
def f5 (x: ℝ) : ℝ := log x / log 2         -- y = log_2 x

-- Problem statement allowing us to prove the number of intersection points
theorem unique_intersections : ∃ n : ℕ, n = 4 ∧
  ∀ x : ℝ, x > 0 →
    (f1 x = f2 x ∨ f1 x = f3 x ∨ f1 x = f4 x ∨ f1 x = f5 x ∨ 
     f2 x = f3 x ∨ f2 x = f4 x ∨ f2 x = f5 x ∨ 
     f3 x = f4 x ∨ f3 x = f5 x ∨ 
     f4 x = f5 x) → n = 4 := sorry

end unique_intersections_l275_275492


namespace amount_of_CaO_required_l275_275976

theorem amount_of_CaO_required (n_H2O : ℝ) (n_CaOH2 : ℝ) (n_CaO : ℝ) 
  (h1 : n_H2O = 2) (h2 : n_CaOH2 = 2) :
  n_CaO = 2 :=
by
  sorry

end amount_of_CaO_required_l275_275976


namespace slices_left_after_all_l275_275936

def total_cakes : ℕ := 2
def total_slices_per_cake : ℕ := 8
def fraction_given_to_friends : ℝ := 1 / 4
def fraction_given_to_family : ℝ := 1 / 3
def slices_eaten_by_alex : ℕ := 3

theorem slices_left_after_all :
  let total_slices := total_cakes * total_slices_per_cake in
  let slices_given_to_friends := (fraction_given_to_friends * total_slices) in
  let slices_left_after_friends := (total_slices - slices_given_to_friends.to_nat) in
  let slices_given_to_family := (fraction_given_to_family * slices_left_after_friends) in
  let slices_left_after_family := (slices_left_after_friends - slices_given_to_family.to_nat) in
  let slices_left_after_alex := (slices_left_after_family - slices_eaten_by_alex) in
  slices_left_after_alex = 5 :=
by
  sorry

end slices_left_after_all_l275_275936


namespace infinite_convex_polyhedra_layer_l275_275399

theorem infinite_convex_polyhedra_layer (P : Type) [convex P] [bounded P]
  (layer P : set (set P)) (infinite P) :
  (∃ (arrangement : list P), infinite (set.of_list arrangement) ∧ bounded (set.univ arrangement) ∧ 
  (∀ p ∈ arrangement, ∃ q ∈ arrangement, p ≠ q ∧ ¬(fully_removable arrangement p))) :=
sorry

end infinite_convex_polyhedra_layer_l275_275399


namespace current_value_of_business_l275_275913

theorem current_value_of_business
  (own_fraction : ℚ)
  (sale_fraction : ℚ)
  (sale_price : ℚ)
  (tax_rate : ℚ)
  (increase_rate : ℚ)
  (business_value : ℚ) :
  own_fraction = 2 / 3 →
  sale_fraction = 3 / 4 →
  sale_price = 30000 →
  tax_rate = 15 / 100 →
  increase_rate = 10 / 100 →
  let pre_tax_value := sale_price / (1 - tax_rate),
      total_shares_value := pre_tax_value / sale_fraction,
      increased_value := total_shares_value / own_fraction * (1 + increase_rate) in
    increased_value = 77647.07 := 
begin
  sorry
end

end current_value_of_business_l275_275913


namespace price_increase_percentage_l275_275243

theorem price_increase_percentage
    (original_price final_price sale_price : ℝ)
    (h1 : 0.85 * original_price = 71.4)
    (h2 : original_price - final_price = 5.25) :
    (final_price - sale_price) / sale_price = 0.102937 :=
by
  unfold original_price final_price sale_price
  sorry

end price_increase_percentage_l275_275243


namespace number_of_ways_to_arrange_students_l275_275638

theorem number_of_ways_to_arrange_students :
  let students : Finset (Fin 8) := {a | ∃ (i : Fin 4), a ∈ ({x : Fin 8 | (x : ℕ) / 2 = i}) }
  let carA := {a : Fin 8 | ... }  -- properly define the students in car A according to the problem conditions
  let carB := students \ carA
  ∃ carA (carA | ... ) (carB := carA \ ... -- exactly how you define carA and carB will depend upon how you encode constraints.
  exact (number arrangements carA == 24).

| -. sorry.

end number_of_ways_to_arrange_students_l275_275638


namespace monomial_properties_l275_275482

def coefficient (c : ℝ) (x y : ℝ) (k l : ℕ) (m : ℝ) := c = -π / 5
def degree (k l : ℕ) := k = 3 ∧ l = 2 ∧ k + l = 5

theorem monomial_properties : 
  coefficient (-π / 5) x y 3 2 (-π * x^3 * y^2 / 5) ∧ degree 3 2 := 
by 
  sorry

end monomial_properties_l275_275482


namespace area_of_right_triangle_l275_275454

theorem area_of_right_triangle (A B C D : Type*) 
  [metric_space D] [normed_space ℝ D] 
  (angle_C : ∠ A C B = π / 2)
  (AB : Real.sqrt (dist A B) = 5)
  (angle_ADC : ∠ A D C = Real.arccos (1 / Real.sqrt 10))
  (DB_len : dist D B = 4 * Real.sqrt 10 / 3) : 
  real_area ABC = 15 / 4 := 
begin
  sorry
end

end area_of_right_triangle_l275_275454


namespace john_total_distance_l275_275023

theorem john_total_distance :
  let speed1 := 35
  let time1 := 2
  let distance1 := speed1 * time1

  let speed2 := 55
  let time2 := 3
  let distance2 := speed2 * time2

  let total_distance := distance1 + distance2

  total_distance = 235 := by
    sorry

end john_total_distance_l275_275023


namespace exists_num_with_digit_sum_div_by_11_l275_275080

-- Helper function to sum the digits of a natural number
def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Main theorem statement
theorem exists_num_with_digit_sum_div_by_11 (N : ℕ) :
  ∃ k : ℕ, k < 39 ∧ (digit_sum (N + k)) % 11 = 0 :=
sorry

end exists_num_with_digit_sum_div_by_11_l275_275080


namespace greenfield_academy_math_count_l275_275947

theorem greenfield_academy_math_count (total_players taking_physics both_subjects : ℕ) 
(h_total: total_players = 30) 
(h_physics: taking_physics = 15) 
(h_both: both_subjects = 3) : 
∃ taking_math : ℕ, taking_math = 21 :=
by
  sorry

end greenfield_academy_math_count_l275_275947


namespace solution_set_inequality_l275_275120

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def is_monotone_increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

theorem solution_set_inequality (f : ℝ → ℝ) :
  is_odd_function f →
  is_monotone_increasing_on f (set.Ioi 0) →
  f 1 = 0 →
  {x : ℝ | (f x - f (-x)) / x > 0} = set.Iio (-1) ∪ set.Ioi 1 :=
by
  intros
  sorry

end solution_set_inequality_l275_275120


namespace remainder_549547_div_7_l275_275163

theorem remainder_549547_div_7 : 549547 % 7 = 5 :=
by
  sorry

end remainder_549547_div_7_l275_275163


namespace probability_empty_chair_on_sides_7_chairs_l275_275046

theorem probability_empty_chair_on_sides_7_chairs :
  (1 : ℚ) / (35 : ℚ) = 0.2 := by
  sorry

end probability_empty_chair_on_sides_7_chairs_l275_275046


namespace negation_of_proposition_l275_275565

-- Conditions
variable {x : ℝ}

-- The proposition
def proposition : Prop := ∃ x : ℝ, Real.exp x > x

-- The proof problem: proving the negation of the proposition
theorem negation_of_proposition : (¬ proposition) ↔ ∀ x : ℝ, Real.exp x ≤ x := by
  sorry

end negation_of_proposition_l275_275565


namespace sum_of_special_primes_l275_275875

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_prime (p : ℕ) : Prop :=
  p > 30 ∧ p < 99 ∧ is_prime p ∧ is_prime (reverse_digits p)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * units + tens

theorem sum_of_special_primes : 
  ( ∑ p in (Finset.filter valid_prime (Finset.range 100)) : ℕ ) = 388 :=
by sorry

end sum_of_special_primes_l275_275875


namespace volume_of_pyramid_l275_275950

-- Define the conditions
variables (a α β V : ℝ)
-- Define the conditions on angles and side length
variables (h1 : α ∈ Icc 0 π) (h2 : β ∈ Icc 0 π) (h3 : a > 0)

-- Prove the volume formula for the pyramid under the given conditions
theorem volume_of_pyramid :
  (β > π / 6 ∧ 2 * α + β ≥ π → V = (a^3 * sin(α + β) / (12 * sin α)) * sqrt(1 - 2 * cos (2 * β))) ∧
  (β ≤ π / 6 ∧ α < π / 3 ∧ α + β > π / 3 →
    V = (a^3 * sin(α + β) / (12 * sin α)) * sqrt(3 * sin^2 β - (2 * cos (2 * α + β) + cos β)^2)) :=
by
  sorry

end volume_of_pyramid_l275_275950


namespace enclosed_area_l275_275479

def parabola (y : ℝ) : ℝ := y^2
def line (y : ℝ) : ℝ := 2 * y + 3

theorem enclosed_area :
  let y₁ := -1
  let y₂ := 3 in
  (∫ y in y₁..y₂, (line y - parabola y)) = (32/3) :=
by
  sorry

end enclosed_area_l275_275479


namespace octagon_cannot_tile_with_triangles_l275_275072

def interior_angle (n : ℕ) : ℝ := 180 * (n - 2) / n

theorem octagon_cannot_tile_with_triangles :
  ¬ ∃ a b, a * interior_angle 8 + b * 60 = 360 := sorry

end octagon_cannot_tile_with_triangles_l275_275072


namespace trapezoid_midsegment_l275_275932

theorem trapezoid_midsegment (h : ℝ) :
  ∃ k : ℝ, (∃ θ : ℝ, θ = 120 ∧ k = 2 * h * Real.cos (θ / 2)) ∧
  (∃ m : ℝ, m = k / 2) ∧
  (∃ midsegment : ℝ, midsegment = m / Real.sqrt 3 ∧ midsegment = h / Real.sqrt 3) :=
by
  -- This is where the proof would go.
  sorry

end trapezoid_midsegment_l275_275932


namespace symmetrical_point_reflection_l275_275001

theorem symmetrical_point_reflection (x y : ℤ) (h : (x, y) = (-1, 2)) :
  ∃ x' y', (x', y') = (1, 2) ∧ (x', y') = (-x, y) :=
by
  use [1, 2]
  split
  { rfl }
  { rw h
    rfl }
  sorry

end symmetrical_point_reflection_l275_275001


namespace angle_between_vectors_l275_275738

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem angle_between_vectors (ha_eq_hb : ∥a∥ = ∥b∥) 
    (h2a_plus_b_dot_b : ∥2 • a + b∥ = 0) :
    real.angle a b = real.pi / 3 := sorry

end angle_between_vectors_l275_275738


namespace finite_partition_multiple_l275_275127

theorem finite_partition_multiple (S : Set ℕ) (P : Set (Set ℕ)) (hP : ∀ t ∈ P, ∀ x y ∈ t, x ≠ y → (x % y ≠ 0 ∧ y % x ≠ 0)) 
(hPartition : ∀ n ∈ S, ∃ t ∈ P, n ∈ t) (hFinite : Finite P) : 
∃ t ∈ P, ∀ n ∈ S, ∃ m ∈ t, n ∣ m :=
  sorry

end finite_partition_multiple_l275_275127


namespace salary_unspent_fraction_l275_275740

theorem salary_unspent_fraction : 
  let week1_spent := 1/4
  let week2_spent := 1/5
  let week3_spent := 1/5
  let week4_spent := 1/5
  let total_spent := week1_spent + week2_spent + week3_spent + week4_spent
  let total_salary := 1
  in total_salary - total_spent = 3/20 := 
by 
  sorry

end salary_unspent_fraction_l275_275740


namespace cannot_represent_parabola_l275_275326

noncomputable def cos (θ : ℝ) : ℝ := sorry -- definition of cosine, skipped here

variable (θ : ℝ)
variable (x y : ℝ)

-- Assumptions: θ belongs to real numbers and we use non-computable cosine function
axiom cosine_real : ∀ θ : ℝ, cos θ ∈ ℝ

theorem cannot_represent_parabola (h : x^2 + y^2 / cos θ = 4) : 
  ¬ (∃ (p : ℝ) (q : ℝ), x^2 + y^2 / cos θ  = 2 * p * x + 2 * q * y) :=
sorry

end cannot_represent_parabola_l275_275326


namespace sarahs_total_distance_correct_l275_275812

structure Journey :=
  (total_distance : ℚ)
  (mountain_path_fraction : ℚ)
  (paved_road_distance : ℚ)
  (highway_fraction : ℚ)

def sarahs_journey : Journey := {
  total_distance := 360 / 7,
  mountain_path_fraction := 1 / 4,
  paved_road_distance := 30,
  highway_fraction := 1 / 6
}

theorem sarahs_total_distance_correct :
  let x := sarahs_journey.total_distance in
  let mountain_path := sarahs_journey.mountain_path_fraction * x in
  let highway := sarahs_journey.highway_fraction * x in
  let paved_road := sarahs_journey.paved_road_distance in
  mountain_path + paved_road + highway = x :=
by {
  let x := sarahs_journey.total_distance,
  let mountain_path := sarahs_journey.mountain_path_fraction * x,
  let highway := sarahs_journey.highway_fraction * x,
  let paved_road := sarahs_journey.paved_road_distance,
  have h : mountain_path + paved_road + highway = x,
  { sorry },
  exact h
}

end sarahs_total_distance_correct_l275_275812


namespace Anoop_joins_after_7_months_l275_275172

-- Define the initial investments
def Arjun_investment : ℕ := 20000
def Anoop_investment : ℕ := 4000

-- Define the total investment time for Arjun and Anoop
def Arjun_time : ℕ := 12
def Anoop_time_join (x : ℕ) : ℕ := 12 - x

-- Define the equality condition of equal profits (ratio of investments * time)
theorem Anoop_joins_after_7_months (x : ℕ) 
  (investment_ratio : ℕ) 
  (total_time_Anoop : ℕ) 
  (total_time_Arjun : ℕ) 
  (H : Arjun_investment * total_time_Arjun = Anoop_investment * total_time_Anoop) 
  : x = 7 := 
by 
  have h1 : Arjun_investment * Arjun_time = Anoop_investment * Anoop_time_join x,
  from H,
  have h2 : 20000 * 12 = 4000 * (12 - x),
  from h1,
  simp at h2,
  sorry

end Anoop_joins_after_7_months_l275_275172


namespace minimum_sum_distances_square_l275_275395

noncomputable def minimum_sum_of_distances
    (A B : ℝ × ℝ)
    (d : ℝ)
    (h_dist: dist A B = d)
    : ℝ :=
(1 + Real.sqrt 2) * d

theorem minimum_sum_distances_square
    (A B : ℝ × ℝ)
    (d : ℝ)
    (h_dist: dist A B = d)
    : minimum_sum_of_distances A B d h_dist = (1 + Real.sqrt 2) * d := by
sorry

end minimum_sum_distances_square_l275_275395


namespace karan_borrowed_amount_l275_275075

theorem karan_borrowed_amount :
  let P := 4257.63 in
  let A1_factor := (1 + 6/100)^3 in
  let A2_factor := (1 + 8/100)^3 in
  let A3_factor := (1 + 10/100)^3 in
  let Compound_Factor := A1_factor * A2_factor * A3_factor in
  Compound_Factor ≈ 1.9995 ∧ P * Compound_Factor = 8510 ->
  P = 4257.63 :=
by
  sorry

end karan_borrowed_amount_l275_275075


namespace zero_in_interval_l275_275132

-- Definition of the function f
def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 6

-- Stating the conditions as given above
theorem zero_in_interval : ∃ (n : ℤ), f (x : ℝ) = 0 ∧ (n : ℝ) < x ∧ x < (n + 1 : ℝ) :=
by
  -- We need to prove that there exists an integer n such that the zero of f(x) lies in the interval (n, n+1)
  use 1
  have h1 : f 1 < 0 := by norm_num [f, Real.exp]
  have h2 : 0 < f 2 := by norm_num [f, Real.exp]
  use h1, h2
  -- Adding intermediate calculation step
  have cont: continuous f := sorry
  have root := intermediate_value_theorem f 1 2
  exact root

end zero_in_interval_l275_275132


namespace find_constant_k_l275_275173

theorem find_constant_k 
  (k : ℝ)
  (h : ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) : 
  k = -16 :=
sorry

end find_constant_k_l275_275173


namespace find_p_fifth_plus_3_l275_275424

theorem find_p_fifth_plus_3 (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^4 + 3)) :
  p^5 + 3 = 35 :=
sorry

end find_p_fifth_plus_3_l275_275424


namespace seq_is_perfect_square_l275_275560

-- Define the sequence a as a function
def seq (a : ℕ → ℕ) : Prop :=
  (∀ n ≥ 2, a (n+1) = 3 * a n - 3 * a (n-1) + a (n-2)) ∧
  (2 * a 1 = a 0 + a 2 - 2) ∧
  (∀ m : ℕ, ∃ k : ℕ, k ≥ m ∧ is_square (a k) ∧ is_square (a (k+1)))

-- The main theorem to be proved
theorem seq_is_perfect_square {a : ℕ → ℕ} (h : seq a) :
  ∀ n : ℕ, is_square (a n) :=
sorry

end seq_is_perfect_square_l275_275560


namespace min_h10_l275_275596

def stringent (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, 1 ≤ x → 1 ≤ y → f(x) + f(y) > 2 * y^2

def sum_h (h : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum h

theorem min_h10 (h : ℕ → ℕ) (H_str : stringent h) (H_sum : sum_h h 16 = 2180) : h 10 = 136 :=
  sorry

end min_h10_l275_275596


namespace average_age_of_5_people_l275_275103

theorem average_age_of_5_people (avg_age_18 : ℕ) (avg_age_9 : ℕ) (age_15th : ℕ) (total_persons: ℕ) (persons_9: ℕ) (remaining_persons: ℕ) : 
  avg_age_18 = 15 ∧ 
  avg_age_9 = 16 ∧ 
  age_15th = 56 ∧ 
  total_persons = 18 ∧ 
  persons_9 = 9 ∧ 
  remaining_persons = 5 → 
  (avg_age_18 * total_persons - avg_age_9 * persons_9 - age_15th) / remaining_persons = 14 := 
sorry

end average_age_of_5_people_l275_275103


namespace find_alpha_l275_275327

theorem find_alpha (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1)
  (h_min : (∀ a b : ℝ, a > 0 → b > 0 → (a + b) * (1/a + 16/b) ≥ 25)) :
  ∃ α : ℝ, (∀ x : ℝ, x > 0 → ((m = 1/5) → (n = 4/5) → (P = (1/25, 1/5)) →
  (curve := (y : ℝ) → y = x^α) → ∀ x, x = 1/25 → curve x = 1/5 → α = 1)) := sorry

end find_alpha_l275_275327


namespace ashley_age_l275_275890

theorem ashley_age (A M : ℕ) (h1 : 4 * M = 7 * A) (h2 : A + M = 22) : A = 8 :=
sorry

end ashley_age_l275_275890


namespace median_of_set_l275_275845

noncomputable def set_of_numbers := {92, 90, 85, 88, 89, y}

def mean_of_set (s : Set ℝ) (mean : ℝ) :=
  (s.sum id : ℝ) / s.card = mean

theorem median_of_set (y : ℝ) (h_mean : mean_of_set set_of_numbers 88.5) :
  ∃ m, median set_of_numbers = 88.5 :=
sorry

end median_of_set_l275_275845


namespace length_PQ_mn_eq_21_l275_275058

def point (x y : ℝ) := (x, y)
def line_1 (p : ℝ) : ℝ × ℝ := (p, (12/5) * p)
def line_2 (p : ℝ) : ℝ × ℝ := (p, (2/7) * p)
def midpoint (P Q R : ℝ × ℝ) : Prop := 
  P.1 + Q.1 = 2 * R.1 ∧ P.2 + Q.2 = 2 * R.2

def length_PQ (P Q: ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def fraction_simplify (n d: ℝ) : ℝ := 
  if d ≠ 0 then n / d else 0

def result (l : ℝ) : ℝ × ℝ :=
  if l ≠ 0 then (l : ℝ, 1 : ℝ) else (0 : ℝ, 1 : ℝ)

theorem length_PQ_mn_eq_21 : 
  ∃ (m n : ℤ), 
    let PQ_length := length_PQ (line_1 (200/7)) (line_2 (140/7)) in
    fraction_simplify PQ_length 1 = PQ_length
    ∧ (m, n) = result PQ_length
    ∧ m + n = 21 := 
begin
  sorry
end

end length_PQ_mn_eq_21_l275_275058


namespace gcd_fact_8_10_l275_275660

-- Definitions based on the conditions in a)
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Question and conditions translated to a proof problem in Lean
theorem gcd_fact_8_10 : Nat.gcd (fact 8) (fact 10) = 40320 := by
  sorry

end gcd_fact_8_10_l275_275660


namespace geometric_sequence_sum_div_l275_275672

theorem geometric_sequence_sum_div :
  ∀ {a : ℕ → ℝ} {q : ℝ},
  (∀ n, a (n + 1) = a n * q) →
  q = -1 / 3 →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by
  intros a q geometric_seq common_ratio
  sorry

end geometric_sequence_sum_div_l275_275672


namespace some_expression_value_l275_275739

theorem some_expression_value {x y : ℝ} (h : x * y = 1) :
  let some_expression := (x + y) - 2 in
  (7 ^ (x + y)) ^ 2 / (7 ^ some_expression) ^ 2 = 2401 :=
by
  let some_expression := (x + y) - 2
  sorry

end some_expression_value_l275_275739


namespace sugarCubeWeight_l275_275185

theorem sugarCubeWeight
  (ants1 : ℕ) (sugar_cubes1 : ℕ) (weight1 : ℕ) (hours1 : ℕ)
  (ants2 : ℕ) (sugar_cubes2 : ℕ) (hours2 : ℕ) :
  ants1 = 15 →
  sugar_cubes1 = 600 →
  weight1 = 10 →
  hours1 = 5 →
  ants2 = 20 →
  sugar_cubes2 = 960 →
  hours2 = 3 →
  ∃ weight2 : ℕ, weight2 = 5 := by
  sorry

end sugarCubeWeight_l275_275185


namespace p_finishes_work_in_10_days_l275_275175

variable (p q r : ℝ)

def work_rate_p := 1 / 24
def work_rate_q := 1 / 9
def work_rate_r := 1 / 12
def work_done_q_r_in_3_days := (work_rate_q + work_rate_r) * 3
def remaining_work := 1 - work_done_q_r_in_3_days
def days_for_p := remaining_work / work_rate_p

theorem p_finishes_work_in_10_days :
  work_rate_p = 1 / 24 →
  work_rate_q = 1 / 9 →
  work_rate_r = 1 / 12 →
  p = 1 →
  q = 1 →
  r = 1 →
  days_for_p = 10 :=
by
  intros h_p h_q h_r h_pval h_qval h_rval
  rw [← h_p, ← h_q, ← h_r, work_rate_p, work_rate_q, work_rate_r, work_done_q_r_in_3_days, remaining_work, days_for_p]
  sorry

end p_finishes_work_in_10_days_l275_275175


namespace part_I_part_II_l275_275342

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

theorem part_I (f : ℝ → ℝ) (a b : ℝ) (h : ∀ x, f x = (a - 1/2) * x^2 + Real.log x)
  (tangent : ∀ x y, (x, y) = (1, h 1) → 2 * x + y + b = 0) :
  a = -1 ∧ b = -1/2 :=
by {
  sorry
}

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 - 2 * a * x + Real.log x

theorem part_II (a : ℝ) (h : ∀ x : ℝ, x > 1 → g a x < 0) :
  -1/2 ≤ a ∧ a ≤ 1/2 :=
by {
  sorry
}

end part_I_part_II_l275_275342


namespace skee_ball_tickets_l275_275606

-- Defining the conditions
def whack_a_mole_tickets : ℕ := 32
def tickets_spent_on_hat : ℕ := 7
def tickets_left : ℕ := 50

-- The theorem we need to prove
theorem skee_ball_tickets : ∃ x : ℕ, x = tickets_left + tickets_spent_on_hat - whack_a_mole_tickets :=
begin
  sorry
end

end skee_ball_tickets_l275_275606


namespace part_a_part_b_l275_275561

noncomputable theory

variables {α : Type*} {n : ℕ}
-- Conditions for part (a)
variables (ξ_n : ℕ → α) (ξ : α)
variables (a_n : ℕ → ℝ) (σ_n2 : ℕ → ℝ) 
variables (a : ℝ) (σ2 : ℝ)
variables (p : ℝ)

-- Assumption: ξ_n ~ 𝒩(a_n, σ_n^2) and ξ_n →d ξ
def isGaussian (ξ_n : ℕ → α) (a_n : ℕ → ℝ) (σ_n2 : ℕ → ℝ) := sorry -- Placeholder for Gaussian dist definition
def weakConvergence (ξ_n : ℕ → α) (ξ : α) := sorry -- Placeholder for weak convergence definition

-- Limits exist for a_n and σ_n^2
axiom lim_a : a = filterlim a_n atTop
axiom lim_σ2 : σ2 = filterlim σ_n2 atTop

-- Part (a): ξ is Gaussian with given parameters
theorem part_a (h₁ : isGaussian ξ_n a_n σ_n2) (h₂ : weakConvergence ξ_n ξ) : 
  isGaussian (λ _, ξ) (λ _, a) (λ _, σ2) :=
sorry

-- Definitions for part (b) implications of convergence
def convergence_in_probability (ξ_n : ℕ → α) (ξ : α) := sorry
def convergence_in_Lp (ξ_n : ℕ → α) (ξ : α) (p : ℝ) := sorry

-- Part (b): ξ_n →p ξ ↔ ξ_n →Lp ξ for any positive number p
theorem part_b (hp_pos : 0 < p) : 
  (convergence_in_probability ξ_n ξ ↔ convergence_in_Lp ξ_n ξ p) :=
sorry

end part_a_part_b_l275_275561


namespace savings_account_balance_l275_275228

-- Define the conditions as constant values
def initial_deposit : ℝ := 5000
def first_quarter_rate : ℝ := 0.07
def second_quarter_rate : ℝ := 0.085

-- Define the statement to be proved
theorem savings_account_balance :
  let first_quarter_interest := initial_deposit * first_quarter_rate,
      balance_after_first_quarter := initial_deposit + first_quarter_interest,
      second_quarter_interest := balance_after_first_quarter * second_quarter_rate,
      final_balance := balance_after_first_quarter + second_quarter_interest
  in final_balance = 5804.25 :=
by {
  sorry
}

end savings_account_balance_l275_275228


namespace inequality_sqrt_ab_l275_275814

theorem inequality_sqrt_ab {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧ (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := 
sorry

end inequality_sqrt_ab_l275_275814


namespace right_triangles_in_rectangle_l275_275488

-- Definitions for the points and the rectangle
structure Point where
  x : ℝ
  y : ℝ

structure Rectangle where
  A B C D : Point
  P Q : Point
  hP : P.x = (A.x + B.x) / 2 ∧ P.y = A.y
  hQ : Q.x = (C.x + D.x) / 3 ∧ Q.y = C.y
  hAB_CD : A.y = B.y ∧ C.y = D.y ∧ A.y ≠ C.y -- Ensuring AB and CD are horizontal lines with a vertical gap in between.

-- Declaration of the proof problem.
theorem right_triangles_in_rectangle (rect : Rectangle) : 
  ∃ T : finset (finset Point), (∀ t ∈ T, ∃ (X Y Z : Point), t = {X,Y,Z} ∧ right_triangle X Y Z) ∧
  finset.card T = 8 :=
by
  sorry

end right_triangles_in_rectangle_l275_275488


namespace find_f_sqrt_1000_l275_275893

noncomputable def f : ℝ → ℝ :=
  λ x, if 0 ≤ x ∧ x < 1 then
    if 0 ≤ x ∧ x < 1/2 then 2^x
    else log10 (x + 31)
  else sorry
  
lemma f_nonneg (x : ℝ) : f x ≥ 0 := 
  sorry

lemma f_functional_eq (x : ℝ) : f (x + 1) = sqrt (9 - (f x)^2) := 
  sorry

lemma periodicity (x : ℝ) : f (x + 2) = f x := 
  sorry

theorem find_f_sqrt_1000 : f (sqrt 1000) = 3 * sqrt 3 / 2 := 
  sorry

end find_f_sqrt_1000_l275_275893


namespace probability_seating_7_probability_seating_n_l275_275039

-- Definitions and theorem for case (a): n = 7
def num_ways_to_seat_7 (n : ℕ) (k : ℕ) : ℕ := n * (n - 1) * (n - 2) / k!

def valid_arrangements_7 : ℕ := 1

theorem probability_seating_7 : 
  let total_ways := num_ways_to_seat_7 7 3 in
  let valid_ways := valid_arrangements_7 in
  total_ways = 35 ∧ (valid_ways : ℚ) / total_ways = 0.2 := 
by
  sorry

-- Definitions and theorem for case (b): general n
def choose (n k : ℕ) : ℕ := n * (n - 1) / 2

def valid_arrangements_n (n : ℕ) : ℚ := (n - 4) * (n - 5) / 2

theorem probability_seating_n (n : ℕ) (hn : n ≥ 6) : 
  let total_ways := choose (n - 1) 2 in
  let valid_ways := valid_arrangements_n n in
  total_ways = (n - 1) * (n - 2) / 2 ∧ 
  (valid_ways : ℚ) / total_ways = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by
  sorry

end probability_seating_7_probability_seating_n_l275_275039


namespace smallest_product_is_neg4_l275_275963

def set : Set Int := {-9, -7, -5, -1, 1, 3, 4}

def valid_pairs (s : Set Int) : Set (Int × Int) :=
  {p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 + p.2 ≥ 0}

def smallest_valid_product (s : Set Int) : Int :=
  Inf ((valid_pairs s).image (λ p => p.1 * p.2))

theorem smallest_product_is_neg4 : smallest_valid_product set = -4 := by
  sorry

end smallest_product_is_neg4_l275_275963


namespace tan_A_l275_275751

-- Define the sides of the triangle and the right angle condition
variables (A B C : Type) [MetricSpace A B C]
variable (angle_BAC : Real)
variable (AB BC : ℝ)

-- Assume the given conditions
axiom BAC90 : angle_BAC = Real.pi / 2
axiom AB40 : AB = 40
axiom BC41 : BC = 41

-- Prove the value of tan A
theorem tan_A (A B C : Type) [MetricSpace A B C] (AB BC : ℝ) (angle_BAC : Real) (BAC90 : angle_BAC = Real.pi / 2)
    (AB40 : AB = 40) (BC41 : BC = 41) : 
    let AC := Real.sqrt (BC^2 - AB^2) in
    AC = 9 → Real.tan angle_BAC = 9 / 40 :=
sorry

end tan_A_l275_275751


namespace triangle_side_ratio_l275_275765

theorem triangle_side_ratio (A B C : ℝ) (n : ℕ) (hn : n ≥ 5) (h_angle : ∃ α β γ : ℝ, α + β + γ = π ∧ 
  min α (min β γ) ≤ π / n) : 
  ∃ (λ : ℝ), λ ≥ 2 * Real.cos (π / n) ∧ 
  ∀ a b c : ℝ, a = λ * b → a = λ * c → (α = π / 2 ∨ β = π / 2 ∨ γ = π / 2 → 
  (a ^ 2 + b ^ 2 = c ^ 2 ∧ α = π / n ∧ β = π / n ∧ γ = π / n → a / c = λ)) := 
sorry

end triangle_side_ratio_l275_275765


namespace find_a_for_even_function_l275_275997

theorem find_a_for_even_function (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = (x + 1) * (x + a) ∧ f (-x) = f x) : a = -1 := by 
  sorry

end find_a_for_even_function_l275_275997


namespace arbitrary_long_roof_l275_275515

theorem arbitrary_long_roof :
  (∀ ε > 0, ∃ n : ℕ, ∑ i in range n, 1 / (2 * i) > ε) :=
begin
  sorry
end

end arbitrary_long_roof_l275_275515


namespace point_P_outside_circle_l275_275848

theorem point_P_outside_circle (m : ℝ) : (m^4 + 25) > 24 :=
by
  calc 
    m^4 + 25 = m^4 + 25 : by sorry -- since m^4 + 25 is always greater than 24 for any real number m

end point_P_outside_circle_l275_275848


namespace quadrilateral_diagonal_midpoints_l275_275682

-- Define the points of the quadrilateral
variables (A B C D : ℝ × ℝ)

-- Define the midpoints of the sides
def P : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def Q : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def R : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
def S : ℝ × ℝ := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

-- Define the squared distances
def dist_sq (X Y : ℝ × ℝ) : ℝ :=
(X.1 - Y.1) ^ 2 + (X.2 - Y.2) ^ 2

-- The main theorem to be proved
theorem quadrilateral_diagonal_midpoints :
  dist_sq A C + dist_sq B D = 2 * (dist_sq P R + dist_sq Q S) :=
sorry

end quadrilateral_diagonal_midpoints_l275_275682


namespace monotonic_decreasing_interval_l275_275966

theorem monotonic_decreasing_interval (ω : ℝ) (h : ω > 0) (symmetry_distance : Real) :
  symmetry_distance = π / 2 →
  monotonic_decreasing_interval 
    (λ x => Real.sin (ω * x + π / 3)) = [π / 12, π / 2] :=
  by
  intro h_dist
  sorry

end monotonic_decreasing_interval_l275_275966


namespace tan_A_l275_275749

-- Define the sides of the triangle and the right angle condition
variables (A B C : Type) [MetricSpace A B C]
variable (angle_BAC : Real)
variable (AB BC : ℝ)

-- Assume the given conditions
axiom BAC90 : angle_BAC = Real.pi / 2
axiom AB40 : AB = 40
axiom BC41 : BC = 41

-- Prove the value of tan A
theorem tan_A (A B C : Type) [MetricSpace A B C] (AB BC : ℝ) (angle_BAC : Real) (BAC90 : angle_BAC = Real.pi / 2)
    (AB40 : AB = 40) (BC41 : BC = 41) : 
    let AC := Real.sqrt (BC^2 - AB^2) in
    AC = 9 → Real.tan angle_BAC = 9 / 40 :=
sorry

end tan_A_l275_275749


namespace a_n_is_arithmetic_seq_sum_of_b_n_l275_275706

-- Define the function f(x)
def f (n x : ℝ) : ℝ := x^2 - 2*(n+1)*x + n^2 + 5*n - 7

-- Define the sequence a_n
def a_n (n : ℕ) : ℝ := 3*n - 8

-- Define the sequence b_n
def b_n (n : ℕ) : ℝ := abs (a_n n)

-- Define the sum of the first n terms of sequence b_n
def S_n (n : ℕ) : ℝ :=
  if n ≤ 2 then
    (13 * n - 3 * n^2) / 2
  else
    (3 * n^2 - 13 * n + 28) / 2

-- Statement to prove that a_n is an arithmetic sequence
theorem a_n_is_arithmetic_seq : ∀ n : ℕ, a_n (n + 1) - a_n n = 3 := sorry

-- Statement to prove the sum of the first n terms of sequence b_n
theorem sum_of_b_n : ∀ n : ℕ, ∑ i in finset.range n, b_n i = S_n n := sorry

end a_n_is_arithmetic_seq_sum_of_b_n_l275_275706


namespace rebecca_items_count_l275_275809

variable (tent_stakes drink_mix_packets water_bottles total_items : ℕ)

-- Conditions
def tent_stakes_count : tent_stakes := 4
def drink_mix_count : drink_mix_packets := 3 * tent_stakes_count
def water_bottles_count : water_bottles := tent_stakes_count + 2

-- Question: Total items
def total_items_count : total_items := 
  tent_stakes_count + drink_mix_count + water_bottles_count

-- Proof (statement only, no proof needed in this scenario)
theorem rebecca_items_count :
  total_items_count = 22 := by
  sorry

end rebecca_items_count_l275_275809


namespace find_abc_square_sum_l275_275372

theorem find_abc_square_sum (a b c : ℝ) 
  (h1 : a^2 + 3 * b = 9) 
  (h2 : b^2 + 5 * c = -8) 
  (h3 : c^2 + 7 * a = -18) : 
  a^2 + b^2 + c^2 = 20.75 := 
sorry

end find_abc_square_sum_l275_275372


namespace sum_of_T_is_correct_nine_hundred_twenty_nine_in_base_3_l275_275060

-- Define the set T of all three-digit numbers in base 3
def T : Finset ℕ := {n | ∃ (d2 d1 d0 : ℕ), d2 ∈ {1, 2} ∧ d1 ∈ {0, 1, 2} ∧ d0 ∈ {0, 1, 2} ∧ n = d2 * 9 + d1 * 3 + d0}.to_finset

-- State the theorem
theorem sum_of_T_is_correct : ∑ n in T, n = 729 :=
by sorry

-- Show that 729 in base 10 converts to 1000000 in base 3
theorem nine_hundred_twenty_nine_in_base_3 : nat.to_digits 3 729 = [1, 0, 0, 0, 0, 0, 0] :=
by sorry

end sum_of_T_is_correct_nine_hundred_twenty_nine_in_base_3_l275_275060


namespace tom_average_speed_l275_275555

theorem tom_average_speed :
  ∀ (d1 d2 s1 s2 : ℕ), d1 = 10 → d2 = 10 → s1 = 12 → s2 = 10 →
  let total_distance := d1 + d2 in
  let time1 := d1 / s1 in
  let time2 := d2 / s2 in
  let total_time := time1 + time2 in
  let average_speed := total_distance / total_time in
  average_speed = 120 / 11 := 
by
  intros d1 d2 s1 s2 h_d1 h_d2 h_s1 h_s2 total_distance time1 time2 total_time average_speed,
  subst h_d1,
  subst h_d2,
  subst h_s1,
  subst h_s2,
  sorry

end tom_average_speed_l275_275555


namespace part1_part2_l275_275307

section
variable (f g : ℝ → ℝ) (a : ℝ)

-- Define the given functions.
noncomputable def f := λ x : ℝ, x^3 - x
noncomputable def g := λ x : ℝ, x^2 + a

-- Part 1: Prove that if x₁ = -1, then a = 3
theorem part1 (x₁ : ℝ) (h : x₁ = -1) (h_tangent : ∀ x₃ : ℝ, f'(x₃) = g'(x₃)) : 
    a = 3 :=
sorry

-- Part 2: Prove the range of values for a is [-1, +∞).
theorem part2 (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, 3 * x₁^2 - 1 = 2 * x₂ ∧ a = x₂^2 - 2 * x₁^3) : 
    a ∈ Set.Ici (-1) :=
sorry

end

end part1_part2_l275_275307


namespace ratio_of_segments_l275_275077

theorem ratio_of_segments (a b x : ℝ) (h₁ : a = 9 * x) (h₂ : b = 99 * x) : b / a = 11 := by
  sorry

end ratio_of_segments_l275_275077


namespace parallel_slope_l275_275532

theorem parallel_slope (x y : ℝ) (h : 3 * x + 6 * y = 12) :
  ∃ m : ℝ, m = -1/2 ∧ (∃ b : ℝ, y = m * x + b) :=
by {
  use -1/2,
  split,
  {
    exact rfl,
  },
  {
    use 2,
    have h1 : 6 * y = -3 * x + 12 := by linarith,
    have h2 : y = (-1/2) * x + 2 := by linarith,
    exact h2,
  }
  sorry
}

end parallel_slope_l275_275532


namespace probability_seven_chairs_probability_n_chairs_l275_275036
-- Importing necessary library to ensure our Lean code can be built successfully

-- Definition for case where n = 7
theorem probability_seven_chairs : 
  let total_seating := 7 * 6 * 5 / 6 
  let favorable_seating := 1 
  let probability := favorable_seating / total_seating 
  probability = 1 / 35 := 
by 
  sorry

-- Definition for general case where n ≥ 6
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) : 
  let total_seating := (n - 1) * (n - 2) / 2 
  let favorable_seating := (n - 4) * (n - 5) / 2 
  let probability := favorable_seating / total_seating 
  probability = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := 
by 
  sorry

end probability_seven_chairs_probability_n_chairs_l275_275036


namespace average_value_of_x_l275_275498

theorem average_value_of_x
  (x : ℝ)
  (h : (5 + 5 + x + 6 + 8) / 5 = 6) :
  x = 6 :=
sorry

end average_value_of_x_l275_275498


namespace not_perfect_square_l275_275582

theorem not_perfect_square (n : ℕ) (h : ∀ i, i < n → (dig10 i = 6) ∨ (dig10 i = 0)) : ¬ (∃ k : ℕ, (n = k^2)) :=
sorry

end not_perfect_square_l275_275582


namespace min_distance_sum_l275_275674

-- Definitions based on conditions
def point_on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 8 * P.1

def distance_point_to_directrix (P : ℝ × ℝ) : ℝ :=
  abs (P.1 + 2)

def distance_point_to_line (P : ℝ × ℝ) : ℝ :=
  abs (4 * P.1 + 3 * P.2 + 8) / sqrt (4^2 + 3^2)

-- Problem statement as Lean 4 theorem
theorem min_distance_sum (P : ℝ × ℝ) (hP : point_on_parabola P) :
  (distance_point_to_directrix P + distance_point_to_line P) = 16 / 5 :=
sorry

end min_distance_sum_l275_275674


namespace v_20_l275_275620

noncomputable def sequence (b : ℝ) : ℕ → ℝ
| 0       := b
| (n + 1) := -1 / (2 * sequence n + 1)

theorem v_20 (b : ℝ) (h : b ≥ 0) : sequence b 19 = -1 / (2 * b + 1) :=
by {
  sorry
}

end v_20_l275_275620


namespace function_A_is_even_and_increasing_function_B_is_not_even_function_C_is_not_even_function_D_is_even_but_not_increasing_l275_275938

-- Define even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define increasing function on (0, 1)
def increasing_on_0_1 (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, 0 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2

-- Function definitions
def f_A (x : ℝ) : ℝ := abs x
def f_B (x : ℝ) : ℝ := 3 - x
def f_C (x : ℝ) : ℝ := 1 / x
def f_D (x : ℝ) : ℝ := -x^2 + 4

-- Prove that f_A is the only even function that is increasing on (0,1)
theorem function_A_is_even_and_increasing : even_function f_A ∧ increasing_on_0_1 f_A :=
  sorry

theorem function_B_is_not_even : ¬even_function f_B :=
  sorry

theorem function_C_is_not_even : ¬even_function f_C :=
  sorry

theorem function_D_is_even_but_not_increasing : even_function f_D ∧ ¬increasing_on_0_1 f_D :=
  sorry

end function_A_is_even_and_increasing_function_B_is_not_even_function_C_is_not_even_function_D_is_even_but_not_increasing_l275_275938


namespace matching_times_l275_275773

noncomputable def chargeAtTime (t : Nat) : ℚ :=
  100 - t / 6

def isMatchingTime (hh mm : Nat) : Prop :=
  hh * 60 + mm = 100 - (hh * 60 + mm) / 6

theorem matching_times:
  isMatchingTime 4 52 ∨
  isMatchingTime 5 43 ∨
  isMatchingTime 6 35 ∨
  isMatchingTime 7 26 ∨
  isMatchingTime 9 9 :=
by
  repeat { sorry }

end matching_times_l275_275773


namespace probability_seating_7_probability_seating_n_l275_275041

-- Definitions and theorem for case (a): n = 7
def num_ways_to_seat_7 (n : ℕ) (k : ℕ) : ℕ := n * (n - 1) * (n - 2) / k!

def valid_arrangements_7 : ℕ := 1

theorem probability_seating_7 : 
  let total_ways := num_ways_to_seat_7 7 3 in
  let valid_ways := valid_arrangements_7 in
  total_ways = 35 ∧ (valid_ways : ℚ) / total_ways = 0.2 := 
by
  sorry

-- Definitions and theorem for case (b): general n
def choose (n k : ℕ) : ℕ := n * (n - 1) / 2

def valid_arrangements_n (n : ℕ) : ℚ := (n - 4) * (n - 5) / 2

theorem probability_seating_n (n : ℕ) (hn : n ≥ 6) : 
  let total_ways := choose (n - 1) 2 in
  let valid_ways := valid_arrangements_n n in
  total_ways = (n - 1) * (n - 2) / 2 ∧ 
  (valid_ways : ℚ) / total_ways = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by
  sorry

end probability_seating_7_probability_seating_n_l275_275041


namespace number_of_paths_l275_275815

theorem number_of_paths (n : ℕ) : 
  (paths (0, 0) (0, 0) (2 * n) = (nat.choose (2 * n) n) ^ 2) :=
sorry

end number_of_paths_l275_275815


namespace largest_integer_no_repeated_digits_rel_prime_to_6_l275_275870

noncomputable def largest_rel_prime_to_six_with_no_repeated_digits : ℕ := 987654301

theorem largest_integer_no_repeated_digits_rel_prime_to_6 :
  ∃ (n : ℕ), (∀ i j : ℕ, i ≠ j → (n.digits 10).nth i ≠ (n.digits 10).nth j) -- n has no repeated digits
  ∧ gcd n 6 = 1 -- n is relatively prime to 6
  ∧ n = largest_rel_prime_to_six_with_no_repeated_digits :=       -- n is equal to 987654301
  by {
  use 987654301,
  split,
  sorry, -- proof for no repeated digits
  split,
  sorry, -- proof for gcd(n, 6) = 1
  refl
}

end largest_integer_no_repeated_digits_rel_prime_to_6_l275_275870


namespace digits_making_number_divisible_by_4_l275_275203

theorem digits_making_number_divisible_by_4 (N : ℕ) (hN : N < 10) :
  (∃ n0 n4 n8, n0 = 0 ∧ n4 = 4 ∧ n8 = 8 ∧ N = n0 ∨ N = n4 ∨ N = n8) :=
by
  sorry

end digits_making_number_divisible_by_4_l275_275203


namespace ratio_areas_l275_275782

noncomputable def triples_on_plane (a b c : ℝ) : set (ℝ × ℝ × ℝ) :=
{ t | t.1 + t.2 + t.2 = a + b + c ∧ t.1 ≥ 0 ∧ t.2 ≥ 0 ∧ t.3 ≥ 0 }

def supports (t s: ℝ × ℝ × ℝ) : Prop :=
((t.1 ≥ s.1 ∧ t.2 ≥ s.2 ∧ t.3 < s.3) ∨
 (t.1 ≥ s.1 ∧ t.2 < s.2 ∧ t.3 ≥ s.3) ∨
 (t.1 < s.1 ∧ t.2 ≥ s.2 ∧ t.3 ≥ s.3))

def subset_supporting (s: ℝ × ℝ × ℝ) (T : set (ℝ × ℝ × ℝ)) : set (ℝ × ℝ × ℝ) :=
{ t | t ∈ T ∧ supports t s }

def area_of_region (T : set (ℝ × ℝ × ℝ)) : ℝ := sorry

theorem ratio_areas (T : set (ℝ × ℝ × ℝ)) (S : set (ℝ × ℝ × ℝ))
  (triple_support : S = subset_supporting (1/4, 1/4, 1/4) T)
  (plane_condition : T = triples_on_plane 1 1 1) :
  (area_of_region S / area_of_region T) = (3 / 2) :=
sorry

end ratio_areas_l275_275782


namespace find_t_l275_275139

theorem find_t 
  (A B C : Point)
  (hA : A = (1, 10))
  (hB : B = (3, 0))
  (hC : C = (9, 0))
  (t : ℝ)
  (hJ : J = (3 - t / 5, t))
  (hK : K = (9 - 4 * t / 5, t))
  (area_AJK : (1 / 2) * abs ((3 / 5) * (6 - t) * (10 - t)) = 7.5) : 
  t = 5 :=
by 
  sorry

end find_t_l275_275139


namespace courtyard_length_proof_l275_275573

-- Definitions for the conditions
def breadth : ℝ := 16 -- breadth of the courtyard in meters
def brick_length : ℝ := 0.20 -- length of one brick in meters
def brick_breadth : ℝ := 0.10 -- breadth of one brick in meters
def num_bricks : ℝ := 16000 -- total number of bricks required

-- The problem: proving the length of the courtyard is 20 meters
theorem courtyard_length_proof : 
  let brick_area := brick_length * brick_breadth,
      total_area := num_bricks * brick_area
  in total_area / breadth = 20 := by
  let brick_area := brick_length * brick_breadth
  let total_area := num_bricks * brick_area
  sorry -- Proof is omitted.

end courtyard_length_proof_l275_275573


namespace correct_statement_is_B_l275_275880

-- Define integers and zero
def is_integer (n : ℤ) : Prop := True
def is_zero (n : ℤ) : Prop := n = 0

-- Define rational numbers
def is_rational (q : ℚ) : Prop := True

-- Positive and negative zero cannot co-exist
def is_positive (n : ℤ) : Prop := n > 0
def is_negative (n : ℤ) : Prop := n < 0

-- Statement A: Integers and negative integers are collectively referred to as integers.
def statement_A : Prop :=
  ∀ n : ℤ, (is_positive n ∨ is_negative n) ↔ is_integer n

-- Statement B: Integers and fractions are collectively referred to as rational numbers.
def statement_B : Prop :=
  ∀ q : ℚ, is_rational q

-- Statement C: Zero can be either a positive integer or a negative integer.
def statement_C : Prop :=
  ∀ n : ℤ, is_zero n → (is_positive n ∨ is_negative n)

-- Statement D: A rational number is either a positive number or a negative number.
def statement_D : Prop :=
  ∀ q : ℚ, (q ≠ 0 → (is_positive q.num ∨ is_negative q.num))

-- The problem is to prove that statement B is the only correct statement.
theorem correct_statement_is_B : statement_B ∧ ¬statement_A ∧ ¬statement_C ∧ ¬statement_D :=
by sorry

end correct_statement_is_B_l275_275880


namespace ratio_of_liquid_level_rises_l275_275522

-- Definitions for given conditions
def radius_small_cone : ℝ := 5
def radius_large_cone : ℝ := 10
def radius_marble : ℝ := 2

-- Volumes of cones initially filled with liquid
def volume_cone (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Volume of the small and large cones with initial liquid levels h1 and h2
def volume_small_cone (h1 : ℝ) : ℝ := volume_cone radius_small_cone h1
def volume_large_cone (h2 : ℝ) : ℝ := volume_cone radius_large_cone h2

-- Condition that both cones contain the same amount of liquid
axiom same_volume (h1 h2 : ℝ) : volume_small_cone h1 = volume_large_cone h2

-- Volume of the spherical marble
def volume_marble : ℝ := (4/3) * Real.pi * radius_marble^3

-- Proving the main question: the ratio of the rise of the liquid level in the smaller cone to the rise of the liquid level in the larger cone
theorem ratio_of_liquid_level_rises (h1 h2 : ℝ) :
  (same_volume h1 h2) →
  4 = 4 :=
by
  sorry

end ratio_of_liquid_level_rises_l275_275522


namespace simplify_and_evaluate_l275_275817

theorem simplify_and_evaluate (a b : ℝ) (h : |a + 2| + (b - 1)^2 = 0) : 
  (a + 3 * b) * (2 * a - b) - 2 * (a - b)^2 = -23 := by
  sorry

end simplify_and_evaluate_l275_275817


namespace area_of_triangle_ADC_l275_275005

-- Define the constants for the problem
variable (BD DC : ℝ)
variable (abd_area adc_area : ℝ)

-- Given conditions
axiom ratio_condition : BD / DC = 5 / 2
axiom area_abd : abd_area = 35

-- Define the theorem to be proved
theorem area_of_triangle_ADC :
  ∃ adc_area, adc_area = 14 ∧ abd_area / adc_area = BD / DC := 
sorry

end area_of_triangle_ADC_l275_275005


namespace union_of_A_and_B_l275_275350

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4} := by
  sorry

end union_of_A_and_B_l275_275350


namespace number_of_valid_digits_l275_275199

theorem number_of_valid_digits :
  let N := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] in
  let valid_digits := [n | n ∈ N ∧ (640 + n) % 4 = 0] in
  valid_digits.length = 5 :=
by sorry

end number_of_valid_digits_l275_275199


namespace sum_log_abs_l275_275325

theorem sum_log_abs (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ (n : ℕ), a n = 32 * (1 / 4) ^ (n - 1)) 
  (h2 : S 6 / S 3 = 65 / 64)
  (h3 : ∀ (n : ℕ), S n = ∑ i in finset.range n, a (i + 1)):
  (∑ i in finset.range 10, | log 2 (a (i + 1)) |) = 58 :=
by
  sorry

end sum_log_abs_l275_275325


namespace find_a_b_l275_275985

theorem find_a_b :
  ∃ a b : ℝ, 
    (a = -4) ∧ (b = -9) ∧
    (∀ x : ℝ, |8 * x + 9| < 7 ↔ a * x^2 + b * x - 2 > 0) := 
sorry

end find_a_b_l275_275985


namespace area_ratio_of_triangle_and_hexagon_l275_275108

theorem area_ratio_of_triangle_and_hexagon (a : ℝ) (h1 : a > 0) :
    let triangle_side := 2 * a 
    let triangle_area := 4 * (Math.sqrt 3 / 4 * (a ^ 2))
    let hexagon_area := 6 * (Math.sqrt 3 / 4 * (a ^ 2))
    triangle_area / hexagon_area = 2 / 3 := by
    let triangle_side := 2 * a
    let triangle_area := 4 * (Math.sqrt 3 / 4 * (a ^ 2))
    let hexagon_area := 6 * (Math.sqrt 3 / 4 * (a ^ 2))
    sorry

end area_ratio_of_triangle_and_hexagon_l275_275108


namespace combined_fish_population_l275_275382

theorem combined_fish_population:
  (n1A n2A mA n1B n2B mB : ℕ)
  (h1 : n1A = 50)
  (h2 : n2A = 60)
  (h3 : mA = 4)
  (h4 : n1B = 30)
  (h5 : n2B = 40)
  (h6 : mB = 3) :
  let N_A := (n1A * n2A) / mA in
  let N_B := (n1B * n2B) / mB in
  N_A + N_B = 1150 := by
{
  sorry
}

end combined_fish_population_l275_275382


namespace new_mean_after_addition_l275_275551

theorem new_mean_after_addition (numbers : List ℝ) (h_len : numbers.length = 15) (h_avg : (numbers.sum / 15) = 40) : 
  ((numbers.map (λ x, x + 13)).sum / 15) = 53 :=
by
  sorry

end new_mean_after_addition_l275_275551


namespace minimize_J_l275_275363

def H (p q : ℝ) : ℝ := -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

def J (p : ℝ) : ℝ := max (H p 0) (H p 1)

theorem minimize_J : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ p = 9 / 16 ∧ (∀ q, 0 ≤ q ∧ q ≤ 1 → J p ≤ J q) :=
by
  sorry

end minimize_J_l275_275363


namespace isosceles_tetrahedron_l275_275432

theorem isosceles_tetrahedron 
  {ABC : Type} [triangle ABC] 
  {I : point} [incenter I ABC] 
  {D : point} [intersection_point D (line_through AI) (circumcircle ABC)] :
  distance D B = distance D C ∧ distance D C = distance D I :=
  sorry

end isosceles_tetrahedron_l275_275432


namespace num_possibilities_l275_275196

def last_digit_divisible_by_4 (n : Nat) : Prop := (60 + n) % 4 = 0

theorem num_possibilities : {n : Nat | n < 10 ∧ last_digit_divisible_by_4 n}.card = 3 := by
  sorry

end num_possibilities_l275_275196


namespace compute_a_plus_b_l275_275988

theorem compute_a_plus_b (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
    (h3 : (∏ i in finset.range (b - a), real.log (i + a + 1) / real.log (i + a)) = 3)
    (h4 : b - a = 1560) : a + b = 1740 :=
sorry

end compute_a_plus_b_l275_275988


namespace Bill_Sunday_miles_l275_275445

-- Definitions based on problem conditions
def Bill_Saturday (B : ℕ) : ℕ := B
def Bill_Sunday (B : ℕ) : ℕ := B + 4
def Julia_Sunday (B : ℕ) : ℕ := 2 * (B + 4)
def Alex_Total (B : ℕ) : ℕ := B + 2

-- Total miles equation based on conditions
def total_miles (B : ℕ) : ℕ := Bill_Saturday B + Bill_Sunday B + Julia_Sunday B + Alex_Total B

-- Proof statement
theorem Bill_Sunday_miles (B : ℕ) (h : total_miles B = 54) : Bill_Sunday B = 14 :=
by {
  -- calculations and proof would go here if not omitted
  sorry
}

end Bill_Sunday_miles_l275_275445


namespace volume_of_cut_pyramid_l275_275922

-- Given Conditions
def base_length_1 : ℝ := 10 * Real.sqrt 2
def base_length_2 : ℝ := 6 * Real.sqrt 2
def slant_edge : ℝ := 12
def height_cut : ℝ := 4

-- Volume Calculation Outcome 
noncomputable def volume_cut_pyramid : ℝ := 
  let h := 2 * Real.sqrt 19
  20 * ((h - height_cut) / h)^3 * (h - height_cut)

-- Theorem: Volume of the cut pyramid
theorem volume_of_cut_pyramid : 
  Volume (Pyramid (base_length_1, base_length_2) slant_edge height_cut) = volume_cut_pyramid := 
sorry

end volume_of_cut_pyramid_l275_275922


namespace range_of_f_l275_275160

def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_of_f : set.Ioc 0 1 = set_of (λ y, ∃ x : ℝ, f x = y) :=
by
  sorry

end range_of_f_l275_275160


namespace find_m_n_l275_275998

theorem find_m_n (m n x1 x2 : ℕ) (hm : 0 < m) (hn : 0 < n) (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (h_eq : x1 * x2 = m + n) (h_sum : x1 + x2 = m * n) :
  (m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 2) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1) := 
sorry

end find_m_n_l275_275998


namespace w_profit_share_l275_275179

noncomputable def investment_time_product (investment: ℚ) (time: ℚ) : ℚ := investment * time

noncomputable def total_investment_time_product (investments: List (ℚ × ℚ)) : ℚ :=
  investments.foldl (λ (acc: ℚ) (p: ℚ × ℚ), acc + investment_time_product p.1 p.2) 0

noncomputable def profit_share (investment_product: ℚ) (total_investment_product: ℚ) (total_profit: ℚ) : ℚ :=
  (investment_product / total_investment_product) * total_profit

theorem w_profit_share :
  let investment_x := 64500
  let investment_y := 78000
  let investment_z := 86200
  let time_xyz := 11 -- months
  let investment_w := 93500
  let time_w := 4 -- months
  let total_profit := 319750

  let investments := [
    (investment_x, time_xyz),
    (investment_y, time_xyz),
    (investment_z, time_xyz),
    (investment_w, time_w)
  ]

  let total_investment_time := total_investment_time_product investments
  let w_investment_time := investment_time_product investment_w time_w

  profit_share w_investment_time total_investment_time total_profit ≈ 41366.05 := 
sorry

end w_profit_share_l275_275179


namespace functions_equivalence_B_l275_275543

def domain (f : ℝ → ℝ) : Set ℝ := { x | ∃ y, f x = y } 

theorem functions_equivalence_B : 
  let f_b := λ x : ℝ, abs x
  let g_b := λ x : ℝ, real.sqrt (x^2)
  domain f_b = domain g_b ∧ ∀ x ∈ domain f_b, f_b x = g_b x :=
by
  sorry

end functions_equivalence_B_l275_275543


namespace digits_making_number_divisible_by_4_l275_275204

theorem digits_making_number_divisible_by_4 (N : ℕ) (hN : N < 10) :
  (∃ n0 n4 n8, n0 = 0 ∧ n4 = 4 ∧ n8 = 8 ∧ N = n0 ∨ N = n4 ∨ N = n8) :=
by
  sorry

end digits_making_number_divisible_by_4_l275_275204


namespace total_balls_in_box_l275_275570

theorem total_balls_in_box :
  ∀ (W B R : ℕ), 
    W = 16 →
    B = W + 12 →
    R = 2 * B →
    W + B + R = 100 :=
by
  intros W B R hW hB hR
  sorry

end total_balls_in_box_l275_275570


namespace distinct_products_l275_275720

theorem distinct_products (S : Finset ℕ) (hS : S = {2, 3, 5, 7, 11}) :
  (S.powerset.filter (λ x, 2 ≤ x.card)).card = 26 :=
by
  sorry

end distinct_products_l275_275720


namespace product_remainder_l275_275508

theorem product_remainder (a b c d : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 4) (hd : d % 7 = 5) :
  (a * b * c * d) % 7 = 1 :=
by
  sorry

end product_remainder_l275_275508


namespace anna_bought_five_chocolate_bars_l275_275598

noncomputable section

def initial_amount : ℝ := 10
def price_chewing_gum : ℝ := 1
def price_candy_cane : ℝ := 0.5
def remaining_amount : ℝ := 1

def chewing_gum_cost : ℝ := 3 * price_chewing_gum
def candy_cane_cost : ℝ := 2 * price_candy_cane

def total_spent : ℝ := initial_amount - remaining_amount
def known_items_cost : ℝ := chewing_gum_cost + candy_cane_cost
def chocolate_bars_cost : ℝ := total_spent - known_items_cost
def price_chocolate_bar : ℝ := 1

def chocolate_bars_bought : ℝ := chocolate_bars_cost / price_chocolate_bar

theorem anna_bought_five_chocolate_bars : chocolate_bars_bought = 5 := 
by
  sorry

end anna_bought_five_chocolate_bars_l275_275598


namespace number_of_valid_squares_l275_275564

def is_valid_square (n : ℕ) : Prop :=
  ∃ m : ℕ, (10 ≤ m ∧ m ≤ 99 ∧ m = 14 * n + 2)

theorem number_of_valid_squares : 
  {n : ℕ | is_valid_square n ∧ n > 2}.finite.to_finset.card = 4 := 
by 
  sorry

end number_of_valid_squares_l275_275564


namespace arithmetic_seq_sum_a3_a15_l275_275696

theorem arithmetic_seq_sum_a3_a15 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_eq : a 1 - a 5 + a 9 - a 13 + a 17 = 117) :
  a 3 + a 15 = 234 :=
sorry

end arithmetic_seq_sum_a3_a15_l275_275696


namespace solve_equation_l275_275819

theorem solve_equation (x : ℝ) :
  x * (x + 3)^2 * (5 - x) = 0 ∧ x^2 + 3 * x + 2 > 0 ↔ x = -3 ∨ x = 0 ∨ x = 5 :=
by
  sorry

end solve_equation_l275_275819


namespace find_ordered_pair_solution_l275_275975

theorem find_ordered_pair_solution :
  ∃ (x y : ℚ), 7 * x = 10 - 3 * y ∧ 4 * x = 5 * y - 14 ∧ x = 8 / 47 ∧ y = 138 / 47 :=
by
  use (8 / 47, 138 / 47)
  sorry

end find_ordered_pair_solution_l275_275975


namespace false_prime_divisibility_l275_275882

theorem false_prime_divisibility :
  ¬ ∃ n : ℕ, ∀ p : ℕ, nat.prime p → n < p → ∀ k : ℕ, ∃ m : ℕ, 10^m ≡ 1 [MOD p] :=
sorry

end false_prime_divisibility_l275_275882


namespace scaled_det_l275_275994

variable (x y z a b c p q r : ℝ)
variable (det_orig : ℝ)
variable (h : Matrix.det ![![x, y, z], ![a, b, c], ![p, q, r]] = 2)

theorem scaled_det (h : Matrix.det ![![x, y, z], ![a, b, c], ![p, q, r]] = 2) :
  Matrix.det ![![3*x, 3*y, 3*z], ![3*a, 3*b, 3*c], ![3*p, 3*q, 3*r]] = 54 :=
by
  sorry

end scaled_det_l275_275994


namespace sum_of_fifth_powers_l275_275439

theorem sum_of_fifth_powers (n : ℕ) (x : Fin n → ℤ)
  (h1 : ∀ i, x i = 0 ∨ x i = 1 ∨ x i = -2)
  (h2 : (∑ i, x i) = -5)
  (h3 : (∑ i, (x i)^2) = 19) :
  (∑ i, (x i)^5) = -125 :=
sorry

end sum_of_fifth_powers_l275_275439


namespace parabola_equation_l275_275579

theorem parabola_equation (p : ℝ) (hp : 0 < p)
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (A B : ℝ × ℝ)
  (hA : A = (x1, y1)) (hB : B = (x2, y2))
  (h_intersect : y1^2 = 2*p*x1 ∧ y2^2 = 2*p*x2)
  (M : ℝ × ℝ) (hM : M = ((x1 + x2) / 2, (y1 + y2) / 2))
  (hM_coords : M = (3, 2)) :
  p = 2 ∨ p = 4 :=
sorry

end parabola_equation_l275_275579


namespace sugar_fraction_in_cup1_is_one_fourth_l275_275357

variable {tea1_initial sugar2_initial tea_transfer external_tea liquid_transfer tea2_sugar2 sugar1 tea_total sugar_total liquid1_total : ℕ}

def cup1_initial_tea := 6
def cup2_initial_sugar := 4

def cup1_after_transfer (init: ℕ) (tran: ℕ) (add: ℕ) : ℕ :=
  init - tran + add

def cup2_after_transfer (init_sugar: ℕ) (add_tea: ℕ) : ℕ :=
  init_sugar + add_tea

def liquid_ratio (sugar: ℕ) (tea: ℕ) : ℕ :=
  3 * sugar / (sugar + tea)

def sugar_in_mixture (init_sugar: ℕ) (init_tea: ℕ) (liquid: ℕ) : ℕ :=
  (3 * init_sugar / (init_sugar + init_tea))

def final_composition_tea (initial: ℕ) (transfer_back: ℕ) : ℕ :=
  initial + transfer_back

def final_composition_sugar (sugar: ℕ) : ℕ :=
  sugar

def total_liquid (tea: ℕ) (sugar: ℕ) : ℕ :=
  tea + sugar

def sugar_fraction (sugar: ℕ) (total: ℕ) : ℕ :=
  sugar / total

theorem sugar_fraction_in_cup1_is_one_fourth :
  tea1_initial = cup1_initial_tea →
  sugar2_initial = cup2_initial_sugar →
  tea_transfer = 2 →
  external_tea = 2 →
  liquid_transfer = 3 →
  let tea1_after := cup1_after_transfer tea1_initial tea_transfer external_tea in
  let sugar2_after := cup2_after_transfer sugar2_initial tea_transfer in
  let tea_back := 1 in  -- as calculated
  let sugar_back := 2 in  -- as calculated
  let final_tea := final_composition_tea tea1_after tea_back in
  let final_sugar := final_composition_sugar sugar_back in
  let total_liquid := total_liquid final_tea final_sugar in
  sugar_fraction final_sugar total_liquid = 1 / 4 :=
by
  intros
  sorry

end sugar_fraction_in_cup1_is_one_fourth_l275_275357


namespace cost_to_replace_and_install_l275_275209

theorem cost_to_replace_and_install (s l : ℕ) 
  (h1 : l = 3 * s) (h2 : 2 * s + 2 * l = 640) 
  (cost_per_foot : ℕ) (cost_per_gate : ℕ) (installation_cost_per_gate : ℕ) 
  (h3 : cost_per_foot = 5) (h4 : cost_per_gate = 150) (h5 : installation_cost_per_gate = 75) : 
  (s * cost_per_foot + 2 * (cost_per_gate + installation_cost_per_gate)) = 850 := 
by 
  sorry

end cost_to_replace_and_install_l275_275209


namespace intersection_A_B_l275_275322

-- Definition of set A
def A := { x : ℤ | ∃ k : ℕ, x = 3 * k + 1 }

-- Definition of set B
def B := { x : ℚ | x ≤ 7 }

-- Proof statement
theorem intersection_A_B :
  A ∩ B = {1, 4, 7} :=
sorry

end intersection_A_B_l275_275322


namespace ratio_third_second_l275_275803

theorem ratio_third_second (k : ℝ) (x y z : ℝ) (h1 : y = 4 * x) (h2 : x = 18) (h3 : z = k * y) (h4 : (x + y + z) / 3 = 78) :
  z = 2 * y :=
by
  sorry

end ratio_third_second_l275_275803


namespace find_t_l275_275338

theorem find_t (t : ℝ) (h : ∀x y : ℝ, (x^2 / t^2 + y^2 / (5 * t) = 1) ∧ (2 * real.sqrt 6 = 2 * real.sqrt (t^2 - 5 * t))) : 
  t = 2 ∨ t = 3 ∨ t = 6 :=
sorry

end find_t_l275_275338


namespace factor_expression_l275_275617

theorem factor_expression (y : ℝ) : 16 * y^3 + 8 * y^2 = 8 * y^2 * (2 * y + 1) :=
by
  sorry

end factor_expression_l275_275617


namespace probability_of_both_occurring_l275_275352

theorem probability_of_both_occurring
  (A B : Event)
  (h_indep : independent A B)
  (h_neither_occurs : P[¬A ∧ ¬B] = 1 / 9) :
  ∃ (p : ℚ), p = 2 / 9 ∧ P[A ∧ B] = p :=
by
  sorry

end probability_of_both_occurring_l275_275352


namespace find_quadratic_function_l275_275675

def quad_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_quadratic_function : ∃ (a b c : ℝ), 
  (∀ x : ℝ, quad_function a b c x = 2 * x^2 + 4 * x - 1) ∧ 
  (quad_function a b c (-1) = -3) ∧ 
  (quad_function a b c 1 = 5) :=
sorry

end find_quadratic_function_l275_275675


namespace max_pairs_l275_275052

open Set

theorem max_pairs (A B : Set ℤ) (hA : A.card = 2000) (hB : B.card = 2016) : 
  (∃ K : ℕ, K = 3015636 ∧ ∀ m ∈ A, ∀ n ∈ B, |m - n| ≤ 1000 → K = ∑ m ∈ A, ∑ n ∈ B, if |m - n| ≤ 1000 then 1 else 0) :=
sorry

end max_pairs_l275_275052


namespace problem_1_problem_2_l275_275316

-- Given functions f(x) = x^3 - x and g(x) = x^2 + a
def f (x : ℝ) : ℝ := x ^ 3 - x
def g (x : ℝ) (a : ℝ) : ℝ := x ^ 2 + a

-- (1) If x1 = -1, prove the value of a is 3 given the tangent line condition.
theorem problem_1 (a : ℝ) : 
  let x1 := -1 in
  let f' (x : ℝ) : ℝ := 3 * x ^ 2 - 1 in
  let g' (x : ℝ) : ℝ := 2 * x in
  f' x1 = 2 →  -- Slope of the tangent line at x1 = -1 for f(x)
  g' 1 = 2 →  -- Slope of the tangent line at x2 = 1 for g(x)
  (f x1 + (2 * (1 + 1)) = g 1 a) → -- The tangent line condition
  a = 3 := 
sorry

-- (2) Find the range of values for a under the tangent line condition.
theorem problem_2 (a : ℝ) : 
  let h (x1 : ℝ) := 9 * x1 ^ 4 - 8 * x1 ^ 3 - 6 * x1 ^ 2 + 1 in
   -- We check that the minimum value is h(1) = -4, which gives a range for a
  (∃ x1, x1 ≠ 1 ∧ a ≥ (-1)) :=
sorry

end problem_1_problem_2_l275_275316


namespace line_ellipse_intersection_product_l275_275494

-- Definitions based on conditions
def line_polar (ρ θ : ℝ) : ℝ := ρ * cos θ + ρ * sin θ - 1

def ellipse_param (θ : ℝ) : ℝ × ℝ :=
  (2 * cos θ, sin θ)

-- Proof goal
theorem line_ellipse_intersection_product (θ : ℝ) :
  (x : ℝ) (y : ℝ), x = 1 → y = 0 →
  ∃ t₁ t₂ : ℝ, (x = 1 + (sqrt 2) / 2 * t) ∧ (y = (sqrt 2) / 2 * t) ∧ 
    (∀ (x y : ℝ), x = 2 * cos θ → y = sin θ → x^2 + 4 * y^2 = 4) → (|t₁ * t₂| = 6 / 5) :=
begin
  sorry
end

end line_ellipse_intersection_product_l275_275494


namespace total_animals_l275_275377

-- Define the number of pigs and giraffes
def num_pigs : ℕ := 7
def num_giraffes : ℕ := 6

-- Theorem stating the total number of giraffes and pigs
theorem total_animals : num_pigs + num_giraffes = 13 :=
by sorry

end total_animals_l275_275377


namespace remaining_apples_l275_275834

-- Define the initial number of apples
def initialApples : ℕ := 356

-- Define the number of apples given away as a mixed number converted to a fraction
def applesGivenAway : ℚ := 272 + 3/5

-- Prove that the remaining apples after giving away are 83
theorem remaining_apples
  (initialApples : ℕ)
  (applesGivenAway : ℚ) :
  initialApples - applesGivenAway = 83 := 
sorry

end remaining_apples_l275_275834


namespace part1_part2_l275_275308

section
variable (f g : ℝ → ℝ) (a : ℝ)

-- Define the given functions.
noncomputable def f := λ x : ℝ, x^3 - x
noncomputable def g := λ x : ℝ, x^2 + a

-- Part 1: Prove that if x₁ = -1, then a = 3
theorem part1 (x₁ : ℝ) (h : x₁ = -1) (h_tangent : ∀ x₃ : ℝ, f'(x₃) = g'(x₃)) : 
    a = 3 :=
sorry

-- Part 2: Prove the range of values for a is [-1, +∞).
theorem part2 (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, 3 * x₁^2 - 1 = 2 * x₂ ∧ a = x₂^2 - 2 * x₁^3) : 
    a ∈ Set.Ici (-1) :=
sorry

end

end part1_part2_l275_275308


namespace total_tiles_is_1352_l275_275588

noncomputable def side_length_of_floor := 39

noncomputable def total_tiles_covering_floor (n : ℕ) : ℕ :=
  (n ^ 2) - ((n / 3) ^ 2)

theorem total_tiles_is_1352 :
  total_tiles_covering_floor side_length_of_floor = 1352 := by
  sorry

end total_tiles_is_1352_l275_275588


namespace digits_making_number_divisible_by_4_l275_275202

theorem digits_making_number_divisible_by_4 (N : ℕ) (hN : N < 10) :
  (∃ n0 n4 n8, n0 = 0 ∧ n4 = 4 ∧ n8 = 8 ∧ N = n0 ∨ N = n4 ∨ N = n8) :=
by
  sorry

end digits_making_number_divisible_by_4_l275_275202


namespace circles_tangent_internally_l275_275124

-- Define the structures of the circles
def circle1 : Set (ℝ × ℝ) := { p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1 }
def circle2 : Set (ℝ × ℝ) := { p | (p.1 - 4)^2 + (p.2 - 3)^2 = 9 }

-- Prove that the circles are tangent internally
theorem circles_tangent_internally :
  ∃ d : ℝ, d = 2 ∧ 
  (d = real.sqrt ((4 - 2) ^ 2 + (3 - 3) ^ 2)) ∧ 
  d = (3 - 1) :=
sorry

end circles_tangent_internally_l275_275124


namespace number_of_possible_denominators_l275_275100

noncomputable def digit : Type := { n : ℕ // n < 10 }

def repeating_decimal (c d : digit) : ℚ :=
  let num := 10 * c.val + d.val in
  num / 99

def valid_denominator (denom : ℕ) : Prop :=
  denom = 3 ∨ denom = 9 ∨ denom = 11 ∨ denom = 33 ∨ denom = 99

lemma denominators_of_repeating_decimal (c d : digit) (h : ¬(c.val = 8 ∧ d.val = 8) ∧ ¬(c.val = 0 ∧ d.val = 0)) :
  ∃ denom, valid_denominator denom ∧ (repeating_decimal c d).denom = denom :=
sorry

theorem number_of_possible_denominators :
  ∃ n, n = 5 ∧ ∀ (c d : digit), ¬(c.val = 8 ∧ d.val = 8) ∧ ¬(c.val = 0 ∧ d.val = 0) → 
  ∃ denom, valid_denominator denom ∧ (repeating_decimal c d).denom = denom :=
sorry

end number_of_possible_denominators_l275_275100


namespace complex_with_real_part_5_and_magnitude_5_l275_275828

theorem complex_with_real_part_5_and_magnitude_5 :
  ∃ (z : ℂ), z.re = 5 ∧ complex.abs z = 5 :=
sorry

end complex_with_real_part_5_and_magnitude_5_l275_275828


namespace min_value_2013_Quanzhou_simulation_l275_275896

theorem min_value_2013_Quanzhou_simulation:
  ∃ (x y : ℝ), (x - y - 1 = 0) ∧ (x - 2) ^ 2 + (y - 2) ^ 2 = 1 / 2 :=
by
  use 2
  use 3
  sorry

end min_value_2013_Quanzhou_simulation_l275_275896


namespace division_remainder_l275_275178

theorem division_remainder (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (hrem : x % y = 3) (hdiv : (x : ℚ) / y = 96.15) : y = 20 :=
sorry

end division_remainder_l275_275178


namespace sqrt_v_ge_tau_l275_275631

open Finset

variable {n : ℕ}
def permutations (n : ℕ) := Equiv.Perm (Fin n)
noncomputable def S := univ : Finset (permutations n)
def good (T : Set (permutations n)) : Prop :=
  ∀ σ ∈ S, ∃ t1 t2 ∈ T, σ = t1.trans t2
def extremely_good (U : Set (permutations n)) : Prop :=
  ∀ σ ∈ S, ∃ s ∈ S, ∃ u ∈ U, σ = s.symm.trans u.trans s
noncomputable def tau := Inf { |T : Set (permutations n)| / S.card | good T }
noncomputable def v := Inf { |U : Set (permutations n)| / S.card | extremely_good U }

theorem sqrt_v_ge_tau : Real.sqrt v ≥ tau := sorry

end sqrt_v_ge_tau_l275_275631


namespace factorial_addition_l275_275241

theorem factorial_addition : 
  (Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10)) + 
  (Nat.factorial 11 / (Nat.factorial 6 * Nat.factorial 5)) = 1190 := 
by 
  sorry

end factorial_addition_l275_275241


namespace find_lambda_l275_275340

noncomputable def f (x y z : ℝ) : ℝ := real.root 5 (x + 1) + real.root 5 (y + 1) + real.root 5 (z + 1)

theorem find_lambda (n : ℕ) (hn : 2 ≤ n) : ∃ (λ : ℝ), (λ = 2 + real.root n 5) ∧ (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x + y + z = 4 → f x y z > λ) :=
by
  use 2 + real.root n 5
  intros x y z hx hy hz hxyz

  -- sorry placeholder for the proof
  sorry

end find_lambda_l275_275340


namespace largest_packet_size_gcd_l275_275024

theorem largest_packet_size_gcd:
    ∀ (n1 n2 : ℕ), n1 = 36 → n2 = 60 → Nat.gcd n1 n2 = 12 :=
by
  intros n1 n2 h1 h2
  -- Sorry is added because the proof is not required as per the instructions
  sorry

end largest_packet_size_gcd_l275_275024


namespace total_pears_after_giving_away_l275_275218

def alyssa_pears : ℕ := 42
def nancy_pears : ℕ := 17
def carlos_pears : ℕ := 25
def pears_given_away_per_person : ℕ := 5

theorem total_pears_after_giving_away :
  (alyssa_pears + nancy_pears + carlos_pears) - (3 * pears_given_away_per_person) = 69 :=
by
  sorry

end total_pears_after_giving_away_l275_275218


namespace initial_books_l275_275513

theorem initial_books (total_books_now : ℕ) (books_added : ℕ) (initial_books : ℕ) :
  total_books_now = 48 → books_added = 10 → initial_books = total_books_now - books_added → initial_books = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end initial_books_l275_275513


namespace find_a2_l275_275694

-- Definitions from the conditions
def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) := ∀ n, a (n + 1) = q * a n
def sum_geom_seq (a : ℕ → ℕ) (q : ℕ) (n : ℕ) := (a 0 * (1 - q^(n + 1))) / (1 - q)

-- Given conditions
def a_n : ℕ → ℕ := sorry -- Define the sequence a_n
def q : ℕ := 2
def S_4 := 60

-- The theorem to be proved
theorem find_a2 (h1: is_geometric_sequence a_n q)
                (h2: sum_geom_seq a_n q 3 = S_4) : 
                a_n 1 = 8 :=
sorry

end find_a2_l275_275694


namespace sum_first_10_terms_l275_275008

-- Define the sequence
def a : ℕ → ℤ
| 0     := 0 -- We assume that a_0 is 0 as a convention since the sequence starts from n=1
| (n+1) := if n = 0 then -2 else 1 + 2 * a n

-- Define the sum of the first 10 terms of the sequence
def S₁₀ : ℤ := (Finset.range 10).sum (λ n, a (n+1))

-- The theorem to be proved
theorem sum_first_10_terms : S₁₀ = -1033 :=
by
  sorry -- Proof goes here

end sum_first_10_terms_l275_275008


namespace transformed_variance_l275_275926

variable {x : Fin 2023 → ℝ} (a b : ℝ)
variable (h_sorted : ∀ i j, i < j → x i < x j)

def mean (x : Fin 2023 → ℝ) : ℝ := (∑ i, x i) / 2023

def variance (x : Fin 2023 → ℝ) : ℝ :=
  (∑ i, (x i - mean x)^2) / 2023

theorem transformed_variance (s : ℝ) (hx : variance x = s^2) :
  variance (λ i, a * x i + b) = a^2 * s^2 :=
sorry

end transformed_variance_l275_275926


namespace sebastian_deducted_salary_correct_l275_275813

def weekly_salary : ℝ := 1043
def workdays_in_week : ℕ := 5
def absent_days : ℕ := 2

def daily_wage (weekly_salary : ℝ) (workdays_in_week : ℕ) := weekly_salary / workdays_in_week
def deduction (daily_wage : ℝ) (absent_days : ℕ) := daily_wage * absent_days
def deducted_salary (weekly_salary : ℝ) (deduction : ℝ) := weekly_salary - deduction

theorem sebastian_deducted_salary_correct : 
  deducted_salary weekly_salary (deduction (daily_wage weekly_salary workdays_in_week) absent_days) = 625.80 := by
  sorry

end sebastian_deducted_salary_correct_l275_275813


namespace polygon_has_4_sides_l275_275063

noncomputable def T (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ a ≤ x ∧ x ≤ 3 * a ∧ a ≤ y ∧ y ≤ 3 * a ∧ x + y ≥ 2 * a ∧ y ≤ x + 2 * a ∧ x ≤ y + 2 * a ∧ x + y ≤ 4 * a

theorem polygon_has_4_sides (a : ℝ) (h : a > 0) :
  ∃ (vertices : list (ℝ × ℝ)), list.length vertices = 4 ∧
  ∀ (x y : ℝ), T a x y ↔ (x, y) ∈ vertices.to_finset.convex_hull :=
sorry

end polygon_has_4_sides_l275_275063


namespace ordered_pairs_count_l275_275999

def number_of_ordered_pairs (n : ℕ) : ℕ :=
  let factors := n.factors.count
  (List.map (λ pair : ℕ, 2 * pair + 1) (List.range factors)).prod

theorem ordered_pairs_count (n : ℕ) (h : 0 < n) :
  ∃ (x y : ℕ), (0 < x ∧ 0 < y ∧ (x * y) / (x + y) = n) ↔
  number_of_ordered_pairs n = (∏ i in range (n.factors.count), 2 * (n.factors.nth i).get_or_else 0 + 1) :=
by
  sorry

end ordered_pairs_count_l275_275999


namespace truck_transport_time_l275_275141

-- Define the conditions as variables
variables (x y : ℝ) (t₁ : ℝ)
-- Total cargo transported in 6 hours by both trucks
# The fraction of cargo each truck can transport in one hour
-- First truck transported 3/5 of cargo before second truck arrived
-- Remaining 2/5 of the cargo was transported in 12 hours
axioms
  (h1 : x + y = 1 / 6)
  (h2 : t₁ * x = 3 / 5)
  (h3 : (12 - t₁) * (x + y) = 2 / 5)

theorem truck_transport_time :
  (x = 1 / 16 ∧ y = 5 / 48) ∨ (x = 1 / 12 ∧ y = 1 / 12) :=
sorry

end truck_transport_time_l275_275141


namespace powers_of_2_count_l275_275628

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n ^ 2 + 2

def iterate_rest_2 (n : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate f k n) = 2

def number_of_satisfying_numbers : Nat :=
  Finset.card (Finset.filter iterate_rest_2 (Finset.range 201))

theorem powers_of_2_count : number_of_satisfying_numbers = 7 :=
  sorry

end powers_of_2_count_l275_275628


namespace probability_empty_chair_on_sides_7_chairs_l275_275043

theorem probability_empty_chair_on_sides_7_chairs :
  (1 : ℚ) / (35 : ℚ) = 0.2 := by
  sorry

end probability_empty_chair_on_sides_7_chairs_l275_275043


namespace smallest_value_l275_275251

noncomputable def Q : Polynomial ℝ := X^4 - 2*X^3 + 3*X^2 - 4*X + 5

theorem smallest_value :
  let q1 := Q.eval 1,
      prod_zeros := Q.leading_coeff * Q.coeff 0, -- Using Vieta's formulas
      sum_coeff := Q.coeffs.sum in
  ∃ (p : ℝ), p = min (min q1 (prod_zeros) (sum_coeff)) 
  → p = the_product_of_non_real_zeros Q
:= sorry

end smallest_value_l275_275251


namespace age_of_15th_student_l275_275891

theorem age_of_15th_student 
  (avg_age_all : ℕ → ℕ → ℕ)
  (avg_age : avg_age_all 15 15 = 15)
  (avg_age_4 : avg_age_all 4 14 = 14)
  (avg_age_10 : avg_age_all 10 16 = 16) : 
  ∃ age15 : ℕ, age15 = 9 := 
by
  sorry

end age_of_15th_student_l275_275891


namespace find_position_of_term_l275_275348

theorem find_position_of_term :
  ∃ n : ℕ, (n ≥ 1) ∧ ((λ n, (sqrt (2 + (n - 1) * 3))) n = 2 * sqrt 5 ∧ n = 7) :=
by { sorry }

end find_position_of_term_l275_275348


namespace find_abc_sum_l275_275419

-- Definitions and statements directly taken from conditions
def Q1 (x y : ℝ) : Prop := y = x^2 + 51/50
def Q2 (x y : ℝ) : Prop := x = y^2 + 23/2
def common_tangent_rational_slope (a b c : ℤ) : Prop :=
  ∃ (x y : ℝ), (a * x + b * y = c) ∧ (Q1 x y ∨ Q2 x y)

theorem find_abc_sum :
  ∃ (a b c : ℕ), 
    gcd (a) (gcd (b) (c)) = 1 ∧
    common_tangent_rational_slope (a) (b) (c) ∧
    a + b + c = 9 :=
  by sorry

end find_abc_sum_l275_275419


namespace trajectory_equation_l275_275789

variable (m x y : ℝ)
def a := (m * x, y + 1)
def b := (x, y - 1)
def is_perpendicular (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0

theorem trajectory_equation 
  (h1: is_perpendicular (a m x y) (b x y)) : 
  m * x^2 + y^2 = 1 :=
sorry

end trajectory_equation_l275_275789


namespace complex_number_in_forth_quadrant_l275_275484

theorem complex_number_in_forth_quadrant 
  (z : ℂ) 
  (h : (z - 3) * (2 - I) = 5 * I) :
  z.conj.re > 0 ∧ z.conj.im < 0 :=
sorry

end complex_number_in_forth_quadrant_l275_275484


namespace equal_elevation_YAxis_or_Circle_l275_275863

noncomputable def equal_elevation_set (h k a : ℝ) : Set (ℝ × ℝ) :=
  if h = k then {p | p.1 = 0} else 
    {p | (p.1^2 + p.2^2 + 2 * a * p.1 * ((k^2 + h^2) / (k^2 - h^2)) + a^2 = 0)}

theorem equal_elevation_YAxis_or_Circle (h k a : ℝ) :
  equal_elevation_set h k a =
  if h = k then {p | p.1 = 0} else 
    {p | (p.1^2 + p.2^2 + 2 * a * p.1 * ((k^2 + h^2) / (k^2 - h^2)) + a^2 = 0)} :=
begin
  sorry
end

end equal_elevation_YAxis_or_Circle_l275_275863


namespace completing_the_square_transformation_l275_275540

theorem completing_the_square_transformation (x : ℝ) : 
  (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro h
  -- Transformation steps will be shown in the proof
  sorry

end completing_the_square_transformation_l275_275540


namespace length_PF_eq_16div3_l275_275858

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * (x + 2)

-- Define the slope of the line through focus making an angle of 60 degrees
def slope (theta : ℝ) : ℝ := Real.tan theta
def theta_60 : ℝ := Real.pi / 3  -- 60 degrees in radians

-- Define the line equation passing through the focus (0, 0) with slope tan(60)
def line (x y : ℝ) : Prop := y = slope theta_60 * x

-- Define the intersection of line and parabola at points A and B
def points_A_B := {p : ℝ × ℝ // parabola p.1 p.2 ∧ line p.1 p.2}

-- Define the bisector of the chord AB intersecting the x-axis to find point P
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the perpendicular bisector intersecting x-axis
def perpendicular_bisector_x (mid : ℝ × ℝ) : ℝ :=
  mid.1 - (mid.2 * slope theta_60⁻¹)

-- Define the intersection point P on the x-axis
def point_P (mid : ℝ × ℝ) : ℝ × ℝ := 
  (perpendicular_bisector_x mid, 0)

-- Define the focus F
def focus : ℝ × ℝ := (0, 0)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Final theorem to prove PF = 16/3
theorem length_PF_eq_16div3 {A B: ℝ × ℝ}
  (hA: parabola A.1 A.2 ∧ line A.1 A.2)
  (hB: parabola B.1 B.2 ∧ line B.1 B.2) :
  let mid := midpoint A B in
  let P := point_P mid in
  distance P focus = 16 / 3 := 
  sorry

end length_PF_eq_16div3_l275_275858


namespace find_initial_games_l275_275607

variable (x : ℝ)    -- Initial number of games Bill played
variable (won_initial_percentage : ℝ := 0.63)  
variable (additional_games : ℝ := 100)
variable (lost_additional_games : ℝ := 43)
variable (new_win_percentage : ℝ := 0.61)

def bill_initial_games (x : ℝ) : Prop := 
  let won_initial := won_initial_percentage * x
  let won_additional := additional_games - lost_additional_games
  let total_games := x + additional_games
  let total_won := won_initial + won_additional
  total_won / total_games = new_win_percentage

theorem find_initial_games : bill_initial_games 200 :=
by
  let won_initial := 0.63 * 200
  let won_additional := 100 - 43
  let total_games := 200 + 100
  let total_won := won_initial + won_additional
  have h_eq : total_won / total_games = 0.61
  -- insert the actual proof steps here
  sorry

end find_initial_games_l275_275607


namespace possible_analytical_expression_for_f_l275_275474

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x)) + (Real.cos (2 * x))

theorem possible_analytical_expression_for_f :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ x : ℝ, f (x - π/4) = f (-x)) ∧
  (∀ x : ℝ, π/8 < x ∧ x < π/2 → f x < f (x - 1)) :=
by
  sorry

end possible_analytical_expression_for_f_l275_275474


namespace minesweeper_probability_l275_275590

-- Definitions for the conditions
def grid_size := (9, 9)
def total_squares := grid_size.1 * grid_size.2
def total_mines := total_squares / 3
def center_square := (4, 4)
def above_center_square := (3, 4)
def below_center_square := (5, 4)

-- Events
def C : Prop := "The center square is a mine"
def A : Prop := "The square above the center shows the number 4"
def B : Prop := "The square below the center shows the number 1"

-- The main statement we want to prove
theorem minesweeper_probability : 
  (Pr(C | A ∧ B) = 88 / 379) := sorry

end minesweeper_probability_l275_275590


namespace find_weight_l275_275217

-- Define the weight of each box before taking out 20 kg as W
variable (W : ℚ)

-- Define the condition given in the problem
def condition : Prop := 7 * (W - 20) = 3 * W

-- The proof goal is to prove W = 35 under the given condition
theorem find_weight (h : condition W) : W = 35 := by
  sorry

end find_weight_l275_275217


namespace max_value_of_f_l275_275375

noncomputable def f (x a b : ℝ) := (1 - x ^ 2) * (x ^ 2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (x : ℝ) 
  (h1 : (∀ x, f x 8 15 = (1 - x^2) * (x^2 + 8*x + 15)))
  (h2 : ∀ x, f x a b = f (-(x + 4)) a b) :
  ∃ m, (∀ x, f x 8 15 ≤ m) ∧ m = 16 :=
by
  sorry

end max_value_of_f_l275_275375


namespace total_tax_in_cents_l275_275591

-- Declare the main variables and constants
def wage_per_hour_cents : ℕ := 2500
def local_tax_rate : ℝ := 0.02
def state_tax_rate : ℝ := 0.005

-- Define the total tax calculation as a proof statement
theorem total_tax_in_cents :
  local_tax_rate * wage_per_hour_cents + state_tax_rate * wage_per_hour_cents = 62.5 :=
by sorry

end total_tax_in_cents_l275_275591


namespace ratio_of_neighborhood_to_gina_l275_275993

variable (Gina_bags : ℕ) (Weight_per_bag : ℕ) (Total_weight_collected : ℕ)

def neighborhood_to_gina_ratio (Gina_bags : ℕ) (Weight_per_bag : ℕ) (Total_weight_collected : ℕ) := 
  (Total_weight_collected - Gina_bags * Weight_per_bag) / (Gina_bags * Weight_per_bag)

theorem ratio_of_neighborhood_to_gina 
  (h₁ : Gina_bags = 2) 
  (h₂ : Weight_per_bag = 4) 
  (h₃ : Total_weight_collected = 664) :
  neighborhood_to_gina_ratio Gina_bags Weight_per_bag Total_weight_collected = 82 := 
by 
  sorry

end ratio_of_neighborhood_to_gina_l275_275993


namespace slices_left_after_all_l275_275937

def total_cakes : ℕ := 2
def total_slices_per_cake : ℕ := 8
def fraction_given_to_friends : ℝ := 1 / 4
def fraction_given_to_family : ℝ := 1 / 3
def slices_eaten_by_alex : ℕ := 3

theorem slices_left_after_all :
  let total_slices := total_cakes * total_slices_per_cake in
  let slices_given_to_friends := (fraction_given_to_friends * total_slices) in
  let slices_left_after_friends := (total_slices - slices_given_to_friends.to_nat) in
  let slices_given_to_family := (fraction_given_to_family * slices_left_after_friends) in
  let slices_left_after_family := (slices_left_after_friends - slices_given_to_family.to_nat) in
  let slices_left_after_alex := (slices_left_after_family - slices_eaten_by_alex) in
  slices_left_after_alex = 5 :=
by
  sorry

end slices_left_after_all_l275_275937


namespace math_problem_l275_275643

theorem math_problem (a b c : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 4) :
  (a^b)^a - (b^a)^b) * c = 2660 := by
  sorry

end math_problem_l275_275643


namespace reflection_of_BC_tangent_to_circumcircle_APQ_l275_275413

open EuclideanGeometry

-- Definitions of the points and angles involved
variables {A B C O P Q : Point}
variables {ABC_circumcircle : Circle}
variables (hABC : Triangle ABC)
variables (hO : IsCircumcenter O hABC)
variables (hP : PointOnLine P (LineThrough A B))
variables (hQ : PointOnLine Q (LineThrough A C))
variables (hBOAngle : AngleAtPoint B O P = AngleAtPoint B A C)
variables (hCOAngle : AngleAtPoint C O Q = AngleAtPoint C A B)
variables (hReflectionBC : Line P Q)

-- Statement of the problem
theorem reflection_of_BC_tangent_to_circumcircle_APQ :
  isTangent (reflection hReflectionBC) (circumcircle (Triangle A P Q)) :=
  sorry

end reflection_of_BC_tangent_to_circumcircle_APQ_l275_275413


namespace maximum_height_l275_275903

-- Define the elevation function
def elevation (t : ℝ) : ℝ := 30 + 180 * t - 20 * t^2

-- The proof that the maximum height reached by the ball is 435 feet
theorem maximum_height : ∃ t_max : ℝ, elevation t_max = 435 :=
by
  -- Define the vertex of the parabola as the time to reach maximum height
  let t_max := - (180 / (2 * (-20)))
  use t_max
  -- Calculate the elevation at t_max and verify it equals 435
  calc 
    elevation t_max = 30 + 180 * t_max - 20 * t_max^2 : by rfl
                ... = 30 + 180 * 4.5 - 20 * (4.5)^2 : by sorry
                ... = 435 : by sorry

end maximum_height_l275_275903


namespace units_digit_3_2017_plus_5_2017_l275_275242

theorem units_digit_3_2017_plus_5_2017 :
  let units_digit_3 : ℕ → ℕ := λ n, [3, 9, 7, 1].nth! (n % 4)
  let units_digit_5 : ℕ → ℕ := λ _, 5
  in (units_digit_3 2017 + units_digit_5 2017) % 10 = 8 :=
by
  let units_digit_3 : ℕ → ℕ := λ n, [3, 9, 7, 1].nth! (n % 4)
  let units_digit_5 : ℕ → ℕ := λ _, 5
  have units_digit_3_2017 : units_digit_3 2017 = 3 := by
    sorry
  have units_digit_5_2017 : units_digit_5 2017 = 5 := by
    sorry
  calc
    (units_digit_3 2017 + units_digit_5 2017) % 10
    = (3 + 5) % 10   : by rw [units_digit_3_2017, units_digit_5_2017]
    ... = 8          : by norm_num

end units_digit_3_2017_plus_5_2017_l275_275242


namespace trains_cross_time_l275_275140

theorem trains_cross_time (length1 length2 : ℕ) (speed1 speed2 : ℕ) :
  length1 = 100 → length2 = 200 → 
  speed1 = 100 → speed2 = 200 → 
  (length1 + length2) / ((speed1 + speed2) * (1000/3600)) = 3.6 :=
by
  intros h_len1 h_len2 h_speed1 h_speed2
  sorry

end trains_cross_time_l275_275140


namespace number_of_checkpoints_l275_275901

/-- Definitions based on conditions in the problem --/
def marathon_length : ℕ := 26
def first_checkpoint_distance : ℕ := 1
def last_checkpoint_distance : ℕ := 1
def checkpoint_spacing : ℕ := 6

/-- Theorem stating the number of checkpoints in the marathon --/
theorem number_of_checkpoints : 
  marathon_length - first_checkpoint_distance - last_checkpoint_distance = 24 
  → 
  24 / checkpoint_spacing = 4 
  → 
  4 + 1 = 5
:=
by
  intros h1 h2
  rw [h1] at h2
  have : 24 = marathon_length - first_checkpoint_distance - last_checkpoint_distance := h1
  have : 4 = 24 / checkpoint_spacing := h2
  sorry

end number_of_checkpoints_l275_275901


namespace angle_triangle_l275_275556

theorem angle_triangle (angle1 angle2 : ℝ) (h1 : angle1 = 48) (h2 : angle2 = 52) :
  let angle3 := 180 - angle1 - angle2 in angle3 = 80 ∧ angle3 < 90 :=
by
  -- Replace the following line with a formal proof
  sorry

end angle_triangle_l275_275556


namespace question_answer_l275_275849

-- Define a condition for a three-digit palindrome
def is_three_digit_palindrome (n : ℕ) : Prop :=
  n > 99 ∧ n < 1000 ∧ (let s := n.digits 10 in s = s.reverse)

-- Define the given condition as a predicate
def product_is_698896 (a b : ℕ) : Prop :=
  a * b = 698896

-- Define the final sum to be proved
def sum_of_palindromes (a b : ℕ) : ℕ :=
  a + b

-- State the main theorem to be proved
theorem question_answer : ∃ a b : ℕ, is_three_digit_palindrome a ∧ is_three_digit_palindrome b ∧ product_is_698896 a b ∧ sum_of_palindromes a b = 1672 :=
sorry

end question_answer_l275_275849


namespace complement_set_example_l275_275710

open Set

variable (U M : Set ℕ)

def complement (U M : Set ℕ) := U \ M

theorem complement_set_example :
  (U = {1, 2, 3, 4, 5, 6}) → 
  (M = {1, 3, 5}) → 
  (complement U M = {2, 4, 6}) := by
  intros hU hM
  rw [complement, hU, hM]
  sorry

end complement_set_example_l275_275710


namespace trigonometric_expression_simplifies_to_neg_one_l275_275954

theorem trigonometric_expression_simplifies_to_neg_one :
  (1 - (1 / Real.cos (30 * Real.pi / 180))) * 
  (1 + (1 / Real.sin (60 * Real.pi / 180))) * 
  (1 - (1 / Real.sin (30 * Real.pi / 180))) * 
  (1 + (1 / Real.cos (60 * Real.pi / 180))) = -1 :=
by
  -- Use known values of trigonometric functions
  have h1 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := by sorry
  have h2 : Real.sin (30 * Real.pi / 180) = 1 / 2 := by sorry
  have h3 : Real.cos (60 * Real.pi / 180) = 1 / 2 := by sorry
  have h4 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := by sorry

  -- Use these values to show the expression simplifies to -1
  calc
    (1 - (1 / Real.cos (30 * Real.pi / 180))) * 
    (1 + (1 / Real.sin (60 * Real.pi / 180))) * 
    (1 - (1 / Real.sin (30 * Real.pi / 180))) * 
    (1 + (1 / Real.cos (60 * Real.pi / 180))) = -1 : sorry

end trigonometric_expression_simplifies_to_neg_one_l275_275954


namespace range_of_m_l275_275421

noncomputable def f (x : ℝ) := Real.exp x * (x - 1)
noncomputable def g (m x : ℝ) := m * x

theorem range_of_m :
  (∀ x₁ ∈ Set.Icc (-2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (1 : ℝ) 2, f x₁ > g m x₂) ↔ m ∈ Set.Iio (-1/2 : ℝ) :=
sorry

end range_of_m_l275_275421


namespace maximum_triangular_sections_l275_275026

theorem maximum_triangular_sections :
  ∀ (P : Type) [polygon P] (h_convex : convex P) (h_12gon : P.n = 12) 
  (lanterns : ∀ v ∈ vertices P, lantern v)
  (additional_lanterns : finite_set_interior_lanterns P)
  (triangular_sections : ∀ t ∈ triangles P, t.vertices ⊆ (vertices P) ∪ additional_lanterns),
  (∀ l ∈ additional_lanterns, 
     l.illuminates_at_least_N_triangles l t 6) 
  → P.number_of_triangular_sections ≤ 24 := 
sorry

end maximum_triangular_sections_l275_275026


namespace curve_C2_param_eq_correct_line_l1_eq_correct_l275_275007

noncomputable def curve_C1_polar_eq (θ : ℝ) : ℝ := 2
noncomputable def curve_C1_cartesian_eq (x y : ℝ) : Prop := x^2 + y^2 = 4
noncomputable def curve_C2_transformed_eq (x y : ℝ) : Prop := (x / 2)^2 + y^2 = 1

noncomputable def curve_C2_param_eq (α : ℝ) : ℝ × ℝ :=
(2 * Real.cos α, Real.sin α)

def point_A := (4 / Real.sqrt 5, 1 / Real.sqrt 5)
def line_l1 := (x y : ℝ) → y = (1 / 4) * x

theorem curve_C2_param_eq_correct :
  ∃ α : ℝ, ∀ x y : ℝ, curve_C2_transformed_eq x y ↔ (x = 2 * Real.cos α ∧ y = Real.sin α) :=
sorry

theorem line_l1_eq_correct :
  ∃ α : ℝ, point_A = (2 * Real.cos α, Real.sin α) → line_l1 (2 * Real.cos α) (Real.sin α) :=
sorry

end curve_C2_param_eq_correct_line_l1_eq_correct_l275_275007


namespace subtraction_result_l275_275239

theorem subtraction_result :
  10_000_000_000_000 - (5_555_555_555_555 * 2) = -1_111_111_111_110 :=
by
  sorry

end subtraction_result_l275_275239


namespace age_problem_l275_275625

theorem age_problem
    (D X : ℕ) 
    (h1 : D = 4 * X) 
    (h2 : D = X + 30) : D = 40 ∧ X = 10 := by
  sorry

end age_problem_l275_275625


namespace root_product_l275_275148

theorem root_product : (Real.sqrt (Real.sqrt 81) * Real.cbrt 27 * Real.sqrt 9 = 27) :=
by
  sorry

end root_product_l275_275148


namespace sunzi_problem_l275_275183

variable (x y : ℝ)

def condition1 := y - x = 4.5
def condition2 := x - y / 2 = 1

theorem sunzi_problem : condition1 x y ∧ condition2 x y ↔ 
  (y - x = 4.5 ∧ x - y / 2 = 1) :=
by
  simp [condition1, condition2]
  sorry

end sunzi_problem_l275_275183


namespace cos_theta_ge_zero_slope_range_l275_275684

noncomputable section

-- Define the conditions
variables (F₁ F₂ P A B N : ℝ × ℝ) (λ μ : ℝ) (k : ℝ)
variable (θ : ℝ)

-- Conditions given in the problem
axiom ellipse_eq : ∀ (x y : ℝ), x^2 + 2*y^2 = 2*λ
axiom lambda_pos : λ > 0
axiom F₁_def : F₁ = (-1, 0)
axiom N_def : N = (-2, 0)
axiom NA_NB : ∀ A B : ℝ × ℝ,  A.y = μ * B.y
axiom μ_range : μ ∈ set.Icc (1/5) (1/3) -- equivalent to \(\mu \in \left[\frac{1}{5}, \frac{1}{3}\right]\)
axiom angle_condition : ∠ F₁ P F₂ = θ

-- Proof problem part (1)
theorem cos_theta_ge_zero : cos θ ≥ 0 :=
sorry

-- Proof problem part (2)
theorem slope_range :
  ∃ k : ℝ, k ∈ set.Icc (sqrt 2 / 6) (1 / 2) :=
sorry

end cos_theta_ge_zero_slope_range_l275_275684


namespace sqrt_product_eq_l275_275955

theorem sqrt_product_eq :
  (16 ^ (1 / 4) : ℝ) * (64 ^ (1 / 2)) = 16 := by
  sorry

end sqrt_product_eq_l275_275955


namespace dusting_days_l275_275601

theorem dusting_days 
    (vacuuming_minutes_per_day : ℕ) 
    (vacuuming_days_per_week : ℕ)
    (dusting_minutes_per_day : ℕ)
    (total_cleaning_minutes_per_week : ℕ)
    (x : ℕ) :
    vacuuming_minutes_per_day = 30 →
    vacuuming_days_per_week = 3 →
    dusting_minutes_per_day = 20 →
    total_cleaning_minutes_per_week = 130 →
    (vacuuming_minutes_per_day * vacuuming_days_per_week + dusting_minutes_per_day * x = total_cleaning_minutes_per_week) →
    x = 2 :=
by
  -- Proof steps go here
  sorry

end dusting_days_l275_275601


namespace trapezoid_area_l275_275493

theorem trapezoid_area (a b H : ℝ) (h_lat1 : a = 10) (h_lat2 : b = 8) (h_height : H = b) : 
∃ S : ℝ, S = 104 :=
by sorry

end trapezoid_area_l275_275493


namespace conditional_probability_of_B2_given_A_l275_275500

theorem conditional_probability_of_B2_given_A :
  let P_B1 := 0.55
  let P_B2 := 0.45
  let P_A_given_B1 := 0.9
  let P_A_given_B2 := 0.98
  let P_A := P_A_given_B1 * P_B1 + P_A_given_B2 * P_B2
  (P_A_given_B2 * P_B2 / P_A) = 0.4712 :=
by
  let P_B1 := 0.55
  let P_B2 := 0.45
  let P_A_given_B1 := 0.9
  let P_A_given_B2 := 0.98
  let P_A := P_A_given_B1 * P_B1 + P_A_given_B2 * P_B2
  have h : P_A = 0.936 := by {
    calc
    P_A = 0.9 * 0.55 + 0.98 * 0.45 : by norm_num
    ... = 0.495 + 0.441 : by norm_num
    ... = 0.936 : by norm_num
  }
  calc
  (P_A_given_B2 * P_B2 / P_A) = (0.98 * 0.45 / 0.936) : by norm_num
  ... = 0.4712 : by norm_num

end conditional_probability_of_B2_given_A_l275_275500


namespace hyperbola_equation_proof_l275_275736

noncomputable def hyperbola_equation (P : ℝ × ℝ) (a b : ℝ) :=
  ∃ λ : ℝ, λ ≠ 0 ∧ (P.1^2 / a^2 - P.2^2 / b^2 = λ ∧ (∀ x y, y = (1 / 3) * x ∨ y = -(1 / 3) * x → (x^2 / a^2 - y^2 / b^2) = λ))

theorem hyperbola_equation_proof :
  hyperbola_equation (6, sqrt 3) 3 1 :=
begin
  unfold hyperbola_equation,
  use 1,
  split,
  { norm_num },
  split,
  { norm_num },  -- proving (6^2 / 3^2 - (sqrt 3)^2 / 1^2 = 1)
  { intros x y h,
    cases h; norm_num }
end

end hyperbola_equation_proof_l275_275736


namespace total_coins_l275_275099

-- Definitions of the quantities and values
def TotalValue : ℝ := 3.10  -- the total value in dollars
def ValueOfDime : ℝ := 0.10  -- the value of one dime in dollars
def ValueOfNickel : ℝ := 0.05  -- the value of one nickel in dollars
def NumberOfDimes : ℕ := 26  -- the number of dimes Steve has

-- Goal: to find the total number of coins
theorem total_coins (TotalValue = 3.10) (ValueOfDime = 0.10) (ValueOfNickel = 0.05) (NumberOfDimes = 26) : 
    TotalValue = NumberOfDimes * ValueOfDime + n * ValueOfNickel → 26 + n = 36 :=
sorry

end total_coins_l275_275099


namespace discount_percentage_proof_l275_275073

noncomputable def cost_price : ℝ := 540
noncomputable def marked_percentage : ℝ := 0.15
noncomputable def selling_price : ℝ := 496.80

def markup (c : ℝ) (m_p : ℝ) : ℝ :=
  m_p * c

def marked_price (c : ℝ) (m : ℝ) : ℝ :=
  c + m

def discount (m_p : ℝ) (s_p : ℝ) : ℝ :=
  m_p - s_p

def discount_percentage (d : ℝ) (m_p : ℝ) : ℝ :=
  (d / m_p) * 100

theorem discount_percentage_proof :
  ∀ (c : ℝ) (m_p : ℝ) (s_p : ℝ),
  c = cost_price →
  m_p = marked_percentage →
  s_p = selling_price →
  let m := markup c m_p in
  let m_p := marked_price c m in
  let d := discount m_p s_p in
  discount_percentage d m_p = 20 :=
by
  intros
  sorry

end discount_percentage_proof_l275_275073


namespace parallelogram_angle_ratio_l275_275055

structure Parallelogram (A B C D : Type*) [EuclideanGeometry] :=
(isParallelogram : ∀ (A B C D : Point), Parallelogram A B C D)

theorem parallelogram_angle_ratio (A B C D E : Point) 
  (hABCD : Parallelogram A B C D)
  (hE : ∃ (P : Point), collinear A D P ∧ collinear B C P ∧ P = E) :
  let S := ∠ C D E + ∠ D C E
  let S' := ∠ B A D + ∠ A B C
  (S / S' = 2) :=
by 
  sorry

end parallelogram_angle_ratio_l275_275055


namespace solution_set_of_inequality_l275_275852

theorem solution_set_of_inequality (x : ℝ) : x * (x + 2) ≥ 0 ↔ x ≤ -2 ∨ x ≥ 0 := 
sorry

end solution_set_of_inequality_l275_275852


namespace sin_two_pi_minus_alpha_eq_half_l275_275728

theorem sin_two_pi_minus_alpha_eq_half
  (α : ℝ)
  (h1 : cos (π + α) = - (real.sqrt 3 / 2))
  (h2 : π < α ∧ α < 2 * π) : 
  sin (2 * π - α) = 1 / 2 :=
by sorry

end sin_two_pi_minus_alpha_eq_half_l275_275728


namespace min_overlap_smartphones_laptops_l275_275076

theorem min_overlap_smartphones_laptops (s l : ℝ) (hs : s = 0.9) (hl : l = 0.8) : ∃ (o : ℝ), o = 0.7 :=
by {
  use 0.7,
  sorry
}

end min_overlap_smartphones_laptops_l275_275076


namespace probability_seven_chairs_probability_n_chairs_l275_275035
-- Importing necessary library to ensure our Lean code can be built successfully

-- Definition for case where n = 7
theorem probability_seven_chairs : 
  let total_seating := 7 * 6 * 5 / 6 
  let favorable_seating := 1 
  let probability := favorable_seating / total_seating 
  probability = 1 / 35 := 
by 
  sorry

-- Definition for general case where n ≥ 6
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) : 
  let total_seating := (n - 1) * (n - 2) / 2 
  let favorable_seating := (n - 4) * (n - 5) / 2 
  let probability := favorable_seating / total_seating 
  probability = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := 
by 
  sorry

end probability_seven_chairs_probability_n_chairs_l275_275035
