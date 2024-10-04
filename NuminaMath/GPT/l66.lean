import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.SquareRoot
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Integration
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Graph.Basic
import Mathlib.Combinatorics.Perm
import Mathlib.Combinatorics.SimpleGraph.Coloring.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Defs
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Polynomial.AlgebraMap
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Group.Perm.Basic
import Mathlib.Init.Data.Int.Basic
import Mathlib.NumberTheory.Primes
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic

namespace weight_loss_months_l66_66665

theorem weight_loss_months (initial_weight : ℕ) (goal_weight : ℕ) (months_passed : ℕ)
  (current_weight : Option ℕ := none) (average_monthly_loss : Option ℕ := none)
  (initial_weight_eq : initial_weight = 222)
  (goal_weight_eq : goal_weight = 190)
  (months_passed_eq : months_passed = 12) :
  current_weight = none ∨ average_monthly_loss = none → 
  ∃ no_solution: ℕ , no_solution = 0 :=
by
  intro h
  exists 0
  sorry

end weight_loss_months_l66_66665


namespace sum_of_odd_divisors_of_90_l66_66086

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l66_66086


namespace points_on_circle_at_distance_l66_66539

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0
def line_eq (x y : ℝ) : Prop := x + y + 1 = 0

noncomputable def distance_point_from_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / (real.sqrt (a^2 + b^2))

theorem points_on_circle_at_distance (n : ℕ) :
  (∀ x y : ℝ, circle_eq x y → (distance_point_from_line x y 1 1 1 = 1) → n = 2)
:= sorry

end points_on_circle_at_distance_l66_66539


namespace hyperbola_focal_length_proof_l66_66275

noncomputable def hyperbola_focal_length (a b : ℝ) (c : ℝ) (focus_dist parabola_p : ℝ) : Prop :=
  a = 2 ∧ b = 1 ∧ focus_dist = 4 ∧ parabola_p = 4 ∧ c = sqrt (a^2 + b^2)

theorem hyperbola_focal_length_proof :
  ∃ (a b c : ℝ), hyperbola_focal_length a b c 4 4 ∧ 2 * c = 2 * sqrt 5 :=
begin
  use [2, 1, sqrt 5],
  split,
  { exact ⟨rfl, rfl, rfl, rfl, rfl⟩ },
  { exact rfl },
end

end hyperbola_focal_length_proof_l66_66275


namespace georgia_makes_muffins_l66_66177

/--
Georgia makes muffins and brings them to her students on the first day of every month.
Her muffin recipe only makes 6 muffins and she has 24 students. 
Prove that Georgia makes 36 batches of muffins in 9 months.
-/
theorem georgia_makes_muffins 
  (muffins_per_batch : ℕ)
  (students : ℕ)
  (months : ℕ) 
  (batches_per_day : ℕ) 
  (total_batches : ℕ)
  (h1 : muffins_per_batch = 6)
  (h2 : students = 24)
  (h3 : months = 9)
  (h4 : batches_per_day = students / muffins_per_batch) : 
  total_batches = months * batches_per_day :=
by
  -- The proof would go here
  sorry

end georgia_makes_muffins_l66_66177


namespace quadratic_function_value_at_point_l66_66613

theorem quadratic_function_value_at_point (a : ℝ) :
  (∃ (P : ℝ × ℝ), P = (1, a) ∧ ∀ (x : ℝ), P.2 = 2 * x^2) → a = 2 :=
by
  intro h
  cases h with P hP
  rw [Prod.ext_iff, and_comm] at hP 
  cases hP with hx ha
  rw hx at ha
  specialize ha 1
  rw mul_one at ha 
  norm_num at ha
  exact ha

end quadratic_function_value_at_point_l66_66613


namespace generalize_numbers_to_ensure_inverses_l66_66220

theorem generalize_numbers_to_ensure_inverses :
  (∀ n1 n2 : ℕ, ∃ (z : ℤ), n1 - n2 = z) ∧
  (∀ n1 n2 : ℤ, n2 ≠ 0 → ∃ (q : ℚ), n1 / n2 = q) ∧
  (∀ r : ℚ, ∃ (s : ℝ), s * s = r) ∧
  (∀ r : ℝ, r < 0 → ∃ (c : ℂ), c * c = r) →
  "The desire to always have inverse operations be executable leads to the generalization of numbers." :=
sorry

end generalize_numbers_to_ensure_inverses_l66_66220


namespace sum_of_positive_odd_divisors_of_90_l66_66075

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l66_66075


namespace martin_rings_big_bell_l66_66694

/-
Problem Statement:
Martin rings the small bell 4 times more than 1/3 as often as the big bell.
If he rings both of them a combined total of 52 times, prove that he rings the big bell 36 times.
-/

theorem martin_rings_big_bell (s b : ℕ) 
  (h1 : s + b = 52) 
  (h2 : s = 4 + (1 / 3 : ℚ) * b) : 
  b = 36 := 
by
  sorry

end martin_rings_big_bell_l66_66694


namespace range_of_a_l66_66356

def P (x : ℝ) : Prop := x^2 - 4 * x - 5 < 0
def Q (x : ℝ) (a : ℝ) : Prop := x < a

theorem range_of_a (a : ℝ) : (∀ x, P x → Q x a) → (∀ x, Q x a → P x) → a ≥ 5 :=
by
  sorry

end range_of_a_l66_66356


namespace algebra_expression_l66_66292

theorem algebra_expression (a b : ℝ) (h : a = b + 1) : 3 + 2 * a - 2 * b = 5 :=
sorry

end algebra_expression_l66_66292


namespace no_more_than_one_100_l66_66751

-- Define the score variables and the conditions
variables (R P M : ℕ)

-- Given conditions: R = P - 3 and P = M - 7
def score_conditions : Prop := R = P - 3 ∧ P = M - 7

-- The maximum score condition
def max_score_condition : Prop := R ≤ 100 ∧ P ≤ 100 ∧ M ≤ 100

-- The goal: it is impossible for Vanya to have scored 100 in more than one exam
theorem no_more_than_one_100 (R P M : ℕ) (h1 : score_conditions R P M) (h2 : max_score_condition R P M) :
  (R = 100 ∧ P = 100) ∨ (P = 100 ∧ M = 100) ∨ (M = 100 ∧ R = 100) → false :=
sorry

end no_more_than_one_100_l66_66751


namespace side_length_significant_digits_square_side_significant_digits_l66_66157

theorem side_length_significant_digits 
  (A : ℝ) 
  (hA : A = 2.3049) : 
  ∃ s : ℝ, s = real.sqrt A ∧ s = 1.5182 :=
sorry

theorem square_side_significant_digits 
  (s : ℝ) 
  (hs : s = 1.5182) : 
  5 = 5 :=
by sorry

end side_length_significant_digits_square_side_significant_digits_l66_66157


namespace find_circumcircle_radius_l66_66022

noncomputable def circumcircle_radius (r1 r2 r3 : ℝ) (a : ℝ) (radius : ℝ) : Prop :=
  ∃ (O1 O2 O3 A B C : Point), 
    distance O1 O2 = a ∧
    distance O2 O3 = a ∧
    distance O3 O1 = a ∧
    distance O1 A = r1 ∧
    distance O2 A = r2 ∧
    distance O2 B = r2 ∧
    distance O3 B = r3 ∧
    distance O3 C = r3 ∧
    distance O1 C = r1 ∧
    distance O2 C = r2 ∧
    distance A B = 2 * (2 * sqrt( (13 - 6 * sqrt 3) / 13 ))  ∧
    radius = 4 * sqrt 3 - 6

theorem find_circumcircle_radius :
  circumcircle_radius 1 1 (2 * sqrt( (13 - 6 * sqrt 3) / 13 )) (sqrt 3) (4 * sqrt 3 - 6) :=
sorry

end find_circumcircle_radius_l66_66022


namespace one_cow_one_bag_l66_66793

-- Define parameters
def cows : ℕ := 26
def bags : ℕ := 26
def days_for_all_cows : ℕ := 26

-- Theorem to prove the number of days for one cow to eat one bag of husk
theorem one_cow_one_bag (cows bags days_for_all_cows : ℕ) (h : cows = bags) (h2 : days_for_all_cows = 26) : days_for_one_cow_one_bag = 26 :=
by {
    sorry -- Proof to be filled in
}

end one_cow_one_bag_l66_66793


namespace coefficient_of_x4_in_expansion_l66_66761

theorem coefficient_of_x4_in_expansion :
  let f := λ (x : ℝ), (x - 3 * Real.sqrt 2) ^ 8
  ∃ c : ℝ, c * x^4 = (f x) → c = 22680 :=
begin
  intro f,
  use (λ (x : ℝ), (x - 3 * Real.sqrt 2) ^ 8),
  intro x,
  sorry
end

end coefficient_of_x4_in_expansion_l66_66761


namespace smallest_prime_reversing_to_composite_l66_66902

theorem smallest_prime_reversing_to_composite (p : ℕ) :
  p = 23 ↔ (p < 100 ∧ p ≥ 10 ∧ Nat.Prime p ∧ 
  ∃ c, c < 100 ∧ c ≥ 10 ∧ ¬ Nat.Prime c ∧ c = (p % 10) * 10 + p / 10 ∧ (p / 10 = 2 ∨ p / 10 = 3)) :=
by
  sorry

end smallest_prime_reversing_to_composite_l66_66902


namespace problem1_problem2_l66_66271

def f (x a b : ℝ) : ℝ := x^2 + a * x + b

def g (a : ℝ) : ℝ :=
if a ≤ -2 then (a^2)/4 + a + 2
else if -2 < a ∧ a ≤ 2 then 1
else (a^2)/4 - a + 2

theorem problem1 (a : ℝ) (b : ℝ) (h : b = a^2 / 4 + 1) : g(a) = 
(if a ≤ -2 then (a^2)/4 + a + 2
 else if -2 < a ∧ a ≤ 2 then 1
 else (a^2)/4 - a + 2) :=
sorry

theorem problem2 (a b : ℝ) (h1 : 0 ≤ b - 2 * a) (h2 : b - 2 * a ≤ 1) (h3 : ∃ x ∈ set.Icc (-1 : ℝ) 1, f x a b = 0) : 
b ∈ set.Icc (-3 : ℝ) (9 - 4 * real.sqrt 5) :=
sorry

end problem1_problem2_l66_66271


namespace integral_sqrt_and_linear_l66_66218

-- The mathematical problem is to prove the value of the integral
theorem integral_sqrt_and_linear: (∫ x in -2..2, (Real.sqrt (4 - x^2) + 2 * x)) = 2 * Real.pi := 
by
  sorry

end integral_sqrt_and_linear_l66_66218


namespace affine_transformation_unique_basis_affine_transformation_unique_triangle_affine_transformation_unique_parallelogram_l66_66117

-- Part (a): Affine Transformation from point and basis
theorem affine_transformation_unique_basis
  (O O' : Point)
  (e1 e2 : Vector)
  (e1' e2' : Vector) :
  ∃! L : AffineTransformation, 
    L.map_point O = O' ∧
    L.map_vector e1 = e1' ∧
    L.map_vector e2 = e2' :=
sorry

-- Part (b): Affine Transformation for triangles
theorem affine_transformation_unique_triangle
  (A B C A1 B1 C1 : Point) :
  ∃! L : AffineTransformation, 
    L.map_point A = A1 ∧ 
    L.map_point B = B1 ∧ 
    L.map_point C = C1 :=
sorry

-- Part (c): Affine Transformation for parallelograms
theorem affine_transformation_unique_parallelogram
  (P1 P2 P3 P4 P1' P2' P3' P4' : Point) :
  ∃! L : AffineTransformation, 
    L.map_point P1 = P1' ∧ 
    L.map_point P2 = P2' ∧ 
    L.map_point P3 = P3' ∧ 
    L.map_point P4 = P4' :=
sorry

end affine_transformation_unique_basis_affine_transformation_unique_triangle_affine_transformation_unique_parallelogram_l66_66117


namespace black_percentage_is_44_l66_66844

noncomputable def area_of_circle (r : ℝ) : ℝ := π * r^2

theorem black_percentage_is_44 :
  let r0 := 3 in
  let r1 := r0 + 3 in
  let r2 := r1 + 3 in
  let r3 := r2 + 3 in
  let r4 := r3 + 3 in
  let area2 := area_of_circle r2 - area_of_circle r1 in
  let area4 := area_of_circle r4 - area_of_circle r3 in
  let total_black_area := area2 + area4 + area_of_circle r0 in
  let total_area := area_of_circle r4 in
  total_black_area / total_area * 100 = 44 :=
by
  let r0 := 3 in
  let r1 := r0 + 3 in
  let r2 := r1 + 3 in
  let r3 := r2 + 3 in
  let r4 := r3 + 3 in
  let area2 := area_of_circle r2 - area_of_circle r1 in
  let area4 := area_of_circle r4 - area_of_circle r3 in
  let total_black_area := area2 + area4 + area_of_circle r0 in
  let total_area := area_of_circle r4 in
  have h : total_black_area / total_area * 100 = (π * (12^2 - 9^2 + 6^2 - 3^2 + 3^2)) / (π * 15^2) * 100 := sorry
  sorry

end black_percentage_is_44_l66_66844


namespace complement_union_eq_l66_66602

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 2, 4}
noncomputable def B : Set ℕ := {2, 3, 5}

theorem complement_union_eq:
  compl A ∪ B = {0, 2, 3, 5} :=
by
  sorry

end complement_union_eq_l66_66602


namespace relationship_between_x_y_l66_66979

def in_interval (x : ℝ) : Prop := (Real.pi / 4) < x ∧ x < (Real.pi / 2)

noncomputable def x_def (α : ℝ) : ℝ := Real.sin α ^ (Real.log (Real.cos α) / Real.log α)

noncomputable def y_def (α : ℝ) : ℝ := Real.cos α ^ (Real.log (Real.sin α) / Real.log α)

theorem relationship_between_x_y (α : ℝ) (h : in_interval α) : 
  x_def α = y_def α := 
  sorry

end relationship_between_x_y_l66_66979


namespace find_length_PB_l66_66343

noncomputable def radius (O : Type*) : ℝ := sorry

structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)

variables {α : Type*}

def Point (α : Type*) := α

variables (P T A B : Point ℝ) (O : Circle ℝ) (r : ℝ)

def PA := (4 : ℝ)
def PT (AB : ℝ) := AB - 2
def PB (AB : ℝ) := 4 + AB

def power_of_a_point (PA PB PT : ℝ) := PA * PB = PT^2

theorem find_length_PB (AB : ℝ) 
  (h1 : power_of_a_point PA (PB AB) (PT AB)) 
  (h2 : PA < PB AB) : 
  PB AB = 18 := 
by 
  sorry

end find_length_PB_l66_66343


namespace evaluate_expression_at_minus3_l66_66522

theorem evaluate_expression_at_minus3:
  (∀ x, x = -3 → (3 + x * (3 + x) - 3^2 + x) / (x - 3 + x^2 - x) = -3/2) :=
by
  sorry

end evaluate_expression_at_minus3_l66_66522


namespace max_area_sum_l66_66504

-- Define the 100x100 lattice \(\textbf{L}\)
def lattice : Type := fin 100 × fin 100

-- Define the set of polygons \(\mathcal{F}\)
structure Polygon :=
(vertices : finset lattice)
(area : ℝ)

-- Define the set of polygons such that each point in the lattice 
-- is the vertex of exactly one polygon
def valid_polygons (l : finset lattice) (polySet : finset Polygon) : Prop :=
  l = finset.univ ∧
  ∀ p ∈ l, ∃ pol ∈ polySet, p ∈ pol.vertices ∧ ∀ q ∈ lattice, q ∈ pol.vertices → p = q

-- Statement: Prove the sum of the areas of the polygons is at most \(8,332,500\)
theorem max_area_sum (l : finset lattice) (polySet : finset Polygon) (h : valid_polygons l polySet) :
  (polySet.sum Polygon.area) ≤ 8332500 :=
sorry

end max_area_sum_l66_66504


namespace cauchy_schwarz_inequality_l66_66450

theorem cauchy_schwarz_inequality (n : ℕ)
  (a b x : ℕ → ℝ)
  (h_pos : ∀ i, 0 < a i ∧ 0 < b i ∧ 0 < x i) :
  (∑ i in finset.range n, a i * x i) * (∑ i in finset.range n, b i / x i) ≥ 
  (∑ i in finset.range n, real.sqrt (a i * b i)) ^ n :=
begin
  sorry
end

end cauchy_schwarz_inequality_l66_66450


namespace correct_answer_A_l66_66947

open Function

variable {α : Type*} [Preorder α] {f : α → ℝ}

-- Conditions
axiom even_function : ∀ x : α, f x = f (-x)
axiom decreasing_on_interval : ∀ x y : α, 0 ≤ x → x ≤ 2 → 0 ≤ y → y ≤ 2 → x < y → f (x - 2) > f (y - 2)

-- Proof statement
theorem correct_answer_A : f 0 < f (-1) ∧ f (-1) < f 2 :=
by
  sorry

end correct_answer_A_l66_66947


namespace min_value_of_f_l66_66537

def f (x : ℝ) : ℝ := ∑ n in (finset.range 19).map (finset.nat_cast ℕ ℝ), |x - n|

theorem min_value_of_f : ∃ (x : ℝ), f x = 90 :=
by {
  use 10,
  -- The proof would go here
  sorry
}

end min_value_of_f_l66_66537


namespace sin_three_pi_four_minus_alpha_l66_66242

theorem sin_three_pi_four_minus_alpha 
  (α : ℝ) 
  (h₁ : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (3 * π / 4 - α) = 3 / 5 :=
by
  sorry

end sin_three_pi_four_minus_alpha_l66_66242


namespace k_great_implies_k_plus_one_great_l66_66629

variable (G : Type) [SimpleGraph G]
variable (V : Finset G)
variable (k : ℕ)

def k_great (G : SimpleGraph G) (V : Finset G) (k : ℕ) : Prop :=
  ∀ (S : Finset G), S.card = k → ∃ (C : Finset (Sym2 G)), C.card = k ∧ ∀ (e ∈ C), ∃ (a b : G), a ≠ b ∧ e = ⟦(a,b)⟧ ∧ a ∈ S ∧ b ∈ S ∧ G.adj a b

theorem k_great_implies_k_plus_one_great {G : SimpleGraph G} {V : Finset G} (hV : 6 ≤ V.card)
  (h6 : k_great G V 6) : k_great G V 7 :=
sorry

end k_great_implies_k_plus_one_great_l66_66629


namespace trisecting_incircle_l66_66580

universe u
variable {α : Type u} [field α]

theorem trisecting_incircle {A B C : α} (h : trisects_median A B C) : ratio_of_sides A B C = (5, 10, 13) := 
sorry

end trisecting_incircle_l66_66580


namespace min_socks_to_guarantee_15_pairs_l66_66821

theorem min_socks_to_guarantee_15_pairs :
  ∀ (r g b y blk : ℕ),
  r = 120 → g = 100 → b = 70 → y = 50 → blk = 30 →
  (∃ n : ℕ, 
    (∀ (socks : ℕ → ℕ),
      socks 1 + socks 2 + socks 3 + socks 4 + socks 5 = n →
      (∃ (pairs : ℕ), pairs ≥ 15)) →
    n = 146) :=
by 
  intros r g b y blk hr hg hb hy hblk,
  use 146,
  intros socks hs,
  have h_num_pairs : ∃ (pairs : ℕ), pairs ≥ 15,
  {
    sorry
  },
  exact h_num_pairs

end min_socks_to_guarantee_15_pairs_l66_66821


namespace pencil_black_length_l66_66152

theorem pencil_black_length (total_length purple_length blue_length black_length : ℕ) 
  (h1 : total_length = 6)
  (h2 : purple_length = 3) 
  (h3 : blue_length = 1) 
  (h4 : black_length = total_length - purple_length - blue_length) : 
  black_length = 2 :=
by
  have h : black_length = 6 - 3 - 1 := by rw [h1, h2, h3]
  rw [h4] at h
  exact h.symm

end pencil_black_length_l66_66152


namespace seq_a_n_100th_term_l66_66966

theorem seq_a_n_100th_term :
  ∃ a : ℕ → ℤ, a 1 = 3 ∧ a 2 = 6 ∧ 
  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) ∧ 
  a 100 = -3 := 
sorry

end seq_a_n_100th_term_l66_66966


namespace smallest_positive_integer_not_in_any_form_l66_66772

def is_sum_of_consecutive_integers (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2 

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k : ℕ, Prime p ∧ n = p ^ k

def is_one_more_than_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = p + 1

def smallest_excluded_integer (n : ℕ) : Prop :=
  ¬ is_sum_of_consecutive_integers n ∧ ¬ is_prime_power n ∧ ¬ is_one_more_than_prime n

theorem smallest_positive_integer_not_in_any_form : ∃ n : ℕ, smallest_excluded_integer n ∧ ∀ m : ℕ, smallest_excluded_integer m → n ≤ m :=
begin
  use 22,
  split,
  {
    split,
    {
      -- proof that 22 is not a sum of consecutive integers
      sorry,
    },
    split,
    {
      -- proof that 22 is not a prime power
      sorry,
    },
    {
      -- proof that 22 is not one more than a prime
      sorry,
    },
  },
  {
    -- proof that 22 is the smallest such integer
    sorry,
  }
end

end smallest_positive_integer_not_in_any_form_l66_66772


namespace least_value_of_x_l66_66997

theorem least_value_of_x 
  (x : ℕ) 
  (p : ℕ) 
  (hx : 0 < x) 
  (hp : Prime p) 
  (h : x = 2 * 11 * p) : x = 44 := 
by
  sorry

end least_value_of_x_l66_66997


namespace range_of_alpha_l66_66592

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 5 * x

theorem range_of_alpha (α : ℝ) (h₀ : -1 < α) (h₁ : α < 1) (h₂ : f (1 - α) + f (1 - α^2) < 0) : 1 < α ∧ α < Real.sqrt 2 := by
  sorry

end range_of_alpha_l66_66592


namespace tile_splitting_l66_66448

theorem tile_splitting (N : ℕ) (a b : ℝ) (h_a_lt_b : a < b)
  (tiles : fin N → ℝ × ℝ) (h_tiles : ∀ i, tiles i = (a, b)) :
  ∃ (f : fin (2 * N) → ℝ × ℝ),
    (∀ i, (f i).1 = a ∧ (f i).2 = a) ∨
    (∀ i, (f i).1 = (b - (a ^ 2 / b)) ∧ (f i).2 = a) :=
by
  sorry

end tile_splitting_l66_66448


namespace cos_angle_identity_l66_66607

theorem cos_angle_identity (α : ℝ) (h : Real.cos (π / 2 - α) = Real.sqrt 2 / 3) :
  Real.cos (π - 2 * α) = - (5 / 9) := by
sorry

end cos_angle_identity_l66_66607


namespace largest_M_l66_66248

def sum_digits (n : ℕ) : ℕ := n.digits.sum

theorem largest_M (n : ℕ) (M : ℕ) :
  (∀ k : ℕ, k ≤ M → sum_digits (M * k) = sum_digits M) ↔ ∃ n : ℕ, M = 10 ^ n - 1 :=
by {
  sorry
}

end largest_M_l66_66248


namespace sum_of_squares_of_roots_l66_66509

theorem sum_of_squares_of_roots (a b c : ℚ) (h_eq : 6 * a^2 + 5 * a - 11 = 0) :
  let x₁ := (-b + √(b^2 - 4*a*c)) / (2*a),
      x₂ := (-b - √(b^2 - 4*a*c)) / (2*a)
  in x₁^2 + x₂^2 = 157 / 36 :=
by sorry

end sum_of_squares_of_roots_l66_66509


namespace least_possible_N_proof_l66_66460

noncomputable def least_possible_N (N : ℕ) (n : ℕ) : Prop :=
  N > 0 ∧
  (1 ≤ n ∧ n ≤ 29) ∧
  (∀ k : ℕ, (1 ≤ k ∧ k ≤ 30) → k ≠ n → k ≠ n + 1 → k ∣ N) ∧
  N = 2230928700

theorem least_possible_N_proof : ∃ (N n : ℕ), least_possible_N N n :=
by
  use 2230928700
  use 28
  -- Proof of the conditions, skipped with sorry
  sorry

end least_possible_N_proof_l66_66460


namespace hypotenuse_length_l66_66437

theorem hypotenuse_length (a b c : ℝ) (hC : (a^2 + b^2) * (a^2 + b^2 + 1) = 12) (right_triangle : a^2 + b^2 = c^2) : 
  c = Real.sqrt 3 := 
by
  sorry

end hypotenuse_length_l66_66437


namespace a_b_intersect_l66_66655

variables {Point Line BrokenLine : Type}

-- Definition of general position
def general_position (pts : set Point) : Prop :=
  ∀ (p1 p2 p3 : Point),
    p1 ∈ pts → p2 ∈ pts → p3 ∈ pts →
    (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) →
    ¬ ∃ (l : Line), p1 ∈ l ∧ p2 ∈ l ∧ p3 ∈ l ∧
    (¬ ∃ (s : Point), s ∈ between l (p1, p2, p3))

-- Intersection counts between segments and broken lines
def intersects_even (seg : set Point) (bl : BrokenLine) : Prop := sorry
def intersects_odd (seg : set Point) (bl : BrokenLine) : Prop := sorry

variables (a b : BrokenLine) (K L M N : Point)

-- Assumptions based on the problem statement
axiom general_pos : general_position {K, L, M, N}
axiom even_intersects_a_KL : intersects_even {K, L} a
axiom even_intersects_a_MN : intersects_even {M, N} a
axiom odd_intersects_a_LM : intersects_odd {L, M} a
axiom odd_intersects_a_NK : intersects_odd {N, K} a
axiom odd_intersects_b_KL : intersects_odd {K, L} b
axiom odd_intersects_b_MN : intersects_odd {M, N} b
axiom even_intersects_b_LM : intersects_even {L, M} b
axiom even_intersects_b_NK : intersects_even {N, K} b

theorem a_b_intersect : ∃ P : Point, P ∈ a ∧ P ∈ b :=
sorry

end a_b_intersect_l66_66655


namespace calculate_n_l66_66863

theorem calculate_n (n : ℕ) : 3^n = 3 * 9^5 * 81^3 -> n = 23 :=
by
  -- Proof omitted
  sorry

end calculate_n_l66_66863


namespace fruits_eaten_total_l66_66755

variable (oranges_per_day : ℕ) (grapes_per_day : ℕ) (days : ℕ)

def total_fruits (oranges_per_day grapes_per_day days : ℕ) : ℕ :=
  (oranges_per_day * days) + (grapes_per_day * days)

theorem fruits_eaten_total 
  (h1 : oranges_per_day = 20)
  (h2 : grapes_per_day = 40) 
  (h3 : days = 30) : 
  total_fruits oranges_per_day grapes_per_day days = 1800 := 
by 
  sorry

end fruits_eaten_total_l66_66755


namespace minimum_value_of_expression_l66_66982

theorem minimum_value_of_expression (x : ℝ) (h : x > 2) : 
  ∃ y, (∀ z, z > 2 → (z^2 - 4 * z + 5) / (z - 2) ≥ y) ∧ 
       y = 2 :=
by
  sorry

end minimum_value_of_expression_l66_66982


namespace inequality_holds_l66_66338

-- Defining the setup for the problem
variables {a b c x y z : ℝ} (F : ℝ)

-- Conditions for the triangle being acute and the variables x, y, z
def acute_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a < b + c ∧ b < a + c ∧ c < a + b ∧
  a^2 < b^2 + c^2 ∧ b^2 < a^2 + c^2 ∧ c^2 < a^2 + b^2

-- Area of the triangle
def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Statement of the problem in Lean
theorem inequality_holds (h_acute : acute_triangle a b c)
  (h_xy_zx : x * y + y * z + z * x = 1) (h_xnonneg : x ≥ 0)
  (h_ynonneg : y ≥ 0) (h_znonneg : z ≥ 0) :
  a^2 * x + b^2 * y + c^2 * z ≥ 4 * area a b c :=
sorry

end inequality_holds_l66_66338


namespace total_chocolate_bars_proof_l66_66145

def large_box_contains := 17
def first_10_boxes_contains := 10
def medium_boxes_per_small := 4
def chocolate_bars_per_medium := 26

def remaining_7_boxes := 7
def first_two_boxes := 2
def first_two_bars := 18
def next_three_boxes := 3
def next_three_bars := 22
def last_two_boxes := 2
def last_two_bars := 30

noncomputable def total_chocolate_bars_in_large_box : Nat :=
  let chocolate_in_first_10 := first_10_boxes_contains * medium_boxes_per_small * chocolate_bars_per_medium
  let chocolate_in_remaining_7 :=
    (first_two_boxes * first_two_bars) +
    (next_three_boxes * next_three_bars) +
    (last_two_boxes * last_two_bars)
  chocolate_in_first_10 + chocolate_in_remaining_7

theorem total_chocolate_bars_proof :
  total_chocolate_bars_in_large_box = 1202 :=
by
  -- Detailed calculation is skipped
  sorry

end total_chocolate_bars_proof_l66_66145


namespace jana_can_always_reach_chocolate_l66_66340

def cell := (ℕ × ℕ)

def adjacent (c1 c2 : cell) : Prop := 
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 + 1 = c2.2)) ∨
  (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c1.1 + 1 = c2.1))

def same_color (color : cell → ℕ) (c1 c2 : cell) : Prop :=
  color c1 = color c2

noncomputable def can_reach_chocolate
  (n : ℕ)
  (color : cell → ℕ)
  (start : cell)
  (chocolate : cell) :
  Prop :=
  ∃ path : list cell,
    path.head = start ∧
    path.last = chocolate ∧
    ∀ (i : ℕ), i < path.length - 1 →
      (i % 2 = 0 → same_color color (path.nth_le i sorry) (path.nth_le (i + 1) sorry)) ∧
      (i % 2 = 1 → adjacent (path.nth_le i sorry) (path.nth_le (i + 1) sorry))

theorem jana_can_always_reach_chocolate
  (n : ℕ)
  (color : cell → ℕ)
  (start : cell)
  (chocolate : cell) :
  ∀ (table : fin (2 * n) × fin (2 * n)),
    (∀ (c : cell), ∃! (c' : cell), same_color color c c') →
    ∃ path : list cell, can_reach_chocolate n color start chocolate :=
sorry

end jana_can_always_reach_chocolate_l66_66340


namespace max_area_at_least_two_ninths_l66_66337

theorem max_area_at_least_two_ninths :
  ∀ (P : ℝ × ℝ),
    let A := (0, 0)
    let B := (1, 1)
    let C := (1, 0)
    let A_PQ := (λ A P Q : ℝ × ℝ, 0.5 * real.sqrt ((P.1 - A.1)^2 + (Q.1 - A.1)^2))
    let P_BR := (λ B P R : ℝ × ℝ, 0.5 * real.sqrt ((B.1 - P.1)^2 + (R.1 - P.1)^2))
    let QCRP := (λ Q C R P : ℝ × ℝ, real.sqrt ((Q.1 - C.1)^2 + (R.1 - P.1)^2) * real.sqrt ((Q.2 - C.2)^2 + (R.2 - P.2)^2))
    A_PQ A P (P.1, 0) + P_BR B P (0, P.2) + QCRP (P.1, 0) C (0, P.2) P >= 2 / 9 :=
by
  sorry

end max_area_at_least_two_ninths_l66_66337


namespace geometric_sequence_product_l66_66267

theorem geometric_sequence_product (a₁ aₙ : ℝ) (n : ℕ) (hn : n > 0) (number_of_terms : n ≥ 1) :
  -- Conditions: First term, last term, number of terms
  ∃ P : ℝ, P = (a₁ * aₙ) ^ (n / 2) :=
sorry

end geometric_sequence_product_l66_66267


namespace problem1_part1_problem1_part2_problem2_l66_66554

noncomputable def problem1_condition1 (m : ℕ) (a : ℕ) : Prop := 4^m = a
noncomputable def problem1_condition2 (n : ℕ) (b : ℕ) : Prop := 8^n = b

theorem problem1_part1 (m n a b : ℕ) (h1 : 4^m = a) (h2 : 8^n = b) : 2^(2*m + 3*n) = a * b :=
by sorry

theorem problem1_part2 (m n a b : ℕ) (h1 : 4^m = a) (h2 : 8^n = b) : 2^(4*m - 6*n) = (a^2) / (b^2) :=
by sorry

theorem problem2 (x : ℕ) (h : 2 * 8^x * 16 = 2^23) : x = 6 :=
by sorry

end problem1_part1_problem1_part2_problem2_l66_66554


namespace square_area_error_l66_66118

theorem square_area_error (s : ℝ) (h : 0 < s) : 
    let measured_side := 1.02 * s
    let actual_area := s^2
    let measured_area := (1.02 * s)^2
    let error_in_area := measured_area - actual_area
    let percentage_error := (error_in_area / actual_area) * 100
  in percentage_error = 4.04 := by
  sorry

end square_area_error_l66_66118


namespace acute_angled_triangles_in_ngon_l66_66136

noncomputable def acute_angled_triangles_count (A : Fin n → Point) : Nat :=
  sorry

theorem acute_angled_triangles_in_ngon (n : Nat) (A : Fin n → Point)
  (h_convex: is_convex n A) (h_inscribed: is_inscribed n A)
  (h_no_diametrically_opposite: no_diametrically_opposite_vertices n A)
  (h_exists_acute : ∃ p q r : Fin n, p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ is_acute_triangle (A p) (A q) (A r)) :
  ∃ T : Finset (Fin 3) → Point, T.card = n - 2 ∧ (∀ B ∈ T, is_acute_triangle (A B.1) (A B.2) (A B.3)) :=
sorry

end acute_angled_triangles_in_ngon_l66_66136


namespace max_pieces_with_one_cut_min_cuts_to_cut_all_pieces_l66_66153

-- Define the conditions
def cake_dimensions : ℕ := 3 * 5
def individual_pieces : ℕ := 15
def piece_size : (ℕ × ℕ) := (1, 1)
def grid_dimension : (ℕ × ℕ) := (3, 5)

-- Maximum pieces of cake with one cut
theorem max_pieces_with_one_cut
  (cake_dim : ℕ = cake_dimensions)
  (pieces : ℕ = individual_pieces)
  (size : (ℕ × ℕ) = piece_size)
  (grid : (ℕ × ℕ) = grid_dimension)
  : ∃ k : ℕ, k = 22 := 
begin
  sorry
end

-- Minimum number of straight cuts to cut each piece at least once
theorem min_cuts_to_cut_all_pieces
  (cake_dim : ℕ = cake_dimensions)
  (pieces : ℕ = individual_pieces)
  (size : (ℕ × ℕ) = piece_size)
  (grid : (ℕ × ℕ) = grid_dimension)
  : ∃ k : ℕ, k = 3 :=
begin
  sorry
end

end max_pieces_with_one_cut_min_cuts_to_cut_all_pieces_l66_66153


namespace cos_B_of_sine_ratios_proof_l66_66660

noncomputable def cos_B_of_sine_ratios (A B C : ℝ) (a b c : ℝ) 
  (h1 : sin A / sin B = 4 / 3)
  (h2 : sin B / sin C = 3 / 2)
  (h3 : a / sin A = b / sin B)
  (h4 : b / sin B = c / sin C)
  (ha : a = 4)
  (hb : b = 3)
  (hc : c = 2) : ℝ :=
(cos B) = (11 / 16)

theorem cos_B_of_sine_ratios_proof :
  ∀ (A B C a b c : ℝ),
  sin A / sin B = 4 / 3 → 
  sin B / sin C = 3 / 2 → 
  a / sin A = b / sin B → 
  b / sin B = c / sin C → 
  a = 4 → 
  b = 3 → 
  c = 2 → 
  cos B = 11 / 16 :=
by
  intros A B C a b c h1 h2 h3 h4 ha hb hc
  sorry

end cos_B_of_sine_ratios_proof_l66_66660


namespace thin_rings_in_each_group_l66_66184

theorem thin_rings_in_each_group :
  ∀ (x : ℕ), 
    let rings_per_group := 2 + x in
    let first_tree_rings := 70 * rings_per_group in
    let second_tree_rings := 40 * rings_per_group in
    first_tree_rings = second_tree_rings + 180 → x = 4 :=
by
  assume x : ℕ,
  let rings_per_group := 2 + x,
  let first_tree_rings := 70 * rings_per_group,
  let second_tree_rings := 40 * rings_per_group,
  assume h: first_tree_rings = second_tree_rings + 180,
  sorry

end thin_rings_in_each_group_l66_66184


namespace satisfactory_fraction_l66_66304

variables (A B C D F : ℕ)
variables (total_students satisfactory_students : ℕ)

axiom grade_distribution : A = 6 ∧ B = 5 ∧ C = 7 ∧ D = 4 ∧ F = 6
axiom total_students_calculation : total_students = A + B + C + D + F
axiom satisfactory_students_calculation : satisfactory_students = A + B + C

theorem satisfactory_fraction (h1 : grade_distribution)
                              (h2 : total_students_calculation)
                              (h3 : satisfactory_students_calculation) :
  satisfactory_students.to_rat / total_students.to_rat = 9 / 14 :=
sorry

end satisfactory_fraction_l66_66304


namespace inequality_series_delta_ge_3sqrt3_r_sq_R_ge_2r_l66_66618

variables {a b c r R Δ : ℝ}
variables {s : ℝ} -- semiperimeter

-- Assume a, b, c are sides of a triangle, and r, R, Δ are inradius, circumradius, and area respectively.
-- Assume the triangle inequality holds.

-- Defining the inequalities to be proven as Lean statements:
theorem inequality_series (h1 : r = Δ / s) (h2 : Δ = abc / (4 * R)) (h3 : s = (a + b + c) / 2) (triangle_inequality: a + b > c ∧ a + c > b ∧ b + c > a) :
    r ≤ (3 / 2) * (abc / ((sqrt (a^2 + b^2 + c^2)) * (a + b + c))) ∧
    (3 / 2) * (abc / ((sqrt (a^2 + b^2 + c^2)) * (a + b + c))) ≤ (3 / 2) * sqrt 3 * (abc / ((a + b + c)^2)) ∧
    (3 / 2) * sqrt 3 * (abc / ((a + b + c)^2)) ≤ (sqrt 3 / 2) * ((abc)^(2/3) / (a + b + c)) ∧
    (sqrt 3 / 2) * ((abc)^(2/3) / (a + b + c)) ≤ (sqrt 3 / 18) * (a + b + c) ∧
    (sqrt 3 / 18) * (a + b + c) ≤ (sqrt (a^2 + b^2 + c^2)) / 6 :=
  sorry

theorem delta_ge_3sqrt3_r_sq (h1 : Δ = abc / (4 * R)) (h2 : r = Δ / s) (h3 : s = (a + b + c) / 2) (triangle_inequality: a + b > c ∧ a + c > b ∧ b + c > a) :
  Δ ≥ 3 * sqrt 3 * r^2 :=
  sorry

theorem R_ge_2r (h1 : r = Δ / s) (h2 : Δ = abc / (4 * R)) (triangle_inequality: a + b > c ∧ a + c > b ∧ b + c > a):
  R ≥ 2 * r :=
  sorry

end inequality_series_delta_ge_3sqrt3_r_sq_R_ge_2r_l66_66618


namespace find_y_l66_66224

noncomputable def y : ℝ :=
  let y := 27 in
  if h : log y 243 = 5 / 3 then y else 0

theorem find_y (log_condition : log (y) 243 = 5 / 3) : y = 27 :=
by {
  sorry
}

end find_y_l66_66224


namespace geometric_sequence_common_ratio_l66_66574

theorem geometric_sequence_common_ratio (a : ℕ → ℚ) (q : ℚ) :
  (∀ n, a n = a 2 * q ^ (n - 2)) ∧ a 2 = 2 ∧ a 6 = 1 / 8 →
  (q = 1 / 2 ∨ q = -1 / 2) :=
by
  sorry

end geometric_sequence_common_ratio_l66_66574


namespace find_k_l66_66798

theorem find_k : ∃ k : ℕ, (∀ (k > 0), (∏ i in finset.range (k + 2), (1 + 1 / (i + 2))) = 2014) → k = 4026 :=
by sorry

end find_k_l66_66798


namespace relation_a_b_c_l66_66254

noncomputable def a (f : ℝ → ℝ) : ℝ := (1 / 2) * f (Real.log 2 ** (1 / 2))
noncomputable def b (f : ℝ → ℝ) : ℝ := Real.log 2 * f (Real.log 2)
noncomputable def c (f : ℝ → ℝ) : ℝ := 2 * f (-2)

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def decreasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x + x * (deriv (deriv f) x) < 0

theorem relation_a_b_c (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : decreasing_on_neg f) :
  a f > b f ∧ b f > c f := by
  sorry

end relation_a_b_c_l66_66254


namespace urn_contains_four_each_color_after_six_steps_l66_66855

noncomputable def probability_urn_four_each_color : ℚ := 2 / 7

def urn_problem (urn_initial : ℕ) (draws : ℕ) (final_urn : ℕ) (extra_balls : ℕ) : Prop :=
urn_initial = 2 ∧ draws = 6 ∧ final_urn = 8 ∧ extra_balls > 0

theorem urn_contains_four_each_color_after_six_steps :
  urn_problem 2 6 8 2 → probability_urn_four_each_color = 2 / 7 :=
by
  intro h
  cases h
  sorry

end urn_contains_four_each_color_after_six_steps_l66_66855


namespace kite_area_is_correct_l66_66234

structure Point (α : Type _) :=
  (x : α)
  (y : α)

def distance (p1 p2 : Point ℝ) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

def kite_area (p1 p2 p3 p4 : Point ℝ) : ℝ :=
  let base1 := distance p1 p3 * 2 -- because 1 grid unit = 2 inches
  let height1 := abs (p2.y - p1.y) * 2
  let area1 := 1 / 2 * base1 * height1
  
  let base2 := distance p1 p3 * 2
  let height2 := abs (p4.y - p3.y) * 2
  let area2 := 1 / 2 * base2 * height2 
  area1 + area2

theorem kite_area_is_correct (p1 p2 p3 p4 : Point ℝ)
  (h1 : p1 = Point.mk 0 10)
  (h2 : p2 = Point.mk 5 14)
  (h3 : p3 = Point.mk 10 10)
  (h4 : p4 = Point.mk 5 0)
  : kite_area p1 p2 p3 p4 = 160 := by
  sorry

end kite_area_is_correct_l66_66234


namespace boxes_neither_pens_nor_pencils_l66_66546

theorem boxes_neither_pens_nor_pencils (total_boxes boxes_with_pencils boxes_with_pens boxes_with_both : ℕ) 
    (h1 : total_boxes = 15)
    (h2 : boxes_with_pencils = 7)
    (h3 : boxes_with_pens = 4)
    (h4 : boxes_with_both == 3) :
    (total_boxes - (boxes_with_pencils + boxes_with_pens - boxes_with_both) = 7) :=
begin
  sorry
end

end boxes_neither_pens_nor_pencils_l66_66546


namespace tip_percentage_is_20_l66_66458

-- Define the given conditions
def total_spent : ℝ := 198
def price_food_before_tax : ℝ := 150
def sales_tax_rate : ℝ := 0.10

-- Define the derived quantities from conditions
def sales_tax : ℝ := sales_tax_rate * price_food_before_tax
def total_cost_before_tip : ℝ := price_food_before_tax + sales_tax
def tip_amount : ℝ := total_spent - total_cost_before_tip
def tip_percentage : ℝ := (tip_amount / total_cost_before_tip) * 100

-- Prove that the tip percentage is 20%
theorem tip_percentage_is_20 : tip_percentage = 20 :=
by
  sorry

end tip_percentage_is_20_l66_66458


namespace tan_alpha_expression_value_l66_66939

-- (I) Prove that tan(α) = 4/3 under the given conditions
theorem tan_alpha (O A B C P : ℝ × ℝ) (α : ℝ)
  (hO : O = (0, 0))
  (hA : A = (Real.sin α, 1))
  (hB : B = (Real.cos α, 0))
  (hC : C = (-Real.sin α, 2))
  (hP : P = (2 * Real.cos α - Real.sin α, 1))
  (h_collinear : ∃ t : ℝ, C = t • (P.1, P.2)) :
  Real.tan α = 4 / 3 := sorry

-- (II) Prove the given expression under the condition tan(α) = 4/3
theorem expression_value (α : ℝ)
  (h_tan : Real.tan α = 4 / 3) :
  (Real.sin (2 * α) + Real.sin α) / (2 * Real.cos (2 * α) + 2 * Real.sin α * Real.sin α + Real.cos α) + Real.sin (2 * α) = 
  172 / 75 := sorry

end tan_alpha_expression_value_l66_66939


namespace smallest_positive_divisor_is_120_l66_66229

noncomputable def smallest_divisor (n:ℕ) : ℕ :=
  if n = 0 then 0 else Nat.find (λ k, k > 0 ∧ (z**n-1)Factorization z^k-1)

theorem smallest_positive_divisor_is_120 : ∀ n ∈ ℕ, smallest_divisor (12+11+8+7+5+3+1) = 120 := by
  sorry

end smallest_positive_divisor_is_120_l66_66229


namespace crowdfunding_highest_level_backing_l66_66488

-- Definitions according to the conditions
def lowest_level_backing : ℕ := 50
def second_level_backing : ℕ := 10 * lowest_level_backing
def highest_level_backing : ℕ := 100 * lowest_level_backing
def total_raised : ℕ := (2 * highest_level_backing) + (3 * second_level_backing) + (10 * lowest_level_backing)

-- Statement of the problem
theorem crowdfunding_highest_level_backing (h: total_raised = 12000) :
  highest_level_backing = 5000 :=
sorry

end crowdfunding_highest_level_backing_l66_66488


namespace impossible_15_vertices_degree_5_l66_66318

theorem impossible_15_vertices_degree_5 :
  ∀ (G : SimpleGraph (Fin 15)), (∀ v : Fin 15, degree G v = 5) → false :=
begin
  sorry

end impossible_15_vertices_degree_5_l66_66318


namespace mode_of_data_set_is_19_l66_66397

def mode (l : List ℕ) : ℕ :=
  l.groupBy.ranges.map (λ r => (r.head, r)).maxBy (λ x => x.snd.length).fst

theorem mode_of_data_set_is_19 :
  mode [3, 8, 8, 19, 19, 19, 19] = 19 :=
by
  sorry

end mode_of_data_set_is_19_l66_66397


namespace length_of_each_train_proof_l66_66028

noncomputable def length_of_each_train (L : ℝ) :=
  let speed_faster := 46 * 1000 / 3600 in
  let speed_slower := 36 * 1000 / 3600 in
  let relative_speed := speed_faster - speed_slower in
  let time_pass := 72 in
  let distance_covered := relative_speed * time_pass in
  2 * L = distance_covered

theorem length_of_each_train_proof : length_of_each_train 1000 :=
by
  let L := 1000
  show length_of_each_train L
  sorry

end length_of_each_train_proof_l66_66028


namespace ratio_of_ages_l66_66120

theorem ratio_of_ages (age_saras age_kul : ℕ) (h_saras : age_saras = 33) (h_kul : age_kul = 22) : 
  age_saras / Nat.gcd age_saras age_kul = 3 ∧ age_kul / Nat.gcd age_saras age_kul = 2 :=
by
  sorry

end ratio_of_ages_l66_66120


namespace existence_of_distinct_positive_integers_l66_66517

theorem existence_of_distinct_positive_integers :
  ∃ (n : ℕ) (a : Fin n → ℕ), (∀ i, 0 < a i) ∧ Function.Injective a ∧ (∑ i, (1 / (a i):ℚ)) = 2019 :=
sorry

end existence_of_distinct_positive_integers_l66_66517


namespace man_work_m_alone_in_15_days_l66_66464

theorem man_work_m_alone_in_15_days (M : ℕ) (h1 : 1/M + 1/10 = 1/6) : M = 15 := sorry

end man_work_m_alone_in_15_days_l66_66464


namespace no_equidistant_points_l66_66196

noncomputable def circle (O : ℝ × ℝ) (r : ℝ) := { P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2 }

def tangent_line (O : ℝ × ℝ) (d : ℝ) := { P : ℝ × ℝ | P.2 = O.2 + d }

def equidistant_points (C : set (ℝ × ℝ)) (L1 L2 : set (ℝ × ℝ)) :=
  { P : ℝ × ℝ | ∃ (d : ℝ), (P ∉ C) ∧ P ∈ L1 ∧ P ∈ L2 }

open set

theorem no_equidistant_points (O : ℝ × ℝ) (r : ℝ) :
  let C := circle O r in
  let L1 := tangent_line O (r + 2) in
  let L2 := tangent_line O (r + 4) in
  equidistant_points C L1 L2 = ∅ :=
by
  sorry

end no_equidistant_points_l66_66196


namespace pencils_per_student_l66_66413

theorem pencils_per_student (total_pencils students : ℕ) (h1 : total_pencils = 18) (h2 : students = 2) : 
  total_pencils / students = 9 :=
by
  rw [h1, h2]
  norm_num

end pencils_per_student_l66_66413


namespace number_of_non_empty_proper_subsets_l66_66598

def M : Finset ℕ := {1, 2, 3, 4, 5, 6}

noncomputable def N : Finset ℕ :=
  (M.toList.product M.toList).filter (λ p, p.1 < p.2).map (λ p, p.1 + p.2) 

theorem number_of_non_empty_proper_subsets :
  Finset.card (Finset.powerset N) - 2 = 510 := by
  sorry

end number_of_non_empty_proper_subsets_l66_66598


namespace discount_percentage_l66_66333

noncomputable def total_number_of_tickets : ℕ := 24
noncomputable def cost_per_ticket : ℝ := 7.0
noncomputable def amount_spent : ℝ := 84.0

theorem discount_percentage :
  let total_cost_without_discount := total_number_of_tickets * cost_per_ticket,
      discount_amount := total_cost_without_discount - amount_spent,
      percentage_of_discount := (discount_amount / total_cost_without_discount) * 100 in
  percentage_of_discount = 50 :=
by
  sorry

end discount_percentage_l66_66333


namespace sequence_sum_l66_66882

noncomputable def a_seq : ℕ → ℝ 
| 1       := a 
| 2       := 1 
| (n+2)   := (2 * max (a_seq (n+1)) 2) / a_seq n

theorem sequence_sum (a : ℝ) (h_a_pos : 0 < a) (h_a_2015 : a_seq a 2015 = 4 * a) : 
  let S := (λ n, ∑ i in range n, a_seq a (i+1)) in
  S 2015 = 7254 :=
by
  sorry

end sequence_sum_l66_66882


namespace algebraic_identity_l66_66528

theorem algebraic_identity (a b c d : ℝ) : a - b + c - d = a + c - (b + d) :=
by
  sorry

end algebraic_identity_l66_66528


namespace sqrt_product_simplifies_l66_66860

theorem sqrt_product_simplifies (p : ℝ) : 
  Real.sqrt (12 * p) * Real.sqrt (20 * p) * Real.sqrt (15 * p^2) = 60 * p^2 := 
by
  sorry

end sqrt_product_simplifies_l66_66860


namespace hyperbola_condition_l66_66952

theorem hyperbola_condition (k : ℝ) : 
  (∀ x y : ℝ, (x^2 / (1 + k)) - (y^2 / (1 - k)) = 1 → (-1 < k ∧ k < 1)) ∧ 
  ((-1 < k ∧ k < 1) → ∀ x y : ℝ, (x^2 / (1 + k)) - (y^2 / (1 - k)) = 1) :=
sorry

end hyperbola_condition_l66_66952


namespace distinct_increasing_digits_count_l66_66285

theorem distinct_increasing_digits_count (n : ℕ) : 
  2030 ≤ n ∧ n ≤ 2600 ∧ 
  (∀ i j : ℕ, i < j → digit_at n i < digit_at n j) ∧ 
  (∀ i j : ℕ, i ≠ j → digit_at n i ≠ digit_at n j) → 
  ∃! k, k = 16 := 
sorry

end distinct_increasing_digits_count_l66_66285


namespace max_area_equilateral_triangle_inscribed_in_rectangle_l66_66833

theorem max_area_equilateral_triangle_inscribed_in_rectangle :
  let p := 338 in
  let q := 3 in
  let r := 507 in
  p * p + q + r = 848 :=
by
  -- condition: rectangle has sides of lengths 12 and 13
  have rect_sides : ℝ × ℝ := (12, 13),
  -- condition: triangle is equilateral and inscribed in rectangle
  sorry

end max_area_equilateral_triangle_inscribed_in_rectangle_l66_66833


namespace daniel_noodles_left_l66_66510

-- Define initial number of noodles Daniel had
def noodles_initial : ℕ := 54

-- Define the percentage of noodles Daniel gave away
def percentage_given : ℝ := 0.25

-- Define the number of noodles Daniel gave away, rounding down
def noodles_given : ℕ := (percentage_given * noodles_initial).toInt

-- Define the remaining noodles Daniel has left
def noodles_left : ℕ := noodles_initial - noodles_given

-- State the theorem and provide the proof stub
theorem daniel_noodles_left : noodles_left = 41 := by
  sorry

end daniel_noodles_left_l66_66510


namespace find_original_number_l66_66257

-- Given conditions
def cond1 (x : ℕ) : Prop := x * 74 = 19832
def cond2 (x : ℝ) : Prop := (x / 100) * 0.74 = 1.9832

-- Proof Problem
theorem find_original_number (x : ℝ) : cond1 (x.to_nat) ∧ cond2 x → x = 268 := by
  -- Proof to be filled in
  sorry

end find_original_number_l66_66257


namespace fermat_little_theorem_l66_66807

theorem fermat_little_theorem (p : ℕ) (a : ℤ) (hp : Nat.Prime p) (hcoprime : Int.gcd a p = 1) : 
  (a ^ (p - 1)) % p = 1 % p := 
sorry

end fermat_little_theorem_l66_66807


namespace convex_quadrilateral_area_l66_66407

theorem convex_quadrilateral_area (a b c d : ℝ) (convex : true) :
  let area := 1/2 * (a + c) * (b + d)
  in area / 2 ≤ (a + c) * (b + d) / 4 :=
by
  -- proof will go here
  sorry

end convex_quadrilateral_area_l66_66407


namespace quadrilateral_not_necessarily_parallelogram_l66_66502

theorem quadrilateral_not_necessarily_parallelogram (ABCD : Type) [convex ABCD]
  {M : Type} {A B C D : ABCD} (hM_inside : M inside ABCD)
  (h_areas_equal : area (triangle A B M) = area (triangle B C M) ∧ 
    area (triangle B C M) = area (triangle C D M) ∧ 
    area (triangle C D M) = area (triangle D A M)):
  (¬is_parallelogram ABCD) ∨ (¬is_intersection_of_diagonals ABCD M) :=
by
  sorry

end quadrilateral_not_necessarily_parallelogram_l66_66502


namespace largest_possible_angle_l66_66719

-- Define the problem based on given conditions
variables (a b : ℝ)

def largest_angle_between_line_and_plane : ℝ :=
  Real.arcsin (1 / 3)

-- Main theorem
theorem largest_possible_angle (h1a : a = b) :
  ∃ (α : ℝ), α = Real.arcsin (1 / 3) := 
  sorry

end largest_possible_angle_l66_66719


namespace existence_of_linear_function_passing_first_and_fourth_quadrants_l66_66436

theorem existence_of_linear_function_passing_first_and_fourth_quadrants :
  ∃ (k b : ℝ), k > 0 ∧ b < 0 ∧ ∀ x, (k * x + b) = (x - 1) :=
by
  use 1
  use -1
  simp
  split
  { linarith }
  { linarith }
  intros
  sorry

end existence_of_linear_function_passing_first_and_fourth_quadrants_l66_66436


namespace cheapest_salon_option_haily_l66_66283

theorem cheapest_salon_option_haily : 
  let gustran_haircut := 45
  let gustran_facial := 22
  let gustran_nails := 30
  let gustran_foot_spa := 15
  let gustran_massage := 50
  let gustran_total := gustran_haircut + gustran_facial + gustran_nails + gustran_foot_spa + gustran_massage
  let gustran_discount := 0.20
  let gustran_final := gustran_total * (1 - gustran_discount)

  let barbara_nails := 40
  let barbara_haircut := 30
  let barbara_facial := 28
  let barbara_foot_spa := 18
  let barbara_massage := 45
  let barbara_total :=
      barbara_nails + barbara_haircut + (barbara_facial * 0.5) + barbara_foot_spa + (barbara_massage * 0.5)

  let fancy_haircut := 34
  let fancy_facial := 30
  let fancy_nails := 20
  let fancy_foot_spa := 25
  let fancy_massage := 60
  let fancy_total := fancy_haircut + fancy_facial + fancy_nails + fancy_foot_spa + fancy_massage
  let fancy_discount := 15
  let fancy_final := fancy_total - fancy_discount

  let avg_haircut := (gustran_haircut + barbara_haircut + fancy_haircut) / 3
  let avg_facial := (gustran_facial + barbara_facial + fancy_facial) / 3
  let avg_nails := (gustran_nails + barbara_nails + fancy_nails) / 3
  let avg_foot_spa := (gustran_foot_spa + barbara_foot_spa + fancy_foot_spa) / 3
  let avg_massage := (gustran_massage + barbara_massage + fancy_massage) / 3

  let luxury_haircut := avg_haircut * 1.10
  let luxury_facial := avg_facial * 1.10
  let luxury_nails := avg_nails * 1.10
  let luxury_foot_spa := avg_foot_spa * 1.10
  let luxury_massage := avg_massage * 1.10
  let luxury_total := luxury_haircut + luxury_facial + luxury_nails + luxury_foot_spa + luxury_massage
  let luxury_discount := 20
  let luxury_final := luxury_total - luxury_discount

  gustran_final > barbara_total ∧ barbara_total < fancy_final ∧ barbara_total < luxury_final := 
by 
  sorry

end cheapest_salon_option_haily_l66_66283


namespace range_a_range_fx_diff_fx_l66_66272

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x

-- Define the conditions, considering any x in the interval [0, π]
variables {a x : ℝ}
variable h1 : 0 ≤ x ∧ x ≤ Real.pi

-- The first problem: The range of a
theorem range_a (extreme_points : ∃ x1 x2 : ℝ, x1 ∈ Icc 0 Real.pi ∧ x2 ∈ Icc 0 Real.pi ∧ x1 < x2 ∧ f a x1 = f a x2) :
  0 < a ∧ a < 1 :=
sorry

-- The second problem: The range of f(x1) - f(x2)
theorem range_fx_diff_fx (x1 x2 : ℝ) (h1 : x1 ∈ Icc 0 Real.pi)
  (h2 : x2 ∈ Icc 0 Real.pi) (h3 : x1 < x2) (h4 : f a x1 = f a x2) :
  0 < f a x1 - f a x2 ∧ f a x1 - f a x2 < 2 :=
sorry

end range_a_range_fx_diff_fx_l66_66272


namespace negation_diagonals_of_parallelogram_l66_66399

theorem negation_diagonals_of_parallelogram :
  (¬ ∀ (P : Type) [parallelogram P], (diagonals_equal P ∧ diagonals_bisect P)) ↔
  ∃ (P : Type) [parallelogram P], (¬ (diagonals_equal P) ∨ ¬ (diagonals_bisect P)) :=
begin
  sorry
end

end negation_diagonals_of_parallelogram_l66_66399


namespace french_books_count_l66_66411

theorem french_books_count (F : ℕ) 
  (h1 : 11 books on English) 
  (h2 : ∀ (arrangement : list char), no two French books may be together in the arrangement) 
  (h3 : (12 choose F) = 220) : 
  F = 3 :=
by
  sorry

end french_books_count_l66_66411


namespace distinct_lines_l66_66588

noncomputable def numDistinctLines (s : Finset ℕ) : ℕ :=
  (s.card * (s.card - 1)) / 2

theorem distinct_lines {s : Finset ℕ} (h : s = {1, 2, 3, 4, 5}) : numDistinctLines s = 18 :=
by {
  rw h,
  simp [numDistinctLines],
  norm_num,
  sorry
}

end distinct_lines_l66_66588


namespace vacuum_pump_reduction_l66_66144

theorem vacuum_pump_reduction (n : ℕ) (h : 0.4 ^ n < 0.005) : n ≥ 6 :=
sorry

end vacuum_pump_reduction_l66_66144


namespace smallest_integer_with_15_divisors_l66_66773

def has_exactly_n_divisors (n d : ℕ) := (finset.filter (λ x : ℕ, d ∣ x) (finset.range (n + 1))).card = d

theorem smallest_integer_with_15_divisors : 
  ∃ (n : ℕ), has_exactly_n_divisors n 15 ∧ ∀ m : ℕ, (has_exactly_n_divisors m 15 → n ≤ m) :=
begin
  use 900,
  split,
  { sorry }, -- We would need to prove that 900 has exactly 15 divisors.
  { intros m hm,
    sorry } -- We would need to prove that any number with exactly 15 divisors is not smaller than 900.
end

end smallest_integer_with_15_divisors_l66_66773


namespace common_internal_tangent_le_distance_centers_l66_66374

variables {O1 O2 P Q : Point}
variables (radius : ℝ)
variables (h1 : Dist(O1, P) = radius)
variables (h2 : Dist(O2, Q) = radius)
variables (h3 : ∀ A B : Point, Dist(A, B) ≥ 0)

theorem common_internal_tangent_le_distance_centers :
  radius > 0 →
  Dist(O1, O2) = radius + radius →
  Dist(O1, Q) = Dist(O2, P) →
  Dist(P, Q) ≤ Dist(O1, O2) :=
by sorry

end common_internal_tangent_le_distance_centers_l66_66374


namespace abs_neg_three_l66_66716

-- Definition of absolute value for a real number
def abs (x : ℝ) : ℝ := if x < 0 then -x else x

-- Proof statement that |-3| = 3
theorem abs_neg_three : abs (-3) = 3 := 
sorry

end abs_neg_three_l66_66716


namespace max_distance_sum_l66_66672

-- Definitions based on the given conditions
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 16) = 1
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)
def M : ℝ × ℝ := (6, 4)

-- The math proof problem in Lean 4 statement
theorem max_distance_sum : 
  ∀ P : ℝ × ℝ, ellipse_eq P.1 P.2 → 
  max_value (|P - M| + |P - F1|) = 15 := 
by 
  sorry

end max_distance_sum_l66_66672


namespace parabola_standard_equation_l66_66260

theorem parabola_standard_equation (directrix : ℝ → Prop) (h : ∀ y, directrix y ↔ y = -1) :
  ∃ (p : ℝ), p = 1 ∧ ∀ x y, x^2 = 4 * p * y :=
by
  use 1
  split
  . rfl
  . intros
    exact sorry

end parabola_standard_equation_l66_66260


namespace number_of_4_digit_palindromes_l66_66974

theorem number_of_4_digit_palindromes (S : Finset ℕ) [Nonempty S] 
  (h₁ : ∀ x ∈ S, x ∈ {1, 3, 5, 7, 9}) 
  (h₂ : ∀ x ∈ S, x < 10): 
  ∃ n, n = 45 := 
sorry

end number_of_4_digit_palindromes_l66_66974


namespace find_m_of_f_monotone_decreasing_l66_66615

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x^2 + m * x) * Real.exp x

theorem find_m_of_f_monotone_decreasing :
  ∃ m : ℝ, (∀ x ∈ Icc (-3/2 : ℝ) (1 : ℝ), deriv (f m) x ≤ 0) ↔ m = -3/2 :=
by sorry

end find_m_of_f_monotone_decreasing_l66_66615


namespace cos_double_angle_l66_66571

theorem cos_double_angle (theta : ℝ) (h : Real.sin (Real.pi - theta) = 1 / 3) : Real.cos (2 * theta) = 7 / 9 := by
  sorry

end cos_double_angle_l66_66571


namespace distinct_remainders_l66_66174

theorem distinct_remainders (p : ℕ) (hp : Nat.Prime p) (h7 : 7 < p) : 
  let remainders := 
    { r | r ∈ Finset.univ.filter (λ r, ∃ k, p^2 = 210 * k + r) } 
  remainders.card = 6 :=
sorry

end distinct_remainders_l66_66174


namespace fruit_seller_l66_66824

theorem fruit_seller (A P : ℝ) (h1 : A = 700) (h2 : A * (100 - P) / 100 = 420) : P = 40 :=
sorry

end fruit_seller_l66_66824


namespace find_a_value_l66_66999

variable (A B C a b c : ℝ)
variable (h1 : ∠A = 2 * ∠C)
variable (h2 : c = 2)
variable (h3 : a^2 = 4 * b - 4)

theorem find_a_value : a = 2 * real.sqrt 3 := 
by 
  sorry  -- proof goes here

end find_a_value_l66_66999


namespace second_vessel_ratio_l66_66029

theorem second_vessel_ratio (mix1_milk mix1_water total_units : ℕ)
  (h1 : mix1_milk = 7) (h2 : mix1_water = 2)
  (h3 : total_units = 9)
  (combined_milk combined_water : ℕ)
  (h4 : combined_milk = 5 * 3) (h5 : combined_water = 3) :
  ∃ (x y : ℕ), x + y = total_units ∧ x : y = 8 :=
by
  use 8, 1
  split
  · exact rfl
  · simp [ratio.eq_of_mk_eq_mk]
  sorry

end second_vessel_ratio_l66_66029


namespace simplify_expression_l66_66349

variable (a b : ℝ) (hab_pos : 0 < a ∧ 0 < b)
variable (h : a^3 - b^3 = a - b)

theorem simplify_expression 
  (a b : ℝ) (hab_pos : 0 < a ∧ 0 < b) (h : a^3 - b^3 = a - b) : 
  (a / b - b / a + 1 / (a * b)) = 2 * (1 / (a * b)) - 1 := 
sorry

end simplify_expression_l66_66349


namespace volume_of_cut_off_pyramid_l66_66476

noncomputable def base_edge_length : ℝ := 12
noncomputable def slant_edge_length : ℝ := 15
noncomputable def height_above_base : ℝ := 4

noncomputable def cut_off_pyramid_volume (base_edge slant_edge height_base_cut : ℝ) : ℝ :=
  let base_diagonal := base_edge * Real.sqrt 2 / 2
  let height := Real.sqrt (slant_edge^2 - base_diagonal^2)
  let new_height := height - height_base_cut
  let new_base_edge := base_edge * (new_height / height)
  (1 / 3) * (new_base_edge^2) * new_height

theorem volume_of_cut_off_pyramid :
  cut_off_pyramid_volume base_edge_length slant_edge_length height_above_base ≈ 321.254901 :=
by
  unfold base_edge_length slant_edge_length height_above_base
  unfold cut_off_pyramid_volume
  -- Perform the calculations and verify that the resulting volume is approximately 321.254901
  sorry

end volume_of_cut_off_pyramid_l66_66476


namespace log_base_5_18_l66_66570

variable (a b : ℝ)
def log_base_10_2 := a
def log_base_10_3 := b

theorem log_base_5_18 : log 18 / log 5 = (a + 2 * b) / (1 - a) :=
by
  sorry

end log_base_5_18_l66_66570


namespace sum_of_odd_divisors_of_90_l66_66090

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l66_66090


namespace mass_percentage_H_correct_l66_66899

-- Definitions of molar masses
def molarMass_H : ℝ := 1.01    -- g/mol
def molarMass_Cl : ℝ := 35.45  -- g/mol
def molarMass_O : ℝ := 16.00   -- g/mol

-- Molar mass of HClO2
def molarMass_HClO2 : ℝ := molarMass_H + molarMass_Cl + 2 * molarMass_O

-- Calculation of mass percentage of H in HClO2
def massPercentage_H_in_HClO2 : ℝ := (molarMass_H / molarMass_HClO2) * 100

-- Theorem stating the required problem
theorem mass_percentage_H_correct :
  massPercentage_H_in_HClO2 = 1.475 := by
  sorry

end mass_percentage_H_correct_l66_66899


namespace excircle_radii_identity_l66_66998

theorem excircle_radii_identity
  (a b c : ℝ)
  (r_a r_b r_c : ℝ)
  (h1 : r_a = (a + b + c) * (a + b - c) / (4 * real.sqrt((a + b + c) / 2 * (a + b + c) / 2 - a * (a / 2 + b / 2 + c / 2) * (b / 2 + c / 2 - a / 2))))
  (h2 : r_b = (a + b + c) * (b + c - a) / (4 * real.sqrt((a + b + c) / 2 * (a + b + c) / 2 - b * (b / 2 + c / 2 + a / 2) * (c / 2 + a / 2 - b / 2))))
  (h3 : r_c = (a + b + c) * (c + a - b) / (4 * real.sqrt((a + b + c) / 2 * (a + b + c) / 2 - c * (c / 2 + a / 2 + b / 2) * (a / 2 + b / 2 - c / 2))))
  : (a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b))) = 2 :=
by
  sorry

end excircle_radii_identity_l66_66998


namespace find_x_for_h_eq_x_l66_66678

-- Define the function h
def h (x : ℝ) : ℝ := (5 * (x - 2) / 3) - 6

-- Main theorem: finding the value of x where h(x) = x
theorem find_x_for_h_eq_x : ∃ x : ℝ, h(x) = x ∧ x = 14 := 
by {
  -- The proof is omitted, so we use sorry to complete the theorem statement.
  sorry
}

end find_x_for_h_eq_x_l66_66678


namespace arithmetic_sequence_difference_l66_66427

theorem arithmetic_sequence_difference :
  let a : ℤ := -11
  let d : ℤ := -4 - a
  let a_1002 := a + 1001 * d
  let a_1008 := a + 1007 * d
  |a_1008 - a_1002| = 42 :=
by
  let a : ℤ := -11
  let d : ℤ := -4 - a
  let a_1002 := a + 1001 * d
  let a_1008 := a + 1007 * d
  have h : d = 7 := by sorry
  have h1 : a_1002 = 6996 := by sorry
  have h2 : a_1008 = 7038 := by sorry
  have h3 := abs_eq_iff.mpr ⟨7038 - 6996, by norm_num⟩
  exact h3

end arithmetic_sequence_difference_l66_66427


namespace geometric_sequence_right_triangle_l66_66583

theorem geometric_sequence_right_triangle (q : ℝ) (h_q : q > 1) :
  ∃ (m : ℝ), (m / q)^2 + m^2 = (m * q)^2 ∧ q^2 = (Math.sqrt 5 + 1) / 2 :=
by
  sorry

end geometric_sequence_right_triangle_l66_66583


namespace part1_l66_66249

theorem part1 (m : ℝ) (h1 : ∀ x ∈ (Set.Icc 1 2), let y := x^2 - (m+2)*x + 3 in true)
  (h2 : ∀ x ∈ (Set.Icc 1 2), let y := x^2 - (m+2)*x + 3 in
    ∃ M N : ℝ, y = M ∧ y = N ∧ M - N ≤ 2) :
  -1 ≤ m ∧ m ≤ 3 :=
sorry

end part1_l66_66249


namespace find_average_age_of_students_l66_66301

-- Given conditions
variables (n : ℕ) (T : ℕ) (A : ℕ)

-- 20 students in the class
def students : ℕ := 20

-- Teacher's age is 42 years
def teacher_age : ℕ := 42

-- When the teacher's age is included, the average age increases by 1
def average_age_increase (A : ℕ) := A + 1

-- Proof problem statement in Lean 4
theorem find_average_age_of_students (A : ℕ) :
  20 * A + 42 = 21 * (A + 1) → A = 21 :=
by
  -- Here should be the proof steps, added sorry to skip the proof
  sorry

end find_average_age_of_students_l66_66301


namespace bryce_received_15_raisins_l66_66971

theorem bryce_received_15_raisins (x : ℕ) (c : ℕ) (h1 : c = x - 10) (h2 : c = x / 3) : x = 15 :=
by
  sorry

end bryce_received_15_raisins_l66_66971


namespace present_death_rate_l66_66007

theorem present_death_rate (birth_rate : ℕ) (percentage_increase : ℝ) (h1 : birth_rate = 32) (h2 : percentage_increase = 0.021) : ℝ :=
by
  let death_rate := birth_rate - 21
  have h : death_rate = 11 :=
    by
      rw [h1, Nat.cast_sub (by norm_num) (by norm_num)]
      norm_num
  exact death_rate

end present_death_rate_l66_66007


namespace sum_odd_divisors_90_eq_78_l66_66052

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l66_66052


namespace coefficient_x3_in_binomial_expansion_l66_66647

theorem coefficient_x3_in_binomial_expansion :
  (binom 20 3) * 2^(20 - 3) = 149442048 := by
  sorry

end coefficient_x3_in_binomial_expansion_l66_66647


namespace min_triangles_l66_66796

def point (n : ℕ) := { x // x < n }
def num_points := 1994
def num_groups := 83
def min_group_size := 3

theorem min_triangles (P : fin num_points → point num_points)
  (h_no_three_collinear : ∀ (a b c : point num_points), a ≠ b → b ≠ c → a ≠ c → ¬collinear a b c)
  (h_groups : ∃ (groups : fin num_groups → finset (point num_points)),
    (∀ i, min_group_size ≤ (groups i).card) ∧
    (∀ i j, i ≠ j → disjoint (groups i) (groups j)) ∧
    finset.univ = finset.bUnion finset.univ groups) :
  ∃ (G : finset (point num_points) × finset (point num_points)),
  (∀ {g}, g ∈ G → g.1 ∈ finset.univ ∧ g.2 ∈ finset.univ ∧ (∃ i, g.1 ∈ groups i ∧ g.2 ∈ groups i)) ∧
  min_triangles P h_no_three_collinear h_groups = 168544 :=
sorry

end min_triangles_l66_66796


namespace slope_of_line_l66_66011

theorem slope_of_line : ∀ (x y : ℝ), 2 * x - 4 * y + 7 = 0 → (y = (1/2) * x - 7 / 4) :=
by
  intro x y h
  -- This would typically involve rearranging the given equation to the slope-intercept form
  -- but as we are focusing on creating the statement, we insert sorry to skip the proof
  sorry

end slope_of_line_l66_66011


namespace smallest_Q2_l66_66358

def Q (x : ℝ) (k m : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 + k*x + m

theorem smallest_Q2 {k m : ℝ} (hk : k = -16) (hm : m = 16) :
  let Q_at_neg2 := Q (-2) k m
  let abs_sum_coeff := |1| + |(-2)| + |3| + |k| + |m|
  let prod_zeros := m -- when leading coefficient 1, product of roots = constant term
  let Q_at_2 := Q (2) k m
  let sum_zeros := 2 -- since sum of roots = - (-2) / 1 = 2 
  Q_at_2 < Q_at_neg2 ∧ Q_at_2 < abs_sum_coeff ∧ Q_at_2 < prod_zeros ∧ Q_at_2 < sum_zeros := 
begin
  sorry
end

end smallest_Q2_l66_66358


namespace total_fruits_in_30_days_l66_66752

-- Define the number of oranges Sophie receives each day
def sophie_daily_oranges : ℕ := 20

-- Define the number of grapes Hannah receives each day
def hannah_daily_grapes : ℕ := 40

-- Define the number of days
def number_of_days : ℕ := 30

-- Calculate the total number of fruits received by Sophie and Hannah in 30 days
theorem total_fruits_in_30_days :
  (sophie_daily_oranges * number_of_days) + (hannah_daily_grapes * number_of_days) = 1800 :=
by
  sorry

end total_fruits_in_30_days_l66_66752


namespace johns_weekly_earnings_l66_66670

-- Define the conditions as constants
constant baskets_per_week : ℕ := 3
constant collections_per_week : ℕ := 2
constant crabs_per_basket : ℕ := 4
constant price_per_crab : ℕ := 3

-- Define what we need to prove as a theorem
theorem johns_weekly_earnings :
  let total_baskets := baskets_per_week * collections_per_week in
  let total_crabs := total_baskets * crabs_per_basket in
  let total_money := total_crabs * price_per_crab in
  total_money = 72 := 
by
  sorry

end johns_weekly_earnings_l66_66670


namespace prob_not_adjacent_B_C_A_correct_l66_66303

-- we need to use noncomputable since we have factorials which are not explicitly computed here
noncomputable def num_total_sequences : ℕ := (5!)

noncomputable def num_valid_sequences : ℕ :=
  let ways_case1 := 24
  let ways_case2 := 12
  ways_case1 + ways_case2

def prob_not_adjacent_B_C_A : ℚ := num_valid_sequences / num_total_sequences

theorem prob_not_adjacent_B_C_A_correct :
  prob_not_adjacent_B_C_A = 3 / 10 :=
begin
  -- proof will be completed here
  sorry
end

end prob_not_adjacent_B_C_A_correct_l66_66303


namespace sumOddDivisorsOf90_l66_66050

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l66_66050


namespace distance_between_foci_of_hyperbola_l66_66535

-- Definition of the hyperbola in question
def hyperbola : (ℝ × ℝ) → Prop := λ p : ℝ × ℝ, let (x, y) := p in
  9 * x^2 - 18 * x - 16 * y^2 + 32 * y = 72

-- The statement to be proven
theorem distance_between_foci_of_hyperbola :
  ∀ (x y : ℝ), hyperbola (x, y) → ∃ c : ℝ, 2 * c ≈ 6.72 :=
sorry

end distance_between_foci_of_hyperbola_l66_66535


namespace line_eq_l66_66896

theorem line_eq :
  ∃ (l : ℝ × ℝ → ℝ), (∀ x y : ℝ, l (x, y) = 0 ↔ 3 * x + 4 * y + c = 0) ∧
  (let a := -(c / 3), 
       b := -(c / 4)
   in a + b = 7 / 3 := 3 * x + 4 * y - 4 = 0) :=
sorry

end line_eq_l66_66896


namespace hexagon_monochromatic_triangle_l66_66889

-- Define the probability of monochromatic triangles in a hexagon
noncomputable def probability_monochromatic_triangle : ℝ := 0.99683

-- Define the main theorem to prove the probability condition
theorem hexagon_monochromatic_triangle :
  ∃ (G : SimpleGraph (Fin 6)),
    (∀ e ∈ G.edgeSet, e.color = Green ∨ e.color = Yellow) →
    (random_colored_prob G ≈ probability_monochromatic_triangle) :=
by
  sorry

end hexagon_monochromatic_triangle_l66_66889


namespace cosine_squared_identity_l66_66238

theorem cosine_squared_identity (α : ℝ) (h : sin(2 * α) = 1 / 3) :
  cos^2 (α - π / 4) = 2 / 3 :=
by
  sorry

end cosine_squared_identity_l66_66238


namespace georgia_makes_muffins_l66_66178

-- Definitions based on conditions
def muffinRecipeMakes : ℕ := 6
def numberOfStudents : ℕ := 24
def durationInMonths : ℕ := 9

-- Theorem to prove the given problem
theorem georgia_makes_muffins :
  (numberOfStudents / muffinRecipeMakes) * durationInMonths = 36 :=
by
  -- We'll skip the proof with sorry
  sorry

end georgia_makes_muffins_l66_66178


namespace integers_satisfying_condition_l66_66031

-- Define the condition
def condition (x : ℤ) : Prop := x * x < 3 * x

-- Define the theorem stating the proof problem
theorem integers_satisfying_condition :
  {x : ℤ | condition x} = {1, 2} :=
by
  sorry

end integers_satisfying_condition_l66_66031


namespace curves_intersect_at_four_points_l66_66779

theorem curves_intersect_at_four_points (a : ℝ) :
  (∃ x y : ℝ, (x^2 + y^2 = a^2 ∧ y = -x^2 + a ) ∧ 
   (0 = x ∧ y = a) ∧ 
   (∃ t : ℝ, x = t ∧ (y = 1 ∧ x^2 = a - 1))) ↔ a = 2 := 
by
  sorry

end curves_intersect_at_four_points_l66_66779


namespace coeff_x4_expansion_l66_66766

theorem coeff_x4_expansion : coeff (expand (x - 3 * real.sqrt 2) 8) 4 = 22680 := by
  sorry

end coeff_x4_expansion_l66_66766


namespace part_one_part_two_zeros_part_two_ones_l66_66959

noncomputable def f (m x : ℝ) := log (1 + m * x) + x^2 / 2 - m * x

theorem part_one (x : ℝ) (hxm : 0 < 1) (hx_range : -1 < x ∧ x ≤ 0):
  f 1 x ≤ x^3 / 3 :=
  sorry

theorem part_two_zeros (m : ℝ) (hm : 0 < m ∧ m < 1):
  ∃! x : ℝ, f m x = 0 :=
  sorry

theorem part_two_ones (hm : m = 1):
  ∃! x : ℝ, f m x = 0 :=
  sorry

end part_one_part_two_zeros_part_two_ones_l66_66959


namespace shaded_area_of_square_with_quarter_circles_l66_66158

theorem shaded_area_of_square_with_quarter_circles :
  let side_len : ℝ := 12
  let square_area := side_len * side_len
  let radius := side_len / 2
  let total_circle_area := 4 * (π * radius^2 / 4)
  let shaded_area := square_area - total_circle_area
  shaded_area = 144 - 36 * π := 
by
  sorry

end shaded_area_of_square_with_quarter_circles_l66_66158


namespace find_x_plus_inv_x_l66_66941

theorem find_x_plus_inv_x (x : ℝ) (hx : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 := 
by 
  sorry

end find_x_plus_inv_x_l66_66941


namespace sum_odd_divisors_of_90_l66_66110

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l66_66110


namespace calc1_calc2_calc3_calc4_l66_66512

theorem calc1 : 327 + 46 - 135 = 238 := by sorry
theorem calc2 : 1000 - 582 - 128 = 290 := by sorry
theorem calc3 : (124 - 62) * 6 = 372 := by sorry
theorem calc4 : 500 - 400 / 5 = 420 := by sorry

end calc1_calc2_calc3_calc4_l66_66512


namespace inequality_solution_l66_66711

theorem inequality_solution (x : ℝ) : 
  (∀ y : ℝ, y = Real.log x 3 → -2 ≤ y ∧ y < 2 ∧ y ≠ -1 →
    (2 * 2^(-y) - 4) * real.sqrt (2 - real.sqrt (y + 2)) / (1 + real.sqrt (y + 5)) >
    (2^(-y) - 2) * real.sqrt (2 - real.sqrt (y + 2)) / (real.sqrt (y + 5) - 2)) →
  x ∈ (Ioi 0 ∩ Iio (1/3)) ∪ (Ioi (1/3) ∩ Iic (1 / real.sqrt 3)) ∪ Ioi (real.sqrt 3) :=
sorry

end inequality_solution_l66_66711


namespace triangle_area_ratio_l66_66797

theorem triangle_area_ratio
    (a b c : ℝ)
    (h_pos_a : 0 < a)
    (h_pos_b : 0 < b)
    (h_pos_c : 0 < c) :
    let S_ABC := 0.5 * a * b * sin (acos ((a^2 + b^2 - c^2) / (2*a*b))) in
    let S_ODCE := ((a^2 * S_ABC + a * c * S_ABC) / ((a+c) * (b+c) * (a+b+c))) - (a * c * S_ABC / ((b+c) * (a+b+c))) in
    S_ABC / S_ODCE = ((a+c)*(b+c)*(a+b+c)) / (a * b * (a+b+2*c)) :=
by
  sorry

end triangle_area_ratio_l66_66797


namespace expected_elements_mn_sum_l66_66170

namespace ProofProblem

noncomputable def expectedElementsInIntersection : ℚ :=
  1 / (36 ^ 3) * (1 * 8 ^ 3 + 2 * 7 ^ 3 + 3 * 6 ^ 3 + 4 * 5 ^ 3 +
                  5 * 4 ^ 3 + 6 * 3 ^ 3 + 7 * 2 ^ 3 + 8 * 1 ^ 3)

theorem expected_elements_mn_sum :
  let m := 178 in
  let n := 243 in
  let E := (1 / (36 ^ 3) * (1 * 8 ^ 3 + 2 * 7 ^ 3 + 3 * 6 ^ 3 + 4 * 5 ^ 3 +
                           5 * 4 ^ 3 + 6 * 3 ^ 3 + 7 * 2 ^ 3 + 8 * 1 ^ 3)) in
  E = m / n → m + n = 421 :=
by
  sorry

end ProofProblem

end expected_elements_mn_sum_l66_66170


namespace max_value_expression_l66_66926

theorem max_value_expression (x y : ℝ) (hx : 0 < x) (hx' : x < (π / 2)) (hy : 0 < y) (hy' : y < (π / 2)) :
  (A : ℝ) = (Real.sqrt (Real.sqrt (Real.sin x * Real.sin y))) / (Real.sqrt (Real.sqrt (Real.tan x)) + Real.sqrt (Real.sqrt (Real.tan y))) ≤ Real.sqrt (Real.sqrt 8) / 4 :=
sorry

end max_value_expression_l66_66926


namespace matrix_non_invertible_at_36_31_l66_66544

-- Define the matrix A
def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2 * x, 9], ![4 - x, 11]]

-- State the theorem
theorem matrix_non_invertible_at_36_31 :
  ∃ x : ℝ, (A x).det = 0 ∧ x = 36 / 31 :=
by {
  sorry
}

end matrix_non_invertible_at_36_31_l66_66544


namespace simple_interest_l66_66740

theorem simple_interest (TD : ℝ) (Sum : ℝ) (SI : ℝ) 
  (h1 : TD = 78) 
  (h2 : Sum = 947.1428571428571) 
  (h3 : SI = Sum - (Sum - TD)) : 
  SI = 78 := 
by 
  sorry

end simple_interest_l66_66740


namespace rationalize_denominator_eq_l66_66701

noncomputable def rationalize_denominator : ℝ :=
  18 / (Real.sqrt 36 + Real.sqrt 2)

theorem rationalize_denominator_eq : rationalize_denominator = (54 / 17) - (9 * Real.sqrt 2 / 17) := 
by
  sorry

end rationalize_denominator_eq_l66_66701


namespace number_of_locations_l66_66370

-- Definition for points and distance
structure Point (α : Type*) := (x : α) (y : α)

-- Definition of distance between points
def distance {α : Type*} [LinearOrderedField α] (P Q : Point α) : α :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Definition for area of a triangle given vertices A, B, and C
def area {α : Type*} [LinearOrderedField α] (A B C : Point α) : α :=
  (1 / 2) * abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)))

-- Main theorem stating the problem
theorem number_of_locations 
  (A B : Point ℝ)
  (hAB : distance A B = 10)
  (hArea : ∀ C : Point ℝ, area A B C = 20 → 
    ∃ (C_count : ℕ), C_count = 8) : 
  True := 
sorry

end number_of_locations_l66_66370


namespace sum_of_solutions_l66_66775

open Real

theorem sum_of_solutions :
  (∑ x in {x | sqrt((x + 5)^2) = 8} :ℝ, x) = -10 :=
by
  sorry

end sum_of_solutions_l66_66775


namespace number_of_correct_propositions_l66_66392

def curve (t : ℝ) : Set ℝ := {p | ∃ x y: ℝ, x^2 / (4 - t) + y^2 / (t - 1) = 1}

def ellipse (t : ℝ) : Prop := 1 < t ∧ t < 4
def hyperbola (t : ℝ) : Prop := t < 1 ∨ t > 4
def circle (t : ℝ) : Prop := t = 5 / 2
def ellipse_foci_on_x_axis (t : ℝ) : Prop := 1 < t ∧ t < 5 / 2

theorem number_of_correct_propositions :
  let prop1 := ellipse (t)
  let prop2 := hyperbola (t)
  let prop3 := ¬ circle (t)
  let prop4 := ellipse_foci_on_x_axis (t)
  (if prop1 then 1 else 0) + (if prop2 then 1 else 0) + (if prop3 then 1 else 0) + (if prop4 then 1 else 0) = 2 :=
begin
  -- proof would go here
  sorry
end

end number_of_correct_propositions_l66_66392


namespace factor_M_l66_66222

theorem factor_M (a b c d : ℝ) : 
  ((a - c)^2 + (b - d)^2) * (a^2 + b^2) - (a * d - b * c)^2 =
  (a * c + b * d - a^2 - b^2)^2 :=
by
  sorry

end factor_M_l66_66222


namespace max_value_in_interval_inequality_l66_66590

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (1 - 2 * a) * x - Real.log x

def max_value_f (a : ℝ) : ℝ :=
  if a >= -1 / 4 then 2 - Real.log 2
  else if -1 / 2 <= a then 1 - 1 / (4 * a) + Real.log (-2 * a)
  else 1 - a

theorem max_value_in_interval (a : ℝ) : ∃ x ∈ Set.Icc 1 2, f a x = max_value_f a :=
  sorry

theorem inequality (a x₀ x₁ x₂ y₁ y₂ : ℝ) (h₀ : x₀ ≠ (x₁ + x₂) / 2) (hx₁ : y₁ = f a x₁) (hx₂ : y₂ = f a x₂) : 
  f' x₀ a > (y₁ - y₂) / (x₁ - x₂) := sorry

noncomputable def f' (x₀ a : ℝ) : ℝ := 2 * a * x₀ + 1 - 2 * a - 1 / x₀

end max_value_in_interval_inequality_l66_66590


namespace compute_floor_expression_l66_66194

theorem compute_floor_expression :
  Int.floor ((2005^3 / (2003 * 2004)) - (2003^3 / (2004 * 2005))) = 8 :=
by sorry

end compute_floor_expression_l66_66194


namespace part1_1_part1_2_part1_3_part2_l66_66200

def operation (a b c : ℝ) : Prop := a^c = b

theorem part1_1 : operation 3 81 4 :=
by sorry

theorem part1_2 : operation 4 1 0 :=
by sorry

theorem part1_3 : operation 2 (1 / 4) (-2) :=
by sorry

theorem part2 (x y z : ℝ) (h1 : operation 3 7 x) (h2 : operation 3 8 y) (h3 : operation 3 56 z) : x + y = z :=
by sorry

end part1_1_part1_2_part1_3_part2_l66_66200


namespace intersect_x_axis_once_l66_66948

theorem intersect_x_axis_once (k : ℝ) : 
  (∀ x : ℝ, (k - 3) * x^2 + 2 * x + 1 = 0 → x = 0) → (k = 3 ∨ k = 4) :=
by
  intro h
  sorry

end intersect_x_axis_once_l66_66948


namespace non_zero_real_y_satisfies_l66_66431

theorem non_zero_real_y_satisfies (y : ℝ) (h : y ≠ 0) : (8 * y) ^ 3 = (16 * y) ^ 2 → y = 1 / 2 :=
by
  -- Lean code placeholders
  sorry

end non_zero_real_y_satisfies_l66_66431


namespace part1_part2_part3_l66_66957

-- Define the function f(x) and the given conditions
def f (x : ℝ) : ℝ := (x / (x^2 - 4))

theorem part1 (h_odd : ∀ x, f (-x) = -f (x))
  (h_value : f 1 = -1/3) : ((1 : ℝ), (0 : ℝ)) := sorry

theorem part2 : ∀ x1 x2 ∈ Ioo (-2 : ℝ) 2, (x1 < x2 → f x1 > f x2) := sorry

theorem part3 (t : ℝ) (ht : t ∈ Ioo (-2 : ℝ) 2) : f (t - 1) + f t < 0 ↔ t ∈ Ioo (1/2) 2 := sorry

end part1_part2_part3_l66_66957


namespace total_money_made_l66_66008

theorem total_money_made (adult_ticket_price child_ticket_price total_tickets_sold child_tickets_sold : ℕ)
    (h1 : adult_ticket_price = 5) (h2 : child_ticket_price = 3)
    (h3 : total_tickets_sold = 42) (h4 : child_tickets_sold = 16) :
    (42 - 16) * 5 + 16 * 3 = 178 :=
by 
  rw [h3, h4, h1, h2]
  sorry

end total_money_made_l66_66008


namespace triangle_angle_sum_abc_gt_120_l66_66396

theorem triangle_angle_sum_abc_gt_120 {A B C D : Point} 
  (h_triangle_ABC : triangle A B C)
  (h_midpoint_D : midpoint D A C)
  (h_median_BD : ∀ (l_AB l_BC : ℝ), 0 < l_AB → 0 < l_BC → 
                  (BD < l_AB / 2) ∧ (BD < l_BC / 2)) : 
  ∠ABC > 120 :=
sorry

end triangle_angle_sum_abc_gt_120_l66_66396


namespace tangent_line_max_value_extreme_value_l66_66955

section part_I
variable (X : Type) [TopologicalSpace X] [ChartedSpace X ℝ] [HasSmoothTangentBundle X]

def f (a b : ℝ) (x : ℝ) : ℝ := a * real.log x + b * x

theorem tangent_line (a b : ℝ) (f : ℝ → ℝ) (hf_def : f = λ x, a * real.log x + b * x) :
  a = 1 ∧ b = 1 → (∀ x : ℝ, x = 2 * 1 - x - 1) :=
by sorry
end part_I

section part_II
def f (a : ℝ) (x : ℝ) : ℝ := a * real.log x - 2 * x

theorem max_value (a : ℝ) :
  a > 0 → 
  (a ≤ 2 → ∀ x ∈ Icc 1 2, f a x ≤ -2) ∧
  (2 < a ∧ a < 4 → ∀ x ∈ Icc 1 2, f a x ≤ a * real.log (a / 2) - a) ∧
  (a ≥ 4 → ∀ x ∈ Icc 1 2, f a x ≤ a * real.log 2 - 4) :=
by sorry
end part_II

section part_III
def g (b : ℝ) (x : ℝ) : ℝ := real.log x + b * x + real.sin x

theorem extreme_value (b : ℝ) :
  (∀ x ∈ Ioo 0 π, 
    (b < 1 - 1 / π → g b x has_local_max (g b x)) ∧ 
    (b ≥ 1 - 1 / π → ∀ u > 0, g b x < g b (x + u))) :=
by sorry
end part_III

end tangent_line_max_value_extreme_value_l66_66955


namespace min_punchers_needed_l66_66828

theorem min_punchers_needed : 
  ∀ (P : ℝ × ℝ) (a b c : ℝ), 
  ∃ (A B C : ℝ × ℝ), (dist P A ∉ ℚ) ∨ (dist P B ∉ ℚ) ∨ (dist P C ∉ ℚ) :=
by {
  sorry
}

end min_punchers_needed_l66_66828


namespace complex_solution_exists_l66_66513

open Complex

theorem complex_solution_exists (z : ℂ) (h : 2 * z - 4 * conj z = -4 - 40 * I) : 
  z = 2 - (20 / 3) * I :=
sorry

end complex_solution_exists_l66_66513


namespace sum_of_odd_divisors_l66_66065

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l66_66065


namespace tan_alpha_plus_pi_over_12_l66_66239

theorem tan_alpha_plus_pi_over_12 (α : ℝ) (h : Real.sin α = 3 * Real.sin (α + π / 6)) :
  Real.tan (α + π / 12) = 2 * Real.sqrt 3 - 4 :=
by
  sorry

end tan_alpha_plus_pi_over_12_l66_66239


namespace first_part_lending_years_l66_66843

-- Definitions and conditions from the problem
def total_sum : ℕ := 2691
def second_part : ℕ := 1656
def rate_first_part : ℚ := 3 / 100
def rate_second_part : ℚ := 5 / 100
def time_second_part : ℕ := 3

-- Calculated first part
def first_part : ℕ := total_sum - second_part

-- Prove that the number of years (n) the first part is lent is 8
theorem first_part_lending_years : 
  ∃ n : ℕ, (first_part : ℚ) * rate_first_part * n = (second_part : ℚ) * rate_second_part * time_second_part ∧ n = 8 :=
by
  -- Proof steps would go here
  sorry

end first_part_lending_years_l66_66843


namespace division_result_l66_66901

theorem division_result :
  let f := λ x : ℕ, x^5 - 25 * x^3 + 13 * x^2 - 16 * x + 12
  let g := λ x : ℕ, x - 3
  let q := λ x : ℕ, x^4 + 3 * x^3 - 16 * x^2 - 35 * x - 121
  let r := -297
  ∃ q r, ∀ x, f x = g x * q x + r
:=
sorry

end division_result_l66_66901


namespace range_of_k_l66_66351

def f (k x : ℝ) : ℝ := (x^4 + k * x^2 + 1) / (x^4 + x^2 + 1)

theorem range_of_k (k : ℝ) :
  (-1/2 < k ∧ k < 4) ↔
  (∀ a b c : ℝ, (f k a + f k b > f k c) ∧ (f k a + f k c > f k b) ∧ (f k b + f k c > f k a)) :=
sorry

end range_of_k_l66_66351


namespace sum_of_positive_odd_divisors_of_90_l66_66072

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l66_66072


namespace hurricane_damage_approximation_l66_66143

-- Define the conditions
def damage_in_yen : ℕ := 4000000000
def yen_to_usd : ℝ := 110

-- Define the expected result
def expected_damage_in_usd : ℝ := 36363636

-- Prove the statement
theorem hurricane_damage_approximation :
  (damage_in_yen / yen_to_usd) ≈ expected_damage_in_usd :=
by
  sorry

end hurricane_damage_approximation_l66_66143


namespace distance_between_points_l66_66771

-- Definition of a coordinate point in a 2D plane
structure Point2D where
  x: ℝ
  y: ℝ

-- Function to calculate the distance between two points
def distance (p1 p2: Point2D) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Prove that the distance between points (3,7) and (-5,0) is √113
theorem distance_between_points :
  distance {x := 3, y := 7} {x := -5, y := 0} = real.sqrt 113 := by
  sorry

end distance_between_points_l66_66771


namespace conics_through_centroid_are_hyperbolas_locus_of_centers_is_steiner_inellipse_l66_66800

-- Definitions and conditions for the problem
variable {O A B : Point}
variable {G : Point} (is_centroid : centroid O A B = G)

-- Part (a): Prove that all conics passing through the points O, A, B, G are hyperbolas.
theorem conics_through_centroid_are_hyperbolas (O A B G : Point) 
    (hG : centroid O A B = G) :
    ∀ (K : ConicSection), passesThrough K O ∧ passesThrough K A ∧ passesThrough K B ∧ passesThrough K G → isHyperbola K :=
sorry

-- Part (b): Locus of centers of these hyperbolas is the Steiner inellipse of triangle OAB.
theorem locus_of_centers_is_steiner_inellipse (O A B G : Point) 
    (hG : centroid O A B = G) :
    locusOfCenters {K : ConicSection | passesThrough K O ∧ passesThrough K A ∧ passesThrough K B ∧ passesThrough K G} 
    = steinerInellipse O A B :=
sorry

end conics_through_centroid_are_hyperbolas_locus_of_centers_is_steiner_inellipse_l66_66800


namespace exists_sum_divisible_by_2022_l66_66757

theorem exists_sum_divisible_by_2022 (a : Fin 2022 → ℤ) :
  ∃ (s : ℤ), (∃ (i j : Fin 2022), i ≤ j ∧ s = (∑ k in Finset.Ico i j.succ, a k) ∧ s % 2022 = 0) :=
by
  sorry

end exists_sum_divisible_by_2022_l66_66757


namespace tenth_integer_from_permutations_l66_66004

theorem tenth_integer_from_permutations : ∃ n : ℕ, nth_permutation [1, 2, 5, 6] n = 2561 :=
by
  sorry

def nth_permutation (digits : List ℕ) (n : ℕ) : ℕ :=
  sorry

end tenth_integer_from_permutations_l66_66004


namespace extreme_value_derivative_derivative_not_extreme_value_l66_66452

theorem extreme_value_derivative {f : ℝ → ℝ} {x₀ : ℝ} :
  (∃ ε > 0, ∀ x ∈ Icc (x₀ - ε) (x₀ + ε), f x ≤ f x₀) → (deriv f x₀ = 0) :=
sorry

theorem derivative_not_extreme_value {f : ℝ → ℝ} {x₀ : ℝ} :
  (deriv f x₀ = 0) → ¬(∃ ε > 0, ∀ x ∈ Icc (x₀ - ε) (x₀ + ε), f x ≤ f x₀) :=
sorry

end extreme_value_derivative_derivative_not_extreme_value_l66_66452


namespace find_n_l66_66854

theorem find_n (a1 a2 : ℕ) (s2 s1 : ℕ) (n : ℕ) :
    a1 = 12 →
    a2 = 3 →
    s2 = 3 * s1 →
    ∃ n : ℕ, a1 / (1 - a2/a1) = 16 ∧
             a1 / (1 - (a2 + n) / a1) = s2 →
             n = 6 :=
by
  intros
  sorry

end find_n_l66_66854


namespace total_capacity_of_two_tanks_l66_66609

-- Conditions
def tank_A_initial_fullness : ℚ := 3 / 4
def tank_A_final_fullness : ℚ := 7 / 8
def tank_A_added_volume : ℚ := 5

def tank_B_initial_fullness : ℚ := 2 / 3
def tank_B_final_fullness : ℚ := 5 / 6
def tank_B_added_volume : ℚ := 3

-- Proof statement
theorem total_capacity_of_two_tanks :
  let tank_A_total_capacity := tank_A_added_volume / (tank_A_final_fullness - tank_A_initial_fullness)
  let tank_B_total_capacity := tank_B_added_volume / (tank_B_final_fullness - tank_B_initial_fullness)
  tank_A_total_capacity + tank_B_total_capacity = 58 := 
sorry

end total_capacity_of_two_tanks_l66_66609


namespace price_change_38_percent_l66_66842

variables (P : ℝ) (x : ℝ)
noncomputable def final_price := P * (1 - (x / 100)^2) * 0.9
noncomputable def target_price := 0.77 * P

theorem price_change_38_percent (h : final_price P x = target_price P):
  x = 38 := sorry

end price_change_38_percent_l66_66842


namespace sum_odd_divisors_of_90_l66_66111

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l66_66111


namespace linear_function_does_not_pass_fourth_quadrant_l66_66733

theorem linear_function_does_not_pass_fourth_quadrant :
  ∀ x, (2 * x + 1 ≥ 0) :=
by sorry

end linear_function_does_not_pass_fourth_quadrant_l66_66733


namespace minimum_value_of_sum_l66_66207

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ (1 / real.cbrt 2) :=
by
  sorry

end minimum_value_of_sum_l66_66207


namespace winter_melon_ratio_l66_66467

theorem winter_melon_ratio (T Ok_sales Choc_sales : ℕ) (hT : T = 50) 
  (hOk : Ok_sales = 3 * T / 10) (hChoc : Choc_sales = 15) :
  (T - (Ok_sales + Choc_sales)) / T = 2 / 5 :=
by
  sorry

end winter_melon_ratio_l66_66467


namespace largest_prime_ensuring_conditions_l66_66897

theorem largest_prime_ensuring_conditions :
  ∃ p : ℕ, 
    (Nat.Prime p) 
    ∧ (∃ x : ℕ, (p + 1 = 2 * x^2)) 
    ∧ (∃ y : ℕ, (p^2 + 1 = 2 * y^2)) 
    ∧ (∀ q : ℕ, (Nat.Prime q 
                  ∧ ∃ x : ℕ, (q + 1 = 2 * x^2) 
                  ∧ ∃ y : ℕ, (q^2 + 1 = 2 * y^2)) → (q ≤ p)) :=
begin
  use 7,
  -- Provide necessary proof steps here
  sorry
end

end largest_prime_ensuring_conditions_l66_66897


namespace union_complement_l66_66599

open Set

-- Definitions based on conditions
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}
def C_UA : Set ℕ := U \ A

-- The theorem to prove
theorem union_complement :
  (C_UA ∪ B) = {0, 2, 4, 5, 6} :=
by
  sorry

end union_complement_l66_66599


namespace maximum_good_triangles_l66_66384

theorem maximum_good_triangles :
  ∃ (a b : ℕ → ℕ) (N : ℕ),
    (∀ i j, 1 ≤ i → i < j → j ≤ 100 → |a i * b j - a j * b i| = 1 → true) ∧
    (N = 197) :=
sorry

end maximum_good_triangles_l66_66384


namespace logan_passengers_correct_l66_66620

def total_passengers : ℝ := 38.3
def passengers_kennedy : ℝ := (1/3) * total_passengers
def direct_flights_kennedy : ℝ := 0.25 * passengers_kennedy
def layovers_kennedy : ℝ := 0.75 * passengers_kennedy
def passengers_miami : ℝ := (1/2) * passengers_kennedy
def direct_flights_miami : ℝ := 0.4 * passengers_miami
def layovers_miami : ℝ := 0.6 * passengers_miami
def passengers_logan : ℝ := 0.2 * layovers_miami
def passengers_sf : ℝ := passengers_logan / 4

theorem logan_passengers_correct : passengers_logan = 0.766 :=
by
  sorry

end logan_passengers_correct_l66_66620


namespace min_shift_periodic_odd_func_l66_66954

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 3)

theorem min_shift_periodic_odd_func (m : ℝ) (hm : m > 0)
  (H : ∀ x, f (x - m) = -f (-x + m)) : m = π / 6 :=
sorry

end min_shift_periodic_odd_func_l66_66954


namespace lines_parallel_l66_66389

-- Definitions of the geometric objects and conditions.
variables {Point : Type} [euclidean_geometry Point]
variables (ω₁ ω₂ : circle Point) (K L : Point) (ell : line Point)
variables (A B C D P Q : Point)

-- Conditions stated in the problem.
hypotheses
  (h_intersect_circles : K ≠ L ∧ K ∈ ω₁ ∧ L ∈ ω₁ ∧ K ∈ ω₂ ∧ L ∈ ω₂)
  (h_line_intersects_ω₁ : A ∈ ell ∧ C ∈ ell ∧ A ∈ ω₁ ∧ C ∈ ω₁)
  (h_line_intersects_ω₂ : B ∈ ell ∧ D ∈ ell ∧ B ∈ ω₂ ∧ D ∈ ω₂)
  (h_ordering : ordered {A, B, C, D} ell)
  (h_projections : P ⊥ KL ∧ P ∈ B ∧ Q ⊥ KL ∧ Q ∈ C)

-- Conclusion to prove.
theorem lines_parallel : parallel (line_through A P) (line_through D Q) :=
sorry

end lines_parallel_l66_66389


namespace bicycle_cost_price_l66_66838

theorem bicycle_cost_price 
  (CP_A : ℝ) 
  (H : CP_A * (1.20 * 0.85 * 1.30 * 0.90) = 285) : 
  CP_A = 285 / (1.20 * 0.85 * 1.30 * 0.90) :=
sorry

end bicycle_cost_price_l66_66838


namespace find_a_n_l66_66278

noncomputable def a_sequence (n : ℕ) : ℕ :=
if n = 1 then 1 else 3 * a_sequence (n - 1) + 2

theorem find_a_n (n : ℕ) : a_sequence n = 2 * 3^(n - 1) - 1 :=
by
  sorry

end find_a_n_l66_66278


namespace polynomial_divisibility_l66_66518

theorem polynomial_divisibility (a : ℤ) : ∃ q : ℤ[X], (X^13 + X + 90 : ℤ[X]) = (X^2 - X + a) * q ↔ a = -2 :=
by sorry

end polynomial_divisibility_l66_66518


namespace rational_numbers_on_circle_l66_66255

theorem rational_numbers_on_circle (a b c d e f : ℚ)
  (h1 : a = |b - c|)
  (h2 : b = d)
  (h3 : c = |d - e|)
  (h4 : d = |e - f|)
  (h5 : e = f)
  (h6 : a + b + c + d + e + f = 1) :
  [a, b, c, d, e, f] = [1/4, 1/4, 0, 1/4, 1/4, 0] :=
sorry

end rational_numbers_on_circle_l66_66255


namespace probability_of_prime_l66_66428

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sectors : List ℕ := [3, 6, 1, 4, 5, 2, 7, 9]
def prime_sectors : List ℕ := sectors.filter is_prime
def total_sectors : ℕ := sectors.length
def favorable_sectors : ℕ := prime_sectors.length
def probability_prime : ℚ := favorable_sectors / total_sectors

theorem probability_of_prime : probability_prime = 1 / 2 := by
  sorry

end probability_of_prime_l66_66428


namespace height_percentage_difference_l66_66791

theorem height_percentage_difference
  (h_B h_A : ℝ)
  (hA_def : h_A = h_B * 0.55) :
  ((h_B - h_A) / h_A) * 100 = 81.82 := by 
  sorry

end height_percentage_difference_l66_66791


namespace liz_prob_at_least_half_l66_66505

noncomputable def binom (n k : ℕ) : ℕ := 
  Nat.choose n k

noncomputable def binom_prob (n k : ℕ) (p : ℚ) : ℚ := 
  (binom n k : ℚ) * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def prob_at_least_half_correct (n : ℕ) (p : ℚ) : ℚ := 
  (Finset.range (n + 1)).filter (fun k => k ≥ n / 2).sum (binom_prob n · p)

theorem liz_prob_at_least_half (n : ℕ) (p : ℚ) (h_n : n = 10) (h_p : p = 1/3) :
  prob_at_least_half_correct n p = 161 / 2187 := 
by
  sorry

end liz_prob_at_least_half_l66_66505


namespace total_amount_l66_66365

theorem total_amount (P Q R : ℝ) (h1 : R = 2 / 3 * (P + Q)) (h2 : R = 3200) : P + Q + R = 8000 := 
by
  sorry

end total_amount_l66_66365


namespace problem_f_3_l66_66918

def f : ℤ → ℤ 
| x := if x ≥ 6 then x - 5 else f (x + 1)

theorem problem_f_3 : f 3 = 1 :=
by {
  -- informal proof outline: recursive calls on 3, 4, 5, and final base case on 6
  sorry
}

end problem_f_3_l66_66918


namespace area_of_triangle_ABC_l66_66914

/-- 
Given a triangle ABC with point M inside it. Perpendiculars 
from M to the sides BC, AC, and AB have lengths k, l, and m respectively.
Also given the angles \(\angle CAB = \alpha\) and \(\angle ABC = \beta\),
the following definitions hold.
-/
variables {A B C M : Type}
variables (k l m : ℝ)
variables (α β : ℝ)

def area_triangle (A B C : Type) : ℝ := 
  let γ := Real.arcsin ((Real.sin (π - α - β))),
      k := 3, l := 2, m := 4 in
  (3 * (Real.sin α) + 2 * (Real.sin β) + 4 * (Real.sin γ))^2 / (2 * (Real.sin α) * (Real.sin β) * (Real.sin γ))

-- Given the specific values
def α : ℝ := π / 6
def β : ℝ := π / 4

-- Expected area
def expected_area : ℝ := 67

-- Prove the area of triangle ABC to be approximately 67
theorem area_of_triangle_ABC : 
  (area_triangle α β) ≈ expected_area := 
by sorry

end area_of_triangle_ABC_l66_66914


namespace correct_system_of_equations_l66_66802

theorem correct_system_of_equations (x y : ℕ) :
  (8 * x - 3 = y ∧ 7 * x + 4 = y) ↔ 
  (8 * x - 3 = y ∧ 7 * x + 4 = y) := 
by 
  sorry

end correct_system_of_equations_l66_66802


namespace annual_growth_rate_equation_l66_66433

theorem annual_growth_rate_equation
  (initial_capital : ℝ)
  (final_capital : ℝ)
  (n : ℕ)
  (x : ℝ)
  (h1 : initial_capital = 10)
  (h2 : final_capital = 14.4)
  (h3 : n = 2) :
  1000 * (1 + x)^2 = 1440 :=
by
  sorry

end annual_growth_rate_equation_l66_66433


namespace tan_alpha_over_tan_beta_l66_66934

theorem tan_alpha_over_tan_beta (α β : ℝ) (h1 : Real.sin (α + β) = 2 / 3) (h2 : Real.sin (α - β) = 1 / 3) :
  (Real.tan α / Real.tan β = 3) :=
sorry

end tan_alpha_over_tan_beta_l66_66934


namespace at_least_one_not_land_equiv_l66_66632

/-- Propositions of trainees landing within the designated area. -/
variables (p q : Prop)

/-- Trainee A lands within the designated area. -/
def traineeALands : Prop := p

/-- Trainee B lands within the designated area. -/
def traineeBLands : Prop := q

/-- At least one trainee does not land within the designated area. -/
def atLeastOneNotLand : Prop := ¬p ∨ ¬q

theorem at_least_one_not_land_equiv (hp : traineeALands p) (hq : traineeBLands q) :
  atLeastOneNotLand p q = (¬p ∨ ¬q) :=
sorry

end at_least_one_not_land_equiv_l66_66632


namespace proposition_C_parallel_to_plane_l66_66852

-- Definitions for the problem
def line (α : Type) := α
def plane (α : Type) := α

variables {α : Type}
variables (m : line α) (α_plane : plane α)

def is_parallel_to_plane (m : line α) (α_plane : plane α) : Prop :=
∀ l : line α, l ∈ α_plane → m ∥ l

-- Propositions
def proposition_A (m : line α) (α_plane : plane α) : Prop :=
∀ l : line α, l ∈ α_plane → m ∥ l

def proposition_B (m : line α) (α_plane : plane α) : Prop :=
∃ S : set (line α), (infinite S ∧ ∀ l ∈ S, l ∈ α_plane ∧ m ∥ l)

def proposition_C (m : line α) (α_plane : plane α) : Prop :=
¬ ∃ p : α, p ∈ m ∧ p ∈ α_plane 

def proposition_D (m : line α) (α_plane : plane α) : Prop :=
∃ l : line α, l ∈ α_plane ∧ m ∥ l

-- The theorem we need to prove
theorem proposition_C_parallel_to_plane :
  proposition_C m α_plane → is_parallel_to_plane m α_plane :=
sorry

end proposition_C_parallel_to_plane_l66_66852


namespace parabola_intercepts_sum_l66_66734

theorem parabola_intercepts_sum :
  let f := λ y : ℝ, y^2 - 4 * y + 4,
      a := f 0,
      ys := [2, 2] in
  a + ys.sum = 8 := by
sorry

end parabola_intercepts_sum_l66_66734


namespace digits_of_2_pow_100_last_three_digits_of_2_pow_100_l66_66604

-- Prove that 2^100 has 31 digits.
theorem digits_of_2_pow_100 : (10^30 ≤ 2^100) ∧ (2^100 < 10^31) :=
by
  sorry

-- Prove that the last three digits of 2^100 are 376.
theorem last_three_digits_of_2_pow_100 : 2^100 % 1000 = 376 :=
by
  sorry

end digits_of_2_pow_100_last_three_digits_of_2_pow_100_l66_66604


namespace combined_score_210_l66_66623

-- Define the constants and variables
def total_questions : ℕ := 50
def marks_per_question : ℕ := 2
def jose_wrong_questions : ℕ := 5
def jose_extra_marks (alisson_score : ℕ) : ℕ := 40
def meghan_less_marks (jose_score : ℕ) : ℕ := 20

-- Define the total possible marks
def total_possible_marks : ℕ := total_questions * marks_per_question

-- Given the conditions, we need to prove the total combined score is 210
theorem combined_score_210 : 
  ∃ (jose_score meghan_score alisson_score combined_score : ℕ), 
  jose_score = total_possible_marks - (jose_wrong_questions * marks_per_question) ∧
  meghan_score = jose_score - meghan_less_marks jose_score ∧
  alisson_score = jose_score - jose_extra_marks alisson_score ∧
  combined_score = jose_score + meghan_score + alisson_score ∧
  combined_score = 210 := by
  sorry

end combined_score_210_l66_66623


namespace fraction_of_sand_is_one_third_l66_66457

-- Definitions based on the problem conditions
def weight_total : ℝ := 48
def weight_water : ℝ := 1/2 * weight_total
def weight_gravel : ℝ := 8
def weight_sand : ℝ := weight_total - weight_water - weight_gravel

-- Assertion of the correct answer
theorem fraction_of_sand_is_one_third :
  (weight_sand / weight_total) = 1/3 :=
by
  sorry

end fraction_of_sand_is_one_third_l66_66457


namespace sample_size_calculation_l66_66823

theorem sample_size_calculation :
  let total_A := 120
  let total_B := 80
  let total_C := 60
  let C_sample := 3
  let total_production := total_A + total_B + total_C
  let proportion_C := (total_C : ℚ) / total_production
  ∃ n : ℚ, proportion_C = C_sample / n ∧ n = 13 :=
begin
  sorry,
end

end sample_size_calculation_l66_66823


namespace infinite_even_and_odd_numbers_in_fn_l66_66233

theorem infinite_even_and_odd_numbers_in_fn 
  (f : ℕ → ℕ) 
  (h_def : ∀ n, f n = (2^n * real.sqrt 69).floor + (2^n * real.sqrt 96).floor) :
  (∀ k : ℕ, ∃ n : ℕ, (k = 2 * (f n / 2)) ∨ (k = 2 * (f n / 2) + 1)) :=
sorry

end infinite_even_and_odd_numbers_in_fn_l66_66233


namespace gift_bags_needed_l66_66497

/-
  Constants
  total_expected: \(\mathbb{N}\) := 90        -- 50 people who will show up + 40 more who may show up
  total_prepared: \(\mathbb{N}\) := 30        -- 10 extravagant gift bags + 20 average gift bags

  The property to be proved:
  prove that (total_expected - total_prepared = 60)
-/

def total_expected : ℕ := 50 + 40
def total_prepared : ℕ := 10 + 20
def additional_needed := total_expected - total_prepared

theorem gift_bags_needed : additional_needed = 60 := by
  sorry

end gift_bags_needed_l66_66497


namespace star_polygon_interior_angle_sum_l66_66794

-- Definitions: n is an integer greater than or equal to 5
variable (n : ℕ) (h : n ≥ 5)

-- Conditions: a convex polygon with sides 1 to n; for k in 1 to n, side k is not parallel to k+2 and they intersect
noncomputable def sum_interior_angles_star_polygon (n : ℕ) [fact (5 ≤ n)] : ℝ :=
  180 * (n - 4)

-- Proof problem: prove S = 180 * (n - 4)
theorem star_polygon_interior_angle_sum (h1 : n ≥ 5) : sum_interior_angles_star_polygon n = 180 * (n - 4) := 
sorry

end star_polygon_interior_angle_sum_l66_66794


namespace circumference_of_circle_l66_66861

theorem circumference_of_circle (R : ℝ) : 
  (C = 2 * Real.pi * R) :=
sorry

end circumference_of_circle_l66_66861


namespace statement_1_statement_4_l66_66355

variables {α β : Type*} [Plane α] [Plane β] {m n : Line}

-- The conditions are that α and β are different planes, and m and n are different lines.

theorem statement_1 (h1 : α ∥ β) (h2 : m ⊆ α) : m ∥ β :=
sorry

theorem statement_4 (h1 : n ⊥ α) (h2 : n ⊥ β) (h3 : m ⊥ α) : m ⊥ β :=
sorry

end statement_1_statement_4_l66_66355


namespace cubic_roots_c_div_d_l66_66018

theorem cubic_roots_c_div_d (a b c d : ℚ) :
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 1/2 ∨ x = 4) →
  (c / d = 9 / 4) :=
by
  intros h
  -- Proof would go here
  sorry

end cubic_roots_c_div_d_l66_66018


namespace sumOddDivisorsOf90_l66_66051

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l66_66051


namespace James_received_more_apples_than_Jane_l66_66693

theorem James_received_more_apples_than_Jane :
  ∀ (initial_apples : ℕ) (apples_given_to_Jane : ℕ) (apples_to_give_away : ℕ),
    initial_apples = 20 →
    apples_given_to_Jane = 5 →
    apples_to_give_away = 16 - apples_given_to_Jane - 4 →
    (apples_to_give_away - apples_given_to_Jane) = 2 :=
by
  intros initial_apples apples_given_to_Jane apples_to_give_away
  assume h1 h2 h3
  rw [h1, h2]
  sorry

end James_received_more_apples_than_Jane_l66_66693


namespace sum_of_positive_odd_divisors_of_90_l66_66079

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l66_66079


namespace angle_AEB_eq_angle_BEO_l66_66353

variable (A B C D O E : Type)
variable (ABCD : is_square A B C D)
variable (angle_EAB_90 : ∠ E A B = 90)
variable (diagonals_intersect_O : intersect_at_midpoint_diagonals A B C D O)

theorem angle_AEB_eq_angle_BEO : ∠ A E B = ∠ B E O := by
  sorry

end angle_AEB_eq_angle_BEO_l66_66353


namespace cos_B_of_sine_ratios_proof_l66_66659

noncomputable def cos_B_of_sine_ratios (A B C : ℝ) (a b c : ℝ) 
  (h1 : sin A / sin B = 4 / 3)
  (h2 : sin B / sin C = 3 / 2)
  (h3 : a / sin A = b / sin B)
  (h4 : b / sin B = c / sin C)
  (ha : a = 4)
  (hb : b = 3)
  (hc : c = 2) : ℝ :=
(cos B) = (11 / 16)

theorem cos_B_of_sine_ratios_proof :
  ∀ (A B C a b c : ℝ),
  sin A / sin B = 4 / 3 → 
  sin B / sin C = 3 / 2 → 
  a / sin A = b / sin B → 
  b / sin B = c / sin C → 
  a = 4 → 
  b = 3 → 
  c = 2 → 
  cos B = 11 / 16 :=
by
  intros A B C a b c h1 h2 h3 h4 ha hb hc
  sorry

end cos_B_of_sine_ratios_proof_l66_66659


namespace simplify_expressions_1_simplify_expressions_2_l66_66380

-- Define our first proof problem as a theorem in Lean 4.
theorem simplify_expressions_1 : log 3 27 + log 10 (1 / 100) + log (Real.exp 1) (sqrt (Real.exp 1)) + 2 ^ (-1 + log 2 3) = 3 :=
sorry

-- Define our second proof problem as a theorem in Lean 4.
theorem simplify_expressions_2 : (-(1 / 27) ^ (-1 / 3)) + (log 3 16 * log 2 (1 / 9)) = -11 :=
sorry

end simplify_expressions_1_simplify_expressions_2_l66_66380


namespace no_nat_n_for_9_pow_n_minus_7_is_product_l66_66202

theorem no_nat_n_for_9_pow_n_minus_7_is_product :
  ¬ ∃ (n k : ℕ), 9 ^ n - 7 = k * (k + 1) :=
by
  sorry

end no_nat_n_for_9_pow_n_minus_7_is_product_l66_66202


namespace pairings_count_l66_66500

-- Define the problem's conditions explicitly
def number_of_bowls : Nat := 6
def number_of_glasses : Nat := 6

-- The theorem stating that the number of pairings is 36
theorem pairings_count : number_of_bowls * number_of_glasses = 36 := by
  sorry

end pairings_count_l66_66500


namespace wharf_length_l66_66335

-- Define the constants
def avg_speed := 2 -- average speed in m/s
def travel_time := 16 -- travel time in seconds

-- Define the formula to calculate length of the wharf
def length_of_wharf := 2 * avg_speed * travel_time

-- The goal is to prove that length_of_wharf equals 64
theorem wharf_length : length_of_wharf = 64 :=
by
  -- Proof would be here
  sorry

end wharf_length_l66_66335


namespace fraction_of_number_l66_66033

-- Given definitions based on the problem conditions
def fraction : ℚ := 3 / 4
def number : ℕ := 40

-- Theorem statement to be proved
theorem fraction_of_number : fraction * number = 30 :=
by
  sorry -- This indicates that the proof is not yet provided

end fraction_of_number_l66_66033


namespace arrange_in_circle_sum_leq_l66_66014

noncomputable def numbers : Type := ℝ
variables (a b c d e : numbers)

-- Given condition: sum of the five numbers equals 1
def sum_eq_one (a b c d e : numbers) : Prop :=
  a + b + c + d + e = 1 ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 

-- The theorem to be proven
theorem arrange_in_circle_sum_leq (a b c d e : numbers) :
  sum_eq_one a b c d e → 
  ∃ (a1 a2 a3 a4 a5 : numbers), 
  a1 + a2 + a3 + a4 + a5 = 1 ∧ 
  (a1 a2 + a2 a3 + a3 a4 + a4 a5 + a5 a1) ≤ 1 / 5 := sorry

end arrange_in_circle_sum_leq_l66_66014


namespace distinct_numbers_sum_div_by_3_l66_66327

theorem distinct_numbers_sum_div_by_3 : 
  (∃ (S : finset ℕ), S.card = 3 ∧
   ∀ n ∈ S, n < 20 ∧ n > 0 ∧ S.sum % 3 = 0) →
  ∃! S, S.card = 3 ∧ (∀ a, a ∈ S → a < 20 ∧ a > 0) ∧ S.sum % 3 = 0 ∧ 
  finset.card S = 327 :=
by
  sorry

end distinct_numbers_sum_div_by_3_l66_66327


namespace race_possibility_l66_66745

-- Define the types for runners and number of races
inductive Runner
| A | B | C
deriving DecidableEq, Repr

def races : List (List Runner) :=
  [[Runner.A, Runner.B, Runner.C],
   [Runner.B, Runner.C, Runner.A],
   [Runner.C, Runner.A, Runner.B]]

-- Define a function to count how many times one runner outruns another
def outruns (a b : Runner) (races : List (List Runner)) : Nat :=
  races.countp (λ race => race.indexOf a < race.indexOf b)

-- Define the conditions
def outruns_more_than_half (a b : Runner) (races : List (List Runner)) : Prop :=
  outruns a b races > races.length / 2

theorem race_possibility :
  outruns_more_than_half Runner.A Runner.B races ∧
  outruns_more_than_half Runner.B Runner.C races ∧
  outruns_more_than_half Runner.C Runner.A races :=
by {
  sorry
}

end race_possibility_l66_66745


namespace B_empty_implies_m_lt_2_no_common_elements_implies_m_ranges_l66_66689

-- Define the sets A and B
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2m - 1 }

-- Proof problem for Question 1
theorem B_empty_implies_m_lt_2 (m : ℝ) : B m = ∅ → m < 2 :=
by sorry

-- Proof problem for Question 2
theorem no_common_elements_implies_m_ranges (m : ℝ) : (∀ x, ¬ (x ∈ A ∧ x ∈ B m)) → (m > 4 ∨ m < 2) :=
by sorry

end B_empty_implies_m_lt_2_no_common_elements_implies_m_ranges_l66_66689


namespace sum_of_odd_divisors_of_90_l66_66083

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l66_66083


namespace min_value_y_max_value_f_l66_66123

-- Problem 1
theorem min_value_y (x : ℝ) (h : x > 1 / 2) : 
  ∃ y_min, y_min = 2x + 4 / (2x - 1) ∧ y_min = 5 :=
by
  sorry

-- Problem 2
theorem max_value_f (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 3) :
  ∃ f_max, f_max = x * y + 2 * x + y ∧ f_max = 7 :=
by
  sorry

end min_value_y_max_value_f_l66_66123


namespace triangle_area_eq_five_l66_66646

variables (PQ RS S Q : ℝ) -- defining the lengths of sides
variables (height : ℝ)   -- defining the height of the trapezoid

-- Condition: PQRS is a trapezoid with an area of 20
def area_trapezoid (PQ RS height : ℝ) : ℝ :=
  ((PQ + RS) / 2) * height

-- Condition: RS is three times the length of PQ
axiom length_relation : RS = 3 * PQ

-- Given that PQRS has an area of 20
axiom trapezoid_area : area_trapezoid PQ RS height = 20

-- Theorem: The area of triangle PQS is 5
theorem triangle_area_eq_five : (PQ * height) / 2 = 5 :=
sorry

end triangle_area_eq_five_l66_66646


namespace maximum_number_of_red_balls_is_125_minimum_number_of_red_balls_is_43_l66_66743

def is_red (n : ℕ) : Prop :=
  sorry

def is_green (n : ℕ) : Prop :=
  sorry

def is_blue (n : ℕ) : Prop :=
  sorry

def balls : Fin 127 → Prop :=
  sorry

-- Conditions

axiom at_least_one_red : ∃ n, is_red n
axiom at_least_one_green : ∃ n, is_green n
axiom at_least_one_blue : ∃ n, is_blue n
axiom left_of_blue_has_red : ∀ n, is_blue n → ∃ m, m < n ∧ is_red m
axiom right_of_green_has_red : ∀ n, is_green n → ∃ m, n < m ∧ is_red m

-- Questions

theorem maximum_number_of_red_balls_is_125 :
  (∑ n in (Finset.filter is_red (Finset.range 127)), 1) = 125 :=
sorry

theorem minimum_number_of_red_balls_is_43 :
  (∑ n in (Finset.filter is_red (Finset.range 127)), 1) = 43 :=
sorry

end maximum_number_of_red_balls_is_125_minimum_number_of_red_balls_is_43_l66_66743


namespace fraction_of_40_l66_66037

theorem fraction_of_40 : (3 / 4) * 40 = 30 :=
by
  -- We'll add the 'sorry' here to indicate that this is the proof part which is not required.
  sorry

end fraction_of_40_l66_66037


namespace sum_odd_divisors_90_eq_78_l66_66059

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l66_66059


namespace problem1_1_problem1_2_problem2_l66_66124

open Set

/-
Given sets U, A, and B, derived from the provided conditions:
  U : Set ℝ
  A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
  B (m : ℝ) : Set ℝ := {x | x < 2 * m - 3}
-/

def U : Set ℝ := univ
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | x < 2 * m - 3}

theorem problem1_1 (m : ℝ) (h : m = 5) : A ∩ B m = {x | -3 ≤ x ∧ x ≤ 5} :=
sorry

theorem problem1_2 (m : ℝ) (h : m = 5) : (compl A) ∪ B m = univ :=
sorry

theorem problem2 (m : ℝ) : A ⊆ B m → 4 < m :=
sorry

end problem1_1_problem1_2_problem2_l66_66124


namespace monomial_sum_l66_66996

theorem monomial_sum (m k : ℕ) (h1 : 3 = m) (h2 : k = 2) : m + k = 5 := by
  rw [h1, h2]
  exact Nat.add_eq_add_left 5

end monomial_sum_l66_66996


namespace matrix_multiplication_correct_l66_66195

def matrixA : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, 0, -3],
    ![1, 3, -1],
    ![0, 5, 2]]

def matrixB : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, -1, 4],
    ![-2, 0, 0],
    ![3, 0, -2]]

def matrixProduct : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![-7, -2, 14],
    ![-8, -1, 6],
    ![-4, 0, -4]]

theorem matrix_multiplication_correct :
  Matrix.mul matrixA matrixB = matrixProduct := by
  sorry

end matrix_multiplication_correct_l66_66195


namespace interest_rate_increase_l66_66162

theorem interest_rate_increase (P : ℝ) (A1 A2 : ℝ) (T : ℝ) (R1 R2 : ℝ) (percentage_increase : ℝ) :
  P = 500 → A1 = 600 → A2 = 700 → T = 2 → 
  (A1 - P) = P * R1 * T →
  (A2 - P) = P * R2 * T →
  percentage_increase = (R2 - R1) / R1 * 100 →
  percentage_increase = 100 :=
by sorry

end interest_rate_increase_l66_66162


namespace rectangular_solid_surface_area_l66_66214

theorem rectangular_solid_surface_area
  (a b c : ℕ)
  (h_prime_a : Prime a)
  (h_prime_b : Prime b)
  (h_prime_c : Prime c)
  (h_volume : a * b * c = 143) :
  2 * (a * b + b * c + c * a) = 382 := by
  sorry

end rectangular_solid_surface_area_l66_66214


namespace depth_of_well_l66_66159

theorem depth_of_well (d : ℝ) (t1 t2 : ℝ)
  (h1 : d = 15 * t1^2)
  (h2 : t2 = d / 1100)
  (h3 : t1 + t2 = 9.5) :
  d = 870.25 := 
sorry

end depth_of_well_l66_66159


namespace cos_double_angle_l66_66553

open Real

theorem cos_double_angle :
  (∀ θ : ℝ, sin (θ - π / 6) = sqrt 3 / 3 → cos (π / 3 - 2 * θ) = 1 / 3) :=
by
  assume θ h
  sorry

end cos_double_angle_l66_66553


namespace tangent_circle_intersection_l66_66870

theorem tangent_circle_intersection
  {O A B T P M : Type}
  [metric_space O]
  [metric_space A]
  [metric_space B]
  [metric_space T]
  [metric_space P]
  [metric_space M]
  (radius_Ω : ℝ)
  (radius_ω : ℝ)
  (rΩ_5 : radius_Ω = 5)
  (rω_1 : radius_ω = 1)
  (on_Ω : ∀ (A B : Type), A ∈ sphere O radius_Ω ∧ B ∈ sphere O radius_Ω)
  (chord_AB_len : ∀ (A B : Type), distance A B = 6)
  (tangent_point_T : ∀ (T : Type), is_tangent T (chord A B))
  (internal_tangent : ∀ (P : Type), is_internally_tangent (circle P radius_ω) (circle O radius_Ω)) :
  distance A T * distance B T = 2 :=
sorry

end tangent_circle_intersection_l66_66870


namespace find_x_values_l66_66569

-- Define the conditions given in the problem
def condition1 (x : ℤ) := |x^2 - x| < 6
def condition2 (x : ℤ) := x ∈ ℤ
def condition3 (x : ℤ) := (¬ (|x^2 - x| ≥ 6 ∧ x ∈ ℤ) ∧ ¬ (x ∈ ℤ))

noncomputable def solution_set := {-1, 0, 1, 2}

theorem find_x_values : 
  { x : ℤ | condition1 x } = solution_set ∧
  { x : ℤ | condition3 x } = ∅ :=
sorry

end find_x_values_l66_66569


namespace area_of_triangle_KBC_l66_66320

-- Defining the conditions 
variables (ABCDEF : Hexagon)
variables (ABJI FEHG : Square)
variables (JBK : Triangle)
variables (KBC: Triangle)
variables (FE BC : ℝ)

-- conditions
def equiangular_hexagon : Prop := ABCDEF.equiangular
def ABJI_square_area : Prop := ABJI.area = 25
def FEHG_square_area : Prop := FEHG.area = 49
def JBK_isosceles : Prop := JBK.isosceles ∧ JBK.JB = JBK.BK
def FE_eq_BC : Prop := FE = BC

-- Problem statement
theorem area_of_triangle_KBC
  (h1 : equiangular_hexagon)
  (h2 : ABJI_square_area)
  (h3 : FEHG_square_area)
  (h4 : JBK_isosceles)
  (h5 : FE_eq_BC) : 
  KBC.area = 17.5 :=
sorry

end area_of_triangle_KBC_l66_66320


namespace smallest_z_square_l66_66812

theorem smallest_z_square (P U M A C : ℝ) (hroots : ∃ (r1 r2 r3 r4 : ℝ), 
  (r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0 ∧ 
  (P * r1^4 + U * r1^3 + M * r1^2 + A * r1 + C = 0) ∧
  (P * r2^4 + U * r2^3 + M * r2^2 + A * r2 + C = 0) ∧
  (P * r3^4 + U * r3^3 + M * r3^2 + A * r3 + C = 0) ∧
  (P * r4^4 + U * r4^3 + M * r4^2 + A * r4 + C = 0))) (hP_ne_zero : P ≠ 0) :
  ∃ z : ℝ, (M^2 - 2 * U * A + z * P * C > 0) ∧ z^2 = 16 := 
begin
  sorry
end

end smallest_z_square_l66_66812


namespace tangent_line_equation_l66_66264

open Real

noncomputable def common_tangent_line : Prop :=
∃ (l : ℝ → ℝ), 
  (∀ x, l x = ln x → l x = (1 / x) * x - 1 + ln x) ∧ 
  (∀ x < 0, l x = -3 - 1/x → l x = (1 / x^2) * x - 2/x - 3) ∧
  (l = (λ x, x - 1)) -- The function representing the tangent line

theorem tangent_line_equation :
  common_tangent_line :=
sorry

end tangent_line_equation_l66_66264


namespace four_cells_diff_colors_l66_66721

theorem four_cells_diff_colors 
    (colors : Fin 4 → Fin 100 → Fin 100 → Fin 4)
    (∀i, ∀j, ∃! c, colors c i j)
    (∀i, ∀ c, ∃! r, colors c i r)
    (∀j, ∀ c, ∃! c, colors c j i) :
    ∃ (i1 i2 j1 j2 : Fin 100),
      i1 ≠ i2 ∧
      j1 ≠ j2 ∧
      colors i1 j1 ≠ colors i1 j2 ∧
      colors i1 j1 ≠ colors i2 j1 ∧
      colors i2 j1 ≠ colors i2 j2 ∧
      colors i1 j2 ≠ colors i2 j2  :=
sorry

end four_cells_diff_colors_l66_66721


namespace limit_ln_eq_half_ln_three_l66_66188

theorem limit_ln_eq_half_ln_three :
  (Real.lim (λ x : ℝ, ln ((exp (x^2) - cos x) * cos (1 / x) + tan (x + Real.pi / 3))) 0) = 1 / 2 * ln 3 :=
sorry

end limit_ln_eq_half_ln_three_l66_66188


namespace least_constant_K_l66_66801

def sequence_property (c : ℝ) (x : ℕ → ℝ) (n : ℕ) (M : ℝ) :=
  0 < c ∧ c < 1 ∧ x 0 = 1 ∧ x n ≥ M ∧ (∀ i < n, x i < x (i + 1))

def sum_property (c : ℝ) (x : ℕ → ℝ) (n : ℕ) (K : ℝ) :=
  ∑ i in Finset.range n, (x (i + 1) - x i) ^ c / x i ^ (c + 1) ≤ K

theorem least_constant_K (c : ℝ) (M : ℝ) :
  (0 < c ∧ c < 1) → (1 < M) →
  ∃ (K : ℝ), K = c⁻¹ (1 - c)⁻¹ (1 - c) ^ -(1 - c) ∧
    (∀ n : ℕ, ∃ x : ℕ → ℝ, sequence_property c x n M ∧ sum_property c x n K) :=
by
  sorry

end least_constant_K_l66_66801


namespace probability_of_sum_being_perfect_square_l66_66025

structure Dice :=
  (sides : Finset ℕ)
  (roll : ℕ → ℕ)

def sum_rolls (d1 d2 : Dice) : Finset ℕ :=
  {s ∈ (d1.sides.product d2.sides) | (λ (pair : ℕ × ℕ), pair.1 + pair.2) s}

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

def probability_of_perfect_square_sum (d1 d2 : Dice) : ℚ :=
  let total_possible_outcomes := (d1.sides.card * d2.sides.card : ℕ) in
  let perfect_square_sums := sum_rolls d1 d2 |>.filter is_perfect_square in
  ⟨perfect_square_sums.card, total_possible_outcomes⟩
 
theorem probability_of_sum_being_perfect_square :
  let d := Dice.mk (Finset.range 10) id in
  probability_of_perfect_square_sum d d = 7 / 50 :=
by
  sorry

end probability_of_sum_being_perfect_square_l66_66025


namespace solution_set_of_inequality_l66_66741

theorem solution_set_of_inequality :
  { x : ℝ | (x - 1) * (x + 1) * (x - 2) < 0 } = set.Ioo ℝ { -∞, -1 } ∪ set.Ioo ℝ {1, 2} :=
by
  sorry

end solution_set_of_inequality_l66_66741


namespace savings_after_one_year_l66_66792

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem savings_after_one_year :
  compound_interest 1000 0.10 2 1 = 1102.50 :=
by
  sorry

end savings_after_one_year_l66_66792


namespace find_ab_range_m_l66_66594

-- Part 1
theorem find_ab (a b: ℝ) (h1 : 3 - 6 * a + b = 0) (h2 : -1 + 3 * a - b + a^2 = 0) :
  a = 2 ∧ b = 9 := 
sorry

-- Part 2
theorem range_m (m: ℝ) (h: ∀ x ∈ (Set.Icc (-2) 1), x^3 + 3 * 2 * x^2 + 9 * x + 4 - m ≤ 0) :
  20 ≤ m :=
sorry

end find_ab_range_m_l66_66594


namespace possible_values_b2_l66_66478

theorem possible_values_b2 :
  ∃ (b2_set : Set ℕ),
    (∀ b2 ∈ b2_set, b2 < 1023 ∧ b2 % 2 = 1 ∧ Int.gcd 1023 b2 = 3)
    ∧ (b2_set.card = 507) :=
by
  sorry

end possible_values_b2_l66_66478


namespace sec_minus_tan_l66_66984

theorem sec_minus_tan {x : ℝ} (h : real.sec x + real.tan x = 7/3) : real.sec x - real.tan x = 3/7 :=
sorry

end sec_minus_tan_l66_66984


namespace median_and_mode_correct_l66_66164

-- Define the given speeds and corresponding number of vehicles
def speeds : List ℕ := [48, 49, 50, 51, 52]
def vehicle_counts : List ℕ := [5, 4, 8, 2, 1]

-- Given conditions
def total_vehicles : ℕ := vehicle_counts.sum -- 20

-- Function to compute median of provided data given the specific structure
def median : ℕ :=
  let values := List.replicate vehicle_counts.head speeds.head ++ 
                List.replicate (vehicle_counts.get 1) speeds.get 1 ++ 
                List.replicate (vehicle_counts.get 2) speeds.get 2 ++ 
                List.replicate (vehicle_counts.get 3) speeds.get 3 ++ 
                List.replicate (vehicle_counts.get 4) speeds.get 4
  let sorted_values := values.sort
  (sorted_values.get 9 + sorted_values.get 10) / 2 

-- Function to compute mode of provided data given the specific structure
def mode : ℕ :=
  let zipped := speeds.zip vehicle_counts
  let max_pair := zipped.maxBy (λ x => x.snd)
  max_pair.fst

-- Theorem stating the facts to prove
theorem median_and_mode_correct : median = 50 ∧ mode = 50 := by
  sorry

end median_and_mode_correct_l66_66164


namespace wronskian_exponential_l66_66226

noncomputable def Wronskian (f g h : ℝ → ℝ) : ℝ → ℝ :=
  λ x, matrix.det ![
    ![f x, g x, h x],
    ![(deriv f) x, (deriv g) x, (deriv h) x],
    ![(deriv (deriv f)) x, (deriv (deriv g)) x, (deriv (deriv h)) x]
  ]

variable (k1 k2 k3 : ℝ)

def y1 (x : ℝ) : ℝ := real.exp (k1 * x)
def y2 (x : ℝ) : ℝ := real.exp (k2 * x)
def y3 (x : ℝ) : ℝ := real.exp (k3 * x)

theorem wronskian_exponential (x : ℝ) :
  Wronskian (y1 k1) (y2 k2) (y3 k3) x =
    real.exp ((k1 + k2 + k3) * x) * ((k2 - k1) * (k3 - k1) * (k3 - k2)) :=
sorry

end wronskian_exponential_l66_66226


namespace at_least_one_boy_one_girl_l66_66493

theorem at_least_one_boy_one_girl :
  (∀ p : ℝ, (0 < p ∧ p < 1) → (4.choose 0 * p^0 * (1-p)^4 + 4.choose 4 * p^4 * (1-p)^0) = (1/8) →
  (4.choose 1 * p^1 * (1-p)^3 + 4.choose 2 * p^2 * (1-p)^2 + 4.choose 3 * p^3 * (1-p)^1) = 7/8) :=
begin
  intro p,
  assume h, -- assumption of equal probability condition
  have h_boy: (4.choose 0 * p^0 * (1-p)^4 + 4.choose 4 * p^4 * (1-p)^0) = 1/16,
  {
    -- detail the probability that all four children are boys or all four children are girls
    sorry
  },
  have h_combined : 1 - (1/16 + 1/16) = 7/8,
  {
    -- the combined probability to get at least one boy and one girl
    sorry
  },
  exact h_combined,
end

end at_least_one_boy_one_girl_l66_66493


namespace sum_odd_divisors_90_eq_78_l66_66055

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l66_66055


namespace second_number_is_three_l66_66015

theorem second_number_is_three (x y : ℝ) (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) : y = 3 :=
by
  -- To be proved: sorry for now
  sorry

end second_number_is_three_l66_66015


namespace cos_double_angle_point_M_l66_66581

def point_M := (-3, 4)
def θ_initial_side_nonnegative_half_axis := true

theorem cos_double_angle_point_M :
  let x := -3
  let y := 4
  let r := Real.sqrt (x^2 + y^2)
  r = 5 →
  cos (2 * θ) = -7 / 25 :=
  begin
    intros,
    sorry
  end

end cos_double_angle_point_M_l66_66581


namespace expand_product_l66_66525

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by 
  sorry

end expand_product_l66_66525


namespace tangent_line_ellipse_l66_66587

theorem tangent_line_ellipse (x y : ℝ) (h : 2^2 / 8 + 1^2 / 2 = 1) :
    x / 4 + y / 2 = 1 := 
  sorry

end tangent_line_ellipse_l66_66587


namespace number_of_valid_three_digit_numbers_l66_66750

def three_digit_numbers_count : Nat :=
  let count_numbers (last_digit : Nat) (remaining_digits : List Nat) : Nat :=
    remaining_digits.length * (remaining_digits.erase last_digit).length

  let count_when_last_digit_is_0 :=
    count_numbers 0 [1, 2, 3, 4, 5, 6, 7, 8, 9]

  let count_when_last_digit_is_5 :=
    count_numbers 5 [0, 1, 2, 3, 4, 6, 7, 8, 9]

  count_when_last_digit_is_0 + count_when_last_digit_is_5

theorem number_of_valid_three_digit_numbers : three_digit_numbers_count = 136 := by
  sorry

end number_of_valid_three_digit_numbers_l66_66750


namespace sequence_identity_l66_66683

def a : ℕ → ℝ
| 1     := 3
| (n+2) := (a (n+1))^2 - 2

def f : ℕ → ℕ
| 1     := 1
| 2     := 1
| (n+2) := f (n+1) + f n

theorem sequence_identity (n : ℕ) (hn : n ≥ 1) : 
  a (n+1) = (f (2^(n+1)) : ℝ) / (f 4 : ℝ) :=
sorry

end sequence_identity_l66_66683


namespace separation_inequality_l66_66147

-- Definitions for centers, radii and distances
def is_center (A B C : ℝ × ℝ) : Prop := true
def is_radius (rA rB rC : ℝ) : Prop := rA > 0 ∧ rB > 0 ∧ rC > 0
def distance (P Q : ℝ × ℝ) : ℝ := (sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2))

-- Separating line conditions
def separation_property (A B C : ℝ × ℝ) : Prop :=
  ∀ (L : ℝ × ℝ → ℝ), (∀ X Y, L X = 0 ∧ L Y = 0 → X ≠ Y) → 
  (L A * L B < 0 ∧ L A * L C > 0 ∧ L B * L C > 0) ∧
  (L B * L C < 0 ∧ L B * L A > 0 ∧ L C * L A > 0) ∧
  (L C * L A < 0 ∧ L C * L B > 0 ∧ L A * L B > 0)

-- Main theorem statement
theorem separation_inequality 
  (A B C : ℝ × ℝ) 
  (rA rB rC : ℝ)
  (h_center : is_center A B C)
  (h_radius : is_radius rA rB rC)
  (h_separation : separation_property A B C) : 
  distance A B + distance B C + distance C A ≤ 2 * sqrt 2 * (rA + rB + rC) :=
sorry

end separation_inequality_l66_66147


namespace volume_multiplication_factor_l66_66987

variables (r h : ℝ) (π : ℝ := Real.pi)

def original_volume : ℝ := π * r^2 * h
def new_height : ℝ := 3 * h
def new_radius : ℝ := 2.5 * r
def new_volume : ℝ := π * (new_radius r)^2 * (new_height h)

theorem volume_multiplication_factor :
  new_volume r h / original_volume r h = 18.75 :=
by
  sorry

end volume_multiplication_factor_l66_66987


namespace vectors_not_coplanar_l66_66491

noncomputable def vec_a : ℝ × ℝ × ℝ := (1, -1, 4)
noncomputable def vec_b : ℝ × ℝ × ℝ := (1, 0, 3)
noncomputable def vec_c : ℝ × ℝ × ℝ := (1, -3, 8)

def scalar_triple_product (a b c : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * (b.2 * c.3 - b.3 * c.2) -
  a.2 * (b.1 * c.3 - b.3 * c.1) +
  a.3 * (b.1 * c.2 - b.2 * c.1)

theorem vectors_not_coplanar : scalar_triple_product vec_a vec_b vec_c ≠ 0 :=
by {
  simp [vec_a, vec_b, vec_c, scalar_triple_product],
  norm_num,
  sorry
}

end vectors_not_coplanar_l66_66491


namespace _l66_66778

-- Definitions based on conditions
def shares_terminal_side (a b : ℤ) : Prop :=
  ∃ k : ℤ, a = b + 360 * k

-- Main theorem statement
example : ¬ shares_terminal_side (-750) 680 :=
by 
  intro h
  cases h with k hk
  sorry -- Proof would go here

end _l66_66778


namespace concurrent_gergonians_l66_66874

open EuclideanGeometry

variable {A B C D E V W X Y Z P : Point}

/--
Consider a convex pentagon \( A B C D E \) circumscribed about a circle.
The lines \( \overline{A V}, \overline{B W}, \overline{C X}, \overline{D Y}, \overline{E Z} \)
are gergonnians connecting each vertex \( A B C D E \) to its opposite tangency points with the circle.

This statement proves:
1. If four gergonnians are concurrent at a point, then all five are concurrent at that point.
2. If one set of three gergonnians is concurrent, then there is another set of three concurrent gergonnians.
-/
theorem concurrent_gergonians
  (circumscribed : CircumscribedPentagon A B C D E)
  (tangency : TangencyPoints A B C D E V W X Y Z)
  (gergonnian_cur : ∀φ, Gergonnian φ A B C D E V W X Y Z ↔ LinesConcurrent φ)
  :
  (∀ (f1 f2 f3 f4 : Gergonnian A B C D E V W X Y Z) (P : Point), four_concurrent f1 f2 f3 f4 P → all_concurrent f1 f2 f3 f4 (f5 P)) ∧
  (∃ (f1 f2 f3 : Gergonnian A B C D E V W X Y Z) (P : Point), three_concurrent f1 f2 f3 P → ∃ (g1 g2 g3 : Gergonnian A B C D E V W X Y Z), three_concurrent g1 g2 g3 P) :=
by
  sorry

end concurrent_gergonians_l66_66874


namespace find_AC_l66_66309

variable (a b c d : ℝ)

def inscribed_quadrilateral (a b c d : ℝ) (ABCD : Type) : Prop :=
∃ A B C D K : ABCD, 
A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
-- Quadrilateral is inscribed in a circle
inscribed_quadrilateral ABCD A B C D ∧
-- Diagonals AC and BD intersect at point K
diagonals_intersect ABCD A B C D K ∧
-- Given segment lengths
segment_length ABCD A B = a ∧ 
segment_length ABCD B K = b ∧ 
segment_length ABCD A K = c ∧ 
segment_length ABCD C D = d

theorem find_AC (a b c d : ℝ) (ABCD : Type) (h_quad : inscribed_quadrilateral a b c d ABCD) : 
  let AC := (c + d * b / a) in
  AC = ( (a * c + b * d) / a ) := 
by
  sorry

end find_AC_l66_66309


namespace coeff_x4_expansion_l66_66767

theorem coeff_x4_expansion : coeff (expand (x - 3 * real.sqrt 2) 8) 4 = 22680 := by
  sorry

end coeff_x4_expansion_l66_66767


namespace angle_adb_is_120_l66_66122

-- Given conditions
variables {A B C D : Type} 
variables (ABC : Triangle A B C) 
variables [RightTriangle ABC C] [Angle_AT_30 ABC A] 
variables (BD_angle_bisector_of_B : AngleBisector BD (Angle ABC))

-- Goal
theorem angle_adb_is_120 (hABC : Triangle ABC)
                        (hRight : RightTriangle ABC C)
                        (hAngle30 : Angle_AT_30 ABC A)
                        (hBD : AngleBisector BD (Angle ABC)) :
  Angle A D B = 120 := 
sorry

end angle_adb_is_120_l66_66122


namespace alternating_multiple_exists_l66_66425

def alternating (m : ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → j < (nat.digits 10 m).length → (nat.digits 10 m).nth i ≠ (nat.digits 10 m).nth j

theorem alternating_multiple_exists (n : ℕ) (hn : 0 < n) :
  (∃ m : ℕ, m % n = 0 ∧ alternating m) → 20 ∣ n → False :=
by
  sorry

end alternating_multiple_exists_l66_66425


namespace price_of_sweater_l66_66479

theorem price_of_sweater
  (price_Tshirt : ℝ := 8) 
  (price_Jacket : ℝ := 80)
  (discount_Jacket : ℝ := 0.10)
  (tax_rate : ℝ := 0.05)
  (num_Tshirts : ℝ := 6)
  (num_Jackets : ℝ := 5)
  (num_Sweaters : ℝ := 4)
  (total_paid : ℝ := 504)
  : 
  S : ℝ :=
  let cost_no_tax := num_Tshirts * price_Tshirt + num_Jackets * price_Jacket * (1 - discount_Jacket) + num_Sweaters * S in
  (cost_no_tax * (1 + tax_rate) = total_paid) → 
  S = 18 :=
by
  sorry

end price_of_sweater_l66_66479


namespace mean_median_mode_equal_l66_66735

theorem mean_median_mode_equal (x : ℕ) (h1 : 2 ≤ x) (h2 : x ≤ 6) :
  let lst := [2, 3, 4, 4, 5, 6, x] in
  let mean := (2 + 3 + 4 + 4 + 5 + 6 + x) / 7 in
  let median := lst.length / 2 in
  let mode := 4 in
  mean = mode ∧ median = mode → x = 4 :=
by
  sorry

end mean_median_mode_equal_l66_66735


namespace total_peaches_in_baskets_l66_66519

def total_peaches (red_peaches : ℕ) (green_peaches : ℕ) (baskets : ℕ) : ℕ :=
  (red_peaches + green_peaches) * baskets

theorem total_peaches_in_baskets :
  total_peaches 19 4 15 = 345 :=
by
  sorry

end total_peaches_in_baskets_l66_66519


namespace number_of_remainders_mod_210_l66_66173

theorem number_of_remainders_mod_210 (p : ℕ) (hp : Nat.Prime p) (hp_gt_7 : p > 7) :
  ∃ (remainders : Finset ℕ), (remainders.card = 12) ∧ ∀ r ∈ remainders, ∃ k, p^2 ≡ r [MOD 210] :=
by
  sorry

end number_of_remainders_mod_210_l66_66173


namespace chord_length_intercepted_l66_66514

/-- Prove that the length of the chord intercepted by the circle x² + y² = 1 on the line x + y - 1 = 0 is √2 --/
theorem chord_length_intercepted
  (x y : ℝ)
  (h1 : x^2 + y^2 = 1)
  (h2 : x + y - 1 = 0) :
  ∃ l : ℝ, l = sqrt 2 :=
sorry

end chord_length_intercepted_l66_66514


namespace alices_number_l66_66848

theorem alices_number :
  ∃ (m : ℕ), (180 ∣ m) ∧ (45 ∣ m) ∧ (1000 ≤ m) ∧ (m ≤ 3000) ∧
    (m = 1260 ∨ m = 1440 ∨ m = 1620 ∨ m = 1800 ∨ m = 1980 ∨
     m = 2160 ∨ m = 2340 ∨ m = 2520 ∨ m = 2700 ∨ m = 2880) :=
by
  sorry

end alices_number_l66_66848


namespace stair_calculation_l66_66167

def already_climbed : ℕ := 74
def left_to_climb : ℕ := 22
def total_stairs : ℕ := 96

theorem stair_calculation :
  already_climbed + left_to_climb = total_stairs :=
by {
  sorry
}

end stair_calculation_l66_66167


namespace trains_meet_time_l66_66749

def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

def total_distance (L₁ L₂ D : ℝ) : ℝ :=
  L₁ + L₂ + D

def relative_speed (S₁ S₂ : ℝ) : ℝ :=
  S₁ + S₂

def time_to_meet (L₁ L₂ D S₁ S₂ : ℝ) : ℝ :=
  total_distance L₁ L₂ D / relative_speed (kmph_to_mps S₁) (kmph_to_mps S₂)

-- The theorem to prove
theorem trains_meet_time :
  time_to_meet 100 200 100 54 72 ≈ 11.43 :=
begin
  sorry
end

end trains_meet_time_l66_66749


namespace sum_of_positive_odd_divisors_of_90_l66_66074

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l66_66074


namespace sum_odd_divisors_l66_66093

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l66_66093


namespace sum_of_areas_of_rectangles_l66_66708

theorem sum_of_areas_of_rectangles : 
  let lengths := [1, 4, 9, 16, 25, 36, 49]
  let width := 2
  let areas := lengths.map (λ l => width * l)
  (areas.sum = 280) :=
by 
  let lengths := [1, 4, 9, 16, 25, 36, 49]
  let width := 2
  let areas := lengths.map (λ l => width * l)
  suffices : (areas.sum = 280)
  from this
  sorry

end sum_of_areas_of_rectangles_l66_66708


namespace count_solutions_l66_66684

noncomputable theory

def g (x : ℝ) := 3 * Real.cos (Real.pi * x)

theorem count_solutions :
  ∃ n : ℕ, n = 115 ∧ (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ g (g (g x)) = g x → n = 115) := sorry

end count_solutions_l66_66684


namespace finite_sets_of_primes_l66_66699

-- Statement only
theorem finite_sets_of_primes (k : ℕ) (k_pos : k > 0) :
  ∃ (S : Finset (Finset ℕ)), ∀ T ∈ S, (∀ p ∈ T, Prime p) ∧ ∏ p in T, p ∣ ∏ p in T, p + k :=
  sorry

end finite_sets_of_primes_l66_66699


namespace sum_odd_divisors_l66_66099

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l66_66099


namespace sum_odd_divisors_l66_66098

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l66_66098


namespace volume_of_sheep_eq_volume_of_dog_l66_66993

variable (V : Type) [VolumeSpace V] -- hypothetical type and class representing volume-related structures
variable (clay : V) (shaped_into_sheep shaped_into_dog : V → V)

-- Assume the volume function is defined
variable [VolumeMeasure V] (volume : V → ℝ)

-- Definition: The same piece of clay is shaped into both forms
def same_piece_of_clay (shape1 shape2 : V) : Prop :=
  ∃ (clay : V), shape1 = shaped_into_sheep clay ∧ shape2 = shaped_into_dog clay

theorem volume_of_sheep_eq_volume_of_dog
  (hs : same_piece_of_clay (shaped_into_sheep clay) (shaped_into_dog clay)) :
  volume (shaped_into_sheep clay) = volume (shaped_into_dog clay) :=
by
  sorry

end volume_of_sheep_eq_volume_of_dog_l66_66993


namespace probability_palindrome_divisible_by_7_and_quotient_palindrome_l66_66469

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

noncomputable def num_7_digit_palindromes : ℕ := 9000

def valid_7_digit_palindrome (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧
  is_palindrome n

theorem probability_palindrome_divisible_by_7_and_quotient_palindrome :
  let valid_palindromes := { n : ℕ | valid_7_digit_palindrome n }
  let valid_div_by_7 := { n ∈ valid_palindromes | n % 7 = 0 ∧ is_palindrome (n / 7) }
  let k := valid_div_by_7.card
  (k : ℝ) / (num_7_digit_palindromes : ℝ) = 1 / 25 := 
sorry

end probability_palindrome_divisible_by_7_and_quotient_palindrome_l66_66469


namespace nearest_integer_sum_is_236_l66_66872

open BigOperators Real

noncomputable def nearestIntegerToSum : ℤ :=
  let seriesSum := 100 * ∑' n : ℕ, (3 ^ n) * (sin (π / (3 ^ n))) ^ 3
  let approximatedPi := 3.14159265
  let nearestInt := rounded_to_nearest_integer (seriesSum * (approximatedPi/ π))
  nearestInt

-- The main proof problem:
theorem nearest_integer_sum_is_236 : nearestIntegerToSum = 236 := by
  sorry

end nearest_integer_sum_is_236_l66_66872


namespace ratio_of_diagonals_l66_66634

-- Define vertices of a regular octagon
structure RegularOctagon :=
(vertices : Fin 8 → ℝ × ℝ)
(is_regular : ∀ i, ∥vertices (i + 1) - vertices i∥ = ∥vertices 0 - vertices 1∥)

-- Define the ratio problem
def diagonal_ratio_in_regular_octagon (O : RegularOctagon) : ℝ :=
  let S := ∥O.vertices 0 - O.vertices 1∥ -- Shortest diagonal (one step apart)
  let L := ∥O.vertices 0 - O.vertices 3∥ -- Longest diagonal (three steps apart)
  S / L

theorem ratio_of_diagonals (O : RegularOctagon) :
  diagonal_ratio_in_regular_octagon O = (Real.sqrt 2) / 2 :=
by sorry

end ratio_of_diagonals_l66_66634


namespace max_y_value_l66_66578

-- Definitions according to the problem conditions
def is_negative_integer (z : ℤ) : Prop := z < 0

-- The theorem to be proven
theorem max_y_value (x y : ℤ) (hx : is_negative_integer x) (hy : is_negative_integer y) 
  (h_eq : y = 10 * x / (10 - x)) : y = -5 :=
sorry

end max_y_value_l66_66578


namespace sum_of_three_numbers_l66_66416

theorem sum_of_three_numbers (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 12)
    (h4 : (a + b + c) / 3 = a + 8) (h5 : (a + b + c) / 3 = c - 18) : 
    a + b + c = 66 := 
sorry

end sum_of_three_numbers_l66_66416


namespace max_value_S_l66_66964

-- Define the conditions
def condition1 (x : ℝ) : Prop := abs (x ^ 2 - 3 * x - 4) < 2 * x + 2
def condition2 (m n : ℝ) : Prop := m ∈ Ioo (-1) 1 ∧ n ∈ Ioo (-1) 1
def condition3 (m n a b : ℝ) : Prop := m * n = a / b
def condition4 (m n a b S : ℝ) : Prop := S = a / (m ^ 2 - 1) + b / (3 * (n ^ 2 - 1))

-- Define the problem
theorem max_value_S : 
  (∃ a b S : ℝ, (∀ x : ℝ, condition1 x → (2 < x ∧ x < 6) ∧ a = 2 ∧ b = 6) ∧ 
                  (∀ (m n : ℝ), condition2 m n → condition3 m n a b → condition4 m n a b S) ∧ 
                  (-6 ≤ S)) :=
sorry

end max_value_S_l66_66964


namespace solve_logarithmic_exponential_equation_l66_66451

theorem solve_logarithmic_exponential_equation (x : ℝ) (h : 7^Real.log10 x - 5^((Real.log10 x) + 1) = 3 * 5^((Real.log10 x) - 1) - 13 * 7^((Real.log10 x) - 1)) : x = 100 :=
sorry

end solve_logarithmic_exponential_equation_l66_66451


namespace gain_percent_is_correct_l66_66138

/-- Define the variables CP and SP --/
def CP : ℝ := 850
def SP : ℝ := 1080

/-- Define the gain as SP - CP --/
def Gain : ℝ := SP - CP

/-- Define the gain percent as (Gain / CP) * 100 --/
def GainPercent : ℝ := (Gain / CP) * 100

/-- The proof statement that Gain Percent is approximately 27.06 --/
theorem gain_percent_is_correct : GainPercent ≈ 27.06 := sorry

end gain_percent_is_correct_l66_66138


namespace equivalence_1_meter_to_jumps_l66_66714

variables (a p q r s t u v : ℕ)

def hops_to_skips (a p : ℕ) : Prop := a * skips = p * hops
def jumps_to_hops (q r : ℕ) : Prop := q * jumps = r * hops
def skips_to_leaps (s t : ℕ) : Prop := s * skips = t * leaps
def leaps_to_meters (u v : ℕ) : Prop := u * leaps = v * meters

theorem equivalence_1_meter_to_jumps :
  hops_to_skips a p →
  jumps_to_hops q r →
  skips_to_leaps s t →
  leaps_to_meters u v →
  1 * meters = (usaq / pvtr) * jumps :=
by
  sorry

end equivalence_1_meter_to_jumps_l66_66714


namespace ben_consistent_speed_l66_66183

theorem ben_consistent_speed :
  (∀ t1 t2 d1 d2 : ℝ, (t1 = 2 ∧ d1 = 3) ∧ (t2 = 480 / 60 ∧ d2 = 12) → (d1 / t1 = d2 / t2)) →
  ∀ D : ℝ, let speed := 1.5 in ((D / speed) * 60) = (D * (60 / speed)) :=
by
  intro h D
  have consistent_speed : ∀ t1 t2 d1 d2, (t1 = 2 ∧ d1 = 3) ∧ (t2 = 480 / 60 ∧ d2 = 12) → (d1 / t1 = d2 / t2) := by
    intros t1 t2 d1 d2 cond
    rw [div_eq_div_iff] at *
    cases cond with first second
    cases first with h1 h2
    cases second with h3 h4
    rw [h1, h2] at *
    rw [h3, h4] at *
    norm_num
  specialize consistent_speed 2 (480 / 60) 3 12 (and.intro ⟨rfl, rfl⟩ ⟨by norm_num, rfl⟩)
  change (D / 1.5) * 60 = D * (60 / 1.5)
  norm_num
  ring
  sorry

end ben_consistent_speed_l66_66183


namespace find_p_over_q_l66_66905

variables (x y p q : ℚ)

theorem find_p_over_q (h1 : (7 * x + 6 * y) / (x - 2 * y) = 27)
                      (h2 : x / (2 * y) = p / q) :
                      p / q = 3 / 2 :=
sorry

end find_p_over_q_l66_66905


namespace convert_speed_l66_66221

theorem convert_speed (v_kmph : ℝ) (conversion_factor : ℝ) : 
  v_kmph = 252 → conversion_factor = 0.277778 → v_kmph * conversion_factor = 70 := by
  intros h1 h2
  rw [h1, h2]
  sorry

end convert_speed_l66_66221


namespace distance_from_point_to_plane_correct_l66_66369

noncomputable def distance_from_point_to_plane
  (A B C T : ℝ×ℝ×ℝ)
  (h1 : (A.1 - T.1) * (B.1 - T.1) = 0)
  (h2 : (A.1 - T.1) * (C.1 - T.1) = 0)
  (h3 : (B.1 - T.1) * (C.1 - T.1) = 0)
  (TA : ℝ := 8)
  (TB : ℝ := 10)
  (TC : ℝ := 15) : ℝ :=
15

theorem distance_from_point_to_plane_correct 
  (A B C T : ℝ×ℝ×ℝ)
  (h1 : (A.1 - T.1) * (B.1 - T.1) = 0)
  (h2 : (A.1 - T.1) * (C.1 - T.1) = 0)
  (h3 : (B.1 - T.1) * (C.1 - T.1) = 0)
  (hTA : dist A T = 8)
  (hTB : dist B T = 10)
  (hTC : dist C T = 15) : 
  distance_from_point_to_plane A B C T h1 h2 h3 hTA hTB hTC = 15 :=
by sorry

end distance_from_point_to_plane_correct_l66_66369


namespace sum_odd_divisors_l66_66097

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l66_66097


namespace quadratic_function_range_l66_66326

theorem quadratic_function_range (a b c : ℝ) (x y : ℝ) :
  (∀ x, x = -4 → y = a * (-4)^2 + b * (-4) + c → y = 3) ∧
  (∀ x, x = -3 → y = a * (-3)^2 + b * (-3) + c → y = -2) ∧
  (∀ x, x = -2 → y = a * (-2)^2 + b * (-2) + c → y = -5) ∧
  (∀ x, x = -1 → y = a * (-1)^2 + b * (-1) + c → y = -6) ∧
  (∀ x, x = 0 → y = a * 0^2 + b * 0 + c → y = -5) →
  (∀ x, x < -2 → y > -5) :=
sorry

end quadratic_function_range_l66_66326


namespace simplified_expr_eval_l66_66710

theorem simplified_expr_eval
  (x : ℚ) (y : ℚ) (h_x : x = -1/2) (h_y : y = 1) :
  (5*x^2 - 10*y^2) = -35/4 := 
by
  subst h_x
  subst h_y
  sorry

end simplified_expr_eval_l66_66710


namespace correct_statements_l66_66262

-- Definitions and assumptions
variable {a : ℕ → ℝ} -- sequence
variable {S : ℕ → ℝ} -- sum sequence
variable (arithmetic : Prop) -- sequence is arithmetic
variable (geometric : Prop) -- sequence is geometric
variable (S_positive : ∀ n, S n > 0) -- sum is always positive
variable (S61 : S 6 = S 11) -- specific sum condition
variable (a1_pos : a 1 > 0) -- initial condition on first term

-- Statement of the theorem
theorem correct_statements :
  (arithmetic → ∃ d, ∀ n, a (n + 1) - a n = d ∧ (∀ n, 2 ^ (a (n + 1) - a n) = 2 ^ d)) ∧ -- statement ①
  (arithmetic → S_positive → ∀ d, d > 0 → ∀ n > 0, a n < a (n + 1)) ∧ -- statement ②
  (¬ geometric ∨ ∀ r, ∃ k, a k ≤ 0 → log 2 (a k) ∉ ℝ∞) ∧ -- statement ③
  (arithmetic → a1_pos → S61 → ∃ n, n = 8 ∨ n = 9 ∧ (∀ m, m ≥ 1 → S m ≤ S n)) -- statement ④
:=
  sorry

end correct_statements_l66_66262


namespace projection_of_a_plus_b_onto_a_l66_66281

-- Definitions based on given conditions
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-2, 1)

-- Function to calculate the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Function to calculate the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Projection calculation
def projection (v1 v2 : ℝ × ℝ) : ℝ :=
  (dot_product v1 v2) / (magnitude v2)

theorem projection_of_a_plus_b_onto_a :
  projection (a.1 + b.1, a.2 + b.2) a = Real.sqrt(2) / 2 :=
by
  sorry

end projection_of_a_plus_b_onto_a_l66_66281


namespace range_of_C₁_x_range_of_C₁_y_polar_of_C₁_intersection_P_intersection_A_intersection_B_area_triangle_PAB_l66_66325

noncomputable def C₁_range_x : Set ℝ := {x | -1 < x ∧ x ≤ 1}

noncomputable def C₁_range_y : Set ℝ := {y | 0 ≤ y ∧ y ≤ 2}

noncomputable def polar_eq_C₁ : (ρ θ : ℝ) → Prop := λ ρ θ, ρ = 2 * Real.sin θ

noncomputable def intersection_p : (ρ θ : ℝ) → Prop := λ ρ θ, ρ = 2 ∧ θ = π / 2

noncomputable def intersection_a : (ρ θ : ℝ) → Prop := λ ρ θ, ρ = 1 ∧ θ = π / 6

noncomputable def intersection_b : (ρ θ : ℝ) → Prop := λ ρ θ, ρ = 3 ∧ θ = π / 6

noncomputable def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  let (x1, y1) := a in
  let (x2, y2) := b in
  let (x3, y3) := c in
  0.5 * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

theorem range_of_C₁_x : C₁_range_x = {x | -1 < x ∧ x ≤ 1} := sorry

theorem range_of_C₁_y : C₁_range_y = {y | 0 ≤ y ∧ y ≤ 2} := sorry

theorem polar_of_C₁ : ∀ ρ θ, polar_eq_C₁ ρ θ ↔ ρ = 2 * Real.sin θ := sorry

theorem intersection_P : intersection_p 2 (π / 2) := sorry

theorem intersection_A : intersection_a 1 (π / 6) := sorry

theorem intersection_B : intersection_b 3 (π / 6) := sorry

theorem area_triangle_PAB :
  triangle_area (2, π / 2) (1, π / 6) (3, π / 6) = sqrt 3 := sorry

end range_of_C₁_x_range_of_C₁_y_polar_of_C₁_intersection_P_intersection_A_intersection_B_area_triangle_PAB_l66_66325


namespace sequence_product_l66_66040

theorem sequence_product : 
  ( ∏ i in Finset.range 2004, (i + 3) / (i + 2) ) = 1003 := 
by
  sorry

end sequence_product_l66_66040


namespace exponentiation_addition_zero_l66_66186

theorem exponentiation_addition_zero : (-2)^(3^2) + 2^(3^2) = 0 := 
by 
  -- proof goes here
  sorry

end exponentiation_addition_zero_l66_66186


namespace find_angle_ABC_l66_66637

noncomputable def triangle := (A B C : Type) [noncomputable geometry.Euclidean] 

variables {A B C D : triangle}
variables (AB AC BC : Real)
variables (angle_ABC angle_ACB angle_ADC : Real)

axiom is_isosceles_triangle : AB = AC
axiom is_perpendicular_bisector : midpoint D BC ∧ angle_ADC = 90
axiom is_AC_equal_AD : AC = AD

open EuclideanGeometry

theorem find_angle_ABC : angle_ABC = 36 := by
  sorry

end find_angle_ABC_l66_66637


namespace number_of_participants_l66_66308

theorem number_of_participants (n : ℕ) :
  (∀ i : ℕ, i < 10 → ∀ j : ℕ, j ≠ i → i ≠ j) →
  (∀ k : Fin n, ∑ l in (Finset.erase Finset.univ k), (if l < 10 then 0.5 else 1) = n * (n - 1) / 2) →
  (∀ i : ℕ, 9 < i ∧ i < n → ∑ j in Finset.Icc 0 10, (if i ≠ j then 1 else 0)) 
  = 25 := 
sorry

end number_of_participants_l66_66308


namespace line_AC_eqn_l66_66568

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

-- Define point A
def A : Point := { x := 3, y := 1 }

-- Define point B
def B : Point := { x := -1, y := 2 }

-- Define the line equation y = x + 1
def line_eq (p : Point) : Prop := p.y = p.x + 1

-- Define the bisector being on line y=x+1 as a condition
axiom bisector_on_line (C : Point) : 
  line_eq C → (∃ k : ℝ, (C.y - B.y) = k * (C.x - B.x))

-- Define the final goal to prove the equation of line AC
theorem line_AC_eqn (C : Point) :
  line_eq C → ((A.x - C.x) * (B.y - C.y) = (B.x - C.x) * (A.y - C.y)) → C.x = -3 ∧ C.y = -2 → 
  (A.x - 2 * A.y = 1) := sorry

end line_AC_eqn_l66_66568


namespace damage_in_usd_l66_66480

-- Defining the conditions
def australian_damage : ℝ := 30_000_000
def exchange_rate : ℝ := 1.5

-- Theorem statement to prove the equivalent American dollars
theorem damage_in_usd : (australian_damage / exchange_rate) = 20_000_000 := by
  sorry

end damage_in_usd_l66_66480


namespace range_of_3a_minus_b_l66_66915

theorem range_of_3a_minus_b (a b : ℝ) (h1 : 2 ≤ a + b ∧ a + b ≤ 5) (h2 : -2 ≤ a - b ∧ a - b ≤ 1) : 
    -2 ≤ 3 * a - b ∧ 3 * a - b ≤ 7 := 
by 
  sorry

end range_of_3a_minus_b_l66_66915


namespace geometric_progression_condition_l66_66875

theorem geometric_progression_condition {b : ℕ → ℝ} (b1_ne_b2 : b 1 ≠ b 2) (h : ∀ n, b (n + 2) = b n / b (n + 1)) :
  (∀ n, b (n+1) / b n = b 2 / b 1) ↔ b 1 = b 2^3 := sorry

end geometric_progression_condition_l66_66875


namespace common_area_approximation_l66_66265

noncomputable def elliptical_domain (x y : ℝ) : Prop :=
  (x^2 / 3 + y^2 / 2) ≤ 1

noncomputable def circular_domain (x y : ℝ) : Prop :=
  (x^2 + y^2) ≤ 2

noncomputable def intersection_area : ℝ :=
  7.27

theorem common_area_approximation :
  ∃ area, 
    elliptical_domain x y ∧ circular_domain x y →
    abs (area - intersection_area) < 0.01 :=
sorry

end common_area_approximation_l66_66265


namespace competition_arrangements_l66_66912

theorem competition_arrangements (students : Finset ℕ) (competitions : Finset ℕ) (A : ℕ)
  (h1 : students.card = 5) (h2 : competitions.card = 4)
  (h3 : A ∈ students) (h4 : ∀ x ∈ students, x ≠ A → x ∉ {1, 2}) :
  ∃ arrangements : Finset (Finset (ℕ × ℕ)), arrangements.card = 72 :=
sorry

end competition_arrangements_l66_66912


namespace piecewise_function_f_2_l66_66243

def f (x: ℝ) : ℝ :=
if x < 0 then sin (π / 2 * x)
else f (x - 1) + 2

theorem piecewise_function_f_2 :
  f 2 = 5 :=
sorry

end piecewise_function_f_2_l66_66243


namespace rectangle_perimeter_eq_circle_circumference_l66_66403

theorem rectangle_perimeter_eq_circle_circumference (l : ℝ) :
  2 * (l + 3) = 10 * Real.pi -> l = 5 * Real.pi - 3 :=
by
  intro h
  sorry

end rectangle_perimeter_eq_circle_circumference_l66_66403


namespace find_valid_numbers_l66_66401

theorem find_valid_numbers :
  let N_remainders (N : ℕ) := (N % 6 = 5) ∧ (N % 7 = 5) ∧ (N % 8 = 5) ∧ (N % 9 = 5)
  in ∃ (count : ℕ), count = 4 ∧ (∀ (N : ℕ), N < 2021 → N_remainders N → N ∈ {5, 509, 1013, 1517}) := by
  sorry

end find_valid_numbers_l66_66401


namespace jellybeans_in_carrie_box_l66_66232

-- Define the conditions
def bert_box_capacity : ℝ := 200
def dimension_scale : ℝ := 1.5
def fullness_ratio : ℝ := 0.75

-- Define the problem
theorem jellybeans_in_carrie_box :
  let volume_ratio := (dimension_scale * dimension_scale * dimension_scale) in
  let carrie_box_full_capacity := volume_ratio * bert_box_capacity in
  fullness_ratio * carrie_box_full_capacity = 506.25 := 
by
  let volume_ratio := (dimension_scale * dimension_scale * dimension_scale)
  let carrie_box_full_capacity := volume_ratio * bert_box_capacity
  show fullness_ratio * carrie_box_full_capacity = 506.25
  sorry

end jellybeans_in_carrie_box_l66_66232


namespace Smarandache_S16_S2016_Smarandache_S7_max_n_Smarandache_infinitely_many_composite_p_l66_66621

-- Definition of the Smarandache function
def Smarandache_function (n : ℕ) : ℕ := Nat.find (λ m, n ∣ m!)

-- Prove that S(16) = 6 and S(2016) = 8
theorem Smarandache_S16_S2016 : Smarandache_function 16 = 6 ∧ Smarandache_function 2016 = 8 :=
by {
  sorry
}

-- Prove that if S(n) = 7, the maximum value of n is 5040
theorem Smarandache_S7_max_n : ∀ n, Smarandache_function n = 7 → n ≤ 5040 :=
by {
  sorry
}

-- Prove that there are infinitely many composite numbers n such that S(n) = p
-- where p is the largest prime factor of n
theorem Smarandache_infinitely_many_composite_p {p : ℕ} [Fact p.Prime] :
  ∃ᶠ (n : ℕ) in at_top, ¬ (Nat.Prime n) ∧ Smarandache_function n = p ∧ Nat.gcd n p = p :=
by {
  sorry
}

end Smarandache_S16_S2016_Smarandache_S7_max_n_Smarandache_infinitely_many_composite_p_l66_66621


namespace chocolate_chip_more_than_raisin_l66_66284

def chocolate_chip_yesterday : ℕ := 19
def chocolate_chip_morning : ℕ := 237
def raisin_cookies : ℕ := 231

theorem chocolate_chip_more_than_raisin : 
  (chocolate_chip_yesterday + chocolate_chip_morning) - raisin_cookies = 25 :=
by 
  sorry

end chocolate_chip_more_than_raisin_l66_66284


namespace a_n_general_form_b_n_general_form_c_sum_3_terms_sum_first_3n_terms_l66_66564

namespace MathProblems

-- Definitions for sequences and conditions
def a_n (n : ℕ) : ℝ := n / 3

def b_n (n : ℕ) : ℕ := 2 ^ n

def c_n (n : ℕ) : ℝ := (b_n n) * Real.tan (a_n n * Real.pi)

-- Prove the general formula for a_n given its conditions
theorem a_n_general_form :
  (∀ n : ℕ, a_n n = (n : ℝ) / 3) :=
sorry

-- Prove the general formula for b_n given its conditions
theorem b_n_general_form :
  (∀ n : ℕ, b_n n = 2 ^ n) :=
sorry

-- Prove the value of c_1 + c_2 + c_3
theorem c_sum_3_terms :
  c_n 1 + c_n 2 + c_n 3 = -2 * Real.sqrt 3 :=
sorry

-- Prove the sum of the first 3n terms of the sequence c_n
theorem sum_first_3n_terms (n : ℕ) :
  ∑ i in finset.range (3 * n), c_n (i + 1) = (2 * Real.sqrt 3 * (1 - 8 ^ n)) / 7 :=
sorry

end MathProblems

end a_n_general_form_b_n_general_form_c_sum_3_terms_sum_first_3n_terms_l66_66564


namespace shopkeeper_oranges_l66_66155

theorem shopkeeper_oranges (O : ℕ) 
  (bananas : ℕ) 
  (percent_rotten_oranges : ℕ) 
  (percent_rotten_bananas : ℕ) 
  (percent_good_condition : ℚ) 
  (h1 : bananas = 400) 
  (h2 : percent_rotten_oranges = 15) 
  (h3 : percent_rotten_bananas = 6) 
  (h4 : percent_good_condition = 88.6) : 
  O = 600 :=
by
  -- This proof needs to be filled in.
  sorry

end shopkeeper_oranges_l66_66155


namespace decrypt_message_base7_l66_66837

noncomputable def base7_to_base10 : Nat := 
  2 * 343 + 5 * 49 + 3 * 7 + 4 * 1

theorem decrypt_message_base7 : base7_to_base10 = 956 := 
by 
  sorry

end decrypt_message_base7_l66_66837


namespace probability_integer_division_l66_66394

open Set

-- Definitions for the sets of possible values for r and k
def valid_r : Set ℤ := {r | -5 < r ∧ r < 7}
def valid_k : Set ℕ := {k | 2 < k ∧ k < 10}

-- Function to determine if the division results in an integer
def is_integer_division (r : ℤ) (k : ℕ) : Bool := r % k = 0

-- The theorem to prove the probability
theorem probability_integer_division : 
  (∑ r in valid_r.to_finset, ∑ k in valid_k.to_finset, if is_integer_division r k then 1 else 0) / 
  (∑ _ in valid_r.to_finset, ∑ _ in valid_k.to_finset, 1) = 2 / 11 := 
sorry

end probability_integer_division_l66_66394


namespace max_value_of_f_and_symmetry_axis_and_g_range_l66_66582

theorem max_value_of_f_and_symmetry_axis_and_g_range :
  ∀ (f : ℝ → ℝ) (a : ℝ),
    (∀ x, f x = 2 * cos x ^ 2 + 2 * sqrt 3 * sin x * cos x + a) →
    (∀ y, y ≤ 2) →
    a = -1 ∧
      ∃ k : ℤ, ∃ x : ℝ, x = (k * π / 2) + (π / 6) ∧
      (∀ x, (x ≥ π / 6 ∧ x ≤ π / 3) → g x = f (x - π / 12) ∧ (sqrt 3 ≤ g x ∧ g x ≤ 2))
  ∧
    (∀ x, g x = f (x - π / 12)) :=
begin
  sorry
end

end max_value_of_f_and_symmetry_axis_and_g_range_l66_66582


namespace sum_of_odd_divisors_of_90_l66_66088

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l66_66088


namespace equal_water_depth_l66_66747

noncomputable def cylinder_volume (r h : ℝ) : ℝ := π * r * r * h

theorem equal_water_depth (h : ℝ) 
    (r1 h1 r2 h2 : ℝ)
    (V1 V2 : ℝ) 
    (condition1 : r1 = 4) (condition2 : h1 = 10)
    (condition3 : r2 = 6) (condition4 : h2 = 8)
    (condition5 : V1 = cylinder_volume r1 h1) 
    (condition6 : V1 = 160 * π) 
    (condition7 : V2 = cylinder_volume r2 h) 
    (condition8 : h = 40 / 13) 
  : cylinder_volume r1 h + cylinder_volume r2 h = V1 :=
by 
  rw [cylinder_volume]
  sorry

end equal_water_depth_l66_66747


namespace minimum_value_OS_l66_66492

noncomputable def minimum_OS (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : ℝ :=
  let ρ := λ θ : ℝ, 1 / (sin θ + cos θ) in
  ρ θ -- deriving from the conditions that ρ = \( \frac{2}{(\sqrt{5} \sin(2θ + φ) + 3)} \) 

theorem minimum_value_OS (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) :
  (minimum_OS θ hθ) >= (sqrt 5 - 1) / 2 :=
sorry

end minimum_value_OS_l66_66492


namespace interval_monotonically_decreasing_find_ABC_sides_area_l66_66282

noncomputable def f (x : ℝ) : ℝ :=
  let m := (Real.sin x, -1)
  let n := (Real.sqrt 3 * Real.cos x, -1 / 2)
  (m.1 + n.1, m.2 + n.2) • n

theorem interval_monotonically_decreasing :
  ∀ (k : ℤ), (∀ x ∈ Set.Icc (k * Real.pi + Real.pi / 3) (k * Real.pi + 5 * Real.pi / 6), 
  ∀ y ∈ Set.Icc (k * Real.pi + Real.pi / 3) (k * Real.pi + 5 * Real.pi / 6), f x ≤ f y → x ≤ y) := sorry

theorem find_ABC_sides_area :
  ∃ (A b S : ℝ), 
  A = Real.pi / 3 ∧ 
  b = 2 ∧ 
  S = 2 * Real.sqrt 3 := sorry

end interval_monotonically_decreasing_find_ABC_sides_area_l66_66282


namespace necessary_but_not_sufficient_converse_implies_l66_66236

theorem necessary_but_not_sufficient (x : ℝ) (hx1 : 1 < x) (hx2 : x < Real.exp 1) : 
  (x * (Real.log x) ^ 2 < 1) → (x * Real.log x < 1) :=
sorry

theorem converse_implies (x : ℝ) (hx1 : 1 < x) (hx2 : x < Real.exp 1) : 
  (x * Real.log x < 1) → (x * (Real.log x) ^ 2 < 1) :=
sorry

end necessary_but_not_sufficient_converse_implies_l66_66236


namespace regular_17_gon_symmetry_sum_l66_66154

theorem regular_17_gon_symmetry_sum : let L := 17 in let R := 360 / 17 in L + R = 649 / 17 :=
by
  sorry

end regular_17_gon_symmetry_sum_l66_66154


namespace sum_odd_divisors_of_90_l66_66102

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l66_66102


namespace train_length_l66_66846

theorem train_length (speed_km_hr : ℕ) (time_seconds : ℝ) (platform_length_m : ℝ) :
  speed_km_hr = 45 → time_seconds = 40.8 → platform_length_m = 150 →
  let speed_m_s := speed_km_hr * (1000 / 3600) in
  let total_distance := speed_m_s * time_seconds in
  let train_length := total_distance - platform_length_m in
  train_length = 360 :=
by
  intros h_speed h_time h_platform
  let speed_m_s := speed_km_hr * (1000 / 3600)
  have speed_calc : speed_m_s = 12.5 := by sorry -- speed conversion calculation
  let total_distance := speed_m_s * time_seconds
  have distance_calc : total_distance = 510 := by sorry -- total distance calculation
  let train_length := total_distance - platform_length_m
  have train_length_calc : train_length = 360 := by sorry -- train length calculation
  exact train_length_calc

end train_length_l66_66846


namespace sum_odd_divisors_90_eq_78_l66_66058

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l66_66058


namespace angle_between_vectors_l66_66969

variable {a b : EuclideanSpace}

-- Given conditions
def condition1 : ℝ := a ⋅ (a + b) = 5
def condition2 : ℝ := ‖a‖ = 2
def condition3 : ℝ := ‖b‖ = 1

-- Statement to prove
theorem angle_between_vectors (a b : EuclideanSpace) (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  ∃ θ : ℝ, θ = Real.pi / 3 :=
by
  sorry

end angle_between_vectors_l66_66969


namespace hyperbola_eccentricity_proof_l66_66963

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let c := 1 in
  c / (a / 2)

theorem hyperbola_eccentricity_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyp_eq : ℝ := (√5 + 1) / 2
  let c := 1
  ∃ (P : ℝ × ℝ), 
  (P.1 = 1) ∧ (P.2 = 1) ∧ (a = (√5 - 1) / 2) ∧ hyperbola_eccentricity a b ha hb = hyp_eq :=
begin
  sorry
end

end hyperbola_eccentricity_proof_l66_66963


namespace edge_coloring_k6_l66_66795

open Finset

noncomputable def complete_graph_6 : SimpleGraph (Fin 6) := ⟨λ a b, a ≠ b, λ a, not_irrefl _⟩

theorem edge_coloring_k6 :
  ∃ (coloring : (Sym2 (Fin 6)) → Fin 5), 
    ∀ (v : Fin 6), (∃ s : Finset (Fin 5), s.card = 5 ∧ ∀ u, u ∈ complete_graph_6.neighborFinset v → coloring ⟦(v, u)⟧ ∈ s) :=
sorry

end edge_coloring_k6_l66_66795


namespace second_number_l66_66454

theorem second_number (x : ℝ) (h : 3 + x + 333 + 33.3 = 399.6) : x = 30.3 :=
sorry

end second_number_l66_66454


namespace find_sum_of_x_and_reciprocal_l66_66944

theorem find_sum_of_x_and_reciprocal (x : ℝ) (hx_condition : x^3 + 1/x^3 = 110) : x + 1/x = 5 := 
sorry

end find_sum_of_x_and_reciprocal_l66_66944


namespace cos_shift_eq_sin_l66_66024

theorem cos_shift_eq_sin (x : ℝ) : cos (2 * (x - π / 4)) = sin (2 * x) :=
by
  sorry

end cos_shift_eq_sin_l66_66024


namespace Fiona_reaches_pad_14_without_predators_l66_66021

-- Define the probability calculation environment and conditions
def lily_pads := Fin 16
def predators (pad : lily_pads) : Prop := pad = 2 ∨ pad = 5 ∨ pad = 8
def morsel (pad : lily_pads) : Prop := pad = 14
def start_pad : lily_pads := 0
def hop_probability : ℚ := 1/2
def jump_probability : ℚ := 1/2

noncomputable def frog_reaches_without_predator : ℚ := 5/1024

-- Now we state the theorem
theorem Fiona_reaches_pad_14_without_predators (Fiona : lily_pads → ℚ) : 
  (∀ pad : lily_pads, predators pad → Fiona pad = 0) ∧ (Fiona start_pad = 1) ∧ (Fiona 14 = frog_reaches_without_predator) := 
sorry

end Fiona_reaches_pad_14_without_predators_l66_66021


namespace sufficient_conditions_for_perpendicular_l66_66953

variables (l a : Line) (α : Plane) (convex_pentagon : Pentagon) (regular_hexagon : Hexagon)

-- Conditions
def cond1 : Prop := l ⟂ convex_pentagon.side1 ∧ l ⟂ convex_pentagon.side2
def cond2 : Prop := ∃ (s1 s2 s3 : Line), s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧ l ⟂ s1 ∧ l ⟂ s2 ∧ l ⟂ s3 ∧ s1 ∈ α ∧ s2 ∈ α ∧ s3 ∈ α
def cond3 : Prop := ∀ (s : Line), s ∈ α → l ⟂ s
def cond4 : Prop := l ⟂ regular_hexagon.side1 ∧ l ⟂ regular_hexagon.side2 ∧ l ⟂ regular_hexagon.side3
def cond5 : Prop := a ⟂ α ∧ l ⟂ a

-- Proof that l is perpendicular to α given certain conditions
theorem sufficient_conditions_for_perpendicular (h2 : cond2) (h4 : cond4) : l ⟂ α :=
by sorry

end sufficient_conditions_for_perpendicular_l66_66953


namespace total_fruits_in_30_days_l66_66753

-- Define the number of oranges Sophie receives each day
def sophie_daily_oranges : ℕ := 20

-- Define the number of grapes Hannah receives each day
def hannah_daily_grapes : ℕ := 40

-- Define the number of days
def number_of_days : ℕ := 30

-- Calculate the total number of fruits received by Sophie and Hannah in 30 days
theorem total_fruits_in_30_days :
  (sophie_daily_oranges * number_of_days) + (hannah_daily_grapes * number_of_days) = 1800 :=
by
  sorry

end total_fruits_in_30_days_l66_66753


namespace number_of_valid_pairings_l66_66020

-- Define the conditions:
def knows (n : ℕ) (a b : ℕ) : Prop :=
  b = (a + 2) % n ∨ b = (a - 2 + n) % n ∨ b = (a + n / 2) % n

-- Define the problem for 12 people and prove there are exactly 5 valid pairings:
theorem number_of_valid_pairings : 
  (finset.univ : finset (fin 12)).card = 12 → 
  (∀ a b : fin 12, knows 12 a b → knows 12 b a) →
  ∃! S : finset (fin 12 × fin 12), S.card = 6 ∧
  (∀ (p : fin 12 × fin 12), p ∈ S → knows 12 p.1 p.2) :=
sorry

end number_of_valid_pairings_l66_66020


namespace a_n_arithmetic_T_n_leq_S_n_l66_66730

def f : ℝ → ℝ

axiom f_property : ∀ x : ℝ, f x + f (1 - x) = 1 / 2

def a_n (n : ℕ) : ℝ :=
  f 0 + (finset.range n).sum (λ k, f (k / n)) + f 1

theorem a_n_arithmetic : ∀ n : ℕ, a_n n = (n + 1) / 4 :=
by sorry

def b_n (n : ℕ) : ℝ := 4 / (4 * a_n n - 1)

def T_n (n : ℕ) : ℝ := (finset.range n).sum (λ k, b_n (k + 1) ^ 2)

def S_n (n : ℕ) : ℝ := 32 - 16 / n

theorem T_n_leq_S_n : ∀ n : ℕ, T_n n ≤ S_n n :=
by sorry

end a_n_arithmetic_T_n_leq_S_n_l66_66730


namespace fruits_eaten_total_l66_66754

variable (oranges_per_day : ℕ) (grapes_per_day : ℕ) (days : ℕ)

def total_fruits (oranges_per_day grapes_per_day days : ℕ) : ℕ :=
  (oranges_per_day * days) + (grapes_per_day * days)

theorem fruits_eaten_total 
  (h1 : oranges_per_day = 20)
  (h2 : grapes_per_day = 40) 
  (h3 : days = 30) : 
  total_fruits oranges_per_day grapes_per_day days = 1800 := 
by 
  sorry

end fruits_eaten_total_l66_66754


namespace distinct_remainders_l66_66175

theorem distinct_remainders (p : ℕ) (hp : Nat.Prime p) (h7 : 7 < p) : 
  let remainders := 
    { r | r ∈ Finset.univ.filter (λ r, ∃ k, p^2 = 210 * k + r) } 
  remainders.card = 6 :=
sorry

end distinct_remainders_l66_66175


namespace expected_value_of_10_sided_die_l66_66163

-- Definition of the conditions
def num_faces : ℕ := 10
def face_values : List ℕ := List.range' 2 num_faces

-- Theorem statement: The expected value of a roll of this die is 6.5
theorem expected_value_of_10_sided_die : 
  (List.sum face_values : ℚ) / num_faces = 6.5 := 
sorry

end expected_value_of_10_sided_die_l66_66163


namespace problem_statement_l66_66681

def x : ℝ := 2023^1011 - 2023^(-1011)
def y : ℝ := 2023^1011 + 2023^(-1011)

theorem problem_statement : x^2 - y^2 = -4 := 
by
  sorry

end problem_statement_l66_66681


namespace impossible_pawn_moves_l66_66364

theorem impossible_pawn_moves :
  let initial_positions := [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
  let target_positions := [(6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8), (8, 6), (8, 7), (8, 8)]
  ∀ (m n k ℓ : ℕ), 
    ({coord | coord ∈ list.map (λ (p : ℕ × ℕ), prod.fst p + 2 * m * n = 0 ∧ prod.snd p + 2 * k * ℓ = 0) initial_positions}).card ≠
    ({coord | coord ∈ initial_positions}).card :=
begin
  sorry
end

end impossible_pawn_moves_l66_66364


namespace complex_sum_identity_l66_66344

noncomputable def omega : ℂ := algebraic_root (λ z : ℂ, z^3 = 1) (λ z : ℂ, z ≠ 1)

theorem complex_sum_identity
  (b : ℕ → ℝ)
  (m : ℕ)
  (h_eq : ∑ k in finset.range m, 1 / (b k + omega) = (3 : ℂ) + 8 * I) :
  ∑ k in finset.range m, (2 * (b k : ℂ) - 1) / ((b k : ℂ)^2 - (b k : ℂ) + 1) = 6 := 
  sorry

end complex_sum_identity_l66_66344


namespace volume_solid_cylinder_plane_l66_66864

theorem volume_solid_cylinder_plane :
  let cylinder := { p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 1 }
  let height (p : ℝ × ℝ) := 2 - p.1 - p.2
  let volume := ∫∫ (x y : ℝ) in cylinder, height (x, y)
  volume = 2 * Real.pi :=
sorry

end volume_solid_cylinder_plane_l66_66864


namespace solve_for_k_l66_66294

noncomputable def polynomial_is_perfect_square (k : ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, (f x) ^ 2 = x^2 - 2*(k+1)*x + 4

theorem solve_for_k :
  ∀ k : ℝ, polynomial_is_perfect_square k → (k = -3 ∨ k = 1) := by
  sorry

end solve_for_k_l66_66294


namespace basketball_game_count_l66_66132

noncomputable def total_games_played (teams games_each_opp : ℕ) : ℕ :=
  (teams * (teams - 1) / 2) * games_each_opp

theorem basketball_game_count (n : ℕ) (g : ℕ) (h_n : n = 10) (h_g : g = 4) : total_games_played n g = 180 :=
by
  -- Use 'h_n' and 'h_g' as hypotheses
  rw [h_n, h_g]
  show (10 * 9 / 2) * 4 = 180
  sorry

end basketball_game_count_l66_66132


namespace find_real_values_x_l66_66530

theorem find_real_values_x (x : ℝ) :
  (1 / (x + 2) + 5 / (x + 4) ≥ 1) ↔ (x ∈ set.Icc (-4 : ℝ) (-3) ∪ set.Icc (-2) (2)) :=
begin
  sorry
end

end find_real_values_x_l66_66530


namespace books_permutation_books_step_counting_l66_66127

-- Declare the necessary combinatorial functions
def perm (n k : ℕ) : ℕ := (list.range n).permutations.length / list.range (n - k).length

-- Declare the number of different books and students
def num_books : ℕ := 5
def num_students : ℕ := 3

-- Problem 1: Permutation problem
theorem books_permutation : perm num_books num_students = 60 :=
by sorry

-- Problem 2: Step counting problem
theorem books_step_counting : num_books ^ num_students = 125 :=
by sorry

end books_permutation_books_step_counting_l66_66127


namespace box_volume_proof_l66_66406

-- Definitions of the given problem conditions
def length : ℝ := 6
def width : ℝ := 6
def shortest_distance : ℝ := 20

-- The height h that makes the shortest_path calculation valid
def height (h : ℝ) : Prop :=
  (length + width + h) ^ 2 + width ^ 2 = shortest_distance ^ 2

-- The volume of the rectangular box
def volume (l w h : ℝ) : ℝ := l * w * h

-- Theorem stating that given the conditions, the volume is 576 cm³
theorem box_volume_proof (h : ℝ) (H : height h) : volume length width h = 576 := 
  sorry

end box_volume_proof_l66_66406


namespace sum_of_positive_odd_divisors_of_90_l66_66076

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l66_66076


namespace roots_of_quadratic_eq_l66_66295

theorem roots_of_quadratic_eq {x1 x2 : ℝ} (h1 : x1 * x1 - 3 * x1 - 5 = 0) (h2 : x2 * x2 - 3 * x2 - 5 = 0) 
                              (h3 : x1 + x2 = 3) (h4 : x1 * x2 = -5) : x1^2 + x2^2 = 19 := 
sorry

end roots_of_quadratic_eq_l66_66295


namespace number_of_valid_polynomials_l66_66900

noncomputable def count_valid_polynomials : ℕ :=
  let ω := complex.exp (2 * real.pi * complex.I / 3) in
  let polynomials := λ (p : polynomial ℝ), 
    p.map complex = polynomial.X ^ 4 + 
                   polynomial.C (complex.of_real p.coeff_3) * polynomial.X ^ 3 + 
                   polynomial.C (complex.of_real p.coeff_2) * polynomial.X ^ 2 + 
                   polynomial.C (complex.of_real p.coeff_1) * polynomial.X + 
                   polynomial.C 2400 ∧ 
    ∀ r : complex, p.is_root r → p.is_root (ω * r) in
  finset.card (finset.filter (λ p, polynomials p) finset.univ)

theorem number_of_valid_polynomials : count_valid_polynomials = 1 :=
  sorry

end number_of_valid_polynomials_l66_66900


namespace dina_overtakes_laura_l66_66336

variable (n s : ℝ)
variable (hn : 1 < n)

theorem dina_overtakes_laura :
  ∃ d : ℝ, d = (n * s) / (n - 1) :=
by
  have h : ∀d, d = (n * s) / (n - 1)
  exact sorry

end dina_overtakes_laura_l66_66336


namespace solution_set_inequality_l66_66276

variable {x a b : ℝ}

theorem solution_set_inequality (h1 : ∀ x, ax - b > 0 → x ∈ Ioi 1) :
  {x | (ax + b) * (x - 2) > 0} = Iio (-1) ∪ Ioi 2 := 
by
  sorry

end solution_set_inequality_l66_66276


namespace find_arc_length_and_area_l66_66579

noncomputable def θ : ℝ := Real.pi / 3
def r : ℝ := 10
def arc_length : ℝ := θ * r
def sector_area : ℝ := (1 / 2) * θ * r^2

theorem find_arc_length_and_area :
  arc_length = (10 * Real.pi) / 3 ∧ sector_area = (50 * Real.pi) / 3 :=
by
  sorry

end find_arc_length_and_area_l66_66579


namespace number_of_solutions_l66_66347

noncomputable def f (x : ℝ) : ℝ :=
  |x^2 - 2 * x - 3|

theorem number_of_solutions : ∃ (s : finset ℝ), ∀ x : ℝ, (f x)^3 - 4 * (f x)^2 - f x + 4 = 0 ↔ x ∈ s ∧ s.card = 7 :=
by
  sorry

end number_of_solutions_l66_66347


namespace number_of_smaller_type_pages_l66_66312

theorem number_of_smaller_type_pages :
  ∃ (x y : ℕ), x + y = 21 ∧ 1800 * x + 2400 * y = 48000 ∧ y = 17 :=
by
  use 4   -- larger type pages
  use 17  -- smaller type pages
  split
  · refl
  split
  · refl
  · refl

end number_of_smaller_type_pages_l66_66312


namespace boys_to_girls_ratio_l66_66388

theorem boys_to_girls_ratio (x y : ℕ) 
  (h1 : 149 * x + 144 * y = 147 * (x + y)) : 
  x = (3 / 2 : ℚ) * y :=
by
  sorry

end boys_to_girls_ratio_l66_66388


namespace coefficient_of_x4_is_correct_l66_66763

noncomputable def coefficient_of_x4_in_expansion : Nat :=
  let x := 1 -- x is just symbolic
  let b := -3 * Real.sqrt 2
  let n := 8
  let k := 4
  -- binomial theorem coefficient
  let binom := Nat.choose n k
  -- compute b^k (note that because k = 4, (-3√2)^4 = (3√2)^4)
  let b_pow_k := (3 * Real.sqrt 2) ^ k
  
  binom * b_pow_k

theorem coefficient_of_x4_is_correct : coefficient_of_x4_in_expansion = 22680 :=
by
  unfold coefficient_of_x4_in_expansion
  rw [Nat.choose_eq_factorial_div_factorial, Nat.factorial, Nat.factorial, Nat.factorial, Nat.factorial, Real.pow_nat_eq, Real.pow_nat_eq, Real.mul_pow]
  -- calculations for binom coefficient and b^k can be detailed here
  -- sorry can be used here to skip the detailed proof outline if desired
  sorry

end coefficient_of_x4_is_correct_l66_66763


namespace fraction_division_l66_66038

-- Definition of fractions involved
def frac1 : ℚ := 4 / 9
def frac2 : ℚ := 5 / 8

-- Statement of the proof problem
theorem fraction_division :
  (frac1 / frac2) = 32 / 45 :=
by {
  sorry
}

end fraction_division_l66_66038


namespace find_x_for_parallel_vectors_l66_66567

def vector := (ℝ × ℝ)

def a (x : ℝ) : vector := (1, x)
def b (x : ℝ) : vector := (2, 2 - x)

def are_parallel (v w : vector) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem find_x_for_parallel_vectors :
  ∀ x : ℝ, are_parallel (a x) (b x) → x = 2/3 :=
by
  sorry

end find_x_for_parallel_vectors_l66_66567


namespace sum_odd_divisors_l66_66100

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l66_66100


namespace factorize_polynomial_l66_66891

theorem factorize_polynomial (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := 
by 
  sorry

end factorize_polynomial_l66_66891


namespace flight_time_is_59_hours_l66_66739

-- Define the conditions formally
def radius_of_earth : ℝ := 5000  -- miles
def jet_speed : ℝ := 550  -- miles per hour
def refueling_stop : ℝ := 2  -- hours

-- Define the circumference considering the conditions
def circumference_of_earth : ℝ := 2 * Real.pi * radius_of_earth

-- Define the total flight time including the stop
def total_flight_time : ℝ := ((circumference_of_earth / jet_speed) + refueling_stop)

-- State the theorem
theorem flight_time_is_59_hours : total_flight_time ≈ 59 := by
    -- We use ≈ to indicate approximation for this problem
    sorry

end flight_time_is_59_hours_l66_66739


namespace points_on_opposite_sides_l66_66992

theorem points_on_opposite_sides (a : ℝ) : ((0 + 0 < a) ∧ (a < 1 + 1)) → (a < 0) ∨ (a > 2) := 
by 
    intro h,
    cases h with h1 h2,
    sorry

end points_on_opposite_sides_l66_66992


namespace conference_handshakes_l66_66808

-- Define the number of attendees at the conference
def attendees : ℕ := 10

-- Define the number of ways to choose 2 people from the attendees
-- This is equivalent to the combination formula C(10, 2)
def handshakes (n : ℕ) : ℕ := n.choose 2

-- Prove that the number of handshakes at the conference is 45
theorem conference_handshakes : handshakes attendees = 45 := by
  sorry

end conference_handshakes_l66_66808


namespace product_of_two_numbers_l66_66016

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 21) (h2 : x^2 + y^2 = 527) : x * y = -43 :=
sorry

end product_of_two_numbers_l66_66016


namespace parabola_directrix_l66_66726

theorem parabola_directrix (p : ℝ) (h : y^2 = 2 * p * x)
  (focus : (2, 0) = (p / 2, 0)) : directrix = (λ x, x = -p / 2) → directrix = (λ x, x = -2) :=
by
  sorry

end parabola_directrix_l66_66726


namespace sum_of_positive_odd_divisors_of_90_l66_66077

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l66_66077


namespace projections_cyclic_l66_66002

variables {A B C D K L M N : Type*}
variable (α : Set Point) -- point set representing the plane α
variables [intersects_edgeA : ∃ K, α ∩ (A line_segment B) = K]
variables [intersects_edgeB : ∃ L, α ∩ (B line_segment C) = L]
variables [intersects_edgeC : ∃ M, α ∩ (C line_segment D) = M]
variables [intersects_edgeD : ∃ N, α ∩ (D line_segment A) = N]

variables [dihedral_angles_equal : ∀ P Q R S, 
  dihedral_angle (Plane PQ) (Plane PR) (Q R P) = 
  dihedral_angle (Plane QR) (Plane QS) (R S Q)]

theorem projections_cyclic : 
  ∀ (A' B' C' D' : Point), 
  project A α = A' → 
  project B α = B' → 
  project C α = C' → 
  project D α = D' → 
  is_cyclic_quad A' B' C' D' :=
sorry

end projections_cyclic_l66_66002


namespace election_total_votes_l66_66441

noncomputable def total_polled_votes (valid_votes : ℝ) (invalid_votes : ℝ) : ℝ :=
  valid_votes + invalid_votes

theorem election_total_votes (valid_votes : ℝ) (invalid_votes : ℝ) :
  valid_votes = 90000 →
  invalid_votes = 83 →
  total_polled_votes valid_votes invalid_votes = 90083 :=
by
  intros hvalid hinvalid
  unfold total_polled_votes
  rw [hvalid, hinvalid]
  norm_num
  sorry

end election_total_votes_l66_66441


namespace metallic_sheet_box_volume_l66_66826

theorem metallic_sheet_box_volume :
  ∃ L : ℝ, (L - 16) * (30 - 16) * 8 = 2688 ∧ L = 40 :=
begin
  use 40,
  split,
  { norm_num },
  { refl }
end

end metallic_sheet_box_volume_l66_66826


namespace volume_of_wall_l66_66393

-- Define the variables involved
variables (W H L V : ℝ)

-- Define the given conditions
def height_eq_6_width : Prop := H = 6 * W
def length_eq_7_height : Prop := L = 7 * H
def width_approx_7 : Prop := W = 6.999999999999999

-- Define the volume calculation
def volume_eq : Prop := V = W * H * L

-- State the theorem
theorem volume_of_wall : 
  height_eq_6_width → 
  length_eq_7_height → 
  width_approx_7 →
  volume_eq → 
  V = 86436 := 
by 
  intros
  sorry

end volume_of_wall_l66_66393


namespace goodColoringsOfPoints_l66_66653

noncomputable def countGoodColorings (k m : ℕ) : ℕ :=
  (k * (k - 1) + 2) * 2 ^ m

theorem goodColoringsOfPoints :
  countGoodColorings 2011 2011 = (2011 * 2010 + 2) * 2 ^ 2011 :=
  by
    sorry

end goodColoringsOfPoints_l66_66653


namespace thomas_water_needed_l66_66627

theorem thomas_water_needed :
  ∀ (nutrient_concentrate water prepared_solution water_fraction : ℝ),
  nutrient_concentrate = 0.05 →
  water = 0.03 →
  prepared_solution = 0.72 →
  water_fraction = water / (nutrient_concentrate + water) →
  (prepared_solution * water_fraction) = 0.27 :=
by {
  intros nutrient_concentrate water prepared_solution water_fraction,
  assume h1 : nutrient_concentrate = 0.05,
  assume h2 : water = 0.03,
  assume h3 : prepared_solution = 0.72,
  assume h4 : water_fraction = water / (nutrient_concentrate + water),
  sorry
}

end thomas_water_needed_l66_66627


namespace fraction_of_40_l66_66036

theorem fraction_of_40 : (3 / 4) * 40 = 30 :=
by
  -- We'll add the 'sorry' here to indicate that this is the proof part which is not required.
  sorry

end fraction_of_40_l66_66036


namespace tiling_induction_l66_66867

def L_shaped (p1 p2 p3 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ p2.2 = p3.2 ∧ p3.1 = p1.1 + 1 ∧ p2.1 = p3.1)
  ∨ (p1.2 = p2.2 ∧ p2.1 = p3.1 ∧ p3.2 = p1.2 + 1 ∧ p2.2 = p3.2) 
  ∨ (p1.1 = p2.1 ∧ p2.2 = p3.2 ∧ p3.1 = p2.1 - 1 ∧ p2.1 = p3.1)
  ∨ (p1.2 = p2.2 ∧ p2.1 = p3.1 ∧ p3.2 = p1.2 - 1 ∧ p2.2 = p3.2)

def tiling_possible (n : ℕ) (S : set (ℕ × ℕ)) : Prop :=
  ∀ (T : set (ℕ × ℕ)), T ⊆ S ⟹ T = empty ∨ 
  ∃ (p1 p2 p3 : ℕ × ℕ), L_shaped p1 p2 p3 ∧ T = {p1, p2, p3}

theorem tiling_induction (n : ℕ) (missing_cell : ℕ × ℕ) :
  tiling_possible (2^n) ((set.univ \ {missing_cell} : set (ℕ × ℕ))) :=
sorry

end tiling_induction_l66_66867


namespace smallest_y_for_fourth_power_l66_66151

noncomputable def x : ℕ := 5 * 27 * 64

theorem smallest_y_for_fourth_power (y : ℕ) (h : y = 4 * 3 * 125) : ∃ y, x * y = (k : ℕ) ^ 4 :=
begin
  use 1500,
  sorry
end

end smallest_y_for_fourth_power_l66_66151


namespace sum_of_odd_divisors_l66_66067

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l66_66067


namespace thirty_percent_less_than_ninety_eq_one_fourth_more_than_n_l66_66415

theorem thirty_percent_less_than_ninety_eq_one_fourth_more_than_n (n : ℝ) :
  0.7 * 90 = (5 / 4) * n → n = 50.4 :=
by sorry

end thirty_percent_less_than_ninety_eq_one_fourth_more_than_n_l66_66415


namespace proof_P_or_q_l66_66252

def P : Prop := 
  let y := λ x : ℝ, exp (-x)
  let tangent_eq_at_point := λ x y t => ∃ k, (y - exp (-t)) = k * (x + 1)
  ∃ k, tangent_eq_at_point (-1) (exp 1) (-1) ∧ k = -exp 1

def q : Prop := 
  ∀ (f : ℝ → ℝ) (x₀ : ℝ), (f' x₀ = 0) ↔ is_extreme_value_point f x₀

lemma P_is_true : P := sorry
lemma q_is_false : ¬ q := sorry

theorem proof_P_or_q : P ∨ q := by
  apply or.inl
  exact P_is_true

end proof_P_or_q_l66_66252


namespace squares_in_region_count_l66_66203

-- conditions
def is_inside_region (x y : ℕ) : Prop :=
  y ≥ 1 ∧ y ≤ 2 * x ∧ x ≤ 7

-- Translate proof problem into Lean 4 statement
theorem squares_in_region_count:
  (∑ x in finset.range 8, ∑ y in finset.range (min (2 * x + 1) 8), if is_inside_region x y then 1 else 0) +
  (∑ x in finset.range 8, ∑ y in finset.range (min (2 * (x - 1) + 1) 8), if x > 0 ∧ y > 0 ∧ is_inside_region x (y - 1) then 1 else 0) = 72 :=
sorry

end squares_in_region_count_l66_66203


namespace coefficient_of_x4_is_correct_l66_66765

noncomputable def coefficient_of_x4_in_expansion : Nat :=
  let x := 1 -- x is just symbolic
  let b := -3 * Real.sqrt 2
  let n := 8
  let k := 4
  -- binomial theorem coefficient
  let binom := Nat.choose n k
  -- compute b^k (note that because k = 4, (-3√2)^4 = (3√2)^4)
  let b_pow_k := (3 * Real.sqrt 2) ^ k
  
  binom * b_pow_k

theorem coefficient_of_x4_is_correct : coefficient_of_x4_in_expansion = 22680 :=
by
  unfold coefficient_of_x4_in_expansion
  rw [Nat.choose_eq_factorial_div_factorial, Nat.factorial, Nat.factorial, Nat.factorial, Nat.factorial, Real.pow_nat_eq, Real.pow_nat_eq, Real.mul_pow]
  -- calculations for binom coefficient and b^k can be detailed here
  -- sorry can be used here to skip the detailed proof outline if desired
  sorry

end coefficient_of_x4_is_correct_l66_66765


namespace max_value_expression_l66_66927

theorem max_value_expression (x y : ℝ) (hx : 0 < x) (hx' : x < (π / 2)) (hy : 0 < y) (hy' : y < (π / 2)) :
  (A : ℝ) = (Real.sqrt (Real.sqrt (Real.sin x * Real.sin y))) / (Real.sqrt (Real.sqrt (Real.tan x)) + Real.sqrt (Real.sqrt (Real.tan y))) ≤ Real.sqrt (Real.sqrt 8) / 4 :=
sorry

end max_value_expression_l66_66927


namespace largest_term_of_sequence_l66_66898

-- Define the sequence
def x (n : ℕ) : ℝ := (n - 1) / (n^2 + 1)

-- The main theorem stating the largest term of the sequence
theorem largest_term_of_sequence : ∃ n : ℕ, x n = 0.2 ∧ (∀ m: ℕ, x m ≤ 0.2) :=
by
  use 2
  use 3
  split
  { -- Showing that x 2 = 0.2
    show x 2 = 0.2,
    sorry
  }
  split
  { -- Showing that x 3 = 0.2
    show x 3 = 0.2,
    sorry
  }
  { -- Showing that for all n, x n ≤ 0.2
    show ∀ m : ℕ, x m ≤ 0.2,
    sorry
  }

end largest_term_of_sequence_l66_66898


namespace geometric_sequence_arithmetic_condition_l66_66910

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

-- Conditions of the problem
def is_geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

def positive_terms (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n n > 0

def arithmetic_sequence_cond (a_n : ℕ → ℝ) : Prop :=
  a_n 2 - (1 / 2) * a_n 3 = (1 / 2) * a_n 3 - a_n 1

-- Problem: Prove the required ratio equals the given value
theorem geometric_sequence_arithmetic_condition
  (h_geo: is_geometric_sequence a_n q)
  (h_pos: positive_terms a_n)
  (h_arith: arithmetic_sequence_cond a_n)
  (h_q_ne_one: q ≠ 1) :
  (a_n 4 + a_n 5) / (a_n 3 + a_n 4) = (1 + Real.sqrt 5) / 2 :=
sorry

end geometric_sequence_arithmetic_condition_l66_66910


namespace proposition_D_true_l66_66515

theorem proposition_D_true : 
  ∀ x : ℝ, x^2 + 1 ≠ 0 :=
by
  intro x
  have h1 : x^2 ≥ 0 := by sorry
  have h2 : x^2 + 1 ≥ 1 := by sorry
  have h3 : 1 ≠ 0 := by sorry
  exact h3

end proposition_D_true_l66_66515


namespace time_to_cover_length_l66_66789

/-- Constants -/
def speed_escalator : ℝ := 10
def length_escalator : ℝ := 112
def speed_person : ℝ := 4

/-- Proof problem -/
theorem time_to_cover_length :
  (length_escalator / (speed_escalator + speed_person)) = 8 := by
  sorry

end time_to_cover_length_l66_66789


namespace girls_attended_festival_l66_66973

variable (g b : ℕ)

theorem girls_attended_festival :
  g + b = 1500 ∧ (2 / 3) * g + (1 / 2) * b = 900 → (2 / 3) * g = 600 := by
  sorry

end girls_attended_festival_l66_66973


namespace select_elements_exactly_k_times_l66_66687

theorem select_elements_exactly_k_times (𝒮 : Finset α) (n k : ℕ) 
  (h_size : 𝒮.card = n) (h_pos : 0 < k) 
  (S : Finₓ (k * n) → Finset α) 
  (h_sub : ∀ i, (S i).card = 2) 
  (h_elem : ∀ e ∈ 𝒮, (Finₓ.filter (λ i, e ∈ S i)).card = 2 * k) :
  ∃ (f : Finₓ (k * n) → α), (∀ i, f i ∈ S i) ∧ (∀ e ∈ 𝒮, (Finₓ.filter (λ i, f i = e)).card = k) := by
  sorry

end select_elements_exactly_k_times_l66_66687


namespace OH_concentration_mixture_l66_66150

open_locale big_operators

theorem OH_concentration_mixture (V1 V2 : ℝ) (C1 C2 : ℝ) (C : ℝ) :
  V1 = 0.050 ∧ C1 = 0.200 ∧ V2 = 0.075 ∧ C2 = 0.100 →
  C = (C1 * V1 + C2 * V2) / (V1 + V2) →
  C = 0.140 :=
by
  intro h1 h2
  sorry

end OH_concentration_mixture_l66_66150


namespace kite_area_correct_l66_66909

-- Define the coordinates of the vertices
def A : (ℝ × ℝ) := (0, 6)
def B : (ℝ × ℝ) := (3, 10)
def C : (ℝ × ℝ) := (6, 6)
def D : (ℝ × ℝ) := (3, 0)

-- Distance between points on the grid
def grid_spacing : ℝ := 2

-- Definition of the area of the kite
def kite_area : ℝ :=
  let base := dist A C * grid_spacing in
  let height := dist D (𝕎) * grid_spacing in
  base * height / 2 * 2

-- Proof statement to show the area of the kite
theorem kite_area_correct : kite_area = 72 := by
  sorry

end kite_area_correct_l66_66909


namespace sin_double_alpha_cos_2alpha_plus_beta_l66_66933

-- Define the conditions
variables 
  (α β : ℝ)
  (h_cos_α : cos α = -3/4)
  (h_sin_β : sin β = 2/3)
  (h_α_quad : π < α ∧ α < 3 * π / 2)
  (h_β_range : π / 2 < β ∧ β < π)

-- Prove the required statements
theorem sin_double_alpha : sin (2 * α) = (3 * real.sqrt 7) / 8 :=
by sorry

theorem cos_2alpha_plus_beta : cos (2 * α + β) = -(real.sqrt 5 + 6 * real.sqrt 7) / 24 :=
by sorry

end sin_double_alpha_cos_2alpha_plus_beta_l66_66933


namespace max_f_l66_66724

noncomputable def is_point_on_ellipse (x y : ℝ) : Prop := 
  x^2 + y^2 / 4 = 1 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2

noncomputable def f (x y : ℝ) : ℝ :=
  -3 * x * y / (2 * real.sqrt (4 * x^2 + y^2 / 4))

theorem max_f (x y : ℝ) (h : is_point_on_ellipse x y) :
  ∃ P_max : ℝ × ℝ, 
    f P_max.1 P_max.2 = (sup (f '' {P : ℝ × ℝ | is_point_on_ellipse P.1 P.2})) ∧
    ∀ P, is_point_on_ellipse P.1 P.2 → f P.1 P.2 ≤ f P_max.1 P_max.2 ∧
    (f '' {P : ℝ × ℝ | is_point_on_ellipse P.1 P.2} = Icc (inf (f '' {P : ℝ × ℝ | is_point_on_ellipse P.1 P.2})) (sup (f '' {P : ℝ × ℝ | is_point_on_ellipse P.1 P.2}))) :=
sorry

end max_f_l66_66724


namespace sumOddDivisorsOf90_l66_66046

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l66_66046


namespace minimum_weighings_to_identify_counterfeit_weight_l66_66850

theorem minimum_weighings_to_identify_counterfeit_weight (n : ℕ) (h : n = 18) : 
  ∃ m : ℕ, m = 2 ∧ ∀ (coins : fin n → ℝ), 
    (∃ i, (∀ j : fin n, j ≠ i → coins j = 1) ∧ (coins i ≠ 1)) → 
    (∃ weighs : fin m → (fin n → fin n), 
      ∃ scale : fin n → (fin m → bool),
        ∀ (coin_index : fin n),
          (∀ j : fin n, j ≠ coin_index → coins j = 1) →
          ((∃ balance : fin m → option bool, 
            (∀ i : fin m, 
              match balance i with
              | some b => (scale coin_index i = b)
              | none   => true
              end) ∧
            (∀ i : fin m, (balance i = some true ∨ balance i = some false ↔ weighs i = some coin_index)))

noncomputable def weighing_strategy : Type :=
  sorry

end minimum_weighings_to_identify_counterfeit_weight_l66_66850


namespace general_term_aequence_can_form_geometric_sequence_l66_66247

-- 1. Prove the general term formula of the sequence {a_n}
theorem general_term_aequence (d : ℕ) (n : ℕ) (d_ne_zero : d ≠ 0) (a₁_eq_one : a 1 = 1) (Sn : ℕ → ℕ) (S_seq_arith : arithmetic_seq (λ n, (Sn n) / (a n))) : 
  ∀ n, a n = n := 
sorry

-- 2. Determine if b₁, bₖ, and bₘ form a geometric sequence if k = 2 and m = 3.
theorem can_form_geometric_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) (lg_def : ∀ n, b n = a n / (3^n)) :
  ∃ k m, k = 2 ∧ m = 3 ∧ geometric_seq {b 1, b k, b m} := 
sorry

end general_term_aequence_can_form_geometric_sequence_l66_66247


namespace three_digit_divisible_by_7_l66_66976

theorem three_digit_divisible_by_7 : ∃ n, ∀ k, 100 ≤ k ∧ k ≤ 999 → 7 ∣ k ↔ k = 128 :=
by
  sorry

end three_digit_divisible_by_7_l66_66976


namespace square_floor_area_l66_66473

theorem square_floor_area 
  (s : ℝ)
  (h1 : ∃ (f : ℝ), 2 * 7 = 14 ∧ s^2 = f ∧ 0.78125 * f = 0.78125 * (s ^ 2))
  (h2 : 0.21875 * s^2 = 14) : 
  s^2 = 64 := 
by 
  apply eq_of_mul_eq_mul_left (ne_of_gt (show 0.21875 > 0, by norm_num)),
  exact h2

end square_floor_area_l66_66473


namespace candidate_percentage_l66_66456

variables (P candidate_votes rival_votes total_votes : ℝ)

-- Conditions
def candidate_lost_by_2460 (candidate_votes rival_votes : ℝ) : Prop :=
  rival_votes = candidate_votes + 2460

def total_votes_cast (candidate_votes rival_votes total_votes : ℝ) : Prop :=
  candidate_votes + rival_votes = total_votes

-- Proof problem
theorem candidate_percentage (h1 : candidate_lost_by_2460 candidate_votes rival_votes)
                             (h2 : total_votes_cast candidate_votes rival_votes 8200) :
  P = 35 :=
sorry

end candidate_percentage_l66_66456


namespace steamers_meet_l66_66219

/--
   Every day at noon, a mail steamer leaves from Le Havre to New York, and at the same time, 
   another steamer leaves New York for Le Havre. Each of these steamers takes exactly 
   seven days to complete their journey, and they travel the same route. 
 -/

/-- A steamer traveling from Le Havre to New York will meet exactly 15 steamers from the 
same company traveling in the opposite direction. -/
theorem steamers_meet (departures_every_day : ∀ n : ℕ, ∃ l : ℕ, 0 ≤ l ∧ l < 7)
  (journey_duration : ∀ ship : ℕ, ship ∈ departures_every_day ship → ship % 7 = 0) :
  ∀ P : ℕ, P ∈ departures_every_day P → 
  15 = (let current_day := P % 7 in 
        let ships_meeting := 7 + 1 + 7 in 
        ships_meeting) := 
by
  sorry

end steamers_meet_l66_66219


namespace find_CM_l66_66819

variables {R : ℝ} {O A B C M : Point}
variables (α β : ℝ)
variables [circle : Circle O R]

-- Conditions
axiom passes_through_A_and_B: circle.on_circumference A ∧ circle.on_circumference B
axiom intersects_BC_at_M: M ∈ segment B C ∧ circle.on_circumference M
axiom tangent_to_AC_at_A: is_tangent circle (line_through A C) A
axiom angle_ACO: \∠ A C O = α
axiom angle_MAB: \∠ M A B = β

-- Conclusion
theorem find_CM :
  CM = R * (sqrt (sin β ^ 2 + cot α ^ 2) - sin β) :=
sorry

end find_CM_l66_66819


namespace sum_sequence_formula_l66_66449

-- Define the sequence terms as a function.
def seq_term (x a : ℕ) (n : ℕ) : ℕ :=
x ^ (n + 1) + (n + 1) * a

-- Define the sum of the first nine terms of the sequence.
def sum_first_nine_terms (x a : ℕ) : ℕ :=
(x * (x ^ 9 - 1)) / (x - 1) + 45 * a

-- State the theorem to prove that the sum S is as expected.
theorem sum_sequence_formula (x a : ℕ) (h : x ≠ 1) : 
  sum_first_nine_terms x a = (x ^ 10 - x) / (x - 1) + 45 * a := by
  sorry

end sum_sequence_formula_l66_66449


namespace tank_empty_time_l66_66190

theorem tank_empty_time (R L : ℝ) (h1 : R = 1 / 7) (h2 : R - L = 1 / 8) : 
  (1 / L) = 56 :=
by
  sorry

end tank_empty_time_l66_66190


namespace divide_point_in_ratio_l66_66533

theorem divide_point_in_ratio (A B: ℝ × ℝ) (m n: ℝ) (Hx: A = (2, 10)) (Hy: B = (8, 4)) (Hr: m = 1) (Hn: n = 3) :
  let P : ℝ × ℝ := ((m * B.1 + n * A.1) / (m + n), (m * B.2 + n * A.2) / (m + n)) in
  P = (3.5, 8.5) :=
by
  cases A with x1 y1
  cases B with x2 y2
  simp [Hx, Hy, Hr, Hn]
  sorry

end divide_point_in_ratio_l66_66533


namespace fans_receive_all_three_items_l66_66231

theorem fans_receive_all_three_items :
  ∑ fan_idx in (finset.range 4800).filter (λ n, n > 0 ∧ n % 45 = 0 ∧ n % 36 = 0 ∧ n % 60 = 0), 1 = 26 :=
by
  sorry

end fans_receive_all_three_items_l66_66231


namespace max_tuesdays_in_36_days_l66_66206

theorem max_tuesdays_in_36_days (days : ℕ) (weeks : ℕ) (extra_days : ℕ) (h1 : days = 36) (h2 : weeks = days / 7) (h3 : extra_days = days % 7) : 
  by 
    have weeks := 5
    have extra_days := 1
    have max_tuesdays := if extra_days = 0 then weeks else weeks + 1
    exact max_tuesdays = 6
  sorry

end max_tuesdays_in_36_days_l66_66206


namespace arrange_books_l66_66371

theorem arrange_books :
  let total_books := 9,
      arabic_books := 2,
      german_books := 3,
      spanish_books := 4,
      books := arabic_books + german_books + spanish_books,
      group_arabic := 1,
      group_spanish := 1,
      book_groups := group_arabic + german_books + group_spanish,
      arrangement_groups := book_groups.factorial,
      arrange_arabic := arabic_books.factorial,
      arrange_spanish := spanish_books.factorial in
  books = total_books →
  group_arabic = 1 ∧ group_spanish = 1 →
  (arrangement_groups * arrange_arabic * arrange_spanish = 5760) :=
begin
  intros,
  exact sorry
end

end arrange_books_l66_66371


namespace rancher_solution_l66_66472

theorem rancher_solution :
  ∃ s c : ℕ, 30 * s + 40 * c = 800 ∧ s ≥ 10 ∧ c ≥ s / 2 ∧ s = 12 ∧ c = 11 :=
by
  use 12, 11
  split
  -- proof would go here but we provide a placeholder sorry.
  sorry

end rancher_solution_l66_66472


namespace Suma_can_complete_in_6_days_l66_66703

-- Define the rates for Renu and their combined rate
def Renu_rate := (1 : ℚ) / 6
def Combined_rate := (1 : ℚ) / 3

-- Define Suma's time to complete the work alone
def Suma_days := 6

-- defining the work rate Suma is required to achieve given the known rates and combined rate
def Suma_rate := Combined_rate - Renu_rate

-- Require to prove 
theorem Suma_can_complete_in_6_days : (1 / Suma_rate) = Suma_days :=
by
  -- Using the definitions provided and some basic algebra to prove the theorem 
  sorry

end Suma_can_complete_in_6_days_l66_66703


namespace find_angle_A_find_area_l66_66279

-- Define the conditions for the triangle
variable {a b c : ℝ}
variable {A B C : ℝ}

-- Given conditions from the problem
def triangle_conditions (a b c A B C: ℝ) : Prop :=
  b = a * (Real.cos C - Real.sin C) ∧ a = Real.sqrt 10 ∧ Real.sin B = Real.sqrt 2 * Real.sin C

-- Problem 1: Prove that A = 3π/4
theorem find_angle_A (h : triangle_conditions a b c A B C) : A = 3 * Real.pi / 4 :=
sorry

-- Problem 2: Prove that the area of triangle ABC is 1
theorem find_area (h : triangle_conditions a b c A B C) : Area of triangle ABC = 1 :=
sorry

end find_angle_A_find_area_l66_66279


namespace simple_interest_rate_l66_66994

theorem simple_interest_rate (P : ℝ) (R : ℝ) (SI : ℝ) (T : ℝ) (h1 : T = 4) (h2 : SI = P / 5) (h3 : SI = (P * R * T) / 100) : R = 5 := by
  sorry

end simple_interest_rate_l66_66994


namespace last_digit_35_exp_l66_66494

theorem last_digit_35_exp : ∀ (n : ℕ), (n > 0) → (nat.digits 10 (35 ^ 18 * 13 ^ 33)).head = 5 :=
by {
  intros n hn,
  sorry
}

end last_digit_35_exp_l66_66494


namespace triangle_transform_correct_l66_66484

open Function

def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def reflectX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def rotate270cw (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  let p1 := rotate180 p
  let p2 := reflectX p1
  rotate270cw p2

noncomputable def resulting_vertices : list (ℝ × ℝ) :=
  [transform (1, -2), transform (-1, -2), transform (1, 1)]

theorem triangle_transform_correct : 
  resulting_vertices = [(2, 1), (2, -1), (-1, -1)] :=
by
  sorry

end triangle_transform_correct_l66_66484


namespace volume_of_new_pyramid_correct_l66_66475

noncomputable def volume_of_new_pyramid
  (base_edge : ℝ)
  (slant_edge : ℝ)
  (distance_from_base : ℝ) : ℝ :=
let half_diagonal := (base_edge / 2) * Real.sqrt 2 in
let original_height := Real.sqrt (slant_edge^2 - half_diagonal^2) in
let remaining_height := original_height - distance_from_base in
let similarity_ratio := remaining_height / original_height in
let new_base_edge := base_edge * similarity_ratio in
let new_base_area := new_base_edge^2 in
(1 / 3) * new_base_area * remaining_height

theorem volume_of_new_pyramid_correct :
  volume_of_new_pyramid 12 15 5 = 48 * ((Real.sqrt 153 - 5) / Real.sqrt 153)^3 * (Real.sqrt 153 - 5) :=
by
  sorry

end volume_of_new_pyramid_correct_l66_66475


namespace coefficient_of_x4_in_expansion_l66_66762

theorem coefficient_of_x4_in_expansion :
  let f := λ (x : ℝ), (x - 3 * Real.sqrt 2) ^ 8
  ∃ c : ℝ, c * x^4 = (f x) → c = 22680 :=
begin
  intro f,
  use (λ (x : ℝ), (x - 3 * Real.sqrt 2) ^ 8),
  intro x,
  sorry
end

end coefficient_of_x4_in_expansion_l66_66762


namespace combined_score_210_l66_66622

-- Define the constants and variables
def total_questions : ℕ := 50
def marks_per_question : ℕ := 2
def jose_wrong_questions : ℕ := 5
def jose_extra_marks (alisson_score : ℕ) : ℕ := 40
def meghan_less_marks (jose_score : ℕ) : ℕ := 20

-- Define the total possible marks
def total_possible_marks : ℕ := total_questions * marks_per_question

-- Given the conditions, we need to prove the total combined score is 210
theorem combined_score_210 : 
  ∃ (jose_score meghan_score alisson_score combined_score : ℕ), 
  jose_score = total_possible_marks - (jose_wrong_questions * marks_per_question) ∧
  meghan_score = jose_score - meghan_less_marks jose_score ∧
  alisson_score = jose_score - jose_extra_marks alisson_score ∧
  combined_score = jose_score + meghan_score + alisson_score ∧
  combined_score = 210 := by
  sorry

end combined_score_210_l66_66622


namespace number_of_valid_tuples_l66_66679

def number_of_tuples (p : ℕ) [fact (odd p) (prime p)] : ℕ := p^(p-2) * (p-1)

theorem number_of_valid_tuples (p : ℕ) [fact (odd p)] [fact (nat.prime p)] :
  ∃ t : fin p -> fin p,
  (∀ i, 1 ≤ t i ∧ t i ≤ p) ∧
  (∑ i in finset.fin_range p, t i ≠ 0 [MOD p]) ∧
  (∑ i in finset.fin_range p, (t i) * (t (i+1 % p)) = 0 [MOD p]) :=
  sorry

end number_of_valid_tuples_l66_66679


namespace merchant_salt_mixture_l66_66466

theorem merchant_salt_mixture (x : ℝ) (h₀ : (0.48 * (40 + x)) = 1.20 * (14 + 0.50 * x)) : x = 0 :=
by
  sorry

end merchant_salt_mixture_l66_66466


namespace budget_spent_on_utilities_l66_66817

noncomputable def budget_is_correct : Prop :=
  let total_budget := 100
  let salaries := 60
  let r_and_d := 9
  let equipment := 4
  let supplies := 2
  let degrees_in_circle := 360
  let transportation_degrees := 72
  let transportation_percentage := (transportation_degrees * total_budget) / degrees_in_circle
  let known_percentages := salaries + r_and_d + equipment + supplies + transportation_percentage
  let utilities_percentage := total_budget - known_percentages
  utilities_percentage = 5

theorem budget_spent_on_utilities : budget_is_correct :=
  sorry

end budget_spent_on_utilities_l66_66817


namespace abc_sum_l66_66713

theorem abc_sum (f : ℝ → ℝ) (a b c : ℝ) :
  f (x - 2) = 2 * x^2 - 5 * x + 3 → f x = a * x^2 + b * x + c → a + b + c = 6 :=
by
  intros h₁ h₂
  sorry

end abc_sum_l66_66713


namespace price_of_turban_is_2_5_l66_66784

noncomputable def price_of_turban (annual_salary : ℝ) (months_worked : ℝ) (cash_given : ℝ) : ℝ :=
  let salary_should_be := (months_worked / 12) * annual_salary
  salary_should_be - cash_given

theorem price_of_turban_is_2_5 :
  price_of_turban 90 9 65 = 2.5 :=
by
  let annual_salary := 90
  let months_worked := 9
  let cash_given := 65
  let expected_turban_price := 2.5
  unfold price_of_turban
  have h : (months_worked / 12) * annual_salary - cash_given = expected_turban_price := by norm_num
  exact h

end price_of_turban_is_2_5_l66_66784


namespace max_value_of_expression_l66_66928

theorem max_value_of_expression (x y : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) (hy1 : 0 < y) (hy2 : y < π / 2) :
  ∃ (A : ℝ), A = \frac{\sqrt[4]{\sin x \sin y}}{\sqrt[4]{\tan x} + \sqrt[4]{\tan y}} ∧ 
  ∀ (x y : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) (hy1 : 0 < y) (hy2 : y < π / 2), 
    A ≤ \frac{\sqrt[4]{8}}{4} :=
sorry

end max_value_of_expression_l66_66928


namespace faye_initial_books_l66_66527

theorem faye_initial_books (X : ℕ) (h : (X - 3) + 48 = 79) : X = 34 :=
sorry

end faye_initial_books_l66_66527


namespace distinct_pairs_count_l66_66885

theorem distinct_pairs_count :
  ∃! (s : Finset (ℕ × ℕ)),
    (∀ (p ∈ s), p.1 < p.2 ∧ 0 < p.1 ∧ 54 = Nat.sqrt 2916 ∧ 54 = Nat.sqrt p.1 + Nat.sqrt p.2) ∧
    s.card = 4 :=
sorry

end distinct_pairs_count_l66_66885


namespace determine_c_l66_66270

-- Definition of the function f(x)
def f (x c : ℝ) : ℝ := x * (x - c)^2

-- The condition that f(x) has a maximum at x = 2
def has_max_at (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x ≤ f a

theorem determine_c (c : ℝ) :
  has_max_at (λ x, f x c) 2 → c = 6 :=
by
  intros h
  have : deriv (λ x, f x c) 2 = 0 := sorry
  have : deriv (λ x, f x c) 2 = 0 := sorry
  -- solve for c using the derivative = 0 condition
  have : (2-c)^2 + 4*(2-c) = 0 := sorry
  -- use the above equation to determine c
  -- check which value of c causes f(x) to have a maximum at x = 2
  -- complete the proof
  sorry

end determine_c_l66_66270


namespace find_subsequence_with_sum_n_l66_66521

theorem find_subsequence_with_sum_n (n : ℕ) (a : Fin n → ℕ) (h1 : ∀ i, a i ∈ Finset.range n) 
  (h2 : (Finset.univ.sum a) < 2 * n) : 
  ∃ s : Finset (Fin n), s.sum a = n := 
sorry

end find_subsequence_with_sum_n_l66_66521


namespace valid_outfits_count_l66_66506

structure Wardrobe :=
  (red_shirts : ℕ)
  (green_shirts : ℕ)
  (blue_shirts : ℕ)
  (pants : ℕ)
  (green_hats : ℕ)
  (red_hats : ℕ)
  (blue_hats : ℕ)
  (green_ties : ℕ)
  (red_ties : ℕ)
  (blue_ties : ℕ)
  (distinct_items : ∀ (x y : String), x ≠ y)

def count_valid_outfits (w : Wardrobe) : ℕ :=
  let case1 := (w.red_shirts + w.blue_shirts) * w.pants * w.green_hats * w.green_ties in
  let case2 := (w.green_shirts + w.blue_shirts) * w.pants * w.red_hats * w.red_ties in
  let case3 := (w.red_shirts + w.green_shirts) * w.pants * w.blue_hats * w.blue_ties in
  case1 + case2 + case3

theorem valid_outfits_count {w : Wardrobe}
  (h_red_shirts : w.red_shirts = 6)
  (h_green_shirts : w.green_shirts = 7)
  (h_blue_shirts : w.blue_shirts = 8)
  (h_pants : w.pants = 9)
  (h_green_hats : w.green_hats = 10)
  (h_red_hats : w.red_hats = 10)
  (h_blue_hats : w.blue_hats = 10)
  (h_green_ties : w.green_ties = 5)
  (h_red_ties : w.red_ties = 5)
  (h_blue_ties : w.blue_ties = 5)
  (h_distinct : ∀ (x y : String), x ≠ y) : 
  count_valid_outfits w = 18900 :=
by {
  unfold count_valid_outfits,
  rw [h_red_shirts, h_green_shirts, h_blue_shirts, h_pants, h_green_hats, h_red_hats, h_blue_hats, h_green_ties, h_red_ties, h_blue_ties],
  norm_num
}

end valid_outfits_count_l66_66506


namespace distance_to_Big_Rock_l66_66836

theorem distance_to_Big_Rock : 
  ∀ (D : ℝ), 
  (∀ (R_s1 R_s2 C_s1 C_s2 T: ℝ),
  R_s1 = 6 →
  R_s2 = 7 → 
  C_s1 = 2 →
  C_s2 = 3 → 
  T = 2 →
  ( (D / (R_s1 - C_s1)) + (D / (R_s2 - C_s2)) = T ) →
  D = 4) :=
begin
  intros D R_s1 R_s2 C_s1 C_s2 T,
  assume (h1 : R_s1 = 6) (h2 : R_s2 = 7) (h3 : C_s1 = 2) (h4 : C_s2 = 3) (h5 : T = 2),
  assume h6, sorry
end

end distance_to_Big_Rock_l66_66836


namespace diagonal_AC_eq_10_l66_66508

-- Define the rectangle and its properties
structure Rectangle (A B C D : Type) :=
  (AB BC : ℝ)
  (sides_eq : AB = 8 ∧ BC = 6)

-- Define the diagonal computation using the Pythagorean theorem
def diagonal_length (AB BC : ℝ) : ℝ :=
  Real.sqrt (AB^2 + BC^2)

-- The proof problem to state that the length of diagonal AC is 10
theorem diagonal_AC_eq_10 {A B C D : Type} (rect : Rectangle A B C D) : 
  diagonal_length rect.AB rect.BC = 10 :=
by
  -- Extract conditions from the rectangle properties
  rcases rect with ⟨AB, BC, sides_eq⟩,
  cases sides_eq with AB_eq BC_eq,
  rw [AB_eq, BC_eq],
  dsimp [diagonal_length],
  norm_num
-- sorry, we placeholder with sorry to meet code build requirement
-- sorry

end diagonal_AC_eq_10_l66_66508


namespace cos_B_given_sin_ratios_in_triangle_l66_66661

theorem cos_B_given_sin_ratios_in_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : sin A / a = sin B / b)
  (h2 : sin B / b = sin C / c)
  (h3 : sin A / sin B = 4 / 3)
  (h4 : sin B / sin C = 3 / 2)
  (h5 : a = 4 * x)
  (h6 : b = 3 * x)
  (h7 : c = 2 * x)
  (h8 : 0 < x)
  : cos B = 11 / 16 := sorry

end cos_B_given_sin_ratios_in_triangle_l66_66661


namespace volume_MLKC_l66_66673

variables {A B C D M L K : Point}
variable {V : ℝ}

-- Assume M is the intersection of the medians of tetrahedron ABCD
def is_median_intersection (M A B C D : Point) :=
  vector.sum [vector.from M A, vector.from M B, vector.from M C, vector.from M D] = vector.zero

-- Assume the volume of tetrahedron ABCD is V
variable (volume_ABCD : volume (A, B, C, D) = V)

-- Define the required Lean 4 theorem statement
theorem volume_MLKC (h : is_median_intersection M A B C D) :
  volume (M, L, K, C) = V / 4 :=
sorry

end volume_MLKC_l66_66673


namespace nine_dim_hypercube_has_five_dim_faces_l66_66130

theorem nine_dim_hypercube_has_five_dim_faces : 
  let num_faces : ℕ := binomial 9 5 * 2^4 
  in num_faces = 2016 :=
by
  sorry

end nine_dim_hypercube_has_five_dim_faces_l66_66130


namespace max_knights_l66_66385

theorem max_knights (people : Fin 10 → Bool) (tokens : Fin 10 → Nat) 
  (statement : Fin 10 → Bool) :
  -- Conditions
  (∀ i : Fin 10, tokens i = 1 ∧ ∃ j : Fin 10, j ≠ i ∧ (tokens j = 1 ∨ tokens j = 0 ∨ tokens j = 2)) → 
  (∀ i : Fin 10, tokens i' = if people i then 1 else 0) →
  (Sum (fun i : Fin 10 => if statement i then tokens i else 0) = 10) →
  (Sum (fun i : Fin 10 => if statement i then 1 else 0) = 5) →
  (Sum (fun i : Fin 10 => tokens i) = 10 )→
  (∀ i, (people i = tt → statement i = (tokens i = 1)) ∧ (people i = ff → statement i = ¬ (tokens i = 1))) →
  -- Conclusion
  (count (Fin 10) people) ≤ 7 :=
sorry

end max_knights_l66_66385


namespace coeff_x4_expansion_l66_66768

theorem coeff_x4_expansion : coeff (expand (x - 3 * real.sqrt 2) 8) 4 = 22680 := by
  sorry

end coeff_x4_expansion_l66_66768


namespace total_employees_345_l66_66635

theorem total_employees_345 :
  ∀ (E U P UP : ℕ) (p prob : ℚ),
  U = 104 → P = 54 → p = 0.125 → prob = 0.5797101449275363 →
  UP = p * U →
  E = (U - UP) + (P - UP) + UP + prob * E →
  E = 345 :=
by
  intros E U P UP p prob hU hP hp hprob hUP hE
  unfold Nat at *
  unfold Int at *
  sorry

end total_employees_345_l66_66635


namespace unit_digit_7_pow_2006_l66_66361

theorem unit_digit_7_pow_2006 : (7 ^ 2006) % 10 = 9 := 
by 
  have pattern := [7, 9, 3, 1] 
  have h : 2006 % 4 = 2 := nat.mod_eq_of_lt (nat.div_eq_to_mod_eq.mp rfl).2
  exact list.nth_le pattern 2 sorry

end unit_digit_7_pow_2006_l66_66361


namespace eccentricity_of_ellipse_l66_66723

-- Define the constants based on the given ellipse equation parameters
def a : ℝ := 3
def b : ℝ := Real.sqrt 5
def c : ℝ := Real.sqrt (a^2 - b^2)
def e : ℝ := c / a

-- Statement of the theorem
theorem eccentricity_of_ellipse : ∀ (x y : ℝ), (x^2 / 9 + y^2 / 5 = 1) → e = 2 / 3 :=
by
  intros x y h
  rw [e] -- use the definition of e
  rw [c] -- use the definition of c
  have h1 : a = 3 := rfl
  have h2 : b = Real.sqrt 5 := rfl
  have h3 : c = Real.sqrt (a^2 - b^2) := rfl
  have h4 : c = Real.sqrt (9 - 5) := by simp [h1, h2]
  have h5 : c = 2 := by simp [h4]
  have h6 : e = 2 / 3 := by simp [h5, h1]
  exact h6

end eccentricity_of_ellipse_l66_66723


namespace least_positive_integer_divisible_by_primes_l66_66769

theorem least_positive_integer_divisible_by_primes :
  let primes : List ℕ := [11, 13, 17]
  in 2431 = primes.foldl (λ acc x, acc * x) 1 := 
by
  sorry

end least_positive_integer_divisible_by_primes_l66_66769


namespace cole_average_speed_l66_66193

theorem cole_average_speed (v2 : ℝ) (T : ℝ) (T1 : ℝ) : 
  v2 = 110 → 
  T = 2 → 
  T1 = 82.5 / 60 → 
  let T2 := T - T1 in 
  let distance := v2 * T2 in 
  let v1 := distance / T1 in 
  v1 = 50 :=
by 
  intros h_v2 h_T h_T1
  let T2 := T - T1
  let distance := v2 * T2
  let v1 := distance / T1
  rw [h_v2, h_T, h_T1] at *
  sorry

end cole_average_speed_l66_66193


namespace sum_of_odd_divisors_of_90_l66_66089

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l66_66089


namespace distance_from_point_to_plane_l66_66486

theorem distance_from_point_to_plane
  (A B C D : Point)
  (h_AB : dist A B = 1)
  (h_AC : dist A C = 1)
  (h_AD : dist A D = 1)
  (h_BC : dist B C = 1)
  (h_BD : dist B D = 1)
  (h_CD : dist C D = 1) :
  distance_from_point_to_plane D (plane_of_points A B C) = real.sqrt (2 / 3) := 
sorry 

end distance_from_point_to_plane_l66_66486


namespace games_played_by_E_l66_66230

variable (A B C D E : Type)
variable (graph : A → A → Prop)

-- Conditions
axiom A_has_played_four_games : ∃ l : list A, l.length = 4 ∧ (∀ x ∈ l, graph A x)
axiom B_has_played_three_games : ∃ l : list A, l.length = 3 ∧ (∀ x ∈ l, graph B x)
axiom C_has_played_two_games : ∃ l : list A, l.length = 2 ∧ (∀ x ∈ l, graph C x)
axiom D_has_played_one_game : ∃ l : list A, l.length = 1 ∧ (∀ x ∈ l, graph D x)

theorem games_played_by_E : ∃ l : list A, l.length = 2 ∧ (∀ x ∈ l, graph E x) :=
sorry

end games_played_by_E_l66_66230


namespace ben_marble_count_l66_66182

theorem ben_marble_count :
  ∃ k : ℕ, 5 * 2^k > 200 ∧ ∀ m < k, 5 * 2^m ≤ 200 :=
sorry

end ben_marble_count_l66_66182


namespace circles_divide_plane_l66_66633

theorem circles_divide_plane (n : ℕ) (f : ℕ → ℕ) 
(h1 : f 1 = 2) 
(h2 : f 2 = 4) 
(h3 : f 3 = 8) 
(h_intersects_at_two_points : ∀ (c1 c2 : ℕ), c1 ≠ c2 → intersection_points c1 c2 = 2)
(h_no_three_common_points : ∀ (c1 c2 c3 : ℕ), intersection_points c1 c2 ≥ 1 → intersection_points c2 c3 ≥ 1 → intersection_points c1 c3 ≥ 1 → common_point c1 c2 c3 = false)
: f n = n^2 - n + 2 := 
sorry

end circles_divide_plane_l66_66633


namespace pow_half_inequality_l66_66919

-- Definitions based on the conditions
variable {x y : ℝ}

-- Hypotheses from the conditions
variable (hx : x > y)
variable (hy : y < 0)

-- The proof statement
theorem pow_half_inequality (hx : x > y) (hy : y < 0) : 
  (1 / 2) ^ x - (1 / 2) ^ y < 0 := 
sorry

end pow_half_inequality_l66_66919


namespace imaginary_unit_root_l66_66990

theorem imaginary_unit_root (a b : ℝ) (h : (Complex.I : ℂ) ^ 2 + a * Complex.I + b = 0) : a + b = 1 := by
  -- Since this is just the statement, we add a sorry to focus on the structure
  sorry

end imaginary_unit_root_l66_66990


namespace smallest_whole_number_larger_than_sum_l66_66903

noncomputable def mixed_number1 : ℚ := 3 + 2/3
noncomputable def mixed_number2 : ℚ := 4 + 1/4
noncomputable def mixed_number3 : ℚ := 5 + 1/5
noncomputable def mixed_number4 : ℚ := 6 + 1/6
noncomputable def mixed_number5 : ℚ := 7 + 1/7

noncomputable def sum_of_mixed_numbers : ℚ :=
  mixed_number1 + mixed_number2 + mixed_number3 + mixed_number4 + mixed_number5

theorem smallest_whole_number_larger_than_sum : 
  ∃ n : ℤ, (n : ℚ) > sum_of_mixed_numbers ∧ n = 27 :=
by
  sorry

end smallest_whole_number_larger_than_sum_l66_66903


namespace evaluate_expression_l66_66217

theorem evaluate_expression : (∃ (a b : ℕ), a = 33 + 12 ∧ b = 12^2 + 33^2 ∧ (a^2 - b = 792)) :=
begin
  use [33 + 12, 12^2 + 33^2],
  split,
  { refl, },
  split,
  { refl, },
  calc
    (33 + 12)^2 - (12^2 + 33^2)
        = 45^2 - 1233 : by congr; assumption
    ... = 2025 - 1233 : by norm_num
    ... = 792 : by norm_num,
end


end evaluate_expression_l66_66217


namespace sum_of_positive_odd_divisors_of_90_l66_66073

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l66_66073


namespace slope_tangent_at_pi_l66_66408

def curve(x : ℝ) : ℝ := 2 * x + Real.sin x

theorem slope_tangent_at_pi : 
  (Real.deriv curve π) = 1 := 
by 
  sorry

end slope_tangent_at_pi_l66_66408


namespace card_T_bound_l66_66799

theorem card_T_bound 
  (n : ℕ) 
  (T : Finset ℕ) 
  (T_sub : ∀ t ∈ T, t ∈ Finset.range (n + 1)) 
  (cond : ∀ i j ∈ T, i ≠ j → ¬ (2 * j) % i = 0) : 
  |T| ≤ ⌊ (4 / 9) * n + real.log2 n + 2 ⌋ :=
by
  sorry

end card_T_bound_l66_66799


namespace log_simplify_exp_simplify_l66_66524

theorem log_simplify : log 3 63 - 2 * log 3 (sqrt 7) = 2 :=
by sorry

theorem exp_simplify (a : ℝ) : 3 * a ^ 5 * 3 * a ^ 7 / a ^ 6 = 1 / (a ^ 2) :=
by sorry

end log_simplify_exp_simplify_l66_66524


namespace log_a_b_leq_one_l66_66558

variable (x : ℝ)

def a := 2^x
def b := 4^(2/3)

theorem log_a_b_leq_one (h : log a b ≤ 1) : x < 0 ∨ x ≥ 4/3 := sorry

end log_a_b_leq_one_l66_66558


namespace original_average_weight_l66_66414

-- Definitions from conditions
def original_team_size : ℕ := 7
def new_player1_weight : ℝ := 110
def new_player2_weight : ℝ := 60
def new_team_size := original_team_size + 2
def new_average_weight : ℝ := 106

-- Statement to prove
theorem original_average_weight (W : ℝ) :
  (7 * W + 110 + 60 = 9 * 106) → W = 112 := by
  sorry

end original_average_weight_l66_66414


namespace cot_arctan_combination_l66_66907

theorem cot_arctan_combination : 
  (Real.cot (Real.arctan 4 + Real.arctan 9 + Real.arctan 17 + Real.arctan 31)) = 8893 / 4259 := 
by
  sorry

end cot_arctan_combination_l66_66907


namespace general_formula_and_sum_sum_bn_sequence_l66_66244

-- Define the geometric sequence {a_n}
def a (n : ℕ) : ℕ := 2^(n-1)

-- Define the sequence {b_n = log2(a_{n+1})}
def b (n : ℕ) : ℕ := n

-- Define the sequence {1 / (b_n * b_{n+1})}
def d (n : ℕ) : ℚ := 1 / (b n * b (n + 1))

-- General formula proof
theorem general_formula_and_sum (n : ℕ) (hn : 0 < n) :
  let a₁ := 1
      a₂ := 2
      a₃ := 2 * 2
      q  := 2
  in a 5 = 16 ∧ a 1 * a 2 * a 3 = 8 ∧ (a n = 2^(n-1)) ∧
     (∑ i in finset.range n, a (i + 1) = 2^n - 1) :=
by
  simp only [a]
  split
  . trivial
  . apply nat.pow_eq
  . trivial
  . sorry

-- Sum of the sequence {1 / (b_n * b_{n+1})}
theorem sum_bn_sequence (n : ℕ) (hn : 0 < n) :
  ∑ i in finset.range n, d i = n / (n + 1) :=
by
  simp only [d, b]
  sorry

end general_formula_and_sum_sum_bn_sequence_l66_66244


namespace sum_of_odd_divisors_l66_66068

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l66_66068


namespace sequence_properties_l66_66645

noncomputable def an_formula (n : ℕ) : ℤ :=
  2 * n - 1

noncomputable def sum_formula (n : ℕ) : ℤ :=
  2 * n^2 - n

theorem sequence_properties (a : ℕ → ℤ) (n : ℕ) :
  (∀ n, a n = 1 + (n - 1) * (a 2 - 1)) → -- Arithmetic sequence
  a 1 = 1 → -- First term is 1
  a 2 > 1 → -- Second term is greater than 1
  (a 2) * (a 14) = (a 5) ^ 2 → -- Terms form a geometric sequence
  (∀ n, a n = an_formula n) ∧ (sum (range n).map (λ k, a (2 * k + 1)) = sum_formula n) :=
by
  intros h_arith_seq ha1 ha2 h_geo_seq
  sorry -- Proof to be filled


end sequence_properties_l66_66645


namespace perpendicular_condition_parallel_condition_opposite_direction_l66_66556

/-- Conditions definitions --/
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

def k_vector_a_plus_b (k : ℝ) : ℝ × ℝ := (k - 3, 2 * k + 2)
def vector_a_minus_3b : ℝ × ℝ := (10, -4)

/-- Problem 1: Prove the perpendicular condition --/
theorem perpendicular_condition (k : ℝ) : (k_vector_a_plus_b k).fst * vector_a_minus_3b.fst + (k_vector_a_plus_b k).snd * vector_a_minus_3b.snd = 0 → k = 19 :=
by
  sorry

/-- Problem 2: Prove the parallel condition --/
theorem parallel_condition (k : ℝ) : (-(k - 3) / 10 = (2 * k + 2) / (-4)) → k = -1/3 :=
by
  sorry

/-- Determine if the vectors are in opposite directions --/
theorem opposite_direction (k : ℝ) (hk : k = -1/3) : k_vector_a_plus_b k = (-(1/3):ℝ) • vector_a_minus_3b :=
by
  sorry

end perpendicular_condition_parallel_condition_opposite_direction_l66_66556


namespace maximum_guards_watched_l66_66888

-- Define the 8x8 board
def Board := Fin 8 × Fin 8

-- Define the directions a guard can look in
inductive Direction
| up
| down
| left
| right

open Direction

-- Define what it means for a guard to watch another guard
def watches (g1 g2 : Board) (dir : Direction) : Prop :=
  match dir with
  | up => g1.snd = g2.snd ∧ g1.fst < g2.fst
  | down => g1.snd = g2.snd ∧ g1.fst > g2.fst
  | left => g1.fst = g2.fst ∧ g1.snd > g2.snd
  | right => g1.fst = g2.fst ∧ g1.snd < g2.snd

-- Define the main theorem statement
theorem maximum_guards_watched (k : ℕ) :
  (∀ g : Board, ∃ dir : Direction, (Σ g', watches g' g dir) ≥ k) → k ≤ 5 :=
sorry

end maximum_guards_watched_l66_66888


namespace sale_in_third_month_l66_66141

theorem sale_in_third_month 
  (sale1 sale2 sale4 sale5 sale6 : ℕ) 
  (avg_sale_months : ℕ) 
  (total_sales : ℕ)
  (h1 : sale1 = 6435) 
  (h2 : sale2 = 6927) 
  (h4 : sale4 = 7230) 
  (h5 : sale5 = 6562) 
  (h6 : sale6 = 7991) 
  (h_avg : avg_sale_months = 7000) 
  (h_total : total_sales = 6 * avg_sale_months) 
  : (total_sales - (sale1 + sale2 + sale4 + sale5 + sale6)) = 6855 :=
by
  have sales_sum := sale1 + sale2 + sale4 + sale5 + sale6
  have required_sales := total_sales - sales_sum
  sorry

end sale_in_third_month_l66_66141


namespace max_rays_non_exceed_l66_66379

open Real

-- Define the conditions
def rays_from_origin (rays : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ r ∈ rays, ∃ θ φ, r = ⟨cos θ * sin φ, sin θ * sin φ, cos φ⟩

def three_dimensional_space : Type := ℝ × ℝ × ℝ

def angle_between_rays_ge (r1 r2 : ℝ × ℝ × ℝ) (θ : ℝ) : Prop :=
  arccos ((r1.1 * r2.1 + r1.2 * r2.2 + r1.3 * r2.3) / 
          (sqrt (r1.1^2 + r1.2^2 + r1.3^2) * sqrt (r2.1^2 + r2.2^2 + r2.3^2))) ≥ θ

-- Define the problem statement to be proven
theorem max_rays_non_exceed (rays : Set (ℝ × ℝ × ℝ)) :
  rays_from_origin rays →
  (∀ r1 r2 ∈ rays, angle_between_rays_ge r1 r2 (π / 4)) →
  rays.card ≤ 27 :=
by sorry

end max_rays_non_exceed_l66_66379


namespace sumOddDivisorsOf90_l66_66043

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l66_66043


namespace least_x_condition_l66_66348

noncomputable def x : ℝ :=
  let y := 5.86 in
  ceil y

theorem least_x_condition :
  ∃ x : ℝ, x > 1 ∧ cos (3 * x) = sin (x^3 - 90 * (π / 180)) ∧ ceil x = 6 :=
by
  use 5.86
  have h1 : 5.86 > 1 := by linarith
  have h2 : cos (3 * 5.86) = sin ((5.86)^3 - 90 * (π / 180)), from sorry
  have h3 : ceil 5.86 = 6 := by norm_num
  exact ⟨h1, h2, h3⟩
  sorry

end least_x_condition_l66_66348


namespace pencil_length_l66_66790

theorem pencil_length (L : ℝ) (h1 : (1 / 8) * L + (1 / 2) * (7 / 8) * L + (7 / 2) = L) : L = 16 :=
by
  sorry

end pencil_length_l66_66790


namespace polynomial_exists_l66_66887

def exists_poly_13 (p : ℚ[X]) : Prop :=
  p.degree = 13 ∧ p.leadingCoeff = (1 / 1001:ℚ) ∧ ∀ n : ℤ, p.eval (n:ℚ) ∈ ℤ

theorem polynomial_exists : ∃ p : ℚ[X], exists_poly_13 p :=
sorry

end polynomial_exists_l66_66887


namespace ten_thousands_written_correctly_ten_thousands_truncated_correctly_l66_66418

-- Definitions to be used in the proof
def ten_thousands_description := "Three thousand nine hundred seventy-six ten thousands"
def num_written : ℕ := 39760000
def truncated_num : ℕ := 3976

-- Theorems to be proven
theorem ten_thousands_written_correctly :
  (num_written = 39760000) :=
sorry

theorem ten_thousands_truncated_correctly :
  (truncated_num = 3976) :=
sorry

end ten_thousands_written_correctly_ten_thousands_truncated_correctly_l66_66418


namespace sumOddDivisorsOf90_l66_66048

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l66_66048


namespace soldiers_joining_fort_l66_66119

noncomputable def initial_soldiers := 1200
noncomputable def initial_consumption_rate := 3 -- kg per day
noncomputable def initial_days := 30

noncomputable def additional_soldiers (x : ℕ) := initial_soldiers + x
noncomputable def new_consumption_rate := 2.5 -- kg per day
noncomputable def new_days := 25

noncomputable def total_provisions := initial_soldiers * initial_consumption_rate * initial_days

theorem soldiers_joining_fort (x : ℕ) : 
  total_provisions = additional_soldiers x * new_consumption_rate * new_days → x = 528 :=
by 
  sorry

end soldiers_joining_fort_l66_66119


namespace prod_of_four_is_square_l66_66738

theorem prod_of_four_is_square (s : Finset ℕ) (h_size : s.card = 48) 
  (h_primes : ∃ (p : Finset ℕ), p.card = 10 ∧ ∀ x ∈ s, ∀ d ∈ (Nat.factors x).toFinset, d ∈ p) : 
  ∃ (a b c d : ℕ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ ∃ k : ℕ, a * b * c * d = k ^ 2 := 
begin
  sorry
end

end prod_of_four_is_square_l66_66738


namespace max_value_fraction_l66_66477

theorem max_value_fraction {a b c : ℝ} (h1 : c = Real.sqrt (a^2 + b^2)) 
  (h2 : a > 0) (h3 : b > 0) (A : ℝ) (hA : A = 1 / 2 * a * b) :
  ∃ x : ℝ, x = (a + b + A) / c ∧ x ≤ (5 / 4) * Real.sqrt 2 :=
by
  sorry

end max_value_fraction_l66_66477


namespace find_solution_set_of_inequality_l66_66923

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x - 1 else -x - 1

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

theorem find_solution_set_of_inequality (f_odd : is_odd f) :
  {x : ℝ | f(x) < 0} = {x : ℝ | x < -1 ∨ (0 < x ∧ x < 1)} :=
by
  sorry

end find_solution_set_of_inequality_l66_66923


namespace min_sum_of_primes_l66_66291

theorem min_sum_of_primes (a b c d s : ℕ) [Prime a] [Prime b] [Prime c] [Prime d] [Prime s]
  (h : s = a + b + c + d) : s = 11 :=
sorry

end min_sum_of_primes_l66_66291


namespace cards_drawing_problem_l66_66412

-- Define the sets and parameters
def total_cards := 16
def group_size := 4
def red_cards := 4
def yellow_cards := 4
def blue_cards := 4
def green_cards := 4
def draw_cards := 3

-- Theorem statement
theorem cards_drawing_problem : 
  (nat.choose total_cards draw_cards) - 
  4 * (nat.choose group_size draw_cards) = 544 := 
by sorry

end cards_drawing_problem_l66_66412


namespace cos_four_alpha_sub_9pi_over_2_l66_66906

open Real

theorem cos_four_alpha_sub_9pi_over_2 (α : ℝ) 
  (cond : 4.53 * (1 + cos (2 * α - 2 * π) + cos (4 * α + 2 * π) - cos (6 * α - π)) /
                  (cos (2 * π - 2 * α) + 2 * cos (2 * α + π) ^ 2 - 1) = 2 * cos (2 * α)) :
  cos (4 * α - 9 * π / 2) = cos (4 * α - π / 2) :=
by sorry

end cos_four_alpha_sub_9pi_over_2_l66_66906


namespace domain_of_g_l66_66204

def g (x : ℝ) : ℝ := Real.sqrt (4 - Real.sqrt (6 - Real.sqrt x))

theorem domain_of_g :
  {x : ℝ | 0 ≤ x ∧ x ≤ 36} = {x : ℝ | ∃ y : ℝ, g x = y} :=
by
  -- The proof steps would go here, but are omitted for this statement task.
  sorry

end domain_of_g_l66_66204


namespace penalty_kicks_l66_66387

-- Define the soccer team data
def total_players : ℕ := 16
def goalkeepers : ℕ := 2
def players_shooting : ℕ := total_players - goalkeepers -- 14

-- Function to calculate total penalty kicks
def total_penalty_kicks (total_players goalkeepers : ℕ) : ℕ :=
  let players_shooting := total_players - goalkeepers
  players_shooting * goalkeepers

-- Theorem stating the number of penalty kicks
theorem penalty_kicks : total_penalty_kicks total_players goalkeepers = 30 :=
by
  sorry

end penalty_kicks_l66_66387


namespace fourDigitEvenNumbers_proof_fiveDigitMultiplesOf5_proof_fourDigitGreaterThan1325_proof_l66_66424

/-- 
(1) Number of four-digit even numbers without repeating digits using {0, 1, 2, 3, 4, 5} is 156
(2) Number of five-digit numbers without repeating digits that are multiples of 5 using {0, 1, 2, 3, 4, 5} is 216
(3) Number of four-digit numbers without repeating digits greater than 1325 using {0, 1, 2, 3, 4, 5} is 270
-/
noncomputable def fourDigitEvenNumbers : ℕ :=
  {n // 1000 ≤ n ∧ n < 10000 ∧ n % 2 = 0 ∧ (∃ d, n.digits = d ∧ d.nodup ∧ ∀ x ∈ d, x ∈ [0, 1, 2, 3, 4, 5])}

noncomputable def fiveDigitMultiplesOf5 : ℕ :=
  {n // 10000 ≤ n ∧ n < 100000 ∧ n % 5 = 0 ∧ (∃ d, n.digits = d ∧ d.nodup ∧ ∀ x ∈ d, x ∈ [0, 1, 2, 3, 4, 5])}

noncomputable def fourDigitGreaterThan1325 : ℕ :=
  {n // 1325 < n ∧ n < 10000 ∧ (∃ d, n.digits = d ∧ d.nodup ∧ ∀ x ∈ d, x ∈ [0, 1, 2, 3, 4, 5])}

theorem fourDigitEvenNumbers_proof : ∃ l, l.length = 156 ∧ ∀ x ∈ l,  x ∈ fourDigitEvenNumbers := sorry
theorem fiveDigitMultiplesOf5_proof : ∃ l, l.length = 216 ∧ ∀ x ∈ l,  x ∈ fiveDigitMultiplesOf5 := sorry
theorem fourDigitGreaterThan1325_proof : ∃ l, l.length = 270 ∧ ∀ x ∈ l,  x ∈ fourDigitGreaterThan1325 := sorry

end fourDigitEvenNumbers_proof_fiveDigitMultiplesOf5_proof_fourDigitGreaterThan1325_proof_l66_66424


namespace trapezoid_area_l66_66166

theorem trapezoid_area (h_base : ℕ) (sum_bases : ℕ) (height : ℕ) (hsum : sum_bases = 36) (hheight : height = 15) :
    (sum_bases * height) / 2 = 270 := by
  sorry

end trapezoid_area_l66_66166


namespace gender_has_impact_on_judgment_l66_66552

theorem gender_has_impact_on_judgment :
  (independence_test is the appropriate statistical method to determine
  if gender has an impact on the judgment of whether the human-machine competition is a victory for humanity)
  given
  (survey data: out of 2548 males, 1560 hold the opposing view, and out of 2452 females, 1200 hold the opposing view) :=
  sorry

end gender_has_impact_on_judgment_l66_66552


namespace graph_transform_l66_66962

theorem graph_transform (x : ℝ) :
  let f := λ x, cos (1 / 2 * x - π / 6)
  let g := λ x, cos (1 / 2 * (x + π / 3) - π / 6)
  let h := λ x, cos x
  (∀ x, g x = cos (x / 2)) → -- transforming the graph left by π/3 units
  (∀ x, f x = g x) → -- halving the x-coordinate
  ∀ x, f (x / 2) = h x := -- resulting graph
by
  intros f g h h₁ h₂
  intro x
  rw [h₂, h₁]
  sorry

end graph_transform_l66_66962


namespace car_clock_time_correct_l66_66390

noncomputable def car_clock (t : ℝ) : ℝ := t * (4 / 3)

theorem car_clock_time_correct :
  ∀ t_real t_car,
  (car_clock 0 = 0) ∧
  (car_clock 0.5 = 2 / 3) ∧
  (car_clock t_real = t_car) ∧
  (t_car = (8 : ℝ)) → (t_real = 6) → (t_real + 1 = 7) :=
by
  intro t_real t_car h
  sorry

end car_clock_time_correct_l66_66390


namespace find_g_of_7_l66_66293

theorem find_g_of_7 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 8) = 2 * x + 11) : g 7 = 21 :=
by
  sorry

end find_g_of_7_l66_66293


namespace polynomial_problem_l66_66225

theorem polynomial_problem :
  ∀ P : Polynomial ℤ,
    (∃ R : Polynomial ℤ, (X^2 + 6*X + 10) * P^2 - 1 = R^2) → 
    P = 0 :=
by { sorry }

end polynomial_problem_l66_66225


namespace complete_work_together_in_3_days_l66_66148

-- The man's work rate is W/4 per day
def man_work_rate (W : ℝ) := W / 4

-- The daughter's work rate is W/12 per day
def daughter_work_rate (W : ℝ) := W / 12

-- Combined work rate when they work together, which equals W/3 per day
def combined_work_rate (W : ℝ) := man_work_rate W + daughter_work_rate W

-- Function to determine the days taken to complete work together
def days_to_complete_together (W : ℝ) := W / combined_work_rate W

-- The goal is to prove that they complete the work together in 3 days
theorem complete_work_together_in_3_days (W : ℝ) : days_to_complete_together W = 3 :=
by
  sorry

end complete_work_together_in_3_days_l66_66148


namespace matrix_product_result_l66_66892

-- Define the general multiplication rule for the given matrices
lemma matrix_mul_rule (a b : ℕ) :
  (matrix ([[1, a], [0, 1]]) : matrix (fin 2) (fin 2) ℕ) ⬝
  (matrix ([[1, b], [0, 1]]) : matrix (fin 2) (fin 2) ℕ) =
  (matrix ([[1, a + b], [0, 1]]) : matrix (fin 2) (fin 2) ℕ) :=
by simp [matrix.mul, matrix.add]

-- Define the sum of the series 1, 2, ..., 100
def sum_series_100 : ℕ := (100 * (100 + 1)) / 2

-- Prove the main statement
theorem matrix_product_result :
  (2 : ℕ) • ∏ i in (finset.range 100), ((matrix ([[1, (i + 1)], [0, 1]]) : matrix (fin 2) (fin 2) ℕ)) =
  ((matrix ([[2, 10100], [0, 2]]) : matrix (fin 2) (fin 2) ℕ) : matrix (fin 2) (fin 2) ℕ) :=
sorry

end matrix_product_result_l66_66892


namespace train_passes_through_tunnel_in_12_seconds_l66_66482

theorem train_passes_through_tunnel_in_12_seconds
   (train_length : ℕ)
   (tunnel_length : ℕ)
   (speed : ℕ)
   (time : ℕ)
   (h_train_length : train_length = 180)
   (h_tunnel_length : tunnel_length = 120)
   (h_speed : speed = 25)
   (h_time : time = 12) :
   time = (train_length + tunnel_length) / speed := by
  rw [h_train_length, h_tunnel_length, h_speed, h_time]
  norm_num
  sorry

end train_passes_through_tunnel_in_12_seconds_l66_66482


namespace dice_probability_l66_66128

theorem dice_probability :
  ∃ (p : ℚ), 
  p = (135 / 4096) ∧
  (
    ∑ i in finset.range 6, if i < 10 then 1 else 0 = 2 * ∑ j in finset.range 6, if j < 10 then 1 else 0  
  ) :=
begin
  sorry
end

end dice_probability_l66_66128


namespace point_C_number_l66_66697

theorem point_C_number (B C: ℝ) (h1 : B = 3) (h2 : |C - B| = 2) :
  C = 1 ∨ C = 5 := 
by {
  sorry
}

end point_C_number_l66_66697


namespace percent_of_x_is_y_l66_66296

theorem percent_of_x_is_y (x y : ℝ) (h : 0.60 * (x - y) = 0.30 * (x + y)) : y = 0.3333 * x :=
by
  sorry

end percent_of_x_is_y_l66_66296


namespace a_2018_eq_3_l66_66922

noncomputable def a : ℕ → ℤ
| 1       := 2
| 2       := 3
| (n + 3) := a (n + 2) - a (n + 1)

theorem a_2018_eq_3 : a 2018 = 3 :=
by {
  sorry
}

end a_2018_eq_3_l66_66922


namespace candle_lighting_time_l66_66746

theorem candle_lighting_time 
  (l : ℕ) -- initial length of the candles
  (t_diff : ℤ := 206) -- the time difference in minutes, correlating to 1:34 PM.
  : t_diff = 206 :=
by sorry

end candle_lighting_time_l66_66746


namespace find_percentage_of_mixture_x_l66_66706

def mixture (X : ℝ) (Y : ℝ) (percent_x : ℝ) : Prop :=
  let P := 0.40 * percent_x + 0.25 * (1 - percent_x) in
  P = 0.32

theorem find_percentage_of_mixture_x (X Y percent_x : ℝ) (h1 : X = 0.4) (h2 : Y = 0.25) : 
  mixture X Y percent_x → percent_x ∈ set.Icc 0 1 → percent_x ≈ 7 / 15 :=
by
  sorry

end find_percentage_of_mixture_x_l66_66706


namespace divides_two_b_l66_66682

theorem divides_two_b (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : ∃ᵢ m n : ℕ, (m > 0 ∧ n > 0) ∧ 
                    (∃ k₁ : ℕ, m^2 + a * n + b = k₁^2) ∧ 
                    (∃ k₂ : ℕ, n^2 + a * m + b = k₂^2)) :
  a ∣ 2 * b :=
sorry

end divides_two_b_l66_66682


namespace first_divisor_is_13_l66_66810

theorem first_divisor_is_13 (x : ℤ) (h : (377 / x) / 29 * (1/4 : ℚ) / 2 = (1/8 : ℚ)) : x = 13 := by
  sorry

end first_divisor_is_13_l66_66810


namespace gcd_84_120_eq_12_l66_66426

theorem gcd_84_120_eq_12 : Int.gcd 84 120 = 12 := by
  sorry

end gcd_84_120_eq_12_l66_66426


namespace sum_odd_divisors_of_90_l66_66103

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l66_66103


namespace complex_in_second_quadrant_l66_66319

-- Define the complex number z based on the problem conditions.
def z : ℂ := Complex.I + (Complex.I^6)

-- State the condition to check whether z is in the second quadrant.
def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Formulate the theorem stating that the complex number z is in the second quadrant.
theorem complex_in_second_quadrant : is_in_second_quadrant z :=
by
  sorry

end complex_in_second_quadrant_l66_66319


namespace cyclic_quad_of_concurrency_l66_66671

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def foot_of_perpendicular (M D : Point) : Point := sorry
noncomputable def are_concurrent (A C M L : Point) : Prop := sorry
def is_cyclic_quad (K L M N : Point) : Prop := sorry

variables {A B C D M N K L : Point}

/-- Given a convex quadrilateral ABCD, let M and N be the midpoints of AB and AD respectively.
Let K be the foot of the perpendicular from M to CD, and L be the foot of the perpendicular from N to BC.
If AC, BD, MK, and NL are concurrent, then KLMN is a cyclic quadrilateral. -/
theorem cyclic_quad_of_concurrency (H_concurrent : are_concurrent (AC B C MKL)) :
  is_cyclic_quad K L M N := 
sorry

end cyclic_quad_of_concurrency_l66_66671


namespace transform_quadratic_to_linear_l66_66299

theorem transform_quadratic_to_linear (x y : ℝ) : 
  x^2 - 4 * x * y + 4 * y^2 = 4 ↔ (x - 2 * y + 2 = 0 ∨ x - 2 * y - 2 = 0) :=
by
  sorry

end transform_quadratic_to_linear_l66_66299


namespace matching_trio_probability_l66_66822

open Classical

def cards := 52
def removed_cards := 3
def remaining_cards := 49
def total_ways_to_choose_3_cards := Nat.choose remaining_cards 3
def number_of_successful_trios := (11 * Nat.choose 4 3) + (Nat.choose 3 3)

theorem matching_trio_probability 
  (m n : ℕ) 
  (rel_prime_m_n : Nat.gcd m n = 1)
  (frac_m_n : m / n = number_of_successful_trios / total_ways_to_choose_3_cards)
  : m + n = 18469 :=
by
  sorry

end matching_trio_probability_l66_66822


namespace domain_of_f_l66_66883

def f (x : ℝ) : ℝ := Real.sqrt x + Real.log (2 - x)

theorem domain_of_f :
  {x : ℝ | 0 ≤ x ∧ x < 2} = {x : ℝ | ∃ y, f y = f x} :=
by
  sorry

end domain_of_f_l66_66883


namespace sum_odd_divisors_l66_66101

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l66_66101


namespace largest_divisor_of_m_p1_l66_66297

theorem largest_divisor_of_m_p1 (m : ℕ) (h1 : m > 0) (h2 : 72 ∣ m^3) : 6 ∣ m :=
sorry

end largest_divisor_of_m_p1_l66_66297


namespace problem1_problem2_l66_66865

-- Proof problem 1
theorem problem1 : (π - 3)^0 + real.sqrt ((-4)^2) - (-1)^2023 = 6 := sorry

-- Proof problem 2
theorem problem2 : (real.sqrt 24 - real.sqrt 6) / real.sqrt 3 - 
                ((real.sqrt 3 - real.sqrt 2) * (real.sqrt 3 + real.sqrt 2)) = 
                real.sqrt 2 - 1 := sorry

end problem1_problem2_l66_66865


namespace num_of_pairs_l66_66538

theorem num_of_pairs (a b : ℕ) :
  (a ≤ 100000) ∧ (b ≤ 100000) ∧ (a^3 - b) * (b^2 + a^2) = (a^3 + b) * (b^2 - a^2) → 
  card ({(a, b) | (a, b) ∈ ℕ × ℕ ∧ a ≤ 100000 ∧ b ≤ 100000 ∧ (a^3 - b) * (b^2 + a^2) = (a^3 + b) * (b^2 - a^2)}.to_finset) = 10 :=
begin
  sorry
end

end num_of_pairs_l66_66538


namespace coefficient_x7_y2_of_expansion_l66_66648

noncomputable def coefficient_x7_y2_expansion : ℤ :=
  let term1 := Int.ofNat (Nat.choose 8 2)
  let term2 := -Int.ofNat (Nat.choose 8 1)
  term1 + term2

-- Statement of the proof problem
theorem coefficient_x7_y2_of_expansion :
  coefficient_x7_y2_expansion = 20 :=
by
  sorry

end coefficient_x7_y2_of_expansion_l66_66648


namespace ten_digit_number_l66_66134

open Nat

theorem ten_digit_number (a : Fin 10 → ℕ) (h1 : a 4 = 2)
  (h2 : a 8 = 3)
  (h3 : ∀ i, i < 8 → a i * a (i + 1) * a (i + 2) = 24) :
  a = ![4, 2, 3, 4, 2, 3, 4, 2, 3, 4] :=
sorry

end ten_digit_number_l66_66134


namespace gift_bags_needed_l66_66496

/-
  Constants
  total_expected: \(\mathbb{N}\) := 90        -- 50 people who will show up + 40 more who may show up
  total_prepared: \(\mathbb{N}\) := 30        -- 10 extravagant gift bags + 20 average gift bags

  The property to be proved:
  prove that (total_expected - total_prepared = 60)
-/

def total_expected : ℕ := 50 + 40
def total_prepared : ℕ := 10 + 20
def additional_needed := total_expected - total_prepared

theorem gift_bags_needed : additional_needed = 60 := by
  sorry

end gift_bags_needed_l66_66496


namespace powerful_rationals_l66_66866

theorem powerful_rationals (a b c : ℚ) (x y z : ℕ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_abc : a * b * c = 1)
  (h_xyz_int : ∃ x y z : ℕ, (a^x + b^y + c^z).denom = 1)
  : ∃ p1 q1 k1 p2 q2 k2 p3 q3 k3, (a = (p1^k1 / q1) ∧ b = (p2^k2 / q2) ∧ c = (p3^k3 / q3)) 
    ∧ (k1 > 1 ∧ k2 > 1 ∧ k3 > 1) 
    ∧ (nat.gcd p1 q1 = 1) ∧ (nat.gcd p2 q2 = 1) ∧ (nat.gcd p3 q3 = 1) := 
sorry

end powerful_rationals_l66_66866


namespace domain_of_func_l66_66722

open Set

noncomputable def func (x : ℝ) : ℝ := 2 / (sqrt (3 - 2 * x - x^2))

theorem domain_of_func :
  {x : ℝ | 3 - 2 * x - x^2 > 0} = Ioo (-3 : ℝ) (1 : ℝ) := sorry

end domain_of_func_l66_66722


namespace sum_odd_divisors_l66_66095

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l66_66095


namespace determine_k_l66_66946

noncomputable def p (x y : ℝ) : ℝ := x^2 - y^2
noncomputable def q (x y : ℝ) : ℝ := Real.log (x - y)

def m (k : ℝ) : ℝ := 2 * k
def w (n : ℝ) : ℝ := n + 1

theorem determine_k (k : ℝ) (c : ℝ → ℝ → ℝ) (v : ℝ → ℝ → ℝ) (n : ℝ) :
  p 32 6 = k * c 32 6 ∧
  p 45 10 = m k * c 45 10 ∧
  q 15 5 = n * v 15 5 ∧
  q 28 7 = w n * v 28 7 →
  k = 1925 / 1976 :=
by
  sorry

end determine_k_l66_66946


namespace cauliflower_production_diff_l66_66462

theorem cauliflower_production_diff
  (area_this_year : ℕ)
  (area_last_year : ℕ)
  (side_this_year : ℕ)
  (side_last_year : ℕ)
  (H1 : side_this_year * side_this_year = area_this_year)
  (H2 : side_last_year * side_last_year = area_last_year)
  (H3 : side_this_year = side_last_year + 1)
  (H4 : area_this_year = 12544) :
  area_this_year - area_last_year = 223 :=
by
  sorry

end cauliflower_production_diff_l66_66462


namespace derivative_of_f_l66_66241

noncomputable def f (x : Real) : Real := Real.exp x + Real.sin x

theorem derivative_of_f : Real.deriv f = fun x => Real.exp x + Real.cos x := by
  sorry

end derivative_of_f_l66_66241


namespace find_n_equals_3k_minus_1_l66_66432

variable {n k : ℕ}
variable {a b : ℝ}

-- Defining conditions
def condition1 : Prop := n ≥ 2
def condition2 : Prop := a ≠ 0 ∧ b ≠ 0
def condition3 : Prop := a = 2 * k * b
def condition4 : Prop := k ≥ 1
def condition5 : Prop := 
  (∑ i in (Finset.range (n+1)).filter (λ i, i = 2), (nat.choose n i) * (2*b)^(n-i) * (k-1)^i) + 
  (∑ i in (Finset.range (n+1)).filter (λ i, i = 3), (nat.choose n i) * (2*b)^(n-i) * (k-1)^i) = 0

-- Proof statement
theorem find_n_equals_3k_minus_1 (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) : 
  n = 3 * k - 1 :=
by
  sorry

end find_n_equals_3k_minus_1_l66_66432


namespace simplified_sqrt_expression_l66_66212

theorem simplified_sqrt_expression (x : ℝ) : (Real.sqrt (x^6 + x^3) = |x| * Real.sqrt (|x|) * Real.sqrt(x^3 + 1)) :=
  sorry

end simplified_sqrt_expression_l66_66212


namespace ratio_of_areas_l66_66168

-- Define the area function for an equilateral triangle
noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (√3 / 4) * s^2

-- Define the given conditions
def large_triangle_side : ℝ := 10
def small_triangle_side : ℝ := 6

-- Define the areas of the triangles based on the sides
def area_large_triangle : ℝ := area_equilateral_triangle large_triangle_side
def area_small_triangle : ℝ := area_equilateral_triangle small_triangle_side

-- Define the area of the trapezoid
def area_trapezoid : ℝ := area_large_triangle - area_small_triangle

-- Prove the ratio of the areas matches the given ratio
theorem ratio_of_areas :
  (area_small_triangle / area_trapezoid) = (9 / 16) :=
by
  sorry

end ratio_of_areas_l66_66168


namespace bags_needed_l66_66499

theorem bags_needed (expected_people extra_people extravagant_bags average_bags : ℕ) 
    (h1 : expected_people = 50) 
    (h2 : extra_people = 40) 
    (h3 : extravagant_bags = 10) 
    (h4 : average_bags = 20) : 
    (expected_people + extra_people - (extravagant_bags + average_bags) = 60) :=
by {
  sorry
}

end bags_needed_l66_66499


namespace central_angle_for_area_3_central_angle_for_max_area_l66_66404

-- Definitions for the conditions
def perimeter_of_sector (r : ℝ) : ℝ := 8 - 2 * r
def sector_area (r : ℝ) : ℝ := 0.5 * perimeter_of_sector(r) * r
def chord_length (r : ℝ) : ℝ := 2 * r * sin 1

-- Problem 1
theorem central_angle_for_area_3 (h : sector_area r = 3) : 
    r = 1 → central_angle = 6 ∧ r = 3 → central_angle = 2 / 3 := sorry

theorem central_angle_for_max_area : 
    r = 2 → (central_angle = 2 ∧ chord_length r = 4 * sin 1) := sorry

end central_angle_for_area_3_central_angle_for_max_area_l66_66404


namespace max_viewers_per_week_l66_66813

theorem max_viewers_per_week :
  ∃ (A B : ℕ), 600000 * A + 200000 * B = 2000000 ∧
               80 * A + 40 * B ≤ 320 ∧
               A + B ≥ 6 := 
begin
  sorry
end

end max_viewers_per_week_l66_66813


namespace range_a_range_b_l66_66931

def set_A : Set ℝ := {x | Real.log x / Real.log 2 > 2}
def set_B (a : ℝ) : Set ℝ := {x | x > a}
def set_C (b : ℝ) : Set ℝ := {x | b + 1 < x ∧ x < 2 * b + 1}

-- Part (1)
theorem range_a (a : ℝ) : (∀ x, x ∈ set_A → x ∈ set_B a) ↔ a ∈ Set.Iic 4 := sorry

-- Part (2)
theorem range_b (b : ℝ) : (set_A ∪ set_C b = set_A) ↔ b ∈ Set.Iic 0 ∪ Set.Ici 3 := sorry

end range_a_range_b_l66_66931


namespace non_right_triangle_option_B_l66_66658

structure Triangle (A B C : ℝ) : Prop :=
  (sum_angles : A + B + C = 180)

def right_triangle {A B C : ℝ} (t : Triangle A B C) : Prop :=
  A = 90 ∨ B = 90 ∨ C = 90

axiom option_A {α β γ : ℝ} (t : Triangle α β γ) : β = α + γ
axiom option_B {α β γ : ℝ} (t : Triangle α β γ) : α / 5 = β / 12 ∧ α / 5 = γ / 13
axiom option_C {a b c : ℝ} : a^2 = b^2 - c^2
axiom option_D {a b c : ℝ} : a / 5 = b / 12 ∧ a / 5 = c / 13

theorem non_right_triangle_option_B :
  (∃ (α β γ : ℝ), Triangle α β γ ∧ option_B (Triangle.mk (by sorry)) ∧ ¬ right_triangle (Triangle.mk (by sorry))) ∧
  (∀ (α β γ : ℝ), Triangle α β γ → option_A (Triangle.mk (by sorry)) → right_triangle (Triangle.mk (by sorry))) ∧
  (∀ (a b c : ℝ), option_C a b c → right_triangle (Triangle.mk (by sorry))) ∧
  (∀ (a b c : ℝ), option_D a b c → right_triangle (Triangle.mk (by sorry))) :=
sorry

end non_right_triangle_option_B_l66_66658


namespace range_of_m_l66_66547

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x ∈ set.Icc (-2 : ℝ) 1 → m * x^3 - x^2 + 4 * x + 3 ≥ 0) ↔ -6 ≤ m ∧ m ≤ -2 :=
by 
  sorry

end range_of_m_l66_66547


namespace function_value_sum_l66_66125

def f (x : ℝ) : ℝ := x^3 + 2*x

theorem function_value_sum : f(5) + f(-5) = 0 := by
  sorry

end function_value_sum_l66_66125


namespace range_of_A_l66_66591

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then -x^2 + 2*x - 5/4 else log (x) / log (1/3) - 1/4

noncomputable def g (A : ℝ) : ℝ → ℝ :=
λ x, abs (A - 2) * sin x

theorem range_of_A (A : ℝ) : 
  (∀ x1 x2 : ℝ, f x1 ≤ g A x2) ↔ (7/4 ≤ A ∧ A ≤ 9/4) := by
  sorry

end range_of_A_l66_66591


namespace entertainment_expense_percentage_l66_66704

noncomputable def salary : ℝ := 10000
noncomputable def savings : ℝ := 2000
noncomputable def food_expense_percentage : ℝ := 0.40
noncomputable def house_rent_percentage : ℝ := 0.20
noncomputable def conveyance_percentage : ℝ := 0.10

theorem entertainment_expense_percentage :
  let E := (1 - (food_expense_percentage + house_rent_percentage + conveyance_percentage) - (savings / salary))
  E = 0.10 :=
by
  sorry

end entertainment_expense_percentage_l66_66704


namespace sum_odd_divisors_90_eq_78_l66_66060

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l66_66060


namespace distance_between_stations_l66_66868

theorem distance_between_stations :
  ∀ (α β : Type) (t : ℝ) (vA vB vC : ℝ) (d : ℝ),
  t = 1 / 3 ∧
  vA = 90 ∧
  vB = 80 ∧
  vC = 60 ∧
  d = 425 →
  d = vA * t + (vB - vC) * t * vA / (vA + vB) + vA * t :=
begin
  sorry
end

end distance_between_stations_l66_66868


namespace number_of_common_points_l66_66211

-- Define the circle equation
def is_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

-- Define the vertical line equation
def is_on_line (x : ℝ) : Prop :=
  x = 3

-- Prove that the number of distinct points common to both graphs is two
theorem number_of_common_points : 
  ∃ y1 y2 : ℝ, is_on_circle 3 y1 ∧ is_on_circle 3 y2 ∧ y1 ≠ y2 :=
by {
  sorry
}

end number_of_common_points_l66_66211


namespace difference_between_ranges_l66_66811

noncomputable def GMAT_scores : Type := ℕ

variables (scores : ℕ → ℕ → GMAT_scores)
variables (rA rB rC rD : GMAT_scores)
variables (avgA avgB avgC avgD : GMAT_scores)

-- Given conditions
def valid_scores (s : GMAT_scores) : Prop := s >= 400 ∧ s <= 700
def rangeA := rA = 40
def rangeB := rB = 70
def rangeC := rC = 100
def rangeD := rD = 130
def avg_in_range (avg : GMAT_scores) : Prop := avg % 20 = 0 ∧ avg >= 450 ∧ avg <= 650

-- Define the maximum and minimum range calculations
def max_range := 700 - 400
def min_range := 505 - 470

-- The final theorem proving the result
theorem difference_between_ranges :
  valid_scores 400 → valid_scores 700 →
  avg_in_range avgA → avg_in_range avgB → avg_in_range avgC → avg_in_range avgD →
  rangeA → rangeB → rangeC → rangeD →
  max_range - min_range = 265 :=
by
  sorry

end difference_between_ranges_l66_66811


namespace max_value_of_expression_l66_66929

theorem max_value_of_expression (x y : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) (hy1 : 0 < y) (hy2 : y < π / 2) :
  ∃ (A : ℝ), A = \frac{\sqrt[4]{\sin x \sin y}}{\sqrt[4]{\tan x} + \sqrt[4]{\tan y}} ∧ 
  ∀ (x y : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) (hy1 : 0 < y) (hy2 : y < π / 2), 
    A ≤ \frac{\sqrt[4]{8}}{4} :=
sorry

end max_value_of_expression_l66_66929


namespace complement_union_l66_66967
-- Define the universal set U, sets A and B
def U : Set ℕ := {x | x < 9}
def A : Set ℕ := {1, 2, 3, 4, 5, 6}
def B : Set ℕ := {4, 5, 6}

-- Goal: proving \((\complement_U A) \cup B = \{4, 5, 6, 7, 8\}\)
theorem complement_union (U A B : Set ℕ) (hU : U = {x | x < 9})
  (hA : A = {1, 2, 3, 4, 5, 6}) (hB : B = {4, 5, 6}) :
  ((U \ A) ∪ B) = {4, 5, 6, 7, 8} :=
by {
  rw [hU, hA, hB],
  sorry
}

end complement_union_l66_66967


namespace sum_odd_divisors_90_eq_78_l66_66061

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l66_66061


namespace binomial_expansion_conditions_l66_66322

noncomputable def binomial_expansion (a b : ℝ) (x y : ℝ) (n : ℕ) : ℝ :=
(1 + a*x + b*y)^n

theorem binomial_expansion_conditions
  (a b : ℝ) (n : ℕ) 
  (h1 : (1 + b)^n = 243)
  (h2 : (1 + |a|)^n = 32) :
  a = 1 ∧ b = 2 ∧ n = 5 := by
  sorry

end binomial_expansion_conditions_l66_66322


namespace sum_of_a_is_9_l66_66357

theorem sum_of_a_is_9 (a b c : ℂ) (triples : List (ℂ × ℂ × ℂ))
  (h1 : ∀ (t : ℂ × ℂ × ℂ), t ∈ triples → fst t + (snd t).2 * (snd t).2 = 9)
  (h2 : ∀ (t : ℂ × ℂ × ℂ), t ∈ triples → (snd t).1 + fst t * (snd t).2 = 15)
  (h3 : ∀ (t : ℂ × ℂ × ℂ), t ∈ triples → (snd t).2 + fst t * (snd t).1 = 15) :
  triples.map (λ t, fst t).sum = 9 := sorry

end sum_of_a_is_9_l66_66357


namespace parity_E_2021_2022_2023_l66_66839

-- Define the sequence with the given initial conditions and recurrence relation
def E : ℕ → ℕ
| 0       := 0
| 1       := 1
| 2       := 1
| (n + 3) := E (n + 2) + E (n + 1) + E n

-- Define the parities of numbers
def parity (n : ℕ) : Prop :=
  (n % 2 = 0)

-- The math problem rephrased: prove the correctness of parities for the specific indices
theorem parity_E_2021_2022_2023 :
  parity (E 2021) ∧ ¬ parity (E 2022) ∧ ¬ parity (E 2023) :=
by sorry

end parity_E_2021_2022_2023_l66_66839


namespace marc_average_speed_l66_66501

theorem marc_average_speed 
  (d : ℝ) -- Define d as a real number representing distance
  (chantal_speed1 : ℝ := 3) -- Chantal's speed for the first half
  (chantal_speed2 : ℝ := 1.5) -- Chantal's speed for the second half
  (chantal_speed3 : ℝ := 2) -- Chantal's speed while descending
  (marc_meeting_point : ℝ := (2 / 3) * d) -- One-third point from the trailhead
  (chantal_time1 : ℝ := d / chantal_speed1) 
  (chantal_time2 : ℝ := (d / chantal_speed2))
  (chantal_time3 : ℝ := (d / 6)) -- Chantal's time for the descent from peak to one-third point
  (total_time : ℝ := chantal_time1 + chantal_time2 + chantal_time3) : 
  marc_meeting_point / total_time = 12 / 13 := 
  by 
  -- Leaving the proof as sorry to indicate where the proof would be
  sorry

end marc_average_speed_l66_66501


namespace sum_of_positive_odd_divisors_of_90_l66_66081

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l66_66081


namespace exists_sum_of_150_consecutive_integers_l66_66435

theorem exists_sum_of_150_consecutive_integers :
  ∃ a : ℕ, 1627395075 = 150 * a + 11175 :=
by
  sorry

end exists_sum_of_150_consecutive_integers_l66_66435


namespace numbered_cells_proximity_l66_66446

-- Definitions based on problem conditions
def is_numbered (board : ℕ → ℕ → ℕ) (x y : ℕ) : Prop := board x y > 0

def within_distance (b: ℕ → ℕ → ℕ) (d: ℕ) (p q : ℕ × ℕ) : Prop :=
  let (x1, y1) := p in
  let (x2, y2) := q in
  (x1 - x2)^2 + (y1 - y2)^2 < d^2

def distance (p q : ℕ × ℕ) : ℕ :=
  let (x1, y1) := p in
  let (x2, y2) := q in
  sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Main theorem statement
theorem numbered_cells_proximity
(board : ℕ → ℕ → ℕ)
(h1 : ∀ ix iy, ¬is_numbered board ix iy → ∃ jx jy, within_distance board 10 (ix, iy) (jx, jy) ∧ is_numbered board jx jy)
: ∃ (x1 y1 x2 y2 : ℕ), within_distance board 150 (x1, y1) (x2, y2) ∧ abs ((board x1 y1) - (board x2 y2)) > 23 :=
begin
  sorry
end

end numbered_cells_proximity_l66_66446


namespace chelsea_victory_bullseyes_l66_66972

theorem chelsea_victory_bullseyes (k : ℕ) (n : ℕ) : 
  (chelsea_lead : k ≥ 50) ∧ (scoring_points : ∀ (i : ℕ), i ∈ {0, 2, 4, 8, 10}) → 
  (min_bullseyes : 6 * n + 200 > 450) → n = 42 :=
by {
  sorry
}

end chelsea_victory_bullseyes_l66_66972


namespace sum_odd_divisors_90_eq_78_l66_66057

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l66_66057


namespace sum_odd_divisors_of_90_l66_66107

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l66_66107


namespace problem_statement_l66_66921

-- Define a "semi-odd function"
def semi_odd_function (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f(x) = -f(2 * a - x)

-- Define the individual functions
def f1 (x : ℝ) : ℝ := sqrt x
def f2 (x : ℝ) : ℝ := exp x
def f3 (x : ℝ) : ℝ := cos (x + 1)
def f4 (x : ℝ) : ℝ := tan x

-- Define the main theorem to prove which functions are semi-odd
theorem problem_statement : 
  ¬ (∃ a ≠ 0, semi_odd_function f1 a) ∧
  ¬ (∃ a ≠ 0, semi_odd_function f2 a) ∧
  (∃ a ≠ 0, semi_odd_function f3 a) ∧
  (∃ a ≠ 0, semi_odd_function f4 a) := 
by
  sorry

end problem_statement_l66_66921


namespace minimum_cubes_structure_l66_66161

/-- 
A structure is constructed using unit cubes where each cube shares at least one full face with 
another cube. We are given that the front view of the structure is a rectangle of height 2 and 
width 1, the side view shows an 'L' shape composed of heights 2 and 1, and the back view shows 
the structure is only one cube in height. Prove that the minimum number of cubes needed to build 
such a structure is 3. 
-/
theorem minimum_cubes_structure : 
  ∃ (n : ℕ), n = 3 ∧ 
  (∃ (S : Set (ℕ × ℕ × ℕ)), 
    (∀ (c : ℕ × ℕ × ℕ), c ∈ S → 
      (∃ d ∈ S, 
        ((c.1 = d.1 + 1 ∨ c.1 + 1 = d.1) ∧ c.2 = d.2 ∧ c.3 = d.3) ∨
        ((c.2 = d.2 + 1 ∨ c.2 + 1 = d.2) ∧ c.1 = d.1 ∧ c.3 = d.3) ∨
        ((c.3 = d.3 + 1 ∨ c.3 + 1 = d.3) ∧ c.1 = d.1 ∧ c.2 = d.2)
      )
    ) ∧ 
    (∀ (x y : ℕ), x ≤ 1 ∧ y ≤ 0 → ∃ z, (x, 0, z) ∈ S) ∧
    (∀ (x y : ℕ), x ≥ 1 ∧ y ≤ 1 → ∃ z, (x, 0, z) ∈ S) ∧
    (∀ (y z : ℕ), y = 0 ∧ z ≤ 1 → ∃ x, (0, y, z) ∈ S) ∧
    (∀ (y z : ℕ), y = 1 ∧ z ≤ 0 → ∃ x, (1, y, z) ∈ S)
  )
:= by
  sorry

end minimum_cubes_structure_l66_66161


namespace find_sum_of_x_and_reciprocal_l66_66943

theorem find_sum_of_x_and_reciprocal (x : ℝ) (hx_condition : x^3 + 1/x^3 = 110) : x + 1/x = 5 := 
sorry

end find_sum_of_x_and_reciprocal_l66_66943


namespace find_AB_length_l66_66619

theorem find_AB_length (A B C : Type*) [Inhabited A] [Inhabited B] [Inhabited C]
  (angle_B_is_right : ∠ABC = 90)
  (tan_A_1_over_3 : tan ∠BAC = 1 / 3)
  (BC_len : dist B C = 45) :
  dist A B ≈ 14.23 :=
begin
  sorry
end

end find_AB_length_l66_66619


namespace log_expression_eval_find_m_from_conditions_l66_66187

-- (1) Prove that lg (5^2) + (2/3) * lg 8 + lg 5 * lg 20 + (lg 2)^2 = 3.
theorem log_expression_eval : 
  Real.logb 10 (5^2) + (2 / 3) * Real.logb 10 8 + Real.logb 10 5 * Real.logb 10 20 + (Real.logb 10 2)^2 = 3 := 
sorry

-- (2) Given 2^a = 5^b = m and 1/a + 1/b = 2, prove that m = sqrt(10).
theorem find_m_from_conditions (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 :=
sorry

end log_expression_eval_find_m_from_conditions_l66_66187


namespace sum_odd_divisors_l66_66096

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l66_66096


namespace volume_of_second_cube_is_twosqrt2_l66_66039

noncomputable def side_length (volume : ℝ) : ℝ :=
  volume^(1/3)

noncomputable def surface_area (side : ℝ) : ℝ :=
  6 * side^2

theorem volume_of_second_cube_is_twosqrt2
  (v1 : ℝ)
  (h1 : v1 = 1)
  (A1 := surface_area (side_length v1))
  (A2 := 2 * A1)
  (s2 := (A2 / 6)^(1/2)) :
  (s2^3 = 2 * Real.sqrt 2) :=
by
  sorry

end volume_of_second_cube_is_twosqrt2_l66_66039


namespace find_y_coordinate_of_C_l66_66965

theorem find_y_coordinate_of_C
  (A : Point := ⟨0, 2⟩)
  (on_parabola : ∀ (p : Point), p.y^2 = p.x + 4)
  (B : Point)
  (C : Point)
  (H_B_parabola : on_parabola B)
  (H_C_parabola : on_parabola C)
  (H_perpendicular : line_through A B ⊥ line_through B C) :
  C.y ≤ 0 ∨ C.y ≥ 4 :=
sorry

end find_y_coordinate_of_C_l66_66965


namespace maximum_quadratic_value_l66_66884

noncomputable def quadratic_expression (q : ℝ) : ℝ := -3 * q^2 + 18 * q + 7

theorem maximum_quadratic_value : 
  ∃ (q0 : ℝ), ∀ (q : ℝ), quadratic_expression q ≤ quadratic_expression q0 ∧ quadratic_expression q0 = 34 :=
begin
  sorry
end

end maximum_quadratic_value_l66_66884


namespace sophie_buys_six_doughnuts_l66_66712

variable (num_doughnuts : ℕ)

theorem sophie_buys_six_doughnuts 
  (h1 : 5 * 2 = 10)
  (h2 : 4 * 2 = 8)
  (h3 : 15 * 0.60 = 9)
  (h4 : 10 + 8 + 9 = 27)
  (h5 : 33 - 27 = 6)
  (h6 : num_doughnuts * 1 = 6) :
  num_doughnuts = 6 := 
  by
    sorry

end sophie_buys_six_doughnuts_l66_66712


namespace range_of_f_l66_66956

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem range_of_f : 
  ∀ x, -1 ≤ x ∧ x ≤ 1 → -2 ≤ f x ∧ f x ≤ 2 :=
by
  intro x Hx
  sorry

end range_of_f_l66_66956


namespace figure_at_1000th_position_position_of_1000th_diamond_l66_66728

-- Define the repeating sequence
def repeating_sequence : List String := ["△", "Λ", "◇", "Λ", "⊙", "□"]

-- Lean 4 statement for (a)
theorem figure_at_1000th_position :
  repeating_sequence[(1000 % repeating_sequence.length) - 1] = "Λ" :=
by sorry

-- Define the arithmetic sequence for diamond positions
def diamond_position (n : Nat) : Nat :=
  3 + (n - 1) * 6

-- Lean 4 statement for (b)
theorem position_of_1000th_diamond :
  diamond_position 1000 = 5997 :=
by sorry

end figure_at_1000th_position_position_of_1000th_diamond_l66_66728


namespace vector_inequality_l66_66924

variables {R : Type*} [NormedField R] [NormedSpace R (EuclideanSpace ℝ (Fin 3))]

theorem vector_inequality 
  (a b : EuclideanSpace ℝ (Fin 3)) 
  (ha : a ≠ 0) (hb : b ≠ 0) :
  ∥a∥ - ∥b∥ ≤ ∥a + b∥ ∧ ∥a + b∥ ≤ ∥a∥ + ∥b∥ ∧ ∥a∥ - ∥b∥ ≤ ∥a - b∥ ∧ ∥a - b∥ ≤ ∥a∥ + ∥b∥ :=
by
  sorry

end vector_inequality_l66_66924


namespace tunnel_length_scale_l66_66626

theorem tunnel_length_scale (map_length_cm : ℝ) (scale_ratio : ℝ) (convert_factor : ℝ) : 
  map_length_cm = 7 → scale_ratio = 38000 → convert_factor = 100000 →
  (map_length_cm * scale_ratio / convert_factor) = 2.66 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end tunnel_length_scale_l66_66626


namespace ticket_sales_revenue_l66_66908

theorem ticket_sales_revenue (total_tickets student_price nonstudent_price student_tickets_sold : ℕ)
  (h1 : total_tickets = 821)
  (h2 : student_price = 2)
  (h3 : nonstudent_price = 3)
  (h4 : student_tickets_sold = 530) :
  (student_tickets_sold * student_price + (total_tickets - student_tickets_sold) * nonstudent_price) = 1933 :=
by {
  rw [h1, h2, h3, h4],
  sorry
}

end ticket_sales_revenue_l66_66908


namespace sum_odd_divisors_of_90_l66_66109

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l66_66109


namespace find_b_l66_66126

-- Define the conditions in the problem
def quadratic_original (b : ℝ) : ℝ[X] := X^2 + C b * X + 56
def quadratic_rewritten (n : ℝ) : ℝ[X] := (X + C n)^2 + 12

-- The statement to be proved
theorem find_b :
  ∃ b n : ℝ, quadratic_original b = quadratic_rewritten n ∧ b = 4 * Real.sqrt 11 :=
sorry

end find_b_l66_66126


namespace combined_score_l66_66625

variable (A J M : ℕ)

-- Conditions
def Jose_score_more_than_Alisson : Prop := J = A + 40
def Meghan_score_less_than_Jose : Prop := M = J - 20
def total_possible_score : ℕ := 100
def Jose_questions_wrong (wrong_questions : ℕ) : Prop := J = total_possible_score - (wrong_questions * 2)

-- Proof statement
theorem combined_score (h1 : Jose_score_more_than_Alisson)
                       (h2 : Meghan_score_less_than_Jose)
                       (h3 : Jose_questions_wrong 5) :
                       A + J + M = 210 := by
  sorry

end combined_score_l66_66625


namespace lcm_sum_not_power_of_two_l66_66367

open Nat

theorem lcm_sum_not_power_of_two :
  ∀ (S : Finset ℕ) (R B : Finset ℕ), 
    (S = R ∪ B) ∧ (∀ x ∈ R, x ∉ B) ∧ (∀ x ∈ B, x ∉ R) ∧ 
    R.nonempty ∧ B.nonempty ∧
    (∀ a b ∈ S, a ≠ b → abs (a - b) = 1) →
    ¬ (∃ k : ℕ, (lcm (R : Set ℕ)).to_finset.sum + (lcm (B : Set ℕ)).to_finset.sum = 2^k) :=
by
  sorry

end lcm_sum_not_power_of_two_l66_66367


namespace problem1_l66_66342

def setA : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def setB (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 2 * m - 2}

theorem problem1 (m : ℝ) : 
  (∀ x, x ∈ setA → x ∈ setB m) ∧ ¬(∀ x, x ∈ setA ↔ x ∈ setB m) → 3 ≤ m :=
sorry

end problem1_l66_66342


namespace max_pieces_of_pie_l66_66375

theorem max_pieces_of_pie : ∃ (PIE PIECE : ℕ), 10000 ≤ PIE ∧ PIE < 100000
  ∧ 10000 ≤ PIECE ∧ PIECE < 100000
  ∧ ∃ (n : ℕ), n = 7 ∧ PIE = n * PIECE := by
  sorry

end max_pieces_of_pie_l66_66375


namespace complete_the_square_b_l66_66112

theorem complete_the_square_b (x : ℝ) : (x ^ 2 - 6 * x + 7 = 0) → ∃ a b : ℝ, (x + a) ^ 2 = b ∧ b = 2 :=
by
sorry

end complete_the_square_b_l66_66112


namespace cafe_combination_l66_66814

theorem cafe_combination (coffee_types syrups : ℕ) (hc : coffee_types = 5) (hs : syrups = 7) : 
  coffee_types * (Nat.choose syrups 3) = 175 :=
by
  rw [hc, hs, Nat.choose]
  have comb : Nat.choose 7 3 = 35 := by
    simp [Nat.choose]
  rw comb
  norm_num

end cafe_combination_l66_66814


namespace inequality_proof_equality_case_l66_66685

variables (x y z : ℝ)
  
theorem inequality_proof 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z ≥ 3) : 
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 := 
sorry

theorem equality_case 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z ≥ 3) 
  (h_eq : (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) = 1) :
  x = 1 ∧ y = 1 ∧ z = 1 := 
sorry

end inequality_proof_equality_case_l66_66685


namespace number_of_remainders_mod_210_l66_66172

theorem number_of_remainders_mod_210 (p : ℕ) (hp : Nat.Prime p) (hp_gt_7 : p > 7) :
  ∃ (remainders : Finset ℕ), (remainders.card = 12) ∧ ∀ r ∈ remainders, ∃ k, p^2 ≡ r [MOD 210] :=
by
  sorry

end number_of_remainders_mod_210_l66_66172


namespace permutation_value_l66_66608

theorem permutation_value (n : ℕ) (h : n * (n - 1) = 12) : n = 4 :=
by
  sorry

end permutation_value_l66_66608


namespace parametric_to_polar_and_MinMax_MN_l66_66890

-- Definitions from the conditions
def x_α (α : ℝ) : ℝ := 2 + 4 * cos α
def y_α (α : ℝ) : ℝ := 2 * sqrt 3 + 4 * sin α

-- Polar coordinates condition
def curve_C1 (ρ θ : ℝ) : Prop :=
  ρ = 8 * cos (θ - π/3)

-- Line C2 condition
def line_C2 (t β : ℝ) : ℝ × ℝ :=
  (2 + t * cos β, sqrt 3 + t * sin β)

-- Main theorem statement
theorem parametric_to_polar_and_MinMax_MN 
    (α β t1 t2 : ℝ) :
  (curve_C1 (sqrt ((x_α α)^2 + (y_α α)^2)) (real.atan2 (y_α α) (x_α α))) ∧
  let M := line_C2 t1 β in 
  let N := line_C2 t2 β in 
  let MN := abs (t2 - t1) in
  ((MN = 8) ∨ (MN = 2 * sqrt 13)) := 
by 
  sorry

end parametric_to_polar_and_MinMax_MN_l66_66890


namespace sumOddDivisorsOf90_l66_66044

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l66_66044


namespace words_needed_for_90_percent_l66_66977

theorem words_needed_for_90_percent (total_words : ℕ) (desired_percentage : ℕ) (correct_answer : ℕ) : 
  total_words = 600 → 
  desired_percentage = 90 → 
  correct_answer = 540 → 
  correct_answer = (desired_percentage * total_words) / 100 := 
by 
  intros h_total_words h_desired_percentage h_correct_answer 
  rw [h_total_words, h_desired_percentage, h_correct_answer] 
  norm_num 
  sorry

end words_needed_for_90_percent_l66_66977


namespace minimal_coins_for_mark_minimal_coins_for_mark_special_l66_66545

theorem minimal_coins_for_mark {a b : ℕ} (h1 : 1 < a) (h2 : a < b) (h3 : b = 2 * a) :
  ∀ n < 100, ∃ m ≤ 14, ∃ c1 c2 c3 : ℕ, c1 + c2 + c3 = m ∧ c1 * 1 + c2 * a + c3 * b = n :=
begin
  sorry
end

-- Specific instance for a = 7, b = 14
theorem minimal_coins_for_mark_special :
  ∀ n < 100, ∃ m ≤ 14, ∃ c1 c2 c3 : ℕ, c1 + c2 + c3 = m ∧ c1 * 1 + c2 * 7 + c3 * 14 = n :=
begin
  sorry
end

end minimal_coins_for_mark_minimal_coins_for_mark_special_l66_66545


namespace translated_midpoint_correct_l66_66707

open Real Int

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

theorem translated_midpoint_correct :
  let s1_p1 : ℝ × ℝ := (2, -3)
  let s1_p2 : ℝ × ℝ := (10, 7)
  let s2_translation : ℝ × ℝ := (-3, -2)
  let midpoint_s1 := midpoint s1_p1 s1_p2
  let midpoint_s2 := translate midpoint_s1 s2_translation.1 s2_translation.2
  midpoint_s2 = (3, 0) :=
by {
  let s1_p1 : ℝ × ℝ := (2, -3)
  let s1_p2 : ℝ × ℝ := (10, 7)
  let s2_translation : ℝ × ℝ := (-3, -2)
  let midpoint_s1 := midpoint s1_p1 s1_p2
  let midpoint_s2 := translate midpoint_s1 s2_translation.1 s2_translation.2
  show midpoint_s2 = (3, 0), from sorry
}

end translated_midpoint_correct_l66_66707


namespace intersection_A_B_l66_66601

open Set

def A : Set (ℝ × ℝ) :=
  { p | ∃ x y, p = (x, y) ∧ y = x + 1 }

def B : Set (ℝ × ℝ) :=
  { p | ∃ x y, p = (x, y) ∧ y = 4 - 2x }

theorem intersection_A_B :
  A ∩ B = { (1, 2) } :=
sorry

end intersection_A_B_l66_66601


namespace smallest_among_5_8_4_l66_66041

theorem smallest_among_5_8_4 : ∀ (x y z : ℕ), x = 5 → y = 8 → z = 4 → z ≤ x ∧ z ≤ y :=
by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  exact ⟨by norm_num, by norm_num⟩

end smallest_among_5_8_4_l66_66041


namespace sumOddDivisorsOf90_l66_66047

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l66_66047


namespace sum_of_positive_odd_divisors_of_90_l66_66080

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l66_66080


namespace number_of_divisors_3276_l66_66228

theorem number_of_divisors_3276 : 
  let n := 3276
  let prime_factors := [(2, 2), (3, 3), (7, 1), (11, 1)]
  n = 2^2 * 3^3 * 7^1 * 11^1 → 
  ∀ k ∈ prime_factors, (1 + (k.snd)) * 48 = 48 :=
by
  let n := 3276
  let prime_factors := [(2, 2), (3, 3), (7, 1), (11, 1)]
  have h : n = 2^2 * 3^3 * 7^1 * 11^1 := rfl
  have t : (2 + 1) * (3 + 1) * (1 + 1) * (1 + 1) = 48 := by norm_num
  sorry

end number_of_divisors_3276_l66_66228


namespace area_of_triangle_ABC_l66_66913

/-- 
Given a triangle ABC with point M inside it. Perpendiculars 
from M to the sides BC, AC, and AB have lengths k, l, and m respectively.
Also given the angles \(\angle CAB = \alpha\) and \(\angle ABC = \beta\),
the following definitions hold.
-/
variables {A B C M : Type}
variables (k l m : ℝ)
variables (α β : ℝ)

def area_triangle (A B C : Type) : ℝ := 
  let γ := Real.arcsin ((Real.sin (π - α - β))),
      k := 3, l := 2, m := 4 in
  (3 * (Real.sin α) + 2 * (Real.sin β) + 4 * (Real.sin γ))^2 / (2 * (Real.sin α) * (Real.sin β) * (Real.sin γ))

-- Given the specific values
def α : ℝ := π / 6
def β : ℝ := π / 4

-- Expected area
def expected_area : ℝ := 67

-- Prove the area of triangle ABC to be approximately 67
theorem area_of_triangle_ABC : 
  (area_triangle α β) ≈ expected_area := 
by sorry

end area_of_triangle_ABC_l66_66913


namespace projection_of_MN_on_y_axis_l66_66019

-- Vertices of convex quadrilateral \(ABCD\) lie on the parabola \(y=x^2\).
def parabola (x : ℝ) : ℝ := x^2

-- \(ABCD\) is cyclic with \( AC \) as a diameter of the circumcircle.
def is_cyclic (A B C D : ℝ × ℝ) : Prop :=
  ∃ O R, (A.1 - O.1)^2 + (A.2 - O.2)^2 = R^2 ∧
         (B.1 - O.1)^2 + (B.2 - O.2)^2 = R^2 ∧
         (C.1 - O.1)^2 + (C.2 - O.2)^2 = R^2 ∧
         (D.1 - O.1)^2 + (D.2 - O.2)^2 = R^2 ∧
         (C.1 - A.1)^2 + (C.2 - A.2)^2 = (2 * R)^2

-- Midpoints \( M \) of \( AC \) and \( N \) of \( BD \)
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- The projection length calculation
def projection_length_y (M N : ℝ × ℝ) : ℝ :=
  abs (M.2 - N.2)

theorem projection_of_MN_on_y_axis
  (a c : ℝ)
  (A := (a, parabola a))
  (C := (c, parabola c))
  (M := midpoint A C)
  (b d : ℝ)
  (B := (b, parabola b))
  (D := (d, parabola d))
  (N := midpoint B D)
  (h_parabola_B : B.2 = parabola B.1)
  (h_parabola_D : D.2 = parabola D.1)
  (h_cyclic : is_cyclic A B C D) :
  projection_length_y M N = 1 :=
by
  sorry

end projection_of_MN_on_y_axis_l66_66019


namespace shopper_pays_correct_amount_l66_66156

-- Define the original price of the coat
def original_price: ℝ := 120

-- Define the discount factor for a 30% discount
def discount_factor: ℝ := 0.30

-- Define the coupon amount
def coupon: ℝ := 10

-- Define the rebate amount
def rebate: ℝ := 5

-- Define the sales tax rate (5%)
def sales_tax_rate: ℝ := 0.05

-- Calculate the final price the shopper pays
noncomputable def final_price (original_price: ℝ) (discount_factor: ℝ) (coupon: ℝ) (rebate: ℝ) (sales_tax_rate: ℝ): ℝ :=
  let discounted_price := original_price * (1 - discount_factor) in
  let after_coupon := discounted_price - coupon in
  let after_rebate := after_coupon - rebate in
  after_rebate * (1 + sales_tax_rate)

-- The theorem to prove
theorem shopper_pays_correct_amount :
  final_price original_price discount_factor coupon rebate sales_tax_rate = 72.45 :=
by
  sorry

end shopper_pays_correct_amount_l66_66156


namespace rectangle_width_length_ratio_l66_66321

theorem rectangle_width_length_ratio (w l P : ℕ) (h_l : l = 10) (h_P : P = 30) (h_perimeter : 2*w + 2*l = P) :
  w / l = 1 / 2 := 
by {
  sorry
}

end rectangle_width_length_ratio_l66_66321


namespace antonette_score_l66_66171

theorem antonette_score :
  ∀ (score1 score2 score3 num1 num2 num3 total score_final : ℕ),
  score1 = 70 * num1 / 100 ∧ score2 = 80 * num2 / 100 ∧ score3 = 90 * num3 / 100 ∧
  num1 = 10 ∧ num2 = 20 ∧ num3 = 30 ∧ total = num1 + num2 + num3 ∧
  score_final = (score1 + score2 + score3) * 100 / total →
  score_final = 83 :=
by
  intros score1 score2 score3 num1 num2 num3 total score_final
  -- Assume conditions
  intro h
  -- Destructuring the conditions tuple
  cases h with hscore1 rest
  cases rest with hscore2 rest
  cases rest with hscore3 rest
  cases rest with hnum1 rest
  cases rest with hnum2 rest
  cases rest with hnum3 rest
  cases rest with htotal hfinal

  -- Skipping the proofs
  sorry

end antonette_score_l66_66171


namespace inequality_neg_mul_l66_66290

theorem inequality_neg_mul (a b : ℝ) (h : a > b) : -3 * a < -3 * b :=
sorry

end inequality_neg_mul_l66_66290


namespace length_of_plot_l66_66445

theorem length_of_plot (B : ℝ) (h1 : ∃ B, L = B + 10)
  (h2 : 26.50 ∙ (4 ∙ B + 20) = 5300) : L = 55 :=
by 
  have hB : B = 45 := by sorry  -- from the given equation 4B + 20 = 200
  rw [hB, L] -- L = B + 10
  exact rfl

end length_of_plot_l66_66445


namespace solve_quadratic_equation_solve_inequality_system_l66_66803

-- Part 1: Solve the quadratic equation
theorem solve_quadratic_equation :
  ∀ (x : ℝ), 2 * x^2 + x - 2 = 0 ↔ (x = (-1 + real.sqrt 17) / 4 ∨ x = (-1 - real.sqrt 17) / 4) :=
by
  intro x
  sorry

-- Part 2: Solve the inequality system
theorem solve_inequality_system :
  ∀ (x : ℝ), (x + 3 > -2 * x) ∧ (2 * x - 5 < 1) ↔ (-1 < x ∧ x < 3) :=
by
  intro x
  sorry

end solve_quadratic_equation_solve_inequality_system_l66_66803


namespace general_terms_sequences_sum_first_n_terms_l66_66596

-- Constants as described in the conditions
variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {c : ℕ → ℝ}
variable {T : ℕ → ℝ}

-- Definitions based on given problem conditions
-- Geometric sequence with a_1 = 2
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = q * a n

def a_n (n : ℕ) := 2 ^ n
def b_n (n : ℕ) := (n * (n + 1)) / 2
def c_n (n : ℕ) := a_n n + (1 / b_n n)
def T_n (n : ℕ) := ∑ i in finset.range n, c_n (i + 1)

-- Questions & Conditions in Lean 4 statements
theorem general_terms_sequences :
  (∀ n, a 1 * a 2 * a 3 * ... * a n = 2 ^ b n) → 
  (a 1 = 2) → 
  (b 3 = b 2 + 3) →
  (∀ n, a n = 2 ^ n ∧ b n = (n * (n + 1)) / 2) :=
sorry

theorem sum_first_n_terms:
  (∀ n, c n = a n + 1 / b n) →
  (∀ n, T n = ∑ i in finset.range n, c (i + 1)) →
  (∀ n, T n = 2^(n+1) - (2 / (n + 1))) :=
sorry

end general_terms_sequences_sum_first_n_terms_l66_66596


namespace bird_costs_l66_66180

-- Define the cost of a small bird and a large bird
def cost_small_bird (x : ℕ) := x
def cost_large_bird (x : ℕ) := 2 * x

-- Define total cost calculations for the first and second ladies
def cost_first_lady (x : ℕ) := 5 * cost_large_bird x + 3 * cost_small_bird x
def cost_second_lady (x : ℕ) := 5 * cost_small_bird x + 3 * cost_large_bird x

-- State the main theorem
theorem bird_costs (x : ℕ) (hx : cost_first_lady x = cost_second_lady x + 20) : 
(cost_small_bird x = 10) ∧ (cost_large_bird x = 20) := 
by {
  sorry
}

end bird_costs_l66_66180


namespace sum_of_odd_divisors_l66_66070

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l66_66070


namespace fraction_of_selected_films_in_color_l66_66783

variables (x y : ℕ)
def B : ℕ := 30 * x
def C : ℕ := 6 * y
def selectedBWFilms : ℕ := 3 * y / 10
def selectedColorFilms : ℕ := 6 * y
def fractionColorFilms : ℚ := selectedColorFilms / (selectedBWFilms + selectedColorFilms)

theorem fraction_of_selected_films_in_color : fractionColorFilms = 20 / 21 := sorry

end fraction_of_selected_films_in_color_l66_66783


namespace find_x_plus_inv_x_l66_66942

theorem find_x_plus_inv_x (x : ℝ) (hx : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 := 
by 
  sorry

end find_x_plus_inv_x_l66_66942


namespace minimize_fractions_sum_l66_66341

theorem minimize_fractions_sum {A B C D E : ℕ}
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : A ≠ E)
  (h5 : B ≠ C) (h6 : B ≠ D) (h7 : B ≠ E)
  (h8 : C ≠ D) (h9 : C ≠ E) (h10 : D ≠ E)
  (h11 : A ≠ 9) (h12 : B ≠ 9) (h13 : C ≠ 9) (h14 : D ≠ 9) (h15 : E ≠ 9)
  (hA : 1 ≤ A) (hB : 1 ≤ B) (hC : 1 ≤ C) (hD : 1 ≤ D) (hE : 1 ≤ E)
  (hA' : A ≤ 9) (hB' : B ≤ 9) (hC' : C ≤ 9) (hD' : D ≤ 9) (hE' : E ≤ 9) :
  A / B + C / D + E / 9 = 125 / 168 :=
sorry

end minimize_fractions_sum_l66_66341


namespace wire_rounds_field_10_times_l66_66847

noncomputable def area_of_square_field : ℝ := 53824
noncomputable def length_of_wire : ℝ := 9280

theorem wire_rounds_field_10_times (area : ℝ) (wire_length : ℝ) 
  (h_area : area = area_of_square_field)
  (h_wire_length : wire_length = length_of_wire) : 
  (wire_length / (4 * (real.sqrt area))) = 10 :=
by
  sorry

end wire_rounds_field_10_times_l66_66847


namespace num_divisors_35_424_l66_66286

theorem num_divisors_35_424 : 
  ∃ n : ℕ, n = 6 ∧ (∀ d ∈ {1, 2, 3, 4, 6, 9}, d ∣ 35424) ∧ (∀ d ∈ {5, 7, 8}, ¬ d ∣ 35424) := 
begin
  use 6,
  split,
  { refl },
  split,
  { intros d hd,
    fin_cases hd; 
    norm_num, 
    { unfold dvd, use 35424 / d, norm_num },
    { unfold dvd, use 17712, refl },
    { unfold dvd, use 11808, refl },
    { unfold dvd, use 8856, refl },
    { unfold dvd, use 5904, refl },
    { unfold dvd, use 3936, refl } },
  { intros d hd,
    fin_cases hd; 
    norm_num,
    { unfold dvd, intro h, rcases h with ⟨k, hk⟩, dsimp at hk, linarith },
    { unfold dvd, intro h, rcases h with ⟨k, hk⟩, dsimp at hk, linarith },
    { unfold dvd, intro h, rcases h with ⟨k, hk⟩, dsimp at hk, linarith } },
end

end num_divisors_35_424_l66_66286


namespace fraction_of_number_l66_66032

-- Given definitions based on the problem conditions
def fraction : ℚ := 3 / 4
def number : ℕ := 40

-- Theorem statement to be proved
theorem fraction_of_number : fraction * number = 30 :=
by
  sorry -- This indicates that the proof is not yet provided

end fraction_of_number_l66_66032


namespace median_to_hypotenuse_of_right_triangle_l66_66313

theorem median_to_hypotenuse_of_right_triangle (DE DF : ℝ) (h₁ : DE = 6) (h₂ : DF = 8) :
  let EF := Real.sqrt (DE^2 + DF^2)
  let N := EF / 2
  N = 5 :=
by
  let EF := Real.sqrt (DE^2 + DF^2)
  let N := EF / 2
  have h : N = 5 :=
    by
      sorry
  exact h

end median_to_hypotenuse_of_right_triangle_l66_66313


namespace count_valid_paths_l66_66131

-- Definitions based on the given conditions
def start_point := (-5, -5) : ℤ × ℤ
def end_point := (5, 5) : ℤ × ℤ
def rectangle (p : ℤ × ℤ) : Prop :=
  (-3 ≤ p.1 ∧ p.1 ≤ 3) ∧ (-1 ≤ p.2 ∧ p.2 ≤ 1)

def step (p1 p2 : ℤ × ℤ) : Prop :=
  (p2.1 = p1.1 + 1 ∧ p2.2 = p1.2) ∨ (p2.1 = p1.1 ∧ p2.2 = p1.2 + 1)

def valid_path (path : List (ℤ × ℤ)) : Prop :=
  path.head = start_point ∧
  path.tail.tail.last = end_point ∧
  ∀ (i : ℕ), i < path.length - 1 → step (path[i]) (path[i+1]) ∧ ¬rectangle (path[i+1])

-- Statement of the problem in Lean 4
theorem count_valid_paths : 
  ∃ (n : ℕ), n = 2126 ∧ ∃ (paths : List (List (ℤ × ℤ))), ∀ (path : List (ℤ × ℤ)), path ∈ paths → valid_path path :=
sorry  -- Proof to be filled in

end count_valid_paths_l66_66131


namespace merchant_discount_percentage_l66_66785

variable (C : ℝ)
def marked_up_percentage := 0.6
def profit_percentage := 0.2
def marked_price (C : ℝ) := C * (1 + marked_up_percentage)
def selling_price (C : ℝ) := C * (1 + profit_percentage)

theorem merchant_discount_percentage (C : ℝ) :
  let M := marked_price C,
      S := selling_price C
  in ((M - S) / M) * 100 = 25 := by
  sorry

end merchant_discount_percentage_l66_66785


namespace trapezoid_original_area_l66_66835

variable (lower_base upper_base height : ℝ)
variable (h1 : upper_base = (3/5) * lower_base)
variable (h2 : lower_base - 8 = lower_base / sqrt 2)
variable (original_area : ℝ)
noncomputable def trapezoid_area : ℝ := ((lower_base + upper_base) * height) / 2

theorem trapezoid_original_area
    (h3 : height = lower_base / sqrt 2) :
  original_area = 192 :=
  by
    have h4 : lower_base = 4 := sorry
    have h5 : height = 4 := sorry
    have h6 : upper_base = (3/5) * 4 := sorry
    exact sorry

end trapezoid_original_area_l66_66835


namespace geometric_series_ratio_l66_66237

theorem geometric_series_ratio (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ (n : ℕ), S n = a 1 * (1 - q^n) / (1 - q))
  (h2 : a 3 + 2 * a 6 = 0)
  (h3 : a 6 = a 3 * q^3)
  (h4 : q^3 = -1 / 2) :
  S 3 / S 6 = 2 := 
sorry

end geometric_series_ratio_l66_66237


namespace area_quadrilateral_ADEC_l66_66323

theorem area_quadrilateral_ADEC {C D E A B : Type} [geometry C D E A B] : 
  let angleC := 90
  let AD := (1 / 3) * DB
  let DE_perp_AB := DE ⊥ AB
  let AB := 30
  let AC := 18
  area ADEC = 26.16 :=
by
  sorry

end area_quadrilateral_ADEC_l66_66323


namespace total_golden_apples_cost_l66_66857

theorem total_golden_apples_cost :
  let Hephaestus := (3 + 2) * 4 + (6 + 2) * 4 + (9 + 2) * 4
  let Athena := 5 * 6 + 7.5 * 6 + 10
  let Ares := 4 * 3 + 6 * 6 + 8 * 3 + 3 * 4
  Hephaestus + Athena + Ares = 265 :=
by
  let Hephaestus := (3 + 2) * 4 + (6 + 2) * 4 + (9 + 2) * 4
  let Athena := 5 * 6 + 7.5 * 6 + 10
  let Ares := 4 * 3 + 6 * 6 + 8 * 3 + 3 * 4
  have h₁ : Hephaestus = 96 := by sorry
  have h₂ : Athena = 85 := by sorry
  have h₃ : Ares = 84 := by sorry
  calc
    Hephaestus + Athena + Ares = 96 + 85 + 84 := by rw [h₁, h₂, h₃]
    ... = 265 := by norm_num

end total_golden_apples_cost_l66_66857


namespace number_of_possible_ones_digits_l66_66359

open Finset

-- Define the condition of being divisible by 6, which entails being divisible by both 2 and 3
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define the set of possible ones digits for an even number
def even_ones_digits : Finset ℕ := {0, 2, 4, 6, 8}

-- Define the condition for the sum of digits being divisible by 3
def sum_of_digits_divisible_by_3 (n : ℕ) : Prop := 
  let digits := n.digits 10 in
  digits.sum % 3 = 0

-- State the problem: how many different ones digits are possible for numbers Ana likes
theorem number_of_possible_ones_digits : 
  (even_ones_digits.filter (λ d, ∃ n, n % 10 = d ∧ divisible_by_6 n)).card = 5 :=
sorry

end number_of_possible_ones_digits_l66_66359


namespace page_number_added_twice_l66_66737

-- Define the sum of natural numbers from 1 to n
def sum_nat (n: ℕ): ℕ := n * (n + 1) / 2

-- Incorrect sum due to one page number being counted twice
def incorrect_sum (n p: ℕ): ℕ := sum_nat n + p

-- Declaring the known conditions as Lean definitions
def n : ℕ := 70
def incorrect_sum_val : ℕ := 2550

-- Lean theorem statement to be proven
theorem page_number_added_twice :
  ∃ p, incorrect_sum n p = incorrect_sum_val ∧ p = 65 := by
  sorry

end page_number_added_twice_l66_66737


namespace cylinder_volume_tripled_and_radius_increased_l66_66989

theorem cylinder_volume_tripled_and_radius_increased :
  ∀ (r h : ℝ), let V := π * r^2 * h in
               let V_new := π * (2.5 * r)^2 * (3 * h) in
               V_new = 18.75 * V :=
by
  intros r h
  let V := π * r^2 * h
  let V_new := π * (2.5 * r)^2 * (3 * h)
  sorry

end cylinder_volume_tripled_and_radius_increased_l66_66989


namespace fraction_of_cream_in_cup1_is_correct_l66_66705

-- Definitions based on the conditions
def initial_coffee_cup1 := 5 -- ounces
def initial_cream_cup2 := 3 -- ounces

def fraction_poured := 1/3

-- Lean function to model the described process of pouring and mixing
def final_fraction_of_cream_in_cup1 : ℚ :=
  let coffee_cup1_step1 := initial_coffee_cup1 - fraction_poured * initial_coffee_cup1 in
  let mixed_cup2_step1 := initial_cream_cup2 + fraction_poured * initial_coffee_cup1 in
  let poured_back_cup1_step1 := fraction_poured * mixed_cup2_step1 in
  let total_liquid_cup1_step1 := coffee_cup1_step1 + poured_back_cup1_step1 in

  let cream_fraction_cup2_step1 := initial_cream_cup2 / mixed_cup2_step1 in
  let cream_transferred_back_cup1_step1 := poured_back_cup1_step1 * cream_fraction_cup2_step1 in

  let coffee_cup1_step2 := coffee_cup1_step1 - fraction_poured * coffee_cup1_step1 in
  let mixed_cup2_step2 := mixed_cup2_step1 - poured_back_cup1_step1 + fraction_poured * coffee_cup1_step1 in
  let poured_back_cup1_step2 := fraction_poured * mixed_cup2_step2 in
  let cream_fraction_cup2_step2 := (mixed_cup2_step1 * cream_fraction_cup2_step1) / mixed_cup2_step2 in
  let cream_transferred_back_cup1_step2 := poured_back_cup1_step2 * cream_fraction_cup2_step2 in
  let total_liquid_cup1_step2 := total_liquid_cup1_step1 + poured_back_cup1_step2 - fraction_poured * total_liquid_cup1_step1 in

  let total_cream_cup1 := cream_transferred_back_cup1_step1 + cream_transferred_back_cup1_step2 in
  total_cream_cup1 / total_liquid_cup1_step2

-- Proof statement comparing computed final fraction with the correct answer's fraction
theorem fraction_of_cream_in_cup1_is_correct : final_fraction_of_cream_in_cup1 = specific_value :=
  by
  -- sorry is used as placeholder for the actual proof
  sorry

end fraction_of_cream_in_cup1_is_correct_l66_66705


namespace shift_right_by_3pi_div_8_l66_66023

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (Real.pi / 4))

noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

theorem shift_right_by_3pi_div_8 :
  ∀ x, f(x) = g(x - 3 * Real.pi / 8) :=
by
  sorry

end shift_right_by_3pi_div_8_l66_66023


namespace equation_of_line_l66_66146

noncomputable theory
open_locale classical

-- Definitions
def point (x y : ℝ) : (ℝ × ℝ) := (x, y)
def ellipse (a b : ℝ) (p : ℝ × ℝ) : Prop := (p.1^2 / a^2) + (p.2^2 / b^2) = 1
def line (m : ℝ × ℝ) (slope : ℝ) (p : ℝ × ℝ) : Prop := 
  p.2 = slope * (p.1 - m.1) + m.2

-- Problem conditions
variables (A B M : ℝ × ℝ)
def line_through (M : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := 
  ∃ (m : ℝ), line M m A ∧ line M m B

def midpoint (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def intersects_ellipse (A B : ℝ × ℝ) : Prop :=
  ellipse 2 3 A ∧ ellipse 2 3 B

-- Statement to prove
theorem equation_of_line 
  (A B : ℝ × ℝ) (M : ℝ × ℝ) 
  (h1 : line_through M A B)
  (h2 : midpoint A B M)
  (h3 : intersects_ellipse A B) :
  ∃ (k : ℝ), line (1,1) (-3/4) (1,1) ∧ (3 * A.1 + 4 * A.2 - 7 = 0) ∧ (3 * B.1 + 4 * B.2 - 7 = 0) :=
sorry

end equation_of_line_l66_66146


namespace log_a_b_integer_probability_l66_66027

-- Define variables and the set
def set : Finset ℕ := (Finset.range 15).map ⟨λ n => 3^(n + 1), sorry⟩

-- Define the problem in Lean 4
theorem log_a_b_integer_probability : 
  (∀ (a b : ℕ), a ∈ set → b ∈ set → a ≠ b → ∃ (z : ℕ), log a b = z) → 
  ∑ x in Finset.range 15, (Nat.floor (15 / (x + 1)) - 1) = 29 → 
  Nat.choose 15 2 = 105 → 
  29 / 105 = 29 / 105 :=
sorry

end log_a_b_integer_probability_l66_66027


namespace squares_sum_l66_66680

theorem squares_sum {r s : ℝ} (h1 : r * s = 16) (h2 : r + s = 8) : r^2 + s^2 = 32 :=
by
  sorry

end squares_sum_l66_66680


namespace function_not_satisfy_equations_l66_66269

theorem function_not_satisfy_equations (a b : ℝ) :
  let f := λ x : ℝ, 2 ^ x in
  ¬ ((∀ a b, f(a + b) = f(a) + f(b)) ∨
     (∀ a b, f(ab) = f(a) + f(b)) ∨
     (∀ a b, f(ab) = f(a) * f(b))) :=
by
  let f := λ x : ℝ, 2 ^ x
  sorry

end function_not_satisfy_equations_l66_66269


namespace find_number_l66_66135

theorem find_number (x : ℕ) (h : x * 48 = 173 * 240) : x = 865 :=
sorry

end find_number_l66_66135


namespace point_C_number_l66_66696

theorem point_C_number (B C: ℝ) (h1 : B = 3) (h2 : |C - B| = 2) :
  C = 1 ∨ C = 5 := 
by {
  sorry
}

end point_C_number_l66_66696


namespace area_of_right_isosceles_triangle_l66_66314

theorem area_of_right_isosceles_triangle (A B C : Type) [Triangle ABC] 
  (right_triangle : is_right_triangle ABC)
  (angle_eq : ∠A = ∠B) 
  (hypotenuse_length : AB = 8 * (real.sqrt 2)) : 
  area ABC = 32 := sorry

end area_of_right_isosceles_triangle_l66_66314


namespace sum_of_odd_divisors_l66_66069

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l66_66069


namespace box_volume_l66_66439

theorem box_volume (L W S : ℕ) 
  (hL : L = 48) 
  (hW : W = 36) 
  (hS : S = 5) : 
  L - 2 * S = 38 ∧ W - 2 * S = 26 → 
  (L - 2 * S) * (W - 2 * S) * S = 9880 := 
by 
  simp [hL, hW, hS] 
  sorry

end box_volume_l66_66439


namespace find_sum_l66_66781

variables (P : ℝ)

def simple_interest_18 (P : ℝ) : ℝ :=
  P * (18 / 100) * 2

def simple_interest_12 (P : ℝ) : ℝ :=
  P * (12 / 100) * 2

theorem find_sum (h : simple_interest_18 P - simple_interest_12 P = 300) : P = 2500 :=
by
  sorry

end find_sum_l66_66781


namespace find_a_from_quadratic_with_complex_root_l66_66259

theorem find_a_from_quadratic_with_complex_root
  (a : ℂ)
  (h1 : (1 + Complex.sqrt 2 * Complex.I) isRoot (λ x : ℂ, x^2 - 2 * a * x + (a^2 - 4 * a + 6))) :
  a = 1 :=
sorry

end find_a_from_quadratic_with_complex_root_l66_66259


namespace b_work_rate_l66_66116

variable {W : Type} [inst : CommSemiring W] [Nontrivial W] [NoZeroDivisors W]

theorem b_work_rate (a b : W) :
  (1 / 18) + (1 / b) = (1 / 6) → b = 9 := 
  by
  sorry

end b_work_rate_l66_66116


namespace nicky_cristina_headstart_l66_66360

theorem nicky_cristina_headstart :
  ∀ (nicky_speed cristina_speed nicky_time : ℝ),
    nicky_speed = 3 ∧ cristina_speed = 5 ∧ nicky_time = 30 →
    ∃ (head_start : ℝ),
    head_start = 12 :=
by
  intros nicky_speed cristina_speed nicky_time h
  rcases h with ⟨h_n_s, h_c_s, h_n_t⟩
  have distance_nicky := h_n_s * h_n_t
  have time_cristina := distance_nicky / h_c_s
  have head_start := h_n_t - time_cristina
  use head_start
  field_simp at *
  linarith only [h_n_s, h_c_s, h_n_t]
  sorry

end nicky_cristina_headstart_l66_66360


namespace find_a_l66_66917

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) 
  (h : (6 * a * (-1) + 6) = 4) : 
  a = 10 / 3 :=
by {
  sorry
}

end find_a_l66_66917


namespace part1_part2_l66_66240

def Q1 (x : ℝ) : Prop :=
  tan (π/4 + x) = -1/2

def Q2 (x : ℝ) : Prop :=
  tan x = -3 ∧ x > π/2 ∧ x < π -- assuming x in the second quadrant

theorem part1 (x : ℝ) (h : Q1 x) : tan (2 * x) = 3 / 4 :=
  sorry

theorem part2 (x : ℝ) (h : Q2 x) :
  sqrt ((1 + sin x) / (1 - sin x)) + sqrt ((1 - sin x) / (1 + sin x)) = 2 * sqrt 10 :=
  sorry

end part1_part2_l66_66240


namespace cos_squared_sum_sin_squared_sum_l66_66691

theorem cos_squared_sum (A B C : ℝ) (h : A + B + C = 180) :
  cos(A)^2 + cos(B)^2 + cos(C)^2 = 1 - 2 * cos(A) * cos(B) * cos(C) :=
by 
  sorry

theorem sin_squared_sum (A B C : ℝ) (h : A + B + C = 180) :
  sin(A)^2 + sin(B)^2 + sin(C)^2 = 2 * cos(A) * cos(B) * cos(C) :=
by 
  sorry

end cos_squared_sum_sin_squared_sum_l66_66691


namespace angle_C_is_30_degrees_l66_66576

theorem angle_C_is_30_degrees (a b c A B C : ℝ) (h1 : a^2 + b^2 - c^2 = 4 * sqrt 3 * (1/2 * a * b * sin C)) : C = 30 :=
by
  sorry

end angle_C_is_30_degrees_l66_66576


namespace series_value_l66_66345

noncomputable def geometric_series_sum (a r : ℝ) : ℝ :=
  a / (1 - r)

theorem series_value (c d : ℝ) (h1 : (c / d) + (c / d^2) + (c / d^3) + ⋯ = 3) :
  (c / (c + 2 * d)) + (c / (c + 2 * d)^2) + (c / (c + 2 * d)^3) + ⋯ = 3 / 5 := by
  sorry

end series_value_l66_66345


namespace homothety_center_l66_66372

-- Given circles and inversion conditions
variables {O : Type} [Group O]
variables (S S_prime : Type) [Group S] [Group S_prime]
variables (ω : Type) [Group ω]
variables (center : O → ω) (inversion_transformation : S → S_prime)

-- Prove that O is a center of homothety of S and S_prime
theorem homothety_center
  (h_inv : inversion_transformation S S_prime)
  (h_center : center O ω) :
  is_center_of_homothety O S S_prime :=
sorry

end homothety_center_l66_66372


namespace prove_inequality_l66_66542

theorem prove_inequality (x : ℝ) (h : 3 * x^2 + x - 8 < 0) : -2 < x ∧ x < 4 / 3 :=
sorry

end prove_inequality_l66_66542


namespace smallest_num_rectangles_to_cover_square_l66_66430

-- Define essential conditions
def area_3by4_rectangle : ℕ := 3 * 4
def area_square (side_length : ℕ) : ℕ := side_length * side_length
def can_be_tiled_with_3by4 (side_length : ℕ) : Prop := (area_square side_length) % area_3by4_rectangle = 0

-- Define the main theorem
theorem smallest_num_rectangles_to_cover_square :
  can_be_tiled_with_3by4 12 → ∃ n : ℕ, n = (area_square 12) / area_3by4_rectangle ∧ n = 12 :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l66_66430


namespace vector_c_condition_l66_66970

variables (a b c : ℝ × ℝ)

def is_perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def is_parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v = (k * w.1, k * w.2)

theorem vector_c_condition (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (2, -3)) 
  (hc : c = (7 / 2, -7 / 4)) :
  is_perpendicular c a ∧ is_parallel b (a - c) :=
sorry

end vector_c_condition_l66_66970


namespace first_sales_amount_l66_66470

-- Conditions from the problem
def first_sales_royalty : ℝ := 8 -- million dollars
def second_sales_royalty : ℝ := 9 -- million dollars
def second_sales_amount : ℝ := 108 -- million dollars
def decrease_percentage : ℝ := 0.7916666666666667

-- The goal is to determine the first sales amount, S, meeting the conditions.
theorem first_sales_amount :
  ∃ S : ℝ,
    (first_sales_royalty / S - second_sales_royalty / second_sales_amount = decrease_percentage * (first_sales_royalty / S)) ∧
    S = 20 :=
sorry

end first_sales_amount_l66_66470


namespace import_tax_percentage_l66_66149

theorem import_tax_percentage
  (total_value : ℝ)
  (non_taxable_portion : ℝ)
  (import_tax_paid : ℝ)
  (h_total_value : total_value = 2610)
  (h_non_taxable_portion : non_taxable_portion = 1000)
  (h_import_tax_paid : import_tax_paid = 112.70) :
  ((import_tax_paid / (total_value - non_taxable_portion)) * 100) = 7 :=
by
  sorry

end import_tax_percentage_l66_66149


namespace number_of_children_l66_66520

theorem number_of_children (total_crayons children_crayons children : ℕ) 
  (h1 : children_crayons = 3) 
  (h2 : total_crayons = 18) 
  (h3 : total_crayons = children_crayons * children) : 
  children = 6 := 
by 
  sorry

end number_of_children_l66_66520


namespace expected_net_profit_theorem_l66_66816

/-- Problem: A math proof converting the expected value calculation of net profit in Lean. Given conditions and proving the expected value of selling one product. -/
noncomputable def expected_net_profit 
  (purchase_price : ℝ)  -- Purchase price of the product
  (pass_rate : ℝ)       -- Pass rate of the product
  (profit_qualified : ℝ) -- Net profit for each qualified product
  (loss_defective : ℝ)  -- Net loss for a defective product
  (X : ℝ → ℝ)          -- Net profit from selling one product
  (E : (ℝ → ℝ) → ℝ)    -- Expectation operator
  : Prop :=
  E X = 1.4

-- Definitions
variables (purchase_price : ℝ) (pass_rate : ℝ) (profit_qualified : ℝ) (loss_defective : ℝ)
          (X : ℝ → ℝ) (E : (ℝ → ℝ) → ℝ)

-- Assign values to variables based on the problem conditions
def purchase_price_val : ℝ := 10
def pass_rate_val : ℝ := 0.95
def profit_qualified_val : ℝ := 2
def loss_defective_val : ℝ := 10

-- Define net profit X
def net_profit (x : ℝ) : ℝ := if x < pass_rate_val then profit_qualified_val else -loss_defective_val

-- Expected value definition (for example purpose)
def expected_value (f : ℝ → ℝ) : ℝ := 0.95 * f 1 + 0.05 * f 0

-- The statement that needs to be proven
theorem expected_net_profit_theorem : expected_value net_profit = 1.4 :=
by
  sorry

end expected_net_profit_theorem_l66_66816


namespace rearrange_cards_l66_66720

def reverse_segment (l : list ℕ) (start len : ℕ) : list ℕ :=
  let (left, mid_right) := l.split_at start in
  let (mid, right) := mid_right.split_at len in
  left ++ (mid.reverse) ++ right

theorem rearrange_cards :
  ∃ (seq : list ℕ), 
    let ops := [ (0, 6), (3, 6), (0, 6) ] in
    let final_seq := list.foldl (λ l (p : ℕ × ℕ), reverse_segment l p.1 p.2) seq ops in
    seq == [7, 8, 9, 4, 5, 6, 1, 2, 3] →
    final_seq == [1, 2, 3, 4, 5, 6, 7, 8, 9] :=
by
  use [7, 8, 9, 4, 5, 6, 1, 2, 3]
  sorry

end rearrange_cards_l66_66720


namespace covered_area_min_max_l66_66563

-- Definitions for the problem
variables (a r : ℝ)

-- Conditions from the problem
def circle_k_centered (r : ℝ) : Prop := r ≥ 0
def circle_k_constraint1 (a r : ℝ) : Prop := r ≤ (a * Real.sqrt 3) / 2
def vertex_circles_constraint (a r : ℝ) : Prop := r ≥ a / 2

-- The covered area function in terms of r
def covered_area (a r : ℝ) : ℝ := π * (3*r^2 - 4*a*r + 2*a^2)

-- Derivation of the minimum and maximum points
def min_area_radius (a : ℝ) : ℝ := 2 * a / 3
def max_area_radius (a : ℝ) : ℝ := (a * Real.sqrt 3) / 2

theorem covered_area_min_max (a : ℝ) : 
  circle_k_centered (min_area_radius a) ∧ vertex_circles_constraint a (min_area_radius a) ∧ circle_k_constraint1 a (min_area_radius a) ∧ 
  circle_k_centered (max_area_radius a) ∧ vertex_circles_constraint a (max_area_radius a) ∧ circle_k_constraint1 a (max_area_radius a) → 
  (covered_area a (min_area_radius a) ≤ covered_area a r ∧ covered_area a r ≤ covered_area a (max_area_radius a)) := 
by
  sorry

end covered_area_min_max_l66_66563


namespace speed_of_stream_l66_66825

-- Definitions of the problem's conditions
def downstream_distance := 72
def upstream_distance := 30
def downstream_time := 3
def upstream_time := 3

-- The unknowns
variables (b s : ℝ)

-- The effective speed equations based on the problem conditions
def effective_speed_downstream := b + s
def effective_speed_upstream := b - s

-- The core conditions of the problem
def condition1 : Prop := downstream_distance = effective_speed_downstream * downstream_time
def condition2 : Prop := upstream_distance = effective_speed_upstream * upstream_time

-- The problem statement transformed into a Lean theorem
theorem speed_of_stream (h1 : condition1) (h2 : condition2) : s = 7 := 
sorry

end speed_of_stream_l66_66825


namespace ellipse_properties_l66_66565

theorem ellipse_properties :
  (let a := 3 in let c := sqrt 5 in let b := 2 in
  let std_eq := ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) in
  let PF1 := 4 in let d := (6/5) * sqrt 5 in
  std_eq ∧ 
  (∀ P : ℝ × ℝ, (P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧ sqrt ((P.1 + c)^2 + P.2^2) = PF1 → sqrt ((P.1 - c)^2 + P.2^2) = 2 ∧ (2 / d = sqrt 5 / 3)))):
sorry

end ellipse_properties_l66_66565


namespace angle_bisector_limits_l66_66717

theorem angle_bisector_limits (A B C D E F : Point) (α : ℝ) :
  (is_triangle A B C) ∧ (A ≠ B) ∧ (A ≠ C) ∧ 
  (bisector_intersects_opposite_sides A B C D E F) ∧
  (length (segment B A) ≠ length (segment A C)) →
  102.663 < α → α < 104.477 →
  angle A B C = α → angle A C B = α → length (segment D E) = length (segment D F).

end angle_bisector_limits_l66_66717


namespace problem_A_problem_C_l66_66113

section
variables {a b : ℝ}

-- A: If a and b are positive real numbers, and a > b, then a^3 + b^3 > a^2 * b + a * b^2.
theorem problem_A (ha : 0 < a) (hb : 0 < b) (h : a > b) : a^3 + b^3 > a^2 * b + a * b^2 := sorry

end

section
variables {a b : ℝ}

-- C: If a and b are real numbers, then "a > b > 0" is a sufficient but not necessary condition for "1/a < 1/b".
theorem problem_C (ha : 0 < a) (hb : 0 < b) (h : a > b) : 1/a < 1/b := sorry

end

end problem_A_problem_C_l66_66113


namespace number_of_4digit_numbers_with_two_identical_digits_l66_66402

def has_two_identical_digits (n : ℕ) : Prop :=
  (n / 1000 = 1) ∧ (∃ x y : ℕ, (x ≠ y) ∧ (x ≠ 1) ∧ (y ≠ 1) ∧ (
    (n % 1000 = 100 * x + 10 * y + 1) ∨
    (n % 1000 = 100 * x + 10 + y) ∨
    (n % 1000 = 10 * x + y * 10 + 1) ∨
    (n % 1000 = 100 * y + x + 10) ∨
    (n % 1000 = 10 * y + x + 100)
  ))

theorem number_of_4digit_numbers_with_two_identical_digits :
  (∃ count : ℕ, count = 360 ∧ count = (Nat.card { n : ℕ // n / 1000 = 1 ∧ has_two_identical_digits n })) :=
by
  sorry

end number_of_4digit_numbers_with_two_identical_digits_l66_66402


namespace marble_problem_l66_66169

theorem marble_problem (x : ℝ) (h1 : ∀ a b c d : ℝ, b = 3 * a → c = 2 * b → d = 4 * c → a + b + c + d = 156) :
    x = 156 / 34 :=
by
  have h2 : 34 * x = 156 := sorry  -- This is to setup the intermediate equation
  exact eq_div_of_mul_eq' h2

end marble_problem_l66_66169


namespace borel_sigma_algebra_generated_by_open_balls_not_borel_sigma_algebra_generated_by_open_balls_non_separable_l66_66640

variables {E : Type*} [metric_space E]

-- Definition of a separable space
def is_separable (E : Type*) [metric_space E] : Prop :=
  ∃ (D : set E), countable D ∧ ∀ x : E, ∃ d ∈ D, ∀ r > 0, ∃ y ∈ D, y ∈ metric.ball d r

-- Definition of the Borel sigma-algebra
def borel_sigma_algebra (E : Type*) [metric_space E] :=
  measurable_space.generate_from {s | is_open s}

-- Statement to prove
theorem borel_sigma_algebra_generated_by_open_balls {E : Type*} [metric_space E] :
  is_separable E → 
  borel_sigma_algebra E = measurable_space.generate_from {s | 
    ∃ (x : E) (r : ℚ), metric.ball x (real.of_rat r)} :=
sorry

theorem not_borel_sigma_algebra_generated_by_open_balls_non_separable {E : Type*} [metric_space E] :
  ¬is_separable E → 
  ¬borel_sigma_algebra E = measurable_space.generate_from {s | 
    ∃ (x : E) (r : ℚ), metric.ball x (real.of_rat r)} :=
sorry

end borel_sigma_algebra_generated_by_open_balls_not_borel_sigma_algebra_generated_by_open_balls_non_separable_l66_66640


namespace friends_recycled_pounds_l66_66366

-- Definitions for the given conditions
def pounds_per_point : ℕ := 4
def paige_recycled : ℕ := 14
def total_points : ℕ := 4

-- The proof statement
theorem friends_recycled_pounds :
  ∃ p_friends : ℕ, 
  (paige_recycled / pounds_per_point) + (p_friends / pounds_per_point) = total_points 
  → p_friends = 4 := 
sorry

end friends_recycled_pounds_l66_66366


namespace sum_of_ages_l66_66742

-- Definitions
def YoungerBrotherAge : ℕ := 27
def OlderBrotherAge (Y : ℕ) : ℕ := 3 * (Y - 10)
def totalAge (Y O : ℕ) := Y + O

-- Hypothesis
variables (Y : ℕ) (O : ℕ)
hypothesis (hY : Y = YoungerBrotherAge)
hypothesis (hO : Y = 3 * (O - 10))

-- Theorem
theorem sum_of_ages : totalAge 27 (OlderBrotherAge 27) = 78 :=
by
  rw YoungerBrotherAge
  rw OlderBrotherAge
  sorry

end sum_of_ages_l66_66742


namespace curve_and_area_l66_66561

noncomputable def curve_parametric_eq (α: ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sin α)

theorem curve_and_area (ρ θ: ℝ) (h : 0 < θ ∧ θ < Real.pi/2) :
  ∃ α ∈ (Set.Ioo 0 (Real.pi/2)),
    (curve_parametric_eq α).1 = 2 * Real.cos θ ∧
    (curve_parametric_eq α).2 = Real.sin θ ∧
    let A := (2, 0) in
    let B := (0, 1) in
    let O := (0, 0) in
    let P := (2 * Real.cos θ, Real.sin θ) in
    let area_OAPB := Real.cos θ + Real.sin θ in
    (by sorry : area_OAPB = Real.sqrt 2) :=
begin
  sorry
end

end curve_and_area_l66_66561


namespace find_common_ratio_l66_66573

noncomputable def a : ℕ → ℝ
noncomputable def q : ℝ

axiom geom_seq (n : ℕ) : a n = a 1 * q ^ (n - 1)
axiom a2 : a 2 = 2
axiom a6 : a 6 = (1 / 8)

theorem find_common_ratio : (q = 1 / 2) ∨ (q = -1 / 2) :=
by
  sorry

end find_common_ratio_l66_66573


namespace milk_butterfat_problem_l66_66444

variable (x : ℝ)

def butterfat_10_percent (x : ℝ) := 0.10 * x
def butterfat_35_percent_in_8_gallons : ℝ := 0.35 * 8
def total_milk (x : ℝ) := x + 8
def total_butterfat (x : ℝ) := 0.20 * (x + 8)

theorem milk_butterfat_problem 
    (h : butterfat_10_percent x + butterfat_35_percent_in_8_gallons = total_butterfat x) : x = 12 :=
by
  sorry

end milk_butterfat_problem_l66_66444


namespace first_positions_of_one_half_nth_occurrence_of_one_half_first_occurrence_of_p_over_q_l66_66877

theorem first_positions_of_one_half :
  ∃ n₁ n₂ n₃ n₄ n₅ : ℕ,
    sequence_pos_frac n₁ (1/2) = 3 ∧ 
    sequence_pos_frac n₂ (1/2) = 14 ∧ 
    sequence_pos_frac n₃ (1/2) = 34 ∧ 
    sequence_pos_frac n₄ (1/2) = 63 ∧ 
    sequence_pos_frac n₅ (1/2) = 101 := sorry

theorem nth_occurrence_of_one_half (n : ℕ) :
  sequence_pos_frac n (1 / 2) = (9 * n^2 - 5 * n + 2) / 2 := sorry

theorem first_occurrence_of_p_over_q (p q : ℕ) (h_coprime : Nat.coprime p q) (h_lt : p < q) :
  sequence_pos_frac p q = ((p + q - 2) * (p + q - 1)) / 2 + q := sorry

-- Definition of sequence_pos_frac might depend on the specific implementation of the sequence
noncomputable def sequence_pos_frac (n : ℕ) (f : ℚ) : ℕ := sorry

end first_positions_of_one_half_nth_occurrence_of_one_half_first_occurrence_of_p_over_q_l66_66877


namespace find_min_point_l66_66960

def f (x : ℝ) : ℝ := x * Real.exp x

theorem find_min_point : ∀ x : ℝ, (f (-1) ≤ f x) :=
begin
  -- Proof goes here. we add sorry to skip the proof.
  sorry
end

end find_min_point_l66_66960


namespace dihedral_angle_range_l66_66256

noncomputable def dihedral_angle_condition (α β : Type) [plane α] [plane β] (A : point α) (B : point β) (l : line α β) :=
  A ∉ l ∧ B ∉ l ∧ ∃ θ1 θ2 : ℝ, 0 < θ1 + θ2 ∧ θ1 + θ2 < π / 2

theorem dihedral_angle_range (α β : Type) [plane α] [plane β] 
 (A : point α) (B : point β) (l : line α β) :
  dihedral_angle_condition α β A B l → 
  0 < θ1 + θ2 ∧ θ1 + θ2 < π / 2 :=
begin
  intros h,
  cases h with hA hRest,
  cases hRest with hB hθ,
  exact hθ,
end

end dihedral_angle_range_l66_66256


namespace log_equation_solution_l66_66937

theorem log_equation_solution (y : ℝ) (h1 : y < 1) (h2 : (Real.log y / Real.log 10)^2 - Real.log₁₀ y^3 = 75) :
  (Real.log y / Real.log 10)^3 - Real.log₁₀ (y^4) = (2808 - 336 * Real.sqrt 309) / 8 - 6 + 2 * Real.sqrt 309 := 
  sorry

end log_equation_solution_l66_66937


namespace log_sum_l66_66523

theorem log_sum (a b : ℝ) (h₁ : log 5 625 = a) (h₂ : log 5 (1/25) = b) : a + b = 2 := by
  have ha : a = 4 := by sorry
  have hb : b = -2 := by sorry
  rw [ha, hb]
  rfl

end log_sum_l66_66523


namespace verify_f_at_minus_one_l66_66346

def f (a b c : ℝ) (x : ℝ) := a * (Real.tan x)^3 - b * Real.sin (3 * x) + c * x + 7

theorem verify_f_at_minus_one (a b c : ℝ) (h : f a b c 1 = 14) : 
  ∃ d : ℝ, f a b c (-1) = d :=
by
  sorry

end verify_f_at_minus_one_l66_66346


namespace find_a_sum_l66_66529

theorem find_a_sum :
  let solutions_exist (a : ℤ) := ∃ x y, y - 2 = x * (x + 2) ∧ x^2 + a^2 + 2 * x = y * (2 * a - y)
  (filter solutions_exist (list.range 3) ∑ (fun x, x)) = 3 := 
by
  sorry

end find_a_sum_l66_66529


namespace minimize_cone_material_l66_66139

noncomputable def minimize_material (r : ℝ) : ℝ :=
  let h := 27 / r^2
  2 * π * r * h + π * r^2

theorem minimize_cone_material (r : ℝ) (V : ℝ) (h : ℝ) :
  (V = 27 * π) → 
  (h = 27 / r^2) → 
  r = 3 :=
sorry

end minimize_cone_material_l66_66139


namespace ajay_gain_l66_66788

-- Definitions of the problem conditions as Lean variables/constants.
variables (kg1 kg2 kg_total : ℕ) 
variables (price1 price2 price3 cost1 cost2 total_cost selling_price gain : ℝ)

-- Conditions of the problem.
def conditions : Prop :=
  kg1 = 15 ∧ 
  kg2 = 10 ∧ 
  kg_total = kg1 + kg2 ∧ 
  price1 = 14.5 ∧ 
  price2 = 13 ∧ 
  price3 = 15 ∧ 
  cost1 = kg1 * price1 ∧ 
  cost2 = kg2 * price2 ∧ 
  total_cost = cost1 + cost2 ∧ 
  selling_price = kg_total * price3 ∧ 
  gain = selling_price - total_cost 

-- The theorem for the gain amount proof.
theorem ajay_gain (h : conditions kg1 kg2 kg_total price1 price2 price3 cost1 cost2 total_cost selling_price gain) : 
  gain = 27.50 :=
  sorry

end ajay_gain_l66_66788


namespace first_train_length_is_290_l66_66483

/--
Conditions:
1. The speed of the first train is 120 kmph.
2. The speed of the second train is 80 kmph.
3. The time taken to cross each other is 9 seconds.
4. The length of the second train is 210.04 meters.

Question:
What is the length of the first train?
-/
def length_of_first_train
  (speed_first_train : ℕ)
  (speed_second_train : ℕ)
  (time_crossing : ℕ)
  (length_second_train : ℝ) : ℝ :=
  let relative_speed := (speed_first_train + speed_second_train) * 1000 / 3600 in
  let combined_length := relative_speed * time_crossing in
  combined_length - length_second_train

theorem first_train_length_is_290
  (h1 : speed_first_train = 120)
  (h2 : speed_second_train = 80)
  (h3 : time_crossing = 9)
  (h4 : length_second_train = 210.04) :
  length_of_first_train 120 80 9 210.04 = 290 :=
by
  sorry

end first_train_length_is_290_l66_66483


namespace instantaneous_acceleration_at_1_second_l66_66950

-- Assume the velocity function v(t) is given as:
def v (t : ℝ) : ℝ := t^2 + 2 * t + 3

-- We need to prove that the instantaneous acceleration at t = 1 second is 4 m/s^2.
theorem instantaneous_acceleration_at_1_second : 
  deriv v 1 = 4 :=
by 
  sorry

end instantaneous_acceleration_at_1_second_l66_66950


namespace students_taking_french_l66_66628

theorem students_taking_french 
  (Total : ℕ) (G : ℕ) (B : ℕ) (Neither : ℕ) (H_total : Total = 87)
  (H_G : G = 22) (H_B : B = 9) (H_neither : Neither = 33) : 
  ∃ F : ℕ, F = 41 := 
by
  sorry

end students_taking_french_l66_66628


namespace sum_odd_divisors_l66_66092

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l66_66092


namespace clarinet_players_count_l66_66317

-- Given weights and counts
def weight_trumpet : ℕ := 5
def weight_clarinet : ℕ := 5
def weight_trombone : ℕ := 10
def weight_tuba : ℕ := 20
def weight_drum : ℕ := 15
def count_trumpets : ℕ := 6
def count_trombones : ℕ := 8
def count_tubas : ℕ := 3
def count_drummers : ℕ := 2
def total_weight : ℕ := 245

-- Calculated known weight
def known_weight : ℕ :=
  (count_trumpets * weight_trumpet) +
  (count_trombones * weight_trombone) +
  (count_tubas * weight_tuba) +
  (count_drummers * weight_drum)

-- Weight carried by clarinets
def weight_clarinets : ℕ := total_weight - known_weight

-- Number of clarinet players
def number_of_clarinet_players : ℕ := weight_clarinets / weight_clarinet

theorem clarinet_players_count :
  number_of_clarinet_players = 9 := by
  unfold number_of_clarinet_players
  unfold weight_clarinets
  unfold known_weight
  calc
    (245 - (
      (6 * 5) + 
      (8 * 10) + 
      (3 * 20) + 
      (2 * 15))) / 5 = 9 := by norm_num

end clarinet_players_count_l66_66317


namespace find_f_at_10_l66_66729

noncomputable def f (x : ℝ) := sorry

axiom functional_equation 
  (x y : ℝ) 
  : f(x) + f(2 * x + y) + 7 * x * y + 3 * y ^ 2 = f(3 * x - y) + 3 * x ^ 2 + 2

theorem find_f_at_10 : f 10 = -123 := 
  sorry

end find_f_at_10_l66_66729


namespace complex_number_problem_l66_66560

-- Given definitions
def is_imaginary (z : ℂ) : Prop :=
  z.re = 0

-- The problem statement
theorem complex_number_problem :
  (∃ (b : ℝ), (z = 3 + b * Complex.i) ∧ is_imaginary ((1 + 3 * Complex.i) * z)) →
  (z = 3 + Complex.i) ∧
  let ω := (3 + Complex.i) / (2 + Complex.i) in
    ω = (7 / 5) - (1 / 5) * Complex.i ∧
    |ω| = Real.sqrt 2 :=
by
  sorry

end complex_number_problem_l66_66560


namespace tetrahedron_sum_l66_66332

theorem tetrahedron_sum :
  let edges := 6
  let corners := 4
  let faces := 4
  edges + corners + faces = 14 :=
by
  sorry

end tetrahedron_sum_l66_66332


namespace tangent_line_at_1_inequality_l66_66958

noncomputable def f : ℝ → ℝ := λ x, Real.exp x / x

def tangent_line_eq (x : ℝ) := (1 : ℝ) = 1 → (f 1) = Real.exp 1

theorem tangent_line_at_1 : tangent_line_eq 1 :=
by
  -- proof of the tangent line equation would go here
  sorry

theorem inequality (x : ℝ) (h : x ≠ 0) : 
  (1 / (x * f x) > 1 - x) :=
by
  -- proof of the inequality would go here
  sorry

end tangent_line_at_1_inequality_l66_66958


namespace alina_happy_count_l66_66849

def Material := {m : Fin 3 // m = 0 ∨ m = 1 ∨ m = 2} -- Types of materials: 0 -> Leather, 1 -> Silicone, 2 -> Plastic
def Keychain := {k : Fin 4 // k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3} -- Keychains: 0 -> bear, 1 -> dinosaur, 2 -> raccoon, 3 -> fairy
def Drawing := {d : Fin 3 // d = 0 ∨ d = 1 ∨ d = 2} -- Drawings: 0 -> moon, 1 -> sun, 2 -> clouds

-- Define the condition: The material and keychain specifications of each case
structure Case :=
  (material : Material)
  (keychain : Keychain)
  (drawing : Drawing)

-- Define the condition for Alina's happiness
def alina_happy (c1 c2 c3 : Case) : Prop :=
  c2.material = ⟨1, or.inr (or.inl rfl)⟩ ∧ -- Silicone with bear keychain in the middle
  c2.keychain = ⟨0, or.inl rfl⟩ ∧
  c1.material ≠ c2.material ∧ c2.material ≠ c3.material ∧ c3.material ≠ c1.material ∧
  c1.keychain ≠ c2.keychain ∧ c2.keychain ≠ c3.keychain ∧ c3.keychain ≠ c1.keychain ∧
  c1.drawing ≠ c2.drawing ∧ c2.drawing ≠ c3.drawing ∧ c3.drawing ≠ c1.drawing ∧
  ((c1.material = ⟨0, or.inl rfl⟩ ∧ c3.material = ⟨2, or.inr (or.inr rfl)⟩) ∨ -- Leather, Silicone, Plastic
   (c1.material = ⟨2, or.inr (or.inr rfl)⟩ ∧ c3.material = ⟨0, or.inl rfl⟩)) -- Plastic, Silicone, Leather

-- Proven statement that there are exactly 72 valid combinations for Alina to be happy
theorem alina_happy_count : 
  ∃ (count : ℕ), count = 72 ∧ 
  count = Fintype.card {c : (Case × Case × Case) // alina_happy c.1 c.2 c.3} :=
by {
  sorry
}

end alina_happy_count_l66_66849


namespace compare_sums_of_roots_l66_66871

theorem compare_sums_of_roots :
  (sqrt 11 + sqrt 7) > (sqrt 13 + sqrt 5) :=
by
  have h1 : (sqrt 11 + sqrt 7)^2 = 18 + 2 * sqrt 77 := sorry,
  have h2 : (sqrt 13 + sqrt 5)^2 = 18 + 2 * sqrt 65 := sorry,
  have h3 : sqrt 77 > sqrt 65 := sorry,
  sorry

end compare_sums_of_roots_l66_66871


namespace smallest_enclosing_sphere_radius_l66_66550

theorem smallest_enclosing_sphere_radius :
  ∃ r : ℝ, (∀ (s₁ s₂ s₃ s₄ : ball (0 : ℝ^3) 1), 
  s₁.center ≠ s₂.center ∧ s₁.center ≠ s₃.center ∧ s₁.center ≠ s₄.center ∧ 
  s₂.center ≠ s₃.center ∧ s₂.center ≠ s₄.center ∧ s₃.center ≠ s₄.center ∧
  (dist s₁.center s₂.center) = 2 ∧ (dist s₁.center s₃.center) = 2 ∧ (dist s₁.center s₄.center) = 2 ∧ 
  (dist s₂.center s₃.center) = 2 ∧ (dist s₂.center s₄.center) = 2 ∧ (dist s₃.center s₄.center) = 2) 
  → r = sqrt (3 / 2) + 1 := by
  sorry

end smallest_enclosing_sphere_radius_l66_66550


namespace centroid_of_cone_intersection_l66_66709

-- Define the given geometry and properties
variables {C A B P Q S : Type} [Point C] [Point A] [Point B] [Point P] [Point Q] [Point S]

-- Conditions: P and Q points on CA and CB respectively with given ratio
def on_CA (P : C) (A : A) : Prop := true
def on_CB (Q : C) (B : B) : Prop := true
def ratio_CP_PA (P : P) (A : A) (C : C) : Prop := true -- e.g., the condition that CP / PA = 3 / 2
def ratio_CQ_QB (Q : Q) (B : B) (C : C) : Prop := true -- e.g., the condition that CQ / QB = 3 / 2

-- Prove: The intersection point S of lines BP and AQ is the centroid
theorem centroid_of_cone_intersection (h1 : on_CA P A)
                                      (h2 : on_CB Q B)
                                      (h3 : ratio_CP_PA P A C)
                                      (h4 : ratio_CQ_QB Q B C) :
  S := sorry

end centroid_of_cone_intersection_l66_66709


namespace checker_bound_l66_66381

theorem checker_bound (n : ℕ) (x : ℕ) 
  (h1 : ∀ i j, (i < n → j < n → ¬ (i, j) ≠ checker → 
    ((i + 1, j) = checker ∨ (i - 1, j) = checker ∨ (i, j + 1) = checker ∨ (i, j - 1) = checker)))
  (h2 : ∀ (c1 c2 : ℕ × ℕ), (c1 = checker ∧ c2 = checker) → (∃ s : List (ℕ × ℕ), (s.head = c1 ∧ s.last = c2 ∧ ∀ (first second : ℕ × ℕ), first ∈ s → second ∈ s → adj_edge first second)))
  : x ≥ (n^2 - 2) / 3 :=
sorry

end checker_bound_l66_66381


namespace petya_vasya_sum_eq_l66_66559

-- Define the set of 15 distinct integers
variable {X : Type*} [DecidableEq X]

theorem petya_vasya_sum_eq (a : Fin 15 → ℤ) (ha : Function.Injective a) :
  ∃ T : ℤ, (∀ (s7 : Finset (Fin 15)) (h7 : s7.card = 7),
    let s8 := (Finset.univ.erase_all (s7.toList : List _)).toFinset in
    s8.card = 8 ∧ s8.sum a = T - s7.sum a) :=
by sorry

end petya_vasya_sum_eq_l66_66559


namespace sum_odd_divisors_l66_66094

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l66_66094


namespace frank_hours_per_day_l66_66551

theorem frank_hours_per_day (total_hours : ℝ) (days : ℝ) (h1 : total_hours = 8.0) (h2 : days = 4.0) : 
(total_hours / days) = 2.0 :=
by 
  rw [h1, h2]
  norm_num
  sorry

end frank_hours_per_day_l66_66551


namespace max_height_l66_66841

def height (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

theorem max_height : ∃ t : ℝ, ∀ t' : ℝ, height t' ≤ height t ∧ height t = 40 :=
by
  use 1
  sorry

end max_height_l66_66841


namespace find_cos_sum_l66_66306

-- Defining the conditions based on the problem
variable (P A B C D : Type) (α β : ℝ)

-- Assumptions stating the given conditions
def regular_quadrilateral_pyramid (P A B C D : Type) : Prop :=
  -- Placeholder for the exact definition of a regular quadrilateral pyramid
  sorry

def dihedral_angle_lateral_base (P A B C D : Type) (α : ℝ) : Prop :=
  -- Placeholder for the exact property stating the dihedral angle between lateral face and base is α
  sorry

def dihedral_angle_adjacent_lateral (P A B C D : Type) (β : ℝ) : Prop :=
  -- Placeholder for the exact property stating the dihedral angle between two adjacent lateral faces is β
  sorry

-- The final theorem that we want to prove
theorem find_cos_sum (P A B C D : Type) (α β : ℝ)
  (H1 : regular_quadrilateral_pyramid P A B C D)
  (H2 : dihedral_angle_lateral_base P A B C D α)
  (H3 : dihedral_angle_adjacent_lateral P A B C D β) :
  2 * Real.cos β + Real.cos (2 * α) = -1 :=
sorry

end find_cos_sum_l66_66306


namespace print_rolls_sold_l66_66461

-- Defining the variables and conditions
def num_sold := 480
def total_amount := 2340
def solid_price := 4
def print_price := 6

-- Proposed theorem statement
theorem print_rolls_sold (S P : ℕ) (h1 : S + P = num_sold) (h2 : solid_price * S + print_price * P = total_amount) : P = 210 := sorry

end print_rolls_sold_l66_66461


namespace Jane_age_l66_66669

theorem Jane_age (J A : ℕ) (h1 : J + A = 54) (h2 : J - A = 22) : A = 16 := 
by 
  sorry

end Jane_age_l66_66669


namespace initial_average_income_l66_66140

variable (A : ℝ) (init_members : ℝ) (deceased_income : ℝ) (new_avg_income : ℝ) (new_members : ℝ)

theorem initial_average_income :
  init_members = 4 ∧ new_avg_income = 650 ∧ deceased_income = 1410 ∧ new_members = 3 →
  A = 840 :=
begin
  intros h,
  -- let extract the conditions from our hypothesis h
  cases h with h1 h234,
  cases h234 with h2 h34,
  cases h34 with h3 h4,
  -- use the conditions to derive the result
  have eq1 : 4 * A - 1410 = 3 * 650 :=
    by rw [h1, h2, h3, h4],
  have eq2 : 4 * A - 1410 = 1950 :=
    by rw eq1,
  have eq3 : 4 * A = 1950 + 1410 :=
    by linarith eq2,
  have eq4 : 4 * A = 3360 :=
    by rw eq3,
  have eq5 : A = 3360 / 4 :=
    by rw [eq4, div_eq_mul_inv],
  norm_num at eq5,
  exact eq5,
end

end initial_average_income_l66_66140


namespace negation_of_all_students_are_punctual_l66_66398

variable (Student : Type)
variable (student : Student → Prop)
variable (punctual : Student → Prop)

theorem negation_of_all_students_are_punctual :
  ¬ (∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) := by
  sorry

end negation_of_all_students_are_punctual_l66_66398


namespace michael_saves_more_using_promotion_A_l66_66840

theorem michael_saves_more_using_promotion_A :
  let price := 50 in
  let promoA_cost := price + (price - 0.30 * price) in
  let promoB_cost := price + (price - 0.20 * price) in
  promoB_cost - promoA_cost = 5 :=
by 
  sorry

end michael_saves_more_using_promotion_A_l66_66840


namespace question1_question2_question3_l66_66700

/-- Question 1 -/
theorem question1 
  (A B C D I₁ I₂ O₁ O₂ P : Point) 
  (hD_on_BC : lies_on D (side B C)) 
  (hI₁_incenter : is_incenter I₁ (triangle (A B D)))
  (hI₂_incenter : is_incenter I₂ (triangle (A C D)))
  (hO₁_circumcenter : is_circumcenter O₁ (triangle (A I₁ D)))
  (hO₂_circumcenter : is_circumcenter O₂ (triangle (A I₂ D)))
  (hP_intersection : P = intersection (line I₁ O₂) (line I₂ O₁))
  : perp (line P D) (line B C) := sorry

/-- Question 2 -/
theorem question2 
  (A I₁ D I₂ O₁ O₂ P : Point) 
  (h_angle_right : ∠ I₁ D I₂ = π / 2) 
  (h_angle_obtuse1 : is_obtuse_angle (∠ D I₁ A)) 
  (h_angle_obtuse2 : is_obtuse_angle (∠ D I₂ A))
  (hO₁_circumcenter : is_circumcenter O₁ (triangle (A I₁ D)))
  (hO₂_circumcenter : is_circumcenter O₂ (triangle (A I₂ D)))
  (hP_intersection : P = intersection (line I₁ O₂) (line I₂ O₁))
  : ∠ I₁ D P = ∠ P D I₂ := sorry

/-- Question 3 -/
theorem question3 
  (D E K F I₁ I₂ X Y O₁ O₂ : Point) 
  (h_rectangle : is_rectangle (quadrilateral D E K F)) 
  (hI₁_on_DE : lies_on I₁ (line D E))
  (hI₂_on_DF : lies_on I₂ (line D F))
  (hX_midpoint : X = midpoint (segment D I₁))
  (hY_midpoint : Y = midpoint (segment D I₂))
  (hO₁_intersection : O₁ = intersection (parallel_through (line X Y) (side D F)) (line E F))
  (hO₂_intersection : O₂ = intersection (parallel_through (line X Y) (side D E)) (line E F)) 
  : coincident (line I₁ O₂) (line I₂ O₁) (line D K) := sorry

end question1_question2_question3_l66_66700


namespace total_ticket_sales_l66_66142

-- Define the parameters and the theorem to be proven.
theorem total_ticket_sales (total_people : ℕ) (kids : ℕ) (adult_ticket_price : ℕ) (kid_ticket_price : ℕ) 
  (adult_tickets := total_people - kids) 
  (adult_ticket_sales := adult_tickets * adult_ticket_price) 
  (kid_ticket_sales := kids * kid_ticket_price) : 
  total_people = 254 → kids = 203 → adult_ticket_price = 28 → kid_ticket_price = 12 → 
  adult_ticket_sales + kid_ticket_sales = 3864 := 
by
  intros h1 h2 h3 h4
  sorry

end total_ticket_sales_l66_66142


namespace tangent_line_equation_l66_66274

/-- Definition of the function -/
def f (x : ℝ) : ℝ := x^3 - 3 * x

/-- The derivative of the function -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 3

/-- Point A (0, 16) -/
def A : ℝ × ℝ := (0, 16)

/-- Equation of the tangent line passing through point A -/
def tangent_line (x₀ y₀ : ℝ) : (ℝ → ℝ) :=
  λ x, y₀ + f' x₀ * (x - x₀)

/-- Prove that the equation of the tangent line to the curve y = x^3 - 3x that
    passes through point A(0, 16) is 9x - y + 16 = 0 -/
theorem tangent_line_equation :
  ∃ (x₀ y₀ : ℝ), y₀ = f x₀ ∧ tangent_line x₀ y₀ = (λ x, 3 * (x₀^2 - 1) * (x - x₀) + y₀) ∧ 
  tangent_line x₀ (f x₀) 0 = 16 ∧ 9 * x + (-y + 16) = 0 :=
by
  sorry

end tangent_line_equation_l66_66274


namespace three_consecutive_odds_l66_66410

theorem three_consecutive_odds (x : ℤ) (h3 : x + 4 = 133) : 
  x + (x + 4) = 3 * (x + 2) - 131 := 
by {
  sorry
}

end three_consecutive_odds_l66_66410


namespace solve_for_x_l66_66727

theorem solve_for_x :
  exists x : ℝ, 11.98 * 11.98 + 11.98 * x + 0.02 * 0.02 = (11.98 + 0.02) ^ 2 ∧ x = 0.04 :=
by
  sorry

end solve_for_x_l66_66727


namespace days_B_to_complete_remaining_work_l66_66455

/-- 
  Given that:
  - A can complete a work in 20 days.
  - B can complete the same work in 12 days.
  - A and B worked together for 3 days before A left.
  
  We need to prove that B will require 7.2 days to complete the remaining work alone. 
--/
theorem days_B_to_complete_remaining_work : 
  (∃ (A_rate B_rate combined_rate work_done_in_3_days remaining_work d_B : ℚ), 
   A_rate = (1 / 20) ∧
   B_rate = (1 / 12) ∧
   combined_rate = A_rate + B_rate ∧
   work_done_in_3_days = 3 * combined_rate ∧
   remaining_work = 1 - work_done_in_3_days ∧
   d_B = remaining_work / B_rate ∧
   d_B = 7.2) := 
by 
  sorry

end days_B_to_complete_remaining_work_l66_66455


namespace length_of_parallel_line_l66_66718

theorem length_of_parallel_line (base height : ℝ) (h_base : base = 20) (h_height : height = 10)
  (area_division : ∀ (A B C : ℝ), A ≠ B ∧ B ≠ C ∧ A ≠ C → A + B + C = base * height / 2 * 1/4 :=
  begin
    sorry
  end) :
  ∃ PQ : ℝ, PQ = 10 :=
begin
  use 10,
  sorry
end

end length_of_parallel_line_l66_66718


namespace jogging_friends_probability_l66_66423

theorem jogging_friends_probability
  (n p q r : ℝ)
  (h₀ : 1 > 0) -- Positive integers condition
  (h₁ : n = p - q * Real.sqrt r)
  (h₂ : ∀ prime, ¬ (r ∣ prime ^ 2)) -- r is not divisible by the square of any prime
  (h₃ : (60 - n)^2 = 1800) -- Derived from 50% meeting probability
  (h₄ : p = 60) -- Identified values from solution
  (h₅ : q = 30)
  (h₆ : r = 2) : 
  p + q + r = 92 :=
by
  sorry

end jogging_friends_probability_l66_66423


namespace find_x_if_delta_phi_x_eq_3_l66_66555

def delta (x : ℚ) : ℚ := 2 * x + 5
def phi (x : ℚ) : ℚ := 9 * x + 6

theorem find_x_if_delta_phi_x_eq_3 :
  ∃ (x : ℚ), delta (phi x) = 3 ∧ x = -7/9 := by
sorry

end find_x_if_delta_phi_x_eq_3_l66_66555


namespace person_speed_is_9_point_6_kmph_l66_66786

-- Definitions / conditions from the problem
def distance_meters : ℝ := 800
def time_minutes : ℝ := 5

-- Conversion factors
def meters_to_kilometers : ℝ := 1 / 1000
def minutes_to_hours : ℝ := 1 / 60

-- The proof statement
theorem person_speed_is_9_point_6_kmph :
  (distance_meters * meters_to_kilometers) / (time_minutes * minutes_to_hours) = 9.6 := by
  sorry

end person_speed_is_9_point_6_kmph_l66_66786


namespace find_number_l66_66468

/--
A number is added to 5, then multiplied by 5, then subtracted by 5, and then divided by 5. 
The result is still 5. Prove that the number is 1.
-/
theorem find_number (x : ℝ) (h : ((5 * (x + 5) - 5) / 5 = 5)) : x = 1 := 
  sorry

end find_number_l66_66468


namespace fraction_of_40_l66_66035

theorem fraction_of_40 : (3 / 4) * 40 = 30 :=
by
  -- We'll add the 'sorry' here to indicate that this is the proof part which is not required.
  sorry

end fraction_of_40_l66_66035


namespace quadrilateral_properties_l66_66656

variables {P A B C D : Type}

-- Assume there is a convex quadrilateral ABCD
variable [has_area P A B C D]

-- Statement that encodes the problem's conditions and the required proof
theorem quadrilateral_properties (H : ∀ P, area (triangle A B P) = area (triangle B C P) ∧ 
                                             area (triangle B C P) = area (triangle C D P) ∧ 
                                             area (triangle C D P) = area (triangle D A P)) :
  (diagonals_bisect A C B D) ∧ (unique_point A B C D P) :=
sorry

end quadrilateral_properties_l66_66656


namespace part1_part2_l66_66961

variable {R : Type} [LinearOrderedField R]

def f (x : R) : R := abs (x - 2) + 2
def g (m : R) (x : R) : R := m * abs x

theorem part1 (x : R) : f x > 5 ↔ x < -1 ∨ x > 5 := by
  sorry

theorem part2 (m : R) : (∀ x : R, f x ≥ g m x) → m ∈ Set.Iic (1 : R) := by
  sorry

end part1_part2_l66_66961


namespace non_triang_solutions_l66_66650

open Real

theorem non_triang_solutions :
  (∃ a b A, a = 30 ∧ b = 25 ∧ A = 150 ∧
  1 = 1) ∧
  (¬∃ a c B, a = 9 ∧ c = 10 ∧ B = 60 ∧
  1 = 1) ∧
  (¬∃ a b A, a = 6 ∧ b = 9 ∧ A = 45 ∧
  2 = 2) ∧
  (¬∃ a b A, a = 7 ∧ b = 14 ∧ A = 30 ∧
  2 = 2) :=
begin
  sorry -- proof not required
end

end non_triang_solutions_l66_66650


namespace roots_not_real_l66_66198

theorem roots_not_real (k : ℝ) (h : 48 - 12 * k < 0) :
  ∀ x : ℂ, (3 * x ^ 2 - 4 * x * (sqrt 3 : ℂ) + (k : ℂ) = 0) → x.re = 0 :=
by
  sorry

end roots_not_real_l66_66198


namespace tenth_integer_from_permutations_l66_66003

theorem tenth_integer_from_permutations : ∃ n : ℕ, nth_permutation [1, 2, 5, 6] n = 2561 :=
by
  sorry

def nth_permutation (digits : List ℕ) (n : ℕ) : ℕ :=
  sorry

end tenth_integer_from_permutations_l66_66003


namespace find_sum_of_x_and_reciprocal_l66_66945

theorem find_sum_of_x_and_reciprocal (x : ℝ) (hx_condition : x^3 + 1/x^3 = 110) : x + 1/x = 5 := 
sorry

end find_sum_of_x_and_reciprocal_l66_66945


namespace intersection_M_N_l66_66690

open Set

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | x > 0 ∧ x < 2}

theorem intersection_M_N : M ∩ N = {1} :=
by {
  sorry
}

end intersection_M_N_l66_66690


namespace caleb_burgers_l66_66442
noncomputable def caleb_double_burgers_count (total_cost single_cost double_cost total_burgers : ℝ) (S D : ℝ) : Prop :=
  (single_cost * S + double_cost * D = total_cost) ∧ (S + D = total_burgers) → D = 49

theorem caleb_burgers:
  caleb_double_burgers_count 74.50 1.00 1.50 50 :=
begin
  intros h,
  sorry
end

end caleb_burgers_l66_66442


namespace derivative_of_x3_mul_3x_l66_66534

theorem derivative_of_x3_mul_3x :
  ∀ x : ℝ, deriv (λ x, x^3 * (3:ℝ)^x) x = x^2 * (3:ℝ)^x * (3 + x * Real.log 3) :=
by
  intro x
  sorry

end derivative_of_x3_mul_3x_l66_66534


namespace n_factorial_divisibility_l66_66911

theorem n_factorial_divisibility:
  (number_of_valid_n : ℕ) (h_number_of_valid_n : number_of_valid_n = 34) :=
  ∀ n : ℕ, n ≤ 50 → n > 0 → (n! % (n * (n + 1) / 2) = 0) ↔ ((n + 1) divides (2 * (n - 1)!)) :=
  sorry

end n_factorial_divisibility_l66_66911


namespace find_circumcenter_l66_66328

noncomputable def circumcenter_coordinates (a b c : ℝ) (A B C : ℝ × ℝ) 
  (O : ℝ × ℝ) (x y z : ℝ) : Prop :=
  O = (x * A + y * B + z * C) ∧ 
  x + y + z = 1 ∧ 
  a = 8 ∧ 
  b = 10 ∧ 
  c = 6 ∧ 
  (a * a + b * b - c * c) * c * c * A =
    (b * b + c * c - a * a) * a * a * B + 
    (c * c + a * a - b * b) * b * b * C

theorem find_circumcenter (A B C O : ℝ × ℝ) (a b c : ℝ) 
  (x y z : ℝ) : 
  circumcenter_coordinates a b c A B C O x y z → 
  (x, y, z) = (1/2 : ℝ, 1/2 : ℝ, 0) :=
sorry

end find_circumcenter_l66_66328


namespace average_after_17th_inning_l66_66780

variable (A : ℕ)

-- Definition of total runs before the 17th inning
def total_runs_before := 16 * A

-- Definition of new total runs after the 17th inning
def total_runs_after := total_runs_before A + 87

-- Definition of new average after the 17th inning
def new_average := A + 4

-- Definition of new total runs in terms of new average
def new_total_runs := 17 * new_average A

-- The statement we want to prove
theorem average_after_17th_inning : total_runs_after A = new_total_runs A → new_average A = 23 := by
  sorry

end average_after_17th_inning_l66_66780


namespace sum_cos_eq_minus_one_sum_sin_eq_zero_l66_66805

theorem sum_cos_eq_minus_one (n : ℕ) (h : n > 1) : 
  (∑ k in Finset.range (n - 1), Real.cos (2 * (k + 1) * Real.pi / n)) = -1 :=
sorry

theorem sum_sin_eq_zero (n : ℕ) (h : n > 1) : 
  (∑ k in Finset.range (n - 1), Real.sin (2 * (k + 1) * Real.pi / n)) = 0 :=
sorry

end sum_cos_eq_minus_one_sum_sin_eq_zero_l66_66805


namespace Keith_spent_correct_amount_l66_66334

noncomputable def costDigimon : ℝ := 4 * 4.45
noncomputable def costSoccer : ℝ := 2 * 3.75
noncomputable def costBaseball : ℝ := 1 * 6.06
noncomputable def costMarvel : ℝ := 3 * 5.25

noncomputable def totalCostBeforeDiscount : ℝ := costDigimon + costSoccer + costBaseball + costMarvel
noncomputable def discount : ℝ := 0.05 * totalCostBeforeDiscount
noncomputable def totalAfterDiscount : ℝ := totalCostBeforeDiscount - discount
noncomputable def salesTax : ℝ := 0.03 * totalAfterDiscount
noncomputable def finalAmount : ℝ := totalAfterDiscount + salesTax

theorem Keith_spent_correct_amount : round finalAmount = 46.10 := 
by 
  sorry

end Keith_spent_correct_amount_l66_66334


namespace remaining_uncracked_seashells_l66_66421

-- Definitions based on conditions given in the problem
def tom_seashells : ℕ := 15
def fred_seashells : ℕ := 43
def cracked_seashells : ℕ := 29
def give_away_percentage : ℕ := 40

-- The statement to prove
theorem remaining_uncracked_seashells :
  let total_seashells := tom_seashells + fred_seashells in
  let uncracked_seashells := total_seashells - cracked_seashells in
  let to_give_away := (give_away_percentage * uncracked_seashells) / 100 in
  uncracked_seashells - to_give_away = 18 :=
by
  sorry

end remaining_uncracked_seashells_l66_66421


namespace min_abs_sum_of_products_l66_66350

noncomputable def g (x : ℝ) : ℝ := x^4 + 10*x^3 + 29*x^2 + 30*x + 9

theorem min_abs_sum_of_products (w : Fin 4 → ℝ) (h_roots : ∀ i, g (w i) = 0)
  : ∃ a b c d : Fin 4, a ≠ b ∧ c ≠ d ∧ (∀ i j, i ≠ j → a ≠ i ∧ b ≠ i ∧ c ≠ i ∧ d ≠ i → a ≠ j ∧ b ≠ j ∧ c ≠ j ∧ d ≠ j) ∧
    |w a * w b + w c * w d| = 6 :=
sorry

end min_abs_sum_of_products_l66_66350


namespace cookies_per_box_correct_l66_66181

variable (cookies_per_box : ℕ)

-- Define the conditions
def morning_cookie : ℕ := 1 / 2
def bed_cookie : ℕ := 1 / 2
def day_cookies : ℕ := 2
def daily_cookies := morning_cookie + bed_cookie + day_cookies

def days : ℕ := 30
def total_cookies := days * daily_cookies

def boxes : ℕ := 2
def total_cookies_in_boxes : ℕ := cookies_per_box * boxes

-- Theorem we want to prove
theorem cookies_per_box_correct :
  total_cookies_in_boxes = 90 → cookies_per_box = 45 :=
by
  sorry

end cookies_per_box_correct_l66_66181


namespace coloring_possible_if_divisible_by_three_divisible_by_three_if_coloring_possible_l66_66832

/- The problem's conditions and questions rephrased for Lean:
  1. Prove: if \( n \) is divisible by 3, then a valid coloring is possible.
  2. Prove: if a valid coloring is possible, then \( n \) is divisible by 3.
-/

def is_colorable (n : ℕ) : Prop :=
  ∃ (colors : Fin 3 → Fin n → Fin 3),
    ∀ (i j : Fin n), i ≠ j → (colors 0 i ≠ colors 0 j ∧ colors 1 i ≠ colors 1 j ∧ colors 2 i ≠ colors 2 j)

theorem coloring_possible_if_divisible_by_three (n : ℕ) (h : n % 3 = 0) : is_colorable n :=
  sorry

theorem divisible_by_three_if_coloring_possible (n : ℕ) (h : is_colorable n) : n % 3 = 0 :=
  sorry

end coloring_possible_if_divisible_by_three_divisible_by_three_if_coloring_possible_l66_66832


namespace reasoning_correct_l66_66115

def InductiveReasoning : Type := "part to whole"
def DeductiveReasoning : Type := "general to specific"
def AnalogicalReasoning : Type := "specific to specific"

theorem reasoning_correct :
  ("Inductive reasoning" = InductiveReasoning) ∧
  ("Deductive reasoning" = DeductiveReasoning) ∧
  ("Analogical reasoning" = AnalogicalReasoning) ->
  {1, 3, 5} = {1, 3, 5} :=
by
  intros h
  exact rfl

end reasoning_correct_l66_66115


namespace length_of_base_vessel_l66_66820

def volume_of_cube(edge: ℝ) : ℝ := edge ^ 3
def volume_displaced_by_cube(V_cube: ℝ) : ℝ := V_cube
def base_area(V_water: ℝ, water_level_rise: ℝ) : ℝ := V_water / water_level_rise
def length_of_base(area_base: ℝ, width: ℝ) : ℝ := area_base / width

theorem length_of_base_vessel
  (edge : ℝ) (water_level_rise : ℝ) (width : ℝ) (V_cube := volume_of_cube edge)
  (V_water := volume_displaced_by_cube V_cube) (A_base := base_area V_water water_level_rise)
  (length_base := length_of_base A_base width)
  (h_edge : edge = 10)
  (h_water_level_rise : water_level_rise = 3.3333333333333335)
  (h_width : width = 15) :
  length_base = 20 :=
by
  sorry

end length_of_base_vessel_l66_66820


namespace wsquared_values_l66_66606

noncomputable def find_wsquared (w : ℝ) : ℝ :=
if (2 * w + 17)^2 = (4 * w + 9) * (3 * w + 6) then w^2 else sorry

theorem wsquared_values (w : ℝ) :
  (2 * w + 17)^2 = (4 * w + 9) * (3 * w + 6) →
  (w^2 = 19.69140625 ∨ w^2 = 43.06640625) :=
by simplify
    sorry

end wsquared_values_l66_66606


namespace cos_B_given_sin_ratios_in_triangle_l66_66662

theorem cos_B_given_sin_ratios_in_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : sin A / a = sin B / b)
  (h2 : sin B / b = sin C / c)
  (h3 : sin A / sin B = 4 / 3)
  (h4 : sin B / sin C = 3 / 2)
  (h5 : a = 4 * x)
  (h6 : b = 3 * x)
  (h7 : c = 2 * x)
  (h8 : 0 < x)
  : cos B = 11 / 16 := sorry

end cos_B_given_sin_ratios_in_triangle_l66_66662


namespace plane_hit_probability_l66_66698

-- Define the probability of person A hitting the plane
def P_A : ℝ := 0.7

-- Define the probability of person B hitting the plane
def P_B : ℝ := 0.5

-- Main theorem stating the probability of the enemy plane being hit.
theorem plane_hit_probability : (1 - (1 - P_A) * (1 - P_B)) = 0.85 := 
by
  -- Sorry is added here to skip the proof.
  sorry

end plane_hit_probability_l66_66698


namespace prove_system_of_equations_l66_66636

variables (x y : ℕ)

def system_of_equations (x y : ℕ) : Prop :=
  x = 2*y + 4 ∧ x = 3*y - 9

theorem prove_system_of_equations :
  ∀ (x y : ℕ), system_of_equations x y :=
by sorry

end prove_system_of_equations_l66_66636


namespace total_surface_area_of_rearranged_solid_is_eight_l66_66137

/-- A cube with a volume of 1 cubic meter is cut into four layers by three parallel cuts along the vertical axis.

Conditions:
1. The first cut is 1/4 meter from the top.
2. The second cut is 1/6 meter below the first.
3. The third cut is 1/12 meter below the second.
4. The pieces are rearranged in the sequence of B, C, A, and D from top to bottom.

Proof that the total surface area of the new solid is 8 square meters. -/
theorem total_surface_area_of_rearranged_solid_is_eight :
  let V := 1 (unit : ℝ), 
      hA := 1 / 4, 
      hB := 1 / 6, 
      hC := 1 / 12, 
      hD := 1 - (hA + hB + hC), 
      front_and_back := 2 * 1, 
      top_and_bottom := 2 * 1, 
      sides := 2 * 1 
  in 2 * (front_and_back + top_and_bottom + sides) = 8 := sorry

end total_surface_area_of_rearranged_solid_is_eight_l66_66137


namespace required_run_rate_l66_66443

-- Definitions of the given conditions
def run_rate_first_10_overs := 3.2
def total_overs := 50
def target_score := 272
def overs_first_part := 10
def remaining_overs := total_overs - overs_first_part

-- Statement of the theorem to be proven
theorem required_run_rate :
  (target_score - run_rate_first_10_overs * overs_first_part) / remaining_overs = 6 := by
  sorry

end required_run_rate_l66_66443


namespace future_cup_defensive_analysis_l66_66643

variables (avg_A : ℝ) (std_dev_A : ℝ) (avg_B : ℝ) (std_dev_B : ℝ)

-- Statement translations:
-- A: On average, Class B has better defensive skills than Class A.
def stat_A : Prop := avg_B < avg_A

-- C: Class B sometimes performs very well in defense, while other times it performs relatively poorly.
def stat_C : Prop := std_dev_B > std_dev_A

-- D: Class A rarely concedes goals.
def stat_D : Prop := avg_A <= 1.9 -- It's implied that 'rarely' indicates consistency and a lower average threshold, so this represents that.

theorem future_cup_defensive_analysis (h_avg_A : avg_A = 1.9) (h_std_dev_A : std_dev_A = 0.3) 
  (h_avg_B : avg_B = 1.3) (h_std_dev_B : std_dev_B = 1.2) :
  stat_A avg_A avg_B ∧ stat_C std_dev_A std_dev_B ∧ stat_D avg_A :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end future_cup_defensive_analysis_l66_66643


namespace t_50_mod_6_l66_66511

def seq_T : ℕ → ℕ 
| 1     := 11
| (n+1) := 11 ^ seq_T n

theorem t_50_mod_6 : seq_T 50 % 6 = 5 :=
by
  /-
  We need to use the properties derived in the solution to model this naturally 
  in Lean, skipping the detailed induction proof.
  -/
  sorry

end t_50_mod_6_l66_66511


namespace correct_sqrt_calculation_l66_66851

theorem correct_sqrt_calculation :
  (sqrt 2 * sqrt 3 = sqrt 6) ∧ 
  (sqrt 8 ≠ 3 * sqrt 2) ∧ 
  (sqrt 2 + sqrt 3 ≠ sqrt 6) ∧ 
  (sqrt 4 / sqrt 2 ≠ 2) :=
by
  sorry

end correct_sqrt_calculation_l66_66851


namespace minimize_product_of_roots_of_quadratic_eq_l66_66586

theorem minimize_product_of_roots_of_quadratic_eq (k : ℝ) :
  (∃ x y : ℝ, 2 * x^2 + 5 * x + k = 0 ∧ 2 * y^2 + 5 * y + k = 0) 
  → k = 25 / 8 :=
sorry

end minimize_product_of_roots_of_quadratic_eq_l66_66586


namespace simplify_expression_l66_66266

-- Given conditions
def a : ℝ := 0.04

-- Mathematical expression to be simplified
def expression (a b c : ℝ) : ℝ :=
  1.24 * (sqrt ((a * b * c + 4) / a + 4 * sqrt (b * c / a)) / (sqrt (a * b * c) + 2))

-- Theorem stating the expression simplifies to a known value
theorem simplify_expression (b c : ℝ) : expression 0.04 b c = 6.2 := by
  sorry

end simplify_expression_l66_66266


namespace correct_propositions_l66_66268

def proposition1 (P : Prop) : Prop :=
  ¬(∀ x : ℝ, cos x > 0) ↔ ∃x : ℝ, cos x ≤ 0

def proposition2 (f : ℝ → ℝ) (a : ℝ) : Prop :=
  0 < a ∧ a < 1 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f(x) = x^2 + a^x - 3

def proposition3 (f : ℝ → ℝ) : Prop :=
  Icc (-π / 12) (5 * π / 12) ⊆ ({x : ℝ | ∀ y z : ℝ, (Icc y z ⊆ Icc (-π / 12) (5 * π / 12) → f y ≤ f z)}

def proposition4 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, x > 0 → deriv f x > 0) → (∀ x : ℝ, x < 0 → deriv f x < 0)

theorem correct_propositions : 
  {n | n = 1 ∨ n = 3 ∨ n = 4} = {1, 3, 4} :=
by
  sorry

end correct_propositions_l66_66268


namespace minimal_pairs_same_rows_l66_66507

def cell_value (rows cols : ℕ) (a : ℕ → ℕ → ℝ) : Type :=
Π (r : fin rows) (c : fin cols), ℝ

structure saddle_pair {rows cols : ℕ} (a : ℕ → ℕ → ℝ) :=
(R : finset (fin rows))
(C : finset (fin cols))
(row_condition : ∀ r' : fin rows, ∃ r ∈ R, ∀ c ∈ C, a r c ≥ a r' c)
(col_condition : ∀ c' : fin cols, ∃ c ∈ C, ∀ r ∈ R, a r c ≤ a r c')

def is_minimal_pair {rows cols : ℕ} (a : ℕ → ℕ → ℝ)
  (sp : saddle_pair a) : Prop :=
∀ (sp' : saddle_pair a),
  sp'.R ⊆ sp.R → sp'.C ⊆ sp.C → sp'.R = sp.R ∧ sp'.C = sp.C

theorem minimal_pairs_same_rows {rows cols : ℕ} (a : ℕ → ℕ → ℝ)
  (sp1 sp2 : saddle_pair a) (h1 : is_minimal_pair a sp1) (h2 : is_minimal_pair a sp2) :
  finset.card sp1.R = finset.card sp2.R :=
sorry

end minimal_pairs_same_rows_l66_66507


namespace question1_question2_question3_l66_66453

-- Given conditions
def arithmetic_square_root_recursive_seq (a : ℕ+ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ a (n + 1) = real.sqrt (a n)

def seq_x (x : ℕ+ → ℝ) : Prop :=
  x 1 = 9/2 ∧ ∀ n, x n > 0 ∧ (x (n + 1) , x n) ∈ {xy : ℝ × ℝ | xy.2 = 2 * xy.1^2 + 2 * xy.1}

def seq_y (x : ℕ+ → ℝ) (y : ℕ+ → ℝ) : Prop :=
  ∀ n, y n = real.log (2 * x n + 1)

def sum_geometric_seq (z : ℕ+ → ℝ) (m k : ℕ+) : Prop :=
  z 1 = (1 / 2)^(m - 1) ∧ (∀ n, z (n+1) = z n / 2^k) ∧ (∑' n, z n) = 16 / 63

-- Question 1
theorem question1 (x : ℕ+ → ℝ):
  seq_x x → arithmetic_square_root_recursive_seq (λ n, 2 * x n + 1) :=
sorry

-- Question 2
theorem question2 (x : ℕ+ → ℝ) (y : ℕ+ → ℝ) :
  seq_x x → seq_y x y →
  ∃ a r, (∀ n, y (n + 1) = r * y n) ∧ a = y 1 ∧ r = 1 / 2 ∧ (∀ n, y n = (1 / 2) ^ (n - 1)) :=
sorry

-- Question 3
theorem question3 (z : ℕ+ → ℝ) (m k : ℕ+) :
  sum_geometric_seq z m k → m = 3 ∧ k = 6 :=
sorry

end question1_question2_question3_l66_66453


namespace moles_of_NaCl_formed_l66_66227

theorem moles_of_NaCl_formed
    (moles_NaOH : ℕ)
    (moles_HCl : ℕ)
    (balanced_eq : ∀ n : ℕ, n * NaOH + n * HCl = n * NaCl + n * H₂O)
    (ratio : ∀ n : ℕ, (n * NaOH) = (n * HCl)) :
    moles_of_NaCl_formed (2 * NaOH + 2  * HCl = 2 * NaCl + 2  * H₂O) := 
sorry

end moles_of_NaCl_formed_l66_66227


namespace shaded_region_volume_is_85pi_l66_66010

theorem shaded_region_volume_is_85pi
  (squares : set (ℝ × ℝ))
  (h1 : ∀ p ∈ squares, p.1 + p.2 < 8)
  (h2 : ∀ p ∈ squares, p.1 < 5)
  (h3 : ∀ p ∈ squares, p.2 < 8)
  (h4 : fintype squares)
  (h5 : finset.card squares = 15) :
  ∃ V : ℝ, V = 85 * real.pi :=
by
  sorry

end shaded_region_volume_is_85pi_l66_66010


namespace desks_increase_l66_66302

theorem desks_increase 
  (rows : ℕ) (first_row_desks : ℕ) (total_desks : ℕ) 
  (d : ℕ) 
  (h_rows : rows = 8) 
  (h_first_row : first_row_desks = 10) 
  (h_total_desks : total_desks = 136)
  (h_desks_sum : 10 + (10 + d) + (10 + 2 * d) + (10 + 3 * d) + (10 + 4 * d) + (10 + 5 * d) + (10 + 6 * d) + (10 + 7 * d) = total_desks) : 
  d = 2 := 
by 
  sorry

end desks_increase_l66_66302


namespace f_equals_g_l66_66235

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) := a * x^2 + b * x + c

axiom f_property (m n : ℝ) : 
  (∃ x : ℝ, f x = m * x + n) ↔ (∃ x : ℝ, g x = m * x + n)

theorem f_equals_g : ∀ x : ℝ, f x = g x :=
sorry

end f_equals_g_l66_66235


namespace total_candy_given_l66_66869

def candy_given_total (a b c : ℕ) : ℕ := a + b + c

def first_10_friends_candy (n : ℕ) := 10 * n

def next_7_friends_candy (n : ℕ) := 7 * (2 * n)

def remaining_friends_candy := 50

theorem total_candy_given (n : ℕ) (h1 : first_10_friends_candy 12 = 120)
  (h2 : next_7_friends_candy 12 = 168) (h3 : remaining_friends_candy = 50) :
  candy_given_total 120 168 50 = 338 := by
  sorry

end total_candy_given_l66_66869


namespace minimum_pyramid_volume_proof_l66_66770

noncomputable def minimum_pyramid_volume (side_length : ℝ) (apex_angle : ℝ) : ℝ :=
  if side_length = 6 ∧ apex_angle = 2 * Real.arcsin (1 / 3 : ℝ) then 5 * Real.sqrt 23 else 0

theorem minimum_pyramid_volume_proof : 
  minimum_pyramid_volume 6 (2 * Real.arcsin (1 / 3)) = 5 * Real.sqrt 23 :=
by
  sorry

end minimum_pyramid_volume_proof_l66_66770


namespace probability_intersection_first_quadrant_l66_66925

theorem probability_intersection_first_quadrant :
  let l₁ (x y : ℝ) := x - 2 * y - 1 = 0
  let l₂ (a b x y : ℝ) := ax - by + 1 = 0
  let points_in_first_quadrant (a b : ℝ) : Prop := a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧
    (∃ x y, l₁ x y ∧ l₂ a b x y ∧ x > 0 ∧ y > 0)
  (Finset.card (Finset.filter (λ (ab : ℝ × ℝ), points_in_first_quadrant ab.1 ab.2)
    (Finset.product (Finset.range 6) (Finset.range 6)))) / 36 = 1/6 := 
sorry

end probability_intersection_first_quadrant_l66_66925


namespace rectangle_width_decreased_by_33_percent_l66_66991

theorem rectangle_width_decreased_by_33_percent
  (L W A : ℝ)
  (hA : A = L * W)
  (newL : ℝ)
  (h_newL : newL = 1.5 * L)
  (W' : ℝ)
  (h_area_unchanged : newL * W' = A) : 
  (1 - W' / W) * 100 = 33.33 :=
by
  sorry

end rectangle_width_decreased_by_33_percent_l66_66991


namespace medicine_supply_duration_l66_66667

-- Define conditions
def takes_two_thirds_pill_daily := (2 / 3)
def total_pills := 90
def days_per_month := 30

-- The lean theorem statement
theorem medicine_supply_duration :
  (total_pills * (3 / 2) / days_per_month) = 4.5 :=
by
  -- The proof is omitted
  sorry

end medicine_supply_duration_l66_66667


namespace sum_of_odd_divisors_of_90_l66_66091

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l66_66091


namespace variance_2X_plus_3_l66_66949

variable {X : Type} [ProbabilityTheory RandomVariable] 

-- Assume the variance of X is 1
axiom variance_X : D X = 1

-- Prove D(2X+3) = 4
theorem variance_2X_plus_3 : D (2 * X + 3) = 4 := by
  sorry

end variance_2X_plus_3_l66_66949


namespace problem_statement_l66_66611

theorem problem_statement (x y z : ℝ) (h : (x - z)^2 - 4 * (x - y) * (y - z) = 0) : x + z - 2 * y = 0 :=
sorry

end problem_statement_l66_66611


namespace hexagon_side_length_l66_66474

theorem hexagon_side_length (d : ℝ) (s : ℝ) (h : d = 9) :
  let altitude := (Math.sqrt 3 / 2) * s
  altitude = d → s = 6 * Math.sqrt 3 :=
by
  intro altitude_def
  intro altitude_eq_d
  rw [altitude_def] at altitude_eq_d
  sorry

end hexagon_side_length_l66_66474


namespace triangle_area_tangent_line_l66_66263

theorem triangle_area_tangent_line :
  let y := λ x : ℝ, (1 / 2) * x^2 + x
  let tangent_line := λ x, 3 * (x - 2) + 4
  let y_intercept := tangent_line 0
  let x_intercept := (0 - 4 + 3 * 2) / 3
  let base := x_intercept - 0
  let height := 4
  (1 / 2) * base * height = 8 / 3 :=
by
  let y := λ x : ℝ, (1 / 2) * x^2 + x
  let tangent_line := λ x, 3 * (x - 2) + 4
  let y_intercept := tangent_line 0
  let x_intercept := (0 - 4 + 3 * 2) / 3
  let base := x_intercept - 0
  let height := 4
  show (1 / 2) * base * height = 8 / 3
  sorry

end triangle_area_tangent_line_l66_66263


namespace product_of_digits_base8_7432_l66_66429

noncomputable def product_of_base_8_digits (n : ℕ) : ℕ :=
  let digits := (n.nat_digits 8).reverse
  digits.foldl (· * ·) 1

theorem product_of_digits_base8_7432 :
  product_of_base_8_digits 7432 = 192 := 
sorry

end product_of_digits_base8_7432_l66_66429


namespace hyperbola_eccentricity_l66_66932

noncomputable theory

open Set

-- Definitions from conditions
def hyperbola (a b : ℝ) : Set (ℝ × ℝ) := { p | (p.1^2 / a^2) - (p.2^2 / b^2) = 1 }

def A (a : ℝ) : ℝ × ℝ := (a, 0)

def F (c : ℝ) : ℝ × ℝ := (-c, 0)

def is_acute_triangle (A P Q : ℝ × ℝ) : Prop :=
  ∃ α β γ : ℝ, α + β + γ = π ∧ 0 < α ∧ α < π / 2 ∧
  ∠(P, A, Q) = α ∧ ∠(A, P, Q) = β ∧ ∠(P, Q, A) = γ

def eccentricity_range (a b : ℝ) : Set ℝ := {e | 1 < e ∧ e < 2}

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (h_pa: a > 0) (h_pb: b > 0)
  (P Q : ℝ × ℝ) (h_F : F (sqrt (a^2 + b^2))) (h_PQ : P ≠ Q)
  (h_line : P.1 = Q.1 ∧ P.1 = -sqrt (a^2 + b^2)):
  is_acute_triangle (A a) P Q → 
  ∃ e : ℝ, 1 < e ∧ e < 2 :=
begin
  sorry
end

end hyperbola_eccentricity_l66_66932


namespace negation_of_existence_is_universal_l66_66277

theorem negation_of_existence_is_universal (p : Prop) :
  (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
sorry

end negation_of_existence_is_universal_l66_66277


namespace math_problem_l66_66916

theorem math_problem
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a^3 + b^3 = 2) :
  (a + b) * (a^5 + b^5) ≥ 4 ∧ a + b ≤ 2 := 
by
  sorry

end math_problem_l66_66916


namespace rectangle_ratio_l66_66549

theorem rectangle_ratio (s x y : ℝ) 
  (h_outer_area : (2 * s) ^ 2 = 4 * s ^ 2)
  (h_inner_sides : s + 2 * y = 2 * s)
  (h_outer_sides : x + y = 2 * s) :
  x / y = 3 :=
by
  sorry

end rectangle_ratio_l66_66549


namespace arc_angle_60_deg_lens_inside_triangle_perpendiculars_intersect_at_single_point_intersection_point_locus_l66_66440

-- First problem
theorem arc_angle_60_deg (ABC : Triangle) (r : ℝ) (O : Circle) (h : ℝ) 
  (MN : Segment) (H : height_of_equilateral_triangle ABC = r) 
  (circumference_O) (M N : Point) :
  angle_subtended_by MN = 60 :=
sorry

theorem lens_inside_triangle (ABC : Triangle) (MN : Segment) (Lens) 
  (Reflection : Segment) (H : height_of_equilateral_triangle ABC = r)
  (reflected_arc : Curve) :
  lens_always_inside_triangle Lens ABC :=
sorry

-- Second problem
theorem perpendiculars_intersect_at_single_point (ABC : Triangle) 
  (MN : Segment) (H : height_of_equilateral_triangle ABC = r)
  (Perpendiculars) :
  intersect_at_single_point Perpendiculars.intersection_point :=
sorry

theorem intersection_point_locus (ABC : Triangle) (MN : Segment) 
  (H : height_of_equilateral_triangle ABC = r) 
  (Locus_fixed_triangle: Path) (Locus_rotating_dygon: Path)
  :
  locus_inside_triangle (Locus_fixed_triangle Locus_rotating_dygon) :=
sorry

end arc_angle_60_deg_lens_inside_triangle_perpendiculars_intersect_at_single_point_intersection_point_locus_l66_66440


namespace martin_rings_big_bell_l66_66695

/-
Problem Statement:
Martin rings the small bell 4 times more than 1/3 as often as the big bell.
If he rings both of them a combined total of 52 times, prove that he rings the big bell 36 times.
-/

theorem martin_rings_big_bell (s b : ℕ) 
  (h1 : s + b = 52) 
  (h2 : s = 4 + (1 / 3 : ℚ) * b) : 
  b = 36 := 
by
  sorry

end martin_rings_big_bell_l66_66695


namespace sum_of_possible_m_values_l66_66774

theorem sum_of_possible_m_values :
  let p1 := Polynomial.C 8 + Polynomial.C (-6) * Polynomial.X + Polynomial.X ^ 2,
      p2 := λ m, Polynomial.C m + Polynomial.C (-7) * Polynomial.X + Polynomial.X ^ 2 in 
  (∃ x : ℝ, p1.eval x = 0 ∧ p2 10 .eval x = 0) ∨
  (∃ x : ℝ, p1.eval x = 0 ∧ p2 12 .eval x = 0) ∧
  (10 + 12 = 22) := 
by
  sorry

end sum_of_possible_m_values_l66_66774


namespace trajectory_and_AB_line_equation_l66_66818

-- Definitions for given conditions
def is_tangent_at (C : Type*) (f : ℝ → ℝ → Prop) (x : ℝ) (y : ℝ) := sorry
def is_center_trajectory (P : Type*) (center : ℝ → ℝ → Prop) (focus : ℝ × ℝ) (line : ℝ → Prop) := sorry
def parabola_equation (x y : ℝ) :=  y^2 = 4 * x

-- Main theorem for the problem
theorem trajectory_and_AB_line_equation (P : ℝ × ℝ) (A B C D : ℝ × ℝ) :
  (∃ C : Type*, is_tangent_at C (λ x y, x = -1) (-1) 0) ∧
  (∃ P : Type*, is_center_trajectory P (λ x y, (x, y) = (1, 0)) (1, 0) (λ x, x = -1)) ∧
  (∃ AB CD : ℝ × ℝ, x1 + x2 = 4 + 4/k^2 ∧ y1 - y2 + 2 = 4 + k^2) ∧
  (area_of_quadrilateral A C B D = 36) →
  (parabola_equation (A.1) (A.2) ∧ parabola_equation (B.1) (B.2)) →
  (y = sqrt(2)*(x-1) ∨ y = sqrt(2)/2*(x-1)) :=
by
  sorry

end trajectory_and_AB_line_equation_l66_66818


namespace imaginary_part_of_complex_l66_66584

noncomputable def complex_number := (1 + complex.i)^2 + complex.i^2011

theorem imaginary_part_of_complex : complex.im complex_number = 1 := by
  sorry

end imaginary_part_of_complex_l66_66584


namespace matrix_determinant_example_l66_66289

theorem matrix_determinant_example :
  let a := 3
  let b := 4
  let c := 1
  let d := 2
  det! (Matrix.of ⟨[a, b], [c, d]⟩ (by decide : 2 = 2, by decide : 2 = 2)) = 2 :=
by
  sorry

end matrix_determinant_example_l66_66289


namespace seating_arrangements_l66_66311

theorem seating_arrangements :
  let total_arrangements := Nat.factorial 10
  let abc_together := Nat.factorial 8 * Nat.factorial 3
  let de_together := Nat.factorial 9 * Nat.factorial 2
  let abc_and_de_together := Nat.factorial 7 * Nat.factorial 3 * Nat.factorial 2
  total_arrangements - abc_together - de_together + abc_and_de_together = 2853600 :=
by
  sorry

end seating_arrangements_l66_66311


namespace total_elem_school_students_l66_66853

theorem total_elem_school_students (num_women num_women_elem num_more_males num_males_not_elem : ℕ)
  (h1 : num_women = 1518)
  (h2 : num_women_elem = 536)
  (h3 : num_more_males = 525)
  (h4 : num_males_not_elem = 1257)
  : num_women_elem + (num_women + num_more_males - num_males_not_elem) = 1322 :=
by
  rw [h1, h2, h3, h4]
  dsimp
  sorry

end total_elem_school_students_l66_66853


namespace sum_odd_divisors_90_eq_78_l66_66054

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l66_66054


namespace equal_grid_numbers_l66_66639

-- Conditions Definition
def infinite_grid_nat_n {n : ℕ} (hn : n > 2) := 
  ∀ (i j k l : ℤ), ∃ (squares : ℤ × ℤ → Fin n), 
  ∀ (P Q : Fin n → ℤ × ℤ), admissible P → admissible Q → congruent P Q → 
  value P = value Q → squares (i, j) = squares (k, l)

-- Axioms
axiom admissible {a b : ℤ} : Fin n → ℤ × ℤ 
axiom congruent {a b : ℤ} : Fin n → ℤ × ℤ → Fin n → ℤ × ℤ → Prop
axiom value : (Fin n → ℤ × ℤ) → ℤ

-- Main Statement to be proven
theorem equal_grid_numbers {n : ℕ} (hn : n > 2) : infinite_grid_nat_n hn := sorry

end equal_grid_numbers_l66_66639


namespace extreme_value_condition_l66_66012

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log x

theorem extreme_value_condition (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ Deriv (f a) x = 0) ↔ a ≤ 0 :=
sorry

end extreme_value_condition_l66_66012


namespace complex_number_sum_l66_66676

variable (ω : ℂ)
variable (h1 : ω^9 = 1)
variable (h2 : ω ≠ 1)

theorem complex_number_sum :
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80 = ω^2 :=
by sorry

end complex_number_sum_l66_66676


namespace min_area_quadrilateral_pacb_l66_66674

theorem min_area_quadrilateral_pacb : 
  ∀ P : Point,
    (3 * P.x - 4 * P.y + 11 = 0) →
    (∃ x y, P = ⟨x, y⟩ ∧ (x^2 + y^2 - 2*x - 2*y + 1 = 0)) →
    (∃ A B : Point, on_circle A ∧ on_circle B ∧ A ≠ B ∧ (3 * A.x - 4 * A.y + 11 = 0) ∧ (3 * B.x - 4 * B.y + 11 = 0)) →
    min_area_quadrilateral P A B = sqrt(3) :=
by
  sorry

end min_area_quadrilateral_pacb_l66_66674


namespace two_talents_count_l66_66215

variable (S D A : Finset Nat) (students : Finset Nat)

-- Each student can either sing, dance, or act
def students_can_either := ∀ x ∈ students, x ∈ S ∨ x ∈ D ∨ x ∈ A

-- Some students have more than one talent, but no student has all three talents
def no_all_three_talents := ∀ x ∈ students, ¬(x ∈ S ∧ x ∈ D ∧ x ∈ A)

-- Constraints on the number of students who cannot sing, dance, or act
def cannot_sing := students.card - S.card = 42
def cannot_dance := students.card - D.card = 65
def cannot_act := students.card - A.card = 29

-- Total number of students
def total_students := students.card = 100

-- Prove that the number of students with exactly two talents is 64
theorem two_talents_count :
  students_can_either S D A students →
  no_all_three_talents S D A students →
  cannot_sing S students →
  cannot_dance D students →
  cannot_act A students →
  total_students students →
  (S ∩ D \ A ⊆ students).card + (S ∩ A \ D ⊆ students).card + (D ∩ A \ S ⊆ students).card = 64 := by
  sorry

end two_talents_count_l66_66215


namespace sum_of_odd_divisors_l66_66066

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l66_66066


namespace vector_projection_l66_66968

-- Definitions for conditions
def b : ℝ × ℝ := (1/2, real.sqrt 3 / 2)
def a_dot_b : ℝ := 1/2
def b_mag : ℝ := real.sqrt ((1/2)^2 + (real.sqrt 3 / 2)^2)

-- Problem statement
theorem vector_projection (a : ℝ × ℝ) 
  (hb : b = (1/2, real.sqrt 3 / 2)) 
  (hadotb : a.1 * b.1 + a.2 * b.2 = a_dot_b) 
  (hbmagnitude : b_mag = 1) :
  (a.1 * b.1 + a.2 * b.2) / b_mag = 1/2 :=
sorry

end vector_projection_l66_66968


namespace reflect_C_final_C_l66_66422

structure Point :=
  (x : ℝ)
  (y : ℝ)

def reflect_y_axis (p : Point) : Point :=
  Point.mk (-p.x) p.y

def reflect_y_eq_x_m2 (p : Point) : Point :=
  let p_translated := Point.mk p.x (p.y + 2)
  let p_reflected := Point.mk p_translated.y p_translated.x
  Point.mk p_reflected.x (p_reflected.y - 2)

theorem reflect_C_final_C'' :
  let C := Point.mk 5 3 in
  let C' := reflect_y_axis C in
  let C'' := reflect_y_eq_x_m2 C' in
  C'' = Point.mk 5 (-7) :=
by
  sorry

end reflect_C_final_C_l66_66422


namespace estimate_weight_between_70_and_78_l66_66420

noncomputable def high_school_boys_weight_estimation
  (total_boys : ℕ)
  (survey_size : ℕ)
  (survey_histogram : ℕ → ℕ)
  (weight_interval : ℕ × ℕ)
  (estimated_boys : ℕ) :=
  total_boys = 2000 ∧
  survey_size = 100 ∧
  estimated_boys = 240 → 
  (weight_interval = (70, 78)) →
  estimated_boys = 240

-- statement in Lean 4
theorem estimate_weight_between_70_and_78 :
  high_school_boys_weight_estimation 2000 100 (λ x, 0) (70, 78) 240 :=
by sorry

end estimate_weight_between_70_and_78_l66_66420


namespace sin_double_angle_l66_66980

-- Define the necessary variables and hypothesis
variable {x : ℝ}

-- State the main theorem
theorem sin_double_angle (h : sin (π + x) + sin (3*π/2 + x) = 1 / 2) : sin (2 * x) = -3/4 :=
by
  sorry

end sin_double_angle_l66_66980


namespace ellipse_equation_and_points_l66_66316

theorem ellipse_equation_and_points :
  let E := Ellipse origin 16 12 in
  let C := x^2 + y^2 - 4 * x + 2 = 0 in
  let F := Focus C in
  let e := 1/2 in
  given (F is one of the foci of E) (e = 1/2) :
  equation E = (x^2 / 16 + y^2 / 12 = 1) ∧
  ( ∃ P : Point, (on_ellipse P E) ∧ 
    ( ∃ l1 l2 : Line, passing_through P l1 ∧ passing_through P l2 ∧ 
      slopes_product l1 l2 = 1/2 ∧ 
        tangents_to_circle l1 C ∧ 
        tangents_to_circle l2 C )) :=
    P = ⟨-2, 3⟩ ∨ 
        P = ⟨-2, -3⟩ ∨ 
        P = ⟨18/5, sqrt 57/5⟩ ∨ 
        P = ⟨18/5, -sqrt 57/5⟩ :=
sorry

end ellipse_equation_and_points_l66_66316


namespace rectangle_area_proof_l66_66759

noncomputable def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (2, 3)
def C : point := (7, 3)
def D : point := (7, 0)

def distance (p1 p2 : point) : ℝ := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def AB := distance A B
def AD := distance A D

def rectangle_area : ℝ := AB * AD

theorem rectangle_area_proof : rectangle_area = 7 * Real.sqrt 13 :=
by
  sorry

end rectangle_area_proof_l66_66759


namespace closest_integer_is_1500_l66_66536

noncomputable def closest_integer_to_sum : ℤ :=
  let s := 2000 * (∑ n in Finset.range (10001 - 2 + 1), 1 / ((n + 2)^2 - 1))
  in Int.round s

theorem closest_integer_is_1500 :
  closest_integer_to_sum = 1500 :=
sorry

end closest_integer_is_1500_l66_66536


namespace hyperbola_with_foci_on_y_axis_has_asymptotes_l66_66487

-- Problem statement and conditions
def problem_conditions :=
  ∃C₁ C₂ (A B : Set (ℝ × ℝ)),
    C₁ = { p : ℝ × ℝ | p.1^2 - p.2^2 / 9 = 1 } ∧
    A = { p : ℝ × ℝ | p.1^2 / 9 - p.2^2 = 1 } ∧
    B = { p : ℝ × ℝ | p.2^2 / 9 - p.1^2 = 1 } ∧
    C₂ = { p : ℝ × ℝ | p.2^2 - p.1^2 / 9 = 1 }

def correct_answer := 
  ∀C₁ C₂ (A B : Set (ℝ × ℝ)),
    C₂ = { p : ℝ × ℝ | p.2^2 / 9 - p.1^2 = 1 } 

-- Proof problem
theorem hyperbola_with_foci_on_y_axis_has_asymptotes :
  (problem_conditions ∧ correct_answer → C₂) := by
  sorry

end hyperbola_with_foci_on_y_axis_has_asymptotes_l66_66487


namespace sumOddDivisorsOf90_l66_66045

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l66_66045


namespace max_area_triangle_AOB_min_area_quadrilateral_l66_66246

-- Definitions based on conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 2) + (y^2) = 1
def triangle_area (A B O : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x0, y0) := O
  0.5 * | x1 * y2 + x2 * y0 + x0 * y1 - x1 * y0 - x0 * y2 - x2 * y1 |

-- Maximum area of triangle AOB
theorem max_area_triangle_AOB : 
  (∃ A B, ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ 
     triangle_area(A,B,(0,0)) = √2/2 
  ) := sorry

-- Lines forming quadrilateral with specific conditions
def line (k m : ℝ) : (ℝ × ℝ) → Prop := λ P, let (x, y) := P in y = k*x + m
def max_area_lines (L : set ((ℝ × ℝ) → Prop)) : Prop :=
  ∀ (l ∈ L), ∃ k : ℝ, l = line k (± √(k^2 + 0.5))

def quadrilateral_area (l1 l2 l3 l4 : (ℝ × ℝ) → Prop) : ℝ := sorry -- calculation of area skipped

theorem min_area_quadrilateral :
  (∃ l1 l2 l3 l4 ∈ { l : (ℝ × ℝ) → Prop | max_area_lines {l} }, 
     (∃ k1 k2 k3 k4, l1 = line k1 _ ∧ l2 = line k2 _ ∧ l3 = line k3 _ ∧ l4 = line k4 _ ∧ 
      k1 + k2 + k3 + k4 = 0 ∧ 
      quadrilateral_area l1 l2 l3 l4 = 2 * √2)
  ) := sorry

end max_area_triangle_AOB_min_area_quadrilateral_l66_66246


namespace sum_of_positive_odd_divisors_of_90_l66_66078

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l66_66078


namespace positive_area_triangles_count_l66_66605

def is_triangle (p1 p2 p3 : ℕ × ℕ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) ≠ 0

def count_all_triangles (n m : ℕ) : ℕ :=
  (nat.choose (n * m) 3)

def count_degenerate_triangles (n m : ℕ) : ℕ :=
  let rows := n * nat.choose m 3
  let cols := m * nat.choose n 3
  let diag1 := 2 * nat.choose n 3
  let diag_other := 2 * (nat.choose (n - 1) 3 + nat.choose (n - 2) 3)
  rows + cols + diag1 + diag_other

theorem positive_area_triangles_count : count_all_triangles 6 6 - count_degenerate_triangles 6 6 = 6800 :=
  sorry

end positive_area_triangles_count_l66_66605


namespace sequence_periodicity_l66_66009

noncomputable def a : ℕ → ℚ
| 0     := 3/5
| (n+1) := if 0 ≤ a n ∧ a n < 1/2 then 2 * a n else 2 * a n - 1

theorem sequence_periodicity :
  a 2014 = 1/5 := sorry

end sequence_periodicity_l66_66009


namespace math_problem_l66_66675

-- Define the conditions
def is_field (P : Set ℚ) : Prop :=
  (∀ x y ∈ P, x + y ∈ P) ∧
  (∀ x y ∈ P, x - y ∈ P) ∧
  (∀ x y ∈ P, x * y ∈ P) ∧
  (∀ x y ∈ P, y ≠ 0 → x / y ∈ P)

-- Statements to be proven
def statement_1 : Prop := ¬ is_field {n : ℚ | ∃ k : ℤ, k = n}
def statement_2 (M : Set ℚ) : Prop := (∀ M ⊆ ℚ, is_field M → ¬ ∃ x : ℚ, x ∉ ℚ)
def statement_3 : Prop := ∀ (P : Set ℚ), is_field P → infinite P
def statement_4 : Prop := ∃ (F : ℕ → Set ℚ), ∀ n : ℕ, is_field (F n)

-- The problem statement in Lean
theorem math_problem : statement_3 ∧ statement_4 :=
sorry

end math_problem_l66_66675


namespace avg_speeds_l66_66026

variable {a b : ℝ}
variable {t1 t2 t3 : ℝ}

noncomputable def avg_speed_on_entire_track := 3 / (t1 + t2 + t3)

noncomputable def avg_speed_on_first_segment := 1 / t1
noncomputable def avg_speed_on_second_segment := 1 / t2
noncomputable def avg_speed_on_third_segment := 1 / t3

theorem avg_speeds (h1 : 2 / (t1 + t2) = a)
                   (h2 : 2 / (t2 + t3) = b)
                   (h3 : 1 / t2 = 2 / (t1 + t3))
                   (h_cond : b / 3 < a ∧ a < 3 * b) :
  avg_speed_on_entire_track = 3 * a * b / (2 * (a + b)) ∧
  avg_speed_on_first_segment = 2 * a * b / (3 * b - a) ∧
  avg_speed_on_second_segment = 2 * a * b / (a + b) ∧
  avg_speed_on_third_segment = 2 * a * b / (3 * a - b) :=
by
  sorry

end avg_speeds_l66_66026


namespace triangle_area_is_168_l66_66531

def curve (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

def x_intercepts : set ℝ := {x | curve x = 0}
def y_intercept : ℝ × ℝ := (0, curve 0)

def base : ℝ := abs (4 - (-3))
def height : ℝ := curve 0

def area_of_triangle (b h : ℝ) : ℝ := 1/2 * b * h

theorem triangle_area_is_168 : area_of_triangle base height = 168 := by
  sorry

end triangle_area_is_168_l66_66531


namespace jane_oldest_babysat_age_l66_66666

-- Given conditions
def jane_babysitting_has_constraints (jane_current_age jane_stop_babysitting_age jane_start_babysitting_age : ℕ) : Prop :=
  jane_current_age - jane_stop_babysitting_age = 10 ∧
  jane_stop_babysitting_age - jane_start_babysitting_age = 2

-- Helper definition for prime number constraint
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (n % m = 0)

-- Main goal: the current age of the oldest person Jane could have babysat is 19
theorem jane_oldest_babysat_age
  (jane_current_age jane_stop_babysitting_age jane_start_babysitting_age : ℕ)
  (H_constraints : jane_babysitting_has_constraints jane_current_age jane_stop_babysitting_age jane_start_babysitting_age) :
  ∃ (child_age : ℕ), child_age = 19 ∧ is_prime child_age ∧
  (child_age = (jane_stop_babysitting_age / 2 + 10) ∨ child_age = (jane_stop_babysitting_age / 2 + 9)) :=
sorry  -- Proof to be filled in.

end jane_oldest_babysat_age_l66_66666


namespace largest_power_of_3_factor_l66_66981

-- Define q as the sum of k^2 * ln(k)
def q : ℝ := ∑ k in finset.range 9, (k:ℝ)^2 * real.log k

-- Prove that the largest power of 3 that is a factor of e^q is 3^45
theorem largest_power_of_3_factor (q : ℝ) (h : q = ∑ k in finset.range 9, (k:ℝ)^2 * real.log k) :
  ∃ n : ℤ, e^q = 3^n ∧ n = 45 :=
by
  sorry

end largest_power_of_3_factor_l66_66981


namespace sum_totient_tau_eq_sigma_l66_66213

theorem sum_totient_tau_eq_sigma (n : ℕ) :
  ∑ d in Finset.filter (λ d, d ∣ n) (Finset.range (n+1)), Nat.totient d * Nat.divisors_count (n / d) 
  = σ n :=
by
  sorry

end sum_totient_tau_eq_sigma_l66_66213


namespace coefficient_of_x5_is_7_l66_66951

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def sum_of_coefficients (n : ℕ) : ℕ :=
  (2 : ℕ)^n

noncomputable def coefficient_of_x5_in_expansion : ℕ :=
  ∑ k in Finset.range (7 + 1), if 7 - 2 * k = 5 then binomial_coefficient 7 k else 0

theorem coefficient_of_x5_is_7 :
  (sum_of_coefficients 7 = 128) →
  coefficient_of_x5_in_expansion = 7 := 
by 
  intro h
  have h₁: 7 = 2^7, from rfl -- Based on sum_of_coefficients 7 = 128
  have h₂: (2^7 = 128), from rfl
  have h₃: ∑ k in Finset.range (7 + 1), if 7 - 2 * k = 5 then binomial_coefficient 7 k else 0 = 7,
    from by sorry
  exact h₃

end coefficient_of_x5_is_7_l66_66951


namespace extreme_point_g_monotonic_f_minus_g_range_of_m_l66_66273

-- Definitions of the functions g, f, and h
def g (x : ℝ) := 2 / x + log x
def f (x m : ℝ) := m * x - (m - 2) / x
def h (x : ℝ) := 2 * Real.exp 1 / x

-- 1. Prove that the extreme point of g is x = 2
-- The extreme point is found where the first derivative is 0
theorem extreme_point_g : ∃ x, x = 2 ∧ (∃ s, s ∈ (set.Ioo 1 e) → ∀ y, (g(s) < g(y))) := sorry

-- 2. Prove that m ∈ (-∞, 0] ∪ [1, +∞) given f(x) - g(x) is monotonic on [1, +∞)
theorem monotonic_f_minus_g (m : ℝ) : (∀ x ≥ 1, (deriv (λ x => f x m - g x)) x ≥ 0) ∨ 
    (∀ x ≥ 1, (deriv (λ x => f x m - g x)) x ≤ 0) → 
    m ∈ set.Iic 0 ∪ set.Ici 1 := sorry

-- 3. Prove that m ∈ (4 e / (e^2 - 1), +∞) if there exists x_0 ∈ [1, e] such that f(x_0) - g(x_0) > h(x_0)
theorem range_of_m (m : ℝ) : (∃ x₀ ∈ (set.Icc 1 (Real.exp 1)), f x₀ m - g x₀ > h x₀) → 
    m ∈ set.Ioi (4 * Real.exp 1 / (Real.exp 1 ^ 2 - 1)) := sorry

end extreme_point_g_monotonic_f_minus_g_range_of_m_l66_66273


namespace find_y_l66_66223

noncomputable def y : ℝ :=
  let y := 27 in
  if h : log y 243 = 5 / 3 then y else 0

theorem find_y (log_condition : log (y) 243 = 5 / 3) : y = 27 :=
by {
  sorry
}

end find_y_l66_66223


namespace total_height_of_buildings_l66_66017

noncomputable def tallest_building := 100
noncomputable def second_tallest_building := tallest_building / 2
noncomputable def third_tallest_building := second_tallest_building / 2
noncomputable def fourth_tallest_building := third_tallest_building / 5

theorem total_height_of_buildings : 
  (tallest_building + second_tallest_building + third_tallest_building + fourth_tallest_building) = 180 := by
  sorry

end total_height_of_buildings_l66_66017


namespace triangle_side_lengths_inequality_l66_66354

theorem triangle_side_lengths_inequality
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end triangle_side_lengths_inequality_l66_66354


namespace find_common_ratio_l66_66572

noncomputable def a : ℕ → ℝ
noncomputable def q : ℝ

axiom geom_seq (n : ℕ) : a n = a 1 * q ^ (n - 1)
axiom a2 : a 2 = 2
axiom a6 : a 6 = (1 / 8)

theorem find_common_ratio : (q = 1 / 2) ∨ (q = -1 / 2) :=
by
  sorry

end find_common_ratio_l66_66572


namespace probability_of_log_event_is_half_l66_66324

open Real

noncomputable def probability_log_event (x : ℝ) : Prop :=
  x ∈ set.Icc 0 3 ∧ set.Icc (-1 : ℝ) 1 (log ((x + 1 / 2) / (1 / 2)) (1 / 2))

theorem probability_of_log_event_is_half :
  probability_log_event x → 
  probability (fun x => -1 ≤ log (x + 1 / 2) (1 / 2) ∧ log (x + 1 / 2) (1 / 2) ≤ 1) = 1 / 2 :=
begin
  sorry
end

end probability_of_log_event_is_half_l66_66324


namespace georgia_makes_muffins_l66_66179

-- Definitions based on conditions
def muffinRecipeMakes : ℕ := 6
def numberOfStudents : ℕ := 24
def durationInMonths : ℕ := 9

-- Theorem to prove the given problem
theorem georgia_makes_muffins :
  (numberOfStudents / muffinRecipeMakes) * durationInMonths = 36 :=
by
  -- We'll skip the proof with sorry
  sorry

end georgia_makes_muffins_l66_66179


namespace high_school_ten_total_games_l66_66386

theorem high_school_ten_total_games:
  ∃ (n : ℕ) (c_nf : ℕ) (nc_g : ℕ),
    n = 10 ∧ c_nf = 45 ∧ nc_g = 20 ∧ (c_nf * 3 + nc_g = 155) :=
by
  use 10
  use 45
  use 20
  repeat {split};
  sorry

end high_school_ten_total_games_l66_66386


namespace cube_root_equation_solutions_l66_66893

theorem cube_root_equation_solutions :
  ∀ x : ℂ,
    (x = 0 ∨ x = 4 / 9 ∨ x = -complex.I * (real.sqrt 3) / 9 ∨ x = complex.I * (real.sqrt 3) / 9) ↔
    complex.cbrt (10 * x - 2) + complex.cbrt (8 * x + 2) = 3 * complex.cbrt (x ^ 2) := 
sorry

end cube_root_equation_solutions_l66_66893


namespace prime_implies_divisor_l66_66339

theorem prime_implies_divisor (k n : ℕ) (h1 : k > 1) (h2 : k.factorial ≠ 0):
  let p := (∏ i in finset.range (k + 1), (n + i)) / k.factorial - 1 in
  nat.prime p → n ∣ k.factorial :=
by
  sorry

end prime_implies_divisor_l66_66339


namespace no_25th_digit_in_sequence_l66_66612

theorem no_25th_digit_in_sequence :
  let sequence := "10987654321"
  in String.length sequence < 25 :=
by 
  let sequence := "10987654321"
  have h : String.length sequence = 11 := by native_decide
  show String.length sequence < 25, from by linarith

end no_25th_digit_in_sequence_l66_66612


namespace probability_of_mismatch_is_half_l66_66692

open Classical

noncomputable def mismatch_probability : ℚ :=
  let pen_colors := {A, B, C}
  let cap_colors := {a, b, c}
  let outcomes := { (A,a), (B,b), (C,c), (A,a), (B,c), (C,b), (A,b), (B,a), (C,c),
                     (A,b), (B,c), (C,a), (A,c), (B,b), (C,a), (A,c), (B,a), (C,b) }
  let mismatch_outcomes := { (A,a), (B,c), (C,b), (A,b), (B,a), (C,c), (A,c), (B,b), (C,a) }
  (mismatch_outcomes.to_finset.card : ℚ) / (outcomes.to_finset.card : ℚ)

theorem probability_of_mismatch_is_half :
  mismatch_probability = 1 / 2 :=
sorry

end probability_of_mismatch_is_half_l66_66692


namespace first_class_students_l66_66532

/-- Define average mark functions for two classes and a combined average for all students. -/
theorem first_class_students (x : ℝ) 
  (h1 : 60 * x = 60) 
  (h2 : 58 * 48 = 2784) 
  (h3 : (60 * x + 2784) / (x + 48) = 59.067961165048544) : 
  x ≈ 55 :=
begin
  sorry
end

end first_class_students_l66_66532


namespace min_cups_required_l66_66030

theorem min_cups_required (n : ℕ) (h_n : n = 100) : 1 + 2 + 3 + ... + n = 5050 := by
  have calc :
    (1 + n) * n / 2 = 5050 := by sorry
  exact calc

end min_cups_required_l66_66030


namespace cross_product_vec1_vec2_l66_66894

def vec1 : ℝ × ℝ × ℝ := (3, -4, 7)
def vec2 : ℝ × ℝ × ℝ := (2, 5, -1)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

theorem cross_product_vec1_vec2 :
  cross_product vec1 vec2 = (-31, 17, 23) := sorry

end cross_product_vec1_vec2_l66_66894


namespace greatest_value_x_plus_inv_x_l66_66288

theorem greatest_value_x_plus_inv_x (x : ℝ) (hx : 10 = x^2 + 1/x^2) : 
  ∃ k: ℝ, k = Real.sqrt 12 ∧ x + 1/x ≤ k := 
begin
  sorry,
end

end greatest_value_x_plus_inv_x_l66_66288


namespace volume_multiplication_factor_l66_66986

variables (r h : ℝ) (π : ℝ := Real.pi)

def original_volume : ℝ := π * r^2 * h
def new_height : ℝ := 3 * h
def new_radius : ℝ := 2.5 * r
def new_volume : ℝ := π * (new_radius r)^2 * (new_height h)

theorem volume_multiplication_factor :
  new_volume r h / original_volume r h = 18.75 :=
by
  sorry

end volume_multiplication_factor_l66_66986


namespace max_printed_cards_l66_66880

def is_valid_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 6 ∨ d = 8 ∨ d = 9

def is_valid_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  is_valid_digit hundreds ∧ is_valid_digit tens ∧ is_valid_digit units

-- The main theorem to prove
theorem max_printed_cards : ∃ max_cards : ℕ, max_cards = 34 := 
begin
  sorry,
end

end max_printed_cards_l66_66880


namespace sum_of_odd_divisors_of_90_l66_66082

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l66_66082


namespace find_a_from_parallel_lines_l66_66585

theorem find_a_from_parallel_lines : 
  (∀ x, ∃ y, y = 3 * x^2 + 2 * x) → 
  (∀ y0 x0, y0 = 3 * x0^2 + 2 * x0 → x0 = 1 → y0 = 5) → 
  (∀ m, m = 6 * 1 + 2 → (6 * 1 + 2) = 8) →
  (∀ a, 2 * a = 8 → a = 4) :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end find_a_from_parallel_lines_l66_66585


namespace functional_relationship_purchasing_plans_adjusted_prices_l66_66133

def bookstore_fund := 7700
def total_sets := 20
def cost_A := 500
def cost_B := 400
def cost_C := 250
def sell_A := 550
def sell_B := 430
def sell_C := 310

def x : ℕ -- Type A books purchased
def y : ℕ -- Type B books purchased

-- Functional relationship between y and x
theorem functional_relationship (x : ℕ) : y = - 5 * x / 3 + 18 :=
by
  sorry

-- Number of possible purchasing plans
theorem purchasing_plans (x : ℕ) (h1: 1 ≤ x) (h2: x ≤ 10) :
  ∃ y z : ℕ, z = 20 - x - y ∧ 1 ≤ y ∧ y = - 5 * x / 3 + 18 ∧
  (x, y, z) = (3, 13, 4) ∨ (x, y, z) = (6, 8, 6) ∨ (x, y, z) = (9, 3, 8) :=
by
  sorry

-- Adjusted selling prices and profit
theorem adjusted_prices (x y z : ℕ) (a : ℕ) :
  (x, y, z) = (6, 8, 6) ∧ a = 10 ∧
  sell_A = 550 ∧ 
  (sell_B + a) * y - (sell_C - a) * z = 20 :=
by
  sorry

end functional_relationship_purchasing_plans_adjusted_prices_l66_66133


namespace bags_needed_l66_66498

theorem bags_needed (expected_people extra_people extravagant_bags average_bags : ℕ) 
    (h1 : expected_people = 50) 
    (h2 : extra_people = 40) 
    (h3 : extravagant_bags = 10) 
    (h4 : average_bags = 20) : 
    (expected_people + extra_people - (extravagant_bags + average_bags) = 60) :=
by {
  sorry
}

end bags_needed_l66_66498


namespace sum_of_squares_difference_l66_66895

theorem sum_of_squares_difference :
  let S_A := 1500 * (1500 + 1) * (2 * 1500 + 1) / 6
  let S_B := 4 * 1499 * 1500 * 2999 / 6
  S_A - S_B = -372750250 :=
by
  let S_A := 1500 * (1500 + 1) * (2 * 1500 + 1) / 6
  let S_B := 4 * (1499 * 1500 * 2999 / 6)
  have S_A_val: S_A = 1125751250 := by sorry
  have S_B_val: S_B = 1498501500 := by sorry
  calc S_A - S_B = 1125751250 - 1498501500 : by rw [S_A_val, S_B_val]
                ... = -372750250 : by norm_num

end sum_of_squares_difference_l66_66895


namespace number_of_scheduling_arrangements_l66_66417

variables 
  (staff : Type) 
  [fintype staff] 
  (days : fin 5)
  (duty : days → option staff)

/-- Three staff members scheduled to be on duty from the first to the fifth day of the lunar month
    with exactly one person on duty each day and each person can be on duty for at most two days --/
def scheduling_conditions (d : days) : Prop :=
  (∀ d1 d2 : days, d1 ≠ d2 → duty d1 ≠ none → duty d2 ≠ none → duty d1 ≠ duty d2) ∧
  (∀ s : staff, (finset.filter (λ d, duty d = some s) finset.univ).card ≤ 2)

theorem number_of_scheduling_arrangements {staff set_of_staff} (h : set_of_staff.card = 3)
  (hd : scheduling_conditions days duty): 
  ∃ total_arrangements, total_arrangements = 90 :=
sorry

end number_of_scheduling_arrangements_l66_66417


namespace tenth_permutation_is_2561_l66_66006

-- Define the digits
def digits : List ℕ := [1, 2, 5, 6]

-- Define the permutations of those digits
def perms := digits.permutations.map (λ l, l.asString.toNat)

-- Define the 10th integer in the sorted list of those permutations
noncomputable def tenth_integer : ℕ := (perms.sort (≤)).nthLe 9 (by norm_num [List.length_permutations, perms])

-- The statement we want to prove
theorem tenth_permutation_is_2561 : tenth_integer = 2561 :=
sorry

end tenth_permutation_is_2561_l66_66006


namespace marble_color_203rd_l66_66845

theorem marble_color_203rd :
  let pattern := [("red", 6), ("blue", 5), ("green", 4)]
  let marbles := List.cycle_replicate 16 pattern
  marbles.nth 202 = some "blue" :=
by
  sorry

end marble_color_203rd_l66_66845


namespace proof_by_contradiction_x_gt_y_implies_x3_gt_y3_l66_66776

theorem proof_by_contradiction_x_gt_y_implies_x3_gt_y3
  (x y: ℝ) (h: x > y) : ¬ (x^3 ≤ y^3) :=
by
  -- We need to show that assuming x^3 <= y^3 leads to a contradiction
  sorry

end proof_by_contradiction_x_gt_y_implies_x3_gt_y3_l66_66776


namespace cover_black_squares_with_triominio_l66_66352

theorem cover_black_squares_with_triominio (n : ℕ) (hodd : n % 2 = 1) (hlarge : n > 1) :
  (∀ (b : ℕ), n = 2 * b + 1 → b ≥ 3 → (∃ k : ℕ, k * k = (b + 1) * (b + 1) ∧
    ∀ m : ℕ, m = 0 ∨ m = 3 ∨ m = 6 ∨ 
    ((m % 3 = 0) ∧ (0 < m) → 
    (∃ l : ℕ, l = m / 3 ∧ l ≤ k ∧ 
    (∀ p : ℕ, p = ((b + 1)^2 l) / (3 * m)))))) := sorry

end cover_black_squares_with_triominio_l66_66352


namespace percent_of_value_l66_66758

theorem percent_of_value : (2 / 5) * (1 / 100) * 450 = 1.8 :=
by sorry

end percent_of_value_l66_66758


namespace perfect_square_trinomial_m_l66_66938

theorem perfect_square_trinomial_m (m : ℤ) :
  ∀ y : ℤ, ∃ a : ℤ, (y^2 - m * y + 1 = (y + a) ^ 2) ∨ (y^2 - m * y + 1 = (y - a) ^ 2) → (m = 2 ∨ m = -2) :=
by 
  sorry

end perfect_square_trinomial_m_l66_66938


namespace kevin_total_dribbles_l66_66649

theorem kevin_total_dribbles :
  let first_three_seconds := 13
  let next_five_seconds := 18
  let num_five_sec_intervals := 3
  let arithmetic_diff := -3
  let remaining_seconds := 27 - 3 - 5
  let last_interval_dribbles := λ n, next_five_seconds + n * arithmetic_diff
  let total_dribbles :=
    first_three_seconds +
    next_five_seconds +
    (List.range num_five_sec_intervals).sum_map last_interval_dribbles 5 +
    last_interval_dribbles num_five_sec_intervals * (remaining_seconds % 5) / 5
  total_dribbles = 83 :=
begin
  sorry
end

end kevin_total_dribbles_l66_66649


namespace jellybeans_remained_l66_66744

theorem jellybeans_remained (total_jellybeans : ℕ) (num_people : ℕ)
  (first_six_take_twice : ∀ n, n = 6 → (first_six_took : ℕ) = 2 * (last_four_took : ℕ))
  (last_four_took : ℕ) :
  total_jellybeans = 8000 →
  num_people = 10 →
  last_four_took = 400 →
  (remaining_jellybeans : ℕ) = total_jellybeans - (6 * (2 * last_four_took) + 4 * last_four_took) :=
by
  intros total_jellybeans_eq num_people_eq last_four_took_eq,
  sorry

end jellybeans_remained_l66_66744


namespace exp_grows_faster_than_power_l66_66654

theorem exp_grows_faster_than_power {a x : ℝ} (h₁ : 1 < a) (h₂ : 0 < x) : 
  is_faster_growth (a ^ x) (x ^ a) :=
sorry

end exp_grows_faster_than_power_l66_66654


namespace find_number_l66_66983

theorem find_number (N : ℝ) (h : 0.4 * (3 / 5) * N = 36) : N = 150 := 
sorry

end find_number_l66_66983


namespace quadrilateral_area_l66_66315

def point := (ℝ × ℝ)

def square (A B C D : point) : Prop :=
  A = (0, 0) ∧ B = (3, 0) ∧ C = (3, 3) ∧ D = (0, 3)

def divides_in_ratio (P Q : point) (ratio : ℕ × ℕ) : Prop := 
  Q.1 = P.1 / (1 + ratio.2 / ratio.1) -- Simplified check, not general

def midpoint (P Q M : point) : Prop :=
  M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2

noncomputable def area (P Q R S : point) : ℝ :=
  (1 / 2) * abs (P.1 * Q.2 + Q.1 * R.2 + R.1 * S.2 + S.1 * P.2 - (P.2 * Q.1 + Q.2 * R.1 + R.2 * S.1 + S.2 * P.1))

theorem quadrilateral_area
  (A B C D E F G H R T S Q : point)
  (h_square : square A B C D)
  (h_EF : divides_in_ratio E A (1, 2) ∧ divides_in_ratio F A (2, 1))
  (h_GH : divides_in_ratio G D (1, 2) ∧ divides_in_ratio H D (2, 1))
  (h_RT : R = (0, 1.5) ∧ T = (3, 1.5)) -- Given R and T divide the square into two equal areas
  (h_midpoints : midpoint A B S ∧ midpoint C D Q) :
  area R Q S T = 1.125 :=
sorry

end quadrilateral_area_l66_66315


namespace find_x_l66_66777

def Hiram_age := 40
def Allyson_age := 28
def Twice_Allyson_age := 2 * Allyson_age
def Four_less_than_twice_Allyson_age := Twice_Allyson_age - 4

theorem find_x (x : ℤ) : Hiram_age + x = Four_less_than_twice_Allyson_age → x = 12 := 
by
  intros h -- introducing the assumption 
  sorry

end find_x_l66_66777


namespace min_value_shifted_function_l66_66298

theorem min_value_shifted_function : 
  ∀ (c : ℝ), (∀ x : ℝ, f x = x^2 + 4 * x + 5 - c) → (∀ x : ℝ, 2 ≤ f x) → (∀ x : ℝ, 2 ≤ f (x - 2015)) := 
by
  -- Proof Placeholder
  sorry

end min_value_shifted_function_l66_66298


namespace calc_expr_l66_66191

theorem calc_expr :
  (Real.sqrt (1 / 9) - Real.sqrt (25 / 4) + 3 * Real.sqrt (4) - (-(2:Real) ^ 3).root 3) = 35 / 6 :=
by
  sorry

end calc_expr_l66_66191


namespace monotonic_decreasing_fx_l66_66205

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem monotonic_decreasing_fx : ∀ (x : ℝ), (0 < x) ∧ (x < (1 / exp 1)) → deriv f x < 0 := 
by
  sorry

end monotonic_decreasing_fx_l66_66205


namespace unit_vectors_parallel_implies_equal_or_neg_l66_66253

variables {V : Type*} [InnerProductSpace ℝ V]

def is_unit_vector (v : V) : Prop := ∥v∥ = 1

def are_parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b

theorem unit_vectors_parallel_implies_equal_or_neg
  (a b : V) (ha : is_unit_vector a) (hb : is_unit_vector b) (hab : are_parallel a b) :
  a = b ∨ a = -b :=
by
  sorry

end unit_vectors_parallel_implies_equal_or_neg_l66_66253


namespace states_solar_ratio_l66_66664

theorem states_solar_ratio :
  let initial := 20
  let decided := 15
  let reverted := 2
  let new_states_adopting_solar := decided - reverted
  let ratio := (new_states_adopting_solar : ℝ) / initial
  ratio.round = 0.7 :=
  by
  sorry

end states_solar_ratio_l66_66664


namespace find_x_plus_inv_x_l66_66940

theorem find_x_plus_inv_x (x : ℝ) (hx : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 := 
by 
  sorry

end find_x_plus_inv_x_l66_66940


namespace sum_odd_divisors_of_90_l66_66105

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l66_66105


namespace problem1_problem2_l66_66804

-- First problem
theorem problem1 :
  (0.027 ^ (-1/3) - ((-1 / 7) ^ -2) + (256 ^ (3 / 4)) - (3 ^ -1) + ((Real.sqrt 2 - 1) ^ 0)) = 19 := 
by
  -- Proof is omitted
  sorry

-- Second problem
theorem problem2 :
  ((Real.log 8 + Real.log 125 - Real.log 2 - Real.log 5) / ((Real.log (Real.sqrt 10)) * (Real.log 0.1))) = -4 :=
by
  -- Proof is omitted
  sorry

end problem1_problem2_l66_66804


namespace elective_schemes_count_l66_66630

open Finset

variable (A B C D E F G H I : Type)
variable [DecidableEq A] [DecidableEq B] [DecidableEq C] 
variable [DecidableEq D] [DecidableEq E] [DecidableEq F]
variable [DecidableEq G] [DecidableEq H] [DecidableEq I]

theorem elective_schemes_count :
  let courses := ({A, B, C, D, E, F, G, H, I} : Finset Type) in
  let abc := ({A, B, C} : Finset Type) in
  let others := (courses \ abc) in
  @card (Set (courses \ ∅)) = 4 →
  @card ((abc \ ∅) ∪ (choose others 3)) + @card (choose others 4) = 75 :=
by
  sorry

end elective_schemes_count_l66_66630


namespace axis_of_symmetry_l66_66725

open Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x + π / 3)

-- Define the interval
def interval : set ℝ := {x | 0 < x ∧ x < π / 2}

-- The theorem statement: axis of symmetry for the graph 
theorem axis_of_symmetry : (x ∈ interval) → function.symm_axis f interval x = π / 12 :=
by sorry

end axis_of_symmetry_l66_66725


namespace decreasing_interval_l66_66577

theorem decreasing_interval (x0 phi : ℝ) (f : ℝ → ℝ) :
  x0 = -π / 6 →
  (∀ x, f x = sin (2 * x + phi)) →
  (∀ x, has_deriv_at f (cos (2 * x + phi) * 2) x) →
  ∃ (k : ℤ), (∀ x, k = 0 → x0 = -π / 6 → phi = 2 * ↑k * π - π / 6 → 
    f x = sin (2 * x - π / 6) → 
    (π / 3 < x ∧ x < 5 * π / 6) → 
    ∃ (a b : ℝ), π / 3 = a ∧ 5 * π / 6 = b ∧ a < b ∧ 
     (∀ x, a < x ∧ x < b → f x = sin (2 * x - π / 6) ∧ deriv f x < 0)) := by 
  sorry

end decreasing_interval_l66_66577


namespace sqrt_product_example_l66_66495

theorem sqrt_product_example :
  (√ (49 * (√25) * (√64))) = 28 :=
by
  have h1 : √25 = 5 := by
    norm_num
  have h2 : √64 = 8 := by
    norm_num
  sorry

end sqrt_product_example_l66_66495


namespace remove_terms_to_get_two_thirds_l66_66876

noncomputable def sum_of_terms : ℚ := 
  (1/3) + (1/6) + (1/9) + (1/12) + (1/15) + (1/18)

noncomputable def sum_of_remaining_terms := 
  (1/3) + (1/6) + (1/9) + (1/18)

theorem remove_terms_to_get_two_thirds :
  sum_of_terms - (1/12 + 1/15) = (2/3) :=
by
  sorry

end remove_terms_to_get_two_thirds_l66_66876


namespace coefficient_of_x4_is_correct_l66_66764

noncomputable def coefficient_of_x4_in_expansion : Nat :=
  let x := 1 -- x is just symbolic
  let b := -3 * Real.sqrt 2
  let n := 8
  let k := 4
  -- binomial theorem coefficient
  let binom := Nat.choose n k
  -- compute b^k (note that because k = 4, (-3√2)^4 = (3√2)^4)
  let b_pow_k := (3 * Real.sqrt 2) ^ k
  
  binom * b_pow_k

theorem coefficient_of_x4_is_correct : coefficient_of_x4_in_expansion = 22680 :=
by
  unfold coefficient_of_x4_in_expansion
  rw [Nat.choose_eq_factorial_div_factorial, Nat.factorial, Nat.factorial, Nat.factorial, Nat.factorial, Real.pow_nat_eq, Real.pow_nat_eq, Real.mul_pow]
  -- calculations for binom coefficient and b^k can be detailed here
  -- sorry can be used here to skip the detailed proof outline if desired
  sorry

end coefficient_of_x4_is_correct_l66_66764


namespace coefficient_of_x4_in_expansion_l66_66760

theorem coefficient_of_x4_in_expansion :
  let f := λ (x : ℝ), (x - 3 * Real.sqrt 2) ^ 8
  ∃ c : ℝ, c * x^4 = (f x) → c = 22680 :=
begin
  intro f,
  use (λ (x : ℝ), (x - 3 * Real.sqrt 2) ^ 8),
  intro x,
  sorry
end

end coefficient_of_x4_in_expansion_l66_66760


namespace trapezoid_diagonal_length_l66_66657

theorem trapezoid_diagonal_length (A B C D O : Point) (AB BC CD AD AC BD : ℝ)
  (h1 : AD = 16) (h2 : BC = 10) (h3 : circleDiameterIntersection A B C D = O)
  (h4 : AC = 10) : BD = 24 := by
  sorry

end trapezoid_diagonal_length_l66_66657


namespace difference_of_two_distinct_members_sum_of_two_distinct_members_l66_66600

theorem difference_of_two_distinct_members (S : Set ℕ) (h : S = {n | n ∈ Finset.range 20 ∧ 1 ≤ n ∧ n ≤ 20}) :
  (∃ N, N = 19 ∧ (∀ n, 1 ≤ n ∧ n ≤ N → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ n = a - b)) :=
by
  sorry

theorem sum_of_two_distinct_members (S : Set ℕ) (h : S = {n | n ∈ Finset.range 20 ∧ 1 ≤ n ∧ n ≤ 20}) :
  (∃ M, M = 37 ∧ (∀ m, 3 ≤ m ∧ m ≤ 39 → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ m = a + b)) :=
by
  sorry

end difference_of_two_distinct_members_sum_of_two_distinct_members_l66_66600


namespace speed_of_stream_l66_66465

variable (b s : ℝ)

theorem speed_of_stream (h1 : 110 = (b + s + 3) * 5)
                        (h2 : 85 = (b - s + 2) * 6) : s = 3.4 :=
by
  sorry

end speed_of_stream_l66_66465


namespace altitudes_intersect_at_single_point_l66_66373

theorem altitudes_intersect_at_single_point {A B C A1 B1 C1 : Type*}
  [AB1 : A = B1]
  [AC1 : A = C1]
  [BC1 : B = C1]
  [BA1 : B = A1]
  [CA1 : C = A1]
  [CB1 : C = B1] : 
  ∃ H : Type*, 
    (perpendicular_from A1 BC H) ∧ 
    (perpendicular_from B1 CA H) ∧ 
    (perpendicular_from C1 AB H) := 
by sorry

end altitudes_intersect_at_single_point_l66_66373


namespace isosceles_triangle_of_sinc_eq_2cosAsinB_shape_of_triangle_ABC_l66_66663

theorem isosceles_triangle_of_sinc_eq_2cosAsinB {A B C : ℝ} (h : (A + B + C = π)) 
  (h₁ : real.sin C = 2 * real.cos A * real.sin B) : A = B :=
by
    sorry

theorem shape_of_triangle_ABC {A B C : ℝ} (h : (A + B + C = π))
  (h₁ : real.sin C = 2 * real.cos A * real.sin B) : 
is_isosceles A B C :=
by
  have h_equal : A = B := isosceles_triangle_of_sinc_eq_2cosAsinB h h₁
  sorry

end isosceles_triangle_of_sinc_eq_2cosAsinB_shape_of_triangle_ABC_l66_66663


namespace relation_among_abc_l66_66557

noncomputable def a : ℝ := (1 / 2) ^ 10
noncomputable def b : ℝ := (1 / 5) ^ (-1 / 2)
noncomputable def c : ℝ := Real.log10 (10) / Real.log10 (1 / 5)

theorem relation_among_abc : b > a ∧ a > c := 
by
  sorry

end relation_among_abc_l66_66557


namespace most_cost_effective_restaurant_B_l66_66856

/-
Annie has $120 to spend on food and wants to buy 8 hamburgers and 6 milkshakes. 
The restaurants have different prices, tax rates, and special offers.
Prove that the most cost-effective option for her is Restaurant B and 
she will have $58.94 left after making the purchase.
-/

def price_restaurant_A_hamburger := 4
def price_restaurant_A_milkshake := 5
def tax_rate_A := 0.08
def special_offer_A_hamburgers := 5 -- buy 5, get 1 free

def price_restaurant_B_hamburger := 3.50
def price_restaurant_B_milkshake := 6
def tax_rate_B := 0.06
def special_offer_B_milkshakes := 3 -- 10% discount if buy at least 3

def price_restaurant_C_hamburger := 5
def price_restaurant_C_milkshake := 4
def tax_rate_C := 0.10
def special_offer_C_milkshakes := 2 -- buy 2, get the 3rd at half price

def annie_budget := 120
def hamburgers_needed := 8
def milkshakes_needed := 6

def total_cost_restaurant_B :=
  let hamburger_cost := hamburgers_needed * price_restaurant_B_hamburger
  let milkshake_cost := milkshakes_needed * price_restaurant_B_milkshake
  let total_cost_pre_discount := hamburger_cost + milkshake_cost
  let discount := if milkshakes_needed >= special_offer_B_milkshakes then total_cost_pre_discount * 0.10 else 0
  let total_cost_post_discount := total_cost_pre_discount - discount
  let tax := total_cost_post_discount * tax_rate_B
  total_cost_post_discount + tax

def money_left := annie_budget - total_cost_restaurant_B

theorem most_cost_effective_restaurant_B : 
  total_cost_restaurant_B = 61.06 ∧ 
  money_left = 58.94 := 
by 
  -- The proof steps will be skipped
  sorry

end most_cost_effective_restaurant_B_l66_66856


namespace trajectory_equation_line_through_fixed_point_l66_66644

-- Definitions for Problem 1
def A1 : ℝ × ℝ := (-2, 0)
def A2 : ℝ × ℝ := (2, 0)

def mn_condition (m n : ℝ) : Prop := m * n = 3

-- Convert the problem and conditions into a Lean 4 theorem
theorem trajectory_equation (m n : ℝ) (H : mn_condition m n) :
  ∃ x y : ℝ, (x ≠ 2 ∧ x ≠ -2) ∧ (x^2 / 4 + y^2 / 3 = 1) := 
sorry

-- Definitions for Problem 2
def F2 : ℝ × ℝ := (1, 0)
def slope_sum_condition (a b : ℝ) : Prop := a + b = π

-- Convert the problem and conditions into a Lean 4 theorem
theorem line_through_fixed_point (k m : ℝ) (trajectory_exists : ∃ x y : ℝ, (x ≠ 2 ∧ x ≠ -2) ∧ (x^2 / 4 + y^2 / 3 = 1)) 
  (H_slope : ∀ P Q : ℝ × ℝ, P ≠ Q → (slope_sum_condition (k * (P.fst - A1.fst) + m) (k * (Q.fst - A2.fst) + m))) :
  ∃ fixed_point : ℝ × ℝ, fixed_point = (4, 0) :=
sorry

end trajectory_equation_line_through_fixed_point_l66_66644


namespace parallel_vectors_sin_cos_eq_three_l66_66603

variable (θ : ℝ)

def a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def b : ℝ × ℝ := (1, -2)

theorem parallel_vectors_sin_cos_eq_three (h : a θ = λ s, b s * 3) : 
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 3 := 
by 
  sorry

end parallel_vectors_sin_cos_eq_three_l66_66603


namespace probability_MN_l66_66310

-- Define the balls and the sample space
def balls : Finset ℕ := {1, 2, 3, 4, 5} -- Here 1, 2, 3 represent red balls A, B, C and 4, 5 represent black balls D, E

-- Define events M and N
def event_M : Finset (ℕ × ℕ) := {(1,1), (1,2), (1,3), (1,4), (1,5), (2,1), (2,2), (2,3), (2,4), (2,5), (3,1), (3,2)}
def event_N : Finset (ℕ × ℕ) := {(1,4), (1,5), (2,4), (2,5), (3,4), (3,5), (4,4), (5,4), (4,5), (5,5)}

-- Total sample space
def sample_space : Finset (ℕ × ℕ) :=
  {(1,2), (1,3), (1,4), (1,5),
   (2,1), (2,3), (2,4), (2,5),
   (3,1), (3,2), (3,4), (3,5),
   (4,1), (4,2), (4,3), (4,5),
   (5,1), (5,2), (5,3), (5,4)}

-- Probability calculations
def prob_M : ℚ := event_M.card / sample_space.card
def prob_N : ℚ := event_N.card / sample_space.card

-- Theorem statement
theorem probability_MN :
  prob_M = 3 / 5 ∧ prob_N = 2 / 5 :=
by
  unfold prob_M prob_N sample_space event_M event_N
  simp [Finset.card_product, Finset.card, Finset.filter_card]
  sorry

end probability_MN_l66_66310


namespace tennis_tournament_possible_l66_66307

theorem tennis_tournament_possible (n : ℕ) : 
  (∃ k : ℕ, n = 8 * k + 1) ↔ (
    ∃ participants : set (ℕ × ℕ), 
    (∀ p ∈ participants, ∃ a b : ℕ, p = (a, b) ∧ a ≠ b) ∧ 
    (∀ p1 p2 ∈ participants, p1 ≠ p2 → (p1.1 = p2.1 ∨ p1.1 = p2.2 ∨ p1.2 = p2.1 ∨ p1.2 = p2.2) → False)
  ) :=
sorry

end tennis_tournament_possible_l66_66307


namespace sum_of_zeros_eq_sixty_l66_66013

open Real

theorem sum_of_zeros_eq_sixty :
  let f := λ x : ℝ, 2 * (5 - x) * sin (π * x) - 1
  in (∑ x in (Finset.filter (λ x, f x = 0) (Finset.range 11)), x) = 60 :=
by
  sorry

end sum_of_zeros_eq_sixty_l66_66013


namespace functional_equation_solution_l66_66121

theorem functional_equation_solution (f g : ℝ → ℝ)
  (H : ∀ x y : ℝ, f (x^2 - g y) = g x ^ 2 - y) :
  (∀ x : ℝ, f x = x) ∧ (∀ x : ℝ, g x = x) :=
by
  sorry

end functional_equation_solution_l66_66121


namespace expand_product_l66_66526

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by 
  sorry

end expand_product_l66_66526


namespace ana_wins_l66_66400

-- Define the game conditions and state
def game_conditions (n : ℕ) (m : ℕ) : Prop :=
  n < m ∧ m < n^2 ∧ Nat.gcd n m = 1

-- Define the losing condition
def losing_condition (n : ℕ) : Prop :=
  n >= 2016

-- Define the predicate for Ana having a winning strategy
def ana_winning_strategy : Prop :=
  ∃ (strategy : ℕ → ℕ), strategy 3 = 5 ∧
  (∀ n, (¬ losing_condition n) → (losing_condition (strategy n)))

theorem ana_wins : ana_winning_strategy :=
  sorry

end ana_wins_l66_66400


namespace find_values_of_x_l66_66936

noncomputable def solution_x (x y : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ 
  x^2 + 1/y = 13 ∧ 
  y^2 + 1/x = 8 ∧ 
  (x = Real.sqrt 13 ∨ x = -Real.sqrt 13)

theorem find_values_of_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x^2 + 1/y = 13) (h2 : y^2 + 1/x = 8) : x = Real.sqrt 13 ∨ x = -Real.sqrt 13 :=
by { sorry }

end find_values_of_x_l66_66936


namespace geometric_sequence_common_ratio_l66_66575

theorem geometric_sequence_common_ratio (a : ℕ → ℚ) (q : ℚ) :
  (∀ n, a n = a 2 * q ^ (n - 2)) ∧ a 2 = 2 ∧ a 6 = 1 / 8 →
  (q = 1 / 2 ∨ q = -1 / 2) :=
by
  sorry

end geometric_sequence_common_ratio_l66_66575


namespace characterize_affine_function_l66_66201

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, a < b → integrable_on f (set.Icc a b)) ∧
  (∀ x : ℝ, ∀ n : ℕ, 1 ≤ n → (f x = (n / 2) * integral (Icc (x - 1 / n) (x + 1 / n)) f))

theorem characterize_affine_function :
  ∀ f : ℝ → ℝ, satisfies_conditions f → ∃ p q : ℝ, ∀ x : ℝ, f x = p * x + q := 
sorry

end characterize_affine_function_l66_66201


namespace parametric_equation_of_line_through_M_l66_66463

noncomputable theory

def point := ℝ × ℝ

def parametric_equation (P Q: point) (a b : ℝ) (t : ℝ) : Prop :=
  ∃ u : ℝ, Q = (P.1 + a * t, P.2 + b * t)

theorem parametric_equation_of_line_through_M 
  (M : point) (t : ℝ)
  (hM : M = (1, 5))
  (hSlopeAngle : ∃ θ, θ = π / 3 ∧ tan θ = b / a) :
  parametric_equation M (1 + 1/2 * t, 5 + (sqrt 3)/2 * t) := 
by
  sorry

end parametric_equation_of_line_through_M_l66_66463


namespace jenny_boxes_sold_l66_66331

/--
Jenny sold some boxes of Trefoils. Each box has 8.0 packs. She sold 192 packs in total.
Prove that Jenny sold 24 boxes.
-/
theorem jenny_boxes_sold (packs_per_box : Real) (total_packs_sold : Real) (num_boxes_sold : Real) 
  (h1 : packs_per_box = 8.0) (h2 : total_packs_sold = 192) : num_boxes_sold = 24 :=
by
  have h3 : num_boxes_sold = total_packs_sold / packs_per_box :=
    by sorry
  sorry

end jenny_boxes_sold_l66_66331


namespace travel_time_l66_66995

noncomputable def convert_kmh_to_mps (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

theorem travel_time
  (speed_kmh : ℝ)
  (distance_m : ℝ) :
  speed_kmh = 63 →
  distance_m = 437.535 →
  (distance_m / convert_kmh_to_mps speed_kmh) = 25 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end travel_time_l66_66995


namespace correct_statements_about_series_l66_66597

-- Definition of the infinite geometric series with first term 3 and common ratio 1/3.
noncomputable def geometric_series := λ (n : ℕ), 3 * (1 / 3) ^ n

-- Sum of the infinite geometric series.
noncomputable def S := tsum (λ (n : ℕ), 3 * (1 / 3) ^ n)

-- The Lean 4 statement of the proof problem.
theorem correct_statements_about_series :
  (3 + 1 + 1/3 + 1/9 + ...) →       -- The geometric series under consideration
  (∀ ε > 0, ∃ N, ∀ n ≥ N, abs (geometric_series n - 0) < ε) ∧  -- Statement 3
  (∀ ε > 0, abs (S - 9/2) < ε) ∧   -- Statement 4
  (∃ L, tendsto (λ n, partial_sum geometric_series n) at_top (𝓝 L)) -- Statement 5
  := sorry

end correct_statements_about_series_l66_66597


namespace find_x_satisfying_log_equation_l66_66541

theorem find_x_satisfying_log_equation :
  ∃ x : ℝ, x > 0 ∧
    log 3 (x^2 - 3) + (log 3 (x - 2)) / (log 3 9) + (log 3 (x^2 - 3)) / (log 3 (1/3)) = 2 ∧
    x = 83 :=
by
  sorry

end find_x_satisfying_log_equation_l66_66541


namespace find_d_e_f_l66_66686

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 37) / 3 + 5 / 3)

theorem find_d_e_f :
  ∃ (d e f : ℕ), (y ^ 50 = 3 * y ^ 48 + 10 * y ^ 45 + 9 * y ^ 43 - y ^ 25 + d * y ^ 21 + e * y ^ 19 + f * y ^ 15) 
    ∧ (d + e + f = 119) :=
sorry

end find_d_e_f_l66_66686


namespace rhombus_diagonal_l66_66834

theorem rhombus_diagonal (side : ℝ) (short_diag : ℝ) (long_diag : ℝ) 
  (h1 : side = 37) (h2 : short_diag = 40) :
  long_diag = 62 :=
sorry

end rhombus_diagonal_l66_66834


namespace minimum_value_of_sum_l66_66208

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ (1 / real.cbrt 2) :=
by
  sorry

end minimum_value_of_sum_l66_66208


namespace nickel_to_dime_ratio_l66_66859

theorem nickel_to_dime_ratio (
  (N D Q: ℕ)
  (H1: 3 * N = 120)
  (H2: Q = 2 * D)
  (H3: 120 + 5 * N + 10 * D + 25 * Q = 800)
) : N / D = 5 := 
sorry

end nickel_to_dime_ratio_l66_66859


namespace units_digit_sum_d1n_d2n_is_9_l66_66543

-- Definitions based on conditions
def a : ℤ := 15
def b : ℤ := 220
def d1 := a + Real.sqrt b
def d2 := a - Real.sqrt b

-- The theorem we aim to prove
theorem units_digit_sum_d1n_d2n_is_9 :
  (d1^19 + d2^19) % 10 = 9 :=
by
  sorry

end units_digit_sum_d1n_d2n_is_9_l66_66543


namespace james_total_distance_l66_66330

-- Definitions of conditions
def distance_first_hour (d2 : ℝ) : ℝ := d2 / 1.2
def distance_third_hour (d2 : ℝ) : ℝ := d2 * 1.25

-- Main statement to be proved
theorem james_total_distance :
  let d2 := 12 in
  let d1 := distance_first_hour d2 in
  let d3 := distance_third_hour d2 in
  d1 + d2 + d3 = 37 :=
by 
  sorry

end james_total_distance_l66_66330


namespace problem1_problem2_l66_66862

theorem problem1 : ( (log 2 2)^2 + log 2 50 + log 2 25 = 2 ) :=
by
  sorry

theorem problem2 : ( (9 / 4 : ℝ)^(3 / 2) + (0.1:ℝ)^(-2) + (1 / 27 : ℝ)^(-1 / 3) + 2 * pi^0 = (867 / 8 : ℝ) ) :=
by
  sorry

end problem1_problem2_l66_66862


namespace cos_alpha_plus_pi_over_4_l66_66287

theorem cos_alpha_plus_pi_over_4
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = 3 / 5)
  (h4 : Real.sin (β - π / 4) = 5 / 13) : 
  Real.cos (α + π / 4) = 56 / 65 :=
by
  sorry 

end cos_alpha_plus_pi_over_4_l66_66287


namespace geometric_sequence_sum_inverse_equals_l66_66652

variable (a : ℕ → ℝ)
variable (n : ℕ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃(r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum_inverse_equals (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_sum : a 5 + a 6 + a 7 + a 8 = 15 / 8)
  (h_prod : a 6 * a 7 = -9 / 8) :
  (1 / a 5) + (1 / a 6) + (1 / a 7) + (1 / a 8) = -5 / 3 :=
by
  sorry

end geometric_sequence_sum_inverse_equals_l66_66652


namespace sum_of_odd_divisors_l66_66063

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l66_66063


namespace volume_ratio_l66_66756

theorem volume_ratio (A B C : ℕ) (h₁ : A = (B + C) / 4) (h₂ : B = (A + C) / 6) : 
  let m := 23 in
  let n := 12 in
  let ratio := m + n in
  ratio = 35 := 
sorry

end volume_ratio_l66_66756


namespace evaluate_fraction_l66_66216

theorem evaluate_fraction : (5 / 6 : ℚ) / (9 / 10) - 1 = -2 / 27 := by
  sorry

end evaluate_fraction_l66_66216


namespace train_B_time_to_reach_destination_l66_66748

theorem train_B_time_to_reach_destination
    (T t : ℝ)
    (train_A_speed : ℝ) (train_B_speed : ℝ)
    (train_A_extra_hours : ℝ)
    (h1 : train_A_speed = 110)
    (h2 : train_B_speed = 165)
    (h3 : train_A_extra_hours = 9)
    (h_eq : 110 * (T + train_A_extra_hours) = 110 * T + 165 * t) :
    t = 6 := 
by
  sorry

end train_B_time_to_reach_destination_l66_66748


namespace right_triangle_area_l66_66001

theorem right_triangle_area
  (a b x : ℝ)
  (h_angle : ∠C = 90)
  (h_perimeter : a + b + 2 * x = 72)
  (h_median_altitude_diff : x - (x - 7) = 7) :
  let S_ABC := x * (x - 7) in
  2 * S_ABC = a * b :=
sorry

end right_triangle_area_l66_66001


namespace sum_odd_divisors_of_90_l66_66108

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l66_66108


namespace sum_odd_divisors_of_90_l66_66104

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l66_66104


namespace regular_polygon_vertices_eq_12_l66_66185

-- Define that points A, B, C, D are consecutive vertices of a regular polygon.
variable (A B C D : Type) [IsRegularPolygon A B C D]

-- Assume angle ABD is 135 degrees
axiom angle_ABD : ∠(A, B, D) = 135

-- State the theorem that proves the number of vertices in this regular polygon is 12
theorem regular_polygon_vertices_eq_12 (n : ℕ) (h : IsRegularPolygon A B C D) (h_angle : ∠(A, B, D) = 135) : n = 12 :=
sorry

end regular_polygon_vertices_eq_12_l66_66185


namespace integer_pairs_count_l66_66975

theorem integer_pairs_count :
  (∃ (x y : ℤ), x^4 + y^2 = 4 * y) ∧
  (∀ (x y : ℤ), x^4 + y^2 = 4 * y → (x = 0 ∧ (y = 0 ∨ y = 4))) ∧
  (∃ (x1 x2 : ℕ), cardinality { p : ℤ × ℤ | p.1 ^ 4 + p.2 ^ 2 = 4 * p.2 } = 2) :=
sorry

end integer_pairs_count_l66_66975


namespace sum_of_roots_l66_66382

-- Define a second-degree polynomial with given conditions
def f (x : ℝ) := a * x^2 + b * x + c

-- Given conditions
def h1 : Prop := f 2 = 1
def h2 : Prop := f 4 = 2
def h3 : Prop := f 8 = 3

theorem sum_of_roots (a b c : ℝ) (h1 h2 h3 : Prop) : - b / a = 18 :=
begin
  sorry
end

end sum_of_roots_l66_66382


namespace eccentricity_is_correct_l66_66930

noncomputable def eccentricity_of_ellipse (a b c : ℝ) (h : 0 < c / a ∧ c / a < 1) 
  (F1 F2 : ℝ × ℝ) (M N : ℝ × ℝ) : ℝ :=
  let e := c / a in
  if (F1 = (-c, 0) ∧ F2 = (c, 0) ∧ (M = (-c, b * (a^2 - c^2)^0.5 / a) ∨ M = (-c, -b * (a^2 - c^2)^0.5 / a))
    ∧ (N = (c, b * (a^2 - c^2)^0.5 / a) ∨ N = (c, -b * (a^2 - c^2)^0.5 / a))
    ∧ (let triangle_is_isosceles_right_angle := (dist (M, N) (N, F2) = dist (N, F2) (F2, M)) in triangle_is_isosceles_right_angle)
  ∧ (a^2 - b^2 = c^2)) then -1 + Real.sqrt 2 else 0

theorem eccentricity_is_correct (a b c : ℝ) (h : 0 < c / a ∧ c / a < 1)
  (F1 F2 : ℝ × ℝ) (M N : ℝ × ℝ) (h1 : F1 = (-c, 0)) (h2 : F2 = (c, 0))
  (h3 : (M = (-c, b * (a^2 - c^2)^0.5 / a) ∨ M = (-c, -b * (a^2 - c^2)^0.5 / a))
       ∧ (N = (c, b * (a^2 - c^2)^0.5 / a) ∨ N = (c, -b * (a^2 - c^2)^0.5 / a)))
  (h4 : let triangle_is_isosceles_right_angle := (dist (M, N) (N, F2) = dist (N, F2) (F2, M)) in triangle_is_isosceles_right_angle)
  (h5 : a^2 - b^2 = c^2) : 
  eccentricity_of_ellipse a b c h F1 F2 M N = -1 + Real.sqrt 2 := by 
    sorry

end eccentricity_is_correct_l66_66930


namespace andover_no_rain_week_l66_66489

theorem andover_no_rain_week 
  (p_monday : ℚ = 1 / 2)
  (p_tuesday : ℚ = 1 / 3)
  (p_wednesday : ℚ = 1 / 4)
  (p_thursday : ℚ = 1 / 5)
  (p_friday : ℚ = 1 / 6)
  (p_saturday : ℚ = 1 / 7)
  (p_sunday : ℚ = 1 / 8) :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ m + n = 9 ∧ ((∏ i in (finset.range 7), 1 - (1/ (1 + i))) = 1 / 8) sorry

end andover_no_rain_week_l66_66489


namespace sum_odd_divisors_90_eq_78_l66_66053

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l66_66053


namespace train_speed_is_79_9664_kmph_l66_66165

def train_length : ℝ := 200  -- Length of the train in meters
def platform_length : ℝ := 288.928  -- Length of the platform in meters
def crossing_time : ℝ := 22  -- Time to cross the platform in seconds

def total_distance : ℝ := train_length + platform_length  -- Total distance covered by the train
def speed_mps : ℝ := total_distance / crossing_time  -- Speed in meters per second

def conversion_factor : ℝ := 3.6  -- Conversion factor from m/s to km/h
def speed_kmph : ℝ := speed_mps * conversion_factor  -- Speed in kilometers per hour

theorem train_speed_is_79_9664_kmph :
  speed_kmph ≈ 79.9664 :=
sorry

end train_speed_is_79_9664_kmph_l66_66165


namespace midpoint_after_transformations_l66_66878

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define basic operations for the transformations
def rotate180 (p₁ p₂ : Point) : Point :=
  ⟨p₁.x * 2 - p₂.x, p₁.y * 2 - p₂.y⟩

def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

def midpoint (p₁ p₂ : Point) : Point :=
  ⟨(p₁.x + p₂.x) / 2, (p₁.y + p₂.y) / 2⟩

-- Points
def P : Point := ⟨1, 2⟩
def Q : Point := ⟨4, 6⟩
def R : Point := ⟨7, 2⟩

-- Rotate Q and R 180 degrees around P
def Q'' := rotate180 P Q
def R'' := rotate180 P R

-- Translate Q'' and R'' three units right and four units down
def Q' := translate Q'' 3 (-4)
def R' := translate R'' 3 (-4)

-- Statement of the proof problem
theorem midpoint_after_transformations :
  midpoint Q' R' = ⟨-1 / 2, -4⟩ := by
  sorry

end midpoint_after_transformations_l66_66878


namespace infinite_even_and_odd_partitions_l66_66383

def partition_function (n : ℕ) : ℕ :=
  -- Definition of p(n) would go here; this is typically noncomputable.
  sorry

theorem infinite_even_and_odd_partitions :
  (∃ᶠ n in at_top, even (partition_function n)) ∧ (∃ᶠ n in at_top, odd (partition_function n)) :=
sorry

end infinite_even_and_odd_partitions_l66_66383


namespace max_sides_convex_polygon_with_obtuse_angles_l66_66610

-- Definition of conditions
def is_convex_polygon (n : ℕ) : Prop := n ≥ 3
def obtuse_angles (n : ℕ) (k : ℕ) : Prop := k = 3 ∧ is_convex_polygon n

-- Statement of the problem
theorem max_sides_convex_polygon_with_obtuse_angles (n : ℕ) :
  obtuse_angles n 3 → n ≤ 6 :=
sorry

end max_sides_convex_polygon_with_obtuse_angles_l66_66610


namespace sara_total_spent_l66_66363

-- Definitions based on the conditions
def ticket_price : ℝ := 10.62
def discount_rate : ℝ := 0.10
def rented_movie : ℝ := 1.59
def bought_movie : ℝ := 13.95
def snacks : ℝ := 7.50
def sales_tax_rate : ℝ := 0.05

-- Problem statement
theorem sara_total_spent : 
  let total_tickets := 2 * ticket_price
  let discount := total_tickets * discount_rate
  let discounted_tickets := total_tickets - discount
  let subtotal := discounted_tickets + rented_movie + bought_movie
  let sales_tax := subtotal * sales_tax_rate
  let total_with_tax := subtotal + sales_tax
  let total_amount := total_with_tax + snacks
  total_amount = 43.89 :=
by
  sorry

end sara_total_spent_l66_66363


namespace total_points_yolanda_season_l66_66438

-- Define the parameters and conditions
def numGames := 15
def avgFreeThrows := 4
def avgTwoPointBaskets := 5
def avgThreePointBaskets := 3
def pointsPerFreeThrow := 1
def pointsPerTwoPointBasket := 2
def pointsPerThreePointBasket := 3

-- Calculate the total points per game from the given averages
def pointsPerGame := 
  avgFreeThrows * pointsPerFreeThrow +
  avgTwoPointBaskets * pointsPerTwoPointBasket +
  avgThreePointBaskets * pointsPerThreePointBasket

-- Total points over the entire season
def totalPoints := pointsPerGame * numGames

/-- Prove that the total points Yolanda scored over the entire season equals 345. -/
theorem total_points_yolanda_season : totalPoints = 345 :=
by 
  -- We define the values as stated in the conditions
  have h1 : pointsPerGame = 4 * 1 + 5 * 2 + 3 * 3 := rfl
  have h2 : totalPoints = pointsPerGame * 15 := rfl
  
  -- Substitute and calculate
  rw h1,
  rw h2,
  norm_num,
  sorry

end total_points_yolanda_season_l66_66438


namespace average_length_tapes_l66_66395

def lengths (l1 l2 l3 l4 l5 : ℝ) : Prop :=
  l1 = 35 ∧ l2 = 29 ∧ l3 = 35.5 ∧ l4 = 36 ∧ l5 = 30.5

theorem average_length_tapes
  (l1 l2 l3 l4 l5 : ℝ)
  (h : lengths l1 l2 l3 l4 l5) :
  (l1 + l2 + l3 + l4 + l5) / 5 = 33.2 := 
by
  sorry

end average_length_tapes_l66_66395


namespace choose_intervals_l66_66879

-- Definitions and conditions based on the problem statement
def intervals (n : ℕ) : Type := finset (set.Ioo (real) (real))

def A (k : ℕ) := {s : intervals k | s.card = k} -- set representing A_k

-- The proof problem statement in Lean
theorem choose_intervals (n : ℕ) (A : ℕ → Type) (H : ∀ k, A k → intervals k)
  (disjoint : ∀ k (a : A k), ∀ i j : set.Ioo real real, i ≠ j → set.disjoint i j)
  (H_Ak : ∀ k (a : A k), card (H k a) = k) :
  ∃ (S : finset (Σ (k : ℕ), A k)), S.card = (n + 1)/2 ∧
    (∀ (k₁ k₂ : ℕ) (a₁ : A k₁) (a₂ : A k₂), (⟨k₁, a₁⟩ ∈ S ∧ ⟨k₂, a₂⟩ ∈ S) → k₁ ≠ k₂) ∧
    ∀ (s t : set.Ioo real real) (k₁ k₂ : ℕ) (a₁ : A k₁) (a₂ : A k₂),
      (⟨k₁, a₁⟩ ∈ S ∧ ⟨k₂, a₂⟩ ∈ S ∧ s ∈ H k₁ a₁ ∧ t ∈ H k₂ a₂) → set.disjoint s t :=
begin
  sorry
end

end choose_intervals_l66_66879


namespace quiz_win_probability_is_13_over_256_l66_66471

open ProbabilityTheory

noncomputable def quizWinProbability : ℚ :=
  let probabilityCorrect := (1 : ℚ) / 4
  let probabilityWrong := (3 : ℚ) / 4
  let probabilityAllFourCorrect := probabilityCorrect ^ 4
  let probabilityExactlyThreeCorrect := 4 * (probabilityCorrect ^ 3 * probabilityWrong)
  probabilityAllFourCorrect + probabilityExactlyThreeCorrect

theorem quiz_win_probability_is_13_over_256 :
  quizWinProbability = (13 : ℚ) / 256 :=
by
  sorry

end quiz_win_probability_is_13_over_256_l66_66471


namespace min_value_equal_l66_66210

noncomputable def min_possible_value (a b c : ℝ) : ℝ :=
  a / (3 * b) + b / (6 * c) + c / (9 * a)

theorem min_value_equal (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  ∃ (d : ℝ), d = 3 / real.cbrt 162 ∧ min_possible_value a b c ≥ d :=
sorry

end min_value_equal_l66_66210


namespace correct_propositions_count_l66_66250

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α) (d : α)

def arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def num_correct_propositions (a : ℕ → α) (d : α) : ℕ :=
  let prop1 := ∀ n, a (n + 1) - a n > 0
  let prop2 := ∀ n, (n + 1) * a (n + 1) - n * a n > 0
  let prop3 := ∀ n, (a (n + 1) / (n + 1)) - (a n / n) > 0
  let prop4 := ∀ n, a (n + 1) + 3 * (n + 1) * d - (a n + 3 * n * d) > 0
  (if prop1 then 1 else 0) + (if prop4 then 1 else 0) -- prop1 and prop4 are true

theorem correct_propositions_count (a : ℕ → α) (d : α) (hd : d > 0) :
  arithmetic_sequence a d →
  num_correct_propositions a d = 2 :=
by
  intros h_seq
  have h1 : ∀ n, a (n + 1) - a n = d := by sorry
  have h_prop1 : ∀ n, a (n + 1) - a n > 0 := by
    intros n
    rw ← h1 n
    exact hd
  have h_prop4 : ∀ n, a (n + 1) + 3 * (n + 1) * d - (a n + 3 * n * d) > 0 := by
    intros n
    calc
      a (n + 1) + 3 * (n + 1) * d - (a n + 3 * n * d)
        = (a (n + 1) - a n) + 3 * d := by ring
      ... = 4 * d := by rw h1 n; ring
      ... > 0 := by exact mul_pos four_pos hd
  -- Not proving prop2 and prop3 explicitly here as they would take more steps.
  -- Just assuming they are false as per the original solution for count accuracy.
  sorry

end correct_propositions_count_l66_66250


namespace arctan_tan_sub_eq_l66_66503

noncomputable def arctan_tan_sub (a b : ℝ) : ℝ := Real.arctan (Real.tan a - 3 * Real.tan b)

theorem arctan_tan_sub_eq (a b : ℝ) (ha : a = 75) (hb : b = 15) :
  arctan_tan_sub a b = 75 :=
by
  sorry

end arctan_tan_sub_eq_l66_66503


namespace function_must_pass_through_point_l66_66731

theorem function_must_pass_through_point :
  ∀ k : ℝ, ∃ x y : ℝ, (y = k * x - k + 2) ∧ (x = 1) ∧ (y = 2) :=
by {
  intros k,
  use 1,
  use 2,
  simp,
  sorry
}

end function_must_pass_through_point_l66_66731


namespace area_of_circle_II_is_36pi_l66_66192

-- Define the radii of the circles
def radius_circle_I : ℝ := 3
def radius_circle_II : ℝ := 2 * radius_circle_I

-- Define the area calculation
def area_circle (r : ℝ) : ℝ := π * r^2

-- The proof statement: prove that the area of Circle II is 36π square inches
theorem area_of_circle_II_is_36pi : area_circle radius_circle_II = 36 * π := by
  sorry

end area_of_circle_II_is_36pi_l66_66192


namespace sqrt_combination_l66_66114

variable (x : ℝ)

theorem sqrt_combination:
  (∃ k : ℝ, sqrt 48 = k * sqrt 3) :=
begin
  use 4,
  simp [sqrt_mul (nat.succ_pos 15) (nat.succ_pos 2)],
  simp [sqrt_eq_rpow],
  ring,
end

end sqrt_combination_l66_66114


namespace range_of_a_l66_66978

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + a * x + a = 0) → a ∈ Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

end range_of_a_l66_66978


namespace problem_1_problem_2_problem_3_l66_66595

open Real

noncomputable def parabola : set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}
noncomputable def line_x4 : set (ℝ × ℝ) := {p | p.1 = 4}

noncomputable def line_pa (p a : ℝ × ℝ) : set (ℝ × ℝ) := {q | q.2 = p.2 + (a.2 - p.2) / (a.1 - p.1) * (q.1 - p.1)}
noncomputable def line_pb (p a : ℝ × ℝ) : set (ℝ × ℝ) := {q | q.2 = p.2 + (a.2 - p.2) / (a.1 - p.1) * (q.1 - p.1)}

theorem problem_1 (p : ℝ × ℝ) (h_p : p.1 < 4 ∧ p.2 ≥ 0) (h_parabola : p ∈ parabola)
    (h_triangle : 1/2 * abs (8 * (4 - p.1)) = 4) : p = (3, 2 * sqrt 3) :=
by
  sorry

theorem problem_2 (p : ℝ × ℝ) (a b : ℝ × ℝ) (h_pa_perp_pb : (line_pa p a).coprime (line_pb p b))
    (h_p : p ∈ line_x4) : dist p a = 4 * sqrt 2 :=
by
  sorry

theorem problem_3 (p m n : ℝ × ℝ) (h_p_m : m ∈ line_pa p) (h_p_n : n ∈ line_pb p)
    (h_pm_eq_pab : abs (m.1 - n.1) * p.2 / 2 = 4) : 1/2 * p.2^2 = 8 :=
by
  sorry

end problem_1_problem_2_problem_3_l66_66595


namespace compute_sum_pq_pr_qr_l66_66702

theorem compute_sum_pq_pr_qr (p q r : ℝ) (h : 5 * (p + q + r) = p^2 + q^2 + r^2) : 
  let N := 150
  let n := -12.5
  N + 15 * n = -37.5 := 
by {
  sorry
}

end compute_sum_pq_pr_qr_l66_66702


namespace degrees_basic_astrophysics_l66_66782

def percentage_mic : ℝ := 9
def percentage_home : ℝ := 14
def percentage_food : ℝ := 10
def percentage_gmo : ℝ := 29
def percentage_indu : ℝ := 8
def total_percentage := 100
def total_degrees := 360

theorem degrees_basic_astrophysics :
    total_degrees * ((total_percentage - (percentage_mic + percentage_home + percentage_food + 
    percentage_gmo + percentage_indu)) / total_percentage) = 108 :=
by
  have h1 : (percentage_mic + percentage_home + percentage_food + percentage_gmo + percentage_indu) = 70 := by sorry
  have h2 : (total_percentage - 70) = 30 := by sorry
  have h3 : total_degrees * (30 / total_percentage) = 108 := by sorry
  exact h3

end degrees_basic_astrophysics_l66_66782


namespace min_period_f_max_f_at_3_l66_66589

def f (x m : ℝ) : ℝ := (Real.sin ((Real.pi / 2) + x) - Real.sin x)^2 + m

theorem min_period_f (m : ℝ) : (∀ x, f (x + Real.pi) m = f x m) :=
by sorry

theorem max_f_at_3 (m : ℝ) : (∀ x, f x m ≤ 3) → m = 1 :=
by sorry

end min_period_f_max_f_at_3_l66_66589


namespace exists_prime_mod_greater_remainder_l66_66688

theorem exists_prime_mod_greater_remainder (a b : ℕ) (h1 : 0 < a) (h2 : a < b) :
  ∃ p : ℕ, Prime p ∧ a % p > b % p :=
sorry

end exists_prime_mod_greater_remainder_l66_66688


namespace math_proof_l66_66815

-- Definitions based on given conditions
def groupA_scores : List ℕ := [5, 6, 6, 6, 6, 6, 7, 9, 9, 10]
def groupB_scores : List ℕ := [5, 6, 6, 6, 7, 7, 7, 7, 9, 10]

-- Preliminary details
def groupA_mean := 7
def groupA_mode := 6
def groupA_variance := 2.6

-- Proof problem statement
theorem math_proof :
  (median groupA_scores = 6) ∧
  (mean groupB_scores = 7) ∧
  (mode groupB_scores = 7) ∧
  (Xiaoming_is_from_group (mean groupA_scores) 7 = "Group A") ∧
  (variance groupB_scores < variance groupA_scores) :=
by {
  sorry -- Proof details to be filled in
}

end math_proof_l66_66815


namespace tenth_permutation_is_2561_l66_66005

-- Define the digits
def digits : List ℕ := [1, 2, 5, 6]

-- Define the permutations of those digits
def perms := digits.permutations.map (λ l, l.asString.toNat)

-- Define the 10th integer in the sorted list of those permutations
noncomputable def tenth_integer : ℕ := (perms.sort (≤)).nthLe 9 (by norm_num [List.length_permutations, perms])

-- The statement we want to prove
theorem tenth_permutation_is_2561 : tenth_integer = 2561 :=
sorry

end tenth_permutation_is_2561_l66_66005


namespace sum_of_odd_divisors_of_90_l66_66084

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l66_66084


namespace inverse_proportion_function_l66_66732

theorem inverse_proportion_function (x y : ℝ) (h : y = 6 / x) : x * y = 6 :=
by
  sorry

end inverse_proportion_function_l66_66732


namespace sum_of_odd_divisors_l66_66064

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l66_66064


namespace imaginary_part_of_z_l66_66920

theorem imaginary_part_of_z
    (z : ℂ)
    (h : (1 - complex.I) * z = 2 * complex.I) :
    complex.im z = 1 :=
sorry

end imaginary_part_of_z_l66_66920


namespace optimal_path_length_l66_66616

theorem optimal_path_length (n : ℕ) (coords : Fin n.succ → ℕ) 
  (h_perm : ∀ i, coords i < n.succ ∧ ∀ i j, i ≠ j → coords i ≠ coords j) :
  (if even n then (∃ d, d = (n^2 + 2 * n - 2) / 2) else (∃ d, d = (n^2 + 2 * n - 1) / 2)) :=
by sorry

end optimal_path_length_l66_66616


namespace limit_ln_eq_half_ln_three_l66_66189

theorem limit_ln_eq_half_ln_three :
  (Real.lim (λ x : ℝ, ln ((exp (x^2) - cos x) * cos (1 / x) + tan (x + Real.pi / 3))) 0) = 1 / 2 * ln 3 :=
sorry

end limit_ln_eq_half_ln_three_l66_66189


namespace sum_of_odd_divisors_of_90_l66_66087

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l66_66087


namespace sumOddDivisorsOf90_l66_66042

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l66_66042


namespace line_RS_bisects_FB_l66_66736
noncomputable theory

open EuclideanGeometry

variable {k : Circle}   -- Circle k
variable {A B F N P Q R S H : Point}  -- Points A, B, F, N, P, Q, R, S, H

-- Given conditions:
variable (h1 : Midpoint (chord A B k) F)          -- F is the midpoint of chord AB of circle k.
variable (h2 : divides_closer A N F 4)            -- N divides AF into fourths (close to A).
variable (h3 : line_through N line_intersects_circle k P Q)  -- A line through N intersects the circle at P and Q.
variable (h4 : circle_intersect_line k P F R)     -- PF intersects the circle again at R.
variable (h5 : circle_intersect_line k Q F S)     -- QF intersects the circle again at S.

-- The statement to prove:
theorem line_RS_bisects_FB (h1 h2 h3 h4 h5 : Prop) : bisects_line RS segment FB :=
sorry

end line_RS_bisects_FB_l66_66736


namespace determine_value_l66_66516

theorem determine_value : 3 - ((-3)⁻³ : ℚ) = 82 / 27 := by
  sorry

end determine_value_l66_66516


namespace percentage_z_exceeds_x_l66_66280

-- Define the conditions
variable {x y z : ℝ}
variable (h1 : x = 0.55 * y)
variable (h2 : y = 1.30 * z)

-- Define the theorem to prove
theorem percentage_z_exceeds_x : 
  z > x → ((z - x) / x) * 100 ≈ 39.86 :=
by
  sorry

end percentage_z_exceeds_x_l66_66280


namespace households_with_bike_only_l66_66631

theorem households_with_bike_only (total households no_car_nor_bike both_cosb having_car : ℕ) :
  total = 90 →
  no_car_nor_bike = 11 →
  both_cosb = 18 →
  having_car = 44 →
  total - no_car_nor_bike - (having_car - both_cosb + both_cosb) = 35 :=
by
  intros h_total h_no_car_nor_bike h_both_cosb h_having_car
  have h_with_car_or_bike := h_total - h_no_car_nor_bike
  have h_only_car := h_having_car - h_both_cosb
  have h_only_bike := h_with_car_or_bike - (h_only_car + h_both_cosb)
  sorry

end households_with_bike_only_l66_66631


namespace board_zero_condition_l66_66831

theorem board_zero_condition (m n : ℕ) (board : array (array ℕ)) : 
  (∃k, ∀i j, (board[i][j] = board[i+1][j] + k ∧ board[i][j+1] = board[i][j] + k)) ↔ 
  Σ_{i,j | (i + j) % 2 = 0} board[i][j] = Σ_{i,j | (i + j) % 2 = 1} board[i][j] := 
sorry

end board_zero_condition_l66_66831


namespace number_of_measures_of_C_l66_66000

theorem number_of_measures_of_C (C D : ℕ) (h1 : C + D = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ C = k * D) : 
  ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_measures_of_C_l66_66000


namespace derivative_f_derivative_f_at_2_l66_66593

noncomputable def f : ℝ → ℝ := λ x, x^2 + x

-- The first statement: derivative of f(x) is 2x + 1
theorem derivative_f : deriv f x = 2 * x + 1 := sorry

-- The second statement: value of the derivative at x = 2 is 5
theorem derivative_f_at_2 : deriv f 2 = 5 := sorry

end derivative_f_derivative_f_at_2_l66_66593


namespace probability_is_seven_fifteenths_l66_66329

-- Define the problem conditions
def total_apples : ℕ := 10
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def yellow_apples : ℕ := 2
def choose_3_from_10 : ℕ := Nat.choose 10 3
def choose_3_red : ℕ := Nat.choose 5 3
def choose_3_green : ℕ := Nat.choose 3 3
def choose_2_red_1_green : ℕ := Nat.choose 5 2 * Nat.choose 3 1
def choose_2_green_1_red : ℕ := Nat.choose 3 2 * Nat.choose 5 1

-- Calculate favorable outcomes
def favorable_outcomes : ℕ :=
  choose_3_red + choose_3_green + choose_2_red_1_green + choose_2_green_1_red

-- Calculate the required probability
def probability_all_red_or_green : ℚ := favorable_outcomes / choose_3_from_10

-- Prove that probability_all_red_or_green is 7/15
theorem probability_is_seven_fifteenths :
  probability_all_red_or_green = 7 / 15 :=
by 
  -- Leaving the proof as a sorry for now
  sorry

end probability_is_seven_fifteenths_l66_66329


namespace profit_percentage_l66_66830

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 500) (hSP : SP = 625) : 
  ((SP - CP) / CP) * 100 = 25 := 
by 
  sorry

end profit_percentage_l66_66830


namespace charity_donations_l66_66490

theorem charity_donations : 
  let total_earnings := 500
  let cost_of_ingredients := 110
  let remaining_amount := total_earnings - cost_of_ingredients
  let homeless_shelter := 0.30 * remaining_amount + 15
  let food_bank := 0.25 * remaining_amount + 15
  let park_committee := 0.20 * remaining_amount + 15
  let animal_rescue := 0.25 * remaining_amount + 15
  in 
  homeless_shelter = 132 ∧ food_bank = 112.50 ∧ park_committee = 93 ∧ animal_rescue = 112.50 :=
by
  sorry

end charity_donations_l66_66490


namespace earnings_difference_l66_66787

theorem earnings_difference (x y : ℕ) 
  (h1 : 3 * 6 + 4 * 5 + 5 * 4 = 58)
  (h2 : x * y = 12500) 
  (total_earnings : (3 * 6 * x * y / 100 + 4 * 5 * x * y / 100 + 5 * 4 * x * y / 100) = 7250) :
  4 * 5 * x * y / 100 - 3 * 6 * x * y / 100 = 250 := 
by 
  sorry

end earnings_difference_l66_66787


namespace sum_of_coefficients_eq_30_l66_66904

theorem sum_of_coefficients_eq_30 :
  let p := 4 * (2 * x^6 - 5 * x^3 + 9) + 6 * (3 * x^7 - 4 * x^4 + 2)
  (eval 1 p) = 30 :=
by
  have p_def : Polynomial ℝ :=  4 * (2 * X^6 - 5 * X^3 + 9) + 6 * (3 * X^7 - 4 * X^4 + 2)
  have eval_p_1 : eval 1 p_def = 30 := sorry
  exact eval_p_1

end sum_of_coefficients_eq_30_l66_66904


namespace cone_surface_area_l66_66806

-- Define the surface area formula for a cone with radius r and slant height l
theorem cone_surface_area (r l : ℝ) : 
  let S := π * r^2 + π * r * l
  S = π * r^2 + π * r * l :=
by sorry

end cone_surface_area_l66_66806


namespace power_function_through_point_l66_66562

theorem power_function_through_point :
  (∃ α : ℝ, ∀ x : ℝ, (x = real.sqrt 2 → x^α = 2 * real.sqrt 2)) → (∃ f : ℝ → ℝ, f = (λ x, x^3)) :=
by
sorry

end power_function_through_point_l66_66562


namespace pinedale_mall_distance_l66_66715

theorem pinedale_mall_distance :
  let times := [4/60, 6/60, 5/60, 7/60] in
  let speeds := [55, 65, 60, 70] in
  let distances := list.zipWith (λ t s => s * t) times speeds in
  list.sum distances = 23.34 :=
by sorry

end pinedale_mall_distance_l66_66715


namespace sum_of_odd_divisors_l66_66062

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l66_66062


namespace sequence_bounded_by_two_l66_66377

def a_seq : ℕ → ℝ
| 0       := 0
| (n + 1) := real.sqrt (1 + a_seq_aux n)
with a_seq_aux : ℕ → ℝ
| 0       := real.sqrt 2
| (n + 1) := real.sqrt (2 + 2 ^ (n + 1) + a_seq_aux n)  

theorem sequence_bounded_by_two (n : ℕ) (h : 1 ≤ n) : a_seq n < 2 :=
by
  sorry

end sequence_bounded_by_two_l66_66377


namespace people_with_diploma_percentage_l66_66638

-- Definitions of the given conditions
def P_j_and_not_d := 0.12
def P_not_j_and_d := 0.15
def P_j := 0.40

-- Definitions for intermediate values
def P_not_j := 1 - P_j
def P_not_j_d := P_not_j * P_not_j_and_d

-- Definition of the result to prove
def P_d := (P_j - P_j_and_not_d) + P_not_j_d

theorem people_with_diploma_percentage : P_d = 0.43 := by
  -- Placeholder for the proof
  sorry

end people_with_diploma_percentage_l66_66638


namespace sum_of_column_products_non_decreasing_l66_66305

open Matrix

theorem sum_of_column_products_non_decreasing {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
  (hpos : ∀ i j, 0 < A i j) :
  -- Define S_original as the sum of column products
  let S_original := (Finset.univ.sum (λ j, ∏ i, A i j)) in
  -- Define S_sorted as the sum of column products after sorting rows
  let A_sorted := fun i => sort (Finset.univ.map A i) in
  
  let S_sorted := (Finset.univ.sum (λ j, ∏ i, A_sorted i j)) in
  S_sorted ≥ S_original :=
begin
  sorry -- proof left as an exercise
end

end sum_of_column_products_non_decreasing_l66_66305


namespace ratio_sum_of_square_diagonal_intersections_l66_66642

theorem ratio_sum_of_square_diagonal_intersections
  (A B C D E F G P Q R : ℝ×ℝ)
  (h_square : (A, B, C, D) in square 8)
  (h_divide : divide_into_equal_segments B C [E, F, G] 4)
  (h_intersect1 : line_through A E ∩ diagonal B D = {P})
  (h_intersect2 : line_through A F ∩ diagonal B D = {Q})
  (h_intersect3 : line_through A G ∩ diagonal B D = {R}):
  let BP := distance B P,
      PQ := distance P Q,
      QR := distance Q R,
      RD := distance R D in
  BP + PQ + QR + RD = 11 :=
sorry

end ratio_sum_of_square_diagonal_intersections_l66_66642


namespace sum_of_digits_of_x_l66_66129

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s.reverse == s

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_x :
  ∃ x : ℕ, (is_palindrome x) ∧ (is_palindrome (x + 40)) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (sum_of_digits x = 16) :=
sorry

end sum_of_digits_of_x_l66_66129


namespace sum_of_odd_divisors_l66_66071

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l66_66071


namespace math_problem_l66_66447

def Q (f : ℝ → ℝ) : Prop :=
  (∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x + y ≠ 0 → f (1 / (x + y)) = f (1 / x) + f (1 / y))
  ∧ (∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x + y ≠ 0 → (x + y) * f (x + y) = x * y * f x * f y)
  ∧ f 1 = 1

theorem math_problem (f : ℝ → ℝ) : Q f → (∀ (x : ℝ), x ≠ 0 → f x = 1 / x) :=
by
  -- Proof goes here
  sorry

end math_problem_l66_66447


namespace slope_of_tangent_line_l66_66409

def f (x : ℝ) : ℝ := x * Real.exp x

theorem slope_of_tangent_line :
  (deriv f 1) = 2 * Real.exp 1 := by 
  sorry

end slope_of_tangent_line_l66_66409


namespace incorrect_judgment_l66_66261

variables {f : ℝ → ℝ} {a b c d : ℝ}
variables (h_decreasing : ∀ {x y : ℝ}, 0 < x → 0 < y → x < y → f(x) > f(y))
variables (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
variables (h_order : a < b ∧ b < c)
variables (h_product : f(a) * f(b) * f(c) < 0)
variables (h_solution : f(d) = 0)

theorem incorrect_judgment :
  ¬ (d > c) := 
sorry

end incorrect_judgment_l66_66261


namespace pos_rel_lines_l66_66405

-- Definition of the lines
def line1 (k : ℝ) (x y : ℝ) : Prop := 2 * x - y + k = 0
def line2 (x y : ℝ) : Prop := 4 * x - 2 * y + 1 = 0

-- Theorem stating the positional relationship between the two lines
theorem pos_rel_lines (k : ℝ) : 
  (∀ x y : ℝ, line1 k x y → line2 x y → 2 * k - 1 = 0) → 
  (∀ x y : ℝ, line1 k x y → ¬ line2 x y → 2 * k - 1 ≠ 0) → 
  (k = 1/2 ∨ k ≠ 1/2) :=
by sorry

end pos_rel_lines_l66_66405


namespace four_brothers_money_l66_66548

theorem four_brothers_money 
  (a_1 a_2 a_3 a_4 : ℝ) 
  (x : ℝ)
  (h1 : a_1 + a_2 + a_3 + a_4 = 48)
  (h2 : a_1 + 3 = x)
  (h3 : a_2 - 3 = x)
  (h4 : 3 * a_3 = x)
  (h5 : a_4 / 3 = x) :
  a_1 = 6 ∧ a_2 = 12 ∧ a_3 = 3 ∧ a_4 = 27 :=
by
  sorry

end four_brothers_money_l66_66548


namespace existence_of_positive_integers_l66_66378

theorem existence_of_positive_integers (n : ℕ) (hn : n > 0) : 
  ∃ x : Fin n → ℕ, (∀ i, x i > 0) ∧ (x.sum = x.prod) :=
by
  sorry

end existence_of_positive_integers_l66_66378


namespace circle_standard_form1_circle_standard_form2_l66_66881

theorem circle_standard_form1 (x y : ℝ) :
  x^2 + y^2 - 4 * x + 6 * y - 3 = 0 ↔ (x - 2)^2 + (y + 3)^2 = 16 :=
by
  sorry

theorem circle_standard_form2 (x y : ℝ) :
  4 * x^2 + 4 * y^2 - 8 * x + 4 * y - 11 = 0 ↔ (x - 1)^2 + (y + 1 / 2)^2 = 4 :=
by
  sorry

end circle_standard_form1_circle_standard_form2_l66_66881


namespace kabadi_kho_kho_players_l66_66809

theorem kabadi_kho_kho_players (total_players kabadi_only kho_kho_only both_games : ℕ)
  (h1 : kabadi_only = 10)
  (h2 : kho_kho_only = 40)
  (h3 : total_players = 50)
  (h4 : kabadi_only + kho_kho_only - both_games = total_players) :
  both_games = 0 := by
  sorry

end kabadi_kho_kho_players_l66_66809


namespace red_ball_removal_l66_66300

theorem red_ball_removal (total_balls : ℕ) (initial_red_percent : ℚ) (final_red_percent : ℚ) (initial_total : total_balls = 800) (initial_red : initial_red_percent = 0.7) (final_red : final_red_percent = 0.6) : ∃ x : ℕ, 560 - x = 0.6 * (800 - x) ∧ x = 200 :=
by
  sorry

end red_ball_removal_l66_66300


namespace option_a_is_correct_l66_66434

theorem option_a_is_correct (a b : ℝ) : 
  (a^2 + a * b) / a = a + b := 
by sorry

end option_a_is_correct_l66_66434


namespace fraction_of_number_l66_66034

-- Given definitions based on the problem conditions
def fraction : ℚ := 3 / 4
def number : ℕ := 40

-- Theorem statement to be proved
theorem fraction_of_number : fraction * number = 30 :=
by
  sorry -- This indicates that the proof is not yet provided

end fraction_of_number_l66_66034


namespace shift_graph_sin_l66_66419

theorem shift_graph_sin :
  ∀ (x : ℝ), (sin (2 * x + 2) = sin 2 (x + 1)) :=
by
  assume x : ℝ
  calc
    sin (2 * x + 2) = sin (2 * (x + 1)) : by sorry

end shift_graph_sin_l66_66419


namespace Simson_line_bisects_angle_between_Simson_lines_locus_of_intersection_points_l66_66985

universe u

variables {P P1 P2 : Type u}

structure TriangleCircumcircle (P : Type u) :=
  (A B C : P)
  (circumcircle : set P)
  (φ : P → P)

def SimsonLine (T : TriangleCircumcircle P) (P : P) : set P :=
  sorry  -- Definition of the Simson line logic

theorem Simson_line_bisects (T : TriangleCircumcircle P) (P : P) :
  let orthocenter : P := sorry  -- calculate orthocenter of T (ABC)
  let midpoint : P := sorry  -- find midpoint of P and orthocenter
  SimsonLine T P bisects (segment P orthocenter) := sorry

theorem angle_between_Simson_lines (T : TriangleCircumcircle P) (P1 P2 : P) :
  let angle := sorry  -- inscribed angle subtended by arc P1P2
  ∃ s1 s2 : set P, s1 = SimsonLine T P1 ∧ s2 = SimsonLine T P2 ∧
  angle_between s1 s2 = angle := sorry

theorem locus_of_intersection_points (T : TriangleCircumcircle P) :
  let nine_point_circle : set P := sorry  -- construct nine-point circle of T (ABC)
  ∀ P1 P2 on variable diameter of T.circumcircle,
  let intersection_points := sorry  -- intersection of Simson lines corresponding to P1 P2
  intersection_points ⊆ nine_point_circle := sorry

end Simson_line_bisects_angle_between_Simson_lines_locus_of_intersection_points_l66_66985


namespace fingers_game_conditions_l66_66651

noncomputable def minNForWinningSubset (N : ℕ) : Prop :=
  N ≥ 220

-- To state the probability condition, we need to express it in terms of actual probabilities
noncomputable def probLeaderWins (N : ℕ) : ℝ := 
  1 / N

noncomputable def leaderWinProbabilityTendsToZero : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, probLeaderWins n < ε

theorem fingers_game_conditions (N : ℕ) (probLeaderWins : ℕ → ℝ) :
  (minNForWinningSubset N) ∧ leaderWinProbabilityTendsToZero :=
by
  sorry

end fingers_game_conditions_l66_66651


namespace combined_score_l66_66624

variable (A J M : ℕ)

-- Conditions
def Jose_score_more_than_Alisson : Prop := J = A + 40
def Meghan_score_less_than_Jose : Prop := M = J - 20
def total_possible_score : ℕ := 100
def Jose_questions_wrong (wrong_questions : ℕ) : Prop := J = total_possible_score - (wrong_questions * 2)

-- Proof statement
theorem combined_score (h1 : Jose_score_more_than_Alisson)
                       (h2 : Meghan_score_less_than_Jose)
                       (h3 : Jose_questions_wrong 5) :
                       A + J + M = 210 := by
  sorry

end combined_score_l66_66624


namespace min_value_equal_l66_66209

noncomputable def min_possible_value (a b c : ℝ) : ℝ :=
  a / (3 * b) + b / (6 * c) + c / (9 * a)

theorem min_value_equal (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  ∃ (d : ℝ), d = 3 / real.cbrt 162 ∧ min_possible_value a b c ≥ d :=
sorry

end min_value_equal_l66_66209


namespace sum_odd_divisors_90_eq_78_l66_66056

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l66_66056


namespace sumOddDivisorsOf90_l66_66049

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l66_66049


namespace hyperbola_eccentricity_is_4_l66_66245

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h_eq1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : ∀ y : ℝ, y^2 = 16 * (4 : ℝ))
  (h_focus : c = 4)
: ℝ := c / a

theorem hyperbola_eccentricity_is_4 (a b c : ℝ)
  (h_eq1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : ∀ y : ℝ, y^2 = 16 * (4 : ℝ))
  (h_focus : c = 4)
  (h_c2 : c^2 = a^2 + b^2)
  (h_bc : b^2 = a^2 * (c^2 / a^2 - 1))
: hyperbola_eccentricity a b c h_eq1 h_eq2 h_focus = 4 := by
  sorry

end hyperbola_eccentricity_is_4_l66_66245


namespace problem1_problem2_l66_66251

-- Define the Ellipse with relevant parameters
structure Ellipse where
  a b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  b_less_a : b < a
  eccentricity : ℝ

-- Define the foci and relevant geometric properties
structure FociPoint where
  x y : ℝ

-- Define a circle structure
structure Circle where
  center : FociPoint
  radius : ℝ

-- Define the problem statement in Lean
theorem problem1 :
  ∃ e : Ellipse, e.eccentricity = (2 * Real.sqrt 2) / 3 →
  (∃ A B : FociPoint, ∃ f1 : FociPoint, ∃ f2 : FociPoint, 
    let AF1 := distance A f1 in
    let M := midpoint A f1 in
    let tangent_circle := Circle.mk M (AF1/2) in
    tangent_circle.radius = 3 →
    2 * e.a = 6) :=
sorry

theorem problem2 (b c a : ℝ) (b_eq : b = 1) (c_eq : c = 2 * Real.sqrt 2) (a_eq : a = 3) :
  ∃ T : FociPoint,
  T = FociPoint.mk (-19 * Real.sqrt 2 / 9) 0 →
  (∃ A B : FociPoint, 
    let ellipse := \frac{x^2}{9} + y^2 = 1 in
    let AB_slope := ∃ k : ℝ in 
    let AB_line := (y = k * (x + 2 * Real.sqrt 2)) in
    let TA := (T.x - A.x) * (T.y - A.y) in
    let TB := (T.x - B.x) * (T.y - B.y) in
    TA * TB = -7 / 81) :=
sorry

end problem1_problem2_l66_66251


namespace number_of_triangles_is_32_l66_66197

-- Define the vertices and new midpoints of the square and its diagonals
variables (A B C D M N P Q X Y : Type)

-- Conditions for the geometric figure
variables [is_square A B C D]
variables [is_diagonal A C]
variables [is_diagonal B D]
variables [is_midpoint M A B]
variables [is_midpoint N B C]
variables [is_midpoint P C D]
variables [is_midpoint Q D A]
variables [is_midpoint X A C]
variables [is_midpoint Y B D]
variables [is_inner_x_shape A B C D M N P Q X Y]

-- State the theorem
theorem number_of_triangles_is_32 : number_of_triangles A B C D M N P Q X Y = 32 :=
by sorry

end number_of_triangles_is_32_l66_66197


namespace cost_to_marked_price_ratio_l66_66160

-- Declare the variables and constants
variable (p : ℝ)

-- Conditions
def selling_price := p * (3 / 4)
def cost_price := selling_price * (4 / 5)

-- The theorem to prove
theorem cost_to_marked_price_ratio (p : ℝ) (hp : p ≠ 0) :
  (cost_price / p) = (3 / 5) :=
by sorry

end cost_to_marked_price_ratio_l66_66160


namespace polar_coordinates_l66_66199

def point_rectangular := (2 * Real.sqrt 2, -2 * Real.sqrt 2)

noncomputable def r := Real.sqrt ((2 * Real.sqrt 2) ^ 2 + (-2 * Real.sqrt 2) ^ 2)

noncomputable def theta := 2 * Real.pi - Real.arctan (2 * Real.sqrt 2 / 2 * Real.sqrt 2)

theorem polar_coordinates : 
  let p := point_rectangular in
  r > 0 ∧ 0 ≤ theta < 2 * Real.pi ∧ (r, theta) = (4, 7 * Real.pi / 4) := 
by
  sorry

end polar_coordinates_l66_66199


namespace Adam_picks_apples_days_l66_66485

theorem Adam_picks_apples_days (total_apples remaining_apples daily_pick : ℕ) 
  (h1 : total_apples = 350) 
  (h2 : remaining_apples = 230) 
  (h3 : daily_pick = 4) : 
  (total_apples - remaining_apples) / daily_pick = 30 :=
by {
  sorry
}

end Adam_picks_apples_days_l66_66485


namespace cylinder_volume_tripled_and_radius_increased_l66_66988

theorem cylinder_volume_tripled_and_radius_increased :
  ∀ (r h : ℝ), let V := π * r^2 * h in
               let V_new := π * (2.5 * r)^2 * (3 * h) in
               V_new = 18.75 * V :=
by
  intros r h
  let V := π * r^2 * h
  let V_new := π * (2.5 * r)^2 * (3 * h)
  sorry

end cylinder_volume_tripled_and_radius_increased_l66_66988


namespace kiwis_to_add_for_25_percent_oranges_l66_66391

theorem kiwis_to_add_for_25_percent_oranges :
  let oranges := 24
  let kiwis := 30
  let apples := 15
  let bananas := 20
  let total_fruits := oranges + kiwis + apples + bananas
  let target_total_fruits := (oranges : ℝ) / 0.25
  let fruits_to_add := target_total_fruits - (total_fruits : ℝ)
  fruits_to_add = 7 := by
  sorry

end kiwis_to_add_for_25_percent_oranges_l66_66391


namespace max_value_ineq_l66_66566

variables {R : Type} [LinearOrderedField R]

theorem max_value_ineq (a b c x y z : R) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : 0 ≤ x) (h5 : 0 ≤ y) (h6 : 0 ≤ z)
  (h7 : a + b + c = 1) (h8 : x + y + z = 1) :
  (a - x^2) * (b - y^2) * (c - z^2) ≤ 1 / 16 :=
sorry

end max_value_ineq_l66_66566


namespace theater_tickets_l66_66481

theorem theater_tickets (O B P : ℕ) (h1 : O + B + P = 550) 
  (h2 : 15 * O + 10 * B + 25 * P = 9750) (h3: P = 5 * O) (h4 : O ≥ 50) : 
  B - O = 179 :=
by
  sorry

end theater_tickets_l66_66481


namespace jim_gave_away_cards_l66_66668

theorem jim_gave_away_cards
  (sets_brother : ℕ := 15)
  (sets_sister : ℕ := 8)
  (sets_friend : ℕ := 4)
  (sets_cousin : ℕ := 6)
  (sets_classmate : ℕ := 3)
  (cards_per_set : ℕ := 25) :
  (sets_brother + sets_sister + sets_friend + sets_cousin + sets_classmate) * cards_per_set = 900 :=
by
  sorry

end jim_gave_away_cards_l66_66668


namespace no_solution_condition_l66_66886

theorem no_solution_condition (b : ℝ) : (∀ x : ℝ, 4 * (3 * x - b) ≠ 3 * (4 * x + 16)) ↔ b = -12 := 
by
  sorry

end no_solution_condition_l66_66886


namespace frog_horizontal_edge_probability_l66_66459

-- Definitions based on the conditions provided
structure Point :=
  (x : ℕ)
  (y : ℕ)

def frog_grid_bounds := { (0,0), (0,5), (7,5), (7,0) }
def start_point : Point := {x := 2, y := 3 }
def jump_length : ℕ := 2
def directions := { "up", "down", "left", "right" }

noncomputable def probability (p : Point) : ℚ :=
if (p.y = 0 ∨ p.y = 5) then 1
else if (p.x = 0 ∨ p.x = 7) then 0
else sorry -- This would require further recursive definition

-- The statement to be proved
theorem frog_horizontal_edge_probability :
  probability start_point = 3 / 4 :=
sorry

end frog_horizontal_edge_probability_l66_66459


namespace length_of_chord_EF_l66_66858

theorem length_of_chord_EF
  (A B C D G E F N O P : Type)
  (AB BC CD AG : ℝ)
  (r : ℝ)
  (H1 : AB = 2 * r)
  (H2 : BC = 2 * r)
  (H3 : CD = 2 * r)
  (H4 : r = 15)
  (H5 : ∃ B C G E F, (AB = BC ∧ BC = CD ∧ AB + BC + CD = (AD : ℝ)))
  (H6 : ∃ G, AG.is_tangent_to_circle_at r P G)
  (H7 : intersects AG N E F) :
  (EF.length = 24) :=
by sorry

end length_of_chord_EF_l66_66858


namespace sum_of_odd_divisors_of_90_l66_66085

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l66_66085


namespace find_three_digit_number_l66_66827

theorem find_three_digit_number (a b c : ℕ) (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c)
  (h_sum : 122 * a + 212 * b + 221 * c = 2003) :
  100 * a + 10 * b + c = 345 :=
by
  sorry

end find_three_digit_number_l66_66827


namespace sum_geq_cbrt3_l66_66935

variable (a b c : ℝ)

theorem sum_geq_cbrt3 (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^3 + b^3 + c^3 = a^2 * b^2 * c^2) :
  a + b + c ≥ 3 * (3 : ℝ).cbrt := 
sorry

end sum_geq_cbrt3_l66_66935


namespace pipe_A_fill_time_l66_66368

theorem pipe_A_fill_time (t : ℝ) (h1 : t > 0) (h2 : ∃ tA tB, tA = t ∧ tB = t / 6 ∧ (tA + tB) = 3) : t = 21 :=
by
  sorry

end pipe_A_fill_time_l66_66368


namespace Shekar_biology_marks_l66_66376

theorem Shekar_biology_marks 
  (math_marks : ℕ := 76) 
  (science_marks : ℕ := 65) 
  (social_studies_marks : ℕ := 82) 
  (english_marks : ℕ := 47) 
  (average_marks : ℕ := 71) 
  (num_subjects : ℕ := 5) 
  (biology_marks : ℕ) :
  (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / num_subjects = average_marks → biology_marks = 85 := 
by 
  sorry

end Shekar_biology_marks_l66_66376


namespace area_of_triangle_l66_66829

open Real

-- Define the parabola and the hyperbola equations
def parabola (x y : ℝ) := y^2 = 12 * x
def hyperbola (x y : ℝ) := x^2 / 3 - y^2 / 9 = -1

-- Define the latus rectum line of the parabola
def latus_rectum (x : ℝ) := x = -3

-- Define the asymptotes of the hyperbola
def asymptote_1 (x y : ℝ) := y = (sqrt 3 / 3) * x
def asymptote_2 (x y : ℝ) := y = -(sqrt 3 / 3) * x

-- State the theorem
theorem area_of_triangle : 
  let triangle_area := sqrt 3 * sqrt 3 * 3 in
  triangle_area = 3 * sqrt 3 :=
sorry

end area_of_triangle_l66_66829


namespace find_sum_of_coordinates_l66_66614

theorem find_sum_of_coordinates 
  (f : ℝ → ℝ)
  (h_f_def : ∀ x, f x = (2^x - 1) * (1 - 2 * sin x^2) / (2^x + 1))
  (h_line : ∀ k ≠ 0, ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1, A.2) lies in (λ x, (x, -x / k)) ∧ (B.1, B.2) lies in (λ x, (x, -x / k)) ∧ f A.1 = A.2 ∧ f B.1 = B.2)
  (C : ℝ × ℝ)
  (hC : C = (9, 3)) : 
  ∃ (D : ℝ × ℝ), let m := D.1, n := D.2 in m + n = 4 :=
sorry

end find_sum_of_coordinates_l66_66614


namespace binomial_coefficient_x4_l66_66258

def combinatorics.C (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_coefficient_x4 :
  let m := combinatorics.C 4 2 * combinatorics.C 6 2 - combinatorics.C 3 2 * combinatorics.C 5 2
  in (Nat.choose 8 2) * (m ^ 2) = 100800 := by
  sorry

end binomial_coefficient_x4_l66_66258


namespace sum_odd_divisors_of_90_l66_66106

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l66_66106


namespace volume_of_rotated_triangle_l66_66641

-- Define the lengths of the sides of the triangle
def AB : ℕ := 3
def BC : ℕ := 4
def AC : ℕ := 5

-- Define the volume of the cone formed by rotating the right triangle around AB

theorem volume_of_rotated_triangle (h_AB : AB = 3) (r_BC : BC = 4) (hypotenuse_AC : AC = 5) :
  ∀ (V : ℝ), (V = (1 / 3 : ℝ) * real.pi * (BC : ℝ)^2 * (AB : ℝ)) -> V = 16 * real.pi :=
by {
  sorry
}

end volume_of_rotated_triangle_l66_66641


namespace symmetric_point_proof_l66_66540

def point := ℝ × ℝ × ℝ

def plane (yCoeff : ℝ) (zCoeff : ℝ) (constant : ℝ) (p : point) : Prop :=
  let ⟨_, y, z⟩ := p
  yCoeff * y + zCoeff * z - constant = 0

def symmetric_point (M M' : point) (plane_eqn : point → Prop) : Prop :=
  ∃ M0 : point, 
    plane_eqn M0 ∧ 
    (let ⟨xM, yM, zM⟩ := M;
         ⟨xM0, yM0, zM0⟩ := M0;
         ⟨xM', yM', zM'⟩ := M' in
       xM0 = (xM + xM') / 2 ∧
       yM0 = (yM + yM') / 2 ∧
       zM0 = (zM + zM') / 2)

def M := (1, 0, -1) : point
def plane_eqn (p : point) : Prop := plane 2 4 1 p

theorem symmetric_point_proof : symmetric_point M (1, 1, 1) plane_eqn :=
sorry

end symmetric_point_proof_l66_66540


namespace max_value_ab_bc_cd_l66_66677

theorem max_value_ab_bc_cd (a b c d : ℝ) (h1 : 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) (h4: 0 ≤ d) (h_sum : a + b + c + d = 200) :
  ab + bc + cd ≤ 2500 :=
by
  sorry

end max_value_ab_bc_cd_l66_66677


namespace traffic_flow_max_l66_66617

noncomputable def v (x : ℝ) : ℝ :=
  if x ≤ 20 then 60
  else if x ≤ 140 then -0.5 * x + 70
  else 0

def f (x : ℝ) : ℝ :=
  x * v x

theorem traffic_flow_max :
  (∀ x, 0 ≤ x ∧ x ≤ 140 → 
       v x = if x ≤ 20 then 60 else if x ≤ 140 then -0.5 * x + 70 else 0) ∧
  (∃ x, 20 < x ∧ x < 140 ∧ f x = 2450) :=
begin
  split,
  { intro x,
    intro h,
    split_ifs with h₁ h₂,
    { exact sorry },
    { exact sorry },
    { exact sorry } },
  { use 70,
    split,
    { linarith },
    { split,
      { linarith },
      { exact eq.symm (calc 
          f 70 = 70 * v 70 : by rfl
          ...   = 70 * (-0.5 * 70 + 70) : by sorry
          ...   = 2450 : by sorry) } }
end

end traffic_flow_max_l66_66617


namespace fifth_equation_pattern_l66_66362

theorem fifth_equation_pattern :
  (1 = 1) →
  (2 + 3 + 4 = 9) →
  (3 + 4 + 5 + 6 + 7 = 25) →
  (4 + 5 + 6 + 7 + 8 + 9 + 10 = 49) →
  (5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81) :=
by 
  intros h1 h2 h3 h4
  sorry

end fifth_equation_pattern_l66_66362


namespace georgia_makes_muffins_l66_66176

/--
Georgia makes muffins and brings them to her students on the first day of every month.
Her muffin recipe only makes 6 muffins and she has 24 students. 
Prove that Georgia makes 36 batches of muffins in 9 months.
-/
theorem georgia_makes_muffins 
  (muffins_per_batch : ℕ)
  (students : ℕ)
  (months : ℕ) 
  (batches_per_day : ℕ) 
  (total_batches : ℕ)
  (h1 : muffins_per_batch = 6)
  (h2 : students = 24)
  (h3 : months = 9)
  (h4 : batches_per_day = students / muffins_per_batch) : 
  total_batches = months * batches_per_day :=
by
  -- The proof would go here
  sorry

end georgia_makes_muffins_l66_66176


namespace log_expression_value_l66_66873

open Real

theorem log_expression_value :
  log 3 5 + log 5 (1 / 3) + log 7 ((49 : ℝ)^(1 / 3)) + (1 / log 2 6) + log 5 3 + log 6 3 - log 3 15 = 2 / 3 :=
by
  sorry

end log_expression_value_l66_66873
