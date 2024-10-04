import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Prime
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Trigonometry.Trigonometric.Basic
import Mathlib.Combinatorics.GraphTheory
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.ProbabilityTheory.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Algebra.Order
import Mathlib.Meta.Prelude
import Mathlib.MetricSpace.Basic
import Mathlib.NumberTheory.Coprime.Basic
import Mathlib.Set
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace salary_single_shot_decrease_l70_70419

theorem salary_single_shot_decrease :
  let overall_factor := 0.92 * 0.86 * 0.82 in
  let percentage_decrease := 100 - (overall_factor * 100) in
  percentage_decrease ≈ 35.1728 :=
by
  let overall_factor := 0.92 * 0.86 * 0.82
  let percentage_decrease := 100 - (overall_factor * 100)
  show percentage_decrease ≈ 35.1728
  sorry

end salary_single_shot_decrease_l70_70419


namespace find_number_l70_70546

theorem find_number (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by
  sorry

end find_number_l70_70546


namespace solve_quadratic_l70_70909

theorem solve_quadratic : ∀ x : ℝ, x ^ 2 - 6 * x + 8 = 0 ↔ x = 2 ∨ x = 4 := by
  sorry

end solve_quadratic_l70_70909


namespace equi_partite_complex_number_a_l70_70018

-- A complex number z = 1 + (a-1)i
def z (a : ℝ) : ℂ := ⟨1, a - 1⟩

-- Definition of an equi-partite complex number
def is_equi_partite (z : ℂ) : Prop := z.re = z.im

-- The theorem to prove
theorem equi_partite_complex_number_a (a : ℝ) : is_equi_partite (z a) ↔ a = 2 := 
by
  sorry

end equi_partite_complex_number_a_l70_70018


namespace relation_between_a_and_b_l70_70064

open EuclideanGeometry

-- Definitions and given conditions:

-- Plane α is perpendicular to plane β
variable (α β : Plane)
variable (l : Line) (h₁ : α ⊥ β)

-- Their intersection is line l
variable (h_inter : intersect α β = l)

-- Line a is contained in α
variable (a : Line) (h₂ : is_in_line a α)

-- Line b is contained in β
variable (b : Line) (h₃ : is_in_line b β)

-- Neither a is perpendicular to l nor b is perpendicular to l
variable (h₄ : ¬ (a ⊥ l))
variable (h₅ : ¬ (b ⊥ l))

-- The relationship between a and b:
theorem relation_between_a_and_b : (∃ l' : Line, a ∥ l' ∧ b ∥ l') ∧ ¬ (a ⊥ b) := sorry

end relation_between_a_and_b_l70_70064


namespace proof_l70_70088

-- Define proposition p
def p : Prop := ∀ x : ℝ, x < 0 → 2^x > x

-- Define proposition q
def q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

theorem proof : p ∨ q :=
by
  have hp : p := 
    -- Here, you would provide the proof of p being true.
    sorry
  have hq : ¬ q :=
    -- Here, you would provide the proof of q being false, 
    -- i.e., showing that ∀ x, x^2 + x + 1 ≥ 0.
    sorry
  exact Or.inl hp

end proof_l70_70088


namespace solve_estate_problem_l70_70069

noncomputable def estate_total (E : ℝ) :=
  let daughters_share := 0.4 * E in
  let husband_share := 1.2 * E in
  let gardener_share := 1000 in
  E = daughters_share + husband_share + gardener_share → E = 2500

theorem solve_estate_problem : ∃ (E : ℝ), estate_total E :=
by 
  have example_estate : 2500 := 2500
  use example_estate
  rw estate_total
  sorry

end solve_estate_problem_l70_70069


namespace monotonic_increase_interval_l70_70014

variable {b : ℝ}

def f (x : ℝ) : ℝ := x + b / x

theorem monotonic_increase_interval (h : ∃ x ∈ Ioo 1 2, deriv f x = 0) : 
  Ioo (-∞ : ℝ) (-2) ⊆ { x | deriv f x > 0 } := 
sorry

end monotonic_increase_interval_l70_70014


namespace conditions_imply_perpendicularity_l70_70833

variable (a b : Line)
variable (α β : Plane)

theorem conditions_imply_perpendicularity:
  (∃ (h₁ : a ⊂ α) (h₂ : b ∥ β) (h₃ : α ⊥ β), a ⊥ b) ∨
  (∃ (h₁ : a ⊥ α) (h₂ : b ⊥ β) (h₃ : α ⊥ β), a ⊥ b) ∨
  (∃ (h₁ : a ⊂ α) (h₂ : b ⊥ β) (h₃ : α ∥ β), a ⊥ b) ∨
  (∃ (h₁ : a ⊥ α) (h₂ : b ∥ β) (h₃ : α ∥ β), a ⊥ b) :=
sorry

end conditions_imply_perpendicularity_l70_70833


namespace probability_divisible_by_15_l70_70015

open Finset

theorem probability_divisible_by_15 (digits : Finset ℕ) (h : digits = {1, 2, 3, 5, 0}) :
  (∀ (n : ℕ), (n ∈ permutations digits.toList) → (n % 15 ≠ 0)) :=
by
  intros n h_perms
  have h_sum : digits.sum = 11 := by
    rw [h, sum_insert, sum_insert, sum_insert, sum_insert, sum_singleton, add_zero]
    norm_num
  have not_divisible_by_3 : (digits.sum % 3 ≠ 0) := by
    rw [h_sum]
    norm_num
  sorry

end probability_divisible_by_15_l70_70015


namespace least_possible_product_of_two_distinct_primes_greater_than_50_l70_70147

open nat

theorem least_possible_product_of_two_distinct_primes_greater_than_50 :
  ∃ p q : ℕ, p ≠ q ∧ prime p ∧ prime q ∧ p > 50 ∧ q > 50 ∧ 
  (∀ p' q' : ℕ, p' ≠ q' → prime p' → prime q' → p' > 50 → q' > 50 → p * q ≤ p' * q') ∧ p * q = 3127 :=
by
  sorry

end least_possible_product_of_two_distinct_primes_greater_than_50_l70_70147


namespace segment_intersection_l70_70882

theorem segment_intersection {k : ℕ} (k_pos : k > 0) :
  ∃ w b : Finset (ℕ × ℕ),
    (∀ i, i < 2 * k - 1 → (∃ j, j < 2 * k - 1 ∧ intersects w i j))
    ∧ (∃ j, j < 2 * k - 1 ∧ (∀ i, w i j)) := 
sorry

end segment_intersection_l70_70882


namespace solve_quadratic_l70_70913

theorem solve_quadratic :
    ∀ x : ℝ, x^2 - 6*x + 8 = 0 ↔ (x = 2 ∨ x = 4) :=
by
  intros x
  sorry

end solve_quadratic_l70_70913


namespace phi_range_l70_70330

noncomputable def f (omega phi x : ℝ) : ℝ := 2 * sin (omega * x + phi)

theorem phi_range (omega : ℝ) (phi : ℝ) (h_omega : omega > 0) (h_phi : abs phi < (π / 2))
  (h_symmetry : 2 * π / omega = 2 * π)
  (h_f_gt_1 : ∀ x, -π / 6 < x ∧ x < π → f omega phi x > 1) :
  π / 4 < phi ∧ phi < π / 3 :=
by
  have h_omega_half : omega = 1 / 2, from sorry,
  have h_phi_bounds : π / 4 < phi ∧ phi < π / 3, from sorry,
  exact h_phi_bounds

end phi_range_l70_70330


namespace three_friends_at_least_50_mushrooms_l70_70470

theorem three_friends_at_least_50_mushrooms (a : Fin 7 → ℕ) (h_sum : (Finset.univ.sum a) = 100) (h_different : Function.Injective a) :
  ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (a i + a j + a k) ≥ 50 :=
by
  sorry

end three_friends_at_least_50_mushrooms_l70_70470


namespace smallest_class_size_is_42_l70_70394

noncomputable def smallest_class_size_exceeding_40 : Nat :=
  let n := 10
  let t := 4 * n + 2
  t

theorem smallest_class_size_is_42 :
  ∃ t : Nat, t > 40 ∧ (∀ n : Nat, (4 * n + 2 > 40 → n >= 10) → t = 4 * 10 + 2) := by
  use 42
  split
  · exact Nat.lt.base 41 -- Prove 42 > 40
  · intro n H
    have hn : 4 * n + 2 > 40 := by
      exact H n
    have Hn : n >= 10 := by
      linarith
    have t_eq_42 : t = 42 := by
      exact Eq.refl 42
    exact t_eq_42

end smallest_class_size_is_42_l70_70394


namespace union_M_N_l70_70341

-- Define the set M
def M : Set ℤ := {x | x^2 - x = 0}

-- Define the set N
def N : Set ℤ := {y | y^2 + y = 0}

-- Prove that the union of M and N is {-1, 0, 1}
theorem union_M_N :
  M ∪ N = {-1, 0, 1} :=
by
  sorry

end union_M_N_l70_70341


namespace find_x_l70_70378

theorem find_x (x y : ℝ) (h1 : y = 1) (h2 : 4 * x - 2 * y + 3 = 3 * x + 3 * y) : x = 2 :=
by
  sorry

end find_x_l70_70378


namespace slices_left_for_Era_l70_70265

def total_burgers : ℕ := 5
def slices_per_burger : ℕ := 8

def first_friend_slices : ℕ := 3
def second_friend_slices : ℕ := 8
def third_friend_slices : ℕ := 5
def fourth_friend_slices : ℕ := 11
def fifth_friend_slices : ℕ := 6

def total_slices : ℕ := total_burgers * slices_per_burger
def slices_given_to_friends : ℕ := first_friend_slices + second_friend_slices + third_friend_slices + fourth_friend_slices + fifth_friend_slices

theorem slices_left_for_Era : total_slices - slices_given_to_friends = 7 :=
by
  rw [total_slices, slices_given_to_friends]
  exact Eq.refl 7

#reduce slices_left_for_Era

end slices_left_for_Era_l70_70265


namespace fraction_meaningful_iff_l70_70780

theorem fraction_meaningful_iff (x : ℝ) : (x ≠ 2) ↔ (x - 2 ≠ 0) := 
by
  sorry

end fraction_meaningful_iff_l70_70780


namespace tan_theta_expression_l70_70440

theorem tan_theta_expression (θ : ℝ) (x : ℝ) (h_acute : 0 < θ ∧ θ < π / 2) 
  (h_sin_half : sin (θ / 2) = sqrt ((x + 1) / (2 * x))) : 
  tan θ = -sqrt (x^2 - 1) :=
by
  sorry

end tan_theta_expression_l70_70440


namespace negation_of_odd_function_proposition_l70_70942

variables {f : ℝ → ℝ}

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem negation_of_odd_function_proposition :
  (¬ (∀ x, is_odd f → is_odd (λ x, f (-x)))) ↔ (∀ x, ¬is_odd f → ¬is_odd (λ x, f (-x))) :=
by
  sorry

end negation_of_odd_function_proposition_l70_70942


namespace lisa_needs_additional_marbles_l70_70868

/-- Lisa has 12 friends and 40 marbles. She needs to ensure each friend gets at least one marble and no two friends receive the same number of marbles. We need to find the minimum number of additional marbles needed to ensure this. -/
theorem lisa_needs_additional_marbles : 
  ∀ (friends marbles : ℕ), friends = 12 → marbles = 40 → 
  ∃ (additional_marbles : ℕ), additional_marbles = 38 ∧ 
  (∑ i in finset.range (friends + 1), i) - marbles = additional_marbles :=
by
  intros friends marbles friends_eq marbles_eq 
  use 38
  split
  · exact rfl
  calc (∑ i in finset.range (12 + 1), i) - 40 = 78 - 40 : by norm_num
                                  ... = 38 : by norm_num

end lisa_needs_additional_marbles_l70_70868


namespace mark_first_vaccine_wait_time_l70_70446

-- Define the variables and conditions
variable (x : ℕ)
variable (total_wait_time : ℕ)
variable (second_appointment_wait : ℕ)
variable (effectiveness_wait : ℕ)

-- Given conditions
axiom h1 : second_appointment_wait = 20
axiom h2 : effectiveness_wait = 14
axiom h3 : total_wait_time = 38

-- The statement to be proven
theorem mark_first_vaccine_wait_time
  (h4 : x + second_appointment_wait + effectiveness_wait = total_wait_time) :
  x = 4 := by
  sorry

end mark_first_vaccine_wait_time_l70_70446


namespace angle_PMN_l70_70807

theorem angle_PMN (P R Q M N : Point) 
  (h1 : ∆ P R Q)
  (h2 : PR = RQ)
  (h3 : PM = MR)
  (h4 : RN = NQ)
  (h5 : angle PQR = 64)
  (isosceles_PMR : isosceles ∆ P M R)
  (isosceles_RNQ : isosceles ∆ R N Q):
  measure (angle PMN) = 116 :=
by
  sorry

end angle_PMN_l70_70807


namespace m_add_n_eq_l70_70051

-- Conditions
variables {V : Type*} [inner_product_space ℝ V] (a b : V)
variables [fact (orthogonal a b)]
variables (m n : ℝ)
def OA : V := a
def OB : V := b
def OC : V := m • a + n • b
def right_angle_A : Prop := (∥OB - OA∥ = ∥OC - OA∥) ∧ ⟪OB - OA, OC - OA⟫ = 0

-- Proof that m + n = 3 or m + n = -1
theorem m_add_n_eq (h : right_angle_A a b m n) : m + n = 3 ∨ m + n = -1 :=
sorry

end m_add_n_eq_l70_70051


namespace inequality_satisfied_for_all_l70_70830

noncomputable def sequence_a : ℕ → ℝ := sorry -- placeholder for the sequence

axiom sequence_positive (n : ℕ) : sequence_a n > 0

axiom sum_geq_sqrt (n : ℕ) (h : n ≥ 1) : (∑ j in finset.range n, sequence_a (j + 1)) ≥ real.sqrt (n : ℝ)

theorem inequality_satisfied_for_all (n : ℕ) (h : n ≥ 1) :
  (∑ j in finset.range n, (sequence_a (j + 1))^2) > 0.25 * (∑ j in finset.range n, 1 / (j + 1 : ℝ)) :=
  sorry

end inequality_satisfied_for_all_l70_70830


namespace erika_rick_money_left_l70_70683

theorem erika_rick_money_left (gift_cost cake_cost erika_savings : ℝ)
  (rick_savings := gift_cost / 2) (total_savings := erika_savings + rick_savings) 
  (total_cost := gift_cost + cake_cost) : (total_savings - total_cost) = 5 :=
by
  -- Given conditions from the problem
  have h_gift_cost : gift_cost = 250 := sorry
  have h_cake_cost : cake_cost = 25 := sorry
  have h_erika_savings : erika_savings = 155 := sorry
  -- Show that the remaining money is $5
  have h_rick_savings : rick_savings = 125 := by
    rw [←h_gift_cost]
    norm_num
  have h_total_savings : total_savings = 280 := by
    rw [←h_erika_savings, ←h_rick_savings]
    norm_num
  have h_total_cost : total_cost = 275 := by
    rw [←h_gift_cost, ←h_cake_cost]
    norm_num
  rw [←h_total_savings, ←h_total_cost]
  norm_num
  done

end erika_rick_money_left_l70_70683


namespace non_zero_payment_cases_l70_70361

/-!
# Proof that the total number of non-zero payment cases is 89
-/

/-- You have the following coins:
  - 4 coins of 500-won
  - 2 coins of 100-won
  - 5 coins of 10-won

We have to prove that the total number of non-zero payment cases is 89.
-/
theorem non_zero_payment_cases :
  let options_500_won := 5,
      options_100_won := 3,
      options_10_won := 6 in
  (options_500_won * options_100_won * options_10_won) - 1 = 89 :=
by
  let options_500_won := 5
  let options_100_won := 3
  let options_10_won := 6
  show (options_500_won * options_100_won * options_10_won) - 1 = 89
  calc
    5 * 3 * 6 - 1 = 90 - 1 := by rfl
    ... = 89 := by rfl

end non_zero_payment_cases_l70_70361


namespace cuberoot_of_5488000_eq_simplified_l70_70475

noncomputable def cuberoot_simplified_form : ℝ :=
  140 * 2^(1/3 : ℝ)

theorem cuberoot_of_5488000_eq_simplified :
  ∛(5488000 : ℝ) = cuberoot_simplified_form :=
sorry

end cuberoot_of_5488000_eq_simplified_l70_70475


namespace positive_difference_l70_70061

def f (n : ℝ) : ℝ :=
  if n < 3 then n^2 + 4 else 2*n - 30

theorem positive_difference : 
  abs (20.5 - real.sqrt 7) = 20.5 - real.sqrt 7 :=
by
  have h_f_neg3 : f (-3) = 13 := by
    -- proof that f(-3) = 13
    sorry
  have h_f_3 : f 3 = -24 := by
    -- proof that f(3) = -24
    sorry
  have h_f_x_eq_11 : ∀ x, f x = 11 ↔ (x = real.sqrt 7 ∨ x = -real.sqrt 7 ∨ x = 20.5) :=
    -- proof that f(x) = 11 has solutions x = sqrt(7), x = -sqrt(7), and x = 20.5
    sorry
  have h_x_values : f (-3) + f 3 + f x = 0 ↔ f x = 11 := by
    -- proof that f(-3) + f(3) + f(x) = 0 implies f(x) = 11
    sorry
  have h_positive_diff : abs (20.5 - real.sqrt 7) = 20.5 - real.sqrt 7 :=
    -- proof of positive difference between 20.5 and sqrt(7)
    sorry
  exact h_positive_diff

end positive_difference_l70_70061


namespace cylinder_surface_area_minimization_l70_70592

theorem cylinder_surface_area_minimization (S V r h : ℝ) (h₁ : π * r^2 * h = V) (h₂ : r^2 + (h / 2)^2 = S^2) : (h / r) = 2 :=
sorry

end cylinder_surface_area_minimization_l70_70592


namespace total_cards_is_56_l70_70897

-- Let n be the number of Pokemon cards each person has
def n : Nat := 14

-- Let k be the number of people
def k : Nat := 4

-- Total number of Pokemon cards
def total_cards : Nat := n * k

-- Prove that the total number of Pokemon cards is 56
theorem total_cards_is_56 : total_cards = 56 := by
  sorry

end total_cards_is_56_l70_70897


namespace top_angle_is_70_l70_70981

theorem top_angle_is_70
  (sum_angles : ℝ)
  (left_angle : ℝ)
  (right_angle : ℝ)
  (top_angle : ℝ)
  (h1 : sum_angles = 250)
  (h2 : left_angle = 2 * right_angle)
  (h3 : right_angle = 60)
  (h4 : sum_angles = left_angle + right_angle + top_angle) :
  top_angle = 70 :=
by
  sorry

end top_angle_is_70_l70_70981


namespace units_digit_of_p_is_6_l70_70323

theorem units_digit_of_p_is_6 (p : ℕ) (h_even : Even p) (h_units_p_plus_1 : (p + 1) % 10 = 7) (h_units_p3_minus_p2 : ((p^3) % 10 - (p^2) % 10) % 10 = 0) : p % 10 = 6 := 
by 
  -- proof steps go here
  sorry

end units_digit_of_p_is_6_l70_70323


namespace polynomial_condition_l70_70700

noncomputable def q (x : ℝ) : ℝ := -2 * x + 4

theorem polynomial_condition (x : ℝ) :
  q(q(x)) = x * q(x) + 2 * x ^ 2 :=
by
  sorry

end polynomial_condition_l70_70700


namespace car_round_trip_time_l70_70109

theorem car_round_trip_time
  (d_AB : ℝ) (v_AB_downhill : ℝ) (v_BA_uphill : ℝ)
  (h_d_AB : d_AB = 75.6)
  (h_v_AB_downhill : v_AB_downhill = 33.6)
  (h_v_BA_uphill : v_BA_uphill = 25.2) :
  d_AB / v_AB_downhill + d_AB / v_BA_uphill = 5.25 := by
  sorry

end car_round_trip_time_l70_70109


namespace checkered_rectangle_minimal_area_checkered_rectangle_possible_perimeters_l70_70588

-- Define the conditions
def is_checkered_rectangle (S : ℕ) : Prop :=
  (∃ (a b : ℕ), a * b = S) ∧
  (∀ x y k l : ℕ, x * 13 + y * 1 = S) ∧
  (S % 39 = 0)

-- Define that S is minimal satisfying the conditions
def minimal_area_checkered_rectangle (S : ℕ) : Prop :=
  is_checkered_rectangle S ∧
  (∀ (S' : ℕ), S' < S → ¬ is_checkered_rectangle S')

-- Prove that S = 78 is the minimal area
theorem checkered_rectangle_minimal_area : minimal_area_checkered_rectangle 78 :=
  sorry

-- Define the condition for possible perimeters
def possible_perimeters (S : ℕ) (p : ℕ) : Prop :=
  (∀ (a b : ℕ), a * b = S → 2 * (a + b) = p)

-- Prove the possible perimeters for area 78
theorem checkered_rectangle_possible_perimeters :
  ∀ p, p = 38 ∨ p = 58 ∨ p = 82 ↔ possible_perimeters 78 p :=
  sorry

end checkered_rectangle_minimal_area_checkered_rectangle_possible_perimeters_l70_70588


namespace decagon_area_l70_70593

theorem decagon_area (perimeter : ℝ) (n : ℕ) (side_length : ℝ)
  (segments : ℕ) (area : ℝ) :
  perimeter = 200 ∧ n = 4 ∧ side_length = perimeter / n ∧ segments = 5 ∧ 
  area = (side_length / segments)^2 * (1 - (1/2)) * 4 * segments  →
  area = 2300 := 
by
  sorry

end decagon_area_l70_70593


namespace find_f2019_l70_70847

def f (x : ℝ) : ℝ := x^2 / (2 * x - 1)

def f_iter : ℕ → (ℝ → ℝ)
| 0     := id
| (n+1) := f ∘ (f_iter n)

theorem find_f2019 (x : ℝ) :
  f_iter 2019 x = x ^ (2^2019) / (x ^ (2^2019) - (x - 1) ^ (2^2019)) :=
sorry

end find_f2019_l70_70847


namespace integer_solutions_for_xyz_l70_70658

theorem integer_solutions_for_xyz (x y z : ℤ) : 
  (x - y - 1)^3 + (y - z - 2)^3 + (z - x + 3)^3 = 18 ↔
  (x = y ∧ y = z) ∨
  (x = y - 1 ∧ y = z) ∨
  (x = y ∧ y = z + 5) ∨
  (x = y + 4 ∧ y = z + 5) ∨
  (x = y + 4 ∧ z = y) ∨
  (x = y - 1 ∧ z = y + 4) :=
by {
  sorry
}

end integer_solutions_for_xyz_l70_70658


namespace prime_solution_unique_l70_70667

theorem prime_solution_unique:
  ∃ p q r : ℕ, (prime p ∧ prime q ∧ prime r ∧ p + q^2 = r^4 ∧ p = 7 ∧ q = 3 ∧ r = 2) ∧
  (∀ p' q' r' : ℕ, prime p' ∧ prime q' ∧ prime r' ∧ p' + q'^2 = r'^4 → (p' = 7 ∧ q' = 3 ∧ r' = 2)) :=
by {
  sorry
}

end prime_solution_unique_l70_70667


namespace least_possible_product_of_two_distinct_primes_greater_than_50_l70_70144

open nat

theorem least_possible_product_of_two_distinct_primes_greater_than_50 :
  ∃ p q : ℕ, p ≠ q ∧ prime p ∧ prime q ∧ p > 50 ∧ q > 50 ∧ 
  (∀ p' q' : ℕ, p' ≠ q' → prime p' → prime q' → p' > 50 → q' > 50 → p * q ≤ p' * q') ∧ p * q = 3127 :=
by
  sorry

end least_possible_product_of_two_distinct_primes_greater_than_50_l70_70144


namespace square_side_measurement_error_l70_70620

theorem square_side_measurement_error (S S' : ℝ) (h1 : S' = S * Real.sqrt 1.0404) : 
  (S' - S) / S * 100 = 2 :=
by
  sorry

end square_side_measurement_error_l70_70620


namespace geometric_series_sum_l70_70238

noncomputable def geometric_sum (a r : ℚ) (h : |r| < 1) : ℚ :=
a / (1 - r)

theorem geometric_series_sum :
  geometric_sum 1 (1/3) (by norm_num) = 3/2 :=
by
  sorry

end geometric_series_sum_l70_70238


namespace musin_problem_l70_70472

theorem musin_problem (a : ℕ → ℤ) (m n : ℕ) 
  (a_nonzero : ∀ i : ℕ, 1 ≤ i ∧ i ≤ m → a i ≠ 0)
  (h : ∀ k : ℕ, k ≤ n → (finset.range m).sum (λ i, a (i + 1) * (i + 1)^k) = 0)
  (hn : n < m - 1) :
  ∃ S : finset ℕ, S.card ≥ n + 1 ∧ (∀ i : ℕ, i ∈ S → i + 1 ∈ S) ∧ (∀ i : ℕ, i ∈ S → (a i) * (a (i + 1)) < 0) := 
sorry

end musin_problem_l70_70472


namespace num_employees_is_143_l70_70946

def b := 143
def is_sol (b : ℕ) := 80 < b ∧ b < 150 ∧ b % 4 = 3 ∧ b % 5 = 3 ∧ b % 7 = 4

theorem num_employees_is_143 : is_sol b :=
by
  -- This is where the proof would be written
  sorry

end num_employees_is_143_l70_70946


namespace trig_intersection_problem_l70_70111

theorem trig_intersection_problem
  (f : ℝ → ℝ) (k α : ℝ)
  (h_f : f = λ x, abs (sin x))
  (h1 : ∀ x, x ≥ 0 → ∃ x1 x2 x3, x1 < x2 ∧ x2 < x3 ∧ (f x1 = k * x1 ∧ f x2 = k * x2 ∧ f x3 = k * x3))
  (h2 : ∀ x, x ∈ (pi, (3 * pi / 2)) → (f x = k * x → x = α))
  (hα_range : α ∈ (pi, (3 * pi / 2))) :
    (cos α / (sin α + sin (3 * α)) = (α^2 + 1) / (4 * α)) :=
begin
  sorry
end

end trig_intersection_problem_l70_70111


namespace percent_same_grade_l70_70985

theorem percent_same_grade (total_students same_A same_B same_C same_D : ℕ) 
  (h_total : total_students = 40)
  (h_same_A : same_A = 3)
  (h_same_B : same_B = 5)
  (h_same_C : same_C = 7)
  (h_same_D : same_D = 2) :
  (same_A + same_B + same_C + same_D) * 100 / total_students = 42.5 :=
by
  sorry

end percent_same_grade_l70_70985


namespace geometric_progression_properties_l70_70062

noncomputable def gp_sum (a r n : ℕ) : ℕ := a * (1 - r^n) / (1 - r)
noncomputable def last_term (a r n : ℕ) : ℕ := a * r^(n-1)

theorem geometric_progression_properties :
  let a := 3; let r := 2; let n := 5 in
  gp_sum a r n = 93 ∧ last_term a r n = 48 :=
by 
  sorry

end geometric_progression_properties_l70_70062


namespace final_value_l70_70740

noncomputable def f : ℕ → ℝ := sorry

axiom f_mul_add (a b : ℕ) : f (a + b) = f a * f b
axiom f_one : f 1 = 2

theorem final_value : 
  (f 1)^2 + f 2 / f 1 + (f 2)^2 + f 4 / f 3 + (f 3)^2 + f 6 / f 5 + (f 4)^2 + f 8 / f 7 = 16 := 
sorry

end final_value_l70_70740


namespace max_value_sqrt_sum_l70_70054

-- Given conditions
variables (x y z : ℝ)
variable h1 : x ≥ 0
variable h2 : y ≥ 0
variable h3 : z ≥ 0
variable h4 : x + y + z = 6

-- Main statement
theorem max_value_sqrt_sum :
  \(\sqrt{3 * x + 1} + \sqrt{3 * y + 1} + \sqrt{3 * z + 1} \leq \sqrt{63}\) :=
sorry

end max_value_sqrt_sum_l70_70054


namespace solve_for_x_l70_70782

theorem solve_for_x (x : ℝ) (h : (3 + 2^(-x)) * (1 - 2^x) = 4) : x = Real.log (1/3) / Real.log 2 := 
by sorry

end solve_for_x_l70_70782


namespace woodcutters_division_l70_70517

theorem woodcutters_division (ivan_loaves : ℕ) (prokhor_loaves : ℕ) (total_loaves : ℕ)
  (people : ℕ) (total_kopecks : ℕ) (per_loaf_value : ℕ) :
  ivan_loaves = 4 →
  prokhor_loaves = 8 →
  total_loaves = 12 →
  people = 3 →
  total_kopecks = 60 →
  per_loaf_value = total_kopecks / total_loaves →
  let ivan_share := ivan_loaves * per_loaf_value in
  let prokhor_share := (prokhor_loaves - (total_loaves / people)) * per_loaf_value + prokhor_loaves / people * per_loaf_value in
  ivan_share = 20 ∧ prokhor_share = 40 :=
by
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end woodcutters_division_l70_70517


namespace part_I_part_II_l70_70335

-- Define the function f
def f (x a : ℝ) := Real.exp x - a * x ^ 2

-- 1. Prove that f(x) is monotonically increasing on the interval (2, +∞) if and only if \(a \leq e^2 / 4\)
theorem part_I (a : ℝ) :
  (∀ x : ℝ, 2 < x → 0 ≤ Real.exp x - 2 * a * x) ↔ a ≤ Real.exp 2 / 4 := sorry

-- 2. Prove the inequality for given conditions
theorem part_II (x1 x2 a : ℝ) (h1 : 2 < x1) (h2 : 2 < x2) (ha : a < x2) :
  (x2 - x1) * (f x1 a + a * x1 ^ 2 + f x2 a + a * x2 ^ 2) > 2 * (Real.exp x2 - Real.exp x1) := sorry

end part_I_part_II_l70_70335


namespace sum_of_22s_l70_70173

theorem sum_of_22s :
  let a := 22000000
  let b := 22000
  let c := 2200
  let d := 22
  a + b + c + d = 22024222 := 
by 
  let a := 22000000
  let b := 22000
  let c := 2200
  let d := 22
  calc 
    a + b + c + d = 22000000 + 22000 + 2200 + 22 : by rfl
    ... = 22022000 + 2200 + 22 : by rfl
    ... = 22024200 + 22 : by rfl
    ... = 22024222 : by rfl

end sum_of_22s_l70_70173


namespace no_ordered_quadruples_l70_70698

theorem no_ordered_quadruples (a b c d : ℝ) :
  (a * d - b * c ≠ 0) →
  (Matrix.inv ![![a, b], ![c, d]] = ![![1 / a^2, -1 / b^2], ![-1 / c^2, 1 / d^2]]) →
  False :=
sorry

end no_ordered_quadruples_l70_70698


namespace tens_digit_of_desired_number_is_one_l70_70842

def productOfDigits (n : Nat) : Nat :=
  match n / 10, n % 10 with
  | a, b => a * b

def sumOfDigits (n : Nat) : Nat :=
  match n / 10, n % 10 with
  | a, b => a + b

def isDesiredNumber (N : Nat) : Prop :=
  N < 100 ∧ N ≥ 10 ∧ N = (productOfDigits N)^2 + sumOfDigits N

theorem tens_digit_of_desired_number_is_one (N : Nat) (h : isDesiredNumber N) : N / 10 = 1 :=
  sorry

end tens_digit_of_desired_number_is_one_l70_70842


namespace solve_trig_equation_l70_70098

theorem solve_trig_equation (x : ℝ) : 
  2 * Real.cos (13 * x) + 3 * Real.cos (3 * x) + 3 * Real.cos (5 * x) - 8 * Real.cos x * (Real.cos (4 * x))^3 = 0 ↔ 
  ∃ (k : ℤ), x = (k * Real.pi) / 12 :=
sorry

end solve_trig_equation_l70_70098


namespace minimum_dimes_l70_70243

-- Given amounts in dollars
def value_of_dimes (n : ℕ) : ℝ := 0.10 * n
def value_of_nickels : ℝ := 0.50
def value_of_one_dollar_bill : ℝ := 1.0
def value_of_four_tens : ℝ := 40.0
def price_of_scarf : ℝ := 42.85

-- Prove the total value of the money is at least the price of the scarf implies n >= 14
theorem minimum_dimes (n : ℕ) :
  value_of_four_tens + value_of_one_dollar_bill + value_of_nickels + value_of_dimes n ≥ price_of_scarf → n ≥ 14 :=
by
  sorry

end minimum_dimes_l70_70243


namespace sqrt_sum_of_differences_l70_70423

-- Define noncomputable to handle the real number operations
noncomputable def sqrt (x : ℝ) := real.sqrt x

-- The proof statement
theorem sqrt_sum_of_differences (m n : ℕ) (hm : m > 1) (hn : n > 1) :
  ∃ (N : fin m → ℕ), real.sqrt m = (∑ i in finset.range m, (sqrt (N i) - sqrt (N i - 1)) ^ (1 / (n : ℝ))) := 
sorry

end sqrt_sum_of_differences_l70_70423


namespace evaluate_magnitude_l70_70684

def z1 : ℂ := 3 * Real.sqrt 5 - 5 * Complex.I
def z2 : ℂ := 2 * Real.sqrt 7 + 4 * Complex.I

theorem evaluate_magnitude:
  Complex.abs (z1 * z2) = 20 * Real.sqrt 77 :=
by
  sorry

end evaluate_magnitude_l70_70684


namespace problem_log_sum_squared_eq_eight_l70_70318

variable {a x₁ x₂ ... x₂₀₀₆ : ℝ}
variables (H : log a (x₁ * x₂ * ... * x₂₀₀₆) = 4)

theorem problem_log_sum_squared_eq_eight :
  log a (x₁^2) + log a (x₂^2) + ... + log a (x₂₀₀₆^2) = 8 :=
sorry

end problem_log_sum_squared_eq_eight_l70_70318


namespace fixed_point_coordinates_l70_70935

theorem fixed_point_coordinates (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, 4) ∧ ∀ x, P = (x, a^(x-1) + 3) :=
by
  use (1, 4)
  sorry

end fixed_point_coordinates_l70_70935


namespace at_least_30_distances_l70_70883

noncomputable theory

open_locale classical

theorem at_least_30_distances (points : set (ℝ × ℝ)) (h : points.finite) (h_count : points.to_finset.card = 2004) :
  ∃ dists : set ℝ, dists.finite ∧ dists.to_finset.card ≥ 30 ∧
  ∀ p1 p2 ∈ points, p1 ≠ p2 → (dist p1 p2) ∈ dists :=
begin
  sorry
end

end at_least_30_distances_l70_70883


namespace time_to_reach_ticket_window_l70_70559

-- Define the conditions as per the problem
def rate_kit : ℕ := 2 -- feet per minute (rate)
def remaining_distance : ℕ := 210 -- feet

-- Goal: To prove the time required to reach the ticket window is 105 minutes
theorem time_to_reach_ticket_window : remaining_distance / rate_kit = 105 :=
by sorry

end time_to_reach_ticket_window_l70_70559


namespace positive_difference_coordinates_l70_70138

variables (A B C R S : ℝ × ℝ)
variables (AC BC : set (ℝ × ℝ))

-- Coordinates of A, B, C
def A := (-1, 7 : ℝ × ℝ)
def B := (4, -2 : ℝ × ℝ)
def C := (9, -2 : ℝ × ℝ)

-- Condition: vertical line intersects AC at R and BC at S
def is_vertical_intersection (R S : ℝ × ℝ) (AC BC : set (ℝ × ℝ)) : Prop :=
  R.1 = S.1 ∧ S.2 = -2 ∧ S ∈ BC ∧ R ∈ AC

-- Condition: area of triangle RSC
def area_triangle (R S C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((R.1 - S.1) * (S.2 - C.2) - (R.1 - C.1) * (S.2 - S.2))

-- Problem statement
theorem positive_difference_coordinates (
  h1 : is_vertical_intersection R S {p : ℝ × ℝ | p.1 + 1 + 0.9 * p.2 = 0} -- this represents line AC
  h2 : area_triangle R S C = 18
) : abs (R.1 - R.2) = 1 :=
sorry

end positive_difference_coordinates_l70_70138


namespace complex_power_equiv_l70_70060

theorem complex_power_equiv :
  let i := Complex.I in
  ((1 + i)/(1 - i)) ^ 2013 = i :=
by
  sorry

end complex_power_equiv_l70_70060


namespace evaluate_expression_l70_70685

theorem evaluate_expression : 
  (3^2 - 3 * 2) - (4^2 - 4 * 2) + (5^2 - 5 * 2) - (6^2 - 6 * 2) = -14 :=
by
  sorry

end evaluate_expression_l70_70685


namespace math_problems_l70_70386

-- Define the conditions of triangle ABC
variables (A B C a b c : ℝ)
axiom h1 : a = 10 * Real.sqrt 3
axiom h2 : A = Real.pi / 3

-- Problem (Ⅰ)
def problem1 (b : ℝ) : Prop :=
b = 10 * Real.sqrt 2 →
  B = Real.pi / 4 ∧ C = 5 * Real.pi / 12

-- Problem (Ⅱ)
def problem2 (c : ℝ) : Prop :=
c = 10 →
  b = 20 ∧ 1 / 2 * b * c * Real.sin A = 50 * Real.sqrt 3

-- The full problem statement 
theorem math_problems :
  (problem1 b) ∧ (problem2 c) :=
  sorry

end math_problems_l70_70386


namespace marbles_exceed_200_on_sunday_l70_70857

theorem marbles_exceed_200_on_sunday:
  ∃ n : ℕ, 3 * 2^n > 200 ∧ (n % 7) = 0 :=
by
  sorry

end marbles_exceed_200_on_sunday_l70_70857


namespace negation_of_universal_proposition_l70_70496

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℝ), x^3 - x^2 + 1 ≤ 0) ↔ ∃ (x₀ : ℝ), x₀^3 - x₀^2 + 1 > 0 :=
by {
  sorry
}

end negation_of_universal_proposition_l70_70496


namespace min_factors_to_end_in_two_l70_70163

theorem min_factors_to_end_in_two (n : ℕ) (h_n : n = 99) :
  let fact_n := Nat.factorial n,
      count_factors_5 := (n / 5) + (n / 25),
      num_factors_to_remove := 22 in
  count_factors_5 = 22 →
  ∃ m : ℕ, m = 20 ∧ (product_of_remaining_factors (remove_factors fact_n m 5) ends_in 2).

end min_factors_to_end_in_two_l70_70163


namespace find_original_petrol_price_l70_70213

noncomputable def original_petrol_price (P : ℝ) : Prop :=
  let original_amount := 300 / P in
  let reduced_price := 0.85 * P in
  let new_amount := 300 / reduced_price in
  new_amount = original_amount + 7

theorem find_original_petrol_price (P : ℝ) (h : original_petrol_price P) : P ≈ 45 / 5.95 :=
by {
  sorry
}

end find_original_petrol_price_l70_70213


namespace inequality_sqrt_l70_70011

theorem inequality_sqrt {m n : ℕ} (h1 : m > n) (h2 : n ≥ 2) : 
  (n.root m : ℝ) - (m.root n : ℝ) > 1 / (m * n : ℝ) :=
by {
  sorry
}

end inequality_sqrt_l70_70011


namespace rectangle_not_equal_square_area_l70_70307

noncomputable def diagonal (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

def new_rectangle_area (a b : ℝ) : ℝ :=
  let d := diagonal a b
  let base := d + b
  let height := d - b
  base * height

def square_area (a b : ℝ) : ℝ :=
  (a + b) ^ 2

theorem rectangle_not_equal_square_area : 
  new_rectangle_area 8 15 ≠ square_area 8 15 :=
by {
  have h : diagonal 8 15 = 17 := by {
    -- Calculation steps of the diagonal
    sorry,
  },
  rw [new_rectangle_area, square_area],
  rw h,
  rw [Real.sqrt_eq_rpow, Real.pow_two (15: ℝ), Real.pow_two (8: ℝ)], -- Explore reals with powers
  norm_num, -- Normalize to the numbers (helps floating point exact calculations)
  norm_num,
  exact ne_of_lt (by norm_num),  -- Prove by non-equality with normalized numbers
}

end rectangle_not_equal_square_area_l70_70307


namespace xy_sum_l70_70001

theorem xy_sum (x y : ℝ) (h1 : 2 / x + 3 / y = 4) (h2 : 2 / x - 3 / y = -2) : x + y = 3 := by
  sorry

end xy_sum_l70_70001


namespace angle_C_when_b_eq_sqrt3_max_area_of_triangle_l70_70387

theorem angle_C_when_b_eq_sqrt3 (a A b : ℝ) (h1 : a = 1) (h2 : A = π / 6) (h3 : b = sqrt 3) :
  C = π / 2 ∨ C = π / 6 := 
sorry

theorem max_area_of_triangle (a A : ℝ) (h1 : a = 1) (h2 : A = π / 6) :
  area ≤ (2 + sqrt 3) / 4 := 
sorry

end angle_C_when_b_eq_sqrt3_max_area_of_triangle_l70_70387


namespace quadratic_inequality_solution_l70_70366

variable (x : ℝ)

theorem quadratic_inequality_solution (hx : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_solution_l70_70366


namespace rectangle_area_l70_70212

theorem rectangle_area (w d : ℝ) 
  (h1 : d = (w^2 + (3 * w)^2) ^ (1/2))
  (h2 : ∃ A : ℝ, A = w * 3 * w) :
  ∃ A : ℝ, A = 3 * (d^2 / 10) := 
by {
  sorry
}

end rectangle_area_l70_70212


namespace exists_poly_degree_6_l70_70262

theorem exists_poly_degree_6 (x : ℝ) :
  ∃ f : ℝ → ℝ, polynomial.degree f = 6 ∧ ∀ x, f (Real.sin x) + f (Real.cos x) = 1 :=
by
  sorry

end exists_poly_degree_6_l70_70262


namespace exists_a_log_eq_l70_70093

theorem exists_a_log_eq (a : ℝ) (h : a = 10 ^ ((Real.log 2 * Real.log 3) / (Real.log 2 + Real.log 3))) :
  ∀ x > 0, Real.log x / Real.log 2 + Real.log x / Real.log 3 = Real.log x / Real.log a :=
by
  sorry

end exists_a_log_eq_l70_70093


namespace circle_projections_l70_70187

open Real

def is_prime (n : ℕ) : Prop := ∃ p : ℕ, nat.prime p ∧ p = n

theorem circle_projections :
  ∀ (r : ℝ) (p q : ℕ) (m n : ℕ),
  r ∈ set.Ioo 0 ⊤ ∧
  r ≠ 0 ∧
  r % 2 = 1 ∧
  is_prime p ∧
  is_prime q ∧
  0 < m ∧ 0 < n ∧
  let u := p^m in
  let v := q^n in
  u^2 + v^2 = r^2 ∧
  u > v →
  let A := (r, 0) in
  let B := (-r, 0) in
  let C := (0, -r) in
  let D := (0, r) in
  let P := (u, v) in
  let M := (u, 0) in
  let N := (0, v) in
  dist A M = 1 ∧
  dist B M = 9 ∧
  dist C N = 8 ∧
  dist D N = 2 :=
begin
  intro r p q m n,
  rintro ⟨hr_pos, hr_nonzero, hr_odd, prime_p, prime_q, hm_pos, hn_pos, hu, hv, huv_eq, h_u_gt_v⟩,
  let u := p^m,
  let v := q^n,
  simp only [u, v] at huv_eq h_u_gt_v,
  have h_hyp := dist_eq.users P M A u v r huv_eq p q m n hr_nonzero hr_pos prime_p prime_q hm_pos hn_pos,
  let A := (r, 0),
  let B := (-r, 0),
  let C := (0, -r),
  let D := (0, r),
  let P := (u, v),
  let M := (u, 0),
  let N := (0, v),
  suffices : dist A M = 1 ∧ dist B M = 9 ∧ dist C N = 8 ∧ dist D N = 2,
  exact this,
  exact sorry
end

end circle_projections_l70_70187


namespace number_of_unicorns_l70_70134

theorem number_of_unicorns :
  (∀ u : Nat, ∀ f : Nat, u.steps = 9000 / 3 ∧ (u.steps * 4) * u = 72000 → u = 6) :=
by
  sorry

end number_of_unicorns_l70_70134


namespace pyramid_volume_l70_70159

-- Given definitions for the conditions
variables (r p q : ℝ)
variables (h₁ : 0 < r) (h₂ : 0 < p) (h₃ : 0 < q) (h₄ : p ≠ q)

-- Statement to prove
theorem pyramid_volume (r p q : ℝ) (h₁ : 0 < r) (h₂ : 0 < p) (h₃ : 0 < q) (h₄ : p ≠ q) : 
  let volume := (4 / 3) * (r^3 * (p + q)^2) / (p * (q - p)) in 
  True := 
begin
  have : volume = volume := by reflexivity,
  trivial,
end

end pyramid_volume_l70_70159


namespace proof_l70_70037

open Graph

-- Define a undirected graph where all vertices have the same degree
def is_k_regular_graph (G : simple_graph V) (k : ℕ) : Prop :=
  ∀ v : V, G.degree v = k

def exists_4_regular_graph_10_vertices : Prop :=
  ∃ (G : simple_graph (fin 10)), is_k_regular_graph G 4

theorem proof : exists_4_regular_graph_10_vertices :=
  sorry

end proof_l70_70037


namespace larger_integer_is_24_l70_70515

theorem larger_integer_is_24 {x : ℤ} (h1 : ∃ x, 4 * x = x + 6) :
  ∃ y, y = 4 * x ∧ y = 24 := by
  sorry

end larger_integer_is_24_l70_70515


namespace least_possible_product_of_primes_gt_50_l70_70152

open Nat

theorem least_possible_product_of_primes_gt_50 : 
  ∃ (p q : ℕ), prime p ∧ prime q ∧ p ≠ q ∧ p > 50 ∧ q > 50 ∧ (p * q = 3127) := 
  by
  exists 53
  exists 59
  repeat { sorry }

end least_possible_product_of_primes_gt_50_l70_70152


namespace sum_f_eq_zero_l70_70709

noncomputable def f (n : ℕ) : ℝ :=
  ∑' k : ℕ, 1 / (n + 2 + k) ^ n

theorem sum_f_eq_zero :
  ∑' n, if n >= 3 then f n else 0 = 0 :=
by
  sorry

end sum_f_eq_zero_l70_70709


namespace polynomial_has_root_l70_70957

theorem polynomial_has_root {a b c d : ℝ} 
  (h : a * c = 2 * b + 2 * d) : 
  ∃ x : ℝ, (x^2 + a * x + b = 0) ∨ (x^2 + c * x + d = 0) :=
by 
  sorry

end polynomial_has_root_l70_70957


namespace number_of_ways_to_arrange_plants_l70_70893

-- Definitions of the plants and lamps
inductive Plant
| rose1 : Plant
| rose2 : Plant
| orchid : Plant

inductive Lamp
| yellow1 : Lamp
| yellow2 : Lamp
| blue1 : Lamp
| blue2 : Lamp

-- Define the problem's main goal
theorem number_of_ways_to_arrange_plants : 
  ∃ (f : Plant → Lamp), 
  (∃! y, f Plant.rose1 = y ∧ f Plant.rose2 = y ∧ f Plant.orchid = y) ∨
  (∃! y, f Plant.rose1 = y ∧ f Plant.rose2 = y ∧ ∃! b, f Plant.orchid = b ∧ b ≠ y) ∨
  (∃! y, f Plant.orchid = y ∧ ∃! b, f Plant.rose1 = b ∧ f Plant.rose2 = b ∧ b ≠ y) →
  (∃! y, f Plant.rose1 = y ∧ ∃! b1, f Plant.orchid = b1 ∧ ∃! b2, f Plant.rose2 = b2 ∧ y ≠ b1 ∧ b1 = b2) →
  (∃! b, f Plant.rose1 = b ∧ ∃! y, f Plant.orchid = y ∧ ∃! y2, f Plant.rose2 = y2 ∧ b ≠ y ∧ y2 ≠ b ∧ y2 = y) →
  (∃! y, f Plant.rose1 = y ∧ ∃! b, f Plant.orchid = b ∧ ∃! b2, f Plant.rose2 = b2 ∧ y ≠ b ∧ b ≠ b2 ∧ b2 = y) →
  (∃! y, f Plant.rose2 = y ∧ ∃! b, f Plant.rose1 = b ∧ ∃! b2, f Plant.orchid = b2 ∧ y ≠ b ∧ b ≠ b2 ∧ b2 = y) →
  f = 14 :=
by sorry

end number_of_ways_to_arrange_plants_l70_70893


namespace supermarkets_medium_sample_l70_70201

theorem supermarkets_medium_sample (L M S sample : ℕ) (hL : L = 200) (hM : M = 400) (hS : S = 1400) (h_sample : sample = 100) :
  let total := L + M + S in
  let proportion_medium := M.to_float / total.to_float in
  sample * proportion_medium = 20 := 
by
  sorry

end supermarkets_medium_sample_l70_70201


namespace box_volume_l70_70166

variable (l w h : ℝ)
variable (lw_eq : l * w = 30)
variable (wh_eq : w * h = 40)
variable (lh_eq : l * h = 12)

theorem box_volume : l * w * h = 120 := by
  sorry

end box_volume_l70_70166


namespace sum_series_odd_l70_70974

theorem sum_series_odd : 
  let n := 1993 in (1 + 2 + 3 + ... + n) % 2 = 1 :=
by
  let n : ℕ := 1993
  have h_sum_formula : (1 + 2 + ... + n) = n * (n + 1) / 2 := sorry
  have h_odd_product : (n * (n + 1) / 2) % 2 = 1 := sorry
  exact h_odd_product

end sum_series_odd_l70_70974


namespace lines_concur_l70_70827

noncomputable def orthocenter (A B C : Point) : Point := sorry

noncomputable def on_circle (X H : Point) (l : Line) : Point := sorry

noncomputable def concur_at_single_point (A1 A2 B1 B2 C1 C2 : Point) : Prop :=
  ∃ P : Point, on_line A1 A2 P ∧ on_line B1 B2 P ∧ on_line C1 C2 P

theorem lines_concur 
  (A B C X : Point)
  (H : Point := orthocenter A B C)
  (circle_with_diameter_XH : Circle := circle_with_diameter X H)
  (A1 := on_circle X H (line_through H A))
  (B1 := on_circle X H (line_through H B))
  (C1 := on_circle X H (line_through H C))
  (A2 := on_circle X H (line_through X A))
  (B2 := on_circle X H (line_through X B))
  (C2 := on_circle X H (line_through X C)) :
  concur_at_single_point A1 A2 B1 B2 C1 C2 :=
sorry

end lines_concur_l70_70827


namespace problem_solution_l70_70053

theorem problem_solution
  (p q : ℝ)
  (h₁ : p ≠ q)
  (h₂ : (x : ℝ) → (x - 5) * (x + 3) = 24 * x - 72 → x = p ∨ x = q)
  (h₃ : p > q) :
  p - q = 20 :=
sorry

end problem_solution_l70_70053


namespace find_f6_l70_70933

noncomputable def f : ℝ → ℝ :=
sorry

theorem find_f6 (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
                (h2 : f 5 = 6) :
  f 6 = 36 / 5 :=
sorry

end find_f6_l70_70933


namespace primes_eq_condition_l70_70659

theorem primes_eq_condition (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) : 
  p + q^2 = r^4 → p = 7 ∧ q = 3 ∧ r = 2 := 
by
  sorry

end primes_eq_condition_l70_70659


namespace g_eq_g_inv_iff_l70_70645

def g (x : ℝ) : ℝ := 3 * x - 7

def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_iff (x : ℝ) : g x = g_inv x ↔ x = 7 / 2 :=
by {
  sorry
}

end g_eq_g_inv_iff_l70_70645


namespace ratio_of_areas_A_to_C_l70_70091

-- Definitions based on conditions
def perimeter_of_square_A : ℝ := 16
def side_length_A : ℝ := perimeter_of_square_A / 4
def multiplication_factor_C_A : ℝ := 1.5
def side_length_C : ℝ := multiplication_factor_C_A * side_length_A

-- Areas of the squares
def area_A : ℝ := side_length_A ^ 2
def area_C : ℝ := side_length_C ^ 2

-- The theorem to prove
theorem ratio_of_areas_A_to_C :
  (area_A / area_C) = 4 / 9 :=
by
  sorry

end ratio_of_areas_A_to_C_l70_70091


namespace exists_R_l70_70614

-- Definitions and assumptions
variables {α : Type*} [metric_space α] {n : ℕ}
variables {A : fin n → α} (h_no_collinear : ∀ i j k : fin n, i ≠ j → j ≠ k → k ≠ i → ¬ collinear {A i, A j, A k})
variables {P Q : α} (h_distinct : P ≠ Q) (h_not_A : ∀ i : fin n, P ≠ A i ∧ Q ≠ A i)
variables (h_distance_sum : ∑ (i : fin n), dist P (A i) = ∑ (i : fin n), dist Q (A i))

-- Theorem statement
theorem exists_R {P Q : α} (h_different : P ≠ Q) (h_not_An : ∀ i : fin n, P ≠ A i ∧ Q ≠ A i)
  (h_equal_sum : ∑ (i : fin n), dist P (A i) = ∑ (i : fin n), dist Q (A i)) :
  ∃ R : α, ∑ (i : fin n), dist R (A i) < ∑ (i : fin n), dist P (A i) := by
  sorry

end exists_R_l70_70614


namespace teresa_ordered_sandwiches_l70_70483

variables (cost_per_sandwich cost_salami cost_brie cost_olives cost_feta cost_bread total_amount_spent : ℝ)

def cost_brie_def : ℝ := 3 * cost_salami
def cost_olives_def : ℝ := 10 * (1 / 4)
def cost_feta_def : ℝ := 8 * (1 / 2)
def cost_other_items : ℝ := cost_salami + cost_brie + cost_olives + cost_feta + cost_bread
def remaining_amount : ℝ := total_amount_spent - cost_other_items

theorem teresa_ordered_sandwiches :
  cost_per_sandwich = 7.75 ∧ cost_salami = 4.00 ∧ cost_brie = cost_brie_def ∧ cost_olives = cost_olives_def ∧ cost_feta = cost_feta_def ∧ cost_bread = 2.00 ∧ total_amount_spent = 40.00
  → remaining_amount / cost_per_sandwich = 2 :=
by
  sorry

end teresa_ordered_sandwiches_l70_70483


namespace total_capacity_l70_70778

def eight_liters : ℝ := 8
def percentage : ℝ := 0.20
def num_containers : ℕ := 40

theorem total_capacity (h : eight_liters = percentage * capacity) :
  40 * (eight_liters / percentage) = 1600 := sorry

end total_capacity_l70_70778


namespace find_angle_4_l70_70023

variable {α : Type} [LinearOrderedAddCommMonoid α]

theorem find_angle_4
  (angle1 angle2 angle3 angle4 angle5 : α)
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (h3 : angle2 + angle5 = 180) :
  angle4 = 90 :=
by
  -- Proof will be here
  sorry

end find_angle_4_l70_70023


namespace compute_c_minus_d_squared_eq_0_l70_70835

-- Defining conditions
def multiples_of_n_under_m (n m : ℕ) : ℕ :=
  (m - 1) / n

-- Defining the specific values
def c : ℕ := multiples_of_n_under_m 9 60
def d : ℕ := multiples_of_n_under_m 9 60  -- Since every multiple of 9 is a multiple of 3

theorem compute_c_minus_d_squared_eq_0 : (c - d) ^ 2 = 0 := by
  sorry

end compute_c_minus_d_squared_eq_0_l70_70835


namespace cauchy_schwarz_inequality_l70_70005

theorem cauchy_schwarz_inequality (n : ℕ) (a b : ℕ → ℝ) : 
  (∑ i in Finset.range n, a i * b i) ^ 2 ≤ (∑ i in Finset.range n, (a i) ^ 2) * (∑ i in Finset.range n, (b i) ^ 2) := 
by
  sorry

end cauchy_schwarz_inequality_l70_70005


namespace min_sqrt_distance_to_origin_l70_70731

theorem min_sqrt_distance_to_origin (x y : ℝ) (h : 2 * x + y + 5 = 0) : sqrt (x^2 + y^2) = sqrt 5 :=
sorry

end min_sqrt_distance_to_origin_l70_70731


namespace smallest_positive_integer_a_l70_70290

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

theorem smallest_positive_integer_a :
  ∃ (a : ℕ), 0 < a ∧ (isPerfectSquare (10 + a)) ∧ (isPerfectSquare (10 * a)) ∧ 
  ∀ b : ℕ, 0 < b ∧ (isPerfectSquare (10 + b)) ∧ (isPerfectSquare (10 * b)) → a ≤ b :=
sorry

end smallest_positive_integer_a_l70_70290


namespace complex_number_evaluation_l70_70746

-- Defining the complex numbers involved
def z1 : ℂ := (2 - complex.i) / (2 + complex.i)
def z2 : ℂ := (2 + complex.i) / (2 - complex.i)

-- The main problem statement to prove
theorem complex_number_evaluation : z1 - z2 = -8 * complex.i / 5 := by
  sorry

end complex_number_evaluation_l70_70746


namespace correlation_coefficient_high_l70_70381

-- Definition of high linear correlation
def is_high_linear_correlation (r : ℝ) : Prop := abs r ≥ 1 - ε 

-- Theorem statement
theorem correlation_coefficient_high (r : ℝ) (ε: ℝ) (hε : 0 < ε) :
  is_high_linear_correlation r → abs r ≈ 1 :=
sorry

end correlation_coefficient_high_l70_70381


namespace least_possible_product_of_two_distinct_primes_greater_than_50_l70_70145

open nat

theorem least_possible_product_of_two_distinct_primes_greater_than_50 :
  ∃ p q : ℕ, p ≠ q ∧ prime p ∧ prime q ∧ p > 50 ∧ q > 50 ∧ 
  (∀ p' q' : ℕ, p' ≠ q' → prime p' → prime q' → p' > 50 → q' > 50 → p * q ≤ p' * q') ∧ p * q = 3127 :=
by
  sorry

end least_possible_product_of_two_distinct_primes_greater_than_50_l70_70145


namespace train_a_distance_traveled_l70_70999

variable (distance : ℝ) (speedA : ℝ) (speedB : ℝ) (relative_speed : ℝ) (time_to_meet : ℝ) 

axiom condition1 : distance = 450
axiom condition2 : speedA = 50
axiom condition3 : speedB = 50
axiom condition4 : relative_speed = speedA + speedB
axiom condition5 : time_to_meet = distance / relative_speed

theorem train_a_distance_traveled : (50 * time_to_meet) = 225 := by
  sorry

end train_a_distance_traveled_l70_70999


namespace tan_in_third_quadrant_l70_70319

-- Define the conditions as hypotheses:
variables {α : Real} (h1 : sin α = -12/13) (h2 : π < α ∧ α < 3 * π / 2)

-- State the goal using the conditions:
theorem tan_in_third_quadrant (h1 : sin α = -12/13) (h2 : π < α ∧ α < 3 * π / 2) : tan α = 12/5 := by
  sorry

end tan_in_third_quadrant_l70_70319


namespace mangoes_ratio_l70_70226

theorem mangoes_ratio (a d_a : ℕ)
  (h1 : a = 60)
  (h2 : a + d_a = 75) : a / (75 - a) = 4 := by
  sorry

end mangoes_ratio_l70_70226


namespace maximum_volume_tetrahedron_l70_70610

-- Define the conditions
variables (x y z : ℝ)
axiom h1 : ∠BAC = 90
axiom h2 : ∠BAD = 90
axiom h3 : ∠CAD = 90
axiom total_edge_length : x + y + z + (Real.sqrt (x^2 + y^2)) + (Real.sqrt (y^2 + z^2)) + (Real.sqrt (z^2 + x^2)) = 1

-- Define the volume function
def volume (x y z : ℝ) : ℝ := (x * y * z) / 6

-- Define the maximum volume value
def max_volume : ℝ := (5 * Real.sqrt 2 - 7) / 162

-- The theorem we need to prove
theorem maximum_volume_tetrahedron : volume x y z ≤ max_volume :=
sorry

end maximum_volume_tetrahedron_l70_70610


namespace linear_regression_passes_through_center_l70_70313

theorem linear_regression_passes_through_center :
  let x_vals := [1, 2, 2, 3] in
  let y_vals := [1, 3, 5, 7] in
  let n := 4 in
  let x_avg := (1 + 2 + 2 + 3) / n in
  let y_avg := (1 + 3 + 5 + 7) / n in
  (x_avg, y_avg) = (2, 4) :=
by
  sorry

end linear_regression_passes_through_center_l70_70313


namespace function_passes_through_point_l70_70936

noncomputable def passes_through_fixed_point (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  (a > 0) ∧ (a ≠ 1) ∧ (y = a^(2 - x) + 2)

theorem function_passes_through_point :
  ∀ (a : ℝ), a > 0 → a ≠ 1 → passes_through_fixed_point a 2 3 :=
by
  intros a ha1 ha2
  unfold passes_through_fixed_point
  split
  exact ha1
  split
  exact ha2
  apply eq.symm
  sorry  -- Proof required here to complete the theorem

end function_passes_through_point_l70_70936


namespace calculate_expression_l70_70631

theorem calculate_expression :
  (56 * 0.57 * 0.85) / (2.8 * 19 * 1.7) = 0.3 :=
by
  sorry

end calculate_expression_l70_70631


namespace place_tokens_l70_70079

theorem place_tokens (initial_tokens : Fin 50 → Fin 50 → Bool) :
  ∃ (new_tokens : Fin 50 → Fin 50 → Bool), 
    (∑ i j, if new_tokens i j then 1 else 0) ≤ 99 ∧
    ∀ i, even (∑ j, if initial_tokens i j ∨ new_tokens i j then 1 else 0) ∧
    ∀ j, even (∑ i, if initial_tokens i j ∨ new_tokens i j then 1 else 0) :=
by sorry

end place_tokens_l70_70079


namespace sufficient_but_not_necessary_l70_70433

theorem sufficient_but_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧ ∃ x y : ℝ, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2) :=
by
  sorry

end sufficient_but_not_necessary_l70_70433


namespace simplify_expression_l70_70096

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := 4 * x^3 + 5 * x^2 + 2 * x + 8 - (3 * x^3 - 7 * x^2 + 4 * x - 6)

-- Define the simplified form
def simplified_expr (x : ℝ) : ℝ := x^3 + 12 * x^2 - 2 * x + 14

-- State the theorem
theorem simplify_expression (x : ℝ) : initial_expr x = simplified_expr x :=
by sorry

end simplify_expression_l70_70096


namespace negation_of_proposition_l70_70944

-- Definition for function being odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - f (x)

-- The proposition we are negating
def proposition (f : ℝ → ℝ) : Prop := is_odd f → is_odd (λ x, f (-x))

-- Negate the proposition
theorem negation_of_proposition :
  ¬ (∀ f : ℝ → ℝ, proposition f) ↔ (∀ f : ℝ → ℝ, ¬ is_odd f → ¬ is_odd (λ x, f (-x))) :=
sorry

end negation_of_proposition_l70_70944


namespace epsilon_max_success_ratio_l70_70656

theorem epsilon_max_success_ratio :
  ∃ (x y z w u v: ℕ), 
  (y ≠ 350) ∧
  0 < x ∧ 0 < z ∧ 0 < u ∧ 
  x < y ∧ z < w ∧ u < v ∧
  x + z + u < y + w + v ∧
  y + w + v = 800 ∧
  (x / y : ℚ) < (210 / 350 : ℚ) ∧ 
  (z / w : ℚ) < (delta_day_2_ratio) ∧ 
  (u / v : ℚ) < (delta_day_3_ratio) ∧ 
  (x + z + u) / 800 = (789 / 800 : ℚ) := 
by
  sorry

end epsilon_max_success_ratio_l70_70656


namespace translate_function_l70_70489

theorem translate_function (x : ℝ) :
  let f := λ x, 3 * x ^ 2 - 6 * x - 1
  let g := λ x, f (x + 1) + 3
  g x = 3 * x ^ 2 - 1 :=
by
  sorry

end translate_function_l70_70489


namespace least_product_of_distinct_primes_greater_than_50_l70_70148

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 :
  ∃ (p q : ℕ), distinct_primes_greater_than_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_50_l70_70148


namespace rhombus_perimeter_l70_70939

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) 
  (rhombus_properties : ∀ a b c d : ℝ, a = d1 / 2 ∧ b = d2 / 2 ∧ 
    (a^2 + b^2 = c^2) ∧ d = 4 * c) : 
  d = 20 := 
by 
  have h : (3^2 + 4^2 = (5: ℝ)^2) := 
    by norm_num
  exact (4 : ℝ) * 5

end rhombus_perimeter_l70_70939


namespace geometric_sequence_arithmetic_sequence_l70_70993

theorem geometric_sequence_arithmetic_sequence (a r : ℕ) :
  a > 0 → r > 0 →
  (a + a * r + a * r^2 = 21) →
  (a * r^2 - 2 * a * r - 9 = 0) →
  ({a, a * r, a * r^2} = {1, 4, 16} ∨ {a, a * r, a * r^2} = {16, 4, 1}) :=
begin
  sorry,
end

end geometric_sequence_arithmetic_sequence_l70_70993


namespace number_of_six_digit_with_sum_51_l70_70359

open Finset

/-- A digit is a number between 0 and 9 inclusive.-/
def is_digit (n : ℕ) : Prop :=
  n >= 0 ∧ n <= 9

/-- Friendly notation for digit sums -/
def digit_sum (n : Fin 6 → ℕ) : ℕ :=
  (Finset.univ.sum n)

def is_six_digit_with_sum_51 (n : Fin 6 → ℕ) : Prop :=
  (∀ i, is_digit (n i)) ∧ (digit_sum n = 51)

/-- There are exactly 56 six-digit numbers such that the sum of their digits is 51. -/
theorem number_of_six_digit_with_sum_51 : 
  card {n : Fin 6 → ℕ // is_six_digit_with_sum_51 n} = 56 :=
by
  sorry

end number_of_six_digit_with_sum_51_l70_70359


namespace sequence_sum_proof_l70_70254

theorem sequence_sum_proof (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) :
  (∀ n, (n > 0) → (n / ∑ i in range n, a i) = 1 / (2 * n + 1)) →
  (∀ n, (n > 0) → b n = (a n + 1) / 4) →
  (∑ k in range 10, 1 / (b k * b (k + 1))) = 10 / 11 :=
by
  sorry

end sequence_sum_proof_l70_70254


namespace solve_a_plus_b_l70_70363

theorem solve_a_plus_b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 7 * a + 2 * b = 54) : a + b = -103 / 31 :=
by
  sorry

end solve_a_plus_b_l70_70363


namespace problem1_problem2_l70_70756

open Set

variable {U : Set ℝ} (A B : Set ℝ)

def UA : U = univ := by sorry
def A_def : A = { x : ℝ | 0 < x ∧ x ≤ 2 } := by sorry
def B_def : B = { x : ℝ | x < -3 ∨ x > 1 } := by sorry

theorem problem1 : A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 2 } := 
by sorry

theorem problem2 : (U \ A) ∩ (U \ B) = { x : ℝ | -3 ≤ x ∧ x ≤ 0 } := 
by sorry

end problem1_problem2_l70_70756


namespace total_bottle_caps_in_collection_l70_70234

-- Statements of given conditions
def small_box_caps : ℕ := 35
def large_box_caps : ℕ := 75
def num_small_boxes : ℕ := 7
def num_large_boxes : ℕ := 3
def individual_caps : ℕ := 23

-- Theorem statement that needs to be proved
theorem total_bottle_caps_in_collection :
  small_box_caps * num_small_boxes + large_box_caps * num_large_boxes + individual_caps = 493 :=
by sorry

end total_bottle_caps_in_collection_l70_70234


namespace cos_angle_BAD_l70_70410

open Real

-- Definitions from conditions
def length_AB : ℝ := 4
def length_AC : ℝ := 7
def length_BC : ℝ := 9

-- The angle bisector condition doesn't need a separate definition for the goal.

-- The goal statement
theorem cos_angle_BAD (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (h_AB : dist A B = length_AB)
  (h_AC : dist A C = length_AC)
  (h_BC : dist B C = length_BC)
  (h_AD_bisects_BAC : AngleBisector (A B C D)): 
  cos_angle (B A D) = √70 / 14 := 
sorry

end cos_angle_BAD_l70_70410


namespace problem1_problem2_problem3_l70_70441

-- Definition of sets A, B, and U
def A : Set ℤ := {1, 2, 3, 4, 5}
def B : Set ℤ := {-1, 1, 2, 3}
def U : Set ℤ := {x | -1 ≤ x ∧ x < 6}

-- The complement of B in U
def C_U (B : Set ℤ) : Set ℤ := {x ∈ U | x ∉ B}

-- Problem statements
theorem problem1 : A ∩ B = {1, 2, 3} := by sorry
theorem problem2 : A ∪ B = {-1, 1, 2, 3, 4, 5} := by sorry
theorem problem3 : (C_U B) ∩ A = {4, 5} := by sorry

end problem1_problem2_problem3_l70_70441


namespace total_cats_l70_70452

theorem total_cats (current_cats : ℕ) (additional_cats : ℕ) (h1 : current_cats = 11) (h2 : additional_cats = 32):
  current_cats + additional_cats = 43 :=
by
  -- We state the given conditions:
  -- current_cats = 11
  -- additional_cats = 32
  -- We need to prove:
  -- current_cats + additional_cats = 43
  sorry

end total_cats_l70_70452


namespace complex_subset_sum_modulus_l70_70849

open Complex

theorem complex_subset_sum_modulus {n : ℕ} (z : Fin n → ℂ) 
  (h : (∑ i in Finset.finRange n, Complex.abs (z i)) = 1) : 
  ∃ (s : Finset (Fin n)), Complex.abs (∑ i in s, z i) ≥ 1 / 6 := 
sorry

end complex_subset_sum_modulus_l70_70849


namespace quadratic_inequality_solution_l70_70369

variable (x : ℝ)

theorem quadratic_inequality_solution (hx : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_solution_l70_70369


namespace six_digit_number_condition_l70_70413

theorem six_digit_number_condition :
  ∃ A B : ℕ, 100 ≤ A ∧ A < 1000 ∧ 100 ≤ B ∧ B < 1000 ∧
            1000 * B + A = 6 * (1000 * A + B) :=
by
  sorry

end six_digit_number_condition_l70_70413


namespace fraction_simplification_l70_70095

theorem fraction_simplification :
  (1 / 330) + (19 / 30) = 7 / 11 :=
by
  sorry

end fraction_simplification_l70_70095


namespace unique_solution_l70_70664

-- Define the condition that p, q, r are prime numbers
variables {p q r : ℕ}

-- Assume that p, q, and r are primes
axiom p_prime : Prime p
axiom q_prime : Prime q
axiom r_prime : Prime r

-- Define the main theorem to state that the only solution is (7, 3, 2)
theorem unique_solution : p + q^2 = r^4 → (p, q, r) = (7, 3, 2) :=
by
  sorry

end unique_solution_l70_70664


namespace unique_solution_l70_70666

-- Define the condition that p, q, r are prime numbers
variables {p q r : ℕ}

-- Assume that p, q, and r are primes
axiom p_prime : Prime p
axiom q_prime : Prime q
axiom r_prime : Prime r

-- Define the main theorem to state that the only solution is (7, 3, 2)
theorem unique_solution : p + q^2 = r^4 → (p, q, r) = (7, 3, 2) :=
by
  sorry

end unique_solution_l70_70666


namespace value_of_x_for_g_equals_g_inv_l70_70648

noncomputable def g (x : ℝ) : ℝ := 3 * x - 7
  
noncomputable def g_inv (x : ℝ) : ℝ := (x + 7) / 3
  
theorem value_of_x_for_g_equals_g_inv : ∃ x : ℝ, g x = g_inv x ∧ x = 3.5 :=
by
  sorry

end value_of_x_for_g_equals_g_inv_l70_70648


namespace initial_stock_initial_stock_l70_70967

-- Define the sequence of grain changes over 3 days
def grain_changes : List Int := [+26, -32, -25, +34, -38, +10]

-- Define the current stock of grain
def current_stock : Int := 480

-- Total change in grain over 3 days
def total_change (changes : List Int) : Int :=
  changes.sum

-- Prove the stock 3 days ago given the current stock and the total change
theorem initial_stock (changes : List Int) (current_stock : Int) :
  total_change changes = -25 → current_stock = 480 → current_stock + 25 = 505 :=
by
  intros h1 h2
  rw [h1, h2]
  linarith

/-
Theorem initial_stock proves that the initial stock of grain 3 days ago was 505 tons.
-/

end initial_stock_initial_stock_l70_70967


namespace max_profit_l70_70202

noncomputable def profit (x : ℝ) : ℝ :=
  20 * x - 3 * x^2 + 96 * Real.log x - 90

theorem max_profit :
  ∃ x : ℝ, 4 ≤ x ∧ x ≤ 12 ∧ 
  (∀ y : ℝ, 4 ≤ y ∧ y ≤ 12 → profit y ≤ profit x) ∧ profit x = 96 * Real.log 6 - 78 :=
by
  sorry

end max_profit_l70_70202


namespace angle_between_bisectors_l70_70576

-- Define the geometric setup
structure Triangle :=
  (A B C : Point)

def externalAngleBisector (A B C : Point) : Line := sorry
def incenter (A B C : Point) : Point := sorry

open Angle

theorem angle_between_bisectors 
  (A B C : Point) 
  (O : Point := incenter A B C)
  (O1 O2 O3 : Point) 
  (h1 : is_external_bisector A B O1)
  (h2 : is_external_bisector B C O2)
  (h3 : is_external_bisector C A O3)
  : angle_between (Line.mk O1 O2) (Line.mk O O3) = 90 :=
sorry

end angle_between_bisectors_l70_70576


namespace angle_between_vectors_is_90_l70_70831

open Real
open Matrix

noncomputable def a : Vector 3 := ![2, -3, -6]
noncomputable def b : Vector 3 := ![sqrt 11, 5, -2]
noncomputable def c : Vector 3 := ![15, -5, 20]

def dot_product (v w : Vector 3) : ℝ :=
  v 0 * w 0 + v 1 * w 1 + v 2 * w 2

def angle_is_90_degrees : Prop :=
  let ab := dot_product a b
  let ac := dot_product a c
  let resultant := (ac • b).to_fs - (ab • c).to_fs
  dot_product a resultant = 0

theorem angle_between_vectors_is_90 :
  angle_is_90_degrees := sorry

end angle_between_vectors_is_90_l70_70831


namespace total_capacity_l70_70776

def eight_liters : ℝ := 8
def percentage : ℝ := 0.20
def num_containers : ℕ := 40

theorem total_capacity (h : eight_liters = percentage * capacity) :
  40 * (eight_liters / percentage) = 1600 := sorry

end total_capacity_l70_70776


namespace protein_percentage_in_powder_l70_70447

def matt_weight := 80 -- Matt's body weight in kg
def protein_per_kg_per_day := 2 -- grams of protein per kilogram per day
def total_powder_per_week := 1400 -- total grams of protein powder per week

theorem protein_percentage_in_powder :
  (1120 * 100 / total_powder_per_week) = 80 :=
by
  have daily_protein := protein_per_kg_per_day * matt_weight
  have weekly_protein := daily_protein * 7
  calc
    (weekly_protein * 100 / total_powder_per_week)
    = (1120 * 100 / total_powder_per_week) : by sorry
    ... = 80 : by sorry

end protein_percentage_in_powder_l70_70447


namespace coprime_lcm_inequality_l70_70344

theorem coprime_lcm_inequality
  (p q : ℕ)
  (hpq_coprime : Nat.gcd p q = 1)
  (hp_gt_1 : p > 1)
  (hq_gt_1 : q > 1)
  (hpq_diff_gt_1 : abs (p - q) > 1) :
  ∃ n : ℕ, Nat.lcm (p + n) (q + n) < Nat.lcm p q :=
by
  sorry

end coprime_lcm_inequality_l70_70344


namespace game_show_prize_guess_l70_70594

noncomputable def total_possible_guesses : ℕ :=
  (Nat.choose 8 3) * (Nat.choose 5 3) * (Nat.choose 2 2) * (Nat.choose 7 3)

theorem game_show_prize_guess :
  total_possible_guesses = 19600 :=
by
  -- Omitted proof steps
  sorry

end game_show_prize_guess_l70_70594


namespace prime_solution_unique_l70_70669

theorem prime_solution_unique:
  ∃ p q r : ℕ, (prime p ∧ prime q ∧ prime r ∧ p + q^2 = r^4 ∧ p = 7 ∧ q = 3 ∧ r = 2) ∧
  (∀ p' q' r' : ℕ, prime p' ∧ prime q' ∧ prime r' ∧ p' + q'^2 = r'^4 → (p' = 7 ∧ q' = 3 ∧ r' = 2)) :=
by {
  sorry
}

end prime_solution_unique_l70_70669


namespace intersection_of_A_and_B_l70_70299

def A : Set ℝ := {y | y > 1}
def B : Set ℝ := {x | Real.log x ≥ 0}
def Intersect : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = Intersect :=
by
  sorry

end intersection_of_A_and_B_l70_70299


namespace total_capacity_is_1600_l70_70775

/-- Eight liters is 20% of the capacity of one container. -/
def capacity_of_one_container := 8 / 0.20

/-- Calculate the total capacity of 40 such containers filled with water. -/
def total_capacity_of_40_containers := 40 * capacity_of_one_container

theorem total_capacity_is_1600 :
  total_capacity_of_40_containers = 1600 := by
    -- Proof is skipped using sorry.
    sorry

end total_capacity_is_1600_l70_70775


namespace mean_relation_l70_70493

theorem mean_relation (m n : ℕ) (x y : ℝ) (a : ℝ)
(hx_mean : (m : ℝ) * x) 
(hy_mean : (n : ℝ) * y)
(hxy_diff : x ≠ y)
(ha_bound : 0 < a ∧ a ≤ 1/2)
(hz_mean : (m + n : ℝ) * (a * x + (1 - a) * y) = m * x + n * y) :
m ≤ n :=
sorry

end mean_relation_l70_70493


namespace solve_for_y_l70_70259

theorem solve_for_y : ∀ (y : ℚ), 
  (y + 4 / 5 = 2 / 3 + y / 6) → y = -4 / 25 :=
by
  sorry

end solve_for_y_l70_70259


namespace quadratic_root_exists_l70_70952

theorem quadratic_root_exists {a b c d : ℝ} (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
by
  sorry

end quadratic_root_exists_l70_70952


namespace trucks_per_lane_l70_70601

theorem trucks_per_lane (number_of_lanes : ℕ) (total_vehicles : ℕ) (cars_in_lane : ℕ → ℕ) (trucks_in_lane : ℕ → ℕ) 
  (h1 : number_of_lanes = 4) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ number_of_lanes → cars_in_lane i = 2 * (∑ j in finset.range number_of_lanes, trucks_in_lane j))
  (h3 : ∑ i in finset.range number_of_lanes, cars_in_lane i + trucks_in_lane i = total_vehicles) :
  (∃ T, ∀ i, 1 ≤ i ∧ i ≤ number_of_lanes → trucks_in_lane i = T ∧ T = 60) :=
begin
  sorry
end

end trucks_per_lane_l70_70601


namespace lisa_needs_additional_marbles_l70_70861

theorem lisa_needs_additional_marbles
  (friends : ℕ) (initial_marbles : ℕ) (total_required_marbles : ℕ) :
  friends = 12 ∧ initial_marbles = 40 ∧ total_required_marbles = (friends * (friends + 1)) / 2 →
  total_required_marbles - initial_marbles = 38 :=
by
  sorry

end lisa_needs_additional_marbles_l70_70861


namespace bus_driver_total_hours_l70_70198

theorem bus_driver_total_hours (R OT : ℕ) (hR : R ≤ 40) (hRT : (R * 14 + OT * 24.5 = 982) : ∃ h : ℕ, h = R + OT := 57 :=
begin
  sorry
end

end bus_driver_total_hours_l70_70198


namespace tenth_term_arithmetic_sequence_l70_70986

theorem tenth_term_arithmetic_sequence (a d : ℕ) 
  (h1 : a + 2 * d = 10) 
  (h2 : a + 5 * d = 16) : 
  a + 9 * d = 24 := 
by 
  sorry

end tenth_term_arithmetic_sequence_l70_70986


namespace complex_expression_equality_l70_70718

noncomputable def z : ℂ := 1 - complex.I

theorem complex_expression_equality :
  (1 / (z^2)) - complex.conj(z) = -1 - (1 / 2) * complex.I := by
sorry

end complex_expression_equality_l70_70718


namespace solve_equation_l70_70908

theorem solve_equation:
  ∀ x : ℝ, (x + 1) / 3 - 1 = (5 * x - 1) / 6 → x = -1 :=
by
  intro x
  intro h
  sorry

end solve_equation_l70_70908


namespace slope_angle_vertical_line_l70_70783

theorem slope_angle_vertical_line : 
  ∀ α : ℝ, (∀ x y : ℝ, x = 1 → y = α) → α = Real.pi / 2 := 
by 
  sorry

end slope_angle_vertical_line_l70_70783


namespace multiple_of_interest_rate_l70_70928

theorem multiple_of_interest_rate (P r : ℝ) (m : ℝ) 
  (h1 : P * r^2 = 40) 
  (h2 : P * m^2 * r^2 = 360) : 
  m = 3 :=
by
  sorry

end multiple_of_interest_rate_l70_70928


namespace quadratic_root_exists_l70_70953

theorem quadratic_root_exists {a b c d : ℝ} (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
by
  sorry

end quadratic_root_exists_l70_70953


namespace average_square_feet_per_person_l70_70503

/-- The population of the United States in 2020 was estimated to be 331,000,000.
The total area of the country including territorial waters is estimated to be 3,800,000 square miles.
There are (5280)^2 square feet in one square mile. The best approximation
for the average number of square feet per person is 150,000. -/
theorem average_square_feet_per_person :
  let population := 331000000
  let area_sq_miles := 3800000
  let sq_feet_per_sq_mile := 5280 * 5280
  let total_sq_feet := area_sq_miles * sq_feet_per_sq_mile
  let avg_sq_feet_per_person := total_sq_feet / population
  abs (avg_sq_feet_per_person - 150000) < 100000 := 
by
  sorry

end average_square_feet_per_person_l70_70503


namespace cubes_sum_expr_l70_70846

variable {a b s p : ℝ}

theorem cubes_sum_expr (h1 : s = a + b) (h2 : p = a * b) : a^3 + b^3 = s^3 - 3 * s * p := by
  sorry

end cubes_sum_expr_l70_70846


namespace zero_in_interval_of_f_l70_70020

open Real

noncomputable def f (x : ℝ) : ℝ := cos x - x

theorem zero_in_interval_of_f (k : ℤ) (h : ∃ x, x ∈ Ioo (↑k - 1) ↑k ∧ f x = 0) : k = 1 :=
sorry

end zero_in_interval_of_f_l70_70020


namespace part_I_part_II_l70_70328
noncomputable def f (x : ℝ) : ℝ := x - (3 / 2) * Real.log x

theorem part_I : f' 1 = -1 / 2 :=
by sorry

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := f x + (1 / 2) * x^2 - b * x

theorem part_II (b : ℝ) (x1 x2 : ℝ) (hx : x1 < x2)
  (hb : b ≥ 7 / 2)
  (h1 : x1 + x2 = b - 1)
  (h2 : x1 * x2 = 3 / 2) :
  g x1 b - g x2 b ≥ (15 / 8) - 2 * Real.log 2 :=
by sorry

end part_I_part_II_l70_70328


namespace solve_eq_g_4_l70_70853

noncomputable def g : ℝ → ℝ := λ x, if x < 0 then 4*x + 8 else 3 * x - 18

theorem solve_eq_g_4 : {x : ℝ | g x = 4} = {-1, 22/3} :=
by sorry

end solve_eq_g_4_l70_70853


namespace perfect_square_expression_l70_70518

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, m^2 = (2 * l - n - k) * (2 * l - n + k) / 2 :=
by 
  sorry

end perfect_square_expression_l70_70518


namespace triangle_area_formula_l70_70278

def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

def area_of_triangle (A B C : ℝ × ℝ × ℝ) : ℝ :=
  let AB := distance A B
  let AC := distance A C
  let BC := distance B C
  let s := (AB + AC + BC) / 2
  Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))

theorem triangle_area_formula :
  let A := (0, 3, 8) : ℝ × ℝ × ℝ
      B := (-2, 2, 4) : ℝ × ℝ × ℝ
      C := (-3, 5, 4) : ℝ × ℝ × ℝ
  area_of_triangle A B C = 
    let AB := distance A B
        AC := distance A C
        BC := distance B C
        s := (AB + AC + BC) / 2
    Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)) := sorry

end triangle_area_formula_l70_70278


namespace at_least_two_squares_same_size_l70_70608

theorem at_least_two_squares_same_size (S : ℝ) : 
  ∃ a b : ℝ, a = b ∧ 
  (∀ i : ℕ, i < 10 → 
   ∀ j : ℕ, j < 10 → 
   (∃ k : ℕ, k < 9 ∧ 
    ((∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ x ≠ y → 
          (i = x ∧ j = y)) → 
        ((S / 10) = (a * k)) ∨ ((S / 10) = (b * k))))) := sorry

end at_least_two_squares_same_size_l70_70608


namespace four_digit_multiples_of_13_and_7_l70_70357

theorem four_digit_multiples_of_13_and_7 : 
  (∃ n : ℕ, 
    (∀ k : ℕ, 1000 ≤ k ∧ k < 10000 ∧ k % 91 = 0 → k = 1001 + 91 * (n - 11)) 
    ∧ n - 11 + 1 = 99) :=
by
  sorry

end four_digit_multiples_of_13_and_7_l70_70357


namespace sum_of_sequence_2015_l70_70308

noncomputable theory

def a : ℕ → ℕ 
| 0       := 5
| (n + 1) := if a n % 2 = 0 then a n / 2 else 3 * a n + 1

def S (n : ℕ) : ℕ := (Finset.range n).sum a

theorem sum_of_sequence_2015 : S 2015 = 4725 :=
sorry

end sum_of_sequence_2015_l70_70308


namespace temperature_on_sunday_l70_70509

theorem temperature_on_sunday (T_mon T_tue T_wed T_thu T_fri T_sat T_avg T_week_sum : ℕ) :
  T_mon = 50 → T_tue = 65 → T_wed = 36 → T_thu = 82 → T_fri = 72 → T_sat = 26 → 
  T_avg = 53 → T_week_sum = 7 * T_avg →
  T_week_sum - (T_mon + T_tue + T_wed + T_thu + T_fri + T_sat) = 40 :=
by
  intros
  rw [←Int.coe_nat_eq_coe_nat_iff] at *
  sorry

end temperature_on_sunday_l70_70509


namespace distance_between_intersections_eq_sqrt1_2_l70_70284

noncomputable def dist_intersections : ℝ :=
  let y1 := (1 / 2) ^ (1 / 3)
  let y2 := -((1 / 2) ^ (1 / 3))
  let x1 := (1 / 2)
  let x2 := - (1 / 2)
  real.sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)

theorem distance_between_intersections_eq_sqrt1_2 :
  dist_intersections = real.sqrt (1 + real.sqrt 2) :=
 sorry

end distance_between_intersections_eq_sqrt1_2_l70_70284


namespace curve_trajectory_a_eq_1_curve_fixed_point_a_ne_1_l70_70747

noncomputable def curve (x y a : ℝ) : ℝ :=
  x^2 + y^2 - 2 * a * x + 2 * (a - 2) * y + 2 

theorem curve_trajectory_a_eq_1 :
  ∃! (x y : ℝ), curve x y 1 = 0 ∧ x = 1 ∧ y = 1 := by
  sorry

theorem curve_fixed_point_a_ne_1 (a : ℝ) (ha : a ≠ 1) :
  curve 1 1 a = 0 := by
  sorry

end curve_trajectory_a_eq_1_curve_fixed_point_a_ne_1_l70_70747


namespace find_length_of_platform_l70_70586

noncomputable def length_of_platform
  (length_of_train : ℝ)
  (time_to_cross_pole : ℝ)
  (time_to_cross_platform : ℝ)
  : ℝ :=
  let speed_of_train := length_of_train / time_to_cross_pole
  let total_distance := speed_of_train * time_to_cross_platform
  total_distance - length_of_train

theorem find_length_of_platform :
  length_of_platform 900 18 39 = 1050 :=
by
  unfold length_of_platform
  norm_num
  sorry

end find_length_of_platform_l70_70586


namespace least_product_of_distinct_primes_greater_than_50_l70_70151

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 :
  ∃ (p q : ℕ), distinct_primes_greater_than_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_50_l70_70151


namespace final_price_after_changes_l70_70504

-- Define the initial conditions
def original_price : ℝ := 400
def decrease_percentage : ℝ := 15 / 100
def increase_percentage : ℝ := 40 / 100

-- Lean 4 statement of the proof problem
theorem final_price_after_changes :
  (original_price * (1 - decrease_percentage)) * (1 + increase_percentage) = 476 :=
by
  -- Proof goes here
  sorry

end final_price_after_changes_l70_70504


namespace least_possible_product_of_primes_gt_50_l70_70153

open Nat

theorem least_possible_product_of_primes_gt_50 : 
  ∃ (p q : ℕ), prime p ∧ prime q ∧ p ≠ q ∧ p > 50 ∧ q > 50 ∧ (p * q = 3127) := 
  by
  exists 53
  exists 59
  repeat { sorry }

end least_possible_product_of_primes_gt_50_l70_70153


namespace largest_angle_convex_hexagon_l70_70502

theorem largest_angle_convex_hexagon : 
  ∃ x : ℝ, (x-3) + (x-2) + (x-1) + x + (x+1) + (x+2) = 720 → (x + 2) = 122.5 :=
by 
  intros,
  sorry

end largest_angle_convex_hexagon_l70_70502


namespace a_2013_is_4_l70_70409

def seq : ℕ → ℕ
| 0       := 2   -- seq starts from a_1 which is indexed as 0 here
| 1       := 7   -- a_2 is indexed as 1 here
| (n + 2) := (seq n * seq (n + 1)) % 10   -- the units digit of a_n * a_{n+1}

theorem a_2013_is_4 : seq 2012 = 4 :=
sorry

end a_2013_is_4_l70_70409


namespace abs_inequality_solution_l70_70971

theorem abs_inequality_solution (x : ℝ) : |x - 2| < 1 ↔ 1 < x ∧ x < 3 :=
by
  -- the proof would go here
  sorry

end abs_inequality_solution_l70_70971


namespace kinetic_energy_and_time_l70_70199

noncomputable def kinetic_energy_at_boundary 
  (q Q : ℝ) (R : ℝ) (ε₀ : ℝ) : ℝ :=
  -q * Q / (8 * π * R * ε₀)

noncomputable def time_to_reach_boundary 
  (q Q : ℝ) (R : ℝ) (ε₀ m : ℝ) : ℝ :=
  (π / 2) * sqrt((4 * π * R^3 * ε₀ * m) / (q * Q))

theorem kinetic_energy_and_time
  (q Q : ℝ) (R : ℝ) (ε₀ m : ℝ)
  (K₀ : ℝ) (t : ℝ)
  (hqQ : q * Q < 0) : 
  K₀ = kinetic_energy_at_boundary q Q R ε₀ ∧ 
  t = time_to_reach_boundary q Q R ε₀ m :=
by 
  -- Placeholders for proof
  sorry

end kinetic_energy_and_time_l70_70199


namespace rectangular_solid_dimension_change_l70_70590

theorem rectangular_solid_dimension_change (a b : ℝ) (h : 2 * a^2 + 4 * a * b = 0.6 * (6 * a^2)) : b = 0.4 * a :=
by sorry

end rectangular_solid_dimension_change_l70_70590


namespace determine_c_l70_70002

theorem determine_c (a b c : ℤ) (h : ∃ d e : ℤ, (fun x : ℤ => (x^3 - x^2 - x - 1) * (d * x + e) = a * x^4 + b * x^3 + c * x^2 + 1)) : 
  c = 1 - a :=
by 
  sorry

end determine_c_l70_70002


namespace problem_5__l70_70728

-- Define the operation ⊗ on positive integers satisfying given conditions
def operation_⊗ (a b : ℕ) : ℕ :=
  -- placeholder definition to be specified by conditions
  sorry

-- State the problem to be proven in Lean 4
theorem problem_5_⊗_18 : ∀ a b : ℕ, a > 0 → b > 0 → 
  (operation_⊗ (a^2 * b) b = a * (operation_⊗ b b)) → 
  ((operation_⊗ a 1) = a^2 ) → 
  ((1 ⊗ 1) = 1) → 
  (operation_⊗ 5 18 = 8100) := 
by
  intros a b ha hb h1 h2 h3
  -- Provide a structure for proof outline
  sorry -- Proof goes here

end problem_5__l70_70728


namespace total_votes_calculation_l70_70025

-- Define the initial conditions
variables
  (V : ℕ) -- Total number of votes
  (votes_A votes_B votes_C votes_D initial_votes_A initial_votes_B : ℕ)
  (change : ℕ := 3000)

-- Initial vote percentages
def initial_vote_percentage_A := (40 : ℝ) / 100
def initial_vote_percentage_B := (30 : ℝ) / 100
def initial_vote_percentage_C := (20 : ℝ) / 100
def initial_vote_percentage_D := (10 : ℝ) / 100

-- Initial votes calculation
def initial_votes_A := initial_vote_percentage_A * V
def initial_votes_B := initial_vote_percentage_B * V
def initial_votes_C := initial_vote_percentage_C * V
def initial_votes_D := initial_vote_percentage_D * V

-- Condition: Candidate A wins with a margin of 10% votes
def candidate_A_margin := (initial_votes_A - initial_votes_B) = 0.10 * V

-- Votes after changes
def votes_A := initial_votes_A - change
def votes_B := initial_votes_B + change

-- Condition: Candidate B wins with a margin of 20% votes after change
def candidate_B_margin := (votes_B - votes_A) = 0.20 * V

-- Condition: Candidate C receives 10% more votes than Candidate D
def candidate_C_D_margin := initial_votes_C = initial_votes_D + 0.10 * V

-- The theorem to prove the total number of votes polled
theorem total_votes_calculation (h1 : candidate_A_margin) (h2 : candidate_B_margin) (h3 : candidate_C_D_margin)
    : V = 20000 :=
  sorry

end total_votes_calculation_l70_70025


namespace laura_owes_amount_l70_70570

-- Define the given conditions as variables
def principal : ℝ := 35
def rate : ℝ := 0.05
def time : ℝ := 1

-- Define the interest calculation
def interest : ℝ := principal * rate * time

-- Define the final amount owed calculation
def amount_owed : ℝ := principal + interest

-- State the theorem we want to prove
theorem laura_owes_amount
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (interest : ℝ := principal * rate * time)
  (amount_owed : ℝ := principal + interest) :
  amount_owed = 36.75 := 
by 
  -- proof would go here
  sorry

end laura_owes_amount_l70_70570


namespace greatest_visible_cubes_from_corner_l70_70229

def cube_visible_units (n : ℕ) : ℕ :=
  let faces := 3 * n^2
  let edges := 3 * (n - 1)
  let corner := 1
  faces - edges + corner

theorem greatest_visible_cubes_from_corner :
  cube_visible_units 9 = 220 :=
by
  unfold cube_visible_units
  rw [Nat.mul_sub_left_distrib, Nat.mul_one, Nat.add_sub_cancel]
  rfl

end greatest_visible_cubes_from_corner_l70_70229


namespace lisa_additional_marbles_l70_70872

theorem lisa_additional_marbles (n : ℕ) (m : ℕ) (s_n : ℕ) :
  n = 12 →
  m = 40 →
  (s_n = (list.sum (list.range (n + 1)))) →
  s_n - m = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp only [list.range_succ, list.sum_range_succ, nat.factorial, nat.succ_eq_add_one, nat.add_succ, mul_add, mul_one, mul_comm n]
  sorry

end lisa_additional_marbles_l70_70872


namespace nasadkas_in_barrel_l70_70404

def capacity (B N V : ℚ) :=
  (B + 20 * V = 3 * B) ∧ (19 * B + N + 15.5 * V = 20 * B + 8 * V)

theorem nasadkas_in_barrel (B N V : ℚ) (h : capacity B N V) : B / N = 4 :=
by
  sorry

end nasadkas_in_barrel_l70_70404


namespace base2_digit_difference_l70_70552

theorem base2_digit_difference :
  let binary_digits (n : ℕ) : ℕ := (nat.log 2 n) + 1 in
  let digit_diff (x y : ℕ) := binary_digits y - binary_digits x in
  digit_diff 300 1500 = 2 :=
by
  sorry

end base2_digit_difference_l70_70552


namespace find_a_range_l70_70074

-- Definitions and conditions based on the problem
def f (x : ℝ) : ℝ := -Real.exp x - x
def g (x : ℝ) (a : ℝ) : ℝ := a * x + 2 * Real.cos x

-- Derivatives of the functions
def f' (x : ℝ) : ℝ := -Real.exp x - 1
def g' (x : ℝ) (a : ℝ) : ℝ := a - 2 * Real.sin x

-- The main theorem statement
theorem find_a_range : { a : ℝ // -1 ≤ a ∧ a ≤ 2 } :=
by
  sorry

end find_a_range_l70_70074


namespace find_f_neg_five_half_l70_70430

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
else if 0 ≤ x + 2 ∧ x + 2 ≤ 1 then 2 * (x + 2) * (1 - (x + 2))
     else -2 * abs x * (1 - abs x)

theorem find_f_neg_five_half (x : ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 2) = f x)
  (h_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)) : 
  f (-5 / 2) = -1 / 2 :=
  by sorry

end find_f_neg_five_half_l70_70430


namespace range_of_x_satisfying_inequality_l70_70836

def f (x : ℝ) : ℝ := -- Define the function f (we will leave this definition open for now)
sorry
@[continuity] axiom f_increasing (x y : ℝ) (h : x < y) : f x < f y
axiom f_2_eq_1 : f 2 = 1
axiom f_xy_eq_f_x_add_f_y (x y : ℝ) : f (x * y) = f x + f y

noncomputable def f_4_eq_2 : f 4 = 2 := sorry

theorem range_of_x_satisfying_inequality (x : ℝ) :
  3 < x ∧ x ≤ 4 ↔ f x + f (x - 3) ≤ 2 :=
sorry

end range_of_x_satisfying_inequality_l70_70836


namespace lisa_needs_additional_marbles_l70_70870

/-- Lisa has 12 friends and 40 marbles. She needs to ensure each friend gets at least one marble and no two friends receive the same number of marbles. We need to find the minimum number of additional marbles needed to ensure this. -/
theorem lisa_needs_additional_marbles : 
  ∀ (friends marbles : ℕ), friends = 12 → marbles = 40 → 
  ∃ (additional_marbles : ℕ), additional_marbles = 38 ∧ 
  (∑ i in finset.range (friends + 1), i) - marbles = additional_marbles :=
by
  intros friends marbles friends_eq marbles_eq 
  use 38
  split
  · exact rfl
  calc (∑ i in finset.range (12 + 1), i) - 40 = 78 - 40 : by norm_num
                                  ... = 38 : by norm_num

end lisa_needs_additional_marbles_l70_70870


namespace triangle_inequality_l70_70826

variables {ABC A1B1C1 : Type} 
variables {a b c a1 b1 c1 R r1 : ℝ}
variables [Triangle ABC a b c R] [Triangle A1B1C1 a1 b1 c1 r1]

theorem triangle_inequality (h1 : Triangle ABC a b c R) 
  (h2 : Triangle A1B1C1 a1 b1 c1 r1) :
  a / a1 + b / b1 + c / c1 ≤ 3 * R / (2 * r1) := 
sorry

end triangle_inequality_l70_70826


namespace min_value_expression_l70_70734

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) :
  ∃ z : ℝ, z = 16 / 7 ∧ ∀ u > 0, ∀ v > 0, u + v = 4 → ((u^2 / (u + 1)) + (v^2 / (v + 2))) ≥ z :=
by
  sorry

end min_value_expression_l70_70734


namespace parametric_line_l70_70116
   
   variable {t : ℝ}

   def line (x y : ℝ) : Prop := y = 2 * x - 30
   def parametric (f : ℝ → ℝ) (x y : ℝ) : Prop := (x, y) = (f t, 20 * t - 10)

   theorem parametric_line (f : ℝ → ℝ) : 
     (∀ t, parametric f (f t) (20 * t - 10) → line (f t) (20 * t - 10)) → 
     f t = 10 * t + 10 :=
   by
     intro h
     sorry
   
end parametric_line_l70_70116


namespace num_four_digit_divisibles_l70_70355

theorem num_four_digit_divisibles : 
  ∃ n, n = 99 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 91 = 0 ↔ x ∈ (set.Ico 1001 9999) :=
by
  sorry

end num_four_digit_divisibles_l70_70355


namespace cell_growth_l70_70135

-- Declare the initial condition
def a_0 : ℕ := 10

-- Recurrence relation as a function
def recurrence (a : ℕ) : ℕ := 2 * (a - 2)

-- Define the sequence using recurrence relation
noncomputable def a (n : ℕ) : ℕ :=
  Nat.recOn n a_0 (λ n a_n, recurrence a_n)

-- Final statement to prove
theorem cell_growth : a 9 = 1540 :=
  sorry

end cell_growth_l70_70135


namespace g_eq_g_inv_at_7_over_2_l70_70655

def g (x : ℝ) : ℝ := 3 * x - 7
def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_at_7_over_2 : g (7 / 2) = g_inv (7 / 2) := by
  sorry

end g_eq_g_inv_at_7_over_2_l70_70655


namespace perfect_square_expression_l70_70522

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 :=
by
  sorry

end perfect_square_expression_l70_70522


namespace problem_probability_C_l70_70720

structure Lattice where
  A : Dot
  C : Dot
  N : Dot
  S : Dot
  E : Dot
  W : Dot
  path : Dot → Dot → Bool

def move_between_dots (start : Dot) (move_sequence : List Dot) (end : Dot) : Prop :=
  sorry -- some definition that models the ant's movement according to the lattice rules

def T_A (L : Lattice) := -- some definition of the transition relations
  sorry

def P_C_after_6_moves (L : Lattice) : ℚ := -- some definition for the probability
  sorry

theorem problem_probability_C (L : Lattice) (h1 : move_between_dots L.A [L.N, L.C, L.S, L.C, L.W, L.C] L.C) 
(h2 : move_between_dots L.A [L.N, L.C, L.S, L.N, L.W, L.C] L.C) :
  P_C_after_6_moves L = (1/2048) := 
  sorry

end problem_probability_C_l70_70720


namespace coefficient_m4n4_in_m_plus_n_expansion_8_l70_70167

theorem coefficient_m4n4_in_m_plus_n_expansion_8 :
  (nat.choose 8 4) = 70 :=
sorry

end coefficient_m4n4_in_m_plus_n_expansion_8_l70_70167


namespace triangle_incircles_perpendicular_l70_70033

theorem triangle_incircles_perpendicular 
  (A B C D : Point)
  (incircle_ABC : Incircle ABC)
  (tangency_D : TangencyPoint incircle_ABC AB = D)
  (incircle_ACD : Incircle ACD)
  (incircle_BCD : Incircle BCD)
  (center_ACD : Center incircle_ACD)
  (center_BCD : Center incircle_BCD)
  : Perpendicular (line center_ACD center_BCD) (line C D) :=
sorry

end triangle_incircles_perpendicular_l70_70033


namespace fraction_area_above_line_l70_70941

/-- The fraction of the area of the square above the line joining (2, 3) and (5, 1), 
given the square with vertices at (2, 1), (5, 1), (5, 4), and (2, 4), is 2/3. -/
theorem fraction_area_above_line : 
  let A := (2, 1)
  let B := (5, 1)
  let C := (5, 4)
  let D := (2, 4)
  let line := (2, 3) to (5, 1)
  let square_area := 9
  let triangle_area := 3
  (square_area - triangle_area) / square_area = (2 : ℚ) / 3 :=
by
  sorry

end fraction_area_above_line_l70_70941


namespace isosceles_base_length_l70_70121

noncomputable def equilateral_triangle_side (p : ℕ) : ℕ := p / 3

noncomputable def isosceles_base (p_iso : ℕ) (side : ℕ) : ℕ := p_iso - 2 * side

theorem isosceles_base_length :
  ∀ (p_equilateral p_isosceles : ℕ),
    p_equilateral = 60 → p_isosceles = 45 →
    isosceles_base p_isosceles (equilateral_triangle_side p_equilateral) = 5 :=
by
  intros p_eq p_iso h_eq h_iso
  rw [h_eq, h_iso]
  simp [equilateral_triangle_side, isosceles_base]
  sorry

end isosceles_base_length_l70_70121


namespace value_m_not_0_l70_70326

theorem value_m_not_0 (m : ℝ) :
  (∀ m, y = x^2 - (m-2)x + 4 → (y = (x - (m-2)/2)^2 - ((m-2)^2)/4 + 4) → 
  (∃ c : ℝ, (m = 2 ∨ m = -2 ∨ m = 6 ∨ m = c) → c ≠ 0)) :=
begin
  sorry
end

end value_m_not_0_l70_70326


namespace geometric_series_sum_l70_70236

theorem geometric_series_sum (a r : ℚ) (h_a : a = 1) (h_r : r = 1 / 3) : 
  (∑' n : ℕ, a * r ^ n) = 3 / 2 := 
by
  sorry

end geometric_series_sum_l70_70236


namespace complement_A_intersection_B_l70_70340

open Set

variable (α : Type*) [LinearOrder α] [TopologicalSpace α]

def A : Set α := { x : α | x > 3 }
def B : Set α := { x : α | 2 < x ∧ x < 4 }

theorem complement_A_intersection_B :
  (compl A) ∩ B = { x : α | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end complement_A_intersection_B_l70_70340


namespace mr_william_land_percentage_l70_70689

-- Define the conditions
def farm_tax_percentage : ℝ := 0.5
def total_tax_collected : ℝ := 3840
def mr_william_tax : ℝ := 480

-- Theorem statement proving the question == answer
theorem mr_william_land_percentage : 
  (mr_william_tax / total_tax_collected) * 100 = 12.5 := 
by
  -- sorry is used to skip the proof
  sorry

end mr_william_land_percentage_l70_70689


namespace range_of_a_distinct_solutions_maximum_value_of_F_l70_70748

-- Problem 1
theorem range_of_a_distinct_solutions (a : ℝ) (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = | x - a |) (hg : ∀ x, g x = a * x) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2) ↔ a ∈ set.Ioo (-1 : ℝ) 0 ∪ set.Ioo 0 1 :=
sorry

-- Problem 2
theorem maximum_value_of_F (a : ℝ) (F : ℝ → ℝ) 
  (hf : ∀ x, f x = | x - a |) (hg : ∀ x, g x = a * x) (hF : ∀ x, F x = g x * f x) (ha : a > 0) :
  ∀ x ∈ set.Icc (1 : ℝ) 2,
    F x = 
    if 0 < a ∧ a < (5/3 : ℝ) then 4 * a - 2 * a^2 
    else if (5/3 : ℝ) ≤ a ∧ a ≤ 2 then a^2 - a
    else if 2 < a ∧ a ≤ 4 then a^3 / 4
    else if 4 < a then 2 * a^2 - 4 * a
    else 0 :=
sorry

end range_of_a_distinct_solutions_maximum_value_of_F_l70_70748


namespace average_meat_consumption_example_l70_70222

-- Define the number of lions and the fact that the number of tigers is twice the number of lions
variables (x : ℕ)

-- Define daily meat consumption per tiger and per lion
noncomputable def meat_per_tiger : ℝ := 4.5
noncomputable def meat_per_lion : ℝ := 3.5

-- The total meat consumption per day for all animals and the total number of animals
def total_meat_consumption_per_day (x : ℕ) : ℝ :=
  (2 * x) * meat_per_tiger + x * meat_per_lion

def total_number_of_animals (x : ℕ) : ℝ :=
  (2:ℕ) * x + x

-- Average daily meat consumption per animal
noncomputable def average_meat_consumption_per_animal (x : ℕ) : ℝ :=
  total_meat_consumption_per_day x / total_number_of_animals x

-- The theorem statement to prove
theorem average_meat_consumption_example (x : ℕ) : average_meat_consumption_per_animal x = 25 / 6 := 
by 
  sorry

end average_meat_consumption_example_l70_70222


namespace perfect_square_expression_l70_70520

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, m^2 = (2 * l - n - k) * (2 * l - n + k) / 2 :=
by 
  sorry

end perfect_square_expression_l70_70520


namespace find_k_l70_70738

noncomputable def intersecting_parabola (k : ℝ) (m : ℝ) : Bool :=
  let x1 := (12 * k + 36).sqrt
  let x2 := -(12 * k + 36).sqrt
  let A := (x1, k * x1 + m)
  let B := (x2, k * x2 + m)
  let AB := (A.1 - B.1)^2 + (A.2 - B.2)^2
  AB == 36

theorem find_k (x : ℝ) (y : ℝ) :
  let p := λ k : ℝ, intersecting_parabola k 3
  p (√2) = true :=
by
  sorry

end find_k_l70_70738


namespace largest_angle_of_convex_hexagon_l70_70499

noncomputable def largest_angle (angles : Fin 6 → ℝ) (consecutive : ∀ i : Fin 5, angles i + 1 = angles (i + 1)) (sum_eq_720 : (∑ i, angles i) = 720) : ℝ :=
  sorry

theorem largest_angle_of_convex_hexagon (angles : Fin 6 → ℝ) (consecutive : ∀ i : Fin 5, angles i + 1 = angles (i + 1)) (sum_eq_720 : (∑ i, angles i) = 720) :
  largest_angle angles consecutive sum_eq_720 = 122.5 :=
  sorry

end largest_angle_of_convex_hexagon_l70_70499


namespace perfect_square_of_expression_l70_70527

theorem perfect_square_of_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 := 
by 
  sorry

end perfect_square_of_expression_l70_70527


namespace measure_angle_y_l70_70406

-- Define the conditions
variable {m n : Line} -- Lines m and n
variable {α : Angle} -- Angle α
variable (h_parallel : Parallel m n) -- m and n are parallel
variable (h_angle : α = 45) -- α is 45 degrees

-- Define the angle y
def y := SupplementaryAngle α -- y is the supplementary angle to α

-- The theorem stating that y is 135 degrees, given the conditions
theorem measure_angle_y : y.measure = 135 := by
  sorry

end measure_angle_y_l70_70406


namespace perfect_square_expression_l70_70524

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 :=
by
  sorry

end perfect_square_expression_l70_70524


namespace smallest_N_l70_70672

variable (a b c N : ℝ)

theorem smallest_N (h1 : a + b = c) (h2 : a > 0) (h3 : b > 0) :
  ∃ (N : ℝ), (∀ (a b c : ℝ), a + b = c → a > 0 → b > 0 → (a^2 + b^2) / c^2 < N) ∧ N = 1/2 :=
by
  use 1/2
  split
  sorry
  rfl

end smallest_N_l70_70672


namespace unique_real_solution_k_l70_70715

-- Definitions corresponding to problem conditions:
def is_real_solution (a b k : ℤ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (∃ (x y : ℝ), x * x = a - 1 ∧ y * y = b - 1 ∧ x + y = Real.sqrt (a * b + k))

-- Theorem statement:
theorem unique_real_solution_k (k : ℤ) : (∀ a b : ℤ, is_real_solution a b k → (a = 2 ∧ b = 2)) ↔ k = 0 :=
sorry

end unique_real_solution_k_l70_70715


namespace candies_for_50_rubles_l70_70584

theorem candies_for_50_rubles : 
  ∀ (x : ℕ), (45 * x = 45) → (50 / x = 50) := 
by
  intros x h
  sorry

end candies_for_50_rubles_l70_70584


namespace find_f_expression_zeros_of_g_l70_70329

noncomputable def f (x : ℝ) : ℝ := - (1 / 3) * x^3 + a * x + b

axiom a_value : a = 3
axiom f_eq_zero_at_2sqrt3 : f (2 * real.sqrt 3) = 0
axiom f_has_local_min_at_neg_sqrt3: ∀ x, (f' x = - x^2 + a) → f' (-real.sqrt 3) = 0

-- The analytical expression of f(x)
theorem find_f_expression : f = λ x, - (1 / 3) * x^3 + 3 * x + 2 * real.sqrt 3 :=
begin
  sorry
end

-- Definition of g(x)
noncomputable def g (x : ℝ) : ℝ := f x - 2 * real.sqrt 3

-- The number of zeros of g(x) in the interval [-sqrt(3), m)
theorem zeros_of_g (m : ℝ) : 
  if -real.sqrt 3 < m ∧ m ≤ 0 then ∀ x ∈ Ico (-real.sqrt 3) m, g x ≠ 0 ∧ ∀ y ∈ Ico (-real.sqrt 3) m, g y > 0 else
  if 0 < m ∧ m ≤ 3 then ∃! x ∈ Ico (-real.sqrt 3) m, g x = 0 else
  if m > 3 then ∃ x1 x2 ∈ Ico (-real.sqrt 3) m, g x1 = 0 ∧ g x2 = 0 ∧ x1 ≠ x2 
  else false :=
begin
  sorry
end

end find_f_expression_zeros_of_g_l70_70329


namespace odds_against_C_l70_70351

def odds_against_winning (p : ℚ) : ℚ := (1 - p) / p

theorem odds_against_C (pA pB pC : ℚ) (hA : pA = 1 / 3) (hB : pB = 1 / 5) (hC : pC = 7 / 15) :
  odds_against_winning pC = 8 / 7 :=
by
  -- Definitions based on the conditions provided in a)
  have h1 : odds_against_winning (1/3) = 2 := by sorry
  have h2 : odds_against_winning (1/5) = 4 := by sorry

  -- Odds against C
  have h3 : 1 - (pA + pB) = pC := by sorry
  have h4 : pA + pB = 8 / 15 := by sorry

  -- Show that odds against C winning is 8/7
  have h5 : odds_against_winning pC = 8 / 7 := by sorry
  exact h5

end odds_against_C_l70_70351


namespace find_x_l70_70582

theorem find_x (x : ℤ) : 
  3^(x - 2) = 9^3 -> x = 8 :=
by
  intro h
  -- Using sorry to skip the proof as required
  sorry

end find_x_l70_70582


namespace base2_digit_difference_l70_70553

theorem base2_digit_difference :
  let binary_digits (n : ℕ) : ℕ := (nat.log 2 n) + 1 in
  let digit_diff (x y : ℕ) := binary_digits y - binary_digits x in
  digit_diff 300 1500 = 2 :=
by
  sorry

end base2_digit_difference_l70_70553


namespace container_capacity_l70_70770

/-- Given a container where 8 liters is 20% of its capacity, calculate the total capacity of 
    40 such containers filled with water. -/
theorem container_capacity (c : ℝ) (h : 8 = 0.20 * c) : 
    40 * c * 40 = 1600 := 
by
  sorry

end container_capacity_l70_70770


namespace perfect_square_of_expression_l70_70526

theorem perfect_square_of_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 := 
by 
  sorry

end perfect_square_of_expression_l70_70526


namespace swimming_pool_radius_l70_70564

theorem swimming_pool_radius 
  (r : ℝ)
  (h1 : ∀ (r : ℝ), r > 0)
  (h2 : π * (r + 4)^2 - π * r^2 = (11 / 25) * π * r^2) :
  r = 20 := 
sorry

end swimming_pool_radius_l70_70564


namespace quadratic_solutions_l70_70917

-- Define the equation x^2 - 6x + 8 = 0
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

-- Lean statement for the equivalence of solutions
theorem quadratic_solutions : ∀ x : ℝ, quadratic_eq x ↔ x = 2 ∨ x = 4 :=
by
  intro x
  dsimp [quadratic_eq]
  sorry

end quadratic_solutions_l70_70917


namespace ratio_of_big_nosed_gnomes_l70_70991

theorem ratio_of_big_nosed_gnomes
    (total_gnomes : ℕ)
    (red_hatted_gnomes : ℕ)
    (blue_hatted_gnomes : ℕ)
    (blue_big_nosed_gnomes : ℕ)
    (red_small_nosed_gnomes : ℕ)
    (total_big_nosed_gnomes : ℕ)
    (ratio_of_big_nosed_to_total : ℚ) :
    total_gnomes = 28 →
    red_hatted_gnomes = 3 * total_gnomes / 4 →
    blue_hatted_gnomes = total_gnomes - red_hatted_gnomes →
    blue_big_nosed_gnomes = 6 →
    red_small_nosed_gnomes = 13 →
    total_big_nosed_gnomes = blue_big_nosed_gnomes + (red_hatted_gnomes - red_small_nosed_gnomes) →
    ratio_of_big_nosed_to_total = total_big_nosed_gnomes / total_gnomes →
    ratio_of_big_nosed_to_total = 1 / 2 :=
by
  intros h0 h1 h2 h3 h4 h5 h6
  have h7 : red_hatted_gnomes = 21 := by linarith
  have h8 : blue_hatted_gnomes = 7 := by linarith
  have h9 : total_big_nosed_gnomes = 14 := by linarith
  have h10 : ratio_of_big_nosed_to_total = 14 / 28 := by linarith
  rw [h10, h9, h0]
  norm_num
  assumption

end ratio_of_big_nosed_gnomes_l70_70991


namespace ways_to_get_off_the_bus_l70_70132

-- Define the number of passengers and stops
def numPassengers : ℕ := 10
def numStops : ℕ := 5

-- Define the theorem that states the number of ways for passengers to get off
theorem ways_to_get_off_the_bus : (numStops^numPassengers) = 5^10 :=
by sorry

end ways_to_get_off_the_bus_l70_70132


namespace bottle_caps_given_l70_70819

variable (initial_caps : ℕ) (final_caps : ℕ) (caps_given_by_rebecca : ℕ)

theorem bottle_caps_given (h1: initial_caps = 7) (h2: final_caps = 9) : caps_given_by_rebecca = 2 :=
by
  -- The proof will be filled here
  sorry

end bottle_caps_given_l70_70819


namespace max_and_min_of_f_on_0_3_l70_70117

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_and_min_of_f_on_0_3 :
  ∃ (x_max x_min : ℝ), (∀ x ∈ set.Icc (0 : ℝ) 3, f x ≤ f x_max) ∧ (∀ x ∈ set.Icc (0 : ℝ) 3, f x_min ≤ f x) ∧ f 0 = 5 ∧ f 3 = -4 ∧ f 2 = -15 ∧ f x_max = 5 ∧ f x_min = -15 :=
by
  sorry

end max_and_min_of_f_on_0_3_l70_70117


namespace bev_taller_than_ana_l70_70514

-- Define the environment and needed entities
structure PersonGrid (α : Type*) :=
(heights : Array (Array α)) -- Heights of people in a 5x5 grid

variables {α : Type*} [LinearOrder α]

def ana (g : PersonGrid α) : α :=
  let shortest_in_rows := Array.map (Array.minimum') g.heights
  Array.maximum' shortest_in_rows

def bev (g : PersonGrid α) : α :=
  let tallest_in_columns := Array.map (λ j => Array.maximum' (Array.map (![g.heights[i] j]) (Finset.range 5))) (Finset.range 5)
  Array.minimum' tallest_in_columns

theorem bev_taller_than_ana (g : PersonGrid α) (h_diff : ∀ (i₁ i₂ j₁ j₂ : ℕ), g.heights[i₁][j₁] ≠ g.heights[i₂][j₂]) (h25 : g.heights.size = 5 ∧ ∀ i, (g.heights[i]).size = 5)
  (h_ana_ne_bev : ana g ≠ bev g) : bev g > ana g := by {
  sorry
}

end bev_taller_than_ana_l70_70514


namespace f3_is_ideal_function_l70_70008

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + f (-x) = 0

def is_strictly_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

noncomputable def f3 (x : ℝ) : ℝ :=
  if x < 0 then x ^ 2 else -x ^ 2

theorem f3_is_ideal_function : is_odd_function f3 ∧ is_strictly_decreasing f3 := 
  sorry

end f3_is_ideal_function_l70_70008


namespace triangle_area_l70_70027

variable (A B C D : Type)
variable [inner_product_space ℝ A]
variable [inner_product_space ℝ B]
variable [inner_product_space ℝ C]
variable [inner_product_space ℝ D]

variable (AB AC AD BC : ℝ)
variable (h1 : AB = 15)
variable (h2 : AC = 13)
variable (h3 : AD = 12)
variable (h4 : BC = sqrt (AB^2 + AC^2 - 2 * AB * AC * cos angle))
variable (angle : ℝ)

theorem triangle_area (P : ℝ) :
  ∀ (AB AC AD BC : ℝ), AB = 15 ∧ AC = 13 ∧ AD = 12 ∧ 
    BC = sqrt (AB^2 + AC^2 - 2 * AB * AC * cos angle) → P = 84 := by 
  sorry

end triangle_area_l70_70027


namespace solve_quadratic_l70_70912

theorem solve_quadratic :
    ∀ x : ℝ, x^2 - 6*x + 8 = 0 ↔ (x = 2 ∨ x = 4) :=
by
  intros x
  sorry

end solve_quadratic_l70_70912


namespace number_division_l70_70542

theorem number_division (x : ℤ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by 
  sorry

end number_division_l70_70542


namespace fraction_is_5_div_9_l70_70779

-- Define the conditions t = f * (k - 32), t = 35, and k = 95
theorem fraction_is_5_div_9 {f k t : ℚ} (h1 : t = f * (k - 32)) (h2 : t = 35) (h3 : k = 95) : f = 5 / 9 :=
by
  sorry

end fraction_is_5_div_9_l70_70779


namespace least_possible_product_of_primes_gt_50_l70_70154

open Nat

theorem least_possible_product_of_primes_gt_50 : 
  ∃ (p q : ℕ), prime p ∧ prime q ∧ p ≠ q ∧ p > 50 ∧ q > 50 ∧ (p * q = 3127) := 
  by
  exists 53
  exists 59
  repeat { sorry }

end least_possible_product_of_primes_gt_50_l70_70154


namespace Carrie_has_50_dollars_left_l70_70633

/-
Conditions:
1. initial_amount = 91
2. sweater_cost = 24
3. tshirt_cost = 6
4. shoes_cost = 11
-/
def initial_amount : ℕ := 91
def sweater_cost : ℕ := 24
def tshirt_cost : ℕ := 6
def shoes_cost : ℕ := 11

/-
Question:
How much money does Carrie have left?
-/
def total_spent : ℕ := sweater_cost + tshirt_cost + shoes_cost
def money_left : ℕ := initial_amount - total_spent

def proof_statement : Prop := money_left = 50

theorem Carrie_has_50_dollars_left : proof_statement :=
by
  sorry

end Carrie_has_50_dollars_left_l70_70633


namespace greatest_number_of_roses_l70_70468

theorem greatest_number_of_roses (a b : ℕ) (h₁ : a = 680) (h₂ : b = 325) :
  let individual_price := 2.30
  let dozen_price := 36
  let two_dozen_price := 50
  let roses_individual := h₁ / individual_price
  let roses_dozen := h₁ / dozen_price * 12
  let roses_two_dozen := h₁ / two_dozen_price * 24
  max (max roses_individual roses_dozen) roses_two_dozen = b := 
sorry

end greatest_number_of_roses_l70_70468


namespace abs_inequality_solution_l70_70970

theorem abs_inequality_solution (x : ℝ) : |x - 2| < 1 ↔ 1 < x ∧ x < 3 :=
by
  -- the proof would go here
  sorry

end abs_inequality_solution_l70_70970


namespace ribbon_per_gift_l70_70418

theorem ribbon_per_gift
  (total_ribbon : ℕ)
  (number_of_gifts : ℕ)
  (ribbon_left : ℕ)
  (used_ribbon := total_ribbon - ribbon_left)
  (ribbon_per_gift := used_ribbon / number_of_gifts)
  (h_total : total_ribbon = 18)
  (h_gifts : number_of_gifts = 6)
  (h_left : ribbon_left = 6) :
  ribbon_per_gift = 2 := by
  sorry

end ribbon_per_gift_l70_70418


namespace cosine_angle_EF_AC1_l70_70805

-- Definitions and coordinates for the points in the cube.
def point_A : ℝ × ℝ × ℝ := (1, 0, 0)
def point_C1 : ℝ × ℝ × ℝ := (0, 1, 1)
def point_E : ℝ × ℝ × ℝ := (1, 0.5, 0)
def point_F : ℝ × ℝ × ℝ := (0, 1, 0.5)

-- Vector calculations for EF and AC1.
def vector_EF : ℝ × ℝ × ℝ := (0 - 1, 1 - 0.5, 0.5 - 0)
def vector_AC1 : ℝ × ℝ × ℝ := (0 - 1, 1 - 0, 1 - 0)

-- Dot product of the vectors EF and AC1.
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Magnitude of a vector.
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1*v.1 + v.2*v.2 + v.3*v.3)

-- The cosine of the angle between the vectors EF and AC1.
noncomputable def cosine_theta : ℝ :=
  dot_product vector_EF vector_AC1 / (magnitude vector_EF * magnitude vector_AC1)

-- Statement to be proved: the cosine of the angle between EF and AC1 is 2√2/3.
theorem cosine_angle_EF_AC1 :
  cosine_theta = 2 * real.sqrt 2 / 3 :=
by sorry

end cosine_angle_EF_AC1_l70_70805


namespace evaluate_f_diff_l70_70052

def f (x : ℝ) := x^5 + 2*x^3 + 7*x

theorem evaluate_f_diff : f 3 - f (-3) = 636 := by
  sorry

end evaluate_f_diff_l70_70052


namespace boat_speed_l70_70183

theorem boat_speed (b s : ℝ) (h1 : b + s = 7) (h2 : b - s = 5) : b = 6 := 
by
  sorry

end boat_speed_l70_70183


namespace min_pos_int_k_l70_70511

noncomputable def minimum_k (x0 : ℝ) : ℝ := (x0 * (Real.log x0 + 1)) / (x0 - 2)

theorem min_pos_int_k : ∃ k : ℝ, (∀ x0 : ℝ, x0 > 2 → k > minimum_k x0) ∧ k = 5 := 
by
  sorry

end min_pos_int_k_l70_70511


namespace min_colors_needed_l70_70990

def vertices (G : SimpleGraph ℕ) : Finset ℕ := { n : ℕ | G.adj n ≠ ∅ }

theorem min_colors_needed (G : SimpleGraph ℕ) (h_vertex_count : vertices G = 20) 
  (h_max_degree : ∀ v ∈ G.verts, G.degree v ≤ 3) : 
  ∃ k, (k = 4 ∧ ∀ v ∈ G.verts, ∃ f : G.edge → Fin 4, ∀ e₁ e₂ ∈ G.edge_set v, f e₁ ≠ f e₂) :=
by sorry

end min_colors_needed_l70_70990


namespace probability_of_correct_guess_l70_70467

noncomputable def prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def even (n : ℕ) : Prop := n % 2 = 0

def two_digit_integer (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def valid_secret_number (n : ℕ) : Prop :=
  two_digit_integer n ∧
  prime (n / 10) ∧
  even (n % 10) ∧
  n > 80

theorem probability_of_correct_guess : (nat.card { n : ℕ | valid_secret_number n } = 5) →
  ∃ (correct_secret_number : ℕ), valid_secret_number correct_secret_number →
  1 / 5 = 1 / nat.card { n : ℕ | valid_secret_number n } :=
by
  intros h_valid_secret_number_cards
  use 90 -- arbitrary correct number satisfying conditions
  intros h_valid
  rw [h_valid_secret_number_cards]
  norm_num


end probability_of_correct_guess_l70_70467


namespace JQ_equals_20_l70_70458

variables {J K L M P Q R : Point}
variables (parallelogram_JKLM : Parallelogram J K L M)
variables (on_extension_P_of_LM : OnExtension P L M)
variables (JP_intersects_diagonal_KM_at_Q : Intersects (LineThrough J P) (LineThrough K M) Q)
variables (JP_intersects_side_ML_at_R : Intersects (LineThrough J P) (LineThrough M L) R)
variables (QR_length : QR = 40)
variables (RP_length : RP = 30)

theorem JQ_equals_20
  (parallelogram_JKLM : Parallelogram J K L M)
  (on_extension_P_of_LM : OnExtension P L M)
  (JP_intersects_diagonal_KM_at_Q : Intersects (LineThrough J P) (LineThrough K M) Q)
  (JP_intersects_side_ML_at_R : Intersects (LineThrough J P) (LineThrough M L) R)
  (QR_length : QR = 40)
  (RP_length : RP = 30) :
  JQ = 20 :=
sorry

end JQ_equals_20_l70_70458


namespace domain_f_l70_70160

noncomputable def f (x : ℝ) : ℝ := log 2 (log 3 (log 4 (log 5 x)))

theorem domain_f (x : ℝ) : f x = log 2 (log 3 (log 4 (log 5 x))) ∧ x > 625 ↔ x > 625 :=
by sorry

end domain_f_l70_70160


namespace kennedy_lost_pawns_l70_70420

-- Definitions based on conditions
def initial_pawns_per_player := 8
def total_pawns := 2 * initial_pawns_per_player -- Total pawns in the game initially
def pawns_lost_by_Riley := 1 -- Riley lost 1 pawn
def pawns_remaining := 11 -- 11 pawns left in the game

-- Translations of conditions to Lean
theorem kennedy_lost_pawns : 
  initial_pawns_per_player - (pawns_remaining - (initial_pawns_per_player - pawns_lost_by_Riley)) = 4 := 
by 
  sorry

end kennedy_lost_pawns_l70_70420


namespace probability_no_full_favorite_song_l70_70634

def song_lengths (n: ℕ) : ℕ :=
  45 + 15 * n

theorem probability_no_full_favorite_song 
  (n : ℕ)
  (hn : n = 12)
  (favorite_song_length : ℕ)
  (favorite_length_proof : favorite_song_length = 240)
  (total_time : ℕ)
  (total_time_proof : total_time = 300) : 
  (1 - (3 / 132) = 43 / 44) :=
  by
    -- Establish the conditions
    have song_count : ℕ := n,
    have min_song_length : ℕ := song_lengths 0,
    have favorite_length : ℕ := favorite_song_length,
    have time_limit : ℕ := total_time,

    sorry

end probability_no_full_favorite_song_l70_70634


namespace lisa_additional_marbles_l70_70874

theorem lisa_additional_marbles (n : ℕ) (m : ℕ) (s_n : ℕ) :
  n = 12 →
  m = 40 →
  (s_n = (list.sum (list.range (n + 1)))) →
  s_n - m = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp only [list.range_succ, list.sum_range_succ, nat.factorial, nat.succ_eq_add_one, nat.add_succ, mul_add, mul_one, mul_comm n]
  sorry

end lisa_additional_marbles_l70_70874


namespace arithmetic_sequence_a5_l70_70801

variable {α : Type} [AddGroup α]

theorem arithmetic_sequence_a5 (a : ℕ → α) (h : a 2 + a 8 = 15 - a 5) : a 5 = 5 := 
sorry

end arithmetic_sequence_a5_l70_70801


namespace problem_statement_l70_70460

-- Definitions for propositions p and q
def proposition_p (a b : ℝ) : Prop := ab = 0 → a = 0
def proposition_q : Prop := 3 ≥ 3

-- Proof statement
theorem problem_statement (a b : ℝ) : ¬ (proposition_p a b) ∧ proposition_q :=
by
  sorry

end problem_statement_l70_70460


namespace seq_contains_all_positive_integers_l70_70968

-- Define the sequence {a_n}
def seq (a : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := Nat.find (λ m, m > 0 ∧ Nat.coprime m (Finset.range(n+1).sum seq) ∧ m ∉ (Finset.range(n+1).image seq))

-- Prove that every positive integer appears in the sequence {a_n}
theorem seq_contains_all_positive_integers (a : ℕ) (h : a > 0) :
  ∀ (m : ℕ), m > 0 → ∃ (n : ℕ), seq a n = m := 
by
  sorry

end seq_contains_all_positive_integers_l70_70968


namespace unique_solution_of_equation_l70_70256

theorem unique_solution_of_equation :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ x^2 + y^2 = x^y :=
by
  use [2, 2]
  split
  -- omitted the proof
  sorry

end unique_solution_of_equation_l70_70256


namespace problem_opposite_sqrt2_problem_reciprocal_sqrt2_problem_cube_root_neg_1_over_8_l70_70120

-- Definition for the opposite of a number
def opposite (a : ℝ) := -a

-- Definition for the reciprocal of a number
def reciprocal (a : ℝ) := 1 / a

-- Definition for the cube root of a number
def cube_root (a : ℝ) := a^(1 / 3 : ℝ)

theorem problem_opposite_sqrt2 : opposite (-real.sqrt 2) = real.sqrt 2 := 
sorry

theorem problem_reciprocal_sqrt2 : reciprocal (real.sqrt 2) = real.sqrt 2 / 2 :=
sorry

theorem problem_cube_root_neg_1_over_8 : cube_root (-1 / 8) = -1 / 2 :=
sorry

end problem_opposite_sqrt2_problem_reciprocal_sqrt2_problem_cube_root_neg_1_over_8_l70_70120


namespace ellipse_area_l70_70277

theorem ellipse_area {x y : ℝ} 
  (h : 4 * x^2 + 2 * x + y^2 + 4 * y + 5 = 0) 
  : area_of_ellipse 4 2 1 4 5 = (17 * Real.pi) / 32 :=  
sorry

end ellipse_area_l70_70277


namespace minimum_volume_bounded_by_tangent_plane_l70_70705

noncomputable def smallest_volume (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) : ℝ :=
  let ellipsoid (x y z : ℝ) := x^2 / a^2 + y^2 / b^2 + z^2 / c^2
  let tangent_plane_volume (x0 y0 z0 : ℝ) := a^2 * b^2 * c^2 / (6 * x0 * y0 * z0)
  tangent_plane_volume (a / real.sqrt 3) (b / real.sqrt 3) (c / real.sqrt 3)

theorem minimum_volume_bounded_by_tangent_plane 
  {a b c : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) : 
  smallest_volume a b c h_pos_a h_pos_b h_pos_c = (real.sqrt 3 / 2) * a * b * c :=
sorry

end minimum_volume_bounded_by_tangent_plane_l70_70705


namespace range_of_m_l70_70331

def f (x : ℝ) : ℝ :=
if x ≤ 0 then 12 * x - x^3 else -2 * x

definition range_of_f_for_m (m : ℝ) : set ℝ :=
{y : ℝ | ∃ x : ℝ, x ≤ m ∧ f x = y}

theorem range_of_m (m : ℝ) : range_of_f_for_m m = {y : ℝ | -16 ≤ y} ↔ -2 ≤ m ∧ m ≤ 8 := by
  sorry

end range_of_m_l70_70331


namespace find_divisor_l70_70989

-- Definitions based on the conditions
def is_divisor (d : ℕ) (a b k : ℕ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ (b - a) / n = k ∧ k = d

-- Problem statement
theorem find_divisor (a b k : ℕ) (H : b = 43 ∧ a = 10 ∧ k = 11) : ∃ d, d = 3 :=
by
  sorry

end find_divisor_l70_70989


namespace min_balls_left_unboxed_l70_70177

theorem min_balls_left_unboxed (balls : ℕ) (big_box_capacity : ℕ) (small_box_capacity : ℕ) :
  balls = 104 → big_box_capacity = 25 → small_box_capacity = 20 →
  ∃ big_boxes small_boxes unboxed_balls, 
    unboxed_balls = balls - (big_boxes * big_box_capacity + small_boxes * small_box_capacity) ∧
    unboxed_balls < big_box_capacity ∧ 
    unboxed_balls < small_box_capacity ∧
    unboxed_balls = 4 :=
begin
  intros hballs hbig hsmall,
  sorry
end

end min_balls_left_unboxed_l70_70177


namespace increasing_interval_range_of_a_l70_70749

noncomputable def f (x : ℝ) := (x^2 - x + 1) / Real.exp x

theorem increasing_interval :
  {x : ℝ | 1 < x ∧ x < 2} = {x : ℝ | f' x > 0} :=
begin
  sorry
end

theorem range_of_a (a : ℝ) (h : ∀ x > 0, Real.exp x * f x ≥ a + Real.log x) :
  a ≤ 1 :=
begin
  sorry
end

end increasing_interval_range_of_a_l70_70749


namespace circumscribed_sphere_surface_area_of_triangular_pyramid_l70_70797

theorem circumscribed_sphere_surface_area_of_triangular_pyramid
  (PC AC AB : ℝ)
  (hPC : PC = 3)
  (hAC : AC = 4)
  (hAB : AB = 5)
  (h_perp : ∃ (P A B C : Type), PC ⊥ span ℝ {0 : ℝ})
  (h_right_angle : ∃ (A B C : Type), angle A B C = π / 2  ):
  let d := Real.sqrt (PC^2 + AC^2 + AB^2) in
  let R := d / 2 in
  let A := 4 * Real.pi * R^2 in
  A = 50 * Real.pi := by
  sorry

end circumscribed_sphere_surface_area_of_triangular_pyramid_l70_70797


namespace probability_condition_l70_70172

namespace SharedPowerBank

def P (event : String) : ℚ :=
  match event with
  | "A" => 3 / 4
  | "B" => 1 / 2
  | _   => 0 -- Default case for any other event

def probability_greater_than_1000_given_greater_than_500 : ℚ :=
  P "B" / P "A"

theorem probability_condition :
  probability_greater_than_1000_given_greater_than_500 = 2 / 3 :=
by 
  sorry

end SharedPowerBank

end probability_condition_l70_70172


namespace domain_of_g_l70_70671

theorem domain_of_g :
  {x : ℝ | -6*x^2 - 7*x + 8 >= 0} = 
  {x : ℝ | (7 - Real.sqrt 241) / 12 ≤ x ∧ x ≤ (7 + Real.sqrt 241) / 12} :=
by
  sorry

end domain_of_g_l70_70671


namespace solution_to_equation_l70_70907

noncomputable def equation (x : ℝ) : ℝ := 
  (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (5 - 9 * x) / (x - 2) + 2

theorem solution_to_equation :
  equation 3.294 = 0 ∧ equation (-0.405) = 0 :=
by
  sorry

end solution_to_equation_l70_70907


namespace quadrilateral_all_sides_equal_l70_70422

-- Definition of our geometrical setting
variable (A B C D E : Type)

-- Definition of required conditions
variable (AB BE AC CE AE: A → B → C → D → E)
variable (eq_AB_BE : AB = BE)

-- The theorem to be proved
theorem quadrilateral_all_sides_equal (H : ∀ P: Type, angle P C E = 90) : 
  (∀ s: Type, AB s = s BE → AB s = s ) := sorry

end quadrilateral_all_sides_equal_l70_70422


namespace coeff_x_squared_l70_70282

def P (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 5
def Q (x : ℝ) : ℝ := 6 * x^2 + 7 * x + 8

theorem coeff_x_squared :
  let coeff : ℝ := 82 in
  ∀ x : ℝ, (P x * Q x).coeff 2 = coeff :=
sorry

end coeff_x_squared_l70_70282


namespace perimeter_to_side_ratio_l70_70465

variable (a b c h_a r : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < h_a ∧ 0 < r ∧ a + b > c ∧ a + c > b ∧ b + c > a)

theorem perimeter_to_side_ratio (P : ℝ) (hP : P = a + b + c) :
  P / a = h_a / r := by
  sorry

end perimeter_to_side_ratio_l70_70465


namespace tom_room_area_l70_70137

theorem tom_room_area
  (cost_removal : ℝ)
  (cost_per_sqft : ℝ)
  (total_cost : ℝ)
  (h1 : cost_removal = 50)
  (h2 : cost_per_sqft = 1.25)
  (h3 : total_cost = 120) :
  let cost_new_floor := total_cost - cost_removal in
  let area := cost_new_floor / cost_per_sqft in
  area = 56 :=
by {
  have h_cost_new_floor : cost_new_floor = 70, from calc
      cost_new_floor = total_cost - cost_removal : rfl
                   ... = 120 - 50 : by rw [h3, h1]
                   ... = 70 : by norm_num,
  have h_area : area = 70 / 1.25, from calc
      area = cost_new_floor / cost_per_sqft : rfl
           ... = 70 / 1.25 : by rw [h_cost_new_floor, h2],
  norm_num at h_area,
  exact h_area
}

end tom_room_area_l70_70137


namespace coffee_vacation_days_l70_70471

theorem coffee_vacation_days 
  (pods_per_day : ℕ := 3)
  (pods_per_box : ℕ := 30)
  (box_cost : ℝ := 8.00)
  (total_spent : ℝ := 32) :
  (total_spent / box_cost) * pods_per_box / pods_per_day = 40 := 
by 
  sorry

end coffee_vacation_days_l70_70471


namespace solve_x_l70_70767

theorem solve_x (x : ℝ) (hx : (1/x + 1/(2*x) + 1/(3*x) = 1/12)) : x = 22 :=
  sorry

end solve_x_l70_70767


namespace height_of_flagpole_l70_70624

variable (x : ℝ) -- height of the flagpole
variable (tree_shadow flagpole_shadow tree_height : ℝ) -- lengths and height

axiom condition1 : tree_shadow = 0.6
axiom condition2 : flagpole_shadow = 1.5
axiom condition3 : tree_height = 3.6

theorem height_of_flagpole : x = 9 := by
  have h1 : 3.6 / 0.6 = x / 1.5 := by
    rw [condition3, condition1]
    exact (proportional _ _ _ _) -- placeholder for actual proportion proof
  have h2 : 3.6 / 0.6 = 6 := by norm_num
  have h3 : 6 * 1.5 = 9 := by norm_num
  rw [h2] at h1
  simp at h1
  have h4 : 0.6 * x = 5.4 := by calc
    3.6 * 1.5 = 5.4 : by norm_num
  have h5 : x = 9 := by linarith
  exact h5

end height_of_flagpole_l70_70624


namespace unique_solution_l70_70665

-- Define the condition that p, q, r are prime numbers
variables {p q r : ℕ}

-- Assume that p, q, and r are primes
axiom p_prime : Prime p
axiom q_prime : Prime q
axiom r_prime : Prime r

-- Define the main theorem to state that the only solution is (7, 3, 2)
theorem unique_solution : p + q^2 = r^4 → (p, q, r) = (7, 3, 2) :=
by
  sorry

end unique_solution_l70_70665


namespace find_pqr_abs_l70_70838

variables {p q r : ℝ}

-- Conditions as hypotheses
def conditions (p q r : ℝ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
  (p^2 + 2/q = q^2 + 2/r) ∧ (q^2 + 2/r = r^2 + 2/p)

-- Statement of the theorem
theorem find_pqr_abs (h : conditions p q r) : |p * q * r| = 2 :=
sorry

end find_pqr_abs_l70_70838


namespace polynomial_existence_and_uniqueness_l70_70703

theorem polynomial_existence_and_uniqueness :
  ∃ q : polynomial ℝ, (q.comp q = X * q + 2 * X^2) ∧ (q = -2 * X + 4) :=
by
  sorry

end polynomial_existence_and_uniqueness_l70_70703


namespace find_top_angle_l70_70980

theorem find_top_angle 
  (sum_of_angles : ∀ (α β γ : ℝ), α + β + γ = 250) 
  (left_is_twice_right : ∀ (α β : ℝ), α = 2 * β) 
  (right_angle_is_60 : ∀ (β : ℝ), β = 60) :
  ∃ γ : ℝ, γ = 70 :=
by
  -- Assume the variables for the angles
  obtain ⟨α, β, γ, h_sum, h_left, h_right⟩ := ⟨_, _, _, sum_of_angles, left_is_twice_right, right_angle_is_60⟩
  -- Your proof here
  sorry

end find_top_angle_l70_70980


namespace count_palindromes_300_to_800_l70_70255

/-- A three-digit integer palindrome between 300 and 800 is of the form aba, 
    where a and b are digits and 3 ≤ a ≤ 7. There are 50 such palindromes. -/
theorem count_palindromes_300_to_800 : 
  let is_palindrome (n : ℕ) := 
        ∃ a b, n = 100 * a + 10 * b + a ∧ (3 ≤ a ∧ a ≤ 7) 
  ∧ 300 ≤ n ∧ n ≤ 800 
  in
  fintype.card {n // is_palindrome n} = 50 := 
by sorry

end count_palindromes_300_to_800_l70_70255


namespace Jason_scores_31_points_l70_70792

theorem Jason_scores_31_points :
  ∀ (x y z : ℕ),
    (x + y + z = 40) → 
    (z = 10) → 
    (0.75 * x + 0.8 * y + 8 = 31) :=
by
  intros x y z h1 h2
  sorry

end Jason_scores_31_points_l70_70792


namespace speed_in_still_water_l70_70176

-- Define variables for speed of the boy in still water and speed of the stream.
variables (v s : ℝ)

-- Define the conditions as Lean statements
def downstream_condition (v s : ℝ) : Prop := (v + s) * 7 = 91
def upstream_condition (v s : ℝ) : Prop := (v - s) * 7 = 21

-- The theorem to prove that the speed of the boy in still water is 8 km/h given the conditions
theorem speed_in_still_water
  (h1 : downstream_condition v s)
  (h2 : upstream_condition v s) :
  v = 8 := 
sorry

end speed_in_still_water_l70_70176


namespace base_number_is_two_l70_70010

theorem base_number_is_two (x n : ℝ) (b : ℝ) (h1 : n = x ^ 0.25) (h2 : n ^ b = 8) (h3 : b = 12.000000000000002) : 
  x = 2 :=
sorry

end base_number_is_two_l70_70010


namespace problem_statement_l70_70510

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | |x| ≥ 1}
def B : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def C_U (S : Set ℝ) : Set ℝ := {x | x ∉ S}

theorem problem_statement : (C_U A) ∩ (C_U B) = {x | -1 < x ∧ x < 1} :=
by {
  sorry,
}

end problem_statement_l70_70510


namespace least_possible_product_of_distinct_primes_gt_50_l70_70141

-- Define a predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the conditions: two distinct primes greater than 50
def distinct_primes_gt_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50 ∧ p ≠ q

-- The least possible product of two distinct primes each greater than 50
theorem least_possible_product_of_distinct_primes_gt_50 :
  ∃ p q : ℕ, distinct_primes_gt_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_possible_product_of_distinct_primes_gt_50_l70_70141


namespace complement_P_l70_70754

def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x^2 ≤ 1}

theorem complement_P :
  (U \ P) = Set.Iio (-1) ∪ Set.Ioi (1) :=
by
  sorry

end complement_P_l70_70754


namespace money_increase_factor_two_years_l70_70626

theorem money_increase_factor_two_years (P : ℝ) (rate : ℝ) (n : ℕ)
  (h_rate : rate = 0.50) (h_n : n = 2) :
  (P * (1 + rate) ^ n) = 2.25 * P :=
by
  -- proof goes here
  sorry

end money_increase_factor_two_years_l70_70626


namespace quadratic_function_expression_l70_70325

theorem quadratic_function_expression : 
  ∃ (a : ℝ), (a ≠ 0) ∧ (∀ x : ℝ, x = -1 → x * (a * (x + 1) * (x - 2)) = 0) ∧
  (∀ x : ℝ, x = 2 → x * (a * (x + 1) * (x - 2)) = 0) ∧
  (∀ y : ℝ, ∃ x : ℝ, x = 0 ∧ y = -2 → y = a * (x + 1) * (x - 2)) 
  → (∀ x : ℝ, ∃ y : ℝ, y = x^2 - x - 2) := 
sorry

end quadratic_function_expression_l70_70325


namespace correct_statements_count_l70_70487

-- Definitions for each of the conditions provided in the problem
def statement_1 (l1 l2 l3 : Line) : Prop :=
  (number_of_intersections l1 l2 l3 = 2) → are_parallel l1 l2 ∨ are_parallel l2 l3 ∨ are_parallel l1 l3

def statement_2 (l1 l2 l3 : Line) : Prop :=
  (are_parallel l1 l2 ∧ have_equal_interior_angles l1 l2 l3) → (are_perpendicular l1 l3 ∧ are_perpendicular l2 l3)

def statement_3 (p : Point) (l : Line) : Prop :=
  ∃! (m : Line), passes_through m p ∧ are_parallel m l

def statement_4 (t1 t2 : Triangle) : Prop :=
  (is_translation_of t1 t2) → (all_corresponding_segments_parallel t1 t2)

-- The Theorem that we want to prove
theorem correct_statements_count : 
  ∀ (l1 l2 l3 : Line) (p : Point) (t1 t2 : Triangle), 
  statement_1 l1 l2 l3 ∧ statement_2 l1 l2 l3 ∧ statement_3 p l1 ∧ statement_4 t1 t2 → number_of_correct_statements = 2 :=
sorry

end correct_statements_count_l70_70487


namespace arithmetic_geometric_is_constant_l70_70769

theorem arithmetic_geometric_is_constant (a : ℕ → ℝ) :
  (∃ d : ℝ, ∀ n ≥ 2, a n - a (n - 1) = d) ∧ (∃ q : ℝ, ∀ n ≥ 2, a n = q * a (n - 1)) →
  ∃ b : ℝ, b ≠ 0 ∧ ∀ n : ℕ, a n = b :=
begin
  sorry
end

end arithmetic_geometric_is_constant_l70_70769


namespace mean_of_second_set_is_76_point_4_l70_70382

theorem mean_of_second_set_is_76_point_4
    (x : ℕ)
    (h : (28 + x + 50 + 78 + 104) / 5 = 62) :
    (48 + 62 + 98 + 124 + x) / 5 = 76.4 := 
sorry

end mean_of_second_set_is_76_point_4_l70_70382


namespace buoy_distance_proof_l70_70628

def distance_from_first_to_next_buoy (distance_first_buoy distance_next_buoy : ℕ) : ℕ :=
  distance_next_buoy - distance_first_buoy

theorem buoy_distance_proof  : distance_from_first_to_next_buoy 72 96 = 24 := 
by
  simp [distance_from_first_to_next_buoy]
  sorry

end buoy_distance_proof_l70_70628


namespace find_a1_plus_a9_l70_70802

variable (a : ℕ → ℝ) (d : ℝ)

-- condition: arithmetic sequence
def is_arithmetic_seq : Prop := ∀ n, a (n + 1) = a n + d

-- condition: sum of specific terms
def sum_specific_terms : Prop := a 3 + a 4 + a 5 + a 6 + a 7 = 450

-- theorem: prove the desired sum
theorem find_a1_plus_a9 (h1 : is_arithmetic_seq a d) (h2 : sum_specific_terms a) : 
  a 1 + a 9 = 180 :=
  sorry

end find_a1_plus_a9_l70_70802


namespace cigarette_boxes_surface_area_l70_70454

def smallest_area_of_packaging (a b c : ℕ) (h₁ : a > b) (h₂ : b > c) : ℕ :=
  min (2 * (10 * a * b + b * c + 10 * a * c))
      (min (2 * (a * 10 * b + 10 * b * c + a * c))
           (min (2 * (a * b + b * c + 10 * a * c)) -- Fixing the formula inconsistency.
                (2 * (2 * a * b + 5 * b * c + 10 * a * c))))

def cigarette_boxes := ∀ a b c : ℕ, a = 88 → b = 58 → c = 22 → (a > b) → (b > c) → (smallest_area_of_packaging a b c (by norm_num) (by norm_num) = 65296)

-- Proof not provided for the theorem
theorem cigarette_boxes_surface_area : cigarette_boxes := sorry

end cigarette_boxes_surface_area_l70_70454


namespace number_is_2250_l70_70537

-- Question: Prove that x = 2250 given the condition.
theorem number_is_2250 (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
sorry

end number_is_2250_l70_70537


namespace volume_of_pyramid_l70_70219

theorem volume_of_pyramid {O A B C : Type} (r : ℝ) (AN NB AM MC : ℝ)
  (h1 : r = sqrt 5) 
  (h2 : AN = NB) (h3 : AN = 1) 
  (h4 : AM = 1) (h5 : MC = 2 * AM)
  (h6 : AM + MC = AC) : 
  volume O A B C = 2 := 
sorry

end volume_of_pyramid_l70_70219


namespace least_possible_product_of_distinct_primes_gt_50_l70_70142

-- Define a predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the conditions: two distinct primes greater than 50
def distinct_primes_gt_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50 ∧ p ≠ q

-- The least possible product of two distinct primes each greater than 50
theorem least_possible_product_of_distinct_primes_gt_50 :
  ∃ p q : ℕ, distinct_primes_gt_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_possible_product_of_distinct_primes_gt_50_l70_70142


namespace amazon_tide_problem_l70_70925

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := sin (2 * x + φ)

theorem amazon_tide_problem (φ : ℝ) (h1 : |φ| < π / 2) 
  (h2 : ∀ x, f x φ = f (-x - π /3) φ) :
  φ ≠ π / 3 ∧ (f (5 * π / 12) φ = 1) ∧ 
    (∀ x, -π / 3 ≤ x ∧ x ≤ -π / 6 → monotone f x φ) ∧ 
    (∃ x, 0 < x ∧ x < π / 2 ∧ critical_point f x φ) :=
begin
  sorry
end

end amazon_tide_problem_l70_70925


namespace hexagon_area_proof_final_result_l70_70046

noncomputable def find_hexagon_area (b : ℝ) : ℝ :=
  -- Definitions of points A and B
  let A := (0 : ℝ, 0 : ℝ) in
  let B := (b, 1 : ℝ) in
  -- Define additional points and complex numbers after rotations
  let F := (-b * (Real.sqrt 3 / 2) + 1 / 2, b / 2 + Real.sqrt 3 / 2 + 3) in
  let other_points := sorry in -- Other points defined similarly with given properties
  
  -- Compute area using vertices and properties
  let area := 13 / Real.sqrt 3 in -- Example calculation leading to the answer
  
  -- Return the area
  13 * Real.sqrt 3

theorem hexagon_area_proof (b : ℝ) : 
  find_hexagon_area b = 13 * Real.sqrt 3 := 
by
  sorry

theorem final_result : 
  13 + 3 = 16 :=
by 
  rfl


end hexagon_area_proof_final_result_l70_70046


namespace grace_clyde_ratio_l70_70244

theorem grace_clyde_ratio (C G : ℕ) (h1 : G = C + 35) (h2 : G = 40) : G / C = 8 :=
by sorry

end grace_clyde_ratio_l70_70244


namespace prime_solution_unique_l70_70670

theorem prime_solution_unique:
  ∃ p q r : ℕ, (prime p ∧ prime q ∧ prime r ∧ p + q^2 = r^4 ∧ p = 7 ∧ q = 3 ∧ r = 2) ∧
  (∀ p' q' r' : ℕ, prime p' ∧ prime q' ∧ prime r' ∧ p' + q'^2 = r'^4 → (p' = 7 ∧ q' = 3 ∧ r' = 2)) :=
by {
  sorry
}

end prime_solution_unique_l70_70670


namespace find_years_lent_to_B_l70_70596

def principal_B := 5000
def principal_C := 3000
def rate := 8
def time_C := 4
def total_interest := 1760

-- Interest calculation for B
def interest_B (n : ℕ) := (principal_B * rate * n) / 100

-- Interest calculation for C (constant time of 4 years)
def interest_C := (principal_C * rate * time_C) / 100

-- Total interest received
def total_interest_received (n : ℕ) := interest_B n + interest_C

theorem find_years_lent_to_B (n : ℕ) (h : total_interest_received n = total_interest) : n = 2 :=
by
  sorry

end find_years_lent_to_B_l70_70596


namespace negative_two_pow_zero_negative_three_pow_neg_three_l70_70241

theorem negative_two_pow_zero : (-2 : ℤ)^0 = 1 := by
  sorry

theorem negative_three_pow_neg_three : (-3 : ℤ) ^ -3 = -(1 / 27 : ℚ) := by
  sorry

end negative_two_pow_zero_negative_three_pow_neg_three_l70_70241


namespace probability_first_two_heads_l70_70812

theorem probability_first_two_heads (flips : list char) 
  (h_length : flips.length = 5)
  (h_heads : list.count 'H' flips = 3) :
  (∃ l1 l2, flips = l1 ++ ['H', 'H'] ++ l2) → 
  (list.count 'H' (['H', 'H'] ++ l2) = 3 → l2.length = 3 → list.count 'H' l2 = 1) →
  (3/10 = ((list.permutations flips).count (λ l, (list.take 2 l = ['H', 'H'])).to_nat) / (list.permutations flips).length.to_nat) :=
sorry

end probability_first_two_heads_l70_70812


namespace number_division_l70_70544

theorem number_division (x : ℤ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by 
  sorry

end number_division_l70_70544


namespace find_100m_plus_n_l70_70235

def expected_digits_prob_conditions (append_digits_from : ℕ → Bool) : Prop :=
  ∀ d, d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ↔ append_digits_from d

def E_def (E : ℚ) (E_A : ℚ) (E_B : ℚ) : Prop :=
  E = E_B ∧
  E_A = 1 + 8 / 9 * E_B ∧
  E_B = 1 + 2 / 9 * E_A + 7 / 9 * E_B

noncomputable def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem find_100m_plus_n (E : ℚ) (m n : ℕ) (append_digits_from : ℕ → Bool) :
  expected_digits_prob_conditions append_digits_from →
  E_def E (81 : ℚ) (171 / 2) →
  relatively_prime 171 2 →
  m = 171 →
  n = 2 →
  100 * m + n = 17102 := 
begin
  intros cond E_equ gcd_rel m_def n_def,
  calc
    100 * m + n = 100 * 171 + 2 : by rw [m_def, n_def]
             ... = 17102        : by norm_num,
end

end find_100m_plus_n_l70_70235


namespace AKS_time_complexity_l70_70474

-- Definitions representing the given conditions
def loop1_complexity (n : ℕ) := O(log n ^ 6)
def loop2_complexity (n : ℕ) := O(log n ^ 13)
def loop3_complexity (n : ℕ) := O(log n ^ 20)

-- The theorem that corresponds to the given mathematical statement
theorem AKS_time_complexity (n : ℕ) :
  (loop1_complexity n + loop2_complexity n + loop3_complexity n) = O(log n ^ 20) :=
sorry

end AKS_time_complexity_l70_70474


namespace max_bishops_ways_chessboard_l70_70437

theorem max_bishops_ways_chessboard (n k : ℕ) 
  (hn : n = 10) 
  (hk : k = 64): 
  n + k = 74 := 
by 
  rw [hn, hk]
  rfl

end max_bishops_ways_chessboard_l70_70437


namespace no_solution_fraction_equation_l70_70919

theorem no_solution_fraction_equation (x : ℝ) (h : x ≠ 2) : 
  (1 - x) / (x - 2) + 2 = 1 / (2 - x) → false :=
by 
  intro h_eq
  sorry

end no_solution_fraction_equation_l70_70919


namespace vasya_can_place_99_tokens_l70_70083

theorem vasya_can_place_99_tokens (board : ℕ → ℕ → Prop) (h : ∀ i j, board i j → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50) :
  ∃ new_tokens : ℕ → ℕ → Prop, (∀ i j, new_tokens i j → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50) ∧
  (∀ i, (∑ j in finset.range 50, if board i j ∨ new_tokens i j then 1 else 0) % 2 = 0) ∧
  (∀ j, (∑ i in finset.range 50, if board i j ∨ new_tokens i j then 1 else 0) % 2 = 0) ∧
  (finset.sum (finset.product (finset.range 50) (finset.range 50)) 
             (λ (ij : ℕ × ℕ), if new_tokens ij.1 ij.2 then 1 else 0) ≤ 99) :=
sorry

end vasya_can_place_99_tokens_l70_70083


namespace equal_tuesdays_thursdays_in_30_day_month_l70_70791

theorem equal_tuesdays_thursdays_in_30_day_month : 
  ∃ (days : Finset ℕ), 
  days.card = 3 ∧ 
  ∀ (start_day ∈ days), 
    let num_tuesdays := 4 + if start_day ∈ {1, 2} then 1 else 0,
        num_thursdays := 4 + if start_day ∈ {3, 4} then 1 else 0
    in num_tuesdays = num_thursdays :=
begin
  sorry
end

end equal_tuesdays_thursdays_in_30_day_month_l70_70791


namespace time_to_buy_coffee_bagel_is_15_l70_70687

-- Define the given conditions within Lean
def time_to_buy_coffee_bagel (x : ℕ) : Prop :=
  let read_and_eat_time := 2 * x in
  x + read_and_eat_time = 45

-- State the proof problem
theorem time_to_buy_coffee_bagel_is_15 (x : ℕ) (h : time_to_buy_coffee_bagel x) : x = 15 :=
by
  sorry

end time_to_buy_coffee_bagel_is_15_l70_70687


namespace increasing_interval_range_of_a_l70_70750

noncomputable def f (x : ℝ) := (x^2 - x + 1) / Real.exp x

theorem increasing_interval :
  {x : ℝ | 1 < x ∧ x < 2} = {x : ℝ | f' x > 0} :=
begin
  sorry
end

theorem range_of_a (a : ℝ) (h : ∀ x > 0, Real.exp x * f x ≥ a + Real.log x) :
  a ≤ 1 :=
begin
  sorry
end

end increasing_interval_range_of_a_l70_70750


namespace range_of_a_l70_70856

open Classical

variable (a : ℝ) (p : Prop) (q : Prop)

def proposition_p := ∀ x : ℝ, (0 < a ∧ a ≠ 1) → (a ^ x > 1 ↔ x < 0)
def proposition_q := ∀ x : ℝ, (x^2 - x + a > 0)

theorem range_of_a (h1 : proposition_p p) (h2 : proposition_q q) 
  (h3 : p ∨ q) (h4 : ¬(p ∧ q)) : a ∈ (Set.Ioc 0 (1/4)) ∪ Set.Ioi 1 := 
sorry

end range_of_a_l70_70856


namespace initial_number_of_poles_l70_70512

theorem initial_number_of_poles (n : ℕ) (h1 : 5000 / (n - 1) + 1.25 = 5000 / (n - 2)) : n = 65 :=
by
  sorry

end initial_number_of_poles_l70_70512


namespace proposition_correctness_l70_70902

theorem proposition_correctness :
  (¬ (∃ x : ℝ, log a (x^2 + 1) > 3) ↔ ∀ x : ℝ, log a (x^2 + 1) ≤ 3) ∧
  ((p ∨ q) = false ↔ (¬ p ∧ ¬ q) = true) ∧
  ((a > 2 → a > 5) = false) ∧
  (∀ f : ℝ → ℝ, (∀ x : ℝ, f (-x) = f x) → f = λ x, (x + 1) * (x - 1)) :=
by {
  sorry,
}

end proposition_correctness_l70_70902


namespace weight_difference_tote_laptop_l70_70823

variable (T B L : ℝ)

-- Given conditions
def weight_tote := (T = 8)
def empty_briefcase := (T = 2 * B)
def full_briefcase := (2 * T = 2 * (T))
def weight_papers := (1 / 6 * (2 * T))

theorem weight_difference_tote_laptop (h1 : weight_tote) (h2 : empty_briefcase) (h3 : full_briefcase) (h4 : weight_papers) :
  L - T = 1.33 :=
by
  sorry

end weight_difference_tote_laptop_l70_70823


namespace inequality_proof_l70_70906

theorem inequality_proof (a b c : ℝ) (h : a * b * c = 1) : 
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 := 
by
  sorry

end inequality_proof_l70_70906


namespace top_angle_is_70_l70_70983

theorem top_angle_is_70
  (sum_angles : ℝ)
  (left_angle : ℝ)
  (right_angle : ℝ)
  (top_angle : ℝ)
  (h1 : sum_angles = 250)
  (h2 : left_angle = 2 * right_angle)
  (h3 : right_angle = 60)
  (h4 : sum_angles = left_angle + right_angle + top_angle) :
  top_angle = 70 :=
by
  sorry

end top_angle_is_70_l70_70983


namespace math_problem_l70_70581

theorem math_problem : 12 - (- 18) + (- 7) - 15 = 8 :=
by
  sorry

end math_problem_l70_70581


namespace max_visible_sum_of_stacked_cubes_l70_70297

def cube_numbers : Set ℕ := {1, 3, 9, 27, 81, 243}

def max_visible_sum_stacked_cubes (cube_numbers : Set ℕ) (num_cubes : ℕ) : ℕ :=
  if cube_numbers = {1, 3, 9, 27, 81, 243} ∧ num_cubes = 4 then 1446 else 0

theorem max_visible_sum_of_stacked_cubes : max_visible_sum_stacked_cubes cube_numbers 4 = 1446 :=
  by sorry

end max_visible_sum_of_stacked_cubes_l70_70297


namespace perfect_square_l70_70532

variable (n k l : ℕ)

theorem perfect_square (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m ^ 2 := by
  have h1 : (2 * l - n - k) * (2 * l - n + k) = (2 * l - n) ^ 2 - k ^ 2 := by sorry
  have h2 : k ^ 2 = 2 * l ^ 2 - n ^ 2 := by sorry
  have h3 : (2 * l - n) ^ 2 - (2 * l ^ 2 - n ^ 2) = 2 * (l - n) ^ 2 := by sorry
  have h4 : (2 * (l - n) ^ 2) / 2 = (l - n) ^ 2 := by sorry
  use (l - n)
  rw [h4]
  sorry

end perfect_square_l70_70532


namespace tangent_line_at_x1_l70_70751

noncomputable def g (x : ℝ) : ℝ := (-x^2 + 5 * x - 3) * Real.exp x

noncomputable def g' (x : ℝ) : ℝ := 
  let term1 := (-2 * x + 5) * Real.exp x
  let term2 := (-x^2 + 5 * x - 3) * Real.exp x
  term1 + term2

theorem tangent_line_at_x1 : 
  let m := g' 1
  let y1 := g 1
  (m = 4 * Real.exp 1) ∧ (y1 = Real.exp 1) → 
  ∀ x y : ℝ, y = 4 * Real.exp 1 * (x - 1) + Real.exp 1 ↔ 4 * Real.exp 1 * x - y - 3 * Real.exp 1 = 0 :=
by
  let m := g' 1
  let y1 := g 1
  have h1 : m = 4 * Real.exp 1 := sorry
  have h2 : y1 = Real.exp 1 := sorry
  intro cond
  cases cond with h1 h2
  intro x y
  split
  case mp => intro hy; sorry -- proving the forward direction
  case mpr => intro heq; sorry -- proving the reverse direction

end tangent_line_at_x1_l70_70751


namespace line_AB_eq_x_plus_3y_zero_l70_70110

/-- 
Consider two circles defined by:
C1: x^2 + y^2 - 4x + 6y = 0
C2: x^2 + y^2 - 6x = 0

Prove that the equation of the line through the intersection points of these two circles (line AB)
is x + 3y = 0.
-/
theorem line_AB_eq_x_plus_3y_zero (x y : ℝ) :
  (x^2 + y^2 - 4 * x + 6 * y = 0) ∧ (x^2 + y^2 - 6 * x = 0) → (x + 3 * y = 0) :=
by
  sorry

end line_AB_eq_x_plus_3y_zero_l70_70110


namespace number_of_acrobats_l70_70232

open Nat

theorem number_of_acrobats (a e c : ℕ) 
  (h1 : 2 * a + 4 * e + 2 * c = 58) 
  (h2 : a + e + c = 22) 
  (h3 : ∀ x, x = 2 → x = 4 → x = 2):
  a = 0 := 
begin
  sorry
end

end number_of_acrobats_l70_70232


namespace domain_of_f_l70_70332

noncomputable def f (x : ℝ) : ℝ := 1 / x + Real.log (x + 2)

theorem domain_of_f :
  {x : ℝ | (x ≠ 0) ∧ (x > -2)} = {x : ℝ | (-2 < x ∧ x < 0) ∨ (0 < x)} :=
by
  sorry

end domain_of_f_l70_70332


namespace caramel_candy_boxes_l70_70225

theorem caramel_candy_boxes 
  (chocolate_boxes : ℕ) 
  (pieces_per_box : ℕ) 
  (total_candies : ℕ) 
  (chocolate_candy_pieces : ℕ := chocolate_boxes * pieces_per_box)
  (caramel_candy_pieces : ℕ := total_candies - chocolate_candy_pieces)
  (caramel_boxes : ℕ := caramel_candy_pieces / pieces_per_box) :
  chocolate_boxes = 2 →
  pieces_per_box = 4 →
  total_candies = 28 →
  caramel_boxes = 5 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [chocolate_candy_pieces, caramel_candy_pieces, caramel_boxes]
  sorry

end caramel_candy_boxes_l70_70225


namespace small_primes_inf_and_all_l70_70056

def a (n : ℕ) : ℕ := 4^(2*n+1) + 3^(n+2)

theorem small_primes_inf_and_all :
  (∃ (p : ℕ), prime p ∧ (∀ k : ℕ, ∃ n : ℕ, n > k ∧ p ∣ a n) ∧ p = 5) ∧
  (∃ (q : ℕ), prime q ∧ (∀ n : ℕ, q ∣ a n) ∧ q = 13) ∧
  ∃ (pq : ℕ), pq = 5 * 13 ∧ pq = 65 :=
by
  sorry

end small_primes_inf_and_all_l70_70056


namespace g_eq_g_inv_iff_l70_70644

def g (x : ℝ) : ℝ := 3 * x - 7

def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_iff (x : ℝ) : g x = g_inv x ↔ x = 7 / 2 :=
by {
  sorry
}

end g_eq_g_inv_iff_l70_70644


namespace quadratic_inequality_l70_70374

theorem quadratic_inequality (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_l70_70374


namespace count_integers_with_three_factors_l70_70744

theorem count_integers_with_three_factors : 
  ∃ a : ℕ, a = 6 ∧ ∀ n : ℕ, (n < 200 ∧ (∃ p : ℕ, p.prime ∧ n = p^2)) → n ∈ {4, 9, 25, 49, 121, 169} :=
by
  -- statement of the problem
  sorry

end count_integers_with_three_factors_l70_70744


namespace place_tokens_even_parity_l70_70077

def board := fin 50 → fin 50 → bool

variable (initial_placement : board)
variable (is_free : fin 50 → fin 50 → bool)
variable (can_place_new_tokens : ∀ (B : board), B(initial_placement)) 

theorem place_tokens_even_parity :
  (∃ (new_tokens : board), (∀ i j, new_tokens i j → is_free i j) ∧ 
   (∀ i, even (finset.univ.filter (λ j, new_tokens i j ∨ initial_placement i j).card)) ∧
   (∀ j, even (finset.univ.filter (λ i, new_tokens i j ∨ initial_placement i j).card)) ∧ 
   finset.univ.filter (λ ij, new_tokens ij.1 ij.2 = tt).card ≤ 99)
  :=
  sorry

end place_tokens_even_parity_l70_70077


namespace range_of_function_l70_70506

theorem range_of_function : ∀ x : ℝ, (-2 <= x ∧ x <= 1) ↔ (3 ^ (-2) <= 3^(-x) ∧ 3^(-x) <= 3 ^ (-1)) :=
by
  intros x
  split
  all_goals { sorry }

end range_of_function_l70_70506


namespace sine_angle_greater_implies_angle_greater_l70_70032

noncomputable def triangle := {ABC : Type* // Π A B C : ℕ, 
  A + B + C = 180 ∧ 0 < A ∧ A < 180 ∧ 0 < B ∧ B < 180 ∧ 0 < C ∧ C < 180}

variables {A B C : ℕ} (T : triangle)

theorem sine_angle_greater_implies_angle_greater (h1 : 0 < A ∧ A < 180) (h2 : 0 < B ∧ B < 180)
  (h3 : 0 < C ∧ C < 180) (h_sum : A + B + C = 180) (h_sine : Real.sin A > Real.sin B) :
  A > B := 
sorry

end sine_angle_greater_implies_angle_greater_l70_70032


namespace liquid_film_radius_proof_l70_70597

noncomputable def circular_film_radius : ℝ :=
  let volume_Y := 9 * 3 * 18 in
  let thickness := 0.2 in
  let π := Real.pi in
  let volume_film := λ r : ℝ, π * r^2 * thickness in
  let r := Real.sqrt (volume_Y / (thickness * π)) in
  r

theorem liquid_film_radius_proof :
  circular_film_radius = Real.sqrt (2430 / Real.pi) :=
by
  sorry

end liquid_film_radius_proof_l70_70597


namespace graph_compact_Hausdorff_if_connected_and_locally_finite_l70_70004

variable (G : Type) [Graph G]

def is_connected : Prop := ...
def is_locally_finite : Prop := ...
def is_compact_Hausdorff_space (G : Type) [TopologicalSpace G] : Prop := ...

theorem graph_compact_Hausdorff_if_connected_and_locally_finite
  (h_connected : is_connected G)
  (h_locally_finite : is_locally_finite G) :
  is_compact_Hausdorff_space (|G|) :=
sorry

end graph_compact_Hausdorff_if_connected_and_locally_finite_l70_70004


namespace count_nice_subsets_l70_70844

open Set

noncomputable def S : Finset ℝ := sorry

def k : ℕ := sorry

def l : ℕ := sorry

def is_nice_subset (A : Finset ℝ) (S : Finset ℝ) (k l : ℕ) : Prop :=
  |(Finset.sum A id / k) - (Finset.sum (S \ A) id / l)| ≤ (k + l) / (2 * k * l)

theorem count_nice_subsets (S : Finset ℝ) (k l : ℕ) (hS : ∀ x ∈ S, x ∈ Icc 0 1)
  (hk : 0 < k) (hl : 0 < l) :
  k + l = S.card →
  ∃ n : ℕ, n ≥ (2 / (k + l)) * Nat.choose (k + l) k ∧
  (∀ A : Finset ℝ, A.card = k → A ⊆ S → is_nice_subset A S k l → A ∈ Finset.powersetLen k S : n) :=
sorry

end count_nice_subsets_l70_70844


namespace impossible_to_use_up_all_parts_l70_70579

theorem impossible_to_use_up_all_parts (p q r : ℕ) :
  (∃ p q r : ℕ,
    2 * p + 2 * r + 2 = A ∧
    2 * p + q + 1 = B ∧
    q + r = C) → false :=
by {
  sorry
}

end impossible_to_use_up_all_parts_l70_70579


namespace ellipse_equilateral_triangle_ratio_l70_70026

theorem ellipse_equilateral_triangle_ratio :
  let ellipse_eq (x y : ℝ) := x^2 / 4 + y^2 / 9 = 1,
      B : ℝ × ℝ := (2, 0),
      f1_f2 : ℝ := 2 * sqrt 5,
      ab_side (y : ℝ) : ℝ := sqrt y^2,
      y_coord : ℝ := sqrt 5 in
  ∀ (A C: ℝ × ℝ),
    A = (2, sqrt 5) ∨ A = (2, -sqrt 5) →
    C = (2, -sqrt 5) ∨ C = (2, sqrt 5) →
    ellipse_eq (A.1) (A.2) →
    ellipse_eq (C.1) (C.2) →
    (ab_side (sqrt 5) / f1_f2) = 1 / 2 :=
by
  intros ellipse_eq B f1_f2 ab_side y_coord A C hA hC hEllipseA hEllipseC
  sorry

end ellipse_equilateral_triangle_ratio_l70_70026


namespace james_hourly_rate_l70_70813

theorem james_hourly_rate (hours_per_day : ℕ) (days_per_week : ℕ) (weekly_earnings : ℕ) (h1 : hours_per_day = 8) (h2 : days_per_week = 4) (h3 : weekly_earnings = 640) : 
  (weekly_earnings / (hours_per_day * days_per_week) = 20) :=
by
  -- conditions and given values
  have hours_per_week : ℕ := hours_per_day * days_per_week,
  have hourly_rate : ℕ := weekly_earnings / hours_per_week,
  have h_hours_per_week : hours_per_week = 32 := by simp [h1, h2],
  have h_hourly_rate : hourly_rate = 20 := by simp [h3, h_hours_per_week],
  exact h_hourly_rate

end james_hourly_rate_l70_70813


namespace triangle_angle_inconsistency_l70_70975

theorem triangle_angle_inconsistency : 
  ∀ (L R T : ℝ), L + R + T = 180 ∧ L = 2 * R ∧ R = 60 → T = 0 → false :=
by
  intros L R T h1 h2,
  obtain ⟨h_sum, h_left, h_right⟩ := h1,
  rw h_right at *,
  rw h_left at *,
  linarith

end triangle_angle_inconsistency_l70_70975


namespace least_possible_product_of_distinct_primes_gt_50_l70_70140

-- Define a predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the conditions: two distinct primes greater than 50
def distinct_primes_gt_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50 ∧ p ≠ q

-- The least possible product of two distinct primes each greater than 50
theorem least_possible_product_of_distinct_primes_gt_50 :
  ∃ p q : ℕ, distinct_primes_gt_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_possible_product_of_distinct_primes_gt_50_l70_70140


namespace combined_operation_l70_70169

def f (x : ℚ) := (3 / 4) * x
def g (x : ℚ) := (5 / 3) * x

theorem combined_operation (x : ℚ) : g (f x) = (5 / 4) * x :=
by
    unfold f g
    sorry

end combined_operation_l70_70169


namespace reservoir_fullness_before_storm_l70_70568

-- Definition of the conditions as Lean definitions
def storm_deposits : ℝ := 120 -- in billion gallons
def reservoir_percentage_after_storm : ℝ := 85 -- percentage
def original_contents : ℝ := 220 -- in billion gallons

-- The proof statement
theorem reservoir_fullness_before_storm (storm_deposits reservoir_percentage_after_storm original_contents : ℝ) : 
    (169 / 340) * 100 = 49.7 := 
  sorry

end reservoir_fullness_before_storm_l70_70568


namespace number_division_l70_70541

theorem number_division (x : ℤ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by 
  sorry

end number_division_l70_70541


namespace loading_time_correct_l70_70184

-- Define the loading times in seconds
def cellphone_loading_time : ℕ := 540
def laptop_loading_time : ℕ := 15

-- Define the loading rates
def cellphone_rate : ℚ := 1 / cellphone_loading_time
def laptop_rate : ℚ := 1 / laptop_loading_time

-- Define the combined rate
def combined_rate : ℚ := cellphone_rate + laptop_rate

-- Define the time to load the video together, which is the reciprocal of combined_rate
noncomputable def combined_loading_time : ℚ := (1 : ℚ) / combined_rate

-- Define the expected time rounded to the nearest hundredth
noncomputable def rounded_combined_loading_time : ℚ := (real.to_rat ∘ round ∘ (*100) ∘ (/100) ∘ rat.to_real) combined_loading_time

-- The theorem to be proved: the rounded_combined_loading_time is approximately 14.59 seconds
theorem loading_time_correct : rounded_combined_loading_time = 14.59 := 
sorry

end loading_time_correct_l70_70184


namespace count_valid_ns_l70_70358

-- Define the problem conditions as statements in Lean
def is_divisible_by_five (m : ℕ) : Prop :=
  m % 5 = 0

def are_roots_valid (r s : ℕ) : Prop :=
  (r % 2 = 0 ∧ s % 2 = 0) ∨ (r % 5 = 0 ∨ s % 5 = 0)

def quadratic_roots (n m : ℕ) : Prop :=
  ∃ r s : ℕ, n = r + s ∧ m = r * s ∧ are_roots_valid r s

-- Main theorem statement to be proven
theorem count_valid_ns : 
  (∃! n : ℕ, n < 150 ∧ 
          ∃ m : ℕ, is_divisible_by_five m ∧ quadratic_roots n m) = 51 :=
sorry

end count_valid_ns_l70_70358


namespace number_of_pillars_l70_70886

def circular_track_length : ℕ := 1200
def interval_length : ℕ := 30

theorem number_of_pillars (track_length interval : ℕ) : track_length = 1200 ∧ interval = 30 → track_length / interval = 40 :=
by
  intro h
  cases h with h_length h_interval
  rw [h_length, h_interval]
  exact Nat.div_eq_of_eq_mul_right (Nat.pos_of_ne_zero (by decide)) rfl 
  sorry

end number_of_pillars_l70_70886


namespace triangle_angle_inconsistency_l70_70976

theorem triangle_angle_inconsistency : 
  ∀ (L R T : ℝ), L + R + T = 180 ∧ L = 2 * R ∧ R = 60 → T = 0 → false :=
by
  intros L R T h1 h2,
  obtain ⟨h_sum, h_left, h_right⟩ := h1,
  rw h_right at *,
  rw h_left at *,
  linarith

end triangle_angle_inconsistency_l70_70976


namespace simplify_cos_diff_l70_70094

theorem simplify_cos_diff :
  let a := Real.cos (36 * Real.pi / 180)
  let b := Real.cos (72 * Real.pi / 180)
  (b = 2 * a^2 - 1) → 
  (a = 1 - 2 * b^2) →
  a - b = 1 / 2 :=
by
  sorry

end simplify_cos_diff_l70_70094


namespace find_value_in_terms_of_a_b_l70_70362

theorem find_value_in_terms_of_a_b (a b : ℝ) (θ : ℝ) 
  (h : (sin θ ^ 6 / a) + (cos θ ^ 6 / b) = 1 / (a + 2 * b)) :
  (sin θ ^ 12 / (a ^ 3)) + (cos θ ^ 12 / (b ^ 3)) = (a ^ 3 + b ^ 3) / (a + 2 * b) ^ 6 :=
by
  sorry

end find_value_in_terms_of_a_b_l70_70362


namespace exists_lcm_lt_l70_70346

theorem exists_lcm_lt (p q : ℕ) (hpq_coprime : Nat.gcd p q = 1) (hp_gt_one : p > 1) (hq_gt_one : q > 1) (hpq_diff_gt_one : (p < q ∧ q - p > 1) ∨ (p > q ∧ p - q > 1)) :
  ∃ n : ℕ, Nat.lcm (p + n) (q + n) < Nat.lcm p q := by
  sorry

end exists_lcm_lt_l70_70346


namespace circumcircle_radius_l70_70737

-- Definitions based on the conditions
def area (A B C : ℝ) : ℝ := 2 * sqrt 3 
def ab := 2 
def bc := 4

-- The theorem statement we need to prove
theorem circumcircle_radius (A B C : ℝ) (area_ABC : area A B C = 2 * sqrt 3) (AB_eq_2 : ab = 2) (BC_eq_4 : bc = 4) : 
  ∃ (R : ℝ), R = 2 * sqrt 21 / 3 :=
sorry

end circumcircle_radius_l70_70737


namespace first_candidate_percentage_l70_70399

theorem first_candidate_percentage (P : ℝ) 
    (total_votes : ℝ) (votes_second : ℝ)
    (h_total_votes : total_votes = 1200)
    (h_votes_second : votes_second = 480) :
    (P / 100) * total_votes + votes_second = total_votes → P = 60 := 
by
  intro h
  rw [h_total_votes, h_votes_second] at h
  sorry

end first_candidate_percentage_l70_70399


namespace y_satisfies_equation_l70_70188

variable {x : ℝ}

def y (x : ℝ) : ℝ := x * Real.sqrt (1 - x^2)

def y' (x : ℝ) : ℝ :=
  (x * Real.sqrt (1 - x^2))'

theorem y_satisfies_equation (x : ℝ) : y x * y' x = x - 2 * x^3 := by
  sorry

end y_satisfies_equation_l70_70188


namespace combined_salaries_l70_70124
-- Import the required libraries

-- Define the salaries and conditions
def salary_c := 14000
def avg_salary_five := 8600
def num_individuals := 5
def total_salary := avg_salary_five * num_individuals

-- Define what we need to prove
theorem combined_salaries : total_salary - salary_c = 29000 :=
by
  -- The theorem statement
  sorry

end combined_salaries_l70_70124


namespace find_number_l70_70548

theorem find_number (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by
  sorry

end find_number_l70_70548


namespace germination_rate_proof_l70_70998

def random_number_table := [[78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279],
                            [43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820],
                            [61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636],
                            [63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421],
                            [42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983]]

noncomputable def first_4_tested_seeds : List Nat :=
  let numbers_in_random_table := [390, 737, 924, 220, 372]
  numbers_in_random_table.filter (λ x => x < 850) |>.take 4

theorem germination_rate_proof :
  first_4_tested_seeds = [390, 737, 220, 372] := 
by 
  sorry

end germination_rate_proof_l70_70998


namespace quadratic_root_exists_l70_70955

theorem quadratic_root_exists {a b c d : ℝ} (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
by
  sorry

end quadratic_root_exists_l70_70955


namespace woman_traveling_rate_l70_70208

theorem woman_traveling_rate :
  ∃ W, 
    (let man_rate := 6 in
     let time_not_moving := 1 / 3 in
     let distance_man_travels := man_rate * time_not_moving in
     let stopping_time := 10 / 60 in
     let distance_man_travels = distance_woman_travels before := distance_woman traveling because stoptime
   
    sorry

end woman_traveling_rate_l70_70208


namespace trajectory_passes_through_orthocenter_l70_70304

noncomputable def vector_perpendicular_to_edge (O A B C P : Point) (λ : ℝ) : Prop :=
  let OP := O.vector_to P
  let OA := O.vector_to A
  let AB := A.vector_to B 
  let AC := A.vector_to C
  let OC := O.vector_to C
  let BC := B.vector_to C
  λ > 0 ∧ 
  OP = OA + λ • ((AB / (AB.norm * (real.cos (angle B)))) + (AC / (AC.norm * (real.cos (angle C)))))

theorem trajectory_passes_through_orthocenter 
  {O A B C P : Point} {λ : ℝ}
  (h1 : vector_perpendicular_to_edge O A B C P λ) : 
  P.lies_on_the_orthocenter_of (triangle A B C) :=
sorry

end trajectory_passes_through_orthocenter_l70_70304


namespace lisa_needs_additional_marbles_l70_70862

theorem lisa_needs_additional_marbles
  (friends : ℕ) (initial_marbles : ℕ) (total_required_marbles : ℕ) :
  friends = 12 ∧ initial_marbles = 40 ∧ total_required_marbles = (friends * (friends + 1)) / 2 →
  total_required_marbles - initial_marbles = 38 :=
by
  sorry

end lisa_needs_additional_marbles_l70_70862


namespace place_tokens_even_parity_l70_70076

def board := fin 50 → fin 50 → bool

variable (initial_placement : board)
variable (is_free : fin 50 → fin 50 → bool)
variable (can_place_new_tokens : ∀ (B : board), B(initial_placement)) 

theorem place_tokens_even_parity :
  (∃ (new_tokens : board), (∀ i j, new_tokens i j → is_free i j) ∧ 
   (∀ i, even (finset.univ.filter (λ j, new_tokens i j ∨ initial_placement i j).card)) ∧
   (∀ j, even (finset.univ.filter (λ i, new_tokens i j ∨ initial_placement i j).card)) ∧ 
   finset.univ.filter (λ ij, new_tokens ij.1 ij.2 = tt).card ≤ 99)
  :=
  sorry

end place_tokens_even_parity_l70_70076


namespace magnitude_of_b_l70_70350

variable {V : Type*} [inner_product_space ℝ V]

variables (a b : V)
hypothesis (h1 : ∥a∥ = 1)
hypothesis (h2 : inner_product_space.inner a b = 3 / 2)
hypothesis (h3 : ∥a + b∥ = 2 * real.sqrt 2)

theorem magnitude_of_b (a b : V) (h1 : ∥a∥ = 1) (h2 : inner_product_space.inner a b = 3 / 2) (h3 : ∥a + b∥ = 2 * real.sqrt 2) :
  ∥b∥ = real.sqrt 5 := by
  sorry

end magnitude_of_b_l70_70350


namespace primes_eq_condition_l70_70661

theorem primes_eq_condition (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) : 
  p + q^2 = r^4 → p = 7 ∧ q = 3 ∧ r = 2 := 
by
  sorry

end primes_eq_condition_l70_70661


namespace mrsHiltTotalMoney_l70_70450

noncomputable def totalMoney (quarters dimes nickels pennies half_dollars one_dollar_coins two_dollar_canadian_coins : Nat) (cad_to_usd : Rat) : Rat :=
  let totalCents : Nat :=
    (quarters * 25) +
    (dimes * 10) +
    (nickels * 5) +
    (pennies * 1) +
    (half_dollars * 50) +
    (one_dollar_coins * 100) +
    (two_dollar_canadian_coins * 2 * 100 * cad_to_usd.num / cad_to_usd.denom)
  totalCents / 100

theorem mrsHiltTotalMoney :
  totalMoney 4 6 8 12 3 5 2 (4/5) = 11.82 := 
by
  sorry

end mrsHiltTotalMoney_l70_70450


namespace Quadrilateral_proof_l70_70248

-- Definitions reflecting the conditions from the problem
def nonconvex_quadrilateral (A B C D E F K L J I : Point) :=
  ∃ (K L J I : Point), 
    is_nonconvex A B C D ∧
    angle C > π ∧ -- \(\angle C > 180^\circ\) corresponds to angle > π radians
    collinear D C F ∧
    collinear B C E ∧
    intersects_AB AD BC CD K L J I

-- The main theorem statement regarding the specific conditions and the final assertion
theorem Quadrilateral_proof (
  A B C D E F K L J I : Point) 
  (h : nonconvex_quadrilateral A B C D E F K L J I)
  (h1 : segment_eq D I C F)
  (h2 : segment_eq B J C E) : 
  segment_eq K J I L :=
sorry

end Quadrilateral_proof_l70_70248


namespace slope_of_line_l70_70127

theorem slope_of_line :
  ∀ (t : ℝ),
  let x := 3 - (Real.sqrt 3 / 2) * t,
      y := 1 + (1 / 2) * t
  in
  (∃ m b : ℝ, y = m * x + b ∧ m = - Real.sqrt 3 / 3) :=
begin
  sorry
end

end slope_of_line_l70_70127


namespace number_division_l70_70543

theorem number_division (x : ℤ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by 
  sorry

end number_division_l70_70543


namespace find_f_l70_70851

noncomputable def f : ℝ → ℝ := sorry

theorem find_f (f : ℝ → ℝ) (h₀ : f 0 = 2) 
  (h₁ : ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2) / 2) + (x + y)^2) :
  ∀ x : ℝ, f x = 2 - 2 * x :=
sorry

end find_f_l70_70851


namespace equivalent_G_l70_70424

noncomputable def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))
noncomputable def G (x : ℝ) : ℝ := F ((5 * x - x ^ 3) / (1 - 5 * x ^ 2))

theorem equivalent_G (x : ℝ) : G x = -3 * (F x) := by
  sorry

end equivalent_G_l70_70424


namespace max_shortest_side_decagon_inscribed_circle_l70_70492

noncomputable def shortest_side_decagon : ℝ :=
  2 * Real.sin (36 * Real.pi / 180 / 2)

theorem max_shortest_side_decagon_inscribed_circle :
  shortest_side_decagon = (Real.sqrt 5 - 1) / 2 :=
by {
  -- Proof details here
  sorry
}

end max_shortest_side_decagon_inscribed_circle_l70_70492


namespace sum_primes_1_to_50_l70_70677

theorem sum_primes_1_to_50 : 
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] in
  primes.minimum = some 2 ∧ primes.maximum = some 47 → (2 + 47) = 49 :=
by
  intros h_min_max
  have h_min : primes.minimum = some 2 := and.left h_min_max
  have h_max : primes.maximum = some 47 := and.right h_min_max
  rw [h_min, h_max]
  norm_num
  sorry

end sum_primes_1_to_50_l70_70677


namespace swimming_lane_length_l70_70261

-- Conditions
def num_round_trips : ℕ := 3
def total_distance : ℕ := 600

-- Hypothesis that 1 round trip is equivalent to 2 lengths of the lane
def lengths_per_round_trip : ℕ := 2

-- Statement to prove
theorem swimming_lane_length :
  (total_distance / (num_round_trips * lengths_per_round_trip) = 100) := by
  sorry

end swimming_lane_length_l70_70261


namespace incorrect_statement_D_l70_70442

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2

theorem incorrect_statement_D :
  (∃ T : ℝ, ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x : ℝ, f (π / 2 + x) = f (π / 2 - x)) ∧
  (f (π / 2 + π / 4) = 0) ∧ ¬(∀ x : ℝ, (π / 2 < x ∧ x < π) → f x < f (x - 0.1)) := by
  sorry

end incorrect_statement_D_l70_70442


namespace hex_tessellation_min_colors_l70_70162

theorem hex_tessellation_min_colors (H : tessellation hexagon) : 
  ∃ n, n = 3 ∧ ∀ coloring : H.color n, no_adjacent_hexagons_same_color coloring := sorry

end hex_tessellation_min_colors_l70_70162


namespace quadratic_solutions_l70_70915

-- Define the equation x^2 - 6x + 8 = 0
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

-- Lean statement for the equivalence of solutions
theorem quadratic_solutions : ∀ x : ℝ, quadratic_eq x ↔ x = 2 ∨ x = 4 :=
by
  intro x
  dsimp [quadratic_eq]
  sorry

end quadratic_solutions_l70_70915


namespace second_player_wins_l70_70947

noncomputable def game_strategy (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ (p1_turns : ℕ) (p2_turns : ℕ) (remaining : ℕ → Prop),
    if remaining n then
      let m := n - (p1_turns + p2_turns) in
      m = 2 →
      ∑ i in (finset.range n).filter remaining, i = 2 (mod 3) →
      (p2_turns > p1_turns ∧ p2_turns = n // 3)
    else
      false

theorem second_player_wins : game_strategy 1000 :=
by
  -- The proof strategy would go here.
  sorry

end second_player_wins_l70_70947


namespace min_mn_value_l70_70113

theorem min_mn_value
  (a : ℝ) (m : ℝ) (n : ℝ)
  (ha_pos : a > 0) (ha_ne_one : a ≠ 1) (hm_pos : m > 0) (hn_pos : n > 0)
  (H : (1 : ℝ) / m + (1 : ℝ) / n = 4) :
  m + n ≥ 1 :=
sorry

end min_mn_value_l70_70113


namespace perimeter_quadrilateral_l70_70407

-- Define the lengths of the sides and hypotenuses based on conditions
def AE : ℝ := 30
def AB : ℝ := AE
def BE : ℝ := AE * Real.sqrt 2
def BC : ℝ := BE
def CE : ℝ := BC * Real.sqrt 2
def CD : ℝ := CE
def DE : ℝ := CD * Real.sqrt 2
def DA : ℝ := AE + DE

-- Statement: perimeter of quadrilateral ABCD
theorem perimeter_quadrilateral :
  AB + BC + CD + DA = 90 + 90 * Real.sqrt 2 :=
by
  sorry

end perimeter_quadrilateral_l70_70407


namespace smallest_k_l70_70339

def u (n : ℕ) : ℕ := n^4 + n^2 + n

def Δ (k : ℕ) (u : ℕ → ℕ) : ℕ → ℕ 
| 0 => u
| k + 1 => λ n, Δ k (λ n, u (n + 1) - u n) n 

theorem smallest_k (n : ℕ) : (∀ n, Δ 5 u n = 0) :=
sorry

end smallest_k_l70_70339


namespace exists_quad_root_l70_70963

theorem exists_quad_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (∃ x, x^2 + a * x + b = 0) ∨ (∃ x, x^2 + c * x + d = 0) :=
sorry

end exists_quad_root_l70_70963


namespace no_odd_terms_in_expansion_l70_70006

theorem no_odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 0) (hq : q % 2 = 0) :
  ∀ k : ℕ, k ≤ 8 → (binomial 8 k * p^ (8 - k) * q^ k) % 2 = 0 := by
  sorry

end no_odd_terms_in_expansion_l70_70006


namespace omega_range_l70_70334

noncomputable def f (ω x : ℝ) : ℝ :=
  (Real.sin (ω * x))^2 + Real.sin (ω * x) * Real.cos (ω * x) - 1

theorem omega_range (ω : ℝ) :
  ( ∃ ω > 0, ∀ (x ∈ Ioo (0 : ℝ) (Real.pi / 2)),
    f ω x = 0 ∧
    (∃! y₁ y₂ y₃ y₄ ∈ Ioo (0 : ℝ) (Real.pi / 2), (y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₁ ≠ y₄ ∧ y₂ ≠ y₃ ∧ y₂ ≠ y₄ ∧ y₃ ≠ y₄)
  → (3 < ω ∧ ω ≤ 9 / 2)) :=
sorry

end omega_range_l70_70334


namespace sequence_count_l70_70885

-- Proof problem statement in Lean 4: 
theorem sequence_count (n : ℕ) (n = 100) : 
  let countSequences (n : ℕ) : ℕ := 5^n - 3^n 
  in countSequences 100 = 5^100 - 3^100 :=
sorry

end sequence_count_l70_70885


namespace celery_cost_l70_70445

noncomputable def supermarket_problem
  (total_money : ℕ)
  (price_cereal discount_cereal price_bread : ℕ)
  (price_milk discount_milk price_potato num_potatoes : ℕ)
  (leftover_money : ℕ) 
  (total_cost : ℕ) 
  (cost_of_celery : ℕ) :=
  (price_cereal * discount_cereal / 100 + 
   price_bread + 
   price_milk * discount_milk / 100 + 
   price_potato * num_potatoes) + 
   leftover_money = total_money ∧
  total_cost = total_money - leftover_money ∧
  (price_cereal * discount_cereal / 100 + 
   price_bread + 
   price_milk * discount_milk / 100 + 
   price_potato * num_potatoes) = total_cost - cost_of_celery

theorem celery_cost (total_money : ℕ := 60) 
  (price_cereal : ℕ := 12) 
  (discount_cereal : ℕ := 50) 
  (price_bread : ℕ := 8) 
  (price_milk : ℕ := 10) 
  (discount_milk : ℕ := 90) 
  (price_potato : ℕ := 1) 
  (num_potatoes : ℕ := 6) 
  (leftover_money : ℕ := 26) 
  (total_cost : ℕ := 34) :
  supermarket_problem total_money price_cereal discount_cereal price_bread price_milk discount_milk price_potato num_potatoes leftover_money total_cost 5 :=
by
  sorry

end celery_cost_l70_70445


namespace sum_of_row_and_column_products_nonzero_l70_70303

noncomputable def matrix_product (M : Matrix (Fin 2007) (Fin 2007) ℤ) : (Fin 2007 → ℤ) :=
  λ i => ∏ j, M i j

noncomputable def column_product (M : Matrix (Fin 2007) (Fin 2007) ℤ) : (Fin 2007 → ℤ) :=
  λ j => ∏ i, M i j

theorem sum_of_row_and_column_products_nonzero (M : Matrix (Fin 2007) (Fin 2007) ℤ)
  (h : ∀ (i j : Fin 2007), M i j = 1 ∨ M i j = -1) :
  (∑ i, matrix_product M i) + (∑ j, column_product M j) ≠ 0 :=
sorry

end sum_of_row_and_column_products_nonzero_l70_70303


namespace min_value_does_not_exist_l70_70302

noncomputable def f (x : ℝ) : ℝ := 
  (1 + 1 / log (sqrt (x^2 + 10) + x)) * (1 + 2 / log (sqrt (x^2 + 10) - x))

theorem min_value_does_not_exist : 
∀ x : ℝ, (0 < x ∧ x < 4.5) → ¬(∃ y : ℝ, is_minimum y (f x)) :=
by sorry

end min_value_does_not_exist_l70_70302


namespace gcd_of_1237_and_1957_is_one_l70_70695

noncomputable def gcd_1237_1957 : Nat := Nat.gcd 1237 1957

theorem gcd_of_1237_and_1957_is_one : gcd_1237_1957 = 1 :=
by
  unfold gcd_1237_1957
  have : Nat.gcd 1237 1957 = 1 := sorry
  exact this

end gcd_of_1237_and_1957_is_one_l70_70695


namespace product_of_roots_l70_70289

theorem product_of_roots (x : ℝ) (h : x + 16 / x = 12) : (8 : ℝ) * (4 : ℝ) = 32 :=
by
  -- Your proof would go here
  sorry

end product_of_roots_l70_70289


namespace find_original_price_of_petrol_l70_70216

open Real

noncomputable def original_price_of_petrol (P : ℝ) : Prop :=
  ∀ G : ℝ, 
  (G * P = 300) ∧ 
  ((G + 7) * 0.85 * P = 300) → 
  P = 7.56

-- Theorems should ideally be defined within certain scopes or namespaces
theorem find_original_price_of_petrol (P : ℝ) : original_price_of_petrol P :=
  sorry

end find_original_price_of_petrol_l70_70216


namespace find_k_l70_70837

theorem find_k (a b c k : ℤ) (g : ℤ → ℤ)
  (h₁ : g 1 = 0)
  (h₂ : 10 < g 5 ∧ g 5 < 20)
  (h₃ : 30 < g 6 ∧ g 6 < 40)
  (h₄ : 3000 * k < g 100 ∧ g 100 < 3000 * (k + 1))
  (h_g : ∀ x, g x = a * x^2 + b * x + c) :
  k = 9 :=
by
  sorry

end find_k_l70_70837


namespace johns_net_earnings_l70_70818

def hourly_rate (day : String) : ℝ :=
  if day = "Monday" then 10
  else if day = "Wednesday" then 12
  else if day = "Friday" then 15
  else if day = "Saturday" then 20
  else 0

def daily_earnings (day : String) : ℝ :=
  4 * hourly_rate(day)

def total_gross_earnings : ℝ :=
  daily_earnings("Monday") + daily_earnings("Wednesday") +
  daily_earnings("Friday") + daily_earnings("Saturday")

def platform_fee (earnings : ℝ) : ℝ :=
  0.2 * earnings

def net_earnings_before_taxes (gross_earnings : ℝ) : ℝ :=
  gross_earnings - platform_fee(gross_earnings)

def tax (earnings : ℝ) : ℝ :=
  0.25 * earnings

def net_earnings_after_taxes (gross_earnings : ℝ) : ℝ :=
  net_earnings_before_taxes(gross_earnings) - 
  tax(net_earnings_before_taxes(gross_earnings))

theorem johns_net_earnings : net_earnings_after_taxes(total_gross_earnings) = 136.80 := 
by 
  sorry

end johns_net_earnings_l70_70818


namespace men_bathing_suits_l70_70195

theorem men_bathing_suits (total : ℕ) (women : ℕ) (men : ℕ) 
  (h1 : total = 19766) (h2 : women = 4969) : men = 14797 :=
by {
  have : men = total - women,
  sorry
}

end men_bathing_suits_l70_70195


namespace intersection_S_T_l70_70300

def S : Set ℝ := { x | 2 * x + 1 > 0 }
def T : Set ℝ := { x | 3 * x - 5 < 0 }

theorem intersection_S_T :
  S ∩ T = { x | -1/2 < x ∧ x < 5/3 } := by
  sorry

end intersection_S_T_l70_70300


namespace quadratic_inequality_solution_l70_70368

variable (x : ℝ)

theorem quadratic_inequality_solution (hx : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_solution_l70_70368


namespace vector_multiplication_not_associative_l70_70364

-- Define vector space, scalar multiplication, and scalar product
structure Vector (α : Type) [Field α] := 
  (x y z : α)
  
namespace Vector

variables {α : Type} [Field α]

def add (a b : Vector α) : Vector α := 
  ⟨a.x + b.x, a.y + b.y, a.z + b.z⟩

def smul (m : α) (a : Vector α) : Vector α := 
  ⟨m * a.x, m * a.y, m * a.z⟩

def dot (a b : Vector α) : α := 
  a.x * b.x + a.y * b.y + a.z * b.z

-- Using these defined operations, we create example vectors
def vector_a : Vector ℝ := ⟨1, 2, 3⟩
def vector_b : Vector ℝ := ⟨4, 5, 6⟩
def vector_c : Vector ℝ := ⟨7, 8, 9⟩
def scalar_m : ℝ := 10

-- State proving requirement: the non-necessary equality
theorem vector_multiplication_not_associative :
  (dot vector_a vector_b) • vector_c ≠ vector_a • (dot vector_b vector_c) :=
sorry

end Vector

end vector_multiplication_not_associative_l70_70364


namespace range_of_m_l70_70940

-- Define the variables and main theorem
theorem range_of_m (m : ℝ) (a b c : ℝ) 
  (h₀ : a = 3) (h₁ : b = (1 - 2 * m)) (h₂ : c = 8)
  : -5 < m ∧ m < -2 :=
by
  -- Given that a, b, and c are sides of a triangle, we use the triangle inequality theorem
  -- This code will remain as a placeholder of that proof
  sorry

end range_of_m_l70_70940


namespace both_pumps_fill_time_l70_70218

theorem both_pumps_fill_time :
  let small_pump_rate := (1 : ℝ) / 3,
      large_pump_rate := 4,
      combined_rate := small_pump_rate + large_pump_rate in
  (1 / combined_rate = 3 / 13) :=
by
  let small_pump_rate := (1 : ℝ) / 3,
      large_pump_rate := 4,
      combined_rate := small_pump_rate + large_pump_rate
  show 1 / combined_rate = 3 / 13
  sorry

end both_pumps_fill_time_l70_70218


namespace congruent_rectangle_perimeter_l70_70931

theorem congruent_rectangle_perimeter (a b p q : ℝ) : 
  let l1 := b - q,
      l2 := a - p in
  2 * (l1 + l2) = 2 * (a + b - p - q) :=
by
  let l1 := b - q
  let l2 := a - p
  sorry

end congruent_rectangle_perimeter_l70_70931


namespace trisect_point_area_under_curve_l70_70516

noncomputable def log_base_2 (x : ℝ) : ℝ := (Real.log x) / (Real.log 2)

theorem trisect_point_area_under_curve :
  let x_1 := 2
  let x_2 := 32
  let f := log_base_2
  ∃ x_3 : ℝ, x_3 = 2^(7/3) ∧ 
  ∃ area : ℝ, area = (1 / Real.log 2) * ((λ x => x * Real.log x - x) (2^(7/3)) - (λ x => x * Real.log x - x) 2) :=
by
  let x_1 := 2
  let x_2 := 32
  let f := log_base_2
  let y1 := f x_1
  let y2 := f x_2
  let yc := (2 / 3) * y1 + (1 / 3) * y2
  let x_3 := 2^(7 / 3)
  let area := (1 / Real.log 2) * ((λ x => x * Real.log x - x) x_3 - (λ x => x * Real.log x - x) x_1)
  have h1 : x_3 = 2^(7/3) := sorry
  have h2 : area = (1 / Real.log 2) * ((λ x => x * Real.log x - x) x_3 - (λ x => x * Real.log x - x) x_1) := sorry
  exact ⟨x_3, h1, area, h2⟩

end trisect_point_area_under_curve_l70_70516


namespace solve_quadratic_l70_70911

theorem solve_quadratic : ∀ x : ℝ, x ^ 2 - 6 * x + 8 = 0 ↔ x = 2 ∨ x = 4 := by
  sorry

end solve_quadratic_l70_70911


namespace parabola_shifted_l70_70484

-- Define the original parabola
def originalParabola (x : ℝ) : ℝ := (x + 2)^2 + 3

-- Shift the parabola by 3 units to the right
def shiftedRight (x : ℝ) : ℝ := originalParabola (x - 3)

-- Then shift the parabola 2 units down
def shiftedRightThenDown (x : ℝ) : ℝ := shiftedRight x - 2

-- The problem asks to prove that the final expression is equal to (x - 1)^2 + 1
theorem parabola_shifted (x : ℝ) : shiftedRightThenDown x = (x - 1)^2 + 1 :=
by
  sorry

end parabola_shifted_l70_70484


namespace first_number_in_set_is_8_l70_70012

noncomputable def mean (s : List ℕ) : ℕ := (s.sum) / s.length

def median (s : List ℕ) : ℕ := ((s.nth_le (s.length / 2 - 1) (nat.div_lt_of_lt (nat.pred_le n (lt_of_succ_lt (nat.succ_pos (List.length s)))))).getOrElse 0 + (s.nth_le (s.length / 2) (nat.div_lt_of_lt (nat.pred_le n (lt_of_succ_lt (nat.succ_pos (List.length s)))))).getOrElse 0) / 2

theorem first_number_in_set_is_8 (x : ℕ) (s : List ℕ) (h1 : s = [x, 16, 24, 32, 40, 48]) (h2 : mean s * median s = 784) : x = 8 :=
by 
  sorry

end first_number_in_set_is_8_l70_70012


namespace robert_coin_arrangement_l70_70895

noncomputable def num_arrangements (gold : ℕ) (silver : ℕ) : ℕ :=
  if gold + silver = 8 ∧ gold = 5 ∧ silver = 3 then 504 else 0

theorem robert_coin_arrangement :
  num_arrangements 5 3 = 504 := 
sorry

end robert_coin_arrangement_l70_70895


namespace range_of_a_l70_70739

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 5

theorem range_of_a 
  (a : ℝ)
  (decreasing_on : ∀ x, x ≤ 2 → deriv (λ x, f x a) x ≤ 0) 
  (bounded_diff : ∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ a + 1 ∧ 1 ≤ x2 ∧ x2 ≤ a + 1 → |f x1 a - f x2 a| ≤ 4) :
  2 ≤ a ∧ a ≤ 3 :=
by sorry

end range_of_a_l70_70739


namespace min_distance_sum_well_l70_70024

theorem min_distance_sum_well (A B C : ℝ) (h1 : B = A + 50) (h2 : C = B + 50) :
  ∃ X : ℝ, X = B ∧ (∀ Y : ℝ, (dist Y A + dist Y B + dist Y C) ≥ (dist B A + dist B B + dist B C)) :=
sorry

end min_distance_sum_well_l70_70024


namespace exists_parallel_lines_intersecting_2ngon_once_l70_70841

open Function

variable {n : ℕ} (h_n : 0 < n) 
variable (A : Fin (2 * n) → ℝ × ℝ) 
variable (h_convex : ConvexPolygon (A))

theorem exists_parallel_lines_intersecting_2ngon_once :
  ∃ i : Fin n, ∃ l : Line, ∃ m : Line,
    Parallel l m ∧
    IntersectOnce l (2 * n) A i ∧
    IntersectOnce m (2 * n) A (i + n) :=
  sorry

end exists_parallel_lines_intersecting_2ngon_once_l70_70841


namespace total_students_in_Lansing_l70_70825

theorem total_students_in_Lansing:
  (number_of_schools : Nat) → 
  (students_per_school : Nat) → 
  (total_students : Nat) →
  number_of_schools = 25 → 
  students_per_school = 247 → 
  total_students = number_of_schools * students_per_school → 
  total_students = 6175 :=
by
  intros number_of_schools students_per_school total_students h_schools h_students h_total
  rw [h_schools, h_students] at h_total
  exact h_total

end total_students_in_Lansing_l70_70825


namespace ratio_of_pieces_l70_70585

theorem ratio_of_pieces (total_length shorter_piece longer_piece : ℕ) 
    (h1 : total_length = 6) (h2 : shorter_piece = 2)
    (h3 : longer_piece = total_length - shorter_piece) :
    ((longer_piece : ℚ) / (shorter_piece : ℚ)) = 2 :=
by
    sorry

end ratio_of_pieces_l70_70585


namespace sin_cos_bounds_l70_70123

theorem sin_cos_bounds (w x y z : ℝ)
  (hw : -Real.pi / 2 ≤ w ∧ w ≤ Real.pi / 2)
  (hx : -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2)
  (hy : -Real.pi / 2 ≤ y ∧ y ≤ Real.pi / 2)
  (hz : -Real.pi / 2 ≤ z ∧ z ≤ Real.pi / 2)
  (h₁ : Real.sin w + Real.sin x + Real.sin y + Real.sin z = 1)
  (h₂ : Real.cos (2 * w) + Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) ≥ 10 / 3) :
  0 ≤ w ∧ w ≤ Real.pi / 6 ∧ 0 ≤ x ∧ x ≤ Real.pi / 6 ∧ 0 ≤ y ∧ y ≤ Real.pi / 6 ∧ 0 ≤ z ∧ z ≤ Real.pi / 6 :=
by
  sorry

end sin_cos_bounds_l70_70123


namespace quadratic_inequality_solution_l70_70367

variable (x : ℝ)

theorem quadratic_inequality_solution (hx : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_solution_l70_70367


namespace combined_weight_of_candles_l70_70270

theorem combined_weight_of_candles 
  (beeswax_weight_per_candle : ℕ)
  (coconut_oil_weight_per_candle : ℕ)
  (total_candles : ℕ)
  (candles_made : ℕ) 
  (total_weight: ℕ) 
  : 
  beeswax_weight_per_candle = 8 → 
  coconut_oil_weight_per_candle = 1 → 
  total_candles = 10 → 
  candles_made = total_candles - 3 →
  total_weight = candles_made * (beeswax_weight_per_candle + coconut_oil_weight_per_candle) →
  total_weight = 63 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end combined_weight_of_candles_l70_70270


namespace projection_result_l70_70555

def vector_v1 : ℝ × ℝ := (-3, 2)
def vector_v2 : ℝ × ℝ := (4, -1)
def dir_vector : ℝ × ℝ := (7, -3)
def proj_q : ℝ × ℝ := (15 / 58, 35 / 58)

theorem projection_result:
  ∀ t : ℝ,
  let q := (7 * t - 3, -3 * t + 2) in
  (q.1 * dir_vector.1 + q.2 * dir_vector.2 = 0) →
  (proj_q = (7 * (27 / 58) - 3, -3 * (27 / 58) + 2)) →
  proj_q = (15 / 58, 35 / 58) := by
  intros t q orthogonal condition
  have eq_t : t = 27 / 58 := by
    sorry
  sorry

end projection_result_l70_70555


namespace erika_rick_money_left_l70_70681

theorem erika_rick_money_left (gift_cost cake_cost erika_savings : ℝ)
  (rick_savings := gift_cost / 2) (total_savings := erika_savings + rick_savings) 
  (total_cost := gift_cost + cake_cost) : (total_savings - total_cost) = 5 :=
by
  -- Given conditions from the problem
  have h_gift_cost : gift_cost = 250 := sorry
  have h_cake_cost : cake_cost = 25 := sorry
  have h_erika_savings : erika_savings = 155 := sorry
  -- Show that the remaining money is $5
  have h_rick_savings : rick_savings = 125 := by
    rw [←h_gift_cost]
    norm_num
  have h_total_savings : total_savings = 280 := by
    rw [←h_erika_savings, ←h_rick_savings]
    norm_num
  have h_total_cost : total_cost = 275 := by
    rw [←h_gift_cost, ←h_cake_cost]
    norm_num
  rw [←h_total_savings, ←h_total_cost]
  norm_num
  done

end erika_rick_money_left_l70_70681


namespace limit_sequence_l70_70602

noncomputable def u : ℕ → ℝ
| 0     := 1
| (n+1) := 1 / (Finset.sum (Finset.range (n + 1)) u)

theorem limit_sequence :
  ∃ (α β : ℝ), α = Real.sqrt 2 ∧ β = 1/2 ∧
  (λ n, (Finset.sum (Finset.range (n + 1)) u) / (α * n^β)) ⟶ 1 at Top :=
begin
  use [Real.sqrt 2, 1/2],
  sorry
end

end limit_sequence_l70_70602


namespace part1_l70_70829

noncomputable def S (a : ℕ → ℕ) (n : ℕ) := (finset.range (n + 1)).sum (λ i, a i)

theorem part1 (a : ℕ → ℕ)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, (S a (n + 1) / a (n + 1)) - (S a n / a n) = 1 / 2) :
  ∀ n, a n = n := by
  sorry

end part1_l70_70829


namespace quadratic_inequality_l70_70376

theorem quadratic_inequality (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_l70_70376


namespace PQRS_value_l70_70843

noncomputable def PQRS (P Q R S : ℝ) :=
  (log10 (P * Q) + log10 (P * S) = 3) ∧
  (log10 (Q * S) + log10 (Q * R) = 4) ∧
  (log10 (R * P) + log10 (R * S) = 5)

theorem PQRS_value (P Q R S : ℝ) (h : PQRS P Q R S) : 
  P * Q * R * S = 10000 :=
sorry

end PQRS_value_l70_70843


namespace min_sum_y1_y2_l70_70312

theorem min_sum_y1_y2 (y : ℕ → ℕ) (h_seq : ∀ n ≥ 1, y (n+2) = (y n + 2013)/(1 + y (n+1))) : 
  ∃ y1 y2, y1 + y2 = 94 ∧ (∀ n, y n > 0) ∧ (y 1 = y1) ∧ (y 2 = y2) := 
sorry

end min_sum_y1_y2_l70_70312


namespace find_letter_S_l70_70260

variable (x : ℕ)

def date_A := x - 2
def date_B := x - 1
def date_C := x

theorem find_letter_S : ∃ y, date_C x + y = date_A x + date_B x ∧ y = x - 3 ∧ letter corresponding to y = "S" := 
by
  sorry

end find_letter_S_l70_70260


namespace trapezoid_area_l70_70692

theorem trapezoid_area (AC BD BC AD : ℝ)
  (hAC : AC = 7)
  (hBD : BD = 8)
  (hBC : BC = 3)
  (hAD : AD = 6) :
  (1 / 4) * (AC + BD) * sqrt ((BC + AD)^2 - (AC - BD)^2) = 12 * sqrt 5 := by
  sorry

end trapezoid_area_l70_70692


namespace intersection_locus_l70_70421

noncomputable def locus_intersection (a b : ℝ) (A ≠ 0) (B ≠ 0) : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ k : ℝ, p = ((k^2 * b) / (a * b + k^2), (a * k * b) / (a * b + k^2)) }

theorem intersection_locus (a b : ℝ) (hA : a ≠ 0) (hB : b ≠ 0) :
    locus_intersection a b A ≠ 0 B projection_ { (0, 0), (b, 0) } :=
sorry

end intersection_locus_l70_70421


namespace largest_angle_of_convex_hexagon_l70_70498

noncomputable def largest_angle (angles : Fin 6 → ℝ) (consecutive : ∀ i : Fin 5, angles i + 1 = angles (i + 1)) (sum_eq_720 : (∑ i, angles i) = 720) : ℝ :=
  sorry

theorem largest_angle_of_convex_hexagon (angles : Fin 6 → ℝ) (consecutive : ∀ i : Fin 5, angles i + 1 = angles (i + 1)) (sum_eq_720 : (∑ i, angles i) = 720) :
  largest_angle angles consecutive sum_eq_720 = 122.5 :=
  sorry

end largest_angle_of_convex_hexagon_l70_70498


namespace correct_calculation_l70_70556

theorem correct_calculation : 
  (\(\sqrt{(2)^2}\) + \(\sqrt{(3)^2}\)) = 5 :=
by sorry

end correct_calculation_l70_70556


namespace certainEvent_l70_70171

def scoopingTheMoonOutOfTheWaterMeansCertain : Prop :=
  ∀ (e : String), e = "scooping the moon out of the water" → (∀ (b : Bool), b = true)

theorem certainEvent (e : String) (h : e = "scooping the moon out of the water") : ∀ (b : Bool), b = true :=
  by
  sorry

end certainEvent_l70_70171


namespace find_value_of_q_l70_70379

theorem find_value_of_q (p q : ℂ) :
  (∀ x : ℂ, 2 * x^2 + p * x + q = 0 → (x = 3 + 2i ∨ x = 3 - 2i)) →
  q = 26 :=
by
  intro h
  sorry

end find_value_of_q_l70_70379


namespace max_prism_intersections_l70_70204

structure Prism where
  base_top_edges : Set ℝ
  base_bottom_edges : Set ℝ
  lateral_edges : Set ℝ
  
structure Plane where
  intersects : Prism → Set ℝ
  
def max_intersections (p : Prism) (pl : Plane) : ℕ := (pl.intersects p).card

noncomputable def intersecting_edges (pl : Plane) (p : Prism) (n : ℕ) : Prop :=
  max_intersections p pl = n

theorem max_prism_intersections (p : Prism) (pl : Plane) :
  intersecting_edges pl p 8 :=
by
  sorry

end max_prism_intersections_l70_70204


namespace quadratic_inequality_l70_70377

theorem quadratic_inequality (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_l70_70377


namespace problem_l70_70764

def f (x : ℤ) : ℤ := 3 * x - 1
def g (x : ℤ) : ℤ := 2 * x + 5

theorem problem (h : ℤ) :
  (g (f (g (3))) : ℚ) / f (g (f (3))) = 69 / 206 :=
by
  sorry

end problem_l70_70764


namespace LisaNeedsMoreMarbles_l70_70866

theorem LisaNeedsMoreMarbles :
  let friends := 12
  let marbles := 40
  let required_marbles := (friends * (friends + 1)) / 2
  let additional_marbles := required_marbles - marbles
  additional_marbles = 38 :=
by
  let friends := 12
  let marbles := 40
  let required_marbles := (friends * (friends + 1)) / 2
  let additional_marbles := required_marbles - marbles
  have h1 : required_marbles = 78 := by
    calc (friends * (friends + 1)) / 2
      _ = (12 * 13) / 2 : by rfl
      _ = 156 / 2 : by rfl
      _ = 78 : by norm_num
  have h2 : additional_marbles = 38 := by
    calc required_marbles - marbles
      _ = 78 - 40 : by rw h1
      _ = 38 : by norm_num
  exact h2

end LisaNeedsMoreMarbles_l70_70866


namespace largest_angle_convex_hexagon_l70_70501

theorem largest_angle_convex_hexagon : 
  ∃ x : ℝ, (x-3) + (x-2) + (x-1) + x + (x+1) + (x+2) = 720 → (x + 2) = 122.5 :=
by 
  intros,
  sorry

end largest_angle_convex_hexagon_l70_70501


namespace polynomial_sum_l70_70100

theorem polynomial_sum (x : ℂ) (h1 : x ≠ 1) (h2 : x^2021 - 4 * x + 3 = 0) :
  x^2020 + x^2019 + ... + x + 1 = 4 := 
sorry

end polynomial_sum_l70_70100


namespace split_evenly_rounding_l70_70193

noncomputable def split_bill_evenly (total_bill : ℝ) (num_people : ℕ) : ℝ :=
  (total_bill / num_people).round(2)

theorem split_evenly_rounding :
  split_bill_evenly 314.12 8 = 39.27 := sorry

end split_evenly_rounding_l70_70193


namespace factorial_div_binom_equality_l70_70240

theorem factorial_div_binom_equality :
  (20! / 15!) = 20 * 19 * 18 * 17 * 16 ∧ (binom 20 5) = 15504 := by
suffices h1: (20! / 15!) = 20 * 19 * 18 * 17 * 16, from
suffices h2: (binom 20 5) = 20 * 19 * 18 * 17 * 16 / 120, from
suffices h3: (20 * 19 * 18 * 17 * 16) / 120 = 15504, from

have : binom 20 5 = (20 * 19 * 18 * 17 * 16) / 120,
  by rw [binom_eq_20C5],

have : (20 * 19 * 18 * 17 * 16) / 120 = 15504,
  by compute,

have : (20! / 15!) = 20 * 19 * 18 * 17 * 16,
  by compute,

done

-- sorry would be here strictly as a placeholder for the proofs
sorry

end factorial_div_binom_equality_l70_70240


namespace num_of_integer_pairs_l70_70699

def log6 (x : ℝ) : ℝ := log x / log 6

theorem num_of_integer_pairs :
  let f := log6
  let g (x : ℝ) := 90 + x - 6^90
  in ∑ i in finset.range (6^90 + 1), 
       if 90 + i - 6^90 ≤ f i then
         f i - (90 + i - 6^90)
       else 0 =
    1/2 * 6^180 - 7/10 * 6^90 + 456/5 :=
by
  sorry

end num_of_integer_pairs_l70_70699


namespace LisaNeedsMoreMarbles_l70_70864

theorem LisaNeedsMoreMarbles :
  let friends := 12
  let marbles := 40
  let required_marbles := (friends * (friends + 1)) / 2
  let additional_marbles := required_marbles - marbles
  additional_marbles = 38 :=
by
  let friends := 12
  let marbles := 40
  let required_marbles := (friends * (friends + 1)) / 2
  let additional_marbles := required_marbles - marbles
  have h1 : required_marbles = 78 := by
    calc (friends * (friends + 1)) / 2
      _ = (12 * 13) / 2 : by rfl
      _ = 156 / 2 : by rfl
      _ = 78 : by norm_num
  have h2 : additional_marbles = 38 := by
    calc required_marbles - marbles
      _ = 78 - 40 : by rw h1
      _ = 38 : by norm_num
  exact h2

end LisaNeedsMoreMarbles_l70_70864


namespace radius_relation_l70_70200

variables (a b : ℝ)
def c : ℝ := Real.sqrt (a^2 + b^2)
def r : ℝ := (a + b - c) / 2
def R : ℝ := c / 2
def ρ : ℝ := a + b - c

theorem radius_relation
  (a b : ℝ) :
  ρ = 2 * r :=
by
  -- proof is omitted, hence sorry
  sorry

end radius_relation_l70_70200


namespace length_longer_leg_of_smallest_triangle_l70_70676

theorem length_longer_leg_of_smallest_triangle :
  ∀ (a b c d e f g h i j k l m n o p q : ℝ), 
  (a = 16) → (b = a / 2) → (c = b * (real.sqrt 3)) → 
  (d = c / 2) → (e = d * (real.sqrt 3)) →
  (f = e / 2) → (g = f * (real.sqrt 3)) → 
  (h = g / 2) → (i = h * (real.sqrt 3)) → 
  (j = i / 2) → (k = j * (real.sqrt 3)) → 
  (l = k / 2) → (m = l * (real.sqrt 3)) → 
  (n = m / 2) → (o = n * (real.sqrt 3)) → 
  (p = o / 2) → (q = p * (real.sqrt 3)) → 
  q = 9 :=
begin
  sorry
end

end length_longer_leg_of_smallest_triangle_l70_70676


namespace parabola_circle_and_min_area_l70_70721

-- Define the conditions for the parabola and circle
def parabola (p : ℝ) (p_pos : p > 0) (x y : ℝ) : Prop :=
  x^2 = 2 * p * y

def circle (r : ℝ) (r_pos : r > 0) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2

-- Define the condition that the chord length is 4 and passes through the focus F
def chord_length_four (F : ℝ × ℝ) (length : ℝ) (length_cond : length = 4) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ A.1 ≠ B.1 ∧ ((F.1 - A.1)^2 + (F.2 - A.2)^2)^0.5 = length

-- Minimum area of the triangle condition
def min_area_triangle (F : ℝ × ℝ) (M : ℝ × ℝ) (min_area : ℝ) : Prop :=
  ∀ (A B : ℝ × ℝ),
    -- Line passing through F intersects parabola at A and B
    parabola 2 (by norm_num) A.1 A.2 ∧ parabola 2 (by norm_num) B.1 B.2 →
    -- Tangent at A intersects x-axis at M
    (M = (A.1 / 2, 0)) →
    -- Calculate area of triangle ABM
    let S = (1 / 2) * (dist A B) * (abs (A.1 / 2 * A.2 + 1) / (sqrt (1 + A.1^2 / 2))) in
    S = min_area

theorem parabola_circle_and_min_area 
(p r : ℝ) (p_pos : p > 0) (r_pos : r > 0) (length_cond : 4 = 4) (min_area : ℝ) :
  (parabola p p_pos, p = 2) ∧
  (circle r r_pos, r^2 = 5) ∧
  (min_area_triangle (0,1) (p/2,0) min_area, min_area = (8 * sqrt 3) / 9) :=
by
  sorry

end parabola_circle_and_min_area_l70_70721


namespace infinite_series_sum_l70_70257

theorem infinite_series_sum : (∑' n : ℕ, if n % 3 = 0 then 1 / (3 * 2^(((n - n % 3) / 3) + 1)) 
                                 else if n % 3 = 1 then -1 / (6 * 2^(((n - n % 3) / 3)))
                                 else -1 / (12 * 2^(((n - n % 3) / 3)))) = 1 / 72 :=
by
  sorry

end infinite_series_sum_l70_70257


namespace convert_polar_to_rectangular_l70_70252

theorem convert_polar_to_rectangular :
  (∀ (r θ : ℝ), r = 4 ∧ θ = (Real.pi / 4) →
    (r * Real.cos θ, r * Real.sin θ) = (2 * Real.sqrt 2, 2 * Real.sqrt 2)) :=
by
  intros r θ h
  cases h with hr hθ
  rw [hr, hθ]
  simp
  sorry

end convert_polar_to_rectangular_l70_70252


namespace sqrt_quadratic_condition_l70_70107

-- Define the condition under which the quadratic radical equation holds
theorem sqrt_quadratic_condition (x : ℝ) : sqrt ((x - 3)^2) = x - 3 ↔ x ≥ 3 :=
by
  -- Proof would go here, but we only need the statement for now
  sorry

end sqrt_quadratic_condition_l70_70107


namespace lisa_additional_marbles_l70_70875

theorem lisa_additional_marbles (n : ℕ) (m : ℕ) (s_n : ℕ) :
  n = 12 →
  m = 40 →
  (s_n = (list.sum (list.range (n + 1)))) →
  s_n - m = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp only [list.range_succ, list.sum_range_succ, nat.factorial, nat.succ_eq_add_one, nat.add_succ, mul_add, mul_one, mul_comm n]
  sorry

end lisa_additional_marbles_l70_70875


namespace truck_toll_is_accurate_l70_70987

def toll (R S : ℝ) (x : ℕ) : ℝ := R + S * (x - 2)

def axles (wheels : ℕ) (front_axle_wheels : ℕ) (other_axle_wheels : ℕ) : ℕ :=
  1 + (wheels - front_axle_wheels) / other_axle_wheels

def base_rate (weight : ℝ) : ℝ :=
  if weight ≤ 5 then 1.50
  else if weight ≤ 10 then 2.50
  else 4.00

def additional_rate (num_wheels_on_axles : List ℕ) : ℝ :=
  if num_wheels_on_axles.all (fun n => n = 2) then 0.60 else 0.80

noncomputable def truck_toll : ℝ :=
  let weight := 12
  let wheels := 18
  let front_axle_wheels := 2
  let other_axle_wheels := 4
  let num_wheels_on_axles := [front_axle_wheels, other_axle_wheels, other_axle_wheels, other_axle_wheels, other_axle_wheels]
  let R := base_rate weight
  let S := additional_rate num_wheels_on_axles
  let x := axles wheels front_axle_wheels other_axle_wheels
  toll R S x

theorem truck_toll_is_accurate : truck_toll = 6.40 := by
  sorry

end truck_toll_is_accurate_l70_70987


namespace sqrt_computation_l70_70245

theorem sqrt_computation : 
  Real.sqrt ((35 * 34 * 33 * 32) + Nat.factorial 4) = 1114 := by
sorry

end sqrt_computation_l70_70245


namespace find_m_l70_70380

theorem find_m (m : ℤ) (h : (-2)^(2 * m) = 2^(12 - m)) : m = 4 := by
  sorry

end find_m_l70_70380


namespace susan_vacation_length_l70_70924

def daily_pay (hourly_rate : ℕ) (hours_per_day : ℕ) : ℕ := hourly_rate * hours_per_day

def unpaid_vacation_days (missed_pay : ℕ) (daily_pay : ℕ) : ℕ := missed_pay / daily_pay

def total_vacation_days (paid_vacation : ℕ) (unpaid_vacation : ℕ) : ℕ := paid_vacation + unpaid_vacation

theorem susan_vacation_length :
  let hourly_rate := 15
  let hours_per_day := 8
  let missed_pay := 480
  let paid_vacation := 6 in
  total_vacation_days paid_vacation (unpaid_vacation_days missed_pay (daily_pay hourly_rate hours_per_day)) = 10 :=
by
  sorry

end susan_vacation_length_l70_70924


namespace polynomial_existence_and_uniqueness_l70_70702

theorem polynomial_existence_and_uniqueness :
  ∃ q : polynomial ℝ, (q.comp q = X * q + 2 * X^2) ∧ (q = -2 * X + 4) :=
by
  sorry

end polynomial_existence_and_uniqueness_l70_70702


namespace perfect_square_l70_70530

variable (n k l : ℕ)

theorem perfect_square (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m ^ 2 := by
  have h1 : (2 * l - n - k) * (2 * l - n + k) = (2 * l - n) ^ 2 - k ^ 2 := by sorry
  have h2 : k ^ 2 = 2 * l ^ 2 - n ^ 2 := by sorry
  have h3 : (2 * l - n) ^ 2 - (2 * l ^ 2 - n ^ 2) = 2 * (l - n) ^ 2 := by sorry
  have h4 : (2 * (l - n) ^ 2) / 2 = (l - n) ^ 2 := by sorry
  use (l - n)
  rw [h4]
  sorry

end perfect_square_l70_70530


namespace sixth_graders_more_than_seventh_l70_70101

theorem sixth_graders_more_than_seventh
  (bookstore_sells_pencils_in_whole_cents : True)
  (seventh_graders : ℕ)
  (sixth_graders : ℕ)
  (seventh_packs_payment : ℕ)
  (sixth_packs_payment : ℕ)
  (each_pack_contains_two_pencils : True)
  (seventh_graders_condition : seventh_graders = 25)
  (seventh_packs_payment_condition : seventh_packs_payment * seventh_graders = 275)
  (sixth_graders_condition : sixth_graders = 36 / 2)
  (sixth_packs_payment_condition : sixth_packs_payment * sixth_graders = 216) : 
  sixth_graders - seventh_graders = 7 := sorry

end sixth_graders_more_than_seventh_l70_70101


namespace sam_total_yellow_marbles_l70_70469

def sam_original_yellow_marbles : Float := 86.0
def sam_yellow_marbles_given_by_joan : Float := 25.0

theorem sam_total_yellow_marbles : sam_original_yellow_marbles + sam_yellow_marbles_given_by_joan = 111.0 := by
  sorry

end sam_total_yellow_marbles_l70_70469


namespace sine_interval_probability_l70_70598

theorem sine_interval_probability : 
  let x := (UniformContinuousRandomVariable (-1 : ℝ) 1) in
  (Prob (λ x, -((1 : ℝ)/2) ≤ sin ((π * x) / 4 ∧ sin ((π * x) / 4) ≤ (sqrt 2)/2)) = 5/6) :=
begin
  sorry
end

end sine_interval_probability_l70_70598


namespace anne_speed_l70_70230

-- Conditions
def time_hours : ℝ := 3
def distance_miles : ℝ := 6

-- Question with correct answer
theorem anne_speed : distance_miles / time_hours = 2 := by 
  sorry

end anne_speed_l70_70230


namespace math_proof_problem_l70_70385

noncomputable def problem1 (A : ℝ) : Prop :=
  (1/2) * real.cos (2 * A) = real.cos A ^ 2 - real.cos A 

noncomputable def problem2 (a b c : ℝ) (A B C : ℝ) : Prop := 
  a = 3 ∧ real.sin B = 2 * real.sin C → 
  1/2 * b * c * real.sin A = (3 * real.sqrt 3) / 2

theorem math_proof_problem (A B C a b c : ℝ) :
  problem1 A → 0 < A → A < real.pi → A = real.pi / 3 ∧
  problem2 a b c A B C := 
  sorry

end math_proof_problem_l70_70385


namespace probability_x_eq_y_cos_cos_l70_70616

theorem probability_x_eq_y_cos_cos (X Y : ℝ) (hX : -5 * π ≤ X ∧ X ≤ 5 * π) (hY : -5 * π ≤ Y ∧ Y ≤ 5 * π) 
(hcos : cos (cos X) = cos (cos Y)) : 
P(X = Y) = 11 / 100 :=
by sorry

end probability_x_eq_y_cos_cos_l70_70616


namespace ratio_of_beans_to_seitan_is_one_l70_70613

noncomputable def protein_ratio_equivalence 
  (total_dishes: ℕ)
  (beans_lentils: ℕ)
  (beans_seitan: ℕ)
  (include_lentils: ℕ)
  (only_protein_dishes: ℕ)
  (only_beans: ℕ)
  (only_seitan: ℕ)
  (half_only_protein: ℕ)
  (same_beans_seitan: Prop) : Prop :=
  total_dishes = 10 ∧
  beans_lentils = 2 ∧
  beans_seitan = 2 ∧
  include_lentils = 4 ∧
  half_only_protein = only_protein_dishes / 2 ∧
  same_beans_seitan = (only_beans = only_seitan) →
  only_beans / only_seitan = 1

theorem ratio_of_beans_to_seitan_is_one : protein_ratio_equivalence 10 2 2 4 4 2 2 2 (by rfl) :=
  sorry

end ratio_of_beans_to_seitan_is_one_l70_70613


namespace rectangle_bisectors_area_l70_70794

theorem rectangle_bisectors_area (a b : ℝ) (h : b > a) : 
  let S := (b - a) ^ 2 / 2 in
  area_of_quadrilateral_formed_by_bisectors a b = S := 
by
  sorry

end rectangle_bisectors_area_l70_70794


namespace solution_set_inequality_l70_70128

theorem solution_set_inequality : {x : ℝ | (x - 2) * (1 - 2 * x) ≥ 0} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 2} :=
by
  sorry  -- Proof to be provided

end solution_set_inequality_l70_70128


namespace quadratic_has_two_roots_l70_70338

variables {R : Type*} [LinearOrderedField R]

theorem quadratic_has_two_roots (a1 a2 a3 b1 b2 b3 : R) 
  (h1 : a1 * a2 * a3 = b1 * b2 * b3) (h2 : a1 * a2 * a3 > 1) : 
  (4 * a1^2 - 4 * b1 > 0) ∨ (4 * a2^2 - 4 * b2 > 0) ∨ (4 * a3^2 - 4 * b3 > 0) :=
sorry

end quadratic_has_two_roots_l70_70338


namespace pb_dot_oa_range_l70_70030

open Real

noncomputable def P : ℝ × ℝ := (1, sqrt 3)

-- Vector OA, OB, and OC have the same magnitude of 1 and their sum is zero
noncomputable def mag_eq_one 
  (O A B C : ℝ × ℝ) : Prop := 
  (dist O A = 1 ∧ dist O B = 1 ∧ dist O C = 1 ∧ 
  (mk_prod (O.1 + A.1 + B.1 + C.1, O.2 + A.2 + B.2 + C.2) = (0, 0)))

-- Define the vector PB and OA
def vec_PB (P B : ℝ × ℝ) : ℝ × ℝ := (B.1 - P.1, B.2 - P.2)
def vec_OA (O A : ℝ × ℝ) : ℝ × ℝ := (A.1 - O.1, A.2 - O.2)

-- Dot product definition
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The theorem to prove the range
theorem pb_dot_oa_range 
  (O A B C : ℝ × ℝ) (h : mag_eq_one O A B C) : 
  let PB := vec_PB P B
  let OA := vec_OA O A
  let dp := dot_prod PB OA
  -5/2 ≤ dp ∧ dp ≤ 3/2 := sorry

end pb_dot_oa_range_l70_70030


namespace convert_polar_to_rectangular_l70_70250

/-- The given point in polar coordinates -/
def polar_point : ℝ × ℝ := (4, Real.pi / 4)

/-- Convert the polar point to rectangular coordinates -/
def rectangular_coordinates (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular :
  rectangular_coordinates 4 (Real.pi / 4) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) :=
by
  sorry

end convert_polar_to_rectangular_l70_70250


namespace negation_of_odd_function_proposition_l70_70943

variables {f : ℝ → ℝ}

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem negation_of_odd_function_proposition :
  (¬ (∀ x, is_odd f → is_odd (λ x, f (-x)))) ↔ (∀ x, ¬is_odd f → ¬is_odd (λ x, f (-x))) :=
by
  sorry

end negation_of_odd_function_proposition_l70_70943


namespace parabola_vertex_on_x_axis_l70_70674

theorem parabola_vertex_on_x_axis (c : ℝ) : 
    (∃ h k, h = -3 ∧ k = 0 ∧ ∀ x, x^2 + 6 * x + c = x^2 + 6 * x + (c - (h^2)/4)) → c = 9 :=
by
    sorry

end parabola_vertex_on_x_axis_l70_70674


namespace cube_ratio_l70_70889

theorem cube_ratio (a x : ℝ) (AM MB : ℝ) (M N K1 : ℝ) (sqrt54 eight : ℝ):
  M * 2 = a →
  AM + MB = a →
  sqrt54 = Real.sqrt 54 →
  eight = 8 →
  AM = x →
  MB = a - x →
  MK1 = a →
  MN = a * (4 / 3) →
  (MK1 / MN) = sqrt54 / eight →
  (AM / MB) = 1 / 4 :=
by sorry

end cube_ratio_l70_70889


namespace find_abs_ab_l70_70480

theorem find_abs_ab {a b : ℤ} (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : ∃ m n : ℤ, (m ≠ n) ∧ (x : ℝ → x^3 + a*x^2 + b*x + 9*a = (x - m)^2 * (x - n)) ∧ (m, n ∈ ℤ)) :
  |a * b| = 96 :=
sorry

end find_abs_ab_l70_70480


namespace part_one_part_two_l70_70726

namespace GeometricSequenceProof

noncomputable def a := -2

def a_n (n : ℕ) : ℝ := 2^(n - 1)

def S_n (n : ℕ) : ℝ := a * (1 - a_n n) / (1 - 2)

def b_n (n : ℕ) : ℝ := (2 * n + 1) * real.logb 2 (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / b_n i

theorem part_one (n : ℕ) (hn : n > 0) :
  2 * S_n n = 2^(n + 1) + a ∧ (∃ k : ℕ, a_n k = 2^(k - 1)) := 
sorry

theorem part_two (n : ℕ) (hn : n > 0) :
  T_n n = n / (2 * n + 1) := 
sorry

end GeometricSequenceProof

end part_one_part_two_l70_70726


namespace expanded_polynomial_term_count_l70_70965

-- Definitions for the polynomial expressions involved.
def polynomial1 := [a_1, a_2, a_3]
def polynomial2 := [b_1, b_2, b_3, b_4]
def polynomial3 := [c_1, c_2, c_3, c_4, c_5]

-- The proof problem statement
theorem expanded_polynomial_term_count : 
  (polynomial1.length * polynomial2.length * polynomial3.length) = 60 :=
  sorry

end expanded_polynomial_term_count_l70_70965


namespace nth_equation_l70_70071

theorem nth_equation (n : ℕ) : 
  n^2 + (n + 1)^2 = (n * (n + 1) + 1)^2 - (n * (n + 1))^2 :=
by
  sorry

end nth_equation_l70_70071


namespace value_of_x_for_g_equals_g_inv_l70_70650

noncomputable def g (x : ℝ) : ℝ := 3 * x - 7
  
noncomputable def g_inv (x : ℝ) : ℝ := (x + 7) / 3
  
theorem value_of_x_for_g_equals_g_inv : ∃ x : ℝ, g x = g_inv x ∧ x = 3.5 :=
by
  sorry

end value_of_x_for_g_equals_g_inv_l70_70650


namespace man_receives_total_amount_l70_70206
noncomputable def total_amount_received : ℝ := 
  let itemA_price := 1300
  let itemB_price := 750
  let itemC_price := 1800
  
  let itemA_loss := 0.20 * itemA_price
  let itemB_loss := 0.15 * itemB_price
  let itemC_loss := 0.10 * itemC_price

  let itemA_selling_price := itemA_price - itemA_loss
  let itemB_selling_price := itemB_price - itemB_loss
  let itemC_selling_price := itemC_price - itemC_loss

  let vat_rate := 0.12
  let itemA_vat := vat_rate * itemA_selling_price
  let itemB_vat := vat_rate * itemB_selling_price
  let itemC_vat := vat_rate * itemC_selling_price

  let final_itemA := itemA_selling_price + itemA_vat
  let final_itemB := itemB_selling_price + itemB_vat
  let final_itemC := itemC_selling_price + itemC_vat

  final_itemA + final_itemB + final_itemC

theorem man_receives_total_amount :
  total_amount_received = 3693.2 := by
  sorry

end man_receives_total_amount_l70_70206


namespace pretty_vs_beautiful_coin_l70_70436

-- Define the problem setup and sequence limits as given
def P_n (n : ℕ) : ℕ := sorry  -- The number of ways Pretty Penny can make n cents

def B_n (n : ℕ) : ℕ := sorry  -- The number of ways Beautiful Bill can make n cents

noncomputable def limit_ratio (f g : ℕ → ℕ) : ℝ := sorry  -- A function to define the limit ratio

theorem pretty_vs_beautiful_coin {
  c : ℝ
  (lim_P_B : limit_ratio P_n B_n = c) 
} : 
  c = 20 :=
by sorry

end pretty_vs_beautiful_coin_l70_70436


namespace f_nondecreasing_on_01_l70_70043

open Set

-- Define the function f and its conditions
noncomputable def f (t: ℝ) (A: List (Set α)) : ℝ :=
  ∑ k in (Finset.range (A.length + 1)).filter (λ k, k > 0),
  ∑ (s : Multiset (Fin A.length)) 
    (h : s.card = k) 
    (hs : s < A.length), 
    (-1: ℤ) ^ (k - 1) * t ^ s.val.toFinset.sup (λ i, (A.nthIfInBounds i).card)

-- Statement: Prove that f is nondecreasing on [0, 1]
theorem f_nondecreasing_on_01 (A : List (Set α)) (hA : ∀ i ∈ A, (A.nthIfInBounds i).Nonempty) :
  MonotoneOn (λ t, f t A) (Icc 0 1) :=
by
  sorry

end f_nondecreasing_on_01_l70_70043


namespace hyperbola_center_l70_70279

theorem hyperbola_center :
  ∃ c : ℝ × ℝ, (c = (3, 4) ∧ ∀ x y : ℝ, 9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0 ↔ (x - 3)^2 / 4 - (y - 4)^2 / 1 = 1) :=
sorry

end hyperbola_center_l70_70279


namespace train_length_is_correct_l70_70611

noncomputable def speed_kmph : ℝ := 72
noncomputable def time_seconds : ℝ := 74.994
noncomputable def tunnel_length_m : ℝ := 1400
noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600
noncomputable def total_distance : ℝ := speed_mps * time_seconds
noncomputable def train_length : ℝ := total_distance - tunnel_length_m

theorem train_length_is_correct :
  train_length = 99.88 := by
  -- the proof will follow here
  sorry

end train_length_is_correct_l70_70611


namespace max_distance_and_perpendicular_line_l70_70729

noncomputable theory

def pointP : ℝ × ℝ := (-2, -1)

def line_l (λ : ℝ) : ℝ × ℝ → ℝ :=
  λ coord, (1 + 3 * λ) * coord.1 + (1 + λ) * coord.2 - 2 - 4 * λ

def pointQ : ℝ × ℝ := (1, 1)

def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem max_distance_and_perpendicular_line :
  (distance pointP pointQ = Real.sqrt 13) ∧
  (∀ p : ℝ × ℝ,
    let slopePQ := (p.2 - pointP.2) / (p.1 - pointP.1) in
    slopePQ = 2 / 3 →
    ∃ a b c : ℝ, a * p.1 + b * p.2 + c = 0 ∧ a/b = -b / (2/3)) :=
sorry

end max_distance_and_perpendicular_line_l70_70729


namespace mother_hubbard_children_l70_70227

theorem mother_hubbard_children :
  (∃ c : ℕ, (2 / 3 : ℚ) = c * (1 / 12 : ℚ)) → c = 8 :=
by
  sorry

end mother_hubbard_children_l70_70227


namespace abs_inequality_l70_70973

theorem abs_inequality (x : ℝ) (h : |x - 2| < 1) : 1 < x ∧ x < 3 := by
  sorry

end abs_inequality_l70_70973


namespace coeff_x2_term_l70_70280

theorem coeff_x2_term (a b c d e f : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : d = 6) (h5 : e = 7) (h6 : f = 8) :
    (a * f + b * e * 1 + c * d) = 82 := 
by
    sorry

end coeff_x2_term_l70_70280


namespace joes_monthly_income_l70_70189

def monthly_income (tax : ℝ) (rate : ℝ) : ℝ := 
  tax / rate

theorem joes_monthly_income : 
  monthly_income 848 0.4 = 2120 := 
  by sorry

end joes_monthly_income_l70_70189


namespace weight_difference_tote_laptop_l70_70822

variable (T B L : ℝ)

-- Given conditions
def weight_tote := (T = 8)
def empty_briefcase := (T = 2 * B)
def full_briefcase := (2 * T = 2 * (T))
def weight_papers := (1 / 6 * (2 * T))

theorem weight_difference_tote_laptop (h1 : weight_tote) (h2 : empty_briefcase) (h3 : full_briefcase) (h4 : weight_papers) :
  L - T = 1.33 :=
by
  sorry

end weight_difference_tote_laptop_l70_70822


namespace point_on_parabola_l70_70921

noncomputable def parabola_focus : Point := (0, 1 / 8)

theorem point_on_parabola (x y : ℝ) 
  (hx : y = 2 * x ^ 2)
  (hy_pos : 0 ≤ x ∧ 0 ≤ y)
  (hfocus_dist : dist (x, y) parabola_focus = 1 / 4) : 
  (x, y) = (1 / 4, 1 / 8) := 
sorry

end point_on_parabola_l70_70921


namespace sum_floor_log2_l70_70690

def floor_log2 (n : ℕ) : ℕ := Nat.log2 n

theorem sum_floor_log2 : (∑ N in Finset.range 1024, floor_log2 (N + 1)) = 8204 :=
by
  sorry

end sum_floor_log2_l70_70690


namespace no_three_consecutive_inc_dec_l70_70028

theorem no_three_consecutive_inc_dec {α : Type*} [DecidableEq α] (s : Finset α) : 
  s = {1, 2, 3, 4, 5, 6} →
  ∃! permut : List α, permut ∈ s.permutations ∧
    ¬ (∀ i, i < permut.length - 2 → (permut.get (i + 2) > permut.get (i + 1) ∧ permut.get (i + 1) > permut.get i) ∨
       (permut.get (i + 2) < permut.get (i + 1) ∧ permut.get (i + 1) < permut.get i)) 
    ∧ (permut.length = 6) :=
by
  have h : (∃ permut : List α, permut ∈ s.permutations ∧ (¬ (∀ i, i < permut.length - 2 → (permut.get (i + 2) > permut.get (i + 1) ∧ permut.get (i + 1) > permut.get i) ∨ (permut.get (i + 2) < permut.get (i + 1) ∧ permut.get (i + 1) < permut.get i)) ∧ (permut.length = 6)) :=
    sorry,
  exact h


end no_three_consecutive_inc_dec_l70_70028


namespace negation_correct_l70_70622

variable (Dragon : Type) (Faery : Type) 
variable (Magical : Faery → Prop) (BreatheFire : Dragon → Prop)

-- Conditions
axiom faeries_all_magical : ∀ (f : Faery), Magical f
axiom no_dragons_magical : ∀ (d : Dragon), ¬Magical d
axiom all_dragons_breathe_fire : ∀ (d : Dragon), BreatheFire d
axiom some_dragons_do_not_breathe_fire : ∃ (d : Dragon), ¬BreatheFire d

-- Proof problem
theorem negation_correct : (∃ (d : Dragon), ¬BreatheFire d) ↔ ¬(∀ (d : Dragon), BreatheFire d) := sorry

end negation_correct_l70_70622


namespace can_use_bisection_method_l70_70617

noncomputable def f1 (x : ℝ) : ℝ := x^2
noncomputable def f2 (x : ℝ) : ℝ := x⁻¹
noncomputable def f3 (x : ℝ) : ℝ := abs x
noncomputable def f4 (x : ℝ) : ℝ := x^3

theorem can_use_bisection_method : ∃ (a b : ℝ), a < b ∧ (f4 a) * (f4 b) < 0 := 
sorry

end can_use_bisection_method_l70_70617


namespace slope_range_l70_70348

theorem slope_range (A B : ℝ × ℝ) (P : ℝ × ℝ) :
  A = (1, -2) → B = (2, 1) → P = (0, -1) →
  {θ : ℝ | (∃ k, k = Real.atan θ ∧ -1 ≤ k ∧ k ≤ 1)}
  = {θ : ℝ | (0 ≤ θ ∧ θ ≤ Real.pi / 4) ∨ (3 * Real.pi / 4 ≤ θ ∧ θ ≤ Real.pi)} :=
by
  intros hA hB hP
  sorry

end slope_range_l70_70348


namespace total_cards_is_56_l70_70898

-- Let n be the number of Pokemon cards each person has
def n : Nat := 14

-- Let k be the number of people
def k : Nat := 4

-- Total number of Pokemon cards
def total_cards : Nat := n * k

-- Prove that the total number of Pokemon cards is 56
theorem total_cards_is_56 : total_cards = 56 := by
  sorry

end total_cards_is_56_l70_70898


namespace solve_k_values_l70_70190

def has_positive_integer_solution (k : ℕ) : Prop :=
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = k * a * b * c

def infinitely_many_solutions (k : ℕ) : Prop :=
  ∃ (a b c : ℕ → ℕ), (∀ n, a n > 0 ∧ b n > 0 ∧ c n > 0 ∧ a n^2 + b n^2 + c n^2 = k * a n * b n * c n) ∧
  (∀ n, ∃ x y: ℤ, x^2 + y^2 = (a n * b n))

theorem solve_k_values :
  ∃ k : ℕ, (k = 1 ∨ k = 3) ∧ has_positive_integer_solution k ∧ infinitely_many_solutions k :=
sorry

end solve_k_values_l70_70190


namespace weight_difference_l70_70820

-- Define the relevant weights
def Karen_tote : ℝ := 8
def Kevin_empty_briefcase : ℝ := Karen_tote / 2
def Kevin_full_briefcase : ℝ := 2 * Karen_tote
def Kevin_contents : ℝ := Kevin_full_briefcase - Kevin_empty_briefcase
def Kevin_work_papers : ℝ := Kevin_contents / 6
def Kevin_laptop : ℝ := Kevin_contents - Kevin_work_papers

-- The main theorem statement
theorem weight_difference : Kevin_laptop - Karen_tote = 2 :=
by
  -- Proof would go here, but is omitted as per instructions
  sorry

end weight_difference_l70_70820


namespace statement_A_statement_B_statement_C_statement_D_l70_70310

def sequence_a : ℕ → ℤ
| 1       := 8
| 2       := 1
| (n + 2) := if n % 2 = 0 then -sequence_a n else sequence_a n - 2

def T (n : ℕ) : ℤ :=
  (finset.range n).sum (λ i, sequence_a (i + 1))

theorem statement_A : sequence_a 11 = -2 := by sorry

theorem statement_B (n : ℕ) : T (2 * n) = -n^2 + 9 * n + 1 := by sorry

theorem statement_C : T 99 = -2049 := by sorry

theorem statement_D : ∀ n : ℕ, T n ≤ 21 := by sorry

end statement_A_statement_B_statement_C_statement_D_l70_70310


namespace find_radius_of_circle_l70_70102

noncomputable def central_angle := 150
noncomputable def arc_length := 5 * Real.pi
noncomputable def arc_length_formula (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 180) * Real.pi * r

theorem find_radius_of_circle :
  (∃ r : ℝ, arc_length_formula central_angle r = arc_length) ↔ 6 = 6 :=
by  
  sorry

end find_radius_of_circle_l70_70102


namespace polynomial_has_root_l70_70956

theorem polynomial_has_root {a b c d : ℝ} 
  (h : a * c = 2 * b + 2 * d) : 
  ∃ x : ℝ, (x^2 + a * x + b = 0) ∨ (x^2 + c * x + d = 0) :=
by 
  sorry

end polynomial_has_root_l70_70956


namespace length_BC_fraction_AD_l70_70087

-- Given
variables {A B C D : Type*} [AddCommGroup D] [Module ℝ D]
variables (A B C D : D)
variables (AB BD AC CD AD BC : ℝ)

-- Conditions
def segment_AD := A + D
def segment_BD := B + D
def segment_AB := A + B
def segment_CD := C + D
def segment_AC := A + C
def relation_AB_BD : AB = 3 * BD := sorry
def relation_AC_CD : AC = 5 * CD := sorry

-- Proof
theorem length_BC_fraction_AD :
  BC = (1/12) * AD :=
sorry

end length_BC_fraction_AD_l70_70087


namespace find_product_l70_70808

def a : ℕ := 4
def g : ℕ := 8
def d : ℕ := 10

theorem find_product (A B C D E F : ℕ) (hA : A % 2 = 0) (hB : B % 3 = 0) (hC : C % 4 = 0) 
  (hD : D % 5 = 0) (hE : E % 6 = 0) (hF : F % 7 = 0) :
  a * g * d = 320 :=
by
  sorry

end find_product_l70_70808


namespace initial_books_count_l70_70456

-- Definitions in conditions
def books_sold : ℕ := 42
def books_left : ℕ := 66

-- The theorem to prove the initial books count
theorem initial_books_count (initial_books : ℕ) : initial_books = books_sold + books_left :=
  by sorry

end initial_books_count_l70_70456


namespace quadratic_solutions_l70_70916

-- Define the equation x^2 - 6x + 8 = 0
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

-- Lean statement for the equivalence of solutions
theorem quadratic_solutions : ∀ x : ℝ, quadratic_eq x ↔ x = 2 ∨ x = 4 :=
by
  intro x
  dsimp [quadratic_eq]
  sorry

end quadratic_solutions_l70_70916


namespace cost_of_white_car_l70_70066

variable (W : ℝ)
variable (red_cars white_cars : ℕ)
variable (rent_red rent_white : ℝ)
variable (rented_hours : ℝ)
variable (total_earnings : ℝ)

theorem cost_of_white_car 
  (h1 : red_cars = 3)
  (h2 : white_cars = 2) 
  (h3 : rent_red = 3)
  (h4 : rented_hours = 3)
  (h5 : total_earnings = 2340) :
  2 * W * (rented_hours * 60) + 3 * rent_red * (rented_hours * 60) = total_earnings → 
  W = 2 :=
by 
  sorry

end cost_of_white_car_l70_70066


namespace sum_of_every_second_term_l70_70603

def seq : ℕ → ℝ := λ n, n
def sum_seq (n : ℕ) : ℝ := ∑ i in finset.range n, seq i

def sum_all : ℝ := 5010
def condition1 : 2000 > 0 := by sorry
def condition2 : ∀ n, seq (n + 1) = seq n + 1 := by sorry
def condition3 : sum_seq 2000 = sum_all := by sorry

theorem sum_of_every_second_term : 
  ∀ (seq : ℕ → ℝ) (sum_all : ℝ), 
  (
    (condition1 -- 2000 > 0
    ∧ condition2 -- ∀ n, seq (n + 1) = seq n + 1
    ∧ condition3 -- sum_seq 2000 = sum_all
    ) →
    (∑ i in finset.filter (λ n, n % 2 = 1) (finset.range 2000), seq i) = 3005
  ) := 
by sorry

end sum_of_every_second_term_l70_70603


namespace fill_tank_in_30_min_l70_70205

theorem fill_tank_in_30_min :
  let T := 30 in
  let rate_A := 1 / 60 in
  let rate_B := 1 / 40 in
  let filled_by_B := (T / 2) * rate_B in
  let filled_by_A_and_B := (T / 2) * (rate_A + rate_B) in
  filled_by_B + filled_by_A_and_B = 1 :=
by
  -- Proof step placeholder
  sorry

end fill_tank_in_30_min_l70_70205


namespace square_perimeter_l70_70964

theorem square_perimeter (p1 p2 : ℝ) (h1 : p1 = 24) (h2 : p2 = 32) :
  let a1 := (p1 / 4)^2,
      a2 := (p2 / 4)^2,
      a3 := a1 + a2,
      s := Real.sqrt a3 in
  4 * s = 40 :=
by
  sorry

end square_perimeter_l70_70964


namespace expected_value_correct_l70_70599

-- Definitions for the problem
def brakePoint (s : List ℕ) (n : ℕ) : Prop :=
  ∀ i ∈ list.range n, i + 1 ∈ s.take n

def correctPartition (perm : List ℕ) : List (List ℕ) :=
  -- This is a placeholder for the actual function that returns the correct partition
  sorry

-- Expected value of correct partitions for permutations of {1, 2, ..., 7}
noncomputable def expected_value : ℚ :=
  let perms := Finset.univ.val.perms in
  let k (σ : List ℕ) : ℕ := (correctPartition σ).length in
  (finset.card perms : ℚ)⁻¹ * ∑ σ in perms, k σ

-- Assertion that the expected value is 151/105
theorem expected_value_correct : expected_value = 151 / 105 :=
by
  -- Placeholder proof
  sorry

end expected_value_correct_l70_70599


namespace tan_A_l70_70029

variables {α : Type*} [linear_ordered_field α] [archimedean α]

def triangle_ABC (A B C : α × α) :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (dist A B) = 15 ∧
  (dist B C) = 20 ∧
  (∃ θ : α, (tan θ) = 3/4) ∧
  ((∠ A B C) = π / 2)

theorem tan_A (A B C : α × α) (h : triangle_ABC A B C) :
  ∃ θ, tan θ = 4 / 3 := 
sorry

end tan_A_l70_70029


namespace function_property_l70_70274

theorem function_property (f : ℤ → ℤ) (h : ∀ a b c : ℤ, a + b + c = 0 → f(a) + f(b) + f(c) = a^2 + b^2 + c^2) :
  ∃ c : ℤ, ∀ x : ℤ, f(x) = x^2 + c * x :=
sorry

end function_property_l70_70274


namespace range_of_a_l70_70322

theorem range_of_a (z : ℂ) (a : ℝ)
  (h_imaginary : ¬(∃ r θ : ℝ, z = r * (complex.cos θ + complex.sin θ * complex.I) ∧ r ≥ 0))
  (h_real_root : ∃ x : ℝ, x^2 - 2 * a * x + 1 - 3 * a = 0 ∧ x = (z + 3/(2 * z)).re) :
  (a ≥ (real.sqrt 13 - 3) / 2 ∨ a ≤ -(real.sqrt 13 + 3) / 2) :=
sorry -- proof goes here

end range_of_a_l70_70322


namespace place_tokens_l70_70081

theorem place_tokens (initial_tokens : Fin 50 → Fin 50 → Bool) :
  ∃ (new_tokens : Fin 50 → Fin 50 → Bool), 
    (∑ i j, if new_tokens i j then 1 else 0) ≤ 99 ∧
    ∀ i, even (∑ j, if initial_tokens i j ∨ new_tokens i j then 1 else 0) ∧
    ∀ j, even (∑ i, if initial_tokens i j ∨ new_tokens i j then 1 else 0) :=
by sorry

end place_tokens_l70_70081


namespace circle_tangency_l70_70286

noncomputable def circle_eq (x y a : ℝ) : ℝ := 
  (x - a)^2 + (y - 2 * a)^2

theorem circle_tangency (a : ℝ) (r : ℝ) (h1 : (circle_eq 3 2 a) = r^2)
  (h2 : (2 * a - 2 * a + 5) / (real.sqrt 5) = r)
  (h3 : 2 * a * 1 + 2 * -r = 5) :
  circle_eq x y a = r^2 :=
by
  sorry

end circle_tangency_l70_70286


namespace perfect_square_l70_70531

variable (n k l : ℕ)

theorem perfect_square (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m ^ 2 := by
  have h1 : (2 * l - n - k) * (2 * l - n + k) = (2 * l - n) ^ 2 - k ^ 2 := by sorry
  have h2 : k ^ 2 = 2 * l ^ 2 - n ^ 2 := by sorry
  have h3 : (2 * l - n) ^ 2 - (2 * l ^ 2 - n ^ 2) = 2 * (l - n) ^ 2 := by sorry
  have h4 : (2 * (l - n) ^ 2) / 2 = (l - n) ^ 2 := by sorry
  use (l - n)
  rw [h4]
  sorry

end perfect_square_l70_70531


namespace determine_a_n_l70_70311

noncomputable def sequence_a_n_pos (n : ℕ) : ℝ :=
sorry -- Definition of the positive sequence

def a_2_eq_3_a_1 (a1 a2 : ℝ) : Prop :=
a2 = 3 * a1

def sum_S_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = ∑ i in finset.range n, a i

def sqrt_S_n_AP (S : ℕ → ℝ) (d : ℝ) :=
∀ n : ℕ, sqrt (S (n+1)) = sqrt (S n) + d

theorem determine_a_n (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_a2 : a_2_eq_3_a_1 (a 0) (a 1)) (h_S : sum_S_n a S) (h_AP : sqrt_S_n_AP S (1 / 2)) :
  a 0 = 1 / 4 ∧ (∀ n, n ≥ 1 → a n = (2 * (n + 1) - 1) / 4) :=
sorry

end determine_a_n_l70_70311


namespace yellow_ball_probability_l70_70401

noncomputable def probability_drawing_yellow_ball (total_balls yellow_balls : ℕ) : ℚ :=
  yellow_balls / total_balls

theorem yellow_ball_probability :
  let total_balls := 8 in
  let yellow_balls := 5 in
  probability_drawing_yellow_ball total_balls yellow_balls = (5 : ℚ) / (8 : ℚ) :=
by
  sorry

end yellow_ball_probability_l70_70401


namespace smaller_number_l70_70131

theorem smaller_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) : y = 15 :=
sorry

end smaller_number_l70_70131


namespace number_of_six_digit_with_sum_51_l70_70360

open Finset

/-- A digit is a number between 0 and 9 inclusive.-/
def is_digit (n : ℕ) : Prop :=
  n >= 0 ∧ n <= 9

/-- Friendly notation for digit sums -/
def digit_sum (n : Fin 6 → ℕ) : ℕ :=
  (Finset.univ.sum n)

def is_six_digit_with_sum_51 (n : Fin 6 → ℕ) : Prop :=
  (∀ i, is_digit (n i)) ∧ (digit_sum n = 51)

/-- There are exactly 56 six-digit numbers such that the sum of their digits is 51. -/
theorem number_of_six_digit_with_sum_51 : 
  card {n : Fin 6 → ℕ // is_six_digit_with_sum_51 n} = 56 :=
by
  sorry

end number_of_six_digit_with_sum_51_l70_70360


namespace compare_rd_levels_and_probability_l70_70589

-- Define counts for successes and failures in teams A and B
def team_a_scores := [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
def team_b_scores := [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1]

-- Average and variance calculations for team A
def avg_score_a := (∑ i in team_a_scores, i) / (team_a_scores.length : ℝ)
def var_score_a := ∑ i in team_a_scores, (i - avg_score_a)^2 / (team_a_scores.length : ℝ)

-- Average and variance calculations for team B
def avg_score_b := (∑ i in team_b_scores, i) / (team_b_scores.length : ℝ)
def var_score_b := ∑ i in team_b_scores, (i - avg_score_b)^2 / (team_b_scores.length : ℝ)

-- Define the event E (exactly one team succeeds)
def event_e := [⟨1, 0⟩, ⟨0, 1⟩, ⟨1, 0⟩, ⟨0, 1⟩, ⟨1, 0⟩, ⟨1, 0⟩, ⟨0, 1⟩]

-- Probability of event E
def prob_e := (event_e.length : ℝ) / 15

theorem compare_rd_levels_and_probability :
    avg_score_a = 2 / 3 ∧ var_score_a = 2 / 9 ∧
    avg_score_b = 3 / 5 ∧ var_score_b = 6 / 25 ∧
    avg_score_a > avg_score_b ∧ var_score_a < var_score_b ∧
    prob_e = 7 / 15 := 
by 
    sorry

end compare_rd_levels_and_probability_l70_70589


namespace collinear_points_l70_70464

structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨-1, -2⟩
def B : Point := ⟨2, -1⟩
def C : Point := ⟨8, 1⟩

theorem collinear_points (A B C : Point) : 
  (A = ⟨-1, -2⟩) → (B = ⟨2, -1⟩) → (C = ⟨8, 1⟩) →
   ∃ k : ℝ, B.x - A.x = k * (C.x - A.x) ∧ B.y - A.y = k * (C.y - A.y) :=
begin
  sorry
end

end collinear_points_l70_70464


namespace transformed_sum_l70_70605

variable (n : ℕ) (s : ℝ)
variable (x : ℕ → ℝ)
hypothesis (h_sum : ∑ i in Finset.range n, x i = s)

theorem transformed_sum :
  (∑ i in Finset.range n, 3 * (x i + 30) - 10) = 3 * s + 80 * n :=
sorry

end transformed_sum_l70_70605


namespace jerry_school_grade_total_students_l70_70814

variable (jerry_best : ℕ) (jerry_worst : ℕ)

theorem jerry_school_grade_total_students (h_best : jerry_best = 60) (h_worst : jerry_worst = 60) :
  jerry_best + jerry_worst - 1 = 119 :=
by {
  have h : jerry_best = jerry_worst := by rw [h_best, h_worst],
  rw ← h,
  linarith 
  }

end jerry_school_grade_total_students_l70_70814


namespace bisected_by_midsegments_implies_parallelogram_l70_70462

theorem bisected_by_midsegments_implies_parallelogram
  (quadrilateral : Type)
  [isConvexQuadrilateral quadrilateral]
  (A B C D E F : quadrilateral)
  (mid_segment1 mid_segment2 : (quadrilateral → ℝ) → ℝ)
  (area : quadrilateral → ℝ)
  (h1 : isMidpoint A B E)
  (h2 : isMidpoint C D F)
  (h3 : mid_segment1 = (λ q, area q))
  (h4 : mid_segment2 = (λ q, area q))
  (h5 : ∀ E F, mid_segment1 E F = 1/2 * area quadrilateral)
  (h6 : ∀ E F, mid_segment2 E F = 1/2 * area quadrilateral) :
  isParallelogram quadrilateral :=
by
  sorry

end bisected_by_midsegments_implies_parallelogram_l70_70462


namespace length_of_angle_bisector_AM_l70_70412

-- Definitions based on conditions in the problem
def AB : ℝ := 8
def AC : ℝ := 6
def angle_BAC : ℝ := 60 * Real.pi / 180

-- The main statement we want to prove
theorem length_of_angle_bisector_AM :
  let AM := ((24 : ℝ) * Real.sqrt 3) / 7
  in AM = sorry :=
by sorry

end length_of_angle_bisector_AM_l70_70412


namespace perfect_square_expression_l70_70521

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, m^2 = (2 * l - n - k) * (2 * l - n + k) / 2 :=
by 
  sorry

end perfect_square_expression_l70_70521


namespace quadratic_inequality_l70_70375

theorem quadratic_inequality (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_l70_70375


namespace exists_quad_root_l70_70962

theorem exists_quad_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (∃ x, x^2 + a * x + b = 0) ∨ (∃ x, x^2 + c * x + d = 0) :=
sorry

end exists_quad_root_l70_70962


namespace no_infinite_sequence_satisfying_condition_l70_70414

theorem no_infinite_sequence_satisfying_condition :
  ¬ ∃ (x : ℕ → ℝ), (∀ n, x n > 0) ∧ (∀ n, x (n + 2) = real.sqrt (x (n + 1)) - real.sqrt (x n)) :=
by {
  -- The proof is omitted as it's not required by the task.
  sorry
}

end no_infinite_sequence_satisfying_condition_l70_70414


namespace calc_pow_results_l70_70629

theorem calc_pow_results :
  2 ^ 345 + 3 ^ 5 * 3 ^ 3 = 2 ^ 345 + 6561 := by
  have h1 : 3 ^ 5 * 3 ^ 3 = 3 ^ (5 + 3) := by sorry -- Power multiplication rule
  have h2 : 3 ^ 8 = 6561 := by sorry     -- Evaluate 3^8
  rw [h1]                                 -- Substitute 3 ^ 5 * 3 ^ 3 with 3 ^ 8
  rw [h2]                                 -- Substitute 3 ^ 8 with 6561
  sorry                                    -- Combine the results

end calc_pow_results_l70_70629


namespace perfect_square_of_expression_l70_70528

theorem perfect_square_of_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 := 
by 
  sorry

end perfect_square_of_expression_l70_70528


namespace c_100_eq_one_third_l70_70217

variable (c : ℕ → ℚ)

noncomputable def sequence_cn (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 1 / 3
  else (2 - sequence_cn (n - 1)) / (3 * sequence_cn (n - 2))

theorem c_100_eq_one_third {
  c_1 : c 1 = 1,
  c_2 : c 2 = 1 / 3,
  h : ∀ n ≥ 3, c n = (2 - c (n - 1)) / (3 * c (n - 2))
} : c 100 = 1 / 3 :=
sorry

end c_100_eq_one_third_l70_70217


namespace degree_polynomial_is_4_l70_70108

variable (a b : ℕ)

def term1 : ℚ := (2 / 3) * a * b^2
def term2 : ℚ := (4 / 3) * a^3 * b
def term3 : ℚ := 1 / 3

def polynomial : ℚ := term1 + term2 + term3

theorem degree_polynomial_is_4 : ∀ a b : ℕ, polynomial a b = term1 a b + term2 a b + term3 a b →
  4 = max (max (1 + 2) (3 + 1)) 0 :=
by sorry

end degree_polynomial_is_4_l70_70108


namespace slopes_product_no_circle_MN_A_l70_70725

-- Define the equation of the ellipse E and the specific points A and B
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the point P which lies on the ellipse
def P (x0 y0 : ℝ) : Prop := ellipse_eq x0 y0 ∧ x0 ≠ -2 ∧ x0 ≠ 2

-- Prove the product of the slopes of lines PA and PB
theorem slopes_product (x0 y0 : ℝ) (hP : P x0 y0) : 
  (y0 / (x0 + 2)) * (y0 / (x0 - 2)) = -1 / 4 := sorry

-- Define point Q
def Q : ℝ × ℝ := (-1, 0)

-- Define points M and N which are intersections of line and ellipse
def MN_line (t y : ℝ) : ℝ := t * y - 1

-- Prove there is no circle with diameter MN passing through A
theorem no_circle_MN_A (t : ℝ) : 
  ¬ ∃ M N : ℝ × ℝ, ellipse_eq M.1 M.2 ∧ ellipse_eq N.1 N.2 ∧
  (∃ x1 y1 x2 y2, (M = (x1, y1) ∧ N = (x2, y2)) ∧
  (MN_line t y1 = x1 ∧ MN_line t y2 = x2) ∧ 
  ((x1 + 2) * (x2 + 2) + y1 * y2 = 0)) := sorry

end slopes_product_no_circle_MN_A_l70_70725


namespace abs_inequality_l70_70972

theorem abs_inequality (x : ℝ) (h : |x - 2| < 1) : 1 < x ∧ x < 3 := by
  sorry

end abs_inequality_l70_70972


namespace captain_times_l70_70578

theorem captain_times (a b c T k : ℝ) (ha : a = c + 3) (hb : b + c = 15)
  (hT : T / 10 = a + b + c + 25) (hk : k * c = 160) :
  let k_val := 20 
  in a = 11 → b = 7 → c = 8 → k = k_val → ka = 200 ∧ kc = 160 := by
  assume h1 h2 h3 h4
  sorry

end captain_times_l70_70578


namespace union_A_B_inter_A_B_diff_U_A_U_B_subset_A_C_l70_70755

universe u

open Set

def U := @univ ℝ
def A := { x : ℝ | 3 ≤ x ∧ x < 10 }
def B := { x : ℝ | 2 < x ∧ x ≤ 7 }
def C (a : ℝ) := { x : ℝ | x > a }

theorem union_A_B : A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
by sorry

theorem inter_A_B : A ∩ B = { x : ℝ | 3 ≤ x ∧ x ≤ 7 } :=
by sorry

theorem diff_U_A_U_B : (U \ A) ∩ (U \ B) = { x : ℝ | x ≤ 2 } ∪ { x : ℝ | 10 ≤ x } :=
by sorry

theorem subset_A_C (a : ℝ) (h : A ⊆ C a) : a < 3 :=
by sorry

end union_A_B_inter_A_B_diff_U_A_U_B_subset_A_C_l70_70755


namespace bricks_needed_l70_70569

def brick := (length : ℝ) × (width : ℝ) × (height : ℝ)
def wall := (length : ℝ) × (width : ℝ) × (height : ℝ)

def volume (b : brick) : ℝ :=
  b.1 * b.2 * b.3

def volume (w : wall) : ℝ :=
  w.1 * w.2 * w.3

def number_of_bricks_needed (w : wall) (b : brick) : ℕ :=
  (volume w / volume b).ceil.to_nat

theorem bricks_needed :
  let brick := (25, 11.25, 6)
  let wall := (400, 200, 25)
  number_of_bricks_needed wall brick = 1186 :=
by
  sorry

end bricks_needed_l70_70569


namespace geometric_sequence_sum_l70_70408

theorem geometric_sequence_sum (S : ℕ → ℝ) (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n : ℕ, n > 0 → S n = 2^n + a) →
  (S 1 = 2 + a) →
  (∀ n ≥ 2, a_n n = S n - S (n - 1)) →
  (a_n 1 = 1) →
  a = -1 :=
by
  sorry

end geometric_sequence_sum_l70_70408


namespace incorrect_statement_C_l70_70894

open Real

def vertex_of_parabola (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem incorrect_statement_C :
  (vertex_of_parabola 1 = -2) ∧
  (∀ x, vertex_of_parabola x = vertex_of_parabola (2 - x)) ∧
  (∃ x > 1, derivative vertex_of_parabola x > 0) ∧
  (∀ x, 0 ≤ (x - 1)^2)
  → "When x > 1, y decreases as x increases." = false :=
by 
  sorry

end incorrect_statement_C_l70_70894


namespace geometric_series_sum_l70_70237

theorem geometric_series_sum (a r : ℚ) (h_a : a = 1) (h_r : r = 1 / 3) : 
  (∑' n : ℕ, a * r ^ n) = 3 / 2 := 
by
  sorry

end geometric_series_sum_l70_70237


namespace sum_of_integers_l70_70691

theorem sum_of_integers (n m : ℕ) (h1 : n * (n + 1) = 300) (h2 : m * (m + 1) * (m + 2) = 300) : 
  n + (n + 1) + m + (m + 1) + (m + 2) = 49 := 
by sorry

end sum_of_integers_l70_70691


namespace at_least_one_root_l70_70951

theorem at_least_one_root 
  (a b c d : ℝ)
  (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
sorry

end at_least_one_root_l70_70951


namespace like_reading_books_l70_70391

theorem like_reading_books (total groupBoth groupSongs : ℕ) (h1 : total = 100) (h2 : groupBoth = 20) (h3 : groupSongs = 70) : 
  ∃ groupReading : ℕ, groupReading = total - groupSongs + groupBoth :=
by
  use 50
  rw [h1, h2, h3]
  norm_num

end like_reading_books_l70_70391


namespace total_students_in_class_l70_70390

theorem total_students_in_class (F G B N T : ℕ)
  (hF : F = 41)
  (hG : G = 22)
  (hB : B = 9)
  (hN : N = 15)
  (hT : T = (F + G - B) + N) :
  T = 69 :=
by
  -- This is a theorem statement, proof is intentionally omitted.
  sorry

end total_students_in_class_l70_70390


namespace smallest_n_for_congruence_l70_70673

theorem smallest_n_for_congruence :
  ∃ n : ℕ, n > 0 ∧ 7 ^ n % 4 = n ^ 7 % 4 ∧ ∀ m : ℕ, (m > 0 ∧ m < n → ¬ (7 ^ m % 4 = m ^ 7 % 4)) :=
by
  sorry

end smallest_n_for_congruence_l70_70673


namespace find_half_sum_enter_exit_times_l70_70197

def car_position (t : ℝ) : ℝ × ℝ :=
  (t, 0)

def storm_center_position (t : ℝ) : ℝ × ℝ :=
  (3 / 2 * t, 150 - 3 / 2 * t)

def storm_radius : ℝ :=
  60

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

axiom enter_and_exit_time (t1 t2 : ℝ) :
  (distance (car_position t1) (storm_center_position t1) = storm_radius) ∧
  (distance (car_position t2) (storm_center_position t2) = storm_radius) ∧
  (t1 ≤ t2)

theorem find_half_sum_enter_exit_times (t1 t2 : ℝ) (h : enter_and_exit_time t1 t2) :
  (t1 + t2) / 2 = 212.4 :=
sorry

end find_half_sum_enter_exit_times_l70_70197


namespace distinct_sum_l70_70834

theorem distinct_sum (a b c d e : ℤ) (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 0)
  (h2 : a ≠ b) (h3 : a ≠ c) (h4 : a ≠ d) (h5 : a ≠ e) (h6 : b ≠ c) (h7 : b ≠ d) (h8 : b ≠ e) (h9 : c ≠ d) (h10 : c ≠ e) (h11 : d ≠ e) :
  a + b + c + d + e = 35 :=
sorry

end distinct_sum_l70_70834


namespace red_notebook_cost_4_l70_70449

noncomputable def cost_of_red_notebooks (total_spent : ℤ) (total_notebooks : ℤ) (red_notebooks : ℤ) (green_notebooks : ℤ) (cost_green : ℤ) (cost_blue : ℤ) : ℤ :=
  let green_total := green_notebooks * cost_green
  let blue_notebooks := total_notebooks - red_notebooks - green_notebooks
  let blue_total := blue_notebooks * cost_blue
  let remaining := total_spent - (green_total + blue_total)
  remaining / red_notebooks

theorem red_notebook_cost_4
  (total_spent : ℤ) (total_notebooks : ℤ) (red_notebooks : ℤ) (green_notebooks : ℤ) (cost_green : ℤ) (cost_blue : ℤ)
  (h_total_spent : total_spent = 37)
  (h_total_notebooks : total_notebooks = 12)
  (h_red_notebooks : red_notebooks = 3)
  (h_green_notebooks : green_notebooks = 2)
  (h_cost_green : cost_green = 2)
  (h_cost_blue : cost_blue = 3) :
  cost_of_red_notebooks total_spent total_notebooks red_notebooks green_notebooks cost_green cost_blue = 4 :=
by
  rw [h_total_spent, h_total_notebooks, h_red_notebooks, h_green_notebooks, h_cost_green, h_cost_blue]
  unfold cost_of_red_notebooks
  simp only [mul_comm, mul_assoc, add_comm, add_assoc, sub_eq_add_neg, int.coe_nat_mul, int.coe_nat_add, int.coe_nat_sub]
  norm_num
  sorry

end red_notebook_cost_4_l70_70449


namespace determine_conic_section_l70_70258

-- Definitions of |y - 3| and sqrt((x + 4)^2 + (y - 1)^2)
def lhs (y : ℝ) : ℝ := abs (y - 3)
def rhs (x y : ℝ) : ℝ := real.sqrt ((x + 4)^2 + (y - 1)^2)

-- Given condition expresssion
def given_expression (x y : ℝ) : Prop := lhs y = rhs x y

-- Statement to validate the conic type
def is_parabola (x y : ℝ) (h : given_expression x y) : Prop := 
  ∃ (a b c : ℝ), a ≠ 0 ∧ y = a * x^2 + b * x + c

-- Approximation of the problem statement in Lean
theorem determine_conic_section (x y : ℝ) (h : given_expression x y) : is_parabola x y h :=
sorry

end determine_conic_section_l70_70258


namespace smallest_n_S_is_integer_l70_70049

-- Necessary definitions based on the problem conditions
def sum_reciprocal_even_digits : ℚ := (1 / 2) + (1 / 4) + (1 / 6) + (1 / 8)

def calc_S (n : ℕ) : ℚ := n * 10^(n - 1) * sum_reciprocal_even_digits

-- The statement of the problem
theorem smallest_n_S_is_integer :
  (∃ n : ℕ, 0 < n ∧ calc_S n ∈ ℚ ∧ ∀ m : ℕ, 0 < m → m < n → calc_S m ∉ ℚ) :=
begin
  sorry -- Skipping the proof, as it is not required
end

end smallest_n_S_is_integer_l70_70049


namespace zoo_ticket_sales_l70_70192

theorem zoo_ticket_sales (A K : ℕ) (h1 : A + K = 254) (h2 : 28 * A + 12 * K = 3864) : K = 202 :=
by {
  sorry
}

end zoo_ticket_sales_l70_70192


namespace prime_solution_unique_l70_70668

theorem prime_solution_unique:
  ∃ p q r : ℕ, (prime p ∧ prime q ∧ prime r ∧ p + q^2 = r^4 ∧ p = 7 ∧ q = 3 ∧ r = 2) ∧
  (∀ p' q' r' : ℕ, prime p' ∧ prime q' ∧ prime r' ∧ p' + q'^2 = r'^4 → (p' = 7 ∧ q' = 3 ∧ r' = 2)) :=
by {
  sorry
}

end prime_solution_unique_l70_70668


namespace total_surface_area_l70_70292

variable (S p r : ℝ)

-- Condition: The area of the base of the prism is S
def area_base (S : ℝ) : Prop := S = p * r

-- Problem: Prove the total surface area of the prism given the conditions
theorem total_surface_area (h_base_area : area_base S) : 
  6 * S = 6 * S := sorry

end total_surface_area_l70_70292


namespace find_y_of_equations_l70_70365

theorem find_y_of_equations (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x = 1 + 1 / y) (h2 : y = 2 + 1 / x) : 
  y = 1 + Real.sqrt 3 ∨ y = 1 - Real.sqrt 3 :=
by
  sorry

end find_y_of_equations_l70_70365


namespace triangle_angle_inconsistency_l70_70977

theorem triangle_angle_inconsistency : 
  ∀ (L R T : ℝ), L + R + T = 180 ∧ L = 2 * R ∧ R = 60 → T = 0 → false :=
by
  intros L R T h1 h2,
  obtain ⟨h_sum, h_left, h_right⟩ := h1,
  rw h_right at *,
  rw h_left at *,
  linarith

end triangle_angle_inconsistency_l70_70977


namespace concurrency_of_lines_l70_70937

-- Define the geometry and conditions
variables (A B C D E F P X Y Z : Type)
variables [Incircle A B C D E F] [PointInTriangle P A B C] [PointOnIncircle X P A] [PointOnIncircle Y P B] [PointOnIncircle Z P C]

-- Define the concurrency of lines
theorem concurrency_of_lines : Concurrent D X E Y F Z := by
  sorry

end concurrency_of_lines_l70_70937


namespace minimum_flips_to_defeat_hydra_l70_70704

/-- Smallest number of flips to defeat a hydra with 100 necks.-/
theorem minimum_flips_to_defeat_hydra (N : ℕ) (H : hydra) :
  (∀ H : hydra, is_defeated H N) ↔ N = 10 :=
begin
  sorry,
end

end minimum_flips_to_defeat_hydra_l70_70704


namespace extra_mangoes_l70_70877

-- Definitions of the conditions
def original_price_per_mango := 433.33 / 130
def new_price_per_mango := original_price_per_mango - 0.10 * original_price_per_mango
def mangoes_at_original_price := 360 / original_price_per_mango
def mangoes_at_new_price := 360 / new_price_per_mango

-- Statement to be proved
theorem extra_mangoes : mangoes_at_new_price - mangoes_at_original_price = 12 := 
by {
  sorry
}

end extra_mangoes_l70_70877


namespace find_top_angle_l70_70979

theorem find_top_angle 
  (sum_of_angles : ∀ (α β γ : ℝ), α + β + γ = 250) 
  (left_is_twice_right : ∀ (α β : ℝ), α = 2 * β) 
  (right_angle_is_60 : ∀ (β : ℝ), β = 60) :
  ∃ γ : ℝ, γ = 70 :=
by
  -- Assume the variables for the angles
  obtain ⟨α, β, γ, h_sum, h_left, h_right⟩ := ⟨_, _, _, sum_of_angles, left_is_twice_right, right_angle_is_60⟩
  -- Your proof here
  sorry

end find_top_angle_l70_70979


namespace total_pokemon_cards_l70_70899

theorem total_pokemon_cards : 
  let n := 14 in
  let total_cards := 4 * n in
  total_cards = 56 :=
by
  let n := 14
  let total_cards := 4 * n
  show total_cards = 56
  sorry

end total_pokemon_cards_l70_70899


namespace combined_weight_of_candles_l70_70268

theorem combined_weight_of_candles :
  (∀ (candles: ℕ), ∀ (weight_per_candle: ℕ),
    (weight_per_candle = 8 + 1) →
    (candles = 10 - 3) →
    (candles * weight_per_candle = 63)) :=
by
  intros
  sorry

end combined_weight_of_candles_l70_70268


namespace odd_function_rhombus_diagonals_bisect_l70_70639

-- First Problem
theorem odd_function (f : ℝ → ℝ) (h : ∀ x, f (-x) = -f x) : (∀ x, f x = x ^ 3) → true :=
by
  assume h1 : (∀ x, f x = x ^ 3)
  sorry

-- Second Problem
theorem rhombus_diagonals_bisect (h : ∀ (P : Type*) [add_comm_group P] [module ℝ P],
  ∀ (a b c d : P), a + b = c + d → a ≠ b → c ≠ d → true) : true :=
by
  sorry

end odd_function_rhombus_diagonals_bisect_l70_70639


namespace LisaNeedsMoreMarbles_l70_70867

theorem LisaNeedsMoreMarbles :
  let friends := 12
  let marbles := 40
  let required_marbles := (friends * (friends + 1)) / 2
  let additional_marbles := required_marbles - marbles
  additional_marbles = 38 :=
by
  let friends := 12
  let marbles := 40
  let required_marbles := (friends * (friends + 1)) / 2
  let additional_marbles := required_marbles - marbles
  have h1 : required_marbles = 78 := by
    calc (friends * (friends + 1)) / 2
      _ = (12 * 13) / 2 : by rfl
      _ = 156 / 2 : by rfl
      _ = 78 : by norm_num
  have h2 : additional_marbles = 38 := by
    calc required_marbles - marbles
      _ = 78 - 40 : by rw h1
      _ = 38 : by norm_num
  exact h2

end LisaNeedsMoreMarbles_l70_70867


namespace quadratic_inequality_solution_l70_70373

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
by
  sorry

end quadratic_inequality_solution_l70_70373


namespace lisa_needs_additional_marbles_l70_70860

theorem lisa_needs_additional_marbles
  (friends : ℕ) (initial_marbles : ℕ) (total_required_marbles : ℕ) :
  friends = 12 ∧ initial_marbles = 40 ∧ total_required_marbles = (friends * (friends + 1)) / 2 →
  total_required_marbles - initial_marbles = 38 :=
by
  sorry

end lisa_needs_additional_marbles_l70_70860


namespace correct_statements_l70_70170

noncomputable theory
open ProbabilityTheory

-- Definitions based on given conditions
def statement_A (A B : Set Ω) (pA pB : ℝ) [MeasurableSet A] [MeasurableSet B] : Prop :=
  A ⊆ B ∧ pA = 0.3 ∧ pB = 0.6 ∧ P[B | A] = 1

def statement_B (ξ : ℝ → ℝ) (delta : ℝ) : Prop :=
  (NormalDist 2 delta).cdf 4 = 0.84 ∧ (NormalDist 2 delta).pdf 2 < (NormalDist 2 delta).pdf 4 ∧ ¬ ((NormalDist 2 delta).cdf 2 = 0.16)

def statement_C (r : ℝ) : Prop :=
  abs r ≈ 1 → strong_linear_relationship r

def statement_D (residuals : List ℝ) : Prop :=
  width_of_residual_band residuals ↑ indicates_worse_regression_effect

theorem correct_statements (A B : Set Ω) (pA pB : ℝ) (ξ : ℝ → ℝ) (delta r : ℝ) (residuals : List ℝ)
  [MeasurableSet A] [MeasurableSet B] :
  (statement_A A B pA pB) ∧ (¬ (statement_B ξ delta)) ∧ (statement_C r) ∧ (¬ (statement_D residuals)) :=
by
  sorry

end correct_statements_l70_70170


namespace range_of_k_l70_70114

variables {A B : Type} [linear_order A] [linear_order B]

def inverse_prop_function (k : A) (x : A) : B := (k + 4) / x

def points_condition (x1 x2 : A) (y1 y2 : B) : Prop := x1 < 0 ∧ 0 < x2 ∧ y1 > y2

theorem range_of_k (k : A) (x1 x2 : A) (y1 y2 : B) :
  (k ≠ -4) →
  points_condition x1 x2 y1 y2 →
  (inverse_prop_function k x1) = y1 →
  (inverse_prop_function k x2) = y2 →
  k < -4 :=
by
  intro hk points_cond hy1 hy2
  sorry

end range_of_k_l70_70114


namespace find_a_2016_l70_70810

noncomputable def a (n : ℕ) : ℕ := sorry

axiom condition_1 : a 4 = 1
axiom condition_2 : a 11 = 9
axiom condition_3 : ∀ n : ℕ, a n + a (n+1) + a (n+2) = 15

theorem find_a_2016 : a 2016 = 5 := sorry

end find_a_2016_l70_70810


namespace largest_angle_of_convex_hexagon_l70_70497

noncomputable def largest_angle (angles : Fin 6 → ℝ) (consecutive : ∀ i : Fin 5, angles i + 1 = angles (i + 1)) (sum_eq_720 : (∑ i, angles i) = 720) : ℝ :=
  sorry

theorem largest_angle_of_convex_hexagon (angles : Fin 6 → ℝ) (consecutive : ∀ i : Fin 5, angles i + 1 = angles (i + 1)) (sum_eq_720 : (∑ i, angles i) = 720) :
  largest_angle angles consecutive sum_eq_720 = 122.5 :=
  sorry

end largest_angle_of_convex_hexagon_l70_70497


namespace number_is_2250_l70_70539

-- Question: Prove that x = 2250 given the condition.
theorem number_is_2250 (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
sorry

end number_is_2250_l70_70539


namespace limit_of_n_squared_I_n_l70_70044

noncomputable def alpha : ℝ := sorry  -- This represents the solution to |x| = e^{-x}
def I_n (n : ℕ) : ℝ := ∫ x in 0..alpha, (x * exp (-n * x) + alpha * x^(n - 1))

theorem limit_of_n_squared_I_n :
  ∃ α, (0 < α ∧ α < 1) ∧ ∀ n : ℕ, I_n n = ∫ x in 0..α, (x * exp (-n * x) + α * x^(n-1)) ∧
  (lim (λ n, n^2 * I_n n) at_top = 1) :=
sorry

end limit_of_n_squared_I_n_l70_70044


namespace angle_AEB_is_90_l70_70890

noncomputable theory

-- Define the points A, B, C, D, M, E
variables {A B C D M E : Type*} [Point A] [Point B] [Point C] [Point D] [Point M] [Point E]
variables [Circle A B] [Chord C D] [Midpoint M A B] [Intersection M C D]

-- The condition that point E lies on the semicircle with diameter CD
variables (semicircle_CD : Is_semicircle C D E)

-- The condition that line ME is perpendicular to CD
variables (ME_perp_CD : Perp ME CD)

-- Prove that the angle AEB is 90 degrees
theorem angle_AEB_is_90 :
  angle A E B = 90 :=
sorry    -- proof to be filled in here

end angle_AEB_is_90_l70_70890


namespace ellipse_equation_1_ellipse_equation_2_l70_70291

-- Proof Problem 1
theorem ellipse_equation_1 (x y : ℝ) 
  (foci_condition : (x+2) * (x+2) + y*y + (x-2) * (x-2) + y*y = 36) :
  x^2 / 9 + y^2 / 5 = 1 :=
sorry

-- Proof Problem 2
theorem ellipse_equation_2 (x y : ℝ)
  (foci_condition : (x^2 + (y+5)^2 = 0) ∧ (x^2 + (y-5)^2 = 0))
  (point_on_ellipse : 3^2 / 15 + 4^2 / (15 + 25) = 1) :
  y^2 / 40 + x^2 / 15 = 1 :=
sorry

end ellipse_equation_1_ellipse_equation_2_l70_70291


namespace total_capacity_l70_70777

def eight_liters : ℝ := 8
def percentage : ℝ := 0.20
def num_containers : ℕ := 40

theorem total_capacity (h : eight_liters = percentage * capacity) :
  40 * (eight_liters / percentage) = 1600 := sorry

end total_capacity_l70_70777


namespace problem_statement_l70_70642

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement :
  (∀ x ∈ set.Ioo 0 (π / 2), cos x * (deriv^[2] f x) + sin x * f x < 0) →
  f (π / 6) > (sqrt 3) * f (π / 3) :=
begin
  sorry
end

end problem_statement_l70_70642


namespace max_value_f_l70_70713

-- Define the function f(x) as the minimum of \( x^2 \), \( 6 - x \), and \( 2x + 15 \)
def f (x : ℝ) : ℝ := min (x^2) (min (6 - x) (2 * x + 15))

-- Theorem stating the maximum value of the function f(x)
theorem max_value_f : ∃ x : ℝ, f x = 9 :=
  sorry

end max_value_f_l70_70713


namespace T_n_formula_l70_70733

def a_n (n : ℕ) : ℕ := 3 * n - 1
def b_n (n : ℕ) : ℕ := 2 ^ n
def T_n (n : ℕ) : ℕ := (Finset.range n).sum (λ k => a_n (k + 1) * b_n (k + 1))

theorem T_n_formula (n : ℕ) : T_n n = 8 - 8 * 2 ^ n + 3 * n * 2 ^ (n + 1) :=
by 
  sorry

end T_n_formula_l70_70733


namespace sum_of_repeating_decimal_digits_l70_70488

theorem sum_of_repeating_decimal_digits (c : ℕ → ℕ) (m : ℕ) :
  (1 / 98^2 : ℚ) = 0.\overline{(λ i, c (m-i-1))} →
  (∑ i in finset.range m, c i) = 900 :=
sorry

end sum_of_repeating_decimal_digits_l70_70488


namespace find_ordered_pair_l70_70994

theorem find_ordered_pair (a b : ℚ) :
  a • (⟨2, 3⟩ : ℚ × ℚ) + b • (⟨-2, 5⟩ : ℚ × ℚ) = (⟨10, -8⟩ : ℚ × ℚ) →
  (a, b) = (17 / 8, -23 / 8) :=
by
  intro h
  sorry

end find_ordered_pair_l70_70994


namespace sum_of_first_n_terms_geom_sequence_l70_70828

theorem sum_of_first_n_terms_geom_sequence (a₁ q : ℚ) (S : ℕ → ℚ)
  (h : ∀ n, S n = a₁ * (1 - q^n) / (1 - q))
  (h_ratio : S 4 / S 2 = 3) :
  S 6 / S 4 = 7 / 3 :=
by
  sorry

end sum_of_first_n_terms_geom_sequence_l70_70828


namespace student_correct_answers_l70_70221

-- Definitions based on the conditions
def total_questions : ℕ := 100
def score (correct incorrect : ℕ) : ℕ := correct - 2 * incorrect
def studentScore : ℕ := 73

-- Main theorem to prove
theorem student_correct_answers (C I : ℕ) (h1 : C + I = total_questions) (h2 : score C I = studentScore) : C = 91 :=
by
  sorry

end student_correct_answers_l70_70221


namespace parabola_y_intercepts_l70_70352

theorem parabola_y_intercepts : ∃ y1 y2 : ℝ, (3 * y1^2 - 6 * y1 + 2 = 0) ∧ (3 * y2^2 - 6 * y2 + 2 = 0) ∧ (y1 ≠ y2) :=
by 
  sorry

end parabola_y_intercepts_l70_70352


namespace g_eq_g_inv_iff_l70_70647

def g (x : ℝ) : ℝ := 3 * x - 7

def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_iff (x : ℝ) : g x = g_inv x ↔ x = 7 / 2 :=
by {
  sorry
}

end g_eq_g_inv_iff_l70_70647


namespace at_least_one_root_l70_70949

theorem at_least_one_root 
  (a b c d : ℝ)
  (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
sorry

end at_least_one_root_l70_70949


namespace solve_quadratic_l70_70910

theorem solve_quadratic : ∀ x : ℝ, x ^ 2 - 6 * x + 8 = 0 ↔ x = 2 ∨ x = 4 := by
  sorry

end solve_quadratic_l70_70910


namespace problem_min_abc_l70_70059

open Real

theorem problem_min_abc : 
  ∀ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 ∧ (a ≤ b ∧ b ≤ c) ∧ (c ≤ 3 * a) → 
  abc = min abc 
:=
by
  sorry

end problem_min_abc_l70_70059


namespace unique_a_for_set_A_l70_70604

def A (a : ℝ) : Set ℝ := {a^2, 2 - a, 4}

theorem unique_a_for_set_A (a : ℝ) : A a = {x : ℝ // x = a^2 ∨ x = 2 - a ∨ x = 4} → a = -1 :=
by
  sorry

end unique_a_for_set_A_l70_70604


namespace silver_cost_l70_70817

theorem silver_cost (S : ℝ) : 
  (1.5 * S) + (3 * 50 * S) = 3030 → S = 20 :=
by
  intro h
  sorry

end silver_cost_l70_70817


namespace find_c_l70_70320

-- Definitions
def is_root (x c : ℝ) : Prop := x^2 - 3*x + c = 0

-- Main statement
theorem find_c (c : ℝ) (h : is_root 1 c) : c = 2 :=
sorry

end find_c_l70_70320


namespace coefficient_of_x5_in_expansion_zero_l70_70693

noncomputable def binomial_expansion (f : ℚ[X]) (g : ℚ[X]) (n : ℕ) : ℚ[X] :=
∑ k in finset.range (n + 1), (nat.choose n k : ℚ) * (f ^ (n - k)) * (g ^ k)

noncomputable def problem_expr := (X^3 / 3 - 3 / X^2)^10 * X^2

theorem coefficient_of_x5_in_expansion_zero :
  polynomial.coeff (problem_expr) 5 = 0 :=
sorry

end coefficient_of_x5_in_expansion_zero_l70_70693


namespace ruby_siblings_l70_70392

structure Child :=
  (name : String)
  (eye_color : String)
  (hair_color : String)

def children : List Child :=
[
  {name := "Mason", eye_color := "Green", hair_color := "Red"},
  {name := "Ruby", eye_color := "Brown", hair_color := "Blonde"},
  {name := "Fiona", eye_color := "Brown", hair_color := "Red"},
  {name := "Leo", eye_color := "Green", hair_color := "Blonde"},
  {name := "Ivy", eye_color := "Green", hair_color := "Red"},
  {name := "Carlos", eye_color := "Green", hair_color := "Blonde"}
]

def is_sibling_group (c1 c2 c3 : Child) : Prop :=
  (c1.eye_color = c2.eye_color ∨ c1.hair_color = c2.hair_color) ∧
  (c2.eye_color = c3.eye_color ∨ c2.hair_color = c3.hair_color) ∧
  (c1.eye_color = c3.eye_color ∨ c1.hair_color = c3.hair_color)

theorem ruby_siblings :
  ∃ (c1 c2 : Child), 
    c1.name ≠ "Ruby" ∧ c2.name ≠ "Ruby" ∧
    c1 ≠ c2 ∧
    is_sibling_group {name := "Ruby", eye_color := "Brown", hair_color := "Blonde"} c1 c2 ∧
    ((c1.name = "Leo" ∧ c2.name = "Carlos") ∨ (c1.name = "Carlos" ∧ c2.name = "Leo")) :=
by
  sorry

end ruby_siblings_l70_70392


namespace ratio_of_sums_l70_70759

noncomputable theory

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (a1 d : α) (n : ℕ) : α :=
  n * (2 * a1 + (n - 1) * d) / 2

def Sn (a : ℕ → α) (n : ℕ) : α := ∑ i in finset.range n, a i
def Tn (b : ℕ → α) (n : ℕ) : α := ∑ i in finset.range n, b i

theorem ratio_of_sums (a b : ℕ → α) (S T : ℕ → α) (n : ℕ) :
  (∀ n, S n = Sn a n) →
  (∀ n, T n = Tn b n) →
  (∀ n, S n / T n = (↑(3 * n) + 1) / (↑n + 3)) →
  (a2_b20_a7_b15 : ℕ → α) : 
  a2_b20_a7_b15 = (a 1 + a 21) / (b 1 + b 21) :=
by
  intros hS hT hRatio
  have key : (∑ i in finset.range 21, a i) / (∑ i in finset.range 21, b i) = ((3 * 21 + 1) : α) / (21 + 3) :=
    hRatio 21
  -- continuing with the proof
  sorry

end ratio_of_sums_l70_70759


namespace probability_x_plus_y_le_six_l70_70210

theorem probability_x_plus_y_le_six :
  let area_rectangle := 4 * 8 in
  let area_triangle := (1 / 2 : ℚ) * 6 * 6 in
  let probability := area_triangle / area_rectangle in
  probability = 9 / 16 :=
by
  let area_rectangle : ℚ := 4 * 8
  let area_triangle : ℚ := (1 / 2 : ℚ) * 6 * 6
  let probability : ℚ := area_triangle / area_rectangle
  have h : probability = 9 / 16 := by sorry
  exact h

end probability_x_plus_y_le_six_l70_70210


namespace down_payment_l70_70901

theorem down_payment {total_loan : ℕ} {monthly_payment : ℕ} {years : ℕ} (h1 : total_loan = 46000) (h2 : monthly_payment = 600) (h3 : years = 5):
  total_loan - (years * 12 * monthly_payment) = 10000 := by
  sorry

end down_payment_l70_70901


namespace proof_problem_l70_70298

theorem proof_problem (x y : ℤ) (h1 : 2 ^ x = 16 ^ (y + 2)) (h2 : 27 ^ y = 3 ^ (x - 8)) : x + 2 * y = 8 := by
  sorry

end proof_problem_l70_70298


namespace ratio_squared_equals_product_of_segments_l70_70623

variables {A B C P E F K J : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]
variables [AddCommGroup E] [AddCommGroup F] [AddCommGroup K] [AddCommGroup J]

-- Given conditions
variable (hP : OnCircumcircle P A B C)
variable (hE : ∃ E, (LineThrough A B).Meet (LineThrough C P) = E)
variable (hF : ∃ F, (LineThrough A C).Meet (LineThrough B P) = F)
variable (hK : ∃ K, (PerpendicularBisector A B).Meet (LineThrough A C) = K)
variable (hJ : ∃ J, (PerpendicularBisector A C).Meet (LineThrough A B) = J)

-- Prove the required relationship
theorem ratio_squared_equals_product_of_segments (hP : OnCircumcircle P A B C)
    (hE : ∃ E, (LineThrough A B).Meet (LineThrough C P) = E)
    (hF : ∃ F, (LineThrough A C).Meet (LineThrough B P) = F)
    (hK : ∃ K, (PerpendicularBisector A B).Meet (LineThrough A C) = K)
    (hJ : ∃ J, (PerpendicularBisector A C).Meet (LineThrough A B) = J) :
    (CE / BF)^2 = (AJ * JE) / (AK * KF) :=
by
  sorry

end ratio_squared_equals_product_of_segments_l70_70623


namespace coeff_x_squared_l70_70283

def P (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 5
def Q (x : ℝ) : ℝ := 6 * x^2 + 7 * x + 8

theorem coeff_x_squared :
  let coeff : ℝ := 82 in
  ∀ x : ℝ, (P x * Q x).coeff 2 = coeff :=
sorry

end coeff_x_squared_l70_70283


namespace find_remainder_l70_70209

theorem find_remainder :
  let number := 220080
  let sum_of_555_and_445 := 555 + 445
  let difference_between_555_and_445 := 555 - 445
  let quotient := 2 * difference_between_555_and_445
  ∃ remainder, number = sum_of_555_and_445 * quotient + remainder ∧ remainder = 80 :=
by
  let number := 220080
  let sum_of_555_and_445 := 555 + 445
  let difference_between_555_and_445 := 555 - 445
  let quotient := 2 * difference_between_555_and_445
  use 80
  sorry

end find_remainder_l70_70209


namespace solve_quadratic_l70_70914

theorem solve_quadratic :
    ∀ x : ℝ, x^2 - 6*x + 8 = 0 ↔ (x = 2 ∨ x = 4) :=
by
  intros x
  sorry

end solve_quadratic_l70_70914


namespace degree_of_f_plus_g_is_three_l70_70922

noncomputable def f (z : ℂ) (a0 a1 a2 a3 : ℂ) : ℂ := a3 * z^3 + a2 * z^2 + a1 * z + a0
noncomputable def g (z : ℂ) (b0 b1 b2 : ℂ) : ℂ := b2 * z^2 + b1 * z + b0

theorem degree_of_f_plus_g_is_three
  (a0 a1 a2 a3 b0 b1 b2 : ℂ)
  (ha3 : a3 ≠ 0)
  (hb2 : b2 ≠ 0)
  (z : ℂ) :
  (f z a0 a1 a2 a3 + g z b0 b1 b2).degree = 3 :=
sorry

end degree_of_f_plus_g_is_three_l70_70922


namespace minimum_roots_of_derivative_l70_70122

-- Define the polynomial P(x) satisfying the condition
def P (x : ℂ) : ℂ := sorry

-- Define the number n (assuming n is a natural number)
variable (n : ℕ)

-- Assume P(x^2) has 2n + 1 distinct roots
axiom P_x2_roots : ∃ (s : Finset ℂ), s.card = 2 * n + 1 ∧ ∀ x ∈ s, P (x^2) = 0

-- Statement to prove: the minimum number of distinct roots of P'(x) is n
theorem minimum_roots_of_derivative {P : ℂ → ℂ} (h : ∃ (s : Finset ℂ), s.card = 2 * n + 1 ∧ ∀ x ∈ s, P (x^2) = 0) :
  ∃ (t : Finset ℂ), t.card = n ∧ ∀ x ∈ t, derivative P x = 0 :=
sorry

end minimum_roots_of_derivative_l70_70122


namespace range_of_angle_B_l70_70398

theorem range_of_angle_B (a b c : ℝ) (A B C : ℝ) (h : ℝ)
  (h1 : 0 < A) (h2 : A < π / 2)
  (h3 : 0 < B) (h4 : B < π / 2)
  (h5 : 0 < C) (h6 : C < π / 2)
  (h7 : a^2 - b^2 = b * c)
  (h8 : a = sqrt (b^2 + b * c))
  (h9 : A + B + C = π)
  (h10 : c = 4)
  : (π / 6 < B ∧ B < π / 4) ∧ (sqrt 3 < h ∧ h < 4) :=
by
  sorry

end range_of_angle_B_l70_70398


namespace poker_tournament_groupings_l70_70640

theorem poker_tournament_groupings :
  let players := 9
  let rounds := 4
  let groups_per_round := 3
  let all_groupings := 20160
  ∃ (grouper : Fin players → Fin rounds → Fin groups_per_round),
    (∀ i j : Fin players, ∃ r : Fin rounds, grouper i r ≠ grouper j r) ∧
    all_groupings
sorry

end poker_tournament_groupings_l70_70640


namespace distance_BC_l70_70859

theorem distance_BC (A B C : Point)
  (hAB : dist A B = 2)
  (hAC : dist A C = 4)
  (hAngleBAC : ∠A B C = 60) : dist B C = 2 * Real.sqrt 3 := by
  sorry

end distance_BC_l70_70859


namespace sale_price_of_article_l70_70125

noncomputable def cost_price : ℝ := 540.35
noncomputable def profit_rate : ℝ := 0.14
noncomputable def tax_rate : ℝ := 0.10

noncomputable def selling_price_before_tax (cp : ℝ) (pr : ℝ) : ℝ :=
  cp * (1 + pr)

noncomputable def sale_price_with_tax (sp : ℝ) (tr : ℝ) : ℝ :=
  sp * (1 + tr)

theorem sale_price_of_article :
  sale_price_with_tax (selling_price_before_tax cost_price profit_rate) tax_rate = 677.60 :=
by
  calc
    let sp := selling_price_before_tax cost_price profit_rate
    show sale_price_with_tax sp tax_rate = 677.60
    sorry

end sale_price_of_article_l70_70125


namespace inverse_sum_l70_70040

noncomputable def g (x : ℝ) : ℝ :=
if x < 15 then 2 * x + 4 else 3 * x - 1

theorem inverse_sum :
  g⁻¹ (10) + g⁻¹ (50) = 20 :=
sorry

end inverse_sum_l70_70040


namespace total_amount_spent_correct_l70_70513

-- Define the conditions as constants.
def cost_of_meal : ℝ := 60.50
def tip_percentage : ℝ := 0.20
def state_tax_percentage : ℝ := 0.05
def city_tax_percentage : ℝ := 0.03
def surcharge_percentage : ℝ := 0.015

-- Define the correct answer.
def correct_total_amount_spent : ℝ := 78.42

theorem total_amount_spent_correct :
  let tip := cost_of_meal * tip_percentage
  let state_tax := cost_of_meal * state_tax_percentage
  let city_tax := cost_of_meal * city_tax_percentage
  let total_tax := state_tax + city_tax
  let subtotal_before_surcharge := cost_of_meal + total_tax
  let surcharge := subtotal_before_surcharge * surcharge_percentage
  let total_amount_spent := cost_of_meal + total_tax + surcharge + tip
  in total_amount_spent = correct_total_amount_spent :=
by
  sorry

end total_amount_spent_correct_l70_70513


namespace standard_equation_of_hyperbola_ratio_on_complementary_slopes_l70_70752

namespace HyperbolaProof

noncomputable def hyperbola_equation_standard (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
(e : ℝ) (h₂ : e = Real.sqrt 5) (d : ℝ) (h₃ : d = 2) : Prop :=
  ∃ (a b : ℝ), 
  (a = 1 ∧ b = 2 ∧ (∀ x y : ℝ, (x ^ 2) - (y ^ 2 / b ^ 2) = 1))

/-- Prove that the standard form of the hyperbola with the given conditions is x² - y² / 4 = 1 -/
theorem standard_equation_of_hyperbola (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
(e : ℝ) (h2 : e = Real.sqrt 5) (d : ℝ) (h3 : d = 2) : 
hyperbola_equation_standard a b h₀ h₁ e h₂ d h₃ := 
sorry

noncomputable def ratio_complementary_slopes (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
(e : ℝ) (h₂ : e = Real.sqrt 5) (d : ℝ) (h₃ : d = 2) 
(M : ℝ × ℝ) (hx : M.1 = 1 / 4) 
(MA MD ME MB : ℝ) 
(slopeAB slopeDE : ℝ) (h₄ : slopeAB + slopeDE = 0) : Prop :=
  ∀ x y : ℝ, (|MA| / |MD|) = (|ME| / |MB|)

/-- Prove that given the slopes of lines AB and DE are complementary, the ratio MA/MD = ME/MB -/
theorem ratio_on_complementary_slopes (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
(e : ℝ) (h₂ : e = Real.sqrt 5) (d : ℝ) (h₃ : d = 2) 
(M : ℝ × ℝ) (hx : M.1 = 1 / 4) 
(MA MD ME MB : ℝ) 
(slopeAB slopeDE : ℝ) (h₄ : slopeAB + slopeDE = 0) : 
ratio_complementary_slopes a b h₀ h₁ e h₂ d h₃ M hx MA MD ME MB slopeAB slopeDE h₄ := 
sorry

end HyperbolaProof

end standard_equation_of_hyperbola_ratio_on_complementary_slopes_l70_70752


namespace MrsSheridanCurrentCats_l70_70070

theorem MrsSheridanCurrentCats :
  (needs_more : ℕ) (total : ℕ) (current : ℕ) 
  (h1 : needs_more = 32) (h2 : total = 43) 
  (h3 : current = total - needs_more) :
  current = 11 :=
by 
  rw [h1, h2, h3]
  have : 43 - 32 = 11 := by norm_num
  assumption

end MrsSheridanCurrentCats_l70_70070


namespace problem_part_a_problem_part_b_problem_part_e_problem_part_c_disproof_problem_part_d_disproof_l70_70558

-- Define Lean goals for the true statements
theorem problem_part_a (x : ℝ) (h : x < 0) : x^3 < x := sorry
theorem problem_part_b (x : ℝ) (h : x^3 > 0) : x > 0 := sorry
theorem problem_part_e (x : ℝ) (h : x > 1) : x^3 > x := sorry

-- Disprove the false statements by showing the negation
theorem problem_part_c_disproof (x : ℝ) (h : x^3 < x) : ¬ (|x| > 1) := sorry
theorem problem_part_d_disproof (x : ℝ) (h : x^3 > x) : ¬ (x > 1) := sorry

end problem_part_a_problem_part_b_problem_part_e_problem_part_c_disproof_problem_part_d_disproof_l70_70558


namespace sin_angle_PBO_l70_70438

open EuclideanGeometry

theorem sin_angle_PBO (A B C D O P : Point)
  (tetra : regular_tetrahedron A B C D)
  (centroid_O : centroid O B C D)
  (P_on_AO : on_line P (line_through A O))
  (P_minimizes : P_min AO P (line_through A O) (λ P, PA + 2 * (PB + PC + PD))) :
  sin_angle P B O = 1 / 6 :=
sorry

end sin_angle_PBO_l70_70438


namespace incorrect_statements_about_sets_l70_70557

/-
Define the statements about sets
-/
def statementA : Prop := ∃ a : Set ℤ, a = {x | x < 0}
def statementB : Prop := (∀ y, y = 2 * x^2 + 1) ↔ (∀ (x y), y = 2 * x^2 + 1)
def statementC : Prop := (Set.insert 1 (Set.insert 2 (Set.insert (abs (- (1/2))) (Set.insert 0.5 (Set.singleton (1/2))))) = {1, 2, 1/2, 0.5})
def statementD : Prop := ∀ s : Set ℕ, Set.empty ⊆ s

/-
Define a theorem to prove that statements A, B, and C are incorrect, and statement D is correct
-/
theorem incorrect_statements_about_sets : ¬statementA ∧ ¬statementB ∧ ¬statementC ∧ statementD :=
by
  sorry

end incorrect_statements_about_sets_l70_70557


namespace trapezium_longer_side_l70_70276

theorem trapezium_longer_side
  (a b h Area : ℝ)
  (h_a : a = 10)
  (h_h : h = 15)
  (h_Area : Area = 210) :
  b = 18 :=
by 
  have h_eq : Area = 1 / 2 * (a + b) * h := sorry,
  calc
    Area = 1 / 2 * (a + b) * h : h_eq
    ... = 210 : by rw [h_Area]
    ... = 7.5 * (10 + b) : sorry
    ... = 7.5 * (10 + 18) : by rw [h_a, h_h]
    ... = 18 : sorry

end trapezium_longer_side_l70_70276


namespace erika_rick_savings_l70_70680

def gift_cost : ℕ := 250
def erika_savings : ℕ := 155
def rick_savings : ℕ := gift_cost / 2
def cake_cost : ℕ := 25

theorem erika_rick_savings :
  let total_savings := erika_savings + rick_savings in
  let total_cost := gift_cost + cake_cost in
  total_savings - total_cost = 5 :=
by
  let total_savings := erika_savings + rick_savings
  let total_cost := gift_cost + cake_cost
  sorry

end erika_rick_savings_l70_70680


namespace max_T_value_l70_70443

def sequence_x (n : ℕ) : ℕ → ℝ := sorry

def sequence_y (n : ℕ) (x : ℕ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in finset.range k, x i / k

def abs_diff_sum (y : ℕ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in finset.range (k - 1), |y i - y (i + 1)|

theorem max_T_value (n : ℕ) (x : ℕ → ℝ)
  (h1 : n = 2008)
  (h2 : ∑ i in finset.range (n - 1), |x i - x (i + 1)| = 2008)
  (h3 : ∀ k, 1 ≤ k → k ≤ n → y k = sequence_y n x k) :
  abs_diff_sum y (n - 1) ≤ 2007 :=
sorry

end max_T_value_l70_70443


namespace smallest_integer_remainder_range_l70_70508

theorem smallest_integer_remainder_range (m : ℕ) (h1 : m > 1)
  (h2 : m % 5 = 1) (h3 : m % 7 = 1) (h4 : m % 8 = 1) :
  210 < m ∧ m < 299 :=
begin
  sorry
end

end smallest_integer_remainder_range_l70_70508


namespace markdown_final_price_l70_70606

theorem markdown_final_price (P : ℝ) : 
  let initial_discount := 0.70 * P,
      final_price := initial_discount - 0.10 * initial_discount in
  (final_price / P * 100) = 63 :=
by
  let initial_discount := 0.70 * P
  let final_price := initial_discount - 0.10 * initial_discount
  sorry

end markdown_final_price_l70_70606


namespace course_selection_l70_70992

theorem course_selection (num_courses num_students : ℕ) (choices : num_courses = 3) (students : num_students = 4) :
  (3 : ℕ) ^ 4 = 81 :=
by
  -- Given conditions
  have h1 : num_courses = 3 := choices,
  have h2 : num_students = 4 := students,
  -- The statement can be skipped with a sorry
  sorry

end course_selection_l70_70992


namespace simson_line_circumcircle_l70_70463

theorem simson_line_circumcircle {A B C P : Point} 
  (ABC_tri : triangle A B C) 
  (H_collinear : collinear [foot_perpendicular P A B, foot_perpendicular P B C, foot_perpendicular P C A]) : 
  lies_on_circumcircle P A B C :=
sorry

end simson_line_circumcircle_l70_70463


namespace second_train_speed_is_correct_l70_70194

noncomputable def speed_of_second_train (length_first : ℝ) (speed_first : ℝ) (time_cross : ℝ) (length_second : ℝ) : ℝ :=
let total_distance := length_first + length_second
let relative_speed := total_distance / time_cross
let relative_speed_kmph := relative_speed * 3.6
relative_speed_kmph - speed_first

theorem second_train_speed_is_correct :
  speed_of_second_train 270 120 9 230.04 = 80.016 :=
by
  sorry

end second_train_speed_is_correct_l70_70194


namespace inverse_isosceles_triangle_l70_70938

theorem inverse_isosceles_triangle :
  (∀ (T : Triangle), (∃ (a b : Angle), T.has_angle a ∧ T.has_angle b ∧ a = b) → T.is_isosceles) :=
sorry

end inverse_isosceles_triangle_l70_70938


namespace perp_line_intersection_l70_70760

variable {α β : Type*} [Plane α] [Plane β] [Line l] [Line a] [Line b]

axiom perp_planes : α ⟂ β
axiom intersection : α ∩ β = l
axiom parallel_line_plane : a ∥ α
axiom perp_line_plane : b ⟂ β

theorem perp_line_intersection : b ⟂ l := by
  sorry

end perp_line_intersection_l70_70760


namespace round_table_problem_l70_70627

-- Define the problem conditions in Lean 4
constant n : ℕ := 30
constant knights : Finset ℕ := {i : ℕ | i < 15}
constant liars : Finset ℕ := {i : ℕ | 15 ≤ i < 30}
constant is_knight : ℕ → Prop := λ i, i ∈ knights
constant is_liar : ℕ → Prop := λ i, i ∈ liars
constant friends : ℕ → ℕ := λ i, if i < 15 then i + 15 else i - 15
constant sitting_next_to : ℕ → ℕ → Prop := λ i j, (j = (i + 1) % n) ∨ (j = (i - 1 + n) % n) 

-- Given conditions
constant Q1 : ∀ i, friends (friends i) = i
constant Q2 : ∀ i, is_knight i ↔ is_liar (friends i)
constant Q3 : ∀ i, friends i ≠ i
constant Q4 : ∀ (i : ℕ), i % 2 = 0 → (sitting_next_to i (friends i) ↔ (is_knight i = false))

-- Prove that the number of remaining individuals who could have answered "Yes" is 0
theorem round_table_problem : 
  (∑ i in (Finset.range n).filter (λ i, ¬(i % 2 = 0)) count_yes_answer = 0 :=
by
  sorry

end round_table_problem_l70_70627


namespace least_product_of_distinct_primes_greater_than_50_l70_70150

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 :
  ∃ (p q : ℕ), distinct_primes_greater_than_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_50_l70_70150


namespace subtraction_multiplication_rounding_l70_70479

theorem subtraction_multiplication_rounding :
  let result := (555.55 - 222.22) * 1.5
  fin_to_dec_nearest_hundredth(result) = 500.00 :=
by
  sorry

end subtraction_multiplication_rounding_l70_70479


namespace sum_of_squares_second_15_eq_8195_l70_70186

theorem sum_of_squares_second_15_eq_8195
  (h : (∑ k in Finset.range 15, (k + 1)^2) = 1260) :
  (∑ k in Finset.range 15, (k + 16)^2) = 8195 :=
sorry

end sum_of_squares_second_15_eq_8195_l70_70186


namespace stones_required_to_pave_hall_l70_70595

noncomputable def hall_length_meters : ℝ := 36
noncomputable def hall_breadth_meters : ℝ := 15
noncomputable def stone_length_dms : ℝ := 4
noncomputable def stone_breadth_dms : ℝ := 5

theorem stones_required_to_pave_hall :
  let hall_length_dms := hall_length_meters * 10
  let hall_breadth_dms := hall_breadth_meters * 10
  let hall_area_dms_squared := hall_length_dms * hall_breadth_dms
  let stone_area_dms_squared := stone_length_dms * stone_breadth_dms
  let number_of_stones := hall_area_dms_squared / stone_area_dms_squared
  number_of_stones = 2700 :=
by
  sorry

end stones_required_to_pave_hall_l70_70595


namespace imaginary_part_of_z_l70_70106

theorem imaginary_part_of_z (a : ℝ) (h1 : a > 0) (h2 : complex.abs (1 - complex.I * a) = real.sqrt 5) :
  complex.im (1 + complex.I * a) = 2 :=
sorry

end imaginary_part_of_z_l70_70106


namespace coeff_x2_term_l70_70281

theorem coeff_x2_term (a b c d e f : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : d = 6) (h5 : e = 7) (h6 : f = 8) :
    (a * f + b * e * 1 + c * d) = 82 := 
by
    sorry

end coeff_x2_term_l70_70281


namespace least_possible_product_of_distinct_primes_gt_50_l70_70143

-- Define a predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the conditions: two distinct primes greater than 50
def distinct_primes_gt_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50 ∧ p ≠ q

-- The least possible product of two distinct primes each greater than 50
theorem least_possible_product_of_distinct_primes_gt_50 :
  ∃ p q : ℕ, distinct_primes_gt_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_possible_product_of_distinct_primes_gt_50_l70_70143


namespace number_of_subsets_of_A_is_4_l70_70343

-- Definitions given in the conditions.
def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 3}
def C_U_A : Set ℤ := {2}
def A : Set ℤ := U \ C_U_A

-- The final statement to be proved.
theorem number_of_subsets_of_A_is_4 : (A.powerset.card = 4) :=
sorry

end number_of_subsets_of_A_is_4_l70_70343


namespace test_easy_hard_l70_70249

-- Define students and problems as types
universe u
variables {Student : Type u} {Problem : Type u}

-- Define the condition (a) and (b)
def condition_a (solves : Student → Problem → Prop) : Prop :=
∀ p : Problem, ∃ s : Student, solves s p

def condition_b (solves : Student → Problem → Prop) : Prop :=
∀ v : Set Problem, v.nonempty → ∃ s : Student, ∀ p ∈ v, solves s p

-- The main theorem
theorem test_easy_hard (solves : Student → Problem → Prop) :
  (condition_a solves) → ¬ (condition_b solves) :=
by
  sorry

end test_easy_hard_l70_70249


namespace ellipse_problem_l70_70742

namespace EllipseProof

-- Definition for the first part of the proof
def ellipse_equation_correct (m : ℝ) : Prop :=
  (∃ a b : ℝ, a > b > 0 ∧ a^2 = m*(m+1) ∧ b^2 = m ∧ 
   (m = 1) ∧
   (a = sqrt 2) ∧ (b = 1) ∧
   (eccentricity = sqrt(2)/2) ∧
   (∃ x y : ℝ, (x^2 / 2) + y^2 = 1))

-- Definition for the second part of the proof
def eccentricity_range (m : ℝ) : Prop :=
  (m ∈ Ioo 0 1  →
  (∃ e : ℝ, e ∈ Ioo 0 (sqrt 2 / 2) ∧
   (2*m + 1 + (m + 1)^2 < 7)))

-- Main theorem combining both parts
theorem ellipse_problem (m : ℝ) : 
  ellipse_equation_correct m ∧ eccentricity_range m :=
by sorry

end EllipseProof

end ellipse_problem_l70_70742


namespace ratio_equivalence_l70_70583

theorem ratio_equivalence (x : ℕ) : 
  (10 * 60 = 600) →
  (15 : ℕ) / 5 = x / 600 →
  x = 1800 :=
by
  intros h1 h2
  sorry

end ratio_equivalence_l70_70583


namespace sum_of_divisors_excluding_n_l70_70923

theorem sum_of_divisors_excluding_n (p : ℕ) (h_prime : Nat.Prime (2^p - 1)) :
  let n := 2^(p-1) * (2^p - 1) in
  let sigma := ∑ k in Nat.divisors (2^(p-1) * (2^p - 1)), k in
  sigma - n = n :=
by
  sorry

end sum_of_divisors_excluding_n_l70_70923


namespace gain_percentage_correct_l70_70179

noncomputable def cloth_gain_percentage (x : ℝ) : ℝ := 
  let cp_1_meter := x
  let cp_15_meters := 15 * x
  let sp_15_meters := 25 * x
  let gain := sp_15_meters - cp_15_meters
  (gain / cp_15_meters) * 100

theorem gain_percentage_correct (x : ℝ) (h : x ≠ 0) :
  cloth_gain_percentage x = 66.67 := 
by
  let cp_1_meter := x
  let cp_15_meters := 15 * x
  let sp_15_meters := 25 * x
  let gain := sp_15_meters - cp_15_meters
  have h_gain : gain = 10 * x := by
    calc
      gain = sp_15_meters - cp_15_meters : rfl
      ...  = 25 * x - 15 * x : rfl
      ...  = 10 * x : by ring
  have h_gain_percentage : cloth_gain_percentage x = (10 * x / 15 * x) * 100 := by
    rw [cloth_gain_percentage, h_gain]
    rfl
  rw h_gain_percentage
  calc
    (10 * x / 15 * x) * 100 = (10 / 15) * 100 : by rw div_mul_cancel _ (ne_of_gt h)
    ...  = (2 / 3) * 100 : by norm_num
    ...  = 66.67 : by norm_num

end gain_percentage_correct_l70_70179


namespace largest_even_digit_multiple_of_5_less_than_10000_l70_70161

def is_even_digit (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ nat.digits 10 n → d % 2 = 0

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem largest_even_digit_multiple_of_5_less_than_10000 : 
  ∃ n : ℕ, is_even_digit n ∧ is_multiple_of_5 n ∧ n < 10000 ∧ ∀ m : ℕ, is_even_digit m ∧ is_multiple_of_5 m ∧ m < 10000 → m ≤ n :=
  -- The largest positive integer with only even digits, that is less than $10,000$, and is a multiple of $5$ is $8800$
  sorry

end largest_even_digit_multiple_of_5_less_than_10000_l70_70161


namespace sequence_properties_l70_70724

def a : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := a n + a (n+1)

theorem sequence_properties (x : ℕ) :
  (a 0 = x ∧ a 1 = x ∧ (∀ n, a (n + 2) = a n + a (n + 1))) →
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 143 * x) ∧
  ((∑ k in finset.range 2023, if a k % 2 = 1 then 1 else 0) = 1349) ∧
  (a 0 + a 1 + a 2 + ⋯ + a 2019 + a 2021 ≠ a 2022) :=
by
  sorry -- Proof not required.

end sequence_properties_l70_70724


namespace g_at_1_eq_binom_l70_70181

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def g (k l : ℕ) (x : ℤ) : ℤ :=
  ∑ i in Finset.range (k + 1), (binomial (k + l - i) (k - i) : ℤ) * x^i

theorem g_at_1_eq_binom (k l : ℕ) : g k l 1 = binomial (k + l) k := by
  sorry

end g_at_1_eq_binom_l70_70181


namespace range_of_a_l70_70316

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x0 : ℝ, x0^2 + 2 * x0 + a ≤ 0
def q (a : ℝ) : Prop := ∀ x > 0, x + 1/x > a

-- The theorem statement: if p is false and q is true, then 1 < a < 2
theorem range_of_a (a : ℝ) (h1 : ¬ p a) (h2 : q a) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l70_70316


namespace inequality_abc_l70_70903

theorem inequality_abc (a b c : ℝ) (h : a * b * c = 1) :
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end inequality_abc_l70_70903


namespace find_k_l70_70016

theorem find_k (k : ℝ) : 
  (∃ (x y : ℝ), 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0 ∧ x + k * y = 0) → k = -1 / 2 :=
by 
  sorry

end find_k_l70_70016


namespace distinct_mult_products_l70_70761

noncomputable def count_distinct_products (s : Finset ℕ) : ℕ :=
  (s.powerset.filter (λ t, 2 ≤ t.card)).image (λ t, t.prod id).card

theorem distinct_mult_products : count_distinct_products ({1, 2, 3, 7, 13, 17} : Finset ℕ) = 57 :=
by
  sorry

end distinct_mult_products_l70_70761


namespace lcm_of_ratio_and_hcf_l70_70573

-- Define the numbers with the given ratio and highest common factor
def number1 (x : ℕ) := 3 * x
def number2 (x : ℕ) := 4 * x
def hcf (a b : ℕ) := nat.gcd a b
def lcm (a b : ℕ) := nat.lcm a b

noncomputable def x := 4

-- The main proposition: the L.C.M of the two numbers is 48
theorem lcm_of_ratio_and_hcf : lcm (number1 x) (number2 x) = 48 := by
  -- Here we should have a formal proof, which we will skip for now
  sorry

end lcm_of_ratio_and_hcf_l70_70573


namespace convert_polar_to_rectangular_l70_70253

theorem convert_polar_to_rectangular :
  (∀ (r θ : ℝ), r = 4 ∧ θ = (Real.pi / 4) →
    (r * Real.cos θ, r * Real.sin θ) = (2 * Real.sqrt 2, 2 * Real.sqrt 2)) :=
by
  intros r θ h
  cases h with hr hθ
  rw [hr, hθ]
  simp
  sorry

end convert_polar_to_rectangular_l70_70253


namespace equal_area_opposite_pairs_l70_70400
-- Import the entire Mathlib library for necessary mathematical tools

-- Define the Equilateral Triangle and Centroid property
variables {A B C P : Type}
variables [isEquilateralTriangle : EquilateralTriangle A B C]
variables [isCentroid : Centroid P A B C]
variables [midpoint_AB : Midpoint P A B]
variables [midpoint_BC : Midpoint P B C]
variables [midpoint_CA : Midpoint P C A]

-- State the theorem to be proved
theorem equal_area_opposite_pairs :
  (∀ P, isCentroid P A B C →
    let triangles := divideIntoTriangles A B C P
    (∀ (Δ₁ Δ₂ : Triangle), Δ₁ ∈ triangles → Δ₂ ∈ triangles → area Δ₁ = area Δ₂)) :=
sorry

end equal_area_opposite_pairs_l70_70400


namespace total_legs_l70_70068

-- Definitions of the given conditions
def num_kangaroos : ℕ := 23
def legs_per_kangaroo : ℕ := 2
def num_goats : ℕ := 3 * num_kangaroos
def legs_per_goat : ℕ := 4
def num_spiders : ℕ := 2 * num_goats
def legs_per_spider : ℕ := 8
def num_birds : ℕ := num_spiders / 2
def legs_per_bird : ℕ := 2

-- Proof statement of the desired result 
theorem total_legs : 
  (num_kangaroos * legs_per_kangaroo) + 
  (num_goats * legs_per_goat) + 
  (num_spiders * legs_per_spider) + 
  (num_birds * legs_per_bird) = 1564 := 
by 
  calc 
    let total_kangaroo_legs := num_kangaroos * legs_per_kangaroo in
    let total_goat_legs := num_goats * legs_per_goat in
    let total_spider_legs := num_spiders * legs_per_spider in
    let total_bird_legs := num_birds * legs_per_bird in
    total_kangaroo_legs + total_goat_legs + total_spider_legs + total_bird_legs = 46 + 276 + 1104 + 138 := by sorry
    ... = 1564 := by sorry

end total_legs_l70_70068


namespace maximal_partition_sets_l70_70158

theorem maximal_partition_sets : 
  ∃(n : ℕ), (∀(a : ℕ), a * n = 16657706 → (a = 5771 ∧ n = 2886)) := 
by
  sorry

end maximal_partition_sets_l70_70158


namespace total_movie_cost_l70_70415

def previous_movie_length : Nat := 120
def new_movie_length : Nat := previous_movie_length + Nat.floor (0.60 * previous_movie_length)
def cost_per_minute_previous : Nat := 50
def cost_per_minute_new : Nat := 2 * cost_per_minute_previous
def total_cost : Nat := new_movie_length * cost_per_minute_new

theorem total_movie_cost :
  total_cost = 19200 := 
sorry

end total_movie_cost_l70_70415


namespace isosceles_triangle_perimeter_l70_70727

theorem isosceles_triangle_perimeter (a b : ℝ) (h_iso : (a = 4 ∨ a = 9) ∧ (b = 4 ∨ b = 9)) (h_triangle : is_isosceles_triangle (a, b, b)) : 
  (∃ p, p = 22) :=
by
  sorry

end isosceles_triangle_perimeter_l70_70727


namespace least_possible_product_of_primes_gt_50_l70_70155

open Nat

theorem least_possible_product_of_primes_gt_50 : 
  ∃ (p q : ℕ), prime p ∧ prime q ∧ p ≠ q ∧ p > 50 ∧ q > 50 ∧ (p * q = 3127) := 
  by
  exists 53
  exists 59
  repeat { sorry }

end least_possible_product_of_primes_gt_50_l70_70155


namespace complement_union_l70_70342

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {3, 4, 5}

theorem complement_union :
  ((U \ A) ∪ B) = {1, 3, 4, 5, 6} :=
by
  sorry

end complement_union_l70_70342


namespace percentage_of_muslim_boys_l70_70395

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℝ) (sikh_percentage : ℝ) (other_boys : ℕ) :
  total_boys = 650 →
  hindu_percentage = 0.28 →
  sikh_percentage = 0.10 →
  other_boys = 117 →
  (100 - ((hindu_percentage + sikh_percentage) * 100 + (other_boys.to_real / total_boys.to_real * 100))) = 44 :=
by
  intros h_total h_hindu h_sikh h_other
  sorry

end percentage_of_muslim_boys_l70_70395


namespace paths_from_A_to_B_l70_70762

theorem paths_from_A_to_B : 
  let total_steps := 9
  let up_steps := 4
  let right_steps := 5
  (total_steps = up_steps + right_steps) →
  (total_steps.choose up_steps) = 126 :=
by
  intros
  have h : total_steps = up_steps + right_steps := by assumption
  rw h
  rw Nat.choose_eq_factorial_div_factorial (Nat.succ_right_steps) (Nat.succ_up_steps)
  rw [Nat.factorial_succ, Nat.factorial_succ]
  sorry

end paths_from_A_to_B_l70_70762


namespace largest_k_for_family_of_three_element_subsets_l70_70435

-- Define the necessary parameters and predicates
variables {M : Type} (n : ℕ) (ψ : finset (finset M))

-- State the conditions
def is_family_of_three_element_subsets (ψ : finset (finset M)) : Prop :=
  ∀ s ∈ ψ, s.card = 3

def nonempty_intersection (ψ : finset (finset M)) : Prop :=
  ∀ s₁ s₂ ∈ ψ, (s₁ ≠ s₂) → (s₁ ∩ s₂).nonempty

-- State the main theorem
theorem largest_k_for_family_of_three_element_subsets 
  (hn : n ≥ 6) (M : finset M) (hM : M.card = n) 
  (ψ : finset (finset M)) (hψ : is_family_of_three_element_subsets ψ) 
  (h_inter : nonempty_intersection ψ) : 
  ψ.card ≤ (nat.choose (n-1) 2) := 
sorry

end largest_k_for_family_of_three_element_subsets_l70_70435


namespace line_circle_relationship_l70_70293

theorem line_circle_relationship (k : ℝ) :
  let line := (3*k+2)*x - k*y - 2 = 0
  let circle := x^2 + y^2 - 2*x - 2*y - 2 = 0
  ( ∃ x y, line = circle ∧ distance (1, 1) (line) = 2) ∨
  ( ∃ x y, line = circle ∧ distance (1, 1) (line) < 2) :=
sorry

end line_circle_relationship_l70_70293


namespace gcd_Sn_S3n_l70_70426

noncomputable def S (n : ℕ) : ℕ := (∑ k in Finset.range (n+1).filter (λ k, k > 0), k^5 + k^7)

theorem gcd_Sn_S3n (n : ℕ) : Nat.gcd (S n) (S (3 * n)) = 81 * n^4 :=
by
  sorry

end gcd_Sn_S3n_l70_70426


namespace find_m_plus_n_l70_70800

-- Definitions based on the given conditions
def points_symmetric_about_z_axis (A B : ℝ × ℝ × ℝ) : Prop := 
  A.1 = -B.1 ∧ A.2 = -B.2 ∧ A.3 = B.3

def A : ℝ × ℝ × ℝ := (m, n, 1)
def B : ℝ × ℝ × ℝ := (3, 2, 1)

-- The theorem to prove
theorem find_m_plus_n (m n : ℝ) (h : points_symmetric_about_z_axis A B) : m + n = -5 := 
by sorry

end find_m_plus_n_l70_70800


namespace distance_rowed_upstream_l70_70196

variables (V_b V_r D_d T_d T_u D_u : ℕ)
noncomputable def boat_speed := 14
noncomputable def distance_downstream := 200
noncomputable def time_downstream := 10
noncomputable def time_upstream := 12

theorem distance_rowed_upstream (x : ℕ) (h1 : V_b = boat_speed) 
                                (h2 : D_d = distance_downstream)
                                (h3 : T_d = time_downstream) 
                                (h4 : T_u = time_upstream)
                                (h5 : D_d = (V_b + V_r) * T_d)
                                (h6 : x = (distance_downstream - boat_speed * 10) / T_d) :
  D_u = (boat_speed - x) * T_u :=
by
  have Vr_eq : V_r = x, by sorry
  rw [h1, h2, h3, h4, h5, Vr_eq] at h6
  -- remaining steps are skipped
  have : D_u = (boat_speed - x) * Time_upstream, by sorry
  exact this

end distance_rowed_upstream_l70_70196


namespace factorize_problem1_factorize_problem2_l70_70688

-- Problem 1
theorem factorize_problem1 (a b : ℝ) : 
    -3 * a^2 + 6 * a * b - 3 * b^2 = -3 * (a - b)^2 := 
by sorry

-- Problem 2
theorem factorize_problem2 (a b x y : ℝ) : 
    9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) := 
by sorry

end factorize_problem1_factorize_problem2_l70_70688


namespace find_b_l70_70719

variable (b : ℝ)

-- Define the function
def f (x : ℝ) := 5 - b * x

-- Define the inverse function condition
axiom inverse_condition : ∃ g : ℝ → ℝ, ∀ y : ℝ, g (f y) = y ∧ (g (-3) = 3)

-- Define b such that the inverse condition holds
theorem find_b : b = 8 / 3 := by
  sorry

end find_b_l70_70719


namespace factor_polynomial_l70_70765

theorem factor_polynomial (x : ℝ) :
  (x^3 - 12 * x + 16) = (x + 4) * ((x - 2)^2) :=
by
  sorry

end factor_polynomial_l70_70765


namespace buddy_cards_on_thursday_is_32_l70_70881

def buddy_cards_on_monday := 30
def buddy_cards_on_tuesday := buddy_cards_on_monday / 2
def buddy_cards_on_wednesday := buddy_cards_on_tuesday + 12
def buddy_cards_bought_on_thursday := buddy_cards_on_tuesday / 3
def buddy_cards_on_thursday := buddy_cards_on_wednesday + buddy_cards_bought_on_thursday

theorem buddy_cards_on_thursday_is_32 : buddy_cards_on_thursday = 32 :=
by sorry

end buddy_cards_on_thursday_is_32_l70_70881


namespace problem1_partI_problem1_partII_l70_70333

-- Part (I) proof problem definition
theorem problem1_partI (a : ℝ) (h1 : ∀ x : ℝ, deriv (λ x, log x + a * x^2) x = (1 / x) + 2 * a * x)
  (h2: deriv (λ x, (1 / x) + 2 * a * x) 1 = 3) : 
  a = 2 := 
by sorry

-- Part (II) proof problem definition
theorem problem1_partII (a : ℝ) (h1 : ∀ x : ℝ, (λ x, log x + a * x) '' Ioi 0 ⊆ set.Ioi 0) : 
  a ≥ 0 := 
by sorry

end problem1_partI_problem1_partII_l70_70333


namespace problem_a_problem_b_l70_70710

-- Definitions based on conditions
def F (n : ℕ) : Set ℕ :=
  { m : ℕ | ∃ x : ℤ, x^2 + m * x + ↑n = 0 }

def S : Set ℕ :=
  { n : ℕ | 0 < n ∧ ∃ m : ℕ, m ∈ F n ∧ m + 1 ∈ F n }

-- Problem (a)
theorem problem_a : (S.countably_infinite ∧ (∑' n in S, 1 / n) ≤ 1) :=
  sorry

-- Problem (b)
theorem problem_b : ∃ f : ℕ → ℕ, (∀ z : ℕ, 0 < z → f z ∈ S) ∧ Function.Injective f :=
  sorry

end problem_a_problem_b_l70_70710


namespace quadratic_is_complete_the_square_l70_70966

theorem quadratic_is_complete_the_square :
  ∃ a b c : ℝ, 15 * (x : ℝ)^2 + 150 * x + 2250 = a * (x + b)^2 + c 
  ∧ a + b + c = 1895 :=
sorry

end quadratic_is_complete_the_square_l70_70966


namespace g_is_even_l70_70035

def g (x : ℝ) := 3 / (2 * x^8 - x^6 + 5)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end g_is_even_l70_70035


namespace primes_eq_condition_l70_70662

theorem primes_eq_condition (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) : 
  p + q^2 = r^4 → p = 7 ∧ q = 3 ∧ r = 2 := 
by
  sorry

end primes_eq_condition_l70_70662


namespace solve_system_of_equations_l70_70478

theorem solve_system_of_equations :
  ∃ x y : ℝ, (x^2 * y + x * y^2 - 2 * x - 2 * y + 10 = 0) ∧
             (x^3 * y - x * y^3 - 2 * x^2 + 2 * y^2 - 30 = 0) ∧
             (x = -4) ∧ (y = -1) :=
by
  use [-4, -1]
  constructor
  · sorry
  constructor
  · sorry
  constructor
  · rfl
  · rfl

end solve_system_of_equations_l70_70478


namespace equilateral_triangle_to_three_parts_l70_70619

theorem equilateral_triangle_to_three_parts :
  ∃ (A B C : Type) [equilateral_triangle A B C], 
    (cut_equilateral_triangle A B C 5).forms_equilateral_triangles (3) :=
by sorry

end equilateral_triangle_to_three_parts_l70_70619


namespace width_at_bottom_of_stream_l70_70572

theorem width_at_bottom_of_stream 
    (top_width : ℝ) (area : ℝ) (height : ℝ) (bottom_width : ℝ) :
    top_width = 10 → area = 640 → height = 80 → 
    2 * area = height * (top_width + bottom_width) → 
    bottom_width = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  -- Finding bottom width
  have h5 : 2 * 640 = 80 * (10 + bottom_width) := h4
  norm_num at h5
  linarith [h5]

#check width_at_bottom_of_stream

end width_at_bottom_of_stream_l70_70572


namespace shaded_area_is_correct_l70_70621

theorem shaded_area_is_correct : 
  ∀ (leg_length : ℕ) (total_partitions : ℕ) (shaded_partitions : ℕ) 
    (tri_area : ℕ) (small_tri_area : ℕ) (shaded_area : ℕ), 
  leg_length = 10 → 
  total_partitions = 25 →
  shaded_partitions = 15 →
  tri_area = (1 / 2 * leg_length * leg_length) → 
  small_tri_area = (tri_area / total_partitions) →
  shaded_area = (shaded_partitions * small_tri_area) →
  shaded_area = 30 :=
by
  intros leg_length total_partitions shaded_partitions tri_area small_tri_area shaded_area
  intros h_leg_length h_total_partitions h_shaded_partitions h_tri_area h_small_tri_area h_shaded_area
  sorry

end shaded_area_is_correct_l70_70621


namespace common_ratio_geometric_progression_l70_70790

theorem common_ratio_geometric_progression (r : ℝ) (a : ℝ) (h : a > 0) (h_r : r > 0) (h_eq : ∀ (n : ℕ), a * r^(n-1) = a * r^n + a * r^(n+1) + a * r^(n+2)) : r^3 + r^2 + r - 1 = 0 := 
by sorry

end common_ratio_geometric_progression_l70_70790


namespace maximize_area_of_triangle_MDE_l70_70411

theorem maximize_area_of_triangle_MDE (A B C M D E : Point)
  (h : ℝ) -- height of triangle ABC from vertex A to base BC
  (height_BC : Line) -- base BC
  (point_on_BC : M ∈ height_BC)
  (parallel_DE_BC : ∃ l, l = Line(DE) ∧ l ∥ height_BC)
  (height_split_eq : A.height = 2 * (A.height / 2)) -- DE divides height h into two equal parts
  : 
  ∃D E, area (Triangle M D E) is maximized ↔ DE divides height h equally
  := sorry

end maximize_area_of_triangle_MDE_l70_70411


namespace g_eq_g_inv_at_7_over_2_l70_70653

def g (x : ℝ) : ℝ := 3 * x - 7
def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_at_7_over_2 : g (7 / 2) = g_inv (7 / 2) := by
  sorry

end g_eq_g_inv_at_7_over_2_l70_70653


namespace range_of_a_l70_70383

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - a| < 4) → -1 < a ∧ a < 7 :=
  sorry

end range_of_a_l70_70383


namespace smallest_number_of_slices_l70_70455

def cheddar_slices : ℕ := 12
def swiss_slices : ℕ := 28
def gouda_slices : ℕ := 18

theorem smallest_number_of_slices : Nat.lcm (Nat.lcm cheddar_slices swiss_slices) gouda_slices = 252 :=
by 
  sorry

end smallest_number_of_slices_l70_70455


namespace exists_integer_sequence_satisfying_conditions_l70_70089

noncomputable def exists_integer_sequence (K : ℝ) : Prop :=
  ∃ (a : ℕ → ℕ), (∀ i j, i < j → a i < a j) ∧ 
                 (∀ n, a n < ⌊1.01 ^ n * K⌋) ∧ 
                 (∀ (s : Finset ℕ), ¬ is_square (s.sum a))

theorem exists_integer_sequence_satisfying_conditions : ∃ K > 0, exists_integer_sequence K :=
sorry

end exists_integer_sequence_satisfying_conditions_l70_70089


namespace num_valid_mappings_l70_70000

noncomputable def count_valid_mappings : ℕ :=
let M := { -1, 0, 1 }
let N := { -2, -1, 0, 1, 2 }
let f (x : ℤ) (n : ℤ → ℤ) := x + n x in
Finset.card { n : ℤ → ℤ |
  ∀ x ∈ M, x + n x ∈ N ∧ x + n x % 2 = 0 }

theorem num_valid_mappings : count_valid_mappings = 12 :=
sorry

end num_valid_mappings_l70_70000


namespace motion_of_Q_is_clockwise_with_2ω_l70_70768

variables {ω t : ℝ} {P Q : ℝ × ℝ}

def moving_counterclockwise (P : ℝ × ℝ) (ω t : ℝ) : Prop :=
  P = (Real.cos (ω * t), Real.sin (ω * t))

def motion_of_Q (x y : ℝ): ℝ × ℝ :=
  (-2 * x * y, y^2 - x^2)

def is_on_unit_circle (Q : ℝ × ℝ) : Prop :=
  Q.fst ^ 2 + Q.snd ^ 2 = 1

theorem motion_of_Q_is_clockwise_with_2ω 
  (P : ℝ × ℝ) (ω t : ℝ) (x y : ℝ) :
  moving_counterclockwise P ω t →
  P = (x, y) →
  is_on_unit_circle P →
  is_on_unit_circle (motion_of_Q x y) ∧
  Q = (x, y) →
  Q.fst = Real.cos (2 * ω * t + 3 * Real.pi / 2) ∧ 
  Q.snd = Real.sin (2 * ω * t + 3 * Real.pi / 2) :=
sorry

end motion_of_Q_is_clockwise_with_2ω_l70_70768


namespace complex_abs_value_z_l70_70840

-- Definition of complex numbers and absolute values in Lean
variable {z w : ℂ}

-- Hypotheses based on the given conditions
axiom h1 : |3 * z - 2 * w| = 30
axiom h2 : |z + 2 * w| = 5
axiom h3 : |z + w| = 2

-- Statement of the theorem to be proved
theorem complex_abs_value_z : |z| = Real.sqrt (19 / 8) :=
by
  sorry

end complex_abs_value_z_l70_70840


namespace find_k_l70_70031

noncomputable def isosceles_right_angled_triangle_area (a : ℝ) : ℝ :=
  (1 / 2) * a * a

theorem find_k
  (QPT_isosceles : ∠QPT = 90)
  (QTS_isosceles : ∠QTS = 90)
  (QSR_isosceles : ∠QSR = 90)
  (combined_area : isosceles_right_angled_triangle_area k + 
                   isosceles_right_angled_triangle_area (sqrt 2 * k) + 
                   isosceles_right_angled_triangle_area (2 * k) = 56) :
  k = 4 :=
  sorry

end find_k_l70_70031


namespace closest_number_to_fraction_l70_70267

theorem closest_number_to_fraction (x : ℝ) : 
  (abs (x - 2000) < abs (x - 1500)) ∧ 
  (abs (x - 2000) < abs (x - 2500)) ∧ 
  (abs (x - 2000) < abs (x - 3000)) ∧ 
  (abs (x - 2000) < abs (x - 3500)) :=
by
  let x := 504 / 0.252
  sorry

end closest_number_to_fraction_l70_70267


namespace evan_ivan_kara_total_weight_eq_432_l70_70686

variable (weight_evan : ℕ) (weight_ivan : ℕ) (weight_kara_cat : ℕ)

-- Conditions
def evans_dog_weight : Prop := weight_evan = 63
def ivans_dog_weight : Prop := weight_evan = 7 * weight_ivan
def karas_cat_weight : Prop := weight_kara_cat = 5 * (weight_evan + weight_ivan)

-- Mathematical equivalence
def total_weight : Prop := weight_evan + weight_ivan + weight_kara_cat = 432

theorem evan_ivan_kara_total_weight_eq_432 :
  evans_dog_weight weight_evan →
  ivans_dog_weight weight_evan weight_ivan →
  karas_cat_weight weight_evan weight_ivan weight_kara_cat →
  total_weight weight_evan weight_ivan weight_kara_cat :=
by
  intros h1 h2 h3
  sorry

end evan_ivan_kara_total_weight_eq_432_l70_70686


namespace total_pebbles_l70_70879

theorem total_pebbles (white_pebbles : ℕ) (red_pebbles : ℕ)
  (h1 : white_pebbles = 20)
  (h2 : red_pebbles = white_pebbles / 2) :
  white_pebbles + red_pebbles = 30 := by
  sorry

end total_pebbles_l70_70879


namespace least_product_of_distinct_primes_greater_than_50_l70_70149

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 :
  ∃ (p q : ℕ), distinct_primes_greater_than_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_50_l70_70149


namespace smallest_n_abs_diff_power_35_not_exists_l70_70211

noncomputable def is_abs_diff_power (n : ℕ) : Prop :=
  ∀ a b : ℕ, n ≠ |(2 : ℕ)^(a) - (3 : ℕ)^(b)|

theorem smallest_n_abs_diff_power_35_not_exists : is_abs_diff_power 35 := 
by
  sorry

end smallest_n_abs_diff_power_35_not_exists_l70_70211


namespace chopstick_consumption_1999_growth_rate_2001_desks_chairs_2001_l70_70997

noncomputable def chopstickSample : List ℝ := [0.6, 3.7, 2.2, 1.5, 2.8, 1.7, 1.2, 2.1, 3.2, 1.0]

def sampleMean (xs : List ℝ) : ℝ := (xs.sum / xs.length)

def totalConsumption (mean : ℝ) (restaurants : ℕ) (days : ℕ) : ℝ := mean * restaurants * days

def annualGrowthRate (initial : ℝ) (final : ℝ) (years : ℕ) : ℝ := (final / initial)^(1/years) - 1

def desksAndChairsSets (dailyUse : ℝ) (pairWeight : ℝ) (density : ℝ) (volumePerSet : ℝ) (boxesPerDay : ℝ) (restaurants : ℕ) (days : ℕ) : ℝ := 
  (boxesPerDay * restaurants * days * 100 * pairWeight / density) / volumePerSet

theorem chopstick_consumption_1999 :
  totalConsumption (sampleMean chopstickSample) 600 350 = 420000 := by
  sorry

theorem growth_rate_2001 :
  annualGrowthRate 2.0 2.42 2 = 0.1 := by
  sorry

theorem desks_chairs_2001 :
  desksAndChairsSets 2.42 0.005 (0.5 * 1000) 0.07 100 600 350 = 7260 := by
  sorry

end chopstick_consumption_1999_growth_rate_2001_desks_chairs_2001_l70_70997


namespace seats_with_middle_empty_l70_70402

-- Define the parameters
def chairs := 5
def people := 4
def middle_empty := 3

-- Define the function to calculate seating arrangements
def number_of_ways (people : ℕ) (chairs : ℕ) (middle_empty : ℕ) : ℕ := 
  if chairs < people + 1 then 0
  else (chairs - 1) * (chairs - 2) * (chairs - 3) * (chairs - 4)

-- The theorem to prove the number of ways given the conditions
theorem seats_with_middle_empty : number_of_ways 4 5 3 = 24 := by
  sorry

end seats_with_middle_empty_l70_70402


namespace solve_log_inequality_l70_70099

theorem solve_log_inequality (a : ℝ) (ha : 1 < a) :
  (∀ x : ℝ, (abs (Real.log x / Real.log a) + abs ((Real.log x / Real.log a) ^ 2 - 1) > a) ↔
    ((1 < a ∧ a < 5 / 4) → (x ∈ (set.Ioo 0 (a ^ (1 - real.sqrt (5 + 4 * a)) / 2)) ∪ 
      set.Ioo (a ^ (-1 - real.sqrt (5 - 4 * a)) / 2) (a ^ (-1 + real.sqrt (5 - 4 * a)) / 2)) ∪ 
      set.Ioo (a ^ (1 - real.sqrt (5 - 4 * a)) / 2) (a ^ (1 + real.sqrt (5 - 4 * a)) / 2) ∪ 
      set.Ioo (a ^ (1 + real.sqrt (5 + 4 * a)) / 2) ∞) ∨ 
    (a ≥ 5 / 4) → (x ∈ (set.Ioo 0 (a ^ (1 - real.sqrt (5 + 4 * a)) / 2)) ∪ 
      set.Ioo (a ^ (-1 + real.sqrt (5 + 4 * a)) / 2) ∞))) :=
sorry

end solve_log_inequality_l70_70099


namespace ratio_of_ages_in_six_years_l70_70876

-- Definitions based on conditions
def EllensCurrentAge : ℕ := 10
def MarthasCurrentAge : ℕ := 32

-- The main statement to prove
theorem ratio_of_ages_in_six_years : 
  let EllensAgeInSixYears := EllensCurrentAge + 6
  let MarthasAgeInSixYears := MarthasCurrentAge + 6
  (MarthasAgeInSixYears : ℚ) / (EllensAgeInSixYears : ℚ) = 19 / 8 := by
  sorry

end ratio_of_ages_in_six_years_l70_70876


namespace tan_ratio_in_triangle_l70_70811

theorem tan_ratio_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : a = c * sin A / sin C) 
  (h2 : b = c * sin B / sin C) 
  (h3 : a * cos B - b * cos A = c / 2) :
  (tan A / tan B) = 3 := 
  sorry

end tan_ratio_in_triangle_l70_70811


namespace acrobat_count_range_l70_70798

def animal_legs (elephants monkeys acrobats : ℕ) : ℕ :=
  4 * elephants + 2 * monkeys + 2 * acrobats

def animal_heads (elephants monkeys acrobats : ℕ) : ℕ :=
  elephants + monkeys + acrobats

theorem acrobat_count_range (e m a : ℕ) (h1 : animal_heads e m a = 18)
  (h2 : animal_legs e m a = 50) : 0 ≤ a ∧ a ≤ 11 :=
by {
  sorry
}

end acrobat_count_range_l70_70798


namespace extremum_and_zeros_range_of_a_l70_70854

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + b * x - a * Real.log x

theorem extremum_and_zeros (a b : ℝ) (x0 : ℝ) (n : ℕ) (h : x0 ∈ set.Ioo (n : ℝ) (n+1)) :
  (∀ x, deriv (λ x, f x a b) x = 0 → x = 2) →
  f 1 a b = 0 →
  f x0 a b = 0 →
  n = 3 :=
sorry

theorem range_of_a (a : ℝ) (h : ∀ b ∈ Icc (-2 : ℝ) (-1), ∃ x ∈ Ioo (1 : ℝ) Real.exp, f x a b < 0) :
  a > 1 :=
sorry

end extremum_and_zeros_range_of_a_l70_70854


namespace ratio_area_circle_to_triangle_l70_70490

theorem ratio_area_circle_to_triangle (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
    (π * r) / (h + r) = (π * r ^ 2) / (r * (h + r)) := sorry

end ratio_area_circle_to_triangle_l70_70490


namespace probability_of_pink_l70_70036

theorem probability_of_pink (B P : ℕ) (h1 : (B : ℚ) / (B + P) = 6 / 7) (h2 : (B^2 : ℚ) / (B + P)^2 = 36 / 49) : 
  (P : ℚ) / (B + P) = 1 / 7 :=
by
  sorry

end probability_of_pink_l70_70036


namespace inequality_satisfied_l70_70850

open Real

theorem inequality_satisfied (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  a * sqrt b + b * sqrt c + c * sqrt a ≤ 1 / sqrt 3 :=
sorry

end inequality_satisfied_l70_70850


namespace integral_sin_squared_l70_70272

theorem integral_sin_squared : ∫ (x : ℝ) in 0..π/2, (sin (x / 2)) ^ 2 = π / 4 - 1 / 2 :=
by
  sorry

end integral_sin_squared_l70_70272


namespace quadratic_inequality_solution_l70_70370

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
by
  sorry

end quadratic_inequality_solution_l70_70370


namespace feathers_before_crossing_road_l70_70485

theorem feathers_before_crossing_road : 
  ∀ (F : ℕ), 
  (F - (2 * 23) = 5217) → 
  F = 5263 :=
by
  intros F h
  sorry

end feathers_before_crossing_road_l70_70485


namespace range_of_dot_product_l70_70732

noncomputable def ellipse : set (ℝ × ℝ) := {p | p.1^2 / 9 + p.2^2 / 8 = 1}
noncomputable def circle : set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 1}

theorem range_of_dot_product (P : ℝ × ℝ) (A B : ℝ × ℝ)
  (hP : P ∈ ellipse)
  (hAB : ∀ t : ℝ, A = (1 + cos t, sin t) ∧ B = (1 - cos t, -sin t)) :
    3 ≤ (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) ∧
    (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) ≤ 15 :=
sorry

end range_of_dot_product_l70_70732


namespace least_N_prime_condition_l70_70288

theorem least_N_prime_condition :
  ∃ N : ℕ, N > 0 ∧ (∀ n : ℕ, prime (1 + N * 2^n) ↔ n % 12 = 0) ∧ N = 556 :=
sorry

end least_N_prime_condition_l70_70288


namespace find_f_6_l70_70063

def f : ℕ → ℤ 
| 0       := 0  -- Placeholder, we don't need f(0)
| (n + 1) := if (1 ≤ n + 1 ∧ n + 1 ≤ 4) then f(n) - (n + 1)
             else if (5 ≤ n + 1 ∧ n + 1 ≤ 8) then f(n) + 2 * (n + 1)
             else if (n + 1 ≥ 9) then f(n) * (n + 1)
             else 0  -- Placeholder for other cases

theorem find_f_6 : f 4 = 15 → f 6 = 37 := 
by 
  sorry

end find_f_6_l70_70063


namespace find_g_of_three_over_five_l70_70112

variable (g : ℝ → ℝ)
variable (h0 : g 0 = 0)
variable (h1 : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g(x) ≤ g(y))
variable (h2 : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g(1 - x) = 1 - g(x))
variable (h3 : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g(2 * x / 5) = g(x) / 3)

theorem find_g_of_three_over_five : g (3/5) = 2/3 := by
  sorry

end find_g_of_three_over_five_l70_70112


namespace cost_two_enchiladas_two_tacos_three_burritos_l70_70466

variables (e t b : ℝ)

theorem cost_two_enchiladas_two_tacos_three_burritos 
  (h1 : 2 * e + 3 * t + b = 5.00)
  (h2 : 3 * e + 2 * t + 2 * b = 7.50) : 
  2 * e + 2 * t + 3 * b = 10.625 :=
sorry

end cost_two_enchiladas_two_tacos_three_burritos_l70_70466


namespace domain_of_f_l70_70285

noncomputable def f (x : ℝ) : ℝ := (4 * x + 2) / Real.sqrt (x - 7)

theorem domain_of_f : {x | ∃ y, f y = x} = {x : ℝ | x > 7} :=
by
  -- Skipping the proof
  sorry

end domain_of_f_l70_70285


namespace digits_difference_l70_70550

-- Defining base-10 integers 300 and 1500
def n1 := 300
def n2 := 1500

-- Defining a function to calculate the number of digits in binary representation
def binary_digits (n : ℕ) : ℕ := (nat.log2 n) + 1

-- Statement to be proven
theorem digits_difference : binary_digits n2 = binary_digits n1 + 2 := sorry

end digits_difference_l70_70550


namespace exists_lcm_lt_l70_70347

theorem exists_lcm_lt (p q : ℕ) (hpq_coprime : Nat.gcd p q = 1) (hp_gt_one : p > 1) (hq_gt_one : q > 1) (hpq_diff_gt_one : (p < q ∧ q - p > 1) ∨ (p > q ∧ p - q > 1)) :
  ∃ n : ℕ, Nat.lcm (p + n) (q + n) < Nat.lcm p q := by
  sorry

end exists_lcm_lt_l70_70347


namespace geometric_series_sum_l70_70239

noncomputable def geometric_sum (a r : ℚ) (h : |r| < 1) : ℚ :=
a / (1 - r)

theorem geometric_series_sum :
  geometric_sum 1 (1/3) (by norm_num) = 3/2 :=
by
  sorry

end geometric_series_sum_l70_70239


namespace lisa_needs_additional_marbles_l70_70863

theorem lisa_needs_additional_marbles
  (friends : ℕ) (initial_marbles : ℕ) (total_required_marbles : ℕ) :
  friends = 12 ∧ initial_marbles = 40 ∧ total_required_marbles = (friends * (friends + 1)) / 2 →
  total_required_marbles - initial_marbles = 38 :=
by
  sorry

end lisa_needs_additional_marbles_l70_70863


namespace part1_part2_l70_70855

-- Given conditions for part (Ⅰ)
variables {a_n : ℕ → ℝ} {S_n : ℕ → ℝ}

-- The general formula for the sequence {a_n}
theorem part1 (a3_eq : a_n 3 = 1 / 8)
  (arith_seq : S_n 2 + 1 / 16 = 2 * S_n 3 - S_n 4) :
  ∀ n, a_n n = (1 / 2)^n := sorry

-- Given conditions for part (Ⅱ)
variables {b_n : ℕ → ℝ} {T_n : ℕ → ℝ}

-- The sum of the first n terms of the sequence {b_n}
theorem part2 (h_general : ∀ n, a_n n = (1 / 2)^n)
  (b_formula : ∀ n, b_n n = a_n n * (Real.log (a_n n) / Real.log (1 / 2))) :
  ∀ n, T_n n = 2 - (n + 2) / 2^n := sorry

end part1_part2_l70_70855


namespace perfect_square_expression_l70_70525

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 :=
by
  sorry

end perfect_square_expression_l70_70525


namespace cat_finishes_food_on_wednesday_l70_70896

theorem cat_finishes_food_on_wednesday :
  (∀ n : ℕ, n ≥ 1 → cat_food_consumed( (3 / 5 : ℝ), n ) = (3 / 5) * n) →
  (cat_food_total 10) →
  (start_day monday) →
  (additional_days_needed 7) →
  (cat_finishes_on wednesday) :=
begin
  sorry
end

end cat_finishes_food_on_wednesday_l70_70896


namespace find_original_price_of_petrol_l70_70215

open Real

noncomputable def original_price_of_petrol (P : ℝ) : Prop :=
  ∀ G : ℝ, 
  (G * P = 300) ∧ 
  ((G + 7) * 0.85 * P = 300) → 
  P = 7.56

-- Theorems should ideally be defined within certain scopes or namespaces
theorem find_original_price_of_petrol (P : ℝ) : original_price_of_petrol P :=
  sorry

end find_original_price_of_petrol_l70_70215


namespace problem_statement_l70_70314

-- Given conditions
def f : ℝ → ℝ := sorry
axiom dec_f (x1 x2 : ℝ) (h: x1 ≠ x2) : (f x1 - f x2) / (x1 - x2) < 0

-- Derived values to be used in proof
def sqrt_6 : ℝ := real.sqrt 6
def pow_0_7_6 : ℝ := 0.7^6
def log_0_7_6 : ℝ := (real.log 6) / (real.log 0.7)

-- Statement to prove
theorem problem_statement : f sqrt_6 < f pow_0_7_6 ∧ f pow_0_7_6 < f log_0_7_6 := by
  sorry

end problem_statement_l70_70314


namespace quadratic_inequality_solution_l70_70371

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
by
  sorry

end quadratic_inequality_solution_l70_70371


namespace place_tokens_l70_70080

theorem place_tokens (initial_tokens : Fin 50 → Fin 50 → Bool) :
  ∃ (new_tokens : Fin 50 → Fin 50 → Bool), 
    (∑ i j, if new_tokens i j then 1 else 0) ≤ 99 ∧
    ∀ i, even (∑ j, if initial_tokens i j ∨ new_tokens i j then 1 else 0) ∧
    ∀ j, even (∑ i, if initial_tokens i j ∨ new_tokens i j then 1 else 0) :=
by sorry

end place_tokens_l70_70080


namespace coprime_lcm_inequality_l70_70345

theorem coprime_lcm_inequality
  (p q : ℕ)
  (hpq_coprime : Nat.gcd p q = 1)
  (hp_gt_1 : p > 1)
  (hq_gt_1 : q > 1)
  (hpq_diff_gt_1 : abs (p - q) > 1) :
  ∃ n : ℕ, Nat.lcm (p + n) (q + n) < Nat.lcm p q :=
by
  sorry

end coprime_lcm_inequality_l70_70345


namespace final_price_after_changes_l70_70505

-- Define the initial conditions
def original_price : ℝ := 400
def decrease_percentage : ℝ := 15 / 100
def increase_percentage : ℝ := 40 / 100

-- Lean 4 statement of the proof problem
theorem final_price_after_changes :
  (original_price * (1 - decrease_percentage)) * (1 + increase_percentage) = 476 :=
by
  -- Proof goes here
  sorry

end final_price_after_changes_l70_70505


namespace solve_equation_l70_70476

theorem solve_equation (x : ℝ) : (x + 1) * (x - 3) = 5 ↔ (x = 4 ∨ x = -2) :=
by
  sorry

end solve_equation_l70_70476


namespace fraction_value_l70_70481

variable {x y : ℝ}

theorem fraction_value (hx : x ≠ 0) (hy : y ≠ 0) (h : (2 * x - 3 * y) / (x + 2 * y) = 3) :
  (x - 2 * y) / (2 * x + 3 * y) = 11 / 15 :=
  sorry

end fraction_value_l70_70481


namespace greatest_possible_gcd_of_ten_numbers_l70_70129

theorem greatest_possible_gcd_of_ten_numbers (a : Fin 10 → ℕ) (h_sum : (∑ i, a i) = 1001) : 
  ∃ d, d = 91 ∧ ∀ i, d ∣ a i :=
by
  sorry

end greatest_possible_gcd_of_ten_numbers_l70_70129


namespace container_capacity_l70_70772

/-- Given a container where 8 liters is 20% of its capacity, calculate the total capacity of 
    40 such containers filled with water. -/
theorem container_capacity (c : ℝ) (h : 8 = 0.20 * c) : 
    40 * c * 40 = 1600 := 
by
  sorry

end container_capacity_l70_70772


namespace exists_quad_root_l70_70960

theorem exists_quad_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (∃ x, x^2 + a * x + b = 0) ∨ (∃ x, x^2 + c * x + d = 0) :=
sorry

end exists_quad_root_l70_70960


namespace total_capacity_is_1600_l70_70774

/-- Eight liters is 20% of the capacity of one container. -/
def capacity_of_one_container := 8 / 0.20

/-- Calculate the total capacity of 40 such containers filled with water. -/
def total_capacity_of_40_containers := 40 * capacity_of_one_container

theorem total_capacity_is_1600 :
  total_capacity_of_40_containers = 1600 := by
    -- Proof is skipped using sorry.
    sorry

end total_capacity_is_1600_l70_70774


namespace functional_relationship_maximum_profit_range_for_a_l70_70174

-- Define the conditions
def cost_price : ℝ := 8
def initial_selling_price : ℝ := 10
def initial_sales : ℝ := 300
def price_increase_effect : ℝ := 50

-- Part 1: Prove that y = -50x + 800
theorem functional_relationship (x : ℝ) : 
  (λ y, y = -price_increase_effect * x + (initial_sales + price_increase_effect * initial_selling_price)) x = -50 * x + 800 :=
by 
  rw [price_increase_effect, initial_sales, initial_selling_price]
  sorry

-- Part 2: Maximize the profit given y >= 250 kg
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (-price_increase_effect * x + (initial_sales + price_increase_effect * initial_selling_price))

theorem maximum_profit (x : ℝ) (hx : -price_increase_effect * x + (initial_sales + price_increase_effect * initial_selling_price) ≥ 250) : 
  ∃ w, max (λ x, (x - cost_price) * (-price_increase_effect * x + (initial_sales + price_increase_effect * initial_selling_price))) = w ∧ w = 750 :=
by 
  rw [price_increase_effect, initial_sales, initial_selling_price]
  sorry

-- Part 3: Find the range for a such that net profit after donation increases for x <= 13
def net_profit_function (x : ℝ) (a : ℝ) : ℝ := (x - cost_price - a) * (-price_increase_effect * x + (initial_sales + price_increase_effect * initial_selling_price))

theorem range_for_a (a : ℝ) (ha : 0 ≤ a ∧ a ≤ 2.5) :
  ∀ x < 13, by (differentiable_on ℝ (λ x, (x - cost_price - a) * (-price_increase_effect * x + (initial_sales + price_increase_effect * initial_selling_price))))
    (Icc 0 13) → ∀ y, y > x → net_profit_function y a > net_profit_function x a → a ∈ Icc 2 2.5 :=
by
  rw [price_increase_effect, initial_sales, initial_selling_price]
  sorry

end functional_relationship_maximum_profit_range_for_a_l70_70174


namespace volume_le_one_eighth_l70_70073

-- Define points in 3D space
structure Point (α : Type) := (x : α) (y : α) (z : α)

-- Define a tetrahedron using four points
structure Tetrahedron (α : Type) := (A B C D : Point α)

-- Define a function to calculate edge length
def edge_length {α : Type} [LinearOrderedField α] (p1 p2 : Point α) : α :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Define a function to calculate the volume of a tetrahedron
noncomputable def volume {α : Type} [LinearOrderedField α] (tet : Tetrahedron α) : α :=
  let det := det (Matrix.of ![
    ![tet.B.x - tet.A.x, tet.B.y - tet.A.y, tet.B.z - tet.A.z],
    ![tet.C.x - tet.A.x, tet.C.y - tet.A.y, tet.C.z - tet.A.z], 
    ![tet.D.x - tet.A.x, tet.D.y - tet.A.y, tet.D.z - tet.A.z]]) in
  (abs det) / 6

-- Postulate the given conditions
axiom tet_exists (α : Type) [LinearOrderedField α] : Tetrahedron α
axiom CD_gt_1 {α : Type} [LinearOrderedField α] (tet : Tetrahedron α) : (edge_length tet.C tet.D) > 1
axiom other_edges_le_1 {α : Type} [LinearOrderedField α] (tet : Tetrahedron α) :
  (edge_length tet.A tet.B) ≤ 1 ∧ (edge_length tet.A tet.C) ≤ 1 ∧ (edge_length tet.A tet.D) ≤ 1 ∧
  (edge_length tet.B tet.C) ≤ 1 ∧ (edge_length tet.B tet.D) ≤ 1

-- Theorem stating the volume of the tetrahedron is at most 1/8
theorem volume_le_one_eighth {α : Type} [LinearOrderedField α] (tet : Tetrahedron α) :
  volume tet ≤ 1 / 8 :=
  sorry

end volume_le_one_eighth_l70_70073


namespace max_sum_of_products_l70_70988

theorem max_sum_of_products :
  ∀ f g h j : ℕ, 
  (∀ n ∈ {f, g, h, j}, n ∈ {6, 7, 8, 9}) →
  (∀ x y : ℕ, x ≠ y → fintype.card {z // z ∈ {f, g, h, j} ∧ z = x} = 1 → 
                        fintype.card {z // z ∈ {f, g, h, j} ∧ z = y} = 1) →
  fg + gh + hj + fj ≤ 221 :=
by sorry

end max_sum_of_products_l70_70988


namespace parabola_and_line_l70_70305

-- Definitions based on conditions in the problem
def parabola_eq (p : ℝ) (h : p > 0) : Prop :=
  ∀ x y : ℝ, y = x^2 / (2 * p)

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p / 2)

def point_P : ℝ × ℝ := (4, 0)

def point_Q (p y₀ : ℝ) : ℝ × ℝ := (4, y₀)

def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

def condition_dist_QF_PQ (p y₀ : ℝ) : Prop :=
  let F := focus p in
  let Q := point_Q p y₀ in
  distance Q F = (5 / 4) * distance point_P Q

-- Main theorem statement
theorem parabola_and_line
  (p : ℝ) (h : p > 0) (y₀ : ℝ)
  (cond1 : parabola_eq p h)
  (cond2 : condition_dist_QF_PQ p y₀)
  (A : ℝ × ℝ := (-4, 4)) :
  (p = 2 ∧ (∃ k : ℝ, parabola_eq 4 sorry ∧ k = 1 ∧ ∀ M N : ℝ × ℝ, M.1^2 = 4 * M.2 ∧ N.1^2 = 4 * N.2 ∧ M.2 = k * M.1 + 4 ∧ N.2 = k * N.1 + 4 →
       (M ≠ N ∧ A ≠ M ∧ A ≠ N ∧ distance M A = distance N A))) :=
sorry

end parabola_and_line_l70_70305


namespace arrange_weights_with_at_most_nine_comparisons_l70_70534

-- Define the set of weights and the comparison function
def weights := fin 5

-- Define the mass function
def mass (w : weights) : ℝ := sorry -- Assume an arbitrary mass function

-- Define the comparison property
def compare (a b c : weights) : Prop := mass a < mass b ∧ mass b < mass c

-- Main statement
theorem arrange_weights_with_at_most_nine_comparisons : 
  ∃ (comparisons : list (weights × weights × weights)),
    comparisons.length ≤ 9 ∧
    ∀ (a b c : weights), compare a b c ↔ a ≠ b ∧ b ≠ c ∧ c ≠ a :=
sorry

end arrange_weights_with_at_most_nine_comparisons_l70_70534


namespace simplify_expression_l70_70429

noncomputable def p (x a b c : ℝ) :=
  (x + 2 * a)^2 / ((a - b) * (a - c)) +
  (x + 2 * b)^2 / ((b - a) * (b - c)) +
  (x + 2 * c)^2 / ((c - a) * (c - b))

theorem simplify_expression (a b c x : ℝ) (h : a ≠ b ∧ a ≠ c ∧ b ≠ c) :
  p x a b c = 4 :=
by
  sorry

end simplify_expression_l70_70429


namespace problem_1_problem_2_l70_70848

variable (n : ℕ)
variable (S : Finset ℕ := Finset.range (n + 1) ∪ {0})
variable (good_subsets : Finset ℕ → Prop := λ X, (X.filter (λ x, x % 2 = 1)).card > (X.filter (λ x, x % 2 = 0)).card)

noncomputable def F (m : ℕ) : ℕ :=
Finset.card (S.powerset.filter good_subsets)

theorem problem_1 (n : ℕ) : F (2 * n + 1) = 2^(2 * n) := sorry

theorem problem_2 (n : ℕ) : F (2 * n) = (2^(2 * n) - Nat.choose (2 * n) n) / 2 := sorry

end problem_1_problem_2_l70_70848


namespace bet_final_result_l70_70920

theorem bet_final_result :
  let M₀ := 64
  let final_money := (3 / 2) ^ 3 * (1 / 2) ^ 3 * M₀
  final_money = 27 ∧ M₀ - final_money = 37 :=
by
  sorry

end bet_final_result_l70_70920


namespace combined_weight_of_candles_l70_70271

theorem combined_weight_of_candles 
  (beeswax_weight_per_candle : ℕ)
  (coconut_oil_weight_per_candle : ℕ)
  (total_candles : ℕ)
  (candles_made : ℕ) 
  (total_weight: ℕ) 
  : 
  beeswax_weight_per_candle = 8 → 
  coconut_oil_weight_per_candle = 1 → 
  total_candles = 10 → 
  candles_made = total_candles - 3 →
  total_weight = candles_made * (beeswax_weight_per_candle + coconut_oil_weight_per_candle) →
  total_weight = 63 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end combined_weight_of_candles_l70_70271


namespace complex_solutions_count_l70_70697

noncomputable def find_complex_numbers : Prop :=
  ∃ (z : ℂ), |z| = 1 ∧ (| (z^2 / (conj z)^2) + ((conj z)^2 / z^2) | = 2)

theorem complex_solutions_count : ∃ (s : Finset ℂ), s.card = 8 ∧ ∀ z ∈ s, find_complex_numbers :=
sorry

end complex_solutions_count_l70_70697


namespace partial_fraction_decomposition_exists_l70_70275

theorem partial_fraction_decomposition_exists :
  ∃ (A B C D : ℝ), 
    (A = 3) ∧ (B = 4) ∧ (C = 1) ∧ (D = -5) ∧ 
    (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1, 
      (-x^3 + 4*x^2 - 5*x + 3) / (x^4 + x^2) = 
      A / x^2 + (B*x + C) / (x^2 + 1) + D / x) :=
sorry

end partial_fraction_decomposition_exists_l70_70275


namespace polygon_sides_l70_70130

theorem polygon_sides (x : ℕ) 
  (h1 : 180 * (x - 2) = 3 * 360) 
  : x = 8 := 
by
  sorry

end polygon_sides_l70_70130


namespace tangent_x_axis_l70_70784

noncomputable def curve (k : ℝ) : ℝ → ℝ := λ x => Real.log x - k * x + 3

theorem tangent_x_axis (k : ℝ) : 
  ∃ t : ℝ, curve k t = 0 ∧ deriv (curve k) t = 0 → k = Real.exp 2 :=
by
  sorry

end tangent_x_axis_l70_70784


namespace smallest_number_among_set_l70_70618

theorem smallest_number_among_set : ∀ (x : Int), (x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1) → (x = -2) ↔ ∀ y, (y = -2 ∨ y = -1 ∨ y = 0 ∨ y = 1) → y ≥ -2 := 
by
  intro x Hx
  split
  . intro H
    intro y Hy
    cases Hy
    . rw [Hy]
      simp
    . cases Hy
      . simp [Hy]
      . cases Hy
        . simp [Hy]
        . simp [Hy]
  . intro H
    exact H _ Hx

-- Proof will be provided

end smallest_number_among_set_l70_70618


namespace problem_a_l70_70191

theorem problem_a (x a : ℝ) (h : (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) = 3 * a^4) :
  x = (-5 * a + a * Real.sqrt 37) / 2 ∨ x = (-5 * a - a * Real.sqrt 37) / 2 :=
by
  sorry

end problem_a_l70_70191


namespace complex_number_norm_l70_70013

theorem complex_number_norm (z : ℂ) (h : (1 - z) / (1 + z) = complex.i) :
  complex.abs (z + 1) = real.sqrt 2 :=
sorry

end complex_number_norm_l70_70013


namespace erin_soup_serving_time_l70_70266

/-- 
Erin works in the school cafeteria serving soup. She has to serve three different soups, each soup in a separate pot. 
The first pot has 8 gallons of soup, the second pot has 5.5 gallons of soup, and the third pot has 3.25 gallons of soup. 
Each bowl of soup has 10 ounces, and Erin can serve 5 bowls per minute from a single pot (she can only serve from one pot at a time). 
If there are 128 ounces in a gallon, how long, in minutes, will it take Erin to serve all the soups, rounded to the nearest minute?
-/
theorem erin_soup_serving_time :
  let gallons_to_ounces (g : ℕ) := g * 128 in
  let ounces_per_bowl := 10 in
  let bowls_per_minute := 5 in
  let total_ounces := gallons_to_ounces 8 + gallons_to_ounces 5.5 + gallons_to_ounces 3.25 in
  let total_bowls := total_ounces / ounces_per_bowl in
  let total_minutes := total_bowls / bowls_per_minute in
  (round total_minutes).toNat = 43 :=
by {
  -- Definitions of gallons_to_ounces and conditions are inline with the proof goal.
  sorry
}

end erin_soup_serving_time_l70_70266


namespace hash_triple_l70_70641

def hash (N : ℝ) : ℝ := 0.5 * (N^2) + 1

theorem hash_triple  : hash (hash (hash 4)) = 862.125 :=
by {
  sorry
}

end hash_triple_l70_70641


namespace max_distance_between_points_on_circles_l70_70085

-- Define the circles and their properties
def C1_center : ℝ × ℝ := (0, -3)
def C1_radius : ℝ := 1

def C2_center : ℝ × ℝ := (4, 0)
def C2_radius : ℝ := 2

-- Calculate the distance between centers
def distance_between_centers : ℝ :=
  let (x1, y1) := C1_center
  let (x2, y2) := C2_center
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

-- Define the problem statement
theorem max_distance_between_points_on_circles :
  distance_between_centers + C1_radius + C2_radius = 8 :=
by
  sorry

end max_distance_between_points_on_circles_l70_70085


namespace range_of_b_not_monotonic_range_of_a_ln_inequality_exist_points_POQ_right_triangle_l70_70336

-- Problem 1: Prove the range of b.
theorem range_of_b_not_monotonic (b: ℝ):
  (∃ x: ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_deriv x = 16 + b) ∧ 
  (∃ x: ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_deriv x = 5 + b) → 
  (-16 < b ∧ b < -5):=
sorry

-- Problem 2: Prove the range of a.
theorem range_of_a_ln_inequality (a: ℝ):
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ exp 1 → ln x ≥ -x^2 + (a + 2) * x) →
  (a ≤ -1) :=
sorry

-- Problem 3: Prove the existence of P and Q
theorem exist_points_POQ_right_triangle (a: ℝ): 
  a > 0 →  
  (∃ P Q : ℝ × ℝ, P != Q ∧ (P.1 = -Q.1 ∧ P.2 = Q.2) ∧ 
  (P.1 ≠ 0 ∧ Q.1 ≠ 0) ∧ 
  (∃ t : ℝ, t > 0 ∧ t ≠ 1 ∧ 
   (if t < 1 then -t^2 + (-t^3 + t^2)*(t^3 + t^2) = 0 
   else -t^2 + a*ln t*(t^3 + t^2) = 0)) :=
sorry

end range_of_b_not_monotonic_range_of_a_ln_inequality_exist_points_POQ_right_triangle_l70_70336


namespace cos_AEC_eq_one_l70_70403

variables {A B C D E : Type} [inner_product_space ℝ (A B C D E)]

-- Define angles and their relationships in the tetrahedron
variables (angle_ADB angle_ADC angle_BDC : ℝ)
variables (sin_CAD sin_CBD : ℝ)

-- Given conditions as assumptions
axiom (h_angle_ADB : angle_ADB = π / 2)
axiom (h_angle_ADC : angle_ADC = π / 2)
axiom (h_angle_BDC : angle_BDC = π / 2)
axiom (h_sin_CAD : sin_CAD = sin angle_ADB)
axiom (h_sin_CBD : sin_CBD = sin angle_BDC)

-- Definition of point E on the plane of A, B, C such that specific angles are right angles
axiom (E : A × B × C)
axiom (h_AEB : ∀ (a b c : A × B × C), angle (a, b) (b, c) = π / 2)
axiom (h_BEC : ∀ (b e c : B × E × C), angle (b, e) (e, c) = π / 2)
axiom (h_AEC : ∀ (a e c : A × E × C), angle (a, e) (e, c) = π / 2)

-- Goal to prove
theorem cos_AEC_eq_one : cos (angle AEC) = 1 :=
begin
  sorry
end

end cos_AEC_eq_one_l70_70403


namespace distance_between_stations_l70_70157

-- Definitions based on the conditions
def speed_slow : ℝ := 20
def speed_fast : ℝ := 25
def extra_distance : ℝ := 75
def T : ℝ := 15
def D : ℝ := 300
def D_fast : ℝ := D + extra_distance

-- The proof statement
theorem distance_between_stations :
  let D := speed_slow * T,
      D_fast := speed_fast * T in
  D_fast - D = extra_distance → 
  D = 300 → 
  D_fast = 375 → 
  D + D_fast = 675 := by
  intros h1 h2 h3
  sorry

end distance_between_stations_l70_70157


namespace find_f_2008_l70_70057

noncomputable def f : ℝ → ℝ := sorry

axiom f_zero (f : ℝ → ℝ) : f 0 = 2008
axiom f_inequality_1 (f : ℝ → ℝ) (x : ℝ) : f (x + 2) - f x ≤ 3 * 2^x
axiom f_inequality_2 (f : ℝ → ℝ) (x : ℝ) : f (x + 6) - f x ≥ 63 * 2^x

theorem find_f_2008 (f : ℝ → ℝ) : f 2008 = 2^2008 + 2007 :=
by
  apply sorry

end find_f_2008_l70_70057


namespace select_number_in_range_46_to_60_l70_70389

def systematic_sampling (n m start k : ℕ) : ℕ := start + m * (k - 1)

theorem select_number_in_range_46_to_60 :
  ∀ start k, start = 6 ∧ k = 4 → systematic_sampling 1200 15 start k = 51 :=
by
  intros start k h
  cases h
  have h1 : start = 6 := h_left
  have h2 : k = 4 := h_right
  rw [h1, h2, systematic_sampling]
  sorry

end select_number_in_range_46_to_60_l70_70389


namespace total_games_proof_l70_70263

variables (G R : ℝ)

def first_100_games := 100
def percent_won_first_100 := 0.65
def percent_won_remaining := 0.5
def percent_won_total := 0.7

def games_won_first_100 := percent_won_first_100 * first_100_games
def games_won_remaining := percent_won_remaining * R

theorem total_games_proof :
  games_won_first_100 + games_won_remaining = percent_won_total * G ∧
  G = first_100_games + R →
  G = 125 :=
sorry

end total_games_proof_l70_70263


namespace game_win_probability_l70_70615

noncomputable def alexWinsProbability : ℝ := 1/2
noncomputable def melWinsProbability : ℝ := 1/4
noncomputable def chelseaWinsProbability : ℝ := 1/4
noncomputable def totalRounds : ℕ := 8

theorem game_win_probability :
  alexWinsProbability * alexWinsProbability * alexWinsProbability * alexWinsProbability *
  melWinsProbability * melWinsProbability * melWinsProbability *
  chelseaWinsProbability *
  (Nat.factorial totalRounds / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 1)) = 35/512 := by
sorry

end game_win_probability_l70_70615


namespace train_length_l70_70612

noncomputable def length_of_train (time_in_seconds : ℝ) (speed_in_kmh : ℝ) : ℝ :=
  let speed_in_mps := speed_in_kmh * (5 / 18)
  speed_in_mps * time_in_seconds

theorem train_length :
  length_of_train 2.3998080153587713 210 = 140 :=
by
  sorry

end train_length_l70_70612


namespace least_possible_product_of_two_distinct_primes_greater_than_50_l70_70146

open nat

theorem least_possible_product_of_two_distinct_primes_greater_than_50 :
  ∃ p q : ℕ, p ≠ q ∧ prime p ∧ prime q ∧ p > 50 ∧ q > 50 ∧ 
  (∀ p' q' : ℕ, p' ≠ q' → prime p' → prime q' → p' > 50 → q' > 50 → p * q ≤ p' * q') ∧ p * q = 3127 :=
by
  sorry

end least_possible_product_of_two_distinct_primes_greater_than_50_l70_70146


namespace zero_pow_2014_l70_70242

-- Define the condition that zero raised to any positive power is zero
def zero_pow_pos {n : ℕ} (h : 0 < n) : (0 : ℝ)^n = 0 := by
  sorry

-- Use this definition to prove the specific case of 0 ^ 2014 = 0
theorem zero_pow_2014 : (0 : ℝ)^(2014) = 0 := by
  have h : 0 < 2014 := by decide
  exact zero_pow_pos h

end zero_pow_2014_l70_70242


namespace find_integers_l70_70996

theorem find_integers (a b c : ℤ) (h1 : ∃ x : ℤ, a = 2 * x ∧ b = 5 * x ∧ c = 8 * x)
  (h2 : a + 6 = b / 3)
  (h3 : c - 10 = 5 * a / 4) :
  a = 36 ∧ b = 90 ∧ c = 144 :=
by
  sorry

end find_integers_l70_70996


namespace sin_a_max_l70_70845

theorem sin_a_max {a b : ℝ} 
    (h : cos (a + b) + sin (a - b) = cos a + cos b) : 
    ∃ (a : ℝ), sin a ≤ 1 :=
by sorry

end sin_a_max_l70_70845


namespace solve_for_x_l70_70097

theorem solve_for_x : ∀ (x : ℝ), 16^x * 16^x * 16^x = 64^3 → x = 3 / 2 :=
by
  intros x h
  -- Proof would follow here
  sorry

end solve_for_x_l70_70097


namespace equation_1_solution_equation_2_solution_l70_70918

noncomputable def equation_1 (x : ℝ) : Prop :=
x ^ 2 - 5 * x + 6 = 0

theorem equation_1_solution (x : ℝ) :
  equation_1 x → (x = 3 ∨ x = 2) :=
by
  intro h
  rw [equation_1] at h
  -- Factorization can directly be translated
  have h_factor : (x - 3) * (x - 2) = 0, sorry
  cases h_factor
  { left, exact h_factor }
  { right, exact h_factor }

noncomputable def equation_2 (x : ℝ) : Prop :=
(x + 2) * (x - 1) = x + 2

theorem equation_2_solution (x : ℝ) :
  equation_2 x → (x = -2 ∨ x = 2) :=
by
  intro h
  rw [equation_2] at h
  -- After manipulating the equation similarly, we get the factors
  -- The subtraction and factorization steps are merged here
  have h_factor : (x + 2) * (x - 2) = 0, sorry
  cases h_factor
  { left, exact h_factor }
  { right, exact h_factor }

end equation_1_solution_equation_2_solution_l70_70918


namespace composite_expression_l70_70891

theorem composite_expression (x m n : ℤ) (hm : m > 0) (hn : n ≥ 0) : 
  ∃ a b, a > 1 ∧ b > 1 ∧ a * b = x^(4*m) + 2^(4*n+2) := sorry

end composite_expression_l70_70891


namespace complex_number_quadrant_l70_70105

theorem complex_number_quadrant:
  let z := (3 - Complex.I) / (1 + Complex.I) in
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_number_quadrant_l70_70105


namespace largest_n_satisfying_sum_l70_70287

open Nat

theorem largest_n_satisfying_sum (n : ℕ) 
  (H : ∑ k in range (n + 1), k * (choose n k) < 200) : n ≤ 6 :=
sorry

end largest_n_satisfying_sum_l70_70287


namespace probability_white_second_given_red_first_l70_70786

theorem probability_white_second_given_red_first :
  let total_balls := 8
  let red_balls := 5
  let white_balls := 3
  let event_A := red_balls
  let event_B_given_A := white_balls

  (event_B_given_A * (total_balls - 1)) / (event_A * total_balls) = 3 / 7 :=
by
  sorry

end probability_white_second_given_red_first_l70_70786


namespace tennis_racket_price_l70_70816

theorem tennis_racket_price (P : ℝ) : 
    (0.8 * P + 515) * 1.10 + 20 = 800 → 
    P = 242.61 :=
by
  sorry

end tennis_racket_price_l70_70816


namespace isosceles_trapezoid_circumcircle_DR_eq_122_l70_70042

theorem isosceles_trapezoid_circumcircle_DR_eq_122 :
  ∀ (A B C D M E N P Q R : Type)
    [has_coords A]
    [has_coords B]
    [has_coords C]
    [has_coords D]
    [midpoint C D M]
    [circumcircle A B C D (\omega : Type) O]
    [ray_intersects A M (\omega : Type) E]
    [midpoint B E N]
    [intersects B E C D P]
    [rays_intersects O N D C Q]
    [angle_eq P R C (45°)]
    [circumcircle P N Q (\omega2 : Type)]
    [point_on_circumcircle P N Q R]
    [distance_expr D R (\frac{m}{n})]
    (m n : ℕ),
    nat.coprime m n →
    m + n = 122 :=
sorry

end isosceles_trapezoid_circumcircle_DR_eq_122_l70_70042


namespace perfect_square_expression_l70_70519

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, m^2 = (2 * l - n - k) * (2 * l - n + k) / 2 :=
by 
  sorry

end perfect_square_expression_l70_70519


namespace quadratic_function_intersection_l70_70405

theorem quadratic_function_intersection (c : ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, (a = 0.5 * c) → (y = 0.5 * c * (x + 2) ^ 2) ∧ (y = cx + 2c)) :=
begin
  sorry
end

end quadratic_function_intersection_l70_70405


namespace at_least_one_root_l70_70948

theorem at_least_one_root 
  (a b c d : ℝ)
  (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
sorry

end at_least_one_root_l70_70948


namespace find_x_l70_70707

-- Define the given conditions
def conditions (x : ℝ) : Prop :=
  0 < x ∧ x < 180 ∧
  tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)

-- State the theorem we want to prove
theorem find_x : ∃ x : ℝ, conditions x ∧ x = 110 :=
by sorry

end find_x_l70_70707


namespace inequality_proof_l70_70315

open Real

theorem inequality_proof (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
    ( (2 * a + b + c)^2 / (2 * a^2 + (b + c)^2) ) +
    ( (2 * b + c + a)^2 / (2 * b^2 + (c + a)^2) ) +
    ( (2 * c + a + b)^2 / (2 * c^2 + (a + b)^2) ) ≤ 8 :=
by
  sorry

end inequality_proof_l70_70315


namespace fraction_meaningful_range_l70_70507

theorem fraction_meaningful_range (x : ℝ) : (x - 2) ≠ 0 ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_range_l70_70507


namespace square_area_increase_l70_70185

theorem square_area_increase (s : ℝ) : 
  let original_area := s^2 in
  let new_side := 1.10 * s in
  let new_area := new_side^2 in
  ((new_area - original_area) / original_area) * 100 = 21 :=
by
  sorry

end square_area_increase_l70_70185


namespace total_capacity_is_1600_l70_70773

/-- Eight liters is 20% of the capacity of one container. -/
def capacity_of_one_container := 8 / 0.20

/-- Calculate the total capacity of 40 such containers filled with water. -/
def total_capacity_of_40_containers := 40 * capacity_of_one_container

theorem total_capacity_is_1600 :
  total_capacity_of_40_containers = 1600 := by
    -- Proof is skipped using sorry.
    sorry

end total_capacity_is_1600_l70_70773


namespace total_cost_of_fencing_l70_70694

def diameter := 36 -- in meters
def rate := 3.50 -- in Rs. per meter
def pi_approx := 3.14159

-- Circumference of the circle based on the diameter
def circumference := pi_approx * diameter

-- Total cost calculation
def total_cost := circumference * rate

theorem total_cost_of_fencing:
  total_cost ≈ 395.85 :=
by
  sorry

end total_cost_of_fencing_l70_70694


namespace gcd_91_eq_13_count_l70_70714

theorem gcd_91_eq_13_count : 
  { n : ℕ | 1 ≤ n ∧ n ≤ 200 ∧ (Nat.gcd 91 n = 13) }.finite.to_finset.card = 13 :=
by
  sorry

end gcd_91_eq_13_count_l70_70714


namespace circle_radius_is_five_l70_70317

-- Define the conditions
variable (P Q R S : Point)
variable (radius : ℝ)
variable (circle_center : Point)

-- Condition 1
def is_square (P Q R S : Point) : Prop :=
  dist P Q = 10 ∧ dist Q R = 10 ∧ dist R S = 10 ∧ dist S P = 10 ∧ dist P R = dist Q S

-- Condition 2
def circle_through_P_S (P S circle_center : Point) (r : ℝ) : Prop :=
  dist circle_center P = r ∧ dist circle_center S = r

-- Condition 3
def tangent_circle (P Q T circle_center : Point) (r : ℝ) : Prop :=
  T = midpoint P Q ∧ dist circle_center T = r

-- Main proof statement
theorem circle_radius_is_five (P Q R S T circle_center : Point) (r : ℝ)
  (h1 : is_square P Q R S)
  (h2 : circle_through_P_S P S circle_center r)
  (h3 : tangent_circle P Q T circle_center r) : 
  r = 5 := sorry

end circle_radius_is_five_l70_70317


namespace negation_of_proposition_l70_70945

-- Definition for function being odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - f (x)

-- The proposition we are negating
def proposition (f : ℝ → ℝ) : Prop := is_odd f → is_odd (λ x, f (-x))

-- Negate the proposition
theorem negation_of_proposition :
  ¬ (∀ f : ℝ → ℝ, proposition f) ↔ (∀ f : ℝ → ℝ, ¬ is_odd f → ¬ is_odd (λ x, f (-x))) :=
sorry

end negation_of_proposition_l70_70945


namespace det_of_matrix_M_l70_70637

open Matrix

def M : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![2, -4, 4], 
    ![0, 6, -2], 
    ![5, -3, 2]]

theorem det_of_matrix_M : Matrix.det M = -68 :=
by
  sorry

end det_of_matrix_M_l70_70637


namespace inequality_proof_l70_70050

theorem inequality_proof (n : ℕ) 
  (a x : Fin n → ℝ)
  (h_pos_a : ∀ i, 0 < a i)
  (h_pos_x : ∀ i, 0 < x i)
  (h_sum_a : (∑ i, a i) = 1)
  (h_sum_x : (∑ i, x i) = 1) :
  2 * (∑ i j in Finset.range n, if i < j then x i * x j else 0) ≤ 
  (n - 2) / (n - 1) + (∑ i in Finset.range n, a i * (x i)^2 / (1 - a i)) :=
by
  sorry

end inequality_proof_l70_70050


namespace g_eq_g_inv_at_7_over_2_l70_70652

def g (x : ℝ) : ℝ := 3 * x - 7
def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_at_7_over_2 : g (7 / 2) = g_inv (7 / 2) := by
  sorry

end g_eq_g_inv_at_7_over_2_l70_70652


namespace kameron_kangaroos_l70_70038

theorem kameron_kangaroos (K : ℕ) (B_now : ℕ) (rate : ℕ) (days : ℕ)
    (h1 : B_now = 20)
    (h2 : rate = 2)
    (h3 : days = 40)
    (h4 : B_now + rate * days = K) : K = 100 := by
  sorry

end kameron_kangaroos_l70_70038


namespace combined_weight_of_candles_l70_70269

theorem combined_weight_of_candles :
  (∀ (candles: ℕ), ∀ (weight_per_candle: ℕ),
    (weight_per_candle = 8 + 1) →
    (candles = 10 - 3) →
    (candles * weight_per_candle = 63)) :=
by
  intros
  sorry

end combined_weight_of_candles_l70_70269


namespace right_angle_triangle_exists_l70_70832

noncomputable def x : ℕ → ℝ
| 0     := 45
| (n+1) := 90 - (y n)

noncomputable def y : ℕ → ℝ
| 0     := 65
| (n+1) := 90 - (z n)

noncomputable def z : ℕ → ℝ
| 0     := 70
| (n+1) := x n

theorem right_angle_triangle_exists :
  ∃ n, n = 4 ∧ (x n = 90 ∨ y n = 90 ∨ z n = 90) :=
by {
  sorry
}

end right_angle_triangle_exists_l70_70832


namespace max_students_above_average_l70_70789

theorem max_students_above_average (n : ℕ) (h : n = 80) :
  ∃ k, k ≤ n ∧ ∀ (scores : fin n → ℝ), (finset.univ.sum scores) / n < scores k → k = 79 :=
by 
  sorry

end max_students_above_average_l70_70789


namespace box_box_13_eq_24_l70_70711

def sum_of_factors (n : ℕ) : ℕ :=
  (Finset.filter (λ d, d ∣ n) (Finset.range (n + 1))).sum

def box (n : ℕ) : ℕ := 
  sum_of_factors n

theorem box_box_13_eq_24 : box (box 13) = 24 :=
by
  sorry

end box_box_13_eq_24_l70_70711


namespace barbara_blackburn_typing_speed_problem_l70_70233

theorem barbara_blackburn_typing_speed_problem :
  let original_speed := 212
  let speed_reduction := 40
  let time := 20
  let current_speed := original_speed - speed_reduction
  let words_in_document := current_speed * time
  in words_in_document = 3440 := 
by
  let original_speed := 212
  let speed_reduction := 40
  let time := 20
  let current_speed := original_speed - speed_reduction
  let words_in_document := current_speed * time
  show words_in_document = 3440
  sorry

end barbara_blackburn_typing_speed_problem_l70_70233


namespace lisa_needs_additional_marbles_l70_70869

/-- Lisa has 12 friends and 40 marbles. She needs to ensure each friend gets at least one marble and no two friends receive the same number of marbles. We need to find the minimum number of additional marbles needed to ensure this. -/
theorem lisa_needs_additional_marbles : 
  ∀ (friends marbles : ℕ), friends = 12 → marbles = 40 → 
  ∃ (additional_marbles : ℕ), additional_marbles = 38 ∧ 
  (∑ i in finset.range (friends + 1), i) - marbles = additional_marbles :=
by
  intros friends marbles friends_eq marbles_eq 
  use 38
  split
  · exact rfl
  calc (∑ i in finset.range (12 + 1), i) - 40 = 78 - 40 : by norm_num
                                  ... = 38 : by norm_num

end lisa_needs_additional_marbles_l70_70869


namespace unique_location_determined_by_optionD_l70_70224

-- Definitions of the conditions
def optionA : Prop := "Row 2 of Huayu Cinema"
def optionB : Prop := "Central Street of Zhaoyuan County"
def optionC : Prop := "Northward 30 degrees east"
def optionD : Prop := "East longitude 118 degrees, north latitude 40 degrees"

-- Prove that only option D can determine a specific geographical location.
theorem unique_location_determined_by_optionD :
  (¬ (optionA → ∃ loc : String, loc = "specific geographical location")) ∧
  (¬ (optionB → ∃ loc : String, loc = "specific geographical location")) ∧
  (¬ (optionC → ∃ loc : String, loc = "specific geographical location")) ∧
  (optionD → ∃ loc : String, loc = "specific geographical location") :=
by
  sorry

end unique_location_determined_by_optionD_l70_70224


namespace base2_digit_difference_l70_70554

theorem base2_digit_difference :
  let binary_digits (n : ℕ) : ℕ := (nat.log 2 n) + 1 in
  let digit_diff (x y : ℕ) := binary_digits y - binary_digits x in
  digit_diff 300 1500 = 2 :=
by
  sorry

end base2_digit_difference_l70_70554


namespace rectangle_overlap_l70_70793

theorem rectangle_overlap {rect : Type} [has_area rect] 
    (large_rectangle : rect) (small_rectangles : fin 9 → rect) 
    (large_rectangle_area : area large_rectangle = 5)
    (small_rectangle_area : ∀ i, area (small_rectangles i) = 1) : 
    ∃ i j, i ≠ j ∧ area (small_rectangles i ∩ small_rectangles j) ≥ 1 / 9 := 
  sorry

end rectangle_overlap_l70_70793


namespace area_under_curve_l70_70103

theorem area_under_curve : 
  ∫ x in (1/2 : ℝ)..(2 : ℝ), (1 / x) = 2 * Real.log 2 := by
  sorry

end area_under_curve_l70_70103


namespace polynomial_condition_l70_70701

noncomputable def q (x : ℝ) : ℝ := -2 * x + 4

theorem polynomial_condition (x : ℝ) :
  q(q(x)) = x * q(x) + 2 * x ^ 2 :=
by
  sorry

end polynomial_condition_l70_70701


namespace luna_smallest_number_l70_70067

-- Conditions: a number must be five digits long and contain each of the digits 0, 1, 2, 3, and 4 exactly once
-- Question: What is the smallest number on Luna's list that is divisible by both 2 and 5?
-- Correct answer: 12340

theorem luna_smallest_number : ∃ n : ℕ, 
  (n.length = 5 ∧ 
   (∀ d ∈ {0, 1, 2, 3, 4}, d ∈ n.digits 10) ∧
   (∀ d, d ∈ n.digits 10 → d ∈ {0, 1, 2, 3, 4}) ∧
   (nat.mod n 2 = 0) ∧ 
   (nat.mod n 5 = 0) ∧ 
   ∀ m, ((m.length = 5 ∧ 
          (∀ d ∈ {0, 1, 2, 3, 4}, d ∈ m.digits 10) ∧
          (∀ d, d ∈ m.digits 10 → d ∈ {0, 1, 2, 3, 4}) ∧ 
          (nat.mod m 2 = 0) ∧ 
          (nat.mod m 5 = 0)
         ) → n ≤ m)
  ) := 
    n = 12340

end luna_smallest_number_l70_70067


namespace intersection_complement_eq_l70_70758

open Set Real

noncomputable def U := univ
def M : Set ℝ := { x | -1 < x ∧ x < 4 }
def N : Set ℝ := { x | 2 < x ∧ x < 4 }

theorem intersection_complement_eq :
  M ∩ (U \ N) = { x | -1 < x ∧ x ≤ 2 } :=
by {
  sorry
}

end intersection_complement_eq_l70_70758


namespace y_completion_time_l70_70575

theorem y_completion_time (X_days_to_complete : ℕ) (X_worked_days : ℕ) (Y_remaining_work_days : ℕ) :
  X_days_to_complete = 40 →
  X_worked_days = 8 →
  Y_remaining_work_days = 32 →
  ∀ Y_days_to_complete : ℕ, Y_days_to_complete = 40 :=
begin
  intros hX_complete hX_worked hY_remaining,
  let d := 40,
  exact sorry, -- Here we would normally provide the proof, but it’s omitted per instructions.
end

end y_completion_time_l70_70575


namespace number_is_2250_l70_70540

-- Question: Prove that x = 2250 given the condition.
theorem number_is_2250 (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
sorry

end number_is_2250_l70_70540


namespace pourings_remain_one_tenth_l70_70453

theorem pourings_remain_one_tenth (n : ℕ) (h : n + 1 = 10) :
  let rec remain (k : ℕ) : ℚ :=
      if k = 0 then 1
      else remain (k - 1) * (k / (k + 1)) 
  in remain n = 1 / 10 :=
sorry

end pourings_remain_one_tenth_l70_70453


namespace rightmost_box_balls_l70_70887

-- Definitions of initial conditions
def total_balls : ℕ := 100
def total_boxes : ℕ := 26
def leftmost_box_balls : ℕ := 4

-- The sum of any four consecutive boxes
def consecutive_sum_condition (f : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, i + 3 < total_boxes → f i + f (i + 1) + f (i + 2) + f (i + 3) = 15

-- Main statement
theorem rightmost_box_balls (f : ℕ → ℕ) (hf1 : f 0 = leftmost_box_balls)
  (hf2 : consecutive_sum_condition f) 
  (hf3 : (∀ i < total_boxes, 0 ≤ f i)) 
  (hf4 : (finset.range total_boxes).sum f = total_balls) :
  f (total_boxes - 1) = 6 :=
sorry

end rightmost_box_balls_l70_70887


namespace polynomial_has_root_l70_70959

theorem polynomial_has_root {a b c d : ℝ} 
  (h : a * c = 2 * b + 2 * d) : 
  ∃ x : ℝ, (x^2 + a * x + b = 0) ∨ (x^2 + c * x + d = 0) :=
by 
  sorry

end polynomial_has_root_l70_70959


namespace point_value_correct_l70_70388

open Function

def scores_execution : List ℝ := [7.5, 8.1, 9.0, 6.0, 8.5, 7.8, 6.2, 9.0]
def scores_technique : List ℝ := [3.0, 4.5, 2.5, 3.8, 4.0, 4.2, 3.5, 4.7]
def degree_of_difficulty : ℝ := 3.2
def aesthetic_bonus : ℝ := 1.8

noncomputable def point_value_of_dive : ℝ :=
  let execution_scores := scores_execution.erase 6.0 in
  let execution_scores := execution_scores.erase 9.0.headI in
  let technique_scores := scores_technique.erase 2.5 in
  let technique_scores := technique_scores.erase 4.7 in
  let score_sum := execution_scores.sum + technique_scores.sum in
  let difficulty_score := score_sum * degree_of_difficulty in
  difficulty_score + aesthetic_bonus

theorem point_value_correct :
  point_value_of_dive = 226.12 := by
sorry

end point_value_correct_l70_70388


namespace unique_providers_count_l70_70039

theorem unique_providers_count :
  let num_children := 4
  let num_providers := 25
  (∀ s : Fin num_children, s.val < num_providers)
  → num_providers * (num_providers - 1) * (num_providers - 2) * (num_providers - 3) = 303600
:= sorry

end unique_providers_count_l70_70039


namespace erika_rick_savings_l70_70678

def gift_cost : ℕ := 250
def erika_savings : ℕ := 155
def rick_savings : ℕ := gift_cost / 2
def cake_cost : ℕ := 25

theorem erika_rick_savings :
  let total_savings := erika_savings + rick_savings in
  let total_cost := gift_cost + cake_cost in
  total_savings - total_cost = 5 :=
by
  let total_savings := erika_savings + rick_savings
  let total_cost := gift_cost + cake_cost
  sorry

end erika_rick_savings_l70_70678


namespace find_number_l70_70547

theorem find_number (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by
  sorry

end find_number_l70_70547


namespace smallest_non_palindromic_sum_l70_70535

/-- An integer is palindromic if the sequence of decimal digits is the same when read backward -/
def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

/-- The theorem states that 21 is the smallest positive integer that cannot be written 
    as the sum of two nonnegative palindromic integers. -/
theorem smallest_non_palindromic_sum : ∀ (n : ℕ), (n < 21 → ∃ a b : ℕ, is_palindromic a ∧ is_palindromic b ∧ n = a + b) ∧
                                               (¬∃ a b : ℕ, is_palindromic a ∧ is_palindromic b ∧ 21 = a + b) :=
by
  sorry

end smallest_non_palindromic_sum_l70_70535


namespace g_eq_g_inv_at_7_over_2_l70_70654

def g (x : ℝ) : ℝ := 3 * x - 7
def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_at_7_over_2 : g (7 / 2) = g_inv (7 / 2) := by
  sorry

end g_eq_g_inv_at_7_over_2_l70_70654


namespace rate_of_interest_l70_70565

theorem rate_of_interest (P1 P2 T1 T2 SI total_interest R: ℝ) 
  (h1 : P1 = 4000) 
  (h2 : T1 = 2) 
  (h3 : P2 = 2000) 
  (h4 : T2 = 4) 
  (h5 : total_interest = 2200) 
  (h6 : SI = (P1 * T1 * R) / 100 + (P2 * T2 * R) / 100) 
  : total_interest = SI → R = 13.75 := 
begin 
  intros H, 
  sorry 
end

end rate_of_interest_l70_70565


namespace domain_of_g_l70_70246

def quadratic_expr (x : ℝ) : ℝ := x^2 - 5 * x + 8

def domain_g (x : ℝ) : Prop := 
  ∀ y, quadratic_expr(y) ≠  floor(y^2 - 5 * y + 8)

theorem domain_of_g : ∀ x : ℝ, x ∉ (1, 7) ↔ domain_g (x) :=
by {
  sorry
}

end domain_of_g_l70_70246


namespace jelly_beans_distribution_l70_70995

theorem jelly_beans_distribution :
  ∃ x : ℕ, 
  let total_beans := 8000 in
  let remaining_beans := 1600 in
  let people_count := 10 in
  let beans_taken := total_beans - remaining_beans in
  let first_six_taken_each := 2 * x in
  let total_beans_taken_first_six := 6 * first_six_taken_each in
  let total_beans_taken_last_four := 4 * x in
  let total_beans_taken := total_beans_taken_first_six + total_beans_taken_last_four in
  total_beans_taken = beans_taken ∧ x = 400 :=
begin
  use 400,
  let total_beans := 8000,
  let remaining_beans := 1600,
  let beans_taken := total_beans - remaining_beans,
  let people_count := 10,
  let x := 400,
  let first_six_taken_each := 2 * x,
  let total_beans_taken_first_six := 6 * first_six_taken_each,
  let total_beans_taken_last_four := 4 * x,
  let total_beans_taken := total_beans_taken_first_six + total_beans_taken_last_four,
  split,
  {
    -- Prove total_beans_taken = beans_taken 
    show total_beans_taken = beans_taken,
    calc
      total_beans_taken
          = 6 * (2 * x) + 4 * x : by refl
      ... = 12 * x + 4 * x        : by refl
      ... = 16 * x                : by rw ←add_mul
      ... = 16 * 400              : by refl
      ... = 6400                  : by norm_num
      ... = total_beans - remaining_beans 
          : by norm_num,
  },
  {
    -- Prove x = 400 
    show x = 400,
    refl,
  },
end

end jelly_beans_distribution_l70_70995


namespace coeff_x2_in_expansion_of_1_plus_2x_to_5_l70_70927

theorem coeff_x2_in_expansion_of_1_plus_2x_to_5 : 
  -- Assertion: Coefficient of x^2 in (1 + 2x)^5 is 40
  let expansion := (1 + 2*x)^5
  find_coefficient (expansion) (2) = 40
:= sorry

end coeff_x2_in_expansion_of_1_plus_2x_to_5_l70_70927


namespace consecutive_odd_integers_sum_l70_70536

theorem consecutive_odd_integers_sum (x : ℤ) (h : x + (x + 4) = 134) : x + (x + 2) + (x + 4) = 201 := 
by sorry

end consecutive_odd_integers_sum_l70_70536


namespace shifted_parabola_correct_l70_70926

def parabola_shift (x : ℝ) : ℝ := 2 * x^2

def shifted_parabola_expression {x : ℝ} :=
  parabola_shift (x - 4) + 1

theorem shifted_parabola_correct :
  ∀ x : ℝ, shifted_parabola_expression = 2 * (x + 4)^2 + 1 :=
by sorry

end shifted_parabola_correct_l70_70926


namespace complex_number_problem_l70_70321

noncomputable def z : ℂ := 2 + complex.I

theorem complex_number_problem :
  (4 * complex.I) / (z * complex.conj z - 1) = complex.I :=
by
  sorry

end complex_number_problem_l70_70321


namespace geom_seq_sum_abs_l70_70723

theorem geom_seq_sum_abs {a : ℕ → ℤ} {b : ℕ → ℤ} (q : ℤ) (h1 : ∀ n, a n = -4 * n + 5)
  (h2 : ∀ n ≥ 2, q = a n - a (n - 1)) (h3 : b 1 = a 2) :
  ∀ n : ℕ, |b 1| + |b 2| + ... + |b n| = 4 ^ n - 1 :=
sorry

end geom_seq_sum_abs_l70_70723


namespace overtaking_time_l70_70182

theorem overtaking_time (t_a t_b t_k : ℝ) (t_b_start : t_b = t_a - 5) 
                       (overtake_eq1 : 40 * t_b = 30 * t_a)
                       (overtake_eq2 : 60 * (t_a - 10) = 30 * t_a) :
                       t_b = 15 :=
by
  sorry

end overtaking_time_l70_70182


namespace distance_between_parallel_lines_l70_70741

theorem distance_between_parallel_lines :
  let d := (7 * Real.sqrt 2) / 4
  ∃ n : ℝ, (n = 2) → ∀ x y : ℝ, 
      ∃ d : ℝ, 
      (d = (Real.abs (5 - (-1))) / (Real.sqrt (2^2 + 2^2))) ∧
      ∀ x y : ℝ, 2 * x + 2 * y + 5 = 0 → d = (7 * Real.sqrt 2) / 4 := 
by
  intro d,
  use 2,
  intro n_eq_2,
  intro x y,
  use (Real.abs (5 + 1)) / (Real.sqrt (2^2 + 2^2)),
  split,
  { 
    show (Real.abs (5 + 1)) / (Real.sqrt (2^2 + 2^2)) = (7 * Real.sqrt 2) / 4,
    sorry
  },
  {
    intros x y hyp,
    have : 2 * x + 2 * y + 5 = 0 := hyp,
    show d = (7 * Real.sqrt 2) / 4,
    sorry
  }

end distance_between_parallel_lines_l70_70741


namespace two_colonies_limit_l70_70563

def doubles_each_day (size: ℕ) (day: ℕ) : ℕ := size * 2 ^ day

theorem two_colonies_limit (habitat_limit: ℕ) (initial_size: ℕ) : 
  (∀ t, doubles_each_day initial_size t = habitat_limit → t = 20) → 
  initial_size > 0 →
  ∀ t, doubles_each_day (2 * initial_size) t = habitat_limit → t = 20 :=
by
  sorry

end two_colonies_limit_l70_70563


namespace num_valid_6tuples_l70_70353

def f (a b : ℕ) : ℕ := a^2 - a * b + b^2

def valid_6tuple (a : Fin 6 → ℕ) :=
  (∀ i : Fin 6, a i ∈ ({1, 2, 3, 4} : Set ℕ)) ∧
  (∃ c : ℕ, ∀ i : Fin 5, f (a i) (a ⟨i+1, sorry⟩) = c ∧ f (a 5) (a 0) = c)

theorem num_valid_6tuples : ∃ t : Fin 6 → ℕ, valid_6tuple t ∧ (Finset.univ.filter valid_6tuple).card = 40 := 
by sorry

end num_valid_6tuples_l70_70353


namespace part1_measure_of_B_part2_area_of_triangle_l70_70427

-- Definitions for a triangle and its sides and angles
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def given_conditions : Prop :=
  (sin B - sin C) * (b + c) = (sin A - sqrt 2 * sin C) * a ∧
  a / sin A = b / sin B ∧
  b / sin B = c / sin C ∧
  b = sqrt 2 ∧
  a + b + c = 2 + 2 * sqrt 2

-- Statement for part 1
theorem part1_measure_of_B (h : given_conditions) : B = π / 4 := 
  sorry

-- Statement for part 2
theorem part2_area_of_triangle (h : given_conditions) : 
  ∃ area : ℝ, area = 1 := 
  sorry


end part1_measure_of_B_part2_area_of_triangle_l70_70427


namespace sum_of_roots_l70_70048

open Real

theorem sum_of_roots (r s : ℝ) (P : ℝ → ℝ) (Q : ℝ × ℝ) (m : ℝ) :
  (∀ (x : ℝ), P x = x^2) → 
  Q = (20, 14) → 
  (∀ m : ℝ, (m^2 - 80 * m + 56 < 0) ↔ (r < m ∧ m < s)) →
  r + s = 80 :=
by {
  -- sketched proof goes here
  sorry
}

end sum_of_roots_l70_70048


namespace base_b_square_of_15_l70_70766

theorem base_b_square_of_15 (b : ℕ) (h : (b + 5) * (b + 5) = 4 * b^2 + 3 * b + 6) : b = 8 :=
sorry

end base_b_square_of_15_l70_70766


namespace lisa_additional_marbles_l70_70873

theorem lisa_additional_marbles (n : ℕ) (m : ℕ) (s_n : ℕ) :
  n = 12 →
  m = 40 →
  (s_n = (list.sum (list.range (n + 1)))) →
  s_n - m = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp only [list.range_succ, list.sum_range_succ, nat.factorial, nat.succ_eq_add_one, nat.add_succ, mul_add, mul_one, mul_comm n]
  sorry

end lisa_additional_marbles_l70_70873


namespace primes_eq_condition_l70_70660

theorem primes_eq_condition (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) : 
  p + q^2 = r^4 → p = 7 ∧ q = 3 ∧ r = 2 := 
by
  sorry

end primes_eq_condition_l70_70660


namespace remaining_calories_l70_70133

theorem remaining_calories (calories_per_serving : ℕ) (total_servings : ℕ) (eaten_servings : ℕ) :
  calories_per_serving = 110 → total_servings = 16 → eaten_servings = 5 → 
  (total_servings - eaten_servings) * calories_per_serving = 1210 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end remaining_calories_l70_70133


namespace number_of_positive_integers_l70_70929

theorem number_of_positive_integers (n : ℕ) (hpos : 0 < n) (h : 24 - 6 * n ≥ 12) : n = 1 ∨ n = 2 :=
sorry

end number_of_positive_integers_l70_70929


namespace fraction_male_first_class_l70_70072

theorem fraction_male_first_class (total_passengers females total_first_class females_coach_class : ℕ)
                                  (H1 : total_passengers = 120)
                                  (H2 : females = 48)
                                  (H3 : total_first_class = 12)
                                  (H4 : females_coach_class = 40) :
                                  (12 - (48 - 40)) / 12 = 1 / 3 :=
by
  have males_in_first_class := total_first_class - (females - females_coach_class)
  have fraction_males := males_in_first_class / total_first_class
  rw [males_in_first_class, fraction_males, H1, H2, H3, H4]
  have := 4 / 12
  rw this
  norm_num
  sorry

end fraction_male_first_class_l70_70072


namespace unique_solution_l70_70663

-- Define the condition that p, q, r are prime numbers
variables {p q r : ℕ}

-- Assume that p, q, and r are primes
axiom p_prime : Prime p
axiom q_prime : Prime q
axiom r_prime : Prime r

-- Define the main theorem to state that the only solution is (7, 3, 2)
theorem unique_solution : p + q^2 = r^4 → (p, q, r) = (7, 3, 2) :=
by
  sorry

end unique_solution_l70_70663


namespace alice_winning_strategy_l70_70045

theorem alice_winning_strategy (n k : ℤ) (h1 : 1 ≤ k) (h2 : k < n) : 
  (∃ a : bool, a = (¬(n % 2 = 0 ∧ k % 2 = 0))) := 
sorry

end alice_winning_strategy_l70_70045


namespace total_cost_function_range_of_x_minimum_cost_when_x_is_2_l70_70635

def transportation_cost (x : ℕ) : ℕ :=
  300 * x + 500 * (12 - x) + 400 * (10 - x) + 800 * (x - 2)

theorem total_cost_function (x : ℕ) : transportation_cost x = 200 * x + 8400 := by
  -- Simply restate the definition in the theorem form
  sorry

theorem range_of_x (x : ℕ) : 2 ≤ x ∧ x ≤ 10 := by
  -- Provide necessary constraints in theorem form
  sorry

theorem minimum_cost_when_x_is_2 : transportation_cost 2 = 8800 := by
  -- Final cost at minimum x
  sorry

end total_cost_function_range_of_x_minimum_cost_when_x_is_2_l70_70635


namespace interval_length_difference_l70_70115

theorem interval_length_difference (a b : ℝ) :
  (∀ (x : ℝ), a ≤ x ∧ x ≤ b → (1 ≤ 4^(|x|)) ∧ (4^(|x|) ≤ 4)) →
  (b - a = 2 ∧ min (b - a) = 1) → 
  ∃ (diff : ℝ), diff = (2 - 1) :=
begin
  intros h_range h_length,
  use 1,
  sorry
end

end interval_length_difference_l70_70115


namespace find_reciprocal_G_l70_70804

noncomputable def G : Complex := 1 / 2 + (1 / 2) * Complex.I
noncomputable def reciprocal_G : Complex := 1 / G

axiom (A B C D E : Complex)

theorem find_reciprocal_G :
    (reciprocal_G = 1 - Complex.I) →
    (∃ x, x ∈ {A, B, C, D, E} ∧ x = reciprocal_G) →
    D = reciprocal_G := by
  sorry

end find_reciprocal_G_l70_70804


namespace log_2_x_squared_y_eq_6_5_l70_70459

variables {x y : ℝ}
variables (h₀ : 0 < x ∧ x ≠ 1) (h₁ : 0 < y ∧ y ≠ 1)
variables (h₂ : log 2 x = log y 32) (h₃ : x * y^2 = 128)

theorem log_2_x_squared_y_eq_6_5 : log 2 (x^2 * y) = 6.5 := by
  sorry

end log_2_x_squared_y_eq_6_5_l70_70459


namespace sequence_expression_l70_70309

theorem sequence_expression (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) - 2 * a n = 2^n) :
  ∀ n, a n = n * 2^(n - 1) :=
by
  sorry

end sequence_expression_l70_70309


namespace possible_r_values_l70_70735

noncomputable def parabola : Set (ℝ × ℝ) := {p | p.snd^2 = 4 * p.fst}
noncomputable def circle (r : ℝ) := {p | (p.fst - 4)^2 + p.snd^2 = r^2}

theorem possible_r_values :
  ∃ l : Set (ℝ × ℝ), (∃ A B M : ℝ × ℝ, 
    A ∈ parabola ∧ B ∈ parabola ∧ 
    M = (A + B) / 2 ∧ 
    M ∈ circle r ∧ 
    M.fst = 2 ∧ 
    ∃ k : ℝ, 
      l = {p | p.snd = k * (p.fst - A.fst) + A.snd} ∧ 
      ∀ p ∈ l, p = A ∨ p = B
  ) ∧ 
  (∀ l1 l2 : Set (ℝ × ℝ), 
    (∃ A1 B1 M1 : ℝ × ℝ, 
      A1 ∈ parabola ∧ B1 ∈ parabola ∧ 
      M1 = (A1 + B1) / 2 ∧ 
      M1 ∈ circle r ∧ 
      M1.fst = 2 ∧
      l1 = {p | p.snd = k * (p.fst - A1.fst) + A1.snd}) ∧ 
    (∃ A2 B2 M2 : ℝ × ℝ, 
      A2 ∈ parabola ∧ B2 ∈ parabola ∧ 
      M2 = (A2 + B2) / 2 ∧ 
      M2 ∈ circle r ∧ 
      M2.fst = 2 ∧ 
      l2 = {p | p.snd = k * (p.fst - A2.fst) + A2.snd})) 
    → l1 = l2 ∨ l1 ∩ l2 = ∅) ↔ (0 < r ∧ r ≤ 2) :=
by 
  sorry

end possible_r_values_l70_70735


namespace find_top_angle_l70_70978

theorem find_top_angle 
  (sum_of_angles : ∀ (α β γ : ℝ), α + β + γ = 250) 
  (left_is_twice_right : ∀ (α β : ℝ), α = 2 * β) 
  (right_angle_is_60 : ∀ (β : ℝ), β = 60) :
  ∃ γ : ℝ, γ = 70 :=
by
  -- Assume the variables for the angles
  obtain ⟨α, β, γ, h_sum, h_left, h_right⟩ := ⟨_, _, _, sum_of_angles, left_is_twice_right, right_angle_is_60⟩
  -- Your proof here
  sorry

end find_top_angle_l70_70978


namespace circle_center_coordinates_l70_70337

theorem circle_center_coordinates :
  (∀ (ρ θ : ℝ), ρ = 5 * real.cos θ - 5 * real.sqrt 3 * real.sin θ → (∃ x y : ℝ, (x, y).polar_coords = (5, 5 * π / 3))) :=
begin
  sorry
end

end circle_center_coordinates_l70_70337


namespace perfect_square_l70_70533

variable (n k l : ℕ)

theorem perfect_square (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m ^ 2 := by
  have h1 : (2 * l - n - k) * (2 * l - n + k) = (2 * l - n) ^ 2 - k ^ 2 := by sorry
  have h2 : k ^ 2 = 2 * l ^ 2 - n ^ 2 := by sorry
  have h3 : (2 * l - n) ^ 2 - (2 * l ^ 2 - n ^ 2) = 2 * (l - n) ^ 2 := by sorry
  have h4 : (2 * (l - n) ^ 2) / 2 = (l - n) ^ 2 := by sorry
  use (l - n)
  rw [h4]
  sorry

end perfect_square_l70_70533


namespace percentage_of_students_receiving_certificates_l70_70625

theorem percentage_of_students_receiving_certificates
  (boys girls : ℕ)
  (pct_boys pct_girls : ℕ)
  (h_boys : boys = 30)
  (h_girls : girls = 20)
  (h_pct_boys : pct_boys = 30)
  (h_pct_girls : pct_girls = 40)
  :
  (pct_boys * boys + pct_girls * girls) / (100 * (boys + girls)) * 100 = 34 :=
by
  sorry

end percentage_of_students_receiving_certificates_l70_70625


namespace container_capacity_l70_70771

/-- Given a container where 8 liters is 20% of its capacity, calculate the total capacity of 
    40 such containers filled with water. -/
theorem container_capacity (c : ℝ) (h : 8 = 0.20 * c) : 
    40 * c * 40 = 1600 := 
by
  sorry

end container_capacity_l70_70771


namespace prize_expectation_l70_70396

theorem prize_expectation :
  let total_people := 100
  let envelope_percentage := 0.4
  let grand_prize_prob := 0.1
  let second_prize_prob := 0.2
  let consolation_prize_prob := 0.3
  let people_with_envelopes := total_people * envelope_percentage
  let grand_prize_winners := people_with_envelopes * grand_prize_prob
  let second_prize_winners := people_with_envelopes * second_prize_prob
  let consolation_prize_winners := people_with_envelopes * consolation_prize_prob
  let empty_envelopes := people_with_envelopes - (grand_prize_winners + second_prize_winners + consolation_prize_winners)
  grand_prize_winners = 4 ∧
  second_prize_winners = 8 ∧
  consolation_prize_winners = 12 ∧
  empty_envelopes = 16 := by
  sorry

end prize_expectation_l70_70396


namespace perfect_square_of_expression_l70_70529

theorem perfect_square_of_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 := 
by 
  sorry

end perfect_square_of_expression_l70_70529


namespace prime_like_count_l70_70643

def is_composite (n : ℕ) : Prop := ¬(Nat.Prime n) ∧ n > 1

def is_prime_like (n : ℕ) : Prop := 
  is_composite n ∧ ¬(n % 2 = 0) ∧ ¬(n % 3 = 0) ∧ ¬(n % 5 = 0) ∧ ¬(n % 7 = 0)

theorem prime_like_count : 
  let primes_under_1200 := 197 in 
  let n := 1200 in 
  ∃ count : ℕ, count = 174 ∧ count = ((n - 1) - 193 - (count_of_union_S_2_to_S_7 n)) :=
  sorry

end prime_like_count_l70_70643


namespace parallel_lines_slope_eq_l70_70930

theorem parallel_lines_slope_eq (m : ℝ) :
  let l1 := 3 * x + 2 * y - 2 = 0
      l2 := (2 * m - 1) * x + m * y + 1 = 0
  in (l1 ∥ l2) → m = 2 :=
sorry

end parallel_lines_slope_eq_l70_70930


namespace congruent_figures_overlap_by_translation_and_rotation_l70_70473

-- Definitions based on the conditions:
def is_congruent (fig1 fig2 : Type) : Prop :=
  ∃ (f : Type → Type), ∃ (g : f fig1 → f fig2), isometry g

def in_same_plane (fig1 fig2 : Type) : Prop :=
  ∃ (plane : Type), in_plane plane fig1 ∧ in_plane plane fig2

def not_parallel (fig1 fig2 : Type) : Prop :=
  ¬ parallel fig1 fig2

-- The main theorem to prove:
theorem congruent_figures_overlap_by_translation_and_rotation
  (fig1 fig2 : Type)
  (hc : is_congruent fig1 fig2)
  (hp : in_same_plane fig1 fig2)
  (hnp : not_parallel fig1 fig2) :
  ∃ (f : Type → Type), ∃ (g : f fig1 → f fig2), 
  is_translation g ∨ is_rotation g
:=
sorry

end congruent_figures_overlap_by_translation_and_rotation_l70_70473


namespace lisa_needs_additional_marbles_l70_70871

/-- Lisa has 12 friends and 40 marbles. She needs to ensure each friend gets at least one marble and no two friends receive the same number of marbles. We need to find the minimum number of additional marbles needed to ensure this. -/
theorem lisa_needs_additional_marbles : 
  ∀ (friends marbles : ℕ), friends = 12 → marbles = 40 → 
  ∃ (additional_marbles : ℕ), additional_marbles = 38 ∧ 
  (∑ i in finset.range (friends + 1), i) - marbles = additional_marbles :=
by
  intros friends marbles friends_eq marbles_eq 
  use 38
  split
  · exact rfl
  calc (∑ i in finset.range (12 + 1), i) - 40 = 78 - 40 : by norm_num
                                  ... = 38 : by norm_num

end lisa_needs_additional_marbles_l70_70871


namespace equal_segments_AE_AF_l70_70439

theorem equal_segments_AE_AF 
  (A B C D E F : Type)
  [IsTriangle A B C]
  (Γ : Circle A B C)
  (tangent_intersect : ∃ D, is_tangent Γ A D ∧ lies_on D (line B C))
  (bisector_intersections : ∃ E F, is_angle_bisector (angle C D A) (line E F) (line A B) (line A C)) :
  segment_length A E = segment_length A F :=
sorry

end equal_segments_AE_AF_l70_70439


namespace power_function_value_at_9_l70_70736

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

-- Given the conditions
variable (α : ℝ)
variable (f : ℝ → ℝ)
variable h1 : f = power_function α
variable h2 : f 2 = (2 : ℝ)^(-1/2)

-- We need to prove
theorem power_function_value_at_9 :
  f 9 = (1 : ℝ) / 3 :=
by
  -- Add the proof steps here
  sorry

end power_function_value_at_9_l70_70736


namespace value_of_x_for_g_equals_g_inv_l70_70651

noncomputable def g (x : ℝ) : ℝ := 3 * x - 7
  
noncomputable def g_inv (x : ℝ) : ℝ := (x + 7) / 3
  
theorem value_of_x_for_g_equals_g_inv : ∃ x : ℝ, g x = g_inv x ∧ x = 3.5 :=
by
  sorry

end value_of_x_for_g_equals_g_inv_l70_70651


namespace evaluate_expression_at_2_l70_70431

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 9) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := 2 * x - 3

theorem evaluate_expression_at_2 : f (g 2) + g (f 2) = 331 / 20 :=
by
  sorry

end evaluate_expression_at_2_l70_70431


namespace exists_quad_root_l70_70961

theorem exists_quad_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (∃ x, x^2 + a * x + b = 0) ∨ (∃ x, x^2 + c * x + d = 0) :=
sorry

end exists_quad_root_l70_70961


namespace student_question_choice_l70_70566

/-- A student needs to choose 8 questions from part A and 5 questions from part B. Both parts contain 10 questions each.
   This Lean statement proves that the student can choose the questions in 11340 different ways. -/
theorem student_question_choice : (Nat.choose 10 8) * (Nat.choose 10 5) = 11340 := by
  sorry

end student_question_choice_l70_70566


namespace find_f_find_sin_2alpha_l70_70324

-- Given definitions and conditions
def f (x : ℝ) : ℝ := sin(ω * x + φ) where ω > 0 and 0 < φ < π
def dist_between_axes (f : ℝ -> ℝ) : ℝ := π / 2
def is_even (f : ℝ -> ℝ) := ∀ x, f(x + π/2) = f(- (x + π/2))

-- Question 1: Find the analytical expression of f(x).
theorem find_f : f(x) = cos(2 * x) :=
sorry

-- Question 2: Given another condition, find the value of sin 2α.
theorem find_sin_2alpha (α : ℝ) (hα : α > 0 ∧ α < π/2) (h : f(α/2 + π/12) = 3/5) : 
sin(2 * α) = (24 + 7 * sqrt 3) / 50 :=
sorry

end find_f_find_sin_2alpha_l70_70324


namespace crackers_per_friend_l70_70448

theorem crackers_per_friend (Total_crackers Left_crackers Friends : ℕ) (h1 : Total_crackers = 23) (h2 : Left_crackers = 11) (h3 : Friends = 2):
  (Total_crackers - Left_crackers) / Friends = 6 :=
by
  sorry

end crackers_per_friend_l70_70448


namespace valid_parametrizations_l70_70491

-- Definitions for the given points and directions
def pointA := (0, 4)
def dirA := (3, -1)

def pointB := (4/3, 0)
def dirB := (1, -3)

def pointC := (-2, 10)
def dirC := (-3, 9)

-- Line equation definition
def line (x y : ℝ) : Prop := y = -3 * x + 4

-- Proof statement
theorem valid_parametrizations :
  (line pointB.1 pointB.2 ∧ dirB.2 = -3 * dirB.1) ∧
  (line pointC.1 pointC.2 ∧ dirC.2 / dirC.1 = 3) :=
by
  sorry

end valid_parametrizations_l70_70491


namespace ellie_runs_8_miles_in_24_minutes_l70_70264

theorem ellie_runs_8_miles_in_24_minutes (time_max : ℝ) (distance_max : ℝ) 
  (time_ellie_fraction : ℝ) (distance_ellie : ℝ) (distance_ellie_final : ℝ)
  (h1 : distance_max = 6) 
  (h2 : time_max = 36) 
  (h3 : time_ellie_fraction = 1/3) 
  (h4 : distance_ellie = 4) 
  (h5 : distance_ellie_final = 8) :
  ((time_ellie_fraction * time_max) / distance_ellie) * distance_ellie_final = 24 :=
by
  sorry

end ellie_runs_8_miles_in_24_minutes_l70_70264


namespace total_pebbles_l70_70878

theorem total_pebbles (white_pebbles : ℕ) (red_pebbles : ℕ)
  (h1 : white_pebbles = 20)
  (h2 : red_pebbles = white_pebbles / 2) :
  white_pebbles + red_pebbles = 30 := by
  sorry

end total_pebbles_l70_70878


namespace vasya_can_place_99_tokens_l70_70082

theorem vasya_can_place_99_tokens (board : ℕ → ℕ → Prop) (h : ∀ i j, board i j → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50) :
  ∃ new_tokens : ℕ → ℕ → Prop, (∀ i j, new_tokens i j → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50) ∧
  (∀ i, (∑ j in finset.range 50, if board i j ∨ new_tokens i j then 1 else 0) % 2 = 0) ∧
  (∀ j, (∑ i in finset.range 50, if board i j ∨ new_tokens i j then 1 else 0) % 2 = 0) ∧
  (finset.sum (finset.product (finset.range 50) (finset.range 50)) 
             (λ (ij : ℕ × ℕ), if new_tokens ij.1 ij.2 then 1 else 0) ≤ 99) :=
sorry

end vasya_can_place_99_tokens_l70_70082


namespace find_constants_m_n_l70_70806

-- Conditions as definitions in Lean
def norm_OA : ℝ := 1
def norm_OB : ℝ := 1
def norm_OC : ℝ := Real.sqrt 2
def tan_angle_AOC : ℝ := 7
def angle_BOC : ℝ := Real.pi / 4

-- Question and Answer
def constants : ℝ × ℝ := (5 / 4, 7 / 4)

theorem find_constants_m_n :
  ∃ (m n : ℝ), 
    (∥overline OC∥ = Real.sqrt 2) ∧ 
    (tan(angle_AOC) = 7) ∧ 
    (angle BOC = Real.pi / 4) ∧ 
    (overline OC = m * overline OA + n * overline OB) ∧ 
    (m, n) = (5 / 4, 7 / 4) :=
sorry

end find_constants_m_n_l70_70806


namespace treasure_distribution_l70_70708

noncomputable def calculate_share (investment total_investment total_value : ℝ) : ℝ :=
  (investment / total_investment) * total_value

theorem treasure_distribution 
  (investment_fonzie investment_aunt_bee investment_lapis investment_skylar investment_orion total_treasure : ℝ)
  (total_investment : ℝ)
  (h : total_investment = investment_fonzie + investment_aunt_bee + investment_lapis + investment_skylar + investment_orion) :
  calculate_share investment_fonzie total_investment total_treasure = 210000 ∧
  calculate_share investment_aunt_bee total_investment total_treasure = 255000 ∧
  calculate_share investment_lapis total_investment total_treasure = 270000 ∧
  calculate_share investment_skylar total_investment total_treasure = 225000 ∧
  calculate_share investment_orion total_investment total_treasure = 240000 :=
by
  sorry

end treasure_distribution_l70_70708


namespace value_of_x_for_g_equals_g_inv_l70_70649

noncomputable def g (x : ℝ) : ℝ := 3 * x - 7
  
noncomputable def g_inv (x : ℝ) : ℝ := (x + 7) / 3
  
theorem value_of_x_for_g_equals_g_inv : ∃ x : ℝ, g x = g_inv x ∧ x = 3.5 :=
by
  sorry

end value_of_x_for_g_equals_g_inv_l70_70649


namespace find_c_range_l70_70017

variable {x c : ℝ}

-- Definitions and Conditions
def three_times_point (x : ℝ) := (x, 3 * x)

def quadratic_curve (x c : ℝ) := -x^2 - x + c

def in_range (x : ℝ) := -3 < x ∧ x < 1

/-- Lean theorem stating the mathematically equivalent proof problem -/
theorem find_c_range (h : in_range x) (h1 : ∃ x, quadratic_curve x c = 3 * x) : -4 ≤ c ∧ c < 5 := 
sorry

end find_c_range_l70_70017


namespace centroids_form_equilateral_l70_70041

noncomputable def centroid (A B C : Point) : Point := sorry

theorem centroids_form_equilateral
  (A B C D E F G1 G2 G3 centroid_G : Point)
  (h_collinear : collinear A B C)
  (h_between : between A B C)
  (h_eq_triangles : equilateral_triangle A B D ∧ equilateral_triangle B C E ∧ equilateral_triangle C A F)
  (h_same_side : same_side_line A C D E)
  (h_opposite_side : opposite_side_line A C F)
  (h_centroids : centroid A B D = G1 ∧ centroid B C E = G2 ∧ centroid C A F = G3)
  (h_centroid_equilateral : centroid G1 G2 G3 = centroid_G)
  (h_centroid_on_AC : lies_on_line centroid_G A C) :
  (equilateral_triangle G1 G2 G3) ∧ (lies_on_line centroid_G A C) :=
begin
  sorry
end

end centroids_form_equilateral_l70_70041


namespace pie_left_is_30_percent_l70_70632

def Carlos_share : ℝ := 0.60
def remaining_after_Carlos : ℝ := 1 - Carlos_share
def Jessica_share : ℝ := 0.25 * remaining_after_Carlos
def final_remaining : ℝ := remaining_after_Carlos - Jessica_share

theorem pie_left_is_30_percent :
  final_remaining = 0.30 :=
sorry

end pie_left_is_30_percent_l70_70632


namespace trajectory_equation_l70_70118

theorem trajectory_equation (x y : ℝ) :
  (√((x - 5)^2 + y^2) / abs (x - 9 / 5) = 5 / 3) →
  x^2 / 9 - y^2 / 16 = 1 :=
by
  sorry

end trajectory_equation_l70_70118


namespace complement_is_correct_l70_70757

variable (U : Set ℕ) (A : Set ℕ)

def complement (U : Set ℕ) (A : Set ℕ) : Set ℕ :=
  { x ∈ U | x ∉ A }

theorem complement_is_correct :
  (U = {1, 2, 3, 4, 5, 6, 7}) →
  (A = {2, 4, 5}) →
  complement U A = {1, 3, 6, 7} :=
by
  sorry

end complement_is_correct_l70_70757


namespace solve_equation_l70_70477

theorem solve_equation : ∀ (x : ℝ), (2 * x + 5 = 3 * x - 2) → (x = 7) :=
by
  intro x
  intro h
  sorry

end solve_equation_l70_70477


namespace simplify_expression_l70_70175

theorem simplify_expression : 
  (81 ^ (1 / Real.logb 5 9) + 3 ^ (3 / Real.logb (Real.sqrt 6) 3)) / 409 * 
  ((Real.sqrt 7) ^ (2 / Real.logb 25 7) - 125 ^ (Real.logb 25 6)) = 1 :=
by 
  sorry

end simplify_expression_l70_70175


namespace kenny_cost_per_book_l70_70824

theorem kenny_cost_per_book (B : ℕ) :
  let lawn_charge := 15
  let mowed_lawns := 35
  let video_game_cost := 45
  let video_games := 5
  let total_earnings := lawn_charge * mowed_lawns
  let spent_on_video_games := video_game_cost * video_games
  let remaining_money := total_earnings - spent_on_video_games
  remaining_money / B = 300 / B :=
by
  sorry

end kenny_cost_per_book_l70_70824


namespace min_value_of_expression_l70_70428

theorem min_value_of_expression (a b : ℝ) (h : 2 * a + b = 6) : 2^a + (sqrt 2)^b ≥ 4 * sqrt 2 := 
sorry

end min_value_of_expression_l70_70428


namespace cereal_original_price_l70_70065

-- Define the known conditions as constants
def initial_money : ℕ := 60
def celery_price : ℕ := 5
def bread_price : ℕ := 8
def milk_full_price : ℕ := 10
def milk_discount : ℕ := 10
def milk_price : ℕ := milk_full_price - (milk_full_price * milk_discount / 100)
def potato_price : ℕ := 1
def potato_quantity : ℕ := 6
def potatoes_total_price : ℕ := potato_price * potato_quantity
def coffee_remaining_money : ℕ := 26
def total_spent_exclude_coffee : ℕ := initial_money - coffee_remaining_money
def spent_on_other_items : ℕ := celery_price + bread_price + milk_price + potatoes_total_price
def spent_on_cereal : ℕ := total_spent_exclude_coffee - spent_on_other_items
def cereal_discount : ℕ := 50

theorem cereal_original_price :
  (spent_on_other_items = celery_price + bread_price + milk_price + potatoes_total_price) →
  (total_spent_exclude_coffee = initial_money - coffee_remaining_money) →
  (spent_on_cereal = total_spent_exclude_coffee - spent_on_other_items) →
  (spent_on_cereal * 2 = 12) :=
by {
  -- proof here
  sorry
}

end cereal_original_price_l70_70065


namespace slope_range_of_line_intersecting_circle_l70_70021

theorem slope_range_of_line_intersecting_circle : 
  ∀ (k : ℝ), let center : ℝ × ℝ := (2, 2)
             let radius : ℝ := 3 * Real.sqrt 2
             let circle_eq := ∀ (x y : ℝ), x^2 + y^2 - 4 * x - 4 * y - 10 = 0
             let line_eq := ∀ (x y : ℝ), y = k * x
             let distance := (2 * k - 2) / Real.sqrt (1 + k^2)
             (exists p1 p2 p3 : ℝ × ℝ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ circle_eq p1.fst p1.snd = 0 ∧ circle_eq p2.fst p2.snd = 0 ∧ circle_eq p3.fst p3.snd = 0 ∧ ∀ p, line_eq p.fst p.snd → Real.dist center p = 2 * Real.sqrt 2) →
             2 - Real.sqrt 3 ≤ k ∧ k ≤ 2 + Real.sqrt 3 := 
by
  intro k center radius circle_eq line_eq distance
  intro h
  sorry

end slope_range_of_line_intersecting_circle_l70_70021


namespace fraction_zero_solution_l70_70168

theorem fraction_zero_solution (x : ℝ) (h1 : |x| - 3 = 0) (h2 : x + 3 ≠ 0) : x = 3 := 
sorry

end fraction_zero_solution_l70_70168


namespace find_m_value_l70_70730

def point (ℝ : Type) := (ℝ × ℝ)

def vector (A B : point ℝ) : point ℝ := (B.1 - A.1, B.2 - A.2)

def dot_product (v1 v2 : point ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : point ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def condition_satisfied (A B C : point ℝ) (m : ℝ) : Prop :=
  let AB := vector A B
  let CB := vector C B
  let AC := vector A C
  dot_product AB CB = magnitude AC

theorem find_m_value (m : ℝ) (h : condition_satisfied (2, m) (1, 2) (3, 1) m) : m = 7 / 3 :=
sorry

end find_m_value_l70_70730


namespace find_b_perpendicular_l70_70934

theorem find_b_perpendicular
  (b : ℝ)
  (line1 : ∀ x y : ℝ, 2 * x - 3 * y + 5 = 0)
  (line2 : ∀ x y : ℝ, b * x - 3 * y + 1 = 0)
  (perpendicular : (2 / 3) * (b / 3) = -1)
  : b = -9/2 :=
sorry

end find_b_perpendicular_l70_70934


namespace matrix_multiplication_correct_l70_70636

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 0], ![7, -2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -1], ![0, 2]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![15, -3], ![35, -11]]

theorem matrix_multiplication_correct : A ⬝ B = C := by
  sorry

end matrix_multiplication_correct_l70_70636


namespace race_result_l70_70561

theorem race_result
    (distance_race : ℕ)
    (distance_diff : ℕ)
    (distance_second_start_diff : ℕ)
    (speed_xm speed_xl : ℕ)
    (h1 : distance_race = 100)
    (h2 : distance_diff = 20)
    (h3 : distance_second_start_diff = 20)
    (xm_wins_first_race : speed_xm * distance_race >= speed_xl * (distance_race - distance_diff)) :
    speed_xm * (distance_race + distance_second_start_diff) >= speed_xl * (distance_race + distance_diff) :=
by
  sorry

end race_result_l70_70561


namespace number_of_sweet_potatoes_sold_to_mrs_adams_l70_70451

def sweet_potatoes_harvested := 80
def sweet_potatoes_sold_to_mr_lenon := 15
def sweet_potatoes_unsold := 45

def sweet_potatoes_sold_to_mrs_adams :=
  sweet_potatoes_harvested - sweet_potatoes_sold_to_mr_lenon - sweet_potatoes_unsold

theorem number_of_sweet_potatoes_sold_to_mrs_adams :
  sweet_potatoes_sold_to_mrs_adams = 20 := by
  sorry

end number_of_sweet_potatoes_sold_to_mrs_adams_l70_70451


namespace snail_stops_at_25_26_l70_70203

def grid_width : ℕ := 300
def grid_height : ℕ := 50

def initial_position : ℕ × ℕ := (1, 1)

def snail_moves_in_spiral (w h : ℕ) (initial : ℕ × ℕ) : ℕ × ℕ := (25, 26)

theorem snail_stops_at_25_26 :
  snail_moves_in_spiral grid_width grid_height initial_position = (25, 26) :=
sorry

end snail_stops_at_25_26_l70_70203


namespace find_johns_pace_l70_70815

def johns_final_push_pace (j s : ℝ) (d_behind d_ahead : ℝ) (t : ℝ) : Prop :=
  s = 3.7 ∧ d_behind = 16 ∧ d_ahead = 2 ∧ t = 36 ∧ (j * t = s * t + d_behind + d_ahead)

theorem find_johns_pace : ∃ j : ℝ, johns_final_push_pace j 3.7 16 2 36 ∧ j = 4.2 :=
begin
  let s := 3.7,
  let d_behind := 16,
  let d_ahead := 2,
  let t := 36,
  let j := 4.2,
  existsi j,
  unfold johns_final_push_pace,
  split,
  { split,
    { refl },
    split,
    { refl },
    split,
    { refl },
    { split,
      { refl },
      { calc
          j * t = 4.2 * 36 : by refl
            ... = 151.2 : by norm_num
            ... = 133.2 + 16 + 2 : by norm_num
            ... = s * t + d_behind + d_ahead : by simp [s, t, d_behind, d_ahead]
        }
    }
  },
  { refl }
end

end find_johns_pace_l70_70815


namespace determine_g_l70_70657

variable {R : Type*} [CommRing R]

theorem determine_g (g : R → R) (x : R) :
  (4 * x^5 + 3 * x^3 - 2 * x + 1 + g x = 7 * x^3 - 5 * x^2 + 4 * x - 3) →
  g x = -4 * x^5 + 4 * x^3 - 5 * x^2 + 6 * x - 4 :=
by
  sorry

end determine_g_l70_70657


namespace distinct_painted_cubes_l70_70591

theorem distinct_painted_cubes :
  ∃ n : ℕ, n = 4 ∧ ( ∀ (c : Cube), c.is_painted_correctly → c.is_distinct (modulo_rotations) (Yellow, Orange, Purple) n ) :=
sorry

end distinct_painted_cubes_l70_70591


namespace tea_garden_problem_pruned_to_wild_conversion_l70_70884

-- Definitions and conditions as per the problem statement
def total_area : ℕ := 16
def total_yield : ℕ := 660
def wild_yield_per_mu : ℕ := 30
def pruned_yield_per_mu : ℕ := 50

-- Lean 4 statement as per the proof problem
theorem tea_garden_problem :
  ∃ (x y : ℕ), (x + y = total_area) ∧ (wild_yield_per_mu * x + pruned_yield_per_mu * y = total_yield) ∧
  x = 7 ∧ y = 9 :=
sorry

-- Additional theorem for the conversion condition
theorem pruned_to_wild_conversion :
  ∀ (a : ℕ), (wild_yield_per_mu * (7 + a) ≥ pruned_yield_per_mu * (9 - a)) → a ≥ 3 :=
sorry

end tea_garden_problem_pruned_to_wild_conversion_l70_70884


namespace largest_internal_right_angles_l70_70058

theorem largest_internal_right_angles (n : ℕ) (hn : n ≥ 5) : 
  ∃ k, ( ∀ (polygon : { p // is_polygon p ∧ vertices p = n }), internal_right_angles polygon = k ) → 
  k = (2 * n) / 3 + 1 :=
begin
  sorry
end

end largest_internal_right_angles_l70_70058


namespace monotonic_decreasing_interval_l70_70494

open Real

def f (x : ℝ) : ℝ := logBase (2 / 3) (x ^ 2 - x)

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (1 < x) → ∃ I : set ℝ, I = {y | 1 < y} ∧ ∀ a b : ℝ, a ∈ I ∧ b ∈ I ∧ a < b → f b ≤ f a := 
by
  sorry

end monotonic_decreasing_interval_l70_70494


namespace find_original_petrol_price_l70_70214

noncomputable def original_petrol_price (P : ℝ) : Prop :=
  let original_amount := 300 / P in
  let reduced_price := 0.85 * P in
  let new_amount := 300 / reduced_price in
  new_amount = original_amount + 7

theorem find_original_petrol_price (P : ℝ) (h : original_petrol_price P) : P ≈ 45 / 5.95 :=
by {
  sorry
}

end find_original_petrol_price_l70_70214


namespace similar_triangles_A_l70_70223

noncomputable def A''_midpoint (B C : Point) : Point := (B + C) / 2

theorem similar_triangles_A''B''C''_ABC
  (A B C G B'_ C'_ A'' C'' B'' : Point)
  (hA : ¬collinear A B C)
  (hG : centroid A B C G)
  (hpar : line_through G B' C' ∧ parallel (line_through B C) (line_through B' C'))
  (hA'': A'' = A''_midpoint B C)
  (hC'': intersect (line_through B' C) (line_through B G) = some C'')
  (hB'': intersect (line_through C'_ B) (line_through C G) = some B'') :
  similar (triangle A'' B'' C'') (triangle A B C) := 
sorry

end similar_triangles_A_l70_70223


namespace final_result_is_102_l70_70609

theorem final_result_is_102 (n : ℕ) (h1 : n = 121) (h2 : 2 * n - 140 = 102) : n = 121 :=
by
  assumption

end final_result_is_102_l70_70609


namespace obtuse_angle_PSQ_l70_70799

-- Define the basic elements of the problem: points, triangle, and angle bisectors
variables {P Q R S : Point}
variables (TRI : Triangle P Q R)
variables (isosceles_right_TRI : IsoscelesRightTriangle TRI)
variables (angle_P : Angle TRI.vertexP = 45)
variables (angle_Q : Angle TRI.vertexQ = 45)
variables (bisector_SP : AngleBisector TRI.vertexP S)
variables (bisector_SQ : AngleBisector TRI.vertexQ S)

-- The statement to be proved
theorem obtuse_angle_PSQ :
  (TRI : IsoscelesRightTriangle TRI) ∧
  (angle_P : Angle TRI.vertexP = 45) ∧
  (angle_Q : Angle TRI.vertexQ = 45) ∧
  (bisector_SP : AngleBisector TRI.vertexP S) ∧
  (bisector_SQ : AngleBisector TRI.vertexQ S) →
  Angle (TRI.vertexP S TRI.vertexQ) = 135 :=
by sorry

end obtuse_angle_PSQ_l70_70799


namespace sale_price_correct_l70_70126

noncomputable def sale_price (cost_price profit_rate tax_rate : ℝ) : ℝ :=
  let selling_price := cost_price + profit_rate * cost_price in
  selling_price + tax_rate * selling_price

theorem sale_price_correct :
  sale_price 526.50 0.17 0.10 = 677.6055 := 
by
  sorry

end sale_price_correct_l70_70126


namespace number_is_2250_l70_70538

-- Question: Prove that x = 2250 given the condition.
theorem number_is_2250 (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
sorry

end number_is_2250_l70_70538


namespace inequality_proof_l70_70905

theorem inequality_proof (a b c : ℝ) (h : a * b * c = 1) : 
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 := 
by
  sorry

end inequality_proof_l70_70905


namespace rectangle_proof_l70_70562

def RectangleLength (AB BC PQ : ℝ) :=
  BC = 19 ∧ PQ = 87 → AB = 193

theorem rectangle_proof
  (AB BC PQ : ℝ)
  (hBC : BC = 19)
  (hPQ : PQ = 87)
  (h1 : XY = YB + BC + CZ)
  (h2 : XY = ZW = WD + DA + AX)
  (h3 : PQ ∥ AB)
  (h4 : 2 * AB + 2 * BC = 4 * XY)
  (h5 : 2 * AB + 2 * BC = 4 * ZW)
  (h6 : 0.5 * (PQ + XY) * (BC / 2) = 0.5 * 19 * AB)
  (h7 : 0.5 * (PQ + ZW) * (BC / 2) = 0.5 * 19 * AB):
  RectangleLength AB BC PQ := by
  sorry

end rectangle_proof_l70_70562


namespace smallest_sum_of_three_distinct_elements_l70_70969

theorem smallest_sum_of_three_distinct_elements : 
  ∃ (a b c : ℤ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ {a, b, c} ⊆ {7, 25, -1, 12, -3} ∧ a + b + c = 3 :=
by
  have setOfNumbers : set ℤ := {7, 25, -1, 12, -3}
  have smallest_numbers : set ℤ := {-3, -1, 7}
  use [-3, -1, 7]
  split
  {
    split
    {
      exact ne_of_lt h
    }
    {
      exact ne_of_lt h
    }
  }
  split
  {
    exact ne_of_lt h
  }
  split
  {
    apply set.subset_set_of_mem h
  }
  {
    apply set.subset_set_of_mem h
  }
  exact sorry

end smallest_sum_of_three_distinct_elements_l70_70969


namespace competition_score_difference_l70_70393

theorem competition_score_difference :
  let perc_60 := 0.20
  let perc_75 := 0.25
  let perc_85 := 0.15
  let perc_90 := 0.30
  let perc_95 := 0.10
  let mean := (perc_60 * 60) + (perc_75 * 75) + (perc_85 * 85) + (perc_90 * 90) + (perc_95 * 95)
  let median := 85
  (median - mean = 5) := by
sorry

end competition_score_difference_l70_70393


namespace linear_function_convex_and_concave_l70_70457

variables {a b : ℝ}

def linear_function (x : ℝ) : ℝ := a * x + b

theorem linear_function_convex_and_concave :
  convex_on ℝ (set.univ) linear_function ∧ concave_on ℝ (set.univ) linear_function :=
sorry

end linear_function_convex_and_concave_l70_70457


namespace original_average_score_of_class_l70_70104

theorem original_average_score_of_class {A : ℝ} 
  (num_students : ℝ) 
  (grace_marks : ℝ) 
  (new_average : ℝ) 
  (h1 : num_students = 35) 
  (h2 : grace_marks = 3) 
  (h3 : new_average = 40)
  (h_total_new : 35 * new_average = 35 * A + 35 * grace_marks) :
  A = 37 :=
by 
  -- Placeholder for proof
  sorry

end original_average_score_of_class_l70_70104


namespace clocks_correct_time_simultaneously_l70_70580

theorem clocks_correct_time_simultaneously :
  ∃ d : ℕ, (d % 720 = 0) ∧ (d % 480 = 0) ∧ d = 1440 :=
by
  existsi 1440
  split
  . sorry
  split
  . sorry
  . rfl

end clocks_correct_time_simultaneously_l70_70580


namespace quadratic_root_exists_l70_70954

theorem quadratic_root_exists {a b c d : ℝ} (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
by
  sorry

end quadratic_root_exists_l70_70954


namespace space_shuttle_speed_kmph_l70_70607

-- Question: Prove that the speed of the space shuttle in kilometers per hour is 32400, given it travels at 9 kilometers per second and there are 3600 seconds in an hour.
theorem space_shuttle_speed_kmph :
  (9 * 3600 = 32400) :=
by
  sorry

end space_shuttle_speed_kmph_l70_70607


namespace triangle_is_isosceles_l70_70785

theorem triangle_is_isosceles (A B C : ℝ) (h : 2 * sin A * cos B = sin C) : 
  ∃ a b c : ℝ, is_isosceles_triangle a b c :=
sorry

end triangle_is_isosceles_l70_70785


namespace stabilize_sequence_stabilized_value_l70_70712

-- Definitions
def O (n : ℕ) : ℕ := 
  if n = 0 then 1 -- O is not defined for 0, assume 1 for convenience
  else (∏ p in (nat.factorization n).keys.filter (λ p => p % 2 = 1), p ^ (nat.factorization n p))

noncomputable def gcd := Nat.gcd

-- Sequence construction
def seq (a b : ℕ) : ℕ → ℕ
| 0       => a
| 1       => b
| (n + 2) => O (seq a b n + seq a b (n + 1))

theorem stabilize_sequence (a b : ℕ) :
  ∃ m δ, ∀ n, m ≤ n → seq a b n = δ :=
sorry

theorem stabilized_value (a b : ℕ) (m δ : ℕ)
  (stabilized : ∀ n, m ≤ n → seq a b n = δ) :
  δ = O (gcd a b) :=
sorry

end stabilize_sequence_stabilized_value_l70_70712


namespace top_angle_is_70_l70_70982

theorem top_angle_is_70
  (sum_angles : ℝ)
  (left_angle : ℝ)
  (right_angle : ℝ)
  (top_angle : ℝ)
  (h1 : sum_angles = 250)
  (h2 : left_angle = 2 * right_angle)
  (h3 : right_angle = 60)
  (h4 : sum_angles = left_angle + right_angle + top_angle) :
  top_angle = 70 :=
by
  sorry

end top_angle_is_70_l70_70982


namespace weight_difference_l70_70821

-- Define the relevant weights
def Karen_tote : ℝ := 8
def Kevin_empty_briefcase : ℝ := Karen_tote / 2
def Kevin_full_briefcase : ℝ := 2 * Karen_tote
def Kevin_contents : ℝ := Kevin_full_briefcase - Kevin_empty_briefcase
def Kevin_work_papers : ℝ := Kevin_contents / 6
def Kevin_laptop : ℝ := Kevin_contents - Kevin_work_papers

-- The main theorem statement
theorem weight_difference : Kevin_laptop - Karen_tote = 2 :=
by
  -- Proof would go here, but is omitted as per instructions
  sorry

end weight_difference_l70_70821


namespace max_cookies_without_ingredients_l70_70416

theorem max_cookies_without_ingredients :
  (∀ (total chocolate_chip nuts raisins sprinkles : ℕ),
    total = 60 →
    chocolate_chip = total / 3 →
    nuts = total / 2 →
    raisins = 2 * total / 3 →
    sprinkles = total / 4 →
    (total - max chocolate_chip (max nuts (max raisins sprinkles))) = 20) :=
begin
  intros total chocolate_chip nuts raisins sprinkles h_total h_chocolate_chip h_nuts h_raisins h_sprinkles,
  sorry
end

end max_cookies_without_ingredients_l70_70416


namespace find_x_l70_70349

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Definition of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- Theorem statement
theorem find_x (x : ℝ) (h_parallel : parallel a (b x)) : x = 6 :=
sorry

end find_x_l70_70349


namespace radius_of_circumscribed_circle_l70_70136

-- Defining the basic conditions:
variable (ABCD : Type) [ Quadrilateral ABCD ]
variable (R t : ℝ)                 -- The radius of the circle and the given distance t
variable (K O : Point)             -- Points K and O
variable [ Inscribed ABCD R ]      -- Quadrilateral ABCD is inscribed in a circle with radius R
variable [ PerpendicularDiagonals ABCD ] -- Diagonals of ABCD are perpendicular
variable [ DistanceBetween K O t ]  -- Distance from K (intersection of diagonals) to center O is t

-- The theorem to be proved:
theorem radius_of_circumscribed_circle :
  let r := \frac{1}{2} \sqrt{2 R^2 - t^2} in
  CircumscribedCircleRadius ABCD K O r :=
sorry

end radius_of_circumscribed_circle_l70_70136


namespace proof_problem_part1_proof_problem_part2_l70_70384

theorem proof_problem_part1 
  (A B a b c : ℝ)
  (h_eq_A_2B : A = 2 * B) 
  (h_law_of_sines : a / sin A = b / sin B) :
  a = 2 * b * cos B :=
sorry

theorem proof_problem_part2 
  (B : ℝ)
  (A : ℝ := 2 * B)
  (b c : ℝ)
  (h_b : b = 2)
  (h_c : c = 4)
  (h_law_of_cosines : a^2 = b^2 + c^2 - 2 * b * c * cos A)
  (h_cos_B : cos B = sqrt 3 / 2 ) :
  B = π/6 :=
sorry

end proof_problem_part1_proof_problem_part2_l70_70384


namespace triangle_CDM_area_correct_l70_70139

-- Define the basic triangle ABC
variables (A B C M D : Type) [triangle : has_triangle A B C]

-- Given conditions
variables (AC BC AD BD : ℝ) (AC_eq : AC = 8) (BC_eq : BC = 15) 
          (right_angle_C : is_right_angle C) 
          (M_midpoint : is_midpoint M A B) (AD_eq : AD = 17) (BD_eq : BD = 17)
          (CDM_area_form : ℚ) (h1 : ∃ m n p : ℕ, 
              m_gcd_p : nat.coprime m p ∧ n_not_div_square_prime : ∀ q : ℕ, prime q → q * q ∣ n → false ∧
              CDM_area_form = (m * sqrt n) / p)

-- The goal is to prove that under these conditions, m + n + p = 1056
theorem triangle_CDM_area_correct :
  ∃ m n p : ℕ, m_gcd_p : nat.coprime m p ∧ n_not_div_square_prime : (∀ q : ℕ, prime q → q * q ∣ n → false) ∧
  CDM_area_form = (m * sqrt n) / p ∧ m + n + p = 1056 :=
sorry

end triangle_CDM_area_correct_l70_70139


namespace num_four_digit_divisibles_l70_70354

theorem num_four_digit_divisibles : 
  ∃ n, n = 99 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 91 = 0 ↔ x ∈ (set.Ico 1001 9999) :=
by
  sorry

end num_four_digit_divisibles_l70_70354


namespace find_g_neg3_l70_70852

def g (x : ℝ) : ℝ :=
  if x < 0 then 3 * x - 4 else 2 * x + 7

theorem find_g_neg3 : g (-3) = -13 :=
by
  -- Placeholder for the proof
  sorry

end find_g_neg3_l70_70852


namespace bob_walked_12_miles_l70_70571

/-- Let distance XY be 24 miles and Yolanda's rate be 3 mph and Bob's rate be 4 mph. 
    Yolanda starts walking from X to Y and Bob starts one hour later from Y to X.
    The statement proves that Bob had walked 12 miles when they met. -/
theorem bob_walked_12_miles :
  ∀ (d : ℝ) (r₁ r₂ t₀ t₁ : ℝ), 
    d = 24 ∧ r₁ = 3 ∧ r₂ = 4 ∧ t₀ = 1 ∧ 
    d = r₁ * (t₀ + t₁) + r₂ * t₁ → 
    r₂ * t₁ = 12 :=
begin
  intros d r₁ r₂ t₀ t₁ h,
  rcases h with ⟨h_d, h_r₁, h_r₂, h_t₀, h_eq⟩,
  sorry
end

end bob_walked_12_miles_l70_70571


namespace number_of_sets_satisfying_union_l70_70425

open Set

theorem number_of_sets_satisfying_union (M : Set ℤ) (hM : M = { x | -1 ≤ x ∧ x < 2 }) :
  {P : Set ℤ | P ⊆ M ∧ P ∪ M = M}.Finite.toFinset.card = 8 := by
{
  sorry
}

end number_of_sets_satisfying_union_l70_70425


namespace complex_div_result_l70_70486

noncomputable def complex_expr : ℂ := (1+1*complex.i) * (2-1*complex.i) / complex.i

theorem complex_div_result :
  complex_expr = (1 - 3*complex.i) :=
sorry

end complex_div_result_l70_70486


namespace BQPeqBOP_l70_70055

-- Definition of the problem
variable {C : Circle} -- Circle C in the xy-plane
variable {A B : Point} -- Points A and B
variable {P : Point} -- Any point P on the circle
variable {Q : Point} -- Intersection of the line through P and A with the x-axis
variable {O : Point} -- The origin (0,0)

-- Definitions based on the conditions given
def conditions (C : Circle) (A B P Q O : Point) : Prop :=
  A = (0, a) ∧ B = (0, b) ∧ 0 < a ∧ a < b ∧
  C.center.y = y ∧ -- C's center is on the y-axis
  C.contains A ∧ C.contains B ∧ -- C passes through A and B
  C.contains P ∧ -- P is any point on C
  Q.y = 0 ∧ -- Q is on the x-axis
  liesOnLine {P, A} Q -- Q is on the line through P and A

-- Statement to be proven
theorem BQPeqBOP (C : Circle) (A B P Q O : Point) (h : conditions C A B P Q O) :
  angle B Q P = angle B O P := 
sorry

end BQPeqBOP_l70_70055


namespace total_volume_correct_l70_70165

-- Definitions based on the conditions
def box_length := 30 -- in cm
def box_width := 1 -- in cm
def box_height := 1 -- in cm
def horizontal_rows := 7
def vertical_rows := 5
def floors := 3

-- The volume of a single box
def box_volume : Int := box_length * box_width * box_height

-- The total number of boxes is the product of rows and floors
def total_boxes : Int := horizontal_rows * vertical_rows * floors

-- The total volume of all the boxes
def total_volume : Int := box_volume * total_boxes

-- The statement to prove
theorem total_volume_correct : total_volume = 3150 := 
by 
  simp [box_volume, total_boxes, total_volume]
  sorry

end total_volume_correct_l70_70165


namespace two_digit_sequence_partition_property_l70_70892

theorem two_digit_sequence_partition_property :
  ∀ (A B : Set ℕ), (A ∪ B = {x | x < 100 ∧ x % 10 < 10}) →
  ∃ (C : Set ℕ), (C = A ∨ C = B) ∧ 
  ∃ (lst : List ℕ), (∀ (x : ℕ), x ∈ lst → x ∈ C) ∧ 
  (∀ (x y : ℕ), (x, y) ∈ lst.zip lst.tail → (y = x + 1 ∨ y = x + 10 ∨ y = x + 11)) :=
by
  intros A B partition_condition
  sorry

end two_digit_sequence_partition_property_l70_70892


namespace max_min_cos_sin_l70_70696

open Real

theorem max_min_cos_sin :
  (-π / 2) < x → x < 0 → (sin x + cos x = 1 / 5)
  → (∃ y_max y_min : ℝ, y_max = 9 / 4 ∧ y_min = 2
    ∧ ∀ x_max ∈ {x | cos x = 1/2}, x_max ∈ {π / 3, -π / 3}
    ∧ ∀ x_min ∈ {x | cos x = 0} ∪ {x | cos x = 1}, x_min ∈ {π / 2, -π / 2, 0}) 
  := by sorry

end max_min_cos_sin_l70_70696


namespace broken_line_intersection_l70_70034

noncomputable theory

open_locale classical

theorem broken_line_intersection :
  ∀ {n : ℕ} (li ai bi : fin n → ℝ), 
    (∀ i, 0 ≤ li i ∧ 0 ≤ ai i ∧ 0 ≤ bi i) → 
    (∀ i, li i = a_dist (ai i, bi i)) → 
    (∑ i, li i = 1000) →
    ∃ k : ℕ, k ≥ 500 ∧
    (∃ s : fin set, s ⊆ fin.range n ∧ (∀ i ∈ s, intersects_parallel (li i) / s k)) sorry

end broken_line_intersection_l70_70034


namespace movie_theater_revenue_l70_70495

theorem movie_theater_revenue :
  let matinee_price := 5
  let evening_price := 12
  let 3D_price := 20
  let early_bird_discount := 0.5
  let group_discount := 0.9
  let surcharge := 2
  let total_matinee := 200
  let early_bird_matinee := 20
  let group_evening := 150
  let total_evening := 300
  let online_3D := 60
  let total_3D := 100
  let revenue_matinee := early_bird_matinee * (matinee_price * early_bird_discount) + (total_matinee - early_bird_matinee) * matinee_price
  let revenue_evening := group_evening * (evening_price * group_discount) + (total_evening - group_evening) * evening_price
  let revenue_3D := online_3D * (3D_price + surcharge) + (total_3D - online_3D) * 3D_price
  let total_revenue := revenue_matinee + revenue_evening + revenue_3D
  in total_revenue = 6490 :=
by
  let matinee_price := 5
  let evening_price := 12
  let 3D_price := 20
  let early_bird_discount := 0.5
  let group_discount := 0.9
  let surcharge := 2
  let total_matinee := 200
  let early_bird_matinee := 20
  let group_evening := 150
  let total_evening := 300
  let online_3D := 60
  let total_3D := 100
  let revenue_matinee := early_bird_matinee * (matinee_price * early_bird_discount) + (total_matinee - early_bird_matinee) * matinee_price
  let revenue_evening := group_evening * (evening_price * group_discount) + (total_evening - group_evening) * evening_price
  let revenue_3D := online_3D * (3D_price + surcharge) + (total_3D - online_3D) * 3D_price
  let total_revenue := revenue_matinee + revenue_evening + revenue_3D
  sorry

end movie_theater_revenue_l70_70495


namespace midpoint_sum_coordinates_l70_70164

def midpoint_3D (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ × ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)

def sum_of_coordinates (p : ℝ × ℝ × ℝ) : ℝ :=
  p.1 + p.2 + p.3

theorem midpoint_sum_coordinates :
  sum_of_coordinates (midpoint_3D 10 3 (-4) (-2) 7 6) = 10 :=
by
  sorry

end midpoint_sum_coordinates_l70_70164


namespace isosceles_triangle_perimeter_l70_70019

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

def is_valid_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem isosceles_triangle_perimeter {a b : ℕ} (h₁ : is_isosceles_triangle a b b) (h₂ : is_valid_triangle a b b) : a + b + b = 15 :=
  sorry

end isosceles_triangle_perimeter_l70_70019


namespace plank_length_l70_70600

theorem plank_length (a b : ℕ) 
  (h1 : (a - 8)^2 + (b + 4)^2 = a^2 + b^2)
  (h2 : (a - 17)^2 + (b + 7)^2 = a^2 + b^2) : 
  sqrt (a^2 + b^2) = 65 :=
sorry

end plank_length_l70_70600


namespace prove_smallest_number_l70_70984

noncomputable def smallest_number (x y z : ℕ) : ℕ :=
  if (x + y = 20) ∧ (x + z = 27) ∧ (y + z = 37) ∧ (x < y) ∧ (y < z) then x else 0

theorem prove_smallest_number (x y z : ℕ) :
  (x + y = 20) ∧ (x + z = 27) ∧ (y + z = 37) ∧ (x < y) ∧ (y < z) → x = 5 :=
by
  sorry

example : prove_smallest_number 5 15 22 := by
  simp [prove_smallest_number]
  sorry

end prove_smallest_number_l70_70984


namespace increasing_function_in_interval_l70_70228

noncomputable def f_A (x : ℝ) : ℝ := -x + 1
noncomputable def f_B (x : ℝ) : ℝ := 3^(1 - x)
noncomputable def f_C (x : ℝ) : ℝ := -(x - 1)^2
noncomputable def f_D (x : ℝ) : ℝ := 1 / (1 - x)

theorem increasing_function_in_interval :
  ∀ x ∈ Ioi 1, (f_A x < f_A (x + 1)) ∧
                (f_B x < f_B (x + 1)) ∧
                (f_C x < f_C (x + 1)) ∧
                (f_D x < f_D (x + 1)) :=
by
  sorry

end increasing_function_in_interval_l70_70228


namespace point_belongs_to_transformed_plane_l70_70577

def A := (1 : ℝ, 1 / 3 : ℝ, -2 : ℝ)
def original_plane (x y z : ℝ) : Prop := x - 3 * y + z + 6 = 0
def k := (1 / 3 : ℝ)
def transformed_plane (x y z : ℝ) : Prop := x - 3 * y + z + k * 6 = 0

theorem point_belongs_to_transformed_plane : transformed_plane 1 (1 / 3) (-2) = true := by
  sorry

end point_belongs_to_transformed_plane_l70_70577


namespace four_digit_multiples_of_13_and_7_l70_70356

theorem four_digit_multiples_of_13_and_7 : 
  (∃ n : ℕ, 
    (∀ k : ℕ, 1000 ≤ k ∧ k < 10000 ∧ k % 91 = 0 → k = 1001 + 91 * (n - 11)) 
    ∧ n - 11 + 1 = 99) :=
by
  sorry

end four_digit_multiples_of_13_and_7_l70_70356


namespace joana_favorite_song_probability_l70_70417

theorem joana_favorite_song_probability :
  let favorite_song : Nat := 240,
      n : Nat := 12,
      total_time : Nat := 5 * 60,
      song_lengths : Fin n → Nat := fun i => 40 + i.val * 20,
      total_ways : Nat := Nat.factorial n,
      total_ways_fav_complete : Nat := Nat.factorial 11 + 2 * Nat.factorial 10,
      prob_fav_complete : Rat := total_ways_fav_complete / total_ways,
      prob_fav_not_complete : Rat := 1 - prob_fav_complete
  in prob_fav_not_complete = 10 / 11 :=
by
  sorry

end joana_favorite_song_probability_l70_70417


namespace digits_difference_l70_70551

-- Defining base-10 integers 300 and 1500
def n1 := 300
def n2 := 1500

-- Defining a function to calculate the number of digits in binary representation
def binary_digits (n : ℕ) : ℕ := (nat.log2 n) + 1

-- Statement to be proven
theorem digits_difference : binary_digits n2 = binary_digits n1 + 2 := sorry

end digits_difference_l70_70551


namespace calculate_expression_l70_70630

theorem calculate_expression :
  2 * Real.sin (60 * Real.pi / 180) + abs (Real.sqrt 3 - 3) + (Real.pi - 1)^0 = 4 :=
by
  sorry

end calculate_expression_l70_70630


namespace nancy_math_problems_l70_70296

theorem nancy_math_problems
  (spelling_problems : ℝ)
  (problems_per_hour : ℝ)
  (hours : ℝ)
  (total_problems : ℝ)
  (h1 : spelling_problems = 15.0)
  (h2 : problems_per_hour = 8.0)
  (h3 : hours = 4.0)
  (h4 : total_problems = problems_per_hour * hours) :
  total_problems - spelling_problems = 17.0 :=
by 
  rw [h1, h2, h3] at h4
  exact h4.symm ▸ rfl

end nancy_math_problems_l70_70296


namespace min_ring_cuts_l70_70180

/-- Prove that the minimum number of cuts needed to pay the owner daily with an increasing 
    number of rings for 11 days, given a chain of 11 rings, is 2. -/
theorem min_ring_cuts {days : ℕ} {rings : ℕ} : days = 11 → rings = 11 → (∃ cuts : ℕ, cuts = 2) :=
by intros; sorry

end min_ring_cuts_l70_70180


namespace marching_band_formations_l70_70178

/-- A marching band of 240 musicians can be arranged in p different rectangular formations 
with s rows and t musicians per row where 8 ≤ t ≤ 30. 
This theorem asserts that there are 8 such different rectangular formations. -/
theorem marching_band_formations (s t : ℕ) (h : s * t = 240) (h_t_bounds : 8 ≤ t ∧ t ≤ 30) : 
  ∃ p : ℕ, p = 8 := 
sorry

end marching_band_formations_l70_70178


namespace binomial_constant_term_l70_70301

noncomputable def integral_value : ℝ :=
  ∫ x in (-(real.pi / 2)), (real.pi / 2), (6 * real.cos x - real.sin x)

theorem binomial_constant_term :
  let n := integral_value
  in n = 12 → 
     (∃ r : ℕ, r = 8 ∧ (binomial_coeff 12 r * (2^r) = 2^8 * binomial_coeff 12 8)) :=
begin
  intro n,
  intro n_eq,
  existsi 8,
  split,
  refl,
  sorry
end

end binomial_constant_term_l70_70301


namespace place_tokens_even_parity_l70_70078

def board := fin 50 → fin 50 → bool

variable (initial_placement : board)
variable (is_free : fin 50 → fin 50 → bool)
variable (can_place_new_tokens : ∀ (B : board), B(initial_placement)) 

theorem place_tokens_even_parity :
  (∃ (new_tokens : board), (∀ i j, new_tokens i j → is_free i j) ∧ 
   (∀ i, even (finset.univ.filter (λ j, new_tokens i j ∨ initial_placement i j).card)) ∧
   (∀ j, even (finset.univ.filter (λ i, new_tokens i j ∨ initial_placement i j).card)) ∧ 
   finset.univ.filter (λ ij, new_tokens ij.1 ij.2 = tt).card ≤ 99)
  :=
  sorry

end place_tokens_even_parity_l70_70078


namespace intersection_M_N_l70_70743

def M := {x : ℝ | x < 1}

def N := {y : ℝ | ∃ x : ℝ, y = Real.exp x}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
  sorry

end intersection_M_N_l70_70743


namespace digits_difference_l70_70549

-- Defining base-10 integers 300 and 1500
def n1 := 300
def n2 := 1500

-- Defining a function to calculate the number of digits in binary representation
def binary_digits (n : ℕ) : ℕ := (nat.log2 n) + 1

-- Statement to be proven
theorem digits_difference : binary_digits n2 = binary_digits n1 + 2 := sorry

end digits_difference_l70_70549


namespace base_k_constraint_l70_70327

theorem base_k_constraint (k : ℕ) : ¬ (k = 5) → ∃ (digits : List ℕ), digits = [3, 2, 5, 0, 1] ∧ ∀ d ∈ digits, d < k :=
by
  intro h
  exists [3, 2, 5, 0, 1]
  split
  { refl }
  { intros d hd
    cases hd with 
    | inl hinl => finish
    | inr hinr => finish
  }
  sorry

end base_k_constraint_l70_70327


namespace axis_of_symmetry_parabola_l70_70009

/-- If a parabola passes through points A(-2,0) and B(4,0), then the axis of symmetry of the parabola is the line x = 1. -/
theorem axis_of_symmetry_parabola (x : ℝ → ℝ) (hA : x (-2) = 0) (hB : x 4 = 0) : 
  ∃ c : ℝ, c = 1 ∧ ∀ y : ℝ, x y = x (2 * c - y) :=
sorry

end axis_of_symmetry_parabola_l70_70009


namespace sum_max_min_values_l70_70247

theorem sum_max_min_values (x y : ℝ) (k : ℝ) :
  (3 * x^2 + 2 * x * y + 4 * y^2 - 13 * x - 24 * y + 48 = 0) →
  let k1 := k, k2 := (-240/192 - k), 
  (-192 * k^2 + 240 * k - 407 = 0) →
  k1 + k2 = 5/4 :=
sorry

end sum_max_min_values_l70_70247


namespace rationalize_denominator_l70_70090

theorem rationalize_denominator :
  (∃ (a b c d e : ℝ), a = sqrt 2 ∧ 
                      b = sqrt 3 ∧ 
                      c = sqrt 5 ∧ 
                      d = 3 + sqrt 6 + sqrt 15 ∧ 
                      e = 6 ∧ 
                      (a / (a + b - c)) = (d / e)) :=
sorry

end rationalize_denominator_l70_70090


namespace perfect_square_expression_l70_70523

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 :=
by
  sorry

end perfect_square_expression_l70_70523


namespace rational_pairs_satisfy_sqrt_eq_l70_70273

theorem rational_pairs_satisfy_sqrt_eq (a b : ℚ) :
  (√a + √b = √(2 + √3)) ↔ ((a = 0.5 ∧ b = 1.5) ∨ (a = 1.5 ∧ b = 0.5)) :=
by
  sorry

end rational_pairs_satisfy_sqrt_eq_l70_70273


namespace f_has_no_extreme_values_l70_70717

theorem f_has_no_extreme_values (x : ℝ) : ¬∃ y : ℝ, is_extreme_value (λ x, x + sin x) y :=
by sorry

end f_has_no_extreme_values_l70_70717


namespace find_expression_l70_70722

noncomputable def sequence_a : ℕ → ℕ 
| 1 := 0
| 2 := 1
| 3 := 9
| n := if n > 3 then 10 * sequence_a (n - 1) + 9 * sequence_a (n - 2) else 0

def sum_S (n : ℕ) : ℕ :=
(nat.rec_on n 0 (λ k ih, ih + sequence_a k))

theorem find_expression :
  ∀ {n : ℕ}, n ≥ 3 → sequence_a n = 9 * 10^(n-3) := 
sorry

end find_expression_l70_70722


namespace exists_three_quadratic_polynomials_l70_70675

theorem exists_three_quadratic_polynomials :
  ∃ (f₁ f₂ f₃ : ℝ → ℝ),
    (∃ x₁, f₁ x₁ = 0) ∧
    (∃ x₂, f₂ x₂ = 0) ∧
    (∃ x₃, f₃ x₃ = 0) ∧
    (∀ x, f₁(x) + f₂(x) ≠ 0) ∧
    (∀ x, f₁(x) + f₃(x) ≠ 0) ∧
    (∀ x, f₂(x) + f₃(x) ≠ 0) ∧
    (∀ x, ∃ (a b c : ℝ), f₁(x) = a*x^2 + b*x + c) ∧
    (∀ x, ∃ (a b c : ℝ), f₂(x) = a*x^2 + b*x + c) ∧
    (∀ x, ∃ (a b c : ℝ), f₃(x) = a*x^2 + b*x + c) := 
sorry

end exists_three_quadratic_polynomials_l70_70675


namespace polynomial_has_root_l70_70958

theorem polynomial_has_root {a b c d : ℝ} 
  (h : a * c = 2 * b + 2 * d) : 
  ∃ x : ℝ, (x^2 + a * x + b = 0) ∨ (x^2 + c * x + d = 0) :=
by 
  sorry

end polynomial_has_root_l70_70958


namespace sum_of_first_n_terms_geometric_seq_l70_70306

variable {a : ℕ → ℝ}
variable (n : ℕ)
variable {q : ℝ}

-- Definitions and conditions
def is_positive_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def condition1 (a : ℕ → ℝ) := a 1 + a 5 = 34
def condition2 (a : ℕ → ℝ) := a 2 * a 4 = 64

-- Proving the sum of first n terms
theorem sum_of_first_n_terms_geometric_seq 
  (a : ℕ → ℝ) (q : ℝ) (h_geo : is_positive_geometric_sequence a q)
  (h1 : condition1 a) (h2 : condition2 a) (hq : q > 0) :
  (Σ i in Finset.range n, a i) = if q = 2 then 2^(n + 1) - 2 else 64 * (1 - 1 / q^n) :=
sorry

end sum_of_first_n_terms_geometric_seq_l70_70306


namespace angle_ABI_eq_IMB_l70_70809

-- Define the structure and properties of the problem
structure TriangleInscribedInCircle :=
  (A B C O I M : Point) -- Defining points
  (ThatCircumcircle: Circle O) -- Circle \odot O
  (ABC_inscribed : A, B, C ⊂ Circumcircle)
  (incenter : Incenter Triangle ABC = I)
  (arcMidpoint : M = midpoint ArcOverABC, C_not_containing) -- midpoint of arc not containing C
  (AC_eq_2AB : Distance A C = 2 * Distance A B)

-- Define the goal statement
theorem angle_ABI_eq_IMB (t : TriangleInscribedInCircle) : 
  ∠(t.A, t.B, t.I) = ∠(t.I, t.M, t.B) :=
sorry

end angle_ABI_eq_IMB_l70_70809


namespace no_primes_in_list_l70_70638

-- Define the initial number
def initial_number : ℕ := 407

-- Define the sequence of numbers by repeating the initial number in the list
def repeated_sequence (n : ℕ) : ℕ :=
  let rep := (10^(3*n) - 1) / 999
  in (initial_number * rep)

-- The main theorem stating there are no prime numbers in the list
theorem no_primes_in_list : ∀ (n : ℕ), ¬ Prime (repeated_sequence (n + 1)) := 
by
  intro n
  -- Proof steps will go here
  sorry

end no_primes_in_list_l70_70638


namespace cross_country_meet_winning_scores_l70_70397

/-- In a triangular cross-country meet with 3 teams, 
    where each team has 3 runners finishing in unique positions 1 through 9,
    the potential winning scores for the team with the lowest score is the set {6, 7, 8, ..., 21}. -/
theorem cross_country_meet_winning_scores :
  ∃ (winning_scores : set ℕ), winning_scores = set.Icc 6 21 ∧
  ∀ (positions : finset ℕ) (team_scores : fin 3 → ℕ), 
  (positions = finset.range 1 \u finset.range 10) →
  ∑ i, team_scores i ∈ positions → 
  min team_scores = ∈ winning_scores := sorry

end cross_country_meet_winning_scores_l70_70397


namespace initial_bananas_per_child_l70_70880

theorem initial_bananas_per_child 
    (absent : ℕ) (present : ℕ) (total : ℕ) (x : ℕ) (B : ℕ)
    (h1 : absent = 305)
    (h2 : present = 305)
    (h3 : total = 610)
    (h4 : B = present * (x + 2))
    (h5 : B = total * x) : 
    x = 2 :=
by
  sorry

end initial_bananas_per_child_l70_70880


namespace determine_a_for_parallel_lines_l70_70753

theorem determine_a_for_parallel_lines (a : ℝ) :
  let l1 : (ℝ → ℝ → ℝ) := λ x y, (a - 1) * x + y + 1
  let l2 : (ℝ → ℝ → ℝ) := λ x y, 2 * a * x + y + 3
  (∀ x y, l1 x y = 0 → ∀ x' y', l2 x' y' = 0 →
                                 (a - 1) = -2 * a) →
                                 a = -1 :=
by
  intros l1 l2 h
  sorry

end determine_a_for_parallel_lines_l70_70753


namespace ratio_division_rhombus_l70_70796

theorem ratio_division_rhombus (α : ℝ) {A B C D E : Type} 
  [angle : α = ∠BAD] 
  [division1 : ∠EAD = α / 3] 
  [division2 : ∠BAE = 2 * α / 3] 
  [CE DE : ℝ] :
  (CE / DE) = (Real.cos (α / 2)) / (Real.cos (α / 6)) :=
sorry

end ratio_division_rhombus_l70_70796


namespace pirate_chest_value_l70_70092

theorem pirate_chest_value :
  (∀ (pirates chests : ℕ) (common_fund paid compensation total_value chest_value : ℕ),
    pirates = 7 →
    chests = 5 →
    common_fund = 50000 →
    paid = 10000 →
    compensation = common_fund / 2 →
    total_value = pirates * compensation →
    chest_value = total_value / chests →
    chest_value = 35000) :=
by
  intros pirates chests common_fund paid compensation total_value chest_value
  assume hp : pirates = 7
  assume hc : chests = 5
  assume hf : common_fund = 50000
  assume hpaid : paid = 10000
  assume hcomp : compensation = common_fund / 2
  assume hval : total_value = pirates * compensation
  assume hchest : chest_value = total_value / chests
  sorry

end pirate_chest_value_l70_70092


namespace geometric_sequence_4th_term_l70_70932

namespace GeometricSequenceProof

theorem geometric_sequence_4th_term (a a6 : ℝ) (h1 : a = 512) (h2 : a * (1 / 2)^5 = 8) : 
  let r := (1 / 2) in 
  a * r^3 = 64 := 
by
  have h3 : r = 1 / 2 := by
    sorry
  rw [h1, h3]
  sorry

end geometric_sequence_4th_term_l70_70932


namespace erika_rick_savings_l70_70679

def gift_cost : ℕ := 250
def erika_savings : ℕ := 155
def rick_savings : ℕ := gift_cost / 2
def cake_cost : ℕ := 25

theorem erika_rick_savings :
  let total_savings := erika_savings + rick_savings in
  let total_cost := gift_cost + cake_cost in
  total_savings - total_cost = 5 :=
by
  let total_savings := erika_savings + rick_savings
  let total_cost := gift_cost + cake_cost
  sorry

end erika_rick_savings_l70_70679


namespace average_of_class_is_49_5_l70_70022

noncomputable def average_score_of_class : ℝ :=
  let total_students := 50
  let students_95 := 5
  let students_0 := 5
  let students_85 := 5
  let remaining_students := total_students - (students_95 + students_0 + students_85)
  let total_marks := (students_95 * 95) + (students_0 * 0) + (students_85 * 85) + (remaining_students * 45)
  total_marks / total_students

theorem average_of_class_is_49_5 : average_score_of_class = 49.5 := 
by sorry

end average_of_class_is_49_5_l70_70022


namespace num_even_sum_subsets_l70_70763

-- The set of numbers
def nums : Finset ℕ := {45, 52, 68, 73, 81, 94, 105, 110}

-- Number of subsets containing four different numbers such that their sum is even
theorem num_even_sum_subsets : (nums.filter (λ s, s.sum % 2 = 0) powerset.filter (λ s, s.card = 4)).card = 37 := sorry

end num_even_sum_subsets_l70_70763


namespace sin_a4_a7_l70_70803

variable (a : ℕ → ℝ) -- a is our arithmetic sequence

-- Conditions: the arithmetic sequence property and given value
axiom arithmetic_sequence (d : ℝ) : ∀ n : ℕ, a (n + 1) = a n + d
axiom known_sum : a 5 + a 6 = (10 * Real.pi) / 3

-- The proof problem statement
theorem sin_a4_a7 : sin (a 4 + a 7) = - (√3) / 2 := 
  by {
    sorry
  }

end sin_a4_a7_l70_70803


namespace solve_quadratic_eq_l70_70716

theorem solve_quadratic_eq (x : ℝ) : 45 - 3 * x ^ 2 = 0 ↔ x = sqrt 15 ∨ x = -sqrt 15 :=
by
  sorry

end solve_quadratic_eq_l70_70716


namespace convert_polar_to_rectangular_l70_70251

/-- The given point in polar coordinates -/
def polar_point : ℝ × ℝ := (4, Real.pi / 4)

/-- Convert the polar point to rectangular coordinates -/
def rectangular_coordinates (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular :
  rectangular_coordinates 4 (Real.pi / 4) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) :=
by
  sorry

end convert_polar_to_rectangular_l70_70251


namespace photos_difference_is_120_l70_70444

theorem photos_difference_is_120 (initial_photos : ℕ) (final_photos : ℕ) (first_day_factor : ℕ) (first_day_photos : ℕ) (second_day_photos : ℕ) : 
  initial_photos = 400 → 
  final_photos = 920 → 
  first_day_factor = 2 →
  first_day_photos = initial_photos / first_day_factor →
  final_photos = initial_photos + first_day_photos + second_day_photos →
  second_day_photos - first_day_photos = 120 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end photos_difference_is_120_l70_70444


namespace tetrahedron_surface_area_inequality_l70_70434

theorem tetrahedron_surface_area_inequality 
  (a b c d e f : ℝ) 
  (S : ℝ) 
  (h_tet : ∀ {A B C D : ℝ}, tetrahedron A B C D a b c d e f = true) :
  S ≤ (Real.sqrt 3 / 6) * (a^2 + b^2 + c^2 + d^2 + e^2 + f^2) :=
sorry

end tetrahedron_surface_area_inequality_l70_70434


namespace find_wall_area_l70_70858

theorem find_wall_area
  (gallons : ℕ)           -- Number of gallons of paint Linda uses
  (coverage_per_gallon : ℕ) -- Area covered by one gallon of paint in square feet
  (coats : ℕ)             -- Number of coats of paint intended
  (total_coverage : ℕ := gallons * coverage_per_gallon) -- Total coverage provided by the gallons
  (wall_area : ℕ := total_coverage / coats)             -- Wall area calculation
  (h_gallons : gallons = 3)                             -- Condition: Linda uses 3 gallons
  (h_coverage : coverage_per_gallon = 400)              -- Condition: Each gallon covers 400 sq. ft.
  (h_coats : coats = 2)                                 -- Condition: She wants to do two coats
  : wall_area = 600 :=                                  -- Proof goal: Wall area should be 600 sq. ft.
begin
  sorry
end

end find_wall_area_l70_70858


namespace percentage_absent_l70_70231

noncomputable def total_students : ℕ := 240
noncomputable def boys : ℕ := 150
noncomputable def girls : ℕ := 90
noncomputable def fraction_boys_absent : ℚ := 1 / 5
noncomputable def fraction_girls_absent : ℚ := 1 / 2

theorem percentage_absent : 
  let absent_boys := fraction_boys_absent * boys,
      absent_girls := fraction_girls_absent * girls,
      total_absent := absent_boys + absent_girls,
      absent_fraction := total_absent / total_students,
      absent_percentage := absent_fraction * 100 in
  absent_percentage = 31.25 :=
by
  sorry

end percentage_absent_l70_70231


namespace solve_for_x_l70_70706

theorem solve_for_x (x : ℝ) :
  (x + 3)^3 = -64 → x = -7 :=
by
  intro h
  sorry

end solve_for_x_l70_70706


namespace quadratic_inequality_solution_l70_70372

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
by
  sorry

end quadratic_inequality_solution_l70_70372


namespace length_of_arc_l70_70788

variable {O A B : Type}
variable (angle_OAB : Real) (radius_OA : Real)

theorem length_of_arc (h1 : angle_OAB = 45) (h2 : radius_OA = 5) :
  (length_of_arc_AB = 5 * π / 4) :=
sorry

end length_of_arc_l70_70788


namespace prob_min_distance_l70_70294

theorem prob_min_distance (x y z : ℝ) :
  sqrt (x^2 + y^2 + z^2) + sqrt ((x + 1)^2 + (y - 2)^2 + (z - 1)^2) ≥ sqrt 6 :=
sorry

end prob_min_distance_l70_70294


namespace domain_h_l70_70482

noncomputable def domain_f : set ℝ := {x | x ∈ Icc (-3) 9}

noncomputable def h (x : ℝ) (f : ℝ → ℝ) : ℝ := f (-3 * x + 1)

theorem domain_h (f : ℝ → ℝ) (hf : ∀ x, x ∈ domain_f → f x ≠ 0) :
  ∀ x, x ∈ Icc (-8/3) (4/3) ↔ x ∈ preimage (λ x, -3 * x + 1) domain_f := by
  sorry

end domain_h_l70_70482


namespace g_eq_g_inv_iff_l70_70646

def g (x : ℝ) : ℝ := 3 * x - 7

def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_iff (x : ℝ) : g x = g_inv x ↔ x = 7 / 2 :=
by {
  sorry
}

end g_eq_g_inv_iff_l70_70646


namespace steiner_point_on_circumcircle_simson_line_parallel_brocard_l70_70567

-- Definitions of the required points and lines
structure Triangle (α : Type) :=
(A B C : α)

structure ParallelLine (α : Type) :=
(A1 B1 C1 : α)

-- Definition of the Steiner point S
def SteinerPoint (α : Type) (tri : Triangle α) (para : ParallelLine α) : α :=
  sorry

-- Part (a): Prove the existence of the Steiner point on the circumcircle
theorem steiner_point_on_circumcircle (α : Type) [EuclideanGeometry α] 
  (tri : Triangle α) (para : ParallelLine α) : 
  let S := SteinerPoint α tri para in
    OnCircumcircle tri S :=
    by sorry

-- Definition of the Simson line of a point
def SimsonLine (α : Type) (pt : α) (tri : Triangle α) : α :=
  sorry

-- Definition of the Brocard diameter
def BrocardDiameter (α : Type) (tri : Triangle α) : α :=
  sorry

-- Part (b): Prove the Simson line of the Steiner point is parallel to the Brocard diameter
theorem simson_line_parallel_brocard (α : Type) [EuclideanGeometry α] 
  (tri : Triangle α) (para : ParallelLine α) :
  let S := SteinerPoint α tri para in
    Parallel (SimsonLine α S tri) (BrocardDiameter α tri) :=
    by sorry

end steiner_point_on_circumcircle_simson_line_parallel_brocard_l70_70567


namespace value_a_plus_c_l70_70432

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c
noncomputable def g (a b c : ℝ) (x : ℝ) := c * x^2 + b * x + a

theorem value_a_plus_c (a b c : ℝ) (h : ∀ x : ℝ, f a b c (g a b c x) = x) : a + c = -1 :=
by
  sorry

end value_a_plus_c_l70_70432


namespace inequality_abc_l70_70904

theorem inequality_abc (a b c : ℝ) (h : a * b * c = 1) :
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end inequality_abc_l70_70904


namespace vasya_can_place_99_tokens_l70_70084

theorem vasya_can_place_99_tokens (board : ℕ → ℕ → Prop) (h : ∀ i j, board i j → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50) :
  ∃ new_tokens : ℕ → ℕ → Prop, (∀ i j, new_tokens i j → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50) ∧
  (∀ i, (∑ j in finset.range 50, if board i j ∨ new_tokens i j then 1 else 0) % 2 = 0) ∧
  (∀ j, (∑ i in finset.range 50, if board i j ∨ new_tokens i j then 1 else 0) % 2 = 0) ∧
  (finset.sum (finset.product (finset.range 50) (finset.range 50)) 
             (λ (ij : ℕ × ℕ), if new_tokens ij.1 ij.2 then 1 else 0) ≤ 99) :=
sorry

end vasya_can_place_99_tokens_l70_70084


namespace min_pos_int_m_l70_70295

open Polynomial
open Nat

theorem min_pos_int_m (k : ℕ) (hk : k > 1) :
  ∃ m > 1, ∃ f : Polynomial ℤ, (∃ r : ℤ, f.eval r = 1) ∧ (∃ s : Finset ℤ, s.card = k ∧ ∀ x ∈ s, f.eval x = m) ∧
    m = (factorial (k / 2)) * (factorial (k - k / 2)) + 1 :=
sorry

end min_pos_int_m_l70_70295


namespace erika_rick_money_left_l70_70682

theorem erika_rick_money_left (gift_cost cake_cost erika_savings : ℝ)
  (rick_savings := gift_cost / 2) (total_savings := erika_savings + rick_savings) 
  (total_cost := gift_cost + cake_cost) : (total_savings - total_cost) = 5 :=
by
  -- Given conditions from the problem
  have h_gift_cost : gift_cost = 250 := sorry
  have h_cake_cost : cake_cost = 25 := sorry
  have h_erika_savings : erika_savings = 155 := sorry
  -- Show that the remaining money is $5
  have h_rick_savings : rick_savings = 125 := by
    rw [←h_gift_cost]
    norm_num
  have h_total_savings : total_savings = 280 := by
    rw [←h_erika_savings, ←h_rick_savings]
    norm_num
  have h_total_cost : total_cost = 275 := by
    rw [←h_gift_cost, ←h_cake_cost]
    norm_num
  rw [←h_total_savings, ←h_total_cost]
  norm_num
  done

end erika_rick_money_left_l70_70682


namespace volume_of_regular_triangular_prism_l70_70795

theorem volume_of_regular_triangular_prism (S : ℝ) : 
    let angle := 45 * (π / 180)
    let base_area := S / (sqrt 2)
    let height := (sqrt (3 * S)) / (sqrt 6)
    V = base_area * height
  ∃ V, V = (sqrt (6 * S^3)) / 2 :=
sorry

end volume_of_regular_triangular_prism_l70_70795


namespace single_fraction_l70_70560

theorem single_fraction (c : ℕ) : (6 + 5 * c) / 5 + 3 = (21 + 5 * c) / 5 :=
by sorry

end single_fraction_l70_70560


namespace tangent_identity_l70_70003

theorem tangent_identity (α β : Real) :
  sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β →
  tan (α - β) = -1 :=
by
  intro h
  sorry

end tangent_identity_l70_70003


namespace fifth_segment_exists_l70_70220

structure Grid5x5 where
  -- Represents a 5x5 grid, additional properties can be added as needed
  marked_points : Set (ℝ × ℝ)
  segments : Set ((ℝ × ℝ) × (ℝ × ℝ))
  no_intersection : ∀ (s1 s2 : (ℝ × ℝ) × (ℝ × ℝ)), s1 ∈ segments → s2 ∈ segments → s1 ≠ s2 → ¬segments_intersect s1 s2

-- Define a function to determine if two segments intersect
def segments_intersect (s1 s2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := 
  sorry -- Implement this function or import from a library

-- The problem statement in Lean 4
theorem fifth_segment_exists (grid : Grid5x5) :
  ∃ (new_segment : (ℝ × ℝ) × (ℝ × ℝ)), 
    new_segment.1 ∈ grid.marked_points ∧ 
    new_segment.2 ∈ grid.marked_points ∧ 
    ¬∃ s ∈ grid.segments, segments_intersect new_segment s :=
sorry

end fifth_segment_exists_l70_70220


namespace at_least_one_root_l70_70950

theorem at_least_one_root 
  (a b c d : ℝ)
  (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
sorry

end at_least_one_root_l70_70950


namespace bx_squared_l70_70047

theorem bx_squared (A B C M N X : Type) 
  (h1 : is_triangle A B C)
  (h2 : midpoint M A C)
  (h3 : ⦃perpendicular_bisector BN N AC AB⦄) 
  (h4 : intersection X median B M bisector B N)
  (h5 : is_equilateral_triangle B X N)
  (h6 : AC_length AC = 2) 
  (h7 : bisects_ratio BN A N C 2 1) : 
  length_squared BX = 4 / 3 :=
sorry

end bx_squared_l70_70047


namespace triangle_angle_proof_l70_70888

open EuclideanGeometry

theorem triangle_angle_proof
  (A B C D L : Point)
  (hD : D ∈ segment A B)
  (hL_interior : interior (triangle A B C) L)
  (hBD_LD : B.distance D = L.distance D)
  (hEqualAngles : ∠ L A B = ∠ L C A ∧ ∠ L C A = ∠ D C B)
  (hSumAngles : ∠ A L D + ∠ A B C = 180) :
  ∠ B L C = 90 := by
  sorry

end triangle_angle_proof_l70_70888


namespace inequality_proof_l70_70007

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
    (x^4 / (y * (1 - y^2))) + (y^4 / (z * (1 - z^2))) + (z^4 / (x * (1 - x^2))) ≥ 1 / 8 :=
sorry

end inequality_proof_l70_70007


namespace range_of_a_l70_70745

theorem range_of_a (a : ℝ)
  (h_circle : ∃ x y : ℝ, x^2 + y^2 - 2 * a * x + 2 * a * y + 2 * a^2 + 2 * a - 1 = 0)
  (h_line : ∃ x y : ℝ, x - y - 1 = 0)
  (h_common : ∃ x y : ℝ, (x^2 + y^2 - 2 * a * x + 2 * a * y + 2 * a^2 + 2 * a - 1 = 0) ∧ (x - y - 1 = 0))
: a ∈ set.Icc (-1/2 : ℝ) (1/2 : ℝ) := sorry

end range_of_a_l70_70745


namespace point_outside_circle_l70_70781

-- Define the line equation
def line (a b : ℝ) : ℝ × ℝ → Prop := fun p => a * p.1 + b * p.2 = 1

-- Define the circle equation
def circle : ℝ × ℝ → Prop := fun p => p.1^2 + p.2^2 = 1

-- Define intersection condition: the line intersects the circle at two distinct points
def intersects (a b : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), circle (x1, y1) ∧ circle (x2, y2) ∧ line a b (x1, y1) ∧ line a b (x2, y2) ∧ (x1, y1) ≠ (x2, y2)

-- Prove that point (a,b) lies outside the circle given the intersection condition
theorem point_outside_circle (a b : ℝ) (h : intersects a b) : a^2 + b^2 > 1 := by
  sorry

end point_outside_circle_l70_70781


namespace perimeter_new_rectangle_l70_70075

theorem perimeter_new_rectangle : 
  ∀ (x : ℝ), 
    3x^2 = (3x - 18) * (x + 8) → 
    2 * ((3 * x - 18) + (x + 8)) = 172 := 
begin
  sorry
end

end perimeter_new_rectangle_l70_70075


namespace largest_angle_convex_hexagon_l70_70500

theorem largest_angle_convex_hexagon : 
  ∃ x : ℝ, (x-3) + (x-2) + (x-1) + x + (x+1) + (x+2) = 720 → (x + 2) = 122.5 :=
by 
  intros,
  sorry

end largest_angle_convex_hexagon_l70_70500


namespace total_pokemon_cards_l70_70900

theorem total_pokemon_cards : 
  let n := 14 in
  let total_cards := 4 * n in
  total_cards = 56 :=
by
  let n := 14
  let total_cards := 4 * n
  show total_cards = 56
  sorry

end total_pokemon_cards_l70_70900


namespace average_speed_is_60_mph_l70_70207

def meters_to_miles (m : ℝ) : ℝ := m / 1609.34

def time_northward (m : ℝ) : ℝ := 3 * meters_to_miles m

def time_southward (m : ℝ) : ℝ := meters_to_miles m / 3

def total_time_in_hours (m : ℝ) : ℝ := ((3 * meters_to_miles m) + (meters_to_miles m / 3)) / 60

def total_distance_in_miles (m : ℝ) : ℝ := 2 * meters_to_miles m

def average_speed (m : ℝ) : ℝ := total_distance_in_miles m / total_time_in_hours m

theorem average_speed_is_60_mph (m : ℝ) : average_speed m = 60 :=
sorry

end average_speed_is_60_mph_l70_70207


namespace inradii_and_exradii_relation_l70_70086

variables {A B C M : Type*} 
variables (r1 r2 r ρ1 ρ2 ρ : ℝ)
variables [InradiusTriangleAMC : (Triangle M A C)]
variables [InradiusTriangleBMC : (Triangle M B C)]
variables [InradiusTriangleABC : (Triangle A B C)]
variables [ExradiusAngleACB_TriangleAMC : ExradiusTriangleAMC A C B]
variables [ExradiusAngleACB_TriangleBMC : ExradiusTriangleBMC A C B]
variables [ExradiusAngleACB_TriangleABC : ExradiusTriangleABC A C B]

theorem inradii_and_exradii_relation 
  (h1: InradiusTriangleAMC r1) 
  (h2: InradiusTriangleBMC r2)
  (h3: InradiusTriangleABC r)
  (h4: ExradiusAngleACB_TriangleAMC ρ1) 
  (h5: ExradiusAngleACB_TriangleBMC ρ2)
  (h6: ExradiusAngleACB_TriangleABC ρ)
  : (r1 / ρ1) * (r2 / ρ2) = r / ρ := 
sorry -- the proof goes here

end inradii_and_exradii_relation_l70_70086


namespace average_speed_l70_70587

theorem average_speed (d1 d2 d3 d4 d5 t: ℕ) 
  (h1: d1 = 120) 
  (h2: d2 = 70) 
  (h3: d3 = 90) 
  (h4: d4 = 110) 
  (h5: d5 = 80) 
  (total_time: t = 5): 
  (d1 + d2 + d3 + d4 + d5) / t = 94 := 
by 
  -- proof will go here
  sorry

end average_speed_l70_70587


namespace systematic_sampling_selection_l70_70787

theorem systematic_sampling_selection:
  ∀ (students : Finset ℕ) (selected : Finset ℕ) (n : ℕ),
  students = Finset.range (900 + 1) →
  ¬ (0 ∈ students) →
  students.card = 900 →
  selected.card = 150 →
  n = 015 →
  (6 ∣ (081 - n)) :=
begin
  intros students selected n h_students h_nonzero h_students_card h_selected_card h_n,
  rw h_students at *,
  rw h_n,
  -- Proof skipped
  sorry,
end

end systematic_sampling_selection_l70_70787


namespace square_area_from_diagonal_l70_70574

-- Define the parameters and conditions
def diagonal_length (d : ℝ) := d = 16
def area_of_square (A : ℝ) := A = 128

-- Translate the proof problem into a Lean 4 statement
theorem square_area_from_diagonal (d : ℝ) (A : ℝ) (h₁: diagonal_length d) : area_of_square A :=
begin
  rw diagonal_length at h₁,
  rw area_of_square,
  sorry -- Proof to show A = 128 based on d = 16 using Pythagorean theorem.
end

end square_area_from_diagonal_l70_70574


namespace coprime_divisibility_by_240_l70_70461

theorem coprime_divisibility_by_240 (n : ℤ) (h : Int.gcd n 30 = 1) : 240 ∣ (-n^4 - 1) := sorry

end coprime_divisibility_by_240_l70_70461


namespace corridor_painting_l70_70156

theorem corridor_painting (corridor_length : ℝ) 
                          (paint_range_p1_start paint_range_p1_length paint_range_p2_end paint_range_p2_length : ℝ) :
  corridor_length = 15 → 
  paint_range_p1_start = 2 → 
  paint_range_p1_length = 9 → 
  paint_range_p2_end = 14 → 
  paint_range_p2_length = 10 → 
  ((paint_range_p1_start + paint_range_p1_length ≤ paint_range_p2_end) 
  ∧ (paint_range_p2_end - paint_range_p2_length ≥ paint_range_p1_start)) 
  → 5 = (paint_range_p2_end - (paint_range_p1_start + paint_range_p1_length - paint_range_p1_start )) :=
by
  intros h_corridor_length h_paint_range_p1_start h_paint_range_p1_length h_paint_range_p2_end h_paint_range_p2_length h_disjoint
  sorry

end corridor_painting_l70_70156


namespace LisaNeedsMoreMarbles_l70_70865

theorem LisaNeedsMoreMarbles :
  let friends := 12
  let marbles := 40
  let required_marbles := (friends * (friends + 1)) / 2
  let additional_marbles := required_marbles - marbles
  additional_marbles = 38 :=
by
  let friends := 12
  let marbles := 40
  let required_marbles := (friends * (friends + 1)) / 2
  let additional_marbles := required_marbles - marbles
  have h1 : required_marbles = 78 := by
    calc (friends * (friends + 1)) / 2
      _ = (12 * 13) / 2 : by rfl
      _ = 156 / 2 : by rfl
      _ = 78 : by norm_num
  have h2 : additional_marbles = 38 := by
    calc required_marbles - marbles
      _ = 78 - 40 : by rw h1
      _ = 38 : by norm_num
  exact h2

end LisaNeedsMoreMarbles_l70_70865


namespace number_proper_subsets_l70_70119

open Set

theorem number_proper_subsets {α : Type*} (a b : α) : 
  (univ : Finset (Set α)).card = 3 →  -- Assumption to bound the universe of sets
  (filter (λ t, t ≠ {a, b}) (powerset ({a, b} : Finset α))).card = 3 := 
begin
  sorry
end

end number_proper_subsets_l70_70119


namespace largest_x_value_l70_70839

theorem largest_x_value (x y z : ℝ) (h1 : x + y + z = 6) (h2 : x * y + x * z + y * z = 9) : x ≤ 4 := 
sorry

end largest_x_value_l70_70839


namespace find_number_l70_70545

theorem find_number (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by
  sorry

end find_number_l70_70545
