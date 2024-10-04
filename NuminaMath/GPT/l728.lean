import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.BigOperators.Finprod
import Mathlib.Algebra.Combination
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Absolute
import Mathlib.Algebra.Polynomial
import Mathlib.Data.Angle
import Mathlib.Data.Fin
import Mathlib.Data.Fin.Vec
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Vec3
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Cyclic
import Mathlib.Init.Data.Int.Basic
import Mathlib.Logic.Basic
import Mathlib.Order.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic

namespace optionA_is_false_optionB_is_true_optionC_is_true_optionD_is_false_l728_728880

-- Boolean validation for option A (it is incorrect)
def optionA (a b c : ℝ³) (ha_ne_zero : a ≠ 0) (h : a ⋅ b = a ⋅ c) : Prop :=
  b = c -- This should be proven false

-- Boolean validation for option B (it is correct)
def optionB (z1 z2 : ℂ) : Prop :=
  abs (z1 * z2) = abs z1 * abs z2

-- Boolean validation for option C (it is correct)
def optionC (a b : ℝ³) (ha_ne_zero : a ≠ 0) (hb_ne_zero : b ≠ 0) (h : abs (a + b) = abs (a - b)) : Prop :=
  a ⋅ b = 0

-- Boolean validation for option D (it is incorrect)
def optionD (z1 z2 : ℂ) (h : abs (z1 + z2) = abs (z1 - z2)) : Prop :=
  z1 * z2 = 0 -- This should be proven false

theorem optionA_is_false (a b c : ℝ³) (ha_ne_zero : a ≠ 0) (h : a ⋅ b = a ⋅ c) : ¬ optionA a b c ha_ne_zero h :=
by sorry

theorem optionB_is_true (z1 z2 : ℂ) : optionB z1 z2 :=
by sorry

theorem optionC_is_true (a b : ℝ³) (ha_ne_zero : a ≠ 0) (hb_ne_zero : b ≠ 0) (h : abs (a + b) = abs (a - b)) : optionC a b ha_ne_zero hb_ne_zero h :=
by sorry

theorem optionD_is_false (z1 z2 : ℂ) (h : abs (z1 + z2) = abs (z1 - z2)) : ¬ optionD z1 z2 h :=
by sorry

end optionA_is_false_optionB_is_true_optionC_is_true_optionD_is_false_l728_728880


namespace propositions_evaluation_l728_728506

theorem propositions_evaluation :
  (¬ (∃ x₀ > 0, x₀ < sin x₀)) ∧
  (∀ α : ℝ, (sin α ≠ 1/2 → α ≠ π / 6)) ∧
  (¬ (∀ a b : ℝ, ln a > ln b ↔ 10^a > 10^b)) ∧
  (¬ (∀ a b : ℝ, ((f a b (-1) = 0) → (f' a b (-1) = 0) → (a = 2 ∧ b = 9) ∨ (a = 1 ∧ b = 3))))
:=
begin
  sorry
end

-- Auxiliary definitions required for the theorem statement.
def f (a b x : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

def f' (a b x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + b

end propositions_evaluation_l728_728506


namespace binom_7_4_eq_35_l728_728176

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728176


namespace min_partitions_to_boundary_l728_728482

theorem min_partitions_to_boundary (n : ℕ) (h : n ≥ 3) : 
  min_partitions_to_reach_boundary n = (n - 2) ^ 3 := 
sorry

end min_partitions_to_boundary_l728_728482


namespace cost_per_pack_l728_728470

theorem cost_per_pack (packs : ℕ) (total_cost : ℕ) (h1 : packs = 6) (h2 : total_cost = 120) : total_cost / packs = 20 :=
by
  rw [h1, h2]
  rfl

end cost_per_pack_l728_728470


namespace find_k_l728_728354

-- Define the conditions
variables (a b : Real) (x y : Real)

-- The problem's conditions
def tan_x : Prop := Real.tan x = a / b
def tan_2x : Prop := Real.tan (x + x) = b / (a + b)
def y_eq_x : Prop := y = x

-- The goal to prove
theorem find_k (ha : tan_x a b x) (hb : tan_2x a b x) (hy : y_eq_x x y) :
  ∃ k, x = Real.arctan k ∧ k = 1 / (a + 2) :=
sorry

end find_k_l728_728354


namespace miles_ridden_further_l728_728423

theorem miles_ridden_further (distance_ridden distance_walked : ℝ) (h1 : distance_ridden = 3.83) (h2 : distance_walked = 0.17) :
  distance_ridden - distance_walked = 3.66 := 
by sorry

end miles_ridden_further_l728_728423


namespace g_prime_zeros_on_unit_circle_l728_728345

def is_on_unit_circle (z : ℂ) : Prop := complex.abs z = 1

def polynomial_on_unit_circle (p : polynomial ℂ) : Prop :=
  ∀ z : ℂ, p.eval z = 0 → is_on_unit_circle z

-- Definitions for the conditions
variables {p : polynomial ℂ} (n : ℕ)
  (hp : p.degree = n)
  (hp_roots : polynomial_on_unit_circle p)

noncomputable def g (z : ℂ) : ℂ :=
p.eval z / z ^ (n / 2)

-- Statement to be proved
theorem g_prime_zeros_on_unit_circle :
  ∀ z : ℂ, (derivative g) z = 0 → is_on_unit_circle z :=
sorry

end g_prime_zeros_on_unit_circle_l728_728345


namespace division_of_power_l728_728786

theorem division_of_power :
  let n := 16^4044
  in n / 16 = 4^8086 := 
by
  let n := 16^4044
  sorry

end division_of_power_l728_728786


namespace card_U_l728_728049

-- Definitions of the sets U, A, and B
variable (U A B : Set α)

-- Given conditions
variable (card_B : ∀ (s : Set α), s = B → s.card = 49)
variable (card_not_A_or_B : ∀ (s : Set α), s = U \ (A ∪ B) → s.card = 59)
variable (card_A_inter_B : ∀ (s : Set α), s = A ∩ B → s.card = 23)
variable (card_A : ∀ (s : Set α), s = A → s.card = 107)

-- Prove the total number of items in set U
theorem card_U : U.card = 192 := by
  sorry

end card_U_l728_728049


namespace positive_difference_of_solutions_eq_4_l728_728438

theorem positive_difference_of_solutions_eq_4 (r : ℝ) (h : r ≠ -5) :
  let eq := (r^2 - 5*r - 24) / (r + 5) = 3*r + 8
  in |neg_add_of_neg_of_nonpos (-8) (neg_nonpos_of_nonneg 4)| = 4 :=
by
  sorry

end positive_difference_of_solutions_eq_4_l728_728438


namespace find_m_l728_728670

-- Define the vectors a and b
def a : ℝ × ℝ := (3, m)
def b : ℝ × ℝ := (2, -1)

-- Condition: a dot b equals 0
def dot_product_zero (m : ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2 = 0)

-- The main statement to prove
theorem find_m:
  dot_product_zero m → m = 6 :=
begin
  sorry
end

end find_m_l728_728670


namespace probability_three_heads_in_eight_tosses_l728_728936

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l728_728936


namespace Fran_speed_l728_728333

def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 3.5

theorem Fran_speed :
  ∀ (s : ℝ), (s * Fran_time = Joann_speed * Joann_time) → (s = 120 / 7) :=
by
  intro s h
  sorry

end Fran_speed_l728_728333


namespace factorial_difference_l728_728539

-- Define factorial function for natural numbers
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- The theorem we want to prove
theorem factorial_difference : factorial 10 - factorial 9 = 3265920 :=
by
  rw [factorial, factorial]
  sorry

end factorial_difference_l728_728539


namespace expansion_coefficient_l728_728754
noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| n 0     := 1
| 0 k     := 0
| n (k+1) := binomial_coefficient (n-1) k + binomial_coefficient n k

theorem expansion_coefficient :
  let a := 2
  let b := (λ (x : ℝ), -1/x)
  let n := 6
  let r := 2
  let general_term (a : ℝ) (b : ℝ → ℝ) (n : ℕ) (r : ℕ) (x : ℝ) := 
    (-1)^r * (a^(n-r)) * (binomial_coefficient n r) * x^(18 - 4 * r)
  in (general_term a b n r 1) = 240 :=
by
  sorry

end expansion_coefficient_l728_728754


namespace domain_ln_4_minus_x_l728_728212

def f (x : ℝ) : ℝ := Real.log (4 - x)

theorem domain_ln_4_minus_x : 
  {x : ℝ | 0 < 4 - x} = set.Iio 4 := 
sorry

end domain_ln_4_minus_x_l728_728212


namespace number_of_elements_in_P_intersection_M_l728_728276

open Set

def P : Set ℝ := {x | x * (x - 1) ≤ 0}
def M : Set ℝ := {0, 1, 3, 4}

theorem number_of_elements_in_P_intersection_M : (P ∩ M).toFinset.card = 2 := by
  sorry

end number_of_elements_in_P_intersection_M_l728_728276


namespace log_eq_value_l728_728695

theorem log_eq_value (x : ℝ) (h : log 8 x = 3 / 2) : x = 16 * real.sqrt 2 := 
by 
sorry

end log_eq_value_l728_728695


namespace unique_valid_centroids_count_l728_728389

structure Point :=
  (x : ℤ) (y : ℤ)

def is_valid_point (p : Point) : Prop :=
  (p.x = 0 ∧ 0 ≤ p.y ∧ p.y ≤ 20) ∨
  (p.y = 20 ∧ 0 ≤ p.x ∧ p.x ≤ 20) ∨
  (p.x = 20 ∧ 0 ≤ p.y ∧ p.y ≤ 20) ∨
  (p.y = 0 ∧ 0 ≤ p.x ∧ p.x ≤ 20)

def centroid (p q r : Point) : Point :=
  { x := (p.x + q.x + r.x) / 3, y := (p.x + q.y + r.y) / 3 }

def is_valid_centroid (g : Point) : Prop :=
  1 ≤ g.x ∧ g.x ≤ 59 ∧ 1 ≤ g.y ∧ g.y ≤ 59

theorem unique_valid_centroids_count :
  ∃ n : ℕ, n = 3481 ∧ ∀ p q r : Point,
    distinct {p, q, r} →
    ¬collinear {p, q, r} →
    is_valid_point p →
    is_valid_point q →
    is_valid_point r →
    is_valid_centroid (centroid p q r) :=
sorry

end unique_valid_centroids_count_l728_728389


namespace trigonometric_identity_l728_728707

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trigonometric_identity_l728_728707


namespace fractions_sum_simplified_l728_728204

noncomputable def frac12over15 : ℚ := 12 / 15
noncomputable def frac7over9 : ℚ := 7 / 9
noncomputable def frac1and1over6 : ℚ := 1 + 1 / 6

theorem fractions_sum_simplified :
  frac12over15 + frac7over9 + frac1and1over6 = 247 / 90 :=
by
  -- This step will be left as a proof to complete.
  sorry

end fractions_sum_simplified_l728_728204


namespace fran_speed_l728_728338

variable (s : ℝ)

theorem fran_speed
  (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ)
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5)
  (h4 : fran_time * s = joann_speed * joann_time)
  : s = 120 / 7 := 
by
  sorry

end fran_speed_l728_728338


namespace total_marbles_l728_728053

-- Definitions based on conditions
def bowl_capacity_ratio (B1 B2 : ℕ) : Prop := B1 = 3 * B2 / 4
def second_bowl_marbles : ℕ := 600

-- Theorem statement
theorem total_marbles : ∃ B1 B2, bowl_capacity_ratio B1 B2 ∧ B2 = second_bowl_marbles ∧ (B1 + B2 = 1050) :=
by
  -- Let B2 be the capacity of the second bowl
  let B2 := second_bowl_marbles
  -- Let B1 be the capacity of the first bowl, which is 3/4 of B2
  let B1 := 3 * B2 / 4
  -- Prove the statement
  use [B1, B2]
  repeat {split}
  -- Prove the capacity ratio
  · exact rfl
  -- Prove B2 is 600
  · exact rfl
  -- Prove the total number of marbles
  · exact sorry

end total_marbles_l728_728053


namespace goals_per_player_is_30_l728_728031

-- Define the total number of goals scored in the league against Barca
def total_goals : ℕ := 300

-- Define the percentage of goals scored by the two players
def percentage_of_goals : ℝ := 0.20

-- Define the combined goals by the two players
def combined_goals := (percentage_of_goals * total_goals : ℝ)

-- Define the number of players
def number_of_players : ℕ := 2

-- Define the number of goals scored by each player
noncomputable def goals_per_player := combined_goals / number_of_players

-- Proof statement: Each of the two players scored 30 goals.
theorem goals_per_player_is_30 :
  goals_per_player = 30 :=
sorry

end goals_per_player_is_30_l728_728031


namespace batteries_in_controllers_l728_728859

theorem batteries_in_controllers
    (b_flashlights b_toys b_total : ℕ)
    (h_flashlights : b_flashlights = 2)
    (h_toys : b_toys = 15)
    (h_total : b_total = 19) :
    b_total - (b_flashlights + b_toys) = 2 := 
by
    rw [h_flashlights, h_toys, h_total]
    sworry

end batteries_in_controllers_l728_728859


namespace probability_exactly_three_heads_l728_728964
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l728_728964


namespace factorial_difference_l728_728545

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem factorial_difference : fact 10 - fact 9 = 3265920 := by
  have h1 : fact 9 = 362880 := sorry
  have h2 : fact 10 = 10 * fact 9 := by rw [fact]
  calc
    fact 10 - fact 9
        = 10 * fact 9 - fact 9 := by rw [h2]
    ... = 9 * fact 9 := by ring
    ... = 9 * 362880 := by rw [h1]
    ... = 3265920 := by norm_num

end factorial_difference_l728_728545


namespace find_arithmetic_sequence_l728_728627

variable {n q : ℕ} (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

-- Conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n * q

def arithmetic_sum (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def geometric_sum (T : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, T n = b 1 * (q ^ n - 1) / (q - 1)

-- Given condition
axiom sum_relation (T S : ℕ → ℕ) (n : ℕ) : T (2 * n) + 1 = S (q ^ n)

-- The proof problem
theorem find_arithmetic_sequence
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (S : ℕ → ℕ) (T : ℕ → ℕ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b q)
  (hS : arithmetic_sum S a)
  (hT : geometric_sum T b)
  (h : sum_relation T S n) :
  ∀ n : ℕ, a n = 2 * n - 1 :=
sorry

end find_arithmetic_sequence_l728_728627


namespace benny_leftover_money_l728_728140

-- Define the conditions
def initial_money : ℕ := 67
def spent_money : ℕ := 34

-- Define the leftover money calculation
def leftover_money : ℕ := initial_money - spent_money

-- Prove that Benny had 33 dollars left over
theorem benny_leftover_money : leftover_money = 33 :=
by 
  -- Proof
  sorry

end benny_leftover_money_l728_728140


namespace exists_int_divisible_by_2_pow_100_with_digits_8_and_9_l728_728818

theorem exists_int_divisible_by_2_pow_100_with_digits_8_and_9 :
  ∃ N : ℤ, (2^100 ∣ N) ∧ (∀ d : ℕ, d ∈ N.digits 10 → d = 8 ∨ d = 9) :=
sorry

end exists_int_divisible_by_2_pow_100_with_digits_8_and_9_l728_728818


namespace coin_toss_probability_l728_728917

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l728_728917


namespace BD_MN_tangent_intersect_at_point_l728_728789

noncomputable theory
open_locale classical

-- Definitions for the problem conditions
variables (A B C D M N : Point)
variables (parallelogram : Parallelogram A B C D)
variables (circle : Circle (midpoint A C) (A.dist(C) / 2))
variables (intersect_AB : Line A B)
variables (intersect_AD : Line A D)
variables (M_on_circle : is_on_circle M circle)
variables (N_on_circle : is_on_circle N circle)

-- The tangent line to the circle at point C
def tangent_line_C := tangent_circle_at_point circle C

-- Statement of the theorem
theorem BD_MN_tangent_intersect_at_point :
  ∃ P : Point, (Line B D).is_on P ∧ (Line M N).is_on P ∧ tangent_line_C.is_on P :=
sorry

end BD_MN_tangent_intersect_at_point_l728_728789


namespace standard_eq_ellipse_range_area_OMN_l728_728250

section proof

variables {a b : ℝ} (h_a_gt_b_gt_0 : a > b) (h_b_gt_0 : b > 0) (h_a_gt_0 : a > 0)
variables (C : Set (ℝ × ℝ)) (hyperbola : Set (ℝ × ℝ)) (line : Set (ℝ × ℝ))

-- Given conditions
def ellipseEquation (a b : ℝ) : Prop := ∃ (x y : ℝ), (x, y) ∈ C ∧ (x^2 / a^2 + y^2 / b^2 = 1)
def hyperbolaEquation : Prop := ∃ (x y : ℝ), (x, y) ∈ hyperbola ∧ (x^2 / 3 - y^2 = 1)
def reciprocalsOfEccentricities : Prop := ∃ e_h e_e : ℝ, e_h * e_e = 1 ∧ hyperbolaEccentricity e_h ∧ ellipseEccentricity e_e
def passesThroughRightVertex : Prop := ∃ x y : ℝ, (x - y - 2 = 0) ∧ (x, y) ∈ rightVertex 

-- Questions to be proven
theorem standard_eq_ellipse (h : ellipseEquation a b) (hr : reciprocalsOfEccentricities) (hp : passesThroughRightVertex)
  : (a = 2) ∧ (b = 1) ∧ (C = {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ x^2 / 4 + y^2 = 1}) :=
sorry

theorem range_area_OMN (lineNotThroughOrigin : line ≠ {p : ℝ × ℝ | p = (0, 0)}) : 0 < areaOMN < 1 :=
sorry

end proof

end standard_eq_ellipse_range_area_OMN_l728_728250


namespace fran_speed_l728_728335

theorem fran_speed :
  ∀ (Joann_speed Fran_time : ℝ), Joann_speed = 15 → Joann_time = 4 → Fran_time = 3.5 →
    (Joann_speed * Joann_time = Fran_time * (120 / 7)) :=
by
  intro Joann_speed Joann_time Fran_time
  sorry

end fran_speed_l728_728335


namespace probability_sum_condition_l728_728473

theorem probability_sum_condition {Ω : Type*} (s : finset Ω) (independent : Ω → Prop) (event_A : finset (finset Ω)) (event_B : finset (finset Ω)) :
  s.card = 10 →
  (∃ (A B : finset Ω), A.card = 5 ∧ B.card = 5 ∧ A ∪ B = s ∧ A ∩ B = ∅) →
  (∀ (x : Ω), x ∈ s → independent x) →
  (∀ (x : Ω), x ∈ s → (x ∈ event_A ∨ x ∈ event_B) ∨ (x ∉ event_A ∧ x ∉ event_B)) →
  (event_A.card = 252) →
  (finset.card (event_A.filter (λ t, t.card = 5 ∧ (t.sum id = 2 ∨ t.sum id = 3))) = 200) →
  (real.to_nnreal (finset.card (event_A.filter (λ t, t.card = 5 ∧ (t.sum id < 2 ∨ t.sum id > 3)))) / real.to_nnreal event_A.card = 38 / 63) :=
by
  intros
  sorry

end probability_sum_condition_l728_728473


namespace arabic_numerals_eq_natural_numbers_l728_728318

-- Define what Arabic numerals are as a universally recognized symbol set
def arabic_numerals : Set ℕ :=
  { n | n ∈ ℕ }

-- State the theorem equivalently in Lean 4
theorem arabic_numerals_eq_natural_numbers : arabic_numerals = {n | n ∈ ℕ} :=
by
  -- Proof goes here
  sorry

end arabic_numerals_eq_natural_numbers_l728_728318


namespace insurance_not_covered_percentage_l728_728052

noncomputable def insurance_monthly_cost : ℝ := 20
noncomputable def insurance_months : ℝ := 24
noncomputable def procedure_cost : ℝ := 5000
noncomputable def amount_saved : ℝ := 3520

theorem insurance_not_covered_percentage :
  ((procedure_cost - amount_saved - (insurance_monthly_cost * insurance_months)) / procedure_cost) * 100 = 20 :=
by
  sorry

end insurance_not_covered_percentage_l728_728052


namespace max_number_of_liars_l728_728307

open Finset

noncomputable def max_liars_in_castle : Nat :=
  let n := 4
  let rooms := Fin n × Fin n
  let neighbors (i j : rooms) : Prop :=
    (i.1 = j.1 ∧ (i.2 = j.2 + 1 ∨ i.2 + 1 = j.2)) ∨
    (i.2 = j.2 ∧ (i.1 = j.1 + 1 ∨ i.1 + 1 = j.1))
  sorry

theorem max_number_of_liars (liar knight : Fin 16 -> Prop)
  (liars_knights_split : ∀ x, liar x ∨ knight x)
  (liar_truth : ∀ (i : Fin 16),
    liar i → (∑ j in (filter (neighbors − [i])) id).card = 0)
  (knight_truth : ∀ (i : Fin 16),
    knight i → (∑ j in (filter (neighbors − [i])) id).card ≥ 1) :
  max_liars_in_castle = 8 :=
sorry

end max_number_of_liars_l728_728307


namespace smallest_positive_integer_n_l728_728630

variable {a : ℕ → ℝ} -- Arithmetic sequence {a_n}
variable {S : ℕ → ℝ} -- Sum of the first n terms of the sequence

-- Define the conditions
def S6_gt_S7 : Prop := S 6 > S 7
def S7_gt_S5 : Prop := S 7 > S 5

-- Define the conclusion
def smallest_n (n : ℕ) : Prop := S n < 0

theorem smallest_positive_integer_n 
    (a : ℕ → ℝ)
    (S : ℕ → ℝ)
    (h1 : S6_gt_S7)
    (h2 : S7_gt_S5) :
    ∃ n : ℕ, smallest_n n ∧ (∀ m : ℕ, m < n → ¬smallest_n m) :=
begin
  -- Given the conditions
  -- S6 > S7 and S7 > S5
  -- We want to show the smallest n such that S_n < 0 is 13
  use 13,
  split,
  { sorry },  -- S_13 < 0
  { intros m hm,
    sorry }   -- for all m < 13, S_m >= 0
end

end smallest_positive_integer_n_l728_728630


namespace tan_alpha_expression_l728_728292

theorem tan_alpha_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 :=
by
  sorry

end tan_alpha_expression_l728_728292


namespace coin_toss_probability_l728_728920

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l728_728920


namespace value_of_T_l728_728373

def one_third_of_one_fifth (T : ℝ) : ℝ := 1/3 * 1/5 * T
def one_fourth_of_one_sixth_of_120 : ℝ := 1/4 * 1/6 * 120

theorem value_of_T : one_third_of_one_fifth T = one_fourth_of_one_sixth_of_120 → T = 75 :=
by
  intro h
  sorry

end value_of_T_l728_728373


namespace ten_factorial_minus_nine_factorial_l728_728566

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem ten_factorial_minus_nine_factorial :
  factorial 10 - factorial 9 = 3265920 := 
by 
  sorry

end ten_factorial_minus_nine_factorial_l728_728566


namespace triangle_inequality_l728_728791

theorem triangle_inequality (a b c S : ℝ)
  (h : a ≠ b ∧ b ≠ c ∧ c ≠ a)   -- a, b, c are sides of a non-isosceles triangle
  (S_def : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  (a^3) / ((a - b) * (a - c)) + (b^3) / ((b - c) * (b - a)) + (c^3) / ((c - a) * (c - b)) > 2 * 3^(3/4) * S :=
by
  sorry

end triangle_inequality_l728_728791


namespace distance_DE_l728_728860

def Point := ℝ × ℝ

variables (A B C D E P : Point)
variables (AB AC BC PC DE : ℝ)

axiom AB_eq : AB = 15
axiom BC_eq : BC = 20
axiom AC_eq : AC = 25
axiom PC_eq : PC = 15
axiom on_BP : ∃ t1 t2 : ℝ, D = (t1 * B.1 + (1 - t1) * P.1, t1 * B.2 + (1 - t1) * P.2) ∧ 
                            E = (t2 * B.1 + (1 - t2) * P.1, t2 * B.2 + (1 - t2) * P.2)

axiom trapezoid_ABCD : ∃ m1 m2 : ℝ, 
  is_linear_combination A B C D ∧ 
  is_linear_combination A B C E

theorem distance_DE : DE = 3 * Real.sqrt 5 :=
sorry

end distance_DE_l728_728860


namespace line_parallel_l728_728899

theorem line_parallel (x y : ℝ) :
  ∃ m b : ℝ, 
    y = m * (x - 2) + (-4) ∧ 
    m = 2 ∧ 
    (∀ (x y : ℝ), y = 2 * x - 8 → 2 * x - y - 8 = 0) :=
sorry

end line_parallel_l728_728899


namespace unit_cubes_with_paint_l728_728191

/-- Conditions:
1. Cubes with each side one inch long are glued together to form a larger cube.
2. The larger cube's face is painted with red color and the entire assembly is taken apart.
3. 23 small cubes are found with no paints on them.
-/
theorem unit_cubes_with_paint (n : ℕ) (h1 : n^3 - (n - 2)^3 = 23) (h2 : n = 4) :
    n^3 - 23 = 41 :=
by
  sorry

end unit_cubes_with_paint_l728_728191


namespace coin_toss_probability_l728_728943

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l728_728943


namespace average_age_correct_l728_728046

-- Definitions for the problem conditions:
def age_youngest : ℝ := 20
def age_sibling1 : ℝ := age_youngest + 2
def age_sibling2 : ℝ := age_youngest + 7
def age_sibling3 : ℝ := age_youngest + 11

-- The average age of the siblings
def average_age (a b c d : ℝ) : ℝ := (a + b + c + d) / 4

-- The statement to be proved
theorem average_age_correct :
  average_age age_youngest age_sibling1 age_sibling2 age_sibling3 = 25 :=
by
  sorry

end average_age_correct_l728_728046


namespace investment_years_l728_728109

theorem investment_years
  (P : ℝ) (r₁ r₂ : ℝ) (I_diff : ℝ) (n : ℝ)
  (hP : P = 12000)
  (h_r₁ : r₁ = 0.15)
  (h_r₂ : r₂ = 0.12)
  (h_I_diff : I_diff = 720) :
  n = I_diff / (P * (r₁ - r₂)) :=
by {
  unfold P r₁ r₂ I_diff at *,
  rw [hP, h_r₁, h_r₂, h_I_diff],
  sorry,
}

end investment_years_l728_728109


namespace number_of_subsets_of_H_l728_728455

def is_natural (x : ℤ) : Prop := x > 0

def H : Set (ℤ × ℤ) := 
  { p | let x := p.1 in let y := p.2 in
        (x - y) ^ 2 + x ^ 2 - 15 * x + 50 = 0 ∧ is_natural x ∧ is_natural y }

theorem number_of_subsets_of_H : (2 ^ (H.card)) = 64 := by
  sorry

end number_of_subsets_of_H_l728_728455


namespace limit_na_n_l728_728193

def L (x : ℝ) : ℝ := x - x^3 / 3

def a_n (n : ℕ) : ℝ := (Iterate.iterate L n) (6 / n)

theorem limit_na_n : tendsto (λ n : ℕ, n * a_n n) at_top (𝓝 6) := 
sorry

end limit_na_n_l728_728193


namespace binom_7_4_eq_35_l728_728166

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728166


namespace modified_lucas_50th_term_mod_5_l728_728838

-- Define the modified Lucas sequence
def modifiedLucas : ℕ → ℕ
| 0     := 2
| 1     := 5
| (n+2) := modifiedLucas n + modifiedLucas (n+1)

-- The theorem to prove
theorem modified_lucas_50th_term_mod_5 : (modifiedLucas 50) % 5 = 0 :=
by
  sorry

end modified_lucas_50th_term_mod_5_l728_728838


namespace man_rate_in_still_water_l728_728093

theorem man_rate_in_still_water (speed_stream : ℝ) (speed_against_stream : ℝ) (h1 : speed_stream = 26) (h2 : speed_against_stream = 12) :
    (speed_stream + speed_against_stream) / 2 = 19 :=
by
  rw [h1, h2]
  norm_num
  sorry

end man_rate_in_still_water_l728_728093


namespace ratio_of_vanilla_chips_l728_728435

-- Definitions from the conditions
variable (V_c S_c V_v S_v : ℕ)
variable (H1 : V_c = S_c + 5)
variable (H2 : S_c = 25)
variable (H3 : V_v = 20)
variable (H4 : V_c + S_c + V_v + S_v = 90)

-- The statement we want to prove
theorem ratio_of_vanilla_chips : S_v / V_v = 3 / 4 := by
  sorry

end ratio_of_vanilla_chips_l728_728435


namespace number_of_green_bows_l728_728735

theorem number_of_green_bows (T : ℕ)
  (h_red : (3 / 20) * T) 
  (h_blue : (3 / 10) * T) 
  (h_green : (1 / 5) * T) 
  (h_purple : (1 / 20) * T) 
  (h_white : 24): 
  (h_green_bows : (1 / 5) * T = 16) :=
begin
 sorry
end

end number_of_green_bows_l728_728735


namespace sophia_estimate_larger_l728_728319

theorem sophia_estimate_larger (x y a b : ℝ) (hx : x > y) (hy : y > 0) (ha : a > 0) (hb : b > 0) :
  (x + a) - (y - b) > x - y := by
  sorry

end sophia_estimate_larger_l728_728319


namespace mandy_yoga_time_l728_728796

theorem mandy_yoga_time (G B Y : ℕ) (h1 : 2 * B = 3 * G) (h2 : 3 * Y = 2 * (G + B)) (h3 : Y = 30) : Y = 30 := by
  sorry

end mandy_yoga_time_l728_728796


namespace probability_exactly_three_heads_l728_728970
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l728_728970


namespace exists_unique_C_exists_lambda_l728_728467

section PartA

variables {n : ℕ} (f : Matrix (Fin n) (Fin n) ℝ → ℝ)

-- Condition (a): f is a linear mapping
-- Question (a): Prove that there exists a unique C such that f(A) = Tr(AC) for all A

theorem exists_unique_C (hf : IsLinearMap ℝ f) :
  ∃! C : Matrix (Fin n) (Fin n) ℝ, ∀ A : Matrix (Fin n) (Fin n) ℝ, f A = Matrix.trace (A * C) :=
sorry

end PartA

section PartB

variables {n : ℕ} (f : Matrix (Fin n) (Fin n) ℝ → ℝ)

-- Additional Condition (b): f(AB) = f(BA) for all A, B
-- Question (b): Prove that there exists λ such that f(A) = λ Tr(A)

theorem exists_lambda (hf : IsLinearMap ℝ f) (sym : ∀ A B : Matrix (Fin n) (Fin n) ℝ, f (A * B) = f (B * A)) :
  ∃ λ : ℝ, ∀ A : Matrix (Fin n) (Fin n) ℝ, f A = λ * Matrix.trace A :=
sorry

end PartB

end exists_unique_C_exists_lambda_l728_728467


namespace range_of_quadratic_function_l728_728412

noncomputable def quadratic_function_range : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = x^2 - 6 * x + 7 }

theorem range_of_quadratic_function :
  quadratic_function_range = { y : ℝ | y ≥ -2 } :=
by
  -- Insert proof here
  sorry

end range_of_quadratic_function_l728_728412


namespace find_m_l728_728259

-- Define the required conditions as a Lean 4 proposition
theorem find_m (m : ℤ) :
  (|m| = 2) ∧ (m ≠ 2) → m = -2 :=
by 
  intro h,
  cases h with h1 h2,
  sorry

end find_m_l728_728259


namespace a_1000_is_3334_l728_728733

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 3001 ∧
  a 2 = 3002 ∧
  ∀ n, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n + 1

theorem a_1000_is_3334 (a : ℕ → ℤ) (h : sequence a) : a 1000 = 3334 :=
by sorry

end a_1000_is_3334_l728_728733


namespace probability_three_heads_in_eight_tosses_l728_728995

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l728_728995


namespace convergence_equiv_convergence_set_divergence_equiv_divergence_set_l728_728359

variable {Ω : Type*} {ξ : ℕ → Ω → ℝ}

-- Definition of the convergence set
def convergence_set (ξ : ℕ → Ω → ℝ) : set Ω :=
  ⋂ n : ℕ, ⋃ m : ℕ, ⋂ k (hk : k ≥ m), {ω | ∀ l (hl : l ≥ k), |ξ l ω - ξ k ω| ≤ n⁻¹}

-- Definition of the divergence set
def divergence_set (ξ : ℕ → Ω → ℝ) : set Ω :=
  ⋃ n : ℕ, ⋂ m : ℕ, ⋃ k (hk : k ≥ m), {ω | ∃ l (hl : l ≥ k), |ξ l ω - ξ k ω| > n⁻¹}

-- Theorem statement for convergence
theorem convergence_equiv_convergence_set :
  {ω | ∃ c, ∀ ε > 0, ∃ N, ∀ m n ≥ N, |ξ m ω - ξ n ω| < ε} = convergence_set ξ := 
sorry

-- Theorem statement for divergence
theorem divergence_equiv_divergence_set :
  {ω | ¬ ∃ c, ∀ ε > 0, ∃ N, ∀ m n ≥ N, |ξ m ω - ξ n ω| < ε} = divergence_set ξ := 
sorry

end convergence_equiv_convergence_set_divergence_equiv_divergence_set_l728_728359


namespace length_of_platform_is_280_l728_728095

-- Add conditions for speed, times and conversions
def speed_kmph : ℕ := 72
def time_platform : ℕ := 30
def time_man : ℕ := 16

-- Conversion from km/h to m/s
def speed_mps : ℤ := speed_kmph * 1000 / 3600

-- The length of the train when it crosses the man
def length_of_train : ℤ := speed_mps * time_man

-- The length of the platform
def length_of_platform : ℤ := (speed_mps * time_platform) - length_of_train

theorem length_of_platform_is_280 :
  length_of_platform = 280 := by
  sorry

end length_of_platform_is_280_l728_728095


namespace binomial_7_4_equals_35_l728_728171

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Problem statement: Prove that binomial(7, 4) = 35
theorem binomial_7_4_equals_35 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_equals_35_l728_728171


namespace probability_two_rainy_days_l728_728400

theorem probability_two_rainy_days : 
  let numbers := ["907", "966", "191", "925", "271", "932", "812", "458", "569", "683",
                  "431", "257", "393", "027", "556", "488", "730", "113", "537", "989"];
  let rain_condition := ['1', '2', '3', '4'];
  let is_two_rainy_days := λ s : String, s.to_list.filter (λ x, x ∈ rain_condition).length = 2;
  let valid_groups := numbers.filter is_two_rainy_days;
  valid_groups.length = 5 →
  (5 : ℚ) / (20 : ℚ) = (1 : ℚ) / (4 : ℚ) :=
by sorry

end probability_two_rainy_days_l728_728400


namespace quadratic_root_a_l728_728421

theorem quadratic_root_a (a : ℝ) : (∃ x : ℝ, x^2 + x + a^2 - 1 = 0) → (a = 1 ∨ a = -1) :=
begin
  intro h,
  cases h with x hx,
  have : x = 0, from sorry, -- Given condition that one root is 0
  rw this at hx,
  simp at hx,
  exact sorry,
end

end quadratic_root_a_l728_728421


namespace original_price_of_shoes_l728_728621

theorem original_price_of_shoes (x : ℝ) (h : 1/4 * x = 18) : x = 72 := by
  sorry

end original_price_of_shoes_l728_728621


namespace binom_7_4_eq_35_l728_728168

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728168


namespace area_of_EPGH_is_l728_728378

noncomputable def area_of_EPGH {EFGH P Q : Type} [MetricSpace EFGH] [NormedSpace ℝ EFGH] 
  (a b : ℝ) (h : a = 12) (h2 : b = 6) (P_mid : midpoint (b / 2)) (Q_mid : midpoint (a / 2)) 
  (diagonal_EQ : ℝ) : ℝ :=
  let area_EFGH := a * b in
  let half_area_EFGH := area_EFGH / 2 in
  let area_EPQ := 1 / 2 * (P_mid) * (Q_mid) in
  half_area_EFGH + area_EPQ

theorem area_of_EPGH_is (EFGH P Q : Type) [MetricSpace EFGH] [NormedSpace ℝ EFGH]
  (a b : ℝ) (h : a = 12) (h2 : b=  6) (P_mid : midpoint (b / 2)) (Q_mid : midpoint (a / 2)) 
  (diagonal_EQ : ℝ) : area_of_EPGH a b h h2 P_mid Q_mid diagonal_EQ = 45 :=
by
  sorry

end area_of_EPGH_is_l728_728378


namespace express_in_scientific_notation_l728_728895

theorem express_in_scientific_notation :
  ∀ (n : ℕ), n = 1300000 → scientific_notation n = "1.3 × 10^6" :=
by
  intros n h
  have h1 : n = 1300000 := by exact h
  sorry

end express_in_scientific_notation_l728_728895


namespace number_of_one_dollar_bills_l728_728813

/-- Define variables and conditions -/
variables {x y : ℕ}

/-- First condition: total number of bills is 58 -/
def total_bills (x y : ℕ) : Prop := x + y = 58

/-- Second condition: total dollar amount is 84 -/
def total_amount (x y : ℕ) : Prop := x + 2 * y = 84

/-- The main theorem: given the conditions, prove that the number of one dollar bills is 32 -/
theorem number_of_one_dollar_bills (x y : ℕ) (h1 : total_bills x y) (h2 : total_amount x y) : x = 32 :=
by
  -- Proof will be filled in here
  sorry

end number_of_one_dollar_bills_l728_728813


namespace sqrt_720_simplified_l728_728002

theorem sqrt_720_simplified : Real.sqrt 720 = 6 * Real.sqrt 5 := sorry

end sqrt_720_simplified_l728_728002


namespace trigonometric_identity_l728_728711

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
by 
  sorry

end trigonometric_identity_l728_728711


namespace quadratic_root_zero_l728_728616

theorem quadratic_root_zero (k : ℝ) :
  (∃ x : ℝ, (k + 2) * x^2 + 6 * x + k^2 + k - 2 = 0) →
  (∃ x : ℝ, x = 0 ∧ ((k + 2) * x^2 + 6 * x + k^2 + k - 2 = 0)) →
  k = 1 :=
by
  sorry

end quadratic_root_zero_l728_728616


namespace minimum_dot_product_l728_728281

-- Define the vectors OP, OA, and OB
def OP : ℝ × ℝ := (2, 1)
def OA : ℝ × ℝ := (1, 7)
def OB : ℝ × ℝ := (5, 1)

-- Define M on the line OP
def OM (λ : ℝ) : ℝ × ℝ := (2 * λ, λ)

-- Define MA and MB with respect to OM
def MA (λ : ℝ) : ℝ × ℝ := (1 - 2 * λ, 7 - λ)
def MB (λ : ℝ) : ℝ × ℝ := (5 - 2 * λ, 1 - λ)

-- Define the dot product function for two vectors
def dotProduct (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Translate the mathematical problem into Lean to determine the minimum value
theorem minimum_dot_product : 
  ∃ λ : ℝ, λ = 2 ∧ dotProduct (MA λ) (MB λ) = -8 :=
by
  sorry

end minimum_dot_product_l728_728281


namespace triangle_angle_relation_l728_728427

theorem triangle_angle_relation (A B C P : Point) (hBC : dist B C = dist A C + dist A B / 2) 
    (hP : 3 * dist B P = dist A P) : angle C A P = 2 * angle C P A := by sorry

end triangle_angle_relation_l728_728427


namespace smallest_n_for_Q_lt_1_over_2023_l728_728738

-- Definitions based on conditions
def Q (n : ℕ) : ℚ :=
  let q (k : ℕ) : ℚ := (k^2 + k) / (k^2 + k + 1)
  (List.prod (List.map q (List.range (n - 1)))) * (1 / (n^2 + n + 1))

theorem smallest_n_for_Q_lt_1_over_2023 : ∃ n : ℕ, Q(n) < (1 / 2023) ∧ ∀ m : ℕ, m < n → Q(m) ≥ (1 / 2023) :=
by {
  existsi 45,
  split,
  {
    -- proving Q(45) < 1/2023
    -- sorry,
  },
  {
    intros m Hm,
    -- proving that if m < 45, Q(m) ≥ 1/2023
    -- sorry,
  }
  sorry
}

end smallest_n_for_Q_lt_1_over_2023_l728_728738


namespace max_cards_arranged_l728_728047

-- Define the set of cards
def cards : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a function to check if a sequence satisfies the adjacency condition
def satisfies_condition (seq : List ℕ) : Prop :=
  ∀ (x y : ℕ), x ∈ seq.tail ∧ y ∈ seq ∧ list.head seq = y → (x % y = 0 ∨ y % x = 0)

-- Define the maximum number of cards that can be arranged
def max_arrangement : ℕ := 8

-- The main theorem
theorem max_cards_arranged (seq : List ℕ) (h : seq.toFinset = cards ∪ {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  satisfies_condition seq →
  seq.length = max_arrangement :=
sorry

end max_cards_arranged_l728_728047


namespace prob_three_heads_in_eight_tosses_l728_728960

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l728_728960


namespace correct_order_of_operations_for_adding_rationals_with_different_signs_l728_728441

theorem correct_order_of_operations_for_adding_rationals_with_different_signs :
  ∀ (a b : ℚ), 
    (let abs_a := |a|,
         abs_b := |b|,
         larger := max abs_a abs_b,
         smaller := min abs_a abs_b,
         sign := if abs_a > abs_b then sign a else sign b,
         magnitude := larger - smaller
     in sign * magnitude = a + b) :=
by
  intros a b
  let abs_a := |a|
  let abs_b := |b|
  let larger := max abs_a abs_b
  let smaller := min abs_a abs_b
  let sign := if abs_a > abs_b then sign a else sign b
  let magnitude := larger - smaller
  show sign * magnitude = a + b
  sorry

end correct_order_of_operations_for_adding_rationals_with_different_signs_l728_728441


namespace find_f_f_neg_one_l728_728655

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 - 2^x else Real.log x / Real.log 2

theorem find_f_f_neg_one : f (f (-1)) = -1 := by
  sorry

end find_f_f_neg_one_l728_728655


namespace water_left_l728_728852

def steady_flow (rate: ℕ) (duration: ℕ) : ℕ := (rate * (duration / 10))

theorem water_left {rate1 rate2 rate3 duration1 duration2 duration3 half : ℕ} 
  (h1 : rate1 = 2) 
  (h2 : duration1 = 30)
  (h3 : rate2 = 2) 
  (h4 : duration2 = 30)
  (h5 : rate3 = 4) 
  (h6 : duration3 = 60)
  (h7 : half = 2) :
  let total_water := steady_flow rate1 duration1 + steady_flow rate2 duration2 + steady_flow rate3 duration3 in
  (total_water / half) = 18 :=
by
  sorry

end water_left_l728_728852


namespace baker_cakes_l728_728474

theorem baker_cakes (P x : ℝ) (h1 : P * x = 320) (h2 : 0.80 * P * (x + 2) = 320) : x = 8 :=
by
  sorry

end baker_cakes_l728_728474


namespace school_prize_problem_l728_728414

def unit_prices (x y: ℕ): Prop :=
  3 * x + 2 * y = 120 ∧
  5 * x + 4 * y = 210

def max_type_a_prizes (a: ℕ) (x y: ℕ): Prop :=
  (∃ a: ℕ, ∃ b: ℕ, a + b = 80 ∧
  0.8 * (x * a + y * b) ≤ 1500 ∧
  a = 45)

theorem school_prize_problem: 
  ∃ x y, unit_prices x y ∧ max_type_a_prizes 45 x y :=
by
  sorry

end school_prize_problem_l728_728414


namespace sqrt_720_eq_12_sqrt_5_l728_728000

theorem sqrt_720_eq_12_sqrt_5 : sqrt 720 = 12 * sqrt 5 :=
by
  sorry

end sqrt_720_eq_12_sqrt_5_l728_728000


namespace b1_value_l728_728041

axiom seq_b (b : ℕ → ℝ) : ∀ n, b 50 = 2 → 
  (∀ n ≥ 2, (∑ i in finset.range n, b (i + 1)) = n^3 * b n) → b 1 = 100

theorem b1_value (b : ℕ → ℝ) (h50 : b 50 = 2)
  (h : ∀ n ≥ 2, (∑ i in finset.range n, b (i + 1)) = n^3 * b n) : b 1 = 100 :=
sorry

end b1_value_l728_728041


namespace angle_equality_tangents_l728_728632

variable (ω : Circle) (O : Point) (ℓ : Line) (Y X A D B C : Point)

-- Definitions derived from the conditions
def tangent_at_Y (ℓ : Line) (ω : Circle) (Y : Point) : Prop := tangent ℓ ω Y
def on_left_of_Y (ℓ : Line) (Y X : Point) : Prop := on_line ℓ X ∧ between X Y ℓ.left
def tangent_perpendicular (ω : Circle) (ℓ : Line) (A D : Point) : Prop := 
  tangent ℓ ω D ∧ perpendicular (line_through A D) ℓ ∧ on_line ℓ A ∧ on_line (line_through A D) D
def equidistant (A X : Point) (B Y : Point) : Prop := distance A X = distance B Y
def tangent_circle (B C : Point) (ω : Circle) : Prop := tangent (line_through B C) ω C

-- Theorem to prove the desired angle equality
theorem angle_equality_tangents {ω : Circle} {O : Point} {ℓ : Line}
  {Y X A D B C : Point}
  (h1 : tangent_at_Y ℓ ω Y)
  (h2 : on_left_of_Y ℓ Y X)
  (h3 : tangent_perpendicular ω ℓ A D)
  (h4 : B ∈ ℓ ∧ B to_right_of Y)
  (h5 : equidistant A X B Y)
  (h6 : tangent_circle B C ω) :
  angle X D A = angle Y D C :=
sorry

end angle_equality_tangents_l728_728632


namespace sin_add_alpha_l728_728636

theorem sin_add_alpha (α : ℝ) (h : cos (π / 4 - α) = 1 / 3) : sin (π / 4 + α) = 1 / 3 :=
sorry

end sin_add_alpha_l728_728636


namespace probability_of_three_heads_in_eight_tosses_l728_728979

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l728_728979


namespace sequence_a500_l728_728742

theorem sequence_a500 (a : ℕ → ℤ)
  (h1 : a 1 = 2010)
  (h2 : a 2 = 2011)
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n) :
  a 500 = 2177 :=
sorry

end sequence_a500_l728_728742


namespace factorial_difference_l728_728554

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728554


namespace two_coins_heads_probability_l728_728072

theorem two_coins_heads_probability :
  let outcomes := [{fst := true, snd := true}, {fst := true, snd := false}, {fst := false, snd := true}, {fst := false, snd := false}],
      favorable := {fst := true, snd := true}
  in (favorable ∈ outcomes) → (favorable :: (List.erase outcomes favorable).length) / outcomes.length = 1 / 4 :=
by
  sorry

end two_coins_heads_probability_l728_728072


namespace count_arrangements_l728_728855

theorem count_arrangements : 
  ∃ n : ℕ, n = 4 ∧ 
  (∀ grid : Matrix (Fin 3) (Fin 3) Char,
    (∀ i : Fin 3, ∀ j : Fin 3,
      grid 0 0 = 'A' ∧
      (∀ k : Fin 3, ∃! l : Fin 3, grid l k = grid k l) ∧
      (∃ rowA : List (Fin 3 × Fin 3),
        ∀ p ∈ rowA, grid p.1 p.2 = 'A') ∧
      (∃ rowB : List (Fin 3 × Fin 3),
        ∀ p ∈ rowB, grid p.1 p.2 = 'B') ∧
      (∃ rowC : List (Fin 3 × Fin 3),
        ∀ p ∈ rowC, grid p.1 p.2 = 'C') ∧
      (∀ i : Fin 3, List.countp (fun p => p.1 = i) (rowA ++ rowB ++ rowC) = 1) ∧
      (∀ j : Fin 3, List.countp (fun p => p.2 = j) (rowA ++ rowB ++ rowC) = 1)) ∧ 
    List.length (rowA ++ rowB ++ rowC) = 9) :=
  sorry

end count_arrangements_l728_728855


namespace correct_options_l728_728881

noncomputable def f1 (x : ℝ) : ℝ := (x^2 + 5) / (Real.sqrt (x^2 + 4))
noncomputable def f2 (x : ℝ) : ℝ := (2*x - 3) / (x - 1)
noncomputable def f3 (x : ℝ) : ℝ := Real.sqrt (x - 1) * Real.sqrt (x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)

theorem correct_options :
  (∀ x, f1 x ≠ 2) ∧
  (∀ a b m, (a > b ∧ b > 0 ∧ m > 0) → (b / a < (b + m) / (a + m))) ∧
  (∀ y, (∃ x, f2 x = y) ↔ (y ∈ Set.Ioo (-∞) 2 ∪ Set.Ioo 2 ∞)) ∧
  (∀ x, f3 x ≠ g x) :=
by
  sorry

end correct_options_l728_728881


namespace weekly_caloric_allowance_l728_728137

-- Define the given conditions
def average_daily_allowance : ℕ := 2000
def daily_reduction_goal : ℕ := 500
def intense_workout_extra_calories : ℕ := 300
def moderate_exercise_extra_calories : ℕ := 200
def days_intense_workout : ℕ := 2
def days_moderate_exercise : ℕ := 3
def days_rest : ℕ := 2

-- Lean statement to prove the total weekly caloric intake
theorem weekly_caloric_allowance :
  (days_intense_workout * (average_daily_allowance - daily_reduction_goal + intense_workout_extra_calories)) +
  (days_moderate_exercise * (average_daily_allowance - daily_reduction_goal + moderate_exercise_extra_calories)) +
  (days_rest * (average_daily_allowance - daily_reduction_goal)) = 11700 := by
  sorry

end weekly_caloric_allowance_l728_728137


namespace probability_three_heads_in_eight_tosses_l728_728993

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l728_728993


namespace ratio_of_areas_l728_728398

theorem ratio_of_areas (s : ℝ) :
  let small_triangle_area := (sqrt 3 / 4) * s ^ 2 in
  let large_triangle_area := (sqrt 3 / 4) * (6 * s) ^ 2 in
  (6 * small_triangle_area) / large_triangle_area = 1 / 6 :=
by
  let small_triangle_area := (sqrt 3 / 4) * s ^ 2
  let large_triangle_area := (sqrt 3 / 4) * (6 * s) ^ 2
  sorry

end ratio_of_areas_l728_728398


namespace factors_2310_l728_728678

theorem factors_2310 : ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ S.card = 5 ∧ (2310 = S.prod id) :=
by
  sorry

end factors_2310_l728_728678


namespace prism_volume_l728_728608

noncomputable def volume_of_inclined_prism (a : ℝ) : ℝ :=
  let base_area := (sqrt 3 / 4) * a^2
  let height := (a * (sqrt 3) / 2)
  base_area * height

theorem prism_volume (a : ℝ) : volume_of_inclined_prism a = (3 * a^3) / 8 := by
  sorry

end prism_volume_l728_728608


namespace shuttlecock_weight_probability_l728_728814

variable (p_lt_4_8 : ℝ) -- Probability that its weight is less than 4.8 g
variable (p_le_4_85 : ℝ) -- Probability that its weight is not greater than 4.85 g

theorem shuttlecock_weight_probability (h1 : p_lt_4_8 = 0.3) (h2 : p_le_4_85 = 0.32) :
  p_le_4_85 - p_lt_4_8 = 0.02 :=
by
  sorry

end shuttlecock_weight_probability_l728_728814


namespace find_a_l728_728713

theorem find_a (a : ℚ) : (∃ b : ℚ, 4 * (x : ℚ)^2 + 14 * x + a = (2 * x + b)^2) → a = 49 / 4 :=
by
  sorry

end find_a_l728_728713


namespace deductive_syllogism_correct_l728_728446

-- Definitions to reflect the premises
def non_repeating_infinite_decimals : Type := { x : ℝ // ∀ n m : ℕ, n ≠ m → (x.nth n ≠ x.nth m) }
def irrational_numbers : Type := { x : ℝ // x ∉ ℚ }

-- The proof problem
theorem deductive_syllogism_correct :
  (∀ x : non_repeating_infinite_decimals, x.val ∈ irrational_numbers) →
  (π ∈ non_repeating_infinite_decimals) →
  (π ∉ ℚ) :=
by
  sorry

end deductive_syllogism_correct_l728_728446


namespace no_other_pair_l728_728463

noncomputable def f : ℝ → ℝ := sorry  -- Define the polynomial f(x)

variables (a b : ℝ)

-- Conditions
axiom h1 : f(a) = b
axiom h2 : f(b) = a
axiom h3 : a ≠ b
axiom h4 : ∀ x, (f(x) - x)^2 - (2 * f(x) - 1)^2 + 1 = 0  -- Expressing that f(x) is quadratic

theorem no_other_pair (c d : ℝ) : (f(c) = d ∧ f(d) = c) → (c = a ∨ c = b) ∧ (d = a ∨ d = b) := sorry

end no_other_pair_l728_728463


namespace order_abc_l728_728238

-- Define the constants a, b, c
def a : ℝ := (1 / 2) ^ (1 / 2)
def b : ℝ := (1 / 3) ^ (1 / 2)
def c : ℝ := Real.log (Real.cbrt Real.e) / Real.log Real.pi

-- The theorem to prove the ordering
theorem order_abc : c < b ∧ b < a := by
  sorry

end order_abc_l728_728238


namespace constant_two_l728_728360

theorem constant_two (p : ℕ) (h_prime : Nat.Prime p) (h_gt_two : p > 2) (c : ℕ) (n : ℕ) (h_n : n = c * p) (h_even_divisors : ∀ d : ℕ, d ∣ n → (d % 2 = 0) → d = 2) : c = 2 := by
  sorry

end constant_two_l728_728360


namespace coin_toss_probability_l728_728945

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l728_728945


namespace probability_three_heads_in_eight_tosses_l728_728999

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l728_728999


namespace factorial_subtraction_l728_728558

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_subtraction : factorial 10 - factorial 9 = 3265920 := by
  sorry

end factorial_subtraction_l728_728558


namespace determine_p_q_l728_728583

theorem determine_p_q:
  ∃ (p q : ℚ), 24^3 = (16^2 / 4) * 2^(6 * p) * 3^(3 * q) ∧ p = 1 / 2 ∧ q = 1 :=
by
  sorry

end determine_p_q_l728_728583


namespace length_of_AB_l728_728299

theorem length_of_AB (k : ℝ) (x1 x2 : ℝ) (y1 y2 : ℝ) :
  (∀ x y, y^2 = 8 * x ↔ y = k * x - 2) →
  (x1 + x2) / 2 = 2 →
  x1 ≠ x2 →
  k = 2 →
  (x1 = x2 - 4/k^2) →
  (x1 + x2 = 4) →
  (y1 = k * x1 - 2) →
  (y2 = k * x2 - 2) →
  ( ∃ y, y^2 = 8 * x1 ∧ ∃ y, y^2 = 8 * x2) →
  abs (y2 - y1) = 2 * sqrt(15) :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end length_of_AB_l728_728299


namespace angle_x_value_l728_728376

def is_center_of_circle (O : Point) (circle : Circle) : Prop :=
  circle.center = O

def is_isosceles (triangle : Triangle) : Prop :=
  (triangle.A = triangle.B) ∨ (triangle.B = triangle.C) ∨ (triangle.A = triangle.C)

def angle_BCO := 32

def angle_AO_isosceles :=
  ∀ (O A B C : Point), 
    is_center_of_circle O (circumcircle O A B C) → 
    is_isosceles (triangle O B C) →
    is_isosceles (triangle O A C) →
    x = 9

theorem angle_x_value :
  ∀ (O A B C : Point),
  is_center_of_circle O (circumcircle O A B C) →
  measure (angle B C O) = 32 ∧  
  (is_isosceles (triangle O B C)) ∧ 
  (is_isosceles (triangle O A C)) → 
  x = 9 := by
  sorry

end angle_x_value_l728_728376


namespace tenth_ordered_permutation_is_4682_l728_728840

-- Definitions of the problem conditions.
def is_valid_permutation (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 : ℕ), 
    n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧
    {d1, d2, d3, d4} = {2, 4, 6, 8}

-- The main theorem to be proved.
theorem tenth_ordered_permutation_is_4682 : 
  ∃ (l : List ℕ), (∀ n ∈ l, is_valid_permutation n) ∧ l.length = 24 ∧ l.nth 9 = some 4682 :=
sorry

end tenth_ordered_permutation_is_4682_l728_728840


namespace geometric_sequence_S28_l728_728220

noncomputable def geom_sequence_sum (S : ℕ → ℝ) (a : ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, S (n * (n + 1) / 2) = a * (1 - r^n) / (1 - r)

theorem geometric_sequence_S28 {S : ℕ → ℝ} (a r : ℝ)
  (h1 : geom_sequence_sum S a r)
  (h2 : S 14 = 3)
  (h3 : 3 * S 7 = 3) :
  S 28 = 15 :=
by
  sorry

end geometric_sequence_S28_l728_728220


namespace min_sqrt_sum_of_squares_eq_sqrt_3_l728_728724

theorem min_sqrt_sum_of_squares_eq_sqrt_3 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) : sqrt (a^2 + b^2 + c^2) = sqrt 3 :=
sorry

end min_sqrt_sum_of_squares_eq_sqrt_3_l728_728724


namespace find_angle_B_find_perimeter_l728_728763

-- Definitions and conditions
variable {A B C a b c : ℝ}
variable (h1 : cos B + sin ((A + C) / 2) = 0)
variable (h2 : a / c = 3 / 5)
variable (h3 : (3 * b * Real.sqrt 3) / (2 * a * c) = 15 * Real.sqrt 3 / 14)

-- Proof problems translated as Lean statements
theorem find_angle_B (h1 : cos B + sin ((A + C) / 2) = 0) : B = 2 * Real.pi / 3 :=
sorry

theorem find_perimeter (h2 : a / c = 3 / 5) (h3 : (3 * b * Real.sqrt 3) / (2 * a * c) = 15 * Real.sqrt 3 / 14)
  (hB : B = 2 * Real.pi / 3) : a + b + c = 15 :=
sorry

end find_angle_B_find_perimeter_l728_728763


namespace probability_exactly_three_heads_l728_728966
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l728_728966


namespace factorial_difference_l728_728546

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem factorial_difference : fact 10 - fact 9 = 3265920 := by
  have h1 : fact 9 = 362880 := sorry
  have h2 : fact 10 = 10 * fact 9 := by rw [fact]
  calc
    fact 10 - fact 9
        = 10 * fact 9 - fact 9 := by rw [h2]
    ... = 9 * fact 9 := by ring
    ... = 9 * 362880 := by rw [h1]
    ... = 3265920 := by norm_num

end factorial_difference_l728_728546


namespace max_parts_divided_by_three_planes_l728_728856

theorem max_parts_divided_by_three_planes (P1 P2 P3 : Plane) (h1: P1 ≠ P2) (h2: P1 ≠ P3) (h3: P2 ≠ P3):
  divides_space_in_parts P1 P2 P3 = 8 :=
sorry

end max_parts_divided_by_three_planes_l728_728856


namespace trig_expression_simplify_l728_728697

theorem trig_expression_simplify (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trig_expression_simplify_l728_728697


namespace totalTaxIsCorrect_l728_728459

-- Define the different income sources
def dividends : ℝ := 50000
def couponIncomeOFZ : ℝ := 40000
def couponIncomeCorporate : ℝ := 30000
def capitalGain : ℝ := (100 * 200) - (100 * 150)

-- Define the tax rates
def taxRateDividends : ℝ := 0.13
def taxRateCorporateBond : ℝ := 0.13
def taxRateCapitalGain : ℝ := 0.13

-- Calculate the tax for each type of income
def taxOnDividends : ℝ := dividends * taxRateDividends
def taxOnCorporateCoupon : ℝ := couponIncomeCorporate * taxRateCorporateBond
def taxOnCapitalGain : ℝ := capitalGain * taxRateCapitalGain

-- Sum of all tax amounts
def totalTax : ℝ := taxOnDividends + taxOnCorporateCoupon + taxOnCapitalGain

-- Prove that total tax equals the calculated figure
theorem totalTaxIsCorrect : totalTax = 11050 := by
  sorry

end totalTaxIsCorrect_l728_728459


namespace sum_difference_even_odd_3000_l728_728869

theorem sum_difference_even_odd_3000:
  let O := (List.range 3000).map (λ k, 2 * k + 1) in
  let E := (List.range 3000).map (λ k, 2 * (k + 1)) in
  (E.sum - O.sum) = 3000 :=
by
  sorry

end sum_difference_even_odd_3000_l728_728869


namespace probability_three_heads_in_eight_tosses_l728_728998

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l728_728998


namespace combination_seven_four_l728_728147

theorem combination_seven_four : nat.choose 7 4 = 35 :=
by
  sorry

end combination_seven_four_l728_728147


namespace remainder_of_sum_mod_18_l728_728603

theorem remainder_of_sum_mod_18 :
  let nums := [85, 86, 87, 88, 89, 90, 91, 92, 93]
  let sum_nums := nums.sum
  let product := 90 * sum_nums
  product % 18 = 10 :=
by
  sorry

end remainder_of_sum_mod_18_l728_728603


namespace area_of_garden_l728_728586

-- Define the garden properties
variables {l w : ℕ}

-- Calculate length from the condition of walking length 30 times
def length_of_garden (total_distance : ℕ) (times : ℕ) := total_distance / times

-- Calculate perimeter from the condition of walking perimeter 12 times
def perimeter_of_garden (total_distance : ℕ) (times : ℕ) := total_distance / times

-- Define the proof statement
theorem area_of_garden (total_distance : ℕ) (times_length_walk : ℕ) (times_perimeter_walk : ℕ)
  (h1 : length_of_garden total_distance times_length_walk = l)
  (h2 : perimeter_of_garden total_distance (2 * times_perimeter_walk) = 2 * (l + w)) :
  l * w = 400 := 
sorry

end area_of_garden_l728_728586


namespace coin_toss_probability_l728_728949

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l728_728949


namespace probability_of_exactly_three_heads_l728_728926

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l728_728926


namespace probability_of_three_heads_in_eight_tosses_l728_728972

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l728_728972


namespace allocation_schemes_4_teachers_3_schools_l728_728136

theorem allocation_schemes_4_teachers_3_schools : 
  ∃ (assignments : Set (Finset (Fin (4)))), 
  (∀ (assignment_group : Finset (Fin (4))), assignment_group ∈ assignments → assignment_group.card = 1 ∨ assignment_group.card = 2) ∧ 
  assignments.card = 36 := 
sorry

end allocation_schemes_4_teachers_3_schools_l728_728136


namespace polar_to_rectangular_l728_728036

def given_polar_coordinates (r θ : ℝ) : Prop :=
  (r = 2 ∧ θ = -π / 6)

def rectangular_coordinates (x y : ℝ) : Prop :=
  (x, y) = (Real.cos (-π / 6) * 2, Real.sin (-π / 6) * 2)

theorem polar_to_rectangular :
  given_polar_coordinates 2 (-π / 6) →
  rectangular_coordinates (Real.cos (-π / 6) * 2) (Real.sin (-π / 6) * 2) :=
by
  intro h
  simp [given_polar_coordinates, rectangular_coordinates, Real.cos, Real.sin]
  split
  calc
    2 * Real.cos (-π / 6) = 2 * (√3 / 2) := by sorry,
    ... = √3 := by sorry,
  calc
    2 * Real.sin (-π / 6) = 2 * (-1 / 2) := by sorry,
    ... = -1 := by sorry

end polar_to_rectangular_l728_728036


namespace solution_set_of_x_squared_lt_one_l728_728606

theorem solution_set_of_x_squared_lt_one : {x : ℝ | x^2 < 1} = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end solution_set_of_x_squared_lt_one_l728_728606


namespace greatest_k_inequality_l728_728213

theorem greatest_k_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ( ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → 
    (a / b + b / c + c / a - 3) ≥ k * (a / (b + c) + b / (c + a) + c / (a + b) - 3 / 2) ) ↔ k = 1 := 
sorry

end greatest_k_inequality_l728_728213


namespace minimum_value_g_f_zero_point_existence_l728_728659

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  Real.exp x - a * x ^ 2 - b * x - 1

noncomputable def g (x : ℝ) (a b : ℝ) : ℝ :=
  Real.exp x - 2 * a * x - b

theorem minimum_value_g (a b : ℝ) :
  (a ≤ 1/2 → ∀ x ∈ set.Icc 0 1, g x a b ≥ g 0 a b) ∧
  (a ≥ Real.exp 1 / 2 → ∀ x ∈ set.Icc 0 1, g x a b ≥ g 1 a b) ∧
  (1/2 < a ∧ a < Real.exp 1 / 2 → ∀ x ∈ set.Icc 0 1, g x a b ≥ g (Real.log (2 * a)) a b) := sorry

theorem f_zero_point_existence (a b : ℝ) (h : f 1 a b = 0) :
  (∃ x ∈ set.Ioo 0 1, f x a b = 0) ∧ e - 2 < a ∧ a < 1 := sorry

end minimum_value_g_f_zero_point_existence_l728_728659


namespace probability_three_heads_in_eight_tosses_l728_728941

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l728_728941


namespace odot_commutative_odot_one_four_l728_728613

section ProofProblem

variables (a b : ℚ)

def odot (a b : ℚ) : ℚ := a - a * b + b + 3

theorem odot_commutative : ∀ (a b : ℚ), odot a b = odot b a :=
by {
  -- This is the mathematical statement that odot is commutative
  intros,
  sorry
}

theorem odot_one_four : odot 1 4 = 4 :=
by {
  -- This is the mathematical statement that odot 1 4 equals 4
  sorry
}

end ProofProblem

end odot_commutative_odot_one_four_l728_728613


namespace find_annual_interest_rate_l728_728604

variable (P : ℕ) (SI : ℕ) (T : ℚ) (R : ℚ)

-- Definition of the variables given in the condition
noncomputable def Principal : ℕ := 69000
noncomputable def SimpleInterest : ℕ := 8625
noncomputable def TimePeriod : ℚ := 9 / 12

-- Main theorem
theorem find_annual_interest_rate (P = Principal) (SI = Simple_interest) (T = TimePeriod) :
  R = SI / (P * T) ↔ R ≈ 0.1667 := 
by
  sorry

end find_annual_interest_rate_l728_728604


namespace find_angle_A_find_b_c_range_l728_728324

-- Part (a) proving angle A
theorem find_angle_A (A : ℝ) (h : sqrt 3 * cos (2 * A) + 1 = 4 * sin (π / 6 + A) * sin (π / 3 - A)) :
  A = π / 4 :=
sorry

-- Part (b) find the range of values for sqrt(2)b - c given conditions
theorem find_b_c_range (a b c B : ℝ) 
  (hA : ∀ A : ℝ, sqrt 3 * cos (2 * A) + 1 = 4 * sin (π / 6 + A) * sin (π / 3 - A) → A = π / 4)
  (ha : a = sqrt 2)
  (hb : b ≥ a)
  (law_of_sines : ∀ (A B C : ℝ), b / sin B = c / sin C)
  (relation_bc : b = 2 * sin B) (c_relation : c = 2 * sin (π / 2 - B)) :
  sqrt 2 * b - c ∈ set.Ico 0 2 :=
sorry

end find_angle_A_find_b_c_range_l728_728324


namespace unique_perimeter_for_quadrilateral_with_given_properties_l728_728614

theorem unique_perimeter_for_quadrilateral_with_given_properties :
  ∀ (p : ℕ), p < 2015 → 
  (∃ (A B C D : ℝ) (integer_sides : ℝ → Prop),
    (integer_sides A) ∧ (integer_sides B) ∧ (integer_sides C) ∧ (integer_sides D) ∧
    (A + B + C + D = p) ∧
    (B = 3) ∧
    (A = D) ∧
    (A^2 + 3^2 = C^2) ∧
    (quadrilateral_shape A B C D)
  ) →
  (unique (λ (p : ℕ), p < 2015 ∧
    (∃ (A B C D : ℝ),
      (A + B + C + D = p) ∧
      (B = 3) ∧
      (A = D) ∧
      (A^2 + 3^2 = C^2) ∧
      (quadrilateral_shape A B C D))
    ) ) := by
      sorry

noncomputable def integer_sides (x : ℝ) : Prop -> Bool := sorry

noncomputable def quadrilateral_shape (A B C D : ℕ) : Prop := sorry

end unique_perimeter_for_quadrilateral_with_given_properties_l728_728614


namespace rearrange_2021_l728_728674

theorem rearrange_2021 : 
  let digits := [2, 0, 2, 1] in
  let all_permutations := (Finset.univ : Finset (List ℕ)).val.filter (λ l, Multiset.card (l.toMultiset - digits.toMultiset) = 0) in
  let valid_permutations := all_permutations.filter (λ l, l.head! ≠ 0 ∧ l.last! ≠ 0) in
  valid_permutations.card = 6 := sorry

end rearrange_2021_l728_728674


namespace geom_seq_sum_l728_728660

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / ((Real.exp x) + 1)

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r

def geometric_conditions (a : ℕ → ℝ) : Prop :=
  geometric_sequence a ∧
  (∀ n : ℕ, a n > 0) ∧
  a 1009 = 1

theorem geom_seq_sum (a : ℕ → ℝ) (h : geometric_conditions a) :
  ∑ k in Finset.range 2017, f (Real.log (a (k+1))) = 2017 / 2 := 
sorry

end geom_seq_sum_l728_728660


namespace ramesh_profit_percentage_is_correct_l728_728379

-- Definitions for conditions
def purchase_price : ℝ := 13500
def discount_rate : ℝ := 0.20
def transport_cost : ℝ := 125
def installation_cost : ℝ := 250
def selling_price : ℝ := 18975

-- Calculate labelled price
def labelled_price (purchase_price : ℝ) (discount_rate : ℝ) : ℝ :=
  purchase_price / (1 - discount_rate)

-- Calculate cost price
def cost_price (purchase_price transport_cost installation_cost : ℝ) : ℝ :=
  purchase_price + transport_cost + installation_cost

-- Calculate profit
def profit (sp cp : ℝ) : ℝ := sp - cp

-- Calculate profit percentage
def profit_percentage (profit cp : ℝ) : ℝ := (profit / cp) * 100

-- Final statement to prove
theorem ramesh_profit_percentage_is_correct :
  profit_percentage (profit selling_price (cost_price purchase_price transport_cost installation_cost)) (cost_price purchase_price transport_cost installation_cost) = 36.73 := by
  sorry

end ramesh_profit_percentage_is_correct_l728_728379


namespace problem_statement_l728_728764

theorem problem_statement (A B C : Real)
  (h1 : A + B + C = 180)
  (h2 : C > 90) : cos B > sin A := by
  sorry

end problem_statement_l728_728764


namespace value_of_f_at_minus_point_two_l728_728450

noncomputable def f (x : ℝ) : ℝ := 1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem value_of_f_at_minus_point_two : f (-0.2) = 0.81873 :=
by {
  sorry
}

end value_of_f_at_minus_point_two_l728_728450


namespace arithmetic_sequence_proof_geometric_sequence_proof_l728_728629

-- Definitions based on the conditions
def sum_of_first_three_terms (a1 a2 a3 : ℤ) : Prop :=
  a1 + a2 + a3 = -3

def product_of_first_three_terms (a1 a2 a3 : ℤ) : Prop :=
  a1 * a2 * a3 = 8

def is_arithmetic_sequence (a1 a2 a3 : ℤ) (d : ℤ) : Prop :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d

def is_geometric_sequence (a2 a3 a1 : ℤ) : Prop :=
  a2 * a2 = a3 * a1

def general_formula (a : ℕ → ℤ) : Prop :=
  (∀ n, a n = -3 * n + 5) ∨ (∀ n, a n = 3 * n - 7)

-- Definitions for the sequence of the absolute values and the sum
def abs_sequence (a : ℕ → ℤ) : ℕ → ℤ
| 0     => abs (a 0)
| (n+1) => abs (a (n + 1))

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℕ → ℤ
| 0     => abs_sequence a 0
| 1     => 4
| k+2 => (3 * (k + 2) - 7) + sum_first_n_terms a (k + 1)

def expected_sum_first_n_terms (n : ℕ) : ℤ :=
if n = 1 then 4 else (3 * n * n - 11 * n + 20) / 2

-- The proof statement
theorem arithmetic_sequence_proof :
  ∀ (a1 a2 a3 : ℤ) (d : ℤ) (a : ℕ → ℤ),
    sum_of_first_three_terms a1 a2 a3 →
    product_of_first_three_terms a1 a2 a3 →
    is_arithmetic_sequence a1 a2 a3 d →
    (∀ n, a n = a1 + (n - 1) * d) →
    general_formula a :=
sorry

theorem geometric_sequence_proof :
  ∀ (a1 a2 a3 : ℤ) (d : ℤ) (a : ℕ → ℤ),
    sum_of_first_three_terms a1 a2 a3 →
    product_of_first_three_terms a1 a2 a3 →
    is_arithmetic_sequence a1 a2 a3 d →
    is_geometric_sequence a1 a2 a3 →
    (∀ n, a n = a1 + (n - 1) * d) →
    (∀ n, sum_first_n_terms a n = expected_sum_first_n_terms n) :=
sorry

end arithmetic_sequence_proof_geometric_sequence_proof_l728_728629


namespace complement_of_A_in_R_intersection_of_C_R_B_and_A_l728_728364

noncomputable def A : set ℝ := { x | 2 ≤ 2^(2-x) ∧ 2^(2-x) < 8 }
noncomputable def B : set ℝ := { x | x < 0 }
noncomputable def R : set ℝ := set.univ

-- Problem 1: The complement of A in R
theorem complement_of_A_in_R :
  { x : ℝ | x ≤ -1 ∨ x > 1 } = (R \ A) :=
by sorry

-- Problem 2: The intersection of C_R B and A
theorem intersection_of_C_R_B_and_A :
  { x : ℝ | 0 ≤ x ∧ x ≤ 1 } = ((R \ B) ∩ A) :=
by sorry

end complement_of_A_in_R_intersection_of_C_R_B_and_A_l728_728364


namespace value_of_a_monotonicity_on_interval_l728_728264

noncomputable def f (a x : ℝ) : ℝ := (x + a) / (x^2 + 2)

theorem value_of_a (a : ℝ) (h : ∀ x : ℝ, f a (-x) = - f a x) : a = 0 := sorry

noncomputable def g (x : ℝ) : ℝ := x / (x^2 + 2)

theorem monotonicity_on_interval : ∀ x : ℝ, 0 < x ∧ x ≤ sqrt 2 → 0 < (deriv g x) := sorry

end value_of_a_monotonicity_on_interval_l728_728264


namespace factorial_subtraction_l728_728562

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_subtraction : factorial 10 - factorial 9 = 3265920 := by
  sorry

end factorial_subtraction_l728_728562


namespace probability_two_randomly_chosen_diagonals_intersect_l728_728310

def convex_hexagon_diagonals := 9
def main_diagonals := 3
def secondary_diagonals := 6
def intersections_from_main_diagonals := 3 * 4
def intersections_from_secondary_diagonals := 6 * 3
def total_unique_intersections := (intersections_from_main_diagonals + intersections_from_secondary_diagonals) / 2
def total_diagonal_pairs := convex_hexagon_diagonals * (convex_hexagon_diagonals - 1) / 2

theorem probability_two_randomly_chosen_diagonals_intersect :
  (total_unique_intersections / total_diagonal_pairs : ℚ) = 5 / 12 := sorry

end probability_two_randomly_chosen_diagonals_intersect_l728_728310


namespace factorial_subtraction_l728_728561

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_subtraction : factorial 10 - factorial 9 = 3265920 := by
  sorry

end factorial_subtraction_l728_728561


namespace dog_food_total_l728_728328

-- Variables and constants
constant cups_morning : ℕ := 1
constant cups_evening : ℕ := 1
constant days : ℕ := 16

-- Total cups of dog food in the bag
theorem dog_food_total : (cups_morning + cups_evening) * days = 32 :=
by
  -- Proof to be added
  sorry

end dog_food_total_l728_728328


namespace pages_read_in_a_year_l728_728716

-- Definition of the problem conditions
def novels_per_month := 4
def pages_per_novel := 200
def months_per_year := 12

-- Theorem statement corresponding to the problem
theorem pages_read_in_a_year (h1 : novels_per_month = 4) (h2 : pages_per_novel = 200) (h3 : months_per_year = 12) : 
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by
  sorry

end pages_read_in_a_year_l728_728716


namespace f_2_equals_12_l728_728649

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x^3 + x^2 else - (2 * (-x)^3 + (-x)^2)

theorem f_2_equals_12 : f 2 = 12 := by
  sorry

end f_2_equals_12_l728_728649


namespace mn_passes_through_incenter_l728_728749

noncomputable theory
open_locale classical

-- Define the geometric objects and the proof problem in Lean 4

-- Assuming an acute triangle ABC
variables {A B C : Point} (acute_triangle : acute A B C)

-- Define the incenter I of triangle ABC
def incenter (A B C : Point) : Point := sorry -- Formal definition needed

-- Assuming we have the bisectors intersecting the circumcircle again at A1, B1, and C1 respectively
variables {A1 B1 C1 : Point}
(HA1 : bisector A (angle_at A) = A1)
(HB1 : bisector B (angle_at B) = B1)
(HC1 : bisector C (angle_at C) = C1)

-- Point M is the intersection of lines AB and B1C1
def M (A B B1 C1 : Point) : Point := sorry -- Formal definition needed

-- Point N is the intersection of lines BC and A1B1
def N (B C A1 B1 : Point) : Point := sorry -- Formal definition needed

-- Statement of the theorem
theorem mn_passes_through_incenter (acute_triangle: acute A B C)
  (HA1 : bisector A (angle_at A) = A1)
  (HB1 : bisector B (angle_at B) = B1)
  (HC1 : bisector C (angle_at C) = C1)
  (M_def : M A B B1 C1)
  (N_def : N B C A1 B1) :
  passes_through (line_through M_def N_def) (incenter A B C) :=
sorry

end mn_passes_through_incenter_l728_728749


namespace area_of_PQR_l728_728746

-- Define the elements and conditions of the problem.
variables {P Q R : Type} [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables (PQ : ℝ) (QR : ℝ) (PR : ℝ)
variables (angleQ angleR : ℝ) (area : ℝ)

-- The given data
def is_right_triangle (P Q R : P) : Prop := ∃ right_angle : ℝ, right_angle = 90
def is_angle_equal (angleQ angleR : ℝ) : Prop := angleQ = angleR
def hypotenuse_length (PR : ℝ) : Prop := PR = 6 * real.sqrt 2
def area_of_triangle (PQ QR : ℝ) (area : ℝ) : Prop := area = 1/2 * PQ * QR

-- The proof statement
theorem area_of_PQR {P Q R : ℝ} (h1 : is_right_triangle P Q R) 
                    (h2 : is_angle_equal angleQ angleR)
                    (h3 : hypotenuse_length PR):
                    area_of_triangle PQ QR 36 := by 
  sorry

end area_of_PQR_l728_728746


namespace binom_sum_equals_fibonacci_l728_728462

-- Definitions for binomial coefficients and Fibonacci numbers

def binom (n k : ℕ) : ℕ := nat.choose n k

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- The statement to prove
theorem binom_sum_equals_fibonacci (n : ℕ) : 
  (∑ i in Finset.range (n + 1), binom (n - i) i) = fibonacci (n + 1) :=
sorry

end binom_sum_equals_fibonacci_l728_728462


namespace imaginary_part_correct_conjugate_correct_l728_728402

-- Define the complex number in question
def complex_number : ℂ := (1 - real.sqrt 3 * complex.I) / (real.sqrt 3 - complex.I)

-- The statement to prove the imaginary part of the complex number
theorem imaginary_part_correct : complex.im complex_number = -1 / 2 :=
by
  sorry

-- The statement to prove the conjugate of the complex number
theorem conjugate_correct : complex.conj complex_number = (real.sqrt 3) / 2 + (1 / 2) * complex.I :=
by
  sorry

end imaginary_part_correct_conjugate_correct_l728_728402


namespace ten_factorial_minus_nine_factorial_l728_728569

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem ten_factorial_minus_nine_factorial :
  factorial 10 - factorial 9 = 3265920 := 
by 
  sorry

end ten_factorial_minus_nine_factorial_l728_728569


namespace smallest_mul_next_smallest_in_set_l728_728048

theorem smallest_mul_next_smallest_in_set :
  (∀ S : Set ℤ, S = {10, 11, 12, 13, 14} → let s1 := 10; let s2 := 11 in s1 * s2 = 110) :=
by
  intros S hS
  simp
  exact rfl

end smallest_mul_next_smallest_in_set_l728_728048


namespace sqrt_720_simplified_l728_728006

theorem sqrt_720_simplified : Real.sqrt 720 = 6 * Real.sqrt 5 := sorry

end sqrt_720_simplified_l728_728006


namespace polynomial_remainder_division_l728_728219

theorem polynomial_remainder_division :
  ∀ (x : ℝ), (3 * x^7 + 2 * x^5 - 5 * x^3 + x^2 - 9) % (x^2 + 2 * x + 1) = 14 * x - 16 :=
by
  sorry

end polynomial_remainder_division_l728_728219


namespace minimum_real_roots_l728_728785

noncomputable def g (x : ℝ) : Polynomial ℝ := sorry
def s : Fin 3010 → ℂ := sorry

theorem minimum_real_roots :
  ∃ g : Polynomial ℝ,
  g.degree = 3010 ∧
  ∀ i, g.root (s i) ∧
  ∃ n, (n < 4) ∧
  (∃ distinct_abs_values : Fin 1505 → ℂ,
  ∀ i, ∃ j, |s (nat_of_fin i)| = |distinct_abs_values j|)

end minimum_real_roots_l728_728785


namespace binary_to_decimal_l728_728188

theorem binary_to_decimal (b : List ℕ) (h : b = [1, 1, 0, 0, 1, 0, 1]) : 
  (List.foldr (λ (b n : ℕ), b + 2*n) 0 b) = 101 := 
by 
  -- Generates binary 1100101's decimal number
  sorry

end binary_to_decimal_l728_728188


namespace coin_toss_probability_l728_728942

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l728_728942


namespace sign_choice_sum_zero_l728_728357

theorem sign_choice_sum_zero
  (n : ℕ) (hn : n ≥ 1)
  (a : Fin n → ℕ) (h_pos : ∀ i, 0 < a i) (h_bound : ∀ i, a i ≤ i + 1) (h_even_sum : (∑ i in Finset.range n, a i) % 2 = 0)
  : ∃ (b : Fin n → ℤ), (∑ i in Finset.range n, b i) = 0 ∧ (∀ i, b i = a i ∨ b i = -a i) :=
begin
  sorry
end

end sign_choice_sum_zero_l728_728357


namespace amanda_graph_is_quadratic_l728_728504

theorem amanda_graph_is_quadratic :
  ∀ r : ℕ, (r ∈ [2, 4, 6, 8, 10]) → 
  let C := 2 * Real.pi * r in
  let A := Real.pi * r^2 in
  (A / Real.pi) = (C / (2 * Real.pi))^2 :=
by
  intros r hr
  have hC : C = 2 * Real.pi * r := rfl
  have hA : A = Real.pi * r^2 := rfl
  rw [hC, hA, Real.div_eq_iff_mul_eq]
  ring
  norm_num
  sorry

end amanda_graph_is_quadratic_l728_728504


namespace combination_seven_four_l728_728146

theorem combination_seven_four : nat.choose 7 4 = 35 :=
by
  sorry

end combination_seven_four_l728_728146


namespace parallelogram_area_l728_728827

variables (a b : ℝ^3) -- Assuming a and b are 3D real vectors
variable (h : ∥a × b∥ = 10)

theorem parallelogram_area :
  ∥(3 • a + 2 • b) × (2 • a - 4 • b)∥ = 40 :=
by
  sorry

end parallelogram_area_l728_728827


namespace prism_volume_l728_728829

theorem prism_volume (a b : ℝ) (h : 0 < a ∧ b ≠ 0) : 
  let base_area := (sqrt 3 / 4) * a^2
  let height := sqrt (4 * a^2 - b^2)
  let volume := base_area * height
  volume = (ab * sqrt (12 * a^2 - 3 * b^2)) / 8 :=
by
  sorry

end prism_volume_l728_728829


namespace pentagon_perpendicular_sums_l728_728103

noncomputable def FO := 2
noncomputable def FQ := 2
noncomputable def FR := 2

theorem pentagon_perpendicular_sums :
  FO + FQ + FR = 6 :=
by
  sorry

end pentagon_perpendicular_sums_l728_728103


namespace number_of_boundaries_is_three_l728_728903

variable (T : ℕ) (Rs Rb : ℕ) (S B : ℕ)

def runs_by_running : ℕ := (T / 2)

def runs_from_sixes : ℕ := (S * Rs)

def runs_from_boundaries : ℕ := (T - runs_by_running T - runs_from_sixes S Rs)

def number_of_boundaries : ℕ := (runs_from_boundaries T Rs (runs_by_running T) (runs_from_sixes S Rs) / Rb)

theorem number_of_boundaries_is_three 
  (hT : T = 120) 
  (hRs : Rs = 6) 
  (hRb : Rb = 4) 
  (hS : S = 8) : B = 3 :=
by
  rw [←hT, ←hRs, ←hRb, ←hS] 
  sorry

end number_of_boundaries_is_three_l728_728903


namespace find_reflex_angle_l728_728232

def four_points_linear {P Q R S T : Type} (linearPQRS : linear_order P Q R S) (anglePQT : ℚ) 
(angleRTS : ℚ) (reflex_angle_y : ℚ) : Prop :=
  anglePQT = 100 ∧ 
  angleRTS = 90 ∧ 
  reflex_angle_y = 350

theorem find_reflex_angle 
  {P Q R S T : Type} 
  (linearPQRS : linear_order P Q R S) 
  (h1 : ∠ P Q T = 100) 
  (h2 : ∠ R T S = 90) :
  reflex_angle_y = 350 :=
begin
  sorry
end

end find_reflex_angle_l728_728232


namespace prob_three_heads_in_eight_tosses_l728_728957

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l728_728957


namespace numbers_lcm_sum_l728_728200

theorem numbers_lcm_sum :
  ∃ A : List ℕ, A.length = 100 ∧
    (A.count 1 = 89 ∧ A.count 2 = 8 ∧ [4, 5, 6] ⊆ A) ∧
    A.sum = A.foldr lcm 1 :=
by
  sorry

end numbers_lcm_sum_l728_728200


namespace distance_between_parallel_lines_l728_728051

theorem distance_between_parallel_lines (r d : ℝ) 
  (h₁ : ∀ P Q, 0 < r)
  (h₂ : 36^2 + 36^2 = 18 * (18^2 + (d/2)^2))
  (h₃ : 30^2 + 30^2 = 15 * (15^2 + (3*d/2)^2))
  : d = 2 * real.sqrt 11 :=
begin
  -- proof would go here
  sorry
end

end distance_between_parallel_lines_l728_728051


namespace units_digit_of_2_pow_2018_l728_728799

theorem units_digit_of_2_pow_2018 : (2 ^ 2018) % 10 = 4 :=
by
  -- Setup: define the repeating pattern of units digits for powers of 2
  have cycle := [2, 4, 8, 6]
  -- Find the relevant index in the cycle for the units digit
  have index := 2018 % cycle.length
  -- Since 2018 % 4 = 2, the units digit will be the same as for 2^2
  have digit := cycle.nth index
  exact digit == 4

end units_digit_of_2_pow_2018_l728_728799


namespace trigonometric_identity_l728_728702

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = (2 / 5) :=
by
  sorry

end trigonometric_identity_l728_728702


namespace polynomial_q_l728_728387

theorem polynomial_q (q : ℂ[X]) (h₀ : q.monic) (h₁ : q.degree = 4) (h₂ : q.eval (2 - I) = 0) (h₃ : q.eval 0 = 32) :
    q = polynomial.C 25.6 + polynomial.X ^ 4 - 5.6 * polynomial.X ^ 3 + 22.4 * polynomial.X ^ 2 - 28 * polynomial.X := 
sorry

end polynomial_q_l728_728387


namespace area_of_inscribed_circle_proof_l728_728776

noncomputable def area_of_inscribed_circle {P F1 F2 : Real} 
  (h1 : ∀ P, P ∈ set_of P | P.x^2 - P.y^2 / 24 = 1 ∧ P.x ≥ 0 ∧ P.y ≥ 0)
  (h2 : dist P F1 = 8) 
  (h3 : dist P F2 = 6) 
  (h4 : dist F1 F2 = 10) 
  : ℝ := 
  let r := 2 in 
  π * r^2

theorem area_of_inscribed_circle_proof
  (P F1 F2 : Real)
  (h1 : ∀ P, P ∈ set_of P | P.x^2 - P.y^2 / 24 = 1 ∧ P.x ≥ 0 ∧ P.y ≥ 0)
  (h2 : dist P F1 = 8) 
  (h3 : dist P F2 = 6) 
  (h4 : dist F1 F2 = 10) :
  area_of_inscribed_circle h1 h2 h3 h4 = 4 * π := 
by sorry

end area_of_inscribed_circle_proof_l728_728776


namespace roberta_money_amount_l728_728380

theorem roberta_money_amount
  (M : ℝ)
  (H_shoes : 45)
  (H_bag : 28)
  (H_lunch : 7)
  (H_left : 78) :
  M = 158 :=
by 
  -- Define the conditions
  have H1 : H_bag = H_shoes - 17 := rfl
  have H2 : H_lunch = 1 / 4 * H_bag := rfl
  have H3 : M - (H_shoes + H_bag + H_lunch) = H_left := rfl

  -- The proof (you don't need to complete it)
  sorry

end roberta_money_amount_l728_728380


namespace vlad_dima_profit_difference_l728_728866

theorem vlad_dima_profit_difference :
  let initial_deposit := 3000
  let vlad_increase_rate := 1.2
  let vlad_fee := 0.1
  let dima_increase_rate := 1.4
  let dima_fee := 0.2
  let vlad_final := initial_deposit * vlad_increase_rate * (1 - vlad_fee)
  let dima_final := initial_deposit * dima_increase_rate * (1 - dima_fee)
  dima_final - vlad_final = 120 :=
by
  let initial_deposit := 3000
  let vlad_increase_rate := 1.2
  let vlad_fee := 0.1
  let dima_increase_rate := 1.4
  let dima_fee := 0.2
  let vlad_final := initial_deposit * vlad_increase_rate * (1 - vlad_fee)
  let dima_final := initial_deposit * dima_increase_rate * (1 - dima_fee)
  have h₁ : vlad_final = 3000 * 1.2 * 0.9 := rfl
  have h₂ : dima_final = 3000 * 1.4 * 0.8 := rfl
  have h₃ : 3000 * 1.2 * 0.9 = 3240 := rfl
  have h₄ : 3000 * 1.4 * 0.8 = 3360 := rfl
  have h₅ : 3360 - 3240 = 120 := rfl
  exact h₅

end vlad_dima_profit_difference_l728_728866


namespace area_of_quadrilateral_BEFC_l728_728775

theorem area_of_quadrilateral_BEFC :
  ∀ (A B C D E F : Type) (AB BC AC BD BE EC AF FC : ℝ),
  -- Conditions
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) →
  (∀ A B C, equilateral_triangle A B C 3) →
  (length B C = 3 ∧ length A B = 3 ∧ length A C = 3) →
  (length A B = length B D) →
  (midpoint E B C) →
  (intersects E D A C F) →
  -- Prove the area of quadrilateral BEFC is 9√3 / 8
  area_of_quadrilateral B E F C = 9 * real.sqrt 3 / 8 :=
by sorry

end area_of_quadrilateral_BEFC_l728_728775


namespace two_coins_heads_probability_l728_728082

/-- 
When tossing two coins of uniform density, the probability that both coins land with heads facing up is 1/4.
-/
theorem two_coins_heads_probability : 
  let outcomes := ["HH", "HT", "TH", "TT"]
  let favorable := "HH"
  probability (favorable) = 1/4 :=
by
  sorry

end two_coins_heads_probability_l728_728082


namespace binomial_7_4_equals_35_l728_728175

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Problem statement: Prove that binomial(7, 4) = 35
theorem binomial_7_4_equals_35 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_equals_35_l728_728175


namespace y_x_cubed_monotonic_increasing_l728_728406

theorem y_x_cubed_monotonic_increasing : 
  ∀ x1 x2 : ℝ, (x1 ≤ x2) → (x1^3 ≤ x2^3) :=
by
  intros x1 x2 h
  sorry

end y_x_cubed_monotonic_increasing_l728_728406


namespace true_propositions_count_l728_728667

variable {a b : Vec}

-- Definition of P
def prop_p (a b : Vec) : Prop := (a = b) → (abs a = abs b)
-- Definition of Converse of P
def converse_p (a b : Vec) : Prop := (abs a = abs b) → (a = b)
-- Definition of Inverse of P
def inverse_p (a b : Vec) : Prop := (a ≠ b) → (abs a ≠ abs b)
-- Definition of Contrapositive of P
def contrapositive_p (a b : Vec) : Prop := (abs a ≠ abs b) → (a ≠ b)

theorem true_propositions_count : (prop_p a b) ∧ (contrapositive_p a b) ∧ ¬(converse_p a b) ∧ ¬(inverse_p a b) := by
  sorry

end true_propositions_count_l728_728667


namespace inequality_transition_l728_728059

variable (n : ℕ)

theorem inequality_transition (k : ℕ) (hk : k > 2) :
  (∑ i in Finset.range(1 + 2 * (k + 1)), if 2 * k + 1 ≤ i ∧ i ≤ 2 * (k + 1) then 1 / i else 0) -
  (∑ i in Finset.range(1 + 2 * n), if n + 1 ≤ i ∧ i ≤ 2 * n then 1 / i else 0) =
  1 / (2 * k + 1) + 1 / (2 * (k + 1)) :=
sorry

end inequality_transition_l728_728059


namespace probability_of_exactly_three_heads_l728_728922

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l728_728922


namespace ending_number_of_range_l728_728847

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def range_digit_sum (a b : ℕ) : ℕ :=
  (list.range' a (b - a + 1)).map digit_sum.sum

theorem ending_number_of_range (a : ℕ) (h1 : range_digit_sum 0 a = 900) (h2 : range_digit_sum 18 21 = 24) : a = 99 :=
by
  sorry

end ending_number_of_range_l728_728847


namespace fraction_clerical_employees_l728_728370

theorem fraction_clerical_employees (total_employees : ℕ) (x : ℚ) 
  (h1 : total_employees = 3600)
  (h2 : x ≥ 0 ∧ x ≤ 1)
  (h3 : 0.2 * total_employees = 720) :
  x = 4 / 15 :=
by
  -- the proof would go here
  sorry

end fraction_clerical_employees_l728_728370


namespace hundredth_day_of_year_n_minus_one_is_saturday_l728_728327

theorem hundredth_day_of_year_n_minus_one_is_saturday
  (N : ℕ)
  (h1 : ∀ k : ℕ, (k ≡ 6 [MOD 7] → nat.succ (k % 365) = 35) → true)
  (h2 : ∀ k : ℕ, (k ≡ 5 [MOD 7] → nat.succ (k % 365) = 300) → true) :
  true := sorry

end hundredth_day_of_year_n_minus_one_is_saturday_l728_728327


namespace fold_point_area_l728_728624

theorem fold_point_area (P A B C : Point) (AB_length BC_length : ℝ) (angle_C : ℝ) :
    (AB_length = 24) →
    (BC_length = 48) →
    (angle_C = 90) →
    (∃ (q r s : ℝ), 
      q * Real.pi - r * Real.sqrt s = measure (set_of_fold_points P A B C) ∧ 
      q = 240 ∧ r = 360 ∧ s = 3) :=
begin
  sorry
end

end fold_point_area_l728_728624


namespace a_nk_sub_b_nk_l728_728344

/-- Let n be a positive integer, and let S_n be the set of all permutations of {1, 2, ..., n}.
    Let k be a non-negative integer.
    Let a_{n, k} be the number of even permutations σ in S_n such that ∑_{i=1}^{n} |σ(i) - i| = 2k.
    Let b_{n, k} be the number of odd permutations σ in S_n such that ∑_{i=1}^{n} |σ(i) - i| = 2k.
    Then a_{n, k} - b_{n, k} = (-1)^k * binom(n-1, k). -/
theorem a_nk_sub_b_nk (n k : ℕ) (h_n : 0 < n) :
  let S_n := {σ : Fin n → Fin n // Function.Bijective σ},
  let a_nk := S_n.count (λ σ, (∑ i, |σ.val i - i| = 2 * k) ∧ Even (Permutation.parity σ.val)),
  let b_nk := S_n.count (λ σ, (∑ i, |σ.val i - i| = 2 * k) ∧ Odd (Permutation.parity σ.val)) in
  a_nk - b_nk = (-1)^k * Nat.choose (n - 1) k := sorry

end a_nk_sub_b_nk_l728_728344


namespace max_regions_5_segments_l728_728873

theorem max_regions_5_segments :
  let R := λ n, (n * (n + 1)) / 2 + 1 in
  R 5 = 16 :=
by
  sorry

end max_regions_5_segments_l728_728873


namespace sum_of_perpendiculars_l728_728102

-- Define the points and variables
variables {A B C D E F O P Q R : Type} [Point A] [Point B] [Point C] [Point D] [Point E] [Point F] [Point O] [Point P] [Point Q] [Point R]

-- Introduce the conditions as hypotheses
variables (isRegularHexagon : isRegularHexagon A B C D E F)
variables (perpendiculars : (AP : perpendicular A DE) (AQ : perpendicular A (extended C D)) (AR : perpendicular A (extended E F)))
variables (centerO : isCenter O A B C D E F)
variables (OP_value : OP = 2)

-- Main statement to be proven
theorem sum_of_perpendiculars (radius : ℝ) : 
  AO + AQ + AR = 3 * radius * sqrt(3) - 2 :=
by 
  sorry

end sum_of_perpendiculars_l728_728102


namespace factorial_difference_l728_728544

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem factorial_difference : fact 10 - fact 9 = 3265920 := by
  have h1 : fact 9 = 362880 := sorry
  have h2 : fact 10 = 10 * fact 9 := by rw [fact]
  calc
    fact 10 - fact 9
        = 10 * fact 9 - fact 9 := by rw [h2]
    ... = 9 * fact 9 := by ring
    ... = 9 * 362880 := by rw [h1]
    ... = 3265920 := by norm_num

end factorial_difference_l728_728544


namespace coin_toss_probability_l728_728948

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l728_728948


namespace product_of_two_digit_numbers_5488_has_smaller_number_56_l728_728037

theorem product_of_two_digit_numbers_5488_has_smaller_number_56 (a b : ℕ) (h_a2 : 10 ≤ a) (h_a3 : a < 100) (h_b2 : 10 ≤ b) (h_b3 : b < 100) (h_prod : a * b = 5488) : a = 56 ∨ b = 56 :=
by {
  sorry
}

end product_of_two_digit_numbers_5488_has_smaller_number_56_l728_728037


namespace general_term_of_sequence_sum_of_first_n_terms_l728_728260

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable (q : ℝ) [lin_ordered_field ℝ]

-- Conditions given in the problem
hypothesis (h1 : ∀ n, a (n + 1) = a n * q)
hypothesis (h2 : q < 1)
hypothesis (h3 : a 2 = 2)
hypothesis (h4 : a 0 + a 1 + a 2 = 7)

-- Define the derived logarithmic sequence
def b (n : ℕ) := log 2 (a n)

-- Statements to be proven
theorem general_term_of_sequence : ∀ n, a n = (1/2)^(n-3) := sorry

theorem sum_of_first_n_terms (n : ℕ) : 
  (∑ k in range n, b k) = (5 * n - n^2) / 2 := sorry

end general_term_of_sequence_sum_of_first_n_terms_l728_728260


namespace coin_toss_probability_l728_728919

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l728_728919


namespace count_seven_digit_palindromes_with_odd_middle_l728_728434

noncomputable def count_palindromes : ℕ :=
  let digits := {5, 6, 7}
  let odd_digits := {5, 7}
  3 * 3 * 3 * 2

theorem count_seven_digit_palindromes_with_odd_middle :
  count_palindromes = 54 :=
by
  sorry

end count_seven_digit_palindromes_with_odd_middle_l728_728434


namespace totalTaxIsCorrect_l728_728460

-- Define the different income sources
def dividends : ℝ := 50000
def couponIncomeOFZ : ℝ := 40000
def couponIncomeCorporate : ℝ := 30000
def capitalGain : ℝ := (100 * 200) - (100 * 150)

-- Define the tax rates
def taxRateDividends : ℝ := 0.13
def taxRateCorporateBond : ℝ := 0.13
def taxRateCapitalGain : ℝ := 0.13

-- Calculate the tax for each type of income
def taxOnDividends : ℝ := dividends * taxRateDividends
def taxOnCorporateCoupon : ℝ := couponIncomeCorporate * taxRateCorporateBond
def taxOnCapitalGain : ℝ := capitalGain * taxRateCapitalGain

-- Sum of all tax amounts
def totalTax : ℝ := taxOnDividends + taxOnCorporateCoupon + taxOnCapitalGain

-- Prove that total tax equals the calculated figure
theorem totalTaxIsCorrect : totalTax = 11050 := by
  sorry

end totalTaxIsCorrect_l728_728460


namespace club_president_vice_president_count_l728_728806

theorem club_president_vice_president_count :
  let boys := 14 in
  let girls := 10 in
  let total_members := 24 in
  let senior_boys := 4 in
  let senior_girls := 2 in
  let senior_members := senior_boys + senior_girls in
  total_members = boys + girls ∧
  senior_members = senior_boys + senior_girls →
  -- The number of ways to choose a president and vice-president with given constraints
  (senior_boys * girls + senior_girls * boys) = 68 := 
by
  intros
  sorry

end club_president_vice_president_count_l728_728806


namespace ratio_of_radii_l728_728751

-- Given conditions
variables {b a c : ℝ}
variables (h1 : π * b^2 - π * c^2 = 2 * π * a^2)
variables (h2 : c = 1.5 * a)

-- Define and prove the ratio
theorem ratio_of_radii (h1: π * b^2 - π * c^2 = 2 * π * a^2) (h2: c = 1.5 * a) :
  a / b = 2 / Real.sqrt 17 :=
sorry

end ratio_of_radii_l728_728751


namespace shaded_region_area_l728_728110

/-- A circle of radius 3 is centered at O. Regular hexagon OABCDF has side length 2. Sides AB and DF are extended past B and F to meet the circle at G and H, respectively.
    The area of the shaded region bounded by BG, FH, and the minor arc connecting G and H is 3π - (9√3)/4. -/
theorem shaded_region_area : 
  let O : point
  let radius_circle := 3
  let hexagon_side := 2
  let circle_area := 3 * π
  let triangle_area := (9 * real.sqrt 3) / 4
  in
  (circle_area - triangle_area = 3 * π - (9 * real.sqrt 3) / 4) := sorry

end shaded_region_area_l728_728110


namespace minimum_score_last_two_games_l728_728755

/-- Given that the player scored 26, 15, 12, and 24 points in the fifteenth, sixteenth, seventeenth,
and eighteenth games respectively, and the average score after twenty games is greater than 20,
prove that the minimum score in the last two games is 58. -/

theorem minimum_score_last_two_games :
  let score_15 := 26
  let score_16 := 15
  let score_17 := 12
  let score_18 := 24
  let sum_15_to_18 := score_15 + score_16 + score_17 + score_18
  sum_15_to_18 = 77 ->
  (avg_20 : ℤ) (H : avg_20 > 20)
  (total_20 : ℤ) (H_total : total_20 = avg_20 * 20)
  (score_1_to_14_plus_19_20 : ℤ) (H_score : score_1_to_14_plus_19_20 = total_20 - sum_15_to_18) :
  ∃ score_19_20, score_19_20 ≥ 58 ∧ score_19_20 = score_1_to_14_plus_19_20 - max_points_1_to_14 :=
  sorry

end minimum_score_last_two_games_l728_728755


namespace binomial_7_4_equals_35_l728_728170

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Problem statement: Prove that binomial(7, 4) = 35
theorem binomial_7_4_equals_35 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_equals_35_l728_728170


namespace tangent_line_range_l728_728647

noncomputable def hasCommonTangentLine (a : ℝ) : Prop := 
  ∃ x : ℝ, 2 * x = (1 / a) * exp x

theorem tangent_line_range (a : ℝ) (h : hasCommonTangentLine a) : a ∈ Ici (exp 2 / 4) :=
sorry

end tangent_line_range_l728_728647


namespace cosine_of_angle_between_AB_and_AC_l728_728210

noncomputable def point := (ℝ × ℝ × ℝ)

def A : point := (0, 3, -6)
def B : point := (9, 3, 6)
def C : point := (12, 3, 3)

def vector_sub (p1 p2 : point) : point :=
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

def dot_product (v1 v2 : point) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : point) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

def cosine (v1 v2 : point) : ℝ :=
  (dot_product v1 v2) / (magnitude v1 * magnitude v2)

theorem cosine_of_angle_between_AB_and_AC : 
  cosine (vector_sub B A) (vector_sub C A) = 0.96 :=
by {
  let vAB := vector_sub B A,
  let vAC := vector_sub C A,
  have : dot_product vAB vAC = 216 := by sorry,
  have : magnitude vAB = 15 := by sorry,
  have : magnitude vAC = 15 := by sorry,
  have : cosine vAB vAC = 216 / (15 * 15) := by sorry,
  have : 216 / 225 = 0.96 := by sorry,
  exact this,
}

end cosine_of_angle_between_AB_and_AC_l728_728210


namespace find_N_in_arithmetic_sequences_l728_728416

/-- The sequence in the row and in each column of squares form three distinct arithmetic sequences.
We need to determine the value of N which lies at the given position in this configuration. -/
theorem find_N_in_arithmetic_sequences (a b c d e f g h : ℤ) : 
    -- First column sequence
    a = 18 → 
    b = 14 → 
    c = 10 →
    d = 6 → 
    -- Numbers given relate row-wise with common difference
    e = 21 → 
    (d - e) / 3 = -5 →
    -- The calculated bottom number of the second column 
    g = -17 → 
    -- The second column common difference 
    (g - h) / 4 = -2 → 
    -- Solving for N
    d = -9 → 
    N = -7 :=
begin
    -- solution provided 
    sorry
end

end find_N_in_arithmetic_sequences_l728_728416


namespace blue_whale_tongue_weight_l728_728834

theorem blue_whale_tongue_weight (ton_in_pounds : ℕ) (tons : ℕ) (blue_whale_tongue_weight : ℕ) :
  ton_in_pounds = 2000 → tons = 3 → blue_whale_tongue_weight = tons * ton_in_pounds → blue_whale_tongue_weight = 6000 :=
  by
  intros h1 h2 h3
  rw [h2] at h3
  rw [h1] at h3
  exact h3

end blue_whale_tongue_weight_l728_728834


namespace probability_of_exactly_three_heads_l728_728925

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l728_728925


namespace second_year_students_count_l728_728863

theorem second_year_students_count (n : ℕ) :
  ∃ (n : ℕ), 
    let total_points := (n + 2) * (n + 1) / 2 in
    (total_points - 8) / n = 4 ∧ (total_points - 8) % n = 0 :=
begin
  use 7,
  let total_points := (7 + 2) * (7 + 1) / 2,
  have h1 : total_points = 36, from rfl,
  have h2 : (total_points - 8) = 28, by rw h1,
  have h3 : 28 / 7 = 4, from rfl,
  exact ⟨4, rfl⟩,
end

end second_year_students_count_l728_728863


namespace number_of_divisors_greater_than_eight_factorial_l728_728285

-- Definitions based on conditions
def nine_factorial := fact 9
def eight_factorial := fact 8

-- Theorem statement
theorem number_of_divisors_greater_than_eight_factorial :
  ∃ (n : ℕ), n = 8 ∧ ∀ d : ℕ, (d ∣ nine_factorial ∧ d > eight_factorial) → n = 8 :=
by
  sorry

end number_of_divisors_greater_than_eight_factorial_l728_728285


namespace inverse_of_given_matrix_l728_728601

theorem inverse_of_given_matrix :
  let matrix := ![![5, 3], ![10, 6]] in
  let zero_matrix := ![![0, 0], ![0, 0]] in
  matrix.det = 0 → Matrix.has_inv matrix → matrix⁻¹ = zero_matrix :=
by
  intro matrix zero_matrix
  have h_det : matrix.det = 0 := by sorry -- the determinant computation and its result
  have h_singular : ¬ matrix.has_inv := by sorry -- conclusion from the determinant
  have h_inv : matrix⁻¹ = zero_matrix := by sorry -- conclusion from the singularity
  
  exact h_inv

end inverse_of_given_matrix_l728_728601


namespace equal_segments_for_regular_polygon_l728_728422

theorem equal_segments_for_regular_polygon (n : ℕ) (m : ℕ) 
  (h₁ : n = 4 * m + 2 ∨ n = 4 * m + 3)
  (A : Fin (2 * n) → Type) : 
  ∃ (i j k l : Fin (2 * n)), (i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ j ≠ l) ∧
  dist (A i) (A j) = dist (A k) (A l) := 
sorry

end equal_segments_for_regular_polygon_l728_728422


namespace find_range_of_a_l728_728634

def have_real_roots (a : ℝ) : Prop := a^2 - 16 ≥ 0

def is_increasing_on_interval (a : ℝ) : Prop := a ≥ -12

theorem find_range_of_a (a : ℝ) : ((have_real_roots a ∨ is_increasing_on_interval a) ∧ ¬(have_real_roots a ∧ is_increasing_on_interval a)) → (a < -12 ∨ (-4 < a ∧ a < 4)) :=
by 
  sorry

end find_range_of_a_l728_728634


namespace total_students_l728_728116

theorem total_students (
  Y R B : Finset ℕ -- sets representing students using yellow, red, and blue respectively
  (hY : Y.card = 46)
  (hR : R.card = 69)
  (hB : B.card = 104)
  (hYB : (Y ∩ B \ R).card = 14)
  (hYR : (Y ∩ R \ B).card = 13)
  (hBR : (B ∩ R \ Y).card = 19)
  (hAll : (Y ∩ R ∩ B).card = 16)
) : Y ∪ R ∪ B.card = 141 :=
by {
  -- Proof is omitted
  sorry
}

end total_students_l728_728116


namespace angle_BED_120_degrees_l728_728183

variables (A B C D E : Type) (quadrilateral : quadrilateral A B C D) (triangle1 : equilateral_triangle A B E) (triangle2 : equilateral_triangle E C D)

def square (A B C D : Type) : Prop := ∀ (a b c d : Type), 
∠ A B C = 90 ∧ ∠ B C D = 90 ∧ ∠ C D A = 90 ∧ ∠ D A B = 90

def equilateral_triangle (A B C : Type) : Prop := ∀ (a b c : Type), 
∠ A B C = 60 ∧ ∠ B C A = 60 ∧ ∠ C A B = 60

theorem angle_BED_120_degrees (h_square : square A B C D) 
                                (h_eq_tri1 : equilateral_triangle A B E)
                                (h_eq_tri2 : equilateral_triangle E C D):
                                ∠ B E D = 120 := sorry

end angle_BED_120_degrees_l728_728183


namespace scientific_notation_361000000_l728_728803

theorem scientific_notation_361000000 :
  361000000 = 3.61 * 10^8 :=
sorry

end scientific_notation_361000000_l728_728803


namespace trigonometric_identity_l728_728704

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trigonometric_identity_l728_728704


namespace factorial_difference_l728_728535

-- Define factorial function for natural numbers
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- The theorem we want to prove
theorem factorial_difference : factorial 10 - factorial 9 = 3265920 :=
by
  rw [factorial, factorial]
  sorry

end factorial_difference_l728_728535


namespace strictly_monotone_function_l728_728342

open Function

-- Define the problem
theorem strictly_monotone_function (f : ℝ → ℝ) (F : ℝ → ℝ → ℝ)
  (hf_cont : Continuous f) (hf_nonconst : ¬ (∃ c, ∀ x, f x = c))
  (hf_eq : ∀ x y : ℝ, f (x + y) = F (f x) (f y)) :
  StrictMono f :=
sorry

end strictly_monotone_function_l728_728342


namespace min_value_of_expression_l728_728781

noncomputable def min_value_expression (a b c : ℝ) : ℝ :=
  a^2 + 4 * a * b + 9 * b^2 + 8 * b * c + 3 * c^2

theorem min_value_of_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b * c = 1) (h5 : a / b = 2) : 
  min_value_expression a b c = 3 * real.cbrt 63 :=
sorry

end min_value_of_expression_l728_728781


namespace Paula_min_correct_answers_l728_728390

theorem Paula_min_correct_answers :
  let n := 35
  let a := 30
  let u := 5
  let p_c := 7
  let p_i := -1
  let p_u := 2
  let S_min := 150
  ∃ k : ℕ, (k ≤ a ∧ p_c * k + p_i * (a - k) + p_u * u ≥ S_min) ∧ k = 20 :=
by
  sorry

end Paula_min_correct_answers_l728_728390


namespace num_possible_integer_values_n_l728_728015

open Real

theorem num_possible_integer_values_n :
  ∃ (ABCD : Type) (E : Point) (perimeter_abe : ℝ) (n : ℝ) (a b : ℝ)
    (is_rectangle : ABCD)
    (perimeter_abe_eq : perimeter_abe = 10 * π)
    (perimeter_ade_eq : n = b + 3 * (Math.sqrt (a^2 + b^2)) / 2),
  (perimeter_abe_eq → is_rectangle → (perimeter_abe = 10 * π)) → 
  47 = (62 - 16 + 1) := sorry

end num_possible_integer_values_n_l728_728015


namespace LeRoy_should_pay_Bernardo_l728_728584

theorem LeRoy_should_pay_Bernardo 
    (initial_loan : ℕ := 100)
    (LeRoy_gas_expense : ℕ := 300)
    (LeRoy_food_expense : ℕ := 200)
    (Bernardo_accommodation_expense : ℕ := 500)
    (total_expense := LeRoy_gas_expense + LeRoy_food_expense + Bernardo_accommodation_expense)
    (shared_expense := total_expense / 2)
    (LeRoy_total_responsibility := shared_expense + initial_loan)
    (LeRoy_needs_to_pay := LeRoy_total_responsibility - (LeRoy_gas_expense + LeRoy_food_expense)) :
    LeRoy_needs_to_pay = 100 := 
by
    sorry

end LeRoy_should_pay_Bernardo_l728_728584


namespace possible_roots_l728_728843

theorem possible_roots (a b p q : ℤ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : a ≠ b)
  (h4 : p = -(a + b))
  (h5 : q = ab)
  (h6 : (a + p) % (q - 2 * b) = 0) :
  a = 1 ∨ a = 3 :=
  sorry

end possible_roots_l728_728843


namespace probability_different_colors_l728_728730

theorem probability_different_colors :
  let total_chips := 16
  let prob_blue := (7 : ℚ) / total_chips
  let prob_yellow := (5 : ℚ) / total_chips
  let prob_red := (4 : ℚ) / total_chips
  let prob_blue_then_nonblue := prob_blue * ((prob_yellow + prob_red) : ℚ)
  let prob_yellow_then_non_yellow := prob_yellow * ((prob_blue + prob_red) : ℚ)
  let prob_red_then_non_red := prob_red * ((prob_blue + prob_yellow) : ℚ)
  let total_prob := prob_blue_then_nonblue + prob_yellow_then_non_yellow + prob_red_then_non_red
  total_prob = (83 : ℚ) / 128 := 
by
  sorry

end probability_different_colors_l728_728730


namespace ratio_of_rises_requires_evaluation_l728_728431

-- Define the conditions as variables and assumptions
variables (r1 r2 h1 h2 : ℝ)
variable (cube_volume : ℝ)
hypothesis h_r1 : r1 = 5
hypothesis h_r2 : r2 = 10
hypothesis h_cube : cube_volume = 8  -- cube side 2 cm leads to volume 8 cm^3
hypothesis h_initial_volumes_equal : π * r1^2 * h1 = π * r2^2 * h2
hypothesis h_h1_h2_relation : h1 = 4 * h2

-- Define the statement that we intend to prove
theorem ratio_of_rises_requires_evaluation :
  (let x := (1 + 24 / (π * 25 * h1))^(1/3),
       y := (1 + 24 / (π * 100 * h2))^(1/3)
   in 4 * (x - 1) / (y - 1)) requires numerical evaluation :=
sorry

end ratio_of_rises_requires_evaluation_l728_728431


namespace is_abs_g_piecewise_l728_728355

def g (x : ℝ) : ℝ :=
if x ≥ -2 then x + 4
else if x ≥ -4 then -(x + 2)^2 + 2
else -3 * x + 3

def abs_g (x : ℝ) : ℝ := abs (g x)

theorem is_abs_g_piecewise :
  abs_g = λ x, 
      if x ≥ -2 then x + 4
      else if x ≥ -4 then -((x + 2)^2 - 2)
      else -3 * x + 3 := 
  sorry

end is_abs_g_piecewise_l728_728355


namespace spherical_to_rectangular_coords_l728_728575

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), ρ = 15 → θ = 5 * Real.pi / 4 → φ = Real.pi / 4 →
  let x := ρ * Real.sin φ * Real.cos θ in
  let y := ρ * Real.sin φ * Real.sin θ in
  let z := ρ * Real.cos φ in
  (x, y, z) = (-15 / 2, -15 / 2, 15 * Real.sqrt 2 / 2) :=
begin
  intros ρ θ φ hρ hθ hφ,
  let x := ρ * Real.sin φ * Real.cos θ,
  let y := ρ * Real.sin φ * Real.sin θ,
  let z := ρ * Real.cos φ,
  rw [hρ, hθ, hφ],
  simp,
  split,
  { rw [Real.sin_pi_div_four, Real.cos_five_pi_div_four],
    simp },
  split,
  { rw [Real.sin_pi_div_four, Real.sin_five_pi_div_four],
    simp },
  { rw [Real.cos_pi_div_four],
    simp }
end

end spherical_to_rectangular_coords_l728_728575


namespace minimum_value_of_t_l728_728437

theorem minimum_value_of_t :
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → sqrt (x * y) ≤ (1 / (2 * sqrt 6)) * (2 * x + 3 * y)) :=
by
  intros x y hx hy
  sorry

end minimum_value_of_t_l728_728437


namespace intersection_of_sets_l728_728277

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {x | (x + 1) * (x - 2) < 0 ∧ x ∈ ℤ}

theorem intersection_of_sets : A ∩ B = {1} := by
  sorry

end intersection_of_sets_l728_728277


namespace manufacturing_section_degrees_l728_728825

theorem manufacturing_section_degrees (percentage : ℝ) (total_degrees : ℝ) (h1 : total_degrees = 360) (h2 : percentage = 35) : 
  ((percentage / 100) * total_degrees) = 126 :=
by
  sorry

end manufacturing_section_degrees_l728_728825


namespace factorial_difference_l728_728529

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728529


namespace scientific_notation_1300000_l728_728898

theorem scientific_notation_1300000 :
  1300000 = 1.3 * 10^6 :=
sorry

end scientific_notation_1300000_l728_728898


namespace factorize_expression_l728_728591

variable {a b : ℝ} -- define a and b as real numbers

theorem factorize_expression : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l728_728591


namespace coin_toss_probability_l728_728915

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l728_728915


namespace max_value_of_trig_function_l728_728837

theorem max_value_of_trig_function :
  ∃ x : ℝ, 
    let y := (sin x * cos x) / (1 + sin x + cos x) in
    y ≤ (sqrt 2 - 1) / 2 :=
sorry

end max_value_of_trig_function_l728_728837


namespace lele_has_enough_money_and_remaining_19_yuan_l728_728841

def price_A : ℝ := 46.5
def price_B : ℝ := 54.5
def total_money : ℝ := 120

theorem lele_has_enough_money_and_remaining_19_yuan : 
  (price_A + price_B ≤ total_money) ∧ (total_money - (price_A + price_B) = 19) :=
by
  sorry

end lele_has_enough_money_and_remaining_19_yuan_l728_728841


namespace binomial_7_4_equals_35_l728_728173

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Problem statement: Prove that binomial(7, 4) = 35
theorem binomial_7_4_equals_35 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_equals_35_l728_728173


namespace divide_cube_into_smaller_cubes_l728_728199

theorem divide_cube_into_smaller_cubes : 
  ∃ (n : ℕ), n = 20 ∧ ∀ (cubes : List ℕ), 
  List.length cubes = n ∧ 
  (∀ (e : ℕ), e ∈ cubes → (∃ k : ℕ, e = k ∧ 1 ≤ k ∧ k ≤ 3)) ∧ 
  (∀ (v₁ v₂ : ℕ), v₁ ∈ cubes → v₂ ∈ cubes → v₁ ≠ v₂ → k ≠ 1 → e ≠ 1 → v₁^3 ≠ v₂^3) ∧ 
  ((3^3) = List.sum (List.map (λ e, e^3) cubes)) :=
by
  sorry

end divide_cube_into_smaller_cubes_l728_728199


namespace total_students_l728_728739

variable (T : ℕ)

-- Conditions
def is_girls_percentage (T : ℕ) := 60 / 100 * T
def is_boys_percentage (T : ℕ) := 40 / 100 * T
def boys_not_in_clubs (number_of_boys : ℕ) := 2 / 3 * number_of_boys

theorem total_students (h1 : is_girls_percentage T + is_boys_percentage T = T)
  (h2 : boys_not_in_clubs (is_boys_percentage T) = 40) : T = 150 :=
by
  sorry

end total_students_l728_728739


namespace factorial_difference_l728_728553

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728553


namespace probability_of_event_A_eq_five_fourteenth_l728_728309

noncomputable theory

open_locale classical 
open_locale big_operators 

def probability_event_exactly_two_students_from_same_school (n m : ℕ) : ℚ :=
  let total_ways := nat.choose 10 4 in
  let ways_event_A := nat.choose 5 1 * nat.choose 2 2 * nat.choose 8 2 in
  ways_event_A / total_ways

theorem probability_of_event_A_eq_five_fourteenth :
  probability_event_exactly_two_students_from_same_school 4 10 = 5/14 :=
by sorry

end probability_of_event_A_eq_five_fourteenth_l728_728309


namespace maximal_negatives_in_exponential_equation_l728_728692

theorem maximal_negatives_in_exponential_equation :
  ∀ (a b c d : ℤ), (a ≤ b) → (c ≤ d) → (5 ^ a + 5 ^ b = 3 ^ c + 3 ^ d) → false := sorry

end maximal_negatives_in_exponential_equation_l728_728692


namespace longer_diagonal_of_parallelogram_l728_728862

noncomputable def DiagonalParallelogram 
  (h1 h2 : ℝ) (α : ℝ) : ℝ :=
  (h1^2 + 2*h1*h2*cos α + h2^2)^(1/2) / sin α

theorem longer_diagonal_of_parallelogram 
  (h1 h2 : ℝ) (α : ℝ) 
  (h1_pos : 0 < h1)
  (h2_pos : 0 < h2)
  (sin_pos : 0 < sin α) :
  DiagonalParallelogram h1 h2 α = 
    (h1^2 + 2*h1*h2*cos α + h2^2)^(1/2) / (sin α) :=
sorry

end longer_diagonal_of_parallelogram_l728_728862


namespace no_equal_number_of_liars_l728_728804

-- Define the concept of fans, teams, and statements:
inductive Team | SuperEagles | SuperLions
inductive Fan | Knight | Liar

-- Declaration made by each fan:
def declaration (f: Fan) (team_right: Team) : Prop :=
  match f with
  | Fan.Knight => team_right = Team.SuperEagles
  | Fan.Liar => team_right ≠ Team.SuperEagles

-- Define the problem conditions:
constant total_fans : ℕ
constant fans_superEagles : ℕ
constant fans_superLions : ℕ
constant knights : ℕ
constant liars : ℕ

axiom h1 : total_fans = 50
axiom h2 : fans_superEagles = 25
axiom h3 : fans_superLions = 25
axiom h4 : total_fans = fans_superEagles + fans_superLions

-- Assume equal number of liars among both teams:
constant k : ℕ
axiom h5 : liars = 2 * k
axiom h6 : k ≤ 25 -- As 50 total fans

-- Proposition asserting the impossibility:
theorem no_equal_number_of_liars :
  ∀ k, ¬ (liars = 2 * k ∧ k ≤ 25 ∧ 
    (declaration Fan.Liar Team.SuperEagles → 
    declaration Fan.Knight Team.SuperLions)) :=
by {
  sorry
}

end no_equal_number_of_liars_l728_728804


namespace number_of_pk_exceeding_one_eighth_l728_728306

-- Define η as the smaller number of two drawn balls from a set of balls numbered from 1 to 8.
def η (a b : ℕ) : ℕ := min a b

-- Define pk as the probability that η equals k.
def pk (k : ℕ) : ℚ := 
if k = 1 then 7 / 28 else
if k = 2 then 6 / 28 else
if k = 3 then 5 / 28 else
if k = 4 then 4 / 28 else
if k = 5 then 3 / 28 else
if k = 6 then 2 / 28 else
1 / 28

-- The main theorem to prove, where the number of pk that are greater than 1/8 is 4.
theorem number_of_pk_exceeding_one_eighth : 
  ∑ k in (finset.range 7).filter (λ k, pk (k + 1) > 1 / 8), 1 = 4 :=
sorry

end number_of_pk_exceeding_one_eighth_l728_728306


namespace duration_of_each_period_is_three_l728_728731

-- Definitions based on the conditions
def time_points := [1, 4, 7] -- hours (converted to integer representation of time)
def bacteria_amounts := [10.0, 11.05, 12.1] -- grams

-- Definition of the same factor increase
def same_factor_increase (times: List ℕ) (amounts: List ℝ) (d: ℕ) :=
  ∀ i, i < amounts.length - 1 → (amounts.get (i+1)) = (amounts.get i) * (amounts.get (1) / amounts.get 0)^(times.get (i+1) - times.get i) / d

-- The statement of the proof problem
theorem duration_of_each_period_is_three (times: List ℕ) (amounts: List ℝ) :
  same_factor_increase time_points bacteria_amounts 3 → ∃ d, d = 3 :=
by
  sorry

end duration_of_each_period_is_three_l728_728731


namespace coin_toss_probability_l728_728944

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l728_728944


namespace place_sllip_with_4_5_l728_728330

section Jessica
variable (slips : List ℝ) (cup : Type) [DecidableEq cup]

-- Slips of paper values
def slips_values : List ℝ := [1, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 4, 4, 4.5, 5, 5.5, 6]

-- Cups labels
def cups : List cup := ["A", "B", "C", "D", "E", "F"]

-- Mapping cups to sums
def cup_sums (A B C D E F : ℝ) : List ℝ := [A, B, C, D, E, F]

-- Several conditions from the problem:
def is_consecutive_even (sums: List ℝ) : Prop := 
  ∀ i ∈ List.range 5, sums.nth i + 2 = sums.nth (i + 1)

-- Assignment constraints
def assignment (xs : cup → ℝ) : Prop := 
  xs "F" = 2 ∧ xs "B" = 3

-- Problem statement
theorem place_sllip_with_4_5 
  (xs : cup → ℝ) 
  (h1 : is_consecutive_even [xs "A", xs "B", xs "C", xs "D", xs "E", xs "F"]) 
  (h2 : assignment xs)
  (h3 : (xs "A" + xs "B" + xs "C" + xs "D" + xs "E" + xs "F") = (49.5 : ℝ))
  : xs "C" = 4.5 :=
  sorry
end Jessica

end place_sllip_with_4_5_l728_728330


namespace abs_z_eq_sqrt_2_l728_728646

noncomputable def z : ℂ :=
sorry

theorem abs_z_eq_sqrt_2 (z : ℂ) (h : conj(z) = (-2 * I) / z + 2) : abs(z) = real.sqrt 2 :=
sorry

end abs_z_eq_sqrt_2_l728_728646


namespace altitude_inequality_not_universally_true_l728_728351

noncomputable def altitudes (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) : Prop :=
  ∃ m_a m_b m_c : ℝ, m_a ≤ m_b ∧ m_b ≤ m_c 

noncomputable def seg_to_orthocenter (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) : Prop :=
  ∃ m_a_star m_b_star m_c_star : ℝ, True

theorem altitude_inequality (a b c m_a m_b m_c : ℝ) 
  (h₀ : a ≥ b) (h₁ : b ≥ c) (h₂ : m_a ≤ m_b) (h₃ : m_b ≤ m_c) :
  (a + m_a ≥ b + m_b) ∧ (b + m_b ≥ c + m_c) :=
by
  sorry

theorem not_universally_true (a b c m_a_star m_b_star m_c_star : ℝ)
  (h₀ : a ≥ b) (h₁ : b ≥ c) :
  ¬(a + m_a_star ≥ b + m_b_star ∧ b + m_b_star ≥ c + m_c_star) :=
by
  sorry

end altitude_inequality_not_universally_true_l728_728351


namespace inequality_solution_l728_728595

theorem inequality_solution (x : ℝ) :
  (1 / (x ^ 2 + 4) > 4 / x + 27 / 10) ↔ x ∈ Ioo (-5/8 : ℝ) 0 ∪ Ioo 0 (2/5 : ℝ) :=
by
  sorry

end inequality_solution_l728_728595


namespace S_10_eq_210_l728_728777

noncomputable def floor_sqrt (x : ℝ) : ℕ :=
  int.to_nat (⌊real.sqrt x⌋)

noncomputable def S (n : ℕ) : ℕ :=
  ∑ i in finset.range (2 * n + 1), floor_sqrt (n^2 + i)

theorem S_10_eq_210 : S 10 = 210 :=
by
  sorry

end S_10_eq_210_l728_728777


namespace smallest_b_for_no_real_root_l728_728067

theorem smallest_b_for_no_real_root :
  ∃ b : ℤ, (b < 8 ∧ b > -8) ∧ (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ -6) ∧ (b = -7) :=
by
  sorry

end smallest_b_for_no_real_root_l728_728067


namespace factorial_difference_l728_728555

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728555


namespace cos_alpha_lt_sqrt3_over_2_l728_728070

theorem cos_alpha_lt_sqrt3_over_2 (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : α > π / 6) : 
  real.cos α < real.cos (π / 6) := by
  sorry

end cos_alpha_lt_sqrt3_over_2_l728_728070


namespace volume_and_surface_area_implies_sum_of_edges_l728_728849

-- Define the problem conditions and prove the required statement
theorem volume_and_surface_area_implies_sum_of_edges :
  ∃ (a r : ℝ), 
    (a / r) * a * (a * r) = 216 ∧ 
    2 * ((a^2 / r) + a^2 * r + a^2) = 288 →
    4 * ((a / r) + a * r + a) = 96 :=
by
  sorry

end volume_and_surface_area_implies_sum_of_edges_l728_728849


namespace area_difference_l728_728111

-- Defining the given parameters
def radius : ℝ := 3
def side_length : ℝ := 6

-- Defining the areas
def circle_area : ℝ := Real.pi * radius^2
def triangle_area : ℝ := (Real.sqrt 3 / 4) * side_length^2

-- Stating the theorem
theorem area_difference :
  circle_area - triangle_area = 9 * (Real.pi - Real.sqrt 3) :=
by
  -- Proof goes here
  sorry

end area_difference_l728_728111


namespace trigonometric_identity_l728_728710

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
by 
  sorry

end trigonometric_identity_l728_728710


namespace radius_of_spherical_circle_correct_l728_728417

noncomputable def radius_of_spherical_circle (rho theta phi : ℝ) : ℝ :=
  if rho = 1 ∧ phi = Real.pi / 4 then Real.sqrt 2 / 2 else 0

theorem radius_of_spherical_circle_correct :
  ∀ (theta : ℝ), radius_of_spherical_circle 1 theta (Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end radius_of_spherical_circle_correct_l728_728417


namespace playerA_wins_optimally_l728_728864

noncomputable def optimalGameWinner : ℕ → ℕ → ℕ → Prop
| 0, _, _ => false  -- no valid outcome if no turns (invalid state)
| n+1, 1, 0 => -- Start on square 1, A's turn
  true  -- A wins
| n+1, 1, m+1 => -- Start on square 1, B's turn
  false  -- B loses
| n+1, sq, 0 =>  -- A's turn at any other square
  (optimalGameWinner n (if sq > 100 then sq-100 else sq) 1 && sq+1 ≤ 100) ||
  (optimalGameWinner n (if sq+10 > 100 then sq-100 else sq+10) 1 && sq+10 <= 100) ||
  (optimalGameWinner n (if sq+11 > 100 then sq-100 else sq+11) 1 && sq+11 <= 100)
| n+1, sq, m+1 =>  -- B's turn at any other square
  (optimalGameWinner n (if sq > 100 then sq-100 else sq) 0 && sq+1 ≤ 100) ||
  (optimalGameWinner n (if sq+10 > 100 then sq-100 else sq+10) 0 && sq+10 <= 100) ||
  (optimalGameWinner n (if sq+11 > 100 then sq-100 else sq+11) 0 && sq+11 ≤ 100)

theorem playerA_wins_optimally : ∀ n, optimalGameWinner n 1 0 := sorry -- prove that A wins starting from square 1

end playerA_wins_optimally_l728_728864


namespace probability_of_subset_l728_728475

variables {Ω : Type} [ProbabilitySpace Ω] {A B : Event Ω}

theorem probability_of_subset (h : A ⊆ B) : P(B) ≥ P(A) :=
sorry

end probability_of_subset_l728_728475


namespace movie_theater_charge_l728_728221

theorem movie_theater_charge 
    (charge_adult : ℝ) 
    (children : ℕ) 
    (adults : ℕ) 
    (total_receipts : ℝ) 
    (charge_child : ℝ) 
    (condition1 : charge_adult = 6.75) 
    (condition2 : children = adults + 20) 
    (condition3 : total_receipts = 405) 
    (condition4 : children = 48) 
    : charge_child = 4.5 :=
sorry

end movie_theater_charge_l728_728221


namespace probability_heads_heads_l728_728077

theorem probability_heads_heads (h_uniform_density : ∀ outcome, outcome ∈ {("heads", "heads"), ("heads", "tails"), ("tails", "heads"), ("tails", "tails")} → True) :
  ℙ({("heads", "heads")}) = 1 / 4 :=
sorry

end probability_heads_heads_l728_728077


namespace construct_points_and_prove_relationship_l728_728507

noncomputable theory
open_locale classical

variables {Point : Type} [EuclideanGeometry Point]
variables (A B C D P Q R : Point)

def is_acute_angle (ABC : Angle) : Prop := 
ABC.isAcute

def is_ray_between (A B D : Point) : Prop := 
∃ (r : ℝ), r > 0 ∧ (B + r • (D - B) = A)

def points_on_rays (A B C P Q : Point) : Prop :=
∃ (s t : ℝ), s > 0 ∧ t > 0 ∧ (P = A + s • (B - A)) ∧ (Q = B + t • (C - B))

def infinite_ruler_construct (P Q R : Point) (BD : Ray Point) : Prop := 
∃ (BD_intersection : Point), BD_intersection = intersection_point (line_through BD) (line_through P Q) 
∧ segment_length (P, BD_intersection) = 2 * segment_length (Q, BD_intersection)

theorem construct_points_and_prove_relationship
(acute_ABC : is_acute_angle (∠ABC))
(interior_ray_BD : is_ray_between A B D)
(points_on_rays_BA_BC : points_on_rays A B C P Q)
(using_infinite_ruler : infinite_ruler_construct P Q R) :
segment_length(P, R) = 2 * segment_length(Q, R) :=
sorry

end construct_points_and_prove_relationship_l728_728507


namespace probability_of_three_heads_in_eight_tosses_l728_728980

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l728_728980


namespace largest_n_sum_positive_l728_728639

variables {a : ℕ → ℝ} (a_1_pos : a 1 > 0) (a_2013 : a 2013) (a_2014 : a 2014)
  (h1 : a 2013 + a 2014 > 0) (h2 : a 2013 * a 2014 < 0)

theorem largest_n_sum_positive (h : ∀ n, a (n+1) - a n = a 2 - a 1) :
  ∃ n : ℕ, n = 4026 ∧ ∑ i in finset.range (n+1), a i > 0 :=
by sorry

end largest_n_sum_positive_l728_728639


namespace find_fx_and_symmetry_and_monotonic_intervals_l728_728265

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 3) - sqrt 3

theorem find_fx_and_symmetry_and_monotonic_intervals :
  (∀ x, f x = sin (2 * x + π / 3) - sqrt 3) ∧
  (∀ k : ℤ, is_axis_of_symmetry (f x) (x = π / 12 + k * π / 2)) ∧
  (∀ k : ℤ, is_increasing_interval (f x) ([- 5 * π / 12 + k * π, π / 12 + k * π])) ∧
  (∀ k : ℤ, is_decreasing_interval (f x) ([π / 12 + k * π, 7 * π / 12 + k * π])) :=
by
  sorry

end find_fx_and_symmetry_and_monotonic_intervals_l728_728265


namespace volume_of_sphere_l728_728418

-- Definitions based on given problem condition
def vec3 := ℝ × ℝ × ℝ

def dot_product (u v : vec3) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def volume_of_solid (u : vec3) : ℝ :=
  if dot_product u u = dot_product u (6, -24, 12) then
    (4 / 3) * Real.pi * ((3 * Real.sqrt 21))^3
  else
    0

-- The proof statement
theorem volume_of_sphere (u : vec3) (h : dot_product u u = dot_product u (6, -24, 12)) :
  volume_of_solid u = 756 * Real.sqrt 21 * Real.pi :=
by
  sorry

end volume_of_sphere_l728_728418


namespace subtract_to_make_perfect_square_l728_728458

theorem subtract_to_make_perfect_square :
  ∃ x : ℕ, x^2 ≤ 92555 ∧ 92555 - (x^2) = 139 :=
begin
  sorry
end

end subtract_to_make_perfect_square_l728_728458


namespace merchant_problem_l728_728489

theorem merchant_problem (P C : ℝ) (h1 : P + C = 60) (h2 : 2.40 * P + 6.00 * C = 180) : C = 10 := 
by
  -- Proof goes here
  sorry

end merchant_problem_l728_728489


namespace range_of_dot_product_l728_728271

theorem range_of_dot_product
  (a b : ℝ)
  (h: ∃ (A B : ℝ × ℝ), (A ≠ B) ∧ ∃ m n : ℝ, A = (m, n) ∧ B = (-m, -n) ∧ m^2 + (n^2 / 9) = 1)
  : ∃ r : Set ℝ, r = (Set.Icc 41 49) :=
  sorry

end range_of_dot_product_l728_728271


namespace no_valid_operation_for_question_mark_l728_728858

-- Definitions
def operations := { "+", "-", "*", "/" }

-- Main statement to be proved
theorem no_valid_operation_for_question_mark : ¬ ∃ (op : String), op ∈ operations ∧ 
    match op with
    | "+" => (12 + 4) - 3 + (6 - 2) = 7
    | "-" => (12 - 4) - 3 + (6 - 2) = 7
    | "*" => (12 * 4) - 3 + (6 - 2) = 7
    | "/" => (12 / 4) - 3 + (6 - 2) = 7
    | _   => false :=
by 
    sorry

end no_valid_operation_for_question_mark_l728_728858


namespace ellipse_properties_l728_728508

-- Definitions based on the given conditions
def ellipse (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def minor_axis_length (b : ℝ) := 2 * b = 6

def eccentricity_relation (a b : ℝ) := a = (√2) * b

def quadratic_relation (a b : ℝ) := a^2 = b^2 + (a/√2)^2

-- Main theorem to prove the equivalent problem statement
theorem ellipse_properties :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
  ellipse a b x y ∧
  minor_axis_length b ∧
  eccentricity_relation a b ∧
  quadratic_relation a b ∧
  (∃ M N B1 B2 : ℝ × ℝ,
     (M ≠ B1 ∧ M ≠ B2) ∧
     (NB1 y' x' B1.1 B1.2 N.1 N.2) ⊥ (MB1 y x B1.1 B1.2 M.1 M.2) ∧
     (NB2 y' x' B2.1 B2.2 N.1 N.2) ⊥ (MB2 y x B2.1 B2.2 M.1 M.2) ∧
     max_area MB2 NB1 (B2.1 + B2.2) M N B1 B2 = 27 * (√2) / 2 )
:= sorry

end ellipse_properties_l728_728508


namespace remainder_1425_1427_1429_mod_12_l728_728457

theorem remainder_1425_1427_1429_mod_12 : 
  (1425 * 1427 * 1429) % 12 = 3 :=
by
  sorry

end remainder_1425_1427_1429_mod_12_l728_728457


namespace order_of_f_values_l728_728362

def f (x : ℝ) : ℝ := if x >= 1 then 3^x - 1 else sorry -- (we would need the definition for x < 1)

theorem order_of_f_values : 
  (∀ x, f (1 + x) = f (1 - x)) ∧ 
  (∀ x, x >= 1 → f x = 3^x - 1) → 
  f (2/3) < f (3/2) ∧ f (3/2) < f (1/3) := 
by
  sorry

end order_of_f_values_l728_728362


namespace exists_m_in_interval_l728_728648

theorem exists_m_in_interval (f : ℝ → ℝ) (h_domain : ∀ x, x ∈ set.Icc (-2 : ℝ) (2 : ℝ) → f x ∈ set.Icc (-2 : ℝ) (2 : ℝ))
  (h_increasing : ∀ x y, x < y → f x < f y)
  (h_inequality : ∀ m, x ∈ set.Icc (-2 : ℝ) (2 : ℝ) → f (1 - m) < f m) :
  ∃ m, (1 / 2 : ℝ) < m ∧ m ≤ 2 :=
begin
  -- Proof would go here
  sorry
end

end exists_m_in_interval_l728_728648


namespace partition_naturals_100_sets_l728_728817

-- Define the function t(n) as the highest power of 2 dividing n.
def t (n : ℕ) : ℕ := if n = 0 then 0 else (Nat.find (λ k => 2^k ≤ n ∧ ¬ 2^(k+1) ≤ n))

theorem partition_naturals_100_sets :
  ∃ (A : ℕ → ℕ), 
    (∀ n : ℕ, 0 < A n ∧ A n < 100) ∧ 
    (∀ a b c : ℕ, a + 99 * b = c → ¬(A a ≠ A b ∧ A a ≠ A c ∧ A b ≠ A c)) := 
sorry

end partition_naturals_100_sets_l728_728817


namespace max_average_daily_profit_total_profit_comparison_l728_728484

noncomputable def sales_volume (x : ℝ) : ℝ := 30 + 2 * (50 - x)
noncomputable def sales_revenue (x : ℝ) : ℝ := x * (sales_volume x)
noncomputable def daily_profit (x : ℝ) : ℝ := (x - 20) * (sales_volume x) - 400

theorem max_average_daily_profit:
  ∃ x : ℝ, 20 ≤ x ∧ x ≤ 50 ∧
  (∀ y : ℝ, 20 ≤ y → y ≤ 50 → daily_profit y ≤ daily_profit (85 / 2)) ∧
  daily_profit (85 / 2) = 612.5 :=
begin
  sorry
end

theorem total_profit_comparison:
  let x_max_profit := 85 / 2,
      x_highest_price := 50,
      daily_sales_volume_max_profit := sales_volume x_max_profit,
      daily_sales_volume_highest_price := sales_volume x_highest_price,
      sales_days_max_profit := 2000 / daily_sales_volume_max_profit,
      sales_days_highest_price := 2000 / daily_sales_volume_highest_price,
      total_profit_max_profit := daily_profit x_max_profit * sales_days_max_profit,
      total_profit_highest_price := (x_highest_price - 20) * 2000 - sales_days_highest_price * 400 in
  total_profit_max_profit = 27562.5 ∧ total_profit_highest_price = 33200 :=
begin
  sorry
end

end max_average_daily_profit_total_profit_comparison_l728_728484


namespace square_pentagon_side_ratio_l728_728127

noncomputable def tan54 : ℝ := Real.tan (Real.pi * 54 / 180)

theorem square_pentagon_side_ratio 
  (s_s s_p : ℝ) 
  (h : s_s^2 = (5 * s_p^2 * tan54) / 4) : 
  s_s / s_p ≈ 1.3115 :=
begin
  sorry
end

end square_pentagon_side_ratio_l728_728127


namespace find_xyz_l728_728258

def divisible_by (n k : ℕ) : Prop := k % n = 0

def is_7_digit_number (a b c d e f g : ℕ) : ℕ := 
  10^6 * a + 10^5 * b + 10^4 * c + 10^3 * d + 10^2 * e + 10 * f + g

theorem find_xyz
  (x y z : ℕ)
  (h : divisible_by 792 (is_7_digit_number 1 4 x y 7 8 z))
  : (100 * x + 10 * y + z) = 644 :=
by
  sorry

end find_xyz_l728_728258


namespace consecutive_integers_sequence_l728_728815

theorem consecutive_integers_sequence :
  ∃ (a : ℤ), ∀ (seq : List ℤ), 
    seq = [a-3, a-2, a-1, a, a+1, a+2, a+3] →
    let swapped_seq := List.updateNth (List.updateNth seq 0 (List.get! seq 6)) 6 (List.get! seq 0) in
    let moved_middle_seq := a :: swapped_seq.tail! in
    let final_seq := moved_middle_seq.take 3 ++ [moved_middle_seq.head!] ++ moved_middle_seq.drop 3 in
    final_seq[3] = a+3 → abs (final_seq[2]) = abs (final_seq[3]) →
    seq = [-3, -2, -1, 0, 1, 2, 3] :=
sorry

end consecutive_integers_sequence_l728_728815


namespace max_sequence_term_value_l728_728761

def a_n (n : ℕ) : ℤ := -2 * n^2 + 29 * n + 3

theorem max_sequence_term_value : ∃ n : ℕ, a_n n = 108 := 
sorry

end max_sequence_term_value_l728_728761


namespace remaining_money_after_shopping_l728_728329

/-- Jerry's grocery shopping problem setup -/
def budget : ℝ := 100
def mustard_oil_price_per_liter : ℝ := 13
def mustard_oil_quantity : ℝ := 2
def mustard_oil_discount : ℝ := 0.10
def penne_pasta_price_per_pound : ℝ := 4
def penne_pasta_quantity : ℝ := 3
def pasta_sauce_price_per_pound : ℝ := 5
def pasta_sauce_quantity : ℝ := 1

/-- Jerry's expected remaining money after grocery shopping -/
def expected_remaining_money_after_shopping : ℝ := 63.60

theorem remaining_money_after_shopping : 
  let cost_mustard_oil := mustard_oil_quantity * mustard_oil_price_per_liter in
  let discount_mustard_oil := mustard_oil_discount * cost_mustard_oil in
  let total_cost_mustard_oil := cost_mustard_oil - discount_mustard_oil in
  let cost_penne_pasta := 2 * penne_pasta_price_per_pound in
  let total_cost_penne_pasta := cost_penne_pasta in
  let total_cost_pasta_sauce := pasta_sauce_quantity * pasta_sauce_price_per_pound in
  let total_cost := total_cost_mustard_oil + total_cost_penne_pasta + total_cost_pasta_sauce in
  let remaining_money := budget - total_cost in
  remaining_money = expected_remaining_money_after_shopping := by
  sorry

end remaining_money_after_shopping_l728_728329


namespace download_time_ratio_l728_728368

-- Define the conditions of the problem
def mac_download_time : ℕ := 10
def audio_glitches : ℕ := 2 * 4
def video_glitches : ℕ := 6
def time_with_glitches : ℕ := audio_glitches + video_glitches
def time_without_glitches : ℕ := 2 * time_with_glitches
def total_time : ℕ := 82

-- Define the Windows download time as a variable
def windows_download_time : ℕ := total_time - (mac_download_time + time_with_glitches + time_without_glitches)

-- Prove the required ratio
theorem download_time_ratio : 
  (windows_download_time / mac_download_time = 3) :=
by
  -- Perform a straightforward calculation as defined in the conditions and solution steps
  sorry

end download_time_ratio_l728_728368


namespace find_a2_b2_l728_728234

noncomputable def imaginary_unit : ℂ := Complex.I

theorem find_a2_b2 (a b : ℝ) (h1 : (a - 2 * imaginary_unit) * imaginary_unit = b - imaginary_unit) : a^2 + b^2 = 5 :=
  sorry

end find_a2_b2_l728_728234


namespace rectangle_width_l728_728300

theorem rectangle_width (w : ℝ) 
  (h1 : ∃ w : ℝ, w > 0 ∧ (2 * w + 2 * (w - 2)) = 16) 
  (h2 : ∀ w, w > 0 → 2 * w + 2 * (w - 2) = 16 → w = 5) : 
  w = 5 := 
sorry

end rectangle_width_l728_728300


namespace solve_system_of_equations_l728_728821

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x^2 + y * real.sqrt (x * y) = 105) ∧ (y^2 + x * real.sqrt (y * x) = 70) ∧ (x = 9 ∧ y = 4) :=
by {
  have key_eq : ∀ x y : ℝ, x^2 + y * real.sqrt (x * y) = 105 ∧ y^2 + x * real.sqrt (y * x) = 70 → (x = 9 ∧ y = 4),
  { intros x y h,
    sorry },
  use 9,
  use 4,
  split,
  norm_num,
  split,
  norm_num,
  split,
  norm_num,
  split,
  norm_num,
  exact key_eq 9 4 ⟨rfl, rfl⟩,
}

end solve_system_of_equations_l728_728821


namespace total_amount_of_money_l728_728476

open Real

theorem total_amount_of_money (a b c total first_part : ℝ)
  (h1 : a / b = 1 / 2)
  (h2 : b / c = 1 / 3)
  (h3 : c / total = 3 / 4)
  (h4 : a = 246.95) :
  total ≈ 782.06 :=
by 
  sorry

end total_amount_of_money_l728_728476


namespace binom_7_4_l728_728158

theorem binom_7_4 : Nat.choose 7 4 = 35 := 
by
  sorry

end binom_7_4_l728_728158


namespace probability_sum_exceeds_ten_l728_728729

def two_dice_probability : ℚ :=
  let total_events := 36
  let favorable_events := 3
  favorable_events / total_events

theorem probability_sum_exceeds_ten :
  two_dice_probability = 1 / 12 :=
by
  rw [two_dice_probability]
  norm_num
  sorry

end probability_sum_exceeds_ten_l728_728729


namespace quadratic_distinct_real_roots_l728_728273

theorem quadratic_distinct_real_roots (a : ℝ) (h : a ≠ 1) : 
(a < 2) → 
(∃ x y : ℝ, x ≠ y ∧ (a-1)*x^2 - 2*x + 1 = 0 ∧ (a-1)*y^2 - 2*y + 1 = 0) :=
sorry

end quadratic_distinct_real_roots_l728_728273


namespace factorize_expression_l728_728590

theorem factorize_expression (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l728_728590


namespace prob_three_heads_in_eight_tosses_l728_728954

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l728_728954


namespace angle_between_vectors_eq_pi_over_4_l728_728280

open Real EuclideanGeometry

variables (a b : EuclideanSpace ℝ ℝ)
variables (ha : ‖a‖ = sqrt 2) (hb : ‖b‖ = 2)
variables (hperp : (a - b) ⬝ a = 0)

theorem angle_between_vectors_eq_pi_over_4 (a b : EuclideanSpace ℝ ℝ)
  (ha : ‖a‖ = sqrt 2) (hb : ‖b‖ = 2) (hperp : (a - b) ⬝ a = 0) :
  angle a b = π / 4 :=
sorry

end angle_between_vectors_eq_pi_over_4_l728_728280


namespace max_value_of_quadratic_l728_728289

theorem max_value_of_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : 
  ∃ y, y = x * (1 - 2 * x) ∧ y ≤ 1 / 8 ∧ (y = 1 / 8 ↔ x = 1 / 4) :=
by sorry

end max_value_of_quadratic_l728_728289


namespace coeff_x9_in_expansion_l728_728065

noncomputable def binomial_coeff (n k : ℕ) : ℤ :=
  if h : k ≤ n then nat.choose n k else 0

theorem coeff_x9_in_expansion : 
  coefficient (x^9) (expand (1 - 3 * x^3) ^ 6) = -540 :=
by
  sorry

end coeff_x9_in_expansion_l728_728065


namespace arithmetic_mean_of_remaining_numbers_l728_728743

-- Definitions and conditions
def initial_set_size : ℕ := 60
def initial_arithmetic_mean : ℕ := 45
def numbers_to_remove : List ℕ := [50, 55, 60]

-- Calculation of the total sum
def total_sum : ℕ := initial_arithmetic_mean * initial_set_size

-- Calculation of the sum of the numbers to remove
def sum_of_removed_numbers : ℕ := numbers_to_remove.sum

-- Sum of the remaining numbers
def new_sum : ℕ := total_sum - sum_of_removed_numbers

-- Size of the remaining set
def remaining_set_size : ℕ := initial_set_size - numbers_to_remove.length

-- The arithmetic mean of the remaining numbers
def new_arithmetic_mean : ℚ := new_sum / remaining_set_size

-- The proof statement
theorem arithmetic_mean_of_remaining_numbers :
  new_arithmetic_mean = 2535 / 57 :=
by
  sorry

end arithmetic_mean_of_remaining_numbers_l728_728743


namespace probability_of_exactly_three_heads_l728_728930

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l728_728930


namespace smallest_integer_with_mod_inverse_l728_728439

theorem smallest_integer_with_mod_inverse :
  ∃ n : ℕ, n > 2 ∧ n.gcd 735 = 1 ∧ ∀ m : ℕ, m > 2 ∧ m.gcd 735 = 1 → m ≥ n :=
begin
  use 4,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  { intros m hm hmgcd,
    by_cases h : m = 4,
    { exact (by rwa [h]), },
    { exact dec_trivial, } }
end

end smallest_integer_with_mod_inverse_l728_728439


namespace concyclic_KK_MK_eq_ML_l728_728356

open EuclideanGeometry

variables {A B C D X K K' L L' M : Point}

theorem concyclic_KK'_LL' (h_triangle : ∠ B C A = 90)
  (h_altitude : foot D A C B)
  (h_X_on_CD : X ∈ segment C D)
  (h_BK_eq_BC : BK = BC)
  (h_AL_eq_AC : AL = AC)
  (h_intersect_M : ∃ (M : Point), M ∈ line A L ∧ M ∈ line B K) :
  concyclic K K' L L' :=
sorry

theorem MK_eq_ML (h_concyclic : concyclic K K' L L')
  (h_intersect_M : ∃ (M : Point), M ∈ line A L ∧ M ∈ line B K) :
  MK = ML :=
sorry

end concyclic_KK_MK_eq_ML_l728_728356


namespace sqrt_720_simplified_l728_728011

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  have h1 : 720 = 2^4 * 3^2 * 5 := by norm_num
  -- Here we use another proven fact or logic per original conditions and definition
  sorry

end sqrt_720_simplified_l728_728011


namespace even_function_l728_728023

-- Definition of the given function
def f (x : ℝ) : ℝ := sin ((2005 / 2) * Real.pi - 2004 * x)

-- Statement that the function is even
theorem even_function : ∀ x : ℝ, f(-x) = f(x) := by
  sorry

end even_function_l728_728023


namespace probability_exactly_three_heads_l728_728969
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l728_728969


namespace quadratic_root_zero_l728_728615

theorem quadratic_root_zero (k : ℝ) :
  (∃ x : ℝ, (k + 2) * x^2 + 6 * x + k^2 + k - 2 = 0) →
  (∃ x : ℝ, x = 0 ∧ ((k + 2) * x^2 + 6 * x + k^2 + k - 2 = 0)) →
  k = 1 :=
by
  sorry

end quadratic_root_zero_l728_728615


namespace simplify_complex_expression_l728_728819

theorem simplify_complex_expression : (2 - 3 * complex.I) ^ 3 = -46 - 9 * complex.I := by
  have h1 : complex.I ^ 2 = -1 := by
    sorry
  sorry

end simplify_complex_expression_l728_728819


namespace number_of_teachers_l728_728740

-- Defining the conditions
variables (T S : ℕ) -- the number of teachers and students

-- Total number of teachers and students
axiom population_eq : T + S = 2400

-- Fraction of students in the sample
axiom sample_fraction : (150 / 160 : ℚ) = (S / 2400 : ℚ)

-- Prove the number of teachers is 150
theorem number_of_teachers : T = 150 :=
by {
    sorry, -- omitting the proof 
}

end number_of_teachers_l728_728740


namespace find_numbers_l728_728430

theorem find_numbers (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (geom_mean_cond : Real.sqrt (a * b) = Real.sqrt 5)
  (harm_mean_cond : 2 / ((1 / a) + (1 / b)) = 2) :
  (a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2) ∨
  (a = (5 - Real.sqrt 5) / 2 ∧ b = (5 + Real.sqrt 5) / 2) :=
by
  sorry

end find_numbers_l728_728430


namespace triangle_side_probability_l728_728620

open Finset

-- We define the set of vertices of a regular decagon as {1, 2, ... ,10}
def vertices : Finset ℕ := (range 10).map Nat.succ

-- Counting the total number of ways to choose 3 vertices from 10
def total_triangles : ℕ := Nat.choose 10 3

-- Function to count favorable outcomes
def favorable_outcomes : ℕ :=
  let one_side := 10 * 6 -- one side is also a side of the decagon
  let two_sides := 10    -- two sides are also sides of the decagon
  one_side + two_sides

-- Probability calculation
def at_least_one_side_probability : ℚ :=
  favorable_outcomes / total_triangles

-- Proof problem statement
theorem triangle_side_probability : at_least_one_side_probability = 7 / 12 := sorry

end triangle_side_probability_l728_728620


namespace probability_of_valid_p_probability_of_valid_p_fraction_l728_728691

def satisfies_equation (p q : ℤ) : Prop := p * q - 6 * p - 3 * q = 3

def valid_p (p : ℤ) : Prop := ∃ q : ℤ, satisfies_equation p q

theorem probability_of_valid_p :
  (finset.filter valid_p (finset.Icc 1 15)).card = 4 :=
sorry

theorem probability_of_valid_p_fraction :
  (finset.filter valid_p (finset.Icc 1 15)).card / 15 = 4 / 15 :=
begin
  have h : (finset.filter valid_p (finset.Icc 1 15)).card = 4 := probability_of_valid_p,
  rw h,
  norm_num,
end

end probability_of_valid_p_probability_of_valid_p_fraction_l728_728691


namespace seashells_count_l728_728367

theorem seashells_count (mary_seashells : ℕ) (keith_seashells : ℕ) (cracked_seashells : ℕ) 
  (h_mary : mary_seashells = 2) (h_keith : keith_seashells = 5) (h_cracked : cracked_seashells = 9) :
  (mary_seashells + keith_seashells = 7) ∧ (cracked_seashells > mary_seashells + keith_seashells) → false := 
by {
  sorry
}

end seashells_count_l728_728367


namespace crackers_given_to_friends_l728_728369

theorem crackers_given_to_friends (crackers_per_friend : ℕ) (number_of_friends : ℕ) (h1 : crackers_per_friend = 6) (h2 : number_of_friends = 6) : (crackers_per_friend * number_of_friends) = 36 :=
by
  sorry

end crackers_given_to_friends_l728_728369


namespace units_digit_of_product_of_skipping_odds_l728_728440

def odd_seq_skip_every_second (start end : ℕ) : List ℕ :=
  List.filterMap (λ n, if (21 + 4 * n ≥ start) ∧ (21 + 4 * n ≤ end) then some (21 + 4 * n) else none) (List.range 20)

def units_digit (n : ℕ) : ℕ :=
  n % 10

def product (lst : List ℕ) : ℕ :=
  lst.foldr (*) 1

theorem units_digit_of_product_of_skipping_odds :
  units_digit (product (odd_seq_skip_every_second 20 100)) = 5 :=
sorry

end units_digit_of_product_of_skipping_odds_l728_728440


namespace ratio_of_perimeters_l728_728687

-- Define lengths of the rectangular patch
def length_rect : ℝ := 400
def width_rect : ℝ := 300

-- Define the length of the side of the square patch
def side_square : ℝ := 700

-- Define the perimeters of both patches
def P_square : ℝ := 4 * side_square
def P_rectangle : ℝ := 2 * (length_rect + width_rect)

-- Theorem stating the ratio of the perimeters
theorem ratio_of_perimeters : P_square / P_rectangle = 2 :=
by sorry

end ratio_of_perimeters_l728_728687


namespace find_b_l728_728573

def h(x : ℝ) : ℝ := 5 * x - 10

theorem find_b (b : ℝ) : h(b) = 0 → b = 2 :=
by 
  sorry

end find_b_l728_728573


namespace number_of_divisors_greater_than_eight_factorial_l728_728286

-- Definitions based on conditions
def nine_factorial := fact 9
def eight_factorial := fact 8

-- Theorem statement
theorem number_of_divisors_greater_than_eight_factorial :
  ∃ (n : ℕ), n = 8 ∧ ∀ d : ℕ, (d ∣ nine_factorial ∧ d > eight_factorial) → n = 8 :=
by
  sorry

end number_of_divisors_greater_than_eight_factorial_l728_728286


namespace negation_of_p_l728_728666

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 := by
  sorry

end negation_of_p_l728_728666


namespace ramesh_installation_cost_l728_728811

noncomputable def labelled_price (discounted_price : ℝ) (discount_rate : ℝ) : ℝ :=
  discounted_price / (1 - discount_rate)

noncomputable def selling_price (labelled_price : ℝ) (profit_rate : ℝ) : ℝ :=
  labelled_price * (1 + profit_rate)

def ramesh_total_cost (purchase_price transport_cost : ℝ) (installation_cost : ℝ) : ℝ :=
  purchase_price + transport_cost + installation_cost

theorem ramesh_installation_cost :
  ∀ (purchase_price discounted_price transport_cost labelled_price profit_rate selling_price installation_cost : ℝ),
  discounted_price = 12500 → transport_cost = 125 → profit_rate = 0.18 → selling_price = 18880 →
  labelled_price = discounted_price / (1 - 0.20) →
  selling_price = labelled_price * (1 + profit_rate) →
  ramesh_total_cost purchase_price transport_cost installation_cost = selling_price →
  installation_cost = 6255 :=
by
  intros
  sorry

end ramesh_installation_cost_l728_728811


namespace attendees_not_from_companies_l728_728113

theorem attendees_not_from_companies :
  let A := 30 
  let B := 2 * A
  let C := A + 10
  let D := C - 5
  let T := 185 
  T - (A + B + C + D) = 20 :=
by
  sorry

end attendees_not_from_companies_l728_728113


namespace sin_double_angle_l728_728256

open Real

theorem sin_double_angle (α : ℝ) (h : tan α = -3/5) : sin (2 * α) = -15/17 :=
by
  -- We are skipping the proof here
  sorry

end sin_double_angle_l728_728256


namespace f_2004_eq_1320_l728_728039

noncomputable def f : ℕ → ℝ := sorry

theorem f_2004_eq_1320 :
  (∀ (a b n : ℕ), a + b = 2^(n + 1) → f a + f b = n^3) →
  f 2004 = 1320 :=
by
  intro h
  have h4 : f 4 = 4 := by
    have := h 4 4 1
    rw [add_comm] at this
    linarith
  have h12 : f 12 = 4^3 - 4 := by
    have := h 12 20 4
    rw [add_comm] at this
    linarith [h4]
  have h20 : f 20 = 5^3 - 65 := by
    have := h 20 32 5
    rw [add_comm] at this
    linarith [h12]
  have h52 : f 52 = 6^3 - 151 := by
    have := h 52 2004 11
    rw [add_comm] at this
    linarith [h20]
  have h2004 : f 2004 = 11^3 - 151 := by
    have := h 2004 4096 12
    rw [add_comm] at this
    linarith [h52]
  linarith [h2004]

end f_2004_eq_1320_l728_728039


namespace CP_bisects_angle_DCE_l728_728673

-- Definitions of the geometric setup
structure Point where
  x y z : ℝ

structure Plane where
  normal : Point
  point : Point

def orthogonal (π1 π2 : Plane) : Prop :=
  π1.normal.x * π2.normal.x + π1.normal.y * π2.normal.y + π1.normal.z * π2.normal.z = 0

def distinct (A B : Point) : Prop :=
  A ≠ B

def line_intersection(π1 π2 : Plane) : set Point :=
  { P | ∃ λ μ : ℝ, P = {x := λ * π1.normal.x + μ * π2.normal.x,
                         y := λ * π1.normal.y + μ * π2.normal.y,
                         z := λ * π1.normal.z + μ * π2.normal.z} }

def on_plane (π : Plane) (P : Point) : Prop :=
  π.normal.x * P.x + π.normal.y * P.y + π.normal.z * P.z = π.normal.x * π.point.x + π.normal.y * π.point.y + π.normal.z * π.point.z

def angle_bisector (B C A : Point) (P : Point) : Prop :=
  sorry -- Details of defining angle bisector

def circumference (π : Plane) (A B : Point) : set Point :=
  sorry -- Details for circumference on π with diameter AB

noncomputable def intersection (π1 π2 : Plane) (S : set Point) : set Point :=
  { P | on_plane π1 P ∧ P ∈ S }

-- Main theorem
theorem CP_bisects_angle_DCE
  (π1 π2 : Plane) (A B C P : Point) (π3 : Plane) (D E : Point)
  (h_ortho : orthogonal π1 π2)
  (h_distinct : distinct A B)
  (h_C_on_π2_not_π1 : on_plane π2 C ∧ ¬ on_plane π1 C)
  (h_P_bisector : angle_bisector B C A P)
  (h_S : ∀ P', P' ∈ circumference π1 A B ↔ ∃ t : ℝ, P' = sorry)
  (h_π3_contains_CP : on_plane π3 C ∧ ∃ t : ℝ, C + t * sorry = P)
  (h_DE : D ≠ E ∧ ∀ P, P ∈ intersection π3 (circumference π1 A B) → P = D ∨ P = E) :
  angle_bisector D C E P :=
sorry

end CP_bisects_angle_DCE_l728_728673


namespace road_completion_days_l728_728509

variable (L : ℕ) (M_1 : ℕ) (W_1 : ℕ) (t1 : ℕ) (M_2 : ℕ)

theorem road_completion_days : L = 10 ∧ M_1 = 30 ∧ W_1 = 2 ∧ t1 = 5 ∧ M_2 = 60 → D = 15 :=
by
  sorry

end road_completion_days_l728_728509


namespace diameter_correct_l728_728830

noncomputable def diameter_of_circle (C : ℝ) (hC : C = 36) : ℝ :=
  let r := C / (2 * Real.pi)
  2 * r

theorem diameter_correct (C : ℝ) (hC : C = 36) : diameter_of_circle C hC = 36 / Real.pi := by
  sorry

end diameter_correct_l728_728830


namespace cubic_polynomial_roots_l728_728640

theorem cubic_polynomial_roots (x y z s t : ℝ)
  (h1 : x + y = s)
  (h2 : xy = t^2)
  (h3 : z = x - 3) :
  (x^3 - sx^2 + (3s + t^2)x - 3t^2 = 0) :=
sorry

end cubic_polynomial_roots_l728_728640


namespace probability_of_three_heads_in_eight_tosses_l728_728986

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l728_728986


namespace coin_toss_probability_l728_728946

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l728_728946


namespace range_of_k_l728_728662

open Real

theorem range_of_k (k : ℝ) (h₁ : 0 < k)
  (h₂ : ∀ A B : ℝ×ℝ, (A ≠ B) → 
        (∃ p1 p2, x.component p1 A ∧ y.component p2 B ∧ p1 ≠ p2 ∧ p1.onCircle 2 ∧ p2.onCircle 2 ∧ 
        (∀ t : ℝ, x.linear t y p1 k p2 → t.intersect O)
        → |(vec OA) + (vec OB)| >= (sqrt(3)/3) * |(vec AB)|) 
  : ∃ k, sqrt(2) <= k ∧ k < 2 * sqrt(2) := 
begin
  sorry,
end

end range_of_k_l728_728662


namespace minimum_value_of_expression_l728_728890

theorem minimum_value_of_expression (x y z w : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) (h : 5 * w = 3 * x ∧ 5 * w = 4 * y ∧ 5 * w = 7 * z) : x - y + z - w = 11 :=
sorry

end minimum_value_of_expression_l728_728890


namespace number_of_sets_l728_728028

theorem number_of_sets (M : Set ℕ) :
  (\{1, 2\} ∪ M = \{1, 2, 3\}) → (∃ S : Finset (Set ℕ), S.card = 4) :=
by
  sorry

end number_of_sets_l728_728028


namespace range_of_values_for_a_l728_728251

variable {α : Type*}
variable (x a : α)

-- Assumptions
axiom p : x > 2
axiom q : x > a
axiom necessary_but_not_sufficient : ∀ {x : α}, (x > 2 → x > a) ∧ ¬(x > a → x > 2)

theorem range_of_values_for_a : a > 2 :=
by
  sorry

end range_of_values_for_a_l728_728251


namespace probability_of_exactly_three_heads_l728_728929

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l728_728929


namespace combined_tax_rate_is_29_percent_l728_728889

-- Definitions for the conditions
def mork_income := X : ℝ
def mindy_income := 4 * mork_income
def mork_tax_rate := 0.45
def mindy_tax_rate := 0.25

-- Questions to answer given the conditions
theorem combined_tax_rate_is_29_percent :
  let mork_tax := mork_tax_rate * mork_income in
  let mindy_tax := mindy_tax_rate * mindy_income in
  let combined_tax_paid := mork_tax + mindy_tax in
  let combined_income := mork_income + mindy_income in
  (combined_tax_paid / combined_income) * 100 = 29 := 
by 
  -- Proof omitted
  sorry

end combined_tax_rate_is_29_percent_l728_728889


namespace find_common_ratio_l728_728756

variable (a : ℕ → ℝ)
variable (q : ℝ)

def arithmetic_mean (x y z : ℝ) : Prop := 2 * x = y + z

def is_acute_angle (A : ℝ × ℝ) (B : ℝ × ℝ) : Prop := 
  let vA := A.1
  let vB := A.2
  let vC := B.1
  let vD := B.2
  -vA * 1 - vB + 1 > 0

theorem find_common_ratio :
  (arithmetic_mean (a 7) (a 8) (a 9)) ∧ is_acute_angle (1, 1) (2, q) → q = -2 :=
by
  sorry

end find_common_ratio_l728_728756


namespace probability_exactly_three_heads_l728_728967
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l728_728967


namespace log_domain_inequality_l728_728723

theorem log_domain_inequality {a : ℝ} : 
  (∀ x : ℝ, x^2 + 2 * x + a > 0) ↔ a > 1 :=
sorry

end log_domain_inequality_l728_728723


namespace net_profit_is_90_l728_728518

theorem net_profit_is_90
    (cost_seeds cost_soil : ℝ)
    (num_plants : ℕ)
    (price_per_plant : ℝ)
    (h0 : cost_seeds = 2)
    (h1 : cost_soil = 8)
    (h2 : num_plants = 20)
    (h3 : price_per_plant = 5) :
    (num_plants * price_per_plant - (cost_seeds + cost_soil)) = 90 := by
  sorry

end net_profit_is_90_l728_728518


namespace two_coins_heads_probability_l728_728080

/-- 
When tossing two coins of uniform density, the probability that both coins land with heads facing up is 1/4.
-/
theorem two_coins_heads_probability : 
  let outcomes := ["HH", "HT", "TH", "TT"]
  let favorable := "HH"
  probability (favorable) = 1/4 :=
by
  sorry

end two_coins_heads_probability_l728_728080


namespace bicycle_race_permutations_l728_728503

theorem bicycle_race_permutations :
  let n := 4 in
  (n.factorial = 24) :=
by
  sorry

end bicycle_race_permutations_l728_728503


namespace factorize_expression_l728_728589

theorem factorize_expression (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l728_728589


namespace unpaid_parking_lots_l728_728511

noncomputable def calculate_unpaid_cars
  (lot_a_total : ℕ) (lot_a_valid_percent : ℚ) (lot_a_perm_fraction : ℚ)
  (lot_b_total : ℕ) (lot_b_valid_percent : ℚ) (lot_b_perm_fraction : ℚ)
  (lot_c_total : ℕ) (lot_c_valid_percent : ℚ) (lot_c_perm_fraction : ℚ) : ℕ :=
  
  let lot_a_valid := floor (lot_a_valid_percent * lot_a_total)
  let lot_a_perm := floor (lot_a_perm_fraction * lot_a_valid)
  let lot_a_unpaid := lot_a_total - (lot_a_valid + lot_a_perm)

  let lot_b_valid := floor (lot_b_valid_percent * lot_b_total)
  let lot_b_perm := floor (lot_b_perm_fraction * lot_b_valid)
  let lot_b_unpaid := lot_b_total - (lot_b_valid + lot_b_perm)

  let lot_c_valid := floor (lot_c_valid_percent * lot_c_total)
  let lot_c_perm := floor (lot_c_perm_fraction * lot_c_valid)
  let lot_c_unpaid := lot_c_total - (lot_c_valid + lot_c_perm)

  lot_a_unpaid + lot_b_unpaid + lot_c_unpaid

theorem unpaid_parking_lots :
  calculate_unpaid_cars 300 (75 / 100) (1 / 5)
                        450 (65 / 100) (1 / 4)
                        600 (80 / 100) (1 / 10) = 187 := by
  sorry

end unpaid_parking_lots_l728_728511


namespace g_2_power_4_l728_728824

-- Definitions for the conditions
variables (f g : ℝ → ℝ)

-- Lean assumptions corresponding to the conditions
axiom f_g_x (x : ℝ) (hx : x ≥ 1) : f (g x) = 2 * x^2
axiom g_f_x (x : ℝ) (hx : x ≥ 1) : g (f x) = x^4
axiom g_4 : g 4 = 16

-- Target statement to prove
theorem g_2_power_4 : [g 2]^4 = 16 :=
by sorry

end g_2_power_4_l728_728824


namespace probability_of_three_heads_in_eight_tosses_l728_728983

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l728_728983


namespace verify_chord_and_distance_l728_728478

open Real

-- Define the hyperbola equation
def hyperbola (x y : Real) : Prop := (x^2) / 9 - (y^2) / 16 = 1

-- Define the right focus F of the hyperbola
def focus_F : (Real × Real) := (5, 0)

-- Define the line AB passing through the focus F and inclined at an angle of π/4
def line_AB (x y : Real) : Prop := y = x - 5

-- Define the distance formula
def distance (p q : (Real × Real)) : Real :=
  sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- The length of the chord AB if it passes through the focus F and inclined at π/4
def length_chord_AB : Real := 10 * sqrt 2

-- The distance from the midpoint C of the chord to the focus F
def distance_CF : Real := (80 * sqrt 2) / 7

-- Mathematical proof problem to verify the required conditions

theorem verify_chord_and_distance :
    (∃ A B : (ℝ × ℝ), (hyperbola A.1 A.2 ∧ line_AB A.1 A.2) ∧
                       (hyperbola B.1 B.2 ∧ line_AB B.1 B.2) ∧
                        distance A B = length_chord_AB) ∧
    (∃ C : (Real × Real), C = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧ distance C focus_F = distance_CF) :=
sorry

end verify_chord_and_distance_l728_728478


namespace final_answer_l728_728771

variable (A B C D I E : Type)
variable (AB AC BC AD CD DE : ℚ)
variable (angleBAC : ℝ)
variable (incenterI : A)
variable (intersectionE : A)
variable (a b c : ℕ)

-- Conditions
axiom h1 : AB = 5
axiom h2 : AC = 8
axiom h3 : BC = 7
axiom h4 : AD = 5
axiom h5 : CD = 3
axiom h6 : incenterI = I
axiom h7 : intersectionE = E
axiom h8 : DE = (a * Real.sqrt b) / c
axiom h9 : Nat.coprime a c
axiom h10 : ∀ p : Nat.prime, ¬ (p ^ 2).is_dvd b
axiom h11 : ∠BAC = π / 3 -- Angle in radians which is 60°

theorem final_answer : a + b + c = 13 := by
  sorry

end final_answer_l728_728771


namespace ten_factorial_minus_nine_factorial_l728_728567

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem ten_factorial_minus_nine_factorial :
  factorial 10 - factorial 9 = 3265920 := 
by 
  sorry

end ten_factorial_minus_nine_factorial_l728_728567


namespace find_g_five_l728_728363

theorem find_g_five (g : ℝ → ℝ) (h : ∀ x : ℝ, g(x) + 2 * g(2 - x) = 4 * x^3 - x^2) : 
  g(5) = -709 / 3 :=
by
  sorry

end find_g_five_l728_728363


namespace find_a_solve_inequality_l728_728656

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ x

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f a 2 = 8 * f a (-1)) : a = 2 ∨ a = 1 / 2 :=
by
  sorry

theorem solve_inequality (x : ℝ) (h4 : 2 ^ (x^2 - 2 * x - 3) < 0) : x ∈ Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo (3 : ℝ) ⊤ :=
by
  sorry

end find_a_solve_inequality_l728_728656


namespace contrapositive_l728_728831

-- Given proposition: If triangle ABC is an isosceles triangle, then its any two interior angles are not equal.
axiom isosceles_triangle (ABC : Triangle) : Prop
axiom interior_angles_not_equal (ABC : Triangle) : Prop

-- Goal: Prove the contrapositive
theorem contrapositive (ABC : Triangle) :
  (∀ (ABC : Triangle), isosceles_triangle ABC → ¬(interior_angles_not_equal ABC)) ↔
  (∀ (ABC : Triangle), (¬interior_angles_not_equal ABC) → isosceles_triangle ABC) :=
sorry

end contrapositive_l728_728831


namespace task2_probability_l728_728449

theorem task2_probability :
  ∀ (P1 P1_not_P2 P2 : ℝ),
    P1 = 2/3 →
    P1_not_P2 = 4/15 →
    (P1 * (1 - P2) = P1_not_P2) →
    P2 = 3/5 :=
by
  intros P1 P1_not_P2 P2 h1 h2 h3
  linarith

end task2_probability_l728_728449


namespace trigonometric_identity_l728_728705

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trigonometric_identity_l728_728705


namespace no_such_m_for_equivalence_existence_of_m_for_implication_l728_728235

def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem no_such_m_for_equivalence :
  ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S x m :=
sorry

theorem existence_of_m_for_implication :
  ∃ m : ℝ, (∀ x : ℝ, S x m → P x) ∧ m ≤ 3 :=
sorry

end no_such_m_for_equivalence_existence_of_m_for_implication_l728_728235


namespace intersection_eq_set_l728_728278

noncomputable def M : Set ℤ := {x : ℤ | x < 3}
noncomputable def N : Set ℝ := {x : ℝ | 1 ≤ exp x ∧ exp x ≤ exp 1}

theorem intersection_eq_set :
  M ∩ {y : ℤ | ∃ x : ℝ, (x = ↑y) ∧ x ∈ N} = {0, 1} :=
by
  sorry

end intersection_eq_set_l728_728278


namespace probability_three_heads_in_eight_tosses_l728_728938

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l728_728938


namespace combination_seven_four_l728_728148

theorem combination_seven_four : nat.choose 7 4 = 35 :=
by
  sorry

end combination_seven_four_l728_728148


namespace coefficient_x2_expansion_l728_728753

open Nat

theorem coefficient_x2_expansion : 
  (∑ k in finset.range 17, binom (k + 3) 2) = 1139 := 
by
  sorry

end coefficient_x2_expansion_l728_728753


namespace length_of_room_length_of_room_l728_728404

-- Definitions of given conditions
def width : ℝ := 3.75
def cost_per_square_meter : ℝ := 400
def total_paving_cost : ℝ := 8250

-- Target to prove that the length is 5.5 meters
theorem length_of_room : width * 5.5 = total_paving_cost / cost_per_square_meter := by
  define length (total cost / cost per square meter)
  calc
    multiply width by length gives correct area
    all conditions hold true
    conclusion hence length is as calculated
add proof here

theorem length_of_room : (total_paving_cost / cost_per_square_meter) / width = 5.5 := by
  calc
    (total_paving_cost / cost_per_square_meter) / width = 20.625 / 3.75 : by sorry
    ... = 5.5 : by sorry

end length_of_room_length_of_room_l728_728404


namespace two_coins_heads_probability_l728_728074

theorem two_coins_heads_probability :
  let outcomes := [{fst := true, snd := true}, {fst := true, snd := false}, {fst := false, snd := true}, {fst := false, snd := false}],
      favorable := {fst := true, snd := true}
  in (favorable ∈ outcomes) → (favorable :: (List.erase outcomes favorable).length) / outcomes.length = 1 / 4 :=
by
  sorry

end two_coins_heads_probability_l728_728074


namespace frac_diff_eq_neg_one_l728_728622

theorem frac_diff_eq_neg_one (x y : ℝ) (h1 : 75^x = 10^(-2)) (h2 : 0.75^y = 10^(-2)) : 
  (1/x) - (1/y) = -1 :=
by sorry

end frac_diff_eq_neg_one_l728_728622


namespace find_abc_l728_728839

theorem find_abc (a b c : ℚ) 
  (h1 : a + b + c = 24)
  (h2 : a + 2 * b = 2 * c)
  (h3 : a = b / 2) : 
  a = 16 / 3 ∧ b = 32 / 3 ∧ c = 8 := 
by 
  sorry

end find_abc_l728_728839


namespace goals_scored_by_each_l728_728030

theorem goals_scored_by_each (total_goals : ℕ) (percentage : ℕ) (two_players_goals : ℕ) (each_player_goals : ℕ)
  (H1 : total_goals = 300)
  (H2 : percentage = 20)
  (H3 : two_players_goals = (percentage * total_goals) / 100)
  (H4 : two_players_goals / 2 = each_player_goals) :
  each_player_goals = 30 := by
  sorry

end goals_scored_by_each_l728_728030


namespace probability_three_heads_in_eight_tosses_l728_728996

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l728_728996


namespace restore_triangle_l728_728826

variables {A B C D E F : Type*}
variables [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D] [decidable_eq E] [decidable_eq F]

-- Assume points D, E, and F are given intersection points
variables (pointD pointE pointF : Type*)

-- Definitions for altitude, median, and angle bisector intersections
def Altitude (A B C D : Type*) := sorry
def Median (A B C F : Type*) := sorry
def AngleBisector (A B C E : Type*) := sorry

-- Main theorem: Reconstructing the triangle ABC from points D, E, F
theorem restore_triangle
  (hD : Altitude B A C D)
  (hE : AngleBisector A C B E)
  (hF : Median A B C F)
  : Type* :=
sorry

end restore_triangle_l728_728826


namespace min_AB_DE_l728_728663

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_focus (k x y : ℝ) : Prop := y = k * (x - 1)

theorem min_AB_DE 
(F : (ℝ × ℝ)) 
(A B D E : ℝ × ℝ) 
(k1 k2 : ℝ) 
(hF : F = (1, 0)) 
(hk : k1^2 + k2^2 = 1) 
(hAB : ∀ x y, parabola x y → line_through_focus k1 x y → A = (x, y) ∨ B = (x, y)) 
(hDE : ∀ x y, parabola x y → line_through_focus k2 x y → D = (x, y) ∨ E = (x, y)) 
: |(A.1 - B.1)| + |(D.1 - E.1)| ≥ 24 := 
sorry

end min_AB_DE_l728_728663


namespace spherical_to_rectangular_coords_l728_728576

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), ρ = 15 → θ = 5 * Real.pi / 4 → φ = Real.pi / 4 →
  let x := ρ * Real.sin φ * Real.cos θ in
  let y := ρ * Real.sin φ * Real.sin θ in
  let z := ρ * Real.cos φ in
  (x, y, z) = (-15 / 2, -15 / 2, 15 * Real.sqrt 2 / 2) :=
begin
  intros ρ θ φ hρ hθ hφ,
  let x := ρ * Real.sin φ * Real.cos θ,
  let y := ρ * Real.sin φ * Real.sin θ,
  let z := ρ * Real.cos φ,
  rw [hρ, hθ, hφ],
  simp,
  split,
  { rw [Real.sin_pi_div_four, Real.cos_five_pi_div_four],
    simp },
  split,
  { rw [Real.sin_pi_div_four, Real.sin_five_pi_div_four],
    simp },
  { rw [Real.cos_pi_div_four],
    simp }
end

end spherical_to_rectangular_coords_l728_728576


namespace simplify_expression_l728_728820

theorem simplify_expression : 
  (1 / ((1 / (1 / 3)^1) + (1 / (1 / 3)^2) + (1 / (1 / 3)^3))) = 1 / 39 :=
by
  sorry

end simplify_expression_l728_728820


namespace sqrt_720_simplified_l728_728009

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  have h1 : 720 = 2^4 * 3^2 * 5 := by norm_num
  -- Here we use another proven fact or logic per original conditions and definition
  sorry

end sqrt_720_simplified_l728_728009


namespace sum_coefficients_l728_728184

theorem sum_coefficients (a : ℕ → ℕ) (n : ℕ) (h : ∀ x, ∑ k in finset.range (2 * n + 1), a k * x^k = (∑ k in finset.range (n + 1), k * x^k)^2) : 
  (∑ k in finset.range (2 * n + 1), if k > n then a k else 0) = (1 / 24) * n * (n + 1) * (5 * n^2 + 5 * n + 2) := 
sorry

end sum_coefficients_l728_728184


namespace diameter_of_circle_l728_728468

theorem diameter_of_circle (AX XB CX XD : ℝ) 
  (h1 : AX = 3) (h2 : XB = 4) (h3 : CX = 6) (h4 : XD = 2)
  (h5 : AX * XB = CX * XD) : 
  ∃ (D : ℝ), D = 4 * sqrt 14 :=
by 
  sorry

end diameter_of_circle_l728_728468


namespace ten_factorial_minus_nine_factorial_l728_728564

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem ten_factorial_minus_nine_factorial :
  factorial 10 - factorial 9 = 3265920 := 
by 
  sorry

end ten_factorial_minus_nine_factorial_l728_728564


namespace coeff_x_9_is_zero_l728_728209

-- Define the expression
def expr := (x^3 / 3 - 3 / x^2)^10

-- State the theorem
theorem coeff_x_9_is_zero : 
  (expr.coeff 9) = 0 := 
by
  sorry

end coeff_x_9_is_zero_l728_728209


namespace union_eq_inter_eq_compl_inter_eq_l728_728365

open Set

variable {R : Type*} [LinearOrder R]

def A : Set R := {x | 3 ≤ x ∧ x < 7}
def B : Set R := {x | 4 < x ∧ x < 10}

theorem union_eq : A ∪ B = {x | 3 ≤ x ∧ x < 10} := sorry

theorem inter_eq : A ∩ B = {x | 4 < x ∧ x < 7} := sorry

theorem compl_inter_eq {x : Set R} : 
  compl A ∩ B = {x | (4 < x ∧ x < 7) ∨ (7 ≤ x ∧ x < 10)} := sorry

end union_eq_inter_eq_compl_inter_eq_l728_728365


namespace factorial_difference_l728_728530

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728530


namespace trajectory_is_ellipse_l728_728261

-- Definition of the circle
def circle (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x + 3)^2 + y^2 = 100

-- Point B on the plane
def B : ℝ × ℝ := (3, 0)

-- Point P is on the circle
def on_circle (P : ℝ × ℝ) : Prop := circle P

-- Definition of the point M, which is on the perpendicular bisector of BP and CP
def on_perpendicular_bisector (P M: ℝ × ℝ) : Prop :=
  let (x1, y1) := B
  let (x2, y2) := P
  let (xM, yM) := M
  (x1 - xM)^2 + y1^2 = (x2 - xM)^2 + y2^2

-- Condition MC + MB = 10
def distance_sum_condition (M : ℝ × ℝ) (C B: ℝ × ℝ) : Prop :=
  let (xM, yM) := M
  let (xC, yC) := C
  let (xB, yB) := B
  Real.sqrt ((xM - xC)^2 + (yM - yC)^2) 
  + Real.sqrt ((xM - xB)^2 + (yM - yB)^2) = 10

-- Point C, the center of the circle
def C : ℝ × ℝ := (-3, 0)

-- Definition of the ellipse 
def ellipse (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (x^2) / 25 + (y^2) / 16 = 1

-- Final theorem definition in Lean 4
theorem trajectory_is_ellipse (M : ℝ × ℝ) (P : ℝ × ℝ) :
  on_circle P → 
  on_perpendicular_bisector P M → 
  distance_sum_condition M C B → 
  ellipse M := 
by
  -- Proof goes here
  sorry

end trajectory_is_ellipse_l728_728261


namespace sum_9_more_likely_than_sum_10_l728_728448

def dice_sums : List (ℕ × ℕ) := [
  (1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
  (2,1), (2,2), (2,3), (2,4), (2,5), (2,6),
  (3,1), (3,2), (3,3), (3,4), (3,5), (3,6),
  (4,1), (4,2), (4,3), (4,4), (4,5), (4,6),
  (5,1), (5,2), (5,3), (5,4), (5,5), (5,6),
  (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)
]

def count_sum (n : ℕ) : ℕ :=
  dice_sums.countp (λ (pair : ℕ × ℕ), pair.1 + pair.2 = n)

theorem sum_9_more_likely_than_sum_10 :
  count_sum 9 > count_sum 10 :=
by
  sorry -- The proof is omitted as per instructions.

end sum_9_more_likely_than_sum_10_l728_728448


namespace find_min_value_l728_728653

noncomputable def f (x a : ℝ) := -x^3 + 3 * x^2 + 9 * x + a

theorem find_min_value :
  ∀ a : ℝ, (∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x a ≤ 20) →
  (∃ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x a = -7) :=
by
  sorry

end find_min_value_l728_728653


namespace probability_of_three_heads_in_eight_tosses_l728_728978

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l728_728978


namespace probability_three_heads_in_eight_tosses_l728_728933

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l728_728933


namespace factorial_difference_l728_728536

-- Define factorial function for natural numbers
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- The theorem we want to prove
theorem factorial_difference : factorial 10 - factorial 9 = 3265920 :=
by
  rw [factorial, factorial]
  sorry

end factorial_difference_l728_728536


namespace find_n_l728_728403

theorem find_n :
  ∃ (n : ℕ),
    10000 ≤ n ∧ n ≤ 99999 ∧
    let a := n % 2 in let b := n % 3 in let c := n % 4 in let d := n % 5 in let e := n % 6 in
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    (a = 1) ∧
    (b ∈ {0, 1, 2}) ∧
    (c ∈ {1, 3}) ∧
    (d ∈ {0, 1, 3}) ∧
    (e ∈ {1, 3, 5}) ∧
    (b = 0 ∧ e = 3 ∨ b = 1 ∧ e = 1 ∨ b = 2 ∧ e = 5) ∧
    n = 11311 :=
begin
  sorry
end

end find_n_l728_728403


namespace sum_binomial_alternating_l728_728876

theorem sum_binomial_alternating :
  (finset.range 50).sum (λ k, (-1 : ℤ)^k * nat.choose 99 (2 * k)) = -2^49 :=
sorry

end sum_binomial_alternating_l728_728876


namespace magic_carpet_transformation_l728_728748

theorem magic_carpet_transformation:
  ∃ (a b : ℕ) (H : a * b = 100) (H1 : 1 * 8 = 8) (H2 : 9 * 12 - 8 = 100), 
  (a = 10 ∧ b = 10) :=
by
  existsi 10, 10
  have H: 10 * 10 = 100 := rfl
  have H1: 1 * 8 = 8 := rfl
  have H2: 9 * 12 - 8 = 100 := by norm_num
  exact ⟨H, H1, H2⟩

end magic_carpet_transformation_l728_728748


namespace binom_7_4_eq_35_l728_728164

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728164


namespace equation_of_circle_l728_728230

def center : ℝ × ℝ := (3, -2)
def radius : ℝ := 5

theorem equation_of_circle (x y : ℝ) :
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔
  (x - 3)^2 + (y + 2)^2 = 25 :=
by
  simp [center, radius]
  sorry

end equation_of_circle_l728_728230


namespace number_of_goose_eggs_l728_728802

variable (E : ℝ) -- E is the total number of goose eggs laid

-- The conditions translated to Lean 4 definitions
def hatched (E : ℝ) := (1/4) * E
def survived_first_month (E : ℝ) := (4/5) * hatched E
def survived_first_year (E : ℝ) := (2/5) * survived_first_month E

-- Given that 120 geese survived the first year
axiom survived_120 : survived_first_year E = 120

-- Proving that the number of goose eggs laid is 1500
theorem number_of_goose_eggs : E = 1500 :=
by
  -- The proof will go here
  sorry

end number_of_goose_eggs_l728_728802


namespace monomial_same_type_l728_728505

-- Define a structure for monomials
structure Monomial where
  coeff : ℕ
  vars : List String

-- Monomials definitions based on the given conditions
def m1 := Monomial.mk 3 ["a"]
def m2 := Monomial.mk 2 ["b"]
def m3 := Monomial.mk 1 ["a", "b"]
def m4 := Monomial.mk 3 ["a", "c"]
def target := Monomial.mk 2 ["a", "b"]

-- Define a predicate to check if two monomials are of the same type
def sameType (m n : Monomial) : Prop :=
  m.vars = n.vars

theorem monomial_same_type :
  sameType m3 target := sorry

end monomial_same_type_l728_728505


namespace binom_7_4_eq_35_l728_728181

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728181


namespace average_percentage_reduction_per_round_l728_728087

-- Define the necessary conditions
variable (x : ℝ) -- percentage of price reduction per round (in decimal form)
variable (original_price : ℝ) (final_price : ℝ)
variable (same_reduction : Prop)
variable (successive_reductions : Prop)

-- State the conditions
def conditions := original_price = 200 ∧ final_price = 98 ∧ same_reduction ∧ successive_reductions

-- Assert the theorem to be proven
theorem average_percentage_reduction_per_round (h : conditions): x = 0.3 :=
by sorry

end average_percentage_reduction_per_round_l728_728087


namespace angle_QRP_eq_50_degrees_l728_728524

theorem angle_QRP_eq_50_degrees (Ω : Type*) [circle Ω]
  (A B C P Q R : Ω)
  (triangle_ABC : triangle A B C)
  (triangle_PQR : triangle P Q R)
  (circumcircle_ABC : circumcircle Ω triangle_ABC)
  (incircle_PQR : incircle Ω triangle_PQR)
  (P_on_BC : point_on_line P B C)
  (Q_on_AB : point_on_line Q A B)
  (R_on_AC : point_on_line R A C)
  (angle_A_eq_50 : angle A = 50)
  (angle_B_eq_70 : angle B = 70)
  (angle_C_eq_60 : angle C = 60) :
  angle Q R P = 50 :=
sorry

end angle_QRP_eq_50_degrees_l728_728524


namespace budget_transportation_percent_l728_728908

theorem budget_transportation_percent :
  let
    salary_degrees := 216
    total_degrees := 360
    research_percent := 9
    utilities_percent := 5
    equipment_percent := 4
    supplies_percent := 2
    salary_percent := (salary_degrees / total_degrees : ℝ) * 100
    total_budget := 100
    known_percentages := research_percent + utilities_percent + equipment_percent + supplies_percent + salary_percent
  in total_budget - known_percentages = 20 := 
by
  sorry

end budget_transportation_percent_l728_728908


namespace canonical_equation_line_l728_728886

theorem canonical_equation_line 
  {x y z : ℝ} 
  (h1: 3 * x + y - z - 6 = 0)
  (h2: 3 * x - y + 2 * z = 0) : 
  ∃ m n p (x0 y0 z0 : ℝ), 
    (m, n, p) = (1, -9, -6) ∧ 
    (x0, y0, z0) = (1, 3, 0) ∧ 
    (∀ x y z, 
      ((x - x0) / m = (y - y0) / n) ∧ 
      ((x - x0) / m = (z - z0) / p)) := 
sorry

end canonical_equation_line_l728_728886


namespace percent_of_y_l728_728304

theorem percent_of_y (y : ℝ) (h : y > 0) : ((1 * y) / 20 + (3 * y) / 10) = (35/100) * y :=
by
  sorry

end percent_of_y_l728_728304


namespace calculate_speed_of_boat_in_still_water_l728_728906

noncomputable def speed_of_boat_in_still_water (V : ℝ) : Prop :=
    let downstream_speed := 16
    let upstream_speed := 9
    let first_half_current := 3 
    let second_half_current := 5
    let wind_speed := 2
    let effective_current_1 := first_half_current - wind_speed
    let effective_current_2 := second_half_current - wind_speed
    let V1 := downstream_speed - effective_current_1
    let V2 := upstream_speed + effective_current_2
    V = (V1 + V2) / 2

theorem calculate_speed_of_boat_in_still_water : 
    ∃ V : ℝ, speed_of_boat_in_still_water V ∧ V = 13.5 := 
sorry

end calculate_speed_of_boat_in_still_water_l728_728906


namespace number_increase_l728_728381

/-- Set S contains exactly 10 numbers with an average of 6.2. After increasing one of these numbers by a certain amount, the new average becomes 6.6. Prove that the number was increased by 4. -/
theorem number_increase (S : Finset ℝ) (a b : ℝ) (hS_card : S.card = 10) (hS_avg : S.sum = 62) (ha : a ∈ S) (hb : b = a + 4) (hS'_sum : S.sum - a + b = 66) : b - a = 4 :=
by 
  sorry

end number_increase_l728_728381


namespace speed_is_correct_l728_728452

-- Define the initial conditions
def distance_meters := 600
def time_minutes := 5

-- Conversion factors and calculations
def distance_kilometers := distance_meters / 1000
def time_hours := time_minutes / 60

-- Define the expected speed in km/hour
def expected_speed := 7.2

-- The proof statement
theorem speed_is_correct : (distance_kilometers / time_hours) = expected_speed := by
  sorry

end speed_is_correct_l728_728452


namespace sqrt_720_simplified_l728_728007

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  have h1 : 720 = 2^4 * 3^2 * 5 := by norm_num
  -- Here we use another proven fact or logic per original conditions and definition
  sorry

end sqrt_720_simplified_l728_728007


namespace factorial_difference_l728_728547

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem factorial_difference : fact 10 - fact 9 = 3265920 := by
  have h1 : fact 9 = 362880 := sorry
  have h2 : fact 10 = 10 * fact 9 := by rw [fact]
  calc
    fact 10 - fact 9
        = 10 * fact 9 - fact 9 := by rw [h2]
    ... = 9 * fact 9 := by ring
    ... = 9 * 362880 := by rw [h1]
    ... = 3265920 := by norm_num

end factorial_difference_l728_728547


namespace return_trip_time_l728_728122

theorem return_trip_time {d p w : ℝ} (h1 : d = 120 * (p-w)) (h2 : ∀ t, t = d / p → d / (p+w) = t - 20) : 
  (d / (p + w) = 80) ∨ (d / (p + w) ≈ 11) :=
by
  sorry

end return_trip_time_l728_728122


namespace range_of_a_l728_728267

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + (1 / 2) * a

theorem range_of_a (a : ℝ) : (∀ x > 0, f a x ≤ 0) → (0 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l728_728267


namespace find_alpha_l728_728291

open Real

theorem find_alpha (α : ℝ) (h1 : sin α = 1/5) (h2 : α ∈ Ioc (π / 2) π) :
  α = π - arcsin (1 / 5) :=
sorry

end find_alpha_l728_728291


namespace four_by_four_increasing_matrices_l728_728570

noncomputable def count_increasing_matrices (n : ℕ) : ℕ := sorry

theorem four_by_four_increasing_matrices :
  count_increasing_matrices 4 = 320 :=
sorry

end four_by_four_increasing_matrices_l728_728570


namespace sum_of_abcd_l728_728783

theorem sum_of_abcd (a b c d: ℝ) (h₁: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h₂: c + d = 10 * a) (h₃: c * d = -11 * b) (h₄: a + b = 10 * c) (h₅: a * b = -11 * d)
  : a + b + c + d = 1210 := by
  sorry

end sum_of_abcd_l728_728783


namespace factorial_difference_l728_728542

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem factorial_difference : fact 10 - fact 9 = 3265920 := by
  have h1 : fact 9 = 362880 := sorry
  have h2 : fact 10 = 10 * fact 9 := by rw [fact]
  calc
    fact 10 - fact 9
        = 10 * fact 9 - fact 9 := by rw [h2]
    ... = 9 * fact 9 := by ring
    ... = 9 * 362880 := by rw [h1]
    ... = 3265920 := by norm_num

end factorial_difference_l728_728542


namespace incorrect_statement_B_l728_728447

/- Definitions based on the conditions -/
def algorithm_finite : Prop := ∀ (A : Type), ∃ n : ℕ, steps n A
def algorithm_definite : Prop := ∀ (A : Type), ∀ step : A → A, clear (step A)
def algorithm_output : Prop := ∀ (A : Type), ∃ r : A, produces_output r
def multiple_algorithms : Prop := ∃ (T : Type), ∃ alg1 alg2 : T → T, alg1 ≠ alg2

/- The question's correct answer as a theorem statement -/
theorem incorrect_statement_B :
  ¬ ∀ (P : Type), ∃! alg : P → P, solves_problem alg :=
by
  sorry

end incorrect_statement_B_l728_728447


namespace integer_solution_l728_728617

theorem integer_solution (n : ℤ) (h : n ≥ 8) : n + 1/(n-7) ∈ ℤ → n = 8 :=
  sorry

end integer_solution_l728_728617


namespace quadratic_value_at_point_l728_728668

theorem quadratic_value_at_point :
  ∃ a b c, 
    (∃ y, y = a * 2^2 + b * 2 + c ∧ y = 7) ∧
    (∃ y, y = a * 0^2 + b * 0 + c ∧ y = -7) ∧
    (∃ y, y = a * 5^2 + b * 5 + c ∧ y = -24.5) := 
sorry

end quadratic_value_at_point_l728_728668


namespace coin_toss_probability_l728_728912

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l728_728912


namespace prob_three_heads_in_eight_tosses_l728_728956

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l728_728956


namespace sequence_sum_correct_l728_728415

-- Define the sequence a_n
def a_seq : ℕ → ℕ
| 0 := 1
| (n+1) := a_seq 0 + a_seq n + n

noncomputable def sum_reciprocals := 
  ∑ i in finset.range 2016, (1 / a_seq (i + 1) : ℚ)

theorem sequence_sum_correct : 
  sum_reciprocals = 4032 / 2017 :=
sorry

end sequence_sum_correct_l728_728415


namespace integral_ex_plus_x_eq_e_sub_half_l728_728520

open Real

noncomputable def integral_ex_plus_x := ∫ x in 0..1, exp(x) + x

theorem integral_ex_plus_x_eq_e_sub_half :
  integral_ex_plus_x = Real.exp(1) - 1/2 := 
sorry

end integral_ex_plus_x_eq_e_sub_half_l728_728520


namespace coordinates_of_S_l728_728035

variable (P Q R S : (ℝ × ℝ))
variable (hp : P = (3, -2))
variable (hq : Q = (3, 1))
variable (hr : R = (7, 1))
variable (h : Rectangle P Q R S)

def Rectangle (P Q R S : (ℝ × ℝ)) : Prop :=
  let (xP, yP) := P
  let (xQ, yQ) := Q
  let (xR, yR) := R
  let (xS, yS) := S
  (xP = xQ ∧ yR = yS) ∧ (xS = xR ∧ yP = yQ) 

theorem coordinates_of_S : S = (7, -2) := by
  sorry

end coordinates_of_S_l728_728035


namespace pages_read_in_a_year_l728_728715

theorem pages_read_in_a_year (novels_per_month : ℕ) (pages_per_novel : ℕ) (months_per_year : ℕ)
  (h1 : novels_per_month = 4) (h2 : pages_per_novel = 200) (h3 : months_per_year = 12) :
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by
  -- Using the given conditions
  rw [h1, h2, h3]
  -- Simplifying the expression
  simp
  sorry

end pages_read_in_a_year_l728_728715


namespace binomial_7_4_equals_35_l728_728172

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Problem statement: Prove that binomial(7, 4) = 35
theorem binomial_7_4_equals_35 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_equals_35_l728_728172


namespace probability_of_three_heads_in_eight_tosses_l728_728977

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l728_728977


namespace evaporated_water_mass_l728_728383

theorem evaporated_water_mass : 
  let initial_solution_weight := 8 in
  let liquid_X_percentage_initial := 0.3 in
  let liquid_X_weight_initial := initial_solution_weight * liquid_X_percentage_initial in
  let evaporated_water : ℝ := E in
  let second_addition_weight := initial_solution_weight in
  let new_solution_weight := initial_solution_weight - evaporated_water + second_addition_weight in
  let new_liquid_X_weight := liquid_X_weight_initial + liquid_X_weight_initial in
  let liquid_X_percentage_new := 0.4125 in
  new_liquid_X_weight / new_solution_weight = liquid_X_percentage_new → evaporated_water = 4.36 :=
by
  intros initial_solution_weight liquid_X_percentage_initial liquid_X_weight_initial evaporated_water second_addition_weight new_solution_weight new_liquid_X_weight liquid_X_percentage_new
  sorry

end evaporated_water_mass_l728_728383


namespace book_cost_l728_728454


theorem book_cost (C1 C2 : ℝ) (h1 : C1 + C2 = 470) (h2 : 0.85 * C1 = 1.19 * C2) : C1 ≈ 274.11 := 
by
  sorry

end book_cost_l728_728454


namespace fraction_spent_at_arcade_l728_728340

-- Definitions based on conditions
def A : ℝ := 2.81
def candy_expense : ℝ := 0.75

-- The problem is to find the fraction F such that
theorem fraction_spent_at_arcade (F : ℝ) (h1 : (2/3) * (1 - F) * A = candy_expense) : F ≈ 3/5 :=
by
  sorry

end fraction_spent_at_arcade_l728_728340


namespace probability_exactly_three_heads_l728_728971
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l728_728971


namespace zero_in_P_two_not_in_P_l728_728124

variables (P : Set Int)

-- Conditions
def condition_1 := ∃ x ∈ P, x > 0 ∧ ∃ y ∈ P, y < 0
def condition_2 := ∃ x ∈ P, x % 2 = 0 ∧ ∃ y ∈ P, y % 2 ≠ 0 
def condition_3 := 1 ∉ P
def condition_4 := ∀ x y, x ∈ P → y ∈ P → x + y ∈ P

-- Proving 0 ∈ P
theorem zero_in_P (h1 : condition_1 P) (h2 : condition_2 P) (h3 : condition_3 P) (h4 : condition_4 P) : 0 ∈ P := 
sorry

-- Proving 2 ∉ P
theorem two_not_in_P (h1 : condition_1 P) (h2 : condition_2 P) (h3 : condition_3 P) (h4 : condition_4 P) : 2 ∉ P := 
sorry

end zero_in_P_two_not_in_P_l728_728124


namespace cost_per_meter_of_fencing_l728_728123

theorem cost_per_meter_of_fencing
  (A : ℝ) (W : ℝ) (total_cost : ℝ)
  (hA : A = 1200) (hW : W = 30) (hC : total_cost = 1680) :
  let L := A / W in
  let D := real.sqrt (L^2 + W^2) in
  let total_length := L + W + D in
  total_cost / total_length = 14 :=
by
  let L := A / W
  let D := real.sqrt (L^2 + W^2)
  let total_length := L + W + D
  have hL : L = 40 := by sorry
  have hD : D = 50 := by sorry
  have htotal_length : total_length = 120 := by sorry
  have hcost : total_cost / total_length = 14 := by sorry
  exact hcost

end cost_per_meter_of_fencing_l728_728123


namespace coin_toss_probability_l728_728947

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l728_728947


namespace divisors_greater_than_8_factorial_l728_728287

theorem divisors_greater_than_8_factorial:
  let f := Nat.factorial
  let n := f 9
  let m := f 8
  (∃ (d : ℕ), d ∣ n ∧ d > m) → (count (λ d, d ∣ n ∧ d > m) (list.range (n+1)) = 8) :=
begin
  intros h,
  sorry
end

end divisors_greater_than_8_factorial_l728_728287


namespace coin_toss_probability_l728_728951

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l728_728951


namespace abc_zero_l728_728033

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = (a * b * c)^3) 
  : a * b * c = 0 := by
  sorry

end abc_zero_l728_728033


namespace right_triangle_area_l728_728835

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 5) (hc : c = 13) : 
  0.5 * a * (real.sqrt(c^2 - a^2)) = 30 := 
by
  sorry

end right_triangle_area_l728_728835


namespace probability_three_heads_in_eight_tosses_l728_728935

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l728_728935


namespace max_intersection_points_is_10_l728_728099

noncomputable def max_num_intersection_points (n : ℕ) : ℕ :=
  if h : n = 10 then 10 else sorry

theorem max_intersection_points_is_10 :
  ∀ (n : ℕ), 
    n = 10 → max_num_intersection_points n = 10 := 
by
  intro n h
  rw [h]
  simp [max_num_intersection_points]
  rfl

end max_intersection_points_is_10_l728_728099


namespace binom_7_4_eq_35_l728_728178

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728178


namespace find_a_b_and_extreme_value_logarithmic_product_inequality_l728_728657

noncomputable def f (x : ℝ) (a b : ℝ) := (a * x + b) * log x - b * x + 3

theorem find_a_b_and_extreme_value :
  let a := 0
  let b := 1
  let f := f x 0 1
  f 1 = 2 ∧ (∀ x, 0 < x → x < 1 → (f x < f 1)) ∧ (∀ x, x > 1 → (f x < f 1)) :=
by {
  sorry
}

theorem logarithmic_product_inequality (n : ℕ) (h : n ≥ 2) :
  (∏ k in Finset.range (n - 1), (log (↑k + 2) / (↑k + 2))) < (1 / ↑n) :=
by {
  sorry
}

end find_a_b_and_extreme_value_logarithmic_product_inequality_l728_728657


namespace quadratic_polynomial_conditions_l728_728218

noncomputable def q (x : ℝ) : ℝ := -4.5 * x^2 + 13.5 * x + 81

theorem quadratic_polynomial_conditions :
  q (-3) = 0 ∧ q 6 = 0 ∧ q 7 = -45 :=
by
  unfold q,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }

end quadratic_polynomial_conditions_l728_728218


namespace largest_quotient_l728_728872

theorem largest_quotient (s : set ℤ) 
  (h : s = {-30, -6, -1, 3, 5, 20}) 
  (neg_cond : ∀ a b ∈ s, a < 0 ∨ b < 0) :
  ∃ a b ∈ s, a < 0 ∨ b < 0 ∧ (a : ℚ) / b = -0.05 :=
by
  have h1 : -1 ∈ s, from by rw h; simp,
  have h2 : 20 ∈ s, from by rw h; simp,
  use [-1, 20],
  repeat {
    split,
    assumption,
  },
  simp,
  norm_num,
  exact neg_cond _ _ h1 h2

end largest_quotient_l728_728872


namespace factorial_difference_l728_728552

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728552


namespace number_2015_is_106th_lucky_number_l728_728301

def is_lucky_number (n : ℕ) : Prop :=
  (n.digits 10).sum = 8

theorem number_2015_is_106th_lucky_number :
  ∃ (k : ℕ), 2015 = (list.filter is_lucky_number (list.range (2015 + 1))).nth k ∧ k = 105 :=
sorry

end number_2015_is_106th_lucky_number_l728_728301


namespace student_B_is_wrong_l728_728836

def is_monotonically_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop := 
  ∀ x y ∈ I, x < y → f y ≤ f x

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop := 
  ∀ x y ∈ I, x < y → f x ≤ f y

def is_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

def exactly_three_are_correct (A B C D : Prop) : Prop :=
  (A ∧ B ∧ C ∧ ¬D) ∨
  (A ∧ B ∧ ¬C ∧ D) ∨
  (A ∧ ¬B ∧ C ∧ D) ∨
  (¬A ∧ B ∧ C ∧ D)

theorem student_B_is_wrong (f : ℝ → ℝ) :
  let A := is_monotonically_decreasing f {x | x ≤ 0},
      B := is_monotonically_increasing f {x | x ≥ 0},
      C := is_symmetric_about_line f 1,
      D := ¬ (∀ x, f x ≥ f 0)
  in exactly_three_are_correct A B C D → ¬ B :=
by
  sorry

end student_B_is_wrong_l728_728836


namespace num_neg_values_of_x_l728_728612

theorem num_neg_values_of_x 
  (n : ℕ) 
  (xn_pos_int : ∃ k, n = k ∧ k > 0) 
  (sqrt_x_169_pos_int : ∀ x, ∃ m, x + 169 = m^2 ∧ m > 0) :
  ∃ count, count = 12 := 
by
  sorry

end num_neg_values_of_x_l728_728612


namespace num_inverses_mod_9_l728_728682

theorem num_inverses_mod_9 : (Finset.filter (λ x, ∃ y, (x * y) % 9 = 1) (Finset.range 9)).card = 6 :=
by
  sorry

end num_inverses_mod_9_l728_728682


namespace probability_of_three_heads_in_eight_tosses_l728_728990

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l728_728990


namespace trigonometric_identity_l728_728709

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
by 
  sorry

end trigonometric_identity_l728_728709


namespace product_QED_l728_728290

def Q : Complex := 7 + 3 * Complex.i
def E : Complex := 2 * Complex.i
def D : Complex := 7 - 3 * Complex.i

theorem product_QED : Q * E * D = 116 * Complex.i := 
by sorry

end product_QED_l728_728290


namespace binomial_7_4_eq_35_l728_728156

-- Define the binomial coefficient using the binomial coefficient formula.
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem to prove.
theorem binomial_7_4_eq_35 : binomial_coefficient 7 4 = 35 :=
sorry

end binomial_7_4_eq_35_l728_728156


namespace unit_cubes_center_property_l728_728758

theorem unit_cubes_center_property :
    ∀ (larger_cube : Set ℝ),
      (∃ (unit_cubes : Set (Set ℝ)), unit_cubes ⊆ larger_cube ∧ unit_cubes.finite ∧ unit_cubes.card = 1001 ∧ 
      (∀ cube ∈ unit_cubes, ∃ center : ℝ × ℝ × ℝ, center ∈ cube ∧ ∀ face, face ⊆ larger_cube ∧ cube ⊥ face)) → 
    ∃ (c1 c2 : Set ℝ), c1 ∈ unit_cubes ∧ c2 ∈ unit_cubes ∧ c1 ≠ c2 ∧
    (∃ center1 center2 : ℝ × ℝ × ℝ, center1 ∈ c1 ∧ center2 ∈ c2 ∧ 
    (center1 ∈ c2 ∨ center2 ∈ c1 ∨ 
    ( dist center1 center2 < 1 ∧ 
      (center1 ∈ CubeInterior (face c1) ∨ center2 ∈ CubeInterior (face c2))))) :=
sorry

end unit_cubes_center_property_l728_728758


namespace A_wins_if_perfect_square_or_prime_l728_728246

theorem A_wins_if_perfect_square_or_prime (n : ℕ) (h_pos : 0 < n) : 
  (∃ A_wins : Bool, A_wins = true ↔ (∃ k : ℕ, n = k^2) ∨ (∃ p : ℕ, Nat.Prime p ∧ n = p)) :=
by
  sorry

end A_wins_if_perfect_square_or_prime_l728_728246


namespace exists_sum_of_eight_digits_equals_36_l728_728322

/-- Prove that there exists a list of 8 different digits such that their sum equals 36. --/
theorem exists_sum_of_eight_digits_equals_36 :
  ∃ (digits : List ℕ), 
    (∀ d ∈ digits, d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧ -- All digits must be in the range 0 to 9
    digits.nodup ∧ -- All digits must be different
    digits.length = 8 ∧ -- There must be exactly 8 digits
    digits.sum = 36 := -- Their sum must be 36
sorry

end exists_sum_of_eight_digits_equals_36_l728_728322


namespace prob_three_heads_in_eight_tosses_l728_728961

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l728_728961


namespace problem_statement_l728_728253

variable (a : ℝ)
def prop_p := ∀ x : ℝ, 2 ≤ x → x ≤ 4 → x^2 - a * x - 8 > 0
def prop_q := ∃ θ : ℝ, a - 1 ≤ sin θ - 2

theorem problem_statement (hp : prop_p a) (hq : prop_q a) : a < -2 :=
sorry

end problem_statement_l728_728253


namespace tangent_line_equation_l728_728665

-- Define the function y = sqrt(x)
def curve (x : ℝ) : ℝ := Real.sqrt x

-- Define the point (1,1)
def point : ℝ × ℝ := (1, 1)

-- Prove that the equation of the tangent line to the curve at the point (1, 1) is x - 2y + 1 = 0
theorem tangent_line_equation : 
  tangent_line curve point = (λ x y, x - 2 * y + 1 = 0) :=
sorry

end tangent_line_equation_l728_728665


namespace certain_number_l728_728108

theorem certain_number (x : ℝ) (h : 0.65 * 40 = (4/5) * x + 6) : x = 25 :=
sorry

end certain_number_l728_728108


namespace probability_exactly_three_heads_l728_728968
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l728_728968


namespace prism_inscribed_in_sphere_volume_l728_728498

noncomputable def prism_volume :=
  let R := 3
  let AD := 2 * Real.sqrt 6
  let h := 2 * Real.sqrt 5
  let r := 2
  let base_area := (Real.sqrt 3 / 4) * (2 * Real.sqrt 3) ^ 2
  base_area * h

theorem prism_inscribed_in_sphere_volume :
  let R := 3 in
  let AD := 2 * Real.sqrt 6 in
  let h := 2 * Real.sqrt 5 in
  let r := 2 in
  let base_area := (Real.sqrt 3 / 4) * (2 * Real.sqrt 3) ^ 2 in
  base_area * h = 6 * Real.sqrt 15 := by
  sorry

end prism_inscribed_in_sphere_volume_l728_728498


namespace find_z2_l728_728652

-- Given conditions
def z1 := 1 - complex.i
def c1 := 1 + complex.i

-- Given problem to prove
theorem find_z2 (z2 : ℂ) : z1 * z2 = c1 → z2 = complex.i :=
by sorry

end find_z2_l728_728652


namespace solve_for_x_l728_728693

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 15 / (x / 3)) : x = 15 * Real.sqrt 3 ∨ x = -15 * Real.sqrt 3 :=
by
  sorry

end solve_for_x_l728_728693


namespace addition_order_correct_l728_728443

noncomputable def correctOrderAddition (a b : ℚ) : Prop :=
  let abs_a := abs a
  let abs_b := abs b
  let larger := max abs_a abs_b
  let smaller := min abs_a abs_b
  let sign := if abs_a > abs_b then (if a > 0 then 1 else -1) else (if b > 0 then 1 else -1)
  let result := (larger - smaller) * sign
  result = a + b -- Statement of the correctness based on provided sequence.

theorem addition_order_correct (a b : ℚ) (h_diff_signs : (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) : correctOrderAddition a b :=
begin
  sorry
end

end addition_order_correct_l728_728443


namespace solve_for_x_l728_728272

-- Define the conditions as mathematical statements in Lean
def conditions (x y : ℝ) : Prop :=
  (2 * x - 3 * y = 10) ∧ (y = -x)

-- State the theorem that needs to be proven
theorem solve_for_x : ∃ x : ℝ, ∃ y : ℝ, conditions x y ∧ x = 2 :=
by 
  -- Provide a sketch of the proof to show that the statement is well-formed
  sorry

end solve_for_x_l728_728272


namespace binom_7_4_l728_728160

theorem binom_7_4 : Nat.choose 7 4 = 35 := 
by
  sorry

end binom_7_4_l728_728160


namespace calculate_result_sin_15_l728_728523

theorem calculate_result_sin_15 :
  (1 - 2 * (Real.sin (15 : ℝ * Real.pi / 180))^2 = Real.cos (30 : ℝ * Real.pi / 180)) :=
by
  sorry

end calculate_result_sin_15_l728_728523


namespace Elberta_has_21_dollars_l728_728283

theorem Elberta_has_21_dollars
  (Granny_Smith : ℕ)
  (Anjou : ℕ)
  (Elberta : ℕ)
  (h1 : Granny_Smith = 72)
  (h2 : Anjou = Granny_Smith / 4)
  (h3 : Elberta = Anjou + 3) :
  Elberta = 21 := 
  by {
    sorry
  }

end Elberta_has_21_dollars_l728_728283


namespace probability_of_exactly_three_heads_l728_728928

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l728_728928


namespace additional_divisor_of_44404_l728_728846

theorem additional_divisor_of_44404 :
  let n := 44402 in
  let m := n + 2 in
  m % 12 = 0 ∧ m % 30 = 0 ∧ m % 74 = 0 ∧ m % 100 = 0 →
  ∃ d, d = 37003 ∧ m % d = 0 :=
by
  intros n m h
  let m := 44404
  sorry

end additional_divisor_of_44404_l728_728846


namespace f_monotonic_increasing_on_neg_infty_0_f_min_value_on_minus3_minus1_f_max_value_on_minus3_minus1_l728_728661

-- Define the function f(x) = 1 / (1 + x^2)
def f (x : ℝ) : ℝ := 1 / (1 + x^2)

-- Prove that f(x) is monotonically increasing on (-∞, 0)
theorem f_monotonic_increasing_on_neg_infty_0 : 
  ∀ x y, x < y → x < 0 → y < 0 → f x ≤ f y :=
by sorry

-- Define the endpoints of the interval [-3, -1]
def a := -3
def b := -1

-- Prove that the minimum value of f(x) on [-3, -1] is 1/10
theorem f_min_value_on_minus3_minus1 : 
  is_min_on f (function.restrict f (set.interval [-3, -1]) ) (1/10) :=
by sorry

-- Prove that the maximum value of f(x) on [-3, -1] is 1/2
theorem f_max_value_on_minus3_minus1 : 
  is_max_on f (function.restrict f (set.interval [-3, -1]) ) (1/2) :=
by sorry

end f_monotonic_increasing_on_neg_infty_0_f_min_value_on_minus3_minus1_f_max_value_on_minus3_minus1_l728_728661


namespace repaired_shoes_last_correct_l728_728477

noncomputable def repaired_shoes_last := 
  let repair_cost: ℝ := 10.50
  let new_shoes_cost: ℝ := 30.00
  let new_shoes_years: ℝ := 2.0
  let percentage_increase: ℝ := 42.857142857142854 / 100
  (T : ℝ) -> 15.00 = (repair_cost / T) * (1 + percentage_increase) → T = 1

theorem repaired_shoes_last_correct : repaired_shoes_last :=
by
  sorry

end repaired_shoes_last_correct_l728_728477


namespace larger_side_has_shorter_median_l728_728377

variable {α : Type*} [EuclideanGeometry α]

/-- In a triangle, the larger side corresponds to the smaller median and vice versa -/
theorem larger_side_has_shorter_median 
  {A B C M1 M2 G : α}
  (h△ : triangle A B C)
  (h_midBC : midpoint B C M1)
  (h_midAC : midpoint A C M2)
  (h_centroid : centroid_of_triangle A B C G)
  (h_side_compare : dist B C > dist A C) :
  dist (centroid_of_triangle B C M1) B > dist (centroid_of_triangle A C M2) A := 
sorry

end larger_side_has_shorter_median_l728_728377


namespace correct_order_of_operations_for_adding_rationals_with_different_signs_l728_728442

theorem correct_order_of_operations_for_adding_rationals_with_different_signs :
  ∀ (a b : ℚ), 
    (let abs_a := |a|,
         abs_b := |b|,
         larger := max abs_a abs_b,
         smaller := min abs_a abs_b,
         sign := if abs_a > abs_b then sign a else sign b,
         magnitude := larger - smaller
     in sign * magnitude = a + b) :=
by
  intros a b
  let abs_a := |a|
  let abs_b := |b|
  let larger := max abs_a abs_b
  let smaller := min abs_a abs_b
  let sign := if abs_a > abs_b then sign a else sign b
  let magnitude := larger - smaller
  show sign * magnitude = a + b
  sorry

end correct_order_of_operations_for_adding_rationals_with_different_signs_l728_728442


namespace calculate_value_l728_728144

theorem calculate_value :
  12 * ( (1 / 3 : ℝ) + (1 / 4) + (1 / 6) )⁻¹ = 16 :=
sorry

end calculate_value_l728_728144


namespace total_points_correct_l728_728374

structure PaperRecycling where
  white_paper_points : ℚ
  colored_paper_points : ℚ

def paige_paper : PaperRecycling := {
  white_paper_points := (12 / 6) * 2,
  colored_paper_points := (18 / 8) * 3
}

def alex_paper : PaperRecycling := {
  white_paper_points := (⟨26, 6, by norm_num⟩.num / ⟨26, 6, by norm_num⟩.den) * 2,
  colored_paper_points := (⟨10, 8, by norm_num⟩.num / ⟨10, 8, by norm_num⟩.den) * 3
}

def jordan_paper : PaperRecycling := {
  white_paper_points := (30 / 6) * 2,
  colored_paper_points := 0
}

def total_points : ℚ :=
  paige_paper.white_paper_points + paige_paper.colored_paper_points +
  alex_paper.white_paper_points + alex_paper.colored_paper_points +
  jordan_paper.white_paper_points + jordan_paper.colored_paper_points

theorem total_points_correct : total_points = 31 := by
  sorry

end total_points_correct_l728_728374


namespace not_decreasing_on_interval_l728_728263

def f (x a : ℝ) : ℝ :=
  (x^2 - a * x) * Real.exp x

theorem not_decreasing_on_interval (a : ℝ) :
  ¬ (∀ x ∈ Icc (-1 : ℝ) 1, (x^2 + (2 - a) * x - a * x) * Real.exp x ≤ 0) ↔ a < 3 / 2 :=
sorry

end not_decreasing_on_interval_l728_728263


namespace find_w_l728_728388

-- Define the roots condition for the first polynomial
def roots_of_first_cubic_polynomial (x p q r : ℝ) : Prop :=
  (x^3 + 4 * x^2 + 5 * x - 13 = 0) ∧
  ∃ (p q r : ℝ), (∃ (h : (x - p) * (x - q) * (x - r) = x^3 + 4 * x^2 + 5 * x - 13), True) 

-- Define the roots condition for the second polynomial
def roots_of_second_cubic_polynomial (x u v w p q r : ℝ) : Prop :=
  (x^3 + u * x^2 + v * x + w = 0) ∧ 
  ∃ (p q r : ℝ), (∃ (h : (x - (p + q)) * (x - (q + r)) * (x - (r + p)) = x^3 + u * x^2 + v * x + w), True)

-- The proof statement for w given the conditions
theorem find_w
  (p q r : ℝ)
  (h1 : roots_of_first_cubic_polynomial 0 p q r)
  (h2 : p + q + r = -4)
  (h3 : roots_of_second_cubic_polynomial 0 0 0 0 p q r) :
  let w := -(p + q) * (q + r) * (r + p)
  in w = 33 :=
by
  sorry

end find_w_l728_728388


namespace ten_factorial_minus_nine_factorial_l728_728568

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem ten_factorial_minus_nine_factorial :
  factorial 10 - factorial 9 = 3265920 := 
by 
  sorry

end ten_factorial_minus_nine_factorial_l728_728568


namespace infinite_solutions_abs_eq_ax_minus_2_l728_728228

theorem infinite_solutions_abs_eq_ax_minus_2 (a : ℝ) :
  (∀ x : ℝ, |x - 2| = ax - 2) ↔ a = 1 :=
by {
  sorry
}

end infinite_solutions_abs_eq_ax_minus_2_l728_728228


namespace positive_difference_smallest_prime_factors_of_96043_l728_728875

theorem positive_difference_smallest_prime_factors_of_96043 : 
  ∃ (a b : ℕ), prime a ∧ prime b ∧ a < b ∧ (a, b).fst = 7 ∧ (a, b).snd = 11 ∧ (b - a) = 4 := by
  sorry

end positive_difference_smallest_prime_factors_of_96043_l728_728875


namespace probability_of_three_heads_in_eight_tosses_l728_728976

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l728_728976


namespace find_m_monotonically_decreasing_l728_728198

noncomputable def f (m x : ℝ) := (m^2 - m - 5) * x^(m + 1)

theorem find_m_monotonically_decreasing :
  ∃ m : ℝ, (f m x).deriv < 0 ∀ x ∈ set.Ioi 0 :=
begin
  let m := -2,
  sorry
end

end find_m_monotonically_decreasing_l728_728198


namespace math_problem_l728_728104

theorem math_problem (p x y : ℤ) (h1 : p^2 - 4 * x + y = 20) (h2 : y > 3) (u v : ℤ) (h3 : y-3 = p^u) (h4 : u ≥ 0) (h5 : y+3 = p^v) (h6 : v ≥ 0) (h7 : gcd (y-3) (y+3) = gcd (y-3, 6)) :
  p = 7 ∧ x = 1 ∧ y = 4 ∨ p = 3 ∧ x = 3 ∧ y = 6 :=
sorry

end math_problem_l728_728104


namespace complex_inequality_thm_l728_728792

noncomputable def complex_inequality (a b : ℕ → ℂ) (n : ℕ) :=
  (Re (∑ k in Finset.range (n + 1), a k * b k) ≤ 
  (1 / (3*n + 2)) * ((∑ k in Finset.range (n + 1), ∥a k∥ ^ 2) + 
  ((9*n^2 + 6*n + 2) / 2) * (∑ k in Finset.range (n + 1), ∥b k∥ ^ 2)))

theorem complex_inequality_thm (a b : ℕ → ℂ) (n : ℕ) : 
  complex_inequality a b n :=
  sorry

end complex_inequality_thm_l728_728792


namespace max_radius_right_angle_triangle_l728_728762

theorem max_radius_right_angle_triangle :
  ∀ (A B C R : ℝ) (h : ℝ), 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ -- Ensuring distinct points
  (∀ x, 0 ≤ x) ∧           -- Radius is non-negative
  B ≠ R ∧ C ≠ R ∧ 
  (h = 5 ∧ A = 3 ∧ B = 4) ∧ -- Using Pythagorean triplet (3, 4, 5)
  (h = sqrt (5^2 - (4 / 2)^2)) 
  → h = sqrt(21) := 
begin
  -- Proof goes here
  sorry
end

end max_radius_right_angle_triangle_l728_728762


namespace integer_solutions_count_l728_728772

noncomputable def f (x b : ℝ) : ℝ := x^2 + b*x + 1

theorem integer_solutions_count (b : ℝ) (h_condition : |b| > 2 ∧ ∀ a : ℤ, a ≠ 0 ∧ a ≠ 1 ∧ a ≠ -1 → b ≠ a + 1/a) :
  ∃ k : ℤ, k = 2 ∧ (∀ x : ℝ, f(f(x, b) + x, b) < 0 → ∃! n : ℤ, x = n) :=
begin
  sorry
end

end integer_solutions_count_l728_728772


namespace probability_of_exactly_three_heads_l728_728931

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l728_728931


namespace polar_coordinates_of_points_A_and_B_max_value_PA_plus_PB_squared_l728_728664

namespace Proof

def coordinates_of_point_A (t: ℝ) : ℝ × ℝ :=
  (2 - t, sqrt 3 * t)

def coordinates_of_point_B : ℝ × ℝ :=
  let (x_A, y_A) := coordinates_of_point_A (-1)
  in (-x_A, -y_A)

def polar_coordinates (x y: ℝ) : ℝ × ℝ :=
  (sqrt (x ^ 2 + y ^ 2), atan2 y x)

def curve_C2_polar (θ: ℝ) : ℝ :=
  6 / sqrt (9 - 3 * sin θ ^ 2)

theorem polar_coordinates_of_points_A_and_B : 
  polar_coordinates (2 - (-1)) (sqrt 3 * (-1)) = (2 * sqrt 3, -π / 6) ∧
  polar_coordinates (- (2 - (-1))) (- (sqrt 3 * (-1))) = (2 * sqrt 3, 5 * π / 6) :=
by {
  sorry
}

theorem max_value_PA_plus_PB_squared :
  ∀ P: ℝ × ℝ, 
    let θ := atan2 P.2 P.1
    in P = (2 * cos θ, sqrt 6 * sin θ) → 
      (2 * cos θ - 3) ^ 2 + (sqrt 6 * sin θ + sqrt 3) ^ 2 
      + (2 * cos θ + 3) ^ 2 + (sqrt 6 * sin θ - sqrt 3) ^ 2 ≤ 36 :=
by {
  sorry
}

end Proof

end polar_coordinates_of_points_A_and_B_max_value_PA_plus_PB_squared_l728_728664


namespace range_of_b_l728_728642

theorem range_of_b (b : ℝ) :
  (∀ x y : ℝ, (x ≠ y) → (y = 1/3 * x^3 + b * x^2 + (b + 2) * x + 3) → (y ≥ 1/3 * x^3 + b * x^2 + (b + 2) * x + 3))
  ↔ (-1 ≤ b ∧ b ≤ 2) :=
sorry

end range_of_b_l728_728642


namespace smallest_n_with_pythagorean_triplet_partition_l728_728353

theorem smallest_n_with_pythagorean_triplet_partition (n : ℕ) (h : n ≥ 5) : 
  (∀ (A B : set ℕ), A ∪ B = {x : ℕ | 2 ≤ x ∧ x ≤ n} → A ∩ B = ∅ → 
    ∃ (x y z ∈ A) (x y z : ℕ), x^2 + y^2 = z^2 ∨ 
    ∃ (x y z ∈ B) (x y z : ℕ), x^2 + y^2 = z^2) :=
begin
  sorry
end

example : ∃ n, n = 5 ∧ 
  (∀ (A B : set ℕ), A ∪ B = {x : ℕ | 2 ≤ x ∧ x ≤ n} → A ∩ B = ∅ → 
    ∃ (x y z ∈ A) (x y z : ℕ), x^2 + y^2 = z^2 ∨ 
    ∃ (x y z ∈ B) (x y z : ℕ), x^2 + y^2 = z^2) :=
begin
  use 5,
  split,
  { refl },
  { intros A B h_union h_disjoint,
    sorry },
end

end smallest_n_with_pythagorean_triplet_partition_l728_728353


namespace triangle_construction_l728_728186

-- Define the given conditions
variables (a : ℝ) (B C A : euclidean_geometry.point) (angle_C : euclidean_geometry.angle) 

-- The main theorem we want to prove
theorem triangle_construction (hBC : euclidean_geometry.dist B C = a)
    (hAngle : euclidean_geometry.angle_at_point C B A = angle_C)
    (hAC_2AB : euclidean_geometry.dist A C = 2 * euclidean_geometry.dist A B) :
    ∃ (A1 B1 C1 : euclidean_geometry.point), 
        (euclidean_geometry.dist B1 C1 = a) ∧ 
        (euclidean_geometry.angle_at_point C1 B1 A1 = angle_C) ∧ 
        (euclidean_geometry.dist A1 C1 = 2 * euclidean_geometry.dist A1 B1) ∧ 
        (triangle ABC) ∧ 
        (triangle AB1C1) := 
sorry

end triangle_construction_l728_728186


namespace ae_times_af_eq_100_l728_728391

-- Given definitions and conditions
variable (O : Type) -- Type representing the circle with center O
variable [MetricSpace O]

-- Definitions for points and line segments
variables (A B C D E F : O)
variable (radius : ℝ)
variable (cd_parallel_to_ab : is_parallel (line_segment C D) (line_segment A B))

-- Definitions for the circle center and radius
noncomputable def circle_center (O : Type) : ℝ := sorry

-- Definitions asserting the parallel chord and tangent line relationships
axiom tangent_at_A (A : O) (A_tangent : ∀ P : O, P ≠ A → line_tangent_at (circle_center O) P)
axiom meet_points (A B C D : O) (E : O) (F : O)

-- Given condition AB = 10
axiom ab_length : |AB| = 10

-- Proof that |AE|⋅|AF| = 100
theorem ae_times_af_eq_100 : |AE| * |AF| = 100 := sorry

end ae_times_af_eq_100_l728_728391


namespace factorial_difference_l728_728538

-- Define factorial function for natural numbers
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- The theorem we want to prove
theorem factorial_difference : factorial 10 - factorial 9 = 3265920 :=
by
  rw [factorial, factorial]
  sorry

end factorial_difference_l728_728538


namespace remaining_pie_l728_728145

theorem remaining_pie (carlos_take: ℝ) (sophia_share : ℝ) (final_remaining : ℝ) :
  carlos_take = 0.6 ∧ sophia_share = (1 - carlos_take) / 4 ∧ final_remaining = (1 - carlos_take) - sophia_share →
  final_remaining = 0.3 :=
by
  intros h
  sorry

end remaining_pie_l728_728145


namespace addition_order_correct_l728_728444

noncomputable def correctOrderAddition (a b : ℚ) : Prop :=
  let abs_a := abs a
  let abs_b := abs b
  let larger := max abs_a abs_b
  let smaller := min abs_a abs_b
  let sign := if abs_a > abs_b then (if a > 0 then 1 else -1) else (if b > 0 then 1 else -1)
  let result := (larger - smaller) * sign
  result = a + b -- Statement of the correctness based on provided sequence.

theorem addition_order_correct (a b : ℚ) (h_diff_signs : (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) : correctOrderAddition a b :=
begin
  sorry
end

end addition_order_correct_l728_728444


namespace max_abcd_is_one_l728_728407

noncomputable def max_abcd {a b c d : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_eq : (1 + a) * (1 + b) * (1 + c) * (1 + d) = 16) : ℝ :=
  if h :  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1)
    then abcd
    else sorry
  where abcd : ℝ := a * b * c * d

theorem max_abcd_is_one {a b c d : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_eq : (1 + a) * (1 + b) * (1 + c) * (1 + d) = 16) : ∃ (m : ℝ), m = max_abcd h_pos_a h_pos_b h_pos_c h_pos_d h_eq :=
begin
  use 1,
  sorry
end

end max_abcd_is_one_l728_728407


namespace second_number_value_l728_728469

theorem second_number_value (x y : ℝ) (h1 : (1/5) * x = (5/8) * y) 
                                      (h2 : x + 35 = 4 * y) : y = 40 := 
by 
  sorry

end second_number_value_l728_728469


namespace ellipse_equation_line_through_F_l728_728747

-- Definitions based on the problem conditions
def c : ℝ := real.sqrt 3
def area_triangle : ℝ := real.sqrt 3
def a : ℝ := 2
def b : ℝ := 1

-- Ellipse equation from the conditions
def ellipse_eq (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Right focus of the ellipse
def F : ℝ × ℝ := (real.sqrt 3, 0)

-- Prove the ellipse equation is as given in the solution 
theorem ellipse_equation :
  ∀ x y : ℝ, ellipse_eq x y ↔ ((x^2 / 4) + y^2 = 1) :=
by sorry

-- Line equation from the solution
def line_eqn (x y : ℝ) (m : ℝ) : Prop := x = m * y + real.sqrt 3

-- Proof for when the slope is +- 2*sqrt(2)
theorem line_through_F :
  ∃ m : ℝ, line_eqn (- b) (a) m ∧ (m = 2 * real.sqrt 2 ∨ m = -2 * real.sqrt 2) :=
by sorry

end ellipse_equation_line_through_F_l728_728747


namespace train_journey_time_l728_728021

theorem train_journey_time {X : ℝ} (h1 : 0 < X) (h2 : X < 60) (h3 : ∀ T_A M_A T_B M_B : ℝ, M_A - T_A = X ∧ M_B - T_B = X) :
    X = 360 / 7 :=
by
  sorry

end train_journey_time_l728_728021


namespace prob_three_heads_in_eight_tosses_l728_728953

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l728_728953


namespace binomial_7_4_eq_35_l728_728155

-- Define the binomial coefficient using the binomial coefficient formula.
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem to prove.
theorem binomial_7_4_eq_35 : binomial_coefficient 7 4 = 35 :=
sorry

end binomial_7_4_eq_35_l728_728155


namespace bailey_dog_treats_l728_728139

-- Definitions based on conditions
def total_charges_per_card : Nat := 5
def number_of_cards : Nat := 4
def chew_toys : Nat := 2
def rawhide_bones : Nat := 10

-- Total number of items bought
def total_items : Nat := total_charges_per_card * number_of_cards

-- Definition of the number of dog treats
def dog_treats : Nat := total_items - (chew_toys + rawhide_bones)

-- Theorem to prove the number of dog treats
theorem bailey_dog_treats : dog_treats = 8 := by
  -- Proof is skipped with sorry
  sorry

end bailey_dog_treats_l728_728139


namespace max_value_g_l728_728341

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (4 - x))

theorem max_value_g : ∃ (x₁ N : ℝ), (0 ≤ x₁ ∧ x₁ ≤ 4) ∧ (N = 16) ∧ (x₁ = 2) ∧ (∀ x, 0 ≤ x ∧ x ≤ 4 → g x ≤ N) :=
by
  sorry

end max_value_g_l728_728341


namespace binomial_7_4_eq_35_l728_728154

-- Define the binomial coefficient using the binomial coefficient formula.
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem to prove.
theorem binomial_7_4_eq_35 : binomial_coefficient 7 4 = 35 :=
sorry

end binomial_7_4_eq_35_l728_728154


namespace paul_weed_eating_money_l728_728375

variable (weekly_spending total_weeks lawn_money : ℕ)

def total_money (weekly_spending total_weeks : ℕ) : ℕ :=
  weekly_spending * total_weeks

def weed_eating_money (total_money lawn_money : ℕ) : ℕ :=
  total_money - lawn_money

theorem paul_weed_eating_money
  (h1 : weekly_spending = 9)
  (h2 : total_weeks = 8)
  (h3 : lawn_money = 44) :
  weed_eating_money (total_money weekly_spending total_weeks) lawn_money = 28 :=
by
  unfold total_money weed_eating_money
  rw [h1, h2, h3]
  norm_num
  sorry

end paul_weed_eating_money_l728_728375


namespace probability_of_exactly_three_heads_l728_728924

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l728_728924


namespace probability_of_product_form_l728_728727

open set classical function

noncomputable def expressions : set (ℚ → ℚ → ℚ) := {λ (x y : ℚ), x + y, λ (x y : ℚ), x + 5*y, λ (x y : ℚ), x - y, λ (x y : ℚ), 5*x + y}

def is_form_x_squared_minus_by_squared (f g : ℚ → ℚ → ℚ) : Prop :=
∃ (b : ℚ), ∀ (x y : ℚ), f x y * g x y = x^2 - (b * y)^2

theorem probability_of_product_form :
  (finset.univ.image (λ (p : finset (ℚ → ℚ → ℚ)), p.1))
    (finset.filter (λ (p : (ℚ → ℚ → ℚ) × (ℚ → ℚ → ℚ)), is_form_x_squared_minus_by_squared p.1 p.2)
      (finset.univ.image (λ (p : finset (ℚ → ℚ → ℚ)), p.1))).card =
  (1 / 6 : ℚ) :=
sorry

end probability_of_product_form_l728_728727


namespace sun_does_not_rise_from_west_l728_728885

noncomputable def isImpossibleEvent (E : Type) : Prop :=
  ∀ (e : E), ¬e

def sunRisesFromTheWestTomorrow : Prop :=
  isImpossibleEvent (λ _ : Unit, False)

theorem sun_does_not_rise_from_west : sunRisesFromTheWestTomorrow :=
by
  -- Proof here, skipped
  sorry

end sun_does_not_rise_from_west_l728_728885


namespace circumsphere_surface_area_l728_728135

def Point := ℝ × ℝ × ℝ

-- Definitions for the points A, B, C, P
def A : Point := (0, 0, 0)
def B : Point := (6, 0, 0)
def C : Point := (3, 3 * Real.sqrt 3, 0)
def P : Point := (3, Real.sqrt 3, 6 * Real.sqrt 3)

-- Definition for the distance between two points
def dist (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

-- Condition: P-AC-B has a dihedral angle of 120 degrees
def dihedral_angle_condition : Bool :=
  true  -- Placeholder, actual geometric verification needed

-- Given that P-AC-B has a dihedral angle of 120 degrees
axiom dihedral_angle_120 : dihedral_angle_condition

-- The surface area of the circumsphere
def surface_area_of_circumsphere (P A B C : Point) : ℝ :=
  let radius := dist (3, Real.sqrt 3, 0) P in
  4 * Real.pi * radius^2

-- Theorem stating the required result
theorem circumsphere_surface_area : surface_area_of_circumsphere P A B C = 84 * Real.pi := by
  sorry

end circumsphere_surface_area_l728_728135


namespace grasshopper_twenty_five_jumps_l728_728115

noncomputable def sum_natural (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem grasshopper_twenty_five_jumps :
  let total_distance := sum_natural 25
  total_distance % 2 = 1 -> 0 % 2 = 0 -> total_distance ≠ 0 :=
by
  intros total_distance_odd zero_even
  sorry

end grasshopper_twenty_five_jumps_l728_728115


namespace sequence_difference_l728_728142

/-- 
Prove that the difference between the sum of the arithmetic sequence 
from 2001 to 2100 with a common difference of 1 and the sum of the 
arithmetic sequence of odd numbers from 51 to 149 is 200050. 
-/
theorem sequence_difference : 
  (∑ i in finset.range (2100 - 2001 + 1), (2001 + i)) 
  - (∑ i in finset.range ((149 - 51) / 2 + 1), (51 + 2 * i)) = 200050 := 
by 
  sorry

end sequence_difference_l728_728142


namespace correct_average_l728_728828

def incorrect_avg := 25
def num_values := 15
def incorrect_numbers := [26, 62, 24]
def correct_numbers := [86, 92, 48]
def incorrect_sum := incorrect_avg * num_values
def diff := (List.sum correct_numbers) - (List.sum incorrect_numbers)

theorem correct_average : 
  (incorrect_sum + diff) / num_values = 32.6 :=
by
  sorry

end correct_average_l728_728828


namespace bug_total_distance_l728_728905

/-!
# Problem Statement
A bug starts crawling on a number line from position -3. It first moves to -7, then turns around and stops briefly at 0 before continuing on to 8. Prove that the total distance the bug crawls is 19 units.
-/

def bug_initial_position : ℤ := -3
def bug_position_1 : ℤ := -7
def bug_position_2 : ℤ := 0
def bug_final_position : ℤ := 8

theorem bug_total_distance : 
  |bug_position_1 - bug_initial_position| + 
  |bug_position_2 - bug_position_1| + 
  |bug_final_position - bug_position_2| = 19 :=
by 
  sorry

end bug_total_distance_l728_728905


namespace sum_of_fractions_l728_728527

theorem sum_of_fractions :
  (∑ i in Finset.range 10.succ, (i + 1) / 7) = 55 / 7 :=
by
  sorry

end sum_of_fractions_l728_728527


namespace binom_7_4_eq_35_l728_728167

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728167


namespace sequence_increasing_or_decreasing_l728_728461

theorem sequence_increasing_or_decreasing (x : ℕ → ℝ) (h1 : x 1 > 0) (h2 : x 1 ≠ 1) 
  (hrec : ∀ n, x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)) :
  ∀ n, x n < x (n + 1) ∨ x n > x (n + 1) :=
by
  sorry

end sequence_increasing_or_decreasing_l728_728461


namespace cube_surface_area_l728_728910

theorem cube_surface_area {R a : ℝ} 
  (h1 : (4/3) * π * R^3 = 4 * √3 * π) 
  (h2 : √3 * a = 2 * R) : 
  6 * a^2 = 24 := 
by 
  -- proof will be here
  sorry

end cube_surface_area_l728_728910


namespace new_salary_each_employee_l728_728202

theorem new_salary_each_employee (
  emily_initial_salary : ℕ,
  employee_initial_salary : ℕ,
  num_employees : ℕ,
  emily_new_salary : ℕ
)
(h1 : emily_initial_salary = 1000000)
(h2 : employee_initial_salary = 20000)
(h3 : num_employees = 10)
(h4 : emily_new_salary = 850000)
: (employee_initial_salary + ((emily_initial_salary - emily_new_salary) / num_employees) = 35000) :=
by
  sorry

end new_salary_each_employee_l728_728202


namespace correct_article_choice_l728_728050

def small_temple_island_article_choice : Prop :=
  ∃ (first_blank_choice second_blank_choice : string),
    first_blank_choice = "a" ∧ second_blank_choice = "no article"

theorem correct_article_choice : small_temple_island_article_choice :=
  sorry

end correct_article_choice_l728_728050


namespace Ma_Xiaohu_score_l728_728517

theorem Ma_Xiaohu_score :
  (∀ (x : ℤ), (6 / x) > 0 ∧ (∃ (m n : ℤ), (m + n) / (m - n) > 0 ∧ (x ≠ -1 → x / (x + 1) = 1) ∧
  (a b : ℤ), (a^2 + b^2) / (a + b) = 1 ∧ (x | x | - 2) / (x + 2) = 0 → x = 2) ∧
  (∀ (x y : ℤ), (2 * y)^2 / (2 * x + 2 * y) = y^2 / (x + y))
  → 4 * 20 = 80) := 
sorry

end Ma_Xiaohu_score_l728_728517


namespace angle_maximized_l728_728788

-- Definition of points A and B in the plane
variable (A B : Point)

-- Definition of line d that does not intersect segment [A B]
variable (d : Line)
variable (h_d : ¬ intersects_segment d A B)

-- Definition of point M on line d
def on_line (M : Point) : Prop := lies_on M d

-- The goal is to prove that M is the point of tangency that maximizes angle ∠AMB
theorem angle_maximized (M : Point) (hM : on_line d M) :
  maximizes_angle_AMB A B M ↔ tangent_point_of_circumcircle A B d M :=
sorry

end angle_maximized_l728_728788


namespace max_value_sqrt_l728_728610

-- Define the conditions as a predicate
def condition (x : ℝ) : Prop :=
  -49 ≤ x ∧ x ≤ 49

-- Maximum value problem as a statement
theorem max_value_sqrt (x : ℝ) (h : condition x) : 
  sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
sorry

end max_value_sqrt_l728_728610


namespace platform_length_l728_728094

-- Definitions for the given conditions
def train1_speed_kmph : ℕ := 48
def train2_speed_kmph : ℕ := 42
def time_to_cross_trains_seconds : ℕ := 12
def time_to_pass_platform_seconds : ℕ := 45

-- Conversion factor: 1 kmph = 5/18 m/s
def kmph_to_mps (kmph : ℕ) : ℝ := kmph * (5 / 18)

-- Speeds in m/s
def train1_speed_mps : ℝ := kmph_to_mps train1_speed_kmph
def train2_speed_mps : ℝ := kmph_to_mps train2_speed_kmph

-- Relative speed when two trains are moving in opposite directions
def relative_speed_mps : ℝ := train1_speed_mps + train2_speed_mps

-- Length of the first train (to be found in the proof)
noncomputable def L : ℝ := 200  -- Given from solution steps

-- Length of the second train is half the first train
def train2_length : ℝ := L / 2

-- Distance covered when trains pass each other
def distance_when_cross : ℝ := L + train2_length

-- Distance formula: Distance = Speed * Time
def calculated_distance_when_cross : ℝ := relative_speed_mps * (time_to_cross_trains_seconds : ℝ)

-- The distance covered when the first train passes the platform
def platform_speed_distance : ℝ := train1_speed_mps * (time_to_pass_platform_seconds : ℝ)

-- Length of the platform (to be proved)
noncomputable def P : ℝ := 399.85

-- Distance when train passes platform: L + P = Speed * Time
theorem platform_length :
  L + P = platform_speed_distance :=
by
  sorry

end platform_length_l728_728094


namespace part_a_l728_728192

noncomputable def S : ℕ → ℕ
| 0       := 1
| (n + 1) := sorry  -- To be defined according to the given recurrence relation.

axiom S_recurrence : ∀ n, S n.succ = S n + n * S (n - 1)

theorem part_a (n : ℕ) : S (n + 1) = S n + n * S (n - 1) :=
  S_recurrence n

end part_a_l728_728192


namespace cookie_ratio_l728_728887

theorem cookie_ratio (f : ℚ) (h_monday : 32 = 32) (h_tuesday : (f : ℚ) * 32 = 32 * (f : ℚ)) 
    (h_wednesday : 3 * (f : ℚ) * 32 - 4 + 32 + (f : ℚ) * 32 = 92) :
    f = 1/2 :=
by
  sorry

end cookie_ratio_l728_728887


namespace total_interest_proof_l728_728845

variable (P R : ℕ)  -- principal and rate of interest

-- Given conditions
def condition1 : Prop := (P * R * 10) / 100 = 1400
def condition2 : Prop := 15 * P * R / 100 = 21

-- Total interest calculation
def total_interest : ℕ := 1400 + 21

-- Theorem to be proved
theorem total_interest_proof (P R : ℕ) (h1 : condition1 P R) (h2 : condition2 P R) :
  1400 + 21 = 1421 :=
by
  rw [h1, h2]
  exact rfl

end total_interest_proof_l728_728845


namespace sin_expression_positive_l728_728718

theorem sin_expression_positive (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π) :
  sin θ + 1/2 * sin (2 * θ) + 1/3 * sin (3 * θ) > 0 :=
sorry

end sin_expression_positive_l728_728718


namespace points_concyclic_l728_728433

variables {O₁ O₂ O₃ O₄ A B C D : Point}

/-- Centers of the coins -/
axiom centers_coins (O₁ O₂ O₃ O₄ : Point) : true

/-- Collinearity of the points with respective centers -/
axiom collinear_O₁_A_O₂ : collinear {O₁, A, O₂}
axiom collinear_O₂_B_O₃ : collinear {O₂, B, O₃}
axiom collinear_O₃_C_O₄ : collinear {O₃, C, O₄}
axiom collinear_O₄_D_O₁ : collinear {O₄, D, O₁}

/-- The goal is to show that points A, B, C, D are concyclic -/
theorem points_concyclic 
  (h₁ : centers_coins O₁ O₂ O₃ O₄)
  (h₂ : collinear_O₁_A_O₂)
  (h₃ : collinear_O₂_B_O₃)
  (h₄ : collinear_O₃_C_O₄)
  (h₅ : collinear_O₄_D_O₁) : 
  cyclic_quad {A, B, C, D} :=
  sorry

end points_concyclic_l728_728433


namespace roots_of_polynomial_are_divisors_of_constant_term_l728_728494

theorem roots_of_polynomial_are_divisors_of_constant_term (a2 : ℤ) : 
  ∀ x : ℤ, (x ∣ -18) → (x^3 + a2 * x^2 - 7 * x - 18 = 0) → x ∈ {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18} :=
by
  sorry

end roots_of_polynomial_are_divisors_of_constant_term_l728_728494


namespace sqrt_720_simplified_l728_728004

theorem sqrt_720_simplified : Real.sqrt 720 = 6 * Real.sqrt 5 := sorry

end sqrt_720_simplified_l728_728004


namespace problem_1_max_min_values_problem_2_value_l728_728266

def f (x : Real) : Real := -Real.cos x + Real.cos (Real.pi / 2 - x)

theorem problem_1_max_min_values :
  (∀ x, x ∈ Set.Icc (0 : Real) Real.pi → f x ≤ Real.sqrt 2) ∧
  (∃ x, x ∈ Set.Icc (0 : Real) Real.pi ∧ f x = Real.sqrt 2) ∧
  (∀ x, x ∈ Set.Icc (0 : Real) Real.pi → f x ≥ -Real.sqrt 2) ∧
  (∃ x, x ∈ Set.Icc (0 : Real) Real.pi ∧ f x = -Real.sqrt 2) :=
begin
  sorry
end

theorem problem_2_value :
  ∀ x, x ∈ Set.Ioo (0 : Real) (Real.pi / 6) ∧ Real.sin (2 * x) = 1 / 3 →
       f x = - (Real.sqrt 6) / 3 :=
begin
  sorry
end

end problem_1_max_min_values_problem_2_value_l728_728266


namespace find_angle_XOZ_l728_728129

noncomputable def triangle (A B C : Type) : Type := sorry -- defining a triangle on types

variables {X Y Z O : Type}

-- Given conditions
variables (circle_inscribed : ∀ {XYZ : triangle}, inscribed_in_circle center O)
variables (angle_XYZ : ∀ {XYZ : triangle}, measure_angle X Y Z = 70)
variables (angle_YXZ : ∀ {XYZ : triangle}, measure_angle Y X Z = 80)
variables (external_bisector_intersection : ∀ {O : Type}, external_bisector_intersection X Y Z O)

-- Proof statement
theorem find_angle_XOZ : 
  ∀ (X Y Z O : Type) (h : ∀ {XYZ : triangle}, inscribed_in_circle center O) 
  (h1 : ∀ {XYZ : triangle}, measure_angle X Y Z = 70) 
  (h2 : ∀ {XYZ : triangle}, measure_angle Y X Z = 80)
  (h3 : ∀ {O : Type}, external_bisector_intersection X Y Z O),
  measure_angle X O Z = 105 :=
  by
  sorry

end find_angle_XOZ_l728_728129


namespace trig_expression_simplify_l728_728696

theorem trig_expression_simplify (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trig_expression_simplify_l728_728696


namespace multiples_5_or_7_not_35_count_l728_728683

theorem multiples_5_or_7_not_35_count :
  (finset.filter (λ n, (n % 5 = 0 ∨ n % 7 = 0) ∧ n % 35 ≠ 0) (finset.range 3001)).card = 943 :=
by
  sorry

end multiples_5_or_7_not_35_count_l728_728683


namespace cistern_length_is_four_l728_728112

noncomputable def length_of_cistern (width depth total_area : ℝ) : ℝ :=
  let L := ((total_area - (2 * width * depth)) / (2 * (width + depth)))
  L

theorem cistern_length_is_four
  (width depth total_area : ℝ)
  (h_width : width = 2)
  (h_depth : depth = 1.25)
  (h_total_area : total_area = 23) :
  length_of_cistern width depth total_area = 4 :=
by 
  sorry

end cistern_length_is_four_l728_728112


namespace election_majority_l728_728096

theorem election_majority (V : ℝ) 
  (h1 : ∃ w l : ℝ, w = 0.70 * V ∧ l = 0.30 * V ∧ w - l = 174) : 
  V = 435 :=
by
  sorry

end election_majority_l728_728096


namespace sequence_10th_term_is_2_l728_728399

def sequence_step (t : ℕ) : ℕ :=
  if even t then t / 2 else 3 * t + 1

def sequence (n : ℕ) (a₀ : ℕ) : ℕ :=
  Nat.recOn n a₀ (λ n aₙ, sequence_step aₙ)

theorem sequence_10th_term_is_2 : sequence 9 20 = 2 :=
by
  sorry

end sequence_10th_term_is_2_l728_728399


namespace dilution_problem_l728_728020

theorem dilution_problem
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (desired_concentration : ℝ)
  (initial_alcohol_content : initial_concentration * initial_volume / 100 = 4.8)
  (desired_alcohol_content : 4.8 = desired_concentration * (initial_volume + N) / 100)
  (N : ℝ) :
  N = 11.2 :=
sorry

end dilution_problem_l728_728020


namespace range_of_m_l728_728296

open Real

theorem range_of_m (a m y1 y2 : ℝ) (h_a_pos : a > 0)
  (hA : y1 = a * (m - 1)^2 + 4 * a * (m - 1) + 3)
  (hB : y2 = a * m^2 + 4 * a * m + 3)
  (h_y1_lt_y2 : y1 < y2) : 
  m > -3 / 2 := 
sorry

end range_of_m_l728_728296


namespace brothers_in_same_team_probability_l728_728320

theorem brothers_in_same_team_probability :
  let num_ways_to_choose_two n : ℕ := n * (n - 1) / 2
  let total_participants := 10
  let monarch_and_loyalists := 4
  let rebels := 5
  let traitor := 1
  let total_pairs := num_ways_to_choose_two total_participants
  let favorable_pairs := num_ways_to_choose_two monarch_and_loyalists + num_ways_to_choose_two rebels
  total_pairs ≠ 0 → (favorable_pairs.to_rat / total_pairs.to_rat) = (16:ℚ/45:ℚ) :=
by
  sorry

end brothers_in_same_team_probability_l728_728320


namespace factorial_difference_l728_728549

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728549


namespace sqrt_720_eq_12_sqrt_5_l728_728001

theorem sqrt_720_eq_12_sqrt_5 : sqrt 720 = 12 * sqrt 5 :=
by
  sorry

end sqrt_720_eq_12_sqrt_5_l728_728001


namespace goldfish_died_l728_728609

theorem goldfish_died (original left died : ℕ) (h₁ : original = 89) (h₂ : left = 57) : died = original - left :=
by {
  rw [h₁, h₂],
  exact eq.refl _
}

# Check the expected outcome is 32
example : goldfish_died 89 57 32 (by rfl) (by rfl) = (32 : ℕ) := rfl

end goldfish_died_l728_728609


namespace minor_arc_circumference_l728_728358

theorem minor_arc_circumference
  (P Q R : Point)
  (radius : ℝ)
  (h_radius : radius = 12)
  (h_angle : ∠(P, R, Q) = 90) :
  arc_circumference P Q = 12 * π := 
sorry

end minor_arc_circumference_l728_728358


namespace probability_of_drawing_red_ball_l728_728314

theorem probability_of_drawing_red_ball :
  let red_balls := 7
  let black_balls := 3
  let total_balls := red_balls + black_balls
  let probability_red := (red_balls : ℚ) / total_balls
  probability_red = 7 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l728_728314


namespace equilateral_triangle_projections_ratio_l728_728088

theorem equilateral_triangle_projections_ratio (A B C P L M N : Point)
  (hABC : equilateral_triangle A B C)
  (hP_in_triangle : point_in_triangle P A B C)
  (hL : perpendicular_projection P A B L)
  (hM : perpendicular_projection P B C M)
  (hN : perpendicular_projection P C A N) :
  (AP / NL) = (BP / LM) = (CP / MN) :=
sorry

end equilateral_triangle_projections_ratio_l728_728088


namespace two_coins_heads_probability_l728_728081

/-- 
When tossing two coins of uniform density, the probability that both coins land with heads facing up is 1/4.
-/
theorem two_coins_heads_probability : 
  let outcomes := ["HH", "HT", "TH", "TT"]
  let favorable := "HH"
  probability (favorable) = 1/4 :=
by
  sorry

end two_coins_heads_probability_l728_728081


namespace intersection_unique_l728_728893

noncomputable def intersection_point (a b c : ℝ) : ℂ :=
  (1/2 : ℂ) + (complex.I * (a + c + 2 * b) / 4)

theorem intersection_unique (a b c : ℝ) (t : ℝ) (Z0 Z1 Z2 Z : ℂ) (Zt : ℝ → ℂ)
  (hZ0 : Z0 = complex.I * a)
  (hZ1 : Z1 = (1/2 : ℂ) + complex.I * b)
  (hZ2 : Z2 = 1 + complex.I * c)
  (hZ : Z = Z0 * complex.cos t ^ 4 + 2 * Z1 * complex.cos t ^ 2 * complex.sin t ^ 2 + Z2 * complex.sin t ^ 4)
  (hx : x = (complex.cos t ^ 2 * complex.sin t ^ 2 + complex.sin t ^ 4))
  (hy : y = (a * complex.cos t ^ 4 + 2 * b * complex.cos t ^ 2 * complex.sin t ^ 2 + c * complex.sin t ^ 4))
  : Z = intersection_point a b c :=
sorry

end intersection_unique_l728_728893


namespace min_possible_value_l728_728293

theorem min_possible_value (a b : ℤ) (h : a > b) :
  (∃ x : ℚ, x = (2 * a + 3 * b) / (a - 2 * b) ∧ (x + 1 / x = (2 : ℚ))) :=
sorry

end min_possible_value_l728_728293


namespace selling_price_percentage_l728_728413

-- Definitions for conditions
def ratio_cara_janet_jerry (c j je : ℕ) : Prop := 4 * (c + j + je) = 4 * c + 5 * j + 6 * je
def total_money (c j je total : ℕ) : Prop := c + j + je = total
def combined_loss (c j loss : ℕ) : Prop := c + j - loss = 36

-- The theorem statement to be proven
theorem selling_price_percentage (c j je total loss : ℕ) (h1 : ratio_cara_janet_jerry c j je) (h2 : total_money c j je total) (h3 : combined_loss c j loss)
    (h4 : total = 75) (h5 : loss = 9) : (36 * 100 / (c + j) = 80) := by
  sorry

end selling_price_percentage_l728_728413


namespace ten_factorial_minus_nine_factorial_l728_728563

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem ten_factorial_minus_nine_factorial :
  factorial 10 - factorial 9 = 3265920 := 
by 
  sorry

end ten_factorial_minus_nine_factorial_l728_728563


namespace not_sufficient_for_parallelogram_l728_728635

-- Definition of the quadrilateral ABCD
structure Quadrilateral (A B C D : Type) :=
(side1 : A -> B -> Prop)
(side2 : B -> C -> Prop)
(side3 : C -> D -> Prop)
(side4 : D -> A -> Prop)
(parallel : (A -> B -> C -> D -> Prop) -> Prop)
(equal   : (A -> B -> C -> D -> Prop) -> Prop)

-- Theorem statement
theorem not_sufficient_for_parallelogram (A B C D : Type)
  (AB_parallel_CD : Quadrilateral.parallel A B C D)
  (BC_parallel_AD : Quadrilateral.parallel B C A D)
  (AB_eq_CD : Quadrilateral.equal A B C D)
  (BC_eq_AD : Quadrilateral.equal B C A D) :
  ¬ (AB_eq_CD ∧ BC_parallel_AD ∨ AB_parallel_CD ∧ BC_eq_AD) → (¬ Quadrilateral.parallel A B C D ∧ ¬ Quadrilateral.equal A B C D) :=
sorry

end not_sufficient_for_parallelogram_l728_728635


namespace pages_read_in_a_year_l728_728717

-- Definition of the problem conditions
def novels_per_month := 4
def pages_per_novel := 200
def months_per_year := 12

-- Theorem statement corresponding to the problem
theorem pages_read_in_a_year (h1 : novels_per_month = 4) (h2 : pages_per_novel = 200) (h3 : months_per_year = 12) : 
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by
  sorry

end pages_read_in_a_year_l728_728717


namespace probability_three_heads_in_eight_tosses_l728_728932

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l728_728932


namespace find_intersection_l728_728486

theorem find_intersection :
  ∃ t u,
    (2 + t = 4 + 5 * u) ∧
    (3 - 4 * t = -6 + 3 * u) ∧
    (2 + t, 3 - 4 * t) = (185 / 23, 21 / 23) :=
begin
  sorry
end

end find_intersection_l728_728486


namespace distance_A_to_line0_l728_728252

def line1 (x y : ℝ) : Prop := 3 * x + 2 * y + 1 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y - 5 = 0
def line0 (x y : ℝ) : Prop := y = -3 / 4 * x - 5 / 2

def A : ℝ × ℝ := (1, -2)

theorem distance_A_to_line0 : 
  let d := abs ((-3 / 4 * A.1 + A.2 + 5 / 2) / real.sqrt (3^2/16 + 1)) in
  d = 1 := by
  sorry

end distance_A_to_line0_l728_728252


namespace fran_speed_l728_728334

theorem fran_speed :
  ∀ (Joann_speed Fran_time : ℝ), Joann_speed = 15 → Joann_time = 4 → Fran_time = 3.5 →
    (Joann_speed * Joann_time = Fran_time * (120 / 7)) :=
by
  intro Joann_speed Joann_time Fran_time
  sorry

end fran_speed_l728_728334


namespace max_value_of_expr_l728_728641

theorem max_value_of_expr 
  (x y z : ℝ) 
  (h₀ : 0 < x) 
  (h₁ : 0 < y) 
  (h₂ : 0 < z)
  (h : x^2 + y^2 + z^2 = 1) : 
  3 * x * y + y * z ≤ (Real.sqrt 10) / 2 := 
  sorry

end max_value_of_expr_l728_728641


namespace common_divisors_4n_7n_l728_728720

theorem common_divisors_4n_7n (n : ℕ) (h1 : n < 50) 
    (h2 : (Nat.gcd (4 * n + 5) (7 * n + 6) > 1)) :
    n = 7 ∨ n = 18 ∨ n = 29 ∨ n = 40 := 
  sorry

end common_divisors_4n_7n_l728_728720


namespace multiples_of_three_l728_728779

theorem multiples_of_three (a b : ℤ) (h : 9 ∣ (a^2 + a * b + b^2)) : 3 ∣ a ∧ 3 ∣ b :=
by {
  sorry
}

end multiples_of_three_l728_728779


namespace fraction_ordering_l728_728436

theorem fraction_ordering :
  (8 : ℚ) / 31 < (11 : ℚ) / 33 ∧
  (11 : ℚ) / 33 < (12 : ℚ) / 29 ∧
  (8 : ℚ) / 31 < (12 : ℚ) / 29 := 
by  
  sorry

end fraction_ordering_l728_728436


namespace octagon_diagonal_ratio_l728_728737

theorem octagon_diagonal_ratio 
  (s : ℝ) -- Side length of the octagon
  (short_diagonal long_diagonal: ℝ) -- Lengths of shortest and longest diagonals
  (h1 : short_diagonal = s * (1 + 2 * (Real.sin(π/8))))
  (h2 : long_diagonal = s * (1 + 2 * (Real.sin(3 * π/8)))) : 
  short_diagonal / long_diagonal = 1 / 2 := 
  by
    sorry

end octagon_diagonal_ratio_l728_728737


namespace fish_price_decrease_l728_728865

variable (x : ℝ)
variable (fish_valley : ℝ) := 0.85 * x
variable (fur_valley : ℝ) := 0.9 * x
variable (fur_hillside : ℝ) := 0.8 * x
variable (ratio_fur_fish : ℝ) := 18/17

theorem fish_price_decrease :
  (fur_valley / fish_valley = ratio_fur_fish) →
  (fur_hillside / y = ratio_fur_fish) →
  y = 0.7556 * x →
  ((x - y) / x) * 100 = 24.4 := 
by
  intros
  sorry

end fish_price_decrease_l728_728865


namespace lance_hourly_earnings_l728_728585

theorem lance_hourly_earnings
  (hours_per_week : ℕ)
  (workdays_per_week : ℕ)
  (daily_earnings : ℕ)
  (total_weekly_earnings : ℕ)
  (hourly_wage : ℕ)
  (h1 : hours_per_week = 35)
  (h2 : workdays_per_week = 5)
  (h3 : daily_earnings = 63)
  (h4 : total_weekly_earnings = daily_earnings * workdays_per_week)
  (h5 : total_weekly_earnings = hourly_wage * hours_per_week)
  : hourly_wage = 9 :=
sorry

end lance_hourly_earnings_l728_728585


namespace sine_pi_identity_l728_728521

-- Define the necessary variables and functions
variables (α : ℝ)

noncomputable def trigonometric_identity : Prop :=
  sin (π - α) + sin (π + α) = 0

-- State the theorem
theorem sine_pi_identity : trigonometric_identity α :=
by sorry

end sine_pi_identity_l728_728521


namespace max_length_OA_l728_728056

noncomputable def sqrt2 : ℝ := Real.sqrt 2

theorem max_length_OA (O A B : Point) (h_angle_O : ∠ O B A = 45) (h_AB : dist A B = 1) :
  ∃ OA : ℝ, OA = sqrt2 :=
begin
  sorry
end

end max_length_OA_l728_728056


namespace coin_toss_probability_l728_728921

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l728_728921


namespace trig_equation_solution_l728_728013

noncomputable def solve_trig_equation (x : ℝ) : Prop :=
  let lhs := (Real.sin (5 * x) + Real.sin (7 * x)) / (Real.sin (4 * x) - Real.sin (2 * x))
  let rhs := -3 * Real.abs (Real.sin (2 * x))
  lhs = rhs

theorem trig_equation_solution (x k : ℤ) (y : ℝ) :
  (trig_sol1 := fun (k : ℤ) => π - Real.arcsin ((3 - Real.sqrt 57) / 8) + 2 * k * π)
  (trig_sol2 := fun (k : ℤ) => π - Real.arcsin ((Real.sqrt 57 - 3) / 8) + 2 * k * π)
  solve_trig_equation x ↔ 
    (∃ k : ℤ, x = trig_sol1 k) ∨ (∃ k : ℤ, x = trig_sol2 k) :=
begin
  sorry
end

end trig_equation_solution_l728_728013


namespace can_have_1001_free_ends_l728_728248

theorem can_have_1001_free_ends : ∃ k : ℕ, 4 * k + 1 = 1001 :=
by
  use 250
  norm_num
  sorry

end can_have_1001_free_ends_l728_728248


namespace probability_heads_heads_l728_728078

theorem probability_heads_heads (h_uniform_density : ∀ outcome, outcome ∈ {("heads", "heads"), ("heads", "tails"), ("tails", "heads"), ("tails", "tails")} → True) :
  ℙ({("heads", "heads")}) = 1 / 4 :=
sorry

end probability_heads_heads_l728_728078


namespace area_ABC_at_least_twice_area_AKL_l728_728500

-- Define a right triangle ABC with right angle at A
variables {A B C D K L : Type} [RealNormedSpace ℝ A B C D K L]
variables (ABC : Triangle A B C) (right_angle_A : is_right_angle ABC A)
variables (D_foot : is_foot_of_altitude A ABC D)
variables (K_L_on_AB_and_AC : Line (incenter (Triangle A B D)) (incenter (Triangle A C D)) ⤳ Line AB ∩ Line AC = {K, L})

theorem area_ABC_at_least_twice_area_AKL :
  area (Triangle A B C) ≥ 2 * area (Triangle A K L) := sorry

end area_ABC_at_least_twice_area_AKL_l728_728500


namespace factorial_difference_l728_728550

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728550


namespace infinite_solutions_eq_one_l728_728227

theorem infinite_solutions_eq_one (a : ℝ) :
  (∃ᶠ x in filter.at_top, abs (x - 2) = a * x - 2) →
  a = 1 :=
by
  sorry

end infinite_solutions_eq_one_l728_728227


namespace only_statement_1_is_correct_l728_728086

theorem only_statement_1_is_correct :
  (let A := ¬ (∃ x₀ : ℝ, 2^x₀ ≤ 0) ∧ (∀ x : ℝ, 2^x > 0),                               -- Statement 1
       B := ∀ x ∈ Icc (-π/2) (π/2), - (sin ((1 / 2) * x + π / 4)) is_strictly_increasing, -- Statement 2
       C := ∀ x : ℝ, min (x^2 + 4 / sqrt (x^2 + 3)) = 2,                                  -- Statement 3
       D := ∃ k ∈ Ioi 1, ∃ x₁ x₂ x₃ : ℝ,                                                 -- Statement 4
             f (x₁) - k * x₁ = 0 ∧ f (x₂) - k * x₂ = 0 ∧ f (x₃) - k * x₃ = 0 ∧
             x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
             (f (x) = x / (1 + abs x))
  in 
    A ∧ ¬B ∧ ¬C ∧ ¬D) := sorry

end only_statement_1_is_correct_l728_728086


namespace largest_expression_is_A_l728_728083

def expr_A := 1 - 2 + 3 + 4
def expr_B := 1 + 2 - 3 + 4
def expr_C := 1 + 2 + 3 - 4
def expr_D := 1 + 2 - 3 - 4
def expr_E := 1 - 2 - 3 + 4

theorem largest_expression_is_A : expr_A = 6 ∧ expr_A > expr_B ∧ expr_A > expr_C ∧ expr_A > expr_D ∧ expr_A > expr_E :=
  by sorry

end largest_expression_is_A_l728_728083


namespace smallest_n_for_2017_digits_l728_728844

-- Define the sequence (x_n) with given properties
def sequence (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else list.perm.max (nat.digits 10 (sequence (n-1) + 1))

-- Check the length of digit representation of a number
def digit_length (k : ℕ) : ℕ :=
  (nat.digits 10 k).length

-- Define the predicate to find the correct n
def correct_n (n : ℕ) : Prop :=
  digit_length (sequence n) = 2017

theorem smallest_n_for_2017_digits :
  ∃ n : ℕ, correct_n n ∧ (∀ m : ℕ, m < n → ¬correct_n m) :=
  sorry

end smallest_n_for_2017_digits_l728_728844


namespace fran_speed_l728_728336

theorem fran_speed :
  ∀ (Joann_speed Fran_time : ℝ), Joann_speed = 15 → Joann_time = 4 → Fran_time = 3.5 →
    (Joann_speed * Joann_time = Fran_time * (120 / 7)) :=
by
  intro Joann_speed Joann_time Fran_time
  sorry

end fran_speed_l728_728336


namespace find_number_mul_l728_728607

theorem find_number_mul (n : ℕ) (h : n * 9999 = 724777430) : n = 72483 :=
by
  sorry

end find_number_mul_l728_728607


namespace binomial_7_4_eq_35_l728_728152

-- Define the binomial coefficient using the binomial coefficient formula.
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem to prove.
theorem binomial_7_4_eq_35 : binomial_coefficient 7 4 = 35 :=
sorry

end binomial_7_4_eq_35_l728_728152


namespace goals_per_player_is_30_l728_728032

-- Define the total number of goals scored in the league against Barca
def total_goals : ℕ := 300

-- Define the percentage of goals scored by the two players
def percentage_of_goals : ℝ := 0.20

-- Define the combined goals by the two players
def combined_goals := (percentage_of_goals * total_goals : ℝ)

-- Define the number of players
def number_of_players : ℕ := 2

-- Define the number of goals scored by each player
noncomputable def goals_per_player := combined_goals / number_of_players

-- Proof statement: Each of the two players scored 30 goals.
theorem goals_per_player_is_30 :
  goals_per_player = 30 :=
sorry

end goals_per_player_is_30_l728_728032


namespace angle_at_point_of_star_is_216_degrees_l728_728497

-- Definitions for the conditions
def pentagram_inscribed_in_circle : Prop := sorry -- Placeholder for the actual definition
def divides_circle_into_equal_sections (n : ℕ) : Prop := sorry -- Placeholder: asserts the circle is divided into n equal sections
def forms_star_with_sharp_points : Prop := sorry -- Placeholder: asserts the shape forms a star with sharp points

-- Given conditions
axiom h1 : pentagram_inscribed_in_circle
axiom h2 : divides_circle_into_equal_sections 10
axiom h3 : forms_star_with_sharp_points

-- Proof problem statement
theorem angle_at_point_of_star_is_216_degrees :
  (h1 ∧ h2 ∧ h3) → angle_at_star_point = 216 := sorry

end angle_at_point_of_star_is_216_degrees_l728_728497


namespace degree_of_interior_angle_of_regular_octagon_l728_728832

theorem degree_of_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (∀ (A : ℝ), A = (n - 2) * 180 / n → A = 135) :=
by
  intros n hn A hA
  rw [hn, hA]
  sorry

end degree_of_interior_angle_of_regular_octagon_l728_728832


namespace probability_male_or_young_but_not_both_l728_728745

/-- In a village with 5,000 people, the population is divided into:
    - 0-30 years old: 35%
    - 31-50 years old: 30%
    - 51-70 years old: 25%
    - over 70 years old: 10%
    60% are male, 40% are female. Each gender is evenly distributed by age,
    and the probabilities of being in each age group and gender category are independent.
    Prove that the probability a randomly chosen person is either male or younger than 31 years old, but not both, is 53%.
--/
theorem probability_male_or_young_but_not_both 
  (total_population : ℕ := 5000)
  (group1_percent : ℝ := 0.35)
  (group2_percent : ℝ := 0.30)
  (group3_percent : ℝ := 0.25)
  (group4_percent : ℝ := 0.10)
  (male_percent : ℝ := 0.60)
  (female_percent : ℝ := 0.40) :
  let group1_population := total_population * group1_percent,
      group2_population := total_population * group2_percent,
      group3_population := total_population * group3_percent,
      group4_population := total_population * group4_percent,
      male_group1 := male_percent * group1_population / total_population,
      male_group2 := male_percent * group2_population / total_population,
      male_group3 := male_percent * group3_population / total_population,
      male_group4 := male_percent * group4_population / total_population,
      female_group1 := female_percent * group1_population / total_population in
  (male_group2 * group2_percent + male_group3 * group3_percent + male_group4 * group4_percent) + female_group1 = 0.53 :=
by
  -- Add proof here
  sorry

end probability_male_or_young_but_not_both_l728_728745


namespace ball_falls_in_middle_pocket_l728_728311

theorem ball_falls_in_middle_pocket (p q : ℕ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  ∃ k : ℕ, (k * p) % (2 * q) = 0 :=
by
  sorry

end ball_falls_in_middle_pocket_l728_728311


namespace divisible_by_75_count_l728_728323

theorem divisible_by_75_count :
  let stars := {0, 2, 4, 5, 7, 9}
  let fixed_digits_sum := 2 + 0 + 1 + 6 + 0 + 2
  (∃ (x1 x2 x3 x4 x5 ∈ stars), (x1 + x2 + x3 + x4 + x5) % 3 = 2) →
  ((stars.to_finset.card ^ 4) * 2 = 2592) := 
by {
  intro h,
  have h_sum : fixed_digits_sum + 5 = 16 := by norm_num,
  obtain ⟨x1, hx1, x2, hx2, x3, hx3, x4, hx4, x5, hx5, hx_sum⟩ := h,
  have h_mod : (16 + x1 + x2 + x3 + x4 + x5) % 3 = 0 := by simpa,
  have mod_eq : (x1 + x2 + x3 + x4 + x5) % 3 = 2 := hx_sum,
  have possibilities : ∀ x ∈ stars, x % 3 = 0 ∨ x % 3 = 2 ∨ x % 3 = 1,
  {
    intros x hx,
    fin_cases x; simp,
  },
  have count_mod_2 : set.count (λ x, x % 3 = 2) stars = 2 := rfl,
  have configurations := (stars.to_finset.card ^ 4),
  exact configurations * 2 = 2592,
  sorry
}

#eval divisible_by_75_count

end divisible_by_75_count_l728_728323


namespace decreasing_interval_f_l728_728600

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 1

theorem decreasing_interval_f :
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 → f(x₁) > f(x₂)) :=
sorry

end decreasing_interval_f_l728_728600


namespace factorial_subtraction_l728_728559

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_subtraction : factorial 10 - factorial 9 = 3265920 := by
  sorry

end factorial_subtraction_l728_728559


namespace furniture_store_revenue_increase_l728_728451

noncomputable def percentage_increase_in_gross (P R : ℕ) : ℚ :=
  ((0.80 * P) * (1.70 * R) - (P * R)) / (P * R) * 100

theorem furniture_store_revenue_increase (P R : ℕ) :
  percentage_increase_in_gross P R = 36 := 
by
  -- We include the conditions directly in the proof.
  -- Follow theorem from the given solution.
  sorry

end furniture_store_revenue_increase_l728_728451


namespace card_sorting_moves_upper_bound_l728_728854

theorem card_sorting_moves_upper_bound (n : ℕ) (cells : Fin (n+1) → Fin (n+1)) (cards : Fin (n+1) → Fin (n+1)) : 
  (∃ (moves : (Fin (n+1) × Fin (n+1)) → ℕ),
    (∀ (i : Fin (n+1)), moves (i, cards i) ≤ 2 * n - 1) ∧ 
    (cards 0 = 0 → moves (0, 0) = 2 * n - 1) ∧ 
    (∃! start_pos : Fin (n+1) → Fin (n+1), 
      moves (start_pos (n), start_pos (0)) = 2 * n - 1)) := sorry

end card_sorting_moves_upper_bound_l728_728854


namespace divisors_greater_than_8_factorial_l728_728288

theorem divisors_greater_than_8_factorial:
  let f := Nat.factorial
  let n := f 9
  let m := f 8
  (∃ (d : ℕ), d ∣ n ∧ d > m) → (count (λ d, d ∣ n ∧ d > m) (list.range (n+1)) = 8) :=
begin
  intros h,
  sorry
end

end divisors_greater_than_8_factorial_l728_728288


namespace total_amount_is_20_yuan_60_cents_l728_728900

-- Conditions
def ten_yuan_note : ℕ := 10
def five_yuan_notes : ℕ := 2 * 5
def twenty_cent_coins : ℕ := 3 * 20

-- Total amount calculation
def total_yuan : ℕ := ten_yuan_note + five_yuan_notes
def total_cents : ℕ := twenty_cent_coins

-- Conversion rates
def yuan_per_cent : ℕ := 100
def total_cents_in_yuan : ℕ := total_cents / yuan_per_cent
def remaining_cents : ℕ := total_cents % yuan_per_cent

-- Proof statement
theorem total_amount_is_20_yuan_60_cents : total_yuan = 20 ∧ total_cents_in_yuan = 0 ∧ remaining_cents = 60 :=
by
  sorry

end total_amount_is_20_yuan_60_cents_l728_728900


namespace range_of_m_l728_728411

noncomputable def quadratic_function : Type := ℝ → ℝ

variable (f : quadratic_function)

axiom quadratic : ∃ a b : ℝ, ∀ x : ℝ, f x = a * (x-2)^2 + b
axiom symmetry : ∀ x : ℝ, f (x + 2) = f (-x + 2)
axiom condition1 : f 0 = 3
axiom condition2 : f 2 = 1
axiom max_value : ∀ m : ℝ, ∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), f x ≤ 3
axiom min_value : ∀ m : ℝ, ∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), 1 ≤ f x

theorem range_of_m : ∀ m : ℝ, (∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), 1 ≤ f x ∧ f x ≤ 3) → 2 ≤ m ∧ m ≤ 4 :=
by
  intro m
  intro h
  sorry

end range_of_m_l728_728411


namespace trigonometric_identity_l728_728706

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trigonometric_identity_l728_728706


namespace sum_of_terms_l728_728350

open Nat

noncomputable def arithmetic_sequence (a_1 d : ℤ) : ℕ → ℤ
| 0       := a_1
| (n + 1) := arithmetic_sequence a_1 d n + d

noncomputable def sum_arithmetic_sequence (a_1 d : ℤ) : ℕ → ℤ
| 0       := a_1
| (n + 1) := sum_arithmetic_sequence a_1 d n + (a_1 + d * (n + 1))

theorem sum_of_terms (a_1 d : ℤ) (S_3 S_6 : ℤ)
  (h1 : S_3 = 3 * a_1 + 3 * (3 - 1) / 2 * d)
  (h2 : S_6 = 6 * a_1 + 6 * (6 - 1) / 2 * d) :
  sum_arithmetic_sequence a_1 d 8 - sum_arithmetic_sequence a_1 d 5 = 45 :=
  sorry

end sum_of_terms_l728_728350


namespace coin_combination_l728_728126

theorem coin_combination (
  price_candy : ℕ := 45,
  price_gum : ℕ := 35,
  price_chocolate : ℕ := 65,
  price_juice : ℕ := 70,
  price_cookie : ℕ := 80,
  qty_candy : ℕ := 2,
  qty_gum : ℕ := 3,
  qty_chocolate : ℕ := 1,
  qty_juice : ℕ := 2,
  qty_cookie : ℕ := 1
) : 
  ∃ (quarters dimes nickels : ℕ), 
    quarters * 25 + dimes * 10 + nickels * 5 = 
    qty_candy * price_candy + 
    qty_gum * price_gum + 
    qty_chocolate * price_chocolate + 
    qty_juice * price_juice + 
    qty_cookie * price_cookie 
    ∧ quarters = 19 
    ∧ dimes = 0 
    ∧ nickels = 1 := 
  sorry

end coin_combination_l728_728126


namespace cannot_arrive_by_noon_l728_728201

-- Define the conditions
def distance : ℝ := 259
def speed : ℝ := 60
def departure_time : ℝ := 8
def arrival_deadline : ℝ := 12

-- Define the function to compute travel time
def travel_time (d : ℝ) (s : ℝ) : ℝ := d / s

-- The math proof statement
theorem cannot_arrive_by_noon : travel_time distance speed + departure_time > arrival_deadline := 
by
  -- proof omitted
  sorry

end cannot_arrive_by_noon_l728_728201


namespace probability_red_ball_l728_728317

theorem probability_red_ball (red_balls black_balls : ℕ) (h_red : red_balls = 7) (h_black : black_balls = 3) :
  (red_balls.to_rat / (red_balls + black_balls).to_rat) = 7 / 10 :=
by
  sorry

end probability_red_ball_l728_728317


namespace range_sum_l728_728038

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 - 4*x) * (Real.exp (x - 2) - Real.exp (2 - x)) + x + 1

theorem range_sum (m M : ℝ) :
  Set.Icc m M = Set.range (f ∘ (λ x : ℝ, x)) ∧ 
  Set.Icc (-1 : ℝ) (5 : ℝ) ⊆ Set.Icc m M -> 
  m + M = 6 :=
begin
  sorry -- Proof goes here
end

end range_sum_l728_728038


namespace product_slope_one_l728_728429

-- Definitions for lines and slopes
def line1_eqn (m : ℝ) : ℝ → ℝ := λ x, m * x
def line2_eqn (n : ℝ) : ℝ → ℝ := λ x, n * x

-- Condition: The angle L1 makes with the horizontal axis is three times the angle L2 makes.
def angle_condition (m n : ℝ) : Prop := 
  ∃ θ, θ ≠ 0 ∧ m = tan (3 * θ) ∧ n = tan θ

-- Condition: m = 3n.
def slope_condition (m n : ℝ) : Prop :=
  m = 3 * n

-- Prove: The product mn = 1.
theorem product_slope_one (m n : ℝ) (h_angle : angle_condition m n) (h_slope : slope_condition m n) (h_not_horizontal : m ≠ 0 ∧ n ≠ 0) : m * n = 1 :=
by sorry

end product_slope_one_l728_728429


namespace combination_seven_four_l728_728149

theorem combination_seven_four : nat.choose 7 4 = 35 :=
by
  sorry

end combination_seven_four_l728_728149


namespace clock_in_2023_hours_l728_728395

theorem clock_in_2023_hours (current_time : ℕ) (h_current_time : current_time = 3) : 
  (current_time + 2023) % 12 = 10 := 
by {
  -- context: non-computational (time kept in modulo world and not real increments)
  sorry
}

end clock_in_2023_hours_l728_728395


namespace probability_of_three_heads_in_eight_tosses_l728_728981

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l728_728981


namespace sin_double_angle_of_tan_eq_two_l728_728645

theorem sin_double_angle_of_tan_eq_two (α : ℝ) (h : ∃ P : ℝ × ℝ, P ∈ {p : ℝ × ℝ | p.2 = 2 * p.1} ∧ ∃ θ, θ = α ∧ tan θ = 2) : sin (2 * α) = 4 / 5 :=
sorry

end sin_double_angle_of_tan_eq_two_l728_728645


namespace log_equivalence_l728_728262

theorem log_equivalence :
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 :=
by
  sorry

end log_equivalence_l728_728262


namespace probability_exactly_three_heads_l728_728962
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l728_728962


namespace solve_system_of_equations_l728_728208

theorem solve_system_of_equations :
  ∃ (x y : ℤ), 2 * x + y = 7 ∧ 4 * x + 5 * y = 11 ∧ x = 4 ∧ y = -1 :=
by
  sorry

end solve_system_of_equations_l728_728208


namespace function_passes_through_third_quadrant_l728_728275

def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

lemma quadratic_eq {α : ℝ} :
  quadratic 1 (-7/2) (5/2) α = 1 → α ∈ {3} := by
sorry

theorem function_passes_through_third_quadrant (α : ℝ) (f : ℝ → ℝ)
  (h1 : f x = (α^2 - 7/2 * α + 5/2) * x^α)
  (h2 : ∃ x < 0, f x < 0) :
  α = 3 :=
begin
  -- Using the quadratic lemma, we can simplify the proof
  have h_quad : quadratic 1 (-7/2) (3/2) α = 0,
  { simp [quadratic_eq, h1, h2], },
  simp [quadratic_eq] at h_quad,
  -- Apply the quadratic solution conditions here
  have h_solution_set : α ∈ {3} := by
  { sorry },
  exact h_solution_set.elim α id,
end

end function_passes_through_third_quadrant_l728_728275


namespace cost_of_one_bag_of_onions_l728_728410

theorem cost_of_one_bag_of_onions (price_per_onion : ℕ) (total_onions : ℕ) (num_bags : ℕ) (h_price : price_per_onion = 200) (h_onions : total_onions = 180) (h_bags : num_bags = 6) :
  (total_onions / num_bags) * price_per_onion = 6000 := 
  by
  sorry

end cost_of_one_bag_of_onions_l728_728410


namespace exists_infinitely_many_odd_abundant_l728_728579

-- Define the predicate of an abundant number
def is_abundant (n : ℕ) : Prop := ∑ k in (finset.filter (λ d : ℕ, d ∣ n) (finset.range (n+1))), k > 2 * n

-- Exists an odd abundant number and there are infinitely many of them
theorem exists_infinitely_many_odd_abundant : 
  ∃ n : ℕ, odd n ∧ is_abundant n ∧ ∃ f : ℕ → ℕ, (∀ m, odd (f m) ∧ is_abundant (f m)) ∧ function.injective f :=
begin
  sorry
end

end exists_infinitely_many_odd_abundant_l728_728579


namespace num_days_c_worked_l728_728453

noncomputable def dailyWages := (Wa Wb Wc : ℝ) (h_ratio : Wa / Wb / Wc = 3 / 4 / 5) (Wc : ℝ) : Prop := 
  Wa = 3/5 * Wc ∧ Wb = 4/5 * Wc ∧ Wc = 95

noncomputable def totalEarnings := (Wa Wb Wc : ℝ) (Da Db Dc : ℕ) : ℝ :=
  Wa * Da + Wb * Db + Wc * Dc

noncomputable def cDaysWorked := 
  (a_days : ℕ) (b_days : ℕ) (Wc : ℝ) (total_earnings : ℝ) (Wa Wb Wc : ℝ) : ℕ :=
  let Ta := Wa * a_days
  let Tb := Wb * b_days
  let Tc := total_earnings - Ta - Tb
  let Dc := Tc / Wc
  Dc

theorem num_days_c_worked :
  ∃ (Dc : ℕ), Dc = 4 :=
  let Wa := (3/5) * 95
  let Wb := (4/5) * 95
  let Wc := 95
  let total_earnings := 1406
  let a_days := 6
  let b_days := 9
  let Ta := Wa * a_days
  let Tb := Wb * b_days
  let Tc := total_earnings - Ta - Tb
  have h1 : Dc = cDaysWorked a_days b_days Wc total_earnings Wa Wb Wc,
  have h2 : Dc = 4 from sorry, -- Apply calculations here
  ⟨h2⟩

end num_days_c_worked_l728_728453


namespace problem_conditions_imply_conclusions_l728_728623

variable {R : Type} [LinearOrder R] [AddGroup R] [AddAction R ℝ] 
variable (f : ℝ → ℝ)

-- The given conditions
def is_monotonically_increasing_on_neg1_0 (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 <= x ∧ x <= 0 → -1 <= y ∧ y <= 0 → x < y → f(x) < f(y)

def symmetry_about_x1 (f : ℝ → ℝ) : Prop :=
  ∀ x, f(1 + x) = f(1 - x)

def symmetry_about_2_0 (f : ℝ → ℝ) : Prop :=
  ∀ x, f(2 + x) = f(2 - x)

-- The conclusions we need to prove
theorem problem_conditions_imply_conclusions 
  (h1 : is_monotonically_increasing_on_neg1_0 f)
  (h2 : symmetry_about_x1 f)
  (h3 : symmetry_about_2_0 f) : 
  (f(0) = f(2)) ∧ 
  (∀ x y, 1 < x ∧ x ≤ 2 → 1 < y ∧ y ≤ 2 → x < y → f(x) > f(y)) ∧ 
  (f(2021) > f(2022) ∧ f(2022) > f(2023)) := 
sorry

end problem_conditions_imply_conclusions_l728_728623


namespace volume_of_larger_prism_is_correct_l728_728190

noncomputable def volume_of_larger_solid : ℝ :=
  let A := (0, 0, 0)
  let B := (2, 0, 0)
  let C := (2, 2, 0)
  let D := (0, 2, 0)
  let E := (0, 0, 2)
  let F := (2, 0, 2)
  let G := (2, 2, 2)
  let H := (0, 2, 2)
  let P := (1, 1, 1)
  let Q := (1, 0, 1)
  
  -- Assume the plane equation here divides the cube into equal halves
  -- Calculate the volume of one half of the cube
  let volume := 2 -- This represents the volume of the larger solid

  volume

theorem volume_of_larger_prism_is_correct :
  volume_of_larger_solid = 2 :=
sorry

end volume_of_larger_prism_is_correct_l728_728190


namespace distinct_prime_factors_2310_l728_728676

theorem distinct_prime_factors_2310 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ (S.card = 5) ∧ (S.prod id = 2310) := by
  sorry

end distinct_prime_factors_2310_l728_728676


namespace binomial_7_4_equals_35_l728_728174

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Problem statement: Prove that binomial(7, 4) = 35
theorem binomial_7_4_equals_35 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_equals_35_l728_728174


namespace cos_C_value_l728_728325

theorem cos_C_value (A B C: ℝ)
  (h1: sin A = 4 / 5)
  (h2: cos B = 5 / 13)
  (h3: A + B + C = π) :
  cos C = 33 / 65 :=
by
  sorry

end cos_C_value_l728_728325


namespace clock_in_2023_hours_l728_728396

theorem clock_in_2023_hours (current_time : ℕ) (h_current_time : current_time = 3) : 
  (current_time + 2023) % 12 = 10 := 
by {
  -- context: non-computational (time kept in modulo world and not real increments)
  sorry
}

end clock_in_2023_hours_l728_728396


namespace least_positive_theta_l728_728597

theorem least_positive_theta :
  (∃ θ : ℝ, θ > 0 ∧ cos 10 = sin 35 + sin θ ∧ θ = 32.5) :=
by
  sorry

end least_positive_theta_l728_728597


namespace scientific_notation_1300000_l728_728897

theorem scientific_notation_1300000 :
  1300000 = 1.3 * 10^6 :=
sorry

end scientific_notation_1300000_l728_728897


namespace ellipse_equation_l728_728631

def is_ellipse (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → (y^2 = 8 * x) → (x = 2)

theorem ellipse_equation
  (m n : ℝ) (h_m_pos : 0 < m) (h_n_pos : 0 < n)
  (eccentricity : ℝ) (h_ecc : eccentricity = 1 / 2)
  (focus_coordinates : ℝ × ℝ) (h_focus : focus_coordinates = (2, 0)) :
  is_ellipse 16 12 h_m_pos h_n_pos := 
by
  sorry

end ellipse_equation_l728_728631


namespace coin_toss_probability_l728_728950

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l728_728950


namespace distances_equal_l728_728638

variable (A B C D X Y P M : Type)
variable [NonIsoscelesAcuteTriangle A B C]
variable (EulerLine : Line)
variable (hD : OnProjection A EulerLine D)
variable (Gamma : Circle)
variable (S : Point)
variable (hGamma : Gamma.Center = S ∧ Gamma.OnCircumference A ∧ Gamma.OnCircumference D)
variable (hX : OnIntersection Gamma AB X)
variable (hY : OnIntersection Gamma AC Y)
variable (hP : OnProjection A BC P)
variable (hM : Midpoint BC M)

theorem distances_equal :
  dist (Circumcenter (triangle X S Y)) P = dist (Circumcenter (triangle X S Y)) M := 
  sorry

end distances_equal_l728_728638


namespace distance_between_points_l728_728397

-- Define the parametric equations for the line
def x(t: ℝ): ℝ := 2 + 3 * t
def y(t: ℝ): ℝ := -1 + t

-- Define the coordinates for t = 0 and t = 1
def x1: ℝ := x(0)
def y1: ℝ := y(0)
def x2: ℝ := x(1)
def y2: ℝ := y(1)

-- State the theorem that distance between the two points is sqrt(10)
theorem distance_between_points : real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = real.sqrt 10 := by
  sorry

end distance_between_points_l728_728397


namespace polygon_area_ratio_l728_728822

theorem polygon_area_ratio 
  (s : ℝ) 
  (D_midpoint_HE : D.x = (H.x + E.x)/2 ∧ D.y = (H.y + E.y)/2)
  (C_midpoint_FG : C.x = (F.x + G.x)/2 ∧ C.y = (F.y + G.y)/2) :
  is_ratio_of_area_AJICB_to_area_of_three_squares (1 / 4) :=
begin
  assume (s > 0),
  sorry
end

end polygon_area_ratio_l728_728822


namespace monotonic_intervals_and_single_zero_point_l728_728269

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 3 * (x^2 + x + 1)

theorem monotonic_intervals_and_single_zero_point :
  (∀ x ∈ Ioo (3 - 2*Real.sqrt 3) (3 + 2*Real.sqrt 3), deriv f x < 0) ∧
  (∀ x ∈ (Iio (3 - 2*Real.sqrt 3) ∪ Ioi (3 + 2*Real.sqrt 3)), deriv f x > 0) ∧
  ∃! x : ℝ, f x = 0 :=
  sorry

end monotonic_intervals_and_single_zero_point_l728_728269


namespace trigonometric_identity_l728_728700

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = (2 / 5) :=
by
  sorry

end trigonometric_identity_l728_728700


namespace sum_first_100_terms_l728_728833

open Nat

def a_n (n : ℕ) : ℤ := (-1)^(n-1) * (4 * n - 3)

def S (n : ℕ) : ℤ := ∑ i in range 1 (n+1), a_n i

-- The statement to prove
theorem sum_first_100_terms : S 100 = -200 := by
  sorry

end sum_first_100_terms_l728_728833


namespace difference_between_numbers_l728_728045

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 24365) (h2 : a % 5 = 0) (h3 : (a / 10) = 2 * b) : a - b = 19931 :=
by sorry

end difference_between_numbers_l728_728045


namespace symmetric_point_origin_l728_728216

noncomputable def point_symmetric (x y z : ℝ) : (ℝ × ℝ × ℝ) :=
  (-x, -y, -z)

theorem symmetric_point_origin:
  let M := (2, -3, 1)
  ∃ N : ℝ × ℝ × ℝ, N = (-2, 3, -1) ∧ point_symmetric (2) (-3) (1) = N :=
begin
  sorry
end

end symmetric_point_origin_l728_728216


namespace coin_toss_probability_l728_728913

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l728_728913


namespace remainder_theorem_division_l728_728625

-- Definitions and conditions
variables {q : ℝ → ℝ}

-- The problem's conditions
def condition_one : Prop := q 2 = 3
def condition_two : Prop := q (-3) = -9

-- The main theorem statement
theorem remainder_theorem_division (q : ℝ → ℝ)
  (h1 : condition_one)
  (h2 : condition_two) :
  ∃ c d, q(x) = (x - 2) * (x + 3) * r(x) + c * x + d ∧
    (q 2 = 3) ∧
    (q (-3) = -9) ∧
    (c = 12 / 5) ∧
    (d = -9 / 5) :=
sorry

end remainder_theorem_division_l728_728625


namespace Xiaohuo_books_l728_728090

def books_proof_problem : Prop :=
  ∃ (X_H X_Y X_Z : ℕ), 
    (X_H + X_Y + X_Z = 1248) ∧ 
    (X_H = X_Y + 64) ∧ 
    (X_Y = X_Z - 32) ∧ 
    (X_H = 448)

theorem Xiaohuo_books : books_proof_problem :=
by
  sorry

end Xiaohuo_books_l728_728090


namespace reach_any_position_l728_728060

/-- We define a configuration of marbles in terms of a finite list of natural numbers, which corresponds to the number of marbles in each hole. A configuration transitions to another by moving marbles from one hole to subsequent holes in a circular manner. -/
def configuration (n : ℕ) := List ℕ 

/-- Define the operation of distributing marbles from one hole to subsequent holes. -/
def redistribute (l : configuration n) (i : ℕ) : configuration n :=
  sorry -- The exact redistribution function would need to be implemented based on the conditions.

theorem reach_any_position (n : ℕ) (m : ℕ) (init_config final_config : configuration n)
  (h_num_marbles : init_config.sum = m)
  (h_final_marbles : final_config.sum = m) :
  ∃ steps, final_config = (steps : List ℕ).foldl redistribute init_config :=
sorry

end reach_any_position_l728_728060


namespace min_max_values_l728_728025

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem min_max_values : 
  (∃ (a b : ℝ), 
    (∀ x ∈ Icc (-3 : ℝ) 3, f x ≥ a) ∧ 
    (∀ x ∈ Icc (-3 : ℝ) 3, f x ≤ b) ∧ 
    (∃ x ∈ Icc (-3 : ℝ) 3, f x = a) ∧ 
    (∃ x ∈ Icc (-3 : ℝ) 3, f x = b)) :=
by
  sorry

end min_max_values_l728_728025


namespace larger_angle_by_diagonal_bisecting_angle_l728_728744

-- Definitions based on conditions in the problem
structure Trapezoid (AB CD : ℝ) :=
(length_ratio : CD = 2 * AB)
(angle_relation : ∃ α, ∃ β, α = (3/2) * β)

-- Theorem statement according to the problem translation
theorem larger_angle_by_diagonal_bisecting_angle
  (AB CD : ℝ)
  (h1 : Trapezoid AB CD)
  (α β : ℝ) 
  (h2 : α = (3 / 2) * β)
  (h3 : ∃ P, segment_bisects_angle AB CD P) :
  ∃ (CAD ABC : ℝ), CAD > ABC :=
sorry

end larger_angle_by_diagonal_bisecting_angle_l728_728744


namespace factorial_difference_l728_728534

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728534


namespace correct_computation_l728_728519

theorem correct_computation (x : ℕ) (h : x - 20 = 52) : x / 4 = 18 :=
  sorry

end correct_computation_l728_728519


namespace crates_with_apples_l728_728117

theorem crates_with_apples :
  ∀ (crates : ℕ) (min_apples max_apples : ℕ) (n : ℕ),
  crates = 175 →
  min_apples = 110 →
  max_apples = 148 →
  n = 4 →
  (∀ (apple_count : ℕ), min_apples ≤ apple_count ∧ apple_count ≤ max_apples → 
    ∃ (count : ℕ), count ≥ n ∧ (count <= crates) ∧ (crates ≤ (max_apples - min_apples + 1) * count)) :=
by
  intros crates min_apples max_apples n h_crates h_min_apples h_max_apples h_n apple_count h_range
  have h_range_length : max_apples - min_apples + 1 = 39 := by sorry
  have h_gte_n : ∃ (count : ℕ), count = n ∧ n = 4 := by sorry
  exact ⟨4, h_gte_n, le_refl 175, le_of_eq h_range_length⟩

end crates_with_apples_l728_728117


namespace simplify_expression_l728_728382

-- Define the initial expression
def expr (q : ℚ) := (5 * q^4 - 4 * q^3 + 7 * q - 8) + (3 - 5 * q^2 + q^3 - 2 * q)

-- Define the simplified expression
def simplified_expr (q : ℚ) := 5 * q^4 - 3 * q^3 - 5 * q^2 + 5 * q - 5

-- The theorem stating that the two expressions are equal
theorem simplify_expression (q : ℚ) : expr q = simplified_expr q :=
by
  sorry

end simplify_expression_l728_728382


namespace combination_seven_four_l728_728150

theorem combination_seven_four : nat.choose 7 4 = 35 :=
by
  sorry

end combination_seven_four_l728_728150


namespace factorial_difference_l728_728540

-- Define factorial function for natural numbers
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- The theorem we want to prove
theorem factorial_difference : factorial 10 - factorial 9 = 3265920 :=
by
  rw [factorial, factorial]
  sorry

end factorial_difference_l728_728540


namespace triangle_AX_length_l728_728321

noncomputable def length_AX (AB AC BC : ℝ) (h1 : AB = 60) (h2 : AC = 34) (h3 : BC = 52) : ℝ :=
  1020 / 43

theorem triangle_AX_length 
  (AB AC BC AX : ℝ)
  (h1 : AB = 60)
  (h2 : AC = 34)
  (h3 : BC = 52)
  (h4 : AX + (AB - AX) = AB)
  (h5 : AX / (AB - AX) = AC / BC) :
  AX = 1020 / 43 := 
sorry

end triangle_AX_length_l728_728321


namespace find_number_l728_728861

theorem find_number : 
  let n := 40 in
  n = λ x : ℚ, x = (80 - 0.25 * 80) → (3 / 2) * x := 40 := λ h, begin
  sorry
end

end find_number_l728_728861


namespace two_coins_heads_probability_l728_728079

/-- 
When tossing two coins of uniform density, the probability that both coins land with heads facing up is 1/4.
-/
theorem two_coins_heads_probability : 
  let outcomes := ["HH", "HT", "TH", "TT"]
  let favorable := "HH"
  probability (favorable) = 1/4 :=
by
  sorry

end two_coins_heads_probability_l728_728079


namespace min_percentage_both_physics_and_chemistry_l728_728514

theorem min_percentage_both_physics_and_chemistry (P C : ℝ) 
  (hP : P = 0.68) (hC : C = 0.72) : ∃ x, x = P + C - 1 ∧ x = 0.40 :=
by
  use 0.40
  split
  sorry

end min_percentage_both_physics_and_chemistry_l728_728514


namespace correct_statements_l728_728812

def vector_problem := 
  let a := (1, 2)
  let b := (1, 1)
  let c := (-2, 6)
  let parallel (v1 v2 : ℝ × ℝ) := v1.1 * v2.2 - v1.2 * v2.1 = 0
  let dot (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2
  let magnitude (v : ℝ × ℝ) := real.sqrt (v.1^2 + v.2^2)
  let acute (v1 v2 : ℝ × ℝ) := dot v1 v2 > 0
  have h1 : ¬(a • b = a • c → b = c) := sorry,
  have h2 : ∀ k, (a.1, k) • c = (a.1, k).1 * c.1 + (a.1, k).2 * c.2 → k = -3 := sorry,
  have h3 : ∀ u v: ℝ × ℝ, u ≠ 0 ∧ v ≠ 0 ∧ (magnitude u = magnitude v) ∧ (magnitude u = magnitude (u - v)) 
    → real.arccos ((dot u (u + v)) / (magnitude u * magnitude (u + v))) = real.pi / 6 := sorry,
  have h4 : ∀ λ : ℝ, acute a (a + (λ • b)) ↔ λ > -5/3 := sorry,
  false.elim

theorem correct_statements : vector_problem := sorry

end correct_statements_l728_728812


namespace jimmy_change_proof_l728_728767

noncomputable def change_jimmy_gets_back : ℝ :=
  let pen_cost := 5 * 1.50
  let notebook_cost := 6 * 3.75
  let folder_cost := 4 * 4.25
  let highlighter_cost := 3 * 2.50
  let total_cost := pen_cost + notebook_cost + folder_cost + highlighter_cost
  let discount := 0.15 * total_cost
  let discounted_total := total_cost - discount
  let tax := 0.07 * discounted_total
  let final_total := discounted_total + tax
  let change := 100 - final_total
  real.floor (change * 100) / 100

theorem jimmy_change_proof : change_jimmy_gets_back = 50.43 := sorry

end jimmy_change_proof_l728_728767


namespace exists_valid_board_configuration_l728_728807

open Matrix

def board (m n : Nat) := Matrix (Fin m) (Fin n) Bool

def is_adjacent {m n : Nat} (i j : Fin m) (k l : Fin n) : Prop :=
  (i = k ∧ abs (j.val - l.val) = 1) ∨ (j = l ∧ abs (i.val - k.val) = 1)

def valid_configuration {m n : Nat} (b : board m n) : Prop :=
  ∀ (i : Fin m) (j : Fin n), b i j = false → 
  ∃ (k : Fin m) (l : Fin n), is_adjacent i j k l ∧ b k l = true

def example_board : board 4 6
| ⟨0,_⟩, ⟨1,_⟩ := true
| ⟨0,_⟩, ⟨3,_⟩ := true
| ⟨1,_⟩, ⟨1,_⟩ := true
| ⟨1,_⟩, ⟨3,_⟩ := true
| ⟨2,_⟩, ⟨1,_⟩ := true
| ⟨2,_⟩, ⟨3,_⟩ := true
| ⟨3,_⟩, ⟨1,_⟩ := true
| _, _ := false

theorem exists_valid_board_configuration : 
  valid_configuration example_board := sorry

end exists_valid_board_configuration_l728_728807


namespace area_of_fourth_rectangle_l728_728432

theorem area_of_fourth_rectangle
  (PR PQ : ℝ)
  (hPR : PR^2 = 25)
  (hPQ : PQ^2 = 49)
  (RS : ℝ)
  (hRS : RS^2 = 64) :
  ∃ PS : ℝ, PS^2 = 89 :=
by
  have PR := Real.sqrt 25
  have RS := Real.sqrt 64
  have PS := Real.sqrt (Real.pow PR 2 + Real.pow RS 2)
  use PS
  have A1 : (PR^2 + RS^2) = 89
  sorry

end area_of_fourth_rectangle_l728_728432


namespace factorial_difference_l728_728528

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728528


namespace intersection_of_A_and_B_l728_728722

open Set

def A := {1, 2, 3, 4}
def B := {x : ℕ | |x| ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l728_728722


namespace total_weight_of_rhinos_l728_728372

def white_rhino_weight : ℕ := 5100
def black_rhino_weight : ℕ := 2000

theorem total_weight_of_rhinos :
  7 * white_rhino_weight + 8 * black_rhino_weight = 51700 :=
by
  sorry

end total_weight_of_rhinos_l728_728372


namespace Mike_savings_l728_728798

theorem Mike_savings
  (price_book1 : ℝ := 33)
  (price_book2 : ℝ)
  (discount_rate : ℝ := 0.5)
  (total_discount : ℝ := 0.2) :
  let 
    paid_books := price_book1 + discount_rate * price_book2
    full_price_books := price_book1 + price_book2
    saved_amount := full_price_books - paid_books
  in saved_amount = (total_discount * full_price_books) ↔ saved_amount = 11 := 
by
  sorry

end Mike_savings_l728_728798


namespace min_value_expression_sin_cos_cot_l728_728215

open Real

theorem min_value_expression_sin_cos_cot (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) :
  ∃ θ, (3 * sin θ + 2 / cos θ + 2 * sqrt 3 * cot θ) = 6 * real.cbrt (sqrt 3) := by
  sorry

end min_value_expression_sin_cos_cot_l728_728215


namespace perp_trans_l728_728726

variables (a b c : Type) [InnerProductSpace ℝ a] [InnerProductSpace ℝ b] [InnerProductSpace ℝ c]

-- Assume (a ⊥ b) and (b ∥ c), we want to prove (a ⊥ c)
theorem perp_trans {a b c : Type} [InnerProductSpace ℝ a] [InnerProductSpace ℝ b] [InnerProductSpace ℝ c]
  (h₁ : ∀ (x y : a), ⟪x, y⟫ = 0)
  (h₂ : ∀ (x y : b), x = y) :
  ∀ (x z : c), ⟪x, z⟫ = 0 := 
by
  sorry

end perp_trans_l728_728726


namespace probability_of_exactly_three_heads_l728_728923

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l728_728923


namespace maximize_xz_l728_728098

-- Define the problem
theorem maximize_xz (x y z t : ℝ) 
  (h1 : x^2 + y^2 = 4)
  (h2 : z^2 + t^2 = 9)
  (h3 : x * t + y * z = 6) :
  x + z = sqrt 13 :=
sorry

end maximize_xz_l728_728098


namespace regular_tetrahedron_has_4_faces_l728_728681

-- Define a regular tetrahedron in terms of its property (which we assume)
constant RegularTetrahedron : Type
constant isRegularTetrahedron : RegularTetrahedron → Prop

-- Define a function that counts the faces of a polyhedron
constant numberOfFaces : RegularTetrahedron → ℕ

-- State the theorem, i.e., the number of faces in a regular tetrahedron is 4
theorem regular_tetrahedron_has_4_faces (T : RegularTetrahedron) (h : isRegularTetrahedron T) : numberOfFaces T = 4 := 
sorry

end regular_tetrahedron_has_4_faces_l728_728681


namespace binomial_expansion_problem_l728_728651

theorem binomial_expansion_problem :
  (∃ (a : Fin 7 → ℤ), 
   (∀ x, (3 * x - 2) ^ 6 = ∑ i, a i * x ^ i) ∧ 
   a 0 + a 1 + 2 * a 2 + 3 * a 3 + 4 * a 4 + 5 * a 5 + 6 * a 6 = 82) :=
sorry

end binomial_expansion_problem_l728_728651


namespace trigonometric_identity_l728_728703

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = (2 / 5) :=
by
  sorry

end trigonometric_identity_l728_728703


namespace min_pos_period_of_func_l728_728405

noncomputable def min_period (f : ℝ → ℝ) : ℝ := 
  inf {T : ℝ | T > 0 ∧ ∀ x, f (x + T) = f x}

theorem min_pos_period_of_func : min_period (λ x => (Real.sin (x + π/3) * Real.sin (x + π/2))) = π := by
  sorry

end min_pos_period_of_func_l728_728405


namespace correct_statements_l728_728884

theorem correct_statements (a b m : ℝ) (f g : ℝ → ℝ) (h1 : a > b ∧ b > 0 ∧ m > 0)
                           (h2 : ∀ x, f x = (2 * x - 3) / (x - 1))
                           (h3 : ∀ x, g x = sqrt (x - 1) * sqrt (x + 1))
                           (h4 : ∀ x, g x = sqrt (x^2 - 1)) :
    (∀ x, (f x ≠ 2 ∧ (f x ∈ (-∞, 2) ∨ f x ∈ (2, ∞)))) ∧
    (∀ x, (x < 0 → false) ∨ (x^2 - 1 < 0 → false)) :=
by
    sorry

end correct_statements_l728_728884


namespace evaluate_at_minus_two_l728_728240

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem evaluate_at_minus_two : f (-2) = -1 := 
by 
  unfold f 
  sorry

end evaluate_at_minus_two_l728_728240


namespace line_intersects_ellipse_slopes_l728_728487

theorem line_intersects_ellipse_slopes :
  {m : ℝ | ∃ x, 4 * x^2 + 25 * (m * x + 8)^2 = 100} = 
  {m : ℝ | m ≤ -Real.sqrt 2.4 ∨ Real.sqrt 2.4 ≤ m} := 
by
  sorry

end line_intersects_ellipse_slopes_l728_728487


namespace total_letters_received_l728_728284

theorem total_letters_received 
  (Brother_received Greta_received Mother_received : ℕ) 
  (h1 : Greta_received = Brother_received + 10)
  (h2 : Brother_received = 40)
  (h3 : Mother_received = 2 * (Greta_received + Brother_received)) :
  Brother_received + Greta_received + Mother_received = 270 := 
sorry

end total_letters_received_l728_728284


namespace quad_completion_l728_728842

theorem quad_completion (a b c : ℤ) 
    (h : ∀ x : ℤ, 8 * x^2 - 48 * x - 128 = a * (x + b)^2 + c) : 
    a + b + c = -195 := 
by
  sorry

end quad_completion_l728_728842


namespace fraction_of_female_parrots_l728_728313

theorem fraction_of_female_parrots 
  (B : ℕ)
  (H1 : ∀ p t : ℕ, 3 * B / 5 = p ∧ 2 * B / 5 = t)
  (H2 : ∀ f : ℚ, 3 * B / 4 / 5 = 3 * B / 10)
  (H3 : ∑ p t : ℕ, p + t = B)
  (H4 : ∀ m : ℚ, m / 2 = 1 / 2) :
  ∑ F : ℚ, F * 3 / 5 + 3 / 10 = 1 / 2 → F = 1 / 3 :=
by
  sorry

end fraction_of_female_parrots_l728_728313


namespace triangle_median_length_l728_728867

noncomputable def length_of_median_DM : ℝ :=
  2 * Real.sqrt 30

theorem triangle_median_length (DE DF EF : ℝ) (h1 : DE = 13) (h2 : DF = 13) (h3 : EF = 14) :
    let DM := (2 : ℝ) * Real.sqrt 30
in DM = length_of_median_DM :=
by
  sorry

end triangle_median_length_l728_728867


namespace probability_of_two_red_shoes_l728_728456

theorem probability_of_two_red_shoes (total_shoes : ℕ) (red_shoes : ℕ) (green_shoes : ℕ) 
    (first_red_probability second_red_probability : ℚ) :
    total_shoes = 8 → red_shoes = 4 → green_shoes = 4 →
    first_red_probability = (4 / 8 : ℚ) →
    second_red_probability = (3 / 7 : ℚ) →
    (first_red_probability * second_red_probability = (3 / 14 : ℚ)) :=
begin
    intros h1 h2 h3 h4 h5,
    rw [h1, h2, h3, h4, h5],
    norm_num,
    linarith,
    sorry,
end

end probability_of_two_red_shoes_l728_728456


namespace average_speed_round_trip_l728_728891

theorem average_speed_round_trip (D : ℝ) (h1 : 5 > 0) (h2 : 100 > 0) :
  let distance := 2 * D,
      time_out := D / 5,
      time_back := D / 100,
      total_time := time_out + time_back,
      average_speed := distance / total_time
  in average_speed = 200 / 21 :=
by {
  sorry
}

end average_speed_round_trip_l728_728891


namespace probability_red_ball_l728_728316

theorem probability_red_ball (red_balls black_balls : ℕ) (h_red : red_balls = 7) (h_black : black_balls = 3) :
  (red_balls.to_rat / (red_balls + black_balls).to_rat) = 7 / 10 :=
by
  sorry

end probability_red_ball_l728_728316


namespace slope_of_intersection_points_l728_728225

theorem slope_of_intersection_points {s x y : ℝ} 
  (h1 : 2 * x - 3 * y = 6 * s - 5) 
  (h2 : 3 * x + y = 9 * s + 4) : 
  ∃ m : ℝ, m = 3 ∧ (∀ s : ℝ, (∃ x y : ℝ, 2 * x - 3 * y = 6 * s - 5 ∧ 3 * x + y = 9 * s + 4) → y = m * x + (23/11)) := 
by
  sorry

end slope_of_intersection_points_l728_728225


namespace origami_papers_per_cousin_l728_728879

theorem origami_papers_per_cousin :
  ∀ (P C : ℝ), P = 48 ∧ C = 6 → P / C = 8 := 
by
  intros P C h
  cases h with hP hC
  rw [hP, hC]
  norm_num
  sorry

end origami_papers_per_cousin_l728_728879


namespace quadratic_common_root_l728_728644

theorem quadratic_common_root (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h1 : ∃ x, x^2 + a * x + b = 0 ∧ x^2 + c * x + a = 0)
  (h2 : ∃ x, x^2 + a * x + b = 0 ∧ x^2 + b * x + c = 0)
  (h3 : ∃ x, x^2 + b * x + c = 0 ∧ x^2 + c * x + a = 0) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end quadratic_common_root_l728_728644


namespace abs_val_of_5_minus_e_l728_728522

theorem abs_val_of_5_minus_e : ∀ (e : ℝ), e = 2.718 → |5 - e| = 2.282 :=
by
  intros e he
  sorry

end abs_val_of_5_minus_e_l728_728522


namespace type_A_to_type_B_time_ratio_l728_728728

def total_examination_time : ℝ := 3 * 60  -- Examination time in minutes.
def time_spent_on_type_A := 25.116279069767444  -- Time spent on type A problems.

noncomputable def time_spent_on_type_B : ℝ := total_examination_time - time_spent_on_type_A

noncomputable def time_ratio : ℝ := time_spent_on_type_A / time_spent_on_type_B

theorem type_A_to_type_B_time_ratio :
  time_ratio ≈ 0.162 := by
  sorry

end type_A_to_type_B_time_ratio_l728_728728


namespace arrangements_4x4_grid_l728_728618

-- The cells in the grid are indexed as (i, j) for i, j in {1, 2, 3, 4}
-- and the letters are represented by 'A', 'B', 'C', 'D'

def in_grid (i j : ℕ) : Prop := 1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4

noncomputable def unique_rows_columns (grid : ℕ → ℕ → char) : Prop :=
  ∀ i j1 j2, j1 ≠ j2 → grid i j1 ≠ grid i j2 ∧
  ∀ j i1 i2, i1 ≠ i2 → grid i1 j ≠ grid i2 j

noncomputable def valid_placement (grid : ℕ → ℕ → char) : Prop :=
  grid 1 1 = 'A' ∧ 
  (∀ i j, in_grid i j → 
    grid i j ∈ {'A', 'B', 'C', 'D'}) ∧ 
  unique_rows_columns grid

theorem arrangements_4x4_grid :
  ∃ (grid : ℕ → ℕ → char), valid_placement grid ∧
  (∃ n : ℕ, n = 12) :=
by {
  -- Definitions and conditions of the problem must be used to prove this theorem
  sorry
}

end arrangements_4x4_grid_l728_728618


namespace smallest_b_for_shift_l728_728017

def g : ℝ → ℝ := sorry

-- Given condition: \( g(x) \) is periodic with a period of 30.
axiom periodic_g : ∀ x : ℝ, g(x) = g(x + 30)

-- Statement to prove
theorem smallest_b_for_shift : ∃ b : ℕ, 0 < b ∧ (∀ x : ℝ, g((x - b) / 3) = g(x / 3)) ∧ b = 90 :=
by
  use 90
  sorry

end smallest_b_for_shift_l728_728017


namespace problem_statement_l728_728782

noncomputable def roots_of_x3_minus_12x2_plus_14x_minus_1 (a b c : ℝ) : Prop :=
  a + b + c = 12 ∧
  a * b + b * c + c * a = 14 ∧
  a * b * c = 1

theorem problem_statement (a b c s : ℝ)
  (h1 : roots_of_x3_minus_12x2_plus_14x_minus_1 a b c)
  (h2 : s = real.sqrt a + real.sqrt b + real.sqrt c) :
  s^4 - 24 * s^2 - 8 * s = -232 :=
  sorry

end problem_statement_l728_728782


namespace smallest_possible_n_l728_728016

theorem smallest_possible_n :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2010 ∧
  (∃ (m n : ℤ), (x! * y! * z! = m * 10^n) ∧ (m % 10 ≠ 0) ∧ n = 492) :=
by
  sorry

end smallest_possible_n_l728_728016


namespace water_left_after_dumping_l728_728851

/--
Given:
1. The water pressure of a sink has a steady flow of 2 cups per 10 minutes for the first 30 minutes.
2. The water pressure still flows at 2 cups per 10 minutes for the next 30 minutes after.
3. For the next hour, the water pressure maximizes to 4 cups per 10 minutes and stops.
4. Shawn has to dump half of the water away.
Prove:
The amount of water left is 18 cups.
-/
theorem water_left_after_dumping (h1 : ∀ t : ℕ, 0 ≤ t ∧ t < 30 → 2 * (t / 10) = water_flow t)
                                   (h2 : ∀ t : ℕ, 30 ≤ t ∧ t < 60 → 2 * ((t - 30) / 10) = water_flow t)
                                   (h3 : ∀ t : ℕ, 60 ≤ t ∧ t < 120 → 4 * ((t - 60) / 10) = water_flow t)
                                   (h4 : Shawn_dumps_half_water) :
                                   total_water_after_dumping = 18 := by 
                                   sorry

end water_left_after_dumping_l728_728851


namespace rhombus_area_l728_728298

theorem rhombus_area (AC BD : ℝ) (h : AC ^ 2 - 65 * AC + 360 = 0) (k : BD ^ 2 - 65 * BD + 360 = 0) :
    let area := 1 / 2 * AC * BD in area = 180 :=
by
  -- Since AC and BD are roots of the given quadratic equation, we note their product
  -- is equal to the constant term divided by the leading coefficient.
  -- Let’s introduce the product of the roots as a lemma:
  have prod_roots : AC * BD = 360 := sorry
  
  -- Considering the definition of the area of the rhombus, we calculate:
  have area_calc : 1 / 2 * AC * BD = 180 := sorry
  
  -- Thus, we conclude that the area of the rhombus is indeed:
  exact area_calc

end rhombus_area_l728_728298


namespace proof_max_ρ_sq_l728_728780

noncomputable def max_ρ_sq (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≥ b) 
    (x y : ℝ) (h₃ : 0 ≤ x) (h₄ : x < a) (h₅ : 0 ≤ y) (h₆ : y < b)
    (h_xy : a^2 + y^2 = b^2 + x^2)
    (h_eq : a^2 + y^2 = (a - x)^2 + (b - y)^2)
    (h_x_le : x ≤ 2 * a / 3) : ℝ :=
  (a / b) ^ 2

theorem proof_max_ρ_sq (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≥ b)
    (x y : ℝ) (h₃ : 0 ≤ x) (h₄ : x < a) (h₅ : 0 ≤ y) (h₆ : y < b)
    (h_xy : a^2 + y^2 = b^2 + x^2)
    (h_eq : a^2 + y^2 = (a - x)^2 + (b - y)^2)
    (h_x_le : x ≤ 2 * a / 3) : (max_ρ_sq a b h₀ h₁ h₂ x y h₃ h₄ h₅ h₆ h_xy h_eq h_x_le) ≤ 9 / 5 := by
  sorry

end proof_max_ρ_sq_l728_728780


namespace problem_1_problem_2_problem_3_l728_728760

-- Define equal variance sequence:
def equal_variance_sequence (a : ℕ → ℝ) (p : ℝ) : Prop :=
  ∀ n : ℕ, a n ^ 2 - a (n + 1) ^ 2 = p

-- Problems to prove in Lean
theorem problem_1 (a : ℕ → ℝ) (p : ℝ) (h : equal_variance_sequence a p) :
  ∃ d : ℝ, ∀ n : ℕ, a n ^ 2 = d * n + a 0 ^ 2 :=
sorry

theorem problem_2 : equal_variance_sequence (λ n, (-1) ^ n) 0 :=
sorry

theorem problem_3 (a : ℕ → ℝ) (p : ℝ) (k : ℕ) (h : equal_variance_sequence a p) :
  equal_variance_sequence (λ n, a (k * n)) (k * p) :=
sorry

end problem_1_problem_2_problem_3_l728_728760


namespace min_value_of_expression_l728_728902

theorem min_value_of_expression (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (habc : a + b + c = 1) (expected_value : 3 * a + 2 * b = 2) :
  ∃ a b, (a + b + (1 - a - b) = 1) ∧ (3 * a + 2 * b = 2) ∧ (∀ a b, ∃ m, m = (2/a + 1/(3*b)) ∧ m = 16/3) :=
sorry

end min_value_of_expression_l728_728902


namespace spherical_to_rectangular_coords_l728_728578

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), ρ = 15 ∧ θ = 5 * Real.pi / 4 ∧ φ = Real.pi / 4 →
  let x := ρ * Real.sin φ * Real.cos θ,
      y := ρ * Real.sin φ * Real.sin θ,
      z := ρ * Real.cos φ in
  (x, y, z) = (-15 / 2, -15 / 2, 15 * Real.sqrt 2 / 2) :=
by
  intros ρ θ φ h
  obtain ⟨hρ, hθ, hφ⟩ := h
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coords_l728_728578


namespace smallest_circle_radius_l728_728092

open Real

-- Define the vertices of the squares
def vertices : List (ℝ × ℝ) :=
  [(0, 0), (0, 1), (1, 0), (1, 1),
   (0.5, 1), (0.5, 2), (1.5, 1), (1.5, 2),
   (1, 0), (1, 1), (2, 0), (2, 1)]

-- Function to compute the centroid of vertices
def centroid (vs : List (ℝ × ℝ)) : ℝ × ℝ :=
  let (sx, sy) := vs.foldl (λ (acc : ℝ × ℝ) (p : ℝ × ℝ) => (acc.1 + p.1, acc.2 + p.2)) (0, 0)
  (sx / vs.length, sy / vs.length)

-- Farthest vertex from a given point
def farthest_vertex (p : ℝ × ℝ) (vs : List (ℝ × ℝ)) : ℝ :=
  vs.foldl (λ acc v => max acc (Real.sqrt ((v.1 - p.1)^2 + (v.2 - p.2)^2))) 0

-- Main statement
theorem smallest_circle_radius : 
  let c := centroid vertices in
  farthest_vertex c vertices = 5 * Real.sqrt 17 / 16 :=
sorry

end smallest_circle_radius_l728_728092


namespace T_area_div_S_area_is_7_over_18_l728_728348

open Set

def T : Set (ℝ × ℝ × ℝ) :=
  { t | ∃ (x y z : ℝ), t = (x, y, z) ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1 }

def supports (t : ℝ × ℝ × ℝ) (a b c : ℝ) : Prop :=
  let (x, y, z) := t in
  (x ≥ a ∧ y ≥ b ∧ z < c) ∨ (x ≥ a ∧ y < b ∧ z ≥ c) ∨ (x < a ∧ y ≥ b ∧ z ≥ c)

def S : Set (ℝ × ℝ × ℝ) :=
  { t | t ∈ T ∧ supports t (1 / 2) (1 / 3) (1 / 6) }

theorem T_area_div_S_area_is_7_over_18 : measure_of S / measure_of T = 7 / 18 := sorry

end T_area_div_S_area_is_7_over_18_l728_728348


namespace add_right_side_term_l728_728464

theorem add_right_side_term (k : ℕ) (h : k ≥ 1) : 
  1 - (1 / 2) + (1 / 3) - (1 / 4) + ... + (1 / (2*k-1)) - (1 / (2*k)) = 
  (1 / (k+1)) + (1 / (k+2)) + ... + (1 / (2*k)) → 
  1 - (1 / 2) + (1 / 3) - (1 / 4) + ... + 1/ (2*(k+1)-1) - (1 / (2*(k+1))) = 
  (1 / (k+1)) + (1 / (k+2)) + ... + (1 / (2*k)) + (1 / (2*k+1)) + (1 / (2*(k+1))) - (1 / (k+1)) :=
by sorry

end add_right_side_term_l728_728464


namespace children_got_off_l728_728107

theorem children_got_off {x : ℕ} 
  (initial_children : ℕ := 22)
  (children_got_on : ℕ := 40)
  (children_left : ℕ := 2)
  (equation : initial_children + children_got_on - x = children_left) :
  x = 60 :=
sorry

end children_got_off_l728_728107


namespace trig_expression_simplify_l728_728699

theorem trig_expression_simplify (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trig_expression_simplify_l728_728699


namespace height_of_isosceles_triangle_l728_728510

variable (s : ℝ) (h : ℝ) (A : ℝ)
variable (triangle : ∀ (s : ℝ) (h : ℝ), A = 0.5 * (2 * s) * h)
variable (rectangle : ∀ (s : ℝ), A = s^2)

theorem height_of_isosceles_triangle (s : ℝ) (h : ℝ) (A : ℝ) (triangle : ∀ (s : ℝ) (h : ℝ), A = 0.5 * (2 * s) * h)
  (rectangle : ∀ (s : ℝ), A = s^2) : h = s := by
  sorry

end height_of_isosceles_triangle_l728_728510


namespace polyomino_Z_placement_l728_728686

theorem polyomino_Z_placement :
  let chessboard := fin 8 × fin 8
  let possible_placements := 
    {z_pos : set (fin 8 × fin 8) // 
      z_pos.card = 4 ∧ 
      (exists f : (fin 8 × fin 8) → (fin 8 × fin 8), 
        bijection f ∧ 
        (z_pos.image f = z_pos ∨ z_pos.image f ≠ z_pos)) ∧ 
      (∀ p : fin 8 × fin 8, z_pos.contains p)} ∧ 
    (∀ z_shape : fin 2 → fin 3, 
      ∃ pos : fin 8 × fin 8, 
      ∃ board_transform : (fin 3 × fin 3) → (fin 8 × fin 8), 
      bijection board_transform ∧ 
      ∃ transform_pos : fin 8 → fin 8, 
      bijection transform_pos ∧ 
      transform_pos (z_shape ⟨0, by linarith⟩) = board_transform pos) in
  possible_placements.card = 168 :=
by
  sorry

end polyomino_Z_placement_l728_728686


namespace probability_of_three_heads_in_eight_tosses_l728_728988

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l728_728988


namespace match_weights_l728_728106

def Item := { name : String }

noncomputable def w (item : Item) : ℕ

def banana := { name := "Banana" }
def orange := { name := "Orange" }
def watermelon := { name := "Watermelon" }
def kiwi := { name := "Kiwi" }
def apple := { name := "Apple" }

def weights : List ℕ := [210, 180, 200, 170, 1400]

axiom condition_1 : ∀ x, x ∈ [banana, orange, watermelon, kiwi, apple]

axiom condition_2 : w watermelon > max (w banana) (max (w orange) (max (w kiwi) (w apple)))

axiom condition_3 : w orange + w kiwi = w banana + w apple

axiom condition_4 : w banana < w orange ∧ w orange < w kiwi

theorem match_weights : 
  (w banana = 170 ∧ 
   w orange = 180 ∧ 
   w watermelon = 1400 ∧ 
   w kiwi = 200 ∧ 
   w apple = 210) :=
sorry

end match_weights_l728_728106


namespace probability_three_heads_in_eight_tosses_l728_728992

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l728_728992


namespace probability_heads_heads_l728_728076

theorem probability_heads_heads (h_uniform_density : ∀ outcome, outcome ∈ {("heads", "heads"), ("heads", "tails"), ("tails", "heads"), ("tails", "tails")} → True) :
  ℙ({("heads", "heads")}) = 1 / 4 :=
sorry

end probability_heads_heads_l728_728076


namespace ball_bearings_per_machine_l728_728769

-- Define the conditions
def num_machines := 10
def normal_cost_per_ball_bearing := 1.0
def sale_cost_per_ball_bearing := 0.75
def bulk_discount_percent := 0.20
def savings := 120.0

-- Define the proof problem as a Lean statement
theorem ball_bearings_per_machine (x : ℝ) : 
  (10 * normal_cost_per_ball_bearing * x) - (10 * sale_cost_per_ball_bearing * (1 - bulk_discount_percent) * x) = savings → 
  x = 30 :=
by sorry

end ball_bearings_per_machine_l728_728769


namespace partition_count_l728_728790

-- Define the set of seven primes
def P : Set ℕ := {2, 3, 5, 7, 11, 13, 17}

-- Define the set of 28 composites each being a product of two elements of P
def C : Set ℕ := { p1 * p2 | p1 p2 : ℕ, p1 ∈ P, p2 ∈ P }.to_finset.to_set

-- Conditions for partitions
def valid_subset (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ ∀ x ∈ s, ∃ y z ∈ s, y ≠ x ∧ z ≠ x ∧ gcd x y > 1 ∧ gcd x z > 1

-- Define a valid partition of C
def is_valid_partition (part : Finset (Finset ℕ)) : Prop :=
  part.card = 7 ∧ ∀ s ∈ part, valid_subset s ∧ s ⊆ C ∧ (∀ t ∈ part, s ≠ t → disjoint s t)

-- Prove that the number of valid partitions of C is 26460
theorem partition_count : (Finset.filter is_valid_partition (Finset.powerset C)).card = 26460 :=
  sorry

end partition_count_l728_728790


namespace number_of_elements_l728_728343

noncomputable def set_mean (S : Set ℝ) : ℝ := sorry

theorem number_of_elements (S : Set ℝ) (M : ℝ)
  (h1 : set_mean (S ∪ {15}) = M + 2)
  (h2 : set_mean (S ∪ {15, 1}) = M + 1) :
  ∃ k : ℕ, (M * k + 15 = (M + 2) * (k + 1)) ∧ (M * k + 16 = (M + 1) * (k + 2)) ∧ k = 4 := sorry

end number_of_elements_l728_728343


namespace water_left_l728_728853

def steady_flow (rate: ℕ) (duration: ℕ) : ℕ := (rate * (duration / 10))

theorem water_left {rate1 rate2 rate3 duration1 duration2 duration3 half : ℕ} 
  (h1 : rate1 = 2) 
  (h2 : duration1 = 30)
  (h3 : rate2 = 2) 
  (h4 : duration2 = 30)
  (h5 : rate3 = 4) 
  (h6 : duration3 = 60)
  (h7 : half = 2) :
  let total_water := steady_flow rate1 duration1 + steady_flow rate2 duration2 + steady_flow rate3 duration3 in
  (total_water / half) = 18 :=
by
  sorry

end water_left_l728_728853


namespace probability_is_one_fifth_l728_728689

def satisfies_equation (p q : ℤ) : Prop :=
  p * q - 6 * p - 3 * q = 3

def valid_p_values : Finset ℤ :=
  (Finset.range 15).map (λ x, x + 1)

def count_valid_p: ℕ :=
  valid_p_values.filter (λ p, ∃ q, satisfies_equation p q).card

theorem probability_is_one_fifth :
  (count_valid_p : ℚ) / valid_p_values.card = 1 / 5 :=
  by
    sorry

end probability_is_one_fifth_l728_728689


namespace factorial_difference_l728_728543

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem factorial_difference : fact 10 - fact 9 = 3265920 := by
  have h1 : fact 9 = 362880 := sorry
  have h2 : fact 10 = 10 * fact 9 := by rw [fact]
  calc
    fact 10 - fact 9
        = 10 * fact 9 - fact 9 := by rw [h2]
    ... = 9 * fact 9 := by ring
    ... = 9 * 362880 := by rw [h1]
    ... = 3265920 := by norm_num

end factorial_difference_l728_728543


namespace sqrt_x_div_sqrt_y_l728_728195

-- Conditions
def condition (x y : ℝ) :=
  (1/9 + 1/16) / (1/25 + 1/36) = 13 * x / (53 * y)

-- Proof statement
theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : condition x y) :
  sqrt x / sqrt y = 1092 / 338 :=
sorry

end sqrt_x_div_sqrt_y_l728_728195


namespace correct_propositions_l728_728270

noncomputable def log_a (a x : ℝ) : ℝ := Real.log x / Real.log a

noncomputable def inv_log_a (a x : ℝ) : ℝ := a^x

theorem correct_propositions (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (∀ x y : ℝ, f x = log_a a x ∧ f⁻¹ y = inv_log_a a y → 
    ((f x ≤ f y ↔ f⁻¹ x ≤ f⁻¹ y) ∧
     (a > 1 → ∀ x, f x ≠ f⁻¹ x) ∧
     (∃ x, f x = f⁻¹ x → x = inv_log_a a x) ∧
     (0 < a ∧ a < 1 → ∃ x, f x = f⁻¹ x))) :=
by
  sorry

end correct_propositions_l728_728270


namespace count_g_mod_5_zero_l728_728352

def g (x : ℝ) := x^2 + 4 * x + 3

def T := finset.range 21

theorem count_g_mod_5_zero :
  ((T.filter (λ t : ℕ, (g t) % 5 = 0)).card = 8) := sorry

end count_g_mod_5_zero_l728_728352


namespace probability_at_least_one_of_ABC_l728_728257

-- Definitions and conditions from part (a)
variable {A B C : Prop}
variable [ProbA : fact (0.2)] [ProbB : fact (0.6)] [ProbC : fact (0.14)]

-- Theorem statement using the conditions
theorem probability_at_least_one_of_ABC :
  independent A B →
  mutually_exclusive A C →
  mutually_exclusive B C →
  (P(A) = 0.2) →
  (P(B) = 0.6) →
  (P(C) = 0.14) →
  P(A ∪ B ∪ C) = 0.82 := 
by
  sorry

end probability_at_least_one_of_ABC_l728_728257


namespace digit_equation_l728_728588

-- Define the digits for the letters L, O, V, E, and S in base 10.
def digit_L := 4
def digit_O := 3
def digit_V := 7
def digit_E := 8
def digit_S := 6

-- Define the numeral representations.
def LOVE := digit_L * 1000 + digit_O * 100 + digit_V * 10 + digit_E
def EVOL := digit_E * 1000 + digit_V * 100 + digit_O * 10 + digit_L
def SOLVES := digit_S * 100000 + digit_O * 10000 + digit_L * 1000 + digit_V * 100 + digit_E * 10 + digit_S

-- Prove that LOVE + EVOL + LOVE = SOLVES in base 10.
theorem digit_equation :
  LOVE + EVOL + LOVE = SOLVES :=
by
  -- Proof is omitted; include a proper proof in your verification process.
  sorry

end digit_equation_l728_728588


namespace factorial_difference_l728_728551

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728551


namespace binom_2023_1_eq_2023_l728_728525

theorem binom_2023_1_eq_2023 :
  nat.choose 2023 1 = 2023 :=
by sorry

end binom_2023_1_eq_2023_l728_728525


namespace remaining_candy_l728_728141

def initial_candy : ℕ := 36
def ate_candy1 : ℕ := 17
def ate_candy2 : ℕ := 15
def total_ate_candy : ℕ := ate_candy1 + ate_candy2

theorem remaining_candy : initial_candy - total_ate_candy = 4 := by
  sorry

end remaining_candy_l728_728141


namespace probability_of_drawing_red_ball_l728_728315

theorem probability_of_drawing_red_ball :
  let red_balls := 7
  let black_balls := 3
  let total_balls := red_balls + black_balls
  let probability_red := (red_balls : ℚ) / total_balls
  probability_red = 7 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l728_728315


namespace binomial_expansion_thm_l728_728196

theorem binomial_expansion_thm :
  coefficient_of_x3 (1 / 2 * x + 1)^5 = 5 / 4 :=
by
  sorry

end binomial_expansion_thm_l728_728196


namespace cos_2theta_plus_pi_l728_728637

-- Given condition
def tan_theta_eq_2 (θ : ℝ) : Prop := Real.tan θ = 2

-- The mathematical statement to prove
theorem cos_2theta_plus_pi (θ : ℝ) (h : tan_theta_eq_2 θ) : Real.cos (2 * θ + Real.pi) = 3 / 5 := 
sorry

end cos_2theta_plus_pi_l728_728637


namespace stephanie_bills_l728_728385

theorem stephanie_bills :
  let electricity_bill := 120
  let electricity_paid := 0.80 * electricity_bill
  let gas_bill := 80
  let gas_paid := (3 / 4) * gas_bill
  let additional_gas_payment := 10
  let water_bill := 60
  let water_paid := 0.65 * water_bill
  let internet_bill := 50
  let internet_paid := 6 * 5
  let internet_remaining_before_discount := internet_bill - internet_paid
  let internet_discount := 0.10 * internet_remaining_before_discount
  let phone_bill := 45
  let phone_paid := 0.20 * phone_bill
  let remaining_electricity := electricity_bill - electricity_paid
  let remaining_gas := gas_bill - (gas_paid + additional_gas_payment)
  let remaining_water := water_bill - water_paid
  let remaining_internet := internet_remaining_before_discount - internet_discount
  let remaining_phone := phone_bill - phone_paid
  (remaining_electricity + remaining_gas + remaining_water + remaining_internet + remaining_phone) = 109 :=
by
  sorry

end stephanie_bills_l728_728385


namespace exponent_fraction_simplification_l728_728069

theorem exponent_fraction_simplification : 
  (2 ^ 2016 + 2 ^ 2014) / (2 ^ 2016 - 2 ^ 2014) = 5 / 3 := 
by {
  -- proof steps would go here
  sorry
}

end exponent_fraction_simplification_l728_728069


namespace max_cardinality_T_l728_728346

-- Given: T is a subset of {1, 2, ..., 2000}
--        For all x, y in T, |x - y| ≠ 6 and |x - y| ≠ 10
-- Prove: The maximum cardinality of T (|T|) is 1000

theorem max_cardinality_T : 
  ∃ T : Finset ℕ, (∀ x y ∈ T, (x ≠ y) → (abs (x - y) ≠ 6) ∧ (abs (x - y) ≠ 10)) 
  ∧ (T ⊆ (Finset.range 2000).map Finset.univ.fintype.1) 
  ∧ T.card = 1000 := 
sorry

end max_cardinality_T_l728_728346


namespace range_of_p_l728_728254

open Set

variable {α : Type _}

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (p : ℝ) : Set ℝ := {x | p+1 ≤ x ∧ x ≤ 2p-1}

theorem range_of_p (p : ℝ) : (A ∩ (B p) = (B p)) ↔ (p ≤ 3) := sorry

end range_of_p_l728_728254


namespace combination_seven_four_l728_728151

theorem combination_seven_four : nat.choose 7 4 = 35 :=
by
  sorry

end combination_seven_four_l728_728151


namespace factors_2310_l728_728680

theorem factors_2310 : ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ S.card = 5 ∧ (2310 = S.prod id) :=
by
  sorry

end factors_2310_l728_728680


namespace tangent_line_eq_area_independent_of_a_l728_728245

open Real

section TangentLineAndArea

def curve (x : ℝ) := x^2 - 1

def tangentCurvey (x : ℝ) := x^2

noncomputable def tangentLine (a : ℝ) (ha : a > 0) : (ℝ → ℝ) :=
  if a > 1 then λ x => (2*(a + 1)) * x - (a+1)^2
  else λ x => (2*(a - 1)) * x - (a-1)^2

theorem tangent_line_eq (a : ℝ) (ha : a > 0) :
  ∃ (line : ℝ → ℝ), (line = tangentLine a ha) :=
sorry

theorem area_independent_of_a (a : ℝ) (ha : a > 0) :
  (∫ x in (a - 1)..a, (tangentCurvey x - tangentLine a ha x)) +
  (∫ x in a..(a + 1), (tangentCurvey x - tangentLine a ha x)) = (2 / 3 : Real) :=
sorry

end TangentLineAndArea

end tangent_line_eq_area_independent_of_a_l728_728245


namespace remainder_when_dividing_25197631_by_17_l728_728294

theorem remainder_when_dividing_25197631_by_17 :
  25197631 % 17 = 10 :=
by
  sorry

end remainder_when_dividing_25197631_by_17_l728_728294


namespace geometric_locus_of_simson_lines_l728_728810

-- Define the elements: Orthocenter, Circumcenter, Nine-point Circle Center, Points on circumcircle, and Midpoints
variables {triangle : Type*} [EuclideanSpace ℝ triangle]
variables (M K F : triangle)
variables (P1 P2 Q1 Q2 : triangle)
variables (s1 s2 : set triangle)

-- Define the conditions
noncomputable def is_orthocenter (M : triangle) : Prop := sorry
noncomputable def is_circumcenter (K : triangle) : Prop := sorry
noncomputable def is_nine_point_circle_center (F : triangle) : Prop := sorry
noncomputable def is_diameter_endpoints (P1 P2 : triangle) : Prop := sorry
noncomputable def is_midpoint (M P Q : triangle) : Prop := sorry
noncomputable def is_simson_line (P : triangle) (s : set triangle) : Prop := sorry

-- The theorem to prove
theorem geometric_locus_of_simson_lines :
  is_orthocenter M →
  is_circumcenter K →
  is_nine_point_circle_center F →
  is_diameter_endpoints P1 P2 →
  is_midpoint M P1 Q1 →
  is_midpoint M P2 Q2 →
  is_simson_line P1 s1 →
  is_simson_line P2 s2 →
  (∀ point, point ∈ (s1 ∩ s2)) → point ∈ nine_point_circle :=
begin
  sorry
end

end geometric_locus_of_simson_lines_l728_728810


namespace find_f_x_l728_728268

theorem find_f_x (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 2 * x - 1) : 
  ∀ x : ℤ, f x = 2 * x - 3 :=
sorry

end find_f_x_l728_728268


namespace evaluate_fraction_l728_728182

theorem evaluate_fraction : 
  ( (20 - 19) + (18 - 17) + (16 - 15) + (14 - 13) + (12 - 11) + (10 - 9) + (8 - 7) + (6 - 5) + (4 - 3) + (2 - 1) ) 
  / 
  ( (1 - 2) + (3 - 4) + (5 - 6) + (7 - 8) + (9 - 10) + (11 - 12) + 13 ) 
  = (10 / 7) := 
by
  -- proof skipped
  sorry

end evaluate_fraction_l728_728182


namespace thermal_energy_l728_728892

-- Define the given conditions
def delta_F : ℝ := 30
def V : ℝ := 2
def fahrenheit_to_celsius (F : ℝ) : ℝ := (5 / 9) * (F - 32)
def heat_energy (V : ℝ) (delta_C : ℝ) : ℝ := 4200 * V * delta_C

-- Convert ΔF to ΔC
def delta_C : ℝ := fahrenheit_to_celsius delta_F

-- Compute the expected result in joules and then convert to kilojoules
def Q_joules : ℝ := heat_energy V delta_C
def Q_kilojoules : ℝ := Q_joules / 1000

-- The goal to prove
theorem thermal_energy : Q_kilojoules ≈ 140 := sorry 

end thermal_energy_l728_728892


namespace min_intersection_points_l728_728502

noncomputable def sqrt (n : ℕ) : ℝ := Real.sqrt n

-- Definitions based on conditions
def john_harvard_statue_position : ℝ × ℝ := (0, 0)

def circle_radius_set : Finset ℝ := (Finset.range 10001).filter (λ n, 2020 ≤ n ∧ n ≤ 10000).image sqrt

def johnston_gate : ℝ := 10

-- Prove the required statement
theorem min_intersection_points (circles : Finset ℝ) (line_segment_length : ℝ) :
  (circle_radius_set.card = 7981) ∧ (line_segment_length = johnston_gate) →
  ∃ m : ℕ, m = 49 :=
begin
  sorry
end

end min_intersection_points_l728_728502


namespace expression_of_24ab_in_P_and_Q_l728_728386

theorem expression_of_24ab_in_P_and_Q (a b : ℕ) (P Q : ℝ)
  (hP : P = 2^a) (hQ : Q = 5^b) : 24^(a*b) = P^(3*b) * 3^(a*b) := 
  by
  sorry

end expression_of_24ab_in_P_and_Q_l728_728386


namespace min_needed_framing_l728_728472

-- Define the original dimensions of the picture
def original_width_inch : ℕ := 5
def original_height_inch : ℕ := 7

-- Define the factor by which the dimensions are doubled
def doubling_factor : ℕ := 2

-- Define the width of the border
def border_width_inch : ℕ := 3

-- Define the function to calculate the new dimensions after doubling
def new_width_inch : ℕ := original_width_inch * doubling_factor
def new_height_inch : ℕ := original_height_inch * doubling_factor

-- Define the function to calculate dimensions including the border
def total_width_inch : ℕ := new_width_inch + 2 * border_width_inch
def total_height_inch : ℕ := new_height_inch + 2 * border_width_inch

-- Define the function to calculate the perimeter of the picture with border
def perimeter_inch : ℕ := 2 * (total_width_inch + total_height_inch)

-- Conversision from inches to feet (1 foot = 12 inches)
def inch_to_foot_conversion_factor : ℕ := 12

-- Define the function to calculate the minimum linear feet of framing needed
noncomputable def min_linear_feet_of_framing : ℕ := (perimeter_inch + inch_to_foot_conversion_factor - 1) / inch_to_foot_conversion_factor

-- The main theorem statement
theorem min_needed_framing : min_linear_feet_of_framing = 6 := by
  -- Proof construction is omitted as per the instructions
  sorry

end min_needed_framing_l728_728472


namespace length_of_AB_l728_728808

theorem length_of_AB
  (AP PB AQ QB : ℝ) 
  (h_ratioP : 5 * AP = 3 * PB)
  (h_ratioQ : 3 * AQ = 2 * QB)
  (h_PQ : AQ = AP + 3 ∧ QB = PB - 3)
  (h_PQ_length : AQ - AP = 3)
  : AP + PB = 120 :=
by {
  sorry
}

end length_of_AB_l728_728808


namespace min_nonempty_piles_eq_binary_ones_l728_728793

-- Define the problem conditions
def initial_piles (n : ℕ) : ℕ := n

def combine_piles (x y : ℕ) : ℕ :=
  if x = y then 1 else 0

-- Define the binary representation of n
def binary_ones (n : ℕ) : ℕ :=
  n.binary_digits.count 1

-- Define the statement of the problem: 
-- For any positive integer n, the smallest number of non-empty piles is the number of 1s in the binary representation of n
theorem min_nonempty_piles_eq_binary_ones (n : ℕ) (h : n > 0) : ∃ k, k = binary_ones n :=
by
  sorry

end min_nonempty_piles_eq_binary_ones_l728_728793


namespace factorial_difference_l728_728541

-- Define factorial function for natural numbers
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- The theorem we want to prove
theorem factorial_difference : factorial 10 - factorial 9 = 3265920 :=
by
  rw [factorial, factorial]
  sorry

end factorial_difference_l728_728541


namespace factorial_difference_l728_728533

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728533


namespace change_calculation_l728_728130

/-!
# Problem
Adam has $5 to buy an airplane that costs $4.28. How much change will he get after buying the airplane?

# Conditions
Adam has $5.
The airplane costs $4.28.

# Statement
Prove that the change Adam will get is $0.72.
-/

theorem change_calculation : 
  let amount := 5.00
  let cost := 4.28
  let change := 0.72
  amount - cost = change :=
by 
  sorry

end change_calculation_l728_728130


namespace lisa_goal_achievable_l728_728138

theorem lisa_goal_achievable (total_quizzes : ℕ) (goal_percentage : ℕ) (halfway_quizzes : ℕ) 
(current_As : ℕ) (remaining_quizzes : ℕ) : 
  total_quizzes = 60 ∧ goal_percentage = 90 ∧ halfway_quizzes = 40 ∧ current_As = 30 ∧ remaining_quizzes = 20 →
  (remaining_quizzes - (goal_percentage * total_quizzes / 100 - current_As)) = 0 :=
begin
  sorry
end

end lisa_goal_achievable_l728_728138


namespace binom_7_4_l728_728161

theorem binom_7_4 : Nat.choose 7 4 = 35 := 
by
  sorry

end binom_7_4_l728_728161


namespace partial_sum_b_l728_728249

def S (a : Nat → ℕ) (n : ℕ) : ℕ := (finset.sum (finset.range n) a)

def a : ℕ → ℕ
| 0 => 1
| (n + 1) => 2 * a n

def b : ℕ → ℕ
| 0 => 3
| (n + 1) => a n + b n

theorem partial_sum_b (n : ℕ) : (finset.sum (finset.range (n + 1)) b) = 2^(n + 1) + 2 * (n + 1) - 1 := sorry

end partial_sum_b_l728_728249


namespace fran_speed_l728_728339

variable (s : ℝ)

theorem fran_speed
  (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ)
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5)
  (h4 : fran_time * s = joann_speed * joann_time)
  : s = 120 / 7 := 
by
  sorry

end fran_speed_l728_728339


namespace row_3_seat_6_representation_l728_728732

-- Given Conditions
def seat_representation (r : ℕ) (s : ℕ) : (ℕ × ℕ) :=
  (r, s)

-- Proof Statement
theorem row_3_seat_6_representation :
  seat_representation 3 6 = (3, 6) :=
by
  sorry

end row_3_seat_6_representation_l728_728732


namespace coin_toss_probability_l728_728914

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l728_728914


namespace union_complement_eq_complement_intersection_eq_l728_728669

-- Define the universal set U and sets A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 5, 7}

-- Theorem 1: A ∪ (U \ B) = {2, 4, 5, 6}
theorem union_complement_eq : A ∪ (U \ B) = {2, 4, 5, 6} := by
  sorry

-- Theorem 2: U \ (A ∩ B) = {1, 2, 3, 4, 6, 7}
theorem complement_intersection_eq : U \ (A ∩ B) = {1, 2, 3, 4, 6, 7} := by
  sorry

end union_complement_eq_complement_intersection_eq_l728_728669


namespace correct_options_l728_728882

noncomputable def f1 (x : ℝ) : ℝ := (x^2 + 5) / (Real.sqrt (x^2 + 4))
noncomputable def f2 (x : ℝ) : ℝ := (2*x - 3) / (x - 1)
noncomputable def f3 (x : ℝ) : ℝ := Real.sqrt (x - 1) * Real.sqrt (x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)

theorem correct_options :
  (∀ x, f1 x ≠ 2) ∧
  (∀ a b m, (a > b ∧ b > 0 ∧ m > 0) → (b / a < (b + m) / (a + m))) ∧
  (∀ y, (∃ x, f2 x = y) ↔ (y ∈ Set.Ioo (-∞) 2 ∪ Set.Ioo 2 ∞)) ∧
  (∀ x, f3 x ≠ g x) :=
by
  sorry

end correct_options_l728_728882


namespace median_contains_A_BC_l728_728633

def Point : Type := (ℤ × ℤ)

def midpoint (P Q : Point) : Point :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def line_equation (P Q : Point) :=
  ((P.2 - Q.2) * (x - P.1) = (P.1 - Q.1) * (y - P.2))

noncomputable def median_line_eq (A B C : Point) : Prop :=
  let D := midpoint B C in
      (x + 3 * y - 5 = 0)

theorem median_contains_A_BC : median_line_eq (2, 1) (-2, 3) (0, 1) :=
by
  sorry

end median_contains_A_BC_l728_728633


namespace binom_7_4_eq_35_l728_728165

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728165


namespace solve_logarithm_eqn_l728_728384

noncomputable def log_five (x : ℝ) := log x / log 5
noncomputable def log_four (x : ℝ) := log x / log 4

theorem solve_logarithm_eqn : 
  2 * log_five 10 + log_five (1 / 4) + 2 ^ (log_four 3) = 2 + real.sqrt 3 :=
by
  sorry

end solve_logarithm_eqn_l728_728384


namespace find_line_m_eqns_find_line_n_eqns_l728_728672

-- Define the given lines l1 and l2
def line1 (x y: ℝ) : Prop := sqrt 3 * x - y + 1 = 0
def line2 (x y: ℝ) : Prop := sqrt 3 * x - y + 3 = 0

-- Define the given point
def point_m : ℝ × ℝ := (sqrt 3, 4)

-- Define lines m and n and their respective conditions
def passes_through (m : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop :=
  m p.1 p.2

def intercepted_length (m : ℝ → ℝ → Prop) (l1 l2 : ℝ → ℝ → Prop) (length: ℝ) : Prop := 
  sorry  -- Define the precise intercepted length condition here

def perpendicular (n l : ℝ → ℝ → Prop) : Prop := 
  sorry  -- Define the precise perpendicular condition here

def triangle_area (n : ℝ → ℝ → Prop) (area: ℝ) : Prop :=
  sorry  -- Define the precise area condition here

-- Statement for Question 1
theorem find_line_m_eqns : ∀ x y, 
  (passes_through (λ x y, x = sqrt 3) point_m ∧ intercepted_length (λ x y, x = sqrt 3) line1 line2 2) 
  ∨ (passes_through (λ x y, y = sqrt 3 / 3 * x + 3) point_m ∧ intercepted_length (λ x y, y = sqrt 3 / 3 * x + 3) line1 line2 2) :=
sorry

-- Statement for Question 2
theorem find_line_n_eqns : ∀ x y,
  (perpendicular (λ x y, y = - sqrt 3 / 3 * x + 2) line1 ∧ triangle_area (λ x y, y = - sqrt 3 / 3 * x + 2) 2sqrt(3)) 
  ∨ (perpendicular (λ x y, y = - sqrt 3 / 3 * x - 2) line1 ∧ triangle_area (λ x y, y = - sqrt 3 / 3 * x - 2) 2sqrt(3)) :=
sorry

end find_line_m_eqns_find_line_n_eqns_l728_728672


namespace probability_of_three_heads_in_eight_tosses_l728_728984

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l728_728984


namespace exists_subset_with_sum_gt_one_l728_728233

noncomputable def vectors := list ℝ

def condition (v : vectors) : Prop :=
  (v.map (λ x, abs x)).sum = 4

theorem exists_subset_with_sum_gt_one (v : vectors) (h : condition v) :
  ∃ u : vectors, (u ⊆ v) ∧ (u.map (λ x, abs x)).sum > 1 :=
sorry

end exists_subset_with_sum_gt_one_l728_728233


namespace probability_of_three_heads_in_eight_tosses_l728_728982

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l728_728982


namespace problem_find_f_l728_728599

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem_find_f {k : ℝ} :
  (∀ x : ℝ, x * (f (x + 1) - f x) = f x) →
  (∀ x y : ℝ, |f x - f y| ≤ |x - y|) →
  (∀ x : ℝ, 0 < x → f x = k * x) :=
by
  intro h1 h2
  apply sorry

end problem_find_f_l728_728599


namespace binomial_7_4_eq_35_l728_728153

-- Define the binomial coefficient using the binomial coefficient formula.
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem to prove.
theorem binomial_7_4_eq_35 : binomial_coefficient 7 4 = 35 :=
sorry

end binomial_7_4_eq_35_l728_728153


namespace find_BC_l728_728326

-- Define the given angles and area
def angle_B := 60
def angle_C := 75
def area_ABC := (1 / 2) * (3 + Real.sqrt 3)

-- State that the length of BC (denoted as c) is 2
theorem find_BC (a b c : ℝ) (A B C : ℝ) 
  (h1 : B = angle_B) 
  (h2 : C = angle_C) 
  (h3 : (1/2) * a * c * Real.sin B = area_ABC) :
  c = 2 :=
by
  sorry

end find_BC_l728_728326


namespace graphs_intersection_points_l728_728823

noncomputable theory

namespace Proof

variables {ℝ : Type*} [linear_ordered_field ℝ]

variables (f : ℝ → ℝ) 

def invertible (f : ℝ → ℝ) : Prop := 
  ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ ∀ y, f (g y) = y

theorem graphs_intersection_points : invertible f → (∃ x y : ℝ, f (x^2) = y ∧ f (x^6) = y) → 3 :=
by
  intro hf h
  sorry

end Proof

end graphs_intersection_points_l728_728823


namespace sqrt_720_simplified_l728_728003

theorem sqrt_720_simplified : Real.sqrt 720 = 6 * Real.sqrt 5 := sorry

end sqrt_720_simplified_l728_728003


namespace mark_pond_depth_l728_728366

def depth_of_Peter_pond := 5

def depth_of_Mark_pond := 3 * depth_of_Peter_pond + 4

theorem mark_pond_depth : depth_of_Mark_pond = 19 := by
  sorry

end mark_pond_depth_l728_728366


namespace sum_of_solutions_l728_728420

theorem sum_of_solutions :
  (∃ S : Finset ℝ, (∀ x ∈ S, x^2 - 8*x + 21 = abs (x - 5) + 4) ∧ S.sum id = 18) :=
by
  sorry

end sum_of_solutions_l728_728420


namespace minimum_shirts_needed_l728_728091

def colors := {red, blue, green}

def shirts : colors → ℕ
| red   := 3
| blue  := 3
| green := 3

def valid_set (s : multiset colors) : Prop :=
  (∃ (c : colors), s.count c ≥ 3) ∨ (s.card ≥ 3 ∧ s.nodup)

theorem minimum_shirts_needed :
  ∀ (d : multiset colors), d.card = 5 → valid_set d :=
sorry

end minimum_shirts_needed_l728_728091


namespace sum_binom_a_b_equals_binom_a_plus_b_sum_binom_k_n_minus_k_equals_binom_l728_728100

-- Definitions for binomial coefficient and sum
def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.choose n k

-- Theorem 1: ∑ (k = 0 to n) binom(a, k) * binom(b, n - k) = binom(a + b, n)
theorem sum_binom_a_b_equals_binom_a_plus_b (a b n : ℕ) :
  (∑ k in Finset.range (n + 1), binom a k * binom b (n - k)) = binom (a + b) n :=
by sorry

-- Theorem 2: ∑ (k = 0 to n) binom(k, a) * binom(n - k, b) = binom(n + 1, a + b + 1)
theorem sum_binom_k_n_minus_k_equals_binom (a b n : ℕ) :
  (∑ k in Finset.range (n + 1), binom k a * binom (n - k) b) = binom (n + 1) (a + b + 1) :=
by sorry

end sum_binom_a_b_equals_binom_a_plus_b_sum_binom_k_n_minus_k_equals_binom_l728_728100


namespace binom_7_4_eq_35_l728_728180

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728180


namespace tuple_count_l728_728685

theorem tuple_count : 
  (finset.univ.filter (λ (a : fin 5 × fin 5 × fin 5 × fin 5 × fin 5), 
  1 ≤ a.1 + 1 ∧ a.1 + 1 ≤ 5 ∧
  1 ≤ a.2.1 + 1 ∧ a.2.1 + 1 ≤ 5 ∧
  1 ≤ a.2.2.1 + 1 ∧ a.2.2.1 + 1 ≤ 5 ∧
  1 ≤ a.2.2.2.1 + 1 ∧ a.2.2.2.1 + 1 ≤ 5 ∧
  1 ≤ a.2.2.2.2 + 1 ∧ a.2.2.2.2 + 1 ≤ 5 ∧ 
  a.1 + 1 < a.2.1 + 1 ∧ a.2.1 + 1 > a.2.2.1 + 1 ∧ 
  a.2.2.1 + 1 < a.2.2.2.1 + 1 ∧ a.2.2.2.1 + 1 > a.2.2.2.2 + 1)).card = 11 := 
begin
  sorry
end

end tuple_count_l728_728685


namespace no_base_for_final_digit_one_l728_728224

theorem no_base_for_final_digit_one (b : ℕ) (h : 3 ≤ b ∧ b ≤ 10) : ¬ (842 % b = 1) :=
by
  cases h with 
  | intro hb1 hb2 => sorry

end no_base_for_final_digit_one_l728_728224


namespace sqrt_720_simplified_l728_728008

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  have h1 : 720 = 2^4 * 3^2 * 5 := by norm_num
  -- Here we use another proven fact or logic per original conditions and definition
  sorry

end sqrt_720_simplified_l728_728008


namespace last_score_is_60_l728_728741

noncomputable def isLastScore (scores : List ℕ) (n : ℕ) : Prop :=
  ∀ (k : ℕ), k < scores.length → 
    (1 + ∑ i in (scores.take k), scores.get! i) % (k + 1) = 0 → 
    ∃ (j : ℕ), j = scores.length - 1 ∧ scores.get! j = n

theorem last_score_is_60 :
  ∃ (scores : List ℕ), scores = [50, 55, 60, 85, 90, 100] ∧ isLastScore scores 60 :=
by
  sorry

end last_score_is_60_l728_728741


namespace parities_of_E2021_E2022_E2023_l728_728194

def E : ℕ → ℕ
| 0     := 1
| 1     := 1
| 2     := 2
| (n+3) := E n + E (n+1) + E (n+2)

theorem parities_of_E2021_E2022_E2023 :
  (E 2021 % 2, E 2022 % 2, E 2023 % 2) = (0, 1, 1) :=
by
  sorry

end parities_of_E2021_E2022_E2023_l728_728194


namespace pages_read_in_a_year_l728_728714

theorem pages_read_in_a_year (novels_per_month : ℕ) (pages_per_novel : ℕ) (months_per_year : ℕ)
  (h1 : novels_per_month = 4) (h2 : pages_per_novel = 200) (h3 : months_per_year = 12) :
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by
  -- Using the given conditions
  rw [h1, h2, h3]
  -- Simplifying the expression
  simp
  sorry

end pages_read_in_a_year_l728_728714


namespace time_after_2023_hours_l728_728394

theorem time_after_2023_hours (current_time : ℕ) (hours_later : ℕ) (modulus : ℕ) : 
    (current_time = 3) → 
    (hours_later = 2023) → 
    (modulus = 12) → 
    ((current_time + (hours_later % modulus)) % modulus = 2) :=
by 
    intros h1 h2 h3
    rw [h1, h2, h3]
    sorry

end time_after_2023_hours_l728_728394


namespace solve_equation_l728_728419

theorem solve_equation : ∃ x : ℝ, 2 * x + 1 = 0 ∧ x = -1 / 2 := by
  sorry

end solve_equation_l728_728419


namespace geom_series_sum_l728_728068

-- Define the conditions for the problem
def a : ℕ := 3
def r : ℕ := 2
def l : ℕ := 3072

-- Calculate the number of terms in the series
def n : ℕ := Nat.log (l / a) / Nat.log r + 1

-- Sum of the geometric series
def sum_geom_series : ℕ := a * ((r ^ n - 1) / (r - 1))

-- Problem statement: Prove that the sum of the series is 6141
theorem geom_series_sum : sum_geom_series = 6141 :=
by
  sorry

end geom_series_sum_l728_728068


namespace polygon_sides_l728_728725

theorem polygon_sides (n : ℕ) 
  (h1 : ∑ i in (finset.range n).filter (λ k, 2 ≤ k), 180 = 180 * (n - 2))
  (h2 : ∑ i in (finset.range n), 360 = 360)
  (h3 : 180 * (n - 2) = 3 * 360) :
  n = 8 := 
  sorry

end polygon_sides_l728_728725


namespace arrangement_proof_l728_728019

/-- The Happy Valley Zoo houses 5 chickens, 3 dogs, and 6 cats in a large exhibit area
    with separate but adjacent enclosures. We need to find the number of ways to place
    the 14 animals in a row of 14 enclosures, ensuring all animals of each type are together,
    and that chickens are always placed before cats, but with no restrictions regarding the
    placement of dogs. -/
def number_of_arrangements : ℕ :=
  let chickens := 5
  let dogs := 3
  let cats := 6
  let chicken_permutations := Nat.factorial chickens
  let dog_permutations := Nat.factorial dogs
  let cat_permutations := Nat.factorial cats
  let group_arrangements := 3 -- Chickens-Dogs-Cats, Dogs-Chickens-Cats, Chickens-Cats-Dogs
  group_arrangements * chicken_permutations * dog_permutations * cat_permutations

theorem arrangement_proof : number_of_arrangements = 1555200 :=
by 
  sorry

end arrangement_proof_l728_728019


namespace percentage_ill_l728_728131

theorem percentage_ill (total_visitors visitors_not_ill : ℕ) (h1 : total_visitors = 500) (h2 : visitors_not_ill = 300) : 
  ((total_visitors - visitors_not_ill) * 100) / total_visitors = 40 :=
begin
  -- Proof will go here
  sorry
end

end percentage_ill_l728_728131


namespace distance_between_foci_l728_728211

theorem distance_between_foci : 
  (let a_sq := 180 in let b_sq := 45 in 2 * Real.sqrt (a_sq - b_sq) = 6 * Real.sqrt 15) := 
begin
  sorry
end

end distance_between_foci_l728_728211


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l728_728014

-- Define the solutions to the given quadratic equations

theorem solve_eq1 (x : ℝ) : 2 * x ^ 2 - 8 = 0 ↔ x = 2 ∨ x = -2 :=
by sorry

theorem solve_eq2 (x : ℝ) : x ^ 2 + 10 * x + 9 = 0 ↔ x = -9 ∨ x = -1 :=
by sorry

theorem solve_eq3 (x : ℝ) : 5 * x ^ 2 - 4 * x - 1 = 0 ↔ x = -1 / 5 ∨ x = 1 :=
by sorry

theorem solve_eq4 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l728_728014


namespace find_a_l728_728658

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin (2 * x) + Real.cos (2 * x)

theorem find_a (a : ℝ) (h_symmetry : (f a (π / 12)) = √(a^2 + 1) ∨ (f a (π / 12)) = -√(a^2 + 1)) :
  a = √3 / 3 :=
by
  sorry

end find_a_l728_728658


namespace ferry_dock_202_trips_l728_728492

theorem ferry_dock_202_trips (trips : ℕ) (starts_at_south : Bool) (even_trips : trips = 202 ∧ 202 % 2 = 0) : 
  starts_at_south = true ∧ even_trips → docked_at_south : Bool :=
by 
  sorry

end ferry_dock_202_trips_l728_728492


namespace equal_intercepts_l728_728044

theorem equal_intercepts (a : ℝ) (h_ne_zero : a ≠ 0) :
    (let l := λ x y, a * x + y - 2 - a in
    let x_intercept := (2 + a) / a in
    let y_intercept := 2 + a in
    x_intercept = y_intercept → a = -2 ∨ a = 1) :=
sorry

end equal_intercepts_l728_728044


namespace point_on_xOz_plane_l728_728197

def point : ℝ × ℝ × ℝ := (1, 0, 4)

theorem point_on_xOz_plane : point.snd = 0 :=
by 
  -- Additional definitions and conditions might be necessary,
  -- but they should come directly from the problem statement:
  -- * Define conditions for being on the xOz plane.
  -- For the purpose of this example, we skip the proof.
  sorry

end point_on_xOz_plane_l728_728197


namespace train_B_time_to_destination_l728_728058

-- Definitions (conditions)
def speed_train_A := 60  -- Train A travels at 60 kmph
def speed_train_B := 90  -- Train B travels at 90 kmph
def time_train_A_after_meeting := 9 -- Train A takes 9 hours after meeting train B

-- Theorem statement
theorem train_B_time_to_destination 
  (speed_A : ℝ)
  (speed_B : ℝ)
  (time_A_after_meeting : ℝ)
  (time_B_to_destination : ℝ) :
  speed_A = speed_train_A ∧
  speed_B = speed_train_B ∧
  time_A_after_meeting = time_train_A_after_meeting →
  time_B_to_destination = 4.5 :=
by
  sorry

end train_B_time_to_destination_l728_728058


namespace factorial_difference_l728_728537

-- Define factorial function for natural numbers
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- The theorem we want to prove
theorem factorial_difference : factorial 10 - factorial 9 = 3265920 :=
by
  rw [factorial, factorial]
  sorry

end factorial_difference_l728_728537


namespace LCM_interval_equality_GCD_parenthesize_LCM_parenthesize_l728_728894

-- Part (a)
theorem LCM_interval_equality (n : ℕ) : 
  Nat.lcm_list (List.range' 1 (2 * n)) = 
  Nat.lcm_list (List.range' (n + 1) n) := 
by
  sorry

-- Part (b)
theorem GCD_parenthesize (a : List ℕ) :
  Nat.gcd_list a = Nat.gcd (a.head!) (Nat.gcd_list a.tail) := 
by
  sorry

-- Part (c)
theorem LCM_parenthesize (a : List ℕ) :
  Nat.lcm_list a = Nat.lcm (a.head!) (Nat.lcm_list a.tail) := 
by
  sorry

end LCM_interval_equality_GCD_parenthesize_LCM_parenthesize_l728_728894


namespace find_nat_pairs_l728_728207

theorem find_nat_pairs (n m : ℕ) : n! + 4! = m^2 ↔ (n = 1 ∧ m = 5) ∨ (n = 5 ∧ m = 12) := 
by
  sorry

end find_nat_pairs_l728_728207


namespace max_plus_cos_squared_l728_728774

open Real

theorem max_plus_cos_squared : 
  ∃ x ∈ Icc (0 : ℝ) (π / 2), 
  let M := (3 * sin x ^ 2 + 8 * sin x * cos x + 9 * cos x ^ 2) in
  M = 11 ∧ M + 100 * cos x ^ 2 = 91 := by
  sorry

end max_plus_cos_squared_l728_728774


namespace digit_a_for_divisibility_l728_728596

theorem digit_a_for_divisibility (a : ℕ) (h1 : (8 * 10^3 + 7 * 10^2 + 5 * 10 + a) % 6 = 0) : a = 4 :=
sorry

end digit_a_for_divisibility_l728_728596


namespace probability_three_heads_in_eight_tosses_l728_728937

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l728_728937


namespace height_second_tree_is_42_l728_728471

-- Definitions based on the conditions
def height_first_tree : ℝ := 28
def shadow_length_first_tree : ℝ := 30
def shadow_length_second_tree : ℝ := 45
def ratio : ℝ := height_first_tree / shadow_length_first_tree

-- Statement to prove that the height of the second tree is 42
theorem height_second_tree_is_42 :
  let height_second_tree := shadow_length_second_tree * ratio in
  height_second_tree = 42 :=
by
  sorry

end height_second_tree_is_42_l728_728471


namespace two_intersecting_planes_divide_space_into_four_parts_l728_728055

/--
Two intersecting planes divide the space into 4 parts.
-/
theorem two_intersecting_planes_divide_space_into_four_parts
  (Plane1 Plane2 : Plane) (h : ∃ (l : Line), Plane1 ∩ Plane2 = l) :
  divides_space_into Plane1 Plane2 4 :=
sorry

end two_intersecting_planes_divide_space_into_four_parts_l728_728055


namespace tree_age_when_23_feet_l728_728425

theorem tree_age_when_23_feet (initial_age initial_height growth_rate final_height : ℕ) 
(h_initial_age : initial_age = 1)
(h_initial_height : initial_height = 5) 
(h_growth_rate : growth_rate = 3) 
(h_final_height : final_height = 23) : 
initial_age + (final_height - initial_height) / growth_rate = 7 := 
by sorry

end tree_age_when_23_feet_l728_728425


namespace rectangle_volume_l728_728496

theorem rectangle_volume {a b c : ℕ} (h1 : a * b - c * a - b * c = 1) (h2 : c * a = b * c + 1) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a * b * c = 6 :=
sorry

end rectangle_volume_l728_728496


namespace part1_prob_excellent_team_part2_min_rounds_l728_728018

def excellent_team_probability (p1 p2 : ℚ) : ℚ :=
  2 * (p1 ^ 2 * p2 * (1 - p2) + p2 ^ 2 * p1 * (1 - p1)) + p1^2 * p2^2

def min_rounds_to_be_excellent_team
  (p1 p2 : ℚ) (required_times : ℕ) : ℕ :=
  let probability := excellent_team_probability p1 p2
  (required_times : ℚ) / probability

theorem part1_prob_excellent_team :
  excellent_team_probability (3/4) (2/3) = 2/3 :=
by
  sorry

theorem part2_min_rounds :
  ∀ p1 p2 : ℚ, p1 + p2 = 6/5 → 
  min_rounds_to_be_excellent_team p1 p2 9 ≥ 19 :=
by
  sorry

end part1_prob_excellent_team_part2_min_rounds_l728_728018


namespace range_of_x_l728_728611

theorem range_of_x (P x : ℝ) : 
  0 ≤ P ∧ P ≤ 4 → (x^2 + P * x > 4 * x + P - 3 ↔ x ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioo 3 ∞) := 
by 
  intro h
  sorry

end range_of_x_l728_728611


namespace circles_marked_points_possible_l728_728736

theorem circles_marked_points_possible :
  ∃ (circles : Set (Set Point)),
    (∀ c ∈ circles, ∃ (marked_points : Set Point), marked_points.card = 4 ∧ ∀ p ∈ marked_points, ∃ cs ∈ circles, p ∈ cs ∧ cs.card = 4) :=
sorry

end circles_marked_points_possible_l728_728736


namespace probability_is_one_fifth_l728_728688

def satisfies_equation (p q : ℤ) : Prop :=
  p * q - 6 * p - 3 * q = 3

def valid_p_values : Finset ℤ :=
  (Finset.range 15).map (λ x, x + 1)

def count_valid_p: ℕ :=
  valid_p_values.filter (λ p, ∃ q, satisfies_equation p q).card

theorem probability_is_one_fifth :
  (count_valid_p : ℚ) / valid_p_values.card = 1 / 5 :=
  by
    sorry

end probability_is_one_fifth_l728_728688


namespace binom_7_4_eq_35_l728_728177

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728177


namespace number_of_possible_values_l728_728297

section
  variable (f : ℕ → (ℝ → ℝ))
  variable (a : ℕ)

  -- Define the function
  def f (a : ℕ) (x : ℝ) : ℝ := Real.log x + a / (x + 1)

  -- Define the derivative of the function
  noncomputable def f_prime (a : ℕ) (x : ℝ) : ℝ := 1 / x - a / ((x + 1) ^ 2)

  -- Define the condition that ensures there is only one extreme value point in the interval (1,3)
  def has_one_extreme_value_point_in_interval (a : ℕ) : Prop :=
    f_prime a 1 * f_prime a 3 < 0

  theorem number_of_possible_values (h : has_one_extreme_value_point_in_interval 5) : 
    (∃! a : ℕ, has_one_extreme_value_point_in_interval a) :=
  sorry
end

end number_of_possible_values_l728_728297


namespace pirate_probability_l728_728493

theorem pirate_probability :
  (∃ n : ℕ, n = 6) →
  (∃ k : ℕ, k = 3) →
  (∃ p_treasure : ℚ, p_treasure = 1 / 4) →
  (∃ p_neither : ℚ, p_neither = 2 / 3) →
  (∃ p_traps : ℚ, p_traps = 1 / 12) →
  ∑ P in finset.powerset_len 3 (finset.range 6), 
  (1 / 4)^3 * (2 / 3)^3 = 5 / 54 :=
begin
  sorry
end

end pirate_probability_l728_728493


namespace probability_of_exactly_three_heads_l728_728927

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l728_728927


namespace distinct_prime_factors_2310_l728_728677

theorem distinct_prime_factors_2310 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ (S.card = 5) ∧ (S.prod id = 2310) := by
  sorry

end distinct_prime_factors_2310_l728_728677


namespace trig_expression_simplify_l728_728698

theorem trig_expression_simplify (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trig_expression_simplify_l728_728698


namespace part1_part2_part3_l728_728244

-- Definition of the linear function f(x) with given conditions
def f (x : ℝ) := -3 * x + 5

-- Q1: Prove f(10) = -25
theorem part1 (h1 : f 1 = 2) (h2 : f 2 = -1) : f 10 = -25 :=
sorry

-- Q2: Prove that f(x) is a decreasing function
theorem part2 (h1 : f 1 = 2) (h2 : f 2 = -1) : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
sorry 

-- Definition of g(x)
def g (x : ℝ) := (1 / x) - 3 * x

-- Q3: Prove that g(x) is an odd function
theorem part3 : ∀ x : ℝ, x ≠ 0 → g (-x) = -g x :=
sorry 

end part1_part2_part3_l728_728244


namespace find_k_l728_728279

noncomputable def k_value : ℝ := -24

-- Definitions of the vectors
def vector_OA (k : ℝ) : ℝ × ℝ := (k, 12)
def vector_OB : ℝ × ℝ := (4, 5)
def vector_OC (k : ℝ) : ℝ × ℝ := (-k, 0)

-- Define vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Define the points are collinear condition
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, v1 = (λ * v2.1, λ * v2.2)

-- Statement to prove k = -24, given the conditions
theorem find_k (k : ℝ) 
  (h1 : vector_sub (vector_OB) (vector_OA k) = vector_sub (vector_OC k) (vector_OA k) )
  : k = k_value :=
by 
  sorry

end find_k_l728_728279


namespace exists_function_l728_728223

theorem exists_function {n : ℕ} (hn : n ≥ 3) (S : Finset ℤ) (hS : S.card = n) :
  ∃ f : Fin (n) → S, 
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ i j k : Fin n, i < j ∧ j < k → 2 * (f j : ℤ) ≠ (f i : ℤ) + (f k : ℤ)) :=
by
  sorry

end exists_function_l728_728223


namespace jim_bought_3_pictures_l728_728125
noncomputable theory

def total_pictures : ℕ := 10
def probability_not_bought (x : ℕ) : ℚ :=
  ((total_pictures - x) * (total_pictures - x - 1) : ℚ) / (total_pictures * (total_pictures - 1) : ℚ)

theorem jim_bought_3_pictures (x : ℕ) (h : probability_not_bought x = 0.4666666666666667) :
  x = 3 :=
sorry

end jim_bought_3_pictures_l728_728125


namespace binom_7_4_eq_35_l728_728169

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728169


namespace probability_of_valid_p_probability_of_valid_p_fraction_l728_728690

def satisfies_equation (p q : ℤ) : Prop := p * q - 6 * p - 3 * q = 3

def valid_p (p : ℤ) : Prop := ∃ q : ℤ, satisfies_equation p q

theorem probability_of_valid_p :
  (finset.filter valid_p (finset.Icc 1 15)).card = 4 :=
sorry

theorem probability_of_valid_p_fraction :
  (finset.filter valid_p (finset.Icc 1 15)).card / 15 = 4 / 15 :=
begin
  have h : (finset.filter valid_p (finset.Icc 1 15)).card = 4 := probability_of_valid_p,
  rw h,
  norm_num,
end

end probability_of_valid_p_probability_of_valid_p_fraction_l728_728690


namespace max_value_of_n_l728_728734

noncomputable def max_n : ℕ := 7

theorem max_value_of_n
  (participants : Fin 8 → Fin n → Bool)
  (pairwise_conditions : ∀ (i j : Fin n) (p : Fin 8),
    True ∨ False ∨ True ∨ False ∨ participants p i = tt ∨ participants p j = tt) :
  ∃ n : ℕ, n ≤ max_n ∧ (∀ m : ℕ, m > n → ¬(pairwise_conditions)) :=
begin
  sorry
end

end max_value_of_n_l728_728734


namespace michael_truck_meet_1_time_l728_728797

theorem michael_truck_meet_1_time
  (michael_speed : ℕ) (pail_distance : ℕ) (truck_speed : ℕ) (truck_stop_time : ℕ)
  (initial_distance : ℕ)
  (H1 : michael_speed = 4)
  (H2 : pail_distance = 300)
  (H3 : truck_speed = 6)
  (H4 : truck_stop_time = 20)
  (H5 : initial_distance = 300) :
  michael_truck_meet_count michael_speed pail_distance truck_speed truck_stop_time initial_distance = 1 :=
sorry

end michael_truck_meet_1_time_l728_728797


namespace distance_between_points_l728_728870

theorem distance_between_points :
  let p1 : ℝ × ℝ × ℝ := (4, -3, 1)
  let p2 : ℝ × ℝ × ℝ := (8, 6, -2)
  let distance (a b : ℝ × ℝ × ℝ) : ℝ :=
    Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2 + (b.3 - a.3)^2)
  distance p1 p2 = Real.sqrt 106 :=
by
  sorry

end distance_between_points_l728_728870


namespace path_length_of_point_B_l728_728512

  theorem path_length_of_point_B (BD : ℝ) (hBD : BD = 4 / real.pi) : 
    let radius := BD in
    let circumference := 2 * real.pi * radius in
    let semicircle_length := circumference / 2 in
    semicircle_length = 4 :=
  by
    have h1 : radius = 4 / real.pi := hBD,
    have h2 : circumference = 2 * real.pi * radius := rfl,
    have h3 : semicircle_length = circumference / 2 := rfl,
    have h4 : semicircle_length = 4,
    from calc
      semicircle_length = (2 * real.pi * (4 / real.pi)) / 2 : by congr; exact h1
      ... = 8 / 2 : by simp
      ... = 4 : by norm_num
    exact h4
  
end path_length_of_point_B_l728_728512


namespace matrix_det_zero_l728_728816

variables {α β γ : ℝ}

theorem matrix_det_zero (h : α + β + γ = π) :
  Matrix.det ![
    ![Real.cos β, Real.cos α, -1],
    ![Real.cos γ, -1, Real.cos α],
    ![-1, Real.cos γ, Real.cos β]
  ] = 0 :=
sorry

end matrix_det_zero_l728_728816


namespace probability_of_three_heads_in_eight_tosses_l728_728974

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l728_728974


namespace line_parabola_tangent_slopes_l728_728274

theorem line_parabola_tangent_slopes :
  let parabola := λ (x y : ℝ), y ^ 2 = 8 * x
  let point : ℝ × ℝ := (-3, 1)
  let line := λ (k x y : ℝ), y - 1 = k * (x + 3)
  ∃ k : ℝ, ∀ x y, line k x y → parabola x y ↔ k = 0 ∨ k = -1 ∨ k = 2 / 3 :=
by
  have parabola : ∀ (x y : ℝ), y ^ 2 = 8 * x, from λ x y h, h
  have point : ℝ × ℝ := (-3, 1), from (-3, 1)
  have line : ∀ (k x y : ℝ), y - 1 = k * (x + 3), from λ k x y h, h
  sorry

end line_parabola_tangent_slopes_l728_728274


namespace perfect_arithmetic_sequence_prob_eq_8_315_l728_728619

theorem perfect_arithmetic_sequence_prob_eq_8_315 :
  let prob := 24 * (∑ a from 1 to ∞, ∑ d from 1 to ∞, 2 ^ (-4 * a - 6 * d)) / 945
  in prob = 8 / 315 := sorry

end perfect_arithmetic_sequence_prob_eq_8_315_l728_728619


namespace factorial_subtraction_l728_728556

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_subtraction : factorial 10 - factorial 9 = 3265920 := by
  sorry

end factorial_subtraction_l728_728556


namespace propositions_false_l728_728347

structure Plane :=
(is_plane : Prop)

structure Line :=
(in_plane : Plane → Prop)

def is_parallel (p1 p2 : Plane) : Prop := sorry
def is_perpendicular (p1 p2 : Plane) : Prop := sorry
def line_parallel (l1 l2 : Line) : Prop := sorry
def line_perpendicular (l1 l2 : Line) : Prop := sorry

variable (α β : Plane)
variable (l m : Line)

axiom α_neq_β : α ≠ β
axiom l_in_α : l.in_plane α
axiom m_in_β : m.in_plane β

theorem propositions_false :
  ¬(is_parallel α β → line_parallel l m) ∧ 
  ¬(line_perpendicular l m → is_perpendicular α β) := 
sorry

end propositions_false_l728_728347


namespace trapezoid_circumcenter_distance_fixed_value_l728_728626

noncomputable def trapezoid_circumcenter_distance (A B C D E O₁ O₂ : ℝ) : Prop :=
  ∀ (AD_parallel_BC : (D-A) * (B-C) = (D-C)* (A-B))
     (E_on_AB : E ∈ segment A B)
     (circumcenter_O₁ : O₁ = circumcenter_triangle A E D)
     (circumcenter_O₂ : O₂ = circumcenter_triangle B E C), 
  (distance O₁ O₂ = DC / (2 * sin B))

theorem trapezoid_circumcenter_distance_fixed_value (A B C D E O₁ O₂: ℝ) 
  (AD_parallel_BC : (D-A) * (B-C) = (D-C)* (A-B))
  (E_on_AB : E ∈ segment A B)
  (circumcenter_O₁ : O₁ = circumcenter_triangle A E D)
  (circumcenter_O₂ : O₂ = circumcenter_triangle B E C):
  distance O₁ O₂ = DC / (2 * sin B) :=
  by sorry

end trapezoid_circumcenter_distance_fixed_value_l728_728626


namespace correct_statements_l728_728883

theorem correct_statements (a b m : ℝ) (f g : ℝ → ℝ) (h1 : a > b ∧ b > 0 ∧ m > 0)
                           (h2 : ∀ x, f x = (2 * x - 3) / (x - 1))
                           (h3 : ∀ x, g x = sqrt (x - 1) * sqrt (x + 1))
                           (h4 : ∀ x, g x = sqrt (x^2 - 1)) :
    (∀ x, (f x ≠ 2 ∧ (f x ∈ (-∞, 2) ∨ f x ∈ (2, ∞)))) ∧
    (∀ x, (x < 0 → false) ∨ (x^2 - 1 < 0 → false)) :=
by
    sorry

end correct_statements_l728_728883


namespace select_best_athlete_l728_728848

theorem select_best_athlete
  (avg_A avg_B avg_C avg_D: ℝ)
  (var_A var_B var_C var_D: ℝ)
  (h_avg_A: avg_A = 185)
  (h_avg_B: avg_B = 180)
  (h_avg_C: avg_C = 185)
  (h_avg_D: avg_D = 180)
  (h_var_A: var_A = 3.6)
  (h_var_B: var_B = 3.6)
  (h_var_C: var_C = 7.4)
  (h_var_D: var_D = 8.1) :
  (avg_A > avg_B ∧ avg_A > avg_D ∧ var_A < var_C) →
  (avg_A = 185 ∧ var_A = 3.6) :=
by
  sorry

end select_best_athlete_l728_728848


namespace walking_speed_in_km_per_hr_l728_728488

def distance : ℝ := 2250 -- distance in meters
def time : ℝ := 15 -- time in minutes
def conversion_factor : ℝ := 1000 / 60 -- factor to convert meters per minute to kilometers per hour

theorem walking_speed_in_km_per_hr :
  (distance / time) * conversion_factor = 2.5 :=
by
  sorry

end walking_speed_in_km_per_hr_l728_728488


namespace coin_toss_probability_l728_728916

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l728_728916


namespace power_function_odd_l728_728650

-- Define the conditions
def f : ℝ → ℝ := sorry
def condition1 (f : ℝ → ℝ) : Prop := f 1 = 3

-- Define the statement of the problem as a Lean theorem
theorem power_function_odd (f : ℝ → ℝ) (h : condition1 f) : ∀ x, f (-x) = -f x := sorry

end power_function_odd_l728_728650


namespace probability_of_three_heads_in_eight_tosses_l728_728985

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l728_728985


namespace largest_x_value_l728_728066

theorem largest_x_value : ∃ x : ℝ, (x / 7 + 3 / (7 * x) = 1) ∧ (∀ y : ℝ, (y / 7 + 3 / (7 * y) = 1) → y ≤ (7 + Real.sqrt 37) / 2) :=
by
  -- (Proof of the theorem is omitted for this task)
  sorry

end largest_x_value_l728_728066


namespace binomial_7_4_eq_35_l728_728157

-- Define the binomial coefficient using the binomial coefficient formula.
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem to prove.
theorem binomial_7_4_eq_35 : binomial_coefficient 7 4 = 35 :=
sorry

end binomial_7_4_eq_35_l728_728157


namespace dart_board_probability_l728_728483

-- Define the areas of the hexagons
def area_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

-- The main theorem stating the problem
theorem dart_board_probability (s : ℝ) : 
  (area_hexagon s / area_hexagon (2 * s)) = 1 / 4 := 
by
  sorry

end dart_board_probability_l728_728483


namespace joan_books_l728_728768

theorem joan_books (Tom_books : ℕ) (Total_books : ℕ) (h_Tom : Tom_books = 38) (h_Total : Total_books = 48) : Total_books - Tom_books = 10 :=
by
  rw [h_Tom, h_Total]
  simp
  sorry

end joan_books_l728_728768


namespace binom_7_4_l728_728163

theorem binom_7_4 : Nat.choose 7 4 = 35 := 
by
  sorry

end binom_7_4_l728_728163


namespace sum_of_common_roots_is_nine_l728_728766

noncomputable def sumCommonRoots (a : ℝ) : ℝ :=
  if a = 2 then 0 else 9

theorem sum_of_common_roots_is_nine :
  ∀ (a : ℝ) (x : ℝ), (x^2 + (2*a - 5)*x + a^2 + 1 = 0) ∧ (x^3 + (2*a - 5)*x^2 + (a^2 + 1)*x + a^2 - 4 = 0) →
  (a = 2 ∨ a = -2) →
  (a = 2 → ¬(∃ x : ℝ, (x^2 + (2*2 - 5)*x + 2^2 + 1 = 0) ∧ (x^3 + (2*2 - 5)*x^2 + (2^2 + 1)*x + 2^2 - 4 = 0))) → 
  sumCommonRoots a = 9 := 
by
  intros a x cond ha h2
  cases ha with ha2 han2
  · rw ha2
    simp [sumCommonRoots]
    sorry
  · rw han2
    simp [sumCommonRoots]
    sorry

end sum_of_common_roots_is_nine_l728_728766


namespace prob_three_heads_in_eight_tosses_l728_728959

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l728_728959


namespace express_in_scientific_notation_l728_728896

theorem express_in_scientific_notation :
  ∀ (n : ℕ), n = 1300000 → scientific_notation n = "1.3 × 10^6" :=
by
  intros n h
  have h1 : n = 1300000 := by exact h
  sorry

end express_in_scientific_notation_l728_728896


namespace greatest_power_of_2_divides_l728_728871

-- Define the conditions as Lean definitions.
def a : ℕ := 15
def b : ℕ := 3
def n : ℕ := 600

-- Define the theorem statement based on the conditions and correct answer.
theorem greatest_power_of_2_divides (x : ℕ) (y : ℕ) (k : ℕ) (h₁ : x = a) (h₂ : y = b) (h₃ : k = n) :
  ∃ m : ℕ, (x^k - y^k) % (2^1200) = 0 ∧ ¬ ∃ m' : ℕ, m' > m ∧ (x^k - y^k) % (2^m') = 0 := sorry

end greatest_power_of_2_divides_l728_728871


namespace prob_three_heads_in_eight_tosses_l728_728952

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l728_728952


namespace composite_29n_plus_11_l728_728121

-- Definitions required by the conditions
def is_square (x : ℕ) : Prop := ∃ a : ℕ, a * a = x

theorem composite_29n_plus_11 (n a b : ℕ) 
  (h1 : 3 * n + 1 = a * a) 
  (h2 : 10 * n + 1 = b * b) : 
  ¬ nat.prime (29 * n + 11) :=
by
  sorry

end composite_29n_plus_11_l728_728121


namespace tangent_slope_angle_at_one_l728_728042

-- Define the given function.
def curve (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 5

-- Define the proposition to find the slope angle of the tangent line at x = 1.
theorem tangent_slope_angle_at_one : 
  let dydx := deriv curve 1 in
  let θ := real.arctan dydx in 
  θ = 3 * real.pi / 4 :=
by
  sorry

end tangent_slope_angle_at_one_l728_728042


namespace empty_vessel_mass_l728_728024

theorem empty_vessel_mass
  (m1 : ℝ) (m2 : ℝ) (rho_K : ℝ) (rho_B : ℝ) (V : ℝ) (m_c : ℝ)
  (h1 : m1 = m_c + rho_K * V)
  (h2 : m2 = m_c + rho_B * V)
  (h_mass_kerosene : m1 = 31)
  (h_mass_water : m2 = 33)
  (h_rho_K : rho_K = 800)
  (h_rho_B : rho_B = 1000) :
  m_c = 23 :=
by
  -- Proof skipped
  sorry

end empty_vessel_mass_l728_728024


namespace chess_club_boys_count_l728_728907

theorem chess_club_boys_count (B G : ℕ) 
  (h1 : B + G = 30)
  (h2 : (2/3 : ℝ) * G + B = 18) : 
  B = 6 :=
by
  sorry

end chess_club_boys_count_l728_728907


namespace sum_k_Pk_eq_fact_l728_728794

noncomputable theory

open Finset

variable {n : ℕ} (P : ℕ → ℕ) (Sn : Finset (Equiv.Perm (Fin n)))

# Check the main theorem to be proved

theorem sum_k_Pk_eq_fact :
  (∑ k in range (n + 1), k * P k) = n! :=
  sorry

end sum_k_Pk_eq_fact_l728_728794


namespace probability_of_three_heads_in_eight_tosses_l728_728975

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l728_728975


namespace value_of_x_l728_728694

theorem value_of_x (x y : ℝ) :
  x / (x + 1) = (y^2 + 3*y + 1) / (y^2 + 3*y + 2) → x = y^2 + 3*y + 1 :=
by
  intro h
  sorry

end value_of_x_l728_728694


namespace housewife_spent_fraction_l728_728485

theorem housewife_spent_fraction
  (initial_amount : ℝ)
  (amount_left : ℝ)
  (initial_amount_eq : initial_amount = 150)
  (amount_left_eq : amount_left = 50) :
  (initial_amount - amount_left) / initial_amount = 2/3 :=
by 
  sorry

end housewife_spent_fraction_l728_728485


namespace trigonometric_identity_solution_l728_728888

theorem trigonometric_identity_solution (x : ℝ) (k : ℤ) :
  (cos (3 * x - real.pi / 6) - sin (3 * x - real.pi / 6) * tan (real.pi / 6) = 
   1 / (2 * cos (7 * real.pi / 6))) ↔ 
  (∃ k : ℤ, x = 2 * real.pi / 9 + 2 * k * real.pi / 3 ∨ x = -2 * real.pi / 9 + 2 * k * real.pi / 3) :=
by
  sorry

end trigonometric_identity_solution_l728_728888


namespace range_of_b_minus_a_l728_728243

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem range_of_b_minus_a (a b : ℝ) :
  (∀ x, x ∈ set.Icc a b → f x ∈ set.Icc (-1) 3) →
  (set.range (λ x, f x) = set.Icc (-1 : ℝ) 3) →
  2 ≤ b - a ∧ b - a ≤ 4 :=
by
  intros h1 h2
  sorry

end range_of_b_minus_a_l728_728243


namespace students_voted_both_issues_l728_728513

-- Define the total number of students.
def total_students : ℕ := 150

-- Define the number of students who voted in favor of the first issue.
def voted_first_issue : ℕ := 110

-- Define the number of students who voted in favor of the second issue.
def voted_second_issue : ℕ := 95

-- Define the number of students who voted against both issues.
def voted_against_both : ℕ := 15

-- Theorem: Number of students who voted in favor of both issues is 70.
theorem students_voted_both_issues : 
  ((voted_first_issue + voted_second_issue) - (total_students - voted_against_both)) = 70 :=
by
  sorry

end students_voted_both_issues_l728_728513


namespace machine_shirts_per_minute_l728_728134

def shirts_made_yesterday : ℕ := 13
def shirts_made_today : ℕ := 3
def minutes_worked : ℕ := 2
def total_shirts_made : ℕ := shirts_made_yesterday + shirts_made_today
def shirts_per_minute : ℕ := total_shirts_made / minutes_worked

theorem machine_shirts_per_minute :
  shirts_per_minute = 8 := by
  sorry

end machine_shirts_per_minute_l728_728134


namespace fran_speed_l728_728337

variable (s : ℝ)

theorem fran_speed
  (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ)
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5)
  (h4 : fran_time * s = joann_speed * joann_time)
  : s = 120 / 7 := 
by
  sorry

end fran_speed_l728_728337


namespace tan_angle_difference_l728_728237

theorem tan_angle_difference (θ : ℝ) (h : tan θ = 1/2) : tan (π/4 - 2*θ) = -1/7 :=
by
  sorry

end tan_angle_difference_l728_728237


namespace binary_to_decimal_11011_l728_728189

theorem binary_to_decimal_11011 : 
  let bin := [1, 1, 0, 1, 1] in 
  let binary_to_decimal := fun l => List.foldr (fun (b bitValue: Nat) acc => acc + b * 2^bitValue) 0 (List.enumFromZero l.reverse) in
  binary_to_decimal bin = 27 :=
by
  let bin := [1, 1, 0, 1, 1]
  let binary_to_decimal := fun l => List.foldr (fun (b bitValue: Nat) acc => acc + b * 2^bitValue) 0 (List.enumFromZero l.reverse)
  have := binary_to_decimal bin = 27
  exact this

end binary_to_decimal_11011_l728_728189


namespace area_of_triangle_ABC_is_correct_l728_728305

noncomputable def triangle_area
  (AB : ℝ) (AC : ℝ) (B : ℝ) : ℝ :=
  if AB = 5 ∧ AC = 7 ∧ B = 120 then
    have a : ℝ := 3, -- derived from solution process
    1/2 * a * 5 * Real.sin (120 * Real.pi / 180)
  else 0

theorem area_of_triangle_ABC_is_correct : triangle_area 5 7 120 = 15 * Real.sqrt 3 / 4 := by
  sorry

end area_of_triangle_ABC_is_correct_l728_728305


namespace modulus_of_complex_number_is_sqrt10_l728_728026

-- Define the complex number from the problem and its properties

def complex_number : ℂ := 1 + 5 / (2 - I)

-- Define the modulus function for the complex number
def modulus (z : ℂ) : ℝ := complex.abs z

-- Define what we need to prove
theorem modulus_of_complex_number_is_sqrt10 : modulus complex_number = Real.sqrt 10 := 
  sorry

end modulus_of_complex_number_is_sqrt10_l728_728026


namespace time_after_2023_hours_l728_728393

theorem time_after_2023_hours (current_time : ℕ) (hours_later : ℕ) (modulus : ℕ) : 
    (current_time = 3) → 
    (hours_later = 2023) → 
    (modulus = 12) → 
    ((current_time + (hours_later % modulus)) % modulus = 2) :=
by 
    intros h1 h2 h3
    rw [h1, h2, h3]
    sorry

end time_after_2023_hours_l728_728393


namespace students_scoring_above_120_l728_728479

noncomputable theory

def class_size : ℕ := 50
def mean_score : ℝ := 110
def stddev_score : ℝ := 10
def lower_bound := 100
def prob_range := 0.36
def upper_bound := 120

def normal_distribution (mean : ℝ) (stddev : ℝ) (x : ℝ) : ℝ := sorry -- Definition for normal distribution PDF

theorem students_scoring_above_120 :
  P (λ x, x > upper_bound) = 0.14 →
  (class_size : ℝ) * 0.14 = 7 :=
sorry

end students_scoring_above_120_l728_728479


namespace trisha_total_distance_walked_l728_728805

def d1 : ℝ := 0.1111111111111111
def d2 : ℝ := 0.1111111111111111
def d3 : ℝ := 0.6666666666666666

theorem trisha_total_distance_walked :
  d1 + d2 + d3 = 0.8888888888888888 := 
sorry

end trisha_total_distance_walked_l728_728805


namespace solution_l728_728787

noncomputable def problem_statement (x y : ℝ) (hx : x > 1) (hy : y > 1) (h : (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5)) : ℝ :=
  (x^2 * y^2)

theorem solution : ∀ x y : ℝ, x > 1 → y > 1 → (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5) →
  (x^2 * y^2) = 225^(Real.sqrt 2) :=
by
  intros x y hx hy h
  sorry

end solution_l728_728787


namespace evaluate_expr_l728_728587

theorem evaluate_expr : (3^2 - 2^3 + 7^1 - 1 + 4^2)⁻¹ * (5 / 6) = 5 / 138 := 
by
  sorry

end evaluate_expr_l728_728587


namespace solution_set_inequality_l728_728784

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (h1 : ∀ x, HasDerivAt f (f' x) x)
variable (h2 : ∀ x, f(x) + f'(x) < 1)
variable (h3 : f(0) = 2016)

theorem solution_set_inequality :
  {x : ℝ | (exp x) * f x - (exp x) > 2015} = set.Iio 0 := 
sorry

end solution_set_inequality_l728_728784


namespace probability_of_three_heads_in_eight_tosses_l728_728991

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l728_728991


namespace log_equality_solution_l728_728295

theorem log_equality_solution (x : ℝ) (h : log 3 (x ^ 3) + log 9 x = 6) : x = 3 ^ (12 / 7) := 
by
  sorry

end log_equality_solution_l728_728295


namespace factorize_expression_l728_728592

variable {a b : ℝ} -- define a and b as real numbers

theorem factorize_expression : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l728_728592


namespace line_relationship_l728_728040

/-- P is a point inside a plane π -/
variable {P : Point} (π : Plane)
axiom P_in_plane_π : P ∈ π

/-- Q is a point outside plane π -/
variable {Q : Point}
axiom Q_not_in_plane_π : ¬ (Q ∈ π)

/-- l is a line within plane π -/
variable {l : Line}
axiom l_in_plane_π : l ∈ π

/-- The relationship between line PQ and line l is either skew or intersecting -/
theorem line_relationship (PQ : Line) :
  (PQ = join P Q) → (PQ ∩ l = ∅ ∨ (∃ R, R ∈ PQ ∧ R ∈ l)) :=
by
  intros
  sorry

end line_relationship_l728_728040


namespace height_of_parallelogram_l728_728214

variable (Area Base Height : ℝ)
variable (p : Area = 200) (q : Base = 10)

theorem height_of_parallelogram : Height = 20 :=
by
  have h1: Height = Area / Base := sorry
  rw [p, q] at h1
  exact h1

end height_of_parallelogram_l728_728214


namespace probability_heads_heads_l728_728075

theorem probability_heads_heads (h_uniform_density : ∀ outcome, outcome ∈ {("heads", "heads"), ("heads", "tails"), ("tails", "heads"), ("tails", "tails")} → True) :
  ℙ({("heads", "heads")}) = 1 / 4 :=
sorry

end probability_heads_heads_l728_728075


namespace factorial_difference_l728_728532

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728532


namespace sum_of_series_l728_728800

theorem sum_of_series (S : ℝ) (h : S = 2^100) : 
  (∑ i in finset.range (200 - 100 + 1), 2^(i + 100)) = 2 * S ^ 2 - S :=
by
  sorry

end sum_of_series_l728_728800


namespace count_valid_numbers_l728_728495

def tens_digit (n : ℕ) : ℕ := n / 10
def ones_digit (n : ℕ) : ℕ := n % 10

def is_valid (n : ℕ) : Prop :=
  n % 12 = 0 ∧ 10 ≤ n ∧ n < 100 ∧ tens_digit n > ones_digit n

theorem count_valid_numbers : (finset.filter is_valid (finset.range 100)).card = 4 := by
  sorry

end count_valid_numbers_l728_728495


namespace probability_exactly_three_heads_l728_728965
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l728_728965


namespace segment_PB_measure_l728_728308

theorem segment_PB_measure (k : ℝ) (M : Point) (A B C P : Point) (h_midpoint_arc : MidpointArc M C A B) 
  (h_perpendicular : Perpendicular (Line M P) (Line A B))
  (h_AC : dist A C = 3 * k) 
  (h_AP : dist A P = 2 * k + 1) 
  (h_BC : dist B C = 5 * k - 1) 
  : dist P B = 2 * k + 1 := 
sorry

end segment_PB_measure_l728_728308


namespace time_for_A_and_D_together_l728_728089

-- Definitions of work rates
def WorkRate_A : ℝ := 1 / 15
def WorkRate_D : ℝ := 1 / 29.999999999999993
def combined_WorkRate : ℝ := WorkRate_A + WorkRate_D

-- Additional definitions or parameters can be added if required.

theorem time_for_A_and_D_together :
  (1 / combined_WorkRate) = 10 :=
by
  -- Proof of the theorem
  sorry

end time_for_A_and_D_together_l728_728089


namespace split_numbers_cubic_l728_728222

theorem split_numbers_cubic (m : ℕ) (hm : 1 < m) (assumption : m^2 - m + 1 = 73) : m = 9 :=
sorry

end split_numbers_cubic_l728_728222


namespace probability_three_heads_in_eight_tosses_l728_728940

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l728_728940


namespace find_x_l728_728877

theorem find_x (a b k : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : k ≠ 0) :
  let x := (-(a^2 - b) + Real.sqrt ((a^2 - b)^2 - 4 * b)) / 2 in
  (a^2 - b + x) / (b + x * k) = ((a + x) * k - b) / (b + x * k) :=
by
  sorry

end find_x_l728_728877


namespace goals_scored_by_each_l728_728029

theorem goals_scored_by_each (total_goals : ℕ) (percentage : ℕ) (two_players_goals : ℕ) (each_player_goals : ℕ)
  (H1 : total_goals = 300)
  (H2 : percentage = 20)
  (H3 : two_players_goals = (percentage * total_goals) / 100)
  (H4 : two_players_goals / 2 = each_player_goals) :
  each_player_goals = 30 := by
  sorry

end goals_scored_by_each_l728_728029


namespace passes_through_quadrants_l728_728085

theorem passes_through_quadrants (k b : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = k * x + b) →
  (k < 0) →
  (b < 0) →
  f = (λ x, -2 * x - 1) → 
  (f (-1) > 0 ∧ f (1) < 0 ∧ f (0) < 0) := sorry

end passes_through_quadrants_l728_728085


namespace find_base_k_l728_728750

-- Define the conversion condition as a polynomial equation.
def base_conversion (k : ℤ) : Prop := k^2 + 3*k + 2 = 42

-- State the theorem to be proven: given the conversion condition, k = 5.
theorem find_base_k (k : ℤ) (h : base_conversion k) : k = 5 :=
by
  sorry

end find_base_k_l728_728750


namespace factorial_difference_l728_728548

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem factorial_difference : fact 10 - fact 9 = 3265920 := by
  have h1 : fact 9 = 362880 := sorry
  have h2 : fact 10 = 10 * fact 9 := by rw [fact]
  calc
    fact 10 - fact 9
        = 10 * fact 9 - fact 9 := by rw [h2]
    ... = 9 * fact 9 := by ring
    ... = 9 * 362880 := by rw [h1]
    ... = 3265920 := by norm_num

end factorial_difference_l728_728548


namespace termite_ridden_fraction_l728_728801

theorem termite_ridden_fraction (T : ℝ) 
    (h1 : 5/8 * T > 0)
    (h2 : 3/8 * T = 0.125) : T = 1/8 :=
by
  sorry

end termite_ridden_fraction_l728_728801


namespace total_dots_not_visible_l728_728231

theorem total_dots_not_visible (visible : list ℕ)
  (h_visible : visible = [1, 2, 3, 3, 4, 5, 5, 6, 6]) : 
  (4 * 21 - visible.sum = 49) :=
by
  sorry

end total_dots_not_visible_l728_728231


namespace quotient_of_base5_division_correct_l728_728143

-- Base 5 to decimal conversion definition
def base5_to_decimal (digits : List ℕ) : ℕ :=
  digits.reverse.enum_from 0 |>.foldl (λ acc (idx, dgt) => acc + dgt * (5^idx)) 0

-- Testable definition of division in base 5 context
def quotient_base5 (numerator_base5 denominator_base5: List ℕ) : List ℕ :=
  let num := base5_to_decimal numerator_base5
  let denom := base5_to_decimal denominator_base5
  let quotient := num / denom
  decimal_to_base5 quotient

-- Decimal to base 5 conversion helper
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec convert (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else convert (m / 5) ((m % 5) :: acc)
  convert n []

-- Example provided conditions
def numerator_base5 := [2, 3, 0, 1]  -- 2301_5 as digits
def denominator_base5 := [2, 2]      -- 22_5 as digits
def expected_quotient_base5 := [1, 0, 2]  -- 102_5 as digits

-- Proof statement
theorem quotient_of_base5_division_correct :
  quotient_base5 numerator_base5 denominator_base5 = expected_quotient_base5 :=
by
  -- proof goes here
  sorry

end quotient_of_base5_division_correct_l728_728143


namespace consecutive_months_62_days_l728_728901

def is_consecutive (m1 m2 : String) : Prop :=
  (m1 = "July" ∧ m2 = "August") ∨ (m1 = "December" ∧ m2 = "January")

def total_days_in_months (m1 m2 : String) : ℕ :=
  if is_consecutive m1 m2 then 62 else 0

theorem consecutive_months_62_days (m1 m2 : String) (h : total_days_in_months m1 m2 = 62) :
  (m1 = "July" ∧ m2 = "August") ∨ (m1 = "December" ∧ m2 = "January") :=
  sorry

end consecutive_months_62_days_l728_728901


namespace prob_three_heads_in_eight_tosses_l728_728958

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l728_728958


namespace incorrect_differentiation_operations_l728_728084

noncomputable def diff_A_is_correct : Prop :=
  ∀ x : ℝ, deriv (λ x, ln x + 3 / x) x = (1 / x - 3 / x^2)

noncomputable def diff_B_is_correct : Prop :=
  ∀ x : ℝ, deriv (λ x, x^2 * exp x) x = (2 * x * exp x + x^2 * exp x)

noncomputable def diff_C_is_correct : Prop :=
  ∀ x : ℝ, deriv (λ x, 3^x * cos (2 * x)) x = (3^x * (log 3 * cos (2 * x) - 2 * sin (2 * x)))

noncomputable def diff_D_is_correct : Prop :=
  ∀ x : ℝ, deriv (λ x, log (1 / 2) + log x / log 2) x = (1 / (x * log 2))

theorem incorrect_differentiation_operations : diff_A_is_correct ∧ diff_B_is_correct ∧ diff_C_is_correct ∧ diff_D_is_correct := by
  sorry

end incorrect_differentiation_operations_l728_728084


namespace travel_time_l728_728490

-- Definitions: 
def speed := 20 -- speed in km/hr
def distance := 160 -- distance in km

-- Proof statement: 
theorem travel_time (s : ℕ) (d : ℕ) (h1 : s = speed) (h2 : d = distance) : 
  d / s = 8 :=
by {
  sorry
}

end travel_time_l728_728490


namespace no_such_n_exists_l728_728594

theorem no_such_n_exists : ¬ ∃ n : ℕ, n > 0 ∧ 
  (∃ S T : Finset ℕ, disjoint S T ∧ (S ∪ T = {n, n+1, n+2, n+3, n+4, n+5}.to_finset) ∧
  S.prod id = T.prod id) :=
by 
  sorry

end no_such_n_exists_l728_728594


namespace sphere_volume_l728_728466

theorem sphere_volume (r : ℝ) (h : 4 * Real.pi * r^2 = 36 * Real.pi) : (4/3) * Real.pi * r^3 = 36 * Real.pi := 
sorry

end sphere_volume_l728_728466


namespace probability_perfect_square_sum_l728_728057

-- Definitions of parameters and conditions
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def possible_sums := {n | ∃ a b : ℕ, (1 ≤ a ∧ a ≤ 8) ∧ (1 ≤ b ∧ b ≤ 8) ∧ n = a + b}

def perfect_square_sums := {n | is_perfect_square n ∧ n ∈ possible_sums }

def total_outcomes : ℕ := 64

def favorable_outcomes : ℕ := 
  let sums := (for n in possible_sums collect n)
  sums.count (ex ∃ n, n ∈ perfect_square_sums)

-- Statement of the proof to be implemented
theorem probability_perfect_square_sum : 
  favorable_outcomes = 12 →
  total_outcomes = 64 →
  (favorable_outcomes.to_real / total_outcomes.to_real) = (3 / 16) :=
sorry

end probability_perfect_square_sum_l728_728057


namespace min_workers_to_profit_l728_728481

-- Given conditions
def dailyOperatingCost : ℕ := 600
def costPerWorkerPerHour : ℕ := 20
def widgetsPerWorkerPerHour : ℕ := 6
def pricePerWidget : ℚ := 3.5
def workdayHours : ℕ := 10

-- Definition for total daily cost
def dailyCost (n : ℕ) : ℕ :=
  dailyOperatingCost + (workdayHours * costPerWorkerPerHour * n)

-- Definition for total daily revenue
def dailyRevenue (n : ℕ) : ℚ :=
  workdayHours * widgetsPerWorkerPerHour * pricePerWidget * n

-- Main theorem we need to prove
theorem min_workers_to_profit (n : ℕ) : n = 61 → dailyRevenue n > dailyCost n :=
by
  intros h1
  -- Simplifying the statement with given n
  have h2 : dailyCost 61 = 600 + 200 * 61 := by sorry
  have h3 : dailyRevenue 61 = 210 * 61 := by sorry
  show dailyRevenue 61 > dailyCost 61, from 
      calc
        210 * 61 > 600 + 200 * 61 : by sorry
        _ = dailyCost 61    : by sorry
        _ = dailyOperatingCost + workdayHours * costPerWorkerPerHour * 61 : by sorry
        _ = 600 + (10 * 20 * 61) : by sorry
        _ > 600 + (200 * 60) : by sorry
        _ > 600 + 200 * 61 : by sorry
        _ = dailyRevenue 61 : by sorry
  sorry

end min_workers_to_profit_l728_728481


namespace original_number_value_l728_728445

theorem original_number_value (x : ℝ) (h : 0 < x) (h_eq : 10^4 * x = 4 / x) : x = 0.02 :=
sorry

end original_number_value_l728_728445


namespace range_m_l728_728241

theorem range_m {m x : ℝ} (p : (x - m)^2 < 9) (q : log 4 (x + 3) < 1) :
  (¬ q → ¬ p ∨ q) → -2 ≤ m ∧ m ≤ 0 :=
by
  have h1 : ∀ m, (m - 3 ≤ -3) := sorry,
  have h2 : ∀ m, (m + 3 ≥ 1) := sorry,
  have m_range := sorry,
  exact m_range

end range_m_l728_728241


namespace ten_factorial_minus_nine_factorial_l728_728565

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem ten_factorial_minus_nine_factorial :
  factorial 10 - factorial 9 = 3265920 := 
by 
  sorry

end ten_factorial_minus_nine_factorial_l728_728565


namespace factorial_subtraction_l728_728560

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_subtraction : factorial 10 - factorial 9 = 3265920 := by
  sorry

end factorial_subtraction_l728_728560


namespace coefficient_of_x_squared_in_derivative_l728_728361

def f (x : ℝ) : ℝ := (1 - 2 * x) ^ 10

theorem coefficient_of_x_squared_in_derivative :
  (∃ c : ℝ, has_deriv_at f' x (-20 * (1 - 2 * x) ^ 9) ∧ -- derivative condition
  ( ∀ x, f' x = -20 * (1 - 2 * x) ^ 9) ∧ -- simplifying f'
  ∀ (u : ℕ), u = 2 → f.coeff u = -2880) :=
begin
  sorry
end

end coefficient_of_x_squared_in_derivative_l728_728361


namespace sin_double_angle_l728_728712

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ = 2) : Real.sin (2 * θ) = 4 / 5 :=
by
  sorry

end sin_double_angle_l728_728712


namespace polar_circle_equation_l728_728759

theorem polar_circle_equation (r θ θ₀ : ℝ) (h : r = 2 ∧ θ₀ = π ∧ r * cos(θ - θ₀) = -4 * cos(θ)) :
  ρ = -4 * cos θ :=
sorry

end polar_circle_equation_l728_728759


namespace inequality_holds_l728_728809

-- Define the function for the inequality condition
def inequality (n : ℕ) (x : ℝ) : Prop :=
  (1 - x + x^2 / 2) ^ n - (1 - x) ^ n ≤ x / 2

theorem inequality_holds :
  ∀ (n : ℕ) (x : ℝ), 0 < n → (0 ≤ x ∧ x ≤ 1) → inequality n x :=
begin
  intros n x hn hx,
  sorry -- Proof goes here
end

end inequality_holds_l728_728809


namespace smallest_positive_x_for_g_max_l728_728526

def g (x : ℝ) : ℝ := Real.sin (x / 5) + Real.sin (x / 7)

theorem smallest_positive_x_for_g_max :
  ∃ x > 0, g x = 2 ∧ ∀ y > 0, g y = 2 → y ≥ 5850 :=
sorry

end smallest_positive_x_for_g_max_l728_728526


namespace tangent_line_equation_l728_728022

-- Definitions for the conditions
def curve (x : ℝ) : ℝ := 2 * x^2 - x
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Statement of the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), (b = 1 - 3 * 1) ∧ 
  (m = 3) ∧ 
  ∀ (x y : ℝ), y = m * x + b → 3 * x - y - 2 = 0 :=
by
  sorry

end tangent_line_equation_l728_728022


namespace Fran_speed_l728_728331

def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 3.5

theorem Fran_speed :
  ∀ (s : ℝ), (s * Fran_time = Joann_speed * Joann_time) → (s = 120 / 7) :=
by
  intro s h
  sorry

end Fran_speed_l728_728331


namespace convert_base_10_to_base_4_l728_728062

theorem convert_base_10_to_base_4 : nat.to_digits 4 256 = [1, 0, 0, 0, 0] :=
by sorry

end convert_base_10_to_base_4_l728_728062


namespace totalDistanceAfterFifthBounce_l728_728499

-- Define the problem conditions
/-- The initial height from which the ball is dropped -/
def initialHeight : ℝ := 150

/-- The rebound ratio of the ball -/
def reboundRatio : ℝ := 1 / 3

/-- Function to calculate the height after nth bounce -/
noncomputable def heightAfterBounce (n : ℕ) : ℝ :=
  initialHeight * (reboundRatio ^ n)

/-- Function to calculate the total distance traveled by the ball after nth bounce -/
noncomputable def totalDistance (n : ℕ) : ℝ :=
  let descents := initialHeight * (1 + (1 / 3 ^ n) * (1 - (1 / 3)) / (1 - (1 / 3)))
  let ascents := initialHeight * (1 / 3) * ((1 - (1 / 3 ^ n)) / (1 - (1 / 3)))
  descents + ascents

/-- Theorem statement: The total distance traveled by the ball when it hits the ground the fifth time -/
theorem totalDistanceAfterFifthBounce : totalDistance 5 = 298.14 := by
  sorry

end totalDistanceAfterFifthBounce_l728_728499


namespace infinite_solutions_abs_eq_ax_minus_2_l728_728229

theorem infinite_solutions_abs_eq_ax_minus_2 (a : ℝ) :
  (∀ x : ℝ, |x - 2| = ax - 2) ↔ a = 1 :=
by {
  sorry
}

end infinite_solutions_abs_eq_ax_minus_2_l728_728229


namespace two_coins_heads_probability_l728_728071

theorem two_coins_heads_probability :
  let outcomes := [{fst := true, snd := true}, {fst := true, snd := false}, {fst := false, snd := true}, {fst := false, snd := false}],
      favorable := {fst := true, snd := true}
  in (favorable ∈ outcomes) → (favorable :: (List.erase outcomes favorable).length) / outcomes.length = 1 / 4 :=
by
  sorry

end two_coins_heads_probability_l728_728071


namespace factorial_subtraction_l728_728557

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_subtraction : factorial 10 - factorial 9 = 3265920 := by
  sorry

end factorial_subtraction_l728_728557


namespace base_4_equiv_of_256_l728_728064

theorem base_4_equiv_of_256 : 
  ∀ (n : ℕ), n = 256 → (∃ (digits : List ℕ), digits = [1, 0, 0, 0, 0] ∧ 
  n = digits.head * 4^4 + digits.nth 1 * 4^3 + digits.nth 2 * 4^2 + digits.nth 3 * 4^1 + digits.nth 4 * 4^0) := 
by
  intros
  exists [1, 0, 0, 0, 0]
  dsimp
  simp
  sorry

end base_4_equiv_of_256_l728_728064


namespace fifth_largest_divisor_1936000000_l728_728580

theorem fifth_largest_divisor_1936000000 : 
  ∃ d, (d = 121000000) ∧ (∃ N_div : ℕ, N_div = 1936000000 ∧
                         (∀ k : ℕ, k < 5 → ∃ m : ℕ, m divides N_div ∧
                                    m > d)) := 
sorry

end fifth_largest_divisor_1936000000_l728_728580


namespace box_filled_with_cubes_no_leftover_l728_728904

-- Define dimensions of the box
def box_length : ℝ := 50
def box_width : ℝ := 60
def box_depth : ℝ := 43

-- Define volumes of different types of cubes
def volume_box : ℝ := box_length * box_width * box_depth
def volume_small_cube : ℝ := 2^3
def volume_medium_cube : ℝ := 3^3
def volume_large_cube : ℝ := 5^3

-- Define the smallest number of each type of cube
def num_large_cubes : ℕ := 1032
def num_medium_cubes : ℕ := 0
def num_small_cubes : ℕ := 0

-- Theorem statement ensuring the number of cubes completely fills the box
theorem box_filled_with_cubes_no_leftover :
  num_large_cubes * volume_large_cube + num_medium_cubes * volume_medium_cube + num_small_cubes * volume_small_cube = volume_box :=
by
  sorry

end box_filled_with_cubes_no_leftover_l728_728904


namespace nested_subtraction_eq_l728_728593

theorem nested_subtraction_eq (n : ℕ) (h : n = 200) : 
  (λ x, (λ y, (y - ((λ f, f ∘ f) (λ g, g ∘ g) (y - 1))))^[n] x) = 1 →
  x = 201 :=
by
  sorry

end nested_subtraction_eq_l728_728593


namespace range_of_a_l728_728671

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  log 4 (a * 2^x - (4 / 3) * a)

noncomputable def g (x : ℝ) : ℝ :=
  log 4 (4^x + 1) - (1 / 2) * x

theorem range_of_a (a : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, f a x = g x) ↔ (a > 1 ∨ a = -3) :=
sorry

end range_of_a_l728_728671


namespace infinite_solutions_implies_d_eq_five_l728_728582

theorem infinite_solutions_implies_d_eq_five (d : ℝ) :
  (∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) ↔ (d = 5) := by
sorry

end infinite_solutions_implies_d_eq_five_l728_728582


namespace factors_2310_l728_728679

theorem factors_2310 : ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ S.card = 5 ∧ (2310 = S.prod id) :=
by
  sorry

end factors_2310_l728_728679


namespace two_coins_heads_probability_l728_728073

theorem two_coins_heads_probability :
  let outcomes := [{fst := true, snd := true}, {fst := true, snd := false}, {fst := false, snd := true}, {fst := false, snd := false}],
      favorable := {fst := true, snd := true}
  in (favorable ∈ outcomes) → (favorable :: (List.erase outcomes favorable).length) / outcomes.length = 1 / 4 :=
by
  sorry

end two_coins_heads_probability_l728_728073


namespace eccentricity_range_l728_728643

open Real

def hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ x^2 / a^2 - y^2 / b^2 = 1}

def vertex (a : ℝ) : ℝ × ℝ := (a, 0)

def focus (a c : ℝ) : ℝ × ℝ := (-c, 0)

def eccentricity (c a : ℝ) : ℝ := c / a

theorem eccentricity_range {a b c : ℝ} (h1 : a > 0) (h2 : b > 0)
  (h3 : c^2 = a^2 + b^2) :
  let F := focus a c in
  let A := vertex a in
  let P Q : ℝ × ℝ := ((-c, 1), (-c, -1)) in -- Assumes P and Q lie on line x = -c
  ∃ e:ℝ, e = eccentricity c a ∧ 1 < e ∧ e < 2 :=
by
  sorry

end eccentricity_range_l728_728643


namespace probability_three_heads_in_eight_tosses_l728_728939

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l728_728939


namespace Fran_speed_l728_728332

def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 3.5

theorem Fran_speed :
  ∀ (s : ℝ), (s * Fran_time = Joann_speed * Joann_time) → (s = 120 / 7) :=
by
  intro s h
  sorry

end Fran_speed_l728_728332


namespace average_after_discarding_l728_728097

theorem average_after_discarding (n1 n2 : ℕ) (avg : ℕ) (sum : ℕ) (count : ℕ) :
  count = 50 → avg = 50 → sum = avg * count → n1 = 45 → n2 = 55 →
  (sum - (n1 + n2)) / (count - 2) = 50 :=
by
  intro h_count h_avg h_sum h_n1 h_n2
  have h_sum_value : sum = 2500 := by
    rw [h_avg, h_count, h_sum]
  rw [h_sum_value, h_n1, h_n2]
  have h_disc_sum : 2500 - (45 + 55) = 2400 := by norm_num
  rw [h_disc_sum]
  have h_new_avg : 2400 / 48 = 50 := by norm_num
  exact h_new_avg

end average_after_discarding_l728_728097


namespace tangent_line_at_1_l728_728598

noncomputable def f (x : ℝ) : ℝ := x + Real.log x

theorem tangent_line_at_1 : ∀ x : ℝ, (1, f(1)) = (1, 1) → ( ∃ m b : ℝ, (x = 1 → f'(x) = 2) ∧ (y = 2*x - 1) ) :=
by
  sorry

end tangent_line_at_1_l728_728598


namespace count_valid_sequences_l728_728684

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_sequence (x : ℕ → ℕ) : Prop :=
  (x 7 % 2 = 0) ∧ (∀ i < 7, (x i % 2 = 0 → x (i + 1) % 2 = 1) ∧ (x i % 2 = 1 → x (i + 1) % 2 = 0))

theorem count_valid_sequences : ∃ n, 
  n = 78125 ∧ 
  ∃ x : ℕ → ℕ, 
    (∀ i < 8, 0 ≤ x i ∧ x i ≤ 9) ∧ valid_sequence x :=
sorry

end count_valid_sequences_l728_728684


namespace point_slope_eq_l728_728034

theorem point_slope_eq (x y : ℝ) (x1 y1 : ℝ) (m : ℝ) :
  x1 = -1 → y1 = -2 → m = 3 → (y - y1 = m * (x - x1)) → (y + 2 = 3 * (x + 1)) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end point_slope_eq_l728_728034


namespace base_4_equiv_of_256_l728_728063

theorem base_4_equiv_of_256 : 
  ∀ (n : ℕ), n = 256 → (∃ (digits : List ℕ), digits = [1, 0, 0, 0, 0] ∧ 
  n = digits.head * 4^4 + digits.nth 1 * 4^3 + digits.nth 2 * 4^2 + digits.nth 3 * 4^1 + digits.nth 4 * 4^0) := 
by
  intros
  exists [1, 0, 0, 0, 0]
  dsimp
  simp
  sorry

end base_4_equiv_of_256_l728_728063


namespace binomial_constant_term_l728_728392

theorem binomial_constant_term : 
  (∃ c : ℕ, ∀ x : ℝ, (x + (1 / (3 * x)))^8 = c * (x ^ (4 * 2 - 8) / 3)) → 
  ∃ c : ℕ, c = 28 :=
sorry

end binomial_constant_term_l728_728392


namespace minimal_height_exists_l728_728185

noncomputable def height_min_material (x : ℝ) : ℝ := 4 / (x^2)

theorem minimal_height_exists
  (x h : ℝ)
  (volume_cond : x^2 * h = 4)
  (surface_area_cond : h = height_min_material x) :
  h = 1 := by
  sorry

end minimal_height_exists_l728_728185


namespace translated_function_area_of_triangle_l728_728401

-- 1: Prove the expression of the translated function
theorem translated_function (b : ℝ) (h : 4 = 3 * 1 + b) : b = 1 :=
by
  sorry

-- 2: Prove the area of the triangle formed by the translated function's graph and the coordinate axes
theorem area_of_triangle (b : ℝ) (h : b = 1) : 
  let y := λ x : ℝ, 3 * x + b in 
  (1 / 2) * (abs (-1 / 3)) * (abs 1) = 1 / 6 :=
by
  sorry

end translated_function_area_of_triangle_l728_728401


namespace probability_of_three_heads_in_eight_tosses_l728_728989

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l728_728989


namespace water_left_after_dumping_l728_728850

/--
Given:
1. The water pressure of a sink has a steady flow of 2 cups per 10 minutes for the first 30 minutes.
2. The water pressure still flows at 2 cups per 10 minutes for the next 30 minutes after.
3. For the next hour, the water pressure maximizes to 4 cups per 10 minutes and stops.
4. Shawn has to dump half of the water away.
Prove:
The amount of water left is 18 cups.
-/
theorem water_left_after_dumping (h1 : ∀ t : ℕ, 0 ≤ t ∧ t < 30 → 2 * (t / 10) = water_flow t)
                                   (h2 : ∀ t : ℕ, 30 ≤ t ∧ t < 60 → 2 * ((t - 30) / 10) = water_flow t)
                                   (h3 : ∀ t : ℕ, 60 ≤ t ∧ t < 120 → 4 * ((t - 60) / 10) = water_flow t)
                                   (h4 : Shawn_dumps_half_water) :
                                   total_water_after_dumping = 18 := by 
                                   sorry

end water_left_after_dumping_l728_728850


namespace min_abs_phi_l728_728426

open Real

theorem min_abs_phi {k : ℤ} :
  ∃ (φ : ℝ), ∀ (k : ℤ), φ = - (5 * π) / 6 + k * π ∧ |φ| = π / 6 := sorry

end min_abs_phi_l728_728426


namespace probability_of_three_heads_in_eight_tosses_l728_728973

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l728_728973


namespace laurent_series_expansion_region1_laurent_series_expansion_region2_laurent_series_expansion_region3_l728_728571

noncomputable def f (z : ℂ) : ℂ := (2 * z + 1) / (z^2 + z - 2)

def region1 (z : ℂ) : Prop := abs z < 1
def region2 (z : ℂ) : Prop := 1 < abs z ∧ abs z < 2
def region3 (z : ℂ) : Prop := abs z > 2

theorem laurent_series_expansion_region1 (z : ℂ) (h : region1 z) :
  f z = -1/2 - (3/4) * z - (7/8) * z^2 - (15/16) * z^3 + sorry := sorry

theorem laurent_series_expansion_region2 (z : ℂ) (h : region2 z) :
  f z = (∑ n : ℕ in Finset.range 1, sorry) / z^n + (∑ n : ℕ, sorry) / (2^n) := sorry

theorem laurent_series_expansion_region3 (z : ℂ) (h : region3 z) :
  f z = 2 / z - (1 / z^2) + (5 / z^3) - (7 / z^4) + sorry := sorry

end laurent_series_expansion_region1_laurent_series_expansion_region2_laurent_series_expansion_region3_l728_728571


namespace probability_three_heads_in_eight_tosses_l728_728997

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l728_728997


namespace percentage_after_10_hours_time_to_reduce_by_more_than_one_third_l728_728911

-- Define the initial conditions and constants
def pollutant_content (P0 : ℝ) (k : ℝ) (t : ℝ) : ℝ :=
  P0 * Real.exp (-k * t)

-- Given condition: 10% eliminated in the first 5 hours
def condition_eliminated_10_percent (P0 : ℝ) (k : ℝ) : Prop :=
  pollutant_content P0 k 5 = 0.9 * P0

-- Prove percentage remaining after 10 hours
theorem percentage_after_10_hours (P0 k : ℝ) (h : condition_eliminated_10_percent P0 k) :
  pollutant_content P0 k 10 = 0.81 * P0 := 
sorry

-- Prove at least 21 hours to reduce pollutant by more than one-third
theorem time_to_reduce_by_more_than_one_third (P0 k : ℝ) (h : condition_eliminated_10_percent P0 k) :
  ∃ t ≥ 21, pollutant_content P0 k t ≤ (2 / 3) * P0 :=
sorry

end percentage_after_10_hours_time_to_reduce_by_more_than_one_third_l728_728911


namespace number_of_meetings_l728_728054

-- Definitions based on the given conditions
def track_circumference : ℕ := 300
def boy1_speed : ℕ := 7
def boy2_speed : ℕ := 3
def both_start_simultaneously := true

-- The theorem to prove
theorem number_of_meetings (h1 : track_circumference = 300) (h2 : boy1_speed = 7) (h3 : boy2_speed = 3) (h4 : both_start_simultaneously) : 
  ∃ n : ℕ, n = 1 := 
sorry

end number_of_meetings_l728_728054


namespace solution_set_m_l728_728654

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -log x - x
  else -log (-x) + x

theorem solution_set_m :
  {m : ℝ | f (1 / m) < log (1 / 2) - 2} =
  {m : ℝ | m ∈ Ioo (-1/2) 0 ∪ Ioo 0 1/2 } :=
by
  sorry

end solution_set_m_l728_728654


namespace spherical_to_rectangular_coords_l728_728577

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), ρ = 15 ∧ θ = 5 * Real.pi / 4 ∧ φ = Real.pi / 4 →
  let x := ρ * Real.sin φ * Real.cos θ,
      y := ρ * Real.sin φ * Real.sin θ,
      z := ρ * Real.cos φ in
  (x, y, z) = (-15 / 2, -15 / 2, 15 * Real.sqrt 2 / 2) :=
by
  intros ρ θ φ h
  obtain ⟨hρ, hθ, hφ⟩ := h
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coords_l728_728577


namespace intersection_is_correct_l728_728255

def M : Set ℤ := {-2, 1, 2}
def N : Set ℤ := {1, 2, 4}

theorem intersection_is_correct : M ∩ N = {1, 2} := 
by {
  sorry
}

end intersection_is_correct_l728_728255


namespace plane_equation_l728_728778

variables {x y z : ℝ}
def w : ℝ^3 := ⟨3, -2, 3⟩
def v : ℝ^3 := ⟨x, y, z⟩
def proj_w_v : ℝ^3 := ⟨6, -4, 6⟩

theorem plane_equation (h : proj_w_v = (real_inner v w / real_inner w w) • w) :
  3 * x - 2 * y + 3 * z - 44 = 0 :=
sorry

end plane_equation_l728_728778


namespace max_marked_points_l728_728371

theorem max_marked_points (segments : ℕ) (ratio : ℚ) (h_segments : segments = 10) (h_ratio : ratio = 3 / 4) : 
  ∃ n, n ≤ (segments * 2 / 2) ∧ n = 10 :=
by
  sorry

end max_marked_points_l728_728371


namespace base7_to_base6_conversion_l728_728187

theorem base7_to_base6_conversion : 
  ∀ (b7 : ℕ), b7 = 3 * 7^2 + 5 * 7^1 + 1 * 7^0 → 
    ∃ (b6 : ℕ), b6 = 5 * 6^2 + 0 * 6^1 + 3 * 6^0 ∧ 
      ∀ n, n = nat.digits 6 b7 → n = [5, 0, 3] := by
  intros b7 h_conv
  use (5 * 6^2 + 0 * 6^1 + 3 * 6^0)
  split
  · refl
  · intros n h_digits
    rw [← h_digits]
    exact nat.digits_eq_digits_of_valid_input 6 b7 h_conv

end base7_to_base6_conversion_l728_728187


namespace club_committee_impossible_l728_728909

theorem club_committee_impossible :
  let members := 11 in
  let min_members := 3 in
  let different_by_one (committee1 committee2 : Finset ℕ) := 
    (|committee1 \ committee2| + |committee2 \ committee1| = 1) in
  let possible_committees := {s : Finset ℕ | s.card ≥ min_members ∧ s ⊆ Finset.range members} in
  ¬(∀ current_state : List (Finset ℕ),
    (∀ (i j : ℕ), i < j → i < current_state.length → j < current_state.length → current_state.nth i ≠ current_state.nth j) ∧
    (∀ (i : ℕ), i < current_state.length - 1 →
      different_by_one (current_state.nth i) (current_state.nth (i + 1))) →
    ∃ used_committees : Finset (Finset ℕ),
      used_committees ⊆ possible_committees ∧
      used_committees.card = possible_committees.card) :=
sorry

end club_committee_impossible_l728_728909


namespace distinct_prime_factors_2310_l728_728675

theorem distinct_prime_factors_2310 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ (S.card = 5) ∧ (S.prod id = 2310) := by
  sorry

end distinct_prime_factors_2310_l728_728675


namespace parallelogram_circle_intersect_l728_728857

open_locale big_operators

variables {A B C D B' C' D' : Type*}
variables [field A] [field B] [field C] [field D] [field B'] [field C'] [field D']
variables [module C A] [module D A] [module B' A] [module C' A] [module D' A]
variables [has_mul AC] [has_mul ('AB)] [has_mul ('AC)] [has_mul ('AD)]
variables [is_parallelogram A B C D B' C' D']

theorem parallelogram_circle_intersect (AC' AB' : A) (AC AD : D') 
  (H : is_parallelogram A B C D B' C' D') : 
  AC * 'AC' = 'AB' * 'AB' + 'AD' * 'AD' :=
sorry

end parallelogram_circle_intersect_l728_728857


namespace yellow_tint_percentage_l728_728119

variable (original_volume : ℕ) (yellow_percentage : ℚ) (additional_yellow : ℕ) (additional_red : ℕ)

-- Define the original amounts and additional volumes
def original_yellow_volume : ℚ := original_volume * yellow_percentage
def new_yellow_volume : ℚ := original_yellow_volume + additional_yellow
def new_total_volume : ℚ := original_volume + additional_yellow + additional_red

-- Proving the final percentage
theorem yellow_tint_percentage
  (h1 : original_volume = 40)
  (h2 : yellow_percentage = 35 / 100)
  (h3 : additional_yellow = 3)
  (h4 : additional_red = 2) :
  (new_yellow_volume / new_total_volume * 100).round = 38 :=
by
  unfold original_yellow_volume new_yellow_volume new_total_volume
  simp [h1, h2, h3, h4]
  norm_num
  sorry

end yellow_tint_percentage_l728_728119


namespace solve_for_q_l728_728012

theorem solve_for_q (k r q : ℕ) (h1 : 4 / 5 = k / 90) (h2 : 4 / 5 = (k + r) / 105) (h3 : 4 / 5 = (q - r) / 150) : q = 132 := 
  sorry

end solve_for_q_l728_728012


namespace speed_of_second_person_l728_728491

-- Definitions based on the conditions
def speed_person1 := 70 -- km/hr
def distance_AB := 600 -- km

def time_traveled := 4 -- hours (from 10 am to 2 pm)

-- The goal is to prove that the speed of the second person is 80 km/hr
theorem speed_of_second_person :
  (distance_AB - speed_person1 * time_traveled) / time_traveled = 80 := 
by 
  sorry

end speed_of_second_person_l728_728491


namespace range_of_c_l728_728242

noncomputable def is_monotonically_decreasing (c: ℝ) : Prop := ∀ x1 x2: ℝ, x1 < x2 → c^x2 ≤ c^x1

def inequality_holds (c: ℝ) : Prop := ∀ x: ℝ, x^2 + x + (1/2)*c > 0

theorem range_of_c (c: ℝ) (h1: c > 0) :
  ((is_monotonically_decreasing c ∨ inequality_holds c) ∧ ¬(is_monotonically_decreasing c ∧ inequality_holds c)) 
  → (0 < c ∧ c ≤ 1/2 ∨ c ≥ 1) := 
sorry

end range_of_c_l728_728242


namespace average_rainfall_total_l728_728128

theorem average_rainfall_total
  (rain_first_30 : ℝ)
  (rain_next_30 : ℝ)
  (rain_next_hour : ℝ)
  (total_time : ℝ)
  (total_rainfall : ℝ) :
  rain_first_30 = 5 →
  rain_next_30 = 5 / 2 →
  rain_next_hour = 1 / 2 →
  total_time = 30 + 30 + 60 →
  total_rainfall = rain_first_30 + rain_next_30 + rain_next_hour →
  total_rainfall = 8 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h5
  have : total_rainfall = 5 + 2.5 + 0.5 := by rw h5
  norm_num at this
  exact this

#check average_rainfall_total

end average_rainfall_total_l728_728128


namespace probability_exactly_three_heads_l728_728963
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l728_728963


namespace binom_7_4_eq_35_l728_728179

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l728_728179


namespace direction_vector_b_l728_728581

theorem direction_vector_b :
  ∃ b : ℚ, ∃ scale : ℚ, 
  (b ≠ 0 ∧ scale ≠ 0 ∧ 
   (b = (scale * (-4))) ∧ 
   (-2 = (scale * (-3)))) ∧ 
  b = -8 / 3 :=
by {
  use (-8) / 3,
  use (2 / 3),
  split,
  { split,
    { norm_num, },
    { norm_num, }, },
  split;
  norm_num,
  sorry -- Placeholder for detailed proof steps
}

end direction_vector_b_l728_728581


namespace solve_equation_l728_728217

theorem solve_equation :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ x : ℝ, x^2 + 10 * x = 34 ∧ x = real.sqrt a - b) ∧ a + b = 64 :=
by sorry

end solve_equation_l728_728217


namespace binom_7_4_l728_728162

theorem binom_7_4 : Nat.choose 7 4 = 35 := 
by
  sorry

end binom_7_4_l728_728162


namespace problem1_problem2_l728_728465

-- Define the necessary variables
variables (a b x : ℝ)

-- Problem 1: Proving the inequality
theorem problem1 (h₁ : a ∈ ℝ) (h₂ : b ∈ ℝ) : 2 * (a^2 + b^2) ≥ (a + b)^2 := 
sorry

-- Problem 2: Proving the non-negativity condition
theorem problem2 (h₁ : x ∈ ℝ) (h₂ : a = x^2 - 1) (h₃ : b = 2 * x + 2) : ¬(a < 0 ∧ b < 0) :=
sorry

end problem1_problem2_l728_728465


namespace mindy_mork_ratio_l728_728516

variable (M K : ℝ)
variable h1 : 0.15 * M + 0.45 * K = 0.21 * (M + K)

theorem mindy_mork_ratio (M K : ℝ) (h1 : 0.15 * M + 0.45 * K = 0.21 * (M + K)) : M / K = 4 := sorry

end mindy_mork_ratio_l728_728516


namespace trigonometric_identity_l728_728708

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
by 
  sorry

end trigonometric_identity_l728_728708


namespace car_speed_constant_l728_728114

-- Conditions
def cyclist_speed : ℝ := 15 -- miles per hour
def car_wait_time_1 : ℝ := 6 / 60 -- hours, first waiting period
def car_wait_time_2 : ℝ := 18 / 60 -- hours, second waiting period
def total_wait_time : ℝ := car_wait_time_1 + car_wait_time_2

-- Cyclist distance during both waiting periods
def cyclist_distance_1 : ℝ := cyclist_speed * car_wait_time_1
def cyclist_distance_2 : ℝ := cyclist_speed * car_wait_time_2
def total_cyclist_distance : ℝ := cyclist_distance_1 + cyclist_distance_2


-- Required to prove that the car's speed (v) is 20 miles per hour
theorem car_speed_constant (v : ℝ) 
  (h1 : cyclist_speed = 15) 
  (h2 : car_wait_time_1 = 6 / 60)
  (h3 : car_wait_time_2 = 18 / 60)
  (h4 : total_cyclist_distance = 6) :
  v = 20 := 
  by
  sorry

end car_speed_constant_l728_728114


namespace min_y_ellipse_l728_728572

-- Defining the ellipse equation
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 49) + ((y - 3)^2 / 25) = 1

-- Problem statement: Prove that the smallest y-coordinate is -2
theorem min_y_ellipse : 
  ∀ x y, ellipse x y → y ≥ -2 :=
sorry

end min_y_ellipse_l728_728572


namespace choose_athlete_B_l728_728515

variable (SA2 : ℝ) (SB2 : ℝ)
variable (num_shots : ℕ) (avg_rings : ℝ)

-- Conditions
def athlete_A_variance := SA2 = 3.5
def athlete_B_variance := SB2 = 2.8
def same_number_of_shots := true -- Implicit condition, doesn't need proof
def same_average_rings := true -- Implicit condition, doesn't need proof

-- Question: prove Athlete B should be chosen
theorem choose_athlete_B 
  (hA_var : athlete_A_variance SA2)
  (hB_var : athlete_B_variance SB2)
  (same_shots : same_number_of_shots)
  (same_avg : same_average_rings) :
  "B" = "B" :=
by 
  sorry

end choose_athlete_B_l728_728515


namespace cubic_roots_l728_728752

theorem cubic_roots (a b x₃ : ℤ)
  (h1 : (2^3 + a * 2^2 + b * 2 + 6 = 0))
  (h2 : (3^3 + a * 3^2 + b * 3 + 6 = 0))
  (h3 : 2 * 3 * x₃ = -6) :
  a = -4 ∧ b = 1 ∧ x₃ = -1 :=
by {
  sorry
}

end cubic_roots_l728_728752


namespace maximize_multiplication_table_sum_l728_728408

-- Lean statement defining the given numbers and the target maximum product
theorem maximize_multiplication_table_sum :
  ∃ (a b c d e f : ℕ), 
  ({a, b, c, d, e, f} = {2, 3, 5, 7, 11, 17, 19}) ∧ 
  (a + b + c + d + e + f = 64) ∧ 
  ((a + b + c) * (d + e + f) = 1024) :=
sorry

end maximize_multiplication_table_sum_l728_728408


namespace fraction_three_fourths_of_forty_five_l728_728868

-- Define the problem: What is (3/4) * 45?
def fraction_mul (a b c : ℝ) : ℝ := (a * c) / b

-- Problem statement
theorem fraction_three_fourths_of_forty_five : fraction_mul 3 4 45 = 33.75 :=
by
  sorry

end fraction_three_fourths_of_forty_five_l728_728868


namespace school_year_hours_per_week_l728_728770

def summer_weekly_hours : ℝ := 60
def summer_total_weeks : ℝ := 10
def summer_earnings : ℝ := 7500
def school_year_total_weeks : ℝ := 50
def school_year_earnings : ℝ := 7500

def hourly_wage (total_earnings : ℝ) (weekly_hours : ℝ) (total_weeks : ℝ) : ℝ :=
  total_earnings / (weekly_hours * total_weeks)

def school_year_weekly_hours (total_earnings : ℝ) (rate : ℝ) (total_weeks : ℝ) : ℝ :=
  total_earnings / (rate * total_weeks)

theorem school_year_hours_per_week :
  let r := hourly_wage summer_earnings summer_weekly_hours summer_total_weeks
  in school_year_weekly_hours school_year_earnings r school_year_total_weeks = 12 :=
by {
  sorry
}

end school_year_hours_per_week_l728_728770


namespace part1_double_root_f_expression_part2_max_value_positive_l728_728247

variables {R : Type*} [LinearOrderedField R]

def quadratic_function (a b c x : R) := a * x^2 + b * x + c

-- Conditions
variables (a b c : R)
variable (h : a < 0)
variable (h₂ : ∀ x : R, (x = 1 ∨ x = 3) ↔ quadratic_function a (b + 2) c x = 0)

-- Part 1
theorem part1_double_root_f_expression (h3 : discriminant a b c + 6 * a = 0) :
  quadratic_function (-1 / 5) (-6 / 5) (-3 / 5) x = 0 := sorry

-- Part 2
theorem part2_max_value_positive (h4 : ∀ x, quadratic_function a b c x < 0) 
  (h5 : ∃ x, 4*5*x^2+5*x-2=0) : 
  a > 1/5 := sorry

end part1_double_root_f_expression_part2_max_value_positive_l728_728247


namespace range_of_a_l728_728303

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → a ∈ set.Icc (-2 : ℝ) 4 :=
by
  sorry

end range_of_a_l728_728303


namespace find_a_a2_a4_a6_a8_sum_l728_728203

theorem find_a_a2_a4_a6_a8_sum :
  let a := (2: ℤ); a1 := (3: ℤ); a2 := (6561: ℤ); a8 := 1 
  in (a + a2 + a4 + a6 + a8) = 3281 :=
by
  sorry

end find_a_a2_a4_a6_a8_sum_l728_728203


namespace max_sum_log2_geometric_seq_l728_728757

noncomputable def geometric_sequence (n : ℕ) (a₃ a₆ : ℝ) (q : ℝ) : ℝ :=
  a₃ * q ^ (n - 3)

theorem max_sum_log2_geometric_seq :
  ∀ (a₃ a₆ : ℝ) (a_n : ℕ → ℝ),
  a₃ = 8 →
  a₆ = 1 →
  (∀ n, a_n n = geometric_sequence n 8 1 (1 / 2)) →
  ∑ n in (finset.range 6).filter (λ n, n ≠ 2) \[remove🇧🇷 condition,
  sorry :=
begin
  -- let's leave this to be filled in the future
  sorry
end

end max_sum_log2_geometric_seq_l728_728757


namespace solution_set_of_inequality_l728_728043

theorem solution_set_of_inequality (x : ℝ) : 
  (|x| * (1 - 2 * x) > 0) ↔ (x ∈ ((Set.Iio 0) ∪ (Set.Ioo 0 (1/2)))) :=
by
  sorry

end solution_set_of_inequality_l728_728043


namespace probability_of_three_heads_in_eight_tosses_l728_728987

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l728_728987


namespace num_subsets_P_l728_728105

-- Define the given sets M and N
def M : Set ℤ := {0, 1, 2, 3, 4}
def N : Set ℤ := {1, 3, 5}

-- Define the intersection P
def P : Set ℤ := M ∩ N

-- The theorem to prove the number of subsets of P
theorem num_subsets_P : ∃ n : ℕ, n = 2 ∧ 2^n = 4 :=
by
  have hP : P = {1, 3} := sorry
  have card_P : (P : Finset ℤ).card = 2 := sorry
  use 2
  simp [card_P]
  sorry

end num_subsets_P_l728_728105


namespace binom_7_4_l728_728159

theorem binom_7_4 : Nat.choose 7 4 = 35 := 
by
  sorry

end binom_7_4_l728_728159


namespace coin_toss_sequence_probability_l728_728312

/-- The probability of getting 4 heads followed by 3 tails and then finishing with 3 heads 
in a sequence of 10 coin tosses is 1/1024. -/
theorem coin_toss_sequence_probability : 
  ( (1 / 2 : ℝ) ^ 4) * ( (1 / 2 : ℝ) ^ 3) * ((1 / 2 : ℝ) ^ 3) = 1 / 1024 := 
by sorry

end coin_toss_sequence_probability_l728_728312


namespace domain_f_odd_function_f_l728_728239

-- Definitions based on the given conditions
def f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x : ℝ) : ℝ := log a ( (1 + x) / (1 - x) )

-- Proving the domain
theorem domain_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : ∀ x, -1 < x ∧ x < 1 ↔ ( (1 + x) / (1 - x) > 0) := 
sorry

-- Proving the parity: odd function
theorem odd_function_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : ∀ x, f a h1 h2 (-x) = - f a h1 h2 x :=
sorry

end domain_f_odd_function_f_l728_728239


namespace find_y_l728_728719

theorem find_y (x y : ℝ) (h1 : sqrt (3 + sqrt x) = 4) (h2 : x + y = 58) : y = -111 :=
by {
  have h3 : sqrt x = 13, from by {
    rw [←sqrt_eq_iff_eq_sq, real.sqrt_eq_iff_sq_eq] at h1,
    have : sqrt (3 + sqrt x) = sqrt 16, from h1,
    rw sqrt_inj (by norm_num) at this,
    exact this,
  },
  have h4 : x = 169, from by {
    rw [real.sqrt_eq_iff_sq_eq] at h3,
    exact h3,
  },
  have h5 : y = 58 - x, from by {
    linarith,
  },
  rw h4 at h5,
  norm_num at h5,
  exact h5,
}

end find_y_l728_728719


namespace positive_difference_median_mode_eq_nineteen_point_five_l728_728874

open Real

def data := [30, 31, 32, 33, 33, 33, 40, 41, 42, 43, 44, 45, 51, 51, 51, 52, 53, 55, 60, 61, 62, 64, 65, 67, 71, 72, 73, 74, 75, 76]
def mode := 33
def median := 52.5

def positive_difference (a b : ℝ) : ℝ := abs (a - b)

theorem positive_difference_median_mode_eq_nineteen_point_five :
  positive_difference median mode = 19.5 :=
by
  sorry

end positive_difference_median_mode_eq_nineteen_point_five_l728_728874


namespace find_box_value_l728_728721

theorem find_box_value :
  ∃ x : ℝ, 1 + 1.1 + 1.11 + x = 4.44 ∧ x = 1.23 :=
by
  exists 1.23
  split
  · norm_num
  · norm_num
  sorry

end find_box_value_l728_728721


namespace convert_base_10_to_base_4_l728_728061

theorem convert_base_10_to_base_4 : nat.to_digits 4 256 = [1, 0, 0, 0, 0] :=
by sorry

end convert_base_10_to_base_4_l728_728061


namespace modular_inverse_addition_l728_728205

theorem modular_inverse_addition :
  (3 * 9 + 9 * 37) % 63 = 45 :=
by
  sorry

end modular_inverse_addition_l728_728205


namespace probability_three_heads_in_eight_tosses_l728_728934

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l728_728934


namespace final_amount_correct_l728_728132

/-- Ali's initial money in various denominations -/
def initial_money : ℝ :=
  (7 * 5) + 10 + (3 * 20) + 50 + 8 + (10 * 0.25)

/-- Money spent and change received in the morning after grocery shopping -/
def morning_spent : ℝ := 50 + 20 + 5
def morning_change : ℝ := 3 + (8 * 0.25) + (10 * 0.10)

/-- Money spent and change received in the coffee shop -/
def coffee_spent : ℝ := 3.75
def coffee_change : ℝ := 5 - 3.75 -- Change received in 1-dollar coins and 25-cent coins

/-- Money received from a friend in the afternoon -/
def afternoon_received : ℝ := 42

/-- Money spent and additional money needed in the evening after buying a book -/
def book_spent : ℝ := 11.25
def evening_paid : ℝ := 10 -- Paid with two 5 dollar bills

/-- Calculate the final amount of money left with Ali after all transactions -/
def final_money : ℝ :=
  initial_money - (morning_spent - morning_change) - coffee_spent +
  coffee_change + afternoon_received - book_spent + evening_paid

theorem final_amount_correct : final_money = 123.50 := by
  unfold final_money initial_money morning_spent morning_change coffee_spent
  unfold coffee_change afternoon_received book_spent evening_paid
  simp only [sub_sub_sub_cancel_right, add_sub_sub_cancel, sub_sub_self, add_sub_assoc, add_sub_cancel], simp -- use basic arithmetic rules
  -- detailed calculations
  calc
    initial_money - (morning_spent - morning_change) - coffee_spent + coffee_change + afternoon_received - book_spent + evening_paid 
    = 165.50 - (75 - 6) - 3.75 + 1.25 + 42 - 11.25 + 10 : by norm_num -- this simplifies and verifies the sequence of operations.
  norm_num -- Computes the final simplified result to confirm the exact value
  sorry

end final_amount_correct_l728_728132


namespace new_median_after_adding_10_l728_728480

def mean (s : List ℝ) : ℝ := s.sum / s.length

theorem new_median_after_adding_10
  (a b c d : ℕ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_collection_mean : mean [a, b, c, d] = 6.5)
  (h_unique_mode : (Multiset.mode (a::b::c::d::[])).unique = [6])
  (h_median : ([a, b, c, d].sort (· ≤ ·))[(4 - 1) / 2] = 7) :
  ([6, 6, 7, 7].sort (· ≤ ·)).nth 2 = some 7 :=
by
  sorry

end new_median_after_adding_10_l728_728480


namespace unique_function_l728_728206

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x y : ℝ, 0 < x → 0 < y → f(x * f(y)) = y * f(x)

axiom h2 : ∀ (ε : ℝ), ε > 0 → ∃ (N : ℝ), ∀ x : ℝ, x > N → f(x) < ε

theorem unique_function : ∀ x : ℝ, 0 < x → f(x) = 1 / x :=
by
  sorry

end unique_function_l728_728206


namespace trigonometric_identity_l728_728701

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = (2 / 5) :=
by
  sorry

end trigonometric_identity_l728_728701


namespace man_slips_each_day_l728_728118

theorem man_slips_each_day (d h : ℕ) (climb slip : ℕ) (days : ℕ)
  (depth_eq : d = 30) (climb_eq : climb = 4) 
  (total_days_eq : days = 27) : (slip = 3) :=
by
  have end_of_26th_day : d - climb = 26 :=
    by rw [depth_eq, climb_eq]; exact rfl
  have total_climb_26_days : climb * (days - 1) = 104 :=
    by rw [climb_eq, total_days_eq]; exact rfl
  have height_after_slip : total_climb_26_days - (d - climb) = 78 :=
    by rw [end_of_26th_day, total_climb_26_days]; exact rfl
  have average_slip_per_day : (total_climb_26_days - (d - climb)) / (days - 1) = 3 :=
    by rw [height_after_slip, total_days_eq]; exact rfl
  show slip = 3 from average_slip_per_day

end man_slips_each_day_l728_728118


namespace Gake_needs_fewer_boards_than_Tom_l728_728424

noncomputable def Tom_boards_needed : ℕ :=
  let width_board : ℕ := 5
  let width_char : ℕ := 9
  (3 * width_char + 2 * 6) / width_board

noncomputable def Gake_boards_needed : ℕ :=
  let width_board : ℕ := 5
  let width_char : ℕ := 9
  (4 * width_char + 3 * 1) / width_board

theorem Gake_needs_fewer_boards_than_Tom :
  Gake_boards_needed < Tom_boards_needed :=
by
  -- Here you will put the actual proof steps
  sorry

end Gake_needs_fewer_boards_than_Tom_l728_728424


namespace baseball_weight_l728_728878

theorem baseball_weight
  (weight_total : ℝ)
  (weight_soccer_ball : ℝ)
  (n_soccer_balls : ℕ)
  (n_baseballs : ℕ)
  (total_weight : ℝ)
  (B : ℝ) :
  n_soccer_balls * weight_soccer_ball + n_baseballs * B = total_weight →
  n_soccer_balls = 9 →
  weight_soccer_ball = 0.8 →
  n_baseballs = 7 →
  total_weight = 10.98 →
  B = 0.54 := sorry

end baseball_weight_l728_728878


namespace find_m_n_l728_728602

theorem find_m_n (m n : ℕ) (positive_m : 0 < m) (positive_n : 0 < n)
  (h1 : m = 3) (h2 : n = 4) :
    Real.arctan (1 / 3) + Real.arctan (1 / 4) + Real.arctan (1 / m) + Real.arctan (1 / n) = π / 2 :=
  by 
    -- Placeholder for the proof
    sorry

end find_m_n_l728_728602


namespace sqrt_720_simplified_l728_728010

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  have h1 : 720 = 2^4 * 3^2 * 5 := by norm_num
  -- Here we use another proven fact or logic per original conditions and definition
  sorry

end sqrt_720_simplified_l728_728010


namespace value_of_a_l728_728302

theorem value_of_a :
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - -1)^2 + (y - 1)^2 = 4) := sorry

end value_of_a_l728_728302


namespace sqrt_720_simplified_l728_728005

theorem sqrt_720_simplified : Real.sqrt 720 = 6 * Real.sqrt 5 := sorry

end sqrt_720_simplified_l728_728005


namespace sufficient_and_necessary_condition_l728_728349

variable {a : ℕ → ℝ}
variable {a1 a2 : ℝ}
variable {q : ℝ}

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
  (∀ n, a n = a1 * q ^ n)

noncomputable def increasing (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < a (n + 1))

theorem sufficient_and_necessary_condition
  (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (h_geom : geometric_sequence a a1 q)
  (h_a1_pos : a1 > 0)
  (h_a1_lt_a2 : a1 < a1 * q) :
  increasing a ↔ a1 < a1 * q := 
sorry

end sufficient_and_necessary_condition_l728_728349


namespace bounded_magnitude_l728_728101

noncomputable def bounded_function {f : ℝ → ℝ} (g : ℝ → ℝ) :=
  ∀ (x : ℝ), f(x) + deriv (deriv f x) = -x * g(x) * deriv f x ∧ g(x) ≥ 0

theorem bounded_magnitude {f : ℝ → ℝ} (g : ℝ → ℝ) (h : bounded_function g) :
  ∃ (c : ℝ), ∀ x, |f x| ≤ c :=
sorry

end bounded_magnitude_l728_728101


namespace monotonic_increasing_interval_l728_728027

open Real

noncomputable def f (x : ℝ) : ℝ := log (1 / 4) (-x^2 + 2*x + 3)

theorem monotonic_increasing_interval : ∀ (x : ℝ), 1 ≤ x ∧ x < 3 → monotone_increasing f [1, 3) :=
begin
  intro x,
  intro hx,
  sorry
end

end monotonic_increasing_interval_l728_728027


namespace coin_toss_probability_l728_728918

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l728_728918


namespace factorial_difference_l728_728531

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l728_728531


namespace binomial_value_19_13_l728_728236

theorem binomial_value_19_13 :
  (binomial 18 11 = 31824) →
  (binomial 18 12 = 18564) →
  (binomial 20 13 = 77520) →
  binomial 19 13 = 27132 :=
by
  intros h1 h2 h3
  -- using the proven values and the properties of binomial coefficients
  have h4 : binomial 19 12 = binomial 18 11 + binomial 18 12, from binomial_pascal_left 18 11,
  rw [h1, h2] at h4 -- substituting given values
  have h5 : binomial 19 12 = 50388, by simpa only [],
  have h6 : binomial 19 13 = binomial 20 13 - binomial 19 12, from binomial_pascal_right 19 13,
  rw [h3, h5] at h6,
  exact eq.symm h6

end binomial_value_19_13_l728_728236


namespace prob_three_heads_in_eight_tosses_l728_728955

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l728_728955


namespace maximal_alternating_sum_l728_728428

theorem maximal_alternating_sum :
  ∃ (x : Fin 100 → ℕ), 
    x 0 = 1 ∧ 
    (∀ k : Fin 99, 0 ≤ x (k + 1) ∧ x (k + 1) ≤ 2 * x k) ∧
    (x 0 - x 1 + x 2 - x 3 + x 4 - x 5 + ... + x 98 - x 99 = 2^99 - 1) := sorry

end maximal_alternating_sum_l728_728428


namespace distance_from_center_of_sphere_to_plane_of_triangle_l728_728409

theorem distance_from_center_of_sphere_to_plane_of_triangle
  (A B C O : Type)
  [metric_space O] [has_dist O]
  (r : ℝ) (AB BC CA : ℝ)
  (h1 : dist O A = r) 
  (h2 : dist O B = r) 
  (h3 : dist O C = r)
  (h4 : dist A B = AB)
  (h5 : dist B C = BC)
  (h6 : dist C A = CA)
  (h7 : r = 30)
  (h8 : AB = 30)
  (h9 : BC = 40)
  (h10 : CA = 50) :
  ∃ p q r : ℕ, p.gcd r = 1 ∧ ¬ (∃ n : ℕ, q = n^2) ∧ 5 * √(11 : ℕ) = (p * (√(q : ℕ))) / (r : ℕ)  ∧ p + q + r = 17 :=
by apply sorry

end distance_from_center_of_sphere_to_plane_of_triangle_l728_728409


namespace largest_number_among_options_l728_728133

def option_a : ℝ := -abs (-4)
def option_b : ℝ := 0
def option_c : ℝ := 1
def option_d : ℝ := -( -3)

theorem largest_number_among_options : 
  max (max option_a (max option_b option_c)) option_d = option_d := by
  sorry

end largest_number_among_options_l728_728133


namespace not_always_true_inequality_l728_728282

variable {x y z : ℝ} {k : ℤ}

theorem not_always_true_inequality :
  x > 0 → y > 0 → x > y → z ≠ 0 → k ≠ 0 → ¬ ( ∀ z, (x / (z^k) > y / (z^k)) ) :=
by
  intro hx hy hxy hz hk
  sorry

end not_always_true_inequality_l728_728282


namespace infinite_solutions_eq_one_l728_728226

theorem infinite_solutions_eq_one (a : ℝ) :
  (∃ᶠ x in filter.at_top, abs (x - 2) = a * x - 2) →
  a = 1 :=
by
  sorry

end infinite_solutions_eq_one_l728_728226


namespace income_before_taxes_l728_728501

noncomputable def tax_brackets (income : ℝ) : ℝ :=
  if income > 20000 then 0.15 * (income - 20000) + 0.1 * (20000 - 10000) + 0.05 * (10000 - 3000)
  else if income > 10000 then 0.10 * (income - 10000) + 0.05 * (10000 - 3000)
  else if income > 3000 then 0.05 * (income - 3000)
  else 0

theorem income_before_taxes (net_income : ℝ) (tax_deduction : ℝ) (tax_credit : ℝ) : 
  net_income = 25000 ∧ tax_deduction = 3000 ∧ tax_credit = 1000 →
  let total_taxes := tax_brackets 30100 in
  (net_income + total_taxes + tax_deduction = 30100) :=
by
  intros h
  obtain ⟨hne, htd, htc⟩ := h
  sorry

end income_before_taxes_l728_728501


namespace probability_three_heads_in_eight_tosses_l728_728994

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l728_728994


namespace geometric_sum_S12_l728_728795

theorem geometric_sum_S12 
  (S : ℕ → ℝ)
  (h_S4 : S 4 = 2) 
  (h_S8 : S 8 = 6) 
  (geom_property : ∀ n, (S (2 * n + 4) - S n) ^ 2 = S n * (S (3 * n + 4) - S (2 * n + 4))) 
  : S 12 = 14 := 
by sorry

end geometric_sum_S12_l728_728795


namespace possible_to_fill_4x4_grid_l728_728765

theorem possible_to_fill_4x4_grid :
  ∃ (a : ℕ → ℕ → ℤ), 
    (∀ i j, 1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4 → true) ∧
    (finset.sum (finset.range 4) (λ i, finset.sum (finset.range 4) (λ j, a i j)) > 0) ∧
    (∀ i j, 1 ≤ i ∧ i ≤ 2 ∧ 1 ≤ j ∧ j ≤ 2 
      → finset.sum (finset.interval (i-1) (i+1)) (λ p, finset.sum (finset.interval (j-1) (j+1)) (λ q, if 1 ≤ p ∧ p ≤ 4 ∧ 1 ≤ q ∧ q ≤ 4 then a p q else 0)) < 0) :=
sorry

end possible_to_fill_4x4_grid_l728_728765


namespace smallest_two_digit_prime_reversed_composite_l728_728605

def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n

theorem smallest_two_digit_prime_reversed_composite : ∃ (p : ℕ), p = 19 ∧ p.prime ∧ 10 ≤ p ∧ p < 100 ∧ is_composite (p % 10 * 10 + p / 10) := 
by {
  sorry
}

end smallest_two_digit_prime_reversed_composite_l728_728605


namespace binary_to_decimal_and_septal_l728_728574

theorem binary_to_decimal_and_septal :
  let bin : ℕ := 110101
  let dec : ℕ := 53
  let septal : ℕ := 104
  let convert_to_decimal (b : ℕ) : ℕ := 
    (b % 10) * 2^0 + ((b / 10) % 10) * 2^1 + ((b / 100) % 10) * 2^2 + 
    ((b / 1000) % 10) * 2^3 + ((b / 10000) % 10) * 2^4 + ((b / 100000) % 10) * 2^5
  let convert_to_septal (n : ℕ) : ℕ :=
    let rec aux (n : ℕ) (acc : ℕ) (place : ℕ) : ℕ :=
      if n = 0 then acc
      else aux (n / 7) (acc + (n % 7) * place) (place * 10)
    aux n 0 1
  convert_to_decimal bin = dec ∧ convert_to_septal dec = septal :=
by
  sorry

end binary_to_decimal_and_septal_l728_728574


namespace movie_store_additional_movie_needed_l728_728120

theorem movie_store_additional_movie_needed (movies shelves : ℕ) (h_movies : movies = 999) (h_shelves : shelves = 5) : 
  (shelves - (movies % shelves)) % shelves = 1 :=
by
  sorry

end movie_store_additional_movie_needed_l728_728120


namespace arithmetic_sequence_sum_S9_l728_728628

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable {S : ℕ → ℝ} -- Define the sum sequence

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = n * (a 1 + a n) / 2

-- Problem statement in Lean
theorem arithmetic_sequence_sum_S9 (h_seq : ∃ d, arithmetic_sequence a d) (h_a2 : a 2 = -2) (h_a8 : a 8 = 6) (h_S_def : sum_of_first_n_terms a S) : S 9 = 18 := 
by {
  sorry
}

end arithmetic_sequence_sum_S9_l728_728628


namespace power_induction_equivalence_l728_728773

-- Defining that f is a function from integers to integers.
def f (a : ℤ) : ℤ := sorry

-- Main statement translated to Lean 4.
theorem power_induction_equivalence
  (f_def : ∀ a b : ℤ, a ≠ 0 → b ≠ 0 → f(a * b) ≥ f(a) + f(b)) :
  ∀ a : ℤ, a ≠ 0 →
  (∀ n : ℕ, f(a^n) = n * f(a)) ↔ (f(a^2) = 2 * f(a)) := 
  by
    sorry

end power_induction_equivalence_l728_728773
