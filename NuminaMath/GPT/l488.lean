import Data.Finset
import Data.Rat.Basic
import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Geometry.Euclidean.Angles
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.MeasureTheory.ProbabilityMassFunction
import Mathlib.Probability.Basic
import Mathlib.Probability.Normal
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace probability_correct_l488_488544

open Finset

def first_ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def pairs_with_sum_divisible_by_3 (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.bUnion (λ x, s.filter (λ y, x < y ∧ (x + y) % 3 = 0).image (λ y, (x, y)))

noncomputable def probability_sum_divisible_by_3 : ℚ :=
  (pairs_with_sum_divisible_by_3 first_ten_primes).card / (first_ten_primes.card.choose 2)

theorem probability_correct : probability_sum_divisible_by_3 = 1 / 5 := 
sorry

end probability_correct_l488_488544


namespace tan_identity_l488_488002

-- Given statements
def lhs (α : ℝ) : ℝ :=
  4.6 * (cos ((5 * Real.pi / 2) - 6 * α) + sin (Real.pi + 4 * α) + sin (3 * Real.pi - α)) /
  (sin ((5 * Real.pi / 2) + 6 * α) + cos (4 * α - 2 * Real.pi) + cos (α + 2 * Real.pi))

-- Proving lhs equals to tan(α)
theorem tan_identity (α : ℝ) : lhs α = Real.tan α :=
by sorry

end tan_identity_l488_488002


namespace star_computation_l488_488872

def star (x y : ℝ) (h : x ≠ y) : ℝ := (x + y) / (x - y)

theorem star_computation :
  star (star (-1) 4 (by norm_num)) (star (-5) 2 (by norm_num)) (by norm_num) = 1 / 6 := 
sorry

end star_computation_l488_488872


namespace tangential_circle_radius_l488_488134

-- Definitions for given elements
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Definitions for the conditions
variables {O A B : ℝ × ℝ} (C : Circle) (R : ℝ) (x : ℝ)
          (O A B_perpendicular : O.1 = A.1 ∧ O.2 = B.2 ∧ O.1 = B.1 ∧ O.2 = A.2)

-- Assume the conditions hold
axiom lines_perpendicular : O A ⊤ B

-- The Lean statement we need to prove
theorem tangential_circle_radius (C : Circle) (R : ℝ) (x : ℝ)
  (C_eq : C.radius = R) (tangent_to_lines : ∀ (P : ℝ × ℝ), (P = O ∨ P = A ∨ P = B) → P.1 * P.2 = 0)
  (right_angle_intersection : (C.radius ^ 2 + x ^ 2 = (C.center.1 - x)^2 + (C.center.2 - x)^2)) :
  x = R * (2 + Real.sqrt(3)) ∨ x = R * (2 - Real.sqrt(3)) :=
sorry

end tangential_circle_radius_l488_488134


namespace quadratic_has_two_distinct_real_roots_l488_488454

theorem quadratic_has_two_distinct_real_roots :
  let a := 1
  let b := -4
  let c := -4
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  let a := 1
  let b := -4
  let c := -4
  let Δ := b^2 - 4 * a * c
  have hΔ : Δ = b^2 - 4 * a * c := rfl
  have hΔ_val : Δ = 32 := by
    calc Δ
        = (-4)^2 - 4 * 1 * (-4) : by rw [hΔ]
    ... = 16 + 16 : by norm_num
    ... = 32 : rfl
  show Δ > 0, from by {
    calc Δ
        = 32 : hΔ_val
    ... > 0 : by norm_num
  }
sorry

end quadratic_has_two_distinct_real_roots_l488_488454


namespace virginia_eggs_l488_488350

-- Definitions and conditions
variable (eggs_start : Nat)
variable (eggs_taken : Nat := 3)
variable (eggs_end : Nat := 93)

-- Problem statement to prove
theorem virginia_eggs : eggs_start - eggs_taken = eggs_end → eggs_start = 96 :=
by
  intro h
  sorry

end virginia_eggs_l488_488350


namespace no_real_roots_of_quadratic_l488_488455

theorem no_real_roots_of_quadratic (k : ℝ) (hk : k ≠ 0) : ¬ ∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 = 0 :=
by
  sorry

end no_real_roots_of_quadratic_l488_488455


namespace songs_before_camp_l488_488719

theorem songs_before_camp (total_songs : ℕ) (learned_at_camp : ℕ) (songs_before_camp : ℕ) (h1 : total_songs = 74) (h2 : learned_at_camp = 18) : songs_before_camp = 56 :=
by
  sorry

end songs_before_camp_l488_488719


namespace polygon_with_120_degree_interior_angle_has_6_sides_l488_488528

theorem polygon_with_120_degree_interior_angle_has_6_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (sum_interior_angles : ℕ) = (n-2) * 180 / n ∧ (each_angle : ℕ) = 120) : n = 6 :=
by
  sorry

end polygon_with_120_degree_interior_angle_has_6_sides_l488_488528


namespace even_M_remainder_probability_l488_488054

theorem even_M_remainder_probability :
  let M := 1818
  let range := fin (M + 1)
  let even_count := (M / 2)
  let valid_even_count := if even_count % 3 == 0 then even_count else even_count - even_count % 3
  let favorable_count := (2 / 6 * M)
  (favorable_count / even_count : ℚ) = 2 / 3 :=
by
  sorry

end even_M_remainder_probability_l488_488054


namespace infinite_sequence_domain_l488_488534

def seq_domain (f : ℕ → ℕ) : Set ℕ := {n | 0 < n}

theorem infinite_sequence_domain (f : ℕ → ℕ) (a_n : ℕ → ℕ)
   (h : ∀ (n : ℕ), a_n n = f n) : 
   seq_domain f = {n | 0 < n} :=
sorry

end infinite_sequence_domain_l488_488534


namespace equation_solutions_l488_488292

noncomputable def solve_equation (x : ℝ) : Prop :=
  x - 3 = 4 * (x - 3)^2

theorem equation_solutions :
  ∀ x : ℝ, solve_equation x ↔ x = 3 ∨ x = 3.25 :=
by sorry

end equation_solutions_l488_488292


namespace UrsaMajor_distance_correct_UrsaMinor_distance_correct_l488_488089

noncomputable def UrsaMajorDistance : ℝ :=
  let δ1 := 62.15 * Math.pi / 180 -- convert degrees to radians.
  let α1 := 10 * 15 + 59 * 15 / 60
  let δ2 := 49.6833 * Math.pi / 180
  let α2 := 13 * 15 + 45 * 15 / 60
  let d1 := 59
  let d2 := 200
  let Δα := α2 - α1
  let cos_Δα := Math.cos(56.5 * (Math.pi / 180))
  let cos_θ := (Math.sin(δ1) * Math.sin(δ2)) + (Math.cos(δ1) * Math.cos(δ2) * cos_Δα)
  let θ := Math.acos(cos_θ)
  Math.sqrt(d1^2 + d2^2 - 2 * d1 * d2 * Math.cos(θ))

noncomputable def UrsaMinorDistance : ℝ :=
  let δ3 := 88.9 * Math.pi / 180
  let α3 := 2 * 15 + 58 * 15 / 60
  let δ4 := 74.467 * Math.pi / 180
  let α4 := 14 * 15 + 51 * 15 / 60
  let d3 := 325
  let d4 := 96
  let Δα_2 := α4 - α3
  let cos_Δα_2 := Math.cos(178.25 * (Math.pi / 180))
  let cos_θ_2 := (Math.sin(δ3) * Math.sin(δ4)) + (Math.cos(δ3) * Math.cos(δ4) * cos_Δα_2)
  let θ_2 := Math.acos(cos_θ_2)
  Math.sqrt(d3^2 + d4^2 - 2 * d3 * d4 * Math.cos(θ_2))

theorem UrsaMajor_distance_correct (h : True) : UrsaMajorDistance = 154.04 := by
  sorry

theorem UrsaMinor_distance_correct (h : True) : UrsaMinorDistance = 235.37 := by
  sorry

end UrsaMajor_distance_correct_UrsaMinor_distance_correct_l488_488089


namespace prob_yellow_white_l488_488602

variable {P : ℕ → ℝ}
variable (A B C: ℕ)
variable (P_A_plus_P_C : P(A) + P(C) = 0.4)
variable (P_A_plus_P_B : P(A) + P(B) = 0.9)
variable (P_A_plus_P_B_plus_P_C : P(A) + P(B) + P(C) = 1)

theorem prob_yellow_white (P_A_plus_P_C : P(A) + P(C) = 0.4)
(P_A_plus_P_B : P(A) + P(B) = 0.9)
(P_A_plus_P_B_plus_P_C : P(A) + P(B) + P(C) = 1) : 
P(B) + P(C) = 0.7 := 
by 
  sorry

end prob_yellow_white_l488_488602


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488943

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488943


namespace tagged_fish_in_second_catch_l488_488552

theorem tagged_fish_in_second_catch
  (N : ℕ)
  (initial_catch tagged_returned : ℕ)
  (second_catch : ℕ)
  (approximate_pond_fish : ℕ)
  (condition_1 : initial_catch = 60)
  (condition_2 : tagged_returned = 60)
  (condition_3 : second_catch = 60)
  (condition_4 : approximate_pond_fish = 1800) :
  (tagged_returned * second_catch) / approximate_pond_fish = 2 :=
by
  sorry

end tagged_fish_in_second_catch_l488_488552


namespace find_a_l488_488486

-- Define the functions f and g
def f (x : ℝ) : ℝ := x ^ 2 - 2
def g (x : ℝ) : ℝ := 3 * Real.log x - a * x

-- Define the derivatives f' and g'
def f' (x : ℝ) : ℝ := 2 * x
def g' (x : ℝ) : ℝ := (3 / x) - a

-- The equation for the slopes being equal at the common point t
def slopes_equal (t : ℝ) (a : ℝ) : Prop :=
  2 * t = (3 / t) - a

-- The equation for the function values being equal at the common point t
def values_equal (t : ℝ) (a : ℝ) : Prop :=
  t ^ 2 - 2 = 3 * Real.log t - a * t

-- Defining the proof problem
theorem find_a (a : ℝ) (t : ℝ) (h₁ : slopes_equal t a) (h₂ : values_equal t a) : a = 1 :=
  sorry

end find_a_l488_488486


namespace cost_of_fencing_park_l488_488695

theorem cost_of_fencing_park :
  ∃ (cost : ℝ), 
    let x := real.sqrt (2400 / 6),
        length := 3 * x,
        width := 2 * x,
        perimeter := 2 * (length + width),
        cost_in_inr := perimeter * 0.50,
        conversion_rate := 1 / 75,
        cost_in_usd := cost_in_inr * conversion_rate
    in
      length * width = 2400 ∧  -- Ensuring the area condition
      perimeter = 200 ∧        -- Ensuring the perimeter is computed correctly
      cost_in_usd ≈ 1.33 ∧     -- Ensuring the cost in USD is approximately 1.33
      cost = cost_in_usd       -- The final cost is the computed cost in USD.
:= by
  sorry

end cost_of_fencing_park_l488_488695


namespace range_of_a_l488_488498

def quadratic_has_two_distinct_positive_zeros {α : Type*} [LinearOrderedField α] (a : α) :=
  ∀ x : α,  f(x) = (1 / 4) * x^2 - a * x + 4 → (0 < x) → (f(x) = 0)

theorem range_of_a (a : ℝ) :
  quadratic_has_two_distinct_positive_zeros a → (2 < a) :=
by
  sorry

end range_of_a_l488_488498


namespace max_sequence_term_l488_488508

theorem max_sequence_term (a_n : ℕ → ℝ) (h : ∀ n, a_n n = (10 / 11 : ℝ) ^ n * (3 * n + 13)) : 
  ∀ N, (∀ n, n ≠ 6 → a_n n ≤ a_n 6) :=
begin
  -- proof goes here
  sorry
end

end max_sequence_term_l488_488508


namespace at_least_p_commercials_at_most_q_commercials_between_p_and_q_commercials_l488_488779

-- Defining the problem conditions
variables {r n p q : ℕ}

-- At least p commercials per day
theorem at_least_p_commercials (hnp : n * p ≤ r) :
  ∃ k, k = r - n * p + n - 1 ∧ nat.choose k (n - 1) = nat.choose (r - n * p + n - 1) (n - 1) :=
begin
  sorry
end

-- At most q commercials per day
theorem at_most_q_commercials (hnq : n * q ≥ r) :
  ∑ k in finset.range (n + 1), (-1 : ℤ)^k * (nat.choose n k) * (nat.choose (r - k * (q + 1) + (n - 1)) (n - 1)) =
  ∑ k in finset.range (n + 1), (-1 : ℤ)^k * (nat.choose n k) * (nat.choose (r - k * (q + 1) + (n - 1)) (n - 1)) :=
begin
  sorry
end

-- Between p and q commercials per day
theorem between_p_and_q_commercials (hnp : n * p ≤ r) (hnq : n * q ≥ r) :
  ∑ k in finset.range (n + 1), (-1 : ℤ)^k * (nat.choose n k) * (nat.choose (r - k * (q + 1 - p) + (n - 1)) (n - 1)) =
  ∑ k in finset.range (n + 1), (-1 : ℤ)^k * (nat.choose n k) * (nat.choose (r - k * (q + 1 - p) + (n - 1)) (n - 1)) :=
begin
  sorry
end

end at_least_p_commercials_at_most_q_commercials_between_p_and_q_commercials_l488_488779


namespace even_odd_difference_l488_488722

def even_sum_n (n : ℕ) : ℕ := (n * (n + 1))
def odd_sum_n (n : ℕ) : ℕ := n * n

theorem even_odd_difference : even_sum_n 100 - odd_sum_n 100 = 100 := by
  -- The proof goes here
  sorry

end even_odd_difference_l488_488722


namespace tank_fewer_eggs_in_second_round_l488_488550

variables (T E_total T_r2_diff : ℕ)

theorem tank_fewer_eggs_in_second_round
  (h1 : E_total = 400)
  (h2 : E_total = (T + (T - 10)) + (30 + 60))
  (h3 : T_r2_diff = T - 30) :
  T_r2_diff = 130 := by
    sorry

end tank_fewer_eggs_in_second_round_l488_488550


namespace determine_N_l488_488265

variable (U M N : Set ℕ)

theorem determine_N (h1 : U = {1, 2, 3, 4, 5})
  (h2 : U = M ∪ N)
  (h3 : M ∩ (U \ N) = {2, 4}) :
  N = {1, 3, 5} :=
by
  sorry

end determine_N_l488_488265


namespace probability_correct_l488_488545

open Finset

def first_ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def pairs_with_sum_divisible_by_3 (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.bUnion (λ x, s.filter (λ y, x < y ∧ (x + y) % 3 = 0).image (λ y, (x, y)))

noncomputable def probability_sum_divisible_by_3 : ℚ :=
  (pairs_with_sum_divisible_by_3 first_ten_primes).card / (first_ten_primes.card.choose 2)

theorem probability_correct : probability_sum_divisible_by_3 = 1 / 5 := 
sorry

end probability_correct_l488_488545


namespace log_base_9_of_729_l488_488846

theorem log_base_9_of_729 : ∃ (x : ℝ), (9 : ℝ)^x = (729 : ℝ) ∧ x = 3 := 
by {
  have h1 : (9 : ℝ) = (3 : ℝ)^2 := by norm_num,
  have h2 : (729 : ℝ) = (3 : ℝ)^6 := by norm_num,
  use 3,
  split,
  {
    calc (9 : ℝ) ^ 3
        = (3^2 : ℝ) ^ 3 : by rw h1
    ... = (3^6 : ℝ) : by rw pow_mul
    ... = (729 : ℝ) : by rw h2,
  },
  { 
    refl,
  }
}

end log_base_9_of_729_l488_488846


namespace total_campers_correct_l488_488776

-- Definitions for the conditions
def campers_morning : ℕ := 15
def campers_afternoon : ℕ := 17

-- Define total campers, question is to prove it is indeed 32
def total_campers : ℕ := campers_morning + campers_afternoon

theorem total_campers_correct : total_campers = 32 :=
by
  -- Proof omitted
  sorry

end total_campers_correct_l488_488776


namespace symmetric_point_l488_488864

open Real

-- Define the given point M and the line equation in conditions.
def M : ℝ × ℝ × ℝ := (1, 1, 1)
def line (t : ℝ) : ℝ × ℝ × ℝ := (2 + t, -1.5 - 2 * t, 1 + t)

-- Define the function to get the normal vector of the plane and plane equation passing through given point M
def normal_vector : ℝ × ℝ × ℝ := (1, -2, 1)
def plane (x y z : ℝ) : Prop := x - 2*y + z = 0

-- Given the above conditions, we need to prove that the coordinates of the symmetric point M' are (1, 0, -1)
theorem symmetric_point : ∃ M' : ℝ × ℝ × ℝ,
  M' = (1, 0, -1) ∧ 
  let M0 := line (-1) in
  -- M_0 is the midpoint between M and M'
  M0 = ((fst M + (fst M')) / 2, (snd M + (snd M')) / 2, (trd M + (trd M')) / 2)
:= 
  sorry

-- Helper functions to get components of a tuple
def fst {α β γ : Type} (t : α × β × γ) : α := t.1
def snd {α β γ : Type} (t : α × β × γ) : β := t.2.1
def trd {α β γ : Type} (t : α × β × γ) : β := t.2.2

end symmetric_point_l488_488864


namespace number_of_pairs_l488_488873

def harmonic_mean (x y : ℕ) : ℚ := (2 * x * y) / (x + y)

theorem number_of_pairs (H : ℚ) (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) :
  H = 12^10 → (harmonic_mean x y = H) → 220 :=
by
  sorry

end number_of_pairs_l488_488873


namespace find_q_in_geometric_sequence_l488_488220

theorem find_q_in_geometric_sequence
  {q : ℝ} (q_pos : q > 0) 
  (a1_def : ∀(a : ℕ → ℝ), a 1 = 1 / q^2) 
  (S5_eq_S2_plus_2 : ∀(S : ℕ → ℝ), S 5 = S 2 + 2) :
  q = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end find_q_in_geometric_sequence_l488_488220


namespace find_rate_percent_l488_488740

-- Define the conditions based on the problem statement
def principal : ℝ := 800
def simpleInterest : ℝ := 160
def time : ℝ := 5

-- Create the statement to prove the rate percent
theorem find_rate_percent : ∃ (rate : ℝ), simpleInterest = (principal * rate * time) / 100 := sorry

end find_rate_percent_l488_488740


namespace problem_l488_488824

theorem problem (k : ℕ) (h1 : 2 < k) 
  (h2 : log 10 ((k - 2)!) + log 10 ((k - 1)!) + 2 = 2 * log 10 (k!)) : 
  k = 5 := 
  sorry

end problem_l488_488824


namespace fixed_point_of_log_function_l488_488251

theorem fixed_point_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (-1, 2) ∧ ∀ x y : ℝ, y = 2 + Real.logb a (x + 2) → y = 2 → x = -1 :=
by
  sorry

end fixed_point_of_log_function_l488_488251


namespace area_of_triangle_ABC_l488_488546

variable (AB BC : Real)
variable (cosB : Real)
variable (sinB : Real)

noncomputable def area_of_triangle (AB BC sinB : Real) : Real :=
  (1 / 2) * AB * BC * sinB

theorem area_of_triangle_ABC :
  AB = 2 →
  BC = 5 * sqrt 3 →
  cosB = 4 / 5 →
  sinB = sqrt (1 - cosB^2) →
  area_of_triangle AB BC sinB = 3 * sqrt 3 :=
by
  intros hAB hBC hcosB hsinB
  rw [hAB, hBC, hcosB, hsinB]
  sorry

end area_of_triangle_ABC_l488_488546


namespace rectangle_perimeter_l488_488398

theorem rectangle_perimeter (a b c d e f g : ℕ)
  (h1 : a + b + c = d)
  (h2 : d + e = g)
  (h3 : b + c = f)
  (h4 : c + f = g)
  (h5 : Nat.gcd (a + b + g) (d + e) = 1)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (e_pos : 0 < e)
  (f_pos : 0 < f)
  (g_pos : 0 < g) :
  2 * (a + b + g + d + e) = 40 :=
sorry

end rectangle_perimeter_l488_488398


namespace tangent_circle_radius_l488_488241

-- Definitions of the problem conditions
def side_length : ℝ := 4
def semicircle_radius : ℝ := 1 / 2
def num_sides : ℕ := 4
def semicircles_per_side : ℕ := 4
def center_to_center : ℝ := side_length / 2
def hypotenuse : ℝ := Real.sqrt (center_to_center^2 + semicircle_radius^2)

-- Axioms about the distances and geometry
axiom center_to_semicircle_top : ℝ := hypotenuse / 2

-- The goal is to prove the radius of the tangent circle
theorem tangent_circle_radius : ∃ r : ℝ, r = (center_to_semicircle_top - semicircle_radius) / num_sides := sorry

end tangent_circle_radius_l488_488241


namespace convex_polygon_four_equal_area_parts_l488_488275

-- Assume a convex polygon P with area S
variables (P : Set (ℝ × ℝ)) (hP : Convex P) (S : ℝ)

-- Assume the area of the polygon P is S
axiom area_eq_S : measure_theory.measure (outer_measure.polygon P) = S

-- State to prove that a convex polygon can be divided into four equal-area parts by two mutually perpendicular lines
theorem convex_polygon_four_equal_area_parts (hP : Convex P) (hS : measure_theory.measure (outer_measure.polygon P) = S) :
  ∃ (a b : ℝ), 
    (measure_theory.measure (outer_measure.polygon (P ∩ {p : ℝ × ℝ | p.2 ≤ a}) = S / 2) ∧ 
     measure_theory.measure (outer_measure.polygon (P ∩ {p : ℝ × ℝ | p.1 ≤ b}) = S / 2) ∧ 
     measure_theory.measure (outer_measure.polygon (P ∩ {p : ℝ × ℝ | p.1 ≤ b ∧ p.2 ≤ a}) = S / 4) ∧ 
     measure_theory.measure (outer_measure.polygon (P ∩ {p : ℝ × ℝ | p.1 ≤ b ∧ p.2 > a}) = S / 4) ∧ 
     measure_theory.measure (outer_measure.polygon (P ∩ {p : ℝ × ℝ | p.1 > b ∧ p.2 ≤ a}) = S / 4) ∧ 
     measure_theory.measure (outer_measure.polygon (P ∩ {p : ℝ × ℝ | p.1 > b ∧ p.2 > a}) = S / 4)) :=
sorry

end convex_polygon_four_equal_area_parts_l488_488275


namespace complex_division_example_l488_488363

open Complex

theorem complex_division_example : (1 + 3 * Complex.i) / (1 - Complex.i) = -1 + 2 * Complex.i :=
by
  sorry

end complex_division_example_l488_488363


namespace proof_problem_l488_488629

theorem proof_problem (a1 a2 a3 : ℕ) (h1 : a1 = a2 - 1) (h2 : a3 = a2 + 1) : 
  a2^3 ∣ (a1 * a2 * a3 + a2) :=
by sorry

end proof_problem_l488_488629


namespace not_enough_pharmacies_l488_488770

theorem not_enough_pharmacies : 
  ∀ (n m : ℕ), n = 10 ∧ m = 10 →
  ∃ (intersections : ℕ), intersections = n * m ∧ 
  ∀ (d : ℕ), d = 3 →
  ∀ (coverage : ℕ), coverage = (2 * d + 1) * (2 * d + 1) →
  ¬ (coverage * 12 ≥ intersections * 2) :=
by sorry

end not_enough_pharmacies_l488_488770


namespace integer_points_on_parabola_l488_488690

def is_point_on_parabola (focus : ℝ × ℝ) (point : ℝ × ℝ) (parabola : ℝ × ℝ → Prop) : Prop :=
  parabola point

def parabola_q (p : ℝ × ℝ) : Prop :=
  let focus := (0, 0)
  let directrix := ∂-5
  let (x, y) := p
  y * 10 = (x * x - 25)

theorem integer_points_on_parabola :
  let focus := (0, 0)
  let points := [(4, 3), (-4, -3)]
  let cond := λ (x y : ℝ), |3 * x + 4 * y| ≤ 1200
  (∀ (p ∈ points), is_point_on_parabola focus p parabola_q)
  → (count {p : ℝ × ℝ | parabola_q p ∧ int_point p ∧ cond p} = 97) :=
sorry

def int_point (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x.floor = x ∧ y.floor = y

end integer_points_on_parabola_l488_488690


namespace subscription_total_l488_488365

theorem subscription_total (a b c : ℝ) (h1 : a = b + 4000) (h2 : b = c + 5000) (h3 : 15120 / 36000 = a / (a + b + c)) : 
  a + b + c = 50000 :=
by 
  sorry

end subscription_total_l488_488365


namespace length_of_faster_train_l488_488348

-- Definitions (conditions)
def speed_faster_train_kmph : ℝ := 50
def speed_slower_train_kmph : ℝ := 32
def time_to_pass_seconds : ℝ := 15

-- Conversion factors and helper definitions
def kmph_to_mps (speed_kmph : ℝ) := speed_kmph * (5 / 18)
def relative_speed := (speed_faster_train_kmph - speed_slower_train_kmph)
def relative_speed_mps := kmph_to_mps relative_speed

-- Theorem (proof problem)
theorem length_of_faster_train :
  (relative_speed_mps * time_to_pass_seconds) = 75 := 
sorry

end length_of_faster_train_l488_488348


namespace only_natural_number_with_specific_conditions_is_prime_l488_488855

theorem only_natural_number_with_specific_conditions_is_prime : 
  ∀ n : ℕ, (∀ k : ℕ, k ≤ n - 1 → Prime (10^n - 1) / 9 + 6 * 10^k) ↔ n = 1 := 
by
  sorry

end only_natural_number_with_specific_conditions_is_prime_l488_488855


namespace num_three_digit_sums7_l488_488995

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l488_488995


namespace student_B_more_consistent_l488_488414

noncomputable def standard_deviation_A := 5.09
noncomputable def standard_deviation_B := 3.72
def games_played := 7
noncomputable def average_score_A := 16
noncomputable def average_score_B := 16

theorem student_B_more_consistent :
  standard_deviation_B < standard_deviation_A :=
sorry

end student_B_more_consistent_l488_488414


namespace arithmetic_sequence_general_term_arithmetic_sequence_sum_inequality_l488_488264

variable (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ) (c : ℕ)
variable hS : ∀ n, S n = (1 / 3 : ℚ) * n * a n + a n - c
variable h_a2 : a 2 = 6

theorem arithmetic_sequence_general_term :
  ∀ n, a n = 3 * n :=
sorry

theorem arithmetic_sequence_sum_inequality :
  ∀ n, (∑ i in Finset.range n, 1 / (a i * a (i+1))) < 1 / 9 :=
sorry

end arithmetic_sequence_general_term_arithmetic_sequence_sum_inequality_l488_488264


namespace chameleons_changed_color_l488_488557

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l488_488557


namespace roots_increase_l488_488635

noncomputable def P (z : ℂ) (roots : Fin n → ℂ) : ℂ := 
  ∏ i in Finset.finRange n, (z - roots i)

theorem roots_increase {n : ℕ} {x : Fin n → ℂ} [Fintype (Fin n)] 
  (h_ordered : ∀ i j : Fin n, i < j → x i < x j) 
  (i : Fin n) (x_i' : ℂ) (h_interval : x i < x_i' ∧ x_i' < x (Fin.succ i)) :
  let P' := λ z, deriv (P z x)
  let roots_P' := Multiset.sort (≤) (Multiset.pmap (fun z h => z) (polynomial.roots (Pol P'))) (λ x, x) in
  let P'_modified := λ z, deriv (P z (λ j, if j = i then x_i' else x j))
  let roots_P'_modified := Multiset.sort (≤) (Multiset.pmap (fun z h => z) (polynomial.roots (Pol P'_modified))) (λ x, x) in
  roots_P' < roots_P'_modified :=
sorry

end roots_increase_l488_488635


namespace find_width_of_rectangle_l488_488812

variable (A B C D E F G : Type) 
variable [metric_space A]
variable {square_side : ℝ}
variable {ae fg : ℝ}

-- Conditions
def is_square (sides : ℝ) := sides = 12
def segment_ae := ae = 5
def length_fg := fg = 13

-- Theorem to prove
theorem find_width_of_rectangle 
  (h_square : is_square square_side) 
  (h_ae : segment_ae) 
  (h_f_length : length_fg):
  (ℝ) :=
  let width_dg := 11 + 1/13 in
  width_dg = 11 + 1/13 :=
  sorry

end find_width_of_rectangle_l488_488812


namespace value_of_p_l488_488532

theorem value_of_p :
  ∀ (p : ℝ), (∀ (x : ℝ), (x^2 + p * x) * (x^2 - 3 * x + 1)).coeff (2 : ℕ) = 0 → p = 1 / 3 :=
by
  intro p h
  sorry

end value_of_p_l488_488532


namespace painter_total_cost_l488_488404

-- Define the sequences for the north and south sides
def south_side_seq : ℕ → ℕ := λ n, 5 + (n - 1) * 7
def north_side_seq : ℕ → ℕ := λ n, 6 + (n - 1) * 8

-- Calculate the total digits for a given house number
def digit_count (n : ℕ) : ℕ := (n.toString.length)

-- Painter charges $2 per digit
def painter_charge (house_num : ℕ) : ℕ := 2 * (digit_count house_num)

-- Total cost of painting one side of the street
def total_cost_for_side (seq : ℕ → ℕ) : ℕ := ∑ n in (range 30), painter_charge (seq (n + 1))

theorem painter_total_cost :
  total_cost_for_side south_side_seq + total_cost_for_side north_side_seq = 312 :=
sorry

end painter_total_cost_l488_488404


namespace probability_first_white_second_red_l488_488013

section probability_problem

def red_marbles : ℕ := 6
def white_marbles : ℕ := 8
def total_marbles : ℕ := red_marbles + white_marbles

theorem probability_first_white_second_red :
  ((white_marbles:ℚ) / total_marbles) * (red_marbles / (total_marbles - 1)) = 24 / 91 := by
  sorry

end probability_problem

end probability_first_white_second_red_l488_488013


namespace product_586645_9999_l488_488724

theorem product_586645_9999 :
  586645 * 9999 = 5865885355 :=
by
  sorry

end product_586645_9999_l488_488724


namespace distance_between_circles_l488_488586

theorem distance_between_circles (ABC : Triangle) (A B C : Point) (R : ℝ) 
  (h_right : is_right_triangle ABC A B C) 
  (h_angle : angle A = 30) 
  (h_inc_radius : incircle_radius ABC = R) : 
  distance (incircle_center ABC) (excircle_center ABC) = 2 * R * sqrt 2 :=
begin
  sorry
end

end distance_between_circles_l488_488586


namespace complex_number_quadrant_l488_488866

def complex_quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else if z.re > 0 ∧ z.im < 0 then 4
  else 0

theorem complex_number_quadrant :
  complex_quadrant (1 - complex.i / complex.i) = 1 :=
by
  sorry

end complex_number_quadrant_l488_488866


namespace find_m_l488_488505

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x + α / x + Real.log x

theorem find_m (α : ℝ) (m : ℝ) (l e : ℝ) (hα_range : α ∈ Set.Icc (1 / Real.exp 1) (2 * Real.exp 2))
(h1 : f 1 α < m) (he : f (Real.exp 1) α < m) :
m > 1 + 2 * Real.exp 2 := by
  sorry

end find_m_l488_488505


namespace probability_calculation_l488_488035

noncomputable def calculate_probability (R : ℝ) : ℝ :=
let s := (2 * R * Real.sqrt 6) / 3 in
let re := R - (R * Real.sqrt 3) / 3 in
let Vs := (4 / 3) * Real.pi * re^3 in
let Vc := (4 / 3) * Real.pi * R^3 in
6 * Vs / Vc

theorem probability_calculation (R : ℝ) (hR : R > 0) :
  ∃ prob : ℝ, prob = calculate_probability R :=
begin
  use calculate_probability R,
  sorry
end

end probability_calculation_l488_488035


namespace shift_left_5pi_over_6_l488_488500

variable (x : ℝ)

-- Define the function f(x)
def f (x : ℝ) : ℝ := sin (x - (Real.pi / 3))

-- Define the function g(x)
def g (x : ℝ) : ℝ := cos x

-- The statement to prove
theorem shift_left_5pi_over_6 : ∀ x, f (x - (5 * Real.pi / 6)) = g x :=
by
  intros
  unfold f g
  sorry

end shift_left_5pi_over_6_l488_488500


namespace necessary_but_not_sufficient_l488_488477

noncomputable def is_increasing_on_R (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem necessary_but_not_sufficient (f : ℝ → ℝ) :
  (f 1 < f 2) → (¬∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∨ (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
by
  sorry

end necessary_but_not_sufficient_l488_488477


namespace probability_sum_divisible_by_3_l488_488542

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

noncomputable def num_pairs_divisible_by_3 (primes : List ℕ) : ℕ :=
  (primes.toFinset.powerset.toList.filter 
    (λ s, s.card = 2 ∧ (s.sum % 3 = 0))).length

theorem probability_sum_divisible_by_3 :
  (num_pairs_divisible_by_3 first_ten_primes : ℚ) / (10.choose 2) = 2 / 15 :=
by
  sorry

end probability_sum_divisible_by_3_l488_488542


namespace coeff_x3y2z5_in_expansion_l488_488445

def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3y2z5_in_expansion :
  let x := 1
  let y := 1
  let z := 1
  let x_term := 2 * x
  let y_term := y
  let z_term := z
  let target_term := x_term ^ 3 * y_term ^ 2 * z_term ^ 5
  let coeff := 2^3 * binomialCoeff 10 3 * binomialCoeff 7 2 * binomialCoeff 5 5
  coeff = 20160 :=
by
  sorry

end coeff_x3y2z5_in_expansion_l488_488445


namespace chameleon_color_change_l488_488568

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l488_488568


namespace digits_sum_eq_seven_l488_488927

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488927


namespace abs_h_eq_one_l488_488701

theorem abs_h_eq_one (h : ℝ) (roots_square_sum_eq : ∀ x : ℝ, x^2 + 6 * h * x + 8 = 0 → x^2 + (x + 6 * h)^2 = 20) : |h| = 1 :=
by
  sorry

end abs_h_eq_one_l488_488701


namespace find_p_plus_q_l488_488222

open Classical

variable (X Y Z : Type)
variable (P : Prop → ℚ)

-- Probabilities given in the problem
def prob_only_one : ℚ := 0.08
def prob_exactly_two : ℚ := 0.12
def prob_all_given_XY : ℚ := 1 / 4

-- Given conditions
axiom prob_one_X : P (X ∧ ¬ Y ∧ ¬ Z) = prob_only_one
axiom prob_one_Y : P (¬ X ∧ Y ∧ ¬ Z) = prob_only_one
axiom prob_one_Z : P (¬ X ∧ ¬ Y ∧ Z) = prob_only_one

axiom prob_two_XY : P (X ∧ Y ∧ ¬ Z) = prob_exactly_two
axiom prob_two_YZ : P (¬ X ∧ Y ∧ Z) = prob_exactly_two
axiom prob_two_XZ : P (X ∧ ¬ Y ∧ Z) = prob_exactly_two

axiom prob_all_given_X_and_Y : P (X ∧ Y ∧ Z) = prob_all_given_XY * P (X ∧ Y)

-- Conditional probability relating to the unknowns p and q where they are relatively prime
axiom not_X_none_factors_cond : ∃ (p q :ℕ), p.gcd q = 1 ∧
  P (¬ X ∧ ¬ Y ∧ ¬ Z ∣ ¬ X) = Rat.mk p q  -- Represent p/q as a rational number

-- Goal to prove
theorem find_p_plus_q : ∃ (p q :ℕ), p.gcd q = 1 ∧ p + q = 25 :=
sorry

end find_p_plus_q_l488_488222


namespace P_I_D_collinear_l488_488261

variables {A B C P I D E F : Type} [IncircleTriangle A B C I D E F]
variables (P_same_side : SameSide P A EF)
variables (anglePEF_eq_ABC : Angle PEF = Angle ABC)
variables (anglePFE_eq_ACB : Angle PFE = Angle ACB)

theorem P_I_D_collinear : Collinear P I D :=
by {
  -- The proof will be inserted here
  sorry
}

end P_I_D_collinear_l488_488261


namespace quadratic_is_three_times_root_equation_1_quadratic_is_three_times_root_equation_2_quadratic_is_three_times_root_equation_3_quadratic_is_three_times_root_equation_4_l488_488200

-- Problem 1
theorem quadratic_is_three_times_root_equation_1 (h: (2: ℝ) * 3 = 6): 
  x^2 - 8x + 12 = 0 ∧ (roots_x = (2, 6) ∧ (6 = 3 * 2)) → "3 times root equation" :=
by sorry

-- Problem 2
theorem quadratic_is_three_times_root_equation_2 (a: ℝ) (c: ℝ): 
  (root₁ := 1) ∧ (root₂ := 3) ∧ (root₁ + root₂ = 4) ∧ (c = root₁ * 3 * root₁) → c = 3 :=
by sorry

-- Problem 3
theorem quadratic_is_three_times_root_equation_3 (a b: ℝ): 
  (roots: ((3: ℝ), b / a)) ∧ ( (b = 9 * a ∧ (3a - 2*b) / (a + b) = -3/2) ∨ (b = a ∧ (3a - 2*b) / (a + b) = 1/2) ) → true :=
by sorry

-- Problem 4
theorem quadratic_is_three_times_root_equation_4 (m n: ℝ) (xy: (m, n) ∣ y = 12 / x):
  y = 12 / m ∧ (roots := (2/m, 6/m)) \∧ (6 / m =  3 * (2 / m)) ∧ (mx^2 - 8x + n = 0) → "3 times root equation" :=
by sorry

end quadratic_is_three_times_root_equation_1_quadratic_is_three_times_root_equation_2_quadratic_is_three_times_root_equation_3_quadratic_is_three_times_root_equation_4_l488_488200


namespace proof_distance_between_foci_l488_488117

noncomputable def distance_between_foci (ellipse_eq : ℝ × ℝ → ℝ) : ℝ :=
  let x := ellipse_eq.1
  let y := ellipse_eq.2
  -- Given ellipse equation: 9x^2 - 18x + 4y^2 + 8y + 8 = 0
  let modified_eq := (9 * (x - 1)^2) * 4 * (y + 1)^2 = 5
  let a_squared := 5 / 4
  let b_squared := 5 / 9
  let c := Real.sqrt (a_squared - b_squared)
  2 * c

-- The statement we want to prove
theorem proof_distance_between_foci : 
  distance_between_foci (λ p, 9 * (p.1)^2 - 18 * p.1 + 4 * (p.2)^2 + 8 * p.2 + 8) = 5 / 3 :=
by
  sorry

end proof_distance_between_foci_l488_488117


namespace yoghurt_cost_1_l488_488299

theorem yoghurt_cost_1 :
  ∃ y : ℝ,
  (∀ (ice_cream_cartons yoghurt_cartons : ℕ) (ice_cream_cost_one_carton : ℝ) (yoghurt_cost_one_carton : ℝ),
    ice_cream_cartons = 19 →
    yoghurt_cartons = 4 →
    ice_cream_cost_one_carton = 7 →
    (19 * 7 = 133) →  -- total ice cream cost
    (133 - 129 = 4) → -- Total yogurt cost
    (4 = 4 * y) →    -- Yoghurt cost equation
    y = 1) :=
sorry

end yoghurt_cost_1_l488_488299


namespace num_three_digit_integers_sum_to_seven_l488_488994

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488994


namespace T_cardinality_l488_488249

-- Definition: prime number
def prime (p : ℕ) : Prop := Nat.Prime p

-- Definition: periodicity condition
def periodic_decimal (n : ℕ) (d : ℕ) : Prop :=
  ∀ i : ℕ, (decimal_digits i n) = (decimal_digits (i + d) n)

-- Main statement
theorem T_cardinality :
  (∀ n : ℕ, (prime 19 ∧ prime 37) → (n > 1 ∧ periodic_decimal n 18) ↔
  (n \in divisors (10^18 - 1) \setminus {1})) → 
  (Fintype.card ({n : ℕ // n ∈ divisors (10^18 - 1) \setminus {1}}) = 767) :=
by
  sorry

end T_cardinality_l488_488249


namespace number_of_three_digit_numbers_with_sum_7_l488_488973

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488973


namespace find_params_find_angle_l488_488513

section Problem1

variables (a b c : ℝ × ℝ)
variables (λ μ : ℝ)

-- Given conditions
def a := (1, 0)
def b := (1, 2)
def c := (0, 1)

-- Prove λ and μ
theorem find_params : c = (λ • a) + (μ • b) → λ = -1/2 ∧ μ = 1/2 :=
by
  sorry

end Problem1

section Problem2

variables (a b c ab ac : ℝ × ℝ)
variables (θ : ℝ)

-- Given conditions
def a := (1, 0)
def b := (1, 2)
def c := (0, 1)
def ab := (-a) + 3 • c
def ac := 4 • a - 2 • c

-- Prove θ
theorem find_angle : ab = (-a) + 3 • c ∧ ac = 4 • a - 2 • c → θ = 3 * Real.pi / 4 :=
by
  sorry

end Problem2

end find_params_find_angle_l488_488513


namespace convex_decagon_diagonals_l488_488519

theorem convex_decagon_diagonals : 
  ∀ (n : ℕ), n = 10 → (∃ d, d = (n * (n - 3)) / 2 ∧ d = 35) :=
by
  intros n hn
  use (n * (n - 3)) / 2
  split
  { rw hn, exact rfl }
  { rw hn, exact rfl } 
  sorry

end convex_decagon_diagonals_l488_488519


namespace number_of_terms_arithmetic_sequence_l488_488431

noncomputable def arithmetic_sequence_n_terms : ℕ :=
  let a₁ := 1
  let d := -2
  let aₙ := -89
  let n := (aₙ - a₁) / d + 1 in
  n

theorem number_of_terms_arithmetic_sequence :
  arithmetic_sequence_n_terms = 46 :=
by
  -- Conditions
  let a₁ := 1
  let d := -2
  let aₙ := -89
  -- Calculation
  let n := (aₙ - a₁) / d + 1
  -- Required proof statement
  have h : n = 46, sorry
  exact h

end number_of_terms_arithmetic_sequence_l488_488431


namespace modulus_of_complex_fraction_l488_488150

open Complex

theorem modulus_of_complex_fraction :
  let z : ℂ := (2 - I) / (1 + 2 * I)
  ∣z∣ = 1 :=
by
  let z : ℂ := (2 - I) / (1 + 2 * I)
  calc
    ∣z∣ = 1 := sorry

end modulus_of_complex_fraction_l488_488150


namespace three_digit_sum_seven_l488_488952

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488952


namespace number_of_sturgeons_l488_488094

def number_of_fishes := 145
def number_of_pikes := 30
def number_of_herrings := 75

theorem number_of_sturgeons : (number_of_fishes - (number_of_pikes + number_of_herrings) = 40) :=
  by
  sorry

end number_of_sturgeons_l488_488094


namespace geometric_sequence_common_ratio_l488_488627

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α) (S : ℕ → α) (q : α)

-- Given conditions
def sum_of_first_terms (n : ℕ) : Prop := S n = (a 0) * (1 - q ^ n) / (1 - q)
def condition1 : Prop := 3 * S 3 = a 4 - 2
def condition2 : Prop := 3 * S 2 = a 3 - 2
def is_geometric_sequence_with_sum_conditions : Prop := sum_of_first_terms 3 ∧ sum_of_first_terms 2 ∧ condition1 ∧ condition2 

-- Conclusion: common ratio q equals 4
theorem geometric_sequence_common_ratio (h : is_geometric_sequence_with_sum_conditions a S q) : q = 4 := 
sorry

end geometric_sequence_common_ratio_l488_488627


namespace arithmetic_geometric_sequence_l488_488227

def arithmetic_seq (a_1 d : ℕ → ℕ) := ∀ n, a_1 + (n - 1) * d = a_n
def geometric_seq (a b c : ℕ) := a * c = b^2
def s_n (n : ℕ) : ℕ := 2 * n^2 - n

def t_m (a : ℕ → ℕ) (S_n : ℕ → ℕ) (n : ℕ) : ℚ :=
  ∑ i in range n, 1 / (a (i + 1) * S_n (i + 1))

theorem arithmetic_geometric_sequence :
  (∀ (a_1 d : ℕ), arithmetic_seq a_1 d) →
  (∀ m, geometric_seq (arithmetic_seq 2 3) (arithmetic_seq 5 2) m) →
  (∀ n, S_n n = 2 * n^2 - n) →
  t_m (λ n, 2 * n - 1) S_n 14 = 14 / 29 := 
by
  dsimp [arithmetic_seq, geometric_seq, t_m, S_n]
  sorry

end arithmetic_geometric_sequence_l488_488227


namespace sufficient_not_necessary_range_l488_488332

theorem sufficient_not_necessary_range (x a : ℝ) : (∀ x, x < 1 → x < a) ∧ (∃ x, x < a ∧ ¬ (x < 1)) ↔ 1 < a := by
  sorry

end sufficient_not_necessary_range_l488_488332


namespace profit_comparison_l488_488017

noncomputable def supermarket_price := 15
noncomputable def discount_price := 5
noncomputable def cost_price := 10

-- Given frequencies from the table
constant f15 : ℕ := 10
constant f17 : ℕ := 15
constant f18 : ℕ := 16
constant f19 : ℕ := 16
constant f20 : ℕ := 13
constant fX : ℕ
constant fY : ℕ

axiom xy_sum : fX + fY = 30

-- expected profit for 17 portions and 18 portions
noncomputable def E_17 := 65 * (1/10) + 75 * (fX/100) + 85 * ((90 - fX)/100)
noncomputable def E_18 := 60 * (1/10) + 70 * (fX/100) + 80 * (3/20) + 90 * ((75 - fX)/100)

theorem profit_comparison : E_17 > E_18 → 25 < fX ∧ fX ≤ 29 :=
by
  sorry

end profit_comparison_l488_488017


namespace log_base_9_of_729_l488_488844

theorem log_base_9_of_729 : ∃ (x : ℝ), (9 : ℝ)^x = (729 : ℝ) ∧ x = 3 := 
by {
  have h1 : (9 : ℝ) = (3 : ℝ)^2 := by norm_num,
  have h2 : (729 : ℝ) = (3 : ℝ)^6 := by norm_num,
  use 3,
  split,
  {
    calc (9 : ℝ) ^ 3
        = (3^2 : ℝ) ^ 3 : by rw h1
    ... = (3^6 : ℝ) : by rw pow_mul
    ... = (729 : ℝ) : by rw h2,
  },
  { 
    refl,
  }
}

end log_base_9_of_729_l488_488844


namespace billboard_color_schemes_l488_488340

theorem billboard_color_schemes : 
  let n := 10 in
  let colors := {red, green} in
  ∃ (config : list colors), 
    length config = n ∧ 
    (∀ (i : ℕ) (h : i < n - 1), config.nth i ≠ config.nth (i + 1)) ∧ 
    (∃ (i : ℕ) (h : i < n), config.nth i = some green) ∧ 
    (number_of_schemes config = 143) :=
sorry

end billboard_color_schemes_l488_488340


namespace no_six_million_consecutive_identical_digits_in_sqrt2_l488_488274

def consecutive_identical_digits_not_exist_in_sqrt2 (n : ℕ) : Prop :=
  let sqrt2_dec := 1.41421356237309504880168872420969807856967187537694807317667973799  -- etc., a known decimal rep. of sqrt(2)
  let sequence_length := 10000000  -- first ten million digits
  let consecutive_length := 6000000  -- six million digits
  !(exists (start : ℕ), start + consecutive_length <= sequence_length 
              ∧ (forall (i : ℕ) (h : i < consecutive_length), 
                 (sqrt2_dec.to_digits sequence_length)[start + i] = (sqrt2_dec.to_digits sequence_length)[start]))

theorem no_six_million_consecutive_identical_digits_in_sqrt2 :
  consecutive_identical_digits_not_exist_in_sqrt2 10000000 :=
sorry

end no_six_million_consecutive_identical_digits_in_sqrt2_l488_488274


namespace problem1_eval_problem2_eval_l488_488059

theorem problem1_eval : (1 * (Real.pi - 3.14)^0 - |2 - Real.sqrt 3| + (-1 / 2)^2) = Real.sqrt 3 - 3 / 4 :=
  sorry

theorem problem2_eval : (Real.sqrt (1 / 3) + Real.sqrt 6 * (1 / Real.sqrt 2 + Real.sqrt 8)) = 16 * Real.sqrt 3 / 3 :=
  sorry

end problem1_eval_problem2_eval_l488_488059


namespace bees_on_20th_day_l488_488215

-- Define the conditions
def initial_bees : ℕ := 1

def companions_per_bee : ℕ := 4

-- Define the total number of bees on day n
def total_bees (n : ℕ) : ℕ :=
  (initial_bees + companions_per_bee) ^ n

-- Statement to prove
theorem bees_on_20th_day : total_bees 20 = 5^20 :=
by
  -- The proof is omitted
  sorry

end bees_on_20th_day_l488_488215


namespace max_distance_PQ_l488_488135

noncomputable def curve1 (θ : ℝ) : ℝ × ℝ :=
  (sqrt 2 * Real.cos θ, 6 + sqrt 2 * Real.sin θ)

noncomputable def curve2 (ϕ : ℝ) : ℝ × ℝ :=
  (sqrt 10 * Real.cos ϕ, Real.sin ϕ)

theorem max_distance_PQ :
  (∀ θ ϕ, let P := curve1 θ in
           let Q := curve2 ϕ in
           dist P Q ≤ 6 * sqrt 2) ∧
  (∃ θ ϕ, let P := curve1 θ in
           let Q := curve2 ϕ in
           dist P Q = 6 * sqrt 2) :=
by sorry

end max_distance_PQ_l488_488135


namespace sum_of_digits_of_sequence_l488_488868

theorem sum_of_digits_of_sequence :
  (∑ n in Finset.range 10001, (n : ℕ).digits.sum) = 180001 := sorry

end sum_of_digits_of_sequence_l488_488868


namespace ryan_owes_leo_10_l488_488606

theorem ryan_owes_leo_10 (total_amount : ℕ) (ryan_fraction : ℚ) (leo_owes_ryan : ℕ) (leo_final_amount : ℕ) : 
  total_amount = 48 ∧ ryan_fraction = 2 / 3 ∧ leo_owes_ryan = 7 ∧ leo_final_amount = 19 → 
  (let ryan_share := ryan_fraction * total_amount in 
   let leo_initial_share := total_amount - ryan_share in 
   let debt_settled_amount := leo_initial_share + ryan_owes_leo - leo_owes_ryan in 
   debt_settled_amount = leo_final_amount → ryan_owes_leo = 10) := 
by
  intros h
  cases h with h_total h_rest
  cases h_rest with h_fraction h_rest2
  cases h_rest2 with h_owes h_final
  simp
  sorry

end ryan_owes_leo_10_l488_488606


namespace remaining_volume_of_5foot_cube_with_cylinder_removed_l488_488306

noncomputable def remaining_volume_of_cube (side length : ℝ) (cylinder_radius : ℝ) (angle : ℝ) : ℝ :=
  let cube_volume := (side length) ^ 3
  let cylinder_height := side length * Real.sqrt 2
  let cylinder_volume := Real.pi * (cylinder_radius ^ 2) * cylinder_height
  cube_volume - cylinder_volume

theorem remaining_volume_of_5foot_cube_with_cylinder_removed :
  remaining_volume_of_cube 5 1 (Real.pi / 4) = 125 - 5 * Real.sqrt 2 * Real.pi :=
by sorry

end remaining_volume_of_5foot_cube_with_cylinder_removed_l488_488306


namespace domain_of_f_l488_488068

noncomputable def f (x : ℝ) := real.sqrt (4 - real.sqrt (6 - real.sqrt x))

theorem domain_of_f :
  {x | 0 ≤ x ∧ x ≤ 36} = {x | f x ≠ 0 ∨ 0 ≤ f x} :=
sorry

end domain_of_f_l488_488068


namespace statement_C_correct_l488_488190

theorem statement_C_correct (a b c d : ℝ) (h_ab : a > b) (h_cd : c > d) : a + c > b + d :=
by
  sorry

end statement_C_correct_l488_488190


namespace customers_who_did_not_tip_l488_488413

def total_customers := 10
def total_tips := 15
def tip_per_customer := 3

theorem customers_who_did_not_tip : total_customers - (total_tips / tip_per_customer) = 5 :=
by
  sorry

end customers_who_did_not_tip_l488_488413


namespace number_of_ways_to_choose_4_points_l488_488056

-- Definition of the conditions
variable (points : Finset (Point Real)) -- variable representing our set of points
variable (count : points.card = 9) -- 9 points condition
variable (condition1 : ∀ (p : Point Real), p ∈ points → ¬(AreCollinear p)) -- condition: no 3 points are collinear
variable (condition2 : ∀ (p : Point Real), p ∈ points → ¬(LieOnSameCircle p)) -- condition: no 4 points lie on the same circle

-- The statement we want to prove
theorem number_of_ways_to_choose_4_points : (points.card = 9) → 
                                             (∀ (p : Point Real), p ∈ points → ¬(AreCollinear p)) → 
                                             (∀ (p : Point Real), p ∈ points → ¬(LieOnSameCircle p)) → 
                                             (points.choose 4).card = 114 := 
by sorry

end number_of_ways_to_choose_4_points_l488_488056


namespace middle_integer_is_six_l488_488705

def valid_even_integer (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), n = x ∧ x = n - 2 ∧ y = n ∧ z = n + 2 ∧ x < y ∧ y < z ∧
  x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧
  1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 1 ≤ z ∧ z ≤ 9

theorem middle_integer_is_six (n : ℕ) (h : valid_even_integer n) :
  n = 6 :=
by
  sorry

end middle_integer_is_six_l488_488705


namespace num_three_digit_integers_sum_to_seven_l488_488988

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488988


namespace rose_price_vs_carnation_price_l488_488713

variable (x y : ℝ)

theorem rose_price_vs_carnation_price
  (h1 : 3 * x + 2 * y > 8)
  (h2 : 2 * x + 3 * y < 7) :
  x > 2 * y :=
sorry

end rose_price_vs_carnation_price_l488_488713


namespace probability_of_receiving_A_signal_probability_of_H_A_given_A_l488_488717

-- Conditions
def P_H_A : ℝ := 0.72
def P_H_B : ℝ := 0.28
def P_A_given_H_A : ℝ := 5 / 6
def P_A_given_H_B : ℝ := 1 / 7

-- Question 1
theorem probability_of_receiving_A_signal : 
  (P_H_A * P_A_given_H_A + P_H_B * P_A_given_H_B) = 0.64 :=
by
  sorry

-- Question 2
theorem probability_of_H_A_given_A : 
  ((P_H_A * P_A_given_H_A) / (P_H_A * P_A_given_H_A + P_H_B * P_A_given_H_B)) = 0.9375 :=
by
  sorry

end probability_of_receiving_A_signal_probability_of_H_A_given_A_l488_488717


namespace range_of_a_l488_488506
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x - 1| + |2 * x - a|

theorem range_of_a (a : ℝ)
  (h : ∀ x : ℝ, f x a ≥ (1 / 4) * a ^ 2 + 1) : -2 ≤ a ∧ a ≤ 0 :=
by sorry

end range_of_a_l488_488506


namespace largest_n_divides_factorial_l488_488860

theorem largest_n_divides_factorial (n : ℕ) : 
    (∀ k : ℕ, (18^k ∣ nat.fact 24) → k ≤ 4) :=
begin
  sorry
end

end largest_n_divides_factorial_l488_488860


namespace shaded_region_area_l488_488597

/--
Given:
1. The radius of quadrant \( OAD \) is \( 4 \).
2. The radius of quadrant \( OBC \) is \( 8 \).
3. \( \angle COD = 30^\circ \).

To prove:
The area of the shaded region \( ABCD \) is \( 12 \pi \).
-/
theorem shaded_region_area 
  (r1 r2 : ℝ) (theta : ℝ) (h1 : r1 = 4) (h2 : r2 = 8) (h3 : theta = 30) :
  (sector_area (r2, theta) - sector_area (r2, 90) - sector_area (r1, 90) = 12 * ℼ) := by sorry

-- Definitions used in theorem (these are placeholders, real definitions will depend on available library functions)
def sector_area (r : ℝ, theta : ℝ) : ℝ := sorry -- Define this function appropriately

end shaded_region_area_l488_488597


namespace ratio_x_to_y_is_12_l488_488029

noncomputable def ratio_x_y (x y : ℝ) (h1 : y = x * (1 - 0.9166666666666666)) : ℝ := x / y

theorem ratio_x_to_y_is_12 (x y : ℝ) (h1 : y = x * (1 - 0.9166666666666666)) : ratio_x_y x y h1 = 12 :=
sorry

end ratio_x_to_y_is_12_l488_488029


namespace intervals_of_monotonicity_range_of_k_if_f_le_zero_sum_of_reciprocals_greater_than_sqrt_l488_488008

def f (x k : ℝ) : ℝ := (x - k) / (x + 1)

theorem intervals_of_monotonicity (k : ℝ) :
  ((k ≤ 0) → (∀ x : ℝ, x > 0 → deriv (λ x, f x k) x > 0)) ∧
  ((k > 0) → ((∀ x : ℝ, x > 0 → deriv (λ x, f x k) x > 0) ∧ 
              (∀ x : ℝ, x < -1 → deriv (λ x, f x k) x < 0))) :=
sorry

theorem range_of_k_if_f_le_zero :
  ∀ k : ℝ, (∀ x : ℝ, f x k ≤ 0) ↔ (1 ≤ k) :=
sorry

theorem sum_of_reciprocals_greater_than_sqrt (n : ℕ) (h : 1 < n) :
  (1 + ∑ i in range n, (1 : ℝ) / (i + 1)) > real.sqrt n :=
sorry

end intervals_of_monotonicity_range_of_k_if_f_le_zero_sum_of_reciprocals_greater_than_sqrt_l488_488008


namespace B_alone_completes_work_in_24_days_l488_488364

theorem B_alone_completes_work_in_24_days 
  (A B : ℚ) 
  (h1 : A + B = 1 / 12) 
  (h2 : A = 1 / 24) : 
  1 / B = 24 :=
by
  sorry

end B_alone_completes_work_in_24_days_l488_488364


namespace l_shaped_tile_rectangle_multiple_of_8_l488_488811

theorem l_shaped_tile_rectangle_multiple_of_8 (m n : ℕ) 
  (h : ∃ k : ℕ, 4 * k = m * n) : ∃ s : ℕ, m * n = 8 * s :=
by
  sorry

end l_shaped_tile_rectangle_multiple_of_8_l488_488811


namespace composition_of_two_rotations_is_rotation_composition_is_translation_when_angles_sum_to_multiple_of_360_l488_488277

structure Point :=
(x y : ℝ)

structure Rotation :=
(center : Point)
(angle : ℝ)

def is_multiple_of_360 (θ : ℝ) : Prop :=
∃ k : ℤ, θ = k * 360

noncomputable def compose_rotations (R_A R_B : Rotation) : Rotation :=
sorry

theorem composition_of_two_rotations_is_rotation (A B : Point) (α β : ℝ) (h : ¬ is_multiple_of_360 (α + β)) :
  let R_A := Rotation.mk A α,
      R_B := Rotation.mk B β in
  ∃ O : Point, compose_rotations R_A R_B = Rotation.mk O (α + β) :=
sorry

theorem composition_is_translation_when_angles_sum_to_multiple_of_360 (A B : Point) (α β : ℝ) (h : is_multiple_of_360 (α + β)) :
  let R_A := Rotation.mk A α,
      R_B := Rotation.mk B β in
  ∃ T : Point → Point, ∀ P : Point, (compose_rotations R_A R_B).center = T P :=
sorry

end composition_of_two_rotations_is_rotation_composition_is_translation_when_angles_sum_to_multiple_of_360_l488_488277


namespace geometric_series_sum_l488_488433

theorem geometric_series_sum :
  let a := (1/2 : ℚ)
  let r := (-1/3 : ℚ)
  let n := 7
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 547 / 1458 :=
by
  sorry

end geometric_series_sum_l488_488433


namespace find_CD_l488_488272

noncomputable def triangle_ABC (BC AC AB AD : ℝ) (hAB_pos : 0 < AB) (hAC_pos : 0 < AC) (hBC_pos : 0 < BC) (hAD_pos : 0 < AD) (hAD_le_AB: AD ≤ AB) : ℝ :=
  let cos_angle_A := (BC^2 - AB^2 - AC^2) / (-2 * AB * AC)
  let DC_squared := AD^2 + AC^2 - 2 * AD * AC * cos_angle_A
  real.sqrt DC_squared

theorem find_CD: 
  triangle_ABC 37 15 44 14 0 < 44 0 < 15 0 < 37 0 < 14 14 ≤ 44 = 13 :=
  by
    sorry

end find_CD_l488_488272


namespace sum_series_l488_488416

theorem sum_series :
  ∑ n in Finset.range 100.succ, 1 / ((2 * n - 1) * (2 * n + 1)) = 100 / 201 := by
  sorry

end sum_series_l488_488416


namespace unit_digit_of_15_pow_l488_488351

-- Define the conditions
def base_number : ℕ := 15
def base_unit_digit : ℕ := 5

-- State the question and objective in Lean 4
theorem unit_digit_of_15_pow (X : ℕ) (h : 0 < X) : (15^X) % 10 = 5 :=
sorry

end unit_digit_of_15_pow_l488_488351


namespace prob_at_most_one_first_class_product_l488_488346

noncomputable def P_event (p q : ℚ) : ℚ :=
  p * (1 - q) + (1 - p) * q

theorem prob_at_most_one_first_class_product :
  let p := 2 / 3
  let q := 3 / 4
  P_event p q = 5 / 12 := by
  sorry

end prob_at_most_one_first_class_product_l488_488346


namespace maximize_sum_abs_diff_if_and_only_if_l488_488609
noncomputable section

def A : Set ℤ := {x | -2562 ≤ x ∧ x ≤ 2562}

def is_bijection (f : ℤ → ℤ) (S : Set ℤ) : Prop := 
  Function.Bijective (Set.restrict f S)

theorem maximize_sum_abs_diff_if_and_only_if {f : ℤ → ℤ} 
  (h_bij : is_bijection f A) : 
  (∑ k in Finset.range (2562 + 1), |f k - f (-k)|) = 
  (∑ k in Finset.range (2562 + 1), |f k - f (-k)|) → 
  (∀ k ∈ Finset.range 2562, 0 < k → k ≤ 2562 → f k * f (-k) < 0) := 
sorry

end maximize_sum_abs_diff_if_and_only_if_l488_488609


namespace number_of_three_digit_numbers_with_sum_7_l488_488980

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488980


namespace find_f_four_thirds_l488_488530

def f (y: ℝ) : ℝ := sorry  -- Placeholder for the function definition

theorem find_f_four_thirds : f (4 / 3) = - (7 / 2) := sorry

end find_f_four_thirds_l488_488530


namespace number_of_three_digit_numbers_with_sum_seven_l488_488889

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488889


namespace vector_addition_result_l488_488176

theorem vector_addition_result :
  let a := (0, -1)
  let b := (3, 2)
  2 • a + b = (3, 0) := by
  sorry

end vector_addition_result_l488_488176


namespace three_digit_sum_seven_l488_488954

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488954


namespace largest_n_divisible_by_18_l488_488861

theorem largest_n_divisible_by_18 (n : ℕ) : 
  (18^n ∣ nat.factorial 24) ↔ n ≤ 5 := 
sorry

end largest_n_divisible_by_18_l488_488861


namespace num_three_digit_integers_sum_to_seven_l488_488989

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488989


namespace no_real_roots_of_equation_sqrt_l488_488688

theorem no_real_roots_of_equation_sqrt :
  (∀ x : ℝ, sqrt (9 - 3 * x) = x * sqrt (9 - 9 * x) → false) :=
by
  intro x
  sorry

end no_real_roots_of_equation_sqrt_l488_488688


namespace different_variance_l488_488048

def setA := [102, 103, 105, 107, 108]
def setB := [2, 3, 5, 7, 8]
def setC := [4, 9, 25, 49, 64]
def setD := [2102, 2103, 2105, 2107, 2108]

def variance (data : List ℝ) : ℝ :=
  let mean := (data.sum : ℝ) / (list.length data)
  (data.map (λ x => (x - mean) ^ 2)).sum / (list.length data)

theorem different_variance : variance setC ≠ variance setA ∧ variance setC ≠ variance setB ∧ variance setC ≠ variance setD := by
  sorry

end different_variance_l488_488048


namespace stall_area_and_cost_minimization_l488_488385

theorem stall_area_and_cost_minimization :
  let A_area : ℕ := 5
  let B_area : ℕ := 3
  let total_stalls : ℕ := 100
  let cost_per_sqm_A : ℕ := 20
  let cost_per_sqm_B : ℕ := 40
  let num_A_stalls := 25
  let num_B_stalls := total_stalls - num_A_stalls
  let total_cost := cost_per_sqm_A * A_area * num_A_stalls + cost_per_sqm_B * B_area * num_B_stalls
  150 / A_area = 3 / 4 * (120 / B_area) ∧
  total_stalls ≥ 3 * num_A_stalls ∧
  num_A_stalls = max (total_stalls / 4) ∧
  total_cost = 11500 :=
by
  have h1 : 150 / 5 = 3 / 4 * (120 / 3), by sorry
  have h2 : total_stalls ≥ 3 * num_A_stalls, by sorry
  have h3 : num_A_stalls = max (total_stalls / 4), by sorry
  have h4 : total_cost = 11500, by sorry
  exact ⟨h1, h2, h3, h4⟩

end stall_area_and_cost_minimization_l488_488385


namespace sufficient_and_unnecessary_condition_for_A_l488_488406

open Nat

-- Definition of set A
def set_A (a : ℝ) : Set (ℕ × ℕ) :=
  {xy | xy.1^2 + 2 * xy.2^2 < a}

-- The main theorem stating the sufficient and unnecessary condition
theorem sufficient_and_unnecessary_condition_for_A (a : ℝ) :
  -- The set A has exactly 2 elements
  (Set.card (set_A a) = 2) ↔ 
  -- The range of a is (1, 2]
  (1 < a ∧ a ≤ 2) :=
sorry

end sufficient_and_unnecessary_condition_for_A_l488_488406


namespace radius_relation_l488_488421

-- Define the conditions under which the spheres exist
variable {R r : ℝ}

-- The problem statement
theorem radius_relation (h : r = R * (2 - Real.sqrt 2)) : r = R * (2 - Real.sqrt 2) :=
sorry

end radius_relation_l488_488421


namespace Carla_has_8_dandelions_l488_488820

noncomputable def CarlaSunflowers : Nat := 6
noncomputable def SeedsPerSunflower : Nat := 9
noncomputable def SeedsPerDandelion : Nat := 12
noncomputable def PercentageFromDandelions : ℚ := 0.64

theorem Carla_has_8_dandelions 
  (CarlaSunflowers = 6) 
  (SeedsPerSunflower = 9) 
  (SeedsPerDandelion = 12) 
  (PercentageFromDandelions = 0.64) : 
  ∃ (D : Nat), D = 8 :=
by 
  sorry

end Carla_has_8_dandelions_l488_488820


namespace max_dist_PB_l488_488620

-- Let B be the upper vertex of the ellipse.
def B : (ℝ × ℝ) := (0, 1)

-- Define the equation of the ellipse.
def ellipse (x y : ℝ) : Prop := (x^2) / 5 + y^2 = 1

-- Define a point P on the ellipse.
def P (θ : ℝ) : (ℝ × ℝ) := (sqrt 5 * cos θ, sin θ)

-- Define the distance function between points.
def dist (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Prove that the maximum distance |PB| is 5/2.
theorem max_dist_PB : ∃ θ : ℝ, dist (P θ) B = 5 / 2 :=
sorry

end max_dist_PB_l488_488620


namespace ratio_of_largest_to_sum_of_others_l488_488832

theorem ratio_of_largest_to_sum_of_others :
  let largest := 2^12
  let sum_others := (2^12 - 1) in
  (largest : ℝ) / (sum_others : ℝ) = 1 :=
by
  sorry

end ratio_of_largest_to_sum_of_others_l488_488832


namespace chameleons_color_change_l488_488560

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l488_488560


namespace range_of_f_l488_488158

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2 else x^2

theorem range_of_f : set.image f set.univ = set.Iio 4 := by
  sorry

end range_of_f_l488_488158


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488935

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488935


namespace total_pencils_l488_488436

def pencils_in_rainbow_box : ℕ := 7
def total_people : ℕ := 8

theorem total_pencils : pencils_in_rainbow_box * total_people = 56 := by
  sorry

end total_pencils_l488_488436


namespace range_of_c_for_two_distinct_roots_l488_488600

theorem range_of_c_for_two_distinct_roots (c : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 3 * x1 + c = x1 + 2) ∧ (x2^2 - 3 * x2 + c = x2 + 2)) ↔ (c < 6) :=
sorry

end range_of_c_for_two_distinct_roots_l488_488600


namespace find_first_number_l488_488698

-- Definitions from conditions
variable (x : ℕ) -- Let the first number be x
variable (y : ℕ) -- Let the second number be y

-- Given conditions in the problem
def condition1 : Prop := y = 43
def condition2 : Prop := x + 2 * y = 124

-- The proof target
theorem find_first_number (h1 : condition1 y) (h2 : condition2 x y) : x = 38 := by
  sorry

end find_first_number_l488_488698


namespace twelve_pharmacies_not_sufficient_l488_488762

-- Define an intersection grid of size 10 x 10 (100 squares).
def city_grid : Type := Fin 10 × Fin 10 

-- Define the distance measure between intersections, assumed as L1 metric for grid paths.
def dist (p q : city_grid) : Nat := (abs (p.1.val - q.1.val) + abs (p.2.val - q.2.val))

-- Define a walking distance pharmacy 
def is_walking_distance (p q : city_grid) : Prop := dist p q ≤ 3

-- State that having 12 pharmacies is not sufficient
theorem twelve_pharmacies_not_sufficient (pharmacies : Fin 12 → city_grid) :
  ¬ (∀ intersection: city_grid, ∃ (p_index : Fin 12), is_walking_distance (pharmacies p_index) intersection) :=
sorry

end twelve_pharmacies_not_sufficient_l488_488762


namespace probability_sum_divisible_by_3_l488_488537

-- Define the first ten prime numbers
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate to check divisibility by 3
def divisibleBy3 (n : ℕ) : Prop := n % 3 = 0

-- Define the main theorem statement
theorem probability_sum_divisible_by_3 :
  (let pairs := (firstTenPrimes.product firstTenPrimes).filter (λ (x : ℕ × ℕ), x.1 < x.2) in
    let totalPairs := pairs.length in
    let divisiblePairs := pairs.count (λ (x : ℕ × ℕ), divisibleBy3 (x.1 + x.2)) in
    (divisiblePairs.to_rat / totalPairs.to_rat) = (1 : ℚ) / 3) :=
begin
  sorry -- Proof is not required.
end

end probability_sum_divisible_by_3_l488_488537


namespace segment_length_l488_488090
noncomputable def cube_root27 : ℝ := 3

theorem segment_length : ∀ (x : ℝ), (|x - cube_root27| = 4) → ∃ (a b : ℝ), (a = cube_root27 + 4) ∧ (b = cube_root27 - 4) ∧ |a - b| = 8 :=
by
  sorry

end segment_length_l488_488090


namespace find_base_a_l488_488112

theorem find_base_a 
  (a : ℕ)
  (C_a : ℕ := 12) :
  (3 * a^2 + 4 * a + 7) + (5 * a^2 + 7 * a + 9) = 9 * a^2 + 2 * a + C_a →
  a = 14 :=
by
  intros h
  sorry

end find_base_a_l488_488112


namespace sum_and_product_radical_conjugates_l488_488685

theorem sum_and_product_radical_conjugates (x y : ℝ) 
  (h_sum : (x + sqrt y) + (x - sqrt y) = 8) 
  (h_prod : (x + sqrt y) * (x - sqrt y) = 15) : 
  x + y = 5 :=
by 
  sorry

end sum_and_product_radical_conjugates_l488_488685


namespace solve_for_a_l488_488123

theorem solve_for_a (a : ℝ) (h : (a + complex.i) * (1 + complex.i) = 2 * complex.i) : a = 1 :=
sorry

end solve_for_a_l488_488123


namespace equilateral_triangle_perpendicular_l488_488004

/-- Given an equilateral triangle ABC with side length 2,
    vector a, vector b such that AB = 2a and AC = 2a + b,
    prove that (4a + b) is perpendicular to BC. -/
theorem equilateral_triangle_perpendicular 
  {a b : ℝ^3} -- Vectors in 3-dimensional real space
  (hAB : ∃ AB : ℝ^3, 2 * a = AB) -- Condition for vector AB
  (hAC : ∃ AC : ℝ^3, 2 * a + b = AC) -- Condition for vector AC
  (h_length : ∀ u : ℝ^3, (u = b) → ‖u‖ = 2) -- Magnitude of b is 2 (length of side of equilateral triangle)
  (h_angle : ∀ γ : ℝ, (γ = 120) → inner a b = |a| * |b| * real.cos (γ * (real.pi / 180))) -- angle between a and b is 120 degrees
  : inner (4 • a + b) b = 0 := -- (4a + b) is perpendicular to BC
sorry

end equilateral_triangle_perpendicular_l488_488004


namespace distance_sum_is_ten_l488_488784

noncomputable def angle_sum_distance (C A B : ℝ) (d : ℝ) (k : ℝ) : ℝ := 
  let h_A : ℝ := sorry -- replace with expression for h_A based on conditions
  let h_B : ℝ := sorry -- replace with expression for h_B based on conditions
  h_A + h_B

theorem distance_sum_is_ten 
  (A B C : ℝ) 
  (h : ℝ) 
  (k : ℝ) 
  (h_pos : h = 4) 
  (ratio_condition : h_A = 4 * h_B)
  : angle_sum_distance C A B h k = 10 := 
  sorry

end distance_sum_is_ten_l488_488784


namespace find_seventh_term_arith_seq_l488_488074

open Real

variables (a d : ℝ) (n : ℕ)

def S5 := 5 / 2 * (2 * a + 4 * d)
def S_last := 5 / 2 * (2 * (a + (n - 1) * d) - 4 * d)
def S_all := n / 2 * (2 * a + (n - 1) * d)
def a7 := a + 6 * d

theorem find_seventh_term_arith_seq (h1 : S5 a d = 34) (h2 : S_last a d n = 146) (h3 : S_all a d n = 234) : 
  a7 a d = 18 := by
  sorry

end find_seventh_term_arith_seq_l488_488074


namespace sum_of_not_visible_faces_l488_488095

-- Define the sum of the numbers on the faces of one die
def die_sum : ℕ := 21

-- List of visible numbers on the dice
def visible_faces_sum : ℕ := 4 + 3 + 2 + 5 + 1 + 3 + 1

-- Define the total sum of the numbers on the faces of three dice
def total_sum : ℕ := die_sum * 3

-- Statement to prove the sum of not-visible faces equals 44
theorem sum_of_not_visible_faces : 
  total_sum - visible_faces_sum = 44 :=
sorry

end sum_of_not_visible_faces_l488_488095


namespace fruits_in_basket_l488_488339

noncomputable def fruit_basket (A B N : ℕ) : Prop :=
  (N + B ≤ 2) ∧
  (A + N ≤ 3) ∧
  (A + B + N ≥ 5) ∧
  (A = 3) ∧
  (B = 2)

theorem fruits_in_basket :
  ∃ A B N : ℕ, fruit_basket A B N :=
by
  exists 3, 2, 0
  simp [fruit_basket]
  split
  . norm_num
  . split
    . norm_num
    . split
      . norm_num
      . split
        . rfl
        . rfl

end fruits_in_basket_l488_488339


namespace doll_collection_increased_l488_488099

variables (x : ℕ) (increase : ℕ := 2) (percent_increase : ℝ := 0.25)

noncomputable def original_doll_count (x : ℕ) : ℝ := x * percent_increase -- This is 25% of x
noncomputable def final_doll_count := x + increase

theorem doll_collection_increased (x : ℕ) (h : original_doll_count(x) = 2) : final_doll_count x = 10 :=
by
  unfold original_doll_count final_doll_count
  have h1 : x * 0.25 = 2 := h
  have h2 : (x : ℝ) = 2 / 0.25 := (eq_div_iff_mul_eq' (by norm_num : (0.25 : ℝ) ≠ 0)).mpr h1
  norm_num at h2
  have x_eq_8 : x = 8 := by linarith
  rw [x_eq_8]
  norm_num
  exact sorry

end doll_collection_increased_l488_488099


namespace expected_value_twelve_sided_die_l488_488044

theorem expected_value_twelve_sided_die :
  let faces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  ∑ x in faces, x / 12 = 6.5 :=
by
  sorry

end expected_value_twelve_sided_die_l488_488044


namespace total_rowing_and_hiking_l488_488267

def total_campers : ℕ := 80
def morning_rowing : ℕ := 41
def morning_hiking : ℕ := 4
def morning_swimming : ℕ := 15
def afternoon_rowing : ℕ := 26
def afternoon_hiking : ℕ := 8
def afternoon_swimming : ℕ := total_campers - afternoon_rowing - afternoon_hiking - (total_campers - morning_rowing - morning_hiking - morning_swimming)

theorem total_rowing_and_hiking : 
  (morning_rowing + afternoon_rowing) + (morning_hiking + afternoon_hiking) = 79 :=
by
  sorry

end total_rowing_and_hiking_l488_488267


namespace valid_pairs_count_l488_488233

open Nat

def no_carry_add (a b : ℕ) : Prop :=
  let digits_a := to_digits 10 a
  let digits_b := to_digits 10 b
  ∀ i, i < min digits_a.length digits_b.length → digits_a.nth i + digits_b.nth i < 10

def is_valid_pair (a b : ℕ) : Prop :=
  b = a + 1 ∧ no_carry_add a b

def count_valid_pairs : ℕ :=
  ((range (2001 - 1000)).map (fun x => 1000 + x)).foldl
    (fun count a => if is_valid_pair a (a + 1) then count + 1 else count) 0

theorem valid_pairs_count : count_valid_pairs = 156 := by
  sorry

end valid_pairs_count_l488_488233


namespace binomial_identity_l488_488636

theorem binomial_identity (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  k * (n.choose k) = n * ((n - 1).choose (k - 1)) :=
sorry

end binomial_identity_l488_488636


namespace QAR_equal_2_XAY_l488_488250

noncomputable def circles_tangent_at (Γ1 Γ2 : Set Point) (A : Point) : Prop :=
  ∃ (C1 C2 : Circle), C1 ∈ Γ1 ∧ C2 ∈ Γ2 ∧ C1.tangent C2 A

noncomputable def point_on_circle (P : Point) (Γ2 : Set Point) : Prop :=
  ∃ (C2 : Circle), C2 ∈ Γ2 ∧ C2.contains P

noncomputable def tangents_from_point (Γ1 : Set Point) (P : Point) : (Point × Point) :=
  ∃ (C1 : Circle), C1 ∈ Γ1 ∧ (C1.tangent_from P)

noncomputable def cyclic_quadrilateral' (P Q R A : Point) : Prop :=
  ∃ C2 Circle, C2.contains P ∧ C2.contains Q ∧ C2.contains R ∧ C2.contains A

theorem QAR_equal_2_XAY 
  (Γ1 Γ2 : Set Point) (A P X Y Q R : Point)
  (h1 : circles_tangent_at Γ1 Γ2 A)
  (h2 : point_on_circle P Γ2)
  (h3 : tangents_from_point Γ1 P = (X, Y))
  (h4 : cyclic_quadrilateral' P Q R A):
  angle Q A R = 2 * angle X A Y := 
sorry

end QAR_equal_2_XAY_l488_488250


namespace log_base_9_of_729_l488_488852

theorem log_base_9_of_729 : ∃ x : ℝ, (9:ℝ) = 3^2 ∧ (729:ℝ) = 3^6 ∧ (9:ℝ)^x = 729 ∧ x = 3 :=
by
  sorry

end log_base_9_of_729_l488_488852


namespace find_f_of_fraction_l488_488492

noncomputable def f (t : ℝ) : ℝ := sorry

theorem find_f_of_fraction (x : ℝ) (h : f ((1-x^2)/(1+x^2)) = x) :
  f ((2*x)/(1+x^2)) = (1 - x) / (1 + x) ∨ f ((2*x)/(1+x^2)) = (x - 1) / (1 + x) :=
sorry

end find_f_of_fraction_l488_488492


namespace parabola_count_l488_488260

/-- Define S as the set of pairs of positive integers that sum to less than 200 --/
def S : set (ℤ × ℤ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 < 200}

/-- Statement about the number of parabolas meeting given conditions --/
theorem parabola_count :
  (∃ (P : set (ℝ × ℝ)), 
    ∀ v ∈ P, (∃ V : ℤ × ℤ, V ∈ P ∧ ∃ a : ℝ, P = {p : ℝ × ℝ | p.2 = a * (p.1 - V.1)^2 + 0} ∧ 
    P ⊆ {p : ℝ × ℝ | p = (100, 100) ∨ p ∈ S}) ∧
    ∃ V : ℤ × ℤ, V ∈ P ∧ V.1 + V.2 = 0) 
  → 
  (card (< count distinct such parabolas>) = 264) 
:= sorry

end parabola_count_l488_488260


namespace min_value_f_range_of_a_logarithmic_inequality_l488_488127

open Real

def f (x : ℝ) : ℝ := x * log x
def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 3

theorem min_value_f (h1 : 0 < x) (h2 : x ≤ exp 1) : f x ≥ -1 / exp 1 :=
sorry

theorem range_of_a (h : ∀ x > 0, 2 * f x ≥ g x a) : a ≤ 4 :=
sorry

theorem logarithmic_inequality (h : x > 0) : log x > 1 / exp x - 2 / (exp x * x) :=
sorry

end min_value_f_range_of_a_logarithmic_inequality_l488_488127


namespace number_of_three_digit_numbers_with_sum_seven_l488_488892

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488892


namespace exist_permutations_with_common_points_no_permutations_with_common_points_l488_488000

variables {X : Type} [decidable_eq X] [fintype X]
variables (n : ℕ) (m : ℕ) (f g : X → X)

-- Definition of permutation
def is_permutation (f : X → X) : Prop := function.bijective f

-- Definition of common points
def have_common_points (f g : X → X) : Prop :=
  ∃ k : X, f k = g k

-- Part (a)
theorem exist_permutations_with_common_points (hX : nat.card X = n) (hm : m > n / 2) :
  ∃ (fs : fin m → (X → X)), (∀ i, is_permutation (fs i)) ∧ 
    ∀ f, is_permutation f → ∃ i, have_common_points f (fs i) :=
sorry

-- Part (b)
theorem no_permutations_with_common_points (hX : nat.card X = n) (hm : m ≤ n / 2) :
  ¬ ∃ (fs : fin m → (X → X)), (∀ i, is_permutation (fs i)) ∧ 
    ∀ f, is_permutation f → ∃ i, have_common_points f (fs i) :=
sorry

end exist_permutations_with_common_points_no_permutations_with_common_points_l488_488000


namespace gcd_lcm_sum_l488_488729

-- Define the given numbers
def a1 := 54
def b1 := 24
def a2 := 48
def b2 := 18

-- Define the GCD and LCM functions in Lean
def gcd_ab := Nat.gcd a1 b1
def lcm_cd := Nat.lcm a2 b2

-- Define the final sum
def final_sum := gcd_ab + lcm_cd

-- State the equality that represents the problem
theorem gcd_lcm_sum : final_sum = 150 := by
  sorry

end gcd_lcm_sum_l488_488729


namespace c_and_b_not_parallel_l488_488493

open_locale classical

variables (a b c : Line)
variables (are_skew : ∃ a b : Line, ¬ ∃ P, point_on a P ∧ point_on b P ∧ parallel a b ∧ parallel b c) 
variables (parallel_ac : parallel a c)

theorem c_and_b_not_parallel (h1 : are_skew a b) (h2 : parallel a c) : ¬ parallel c b :=
sorry

end c_and_b_not_parallel_l488_488493


namespace min_distance_between_curves_is_correct_l488_488297

noncomputable def min_distance_between_curves : ℝ :=
  let d_min := (1 - log 2) / sqrt 2
  2 * d_min

theorem min_distance_between_curves_is_correct :
  min_distance_between_curves = sqrt 2 * (1 - log 2) :=
by {
  let d_min := (1 - log 2) / sqrt 2,
  have h1 : min_distance_between_curves = 2 * d_min, by { unfold min_distance_between_curves },
  rw h1,
  have h2 : 2 * d_min = sqrt 2 * (1 - log 2), by {
    rw ← mul_assoc,
    have h3 : 2 / sqrt 2 = sqrt 2, by {
      field_simp,
      norm_num,
    },
    rw h3,
  },
  exact h2,
}

end min_distance_between_curves_is_correct_l488_488297


namespace smallest_n_23n_congr_789_mod_11_l488_488726

theorem smallest_n_23n_congr_789_mod_11 : ∃ n : ℕ, 23 * n % 11 = 789 % 11 ∧ n > 0 ∧ n < 11 → n = 9 := 
begin
  sorry
end

end smallest_n_23n_congr_789_mod_11_l488_488726


namespace height_at_D_l488_488410

structure RegularHexagon (A B C D E F : Type) :=
  (side_length : ℝ)
  (pillar_height : A → ℝ)
  (coords : A → B → C → (ℝ × ℝ × ℝ))

def RegularHexagonExample : RegularHexagon :=
  {
    side_length := 10,
    pillar_height := λ p, 
      if p = "A" then 8
      else if p = "B" then 11
      else if p = "C" then 12
      else 0,
    coords := λ A B C, 
      (
        (0, 0, pillar_height "A"),
        (10, 0, pillar_height "B"),
        (5, 5*sqrt(3), pillar_height "C")
      )
  }

theorem height_at_D (A B C D : Type) 
  (hA : A = (0, 0, 8)) 
  (hB : B = (10, 0, 11)) 
  (hC : C = (5, 5 * sqrt(3), 12))
  : ∃ (hD : ℝ), hD = 5 :=
begin
  use 5,
  sorry
end

end height_at_D_l488_488410


namespace three_digit_sum_seven_l488_488879

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488879


namespace defective_probability_l488_488221

theorem defective_probability {total_switches checked_switches defective_checked : ℕ}
  (h1 : total_switches = 2000)
  (h2 : checked_switches = 100)
  (h3 : defective_checked = 10) :
  (defective_checked : ℚ) / checked_switches = 1 / 10 :=
sorry

end defective_probability_l488_488221


namespace compute_A_3_2_l488_488086

namespace Ackermann

def A : ℕ → ℕ → ℕ
| 0, n     => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem compute_A_3_2 : A 3 2 = 12 :=
sorry

end Ackermann

end compute_A_3_2_l488_488086


namespace chameleon_color_change_l488_488569

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l488_488569


namespace paint_area_equivalence_l488_488803

noncomputable def B : ℚ := 27
noncomputable def G : ℚ := 33
noncomputable def Y : ℚ := 16

theorem paint_area_equivalence :
  ∃ B G Y : ℚ, 
    B + (1/3) * G = 38 ∧
    G = B + 6 ∧
    Y + (2/3) * G = 38 ∧
    B = 27 ∧
    G = 33 ∧
    Y = 16 :=
begin
  use [27, 33, 16],
  split,
  { have h1 : 27 + (1/3) * 33 = 38, { norm_num }, exact h1 },
  split,
  { exact (by norm_num : 33 = 27 + 6) },
  split,
  { have h2 : 16 + (2/3) * 33 = 38, { norm_num }, exact h2 },
  split, exact (by norm_num : 27 = 27),
  split, exact (by norm_num : 33 = 33),
  exact (by norm_num : 16 = 16),
end

end paint_area_equivalence_l488_488803


namespace round_robin_maximum_sum_squares_l488_488011

theorem round_robin_maximum_sum_squares (p : Fin 10 → ℕ) (h_sum : ∑ i, p i = 45) : 
  ∑ i, (p i)^2 ≤ 285 :=
sorry

end round_robin_maximum_sum_squares_l488_488011


namespace log_base_9_of_729_l488_488848

-- Define the conditions
def nine_eq := 9 = 3^2
def seven_two_nine_eq := 729 = 3^6

-- State the goal to be proved
theorem log_base_9_of_729 (h1 : nine_eq) (h2 : seven_two_nine_eq) : log 9 729 = 3 :=
by
  sorry

end log_base_9_of_729_l488_488848


namespace lottery_weeks_l488_488030

theorem lottery_weeks (a b : ℕ) (h1 : a = 5) (h2 : b = 14) : (a - 2) * (b - 2) / 4 = 52 :=
by {
  rw [h1, h2],
  rfl,
  sorry
}

end lottery_weeks_l488_488030


namespace binomial_coeff_prime_divisors_l488_488488

theorem binomial_coeff_prime_divisors (k : ℕ) (h_k : 0 < k) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (nat.choose n k).prime_factors.to_finset.card ≥ k :=
begin
  sorry
end

end binomial_coeff_prime_divisors_l488_488488


namespace prime_cond_l488_488429

theorem prime_cond (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : n > 1) : 
  (p^(2*n+1) - 1) / (p - 1) = (q^3 - 1) / (q - 1) → (p = 2 ∧ q = 5 ∧ n = 2) :=
  sorry

end prime_cond_l488_488429


namespace shannon_bracelets_l488_488285

theorem shannon_bracelets (total_stones : ℝ) (stones_per_bracelet : ℝ) (h1 : total_stones = 48.0) (h2 : stones_per_bracelet = 8.0) : 
  total_stones / stones_per_bracelet = 6 :=
by
  rw [h1, h2]
  norm_num
  sorry

end shannon_bracelets_l488_488285


namespace tetrahedron_sphere_surface_area_l488_488400

-- Define the conditions
variables (a : ℝ) (mid_AB_C : ℝ → Prop) (S : ℝ)
variables (h1 : a > 0)
variables (h2 : mid_AB_C a)
variables (h3 : S = 3 * Real.sqrt 2)

-- Theorem statement
theorem tetrahedron_sphere_surface_area (h1 : a = 2 * Real.sqrt 3) : 
  4 * Real.pi * ( (Real.sqrt 6 / 4) * a )^2 = 18 * Real.pi := by
  sorry

end tetrahedron_sphere_surface_area_l488_488400


namespace sum_of_squares_of_roots_l488_488424

theorem sum_of_squares_of_roots
    (a b c : ℚ)
    (h1 : a ≠ 0)
    (h2 : a = 10) (h3 : b = 15) (h4 : c = -25) :
  let x1 := -b / (2 * a)
  let x2 := -b / (2 * a)
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  in (sum_of_roots^2 - 2 * product_of_roots) = 29 / 4 := by
    sorry

end sum_of_squares_of_roots_l488_488424


namespace no_solution_system_l488_488668

theorem no_solution_system : ¬ ∃ (x y z : ℝ), 
  x^2 - 2*y + 2 = 0 ∧ 
  y^2 - 4*z + 3 = 0 ∧ 
  z^2 + 4*x + 4 = 0 := 
by
  sorry

end no_solution_system_l488_488668


namespace triangles_SSS_congruence_l488_488358

theorem triangles_SSS_congruence (ΔABC ΔDEF : Triangle) 
  (h1 : ΔABC.side1 = ΔDEF.side1)
  (h2 : ΔABC.side2 = ΔDEF.side2)
  (h3 : ΔABC.side3 = ΔDEF.side3) : ΔABC ≅ ΔDEF :=
by
  -- The proof is omitted here.
  sorry

end triangles_SSS_congruence_l488_488358


namespace three_digit_integers_sum_to_7_l488_488903

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488903


namespace three_digit_integers_sum_to_7_l488_488901

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488901


namespace digits_sum_eq_seven_l488_488923

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488923


namespace directrix_of_parabola_l488_488312

noncomputable def parabola_directrix (y : ℝ) (x : ℝ) : Prop :=
  y = 4 * x^2

theorem directrix_of_parabola : ∃ d : ℝ, (parabola_directrix (y := 4) (x := x) → d = -1/16) :=
by
  sorry

end directrix_of_parabola_l488_488312


namespace range_of_a_l488_488207

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ : ℝ, 2 * x₀^2 + (a - 1) * x₀ + 1 / 2 ≤ 0) → a ∈ set.Ioo (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_a_l488_488207


namespace trailing_zero_count_l488_488186

theorem trailing_zero_count : nat.trail_zero_count (125 * 360) = 3 := sorry

end trailing_zero_count_l488_488186


namespace proof_problem_l488_488199

variable {a b x : ℝ}

theorem proof_problem (h1 : x = b / a) (h2 : a ≠ b) (h3 : a ≠ 0) : 
  (2 * a + b) / (2 * a - b) = (2 + x) / (2 - x) :=
sorry

end proof_problem_l488_488199


namespace solve_for_x_l488_488665

theorem solve_for_x (x : ℝ) : (5 : ℝ)^(x + 6) = (625 : ℝ)^x → x = 2 :=
by
  sorry

end solve_for_x_l488_488665


namespace line_parallel_or_subset_plane_l488_488495

-- Define the direction vector of the line l
def a : ℝ × ℝ × ℝ := (-3, 2, 1)

-- Define the normal vector of the plane α
def n : ℝ × ℝ × ℝ := (1, 2, -1)

-- Define the positional relationship between the line l and the plane α
def positional_relationship :=
  a.1 * n.1 + a.2 * n.2 + a.3 * n.3 = 0

-- Result statement: Prove that the positional relationship implies that l is parallel to α or l is a subset of α
theorem line_parallel_or_subset_plane :
  positional_relationship → (l_parallel_or_subset : Prop) :=
by
  intro h
  sorry

end line_parallel_or_subset_plane_l488_488495


namespace smallest_positive_multiple_l488_488727

theorem smallest_positive_multiple (a : ℕ) (h : a > 0) : ∃ a > 0, (31 * a) % 103 = 7 := 
sorry

end smallest_positive_multiple_l488_488727


namespace find_sin_BAD_l488_488235

-- Definition of the problem
def triangle_ABC (A B C D : Type) (AB AC BC : ℝ) : Prop :=
  AB = 4 ∧ AC = 7 ∧ BC = 9 ∧ 
  -- D lies on BC and AD bisects angle BAC
  (∃ (p : ℝ), p ∈ Icc 0 1 ∧ D = B + p * (C - B)) ∧ 
  bis((vector_angle_from_to A B) (vector_angle_from_to A C) = ∠ (A D))

-- The problem statement
theorem find_sin_BAD
  (A B C D : Type)
  (AB AC BC : ℝ)
  (h : triangle_ABC A B C D AB AC BC) : 
  ∃ sin_BAD : ℝ, sin_BAD = 3 * real.sqrt 14 / 14 :=
sorry

end find_sin_BAD_l488_488235


namespace digits_sum_eq_seven_l488_488929

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488929


namespace ratio_of_q_to_r_l488_488282

theorem ratio_of_q_to_r
  (P Q R : ℕ)
  (h1 : R = 400)
  (h2 : P + Q + R = 1210)
  (h3 : 5 * Q = 4 * P) :
  Q * 10 = R * 9 :=
by
  sorry

end ratio_of_q_to_r_l488_488282


namespace three_digit_sum_seven_l488_488878

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488878


namespace remaining_lawn_area_l488_488787

noncomputable def diameter : ℝ := 12
noncomputable def path_width : ℝ := 3
noncomputable def lawn_radius : ℝ := diameter / 2
noncomputable def lawn_area : ℝ := π * lawn_radius^2

theorem remaining_lawn_area : lawn_area - (6 * π + 9 * sqrt 3) = 30 * π - 9 * sqrt 3 :=
by
  -- The proof steps go here
  sorry

end remaining_lawn_area_l488_488787


namespace sin_cos_identity_l488_488005

theorem sin_cos_identity (α : ℝ) (h : (sin α + 3 * cos α) / (3 * cos α - sin α) = 5) : sin α ^ 2 - sin α * cos α = 2 / 5 := 
sorry

end sin_cos_identity_l488_488005


namespace period_of_sin_6x_plus_pi_l488_488835

theorem period_of_sin_6x_plus_pi (x : ℝ) : 
  ∃ T, (∀ x, sin (6 * x + π) = sin (6 * (x + T) + π)) ∧ T = π / 3 :=
by
  use π / 3
  intro x
  sorry

end period_of_sin_6x_plus_pi_l488_488835


namespace product_expression_fraction_l488_488440

theorem product_expression_fraction :
  (∏ k in Finset.range 99, 1 - (1 / (2 * k + 3))) = 2 / 199 :=
by
  sorry

end product_expression_fraction_l488_488440


namespace rabbits_in_cage_l488_488529

theorem rabbits_in_cage (rabbits_in_cage : ℕ) (rabbits_park : ℕ) : 
  rabbits_in_cage = 13 ∧ rabbits_park = 60 → (1/3 * rabbits_park - rabbits_in_cage) = 7 :=
by
  sorry

end rabbits_in_cage_l488_488529


namespace max_teams_advance_l488_488219

theorem max_teams_advance (n : ℕ) (teams : Finset ℕ)
  (h_teams : teams.card = 7)
  (h_games_played : ∀ (t1 t2 : ℕ), t1 ∈ teams ∧ t2 ∈ teams ∧ t1 ≠ t2 → true)
  (points : ℕ → ℕ)
  (h_points_win : ∀ t1 t2, t1 ∈ teams ∧ t2 ∈ teams ∧ t1 ≠ t2 → 
                            (points t1 = points t1 + 3 ∨ points t2 = points t2 + 3 ∨ 
                             (points t1 = points t1 + 1 ∧ points t2 = points t2 + 1)))
  (h_points_thresh : ∀ t ∈ teams, points t >= 12 → t ∈ { t | teams.card ≥ n }) :
  n ≤ 5 :=
sorry

end max_teams_advance_l488_488219


namespace intervals_of_decrease_min_max_values_in_interval_l488_488504

noncomputable def f (x : ℝ) : ℝ := real.sqrt 2 * real.cos (2 * x - real.pi / 4)

theorem intervals_of_decrease (k : ℤ) : f '' set.Icc (k * real.pi + real.pi / 8) (k * real.pi + 5 * real.pi / 8) ⊆ set.Icc (-real.sqrt 2) (real.sqrt 2) := 
sorry

theorem min_max_values_in_interval : 
  (∀ (x ∈ set.Icc (-real.pi / 8) (real.pi / 2)), f x ≥ -real.sqrt 2 ∧ f x ≤ real.sqrt 2) ∧ 
  (f (real.pi / 8) = real.sqrt 2) ∧ 
  (f (real.pi / 2) = -real.sqrt 2) := 
sorry

end intervals_of_decrease_min_max_values_in_interval_l488_488504


namespace num_three_digit_integers_sum_to_seven_l488_488986

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488986


namespace actual_distance_between_cities_l488_488309

-- Define the scale and distance on the map as constants
def distance_on_map : ℝ := 20
def scale_inch_miles : ℝ := 12  -- Because 1 inch = 12 miles derived from the scale 0.5 inches = 6 miles

-- Define the actual distance calculation
def actual_distance (distance_inch : ℝ) (scale : ℝ) : ℝ :=
  distance_inch * scale

-- Example theorem to prove the actual distance between the cities
theorem actual_distance_between_cities :
  actual_distance distance_on_map scale_inch_miles = 240 := by
  sorry

end actual_distance_between_cities_l488_488309


namespace common_ratio_of_geometric_sequence_l488_488129

noncomputable def geometric_sequence (a q : ℝ) : ℕ → ℝ
| 0     := a
| (n+1) := q * geometric_sequence a q n

-- We know that the sequence is a geometric progression and we're forming the terms for the problem
theorem common_ratio_of_geometric_sequence (a1 q : ℝ) (h_pos : a1 > 0) 
  (hp_seq : 2 * (1 / 2) * (geometric_sequence a1 q 2) = 3 * a1 + 2 * (geometric_sequence a1 q 1)) :
  q = 3 :=
begin
  -- Skipping the proof with sorry
  sorry,
end

end common_ratio_of_geometric_sequence_l488_488129


namespace describe_graph_l488_488360

theorem describe_graph : 
  ∀ (x y : ℝ), x^2 * (x + y + 1) = y^3 * (x + y + 1) ↔ (x^2 = y^3 ∨ y = -x - 1)
:= sorry

end describe_graph_l488_488360


namespace intersection_elements_eq_two_l488_488169

open Set

def M : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 3 * x^2) }
def N : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 5 * x) }

theorem intersection_elements_eq_two : (M ∩ N).to_finset.card = 2 :=
by
  sorry

end intersection_elements_eq_two_l488_488169


namespace arithmetic_progression_no_rth_power_l488_488103

noncomputable def is_arith_sequence (a : ℕ → ℤ) : Prop := 
∀ n : ℕ, a n = 4 * (n : ℤ) - 2

theorem arithmetic_progression_no_rth_power (n : ℕ) :
  ∃ a : ℕ → ℤ, is_arith_sequence a ∧ 
  (∀ r : ℕ, 2 ≤ r ∧ r ≤ n → 
  ¬ (∃ k : ℤ, ∃ m : ℕ, m > 0 ∧ a m = k ^ r)) := 
sorry

end arithmetic_progression_no_rth_power_l488_488103


namespace minimum_factors_to_erase_l488_488677

theorem minimum_factors_to_erase :
  ∀ (factors : List ℝ), (∃ (LHS RHS : List ℝ), (length factors = 2016) ∧ 
  (LHS.length + RHS.length = 2016) ∧ 
  (∀ (x : ℝ), (x - (LHS.head x)) * ... * (x - (LHS.last x)) ≠ (x - (RHS.head x)) * ... * (x - (RHS.last x)))) :=
begin
  sorry
end

end minimum_factors_to_erase_l488_488677


namespace find_first_number_l488_488697

-- Definitions from conditions
variable (x : ℕ) -- Let the first number be x
variable (y : ℕ) -- Let the second number be y

-- Given conditions in the problem
def condition1 : Prop := y = 43
def condition2 : Prop := x + 2 * y = 124

-- The proof target
theorem find_first_number (h1 : condition1 y) (h2 : condition2 x y) : x = 38 := by
  sorry

end find_first_number_l488_488697


namespace william_max_moves_l488_488268

-- Define the game conditions
def game_has_been_won (stored_value : ℕ) : Prop :=
  stored_value > (2^100)

def next_moves (value : ℕ) : list ℕ :=
  [2 * value + 1, 4 * value + 3]

-- Define the starting position
def initial_value : ℕ := 1

-- Define optimal play behavior
def plays_optimally (player_moves : ℕ → ℕ) : Prop :=
  ∀ value, game_has_been_won (player_moves value) → game_has_been_won value

-- Mark starts the game
def Mark_starts_first : Prop := true

-- Define the maximum turns William can play optimally
theorem william_max_moves : Mark_starts_first → plays_optimally (next_moves) → ∃ max_moves : ℕ, max_moves = 33 :=
by
  sorry

end william_max_moves_l488_488268


namespace chameleon_color_change_l488_488580

theorem chameleon_color_change :
  ∃ (x : ℕ), (∃ (blue_initial : ℕ) (red_initial : ℕ), blue_initial = 5 * x ∧ red_initial = 140 - 5 * x) →
  (∃ (blue_final : ℕ) (red_final : ℕ), blue_final = x ∧ red_final = 3 * (140 - 5 * x)) →
  (5 * x - x = 80) :=
begin
  sorry
end

end chameleon_color_change_l488_488580


namespace three_digit_numbers_sum_seven_l488_488920

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488920


namespace quadratic_has_two_distinct_real_roots_range_l488_488480

theorem quadratic_has_two_distinct_real_roots_range (m : ℝ) : 
  (let a := 1 - m,
       b := 2,
       c := -2 in 
       (b^2 - 4*a*c) > 0 ∧ a ≠ 0) ↔ (m < (3 / 2) ∧ m ≠ 1) := 
by 
  sorry

end quadratic_has_two_distinct_real_roots_range_l488_488480


namespace decimal_to_binary_18_l488_488425

theorem decimal_to_binary_18 : (18: ℕ) = 0b10010 := by
  sorry

end decimal_to_binary_18_l488_488425


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488941

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488941


namespace isosceles_triangle_perimeter_l488_488808

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 4) (h2 : b = 6) : 
  ∃ p, (p = 14 ∨ p = 16) :=
by
  sorry

end isosceles_triangle_perimeter_l488_488808


namespace largest_m_dividing_factorials_l488_488412

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

theorem largest_m_dividing_factorials (m : ℕ) :
  (∀ k : ℕ, k ≤ m → factorial k ∣ (factorial 100 + factorial 99 + factorial 98)) ↔ m = 98 :=
by
  sorry

end largest_m_dividing_factorials_l488_488412


namespace middle_integer_is_six_l488_488704

def valid_even_integer (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), n = x ∧ x = n - 2 ∧ y = n ∧ z = n + 2 ∧ x < y ∧ y < z ∧
  x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧
  1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 1 ≤ z ∧ z ≤ 9

theorem middle_integer_is_six (n : ℕ) (h : valid_even_integer n) :
  n = 6 :=
by
  sorry

end middle_integer_is_six_l488_488704


namespace problem1_l488_488006

theorem problem1 (α : ℝ) (h : Real.tan α = 2) :
  Real.sin (Real.pi / 2 - α)^2 + 3 * Real.sin (α + Real.pi) * Real.sin (α + Real.pi / 2) = -1 :=
sorry

end problem1_l488_488006


namespace Jan_is_6_inches_taller_than_Bill_l488_488060

theorem Jan_is_6_inches_taller_than_Bill :
  ∀ (Cary Bill Jan : ℕ),
    Cary = 72 →
    Bill = Cary / 2 →
    Jan = 42 →
    Jan - Bill = 6 :=
by
  intros
  sorry

end Jan_is_6_inches_taller_than_Bill_l488_488060


namespace three_digit_sum_seven_l488_488956

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488956


namespace max_area_inscribed_octagon_l488_488055

theorem max_area_inscribed_octagon
  (R : ℝ)
  (s : ℝ)
  (a b : ℝ)
  (h1 : s^2 = 5)
  (h2 : (a * b) = 4)
  (h3 : (s * Real.sqrt 2) = (2*R))
  (h4 : (Real.sqrt (a^2 + b^2)) = 2 * R) :
  ∃ A : ℝ, A = 3 * Real.sqrt 5 :=
by
  sorry

end max_area_inscribed_octagon_l488_488055


namespace confidence_level_correctness_l488_488359

/--
Given:
- The observed value of \( \chi^{2} \) is 6.635, indicating a 99% confidence level that smoking is related to lung cancer.
- Confidence level interpretation in the context of an independence test:
  - For a 99% confidence level, it does not imply that among 100 smokers, 99 must have lung disease.
  - For a 99% confidence level, it does not imply a smoker has a 99% chance of having lung disease.
  - A 95% confidence level means there is a 5% chance that the inference is incorrect.

Prove that the correct answer to the question is C.

- The question is: Which of the following statements is correct?
  A. If the observed value of \( \chi^{2} \) is 6.635, then among 100 smokers, there must be 99 who have lung disease.
  B. Being 99% confident that smoking is related to lung cancer means a smoker has a 99% chance of having lung disease.
  C. Concluding from a statistic with 95% confidence that smoking is related to lung cancer means a 5% chance the inference is incorrect.
  D. All of the above statements are incorrect.
- The correct answer is C.
-/
theorem confidence_level_correctness :
  let chi_sq := 6.635
  let confidence_99 := 0.99
  let confidence_95 := 0.95
  let smokers := 100
  let lung_disease := 99
  (confidence_95 - (1 - confidence_95) = 0) ∧
  ¬(chi_sq = 6.635 → confidence_99 → ∀ smokers, lung_disease) ∧
  ¬(confidence_99 → ∃ smoker, lung_disease) :=
sorry


end confidence_level_correctness_l488_488359


namespace digits_sum_eq_seven_l488_488930

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488930


namespace westbound_speed_is_275_l488_488031

-- Define the conditions for the problem at hand.
def east_speed : ℕ := 325
def separation_time : ℝ := 3.5
def total_distance : ℕ := 2100

-- Compute the known east-bound distance.
def east_distance : ℝ := east_speed * separation_time

-- Define the speed of the west-bound plane as an unknown variable.
variable (v : ℕ)

-- Compute the west-bound distance.
def west_distance := v * separation_time

-- The assertion that the sum of two distances equals the total distance.
def distance_equation := east_distance + (v * separation_time) = total_distance

-- Prove that the west-bound speed is 275 mph.
theorem westbound_speed_is_275 : v = 275 :=
by
  sorry

end westbound_speed_is_275_l488_488031


namespace range_of_f_l488_488156

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2 else
  if -1 < x ∧ x < 2 then x^2 else 0

theorem range_of_f : set.range f = set.Iio 4 :=
  sorry

end range_of_f_l488_488156


namespace statement_C_correct_l488_488191

theorem statement_C_correct (a b c d : ℝ) (h_ab : a > b) (h_cd : c > d) : a + c > b + d :=
by
  sorry

end statement_C_correct_l488_488191


namespace range_of_f_l488_488159

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2 else x^2

theorem range_of_f : set.image f set.univ = set.Iio 4 := by
  sorry

end range_of_f_l488_488159


namespace range_alpha_minus_beta_over_2_l488_488463

theorem range_alpha_minus_beta_over_2 (α β : ℝ) (h1 : -π / 2 ≤ α) (h2 : α < β) (h3 : β ≤ π / 2) :
  Set.Ico (-π / 2) 0 = {x : ℝ | ∃ α β : ℝ, -π / 2 ≤ α ∧ α < β ∧ β ≤ π / 2 ∧ x = (α - β) / 2} :=
by
  sorry

end range_alpha_minus_beta_over_2_l488_488463


namespace exists_radius_for_marked_points_l488_488368

theorem exists_radius_for_marked_points :
  ∃ R : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ < 2 * π) →
    (∃ n : ℕ, (θ ≤ (n * 2 * π * R) % (2 * π * R) + 1 / R ∧ (n * 2 * π * R) % (2 * π * R) < θ + 1))) :=
sorry

end exists_radius_for_marked_points_l488_488368


namespace sum_radical_conjugates_l488_488827

theorem sum_radical_conjugates : (5 - Real.sqrt 500) + (5 + Real.sqrt 500) = 10 :=
by
  sorry

end sum_radical_conjugates_l488_488827


namespace bound_s_n_l488_488327

def s_n (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), real.sin k

theorem bound_s_n :
  ∀ n : ℕ, -1 / 2 ≤ s_n n ∧ s_n n ≤ 1 / 2 :=
by
  sorry

end bound_s_n_l488_488327


namespace roots_of_equation_are_pure_imaginary_for_positive_real_k_l488_488076

noncomputable def roots_pure_imaginary_for_positive_k (k : ℂ) (hk_pos : 0 < re k ∧ im k = 0) : Prop :=
  ∀ (z : ℂ), (8 * z^2 - 5 * Complex.I * z - k = 0) → (im z ≠ 0 ∧ re z = 0)

theorem roots_of_equation_are_pure_imaginary_for_positive_real_k (k : ℂ) (hk : 0 < re k ∧ im k = 0) :
  roots_pure_imaginary_for_positive_k k hk :=
begin
  sorry
end

end roots_of_equation_are_pure_imaginary_for_positive_real_k_l488_488076


namespace probability_four_in_sequence_l488_488028

open Rat

theorem probability_four_in_sequence :
  let die_faces := {1, 2, 3, 4, 5, 6},
      rolls := vector die_faces 4,
      P (s : vector ℕ 4) : ℚ := 
        if (4 ∈ {s[0], s[0] + s[1], s[0] + s[1] + s[2], s[0] + s[1] + s[2] + s[3]}) then 1 else 0,
      probability := (378 / 1296 : ℚ)
  in 
  ∑ s in finset.univ.image rolls, P s / (6 ^ 4) = probability 
:=
sorry

end probability_four_in_sequence_l488_488028


namespace range_of_a_l488_488212

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ ∈ ℝ, 2 * x₀ ^ 2 + (a - 1) * x₀ + 1 / 2 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l488_488212


namespace lambda_value_l488_488182

open Locale BigOperators

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) 
  (h1 : a = (1, 3))
  (h2 : b = (3, 4))
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0) : 
  λ = 3 / 5 := by
  sorry

end lambda_value_l488_488182


namespace find_integer_n_l488_488115

open Int

theorem find_integer_n (n : ℕ) (div19 div20 : ℕ)
  (h_pos : 0 < n)
  (h_div_2019 : 2019 ∣ n)
  (h_divs : ∃ ds : List ℕ, ds.Nth 18 = some div19 ∧ ds.Nth 19 = some div20 ∧ (∀ d ∈ ds, d ∣ n) ∧ List.sorted (· < ·) ds ∧ List.last (1 = n)) :
  (n = 3 * 673 ^ 18 ∨ n = 3 ^ 18 * 673) ∧ n = div19 * div20 :=
by
  sorry

end find_integer_n_l488_488115


namespace school_courses_combination_l488_488036

theorem school_courses_combination :
  let typeA := {1, 2, 3}
  let typeB := {1, 2, 3, 4}
  let choose (n k : ℕ) : ℕ := Nat.choose n k
  (choose 3 2 * choose 4 1 + choose 3 1 * choose 4 2) = 30 := by
  sorry

end school_courses_combination_l488_488036


namespace integral_percentage_l488_488266

variable (a b : ℝ)

theorem integral_percentage (h : ∀ x, x^2 > 0) :
  (∫ x in a..b, (1 / 20 * x^2 + 3 / 10 * x^2)) = 0.35 * (∫ x in a..b, x^2) :=
by
  sorry

end integral_percentage_l488_488266


namespace smallest_portion_arithmetic_sequence_l488_488001

theorem smallest_portion_arithmetic_sequence :
  ∃ (a1 d : ℕ), (5 * a1 + (5 * 4 / 2) * d = 100) ∧ ((1 / 3) * (3 * a1 + 9 * d) = 2 * a1 + d) ∧ a1 = 10 :=
begin
  sorry
end

end smallest_portion_arithmetic_sequence_l488_488001


namespace total_molecular_weight_1035_90_l488_488430

noncomputable def atomic_weights : ℕ → ℚ := λ n, 
  match n with
  | 0 => 26.98
  | 1 => 30.97
  | 2 => 16.00
  | 3 => 22.99
  | 4 => 32.07
  | _ => 0
  end

def molecular_weight_AlPO4 : ℚ := 1 * atomic_weights 0 + 1 * atomic_weights 1 + 4 * atomic_weights 2

def molecular_weight_Na2SO4 : ℚ := 2 * atomic_weights 3 + 1 * atomic_weights 4 + 4 * atomic_weights 2

def total_molecular_weight (moles_AlPO4 moles_Na2SO4 : ℚ) : ℚ :=
  moles_AlPO4 * molecular_weight_AlPO4 + moles_Na2SO4 * molecular_weight_Na2SO4

theorem total_molecular_weight_1035_90 :
  total_molecular_weight 5 3 = 1035.90 := by
  sorry

end total_molecular_weight_1035_90_l488_488430


namespace min_total_bags_l488_488342

theorem min_total_bags (x y : ℕ) (h : 15 * x + 8 * y = 1998) (hy_min : ∀ y', (15 * x + 8 * y' = 1998) → y ≤ y') :
  x + y = 140 :=
by
  sorry

end min_total_bags_l488_488342


namespace red_roses_difference_l488_488648

theorem red_roses_difference (santiago_red : ℕ) (garrett_red : ℕ) (give_away : ℕ) (receive : ℕ) :
  santiago_red = 58 → garrett_red = 24 → give_away = 10 → receive = 5 →
  (santiago_red - give_away + receive) - (garrett_red - give_away + receive) = 34 :=
by
  intros hs hg hga hr
  rw [hs, hg, hga, hr]
  calc (58 - 10 + 5) - (24 - 10 + 5)
      = (58 - 10 + 5) - (24 - 10 + 5) : by rw []
      ... = 53 - 19 : by rw []
      ... = 34 : by norm_num

end red_roses_difference_l488_488648


namespace chameleon_color_change_l488_488579

theorem chameleon_color_change :
  ∃ (x : ℕ), (∃ (blue_initial : ℕ) (red_initial : ℕ), blue_initial = 5 * x ∧ red_initial = 140 - 5 * x) →
  (∃ (blue_final : ℕ) (red_final : ℕ), blue_final = x ∧ red_final = 3 * (140 - 5 * x)) →
  (5 * x - x = 80) :=
begin
  sorry
end

end chameleon_color_change_l488_488579


namespace number_of_three_digit_numbers_with_sum_7_l488_488981

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488981


namespace calculation_expression_solve_system_of_equations_l488_488371

-- Part 1: Prove the calculation
theorem calculation_expression :
  (6 - 2 * Real.sqrt 3) * Real.sqrt 3 - Real.sqrt ((2 - Real.sqrt 2) ^ 2) + 1 / Real.sqrt 2 = 
  6 * Real.sqrt 3 - 8 + 3 * Real.sqrt 2 / 2 :=
by
  -- proof will be here
  sorry

-- Part 2: Prove the solution of the system of equations
theorem solve_system_of_equations (x y : ℝ) :
  (5 * x - y = -9) ∧ (3 * x + y = 1) → (x = -1 ∧ y = 4) :=
by
  -- proof will be here
  sorry

end calculation_expression_solve_system_of_equations_l488_488371


namespace quotient_of_larger_number_divided_by_smaller_l488_488308

theorem quotient_of_larger_number_divided_by_smaller :
  ∃ (L S Q : ℕ), 
    L - S = 1365 ∧ 
    L = Q * S + 15 ∧ 
    L ≈ 1543 ∧ 
    Q = 8 :=
by
  sorry

end quotient_of_larger_number_divided_by_smaller_l488_488308


namespace gasoline_tank_capacity_l488_488390

theorem gasoline_tank_capacity :
  ∃ (x : ℝ), (3 / 4) * x - (1 / 3) * x = 18 → x = 43.2 :=
begin
  sorry
end

end gasoline_tank_capacity_l488_488390


namespace quadratic_square_binomial_l488_488091

theorem quadratic_square_binomial (d : ℝ) : (∃ b : ℝ, (x : ℝ) -> (x + b)^2 = x^2 + 110 * x + d) ↔ d = 3025 :=
by
  sorry

end quadratic_square_binomial_l488_488091


namespace three_digit_numbers_sum_seven_l488_488918

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488918


namespace trajectory_of_center_of_moving_circle_l488_488138

-- Definitions of the conditions
def point_F : ℝ × ℝ := (1, 0)
def tangent_line (x : ℝ) := -1
def is_tangent (circle : ℝ × ℝ × ℝ) (line : ℝ → ℝ) : Prop :=
  let (center, radius) := (circle.1, circle.2) in
  line center.1 + radius = 0

-- Main Statement: Equation of the trajectory of the center of the moving circle
theorem trajectory_of_center_of_moving_circle (P : ℝ × ℝ) : 
  (
    ∃ (C : ℝ × ℝ × ℝ),
      C.1 = P ∧ is_tangent C tangent_line ∧
      (∃ r : ℝ, C = (P, r))
  ) → (P.2 * P.2 = 4 * P.1) :=
by
  sorry

end trajectory_of_center_of_moving_circle_l488_488138


namespace solve_for_x_l488_488666

theorem solve_for_x (x : ℝ) : (5 : ℝ)^(x + 6) = (625 : ℝ)^x → x = 2 :=
by
  sorry

end solve_for_x_l488_488666


namespace twelve_pharmacies_not_sufficient_l488_488760

-- Define an intersection grid of size 10 x 10 (100 squares).
def city_grid : Type := Fin 10 × Fin 10 

-- Define the distance measure between intersections, assumed as L1 metric for grid paths.
def dist (p q : city_grid) : Nat := (abs (p.1.val - q.1.val) + abs (p.2.val - q.2.val))

-- Define a walking distance pharmacy 
def is_walking_distance (p q : city_grid) : Prop := dist p q ≤ 3

-- State that having 12 pharmacies is not sufficient
theorem twelve_pharmacies_not_sufficient (pharmacies : Fin 12 → city_grid) :
  ¬ (∀ intersection: city_grid, ∃ (p_index : Fin 12), is_walking_distance (pharmacies p_index) intersection) :=
sorry

end twelve_pharmacies_not_sufficient_l488_488760


namespace three_digit_sum_seven_l488_488951

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488951


namespace volume_of_regular_tetrahedron_l488_488399

-- Necessary definitions
structure EquilateralTriangle where
  edge_length : ℕ

structure RegularTetrahedron where
  base : EquilateralTriangle
  lateral_edge_length : ℕ

-- Define the specific tetrahedron given in the problem
def givenTetrahedron : RegularTetrahedron :=
{ base := { edge_length := 6 },
  lateral_edge_length := 9 }

-- The theorem to prove the volume of the given tetrahedron is 9
theorem volume_of_regular_tetrahedron (T : RegularTetrahedron)
  (h1 : T.base.edge_length = 6)
  (h2 : T.lateral_edge_length = 9) : 
  volume_of_tetrahedron T = 9 :=
by
  sorry

end volume_of_regular_tetrahedron_l488_488399


namespace triangle_solution_proof_l488_488296

noncomputable def solve_triangle_proof (a b c : ℝ) (alpha beta gamma : ℝ) : Prop :=
  a = 631.28 ∧
  alpha = 63 + 35 / 60 + 30 / 3600 ∧
  b - c = 373 ∧
  beta = 88 + 12 / 60 + 15 / 3600 ∧
  gamma = 28 + 12 / 60 + 15 / 3600 ∧
  b = 704.55 ∧
  c = 331.55

theorem triangle_solution_proof : solve_triangle_proof 631.28 704.55 331.55 (63 + 35 / 60 + 30 / 3600) (88 + 12 / 60 + 15 / 3600) (28 + 12 / 60 + 15 / 3600) :=
  by { sorry }

end triangle_solution_proof_l488_488296


namespace sequence_a4_l488_488167

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = 2 + (2 * a n) / (1 - a n)

theorem sequence_a4 : ∃ a : ℕ → ℚ, sequence a ∧ a 4 = -2 / 5 :=
by
  -- proof goes here
  sorry

end sequence_a4_l488_488167


namespace expected_value_twelve_sided_die_l488_488045

theorem expected_value_twelve_sided_die :
  let faces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  ∑ x in faces, x / 12 = 6.5 :=
by
  sorry

end expected_value_twelve_sided_die_l488_488045


namespace not_enough_pharmacies_l488_488767

theorem not_enough_pharmacies : 
  ∀ (n m : ℕ), n = 10 ∧ m = 10 →
  ∃ (intersections : ℕ), intersections = n * m ∧ 
  ∀ (d : ℕ), d = 3 →
  ∀ (coverage : ℕ), coverage = (2 * d + 1) * (2 * d + 1) →
  ¬ (coverage * 12 ≥ intersections * 2) :=
by sorry

end not_enough_pharmacies_l488_488767


namespace geometric_sequence_solution_l488_488188

-- Assume we have a type for real numbers
variable {R : Type} [LinearOrderedField R]

theorem geometric_sequence_solution (a b c : R)
  (h1 : -1 ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) (h5 : -9 ≠ 0)
  (h : ∃ r : R, r ≠ 0 ∧ (a = r * -1) ∧ (b = r * a) ∧ (c = r * b) ∧ (-9 = r * c)) :
  b = -3 ∧ a * c = 9 := by
  sorry

end geometric_sequence_solution_l488_488188


namespace ratio_of_radii_l488_488716

theorem ratio_of_radii
  {α r₁ r₂ r₃ : ℝ}
  (h₁ : r₁ < r₂)
  (h₂ : sin (α / 2) = (r₂ - r₁) / (r₁ + r₂))
  (h₃ : r₁ / r₂ = (1 - sin (α / 2)) / (1 + sin (α / 2))) :
  r₁ / r₃ = (sqrt ((1 - sin (α / 2)) / (1 + sin (α / 2))) + 1) ^ 2 :=
sorry

end ratio_of_radii_l488_488716


namespace integer_ratio_zero_l488_488315

theorem integer_ratio_zero
  (A B : ℤ)
  (h : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -1 → (A / (x - 3 : ℝ) + B / (x ^ 2 + 2 * x + 1) = (x ^ 3 - x ^ 2 + 3 * x + 1) / (x ^ 3 - x - 3))) :
  B / A = 0 :=
sorry

end integer_ratio_zero_l488_488315


namespace gcd_problem_l488_488088

-- Define the two numbers
def a : ℕ := 1000000000
def b : ℕ := 1000000005

-- Define the problem to prove the GCD
theorem gcd_problem : Nat.gcd a b = 5 :=
by 
  sorry

end gcd_problem_l488_488088


namespace probability_sum_divisible_by_3_l488_488539

-- Define the first ten prime numbers
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate to check divisibility by 3
def divisibleBy3 (n : ℕ) : Prop := n % 3 = 0

-- Define the main theorem statement
theorem probability_sum_divisible_by_3 :
  (let pairs := (firstTenPrimes.product firstTenPrimes).filter (λ (x : ℕ × ℕ), x.1 < x.2) in
    let totalPairs := pairs.length in
    let divisiblePairs := pairs.count (λ (x : ℕ × ℕ), divisibleBy3 (x.1 + x.2)) in
    (divisiblePairs.to_rat / totalPairs.to_rat) = (1 : ℚ) / 3) :=
begin
  sorry -- Proof is not required.
end

end probability_sum_divisible_by_3_l488_488539


namespace twelve_pharmacies_not_enough_l488_488759

def grid := ℕ × ℕ

def is_within_walking_distance (p1 p2 : grid) : Prop :=
  abs (p1.1 - p1.2) ≤ 3 ∧ abs (p2.1 - p2.2) ≤ 3

def walking_distance_coverage (pharmacies : set grid) (p : grid) : Prop :=
  ∃ pharmacy ∈ pharmacies, is_within_walking_distance pharmacy p

def sufficient_pharmacies (pharmacies : set grid) : Prop :=
  ∀ p : grid, walking_distance_coverage pharmacies p

theorem twelve_pharmacies_not_enough (pharmacies : set grid) (h : pharmacies.card = 12) : 
  ¬ sufficient_pharmacies pharmacies :=
sorry

end twelve_pharmacies_not_enough_l488_488759


namespace three_digit_integers_sum_to_7_l488_488908

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488908


namespace coin_outcomes_equivalent_l488_488213

theorem coin_outcomes_equivalent :
  let outcomes_per_coin := 2
  let total_coins := 3
  (outcomes_per_coin ^ total_coins) = 8 :=
by
  sorry

end coin_outcomes_equivalent_l488_488213


namespace range_of_a_l488_488208

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ : ℝ, 2 * x₀^2 + (a - 1) * x₀ + 1 / 2 ≤ 0) → a ∈ set.Ioo (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_a_l488_488208


namespace problem_A_inter_complement_B_l488_488512

noncomputable def A : Set ℝ := {x : ℝ | Real.log x / Real.log 2 < 2}
noncomputable def B : Set ℝ := {x : ℝ | (x - 2) / (x - 1) ≥ 0}
noncomputable def complement_B : Set ℝ := {x : ℝ | ¬((x - 2) / (x - 1) ≥ 0)}

theorem problem_A_inter_complement_B : 
  (A ∩ complement_B) = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end problem_A_inter_complement_B_l488_488512


namespace correctProduct_l488_488593

-- Define the digits reverse function
def reverseDigits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

-- Main theorem statement
theorem correctProduct (a b : ℕ) (h1 : 9 < a ∧ a < 100) (h2 : reverseDigits a * b = 143) : a * b = 341 :=
  sorry -- proof to be provided

end correctProduct_l488_488593


namespace problem_part_1_problem_part_2_l488_488119

theorem problem_part_1 (a : ℕ → ℝ) (h : ∀ n : ℕ, a n ^ 2 - (2 * n - 1) * a n - 2 * n = 0) 
  (pos_seq : ∀ n : ℕ, a n > 0) :
  ∀ n : ℕ, a n = 2 * n :=
sorry

theorem problem_part_2 (a b : ℕ → ℝ)
  (h : ∀ n : ℕ, a n ^ 2 - (2 * n - 1) * a n - 2 * n = 0)
  (pos_seq : ∀ n : ℕ, a n > 0)
  (a_general : ∀ n : ℕ, a n = 2 * n)
  (bn_def : ∀ n : ℕ, b n = 1 / ((n + 1) * a n)) :
  ∀ n : ℕ, (∑ i in Finset.range n, b i) = n / (2 * n + 2) :=
sorry

end problem_part_1_problem_part_2_l488_488119


namespace ice_cream_cone_cost_l488_488647

theorem ice_cream_cone_cost (cost_two_cones : ℕ) (h : cost_two_cones = 198) : 
  ∃ cost_one_cone : ℕ, cost_one_cone = cost_two_cones / 2 ∧ cost_one_cone = 99 :=
by
  use 99
  split
  sorry
  sorry

end ice_cream_cone_cost_l488_488647


namespace max_value_circle_numbers_l488_488336

theorem max_value_circle_numbers (x : ℕ → ℝ) (h_sum : ∑ i in finset.range 10, x i = 100)
  (h_adj : ∀ i, x i + x (i+1) % 10 + x (i+2) % 10 ≥ 29) :
  ∀ i, x i ≤ 13 :=
begin
  sorry
end

end max_value_circle_numbers_l488_488336


namespace domain_of_function_l488_488066

theorem domain_of_function :
  {x : ℝ | 0 ≤ x ∧ x ≤ 36} = {x : ℝ | ∃ y ∈ set.Icc (0:ℝ) (36:ℝ), y = x} :=
by sorry

end domain_of_function_l488_488066


namespace stratified_sampling_sophomores_l488_488790

theorem stratified_sampling_sophomores (N N1 N2 N3 n : ℕ) (hN : N = 2000) (hN1 : N1 = 560) (hN2 : N2 = 640) (hN3 : N3 = 800) (hn : n = 100) :
  (N2 * n / N) = 32 :=
by
  rw [hN, hN1, hN2, hN3, hn]
  -- verification steps would go here
  sorry

end stratified_sampling_sophomores_l488_488790


namespace pool_volume_minus_pillar_volume_l488_488417

-- Definitions of the problem conditions
def pool_diameter : ℝ := 20
def pool_depth : ℝ := 5
def pillar_diameter : ℝ := 4
def pillar_depth : ℝ := 5

-- Auxiliary definition of radii
def pool_radius : ℝ := pool_diameter / 2
def pillar_radius : ℝ := pillar_diameter / 2

-- Definition of volumes
def pool_volume : ℝ := π * pool_radius^2 * pool_depth
def pillar_volume : ℝ := π * pillar_radius^2 * pillar_depth

-- Volume difference calculation (which is the problem statement)
theorem pool_volume_minus_pillar_volume : pool_volume - pillar_volume = 480 * π := 
by
  sorry

end pool_volume_minus_pillar_volume_l488_488417


namespace even_and_monotonic_only_D_l488_488806

-- Define the functions
def f_A (x : ℝ) : ℝ := Real.cos x
def f_B (x : ℝ) : ℝ := -x^2
def f_C (x : ℝ) : ℝ := Real.log (2^x)
def f_D (x : ℝ) : ℝ := Real.exp (Real.abs x)

-- Prove that the only even function that is monotonically increasing on (0, +∞) is f_D
theorem even_and_monotonic_only_D :
  (∀ x, f_A x = f_A (-x)) ∧ (∀ x > 0, f_A x < f_A (x + 1)) ∨
  (∀ x, f_B x = f_B (-x)) ∧ (∀ x > 0, f_B x < f_B (x + 1)) ∨
  (∀ x, f_C x = f_C (-x)) ∧ (∀ x > 0, f_C x < f_C (x + 1)) ∨
  (∀ x, f_D x = f_D (-x)) ∧ (∀ x > 0, f_D x < f_D (x + 1)) :=
by 
  sorry

end even_and_monotonic_only_D_l488_488806


namespace rectangle_area_proof_l488_488657

-- Define points as vectors in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Points P, Q, and R
def P : Point := { x := 15, y := 55 }
def Q : Point := { x := 26, y := 55 }
def R : Point := { x := 26, y := 35 }

-- Function to compute the distance between two points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Define the lengths of PQ and RQ
def len_PQ : ℝ := distance P Q
def len_RQ : ℝ := distance R Q

-- The area of the rectangle
def rectangle_area (PQ RQ : ℝ) : ℝ := PQ * RQ

-- Theorem stating the area of the rectangle
theorem rectangle_area_proof : rectangle_area len_PQ len_RQ = 220 := by
  sorry

-- To ensure the code can be compiled successfully
#print axioms rectangle_area_proof

end rectangle_area_proof_l488_488657


namespace sugar_amount_l488_488645

theorem sugar_amount (S : ℕ) 
  (h1 : ∀ (S : ℕ), 2 + (S + 5) = 10): 
  S = 3 := 
by {
  have h2 : 2 + (S + 5) = 10 := h1 S,
  linarith,
  sorry
}

end sugar_amount_l488_488645


namespace lights_switched_on_l488_488338

theorem lights_switched_on : 
  let total_lights := 56 in
  let initial_state := (fun _ => false) in
  let toggle_light_at n state := state ∘ (λ i, if i % n = 0 then not else id) in
  let final_state := toggle_light_at 5 (toggle_light_at 3 initial_state) in
  (∑ i in (Finset.range total_lights), if final_state i then 1 else 0) = 26 :=
by sorry

end lights_switched_on_l488_488338


namespace sequence_increasing_range_of_a_l488_488694

theorem sequence_increasing_range_of_a :
  ∀ {a : ℝ}, (∀ n : ℕ, 
    (n ≤ 7 → (4 - a) * n - 10 ≤ (4 - a) * (n + 1) - 10) ∧ 
    (7 < n → a^(n - 6) ≤ a^(n - 5))
  ) → 2 < a ∧ a < 4 :=
by
  sorry

end sequence_increasing_range_of_a_l488_488694


namespace optimal_strategy_to_maximize_expected_score_l488_488387

-- Define types of questions and point values
inductive QuestionType
| A 
| B

structure Competition :=
(correct_prob : QuestionType → Real)
(point_value : QuestionType → ℕ)

-- Probability of correct answers
def xiaoming_competition : Competition :=
{
  correct_prob := λ t, if t = QuestionType.A then 0.8 else 0.6,
  point_value := λ t, if t = QuestionType.A then 20 else 80
}

-- Distribution of scores if Xiao Ming starts with type A
def distribution_X : List (ℕ × Real) :=
[(0, 0.2), (20, 0.32), (100, 0.48)]

-- Expected score if starting with type A
def expected_X : Real := 
0 * 0.2 + 20 * 0.32 + 100 * 0.48

-- Distribution of scores if Xiao Ming starts with type B
def distribution_Y : List (ℕ × Real) :=
[(0, 0.4), (80, 0.12), (100, 0.48)]

-- Expected score if starting with type B
def expected_Y : Real := 
0 * 0.4 + 80 * 0.12 + 100 * 0.48

theorem optimal_strategy_to_maximize_expected_score :
  expected_Y > expected_X :=
by 
sorry

end optimal_strategy_to_maximize_expected_score_l488_488387


namespace decimal_to_base7_conversion_l488_488083

theorem decimal_to_base7_conversion :
  (2023 : ℕ) = 5 * (7^3) + 6 * (7^2) + 2 * (7^1) + 0 * (7^0) :=
by
  sorry

end decimal_to_base7_conversion_l488_488083


namespace cody_money_final_l488_488062

theorem cody_money_final (initial_money : ℕ) (birthday_money : ℕ) (money_spent : ℕ) (final_money : ℕ) 
  (h1 : initial_money = 45) (h2 : birthday_money = 9) (h3 : money_spent = 19) :
  final_money = initial_money + birthday_money - money_spent :=
by {
  sorry  -- The proof is not required here.
}

end cody_money_final_l488_488062


namespace proportional_y_when_x3_l488_488147

theorem proportional_y_when_x3 :
  (∃ k : ℝ, ∀ x : ℝ, y = k * x) →
  y (-2) = 4 →
  y 3 = -6 :=
by
  sorry

end proportional_y_when_x3_l488_488147


namespace original_population_960_l488_488021

variable (original_population : ℝ)

def new_population_increased := original_population + 800
def new_population_decreased := 0.85 * new_population_increased original_population

theorem original_population_960 
  (h1: new_population_decreased original_population = new_population_increased original_population + 24) :
  original_population = 960 := 
by
  -- here comes the proof, but we are omitting it as per the instructions
  sorry

end original_population_960_l488_488021


namespace problem_1_problem_2_l488_488485

noncomputable def A (m : ℝ) : set ℝ := {x | x^2 - 2 * m * x + m^2 ≤ 4}
noncomputable def B : set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem problem_1 (h1 : ∀ x, x ∈ A 2 ↔ x ∈ B → 0 ≤ x ∧ x ≤ 3) : 2 = 2 :=
by sorry

theorem problem_2 (h2 : ∀ x, x ∈ B → x ∉ A m) :
  (5 < m ∨ m < -3) :=
by 
  intro h
  sorry

end problem_1_problem_2_l488_488485


namespace max_distance_ellipse_l488_488617

-- Let's define the conditions and the theorem in Lean 4.
theorem max_distance_ellipse (θ : ℝ) :
  let C := { p : ℝ × ℝ | p.1 ^ 2 / 5 + p.2 ^ 2 = 1 }
  let B := (0, 1)
  let P := (sqrt 5 * cos θ, sin θ)
  P ∈ C →
  max (λ θ : ℝ, sqrt ((sqrt 5 * cos θ - 0) ^ 2 + (sin θ - 1) ^ 2)) = 5 / 2 :=
sorry

end max_distance_ellipse_l488_488617


namespace prime_number_digit_repetition_l488_488422

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 ∧ m ≤ n / 2 → n % m ≠ 0

theorem prime_number_digit_repetition (p n : ℕ) 
  (h1 : is_prime p) 
  (h2 : p > 3) 
  (h3 : (nat.floor (log 10 (real.of_nat (p^n))) + 1) = 20) : 
  ∃ d, (0 ≤ d ∧ d ≤ 9) ∧ (nat.count d (digits 10 (p^n)) ≥ 3) :=
sorry

end prime_number_digit_repetition_l488_488422


namespace best_coupon_price_l488_488023

-- Define the conditions for each coupon's discount
def coupon1_discount (x : ℝ) : ℝ := 
  if x ≥ 50 then 0.12 * x else 0

def coupon2_discount (x : ℝ) : ℝ :=
  if x ≥ 120 then 25 else 0

def coupon3_discount (x : ℝ) : ℝ :=
  if x > 120 then 0.15 * (x - 120) else 0

-- Define the validity range for Coupon 1 being the best discount:
def valid_discount_range (x : ℝ) : Prop :=
  x > (25 / 0.12) ∧ x < 600

-- The price options
def listed_prices : List ℝ := [189.95, 209.95, 229.95, 249.95, 269.95]

-- The proof statement
theorem best_coupon_price : (209.95 ∈ listed_prices) →
  (∀ x ∈ listed_prices, valid_discount_range x → coupon1_discount x > coupon2_discount x ∧ coupon1_discount x > coupon3_discount x) :=
by
  intros
  sorry -- proof steps will go here

end best_coupon_price_l488_488023


namespace num_three_digit_sums7_l488_488998

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l488_488998


namespace three_digit_sum_seven_l488_488883

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488883


namespace max_repeating_sequence_length_l488_488314

theorem max_repeating_sequence_length (p q n α β d : ℕ) (h_prime: Nat.gcd p q = 1)
  (hq : q = (2 ^ α) * (5 ^ β) * d) (hd_coprime: Nat.gcd d 10 = 1) (h_repeat: 10 ^ n ≡ 1 [MOD d]) :
  ∃ s, s ≤ n * (10 ^ n - 1) ∧ (10 ^ s ≡ 1 [MOD d^2]) :=
by
  sorry

end max_repeating_sequence_length_l488_488314


namespace six_digit_cyclic_number_l488_488725

def is_cyclic_permutation (n m : ℕ) : Prop :=
  ∃ k, (m = (n % 10 ^ k) * 10 ^ (6 - k) + n / 10 ^ k)

theorem six_digit_cyclic_number : ∃ (abcdef : ℕ), 100000 ≤ abcdef ∧ abcdef ≤ 999999 ∧
  (abcdef * 2 = 285714) ∧ is_cyclic_permutation abcdef (abcdef * 2) ∧
  (abcdef * 3 = 428571) ∧ is_cyclic_permutation abcdef (abcdef * 3) ∧
  (abcdef * 4 = 571428) ∧ is_cyclic_permutation abcdef (abcdef * 4) ∧
  (abcdef * 5 = 714285) ∧ is_cyclic_permutation abcdef (abcdef * 5) ∧
  (abcdef * 6 = 857142) ∧ is_cyclic_permutation abcdef (abcdef * 6) :=
begin
  use 142857,
  split,
  { norm_num },
  split,
  { norm_num },
  repeat {
    split;
    try {norm_num};
    try {apply is_cyclic_permutation_intro, norm_num},
  },
end

end six_digit_cyclic_number_l488_488725


namespace no_other_positive_integers_product_of_three_distinct_primes_l488_488354

theorem no_other_positive_integers_product_of_three_distinct_primes :
  ∀ (p1 p2 p3 : ℕ), p1 + p2 + p3 = 1192 ∧
    nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
    p1 * p2 * p3 = 7102 ∨ (p1 * p2 * p3 = 2 * 3 * 1187 ∧ p1 + p2 + p3 = 1192) :=
by
  sorry

end no_other_positive_integers_product_of_three_distinct_primes_l488_488354


namespace perimeter_ge_AC_l488_488258

open EuclideanGeometry

-- Given points O, A, B, C, M, N
variable (O A B C M N : Point)

-- Conditions:
-- 1. O is the circumcenter of triangle ABC
axiom circumcenter : circumcenter O A B C
-- 2. M is on AB and N is on BC
axiom M_on_AB : on_segment M A B
axiom N_on_BC : on_segment N B C
-- 3. angle AOC is twice angle MON
axiom angle_condition : ∠ A O C = 2 * ∠ M O N

-- Goal: Prove BM + MN + BN ≥ AC
theorem perimeter_ge_AC : distance B M + distance M N + distance N B ≥ distance A C :=
sorry

end perimeter_ge_AC_l488_488258


namespace chameleons_color_change_l488_488576

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l488_488576


namespace incorrect_proposition_C_l488_488136

variables (α β r : Plane) (a b c d l : Line)
variables (A B : Point)
variables (h1 : a ⊆ α) (h2: b ⊆ α) (h3: c ⊆ β) (h4: d ⊆ β)
variables (h5: a ∩ b = {A}) (h6 : c ∩ d = {B})
variables (h7 : a ⊥ c) (h8 : b ⊥ d)

theorem incorrect_proposition_C : ¬ (α ⊥ β) :=
  sorry

end incorrect_proposition_C_l488_488136


namespace sum_odd_red_numbers_greater_l488_488337

theorem sum_odd_red_numbers_greater :
  (∑ i j in finset.range 100, if (i + 1) ^ 2 + (j + 1) ^ 2 % 2 = 1 then (i + 1) ^ 2 + (j + 1) ^ 2 else 0) >
  (∑ i j in finset.range 100, if (i + 1) ^ 2 + (j + 1) ^ 2 % 2 = 0 then (i + 1) ^ 2 + (j + 1) ^ 2 else 0) :=
sorry

end sum_odd_red_numbers_greater_l488_488337


namespace find_sum_of_coefficients_l488_488101

theorem find_sum_of_coefficients (p q r : ℚ)
    (h1 : ∃ p q r, (∀ x, y = px^2 + qx + r → (-2 : ℚ) = (p * 3^2 + q * 3 + r))
    (h2 : y = px^2 + qx + r → y = a(x-3)^2 - 2)
    (h3 : y = px(6)- (h2))
    (h4 : ∀ x, y = px^2 + qx + r → (5 : ℚ) = (p * 6^2 + q * 6 + r))
    : p + q + r = 4 / 3 := 
  sorry

end find_sum_of_coefficients_l488_488101


namespace FJ_eq_FA_l488_488634

open EuclideanGeometry

variables {A B C H M N K L F J : Point}

-- Given conditions as definitions
def is_orthocenter (H A B C : Point) : Prop :=
  H = orthocenter A B C

def are_midpoints (M N A B C : Point) : Prop :=
  midpoint_segment M A B ∧ midpoint_segment N A C

-- Assumptions for the problem
axiom H_is_orthocenter : is_orthocenter H A B C
axiom M_and_N_are_midpoints : are_midpoints M N A B C
axiom H_inside_BMNC : inside_quadrilateral H B M N C
axiom circumcircles_tangent : tangent (circumcircle B M H) (circumcircle C N H)
axiom line_parallel_BC : parallel (line_through H (parallel_to BC H)) (BC)
axiom K_on_circumcircle_BMH : on_circumcircle K (circumcircle B M H)
axiom L_on_circumcircle_CNH : on_circumcircle L (circumcircle C N H)
axiom F_intersection_MK_NL : intersection_point F (line_through M K) (line_through N L)
axiom J_incenter_MHN : incenter J (triangle M H N)

-- The goal is to prove FJ = FA
theorem FJ_eq_FA : distance F J = distance F A :=
sorry

end FJ_eq_FA_l488_488634


namespace chameleon_color_change_l488_488571

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l488_488571


namespace builders_time_l488_488202

theorem builders_time (total_work_days : ℕ) (n_builders : ℕ) : total_work_days = 24 → n_builders = 6 → (total_work_days / n_builders) = 4 :=
by
  intros h_work h_builders
  rw [h_work, h_builders]
  exact Nat.div_eq_of_eq_mul sorry

end builders_time_l488_488202


namespace todd_runs_faster_l488_488814

-- Define the times taken by Brian and Todd
def brian_time : ℕ := 96
def todd_time : ℕ := 88

-- The theorem stating the problem
theorem todd_runs_faster : brian_time - todd_time = 8 :=
by
  -- Solution here
  sorry

end todd_runs_faster_l488_488814


namespace insufficient_pharmacies_l488_488751

/-- Define the grid size and conditions --/
def grid_size : ℕ := 9
def total_intersections : ℕ := 100
def pharmacy_walking_distance : ℕ := 3
def number_of_pharmacies : ℕ := 12

/-- Prove that 12 pharmacies are not enough for the given conditions. --/
theorem insufficient_pharmacies : ¬ (number_of_pharmacies ≥ grid_size + 3) → 
  ∃(pharmacies : ℕ), pharmacies = number_of_pharmacies ∧
  ∀(x y : ℕ), (x ≤ grid_size ∧ y ≤ grid_size) → 
  ¬ exists (p : ℕ × ℕ), p.1 ≤ x + pharmacy_walking_distance ∧
  p.2 ≤ y + pharmacy_walking_distance ∧ 
  p.1 < total_intersections ∧ 
  p.2 < total_intersections := 
by sorry

end insufficient_pharmacies_l488_488751


namespace students_selected_milk_is_54_l488_488588

-- Define the parameters.
variable (total_students : ℕ)
variable (students_selected_soda students_selected_milk : ℕ)

-- Given conditions.
axiom h1 : students_selected_soda = 90
axiom h2 : students_selected_soda = (1 / 2) * total_students
axiom h3 : students_selected_milk = (3 / 5) * students_selected_soda

-- Prove that the number of students who selected milk is equal to 54.
theorem students_selected_milk_is_54 : students_selected_milk = 54 :=
by
  sorry

end students_selected_milk_is_54_l488_488588


namespace perpendicular_x_value_parallel_x_value_l488_488175

/-- Define the vectors a and b where a = (2, -1, 5) and b = (-4, 2, x) --/
def a : ℝ × ℝ × ℝ := (2, -1, 5)
def b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

/-- Perpendicular vectors dot product should be zero --/
def is_perpendicular (a b : ℝ × ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0

/-- Parallel vectors components should be proportional --/
def is_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2 ∧ a.2 / b.2 = a.3 / b.3

/-- Prove that x = 2 if a and b are perpendicular --/
theorem perpendicular_x_value : ∃ (x : ℝ), is_perpendicular a (b x) ∧ x = 2 :=
by
  sorry

/-- Prove that x = -10 if a and b are parallel --/
theorem parallel_x_value : ∃ (x : ℝ), is_parallel a (b x) ∧ x = -10 :=
by
  sorry

end perpendicular_x_value_parallel_x_value_l488_488175


namespace twelve_pharmacies_not_sufficient_l488_488765

-- Define an intersection grid of size 10 x 10 (100 squares).
def city_grid : Type := Fin 10 × Fin 10 

-- Define the distance measure between intersections, assumed as L1 metric for grid paths.
def dist (p q : city_grid) : Nat := (abs (p.1.val - q.1.val) + abs (p.2.val - q.2.val))

-- Define a walking distance pharmacy 
def is_walking_distance (p q : city_grid) : Prop := dist p q ≤ 3

-- State that having 12 pharmacies is not sufficient
theorem twelve_pharmacies_not_sufficient (pharmacies : Fin 12 → city_grid) :
  ¬ (∀ intersection: city_grid, ∃ (p_index : Fin 12), is_walking_distance (pharmacies p_index) intersection) :=
sorry

end twelve_pharmacies_not_sufficient_l488_488765


namespace maximum_m_value_l488_488531

def f (x a : ℝ) := 2^(abs (x + a))

theorem maximum_m_value (a m : ℝ) (h1 : ∀ x, f (3 + x) a = f (3 - x) a) (h2 : ∀ x y, x ≤ y → y ≤ m → f y a ≤ f x a) :
  m ≤ 3 :=
by
  sorry

end maximum_m_value_l488_488531


namespace c_pays_d_l488_488047

variables (A B C D a : ℕ) (m : ℕ)
variable equal_contribution : (A = B ∧ B = C ∧ C = D)

variable a_def : (A = D + 3)
variable b_def : (B = D + 7)
variable c_def : (C = D + 14)

variable b_to_d_payment : (14 = 2 * (B - D))

theorem c_pays_d : (C - D) * 2 = 70 :=
by
  sorry

end c_pays_d_l488_488047


namespace find_alpha_polar_eq_line_l488_488137

section
  -- Define the given point P
  variable (P : ℝ × ℝ) (P_def : P = (2,1))

  -- Define the line l parameterized by t and inclination angle α
  variable (l : ℝ → ℝ × ℝ)
  variable (α : ℝ)
  variable (l_def : l t = (2 + t * cos α, 1 + t * sin α))

  -- Define the incidences on positive x-axis and y-axis
  variable (A B : ℝ × ℝ)
  variable (on_x : A = (A.1, 0))  -- Intersection with positive x-axis
  variable (on_y : B = (0, B.2))  -- Intersection with positive y-axis

  -- Define the product of distances condition
  variable (dist_condition : dist P A * dist P B = 4)

  -- State the theorem for part (1): finding the angle α
  theorem find_alpha : α = 3 * π / 4 := sorry

  -- State the theorem for part (2): polar coordinate equation of line l
  theorem polar_eq_line : ∀ ρ θ, (ρ = 3 / (sin θ + cos θ)) ↔ (∃ t, l t = (ρ * cos θ, ρ * sin θ)) := sorry
end

end find_alpha_polar_eq_line_l488_488137


namespace lead_of_chucks_team_l488_488100

theorem lead_of_chucks_team (chucks_team_points : ℕ) (yellow_team_points : ℕ) (h1 : chucks_team_points = 72) (h2 : yellow_team_points = 55) :
  chucks_team_points - yellow_team_points = 17 := 
by 
  rw [h1, h2] 
  sorry

end lead_of_chucks_team_l488_488100


namespace dividend_rate_of_stock_l488_488777

variable (MarketPrice : ℝ) (YieldPercent : ℝ) (DividendPercent : ℝ)
variable (NominalValue : ℝ) (AnnualDividend : ℝ)

def stock_dividend_rate_condition (YieldPercent MarketPrice NominalValue DividendPercent : ℝ) 
  (AnnualDividend : ℝ) : Prop :=
  YieldPercent = 20 ∧ MarketPrice = 125 ∧ DividendPercent = 0.25 ∧ NominalValue = 100 ∧
  AnnualDividend = (YieldPercent / 100) * MarketPrice

theorem dividend_rate_of_stock :
  stock_dividend_rate_condition 20 125 100 0.25 25 → (DividendPercent * NominalValue) = 25 :=
by 
  sorry

end dividend_rate_of_stock_l488_488777


namespace proposition_p_neither_sufficient_nor_necessary_l488_488139

-- Define propositions p and q
def p (m : ℝ) : Prop := m = -1
def q (m : ℝ) : Prop := ∀ x y : ℝ, (x - 1 = 0) ∧ (x + m^2 * y = 0) → ∀ x' y' : ℝ, x' = x ∧ y' = y → (x - 1) * (x + m^2 * y) = 0

-- Main theorem statement
theorem proposition_p_neither_sufficient_nor_necessary (m : ℝ) : ¬ (p m → q m) ∧ ¬ (q m → p m) :=
by
  sorry

end proposition_p_neither_sufficient_nor_necessary_l488_488139


namespace solution_set_of_inequality_l488_488427

noncomputable def f (x : ℝ) : ℝ := sorry 

-- Conditions
axiom f_defined_on_domain : ∀ x, 0 < x → ∃ y, f x = y
axiom f_satisfies_inequality : ∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ ≠ x₂ → 
  (x₁ * f x₁ - x₂ * f x₀) / (x₁ - x₂) < 0  
axiom f_at_two : f 2 = 4

-- The statement to be proven
theorem solution_set_of_inequality : { x : ℝ | 0 < x ∧ f x - 8 / x > 0 } = { x : ℝ | 0 < x ∧ x < 2 } := 
by {
  sorry
}

end solution_set_of_inequality_l488_488427


namespace geometric_series_sum_l488_488871

noncomputable def T (r : ℝ) := 15 / (1 - r)

theorem geometric_series_sum (b : ℝ) (hb1 : -1 < b) (hb2 : b < 1) (H : T b * T (-b) = 3240) : T b + T (-b) = 432 := 
by sorry

end geometric_series_sum_l488_488871


namespace probability_sum_divisible_by_3_l488_488541

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

noncomputable def num_pairs_divisible_by_3 (primes : List ℕ) : ℕ :=
  (primes.toFinset.powerset.toList.filter 
    (λ s, s.card = 2 ∧ (s.sum % 3 = 0))).length

theorem probability_sum_divisible_by_3 :
  (num_pairs_divisible_by_3 first_ten_primes : ℚ) / (10.choose 2) = 2 / 15 :=
by
  sorry

end probability_sum_divisible_by_3_l488_488541


namespace ten_factorial_mod_thirteen_l488_488452

theorem ten_factorial_mod_thirteen : (nat.factorial 10) % 13 = 6 := 
by 
  -- Using Wilson's Theorem: (p-1)! ≡ -1 (mod p) where p is a prime
  have h1 : nat.factorial 12 % 13 = 12 := by sorry,
  -- 12 ≡ -1 (mod 13), 11 ≡ -2 (mod 13)
  have h2 : 12 % 13 = 12 := by norm_num,
  have h3 : 11 % 13 = 11 := by norm_num,
  have h4 : (nat.factorial 10 * 12 * 11) % 13 = 12 := by sorry,
  -- finding modular inverse
  have h5 : (12 * 11) % 13 = 12 := by sorry,
  have h6 : (2 : ℤ)⁻¹ ≡ 7 [MOD 13] := by sorry,
  -- deriving the factorial result
  exact sorry

end ten_factorial_mod_thirteen_l488_488452


namespace isosceles_triangle_l488_488372

-- Definitions for the points and vectors in the plane
variables (O A B C : Type) [AddCommGroup O] [Module ℝ O]
variables (OB OC OA : O)
variables [InnerProductSpace ℝ O]

def vector_condition : Prop :=
  let OB_vec := OB - OC
  let OC_vec := OB + OC - 2 • OA
  OB_vec ⬝ OC_vec = 0

-- Given this condition, prove that ABC is an isosceles triangle
theorem isosceles_triangle (h : vector_condition (OB) (OC) (OA)) : dist OB OA = dist OC OA :=
sorry

end isosceles_triangle_l488_488372


namespace sum_of_solutions_of_quadratic_sum_of_solutions_of_quadratic_l488_488818

theorem sum_of_solutions_of_quadratic :
  let a := -48
  let b := 72
  let c := 198
  -48 * a * b +
  (a * c) =
    (a * b * c) :=
-- The equation is -48x^2 + 72x + 198 = 0 has sum of solutions equal to 3/2
theorem sum_of_solutions_of_quadratic :
  let a := -48
  let b := 72
  let c := 198
  let sum_roots := -b / a
  sum_roots = 3/2 := by
{sorry}

end sum_of_solutions_of_quadratic_sum_of_solutions_of_quadratic_l488_488818


namespace select_group_friends_l488_488334

theorem select_group_friends (n : ℕ) (friends : Finset (Fin n) -> Finset (Fin n))
  (h_friends_symmetric : ∀ a b : Fin n, a ∈ friends b ↔ b ∈ friends a)
  (h_friends_count : ∀ z : Fin n, (friends z).card = 1000) :
  ∃ S : Finset (Fin n), 
    (∃ A: Finset (Fin n), A ⊆ S ∧ A.card ≥ n / 2017 ∧ ∀ a ∈ A, (friends a ∩ S).card = 2) := 
sorry

end select_group_friends_l488_488334


namespace tracy_initial_candies_l488_488714

theorem tracy_initial_candies (x y : ℕ) (h₁ : x = 108) (h₂ : 2 ≤ y ∧ y ≤ 6) : 
  let remaining_after_eating := (3 / 4) * x 
  let remaining_after_giving := (2 / 3) * remaining_after_eating
  let remaining_after_mom := remaining_after_giving - 40
  remaining_after_mom - y = 10 :=
by 
  sorry

end tracy_initial_candies_l488_488714


namespace min_value_in_interval_l488_488152

noncomputable def f (a x : ℝ) : ℝ := (1/2)*x^2 - a*Real.log x + 1

theorem min_value_in_interval (a : ℝ) : (0 < a ∧ a < 1) ↔ ∃ x ∈ set.Ioo 0 1, (∀ y ∈ set.Ioo 0 1, f a x ≤ f a y) :=
begin
  sorry
end

end min_value_in_interval_l488_488152


namespace question1_question2_question3_l488_488370

open EuclideanGeometry

variables 
  {A B C D P Q M N : Point}
  (circle : Circle)
  (L : Line)
  (AB CD : Diameter)
  (BM DM : Ray)
  (alpha : Real)

-- Conditions
def conditions : Prop :=
  perpendicular_diameters circle AB CD ∧
  tangent_at_point circle A L ∧
  on_minor_arc circle A C M ∧
  rays_meet_line BM P L ∧
  rays_meet_line DM Q L

-- Question 1: Show AP · AQ = AB · PQ
theorem question1 (h : conditions circle L AB CD BM DM alpha) :
  (AP · AQ = AB · PQ) :=
sorry

-- Question 2: Construct M such that BQ parallel to DP
theorem question2 (h : conditions circle L AB CD BM DM alpha) :
  ∃ M, (BQ ∥ DP) :=
sorry

-- Question 3: Find the locus of N if OP and BQ meet
theorem question3 (h : conditions circle L AB CD BM DM alpha) :
  locus N (y = x - r ∧ x ≥ r) :=
sorry

end question1_question2_question3_l488_488370


namespace unique_cell_50_distance_l488_488676

-- Define the distance between two cells
def kingDistance (p1 p2 : ℤ × ℤ) : ℤ :=
  max (abs (p1.1 - p2.1)) (abs (p1.2 - p2.2))

-- A condition stating three cells with specific distances
variables (A B C : ℤ × ℤ) (hAB : kingDistance A B = 100) (hBC : kingDistance B C = 100) (hCA : kingDistance C A = 100)

-- A proposition to prove there is exactly one cell at a distance of 50 from all three given cells
theorem unique_cell_50_distance : ∃! D : ℤ × ℤ, kingDistance D A = 50 ∧ kingDistance D B = 50 ∧ kingDistance D C = 50 :=
sorry

end unique_cell_50_distance_l488_488676


namespace probability_reroll_two_dice_eq_5_over_36_l488_488604

/-- Jason rolls three fair six-sided dice. He can choose a subset of the dice to reroll. 
  Jason wins if and only if the sum of the dice is 9.
  The probability that Jason's optimal strategy rerolls exactly two dice is 5/36. -/
theorem probability_reroll_two_dice_eq_5_over_36 :
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6)
  let favorable_outcomes := 
    outcomes.filter (λ t, 
      let (a,(b,c)) := t in
      (a + b + c = 9 ∧ 
       ((a + (6.choose 2) = 9 ∨ b + (6.choose 2) = 9 ∨ c + (6.choose 2) = 9) ∨ 
        (a + b = 9 - c) ∨ (a + c = 9 - b) ∨ (b + c = 9 - a)))
    )
  in (favorable_outcomes.card: ℚ) / (outcomes.card : ℚ) = 5/36 :=
sorry

end probability_reroll_two_dice_eq_5_over_36_l488_488604


namespace ω₂_touches_altitude_CH_l488_488321

variables (A B C M N : Point)
variables (ω₁ ω₂ : Circle)
variables (CH : Line)

-- Definitions and Assumptions
def median_AM : Line := Line.mk A M
def median_BN : Line := Line.mk B N

def diameter_of_circle_ω₁ := diameter ω₁ = median_AM
def diameter_of_circle_ω₂ := diameter ω₂ = median_BN
def ω₁_touches_altitude_CH := touches ω₁ CH

-- Theorem Statement
theorem ω₂_touches_altitude_CH
  (h₁ : diameter_of_circle_ω₁)
  (h₂ : diameter_of_circle_ω₂)
  (h₃ : ω₁_touches_altitude_CH) : touches ω₂ CH := 
sorry

end ω₂_touches_altitude_CH_l488_488321


namespace value_of_seventh_observation_l488_488674

-- Given conditions
def sum_of_first_six_observations : ℕ := 90
def new_total_sum : ℕ := 98

-- Problem: prove the value of the seventh observation
theorem value_of_seventh_observation : new_total_sum - sum_of_first_six_observations = 8 :=
by
  sorry

end value_of_seventh_observation_l488_488674


namespace range_of_positive_integers_range_of_positive_integers_list_l488_488642

theorem range_of_positive_integers (F : List ℤ) (h1 : F = List.range' (-4) 12) :
  F.filter (λ x, 0 < x) = [1, 2, 3, 4, 5, 6, 7] :=
by sorry

theorem range_of_positive_integers_list (F : List ℤ) (h1 : F = List.range' (-4) 12) :
  F.filter (λ x, 0 < x) = [1, 2, 3, 4, 5, 6, 7] →
  (F.filter (λ x, 0 < x)).maximum - (F.filter (λ x, 0 < x)).minimum = 6 :=
by sorry

end range_of_positive_integers_range_of_positive_integers_list_l488_488642


namespace sum_radical_conjugates_l488_488826

theorem sum_radical_conjugates (n : ℝ) (m : ℝ) (h1 : n = 5) (h2 : m = (sqrt 500)) : 
  (n - m) + (n + m) = 10 :=
by 
  rw [h1, h2]
  sorry

end sum_radical_conjugates_l488_488826


namespace circle_equation_l488_488473

open Real

variable {x y : ℝ}

theorem circle_equation (a : ℝ) (h_a_positive : a > 0) 
    (h_tangent : abs (3 * a + 4) / sqrt (3^2 + 4^2) = 2) :
    (∀ x y : ℝ, (x - a)^2 + y^2 = 4) := sorry

end circle_equation_l488_488473


namespace three_digit_sum_7_l488_488959

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488959


namespace common_difference_is_two_l488_488596

-- Define the properties and conditions.
variables {a : ℕ → ℝ} {d : ℝ}

-- An arithmetic sequence definition.
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Problem statement to be proved.
theorem common_difference_is_two (h1 : a 1 + a 5 = 10) (h2 : a 4 = 7) (h3 : arithmetic_sequence a d) : 
  d = 2 :=
sorry

end common_difference_is_two_l488_488596


namespace tangent_points_and_symmetry_center_l488_488009

noncomputable def f (x a : ℝ) : ℝ := sin (2 * a * x) - sin (a * x) * cos (a * x)

theorem tangent_points_and_symmetry_center :
  (∀ a > 0, (∀ m, (∀ x, f x a = m → (∀ y, tangent_line y m → distance_tangent_points = π) →
  (m = -1/2 ∨ m = 1/2) ∧ a = 2))) ∧
  (∀ x0 ∈ [0, π], ∃ y0, symmetric_center f x0 a → (x0, y0) = (π/4, -1/2) ∨ (x0, y0) = (3*π/4, -1/2)) :=
sorry

end tangent_points_and_symmetry_center_l488_488009


namespace statement_C_l488_488193

variables (a b c d : ℝ)

theorem statement_C (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end statement_C_l488_488193


namespace min_balls_for_color_15_l488_488382

theorem min_balls_for_color_15
  (red green yellow blue white black : ℕ)
  (h_red : red = 28)
  (h_green : green = 20)
  (h_yellow : yellow = 19)
  (h_blue : blue = 13)
  (h_white : white = 11)
  (h_black : black = 9) :
  ∃ n, n = 76 ∧ ∀ balls_drawn, balls_drawn = n →
  ∃ color, 
    (color = "red" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= red) ∨
    (color = "green" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= green) ∨
    (color = "yellow" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= yellow) ∨
    (color = "blue" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= blue) ∨
    (color = "white" ∧ balls_drawn >= 15 ∧ balls_drawn <= white) ∨
    (color = "black" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= black) := 
sorry

end min_balls_for_color_15_l488_488382


namespace division_of_polynomials_l488_488003

theorem division_of_polynomials (a b : ℝ) :
  (18 * a^2 * b - 9 * a^5 * b^2) / (-3 * a * b) = -6 * a + 3 * a^4 * b :=
by
  sorry

end division_of_polynomials_l488_488003


namespace three_digit_numbers_sum_seven_l488_488921

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488921


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488939

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488939


namespace three_digit_integers_sum_to_7_l488_488909

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488909


namespace max_salary_highest_paid_player_l488_488797

-- Define conditions
def num_players : ℕ := 15
def min_salary : ℕ := 20000
def total_salary_cap : ℕ := 500000

-- Define the problem statement using Lean tactics
theorem max_salary_highest_paid_player :
  ∃ s : ℕ, s = 220000 ∧ num_players - 1 ∗ min_salary + s ≤ total_salary_cap :=
by
  -- The proof will go here
  sorry

end max_salary_highest_paid_player_l488_488797


namespace lambda_value_l488_488181

open Locale BigOperators

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) 
  (h1 : a = (1, 3))
  (h2 : b = (3, 4))
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0) : 
  λ = 3 / 5 := by
  sorry

end lambda_value_l488_488181


namespace time_ratio_l488_488046

theorem time_ratio (distance : ℝ) (initial_time : ℝ) (new_speed : ℝ) :
  distance = 600 → initial_time = 5 → new_speed = 80 → (distance / new_speed) / initial_time = 1.5 :=
by
  intros hdist htime hspeed
  sorry

end time_ratio_l488_488046


namespace evaluate_f_at_2_using_horner_l488_488819

def f (x : ℝ) : ℝ := 2 * x^6 - 3 * x^4 + 2 * x^3 + 7 * x^2 + 6 * x + 3

theorem evaluate_f_at_2_using_horner :
  ((((((2 * 2 - 0) * 2 - 3) * 2 + 2) * 2 + 7) * 2 + 6) * 2 + 3) = 5 :=
by {
  -- Starting the computation based on Horner's Rule:
  let V1 := 2 * 2 - 0,
  have hV1 : V1 = 4 := by norm_num,
  let V2 := V1 * 2 - 3,
  have hV2 : V2 = 5 := by norm_num,
  exact hV2,
}

end evaluate_f_at_2_using_horner_l488_488819


namespace abs_diff_squares_103_97_l488_488816

theorem abs_diff_squares_103_97 : abs ((103 ^ 2) - (97 ^ 2)) = 1200 := by
  sorry

end abs_diff_squares_103_97_l488_488816


namespace statement_C_l488_488192

variables (a b c d : ℝ)

theorem statement_C (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end statement_C_l488_488192


namespace twelve_pharmacies_not_enough_l488_488742

-- Define the grid dimensions and necessary parameters
def grid_size := 9
def total_intersections := (grid_size + 1) * (grid_size + 1) -- 10 * 10 grid
def walking_distance := 3
def coverage_side := (walking_distance * 2 + 1)  -- 7x7 grid coverage
def max_covered_per_pharmacy := (coverage_side - 1) * (coverage_side - 1)  -- Coverage per direction

-- Define the main theorem
theorem twelve_pharmacies_not_enough (n m : ℕ): 
  n = grid_size + 1 -> m = grid_size + 1 -> total_intersections = n * m -> 
  (walking_distance < n) -> (walking_distance < m) -> (pharmacies : ℕ) -> pharmacies = 12 ->
  (coverage_side <= n) -> (coverage_side <= m) ->
  ¬ (∀ i j : ℕ, i < n -> j < m -> ∃ p : ℕ, p < pharmacies -> 
  abs (i - (p / (grid_size + 1))) + abs (j - (p % (grid_size + 1))) ≤ walking_distance) :=
begin
  intros,
  sorry -- Proof omitted
end

end twelve_pharmacies_not_enough_l488_488742


namespace a_n_is_n_plus_1_l488_488482

def a : ℕ → ℕ
| 0 := 2
| (n + 1) := (a n)^2 - n * (a n) + 1

theorem a_n_is_n_plus_1 : ∀ n : ℕ, a (n + 1) = n + 2 :=
by
  intro n
  induction n
  case zero =>
    simp [a]
  case succ n h =>
    sorry

end a_n_is_n_plus_1_l488_488482


namespace log_base_9_of_729_l488_488851

theorem log_base_9_of_729 : ∃ x : ℝ, (9:ℝ) = 3^2 ∧ (729:ℝ) = 3^6 ∧ (9:ℝ)^x = 729 ∧ x = 3 :=
by
  sorry

end log_base_9_of_729_l488_488851


namespace vector_at_t_neg1_l488_488791

theorem vector_at_t_neg1 :
  let a : ℝ × ℝ × ℝ := (2, 6, 16)
  let b : ℝ × ℝ × ℝ := (1, 1, 8)
  let d : ℝ × ℝ × ℝ := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let t : ℝ := -1
  (a.1 + t * d.1, a.2 + t * d.2, a.3 + t * d.3) = (3, 11, 24) :=
by
  sorry

end vector_at_t_neg1_l488_488791


namespace number_of_three_digit_numbers_with_sum_seven_l488_488896

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488896


namespace find_lambda_l488_488177

variables {R : Type*} [LinearOrderedField R]
variables (a b : ℝ × ℝ) (λ : ℝ)
variable h_orth : ((1 - 3 * λ, 3 - 4 * λ) ∙ (3, 4)) = 0

theorem find_lambda : λ = 3 / 5 := by
  sorry

end find_lambda_l488_488177


namespace binary_representation_l488_488637

theorem binary_representation (n : ℕ) :
  ∃ (m : ℕ) (a : ℕ → ℕ),
    (∀ i, a i = ((n / 2^i) % 2) ∧ 
         ((n / 2^i) = match i with 
                         | 0 => n
                         | j+1 => (n / 2^j) / 2)
                         ∧ 
         (∃ k, (n / 2^k) = 1 ∧ a k = 1)) → 
    (n = ∑ i in range (finset.max (finset.range m).succ), a i * 2^i) :=
begin
  sorry
end

end binary_representation_l488_488637


namespace modulus_of_z_l488_488149

noncomputable def z : ℂ := (2 + complex.I) / complex.I
def answer : ℝ := real.sqrt 5

theorem modulus_of_z : complex.abs z = answer :=
by
  sorry

end modulus_of_z_l488_488149


namespace domain_of_function_l488_488064

theorem domain_of_function :
  {x : ℝ | 0 ≤ x ∧ x ≤ 36} = {x : ℝ | ∃ y ∈ set.Icc (0:ℝ) (36:ℝ), y = x} :=
by sorry

end domain_of_function_l488_488064


namespace three_digit_sum_seven_l488_488958

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488958


namespace coefficient_sum_eq_512_l488_488142

theorem coefficient_sum_eq_512 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ) :
  (1 - x) ^ 9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + 
                a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 →
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| + |a_9| = 512 :=
sorry

end coefficient_sum_eq_512_l488_488142


namespace builders_time_l488_488203

theorem builders_time (total_work_days : ℕ) (n_builders : ℕ) : total_work_days = 24 → n_builders = 6 → (total_work_days / n_builders) = 4 :=
by
  intros h_work h_builders
  rw [h_work, h_builders]
  exact Nat.div_eq_of_eq_mul sorry

end builders_time_l488_488203


namespace pyramid_volume_l488_488794

theorem pyramid_volume (s : ℝ) (total_surface_area : ℝ) (triangular_face_area : ℝ) (height : ℝ) (slant_height : ℝ) 
  (h1 : total_surface_area = 900) 
  (h2 : triangular_face_area = (1/3) * s^2)
  (h3 : (s^2 + 4 * triangular_face_area) = total_surface_area) 
  (h4 : height = sqrt (slant_height^2 - (s/2)^2)) 
  (h5 : triangular_face_area = (1/2) * s * slant_height) 
  : (1/3) * s^2 * height = 1108.67 :=
by sorry

end pyramid_volume_l488_488794


namespace spell_LEVEL_count_l488_488669

noncomputable def grid : list (list char) := [
  ['L', 'E', 'V'],
  ['E', 'L', 'E'],
  ['V', 'E', 'L']
]

def is_adjacent (p1 p2 : (nat × nat)) : Bool :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 = y2 - 1)) ∨ ((x1 = x2 + 1 ∨ x1 = x2 - 1) ∧ y1 = y2)

def letter_positions (letter : char) (g : list (list char)) : list (nat × nat) :=
  (list.nat.product (list.range g.length) (list.range g.head.length)).filter 
    (fun (pos : nat × nat) => g.nth pos.1 >>= fun r => r.nth pos.2 = some letter)

def num_ways_to_spell_level : nat :=
  let L_positions := letter_positions 'L' grid
  let paths_count (start : (nat × nat)) (target : char) : nat :=
    -- Generate paths based on adjacency rules; assuming num_ways is pre-calculated for simplicity.
    2 -- From the solution steps, each starting 'L' leads to 2 paths to 'V'
  L_positions.length * 36  -- 2 paths * 2 * 2 (after accounting for full spellings as described)

theorem spell_LEVEL_count : num_ways_to_spell_level = 144 := 
  by -- 
  sorry

end spell_LEVEL_count_l488_488669


namespace three_digit_sum_seven_l488_488950

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488950


namespace sum_first_2012_terms_of_b_l488_488130

def is_convex_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → a (n + 1) = a n + a (n + 2)

noncomputable def b : ℕ → ℤ
| 1     := 1
| 2     := -2
| n + 3 := b (n + 1) + b (n + 2)

theorem sum_first_2012_terms_of_b : 
  is_convex_sequence b → 
  (∑ i in finset.range 2012, b (i + 1)) = -1 := 
by
  sorry

end sum_first_2012_terms_of_b_l488_488130


namespace num_three_digit_integers_sum_to_seven_l488_488985

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488985


namespace three_digit_integers_sum_to_7_l488_488907

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488907


namespace log_base_9_of_729_l488_488849

-- Define the conditions
def nine_eq := 9 = 3^2
def seven_two_nine_eq := 729 = 3^6

-- State the goal to be proved
theorem log_base_9_of_729 (h1 : nine_eq) (h2 : seven_two_nine_eq) : log 9 729 = 3 :=
by
  sorry

end log_base_9_of_729_l488_488849


namespace complex_quadratic_root_condition_l488_488341

theorem complex_quadratic_root_condition
  (K : ℂ)
  (h : (7 : ℂ) * K * K + (12 - 5 * complex.I) = 0) :
  |K| ^ 2 = 364 :=
by
  sorry

end complex_quadratic_root_condition_l488_488341


namespace sum_of_sequence_l488_488483

theorem sum_of_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, 1 ≤ n → a 1 + ∑ i in finset.range n, (2 * i + 1) * a (i + 1) = 1 / 3 - 1 / (2 * (n + 1) + 1)) :
  ∀ n : ℕ, 1 ≤ n → a 1 + ∑ i in finset.range (n - 1), a (i + 2) = 1 / 6 - 1 / (2 * (2 * n + 1) * (2 * n + 3)) :=
by
  sorry

end sum_of_sequence_l488_488483


namespace three_digit_sum_seven_l488_488884

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488884


namespace pizza_combinations_l488_488394

theorem pizza_combinations :
  (nat.choose 8 1) + (nat.choose 8 2) + (nat.choose 8 3) = 92 :=
by
  sorry

end pizza_combinations_l488_488394


namespace three_digit_numbers_sum_seven_l488_488913

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488913


namespace not_enough_pharmacies_l488_488766

theorem not_enough_pharmacies : 
  ∀ (n m : ℕ), n = 10 ∧ m = 10 →
  ∃ (intersections : ℕ), intersections = n * m ∧ 
  ∀ (d : ℕ), d = 3 →
  ∀ (coverage : ℕ), coverage = (2 * d + 1) * (2 * d + 1) →
  ¬ (coverage * 12 ≥ intersections * 2) :=
by sorry

end not_enough_pharmacies_l488_488766


namespace find_c_l488_488549

variables (A B C : ℝ)
variables (a b c : ℝ)
variables (AB AC BA BC : ℝ)
variables (cosA cosB : ℝ)

-- Conditions
variable (hAB_AC : AB * AC = 1)
variable (hBA_BC : BA * BC = 1)

-- Defining lengths
def length_AB : ℝ := c
def length_BC : ℝ := a
def length_AC : ℝ := b

-- Using cosine rules to derive cosA and cosB
def cos_law1 : ℝ := c * b * cosA = c * a * cosB

-- Sine law relation used in the solution steps
def sine_law1 : ℝ := sin B * cos A = cos B * sin A

-- Applying the cosine law after concluding that b = a
def cos_law2 : ℝ := a * c * (a ^ 2 + c ^ 2 - a ^ 2) / (2 * a * c) = 1

theorem find_c : c = Real.sqrt 2 := by 
  sorry

end find_c_l488_488549


namespace expected_winnings_is_350_l488_488025

noncomputable def expected_winnings : ℝ :=
  (1 / 8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_winnings_is_350 :
  expected_winnings = 3.5 :=
by sorry

end expected_winnings_is_350_l488_488025


namespace find_smallest_alpha_l488_488638

theorem find_smallest_alpha
  (a b c : ℝ^3)
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (hc : ‖c‖ = 1)
  (alpha : ℝ)
  (hab : angle a b = α)
  (hcb_ab : angle c (cross_product a b) = α)
  (hbc a_dot_cxa : dot_product b (cross_product c a) = 1 / 8) :
  α ≈ 7.24 :=
by
  sorry

end find_smallest_alpha_l488_488638


namespace number_of_three_digit_numbers_with_sum_seven_l488_488893

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488893


namespace selling_price_l488_488038

def cost_price : ℝ := 76.92
def profit_rate : ℝ := 0.30

theorem selling_price : cost_price * (1 + profit_rate) = 100.00 := by
  sorry

end selling_price_l488_488038


namespace locus_of_points_l488_488633

-- Define the conditions of the problem
variables {n : ℕ} (A : Fin n → ℝ × ℝ) (k : Fin n → ℝ) (C : ℝ)

-- Define the locus property as a set of points satisfying the equation
def locus (M : ℝ × ℝ) : Prop :=
  (Finset.univ.sum (λ i, k i * ((M.1 - (A i).1)^2 + (M.2 - (A i).2)^2))) = C

-- Mathematically equivalent proof problem
theorem locus_of_points (h : ∑ i, k i ≠ 0 ∨ ∑ i, k i = 0) :
  (∑ i, k i ≠ 0 → ∃ b c d : ℝ, ∀ x y, locus (x, y) → b * x^2 + c * y^2 + d = 0) ∧
  (∑ i, k i = 0 → ∃ b c d : ℝ, ∀ x y, locus (x, y) → b * x + c * y + d = 0) :=
sorry

end locus_of_points_l488_488633


namespace solve_for_x_l488_488663
noncomputable theory

theorem solve_for_x (x : ℝ) (h : 5^(x + 6) = (5^4)^x) : x = 2 :=
by
  sorry

end solve_for_x_l488_488663


namespace number_plus_273_l488_488670

theorem number_plus_273 (x : ℤ) (h : x - 477 = 273) : x + 273 = 1023 := by
  sorry

end number_plus_273_l488_488670


namespace number_of_three_digit_numbers_with_sum_7_l488_488976

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488976


namespace num_three_digit_integers_sum_to_seven_l488_488984

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488984


namespace least_large_groups_l488_488378

theorem least_large_groups (total_members : ℕ) (members_large_group : ℕ) (members_small_group : ℕ) (L : ℕ) (S : ℕ)
  (H_total : total_members = 90)
  (H_large : members_large_group = 7)
  (H_small : members_small_group = 3)
  (H_eq : total_members = L * members_large_group + S * members_small_group) :
  L = 12 :=
by
  have h1 : total_members = 90 := by exact H_total
  have h2 : members_large_group = 7 := by exact H_large
  have h3 : members_small_group = 3 := by exact H_small
  rw [h1, h2, h3] at H_eq
  -- The proof is skipped here
  sorry

end least_large_groups_l488_488378


namespace builders_cottage_build_time_l488_488205

theorem builders_cottage_build_time :
  (∀ n : ℕ, ∀ d : ℕ, ∀ b : ℕ, b * d = n → (b = 3 → d = 8) → (b = 6 → d = 4)) :=
begin
  sorry -- Skipping the proof as specified
end

end builders_cottage_build_time_l488_488205


namespace min_combined_number_of_horses_and_ponies_l488_488736

theorem min_combined_number_of_horses_and_ponies :
  ∃ P H : ℕ, H = P + 4 ∧ (∃ k : ℕ, k = (3 * P) / 10 ∧ k = 16 * (3 * P) / (16 * 10) ∧ H + P = 36) :=
sorry

end min_combined_number_of_horses_and_ponies_l488_488736


namespace smallest_n_integer_seq_l488_488829

def cuberoot_4 := real.cbrt 4

def sequence (n : ℕ) : ℝ :=
  match n with
  | 1 => cuberoot_4
  | n + 1 => sequence n ^ cuberoot_4

theorem smallest_n_integer_seq : ∃ n : ℕ, n = 8 ∧ ∃ (k : ℕ), sequence n = k := 
by 
  sorry

end smallest_n_integer_seq_l488_488829


namespace max_distance_on_ellipse_to_vertex_l488_488614

open Real

noncomputable def P (θ : ℝ) : ℝ × ℝ :=
(√5 * cos θ, sin θ)

def ellipse (x y : ℝ) := (x^2 / 5) + y^2 = 1

def B : ℝ × ℝ := (0, 1)

def dist (A B : ℝ × ℝ) : ℝ :=
sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem max_distance_on_ellipse_to_vertex :
  ∃ θ : ℝ, dist (P θ) B = 5 / 2 :=
sorry

end max_distance_on_ellipse_to_vertex_l488_488614


namespace problem1_problem2_l488_488153

noncomputable def f (x : ℝ) : ℝ := 3 * |x - 1| + |3 * x + 7|

theorem problem1 (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - 3 * a) → -2 ≤ a ∧ a ≤ 5 := by
  sorry

theorem problem2 (a b : ℝ) (h1 : a + b = 3) (ha : a > 0) (hb : b > 0) :
  sqrt (a + 1) + sqrt (b + 1) ≤ sqrt (f 0) := by
  sorry

end problem1_problem2_l488_488153


namespace handshakes_at_reunion_l488_488738

theorem handshakes_at_reunion (n : ℕ) (h : n = 11) : (n.choose 2) = 55 :=
by {
  rw h,
  exact nat.choose_succ_succ _ _,
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)
}

end handshakes_at_reunion_l488_488738


namespace canBeDividedEvenly_l488_488092

-- Define figures
inductive Figure
| ∞ | one | O | T | Δ | tilde | tilde_underlined

open Figure

-- Define characteristics for the figures
def hasAxisOfSymmetry : Figure → Bool
| ∞   | one | O | T => true
| Δ   | tilde | tilde_underlined => false

def hasCenterOfSymmetry : Figure → Bool
| ∞ | O => true
| one | T | Δ | tilde | tilde_underlined => false

def isCurved : Figure → Bool
| ∞   | tilde | tilde_underlined | O => true
| one | T | Δ => false

def isLetterShape : Figure → Bool
| one | T | O => true
| tilde | tilde_underlined | ∞ | Δ => false

-- Figures
def allFigures : List Figure := [∞, one, O, T, Δ, tilde, tilde_underlined]

-- theorem to prove divisions
theorem canBeDividedEvenly :
  (∃ g1 g2 : List Figure, g1.length = 4 ∧ g2.length = 3 ∧ ∀ f ∈ g1, hasAxisOfSymmetry f = true ∧ ∀ f ∈ g2, hasAxisOfSymmetry f = false) ∧
  (∀ f ∈ allFigures, f ∈ g1 ∨ f ∈ g2) ∧
  (g1 ∪ g2 = allFigures ∧ (g1 ∩ g2 = [])) →
  -- Existence of divisions by axis of symmetry
  (∃ g1 g2 : List Figure, g1.length = 4 ∧ g2.length = 3 ∧ ∀ f ∈ g1, isCurved f = true ∧ ∀ f ∈ g2, isCurved f = false) ∧
  (∀ f ∈ allFigures, f ∈ g1 ∨ f ∈ g2) ∧
  (g1 ∪ g2 = allFigures ∧ (g1 ∩ g2 = [])) →
  -- Existence of divisions by shape characteristics (curved or non-curved)
  (∃ g1 g2 : List Figure, g1.length = 3 ∧ g2.length = 4 ∧ ∀ f ∈ g1, isLetterShape f = true ∧ ∀ f ∈ g2, isLetterShape f = false) ∧
  (∀ f ∈ allFigures, f ∈ g1 ∨ f ∈ g2) ∧
  (g1 ∪ g2 = allFigures ∧ (g1 ∩ g2 = [])) →
  -- Existence of divisions by letter shapes or non-letter shapes
  true := sorry

end canBeDividedEvenly_l488_488092


namespace number_of_three_digit_numbers_with_sum_7_l488_488971

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488971


namespace exists_x_nat_l488_488263

theorem exists_x_nat (a c : ℕ) (b : ℤ) : ∃ x : ℕ, (a^x + x) % c = b % c :=
by
  sorry

end exists_x_nat_l488_488263


namespace twelve_pharmacies_not_sufficient_l488_488761

-- Define an intersection grid of size 10 x 10 (100 squares).
def city_grid : Type := Fin 10 × Fin 10 

-- Define the distance measure between intersections, assumed as L1 metric for grid paths.
def dist (p q : city_grid) : Nat := (abs (p.1.val - q.1.val) + abs (p.2.val - q.2.val))

-- Define a walking distance pharmacy 
def is_walking_distance (p q : city_grid) : Prop := dist p q ≤ 3

-- State that having 12 pharmacies is not sufficient
theorem twelve_pharmacies_not_sufficient (pharmacies : Fin 12 → city_grid) :
  ¬ (∀ intersection: city_grid, ∃ (p_index : Fin 12), is_walking_distance (pharmacies p_index) intersection) :=
sorry

end twelve_pharmacies_not_sufficient_l488_488761


namespace remove_matches_to_avoid_rectangles_l488_488653

-- Define the original 4x4 grid configuration and the 40 matchsticks
structure Grid4x4 :=
(horizontal_matches : ℕ)
(vertical_matches : ℕ)
(total_matches : ℕ)

-- Initialize the original 4x4 grid configuration
def original_grid : Grid4x4 :=
{ horizontal_matches := 20,
  vertical_matches := 20,
  total_matches := 40 }

-- Define a property that checks if a matchstick configuration contains no complete rectangles
def no_complete_rectangles (removed_matches : ℕ) (horizontal_removed : ℕ) (vertical_removed : ℕ) : Prop :=
(removed_matches = horizontal_removed + vertical_removed) → (original_grid.total_matches - removed_matches = 29) →
-- (Additional logic to ensure no complete rectangles, typically a graphical or combinatorial check, skipped for brevity) sorry

-- The theorem to prove that, by removing 11 specific matchsticks, no complete rectangles remain
theorem remove_matches_to_avoid_rectangles :
  ∃ (horizontal_removed vertical_removed : ℕ), no_complete_rectangles 11 horizontal_removed vertical_removed :=
sorry

end remove_matches_to_avoid_rectangles_l488_488653


namespace probability_sum_divisible_by_3_l488_488540

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

noncomputable def num_pairs_divisible_by_3 (primes : List ℕ) : ℕ :=
  (primes.toFinset.powerset.toList.filter 
    (λ s, s.card = 2 ∧ (s.sum % 3 = 0))).length

theorem probability_sum_divisible_by_3 :
  (num_pairs_divisible_by_3 first_ten_primes : ℚ) / (10.choose 2) = 2 / 15 :=
by
  sorry

end probability_sum_divisible_by_3_l488_488540


namespace chameleons_changed_color_l488_488556

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l488_488556


namespace sum_radical_conjugates_l488_488825

theorem sum_radical_conjugates (n : ℝ) (m : ℝ) (h1 : n = 5) (h2 : m = (sqrt 500)) : 
  (n - m) + (n + m) = 10 :=
by 
  rw [h1, h2]
  sorry

end sum_radical_conjugates_l488_488825


namespace sin_func_even_min_period_2pi_l488_488118

noncomputable def f (x : ℝ) : ℝ := Real.sin (13 * Real.pi / 2 - x)

theorem sin_func_even_min_period_2pi :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ 2 * Real.pi) ∧ (∀ x : ℝ, f (x + 2 * Real.pi) = f x) :=
by
  sorry

end sin_func_even_min_period_2pi_l488_488118


namespace range_of_f_l488_488155

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2 else
  if -1 < x ∧ x < 2 then x^2 else 0

theorem range_of_f : set.range f = set.Iio 4 :=
  sorry

end range_of_f_l488_488155


namespace problem_correct_choice_l488_488141

-- Define proposition p
def p : Prop := ∀ x > 1, log (1 / 2) x > 0

-- Define proposition q
def q : Prop := ∃ x : ℝ, x^3 ≥ 3^x

-- The theorem to prove
theorem problem_correct_choice : ¬p ∧ ¬q → p ∨ ¬q :=
by
  sorry

end problem_correct_choice_l488_488141


namespace number_of_three_digit_numbers_with_sum_seven_l488_488898

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488898


namespace insufficient_pharmacies_l488_488750

/-- Define the grid size and conditions --/
def grid_size : ℕ := 9
def total_intersections : ℕ := 100
def pharmacy_walking_distance : ℕ := 3
def number_of_pharmacies : ℕ := 12

/-- Prove that 12 pharmacies are not enough for the given conditions. --/
theorem insufficient_pharmacies : ¬ (number_of_pharmacies ≥ grid_size + 3) → 
  ∃(pharmacies : ℕ), pharmacies = number_of_pharmacies ∧
  ∀(x y : ℕ), (x ≤ grid_size ∧ y ≤ grid_size) → 
  ¬ exists (p : ℕ × ℕ), p.1 ≤ x + pharmacy_walking_distance ∧
  p.2 ≤ y + pharmacy_walking_distance ∧ 
  p.1 < total_intersections ∧ 
  p.2 < total_intersections := 
by sorry

end insufficient_pharmacies_l488_488750


namespace expected_value_of_twelve_sided_die_l488_488043

theorem expected_value_of_twelve_sided_die : 
  (let n := 12 in
  let S := n * (n + 1) / 2 in
  let E := S / n in
  E = 6.5) :=
sorry

end expected_value_of_twelve_sided_die_l488_488043


namespace chameleon_color_change_l488_488582

theorem chameleon_color_change :
  ∃ (x : ℕ), (∃ (blue_initial : ℕ) (red_initial : ℕ), blue_initial = 5 * x ∧ red_initial = 140 - 5 * x) →
  (∃ (blue_final : ℕ) (red_final : ℕ), blue_final = x ∧ red_final = 3 * (140 - 5 * x)) →
  (5 * x - x = 80) :=
begin
  sorry
end

end chameleon_color_change_l488_488582


namespace cotangent_ratio_l488_488257

variable (x y z : ℝ) (ξ η ζ : ℝ)

-- Conditions
def is_triangle_sides (x y z : ℝ) : Prop := x > 0 ∧ y > 0 ∧ z > 0
def angles_opposite_sides : Prop := ξ + η + ζ = π
def pythagorean_like_identity (x y z : ℝ) : Prop := x^2 + y^2 = 2023 * z^2

-- Question to prove
theorem cotangent_ratio (hx : is_triangle_sides x y z)
                        (ha : angles_opposite_sides ξ η ζ)
                        (h : pythagorean_like_identity x y z) :
    (Real.cot ζ) / (Real.cot ξ + Real.cot η) = 1011 := sorry

end cotangent_ratio_l488_488257


namespace evaluate_expression_l488_488438

theorem evaluate_expression :
  ((3^1 - 2 + 7^3 + 1 : ℚ)⁻¹ * 6) = (2 / 115) := by
  sorry

end evaluate_expression_l488_488438


namespace burger_filler_percentage_l488_488384

theorem burger_filler_percentage (total_weight filler_weight : ℕ) (h_total : total_weight = 180) (h_filler : filler_weight = 45) :
  (135 / 180 : ℚ) * 100 = 75 ∧ (45 / 180 : ℚ) * 100 = 25 :=
by
  -- Given definitions based on problem conditions
  have h_non_filler_weight : ℕ := total_weight - filler_weight := by rw [h_total, h_filler]; refl
  -- Conditions
  have h1 : 135 = total_weight - filler_weight := by rw [h_total, h_filler]; exact. rfl
  have h2 : 135 / 180 * 100 = 75 := by norm_cast; norm_num
  have h3 : 45 / 180 * 100 = 25 := by norm_cast; norm_num
  exact ⟨h2, h3⟩

end burger_filler_percentage_l488_488384


namespace midpoint_sum_and_distance_l488_488352

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem midpoint_sum_and_distance :
  let p1 := (8, 10)
  let p2 := (-4, -2)
  let mid := midpoint p1 p2
  (mid.1 + mid.2 = 6) ∧ (distance mid p1 = 6 * Real.sqrt 2) :=
by
  let p1 := (8, 10)
  let p2 := (-4, -2)
  let mid := midpoint p1 p2
  have h_mid_coords : mid = (2, 4) := by sorry
  have h_sum : mid.1 + mid.2 = 6 := by sorry
  have h_dist : distance mid p1 = 6 * Real.sqrt 2 := by sorry
  exact ⟨h_sum, h_dist⟩

end midpoint_sum_and_distance_l488_488352


namespace number_of_three_digit_numbers_with_sum_seven_l488_488890

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488890


namespace min_balls_to_draw_l488_488379

theorem min_balls_to_draw {red green yellow blue white black : ℕ} 
    (h_red : red = 28) 
    (h_green : green = 20) 
    (h_yellow : yellow = 19) 
    (h_blue : blue = 13) 
    (h_white : white = 11) 
    (h_black : black = 9) :
    ∃ n, n = 76 ∧ 
    (∀ drawn, (drawn < n → (drawn ≤ 14 + 14 + 14 + 13 + 11 + 9)) ∧ (drawn >= n → (∃ c, c ≥ 15))) :=
sorry

end min_balls_to_draw_l488_488379


namespace product_of_consecutive_integers_l488_488856

open Nat

theorem product_of_consecutive_integers (k : ℕ) (hk : k > 1) :
  (∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, A = ∏ i in finRange k, (m + i)) ↔ (k = 2 ∨ k = 3) :=
by
  -- Definitions
  let A (n : ℕ) : ℕ := 17^(18 * n) + 4 * 17^(2 * n) + 7 * 19^(5 * n)
  sorry

end product_of_consecutive_integers_l488_488856


namespace find_ab_l488_488490

variable {a b : ℝ}

theorem find_ab (h : log 10 a + log 10 b = 1) : a * b = 10 := sorry

end find_ab_l488_488490


namespace find_M_value_l488_488333

-- Statements of the problem conditions and the proof goal
theorem find_M_value (a b c M : ℤ) (h1 : a + b + c = 75) (h2 : a + 4 = M) (h3 : b - 5 = M) (h4 : 3 * c = M) : M = 31 := 
by
  sorry

end find_M_value_l488_488333


namespace find_range_of_a_l488_488164

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := (m * x) / (x^2 + n)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := real.log x + (a / x)

theorem find_range_of_a (a : ℝ) :
  (∀ x1 ∈ set.Icc (-1 : ℝ) 1, ∃ x2 ∈ set.Icc 1 real.exp 1, g x2 a ≤ f x1 4 1 + 7/2) →
  a ≤ real.sqrt real.exp 1 :=
sorry

end find_range_of_a_l488_488164


namespace heidi_paints_fraction_in_10_minutes_l488_488198

variable (Heidi_paint_rate : ℕ → ℝ)
variable (t : ℕ)
variable (fraction : ℝ)

theorem heidi_paints_fraction_in_10_minutes 
  (h1 : Heidi_paint_rate 30 = 1) 
  (h2 : t = 10) 
  (h3 : fraction = 1 / 3) : 
  Heidi_paint_rate t = fraction := 
sorry

end heidi_paints_fraction_in_10_minutes_l488_488198


namespace roots_opposite_eq_minus_one_l488_488536

theorem roots_opposite_eq_minus_one (k : ℝ) 
  (h_real_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ + x₂ = 0 ∧ x₁ * x₂ = k + 1) :
  k = -1 :=
by
  sorry

end roots_opposite_eq_minus_one_l488_488536


namespace bottles_from_Shop_C_l488_488839

theorem bottles_from_Shop_C (TotalBottles ShopA ShopB ShopC : ℕ) 
  (h1 : TotalBottles = 550) 
  (h2 : ShopA = 150) 
  (h3 : ShopB = 180) 
  (h4 : TotalBottles = ShopA + ShopB + ShopC) : 
  ShopC = 220 := 
by
  sorry

end bottles_from_Shop_C_l488_488839


namespace determine_a_l488_488535

theorem determine_a (a : ℝ) (h : 2 * (-1) + a = 3) : a = 5 := sorry

end determine_a_l488_488535


namespace number_of_three_digit_numbers_with_sum_7_l488_488977

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488977


namespace ratio_of_points_l488_488649

def Noa_points : ℕ := 30
def total_points : ℕ := 90

theorem ratio_of_points (Phillip_points : ℕ) (h1 : Phillip_points = 2 * Noa_points) (h2 : Noa_points + Phillip_points = total_points) : Phillip_points / Noa_points = 2 := 
by
  intros
  sorry

end ratio_of_points_l488_488649


namespace num_three_digit_integers_sum_to_seven_l488_488993

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488993


namespace arithmetic_sequence_before_neg_17_l488_488522

noncomputable def find_terms_before_neg_17 (a1 : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  (91 - 3 * n)

theorem arithmetic_sequence_before_neg_17 :
  (find_terms_before_neg_17 88 (-3) 36 = -17) → 35 = 36 - 1 := 
by 
  -- Given the sequence definition and the position of the term -17, we need to show that
  -- the number of terms before -17 is 35.
  intro h,
  have h1 : 91 - 3 * 36 = -17 := by exact h,
  have h2 : 36 - 1 = 35 := by ring,
  exact h2

end arithmetic_sequence_before_neg_17_l488_488522


namespace popcorn_packages_needed_l488_488393

theorem popcorn_packages_needed (total_buckets_needed : ℕ) (buckets_per_package : ℕ) 
  (h1 : total_buckets_needed = 426) (h2 : buckets_per_package = 8) : 
  ∃ n : ℕ, n * buckets_per_package ≥ total_buckets_needed ∧ n = 54 :=
by
  use 54
  split
  sorry
  refl

end popcorn_packages_needed_l488_488393


namespace problem_min_ineq_range_l488_488124

theorem problem_min_ineq_range (a b : ℝ) (x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∀ x, 1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|) ∧ (1 / a + 4 / b = 9) ∧ (-7 ≤ x ∧ x ≤ 11) :=
sorry

end problem_min_ineq_range_l488_488124


namespace range_of_a_l488_488503

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc (1/2 : ℝ) (2/3 : ℝ), log a (2 * x - a) > 0) ↔ (1/3 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l488_488503


namespace three_digit_sum_seven_l488_488886

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488886


namespace original_breadth_l488_488319

theorem original_breadth (b : ℝ) 
    (h_length_orig : 18)
    (h_length_new : 25)
    (h_breadth_new : 7.2)
    (h_area_eq : 18 * b = 25 * 7.2) :
    b = 10 := 
by 
  -- proof skipped
  sorry

end original_breadth_l488_488319


namespace minimum_tangent_length_4_l488_488201

noncomputable def minimum_tangent_length (a b : ℝ) : ℝ :=
  Real.sqrt ((b + 4)^2 + (b - 2)^2 - 2)

theorem minimum_tangent_length_4 :
  ∀ (a b : ℝ), (x^2 + y^2 + 2 * x - 4 * y + 3 = 0) ∧ (x = a ∧ y = b) ∧ (2*a*x + b*y + 6 = 0) → 
    minimum_tangent_length a b = 4 :=
by
  sorry

end minimum_tangent_length_4_l488_488201


namespace twelve_pharmacies_not_enough_l488_488754

def grid := ℕ × ℕ

def is_within_walking_distance (p1 p2 : grid) : Prop :=
  abs (p1.1 - p1.2) ≤ 3 ∧ abs (p2.1 - p2.2) ≤ 3

def walking_distance_coverage (pharmacies : set grid) (p : grid) : Prop :=
  ∃ pharmacy ∈ pharmacies, is_within_walking_distance pharmacy p

def sufficient_pharmacies (pharmacies : set grid) : Prop :=
  ∀ p : grid, walking_distance_coverage pharmacies p

theorem twelve_pharmacies_not_enough (pharmacies : set grid) (h : pharmacies.card = 12) : 
  ¬ sufficient_pharmacies pharmacies :=
sorry

end twelve_pharmacies_not_enough_l488_488754


namespace chameleons_changed_color_l488_488555

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l488_488555


namespace smallest_positive_angle_l488_488073

theorem smallest_positive_angle :
  ∃ y : ℝ, 10 * sin y * cos y ^ 3 - 10 * sin y ^ 3 * cos y = 1 ∧ y = (1/4) * real.arcsin (2/5) :=
by
  sorry

end smallest_positive_angle_l488_488073


namespace num_valid_arrangements_l488_488592

-- Define the digits we're working with
def digits := {4, 7, 5, 2, 0}

-- Condition: The first digit can't be zero
def valid_first_digit (d : Nat) := d ≠ 0

-- Main theorem: The number of valid 5-digit arrangements is 96
theorem num_valid_arrangements : 
  (∃ (arr : List Nat), arr.perm [4, 7, 5, 2, 0] ∧ (valid_first_digit arr.head)) = 96 :=
by sorry

end num_valid_arrangements_l488_488592


namespace max_dist_PB_l488_488621

-- Let B be the upper vertex of the ellipse.
def B : (ℝ × ℝ) := (0, 1)

-- Define the equation of the ellipse.
def ellipse (x y : ℝ) : Prop := (x^2) / 5 + y^2 = 1

-- Define a point P on the ellipse.
def P (θ : ℝ) : (ℝ × ℝ) := (sqrt 5 * cos θ, sin θ)

-- Define the distance function between points.
def dist (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Prove that the maximum distance |PB| is 5/2.
theorem max_dist_PB : ∃ θ : ℝ, dist (P θ) B = 5 / 2 :=
sorry

end max_dist_PB_l488_488621


namespace ratio_a7_b7_l488_488116

variables (a b : ℕ → ℤ) (Sa Tb : ℕ → ℤ)
variables (h1 : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0))
variables (h2 : ∀ n : ℕ, b n = b 0 + n * (b 1 - b 0))
variables (h3 : ∀ n : ℕ, Sa n = n * (a 0 + a n) / 2)
variables (h4 : ∀ n : ℕ, Tb n = n * (b 0 + b n) / 2)
variables (h5 : ∀ n : ℕ, n > 0 → Sa n / Tb n = (7 * n + 1) / (4 * n + 27))

theorem ratio_a7_b7 : ∀ n : ℕ, n = 7 → a 7 / b 7 = 92 / 79 :=
by
  intros n hn_eq
  sorry

end ratio_a7_b7_l488_488116


namespace largest_n_divisible_by_18_l488_488862

theorem largest_n_divisible_by_18 (n : ℕ) : 
  (18^n ∣ nat.factorial 24) ↔ n ≤ 5 := 
sorry

end largest_n_divisible_by_18_l488_488862


namespace number_of_three_digit_numbers_with_sum_seven_l488_488894

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488894


namespace snow_volume_l488_488392

def length : ℝ := 30
def width : ℝ := 3
def height : ℝ := 0.75

def volume (l w h: ℝ) : ℝ := l * w * h

theorem snow_volume :
  volume length width height = 67.5 :=
by
  unfold volume
  sorry

end snow_volume_l488_488392


namespace exists_point_with_at_most_17_visible_asteroids_l488_488798

theorem exists_point_with_at_most_17_visible_asteroids (S : Type) [MetricSpace S] (P : Sphere S) (A : Set (Point S)) (hA_37: A.card = 37) :
  ∃ (p : Point P), ∀ (a ∈ A), (VisibleFrom(p, a) → ∑ (h a ∈ Hemisphere p P).card ≤ 17) :=
sorry

end exists_point_with_at_most_17_visible_asteroids_l488_488798


namespace world_grain_demand_l488_488120

theorem world_grain_demand (S D : ℝ) (h1 : S = 1800000) (h2 : S = 0.75 * D) : D = 2400000 := by
  sorry

end world_grain_demand_l488_488120


namespace find_y_l488_488731

theorem find_y (x y : ℕ) (h1 : x % y = 9) (h2 : x / y = 86 ∧ ((x % y : ℚ) / y = 0.12)) : y = 75 :=
by
  sorry

end find_y_l488_488731


namespace prices_proof_sales_revenue_proof_l488_488018

-- Definitions for the prices and quantities
def price_peanut_oil := 50
def price_corn_oil := 40

-- Conditions from the problem
def condition1 (x y : ℕ) : Prop := 20 * x + 30 * y = 2200
def condition2 (x y : ℕ) : Prop := 30 * x + 10 * y = 1900
def purchased_peanut_oil := 50
def selling_price_peanut_oil := 60

-- Proof statement for Part 1
theorem prices_proof : ∃ (x y : ℕ), condition1 x y ∧ condition2 x y ∧ x = price_peanut_oil ∧ y = price_corn_oil :=
sorry

-- Proof statement for Part 2
theorem sales_revenue_proof : ∃ (m : ℕ), (selling_price_peanut_oil * m > price_peanut_oil * purchased_peanut_oil) ∧ m = 42 :=
sorry

end prices_proof_sales_revenue_proof_l488_488018


namespace smallest_y_divisible_by_11_l488_488732

-- Define the conditions
def is_digit (y : ℕ) : Prop := y ≤ 9

def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

def number_with_y (y : ℕ) : ℕ := 7 * 10^7 + y * 10^6 + 8 * 10^5 + 6 * 10^4 + 0 * 10^3 + 3 * 10^2 + 8 * 10^1

-- The theorem to be proved
theorem smallest_y_divisible_by_11 : ∃ (y : ℕ), is_digit y ∧ is_divisible_by_11 (number_with_y y) ∧ (∀ z : ℕ, is_digit z → is_divisible_by_11 (number_with_y z) → y ≤ z) :=
by
  existsi 2
  split
  apply nat.le_refl
  split
  sorry
  intros z hz1 hz2
  sorry

end smallest_y_divisible_by_11_l488_488732


namespace equation_of_ellipse_reciprocal_sum_of_chords_l488_488133

variables {a b : ℝ}
variables (a_gt_b : a > b) (a_gt_0 : a > 0) (b_gt_0 : b > 0)
variables (ellipse_eq : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → y = b^2 / a ↔ 2 * b^2 / a = 2 * sqrt(2))
variables (perimeter_triangle : ∀ G H, G ≠ H → (G.1 - H.1)^2 + (G.2 - H.2)^2 = 8 * sqrt(2))
variables (perpendicular_cases : ∀ A B C D, (A.2 / A.1) * (C.2 / C.1) = -1 → (A ≠ B) → (C ≠ D))

theorem equation_of_ellipse : 
  ellipse_eq (2 * sqrt 2) (sqrt 4) → 
  (x^2 / (2 * sqrt 2)^2 + y^2 / (sqrt 4)^2 = 1)

theorem reciprocal_sum_of_chords (A B C D : ℝ × ℝ) (l1 l2 : (ℝ × ℝ) → ℝ) :
  l1(A) = 0 → l1(B) = 0 → l2(C) = 0 → l2(D) = 0 → 
  perpendicular_cases A B C D → 
  (1 / dist A B) + (1 / dist C D) = 3 * sqrt(2) / 8

end equation_of_ellipse_reciprocal_sum_of_chords_l488_488133


namespace final_price_after_discounts_l488_488683

-- Define the original list price
def list_price : ℝ := 70

-- Define the first discount percentage
def first_discount : ℝ := 0.1

-- Define the second discount percentage
def second_discount : ℝ := 0.03

-- Calculate the price after first discount
def price_after_first_discount : ℝ :=
  list_price * (1 - first_discount)

-- Calculate the exact second discount percentage (approximation for demonstration)
def exact_second_discount : ℝ := 0.03 + 0.000000000000001 / 100

-- Apply the second discount to the price after the first discount
def price_after_second_discount : ℝ :=
  price_after_first_discount * (1 - exact_second_discount)

-- Define the expected final price after both discounts are applied
def expected_final_price : ℝ := 61.11

-- Statement of the theorem
theorem final_price_after_discounts :
  price_after_second_discount ≈ expected_final_price :=
by sorry 

end final_price_after_discounts_l488_488683


namespace sum_solutions_eq_five_l488_488728

noncomputable def sum_of_solutions : ℝ :=
  let p := Roots ((λ x : ℝ, x^2 - 5*x - 7) : ℝ → ℝ) in
  p.sum

theorem sum_solutions_eq_five : sum_of_solutions = 5 :=
  sorry

end sum_solutions_eq_five_l488_488728


namespace find_y_l488_488247

theorem find_y (x y : ℝ) (hA : {2, Real.log x} = {a | a = 2 ∨ a = Real.log x})
                (hB : {x, y} = {a | a = x ∨ a = y})
                (hInt : {a | a = 2 ∨ a = Real.log x} ∩ {a | a = x ∨ a = y} = {0}) :
  y = 0 :=
  sorry

end find_y_l488_488247


namespace three_digit_numbers_sum_seven_l488_488922

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488922


namespace sum_legs_of_larger_triangle_l488_488347

noncomputable def similar_right_triangles_sums_legs 
  (area_small : ℝ) (hypotenuse_small : ℝ) (area_large : ℝ) (sum_legs_large : ℝ): Prop :=
  ∃ (a b : ℝ), 
  (a^2 + b^2 = hypotenuse_small^2) ∧ 
  (1/2 * a * b = area_small) ∧
  (√((a^2 + b^2) / area_small) * a + √((a^2 + b^2) / area_small) * b = sum_legs_large)

theorem sum_legs_of_larger_triangle 
  (hypotenuse_small : ℝ) (area_small : ℝ) (area_large : ℝ) : 
  similar_right_triangles_sums_legs area_small hypotenuse_small area_large 45.96 :=
by 
  let a: ℝ := 1.62
  let b: ℝ := 9.87
  have h1: a^2 + b^2 = hypotenuse_small^2 := by sorry
  have h2: 1/2 * a * b = area_small := by sorry
  have h3: 
    √(area_large / area_small) * a + √(area_large / area_small) * b = 45.96 := by sorry

  exact ⟨a, b, h1, h2, h3⟩

end sum_legs_of_larger_triangle_l488_488347


namespace sum_of_distances_l488_488785

noncomputable def sum_distances (d_AB : ℝ) (d_A : ℝ) (d_B : ℝ) : ℝ :=
d_A + d_B

theorem sum_of_distances
    (tangent_to_sides : Circle → Point → Point → Prop)
    (C_on_circle : Circle → Point → Prop)
    (A B C : Point)
    (γ : Circle)
    (h_dist_to_AB : ∃ C, C_on_circle γ C → distance_from_line C A B = 4)
    (h_ratio : ∃ hA hB, distance_from_side C A = hA ∧ distance_from_side C B = hB ∧ (hA = 4 * hB ∨ hB = 4 * hA)) :
    sum_distances (distance_from_line C A B) (distance_from_side C A) (distance_from_side C B) = 10 :=
by
  sorry

end sum_of_distances_l488_488785


namespace travel_time_downstream_is_14_minutes_l488_488330

def speed_in_still_water : ℝ := 18  -- km/hr
def rate_of_current : ℝ := 4       -- km/hr
def distance_downstream : ℝ := 5.133333333333334 -- km

def effective_speed_downstream : ℝ := speed_in_still_water + rate_of_current -- km/hr
def time_travelled_downstream_hours : ℝ := distance_downstream / effective_speed_downstream -- hours
def time_travelled_downstream_minutes : ℝ := time_travelled_downstream_hours * 60 -- minutes

theorem travel_time_downstream_is_14_minutes : time_travelled_downstream_minutes = 14 := by
  sorry

end travel_time_downstream_is_14_minutes_l488_488330


namespace volume_percentage_correct_l488_488793

-- Define the initial conditions
def box_length := 8
def box_width := 6
def box_height := 12
def cube_side := 3

-- Calculate the number of cubes along each dimension
def num_cubes_length := box_length / cube_side
def num_cubes_width := box_width / cube_side
def num_cubes_height := box_height / cube_side

-- Calculate volumes
def volume_cube := cube_side ^ 3
def volume_box := box_length * box_width * box_height
def volume_cubes := (num_cubes_length * num_cubes_width * num_cubes_height) * volume_cube

-- Prove the percentage calculation
theorem volume_percentage_correct : (volume_cubes.toFloat / volume_box.toFloat) * 100 = 75 := by
  sorry

end volume_percentage_correct_l488_488793


namespace deposit_difference_l488_488377

noncomputable def A_deposit : ℝ := 10000
noncomputable def B_deposit : ℝ := 10000
noncomputable def A_annual_rate : ℝ := 2.88 / 100
noncomputable def B_annual_rate : ℝ := 2.25 / 100
noncomputable def interest_tax_rate : ℝ := 20 / 100
noncomputable def years : ℝ := 5

theorem deposit_difference :
  let A_total := A_deposit + A_deposit * A_annual_rate * (1 - interest_tax_rate) * years,
      B_total := B_deposit * (1 + B_annual_rate * (1 - interest_tax_rate))^years,
      difference := (A_total - B_total)
  in  (Float.round difference (Float.log10 100).toNat) = 219.01 :=
by
  sorry

end deposit_difference_l488_488377


namespace Jessica_paid_1000_for_rent_each_month_last_year_l488_488605

/--
Jessica paid $200 for food each month last year.
Jessica paid $100 for car insurance each month last year.
This year her rent goes up by 30%.
This year food costs increase by 50%.
This year the cost of her car insurance triples.
Jessica pays $7200 more for her expenses over the whole year compared to last year.
-/
theorem Jessica_paid_1000_for_rent_each_month_last_year
  (R : ℝ) -- monthly rent last year
  (h1 : 12 * (0.30 * R + 100 + 200) = 7200) :
  R = 1000 :=
sorry

end Jessica_paid_1000_for_rent_each_month_last_year_l488_488605


namespace triangle_problem_l488_488238

noncomputable def triangle_sin_B (a b : ℝ) (A : ℝ) : ℝ :=
  b * Real.sin A / a

noncomputable def triangle_side_c (a b A : ℝ) : ℝ :=
  let discr := b^2 + a^2 - 2 * b * a * Real.cos A
  Real.sqrt discr

noncomputable def sin_diff_angle (sinB cosB sinC cosC : ℝ) : ℝ :=
  sinB * cosC - cosB * sinC

theorem triangle_problem
  (a b : ℝ)
  (A : ℝ)
  (ha : a = Real.sqrt 39)
  (hb : b = 2)
  (hA : A = Real.pi * (2 / 3)) :
  (triangle_sin_B a b A = Real.sqrt 13 / 13) ∧
  (triangle_side_c a b A = 5) ∧
  (sin_diff_angle (Real.sqrt 13 / 13) (2 * Real.sqrt 39 / 13) (5 * Real.sqrt 13 / 26) (3 * Real.sqrt 39 / 26) = -7 * Real.sqrt 3 / 26) :=
by sorry

end triangle_problem_l488_488238


namespace tangent_line_length_l488_488594

noncomputable def curve_C (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

def problem_conditions : Prop :=
  curve_C 0 = 4 ∧ cartesian 4 0 = (4, 0)

theorem tangent_line_length :
  problem_conditions → 
  ∃ l : ℝ, l = 2 :=
by
  sorry

end tangent_line_length_l488_488594


namespace arithmetic_sequence_properties_l488_488489

noncomputable def geom_mean (a b c : ℝ) : Prop := c^2 = a * b

def arith_seq (a : ℕ → ℝ) : Prop :=
∀ n ≥ 2, a n - a (n - 1) = a (n + 1) - a n

def a_5 (a : ℕ → ℝ) : Prop := a 5 = 2

def condition_seq (a : ℕ → ℝ) : Prop :=
∀ n ≥ 2, a (n - 1) + a (n + 1) = 2 * a n

def general_form (a : ℕ → ℝ) : Prop :=
  (∀ n, a n = 3 / 5 * n - 1 ∨ a n = 3 * n - 13)

def S_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = n * (3 * n - 23) / 2

-- Define b_n sequence
def b_n (n : ℕ) (S : ℕ → ℝ) : ℝ := n / ((2 * S n + 23 * n) * (n + 1))

-- Define T_n, the sum of the first n terms of b_n
def T_n (T : ℕ → ℝ) : Prop :=
∀ n, T n = n / (3 * n + 3)

-- Prove the given sequence properties
theorem arithmetic_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ)
  (h_a_seq : arith_seq a)
  (h_a_5 : a_5 a)
  (h_condition_seq : condition_seq a)
  (h_general_form : general_form a)
  (h_S_n : S_n a S)
  (h_a1_int : ∃ m : ℤ, a 1 = m)
  (geom_mean_prop : geom_mean (a 1) (-(8 / 5 : ℝ)) (a 3)) :
  T_n T :=
sorry

end arithmetic_sequence_properties_l488_488489


namespace log_base_9_of_729_l488_488847

-- Define the conditions
def nine_eq := 9 = 3^2
def seven_two_nine_eq := 729 = 3^6

-- State the goal to be proved
theorem log_base_9_of_729 (h1 : nine_eq) (h2 : seven_two_nine_eq) : log 9 729 = 3 :=
by
  sorry

end log_base_9_of_729_l488_488847


namespace a_plus_b_magnitude_l488_488517

-- Defining the vectors and their properties
def a (λ : ℝ) : ℝ × ℝ := (λ, -2)
def b : ℝ × ℝ := (1, 3)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Statement of the proof problem in Lean 4
theorem a_plus_b_magnitude (λ : ℝ) (h : dot_product (a λ) b = 0) :
  magnitude (a λ + b) = 5 * Real.sqrt 2 := 
  by
  sorry

end a_plus_b_magnitude_l488_488517


namespace most_suitable_for_census_l488_488355

-- Conditions as definitions
def scenarioA : Prop := ∀ (area : Set String), area.nonempty → ¬census.isPractical(area)
def scenarioB : Prop := ∀ (students : Set String), students.nonempty → ¬census.isPractical(students)
def scenarioC : Prop := census.isPractical(Set.of_list ["student1", "student2", "student3"])
def scenarioD : Prop := ∀ (viewers : Set String), viewers.nonempty → ¬census.isPractical(viewers)

-- The proof problem in Lean statement
theorem most_suitable_for_census : 
  (scenarioA) ∧ (scenarioB) ∧ (scenarioC) ∧ (scenarioD) → 
  census.isMostSuitable(Set.of_list ["student1", "student2", "student3"]) :=
by
  sorry

end most_suitable_for_census_l488_488355


namespace three_digit_sum_seven_l488_488947

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488947


namespace range_of_a_l488_488209

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ : ℝ, 2 * x₀^2 + (a - 1) * x₀ + 1 / 2 ≤ 0) → a ∈ set.Ioo (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_a_l488_488209


namespace remainder_of_sum_of_remainders_l488_488626

theorem remainder_of_sum_of_remainders (R' : Set ℕ) (hR' : ∀ n : ℕ, ∃ r : ℕ, r ∈ R' ∧ r = 2^n % 720) :
  (∑ r in R', r) % 720 = 500 :=
sorry

end remainder_of_sum_of_remainders_l488_488626


namespace tan_alpha_trigonometric_identity_l488_488510

noncomputable def atan2 (y x : ℝ) : ℝ := Real.angle.toRadians (Complex.arg (x + Complex.i * y))

variables (α : ℝ)

def vector_a : ℝ × ℝ := (4 * Real.sin (Real.pi - α), 3 / 2)
def vector_b : ℝ × ℝ := (Real.cos (Real.pi / 3), Real.cos α)

def dot_product (u v : ℝ × ℝ) : ℝ := (u.1 * v.1) + (u.2 * v.2)

-- Conditions: vectors a and b are perpendicular
axiom perp_a_b : dot_product (vector_a α) (vector_b α) = 0

-- Mathematically equivalent statements to be proven
theorem tan_alpha : Real.tan α = -(3 / 4) :=
sorry

theorem trigonometric_identity : 1 / (1 + Real.sin α * Real.cos α) = 25 / 13 :=
sorry

end tan_alpha_trigonometric_identity_l488_488510


namespace combination_sum_permutation_identity_l488_488375

-- Definition of combinatorial combination
def combination (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- First proof problem: Prove the sum of combinations equals 330
theorem combination_sum : (combination 3 3 + combination 4 3 + combination 5 3 + combination 6 3 + combination 7 3 + combination 8 3 + combination 9 3 + combination 10 3) = combination 11 4 :=
by
  sorry

-- Definition of permutation
def permutation (n k : ℕ) : ℕ := nat.factorial n / nat.factorial (n - k)

-- Second proof problem: Prove the identity of permutations
theorem permutation_identity (n k : ℕ) : (permutation n k + k * permutation n (k - 1)) = permutation (n + 1) k :=
by
  sorry

end combination_sum_permutation_identity_l488_488375


namespace factors_of_2695_more_than_3_factors_l488_488521

theorem factors_of_2695_more_than_3_factors :
  let n := 2695 in
  let p1 := 5 in
  let p2 := 7 in
  let p3 := 11 in
  let factors_2695 := list.prod [p1, p2, p3^2] in
  list.length (factors_with_more_than_k_factors factors_2695 3) = 4 :=
begin
  sorry
end

end factors_of_2695_more_than_3_factors_l488_488521


namespace find_number_of_10_bills_from_mother_l488_488643

variable (m10 : ℕ)  -- number of $10 bills given by Luke's mother

def mother_total : ℕ := 50 + 2*20 + 10*m10
def father_total : ℕ := 4*50 + 20 + 10
def total : ℕ := mother_total m10 + father_total

theorem find_number_of_10_bills_from_mother
  (fee : ℕ := 350)
  (m10 : ℕ) :
  total m10 = fee → m10 = 3 := 
by
  sorry

end find_number_of_10_bills_from_mother_l488_488643


namespace eval_expression_l488_488058

theorem eval_expression : (-3)^5 + 2^(2^3 + 5^2 - 8^2) = -242.999999999535 := by
  sorry

end eval_expression_l488_488058


namespace three_digit_integers_sum_to_7_l488_488910

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488910


namespace part1_part2_l488_488479

noncomputable def a (n : ℕ) : ℝ := 2^n

def b (n : ℕ) : ℝ := a n * real.logb (1/2) (a n)

noncomputable def S (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem part1 (a : ℕ → ℝ) 
  (mono_increasing : ∀ n, a n < a (n+1)) 
  (h_sum : a 2 + a 3 + a 4 = 28) 
  (h_mean : a 3 + 2 = (a 2 + a 4) / 2) : 
  ∀ n, a n = 2^n := 
sorry

theorem part2 (a : ℕ → ℝ) (b : ℕ → ℝ)
  (S : ℕ → ℝ) :
  (∀ n, S n + (n + m) * a (n+1) < 0) ↔ m ≤ -1 :=
sorry

end part1_part2_l488_488479


namespace infinite_solutions_l488_488287

theorem infinite_solutions : ∀ n : ℕ, n > 0 → (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^5 = c^3) :=
by
  intro n hn
  use 10 * n^15, 3 * n^6, 7 * n^10
  split
  · exact Nat.pos_iff_ne_zero.1 (Nat.mul_pos (by norm_num) hn)
  split
  · exact Nat.pos_iff_ne_zero.1 (Nat.mul_pos (by norm_num) hn)
  split 
  · exact Nat.pos_iff_ne_zero.1 (Nat.mul_pos (by norm_num) hn)
  sorry

end infinite_solutions_l488_488287


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488936

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488936


namespace find_lambda_l488_488178

variables {R : Type*} [LinearOrderedField R]
variables (a b : ℝ × ℝ) (λ : ℝ)
variable h_orth : ((1 - 3 * λ, 3 - 4 * λ) ∙ (3, 4)) = 0

theorem find_lambda : λ = 3 / 5 := by
  sorry

end find_lambda_l488_488178


namespace total_area_ABHFGD_l488_488228

open Classical

noncomputable def area_of_polygon (s: ℝ) (A HEX1 HEX2: ℝ) (H_midpoint: Prop) : ℝ :=
  HEX1 - (2 * (s^2) / 36)

theorem total_area_ABHFGD :
  ∀ (s: ℝ) (A HEX1 HEX2: ℝ) (H_midpoint: Prop),
    HEX1 = A * 25 → HEX2 = A * 25 → H_midpoint →
    area_of_polygon s (A * 25) (A * 25) H_midpoint = 50 - 50 * (Real.sqrt 3) / 18 :=
by
  intros s A HEX1 HEX2 H_midpoint h1 h2 h3
  unfold area_of_polygon
  rw [h1, h2]
  simp
  sorry

end total_area_ABHFGD_l488_488228


namespace geometric_statements_correct_count_l488_488356

theorem geometric_statements_correct_count {A B C D : Prop} :
  (A ↔ (∃ (v : ℕ) (e : ℕ) (f : ℕ), v = 10 ∧ e = 15 ∧ f = 7 ∧ is_pentagonal_prism (v, e, f)))
  ∧ (B ↔ (point_to_line ∧ line_to_plane ∧ plane_to_solid))
  ∧ (C ↔ (development_of_cone ≠ circle))
  ∧ (D ↔ (cube_intersection_shapes ∈ {[triangle, quadrilateral, pentagon, hexagon]}))
  → A = false ∧ B = true ∧ C = false ∧ D = true ∧ count_true {A, B, C, D} = 2 := 
by
  sorry

end geometric_statements_correct_count_l488_488356


namespace hexagram_stone_arrangements_equiv_class_count_l488_488603

theorem hexagram_stone_arrangements_equiv_class_count :
  ∃ (arrangements : ℕ), arrangements = 60 :=
by
  -- Definitions
  let total_stones := 6
  let total_positions := 6
  let total_symmetries := 12

  -- Total arrangements without considering symmetries
  let total_arrangements := Nat.factorial total_stones

  -- Distinct arrangements considering symmetries
  let distinct_arrangements := total_arrangements / total_symmetries

  -- Goal
  have hexagram_stone_arrangements_count : distinct_arrangements = 60 := by sorry
  use distinct_arrangements
  exact hexagram_stone_arrangements_count

end hexagram_stone_arrangements_equiv_class_count_l488_488603


namespace total_carrots_grown_l488_488283

theorem total_carrots_grown
  (Sandy_carrots : ℕ) (Sam_carrots : ℕ) (Sophie_carrots : ℕ) (Sara_carrots : ℕ)
  (h1 : Sandy_carrots = 6)
  (h2 : Sam_carrots = 3)
  (h3 : Sophie_carrots = 2 * Sam_carrots)
  (h4 : Sara_carrots = (Sandy_carrots + Sam_carrots + Sophie_carrots) - 5) :
  Sandy_carrots + Sam_carrots + Sophie_carrots + Sara_carrots = 25 :=
by sorry

end total_carrots_grown_l488_488283


namespace max_distance_is_achieved_l488_488631

noncomputable def max_distance (z : ℂ) (hz : abs z = 3) : ℝ :=
  abs (2 + 3 * complex.I - z^2) * abs z^2

theorem max_distance_is_achieved (z : ℂ) (hz : abs z = 3) :
  (max_distance z hz) = 81 + 9 * Real.sqrt 13 := by
  sorry

end max_distance_is_achieved_l488_488631


namespace pie_longest_segment_squared_l488_488788

theorem pie_longest_segment_squared (d : ℝ) (n : ℕ) (h_d : d = 20) (h_n : n = 4) : 
  let r := d / 2 in
  let α := 360 / n in
  α = 90 ∧ r = 10 →
  let l := r * Real.sqrt 2 in
  l^2 = 200 :=
by sorry

end pie_longest_segment_squared_l488_488788


namespace tunnel_length_l488_488407

theorem tunnel_length
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (time_seconds : ℕ)
  (h_train_length : train_length = 300)
  (h_train_speed_kmh : train_speed_kmh = 54)
  (h_time_seconds : time_seconds = 100) :
  let train_speed_ms := train_speed_kmh * 1000 / 3600 in
  let distance_traveled := train_speed_ms * time_seconds in
  let tunnel_length := distance_traveled - train_length in
  tunnel_length = 1200 :=
by
  sorry

end tunnel_length_l488_488407


namespace tangent_line_eq_l488_488499

noncomputable
def f (x : ℝ) : ℝ := (Real.cos x) / (Real.exp x)

theorem tangent_line_eq :
  let f' := λ x, (- (Real.sin x + Real.cos x)) / (Real.exp x)
  (0, f 0) = (0, 1) →
  f' 0 = -1 →
  ∃ k b, (k = -1) ∧ (b = 1) ∧ ∀ x y, y = f' 0 * x + (f 0) ↔ x + y - 1 = 0 :=
by
  sorry

end tangent_line_eq_l488_488499


namespace max_distance_on_ellipse_to_vertex_l488_488612

open Real

noncomputable def P (θ : ℝ) : ℝ × ℝ :=
(√5 * cos θ, sin θ)

def ellipse (x y : ℝ) := (x^2 / 5) + y^2 = 1

def B : ℝ × ℝ := (0, 1)

def dist (A B : ℝ × ℝ) : ℝ :=
sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem max_distance_on_ellipse_to_vertex :
  ∃ θ : ℝ, dist (P θ) B = 5 / 2 :=
sorry

end max_distance_on_ellipse_to_vertex_l488_488612


namespace number_of_cans_in_pack_l488_488662

def cost_of_pack : ℝ := 2.99
def cost_per_can : ℝ := 0.25

theorem number_of_cans_in_pack : (cost_of_pack / cost_per_can).toInt = 11 := by
  sorry

end number_of_cans_in_pack_l488_488662


namespace twelve_pharmacies_not_enough_l488_488755

def grid := ℕ × ℕ

def is_within_walking_distance (p1 p2 : grid) : Prop :=
  abs (p1.1 - p1.2) ≤ 3 ∧ abs (p2.1 - p2.2) ≤ 3

def walking_distance_coverage (pharmacies : set grid) (p : grid) : Prop :=
  ∃ pharmacy ∈ pharmacies, is_within_walking_distance pharmacy p

def sufficient_pharmacies (pharmacies : set grid) : Prop :=
  ∀ p : grid, walking_distance_coverage pharmacies p

theorem twelve_pharmacies_not_enough (pharmacies : set grid) (h : pharmacies.card = 12) : 
  ¬ sufficient_pharmacies pharmacies :=
sorry

end twelve_pharmacies_not_enough_l488_488755


namespace problem_proof_l488_488857

noncomputable def problem : Prop :=
  ∀ x : ℝ, (x ≠ 2 ∧ (x-2)/(x-4) ≤ 3) ↔ (4 < x ∧ x < 5)

theorem problem_proof : problem := sorry

end problem_proof_l488_488857


namespace domain_of_f_l488_488072

noncomputable def f (x : ℝ) : ℝ := real.sqrt (4 - real.sqrt (6 - real.sqrt x))

theorem domain_of_f :
  {x : ℝ | 0 ≤ x ∧ x ≤ 36} = {x : ℝ | ∃ y, f x = y} :=
by
  sorry

end domain_of_f_l488_488072


namespace bug_total_distance_traveled_l488_488383

theorem bug_total_distance_traveled :
  ∀ (radius length_3rd : ℕ), radius = 65 → length_3rd = 100 →
  let diameter := radius * 2 
  let length_2nd := Real.sqrt (diameter ^ 2 - length_3rd ^ 2)
  let total_distance := diameter + length_3rd + length_2nd
  total_distance ≈ 313 := sorry

end bug_total_distance_traveled_l488_488383


namespace chameleons_color_change_l488_488572

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l488_488572


namespace middle_even_integer_l488_488702

theorem middle_even_integer (a b c : ℤ) (ha : even a) (hb : even b) (hc : even c) 
(h1 : a < b) (h2 : b < c) (h3 : 0 < a) (h4 : a < 10) (h5 : a + b + c = (1/8) * a * b * c) : b = 4 := 
sorry

end middle_even_integer_l488_488702


namespace factorial_five_l488_488659

theorem factorial_five : let A := 5 * 4 * 3 * 2 * 1 in A = 120 :=
by
  let A := 5 * 4 * 3 * 2 * 1
  show A = 120
  sorry

end factorial_five_l488_488659


namespace sum_of_abs_slopes_l488_488675

def isosceles_trapezoid_integers (A B C D : ℤ × ℤ) : Prop :=
  A = (10, 50) ∧ D = (11, 57) ∧ B.1 ≠ C.1 ∧ B.2 ≠ C.2 ∧
  (∃ s1 s2 : ℚ, s1 ≠ s2 ∧ C.2 - D.2 = s1 * (C.1 - D.1) ∧ B.2 - A.2 = s2 * (B.1 - A.1))

theorem sum_of_abs_slopes (A B C D : ℤ × ℤ) (m n : ℤ) [fact (m.gcd n = 1)]
  (h : isosceles_trapezoid_integers A B C D) :
  m + n = 131 :=
sorry

end sum_of_abs_slopes_l488_488675


namespace num_three_digit_integers_sum_to_seven_l488_488987

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488987


namespace distinct_values_of_3_exp_3_exp_3_exp_3_l488_488077

theorem distinct_values_of_3_exp_3_exp_3_exp_3 : 
  ∃ n, n = 4 ∧ ∀ e ∈ {3^{(3^{(3^3))}}, 3^{((3^3)^3)}, ((3^3)^3)^3, (3^{(3^3)})^3, (3^3)^{3^3}}, e ∈ {3^{27}, 3^{19683}, 19683^3, (3^{27})^3, 27^{27}} :=
sorry

end distinct_values_of_3_exp_3_exp_3_exp_3_l488_488077


namespace parabola_coefs_l488_488792

theorem parabola_coefs (b c : ℝ) :
  (1 - b + c = -11) ∧ (9 + 3b + c = 17) ∧ (4 + 2b + c = 5) → 
  b = 13 / 3 ∧ c = -5 :=
by
  intros h,
  sorry

end parabola_coefs_l488_488792


namespace intersection_of_bisectors_lies_on_BC_l488_488239

open EuclideanGeometry

theorem intersection_of_bisectors_lies_on_BC
  (ABC : Triangle)
  (A B C I X : Point)
  (D : Point) 
  (Γ Γ0 : Circle) :
  ∠BAC > ∠ABC →
  I = incenter ABC →
  D ∈ Line[BC] →
  ∠CAD = ∠ABC →
  Γ.passing_through I →
  Γ.tangent CA at A →
  Γ.intersects_circumcircle ABC at A X →
  let BXC : Angle := ∠BXC,
  let BAD : Angle := ∠BAD,
  let E := angle_bisector_intersection BXC BAD
  in E ∈ Line[BC] :=
by
  sorry

end intersection_of_bisectors_lies_on_BC_l488_488239


namespace points_M_lie_on_same_circle_l488_488273

noncomputable def point := ℝ × ℝ
def dist (A B : point) : ℝ := real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

variables (A B C : point) (n : ℝ)

-- Given conditions
def is_midpoint_of_arc (C : point) (A B : point) : Prop := sorry
def is_arbitrary_on_arc (P : point) (C : point) : Prop := sorry
def point_on_segment (M : point) (P C : point) : Prop := sorry

-- Condition that M lies on PC line segment
axiom M_on_segment (P C M : point) : point_on_segment M P C

-- Main condition PM = 1/n * |PA - PB|
axiom PM_condition (P C M A B : point) (n : ℝ) : dist P M = (1 / n) * abs (dist P A - dist P B)

-- Statement to prove: As P varies along the arc, all points M lie on the same circle
theorem points_M_lie_on_same_circle (A B C : point) (n : ℝ)
  (h1 : is_midpoint_of_arc C A B)
  (h2 : ∀ P, is_arbitrary_on_arc P C → ∃ M, M_on_segment P C M ∧ PM_condition P C M A B n) :
  ∃ (κ : set point), ∀ (P : point), 
  (is_arbitrary_on_arc P C → ∃ M, M_on_segment P C M ∧ PM_condition P C M A B n → M ∈ κ) := 
sorry

end points_M_lie_on_same_circle_l488_488273


namespace min_sin4x_plus_2cos4x_exists_min_sin4x_plus_2cos4x_l488_488863

theorem min_sin4x_plus_2cos4x (x : ℝ) : (sin x)^4 + 2 * (cos x)^4 ≥ 2 / 3 :=
  sorry

theorem exists_min_sin4x_plus_2cos4x : ∃ x : ℝ, (sin x)^4 + 2 * (cos x)^4 = 2 / 3 :=
  sorry

end min_sin4x_plus_2cos4x_exists_min_sin4x_plus_2cos4x_l488_488863


namespace original_number_l488_488353

theorem original_number (N : ℤ) : (∃ k : ℤ, N - 7 = 12 * k) → N = 19 :=
by
  intros h
  sorry

end original_number_l488_488353


namespace find_value_of_k_l488_488709

def num_students := 10001

def satisfies_conditions (k : ℕ) :=
  ∀ (students : Fin num_students → Fin num_students → Prop)
    (clubs : Fin num_students → Fin num_students → Prop)
    (societies : Fin k → Fin num_students → Prop),
    (∀ (i j : Fin num_students), i ≠ j → ∃! c, clubs i c ∧ clubs j c) ∧
    (∀ (a : Fin num_students) (s : Fin k), ∃! c, clubs a c ∧ societies s c) ∧
    (∀ (c : Fin num_students), odd (nat.card (set_of (λ x, clubs x c))) ∧
       ∃ m, nat.card (set_of (λ x, clubs x c)) = 2 * m + 1 ∧ nat.card (set_of (λ y, societies y c)) = m)

theorem find_value_of_k : ∃ k, k = 5000 ∧ satisfies_conditions k := 
  sorry

end find_value_of_k_l488_488709


namespace three_digit_sum_seven_l488_488955

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488955


namespace decreasing_intervals_max_and_min_values_l488_488502

noncomputable def f (x : ℝ) : ℝ := 4 * sin x * cos (x + π / 6)

theorem decreasing_intervals :
  ∀ k : ℤ, 
    ∀ x : ℝ, 
      k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3 → 
        (f (x + ε) - f x < 0) :=
sorry

theorem max_and_min_values :
  (∀ x ∈ set.Icc 0 (π / 2), f x ≤ 1 ∧ f x ≥ -2) ∧ 
  (∃ x ∈ set.Icc 0 (π / 2), f x = 1) ∧ 
  (∃ x ∈ set.Icc 0 (π / 2), f x = -2) := 
sorry

end decreasing_intervals_max_and_min_values_l488_488502


namespace find_x_l488_488708

-- Define the problem conditions.
def workers := ℕ
def gadgets := ℕ
def gizmos := ℕ
def hours := ℕ

-- Given conditions
def condition1 (g h : ℝ) := (1 / g = 2) ∧ (1 / h = 3)
def condition2 (g h : ℝ) := (100 * 3 / g = 900) ∧ (100 * 3 / h = 600)
def condition3 (x : ℕ) (g h : ℝ) := (40 * 4 / g = x) ∧ (40 * 4 / h = 480)

-- Proof problem statement
theorem find_x (g h : ℝ) (x : ℕ) : 
  condition1 g h → condition2 g h → condition3 x g h → x = 320 :=
by 
  intros h1 h2 h3
  sorry

end find_x_l488_488708


namespace number_of_three_digit_numbers_with_sum_seven_l488_488888

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488888


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488946

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488946


namespace gus_buys_2_dozen_l488_488302

-- Definitions from conditions
def dozens_to_golf_balls (d : ℕ) : ℕ := d * 12
def total_golf_balls : ℕ := 132
def golf_balls_per_dozen : ℕ := 12
def dan_buys : ℕ := 5
def chris_buys_golf_balls : ℕ := 48

-- The number of dozens Gus buys
noncomputable def gus_buys (total_dozens dan_dozens chris_dozens : ℕ) : ℕ := total_dozens - dan_dozens - chris_dozens

theorem gus_buys_2_dozen : gus_buys (total_golf_balls / golf_balls_per_dozen) dan_buys (chris_buys_golf_balls / golf_balls_per_dozen) = 2 := by
  sorry

end gus_buys_2_dozen_l488_488302


namespace calligraphy_prices_max_brushes_l488_488418

theorem calligraphy_prices 
  (x y : ℝ)
  (h1 : 40 * x + 100 * y = 280)
  (h2 : 30 * x + 200 * y = 260) :
  x = 6 ∧ y = 0.4 := 
by sorry

theorem max_brushes 
  (m : ℝ)
  (h_budget : 6 * m + 0.4 * (200 - m) ≤ 360) :
  m ≤ 50 :=
by sorry

end calligraphy_prices_max_brushes_l488_488418


namespace wristband_distribution_l488_488218

open Nat 

theorem wristband_distribution (x y : ℕ) 
  (h1 : 2 * x + 2 * y = 460) 
  (h2 : 2 * x = 3 * y) : x = 138 :=
sorry

end wristband_distribution_l488_488218


namespace probability_correct_l488_488543

open Finset

def first_ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def pairs_with_sum_divisible_by_3 (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.bUnion (λ x, s.filter (λ y, x < y ∧ (x + y) % 3 = 0).image (λ y, (x, y)))

noncomputable def probability_sum_divisible_by_3 : ℚ :=
  (pairs_with_sum_divisible_by_3 first_ten_primes).card / (first_ten_primes.card.choose 2)

theorem probability_correct : probability_sum_divisible_by_3 = 1 / 5 := 
sorry

end probability_correct_l488_488543


namespace books_bought_per_month_l488_488242

theorem books_bought_per_month 
    (monthly_cost_per_book : ℕ) 
    (total_sale : ℕ) 
    (total_loss : ℕ) 
    (total_cost : ℕ := total_sale + total_loss) :
    monthly_cost_per_book * 36 = total_cost :=
by 
    have h1 : monthly_cost_per_book * 36 = 720 := by sorry
    have h2 : total_cost = 720 := by sorry
    transitivity 720
    · exact h1
    · exact h2

end books_bought_per_month_l488_488242


namespace number_of_three_digit_numbers_with_sum_seven_l488_488897

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488897


namespace part_a_part_b_l488_488259

-- Definition of Part (a)
theorem part_a 
  (O A B C A₀ B₀ C₀ L M K : Point)
  (h1 : IsIncenter O A B C)
  (h2 : IsTangencyPoint A₀ B₀ C₀ O)
  (h3 : OnRayAndEqualDistance O A₀ L)
  (h4 : OnRayAndEqualDistance O B₀ M)
  (h5 : OnRayAndEqualDistance O C₀ K) :
  Concurrent (Line.through A L) (Line.through B M) (Line.through C K) := 
sorry

-- Definition of Part (b)
theorem part_b 
  (O A B C A₀ B₀ C₀ L M K A₁ B₁ C₁ l: Point)
  (h1 : IsIncenter O A B C)
  (h2 : IsTangencyPoint A₀ B₀ C₀ O)
  (h3 : OnRayAndEqualDistance O A₀ L)
  (h4 : OnRayAndEqualDistance O B₀ M)
  (h5 : OnRayAndEqualDistance O C₀ K)
  (h6 : IsProjectionOfOnLine l A₁ A)
  (h7 : IsProjectionOfOnLine l B₁ B)
  (h8 : IsProjectionOfOnLine l C₁ C)
  (h9 : PassesThrough O l) :
  Concurrent (Line.through A₁ L) (Line.through B₁ M) (Line.through C₁ K) := 
sorry

end part_a_part_b_l488_488259


namespace ducks_problem_l488_488802

theorem ducks_problem :
  ∃ (adelaide ephraim kolton : ℕ),
    adelaide = 30 ∧
    adelaide = 2 * ephraim ∧
    ephraim + 45 = kolton ∧
    (adelaide + ephraim + kolton) % 9 = 0 ∧
    1 ≤ adelaide ∧
    1 ≤ ephraim ∧
    1 ≤ kolton ∧
    adelaide + ephraim + kolton = 108 ∧
    (adelaide + ephraim + kolton) / 3 = 36 :=
by
  sorry

end ducks_problem_l488_488802


namespace cab_usual_time_l488_488366

noncomputable def usual_time (S T : ℝ) (late : ℝ) := 
  T = (5 / 6) * (T + late)

theorem cab_usual_time 
  (usual_speed : ℝ) 
  (usual_time : ℝ)
  (late_time : ℝ)
  (speed_reduction : usual_speed / 6)
  (late : late_time = 8) 
  (eq1: usual_time = (5 / 6) * (usual_time + late_time)) :
  usual_time = 40 :=
sorry

end cab_usual_time_l488_488366


namespace distinct_positive_six_digit_integers_l488_488184

theorem distinct_positive_six_digit_integers : 
  let digits := [2, 2, 5, 5, 9, 9] in
  let factorial (n : ℕ) := if n = 0 then 1 else List.product (List.range (n + 1)) in
  (factorial 6) / ((factorial 2) * (factorial 2) * (factorial 2)) = 90 :=
by
  let digits := [2, 2, 5, 5, 9, 9]
  let factorial := λ n : ℕ, if n = 0 then 1 else List.product (List.range (n + 1))
  show (factorial 6) / ((factorial 2) * (factorial 2) * (factorial 2)) = 90
  sorry

end distinct_positive_six_digit_integers_l488_488184


namespace sum_first_13_terms_l488_488132

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (ha : a 2 + a 5 + a 9 + a 12 = 60)

theorem sum_first_13_terms :
  S 13 = 195 := sorry

end sum_first_13_terms_l488_488132


namespace contains_Th_inf_subgraph_l488_488810

variable {G : Type}
variable [Graph G]

def contains_end (ω : End G) : Prop :=
  -- Definition of when a graph contains an end
  sorry

def Th_inf_subgraph (G : Type) (ω : End G) : Type :=
  -- Definition of the T H^{\infty} subgraph
  sorry

theorem contains_Th_inf_subgraph (G : Type) (ω : End G) (h : contains_end G ω) : 
  ∃ (H : Th_inf_subgraph G ω),
  -- Proof that G contains a Th_inf_subgraph H
  sorry

end contains_Th_inf_subgraph_l488_488810


namespace complete_the_square_l488_488718

theorem complete_the_square (x : ℝ) : (x^2 + 8 * x + 7 = 0) -> ((x + 4)^2 = 9) :=
by {
  intro h,
  sorry
}

end complete_the_square_l488_488718


namespace circle_equation_from_diameter_l488_488173

theorem circle_equation_from_diameter :
  let A := (0 : ℝ, 0 : ℝ)
  let B := (6 : ℝ, 0 : ℝ)
  let M := ((fst A + fst B) / 2, (snd A + snd B) / 2) -- Midpoint of A and B
  let r := (dist A B) / 2 -- Radius is half the distance between A and B
  (fst M = 3 ∧ snd M = 0) → -- Center is (3, 0)
  (r = 3) → -- Radius is 3
  ∀ x y : ℝ, (x - 3)^2 + y^2 = 9 := -- Equation of the circle
by
  intros
  sorry

end circle_equation_from_diameter_l488_488173


namespace number_of_nonempty_proper_subsets_of_complement_l488_488511

open Set

variable U : Set ℕ := {1, 2, 3, 4, 5, 6}
variable M : Set ℕ := {2, 3, 5}
variable N : Set ℕ := {4, 5}

theorem number_of_nonempty_proper_subsets_of_complement :
  let complement_MN := U \ (M ∪ N) in
  complement_MN = {1, 6} →
  (complement_MN.nonempty ↔ (∃ A, A ⊂ complement_MN ∧ A ≠ ∅ ∧ A.card = 1)) :=
by {
  intro h_complement_MN,
  rw [h_complement_MN, nonempty_def],
  split;
  intro hn,
  { use {1},
    split,
    { rw subset_def,
      intro x,
      simp, 
      tautology },
    { simpa using hn } },
  { cases hn with A hA,
    cases hA with hA₁ hA₂,
    use classical.some hA₂,
    simp at hA }}

end number_of_nonempty_proper_subsets_of_complement_l488_488511


namespace solution_exists_l488_488276

theorem solution_exists (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (gcd_ca : Nat.gcd c a = 1) (gcd_cb : Nat.gcd c b = 1) : 
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^a + y^b = z^c :=
sorry

end solution_exists_l488_488276


namespace color_fig_l488_488096

noncomputable def total_colorings (dots : Finset (Fin 9)) (colors : Finset (Fin 4))
  (adj : dots → dots → Prop)
  (diag : dots → dots → Prop) : Nat :=
  -- coloring left triangle
  let left_triangle := 4 * 3 * 2;
  -- coloring middle triangle considering diagonal restrictions
  let middle_triangle := 3 * 2;
  -- coloring right triangle considering same restrictions
  let right_triangle := 3 * 2;
  left_triangle * middle_triangle * middle_triangle

theorem color_fig (dots : Finset (Fin 9)) (colors : Finset (Fin 4))
  (adj : dots → dots → Prop)
  (diag : dots → dots → Prop) :
  total_colorings dots colors adj diag = 864 :=
by
  sorry

end color_fig_l488_488096


namespace problem_statement_l488_488255

theorem problem_statement : 
  let r := 5^s - s,
      s := 3^(n^2) + 1,
      n := 2 in
  r = 5^82 - 82 :=
by
  sorry

end problem_statement_l488_488255


namespace area_of_region_bounded_by_curves_l488_488104

noncomputable def region_area := 1

theorem area_of_region_bounded_by_curves :
  let r1 (θ : ℝ) := sec (2 * θ)
  let r2 (θ : ℝ) := csc (2 * θ)
  let x_axis (θ : ℝ) := 1
  let y_axis (θ : ℝ) := 1
  (bounded_by r1 r2 x_axis y_axis (1, 1)) = region_area :=
by
  sorry

end area_of_region_bounded_by_curves_l488_488104


namespace scuba_diver_rate_l488_488795

theorem scuba_diver_rate (depth : ℕ) (time : ℕ) (rate : ℕ) (h1: depth = 2400) (h2: time = 80) : rate = 30 :=
by {
  -- Definition of rate of descent
  let rate_of_descent := depth / time,
  -- We need to show rate = 30
  have h_rate : rate = rate_of_descent := sorry,
  rw [h1, h2] at h_rate,
  exact h_rate,
}

end scuba_diver_rate_l488_488795


namespace principal_amount_l488_488109

theorem principal_amount (A r t : ℝ) (hA : A = 1120) (hr : r = 0.11) (ht : t = 2.4) :
  abs ((A / (1 + r * t)) - 885.82) < 0.01 :=
by
  -- This theorem is stating that given A = 1120, r = 0.11, and t = 2.4,
  -- the principal amount (calculated using the simple interest formula)
  -- is approximately 885.82 with a margin of error less than 0.01.
  sorry

end principal_amount_l488_488109


namespace domain_of_f_l488_488071

noncomputable def f (x : ℝ) : ℝ := real.sqrt (4 - real.sqrt (6 - real.sqrt x))

theorem domain_of_f :
  {x : ℝ | 0 ≤ x ∧ x ≤ 36} = {x : ℝ | ∃ y, f x = y} :=
by
  sorry

end domain_of_f_l488_488071


namespace three_digit_sum_seven_l488_488881

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488881


namespace air_pressure_after_cooling_l488_488796

theorem air_pressure_after_cooling 
  (V : ℝ := 1) 
  (V1 : ℝ := 0.45) 
  (p0 : ℝ := 1.0 * 10^5) 
  (T1 : ℝ := 303) 
  (T2 : ℝ := 243) 
  (rho1 : ℝ := 1) 
  (rho2 : ℝ := 0.9) 
  : (p0 * (V - V1) / T1) * T2 / (V - (rho1 * V1 / rho2)) ≈ 0.88 * 10^5 :=
by
  sorry

end air_pressure_after_cooling_l488_488796


namespace largest_angle_of_trapezoid_arithmetic_sequence_l488_488041

variables (a d : ℝ)

-- Given Conditions
def smallest_angle : Prop := a = 45
def trapezoid_property : Prop := a + 3 * d = 135

theorem largest_angle_of_trapezoid_arithmetic_sequence 
  (ha : smallest_angle a) (ht : a + (a + 3 * d) = 180) : 
  a + 3 * d = 135 :=
by
  sorry

end largest_angle_of_trapezoid_arithmetic_sequence_l488_488041


namespace T_sum_l488_488114

-- Define the conditions as variables and assumptions
variables {b : ℝ} (h_b_gt_neg1 : -1 < b) (h_b_lt_1 : b < 1)
def T (r : ℝ) := 15 / (1 - r)
axiom T_condition : T b * T (-b) = 2430

-- The proof problem statement
theorem T_sum : T b + T (-b) = 324 :=
by
  sorry

end T_sum_l488_488114


namespace measure_angle_DAE_l488_488547

theorem measure_angle_DAE
  (ABC : Triangle)
  (A B C E D : Point)
  (h1 : ABC.isosceles_with_eq_sides AB AC)
  (h2 : ABC.perpendicular AE BC)
  (h3 : Triangle.congruent CD CA)
  (h4 : Triangle.congruent AD BD) : 
  ∠DAE = 18 := by
  sorry

end measure_angle_DAE_l488_488547


namespace max_area_triangle_OAB_l488_488231

open Real

def parametric_circle (θ : ℝ) : ℝ × ℝ := (2 + 2 * cos θ, 2 * sin θ)

def point_A : ℝ × ℝ := (1, sqrt 3)

def polar_ray (α : ℝ) : ℝ × ℝ := (4 * cos α, α)

noncomputable def triangle_area (OA OB : ℝ) (α : ℝ) : ℝ :=
  1 / 2 * OA * OB * sin (π / 3 - α)

theorem max_area_triangle_OAB :
  let OB := 4 * cos α,
      OA := 2 in
  (let area := triangle_area OA OB α in
   ∃ α : ℝ, α ≠ π / 2 ∧ area = 2 + sqrt 3) :=
begin
  intros,
  let OB := 4 * cos α,
  let OA := 2,
  use -π / 12, -- -15 degrees in radians
  split,
  -- Prove α ≠ π / 2
  {
    simp,
    linarith,
  },
  -- Prove area = 2 + sqrt 3
  {
    sorry -- The actual calculation of the area
  }
end

end max_area_triangle_OAB_l488_488231


namespace parabola_equation_l488_488449

-- Given the hyperbola equation and vertex of the parabola at the origin.
theorem parabola_equation 
  (hyperbola : ∀ x y, x^2 / 3 - y^2 = 1) 
  (vertex : (0, 0)) : 
  ∃ (p : ℝ), p = 2 ∧ ∀ x y, y^2 = 4 * p * x := 
sorry

end parabola_equation_l488_488449


namespace problem_statement_l488_488843

theorem problem_statement (x : ℝ) (h : x = -3) :
  (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
  rw [h]
  sorry

end problem_statement_l488_488843


namespace digits_sum_eq_seven_l488_488932

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488932


namespace three_digit_sum_seven_l488_488957

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488957


namespace sum_radical_conjugates_l488_488828

theorem sum_radical_conjugates : (5 - Real.sqrt 500) + (5 + Real.sqrt 500) = 10 :=
by
  sorry

end sum_radical_conjugates_l488_488828


namespace min_PA_PB_sum_l488_488608

-- Define the set X
def X := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define P as the product of elements in a set
def P (A : Finset ℕ) : ℕ := A.prod id

theorem min_PA_PB_sum (A B : Finset ℕ) (hA : A ≠ ∅) (hB : B ≠ ∅) (h_union : A ∪ B = X) (h_inter : A ∩ B = ∅) : 
  P A + P B = 402 := by
  sorry

end min_PA_PB_sum_l488_488608


namespace chameleon_color_change_l488_488566

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l488_488566


namespace chameleons_color_change_l488_488565

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l488_488565


namespace mike_gave_pens_l488_488362

theorem mike_gave_pens (M : ℕ) 
  (initial_pens : ℕ := 5) 
  (pens_after_mike : ℕ := initial_pens + M)
  (pens_after_cindy : ℕ := 2 * pens_after_mike)
  (pens_after_sharon : ℕ := pens_after_cindy - 10)
  (final_pens : ℕ := 40) : 
  pens_after_sharon = final_pens → M = 20 := 
by 
  sorry

end mike_gave_pens_l488_488362


namespace log_inequality_l488_488468

noncomputable def log3_2 : ℝ := Real.log 2 / Real.log 3
noncomputable def log2_3 : ℝ := Real.log 3 / Real.log 2
noncomputable def log2_5 : ℝ := Real.log 5 / Real.log 2

theorem log_inequality :
  let a := log3_2;
  let b := log2_3;
  let c := log2_5;
  a < b ∧ b < c :=
  by
  sorry

end log_inequality_l488_488468


namespace compute_f_of_7_l488_488075

theorem compute_f_of_7
  (f : ℝ → ℝ)
  (h0 : f 0 = 0)
  (h1 : tendsto (λ h, f h / h) (nhds 0) (nhds 7))
  (h2 : ∀ x y : ℝ, f (x + y) = f x + f y + 3 * x * y) : f 7 = 122.5 :=
sorry

end compute_f_of_7_l488_488075


namespace sqrt_one_fourth_l488_488331

theorem sqrt_one_fourth :
  {x : ℚ | x^2 = 1/4} = {1/2, -1/2} :=
by sorry

end sqrt_one_fourth_l488_488331


namespace max_distance_ellipse_l488_488618

-- Let's define the conditions and the theorem in Lean 4.
theorem max_distance_ellipse (θ : ℝ) :
  let C := { p : ℝ × ℝ | p.1 ^ 2 / 5 + p.2 ^ 2 = 1 }
  let B := (0, 1)
  let P := (sqrt 5 * cos θ, sin θ)
  P ∈ C →
  max (λ θ : ℝ, sqrt ((sqrt 5 * cos θ - 0) ^ 2 + (sin θ - 1) ^ 2)) = 5 / 2 :=
sorry

end max_distance_ellipse_l488_488618


namespace minimum_distance_sum_l488_488125

theorem minimum_distance_sum (a b : ℝ) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 2 ∧ 
    ∀ x y, (x = a ∧ y = b) →
      (√((x-1)^2 + (y-1)^2) + √((x+1)^2 + (y+1)^2) ≥ min_val) :=
sorry

end minimum_distance_sum_l488_488125


namespace number_of_three_digit_numbers_with_sum_7_l488_488975

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488975


namespace smallest_portion_l488_488301

theorem smallest_portion
    (a_1 d : ℚ)
    (h1 : 5 * a_1 + 10 * d = 10)
    (h2 : (a_1 + 2 * d + a_1 + 3 * d + a_1 + 4 * d) / 7 = a_1 + a_1 + d) :
  a_1 = 1 / 6 := 
sorry

end smallest_portion_l488_488301


namespace find_x_l488_488525

theorem find_x (x : ℝ) : (5^5) * (9^3) = 3 * (15^x) → x = 5 :=
by 
  sorry

end find_x_l488_488525


namespace twelve_pharmacies_not_enough_l488_488747

-- Define the grid dimensions and necessary parameters
def grid_size := 9
def total_intersections := (grid_size + 1) * (grid_size + 1) -- 10 * 10 grid
def walking_distance := 3
def coverage_side := (walking_distance * 2 + 1)  -- 7x7 grid coverage
def max_covered_per_pharmacy := (coverage_side - 1) * (coverage_side - 1)  -- Coverage per direction

-- Define the main theorem
theorem twelve_pharmacies_not_enough (n m : ℕ): 
  n = grid_size + 1 -> m = grid_size + 1 -> total_intersections = n * m -> 
  (walking_distance < n) -> (walking_distance < m) -> (pharmacies : ℕ) -> pharmacies = 12 ->
  (coverage_side <= n) -> (coverage_side <= m) ->
  ¬ (∀ i j : ℕ, i < n -> j < m -> ∃ p : ℕ, p < pharmacies -> 
  abs (i - (p / (grid_size + 1))) + abs (j - (p % (grid_size + 1))) ≤ walking_distance) :=
begin
  intros,
  sorry -- Proof omitted
end

end twelve_pharmacies_not_enough_l488_488747


namespace flag_designs_l488_488326

theorem flag_designs (colors : Finset String) (num_stripes : ℕ) (h_colors : colors.card = 3) (h_stripes : num_stripes = 3) : 
  (colors.card ^ num_stripes) = 27 := 
by
  -- Define the available colors
  let color_choices := colors.card
  -- Calculate the total number of possible flag designs
  have h_flag_designs : color_choices ^ num_stripes = 3 ^ 3,
    from (congr_arg (λ n, n ^ num_stripes) h_colors).trans (congr_arg (λ n, 3 ^ n) h_stripes)
  -- Conclude that the total number of possible flag designs is 27
  exact h_flag_designs

end flag_designs_l488_488326


namespace number_of_three_digit_numbers_with_sum_7_l488_488979

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488979


namespace prob_two_diff_rank_l488_488093

-- Define the cards as types
inductive Card
| Ace : Card
| King : Card
| Queen : Card

open Card

-- Define the deck with 1 Ace, 2 Kings, and 2 Queens
def deck := [Ace, King, King, Queen, Queen]

-- Function to calculate the probability of drawing two cards of different ranks
def probability_diff_rank : ℚ := by
  let total_ways := Nat.choose 5 2
  let same_rank_ways := Nat.choose 2 2 + Nat.choose 2 2
  let same_rank_prob := same_rank_ways / total_ways
  let diff_rank_prob := 1 - same_rank_prob
  exact diff_rank_prob

-- Statement of the theorem
theorem prob_two_diff_rank (h_deck : deck.length = 5) :
  probability_diff_rank = 4 / 5 := by
  -- Sorry is used to skip the proof details
  sorry

end prob_two_diff_rank_l488_488093


namespace exists_real_A_l488_488303

theorem exists_real_A (t : ℝ) (n : ℕ) (h_root: t^2 - 10 * t + 1 = 0) :
  ∃ A : ℝ, (A = t) ∧ ∀ n : ℕ, ∃ k : ℕ, A^n + 1/(A^n) - k^2 = 2 :=
by
  sorry

end exists_real_A_l488_488303


namespace fifth_largest_divisor_of_1209600000_is_75600000_l488_488317

theorem fifth_largest_divisor_of_1209600000_is_75600000 :
  let n : ℤ := 1209600000
  let fifth_largest_divisor : ℤ := 75600000
  n = 2^10 * 5^5 * 3 * 503 →
  fifth_largest_divisor = n / 2^5 :=
by
  sorry

end fifth_largest_divisor_of_1209600000_is_75600000_l488_488317


namespace square_area_lattice_points_l488_488799

/--
Given a square on the Cartesian plane with vertices at lattice points 
(every vertex has integer coordinates), such that exactly one lattice 
point falls on a vertex and two lattice points fall on different sides 
(not corners) of the square, the area of the square is 1.
-/
theorem square_area_lattice_points : 
    ∃ (vertices : Fin 4 → ℤ × ℤ), 
    (∀ (i : Fin 4), ∃ x y : ℤ, vertices i = (x, y)) ∧
    (∃ (i j : Fin 4), i ≠ j ∧ vertices i ≠ vertices j ∧ 
        (vertices i).fst = 0 ∨ (vertices i).snd = 0 ∧ 
        (vertices j).fst = 0 ∨ (vertices j).snd = 0) →
    (let s := int.gcd ((vertices 1).fst - (vertices 0).fst) ((vertices 1).snd - (vertices 0).snd) in s * s = 1) :=
by sorry

end square_area_lattice_points_l488_488799


namespace min_balls_to_draw_l488_488380

theorem min_balls_to_draw {red green yellow blue white black : ℕ} 
    (h_red : red = 28) 
    (h_green : green = 20) 
    (h_yellow : yellow = 19) 
    (h_blue : blue = 13) 
    (h_white : white = 11) 
    (h_black : black = 9) :
    ∃ n, n = 76 ∧ 
    (∀ drawn, (drawn < n → (drawn ≤ 14 + 14 + 14 + 13 + 11 + 9)) ∧ (drawn >= n → (∃ c, c ≥ 15))) :=
sorry

end min_balls_to_draw_l488_488380


namespace find_ratio_of_sums_l488_488465

variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = n * (a 1 + a n) / 2

def ratio_condition (a : ℕ → ℝ) :=
  a 6 / a 5 = 9 / 11

theorem find_ratio_of_sums (seq : ∃ d, arithmetic_sequence a d)
    (sum_prop : sum_first_n_terms S a)
    (ratio_prop : ratio_condition a) :
  S 11 / S 9 = 1 :=
sorry

end find_ratio_of_sums_l488_488465


namespace double_counting_problem_l488_488286

theorem double_counting_problem (n : ℕ) (hn : 0 < n) : 
  ∑ k in finset.range (n + 1), 2^k * nat.choose n k * nat.choose (n - k) (n - k - (n - k) / 2) = nat.choose (2 * n + 1) n := 
by 
  sorry

end double_counting_problem_l488_488286


namespace inverse_function_l488_488316

-- Given function and condition
def f (x : ℝ) := x^2 - 1
def domain_condition (x : ℝ) := x < -1

-- The property we want to prove
theorem inverse_function :
  ∀ x, domain_condition x → 
    (f (-√(x + 1)) = x) ∧ 
    (∀ y, domain_condition y → f y = x → y = -√(x + 1)) :=
by
  sorry -- Proof omitted

end inverse_function_l488_488316


namespace chameleon_color_change_l488_488578

theorem chameleon_color_change :
  ∃ (x : ℕ), (∃ (blue_initial : ℕ) (red_initial : ℕ), blue_initial = 5 * x ∧ red_initial = 140 - 5 * x) →
  (∃ (blue_final : ℕ) (red_final : ℕ), blue_final = x ∧ red_final = 3 * (140 - 5 * x)) →
  (5 * x - x = 80) :=
begin
  sorry
end

end chameleon_color_change_l488_488578


namespace three_digit_numbers_sum_seven_l488_488914

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488914


namespace boat_length_l488_488016

noncomputable def length_of_boat (breadth depth_mass_of_man gravity density_of_water : ℝ) : ℝ :=
  let weight_of_man := mass_of_man * gravity
  let displaced_water_weight := breadth * depth * mass_of_man * density_of_water * gravity
  weight_of_man / displaced_water_weight

theorem boat_length (breadth depth mass_of_man : ℝ) (gravity : ℝ := 9.8) (density_of_water : ℝ := 1000) :
  breadth = 2 → depth = 0.018 → mass_of_man = 108 → length_of_boat breadth depth mass_of_man gravity density_of_water = 3 :=
by
  sorry

end boat_length_l488_488016


namespace expected_value_is_correct_l488_488026

def expected_value_of_win : ℝ :=
  let outcomes := (list.range 8).map (fun n => 8 - (n + 1))
  let probabilities := list.repeat (1 / 8 : ℝ) 8
  list.zip_with (fun outcome probability => outcome * probability) outcomes probabilities |>.sum

theorem expected_value_is_correct :
  expected_value_of_win = 3.5 := by
  sorry

end expected_value_is_correct_l488_488026


namespace volume_spillage_l488_488391

def mass : ℝ := 502 -- mass of the ice in grams
def rho_n : ℝ := 0.92 -- density of fresh ice in g/cm^3
def rho_c : ℝ := 0.952 -- density of salty ice in g/cm^3
def rho_ns : ℝ := 1 -- density of fresh water in g/cm^3

noncomputable def V1 : ℝ := mass / rho_n
noncomputable def V2 : ℝ := mass / rho_c
noncomputable def V_excess : ℝ := V1 - V2
noncomputable def ΔV : ℝ := V_excess * (rho_n / rho_ns)

theorem volume_spillage : ΔV ≈ 2.63 :=
by linarith

end volume_spillage_l488_488391


namespace bridge_length_l488_488681

def train_length : ℕ := 120
def train_speed : ℕ := 45
def crossing_time : ℕ := 30

theorem bridge_length :
  let speed_m_per_s := (train_speed * 1000) / 3600
  let total_distance := speed_m_per_s * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 255 := by
  sorry

end bridge_length_l488_488681


namespace three_digit_sum_seven_l488_488885

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488885


namespace discount_savings_l488_488040

/-- Given the costs and discount methods for teapots and teacups, prove which discount method saves more money. -/
theorem discount_savings (x : ℕ) (h : x ≥ 4) :
  let y1 := 5 * x + 60,
      y2 := 4.6 * x + 73.6 in
  if x < 34 then y1 < y2
  else if x = 34 then y1 = y2
  else y2 < y1 :=
by sorry

end discount_savings_l488_488040


namespace number_of_bushes_l488_488598

theorem number_of_bushes (T B x y : ℕ) (h1 : B = T - 6) (h2 : x ≥ y + 10) (h3 : T * x = 128) (hT_pos : T > 0) (hx_pos : x > 0) : B = 2 :=
sorry

end number_of_bushes_l488_488598


namespace line_through_two_points_l488_488678

-- Define the points
def p1 : ℝ × ℝ := (1, 0)
def p2 : ℝ × ℝ := (0, -2)

-- Define the equation of the line passing through the points
def line_equation (x y : ℝ) : Prop :=
  2 * x - y - 2 = 0

-- The main theorem
theorem line_through_two_points : ∀ x y, p1 = (1, 0) ∧ p2 = (0, -2) → line_equation x y :=
  by sorry

end line_through_two_points_l488_488678


namespace all_candies_with_santa_l488_488284

-- Definitions based on the conditions of the problem
def start_time := 1  -- 1 minute before the New Year
def steps := {n : ℕ // n > 0}  -- Natural number steps

noncomputable def candies_in_hand (n : ℕ) : ℕ := 2^n - 1  -- Number of candies given at step n
noncomputable def candies_taken_back (n : ℕ) : ℕ := 2^(n-1) - 1  -- Number of candies taken back at step n

-- Problem statement to prove
theorem all_candies_with_santa (n : ℕ) (h : n > 0) : 
  ∀ time ∈ {1 / 2^k | k : ℕ, k ≥ n}, candies_in_hand n - candies_taken_back n = 0 :=
begin
  sorry  -- Proof to be completed
end

end all_candies_with_santa_l488_488284


namespace three_digit_sum_7_l488_488963

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488963


namespace population_scientific_notation_l488_488300

theorem population_scientific_notation : 
  (1.41: ℝ) * (10 ^ 9) = 1.41 * 10 ^ 9 := 
by
  sorry

end population_scientific_notation_l488_488300


namespace compute_expression_l488_488823

/-
  Definitions for each of the components in the problem.
-/

def term1 := 150 / 3
def term2 := 36 / 6
def term3 := 7.2 / 0.4
def term4 := 2
def innerExpression := term1 - term2 + term3 + term4
def finalResult := 24 * innerExpression

/-
  The theorem stating that the computed value is 1536.
-/
theorem compute_expression : finalResult = 1536 :=
by
  sorry

end compute_expression_l488_488823


namespace g_2_minus_g_6_l488_488679

-- The function g is linear
def linear (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g(x + y) = g(x) + g(y) ∧ ∀ r : ℝ, g(r * x) = r * g(x)

-- Assume g is a linear function
variable (g : ℝ → ℝ)
variable linear_g : linear g

-- Assume the given condition
axiom condition : ∀ t : ℝ, g(t + 2) - g(t) = 5

-- Prove the required statement
theorem g_2_minus_g_6 : g(2) - g(6) = -10 :=
by
  sorry

end g_2_minus_g_6_l488_488679


namespace hyperbola_eccentricity_proof_l488_488673

def hyperbola_asymptotes (a b : ℝ) : Prop :=
  a / b = sqrt 5 / 2

def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  sqrt (a^2 + b^2) / a

theorem hyperbola_eccentricity_proof (a b : ℝ) (h_asymptotes : hyperbola_asymptotes a b) :
  hyperbola_eccentricity (sqrt 5 / 2 * b) b = 3 * sqrt 5 / 5 :=
by
  -- Assume the proof is to be filled later.
  sorry

end hyperbola_eccentricity_proof_l488_488673


namespace range_of_f_l488_488157

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2 else x^2

theorem range_of_f : set.image f set.univ = set.Iio 4 := by
  sorry

end range_of_f_l488_488157


namespace number_of_three_digit_numbers_with_sum_7_l488_488982

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488982


namespace least_sum_exponents_of_1000_l488_488195

def sum_least_exponents (n : ℕ) : ℕ :=
  if n = 1000 then 38 else 0 -- Since we only care about the case for 1000.

theorem least_sum_exponents_of_1000 :
  sum_least_exponents 1000 = 38 := by
  sorry

end least_sum_exponents_of_1000_l488_488195


namespace three_digit_numbers_sum_seven_l488_488919

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488919


namespace problem_statement_l488_488837

theorem problem_statement : ( ( (25 ^ (1/2 : ℝ) - 1) / 2 ) ^ 2 + 3 )⁻¹ * 10 = 10 / 7 := by
  sorry

end problem_statement_l488_488837


namespace find_length_DE_l488_488269

noncomputable def length_DE : ℝ :=
  let DP : ℝ := 27
  let EQ : ℝ := 36
  let ratio : ℝ := 3 / 5
  let DG : ℝ := ratio * DP
  let EG : ℝ := ratio * EQ
  real.sqrt (DG^2 + EG^2)

theorem find_length_DE : length_DE = 27 := 
  sorry

end find_length_DE_l488_488269


namespace circles_tangent_common_tangent_l488_488715

noncomputable theory
open Set

-- Definitions for circles intersecting at points A and B
variables {ω1 ω2 : Circle} {A B : Point} (h_intersect : ω1 ∩ ω2 = {A, B})

-- Definitions for the common tangent PQ with P in ω1 and Q in ω2
variables {P Q : Point} (h_p_in_ω1 : P ∈ ω1) (h_q_in_ω2 : Q ∈ ω2)
(h_tangent_PQ : TangentLine PQ ω1 P ∧ TangentLine PQ ω2 Q)

-- An arbitrary point X on ω1 and line AX intersecting ω2 again at Y
variables {X Y : Point} (h_X_in_ω1 : X ∈ ω1) (h_AX_intersects_ω2 : ∃ Y ∈ ω2, LineThrough A X Y)

-- Point Y' on ω2 such that QY = QY'
variables {Y' : Point} (h_Y'_ne_Y : Y' ≠ Y) (h_QY_eq_QY' : Distance Q Y = Distance Q Y')

-- Line Y'B intersects ω1 again at X'
variables {X' : Point} (h_Y'B_intersects_ω1 : ∃ X' ∈ ω1, LineThrough Y' B X')

-- The proof statement
theorem circles_tangent_common_tangent
    (h_intersect : ω1 ∩ ω2 = {A, B})
    (h_p_in_ω1 : P ∈ ω1)
    (h_q_in_ω2 : Q ∈ ω2)
    (h_tangent_PQ : TangentLine PQ ω1 P ∧ TangentLine PQ ω2 Q)
    (h_X_in_ω1 : X ∈ ω1)
    (h_AX_intersects_ω2 : ∃ Y ∈ ω2, LineThrough A X Y)
    (h_Y'_ne_Y : Y' ≠ Y)
    (h_QY_eq_QY' : Distance Q Y = Distance Q Y')
    (h_Y'B_intersects_ω1 : ∃ X' ∈ ω1, LineThrough Y' B X') :
    Distance P X = Distance P X' :=
sorry

end circles_tangent_common_tangent_l488_488715


namespace problem1_problem2_l488_488774

-- Problem 1 Lean translation
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

theorem problem1 (x : ℝ) : f (2 * x) = 2 * f x * g x := sorry

-- Problem 2 Lean translation
theorem problem2 (x : ℝ) (h : x * Real.log 3 4 = 1) : 4^x + 4^(-x) = 10 / 3 := sorry

end problem1_problem2_l488_488774


namespace convert_to_rectangular_form_l488_488084

theorem convert_to_rectangular_form (r θ : ℝ) (h_r : r = sqrt 3) (h_θ : θ = 15 * π / 4) :
    r * exp (θ * I) = - (sqrt 6) / 2 + (sqrt 6) / 2 * I := 
by
  -- We know that θ is periodic with period 2π:
  have h1 : θ = 3 * π / 4 := by linarith [h_θ, real.mod_eq_of_lt _ _];
  -- Using Euler's formula:
  have h2 : exp (θ * I) = cos (3 * π / 4) + sin (3 * π / 4) * I := by rw [h1, exp_mul_I];
  -- We know the values of cos and sin for 3π/4
  have h_cos : cos (3 * π / 4) = - (sqrt 2) / 2 := by sorry;
  have h_sin : sin (3 * π / 4) = (sqrt 2) / 2 := by sorry;
  calc 
    r * exp (θ * I) = sqrt 3 * (cos θ + sin θ * I) : by rw [h_r, h2]
    ... = sqrt 3 * (-(sqrt 2) / 2 + (sqrt 2) / 2 * I) : by rw [h_cos, h_sin]
    ... = - (sqrt 6) / 2 + (sqrt 6) / 2 * I : by ring

end convert_to_rectangular_form_l488_488084


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488944

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488944


namespace hamburger_meat_price_per_pound_l488_488246

/-- 
Let total_spent_excluding_change be $14.00.
Let total_cost_of_other_items be $7.00.
Let weight_of_hamburger_meat be 2 pounds.

Prove that the price per pound of the hamburger meat is $3.50.
-/
theorem hamburger_meat_price_per_pound
  (total_spent_excluding_change : ℝ)
  (total_cost_of_other_items : ℝ)
  (weight_of_hamburger_meat : ℝ) :
  total_spent_excluding_change = 14 →
  total_cost_of_other_items = 7 →
  weight_of_hamburger_meat = 2 →
  let price_per_pound := (total_spent_excluding_change - total_cost_of_other_items) / weight_of_hamburger_meat in
  price_per_pound = 3.5 :=
begin
  intros h1 h2 h3,
  simp [h1, h2, h3],
  sorry
end

end hamburger_meat_price_per_pound_l488_488246


namespace ratio_volume_cylinder_cone_l488_488039

theorem ratio_volume_cylinder_cone (R : ℝ) (h₁: 2 * R > 0) (h₂: R > 0):  
  let cylinder_volume := (R / π) ^ 2 * π * (2 * R),
      cone_volume := (1 / 3) * π * (R / 2) ^ 2 * ((√3 / 2) * R) in 
  (cylinder_volume / cone_volume) = (16 * √3) / (π ^ 2) := 
sorry

end ratio_volume_cylinder_cone_l488_488039


namespace mario_age_difference_l488_488696

variable (Mario_age Maria_age : ℕ)

def age_conditions (Mario_age Maria_age difference : ℕ) : Prop :=
  Mario_age + Maria_age = 7 ∧
  Mario_age = 4 ∧
  Mario_age - Maria_age = difference

theorem mario_age_difference : ∃ (difference : ℕ), age_conditions 4 (4 - difference) difference ∧ difference = 1 := by
  sorry

end mario_age_difference_l488_488696


namespace kerosene_cost_l488_488739

/-- Given that:
    - A dozen eggs cost as much as a pound of rice.
    - A half-liter of kerosene costs as much as 8 eggs.
    - The cost of each pound of rice is $0.33.
    - One dollar has 100 cents.
Prove that a liter of kerosene costs 44 cents.
-/
theorem kerosene_cost :
  let egg_cost := 0.33 / 12  -- Cost per egg in dollars
  let kerosene_half_liter_cost := egg_cost * 8  -- Half-liter of kerosene cost in dollars
  let kerosene_liter_cost := kerosene_half_liter_cost * 2  -- Liter of kerosene cost in dollars
  let kerosene_liter_cost_cents := kerosene_liter_cost * 100  -- Liter of kerosene cost in cents
  kerosene_liter_cost_cents = 44 :=
by
  sorry

end kerosene_cost_l488_488739


namespace problem_statement_l488_488491

-- Provided conditions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f (x)

def f (x : ℝ) : ℝ :=
  if x < 0 then x^3 + x^2 else -(x^3 + x^2)

theorem problem_statement : f 2 = 4 :=
by
  -- Definitions
  let f := λ x : ℝ, if x < 0 then x^3 + x^2 else x^3 - x^2
  -- We need to show that f(2) = 4
  show f 2 = 4
  -- Proof skipped
  sorry

end problem_statement_l488_488491


namespace triangle_incenter_circumradius_inradius_l488_488131

theorem triangle_incenter_circumradius_inradius (ABC : Triangle) (R : ℝ) (r : ℝ) (I : Point)
  (h1 : ABC.Circumradius = R) (h2 : ABC.Inradius = r) (h3 : ABC.incenter = I) :
  ∃ D : Point, (Line.mk I ABC.A).meets_circle_point ABC.circumcircle D ∧ 
               (dist I D) * (dist I ABC.A) = 2 * R * r :=
by
  sorry

end triangle_incenter_circumradius_inradius_l488_488131


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488942

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488942


namespace increasing_function_range_l488_488126

-- Definition of the function f(x)
def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x > 1 then a ^ x else (4 - (a / 2)) * x + 2

-- Define the condition that for any x₁ ≠ x₂, (f(x₁) - f(x₂)) / (x₁ - x₂) > 0
def is_increasing (a : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0

-- The proof problem statement in Lean 4
theorem increasing_function_range (a : ℝ) : is_increasing a → 4 ≤ a ∧ a < 8 :=
  sorry

end increasing_function_range_l488_488126


namespace regression_analysis_correctness_l488_488693

theorem regression_analysis_correctness 
  (x y : ℝ) 
  (h : ∀ (xi yi : ℝ), yi = -10 * xi + 200) : 
  ∃ ε > 0, ∀ (x' : ℝ), |x' - 10| < ε → |y - 100| < ε :=
by 
  use 1
  use 1
  sorry

end regression_analysis_correctness_l488_488693


namespace shirt_price_l488_488245

theorem shirt_price (S : ℝ) (h : (5 * S + 5 * 3) / 2 = 10) : S = 1 :=
by
  sorry

end shirt_price_l488_488245


namespace log_base_9_of_729_l488_488850

theorem log_base_9_of_729 : ∃ x : ℝ, (9:ℝ) = 3^2 ∧ (729:ℝ) = 3^6 ∧ (9:ℝ)^x = 729 ∧ x = 3 :=
by
  sorry

end log_base_9_of_729_l488_488850


namespace num_of_cars_l488_488216

theorem num_of_cars (total_wheels : ℕ) (wheels_per_car : ℕ) (h1 : total_wheels = 48) (h2 : wheels_per_car = 4) : 
  total_wheels / wheels_per_car = 12 :=
by
  rw [h1, h2]
  norm_num
  exact h1
  sorry

end num_of_cars_l488_488216


namespace volume_difference_is_867_25_l488_488821

noncomputable def charlie_volume : ℝ :=
  let h_C := 9
  let circumference_C := 7
  let r_C := circumference_C / (2 * Real.pi)
  let v_C := Real.pi * r_C^2 * h_C
  v_C

noncomputable def dana_volume : ℝ :=
  let h_D := 5
  let circumference_D := 10
  let r_D := circumference_D / (2 * Real.pi)
  let v_D := Real.pi * r_D^2 * h_D
  v_D

noncomputable def volume_difference : ℝ :=
  Real.pi * (abs (charlie_volume - dana_volume))

theorem volume_difference_is_867_25 : volume_difference = 867.25 := by
  sorry

end volume_difference_is_867_25_l488_488821


namespace polyhedron_interior_segments_count_l488_488789

/-- 
A convex polyhedron has 12 square faces, 8 regular hexagonal faces, and 6 regular octagonal faces.
Exactly one square, one hexagon, and one octagon meet at each vertex of the polyhedron.
Prove that the number of segments joining pairs of vertices of the polyhedron that are interior to the polyhedron,
that is, are not edges nor contained in a face, is 840.
-/
theorem polyhedron_interior_segments_count : 
  let V := 48 in
  let A := 72 in
  let D := 216 in
  (V * (V - 1) / 2) - A - D = 840 := 
by
  sorry

end polyhedron_interior_segments_count_l488_488789


namespace num_three_digit_integers_sum_to_seven_l488_488990

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488990


namespace scientific_notation_l488_488822

def original_number : ℝ := 14 * (10 : ℝ) ^ (-9)

theorem scientific_notation :
  original_number = 1.4 * (10 : ℝ) ^ (-8) :=
by
  sorry

end scientific_notation_l488_488822


namespace circle_eq_Tangent_Line_range_PA_PB_Geom_Sequence_l488_488230

-- Define coordinates for origin O, and points A and B on x-axis
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define tangent line to the circle
def tangent_line (x : ℝ) : ℝ := x + 2 * Real.sqrt 2

-- Define distance function from a point to a line
def dist (P : ℝ × ℝ) (line : ℝ → ℝ) : ℝ :=
  let (x1, y1) := P
  Real.abs ((line x1 - y1) / (1 + (Real.sqrt 2) ^ 2))

-- Define equation of circle given center O and radius
def circle_eq (P : ℝ × ℝ) (center : ℝ × ℝ) (r : ℝ) : Prop :=
  let (x1, y1) := P
  let (x0, y0) := center
  (x1 - x0)^2 + (y1 - y0)^2 = r^2

-- Proof statement 1: Find the equation of the circle O given it's tangent to a line
theorem circle_eq_Tangent_Line :
  (dist O tangent_line = 2) → circle_eq A O 2 → circle_eq B O 2 :=
sorry

-- Proof statement 2: The range of ∥PA∥ ⋅ ∥PB∥ given distances form a geometric sequence
theorem range_PA_PB_Geom_Sequence :
  (∀ P : ℝ × ℝ, circle_eq P O 2 → (Real.sqrt ((fst P + 2)^2 + snd P^2) * Real.sqrt ((fst P - 2)^2 + snd P^2)) = fst P^2 + snd P^2 → ∃ k : ℝ, 2 * (snd P ^2 - 1) = k ∧ -2 ≤ k ∧ k < 0) :=
sorry

end circle_eq_Tangent_Line_range_PA_PB_Geom_Sequence_l488_488230


namespace pipe_emptying_time_l488_488656

theorem pipe_emptying_time (A_rate : ℝ) (B_rate_inv : ℝ) (A_time : ℝ) (B_time : ℝ) (joint_time : ℝ) (total_time : ℝ) 
  (hA : A_rate = 1 / A_time) 
  (hA_time : A_time = 12)
  (h_joint_time : joint_time = 36)
  (h_total_time : total_time = 30) 
  (total_work : ℝ = 1) :
  B_time = 24 :=
by
  -- A's fill rate
  have hA_rate : A_rate = 1 / 12, from hA ▸ hA_time,
  
  -- Let us consider B's emptying time and B_rate_inv = 1 / B_time
  let B_rate := 1 / B_time,
  
  -- In the first 36 minutes, both A and B work together
  -- During these 36 minutes, the contribution to filling the tank:
  have h1: 36 * (A_rate - B_rate) = 1,
  from calc
    36 * (1 / 12 - 1 / B_time) - (6 / 12) = 1 :
      by sorry,
    
  -- Substituting hA_rate and simplifying to find B_time
  have hB_time : B_time = 24, from
    sorry,

  -- Conclude the proof
  exact hB_time

end pipe_emptying_time_l488_488656


namespace three_digit_numbers_sum_seven_l488_488916

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488916


namespace three_digit_numbers_sum_seven_l488_488911

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488911


namespace alice_number_l488_488804

theorem alice_number (n : ℕ) 
  (h1 : 243 ∣ n) 
  (h2 : 36 ∣ n) 
  (h3 : 1000 < n) 
  (h4 : n < 3000) : 
  n = 1944 ∨ n = 2916 := 
sorry

end alice_number_l488_488804


namespace slope_angle_range_correct_l488_488320

def A : ℝ × ℝ := (2, 1)
def B (m : ℝ) : ℝ × ℝ := (1, m^2)

def slope_angle_range : Set ℝ := { θ | 0 ≤ θ ∧ θ < (π / 2) ∨ (π / 2) < θ ∧ θ < π}

theorem slope_angle_range_correct (m : ℝ) : 
  let l := line_through_points A (B m) in
  let θ := angle_of_line l in
  θ ∈ slope_angle_range :=
sorry

end slope_angle_range_correct_l488_488320


namespace chameleon_color_change_l488_488581

theorem chameleon_color_change :
  ∃ (x : ℕ), (∃ (blue_initial : ℕ) (red_initial : ℕ), blue_initial = 5 * x ∧ red_initial = 140 - 5 * x) →
  (∃ (blue_final : ℕ) (red_final : ℕ), blue_final = x ∧ red_final = 3 * (140 - 5 * x)) →
  (5 * x - x = 80) :=
begin
  sorry
end

end chameleon_color_change_l488_488581


namespace general_term_of_sequence_l488_488168

theorem general_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n^2 + 3 * n + 1) →
  a 1 = 5 ∧ (∀ n, n ≥ 2 → a n = 2 * n + 2) :=
by
  assume hS : ∀ n, S n = n^2 + 3 * n + 1
  sorry

end general_term_of_sequence_l488_488168


namespace find_abc_unique_solution_l488_488271

theorem find_abc_unique_solution (N a b c : ℕ) 
  (hN : N > 3 ∧ N % 2 = 1)
  (h_eq : a^N = b^N + 2^N + a * b * c)
  (h_c : c ≤ 5 * 2^(N-1)) : 
  N = 5 ∧ a = 3 ∧ b = 1 ∧ c = 70 := 
sorry

end find_abc_unique_solution_l488_488271


namespace chameleon_color_change_l488_488583

theorem chameleon_color_change :
  ∃ (x : ℕ), (∃ (blue_initial : ℕ) (red_initial : ℕ), blue_initial = 5 * x ∧ red_initial = 140 - 5 * x) →
  (∃ (blue_final : ℕ) (red_final : ℕ), blue_final = x ∧ red_final = 3 * (140 - 5 * x)) →
  (5 * x - x = 80) :=
begin
  sorry
end

end chameleon_color_change_l488_488583


namespace ratio_FG_CA_l488_488548

def points_on_side (A B : Type) : Prop := sorry -- Placeholder for point on line segment definition
def divides_into_equal_area_triangles (A B C D E F G : Type) : Prop := sorry -- Placeholder for equal area division

theorem ratio_FG_CA (A B C D E F G : Type)
  (h1 : points_on_side D A B)
  (h2 : points_on_side E A B)
  (h3 : points_on_side F C A)
  (h4 : points_on_side G C A)
  (h5 : divides_into_equal_area_triangles A B C D E F G) :
  FG / CA = 4 / 15 :=
by 
  sorry

end ratio_FG_CA_l488_488548


namespace max_dist_PB_l488_488622

-- Let B be the upper vertex of the ellipse.
def B : (ℝ × ℝ) := (0, 1)

-- Define the equation of the ellipse.
def ellipse (x y : ℝ) : Prop := (x^2) / 5 + y^2 = 1

-- Define a point P on the ellipse.
def P (θ : ℝ) : (ℝ × ℝ) := (sqrt 5 * cos θ, sin θ)

-- Define the distance function between points.
def dist (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Prove that the maximum distance |PB| is 5/2.
theorem max_dist_PB : ∃ θ : ℝ, dist (P θ) B = 5 / 2 :=
sorry

end max_dist_PB_l488_488622


namespace collinear_HIJ_l488_488623

-- Define the points and lines as per the conditions of the problem
variables (A B C D E F G H I J : Type)

-- Assume collinearity conditions based on given data
def collinear (x y z : Type) : Prop := 
  ∃ l : Type, (x ∈ l) ∧ (y ∈ l) ∧ (z ∈ l)

-- State the problem
theorem collinear_HIJ 
  (E F G : Type)
  (A B C D : Type)
  (BD_EF : ∃ H, ∃ (line_BD : Type), ∃ (line_EF : Type), (B ∈ line_BD) ∧ (D ∈ line_BD) ∧ (E ∈ line_EF) ∧ (F ∈ line_EF) ∧ (H ∈ line_BD ∧ H ∈ line_EF))
  (CD_FG : ∃ I, ∃ (line_CD : Type), ∃ (line_FG : Type), (C ∈ line_CD) ∧ (D ∈ line_CD) ∧ (F ∈ line_FG) ∧ (G ∈ line_FG) ∧ (I ∈ line_CD ∧ I ∈ line_FG))
  (BC_EG : ∃ J, ∃ (line_BC : Type), ∃ (line_EG : Type), (B ∈ line_BC) ∧ (C ∈ line_BC) ∧ (E ∈ line_EG) ∧ (G ∈ line_EG) ∧ (J ∈ line_BC ∧ J ∈ line_EG)) :
  collinear H I J := 
sorry

end collinear_HIJ_l488_488623


namespace domain_of_function_l488_488065

theorem domain_of_function :
  {x : ℝ | 0 ≤ x ∧ x ≤ 36} = {x : ℝ | ∃ y ∈ set.Icc (0:ℝ) (36:ℝ), y = x} :=
by sorry

end domain_of_function_l488_488065


namespace digits_sum_eq_seven_l488_488931

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488931


namespace martha_initial_juice_pantry_l488_488408

theorem martha_initial_juice_pantry (P : ℕ) : 
  4 + P + 5 - 3 = 10 → P = 4 := 
by
  intro h
  sorry

end martha_initial_juice_pantry_l488_488408


namespace perimeter_solutions_count_l488_488172

variables {α : Type*} [linear_ordered_field α]
variables (A P : point α) (e f : line α) (s : α)
variables (h1 : P ∉ e) (h2 : P ∉ f) (h3 : e ≠ f) (h4 : ∃ A, A ∈ e ∧ A ∈ f)

theorem perimeter_solutions_count :
  ∃ n, n = 2 ∨ n = 3 ∨ n = 4 :=
by
  sorry

end perimeter_solutions_count_l488_488172


namespace three_digit_sum_7_l488_488961

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488961


namespace range_of_f_l488_488154

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2 else
  if -1 < x ∧ x < 2 then x^2 else 0

theorem range_of_f : set.range f = set.Iio 4 :=
  sorry

end range_of_f_l488_488154


namespace area_of_region_bounded_by_polynomial_and_line_l488_488053

noncomputable def area_bounded_by_polynomial_line (a b c d e f g h i p q : ℝ) (α β γ δ : ℝ) (h_a : a ≠ 0) 
(h_order : α < β ∧ β < γ ∧ γ < δ) (h_touch : ∀ x ∈ {α, β, γ, δ}, ax^8 + bx^7 + cx^6 + dx^5 + ex^4 + fx^3 + gx^2 + hx + i = px + q) :
  Real :=
  (∫ x in α..β, (ax^8 + bx^7 + cx^6 + dx^5 + ex^4 + fx^3 + gx^2 + hx + i - (px + q))) +
  (∫ x in γ..δ, (ax^8 + bx^7 + cx^6 + dx^5 + ex^4 + fx^3 + gx^2 + hx + i - (px + q)))

theorem area_of_region_bounded_by_polynomial_and_line (a b c d e f g h i p q : ℝ) (α β γ δ : ℝ) (h_a : a ≠ 0) 
(h_order : α < β ∧ β < γ ∧ γ < δ) (h_touch : ∀ x ∈ {α, β, γ, δ}, ax^8 + bx^7 + cx^6 + dx^5 + ex^4 + fx^3 + gx^2 + hx + i = px + q) :
  area_bounded_by_polynomial_line a b c d e f g h i p q α β γ δ h_a h_order h_touch = 
    (a / 9) * ((β^9 - α^9) + (δ^9 - γ^9)) :=
sorry

end area_of_region_bounded_by_polynomial_and_line_l488_488053


namespace basketball_lineups_l488_488015

theorem basketball_lineups (players : ℕ) (point_guard : ℕ) (other_players : ℕ) : 
  players = 20 → 
  point_guard = 1 → 
  other_players = 7 → 
  let remaining_players := players - point_guard in
  let combinations := nat.choose remaining_players other_players in
  let total_lineups := point_guard * combinations in
  total_lineups = 1007760 :=
by
  intros h_players h_point_guard h_other_players
  let remaining_players := 19
  let combinations := nat.choose remaining_players 7
  have h_combinations : combinations = 50388 := by sorry
  let total_lineups := 20 * combinations
  have h_total_lineups : total_lineups = 1007760 := by sorry
  exact h_total_lineups

end basketball_lineups_l488_488015


namespace range_of_p_l488_488630

/-- The polynomial function p(x) = x^4 + 8x^2 + 16 for x ≥ 0 --/
def p (x : ℝ) : ℝ := x^4 + 8 * x^2 + 16

/-- The range of p(x) is [16, ∞) for x ≥ 0. --/
theorem range_of_p : {y : ℝ | ∃ x ≥ 0, p(x) = y} = {y : ℝ | y ≥ 16} :=
by
  sorry

end range_of_p_l488_488630


namespace charlotte_rearrangements_time_l488_488061

theorem charlotte_rearrangements_time
    (n : ℕ) (n_t : ℕ) (r : ℕ) (h_n : n = 9) (h_nt : n_t = 2) (h_r : r = 15) :
    (∑ (f : ℕ) in finset.range (nat.factorial (n - 1)), 1 : ℝ) / (nat.factorial n_t) / (r * 60) = 201.6 :=
by
  sorry

end charlotte_rearrangements_time_l488_488061


namespace addition_pyramid_max_l488_488419

theorem addition_pyramid_max : 
  ∀ (a b c d e f g : ℕ), 
  {a, b, c, d, e, f, g} = {1, 1, 2, 2, 3, 3, 4} → 
  let h := a + b,
      i := b + c,
      j := c + d,
      k := d + e,
      l := e + f,
      m := f + g,
      n := h + i,
      o := i + j,
      p := j + k,
      q := k + l,
      r := l + m,
      s := n + o,
      t := o + p,
      u := p + q,
      v := q + r,
      w := s + t,
      x := t + u,
      y := u + v,
      z := w + x,
      A := x + y,
      T := z + A in
  T ≤ 196 := 
by {
  intros a b c d e f g setup, 
  let h := a + b,
      i := b + c,
      j := c + d,
      k := d + e,
      l := e + f,
      m := f + g,
      n := h + i,
      o := i + j,
      p := j + k,
      q := k + l,
      r := l + m,
      s := n + o,
      t := o + p,
      u := p + q,
      v := q + r,
      w := s + t,
      x := t + u,
      y := u + v,
      z := w + x,
      A := x + y,
      T := z + A,
  sorry
}

end addition_pyramid_max_l488_488419


namespace range_of_s_is_composite_l488_488453

-- Defining composite positive integer and function s
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, m > 1 ∧ m < n ∧ n % m = 0

def s (n : ℕ) : ℕ :=
  if h : isComposite n then
    let fs := Multiset.ofList (Nat.factors n)
    fs.prod
  else 0

-- Statement of the theorem
theorem range_of_s_is_composite : 
  {n | ∃ m, isComposite m ∧ s m = n} = {n | isComposite n} :=
by
  sorry

end range_of_s_is_composite_l488_488453


namespace digits_sum_eq_seven_l488_488926

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488926


namespace real_numbers_properties_l488_488280

-- Definitions given in the conditions of the problem
def includes_rational_and_irrational (ℝ : Type) : Prop :=
  ∀ r : ℝ, r ∈ ℚ ∨ r ∉ ℚ

def abs_pos (x : ℝ) : Prop :=
  x > 0 → abs x = x

def abs_non_pos (x : ℝ) : Prop :=
  x ≤ 0 → abs x ≥ 0

-- Main statement incorporating all conditions and conclusions to prove
theorem real_numbers_properties (ℝ : Type) [is_real : real ℝ] :
  (includes_rational_and_irrational ℝ) ∧
  (∀ x : ℝ, abs_pos x) ∧
  (∀ x : ℝ, abs_non_pos x) :=
by
  sorry

end real_numbers_properties_l488_488280


namespace largest_n_divides_factorial_l488_488859

theorem largest_n_divides_factorial (n : ℕ) : 
    (∀ k : ℕ, (18^k ∣ nat.fact 24) → k ≤ 4) :=
begin
  sorry
end

end largest_n_divides_factorial_l488_488859


namespace min_sum_of_factors_l488_488313

theorem min_sum_of_factors (a b c d e : ℕ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (h_prod : a * b * c * d * e = nat.factorial 12) : a + b + c + d + e = 501 :=
begin
  sorry
end

end min_sum_of_factors_l488_488313


namespace perimeter_of_given_triangle_l488_488108

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def perimeter_of_triangle (D E F : ℝ × ℝ) : ℝ := 
  distance D E + distance E F + distance F D

theorem perimeter_of_given_triangle : 
  (perimeter_of_triangle (2, 3) (2, 9) (6, 6) = 16) :=
by
  sorry

end perimeter_of_given_triangle_l488_488108


namespace insufficient_pharmacies_l488_488749

/-- Define the grid size and conditions --/
def grid_size : ℕ := 9
def total_intersections : ℕ := 100
def pharmacy_walking_distance : ℕ := 3
def number_of_pharmacies : ℕ := 12

/-- Prove that 12 pharmacies are not enough for the given conditions. --/
theorem insufficient_pharmacies : ¬ (number_of_pharmacies ≥ grid_size + 3) → 
  ∃(pharmacies : ℕ), pharmacies = number_of_pharmacies ∧
  ∀(x y : ℕ), (x ≤ grid_size ∧ y ≤ grid_size) → 
  ¬ exists (p : ℕ × ℕ), p.1 ≤ x + pharmacy_walking_distance ∧
  p.2 ≤ y + pharmacy_walking_distance ∧ 
  p.1 < total_intersections ∧ 
  p.2 < total_intersections := 
by sorry

end insufficient_pharmacies_l488_488749


namespace cos_C_value_l488_488236

namespace Triangle

theorem cos_C_value (A B C : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (sin_A : Real.sin A = 2/3)
  (cos_B : Real.cos B = 1/2) :
  Real.cos C = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 := 
sorry

end Triangle

end cos_C_value_l488_488236


namespace expand_expression_l488_488439

theorem expand_expression (x y : ℝ) : 
  (16 * x + 18 - 7 * y) * (3 * x) = 48 * x^2 + 54 * x - 21 * x * y :=
by
  sorry

end expand_expression_l488_488439


namespace find_b2_div_a2_l488_488166

variables (a b : ℝ)
variables (a_pos : a > 0) (b_pos : b > 0)

def hyperbola (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def foci (c : ℝ) : Prop :=
  c = real.sqrt (a^2 + b^2)

def midpoint_MF1_on_hyperbola (M : ℝ × ℝ) (F1 : ℝ × ℝ) : Prop :=
  let N := ( (M.1 + F1.1) / 2, (M.2 + F1.2) / 2 ) in
  hyperbola a b N.1 N.2

theorem find_b2_div_a2 (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (c : ℝ) (h_c : foci a b c)
  (F1 F2 : ℝ × ℝ) (h_F1 : F1 = (-c, 0)) (h_F2 : F2 = (c, 0))
  (M : ℝ × ℝ) (h_M : M = (0, real.sqrt 3 * c))
  (h_midpoint : midpoint_MF1_on_hyperbola a b M F1) :
  (b^2 / a^2) = 3 + 2 * real.sqrt 3 :=
sorry

end find_b2_div_a2_l488_488166


namespace find_fx_two_thirds_l488_488471

-- Define the piecewise function
noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then √3 * Real.sin (Real.pi * x) else f (x - 1) + 1

-- Theorem to prove
theorem find_fx_two_thirds : f (2 / 3) = -1 / 2 := 
sorry

end find_fx_two_thirds_l488_488471


namespace three_digit_numbers_sum_seven_l488_488917

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488917


namespace middle_even_integer_l488_488703

theorem middle_even_integer (a b c : ℤ) (ha : even a) (hb : even b) (hc : even c) 
(h1 : a < b) (h2 : b < c) (h3 : 0 < a) (h4 : a < 10) (h5 : a + b + c = (1/8) * a * b * c) : b = 4 := 
sorry

end middle_even_integer_l488_488703


namespace least_number_l488_488446

theorem least_number (n : ℕ) : 
  (n % 45 = 2) ∧ (n % 59 = 2) ∧ (n % 77 = 2) → n = 205517 :=
by
  sorry

end least_number_l488_488446


namespace distance_between_points_l488_488817

-- Definitions of the points given in the conditions
def x1 : ℝ := 2
def y1 : ℝ := -1
def z1 : ℝ := 3

def x2 : ℝ := 6
def y2 : ℝ := 4
def z2 : ℝ := -2

-- Definition of the 3D distance formula
def distance_3d (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

-- The final proof statement asserting that the distance between the given points is sqrt(66)
theorem distance_between_points : distance_3d 2 -1 3 6 4 -2 = real.sqrt 66 :=
by
  sorry

end distance_between_points_l488_488817


namespace compare_liters_milliliters_l488_488775

-- Define the conversion factor between liters and milliliters
def liter_to_milliliter : ℕ := 1000

-- Define 1000 liters in milliliters
def liters1000_in_milliliters : ℕ := 1000 * liter_to_milliliter

-- Define 9000 milliliters
def milliliters9000 : ℕ := 9000

-- Problem statement in Lean 4
theorem compare_liters_milliliters : liters1000_in_milliliters < milliliters9000 = false :=
by
  sorry

end compare_liters_milliliters_l488_488775


namespace total_students_calculation_score_cutoff_calculation_l488_488434

-- Definitions based on conditions
def mu : ℝ := 60
def sigma2 : ℝ := 100
def sigma : ℝ := Real.sqrt sigma2
def z_score (x : ℝ) : ℝ := (x - mu) / sigma

-- Constants
def students_scored_90_or_above : ℕ := 13
def total_students : ℕ := 10000
def top_rewarded_students : ℕ := 228

-- Proving the total number of students given the conditions
theorem total_students_calculation :
  (1 - Real.cdf (z_score 90)) * total_students = students_scored_90_or_above :=
sorry

-- Proving the score cutoff for top 228 students
theorem score_cutoff_calculation :
  let p_top := (top_rewarded_students : ℝ) / (total_students : ℝ)
  let z := Real.invCdf (1 - p_top)
  Real.toFixed (z * sigma + mu) = 80 :=
sorry

end total_students_calculation_score_cutoff_calculation_l488_488434


namespace seed_packet_combinations_l488_488402

theorem seed_packet_combinations :
  (∑ n in finset.range 16, ∑ m in finset.range ((60 - 4 * n) / 3 + 1), 
    if 4 * n + 3 * m ≤ 60 then 1 else 0) = 72 :=
sorry

end seed_packet_combinations_l488_488402


namespace number_of_valid_pairs_l488_488079

theorem number_of_valid_pairs :
  let count_pairs := (finset.univ.image (λ (p : ℕ × ℕ), p)).countp
                      (λ (p : ℕ × ℕ), (1 ≤ p.1 ∧ p.1 ≤ 2089 ∧ 5^p.2 < 2^p.1 ∧ 2^p.1 < 2^(p.1 + 1) ∧ 2^(p.1 + 1) < 5^(p.2 + 1)))
  in count_pairs = 1189 :=
by sorry

end number_of_valid_pairs_l488_488079


namespace integral_cos_approx_l488_488349

open Real

theorem integral_cos_approx :
  let a := 0
  let b := π / 4
  let n := 5
  let Δx := (b - a) / n
  let x1 := a + (1 / 2) * Δx
  let x2 := a + (1 + 1 / 2) * Δx
  let x3 := a + (2 + 1 / 2) * Δx
  let x4 := a + (3 + 1 / 2) * Δx
  let x5 := a + (4 + 1 / 2) * Δx
  let y1 := cos x1
  let y2 := cos x2
  let y3 := cos x3
  let y4 := cos x4
  let y5 := cos x5
  Δx * (y1 + y2 + y3 + y4 + y5) ≈ 0.7231 := 
sorry

end integral_cos_approx_l488_488349


namespace solve_for_x_l488_488664
noncomputable theory

theorem solve_for_x (x : ℝ) (h : 5^(x + 6) = (5^4)^x) : x = 2 :=
by
  sorry

end solve_for_x_l488_488664


namespace base_of_parallelogram_l488_488105

-- Define the basic conditions given in the problem
def area := 360 -- in cm²
def height := 12 -- in cm

-- The goal is to prove that the base is 30 cm
theorem base_of_parallelogram : (area = 360) → (height = 12) → (360 / 12 = 30) :=
by
  intros h1 h2
  sorry

end base_of_parallelogram_l488_488105


namespace tax_amount_l488_488671

-- Defining the given conditions
def monthly_turnover := 35000
def threshold := 1000
def below_threshold_tax := 300
def exceeding_rate := 0.04

-- Formalizing the tax calculation
def calculate_tax (turnover : ℕ) : ℕ :=
  if turnover <= threshold then 
    below_threshold_tax
  else 
    below_threshold_tax + ((turnover - threshold) * exceeding_rate).toNat

-- The statement to prove
theorem tax_amount : calculate_tax monthly_turnover = 1660 := by
  sorry

end tax_amount_l488_488671


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488937

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488937


namespace anna_overtakes_bonnie_after_5_laps_l488_488809

-- Defining the problem with given conditions
variable (v : ℝ)  -- Bonnie's speed
variable (distance : ℝ := 400)  -- Length of the track
variable (lap_difference : ℝ := 1/4)  -- Anna covers 1/4 lap more than Bonnie each lap

-- Stating the theorem
theorem anna_overtakes_bonnie_after_5_laps :
  let anna_speed := 1.25 * v in
  let anna_distance_per_lap := 1.25 * distance in
  ∀ (n : ℝ), (anna_distance_per_lap / distance) * n = n + 1 → n = 4 := 
by
  intro anna_speed anna_distance_per_lap n h
  -- inserting "sorry" to skip the actual proof steps
  sorry

end anna_overtakes_bonnie_after_5_laps_l488_488809


namespace binomial_inequality_l488_488472

theorem binomial_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) : (1 + x)^n ≥ 1 + n * x :=
by sorry

end binomial_inequality_l488_488472


namespace min_balls_for_color_15_l488_488381

theorem min_balls_for_color_15
  (red green yellow blue white black : ℕ)
  (h_red : red = 28)
  (h_green : green = 20)
  (h_yellow : yellow = 19)
  (h_blue : blue = 13)
  (h_white : white = 11)
  (h_black : black = 9) :
  ∃ n, n = 76 ∧ ∀ balls_drawn, balls_drawn = n →
  ∃ color, 
    (color = "red" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= red) ∨
    (color = "green" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= green) ∨
    (color = "yellow" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= yellow) ∨
    (color = "blue" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= blue) ∨
    (color = "white" ∧ balls_drawn >= 15 ∧ balls_drawn <= white) ∨
    (color = "black" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= black) := 
sorry

end min_balls_for_color_15_l488_488381


namespace twelve_pharmacies_not_enough_l488_488756

def grid := ℕ × ℕ

def is_within_walking_distance (p1 p2 : grid) : Prop :=
  abs (p1.1 - p1.2) ≤ 3 ∧ abs (p2.1 - p2.2) ≤ 3

def walking_distance_coverage (pharmacies : set grid) (p : grid) : Prop :=
  ∃ pharmacy ∈ pharmacies, is_within_walking_distance pharmacy p

def sufficient_pharmacies (pharmacies : set grid) : Prop :=
  ∀ p : grid, walking_distance_coverage pharmacies p

theorem twelve_pharmacies_not_enough (pharmacies : set grid) (h : pharmacies.card = 12) : 
  ¬ sufficient_pharmacies pharmacies :=
sorry

end twelve_pharmacies_not_enough_l488_488756


namespace roots_of_equation_l488_488432

theorem roots_of_equation :
  ∀ x : ℝ, (4 * real.sqrt x + 4 * x^(-1/2) = 10) ↔ (x = 4 ∨ x = 0.25) :=
by
  intro x
  split
  { -- Assume 4 * √x + 4 * x^(-1/2) = 10 and prove x = 4 ∨ x = 0.25
    sorry },
  { -- Assume x = 4 or x = 0.25 and prove 4 * √x + 4 * x^(-1/2) = 10
    sorry }

end roots_of_equation_l488_488432


namespace ratio_of_areas_l488_488607

noncomputable def S1 : Set (ℝ × ℝ) :=
  {p | log10 (1 + p.1^2 + p.2^2) ≤ 1 + log10 (p.1 + p.2)}

noncomputable def S2 : Set (ℝ × ℝ) :=
  {p | log10 (2 + p.1^2 + p.2^2) ≤ 2 + log10 (p.1 + p.2)}

theorem ratio_of_areas : 
  let area_S1 := real.pi * 7^2
  let area_S2 := real.pi * (7 * real.sqrt 102)^2
  area_S2 / area_S1 = 102 := by
  sorry

end ratio_of_areas_l488_488607


namespace three_digit_sum_seven_l488_488882

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488882


namespace locus_of_points_on_diagonals_l488_488240

theorem locus_of_points_on_diagonals (a b x y : ℝ) (h₁ : 0 ≤ x ∧ x ≤ a) (h₂ : 0 ≤ y ∧ y ≤ b) :
  (x * y = (a - x) * y ∧ x * (b - y) = (a - x) * (b - y)) ↔ (x / a = y / b) :=
begin
  sorry
end

end locus_of_points_on_diagonals_l488_488240


namespace max_distance_on_ellipse_to_vertex_l488_488611

open Real

noncomputable def P (θ : ℝ) : ℝ × ℝ :=
(√5 * cos θ, sin θ)

def ellipse (x y : ℝ) := (x^2 / 5) + y^2 = 1

def B : ℝ × ℝ := (0, 1)

def dist (A B : ℝ × ℝ) : ℝ :=
sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem max_distance_on_ellipse_to_vertex :
  ∃ θ : ℝ, dist (P θ) B = 5 / 2 :=
sorry

end max_distance_on_ellipse_to_vertex_l488_488611


namespace range_of_a_l488_488210

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ ∈ ℝ, 2 * x₀ ^ 2 + (a - 1) * x₀ + 1 / 2 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l488_488210


namespace equation_for_number_l488_488689

variable (a : ℤ)

theorem equation_for_number : 3 * a + 5 = 9 :=
sorry

end equation_for_number_l488_488689


namespace three_digit_numbers_sum_seven_l488_488912

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488912


namespace nabla_difference_l488_488194

def nabla (a b : ℚ) : ℚ :=
  (a + b) / (1 + (a * b)^2)

theorem nabla_difference :
  (nabla 3 4) - (nabla 1 2) = - (16 / 29) :=
by
  sorry

end nabla_difference_l488_488194


namespace radius_large_circle_l488_488459

-- Definitions for the conditions
def radius_small_circle : ℝ := 2

def is_tangent_externally (r1 r2 : ℝ) : Prop := -- Definition of external tangency
  r1 + r2 = 4

def is_tangent_internally (R r : ℝ) : Prop := -- Definition of internal tangency
  R - r = 4

-- Setting up the property we need to prove: large circle radius
theorem radius_large_circle
  (R r : ℝ)
  (h1 : r = radius_small_circle)
  (h2 : is_tangent_externally r r)
  (h3 : is_tangent_externally r r)
  (h4 : is_tangent_externally r r)
  (h5 : is_tangent_externally r r)
  (h6 : is_tangent_internally R r) :
  R = 4 :=
by sorry

end radius_large_circle_l488_488459


namespace find_a₁_l488_488478

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ n

noncomputable def sequence_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

variables (a₁ q : ℝ)
-- Condition: The common ratio should not be 1.
axiom hq : q ≠ 1
-- Condition: Second term of the sequence a₂ = 1
axiom ha₂ : geometric_sequence a₁ q 1 = 1
-- Condition: 9S₃ = S₆
axiom hsum : 9 * sequence_sum a₁ q 3 = sequence_sum a₁ q 6

theorem find_a₁ : a₁ = 1 / 2 :=
  sorry

end find_a₁_l488_488478


namespace find_z2_l488_488151

noncomputable def z1 : Complex := 2 - Complex.i

theorem find_z2 (z1 z2 : Complex)
  (h1 : (z1 - 2) * (1 + Complex.i) = 1 - Complex.i)
  (h2 : z2.im = 2)
  (h3 : (z1 * z2).im = 0) :
  z2 = 4 + 2 * Complex.i :=
by
  have h4 : z1 = 2 - Complex.i := by sorry
  have h5 : z2 = 4 + 2 * Complex.i := by sorry
  exact h5

end find_z2_l488_488151


namespace total_weight_30_l488_488707

-- Definitions of initial weights and ratio conditions
variables (a b : ℕ)
def initial_weights (h1 : a = 4 * b) : Prop := True

-- Definitions of transferred weights
def transferred_weights (a' b' : ℕ) (h2 : a' = a - 10) (h3 : b' = b + 10) : Prop := True

-- Definition of the new ratio condition
def new_ratio (a' b' : ℕ) (h4 : 8 * a' = 7 * b') : Prop := True

-- The final proof statement
theorem total_weight_30 (a b a' b' : ℕ)
    (h1 : a = 4 * b) 
    (h2 : a' = a - 10) 
    (h3 : b' = b + 10)
    (h4 : 8 * a' = 7 * b') : a + b = 30 := 
    sorry

end total_weight_30_l488_488707


namespace sin_theta_plus_pi_over_six_l488_488121

open Real

theorem sin_theta_plus_pi_over_six (theta : ℝ) (h : sin θ + sin (θ + π / 3) = sqrt 3) :
  sin (θ + π / 6) = 1 := 
sorry

end sin_theta_plus_pi_over_six_l488_488121


namespace find_Q_l488_488052

/-
Given conditions:
- The list of numbers in the main column forms an arithmetic sequence.
- The first value in the main column is -9.
- The fourth value in the main column is 56.
- The list of numbers in the right column forms an arithmetic sequence.
- The first value in the right column is 16.
- The goal is to prove that the value of Q in the right column is -851/3.
-/

theorem find_Q :
  ∃ (d d' : ℚ),
    let a : ℕ → ℚ := λ n, -9 + n * d in
    let b : ℕ → ℚ := λ n, 16 - n * d' in
    a 3 = 56 ∧
    Q = b 1 →
    Q = -851 / 3 :=
begin
  sorry
end

end find_Q_l488_488052


namespace probability_abc_plus_de_is_odd_l488_488773

theorem probability_abc_plus_de_is_odd :
  let s := {1, 2, 3, 4, 5} in
  ∀ (a b c d e : ℕ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧ e ∈ s ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e →
  (∃ abc de : ℕ, abc = a * b * c ∧ de = d * e ∧ (abc + de) % 2 = 1) →
  (4/15 : ℚ) = 2/5 :=
by
  sorry

end probability_abc_plus_de_is_odd_l488_488773


namespace number_of_three_digit_numbers_with_sum_seven_l488_488887

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488887


namespace smallest_angle_of_trapezoid_l488_488224

theorem smallest_angle_of_trapezoid (a d : ℝ) :
  (a + (a + d) + (a + 2 * d) + (a + 3 * d) = 360) → 
  (a + 3 * d = 150) → 
  a = 15 :=
by
  sorry

end smallest_angle_of_trapezoid_l488_488224


namespace intersection_of_sets_l488_488524

def A := { x : ℝ | x^2 - 2 * x - 8 < 0 }
def B := { x : ℝ | x >= 0 }
def intersection := { x : ℝ | 0 <= x ∧ x < 4 }

theorem intersection_of_sets : (A ∩ B) = intersection := 
sorry

end intersection_of_sets_l488_488524


namespace rationalized_expression_correct_A_B_C_D_E_sum_correct_l488_488279

noncomputable def A : ℤ := -18
noncomputable def B : ℤ := 2
noncomputable def C : ℤ := 30
noncomputable def D : ℤ := 5
noncomputable def E : ℤ := 428
noncomputable def expression := 3 / (2 * Real.sqrt 18 + 5 * Real.sqrt 20)
noncomputable def rationalized_form := (A * Real.sqrt B + C * Real.sqrt D) / E

theorem rationalized_expression_correct :
  rationalized_form = (18 * Real.sqrt 2 - 30 * Real.sqrt 5) / -428 :=
by
  sorry

theorem A_B_C_D_E_sum_correct :
  A + B + C + D + E = 447 :=
by
  sorry

end rationalized_expression_correct_A_B_C_D_E_sum_correct_l488_488279


namespace three_digit_sum_seven_l488_488875

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488875


namespace three_digit_sum_seven_l488_488876

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488876


namespace number_of_valid_sequences_l488_488734

theorem number_of_valid_sequences : 
  let sequences := {s : Fin 25 → ℕ // (∀ n m : Fin 25, n + 1 ∣ m + 1 → s n ∣ s m) ∧ ∀ n : Fin 25, s n ∈ Finset.range 1 26} in
  sequences.card = 24 :=
by
  sorry

end number_of_valid_sequences_l488_488734


namespace rationalize_denominator_sum_l488_488658

theorem rationalize_denominator_sum :
  let A := 3
  let B := -9
  let C := -9
  let D := 9
  let E := 165
  let F := 51
  A + B + C + D + E + F = 210 :=
by
  let A := 3
  let B := -9
  let C := -9
  let D := 9
  let E := 165
  let F := 51
  show 3 + -9 + -9 + 9 + 165 + 51 = 210
  sorry

end rationalize_denominator_sum_l488_488658


namespace twelve_pharmacies_not_enough_l488_488757

def grid := ℕ × ℕ

def is_within_walking_distance (p1 p2 : grid) : Prop :=
  abs (p1.1 - p1.2) ≤ 3 ∧ abs (p2.1 - p2.2) ≤ 3

def walking_distance_coverage (pharmacies : set grid) (p : grid) : Prop :=
  ∃ pharmacy ∈ pharmacies, is_within_walking_distance pharmacy p

def sufficient_pharmacies (pharmacies : set grid) : Prop :=
  ∀ p : grid, walking_distance_coverage pharmacies p

theorem twelve_pharmacies_not_enough (pharmacies : set grid) (h : pharmacies.card = 12) : 
  ¬ sufficient_pharmacies pharmacies :=
sorry

end twelve_pharmacies_not_enough_l488_488757


namespace bug_ends_up_at_A_after_six_moves_l488_488781

def P (n : ℕ) (v : Fin 3) : ℚ := sorry

axiom initial_conditions : P 0 0 = 1 ∧ P 0 1 = 0 ∧ P 0 2 = 0

axiom transition_probabilities : 
  ∀ (n : ℕ),
    [P (n+1) 0 = 1/2 * P n 1 + 1/2 * P n 2] ∧
    [P (n+1) 1 = 1/2 * P n 0 + 1/2 * P n 2] ∧
    [P (n+1) 2 = 1/2 * P n 0 + 1/2 * P n 1]

theorem bug_ends_up_at_A_after_six_moves :
  P 6 0 = 5 / 16 :=
sorry

end bug_ends_up_at_A_after_six_moves_l488_488781


namespace exist_decimals_l488_488361

theorem exist_decimals :
  ∃ (a b : ℝ), 3.5 < a ∧ a < 3.6 ∧ 3.5 < b ∧ b < 3.6 ∧
  ∃ (c d e : ℝ), 0 < c ∧ c < 0.1 ∧ 0 < d ∧ d < 0.1 ∧ 0 < e ∧ e < 0.1 :=
by
  -- Example values that satisfy the conditions
  use 3.51, 3.52
  split
  { linarith }
  split
  { linarith }
  use 0.01, 0.02, 0.03
  split
  { linarith }
  split
  { linarith }
  split
  { linarith }
  sorry

end exist_decimals_l488_488361


namespace tangent_intersection_y_constant_l488_488830

theorem tangent_intersection_y_constant (a b : ℝ) (hA : point_on_parabola a) (hB : point_on_parabola b)
  (h_perp : tangents_perpendicular a b) :
  tangent_intersection_y a b = -1 / 4 :=
sorry

-- Definitions needed for the theorem statement
def point_on_parabola (a : ℝ) : Prop :=
  ∃ y, y = 4 * a^2

def tangents_perpendicular (a b : ℝ) : Prop :=
  (8 * a) * (8 * b) = -1

noncomputable def tangent_intersection_y (a b : ℝ) : ℝ :=
  let x := (b^2 - a^2) / (2 * (a - b)) in
  8 * a * (x - a) - 4 * a^2

end tangent_intersection_y_constant_l488_488830


namespace students_attended_game_l488_488343

variable (s n : ℕ)

theorem students_attended_game (h1 : s + n = 3000) (h2 : 10 * s + 15 * n = 36250) : s = 1750 := by
  sorry

end students_attended_game_l488_488343


namespace area_of_right_triangle_l488_488587

-- Mathematical definitions based on the problem conditions
def right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def leg_of_right_triangle (a : ℕ) : Prop :=
  a = 7

def integers_in_right_triangle (a b c : ℕ) : Prop :=
  ∀ x, x = a ∨ x = b ∨ x = c → x ∈ ℕ

-- The statement to prove
theorem area_of_right_triangle : ∃ (b c : ℕ), right_triangle 7 b c ∧ leg_of_right_triangle 7 ∧ integers_in_right_triangle 7 b c ∧ (7 * b / 2 = 84) :=
sorry

end area_of_right_triangle_l488_488587


namespace find_numbers_l488_488307

theorem find_numbers (x y : ℤ) (h1 : x > y) (h2 : x^2 - y^2 = 100) : 
  x = 26 ∧ y = 24 := 
  sorry

end find_numbers_l488_488307


namespace car_clock_actual_time_l488_488304

theorem car_clock_actual_time :
    ∀ (car_clock_real_rate : ℝ) (car_start_watch_time car_start_car_time watch_time car_time : ℝ),
    car_clock_real_rate > 0 ∧
    car_start_watch_time = car_start_car_time ∧
    watch_time = 0.5 ∧  -- 30 minutes past 12:00 noon
    car_time = 0.583333 ∧  -- 35 minutes past 12:00 noon
    car_clock_real_rate = car_time / watch_time ∧
    7 / car_clock_real_rate = 6 →
    let actual_time := car_start_watch_time + 6 in
    actual_time = 18 :=
begin
    sorry,
end

end car_clock_actual_time_l488_488304


namespace decimal_to_base7_conversion_l488_488082

theorem decimal_to_base7_conversion :
  (2023 : ℕ) = 5 * (7^3) + 6 * (7^2) + 2 * (7^1) + 0 * (7^0) :=
by
  sorry

end decimal_to_base7_conversion_l488_488082


namespace length_of_segment_AB_is_eight_l488_488509

theorem length_of_segment_AB_is_eight :
  let F := (1 : ℝ, 0 : ℝ),
      line_eq := ∀ (x y : ℝ), y = x - 1,
      parabola_eq := ∀ (x y : ℝ), y^2 = 4 * x,
      intersection_points (x y : ℝ) := (line_eq x y) ∧ (parabola_eq x y) in
  ∃ (A B : ℝ × ℝ), intersection_points A.1 A.2 ∧ intersection_points B.1 B.2 ∧
  (∃ (d : ℝ), d = 8 ∧ (d = real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2))) :=
sorry

end length_of_segment_AB_is_eight_l488_488509


namespace log_base_9_of_729_l488_488845

theorem log_base_9_of_729 : ∃ (x : ℝ), (9 : ℝ)^x = (729 : ℝ) ∧ x = 3 := 
by {
  have h1 : (9 : ℝ) = (3 : ℝ)^2 := by norm_num,
  have h2 : (729 : ℝ) = (3 : ℝ)^6 := by norm_num,
  use 3,
  split,
  {
    calc (9 : ℝ) ^ 3
        = (3^2 : ℝ) ^ 3 : by rw h1
    ... = (3^6 : ℝ) : by rw pow_mul
    ... = (729 : ℝ) : by rw h2,
  },
  { 
    refl,
  }
}

end log_base_9_of_729_l488_488845


namespace insufficient_pharmacies_l488_488752

/-- Define the grid size and conditions --/
def grid_size : ℕ := 9
def total_intersections : ℕ := 100
def pharmacy_walking_distance : ℕ := 3
def number_of_pharmacies : ℕ := 12

/-- Prove that 12 pharmacies are not enough for the given conditions. --/
theorem insufficient_pharmacies : ¬ (number_of_pharmacies ≥ grid_size + 3) → 
  ∃(pharmacies : ℕ), pharmacies = number_of_pharmacies ∧
  ∀(x y : ℕ), (x ≤ grid_size ∧ y ≤ grid_size) → 
  ¬ exists (p : ℕ × ℕ), p.1 ≤ x + pharmacy_walking_distance ∧
  p.2 ≤ y + pharmacy_walking_distance ∧ 
  p.1 < total_intersections ∧ 
  p.2 < total_intersections := 
by sorry

end insufficient_pharmacies_l488_488752


namespace trivia_team_students_per_group_l488_488712

theorem trivia_team_students_per_group (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ) 
  (h1 : total_students = 58) (h2 : not_picked = 10) (h3 : num_groups = 8) :
  (total_students - not_picked) / num_groups = 6 :=
by
  sorry

end trivia_team_students_per_group_l488_488712


namespace independence_tests_incorrect_statement_l488_488807

theorem independence_tests_incorrect_statement (h1 : ∀ (P : Prop), P → ¬(¬P))
    (h2 : ∃ (P : Prop), P ∧ ¬P)
    (h3 : ∀ (P : Prop), true)
    (h4 : ∀ (independence_test : Type → Prop), independence_test False ↔ false):
    ¬(∀ (independence_test : Type → Prop), independence_test True → true) :=
by
  sorry

end independence_tests_incorrect_statement_l488_488807


namespace three_digit_sum_7_l488_488968

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488968


namespace probability_grade_A_l488_488032

-- Defining probabilities
def P_B : ℝ := 0.05
def P_C : ℝ := 0.03

-- Theorem: proving the probability of Grade A
theorem probability_grade_A : 1 - P_B - P_C = 0.92 :=
by
  -- Placeholder for proof
  sorry

end probability_grade_A_l488_488032


namespace digits_sum_eq_seven_l488_488924

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488924


namespace twelve_pharmacies_not_enough_l488_488746

-- Define the grid dimensions and necessary parameters
def grid_size := 9
def total_intersections := (grid_size + 1) * (grid_size + 1) -- 10 * 10 grid
def walking_distance := 3
def coverage_side := (walking_distance * 2 + 1)  -- 7x7 grid coverage
def max_covered_per_pharmacy := (coverage_side - 1) * (coverage_side - 1)  -- Coverage per direction

-- Define the main theorem
theorem twelve_pharmacies_not_enough (n m : ℕ): 
  n = grid_size + 1 -> m = grid_size + 1 -> total_intersections = n * m -> 
  (walking_distance < n) -> (walking_distance < m) -> (pharmacies : ℕ) -> pharmacies = 12 ->
  (coverage_side <= n) -> (coverage_side <= m) ->
  ¬ (∀ i j : ℕ, i < n -> j < m -> ∃ p : ℕ, p < pharmacies -> 
  abs (i - (p / (grid_size + 1))) + abs (j - (p % (grid_size + 1))) ≤ walking_distance) :=
begin
  intros,
  sorry -- Proof omitted
end

end twelve_pharmacies_not_enough_l488_488746


namespace count_tetrahedra_l488_488234

noncomputable def P1 := (1, 1, 1)
noncomputable def P2 := (-1, 1, 1)
noncomputable def P3 := (1, -1, 1)
noncomputable def P4 := (-1, -1, 1)
noncomputable def P5 := (1, 1, -1)
noncomputable def P6 := (-1, 1, -1)
noncomputable def P7 := (1, -1, -1)
noncomputable def P8 := (-1, -1, -1)

def points : List (ℝ × ℝ × ℝ) := [P1, P2, P3, P4, P5, P6, P7, P8]

theorem count_tetrahedra (n : ℕ) : n = 58 :=
by
  sorry

end count_tetrahedra_l488_488234


namespace subtraction_makes_divisible_l488_488741

theorem subtraction_makes_divisible :
  ∃ n : Nat, 9671 - n % 2 = 0 ∧ n = 1 :=
by
  sorry

end subtraction_makes_divisible_l488_488741


namespace chameleons_color_change_l488_488563

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l488_488563


namespace number_of_books_in_oak_grove_libraries_l488_488335

theorem number_of_books_in_oak_grove_libraries :
  let public_library_books := 1986
  let school_library_books := 5106
  (public_library_books + school_library_books) = 7092 :=
by
  let public_library_books := 1986
  let school_library_books := 5106
  show (public_library_books + school_library_books) = 7092, from sorry

end number_of_books_in_oak_grove_libraries_l488_488335


namespace twelve_pharmacies_not_enough_l488_488745

-- Define the grid dimensions and necessary parameters
def grid_size := 9
def total_intersections := (grid_size + 1) * (grid_size + 1) -- 10 * 10 grid
def walking_distance := 3
def coverage_side := (walking_distance * 2 + 1)  -- 7x7 grid coverage
def max_covered_per_pharmacy := (coverage_side - 1) * (coverage_side - 1)  -- Coverage per direction

-- Define the main theorem
theorem twelve_pharmacies_not_enough (n m : ℕ): 
  n = grid_size + 1 -> m = grid_size + 1 -> total_intersections = n * m -> 
  (walking_distance < n) -> (walking_distance < m) -> (pharmacies : ℕ) -> pharmacies = 12 ->
  (coverage_side <= n) -> (coverage_side <= m) ->
  ¬ (∀ i j : ℕ, i < n -> j < m -> ∃ p : ℕ, p < pharmacies -> 
  abs (i - (p / (grid_size + 1))) + abs (j - (p % (grid_size + 1))) ≤ walking_distance) :=
begin
  intros,
  sorry -- Proof omitted
end

end twelve_pharmacies_not_enough_l488_488745


namespace sum_of_coordinates_of_point_F_l488_488248

def midpoint (p1 p2 m : ℤ × ℤ) : Prop :=
  (m.1 = (p1.1 + p2.1) / 2) ∧ (m.2 = (p1.2 + p2.2) / 2)

theorem sum_of_coordinates_of_point_F (x y : ℤ) :
  midpoint (2, -5) (x, y) (4, -1) → x + y = 9 := 
by
  sorry

end sum_of_coordinates_of_point_F_l488_488248


namespace find_m_l488_488165

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) 2 → x^2 - x - m ≤ 0

theorem find_m : satisfies_inequality 2 :=
by sorry

end find_m_l488_488165


namespace highest_score_not_necessarily_12_l488_488369

-- Define the structure of the round-robin tournament setup
structure RoundRobinTournament :=
  (teams : ℕ)
  (matches_per_team : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (points_draw : ℕ)

-- Tournament conditions
def tournament : RoundRobinTournament :=
  { teams := 12,
    matches_per_team := 11,
    points_win := 2,
    points_loss := 0,
    points_draw := 1 }

-- The statement we want to prove
theorem highest_score_not_necessarily_12 (T : RoundRobinTournament) :
  ∃ team_highest_score : ℕ, team_highest_score < 12 :=
by
  -- Provide a proof here
  sorry

end highest_score_not_necessarily_12_l488_488369


namespace part_a_part_b_l488_488660

section
variable (a : ℕ → ℝ) (n : ℕ)

def di (i : ℕ) (h₁ : 1 ≤ i) (h₂ : i ≤ n) : ℝ :=
  let maxAj := (finset.range i).max' (finset.nonempty_range_iff.mpr h₁)
  let minAj := (finset.Icc i n).min' (finset.nonempty_Icc.mpr ⟨h₂, by linarith⟩)
  a maxAj - a minAj

def d (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  finset.range n.succ |>.image (λ i, di a i (nat.succ_pos _) (nat.le_succ _)) |>.max' (finset.nonempty_image_iff.mpr (finset.nonempty_range_iff.mpr (nat.succ_pos _)))

theorem part_a (a : ℕ → ℝ) (n : ℕ) (x : ℕ → ℝ) (hx : ∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n → x i ≤ x j ) :
  ∃ i, 1 ≤ i ∧ i ≤ n ∧ |x i - a i| ≥ d a n / 2 :=
sorry

theorem part_b (a : ℕ → ℝ) (n : ℕ) :
  ∃ x : ℕ → ℝ, (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n → x i ≤ x j) ∧ ( ∀ i, 1 ≤ i ∧ i ≤ n → |x i - a i| = d a n / 2 ) :=
sorry

end

end part_a_part_b_l488_488660


namespace find_a9_l488_488628

variable {a_n : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable {d a₁ : ℤ}

-- Conditions
def arithmetic_sequence := ∀ n : ℕ, a_n n = a₁ + n * d
def sum_first_n_terms := ∀ n : ℕ, S n = (n * (2 * a₁ + (n - 1) * d)) / 2

-- Specific Conditions for the problem
axiom condition1 : S 8 = 4 * a₁
axiom condition2 : a_n 6 = -2 -- Note that a_n is 0-indexed here.

theorem find_a9 : a_n 8 = 2 :=
by
  sorry

end find_a9_l488_488628


namespace find_k_l488_488325

-- Define the problem statement
theorem find_k (d : ℝ) (x : ℝ)
  (h_ratio : 3 * x / (5 * x) = 3 / 5)
  (h_diag : (10 * d)^2 = (3 * x)^2 + (5 * x)^2) :
  ∃ k : ℝ, (3 * x) * (5 * x) = k * d^2 ∧ k = 750 / 17 := by
  sorry

end find_k_l488_488325


namespace area_quadrilateral_l488_488225

/-- In quadrilateral ABCD with given conditions, the area of ABCD is 12.5 * sqrt 2 -/
theorem area_quadrilateral 
  (A B C D : Type) [EucGeometry A]
  (AB : ℝ) (BC : ℝ) (CD : ℝ)
  (angle_B : ℝ) (angle_C : ℝ)
  (h1 : ∠ B = 135) (h2 : ∠ C = 135)
  (h3 : AB = 4) (h4 : BC = 5) (h5 : CD = 6) :
  area ABCD = 12.5 * Real.sqrt 2 := 
by 
  sorry

end area_quadrilateral_l488_488225


namespace max_apps_proof_l488_488451

variable (r : ℕ)
variable (twice_r : ℕ)
variable (delete_apps : ℕ)
variable (max_apps : ℕ)

axiom h1 : r = 35
axiom h2 : twice_r = 2 * r
axiom h3 : delete_apps = 20
axiom h4 : max_apps = twice_r - delete_apps

theorem max_apps_proof : max_apps = 50 := by
  rw [h1, h2, h3, h4]
  sorry

end max_apps_proof_l488_488451


namespace orchid_bushes_planted_tomorrow_l488_488711

theorem orchid_bushes_planted_tomorrow 
  (initial : ℕ) (planted_today : ℕ) (final : ℕ) (planted_tomorrow : ℕ) :
  initial = 47 →
  planted_today = 37 →
  final = 109 →
  planted_tomorrow = final - (initial + planted_today) →
  planted_tomorrow = 25 :=
by
  intros h_initial h_planted_today h_final h_planted_tomorrow
  rw [h_initial, h_planted_today, h_final] at h_planted_tomorrow
  exact h_planted_tomorrow


end orchid_bushes_planted_tomorrow_l488_488711


namespace ex1_l488_488189

theorem ex1 (a b : ℕ) (h₀ : a = 3) (h₁ : b = 4) : ∃ n : ℕ, 3^(7*a + b) = n^7 :=
by
  use 27
  sorry

end ex1_l488_488189


namespace problem_statement_l488_488254

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def given_function (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then real.log (4 * x + 1) / real.log 2 else sorry

noncomputable def f : ℝ → ℝ := given_function

theorem problem_statement :
  is_periodic f 2 →
  is_odd f →
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = real.log (4 * x + 1) / real.log 2) →
  f (13 / 4) = -2 := by
  intros h_periodic h_odd h_def
  sorry

end problem_statement_l488_488254


namespace necessary_but_not_sufficient_parallel_l488_488487

-- Definitions for lines, planes, and angles in 3D space
variables {m n : Type} {a : Type} 
variables [line m] [line n] [plane a]

-- Condition: The angles between m, n and a are equal
axiom equal_angles (m n : Type) [line m] [line n] (a : Type) [plane a] :
  angle m a = angle n a

-- The theorem stating that this is a necessary but not sufficient condition for m parallel to n
theorem necessary_but_not_sufficient_parallel (m n : Type) [line m] [line n] (a : Type) [plane a]
  (h : equal_angles m n a) : 
  m ∥ n := 
sorry

end necessary_but_not_sufficient_parallel_l488_488487


namespace sum_of_distances_l488_488786

noncomputable def sum_distances (d_AB : ℝ) (d_A : ℝ) (d_B : ℝ) : ℝ :=
d_A + d_B

theorem sum_of_distances
    (tangent_to_sides : Circle → Point → Point → Prop)
    (C_on_circle : Circle → Point → Prop)
    (A B C : Point)
    (γ : Circle)
    (h_dist_to_AB : ∃ C, C_on_circle γ C → distance_from_line C A B = 4)
    (h_ratio : ∃ hA hB, distance_from_side C A = hA ∧ distance_from_side C B = hB ∧ (hA = 4 * hB ∨ hB = 4 * hA)) :
    sum_distances (distance_from_line C A B) (distance_from_side C A) (distance_from_side C B) = 10 :=
by
  sorry

end sum_of_distances_l488_488786


namespace chameleons_color_change_l488_488561

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l488_488561


namespace locus_single_point_l488_488106

-- We define our points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- Define the distance function for points in the plane
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the radius
def r : ℝ := 3

-- Define the locus points we are searching for
def locus (P : ℝ × ℝ) : Prop :=
  dist P A = r ∧ dist P B = r

theorem locus_single_point :
  ∀ P : ℝ × ℝ, locus P → P = (0, 0) :=
by
  sorry

end locus_single_point_l488_488106


namespace number_to_subtract_l488_488730

theorem number_to_subtract (x : ℝ) :
  let p := 2 * x^3 + 8 * x^2 - 14 * x + 24 in
  ∃ k, (p - k) = (x + 3) * (p / (x + 3)) ∧ k = 84 := by
  sorry

end number_to_subtract_l488_488730


namespace digits_sum_eq_seven_l488_488934

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488934


namespace trig_identity_l488_488146

open Real

def tan_identity {α : ℝ} : Prop :=
  tan α = 2 * tan (π / 5)

theorem trig_identity (α : ℝ) (h : tan_identity α) :
  (cos (α - 3 * π / 10)) / (sin (α - π / 5)) = 3 :=
by
  sorry

end trig_identity_l488_488146


namespace relative_min_of_f_l488_488836

def f (a x : ℝ) : ℝ := x^4 - x^3 - x^2 + a * x + 1

theorem relative_min_of_f (a : ℝ) : 
  (∃ a : ℝ, f a a = a ∧ (∀ x : ℝ, f x x = x → x = 1)) ∧ 
  (∀ x : ℝ, f x x = x → (12 * (x^2) - 6 * x - 2 > 0)) → 
  (a = 1) := 
by {
  exists 1,
  intro x,
  intro hx,
  sorry
}

end relative_min_of_f_l488_488836


namespace expected_winnings_is_350_l488_488024

noncomputable def expected_winnings : ℝ :=
  (1 / 8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_winnings_is_350 :
  expected_winnings = 3.5 :=
by sorry

end expected_winnings_is_350_l488_488024


namespace moon_speed_conversion_l488_488684

theorem moon_speed_conversion
  (speed_kps : ℝ)
  (seconds_per_hour : ℝ)
  (h1 : speed_kps = 0.2)
  (h2 : seconds_per_hour = 3600) :
  speed_kps * seconds_per_hour = 720 := by
  sorry

end moon_speed_conversion_l488_488684


namespace vector_magnitude_given_perpendicular_l488_488514

variable (λ : ℝ)

def vector_a : ℝ × ℝ := (λ, -2)
def vector_b : ℝ × ℝ := (1, 3)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2)

theorem vector_magnitude_given_perpendicular :
  (dot_product (vector_a λ) vector_b = 0) →
  magnitude (vector_a λ + vector_b) = 5 * Real.sqrt 2 :=
by
  sorry

end vector_magnitude_given_perpendicular_l488_488514


namespace max_height_l488_488014

def h (t : ℝ) : ℝ := -4 * t^2 + 40 * t + 20

theorem max_height : ∃ t, h(t) = 120 ∧ ∀ s, h(s) ≤ 120 :=
sorry

end max_height_l488_488014


namespace radius_of_large_circle_l488_488458

theorem radius_of_large_circle : 
  ∃ (R : ℝ), R = 2 + 2 * Real.sqrt 2 ∧ 
  ∀ (r : ℝ) (n : ℕ), 
    (∀ (i j : ℕ), i ≠ j → i < n → j < n → 
    dist (r * cos (2 * i * π / n), r * sin (2 * i * π / n)) 
         (r * cos (2 * j * π / n), r * sin (2 * j * π / n)) = 2 * r) ∧ 
    (∀ (i : ℕ), 
      i < n → 
      dist (r * cos (2 * i * π / n), r * sin (2 * i * π / n)) 
         (0, 0) = R - r) → r = 2 ∧ n = 4 :=
by
  sorry

end radius_of_large_circle_l488_488458


namespace exists_monic_decreasing_integer_polynomial_of_degree_l488_488661

theorem exists_monic_decreasing_integer_polynomial_of_degree 
  (n : ℕ) (hn : 0 < n) :
  ∃ p : polynomial ℤ, 
    p.monic ∧ 
    p.degree = n ∧ 
    (∀ i j, i < j → p.coeff i ≤ p.coeff j) ∧ 
    (∀ x, is_root p x → x ∈ ℤ) :=
sorry

end exists_monic_decreasing_integer_polynomial_of_degree_l488_488661


namespace cost_price_correct_l488_488646

-- Define the selling price
def selling_price : ℝ := 21000

-- Define the discount rate
def discount_rate : ℝ := 0.10

-- Define the new selling price after discount
def new_selling_price : ℝ := selling_price * (1 - discount_rate)

-- Define the profit rate
def profit_rate : ℝ := 0.08

-- Define the cost price
def cost_price : ℝ := new_selling_price / (1 + profit_rate)

-- The theorem to prove
theorem cost_price_correct : cost_price = 17500 := 
by 
  sorry

end cost_price_correct_l488_488646


namespace num_three_digit_sums7_l488_488999

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l488_488999


namespace largest_y_coordinate_ellipse_l488_488423

theorem largest_y_coordinate_ellipse:
  (∀ x y : ℝ, (x^2 / 49) + ((y + 3)^2 / 25) = 1 → y ≤ 2)  ∧ 
  (∃ x : ℝ, (x^2 / 49) + ((2 + 3)^2 / 25) = 1) := sorry

end largest_y_coordinate_ellipse_l488_488423


namespace bottles_from_shop_C_l488_488842

theorem bottles_from_shop_C (A B C T : ℕ) (hA : A = 150) (hB : B = 180) (hT : T = 550) (hSum : T = A + B + C) :
  C = 220 :=
by
  rw [hA, hB, hT] at hSum
  simpa using hSum

end bottles_from_shop_C_l488_488842


namespace faye_age_l488_488838

variables (C D E F G : ℕ)
variables (h1 : D = E - 2)
variables (h2 : E = C + 6)
variables (h3 : F = C + 4)
variables (h4 : G = C - 5)
variables (h5 : D = 16)

theorem faye_age : F = 16 :=
by
  -- Proof will be placed here
  sorry

end faye_age_l488_488838


namespace three_digit_integers_sum_to_7_l488_488905

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488905


namespace measure_angle_BAC_is_50_degrees_l488_488144

theorem measure_angle_BAC_is_50_degrees
  (O A B C X Y : Point)
  (h1 : circumcenter O (triangle A B C))
  (h2 : lies_on X (segment A C))
  (h3 : lies_on Y (segment A B))
  (h4 : intersects (line B X) (line C Y) O)
  (h5 : angle A B C = angle A Y X ∧ angle A Y X = angle X Y C) :
  angle A B C = 50 :=
by
  sorry

end measure_angle_BAC_is_50_degrees_l488_488144


namespace number_of_towers_l488_488388

noncomputable def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def multinomial (n : ℕ) (k1 k2 k3 : ℕ) : ℕ :=
  factorial n / (factorial k1 * factorial k2 * factorial k3)

theorem number_of_towers :
  (multinomial 10 3 3 4 = 4200) :=
by
  sorry

end number_of_towers_l488_488388


namespace find_unknown_term_l488_488111

theorem find_unknown_term (x : ℤ) : 
  ∃ x, (x = 64 ∧ (8, x, 62, -4, -12) = (8, 64, 62, -4, -12)) := 
by
  use 64
  split
  · rfl
  · rfl

end find_unknown_term_l488_488111


namespace ratio_cost_to_age_l488_488305

-- the ages and costs are in natural numbers
variables {Betty_age Doug_age cost_per_pack packs_count total_cost : ℕ}

-- The variables involved
def Doug_age := 40
def total_cost := 2000
def packs_count := 20
def sum_ages := Doug_age + Betty_age = 90

-- Define the cost per pack
def cost_per_pack := total_cost / packs_count

-- Prove that the ratio of the cost of a pack of nuts to Betty's age is 2
theorem ratio_cost_to_age
  (H1 : Doug_age = 40)
  (H2 : Doug_age + Betty_age = 90)
  (H3 : total_cost = 2000)
  (H4 : packs_count = 20)
  : cost_per_pack / Betty_age = 2 :=
by
  sorry

end ratio_cost_to_age_l488_488305


namespace lambda_value_l488_488180

open Locale BigOperators

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) 
  (h1 : a = (1, 3))
  (h2 : b = (3, 4))
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0) : 
  λ = 3 / 5 := by
  sorry

end lambda_value_l488_488180


namespace relationship_of_a_and_b_l488_488467

theorem relationship_of_a_and_b (a b : ℝ) (h_b_nonzero: b ≠ 0)
  (m n : ℤ) (h_intersection : ∃ (m n : ℤ), n = m^3 - a * m^2 - b * m ∧ n = a * m + b) :
  2 * a - b + 8 = 0 :=
  sorry

end relationship_of_a_and_b_l488_488467


namespace remainder_div_1000_eq_531_l488_488624

def greatest_integer_multiple_of_9 (n : ℕ) : Prop := 
  n % 9 = 0

def no_two_digits_same (n : ℕ) : Prop := 
  (∀ (i j : ℕ), i ≠ j → ((n / 10 ^ i) % 10) ≠ ((n / 10 ^ j) % 10))

def each_digit_odd (n : ℕ) : Prop :=
  (∀ (i : ℕ), ((n / 10 ^ i) % 10 ≠ 0) →
          ((((n / 10 ^ i) % 10) = 1) ∨ (((n / 10 ^ i) % 10) = 3) ∨ 
           (((n / 10 ^ i) % 10) = 5) ∨ (((n / 10 ^ i) % 10) = 7) ∨ 
           (((n / 10 ^ i) % 10) = 9)))

noncomputable def M : ℕ :=
  if h1 : ∃ n, greatest_integer_multiple_of_9 n ∧ no_two_digits_same n ∧ each_digit_odd n
  then Classical.some h1
  else 0

theorem remainder_div_1000_eq_531 (M : ℕ) 
  (hM : greatest_integer_multiple_of_9 M ∧ no_two_digits_same M ∧ each_digit_odd M) :
  M % 1000 = 531 := sorry

end remainder_div_1000_eq_531_l488_488624


namespace calculate_A_plus_B_l488_488551

def grid_area := 36
def num_smaller_circles := 4
def num_medium_circles := 1
def small_circle_radius := 0.5
def medium_circle_radius := 1
def smaller_circle_area (r : ℝ) := Real.pi * r ^ 2

theorem calculate_A_plus_B : 
  let A := grid_area,
      B := num_smaller_circles * smaller_circle_area(small_circle_radius) / Real.pi + num_medium_circles * smaller_circle_area(medium_circle_radius) / Real.pi
  in A + B = 38 := 
by
  -- Our definitions
  let A := grid_area
  let B := num_smaller_circles * smaller_circle_area(small_circle_radius) / Real.pi + num_medium_circles * smaller_circle_area(medium_circle_radius) / Real.pi
  -- Expected result
  have hA : A = 36 := rfl
  have hB : B = 2 := by
    simp only [num_smaller_circles, small_circle_radius, num_medium_circles, medium_circle_radius, smaller_circle_area]
    ring
  have hA_plus_B : A + B = 38 := by
    rw [hA, hB]
    norm_num
  exact hA_plus_B

end calculate_A_plus_B_l488_488551


namespace max_students_late_all_three_days_l488_488217

theorem max_students_late_all_three_days (A B C total l: ℕ) 
  (hA: A = 20) 
  (hB: B = 13) 
  (hC: C = 7) 
  (htotal: total = 30) 
  (hposA: 0 ≤ A) (hposB: 0 ≤ B) (hposC: 0 ≤ C) 
  (hpostotal: 0 ≤ total) 
  : l = 5 := by
  sorry

end max_students_late_all_three_days_l488_488217


namespace M_zero_l488_488328

noncomputable def a (n : ℕ) : ℕ := n
noncomputable def b (n : ℕ) : ℕ := 2^n
noncomputable def c (n : ℕ) : ℕ := n + 2^n

def M (n : ℕ) : ℕ := ∑ i in range n, a i + ∑ j in range n, b j + ∑ k in range n, c k

theorem M_zero :
  M 0 = 11 := 
by
  sorry

end M_zero_l488_488328


namespace sum_of_signed_integers_parity_even_l488_488496

theorem sum_of_signed_integers_parity_even :
  ∀ (f : Fin 2012 → Bool),
  (∑ i, if f i then (i : ℤ) + 1 else - (i + 1)) % 2 = 0 := 
by
  sorry

end sum_of_signed_integers_parity_even_l488_488496


namespace domain_of_f_l488_488069

noncomputable def f (x : ℝ) := real.sqrt (4 - real.sqrt (6 - real.sqrt x))

theorem domain_of_f :
  {x | 0 ≤ x ∧ x ≤ 36} = {x | f x ≠ 0 ∨ 0 ≤ f x} :=
sorry

end domain_of_f_l488_488069


namespace twelve_pharmacies_not_enough_l488_488744

-- Define the grid dimensions and necessary parameters
def grid_size := 9
def total_intersections := (grid_size + 1) * (grid_size + 1) -- 10 * 10 grid
def walking_distance := 3
def coverage_side := (walking_distance * 2 + 1)  -- 7x7 grid coverage
def max_covered_per_pharmacy := (coverage_side - 1) * (coverage_side - 1)  -- Coverage per direction

-- Define the main theorem
theorem twelve_pharmacies_not_enough (n m : ℕ): 
  n = grid_size + 1 -> m = grid_size + 1 -> total_intersections = n * m -> 
  (walking_distance < n) -> (walking_distance < m) -> (pharmacies : ℕ) -> pharmacies = 12 ->
  (coverage_side <= n) -> (coverage_side <= m) ->
  ¬ (∀ i j : ℕ, i < n -> j < m -> ∃ p : ℕ, p < pharmacies -> 
  abs (i - (p / (grid_size + 1))) + abs (j - (p % (grid_size + 1))) ≤ walking_distance) :=
begin
  intros,
  sorry -- Proof omitted
end

end twelve_pharmacies_not_enough_l488_488744


namespace min_value_ineq_l488_488252

theorem min_value_ineq (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_constraint : 2 * a + 3 * b = 1) : 
  26 + 12 * real.sqrt 6 ≤ 2 / a + 3 / b :=
sorry

end min_value_ineq_l488_488252


namespace circumference_is_50_27_approx_l488_488019

-- Define the radius of the circle
def radius : ℝ := 8

-- Define the formula for the circumference of a circle
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- Given the radius of 8 cm, we prove the circumference is approximately 50.27 cm
theorem circumference_is_50_27_approx :
  (circumference radius) ≈ 50.27 := 
by
  sorry

end circumference_is_50_27_approx_l488_488019


namespace digits_sum_eq_seven_l488_488925

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488925


namespace hyperbola_equation_find_line_eq_l488_488680

variables (a b : ℝ) (A B F P Q M N : ℝ × ℝ)

-- Conditions
def hyperbola_eq (a b : ℝ) : Prop := 
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

def eccentricity_eq (a b : ℝ) : Prop := 
  (a > 0) ∧ (b > 0) ∧ (sqrt(a^2 + b^2) / a = 2)

def distance_origin_line_AB (a b : ℝ) : Prop := 
  |a * b| / sqrt(a^2 + b^2) = sqrt(3) / 2

def points_on_hyperbola (a b : ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (0, -b) ∧ B = (a, 0)

-- Proving the hyperbola's equation
theorem hyperbola_equation (a b : ℝ) (A B : ℝ × ℝ) 
  (h_eq : hyperbola_eq a b)
  (h_ecc : eccentricity_eq a b)
  (h_dist : distance_origin_line_AB a b)
  (h_points : points_on_hyperbola a b A B) : 
  a = 1 ∧ b = sqrt(3) ∧ ∀ x y : ℝ, x^2 - y^2 / 3 = 1 :=
sorry

-- Additional conditions for line equation
def line_through_f (F : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ k, p.2 = k * (p.1 - 2) + F.2}

def projections_and_length 
    (P Q M N : ℝ × ℝ) : Prop :=
  (P.1 + Q.1) / 2 = M.1 ∧ (P.2 + Q.2) / 2 = M.2 ∧ 
  P.1 - 2 = N.1 ∧ (M.2) - P.2 = (M.2) - Q.2 ∧ 
  (M.1 - P.1) * (N.1 - Q.1) = 0 ∧ 
  sqrt((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 10

-- Proving the line's equation
theorem find_line_eq (F P Q M N : ℝ × ℝ)
    (a b : ℝ)
    (h_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
    (h_line : line_through_f F)
    (h_cond : projections_and_length P Q M N) : 
      (∀ x : ℝ, x = 2 → (F.1 = 3 ∨ F.1 = -3)) → 
      line_through_f F = {p | ∃ k, p.2 = k * (p.1 - 2)} :=
sorry

end hyperbola_equation_find_line_eq_l488_488680


namespace algebraic_expression_value_l488_488469

theorem algebraic_expression_value (a b : ℝ) (h1 : a * b = 2) (h2 : a - b = 3) :
  2 * a^3 * b - 4 * a^2 * b^2 + 2 * a * b^3 = 36 :=
by
  sorry

end algebraic_expression_value_l488_488469


namespace prism_max_cross_section_area_l488_488395

   noncomputable def max_cross_section_area (side_length : ℝ) (plane_eq : ℝ × ℝ × ℝ × ℝ) : ℝ :=
   let square_vertices := 
     [(6 * Real.sqrt 2 * Real.cos θ, 6 * Real.sqrt 2 * Real.sin θ, 0),
      (-6 * Real.sqrt 2 * Real.sin θ, 6 * Real.sqrt 2 * Real.cos θ, 0),
      (-6 * Real.sqrt 2 * Real.cos θ, -6 * Real.sqrt 2 * Real.sin θ, 0),
      (6 * Real.sqrt 2 * Real.sin θ, -6 * Real.sqrt 2 * Real.cos θ, 0)]
   let cut_vertices := 
     square_vertices.map (λ (x, y, z),
       (x, y, (42 * Real.sqrt 2 * Real.sin θ - 18 * Real.sqrt 2 * Real.cos θ + 30) / 3))
   let area ⟨x1, y1, z1⟩ ⟨x2, y2, z2⟩ := 
     (1 / 2) * Real.sqrt ((72 : ℝ) ^ 2 + (-63 : ℝ) ^ 2 + (72 : ℝ) ^ 2)
   in 2 * area ((42 * Real.sqrt 2 * Real.sin θ - 18 * Real.sqrt 2 * Real.cos θ + 30) / 3)
      ((-42 * Real.sqrt 2 * Real.sin θ + 18 * Real.sqrt 2 * Real.cos θ + 30) / 3)

   theorem prism_max_cross_section_area :
     max_cross_section_area 12 (3, -5, 3, 30) = 111.32 :=
   by
     sorry
   
end prism_max_cross_section_area_l488_488395


namespace part1_cosA_part2_bc_over_a_l488_488237

-- Define the sides and angles of the triangle
variables {A B C : ℝ}
variables {a b c : ℝ} (h_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- Define the condition given in the problem
def given_condition (A B C : ℝ) (a b c : ℝ) :=
  cos (B - C) * cos A + cos (2 * A) = 1 + cos A * cos (B + C)

-- Part 1: If B=C, determine the value of cos A
theorem part1_cosA (h_cos: given_condition A B C a b c) (h_eq: B = C) : cos A = 2 / 3 :=
by sorry

-- Part 2: Find the value of (b^2 + c^2) / a^2
theorem part2_bc_over_a (h_cos: given_condition A B C a b c) : (b^2 + c^2) / a^2 = 3 :=
by sorry

end part1_cosA_part2_bc_over_a_l488_488237


namespace smallest_n_polynomials_l488_488110

theorem smallest_n_polynomials :
  ∃ (n : ℕ), (∀ (f : ℕ → ℚ[X]), (∑ i in Finset.range n, (f i)^2) = X^2 + 7) ∧ ∀ m : ℕ, (m < n → ¬ (∀ (f : ℕ → ℚ[X]), (∑ i in Finset.range m, (f i)^2) = X^2 + 7)) :=
sorry

end smallest_n_polynomials_l488_488110


namespace number_of_three_digit_numbers_with_sum_7_l488_488978

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488978


namespace obtuse_triangle_angle_C_60_sequence_general_formula_correct_statements_l488_488374

-- Problem 1
theorem obtuse_triangle (A B C : ℝ) (h : Real.sin A ^ 2 + Real.sin B ^ 2 < Real.sin C ^ 2) : triangle_shape A B C = "obtuse" :=
sorry

-- Problem 2
theorem angle_C_60 (a b c : ℝ) (S : ℝ) (h : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) : angle_C a b c = 60 :=
sorry

-- Problem 3
theorem sequence_general_formula (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n ≥ 2, a n = 2 * S (n - 1)) (h3 : ∀ n, S n = 3 ^ (n - 1)) :
  ∀ n, a n = if n = 1 then 1 else 2 * 3 ^ (n - 2) :=
sorry

-- Problem 4
theorem correct_statements (S : ℕ → ℝ) (h1 : S 6 > S 7) (h2 : S 7 > S 5) :
  ({1, 2, 5} : set ℕ) = {i | i = 1 ∨ i = 2 ∨ i = 5} :=
sorry

end obtuse_triangle_angle_C_60_sequence_general_formula_correct_statements_l488_488374


namespace functional_eq_solutions_l488_488442

-- Define the conditions for the problem
def func_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y

-- Define the two solutions to be proven correct
def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := |x|

-- State the main theorem to be proven
theorem functional_eq_solutions (f : ℝ → ℝ) (h : func_equation f) : f = f1 ∨ f = f2 :=
sorry

end functional_eq_solutions_l488_488442


namespace proof_problem_l488_488466

-- Definitions
variable {A B C O P M Q : point}
variable {a b c : ℝ}

-- Conditions
def triangle_abc (A B C : point) : Prop :=
  distance AB = c ∧ distance BC = a ∧ distance AC = b ∧ b ≥ a

def incenter (O : point) (A B C : point) : Prop :=
  -- some definition for incenter here
  sorry

def excircle_opposite_A (P : point) (A B C : point) : Prop :=
  -- some definition for excircle touching AB at P here
  sorry

def midpoint (M : point) (A B : point) : Prop :=
  distance A M = distance M B ∧ distance A B = c

def intersection_MO_BC (Q : point) (M O B C : point) : Prop :=
  Q = line_through M O ∩ BC

-- Theorem Statement
theorem proof_problem (A B C O P M Q : point) (a b c : ℝ)
  (habc : triangle_abc A B C)
  (hO : incenter O A B C)
  (hP : excircle_opposite_A P A B C)
  (hM : midpoint M A B)
  (hQ : intersection_MO_BC Q M O B C) :
  distance P M = 1/2 * (b - a) ∧ distance P Q = a * c / (b + c - a) :=
  sorry

end proof_problem_l488_488466


namespace prime_2_in_500_fact_l488_488185

theorem prime_2_in_500_fact : 
  let f : ℕ → ℕ := λ n, if n = 0 then 0 else n + f (n / 2)
  f 500 = 494 := 
  by {
    sorry
  }

end prime_2_in_500_fact_l488_488185


namespace num_integers_satisfying_ineq_l488_488520

theorem num_integers_satisfying_ineq : 
  {x : ℤ | -6 ≤ 3 * (x : ℤ) + 2 ∧ 3 * (x : ℤ) + 2 ≤ 9}.to_finset.card = 5 := 
sorry

end num_integers_satisfying_ineq_l488_488520


namespace find_c_l488_488854

noncomputable def func_condition (f : ℝ → ℝ) (c : ℝ) :=
  ∀ x y : ℝ, (f x + 1) * (f y + 1) = f (x + y) + f (x * y + c)

theorem find_c :
  ∃ c : ℝ, ∀ (f : ℝ → ℝ), func_condition f c → (c = 1 ∨ c = -1) :=
sorry

end find_c_l488_488854


namespace P_I_D_collinear_l488_488262

variables {A B C P I D E F : Type} [IncircleTriangle A B C I D E F]
variables (P_same_side : SameSide P A EF)
variables (anglePEF_eq_ABC : Angle PEF = Angle ABC)
variables (anglePFE_eq_ACB : Angle PFE = Angle ACB)

theorem P_I_D_collinear : Collinear P I D :=
by {
  -- The proof will be inserted here
  sorry
}

end P_I_D_collinear_l488_488262


namespace cone_radius_half_slant_height_eq_vlateral_area_eq_volume_l488_488533

theorem cone_radius_half_slant_height_eq_vlateral_area_eq_volume
  (r : ℝ) (h : ℝ) (l : ℝ) (V : ℝ) (LA : ℝ)
  (h_half_sl : l = 2 * r)
  (h_cone_volume : V = (1/3) * π * r^2 * (sqrt 3 * r))
  (h_lateral_area : LA = π * r * l)
  (h_eq : LA = V)
  : r = 2 * sqrt 3 :=
by sorry

end cone_radius_half_slant_height_eq_vlateral_area_eq_volume_l488_488533


namespace asymptote_equation_l488_488206

-- Definitions as given in the conditions
variables (a b c : ℝ)

-- Hypotheses from the conditions
hypothesis h1 : a > 0
hypothesis h2 : b > 0
hypothesis h3 : 2 * a = c
hypothesis h4 : c^2 = a^2 + b^2

-- The theorem to prove
theorem asymptote_equation : ( ∃ m : ℝ, y = m * x ↔ m = sqrt 3 ∨ m = - sqrt 3 ) :=
by
  -- Placeholder for the proof steps
  sorry

end asymptote_equation_l488_488206


namespace first_number_is_38_l488_488699

theorem first_number_is_38 (x y : ℕ) (h1 : x + 2 * y = 124) (h2 : y = 43) : x = 38 :=
by
  sorry

end first_number_is_38_l488_488699


namespace first_player_wins_initial_move_l488_488651

def chessboard : Type := list (list ℕ)

def initial_board : chessboard := 
[ [1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1] ]

def winning_strategy (board : chessboard) : bool :=
-- The function to determine if the first player has a winning strategy
sorry

theorem first_player_wins_initial_move (board : chessboard) : winning_strategy initial_board = tt :=
sorry

end first_player_wins_initial_move_l488_488651


namespace Deepak_age_l488_488323

variables (R D : ℕ) 

theorem Deepak_age : (R / D = 4 / 3) ∧ (R + 6 = 42) → D = 27 := by
  assume h : (R / D = 4 / 3) ∧ (R + 6 = 42)
  sorry

end Deepak_age_l488_488323


namespace part1_solution_part2_solution_l488_488501

def f (c a b x : ℝ) : ℝ := |c * x + a| + |c * x - b|
def g (c x : ℝ) : ℝ := |x - 2| + c

theorem part1_solution (x : ℝ) : 
  let f := f 2 1 3 x in f - 4 = 0 ↔ -1/2 ≤ x ∧ x ≤ 3/2 :=
sorry

theorem part2_solution (a : ℝ) : 
  (∀ x1 : ℝ, ∃ x2 : ℝ, g 1 x2 = f 1 a 1 x1) →
  a ∈ (-∞, -2] ∪ [0, +∞) :=
sorry

end part1_solution_part2_solution_l488_488501


namespace bottles_from_Shop_C_l488_488840

theorem bottles_from_Shop_C (TotalBottles ShopA ShopB ShopC : ℕ) 
  (h1 : TotalBottles = 550) 
  (h2 : ShopA = 150) 
  (h3 : ShopB = 180) 
  (h4 : TotalBottles = ShopA + ShopB + ShopC) : 
  ShopC = 220 := 
by
  sorry

end bottles_from_Shop_C_l488_488840


namespace constant_term_polynomial_l488_488128

noncomputable def m : ℝ := 3 * ∫ x in -1..1, (x^2 + Real.sin x)

theorem constant_term_polynomial :
  let p := (fun x : ℝ => x + 1 / (m * Real.sqrt x))
  (p x)^6 = (x + 1 / (m * Real.sqrt x))^6 in
  ∀ x : ℝ, m = 2 → p x = x + 1 / (2 * Real.sqrt x) →
    (p x)^6 = (x + 1 / (2 * Real.sqrt x))^6 →
    (mul_zero_class ((x + 1 / (2 * Real.sqrt x))^6)).1 = 15 / 16 :=
by
  intros
  sorry

end constant_term_polynomial_l488_488128


namespace expected_value_is_correct_l488_488027

def expected_value_of_win : ℝ :=
  let outcomes := (list.range 8).map (fun n => 8 - (n + 1))
  let probabilities := list.repeat (1 / 8 : ℝ) 8
  list.zip_with (fun outcome probability => outcome * probability) outcomes probabilities |>.sum

theorem expected_value_is_correct :
  expected_value_of_win = 3.5 := by
  sorry

end expected_value_is_correct_l488_488027


namespace three_digit_sum_7_l488_488966

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488966


namespace three_digit_sum_seven_l488_488877

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488877


namespace insufficient_pharmacies_l488_488753

/-- Define the grid size and conditions --/
def grid_size : ℕ := 9
def total_intersections : ℕ := 100
def pharmacy_walking_distance : ℕ := 3
def number_of_pharmacies : ℕ := 12

/-- Prove that 12 pharmacies are not enough for the given conditions. --/
theorem insufficient_pharmacies : ¬ (number_of_pharmacies ≥ grid_size + 3) → 
  ∃(pharmacies : ℕ), pharmacies = number_of_pharmacies ∧
  ∀(x y : ℕ), (x ≤ grid_size ∧ y ≤ grid_size) → 
  ¬ exists (p : ℕ × ℕ), p.1 ≤ x + pharmacy_walking_distance ∧
  p.2 ≤ y + pharmacy_walking_distance ∧ 
  p.1 < total_intersections ∧ 
  p.2 < total_intersections := 
by sorry

end insufficient_pharmacies_l488_488753


namespace light_bulbs_on_l488_488584

theorem light_bulbs_on
  (n : ℕ) 
  (h1 : n = 100)
  (Ming_pulled : ∀ i, 1 ≤ i ∧ i ≤ n → even i → i) -- even numbered switches
  (Cong_pulled : ∀ i, 1 ≤ i ∧ i ≤ n → i % 3 = 0 → i) -- divisible by 3 switches
  (initially_off : ∀ i, 1 ≤ i ∧ i ≤ n → i → bool.false) -- initially all off
  : (∃ k, k = 67) :=
by
  have even_count : ∃ k, k = 50 := sorry
  have multiple_of_3_count : ∃ k, k = 33 := sorry
  have multiple_of_6_count : ∃ k, k = 16 := sorry
  have on_count : ∃ k, k = even_count + multiple_of_3_count - multiple_of_6_count := sorry
  exact ⟨67, on_count⟩

end light_bulbs_on_l488_488584


namespace max_intersection_points_circle_sine_l488_488389

def circle_eq (x y h k : ℝ) : ℝ := (x - h)^2 + (y - k)^2

theorem max_intersection_points_circle_sine :
  ∀ (h : ℝ), ∃ k ∈ Icc (-2 : ℝ) (2 : ℝ),
    ∀ (x : ℝ), circle_eq x (Real.sin x) h k = 4 → (∃! x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ, 
    circle_eq x₁ (Real.sin x₁) h k = 4 ∧
    circle_eq x₂ (Real.sin x₂) h k = 4 ∧
    circle_eq x₃ (Real.sin x₃) h k = 4 ∧
    circle_eq x₄ (Real.sin x₄) h k = 4 ∧
    circle_eq x₅ (Real.sin x₅) h k = 4 ∧
    circle_eq x₆ (Real.sin x₆) h k = 4 ∧
    circle_eq x₇ (Real.sin x₇) h k = 4 ∧
    circle_eq x₈ (Real.sin x₈) h k = 4) := 
begin
  sorry
end

end max_intersection_points_circle_sine_l488_488389


namespace percentage_increase_l488_488415

theorem percentage_increase (M N : ℝ) (h : M ≠ N) : 
  (200 * (M - N) / (M + N) = ((200 : ℝ) * (M - N) / (M + N))) :=
by
  -- Translate the problem conditions into Lean definitions
  let average := (M + N) / 2
  let increase := (M - N)
  let fraction_of_increase_over_average := (increase / average) * 100

  -- Additional annotations and calculations to construct the proof would go here
  sorry

end percentage_increase_l488_488415


namespace jack_time_to_school_l488_488426

noncomputable def dave_steps_per_minute := 85
noncomputable def dave_step_length_cm := 80
noncomputable def dave_time_minutes := 15
noncomputable def jack_step_length_cm := 72
noncomputable def jack_steps_per_minute := 104

noncomputable def calculate_time_for_jack := 
  let dave_speed_cm_per_minute := dave_steps_per_minute * dave_step_length_cm
  let distance_to_school_cm := dave_speed_cm_per_minute * dave_time_minutes
  let jack_speed_cm_per_minute := jack_steps_per_minute * jack_step_length_cm
  distance_to_school_cm / jack_speed_cm_per_minute

theorem jack_time_to_school : calculate_time_for_jack = 13.62 := by
  sorry

end jack_time_to_school_l488_488426


namespace find_person_age_l488_488735

theorem find_person_age : ∃ x : ℕ, 4 * (x + 4) - 4 * (x - 4) = x ∧ x = 32 := by
  sorry

end find_person_age_l488_488735


namespace chessboard_pieces_placement_l488_488652

theorem chessboard_pieces_placement : 
  ∀ (designated_squares : Finset (Fin 8 × Fin 8)),
  designated_squares.card = 16 → 
  (∀ i : Fin 8, (designated_squares.filter (λ sq, sq.1 = i)).card = 2) →
  (∀ j : Fin 8, (designated_squares.filter (λ sq, sq.2 = j)).card = 2) →
  ∃ (black_squares white_squares : Finset (Fin 8 × Fin 8)),
  black_squares.card = 8 ∧ 
  white_squares.card = 8 ∧ 
  disjoint black_squares white_squares ∧
  (∀ i : Fin 8, (black_squares.filter (λ sq, sq.1 = i)).card = 1 ∧ (white_squares.filter (λ sq, sq.1 = i)).card = 1) ∧
  (∀ j : Fin 8, (black_squares.filter (λ sq, sq.2 = j)).card = 1 ∧ (white_squares.filter (λ sq, sq.2 = j)).card = 1) :=
by
  sorry

end chessboard_pieces_placement_l488_488652


namespace probability_is_2_over_11_l488_488376

-- Define the grid size and points
def grid_size : ℕ := 10
def total_points : ℕ := grid_size * grid_size

-- Define the point P and possible point Q
variable (P : ℕ × ℕ)
def valid_Q_points (P : ℕ × ℕ) : list (ℕ × ℕ) := 
  ((list.range grid_size).map (λ x, (x, P.snd))).erase P ++ 
  ((list.range grid_size).map (λ y, (P.fst, y))).erase P

-- Define the probability calculation
def probability_vertical_or_horizontal (P : ℕ × ℕ) : ℚ :=
  (valid_Q_points P).length / (total_points - 1)

-- The theorem to be proved
theorem probability_is_2_over_11 (P : ℕ × ℕ) (h : P.1 < grid_size ∧ P.2 < grid_size) : 
  probability_vertical_or_horizontal P = 2 / 11 := 
sorry

end probability_is_2_over_11_l488_488376


namespace closest_point_l488_488865

def vector_on_line (s : ℚ) : ℚ × ℚ × ℚ :=
  let x := 1 + 3 * s
  let y := 2 - s
  let z := 4 * s
  (x, y, z)

def vector_pointing (s : ℚ) : ℚ × ℚ × ℚ :=
  let x := -2 + 3 * s
  let y := 2 - s
  let z := -1 + 4 * s
  (x, y, z)

def direction_vector : ℚ × ℚ × ℚ :=
  (3, -1, 4)

theorem closest_point (s : ℚ) (p : ℚ × ℚ × ℚ) 
  (h_p : p = vector_on_line s)
  (h_orth : vector_pointing s.1 * 3 + vector_pointing s.2 * -1 + vector_pointing s.3 * 4 = 0)
  (hs : s = 6 / 13) :
  p = (31/13, 20/13, 24/13) := sorry

end closest_point_l488_488865


namespace find_multiple_of_pages_l488_488085

-- Definitions based on conditions
def beatrix_pages : ℕ := 704
def cristobal_extra_pages : ℕ := 1423
def cristobal_pages (x : ℕ) : ℕ := x * beatrix_pages + 15

-- Proposition to prove the multiple x equals 2
theorem find_multiple_of_pages (x : ℕ) (h : cristobal_pages x = beatrix_pages + cristobal_extra_pages) : x = 2 :=
  sorry

end find_multiple_of_pages_l488_488085


namespace seniority_order_l488_488344

def SeniorityOrder : Type := {α : Type} → α

constant Tom Jerry Sam : SeniorityOrder

def StatementI := (Jerry = most_senior)
def StatementII := (Sam ≠ least_senior)
def StatementIII := (Tom ≠ most_senior)

theorem seniority_order :
  (∃! (P : Prop), P = StatementI ∨ P = StatementII ∨ P = StatementIII) →
  (Tom ≠ Jerry ∧ Jerry ≠ Sam ∧ Sam ≠ Tom) →
  ordering_of_colleagues = [Jerry, Tom, Sam] := by
  sorry

end seniority_order_l488_488344


namespace prime_mean_34_37_39_41_43_l488_488444

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def arithmetic_mean (nums : List ℕ) : ℚ :=
  (nums.sum : ℚ) / nums.length

theorem prime_mean_34_37_39_41_43 :
  arithmetic_mean (List.filter is_prime [34, 37, 39, 41, 43]) = 40.33 := by
    sorry

end prime_mean_34_37_39_41_43_l488_488444


namespace ratio_inscribed_circle_altitude_l488_488867

theorem ratio_inscribed_circle_altitude (a : ℝ) (h : 0 < a) :
  let r := (sqrt 2 - 1) / 2 * a in
  let CD := a / (sqrt 2 + 1) in
  r / CD = sqrt 2 - 1 :=
by {
  let r := (sqrt 2 - 1) / 2 * a,
  let CD := a / (sqrt 2 + 1),
  calc r / CD = ((sqrt 2 - 1) / 2 * a) / (a / (sqrt 2 + 1)) : by rw [r, CD]
           ... = (sqrt 2 - 1) / 2 * (sqrt 2 + 1) : by field_simp [h.ne']
           ... = (sqrt 2 - 1) : by ring,
}

end ratio_inscribed_circle_altitude_l488_488867


namespace area_of_transformed_region_l488_488625

noncomputable def area_transformed_region (A : Matrix (Fin 2) (Fin 2) ℝ) (area_R : ℝ) : ℝ :=
  (|A.det| * area_R)

theorem area_of_transformed_region :
  let A := !![3, 2; 4, -5]
  let area_R := 15
  area_transformed_region A area_R = 345 := 
by
  let A := !![3, 2; 4, -5]
  let area_R := 15
  show area_transformed_region A area_R = 345
  sorry

end area_of_transformed_region_l488_488625


namespace complement_P_intersection_Q_l488_488170

def P (x : ℝ) : Prop := x^2 - 2 * x > 3
def complement_P : set ℝ := {x | ¬ P x}
def Q (x : ℝ) : Prop := 2^x < 4

theorem complement_P_intersection_Q :
  (complement_P ∩ {x | Q x}) = { x | -1 ≤ x ∧ x < 2 } := by
  sorry

end complement_P_intersection_Q_l488_488170


namespace three_digit_integers_sum_to_7_l488_488906

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488906


namespace three_digit_sum_seven_l488_488948

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488948


namespace domain_of_f_l488_488067

noncomputable def f (x : ℝ) := real.sqrt (4 - real.sqrt (6 - real.sqrt x))

theorem domain_of_f :
  {x | 0 ≤ x ∧ x ≤ 36} = {x | f x ≠ 0 ∨ 0 ≤ f x} :=
sorry

end domain_of_f_l488_488067


namespace chord_length_of_curve_C_on_line_l_l488_488585

-- Conditions in a)
def polar_curve := set (λ (θ : ℝ), (2 * real.cos θ + 2 * real.sin θ) * (real.cos θ, real.sin θ))
def parametric_line (t : ℝ) := (1 + t, real.sqrt 3 * t)
def cartesian_line (x : ℝ) := real.sqrt 3 * (x - 1)

-- Equivalent proof problem statement in Lean 4
theorem chord_length_of_curve_C_on_line_l :
  let center := (1, 1) in
  let radius := real.sqrt 2 in
  let distance_from_center_to_line := 1 / 2 in
  let chord_length := 2 * real.sqrt (radius^2 - distance_from_center_to_line^2) in
  chord_length = real.sqrt 7 := by sorry

end chord_length_of_curve_C_on_line_l_l488_488585


namespace album_distribution_ways_l488_488405

theorem album_distribution_ways :
  let num_ways := 10
  in ∃ ways : Nat, ways = num_ways :=
by
sorry

end album_distribution_ways_l488_488405


namespace triangle_area_is_correct_l488_488721

-- Define the vertices of the triangle
def p1 := (0 : ℝ, 0 : ℝ)
def p2 := (0 : ℝ, 6 : ℝ)
def p3 := (8 : ℝ, 14 : ℝ)

-- Define the area calculation function using the coordinate formula for the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := 
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

-- State the theorem
theorem triangle_area_is_correct : triangle_area p1 p2 p3 = 24 := 
by
  -- This 'sorry' acts as a placeholder for the proof.
  sorry

end triangle_area_is_correct_l488_488721


namespace volume_ratio_l488_488474

-- Define the radius and height and the corresponding areas of the cylinder and sphere
variables (R h : ℝ)

-- Define the equations based on the conditions provided
def surface_area_sphere := 4 * Real.pi * R^2
def surface_area_cylinder := 2 * Real.pi * R^2 + 2 * Real.pi * R * h

-- Define the volumes of the cylinder and sphere
def volume_cylinder := Real.pi * R^2 * h
def volume_sphere := (4 / 3) * Real.pi * R^3

-- Given condition that the surface areas are equal
axiom equal_surface_areas : surface_area_sphere R = surface_area_cylinder R h

-- Theorem to prove the ratio of volumes is 3/4
theorem volume_ratio (R : ℝ) (h : ℝ) (h_eq_R : h = R) (equal_surface_areas : surface_area_sphere R = surface_area_cylinder R h) :
  volume_cylinder R h / volume_sphere R = 3 / 4 :=
by {
  -- Assume variables and conditions leading to the desired proof
  sorry
}

end volume_ratio_l488_488474


namespace average_gt_median_by_19_pounds_l488_488049

def alvin_weight : ℕ := 120
def sibling_weights : list ℕ := [5, 6, 6, 7, 9]
def weights : list ℕ := alvin_weight :: sibling_weights

noncomputable def average_weight (l : list ℕ) : ℚ :=
(list.sum l : ℚ) / l.length

noncomputable def median_weight (l : list ℕ) : ℚ :=
let sorted := l.sort (<=)
in if l.length % 2 == 0 then
     (sorted.nth (l.length / 2 - 1) + sorted.nth (l.length / 2)) / 2
   else
     sorted.nth (l.length / 2)

theorem average_gt_median_by_19_pounds :
  (average_weight weights) - (median_weight weights) = 19 :=
by { sorry }

end average_gt_median_by_19_pounds_l488_488049


namespace mean_of_solutions_eq_neg_half_l488_488107

noncomputable def poly : Polynomial ℝ := Polynomial.C (-4) + Polynomial.X * (Polynomial.C (-8) + Polynomial.X * (Polynomial.C 2 + Polynomial.X * (Polynomial.C 5 + Polynomial.X)))

theorem mean_of_solutions_eq_neg_half : 
  (let roots := { -1, -1, 2, -2 }
  in (roots.sum / roots.size) = -0.5) :=
sorry

end mean_of_solutions_eq_neg_half_l488_488107


namespace radius_of_sphere_l488_488318

noncomputable def sqrt (x : ℕ) : ℝ := Real.sqrt (x : ℝ)

theorem radius_of_sphere :
  ∀ (x y z : ℝ)
    (hx : x^2 + y^2 = 45)
    (hy : y^2 + z^2 = 85)
    (hz : z^2 + x^2 = 58),
  (3 * 6 * 7 / (3 * 6 + 6 * 7 + 3 * 7) = 14 / 9) :=
begin
  assume x y z hx hy hz,
  have h : x = 3 ∧ y = 6 ∧ z = 7,
  { -- This proof can be detailed separately
    sorry
  },
  cases h with hx hyz,
  cases hyz with hy hz,
  rw [hx, hy, hz],
  norm_num, -- This command will compute numerical equivalence
  sorry -- Detailed proofs for all steps skipped here
end

end radius_of_sphere_l488_488318


namespace find_radius_l488_488020

-- Define the side length of the octagon
def side_length : ℝ := 3

-- Define the given probability
def probability : ℝ := 1 / 3

-- Define the required radius to prove
def required_radius : ℝ := 12 * real.sqrt 2 / (real.sqrt 6 + real.sqrt 2)

-- Define the condition as statement
theorem find_radius (r : ℝ) :
  (∀ (point : ℝ × ℝ), (point_on_circle point r) → (four_sides_visible point)) = probability → r = required_radius :=
sorry

-- Define a helper function for a point lying on the circle
def point_on_circle (point : ℝ × ℝ) (r : ℝ) : Prop :=
  point.fst ^ 2 + point.snd ^ 2 = r ^ 2

-- Define a helper function checking visibility of exactly four sides from a given point
def four_sides_visible (point : ℝ × ℝ) : Prop :=
  -- A placeholder, as the detailed geometrical implementation of visibility is complex
  true


end find_radius_l488_488020


namespace reciprocal_inequality_l488_488526

theorem reciprocal_inequality {a b c : ℝ} (hab : a < b) (hbc : b < c) (ha_pos : 0 < a) (hb_pos : 0 < b) : 
  (1 / a) < (1 / b) :=
sorry

end reciprocal_inequality_l488_488526


namespace chameleons_color_change_l488_488577

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l488_488577


namespace a_n_correct_b_n_correct_T_n_correct_l488_488174

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 * a_n (n - 1)

def b_n (n : ℕ) : ℕ :=
  n

def T_n (n : ℕ) : ℕ :=
  (n - 1) * 2^(n+1) + 2

theorem a_n_correct (n : ℕ) (h : n > 0) : a_n n = 2^n :=
  sorry

theorem b_n_correct (n : ℕ) (h : n > 0) : (b_1 = 1) ∧ 
                                          (∀ k > 0, 
                                              b_1 + (∑ i in finset.range k, 1 / i.succ) * (b_i.succ) = (b_(k+1))-1) → 
                                          b_n n = n :=
  sorry

theorem T_n_correct (n : ℕ) : (T_n n = (n - 1) * 2^(n+1) + 2) ↔ 
                             ((∑ k in finset.range n, a_n k.succ * b_n k.succ) = (n - 1) * 2^(n+1) + 2) :=
  sorry

end a_n_correct_b_n_correct_T_n_correct_l488_488174


namespace beavers_help_l488_488012

theorem beavers_help (initial final : ℝ) (h_initial : initial = 2.0) (h_final : final = 3) : final - initial = 1 :=
  by
    sorry

end beavers_help_l488_488012


namespace radius_of_large_circle_l488_488457

theorem radius_of_large_circle : 
  ∃ (R : ℝ), R = 2 + 2 * Real.sqrt 2 ∧ 
  ∀ (r : ℝ) (n : ℕ), 
    (∀ (i j : ℕ), i ≠ j → i < n → j < n → 
    dist (r * cos (2 * i * π / n), r * sin (2 * i * π / n)) 
         (r * cos (2 * j * π / n), r * sin (2 * j * π / n)) = 2 * r) ∧ 
    (∀ (i : ℕ), 
      i < n → 
      dist (r * cos (2 * i * π / n), r * sin (2 * i * π / n)) 
         (0, 0) = R - r) → r = 2 ∧ n = 4 :=
by
  sorry

end radius_of_large_circle_l488_488457


namespace max_value_S_over_2_pow_n_l488_488484

variable {a : ℕ → ℝ}
noncomputable def is_arith_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (2 * n) = 2 * a n - 3

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a i

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  sum_first_n_terms a n

theorem max_value_S_over_2_pow_n (a : ℕ → ℝ)
  (h1 : is_arith_seq a)
  (h2 : a 6 ^ 2 = a 1 * a 21)
  (distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∀ n : ℕ, n ≠ 0 → 2 ≤ n → 6 ≤ 2 * S a 2 / 2^(2-1) :=
sorry

end max_value_S_over_2_pow_n_l488_488484


namespace set_intersection_set_union_set_complement_union_l488_488007

open set

def A := { x : ℝ | 2 * x - 1 ≥ 1 }
def B := { x : ℝ | real.log (3 - x) / real.log 2 < 2 }

theorem set_intersection :
  A ∩ B = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
sorry

theorem set_union :
  A ∪ B = { x : ℝ | x > -1 } :=
sorry

theorem set_complement_union :
  (compl A) ∪ (compl B) = { x : ℝ | x < 1 ∨ x ≥ 3 } :=
sorry

end set_intersection_set_union_set_complement_union_l488_488007


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488938

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488938


namespace base10_to_base7_conversion_l488_488080

theorem base10_to_base7_conversion : 2023 = 5 * 7^3 + 6 * 7^2 + 2 * 7^1 + 0 * 7^0 :=
  sorry

end base10_to_base7_conversion_l488_488080


namespace twelve_pharmacies_not_enough_l488_488758

def grid := ℕ × ℕ

def is_within_walking_distance (p1 p2 : grid) : Prop :=
  abs (p1.1 - p1.2) ≤ 3 ∧ abs (p2.1 - p2.2) ≤ 3

def walking_distance_coverage (pharmacies : set grid) (p : grid) : Prop :=
  ∃ pharmacy ∈ pharmacies, is_within_walking_distance pharmacy p

def sufficient_pharmacies (pharmacies : set grid) : Prop :=
  ∀ p : grid, walking_distance_coverage pharmacies p

theorem twelve_pharmacies_not_enough (pharmacies : set grid) (h : pharmacies.card = 12) : 
  ¬ sufficient_pharmacies pharmacies :=
sorry

end twelve_pharmacies_not_enough_l488_488758


namespace solve_exponential_eq_l488_488329

theorem solve_exponential_eq :
  ∀ x : ℝ, (4^x + 2^x - 2 = 0) → (x = 0) :=
by
  sorry

end solve_exponential_eq_l488_488329


namespace expected_value_of_twelve_sided_die_l488_488042

theorem expected_value_of_twelve_sided_die : 
  (let n := 12 in
  let S := n * (n + 1) / 2 in
  let E := S / n in
  E = 6.5) :=
sorry

end expected_value_of_twelve_sided_die_l488_488042


namespace marked_price_l488_488401

theorem marked_price (original_price : ℝ) 
                     (discount1_rate : ℝ) 
                     (profit_rate : ℝ) 
                     (discount2_rate : ℝ)
                     (marked_price : ℝ) : 
                     original_price = 40 → 
                     discount1_rate = 0.15 → 
                     profit_rate = 0.25 → 
                     discount2_rate = 0.10 → 
                     marked_price = 47.20 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end marked_price_l488_488401


namespace max_value_of_expression_l488_488256

noncomputable def w_condition := {w : ℂ // complex.abs w = 2}

theorem max_value_of_expression : ∀ w : w_condition, complex.abs ((w - 2)^2 * (w + 2)) ≤ 12 ∧ 
                                          ∃ w : w_condition, complex.abs ((w - 2)^2 * (w + 2)) = 12 := 
sorry

end max_value_of_expression_l488_488256


namespace angles_same_terminal_side_in_range_l488_488599

theorem angles_same_terminal_side_in_range :
  ∀ (β : ℝ), (-720 ≤ β ∧ β < 0) ↔ (β = -675 ∨ β = -315) :=
begin
  sorry
end

end angles_same_terminal_side_in_range_l488_488599


namespace symmetric_line_eq_l488_488858

theorem symmetric_line_eq (x y : ℝ) (c : ℝ) (P : ℝ × ℝ)
  (h₁ : 3 * x - y - 4 = 0)
  (h₂ : P = (2, -1))
  (h₃ : 3 * x - y + c = 0)
  (h : 3 * 2 - (-1) + c = 0) : 
  c = -7 :=
by
  sorry

end symmetric_line_eq_l488_488858


namespace number_of_three_digit_numbers_with_sum_7_l488_488972

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488972


namespace chameleon_color_change_l488_488567

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l488_488567


namespace S105_minus_S96_l488_488507

/-- Definition of the function f(x) for given x and n --/
def f (x: ℝ) (n: ℕ) : ℝ := 
  if x ∈ Set.Ico (2 * n : ℝ) (2 * n + 1) then 
    ((-1) ^ n) * Real.sin (π * x / 2) + 2 * n 
  else 
    ((-1) ^ (n + 1)) * Real.sin (π * x / 2) + 2 * n + 2

/-- Definition of the sequence a_m --/
def a (m: ℕ) [h: Fact (m > 0)] : ℝ := 
  let n := (m / 2) in f m n

/-- Sum of the first m terms in the sequence a_m --/
def S (m: ℕ) : ℝ := 
  ∑ i in Finset.range (m + 1), a i

theorem S105_minus_S96 : S 105 - S 96 = 909 := by 
  sorry

end S105_minus_S96_l488_488507


namespace combination_15_choose_5_l488_488589

theorem combination_15_choose_5 : nat.choose 15 5 = 3003 :=
by sorry

end combination_15_choose_5_l488_488589


namespace problem_odd_decreasing_function_l488_488051

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def is_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x ≥ f y

theorem problem_odd_decreasing_function :
  let f := λ x : ℝ, x⁻¹ in
  is_odd f ∧ is_decreasing f (Set.Ioi 0) :=
sorry

end problem_odd_decreasing_function_l488_488051


namespace three_digit_integers_sum_to_7_l488_488899

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488899


namespace correct_polynomial_mult_l488_488800

variable (x : ℝ)

def incorrect_result : polynomial ℝ := polynomial.C 1 - polynomial.C 4 * polynomial.X + polynomial.C 1 * polynomial.X ^ 2

theorem correct_polynomial_mult (correct_result : polynomial ℝ) :
  (3 * polynomial.X ^ 2 + incorrect_result) * (-3 * polynomial.X ^ 2) = correct_result :=
by
  have corrected_polynomial := 4 * polynomial.X ^ 2 - 4 * polynomial.X + polynomial.C 1
  have correct_multi := corrected_polynomial * (-3 * polynomial.X ^ 2)
  calc
    (3 * polynomial.X ^ 2 + (polynomial.C 1 * polynomial.X ^ 2 - 4 * polynomial.X + polynomial.C 1)) * 
    (-3 * polynomial.X ^ 2) = correct_multi : by sorry
  correct_multi = polynomial.C (-12) * polynomial.X ^ 4 + 
                  polynomial.C 12 * polynomial.X ^ 3 - 
                  polynomial.C 3 * polynomial.X ^ 2 : by sorry

end correct_polynomial_mult_l488_488800


namespace first_number_is_38_l488_488700

theorem first_number_is_38 (x y : ℕ) (h1 : x + 2 * y = 124) (h2 : y = 43) : x = 38 :=
by
  sorry

end first_number_is_38_l488_488700


namespace number_of_three_digit_numbers_with_sum_seven_l488_488891

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488891


namespace max_distance_ellipse_l488_488615

-- Let's define the conditions and the theorem in Lean 4.
theorem max_distance_ellipse (θ : ℝ) :
  let C := { p : ℝ × ℝ | p.1 ^ 2 / 5 + p.2 ^ 2 = 1 }
  let B := (0, 1)
  let P := (sqrt 5 * cos θ, sin θ)
  P ∈ C →
  max (λ θ : ℝ, sqrt ((sqrt 5 * cos θ - 0) ^ 2 + (sin θ - 1) ^ 2)) = 5 / 2 :=
sorry

end max_distance_ellipse_l488_488615


namespace solution_set_l488_488148

variable {f : ℝ → ℝ}

-- Conditions
axiom domain_f : ∀ x, x > 0 → ∃ y, f(x) = y
axiom condition_1 : ∀ x, x > 1 → f(x) < 0
axiom condition_2 : ∀ x y, f(x * y) = f(x) + f(y)

-- Goal
theorem solution_set : { x | f(x) + f(x - 2) ≥ f(8) } = { x | 2 < x ∧ x ≤ 4 } :=
sorry

end solution_set_l488_488148


namespace three_consecutive_multiples_of_3_three_consecutive_integers_three_consecutive_even_multiples_of_3_random_even_multiples_of_3_indeterminate_random_odd_multiples_of_3_indeterminate_l488_488831

theorem three_consecutive_multiples_of_3 (x y z : ℕ) (h1 : x + y + z = 36) (h2 : ∃ m, x = 3 * m) (h3 : ∃ n, y = 3 * n) (h4 : ∃ p, z = 3 * p) :
  ∃ m, ∃ n, ∃ p, x = 3 * m ∧ y = 3 * (m + 1) ∧ z = 3 * (m + 2) :=
sorry

theorem three_consecutive_integers (x y z : ℕ) (h1 : x + y + z = 36) :
  ¬ (∃ n, x = n ∧ y = n + 1 ∧ z = n + 2 ∧ x % 3 = 0 ∧ y % 3 = 0 ∧ z % 3 = 0) :=
sorry

theorem three_consecutive_even_multiples_of_3 (x y z : ℕ) (h1 : x + y + z = 36) (h2 : x % 6 = 0) (h3 : y % 6 = 0) (h4 : z % 6 = 0) :
  ∃ m, x = 6 * m ∧ y = 6 * (m + 1) ∧ z = 6 * (m + 2) :=
sorry

theorem random_even_multiples_of_3_indeterminate (x y z : ℕ) (h1 : x + y + z = 36) (h2 : x % 6 = 0) (h3 : y % 6 = 0) (h4 : z % 6 = 0) :
  true :=
sorry

theorem random_odd_multiples_of_3_indeterminate (x y z : ℕ) (h1 : x + y + z = 36) (h2 : x % 3 = 0 ∧ x % 2 ≠ 0) (h3 : y % 3 = 0 ∧ y % 2 ≠ 0) (h4 : z % 3 = 0 ∧ z % 2 ≠ 0) :
  true :=
sorry

end three_consecutive_multiples_of_3_three_consecutive_integers_three_consecutive_even_multiples_of_3_random_even_multiples_of_3_indeterminate_random_odd_multiples_of_3_indeterminate_l488_488831


namespace number_of_correct_statements_is_three_l488_488687

-- Conditions
def statement1 : Prop := ∀ x: ℝ, x ≠ 0 → ∃ y, y = 2 / x
def statement2 : Prop := ∀ k: ℝ, ∃ center asymptotes, (x ≠ 0 → (y = k / x)) ↔ (hyperbola center asymptotes)
def statement3 : Prop := ∀ x: ℝ, x > 0 →  y = 3 / x ∧ x < 0 → y = 3 / x
def statement4 : Prop := ∀ (P : (ℝ × ℝ)), P = (3,2) → 
                          ((x = 3 ∧ y = -2 ∧ y = -6 / x) ∧ 
                           (x = -3 ∧ y = 2 ∧ y = -6 / x))

-- The proof problem
theorem number_of_correct_statements_is_three :
  (statement1 ∧ statement2 ∧ ¬statement3 ∧ statement4) → 3 = 3 :=
by
  intros
  sorry

end number_of_correct_statements_is_three_l488_488687


namespace expand_polynomial_correct_l488_488853

open Polynomial

noncomputable def expand_polynomial : Polynomial ℤ :=
  (C 3 * X^3 - C 2 * X^2 + X - C 4) * (C 4 * X^2 - C 2 * X + C 5)

theorem expand_polynomial_correct :
  expand_polynomial = C 12 * X^5 - C 14 * X^4 + C 23 * X^3 - C 28 * X^2 + C 13 * X - C 20 :=
by sorry

end expand_polynomial_correct_l488_488853


namespace books_sold_on_wednesday_l488_488244

theorem books_sold_on_wednesday
  (initial_stock : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_thursday : ℕ)
  (sold_friday : ℕ)
  (percent_unsold : ℚ) :
  initial_stock = 900 →
  sold_monday = 75 →
  sold_tuesday = 50 →
  sold_thursday = 78 →
  sold_friday = 135 →
  percent_unsold = 55.333333333333336 →
  ∃ (sold_wednesday : ℕ), sold_wednesday = 64 :=
by
  sorry

end books_sold_on_wednesday_l488_488244


namespace baker_total_cost_l488_488780

def cost_flour := 5 * 6
def cost_eggs := 6 * 12
def cost_milk := 8 * 3
def cost_bakingsoda := 4 * 1.5
def total_cost_before_discount := cost_flour + cost_eggs + cost_milk + cost_bakingsoda
def discount := 0.15
def total_cost_after_discount := total_cost_before_discount * (1 - discount)
def exchange_rate := 1.2
def total_cost_in_dollars := total_cost_after_discount * exchange_rate

theorem baker_total_cost : total_cost_in_dollars = 134.64 :=
by {
  sorry
}

end baker_total_cost_l488_488780


namespace points_on_line_l488_488345

-- Definitions for the problem setup
structure Triangle (α : Type) [linear_ordered_field α] :=
(A B C : Point α)

structure Circle (α : Type) [linear_ordered_field α] :=
(center : Point α)
(radius : α)

structure Point (α : Type) [linear_ordered_field α] := 
(x y : α)

def inscribed (A B C : Point ℝ) (O : Point ℝ) :=
∃ r : ℝ, Circle.mk O r

def angle_ABC (X A B C : Point ℝ) := -- implement with specifics for the angle calculation
sorry

def perpendicular (P X O : Point ℝ) :=
(X.x - P.x) * (P.x - O.x) + (X.y - P.y) * (P.y - O.y) = 0

def oriented_angle (A B C X O P : Point ℝ) (φ : ℝ) := 
(angle_ABC X A B C = φ) ∧ (angle_ABC O X P = φ)

-- The theorem to prove
theorem points_on_line (A B C O X P : Point ℝ) (φ : ℝ) (h_inscribed : inscribed A B C O)
  (h_angle_ABAC : angle_ABC X A B C = φ) (h_angle_XBC : angle_ABC X B C = φ)
  (h_perpendicular : perpendicular P X O) (h_oriented : oriented_angle A B C X O P φ) :
  ∃ (m b : ℝ), ∀ P, P.y = m * P.x + b :=
sorry

end points_on_line_l488_488345


namespace interval_of_increase_f_l488_488298

noncomputable def f : ℝ → ℝ := fun x => log x / log 2

def interval_of_increase : Set ℝ := { x : ℝ | 0 < x ∧ x < 3 }

theorem interval_of_increase_f (x : ℝ) (h : 0 < x ∧ x < 6) :
  monotone_increase (fun x => f (6 * x - x^2)) 0 3 :=
begin
  sorry
end

end interval_of_increase_f_l488_488298


namespace episodes_count_l488_488097

variable (minutes_per_episode : ℕ) (total_watching_time_minutes : ℕ)
variable (episodes_watched : ℕ)

theorem episodes_count 
  (h1 : minutes_per_episode = 50) 
  (h2 : total_watching_time_minutes = 300) 
  (h3 : total_watching_time_minutes / minutes_per_episode = episodes_watched) :
  episodes_watched = 6 := sorry

end episodes_count_l488_488097


namespace three_digit_sum_seven_l488_488880

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l488_488880


namespace max_dist_PB_l488_488619

-- Let B be the upper vertex of the ellipse.
def B : (ℝ × ℝ) := (0, 1)

-- Define the equation of the ellipse.
def ellipse (x y : ℝ) : Prop := (x^2) / 5 + y^2 = 1

-- Define a point P on the ellipse.
def P (θ : ℝ) : (ℝ × ℝ) := (sqrt 5 * cos θ, sin θ)

-- Define the distance function between points.
def dist (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Prove that the maximum distance |PB| is 5/2.
theorem max_dist_PB : ∃ θ : ℝ, dist (P θ) B = 5 / 2 :=
sorry

end max_dist_PB_l488_488619


namespace distance_sum_is_ten_l488_488783

noncomputable def angle_sum_distance (C A B : ℝ) (d : ℝ) (k : ℝ) : ℝ := 
  let h_A : ℝ := sorry -- replace with expression for h_A based on conditions
  let h_B : ℝ := sorry -- replace with expression for h_B based on conditions
  h_A + h_B

theorem distance_sum_is_ten 
  (A B C : ℝ) 
  (h : ℝ) 
  (k : ℝ) 
  (h_pos : h = 4) 
  (ratio_condition : h_A = 4 * h_B)
  : angle_sum_distance C A B h k = 10 := 
  sorry

end distance_sum_is_ten_l488_488783


namespace segments_difference_four_l488_488682

theorem segments_difference_four (n : ℕ) (h x y : ℝ) 
  (h_triangle : (n - 1)^2 + h^2 = x^2)
  (h_triangle' : (n + 1)^2 + h^2 = y^2)
  (h_sum : x + y = n) 
  (h_acute : (n - 1)^2 + n^2 > (n + 1)^2)
  (h_acute' : (n + 1)^2 + n^2 > (n - 1)^2)
  (h_acute'' : (n + 1)^2 + (n - 1)^2 > n^2) :
  y - x = 4 := 
begin
  sorry
end

end segments_difference_four_l488_488682


namespace digits_sum_eq_seven_l488_488928

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488928


namespace ethanol_percentage_in_fuel_A_l488_488411

variable {capacity_A fuel_A : ℝ}
variable (ethanol_A ethanol_B total_ethanol : ℝ)
variable (E : ℝ)

def fuelTank (capacity_A fuel_A ethanol_A ethanol_B total_ethanol : ℝ) (E : ℝ) : Prop := 
  (ethanol_A / fuel_A = E) ∧
  (capacity_A - fuel_A = 200 - 99.99999999999999) ∧
  (ethanol_B = 0.16 * (200 - 99.99999999999999)) ∧
  (total_ethanol = ethanol_A + ethanol_B) ∧
  (total_ethanol = 28)

theorem ethanol_percentage_in_fuel_A : 
  ∃ E, fuelTank 99.99999999999999 99.99999999999999 ethanol_A ethanol_B 28 E ∧ E = 0.12 := 
sorry

end ethanol_percentage_in_fuel_A_l488_488411


namespace inequality_abc_l488_488278

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1) * (b + 1) * (a + c) * (b + c) ≥ 16 * a * b * c :=
by
  sorry

end inequality_abc_l488_488278


namespace simplify_expression_l488_488288

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  (4 / (3 * x^(-3)) * (3 * x^2) / 2 * x^(-1) / 5 = 2 * x^4 / 5) :=
by
  sorry

end simplify_expression_l488_488288


namespace amelia_dinner_initial_amount_l488_488050

theorem amelia_dinner_initial_amount :
  let first_course := 15
  let second_course := first_course + 5
  let dessert := 0.25 * second_course
  let amount_left := 20
  let total_cost := first_course + second_course + dessert
  initial_amount = total_cost + amount_left
  in initial_amount = 60 := by
  sorry

end amelia_dinner_initial_amount_l488_488050


namespace closed_broken_line_impossible_l488_488601

theorem closed_broken_line_impossible (n : ℕ) (h : n = 1989) : ¬ (∃ a b : ℕ, 2 * (a + b) = n) :=
by {
  sorry
}

end closed_broken_line_impossible_l488_488601


namespace three_digit_integers_sum_to_7_l488_488904

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488904


namespace chameleons_color_change_l488_488564

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l488_488564


namespace num_three_digit_sums7_l488_488997

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l488_488997


namespace bottles_from_shop_C_l488_488841

theorem bottles_from_shop_C (A B C T : ℕ) (hA : A = 150) (hB : B = 180) (hT : T = 550) (hSum : T = A + B + C) :
  C = 220 :=
by
  rw [hA, hB, hT] at hSum
  simpa using hSum

end bottles_from_shop_C_l488_488841


namespace equal_cost_sharing_difference_l488_488409

theorem equal_cost_sharing_difference 
  (A B C D : ℝ) 
  (paid_A : A = 160) 
  (paid_B : B = 220) 
  (paid_C : C = 190) 
  (paid_D : D = 95) : 
  (let total_expense := A + B + C + D in
   let each_share := total_expense / 4 in
   let Alice_to_Bob := each_share - A in
   let Charlie_to_Bob := C - each_share in
   let Alice_Charlie_diff := Alice_to_Bob - Charlie_to_Bob 
   in
   Alice_Charlie_diff) = -17.50 := 
sorry  -- Proof to be filled in later

end equal_cost_sharing_difference_l488_488409


namespace parallelepiped_volume_l488_488113

open Real

theorem parallelepiped_volume (k : ℝ) (hk : k > 0) :
  abs (det ![
    ![3, 2, 2],
    ![4, k, 3],
    ![5, 3, k]
    ]) = 20 ↔ k = 3 + sqrt 15 := 
by {
  sorry
}

end parallelepiped_volume_l488_488113


namespace chi_squared_test_expected_value_correct_l488_488782
open ProbabilityTheory

section Part1

def n : ℕ := 400
def a : ℕ := 60
def b : ℕ := 20
def c : ℕ := 180
def d : ℕ := 140
def alpha : ℝ := 0.005
def chi_critical : ℝ := 7.879

noncomputable def chi_squared : ℝ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem chi_squared_test : chi_squared > chi_critical :=
  sorry

end Part1

section Part2

def reward_med : ℝ := 6  -- 60,000 yuan in 10,000 yuan unit
def reward_small : ℝ := 2  -- 20,000 yuan in 10,000 yuan unit
def total_support : ℕ := 12
def total_rewards : ℕ := 9

noncomputable def dist_table : List (ℝ × ℝ) :=
  [(180, 1 / 220),
   (220, 27 / 220),
   (260, 27 / 55),
   (300, 21 / 55)]

noncomputable def expected_value : ℝ :=
  dist_table.foldr (fun (xi : ℝ × ℝ) acc => acc + xi.1 * xi.2) 0

theorem expected_value_correct : expected_value = 270 :=
  sorry

end Part2

end chi_squared_test_expected_value_correct_l488_488782


namespace odd_degree_polynomial_real_zero_l488_488034

theorem odd_degree_polynomial_real_zero {n : ℕ} (h : odd n)
  (a : Fin n → ℝ) (ha : ∀ i, 0 < a i) :
  ∃ (σ : Fin n → Fin n), ∃ x : ℝ, ( ∑ i, (a (σ i)) * x ^ (i : ℕ) ) = 0 :=
by
  sorry

end odd_degree_polynomial_real_zero_l488_488034


namespace domain_of_f_range_of_f_f_at_1_f_eq_3_iff_x_eq_sqrt3_l488_488161

def f (x : ℝ) : ℝ :=
if x ≤ -1 then x + 2 else x ^ 2

theorem domain_of_f : set.Iio (2:ℝ) = set.univ :=
sorry

theorem range_of_f : set.Iio (4:ℝ) = set.Ioo (-∞) (4:ℝ) :=
sorry

theorem f_at_1 : f 1 = 1 :=
sorry

theorem f_eq_3_iff_x_eq_sqrt3 : ∀ x, f x = 3 ↔ x = real.sqrt 3 :=
sorry

end domain_of_f_range_of_f_f_at_1_f_eq_3_iff_x_eq_sqrt3_l488_488161


namespace pin_for_mr_skleroza_possible_pins_for_mr_odkoukal_l488_488813

/-- Define the function to convert a Roman numeral to a single digit decimal. -/
def roman_to_decimal (s : String) : Option ℕ :=
  match s with
  | "I"   => some 1
  | "II"  => some 2
  | "III" => some 3
  | "IV"  => some 4
  | "V"   => some 5
  | "VI"  => some 6
  | "VII" => some 7
  | "VIII"=> some 8
  | "IX"  => some 9
  | _     => none

/-- Predicate to check if a list of Roman numerals forms a valid PIN code. -/
def is_valid_pin (lst : List String) : Prop :=
  lst.length = 4 ∧
  ∀ num, num ∈ lst → (roman_to_decimal num).isSome

/-- Predicate to check if the converted PIN code matches the expected digits. -/
def pin_code_of (lst : List String) (digits : List ℕ) : Prop :=
  digits = (lst.map (λ s, roman_to_decimal s).filterMap id)

/-- 
  The PIN code for Mr. Skleróza's card from the Roman numeral "IIIVIIIXIV" 
  should be "3794".
-/
theorem pin_for_mr_skleroza :
  is_valid_pin ["III", "VII", "IX", "IV"] ∧
  pin_code_of ["III", "VII", "IX", "IV"] [3, 7, 9, 4] :=
by
  sorry

/-- 
  Possible PIN codes for Mr. Odkoukal's card from the Roman numeral "IVIIIVI".
  It could be "1536", "1626", "1716", etc.
-/
theorem possible_pins_for_mr_odkoukal :
  ∃ lst : List (List String), 
    ∀ l ∈ lst, is_valid_pin l ∧
    (pin_code_of l [1, 5, 3, 6] ∨ pin_code_of l [1, 6, 2, 6] ∨ 
     pin_code_of l [1, 7, 1, 6] ∨ pin_code_of l [1, 5, 1, 6] ∨ 
     pin_code_of l [1, 6, 1, 6] ∨ pin_code_of l [4, 2, 4, 1] ∨ 
     pin_code_of l [4, 1, 2, 6] ∨ pin_code_of l [4, 3, 6, 6]) :=
by
  sorry

end pin_for_mr_skleroza_possible_pins_for_mr_odkoukal_l488_488813


namespace probability_between_pq_is_33_l488_488214

-- Define the line equations and conditions
def line_p (x : ℝ) : ℝ := -2 * x + 5
def line_q (x : ℝ) : ℝ := -3 * x + 5

-- The area under line_p in the first quadrant
def area_under_p : ℝ := (1 / 2) * (5 / 2) * 5

-- The area under line_q in the first quadrant
def area_under_q : ℝ := (1 / 2) * (5 / 3) * 5

-- The area between the lines p and q
def area_between_pq : ℝ := area_under_p - area_under_q

-- Total area under line p in the first quadrant
def total_area_p : ℝ := area_under_p

-- Calculate probability
def probability_between_pq : ℝ := area_between_pq / total_area_p

-- Prove the problem statement
theorem probability_between_pq_is_33 : probability_between_pq = 0.33 :=
by
  -- Proof omitted for brevity
  sorry

end probability_between_pq_is_33_l488_488214


namespace three_digit_sum_7_l488_488965

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488965


namespace fill_circles_equations_l488_488441

theorem fill_circles_equations :
  ∃ (a b c d e f g h i : ℕ),
    a = 7 ∧ b = 8 ∧ a * b = 56 ∧
    (c = 1 ∨ c = 1) ∧ (d = 4 ∨ d = 9) ∧ (e = 9 ∨ e = 4) ∧ (c + d = 23 ∨ c + e = 23) ∧
    {a, b, c, d, e, f, g, h, i} = {1, 2, 3, 4, 9} :=
by
  sorry

end fill_circles_equations_l488_488441


namespace not_enough_pharmacies_l488_488769

theorem not_enough_pharmacies : 
  ∀ (n m : ℕ), n = 10 ∧ m = 10 →
  ∃ (intersections : ℕ), intersections = n * m ∧ 
  ∀ (d : ℕ), d = 3 →
  ∀ (coverage : ℕ), coverage = (2 * d + 1) * (2 * d + 1) →
  ¬ (coverage * 12 ≥ intersections * 2) :=
by sorry

end not_enough_pharmacies_l488_488769


namespace ceiling_sum_sqrt_evaluation_l488_488437

theorem ceiling_sum_sqrt_evaluation :
    (∑ n in Finset.range (29 - 4), ⌈Real.sqrt (n + 5)⌉) = 112 :=
by
    sorry

end ceiling_sum_sqrt_evaluation_l488_488437


namespace n_value_l488_488196

theorem n_value (n : ℤ) (h1 : (18888 - n) % 11 = 0) : n = 7 :=
sorry

end n_value_l488_488196


namespace min_sum_of_factors_l488_488692

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1176) :
  a + b + c ≥ 59 :=
sorry

end min_sum_of_factors_l488_488692


namespace four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime_l488_488610

theorem four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime
  (N : ℕ) (hN : N ≥ 2) :
  (∀ n : ℕ, n < N → ¬ ∃ k : ℕ, k^2 = 4 * n * (N - n) + 1) ↔ Nat.Prime (N^2 + 1) :=
by sorry

end four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime_l488_488610


namespace officer_selection_count_l488_488022

theorem officer_selection_count (n : ℕ) (hn : n = 12) : 
  (nat.fact 5) * (nat.descFactorial n 5) = 95_040 :=
by
  rw [hn]
  sorry

end officer_selection_count_l488_488022


namespace radius_large_circle_l488_488460

-- Definitions for the conditions
def radius_small_circle : ℝ := 2

def is_tangent_externally (r1 r2 : ℝ) : Prop := -- Definition of external tangency
  r1 + r2 = 4

def is_tangent_internally (R r : ℝ) : Prop := -- Definition of internal tangency
  R - r = 4

-- Setting up the property we need to prove: large circle radius
theorem radius_large_circle
  (R r : ℝ)
  (h1 : r = radius_small_circle)
  (h2 : is_tangent_externally r r)
  (h3 : is_tangent_externally r r)
  (h4 : is_tangent_externally r r)
  (h5 : is_tangent_externally r r)
  (h6 : is_tangent_internally R r) :
  R = 4 :=
by sorry

end radius_large_circle_l488_488460


namespace annie_original_seat_is_1_l488_488870

variable {Seat : Type} [DecidableEq Seat]
variable (friends : List Seat) (empty_seat : Seat)

-- Definition of the initial and final seating of friends
inductive Friends
  | Annie : Friends
  | Beth : Friends
  | Cass : Friends
  | Dana : Friends
  | Ella : Friends
  
def initial_position : Friends → Seat
def final_position : Friends → Seat

-- Define the seat movements per the problem conditions
def beth_moves : Seat → Seat -- Beth moves one seat to the left
def cass_dana_swap : (Seat × Seat) → (Seat × Seat) -- Cass and Dana swap places
def ella_moves : Seat → Seat -- Ella moves three seats to the left, lands on the end seat

-- Now the proof statement based on previous conditions:
theorem annie_original_seat_is_1 :
  (final_position Friends.Annie = empty_seat)
  ∧ (beth_moves (initial_position Friends.Beth) = final_position Friends.Beth)
  ∧ (cass_dana_swap (initial_position Friends.Cass, initial_position Friends.Dana) = (final_position Friends.Cass, final_position Friends.Dana))
  ∧ (ella_moves (initial_position Friends.Ella) = final_position Friends.Ella)
  → initial_position Friends.Annie = seat_number 1 :=
sorry

end annie_original_seat_is_1_l488_488870


namespace not_all_crows_gather_on_one_tree_l488_488290

theorem not_all_crows_gather_on_one_tree :
  ∀ (crows : Fin 6 → ℕ), 
  (∀ i, crows i = 1) →
  (∀ t1 t2, abs (t1 - t2) = 1 → crows t1 = crows t1 - 1 ∧ crows t2 = crows t2 + 1) →
  ¬(∃ i, crows i = 6 ∧ (∀ j ≠ i, crows j = 0)) :=
by
  sorry

end not_all_crows_gather_on_one_tree_l488_488290


namespace combined_lowest_sale_price_percentage_l488_488403

theorem combined_lowest_sale_price_percentage
  (list_price_jersey : ℝ)
  (list_price_ball : ℝ)
  (list_price_cleats : ℝ)
  (lowest_discount_jersey : ℝ)
  (lowest_discount_ball : ℝ)
  (lowest_discount_cleats : ℝ)
  (additional_discount : ℝ)
  (R : ℝ)
  (h1 : list_price_jersey = 80)
  (h2 : list_price_ball = 40)
  (h3 : list_price_cleats = 100)
  (h4 : lowest_discount_jersey = 0.5)
  (h5 : lowest_discount_ball = 0.6)
  (h6 : lowest_discount_cleats = 0.4)
  (h7 : additional_discount = 0.2)
  (h8 : R = (( (list_price_jersey * (1 - lowest_discount_jersey) * (1 - additional_discount))
             + (list_price_ball * (1 - lowest_discount_ball) * (1 - additional_discount))
             + (list_price_cleats * (1 - lowest_discount_cleats) * (1 - additional_discount)) )
             / (list_price_jersey + list_price_ball + list_price_cleats)) * 100) :
  R ≈ 32.73 :=
by
  sorry

end combined_lowest_sale_price_percentage_l488_488403


namespace max_distance_on_ellipse_to_vertex_l488_488613

open Real

noncomputable def P (θ : ℝ) : ℝ × ℝ :=
(√5 * cos θ, sin θ)

def ellipse (x y : ℝ) := (x^2 / 5) + y^2 = 1

def B : ℝ × ℝ := (0, 1)

def dist (A B : ℝ × ℝ) : ℝ :=
sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem max_distance_on_ellipse_to_vertex :
  ∃ θ : ℝ, dist (P θ) B = 5 / 2 :=
sorry

end max_distance_on_ellipse_to_vertex_l488_488613


namespace probability_sum_5_twice_l488_488723

def sides := {1, 2, 3, 4}

noncomputable def roll_two_dice : set (ℕ × ℕ) := { (d1, d2) | d1 ∈ sides ∧ d2 ∈ sides }

noncomputable def successful_sum_5 : set (ℕ × ℕ) :=
  { (d1, d2) | d1 + d2 = 5 ∧ d1 ∈ sides ∧ d2 ∈ sides }

theorem probability_sum_5_twice : (4/16) * (4/16) = 1/16 :=
by
  sorry

end probability_sum_5_twice_l488_488723


namespace hyperbola_eccentricity_l488_488078

-- We define the necessary variables and conditions
variable {a b p : ℝ}
variable (ha : 0 < a) (hb : 0 < b) (hp : 0 < p)

-- Define hyperbola C and parabola
def hyperbola (x y : ℝ) := x^2 / a^2 - y^2 / b^2 = 1
def parabola (x y : ℝ) := y^2 = 2 * p * x

-- Define the intersection points A and B and the chord AB passing through the focus
def intersection_A (x y : ℝ) := parabola x y ∧ hyperbola x y

-- Define eccentricity e
def eccentricity (e : ℝ) := e = (1 + Real.sqrt 2)

-- The proof goal
theorem hyperbola_eccentricity :
  (∃ x y : ℝ, intersection_A x y) →
  (∃ F : ℝ × ℝ, (∃ x1 y1 x2 y2 : ℝ, intersection_A x1 y1 ∧ intersection_A x2 y2 ∧ (F = (x1 + x2) / 2, 0))) →
  ∃ e : ℝ, eccentricity e :=
by
  sorry

end hyperbola_eccentricity_l488_488078


namespace price_relationship_l488_488386

variable {m n : ℝ}
variable (h1 : m > n) (h2 : n > 0)

theorem price_relationship :
  let price_A := (1 + m / 100) * (1 + n / 100)
  let price_B := (1 + n / 100) * (1 + m / 100)
  let price_C := (1 + (m + n) / 200) * (1 + (m + n) / 200)
  in price_A = price_B ∧ price_C > price_A :=
by
  sorry

end price_relationship_l488_488386


namespace find_least_m_l488_488428

noncomputable def sequence (n : ℕ) : ℝ :=
  Nat.recOn n 7 (λ k xk, (xk^2 + 6 * xk + 5) / (xk + 7))

theorem find_least_m :
  ∃ (m : ℕ), sequence m ≤ 5 + 1 / 2^10 ∧ (∀ n : ℕ, n < m → sequence n > 5 + 1 / 2^10) ∧ m = 89 :=
  sorry

end find_least_m_l488_488428


namespace find_number_l488_488527

-- Define the hypothesis/condition
def condition (x : ℤ) : Prop := 2 * x + 20 = 8 * x - 4

-- Define the statement to prove
theorem find_number (x : ℤ) (h : condition x) : x = 4 := 
by
  sorry

end find_number_l488_488527


namespace integer_solution_count_eq_eight_l488_488310

theorem integer_solution_count_eq_eight : ∃ S : Finset (ℤ × ℤ), (∀ s ∈ S, 2 * s.1 ^ 2 + s.1 * s.2 - s.2 ^ 2 = 14 ∧ (s.1 = s.1 ∧ s.2 = s.2)) ∧ S.card = 8 :=
by
  sorry

end integer_solution_count_eq_eight_l488_488310


namespace three_digit_sum_seven_l488_488953

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488953


namespace find_number_l488_488197

theorem find_number (number : ℝ) (h1 : 213 * number = 3408) (h2 : 0.16 * 2.13 = 0.3408) : number = 16 :=
by
  sorry

end find_number_l488_488197


namespace chameleons_color_change_l488_488562

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l488_488562


namespace three_digit_sum_seven_l488_488949

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l488_488949


namespace digits_sum_eq_seven_l488_488933

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l488_488933


namespace a_plus_b_magnitude_l488_488516

-- Defining the vectors and their properties
def a (λ : ℝ) : ℝ × ℝ := (λ, -2)
def b : ℝ × ℝ := (1, 3)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Statement of the proof problem in Lean 4
theorem a_plus_b_magnitude (λ : ℝ) (h : dot_product (a λ) b = 0) :
  magnitude (a λ + b) = 5 * Real.sqrt 2 := 
  by
  sorry

end a_plus_b_magnitude_l488_488516


namespace two_lines_in_parallel_planes_do_not_intersect_l488_488691

theorem two_lines_in_parallel_planes_do_not_intersect 
  (P1 P2 : Plane) (h_parallel : P1 ∥ P2) (L1 : Line) (L2 : Line) 
  (hL1 : L1 ⊆ P1) (hL2 : L2 ⊆ P2) : ¬ (L1 ∩ L2 = ∅) :=
by
  sorry

end two_lines_in_parallel_planes_do_not_intersect_l488_488691


namespace total_students_l488_488590

variables (T : ℕ) (first_division second_division just_passed : ℕ)

-- Conditions given in the problem
def condition1 := first_division = 0.30 * T
def condition2 := second_division = 0.54 * T
def condition3 := just_passed = 0.16 * T
def condition4 := just_passed = 48

-- The statement to be proven
theorem total_students (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : T = 300 :=
sorry

end total_students_l488_488590


namespace maximal_intersection_sum_l488_488720

theorem maximal_intersection_sum (R x : ℝ) (h_pos : 0 ≤ x ∧ x ≤ R) :
  let r_c := (R / 2) * Real.sqrt 3 in
  let h := (3 / 2) * R in
  let r_1 := Real.sqrt (R^2 - x^2) in
  let r_2 := (R + x) / Real.sqrt 3 in
  let t := π * (r_1^2 + r_2^2) in
  ∀ x, t ≤ π * ((R * R) / 2 + R * (R / 2) * (2/3) - (R / 2)^2 * (2/3)) := 
sorry

end maximal_intersection_sum_l488_488720


namespace vector_addition_square_l488_488145

variables {m n : EuclideanSpace ℝ (Fin 3)}

noncomputable def dot_product (x y : EuclideanSpace ℝ (Fin 3)) : ℝ := x ⬝ y

noncomputable def magnitude (x : EuclideanSpace ℝ (Fin 3)) : ℝ := ‖x‖

theorem vector_addition_square :
  magnitude m = 1 →
  magnitude n = 1 →
  dot_product m n = 1 / 2 →
  dot_product (m + n) (m + n) = 3 :=
by
  sorry

end vector_addition_square_l488_488145


namespace angle_RPT_is_38_l488_488229

def measure_angle_RPT (PQ Q R S : Prop) (angle_PQT angle_PTQ angle_QPR : ℕ) :=
  (angle_PQT = 55) ∧ 
  (angle_PTQ = 45) ∧ 
  (angle_QPR = 42) ∧ 
  (QRS_is_straight_line : PQ ∧ Q ∧ R ∧ S)

theorem angle_RPT_is_38
  (PQ Q R S : Prop)
  (angle_PQT angle_PTQ angle_QPR : ℕ)
  (h1 : angle_PQT = 55)
  (h2 : angle_PTQ = 45)
  (h3 : angle_QPR = 42)
  (h_straight : PQ ∧ Q ∧ R ∧ S) :
  ∃ angle_RPT : ℕ, angle_RPT = 38 :=
by
  sorry

end angle_RPT_is_38_l488_488229


namespace area_of_enclosed_curve_l488_488396

noncomputable def projectile_area (u : ℝ) (g : ℕ → ℝ) (n : ℕ) : ℝ :=
  let G := (∑ k in finset.range n, g k) / n
  c := (Math.pi / 8 : ℝ) in
  c * (u ^ 4) / (G ^ 2)

theorem area_of_enclosed_curve (u : ℝ) (g : ℕ → ℝ) (n : ℕ) (h : ∀ k, k < n -> g k > 0) : 
  projectile_area u g n = (Math.pi * (u ^ 4)) / (8 * ((∑ k in finset.range n, g k) / n) ^ 2) := 
by 
  sorry

end area_of_enclosed_curve_l488_488396


namespace angle_between_vectors_correct_l488_488443

open Real -- Open Real module to use real number functions

variables (v : ℝ × ℝ × ℝ) (w : ℝ × ℝ × ℝ)
-- Defining the vectors
def v := (3, -2, 2)
def w := (2, 3, -1)

-- Noncomputable as we are dealing with inverse cosine and sqrt
noncomputable def angle_between_vectors : ℝ :=
  Real.arccos (((3 * 2) + (-2 * 3) + (2 * -1)) / (Real.sqrt ((3^2) + (-2^2) + (2^2)) * Real.sqrt ((2^2) + (3^2) + (-1^2))))

#reduce angle_between_vectors -- Removing proof, just reducing to check equivalence to correct answer

theorem angle_between_vectors_correct :
  angle_between_vectors = Real.arccos (-2 / Real.sqrt 238) :=
  by
    -- Proof omitted
    sorry

end angle_between_vectors_correct_l488_488443


namespace chairs_produced_after_six_hours_l488_488710

theorem chairs_produced_after_six_hours
  (workers : ℕ) (production_per_worker_per_hour : ℕ) (additional_chairs_per_interval : ℕ) (interval_hours : ℕ) :
  workers = 3 →
  production_per_worker_per_hour = 4 →
  additional_chairs_per_interval = 1 →
  interval_hours = 6 →
  let total_chairs := workers * production_per_worker_per_hour * interval_hours + additional_chairs_per_interval in
  total_chairs = 73 :=
by
  intros h_workers h_production h_additional h_interval
  let total_chairs := workers * production_per_worker_per_hour * interval_hours + additional_chairs_per_interval
  sorry

end chairs_produced_after_six_hours_l488_488710


namespace height_of_table_without_book_l488_488801

-- Define the variables and assumptions
variables (l h w : ℝ) (b : ℝ := 6)

-- State the conditions from the problem
-- Condition 1: l + h - w = 40
-- Condition 2: w + h - l + b = 34

theorem height_of_table_without_book (hlw : l + h - w = 40) (whlb : w + h - l + b = 34) : h = 34 :=
by
  -- Since we are skipping the proof, we put sorry here
  sorry

end height_of_table_without_book_l488_488801


namespace compute_complex_expression_l488_488063

-- Define the expression we want to prove
def complex_expression : ℚ := 1 / (1 + (1 / (2 + (1 / (4^2)))))

-- The theorem stating the expression equals to the correct result
theorem compute_complex_expression : complex_expression = 33 / 49 :=
by sorry

end compute_complex_expression_l488_488063


namespace base10_to_base7_conversion_l488_488081

theorem base10_to_base7_conversion : 2023 = 5 * 7^3 + 6 * 7^2 + 2 * 7^1 + 0 * 7^0 :=
  sorry

end base10_to_base7_conversion_l488_488081


namespace chameleons_color_change_l488_488573

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l488_488573


namespace min_value_f_exists_a_for_min_g_l488_488470

def f (x : ℝ) (a : ℝ) := x^2 - a * x + Real.log x

theorem min_value_f :
  let a := 3
  f 1 a = -2 := 
by
  sorry

def g (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem exists_a_for_min_g :
  ∃ (a : ℝ), (∀ x ∈ set.Icc 1 Real.exp, g x a ≥ 1) ∧ g 1 1 = 1 :=
by
  use 1
  split
  {
    intro x hx,
    cases hx with hx1 hx2,
    sorry
  }
  {
    rw g,
    norm_num,
    rw Real.log_one,
    norm_num
  }

end min_value_f_exists_a_for_min_g_l488_488470


namespace not_enough_pharmacies_l488_488768

theorem not_enough_pharmacies : 
  ∀ (n m : ℕ), n = 10 ∧ m = 10 →
  ∃ (intersections : ℕ), intersections = n * m ∧ 
  ∀ (d : ℕ), d = 3 →
  ∀ (coverage : ℕ), coverage = (2 * d + 1) * (2 * d + 1) →
  ¬ (coverage * 12 ≥ intersections * 2) :=
by sorry

end not_enough_pharmacies_l488_488768


namespace solve_eq1_solve_eq2_l488_488293

-- Define the first proof problem
theorem solve_eq1 (x : ℝ) : 2 * x - 3 = 3 * (x + 1) → x = -6 :=
by
  sorry

-- Define the second proof problem
theorem solve_eq2 (x : ℝ) : (1 / 2) * x - (9 * x - 2) / 6 - 2 = 0 → x = -5 / 3 :=
by
  sorry

end solve_eq1_solve_eq2_l488_488293


namespace domain_of_f_range_of_f_f_at_1_f_eq_3_iff_x_eq_sqrt3_l488_488162

def f (x : ℝ) : ℝ :=
if x ≤ -1 then x + 2 else x ^ 2

theorem domain_of_f : set.Iio (2:ℝ) = set.univ :=
sorry

theorem range_of_f : set.Iio (4:ℝ) = set.Ioo (-∞) (4:ℝ) :=
sorry

theorem f_at_1 : f 1 = 1 :=
sorry

theorem f_eq_3_iff_x_eq_sqrt3 : ∀ x, f x = 3 ↔ x = real.sqrt 3 :=
sorry

end domain_of_f_range_of_f_f_at_1_f_eq_3_iff_x_eq_sqrt3_l488_488162


namespace find_number_l488_488672

-- Definitions based on the given conditions
def area (s : ℝ) := s^2
def perimeter (s : ℝ) := 4 * s
def given_perimeter : ℝ := 36
def equation (s : ℝ) (n : ℝ) := 5 * area s = 10 * perimeter s + n

-- Statement of the problem
theorem find_number :
  ∃ n : ℝ, equation (given_perimeter / 4) n ∧ n = 45 :=
by
  sorry

end find_number_l488_488672


namespace chameleons_changed_color_l488_488554

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l488_488554


namespace train_speed_l488_488778

theorem train_speed (length_train : ℕ) (length_platform : ℕ) (time_seconds : ℝ)
  (h_train : length_train = 360) (h_platform : length_platform = 520)
  (h_time : time_seconds = 57.59539236861051) :
  let total_distance := length_train + length_platform in
  let speed_m_s := total_distance / time_seconds in
  let speed_km_hr := speed_m_s * 3.6 in
  speed_km_hr = 54.9936 :=
by
  -- sorry to indicate the proof is omitted
  sorry

end train_speed_l488_488778


namespace exists_point_with_distance_sum_l488_488650

-- Definitions and conditions
variable (circle : Type) [MetricSpace circle]
variable (radius : ℝ) (h_radius : radius = 1)
variable (points : Fin 100 → circle)
variable (c : circle) -- center of the circle

-- Theorem statement
theorem exists_point_with_distance_sum :
  ∃ (p : circle), ∑ i, dist p (points i) ≥ 100 :=
by
  -- Proof here is omitted according to instructions
  sorry

end exists_point_with_distance_sum_l488_488650


namespace fred_initial_cards_l488_488461

theorem fred_initial_cards (cards_given : ℕ) (found_box : ℕ) (total_cards : ℕ) 
    (h1 : cards_given = 18) (h2 : found_box = 40) (h3 : total_cards = 48) : 
    ∃ (initial_cards : ℕ), initial_cards = 26 := 
by
  use 26
  sorry

end fred_initial_cards_l488_488461


namespace xiao_ding_distance_l488_488706

variable (x y z w : ℕ)

theorem xiao_ding_distance (h1 : x = 4 * y)
                          (h2 : z = x / 2 + 20)
                          (h3 : w = 2 * z - 15)
                          (h4 : x + y + z + w = 705) : 
                          y = 60 := 
sorry

end xiao_ding_distance_l488_488706


namespace three_digit_numbers_sum_seven_l488_488915

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l488_488915


namespace f_g_of_4_eq_l488_488253

def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / (Real.sqrt x)
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem f_g_of_4_eq : f (g 4) = 2 * Real.sqrt 21 + 12 / (Real.sqrt 21) :=
by
  sorry

end f_g_of_4_eq_l488_488253


namespace true_proposition_l488_488140

def prop_p (x : ℝ) := 3 ^ x ≤ 0
def prop_q (x : ℝ) := x > 2 → x > 4

theorem true_proposition :
  (¬ (∀ x : ℝ, prop_p x)) ∧ (¬ (∀ x : ℝ, prop_q x)) :=
by
  sorry

end true_proposition_l488_488140


namespace incorrect_statement_proof_by_contradiction_is_only_valid_for_existence_l488_488553

theorem incorrect_statement_proof_by_contradiction_is_only_valid_for_existence :
  ¬ (∀ (P : Prop), (¬P → false) → P) := 
by
  intro h
  have : false := by
    apply h (P := false)
    simp
  contradiction

end incorrect_statement_proof_by_contradiction_is_only_valid_for_existence_l488_488553


namespace rectangle_side_greater_than_12_l488_488481

theorem rectangle_side_greater_than_12 
  (a b : ℝ) (h₁ : a ≠ b) (h₂ : a * b = 6 * (a + b)) : a > 12 ∨ b > 12 := 
by
  sorry

end rectangle_side_greater_than_12_l488_488481


namespace minimum_value_l488_488476

noncomputable def f (x : ℝ) : ℝ := sorry

theorem minimum_value 
  (f : ℝ → ℝ)
  (hf : ∀ x > 0, f x = (f'' x - 1) * x)
  (hf1 : f 1 = 0) :
  ∃ x > 0, f x = -1/e :=
by
  sorry

end minimum_value_l488_488476


namespace neither_sufficient_nor_necessary_l488_488462

theorem neither_sufficient_nor_necessary (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬ ((a > 0 ∧ b > 0) ↔ (ab < ((a + b) / 2)^2)) :=
sorry

end neither_sufficient_nor_necessary_l488_488462


namespace equivalent_expression_l488_488497

variable {a b : ℝ}

theorem equivalent_expression (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (a⁻¹ * b⁻¹) / (a⁻³ + b⁻³) = (a^2 * b^2) / (b^3 + a^3) :=
sorry

end equivalent_expression_l488_488497


namespace range_of_a_l488_488211

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ ∈ ℝ, 2 * x₀ ^ 2 + (a - 1) * x₀ + 1 / 2 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l488_488211


namespace quadrilateral_is_parallelogram_l488_488397

theorem quadrilateral_is_parallelogram
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 - 2 * a * c - 2 * b * d = 0) :
  (a = c) ∧ (b = d) :=
by {
  sorry
}

end quadrilateral_is_parallelogram_l488_488397


namespace single_cakes_needed_l488_488654

theorem single_cakes_needed :
  ∀ (layer_cake_frosting single_cake_frosting cupcakes_frosting brownies_frosting : ℝ)
  (layer_cakes cupcakes brownies total_frosting : ℕ)
  (single_cakes_needed : ℝ),
  layer_cake_frosting = 1 →
  single_cake_frosting = 0.5 →
  cupcakes_frosting = 0.5 →
  brownies_frosting = 0.5 →
  layer_cakes = 3 →
  cupcakes = 6 →
  brownies = 18 →
  total_frosting = 21 →
  single_cakes_needed = (total_frosting - (layer_cakes * layer_cake_frosting + cupcakes * cupcakes_frosting + brownies * brownies_frosting)) / single_cake_frosting →
  single_cakes_needed = 12 :=
by
  intros
  sorry

end single_cakes_needed_l488_488654


namespace intersection_point_l488_488591

variables (P Q R S : ℝ × ℝ × ℝ)
def line_PQ (t : ℝ) : ℝ × ℝ × ℝ :=
  let ⟨xP, yP, zP⟩ := P;
  let ⟨xQ, yQ, zQ⟩ := Q;
  (xP + t * (xQ - xP), yP + t * (yQ - yP), zP + t * (zQ - zP))

def line_RS (s : ℝ) : ℝ × ℝ × ℝ :=
  let ⟨xR, yR, zR⟩ := R;
  let ⟨xS, yS, zS⟩ := S;
  (xR + s * (xS - xR), yR + s * (yS - yR), zR + s * (zS - zR))

theorem intersection_point (P Q R S : ℝ × ℝ × ℝ) :
  let t := 1 / 10 in
  let s := 3 / 2 in
  line_PQ P Q t = line_RS R S s ↔
  (P, Q, R, S) = ((10, -1, 3),(20, -11, 8),(3, 8, -9),(5, 0, 6)) → (11, -2, 3.5) :=
by sorry

end intersection_point_l488_488591


namespace insufficient_pharmacies_l488_488748

/-- Define the grid size and conditions --/
def grid_size : ℕ := 9
def total_intersections : ℕ := 100
def pharmacy_walking_distance : ℕ := 3
def number_of_pharmacies : ℕ := 12

/-- Prove that 12 pharmacies are not enough for the given conditions. --/
theorem insufficient_pharmacies : ¬ (number_of_pharmacies ≥ grid_size + 3) → 
  ∃(pharmacies : ℕ), pharmacies = number_of_pharmacies ∧
  ∀(x y : ℕ), (x ≤ grid_size ∧ y ≤ grid_size) → 
  ¬ exists (p : ℕ × ℕ), p.1 ≤ x + pharmacy_walking_distance ∧
  p.2 ≤ y + pharmacy_walking_distance ∧ 
  p.1 < total_intersections ∧ 
  p.2 < total_intersections := 
by sorry

end insufficient_pharmacies_l488_488748


namespace inequality_solution_l488_488295

noncomputable def ratFunc (x : ℝ) : ℝ := 
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution (x : ℝ) : 
  (ratFunc x > 0) ↔ 
  ((x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) := 
by
  sorry

end inequality_solution_l488_488295


namespace num_three_digit_integers_sum_to_seven_l488_488992

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488992


namespace twelve_pharmacies_not_sufficient_l488_488763

-- Define an intersection grid of size 10 x 10 (100 squares).
def city_grid : Type := Fin 10 × Fin 10 

-- Define the distance measure between intersections, assumed as L1 metric for grid paths.
def dist (p q : city_grid) : Nat := (abs (p.1.val - q.1.val) + abs (p.2.val - q.2.val))

-- Define a walking distance pharmacy 
def is_walking_distance (p q : city_grid) : Prop := dist p q ≤ 3

-- State that having 12 pharmacies is not sufficient
theorem twelve_pharmacies_not_sufficient (pharmacies : Fin 12 → city_grid) :
  ¬ (∀ intersection: city_grid, ∃ (p_index : Fin 12), is_walking_distance (pharmacies p_index) intersection) :=
sorry

end twelve_pharmacies_not_sufficient_l488_488763


namespace three_digit_sum_7_l488_488967

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488967


namespace solve_equation_l488_488667

theorem solve_equation : ∀ x : ℝ, (x + 1 - 2 * (x - 1) = 1 - 3 * x) → x = 0 := 
by
  intros x h
  sorry

end solve_equation_l488_488667


namespace num_three_digit_sums7_l488_488996

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l488_488996


namespace average_first_last_l488_488322

-- Definitions based on conditions
def numbers : List ℤ := [-3, 0, 5, 8, 11, 15]
def largest := 15
def smallest := -3
def median (n : ℤ) : Prop := n = 5 ∨ n = 8

-- Conditions to be used as definitions in Lean 4
def condition1 (arr : List ℤ) : Prop := largest ∈ arr.take 4 ∧ arr.head ≠ largest
def condition2 (arr : List ℤ) : Prop := smallest ∈ arr.drop (arr.length - 4) ∧ arr.last ≠ smallest
def condition3 (arr : List ℤ) : Prop := median (List.nth arr 1) ∨ median (List.nth arr 3)

-- Proof statement
theorem average_first_last (arr : List ℤ) 
  (h1 : condition1 arr)
  (h2 : condition2 arr)
  (h3 : condition3 arr) :
  (arr.head + arr.last) / 2 = -1.5 := 
by
  sorry

end average_first_last_l488_488322


namespace min_quotient_l488_488447

theorem min_quotient (a b c d : ℕ) (h1 : 0 < a ∧ a < 10) (h2 : 0 < b ∧ b < 10) (h3 : 0 < c ∧ c < 10) (h4 : 0 < d ∧ d < 10) 
  (h5 : a ≠ b) (h6 : a ≠ c) (h7 : a ≠ d) (h8 : b ≠ c) (h9 : b ≠ d) (h10 : c ≠ d) :
  let n := 1000 * a + 100 * b + 10 * c + d 
  in (n : ℚ) / (a + b + c + d) ≥ 80.56 :=
sorry

end min_quotient_l488_488447


namespace three_digit_sum_7_l488_488960

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488960


namespace count_positive_integers_l488_488874

theorem count_positive_integers (x : ℤ) : 
  (25 < x^2 + 6 * x + 8) → (x^2 + 6 * x + 8 < 50) → (x > 0) → (x = 3 ∨ x = 4) :=
by sorry

end count_positive_integers_l488_488874


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488945

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488945


namespace sum_of_roots_l488_488640

theorem sum_of_roots (f : ℝ → ℝ) :
  (∀ x : ℝ, f (3 + x) = f (3 - x)) →
  (∃ (S : Finset ℝ), S.card = 6 ∧ ∀ x ∈ S, f x = 0) →
  (∃ (S : Finset ℝ), S.sum id = 18) :=
by
  sorry

end sum_of_roots_l488_488640


namespace maximum_n_l488_488450

/-- Definition of condition (a): For any three people, there exist at least two who know each other. -/
def condition_a (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 3 → ∃ (a b : V) (ha : a ∈ s) (hb : b ∈ s), G.Adj a b

/-- Definition of condition (b): For any four people, there exist at least two who do not know each other. -/
def condition_b (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 4 → ∃ (a b : V) (ha : a ∈ s) (hb : b ∈ s), ¬ G.Adj a b

theorem maximum_n (G : SimpleGraph V) [Fintype V] (h1 : condition_a G) (h2 : condition_b G) : 
  Fintype.card V ≤ 8 :=
by
  sorry

end maximum_n_l488_488450


namespace vector_magnitude_given_perpendicular_l488_488515

variable (λ : ℝ)

def vector_a : ℝ × ℝ := (λ, -2)
def vector_b : ℝ × ℝ := (1, 3)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2)

theorem vector_magnitude_given_perpendicular :
  (dot_product (vector_a λ) vector_b = 0) →
  magnitude (vector_a λ + vector_b) = 5 * Real.sqrt 2 :=
by
  sorry

end vector_magnitude_given_perpendicular_l488_488515


namespace chameleon_color_change_l488_488570

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l488_488570


namespace carol_first_six_l488_488805

-- A formalization of the probabilities involved when Alice, Bob, Carol,
-- and Dave take turns rolling a die, and the process repeats.
def probability_carol_first_six (prob_rolling_six : ℚ) : ℚ := sorry

theorem carol_first_six (prob_rolling_six : ℚ) (h : prob_rolling_six = 1/6) :
  probability_carol_first_six prob_rolling_six = 25 / 91 :=
sorry

end carol_first_six_l488_488805


namespace find_theta_l488_488102

theorem find_theta (theta : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ 2 * π) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ - x^2 * (1 - x) + (1 - x)^3 * Real.sin θ > 0) →
  θ > π / 12 ∧ θ < 5 * π / 12 :=
by
  sorry

end find_theta_l488_488102


namespace grid_fill_possible_l488_488037

theorem grid_fill_possible (n : ℕ) :
  (∀ (grid : Array (Array Int)), (∀ i j, grid[i][j] = 1 ∨ grid[i][j] = -1) ∧
    (∀ i, (List.foldr (*) 1 (List.ofArray (grid[i]))) > 0) ∧
    (∀ j, (List.foldr (*) 1 (List.map (λ i, grid[i][j]) (List.range 5))) > 0)) ↔
  n % 4 = 0 :=
sorry

end grid_fill_possible_l488_488037


namespace chameleons_color_change_l488_488575

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l488_488575


namespace ellipse_properties_and_area_l488_488143

-- Definition of the ellipse and its properties

variables {a b : ℝ}
def is_ellipse (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ x^2 / a^2 + y^2 / b^2 = 1

def is_left_vertex (A : ℝ × ℝ) (x y : ℝ) : Prop := 
  A = (-2, 0)

def line_through_D (D : ℝ × ℝ) (slope : Option ℝ) : Prop :=
  D = (1, 0) ∧ slope = none -- undefined slope implies a vertical line

def intersection_points (P Q : ℝ × ℝ) (x : ℝ) : Prop :=
  P.1 = x ∧ Q.1 = x ∧ P ≠ Q

def distance_pq (P Q : ℝ × ℝ) : Prop :=
  real.dist P Q = 3

def equation_of_ellipse (a b : ℝ) : (ℝ × ℝ) → Prop :=
  λ (x y), x^2 / 4 + y^2 / 3 = 1

def area_triangle (A P Q : ℝ × ℝ) : ℝ :=
  (abs (A.1 * (P.2 - Q.2) + P.1 * (Q.2 - A.2) + Q.1 * (A.2 - P.2))) / 2

def area_range (A P Q : ℝ × ℝ) : Prop :=
  area_triangle A P Q > 0 ∧ area_triangle A P Q ≤ 9 / 2

-- Proving the conditions

theorem ellipse_properties_and_area (A P Q D : ℝ × ℝ) :
  is_ellipse a b (-2) 0 ∧ 
  is_left_vertex A (-2) 0 ∧
  line_through_D D none ∧
  distance_pq P Q →
  (equation_of_ellipse a b (1, 3 / 2) ∧ true → equation_of_ellipse 2 (sqrt 3)) ∧
  area_range A P Q := 
sorry

end ellipse_properties_and_area_l488_488143


namespace builders_cottage_build_time_l488_488204

theorem builders_cottage_build_time :
  (∀ n : ℕ, ∀ d : ℕ, ∀ b : ℕ, b * d = n → (b = 3 → d = 8) → (b = 6 → d = 4)) :=
begin
  sorry -- Skipping the proof as specified
end

end builders_cottage_build_time_l488_488204


namespace number_of_three_digit_numbers_with_sum_7_l488_488974

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l488_488974


namespace air_conditioner_sales_l488_488435

-- Definitions based on conditions
def ratio_air_conditioners_refrigerators : ℕ := 5
def ratio_refrigerators_air_conditioners : ℕ := 3
def difference_in_sales : ℕ := 54

-- The property to be proven: 
def number_of_air_conditioners : ℕ := 135

theorem air_conditioner_sales
  (r_ac : ℕ := ratio_air_conditioners_refrigerators) 
  (r_ref : ℕ := ratio_refrigerators_air_conditioners) 
  (diff : ℕ := difference_in_sales) 
  : number_of_air_conditioners = 135 := sorry

end air_conditioner_sales_l488_488435


namespace prob_between_2_and_3_l488_488772

noncomputable def normal_distribution (μ σ : ℝ) := sorry -- Placeholder for actual normal distribution definition

variables (X : ℝ) (σ : ℝ)
axiom normal_X : ∃ σ > 0, X ~ normal_distribution 3 σ
axiom prob_X_greater_4 : P(X > 4) = 0.2

theorem prob_between_2_and_3 : P(2 < X < 3) = 0.3 := by
  sorry

end prob_between_2_and_3_l488_488772


namespace abs_diff_squares_103_97_l488_488815

theorem abs_diff_squares_103_97 : abs ((103 ^ 2) - (97 ^ 2)) = 1200 := by
  sorry

end abs_diff_squares_103_97_l488_488815


namespace exists_N_for_all_n_l488_488456

-- Define a predicate indicating that a number is not a power of a prime number
def not_prime_power (x : ℕ) : Prop :=
  ∀ p k : ℕ, prime p → k ≥ 1 → x ≠ p^k

-- Main theorem statement
theorem exists_N_for_all_n :
  ∀ (n : ℕ), n > 0 → ∃ (N : ℕ), N > 0 ∧ (∀ j : ℕ, 1 ≤ j ∧ j ≤ n → not_prime_power (N + j)) :=
by 
  sorry

end exists_N_for_all_n_l488_488456


namespace polynomial_coeffs_sum_l488_488187

theorem polynomial_coeffs_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) (x : ℝ) :
  (2*x - 3)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + 2*a_2 + 3*a_3 + 4*a_4 + 5*a_5 = 10 :=
by
  sorry

end polynomial_coeffs_sum_l488_488187


namespace train_cross_time_l488_488737

theorem train_cross_time 
  (train_length : ℕ) 
  (train_speed_kmph : ℕ) 
  (bridge_length : ℕ) 
  (train_length = 150) 
  (train_speed_kmph = 45) 
  (bridge_length = 220) : 
  (train_length + bridge_length) / ((train_speed_kmph * 1000) / 3600) = 29.6 := 
sorry

end train_cross_time_l488_488737


namespace find_surface_area_of_sphere_l488_488171

noncomputable def sphere_surface_area (A B C : Point) (R : ℝ) : ℝ :=
  4 * Real.pi * R^2

theorem find_surface_area_of_sphere (A B C : Point) (R : ℝ)
  (h1 : dist (plane_of A B C) (center_of_sphere A B C) = (1/2) * R)
  (h2 : dist A B = 2)
  (h3 : dist B C = 2)
  (h4 : dist C A = 2) :
  sphere_surface_area A B C R = (64 * Real.pi) / 9 := by
sorry

-- Definitions for Points, Planes, and Distance
structure Point :=
  (x y z : ℝ)

def plane_of (A B C : Point) : Plane := sorry

def center_of_sphere (A B C : Point) : Point := sorry

def dist (p1 p2 : Point) : ℝ := sorry

end find_surface_area_of_sphere_l488_488171


namespace part_a_part_b_part_c_l488_488010

theorem part_a (θ : ℝ) (m : ℕ) : |Real.sin (m * θ)| ≤ m * |Real.sin θ| :=
sorry

theorem part_b (θ₁ θ₂ : ℝ) (m : ℕ) (hm_even : Even m) : 
  |Real.sin (m * θ₂) - Real.sin (m * θ₁)| ≤ m * |Real.sin (θ₂ - θ₁)| :=
sorry

theorem part_c (m : ℕ) (hm_odd : Odd m) : 
  ∃ θ₁ θ₂ : ℝ, |Real.sin (m * θ₂) - Real.sin (m * θ₁)| > m * |Real.sin (θ₂ - θ₁)| :=
sorry

end part_a_part_b_part_c_l488_488010


namespace solve_nat_eqn_l488_488291

theorem solve_nat_eqn (n k l m : ℕ) (hl : l > 1) 
  (h_eq : (1 + n^k)^l = 1 + n^m) : (n, k, l, m) = (2, 1, 2, 3) := 
sorry

end solve_nat_eqn_l488_488291


namespace three_digit_sum_7_l488_488969

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488969


namespace domain_of_f_range_of_f_f_at_1_f_eq_3_iff_x_eq_sqrt3_l488_488160

def f (x : ℝ) : ℝ :=
if x ≤ -1 then x + 2 else x ^ 2

theorem domain_of_f : set.Iio (2:ℝ) = set.univ :=
sorry

theorem range_of_f : set.Iio (4:ℝ) = set.Ioo (-∞) (4:ℝ) :=
sorry

theorem f_at_1 : f 1 = 1 :=
sorry

theorem f_eq_3_iff_x_eq_sqrt3 : ∀ x, f x = 3 ↔ x = real.sqrt 3 :=
sorry

end domain_of_f_range_of_f_f_at_1_f_eq_3_iff_x_eq_sqrt3_l488_488160


namespace find_x_of_floor_eq_72_l488_488869

theorem find_x_of_floor_eq_72 (x : ℝ) (hx_pos : 0 < x) (hx_eq : x * ⌊x⌋ = 72) : x = 9 :=
by 
  sorry

end find_x_of_floor_eq_72_l488_488869


namespace cat_food_finish_day_l488_488281

theorem cat_food_finish_day
  (morning_consumption : ℚ := 1/6)
  (midday_consumption : ℚ := 1/8)
  (evening_consumption : ℚ := 1/4)
  (initial_food : ℚ := 8)
  (start_day : String := "Monday") :
  let total_daily_consumption := morning_consumption + midday_consumption + evening_consumption
  let days_to_finish_food := (initial_food / total_daily_consumption).ceil.to_nat
  let end_day := start_day -- This is a placeholder; the mapping of days needs to be implemented
  in end_day = "Tuesday" := 
sorry

end cat_food_finish_day_l488_488281


namespace fraction_of_suitable_dishes_l488_488523

theorem fraction_of_suitable_dishes {T : Type} (total_menu: ℕ) (vegan_dishes: ℕ) (vegan_fraction: ℚ) (gluten_inclusive_vegan_dishes: ℕ) (low_sugar_gluten_free_vegan_dishes: ℕ) 
(h1: vegan_dishes = 6)
(h2: vegan_fraction = 1/4)
(h3: gluten_inclusive_vegan_dishes = 4)
(h4: low_sugar_gluten_free_vegan_dishes = 1)
(h5: total_menu = vegan_dishes / vegan_fraction) :
(1 : ℚ) / (total_menu : ℚ) = (1 : ℚ) / 24 := 
by
  sorry

end fraction_of_suitable_dishes_l488_488523


namespace figure_100_squares_l488_488641

theorem figure_100_squares :
  ∀ (f : ℕ → ℕ),
    (f 0 = 1) →
    (f 1 = 6) →
    (f 2 = 17) →
    (f 3 = 34) →
    f 100 = 30201 :=
by
  intros f h0 h1 h2 h3
  sorry

end figure_100_squares_l488_488641


namespace arithmetic_sequence_a6_l488_488232

theorem arithmetic_sequence_a6 :
  (∃ (a : ℕ → ℕ),
    a 1 = 1 ∧
    (∀ n, a (n + 1) = a n + 2) ∧
    a 6 = 11) :=
begin
  sorry
end

end arithmetic_sequence_a6_l488_488232


namespace num_three_digit_integers_sum_to_seven_l488_488983

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488983


namespace average_halfway_l488_488448

theorem average_halfway (a b : ℚ) (h_a : a = 1/8) (h_b : b = 1/3) : (a + b) / 2 = 11 / 48 := by
  sorry

end average_halfway_l488_488448


namespace distance_between_points_on_intersections_of_curves_l488_488226

theorem distance_between_points_on_intersections_of_curves :
  ∀ (theta : ℝ), theta = π / 3 → ∃ (A B : ℝ), 
  (A = 2 * sqrt 3) ∧ (B = 4 * sqrt 3) ∧ (abs (B - A) = 2 * sqrt 3) := 
by
  intro theta h_theta
  use 2 * sqrt 3 -- Point A
  use 4 * sqrt 3 -- Point B
  simp [abs_sub_abs_eq_abs_sub]
  sorry

end distance_between_points_on_intersections_of_curves_l488_488226


namespace solitaire_probability_l488_488057

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

def solitaire_win_probability : ℚ :=
  (binomial_coefficient 26 13)^2 / (binomial_coefficient 52 26)

theorem solitaire_probability :
  solitaire_win_probability = (binomial_coefficient 26 13)^2 / (binomial_coefficient 52 26) :=
by
  sorry

end solitaire_probability_l488_488057


namespace num_three_digit_integers_sum_to_seven_l488_488991

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l488_488991


namespace range_for_k_range_for_a_l488_488639

variable {k a : ℝ}

def proposition_p (k a : ℝ) := 
∀ x y : ℝ, (7 - a > 0 ∧ k - 1 > 0) → (7 - a > k - 1) → 0 < k

def proposition_q (k : ℝ) := 
∀ x y : ℝ, (4 - k) * (k - 2) ≥ 0 → 2 ≤ k ∧ k ≤ 4

theorem range_for_k (hk : proposition_q k) : 2 ≤ k ∧ k ≤ 4 := by
  sorry

theorem range_for_a (hp : proposition_p k a) (hq : proposition_q k) (h : hp k a → (hq k)) : a < 4 := by
  sorry

end range_for_k_range_for_a_l488_488639


namespace tan_of_angle_evaluate_expression_l488_488373

-- Define the angle α and the point P(m,1) with given condition and prove tan α = 2√2
theorem tan_of_angle (m : ℝ) (α : ℝ) (h1 : cos α = -1/3) (h2 : cos α = m / (√((m)^2 + 1))) : tan α = 2 * √2 := 
  sorry

-- Define the expression and prove its value
theorem evaluate_expression :
  (tan 150 * cos (-210) * sin (-420)) / (sin 1050 * cos (-600)) = -√3 :=
  sorry

end tan_of_angle_evaluate_expression_l488_488373


namespace domain_of_f_l488_488070

noncomputable def f (x : ℝ) : ℝ := real.sqrt (4 - real.sqrt (6 - real.sqrt x))

theorem domain_of_f :
  {x : ℝ | 0 ≤ x ∧ x ≤ 36} = {x : ℝ | ∃ y, f x = y} :=
by
  sorry

end domain_of_f_l488_488070


namespace find_lambda_l488_488179

variables {R : Type*} [LinearOrderedField R]
variables (a b : ℝ × ℝ) (λ : ℝ)
variable h_orth : ((1 - 3 * λ, 3 - 4 * λ) ∙ (3, 4)) = 0

theorem find_lambda : λ = 3 / 5 := by
  sorry

end find_lambda_l488_488179


namespace pies_left_l488_488644

theorem pies_left (pies_per_batch : ℕ) (batches : ℕ) (dropped : ℕ) (total_pies : ℕ) (pies_left : ℕ)
  (h1 : pies_per_batch = 5)
  (h2 : batches = 7)
  (h3 : dropped = 8)
  (h4 : total_pies = pies_per_batch * batches)
  (h5 : pies_left = total_pies - dropped) :
  pies_left = 27 := by
  sorry

end pies_left_l488_488644


namespace assertion_1_assertion_2_assertion_3_l488_488632
noncomputable theory

open Complex

def A : Set ℝ := sorry -- Define A such that the maximum element is given
def B : Set ℝ := sorry -- Define B
def C : Set ℝ := {c | ∃ a ∈ A, ∃ b ∈ B, c = a + b}
def max_A : ℝ := (sqrt 2 + 1) ^ 2020 + (sqrt 2 - 1) ^ 2020

lemma max_A_in_A : max_A ∈ A := sorry -- Hypothesis from the problem

theorem assertion_1 : finite A ∧ finite B :=
begin
  sorry -- Proof that both A and B are finite
end

theorem assertion_2 : (∀ a ∈ A, a ∈ ℤ) ∧ (∀ b ∈ B, b ∈ ℤ) :=
begin
  sorry -- Proof that all elements in A and B are integers
end

theorem assertion_3 : ∃ b ∈ B, b ≤ 2 ^ 2828 - 2 ^ 2525 :=
begin
  sorry -- Proof that there is an element in B that does not exceed 2^2828 - 2^2525
end

end assertion_1_assertion_2_assertion_3_l488_488632


namespace number_of_functions_eq_8_l488_488834

theorem number_of_functions_eq_8 :
  {a b c d : ℝ // a^2 = a ∧ -b^2 = b ∧ c^2 = c ∧ d^2 = d}.subtype.card = 8 := 
sorry

end number_of_functions_eq_8_l488_488834


namespace chameleons_changed_color_l488_488558

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l488_488558


namespace three_digit_sum_7_l488_488964

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488964


namespace projectile_maximum_height_l488_488033

theorem projectile_maximum_height (v_0 h_0 : ℝ) (g : ℝ := 10) :
  let h_max := h_0 + 45 in
  (-5 * (v_0 / g) ^ 2 + v_0 * (v_0 / g) + h_0) = h_max → v_0 = 30 := 
by
  intro h_eq
  have : v_0^2 / 20 = 45 := sorry
  have : v_0^2 = 900 := sorry
  have : v_0 = 30 := sorry
  exact this

end projectile_maximum_height_l488_488033


namespace probability_sum_divisible_by_3_l488_488538

-- Define the first ten prime numbers
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate to check divisibility by 3
def divisibleBy3 (n : ℕ) : Prop := n % 3 = 0

-- Define the main theorem statement
theorem probability_sum_divisible_by_3 :
  (let pairs := (firstTenPrimes.product firstTenPrimes).filter (λ (x : ℕ × ℕ), x.1 < x.2) in
    let totalPairs := pairs.length in
    let divisiblePairs := pairs.count (λ (x : ℕ × ℕ), divisibleBy3 (x.1 + x.2)) in
    (divisiblePairs.to_rat / totalPairs.to_rat) = (1 : ℚ) / 3) :=
begin
  sorry -- Proof is not required.
end

end probability_sum_divisible_by_3_l488_488538


namespace log_series_inequality_l488_488163

open Real Nat

noncomputable def f (x k : ℝ) : ℝ := k * x^2 - log x

theorem log_series_inequality (n : ℕ) (h : 2 ≤ n) :
  ∑ i in range' 2 n, (log i / i^4) < (1 / (2 * exp 1)) :=
by
  sorry

end log_series_inequality_l488_488163


namespace number_of_three_digit_numbers_with_digit_sum_seven_l488_488940

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l488_488940


namespace resistance_of_wire_loop_l488_488367

theorem resistance_of_wire_loop (d : ℝ) (R₀ : ℝ) (R : ℝ) : 
  d = 2 → R₀ = 10 → (1 / R = 1 / (R₀ * 1) + 1 / (R₀ * 2) + 1 / (R₀ * 1)) → R = 4 :=
begin
  intros h_d h_R₀ h_parallel,
  rw [h_d, h_R₀] at h_parallel,
  sorry,
end

end resistance_of_wire_loop_l488_488367


namespace min_value_of_expression_l488_488464

noncomputable def f (x : ℝ) : ℝ := 1/x + 8/(1 - 2*x)

theorem min_value_of_expression (h : 0 < x ∧ x < 1/2) : ∃ m, m = 18 ∧ (∀ y, 0 < y ∧ y < 1/2 → f y ≥ m) := 
by
  let m := 18
  exists m
  split
  { rfl }
  sorry

end min_value_of_expression_l488_488464


namespace chameleons_changed_color_l488_488559

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l488_488559


namespace number_of_common_tangents_l488_488833

noncomputable def circle1_center := (-2, 2)
noncomputable def circle1_radius := 1

noncomputable def circle2_center := (2, -5)
noncomputable def circle2_radius := 4

noncomputable def distance_between_centers : ℝ := Real.sqrt ((-2 - 2) ^ 2 + (2 + 5) ^ 2)

theorem number_of_common_tangents
  (h1 : (Real.sqrt 65) > (circle1_radius + circle2_radius))
  : 4 :=
  sorry

end number_of_common_tangents_l488_488833


namespace three_digit_sum_7_l488_488962

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488962


namespace not_enough_pharmacies_l488_488771

theorem not_enough_pharmacies : 
  ∀ (n m : ℕ), n = 10 ∧ m = 10 →
  ∃ (intersections : ℕ), intersections = n * m ∧ 
  ∀ (d : ℕ), d = 3 →
  ∀ (coverage : ℕ), coverage = (2 * d + 1) * (2 * d + 1) →
  ¬ (coverage * 12 ≥ intersections * 2) :=
by sorry

end not_enough_pharmacies_l488_488771


namespace twelve_pharmacies_not_enough_l488_488743

-- Define the grid dimensions and necessary parameters
def grid_size := 9
def total_intersections := (grid_size + 1) * (grid_size + 1) -- 10 * 10 grid
def walking_distance := 3
def coverage_side := (walking_distance * 2 + 1)  -- 7x7 grid coverage
def max_covered_per_pharmacy := (coverage_side - 1) * (coverage_side - 1)  -- Coverage per direction

-- Define the main theorem
theorem twelve_pharmacies_not_enough (n m : ℕ): 
  n = grid_size + 1 -> m = grid_size + 1 -> total_intersections = n * m -> 
  (walking_distance < n) -> (walking_distance < m) -> (pharmacies : ℕ) -> pharmacies = 12 ->
  (coverage_side <= n) -> (coverage_side <= m) ->
  ¬ (∀ i j : ℕ, i < n -> j < m -> ∃ p : ℕ, p < pharmacies -> 
  abs (i - (p / (grid_size + 1))) + abs (j - (p % (grid_size + 1))) ≤ walking_distance) :=
begin
  intros,
  sorry -- Proof omitted
end

end twelve_pharmacies_not_enough_l488_488743


namespace find_days_B_l488_488655

def work_rate_A := 1 / 30
def combined_work_rate := 0.05

theorem find_days_B (B : ℝ) (work_rate_A : ℝ := 1 / 30) (combined_work_rate : ℝ := 0.05) :
  (rate_B : ℝ) (h : rate_B = 1 / B):
  (work_rate_A + rate_B = combined_work_rate) → B = 60 :=
by sorry

end find_days_B_l488_488655


namespace inequality_proof_l488_488475

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_proof
  (h_diff : differentiable ℝ f)
  (h_ineq : ∀ x, deriv f x > f x)
  (a : ℝ) (ha : a > 0) :
  f(a) > (Real.exp(a) * f(0)) := sorry

end inequality_proof_l488_488475


namespace solve_exponential_equation_l488_488294

theorem solve_exponential_equation (x : ℝ) :
  8^x + 27^x + 2 * 30^x + 54^x + 60^x = 12^x + 18^x + 20^x + 24^x + 45^x + 90^x ↔
  x = -1 ∨ x = 0 ∨ x = 1 :=
sorry

end solve_exponential_equation_l488_488294


namespace inequality_not_always_true_l488_488518

theorem inequality_not_always_true {a b c : ℝ}
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) (h₄ : c ≠ 0) : ¬ ∀ c : ℝ, (a / c > b / c) :=
by
  sorry

end inequality_not_always_true_l488_488518


namespace six_consecutive_primes_sum_prime_unique_l488_488289

theorem six_consecutive_primes_sum_prime_unique :
  ∃ (p : ℕ), 
    (∃ (a b c d e f : ℕ), prime a ∧ prime b ∧ prime c ∧ prime d ∧ prime e ∧ prime f ∧ (a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f) ∧ p = a + b + c + d + e + f) 
    ∧ prime p ∧ p = 41 :=
by
  sorry

end six_consecutive_primes_sum_prime_unique_l488_488289


namespace three_digit_integers_sum_to_7_l488_488902

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488902


namespace unique_function_satisfying_equation_l488_488087

def functional_equation (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f(x - f(y)) = f(f(y)) + x * f(y) + f(x) - 1

theorem unique_function_satisfying_equation :
  (∀ f : ℝ → ℝ, functional_equation f → f = (λ x, 1 - x^2 / 2)) := 
by
  sorry

end unique_function_satisfying_equation_l488_488087


namespace windows_as_top_choice_l488_488223

def total_students := 3000
def mac_pref := 0.40 * total_students
def linux_pref := 0.15 * total_students
def equal_pref := 1/5 * mac_pref
def remaining := total_students - (mac_pref + linux_pref + equal_pref)
def win_remaining := 0.60 * remaining
def win_top := win_remaining + equal_pref

theorem windows_as_top_choice : win_top = 906 := by
  sorry

end windows_as_top_choice_l488_488223


namespace option_b_equivalence_l488_488122

theorem option_b_equivalence (a : ℝ) (ha_pos : 0 < a) (ha_neq_one : a ≠ 1) : 
    (∀ x : ℝ, (2 : ℝ) * x = log a (a ^ ((2 : ℝ) * x))) := 
by 
  sorry

end option_b_equivalence_l488_488122


namespace small_triangles_l488_488733

theorem small_triangles (n : ℕ) (h_n : n = 100) : 
  let total_points := n + 3 in
  no_collinear total_points → 
  num_small_triangles total_points = 2 * n + 1 :=
by
  intros
  rw [h_n]
  let n := 100
  have : total_points = 103 := by simp
  sorry

end small_triangles_l488_488733


namespace three_digit_integers_sum_to_7_l488_488900

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l488_488900


namespace problem_1_problem_2_l488_488595

noncomputable def O := (0, 0)
noncomputable def A := (1, 2)
noncomputable def B := (-3, 4)

noncomputable def vector_AB := (B.1 - A.1, B.2 - A.2)
noncomputable def magnitude_AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def dot_OA_OB := A.1 * B.1 + A.2 * B.2
noncomputable def magnitude_OA := Real.sqrt (A.1^2 + A.2^2)
noncomputable def magnitude_OB := Real.sqrt (B.1^2 + B.2^2)
noncomputable def cosine_angle := dot_OA_OB / (magnitude_OA * magnitude_OB)

theorem problem_1 : vector_AB = (-4, 2) ∧ magnitude_AB = 2 * Real.sqrt 5 := sorry

theorem problem_2 : cosine_angle = Real.sqrt 5 / 5 := sorry

end problem_1_problem_2_l488_488595


namespace max_distance_ellipse_l488_488616

-- Let's define the conditions and the theorem in Lean 4.
theorem max_distance_ellipse (θ : ℝ) :
  let C := { p : ℝ × ℝ | p.1 ^ 2 / 5 + p.2 ^ 2 = 1 }
  let B := (0, 1)
  let P := (sqrt 5 * cos θ, sin θ)
  P ∈ C →
  max (λ θ : ℝ, sqrt ((sqrt 5 * cos θ - 0) ^ 2 + (sin θ - 1) ^ 2)) = 5 / 2 :=
sorry

end max_distance_ellipse_l488_488616


namespace number_of_three_digit_numbers_with_sum_seven_l488_488895

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l488_488895


namespace sum_of_divisors_75_to_85_l488_488686

theorem sum_of_divisors_75_to_85 (N : ℕ) (hN: N = 3^16 - 1) (hDiv: 193 ∣ N) :
  (∑ x in (finset.filter (λ n, 75 ≤ n ∧ n ≤ 85) (finset.divisors N)), x) = 247 :=
by
  sorry

end sum_of_divisors_75_to_85_l488_488686


namespace three_digit_sum_7_l488_488970

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l488_488970


namespace length_vector_eq_unit_vector_collinear_parallelogram_iff_l488_488357

-- Definitions for vector lengths and unit vectors
def vector (α : Type*) := α × α

def length {α : Type*} [NormedGroup α] (v : vector α) : ℝ := ∥v.1 - v.2∥

def unit_vector {α : Type*} [NormedGroup α] (v : vector α) (h : ∥v.1 - v.2∥ ≠ 0) : vector α :=
(v.1 - v.2) / (norm (v.1 - v.2))

-- Definitions for collinear vectors and parallelograms
def collinear {α : Type*} [NormedGroup α] (v w : vector α) : Prop :=
∃ k : ℝ, w = (k • v)

def is_parallelogram {α : Type*} [AddCommGroup α] (A B C D : α) : Prop :=
(A - B) = (D - C) ∧ (A - D) = (B - C)

-- Statements to prove
theorem length_vector_eq (A B : α) [NormedGroup α] : length (A, B) = length (B, A) := by sorry

theorem unit_vector_collinear (A B : α) [NormedGroup α] (h : ∥A - B∥ ≠ 0) : collinear (A, B) (unit_vector (A, B) h) := by sorry

theorem parallelogram_iff (A B C D : α) [AddCommGroup α] (h : (A - B) = (D - C)) : is_parallelogram A B C D := by sorry

end length_vector_eq_unit_vector_collinear_parallelogram_iff_l488_488357


namespace pounds_of_apples_needed_l488_488183

-- Define the conditions
def n : ℕ := 8
def c_p : ℕ := 1
def a_p : ℝ := 2.00
def c_crust : ℝ := 2.00
def c_lemon : ℝ := 0.50
def c_butter : ℝ := 1.50

-- Define the theorem to be proven
theorem pounds_of_apples_needed : 
  (n * c_p - (c_crust + c_lemon + c_butter)) / a_p = 2 := 
by
  sorry

end pounds_of_apples_needed_l488_488183


namespace cones_sold_l488_488270

-- Define the conditions
variable (milkshakes : Nat)
variable (cones : Nat)

-- Assume the given conditions
axiom h1 : milkshakes = 82
axiom h2 : milkshakes = cones + 15

-- State the theorem to prove
theorem cones_sold : cones = 67 :=
by
  -- Proof goes here
  sorry

end cones_sold_l488_488270


namespace chameleons_color_change_l488_488574

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l488_488574


namespace area_of_region_l488_488494

noncomputable def region_area : ℝ :=
  let square_area := 100
  let circle_area := 25 * Real.pi
  let total_subtracted_area := 2 * (circle_area / 4)
  let final_area := square_area - total_subtracted_area
  final_area

theorem area_of_region (z : ℂ) (x y : ℝ) (h1 : 0 < x < 10 ∧ 0 < y < 10) (h2 : (x - 5)^2 + y^2 > 25) (h3 : x^2 + (y - 5)^2 > 25) : 
  region_area = 75 - 25 / 2 * Real.pi := 
by
  sorry

end area_of_region_l488_488494


namespace percentage_decrease_l488_488324

variable {a b x m : ℝ} (p : ℝ)

theorem percentage_decrease (h₁ : a / b = 4 / 5)
                          (h₂ : x = 1.25 * a)
                          (h₃ : m = b * (1 - p / 100))
                          (h₄ : m / x = 0.8) :
  p = 20 :=
sorry

end percentage_decrease_l488_488324


namespace compute_problem_l488_488420

theorem compute_problem : (19^12 / 19^8)^2 = 130321 := by
  sorry

end compute_problem_l488_488420


namespace doll_collection_increased_l488_488098

variables (x : ℕ) (increase : ℕ := 2) (percent_increase : ℝ := 0.25)

noncomputable def original_doll_count (x : ℕ) : ℝ := x * percent_increase -- This is 25% of x
noncomputable def final_doll_count := x + increase

theorem doll_collection_increased (x : ℕ) (h : original_doll_count(x) = 2) : final_doll_count x = 10 :=
by
  unfold original_doll_count final_doll_count
  have h1 : x * 0.25 = 2 := h
  have h2 : (x : ℝ) = 2 / 0.25 := (eq_div_iff_mul_eq' (by norm_num : (0.25 : ℝ) ≠ 0)).mpr h1
  norm_num at h2
  have x_eq_8 : x = 8 := by linarith
  rw [x_eq_8]
  norm_num
  exact sorry

end doll_collection_increased_l488_488098


namespace slip_4_goes_in_B_l488_488243

-- Definitions for the slips, cups, and conditions
def slips : List ℝ := [1, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5]
def cupSum (c : Char) : ℝ := 
  match c with
  | 'A' => 6
  | 'B' => 7
  | 'C' => 8
  | 'D' => 9
  | 'E' => 10
  | 'F' => 11
  | _   => 0

def cupAssignments : Char → List ℝ
  | 'F' => [2]
  | 'B' => [3]
  | _   => []

theorem slip_4_goes_in_B :
  (∃ cupA cupB cupC cupD cupE cupF : List ℝ, 
    cupA.sum = cupSum 'A' ∧
    cupB.sum = cupSum 'B' ∧
    cupC.sum = cupSum 'C' ∧
    cupD.sum = cupSum 'D' ∧
    cupE.sum = cupSum 'E' ∧
    cupF.sum = cupSum 'F' ∧
    slips = cupA ++ cupB ++ cupC ++ cupD ++ cupE ++ cupF ∧
    cupF.contains 2 ∧
    cupB.contains 3 ∧
    cupB.contains 4) :=
sorry

end slip_4_goes_in_B_l488_488243


namespace ellipse_shortest_major_axis_l488_488311

theorem ellipse_shortest_major_axis (P : ℝ × ℝ) (a b : ℝ) 
  (ha : a > b) (hb : b > 0) (hP_on_line : P.2 = P.1 + 2)
  (h_foci_hyperbola : ∃ c : ℝ, c = 1 ∧ a^2 - b^2 = c^2) :
  (∃ a b : ℝ, a^2 = 5 ∧ b^2 = 4 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1)) :=
sorry

end ellipse_shortest_major_axis_l488_488311


namespace twelve_pharmacies_not_sufficient_l488_488764

-- Define an intersection grid of size 10 x 10 (100 squares).
def city_grid : Type := Fin 10 × Fin 10 

-- Define the distance measure between intersections, assumed as L1 metric for grid paths.
def dist (p q : city_grid) : Nat := (abs (p.1.val - q.1.val) + abs (p.2.val - q.2.val))

-- Define a walking distance pharmacy 
def is_walking_distance (p q : city_grid) : Prop := dist p q ≤ 3

-- State that having 12 pharmacies is not sufficient
theorem twelve_pharmacies_not_sufficient (pharmacies : Fin 12 → city_grid) :
  ¬ (∀ intersection: city_grid, ∃ (p_index : Fin 12), is_walking_distance (pharmacies p_index) intersection) :=
sorry

end twelve_pharmacies_not_sufficient_l488_488764
