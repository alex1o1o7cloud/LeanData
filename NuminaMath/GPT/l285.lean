import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Floor
import Mathlib.Algebra.Order
import Mathlib.Algebra.Parity
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.SinCos
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Init.Data.Nat.Basic
import Mathlib.LinearAlgebra.FinVect
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import ProbTheory

namespace not_prime_2010_ones_prime_factor_form_2011_ones_l285_285059

-- Part a: Prove that 111...11 (with 2010 ones) is not a prime number
theorem not_prime_2010_ones :
  let N := (10^2010 - 1) / 9 in
  ¬(N.prime) :=
by
  let N := (10^2010 - 1) / 9
  sorry

-- Part b: Prove every prime factor of 111...11 (with 2011 ones) is of the form 4022j + 1
theorem prime_factor_form_2011_ones :
  let N := (10^2011 - 1) / 9 in
  ∀ p : ℕ, p.prime ∧ p ∣ N → ∃ (j : ℕ), p = 4022 * j + 1 :=
by
  let N := (10^2011 - 1) / 9
  sorry

end not_prime_2010_ones_prime_factor_form_2011_ones_l285_285059


namespace part1_part2_l285_285891

-- Part 1: The theorem statement for part (1)
theorem part1 (c : ℕ → ℕ) : ∃ (n : ℕ) (i j : ℕ), i ∈ {1, 2, 3, 4} ∧ j ∈ {1, 2, 3, 4} ∧ i ≠ j ∧ 
  (∑ d in (finset.divisors n).filter (λ d, c d = i), 1) ≥ 
  (∑ d in (finset.divisors n).filter (λ d, c d = j), 1) + 3 :=
sorry

-- Part 2: The theorem statement for part (2)
theorem part2 (c : ℕ → ℕ) (A : ℕ) : ∃ (n : ℕ) (i j : ℕ), i ∈ {1, 2, 3, 4} ∧ j ∈ {1, 2, 3, 4} ∧ i ≠ j ∧ 
  (∑ d in (finset.divisors n).filter (λ d, c d = i), 1) ≥ 
  (∑ d in (finset.divisors n).filter (λ d, c d = j), 1) + A :=
sorry

end part1_part2_l285_285891


namespace even_five_digit_numbers_count_l285_285862

-- Define the conditions
def digits : Finset ℕ := {1, 2, 3, 4, 5}
def five_digit_numbers : Finset (List ℕ) := 
  { nums | nums.length = 5 ∧ (nums.to_finset = digits)}

-- Define the property of even numbers
def even (n : ℕ) : Prop := n % 2 = 0

-- Define the set of even numbers from five_digit_numbers
def even_five_digit_numbers : Finset (List ℕ) :=
  { nums ∈ five_digit_numbers | even (nums.ilast sorry) }

-- The statement we want to prove
theorem even_five_digit_numbers_count :
  even_five_digit_numbers.card = 48 :=
sorry

end even_five_digit_numbers_count_l285_285862


namespace max_p_value_l285_285127

theorem max_p_value : ∃ x : ℝ,  let LHS := 2 * (cos (2*π - (π * x^2) / 6) * cos ((π / 3) * sqrt (9 - x^2))) - 3,
                                      RHS := p - 2 * (sin (-(π * x^2) / 6) * cos ((π / 3) * sqrt (9 - x^2)))
                                in LHS = RHS → p ≤ -2 :=
begin
  sorry
end

end max_p_value_l285_285127


namespace bug_reaches_5_5_odd_moves_l285_285795

theorem bug_reaches_5_5_odd_moves :
  ∀ (startRow startCol finalRow finalCol : ℕ),
  startRow = 1 ∧ startCol = 1 ∧ finalRow = 5 ∧ finalCol = 5 →
  forall (moves: ℕ),
    ( ∃ h_steps: ℕ, ∃ v_steps: ℕ,
      moves = h_steps + v_steps ∧
      h_steps % 2 = 0 ∧ -- h_steps is even, 2 squares per move horizontally
      v_steps % 1 = 0 ∧ -- v_steps is any, 1 square per move vertically
      h_steps + v_steps = (finalRow - startRow + finalCol - startCol)) →
    odd moves :=
by
  sorry

end bug_reaches_5_5_odd_moves_l285_285795


namespace part1_part2_l285_285588

theorem part1 (a : ℝ) (h : a * (-a)^2 + (a - 1) * (-a) - 1 > 0) : a > 1 :=
sorry

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x : ℝ | x < -1}
  else if a = -1 then ∅
  else if a > 0 then {x : ℝ | x < -1} ∪ {x : ℝ | x > 1 / a}
  else if a < -1 then {x : ℝ | -1 < x ∧ x < 1 / a}
  else {x : ℝ | 1 / a < x ∧ x < -1}

theorem part2 (a : ℝ) :
  ∀ x : ℝ, (a * x^2 + (a - 1) * x - 1 > 0) ↔ (x ∈ (solution_set a)) :=
sorry

end part1_part2_l285_285588


namespace marble_difference_l285_285477

theorem marble_difference
  (ben_initial : ℕ := 18)
  (john_initial : ℕ := 17)
  (ben_to_john : ben_initial / 2)
  (ben_remaining_after_john : ben_initial - ben_to_john)
  (ben_to_lisa : ben_remaining_after_john / 4)
  (ben_remaining : 7)
  (john_after_ben : john_initial + ben_to_john)
  (john_to_lisa : ben_to_john / 3)
  (john_remaining : 23) :
  john_remaining - ben_remaining = 16 :=
by {
  simp only [ben_initial, john_initial, ben_to_john, ben_remaining_after_john, ben_to_lisa, ben_remaining, john_after_ben, john_to_lisa, john_remaining],
  sorry
}

end marble_difference_l285_285477


namespace find_a_value_l285_285565

noncomputable theory

-- Definitions and conditions
def a_prop (a : ℝ) : Prop :=
  0 < a ∧ ∥(a : ℂ) + complex.I∥ = 2 * ∥complex.I∥

-- Main theorem statement
theorem find_a_value (a : ℝ) (ha : a_prop a) : a = real.sqrt 3 := 
sorry

end find_a_value_l285_285565


namespace least_possible_value_of_k_l285_285611

def k : ℝ := 4.9956356288922485
def num : ℝ := 0.00010101
def threshold : ℝ := 10 ^ k

theorem least_possible_value_of_k :
  ∀ k : ℤ, k = 4 → num * 10 ^ k > 1
  sorry

end least_possible_value_of_k_l285_285611


namespace max_composite_rel_prime_set_l285_285836

theorem max_composite_rel_prime_set : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 10 ≤ n ∧ n ≤ 99 ∧ ¬Nat.Prime n) ∧ 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → Nat.gcd a b = 1) ∧ 
  S.card = 4 := by
sorry

end max_composite_rel_prime_set_l285_285836


namespace area_of_triangle_l285_285576

-- Defining the ellipse and its foci
def ellipse_equation (x y : ℝ) : Prop := (x^2 / 49) + (y^2 / 24) = 1

def is_focus (x : ℝ) : Prop := x = 5 ∨ x = -5

-- The condition that lines connecting point P to the foci are perpendicular
def perp_condition (m n : ℝ) : Prop := (n / (m + 5)) * (n / (m - 5)) = -1

-- Point P lies on the ellipse
def point_on_ellipse (m n : ℝ) : Prop := ellipse_equation m n

-- Proving that the area of triangle PF1F2 is 24
theorem area_of_triangle (m n : ℝ) (hm : is_focus m) (hn : perp_condition m n) (he : point_on_ellipse m n) :
  let c := 5 in
  let area := 1/2 * 2 * c * |n| in
  area = 24 :=
by
  sorry

end area_of_triangle_l285_285576


namespace sequences_identity_l285_285329

variables {α β γ : ℤ}
variables {a b : ℕ → ℤ}

-- Define the recurrence relations conditions
def conditions (a b : ℕ → ℤ) (α β γ : ℤ) : Prop :=
  a 0 = 1 ∧ b 0 = 1 ∧
  (∀ n, a (n + 1) = α * a n + β * b n) ∧
  (∀ n, b (n + 1) = β * a n + γ * b n) ∧
  α < γ ∧ α * γ = β^2 + 1

-- Define the main statement
theorem sequences_identity (a b : ℕ → ℤ) 
  (h : conditions a b α β γ) (m n : ℕ) :
  a (m + n) + b (m + n) = a m * a n + b m * b n :=
sorry

end sequences_identity_l285_285329


namespace cot20_plus_tan10_eq_csc20_l285_285723

theorem cot20_plus_tan10_eq_csc20 : Real.cot 20 * Real.pi / 180 + Real.tan 10 * Real.pi / 180 = Real.csc 20 * Real.pi / 180 := 
by
  sorry

end cot20_plus_tan10_eq_csc20_l285_285723


namespace largest_value_of_a_l285_285147

theorem largest_value_of_a :
  ∃ (a : ℝ), (a = Real.sqrt 2023 ∧ (∃ (b c : ℝ), b > 1 ∧ c > 1 ∧ (a ^ log b c) * (b ^ log c a) = 2023)) :=
sorry

end largest_value_of_a_l285_285147


namespace sum_heights_30_students_l285_285842

/-- Sum of the heights of all 30 students given the specified conditions. -/
theorem sum_heights_30_students (S10 S20 : ℕ) 
  (hS10 : S10 = 1450) 
  (hS20 : S20 = 3030) 
  (d : ℕ) 
  (heights : Fin 31 → ℕ) 
  (sum_first_10 : (Finset.range 10).sum (λ i, heights ⟨i, nat.lt_succ_self _⟩) = S10)
  (sum_first_20 : (Finset.range 20).sum (λ i, heights ⟨i, nat.lt_succ_self _⟩) = S20)
  (constant_diff : ∀ i (hi : i < 29), heights ⟨i + 1, nat.lt_succ_self (i + 1)⟩ - heights ⟨i, hi⟩ = d) :
  (Finset.range 30).sum (λ i, heights ⟨i, nat.lt_succ_self _⟩) = 4610 :=
begin
  sorry
end

end sum_heights_30_students_l285_285842


namespace product_of_distances_l285_285885

open Real

noncomputable def power_of_point (P A : ℝ) (r : ℝ) : ℝ :=
  P ^ 2 - r ^ 2

theorem product_of_distances
(P A : Point) (Γ ω : Circle)
(h₁ : Γ.radius = 1)
(h₂ : ω.radius = 7)
(hA : ω.contains A)
(hPA : dist P A = 4)
(hX : P ∈ ω)
(hY : P ∈ ω)
(hXY₁ : X ∈ Γ)
(hXY₂ : Y ∈ Γ)
(hXY_intersect : X ≠ Y ∧ X ∈ ω ∧ Y ∈ ω) :
(PX * PY = power_of_point (dist P A) Γ.radius) :=
by
  -- Proof omitted
  sorry

end product_of_distances_l285_285885


namespace qx_polynomial_l285_285677

theorem qx_polynomial (q : ℝ → ℝ) (h_mon : polynomial.monotonicity q 4) (h_q_neg1 : q (-1) = -23) (h_q_1 : q 1 = 11) (h_q_3 : q 3 = 35) :
  q 0 + q 4 = 71 :=
by
  sorry

end qx_polynomial_l285_285677


namespace find_right_triangle_area_l285_285630

noncomputable def area_of_right_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

theorem find_right_triangle_area (h1 : (∃ a : ℕ, a^2 = 36) ∧ (∃ b : ℕ, b^2 = 64)) :
  ∃ area : ℕ, area = 24 :=
by
  -- Assume the conditions
  obtain ⟨a, ha⟩ := h1.1
  obtain ⟨b, hb⟩ := h1.2
  -- Proceed with proof
  -- sorry would be placed here

end find_right_triangle_area_l285_285630


namespace simplify_cot_tan_l285_285733

theorem simplify_cot_tan : Real.cot (Real.pi / 9) + Real.tan (Real.pi / 18) = Real.csc (Real.pi / 9) := 
by
  sorry

end simplify_cot_tan_l285_285733


namespace max_value_function_interval_l285_285782

open Interval

def max_value_of_function : ℝ :=
  let f (x : ℝ) := - (1 / (x + 1))
  let interval := Icc (1 : ℝ) (2 : ℝ)
  let max_y := - (1 / (2 + 1))
  max_y

theorem max_value_function_interval : 
  ∀ x ∈ (Icc (1 : ℝ) (2 : ℝ)), let y := - (1 / (x + 1)) in y ≤ - (1 / (2 + 1)) :=
by
  sorry

end max_value_function_interval_l285_285782


namespace actual_distance_traveled_l285_285408

theorem actual_distance_traveled (D : ℕ) 
  (h : D / 10 = (D + 36) / 16) : D = 60 := by
  sorry

end actual_distance_traveled_l285_285408


namespace lcm_of_18_and_30_l285_285025

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end lcm_of_18_and_30_l285_285025


namespace sum_c_d_l285_285301

structure Point where
  x : ℝ
  y : ℝ

def distance (P Q : Point) : ℝ := 
  Real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

def E : Point := ⟨1, 3⟩
def F : Point := ⟨3, 6⟩
def G : Point := ⟨6, 3⟩
def H : Point := ⟨9, 1⟩

noncomputable def perimeter : ℝ := 
  distance E F + distance F G + distance G H + distance H E

theorem sum_c_d : 
  let c := 5
  let d := 2
  c + d = 7 := 
by
  sorry

end sum_c_d_l285_285301


namespace simplify_trig_expression_l285_285333

theorem simplify_trig_expression :
  (sin (20 * (real.pi / 180)) + sin (40 * (real.pi / 180)) + sin (60 * (real.pi / 180)) + sin (80 * (real.pi / 180)))
  / (cos (10 * (real.pi / 180)) * cos (20 * (real.pi / 180)) * sin (30 * (real.pi / 180))) = 8 * cos (40 * (real.pi / 180)) :=
by
  sorry

end simplify_trig_expression_l285_285333


namespace trains_cross_time_l285_285414

theorem trains_cross_time
  (len1 len2 : ℕ) (speed1_kmh speed2_kmh : ℕ)
  (speed1_pos : 0 < speed1_kmh) (speed2_pos : 0 < speed2_kmh)
  (opposite_directions : true)
  (crossing_time_approx : ℝ := 12.23) :
  len1 = 140 →
  len2 = 200 →
  speed1_kmh = 60 →
  speed2_kmh = 40 →
  (len1 + len2) / ((speed1_kmh + speed2_kmh) * (5.0 / 18.0)) ≈ crossing_time_approx :=
by
  intros
  sorry

end trains_cross_time_l285_285414


namespace min_students_choir_l285_285433

-- Defining the least common multiple function explicitly in Lean
def lcm(a b : Nat) : Nat := a * (b / Nat.gcd a b)

-- The LCM of 8, 9, and 10
def lcm_8_9_10 : Nat := lcm 8 (lcm 9 10)

theorem min_students_choir : (∃ n : Nat, n % lcm_8_9_10 = 0 ∧ ∃ k : Nat, n = k^2) ∧ ∀ m, (m % lcm_8_9_10 = 0 ∧ ∃ k : Nat, m = k^2) → m = 32400 → n = 32400 :=
by {
  -- Placeholder proof
  sorry
}

end min_students_choir_l285_285433


namespace lighthouse_signal_time_l285_285073

/-
Theorem: Prove that the two signals emitted by the lighthouse will be seen for the first time together 92 seconds after midnight.
-/
theorem lighthouse_signal_time :
  ∃ x : ℕ, x = 92 ∧ (x ≡ 2 [MOD 15]) ∧ (x ≡ 8 [MOD 28]) :=
by {
  have h1 : 92 ≡ 2 [MOD 15] := by exact_mod_cast nat.modeq.refl 92,
  have h2 : 92 ≡ 8 [MOD 28] := by exact_mod_cast nat.modeq.refl 92,
  exact ⟨92, rfl, h1, h2⟩,
  sorry
}

end lighthouse_signal_time_l285_285073


namespace lcm_18_30_l285_285020

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l285_285020


namespace incorrect_for_loop_statement_l285_285045

-- Conditions definitions
def statementA : Prop :=
  "In a for loop, the loop expression is also known as the loop body."

def statementB : Prop :=
  "In a for loop, if the step size is 1, it can be omitted; if it is any other value, it cannot be omitted."

def statementC : Prop :=
  "When using a for loop, you must know the final value to proceed."

def statementD : Prop :=
  "In a for loop, the end controls the termination of one loop and the start of a new loop."

-- Lean theorem statement
theorem incorrect_for_loop_statement :
  statementD = false :=
sorry

end incorrect_for_loop_statement_l285_285045


namespace sum_y_coords_of_circle_y_axis_points_l285_285888

theorem sum_y_coords_of_circle_y_axis_points 
  (h : ∀ x y : ℝ, (x + 3)^2 + (y - 5)^2 = 64) :
  (-3, 5).snd + sqrt 55 + (-3, 5).snd - sqrt 55 = 10 :=
by
  sorry

end sum_y_coords_of_circle_y_axis_points_l285_285888


namespace value_of_k_l285_285248

theorem value_of_k (k x : ℕ) (h1 : 2^x - 2^(x - 2) = k * 2^10) (h2 : x = 12) : k = 3 := by
  sorry

end value_of_k_l285_285248


namespace find_lines_passing_through_point_and_intercepting_chord_length_l285_285146

-- Circle center and radius derived from condition
def center : ℝ × ℝ := (1, 2)
def radius : ℝ := 4

-- Given conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y - 11 = 0
def chord_length : ℝ := 4 * real.sqrt 3

-- Required line equations
def line1 : ℝ → Prop := λ y, (-1, y)
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y - 1 = 0

-- Lean statement proving both required line equations
theorem find_lines_passing_through_point_and_intercepting_chord_length :
  (∃ y, circle_eq -1 y ∧ chord_length = real.dist (1, 2) (-1, y)) ∨
  (∃ k, circle_eq (3 * k + 4 - 1) (k + 1) ∧ chord_length = 2 * real.sqrt (radius^2 - (real.abs (2 * k - 1) / real.sqrt (k^2 + 1))^2)) :=
sorry

end find_lines_passing_through_point_and_intercepting_chord_length_l285_285146


namespace part1_part2_l285_285292

def P (a : ℝ) := ∀ x : ℝ, x^2 - a * x + a + 5 / 4 > 0
def Q (a : ℝ) := 4 * a + 7 ≠ 0 ∧ a - 3 ≠ 0 ∧ (4 * a + 7) * (a - 3) < 0

theorem part1 (h : Q a) : -7 / 4 < a ∧ a < 3 := sorry

theorem part2 (h : (P a ∨ Q a) ∧ ¬(P a ∧ Q a)) :
  (-7 / 4 < a ∧ a ≤ -1) ∨ (3 ≤ a ∧ a < 5) := sorry

end part1_part2_l285_285292


namespace popped_white_probability_l285_285425

theorem popped_white_probability :
  let P_white := 2 / 3
  let P_yellow := 1 / 3
  let P_pop_given_white := 1 / 2
  let P_pop_given_yellow := 2 / 3

  let P_white_and_pop := P_white * P_pop_given_white
  let P_yellow_and_pop := P_yellow * P_pop_given_yellow
  let P_pop := P_white_and_pop + P_yellow_and_pop

  let P_white_given_pop := P_white_and_pop / P_pop

  P_white_given_pop = 3 / 5 := sorry

end popped_white_probability_l285_285425


namespace part1_part2_part3_l285_285233

-- Definitions for the given functions
def y1 (x : ℝ) : ℝ := -x + 1
def y2 (x : ℝ) : ℝ := -3 * x + 2

-- Part (1)
theorem part1 (a : ℝ) : (∃ x : ℝ, y1 x = a + y2 x ∧ x > 0) ↔ (a > -1) := sorry

-- Part (2)
theorem part2 (x y : ℝ) (h1 : y = y1 x) (h2 : y = y2 x) : 12*x^2 + 12*x*y + 3*y^2 = 27/4 := sorry

-- Part (3)
theorem part3 (A B : ℝ) (x : ℝ) (h : (4 - 2 * x) / ((3 * x - 2) * (x - 1)) = A / y1 x + B / y2 x) : (A / B + B / A) = -17 / 4 := sorry

end part1_part2_part3_l285_285233


namespace simplify_trig_identity_l285_285756

-- Defining the trigonometric functions involved
def cot (x : Real) : Real := (cos x) / (sin x)
def tan (x : Real) : Real := (sin x) / (cos x)
def csc (x : Real) : Real := 1 / (sin x)

theorem simplify_trig_identity :
  cot 20 + tan 10 = csc 20 :=
by
  sorry

end simplify_trig_identity_l285_285756


namespace can_rotate_projector_l285_285866

noncomputable def octant_illuminated (P : Point) : Prop := 
  -- Assume this defines whether a point P is within the illuminated octant
  sorry

theorem can_rotate_projector (center : Point) (cube_vertices : Set Point) : 
    (∀ v ∈ cube_vertices, v ≠ center) ∧ 
    (center_of_cube center) ∧ 
    (illuminates_octant center) → 
    (∃ rotation, ∀ v ∈ cube_vertices, ¬octant_illuminated (rotation v)) :=
sorry

end can_rotate_projector_l285_285866


namespace evaluate_at_neg_one_l285_285579

def f (x: ℝ) : ℝ :=
  if x >= 0 then x - 2 else 2^x

theorem evaluate_at_neg_one : f (-1) = 1 / 2 := by
  sorry

end evaluate_at_neg_one_l285_285579


namespace find_m_l285_285573

theorem find_m (a b c d : ℕ) (m : ℕ) (a_n b_n c_n d_n: ℕ → ℕ)
  (ha : ∀ n, a_n n = a * n + b)
  (hb : ∀ n, b_n n = c * n + d)
  (hc : ∀ n, c_n n = a_n n * b_n n)
  (hd : ∀ n, d_n n = c_n (n + 1) - c_n n)
  (ha1b1 : m = a_n 1 * b_n 1)
  (hca2b2 : a_n 2 * b_n 2 = 4)
  (hca3b3 : a_n 3 * b_n 3 = 8)
  (hca4b4 : a_n 4 * b_n 4 = 16) :
  m = 4 := 
by sorry

end find_m_l285_285573


namespace matrix_cubic_l285_285495

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![2, -1],
  ![1, 1]
]

theorem matrix_cubic :
  matrix_a ^ 3 = ![
    ![3, -6],
    ![6, -3]
  ] := by
  sorry

end matrix_cubic_l285_285495


namespace speed_of_goods_train_is_42_l285_285051

-- Definitions based on the problem conditions
def speed_of_mans_train : ℝ := 70
def length_of_goods_train : ℝ := 280 / 1000  -- convert meters to kilometers
def time_for_goods_train_to_pass : ℝ := 9 / 3600  -- convert seconds to hours

-- Main statement: Prove that the speed of the goods train is 42 km/h
theorem speed_of_goods_train_is_42 :
  let relative_speed := (length_of_goods_train / time_for_goods_train_to_pass)
  let speed_of_goods_train := relative_speed - speed_of_mans_train
  speed_of_goods_train = 42 := 
by
  sorry

end speed_of_goods_train_is_42_l285_285051


namespace perpendicularOfAltitudes_l285_285685

-- Let's define the terms and conditions that will be used in the proof.

variables {A B C O H_B H_C : Type}
variables [Triangle ABC] [Circumcenter O ABC] [Feet H_B B] [Feet H_C C]

theorem perpendicularOfAltitudes (h_circumcenter : is_circumcenter O A B C) (h_feetB : is_foot H_B B A C) (h_feetC : is_foot H_C C A B) :
  is_perpendicular H_B H_C O A :=
sorry

end perpendicularOfAltitudes_l285_285685


namespace beautiful_point_range_l285_285166

def beautiful_point (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  f x0 + f (-x0) = 0

def f (k : ℝ) : ℝ → ℝ :=
λ x, if x < 0 then x^2 + 2*x else k * x + 2

theorem beautiful_point_range (k : ℝ) :
  (∃ x0 : ℝ, beautiful_point (f k) x0) ↔ k ≤ 2 - 2 * real.sqrt 2 :=
sorry

end beautiful_point_range_l285_285166


namespace bridge_length_is_219_l285_285831

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℤ) (time_seconds : ℕ) : ℝ :=
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  let total_distance : ℝ := train_speed_ms * time_seconds
  total_distance - train_length

theorem bridge_length_is_219 :
  length_of_bridge 156 45 30 = 219 :=
by
  sorry

end bridge_length_is_219_l285_285831


namespace a8a9_eq_20_l285_285644

variable {a : ℕ → ℝ}

-- The sequence {a_n} is a geometric sequence
def geometric_sequence (a : ℕ → ℝ) := 
  ∃ r : ℝ, ∀ n : ℕ, a(n+1) = r * a(n)

-- Given conditions
variable (geo_seq : geometric_sequence a)
variable (h1 : a 2 * a 3 = 5)
variable (h2 : a 5 * a 6 = 10)

-- Goal to prove
theorem a8a9_eq_20 : a 8 * a 9 = 20 :=
by
  sorry

end a8a9_eq_20_l285_285644


namespace is_possible_l285_285253

-- Define the context and conditions
structure Grade8Class1 where
  num_students : ℕ
  avg_height : ℝ
  height_xiao_ming : ℝ
  num_taller_than_xiao_ming : ℕ
  num_shorter_than_xiao_ming : ℕ

-- Given conditions
def conditions : Grade8Class1 :=
  { num_students := 46,
    avg_height := 1.58,
    height_xiao_ming := 1.59,
    num_taller_than_xiao_ming := 25,
    num_shorter_than_xiao_ming := 20 }

-- Statement to be proved
theorem is_possible (c : Grade8Class1) :
  c.num_students = 46 →
  c.avg_height = 1.58 →
  c.height_xiao_ming = 1.59 →
  c.num_taller_than_xiao_ming = 25 →
  c.num_shorter_than_xiao_ming = 20 →
  ∃ (students : List ℝ), (∀ student_height ∈ students, true) ∧
    List.length students = c.num_students ∧
    (List.sum students) / (c.num_students : ℝ) = c.avg_height ∧
    List.count (λ h, h > c.height_xiao_ming) students = c.num_taller_than_xiao_ming ∧
    List.count (λ h, h < c.height_xiao_ming) students = c.num_shorter_than_xiao_ming :=
by
  -- Define the variables and add the proof steps here
  sorry

end is_possible_l285_285253


namespace projection_of_a_on_a_plus_b_l285_285562

variables {a b : ℝ^3}
noncomputable def norm (v : ℝ^3) := sqrt (v.1^2 + v.2^2 + v.3^2)

axiom unit_vectors (ha : norm a = 1) (hb : norm b = 1)
axiom vector_condition (h : norm (a + b) = sqrt 2 * norm (a - b))

theorem projection_of_a_on_a_plus_b :
  let projection := (a • (a + b)) / (norm (a + b)) in
  projection = sqrt 6 / 3 :=
sorry

end projection_of_a_on_a_plus_b_l285_285562


namespace problem_statement_l285_285669

variable {S : ℕ → ℚ}
variable {a : ℕ → ℚ}
variable {n : ℕ}

-- Condition: \( S_n \) is the sum of the first n terms of the sequence \(\{a_n\}\).
def sum_seq (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, S n = (Finset.range n).sum a

-- Condition: \( a_1 = 1 \).
def initial_term (a : ℕ → ℚ) : Prop :=
  a 1 = 1

-- Condition: \(\frac{S_n}{a_n}\) forms an arithmetic sequence with a common difference of \(\frac{1}{3}\).
def arithmetic_seq (S a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, d = 1/3 ∧ ∀ n, (S n / a n) = (S 1 / a 1) + d * (n - 1)

-- The general formula for \(a_n\).
def general_formula (a : ℕ → ℚ) : Prop :=
  ∀ n, a n = n * (n + 1) / 2

-- The harmonic series sum we want to prove is less than 2.
def harmonic_series (a : ℕ → ℚ) : Prop :=
  ∀ n, (Finset.range n).sum (λ k, 1 / a (k + 1)) < 2

-- Combining all goals into a single theorem to be proved.
theorem problem_statement (S a : ℕ → ℚ) :
  sum_seq a S →
  initial_term a →
  arithmetic_seq S a →
  general_formula a ∧ harmonic_series a :=
by
  intros
  split
  -- Prove the general formula
  { sorry }
  -- Prove the harmonic series inequality
  { sorry }

end problem_statement_l285_285669


namespace math_problem_l285_285175

noncomputable def problem_statement : Prop :=
  ∃ b c : ℝ, 
  (∀ x : ℝ, (x^2 - b * x + c < 0) ↔ (-3 < x ∧ x < 2)) ∧ 
  (b + c = -7)

theorem math_problem : problem_statement := 
by
  sorry

end math_problem_l285_285175


namespace range_of_a_decreasing_l285_285362

theorem range_of_a_decreasing (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y ≤ 4 → f(x) >= f(y)) → a ∈ set.Ici 5 :=
by
  let f := λ x : ℝ, x^2 + 2 * (1 - a) * x + 2
  sorry

end range_of_a_decreasing_l285_285362


namespace find_sample_size_l285_285455

theorem find_sample_size (f r : ℝ) (h1 : f = 20) (h2 : r = 0.125) (h3 : r = f / n) : n = 160 := 
by {
  sorry
}

end find_sample_size_l285_285455


namespace initial_notebooks_l285_285249

variable (a n : ℕ)
variable (h1 : n = 13 * a + 8)
variable (h2 : n = 15 * a)

theorem initial_notebooks : n = 60 := by
  -- additional details within the proof
  sorry

end initial_notebooks_l285_285249


namespace find_distance_P_to_O_l285_285106

def circle_radius := 1

variables {A B C D P O : Type} 

-- Assume C and D are points on the circumference of the circle with center O
-- The tangents to the circle at points C and D meet at point P

noncomputable def distance_P_to_O 
  (h1 : ∠A Q B = 2 * ∠C O D)
  (h2 : circle_radius = 1)
  : ℝ := sorry

theorem find_distance_P_to_O (h1 : ∠A Q B = 2 * ∠C O D)
                            (h2 : circle_radius = 1) : 
                            distance_P_to_O h1 h2 = (2 / sqrt 3) := by
  sorry

end find_distance_P_to_O_l285_285106


namespace minimum_omega_l285_285172

theorem minimum_omega 
  (ω : ℝ)
  (hω : ω > 0)
  (h_shift : ∃ T > 0, T = 2 * π / ω ∧ T = 2 * π / 3) : 
  ω = 3 := 
sorry

end minimum_omega_l285_285172


namespace max_value_seven_a_minus_nine_b_l285_285762

theorem max_value_seven_a_minus_nine_b (r1 r2 r3 a b : ℝ)
  (h1 : 0 < r1) (h2 : r1 < 1)
  (h3 : 0 < r2) (h4 : r2 < 1)
  (h5 : 0 < r3) (h6 : r3 < 1)
  (h7 : r1 + r2 + r3 = 1)
  (h8 : r1 * r2 + r2 * r3 + r3 * r1 = a)
  (h9 : r1 * r2 * r3 = b) :
  7 * a - 9 * b ≤ 2 :=
begin
  sorry,
end

end max_value_seven_a_minus_nine_b_l285_285762


namespace part1_part2_l285_285599

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (real.cos (3 * x / 2), real.sin (3 * x / 2))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (-real.sin (x / 2), -real.cos (x / 2))

noncomputable def f (x : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 + ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2

theorem part1 (x : ℝ) (h1 : |((a x).1 + (b x).1, (a x).2 + (b x).2)| = sqrt 3) : 
  x = (7 * real.pi / 12) ∨ x = (11 * real.pi / 12) :=
sorry

theorem part2 (c : ℝ) (h2 : ∀ x ∈ Icc (real.pi / 2) real.pi, c > f x) : 
  c > 5 :=
sorry

end part1_part2_l285_285599


namespace eliminating_y_l285_285167

theorem eliminating_y (x y : ℝ) (h1 : y = x + 3) (h2 : 2 * x - y = 5) : 2 * x - x - 3 = 5 :=
by {
  sorry
}

end eliminating_y_l285_285167


namespace problem_1_problem_2a_problem_2b_problem_2c_problem_3_l285_285218

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 1
noncomputable def f_prime (x a : ℝ) : ℝ := 2 * x + 2 * a
noncomputable def g (x a : ℝ) : ℝ :=
  if f x a ≥ f_prime x a then f_prime x a else f x a

theorem problem_1 (x a : ℝ) (h1 : x ∈ set.closed_interval (-2 : ℝ) (-1)) (h2 : a ≥ 3 / 2) :
  f x a ≤ f_prime x a := sorry

theorem problem_2a (x a : ℝ) (h : a < -1) :
  f x a = |f_prime x a| → (x = -1 ∨ x = 1 - 2 * a) := sorry

theorem problem_2b (x a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  f x a = |f_prime x a| →
  (x = 1 ∨ x = -1 ∨ x = 1 - 2 * a ∨ x = -(1 + 2 * a)) := sorry

theorem problem_2c (x a : ℝ) (h : a > 1) :
  f x a = |f_prime x a| → (x = 1 ∨ x = -(1 + 2 * a)) := sorry

theorem problem_3 (x a : ℝ) (h : x ∈ set.closed_interval (2 : ℝ) (4)) :
  g x a = 
  if a ≥ -1 / 2 then 2 * a + 4
  else if a < -3 / 2 then 
    if -2 ≤ a then 4 * a + 5
    else if -4 ≤ a then 1 - a^2
    else 8 * a + 17
  else 4 * a + 5 := sorry

end problem_1_problem_2a_problem_2b_problem_2c_problem_3_l285_285218


namespace integer_root_count_of_two_to_the_power_l285_285170

noncomputable def is_integer_root (n : ℕ) : ℕ → Prop
| k := (∃ (m : ℕ), n = m ^ k)

theorem integer_root_count_of_two_to_the_power (n : ℕ) (h : n = 65536) : 
  (finset.card (finset.filter (λ k, is_integer_root n k) (finset.range n))) = 5 := 
by
  sorry

end integer_root_count_of_two_to_the_power_l285_285170


namespace max_min_f_values_l285_285549

noncomputable def f (a b c d : ℝ) : ℝ := (Real.sqrt (5 * a + 9) + Real.sqrt (5 * b + 9) + Real.sqrt (5 * c + 9) + Real.sqrt (5 * d + 9))

theorem max_min_f_values (a b c d : ℝ) (h₀ : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h₁ : a + b + c + d = 32) :
  (f a b c d ≤ 28) ∧ (f a b c d ≥ 22) := by
  sorry

end max_min_f_values_l285_285549


namespace number_of_people_needed_to_lift_car_l285_285279

-- Define the conditions as Lean definitions
def twice_as_many_people_to_lift_truck (C T : ℕ) : Prop :=
  T = 2 * C

def people_needed_for_cars_and_trucks (C T total_people : ℕ) : Prop :=
  60 = 6 * C + 3 * T

-- Define the theorem statement using the conditions
theorem number_of_people_needed_to_lift_car :
  ∃ C, (∃ T, twice_as_many_people_to_lift_truck C T) ∧ people_needed_for_cars_and_trucks C T 60 ∧ C = 5 :=
sorry

end number_of_people_needed_to_lift_car_l285_285279


namespace men_in_second_group_l285_285841

theorem men_in_second_group (m w : ℝ) (x : ℝ) 
  (h1 : 3 * m + 8 * w = x * m + 2 * w) 
  (h2 : 2 * m + 2 * w = (3 / 7) * (3 * m + 8 * w)) : x = 6 :=
by
  sorry

end men_in_second_group_l285_285841


namespace complex_numbers_count_l285_285371

theorem complex_numbers_count : 
  let real_part := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let imaginary_part := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  #|real_part| * #|imaginary_part| = 90 :=
by
  sorry

end complex_numbers_count_l285_285371


namespace mosquito_distance_ratio_l285_285349

-- Definition of the clock problem conditions
structure ClockInsects where
  distance_from_center : ℕ
  initial_time : ℕ := 1

-- Prove the ratio of distances traveled by mosquito and fly over 12 hours
theorem mosquito_distance_ratio (c : ClockInsects) :
  let mosquito_distance := (83 : ℚ)/12
  let fly_distance := (73 : ℚ)/12
  mosquito_distance / fly_distance = 83 / 73 :=
by 
  sorry

end mosquito_distance_ratio_l285_285349


namespace matrix_cubic_l285_285497

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![2, -1],
  ![1, 1]
]

theorem matrix_cubic :
  matrix_a ^ 3 = ![
    ![3, -6],
    ![6, -3]
  ] := by
  sorry

end matrix_cubic_l285_285497


namespace smallest_f_coprime_l285_285953

def f (n : ℕ) : ℕ :=
  if n < 4 then 0 else
    (n + 1) / 2 + (n + 1) / 3 - (n + 1) / 6 + 1

theorem smallest_f_coprime (n : ℕ) (hn : n ≥ 4) (m : ℕ) (hm : m > 0) :
  ∀ (S : set ℕ) (hS : S ⊆ {m, m+1, ..., m+n-1}) (hSf : S.card = f n),
  ∃ (A ⊆ S), A.card = 3 ∧ pairwise coprime A :=
sorry

end smallest_f_coprime_l285_285953


namespace ellipse_equation_l285_285189

variables (a b c: ℝ) (α: ℝ)
variables (F1 F2 A B: ℝ × ℝ)
variables (P Q M : ℝ × ℝ)
variable (e : ℝ)
variable (t : ℝ)

-- Given conditions
def conditions : Prop :=
  a > b ∧ b > 0 ∧ -- a > b > 0
  c = 3 ∧ -- c = 3 as derived from the condition with the distance given the inclination angle α
  (5 * (A.1 - F1.1) = 8 * (B.1 - F1.1)) ∧ -- 5 \overrightarrow{F_{1} A} = 8 \overrightarrow{B F_{1}}
  (2 * c * real.sin α = 72 / 13) ∧ -- derived from given the distance calculation
  (real.cos α = 5 / 13) ∧ -- cos α = 5 / 13
  (e = c / a) ∧ -- e = c / a
  (a = 5) ∧ -- result from solution
  (b = 4)   -- result from solution

-- Prove the ellipse equation given the conditions
theorem ellipse_equation : conditions α a b c F1 F2 A B e t → 
  ∃ (x y: ℝ), (x ^ 2) / 25 + (y ^ 2) / 16 = 1 :=
by
  sorry

end ellipse_equation_l285_285189


namespace solution_set_of_inequality_l285_285157

theorem solution_set_of_inequality : 
  { x : ℝ | (1 : ℝ) * (2 * x + 1) < (x + 1) } = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end solution_set_of_inequality_l285_285157


namespace determine_n_l285_285120

theorem determine_n (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∃ k d, k = (nat.factors n).head ∧ n = k^2 + d^2 ∧ d ∣ n) :
  n = 8 ∨ n = 20 :=
by
  sorry

end determine_n_l285_285120


namespace lcm_of_18_and_30_l285_285024

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end lcm_of_18_and_30_l285_285024


namespace bobs_rice_id_possibilities_l285_285876

/-- Proving that the number of different 6-digit ID numbers satisfying the given conditions is 324. -/
theorem bobs_rice_id_possibilities :
  ∃ S : Finset (Vector ℕ 6), 
  (∀ x ∈ S, 
     (∀ i, 1 ≤ x[i] ∧ x[i] ≤ 9) ∧
     (x[1] % 2 = 0) ∧
     ((x[0] + x[1] + x[2]) % 3 = 0) ∧
     ((10 * x[2] + x[3]) % 4 = 0) ∧
     (x[4] = 5) ∧
     (Array.toList x).foldl (+) 0 % 6 = 0
  ) ∧ S.card = 324 := sorry

end bobs_rice_id_possibilities_l285_285876


namespace greatest_prime_factor_294_l285_285810

theorem greatest_prime_factor_294 : ∃ p, Nat.Prime p ∧ p ∣ 294 ∧ ∀ q, Nat.Prime q ∧ q ∣ 294 → q ≤ p := 
by
  let prime_factors := [2, 3, 7]
  have h1 : 294 = 2 * 3 * 7 * 7 := by
    -- Proof of factorization should be inserted here
    sorry

  have h2 : ∀ p, p ∣ 294 → p = 2 ∨ p = 3 ∨ p = 7 := by
    -- Proof of prime factor correctness should be inserted here
    sorry

  use 7
  -- Prove 7 is the greatest prime factor here
  sorry

end greatest_prime_factor_294_l285_285810


namespace lcm_18_30_is_90_l285_285036

theorem lcm_18_30_is_90 : Nat.lcm 18 30 = 90 := 
by 
  unfold Nat.lcm
  sorry

end lcm_18_30_is_90_l285_285036


namespace find_f_neg1_l285_285777

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_neg1 (h1 : ∀ x : ℝ, x ≠ 1 → (x-1) * f((x+1)/(x-1)) = x + f(x)) :
  f (-1) = -1 :=
sorry

end find_f_neg1_l285_285777


namespace circle_equation_from_diameter_l285_285356

theorem circle_equation_from_diameter :
 {x y : ℝ} (x+2)^2 + (y - (3/2))^2 = 25/4 :=
begin
  -- Given the line 3x - 4y + 12 = 0
  -- Find the points of intersection with the x and y axes
  let x_intercept := (-4, 0),
  let y_intercept := (0, 3),
  
  -- The midpoint is hence (-2, 3/2)
  let midpoint := (-2, 3/2),
  
  -- The radius is the distance from (-2, 3/2) to (-4, 0) or (0, 3)
  -- As the line segment from -4, 0 to 0, 3 is the diameter
  let radius := (5/2),
  
  -- The standard equation of the circle is then:
  sorry
end

end circle_equation_from_diameter_l285_285356


namespace part1_distance_AB_part2_min_distance_C2_to_l_l285_285589

-- Definitions based on the conditions
def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + (1/2)*t, (sqrt 3 / 2)*t)

def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (cos θ, sin θ)

def curve_C2 (θ : ℝ) : ℝ × ℝ :=
  ((1/2)*cos θ, (sqrt 3 / 2)*sin θ)

-- Part (I): Proof that the distance |AB| = 1
theorem part1_distance_AB :
  let A := (1 : ℝ, 0 : ℝ),
    B := (1/2 : ℝ, -sqrt 3 / 2 : ℝ) in
  dist A B = 1 :=
sorry

-- Part (II): Proof that the minimum distance from any point on curve C2 to line l is (sqrt 6 / 4)(sqrt 2 - 1)
theorem part2_min_distance_C2_to_l :
  ∀ P : ℝ × ℝ, P ∈ (Set.range curve_C2) →
  ∃ θ : ℝ, dist P (line_l θ) = (sqrt 6 / 4)*(sqrt 2 - 1) :=
sorry

end part1_distance_AB_part2_min_distance_C2_to_l_l285_285589


namespace vector_correctness_l285_285596

section Problem
variable (A B C : ℝ × ℝ × ℝ)
variable (AB AC BC : ℝ × ℝ × ℝ)

def vector (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (b.1 - a.1, b.2 - a.2, b.3 - a.3)

theorem vector_correctness :
  A = (0, 1, 0) ∧ B = (2, 2, 0) ∧ C = (-1, 1, 1) →
  vector A B = (2, 1, 0) ∧
  vector A C = (-1, 0, 1) ∧
  vector B C = (-3, -1, 1) ∧
  (∃ u : ℝ × ℝ × ℝ, u = (2 * real.sqrt 5 / 5, real.sqrt 5 / 5, 0)) ∧
  (∃ n : ℝ × ℝ × ℝ, n = (1, -2, 1))
:= by
  intros h
  cases h with hA hB
  cases hB with hB hC
  simp [vector, hA, hB, hC]
  refine ⟨rfl, rfl, rfl, ⟨(2 * real.sqrt 5 / 5, real.sqrt 5 / 5, 0), rfl⟩, ⟨(1, -2, 1), rfl⟩⟩
end

end vector_correctness_l285_285596


namespace simplify_cotAndTan_l285_285716

theorem simplify_cotAndTan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := 
by
  sorry

end simplify_cotAndTan_l285_285716


namespace common_difference_of_arithmetic_seq_l285_285640

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * d

theorem common_difference_of_arithmetic_seq :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  arithmetic_sequence a d →
  (a 4 + a 8 = 10) →
  (a 10 = 6) →
  d = 1 / 4 :=
by
  intros a d h_seq h1 h2
  sorry

end common_difference_of_arithmetic_seq_l285_285640


namespace total_sections_l285_285413

theorem total_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 216) : 
  let gcd := Nat.gcd boys girls,
      sections_boys := boys / gcd,
      sections_girls := girls / gcd
  in sections_boys + sections_girls = 26 :=
by
  sorry

end total_sections_l285_285413


namespace problem_l285_285231

open Set

theorem problem (I M N : Set ℕ) (hI : I = {1, 2, 3, 4, 5, 6, 7, 8}) (hM : M = {3, 4, 5}) (hN : N = {1, 3, 6}) :
  \{2, 7, 8} = (I \ M) ∩ (I \ N) :=
by {
  sorry
}

end problem_l285_285231


namespace total_number_of_people_l285_285100

def floor_info := 
  Σ (floors : ℕ) (apartments : ℕ) (occupancy_rate : ℝ) (people_per_apt : ℕ), 
    { floors := floors, apartments := apartments, occupancy_rate := occupancy_rate, people_per_apt := people_per_apt }

def floors_1_to_4 : floor_info := ⟨4, 10, 1.0, 4⟩
def floors_5_to_8 : floor_info := ⟨4, 8, 0.8, 5⟩
def floors_9_to_12 : floor_info := ⟨4, 6, 0.6, 6⟩

def total_people (info : floor_info) : ℕ :=
  let floors := info.1
  let apartments := info.2
  let occupancy_rate := info.3
  let people_per_apt := info.4
  floors * apartments * (occupancy_rate.floor : ℕ) * people_per_apt

theorem total_number_of_people : 
  total_people floors_1_to_4 + total_people floors_5_to_8 + total_people floors_9_to_12 = 369 := 
by
  -- Proof omitted
  sorry

end total_number_of_people_l285_285100


namespace product_of_nonreal_roots_l285_285940

open Complex Polynomial

noncomputable def poly : Polynomial ℂ := X^4 - 4*X^3 + 6*X^2 - 4*X - 2010

theorem product_of_nonreal_roots :
  (∏ x in (poly.map Complex.ofReal).roots.filter (λ x, x.im ≠ 0), x) = 1 + Real.sqrt 2011 :=
by
  sorry

end product_of_nonreal_roots_l285_285940


namespace lcm_18_30_eq_90_l285_285014

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end lcm_18_30_eq_90_l285_285014


namespace new_angle_twice_original_l285_285616

variable {r l : ℝ} (h_r : r > 0) (h_l : l > 0)

def original_central_angle (r l : ℝ) : ℝ := l / r

def new_radius (r : ℝ) : ℝ := 1 / 2 * r

def new_central_angle (l : ℝ) (new_radius : ℝ) : ℝ := l / new_radius

theorem new_angle_twice_original (r l : ℝ) (h_r : r > 0) (h_l : l > 0) :
  new_central_angle l (new_radius r) = 2 * original_central_angle r l :=
by
  -- Proof omitted
  sorry

end new_angle_twice_original_l285_285616


namespace range_of_x_l285_285567

theorem range_of_x (x : ℝ) (a : ℝ)
  (h1 : a ∈ set.Icc (-2 : ℝ) (2 : ℝ))
  (h2 : x^2 + a * x ≥ 3 - a) :
  x ≤ -1 - real.sqrt 2 ∨ x ≥ 1 + real.sqrt 6 :=
sorry

end range_of_x_l285_285567


namespace rectangle_area_correct_l285_285780

theorem rectangle_area_correct (l r s : ℝ) (b : ℝ := 10) (h1 : l = (1 / 4) * r) (h2 : r = s) (h3 : s^2 = 1225) :
  l * b = 87.5 :=
by
  sorry

end rectangle_area_correct_l285_285780


namespace sphere_volume_given_conditions_l285_285202

noncomputable def h_eq_r (r : ℝ) : ℝ := r

noncomputable def cone_volume (r : ℝ) : ℝ := (1 / 3) * π * r^3

noncomputable def cone_lateral_surface_area_eq_sphere_surface_area (r R: ℝ) : bool := 
  π * r * (real.sqrt (r ^ 2 + (h_eq_r r) ^ 2)) = 4 * π * R ^ 2

noncomputable def volume_of_sphere (R : ℝ) : ℝ := (4 / 3) * π * R ^ 3

noncomputable def sphere_radius (r : ℝ) : ℝ := (r * real.sqrt (2) ^ (1 / 4)) / 2

theorem sphere_volume_given_conditions (r R : ℝ) 
  (h_eq_r : h_eq_r r = r)
  (cone_volume_condition : cone_volume r = (8 * π) / 3)
  (surface_area_condition : cone_lateral_surface_area_eq_sphere_surface_area r R = true)
    : volume_of_sphere (sphere_radius r) = (4 / 3) * π * real.sqrt(8)^ (1 / 4) := sorry

end sphere_volume_given_conditions_l285_285202


namespace expand_product_l285_285914

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 :=
by
  sorry

end expand_product_l285_285914


namespace sum_of_integers_l285_285790

theorem sum_of_integers (m n p q : ℤ) 
(h1 : m ≠ n) (h2 : m ≠ p) 
(h3 : m ≠ q) (h4 : n ≠ p) 
(h5 : n ≠ q) (h6 : p ≠ q) 
(h7 : (5 - m) * (5 - n) * (5 - p) * (5 - q) = 9) : 
m + n + p + q = 20 :=
by
  sorry

end sum_of_integers_l285_285790


namespace simplify_cot_tan_l285_285713

theorem simplify_cot_tan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := by
  sorry

end simplify_cot_tan_l285_285713


namespace ratio_a_to_c_l285_285617

variable (a b c : ℕ)

theorem ratio_a_to_c (h1 : a / b = 8 / 3) (h2 : b / c = 1 / 5) : a / c = 8 / 15 := 
sorry

end ratio_a_to_c_l285_285617


namespace sum_leq_n_div_3_l285_285320

theorem sum_leq_n_div_3 (n : ℕ) (x : Fin n → ℝ) 
  (h1 : ∀ i, -1 ≤ x i ∧ x i ≤ 1) 
  (h2 : ∑ i, (x i)^3 = 0) : 
  ∑ i, x i ≤ n / 3 :=
sorry

end sum_leq_n_div_3_l285_285320


namespace conjugate_of_z_l285_285210

def z : ℂ := 1 - Complex.i
def conj_z : ℂ := conj z

theorem conjugate_of_z : conj_z = 1 + Complex.i := by
  unfold conj_z z
  sorry

end conjugate_of_z_l285_285210


namespace corn_purchase_l285_285896

theorem corn_purchase : ∃ c b : ℝ, c + b = 30 ∧ 89 * c + 55 * b = 2170 ∧ c = 15.3 := 
by
  sorry

end corn_purchase_l285_285896


namespace verify_average_speed_and_total_cost_l285_285067

noncomputable def average_speed_and_total_cost : ℕ :=
  let distance_1 := 30 -- in kilometers
  let speed_1 := 45 -- in kph
  let distance_2 := 35 -- in kilometers
  let speed_2 := 55 -- in kph
  let speed_3 := 70 -- in kph
  let time_3 := 50 / 60.0 -- in hours
  let speed_4 := 55 -- in kph
  let time_4 := 20 / 60.0 -- in hours

  let cost_under_50 := 0.20 -- cost per kilometer for speeds up to 50 kph
  let cost_over_50 := 0.25 -- cost per kilometer for speeds over 50 kph

  let distance_3 := speed_3 * time_3 -- distance covered in segment 3
  let distance_4 := speed_4 * time_4 -- distance covered in segment 4

  let total_distance := distance_1 + distance_2 + distance_3 + distance_4
  let total_time := (distance_1 / speed_1) + (distance_2 / speed_2) + time_3 + time_4
  let average_speed := total_distance / total_time

  let cost_1 := distance_1 * cost_under_50
  let cost_2 := distance_2 * cost_over_50
  let cost_3 := distance_3 * cost_over_50
  let cost_4 := distance_4 * cost_over_50
  let total_cost := cost_1 + cost_2 + cost_3 + cost_4

  (average_speed, total_cost)

theorem verify_average_speed_and_total_cost : average_speed_and_total_cost = (57, 33.91) :=
  sorry

end verify_average_speed_and_total_cost_l285_285067


namespace tan_sine_condition_necesary_but_not_sufficient_l285_285238

theorem tan_sine_condition_necesary_but_not_sufficient (x : ℝ) (h₀ : 0 < x) (h₁ : x < π / 2) : 
  (x * tan x > 1 → x * sin x > 1) ∧ (¬(x * sin x > 1 → x * tan x > 1)) :=
by
  sorry

end tan_sine_condition_necesary_but_not_sufficient_l285_285238


namespace fill_tank_l285_285778

theorem fill_tank (full_capacity : ℝ) (fraction_full : ℝ) (current_volume_needed : ℝ) : 
  fraction_full = 1 / 3 → 
  full_capacity = 24 → 
  current_volume_needed = 16 := by
  intro h_fraction_full h_full_capacity
  have h_current_volume: full_capacity * fraction_full = 8 := by
    rw [h_fraction_full, h_full_capacity]
    norm_num
  have h_added_volume: full_capacity - (full_capacity * fraction_full) = current_volume_needed := by
    rw [h_current_volume]
    norm_num
  exact h_added_volume.symm

end fill_tank_l285_285778


namespace expand_polynomial_l285_285921

theorem expand_polynomial (x : ℝ) :
  (x + 4) * (x - 9) = x^2 - 5 * x - 36 := 
sorry

end expand_polynomial_l285_285921


namespace passengers_landed_in_Newberg_l285_285284

theorem passengers_landed_in_Newberg (on_time late total : ℕ) 
  (h1 : on_time = 14507) 
  (h2 : late = 213) 
  (h3 : total = on_time + late) :
  total = 14620 :=
by
  rw [h1, h2] at h3
  exact h3
sorry

end passengers_landed_in_Newberg_l285_285284


namespace mary_score_l285_285305

theorem mary_score :
  ∃ s c w : ℕ,
    s > 80 ∧
    (∀ t, 80 < t < s →
       ∀ c' w' : ℕ, s = 30 + 4 * c' - w' →
       t ≠ 30 + 4 * c' - w' ) ∧
    s = 30 + 4 * c - w ∧
    c ≥ 0 ∧
    w ≥ 0 ∧
    c + w ≤ 30 ∧
    ∃ c_final : ℕ, s = 30 + 4 * c_final - (30 - c_final)
 :=
begin
  use 119,
  use 23,
  use 3,
  split,
  { linarith },
  split,
  { intros t ht c' w' H1 H2,
    exfalso,
    apply H1,
    sorry },
  split,
  { refl },
  split,
  { linarith },
  split,
  { linarith },
  split,
  { linarith },
  { use 23,
    linarith },
end

end mary_score_l285_285305


namespace nat_numbers_representation_l285_285119

theorem nat_numbers_representation (n : ℕ) (h : n ≥ 2) (k : ℕ) (d : ℕ) 
  (h1 : k = Nat.find (Nat.exists_factor_two_mul (exists:= n.exists_dvd_ne_one)))
   (h2 : d ∣ n) (h3 : n = k^2 + d^2) : n = 8 ∨ n = 20 :=
by
  sorry

end nat_numbers_representation_l285_285119


namespace center_of_incircle_lies_on_MN_l285_285291

variables {A B C D M N O : Type}

-- Definitions for the geometric configurations
variables [IsInscribedTrapezoid A B C D]
variables (h_parallel : A B ∥ C D)
variables (h_proportioned : A B > C D)
variables [IsTangentToIncircle M A B]
variables [IsTangentToIncircle N A C]

-- The main theorem statement
theorem center_of_incircle_lies_on_MN :
  lies_on_center_of_incircle A B C D M N :=
begin
  -- skipped proof
  sorry
end

end center_of_incircle_lies_on_MN_l285_285291


namespace equation_of_line_through_center_arithmetic_sequence_l285_285890

theorem equation_of_line_through_center_arithmetic_sequence 
  (P M N Q : ℝ × ℝ)
  (F : Set (ℝ × ℝ))
  (C : Set (ℝ × ℝ))
  (centerF : ℝ × ℝ := (1, 0))
  (conditionF : ∀ (x y : ℝ), (x, y) ∈ F ↔ x^2 + y^2 - 2*x = 0)
  (conditionC : ∀ (x y : ℝ), (x, y) ∈ C ↔ y^2 = 4*x)
  (line_through_center : ∀ (x y k : ℝ), k ≠ 0 → y = k*(x - 1))
  (intersections : List (ℝ × ℝ) := [(P, M), (M, N), (N, Q)])
  (arithmetic_seq_cond : dist P M + dist N Q = 2 * dist M N)
  (distances : ∀ (A B : ℝ × ℝ), dist A B = ℝ := λ A B, real.sqrt ((fst A - fst B)^2 + (snd A - snd B)^2)) :
  (line_through_center x y (sqrt 2) ∨ line_through_center x y (-sqrt 2) → 
  arithmetic_seq_cond P M N Q) :=
sorry

end equation_of_line_through_center_arithmetic_sequence_l285_285890


namespace outer_term_in_proportion_l285_285259

theorem outer_term_in_proportion (a b x : ℝ) (h_ab : a * b = 1) (h_x : x = 0.2) : b = 5 :=
by
  sorry

end outer_term_in_proportion_l285_285259


namespace remainder_19_pow_19_plus_19_mod_20_l285_285417

theorem remainder_19_pow_19_plus_19_mod_20 : (19^19 + 19) % 20 = 18 := 
by
  sorry

end remainder_19_pow_19_plus_19_mod_20_l285_285417


namespace notebooks_per_child_if_half_l285_285169

theorem notebooks_per_child_if_half (C N : ℕ) 
    (h1 : N = C / 8) 
    (h2 : C * N = 512) : 
    512 / (C / 2) = 16 :=
by
    sorry

end notebooks_per_child_if_half_l285_285169


namespace annes_speed_ratio_l285_285478

def clean_rate (t : ℝ) := 1 / t

def clean_house_together_in (t : ℝ) (rA rB : ℝ) : Prop :=
  rA + rB = clean_rate t

def original_rate_A : ℝ := clean_rate 12

def rate_after_change : ℝ := clean_rate 3

theorem annes_speed_ratio (rB : ℝ) (A' : ℝ) (h1 : clean_house_together_in 4 original_rate_A rB) 
  (h2 : clean_house_together_in 3 A' rB) :
  A' / original_rate_A = 2 :=
sorry

end annes_speed_ratio_l285_285478


namespace suresh_completion_time_l285_285344

theorem suresh_completion_time (S : ℕ) 
  (ashu_time : ℕ := 30) 
  (suresh_work_time : ℕ := 9) 
  (ashu_remaining_time : ℕ := 12) 
  (ashu_fraction : ℚ := ashu_remaining_time / ashu_time) :
  (suresh_work_time / S + ashu_fraction = 1) → S = 15 :=
by
  intro h
  -- Proof here
  sorry

end suresh_completion_time_l285_285344


namespace vector_magnitude_l285_285600

variables {V : Type*} [inner_product_space ℝ V]

-- Define the vectors a and b
variables (a b : V)

-- Given conditions
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 1
axiom dot_ab : ⟪a, b⟫ = -1 / 2

-- The theorem to be proven
theorem vector_magnitude : ∥a + 2 • b∥ = real.sqrt 3 :=
sorry

end vector_magnitude_l285_285600


namespace tournament_nk_condition_l285_285098

def t_k (k : ℕ) : ℕ :=
  Nat.find (λ m : ℕ, 2^(m - 1) < (k + 1) ∧ (k + 1) ≤ 2^m)

def power_of_two (n m : ℕ) : Prop :=
  ∃ t : ℕ, n = t * 2^m

def nk_tournament_exists (n k : ℕ) : Prop :=
  ∃ (meetings : fin k → fin n → fin n → Prop),
    (∀ (i : fin k) (a b : fin n), a ≠ b → meetings i a b) ∧
    (∀ (i j : fin k) (a b c d : fin n),
      a ≠ b → c ≠ d →
      meetings i a b → meetings i c d →
      meetings j a c → meetings j b d)

theorem tournament_nk_condition (n k : ℕ) :
  nk_tournament_exists n k ↔ power_of_two n (t_k k) := sorry

end tournament_nk_condition_l285_285098


namespace units_digit_of_sum_of_squares_plus_7_l285_285815

theorem units_digit_of_sum_of_squares_plus_7 :
  ∃ d : ℕ, (d < 10) ∧ (d = ( ( ∑ k in finset.range (1011 + 1), (if k % 2 = 1 then k else 0) ^ 2 ) + 7) % 10) :=
by
  sorry

end units_digit_of_sum_of_squares_plus_7_l285_285815


namespace percentage_of_copper_in_first_alloy_l285_285465

-- Setting up the problem in Lean
variables (x : ℝ)  -- The percentage of copper in the first alloy in decimal form

-- Conditions
def first_alloy_copper_mass := 200 * x
def second_alloy_copper_percent := 0.50
def second_alloy_copper_mass := 800 * second_alloy_copper_percent
def total_mass := 1000
def target_copper_percent := 0.45
def total_copper_mass := total_mass * target_copper_percent

-- The proof statement
theorem percentage_of_copper_in_first_alloy 
  (h : first_alloy_copper_mass + second_alloy_copper_mass = total_copper_mass) :
  x = 0.25 :=
by
  sorry

end percentage_of_copper_in_first_alloy_l285_285465


namespace part1_part2_l285_285270

noncomputable def concentration1 (x : ℝ) (u : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 4 then (u * (16 / (8 - x) - 1))
  else (u * (5 - x / 2))

noncomputable def concentration2 (x : ℝ) (a : ℝ) : ℝ := 
  2 * (5 - x / 2) + a * (16 / (8 - (x - 6)) - 1)

theorem part1 (u x : ℝ) (h₀ : u = 4) (hx₀ : 0 ≤ x ∧ x ≤ 8) :
  concentration1 x u ≥ 4 :=
by
  rw [concentration1]
  split_ifs
  · exact sorry -- Calculation steps for first interval
  · exact sorry -- Calculation steps for second interval

theorem part2 (a : ℝ) (hx₀ : 1 ≤ a ∧ a ≤ 4) :
  1.6 ≤ a :=
by
  have key : 8 * sqrt 2 - 4 = 1.6,
  { -- This uses the approximation sqrt 2 ≈ 1.4
    have : sqrt 2 = 14/10,
    linarith, },
  suffices : ∀ a, 1 ≤ a ∧ a ≤ 4 → (8 * sqrt a - a - 4) ≥ 4 ↔ a = 1.6,
  exact sorry -- Detailed calculation steps

end part1_part2_l285_285270


namespace calculate_expression_l285_285881

theorem calculate_expression : (36 / (9 + 2 - 6)) * 4 = 28.8 := 
by
    sorry

end calculate_expression_l285_285881


namespace domain_of_f_l285_285355

theorem domain_of_f : 
  ∀ x, (2 - x ≥ 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x ≤ 2) := by
  sorry

end domain_of_f_l285_285355


namespace graph_shift_l285_285990

def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

def axis_of_symmetry (φ : ℝ) : ℝ := π / 6

theorem graph_shift (φ : ℝ) (hφ : |φ| < π / 2) : 
  f (x + π / 12) φ = Real.sin (2 * x + φ) :=
begin
  sorry
end

end graph_shift_l285_285990


namespace part_I_part_II_l285_285302

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}
variable {a_1 d : ℤ}

-- Conditions for part (I)
def cond_1 : Prop := a_n 11 = 0
def cond_2 : Prop := S_n 14 = 98

-- Problem statement for part (I)
theorem part_I (h1 : cond_1) (h2 : cond_2) : ∀ n, a_n n = 22 - 2 * n := 
sorry

-- Conditions for part (II)
def cond_3 : Prop := a_1 ≥ 6
def cond_4 : Prop := a_n 11 > 0
def cond_5 : Prop := S_n 14 ≤ 77

-- Problem Statement for part (II)
theorem part_II (h3 : cond_3) (h4 : cond_4) (h5 : cond_5) : 
  (∀ n, a_n n = 12 - n) ∨ (∀ n, a_n n = 13 - n) := 
sorry

end part_I_part_II_l285_285302


namespace problem_rect_ratio_l285_285263

theorem problem_rect_ratio (W X Y Z U V R S : ℝ × ℝ) 
  (hYZ : Y = (0, 0))
  (hW : W = (0, 6))
  (hZ : Z = (7, 6))
  (hX : X = (7, 4))
  (hU : U = (5, 0))
  (hV : V = (4, 4))
  (hR : R = (5 / 3, 4))
  (hS : S = (0, 4))
  : (dist R S) / (dist X V) = 5 / 9 := 
sorry

end problem_rect_ratio_l285_285263


namespace speed_of_train_A_l285_285389

-- Definitions for the conditions
def length_train_A : ℕ := 125 -- Length of train A in meters
def length_train_B : ℕ := 150 -- Length of train B in meters
def time_to_cross : ℕ := 11 -- Time taken to completely cross train B in seconds
def speed_train_B_kmh : ℕ := 36 -- Speed of train B in km per hour

-- Conversion from km/h to m/s for train B's speed
def speed_train_B := speed_train_B_kmh * 1000 / 3600

-- The proof problem
theorem speed_of_train_A : 
  ∀ (length_A length_B time_cross speed_B : ℕ), 
  length_A = length_train_A →
  length_B = length_train_B →
  time_cross = time_to_cross →
  speed_B = speed_train_B_kmh →
  (length_A + length_B) / time_cross = 25 →
  25 - (speed_B * 1000 / 3600) = 15 →
  (15 * 3600) / 1000 = 54
:= by
  intros _ _ _ _ h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h5 h6
  sorry

end speed_of_train_A_l285_285389


namespace isosceles_triangle_perimeter_l285_285252

-- Definitions of the conditions
def is_isosceles (a b : ℕ) : Prop :=
  a = b

def has_side_lengths (a b : ℕ) (c : ℕ) : Prop :=
  true

-- The statement to be proved
theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h₁ : is_isosceles a b) (h₂ : has_side_lengths a b c) :
  (a + b + c = 16 ∨ a + b + c = 17) :=
sorry

end isosceles_triangle_perimeter_l285_285252


namespace factorial_division_l285_285506

-- Definition of factorial in Lean
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Statement of the problem:
theorem factorial_division : fact 12 / fact 11 = 12 :=
by
  sorry

end factorial_division_l285_285506


namespace problem_solution_l285_285528

variable (n : ℕ)

def solution_condition (x : Fin n → ℝ) : Prop :=
  (∀ i, x i ≠ 0) ∧ (∃ j, x j = -1)

def equation (x : Fin n → ℝ) : ℝ :=
  1 + ∑ i in Finset.range n, ∏ j in Finset.range (i + 1), (if j = 0 then (1 / x j) else (x (j - 1) + 1) / x j)

theorem problem_solution (x : Fin n → ℝ) : 
  equation n x = 0 ↔ solution_condition n x := sorry

end problem_solution_l285_285528


namespace machines_working_time_l285_285384

theorem machines_working_time (y: ℝ) 
  (h1 : y + 8 > 0)  -- condition for time taken by S
  (h2 : y + 2 > 0)  -- condition for time taken by T
  (h3 : 2 * y > 0)  -- condition for time taken by U
  : (1 / (y + 8) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) ↔ (y = 3 / 2) := 
by
  have h4 : y ≠ 0 := by linarith [h1, h2, h3]
  sorry

end machines_working_time_l285_285384


namespace pollution_relationship_slowest_growth_time_l285_285134

-- Definitions based on given conditions
def data_points : List (ℕ × ℝ) := [(1, real.exp 1), (2, real.exp 3), (3, real.exp 4), (4, real.exp 5)]

def log_data_points : List (ℕ × ℝ) := data_points.map (fun (t, y) => (t, real.log y))

def mean (l : List ℝ) : ℝ := l.sum / (l.length : ℝ)

def calc_mean_t (data : List (ℕ × ℝ)) : ℝ :=
  mean (data.map Prod.fst)

def calc_mean_z (data : List (ℕ × ℝ)) : ℝ :=
  mean (data.map Prod.snd)

def calc_sum_tz (data : List (ℕ × ℝ)) : ℝ :=
  (data.map (fun (t, z) => t * z)).sum

def calc_sum_tt (data : List (ℕ × ℝ)) : ℝ :=
  (data.map (fun (t, _) => t ^ 2)).sum

def calc_beta (data : List (ℕ × ℝ)) : ℝ :=
  let n := data.length
  let mean_t := calc_mean_t data
  let mean_z := calc_mean_z data
  let sum_tz := calc_sum_tz data
  let sum_tt := calc_sum_tt data
  (sum_tz - n * mean_t * mean_z) / (sum_tt - n * mean_t ^ 2)

def calc_a (data : List (ℕ × ℝ)) : ℝ :=
  let mean_t := calc_mean_t data
  let mean_z := calc_mean_z data
  mean_z - calc_beta data * mean_t

theorem pollution_relationship :
  (calc_beta log_data_points = 1.3) ∧ (calc_a log_data_points = 0) ∧ 
  ∀ t ∈ [1, 2, 3, 4], (y = real.exp (1.3 * t)) :=
by sorry

theorem slowest_growth_time :
  let f := λ t : ℝ, t⁻¹ * real.exp (1.3 * t) in
  t = 1 / 1.3 ∧ ∀ t' > 0, f t' ≥ f (1 / 1.3) :=
by sorry

end pollution_relationship_slowest_growth_time_l285_285134


namespace distance_from_center_to_plane_l285_285460

theorem distance_from_center_to_plane (r : ℝ) (a b c : ℝ) (O : ℝ) :
  r = 10 → a = 18 → b = 18 → c = 30 →
  (distance_from_center_to_plane O) = (10 * sqrt 37 / 33) :=
sorry

end distance_from_center_to_plane_l285_285460


namespace lateral_surface_area_of_prism_is_32_l285_285086

-- Given conditions
variable (S : ℝ) (R : ℝ) (a : ℝ)

-- Given conditions in Lean
def sphere_surface_area := 24 * Real.pi
def prism_height := 4

axiom radius_eq_sqrt_six : R = Real.sqrt 6
axiom prism_side_length_eq_two : a = 2

-- Proof goal in Lean
theorem lateral_surface_area_of_prism_is_32 : 
  (4 * a * prism_height) = 32 :=
by
  -- Sorry for the proof to indicate we are skipping it
  sorry

end lateral_surface_area_of_prism_is_32_l285_285086


namespace sum_x_y_eq_two_l285_285298

theorem sum_x_y_eq_two (x y : ℝ) 
  (h1 : (x-1)^3 + 2003*(x-1) = -1) 
  (h2 : (y-1)^3 + 2003*(y-1) = 1) : 
  x + y = 2 :=
by
  sorry

end sum_x_y_eq_two_l285_285298


namespace simplify_cot_tan_l285_285735

theorem simplify_cot_tan : Real.cot (Real.pi / 9) + Real.tan (Real.pi / 18) = Real.csc (Real.pi / 9) := 
by
  sorry

end simplify_cot_tan_l285_285735


namespace find_value_of_P_l285_285761

theorem find_value_of_P :
  ∃ P : ℤ, 5 < P ∧ P < 20 ∧ (∃ m : ℤ, (x^2 - 2 * (2 * P - 3) * x + 4 * P^2 - 14 * P + 8).discriminant = m^2) ∧ P = 12 :=
sorry

end find_value_of_P_l285_285761


namespace lattice_points_distance_5_from_origin_l285_285271

theorem lattice_points_distance_5_from_origin :
  {p : Int × Int × Int // p.1 ^ 2 + p.2.1 ^ 2 + p.2.2 ^ 2 = 25}.to_finset.card = 42 :=
by
  sorry

end lattice_points_distance_5_from_origin_l285_285271


namespace tracy_sold_paintings_l285_285804

-- Definitions of conditions
def total_customers := 20
def first_group_customers := 4
def paintings_per_first_group_customer := 2
def second_group_customers := 12
def paintings_per_second_group_customer := 1
def third_group_customers := 4
def paintings_per_third_group_customer := 4

-- Statement of the problem
theorem tracy_sold_paintings :
  (first_group_customers * paintings_per_first_group_customer) +
  (second_group_customers * paintings_per_second_group_customer) +
  (third_group_customers * paintings_per_third_group_customer) = 36 :=
by
  sorry

end tracy_sold_paintings_l285_285804


namespace count_skew_lines_l285_285554

def is_skew (l1 l2 : Line) : Prop :=
  ¬ (l1 ∥ l2) ∧ ¬ (l1.intersects l2)

def given_lines : List Line :=
  [AB', BA', CD', DC', AD', DA', BC', CB', AC, BD, A'C', B'D']

theorem count_skew_lines : ∃ n, n = 30 ∧ 
  (∀ l1 l2, l1 ∈ given_lines ∧ l2 ∈ given_lines ∧ l1 ≠ l2 → is_skew l1 l2) :=
sorry

end count_skew_lines_l285_285554


namespace probability_red_blue_green_marble_l285_285812

theorem probability_red_blue_green_marble :
  let total_marbles := 4 + 3 + 2 + 6 in
  let favorable_marbles := 4 + 3 + 2 in
  (favorable_marbles / total_marbles : ℝ) = 0.6 :=
by
  let total_marbles := 4 + 3 + 2 + 6
  let favorable_marbles := 4 + 3 + 2
  have htotal : total_marbles = 15 := by norm_num
  have hfav : favorable_marbles = 9 := by norm_num
  calc (favorable_marbles : ℝ) / total_marbles
      = (9 : ℝ) / 15 : by rw [htotal, hfav]
  ... = 0.6 : by norm_num

end probability_red_blue_green_marble_l285_285812


namespace bill_order_combinations_l285_285873

theorem bill_order_combinations :
  let k := 3  -- number of specific kinds with restriction
      m := 8  -- total number of donuts
      n := 6  -- total number of kinds
  in (∀ (purchase : Fin k → Nat), (∀ i, purchase i ≥ 2) ∧ (∑ i, purchase i = m)).count
    = 21 := by
    sorry

end bill_order_combinations_l285_285873


namespace first_dig_site_date_difference_l285_285865

-- Definitions for the conditions
def F : Int := sorry  -- The age of the first dig site
def S : Int := sorry  -- The age of the second dig site
def T : Int := sorry  -- The age of the third dig site
def Fo : Int := 8400  -- The age of the fourth dig site
def x : Int := (S - F)

-- The conditions
axiom condition1 : F = S + x
axiom condition2 : T = F + 3700
axiom condition3 : Fo = 2 * T
axiom condition4 : S = 852
axiom condition5 : S > F  -- Ensuring S is older than F for meaningfulness

-- The theorem to prove
theorem first_dig_site_date_difference : x = 352 :=
by
  -- Proof goes here
  sorry

end first_dig_site_date_difference_l285_285865


namespace system1_solution_system2_solution_l285_285337

theorem system1_solution (x y : ℝ) (h1 : 3 * x + y = 4) (h2 : 3 * x + 2 * y = 6) : x = 2 / 3 ∧ y = 2 :=
by
  sorry

theorem system2_solution (x y : ℝ) (h1 : 2 * x + y = 3) (h2 : 3 * x - 5 * y = 11) : x = 2 ∧ y = -1 :=
by
  sorry

end system1_solution_system2_solution_l285_285337


namespace area_of_union_circles_l285_285348

def radius1 : ℝ := 2
def radius2 : ℝ := 5

theorem area_of_union_circles :
  ∀ (R1 R2 : ℝ), R1 = radius1 ∧ R2 = radius2 →
  let θ := Real.arccos (1 / 5) in
  let area_R1 := π * R1^2 in
  let area_R2 := π * R2^2 in
  let overlap_area := 4 * θ + 25 * θ + 4 * Real.sqrt 6 in
  area_R1 + area_R2 - overlap_area = 4 * π + 46 * θ + 4 * Real.sqrt 6 :=
begin
  intros,
  simp [radius1, radius2],
  sorry
end

end area_of_union_circles_l285_285348


namespace exists_nat_with_k_distinct_prime_factors_of_sum_l285_285981

theorem exists_nat_with_k_distinct_prime_factors_of_sum (k : ℕ) (m : ℕ) (hk : 0 < k) (hm : m % 2 = 1) : 
  ∃ (n : ℕ), m^n + n^m ≥ k ∧ ((nat.prime_factors (m^n + n^m)).length >= k) :=
sorry

end exists_nat_with_k_distinct_prime_factors_of_sum_l285_285981


namespace partition_set_relatively_prime_l285_285708

theorem partition_set_relatively_prime (n : ℕ) (h : n % 2 = 1) : 
  ∃ (A B : Finset ℕ), 
    A ∪ B = Finset.range (33) + n ∧ A.card = 14 ∧ B.card = 19 ∧ 
    (∀ a ∈ A, ∀ b ∈ A, (a ≠ b) → Nat.gcd a b = 1) ∧ 
    (∀ a ∈ B, ∀ b ∈ B, (a ≠ b) → Nat.gcd a b = 1) :=
by
  sorry

end partition_set_relatively_prime_l285_285708


namespace sqrt_two_over_two_not_covered_l285_285513

theorem sqrt_two_over_two_not_covered 
  (a b : ℕ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≠ 0) (h4 : Nat.gcd a b = 1) :
  ¬ (real.sqrt 2 / 2 ∈ set.Icc (a / b - 1 / (4 * b ^ 2) : ℝ) (a / b + 1 / (4 * b ^ 2) : ℝ)) :=
sorry

end sqrt_two_over_two_not_covered_l285_285513


namespace lattice_points_at_distance_5_l285_285273

theorem lattice_points_at_distance_5 :
  ∃ S : Finset (ℤ × ℤ × ℤ), (∀ p ∈ S, let ⟨x, y, z⟩ := p in x^2 + y^2 + z^2 = 25) ∧ S.card = 18 := by
  sorry

end lattice_points_at_distance_5_l285_285273


namespace sequence_2023rd_letter_is_B_l285_285012

theorem sequence_2023rd_letter_is_B :
  let sequence := "ABCDDCBA"
  let n := 2023
  let position := n % 8
  position = 7 → sequence[position - 1] = 'B' :=
by
  intros
  unfold sequence position
  rw [mod_eq_of_lt]
  sorry

end sequence_2023rd_letter_is_B_l285_285012


namespace angle_sum_equals_l285_285645

theorem angle_sum_equals (
  ABC : Triangle,
  E B D F C : Point,
  h1 : EB ≃ ED,
  h2 : FC ≃ FD,
  h3 : ∠ EDF = 72°
) : ∠ AED + ∠ AFD = 216° :=
sorry

end angle_sum_equals_l285_285645


namespace function_identity_l285_285968

theorem function_identity (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end function_identity_l285_285968


namespace number_of_sons_l285_285075

noncomputable def land_area_hectares : ℕ := 3
noncomputable def hectare_to_m2 : ℕ := 10000
noncomputable def profit_per_section_per_3months : ℕ := 500
noncomputable def section_area_m2 : ℕ := 750
noncomputable def profit_per_son_per_year : ℕ := 10000
noncomputable def months_in_year : ℕ := 12
noncomputable def months_per_season : ℕ := 3

theorem number_of_sons :
  let total_land_area_m2 := land_area_hectares * hectare_to_m2
  let yearly_profit_per_section := profit_per_section_per_3months * (months_in_year / months_per_season)
  let number_of_sections := total_land_area_m2 / section_area_m2
  let total_yearly_profit := number_of_sections * yearly_profit_per_section
  let n := total_yearly_profit / profit_per_son_per_year
  n = 8 :=
by
  sorry

end number_of_sons_l285_285075


namespace perfect_square_trinomial_m_l285_285247

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ (a b : ℝ), ∀ x : ℝ, x^2 + (m-1)*x + 9 = (a*x + b)^2) ↔ (m = 7 ∨ m = -5) :=
sorry

end perfect_square_trinomial_m_l285_285247


namespace triangle_side_b_ge_sqrt_2_l285_285186

theorem triangle_side_b_ge_sqrt_2 {a b c : ℝ} (h_area : 1 = (1 / 2) * a * b * real.sin (real.angle a b c))
  (h_sides : a ≤ b ∧ b ≤ c) 
  (h_trig_sin : real.sin (real.angle a b c) ≤ 1) : b ≥ real.sqrt 2 := 
sorry

end triangle_side_b_ge_sqrt_2_l285_285186


namespace F_101_eq_52_l285_285239

def F : ℕ → ℚ
| 1 := 2
| (n + 1) := (2 * (F n) + 1) / 2

theorem F_101_eq_52 : F 101 = 52 :=
by {
  sorry
}

end F_101_eq_52_l285_285239


namespace count_valid_sequences_length_21_l285_285237

def valid_sequences_count (n : ℕ) : ℕ :=
  if n = 3 then 1 else
  if n = 4 then 1 else
  if n = 5 then 1 else
  if n = 6 then 2 else
  valid_sequences_count (n - 4) + 
  2 * valid_sequences_count (n - 5) + 
  2 * valid_sequences_count (n - 6)

theorem count_valid_sequences_length_21 : valid_sequences_count 21 = 135 := by
  sorry

end count_valid_sequences_length_21_l285_285237


namespace min_S_min_S_values_range_of_c_l285_285959

-- Part 1
theorem min_S (a b c : ℝ) (h : a + b + c = 1) : 
  2 * a^2 + 3 * b^2 + c^2 ≥ (6 / 11) :=
sorry

-- Part 1, finding exact values of a, b, c where minimum is reached
theorem min_S_values (a b c : ℝ) (h : a + b + c = 1) :
  2 * a^2 + 3 * b^2 + c^2 = (6 / 11) ↔ a = (3 / 11) ∧ b = (2 / 11) ∧ c = (6 / 11) :=
sorry
  
-- Part 2
theorem range_of_c (a b c : ℝ) (h1 : 2 * a^2 + 3 * b^2 + c^2 = 1) : 
  (1 / 11) ≤ c ∧ c ≤ 1 :=
sorry

end min_S_min_S_values_range_of_c_l285_285959


namespace coast_guard_overtake_smuggler_l285_285856

noncomputable def time_of_overtake (initial_distance : ℝ) (initial_time : ℝ) 
                                   (smuggler_speed1 coast_guard_speed : ℝ) 
                                   (duration1 new_smuggler_speed : ℝ) : ℝ :=
  let distance_after_duration1 := initial_distance + (smuggler_speed1 * duration1) - (coast_guard_speed * duration1)
  let relative_speed_new := coast_guard_speed - new_smuggler_speed
  duration1 + (distance_after_duration1 / relative_speed_new)

theorem coast_guard_overtake_smuggler : 
  time_of_overtake 15 0 18 20 1 16 = 4.25 := by
  sorry

end coast_guard_overtake_smuggler_l285_285856


namespace monotone_increasing_f_l285_285613

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - Real.sin x) / (Real.cos x)

theorem monotone_increasing_f (a : ℝ) :
  (∀ x ∈ Set.Ioo (Real.pi / 6) (Real.pi / 3), 0 ≤ ((a - Real.sin x) / (Real.cos x))') ↔ 2 ≤ a :=
sorry

end monotone_increasing_f_l285_285613


namespace tracy_sold_paintings_l285_285799

-- Definitions based on conditions
def total_customers := 20
def customers_2_paintings := 4
def customers_1_painting := 12
def customers_4_paintings := 4

def paintings_per_customer_2 := 2
def paintings_per_customer_1 := 1
def paintings_per_customer_4 := 4

-- Theorem statement
theorem tracy_sold_paintings : 
  (customers_2_paintings * paintings_per_customer_2) + 
  (customers_1_painting * paintings_per_customer_1) + 
  (customers_4_paintings * paintings_per_customer_4) = 36 := 
by
  sorry

end tracy_sold_paintings_l285_285799


namespace commute_time_l285_285911

variables (v_real : ℝ) (v_freeway : ℝ) (distance_freeway : ℝ) (distance_local : ℝ) (time_local : ℝ) (time_freeway : ℝ) (total_time : ℝ)

-- Define the conditions
def conditions : Prop :=
  distance_freeway = 100 ∧
  distance_local = 25 ∧
  v_freeway = 2 * v_real ∧
  time_local = 50 ∧
  time_local = distance_local / v_real ∧
  time_freeway = distance_freeway / v_freeway

-- Question: Prove that total time is 150 minutes given the conditions.
theorem commute_time (h : conditions) : total_time = 150 :=
by
  rw [conditions] at h
  sorry

end commute_time_l285_285911


namespace tennis_percentage_in_combined_schools_l285_285784

theorem tennis_percentage_in_combined_schools :
  (let total_njhs := 1800
       percent_tennis_njhs := 25
       total_sms := 2700
       percent_tennis_sms := 35
       num_tennis_njhs := total_njhs * percent_tennis_njhs / 100
       num_tennis_sms := total_sms * percent_tennis_sms / 100
       total_tennis := num_tennis_njhs + num_tennis_sms
       total_students := total_njhs + total_sms
       percentage_tennis_combined := total_tennis * 100 / total_students
   in percentage_tennis_combined = 31) := sorry

end tennis_percentage_in_combined_schools_l285_285784


namespace cot_tan_simplify_l285_285742

theorem cot_tan_simplify : cot (20 * π / 180) + tan (10 * π / 180) = csc (20 * π / 180) :=
by
  sorry

end cot_tan_simplify_l285_285742


namespace extreme_value_f_at_a_eq_1_monotonic_intervals_f_exists_no_common_points_l285_285214

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a * Real.log (x - 1)

-- Problem 1: Prove that the extreme value of f(x) when a = 1 is \frac{3}{4} + \ln 2
theorem extreme_value_f_at_a_eq_1 : 
  f (3/2) 1 = 3/4 + Real.log 2 :=
sorry

-- Problem 2: Prove the monotonic intervals of f(x) based on the value of a
theorem monotonic_intervals_f :
  ∀ a : ℝ, 
    (if a ≤ 0 then 
      ∀ x, 1 < x → f x' a > 0
     else
      ∀ x, 1 < x ∧ x ≤ (a + 2) / 2 → f x a ≤ 0 ∧ ∀ x, x ≥ (a + 2) / 2 → f x a > 0) :=
sorry

-- Problem 3: Prove that for a ≥ 1, there exists an a such that f(x) has no common points with y = \frac{5}{8} + \ln 2
theorem exists_no_common_points (h : 1 ≤ a) :
  ∃ x : ℝ, f x a ≠ 5/8 + Real.log 2 :=
sorry

end extreme_value_f_at_a_eq_1_monotonic_intervals_f_exists_no_common_points_l285_285214


namespace simplify_cot_tan_l285_285714

theorem simplify_cot_tan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := by
  sorry

end simplify_cot_tan_l285_285714


namespace find_positive_integer_k_l285_285929

theorem find_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ (k.digits.product = (11*k / 4) - 199) ∧ k = 84 :=
by
  sorry

end find_positive_integer_k_l285_285929


namespace price_decrease_approx_l285_285404

noncomputable def original_price : ℝ := 77.95
noncomputable def sale_price : ℝ := 59.95
noncomputable def amount_decrease : ℝ := original_price - sale_price
noncomputable def percentage_decrease : ℝ := (amount_decrease / original_price) * 100

theorem price_decrease_approx : percentage_decrease ≈ 23.08 :=
by sorry

end price_decrease_approx_l285_285404


namespace question1_proof_question2_proof_l285_285946

noncomputable def expr1 := (9 / 4) ^ (1 / 2) - 1 - (27 / 8) ^ (-2 / 3) + (3 / 2) ^ (-2)
noncomputable def expr2 := (1 / 4) * (log 27 / log 3) - 1 + 2 * (log 5 / log 10) + log 4 / log 10

theorem question1_proof : expr1 = 13 / 18 :=
by
  sorry

theorem question2_proof : expr2 = 7 / 4 :=
by
  sorry

end question1_proof_question2_proof_l285_285946


namespace ellipse_equation_eccentricity_line_equation_AB_AC_passes_through_midpoint_l285_285190

/-- Problem (I): Determining the equation of an ellipse and its eccentricity -/
theorem ellipse_equation_eccentricity :
  ∃ a b x y e : ℝ,
  (a > b > 0) ∧ (2 * b = 2) ∧ (1 < a) ∧ 
  (b = 1) ∧ (a = sqrt 2) ∧ (e = sqrt 2 / 2) ∧ 
  ((x^2 / a^2) + (y^2 / b^2) = 1) := 
  sorry

/-- Problem (II): Finding the equation of line AB given a constraint on segments BC and AD lengths -/
theorem line_equation_AB (BC AD : ℝ) :
  (BC = (1 / 3) * AD) → 
  (∃ k : ℝ, k = 1 ∨ k = -1) ∧ 
  (∃ x y : ℝ, y = k * (x - 1) ∧ ((x^2 / 2) + y^2 = 1) 
  → ((x - y - 1 = 0) ∨ (x + y - 1 = 0))) :=
  sorry

/-- Problem (III): Proving that line AC passes through the midpoint of segment EF -/
theorem AC_passes_through_midpoint (F E : ℝ × ℝ) (A C N : ℝ × ℝ) :
  F = (1, 0) → E = (2, 0) → N = (3/2, 0) →
  (A.1, A.2) = (1, y) → (C.1, C.2) = (2, -y) → 
  ((A.1 + C.1) / 2 = 3/2) → 
  ((A.1, A.2), (C.1, C.2)) = ((1, y), (2, -y)) → 
  (N = (3/2, 0)) :=
  sorry

end ellipse_equation_eccentricity_line_equation_AB_AC_passes_through_midpoint_l285_285190


namespace candidate_wage_difference_l285_285054

noncomputable def wage_difference (P Q : ℝ) : Prop :=
  (Q * ((360 / (1.5 * Q)) + 10) = 360) ∧
  (1.5 * Q * (360 / (1.5 * Q)) = 360) ∧
  P = 1.5 * Q → 
  P - Q = 6

theorem candidate_wage_difference : ∀ (P Q : ℝ), wage_difference P Q :=
begin
  sorry
end

end candidate_wage_difference_l285_285054


namespace find_largest_d_l285_285290

theorem find_largest_d (m n d : ℕ) (h₁ : m ≥ 3) (h₂ : n > m * (m - 2)) :
  (d ∣ n! ∧ ∀ k, (m ≤ k ∧ k ≤ n) → ¬ (k ∣ d)) → d = m - 1 :=
begin
  sorry
end

end find_largest_d_l285_285290


namespace find_k_and_prove_geometric_sequence_l285_285971

/-
Given conditions:
1. Sequence sa : ℕ → ℝ with sum sequence S : ℕ → ℝ satisfying the recurrence relation S (n + 1) = (k + 1) * S n + 2
2. Initial terms a_1 = 2 and a_2 = 1
-/

def sequence_sum_relation (S : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n : ℕ, S (n + 1) = (k + 1) * S n + 2

def init_sequence_terms (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ a 2 = 1

/-
Proof goal:
1. Prove k = -1/2 given the conditions.
2. Prove sequence a is a geometric sequence with common ratio 1/2 given the conditions.
-/

theorem find_k_and_prove_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ) :
  sequence_sum_relation S k →
  init_sequence_terms a →
  (k = (-1:ℝ)/2) ∧ (∀ n: ℕ, n ≥ 1 → a (n+1) = (1/2) * a n) :=
by
  sorry

end find_k_and_prove_geometric_sequence_l285_285971


namespace sandy_final_position_l285_285326

open Function

-- Define Sandy's movements as points in a 2-dimensional plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the initial point A
def A : Point := { x := 0, y := 0 }

-- Define the point after Sandy walks 20 meters south
def B : Point := { x := 0, y := -20 }

-- Define the point after Sandy walks 20 meters east
def C : Point := { x := 20, y := -20 }

-- Define the point after Sandy walks 20 meters north
def D : Point := { x := 20, y := 0 }

-- Define the point after Sandy walks 25 meters east
def E : Point := { x := 45, y := 0 }

-- The theorem to prove the final distance and direction from starting point
theorem sandy_final_position
  (A B C D E : Point)
  (h1 : B = { x := A.x, y := A.y - 20 })
  (h2 : C = { x := B.x + 20, y := B.y })
  (h3 : D = { x := C.x, y := C.y + 20 })
  (h4 : E = { x := D.x + 25, y := D.y }) :
  dist E A = 25 ∧ E.x > A.x := sorry

end sandy_final_position_l285_285326


namespace simplify_cot_tan_l285_285731

theorem simplify_cot_tan : Real.cot (Real.pi / 9) + Real.tan (Real.pi / 18) = Real.csc (Real.pi / 9) := 
by
  sorry

end simplify_cot_tan_l285_285731


namespace formation_of_KCl_l285_285236

-- Define the constants and conditions
def moles_HCl : ℕ := 2
def moles_KOH : ℕ := 2
def moles_H2O : ℕ := 2

-- Assume the stoichiometric ratio is 1:1 for all reactants and products
axiom stoichiometry_HCl_KOH_KCl_H2O : ∀ (x : ℕ), (HCl x) ∧ (KOH x) -> (KCl x) ∧ (H2O x)

-- The proof problem statement
theorem formation_of_KCl
  (h_HCl : moles_HCl = 2)
  (h_KOH : moles_KOH = 2)
  (h_H2O : moles_H2O = 2) :
  moles_KOH = moles_H2O → moles_KOH = 2 → moles_HCl = moles_KCl ∧ moles_H2O = moles_KOH := 
begin
    sorry
end

end formation_of_KCl_l285_285236


namespace monotonic_increasing_interval_l285_285783

noncomputable def u (x : ℝ) : ℝ := -x^2 - 2*x + 3

theorem monotonic_increasing_interval :
  (∃ I : set ℝ, I = {-3 < x | x ≤ -1} ∧ 
               (∀ x ∈ I, u(x) > 0) ∧ 
               (∀ x1 x2 ∈ I, x1 < x2 → u(x1) ≤ u(x2)) ∧ 
               (∀ x1 x2 ∈ I, x1 < x2 → log (u x1) ≤ log (u x2))) := 
begin
  sorry
end

end monotonic_increasing_interval_l285_285783


namespace students_in_second_class_find_students_in_second_class_l285_285434

theorem students_in_second_class 
  (students_class1 : ℕ)
  (average_marks_class1 : ℝ)
  (average_marks_class2 : ℝ)
  (total_students_avg_marks : ℝ)
  (total_marks_class1 := students_class1 * average_marks_class1) -- Simplifying condition
  (total_marks_class2 : ℝ := average_marks_class2 * students_class2)
  (total_students := students_class1 + students_class2)
  (combined_total_marks := total_marks_class1 + total_marks_class2) : 
  students_class2 = 28 :=
by
  sorry

-- Definitions and conditions
variables 
  {students_class1 : ℕ} -- total number of students in the first class
  {average_marks_class1 average_marks_class2 total_students_avg_marks : ℝ} -- average marks of classes and total average
  {students_class2 : ℕ}
  (h1 : students_class1 = 22)
  (h2 : average_marks_class1 = 40)
  (h3 : average_marks_class2 = 60)
  (h4 : total_students_avg_marks = 51.2)

include h1 h2 h3 h4

theorem find_students_in_second_class (h : (22 * 40 + students_class2 * 60) / (22 + students_class2) = 51.2) :
  students_class2 = 28 :=
by 
  sorry

end students_in_second_class_find_students_in_second_class_l285_285434


namespace problem_conditions_l285_285665

noncomputable def S (n : ℕ) (a : ℕ → ℚ) := ∑ i in finset.range n, a (i + 1)

theorem problem_conditions
  (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≠ 0 → ((S n a) / (a n)) = 1 + (1 / 3) * (n - 1)) :
  (∀ n : ℕ, a n = (n * (n + 1)) / 2) ∧
  (∀ n : ℕ, (∑ i in finset.range n, 1 / a (i + 1)) < 2)
  :=
by
  sorry

end problem_conditions_l285_285665


namespace right_triangle_eqn_roots_indeterminate_l285_285183

theorem right_triangle_eqn_roots_indeterminate 
  (a b c : ℝ) (h : a^2 + c^2 = b^2) : 
  ¬(∃ Δ, Δ = 4 - 4 * c^2 ∧ (Δ > 0 ∨ Δ = 0 ∨ Δ < 0)) →
  (¬∃ x, a * (x^2 - 1) - 2 * x + b * (x^2 + 1) = 0 ∨
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ a * (x₁^2 - 1) - 2 * x₁ + b * (x₁^2 + 1) = 0 ∧ a * (x₂^2 - 1) - 2 * x₂ + b * (x₂^2 + 1) = 0) :=
by
  sorry

end right_triangle_eqn_roots_indeterminate_l285_285183


namespace geom_arith_seq_first_term_is_two_l285_285360

theorem geom_arith_seq_first_term_is_two (b q a d : ℝ) 
  (hq : q ≠ 1) 
  (h_geom_first : b = a + d) 
  (h_geom_second : b * q = a + 3 * d) 
  (h_geom_third : b * q^2 = a + 6 * d) 
  (h_prod : b * b * q * b * q^2 = 64) :
  b = 2 :=
by
  sorry

end geom_arith_seq_first_term_is_two_l285_285360


namespace extra_people_got_on_the_train_l285_285793

-- Definitions corresponding to the conditions
def initial_people_on_train : ℕ := 78
def people_got_off : ℕ := 27
def current_people_on_train : ℕ := 63

-- The mathematical equivalent proof problem
theorem extra_people_got_on_the_train :
  (initial_people_on_train - people_got_off + extra_people = current_people_on_train) → (extra_people = 12) :=
by
  sorry

end extra_people_got_on_the_train_l285_285793


namespace solution_l285_285212

open Real

def statement1 (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) (h3 : α > β) : Prop :=
  ¬ (sin α > sin β)

def statement2 : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → y ≤ (5 * π) / 12 → sin(2 * x - π / 3) ≤ sin(2 * y - π / 3)

def statement3 : Prop :=
  ¬ (cos (2 * -π / 6 + π / 3) = 0)

def statement4 : Prop :=
  ∀ x, -1 ≤ min (sin x) (cos x) ∧ min (sin x) (cos x) ≤ sqrt 2 / 2

def problem : Prop :=
(statement2 ∧ statement4)  ∧ (¬ statement1 ∧ ¬ statement3)

theorem solution : problem := sorry

end solution_l285_285212


namespace factorial_division_l285_285501

open Nat

theorem factorial_division : 12! / 11! = 12 := sorry

end factorial_division_l285_285501


namespace factorize_one_factorize_two_l285_285523

variable (x a b : ℝ)

-- Problem 1: Prove that 4x^2 - 64 = 4(x + 4)(x - 4)
theorem factorize_one : 4 * x^2 - 64 = 4 * (x + 4) * (x - 4) :=
sorry

-- Problem 2: Prove that 4ab^2 - 4a^2b - b^3 = -b(2a - b)^2
theorem factorize_two : 4 * a * b^2 - 4 * a^2 * b - b^3 = -b * (2 * a - b)^2 :=
sorry

end factorize_one_factorize_two_l285_285523


namespace base_7_even_last_digit_l285_285955

theorem base_7_even_last_digit :
  ∃ (b : ℕ), (b = 7) ∧ (b^3 ≤ 625) ∧ (625 < b^4) ∧ (Nat.digits b 625).length = 4 ∧ (Nat.digits b 625).head = 2 :=
by
  sorry

end base_7_even_last_digit_l285_285955


namespace cot20_tan10_eq_csc20_l285_285746

theorem cot20_tan10_eq_csc20 : 
  Real.cot 20 * (Real.pi / 180) + Real.tan 10 * (Real.pi / 180) = Real.csc 20 * (Real.pi / 180) :=
by sorry

end cot20_tan10_eq_csc20_l285_285746


namespace lcm_18_30_l285_285019

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l285_285019


namespace fifth_grade_soccer_students_l285_285643

variable (T B Gnp GP S : ℕ)
variable (p : ℝ)

theorem fifth_grade_soccer_students
  (hT : T = 420)
  (hB : B = 296)
  (hp_percent : p = 86 / 100)
  (hGnp : Gnp = 89)
  (hpercent_boys_playing_soccer : (1 - p) * S = GP)
  (hpercent_girls_playing_soccer : GP = 35) :
  S = 250 := by
  sorry

end fifth_grade_soccer_students_l285_285643


namespace box_volume_l285_285438

theorem box_volume (l w s : ℕ) (h_l : l = 48) (h_w : w = 36) (h_s : s = 6) 
  : let new_length := l - 2 * s
        new_width := w - 2 * s
        height := s
    in new_length * new_width * height = 5184 :=
by
  let new_length := l - 2 * s
  let new_width := w - 2 * s
  let height := s
  exact sorry

end box_volume_l285_285438


namespace eggs_processed_per_day_l285_285626

/-- In a certain egg-processing plant, every egg must be inspected, and is either accepted for processing or rejected. For every 388 eggs accepted for processing, 12 eggs are rejected.

If, on a particular day, 37 additional eggs were accepted, but the overall number of eggs inspected remained the same, the ratio of those accepted to those rejected would be 405 to 3.

Prove that the number of eggs processed per day, given these conditions, is 125763.
-/
theorem eggs_processed_per_day : ∃ (E : ℕ), (∃ (R : ℕ), 38 * R = 3 * (E - 37) ∧  E = 32 * R + E / 33 ) ∧ (E = 125763) :=
sorry

end eggs_processed_per_day_l285_285626


namespace max_d_minus_r_proof_l285_285618

noncomputable def max_d_minus_r : ℕ := 35

theorem max_d_minus_r_proof (d r : ℕ) (h1 : 2017 % d = r) (h2 : 1029 % d = r) (h3 : 725 % d = r) :
  d - r ≤ max_d_minus_r :=
  sorry

end max_d_minus_r_proof_l285_285618


namespace tetrahedron_vertices_l285_285520

-- Definitions
variables {P Q R S : Type} [Point P] [Point Q] [Point R] [Point S]

-- Midpoints of the edges serving as conditions
variable {midPQ : Point}
variable {midPR : Point}
variable {midPS : Point}
variable {midQS : Point}
variable {midQR : Point}
variable {midRS : Point}

-- Prove that given these midpoints, either there are 12 distinct tetrahedrons 
-- or infinitely many tetrahedrons
theorem tetrahedron_vertices (P Q R S : Point)
  (midPQ midPR midPS midQS midQR midRS : Point) :
  (∃ Tetrahedrons, non_coplanar midPQ midPR midPS midQS → set.finite Tetrahedrons ∧ finset.card Tetrahedrons = 12) 
  ∨ (∀ Tetrahedrons, coplanar midPQ midPR midPS midQS → ∃! Tetrahedrons, has_infinite_distinct_elements Tetrahedrons) :=
by sorry

end tetrahedron_vertices_l285_285520


namespace unit_cubes_intersected_by_plane_l285_285853

theorem unit_cubes_intersected_by_plane :
  ∀ (n : ℕ), n = 4 →
  ∀ (k : ℕ), k = 64 →
  ∀ (ratio : ℕ × ℕ), ratio = (1, 3) →
  ∃ (intersect_cubes : ℕ), intersect_cubes = 32 :=
by intros n h1 k h2 ratio h3
   use 32
   sorry

end unit_cubes_intersected_by_plane_l285_285853


namespace sum_f_values_l285_285537

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x  
axiom f_period_2 : ∀ x : ℝ, f (x + 2) = f x  

theorem sum_f_values : (List.range 2008).sum (λ n, f (n + 1)) = 0 := 
by
  sorry

end sum_f_values_l285_285537


namespace triangle_division_possible_l285_285646

-- Definitions for the conditions
variables {T : Type} [triangle T]

-- Statement of the problem
theorem triangle_division_possible :
  ∀ (Δ : T), ∃ (F1 F2 F3 : set T), 
    (area F1 > 1/4 * area Δ) ∧
    (area F2 > 1/4 * area Δ) ∧
    (area F3 > 1/4 * area Δ) ∧
    (F1 ∪ F2 ∪ F3 = Δ) ∧
    (F1 ∩ F2 = ∅) ∧
    (F2 ∩ F3 = ∅) ∧
    (F1 ∩ F3 = ∅) := sorry

end triangle_division_possible_l285_285646


namespace number_of_real_values_p_l285_285130

theorem number_of_real_values_p (p : ℝ) :
  (∀ p: ℝ, x^2 - (p + 1) * x + (p + 1)^2 = 0 -> (p + 1) ^ 2 = 0) ↔ p = -1 := by
  sorry

end number_of_real_values_p_l285_285130


namespace geometric_sequence_first_term_l285_285358

variable {b q a d : ℝ}

-- Hypotheses and Conditions
hypothesis h_geom_seq : ∀ n, b * q ^ n
hypothesis h_arith_seq : ∀ n, a + n * d
hypothesis h_product : b * (b * q) * (b * q^2) = 64

-- Theorems to prove b == 8/3
theorem geometric_sequence_first_term :
  (∃ b q a d : ℝ, (∀ n, b * q ^ n = a + n * d) ∧ (b * (b * q) * (b * q^2) = 64)) → b = 8/3 := by 
  sorry

end geometric_sequence_first_term_l285_285358


namespace sum_k_plus_3_sq_eq_334_l285_285684

theorem sum_k_plus_3_sq_eq_334 (x : ℕ → ℝ)
  (h1 : ∑ k in finset.range 7, (k + 1) ^ 2 * x k = 1)
  (h2 : ∑ k in finset.range 7, (k + 2) ^ 2 * x k = 12)
  (h3 : ∑ k in finset.range 7, (k + 3) ^ 2 * x k = 123) :
  ∑ k in finset.range 7, (k + 4) ^ 2 * x k = 334 :=
sorry

end sum_k_plus_3_sq_eq_334_l285_285684


namespace clock_correction_l285_285091

def gain_per_day : ℚ := 13 / 4
def hours_per_day : ℕ := 24
def days_passed : ℕ := 9
def extra_hours : ℕ := 8
def total_hours : ℕ := days_passed * hours_per_day + extra_hours
def gain_per_hour : ℚ := gain_per_day / hours_per_day
def total_gain : ℚ := total_hours * gain_per_hour
def required_correction : ℚ := 30.33

theorem clock_correction :
  total_gain = required_correction :=
  by sorry

end clock_correction_l285_285091


namespace us2_eq_3958_div_125_l285_285294

-- Definitions based on conditions
def t (x : ℚ) : ℚ := 5 * x - 12
def s (t_x : ℚ) : ℚ := (2 : ℚ) ^ 2 + 3 * 2 - 2
def u (s_t_x : ℚ) : ℚ := (14 : ℚ) / 5 ^ 3 + 2 * (14 / 5) ^ 2 - 14 / 5 + 4

-- Prove that u(s(2)) = 3958 / 125
theorem us2_eq_3958_div_125 : u (s (2)) = 3958 / 125 := by
  sorry

end us2_eq_3958_div_125_l285_285294


namespace bronson_yellow_leaves_l285_285878

-- Bronson collects 12 leaves on Thursday
def leaves_thursday : ℕ := 12

-- Bronson collects 13 leaves on Friday
def leaves_friday : ℕ := 13

-- 20% of the leaves are Brown (as a fraction)
def percent_brown : ℚ := 0.2

-- 20% of the leaves are Green (as a fraction)
def percent_green : ℚ := 0.2

theorem bronson_yellow_leaves : 
  (leaves_thursday + leaves_friday) * (1 - percent_brown - percent_green) = 15 := by
sorry

end bronson_yellow_leaves_l285_285878


namespace absolute_error_2175000_absolute_error_1730000_l285_285276

noncomputable def absolute_error (a : ℕ) : ℕ :=
  if a = 2175000 then 1
  else if a = 1730000 then 10000
  else 0

theorem absolute_error_2175000 : absolute_error 2175000 = 1 :=
by sorry

theorem absolute_error_1730000 : absolute_error 1730000 = 10000 :=
by sorry

end absolute_error_2175000_absolute_error_1730000_l285_285276


namespace vector_eq_l285_285884

variables (a b : Type) [AddCommGroup a] [VectorSpace ℝ a]

theorem vector_eq : 4 • a - 3 • (a + b) = a - 3 • b := by
  sorry

end vector_eq_l285_285884


namespace cot_tan_simplify_l285_285737

theorem cot_tan_simplify : cot (20 * π / 180) + tan (10 * π / 180) = csc (20 * π / 180) :=
by
  sorry

end cot_tan_simplify_l285_285737


namespace minimum_value_f_m_plus_f_prime_n_l285_285578

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 - 4

noncomputable def f' (x : ℝ) : ℝ := -3 * x^2 + 6 * x

def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem minimum_value_f_m_plus_f_prime_n :
  ∃ (m n : ℝ), m ∈ interval ∧ n ∈ interval ∧ f(m) + f'(n) = -13 :=
by
  sorry

end minimum_value_f_m_plus_f_prime_n_l285_285578


namespace central_cell_value_l285_285254

noncomputable def x_bottom_left : ℚ := 341 / (31 * 121)

def cell_value (i j : ℕ) : ℚ :=
  let x := x_bottom_left in
  (2^i) * (3^j) * x

def sum_cells : ℚ :=
  ∑ i in finset.range 5, ∑ j in finset.range 5, cell_value i j

theorem central_cell_value :
  let central_value := cell_value 2 2 in
  sum_cells = 341 → central_value = 36 / 11 :=
by
  let central_value := cell_value 2 2
  let x := x_bottom_left
  have h1 : central_value = 36 * x := by sorry
  have h2 : x = 1 / 11 := by sorry
  rw [h1, h2]
  norm_num
  exact id

end central_cell_value_l285_285254


namespace planting_ratio_is_three_to_one_l285_285439

-- Definitions based on conditions
def trees_chopped_down_first_half : ℕ := 200
def trees_chopped_down_second_half : ℕ := 300
def total_trees_to_plant : ℕ := 1500

-- The proof goal
theorem planting_ratio_is_three_to_one :
  let total_trees_chopped := trees_chopped_down_first_half + trees_chopped_down_second_half in
  total_trees_to_plant / total_trees_chopped = 3 :=
by
  sorry

end planting_ratio_is_three_to_one_l285_285439


namespace roots_are_complex_l285_285534

-- Define the polynomial
def polynomial (x : ℂ) : ℂ := 5 * x^4 - 28 * x^3 + 57 * x^2 - 28 * x + 5

-- Statement that the roots of the polynomial are complex numbers
theorem roots_are_complex : ∀ x : ℂ, polynomial x = 0 → x ∈ ℂ := by
  sorry

end roots_are_complex_l285_285534


namespace greenwood_school_l285_285473

theorem greenwood_school (f s : ℕ) (h : (3 / 4) * f = (1 / 3) * s) : s = 3 * f :=
by
  sorry

end greenwood_school_l285_285473


namespace ab_not_necessary_nor_sufficient_l285_285199

theorem ab_not_necessary_nor_sufficient (a b : ℝ) : 
  (ab > 0 → a + b > 0) ↔ false ∧
  (a + b > 0 → ab > 0) ↔ false := 
sorry

end ab_not_necessary_nor_sufficient_l285_285199


namespace find_interesting_pairs_l285_285658

noncomputable theory
open classical

def odd (n : ℕ) : Prop := ¬even n

def is_interesting_pair (a b : ℕ) : Prop :=
∀ n : ℕ, ∃ k : ℕ, 2 ^ n ∣ (a^k + b)

theorem find_interesting_pairs (k l q : ℕ) (hk : 2 ≤ k) (hl : odd l) (hq : odd q) :
  (is_interesting_pair (2^k * l + 1) (2^k * q - 1)) ∧
  (is_interesting_pair (2^k * l - 1) (2^k * q + 1)) :=
by
  sorry

end find_interesting_pairs_l285_285658


namespace car_second_hour_speed_l285_285379

theorem car_second_hour_speed :
  ∃ x : ℝ, 
    let speed_first_hour := 90 in
    let avg_speed := 60 in
    let total_distance := avg_speed * 2 in
    let distance_second_hour := total_distance - speed_first_hour in
    let speed_second_hour := distance_second_hour in
    speed_second_hour = 30 :=
by {
    sorry
}

end car_second_hour_speed_l285_285379


namespace figure_symmetry_l285_285321

/-- Prove that a figure with an axis of symmetry and a center of symmetry lying on it has
another axis of symmetry perpendicular to the first and passing through the center of symmetry. -/
theorem figure_symmetry
    (figure : Type)
    (axis_of_symmetry : figure → Prop)
    (center_of_symmetry : figure)
    (h1 : axis_of_symmetry figure)
    (h2 : axis_of_symmetry center_of_symmetry)
    : ∃ (perpendicular_axis : figure → Prop),
        (perpendicular_axis figure) ∧ (perpendicular_axis center_of_symmetry) :=
sorry

end figure_symmetry_l285_285321


namespace hyperbrick_hyperbox_probability_l285_285161

open ProbabilityTheory

-- Define the set of numbers and draws
def U : Set ℕ := { n | 1 ≤ n ∧ n ≤ 500 }
def draws_a := Finset ℕ := {a1, a2, a3, a4, a5 | a1 ≠ a2 ∧ a2 ≠ a3 ∧ a3 ≠ a4 ∧ a4 ≠ a5 ∧ a1 ∈ U ∧ a2 ∈ U ∧ a3 ∈ U ∧ a4 ∈ U ∧ a5 ∈ U}
def draws_b := Finset ℕ := {b1, b2, b3, b4 | b1 ≠ b2 ∧ b2 ≠ b3 ∧ b3 ≠ b4 ∧ b1 ∈ (U \ draws_a) ∧ b2 ∈ (U \ draws_a) ∧ b3 ∈ (U \ draws_a) ∧ b4 ∈ (U \ draws_a)}

-- Theorem statement
theorem hyperbrick_hyperbox_probability :
  let p := Probability (enclosed_hyperbrick_hyperbox draws_a draws_b)
  let (numer, denom) := p.lowest_terms
  numer + denom = 89 := sorry

end hyperbrick_hyperbox_probability_l285_285161


namespace Dad_steps_l285_285897

variable (d m y : ℕ)

-- Conditions
def condition_1 : Prop := d = 3 → m = 5
def condition_2 : Prop := m = 3 → y = 5
def condition_3 : Prop := m + y = 400

-- Question and Answer
theorem Dad_steps : condition_1 d m → condition_2 m y → condition_3 m y → d = 90 :=
by
  intros
  sorry

end Dad_steps_l285_285897


namespace number_of_solutions_l285_285902

theorem number_of_solutions : 
  ∀ x ∈ Icc (0 : ℝ) (2 * Real.pi),
    (2 * Real.cos x ^ 3 - 5 * Real.cos x ^ 2 + 2 * Real.cos x = 0) → 
    (∃ y ∈ {0 : ℝ, Real.pi / 2, 3 * Real.pi / 2, Real.pi / 3, 5 * Real.pi / 3}, x = y) :=
by
  sorry

end number_of_solutions_l285_285902


namespace union_complement_l285_285595

open Set

theorem union_complement :
  let U : Set ℝ := univ,
      A : Set ℝ := {x | x > 1},
      B : Set ℝ := {x | 0 < x ∧ x < 2}
  in (U \ A) ∪ B = {x | x < 2} :=
by
  let U : Set ℝ := univ
  let A : Set ℝ := {x | x > 1}
  let B : Set ℝ := {x | 0 < x ∧ x < 2}
  have h1 : U \ A = {x | x ≤ 1}, by sorry
  have h2, by sorry
  exact h2

end union_complement_l285_285595


namespace value_of_half_plus_five_l285_285601

theorem value_of_half_plus_five (n : ℕ) (h₁ : n = 20) : (n / 2) + 5 = 15 := 
by {
  sorry
}

end value_of_half_plus_five_l285_285601


namespace part_a_part_b_l285_285287

def a : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := Real.sqrt ((1 + a n) / 2)

theorem part_a (n : ℕ) : ∃ θ_n : ℝ, 0 < θ_n ∧ θ_n < π / 2 ∧ a n = Real.cos θ_n := sorry

theorem part_b : (tendsto (fun n => 4^n * (1 - a n)) at_top (nhds (π^2 / 18))) := sorry

end part_a_part_b_l285_285287


namespace number_of_arrangements_l285_285382

theorem number_of_arrangements (V T : ℕ) (hV : V = 3) (hT : T = 4) :
  ∃ n : ℕ, n = 36 :=
by
  sorry

end number_of_arrangements_l285_285382


namespace trigonometric_identity_l285_285159

theorem trigonometric_identity :
  sin 68 * sin 67 - sin 23 * cos 68 = - (Real.sqrt 2) / 2 :=
by
  sorry

end trigonometric_identity_l285_285159


namespace max_value_frac_l285_285962
noncomputable section

open Real

variables (a b x y : ℝ)

theorem max_value_frac :
  a > 1 → b > 1 → 
  a^x = 2 → b^y = 2 →
  a + sqrt b = 4 →
  (2/x + 1/y) ≤ 4 :=
by
  intros ha hb hax hby hab
  sorry

end max_value_frac_l285_285962


namespace solve_for_x_l285_285420

theorem solve_for_x : ∃ x : ℕ, 289 + 2 * 17 * 4 + 16 = x ∧ x = 441 :=
by
  let x := 289 + 2 * 17 * 4 + 16
  have h : x = 441 := by
    calc
      x = 289 + 2 * 17 * 4 + 16 : by rfl
      ... = 289 + 136 + 16       : by norm_num
      ... = 441                 : by norm_num
  use x
  exact ⟨by rfl, h⟩

end solve_for_x_l285_285420


namespace simplify_cot_tan_l285_285709

theorem simplify_cot_tan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := by
  sorry

end simplify_cot_tan_l285_285709


namespace number_of_people_is_ten_l285_285691

-- Define the total number of Skittles and the number of Skittles per person.
def total_skittles : ℕ := 20
def skittles_per_person : ℕ := 2

-- Define the number of people as the total Skittles divided by the Skittles per person.
def number_of_people : ℕ := total_skittles / skittles_per_person

-- Theorem stating that the number of people is 10.
theorem number_of_people_is_ten : number_of_people = 10 := sorry

end number_of_people_is_ten_l285_285691


namespace expand_product_l285_285916

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 :=
by
  sorry

end expand_product_l285_285916


namespace mean_temperature_correct_l285_285765

def temperatures : List ℤ := [-6, -3, -3, -4, 2, 4, 1]

def mean_temperature (temps : List ℤ) : ℚ :=
  (temps.sum : ℚ) / temps.length

theorem mean_temperature_correct :
  mean_temperature temperatures = -9 / 7 := 
by
  sorry

end mean_temperature_correct_l285_285765


namespace lcm_18_30_is_90_l285_285033

noncomputable def LCM_of_18_and_30 : ℕ := 90

theorem lcm_18_30_is_90 : nat.lcm 18 30 = LCM_of_18_and_30 := 
by 
  -- definition of lcm should be used here eventually
  sorry

end lcm_18_30_is_90_l285_285033


namespace prob_absolute_value_less_than_196_l285_285689

variable (ξ : ℝ)
variable (h0 : ∀ x : ℝ, (∫ t in -∞..x, pdf (Normal 0 1) t) = P (λ x, x < a))
variable (h1 : P (λ x, x < -1.96) = 0.025)

theorem prob_absolute_value_less_than_196 :
  P (λ x, |ξ x| < 1.96) = 0.950 := sorry

end prob_absolute_value_less_than_196_l285_285689


namespace concert_revenue_l285_285858

variable {students non_students : ℕ}
variable {price_per_student_ticket price_per_non_student_ticket : ℕ}
variable {total_tickets total_student_tickets total_non_student_tickets : ℕ}

theorem concert_revenue 
  (total_tickets_sold : total_tickets = 150) 
  (students_tickets_price : price_per_student_ticket = 5) 
  (non_students_tickets_price : price_per_non_student_ticket = 8) 
  (students_tickets_count : total_student_tickets = 90) 
  (non_students_tickets_count : total_non_student_tickets = 60) 
  : (total_student_tickets * price_per_student_ticket + total_non_student_tickets * price_per_non_student_ticket = 930) :=
by
  rw [students_tickets_count, students_tickets_price]
  rw [non_students_tickets_count, non_students_tickets_price]
  norm_num
  exact congr_arg _ rfl

end concert_revenue_l285_285858


namespace no_nat_pairs_divisibility_l285_285141

theorem no_nat_pairs_divisibility (a b : ℕ) (hab : b^a ∣ a^b - 1) : false :=
sorry

end no_nat_pairs_divisibility_l285_285141


namespace problem_statement_l285_285607

theorem problem_statement (x y : ℝ) (h : (x - 1)^2 + |2y + 1| = 0) : x + y = 1 / 2 :=
sorry

end problem_statement_l285_285607


namespace only_A_forms_triangle_l285_285044

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem only_A_forms_triangle :
  (triangle_inequality 5 6 10) ∧ ¬(triangle_inequality 5 2 9) ∧ ¬(triangle_inequality 5 7 12) ∧ ¬(triangle_inequality 3 4 8) :=
by
  sorry

end only_A_forms_triangle_l285_285044


namespace multiply_polynomials_l285_285310

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l285_285310


namespace prove_a5_l285_285605

-- Definition of the conditions
def expansion (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) :=
  (x - 1) ^ 8 = a_0 + a_1 * (1 + x) + a_2 * (1 + x)^2 + a_3 * (1 + x)^3 + a_4 * (1 + x)^4 + 
               a_5 * (1 + x)^5 + a_6 * (1 + x)^6 + a_7 * (1 + x)^7 + a_8 * (1 + x)^8

-- Given condition
axiom condition (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : ∀ x : ℤ, expansion x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8

-- The target problem: proving a_5 = -448
theorem prove_a5 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : a_5 = -448 :=
by
  sorry

end prove_a5_l285_285605


namespace length_of_PS_l285_285264

theorem length_of_PS {O P Q R S : Type} [PlaneGeometry O P Q R S]
  (radius : ℝ)
  (diameter_semi: radius * 2 = 20)
  (PQ_length: ℝ)
  (PQ_condition: PQ_length = 16)
  (PQRS_is_rectangle: IsRectangle PQRS)
  (PQ_on_diameter: OnDiameter PQ)
  (R_S_on_semicircle: OnSemicircle R S) :
  Length PS = 6 :=
by
  sorry

end length_of_PS_l285_285264


namespace same_function_l285_285097

noncomputable def f (x : ℝ) : ℝ := x
noncomputable def g (t : ℝ) : ℝ := (t^3 + t) / (t^2 + 1)

theorem same_function : ∀ x : ℝ, f x = g x :=
by sorry

end same_function_l285_285097


namespace regular_polygon_sides_l285_285080

theorem regular_polygon_sides (O A B : Type) (angle_OAB : ℝ) 
  (h_angle : angle_OAB = 72) : 
  (360 / angle_OAB = 5) := 
by 
  sorry

end regular_polygon_sides_l285_285080


namespace a_n_integers_l285_285378

def sequence_a : ℕ → ℤ
| 0     := 1
| 1     := 1
| (n+2) := (1 + (sequence_a (n+1))^2) / (sequence_a n)

theorem a_n_integers : ∀ n : ℕ, ∃ k : ℤ, sequence_a n = k := by
  sorry

end a_n_integers_l285_285378


namespace lcm_of_18_and_30_l285_285027

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end lcm_of_18_and_30_l285_285027


namespace lcm_18_30_l285_285023

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l285_285023


namespace arrangements_count_l285_285539

-- Define the grid and conditions
def grid : Type := matrix (fin 4) (fin 4) (fin 4 → char)

def letters : fin 4 → char
| ⟨0,_⟩ := 'A'
| ⟨1,_⟩ := 'B'
| ⟨2,_⟩ := 'C'
| ⟨3,_⟩ := 'D'

def valid_arrangements (g : grid) : Prop :=
  (∀ i, ∀ j, ∃! k, g i k = letters j) ∧  -- Each letter appears exactly once in each row
  (∀ j, ∀ i, ∃! k, g k j = letters i)    -- Each letter appears exactly once in each column
  ∧ g ⟨0, by sorry⟩ ⟨0, by sorry⟩ = 'A'  -- 'A' is fixed in the top-left corner

def number_of_arrangements : nat := 72

theorem arrangements_count : ∃ n, valid_arrangements ∧ n = number_of_arrangements :=
by
  existsi 72
  sorry

end arrangements_count_l285_285539


namespace lcm_18_30_is_90_l285_285029

noncomputable def LCM_of_18_and_30 : ℕ := 90

theorem lcm_18_30_is_90 : nat.lcm 18 30 = LCM_of_18_and_30 := 
by 
  -- definition of lcm should be used here eventually
  sorry

end lcm_18_30_is_90_l285_285029


namespace ways_to_make_30_cents_is_17_l285_285343

-- Define the value of each type of coin
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Define the function that counts the number of ways to make 30 cents
def count_ways_to_make_30_cents : ℕ :=
  let ways_with_1_quarter := (if 30 - quarter_value == 5 then 2 else 0)
  let ways_with_0_quarters :=
    let ways_with_2_dimes := (if 30 - 2 * dime_value == 10 then 3 else 0)
    let ways_with_1_dime := (if 30 - dime_value == 20 then 5 else 0)
    let ways_with_0_dimes := (if 30 == 30 then 7 else 0)
    ways_with_2_dimes + ways_with_1_dime + ways_with_0_dimes
  2 + ways_with_1_quarter + ways_with_0_quarters

-- Proof statement
theorem ways_to_make_30_cents_is_17 : count_ways_to_make_30_cents = 17 := sorry

end ways_to_make_30_cents_is_17_l285_285343


namespace house_ordering_solution_l285_285808

def house_ordering_problem : Prop :=
  ∃ (sequence : List String),
    sequence.length = 5 ∧
    sequence.nodup ∧  -- All houses should be uniquely painted and patterned
    (∃ (idxR idxG idxB idxP idxY : ℕ),
      idxR < idxG ∧   -- Condition 1: Red stripes come before green dots
      idxY < idxB ∧   -- Condition 2a: Yellow zigzags come before blue waves
      idxB < idxP ∧   -- Condition 2b: Blue waves come before pink spirals
      (idxB ≠ idxP + 1 ∧ idxB ≠ idxP - 1) ∧  -- Condition 3a: Blue waves not next to pink spirals
      (idxB ≠ idxY + 1 ∧ idxB ≠ idxY - 1) ∧  -- Condition 3b: Blue waves not next to yellow zigzags
      sequence[idxR] = "R" ∧ sequence[idxG] = "G" ∧
      sequence[idxB] = "B" ∧ sequence[idxP] = "P" ∧ sequence[idxY] = "Y")

theorem house_ordering_solution : ∃ sequence : List String, house_ordering_problem ∧ sequence.length = 4 :=
by
  sorry

end house_ordering_solution_l285_285808


namespace AD_perpendicular_BC_l285_285346

variable {Point : Type} [EuclideanSpace Point] 
variables {A B C D BB1 CC1 BC AD : Point}
variables {tetrahedron : Point → Point → Point → Point → Prop}

-- Conditions
axiom altitudes_intersect : tetrahedron A B C D → ∃ X, X ∈ line B BB1 ∧ X ∈ line C CC1

-- Proof Statement
theorem AD_perpendicular_BC (h : tetrahedron A B C D) : altitudes_intersect h → AD ⟂ BC :=
by 
  sorry

end AD_perpendicular_BC_l285_285346


namespace find_a4_in_geometric_seq_l285_285265

variable {q : ℝ} -- q is the common ratio of the geometric sequence

noncomputable def geometric_seq (q : ℝ) (n : ℕ) : ℝ := 16 * q ^ (n - 1)

theorem find_a4_in_geometric_seq (h1 : geometric_seq q 1 = 16)
  (h2 : geometric_seq q 6 = 2 * geometric_seq q 5 * geometric_seq q 7) :
  geometric_seq q 4 = 2 := 
  sorry

end find_a4_in_geometric_seq_l285_285265


namespace relationship_among_a_b_c_l285_285546

noncomputable def a : ℝ := Real.sin (2 * Real.pi / 5)
noncomputable def b : ℝ := Real.cos (5 * Real.pi / 6)
noncomputable def c : ℝ := Real.tan (7 * Real.pi / 5)

theorem relationship_among_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_among_a_b_c_l285_285546


namespace partA_l285_285419

theorem partA (a b : ℝ) : (a - b) ^ 2 ≥ 0 → (a^2 + b^2) / 2 ≥ a * b := 
by
  intro h
  sorry

end partA_l285_285419


namespace problem_solution_l285_285682

theorem problem_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x * f y - x) = x * y - f x) :
  let n := {x : ℝ | ∃ y, f y = x}.to_finset.card,
      s := ({x : ℝ | ∃ y, f y = x}.to_finset.sum (λ x, x))
  in n * s = 0 :=
by
  sorry

end problem_solution_l285_285682


namespace age_ratio_l285_285076

theorem age_ratio (S : ℕ) (M : ℕ) (h1 : S = 28) (h2 : M = S + 30) : 
  ((M + 2) / (S + 2) = 2) := 
by
  sorry

end age_ratio_l285_285076


namespace bronson_yellow_leaves_l285_285877

-- Bronson collects 12 leaves on Thursday
def leaves_thursday : ℕ := 12

-- Bronson collects 13 leaves on Friday
def leaves_friday : ℕ := 13

-- 20% of the leaves are Brown (as a fraction)
def percent_brown : ℚ := 0.2

-- 20% of the leaves are Green (as a fraction)
def percent_green : ℚ := 0.2

theorem bronson_yellow_leaves : 
  (leaves_thursday + leaves_friday) * (1 - percent_brown - percent_green) = 15 := by
sorry

end bronson_yellow_leaves_l285_285877


namespace Jake_not_drop_coffee_l285_285654

theorem Jake_not_drop_coffee :
  (40% / 100) * (25% / 100) = 10% / 100 → 
  100% / 100 - 10% / 100 = 90% / 100 :=
begin
  sorry
end

end Jake_not_drop_coffee_l285_285654


namespace octagon_area_fraction_product_l285_285926

noncomputable def octagon_fraction_area_product (s : ℝ) : ℝ := 
  let r := s * Real.sqrt 2 / 2
  let total_octagon_area := 8 * r^2 * Real.sqrt 2
  let fido_reach_area := Real.pi * r^2
  let fraction := fido_reach_area / total_octagon_area
  let simplified_fraction := fraction * (Real.sqrt 2 / Real.sqrt 2)
  a = 2
  b = 16
  ab = a * b
  ab

theorem octagon_area_fraction_product (s : ℝ) : octagon_fraction_area_product s = 32 :=
  by
    sorry

end octagon_area_fraction_product_l285_285926


namespace hexagon_planting_schemes_l285_285447

theorem hexagon_planting_schemes :
  let A := 0
  let B := 1
  let C := 2
  let D := 3
  let E := 4
  let F := 5
  let regions := {A, B, C, D, E, F}
  let number_of_plants := 4
  let adjacency := [(A, B), (B, C), (C, D), (D, E), (E, F), (F, A), (A, C), (C, E), (E, A)]
  (∀ region1 region2 ∈ regions, region1 ≠ region2 ∧ (region1, region2) ∈ adjacency -> plant(region1) ≠ plant(region2)) →
  ∃ planting_scheme, planting_scheme = 732 :=
sorry

end hexagon_planting_schemes_l285_285447


namespace determine_h_l285_285117

theorem determine_h (x : ℝ) (h : ℝ → ℝ) :
  2 * x ^ 5 + 4 * x ^ 3 + h x = 7 * x ^ 3 - 5 * x ^ 2 + 9 * x + 3 →
  h x = -2 * x ^ 5 + 3 * x ^ 3 - 5 * x ^ 2 + 9 * x + 3 :=
by
  intro h_eq
  sorry

end determine_h_l285_285117


namespace max_p_solution_l285_285125

theorem max_p_solution :
  ∃ (x ∈ ℝ), ∀ (p : ℝ), (2 * cos (2 * real.pi - real.pi * x^2 / 6) * cos (real.pi / 3 * sqrt (9 - x^2)) - 3 =
                       p - 2 * sin (-real.pi * x^2 / 6) * cos (real.pi / 3 * sqrt (9 - x^2))) → 
                       p ≤ -2 := 
sorry

end max_p_solution_l285_285125


namespace solve_for_x_l285_285168

theorem solve_for_x : ∃ x : ℝ, 10^(2 * x) * 100^x = 1000^4 ∧ x = 3 := by
  sorry

end solve_for_x_l285_285168


namespace find_eigenvalues_l285_285931

open Matrix

noncomputable theory

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 8, 0], ![2, 1, 2], ![0, 2, 1]]

theorem find_eigenvalues : (∃ v : Fin 3 → ℝ, v ≠ 0 ∧ A.mul_vec v = k • v) ↔ (k = 9 ∨ k = -7 ∨ k = 3 ∨ k = -1) := 
sorry

end find_eigenvalues_l285_285931


namespace solution_set_of_inequality_l285_285900

noncomputable def f : ℝ → ℝ := sorry

axiom f_continuous : ∀ x : ℝ, continuous_at f x
axiom f_even_sum : ∀ x : ℝ, f(x) + f(-x) = x^2
axiom f_deriv_lessthan_x : ∀ x : ℝ, x < 0 → (deriv f x) < x

theorem solution_set_of_inequality :
  {x : ℝ | f(x) - f(1 - x) ≥ x - 1/2} = set.Iic (1/2) :=
begin
  sorry
end

end solution_set_of_inequality_l285_285900


namespace lcm_18_30_is_90_l285_285034

theorem lcm_18_30_is_90 : Nat.lcm 18 30 = 90 := 
by 
  unfold Nat.lcm
  sorry

end lcm_18_30_is_90_l285_285034


namespace factor_evaluate_l285_285136

theorem factor_evaluate (a b : ℤ) (h1 : a = 2) (h2 : b = -2) : 
  5 * a * (b - 2) + 2 * a * (2 - b) = -24 := by
  sorry

end factor_evaluate_l285_285136


namespace ratio_of_heights_eq_three_twentieths_l285_285081

noncomputable def base_circumference : ℝ := 32 * Real.pi
noncomputable def original_height : ℝ := 60
noncomputable def shorter_volume : ℝ := 768 * Real.pi

theorem ratio_of_heights_eq_three_twentieths
  (base_circumference : ℝ)
  (original_height : ℝ)
  (shorter_volume : ℝ)
  (h' : ℝ)
  (ratio : ℝ) :
  base_circumference = 32 * Real.pi →
  original_height = 60 →
  shorter_volume = 768 * Real.pi →
  (1 / 3 * Real.pi * (base_circumference / (2 * Real.pi))^2 * h') = shorter_volume →
  ratio = h' / original_height →
  ratio = 3 / 20 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end ratio_of_heights_eq_three_twentieths_l285_285081


namespace basketball_win_requirement_l285_285427

theorem basketball_win_requirement (games_played : ℕ) (games_won : ℕ) (games_left : ℕ) (total_games : ℕ) (required_win_percentage : ℚ) : 
  games_played = 50 → games_won = 35 → games_left = 25 → total_games = games_played + games_left → required_win_percentage = 64 / 100 → 
  ∃ games_needed_to_win : ℕ, games_needed_to_win = (required_win_percentage * total_games).natCeil - games_won ∧ games_needed_to_win = 13 :=
by
  sorry

end basketball_win_requirement_l285_285427


namespace cot20_plus_tan10_eq_csc20_l285_285728

theorem cot20_plus_tan10_eq_csc20 : Real.cot 20 * Real.pi / 180 + Real.tan 10 * Real.pi / 180 = Real.csc 20 * Real.pi / 180 := 
by
  sorry

end cot20_plus_tan10_eq_csc20_l285_285728


namespace area_of_PQRS_leq_quarter_area_of_ABCD_l285_285558

variables {A B C D Q S P R : Type*}
variables [trapezoid ABCD] [point_on_base Q BC] [point_on_base S AD]
variables [intersects_at AQ BS P] [intersects_at CS DQ R]

theorem area_of_PQRS_leq_quarter_area_of_ABCD : 
  S_{PQRS} ≤ (1 / 4) * S_{ABCD} :=
sorry

end area_of_PQRS_leq_quarter_area_of_ABCD_l285_285558


namespace number_of_answer_choices_l285_285854

theorem number_of_answer_choices (n : ℕ) (H1 : (n + 1)^4 = 625) : n = 4 :=
sorry

end number_of_answer_choices_l285_285854


namespace num_congruent_1_mod_9_l285_285391

theorem num_congruent_1_mod_9 : {n ∈ finset.range 1000 | n % 9 = 1}.card = 112 := by
  sorry

end num_congruent_1_mod_9_l285_285391


namespace max_grid_sum_l285_285138

-- Definition of valid grids
def is_valid_grid (grid : List (List ℕ)) : Prop :=
  grid.length = 3 ∧ (∀ row, row ∈ grid → row.length = 3) ∧
  (∀ row, ∀ x, x ∈ row → x ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (List.join grid).nodup

-- Definition of the sum of numbers in the grid
def grid_sum (grid : List (List ℕ)) : ℕ :=
  grid.foldr (λ row acc, acc + (row.foldr (λ x acc, acc * 10 + x) 0)) 0

-- The main theorem to prove
theorem max_grid_sum : ∃ grid, is_valid_grid grid ∧ grid_sum grid = 3972 :=
  sorry

end max_grid_sum_l285_285138


namespace distance_traveled_in_second_part_l285_285461

theorem distance_traveled_in_second_part
  (d1 : ℝ) (t1 t2 : ℝ) (v : ℝ) (total_distance : ℝ) (distance_second : ℝ) :
  d1 = 225 → t1 = 3.5 → t2 = 5 → v = 70 → total_distance = v * (t1 + t2) →
  distance_second = total_distance - d1 →
  distance_second = 370 := by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at *
  have : total_distance = 70 * 8.5, by
    simp [h5]
  have : distance_second = total_distance - 225, by
    simp [h6]
  have : distance_second = 595 - 225, by
    rw this
  have : distance_second = 370, by
    norm_num at this
  exact this

end distance_traveled_in_second_part_l285_285461


namespace present_age_ratio_l285_285376

-- Define the conditions as functions in Lean.
def age_difference (M R : ℝ) : Prop := M - R = 7.5
def future_age_ratio (M R : ℝ) : Prop := (R + 10) / (M + 10) = 2 / 3

-- Define the goal as a proof problem in Lean.
theorem present_age_ratio (M R : ℝ) 
  (h1 : age_difference M R) 
  (h2 : future_age_ratio M R) : 
  R / M = 2 / 5 := 
by 
  sorry  -- Proof to be completed

end present_age_ratio_l285_285376


namespace largest_n_inequality_l285_285935

theorem largest_n_inequality :
  ∃ n : ℕ, (∀ a b c : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 →
    ((a + b + c) / (a * b * c + 1) + (a * b * c)^(1 / n) ≤ 5 / 2)) ∧
    ∀ m : ℕ, m > n →
    ∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧
    ((a + b + c) / (a * b * c + 1) + (a * b * c)^(1 / m) > 5 / 2)) :=
begin
  use 3,
  sorry
end

end largest_n_inequality_l285_285935


namespace max_f_on_interval_increasing_interval_of_f_l285_285818

noncomputable def f (x a : ℝ) : ℝ := -x^2 + x + a * (x^2 - 1)

theorem max_f_on_interval (a : ℝ) (h : a = 1) : ∀ x ∈ set.Icc (1:ℝ) (4:ℝ), f x a ≤ 10 / 3 :=
by {
  intro x,
  intro hx,
  sorry
}

theorem increasing_interval_of_f (a : ℝ) (h : a > -1/9) : ∀ x > 2/3, (f' x a > 0) :=
by {
  intro x,
  intro hx,
  sorry
}

end max_f_on_interval_increasing_interval_of_f_l285_285818


namespace find_length_C_l285_285834

noncomputable def length_C'B (AC BC : ℝ) (angle_C : ℝ) : ℝ :=
  if AC = BC ∧ AC = sqrt 2 ∧ angle_C = 90 then sqrt 3 - 1 else 0

theorem find_length_C'B (AC BC : ℝ) (angle_C : ℝ) (angle_BAC : ℝ) (AB : ℝ)
  (triangle_rotation : ℝ) (BB' : ℝ) (BD : ℝ) (C'D : ℝ) : 
  (AC = sqrt 2) ∧ (BC = sqrt 2) ∧ (angle_C = 90) ∧ (angle_BAC = 45) ∧ 
  (AB = 2) ∧ (triangle_rotation = 60) ∧ (BB' = 2) ∧ 
  (BD = sqrt 3) ∧ (C'D = 1) → length_C'B AC BC angle_C = sqrt 3 - 1 :=
begin
  intros,
  sorry
end

end find_length_C_l285_285834


namespace equal_areas_l285_285285

variable (ABC : Type) [IsAcuteTriangle ABC] 
variable {A B C T O : ABC}
variable [HasFootOfAltitude T]
variable [IsCircumcenter O]

theorem equal_areas (h1 : AC > BC) : 
    area (quad AT O C) = area (quad BT O C) :=
sorry

end equal_areas_l285_285285


namespace num_common_elements_in_sequences_l285_285597

theorem num_common_elements_in_sequences
  (U: Finset ℕ := Finset.image (λ n, 2*n) (Finset.range 150))
  (V: Finset ℕ := Finset.image (λ m, 3*m) (Finset.range 150)):
  U.filter (λ x, x ∈ V).card = 25 :=
by
  sorry

end num_common_elements_in_sequences_l285_285597


namespace antiderivative_comp_l285_285680

variables {a b p q : ℝ}
variables {f F : ℝ → ℝ} {φ : ℝ → ℝ}

-- Conditions
noncomputable def is_antiderivative (F f : ℝ → ℝ) (a b : ℝ) := ∀ x ∈ set.Icc a b, has_deriv_at F (f x) x
noncomputable def is_differentiable_on (φ : ℝ → ℝ) (p q : ℝ) := ∀ y ∈ set.Icc p q, differentiable_at ℝ φ y
noncomputable def image_in_interval (φ : ℝ → ℝ) (a b : ℝ) (p q : ℝ) := ∀ y ∈ set.Icc p q, φ y ∈ set.Icc a b
noncomputable def injective_in_neigh (φ : ℝ → ℝ) (p q : ℝ) := 
  ∀ y₀ ∈ set.Icc p q, ∃ U : set ℝ, is_open U ∧ y₀ ∈ U ∧ ∀ y ∈ U, y ≠ y₀ → φ y ≠ φ y₀

-- Mathematically equivalent proof problem
theorem antiderivative_comp {a b p q : ℝ} {f F : ℝ → ℝ} {φ : ℝ → ℝ}
  (h₁ : is_antiderivative F f a b)
  (h₂ : is_differentiable_on φ p q)
  (h₃ : image_in_interval φ a b p q)
  (h₄ : injective_in_neigh φ p q) :
  is_antiderivative (λ y, F (φ y)) (λ y, (f (φ y)) * (deriv φ y)) p q :=
  sorry

end antiderivative_comp_l285_285680


namespace shopkeeper_revenue_l285_285084

variable (T : ℕ) (revenue : ℕ)

theorem shopkeeper_revenue : 
  (0.70 * T = 210) → 
  revenue = 2 * (T - 210) → 
  revenue = 180 :=
by
  intros h₁ h₂
  have h₃ : T = 300 := by
    linarith
  rw h₃ at h₂
  linarith

#check shopkeeper_revenue

end shopkeeper_revenue_l285_285084


namespace find_f_neg_eight_l285_285165

-- Conditions based on the given problem
variable (f : ℤ → ℤ)
axiom func_property : ∀ x y : ℤ, f (x + y) = f x + f y + x * y + 1
axiom f1_is_one : f 1 = 1

-- Main theorem
theorem find_f_neg_eight : f (-8) = 19 := by
  sorry

end find_f_neg_eight_l285_285165


namespace find_other_polynomial_l285_285211

variables {a b c d : ℤ}

theorem find_other_polynomial (h : ∀ P Q : ℤ, P - Q = c^2 * d^2 - a^2 * b^2) 
  (P : ℤ) (hP : P = a^2 * b^2 + c^2 * d^2 - 2 * a * b * c * d) : 
  (∃ Q : ℤ, Q = 2 * c^2 * d^2 - 2 * a * b * c * d) ∨ 
  (∃ Q : ℤ, Q = 2 * a^2 * b^2 - 2 * a * b * c * d) :=
by {
  sorry
}

end find_other_polynomial_l285_285211


namespace desired_gold_percentage_l285_285099

def original_alloy_weight : ℝ := 48 -- ounces
def original_gold_percentage : ℝ := 0.25 -- 25%
def added_gold_weight : ℝ := 12 -- ounces

theorem desired_gold_percentage :
  let original_gold_weight := original_alloy_weight * original_gold_percentage,
      total_gold_weight := original_gold_weight + added_gold_weight,
      new_alloy_weight := original_alloy_weight + added_gold_weight,
      gold_percentage := (total_gold_weight / new_alloy_weight) * 100
  in gold_percentage = 40 := 
by
  sorry

end desired_gold_percentage_l285_285099


namespace sum_of_reversed_base_numbers_eq_118_l285_285943

def sum_of_reversed_base_numbers (d : ℕ) := 
  ∑ n in { n | 
    let digits_9 := (nat.digits 9 n) in 
    let digits_12 := (nat.digits 12 n) in 
    digits_9 = digits_12.reverse ∧
    n > 0 }, id

theorem sum_of_reversed_base_numbers_eq_118 : 
  sum_of_reversed_base_numbers 2 = 118 :=
by 
  -- proof is left as a sorry placeholder as we are focusing on the statement
  sorry

end sum_of_reversed_base_numbers_eq_118_l285_285943


namespace lcm_18_30_is_90_l285_285037

theorem lcm_18_30_is_90 : Nat.lcm 18 30 = 90 := 
by 
  unfold Nat.lcm
  sorry

end lcm_18_30_is_90_l285_285037


namespace simplify_cot_tan_l285_285715

theorem simplify_cot_tan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := by
  sorry

end simplify_cot_tan_l285_285715


namespace lap_time_improvement_l285_285657

theorem lap_time_improvement (initial_laps initial_time current_laps current_time : ℕ)
  (h1 : initial_laps = 15) 
  (h2 : initial_time = 40) 
  (h3 : current_laps = 18) 
  (h4 : current_time = 36):
  (initial_time / initial_laps - current_time / current_laps : ℚ) = 2 / 3 := by
  have initial_lap_time := (initial_time : ℚ) / initial_laps
  have current_lap_time := (current_time : ℚ) / current_laps
  have improvement := initial_lap_time - current_lap_time
  have h5 : initial_lap_time = 40 / 15 := by sorry
  have h6 : current_lap_time = 36 / 18 := by sorry
  have h7 : improvement = (40 / 15) - (36 / 18) := by sorry
  have h8 : (40 / 15) - (36 / 18) = 2 / 3 := by sorry
  show (initial_time / initial_laps - current_time / current_laps : ℚ) = 2 / 3 from sorry

end lap_time_improvement_l285_285657


namespace b_alone_work_time_l285_285048

def work_rate_combined (a_rate b_rate : ℝ) : ℝ := a_rate + b_rate

theorem b_alone_work_time
  (a_rate b_rate : ℝ)
  (h1 : work_rate_combined a_rate b_rate = 1/16)
  (h2 : a_rate = 1/20) :
  b_rate = 1/80 := by
  sorry

end b_alone_work_time_l285_285048


namespace factorial_division_l285_285503
-- Definition of factorial
def fact : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * fact n

-- Problem statement
theorem factorial_division : fact 12 / fact 11 = 12 :=
by sorry

end factorial_division_l285_285503


namespace cot20_plus_tan10_eq_csc20_l285_285729

theorem cot20_plus_tan10_eq_csc20 : Real.cot 20 * Real.pi / 180 + Real.tan 10 * Real.pi / 180 = Real.csc 20 * Real.pi / 180 := 
by
  sorry

end cot20_plus_tan10_eq_csc20_l285_285729


namespace sum_y_coords_l285_285886

theorem sum_y_coords (h1 : ∃(y : ℝ), (0 + 3)^2 + (y - 5)^2 = 64) : 
  ∃ y1 y2 : ℝ, y1 + y2 = 10 ∧ (0, y1) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) ∧ 
                            (0, y2) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) := 
by
  sorry

end sum_y_coords_l285_285886


namespace select_group_of_100_l285_285470

/-- Given 198 cats each with a unique weight and unique speed, 
    where the heaviest cat in any group of 100 cats is also the fastest,
    prove that we can form a group of 100 cats such that the lightest cat is also the slowest. -/
theorem select_group_of_100 (cats : Finset (ℕ × ℕ)) (h_size : cats.card = 198)
  (unique_weights : ∀ {w1 w2 s1 s2 : ℕ}, (w1, s1) ∈ cats → (w2, s2) ∈ cats → w1 = w2 → s1 = s2 ∧ (w1, s1) = (w2, s2))
  (heaviest_is_fastest : ∀ (group : Finset (ℕ × ℕ)), group.card = 100 → ∃ (heaviest : (ℕ × ℕ)), heaviest ∈ group ∧ 
    (∀ (cat ∈ group), heaviest.fst < cat.fst → heaviest.snd < cat.snd)) :
  ∃ group_100 : Finset (ℕ × ℕ), group_100.card = 100 ∧
    let lightest := Finset.min' group_100 ⟨0, 0⟩ in
    ∀ cat ∈ group_100, lightest.fst > cat.fst → lightest.snd > cat.snd :=
by
  sorry

end select_group_of_100_l285_285470


namespace magnitude_angle_DAB_l285_285839

-- Given conditions
variables {A B C D : Type}
variables [affine_space ℝ {Triangle}]
variables (tri_ABC : Triangle A B C) (tri_BAD : Triangle B A D)

-- Define the properties of the triangles based on given conditions
def right_angle_at_B (tri_BAD : Triangle B A D) : Prop :=
∠ BAD = 90

def midpoint_C (A D C : Point) : Prop := 
dist A C = dist C D

def equal_sides (A B C : Point) : Prop :=
dist A B = dist B C

-- The theorem to prove
theorem magnitude_angle_DAB 
(triangle_BAD: Triangle) 
(right_angle_at_B: right_angle_at_B triangle_BAD)
(midpoint_C: midpoint_C A D C)
(equal_sides: equal_sides A B C): 
∠ DAB = 60 := 
sorry

end magnitude_angle_DAB_l285_285839


namespace starting_point_for_drawing_l285_285418

-- Define vertices and their degrees
structure GraphVertex :=
  (label : String)
  (degree : Nat)

-- Define the problem
def graph_vertices : List GraphVertex := [
  {label := "T", degree := 2},
  {label := "Q", degree := 2},
  {label := "P", degree := 3},
  {label := "R", degree := 3},
  {label := "S", degree := 3}
]

-- Define the statement of the problem
theorem starting_point_for_drawing (v1 v2 : String) :
  (v1 = "R" ∧ v2 = "S") ∨ (v1 = "S" ∧ v2 = "R") :=
begin
  sorry
end

end starting_point_for_drawing_l285_285418


namespace range_of_k_intersecting_AB_l285_285639

theorem range_of_k_intersecting_AB 
  (A B : ℝ × ℝ) 
  (hA : A = (2, 7)) 
  (hB : B = (9, 6)) 
  (k : ℝ) 
  (hk : k ≠ 0) 
  (H : ∃ x : ℝ, A.2 = k * A.1 ∧ B.2 = k * B.1):
  (2 / 3) ≤ k ∧ k ≤ 7 / 2 :=
by sorry

end range_of_k_intersecting_AB_l285_285639


namespace min_pairs_acquaintances_l285_285269

def residents := 200

def can_be_strangely_seated (R : Finset (Fin residents)) : Prop :=
  R.card = 6 ∧ ∀ (A B : Fin residents) (hA : A ∈ R) (hB : B ∈ R), 
    ∃ (x y : R), x ≠ y ∧ A = x ∧ B = y ∨ B = x ∧ A = y

theorem min_pairs_acquaintances (H : 
  ∀ R : Finset (Fin residents), R.card = 6 → 
  ∃ (A_1 A_2 A_3 A_4 A_5 A_6 : Fin residents), 
    {A_1, A_2, A_3, A_4, A_5, A_6} = R ∧ 
    (∃ (N_1 N_2 N_3 N_4 N_5 N_6 : Fin residents), 
      (N_1 = A_1 ∧ N_2 = A_2 ∧ N_3 = A_3 ∧ N_4 = A_4 ∧ N_5 = A_5 ∧ N_6 = A_6))) :
  ∃ p : ℕ, p = 19600 :=
begin
  sorry
end

end min_pairs_acquaintances_l285_285269


namespace Jake_not_drop_coffee_l285_285655

theorem Jake_not_drop_coffee :
  (40% / 100) * (25% / 100) = 10% / 100 → 
  100% / 100 - 10% / 100 = 90% / 100 :=
begin
  sorry
end

end Jake_not_drop_coffee_l285_285655


namespace ratio_of_perimeters_l285_285832

theorem ratio_of_perimeters (s : ℝ) :
  let d := s * Real.sqrt 2 in
  let d' := 11 * d in
  let P1 := 4 * s in
  let P2 := 4 * (d' / Real.sqrt 2) in
  P2 / P1 = 11 :=
by
  sorry

end ratio_of_perimeters_l285_285832


namespace badgers_win_at_least_five_games_l285_285769

-- Define the problem conditions and the required probability calculation
theorem badgers_win_at_least_five_games :
  let p := 0.5 in
  let n := 9 in
  let probability_at_least_five_wins :=
    ∑ k in Finset.range (n + 1), if k >= 5 then (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) else 0
  in
  probability_at_least_five_wins = 1 / 2 :=
by
  sorry

end badgers_win_at_least_five_games_l285_285769


namespace total_distance_is_correct_l285_285431

noncomputable def boat_speed : ℝ := 20 -- boat speed in still water (km/hr)
noncomputable def current_speed_first : ℝ := 5 -- current speed for the first 6 minutes (km/hr)
noncomputable def current_speed_second : ℝ := 8 -- current speed for the next 6 minutes (km/hr)
noncomputable def current_speed_third : ℝ := 3 -- current speed for the last 6 minutes (km/hr)
noncomputable def time_in_hours : ℝ := 6 / 60 -- 6 minutes in hours (0.1 hours)

noncomputable def total_distance_downstream := 
  (boat_speed + current_speed_first) * time_in_hours +
  (boat_speed + current_speed_second) * time_in_hours +
  (boat_speed + current_speed_third) * time_in_hours

theorem total_distance_is_correct : total_distance_downstream = 7.6 :=
  by 
  sorry

end total_distance_is_correct_l285_285431


namespace number_of_solutions_of_sine_eq_third_to_x_l285_285152

open Real

theorem number_of_solutions_of_sine_eq_third_to_x : 
  (set.Icc 0 (200 * π)).countable.count (λ x, sin x = (1/3)^x) = 200 :=
by
  sorry

end number_of_solutions_of_sine_eq_third_to_x_l285_285152


namespace smallest_add_to_multiple_of_4_l285_285813

theorem smallest_add_to_multiple_of_4 : ∃ n : ℕ, n > 0 ∧ (587 + n) % 4 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (587 + m) % 4 = 0 → n ≤ m :=
  sorry

end smallest_add_to_multiple_of_4_l285_285813


namespace find_k_l285_285614

-- Define the problem with given conditions
theorem find_k (k : ℤ) : 
  (2 * real.sqrt (325 + k) = real.sqrt (49 + k) + real.sqrt (784 + k)) → 
  k = 44 := 
  by
  sorry

end find_k_l285_285614


namespace problem1_problem2_l285_285840

open Real

-- (1)
theorem problem1 (x y : ℝ) (hx : x = log 2) (hy : y = log 5) : 
    x^2 + x * y + y = 1 := by sorry

-- (2)
theorem problem2 (a b c d e f : ℝ) 
  (ha : a = 2^(1/3)) 
  (hb : b = 3^(1/2)) 
  (hc : c = 8) 
  (hd : d = 16 / 49) 
  (he : e = 2^(1/4)) 
  (hf : f = -2016) :
  (a * b)^6 - c * d^(-1/2) - e * 8^(1/4) - f^0 = 91 := by sorry

end problem1_problem2_l285_285840


namespace compute_Mu_l285_285181

variable (M : Matrix (Fin 3) (Fin 3) ℤ)
variable (i j k u : Fin 3 → ℤ)
variable h1 : M.mulVec i = ![-1, 4, 9]
variable h2 : M.mulVec j = ![6, 2, -3]
variable h3 : M.mulVec k = ![0, -7, 5]
variable hu : u = ![1, 2, -1]

theorem compute_Mu : M.mulVec u = ![11, 15, -2] :=
by
  -- h1 : M.mulVec i = ![-1, 4, 9]
  -- h2 : M.mulVec j = ![6, 2, -3]
  -- h3 : M.mulVec k = ![0, -7, 5]
  -- hu : u = ![1, 2, -1]
  sorry

end compute_Mu_l285_285181


namespace car_distance_covered_by_car_l285_285065

theorem car_distance_covered_by_car
  (V : ℝ)                               -- Initial speed of the car
  (D : ℝ)                               -- Distance covered by the car
  (h1 : D = V * 6)                      -- The car takes 6 hours to cover the distance at speed V
  (h2 : D = 56 * 9)                     -- The car takes 9 hours to cover the distance at speed 56
  : D = 504 :=                          -- Prove that the distance D is 504 kilometers
by
  sorry

end car_distance_covered_by_car_l285_285065


namespace trig_ratio_calc_l285_285108

theorem trig_ratio_calc :
  let deg := 330
  let expr := λ (deg : ℝ) (pi : ℝ): ℝ,
    (Real.sin (deg * Real.pi / 180) * Real.tan (-13 / 3 * pi)) /
    (Real.cos (-19 / 6 * pi) * Real.cos (690 * Real.pi / 180))
  let sin_30 := 1 / 2
  let tan_pi_over_3 := Real.sqrt 3
  let cos_pi_over_6 := Real.sqrt 3 / 2
  expr deg Real.pi = - (2 * Real.sqrt 3 / 3) := by
  let sin_330 := -sin_30
  let tan_neg_13pi_over_3 := -tan_pi_over_3
  let cos_neg_19pi_over_6 := cos_pi_over_6
  let cos_690 := cos_pi_over_6
  sorry

end trig_ratio_calc_l285_285108


namespace star_equation_l285_285517

def star (X Y : ℝ) : ℝ := (X + Y) / 4

theorem star_equation : star (star 3 9) 4 = 7 / 4 :=
by
  sorry

end star_equation_l285_285517


namespace number_of_sets_summing_to_150_l285_285372

-- Define the conditions
def sum_consecutive_integers (a n : ℕ) : ℕ :=
  n * (2 * a + n - 1) / 2

def is_valid_set (a n : ℕ) : Prop :=
  sum_consecutive_integers a n = 150 ∧ a > 0

def count_valid_sets : ℕ :=
  {n | ∃ a, is_valid_set a n ∧ n > 1}.toFinset.card

-- Statement of the proof problem
theorem number_of_sets_summing_to_150 : count_valid_sets = 3 := 
sorry

end number_of_sets_summing_to_150_l285_285372


namespace badgers_win_at_least_five_games_prob_l285_285767

noncomputable def probability_Badgers_win_at_least_five_games : ℚ :=
  let p : ℚ := 1 / 2
  let n : ℕ := 9
  (1 / 2)^n * ∑ k in finset.range (n + 1), if k >= 5 then (nat.choose n k : ℚ) else 0

theorem badgers_win_at_least_five_games_prob :
  probability_Badgers_win_at_least_five_games = 1 / 2 :=
by sorry

end badgers_win_at_least_five_games_prob_l285_285767


namespace m_gt_zero_sufficient_not_necessary_l285_285293

-- Define the line l
def line (m : ℝ) : affine ! ℝ :=
  λ x : ℝ, m * x + 1

-- Define the circle C
def circle : ℝ × ℝ → Prop :=
  λ p, p.1 ^ 2 + p.2 ^ 2 = 1

-- Prove the statement
theorem m_gt_zero_sufficient_not_necessary (m : ℝ) :
  (∃ x y : ℝ, circle (x, y) ∧ y = line m x) ↔ (m > 0) :=
sorry

end m_gt_zero_sufficient_not_necessary_l285_285293


namespace largest_prime_divisor_l285_285811

-- Definition of prime number.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Given conditions (facts) of the problem.
def divisors_of_462 : set ℕ := {1, 2, 3, 6, 7, 14, 21, 33, 42, 66, 77, 154, 231, 462}
def divisors_of_385 : set ℕ := {1, 5, 7, 11, 35, 55, 77, 385}

-- The theorem that needs to be proved.
theorem largest_prime_divisor : ∃ d, d ∣ 462 ∧ d ∣ 385 ∧ is_prime d ∧ ∀ x, x ∣ 462 ∧ x ∣ 385 ∧ is_prime x → x ≤ d ∧ d = 7 :=
sorry

end largest_prime_divisor_l285_285811


namespace expand_binomials_l285_285923

theorem expand_binomials : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by 
  sorry

end expand_binomials_l285_285923


namespace number_of_n_mod_5_multiples_l285_285952

theorem number_of_n_mod_5_multiples:
  (Finset.card {n : ℕ | 3 ≤ n ∧ n ≤ 100 ∧ (7 + 2 * n + n^2 + 3 * n^3 + 4 * n^4 + 2 * n^5) % 5 = 0} = 19) := 
sorry

end number_of_n_mod_5_multiples_l285_285952


namespace revenue_fall_percentage_l285_285830

noncomputable def old_revenue : Float := 69.0
noncomputable def new_revenue : Float := 48.0

theorem revenue_fall_percentage :
  ((old_revenue - new_revenue) / old_revenue) * 100 ≈ 30.43 := by
  sorry

end revenue_fall_percentage_l285_285830


namespace evaluate_expression_l285_285522

theorem evaluate_expression : (16 : ℝ) ^ (-(2 ^ (-3) : ℝ)) = 1 / real.sqrt (2 : ℝ) := 
sorry

end evaluate_expression_l285_285522


namespace f_at_zero_f_negative_abs_f_monotonic_xf_less_than_zero_l285_285988

noncomputable theory

section
variables {f : ℝ → ℝ}

-- Condition: f(x) is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

-- Condition: When x > 0, f(x) = x - 6 / (x + 1)
def f_positive_defined (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f x = x - 6 / (x + 1)

-- Prove: f(0) = 0
theorem f_at_zero (hf : odd_function f) (hfp : f_positive_defined f) : f 0 = 0 :=
sorry

-- Prove: ∀ x < 0, f(x) = x + 6 / (1 - x)
theorem f_negative (hf : odd_function f) (hfp : f_positive_defined f) :
  ∀ x < 0, f x = x + 6 / (1 - x) :=
sorry

-- Prove: |f(x)| is monotonically increasing on [-2, 0) and [2, +∞)
theorem abs_f_monotonic (hf : odd_function f) (hfp : f_positive_defined f) :
  (∀ x ∈ Ico (-2 : ℝ) 0, abs (f x) ≤ abs (f (x + 1))) ∧ (∀ x ∈ Ioo 2 (⊤ : ℝ), abs (f x) ≤ abs (f (x + 1))) :=
sorry

-- Prove: The solution set of x * f(x) < 0 is (-2, 0) ∪ (0, 2)
theorem xf_less_than_zero (hf : odd_function f) (hfp : f_positive_defined f) :
  ∀ x, x * f x < 0 ↔ x ∈ Ioo (-2 : ℝ) 0 ∨ x ∈ Ioo 0 2 :=
sorry

end

end f_at_zero_f_negative_abs_f_monotonic_xf_less_than_zero_l285_285988


namespace nat_numbers_representation_l285_285118

theorem nat_numbers_representation (n : ℕ) (h : n ≥ 2) (k : ℕ) (d : ℕ) 
  (h1 : k = Nat.find (Nat.exists_factor_two_mul (exists:= n.exists_dvd_ne_one)))
   (h2 : d ∣ n) (h3 : n = k^2 + d^2) : n = 8 ∨ n = 20 :=
by
  sorry

end nat_numbers_representation_l285_285118


namespace quadratic_inequality_solution_l285_285535

theorem quadratic_inequality_solution (x : ℝ) : 
    (x^2 - 3*x - 4 > 0) ↔ (x < -1 ∨ x > 4) :=
sorry

end quadratic_inequality_solution_l285_285535


namespace parabola_intersects_x_axis_at_1_0_l285_285352

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := x^2 - 2 * x + 1

-- The proof statement checking the intersection with the x-axis at point (1, 0)
theorem parabola_intersects_x_axis_at_1_0 :
  ∃ x : ℝ, parabola x = 0 ∧ x = 1 :=
begin
  use 1,
  split,
  { simp [parabola], },
  { refl, }
end

end parabola_intersects_x_axis_at_1_0_l285_285352


namespace sum_y_coords_l285_285887

theorem sum_y_coords (h1 : ∃(y : ℝ), (0 + 3)^2 + (y - 5)^2 = 64) : 
  ∃ y1 y2 : ℝ, y1 + y2 = 10 ∧ (0, y1) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) ∧ 
                            (0, y2) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) := 
by
  sorry

end sum_y_coords_l285_285887


namespace solution_set_of_inequality_l285_285180

noncomputable def f : ℝ → ℝ := sorry

axiom ax1 : ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → (x1 ≠ x2) → 
  (x1 * f x2 - x2 * f x1) / (x2 - x1) > 1

axiom ax2 : f 3 = 2

theorem solution_set_of_inequality :
  {x : ℝ | 0 < x ∧ f x < x - 1} = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end solution_set_of_inequality_l285_285180


namespace intersection_sets_l285_285593

def M := {1, 2, 3}
def N := {1, 3, 4}

theorem intersection_sets (M N : set ℕ) : M ∩ N = {1, 3} := by
  sorry

-- Providing the instance for the theorem applied on specific sets M and N
example : ({1, 2, 3} : set ℕ) ∩ ({1, 3, 4} : set ℕ) = {1, 3} := by
  exact intersection_sets M N

end intersection_sets_l285_285593


namespace f_zero_count_in_0_6_l285_285200

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x < 2 then x^3 - x
  else f (x - 2 * real.floor (x / 2))

theorem f_zero_count_in_0_6 :
  ∃ n, n = 7 ∧ ∀ x ∈ set.Icc 0 6, f x = 0 → ∃! y, y ∈ set.Icc 0 6 ∧ f y = 0 ∧ y = x :=
sorry

end f_zero_count_in_0_6_l285_285200


namespace simplify_cot_tan_l285_285732

theorem simplify_cot_tan : Real.cot (Real.pi / 9) + Real.tan (Real.pi / 18) = Real.csc (Real.pi / 9) := 
by
  sorry

end simplify_cot_tan_l285_285732


namespace part1_part2_l285_285213

theorem part1 (f : ℝ → ℝ) (h : ∀ x, f x = (x - 1)^2 + 2) :
  ∃ m : ℝ, ∀ x : ℝ, m + f x > 0 ↔ m > -2 :=
begin
  sorry,
end

theorem part2 (f : ℝ → ℝ) (h : ∀ x, f x = (x - 1)^2 + 2) :
  (∃ x : ℝ, m - f x > 0) → m > 2 :=
begin
  sorry,
end

end part1_part2_l285_285213


namespace total_number_of_fruits_picked_l285_285698

theorem total_number_of_fruits_picked :
  let melanie_plums := 4
  let melanie_apples := 6
  let dan_plums := 9
  let dan_oranges := 2
  let sally_plums := 3
  let sally_cherries := 10
  let thomas_plums := 15
  let thomas_peaches := 5
  let melanie_total := melanie_plums + melanie_apples
  let dan_total := dan_plums + dan_oranges
  let sally_total := sally_plums + sally_cherries
  let thomas_total := thomas_plums + thomas_peaches
  let total_fruits := melanie_total + dan_total + sally_total + thomas_total
  in total_fruits = 54 :=
by
  sorry

end total_number_of_fruits_picked_l285_285698


namespace altered_volume_percentage_l285_285085

noncomputable def percentVolumeAltered (length width height : ℕ) (sideRemovedCube sideAddedCube : ℕ) : ℕ :=
  let originalVolume := length * width * height
  let removedVolume := 4 * sideRemovedCube^3
  let addedVolume := 4 * sideAddedCube^3
  let netVolumeChange := removedVolume - addedVolume
  (netVolumeChange * 100) / originalVolume

-- Given the problem statement conditions
theorem altered_volume_percentage :
  percentVolumeAltered 20 15 12 4 2 = 6.22 := by
  -- To prove this theorem, which verifies that 6.22% of the original volume is altered.
  sorry

end altered_volume_percentage_l285_285085


namespace ac_plus_bd_eq_neg_10_l285_285243

theorem ac_plus_bd_eq_neg_10 (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 3)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 6) :
  a * c + b * d = -10 :=
by
  sorry

end ac_plus_bd_eq_neg_10_l285_285243


namespace expand_binomials_l285_285924

theorem expand_binomials : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by 
  sorry

end expand_binomials_l285_285924


namespace simplify_trig_identity_l285_285752

-- Defining the trigonometric functions involved
def cot (x : Real) : Real := (cos x) / (sin x)
def tan (x : Real) : Real := (sin x) / (cos x)
def csc (x : Real) : Real := 1 / (sin x)

theorem simplify_trig_identity :
  cot 20 + tan 10 = csc 20 :=
by
  sorry

end simplify_trig_identity_l285_285752


namespace cot_tan_simplify_l285_285740

theorem cot_tan_simplify : cot (20 * π / 180) + tan (10 * π / 180) = csc (20 * π / 180) :=
by
  sorry

end cot_tan_simplify_l285_285740


namespace diagonal_of_rectangle_l285_285422

theorem diagonal_of_rectangle (h P : ℝ) (H_h : h = 16) (H_P : P = 12) : 
  let l := P in
  let w := h in
  let d := real.sqrt((l ^ 2) + (w ^ 2)) in
  d = 20 :=
by
  have l_def : l = 12 := by rw [H_P]
  have w_def : w = 16 := by rw [H_h]
  have d_def : d = real.sqrt((12:ℝ)^2 + (16:ℝ)^2) := by simp [l_def, w_def]
  rw [d_def]
  norm_num
  sorry

end diagonal_of_rectangle_l285_285422


namespace path_length_of_point_P_l285_285966

-- Definitions based on the conditions
def cube_edge_length : ℝ := 1
def tetrahedron_volume : ℝ := 1 / 3

-- Define the geometric setup and give the proof goal
theorem path_length_of_point_P : 
  ∃ P : ℝ, (edge_length = 1) ∧ (volume_of_tetrahedron P = 1 / 3) → P = 2 :=
by
  sorry

end path_length_of_point_P_l285_285966


namespace trigonometric_problem_l285_285606

open Real

theorem trigonometric_problem
  (θ : ℝ)
  (h : (cot θ - 1) / (2 * cot θ + 1) = 1) :
  cos (2 * θ) / (1 + sin (2 * θ)) = 3 :=
by
  sorry

end trigonometric_problem_l285_285606


namespace vishal_investment_more_than_trishul_l285_285008

theorem vishal_investment_more_than_trishul:
  ∀ (V T R : ℝ),
  R = 2100 →
  T = 0.90 * R →
  V + T + R = 6069 →
  ((V - T) / T) * 100 = 10 :=
by
  intros V T R hR hT hSum
  sorry

end vishal_investment_more_than_trishul_l285_285008


namespace madeline_min_cost_l285_285693

/-- Madeline's garden problem: given the conditions specified, prove the minimum amount 
she can spend on seeds to plant 20 flowers with equal number of surviving flowers for each type is $16.80. -/
theorem madeline_min_cost :
  ∃ (cost : ℝ), 
    (∀ (roses daisies sunflowers : ℕ),
       (roses + daisies + sunflowers = 20) →
       (roses * 0.4 ≃ 7) ∧ (daisies * 0.6 ≃ 7) ∧ (sunflowers * 0.5 ≃ 6) →
       let 
           rose_pack_1 := 15,
           rose_pack_1_price := 5,
           rose_pack_2 := 40,
           rose_pack_2_price := 10,
           daisy_pack_1 := 20,
           daisy_pack_1_price := 4,
           daisy_pack_2 := 50,
           daisy_pack_2_price := 9,
           sunflower_pack_1 := 10,
           sunflower_pack_1_price := 3,
           sunflower_pack_2 := 30,
           sunflower_pack_2_price := 7,
           discount_rate := 0.2
       in
       if (roses ≥ 18) then
         if (daisies ≥ 12) then
           if (sunflowers ≥ 12) then
             let base_cost := rose_pack_2_price + daisy_pack_1_price + sunflower_pack_2_price in
             cost = base_cost - discount_rate * base_cost) := 16.80 :=
sorry

end madeline_min_cost_l285_285693


namespace earnings_per_weed_is_six_l285_285692

def flower_bed_weeds : ℕ := 11
def vegetable_patch_weeds : ℕ := 14
def grass_weeds : ℕ := 32
def grass_weeds_half : ℕ := grass_weeds / 2
def soda_cost : ℕ := 99
def money_left : ℕ := 147
def total_weeds : ℕ := flower_bed_weeds + vegetable_patch_weeds + grass_weeds_half
def total_money : ℕ := money_left + soda_cost

theorem earnings_per_weed_is_six :
  total_money / total_weeds = 6 :=
by
  sorry

end earnings_per_weed_is_six_l285_285692


namespace area_of_45_45_90_triangle_l285_285779

noncomputable def leg_length (hypotenuse : ℝ) : ℝ :=
  hypotenuse / Real.sqrt 2

theorem area_of_45_45_90_triangle (hypotenuse : ℝ) (h : hypotenuse = 13) : 
  (1 / 2) * (leg_length hypotenuse) * (leg_length hypotenuse) = 84.5 :=
by
  sorry

end area_of_45_45_90_triangle_l285_285779


namespace number_replacement_l285_285817

theorem number_replacement :
  ∃ x : ℝ, ( (x / (1 / 2) * x) / (x * (1 / 2) / x) = 25 ) ↔ x = 2.5 :=
by 
  sorry

end number_replacement_l285_285817


namespace find_constants_l285_285982

theorem find_constants (a b c : ℝ) (h_neq_0_a : a ≠ 0) (h_neq_0_b : b ≠ 0) 
(h_neq_0_c : c ≠ 0) 
(h_eq1 : a * b = 3 * (a + b)) 
(h_eq2 : b * c = 4 * (b + c)) 
(h_eq3 : a * c = 5 * (a + c)) : 
a = 120 / 17 ∧ b = 120 / 23 ∧ c = 120 / 7 := 
  sorry

end find_constants_l285_285982


namespace coordinates_of_P60_l285_285184

-- Conditions and definitions
def is_valid_point (p : ℕ × ℕ) (n : ℕ) : Prop := p.1 + p.2 = n + 1

def point_sequence : List (ℕ × ℕ) :=
  List.concat (List.concatMap (λ n : ℕ, List.map (λ x : ℕ, (x, n + 1 - x)) (List.range (n + 1))) (List.range 12))

-- Statement of the problem
theorem coordinates_of_P60 : point_sequence.nth 59 = some (5, 7) :=
by sorry

end coordinates_of_P60_l285_285184


namespace range_of_set_is_8_l285_285855

theorem range_of_set_is_8 (a b c : ℝ) (h_sum : a + b + c = 18) (h_median : ∃ a b c, list.median [a, b, c] = 6) (h_min : a = 2) : c - a = 8 :=
by
  -- Proof goes here
  sorry

end range_of_set_is_8_l285_285855


namespace sin_bound_implication_l285_285683

theorem sin_bound_implication (n : ℕ) (an : Fin n → ℝ) 
  (h : ∀ x : ℝ, |(Finset.univ.sum (λ k : Fin n, an k * sin ((k + 1 : ℕ) * x)))| ≤ |sin x|) : 
  |(Finset.univ.sum (λ k : Fin n, (k + 1) * an k))| ≤ 1 :=
sorry

end sin_bound_implication_l285_285683


namespace tangent_line_eq_uniq_zero_max_g_l285_285999

def f (x : ℝ) (a : ℝ) : ℝ :=
  (x^2 - 2*x) * Real.log x + a * x^2 + 2

def g (x : ℝ) (a : ℝ) : ℝ :=
  f x a - x - 2

-- Part (I)
theorem tangent_line_eq (a : ℝ) (h : a = -1) :
  let f' (x : ℝ) := (2*x - 2) * Real.log x + (x - 2) - 2*x in
  let f1 := f 1 a in
  f' 1 = -3 ∧ f1 = 1 → 
  3 * 1 + (-3) * 1 + 1 = 0 := 
by
  sorry

-- Part (II)
theorem uniq_zero (a : ℝ) (h : a > 0) (h_zero : ∀ x, g x a = 0 ↔ false) :
  a = 1 := 
by
  sorry

-- Part (III)
theorem max_g (a : ℝ) (h : a = 1) (m : ℝ) (h_m : ∀ x, e^(-2) < x ∧ x < e → g x a ≤ m) : 
  m ≥ 2 * e^2 - 3 * e := 
by
  sorry

end tangent_line_eq_uniq_zero_max_g_l285_285999


namespace formula_for_an_harmonic_sum_lt_two_l285_285668

def a (n : ℕ) : ℕ := n * (n + 1) / 2

theorem formula_for_an (n : ℕ) (h : n > 0) : a n = n * (n + 1) / 2 := sorry

theorem harmonic_sum_lt_two (n : ℕ) (h : n > 0) : 
  (∑ k in finset.range (n + 1), (1 : ℝ) / a (k + 1)) < 2 := sorry

end formula_for_an_harmonic_sum_lt_two_l285_285668


namespace matrix_power_is_correct_l285_285490

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]
def A_cubed : Matrix (Fin 2) (Fin 2) ℤ := !![3, -6; 6, -3]

theorem matrix_power_is_correct : A ^ 3 = A_cubed := by 
  sorry

end matrix_power_is_correct_l285_285490


namespace find_n_in_range_l285_285341

-- Define the conditions
def a : ℤ := 23
def b : ℤ := 58
def n : ℤ := 150

theorem find_n_in_range :
  (a - b) % 37 = n % 37 ∧ (n ≥ 150 ∧ n ≤ 191) :=
by {
  -- These conditions hold based on problem statement
  have ha : a % 37 = 23 % 37 := by refl,
  have hb : b % 37 = 58 % 37 := by refl,

  -- We derive the result 21 for b modulo 37,
  let b_mod := 21,

  -- Compute the term a - b mod 37
  have h_sub : (a - b) % 37 = (23 - 21) % 37 := mod_sub_eq_mod a b 37,
  rw [ha, sub_eq_add_neg, add_comm, ←mod_neg_eq_sub_mod] at h_sub,

  -- The integer n found to satisfy the conditions
  let potential_n := 150,

  -- Directly infer the theorem result
  exact ⟨h_sub, by norm_num⟩,
}

end find_n_in_range_l285_285341


namespace part1_tangent_line_part2_monotonically_increasing_l285_285220

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.log (x + 1)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x

theorem part1_tangent_line (a : ℝ) :
  a = 2 → (∃ y : ℝ → ℝ, ∀ x : ℝ, y x = 2 * x) :=
by
  intro ha
  have h_def : f 2 0 = 0,
    from Eq.trans (congr_arg (fun x => (x + 2) * Real.log (x + 1)) (rfl)) (by simp [Real.log])
  have h_deriv : (Real.log(1) + (2 : ℝ)) = 2,
    by simp
  use (fun x => 2 * x)
  intro x
  exact (by simp)

theorem part2_monotonically_increasing (a : ℝ) :
  (∀ x : ℝ, 0 < x → (Real.log(x + 1) + (a - 1) / (x + 1)) ≥ 0) → a ≥ 1 :=
by
  intro h
  have key : ∀ x : ℝ, 0 < x → (a - 1) ≥ - (x + 1) * Real.log (x + 1),
    from (h 0.1)
  have ha : a - 1 ≥ 0,
    sorry -- further analysis of the inequality

  exact ha

end part1_tangent_line_part2_monotonically_increasing_l285_285220


namespace symmetric_points_sum_l285_285984

theorem symmetric_points_sum (a b : ℝ) (P Q : ℝ × ℝ) 
    (hP : P = (3, a)) (hQ : Q = (b, 2))
    (symm : P = (-Q.1, Q.2)) : a + b = -1 := by
  sorry

end symmetric_points_sum_l285_285984


namespace option_A_option_B_option_C_option_D_l285_285987

-- Define the equation of the curve
def curve (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 1) + y^2 / (5 - k) = 1

-- Prove that when k=2, the curve is a circle
theorem option_A (x y : ℝ) : curve 2 x y ↔ x^2 + y^2 = 3 :=
by
  sorry

-- Prove the necessary and sufficient condition for the curve to be an ellipse
theorem option_B (k : ℝ) : (-1 < k ∧ k < 5) ↔ ∃ x y, curve k x y ∧ (k ≠ 2) :=
by
  sorry

-- Prove the condition for the curve to be a hyperbola with foci on the y-axis
theorem option_C (k : ℝ) : k < -1 ↔ ∃ x y, curve k x y ∧ (k < -1 ∧ k < 5) :=
by
  sorry

-- Prove that there does not exist a real number k such that the curve is a parabola
theorem option_D : ¬ (∃ k x y, curve k x y ∧ ∃ a b, x = a ∧ y = b) :=
by
  sorry

end option_A_option_B_option_C_option_D_l285_285987


namespace fred_balloons_remaining_l285_285540

theorem fred_balloons_remaining 
    (initial_balloons : ℕ)         -- Fred starts with these many balloons
    (given_to_sandy : ℕ)           -- Fred gives these many balloons to Sandy
    (given_to_bob : ℕ)             -- Fred gives these many balloons to Bob
    (h1 : initial_balloons = 709) 
    (h2 : given_to_sandy = 221) 
    (h3 : given_to_bob = 153) : 
    (initial_balloons - given_to_sandy - given_to_bob = 335) :=
by
  sorry

end fred_balloons_remaining_l285_285540


namespace man_speed_same_direction_l285_285061

theorem man_speed_same_direction (L t : ℝ) (speed_train_kmph speed_cross : ℝ)
  (hL : L = 500) (hspeed_train_kmph : speed_train_kmph = 63) (ht : t = 29.997600191984642)
  (hspeed_cross : speed_cross = speed_train_kmph * 1000 / 3600) :
  let speed_man := speed_cross - (L / t)
  in speed_man ≈ 0.833 :=
by
  have h1 : speed_cross = speed_train_kmph * 1000 / 3600 := hspeed_cross,
  have h2 : speed_cross = 63 * 1000 / 3600 := by rw [hspeed_train_kmph, h1],
  have h_speed_m_s : speed_cross = 17.5 := by norm_num,
  have h3 := (L / t), -- compute the relative speed
  have h4: 500 / 29.997600191984642 ≈ 16.667222222222222, by norm_num,
  have speed_man := 17.5 - h3,
  show speed_man ≈ 0.833, 
  from calc speed_man = 17.5 - (500 / 29.997600191984642) : by rw h4
              ≈ 0.833 : by norm_num

-- sorry to skip the actual detailed proof steps
sorry

end man_speed_same_direction_l285_285061


namespace brick_height_calc_l285_285848

theorem brick_height_calc 
  (length_wall : ℝ) (height_wall : ℝ) (width_wall : ℝ) 
  (num_bricks : ℕ) 
  (length_brick : ℝ) (width_brick : ℝ) 
  (H : ℝ) 
  (volume_wall : ℝ) 
  (volume_brick : ℝ)
  (condition1 : length_wall = 800) 
  (condition2 : height_wall = 600) 
  (condition3 : width_wall = 22.5)
  (condition4 : num_bricks = 3200) 
  (condition5 : length_brick = 50) 
  (condition6 : width_brick = 11.25) 
  (condition7 : volume_wall = length_wall * height_wall * width_wall) 
  (condition8 : volume_brick = length_brick * width_brick * H) 
  (condition9 : num_bricks * volume_brick = volume_wall) 
  : H = 6 := 
by
  sorry

end brick_height_calc_l285_285848


namespace range_of_x_coordinate_of_point_l285_285178

theorem range_of_x_coordinate_of_point 
  (P : ℝ × ℝ) 
  (m : ℝ) 
  (C : set (ℝ × ℝ)) 
  (l : set (ℝ × ℝ)) 
  (hC : C = {p | (p.1 + 1)^2 + (p.2 - 1)^2 = 1}) 
  (hl : l = {p | p.2 = 2 * p.1 - 4}) 
  (hP : P ∈ l)
  (hAB : ∀ A B ∈ C, A ≠ B ∧ ∃ P ∈ l, dist P A = 2 * dist A B) :
  9 - 2 * real.sqrt 19 ≤ m ∧ m ≤ 9 + 2 * real.sqrt 19 :=
begin
  sorry
end

end range_of_x_coordinate_of_point_l285_285178


namespace multiply_polynomials_l285_285312

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l285_285312


namespace arrangement_count_l285_285195

theorem arrangement_count : 
  let elements := ['A', 'B', 'C', 'D', 'E', 'F'] in
  ∃ (n : ℕ), 
    -- Condition: A is not at either end
    (∀(arrangement : list Char), arrangement.length = 6 → arrangements elements → ∀i : ℕ, i ≠ 0 ∧ i ≠ 5 → 
      arrangement.nth i = some 'A' →  (
        -- Condition: B and C are adjacent
        (arrangement.nth (i - 1) = some 'B' ∧ arrangement.nth (i - 2) = some 'C') ∨
        (arrangement.nth (i + 1) = some 'B' ∧ arrangement.nth (i + 2) = some 'C')
      )
    )  → n = 144 :=
sorry

end arrangement_count_l285_285195


namespace largest_common_term_l285_285851

theorem largest_common_term (n : ℕ) (h : 1 ≤ n ∧ n ≤ 200) :
  (∃ m : ℕ, n = 3 + 8 * m) ∧ (∃ k : ℕ, n = 5 + 9 * k) ↔ n = 187 :=
by
  skip

end largest_common_term_l285_285851


namespace degree_sum_of_polynomials_l285_285245

-- Define the degree of a polynomial
def degree (p : Polynomial ℤ) : ℕ := p.degree.toNat  -- Assuming p.degree returns the degree in a form that can be converted to ℕ

-- Define f(z) as a polynomial of degree 3
def f : Polynomial ℤ := Polynomial.Coeff (λ n, if n = 3 then 1 else 0)

-- Define g(z) as a polynomial of degree 1
def g : Polynomial ℤ := Polynomial.Coeff (λ n, if n = 1 then 1 else 0)

-- The statement to prove
theorem degree_sum_of_polynomials :
  degree (f + g) = 3 :=
sorry

end degree_sum_of_polynomials_l285_285245


namespace find_fine_for_inappropriate_items_in_recycling_bin_l285_285696

noncomputable def trash_bin_cost_per_week := 10
noncomputable def recycling_bin_cost_per_week := 5
noncomputable def num_trash_bins := 2
noncomputable def num_recycling_bins := 1
noncomputable def weeks_per_month := 4
noncomputable def elderly_discount_rate := 0.18
noncomputable def final_bill := 102

theorem find_fine_for_inappropriate_items_in_recycling_bin :
  let weekly_cost := (num_trash_bins * trash_bin_cost_per_week) + (num_recycling_bins * recycling_bin_cost_per_week)
  let monthly_cost := weekly_cost * weeks_per_month
  let discount := monthly_cost * elderly_discount_rate
  let discounted_cost := monthly_cost - discount
  let fine := final_bill - discounted_cost
  fine = 20 :=
by
  sorry

end find_fine_for_inappropriate_items_in_recycling_bin_l285_285696


namespace polynomial_evaluation_l285_285577

theorem polynomial_evaluation (x y : ℝ) (h : 2 * x^2 + 3 * y + 7 = 8) : -2 * x^2 - 3 * y + 10 = 9 :=
by
  have h1 : 2 * x^2 + 3 * y = 1 :=
    calc 2 * x^2 + 3 * y
        = 1 : by linarith [h]
  calc -2 * x^2 - 3 * y + 10
        = -1 + 10 : by rw [h1]
        = 9 : by norm_num

end polynomial_evaluation_l285_285577


namespace eql_sol_ineq_sol_l285_285581

def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 4

theorem eql_sol (
  x : ℝ
) : f x = 1 / 4 ↔ x = Real.sqrt 2 ∧ 1 ≤ x :=
by sorry -- proof placeholder

theorem ineq_sol (
  x : ℝ
) : f x ≤ 2 ↔ -1 ≤ x ∧ x ≤ 16 :=
by sorry -- proof placeholder

end eql_sol_ineq_sol_l285_285581


namespace cloth_sold_l285_285458

theorem cloth_sold (total_sell_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ)
  (h1 : total_sell_price = 15000)
  (h2 : loss_per_metre = 10)
  (h3 : cost_price_per_metre = 40) :
  let sell_price_per_metre := cost_price_per_metre - loss_per_metre in
  let metres_sold := total_sell_price / sell_price_per_metre in
  metres_sold = 500 := by
  sorry

end cloth_sold_l285_285458


namespace max_value_l285_285976

-- Definitions for conditions
variables {a b : ℝ}
variables (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (1 / b) = 2)

-- Statement of the theorem
theorem max_value : (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (1 / a) + (1 / b) = 2 ∧ ∀ y : ℝ,
  (1 / y) * ((2 / (y * (3 * y - 1)⁻¹)) + 1) ≤ 25 / 8) :=
sorry

end max_value_l285_285976


namespace vector_subtraction_l285_285487

def vector1 : ℝ × ℝ := (3, -5)
def vector2 : ℝ × ℝ := (2, -6)
def scalar1 : ℝ := 4
def scalar2 : ℝ := 3

theorem vector_subtraction :
  (scalar1 • vector1 - scalar2 • vector2) = (6, -2) := by
  sorry

end vector_subtraction_l285_285487


namespace regular_hexagon_area_l285_285394

theorem regular_hexagon_area (r : ℝ) (h₀ : π * r ^ 2 = 400 * π) : 
  (6 * 100 * real.sqrt 3 = 600 * real.sqrt 3) :=
by {
  sorry
}

end regular_hexagon_area_l285_285394


namespace factorize_polynomial_l285_285524

theorem factorize_polynomial (x y : ℝ) : 
  (2 * x^2 * y - 4 * x * y^2 + 2 * y^3) = 2 * y * (x - y)^2 :=
sorry

end factorize_polynomial_l285_285524


namespace percentage_not_drop_l285_285651

def P_trip : ℝ := 0.40
def P_drop_given_trip : ℝ := 0.25
def P_not_drop : ℝ := 0.90

theorem percentage_not_drop :
  (1 - P_trip * P_drop_given_trip) = P_not_drop :=
sorry

end percentage_not_drop_l285_285651


namespace not_perfect_square_l285_285056

theorem not_perfect_square (n : ℕ) (h : 0 < n) : ¬ ∃ k : ℕ, k * k = 2551 * 543^n - 2008 * 7^n :=
by
  sorry

end not_perfect_square_l285_285056


namespace cot20_tan10_eq_csc20_l285_285749

theorem cot20_tan10_eq_csc20 : 
  Real.cot 20 * (Real.pi / 180) + Real.tan 10 * (Real.pi / 180) = Real.csc 20 * (Real.pi / 180) :=
by sorry

end cot20_tan10_eq_csc20_l285_285749


namespace smallest_positive_period_of_f_max_and_min_values_of_f_on_interval_l285_285235

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin (π / 2 * x), Real.cos (π / 2 * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin (π / 2 * x), Real.sqrt 3 * Real.sin (π / 2 * x))

noncomputable def f (x : ℝ) : ℝ :=
  let a1 := a x
  let b1 := b x
  a1.1 * (a1.1 + 2 * b1.1) + a1.2 * (a1.2 + 2 * b1.2)

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = 2 := sorry

theorem max_and_min_values_of_f_on_interval :
  ∃ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ 1 ∧ f x₁ = 4) ∧ (0 ≤ x₂ ∧ x₂ ≤ 1 ∧ f x₂ = 1) := sorry

end smallest_positive_period_of_f_max_and_min_values_of_f_on_interval_l285_285235


namespace hyperbola_asymptote_focus_directrix_l285_285203

def hyperbola_equation (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : Prop :=
  (∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)

theorem hyperbola_asymptote_focus_directrix : 
  ∃ a b : ℝ, 
  a > 0 ∧ b > 0 ∧ 
  (∃ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) ∧ 
  (b / a = sqrt 3) ∧ 
  (a^2 + b^2 = 36) ∧
  (∃ F : ℝ × ℝ, F = (-6, 0)) ∧ 
  hyperbola_equation a b :=
begin
  use [3, 3*sqrt 3],
  split, { exact zero_lt_three },
  split, { exact mul_pos zero_lt_three (sqrt_pos zero_lt_three) },
  split,
  { use [x, y],
    exact sorry
  },
  split,
  { exact sorry },
  split,
  { exact sorry },
  split,
  { use (-6, 0),
    exact sorry
  },
  exact sorry
end

end hyperbola_asymptote_focus_directrix_l285_285203


namespace simplify_cotAndTan_l285_285719

theorem simplify_cotAndTan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := 
by
  sorry

end simplify_cotAndTan_l285_285719


namespace sum_of_binary_digits_435_l285_285041

theorem sum_of_binary_digits_435 : 
  let n := 435
  let bin_n := nat.binary n
  bin_n.foldr (fun digit sum => digit + sum) 0 = 6 :=
by
  let n := 435
  let bin_n := nat.binary n
  have h_binary : bin_n = [1, 1, 0, 1, 1, 0, 0, 1, 1] := sorry
  have h_sum : [1, 1, 0, 1, 1, 0, 0, 1, 1].foldr (fun digit sum => digit + sum) 0 = 6 := by
    -- calculate the sum here
    sorry
  exact h_sum


end sum_of_binary_digits_435_l285_285041


namespace leak_empties_tank_in_8_hours_l285_285072

theorem leak_empties_tank_in_8_hours (capacity : ℕ) (inlet_rate_per_minute : ℕ) (time_with_inlet_open : ℕ) (time_without_inlet_open : ℕ) : 
  capacity = 8640 ∧ inlet_rate_per_minute = 6 ∧ time_with_inlet_open = 12 ∧ time_without_inlet_open = 8 := 
by 
  sorry

end leak_empties_tank_in_8_hours_l285_285072


namespace inequality_proof_l285_285686

variables {x y z : ℝ}

theorem inequality_proof (h1: 0 < x) (h2: 0 < y) (h3: 0 < z) 
  (h4 : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x^2 + yz) / Real.sqrt(2 * x^2 * (y + z)) + 
  (y^2 + zx) / Real.sqrt(2 * y^2 * (z + x)) + 
  (z^2 + xy) / Real.sqrt(2 * z^2 * (x + y)) ≥ 1 := 
sorry

end inequality_proof_l285_285686


namespace product_of_roots_is_4_l285_285155

def fourth_root_of_16 := 16^(1/4 : ℝ)
def fifth_root_of_32 := 32^(1/5 : ℝ)

theorem product_of_roots_is_4 : fourth_root_of_16 * fifth_root_of_32 = 4 := by
  sorry

end product_of_roots_is_4_l285_285155


namespace bill_earnings_per_ounce_l285_285874

-- Given conditions
def ounces_sold : Nat := 8
def fine : Nat := 50
def money_left : Nat := 22
def total_money_earned : Nat := money_left + fine -- $72

-- The amount earned for every ounce of fool's gold
def price_per_ounce : Nat := total_money_earned / ounces_sold -- 72 / 8

-- The proof statement
theorem bill_earnings_per_ounce (h: price_per_ounce = 9) : True :=
by
  trivial

end bill_earnings_per_ounce_l285_285874


namespace ratio_of_surface_areas_l285_285786

theorem ratio_of_surface_areas (a : ℝ) 
  (h1 : a = 2 * inscribed_radius) 
  (h2 : sqrt 3 * a = 2 * circumscribed_radius) :
  inscribed_surface_area / circumscribed_surface_area = 1 / 3 :=
by
  let inscribed_radius := a / 2
  let circumscribed_radius := sqrt 3 * a / 2
  let inscribed_surface_area := 4 * π * inscribed_radius ^ 2
  let circumscribed_surface_area := 4 * π * circumscribed_radius ^ 2
  sorry

end ratio_of_surface_areas_l285_285786


namespace no_suitable_c_l285_285131

def has_rational_roots (a b c : ℤ) : Prop :=
  ∃ k : ℤ, k^2 = b^2 - 4 * a * c

def sum_of_roots_exceeds (a b c : ℤ) (x : ℚ) : Prop :=
  - (b : ℚ) / (a : ℚ) > x

theorem no_suitable_c :
  ∑' (c : ℕ) in {c : ℕ | has_rational_roots 3 7 c ∧ sum_of_roots_exceeds 3 7 c 4}, c = 0 :=
by sorry

end no_suitable_c_l285_285131


namespace ratio_of_perimeters_l285_285533

theorem ratio_of_perimeters (s : ℝ) (h1: s ≠ 0) (a : ℝ) (h2 : a = (Real.sqrt 5 + 1)/2) :
  let s1 := s * a in
  let P1 := 4 * s1 in
  let P2 := 4 * s in
  let d := Real.sqrt 2 * s1 in
  d = s → (P1 / P2 = a) :=
by
  intro h
  unfold s1 P1 P2 d
  sorry

end ratio_of_perimeters_l285_285533


namespace length_of_second_train_is_229_95_l285_285844

noncomputable def length_of_second_train
  (l1 : ℕ) -- length of the first train in meters
  (v1 : ℕ) -- speed of the first train in km/h
  (v2 : ℕ) -- speed of the second train in km/h
  (t : ℕ)  -- time in seconds for the trains to cross
  (conversion_ratio : ℝ := 1000 / 3600) -- conversion factor from km/h to m/s
  : ℝ :=
  let v1_m_s : ℝ := v1 * conversion_ratio in
  let v2_m_s : ℝ := v2 * conversion_ratio in
  let relative_speed : ℝ := v1_m_s + v2_m_s in
  let total_distance : ℝ := relative_speed * t in
  total_distance - l1

theorem length_of_second_train_is_229_95
  (h_l1 : l1 = 270)
  (h_v1 : v1 = 120)
  (h_v2 : v2 = 80)
  (h_t : t = 9) :
  length_of_second_train l1 v1 v2 t = 229.95 :=
by
  rw [h_l1, h_v1, h_v2, h_t]
  simp [length_of_second_train]
  norm_num
  sorry

end length_of_second_train_is_229_95_l285_285844


namespace part_I_part_II_l285_285266

-- Define the coordinate systems and the equations given in the problem.
def circle_polar_equation (ρ θ : ℝ) : Prop :=
  ρ = 2 * real.sqrt 2 * real.cos (θ + π / 4)

def line_parametric_equation (t : ℝ) (x y : ℝ) : Prop :=
  x = t ∧ y = -1 + 2 * real.sqrt 2 * t

-- Define the intersection points and the arbitrary point on the circle
variables (A B P : ℝ × ℝ)
variables (ρ θ : ℝ)

-- Conditions
def is_point_on_circle (P : ℝ × ℝ) : Prop :=
  ∃ ρ θ, P = (ρ * real.cos θ, ρ * real.sin θ) ∧ circle_polar_equation ρ θ

def line_intersects_circle (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂, line_parametric_equation t₁ A.1 A.2 ∧ line_parametric_equation t₂ B.1 B.2

-- Questions translated to Lean proofs
noncomputable def polar_coordinates_of_circle_center : Prop :=
  let center := (1, -1) in
  let polar_center := (real.sqrt 2, 7 * π / 4) in
  is_point_on_circle center ∧ polar_center = (real.sqrt 2, 7 * π / 4)

noncomputable def max_area_of_triangle_PAB : Prop :=
  is_point_on_circle P ∧ line_intersects_circle A B ∧
  let max_area := (10 * real.sqrt 5) / 9 in
  area_of_triangle P A B = max_area

-- Final theorem statements
theorem part_I : polar_coordinates_of_circle_center := 
  sorry

theorem part_II : max_area_of_triangle_PAB := 
  sorry

end part_I_part_II_l285_285266


namespace m_times_t_l285_285296

noncomputable def problem_g (g : ℝ → ℝ) :=
  ∀ x y : ℝ, g (x * g y - x) = x * y - g x

theorem m_times_t (g : ℝ → ℝ)
  (h : problem_g g) :
  let m : ℕ := 2,
  let t : ℝ := 0 in
  m * t = 0 := 
by
  sorry

end m_times_t_l285_285296


namespace defect_rate_of_line_A_l285_285068

theorem defect_rate_of_line_A (total_chips : ℕ) (chips_A : ℕ) (chips_B : ℕ) (defect_rate_B : ℝ) (overall_defect_rate : ℝ) :
  total_chips = 20 →
  chips_A = 12 →
  chips_B = 8 →
  defect_rate_B = 1 / 20 →
  overall_defect_rate = 0.08 →
  let defect_rate_A := 12 * (1 / (20 * overall_defect_rate - 8 * defect_rate_B)) in
  defect_rate_A = 1 / 10 :=
by
  intros h1 h2 h3 h4 h5
  let defect_rate_A := 12 * (1 / (20 * overall_defect_rate - 8 * defect_rate_B))
  sorry

end defect_rate_of_line_A_l285_285068


namespace number_of_subsets_P_l285_285974

-- Define the sets M and N
def M := {2, 4}
def N := {1, 2}

-- Define the set P
def P := {x : ℚ | ∃ a ∈ M, ∃ b ∈ N, x = a / b}

-- State the theorem to be proved
theorem number_of_subsets_P : ∃ (n : ℕ), n = 2 ^ 3 ∧ (∃ s : finset ℚ, s = P ∧ s.powerset.card = n) :=
by
  -- Definitions of M and N
  let M := {2, 4}
  let N := {1, 2}
  -- Definition of P derived from M and N
  let P_set := {x : ℚ | ∃ a ∈ M, ∃ b ∈ N, x = a / b}
  -- Prove the number of subsets of P is 8
  use 8
  sorry

end number_of_subsets_P_l285_285974


namespace number_of_sets_X_when_n_5_average_a_X_l285_285228

def A (n : ℕ) : Set ℕ := {x | 1 ≤ x ∧ x ≤ n}

def is_valid_X (n : ℕ) (X : Set ℕ) : Prop :=
  X ⊆ A n ∧ 2 ≤ X.card ∧ X.card ≤ n - 2

def a_X (X : Finset ℕ) : ℕ :=
  if X.card < 2 then 0 else X.min' (by sorry) + X.max' (by sorry)

theorem number_of_sets_X_when_n_5 : (Finset.powersetLen 2 (Finset.range 5)) ∪
                                    (Finset.powersetLen 3 (Finset.range 5)).card = 20 :=
by sorry

theorem average_a_X (n : ℕ) (h : n ≥ 4) :
  let X_sets := {X : Set ℕ | is_valid_X n X} in
  (∑ X in X_sets.to_finset, a_X X.to_finset) / X_sets.to_finset.card = n + 1 :=
by sorry

end number_of_sets_X_when_n_5_average_a_X_l285_285228


namespace overall_loss_l285_285437

theorem overall_loss
  (house_selling_price : ℝ) (house_loss_percent : ℝ)
  (store_selling_price : ℝ) (store_gain_percent : ℝ)
  (car_selling_price : ℝ) (car_loss_percent : ℝ) :
  house_selling_price = 10000 → house_loss_percent = 25 →
  store_selling_price = 15000 → store_gain_percent = 25 →
  car_selling_price = 8000 → car_loss_percent = 20 →
  let house_cost_price := house_selling_price / ((100 - house_loss_percent) / 100),
      store_cost_price := store_selling_price / ((100 + store_gain_percent) / 100),
      car_cost_price := car_selling_price / ((100 - car_loss_percent) / 100),
      total_cost := house_cost_price + store_cost_price + car_cost_price,
      total_selling_price := house_selling_price + store_selling_price + car_selling_price in
  total_cost - total_selling_price = 2333.33 :=
by
  intros hsp hlp ssp sgp csp clp Hhsp Hhlp Hssp Hsgp Hcsp Hclp
  let house_cost_price := hsp / ((100 - hlp) / 100)
  let store_cost_price := ssp / ((100 + sgp) / 100)
  let car_cost_price := csp / ((100 - clp) / 100)
  let total_cost := house_cost_price + store_cost_price + car_cost_price
  let total_selling_price := hsp + ssp + csp
  have Hh : house_cost_price = 13333.33 := sorry
  have Hs : store_cost_price = 12000 := sorry
  have Hc : car_cost_price = 10000 := sorry
  rw [Hhsp, Hhlp, Hssp, Hsgp, Hcsp, Hclp] at *
  rw [Hh, Hs, Hc]
  suffices : 13333.33 + 12000 + 10000 - (10000 + 15000 + 8000) = 2333.33, by assumption
  sorry

end overall_loss_l285_285437


namespace proof_correct_option_c_l285_285564

theorem proof_correct_option_c (a b : ℝ) (θ : ℝ) (ha : 1 < b) (hb : b < a) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) :
  a * Real.log base b (Real.sin θ) < b * Real.log base a (Real.sin θ) :=
sorry

end proof_correct_option_c_l285_285564


namespace minimum_value_xy_minimum_value_x_plus_2y_l285_285295

-- (1) Prove that the minimum value of \(xy\) given \(x, y > 0\) and \(\frac{1}{x} + \frac{9}{y} = 1\) is \(36\).
theorem minimum_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 9 / y = 1) : x * y ≥ 36 := 
sorry

-- (2) Prove that the minimum value of \(x + 2y\) given \(x, y > 0\) and \(\frac{1}{x} + \frac{9}{y} = 1\) is \(19 + 6\sqrt{2}\).
theorem minimum_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 9 / y = 1) : x + 2 * y ≥ 19 + 6 * Real.sqrt 2 := 
sorry

end minimum_value_xy_minimum_value_x_plus_2y_l285_285295


namespace cheese_cost_l285_285695

theorem cheese_cost (bread_cost cheese_cost total_paid total_change coin_change nickels_value : ℝ) 
                    (quarter dime nickels_count : ℕ)
                    (h1 : bread_cost = 4.20)
                    (h2 : total_paid = 7.00)
                    (h3 : quarter = 1)
                    (h4 : dime = 1)
                    (h5 : nickels_count = 8)
                    (h6 : coin_change = (quarter * 0.25) + (dime * 0.10) + (nickels_count * 0.05))
                    (h7 : total_change = total_paid - bread_cost)
                    (h8 : cheese_cost = total_change - coin_change) :
                    cheese_cost = 2.05 :=
by {
    sorry
}

end cheese_cost_l285_285695


namespace find_a_2023_l285_285622

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, (1 / a n - 1 / a (n + 1) - 1 / (a n * a (n + 1)) = 1)

theorem find_a_2023 (a : ℕ → ℚ) (h : sequence a) : a 2023 = -1 / 2 :=
  sorry

end find_a_2023_l285_285622


namespace sufficient_but_not_necessary_condition_l285_285300

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (2 * x^2 + x - 1 ≥ 0) → (x ≥ 1/2) ∨ (x ≤ -1) :=
by
  -- The given inequality and condition imply this result.
  sorry

end sufficient_but_not_necessary_condition_l285_285300


namespace johns_honey_production_l285_285282

theorem johns_honey_production :
  let S := 0.5 in
  let hive1_honey := 1000 * S in
  let hive2_honey := 800 * 1.4 * S in
  let hive3_honey := 1200 * 0.7 * S in
  hive1_honey + hive2_honey + hive3_honey = 1480 :=
by
  let S := 0.5
  let hive1_honey := 1000 * S
  let hive2_honey := 800 * 1.4 * S
  let hive3_honey := 1200 * 0.7 * S
  calc
    hive1_honey + hive2_honey + hive3_honey = 1000 * 0.5 + 800 * 1.4 * 0.5 + 1200 * 0.7 * 0.5 : by sorry
    ... = 500 + 560 + 420 : by sorry
    ... = 1480 : by sorry

end johns_honey_production_l285_285282


namespace prob_even_product_first_second_spinner_l285_285005

def first_spinner_values := [2, 3, 5, 7, 11]
def second_spinner_values := [4, 6, 9, 10, 13, 15]

def probability_even_product (spinner1 spinner2 : List ℕ) : ℚ :=
  let total_outcomes := spinner1.length * spinner2.length
  let is_even (n : ℕ) : Bool := (n % 2 = 0)
  let is_odd (n : ℕ) : Bool := ¬ is_even n
  let odd_values1 := spinner1.filter is_odd
  let odd_values2 := spinner2.filter is_odd
  let odd_outcomes := odd_values1.length * odd_values2.length
  1 - (odd_outcomes : ℚ) / (total_outcomes : ℚ)

theorem prob_even_product_first_second_spinner :
  probability_even_product first_spinner_values second_spinner_values = 7 / 10 :=
by
  sorry

end prob_even_product_first_second_spinner_l285_285005


namespace lcm_18_30_is_90_l285_285032

noncomputable def LCM_of_18_and_30 : ℕ := 90

theorem lcm_18_30_is_90 : nat.lcm 18 30 = LCM_of_18_and_30 := 
by 
  -- definition of lcm should be used here eventually
  sorry

end lcm_18_30_is_90_l285_285032


namespace pencils_ratio_l285_285390

theorem pencils_ratio
  (Sarah_pencils : ℕ)
  (Tyrah_pencils : ℕ)
  (Tim_pencils : ℕ)
  (h1 : Tyrah_pencils = 12)
  (h2 : Tim_pencils = 16)
  (h3 : Tim_pencils = 8 * Sarah_pencils) :
  Tyrah_pencils / Sarah_pencils = 6 :=
by
  sorry

end pencils_ratio_l285_285390


namespace percentage_other_schools_l285_285642

variable (totalApplicants : ℕ) (acceptanceRate : ℚ) (attendanceRate : ℚ) (studentsAttending : ℕ)
variable (h1 : totalApplicants = 20000)
variable (h2 : acceptanceRate = 0.05)
variable (h3 : attendanceRate = 0.9)
variable (h4 : studentsAttending = 900)

theorem percentage_other_schools
  (h1 : totalApplicants = 20000)
  (h2 : acceptanceRate = 0.05)
  (h3 : attendanceRate = 0.9)
  (h4 : studentsAttending = 900)
  (acceptedStudents : ℕ) (h5 : acceptedStudents = totalApplicants * (acceptanceRate.toReal))
  (Hattending_correct : studentsAttending = acceptedStudents * (attendanceRate.toReal)) :
  (100 - (attendanceRate * 100 : ℚ)) = 10 := 
sorry

end percentage_other_schools_l285_285642


namespace determine_n_l285_285121

theorem determine_n (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∃ k d, k = (nat.factors n).head ∧ n = k^2 + d^2 ∧ d ∣ n) :
  n = 8 ∨ n = 20 :=
by
  sorry

end determine_n_l285_285121


namespace find_a_l285_285986

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * a * x)

def tangent_slope_at_zero (a : ℝ) : ℝ := 
  let f_prime := deriv (f a)
  f_prime 0

def line_slope (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + b

theorem find_a (a : ℝ) : 
  (tangent_slope_at_zero a = -1/2) ↔ (a = -1/4) :=
by
  sorry

end find_a_l285_285986


namespace log_base_order_l285_285240

theorem log_base_order (a b : ℝ) (h_cond : log a 2 < log b 2 ∧ log b 2 < 0) : 0 < b ∧ b < a ∧ a < 1 :=
begin
  sorry
end

end log_base_order_l285_285240


namespace composite_A_l285_285323

def A : ℕ := 10^1962 + 1

theorem composite_A : ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ A = p * q :=
  sorry

end composite_A_l285_285323


namespace third_place_amount_l285_285283

noncomputable def total_people : ℕ := 13
noncomputable def money_per_person : ℝ := 5
noncomputable def total_money : ℝ := total_people * money_per_person

noncomputable def first_place_percentage : ℝ := 0.65
noncomputable def second_third_place_percentage : ℝ := 0.35
noncomputable def split_factor : ℝ := 0.5

noncomputable def first_place_money : ℝ := first_place_percentage * total_money
noncomputable def second_third_place_money : ℝ := second_third_place_percentage * total_money
noncomputable def third_place_money : ℝ := split_factor * second_third_place_money

theorem third_place_amount : third_place_money = 11.38 := by
  sorry

end third_place_amount_l285_285283


namespace proposition_induction_l285_285553

variable (P : ℕ → Prop)
variable (k : ℕ)

theorem proposition_induction (h : ∀ k : ℕ, P k → P (k + 1))
    (h9 : ¬ P 9) : ¬ P 8 :=
by
  sorry

end proposition_induction_l285_285553


namespace formula_for_an_harmonic_sum_lt_two_l285_285666

def a (n : ℕ) : ℕ := n * (n + 1) / 2

theorem formula_for_an (n : ℕ) (h : n > 0) : a n = n * (n + 1) / 2 := sorry

theorem harmonic_sum_lt_two (n : ℕ) (h : n > 0) : 
  (∑ k in finset.range (n + 1), (1 : ℝ) / a (k + 1)) < 2 := sorry

end formula_for_an_harmonic_sum_lt_two_l285_285666


namespace sum_of_odd_terms_equals_2151_l285_285055

theorem sum_of_odd_terms_equals_2151 (x : ℕ) (terms : Fin 2010 → ℕ) 
  (h_sequence : ∀ n : Fin 2009, terms n.succ = terms n + 1) 
  (h_sum : ∑ n : Fin 2010, terms n = 5307) :
  ∑ i in Finset.filter (λ n : Fin 2010, n.val % 2 = 0) Finset.univ, terms i = 2151 := 
sorry

end sum_of_odd_terms_equals_2151_l285_285055


namespace daniel_thrice_jane_l285_285114

-- Step 1: Define the variables for the given conditions
def daniel_age_current : ℕ := 40
def jane_age_current : ℕ := 26

-- Step 2: State the resulting equation and correct answer in Lean
theorem daniel_thrice_jane :
    ∃ x : ℕ, daniel_age_current - x = 3 * (jane_age_current - x) → x = 19 :=
begin
  sorry
end

end daniel_thrice_jane_l285_285114


namespace quadratic_function_properties_l285_285833

noncomputable def f (x : ℝ) : ℝ := (x + 1) * (x + 1)

theorem quadratic_function_properties :
  (f(-1) = 0) ∧ (∀ x : ℝ, x ≤ f(x) ∧ f(x) ≤ x^2 + 1) ∧ (f(1) = 1) ∧
  (∀ x : ℝ, f(x) = (x + 1)^2) ∧ 
  (1 + 1 + 1 > 2) :=
by
  split
  {
    rw f,
    unfold f,
    norm_num,
  }
  {
    split
    { 
      intro x,
      split,
      {
          unfold f,
          apply le_add_of_nonneg_left,
          norm_num,
      },
      {
          unfold f,
          calc
            (x + 1) * (x + 1) ≤ x^2 + 2 * x + 1 : by norm_num
            ... ≤ x^2 + 1 : by sorry -- Needs more steps to complete
      }
    },
    split
    {
      rw f,
      norm_num,
    },
    split
    {
      intro x,
      unfold f,
      norm_num,
    },
    exact sorry -- Skip proof
  }  

end quadratic_function_properties_l285_285833


namespace domain_of_fractional_sqrt_function_l285_285354

theorem domain_of_fractional_sqrt_function :
  ∀ x : ℝ, (x + 4 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ∈ (Set.Ici (-4) \ {1})) :=
by
  sorry

end domain_of_fractional_sqrt_function_l285_285354


namespace symmetric_range_l285_285996

noncomputable def symmetric_points_on_ellipse (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    (3 * x1^2 + 4 * y1^2 - 12 = 0) ∧ 
    (3 * x2^2 + 4 * y2^2 - 12 = 0) ∧ 
    ((y1 - y2) / (x1 - x2) = -1 / 4) ∧ 
    (4 * (-m) + m - y1) / (4 * (-m) + m - y2) = 1

theorem symmetric_range {m : ℝ} :
  symmetric_points_on_ellipse m ↔ -2 * real.sqrt 13 / 13 < m ∧ m < 2 * real.sqrt 13 / 13 :=
  sorry

end symmetric_range_l285_285996


namespace quadrilateral_EFGH_l285_285636

variable {EF FG GH HE EH : ℤ}

theorem quadrilateral_EFGH (h1 : EF = 6) (h2 : FG = 18) (h3 : GH = 6) (h4 : HE = 10) (h5 : 12 < EH) (h6 : EH < 24) : EH = 12 := 
sorry

end quadrilateral_EFGH_l285_285636


namespace max_value_quotient_of_four_digit_sum_l285_285937

theorem max_value_quotient_of_four_digit_sum :
  ∃ N : ℕ, (∃ a b c d : ℕ, a ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ b < 10 ∧
    N = 1000 * a + 100 * b + 10 * c + d) ∧
  (∃ S : ℕ, S = (
    let (a, b, c, d) := (N / 1000, (N % 1000) / 100, (N % 100) / 10, N % 10)
    in a + b + c + d)) ∧
  ∀ n : ℕ, (∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧ a ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ b < 10) → 
         ∃ s : ℕ, s = (
           let (a, b, c, d) := (n / 1000, (n % 1000) / 100, (n % 100) / 10, n % 10)
           in a + b + c + d) → n / s ≤ 337 := sorry

end max_value_quotient_of_four_digit_sum_l285_285937


namespace pentagon_area_relation_l285_285628

theorem pentagon_area_relation 
  (A B C D E : Type) [convex_pentagon A B C D E] 
  (S a b c d e : ℝ)
  (h_abcd : area_of_triangle A B C = a)
  (h_bcde : area_of_triangle B C D = b)
  (h_cdea : area_of_triangle C D E = c)
  (h_deab : area_of_triangle D E A = d)
  (h_eabc : area_of_triangle E A B = e)
  (h_pentagon : area_of_pentagon A B C D E = S) :
  S^2 - S*(a + b + c + d + e) + a*b + b*c + c*d + d*e + e*a = 0 := 
sorry

end pentagon_area_relation_l285_285628


namespace cot20_plus_tan10_eq_csc20_l285_285724

theorem cot20_plus_tan10_eq_csc20 : Real.cot 20 * Real.pi / 180 + Real.tan 10 * Real.pi / 180 = Real.csc 20 * Real.pi / 180 := 
by
  sorry

end cot20_plus_tan10_eq_csc20_l285_285724


namespace domain_of_sqrt_function_l285_285934

theorem domain_of_sqrt_function :
  ∀ x : ℝ, (-12 * x^2 + 7 * x + 13 ≥ 0) ↔ (x ∈ set.Icc ((7 - real.sqrt 673) / 24) ((7 + real.sqrt 673) / 24)) :=
by
  sorry

end domain_of_sqrt_function_l285_285934


namespace positive_difference_of_diagonal_sums_l285_285843

def originalCalendar : List (List ℕ) :=
  [[1, 2, 3, 4, 5],
   [6, 7, 8, 9, 10],
   [11, 12, 13, 14, 15],
   [16, 17, 18, 19, 20],
   [21, 22, 23, 24, 25]]

def newCalendar : List (List ℕ) :=
  [[1, 2, 3, 4, 5],
   [6, 7, 8, 9, 10],
   [15, 14, 13, 12, 11],
   [16, 17, 18, 19, 20],
   [25, 24, 23, 22, 21]]

def mainDiagonal (matrix : List (List ℕ)) : List ℕ :=
  [matrix[0]![0], matrix[1]![1], matrix[2]![2], matrix[3]![3], matrix[4]![4]]

def secondaryDiagonal (matrix : List (List ℕ)) : List ℕ :=
  [matrix[0]![4], matrix[1]![3], matrix[2]![2], matrix[3]![1], matrix[4]![0]]

def sumList (l : List ℕ) : ℕ := l.sum

theorem positive_difference_of_diagonal_sums :
  | (sumList (mainDiagonal newCalendar) - sumList (secondaryDiagonal newCalendar)) | = 17 := by
  sorry

end positive_difference_of_diagonal_sums_l285_285843


namespace fifteenth_term_value_l285_285013

noncomputable def geometric_fifteenth_term (a r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ (n - 1)

-- Definitions from the conditions
def first_term : ℝ := 5
def common_ratio : ℝ := 1 / 2
def fifteenth_term : ℕ := 15

-- The proof statement:
theorem fifteenth_term_value :
  geometric_fifteenth_term first_term common_ratio fifteenth_term = 5 / 16384 := by
  sorry

end fifteenth_term_value_l285_285013


namespace number_of_valid_integers_up_to_2000_l285_285938

-- Definitions for the problem
def valid_integer_forms (x : ℝ) (n : ℕ) : Prop :=
  ∃ (m : ℕ), (m = ⌊x⌋) ∧ (n = 7 * m ∨ n = 7 * m + 1 ∨ n = 7 * m + 3 ∨ n = 7 * m + 4)

def count_valid_integers (limit : ℕ) : ℕ :=
  (Finset.range limit).count (λ n, ∃ x : ℝ, valid_integer_forms x n)

-- Statement of the problem
theorem number_of_valid_integers_up_to_2000 : count_valid_integers 2001 = 1140 := 
by
  sorry

end number_of_valid_integers_up_to_2000_l285_285938


namespace value_of_m_l285_285909

theorem value_of_m : 5^2 + 7 = 4^3 + m → m = -32 :=
by
  intro h
  sorry

end value_of_m_l285_285909


namespace relationship_between_a_b_c_l285_285545

noncomputable def a : ℝ := 0.4 ^ 0.3
noncomputable def b : ℝ := 0.3 ^ 0.4
noncomputable def c : ℝ := Real.logBase 0.4 0.3

theorem relationship_between_a_b_c : c > a ∧ a > b :=
by
  sorry

end relationship_between_a_b_c_l285_285545


namespace find_m_l285_285143

theorem find_m (a b c m : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_m : 0 < m) (h : a * b * c * m = 1 + a^2 + b^2 + c^2) : 
  m = 4 :=
sorry

end find_m_l285_285143


namespace novel_pages_l285_285313

theorem novel_pages (x : ℕ)
  (h1 : x - ((1 / 6 : ℝ) * x + 10) = (5 / 6 : ℝ) * x - 10)
  (h2 : (5 / 6 : ℝ) * x - 10 - ((1 / 5 : ℝ) * ((5 / 6 : ℝ) * x - 10) + 20) = (2 / 3 : ℝ) * x - 28)
  (h3 : (2 / 3 : ℝ) * x - 28 - ((1 / 4 : ℝ) * ((2 / 3 : ℝ) * x - 28) + 25) = (1 / 2 : ℝ) * x - 46) :
  (1 / 2 : ℝ) * x - 46 = 80 → x = 252 :=
by
  sorry

end novel_pages_l285_285313


namespace min_distance_curve_C1_on_line_C_l285_285590

noncomputable def curve_C1 (x y : ℝ) : Prop :=
(x^2) / 9 + (y^2) / 4 = 1

noncomputable def line_C (x y : ℝ) : Prop :=
x + 2*y - 10 = 0

theorem min_distance_curve_C1_on_line_C : 
    ∃ (M : ℝ × ℝ), 
    curve_C1 M.1 M.2 ∧ 
    ∀ (P : ℝ × ℝ), curve_C1 P.1 P.2 → dist (P.1, P.2) (line_C x y) ≤ sqrt 5 :=
sorry

end min_distance_curve_C1_on_line_C_l285_285590


namespace subtract_from_40_squared_l285_285797

theorem subtract_from_40_squared : 39 * 39 = 40 * 40 - 79 := by
  sorry

end subtract_from_40_squared_l285_285797


namespace tracy_sold_paintings_l285_285802

-- Definitions of conditions
def total_customers := 20
def first_group_customers := 4
def paintings_per_first_group_customer := 2
def second_group_customers := 12
def paintings_per_second_group_customer := 1
def third_group_customers := 4
def paintings_per_third_group_customer := 4

-- Statement of the problem
theorem tracy_sold_paintings :
  (first_group_customers * paintings_per_first_group_customer) +
  (second_group_customers * paintings_per_second_group_customer) +
  (third_group_customers * paintings_per_third_group_customer) = 36 :=
by
  sorry

end tracy_sold_paintings_l285_285802


namespace vector_subtraction_l285_285488

def vector1 : ℝ × ℝ := (3, -5)
def vector2 : ℝ × ℝ := (2, -6)
def scalar1 : ℝ := 4
def scalar2 : ℝ := 3

theorem vector_subtraction :
  (scalar1 • vector1 - scalar2 • vector2) = (6, -2) := by
  sorry

end vector_subtraction_l285_285488


namespace mutually_exclusive_non_opposite_events_l285_285400

def event1 (balls : List String) : Prop :=
  (balls.contains "Black" ∧ balls.length = 2)

def event2 (balls : List String) : Prop :=
  (balls.count ("Black" = ·) = 2)

def event3 (balls : List String) : Prop :=
  (balls.count ("Black" = ·) = 1)

def event4 (balls : List String) : Prop :=
  (balls.count ("Red" = ·) = 2)

axiom two_red_two_black : List String := ["Red", "Red", "Black", "Black"]

theorem mutually_exclusive_non_opposite_events :
  let selected_balls := [two_red_two_black.nth 0, two_red_two_black.nth 1] in
  event3 selected_balls ∧ ¬event2 selected_balls := 
sorry

end mutually_exclusive_non_opposite_events_l285_285400


namespace distance_between_lines_l285_285499

noncomputable def vector_2d := ℝ × ℝ
noncomputable def dot_product (v1 v2 : vector_2d) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)
noncomputable def proj (a d : vector_2d) : vector_2d := 
  let scalar := (dot_product a d) / (dot_product d d)
  (scalar * d.1, scalar * d.2)

noncomputable def vector_diff (v1 v2 : vector_2d) : vector_2d := (v1.1 - v2.1, v1.2 - v2.2)
noncomputable def vector_mag (v : vector_2d) : ℝ := real.sqrt ((v.1 ^ 2) + (v.2 ^ 2))

theorem distance_between_lines 
  (a b d : vector_2d)
  (ha : a = (3, -4)) 
  (hb : b = (2, -6)) 
  (hd : d = (2, -5)) 
  : vector_mag (vector_diff (vector_diff a b) (proj (vector_diff a b) d)) = real.sqrt 2349 / 29 := 
sorry

end distance_between_lines_l285_285499


namespace sine_graph_shift_right_l285_285000

-- Definitions
def sin_shifted_right (x: ℝ) (k: ℝ) : ℝ := Real.sin (x - k)

-- Theorem to be proved
theorem sine_graph_shift_right (x: ℝ) :
  (∀ x, sin_shifted_right x (π / 5) = Real.sin (x - π / 5)) → 
  shift_direction = "right" ∧ shift_amount = π / 5 :=
sorry

end sine_graph_shift_right_l285_285000


namespace tan_neg_240_eq_neg_sqrt_3_l285_285944

theorem tan_neg_240_eq_neg_sqrt_3 : Real.tan (-4 * Real.pi / 3) = -Real.sqrt 3 :=
by
  sorry

end tan_neg_240_eq_neg_sqrt_3_l285_285944


namespace cot_tan_simplify_l285_285739

theorem cot_tan_simplify : cot (20 * π / 180) + tan (10 * π / 180) = csc (20 * π / 180) :=
by
  sorry

end cot_tan_simplify_l285_285739


namespace parabolas_intersection_points_l285_285895

-- Definition of the parameters for the problem
def parabolas : List (ℤ × ℤ) := do
  a <- [-3, -2, -1, 0, 1, 2, 3]
  b <- [-4, -3, -2, -1, 0, 1, 2, 3, 4]
  pure (a, b)

-- We will define the focus point
def focus : (ℤ × ℤ) := (0, 0)

-- Define number of ways to select 2 parabolas from the given set.
def choose_two (n : ℕ) : ℕ := (n * (n - 1)) / 2

-- Compute number of ways to choose 2 parabolas from 45
def total_combinations := choose_two 45

-- Number of non-intersecting pairs for given a values (7 possibilities)
def non_intersecting_pairs := 7 * (16 : ℕ)

-- The number of intersection points calculation
def intersection_points := 2 * (total_combinations - non_intersecting_pairs)

-- Prove the number of intersection points is 1756.
theorem parabolas_intersection_points : intersection_points = 1756 := by
  sorry

end parabolas_intersection_points_l285_285895


namespace two_people_lying_l285_285317

def is_lying (A B C D : Prop) : Prop :=
  (A ↔ ¬B) ∧ (B ↔ ¬C) ∧ (C ↔ ¬B) ∧ (D ↔ ¬A)

theorem two_people_lying (A B C D : Prop) (LA LB LC LD : Prop) :
  is_lying A B C D → (LA → ¬A) → (LB → ¬B) → (LC → ¬C) → (LD → ¬D) → (LA ∧ LC ∧ ¬LB ∧ ¬LD) :=
by
  sorry

end two_people_lying_l285_285317


namespace cosine_BHD_l285_285260

variables (rectangular_solid : Type)
variables (D H G F B : rectangular_solid)
variables [Angle.rectangular_solid D H G 45]
variables [Angle.rectangular_solid F H B 45]

theorem cosine_BHD :
  ∀ (rectangular_solid : Type) 
    (D H G F B : rectangular_solid) 
    [Angle.rectangular_solid D H G 45] 
    [Angle.rectangular_solid F H B 45], 
  cos (angle B H D) = (real.sqrt 2) / 4 := 
begin
  sorry
end

end cosine_BHD_l285_285260


namespace area_of_triangle_ABC_l285_285893

theorem area_of_triangle_ABC :
  ∀ (ABC : Triangle) (BC : ℝ),
    ABC.angle B = 90 ∧ ABC.angle A = 60 ∧ BC = 16 →
    area ABC = 128 * Real.sqrt 3 :=
by
  intros ABC BC h
  sorry

end area_of_triangle_ABC_l285_285893


namespace find_a_and_b_l285_285681

theorem find_a_and_b (a b : ℤ) (h1 : 3 * (b + a^2) = 99) (h2 : 3 * a * b^2 = 162) : a = 6 ∧ b = -3 :=
sorry

end find_a_and_b_l285_285681


namespace find_w_when_x_is_six_l285_285760

variable {x w : ℝ}
variable (h1 : x = 3)
variable (h2 : w = 16)
variable (h3 : ∀ (x w : ℝ), x^4 * w^(1 / 4) = 162)

theorem find_w_when_x_is_six : x = 6 → w = 1 / 4096 :=
by
  intro hx
  sorry

end find_w_when_x_is_six_l285_285760


namespace probability_of_C_l285_285430

theorem probability_of_C {P : Type → ℝ} : 
  P(A) = 1/4 → 
  P(B) = 1/3 → 
  P(D) = 1/6 → 
  P(A) + P(B) + P(C) + P(D) = 1 → 
  P(C) = 1/4 :=
by
  sorry

end probability_of_C_l285_285430


namespace correct_answer_is_fB_l285_285863

-- Define all the functions
def fA (x : ℝ) := sqrt x
def fB (x : ℝ) := 1 / sqrt x
def fC (x : ℝ) := 1 / x
def fD (x : ℝ) := x^2 + x + 1

-- Define the range of each function
def range_fA := { y : ℝ | ∃ x, y = fA x }
def range_fB := { y : ℝ | ∃ x, y = fB x }
def range_fC := { y : ℝ | ∃ x, y = fC x }
def range_fD := { y : ℝ | ∃ x, y = fD x }

-- Define the correct range we are looking for
def target_range := (0 : ℝ, ⊤)

-- Final problem statement
theorem correct_answer_is_fB : range_fB = target_range := sorry

end correct_answer_is_fB_l285_285863


namespace probability_bottom_vertex_l285_285850

noncomputable def vertices := finset (fin 12) -- assuming 12 vertices in the dodecahedron

structure dodecahedron :=
(top bottom : vertices)
(adj : vertices → finset vertices) 
(is_valid : ∀ v ∈ adj top, (∃ w ∈ adj v, w = bottom)) -- the structure condition

open dodecahedron

-- probability for a random walk from top to bottom vertex via adjacent vertices
theorem probability_bottom_vertex (D : dodecahedron) :
  ∃ (A : vertices), A ∈ D.adj D.top ∧ (D.adj A).count (~_.bottom≤D) = 1/5 :=
by
  sorry

end probability_bottom_vertex_l285_285850


namespace paper_strip_length_l285_285087

-- Define the conditions
def paper_strip_width := 5 -- in cm
def initial_diameter := 2  -- in cm
def final_diameter := 10   -- in cm
def number_of_turns := 600 -- number of concentric cylindrical layers
def diameters_increase_uniformly := true

-- Define the question
def total_length_of_paper_strip : ℝ :=
  let radius_increment := (final_diameter - initial_diameter) / (2 * number_of_turns)
  let average_radius := (initial_diameter/2 + final_diameter/2) / 2
  let average_circumference := 2 * Real.pi * average_radius
  let total_length := average_circumference * number_of_turns
  total_length / 100  -- converting cm to meters

-- The proof statement
theorem paper_strip_length :
  total_length_of_paper_strip = 36 * Real.pi :=
by
  sorry

end paper_strip_length_l285_285087


namespace smallest_value_a_b_l285_285196

theorem smallest_value_a_b (a b : ℕ) (h : 2^6 * 3^9 = a^b) : a > 0 ∧ b > 0 ∧ (a + b = 111) :=
by
  sorry

end smallest_value_a_b_l285_285196


namespace repeated_digit_squares_l285_285158

theorem repeated_digit_squares :
  {n : ℕ | ∃ d : Fin 10, n = d ^ 2 ∧ (∀ m < n, m % 10 = d % 10)} ⊆ {0, 1, 4, 9} := by
  sorry

end repeated_digit_squares_l285_285158


namespace valid_triangle_inequality_l285_285615

theorem valid_triangle_inequality (a : ℝ) 
  (h1 : 4 + 6 > a) 
  (h2 : 4 + a > 6) 
  (h3 : 6 + a > 4) : 
  a = 5 :=
sorry

end valid_triangle_inequality_l285_285615


namespace abs_neg_two_l285_285345

def abs (x : ℤ) : ℤ := if x < 0 then -x else x

theorem abs_neg_two : abs (-2) = 2 :=
by
  sorry

end abs_neg_two_l285_285345


namespace probability_of_rolling_perfect_cube_probability_perfect_cube_l285_285342

-- Define what a standard 6-sided die entails
def is_standard_die (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

-- Define what a perfect cube is
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

-- Define an event where a perfect cube is rolled on a 6-sided die
def is_perfect_cube_rolled : Prop := is_standard_die 1 ∧ is_perfect_cube 1

-- Assertion of the probability of rolling a perfect cube
theorem probability_of_rolling_perfect_cube : ℚ :=
if h : is_perfect_cube_rolled then 1 / 6 else 0

-- Proof statement
theorem probability_perfect_cube : 
  probability_of_rolling_perfect_cube = 1 / 6 :=
sorry

end probability_of_rolling_perfect_cube_probability_perfect_cube_l285_285342


namespace find_f_comp_f_pi_l285_285960

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 1
  else Real.sin x - 2

theorem find_f_comp_f_pi : f (f Real.pi) = -3/4 :=
by
  sorry

end find_f_comp_f_pi_l285_285960


namespace gcd_condition_implies_equality_l285_285286

theorem gcd_condition_implies_equality (a b : ℤ) (h : ∀ n : ℤ, n ≥ 1 → Int.gcd (a + n) (b + n) > 1) : a = b :=
sorry

end gcd_condition_implies_equality_l285_285286


namespace cot_tan_simplify_l285_285738

theorem cot_tan_simplify : cot (20 * π / 180) + tan (10 * π / 180) = csc (20 * π / 180) :=
by
  sorry

end cot_tan_simplify_l285_285738


namespace minimize_frac_inv_l285_285559

theorem minimize_frac_inv (a b : ℕ) (h1: 4 * a + b = 30) (h2: a > 0) (h3: b > 0) :
  (a, b) = (5, 10) :=
sorry

end minimize_frac_inv_l285_285559


namespace Ed_lost_marbles_l285_285521

variable (D : ℕ)

def Ed_initial_marble (D : ℕ) : ℕ := D + 19
def Ed_current_marble (D : ℕ) : ℕ := D + 8

theorem Ed_lost_marbles : Ed_initial_marble D - Ed_current_marble D = 11 :=
by 
  simp [Ed_initial_marble, Ed_current_marble]
  sorry

end Ed_lost_marbles_l285_285521


namespace hyperbola_eccentricity_l285_285223

variables {x y m : ℝ}

-- Define the hyperbola C
def hyperbola_C := ∀ {x : ℝ}, ∃ {y : ℝ}, (x ^ 2 / m ^ 2) - (y ^ 2 / (m ^ 2 - 1)) = 1

-- Define the eccentricity
def eccentricity (a c : ℝ) : ℝ := c / a

-- Given conditions
def a_squared := 4
def b_squared := 3
def c_squared := a_squared + b_squared
def e := eccentricity (real.sqrt a_squared) (real.sqrt c_squared)

-- Prove that the eccentricity is equal to sqrt(7)/2
theorem hyperbola_eccentricity : 
  ∀ (m : ℝ), (∃ P : ℝ × ℝ, (hyperbola_C ∧ (PF_1 ⊥ PF_2 ∧ area_PF_1F_2 = 3))) 
  → e = real.sqrt 7 / 2 :=
sorry

end hyperbola_eccentricity_l285_285223


namespace sum_f_zero_l285_285552

def f (n : ℕ) : ℝ := Real.tan ((n / 2 : ℝ) * Real.pi + Real.pi / 4)

theorem sum_f_zero : (Finset.range 2016).sum f = 0 := 
by sorry

end sum_f_zero_l285_285552


namespace repertoire_songs_l285_285788

theorem repertoire_songs (A B C D : ℕ) (hA : A = 5) (hB : B = 7) (hC : C = 2) (hD : D = 8) : 
  A + B + C + 2 * D = 30 :=
by 
  rw [hA, hB, hC, hD] 
  norm_num

end repertoire_songs_l285_285788


namespace arrangement_count_l285_285104

def Student := {name : String} 
noncomputable def A := Student.mk "A"
noncomputable def B := Student.mk "B"
noncomputable def C := Student.mk "C"
noncomputable def D := Student.mk "D"
noncomputable def E := Student.mk "E"

def is_adj (x y : Student) (lst : List Student) : Prop :=
  ∃ i, (lst.nth i = some x ∧ lst.nth (i+1) = some y) ∨ (lst.nth i = some y ∧ lst.nth (i+1) = some x)

def exactly_one_between (x y : Student) (lst : List Student) : Prop :=
  ∃ i, (lst.nth i = some x ∧ lst.nth (i+2) = some y) ∨ (lst.nth i = some y ∧ lst.nth (i+2) = some x)

def total_arrangements : List Student :=
  [A, B, C, D, E].permutations

def valid_arrangements (arrangements : List (List Student)) : List (List Student) :=
  arrangements.filter (λ lst, (is_adj A B lst) ∧ (exactly_one_between A C lst))

theorem arrangement_count : valid_arrangements total_arrangements).length = 20 := sorry

end arrangement_count_l285_285104


namespace water_flow_into_sea_per_min_l285_285453

def depth : ℝ := 5
def width : ℝ := 19
def flow_rate_kmph : ℝ := 4
def flow_rate_m_per_min (rate_kmph : ℝ) : ℝ := (rate_kmph * 1000) / 60

theorem water_flow_into_sea_per_min 
  (d : ℝ) (w : ℝ) (fr_kmph : ℝ) 
  (area : ℝ := d * w) 
  (fr_m_per_min : ℝ := flow_rate_m_per_min fr_kmph) :
  area * fr_m_per_min = 6333.65 :=
sorry

end water_flow_into_sea_per_min_l285_285453


namespace loss_percentage_correct_l285_285353

def cost_price : ℝ := 1800
def selling_price : ℝ := 1430

def loss : ℝ := cost_price - selling_price

def loss_percentage : ℝ := (loss / cost_price) * 100

theorem loss_percentage_correct : loss_percentage = 20.56 := by
  sorry

end loss_percentage_correct_l285_285353


namespace sum_of_values_for_one_solution_l285_285383

noncomputable def sum_of_a_values (a1 a2 : ℝ) : ℝ :=
  a1 + a2

theorem sum_of_values_for_one_solution :
  ∃ a1 a2 : ℝ, 
  (∀ x : ℝ, 4 * x^2 + (a1 + 8) * x + 9 = 0 ∨ 4 * x^2 + (a2 + 8) * x + 9 = 0) ∧
  ((a1 + 8)^2 - 144 = 0) ∧ ((a2 + 8)^2 - 144 = 0) ∧
  sum_of_a_values a1 a2 = -16 :=
by
  sorry

end sum_of_values_for_one_solution_l285_285383


namespace triangle_internal_angles_l285_285623

theorem triangle_internal_angles (A B C : ℝ) (h1 : sin A + cos A = sqrt 2) (h2 : sqrt 3 * cos A = -sqrt 2 * cos (π - B))
    (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (h_sum : A + B + C = π) : 
    A = π / 4 ∧ B = π / 6 ∧ C = 7 * π / 12 :=
begin
  sorry
end

end triangle_internal_angles_l285_285623


namespace trains_cross_time_l285_285415

def L : ℕ := 120 -- Length of each train in meters

def t1 : ℕ := 10 -- Time for the first train to cross the telegraph post in seconds
def t2 : ℕ := 12 -- Time for the second train to cross the telegraph post in seconds

def V1 : ℕ := L / t1 -- Speed of the first train (in m/s)
def V2 : ℕ := L / t2 -- Speed of the second train (in m/s)

def Vr : ℕ := V1 + V2 -- Relative speed when traveling in opposite directions

def TotalDistance : ℕ := 2 * L -- Total distance when both trains cross each other

def T : ℚ := TotalDistance / Vr -- Time for the trains to cross each other

theorem trains_cross_time : T = 11 := sorry

end trains_cross_time_l285_285415


namespace competitive_exam_candidates_l285_285410

theorem competitive_exam_candidates (x : ℝ)
  (A_selected : ℝ := 0.06 * x) 
  (B_selected : ℝ := 0.07 * x) 
  (h : B_selected = A_selected + 81) :
  x = 8100 := by
  sorry

end competitive_exam_candidates_l285_285410


namespace grandson_age_l285_285304

theorem grandson_age (M S G : ℕ) (h1 : M = 2 * S) (h2 : S = 2 * G) (h3 : M + S + G = 140) : G = 20 :=
by 
  sorry

end grandson_age_l285_285304


namespace lcm_of_18_and_30_l285_285028

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end lcm_of_18_and_30_l285_285028


namespace dress_price_l285_285007

namespace VanessaClothes

def priceOfDress (total_revenue : ℕ) (num_dresses num_shirts price_of_shirt : ℕ) : ℕ :=
  (total_revenue - num_shirts * price_of_shirt) / num_dresses

theorem dress_price :
  priceOfDress 69 7 4 5 = 7 :=
by 
  calc
    priceOfDress 69 7 4 5 = (69 - 4 * 5) / 7 : rfl
                     ... = 49 / 7 : by norm_num
                     ... = 7 : by norm_num

end VanessaClothes

end dress_price_l285_285007


namespace lattice_points_at_distance_5_l285_285274

theorem lattice_points_at_distance_5 :
  ∃ S : Finset (ℤ × ℤ × ℤ), (∀ p ∈ S, let ⟨x, y, z⟩ := p in x^2 + y^2 + z^2 = 25) ∧ S.card = 18 := by
  sorry

end lattice_points_at_distance_5_l285_285274


namespace number_of_even_multiples_of_3_l285_285123

theorem number_of_even_multiples_of_3 :
  ∃ n, n = (198 - 6) / 6 + 1 := by
  sorry

end number_of_even_multiples_of_3_l285_285123


namespace exists_x_y_not_divisible_by_3_l285_285289

theorem exists_x_y_not_divisible_by_3 (k : ℕ) (h_pos : 0 < k) :
  ∃ x y : ℤ, (x^2 + 2 * y^2 = 3^k) ∧ (x % 3 ≠ 0) ∧ (y % 3 ≠ 0) := 
sorry

end exists_x_y_not_divisible_by_3_l285_285289


namespace sum_rational_roots_h_l285_285536

def h (x : ℚ) : ℚ := x^3 - 8 * x^2 + 15 * x - 6

theorem sum_rational_roots_h : 
  (∑ r in (multiset.filter (λ x, h x = 0) (multiset.of_list [1, -1, 2, -2, 3, -3, 6, -6])), r) = 2 := 
  sorry

end sum_rational_roots_h_l285_285536


namespace type_a_time_approx_17_l285_285409

-- Definitions of the conditions
def total_questions : ℕ := 200
def type_a_questions : ℕ := 10
def type_b_questions : ℕ := total_questions - type_a_questions
def total_time_minutes : ℕ := 3 * 60
def time_per_type_b_problem : ℝ := 180 / 210

-- We need to prove that the total time spent on Type A problems is approximately 17 minutes
theorem type_a_time_approx_17 :
  let time_per_type_a_problem := 2 * time_per_type_b_problem in
  let total_time_type_a := type_a_questions * time_per_type_a_problem in
  abs (total_time_type_a - 17) < 1 :=
by
  sorry

end type_a_time_approx_17_l285_285409


namespace incorrect_statement_l285_285185

def data_set : list ℕ := [5, 8, 8, 9, 10]

def mean (l : list ℕ) := (list.sum l) / l.length

def mode (l : list ℕ) := l.sort.nth (l.length / 2) -- classical median function

def median (l : list ℕ) := l.sort.nth (l.length / 2)

def variance (l : list ℕ) :=
  let m := mean l in
  (list.sum (l.map (λ x, (x - m) ^ 2))) / l.length

theorem incorrect_statement :
  ¬(variance data_set = 8) :=
by
  sorry

end incorrect_statement_l285_285185


namespace minimum_value_at_1_range_of_a_l285_285221

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + log x

-- Definition for domain of real numbers strictly greater than 0
def domain (x : ℝ) : Prop := 0 < x

-- First problem: a = 1 implies the minimum value of f(x) at x = 1
theorem minimum_value_at_1 (x : ℝ) (h₁ : domain x) : 
  f 1 1 = -2 := 
sorry

-- Second problem: range of values for a given the minimum value condition
theorem range_of_a (a x : ℝ) (h₂ : domain x) (h₃ : 1 ≤ x ∧ x ≤ exp 1) :
  (∀ x, f a x ≥ -2) → (1 ≤ a) :=
sorry

end minimum_value_at_1_range_of_a_l285_285221


namespace symmetric_graph_to_2_power_x_is_2_power_neg_x_l285_285464

theorem symmetric_graph_to_2_power_x_is_2_power_neg_x (f : ℝ → ℝ) :
  (∀ x, f x = 2^x) →
  (∃ g : ℝ → ℝ, (∀ x, g x = 2^(-x))) :=
by
  intro h
  use (λ x, 2^(-x))
  intro x
  sorry

end symmetric_graph_to_2_power_x_is_2_power_neg_x_l285_285464


namespace longest_diagonal_is_20_l285_285451

noncomputable def length_of_longest_diagonal (area : ℝ) (ratio1 ratio2 : ℝ) : ℝ :=
  let x := (area * 2) / (ratio1 * ratio2)
  in ratio1 * (x / 6) -- as we derived 150 = 6 * x^2, leading to x/6 part

theorem longest_diagonal_is_20 :
  length_of_longest_diagonal 150 4 3 = 20 := by
{
  -- specify the structure of the proof and then leave it as a sorry
  sorry
}

end longest_diagonal_is_20_l285_285451


namespace largest_times_smallest_product_l285_285402

def digits := {2, 4, 6, 8, 9}

noncomputable def largest_two_digit_number : ℕ :=
  98

noncomputable def smallest_two_digit_number : ℕ :=
  24

theorem largest_times_smallest_product : 
  largest_two_digit_number * smallest_two_digit_number = 2352 :=
by
  sorry

end largest_times_smallest_product_l285_285402


namespace range_of_a_l285_285687

noncomputable def f : ℝ → ℝ :=
λ x, if h : x < 2 then 2^x else x^2

theorem range_of_a (a : ℝ) (h : f (a + 1) ≥ f (2 * a - 1)) : a ≤ 2 :=
by
  sorry

end range_of_a_l285_285687


namespace pencils_problem_l285_285789

theorem pencils_problem (x : ℕ) :
  2 * x + 6 * 3 + 2 * 1 = 24 → x = 2 :=
by
  sorry

end pencils_problem_l285_285789


namespace find_three_digit_number_l285_285787

theorem find_three_digit_number : 
  ∀ (c d e : ℕ), 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ 0 ≤ e ∧ e < 10 ∧ 
  (10 * c + d) / 99 + (100 * c + 10 * d + e) / 999 = 44 / 99 → 
  100 * c + 10 * d + e = 400 :=
by {
  sorry
}

end find_three_digit_number_l285_285787


namespace father_twice_as_old_in_years_l285_285137

-- Conditions
def father_age : ℕ := 42
def son_age : ℕ := 14
def years : ℕ := 14

-- Proof statement
theorem father_twice_as_old_in_years : (father_age + years) = 2 * (son_age + years) :=
by
  -- Proof content is omitted as per the instruction.
  sorry

end father_twice_as_old_in_years_l285_285137


namespace volume_of_quadrilateral_pyramid_l285_285947

theorem volume_of_quadrilateral_pyramid 
  (r : ℝ) (α : ℝ) 
  (h_r_pos : r > 0) 
  (h_alpha_pos : α > 0) :
  let V := (2 * r^3 * (sqrt (2 * (tan α)^2 + 1) + 1)^3) / (3 * (tan α)^2)
  in V = (2 * r^3 * (sqrt (2 * (tan α)^2 + 1) + 1)^3) / (3 * (tan α)^2) :=
by
  sorry

end volume_of_quadrilateral_pyramid_l285_285947


namespace anyas_hair_loss_l285_285103

theorem anyas_hair_loss (H : ℝ) 
  (washes_hair_loss : H > 0) 
  (brushes_hair_loss : H / 2 > 0) 
  (grows_back : ∃ h : ℝ, h = 49 ∧ H + H / 2 + 1 = h) :
  H = 32 :=
by
  sorry

end anyas_hair_loss_l285_285103


namespace no_real_solutions_cubic_eq_l285_285932

theorem no_real_solutions_cubic_eq : ∀ x : ℝ, ¬ (∃ (y : ℝ), y = x^(1/3) ∧ y = 15 / (6 - y)) :=
by
  intro x
  intro hexist
  obtain ⟨y, hy1, hy2⟩ := hexist
  have h_cubic : y * (6 - y) = 15 := by sorry -- from y = 15 / (6 - y)
  have h_quad : y^2 - 6 * y + 15 = 0 := by sorry -- after expanding y(6 - y) = 15
  sorry -- remainder to show no real solution due to negative discriminant

end no_real_solutions_cubic_eq_l285_285932


namespace find_peters_magnets_l285_285093

namespace MagnetProblem

variable (AdamInitial Peter : ℕ) (AdamRemaining : ℕ)

axiom Adam_initial_has_18_magnets : AdamInitial = 18
axiom Adam_gives_away_third : AdamRemaining = AdamInitial - (AdamInitial / 3)
axiom Adam_has_half_of_Peter : AdamRemaining = Peter / 2

theorem find_peters_magnets : Peter = 24 :=
by
  have h1 : AdamRemaining = 18 - (18 / 3) := by rw [Adam_initial_has_18_magnets, nat.div_eq_of_eq_mul_right (by decide : 3 > 0) rfl]
  have h2 : AdamRemaining = 18 - 6 := by rw [h1]
  have h3 : AdamRemaining = 12 := by norm_num at h2
  have h4 : 12 = Peter / 2 := by rw [Adam_has_half_of_Peter, h3]
  have h5 : Peter = 24 := by linarith [h4]
  exact h5

end find_peters_magnets_l285_285093


namespace tangent_length_to_circle_l285_285964

-- Definitions capturing the conditions
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y + 1 = 0
def line_l (x y a : ℝ) : Prop := x + a * y - 1 = 0
def point_A (a : ℝ) : ℝ × ℝ := (-4, a)

-- Main theorem statement proving the question against the answer
theorem tangent_length_to_circle (a : ℝ) (x y : ℝ) (hC : circle_C x y) (hl : line_l 2 1 a) :
  (a = -1) -> (point_A a = (-4, -1)) -> ∃ b : ℝ, b = 6 := 
sorry

end tangent_length_to_circle_l285_285964


namespace coordinates_of_P_l285_285570

theorem coordinates_of_P (m : ℝ) (P : ℝ × ℝ) :
  P = (2 * m, m + 8) ∧ 2 * m = 0 → P = (0, 8) := by
  intros hm
  sorry

end coordinates_of_P_l285_285570


namespace Carson_skipped_times_l285_285484

variable (length width total_circles actual_distance perimeter distance_skipped : ℕ)
variable (total_distance : ℕ)

def perimeter_calculation (length width : ℕ) : ℕ := 2 * (length + width)

def total_distance_calculation (total_circles perimeter : ℕ) : ℕ := total_circles * perimeter

def distance_skipped_calculation (total_distance actual_distance : ℕ) : ℕ := total_distance - actual_distance

def times_skipped_calculation (distance_skipped perimeter : ℕ) : ℕ := distance_skipped / perimeter

theorem Carson_skipped_times (h_length : length = 600) 
                             (h_width : width = 400) 
                             (h_total_circles : total_circles = 10) 
                             (h_actual_distance : actual_distance = 16000) 
                             (h_perimeter : perimeter = perimeter_calculation length width) 
                             (h_total_distance : total_distance = total_distance_calculation total_circles perimeter) 
                             (h_distance_skipped : distance_skipped = distance_skipped_calculation total_distance actual_distance) :
                             times_skipped_calculation distance_skipped perimeter = 2 := 
by
  simp [perimeter_calculation, total_distance_calculation, distance_skipped_calculation, times_skipped_calculation]
  sorry

end Carson_skipped_times_l285_285484


namespace tracy_sold_paintings_l285_285803

-- Definitions of conditions
def total_customers := 20
def first_group_customers := 4
def paintings_per_first_group_customer := 2
def second_group_customers := 12
def paintings_per_second_group_customer := 1
def third_group_customers := 4
def paintings_per_third_group_customer := 4

-- Statement of the problem
theorem tracy_sold_paintings :
  (first_group_customers * paintings_per_first_group_customer) +
  (second_group_customers * paintings_per_second_group_customer) +
  (third_group_customers * paintings_per_third_group_customer) = 36 :=
by
  sorry

end tracy_sold_paintings_l285_285803


namespace number_of_mixed_pairs_l285_285541

/-- Given 2n men and 2n women, the number of ways to form n mixed pairs is (2n!)^2 / (n! * 2^n) -/
theorem number_of_mixed_pairs (n : ℕ) : 
  let men_and_women_pairs := (2 * n)!
  let mixed_pairs_count := men_and_women_pairs * men_and_women_pairs / (n! * 2^n)
  mixed_pairs_count = (2 * n)! * (2 * n)! / (n! * 2 ^ n) :=
by
  sorry

end number_of_mixed_pairs_l285_285541


namespace sufficient_condition_l285_285580

def f (x : ℝ) : ℝ := Real.exp (x^2 - 3 * x)

theorem sufficient_condition (x : ℝ) (h : 0 < x ∧ x < 1) : f x < 1 :=
by
  have h₀ := h.1
  have h₁ := h.2
  -- We need to show that x^2 - 3x < 0 for 0 < x < 1
  have : x^2 < 3*x :=
    calc
      x^2 < x * 3 := by linarith [h₀, h₁]
      ... = 3*x : by ring
  have : Real.exp (x^2 - 3*x) < 1 := by
    rw [Real.exp_lt_one_iff]
    exact this
  exact this

end sufficient_condition_l285_285580


namespace acute_angle_l285_285993

/-- Given that the terminal side of the acute angle α passes through point P (cos 40° + 1, sin 40°),
prove that the acute angle α is 20°. -/
theorem acute_angle (α : ℝ) (h : 0 < α ∧ α < π / 2)
  (hP : ∃ x y, cos α = x / (x^2 + y^2) ∧ sin α = y / (x^2 + y^2)
         ∧ x = cos 40 + 1 ∧ y = sin 40) :
  α = 20 := sorry

end acute_angle_l285_285993


namespace largest_angle_of_pentagon_l285_285849

theorem largest_angle_of_pentagon (x : ℝ) : 
  (2*x + 2) + 3*x + 4*x + 5*x + (6*x - 2) = 540 → 
  6*x - 2 = 160 :=
by
  intro h
  sorry

end largest_angle_of_pentagon_l285_285849


namespace shortest_student_height_l285_285380

def male_students : List ℝ := [161.5, 154.3, 143.7]
def female_students : List ℝ := [160.1, 158.0, 153.5, 147.8]
def all_students : List ℝ := male_students ++ female_students

theorem shortest_student_height :
  all_students.minimum = 143.7 :=
sorry

end shortest_student_height_l285_285380


namespace oil_production_per_capita_l285_285010

-- Defining the given conditions
def oilProduction_West : ℝ := 55084
def oilProduction_NonWest : ℝ := 1480689
def population_NonWest : ℝ := 6900000
def oilProduction_Russia_part : ℝ := 13737.1
def percent_Russia_part : ℝ := 9 / 100
def population_Russia : ℝ := 147000000

-- Defining the correct answers
def perCapita_West : ℝ := oilProduction_West
def perCapita_NonWest : ℝ := oilProduction_NonWest / population_NonWest
def totalOilProduction_Russia : ℝ := oilProduction_Russia_part / percent_Russia_part
def perCapita_Russia : ℝ := totalOilProduction_Russia / population_Russia

-- Statement of the theorem
theorem oil_production_per_capita :
    (perCapita_West = 55084) ∧
    (perCapita_NonWest = 214.59) ∧
    (perCapita_Russia = 1038.33) :=
by sorry

end oil_production_per_capita_l285_285010


namespace participants_initial_count_l285_285258

theorem participants_initial_count (n : ℕ) (h1 : 0.60 * n = n - 0.40 * n)
    (h2 : 0.40 * n * (1 / 4) = 0.10 * n)
    (h3 : 0.10 * n = 30) :
    n = 300 :=
by
  sorry

end participants_initial_count_l285_285258


namespace avg_temp_correct_l285_285078

-- Defining the temperatures for each day from March 1st to March 5th
def day_1_temp := 55.0
def day_2_temp := 59.0
def day_3_temp := 60.0
def day_4_temp := 57.0
def day_5_temp := 64.0

-- Calculating the average temperature
def avg_temp := (day_1_temp + day_2_temp + day_3_temp + day_4_temp + day_5_temp) / 5.0

-- Proving that the average temperature equals 59.0°F
theorem avg_temp_correct : avg_temp = 59.0 := sorry

end avg_temp_correct_l285_285078


namespace expand_polynomial_l285_285919

theorem expand_polynomial (x : ℝ) :
  (x + 4) * (x - 9) = x^2 - 5 * x - 36 := 
sorry

end expand_polynomial_l285_285919


namespace min_value_expression_l285_285977

theorem min_value_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : ∀ x y : ℝ, x + y + a = 0 → (x - b)^2 + (y - 1)^2 = 2) : 
    (∃ c : ℝ,  c = 4 ∧ ∀ a b : ℝ, (0 < a → 0 < b → x + y + a = 0 → (x - b)^2 + (y - 1)^2 = 2 →  (3 - 2 * b)^2 / (2 * a) ≥ c)) :=
by
  sorry

end min_value_expression_l285_285977


namespace range_f_l285_285905

-- Definition of the function y = 3^x
def f (x : ℝ) : ℝ := 3^x

-- The range of the function f
theorem range_f : set.Ioi 0 = {y : ℝ | ∃ x : ℝ, f x = y} :=
sorry

end range_f_l285_285905


namespace ratio_of_areas_l285_285369

theorem ratio_of_areas {α β γ a b c : ℝ}
  (h₁: α = 1/3 * a)
  (h₂: β = 1/3 * b)
  (h₃: γ = 1/3 * c)
  (median_relation: (3* β) = a ∧ (3 * γ) = b ∧ (3 * α) = c ):
  let area_original := α * β / 2 in
  let area_median_constructed := (4 / 3) * area_original in
  area_original / area_median_constructed = 3 / 4 :=
by {
  sorry
}

end ratio_of_areas_l285_285369


namespace cot_tan_simplify_l285_285743

theorem cot_tan_simplify : cot (20 * π / 180) + tan (10 * π / 180) = csc (20 * π / 180) :=
by
  sorry

end cot_tan_simplify_l285_285743


namespace new_marketing_percentage_l285_285474

theorem new_marketing_percentage 
  (total_students : ℕ)
  (initial_finance_percentage : ℕ)
  (initial_marketing_percentage : ℕ)
  (initial_operations_management_percentage : ℕ)
  (new_finance_percentage : ℕ)
  (operations_management_percentage : ℕ)
  (total_percentage : ℕ) :
  total_students = 5000 →
  initial_finance_percentage = 85 →
  initial_marketing_percentage = 80 →
  initial_operations_management_percentage = 10 →
  new_finance_percentage = 92 →
  operations_management_percentage = 10 →
  total_percentage = 175 →
  initial_marketing_percentage - (new_finance_percentage - initial_finance_percentage) = 73 :=
by
  sorry

end new_marketing_percentage_l285_285474


namespace smallest_positive_period_l285_285907

def omega := (π / 2)
def f (x : ℝ) := 3 * Real.cos (omega * x - π / 8)
def period := (2 * π / omega)

theorem smallest_positive_period : period = 4 := by
  sorry

end smallest_positive_period_l285_285907


namespace geometric_sequence_first_term_l285_285359

variable {b q a d : ℝ}

-- Hypotheses and Conditions
hypothesis h_geom_seq : ∀ n, b * q ^ n
hypothesis h_arith_seq : ∀ n, a + n * d
hypothesis h_product : b * (b * q) * (b * q^2) = 64

-- Theorems to prove b == 8/3
theorem geometric_sequence_first_term :
  (∃ b q a d : ℝ, (∀ n, b * q ^ n = a + n * d) ∧ (b * (b * q) * (b * q^2) = 64)) → b = 8/3 := by 
  sorry

end geometric_sequence_first_term_l285_285359


namespace meters_of_cloth_sold_l285_285089

-- Definitions based on conditions
def total_selling_price : ℕ := 8925
def profit_per_meter : ℕ := 20
def cost_price_per_meter : ℕ := 85
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- The proof statement
theorem meters_of_cloth_sold : ∃ x : ℕ, selling_price_per_meter * x = total_selling_price ∧ x = 85 := by
  sorry

end meters_of_cloth_sold_l285_285089


namespace strawberry_picking_l285_285042

noncomputable def entrance_fee : ℝ := 4
noncomputable def price_per_pound : ℝ := 20
noncomputable def num_people : ℝ := 3
noncomputable def total_paid : ℝ := 128 

theorem strawberry_picking :
  let total_entrance_fees := num_people * entrance_fee in
  let total_cost_strawberries := total_paid + total_entrance_fees in
  let weight_in_pounds := total_cost_strawberries / price_per_pound in
  weight_in_pounds = 7 :=
by
  sorry

end strawberry_picking_l285_285042


namespace sum_arithmetic_sequence_l285_285641

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
    ∃ d, ∀ n, a (n+1) = a n + d

-- The conditions
def condition_1 (a : ℕ → ℝ) : Prop :=
    (a 1 + a 2 + a 3 = 6)

def condition_2 (a : ℕ → ℝ) : Prop :=
    (a 10 + a 11 + a 12 = 9)

-- The Theorem statement
theorem sum_arithmetic_sequence :
    is_arithmetic_sequence a →
    condition_1 a →
    condition_2 a →
    (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 = 30) :=
by
  intro h1 h2 h3
  sorry

end sum_arithmetic_sequence_l285_285641


namespace sum_of_divisors_gt_5_eq_15_l285_285942

def is_divisor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

def positive_divisors (n : ℕ) : set ℕ := {d | is_divisor n d}

def common_divisors (a b : ℕ) : set ℕ := {d | is_divisor a d ∧ is_divisor b d}

def sum_of_divisors_gt_5 (a b : ℕ) : ℕ :=
  (common_divisors a b).filter (λ d, d > 5).sum

theorem sum_of_divisors_gt_5_eq_15 :
  sum_of_divisors_gt_5 75 45 = 15 :=
by
  sorry

end sum_of_divisors_gt_5_eq_15_l285_285942


namespace max_min_values_of_function_l285_285781

theorem max_min_values_of_function :
  (∀ x, 0 ≤ 2 * Real.sin x + 2 ∧ 2 * Real.sin x + 2 ≤ 4) ↔ (∃ x, 2 * Real.sin x + 2 = 0) ∧ (∃ y, 2 * Real.sin y + 2 = 4) :=
by
  sorry

end max_min_values_of_function_l285_285781


namespace necessity_of_B_for_C_l285_285998

variables {Point Line Plane : Type}
variables (l m : Line) (α β : Plane)

-- Definitions of the properties given in the problem conditions
def propA : Prop := (∃ X : Point, X ∈ l ∧ X ∈ m ∧ X ∈ α ∧ X ∉ β)
def propB : Prop := (∃ X : Point, (X ∈ l ∨ X ∈ m) ∧ X ∈ β)
def propC : Prop := (∃ X : Point, X ∈ α ∧ X ∈ β)

-- The statement combining the conditions and the correct answer to be proved
theorem necessity_of_B_for_C (hA : propA l m α β) : (propC α β → propB l m β) ∧ ¬(propB l m β → propC α β) :=
by
  sorry

end necessity_of_B_for_C_l285_285998


namespace total_students_eq_seventeen_l285_285625

theorem total_students_eq_seventeen 
    (N : ℕ)
    (initial_students : N - 1 = 16)
    (avg_first_day : 77 * (N - 1) = 77 * 16)
    (avg_second_day : 78 * N = 78 * N)
    : N = 17 :=
sorry

end total_students_eq_seventeen_l285_285625


namespace find_a_and_mono_l285_285547

open Real

noncomputable def f (x : ℝ) (a : ℝ) := (a * 2^x + a - 2) / (2^x + 1)

theorem find_a_and_mono :
  (∀ x : ℝ, f x a + f (-x) a = 0) →
  a = 1 ∧ f 3 1 = 7 / 9 ∧ ∀ x1 x2 : ℝ, x1 < x2 → f x1 1 < f x2 1 :=
by
  sorry

end find_a_and_mono_l285_285547


namespace ice_melting_volume_l285_285829

theorem ice_melting_volume (V : ℝ) (h₁ : V / 4 = V_1) (h₂ : V_1 / 4 = 0.2) : V = 3.2 :=
by
  have h₃ : V_1 = V / 4 := by sorry
  have h₄ : V_1 / 4 = 0.2 := by sorry
  have h₅ : V / 16 = 0.2 := by sorry
  calc V = 0.2 * 16 : by sorry
      ... = 3.2 : by sorry

end ice_melting_volume_l285_285829


namespace find_z_coordinate_l285_285074

noncomputable def z_coordinate_on_line (p1 p2 : ℝ × ℝ × ℝ) (x_target : ℝ) : ℝ :=
  let direction_vector := (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)
  let t := (x_target - p1.1) / direction_vector.1
  p1.3 + t * direction_vector.3

theorem find_z_coordinate : 
  z_coordinate_on_line (1, 3, 4) (4, 2, 0) 7 = -4 := 
by 
  unfold z_coordinate_on_line 
  rw [prod.smul_mk] 
  -- computation steps and simplifications
  sorry

end find_z_coordinate_l285_285074


namespace intersection_of_A_and_B_l285_285193

def A : set ℝ := { x | 2 * x - 4 < 0 }
def B : set ℝ := { x | real.log10 x < 1 }

theorem intersection_of_A_and_B : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_B_l285_285193


namespace product_of_A_and_B_l285_285939

theorem product_of_A_and_B (A B : ℕ) (h1 : 3 / 9 = 6 / A) (h2 : B / 63 = 6 / A) : A * B = 378 :=
  sorry

end product_of_A_and_B_l285_285939


namespace sqrt_expression_equality_l285_285058

theorem sqrt_expression_equality :
  ∃ (a b c : ℤ), c = 3 ∧ ab = 15 ∧ a ^ 2 + 3 * (b ^ 2) = 72 ∧ a + b + c = 14 :=
by
  let c := 3
  have H1 : c = 3 := rfl
  use 6, 5, c
  split
  · exact H1
  split
  · exact dec_trivial
  split
  · exact dec_trivial
  · exact dec_trivial
  sorry

end sqrt_expression_equality_l285_285058


namespace calculate_DE_length_l285_285805

variable (A B C D E : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variable {AB AC BC : ℝ}
variable (h1 : AB = 24)
variable (h2 : AC = 26)
variable (h3 : BC = 22)
variable (DE_parallel_BC : parallel DE BC)
variable (DE_contains_incenter : contains_incenter DE)

theorem calculate_DE_length (h1 : AB = 24) (h2 : AC = 26) (h3 : BC = 22) 
  (DE_parallel_BC : DE.parallel BC) (DE_contains_incenter : DE.contains_incenter) : 
  DE = 275 / 18 :=
by
  sorry

end calculate_DE_length_l285_285805


namespace condition_a_neither_necessary_nor_sufficient_for_b_l285_285569

theorem condition_a_neither_necessary_nor_sufficient_for_b {x y : ℝ} (h : ¬(x = 1 ∧ y = 2)) (k : ¬(x + y = 3)) : ¬((x ≠ 1 ∧ y ≠ 2) ↔ (x + y ≠ 3)) :=
by
  sorry

end condition_a_neither_necessary_nor_sufficient_for_b_l285_285569


namespace factorial_division_l285_285505
-- Definition of factorial
def fact : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * fact n

-- Problem statement
theorem factorial_division : fact 12 / fact 11 = 12 :=
by sorry

end factorial_division_l285_285505


namespace lcm_18_30_eq_90_l285_285015

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end lcm_18_30_eq_90_l285_285015


namespace find_a_for_constant_term_l285_285299

theorem find_a_for_constant_term (a : ℝ) (h : a > 0) :
  (let term := (x : ℝ) ^ 2 + a / x.sqrt in ((term^5).coeff 0 = 80) → a = 2) :=
begin
  sorry
end

end find_a_for_constant_term_l285_285299


namespace sequence_correct_l285_285555

noncomputable def sequence (n : ℕ) : ℕ :=
  if h : n = 1 then 5 else 2^(n-1)

def sum_of_first_n_terms (n : ℕ) : ℕ := 3 + 2^n

def a (n : ℕ) : ℕ :=
  if n = 1 then sum_of_first_n_terms n
  else sum_of_first_n_terms n - sum_of_first_n_terms (n - 1)

theorem sequence_correct : ∀ n, sequence n = a n :=
by
  intros n
  sorry

end sequence_correct_l285_285555


namespace determine_k_l285_285571

noncomputable def roots_of_quadratic (a b c : ℝ) : ℝ × ℝ :=
((-b + Real.sqrt (b * b - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b * b - 4 * a * c)) / (2 * a))

theorem determine_k : ∃ k : ℝ, (k = 3 ∨ k = -11) ∧
  let (x1, x2) := roots_of_quadratic 2 k (-2 * k + 1) in
    x1^2 + x2^2 = 29 / 4 :=
begin
  sorry
end

end determine_k_l285_285571


namespace largest_median_value_l285_285194

noncomputable def largest_median_possible : ℕ :=
  let given_numbers := [3, 5, 6, 9, 10, 4, 7]
  let max_val := 10
  let required_length := 11
  ↑(10) -- The correct answer is 10

theorem largest_median_value :
  ∀ (eleven_list : List ℕ),
    (∀ x ∈ eleven_list, x > 0 ∧ x ≤ 10) ∧
    given_numbers ⊆ eleven_list ∧
    eleven_list.length = 11 →
    List.median eleven_list = largest_median_possible :=
by
  sorry

end largest_median_value_l285_285194


namespace simplify_cot_tan_l285_285712

theorem simplify_cot_tan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := by
  sorry

end simplify_cot_tan_l285_285712


namespace max_p_solution_l285_285124

theorem max_p_solution :
  ∃ (x ∈ ℝ), ∀ (p : ℝ), (2 * cos (2 * real.pi - real.pi * x^2 / 6) * cos (real.pi / 3 * sqrt (9 - x^2)) - 3 =
                       p - 2 * sin (-real.pi * x^2 / 6) * cos (real.pi / 3 * sqrt (9 - x^2))) → 
                       p ≤ -2 := 
sorry

end max_p_solution_l285_285124


namespace tan_of_cos_neg_eq_k_l285_285543

theorem tan_of_cos_neg_eq_k (k : ℝ) (h : real.cos (-80 * real.pi / 180) = k) :
  real.tan (80 * real.pi / 180) = (real.sqrt (1 - k^2)) / k :=
sorry

end tan_of_cos_neg_eq_k_l285_285543


namespace exists_c_passes_intersections_l285_285191

theorem exists_c_passes_intersections (a b : ℕ) (a_ne_b : a ≠ b) :
  ∃ c : ℕ, c = 2 * (a^2 - b^2) + a ∧ c ≠ a ∧ c ≠ b ∧
           ∀ x : ℝ, sin (a * x) = sin (b * x) → sin (c * x) = sin (a * x) :=
by sorry

end exists_c_passes_intersections_l285_285191


namespace central_polygon_area_is_25_l285_285257

-- Define the conditions
def large_square_area : ℝ := 100
def side_length (area : ℝ) : ℝ := real.sqrt area
def midpoints :=
  (side_length large_square_area) / 2
def diagonal (side_length : ℝ) : ℝ := real.sqrt (side_length^2 + side_length^2)
def smaller_square_side (diagonal : ℝ) : ℝ := diagonal / 2 / real.sqrt 2
def smaller_square_area (side : ℝ) : ℝ := side^2

-- Prove the area of the central polygon is 25 square units
theorem central_polygon_area_is_25 :
  smaller_square_area (smaller_square_side (diagonal (side_length large_square_area))) = 25 := by
  -- Here's the place to insert the proof steps.
  sorry

end central_polygon_area_is_25_l285_285257


namespace similarity_triangles_l285_285991

-- Define the triangles with their specific side lengths and conditions

theorem similarity_triangles (a : ℝ) : 
(∃ P Q R : Triangle, 
  P.sides = {a, sqrt 3 * a, 2 * a} ∧
  Q.sides = {3 * a, 2 * sqrt 3 * a, sqrt 3 * a} ∧
  R.is_right_triangle ∧
  R.has_angle 30.0) →
  (P ∼ Q ∧ Q ∼ R ∧ P ∼ R) :=
by
  -- definitions and conditions are yet to be translated to exact code
  sorry

end similarity_triangles_l285_285991


namespace find_m_l285_285375

theorem find_m (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (m^2 - 5*m + 7)*x^(m-2)) 
  (h2 : ∀ x, f (-x) = - f x) : 
  m = 3 :=
by
  sorry

end find_m_l285_285375


namespace simplify_cotAndTan_l285_285718

theorem simplify_cotAndTan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := 
by
  sorry

end simplify_cotAndTan_l285_285718


namespace birds_in_2003_l285_285334

variable (x : ℝ)

def birds_2001 := x
def birds_2002 := 1.5 * birds_2001
def birds_2003 := 2 * birds_2002

theorem birds_in_2003 : birds_2003 = 3 * x := by
  -- proof steps would go here
  sorry

end birds_in_2003_l285_285334


namespace minimum_real_roots_l285_285674

-- Define g(x) as a polynomial of degree 1004 with real coefficients
def g : polynomial ℝ := sorry

-- Assume the roots of g(x) are s_1, s_2, ..., s_1004
variables {s : ℕ → ℂ} (hs : ∀ i, root (g : polynomial ℂ) (s i))

-- Assume there are exactly 502 distinct values among |s_1|, |s_2|, ..., |s_1004|
def distinct_magnitudes : finset ℝ := (finset.range 1005).image (λ i, complex.abs (s i))

axiom h_distinct : distinct_magnitudes.card = 502

-- The theorem to prove
theorem minimum_real_roots (h_real_coeff : ∀ i, g.coeff i ∈ ℝ) :
  ∃ (r : ℕ), r = 0 := sorry

end minimum_real_roots_l285_285674


namespace radius_of_circle_l285_285941

theorem radius_of_circle (x y : ℝ) :
  4 * x^2 + 8 * x + 4 * y^2 - 16 * y + 20 = 0 → 
  (∃ xc yc : ℝ, xc = -1 ∧ yc = 2 ∧ ∀ x y : ℝ, (x - xc)^2 + (y - yc)^2 = 0) :=
by
  intro h
  use -1, 2
  split; [refl, split; [refl, intro x y]]
  sorry -- Proof to be completed

end radius_of_circle_l285_285941


namespace cone_volume_l285_285452

theorem cone_volume (r h : ℝ) (π : ℝ) (V : ℝ) :
    r = 3 → h = 4 → π = Real.pi → V = (1/3) * π * r^2 * h → V = 37.68 :=
by
  sorry

end cone_volume_l285_285452


namespace prove_problem1_prove_problem2_prove_problem3_l285_285481

noncomputable def problem1 : ℝ :=
  real.sqrt 5 * real.sqrt 15 - real.sqrt 12

noncomputable def problem2 : ℝ :=
  (real.sqrt 3 + real.sqrt 2) * (real.sqrt 3 - real.sqrt 2)

noncomputable def problem3 : ℝ :=
  (real.sqrt 20 + 5) / real.sqrt 5

theorem prove_problem1 : problem1 = 3 * real.sqrt 3 := by
  sorry

theorem prove_problem2 : problem2 = 1 := by
  sorry

theorem prove_problem3 : problem3 = 2 + real.sqrt 5 := by
  sorry

end prove_problem1_prove_problem2_prove_problem3_l285_285481


namespace oil_production_per_capita_l285_285011

-- Defining the given conditions
def oilProduction_West : ℝ := 55084
def oilProduction_NonWest : ℝ := 1480689
def population_NonWest : ℝ := 6900000
def oilProduction_Russia_part : ℝ := 13737.1
def percent_Russia_part : ℝ := 9 / 100
def population_Russia : ℝ := 147000000

-- Defining the correct answers
def perCapita_West : ℝ := oilProduction_West
def perCapita_NonWest : ℝ := oilProduction_NonWest / population_NonWest
def totalOilProduction_Russia : ℝ := oilProduction_Russia_part / percent_Russia_part
def perCapita_Russia : ℝ := totalOilProduction_Russia / population_Russia

-- Statement of the theorem
theorem oil_production_per_capita :
    (perCapita_West = 55084) ∧
    (perCapita_NonWest = 214.59) ∧
    (perCapita_Russia = 1038.33) :=
by sorry

end oil_production_per_capita_l285_285011


namespace contrapositive_necessary_condition_l285_285978

theorem contrapositive_necessary_condition (a b : Prop) (h : a → b) : ¬b → ¬a :=
by
  sorry

end contrapositive_necessary_condition_l285_285978


namespace cannot_form_right_triangle_l285_285822

theorem cannot_form_right_triangle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  a^2 + b^2 ≠ c^2 :=
by
  rw [h1, h2, h3]
  sorry

end cannot_form_right_triangle_l285_285822


namespace hyperbola_real_axis_length_l285_285368

theorem hyperbola_real_axis_length :
  let a := Real.sqrt 2 in let b := 1 in
  (2 * a = 2 * Real.sqrt 2) :=
by
  trivial
  sorry

end hyperbola_real_axis_length_l285_285368


namespace hypothesis_test_l285_285872

def X : List ℕ := [3, 4, 6, 10, 13, 17]
def Y : List ℕ := [1, 2, 5, 7, 16, 20, 22]

def alpha : ℝ := 0.01
def W_lower : ℕ := 24
def W_upper : ℕ := 60
def W1 : ℕ := 41

-- stating the null hypothesis test condition
theorem hypothesis_test : (24 < 41) ∧ (41 < 60) :=
by
  sorry

end hypothesis_test_l285_285872


namespace solve_for_a_l285_285945

noncomputable def a := 3.6

theorem solve_for_a (h : 4 * ((a * 0.48 * 2.5) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) : 
    a = 3.6 :=
by
  sorry

end solve_for_a_l285_285945


namespace jake_not_drop_coffee_percentage_l285_285649

-- Definitions for the conditions
def trip_probability : ℝ := 0.40
def drop_when_trip_probability : ℝ := 0.25

-- The question and proof statement
theorem jake_not_drop_coffee_percentage :
  100 * (1 - trip_probability * drop_when_trip_probability) = 90 :=
by
  sorry

end jake_not_drop_coffee_percentage_l285_285649


namespace infinite_positive_integers_with_triangle_l285_285330

theorem infinite_positive_integers_with_triangle : ∃ (n : ℕ), ∃ (a b c : ℕ), a + b > c ∧ a + c > b ∧ b + c > a ∧ 
  let s := (a + b + c) / 2 in
  let A := Math.sqrt (s * (s - a) * (s - b) * (s - c)) in
  let r := A / s in
  s / r = n := sorry

end infinite_positive_integers_with_triangle_l285_285330


namespace mandy_gets_15_pieces_l285_285846

def initial_pieces : ℕ := 75
def michael_takes (pieces : ℕ) : ℕ := pieces / 3
def paige_takes (pieces : ℕ) : ℕ := (pieces - michael_takes pieces) / 2
def ben_takes (pieces : ℕ) : ℕ := 2 * (pieces - michael_takes pieces - paige_takes pieces) / 5
def mandy_takes (pieces : ℕ) : ℕ := pieces - michael_takes pieces - paige_takes pieces - ben_takes pieces

theorem mandy_gets_15_pieces :
  mandy_takes initial_pieces = 15 :=
by
  sorry

end mandy_gets_15_pieces_l285_285846


namespace greatest_multiple_of_5_and_4_less_than_700_l285_285395

theorem greatest_multiple_of_5_and_4_less_than_700 : 
  ∃ n, n % 5 = 0 ∧ n % 4 = 0 ∧ n < 700 ∧ ∀ m, m % 5 = 0 ∧ m % 4 = 0 ∧ m < 700 → n ≥ m :=
  ∃ (n : ℕ), n = 680 ∧ 
    (n % 5 = 0 ∧ n % 4 = 0 ∧ n < 700 ∧ ∀ m, m % 5 = 0 ∧ m % 4 = 0 ∧ m < 700 → n ≥ m) :=
  begin
    sorry
  end

end greatest_multiple_of_5_and_4_less_than_700_l285_285395


namespace lcm_of_18_and_30_l285_285026

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end lcm_of_18_and_30_l285_285026


namespace problem_conditions_l285_285664

noncomputable def S (n : ℕ) (a : ℕ → ℚ) := ∑ i in finset.range n, a (i + 1)

theorem problem_conditions
  (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≠ 0 → ((S n a) / (a n)) = 1 + (1 / 3) * (n - 1)) :
  (∀ n : ℕ, a n = (n * (n + 1)) / 2) ∧
  (∀ n : ℕ, (∑ i in finset.range n, 1 / a (i + 1)) < 2)
  :=
by
  sorry

end problem_conditions_l285_285664


namespace div_by_seven_condition_l285_285678

theorem div_by_seven_condition {x y : ℤ} :
  (7 ∣ (2 * x + 3 * y)) ↔ (7 ∣ (5 * x + 4 * y)) :=
begin
  sorry
end

end div_by_seven_condition_l285_285678


namespace path_width_l285_285446

theorem path_width (x : ℝ)
    (length_field : ℝ) (length_field = 20)
    (width_field : ℝ) (width_field = 15)
    (area_path : ℝ) (area_path = 246)
    : (2*x^2 + 35*x - 123 = 0) → x = 3 := 
by
  intros h
  sorry

end path_width_l285_285446


namespace jake_not_drop_coffee_percentage_l285_285647

-- Definitions for the conditions
def trip_probability : ℝ := 0.40
def drop_when_trip_probability : ℝ := 0.25

-- The question and proof statement
theorem jake_not_drop_coffee_percentage :
  100 * (1 - trip_probability * drop_when_trip_probability) = 90 :=
by
  sorry

end jake_not_drop_coffee_percentage_l285_285647


namespace lcm_18_30_eq_90_l285_285016

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end lcm_18_30_eq_90_l285_285016


namespace range_of_a_l285_285222

theorem range_of_a (a : ℝ) (f g : ℝ → ℝ) 
  (Hf : ∀ x, f x = x + 4 / x)
  (Hg : ∀ x, g x = 2^x + a)
  (H : ∀ x1 ∈ set.Icc (1/2 : ℝ) 1, ∃ x2 ∈ set.Icc (2 : ℝ) 3, f x1 ≥ g x2) :
  a ≤ 1 :=
sorry

end range_of_a_l285_285222


namespace record_sale_correct_l285_285860

theorem record_sale_correct (purchase_record : ℤ) (purchase_quantity : ℤ) (sale_quantity : ℤ) :
  purchase_record = +10 → purchase_quantity = 10 → sale_quantity = 6 → -sale_quantity = -6 := by
  intros hpq hq hs
  rw [hs]
  exact rfl

end record_sale_correct_l285_285860


namespace min_students_two_in_same_class_min_students_three_in_same_class_l285_285758

-- Define the conditions
def class := Type
def student := Type
def belongs_to : student → class → Prop

-- Part 1: Proving at least two students in the same class with n = 4
theorem min_students_two_in_same_class (students : list student) (belongs_to : student → class → Prop) (c1 c2 c3 : class) 
  (h_class : ∀ s, belongs_to s c1 ∨ belongs_to s c2 ∨ belongs_to s c3) (h_distinct : ∀ s t c, belongs_to s c → belongs_to t c → s ≠ t → false) 
  : students.length ≥ 4 →
  (∃ c, ∃ s1 s2, belongs_to s1 c ∧ belongs_to s2 c ∧ s1 ≠ s2) :=
sorry

-- Part 2: Proving at least three students in the same class with n = 7
theorem min_students_three_in_same_class (students : list student) (belongs_to : student → class → Prop) (c1 c2 c3 : class) 
  (h_class : ∀ s, belongs_to s c1 ∨ belongs_to s c2 ∨ belongs_to s c3) (h_distinct : ∀ s t u c, belongs_to s c → belongs_to t c → belongs_to u c → s ≠ t → s ≠ u → t ≠ u → false) 
  : students.length ≥ 7 →
  (∃ c, ∃ s1 s2 s3, belongs_to s1 c ∧ belongs_to s2 c ∧ belongs_to s3 c ∧ s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3 ) :=
sorry

end min_students_two_in_same_class_min_students_three_in_same_class_l285_285758


namespace greatest_integer_less_than_N_div_100_l285_285975

theorem greatest_integer_less_than_N_div_100 
    (N : ℕ)
    (h : 1/(2!*17!) + 1/(3!*16!) + 1/(4!*15!) + 1/(5!*14!) + 1/(6!*13!) + 1/(7!*12!) + 1/(8!*11!) + 1/(9!*10!) = N / (1!*18!)) :
    (⌊ N / 100 ⌋ : ℕ) = 137 := 
sorry

end greatest_integer_less_than_N_div_100_l285_285975


namespace geom_arith_seq_first_term_is_two_l285_285361

theorem geom_arith_seq_first_term_is_two (b q a d : ℝ) 
  (hq : q ≠ 1) 
  (h_geom_first : b = a + d) 
  (h_geom_second : b * q = a + 3 * d) 
  (h_geom_third : b * q^2 = a + 6 * d) 
  (h_prod : b * b * q * b * q^2 = 64) :
  b = 2 :=
by
  sorry

end geom_arith_seq_first_term_is_two_l285_285361


namespace problem_proof_l285_285582
open Real

noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x, (1 / 3) * x ^ 3 + 2 * x ^ 2 + a * x + b

noncomputable def g (c d : ℝ) : ℝ → ℝ := λ x, exp x * (c * x + d)

theorem problem_proof (a b c d : ℝ)
  (h1 : ∀ a b, f a b 0 = 2)
  (h2 : ∀ c d, g c d 0 = 2)
  (h3 : ∀ a b, derivative (f a b) 0 = 4)
  (h4 : ∀ c d, derivative (g c d) 0 = 4)
  (h5 : derivative (f a b) = λ x, x^2 + 4 * x + a)
  (h6 : derivative (g c d) = λ x, exp x * (c * x + d + c)) :
  (a = 4) ∧ (b = 2) ∧ (c = 2) ∧ (d = 2) ∧ (1 ≤ m ∧ m ≤ exp 2) → (∀ m x, x ≥ -2 → m * g c d x ≥ f' x - 2) :=
by
  sorry

end problem_proof_l285_285582


namespace min_b_div_a_l285_285583

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log x + (2 * Real.exp 2 - a) * x - b / 2

theorem min_b_div_a (a b : ℝ) (h1 : ∀ x : ℝ, 0 < x → f x a b ≤ 0) : 
  ∃ (c : ℝ), c = -2 / Real.exp 2 ∧ ∀ y : ℝ, y = b / a → y ≥ c :=
begin
  sorry
end

end min_b_div_a_l285_285583


namespace Carol_max_chance_l285_285463

-- Definitions of the conditions
def Alice_random_choice (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1
def Bob_random_choice (b : ℝ) : Prop := 0.4 ≤ b ∧ b ≤ 0.6
def Carol_wins (a b c : ℝ) : Prop := (a < c ∧ c < b) ∨ (b < c ∧ c < a)

-- Statement that Carol maximizes her chances by picking 0.5
theorem Carol_max_chance : ∃ c : ℝ, (∀ a b : ℝ, Alice_random_choice a → Bob_random_choice b → Carol_wins a b c) ∧ c = 0.5 := 
sorry

end Carol_max_chance_l285_285463


namespace inequality_transformation_l285_285177

theorem inequality_transformation (x : ℝ) (n : ℕ) (hn : 0 < n) (hx : 0 < x) : 
  x + (n^n) / (x^n) ≥ n + 1 := 
sorry

end inequality_transformation_l285_285177


namespace distance_between_points_not_right_triangle_l285_285530

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem distance_between_points :
  distance 3 7 (-4) (-1) = Real.sqrt 113 :=
by
  sorry

def forms_right_triangle (x1 y1 x2 y2 : ℝ) : Prop :=
  let d1 := distance x1 y1 0 0
  let d2 := distance x2 y2 0 0
  let d3 := distance x1 y1 x2 y2
  d1^2 + d2^2 = d3^2 ∨ d2^2 + d3^2 = d1^2 ∨ d3^2 + d1^2 = d2^2

theorem not_right_triangle :
  ¬ forms_right_triangle 3 7 (-4) (-1) :=
by
  sorry

end distance_between_points_not_right_triangle_l285_285530


namespace problem_l285_285958

theorem problem (a b : ℝ) (h1 : 3^a = 2) (h2 : 9^b = 5) : 3^(a + 2 * b) = 10 := 
by
  sorry

end problem_l285_285958


namespace sequence_sum_pattern_l285_285557

theorem sequence_sum_pattern (S n : ℕ)  :
  (S = λ n, 1 * (nat.choose n 0) + 2 * (nat.choose n 1) + 3 * (nat.choose n 2) + ⋯ + n * (nat.choose n n)) →
  (n ≥ 1) →
  S n = (n + 2) * 2^(n - 1) :=
by sorry

end sequence_sum_pattern_l285_285557


namespace tan_cot_sum_15_degrees_l285_285047

theorem tan_cot_sum_15_degrees : 
  (Real.tan (Real.pi / 12) + Real.cot (Real.pi / 12) = 4) :=
by
  sorry

end tan_cot_sum_15_degrees_l285_285047


namespace truck_travel_distance_in_meters_l285_285859

theorem truck_travel_distance_in_meters
  (b : ℝ) (t : ℝ) (h1 : t ≠ 0) (h2 : 0.914 ≠ 0) :
  let distance_feet := (b / 4) * (240 / t) in
  let feet_to_meters := (0.914 / 3) in
  (distance_feet * feet_to_meters) = (18.28 * b / t) :=
by
  -- Proof is omitted
  sorry

end truck_travel_distance_in_meters_l285_285859


namespace price_per_salad_l285_285806

theorem price_per_salad :
  ∀ (price_hotdog price_salad : ℝ)
  (num_hotdogs num_salads : ℤ)
  (total_money change : ℝ),
  num_hotdogs = 5 → price_hotdog = 1.50 →
  num_salads = 3 →
  total_money = 20 →
  change = 5 →
  total_money - change = (num_hotdogs * price_hotdog).toℝ + (num_salads * price_salad).toℝ →
  price_salad = 2.50 :=
by
  intros price_hotdog price_salad num_hotdogs num_salads total_money change
  intros h1 h2 h3 h4 h5 h6
  sorry

end price_per_salad_l285_285806


namespace longest_side_of_obtuse_triangle_l285_285206

-- Conditions of the problem
def is_consecutive_integers (a b c : ℕ) : Prop :=
  (a + 1 = b) ∧ (b + 1 = c)

def is_obtuse_angle (a b c : ℕ) (angle : ℝ) : Prop :=
  angle > (Float.pi / 2)

-- Main statement of the problem
theorem longest_side_of_obtuse_triangle 
  (a b c : ℕ) (h1 : is_consecutive_integers a b c) (h2 : a ≥ 1) 
  (h3 : ∃ angle, is_obtuse_angle c b angle) :
  c = 4 :=
by
  sorry

end longest_side_of_obtuse_triangle_l285_285206


namespace garden_area_computed_l285_285825

noncomputable def pi : ℝ := Real.pi

def radius_ground : ℝ := 15
def garden_broad : ℝ := 1.2

def radius_total : ℝ := radius_ground + garden_broad

def area_circle (r : ℝ) : ℝ := pi * r ^ 2

def area_ground := area_circle radius_ground
def area_total := area_circle radius_total

def area_garden : ℝ := area_total - area_ground

theorem garden_area_computed:
  area_garden ≈ 117.45 :=
by
  sorry

end garden_area_computed_l285_285825


namespace correct_statements_l285_285864

-- Define the conditions as properties or axioms:
def each_step_reversible (alg : Type) : Prop := ∀ step, reversible step
def yields_definite_result (alg : Type) : Prop := ∀ input, ∃! result, execute alg input = result
def not_unique_algorithms (P : Type) : Prop := ∃ alg1 alg2 : P, alg1 ≠ alg2
def terminates_finite_steps (alg : Type) : Prop := ∀ input, ∃ n : ℕ, steps alg input = n

-- Prove the theorem that specifies which claims about algorithms are true:
theorem correct_statements (alg : Type) :
  (¬ each_step_reversible alg) ∧ yields_definite_result alg ∧ not_unique_algorithms alg ∧ terminates_finite_steps alg :=
sorry

end correct_statements_l285_285864


namespace teamAPointDifferenceTeamB_l285_285624

-- Definitions for players' scores and penalties
structure Player where
  name : String
  points : ℕ
  penalties : List ℕ

def TeamA : List Player := [
  { name := "Beth", points := 12, penalties := [1, 2] },
  { name := "Jan", points := 18, penalties := [1, 2, 3] },
  { name := "Mike", points := 5, penalties := [] },
  { name := "Kim", points := 7, penalties := [1, 2] },
  { name := "Chris", points := 6, penalties := [1] }
]

def TeamB : List Player := [
  { name := "Judy", points := 10, penalties := [1, 2] },
  { name := "Angel", points := 9, penalties := [1] },
  { name := "Nick", points := 12, penalties := [] },
  { name := "Steve", points := 8, penalties := [1, 2, 3] },
  { name := "Mary", points := 5, penalties := [1, 2] },
  { name := "Vera", points := 4, penalties := [1] }
]

-- Helper function to calculate total points for a player considering penalties
def Player.totalPoints (p : Player) : ℕ :=
  p.points - p.penalties.sum

-- Helper function to calculate total points for a team
def totalTeamPoints (team : List Player) : ℕ :=
  team.foldr (λ p acc => acc + p.totalPoints) 0

def teamAPoints : ℕ := totalTeamPoints TeamA
def teamBPoints : ℕ := totalTeamPoints TeamB

theorem teamAPointDifferenceTeamB :
  teamAPoints - teamBPoints = 1 :=
  sorry

end teamAPointDifferenceTeamB_l285_285624


namespace distance_from_origin_is_correct_l285_285079

noncomputable def is_distance_8_from_x_axis (x y : ℝ) := y = 8
noncomputable def is_distance_12_from_point (x y : ℝ) := (x - 1)^2 + (y - 6)^2 = 144
noncomputable def x_greater_than_1 (x : ℝ) := x > 1
noncomputable def distance_from_origin (x y : ℝ) := Real.sqrt (x^2 + y^2)

theorem distance_from_origin_is_correct (x y : ℝ)
  (h1 : is_distance_8_from_x_axis x y)
  (h2 : is_distance_12_from_point x y)
  (h3 : x_greater_than_1 x) :
  distance_from_origin x y = Real.sqrt (205 + 2 * Real.sqrt 140) :=
by
  sorry

end distance_from_origin_is_correct_l285_285079


namespace formula_for_an_harmonic_sum_lt_two_l285_285667

def a (n : ℕ) : ℕ := n * (n + 1) / 2

theorem formula_for_an (n : ℕ) (h : n > 0) : a n = n * (n + 1) / 2 := sorry

theorem harmonic_sum_lt_two (n : ℕ) (h : n > 0) : 
  (∑ k in finset.range (n + 1), (1 : ℝ) / a (k + 1)) < 2 := sorry

end formula_for_an_harmonic_sum_lt_two_l285_285667


namespace smallest_portion_proof_l285_285771

theorem smallest_portion_proof :
  ∃ (a d : ℚ), 5 * a = 100 ∧ 3 * (a + d) = 2 * d + 7 * (a - 2 * d) ∧ a - 2 * d = 5 / 3 :=
by
  sorry

end smallest_portion_proof_l285_285771


namespace problem_one_problem_two_zero_zeros_problem_two_one_zero_problem_two_two_zeros_l285_285174

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (2) (1 / x + a)

-- Problem 1
theorem problem_one (a : ℝ) : f a 1 < 2 ↔ -1 < a ∧ a < 3 := 
by sorry

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 
  f a x - log (2) ((a - 4) * x + 2 * a - 5)

-- Problem 2
theorem problem_two_zero_zeros (a : ℝ) : 
  (a ≤ 4 / 5) ↔ ∀ x, g a x ≠ 0 := 
by sorry

theorem problem_two_one_zero (a : ℝ) : 
  (4 / 5 < a ∧ a ≤ 1) ↔ ∃! x, g a x = 0 := 
by sorry

theorem problem_two_two_zeros (a : ℝ) : 
  (1 < a ∧ a ≠ 3 ∧ a ≠ 4) ↔ ∃ x₁ x₂, x₁ ≠ x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0 := 
by sorry

end problem_one_problem_two_zero_zeros_problem_two_one_zero_problem_two_two_zeros_l285_285174


namespace max_single_player_salary_l285_285456

theorem max_single_player_salary
    (num_players : ℕ) (min_salary : ℕ) (total_salary_cap : ℕ)
    (num_player_min_salary : ℕ) (max_salary : ℕ)
    (h1 : num_players = 18)
    (h2 : min_salary = 20000)
    (h3 : total_salary_cap = 600000)
    (h4 : num_player_min_salary = 17)
    (h5 : num_players = num_player_min_salary + 1)
    (h6 : total_salary_cap = num_player_min_salary * min_salary + max_salary) :
    max_salary = 260000 :=
by
  sorry

end max_single_player_salary_l285_285456


namespace yellow_curved_probability_l285_285053

variable (P_Green : ℚ) (P_Straight : ℚ)
-- Define the probabilities as variables in the context.

-- Define the conditions mentioned in the problem
axiom h1 : P_Green = 3 / 4
axiom h2 : P_Straight = 1 / 2

-- Define the problem to be proven: the probability of picking a yellow and curved flower
theorem yellow_curved_probability : 
  let P_Yellow : ℚ := 1 - P_Green in
  let P_Curved : ℚ := 1 - P_Straight in
  P_Yellow * P_Curved = 1 / 8 :=
by
  sorry

end yellow_curved_probability_l285_285053


namespace basketball_team_win_rate_l285_285428

/-- 
  A basketball team won 35 out of 50 games and has 25 games left to play.
  Prove that they need to win 13 of the remaining games to achieve a 64% win rate for the entire season.
-/
theorem basketball_team_win_rate (games_won: ℕ) (games_played: ℕ) 
  (games_left: ℕ) (required_win_rate: ℚ) :
  games_won = 35 → games_played = 50 → games_left = 25 →
  required_win_rate = 0.64 →
  let total_games := games_played + games_left in
  let required_wins := total_games * required_win_rate in
  let additional_wins := required_wins - games_won in
  additional_wins = 13 :=
by
  intros h1 h2 h3 h4
  let total_games := games_played + games_left
  let required_wins := total_games * required_win_rate
  let additional_wins := required_wins - games_won
  sorry

end basketball_team_win_rate_l285_285428


namespace cot20_plus_tan10_eq_csc20_l285_285727

theorem cot20_plus_tan10_eq_csc20 : Real.cot 20 * Real.pi / 180 + Real.tan 10 * Real.pi / 180 = Real.csc 20 * Real.pi / 180 := 
by
  sorry

end cot20_plus_tan10_eq_csc20_l285_285727


namespace difference_between_hit_and_unreleased_l285_285335

-- Define the conditions as constants
def hit_songs : Nat := 25
def top_100_songs : Nat := hit_songs + 10
def total_songs : Nat := 80

-- Define the question, conditional on the definitions above
theorem difference_between_hit_and_unreleased : 
  let unreleased_songs := total_songs - (hit_songs + top_100_songs)
  hit_songs - unreleased_songs = 5 :=
by
  sorry

end difference_between_hit_and_unreleased_l285_285335


namespace swimming_problem_l285_285077

/-- The swimming problem where a man swims downstream 30 km and upstream a certain distance 
    taking 6 hours each time. Given his speed in still water is 4 km/h, we aim to prove the 
    distance swam upstream is 18 km. -/
theorem swimming_problem 
  (V_m : ℝ) (Distance_downstream : ℝ) (Time_downstream : ℝ) (Time_upstream : ℝ) 
  (Distance_upstream : ℝ) (V_s : ℝ)
  (h1 : V_m = 4)
  (h2 : Distance_downstream = 30)
  (h3 : Time_downstream = 6)
  (h4 : Time_upstream = 6)
  (h5 : V_m + V_s = Distance_downstream / Time_downstream)
  (h6 : V_m - V_s = Distance_upstream / Time_upstream) :
  Distance_upstream = 18 := 
sorry

end swimming_problem_l285_285077


namespace largest_less_than_1_11_among_five_numbers_l285_285928

theorem largest_less_than_1_11_among_five_numbers :
  let S := {4.0, 9 / 10, 1.2, 0.5, 13 / 10}
  let T := {x ∈ S | x < 1.11}
  ∃ m ∈ T, ∀ y ∈ T, m ≥ y ∧ m = 0.9 :=
by
  let S := {4.0, 9 / 10, 1.2, 0.5, 13 / 10}
  let T := {x ∈ S | x < 1.11}
  existsi (0.9 : ℝ)
  split
  · show 0.9 ∈ T
    sorry
  · intro y hy
    split
    · show 0.9 ≥ y
      sorry
    · show 0.9 = 0.9
      rfl

end largest_less_than_1_11_among_five_numbers_l285_285928


namespace ratio_mother_to_initial_l285_285525

def initial_money : ℕ := 20
def cupcakes_price : ℕ := 150
def cupcakes_count : ℕ := 10
def cookies_price : ℕ := 300
def cookies_count : ℕ := 5
def money_left : ℕ := 3000

theorem ratio_mother_to_initial :
  let total_spent := (cupcakes_price * cupcakes_count) / 100 + (cookies_price * cookies_count) / 100 in
  let money_after_given := (money_left / 100) + total_spent in
  let money_given := money_after_given - initial_money in
  (money_given / initial_money) = 2 :=
by
  sorry

end ratio_mother_to_initial_l285_285525


namespace irrational_t3_sqrt2_l285_285322

theorem irrational_t3_sqrt2 (t : ℝ) (r : ℚ) (h : t + real.sqrt 2 = r) : ¬ ∃ s : ℚ, t^3 + real.sqrt 2 = s := 
sorry

end irrational_t3_sqrt2_l285_285322


namespace arun_and_tarun_together_complete_work_l285_285471

-- Defining the variables and conditions:
variables (W : ℝ) (R_A R_T R_AplusT : ℝ) (X : ℝ)

-- Aron's rate of work:
def rate_Arun := W / 60

-- Combined rate if Arun and Tarun work together:
def combined_rate_Arun_Tarun := W / X

-- Work done by Arun and Tarun together in 4 days:
def work_Arun_Tarun_4d := 4 * combined_rate_Arun_Tarun

-- Work done by Arun alone in 36 days:
def work_Arun_36d := 36 * rate_Arun

-- The main goal is to show that they will complete the work together in 10 days
theorem arun_and_tarun_together_complete_work : 
  (∀ (W : ℝ) (X : ℝ), rate_Arun = W / 60 →
                      combined_rate_Arun_Tarun = W / X → 
                      work_Arun_Tarun_4d + work_Arun_36d = W → 
                      X = 10) :=
by
  intros W X rate_Arun_def combined_rate_Arun_Tarun_def work_def
  have h1 : 4 * W / X + 36 * W / 60 = W,
  by rw [rate_Arun_def, combined_rate_Arun_Tarun_def] at work_def; exact work_def
  have h2 : 4 / X + 3 / 5 = 1,
  by sorry
  have h3 : X = 10,
  by sorry
  exact h3

end arun_and_tarun_together_complete_work_l285_285471


namespace problem1_problem2_l285_285509

noncomputable def f (x a : ℝ) : ℝ := abs (x + real.sqrt (1 - a)) - abs (x - real.sqrt a)

-- Problem 1
theorem problem1 (x : ℝ) : 
  (f x 0) ≥ 0 ↔ x ≥ -1/2 := sorry

-- Problem 2
theorem problem2 (b : ℝ) : 
  ((∀ a ∈ set.Icc (0:ℝ) 1, ∃ x : ℝ, f x a ≥ b) ↔ b ≤ 1) := sorry

end problem1_problem2_l285_285509


namespace orthocenter_divides_altitude_l285_285634

theorem orthocenter_divides_altitude
  (A B C : Type)
  [hA : is_acute_angle A] [hB : is_acute_angle B] [hC : is_acute_angle C]
  (angle_A : Real) (angle_B : Real) (angle_C : Real)
  (triangle_ABC : Triangle A B C)
  (orthocenter_O : is_orthocenter O triangle_ABC)
  (altitude_AD : Altitude AD A B C) :
  divides_altitude orthocenter_O altitude_AD = cos angle_A / (cos angle_B * cos angle_C) := 
  sorry

end orthocenter_divides_altitude_l285_285634


namespace simplify_cot_tan_l285_285711

theorem simplify_cot_tan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := by
  sorry

end simplify_cot_tan_l285_285711


namespace floor_T_squared_l285_285498

def T : ℝ :=
  ∑ i in Finset.range 2020, real.sqrt (1 + 1 / (i + 1)^2 + 1 / ((i + 1) + 3)^2)

theorem floor_T_squared : ⌊T^2⌋ = 4092929 := by
  sorry

end floor_T_squared_l285_285498


namespace no_integer_solution_l285_285331

theorem no_integer_solution (a b : ℤ) : ¬ (1 < a + b * (Real.sqrt 5) ∧ a + b * (Real.sqrt 5) < 9 + 4 * (Real.sqrt 5)) :=
by
  assume h : 1 < a + b * (Real.sqrt 5) ∧ a + b * (Real.sqrt 5) < 9 + 4 * (Real.sqrt 5)
  -- Proof omitted
  sorry

end no_integer_solution_l285_285331


namespace range_of_g_l285_285904

def g (x : ℝ) : ℝ := (⌊x⌋) - 2 * x

theorem range_of_g :
  Set.range g = Set.Iic 0 :=
sorry

end range_of_g_l285_285904


namespace q1_q2_l285_285219

noncomputable def f (x a : ℝ) : ℝ :=
  (x - a - 1) * Real.exp(x - 1) - 1 / 2 * x^2 + a * x

theorem q1 {a : ℝ} (h : ∀ x > 0, x ≥ a → Real.exp(x - 1) ≥ x / (x - a)) :
  a = 1 :=
sorry

theorem q2 {a : ℤ} (h : ∀ x > 0, a > 0 → a < 2 → ∀ ε > 0, ∃ δ > 0, ∀ y, |x - y| < δ → f y a > f x a) :
  a = 1 ∨ a = 2 → (1 + 2) = 3 :=
sorry

end q1_q2_l285_285219


namespace volume_proof_l285_285387

noncomputable def volume_of_tetrahedron : ℝ :=
let XY := 9 in
let XZ := 9 in
let YZ := real.sqrt 41 in
let h := 6 in
let H := 5 in
-- The area of triangle XYZ
let area_XYZ := 1 / 2 * YZ * h in
-- The volume of the tetrahedron
(1 / 3 * area_XYZ * H)

theorem volume_proof : volume_of_tetrahedron = 5 * real.sqrt 41 :=
by
  sorry

end volume_proof_l285_285387


namespace volume_of_tetrahedron_O_M_N_B₁_l285_285179

-- Definitions according to the problem conditions
def cube_edge_length : ℝ := 1
def O : EuclideanSpace ℝ (Fin 3) := (0.5, 0.5, 0)
def M : EuclideanSpace ℝ (Fin 3) := (1, 0.5, 1)
def N : EuclideanSpace ℝ (Fin 3) := (0.5, 0.5, 1)
def B₁ : EuclideanSpace ℝ (Fin 3) := (1, 0, 1)

-- Volume of tetrahedron O - M N B₁
noncomputable def volume_tetrahedron_OMNB₁ : ℝ :=
  1/6 * abs ((M - O) ⬝ ((N - O) × (B₁ - O)))

-- Statement to be proved
theorem volume_of_tetrahedron_O_M_N_B₁ :
  volume_tetrahedron_OMNB₁ = 7 / 48 :=
sorry

end volume_of_tetrahedron_O_M_N_B₁_l285_285179


namespace experimental_mean_is_correct_l285_285468

def experimental_group : List ℚ := [7.8, 9.2, 11.4, 12.4, 13.2, 15.5, 16.5, 18.0, 18.8, 19.2, 19.8, 20.2, 21.6, 22.8, 23.6, 23.9, 25.1, 28.2, 32.3, 36.5]
def control_group : List ℚ := [15.2, 18.8, 20.2, 21.3, 22.5, 23.2, 25.8, 26.5, 27.5, 30.1, 32.6, 34.3, 34.8, 35.6, 35.6, 35.8, 36.2, 37.3, 40.5, 43.2]

-- Sample mean for experimental group
def mean_experimental_group : ℚ := 19.8

-- Median of 40 mice's weight increases
def median : ℚ := 23.4

-- Contingency table
def contingency_table : ((ℕ, ℕ), (ℕ, ℕ)) := ((6, 14), (14, 6))

-- Calculated K^2 value
def K_squared : ℚ := 6.4

-- Lean theorem to prove the above assertions
theorem experimental_mean_is_correct : 
  (List.sum experimental_group) / (experimental_group.length) = mean_experimental_group ∧ 
  median = ((List.nth_le (List.sort (experimental_group ++ control_group)) 19 (by decide) + List.nth_le (List.sort (experimental_group ++ control_group)) 20 (by decide)) / 2) ∧ 
  contingency_table = ((List.countp (λ x => x < median) control_group, List.countp (λ x => x >= median) control_group), (List.countp (λ x => x < median) experimental_group, List.countp (λ x => x >= median) experimental_group)) ∧ 
  K_squared = (40 * (6 * 6 - 14 * 14)^2) / ((20 * 20 * 20 * 20)) :=
  sorry

end experimental_mean_is_correct_l285_285468


namespace quadrilateral_area_proof_l285_285443

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := {x := 0, y := 0}
def B : Point := {x := 0, y := 2}
def C : Point := {x := 3, y := 2}
def D : Point := {x := 5, y := 5}

def triangle_area (p1 p2 p3 : Point) : ℝ :=
  abs ((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2)

def quadrilateral_area (p1 p2 p3 p4 : Point) : ℝ :=
  triangle_area p1 p2 p4 + triangle_area p2 p3 p4

theorem quadrilateral_area_proof : quadrilateral_area A B C D = 9.5 := 
  by
  sorry

end quadrilateral_area_proof_l285_285443


namespace bronson_yellow_leaves_l285_285879

theorem bronson_yellow_leaves :
  let thursday_leaves := 12 in
  let friday_leaves := 13 in
  let total_leaves := thursday_leaves + friday_leaves in
  let brown_leaves := total_leaves * 20 / 100 in
  let green_leaves := total_leaves * 20 / 100 in
  let yellow_leaves := total_leaves - (brown_leaves + green_leaves) in
  yellow_leaves = 15 :=
by
  sorry

end bronson_yellow_leaves_l285_285879


namespace parity_of_T2021_T2022_T2023_l285_285457

def T : ℕ → ℤ
| 0 := 1
| 1 := 1
| 2 := 0
| n := T (n - 1) + T (n - 2) - T (n - 3)

def is_odd (n : ℤ) : Prop := ∃ (k : ℤ), n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ (k : ℤ), n = 2 * k

theorem parity_of_T2021_T2022_T2023 :
  is_odd (T 2021) ∧ is_even (T 2022) ∧ is_odd (T 2023) := 
by sorry

end parity_of_T2021_T2022_T2023_l285_285457


namespace competition_sequences_count_l285_285764

theorem competition_sequences_count :
  ∀ (A B : Fin 7 → Type),
  ∀ (predetermined_order : (Fin 7 → Type) × (Fin 7 → Type) → Prop),
  ∀ (competes_first : ∀ (a b : Type) (A B : Fin 7 → Type), Prop),
  ∀ (loser_eliminated : ∀ (winner loser : Type) (A B : Fin 7 → Type), Prop),
  ∀ (winner_advances : ∀ (winner : Type) (loser : Fin 6 → Type) (A : Fin 7 → Type) (B : Fin 7 → Type), Prop),
  ∀ (competition_continues : ∀ (A : Fin 7 → Type) (B : Fin 7 → Type), Prop),
  (choose 14 7) = 3432 :=
begin
  sorry
end

end competition_sequences_count_l285_285764


namespace false_proposition_D_l285_285207

noncomputable def xi : ℝ → ℝ := sorry -- Definition of the normal distribution random variable ξ

variable (a : ℝ)

def prop_A := P (ξ > a + 1) > P (ξ > a + 2)
def prop_B := P (ξ ≤ a) = 0.5
def prop_C := P (ξ > a + 1) = P (ξ < a - 1)
def prop_D := P (a - 1 < ξ ∧ ξ < 3 + a) < P (a < ξ ∧ ξ < 4 + a)

theorem false_proposition_D : prop_A ∧ prop_B ∧ prop_C ∧ ¬prop_D :=
by
  sorry

end false_proposition_D_l285_285207


namespace inverse_false_negation_false_l285_285989

theorem inverse_false_negation_false (p : Prop) (h : ¬ p -> False) : ¬ ¬ p := 
begin
  sorry
end

end inverse_false_negation_false_l285_285989


namespace range_of_m_l285_285215
/-
Problem: Given the function f(x) = (1/3)x^3 - x,
prove that if f has a maximum value on the interval (2m, 1-m),
then the range of the real number m is [-1, -1/2).
-/


noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x

theorem range_of_m (m : ℝ) :
  (∃ x ∈ Ioo (2*m) (1-m), f x = max (f a) (f b))
  → -1 ≤ m ∧ m < -1/2 :=
by
  sorry

end range_of_m_l285_285215


namespace nonnegative_values_ineq_l285_285129

theorem nonnegative_values_ineq {x : ℝ} : 
  (x^2 - 6*x + 9) / (9 - x^3) ≥ 0 ↔ x ∈ Set.Iic 3 := 
sorry

end nonnegative_values_ineq_l285_285129


namespace simplify_trig_identity_l285_285757

-- Defining the trigonometric functions involved
def cot (x : Real) : Real := (cos x) / (sin x)
def tan (x : Real) : Real := (sin x) / (cos x)
def csc (x : Real) : Real := 1 / (sin x)

theorem simplify_trig_identity :
  cot 20 + tan 10 = csc 20 :=
by
  sorry

end simplify_trig_identity_l285_285757


namespace matrix_cube_l285_285494

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by
  sorry

end matrix_cube_l285_285494


namespace simplify_trig_identity_l285_285751

-- Defining the trigonometric functions involved
def cot (x : Real) : Real := (cos x) / (sin x)
def tan (x : Real) : Real := (sin x) / (cos x)
def csc (x : Real) : Real := 1 / (sin x)

theorem simplify_trig_identity :
  cot 20 + tan 10 = csc 20 :=
by
  sorry

end simplify_trig_identity_l285_285751


namespace sum_of_greatest_odd_divisors_l285_285951

theorem sum_of_greatest_odd_divisors (n : ℕ) :
  (∑ i in finset.range (2 * n + 1), if (n + 1 ≤ i ∧ i ≤ 2 * n) then (i.divisors.filter (λ d, d % 2 = 1)).max' sorry else 0) = n^2 :=
by sorry

end sum_of_greatest_odd_divisors_l285_285951


namespace window_height_is_51_l285_285910

theorem window_height_is_51
  (n_panes : ℕ) 
  (rows columns : ℕ)
  (border_width pane_height_ratio pane_width_ratio : ℕ)
  (height width : ℕ)
  (h_n_panes : n_panes = 8)
  (h_rows : rows = 4)
  (h_columns : columns = 2)
  (h_pane_ratio : pane_height_ratio = 3 ∧ pane_width_ratio = 4)
  (h_border_width : border_width = 3)
  (h_height : height = (rows * pane_height_ratio * (width / pane_width_ratio)) + ((rows + 1) * border_width))
  (h_width : width = (columns * pane_width_ratio * (height / pane_height_ratio)) + ((columns + 1) * border_width)) :
  height = 51 :=
by
  -- Assuming a reasonable value for computations
  have x := 3
  have pane_height := pane_height_ratio * x
  have pane_width := pane_width_ratio * x
  have total_height := (rows * pane_height) + ((rows + 1) * border_width)
  have total_height_computed := 51
  exact h_height.symm.trans total_height_computed.symm sorry

end window_height_is_51_l285_285910


namespace exists_broken_line_of_length_2n_plus_1_l285_285659

theorem exists_broken_line_of_length_2n_plus_1 {n : ℕ} (hn : 0 < n) (points : Fin n^2 → ℝ × ℝ) :
  ∃ path : List (ℝ × ℝ), 
    (∀ p ∈ points, p ∈ path) ∧
    (length_of_broken_line path ≤ 2 * n + 1) := sorry

end exists_broken_line_of_length_2n_plus_1_l285_285659


namespace maximum_profit_l285_285837

def radioactive_marble_problem : ℕ :=
    let total_marbles := 100
    let radioactive_marbles := 1
    let non_radioactive_profit := 1
    let measurement_cost := 1
    let max_profit := 92 
    max_profit

theorem maximum_profit 
    (total_marbles : ℕ := 100) 
    (radioactive_marbles : ℕ := 1) 
    (non_radioactive_profit : ℕ := 1) 
    (measurement_cost : ℕ := 1) :
    radioactive_marble_problem = 92 :=
by sorry

end maximum_profit_l285_285837


namespace sufficient_but_not_necessary_condition_l285_285550

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3
def g (k x : ℝ) : ℝ := k * x - 1

theorem sufficient_but_not_necessary_condition (k : ℝ) :
  (∀ x : ℝ, f x ≥ g k x) ↔ (-6 ≤ k ∧ k ≤ 2) :=
sorry

end sufficient_but_not_necessary_condition_l285_285550


namespace union_complement_l285_285954

open Set Real

def P : Set ℝ := { x | x^2 - 4 * x + 3 ≤ 0 }
def Q : Set ℝ := { x | x^2 - 4 < 0 }

theorem union_complement :
  P ∪ (compl Q) = (Iic (-2)) ∪ Ici 1 :=
by
  sorry

end union_complement_l285_285954


namespace avg_cost_apple_tv_200_l285_285869

noncomputable def average_cost_apple_tv (iphones_sold ipads_sold apple_tvs_sold iphone_cost ipad_cost overall_avg_cost: ℝ) : ℝ :=
  (overall_avg_cost * (iphones_sold + ipads_sold + apple_tvs_sold) - (iphones_sold * iphone_cost + ipads_sold * ipad_cost)) / apple_tvs_sold

theorem avg_cost_apple_tv_200 :
  let iphones_sold := 100
  let ipads_sold := 20
  let apple_tvs_sold := 80
  let iphone_cost := 1000
  let ipad_cost := 900
  let overall_avg_cost := 670
  average_cost_apple_tv iphones_sold ipads_sold apple_tvs_sold iphone_cost ipad_cost overall_avg_cost = 200 :=
by
  sorry

end avg_cost_apple_tv_200_l285_285869


namespace basketball_win_requirement_l285_285426

theorem basketball_win_requirement (games_played : ℕ) (games_won : ℕ) (games_left : ℕ) (total_games : ℕ) (required_win_percentage : ℚ) : 
  games_played = 50 → games_won = 35 → games_left = 25 → total_games = games_played + games_left → required_win_percentage = 64 / 100 → 
  ∃ games_needed_to_win : ℕ, games_needed_to_win = (required_win_percentage * total_games).natCeil - games_won ∧ games_needed_to_win = 13 :=
by
  sorry

end basketball_win_requirement_l285_285426


namespace bounded_region_area_l285_285894

theorem bounded_region_area : 
  (∀ x y : ℝ, (y^2 + 4*x*y + 50*|x| = 500) → (x ≥ 0 ∧ y = 25 - 4*x) ∨ (x ≤ 0 ∧ y = -12.5 - 4*x)) →
  ∃ (A : ℝ), A = 156.25 :=
by
  sorry

end bounded_region_area_l285_285894


namespace green_area_percentage_l285_285113

theorem green_area_percentage (s : ℝ) (h1 : s > 0) 
  (h_cross : 0.44 * s^2)
  (h_circle : 0.20 * s^2) :
  (0.24 * s^2 / s^2) * 100 = 24 :=
by
  sorry

end green_area_percentage_l285_285113


namespace sum_of_c_for_8_solutions_l285_285510

-- Define the polynomial function g
def g : ℝ → ℝ := sorry -- g is an unspecified polynomial function with real coefficients

theorem sum_of_c_for_8_solutions :
  (∃ c : ℝ, (c = -2 ∨ c = -3) ∧ (∀ x : ℝ, g x = c → (∃^8 y : ℝ, g y = c))) →
  -2 + -3 = -5 :=
by 
  intro H,
  cases H with c Hc,
  cases Hc with Hc_value Hc_property,
    { rw Hc_value, 
      have : ∀ x : ℝ, g x = c → (∃^8 y : ℝ, g y = c) := Hc_property,
      norm_num },
    { rw Hc_value, 
      have : ∀ x : ℝ, g x = c → (∃^8 y : ℝ, g y = c) := Hc_property,
      norm_num }

end sum_of_c_for_8_solutions_l285_285510


namespace simplify_cotAndTan_l285_285717

theorem simplify_cotAndTan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := 
by
  sorry

end simplify_cotAndTan_l285_285717


namespace tracy_sold_paintings_l285_285801

-- Definitions based on conditions
def total_customers := 20
def customers_2_paintings := 4
def customers_1_painting := 12
def customers_4_paintings := 4

def paintings_per_customer_2 := 2
def paintings_per_customer_1 := 1
def paintings_per_customer_4 := 4

-- Theorem statement
theorem tracy_sold_paintings : 
  (customers_2_paintings * paintings_per_customer_2) + 
  (customers_1_painting * paintings_per_customer_1) + 
  (customers_4_paintings * paintings_per_customer_4) = 36 := 
by
  sorry

end tracy_sold_paintings_l285_285801


namespace p_sufficient_not_necessary_q_l285_285192

def p (x : ℝ) := 5x - 6 ≥ x^2
def q (x : ℝ) := |x + 1| > 2

theorem p_sufficient_not_necessary_q : 
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l285_285192


namespace polynomial_coefficient_B_l285_285861

theorem polynomial_coefficient_B : 
  ∃ (A C D : ℤ), 
    (∀ z : ℤ, (z > 0) → (z^6 - 15 * z^5 + A * z^4 + B * z^3 + C * z^2 + D * z + 64 = 0)) ∧ 
    (B = -244) := 
by
  sorry

end polynomial_coefficient_B_l285_285861


namespace largest_n_not_exceeding_2019_l285_285267

noncomputable def sequence (n : ℕ) : ℕ → ℤ
| 0       := 0
| (k + 1) := if sequence k < k + 1 then k + 1 else -(k + 1)

def partial_sum (n : ℕ) : ℤ :=
  (Finset.range n).sum sequence

theorem largest_n_not_exceeding_2019 (n : ℕ) :
  n ≤ 2019 ∧ partial_sum n = 0 → n = 1092 :=
sorry

end largest_n_not_exceeding_2019_l285_285267


namespace basketball_team_win_rate_l285_285429

/-- 
  A basketball team won 35 out of 50 games and has 25 games left to play.
  Prove that they need to win 13 of the remaining games to achieve a 64% win rate for the entire season.
-/
theorem basketball_team_win_rate (games_won: ℕ) (games_played: ℕ) 
  (games_left: ℕ) (required_win_rate: ℚ) :
  games_won = 35 → games_played = 50 → games_left = 25 →
  required_win_rate = 0.64 →
  let total_games := games_played + games_left in
  let required_wins := total_games * required_win_rate in
  let additional_wins := required_wins - games_won in
  additional_wins = 13 :=
by
  intros h1 h2 h3 h4
  let total_games := games_played + games_left
  let required_wins := total_games * required_win_rate
  let additional_wins := required_wins - games_won
  sorry

end basketball_team_win_rate_l285_285429


namespace perp_cond_l285_285979

-- Definitions of lines and perpendicularity in the context of the problem
variable (α : Type) [plane : planar α]
variable (a b c : line α)

-- Conditions: b and c are in plane α
axiom b_in_plane : b ∈ α
axiom c_in_plane : c ∈ α

-- Condition: a is a line that could be perpendicular to the α
variable (a_perp_alpha : a ⊥ α)

-- Problem: "a ⊥ α" is a sufficient condition for "a ⊥ b" and "a ⊥ c", but not necessary.
theorem perp_cond (h1: a_perp_alpha) (h2: b ≠ c) (h3: intersect b c α) :
  (a ⊥ b ∧ a ⊥ c) ↔ (a ⊥ α) := by
  sorry

end perp_cond_l285_285979


namespace pairs_sum_gcd_l285_285142

theorem pairs_sum_gcd (a b : ℕ) (h_sum : a + b = 288) (h_gcd : Int.gcd a b = 36) :
  (a = 36 ∧ b = 252) ∨ (a = 252 ∧ b = 36) ∨ (a = 108 ∧ b = 180) ∨ (a = 180 ∧ b = 108) :=
by {
   sorry
}

end pairs_sum_gcd_l285_285142


namespace curve_representation_l285_285775

theorem curve_representation (x y : ℝ) (h : x ≥ 1) :
  (x + y - 1) * (real.sqrt (x - 1)) = 0 ↔ (x = 1) ∨ (x + y - 1 = 0 ∧ x ≥ 1) := 
by sorry

end curve_representation_l285_285775


namespace binomial_expansion_terms_l285_285572

theorem binomial_expansion_terms (x : ℝ) :
  let n := 8;
  let term_containing_x_to_power_1 := ((8.choose 4) * 2^(-4)) * x;
  let rational_terms := [x^4, ((8.choose 4) * 2^(-4)) * x, (8.choose 8) * 2^(-8) / x^2];
  let largest_coeff_terms := [(8.choose 3) * 2^(-3) * x^(5/2), (8.choose 4) * 2^(-4) * x^(7/4)];
  term_containing_x_to_power_1 = 35 / 8 * x ∧
  rational_terms = [x^4, 35 / 8 * x, 1 / 256 / x^2] ∧
  largest_coeff_terms = [7 * x^(5/2), 7 * x^(7/4)] :=
by
  sorry

end binomial_expansion_terms_l285_285572


namespace AB_eq_length_l285_285701

variable (A B C O : Type)
variable [Circle A B C O]
variable (r : ℝ)

noncomputable def length_of_major_arc (a b r : ℝ) := 2 * r * Real.sin ((b - a) / 2)
noncomputable def length_of_minor_arc (a b r : ℝ) := 2 * r * Real.sin ((b - a) / 2)

axiom A_eq_C (A C : Point) : A = C
axiom A_B_gt_r (A B : Point) : dist A B > r
axiom minor_arc_BC_len (B C : Point) (r : ℝ) : length_of_minor_arc B C r = π * r / 3

def AB_length (r : ℝ) : ℝ := r * Real.sqrt (2 + Real.sqrt 3)

theorem AB_eq_length (A B C : Point) (r : ℝ)
  (h1 : A_eq_C A C)
  (h2 : A_B_gt_r A B)
  (h3 : minor_arc_BC_len B C r) :
  dist A B = AB_length r :=
sorry

end AB_eq_length_l285_285701


namespace score_of_58_is_2_stdevs_below_l285_285163

theorem score_of_58_is_2_stdevs_below :
  ∃ σ : ℝ, (98 = 74 + 3 * σ) ∧ (58 = 74 - 2 * σ) :=
by
  use 8
  split
  · exact by linarith
  · exact by linarith
  sorry

end score_of_58_is_2_stdevs_below_l285_285163


namespace cubic_sequence_exists_l285_285423

-- Statement in Lean 4:
theorem cubic_sequence_exists (b c d : ℤ) :
  ∃ (a : ℕ → ℤ), 
  (∀ n, a n = n^3 + b * n^2 + c * n + d) ∧ 
  ((∃ k : ℤ, a 2015 = k * k) ∧
   (∃ k : ℤ, a 2016 = k * k) ∧ 
   (∀ n, (a n).natAbs = n^3 + b * n^2 + c * n + d → (n ≠ 2015 ∧ n ≠ 2016 → ¬(∃ k : ℤ, a n = k * k)))) ∧
  a 2015 * a 2016 = 0 :=
begin
  sorry
end

end cubic_sequence_exists_l285_285423


namespace negation_of_p_l285_285591

-- Define the function f from reals to reals
variable (f : ℝ → ℝ)

-- The proposition p
def p := ∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0

-- Negation of the proposition p
def neg_p := ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0

-- The theorem to be proved
theorem negation_of_p : ¬ p ↔ neg_p := 
sorry

end negation_of_p_l285_285591


namespace amount_spent_on_giftwrapping_and_expenses_l285_285610

theorem amount_spent_on_giftwrapping_and_expenses (total_spent : ℝ) (cost_of_gifts : ℝ) (h_total_spent : total_spent = 700) (h_cost_of_gifts : cost_of_gifts = 561) : 
  total_spent - cost_of_gifts = 139 :=
by
  rw [h_total_spent, h_cost_of_gifts]
  norm_num

end amount_spent_on_giftwrapping_and_expenses_l285_285610


namespace sum_of_fractions_to_decimal_l285_285883

theorem sum_of_fractions_to_decimal :
  ((2 / 40 : ℚ) + (4 / 80) + (6 / 120) + (9 / 180) : ℚ) = 0.2 :=
by
  sorry

end sum_of_fractions_to_decimal_l285_285883


namespace tetrahedron_volume_bound_l285_285319

-- Definitions for conditions
variable {A B C D : Type}
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]

def edge_lengths (A : Type) (B : Type) (C : Type) (D : Type) (dist : A → B → ℝ) :=
  (dist A B > 1) ∧ (dist A C ≤ 1) ∧ (dist A D ≤ 1) ∧ (dist B C ≤ 1) ∧ (dist B D ≤ 1) ∧ (dist C D ≤ 1)

-- Hypotheses
theorem tetrahedron_volume_bound {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
[has_volume A] [has_volume B] [has_volume C] [has_volume D] (dist : A → B → ℝ) 
(h : edge_lengths A B C D dist) :
  volume (tetrahedron.mk A B C D) ≤ (1 / 8) :=
by
  sorry

end tetrahedron_volume_bound_l285_285319


namespace eggs_per_basket_l285_285656

theorem eggs_per_basket (n : ℕ) (total_eggs_red total_eggs_orange min_eggs_per_basket : ℕ) (h_red : total_eggs_red = 20) (h_orange : total_eggs_orange = 30) (h_min : min_eggs_per_basket = 5) (h_div_red : total_eggs_red % n = 0) (h_div_orange : total_eggs_orange % n = 0) (h_at_least : n ≥ min_eggs_per_basket) : n = 5 :=
sorry

end eggs_per_basket_l285_285656


namespace factorial_division_l285_285500

open Nat

theorem factorial_division : 12! / 11! = 12 := sorry

end factorial_division_l285_285500


namespace replace_asterisk_l285_285819

theorem replace_asterisk :
  ∃ x : ℤ, (x / 21) * (63 / 189) = 1 ∧ x = 63 := sorry

end replace_asterisk_l285_285819


namespace no_fixed_points_l285_285967

def f (x a : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem no_fixed_points (a : ℝ) :
  (∀ x : ℝ, f x a ≠ x) ↔ (-1/2 < a ∧ a < 3/2) := by
    sorry

end no_fixed_points_l285_285967


namespace inscribe_parallelepiped_in_white_sphere_l285_285060

-- Define the problem conditions
variable (S : ℝ) (area_red : ℝ)

-- Assume 12% of the surface area of the white sphere is painted red.
axiom surface_area_red_painted : area_red = 0.12 * S

-- Define the sphere and the requirement of the problem
def sphere (r : ℝ) := { x : EuclideanSpace ℝ (Fin 3) // ∥x∥ = r }

-- Proof Statement
theorem inscribe_parallelepiped_in_white_sphere :
  ∃ (p : parallelepiped (sphere r)), 
    (∀ vertice ∈ p.vertices, vertice ∉ area_red) :=
sorry

end inscribe_parallelepiped_in_white_sphere_l285_285060


namespace geometric_sequence_S5_eq_11_l285_285673

theorem geometric_sequence_S5_eq_11 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (q : ℤ)
  (h1 : a 1 = 1)
  (h4 : a 4 = -8)
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_S : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  S 5 = 11 := 
by
  -- Proof omitted
  sorry

end geometric_sequence_S5_eq_11_l285_285673


namespace pure_imaginary_condition_l285_285209

-- Define the problem
theorem pure_imaginary_condition (θ : ℝ) :
  (∀ k : ℤ, θ = (3 * Real.pi / 4) + k * Real.pi) →
  ∀ z : ℂ, z = (Complex.cos θ - Complex.sin θ * Complex.I) * (1 + Complex.I) →
  ∃ k : ℤ, θ = (3 * Real.pi / 4) + k * Real.pi → 
  (Complex.re z = 0 ∧ Complex.im z ≠ 0) :=
  sorry

end pure_imaginary_condition_l285_285209


namespace number_order_l285_285566

theorem number_order (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : (1 / a > real.sqrt a ∧ real.sqrt a > a ∧ a > a^2) :=
by
  sorry

end number_order_l285_285566


namespace centroid_max_l285_285702

variable (A B C A1 B1 C1 M : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited A1] [Inhabited B1] [Inhabited C1] [Inhabited M]

-- Assume points A1, B1, C1 are on sides BC, CA, AB of some triangle ABC
variable (onSide_BC : A1 ∈ line BC)
variable (onSide_CA : B1 ∈ line CA)
variable (onSide_AB : C1 ∈ line AB)

-- Assume AA1, BB1, CC1 intersect at M
variable (intersectionM : (line AA1) ∩ (line BB1) ∩ (line CC1) = {M})

-- Define α, β, γ as the ratios
variables (α β γ : ℝ)

-- α = MA1 / AA1, β = MB1 / BB1, γ = MC1 / CC1
axiom ratio_α : α = distance M A1 / distance A A1
axiom ratio_β : β = distance M B1 / distance B B1
axiom ratio_γ : γ = distance M C1 / distance C C1

-- The sum of these ratios must satisfy α + β + γ = 1
axiom ratio_sum : α + β + γ = 1

-- Now, we state the theorem:
theorem centroid_max {
  let_product := α * β * γ } :
  product = 1 / 27 :=
sorry

end centroid_max_l285_285702


namespace cot20_tan10_eq_csc20_l285_285748

theorem cot20_tan10_eq_csc20 : 
  Real.cot 20 * (Real.pi / 180) + Real.tan 10 * (Real.pi / 180) = Real.csc 20 * (Real.pi / 180) :=
by sorry

end cot20_tan10_eq_csc20_l285_285748


namespace inequality_always_holds_l285_285563

variable {a b : ℝ}

theorem inequality_always_holds (ha : a > 0) (hb : b < 0) : 1 / a > 1 / b :=
by
  sorry

end inequality_always_holds_l285_285563


namespace contracting_schemes_count_l285_285350

noncomputable def combination (n k : ℕ) : ℕ :=
nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem contracting_schemes_count (n a b c : ℕ) (h : a + b + c = n) :
  combination n a * combination (n - a) b * combination (n - a - b) c = 60 :=
by
  have h1 : combination 6 3 = 20 := by
    sorry
  have h2 : combination 3 2 = 3 := by
    sorry
  have h3 : combination 1 1 = 1 := by
    sorry
  calc
    combination 6 3 * combination (6 - 3) 2 * combination (6 - 3 - 2) 1
        = 20 * 3 * 1                           : by rw [h1, h2, h3]
    ... = 60                                   : by norm_num

end contracting_schemes_count_l285_285350


namespace find_a_l285_285232

noncomputable def U (a : ℝ) : Set ℝ := {2, 4, a^2 - a + 1}
noncomputable def A (a : ℝ) : Set ℝ := {a + 4, 4}
noncomputable def complement_U_A (a : ℝ) : Set ℝ := {7}

theorem find_a (a : ℝ) (hU : U a = {2, 4, a^2 - a + 1})
  (hA : A a = {a + 4, 4})
  (hcompA : complement_U_A a = {7}) :
  a = -2 :=
sorry

end find_a_l285_285232


namespace min_A_div_B_l285_285980

theorem min_A_div_B (x A B : ℝ) (hx_pos : 0 < x) (hA_pos : 0 < A) (hB_pos : 0 < B) 
  (h1 : x^2 + 1 / x^2 = A) (h2 : x - 1 / x = B + 3) : 
  (A / B) = 6 + 2 * Real.sqrt 11 :=
sorry

end min_A_div_B_l285_285980


namespace relationship_f_l285_285961

   def f (x : ℝ) : ℝ := 2 * x + 1
   def a : ℝ := Real.log 0.7 / Real.log 2  -- ln(0.7) / ln(2) is equal to log_2(0.7)
   def b : ℝ := Real.exp (0.2 * (Real.log 3))  -- exp(0.2 * ln(3)) is equal to 3^0.2
   def c : ℝ := Real.exp (1.3 * (Real.log 0.2))  -- exp(1.3 * ln(0.2)) is equal to 0.2^1.3

   theorem relationship_f (h1 : f a = 2 * a + 1) (h2 : f b = 2 * b + 1) (h3 : f c = 2 * c + 1) :
     f a < f c ∧ f c < f b := sorry
   
end relationship_f_l285_285961


namespace calculate_expression_l285_285479

theorem calculate_expression : 
  (-6)^6 / 6^4 - 2^5 + 9^2 = 85 := 
by sorry

end calculate_expression_l285_285479


namespace find_a_for_two_distinct_roots_l285_285901

-- Defining the problem context
def has_distinct_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 + |x1| = 4 * real.sqrt (a * (x1 - 3) + 2) ∧ x2 + |x2| = 4 * real.sqrt (a * (x2 - 3) + 2))

-- Statement of the problem
theorem find_a_for_two_distinct_roots : ∀ a : ℝ, has_distinct_roots a ↔ (a < 1 ∨ a > 2) :=
sorry

end find_a_for_two_distinct_roots_l285_285901


namespace four_digit_prime_and_multiple_of_3_count_l285_285661

/--
Let P equal the number of four-digit prime numbers.
Let M equal the number of four-digit multiples of 3.
Prove that P + M = 4061.
-/
theorem four_digit_prime_and_multiple_of_3_count : 
    let P := 1061 in 
    let M := 3000 in 
    P + M = 4061 := 
by
  -- introduce constants
  let P := 1061
  let M := 3000
  -- prove the statement
  show P + M = 4061  
  sorry

end four_digit_prime_and_multiple_of_3_count_l285_285661


namespace simplify_cotAndTan_l285_285720

theorem simplify_cotAndTan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := 
by
  sorry

end simplify_cotAndTan_l285_285720


namespace multiply_polynomials_l285_285309

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l285_285309


namespace factorial_division_l285_285508

-- Definition of factorial in Lean
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Statement of the problem:
theorem factorial_division : fact 12 / fact 11 = 12 :=
by
  sorry

end factorial_division_l285_285508


namespace probability_kevin_takes_A_bus_l285_285472

noncomputable def A_bus_arrival_time : ℝ := 20
noncomputable def B_bus_arrival_time : ℝ := 18
noncomputable def kevin_waiting_time : ℝ := 5

theorem probability_kevin_takes_A_bus :
  let total_sample_space := A_bus_arrival_time * B_bus_arrival_time
      rect_area := kevin_waiting_time * B_bus_arrival_time
      tri_base := A_bus_arrival_time - kevin_waiting_time
      tri_height := B_bus_arrival_time - kevin_waiting_time
      tri_area := 0.5 * tri_base * tri_height
      desired_area := rect_area + tri_area
  in (desired_area / total_sample_space) = 349 / 720 :=
sorry

end probability_kevin_takes_A_bus_l285_285472


namespace number_of_ways_to_pick_two_cards_with_at_least_one_joker_l285_285436

theorem number_of_ways_to_pick_two_cards_with_at_least_one_joker :
  let deck_size := 54
  let jokers := 2
  let standard_cards := 52
  ∃ (number_of_ways : Nat), number_of_ways = 210 :=
by
  let deck_size := 54
  let jokers := 2
  let standard_cards := 52
  let first_case := 1 * 53
  let second_case := 1 * 53
  let third_case := 52 * 2
  let total_number_of_ways := first_case + second_case + third_case
  exists 210
  have correct_answer : total_number_of_ways = 210 := by sorry
  exact correct_answer

#eval number_of_ways_to_pick_two_cards_with_at_least_one_joker

end number_of_ways_to_pick_two_cards_with_at_least_one_joker_l285_285436


namespace tracy_sold_paintings_l285_285800

-- Definitions based on conditions
def total_customers := 20
def customers_2_paintings := 4
def customers_1_painting := 12
def customers_4_paintings := 4

def paintings_per_customer_2 := 2
def paintings_per_customer_1 := 1
def paintings_per_customer_4 := 4

-- Theorem statement
theorem tracy_sold_paintings : 
  (customers_2_paintings * paintings_per_customer_2) + 
  (customers_1_painting * paintings_per_customer_1) + 
  (customers_4_paintings * paintings_per_customer_4) = 36 := 
by
  sorry

end tracy_sold_paintings_l285_285800


namespace ab_div_10_eq_0_l285_285518

def double_factorial_even (n : ℕ) : ℕ := 2^n * n!
def double_factorial_odd (n : ℕ) : ℕ := (2*n + 1) * double_factorial_odd (n - 1)
def sum_S : ℚ := ∑ i in finset.range 2010, (2 * i)!! / ((2 * i + 1)!!)

theorem ab_div_10_eq_0 : (∃ a b : ℕ, (∀ b, b%2 = 1) → sum_S = 2^a * b → (a * b) / 10 = 0) :=
sorry

end ab_div_10_eq_0_l285_285518


namespace sum_a_n_up_to_1000_l285_285164

def a_n (n : ℕ) : ℕ :=
  if n % 182 = 0 then 10
  else if n % 154 = 0 then 12
  else if n % 143 = 0 then 15
  else 0

theorem sum_a_n_up_to_1000 : (∑ n in Finset.range 1000, a_n (n + 1)) = 227 :=
by
  sorry

end sum_a_n_up_to_1000_l285_285164


namespace least_number_to_add_l285_285416

theorem least_number_to_add (n d : ℕ) (h₁ : n = 1054) (h₂ : d = 23) : ∃ x, (n + x) % d = 0 ∧ x = 4 := by
  sorry

end least_number_to_add_l285_285416


namespace probability_of_divisibility_by_11_among_5_digit_numbers_sum_40_l285_285609

-- Define the conditions
def is_five_digit (n : Nat) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digits_sum_to_40 (n : Nat) : Prop :=
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  digits.sum = 40

def divisible_by_11 (n : Nat) : Prop := n % 11 = 0

-- Define the main theorem
theorem probability_of_divisibility_by_11_among_5_digit_numbers_sum_40 :
  (∑ k in Finset.image (fun p => p.1)
             ({'⟨99994, digits_sum_to_40 99994⟩,'⟨99985, digits_sum_to_40 99985⟩ 
               : {p // p.1 ∈ ({99994, 99985} : Finset Nat)∧ p.2}})
             (λ n, if divisible_by_11 n then 1 else 0) : ℕ) / 25 = 
  (2 : ℚ) / 25 :=
sorry

end probability_of_divisibility_by_11_among_5_digit_numbers_sum_40_l285_285609


namespace problem_conditions_l285_285663

noncomputable def S (n : ℕ) (a : ℕ → ℚ) := ∑ i in finset.range n, a (i + 1)

theorem problem_conditions
  (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≠ 0 → ((S n a) / (a n)) = 1 + (1 / 3) * (n - 1)) :
  (∀ n : ℕ, a n = (n * (n + 1)) / 2) ∧
  (∀ n : ℕ, (∑ i in finset.range n, 1 / a (i + 1)) < 2)
  :=
by
  sorry

end problem_conditions_l285_285663


namespace tree_survival_difference_l285_285096

theorem tree_survival_difference 
  (initial_trees : Nat)
  (died_trees : Nat)
  (survived_trees : Nat)
  : initial_trees = 11 → died_trees = 2 → survived_trees = initial_trees - died_trees → survived_trees - died_trees = 7 :=
by
  intros h_initial h_died h_survived
  rw [h_initial, h_died, h_survived]
  sorry

end tree_survival_difference_l285_285096


namespace derivative_at_zero_l285_285244

-- Define the function f, incorporating the condition
def f (x : ℝ) : ℝ := x^2 + 2 * x * f' 1

-- State the theorem to prove
theorem derivative_at_zero (f' : ℝ → ℝ) (h : ∀ x, f' x = 2 * x + 2 * f' 1) : f' 0 = -4 :=
by
  -- We skip the proof details here
  sorry

end derivative_at_zero_l285_285244


namespace cot20_plus_tan10_eq_csc20_l285_285726

theorem cot20_plus_tan10_eq_csc20 : Real.cot 20 * Real.pi / 180 + Real.tan 10 * Real.pi / 180 = Real.csc 20 * Real.pi / 180 := 
by
  sorry

end cot20_plus_tan10_eq_csc20_l285_285726


namespace fraction_unchanged_l285_285526

-- Define the digit rotation
def rotate (d : ℕ) : ℕ :=
  match d with
  | 0 => 0
  | 1 => 1
  | 6 => 9
  | 8 => 8
  | 9 => 6
  | _ => d  -- for completeness, though we assume d only takes {0, 1, 6, 8, 9}

-- Define the condition for a fraction to be unchanged when flipped
def unchanged_when_flipped (numerator denominator : ℕ) : Prop :=
  let rotated_numerator := rotate numerator
  let rotated_denominator := rotate denominator
  rotated_numerator * denominator = rotated_denominator * numerator

-- Define the specific fraction 6/9
def specific_fraction_6_9 : Prop :=
  unchanged_when_flipped 6 9 ∧ 6 < 9

-- Theorem stating 6/9 is unchanged when its digits are flipped and it's a valid fraction
theorem fraction_unchanged : specific_fraction_6_9 :=
by
  sorry

end fraction_unchanged_l285_285526


namespace min_value_expression_l285_285173

variable {a b : ℝ}

theorem min_value_expression : 
  a > b → b > 1 → 2 * log a b + 3 * log b a = 7 → a + 1 / (b^2 - 1) ≥ 3 :=
by
  intro ha hb hlog
  sorry

end min_value_expression_l285_285173


namespace comparison_of_functions_l285_285586

noncomputable theory

open Real

def f (a x : ℝ) : ℝ := a * log x + 1 / x

def g (a x : ℝ) : ℝ := f a x - 1 / x

theorem comparison_of_functions (a x m n : ℝ) (h₁ : 0 < m) (h₂ : m < n) (h₃ : x = 1) (h₄ : f a 1 = 0) :
  a = 1 → (g a n - g a m) / 2 > (n - m) / (n + m) :=
begin
  intro ha,
  simp [f, g, ha],
  sorry
end

end comparison_of_functions_l285_285586


namespace jake_not_drop_coffee_percentage_l285_285648

-- Definitions for the conditions
def trip_probability : ℝ := 0.40
def drop_when_trip_probability : ℝ := 0.25

-- The question and proof statement
theorem jake_not_drop_coffee_percentage :
  100 * (1 - trip_probability * drop_when_trip_probability) = 90 :=
by
  sorry

end jake_not_drop_coffee_percentage_l285_285648


namespace circle_area_eqn_l285_285393

theorem circle_area_eqn (x y : ℝ) (h : x^2 + y^2 + 8 * x + 18 * y = -72) : 
    let r := 5 in let pi := Real.pi in pi * r^2 = 25 * pi :=
    by
      have eq : (x + 4)^2 + (y + 9)^2 = 25 := 
        by sorry -- This would be the step to show completion of the square
      let r := 5
      let area := pi * r^2
      exact area = 25 * pi

end circle_area_eqn_l285_285393


namespace find_length_of_train_l285_285405

-- Definitions based on the conditions
def speed_km_per_hr : ℕ := 90
def time_sec : ℕ := 5

-- Conversion from km/hr to m/s
def speed_m_per_s : ℕ := (speed_km_per_hr * 1000) / 3600

-- Define the length of the train
def length_of_train (speed : ℕ) (time : ℕ) : ℕ :=
  speed * time

-- Problem statement
theorem find_length_of_train 
  (speed_km_per_hr : ℕ) 
  (time_sec : ℕ) 
  (speed_m_per_s : ℕ) 
  (length_of_train : ℕ) :
  speed_km_per_hr = 90 →
  time_sec = 5 →
  speed_m_per_s = (90 * 1000) / 3600 →
  length_of_train = (speed_m_per_s * time_sec) →
  length_of_train = 125 :=
by {
  intros,
  calc length_of_train = (90 * 1000 / 3600) * 5 : by rw [a_3]
                      ... = 125                 : by norm_num
}

end find_length_of_train_l285_285405


namespace sin_exp_eq_200_l285_285150

theorem sin_exp_eq_200 (x : ℝ) :
  (∃ x ∈ (0, 200 * real.pi), sin x = (1 / 3) ^ x) ↔ 200 := sorry

end sin_exp_eq_200_l285_285150


namespace smallest_number_of_marbles_l285_285255

theorem smallest_number_of_marbles 
  (r w b g n : ℕ) 
  (h_condition1 : r + w + b + g = n)
  (h_condition2 : ∀ (k : ℕ), 
     binomial_coefficient r 2 * binomial_coefficient b 2 = binomial_coefficient w 2 * binomial_coefficient g 2 ∧ 
     binomial_coefficient r 2 * binomial_coefficient b 2 = r * w * b * g) 
  : n = 10 :=
begin
  sorry
end

end smallest_number_of_marbles_l285_285255


namespace probability_exactly_3400_l285_285316

def spinnerAmounts := ["Bankrupt", "$2000", "$600", "$3500", "$800"]
def nonBankruptcyAmounts := ["$2000", "$600", "$3500", "$800"]

theorem probability_exactly_3400 :
  let outcomesPerSpin := 4
  let totalPossibleOutcomes := outcomesPerSpin ^ 3
  let favorableOutcomes := 6
  let probability := favorableOutcomes.toRat / totalPossibleOutcomes.toRat
  probability = 3 / 32 :=
begin
  sorry
end

end probability_exactly_3400_l285_285316


namespace anna_goal_l285_285102

/-
Anna wants to average 5 km per day, and March has 31 days. By the night of the 16th of March,
Anna has walked 95 km so far. We want to prove that she needs to walk an average of 4 km per day for 
the remaining days in March to achieve her goal.
-/

theorem anna_goal
  (avg_daily_target : ℝ)
  (days_in_march : ℕ)
  (walked_so_far : ℝ)
  (current_day : ℕ)
  (remaining_days : ℕ)
  (remaining_distance : ℝ)
  (avg_needed_per_day : ℝ) :
  avg_daily_target = 5 →
  days_in_march = 31 →
  walked_so_far = 95 →
  current_day = 16 →
  remaining_days = days_in_march - current_day →
  remaining_distance = (days_in_march * avg_daily_target) - walked_so_far →
  avg_needed_per_day = remaining_distance / remaining_days →
  avg_needed_per_day = 4 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  rw [h1, h2, h3, h4, h5, h6, h7],
  sorry
end

end anna_goal_l285_285102


namespace smallest_four_digit_multiple_of_6_with_sum_12_l285_285397

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def sum_of_digits_eq (n : ℕ) (m : ℕ) : Prop :=
  let digits := n.to_string.data.map (λ c, c.to_nat - '0'.to_nat)
  digits.sum = m

def is_divisible_by (n : ℕ) (k : ℕ) : Prop :=
  n % k = 0

theorem smallest_four_digit_multiple_of_6_with_sum_12 : ∃ (n : ℕ), 
  is_four_digit n ∧ 
  is_divisible_by n 6 ∧ 
  sum_of_digits_eq n 12 ∧ 
  ∀ m : ℕ, 
    is_four_digit m ∧ 
    is_divisible_by m 6 ∧ 
    sum_of_digits_eq m 12 → 
    n ≤ m := 
begin
  -- Proof to be provided
  sorry
end

end smallest_four_digit_multiple_of_6_with_sum_12_l285_285397


namespace sum_even_coefficients_l285_285542

theorem sum_even_coefficients (n : ℕ) :
  let a := λ i, (finset.range (2 * n + 1)).sum (λ k, if k < i then 0 else if i = 0 then 1 else 0)
  in (finset.range (n + 1)).sum (λ m, a (2 * m)) = (3 ^ n - 1) / 2 :=
sorry

end sum_even_coefficients_l285_285542


namespace graph_passes_through_point_l285_285364

theorem graph_passes_through_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ (x y : ℝ), x = 2 ∧ y = 2 ∧ y = a^(x-2) + 1 :=
by
  use 2
  use 2
  split
  · refl
  split
  · refl
  · have h₂ : a^0 = 1 := by sorry
    rw [h₂]

end graph_passes_through_point_l285_285364


namespace student_failed_by_40_marks_l285_285088

theorem student_failed_by_40_marks (total_marks : ℕ) (passing_percentage : ℝ) (marks_obtained : ℕ) (h1 : total_marks = 500) (h2 : passing_percentage = 33) (h3 : marks_obtained = 125) :
  ((passing_percentage / 100) * total_marks - marks_obtained : ℝ) = 40 :=
sorry

end student_failed_by_40_marks_l285_285088


namespace right_triangles_proof_l285_285445

axiom rectangle_points (E F G H I J K L : Point) : 
  divides_into_four_congruent_rectangles EF GH IJ KL ∧
  parallel (segment IJ) (segment EF) ∧
  parallel (segment KL) (segment EH)

noncomputable def number_of_right_triangles : ℕ :=
  20

theorem right_triangles_proof :
  ∀ (EF GH IJ KL : Segment) (E F G H I J K L : Point),
  divides_into_four_congruent_rectangles EF GH IJ KL ∧
  parallel (segment IJ) (segment EF) ∧
  parallel (segment KL) (segment EH) →
  right_triangles_using_points {E, F, G, H, I, J, K, L} = 20 :=
by
  sorry

end right_triangles_proof_l285_285445


namespace sqrt_two_over_two_not_covered_l285_285514

theorem sqrt_two_over_two_not_covered 
  (a b : ℕ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≠ 0) (h4 : Nat.gcd a b = 1) :
  ¬ (real.sqrt 2 / 2 ∈ set.Icc (a / b - 1 / (4 * b ^ 2) : ℝ) (a / b + 1 / (4 * b ^ 2) : ℝ)) :=
sorry

end sqrt_two_over_two_not_covered_l285_285514


namespace perimeter_of_rectangle_l285_285373

theorem perimeter_of_rectangle (L W : ℝ) (h1 : L / W = 5 / 2) (h2 : L * W = 4000) : 2 * L + 2 * W = 280 :=
sorry

end perimeter_of_rectangle_l285_285373


namespace rug_inner_length_is_4_l285_285454

-- Define the conditions
def areas_form_arithmetic_progression (a b c : ℕ) : Prop :=
  b - a = c - b

def rug_conditions (inner_length middle_length outer_length : ℕ) : Prop :=
  let inner_area := 2 * inner_length in
  let middle_area := 6 * (inner_length + 4) in
  let outer_area := 10 * (inner_length + 8) in
  areas_form_arithmetic_progression inner_area middle_area outer_area

-- Main statement to prove
theorem rug_inner_length_is_4 : ∃ (x : ℕ), rug_conditions x (x + 4) (x + 8) ∧ x = 4 :=
sorry

end rug_inner_length_is_4_l285_285454


namespace vec_problem_l285_285486

def vec1 : ℤ × ℤ := (3, -5)
def vec2 : ℤ × ℤ := (2, -6)
def scalar_mult (a : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (a * v.1, a * v.2)
def vec_sub (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 - v2.1, v1.2 - v2.2)
def result := (6, -2)

theorem vec_problem :
  vec_sub (scalar_mult 4 vec1) (scalar_mult 3 vec2) = result := 
by 
  sorry

end vec_problem_l285_285486


namespace gary_earnings_l285_285957

section
  open Real

  -- Define the constants and initial conditions.
  def total_flour_pounds : ℝ := 6
  def flour_for_cakes : ℝ := 4
  def flour_per_cake : ℝ := 0.5
  def remaining_flour_for_cupcakes : ℝ := 2
  def flour_per_cupcake : ℝ := 1 / 5
  def price_per_cake : ℝ := 2.5
  def price_per_cupcake : ℝ := 1

  -- Calculate the number of cakes and cupcakes.
  def number_of_cakes : ℝ := flour_for_cakes / flour_per_cake
  def number_of_cupcakes : ℝ := remaining_flour_for_cupcakes * 5  -- (remaining_flour_for_cupcakes / flour_per_cupcake)

  -- Calculate earnings from cakes and cupcakes.
  def earnings_from_cakes : ℝ := number_of_cakes * price_per_cake
  def earnings_from_cupcakes : ℝ := number_of_cupcakes * price_per_cupcake

  -- Total earnings.
  def total_earnings : ℝ := earnings_from_cakes + earnings_from_cupcakes

  -- The theorem to prove.
  theorem gary_earnings : total_earnings = 30 := by
    sorry
end

end gary_earnings_l285_285957


namespace solve_for_x_l285_285544

theorem solve_for_x (x y : ℕ) (h1 : x / y = 15 / 5) (h2 : y = 25) : x = 75 := 
by
  sorry

end solve_for_x_l285_285544


namespace find_circumference_l285_285057

theorem find_circumference
  (C : ℕ)
  (h1 : ∃ (vA vB : ℕ), C > 0 ∧ vA > 0 ∧ vB > 0 ∧ 
                        (120 * (C/2 + 80)) = ((C - 80) * (C/2 - 120)) ∧
                        (C - 240) / vA = (C + 240) / vB) :
  C = 520 := 
  sorry

end find_circumference_l285_285057


namespace factorial_division_l285_285507

-- Definition of factorial in Lean
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Statement of the problem:
theorem factorial_division : fact 12 / fact 11 = 12 :=
by
  sorry

end factorial_division_l285_285507


namespace initial_kittens_l285_285796

-- Define the number of kittens given to Jessica and Sara, and the number of kittens currently Tim has.
def kittens_given_to_Jessica : ℕ := 3
def kittens_given_to_Sara : ℕ := 6
def kittens_left_with_Tim : ℕ := 9

-- Define the theorem to prove the initial number of kittens Tim had.
theorem initial_kittens (kittens_given_to_Jessica kittens_given_to_Sara kittens_left_with_Tim : ℕ) 
    (h1 : kittens_given_to_Jessica = 3)
    (h2 : kittens_given_to_Sara = 6)
    (h3 : kittens_left_with_Tim = 9) :
    (kittens_given_to_Jessica + kittens_given_to_Sara + kittens_left_with_Tim) = 18 := 
    sorry

end initial_kittens_l285_285796


namespace range_h_l285_285532

open set

noncomputable def h (t : ℝ) : ℝ := (t^2 + (1/2) * t) / (t^2 + 1)

theorem range_h : range h = {1 / 2} := by
  -- Proof is omitted
  sorry

end range_h_l285_285532


namespace find_NC_l285_285568

-- Define the properties and conditions based on the problem
noncomputable def isosceles_right_triangle : Type := {
  A B C : ℝ
  h₁ : B ≠ A
  h₂ : C ≠ A
  h₃ : A ≠ C
  h₄ : AB = 6 * sqrt 2
  h₅ : AC = 6 * sqrt 2
  right_angle : ∠ BAC = 90
}

noncomputable def points_on_hypotenuse (A B C M N : ℝ) : Prop :=
  M ∈ segment B C ∧ N ∈ segment B C

noncomputable def given_conditions (A B C M N : ℝ) : Prop :=
  isosceles_right_triangle A B C ∧ points_on_hypotenuse A B C M N ∧ BM = 3 ∧ ∠ MAN = 45

theorem find_NC (A B C M N : ℝ) (h : given_conditions A B C M N) : NC = 4 := by
  sorry

end find_NC_l285_285568


namespace area_of_enclosed_region_l285_285363

open Real

noncomputable def integral_area : ℝ :=
  2 * (∫ (θ : ℝ) in (π / 4)..(3 * π / 4), 4 + 4 * cos (2 * θ) + (cos (2 * θ)) ^ 2) sorry

theorem area_of_enclosed_region :
  integral_area = (9 * π / 2) - 8 := sorry

end area_of_enclosed_region_l285_285363


namespace ratio_of_sweater_vests_to_shirts_l285_285483

theorem ratio_of_sweater_vests_to_shirts (S V O : ℕ) (h1 : S = 3) (h2 : O = 18) (h3 : O = V * S) : (V : ℚ) / (S : ℚ) = 2 := 
  by
  sorry

end ratio_of_sweater_vests_to_shirts_l285_285483


namespace lcm_18_30_is_90_l285_285030

noncomputable def LCM_of_18_and_30 : ℕ := 90

theorem lcm_18_30_is_90 : nat.lcm 18 30 = LCM_of_18_and_30 := 
by 
  -- definition of lcm should be used here eventually
  sorry

end lcm_18_30_is_90_l285_285030


namespace planes_parallel_from_plane_l285_285820

-- Define the relationship functions
def parallel (P Q : Plane) : Prop := sorry -- Define parallelism predicate
def perpendicular (l : Line) (P : Plane) : Prop := sorry -- Define perpendicularity predicate

-- Declare the planes α, β, and γ
variable (α β γ : Plane)

-- Main theorem statement
theorem planes_parallel_from_plane (h1 : parallel γ α) (h2 : parallel γ β) : parallel α β := 
sorry

end planes_parallel_from_plane_l285_285820


namespace area_of_BEIH_l285_285824

structure Point where
  x : ℚ
  y : ℚ

def B := { x := 0, y := 0 } : Point
def A := { x := 0, y := 3 } : Point
def C := { x := 3, y := 0 } : Point
def D := { x := 3, y := 3 } : Point
def E := { x := 0, y := 1.5 } : Point
def F := { x := 1.5, y := 0 } : Point
def I := { x := 3/7, y := 9/7 } : Point
def H := { x := 3/5, y := 3/5 } : Point

def area_BEIH (B E I H : Point) : ℚ :=
  (1 / 2) * abs (B.x * (E.y + I.y) + E.x * (I.y + H.y) + I.x * (H.y + B.y) + H.x * (B.y + E.y)
               - (B.y * (E.x + I.x) + E.y * (I.x + H.x) + I.y * (H.x + B.x) + H.y * (B.x + E.x)))

theorem area_of_BEIH :
  area_BEIH B E I H = 9 / 35 := by
  sorry

end area_of_BEIH_l285_285824


namespace expand_product_l285_285917

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 :=
by
  sorry

end expand_product_l285_285917


namespace sqrt2_over_2_not_covered_by_rationals_l285_285512

noncomputable def rational_not_cover_sqrt2_over_2 : Prop :=
  ∀ (a b : ℤ) (h_ab : Int.gcd a b = 1) (h_b_pos : b > 0)
  (h_frac : (a : ℚ) / b ∈ Set.Ioo 0 1),
  abs ((Real.sqrt 2) / 2 - (a : ℚ) / b) > 1 / (4 * b^2)

-- Placeholder for the proof
theorem sqrt2_over_2_not_covered_by_rationals :
  rational_not_cover_sqrt2_over_2 := 
by sorry

end sqrt2_over_2_not_covered_by_rationals_l285_285512


namespace max_p_value_l285_285126

theorem max_p_value : ∃ x : ℝ,  let LHS := 2 * (cos (2*π - (π * x^2) / 6) * cos ((π / 3) * sqrt (9 - x^2))) - 3,
                                      RHS := p - 2 * (sin (-(π * x^2) / 6) * cos ((π / 3) * sqrt (9 - x^2)))
                                in LHS = RHS → p ≤ -2 :=
begin
  sorry
end

end max_p_value_l285_285126


namespace find_n_value_l285_285632

theorem find_n_value : 
  ∃ (n : ℕ), ∀ (a b c : ℕ), 
    a + b + c = 200 ∧ 
    (∃ bc ca ab : ℕ, bc = b * c ∧ ca = c * a ∧ ab = a * b ∧ n = bc ∧ n = ca ∧ n = ab) → 
    n = 199 := sorry

end find_n_value_l285_285632


namespace initial_potatoes_count_l285_285515

-- Given conditions:
variables (initial_potatoes : ℕ) (eaten_potatoes : ℕ) (remaining_potatoes : ℕ)

-- Assigning the specific values given in the conditions:
axiom initial_condition : eaten_potatoes = 4
axiom after_eating : remaining_potatoes = 3

-- Define theorem to prove that initial number of potatoes is 7:
theorem initial_potatoes_count :
  (initial_potatoes = remaining_potatoes + eaten_potatoes) → initial_potatoes = 7 :=
by
  intros h
  subst initial_condition
  subst after_eating
  rw [h]
  norm_num
  sorry

end initial_potatoes_count_l285_285515


namespace probability_sum_of_five_l285_285707

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 4

theorem probability_sum_of_five :
  favorable_outcomes / total_outcomes = 1 / 9 := 
by
  sorry

end probability_sum_of_five_l285_285707


namespace sum_b_n_2019_l285_285770

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else (fibonacci (n - 1)) + (fibonacci (n - 2))

def b_n (n : ℕ) : ℤ :=
  (fibonacci n : ℤ) * (fibonacci (n + 2) : ℤ) - (fibonacci (n + 1) : ℤ)^2

theorem sum_b_n_2019 :
  ∑ n in Finset.range 2020, b_n n = 1 :=
by
  sorry

end sum_b_n_2019_l285_285770


namespace sin_neg_1920_eq_neg_sqrt3_div_2_l285_285816

open Real

theorem sin_neg_1920_eq_neg_sqrt3_div_2 : sin (-1920 * pi / 180) = - (sqrt 3 / 2) :=
by
  -- Proof omitted, focuses on the statement
  sorry

end sin_neg_1920_eq_neg_sqrt3_div_2_l285_285816


namespace matrix_cube_l285_285493

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by
  sorry

end matrix_cube_l285_285493


namespace vec_problem_l285_285485

def vec1 : ℤ × ℤ := (3, -5)
def vec2 : ℤ × ℤ := (2, -6)
def scalar_mult (a : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (a * v.1, a * v.2)
def vec_sub (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 - v2.1, v1.2 - v2.2)
def result := (6, -2)

theorem vec_problem :
  vec_sub (scalar_mult 4 vec1) (scalar_mult 3 vec2) = result := 
by 
  sorry

end vec_problem_l285_285485


namespace cheese_stick_problem_l285_285280

theorem cheese_stick_problem : 
  ∀ (P : ℕ), 
  let cheddar := 15 in
  let mozzarella := 30 in
  let total := cheddar + mozzarella + P in
  (P : ℝ) / total = 0.5 →
  P = 45 :=
by
  intros P cheddar mozzarella total h
  sorry

end cheese_stick_problem_l285_285280


namespace deduce_toppings_l285_285325

-- Define the topping set as a finset of natural numbers from 1 to 5
def Toppings : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the pizzas as a list of finsets (subsets of toppings)
def Pizzas := List (Finset ℕ)

-- Define the condition that each pizza contains exactly one topping
def valid_pizzas (pizzas : Pizzas) : Prop :=
  ∀ pizza ∈ pizzas, ∃ t ∈ Toppings, pizza = {t}

-- The main theorem to prove
theorem deduce_toppings (pizzas : Pizzas) (h : pizzas.length = 5) (hv : valid_pizzas pizzas):
  ∀ t ∈ Toppings, ∃ pizza ∈ pizzas, pizza = {t} :=
by
  sorry

end deduce_toppings_l285_285325


namespace candidate_votes_percentage_l285_285847

-- Conditions
variables {P : ℝ} 
variables (totalVotes : ℝ := 8000)
variables (differenceVotes : ℝ := 2400)

-- Proof Problem
theorem candidate_votes_percentage (h : ((P / 100) * totalVotes + ((P / 100) * totalVotes + differenceVotes) = totalVotes)) : P = 35 :=
by
  sorry

end candidate_votes_percentage_l285_285847


namespace dress_price_l285_285006

namespace VanessaClothes

def priceOfDress (total_revenue : ℕ) (num_dresses num_shirts price_of_shirt : ℕ) : ℕ :=
  (total_revenue - num_shirts * price_of_shirt) / num_dresses

theorem dress_price :
  priceOfDress 69 7 4 5 = 7 :=
by 
  calc
    priceOfDress 69 7 4 5 = (69 - 4 * 5) / 7 : rfl
                     ... = 49 / 7 : by norm_num
                     ... = 7 : by norm_num

end VanessaClothes

end dress_price_l285_285006


namespace complement_intersection_is_correct_l285_285594

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {1, 2}
noncomputable def B : Set ℕ := {0, 2, 5}
noncomputable def complementA := (U \ A)

theorem complement_intersection_is_correct :
  complementA ∩ B = {0, 5} :=
by
  sorry

end complement_intersection_is_correct_l285_285594


namespace triangle_quadrilateral_area_ratio_l285_285275

noncomputable def area_triang {A B C : ℝ} (AB AC BC AD AE : ℝ) (conditions : Prop) : ℝ := sorry

theorem triangle_quadrilateral_area_ratio
  (A B C : Type) [AddCommGroup A] [Module ℝ A] (AB BC AC AD AE : ℝ)
  (h1 : AB = 24) (h2 : BC = 40) (h3 : AC = 50)
  (h4 : AD = 10) (h5 : AE = 20)
  (hF : ∀ (F : A), F = (A +ₗ B) / 2):
  area_triang AB AC BC AD AE (h1 ∧ h2 ∧ h3 ∧ h4 ∧ h5)
  = 25 / 119 := 
sorry

end triangle_quadrilateral_area_ratio_l285_285275


namespace remainder_r15_minus_1_l285_285156

theorem remainder_r15_minus_1 (r : ℝ) : 
    (r^15 - 1) % (r - 1) = 0 :=
sorry

end remainder_r15_minus_1_l285_285156


namespace sequence_term_2023_l285_285619

theorem sequence_term_2023 (a : ℕ → ℚ) (h₁ : a 1 = 2) 
  (h₂ : ∀ n, 1 / a n - 1 / a (n + 1) - 1 / (a n * a (n + 1)) = 1) : 
  a 2023 = -1 / 2 := 
sorry

end sequence_term_2023_l285_285619


namespace max_min_dot_product_and_norm_l285_285201

noncomputable def vector_a : ℝ × ℝ := (1, 0)
noncomputable def vector_b : ℝ × ℝ := (-1, real.sqrt 3)
noncomputable def μ : ℝ := 4 / 7
noncomputable def vector_c : ℝ × ℝ := (2 - 2 * μ, real.sqrt 3 * μ)
noncomputable def vector_c_norm : ℝ := real.sqrt ((2 - 2 * μ)^2 + (real.sqrt 3 * μ)^2)

theorem max_min_dot_product_and_norm :
  (∃ μ : ℝ, 
    μ = (4 / 7) ∧ 
    ∀ λ μ : ℝ, λ + μ = 2 → 
    (|vector_c| = vector_c_norm) ∧ 
    (min (vector_a.1 * vector_c.1 + vector_a.2 * vector_c.2) (vector_b.1 * vector_c.1 + vector_b.2 * vector_c.2) = (2 - 2 * μ))) :=
begin
  use (4 / 7),
  split,
  { norm_num, },
  { intros λ μ h,
    split,
    { sorry }, -- Proof that |vector_c| = vector_c_norm
    { sorry }, -- Proof that min(dot products) = 2 - 2 * μ
  }
end

end max_min_dot_product_and_norm_l285_285201


namespace total_sales_amount_l285_285785

/-
Problem: Prove that the total sales amount of qualified products among 8 bags of laundry detergent is 60.2 yuan given the following conditions.
- Each bag of laundry detergent has a weight measured in grams.
- The standard weight is 448 grams.
- Excess or deficient weight is calculated as the bag's weight minus the standard weight.
- A product cannot be sold if the absolute value of its excess or deficient weight is greater than 4 grams.
- Price of each bag is 8.6 yuan.
- The measured weights are 444, 447, 448, 450, 451, 454, 455, and 449 grams.
-/

theorem total_sales_amount :
  let weights := [444, 447, 448, 450, 451, 454, 455, 449]
  let standard_weight := 448
  let exces_deficient_weight w := w - standard_weight
  let price := 8.6
  let qualified := λ w, abs (exces_deficient_weight w) ≤ 4
  let total_sales_amount := (weights.filter qualified).length * price
  total_sales_amount = 60.2 := sorry

end total_sales_amount_l285_285785


namespace sum_of_a_values_l285_285374

theorem sum_of_a_values (a b c : ℕ) (h : ∀ a ≤ 100, ∃ b c, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 1 / (a ^ 2 : ℝ) + 1 / (b ^ 2 : ℝ) = 1 / (c ^ 2 : ℝ)) :
  (∑ a in (finset.filter (λ a, ∃ b c, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 1 / (a ^ 2 : ℝ) + 1 / (b ^ 2 : ℝ) = 1 / (c ^ 2 : ℝ)) (finset.Icc 1 100)), a) = 620 :=
by
  sorry

end sum_of_a_values_l285_285374


namespace cannot_form_3x3_square_l285_285823

def square_pieces (squares : ℕ) (rectangles : ℕ) (triangles : ℕ) := 
  squares = 4 ∧ rectangles = 1 ∧ triangles = 1

def area (squares : ℕ) (rectangles : ℕ) (triangles : ℕ) : ℕ := 
  squares * 1 * 1 + rectangles * 2 * 1 + triangles * (1 * 1 / 2)

theorem cannot_form_3x3_square : 
  ∀ squares rectangles triangles, 
  square_pieces squares rectangles triangles → 
  area squares rectangles triangles < 9 := by
  intros squares rectangles triangles h
  unfold square_pieces at h
  unfold area
  sorry

end cannot_form_3x3_square_l285_285823


namespace molly_vs_xanthia_time_difference_henry_vs_molly_time_difference_l285_285046

-- Definitions of reading speeds and book length
def reading_speed_xanthia : ℕ := 120  -- pages per hour
def reading_speed_molly : ℕ := 40     -- pages per hour
def reading_speed_henry : ℕ := 60     -- pages per hour
def book_length : ℕ := 300            -- pages

-- Calculating reading times in hours
def time_xanthia : ℝ := (book_length : ℝ) / reading_speed_xanthia
def time_molly : ℝ := (book_length : ℝ) / reading_speed_molly
def time_henry : ℝ := (book_length : ℝ) / reading_speed_henry

-- Calculating time differences in hours
def molly_xanthia_diff_hours : ℝ := time_molly - time_xanthia
def molly_henry_diff_hours : ℝ := time_molly - time_henry

-- Converting time differences to minutes
def molly_xanthia_diff_minutes : ℝ := molly_xanthia_diff_hours * 60
def molly_henry_diff_minutes : ℝ := molly_henry_diff_hours * 60

-- Theorem we want to prove
theorem molly_vs_xanthia_time_difference :
  molly_xanthia_diff_minutes = 300 := by
  sorry

theorem henry_vs_molly_time_difference :
  molly_henry_diff_minutes = 150 := by
  sorry

end molly_vs_xanthia_time_difference_henry_vs_molly_time_difference_l285_285046


namespace badgers_win_at_least_five_games_prob_l285_285766

noncomputable def probability_Badgers_win_at_least_five_games : ℚ :=
  let p : ℚ := 1 / 2
  let n : ℕ := 9
  (1 / 2)^n * ∑ k in finset.range (n + 1), if k >= 5 then (nat.choose n k : ℚ) else 0

theorem badgers_win_at_least_five_games_prob :
  probability_Badgers_win_at_least_five_games = 1 / 2 :=
by sorry

end badgers_win_at_least_five_games_prob_l285_285766


namespace simplify_cotAndTan_l285_285721

theorem simplify_cotAndTan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := 
by
  sorry

end simplify_cotAndTan_l285_285721


namespace max_integer_k_l285_285241

theorem max_integer_k (a b c : ℝ) (h : a > b ∧ b > c) : 
  ∃ k : ℕ, (∀ a b c : ℝ, a > b ∧ b > c → (1 / (a - b) + 1 / (b - c) ≥ k / (a - c))) ∧ 
           (∀ n : ℕ, (∀ a b c : ℝ, a > b ∧ b > c → (1 / (a - b) + 1 / (b - c) ≥ n / (a - c))) → n ≤ k) :=
begin
  use 4,
  sorry
end

end max_integer_k_l285_285241


namespace cost_of_first_batch_eq_30_price_for_profit_margin_eq_50_l285_285798

noncomputable def cost_per_shirt_in_first_batch : ℕ := sorry
noncomputable def price_per_shirt_for_50_percent_profit_margin : ℕ := sorry

theorem cost_of_first_batch_eq_30 :
  (3000 / x = 100) →
  (6600 / (x + 3) = 200) →
  (2 × 3000 / x = 6600 / (x + 3)) →
  x = 30 :=
by
  intros h1 h2 h3
  have h4 : 3000 / x = 100 := by sorry
  have h5 : 6600 / (x + 3) = 200 := by sorry
  have h6 : 2 * (3000 / x) = (6600 / (x + 3)) := by sorry
  -- Proof
  sorry

theorem price_for_profit_margin_eq_50 :
  (270y + 30 * 0.6y - 9600 = 0.5 * 9600) →
  y = 50 :=
by
  intros h1
  have h2 : (270y + 18y - 9600 = 4800) := by sorry
  have h3 : (288y = 14400) := by sorry
  -- Proof
  sorry

end cost_of_first_batch_eq_30_price_for_profit_margin_eq_50_l285_285798


namespace sum_nonzero_l285_285392

-- Definitions based on the given conditions
def isOdd (n : ℕ) : Prop := n % 2 = 1
def productOfRow (board : ℕ → ℕ → ℤ) (k n : ℕ) : ℤ := ∏ j in Finset.range n, board k j
def productOfColumn (board : ℕ → ℕ → ℤ) (k n : ℕ) : ℤ := ∏ i in Finset.range n, board i k

-- Main theorem statement
theorem sum_nonzero (n : ℕ) (board : ℕ → ℕ → ℤ)
  (hn : isOdd n)
  (hboard : ∀ i j, board i j = 1 ∨ board i j = -1) :
  let a := λ k, productOfRow board k n,
      b := λ k, productOfColumn board k n in
  (∑ k in Finset.range n, a k) + (∑ k in Finset.range n, b k) ≠ 0 :=
by
  sorry

end sum_nonzero_l285_285392


namespace distance_to_x_axis_l285_285776

def point_P : Point := (-4, 1)

theorem distance_to_x_axis (P : Point) (hx : P.2 = 1) : distance_to_x_axis P = 1 :=
by {
  sorry
}

end distance_to_x_axis_l285_285776


namespace sequence_properties_and_inequality_l285_285227

theorem sequence_properties_and_inequality :
  ( ∃ {a : ℕ → ℚ}, 
      (a 1 = 1/2) ∧ 
      (∀ n, a (n + 1) = a n - (a n)^2) ∧
      (a 2 = 1/4) ∧ 
      (a 3 = 3/16) ∧
      (∀ n, a n ≤ 2 * a (n + 1))
  ) := 
sorry

end sequence_properties_and_inequality_l285_285227


namespace sum_of_divisors_360_l285_285398

theorem sum_of_divisors_360 : ∑ d in Finset.filter (λ x => x ∣ 360) (Finset.range (360 + 1)) = 1170 := by
  sorry

end sum_of_divisors_360_l285_285398


namespace perp_GH_FC_l285_285160

noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry
noncomputable def E : Point := sorry
noncomputable def F : Point := sorry
noncomputable def G : Point := circumcenter A D F
noncomputable def H : Point := circumcenter B E F

-- Given conditions
axiom distinct_points : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ A
axiom collinear_points : collinear {A, B, C, D, E}
axiom equal_segments : dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E
axiom point_outside_line : ¬collinear {A, F, D}

-- Proof obligation
theorem perp_GH_FC : ∀ (A B C D E F G H : Point),
  distinct_points ∧ collinear_points ∧ equal_segments ∧ point_outside_line ->
  is_perpendicular (line_through G H) (line_through F C) := by
  sorry

end perp_GH_FC_l285_285160


namespace find_a_2023_l285_285621

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, (1 / a n - 1 / a (n + 1) - 1 / (a n * a (n + 1)) = 1)

theorem find_a_2023 (a : ℕ → ℚ) (h : sequence a) : a 2023 = -1 / 2 :=
  sorry

end find_a_2023_l285_285621


namespace longest_diagonal_l285_285449

-- Define the given conditions of the rhombus
def area (d1 d2 : ℝ) : ℝ := (1 / 2) * d1 * d2
def ratio (d1 d2 : ℝ) : ℝ := d1 / d2

-- The main theorem we want to prove
theorem longest_diagonal (d1 d2 : ℝ) 
  (h_area : area d1 d2 = 150) 
  (h_ratio : ratio d1 d2 = 4 / 3) : d1 = 20 := 
sorry

end longest_diagonal_l285_285449


namespace problem_statement_l285_285001

-- Definitions
def triangle (A B C : Type) := 
  ∃ (side : ℝ), side = 4 ∧ dist A B = side ∧ dist B C = side ∧ dist C A = side

def midpoint (M B C : Type) :=
  dist B M = 2 ∧ dist M C = 2

def points_on_sides (N P C A B : Type) :=
  N ∈ segment C A ∧ P ∈ segment A B ∧ dist A N > dist A P

def cyclic_quadrilateral (A N M P : Type) :=
  ∃ (circumcircle : Type), point_on_circle N circumcircle ∧ point_on_circle M circumcircle ∧ 
  point_on_circle P circumcircle ∧ point_on_circle A circumcircle

def triangle_area (N M P : Type) (area : ℝ) :=
  area = 3

-- Statement
theorem problem_statement (A B C M N P : Type) 
  [triangle A B C]
  [midpoint M B C]
  [points_on_sides N P C A B]
  [cyclic_quadrilateral A N M P]
  [triangle_area N M P 3] :
  ∃ (a b c : ℕ), dist C N = (a - real.sqrt b) / c ∧ a + b + c = 6 := 
sorry

end problem_statement_l285_285001


namespace sum_of_squares_of_coefficients_l285_285332

def polynomial1 : Polynomial ℚ := 2 * (Polynomial.X ^ 3) - 3 * (Polynomial.X ^ 2) + 4
def polynomial2 : Polynomial ℚ := Polynomial.X ^ 4 - 2 * (Polynomial.X ^ 3) + 3 * Polynomial.X - 2

theorem sum_of_squares_of_coefficients :
  let simplified_poly := 5 * polynomial1 - 6 * polynomial2
  let coeffs := simplified_poly.coeffs
  (coeffs.map (λ c, c * c)).sum = 2093 := by
  sorry

end sum_of_squares_of_coefficients_l285_285332


namespace int_pairs_satisfy_conditions_l285_285122

theorem int_pairs_satisfy_conditions (m n : ℤ) :
  (∃ a b : ℤ, m^2 + n = a^2 ∧ n^2 + m = b^2) ↔ 
  ∃ k : ℤ, (m = 0 ∧ n = k^2) ∨ (m = k^2 ∧ n = 0) ∨ (m = 1 ∧ n = -1) ∨ (m = -1 ∧ n = 1) := by
  sorry

end int_pairs_satisfy_conditions_l285_285122


namespace gcd_bezout_663_182_l285_285933

theorem gcd_bezout_663_182 :
  let a := 182
  let b := 663
  ∃ d u v : ℤ, d = Int.gcd a b ∧ d = a * u + b * v ∧ d = 13 ∧ u = 11 ∧ v = -3 :=
by 
  let a := 182
  let b := 663
  use 13, 11, -3
  sorry

end gcd_bezout_663_182_l285_285933


namespace perpendiculars_intersect_at_one_point_l285_285560

-- Definitions based on given conditions:
variable (A B C D O H1 H2 H3 H4 : Type)
variable [square : is_square A B C D]
variable [point_O_inside_square : is_point_inside_square O A B C D]
variable [perpendiculars : ∃ (H1 H2 H3 H4 : Type), 
  is_perpendicular A H1 (line_through B O) ∧ 
  is_perpendicular B H2 (line_through C O) ∧ 
  is_perpendicular C H3 (line_through D O) ∧ 
  is_perpendicular D H4 (line_through A O)]

-- The theorem stating the math proof problem
theorem perpendiculars_intersect_at_one_point :
  ∃ P : Type, intersects_at_one_point [perpendicular (A H1 (line_through B O)), 
  perpendicular (B H2 (line_through C O)), 
  perpendicular (C H3 (line_through D O)), 
  perpendicular (D H4 (line_through A O))] P :=
sorry

end perpendiculars_intersect_at_one_point_l285_285560


namespace construct_tangent_intersects_segment_l285_285807

-- Given definitions
variables {O : Point} {A B T : Point} {R : ℝ} {l : Line} -- Points and radius
axiom circle (O : Point) (R : ℝ) : Circle
axiom line_segment (A B : Point) : LineSegment
axiom tangent_line (T : Point) (l : Line) (c : Circle) : Prop -- Line tangent to a circle at point T

-- Given conditions
def center_point : Point := O
def radius : ℝ := R
def given_circle : Circle := circle O R
def given_segment : LineSegment := line_segment A B

-- Problem statement: Construct a tangent to the given circle that intercepts the given line segment AB
theorem construct_tangent_intersects_segment :
  ∃ T l, is_right_triangle O T A ∧ tangent_line T l given_circle ∧ intercept_segment l A B := 
sorry

end construct_tangent_intersects_segment_l285_285807


namespace sequence_contains_multiple_of_p_l285_285083

noncomputable def sequence (a : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := (sequence n * (sequence n + 1)) / 2 - 10

theorem sequence_contains_multiple_of_p (a_1 : ℕ) (h : 5 < a_1) (p : ℕ) [fact (nat.prime p)] :
  (∃ n : ℕ, sequence a_1 n % p = 0) ↔ p = 2 :=
by sorry

#check sequence_contains_multiple_of_p

end sequence_contains_multiple_of_p_l285_285083


namespace walkway_time_l285_285826

theorem walkway_time {v_p v_w : ℝ} 
  (cond1 : 60 = (v_p + v_w) * 30) 
  (cond2 : 60 = (v_p - v_w) * 120) 
  : 60 / v_p = 48 := 
by
  sorry

end walkway_time_l285_285826


namespace perimeter_sum_of_triangles_l285_285694

/-- Sum of perimeters of a large right triangle with legs 6 and 8 units and a smaller triangle
    with half the area of the large right triangle is equal to 24 + 6 * sqrt 3 + 2 * sqrt 15. -/
theorem perimeter_sum_of_triangles :
  let large_leg1 := 6
  let large_leg2 := 8
  let large_hypotenuse := Real.sqrt (large_leg1^2 + large_leg2^2)
  -- Large triangle properties
  let large_perimeter := large_leg1 + large_leg2 + large_hypotenuse
  let large_area := 1/2 * large_leg1 * large_leg2
  -- Small triangle properties
  let small_area := large_area / 2
  (∃ (small_leg1 small_leg2 small_hypotenuse : ℝ), (1/2) * small_leg1 * small_leg2 = small_area 
    ∧ small_hypotenuse = Real.sqrt (small_leg1^2 + small_leg2^2) 
    ∧ let small_perimeter := small_leg1 + small_leg2 + small_hypotenuse 
    -- Sum of perimeters
    in large_perimeter + small_perimeter = 24 + 6 * Real.sqrt 3 + 2 * Real.sqrt 15) := 
begin
  sorry
end

end perimeter_sum_of_triangles_l285_285694


namespace spinner_div_by_4_prob_l285_285857

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def num_from_spins (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem spinner_div_by_4_prob :
  let outcomes := [(a, b, c) | a in [1, 2, 3, 4], b in [1, 2, 3, 4], c in [1, 2, 3, 4]],
      valid_outcomes := [(a, b, c) | (a, b, c) ∈ outcomes, is_divisible_by_4 (num_from_spins a b c)]
  in (valid_outcomes.length : ℚ) / (outcomes.length : ℚ) = 1 / 2 :=
by
  sorry

end spinner_div_by_4_prob_l285_285857


namespace sequence_ratio_l285_285225

noncomputable def seq (n : ℕ) : ℝ :=
if n = 0 then 5 else 
nat.rec_on n 5 (λ n an, 2^n / an)

theorem sequence_ratio :
  (seq 6 / seq 2) = 4 :=
sorry

end sequence_ratio_l285_285225


namespace largest_three_digit_divisible_by_13_l285_285128

theorem largest_three_digit_divisible_by_13 :
  ∃ n, (n ≤ 999 ∧ n ≥ 100 ∧ 13 ∣ n) ∧ (∀ m, m ≤ 999 ∧ m ≥ 100 ∧ 13 ∣ m → m ≤ 987) :=
by
  sorry

end largest_three_digit_divisible_by_13_l285_285128


namespace infinite_chain_on_parabola_l285_285365

theorem infinite_chain_on_parabola :
  ∀ (x y : ℝ), 
  let n := sqrt (x^2 + y^2),
      m := y in
  (n - m = 6 ∨ n + m = 4) →
  (y = x^2 / 12 - 3 ∨ y = - (x^2) / 6 + 3 / 2) :=
by
  intros x y h
  cases h
  · -- Case (n - m = 6): Prove y = x^2 / 12 - 3.
    sorry
  · -- Case (n + m = 4): Prove y = - (x^2) / 6 + 3 / 2.
    sorry

end infinite_chain_on_parabola_l285_285365


namespace distance_between_trees_l285_285424

theorem distance_between_trees
  (yard_length : ℝ)
  (num_trees : ℕ)
  (tree_positions : fin num_trees → ℝ)
  (equal_distances : ∀ i j : fin num_trees, i < j → ∃ d, (tree_positions j - tree_positions i) = d ∧ ∀ k, (tree_positions (i + k)) = (tree_positions i + k * d))
  (trees_at_ends : tree_positions 0 = 0 ∧ tree_positions (num_trees - 1) = yard_length) :
  num_trees = 42 → yard_length = 660 → ∃ d, d = 16.1 := sorry

end distance_between_trees_l285_285424


namespace deepak_meet_wife_l285_285412

/-- 
  Proof that Deepak and his wife will meet for the first time after approximately 4.8 minutes,
  given the circumference of the track, their respective speeds, and that they walk in opposite directions.
-/
theorem deepak_meet_wife :
  ∀ (C : ℝ) (v1 v2 : ℝ), 
    C = 660 ∧ v1 = (4.5 * 1000 / 60) ∧ v2 = (3.75 * 1000 / 60) →
    (C / (v1 + v2)) ≈ 4.8 :=
begin
  sorry
end

end deepak_meet_wife_l285_285412


namespace expr_simplify_l285_285242

variable {a b c d m : ℚ}
variable {b_nonzero : b ≠ 0}
variable {m_nat : ℕ}
variable {m_bound : 0 ≤ m_nat ∧ m_nat < 2}

def expr_value (a b c d m : ℚ) : ℚ :=
  m - (c * d) + (a + b) / 2023 + a / b

theorem expr_simplify (h1 : a = -b) (h2 : c * d = 1) (h3 : m = (m_nat : ℚ)) :
  expr_value a b c d m = -1 ∨ expr_value a b c d m = -2 := by
  sorry

end expr_simplify_l285_285242


namespace min_f_triangle_sides_l285_285234

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x ^ 2, Real.sqrt 3)
  let b := (1, Real.sin (2 * x))
  (a.1 * b.1 + a.2 * b.2) - 2

theorem min_f (x : ℝ) (h1 : -Real.pi / 6 ≤ x) (h2 : x ≤ Real.pi / 3) :
  ∃ x₀, f x₀ = -2 ∧ ∀ x, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → f x ≥ -2 :=
  sorry

theorem triangle_sides (a b C : ℝ) (h1 : f C = 1) (h2 : C = Real.pi / 6)
  (h3 : 1 = 1) (h4 : a * b = 2 * Real.sqrt 3) (h5 : a > b) :
  a = 2 ∧ b = Real.sqrt 3 :=
  sorry

end min_f_triangle_sides_l285_285234


namespace mark_points_on_circle_impossible_l285_285965

theorem mark_points_on_circle_impossible :
  ∀ (C : ℝ) (n : ℕ), C = 90 → n = 10 → 
  ¬ (∃ (points : Fin n → ℝ), 
    (∀ (i j : Fin n), (i ≠ j → (∃ k ∈ {1..89}, (abs (points i - points j) = k ∨ abs (C - abs (points i - points j)) = k))))):
  sorry

end mark_points_on_circle_impossible_l285_285965


namespace find_perpendicular_point_l285_285574

def exp_tangent_slope_at_origin : ℝ := 1

def is_perpendicular_tangent (x0 : ℝ) (h0 : x0 > 0) : Prop := 
  let f := fun x => 1 / x
  let f' := -1 / (x0 ^ 2)
  exp_tangent_slope_at_origin * f' = -1

theorem find_perpendicular_point : 
  (∃ (x0 y0 : ℝ), 
   x0 = 1 ∧ y0 = 1 ∧
   is_perpendicular_tangent x0 (by norm_num)) := 
begin
  use [1, 1],
  split,
  { refl },
  split,
  { refl },
  {
    unfold is_perpendicular_tangent,
    simp,
    exact dec_trivial
  }
end

end find_perpendicular_point_l285_285574


namespace mathematicians_contemporaries_l285_285004

noncomputable def probability_contemporaries (span : ℕ) (lifespan : ℕ) : ℚ :=
  let area_total := 1
  let base_unshaded := (span - lifespan) / span
  let height_unshaded := (span - lifespan) / span
  let area_unshaded := 2 * (1 / 2) * base_unshaded * height_unshaded
  area_total - area_unshaded

theorem mathematicians_contemporaries :
  probability_contemporaries 500 100 = 9 / 25 :=
by
  have span := 500
  have lifespan := 100
  have base_unshaded := (span - lifespan : ℕ) / span
  have height_unshaded := (span - lifespan : ℕ) / span
  have area_unshaded := 2 * (1 / 2) * base_unshaded * height_unshaded
  have area_total := 1
  have area_shaded := area_total - area_unshaded
  rw [area_shaded]
  sorry

end mathematicians_contemporaries_l285_285004


namespace optimal_price_l285_285062

-- We define the revenue function R(p)
def R (p : ℝ) : ℝ := 150 * p - 6 * p^2

-- We state the theorem to prove that the maximum revenue is achieved at p = 12.5 under the condition p ≤ 30
theorem optimal_price : ∀ (p : ℝ), (p ≤ 30) → (∀ q, (q ≤ 30) → R(p) ≥ R(q)) → p = 12.5 := 
by
  intros p h_le_max h_max_value
  /- Proof goes here -/
  sorry

end optimal_price_l285_285062


namespace find_good_integers_l285_285527

noncomputable def is_good_integer (n : ℕ) : Prop :=
  ∀ m : ℕ, 1 < m ∧ m < n ∧ Nat.coprime n m → Nat.prime m

theorem find_good_integers :
  {n : ℕ | 1 < n ∧ n < 1979 ∧ is_good_integer n} = {2, 3, 4, 6, 12, 18, 24, 30} :=
by
  sorry

end find_good_integers_l285_285527


namespace bronson_yellow_leaves_l285_285880

theorem bronson_yellow_leaves :
  let thursday_leaves := 12 in
  let friday_leaves := 13 in
  let total_leaves := thursday_leaves + friday_leaves in
  let brown_leaves := total_leaves * 20 / 100 in
  let green_leaves := total_leaves * 20 / 100 in
  let yellow_leaves := total_leaves - (brown_leaves + green_leaves) in
  yellow_leaves = 15 :=
by
  sorry

end bronson_yellow_leaves_l285_285880


namespace integral_of_exponential_l285_285927

theorem integral_of_exponential:
  ∀ (x : ℝ), (∫ (λ x, 3^(7*x - 1/9)) dx) = (λ x, 3^(7*x - 1/9) / (7 * log 3) + C) := 
by
  sorry

end integral_of_exponential_l285_285927


namespace monotonic_intervals_and_value_diff_and_c_range_l285_285585

noncomputable def f (x a b c : ℕ) := x^3 + 3*a*x^2 + 3*b*x + c

theorem monotonic_intervals_and_value_diff_and_c_range 
  (a b c : ℝ) 
  (h_extreme : 4 * a + b + 4 = 0) 
  (h_tangent : 2 * a + b + 2 = 0)
  (h_inequality : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x a b c > 1 - 4 * c^2) : 
  (∀ x y : ℝ, x < y → (f x a b c < f y a b c)) ∨ (f x a b c = f y a b c) ∧
  (∀ x y : ℝ, 0 < x ∧ x < 2 → f x a b c > f y a b c) ∧
  (f 0 a b c - f 2 a b c = 4) ∧
  (c > 1 ∨ c < (-5/4)) :=
begin
  sorry
end

end monotonic_intervals_and_value_diff_and_c_range_l285_285585


namespace determine_real_numbers_l285_285906

theorem determine_real_numbers (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
    (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) :=
sorry

end determine_real_numbers_l285_285906


namespace badgers_win_at_least_five_games_l285_285768

-- Define the problem conditions and the required probability calculation
theorem badgers_win_at_least_five_games :
  let p := 0.5 in
  let n := 9 in
  let probability_at_least_five_wins :=
    ∑ k in Finset.range (n + 1), if k >= 5 then (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) else 0
  in
  probability_at_least_five_wins = 1 / 2 :=
by
  sorry

end badgers_win_at_least_five_games_l285_285768


namespace abc_product_range_l285_285969

def f (x : ℝ) : ℝ :=
if h : x > 0 then
  if x ≤ 9 then |real.log x / real.log 3 - 1|
  else 4 - real.sqrt x
else 0

theorem abc_product_range (a b c : ℝ) (h1 : f(a) = f(b)) (h2 : f(b) = f(c)) (h3 : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  81 < a * b * c ∧ a * b * c < 144 :=
by sorry

end abc_product_range_l285_285969


namespace simplify_trig_identity_l285_285754

-- Defining the trigonometric functions involved
def cot (x : Real) : Real := (cos x) / (sin x)
def tan (x : Real) : Real := (sin x) / (cos x)
def csc (x : Real) : Real := 1 / (sin x)

theorem simplify_trig_identity :
  cot 20 + tan 10 = csc 20 :=
by
  sorry

end simplify_trig_identity_l285_285754


namespace virus_diameter_scientific_notation_l285_285328

theorem virus_diameter_scientific_notation :
    (0.000000105 : ℝ) = 1.05 * 10^(-7) := 
by
  sorry

end virus_diameter_scientific_notation_l285_285328


namespace real_part_of_i_squared_times_1_plus_i_l285_285377

noncomputable def imaginary_unit : ℂ := Complex.I

theorem real_part_of_i_squared_times_1_plus_i :
  (Complex.re (imaginary_unit^2 * (1 + imaginary_unit))) = -1 :=
by
  sorry

end real_part_of_i_squared_times_1_plus_i_l285_285377


namespace partition_subset_sum_l285_285970

variable {p k : ℕ}

def V_p (p : ℕ) := {k : ℕ | p ∣ (k * (k + 1) / 2) ∧ k ≥ 2 * p - 1}

theorem partition_subset_sum (p : ℕ) (hp : Nat.Prime p) (k : ℕ) : k ∈ V_p p := sorry

end partition_subset_sum_l285_285970


namespace lcm_18_30_eq_90_l285_285017

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end lcm_18_30_eq_90_l285_285017


namespace exists_4_distinct_positive_sum_3_prime_not_exists_5_distinct_positive_sum_3_prime_l285_285828

-- Part (a)
theorem exists_4_distinct_positive_sum_3_prime :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ (x y z : ℕ), {x, y, z} ⊆ {a, b, c, d} ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z → Nat.Prime (x + y + z)) :=
sorry

-- Part (b)
theorem not_exists_5_distinct_positive_sum_3_prime :
  ¬ ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    (∀ (x y z : ℕ), {x, y, z} ⊆ {a, b, c, d, e} ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z → Nat.Prime (x + y + z)) :=
sorry

end exists_4_distinct_positive_sum_3_prime_not_exists_5_distinct_positive_sum_3_prime_l285_285828


namespace cheese_mouse_problem_l285_285441

theorem cheese_mouse_problem :
  let cheese_x := 15
  let cheese_y := 14
  let mouse_start_x := 5
  let mouse_start_y := -3
  let mouse_line (x : ℝ) := -4 * x + 22 
  let perpendicular_slope := 1 / 4
  let perpendicular_line (x : ℝ) := (1 / 4) * x + (41 / 4)
  let a := 45 / 17 
  let b := 194 / 17
  in a + b = 239 / 17 :=
by
  -- The proof will go here
  sorry

end cheese_mouse_problem_l285_285441


namespace parallelogram_area_l285_285529

-- Define the base and height of the parallelogram
def base : ℕ := 32
def height : ℕ := 14

-- Define the area calculation function
def area_parallelogram (b h : ℕ) : ℕ := b * h

-- The theorem to prove the area of the parallelogram is 448 square centimeters
theorem parallelogram_area : area_parallelogram base height = 448 := by
  -- Calculations encapsulated within the theorem
  sorry

end parallelogram_area_l285_285529


namespace Mary_more_than_Tim_l285_285697

-- Define the incomes
variables (J T M : ℝ)

-- Conditions
def Tim_income : Prop := T = 0.80 * J
def Mary_income : Prop := M = 1.28 * J

-- Theorem statement to prove
theorem Mary_more_than_Tim (J T M : ℝ) (h1 : Tim_income J T)
  (h2 : Mary_income J M) : ((M - T) / T) * 100 = 60 :=
by
  -- Including sorry to skip the proof
  sorry

end Mary_more_than_Tim_l285_285697


namespace angle_AA1K_eq_angle_BB1M_l285_285268

variables {A B C A1 B1 M K : Type*} [linear_ordered_field A]

-- Heights in a triangle
variable (hA : line A A1)
variable (hB : line B B1)

-- Points M and K on line AB with specific properties
variable (M : point)
variable (K : point)
variable (h_M_on_AB : M.is_on_line A B)
variable (h_K_on_AB : K.is_on_line A B)
variable (h_B1K_parallel_BC : parallel B1 K B C)
variable (h_A1M_parallel_AC : parallel A1 M A C)

-- Prove the angles are equal given the conditions
theorem angle_AA1K_eq_angle_BB1M :
  ∠(A, A1, K) = ∠(B, B1, M) := sorry

end angle_AA1K_eq_angle_BB1M_l285_285268


namespace max_value_of_f_l285_285531

noncomputable def f (x : ℝ) : ℝ := sin x + (√3) * cos x - 2 * sin (3 * x)

theorem max_value_of_f : ∃ (x : ℝ), f x = 16 * √3 / 9 :=
by
  -- Proof to be filled in here
  sorry

end max_value_of_f_l285_285531


namespace sequence_term_2023_l285_285620

theorem sequence_term_2023 (a : ℕ → ℚ) (h₁ : a 1 = 2) 
  (h₂ : ∀ n, 1 / a n - 1 / a (n + 1) - 1 / (a n * a (n + 1)) = 1) : 
  a 2023 = -1 / 2 := 
sorry

end sequence_term_2023_l285_285620


namespace simplify_cot_tan_l285_285734

theorem simplify_cot_tan : Real.cot (Real.pi / 9) + Real.tan (Real.pi / 18) = Real.csc (Real.pi / 9) := 
by
  sorry

end simplify_cot_tan_l285_285734


namespace find_k_from_inequality_l285_285992

variable (k x : ℝ)

theorem find_k_from_inequality (h : ∀ x ∈ Set.Ico (-2 : ℝ) 1, 1 + k / (x - 1) ≤ 0)
  (h₂: 1 + k / (-2 - 1) = 0) :
  k = 3 :=
by
  sorry

end find_k_from_inequality_l285_285992


namespace find_intersection_l285_285229

def set_A : Set ℝ := { x | (sqrt (x^2 - 1) / sqrt x) = 0 }

def set_B : Set ℝ := { y | -2 ≤ y ∧ y ≤ 2 }

theorem find_intersection : (set_A ∩ set_B) = { 1 } :=
by
  sorry

end find_intersection_l285_285229


namespace simplify_trig_identity_l285_285755

-- Defining the trigonometric functions involved
def cot (x : Real) : Real := (cos x) / (sin x)
def tan (x : Real) : Real := (sin x) / (cos x)
def csc (x : Real) : Real := 1 / (sin x)

theorem simplify_trig_identity :
  cot 20 + tan 10 = csc 20 :=
by
  sorry

end simplify_trig_identity_l285_285755


namespace hyperbola_eccentricity_proof_l285_285587

-- Define necessary components and conditions for the hyperbola
def hyperbola_asymptote_ratio (a b : ℝ) : Prop :=
  a / b = (real.sqrt 3) / 2

def hyperbola_eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Given the specific hyperbola equations and asymptotes conditions
theorem hyperbola_eccentricity_proof {a b : ℝ}
  (asymptote_cond : hyperbola_asymptote_ratio a b) :
  hyperbola_eccentricity a (real.sqrt (a^2 + b^2)) = (real.sqrt 21) / 3 :=
by
sorry

end hyperbola_eccentricity_proof_l285_285587


namespace simplify_cot_tan_l285_285710

theorem simplify_cot_tan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := by
  sorry

end simplify_cot_tan_l285_285710


namespace problem_statement_l285_285675

def h (x : ℝ) : ℝ := x + 3
def j (x : ℝ) : ℝ := x / 4
def h_inv (x : ℝ) : ℝ := x - 3
def j_inv (x : ℝ) : ℝ := 4 * x

theorem problem_statement : 
  h (j_inv (h_inv (h_inv (j (h 25))))) = 7 := 
by
  sorry

end problem_statement_l285_285675


namespace AM_eq_AN_l285_285835

open Triangle

-- Definitions based on the provided conditions
variables {A B C D E F G M N : Point}
variables {circumcircle : Circle}
variables (hABC_acute : AcuteTriangle A B C)
variables (hBAC_not60 : ∠BAC ≠ 60)
variables (hBD_tangent : Tangent BD (circumcircle ∆ABC) B)
variables (hCE_tangent : Tangent CE (circumcircle ∆ABC) C)
variables (hBD_CE_equal : BD = CE)
variables (hBC : BD = BC)
variables (hDE_intersects_AB_AC : ∃ F G, Line DE ∩ Extension AB = F ∧ Line DE ∩ Extension AC = G)
variables (hCF_BD_intersect_M : ∃ M, Line CF ∩ Line BD = M)
variables (hCE_BG_intersect_N : ∃ N, Line CE ∩ Line BG = N)

theorem AM_eq_AN (A B C D E F G M N : Point)
  (circumcircle : Circle)
  (hABC_acute : AcuteTriangle A B C)
  (hBAC_not60 : ∠ BAC ≠ 60)
  (hBD_tangent : Tangent BD (circumcircle ∆ABC) B)
  (hCE_tangent : Tangent CE (circumcircle ∆ABC) C)
  (hBD_CE_equal : BD = CE)
  (hBC : BD = BC)
  (hDE_intersects_AB_AC : ∃ F G, Line DE ∩ Extension AB = F ∧ Line DE ∩ Extension AC = G)
  (hCF_BD_intersect_M : ∃ M, Line CF ∩ Line BD = M)
  (hCE_BG_intersect_N : ∃ N, Line CE ∩ Line BG = N) : AM = AN := 
sorry

end AM_eq_AN_l285_285835


namespace area_of_annulus_l285_285466

theorem area_of_annulus (b c a : ℝ) (hb_gt_hc : b > c) (h4a2_eq_b2_sub_c2 : 4 * a^2 = b^2 - c^2) : 
  π * (b^2 - c^2) = 4 * π * a^2 :=
by
  calc
    π * (b^2 - c^2)
        = π * 4 * a^2 : by rw [h4a2_eq_b2_sub_c2]
    _   = 4 * π * a^2 : by ring

end area_of_annulus_l285_285466


namespace Peter_magnets_l285_285095

theorem Peter_magnets (initial_magnets_Adam : ℕ) (given_away_fraction : ℚ) (half_magnets_factor : ℚ) 
  (Adam_left : ℕ) (Peter_magnets : ℕ) :
  initial_magnets_Adam = 18 →
  given_away_fraction = 1/3 →
  initial_magnets_Adam * (1 - given_away_fraction) = Adam_left →
  Adam_left = half_magnets_factor * Peter_magnets →
  half_magnets_factor = 1/2 →
  Peter_magnets = 24 :=
by
  -- Proof goes here
  sorry

-- Providing the values for conditions
#eval Peter_magnets 18 (1/3) (1/2) 12 24

end Peter_magnets_l285_285095


namespace candidate_marks_secured_l285_285064

theorem candidate_marks_secured:
  ∀ (x : ℝ), 
  (∀ (max_marks: ℝ), max_marks = 152.38 → 
  (∀ (pass_percentage: ℝ), pass_percentage = 0.42 → 
  (∀ (fail_by: ℝ), fail_by = 22 →
  let passing_marks := Float.to_int (pass_percentage * max_marks)
  in x + fail_by = passing_marks)))
  → x = 42 :=
by
  intros x max_marks h1 pass_percentage h2 fail_by h3 passing_marks h4
  have h_passing_marks: passing_marks = 64 := by sorry
  have h_fail_expression: x + 22 = 64 := by rw [←h4, h_passing_marks]
  linarith

end candidate_marks_secured_l285_285064


namespace sphere_volume_of_cuboid_vertices_l285_285435

-- Definitions and conditions
def cuboid (a b c : ℝ) : Prop := (a = 1) ∧ (b = 2) ∧ (c = 2)
def body_diagonal (a b c : ℝ) : ℝ := real.sqrt (a^2 + b^2 + c^2)
def sphere_radius (d : ℝ) : ℝ := d / 2
def sphere_volume (r : ℝ) : ℝ := (4 / 3) * real.pi * r^3

-- Theorem statement
theorem sphere_volume_of_cuboid_vertices (a b c : ℝ) 
  (h_cuboid : cuboid a b c) (h_diagonal : body_diagonal a b c = 3) :
  sphere_volume (sphere_radius (body_diagonal a b c)) = (9 / 2) * real.pi :=
by
  -- Proof skipped
  sorry

end sphere_volume_of_cuboid_vertices_l285_285435


namespace order_of_magnitude_l285_285903

noncomputable def a := 5 ^ 0.6
noncomputable def b := 0.6 ^ 5
noncomputable def c := Real.log 5 / Real.log 0.5  -- log_{0.5} 5

theorem order_of_magnitude (a b c : ℝ) (ha : a = 5 ^ 0.6) (hb : b = 0.6 ^ 5) (hc : c = Real.log 5 / Real.log 0.5) :
  c < b ∧ b < a :=
by
  sorry

end order_of_magnitude_l285_285903


namespace M_inequality_l285_285704

def ceiling (x : ℝ) : ℤ := ⌈x⌉

def M (n k h : ℤ) : ℤ := -- Assume this is already defined

theorem M_inequality (n k h : ℤ) : 
  M n k h ≥ ceiling (n / (n - h) * ceiling ((n - 1) / (n - h - 1) * ceiling (k + 1 / (k - h + 1)))) := 
sorry

end M_inequality_l285_285704


namespace trajectory_M_trajectory_P_l285_285182

-- Definitions based on given conditions
def point (α : Type*) := (α × α)
def distance (M A : point ℝ) := real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)

def ratio_of_distances (M A B : point ℝ) (k : ℝ) :=
  distance M A / distance M B = k

-- Given conditions
def A : point ℝ := (1, 0)
def B : point ℝ := (4, 0)

-- Prove the trajectory equation of M
theorem trajectory_M (M : point ℝ) 
  (h : ratio_of_distances M A B (1/2)) :
  M.1^2 + M.2^2 = 4 :=
by sorry

-- Midpoint and associated proof
def midpoint (A B : point ℝ) : point ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def P (M : point ℝ) : point ℝ := midpoint A M

theorem trajectory_P (P : point ℝ) (M : point ℝ)
  (hM : ratio_of_distances M A B (1/2))
  (hP : P = midpoint A M) :
  (P.1 - 1/2)^2 + P.2^2 = 1 :=
by sorry

end trajectory_M_trajectory_P_l285_285182


namespace determine_b_l285_285340

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 1 / (3 * x + b)
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_b (b : ℝ) :
    (∀ x : ℝ, f_inv (f x b) = x) ↔ b = -3 :=
by
  sorry

end determine_b_l285_285340


namespace lcm_18_30_is_90_l285_285035

theorem lcm_18_30_is_90 : Nat.lcm 18 30 = 90 := 
by 
  unfold Nat.lcm
  sorry

end lcm_18_30_is_90_l285_285035


namespace circles_common_tangents_cannot_be_1_or_4_l285_285388

theorem circles_common_tangents_cannot_be_1_or_4
  (r1 r2 : ℝ) (h : r1 = 2 * r2) :
  ∀ n, ¬(((n = 1) ∨ (n = 4)) ↔ ∃ (C1 C2 : ℝ×ℝ × ℝ), (C1.snd ≤ C2.snd) 
    ∧ (C1.snd = r1) ∧ (C2.snd = r2) ∧ number_of_common_tangents C1 C2 = n) :=
by
  sorry

end circles_common_tangents_cannot_be_1_or_4_l285_285388


namespace travel_hours_first_day_l285_285476

theorem travel_hours_first_day
  (h2_1: let d2_1 := 6 * 6)
  (h2_2: let d2_2 := 3 * 3)
  (h2: let d2 := d2_1 + d2_2)
  (h3: let d3 := 7 * 5)
  (h_total: 115 = d2 + d3 + 35)
  (h1_speed: let s1 := 5) : 
  35 / s1 = 7 :=
by
  sorry

end travel_hours_first_day_l285_285476


namespace car_z_mpg_decrease_l285_285482

theorem car_z_mpg_decrease :
  let mpg_45 := 51
  let mpg_60 := 408 / 10
  let decrease := mpg_45 - mpg_60
  let percentage_decrease := (decrease / mpg_45) * 100
  percentage_decrease = 20 := by
  sorry

end car_z_mpg_decrease_l285_285482


namespace overlap_length_in_mm_l285_285403

theorem overlap_length_in_mm {sheets : ℕ} {length_per_sheet : ℝ} {perimeter : ℝ} 
  (h_sheets : sheets = 12)
  (h_length_per_sheet : length_per_sheet = 18)
  (h_perimeter : perimeter = 210) : 
  (length_per_sheet * sheets - perimeter) / sheets * 10 = 5 := by
  sorry

end overlap_length_in_mm_l285_285403


namespace question1_question2_question3_l285_285551

open Real

noncomputable
def f (x : ℝ) := x * log x

noncomputable
def g (x a : ℝ) := x^3 + a * x^2 - x + 2 * f(x)

-- Proof of Question 1
theorem question1 (a : ℝ) : (∀ x ∈ Ioo (-1/3 : ℝ) 1, (3 * x^2 + 2 * a * x - 1) < 0) ↔ a = -1 := sorry

-- Proof of Question 2
theorem question2 : g (-1) (-1) = 1 ∧ (deriv (λ x => g x (-1))) (-1) = 4 → (∀ x, 4 * x - (g x (-1) - 1 + 4) + 5 = 0) := sorry

-- Proof of Question 3
theorem question3 (a : ℝ) : (∃ x > 0, 2 * f(x) ≤ 6 * x + 2 * a + 2) ↔ a ≥ -2 := sorry

end question1_question2_question3_l285_285551


namespace problem_to_prove_l285_285187

-- Define an arithmetic sequence with common difference d (d ≠ 0)
variables {a : ℕ → ℝ} (d : ℝ)
hypothesis (h_d_nonzero : d ≠ 0)

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 0 + (n - 1) * d)

-- Given condition: a_{10} = S_4
hypothesis (h_sequence : a 10 = S d 4)

-- Define the value we need to prove is 4
theorem problem_to_prove : (S d 8) / (a 9) = 4 :=
sorry

end problem_to_prove_l285_285187


namespace find_a_if_purely_imaginary_l285_285612

-- Define the conditions and the problem
def is_purely_imaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

theorem find_a_if_purely_imaginary :
  ∃ a : ℝ, is_purely_imaginary ((2 + Complex.i) * (1 + a * Complex.i)) ∧ a = 2 :=
by
  sorry

end find_a_if_purely_imaginary_l285_285612


namespace area_of_triangle_ABC_l285_285107

theorem area_of_triangle_ABC (A B C : Point) (a b c : ℝ) 
  (ha : ∠ A B C = 90)
  (hb : ∠ B A C = 45)
  (hc : dist B C = 20) :
  area_of_triangle A B C = 100 :=
sorry

end area_of_triangle_ABC_l285_285107


namespace min_value_of_z_l285_285973

-- Define the conditions and objective function
def constraints (x y : ℝ) : Prop :=
  (y ≥ x + 2) ∧ 
  (x + y ≤ 6) ∧ 
  (x ≥ 1)

def z (x y : ℝ) : ℝ :=
  2 * |x - 2| + |y|

-- The formal theorem stating the minimum value of z under the given constraints
theorem min_value_of_z : ∃ x y : ℝ, constraints x y ∧ z x y = 4 :=
sorry

end min_value_of_z_l285_285973


namespace bob_average_calories_l285_285875

-- Definitions of the given conditions
def slices1 : ℕ := 3
def calories1 : ℕ := 300
def slices2 : ℕ := 4
def calories2 : ℕ := 400

def totalCalories : ℕ := slices1 * calories1 + slices2 * calories2
def totalSlices : ℕ := slices1 + slices2
def averageCalories (totalCalories totalSlices : ℕ) : ℚ := totalCalories / totalSlices

-- The problem statement
theorem bob_average_calories :
  averageCalories totalCalories totalSlices ≈ 357.14 :=
by
  -- We skip the proof; it can be filled in as needed.
  sorry

end bob_average_calories_l285_285875


namespace lcm_18_30_is_90_l285_285038

theorem lcm_18_30_is_90 : Nat.lcm 18 30 = 90 := 
by 
  unfold Nat.lcm
  sorry

end lcm_18_30_is_90_l285_285038


namespace inequality_condition_nec_not_suff_l285_285538

variable (a b c : ℝ)
variable (hc : c^2 > 0)

theorem inequality_condition_nec_not_suff : 
  (a > b) → ¬ (∀ c : ℝ, c^2 > 0 → (ac^2 > bc^2) → (a > b)) ∧
  (exists c : ℝ, c^2 > 0 ∧ ac^2 > bc^2) → (a > b) :=
by
  sorry

end inequality_condition_nec_not_suff_l285_285538


namespace pet_store_cages_l285_285440

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) (h_initial : initial_puppies = 13) (h_sold : sold_puppies = 7) (h_per_cage : puppies_per_cage = 2) : (initial_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l285_285440


namespace remainder_div_29_l285_285049

theorem remainder_div_29 (k : ℤ) (N : ℤ) (h : N = 899 * k + 63) : N % 29 = 10 :=
  sorry

end remainder_div_29_l285_285049


namespace height_inequality_triangle_l285_285297

theorem height_inequality_triangle (a b c h_a h_b h_c Δ : ℝ) (n : ℝ) 
  (ha : h_a = 2 * Δ / a)
  (hb : h_b = 2 * Δ / b)
  (hc : h_c = 2 * Δ / c)
  (n_pos : n > 0) :
  (a * h_b)^n + (b * h_c)^n + (c * h_a)^n ≥ 3 * 2^n * Δ^n := 
sorry

end height_inequality_triangle_l285_285297


namespace closest_point_on_line_l285_285154

theorem closest_point_on_line 
  (p : ℝ × ℝ × ℝ) (line_point : ℝ × ℝ × ℝ) (line_dir : ℝ × ℝ × ℝ) (closest_point : ℝ × ℝ × ℝ) :
  p = (1, 4, 2) →
  line_point = (5, -2, 3) →
  line_dir = (3, -1, 2) →
  closest_point = (5/7, -4/7, 1/7) →
  ∃ s : ℝ, closest_point = (,
    (line_point.1 + s * line_dir.1,
    line_point.2 + s * line_dir.2,
    line_point.3 + s * line_dir.3)) ∧
    (closest_point.1 - p.1) * (line_dir.1) +
    (closest_point.2 - p.2) * (line_dir.2) +
    (closest_point.3 - p.3) * (line_dir.3) = 0 :=
begin
  intros hp hline_point hline_dir hclosest_point,
  use -10/7,
  split,
  { rw [hline_point, hline_dir, hclosest_point],
    simp },
  { rw [hclosest_point],
    simp,
    linarith },
end

end closest_point_on_line_l285_285154


namespace ratio_a_b_l285_285230

-- Define the given conditions
variables {x y a b : ℝ}
hypothesis eq1 : 4 * x - 2 * y = a
hypothesis eq2 : 6 * y - 12 * x = b
hypothesis nonzero_b : b ≠ 0

-- State the goal to prove
theorem ratio_a_b : a / b = -1 / 3 :=
by
  sorry

end ratio_a_b_l285_285230


namespace sum_of_digits_of_square_99999_l285_285814

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_square_99999 : sum_of_digits ((99999 : ℕ)^2) = 45 := by
  sorry

end sum_of_digits_of_square_99999_l285_285814


namespace cubic_function_increasing_l285_285575

noncomputable def cubic_function (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + 7 * a * x

theorem cubic_function_increasing (a : ℝ) :
  (∀ (x : ℝ), (3 * x^2 + 2 * a * x + 7 * a) ≥ 0) → (0 ≤ a ∧ a ≤ 21) :=
begin
  intro h,
  sorry,
end

end cubic_function_increasing_l285_285575


namespace lcm_18_30_eq_90_l285_285018

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end lcm_18_30_eq_90_l285_285018


namespace problem_statement_l285_285670

variable {S : ℕ → ℚ}
variable {a : ℕ → ℚ}
variable {n : ℕ}

-- Condition: \( S_n \) is the sum of the first n terms of the sequence \(\{a_n\}\).
def sum_seq (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, S n = (Finset.range n).sum a

-- Condition: \( a_1 = 1 \).
def initial_term (a : ℕ → ℚ) : Prop :=
  a 1 = 1

-- Condition: \(\frac{S_n}{a_n}\) forms an arithmetic sequence with a common difference of \(\frac{1}{3}\).
def arithmetic_seq (S a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, d = 1/3 ∧ ∀ n, (S n / a n) = (S 1 / a 1) + d * (n - 1)

-- The general formula for \(a_n\).
def general_formula (a : ℕ → ℚ) : Prop :=
  ∀ n, a n = n * (n + 1) / 2

-- The harmonic series sum we want to prove is less than 2.
def harmonic_series (a : ℕ → ℚ) : Prop :=
  ∀ n, (Finset.range n).sum (λ k, 1 / a (k + 1)) < 2

-- Combining all goals into a single theorem to be proved.
theorem problem_statement (S a : ℕ → ℚ) :
  sum_seq a S →
  initial_term a →
  arithmetic_seq S a →
  general_formula a ∧ harmonic_series a :=
by
  intros
  split
  -- Prove the general formula
  { sorry }
  -- Prove the harmonic series inequality
  { sorry }

end problem_statement_l285_285670


namespace Turner_ferris_wheel_rides_l285_285003

theorem Turner_ferris_wheel_rides
  (r_coaster : ℕ) (c_coaster : ℕ) (r_catapult : ℕ) (c_catapult : ℕ)
  (r_ferris_wheel : ℕ) (t : ℕ) : 
  r_coaster * c_coaster + r_catapult * c_catapult + r_ferris_wheel = t → r_coaster = 3 → c_coaster = 4 →
  r_catapult = 2 → c_catapult = 4 → r_ferris_wheel = ?m → t = 21 → ?m = 1 :=
by
  intros h1 h2 h3 h4 h5 h6
  have h7 : r_ferris_wheel = 1 := sorry
  exact h7

#extraction lean_proof

-- Sorry is used to skip the actual proof.

end Turner_ferris_wheel_rides_l285_285003


namespace sin_exp_eq_200_l285_285151

theorem sin_exp_eq_200 (x : ℝ) :
  (∃ x ∈ (0, 200 * real.pi), sin x = (1 / 3) ^ x) ↔ 200 := sorry

end sin_exp_eq_200_l285_285151


namespace Freddy_age_l285_285956

theorem Freddy_age :
  ∀ (S J T F Ti : ℕ),
    (S = 4 * J) ∧
    (J = 5) ∧
    (T = 2 * J) ∧
    (F + 2.5 = S) ∧
    (T = 2 * Ti) ∧
    (Ti + 2 = F)
    → F = 7 := by
  sorry

end Freddy_age_l285_285956


namespace mixed_feed_cost_l285_285386

/-- Tim and Judy mix two kinds of feed for pedigreed dogs. They made 35 pounds of feed by mixing 
    one kind worth $0.18 per pound with another worth $0.53 per pound. They used 17 pounds of the cheaper kind in the mix.
    We are to prove that the cost per pound of the mixed feed is $0.36 per pound. -/
theorem mixed_feed_cost
  (total_weight : ℝ) (cheaper_cost : ℝ) (expensive_cost : ℝ) (cheaper_weight : ℝ)
  (total_weight_eq : total_weight = 35)
  (cheaper_cost_eq : cheaper_cost = 0.18)
  (expensive_cost_eq : expensive_cost = 0.53)
  (cheaper_weight_eq : cheaper_weight = 17) :
  ((cheaper_weight * cheaper_cost + (total_weight - cheaper_weight) * expensive_cost) / total_weight) = 0.36 :=
by
  sorry

end mixed_feed_cost_l285_285386


namespace f_neg_alpha_l285_285217

noncomputable def f (x : ℝ) := (Real.tan x) + (1 / (Real.tan x))

theorem f_neg_alpha (α : ℝ) (h : α ≠ (π / 2) + Real.mul_pi_h α ∀ h ∈ ℤ) (hα : f α = 5) : f (-α) = -5 :=
  by
  sorry

end f_neg_alpha_l285_285217


namespace chord_length_l285_285637

theorem chord_length (x y : ℝ) :
    let center := (2 : ℝ, -1 : ℝ),
        radius := 2,
        line := λ (x y : ℝ), x + 2 * y - 3,
        circle := λ (x y : ℝ), (x - 2) ^ 2 + (y + 1) ^ 2 - 4 in
    (∀ (point : ℝ × ℝ), point = center → 
       let d := (|point.fst - 2 - 3|) / (real.sqrt (1 + 4)) in 
       2 * real.sqrt (radius ^ 2 - d ^ 2) = (2 * real.sqrt 55) / 5) :=
by
  sorry

end chord_length_l285_285637


namespace initially_tagged_fish_l285_285627

theorem initially_tagged_fish (second_catch_total : ℕ) (second_catch_tagged : ℕ)
  (total_fish_pond : ℕ) (approx_ratio : ℚ) 
  (h1 : second_catch_total = 50)
  (h2 : second_catch_tagged = 2)
  (h3 : total_fish_pond = 1750)
  (h4 : approx_ratio = (second_catch_tagged : ℚ) / second_catch_total) :
  ∃ T : ℕ, T = 70 :=
by
  sorry

end initially_tagged_fish_l285_285627


namespace expand_polynomial_l285_285918

theorem expand_polynomial (x : ℝ) :
  (x + 4) * (x - 9) = x^2 - 5 * x - 36 := 
sorry

end expand_polynomial_l285_285918


namespace min_training_iterations_l285_285898

/-- The model of exponentially decaying learning rate is given by L = L0 * D^(G / G0)
    where
    L  : the learning rate used in each round of optimization,
    L0 : the initial learning rate,
    D  : the decay coefficient,
    G  : the number of training iterations,
    G0 : the decay rate.

    Given:
    - the initial learning rate L0 = 0.5,
    - the decay rate G0 = 18,
    - when G = 18, L = 0.4,

    Prove: 
    The minimum number of training iterations required for the learning rate to decay to below 0.1 (excluding 0.1) is 130.
-/
theorem min_training_iterations
  (L0 : ℝ) (G0 : ℝ) (D : ℝ) (G : ℝ) (L : ℝ)
  (h1 : L0 = 0.5)
  (h2 : G0 = 18)
  (h3 : L = 0.4)
  (h4 : G = 18)
  (h5 : L0 * D^(G / G0) = 0.4)
  : ∃ G, G ≥ 130 ∧ L0 * D^(G / G0) < 0.1 := sorry

end min_training_iterations_l285_285898


namespace mul_72516_9999_l285_285406

theorem mul_72516_9999 : 72516 * 9999 = 724787484 :=
by
  sorry

end mul_72516_9999_l285_285406


namespace batch_preparation_l285_285870

theorem batch_preparation (total_students cupcakes_per_student cupcakes_per_batch percent_not_attending : ℕ)
    (hlt1 : total_students = 150)
    (hlt2 : cupcakes_per_student = 3)
    (hlt3 : cupcakes_per_batch = 20)
    (hlt4 : percent_not_attending = 20)
    : (total_students * (80 / 100) * cupcakes_per_student) / cupcakes_per_batch = 18 := by
  sorry

end batch_preparation_l285_285870


namespace total_monthly_bill_working_from_home_l285_285306

-- Definitions based on conditions
def original_bill : ℝ := 60
def increase_rate : ℝ := 0.45
def additional_internet_cost : ℝ := 25
def additional_cloud_cost : ℝ := 15

-- The theorem to prove
theorem total_monthly_bill_working_from_home : 
  original_bill * (1 + increase_rate) + additional_internet_cost + additional_cloud_cost = 127 := by
  sorry

end total_monthly_bill_working_from_home_l285_285306


namespace option_C_correct_l285_285043

theorem option_C_correct (a b : ℝ) : (-2 * a * b^3)^2 = 4 * a^2 * b^6 :=
by
  sorry

end option_C_correct_l285_285043


namespace smallest_whole_number_larger_than_any_triangle_perimeter_l285_285040

theorem smallest_whole_number_larger_than_any_triangle_perimeter
  (s : ℝ) (h1 : 7 < s) (h2 : s < 17) : 34 = 34 :=
by
  -- We have the sides of the triangle are 5, 12, and s
  let perimeter := 5 + 12 + s
  -- Based on conditions, the maximum value for the perimeter is just less than 34
  have h_perimeter_max : perimeter < 34 :=
    by 
    simp [perimeter]
    linarith
  -- Hence, the smallest whole number larger than the perimeter is 34
  exact (eq.refl 34)

end smallest_whole_number_larger_than_any_triangle_perimeter_l285_285040


namespace factorial_division_l285_285504
-- Definition of factorial
def fact : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * fact n

-- Problem statement
theorem factorial_division : fact 12 / fact 11 = 12 :=
by sorry

end factorial_division_l285_285504


namespace simplify_cotAndTan_l285_285722

theorem simplify_cotAndTan :
  Real.cot (20 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (20 * Real.pi / 180) := 
by
  sorry

end simplify_cotAndTan_l285_285722


namespace evaluate_product_l285_285913

theorem evaluate_product : 
  ∏ i in finset.range 2009, (i + 2) / (i + 1) * (1 / 2) = 502.5 := 
by sorry

end evaluate_product_l285_285913


namespace exponent_zero_nonneg_l285_285706

theorem exponent_zero_nonneg (a : ℝ) (h : a ≠ -1) : (a + 1) ^ 0 = 1 :=
sorry

end exponent_zero_nonneg_l285_285706


namespace log_evaluation_l285_285246

theorem log_evaluation
  (x : ℝ)
  (h : x = (Real.log 3 / Real.log 5) ^ (Real.log 5 / Real.log 3)) :
  Real.log x / Real.log 7 = -(Real.log 5 / Real.log 3) * (Real.log (Real.log 5 / Real.log 3) / Real.log 7) :=
by
  sorry

end log_evaluation_l285_285246


namespace largest_whole_number_lt_div_l285_285367

theorem largest_whole_number_lt_div {x : ℕ} (hx : 8 * x < 80) : x ≤ 9 :=
by
  sorry

end largest_whole_number_lt_div_l285_285367


namespace tetrahedron_edge_length_correct_l285_285948

noncomputable def radius := Real.sqrt 2
noncomputable def center_to_center_distance := 2 * radius
noncomputable def tetrahedron_edge_length := center_to_center_distance

theorem tetrahedron_edge_length_correct :
  tetrahedron_edge_length = 2 * Real.sqrt 2 := by
  sorry

end tetrahedron_edge_length_correct_l285_285948


namespace problem_statement_l285_285671

variable {S : ℕ → ℚ}
variable {a : ℕ → ℚ}
variable {n : ℕ}

-- Condition: \( S_n \) is the sum of the first n terms of the sequence \(\{a_n\}\).
def sum_seq (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, S n = (Finset.range n).sum a

-- Condition: \( a_1 = 1 \).
def initial_term (a : ℕ → ℚ) : Prop :=
  a 1 = 1

-- Condition: \(\frac{S_n}{a_n}\) forms an arithmetic sequence with a common difference of \(\frac{1}{3}\).
def arithmetic_seq (S a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, d = 1/3 ∧ ∀ n, (S n / a n) = (S 1 / a 1) + d * (n - 1)

-- The general formula for \(a_n\).
def general_formula (a : ℕ → ℚ) : Prop :=
  ∀ n, a n = n * (n + 1) / 2

-- The harmonic series sum we want to prove is less than 2.
def harmonic_series (a : ℕ → ℚ) : Prop :=
  ∀ n, (Finset.range n).sum (λ k, 1 / a (k + 1)) < 2

-- Combining all goals into a single theorem to be proved.
theorem problem_statement (S a : ℕ → ℚ) :
  sum_seq a S →
  initial_term a →
  arithmetic_seq S a →
  general_formula a ∧ harmonic_series a :=
by
  intros
  split
  -- Prove the general formula
  { sorry }
  -- Prove the harmonic series inequality
  { sorry }

end problem_statement_l285_285671


namespace f_zero_f_increasing_on_negative_l285_285205

noncomputable def f : ℝ → ℝ := sorry
variable {x : ℝ}

-- Assume f is an odd function
axiom odd_f : ∀ x, f (-x) = -f x

-- Assume f is increasing on (0, +∞)
axiom increasing_f_on_positive :
  ∀ ⦃x₁ x₂⦄, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂

-- Prove that f(0) = 0
theorem f_zero : f 0 = 0 := sorry

-- Prove that f is increasing on (-∞, 0)
theorem f_increasing_on_negative :
  ∀ ⦃x₁ x₂⦄, x₁ < x₂ → x₂ < 0 → f x₁ < f x₂ := sorry

end f_zero_f_increasing_on_negative_l285_285205


namespace shaded_rectangle_area_fraction_l285_285315

theorem shaded_rectangle_area_fraction :
  let grid_side := 6
  let points := [(2, 2), (4, 4), (2, 4), (4, 6)]
  let rectangle_side := 2
  let diagonal := 2 * Real.sqrt 2
  let rectangle_area := rectangle_side * diagonal
  let grid_area := grid_side ^ 2
  let fraction := rectangle_area / grid_area
  fraction = Real.sqrt 2 / 9 :=
by
  simp only [grid_side, points, rectangle_side, diagonal, rectangle_area, grid_area, fraction]
  -- to be filled in during proof
  -- by logic of solution provided.
  sorry

end shaded_rectangle_area_fraction_l285_285315


namespace largest_inscribed_rectangle_l285_285774

theorem largest_inscribed_rectangle {a b m : ℝ} (h : m ≥ b) :
  ∃ (base height area : ℝ),
    base = a * (b + m) / m ∧ 
    height = (b + m) / 2 ∧ 
    area = a * (b + m)^2 / (2 * m) :=
sorry

end largest_inscribed_rectangle_l285_285774


namespace notebooks_bought_l285_285516

def dan_total_spent : ℕ := 32
def backpack_cost : ℕ := 15
def pens_cost : ℕ := 1
def pencils_cost : ℕ := 1
def notebook_cost : ℕ := 3

theorem notebooks_bought :
  ∃ x : ℕ, dan_total_spent - (backpack_cost + pens_cost + pencils_cost) = x * notebook_cost ∧ x = 5 := 
by
  sorry

end notebooks_bought_l285_285516


namespace cyclic_points_l285_285972

theorem cyclic_points 
  (A B C D E F X Y Z : Point) 
  (h1 : IncircleTouchPoints A B C D E F)
  (h2 : InTriangle A B C X)
  (h3 : IncircleTouchesBC X B C D)
  (h4 : IncircleTouchesCX X C E Y)
  (h5 : IncircleTouchesBX X B F Z) :
  concyclic {E, F, Z, Y} := 
sorry

end cyclic_points_l285_285972


namespace num_2_coins_l285_285604

open Real

theorem num_2_coins (x y z : ℝ) (h1 : x + y + z = 900)
                     (h2 : x + 2 * y + 5 * z = 1950)
                     (h3 : z = 0.5 * x) : y = 450 :=
by sorry

end num_2_coins_l285_285604


namespace projection_matrix_onto_vector_1_neg4_l285_285936

open Matrix

-- Define the vector
def u : Vector ℝ 2 := ![1, -4]

-- Projection matrix
def projectionMatrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  [1/17, -4/17],
  [-4/17, 16/17]
]

-- Statement of the problem
theorem projection_matrix_onto_vector_1_neg4 :
  ∀ (x : ℝ) (y : ℝ), 
    let v := ![x, y]
    let proj_v := projectionMatrix.mulVec v
    proj_v = ((v.dotProduct u) / (u.dotProduct u)) • u :=
by
  -- Proof omitted
  sorry

end projection_matrix_onto_vector_1_neg4_l285_285936


namespace standard_eq_of_largest_circle_l285_285638

theorem standard_eq_of_largest_circle 
  (m : ℝ)
  (hm : 0 < m) :
  ∃ r : ℝ, 
  (∀ x y : ℝ, (x^2 + (y - 1)^2 = 8) ↔ 
      (x^2 + (y - 1)^2 = r)) :=
sorry

end standard_eq_of_largest_circle_l285_285638


namespace min_intersecting_triples_count_l285_285115

open Set

def minimally_intersecting (A B C : Set Nat) : Prop :=
  (| A ∩ B | = 1) ∧ (| B ∩ C | = 1) ∧ (| C ∩ A | = 1) ∧ (A ∩ B ∩ C = ∅)

def subsets_of_eight : Set (Set Nat) :=
  { S | S ⊆ {1, 2, 3, 4, 5, 6, 7, 8}}

def number_min_intersecting_triples : Nat :=
  Nat.card { (A, B, C) | A ∈ subsets_of_eight ∧ B ∈ subsets_of_eight ∧ C ∈ subsets_of_eight ∧ minimally_intersecting A B C }

theorem min_intersecting_triples_count :
  number_min_intersecting_triples = 344064 := sorry

end min_intersecting_triples_count_l285_285115


namespace calc_hash_2_5_3_l285_285688

def operation_hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem calc_hash_2_5_3 : operation_hash 2 5 3 = 1 := by
  sorry

end calc_hash_2_5_3_l285_285688


namespace cos_product_sum_l285_285963

theorem cos_product_sum {α : ℝ} (hα : α = Real.pi / 13) : 
  let x := Real.cos (Real.pi / 13)
      y := Real.cos (3 * Real.pi / 13)
      z := Real.cos (9 * Real.pi / 13)
  in x * y + y * z + z * x = - 1 / 4 := 
begin
  let x := Real.cos (Real.pi / 13),
  let y := Real.cos (3 * Real.pi / 13),
  let z := Real.cos (9 * Real.pi / 13),
  have hx : x = Real.cos (Real.pi / 13) := by rfl,
  have hy : y = Real.cos (3 * Real.pi / 13) := by rfl,
  have hz : z = Real.cos (9 * Real.pi / 13) := by rfl,
  show x * y + y * z + z * x = - 1 / 4,
  sorry
end

end cos_product_sum_l285_285963


namespace area_of_enclosed_figure_l285_285347

def line (x : ℝ) : ℝ := 2 - x
def curve (x : ℝ) : ℝ := x^3

noncomputable def enclosed_area : ℝ :=
  ∫ x in 0..1, curve x + ∫ x in 1..2, line x

theorem area_of_enclosed_figure :
  enclosed_area = 3 / 4 :=
sorry

end area_of_enclosed_figure_l285_285347


namespace tan_sum_half_angles_leq_circumradius_area_l285_285985

theorem tan_sum_half_angles_leq_circumradius_area (R S a b c : ℝ) (A B C : ℝ) :
  (tan (A / 2) + tan (B / 2) + tan (C / 2)) ≤ (9 * R ^ 2) / (4 * S) :=
sorry

end tan_sum_half_angles_leq_circumradius_area_l285_285985


namespace no_nonnegative_integer_solutions_to_eq_l285_285139

theorem no_nonnegative_integer_solutions_to_eq :
  ¬ ∃ (x: Fin 14 → ℕ), (∑ i, (x i)^4) = 1599 := by
  sorry

end no_nonnegative_integer_solutions_to_eq_l285_285139


namespace max_lambda_l285_285690

def vector (α : Type) := list α

def dot_product (v1 v2 : vector ℝ) : ℝ :=
  (v1.headD 0) * (v2.headD 0) + (v1.tail.headD 0) * (v2.tail.headD 0)

theorem max_lambda (x y : ℝ) (h1 : x > 0) (h2 : x^2 - y^2 = 1)
  (h3 : ∀ P : vector ℝ, dot_product P (vector.cons x (vector.cons y vector.nil)) = 1
      → distance_to_line P x (-y) 1 0 > λ) :
  λ ≤ (↑(sqrt 2) / 2) :=
sorry

end max_lambda_l285_285690


namespace max_bishops_correct_bishop_position_count_correct_l285_285462

-- Define the parameters and predicates
def chessboard_size : ℕ := 2015

def max_bishops (board_size : ℕ) : ℕ := 2 * board_size - 1 - 1

def bishop_position_count (board_size : ℕ) : ℕ := 2 ^ (board_size - 1) * 2 * 2

-- State the equalities to be proved
theorem max_bishops_correct : max_bishops chessboard_size = 4028 := by
  -- proof will be here
  sorry

theorem bishop_position_count_correct : bishop_position_count chessboard_size = 2 ^ 2016 := by
  -- proof will be here
  sorry

end max_bishops_correct_bishop_position_count_correct_l285_285462


namespace cot20_tan10_eq_csc20_l285_285745

theorem cot20_tan10_eq_csc20 : 
  Real.cot 20 * (Real.pi / 180) + Real.tan 10 * (Real.pi / 180) = Real.csc 20 * (Real.pi / 180) :=
by sorry

end cot20_tan10_eq_csc20_l285_285745


namespace sum_max_min_g_eq_four_l285_285204

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f x + 2

theorem sum_max_min_g_eq_four (a : ℝ) (h_odd : ∀ x : ℝ, -a ≤ x ∧ x ≤ a → f (-x) = -f x) :
  ∃ m : ℝ, (∀ x : ℝ, -a ≤ x ∧ x ≤ a → m ≤ f x) ∧ (∀ x : ℝ, -a ≤ x ∧ x ≤ a → f x ≤ -m) 
  ∧ (let g_max := 2 - m in let g_min := m + 2 in g_max + g_min = 4) :=
sorry

end sum_max_min_g_eq_four_l285_285204


namespace least_n_prime_condition_l285_285148

theorem least_n_prime_condition : ∃ n : ℕ, (∀ p : ℕ, Prime p → ¬ Prime (p^2 + n)) ∧ (∀ m : ℕ, 
 (m > 0 ∧ ∀ p : ℕ, Prime p → ¬ Prime (p^2 + m)) → m ≥ 5) ∧ n = 5 := by
  sorry

end least_n_prime_condition_l285_285148


namespace actual_selling_price_approx_14_l285_285852

noncomputable def CP := 18.94 / 1.15
noncomputable def SP_loss := CP * 0.85

theorem actual_selling_price_approx_14 :
  SP_loss ≈ 14.00 :=
by
  sorry

end actual_selling_price_approx_14_l285_285852


namespace cot20_plus_tan10_eq_csc20_l285_285725

theorem cot20_plus_tan10_eq_csc20 : Real.cot 20 * Real.pi / 180 + Real.tan 10 * Real.pi / 180 = Real.csc 20 * Real.pi / 180 := 
by
  sorry

end cot20_plus_tan10_eq_csc20_l285_285725


namespace equidistant_planes_tetrahedron_l285_285602

-- Define the four points A, B, C, D
variables (A B C D : Type) [Point A] [Point B] [Point C] [Point D]

-- Define the condition that A, B, C, D do not lie in the same plane
def not_coplanar (A B C D : Point) : Prop := 
  ¬(∃ (plane : Plane), A ∈ plane ∧ B ∈ plane ∧ C ∈ plane ∧ D ∈ plane)

-- Define a tetrahedron formed by points A, B, C, D
def is_tetrahedron (A B C D : Point) : Prop := 
  ¬coplanar A B C D

-- Define the number of planes equidistant from the vertices of the tetrahedron
def equidistant_planes : ℕ :=
  7

-- Theorem: There are exactly 7 planes equidistant from four non-coplanar points A, B, C, D
theorem equidistant_planes_tetrahedron (A B C D : Point) (h : is_tetrahedron A B C D) :
  ∃ n : ℕ, n = 7 := by
  existsi 7
  exact sorry

end equidistant_planes_tetrahedron_l285_285602


namespace compute_expression_l285_285111

theorem compute_expression (y : ℕ) (h : y = 3) : (y^8 + 10 * y^4 + 25) / (y^4 + 5) = 86 :=
by
  rw [h]
  sorry

end compute_expression_l285_285111


namespace cabbage_pies_in_segment_l285_285791

def cabbage_pies_exists (total_pies : ℕ) (cabbage_pies : ℕ) (segment_length : ℕ) (k : ℕ) : Prop :=
  ∀ (arrangement : Fin total_pies → bool),
  (∃ start_idx,
    (finset.range segment_length).sum (λ i, if arrangement ((start_idx + i) % total_pies) then 1 else 0) = k)

theorem cabbage_pies_in_segment :
  cabbage_pies_exists 100 53 67 35 ∧ cabbage_pies_exists 100 53 67 36 :=
by
  sorry

end cabbage_pies_in_segment_l285_285791


namespace isosceles_triangle_vertex_angle_l285_285101

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) (h_triangle : α + β + γ = 180)
  (h_isosceles : α = β ∨ β = α ∨ α = γ ∨ γ = α ∨ β = γ ∨ γ = β)
  (h_ratio : α / γ = 1 / 4 ∨ γ / α = 1 / 4) :
  (γ = 20 ∨ γ = 120) :=
sorry

end isosceles_triangle_vertex_angle_l285_285101


namespace value_of_fraction_sum_l285_285660

variables (a b c x y z : ℝ)
hypothesis h1 : 17 * x + b * y + c * z = 0
hypothesis h2 : a * x + 29 * y + c * z = 0
hypothesis h3 : a * x + b * y + 53 * z = 0
hypothesis h4 : a ≠ 17
hypothesis h5 : x ≠ 0

theorem value_of_fraction_sum : (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 :=
sorry

end value_of_fraction_sum_l285_285660


namespace cot20_tan10_eq_csc20_l285_285750

theorem cot20_tan10_eq_csc20 : 
  Real.cot 20 * (Real.pi / 180) + Real.tan 10 * (Real.pi / 180) = Real.csc 20 * (Real.pi / 180) :=
by sorry

end cot20_tan10_eq_csc20_l285_285750


namespace brads_zip_code_l285_285105

theorem brads_zip_code (A B C D E : ℕ) (h1 : A + B + C + D + E = 20)
                        (h2 : B = A + 1) (h3 : C = A)
                        (h4 : D = 2 * A) (h5 : D + E = 13)
                        (h6 : Nat.Prime (A*10000 + B*1000 + C*100 + D*10 + E)) :
                        A*10000 + B*1000 + C*100 + D*10 + E = 34367 := 
sorry

end brads_zip_code_l285_285105


namespace multiply_polynomials_l285_285311

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l285_285311


namespace height_of_pillar_S_l285_285868

noncomputable theory
open Real

def vertices (P Q R : ℝ × ℝ) : Prop :=
  P = (0, 0) ∧ Q = (10, 0) ∧ R = (0, 15)

def slope_angle (angle : ℝ) : Prop :=
  angle = π / 6  -- 30 degrees in radians

def pillar_heights (hP hQ hR : ℝ) : Prop :=
  hP = 7 ∧ hQ = 10 ∧ hR = 12

theorem height_of_pillar_S
  (P Q R : ℝ × ℝ)
  (angle : ℝ)
  (hP hQ hR : ℝ)
  (z_S : ℝ)
  (hv : vertices P Q R)
  (ha : slope_angle angle)
  (hp : pillar_heights hP hQ hR) :
  z_S = 7.5 * sqrt 3 := sorry

end height_of_pillar_S_l285_285868


namespace conference_handshakes_l285_285871

theorem conference_handshakes :
    ∀ (A B : Finset ℕ), A.card = 25 ∧ B.card = 5 ∧ 
    (∀ (a a' ∈ A), a ≠ a' → knows a a') ∧
    (∀ (b ∈ B), ¬knows b ∅) →
    (∑ x in B, ∑ y in B \ {x}, 1) = 10 :=
by
  intros A B h
  sorry -- proof to be provided

end conference_handshakes_l285_285871


namespace expand_product_l285_285915

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 :=
by
  sorry

end expand_product_l285_285915


namespace train_length_l285_285090

theorem train_length (L : ℕ) (speed : ℕ) 
  (h1 : L + 1200 = speed * 45) 
  (h2 : L + 180 = speed * 15) : 
  L = 330 := 
sorry

end train_length_l285_285090


namespace original_selling_price_l285_285308

theorem original_selling_price (cost_price : ℕ) (discount_rate profit_margin : ℚ) 
  (hp : cost_price = 20000) (hd : discount_rate = 0.1) (hm : profit_margin = 0.08) :
  let selling_price := 24000 in
  (selling_price - (discount_rate * selling_price)) = cost_price * (1 + profit_margin) :=
by
  sorry

end original_selling_price_l285_285308


namespace sum_y_coords_of_circle_y_axis_points_l285_285889

theorem sum_y_coords_of_circle_y_axis_points 
  (h : ∀ x y : ℝ, (x + 3)^2 + (y - 5)^2 = 64) :
  (-3, 5).snd + sqrt 55 + (-3, 5).snd - sqrt 55 = 10 :=
by
  sorry

end sum_y_coords_of_circle_y_axis_points_l285_285889


namespace range_of_f_l285_285176

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := 3 * x ^ k

def valid_k (k : ℝ) : Prop := k > 0

def valid_domain (x : ℝ) : Prop := x ≥ 2

theorem range_of_f (k : ℝ) (h_k : valid_k k) : 
  (set.range (λ x, f x k) ∩ set.Ici 2) = set.Ici (3 * 2^k) :=
sorry

end range_of_f_l285_285176


namespace car_travel_distance_l285_285066

theorem car_travel_distance (speed time : ℕ) (h1 : speed = 65) (h2 : time = 11) : speed * time = 715 :=
by
  rw [h1, h2]
  norm_num
  sorry

end car_travel_distance_l285_285066


namespace f_periodic_l285_285303

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder since we assume some \( f(x) \) exists

variable (c : ℝ) 
variable (x y : ℝ)

-- Conditions
axiom functional_eq : ∀ (x y : ℝ), f (x + y) + f (x - y) = 2 * f x * f y
axiom exists_c : c > 0 ∧ f (c / 2) = 0

-- The periodicity assertion
theorem f_periodic : ∃ T : ℝ, ∀ x : ℝ, f (x + T) = f x := 
begin 
  -- Proof would go here based on the steps shown in the solution,
  -- but we acknowledge the existence of such T, where T = 2c.
  use 2 * c,
  intros x,
  -- Placeholder proof (actual proof left as an exercise)
  sorry
end

end f_periodic_l285_285303


namespace find_b_eq_sqrt2_l285_285994

open Real

theorem find_b_eq_sqrt2
  (C : ℝ → ℝ × ℝ) (l : ℝ → ℝ × ℝ) (b : ℝ)
  (theta : ℝ → ℝ)
  (C_def : ∀ θ, C θ = (2 * cos θ, 2 * sin θ))
  (l_def : ∀ t, l t = (t, t + b))
  (equidistant_pts : (∃ θ1 θ2 θ3,
    (C θ1 = (2 * cos θ1, 2 * sin θ1)) ∧
    (C θ2 = (2 * cos θ2, 2 * sin θ2)) ∧
    (C θ3 = (2 * cos θ3, 2 * sin θ3)) ∧
    (θ1 ≠ θ2 ∧ θ1 ≠ θ3 ∧ θ2 ≠ θ3) ∧
    (∀ θ, (C θ = (2 * cos θ, 2 * sin θ)) → 
    (|2 * (cos θ - sin θ) + b| / √2 = 1))) :
  b = sqrt 2 ∨ b = -sqrt 2 :=
by
  sorry

end find_b_eq_sqrt2_l285_285994


namespace eccentricity_of_ellipse_slope_of_l_equation_of_ellipse_with_M_N_l285_285188

noncomputable def ellipse_eq (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b c : ℝ) : ℝ :=
  c / a

-- Given conditions
variables (a b : ℝ) (h_ab : a > b) (h_b0 : b > 0)
variables (c : ℝ) (h_c : c^2 = a^2 - b^2)
variables (x y : ℝ)
def point_on_ellipse : Prop :=
  ellipse_eq a b x y

-- Part (I) 
theorem eccentricity_of_ellipse (h_triangle : b / c = sqrt 3 / 3) :
  eccentricity a b c = sqrt 3 / 2 :=
sorry

-- Part (II) 
-- (i)
variables (k : ℝ)
theorem slope_of_l (h_s_sq : a^2 = 4 * b^2)
  (h_system_sol : ∃ (xA xC yC : ℝ), 
    xA = 0 ∧ xC = 8 * k * b / (4 * k^2 + 1) ∧ yC = (4 * k^2 * b - b) / (4 * k^2 + 1))
  (h_PQ : |((4 * k * b + 2 * b + 8 * k^2 * b) / (8 * k^2 * b)| = 7 / 4) :
  k = 1 :=
sorry

-- (ii) 
variables (Mx My : ℝ) (N : ℝ × ℝ)
theorem equation_of_ellipse_with_M_N (h_M : Mx = -4/5 ∧ My = -4/5) 
  (h_parallelogram : quadrilateral_is_parallelogram (0, -b) (8 * b / 5, 3 * b / 5) (Mx, My) N)
  (h_N_ellipse : point_on_ellipse N.1 N.2) :
  ellipse_eq 4 2 N.1 N.2 :=
sorry

end eccentricity_of_ellipse_slope_of_l_equation_of_ellipse_with_M_N_l285_285188


namespace min_abs_val_sum_l285_285370

theorem min_abs_val_sum : ∃ x : ℝ, (∀ y : ℝ, |y - 1| + |y - 2| + |y - 3| ≥ |x - 1| + |x - 2| + |x - 3|) ∧ |x - 1| + |x - 2| + |x - 3| = 1 :=
sorry

end min_abs_val_sum_l285_285370


namespace no_multiple_of_2_pow_101_minus_1_is_fancy_l285_285442

/-- A positive integer is called fancy if it can be expressed as a sum of at most 100 powers of 2. -/
def isFancy (n : ℕ) : Prop :=
  ∃ (a : Fin 101 → ℕ), n = (Finset.univ.sum (λ (i : Fin 101), 2 ^ (a i)))

theorem no_multiple_of_2_pow_101_minus_1_is_fancy :
  ∀ m : ℕ, ¬isFancy (m * (2 ^ 101 - 1)) :=
by
  intro m
  sorry

end no_multiple_of_2_pow_101_minus_1_is_fancy_l285_285442


namespace matrix_power_is_correct_l285_285489

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]
def A_cubed : Matrix (Fin 2) (Fin 2) ℤ := !![3, -6; 6, -3]

theorem matrix_power_is_correct : A ^ 3 = A_cubed := by 
  sorry

end matrix_power_is_correct_l285_285489


namespace number_of_newborn_members_in_group_l285_285256

noncomputable def N : ℝ :=
  let p_death := 1 / 10
  let p_survive := 1 - p_death
  let prob_survive_3_months := p_survive * p_survive * p_survive
  218.7 / prob_survive_3_months

theorem number_of_newborn_members_in_group : N = 300 := by
  sorry

end number_of_newborn_members_in_group_l285_285256


namespace median_is_106_l285_285631

noncomputable def cumulative_count (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

noncomputable def median_position (total_elements : ℕ) : ℕ :=
  (total_elements + 1) / 2

noncomputable def total_elements : ℕ :=
  cumulative_count 150

noncomputable def median_value : ℕ :=
  let pos := median_position total_elements in
  Nat.find (λ n, cumulative_count n ≥ pos)

theorem median_is_106 : median_value = 106 :=
sorry

end median_is_106_l285_285631


namespace target_heart_rate_30_58_l285_285867

def adjusted_max_heart_rate (age : ℕ) : ℕ :=
  205 - Nat.div (age * 1) 2

def target_heart_rate (max_hr : ℕ) : ℚ :=
  0.75 * max_hr

def adjusted_target_heart_rate (age : ℕ) (rest_hr : ℕ) : ℚ :=
  let max_hr := adjusted_max_heart_rate age
  let target_hr := target_heart_rate max_hr
  if rest_hr < 60 then target_hr + 5 else target_hr

theorem target_heart_rate_30_58 :
  (adjusted_target_heart_rate 30 58).round = 148 :=
by sorry

end target_heart_rate_30_58_l285_285867


namespace range_of_a_eqn_unique_solution_l285_285997

theorem range_of_a_eqn_unique_solution :
  ∀ a : ℝ, (∀ x : ℝ, 1 < x ∧ x < 3 → log (x - 1) + log (3 - x) = log (x - a) → false) ↔ 
  (a = 3 / 4 ∨ (1 ≤ a ∧ a < 3)) := 
by
  intros a
  constructor
  · intro H
    sorry
  · intro H
    sorry

end range_of_a_eqn_unique_solution_l285_285997


namespace sequence_10th_term_l285_285226

theorem sequence_10th_term (a : ℕ → ℝ) 
  (h_initial : a 1 = 1) 
  (h_recursive : ∀ n, a (n + 1) = 2 * a n / (a n + 2)) : 
  a 10 = 2 / 11 :=
sorry

end sequence_10th_term_l285_285226


namespace prob_A_not_losing_prob_A_not_winning_l285_285629

-- Definitions based on the conditions
def prob_winning : ℝ := 0.41
def prob_tie : ℝ := 0.27

-- The probability of A not losing
def prob_not_losing : ℝ := prob_winning + prob_tie

-- The probability of A not winning
def prob_not_winning : ℝ := 1 - prob_winning

-- Proof problems
theorem prob_A_not_losing : prob_not_losing = 0.68 := by
  sorry

theorem prob_A_not_winning : prob_not_winning = 0.59 := by
  sorry

end prob_A_not_losing_prob_A_not_winning_l285_285629


namespace B_percentage_more_than_C_l285_285475

variables (A B C D : ℕ)
variable (full_marks : ℕ)
variable (A_marks : ℕ)

def marks_obtained_D : ℕ := (80 * full_marks) / 100
def marks_obtained_C : ℕ := marks_obtained_D - ((20 * marks_obtained_D) / 100)
def marks_obtained_B : ℕ := (100 * A_marks) / 90

theorem B_percentage_more_than_C 
  (h_full_marks : full_marks = 500)
  (h_A_marks : A_marks = 360)
  (h_B_def : A_marks = marks_obtained_B)
  (h_C_def : marks_obtained_C = marks_obtained_D - ((20 * marks_obtained_D) / 100))
  (h_D_def : marks_obtained_D = (80 * full_marks) / 100) :
  ((marks_obtained_B - marks_obtained_C) * 100) / marks_obtained_C = 25 :=
  sorry

end B_percentage_more_than_C_l285_285475


namespace operation_is_multiplication_l285_285592

/-!
Given the set P of positive odd numbers, and the set M defined by the operation 
⊕ on P, if M ⊆ P, then we need to prove that ⊕ corresponds to multiplication.
-/

def odd (n : ℕ) : Prop := n % 2 = 1

noncomputable def P : set ℕ := { n | n > 0 ∧ odd n }

def M (f : ℕ → ℕ → ℕ) : set ℕ := { x | ∃ a b, a ∈ P ∧ b ∈ P ∧ x = f a b }

theorem operation_is_multiplication (f : ℕ → ℕ → ℕ) 
  (h : M f ⊆ P) : f = λ a b, a * b :=
begin
  sorry
end

end operation_is_multiplication_l285_285592


namespace average_temp_addington_l285_285882

def temperatures : List ℚ := [60, 59, 56, 53, 49, 48, 46]

def average_temp (temps : List ℚ) : ℚ := (temps.sum) / temps.length

theorem average_temp_addington :
  average_temp temperatures = 53 := by
  sorry

end average_temp_addington_l285_285882


namespace interval_length_and_range_l285_285116

theorem interval_length_and_range (a : ℝ) :
  let f := (x : ℝ) → (x^2 + (2 * a^2 + 2) * x - a^2 + 4 * a - 7) / 
                     (x^2 + (a^2 + 4 * a - 5) * x - a^2 + 4 * a - 7)
  (solution_set : set ℝ := {x | f x < 0})
  (intervals : set (set ℝ) := {I | ∃ a b, I = set.Ioo a b ∨ I = set.Ico a b ∨ I = set.Icc a b ∨ I = set.Ioc a b})
  (length : ℝ := ∑ I in intervals, I.upper - I.lower)
  (h_sum_length : length ≥ 4) :
  a ≤ 1 ∨ a ≥ 3 :=
by {
  sorry,
}

end interval_length_and_range_l285_285116


namespace symmetric_line_equation_y_axis_l285_285357

theorem symmetric_line_equation_y_axis (x y : ℝ) : 
  (∃ m n : ℝ, (y = 3 * x + 1) ∧ (x + m = 0) ∧ (y = n) ∧ (n = 3 * m + 1)) → 
  y = -3 * x + 1 :=
by
  sorry

end symmetric_line_equation_y_axis_l285_285357


namespace longest_diagonal_l285_285448

-- Define the given conditions of the rhombus
def area (d1 d2 : ℝ) : ℝ := (1 / 2) * d1 * d2
def ratio (d1 d2 : ℝ) : ℝ := d1 / d2

-- The main theorem we want to prove
theorem longest_diagonal (d1 d2 : ℝ) 
  (h_area : area d1 d2 = 150) 
  (h_ratio : ratio d1 d2 = 4 / 3) : d1 = 20 := 
sorry

end longest_diagonal_l285_285448


namespace volume_of_pyramid_l285_285892

theorem volume_of_pyramid :
  ∃ V : ℝ,
  let A := (0, 0) : ℝ × ℝ,
      B := (30, 0) : ℝ × ℝ,
      C := (15, 20) : ℝ × ℝ,
      D := ((B.1 + C.1) / 2, (B.2 + C.2) / 2) : ℝ × ℝ,
      E := ((C.1 + A.1) / 2, (C.2 + A.2) / 2) : ℝ × ℝ,
      F := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) : ℝ × ℝ,
      G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) : ℝ × ℝ,
      h := (2 / 3) * 10,
      area_ABC := (1/2) * abs(0 * (0 - 20) + 30 * (20 - 0) + 15 * (0 - 0)),
      V := (1 / 3) * area_ABC * h
  in V = 2000 / 3 :=
begin
  let A := (0, 0) : ℝ × ℝ,
  let B := (30, 0) : ℝ × ℝ,
  let C := (15, 20) : ℝ × ℝ,
  let D := ((B.1 + C.1) / 2, (B.2 + C.2) / 2) : ℝ × ℝ,
  let E := ((C.1 + A.1) / 2, (C.2 + A.2) / 2) : ℝ × ℝ,
  let F := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) : ℝ × ℝ,
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) : ℝ × ℝ,
  let h := (2 / 3) * 10,
  let area_ABC := (1 / 2) * (abs (0 * (0 - 20) + 30 * (20 - 0) + 15 * (0 - 0))),
  let V := (1 / 3) * area_ABC * h,
  use V,
  exact sorry,
end

end volume_of_pyramid_l285_285892


namespace kabadi_players_l285_285338

def people_play_kabadi (Kho_only Both Total : ℕ) : Prop :=
  ∃ K : ℕ, Kho_only = 20 ∧ Both = 5 ∧ Total = 30 ∧ K = Total - Kho_only ∧ (K + Both) = 15

theorem kabadi_players :
  people_play_kabadi 20 5 30 :=
by
  sorry

end kabadi_players_l285_285338


namespace polynomial_division_l285_285519

theorem polynomial_division :
  ∃ (q r : Polynomial ℚ), 
    (10 * Polynomial.X ^ 4 - 3 * Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 - Polynomial.X + 6 = 
    (3 * Polynomial.X + 4) * q + r ∧ degree r < degree (3 * Polynomial.X + 4) ∧
    q = (10 / 3) * Polynomial.X ^ 3 - (49 / 9) * Polynomial.X ^ 2 + (427 / 27) * Polynomial.X - (287 / 54) ∧
    r = 914 / 27) :=
begin
  sorry
end

end polynomial_division_l285_285519


namespace final_sale_price_l285_285459

def initial_price : ℝ := 450
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.10
def third_discount : ℝ := 0.05

def price_after_first_discount (initial : ℝ) (discount : ℝ) : ℝ :=
  initial * (1 - discount)
  
def price_after_second_discount (price_first : ℝ) (discount : ℝ) : ℝ :=
  price_first * (1 - discount)
  
def price_after_third_discount (price_second : ℝ) (discount : ℝ) : ℝ :=
  price_second * (1 - discount)

theorem final_sale_price :
  price_after_third_discount
    (price_after_second_discount
      (price_after_first_discount initial_price first_discount)
      second_discount)
    third_discount = 288.5625 := 
sorry

end final_sale_price_l285_285459


namespace least_common_multiple_l285_285396

theorem least_common_multiple (x : ℕ) (hx : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end least_common_multiple_l285_285396


namespace find_y_when_x_is_6_l285_285381

variable (x y : ℝ)
variable (h₁ : x > 0)
variable (h₂ : y > 0)
variable (k : ℝ)

axiom inverse_proportional : 3 * x^2 * y = k
axiom initial_condition : 3 * 3^2 * 30 = k

theorem find_y_when_x_is_6 (h : x = 6) : y = 7.5 :=
by
  sorry

end find_y_when_x_is_6_l285_285381


namespace initial_amount_approx_l285_285145

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem initial_amount_approx (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) (CI : ℝ) :
  P * (1 + r / n) ^ (n * t) = P + CI → 
  r = 0.15 → n = 1 → t = 2.33333̅ → CI = 3109 → P ≈ 7677.46 :=
by
  sorry

end initial_amount_approx_l285_285145


namespace geometric_sequence_ratio_l285_285949

theorem geometric_sequence_ratio :
  ∀ (a : ℝ), let r := -2 in
  let S_odd := a + a*r^2 + a*r^4 in
  let S_even := a*r + a*r^3 in
  S_odd / S_even = -21 / 10 :=
by
  intros a
  let r := -2
  let S_odd := a + a * r^2 + a * r^4
  let S_even := a * r + a * r^3
  sorry -- skipping the proof

end geometric_sequence_ratio_l285_285949


namespace machineA_produces_x_in_10_minutes_l285_285401

theorem machineA_produces_x_in_10_minutes
  (x : ℕ) 
  (B_rate_eq : (2 * x)/5) 
  (combined_rate_eq : (x/2)) 
  (together_production_eq : (5 * x)/10) : 
  ∃ T : ℕ, T = 10 :=
by {
  sorry
}

end machineA_produces_x_in_10_minutes_l285_285401


namespace cot20_tan10_eq_csc20_l285_285747

theorem cot20_tan10_eq_csc20 : 
  Real.cot 20 * (Real.pi / 180) + Real.tan 10 * (Real.pi / 180) = Real.csc 20 * (Real.pi / 180) :=
by sorry

end cot20_tan10_eq_csc20_l285_285747


namespace percentage_not_drop_l285_285652

def P_trip : ℝ := 0.40
def P_drop_given_trip : ℝ := 0.25
def P_not_drop : ℝ := 0.90

theorem percentage_not_drop :
  (1 - P_trip * P_drop_given_trip) = P_not_drop :=
sorry

end percentage_not_drop_l285_285652


namespace round_robin_cycles_l285_285633

-- We set up the conditions and the question as definitions.
def is_round_robin_tournament (teams : Finset ℕ) : Prop :=
  ∀ (t ∈ teams), t * 0 = 9 ∧ t * 1 = 9 ∧ t * 2 = 9

-- The theorem we need to prove: 
theorem round_robin_cycles (teams : Finset ℕ) (h : is_round_robin_tournament teams) : 
  ∑ (A ∈ teams) ∑ (B ∈ teams) ∑ (C ∈ teams), (if A ≠ B ∧ B ≠ C ∧ C ≠ A then
    (A, B, C) else 0) = 969 := 
sorry

end round_robin_cycles_l285_285633


namespace find_sine_sum_l285_285983

theorem find_sine_sum {A B C : ℝ} (a b c : ℝ) (h1 : a = 2 * b) (h2 : cos B = 2 * sqrt 2 / 3) :
  sin ((A - B) / 2) + sin (C / 2) = sqrt 10 / 3 := sorry

end find_sine_sum_l285_285983


namespace sandy_total_spent_l285_285327

def shorts_price : ℝ := 13.99
def shirt_price : ℝ := 12.14
def jacket_price : ℝ := 7.43
def total_spent : ℝ := shorts_price + shirt_price + jacket_price

theorem sandy_total_spent : total_spent = 33.56 :=
by
  sorry

end sandy_total_spent_l285_285327


namespace find_f_in_interval_l285_285759

variable f : ℝ → ℝ
variable (h1 : f 2 = 0)
variable (h2 : ∀ x, 0 ≤ x ∧ x < 2 → f x ≤ 0)
variable (h3 : ∀ x y, 0 < x → 0 < y → f (x * f y) * f y ≤ f (x + y))

theorem find_f_in_interval :
  ∀ x, 0 ≤ x → x < 2 → f x = 0 :=
by
  intro x hx1 hx2
  sorry

end find_f_in_interval_l285_285759


namespace region_area_equilateral_triangle_inscribed_circle_equals_l285_285009

-- Define the equilateral triangle and the inscribed circle with radius 1
noncomputable def side_length_of_inscribed_equilateral_triangle (r : ℝ) : ℝ :=
  2 * sqrt 3

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (s^2 * sqrt 3) / 4

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  real.pi * r^2

noncomputable def area_of_region_inside_triangle_but_outside_circle (s r : ℝ) : ℝ :=
  area_of_equilateral_triangle(s) - area_of_circle(r)

theorem region_area_equilateral_triangle_inscribed_circle_equals (r : ℝ) (s := side_length_of_inscribed_equilateral_triangle r) :
  area_of_region_inside_triangle_but_outside_circle(s, r) = 3 * sqrt 3 - real.pi :=
by
  sorry  -- Proof not required

end region_area_equilateral_triangle_inscribed_circle_equals_l285_285009


namespace factorial_division_l285_285502

open Nat

theorem factorial_division : 12! / 11! = 12 := sorry

end factorial_division_l285_285502


namespace arithmetic_geometric_sequence_general_term_l285_285772

theorem arithmetic_geometric_sequence_general_term :
  ∃ q a1 : ℕ, (∀ n : ℕ, a2 = 6 ∧ 6 * a1 + a3 = 30) →
  (∀ n : ℕ, (q = 2 ∧ a1 = 3 → a_n = 3 * 3^(n-1)) ∨ (q = 3 ∧ a1 = 2 → a_n = 2 * 2^(n-1))) :=
by
  sorry

end arithmetic_geometric_sequence_general_term_l285_285772


namespace cubic_roots_sum_cubes_l285_285676

theorem cubic_roots_sum_cubes
  (p q r : ℂ)
  (h_eq_root : ∀ x, x = p ∨ x = q ∨ x = r → x^3 - 2 * x^2 + 3 * x - 1 = 0)
  (h_sum : p + q + r = 2)
  (h_prod_sum : p * q + q * r + r * p = 3)
  (h_prod : p * q * r = 1) :
  p^3 + q^3 + r^3 = -7 := by
  sorry

end cubic_roots_sum_cubes_l285_285676


namespace find_x_plus_y_l285_285548

theorem find_x_plus_y (x y : ℝ) (hx : |x| = 3) (hy : |y| = 2) (hxy : |x - y| = y - x) :
  (x + y = -1) ∨ (x + y = -5) :=
sorry

end find_x_plus_y_l285_285548


namespace min_distance_curve_to_line_l285_285250

theorem min_distance_curve_to_line {x : ℝ} (hx : x > 0) :
  let P := (x, x^2 - Real.log x),
      L := λ x : ℝ, x - 2 in
      dist P (P.1, L P.1) = 1 :=
begin
  sorry
end

end min_distance_curve_to_line_l285_285250


namespace find_peters_magnets_l285_285092

namespace MagnetProblem

variable (AdamInitial Peter : ℕ) (AdamRemaining : ℕ)

axiom Adam_initial_has_18_magnets : AdamInitial = 18
axiom Adam_gives_away_third : AdamRemaining = AdamInitial - (AdamInitial / 3)
axiom Adam_has_half_of_Peter : AdamRemaining = Peter / 2

theorem find_peters_magnets : Peter = 24 :=
by
  have h1 : AdamRemaining = 18 - (18 / 3) := by rw [Adam_initial_has_18_magnets, nat.div_eq_of_eq_mul_right (by decide : 3 > 0) rfl]
  have h2 : AdamRemaining = 18 - 6 := by rw [h1]
  have h3 : AdamRemaining = 12 := by norm_num at h2
  have h4 : 12 = Peter / 2 := by rw [Adam_has_half_of_Peter, h3]
  have h5 : Peter = 24 := by linarith [h4]
  exact h5

end find_peters_magnets_l285_285092


namespace linear_A_not_linear_B_not_linear_C_not_linear_D_l285_285821

def linear (f : ℝ → ℝ) : Prop :=
  ∃ m b, ∀ x, f x = m * x + b

def f_A (x : ℝ) : ℝ := -2 * x
def f_B (x : ℝ) : ℝ := -x^2 + x
def f_C (x : ℝ) : ℝ := -(1 / x)
def f_D (x : ℝ) : ℝ := Real.sqrt x + 1

theorem linear_A : linear f_A :=
sorry

theorem not_linear_B : ¬ linear f_B :=
sorry

theorem not_linear_C : ¬ linear f_C :=
sorry

theorem not_linear_D : ¬ linear f_D :=
sorry

end linear_A_not_linear_B_not_linear_C_not_linear_D_l285_285821


namespace number_of_distinct_sequences_eq_catalan_l285_285827

theorem number_of_distinct_sequences_eq_catalan (n : ℕ) :
  let a : ℕ → ℤ := λ i, if i > 0 then -1 else 1 in
  (∃ (s : ℕ → ℤ), 
    (∀ i, s i = 1 ∨ s i = -1) ∧
    (s 0 = 1) ∧
    (∀ k, (∑ i in range k, s i) ≥ 0) ∧
    (∑ i in range (2 * n), s i = 0)
  ) = (Catalan.nth n) :=
by
  sorry

end number_of_distinct_sequences_eq_catalan_l285_285827


namespace infinite_triples_of_distinct_natural_numbers_with_same_P_l285_285950

theorem infinite_triples_of_distinct_natural_numbers_with_same_P :
  ∃ (f : ℕ → ℕ), function.injective f ∧ 
  ∀ a b c, f a ≠ f b → f b ≠ f c → f a ≠ f c → 
  let P (n : ℕ) := nat.greatest_prime_divisor (n^2 + 1)
  in P (f a) = P (f b) ∧ P (f b) = P (f c) :=
sorry

end infinite_triples_of_distinct_natural_numbers_with_same_P_l285_285950


namespace matrix_det_l285_285135

theorem matrix_det (x : ℝ) : 
  let A := matrix.of ![![2*x + 2, 2*x, 2*x], ![2*x, 2*x + 2, 2*x], ![2*x, 2*x, 2*x + 2]] in
  matrix.det A = 20*x + 8 :=
by sorry

end matrix_det_l285_285135


namespace perimeter_of_ABCD_l285_285421

theorem perimeter_of_ABCD (AE BE CF : ℕ) (hAE : AE = 6) (hBE : BE = 15) (hCF : CF = 5) : 
  ∃ (m n : ℕ), (m + n = 808) ∧ (nat.gcd m n = 1) ∧  (P = m / n) :=
by
  -- Definitions and conditions
  let P := 2 * (nat.sqrt (261) + 143 / nat.sqrt (261))
  -- P is the perimeter of the rectangle ABCD given AE = 6, BE = 15, and CF = 5
  -- We claim m=808 and n=1
  let m := 808
  let n := 1
  have hm : nat.gcd m n = 1 := by sorry  -- gcd calculation
  existsi m  -- ∃ m
  existsi n  -- ∃ n
  split  -- Prove the conjunction
  -- Check the sum m+n
  calc m + n 
      = 808 + 1 
      = 808 : by rfl
  -- Check the gcd
  calc nat.gcd m n 
      = 1 : by sorry
  -- Check the equivalence with perimeter P
  calc P = m / n : by sorry

end perimeter_of_ABCD_l285_285421


namespace score_of_58_is_2_stdevs_below_l285_285162

theorem score_of_58_is_2_stdevs_below :
  ∃ σ : ℝ, (98 = 74 + 3 * σ) ∧ (58 = 74 - 2 * σ) :=
by
  use 8
  split
  · exact by linarith
  · exact by linarith
  sorry

end score_of_58_is_2_stdevs_below_l285_285162


namespace find_z_l285_285608

noncomputable def w : ℝ := sorry
noncomputable def x : ℝ := (5 * w) / 4
noncomputable def y : ℝ := 1.40 * w

theorem find_z (z : ℝ) : x = (1 - z / 100) * y → z = 10.71 :=
by
  sorry

end find_z_l285_285608


namespace jaydee_typing_speed_l285_285281

theorem jaydee_typing_speed (hours : ℕ) (total_words : ℕ) (minutes_per_hour : ℕ := 60) 
  (h1 : hours = 2) (h2 : total_words = 4560) : (total_words / (hours * minutes_per_hour) = 38) :=
by
  sorry

end jaydee_typing_speed_l285_285281


namespace flower_bouquets_l285_285082

theorem flower_bouquets (r t : ℕ) (h : 4 * r + 3 * t = 60) : r ∈ {0, 3, 6, 9, 12, 15} → ∃ n : ℕ, n = 6 :=
by {
  assume hr_values,
  -- proof would go here
  sorry
}

end flower_bouquets_l285_285082


namespace fund_unclaimed_fraction_is_zero_l285_285133

theorem fund_unclaimed_fraction_is_zero (T : ℝ) (hT : T > 0) :
  let dina_share := (4 / (4 + 3 + 1)) * T,
      eva_share := (3 / (4 + 3 + 1)) * T,
      frank_share := (1 / (4 + 3 + 1)) * T,
      remaining_after_dina := T - dina_share,
      remaining_after_eva := remaining_after_dina - eva_share,
      remaining_after_frank := remaining_after_eva - frank_share
  in remaining_after_frank = 0 := sorry

end fund_unclaimed_fraction_is_zero_l285_285133


namespace arithmetic_series_sum_l285_285480

theorem arithmetic_series_sum :
  let a1 : ℚ := 22
  let d : ℚ := 3 / 7
  let an : ℚ := 73
  let n := (an - a1) / d + 1
  let S := n * (a1 + an) / 2
  S = 5700 := by
  sorry

end arithmetic_series_sum_l285_285480


namespace calculate_triple_transform_l285_285899

def transformation (N : ℝ) : ℝ :=
  0.4 * N + 2

theorem calculate_triple_transform :
  transformation (transformation (transformation 20)) = 4.4 :=
by
  sorry

end calculate_triple_transform_l285_285899


namespace finish_order_l285_285763

-- Definitions for the consumption rates
def Черныш_rate := 2  -- Черныш eats 2 sausages per time period T
def Тиграша_rate := 5 -- Тиграша eats 5 sausages per time period T
def Снежок_rate := 3  -- Снежок eats 3 sausages per time period T
def Пушок_rate := 4   -- Пушок eats 4 sausages per time period T

-- Total time period between the two photographs
variables (T : ℝ) (initial_sausages : ℝ)

-- Lean statement for the proof problem
theorem finish_order (h1 : initial_sausages > 0) : 
  -- Determine who finishes first and who finishes last
  (Тиграша_rate * T < Пушок_rate * T ∧ Тиграша_rate * T < Черныш_rate * T ∧ Тиграша_rate * T < Снежок_rate * T) ∧ 
  (Снежок_rate * T > Пушок_rate * T ∧ Снежок_rate * T > Черныш_rate * T ∧ Снежок_rate * T > Тиграша_rate * T) :=
by
  sorry

end finish_order_l285_285763


namespace common_tangent_circles_l285_285635

variables {A B C D M : Type*}
variables [EuclideanGeometry A B C D M]

-- Defining the given parallelogram ABCD
def is_parallelogram (A B C D : Point) : Prop := 
  parallelogram A B C D

-- Definitions of diagonals and intersection point
def diagonal_AC_longer_BD (A C B D : Point) [AC_length : longer_diagonal (segment A C) (segment B D)] := true

-- Circle passes through B, C, D and intersects AC at M
def circle_through_BCD (M B C D : Point) (circle : circle B C D) [intersects_diagonal (circle M) (segment A C)] := true

-- Proof problem statement
theorem common_tangent_circles {A B C D M : Point} [p : parallelogram A B C D] [diagonal_AC_longer_BD A C B D] [circle_through_BCD M B C D circle] :
  tangent (line B D) (circumcircle_triangles A M B) ∧ tangent (line B D) (circumcircle_triangles A M D) :=
by
  sorry

end common_tangent_circles_l285_285635


namespace max_value_seq_l285_285556

theorem max_value_seq : 
  ∃ a : ℕ → ℝ, 
    a 1 = 1 ∧ 
    a 2 = 4 ∧ 
    (∀ n ≥ 2, 2 * a n = (n - 1) / n * a (n - 1) + (n + 1) / n * a (n + 1)) ∧ 
    ∀ n : ℕ, n > 0 → 
      ∃ m : ℕ, m > 0 ∧ 
        ∀ k : ℕ, k > 0 → (a k) / k ≤ 2 ∧ (a 2) / 2 = 2 :=
sorry

end max_value_seq_l285_285556


namespace sculpture_and_base_height_l285_285052

theorem sculpture_and_base_height :
  let sculpture_height_in_feet := 2
  let sculpture_height_in_inches := 10
  let base_height_in_inches := 2
  let total_height_in_inches := (sculpture_height_in_feet * 12) + sculpture_height_in_inches + base_height_in_inches
  let total_height_in_feet := total_height_in_inches / 12
  total_height_in_feet = 3 :=
by
  sorry

end sculpture_and_base_height_l285_285052


namespace cot_tan_simplify_l285_285741

theorem cot_tan_simplify : cot (20 * π / 180) + tan (10 * π / 180) = csc (20 * π / 180) :=
by
  sorry

end cot_tan_simplify_l285_285741


namespace sin_squared_sum_l285_285112

theorem sin_squared_sum : (∑ k in Finset.range 36, (Real.sin ((k + 1) * 5 * Real.pi / 180))^2) = 18.5 :=
by
  sorry

end sin_squared_sum_l285_285112


namespace arcsin_sqrt2_over_2_eq_pi_over_4_l285_285110

theorem arcsin_sqrt2_over_2_eq_pi_over_4 :
  Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
sorry

end arcsin_sqrt2_over_2_eq_pi_over_4_l285_285110


namespace min_subsets_not_prime_diff_l285_285809

theorem min_subsets_not_prime_diff : ∃ n : ℕ, n ≤ 4 ∧ (∀ (S : Fin n → Set ℤ), (∀ (i : Fin n), ∀ (x y : ℤ), x ∈ S i → y ∈ S i → x ≠ y → ¬ is_prime (x - y))) :=
begin
  sorry,
end

end min_subsets_not_prime_diff_l285_285809


namespace quadrilateral_inscribed_in_semicircle_l285_285705

/-
Given:
1. Quadrilateral XABY is inscribed in the semicircle ω with diameter XY.
2. Segments AY and BX meet at P.
3. Point Z is the foot of the perpendicular from P to line XY.
4. Point C lies on ω such that line XC is perpendicular to line AZ.
5. Q is the intersection of segments AY and XC.
Prove: (BY / XP) + (CY / XQ) = (AY / AX).
-/

theorem quadrilateral_inscribed_in_semicircle
  (X A B Y P Z C Q : Type)
  (ω : set (X × Y))
  (diameter_XY : ∀ (x y : X × Y), x ∈ ω → y ∈ ω → (fst x - fst y)^2 + (snd x - snd y)^2 = 1)
  (inscribed_XABY : (X, A) ∈ ω ∧ (A, B) ∈ ω ∧ (B, Y) ∈ ω)
  (meet_P : ∃ R, R ∈ line(A, Y) ∧ R ∈ line(B, X) ∧ R = P)
  (foot_Z : ∃ foot, is_perpendicular foot P (line(X, Y)) ∧ foot = Z)
  (on_circle_C : (X, C) ∈ ω)
  (perp_XC_AZ : is_perpendicular (line(X, C)) (line(A, Z)))
  (intersection_Q : Q ∈ segment(A, Y) ∧ Q ∈ segment(X, C)):
  BY / XP + CY / XQ = AY / AX :=
sorry

end quadrilateral_inscribed_in_semicircle_l285_285705


namespace range_of_x_l285_285198

open Real

theorem range_of_x (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 / a + 1 / b = 1) :
  a + b ≥ 3 + 2 * sqrt 2 :=
sorry

end range_of_x_l285_285198


namespace lcm_18_30_l285_285021

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l285_285021


namespace lcm_18_30_l285_285022

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l285_285022


namespace proof_P_Q_R_S_P_Q_2_l285_285662

def P (x : ℝ) := 3 * Real.sqrt x
def Q (x : ℝ) := x^3
def R (x : ℝ) := x + 2
def S (x : ℝ) := 2 * x

theorem proof_P_Q_R_S_P_Q_2 :
  P (Q (R (S (P (Q 2))))) = 3 * Real.sqrt ((12 * Real.sqrt 2 + 2)^3) := by
  sorry

end proof_P_Q_R_S_P_Q_2_l285_285662


namespace project_time_for_A_l285_285063

/--
A can complete a project in some days and B can complete the same project in 30 days.
If A and B start working on the project together and A quits 5 days before the project is 
completed, the project will be completed in 15 days.
Prove that A can complete the project alone in 20 days.
-/
theorem project_time_for_A (x : ℕ) (h : 10 * (1 / x + 1 / 30) + 5 * (1 / 30) = 1) : x = 20 :=
sorry

end project_time_for_A_l285_285063


namespace expand_polynomial_l285_285920

theorem expand_polynomial (x : ℝ) :
  (x + 4) * (x - 9) = x^2 - 5 * x - 36 := 
sorry

end expand_polynomial_l285_285920


namespace simplify_cot_tan_l285_285736

theorem simplify_cot_tan : Real.cot (Real.pi / 9) + Real.tan (Real.pi / 18) = Real.csc (Real.pi / 9) := 
by
  sorry

end simplify_cot_tan_l285_285736


namespace longest_diagonal_is_20_l285_285450

noncomputable def length_of_longest_diagonal (area : ℝ) (ratio1 ratio2 : ℝ) : ℝ :=
  let x := (area * 2) / (ratio1 * ratio2)
  in ratio1 * (x / 6) -- as we derived 150 = 6 * x^2, leading to x/6 part

theorem longest_diagonal_is_20 :
  length_of_longest_diagonal 150 4 3 = 20 := by
{
  -- specify the structure of the proof and then leave it as a sorry
  sorry
}

end longest_diagonal_is_20_l285_285450


namespace cistern_filling_time_l285_285050

/-- Define the rates at which the cistern is filled and emptied -/
def fill_rate := (1 : ℚ) / 3
def empty_rate := (1 : ℚ) / 8

/-- Define the net rate of filling when both taps are open -/
def net_rate := fill_rate - empty_rate

/-- Define the volume of the cistern -/
def cistern_volume := (1 : ℚ)

/-- Compute the time to fill the cistern given the net rate -/
def fill_time := cistern_volume / net_rate

theorem cistern_filling_time :
  fill_time = 4.8 := by
sorry

end cistern_filling_time_l285_285050


namespace find_lambda_perpendicular_l285_285224

theorem find_lambda_perpendicular (a b : ℝ × ℝ) (h1 : a = (1, -3)) (h2 : b = (4, -2)) :
  ∃ λ : ℝ, (λ * a.1 + b.1, λ * a.2 + b.2) • a = 0 ↔ λ = -1 :=
by {
  let v := (λ * a.1 + b.1, λ * a.2 + b.2),
  rw dot_product_eq_zero,
  sorry
}

end find_lambda_perpendicular_l285_285224


namespace determinant_of_matrixA_l285_285912

variable (x : ℝ)

def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x + 2, x, x], ![x, x + 2, x], ![x, x, x + 2]]

theorem determinant_of_matrixA : Matrix.det (matrixA x) = 8 * x + 8 := by
  sorry

end determinant_of_matrixA_l285_285912


namespace ratio_a_d_l285_285407

theorem ratio_a_d 
  (a b c d : ℕ) 
  (h1 : a / b = 1 / 4) 
  (h2 : b / c = 13 / 9) 
  (h3 : c / d = 5 / 13) : 
  a / d = 5 / 36 :=
sorry

end ratio_a_d_l285_285407


namespace transformed_function_is_g_l285_285584

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) - 2 * (sin x) ^ 2 + 1

noncomputable def g (x : ℝ) : ℝ := sqrt 2 * sin (4 * x - 3 * Real.pi / 4)

theorem transformed_function_is_g :
  ∀ x : ℝ, g x = sqrt 2 * sin (4 * (x - Real.pi / 4) + Real.pi / 4) :=
sorry

end transformed_function_is_g_l285_285584


namespace gas_pipe_probability_l285_285070

-- Define the problem statement in Lean.
theorem gas_pipe_probability :
  let total_area := 400 * 400 / 2
  let usable_area := (300 - 100) * (300 - 100) / 2
  usable_area / total_area = 1 / 4 :=
by
  -- Sorry will be placeholder for the proof
  sorry

end gas_pipe_probability_l285_285070


namespace arrangement_impossible_l285_285277

noncomputable def impossible_arrangement : Prop :=
  ∀ (a b : Fin 1987 → ℕ), 
    (∀ k, b k - a k = k + 1) →
    ¬ ∃ (perm : List (Fin 1987)), 
      (∀ k, perm.indexOf a k < perm.indexOf b k) ∧ 
      (∀ k, perm.indexOf b k - perm.indexOf a k = k + 1)

theorem arrangement_impossible : impossible_arrangement :=
sorry

end arrangement_impossible_l285_285277


namespace Euler_line_property_l285_285838

theorem Euler_line_property (ABC : Triangle)
  (H : Point) (O : Point) (M : Point) 
  (hH : is_orthocenter H ABC) 
  (hO : is_circumcenter O ABC) 
  (hM : is_centroid M ABC) :
  collinear H O M ∧ segment_len M H = 2 * segment_len M O ∧ 
  between M O H := 
sorry

end Euler_line_property_l285_285838


namespace lcm_18_30_is_90_l285_285031

noncomputable def LCM_of_18_and_30 : ℕ := 90

theorem lcm_18_30_is_90 : nat.lcm 18 30 = LCM_of_18_and_30 := 
by 
  -- definition of lcm should be used here eventually
  sorry

end lcm_18_30_is_90_l285_285031


namespace find_value_of_t_l285_285598

def vector_collinear (v w : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, c • v = w

theorem find_value_of_t (t : ℝ) :
  let m := (sqrt 3, 1)
  let n := (0, -1)
  let k := (t, sqrt 3)
  vector_collinear (m.1 - 2 * n.1, m.2 - 2 * n.2) k →
  t = 1 :=
by
  sorry

end find_value_of_t_l285_285598


namespace find_y_l285_285411

theorem find_y (x y : ℤ)
  (h1 : (100 + 200300 + x) / 3 = 250)
  (h2 : (300 + 150100 + x + y) / 4 = 200) :
  y = -4250 :=
sorry

end find_y_l285_285411


namespace cot20_tan10_eq_csc20_l285_285744

theorem cot20_tan10_eq_csc20 : 
  Real.cot 20 * (Real.pi / 180) + Real.tan 10 * (Real.pi / 180) = Real.csc 20 * (Real.pi / 180) :=
by sorry

end cot20_tan10_eq_csc20_l285_285744


namespace sum_of_angles_in_triangles_l285_285444

-- Define a generic quadrilateral with conditions on the diagonals
structure Quadrilateral :=
  (A B C D : Type)    -- Vertices of the quadrilateral
  (diagonal1 diagonal2 : A → B → C → D → Prop) -- Properties of the diagonals
  (intersection : A × B → C × D → Prop) -- Property of the intersection point of diagonals
  (no_side_through_intersection : ¬ ∃ (P : A), intersection (A, B) (C, D)) -- No side passes through the intersection

-- Sum of internal angles of a quadrilateral
def sum_of_internal_angles (q : Quadrilateral) : ℝ := 360

-- Additional angle contribution due to diagonal intersection
def angle_contributions_by_diagonal_intersection (q : Quadrilateral) : ℝ := 360

-- Total sum of angles in the quadrilateral divided into four triangles
def total_angle_sum (q : Quadrilateral) : ℝ :=
  sum_of_internal_angles q + angle_contributions_by_diagonal_intersection q

-- Theorem to be proven
theorem sum_of_angles_in_triangles (q : Quadrilateral) : total_angle_sum q = 720 := by
  sorry

end sum_of_angles_in_triangles_l285_285444


namespace slope_angle_tangent_is_pi_over_4_l285_285995

def f (x : ℝ) : ℝ := x^2 - 2*x

noncomputable def slope_angle_tangent_at (x : ℝ) : ℝ := 
  let slope := 2 * x - 2 in
  if slope = 1 then real.atan(1) else sorry

-- Statement of the problem: we need to show that the slope angle is π/4 when x = 3/2
theorem slope_angle_tangent_is_pi_over_4 :
  slope_angle_tangent_at (3/2) = π / 4 :=
by 
  sorry

end slope_angle_tangent_is_pi_over_4_l285_285995


namespace distinct_values_z_l285_285208

theorem distinct_values_z :
  ∃ (S : Finset ℕ), (∀ (x y : ℕ), (x < 1000 → y = (100 * (x % 10) + 10 * ((x / 10) % 10) + x / 100) → |x - y| ∈ S)) ∧ S.card = 10 :=
sorry

end distinct_values_z_l285_285208


namespace matrix_power_is_correct_l285_285491

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]
def A_cubed : Matrix (Fin 2) (Fin 2) ℤ := !![3, -6; 6, -3]

theorem matrix_power_is_correct : A ^ 3 = A_cubed := by 
  sorry

end matrix_power_is_correct_l285_285491


namespace expand_binomials_l285_285922

theorem expand_binomials : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by 
  sorry

end expand_binomials_l285_285922


namespace reduced_price_after_exchange_rate_fluctuation_l285_285071

-- Definitions based on conditions
variables (P : ℝ) -- Original price per kg

def reduced_price_per_kg : ℝ := 0.9 * P

axiom six_kg_costs_900 : 6 * reduced_price_per_kg P = 900

-- Additional conditions
def exchange_rate_factor : ℝ := 1.02

-- Question restated as the theorem to prove
theorem reduced_price_after_exchange_rate_fluctuation : 
  ∃ P : ℝ, reduced_price_per_kg P * exchange_rate_factor = 153 :=
sorry

end reduced_price_after_exchange_rate_fluctuation_l285_285071


namespace percentage_emails_moved_to_work_folder_l285_285339

def initialEmails : ℕ := 400
def trashedEmails : ℕ := initialEmails / 2
def remainingEmailsAfterTrash : ℕ := initialEmails - trashedEmails
def emailsLeftInInbox : ℕ := 120
def emailsMovedToWorkFolder : ℕ := remainingEmailsAfterTrash - emailsLeftInInbox

theorem percentage_emails_moved_to_work_folder :
  (emailsMovedToWorkFolder * 100 / remainingEmailsAfterTrash) = 40 := by
  sorry

end percentage_emails_moved_to_work_folder_l285_285339


namespace find_weight_of_first_new_player_l285_285792

variable (weight_of_first_new_player : ℕ)
variable (weight_of_second_new_player : ℕ := 60) -- Second new player's weight is a given constant
variable (num_of_original_players : ℕ := 7)
variable (avg_weight_of_original_players : ℕ := 121)
variable (new_avg_weight : ℕ := 113)
variable (num_of_new_players : ℕ := 2)

def total_weight_of_original_players : ℕ := 
  num_of_original_players * avg_weight_of_original_players

def total_weight_of_new_players : ℕ :=
  num_of_new_players * new_avg_weight

def combined_weight_without_first_new_player : ℕ := 
  total_weight_of_original_players + weight_of_second_new_player

def weight_of_first_new_player_proven : Prop :=
  total_weight_of_new_players - combined_weight_without_first_new_player = weight_of_first_new_player

theorem find_weight_of_first_new_player : weight_of_first_new_player = 110 :=
by 
  sorry

end find_weight_of_first_new_player_l285_285792


namespace number_of_solutions_of_sine_eq_third_to_x_l285_285153

open Real

theorem number_of_solutions_of_sine_eq_third_to_x : 
  (set.Icc 0 (200 * π)).countable.count (λ x, sin x = (1/3)^x) = 200 :=
by
  sorry

end number_of_solutions_of_sine_eq_third_to_x_l285_285153


namespace Peter_magnets_l285_285094

theorem Peter_magnets (initial_magnets_Adam : ℕ) (given_away_fraction : ℚ) (half_magnets_factor : ℚ) 
  (Adam_left : ℕ) (Peter_magnets : ℕ) :
  initial_magnets_Adam = 18 →
  given_away_fraction = 1/3 →
  initial_magnets_Adam * (1 - given_away_fraction) = Adam_left →
  Adam_left = half_magnets_factor * Peter_magnets →
  half_magnets_factor = 1/2 →
  Peter_magnets = 24 :=
by
  -- Proof goes here
  sorry

-- Providing the values for conditions
#eval Peter_magnets 18 (1/3) (1/2) 12 24

end Peter_magnets_l285_285094


namespace midpoint_theorem_l285_285318

/-- Define midpoint operations --/
def midpoint (A B : ℝ) : ℝ := (A + B) / 2

theorem midpoint_theorem : ∀ (A B C D E F G : ℝ), 
  C = midpoint A B →
  D = midpoint A C →
  E = midpoint A D →
  F = midpoint A E →
  G = midpoint A F →
  A - G = 4 →
  A - B = 128 :=
by
  intros A B C D E F G hC_midpoint hD_midpoint hE_midpoint hF_midpoint hG_midpoint hAG 
  sorry

end midpoint_theorem_l285_285318


namespace Jake_not_drop_coffee_l285_285653

theorem Jake_not_drop_coffee :
  (40% / 100) * (25% / 100) = 10% / 100 → 
  100% / 100 - 10% / 100 = 90% / 100 :=
begin
  sorry
end

end Jake_not_drop_coffee_l285_285653


namespace monday_rainfall_l285_285278

theorem monday_rainfall (tuesday_rainfall monday_rainfall: ℝ) 
(less_rain: ℝ) (h1: tuesday_rainfall = 0.2) 
(h2: less_rain = 0.7) 
(h3: tuesday_rainfall = monday_rainfall - less_rain): 
monday_rainfall = 0.9 :=
by sorry

end monday_rainfall_l285_285278


namespace matrix_cube_l285_285492

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by
  sorry

end matrix_cube_l285_285492


namespace car_travel_distance_per_liter_l285_285432

theorem car_travel_distance_per_liter
    (gallons_consumed : ℝ) (hours : ℝ) (speed_mph : ℝ)
    (gallon_to_liter : ℝ) (mile_to_km : ℝ)
    (h_gallons : gallons_consumed = 3.9)
    (h_hours : hours = 5.7)
    (h_speed : speed_mph = 104)
    (h_g2l : gallon_to_liter = 3.8)
    (h_m2k : mile_to_km = 1.6) :
    (104 * 5.7 * 1.6) / (3.9 * 3.8) ≈ 64.01 := by
    sorry

end car_travel_distance_per_liter_l285_285432


namespace mean_of_combined_sets_l285_285699

theorem mean_of_combined_sets 
  (mean1 mean2 mean3 : ℚ)
  (count1 count2 count3 : ℕ)
  (h1 : mean1 = 15)
  (h2 : mean2 = 20)
  (h3 : mean3 = 12)
  (hc1 : count1 = 7)
  (hc2 : count2 = 8)
  (hc3 : count3 = 5) :
  ((count1 * mean1 + count2 * mean2 + count3 * mean3) / (count1 + count2 + count3)) = 16.25 :=
by
  sorry

end mean_of_combined_sets_l285_285699


namespace find_p_q_sum_l285_285467

-- Define the equation of the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 + 9*y^2 = 9

-- Define the conditions
def is_equilateral_triangle (A B C : (ℝ × ℝ)) : Prop :=
  let (Ax, Ay) := A;
  let (Bx, By) := B;
  let (Cx, Cy) := C;
  (Ax = 0 ∧ Ay = 1) ∧
  ((Bx = -Cx ∧ By = Cy) ∧ 
   (Bx ≠ 0 ∧ Cy ≠ 1) ∧ 
   Ax^2 + 9*Ay^2 = 9 ∧
   Bx^2 + 9*By^2 = 9 ∧
   Cx^2 + 9*Cy^2 = 9 ∧
   -- Condition for the triangle to be equilateral
   ((Bx - Ax)^2 + (By - Ay)^2 = (Cx - Bx)^2 + (Cy - By)^2) ∧
   ((Cx - Ax)^2 + (Cy - Ay)^2 = (Bx - Ax)^2 + (By - Ay)^2))

-- Define that the length of the side squared is p/q
def length_squared_is_p_div_q (A B : (ℝ × ℝ)) (p q : ℕ) : Prop :=
  let (Ax, Ay) := A;
  let (Bx, By) := B;
  (Ax - Bx)^2 + (Ay - By)^2 = (p : ℝ) / (q : ℝ) ∧ Nat.gcd p q = 1 

-- The main statement to be proven 
theorem find_p_q_sum (A B C : (ℝ × ℝ)) (p q : ℕ) :
  is_equilateral_triangle A B C →
  length_squared_is_p_div_q A B p q →
  p + q = 292 :=
by
  intro h_triangle h_length
  sorry

end find_p_q_sum_l285_285467


namespace eval_expression_l285_285132

theorem eval_expression : |(-3 : ℤ)| ^ 0 + (-8 : ℤ) ^ (1 / 3 : ℚ) = -1 := by
  sorry

end eval_expression_l285_285132


namespace range_of_omega_l285_285216

-- Define the conditions and the function
def f (ω x : ℝ) : ℝ :=
  sin (ω * x / 2) ^ 2 + 1 / 2 * sin (ω * x) + 1 / 2

-- The main theorem to prove the range of ω
theorem range_of_omega (ω : ℝ) : 
  (∀ x, 0 < x ∧ x < π → f ω x ≠ 0) ↔ -3/4 ≤ ω ∧ ω ≤ 1/4 :=
by {
  sorry
}

end range_of_omega_l285_285216


namespace shortest_path_on_tetrahedron_l285_285469

-- Definitions corresponding to the problem's conditions
def regular_tetrahedron (edge_length : ℝ) : Prop :=
∀ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → ∀ e f g : ℝ, e = edge_length → f = edge_length → g = edge_length

def midpoint (a b : ℝ) : ℝ := (a + b) / 2

def opposite_edges (a b c d : ℝ) : Prop := a ≠ c ∧ b ≠ d ∧ a ≠ d ∧ b ≠ c

-- Theorem statement for proving the shortest path
theorem shortest_path_on_tetrahedron
  (a b c d : ℝ)
  (edge_length : ℝ)
  (h_tetra : regular_tetrahedron edge_length)
  (h_opposite : opposite_edges a b c d) :
  ∀ (p q : ℝ), p = midpoint a b → q = midpoint c d → dist p q = edge_length :=
by
  sorry

end shortest_path_on_tetrahedron_l285_285469


namespace sqrt5_lt_sqrt2_plus_1_l285_285109

theorem sqrt5_lt_sqrt2_plus_1 : Real.sqrt 5 < Real.sqrt 2 + 1 :=
sorry

end sqrt5_lt_sqrt2_plus_1_l285_285109


namespace find_y_l285_285399

theorem find_y (x y : ℕ) (h_pos_y : 0 < y) (h_rem : x % y = 7) (h_div : x = 86 * y + (1 / 10) * y) :
  y = 70 :=
sorry

end find_y_l285_285399


namespace functional_equation_solution_l285_285140

-- Define ℕ* (positive integers) as a subtype of ℕ
def Nat.star := {n : ℕ // n > 0}

-- Define the problem statement
theorem functional_equation_solution (f : Nat.star → Nat.star) :
  (∀ m n : Nat.star, m.val ^ 2 + (f n).val ∣ m.val * (f m).val + n.val) →
  (∀ n : Nat.star, f n = n) :=
by
  intro h
  sorry

end functional_equation_solution_l285_285140


namespace distance_between_circles_l285_285366

noncomputable def isosceles_triangle_distance (h α : ℝ) (h_alpha : α ≤ (Real.pi / 6)) : ℝ :=
  have cos_part := Real.cos (π / 4 + 3 * α / 2)
  have cos_alpha_sq := Real.cos α ^ 2
  have tan_factor := Real.cos (π / 4 - α / 2)
  h * cos_part / (2 * cos_alpha_sq * tan_factor)


theorem distance_between_circles (h α : ℝ) (h_alpha : α ≤ Real.pi / 6) :
  isosceles_triangle_distance h α h_alpha = h * Real.cos (π / 4 + 3 * α / 2) / (2 * Real.cos α ^ 2 * Real.cos (π / 4 - α / 2)) :=
by 
  sorry

end distance_between_circles_l285_285366


namespace problem1_problem2_l285_285336

theorem problem1 (x : ℝ) (h : x + 1 > 2x - 3) : x < 4 :=
sorry

theorem problem2 (x : ℝ) (h1 : 2x - 1 > x) (h2 : (x + 5)/2 - x ≥ 1) : 1 < x ∧ x ≤ 3 :=
sorry

end problem1_problem2_l285_285336


namespace triangle_perimeter_l285_285561

variable (F1 F2 M N : Point)
variable (a b : ℝ) (h : a^2 = 3) (k : b^2 = 4)

def isEllipse (P1 P2 : Point) (a b : ℝ) :=
  ∀ (P : Point), |distance P1 P + distance P P2| = 2 * a

theorem triangle_perimeter (h1 : isEllipse F1 F2 3 4) 
    (h2 : distance M F1 + distance M F2 = 4) 
    (h3 : distance N F1 + distance N F2 = 4) 
    : perimeter (triangle F2 M N) = 8 := 
by
  sorry

end triangle_perimeter_l285_285561


namespace largest_polygon_perimeter_l285_285385

theorem largest_polygon_perimeter : 
  ∀ (polygons : list (nat × ℝ)), 
    polygons.length = 3 →
    polygons.all (λ p, (p.2 = 1)) →
    let square := polygons.any (λ p, p.1 = 4) in
    let others := polygons.filter (λ p, p.1 ≠ 4) in
    others.length = 2 →
    others.all (λ p, p.1 = 8) →
    (others.length * 6 + 3 = 15) :=
by
  intro polygons h1 h2 h3 h4 h5
  sorry

end largest_polygon_perimeter_l285_285385


namespace units_digit_periodic_10_l285_285324

theorem units_digit_periodic_10:
  ∀ n: ℕ, (n * (n + 1) * (n + 2)) % 10 = ((n + 10) * (n + 11) * (n + 12)) % 10 :=
by
  sorry

end units_digit_periodic_10_l285_285324


namespace time_difference_beijing_new_york_time_new_york_when_xiao_ming_calls_time_in_sydney_after_flight_beijing_double_moscow_l285_285262

theorem time_difference_beijing_new_york : 
  ∀ (beijing new_york : Int), 
  beijing = 8 ∧ new_york = -4 → beijing - new_york = 12 := 
by
  intros beijing new_york h
  cases h with h_beijing h_new_york
  rw [h_beijing, h_new_york]
  norm_num

theorem time_new_york_when_xiao_ming_calls : 
  ∀ (sydney new_york : Int), 
  sydney = 21 ∧ new_york = -4 → 
  new_york = sydney - 15 := 
by
  intros sydney new_york h
  cases h with h_sydney h_new_york
  rw [h_sydney, h_new_york]
  norm_num

theorem time_in_sydney_after_flight : 
  ∀ (beijing sydney : Int), 
  beijing = 8 ∧ sydney = 11 → 
  (∃ (flight_time : Int), flight_time = 12) → 
  (23 + 12) % 24 + sydney - beijing = 14 := 
by
  intros beijing sydney h flight_time hf
  cases h with h_beijing h_sydney
  cases hf with h_flight_time
  rw [h_beijing, h_sydney, h_flight_time]
  norm_num

theorem beijing_double_moscow : 
  ∀ (beijing moscow : Int), 
  beijing = 8 ∧ moscow = 3 → 
  ∃ (x : Int), x = 10 ∧ beijing - moscow = x := 
by
  intros beijing moscow h
  cases h with h_beijing h_moscow
  use 10
  rw [h_beijing, h_moscow]
  norm_num
  split
  norm_num
  norm_num

end time_difference_beijing_new_york_time_new_york_when_xiao_ming_calls_time_in_sydney_after_flight_beijing_double_moscow_l285_285262


namespace ball_hits_ground_time_l285_285845

theorem ball_hits_ground_time :
  ∃ t : ℝ, -16 * t^2 - 30 * t + 180 = 0 ∧ t = 1.25 := by
  sorry

end ball_hits_ground_time_l285_285845


namespace max_real_part_sum_w_l285_285679

noncomputable def z (j : ℕ) : ℂ :=
  27 * (complex.cos (2 * real.pi * j / 12) + complex.I * complex.sin (2 * real.pi * j / 12))

noncomputable def w (j : ℕ) : ℂ :=
  if complex.cos (2 * real.pi * j / 12) ≥ 0 then 
    z j 
  else 
    -complex.I * z j

theorem max_real_part_sum_w : 
  ∑ j in finset.range 12, w j = 81 :=
sorry

end max_real_part_sum_w_l285_285679


namespace simplify_trig_identity_l285_285753

-- Defining the trigonometric functions involved
def cot (x : Real) : Real := (cos x) / (sin x)
def tan (x : Real) : Real := (sin x) / (cos x)
def csc (x : Real) : Real := 1 / (sin x)

theorem simplify_trig_identity :
  cot 20 + tan 10 = csc 20 :=
by
  sorry

end simplify_trig_identity_l285_285753


namespace floor_equation_interval_l285_285930

theorem floor_equation_interval :
  {x : ℝ | ⌊x * ⌊x⌋⌋ = 49} = set.Ico 7 (50 / 7) :=
by
  sorry

end floor_equation_interval_l285_285930


namespace angle_SQR_measure_l285_285672

theorem angle_SQR_measure
    (angle_PQR : ℝ)
    (angle_PQS : ℝ)
    (h1 : angle_PQR = 40)
    (h2 : angle_PQS = 15) : 
    angle_PQR - angle_PQS = 25 := 
by
    sorry

end angle_SQR_measure_l285_285672


namespace rational_sum_eq_neg2_l285_285251

theorem rational_sum_eq_neg2 (a b : ℚ) (h : |a + 6| + (b - 4)^2 = 0) : a + b = -2 :=
sorry

end rational_sum_eq_neg2_l285_285251


namespace minimum_value_expression_l285_285149

theorem minimum_value_expression (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 :=
by
  sorry

end minimum_value_expression_l285_285149


namespace pages_written_on_wednesday_l285_285314

variable (minutesMonday minutesTuesday rateMonday rateTuesday : ℕ)
variable (totalPages : ℕ)

def pagesOnMonday (minutesMonday rateMonday : ℕ) : ℕ :=
  minutesMonday / rateMonday

def pagesOnTuesday (minutesTuesday rateTuesday : ℕ) : ℕ :=
  minutesTuesday / rateTuesday

def totalPagesMondayAndTuesday (minutesMonday minutesTuesday rateMonday rateTuesday : ℕ) : ℕ :=
  pagesOnMonday minutesMonday rateMonday + pagesOnTuesday minutesTuesday rateTuesday

def pagesOnWednesday (minutesMonday minutesTuesday rateMonday rateTuesday totalPages : ℕ) : ℕ :=
  totalPages - totalPagesMondayAndTuesday minutesMonday minutesTuesday rateMonday rateTuesday

theorem pages_written_on_wednesday :
  pagesOnWednesday 60 45 30 15 10 = 5 := by
  sorry

end pages_written_on_wednesday_l285_285314


namespace expand_binomials_l285_285925

theorem expand_binomials : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by 
  sorry

end expand_binomials_l285_285925


namespace max_valid_pairs_l285_285171

def valid_pairs (pairs : List (ℕ × ℕ)) : Prop := 
  ∀ (i j : ℕ) (ai bi aj bj : ℕ), 
    ((i < pairs.length) ∧ (j < pairs.length)) →
    ((pairs.nth i = some (ai, bi)) ∧ (pairs.nth j = some (aj, bj))) →
    (i ≠ j → ai ≠ aj ∧ ai ≠ bj ∧ bi ≠ aj ∧ bi ≠ bj) ∧
    (ai < bi) ∧
    ((i ≠ j) → (ai + bi ≠ aj + bj))

theorem max_valid_pairs :
  ∃ (pairs : List (ℕ × ℕ)), valid_pairs pairs ∧ pairs.length = 803 := 
sorry

end max_valid_pairs_l285_285171


namespace problem_solution_l285_285144

theorem problem_solution (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2):
    (x ∈ Set.Iio (-2) ∪ Set.Ioo (-2) ((1 - Real.sqrt 129)/8) ∪ Set.Ioo 2 3 ∪ Set.Ioi ((1 + (Real.sqrt 129))/8)) ↔
    (2 * x^2 / (x + 2) ≥ 3 / (x - 2) + 6 / 4) :=
by
  sorry

end problem_solution_l285_285144


namespace valid_arrangements_count_l285_285603

noncomputable def count_valid_arrangements : Nat :=
  let multiples_of_2 := [2, 4, 6, 8, 10]
  let multiples_of_3 := [9, 6, 3]
  let remaining_numbers := [1, 5, 7]
  -- Calculate permutations of remaining_numbers
  let permutations_of_remaining := Nat.factorial (List.length remaining_numbers)
  -- Calculate the total number of valid arrangements
  10.choose 7 * permutations_of_remaining -- 10C7 ways of placing the 7 non-fixed in 10 slots, with fixed order constraints

theorem valid_arrangements_count :
  count_valid_arrangements = 6480 := by
  sorry

end valid_arrangements_count_l285_285603


namespace matrix_cubic_l285_285496

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![2, -1],
  ![1, 1]
]

theorem matrix_cubic :
  matrix_a ^ 3 = ![
    ![3, -6],
    ![6, -3]
  ] := by
  sorry

end matrix_cubic_l285_285496


namespace probability_two_shirts_one_pants_one_socks_l285_285261

def total_items := 5 + 6 + 7

def ways_pick_4 := Nat.choose total_items 4

def ways_pick_2_shirts := Nat.choose 5 2

def ways_pick_1_pants := Nat.choose 6 1

def ways_pick_1_socks := Nat.choose 7 1

def favorable_outcomes := ways_pick_2_shirts * ways_pick_1_pants * ways_pick_1_socks

def probability := (favorable_outcomes : ℚ) / (ways_pick_4 : ℚ)

theorem probability_two_shirts_one_pants_one_socks :
  probability = 7 / 51 :=
by
  sorry

end probability_two_shirts_one_pants_one_socks_l285_285261


namespace not_coincidence_l285_285773

theorem not_coincidence (G : Type) [Fintype G] [DecidableEq G]
    (friend_relation : G → G → Prop)
    (h_friend : ∀ (a b : G), friend_relation a b → friend_relation b a)
    (initial_condition : ∀ (subset : Finset G), subset.card = 4 → 
         ∃ x ∈ subset, ∀ y ∈ subset, x ≠ y → friend_relation x y) :
    ∀ (subset : Finset G), subset.card = 4 → 
        ∃ x ∈ subset, ∀ y ∈ Finset.univ, x ≠ y → friend_relation x y :=
by
  intros subset h_card
  -- The proof would be constructed here
  sorry

end not_coincidence_l285_285773


namespace conic_is_parabola_l285_285908

-- Define the condition involving the given conic section equation
def conic_section_equation (x y : ℝ) : Prop :=
  abs (y - 3) = sqrt ((x + 1)^2 + (y - 1)^2)

-- State that this conic section is a parabola ("P")
theorem conic_is_parabola (x y : ℝ) (h : conic_section_equation x y) : 
  'P' = 'P' := 
sorry

end conic_is_parabola_l285_285908


namespace right_triangle_incircle_ratio_l285_285002

theorem right_triangle_incircle_ratio {P Q R S : Point}
  (PQ : dist P Q = 5) (QR : dist Q R = 12) (PR : dist P R = 13)
  (triangle_PQR : right_triangle P Q R)
  (on_PR : S ∈ segment P R)
  (angle_bisector : bisects ∠PQS Q ⊥ bisects ∠SQR)
  (r_p : radius (incircle (triangle P Q S)))
  (r_q : radius (incircle (triangle Q R S))) :
  r_p / r_q = 5 / 12 :=
sorry

end right_triangle_incircle_ratio_l285_285002


namespace least_positive_integer_remainder_2_l285_285039

theorem least_positive_integer_remainder_2 :
  ∃ n : ℕ, n > 1 ∧ (∀ m ∈ {3, 4, 5, 6, 7}, n % m = 2) ∧ n = 422 :=
by
  sorry

end least_positive_integer_remainder_2_l285_285039


namespace find_k_l285_285197

noncomputable def sin_cos_roots (θ : ℝ) (hθ : 0 < θ ∧ θ < 2 * Real.pi) (k : ℝ) : Prop :=
  let x := Polynomial.X in
  let f := x^2 - Polynomial.C k * x + Polynomial.C (k + 1) in
  f.roots = [sin θ, cos θ].toFinset

theorem find_k (θ : ℝ) (hθ : 0 < θ ∧ θ < 2 * Real.pi) (k : ℝ) 
  (h : sin_cos_roots θ hθ k) : k = -1 :=
sorry

end find_k_l285_285197


namespace serpent_ridge_trail_length_l285_285700

/-- Phoenix hiked the Serpent Ridge Trail last week. It took her five days to complete the trip.
The first two days she hiked a total of 28 miles. The second and fourth days she averaged 15 miles per day.
The last three days she hiked a total of 42 miles. The total hike for the first and third days was 30 miles.
How many miles long was the trail? -/
theorem serpent_ridge_trail_length
  (a b c d e : ℕ)
  (h1 : a + b = 28)
  (h2 : b + d = 30)
  (h3 : c + d + e = 42)
  (h4 : a + c = 30) :
  a + b + c + d + e = 70 :=
sorry

end serpent_ridge_trail_length_l285_285700


namespace simplify_cot_tan_l285_285730

theorem simplify_cot_tan : Real.cot (Real.pi / 9) + Real.tan (Real.pi / 18) = Real.csc (Real.pi / 9) := 
by
  sorry

end simplify_cot_tan_l285_285730


namespace ap_divides_aq_iff_p_divides_q_l285_285288

def sequence_a (d : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 1 else (sequence_a d (n - 1)) ^ d + 1

theorem ap_divides_aq_iff_p_divides_q (d p q : ℕ) (hd : d ≥ 2) :
  sequence_a d p ∣ sequence_a d q ↔ p ∣ q :=
by
  sorry

end ap_divides_aq_iff_p_divides_q_l285_285288


namespace max_kids_can_spell_names_l285_285794

-- Define the available letters and counts
def available_letters : Multiset Char := ['A', 'A', 'A', 'A', 'B', 'B', 'D', 'I', 'I', 'M', 'M', 'N', 'N', 'N', 'Ya', 'Ya']

-- Define the letters required for each name
def anna_req : Multiset Char := ['A', 'A', 'N', 'N']
def vanya_req : Multiset Char := ['B', 'A', 'N', 'Ya']
def danya_req : Multiset Char := ['D', 'A', 'N', 'Ya']
def dima_req : Multiset Char := ['D', 'I', 'M', 'A']

-- The statement that exactly 3 children can spell their names with the available letters
theorem max_kids_can_spell_names : 
  ∃ (anna vanya dani dima : Bool), anna && vanya && dima && ¬dani ∧ 
                                      Multiset.card (anna_req + vanya_req + dima_req) ≤ 16 ∧ 
                                      anna && (anna_req ⊆ available_letters) ∧ 
                                      vanya && (vanya_req ⊆ available_letters) ∧
                                      dima && (dima_req ⊆ available_letters) ∧
                                      ¬dani && (danya_req ∩ available_letters).card < 4 
:= sorry

end max_kids_can_spell_names_l285_285794


namespace cuberoot_inequality_l285_285703

theorem cuberoot_inequality (a b : ℝ) : a < b → (∃ x y : ℝ, x^3 = a ∧ y^3 = b ∧ (x = y ∨ x > y)) := 
sorry

end cuberoot_inequality_l285_285703


namespace quadratic_intersection_y_axis_l285_285351

theorem quadratic_intersection_y_axis :
  (∃ y, y = 3 * (0: ℝ)^2 - 4 * (0: ℝ) + 5 ∧ (0, y) = (0, 5)) :=
by
  sorry

end quadratic_intersection_y_axis_l285_285351


namespace lattice_points_distance_5_from_origin_l285_285272

theorem lattice_points_distance_5_from_origin :
  {p : Int × Int × Int // p.1 ^ 2 + p.2.1 ^ 2 + p.2.2 ^ 2 = 25}.to_finset.card = 42 :=
by
  sorry

end lattice_points_distance_5_from_origin_l285_285272


namespace mickey_final_number_zero_l285_285307

theorem mickey_final_number_zero (n : ℕ) (h : ∀ k, k < n → (2023 + k ∈ finset.range n)) :
  (mickey_final_number_zero_condition n = true) :=
by 
  -- Conditions as definitions in Lean 4
  let initial_seq := list.range (2022 + n),
  let sum_parity := list.sum initial_seq % 2 = 0 in
  -- Proof of the main theorem
  (n % 4 = 0 ∨ n % 4 = 3 ∧ n ≥ 4047) :=
sorry

end mickey_final_number_zero_l285_285307


namespace minimum_cubes_needed_l285_285069

/-- 
  A figure is constructed from unit cubes, where each cube shares at least one face with another cube. 
  Prove that the minimum number of cubes needed to build a figure with the front and side views shown 
  is equal to 6.

  FRONT:
   __
  |__|__|
  |__|__|__|

  SIDE:
  __
  |__|
  |__|__|
  |__|
 -/
theorem minimum_cubes_needed (H : ∀ (front side : nat → nat → bool), 
  (front 0 0 = tt ∧ front 0 1 = tt ∧ front 1 0 = tt ∧ front 1 1 = tt ∧ front 1 2 = tt) ∧
  (side 0 0 = tt ∧ side 1 0 = tt ∧ side 1 1 = tt ∧ side 2 0 = tt ∧ side 2 1 = tt) ∧
  (∀ x y, front x y = tt → ∃ dx dy, dx ≠ 0 ∨ dy ≠ 0 ∧ front (x + dx) (y + dy) = tt) ∧
  (∀ x y, side x y = tt → ∃ dx dy, dx ≠ 0 ∨ dy ≠ 0 ∧ side (x + dx) (y + dy) = tt)) : 
  ∃ n, n = 6 :=
by 
  sorry

end minimum_cubes_needed_l285_285069


namespace sqrt2_over_2_not_covered_by_rationals_l285_285511

noncomputable def rational_not_cover_sqrt2_over_2 : Prop :=
  ∀ (a b : ℤ) (h_ab : Int.gcd a b = 1) (h_b_pos : b > 0)
  (h_frac : (a : ℚ) / b ∈ Set.Ioo 0 1),
  abs ((Real.sqrt 2) / 2 - (a : ℚ) / b) > 1 / (4 * b^2)

-- Placeholder for the proof
theorem sqrt2_over_2_not_covered_by_rationals :
  rational_not_cover_sqrt2_over_2 := 
by sorry

end sqrt2_over_2_not_covered_by_rationals_l285_285511


namespace percentage_not_drop_l285_285650

def P_trip : ℝ := 0.40
def P_drop_given_trip : ℝ := 0.25
def P_not_drop : ℝ := 0.90

theorem percentage_not_drop :
  (1 - P_trip * P_drop_given_trip) = P_not_drop :=
sorry

end percentage_not_drop_l285_285650
