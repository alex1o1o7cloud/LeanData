import Mathlib
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.Geometry
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Archimedean
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.CombinatorialNum
import Mathlib.Data.Int.Gcd
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Base
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Angle.Basic
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Triangle.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.Order.Basic
import Mathlib.Probability
import Mathlib.Set.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Mathlib.Topology.Algebra.Order.Basic

namespace chipmunk_acorns_l710_710805

-- Define the conditions and goal for the proof
theorem chipmunk_acorns :
  ∃ x : ℕ, (∀ h_c h_s : ℕ, h_c = h_s + 4 → 3 * h_c = x ∧ 4 * h_s = x) → x = 48 :=
by {
  -- We assume the problem conditions as given
  sorry
}

end chipmunk_acorns_l710_710805


namespace log_bound_sum_l710_710409

theorem log_bound_sum (c d : ℕ) (h_c : c = 10) (h_d : d = 11) (h_bound : 10 < Real.log 1350 / Real.log 2 ∧ Real.log 1350 / Real.log 2 < 11) : c + d = 21 :=
by
  -- omitted proof
  sorry

end log_bound_sum_l710_710409


namespace intersection_of_sets_l710_710486

noncomputable def A (x : ℝ) : Prop := -1 < x ∧ x < 1
noncomputable def B (y : ℝ) : Prop := (1 / 2) < y ∧ y < 2

theorem intersection_of_sets (x y : ℝ) :
  (A x → y = 2^x) ∧ (y = log (1 - x) / log 2) < 1 → (y ∈ set.Ioo (1 / 2) 1) :=
by
  sorry

end intersection_of_sets_l710_710486


namespace percentage_B_of_C_l710_710370

variable (A B C : ℝ)

theorem percentage_B_of_C (h1 : A = 0.08 * C) (h2 : A = 0.5 * B) : B = 0.16 * C :=
by
  sorry

end percentage_B_of_C_l710_710370


namespace b_arithmetic_sequence_a_general_formula_l710_710046

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710046


namespace trig_identity_simplification_l710_710983

theorem trig_identity_simplification :
  (sin 10 * pi / 180 + sin 20 * pi / 180 + sin 30 * pi / 180 +
   sin 40 * pi / 180 + sin 50 * pi / 180 + sin 60 * pi / 180 +
   sin 70 * pi / 180 + sin 80 * pi / 180) / (cos 5 * pi / 180 * cos 10 * pi / 180 * cos 20 * pi / 180) 
  = 4 * sqrt 2 := 
  sorry

end trig_identity_simplification_l710_710983


namespace group4_equations_groupN_equations_find_k_pos_l710_710794

-- Conditions from the problem
def group1_fractions := (1 : ℚ) / 1 + (1 : ℚ) / 3 = 4 / 3
def group1_pythagorean := 4^2 + 3^2 = 5^2

def group2_fractions := (1 : ℚ) / 3 + (1 : ℚ) / 5 = 8 / 15
def group2_pythagorean := 8^2 + 15^2 = 17^2

def group3_fractions := (1 : ℚ) / 5 + (1 : ℚ) / 7 = 12 / 35
def group3_pythagorean := 12^2 + 35^2 = 37^2

-- Proof Statements
theorem group4_equations :
  ((1 : ℚ) / 7 + (1 : ℚ) / 9 = 16 / 63) ∧ (16^2 + 63^2 = 65^2) := 
  sorry

theorem groupN_equations (n : ℕ) :
  ((1 : ℚ) / (2 * n - 1) + (1 : ℚ) / (2 * n + 1) = 4 * n / (4 * n^2 - 1)) ∧
  ((4 * n)^2 + (4 * n^2 - 1)^2 = (4 * n^2 + 1)^2) :=
  sorry

theorem find_k_pos (k : ℕ) : 
  k^2 + 9603^2 = 9605^2 → k = 196 := 
  sorry

end group4_equations_groupN_equations_find_k_pos_l710_710794


namespace length_of_XY_l710_710426

open Real

theorem length_of_XY
  (X Y Z : Type)
  [Inhabited X] [Inhabited Y] [Inhabited Z]
  (triangle_XYZ : Triangle X Y Z)
  (angle_XYZ : ∡ XYZ = π / 6)
  (side_XZ : distance X Z = 12) :
  distance X Y = 4 * sqrt 3 :=
by 
  sorry

end length_of_XY_l710_710426


namespace minimum_deposits_needed_l710_710735

noncomputable def annual_salary_expense : ℝ := 100000
noncomputable def annual_fixed_expense : ℝ := 170000
noncomputable def interest_rate_paid : ℝ := 0.0225
noncomputable def interest_rate_earned : ℝ := 0.0405

theorem minimum_deposits_needed :
  ∃ (x : ℝ), 
    (interest_rate_earned * x = annual_salary_expense + annual_fixed_expense + interest_rate_paid * x) →
    x = 1500 :=
by
  sorry

end minimum_deposits_needed_l710_710735


namespace maximum_value_is_16_l710_710568

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
(x^2 - 2 * x * y + 2 * y^2) * (x^2 - 2 * x * z + 2 * z^2) * (y^2 - 2 * y * z + 2 * z^2)

theorem maximum_value_is_16 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  maximum_value x y z ≤ 16 :=
by
  sorry

end maximum_value_is_16_l710_710568


namespace problem1_problem2_l710_710757

theorem problem1 : (82 - 15) * (32 + 18) = 3350 :=
by
  sorry

theorem problem2 : (25 + 4) * 75 = 2175 :=
by
  sorry

end problem1_problem2_l710_710757


namespace part1_arithmetic_sequence_part2_general_formula_l710_710097

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710097


namespace solution_set_l710_710478

def f (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 1/2 then 1/2 * x + 1
  else if h : 1/2 ≤ x ∧ x < 1 then 2^(-4*x) + 1
  else 0

theorem solution_set :
  {x | 0 < x ∧ x < 1 ∧ f x > real.sqrt 2 / 8 + 1} = 
  {x | real.sqrt 2 / 4 < x ∧ x < 5 / 8} :=
sorry

end solution_set_l710_710478


namespace price_of_fruits_l710_710435

theorem price_of_fruits
  (x y : ℝ)
  (h1 : 9 * x + 10 * y = 73.8)
  (h2 : 17 * x + 6 * y = 69.8)
  (hx : x = 2.2)
  (hy : y = 5.4) : 
  9 * 2.2 + 10 * 5.4 = 73.8 ∧ 17 * 2.2 + 6 * 5.4 = 69.8 :=
by
  sorry

end price_of_fruits_l710_710435


namespace relationship_y1_y2_l710_710888

theorem relationship_y1_y2 (y1 y2 : ℝ) (m : ℝ) (h_m : m ≠ 0) 
  (hA : y1 = m * (-2) + 4) (hB : 3 = m * 1 + 4) (hC : y2 = m * 3 + 4) : y1 > y2 :=
by
  sorry

end relationship_y1_y2_l710_710888


namespace smallest_integer_neither_prime_nor_square_no_prime_factors_lt_55_l710_710319

theorem smallest_integer_neither_prime_nor_square_no_prime_factors_lt_55 :
  ∃ (n : ℕ), n = 3599 ∧ n > 0 ∧ ¬ prime n ∧ ¬ is_square n ∧ ∀ p : ℕ, p ∣ n → prime p → p ≥ 55 :=
by
  -- proof
  sorry

end smallest_integer_neither_prime_nor_square_no_prime_factors_lt_55_l710_710319


namespace diameter_is_chord_l710_710324

theorem diameter_is_chord {C : Circle} (d : diameter C) : chord C d :=
sorry

end diameter_is_chord_l710_710324


namespace Mark_speeding_ticket_owed_amount_l710_710961

theorem Mark_speeding_ticket_owed_amount :
  let base_fine := 50
  let additional_penalty_per_mph := 2
  let mph_over_limit := 45
  let school_zone_multiplier := 2
  let court_costs := 300
  let lawyer_fee_per_hour := 80
  let lawyer_hours := 3
  let additional_penalty := additional_penalty_per_mph * mph_over_limit
  let pre_school_zone_fine := base_fine + additional_penalty
  let doubled_fine := pre_school_zone_fine * school_zone_multiplier
  let total_fine_with_court_costs := doubled_fine + court_costs
  let lawyer_total_fee := lawyer_fee_per_hour * lawyer_hours
  let total_owed := total_fine_with_court_costs + lawyer_total_fee
  total_owed = 820 :=
by
  sorry

end Mark_speeding_ticket_owed_amount_l710_710961


namespace part1_arithmetic_sequence_part2_general_formula_l710_710066

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710066


namespace inverse_value_exists_l710_710860

noncomputable def f (a x : ℝ) := a^x - 1

theorem inverse_value_exists (a : ℝ) (h : f a 1 = 1) : (f a)⁻¹ 3 = 2 :=
by
  sorry

end inverse_value_exists_l710_710860


namespace part1_arithmetic_sequence_part2_general_formula_l710_710104

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710104


namespace part1_arithmetic_sequence_part2_general_formula_l710_710068

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710068


namespace third_digit_after_decimal_point_l710_710321

-- Define the key elements in the problem
def cuberoot (x : ℝ) := x^(1/3)

-- Define the sum in question
def x := cuberoot (2 + real.sqrt 5) + cuberoot (2 - real.sqrt 5)

-- Prove the third digit after the decimal point of x is 0
theorem third_digit_after_decimal_point : (x^3 + 3 * x = 4) ∧ x = 1 → third_digit_after_decimal_point x = 0 := 
by
  sorry

end third_digit_after_decimal_point_l710_710321


namespace inequality_solution_l710_710989

def satisfies_inequality (x : ℝ) : Prop :=
  abs ((3 * x + 2) / (x - 2)) ≥ 3

theorem inequality_solution :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | x ∈ Icc (2 / 3 : ℝ) ⊤} :=
sorry

end inequality_solution_l710_710989


namespace area_ratio_l710_710837

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the vector positions
def AP (AB AC : ℝ × ℝ) : ℝ × ℝ :=
  (2 / 5 * AB.1 + 1 / 5 * AC.1, 2 / 5 * AB.2 + 1 / 5 * AC.2)

def AQ (AB AC : ℝ × ℝ) : ℝ × ℝ :=
  (2 / 3 * AB.1 + 1 / 3 * AC.1, 2 / 3 * AB.2 + 1 / 3 * AC.2)

-- Define the area ratio
theorem area_ratio (AB AC : ℝ × ℝ) :
  let AP := AP AB AC
  let AQ := AQ AB AC
  (vector_magnitude AP / vector_magnitude AQ) = (3 / 5) := 
begin
  sorry
end

end area_ratio_l710_710837


namespace tomato_land_correct_l710_710199

-- Define the conditions
def total_land : ℝ := 4999.999999999999
def cleared_fraction : ℝ := 0.9
def grapes_fraction : ℝ := 0.1
def potato_fraction : ℝ := 0.8

-- Define the calculated values based on conditions
def cleared_land : ℝ := cleared_fraction * total_land
def grapes_land : ℝ := grapes_fraction * cleared_land
def potato_land : ℝ := potato_fraction * cleared_land
def tomato_land : ℝ := cleared_land - (grapes_land + potato_land)

-- Prove the question using conditions, which should end up being 450 acres.
theorem tomato_land_correct : tomato_land = 450 :=
by sorry

end tomato_land_correct_l710_710199


namespace part1_sequence_arithmetic_part2_general_formula_l710_710137

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710137


namespace part1_arithmetic_sequence_part2_general_formula_l710_710077

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710077


namespace constant_term_poly_expr_l710_710288

-- Define the polynomial expression
def poly_expr (x : ℝ) : ℝ := (3 * x + 2 / x) ^ 8

-- Statement of the theorem
theorem constant_term_poly_expr : 
  let x := var in
  (polynomial.constant_coeff (C (3 * x) + C (2 / x))) = 90720 :=
begin
  sorry
end

end constant_term_poly_expr_l710_710288


namespace b_arithmetic_a_general_formula_l710_710010

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710010


namespace decode_plaintext_l710_710450

theorem decode_plaintext (a x y : ℕ) (h1 : y = a^x - 2) (h2 : 6 = a^3 - 2) (h3 : y = 14) : x = 4 := by
  sorry

end decode_plaintext_l710_710450


namespace convert_38_to_binary_l710_710773

theorem convert_38_to_binary : nat.to_digits 2 38 = [1, 0, 0, 1, 1, 0] := 
by 
  sorry

end convert_38_to_binary_l710_710773


namespace second_order_arithmetic_sequence_a30_l710_710808

theorem second_order_arithmetic_sequence_a30 {a : ℕ → ℝ}
  (h₁ : ∀ n, a (n + 1) - a n - (a (n + 2) - a (n + 1)) = 20)
  (h₂ : a 10 = 23)
  (h₃ : a 20 = 23) :
  a 30 = 2023 := 
sorry

end second_order_arithmetic_sequence_a30_l710_710808


namespace valid_votes_b_l710_710694

noncomputable def numVotesPolled : ℕ := 8720

def percentInvalid : ℝ := 0.20

def percentDifference : ℝ := 0.15

noncomputable def validVotes : ℕ := (1 - percentInvalid) * numVotesPolled

def aVotesExceedsBVotes : ℕ := percentDifference * numVotesPolled

theorem valid_votes_b :
  ∃ (B_votes : ℕ), B_votes = 2834 ∧
  validVotes = B_votes + (B_votes + aVotesExceedsBVotes) := 
sorry

end valid_votes_b_l710_710694


namespace min_value_of_sum_l710_710820

variable {a b : ℝ}

noncomputable def min_sum : ℝ :=
  4 * Real.sqrt 5

theorem min_value_of_sum (h1 : Real.log10 a + Real.log10 b = 1) (h2 : a > 0) (h3 : b > 0) :
  ∃ c, c = a + 2 * b ∧ c = min_sum :=
by
  sorry

end min_value_of_sum_l710_710820


namespace AmphibiansFrogs_l710_710924

-- Definitions for the species
inductive Species
| toad
| frog

open Species

-- Definitions for the amphibians
def Brian : Species
def Chris : Species
def LeRoy : Species
def Mike : Species
def Jack : Species

-- Statements by the amphibians
def Brian_statement := (Brian ≠ Mike)
def Chris_statement := (LeRoy = frog)
def LeRoy_statement := (∃f, fro(H

def LeRoy_statement := Exists (λ (frogs : Nat), (frogs ≥ 3) ∧ (frogs = count_frogs ([Brian, Chris, LeRoy, Mike, Jack])))
let count_frogs (lst : List Species) : Nat := lst.countp (λ s, s = frog)
def Mike_statement := (count_toads ([Brian, Chris, LeRoy, Mike, Jack]) = 3)
def Jack_statement := (Chris = toad)

-- Definitions for counting functions
def count_frogs (lst : List Species) : Nat := lst.countp (λ s, s = frog)
def count_toads (lst : List Species) : Nat := lst.countp (λ s, s = toad)

theorem AmphibiansFrogs : 
  (Brian = toad ∨ Brian = frog) ∧ 
  (Chris = toad ∨ Chris = frog) ∧ 
  (LeRoy = toad ∨ LeRoy = frog) ∧ 
  (Mike = toad ∨ Mike = frog) ∧ 
  (Jack = toad ∨ Jack = frog) ∧ 
  (Brian_statement ↔ (Brian = toad → Mike = frog ∧ Brian = frog → Mike = toad)) ∧ 
  (Chris_statement ↔ (Chris = toad ∧ LeRoy = frog → Chris = frog ∧ LeRoy = toad)) ∧ 
  (LeRoy_statement ↔ (count_frogs ([Brian, Chris, LeRoy, Mike, Jack]) < 3)) ∧ 
  (Mike_statement ↔ (count_toads ([Brian, Chris, LeRoy, Mike, Jack]) = 3)) ∧ 
  (Jack_statement ↔ (Jack = toad ∧ Chris = toad → Jack = frog ∧ Chris = frog)) → 
  count_frogs ([Brian, Chris, LeRoy, Mike, Jack]) = 2 :=
by
  sorry

end AmphibiansFrogs_l710_710924


namespace find_speed_l710_710729

theorem find_speed (v d : ℝ) (h1 : d > 0) (h2 : 1.10 * v > 0) (h3 : 84 = 2 * d / (d / v + d / (1.10 * v))) : v = 80.18 := 
sorry

end find_speed_l710_710729


namespace infinite_rel_prime_divisible_pairs_l710_710212

theorem infinite_rel_prime_divisible_pairs :
  ∀ n ≥ 2, let a := 2^n - 1; let b := 2^n + 1 in 
    nat.coprime a b ∧ (a^b + b^a) % (a + b) = 0 := 
by
  intros n hn
  let a := 2^n - 1
  let b := 2^n + 1
  sorry

end infinite_rel_prime_divisible_pairs_l710_710212


namespace find_divisors_l710_710897

theorem find_divisors (N : ℕ) :
  (∃ k : ℕ, 2014 = k * (N + 1) ∧ k < N) ↔ (N = 2013 ∨ N = 1006 ∨ N = 105 ∨ N = 52) := by
  sorry

end find_divisors_l710_710897


namespace plot_area_is_correct_l710_710612

-- Define the problem conditions
def actual_plot_area_acres
  (base1 : ℝ) (base2 : ℝ) (height : ℝ)
  (scale : ℝ)
  (sq_miles_to_acres : ℝ) : ℝ :=
let area_cm := 0.5 * (base1 + base2) * height in
let area_miles := (area_cm * (scale ^ 2)) in
area_miles * sq_miles_to_acres

-- Prove the actual plot area in acres given the conditions
theorem plot_area_is_correct :
  actual_plot_area_acres 20 25 15 5 640 = 5400000 :=
by
  sorry

end plot_area_is_correct_l710_710612


namespace expression_result_l710_710768

theorem expression_result :
  3 * 3^4 - 9^20 / 9^18 + 5^3 = 287 := 
begin
  sorry
end

end expression_result_l710_710768


namespace find_n_in_arithmetic_sequence_l710_710387

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 2) = a (n + 1) + d

theorem find_n_in_arithmetic_sequence (x : ℝ) (n : ℕ) (b : ℕ → ℝ)
  (h1 : b 1 = Real.exp x) 
  (h2 : b 2 = x) 
  (h3 : is_arithmetic_sequence b) : 
  b n = 1 + Real.exp x ↔ n = (1 + x) / (x - Real.exp x) :=
sorry

end find_n_in_arithmetic_sequence_l710_710387


namespace solve_complex_eq_l710_710407

def complex_num_z : ℂ := -ℚ.ofRat 17 / 25 + ℚ.ofRat 6 / 25 * Complex.I

theorem solve_complex_eq : 
  ∃ (a b : ℚ), (z : ℂ) = a + b * Complex.I ∧ (conjugate_z : ℂ) = a - b * Complex.I ∧ 
   (3 * z + 4 * Complex.I * conjugate_z = -3 - 2 * Complex.I) :=
by
  sorry

end solve_complex_eq_l710_710407


namespace solve_for_y_l710_710222

theorem solve_for_y (y : ℚ) (h : 1/3 + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end solve_for_y_l710_710222


namespace probability_of_odd_divisor_l710_710264

noncomputable def prime_factorization_15! : ℕ :=
  (2 ^ 11) * (3 ^ 6) * (5 ^ 3) * (7 ^ 2) * 11 * 13

def total_factors_15! : ℕ :=
  (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

def odd_factors_15! : ℕ :=
  (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

def probability_odd_divisor_15! : ℚ :=
  odd_factors_15! / total_factors_15!

theorem probability_of_odd_divisor : probability_odd_divisor_15! = 1 / 12 :=
by
  sorry

end probability_of_odd_divisor_l710_710264


namespace total_sales_correct_l710_710621

def normal_sales_per_month : ℕ := 21122
def additional_sales_in_june : ℕ := 3922
def sales_in_june : ℕ := normal_sales_per_month + additional_sales_in_june
def sales_in_july : ℕ := normal_sales_per_month
def total_sales : ℕ := sales_in_june + sales_in_july

theorem total_sales_correct :
  total_sales = 46166 :=
by
  -- Proof goes here
  sorry

end total_sales_correct_l710_710621


namespace b_seq_arithmetic_a_seq_formula_l710_710158

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710158


namespace not_divides_two_pow_n_sub_one_l710_710981

theorem not_divides_two_pow_n_sub_one (n : ℕ) (h1 : n > 1) : ¬ n ∣ (2^n - 1) :=
sorry

end not_divides_two_pow_n_sub_one_l710_710981


namespace find_smallest_pos_angle_l710_710402

theorem find_smallest_pos_angle (y : Real) (h : 6 * sin y * (cos y)^3 - 6 * (sin y)^3 * cos y = 3/2) : y = 7.5 * (Real.pi / 180) := 
by
  sorry

end find_smallest_pos_angle_l710_710402


namespace circle_diameter_l710_710997

theorem circle_diameter (C : ℝ) (hC : C = 100) : ∃ d : ℝ, C = π * d ∧ d = 100 / π :=
by {
  use 100 / π,
  split,
  { exact hC, },
  { sorry } }.

end circle_diameter_l710_710997


namespace Brittany_older_by_3_years_l710_710754

-- Define the necessary parameters as assumptions
variable (Rebecca_age : ℕ) (Brittany_return_age : ℕ) (vacation_years : ℕ)

-- Initial conditions
axiom h1 : Rebecca_age = 25
axiom h2 : Brittany_return_age = 32
axiom h3 : vacation_years = 4

-- Definition to capture Brittany's age before vacation
def Brittany_age_before_vacation (return_age vacation_period : ℕ) : ℕ := return_age - vacation_period

-- Theorem stating that Brittany is 3 years older than Rebecca
theorem Brittany_older_by_3_years :
  Brittany_age_before_vacation Brittany_return_age vacation_years - Rebecca_age = 3 :=
by
  sorry

end Brittany_older_by_3_years_l710_710754


namespace problem_conditions_l710_710081

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710081


namespace bn_is_arithmetic_an_general_formula_l710_710116

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710116


namespace vehicles_not_speedsters_l710_710352

theorem vehicles_not_speedsters (V : ℕ) (h : (9/20 : ℚ) * V = 54) : (V - (3/4 : ℚ) * V).natAbs = 30 :=
by
  sorry

end vehicles_not_speedsters_l710_710352


namespace problem_conditions_l710_710082

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710082


namespace units_digit_proof_l710_710431

def units_digit (n : ℤ) : ℤ := n % 10

theorem units_digit_proof :
  ∀ (a b c : ℤ),
  a = 8 →
  b = 18 →
  c = 1988 →
  units_digit (a * b * c - a^3) = 0 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  -- Proof will go here
  sorry

end units_digit_proof_l710_710431


namespace total_siblings_weight_l710_710389

variable (Antonio_weight : ℕ) (sister_diff : ℕ)
def sister_weight := Antonio_weight - sister_diff
def total_weight := Antonio_weight + sister_weight Antonio_weight sister_diff

theorem total_siblings_weight 
  (h_Antonio_weight : Antonio_weight = 50) 
  (h_sister_diff : sister_diff = 12) : 
  total_weight Antonio_weight sister_diff = 88 := 
by
  rw [total_weight, sister_weight, h_Antonio_weight, h_sister_diff]
  norm_num
  sorry

end total_siblings_weight_l710_710389


namespace solution_l710_710840

noncomputable def problem_statement : Prop :=
  ∃ (m n s t : ℝ), m ∈ ℝ* ∧ n ∈ ℝ* ∧ s ∈ ℝ* ∧ t ∈ ℝ* ∧
  m + n = 3 ∧
  m < n ∧
  (m / s + n / t) = 1 ∧
  s + t = 3 + 2 * Real.sqrt 2 ∧
  (∃ (m n : ℝ), (m, n) = (1, 2)) ∧ -- Because we deduced m = 1, n = 2
  (∃ (x1 y1 x2 y2 : ℝ), 
  x1 + x2 = 2 ∧ 
  y1 + y2 = 4 ∧
  4 * x1 ^ 2 + y1 ^ 2 = 16 ∧ 
  4 * x2 ^ 2 + y2 ^ 2 = 16 ∧ 
  (y2 - y1) / (x2 - x1) = -2) →
  ∀ (x y : ℝ), 
  (2 * x + y - 4 = 0)

theorem solution : problem_statement :=
begin
  sorry
end

end solution_l710_710840


namespace max_value_neg_expr_l710_710506

theorem max_value_neg_expr (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  - (1 / (2 * a)) - (2 / b) ≤ - (9 / 2) :=
by 
  sorry

end max_value_neg_expr_l710_710506


namespace gardener_majestic_tree_count_l710_710528

def is_majestic_tree (height : ℕ) : Prop := height ≥ 10^6

def gardener_strategy_increases_trees (board : matrix ℤ 2022 2022 ℕ) (coords : fin 2022 × fin 2022) : matrix ℤ 2022 2022 ℕ :=
  let (i, j) := coords in
  matrix.update board i j (board i j + 1)
    |> matrix.update (fin.nat_add i 1) j        (board (fin.nat_add i 1) j + 1)
    |> matrix.update (fin.nat_add i 1) (fin.nat_add j 1) (board (fin.nat_add i 1) (fin.nat_add j 1) + 1)
    |> matrix.update i (fin.nat_add j 1)        (board i (fin.nat_add j 1) + 1)
    |> matrix.update (fin.nat_sub i 1) j        (board (fin.nat_sub i 1) j + 1)
    |> matrix.update (fin.nat_sub i 1) (fin.nat_add j 1) (board (fin.nat_sub i 1) (fin.nat_add j 1) + 1)
    |> matrix.update i (fin.nat_add j 1)        (board i (fin.nat_add j 1) + 1)

def lumberjack_strategy_decreases_trees (board : matrix ℤ 2022 2022 ℕ) (coords : (list (fin 2022 × fin 2022))) : matrix ℤ 2022 2022 ℕ :=
  coords.foldl (λ b (i, j), matrix.update b i j (board i j - 1)) board

noncomputable def ensure_majestic_trees (initial_board : matrix ℤ 2022 2022 ℕ) : Prop :=
  ∃ K ≥ 2271380, ∀ (seq: list ((fin 2022 × fin 2022) × (list (fin 2022 × fin 2022)))),
    (seq.length % 2 = 0 →
      ∀ s ∈ seq.filter_map (λ x, if h : seq.index_of x % 2 = 0 then some x.1 else none),
      board = seq.foldl (λ b move, (if move.index_of % 2 = 0 then gardener_strategy_increases_trees b move.1 else lumberjack_strategy_decreases_trees b move.2)) initial_board 
      → (matrix.countp (λ h, is_majestic_tree h) board) ≥ K)

theorem gardener_majestic_tree_count :
  ensure_majestic_trees (matrix.zero_matrix 2022 2022) :=
sorry

end gardener_majestic_tree_count_l710_710528


namespace concrete_pillars_l710_710549

-- Definitions based on the conditions of the problem
def C_deck : ℕ := 1600
def C_anchor : ℕ := 700
def C_total : ℕ := 4800

-- Theorem to prove the concrete required for supporting pillars
theorem concrete_pillars : C_total - (C_deck + 2 * C_anchor) = 1800 :=
by sorry

end concrete_pillars_l710_710549


namespace inscribed_quadrilateral_inequality_l710_710629

-- Define the problem as a Lean 4 statement
theorem inscribed_quadrilateral_inequality (A B C D O : Point) (r : ℝ)
  (h_inscribed : circle ω center O radius r ∧ quadrilateral A B C D ∈ ω)
  (M N : Point) (h_M : midpoint M B C) (h_N : midpoint N A D) :
  2 * r ≤ dist M N :=
sorry

end inscribed_quadrilateral_inequality_l710_710629


namespace abs_diff_of_x_y_l710_710238

def arithmetic_mean_base8 (x y : ℕ) : ℕ :=
(x + y) / 2

def geometric_mean_base8 (x y : ℕ) : ℕ :=
nat.sqrt (x * y)

theorem abs_diff_of_x_y (x y a b : ℕ) 
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hxy_distinct : x ≠ y)
  (ha_range : 1 ≤ a ∧ a ≤ 7)
  (hb_range : 0 ≤ b ∧ b ≤ 7)
  (h_arith : arithmetic_mean_base8 x y = 8 * a + b)
  (h_geom : geometric_mean_base8 x y = 8 * b + a) :
  |x - y| = 66 := sorry

end abs_diff_of_x_y_l710_710238


namespace work_completion_l710_710688

theorem work_completion (W : ℝ) (A_rate B_rate Combined_rate : ℝ) :
  (B_rate = W / 30) ∧ (A_rate = W / 45) ∧ (Combined_rate = A_rate + B_rate) →
  18 = W / Combined_rate :=
by
  intros h
  cases h with hb ha
  cases ha with ha hc
  sorry

end work_completion_l710_710688


namespace penelope_vs_greta_l710_710972

-- Definitions and given conditions
def pig_food_intake : ℕ := 20

def goose_food_intake (G : ℕ) : ℕ := G

def mouse_food_intake (G : ℕ) : ℕ := G / 100

def elephant_food_intake (G : ℕ) : ℕ := 4000 * (G / 100)

def elephant_more_than_pig : ℕ := 60

-- The Lean 4 statement for the proof problem
theorem penelope_vs_greta (G : ℕ) (h1 : elephant_food_intake G = pig_food_intake + elephant_more_than_pig) : 
  20 / G = 10 :=
begin
  -- Given condition translated into a Lean hypothesis
  have h2 : elephant_food_intake G = 40 * G, from rfl,
  -- Solve the system of equations based on the hypotheses
  sorry
end

end penelope_vs_greta_l710_710972


namespace bn_is_arithmetic_an_general_formula_l710_710111

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710111


namespace no_third_quadrant_l710_710501

theorem no_third_quadrant {a b : ℝ} (h1 : 0 < a) (h2 : a < 1) (h3 : -1 < b) : ∀ x y : ℝ, (y = a^x + b) → ¬ (x < 0 ∧ y < 0) :=
by
  intro x y h
  sorry

end no_third_quadrant_l710_710501


namespace coordinates_of_A_eq_l710_710702

noncomputable def point := (ℝ × ℝ × ℝ)

def A (z : ℝ) : point := (0, 0, z)
def B : point := (-13, 4, 6)
def C : point := (10, -9, 5)

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem coordinates_of_A_eq (z : ℝ) (h : dist (A z) B = dist (A z) C) : z = 7.5 :=
by
  sorry

end coordinates_of_A_eq_l710_710702


namespace island_people_count_l710_710203

-- Define the problem statement as a theorem
theorem island_people_count (n : ℕ) :
  (∀ i : ℕ, i < n → (∃ j : ℕ, j ≠ i ∧ j < n ∧ 
    (∃ k : ℕ, k ≠ i ∧ k ≠ j ∧ k < n ∧ 
      (¬ (knight i ↔ liar j) ∨ ¬ (knight i ↔ liar k))))) →
  (∀ i : ℕ, i < n → (∃ j : ℕ, j ≠ i ∧ j < n ∧ 
    (∃ k : ℕ, k ≠ i ∧ k ≠ j ∧ k < n ∧ 
      (knight i → (liar j ∧ liar k))))) →
  n ≠ 2017 :=
by
  sorry

-- Dummy definitions for knights and liars to allow compilation
constant knight : ℕ → Prop
constant liar : ℕ → Prop

end island_people_count_l710_710203


namespace henry_books_l710_710495

theorem henry_books (initial_books packed_boxes each_box room_books coffee_books kitchen_books taken_books : ℕ)
  (h1 : initial_books = 99)
  (h2 : packed_boxes = 3)
  (h3 : each_box = 15)
  (h4 : room_books = 21)
  (h5 : coffee_books = 4)
  (h6 : kitchen_books = 18)
  (h7 : taken_books = 12) :
  initial_books - (packed_boxes * each_box + room_books + coffee_books + kitchen_books) + taken_books = 23 :=
by
  sorry

end henry_books_l710_710495


namespace shoe_cost_percentage_increase_l710_710720

/-- Given:
- The cost to repair used shoes is $14.50 and they will last for 1 year.
- The cost to buy new shoes is $32.00 and they will last for 2 years.

Prove that the percentage increase in the average cost per year of the new shoes over the repaired shoes is 10.34%. -/
theorem shoe_cost_percentage_increase 
    (cost_repair : ℝ := 14.5) 
    (years_repair : ℕ := 1) 
    (cost_new : ℝ := 32) 
    (years_new : ℕ := 2) : 
    ((cost_new / years_new - cost_repair / years_repair) / (cost_repair / years_repair) * 100) = 10.34 :=
by
  have avg_repair := cost_repair / years_repair
  have avg_new := cost_new / years_new
  have diff := avg_new - avg_repair
  have percent_increase := (diff / avg_repair) * 100
  show percent_increase = 10.34
  sorry

end shoe_cost_percentage_increase_l710_710720


namespace part1_sequence_arithmetic_part2_general_formula_l710_710140

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710140


namespace smallest_missing_unit_digit_of_odd_number_l710_710408

theorem smallest_missing_unit_digit_of_odd_number : ∀ d : ℕ, d ∈ {0, 2, 4, 6, 8} → d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9 → d = 0 :=
by
  sorry

end smallest_missing_unit_digit_of_odd_number_l710_710408


namespace ellipse_slope_angle_l710_710854

noncomputable def eccentricity := (sqrt 3) / 2
noncomputable def area := 4
noncomputable def a := 2
noncomputable def b := 1
noncomputable def A := (-(a : ℝ), 0 : ℝ)
noncomputable def dist := (4 * sqrt 2) / 5

theorem ellipse_slope_angle:
  (∀ a b : ℝ, a > b ∧ b > 0 ∧ eccentricity = sqrt ((a ^ 2 - b ^ 2) / a ^ 2) →
  (1/2) * 2 * a * 2 * b = area →
  (a = 2 ∧ b = 1) ∧
  ∃ l : ℝ → ℝ, l = λ x, k * (x + 2) ∧ |(A.1, A.2) - (l.1, l.2)| = dist →
  (x y : ℝ, y = l x ∧ (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) →
  (√(1 + k ^ 2) / (1 + 4 * k ^ 2) = √2 / 5) →
  32 * k ^ 4 - 9 * k ^ 2 - 23 = 0) →
  k = ±1 →
  (slope : ℝ, slope = tan ^ (-1) k → (slope = π / 4 ∨ slope = 3 * π / 4)) :=
by
  sorry

end ellipse_slope_angle_l710_710854


namespace expected_worth_of_coin_flip_l710_710746

theorem expected_worth_of_coin_flip :
  let p_heads := 2 / 3
  let p_tails := 1 / 3
  let gain_heads := 5
  let loss_tails := -9
  (p_heads * gain_heads) + (p_tails * loss_tails) = 1 / 3 :=
by
  -- Proof will be here
  sorry

end expected_worth_of_coin_flip_l710_710746


namespace telephone_number_problem_l710_710379

theorem telephone_number_problem :
  ∃ A B C D E F G H I J : ℕ,
    (A > B) ∧ (B > C) ∧ (D > E) ∧ (E > F) ∧ (G > H) ∧ (H > I) ∧ (I > J) ∧
    (D = E + 1) ∧ (E = F + 1) ∧ (D % 2 = 0) ∧ 
    (G = H + 2) ∧ (H = I + 2) ∧ (I = J + 2) ∧ (G % 2 = 1) ∧ (H % 2 = 1) ∧ (I % 2 = 1) ∧ (J % 2 = 1) ∧
    (A + B + C = 7) ∧ (B + C + F = 10) ∧ (A = 7) :=
sorry

end telephone_number_problem_l710_710379


namespace sum_reciprocal_geo_seq_l710_710541

theorem sum_reciprocal_geo_seq {a_5 a_6 a_7 a_8 : ℝ}
  (h_sum : a_5 + a_6 + a_7 + a_8 = 15 / 8)
  (h_prod : a_6 * a_7 = -9 / 8) :
  (1 / a_5) + (1 / a_6) + (1 / a_7) + (1 / a_8) = -5 / 3 := by
  sorry

end sum_reciprocal_geo_seq_l710_710541


namespace hyperbola_asymptotes_l710_710994

theorem hyperbola_asymptotes :
  ∀ {x y : ℝ},
    (x^2 / 9 - y^2 / 16 = 1) →
    (y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intro x y h
  sorry

end hyperbola_asymptotes_l710_710994


namespace trig_equation_solution_l710_710228

theorem trig_equation_solution (x : ℝ) (k p : ℤ) :
  (cos (8 * x) / (cos (3 * x) + sin (3 * x)) + 
   sin (8 * x) / (cos (3 * x) - sin (3 * x)) = real.sqrt 2) →
  (x = real.pi / 44 + 2 * real.pi * k / 11 ∧ k ≠ 11 * p + 4) :=
sorry

end trig_equation_solution_l710_710228


namespace lcm_is_perfect_square_l710_710194

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : ∃ k : ℕ, k^2 = Nat.lcm a b :=
by
  sorry

end lcm_is_perfect_square_l710_710194


namespace part1_part2_l710_710481

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2) / 2 - (a + 1) * x + a * Real.log x

theorem part1 (a : ℝ) (h : a = -1/2) : ∃ x : ℝ, f x a = 0 ∧ (∀ y : ℝ, f y a ≥ 0) :=
by {
  -- Placeholder for proof
  sorry
}

theorem part2 (a : ℝ) :
  (a ≤ 0 → (∀ x : ℝ, (0 < x ∧ x < 1) → f' x a < 0) ∧ (∀ x : ℝ, 1 < x → f' x a > 0)) ∧
  (0 < a ∧ a < 1 → (∀ x : ℝ, x < a → f' x a > 0) ∧ (∀ x : ℝ, a < x ∧ x < 1 → f' x a < 0) ∧ (∀ x : ℝ, 1 < x → f' x a > 0)) ∧
  (a = 1 → ∀ x : ℝ, 0 < x → f' x a ≥ 0) ∧
  (a > 1 → (∀ x : ℝ, x < 1 → f' x a > 0) ∧ (∀ x : ℝ, 1 < x ∧ x < a → f' x a < 0) ∧ (∀ x : ℝ, a < x → f' x a > 0)) :=
by {
  -- Placeholder for proof
  sorry
}

end part1_part2_l710_710481


namespace square_TU_squared_l710_710990

theorem square_TU_squared (P Q R S T U : ℝ × ℝ)
  (side : ℝ) (RT SU PT QU : ℝ)
  (hpqrs : (P.1 - S.1)^2 + (P.2 - S.2)^2 = side^2 ∧ (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = side^2 ∧ 
            (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = side^2 ∧ (S.1 - R.1)^2 + (S.2 - R.2)^2 = side^2)
  (hRT : (R.1 - T.1)^2 + (R.2 - T.2)^2 = RT^2)
  (hSU : (S.1 - U.1)^2 + (S.2 - U.2)^2 = SU^2)
  (hPT : (P.1 - T.1)^2 + (P.2 - T.2)^2 = PT^2)
  (hQU : (Q.1 - U.1)^2 + (Q.2 - U.2)^2 = QU^2)
  (side_eq_17 : side = 17) (RT_SU_eq_8 : RT = 8) (PT_QU_eq_15 : PT = 15) :
  (T.1 - U.1)^2 + (T.2 - U.2)^2 = 979.5 :=
by
  -- proof to be filled in
  sorry

end square_TU_squared_l710_710990


namespace P_Q_sum_equals_44_l710_710173

theorem P_Q_sum_equals_44 (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3))) :
  P + Q = 44 :=
sorry

end P_Q_sum_equals_44_l710_710173


namespace relationship_between_z1_and_z2_l710_710946

noncomputable def z1 : ℂ := (1:ℂ)^(-4) + (complex.I)^(-5) + (complex.I)^(-6) + (complex.I)^(-7) + (complex.I)^(-8) + (complex.I)^(-9) + (complex.I)^(-10) + (complex.I)^(-11) + (complex.I)^(-12)
noncomputable def z2 : ℂ := (1:ℂ)^(-4) - (complex.I)^(-5) - (complex.I)^(-6) - (complex.I)^(-7) - (complex.I)^(-8) - (complex.I)^(-9) - (complex.I)^(-10) - (complex.I)^(-11) - (complex.I)^(-12)

theorem relationship_between_z1_and_z2 : z1 = z2 := 
by 
-- The proof code would go here
sorry

end relationship_between_z1_and_z2_l710_710946


namespace average_score_first_6_matches_l710_710241

-- Definition of conditions
variable (total_avg : ℝ) (num_matches : ℕ) (last_avg : ℝ) (last_matches : ℕ)
variable (total_avg_eq : total_avg = 38.9) (num_matches_eq : num_matches = 10)
variable (last_avg_eq : last_avg = 35.75) (last_matches_eq : last_matches = 4)

theorem average_score_first_6_matches 
  (total_avg : ℝ) (num_matches : ℕ) (last_avg : ℝ) (last_matches : ℕ)
  (total_avg_eq : total_avg = 38.9) (num_matches_eq : num_matches = 10)
  (last_avg_eq : last_avg = 35.75) (last_matches_eq : last_matches = 4) : 
  let total_runs := total_avg * num_matches
  let last_runs := last_avg * last_matches
  let first_runs := total_runs - last_runs
  let total_first_matches := num_matches - last_matches
  total_first_matches = 6 → 
  first_runs / total_first_matches = 41 := 
by
  intros h_total_first_matches
  have h_total_runs : total_runs = 389 := by sorry
  have h_last_runs : last_runs = 143 := by sorry
  have h_first_runs : first_runs = 246 := by sorry
  has_field.to_div (first_runs / total_first_matches)
  simp [h_first_runs, h_total_first_matches]
  exact norm_numする
  done

end average_score_first_6_matches_l710_710241


namespace price_reduction_l710_710360

theorem price_reduction (x : ℝ) :
  (20 + 2 * x) * (40 - x) = 1200 → x = 20 :=
by
  sorry

end price_reduction_l710_710360


namespace lcm_perfect_square_l710_710189

-- Define the conditions and the final statement in Lean 4
theorem lcm_perfect_square (a b : ℕ) 
  (h: (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : 
  ∃ k : ℕ, lcm a b = k^2 :=
sorry

end lcm_perfect_square_l710_710189


namespace arithmetic_sequence_problem_l710_710916

theorem arithmetic_sequence_problem
  (a : ℕ → ℚ)
  (h : a 2 + a 4 + a 9 + a 11 = 32) :
  a 6 + a 7 = 16 :=
sorry

end arithmetic_sequence_problem_l710_710916


namespace function_not_strictly_decreasing_l710_710867

theorem function_not_strictly_decreasing (b : ℝ)
  (h : ¬ ∀ x1 x2 : ℝ, x1 < x2 → (-x1^3 + b*x1^2 - (2*b + 3)*x1 + 2 - b > -x2^3 + b*x2^2 - (2*b + 3)*x2 + 2 - b)) : 
  b < -1 ∨ b > 3 :=
by
  sorry

end function_not_strictly_decreasing_l710_710867


namespace intersection_of_A_and_B_l710_710838

open Set

def set_A := {x : ℕ | |x| < 3}
def set_B := {x : ℤ | -2 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : (set_A : Set ℤ) ∩ set_B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l710_710838


namespace chapatis_ordered_l710_710386

theorem chapatis_ordered (C : ℕ) 
  (chapati_cost : ℕ) (plates_rice : ℕ) (rice_cost : ℕ)
  (plates_mixed_veg : ℕ) (mixed_veg_cost : ℕ)
  (ice_cream_cups : ℕ) (ice_cream_cost : ℕ)
  (total_amount_paid : ℕ)
  (cost_eq : chapati_cost = 6)
  (plates_rice_eq : plates_rice = 5)
  (rice_cost_eq : rice_cost = 45)
  (plates_mixed_veg_eq : plates_mixed_veg = 7)
  (mixed_veg_cost_eq : mixed_veg_cost = 70)
  (ice_cream_cups_eq : ice_cream_cups = 6)
  (ice_cream_cost_eq : ice_cream_cost = 40)
  (total_paid_eq : total_amount_paid = 1051) :
  6 * C + 5 * 45 + 7 * 70 + 6 * 40 = 1051 → C = 16 :=
by
  intro h
  sorry

end chapatis_ordered_l710_710386


namespace mason_grandmother_age_l710_710185

theorem mason_grandmother_age (mason_age: ℕ) (sydney_age: ℕ) (father_age: ℕ) (grandmother_age: ℕ)
  (h1: mason_age = 20)
  (h2: mason_age * 3 = sydney_age)
  (h3: sydney_age + 6 = father_age)
  (h4: father_age * 2 = grandmother_age) : 
  grandmother_age = 132 :=
by
  sorry

end mason_grandmother_age_l710_710185


namespace geom_sequence_a4_times_a7_l710_710523

theorem geom_sequence_a4_times_a7 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_q : q = 2) 
  (h_a2_a5 : a 2 * a 5 = 32) : 
  a 4 * a 7 = 512 :=
by 
  sorry

end geom_sequence_a4_times_a7_l710_710523


namespace propositions_truth_l710_710858

variable (P₁ P₂ P₃ P₄ : Prop)

def parallel_lines_in_plane_parallel_planes (P₁ : Prop) : Prop :=
  P₁ ↔ (∀ (L1 L2 : ℝ), (L1 ∥ P₃ ∧ L2 ∥ P₃) → (P₁ ∥ P₄))

def plane_perpendicularity (P₂ : Prop) : Prop :=
  P₂ ↔ (∀ (L : ℝ), (L ⊥ P₁ ∧ L ∈ P₄) → (P₁ ⊥ P₄))

def two_lines_perpendicular_same_line_parallel (P₃ : Prop) : Prop :=
  P₃ ↔ (∀ (L1 L2 L : ℝ), (L1 ⊥ L ∧ L2 ⊥ L) → (L1 ∥ L2))

def line_in_one_plane_not_perpendicular (P₄ : Prop) : Prop :=
  P₄ ↔ (∀ (L : ℝ), (L ∈ P₁ ∧ ¬ L ⊥ (P₁ ∩ P₄)) → ¬(L ⊥ P₄))

theorem propositions_truth :
  (parallel_lines_in_plane_parallel_planes P₁ → false) ∧
  plane_perpendicularity P₂ ∧
  (two_lines_perpendicular_same_line_parallel P₃ → false) ∧
  line_in_one_plane_not_perpendicular P₄ :=
by sorry

end propositions_truth_l710_710858


namespace season_duration_l710_710652

theorem season_duration (total_games : ℕ) (games_per_month : ℕ) (H1 : total_games = 323) (H2 : games_per_month = 19) :
  total_games / games_per_month = 17 :=
by
  rw [H1, H2]
  norm_num
  -- sorry

end season_duration_l710_710652


namespace problem_conditions_l710_710088

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710088


namespace find_n_l710_710792

theorem find_n (n : ℕ) (hn : 0 ≤ n ∧ n ≤ 180) : cos (n : ℝ) = cos (317 : ℝ) → n = 43 := by 
  sorry

end find_n_l710_710792


namespace standard_equation_of_ellipse_maximum_area_triangle_OEN_l710_710449

-- Definitions for the standard equation of the ellipse

def ellipse_c (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem standard_equation_of_ellipse (a b c : ℝ) (h_c : c = 1) (h_rat : (b^2 / a^2) = 3/4) :
  ellipse_c x y a b ↔ ellipse_c x y 2 (sqrt 3) := sorry

-- Definitions for conditions to calculate the area of the triangle

def line_m (x a : ℝ) : Prop :=  x = -2 * a

theorem maximum_area_triangle_OEN (F : ℝ × ℝ) (M N : ℝ × ℝ) (O E : ℝ × ℝ) (a : ℝ)
  (h_F : F = (-1, 0)) (h_line_m : ∀ x y, line_m x a → (x, y) = M ∨ (x, y) = N )
  (h_MN_inters_ellipse : ∀ x y, ellipse_c x y a (sqrt (a^2 - 1)) →
     (x, y) = M ∨ (x, y) = N)
  (h_ME_perpendicular : ∀ x y,  E = ( -4, y))
  (h_area_condition : true): 
  ∃ t : ℝ, t ≥ 1 ∧ (let area := (15 * sqrt( ( t^2 - 1 ) )  / ( 3 * ( t^2 - 1 ) + 4 )) in area = 15 / 4 ) :=
sorry

end standard_equation_of_ellipse_maximum_area_triangle_OEN_l710_710449


namespace geometric_inequality_l710_710909

theorem geometric_inequality 
  {A B C P D E M : Type}
  [triangle A B C] -- Assume A, B, C form an acute triangle
  (midpoint_M : midpoint B C M) -- M is midpoint of B and C
  (interior_P : point_in_interior A B C P) -- P is in the interior of ABC
  (angle_bisector : angle_bisects A P B C) -- AP bisects angle BAC
  (circumcircle_ABP : on_circumcircle P A B D) -- D is on circumcircle through A, B, P
  (circumcircle_ACP : on_circumcircle P A C E) -- E is on circumcircle through A, C, P
  (DE_eq_MP : dist D E = dist M P) -- DE = MP
: dist B C = 2 * dist B P := 
sorry

end geometric_inequality_l710_710909


namespace symmetric_point_origin_l710_710245

theorem symmetric_point_origin (x y : ℤ) (hx : x = -2) (hy : y = 3) : 
    let symmetric_x := -x,
        symmetric_y := -y
    in (symmetric_x, symmetric_y) = (2, -3) :=
by {
  sorry
}

end symmetric_point_origin_l710_710245


namespace sarah_least_days_to_repay_at_least_three_times_l710_710213

theorem sarah_least_days_to_repay_at_least_three_times (
    borrow_amount : ℕ := 50,
    interest_rate : ℕ := 10,
    three_times : ℕ := 3
) : 
    ∃ x : ℕ, borrow_amount + x * (borrow_amount * interest_rate / 100) ≥ (three_times * borrow_amount) ∧ x ≥ 20 :=
by
  sorry

end sarah_least_days_to_repay_at_least_three_times_l710_710213


namespace evaluate_f_29_4_and_41_6_l710_710847

def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then
  x * (1 - x)
else if 1 < x ∧ x ≤ 2 then
  Real.sin (π * x)
else
  sorry

theorem evaluate_f_29_4_and_41_6 :
  f (29 / 4) + f (41 / 6) = 5 / 16 :=
sorry

end evaluate_f_29_4_and_41_6_l710_710847


namespace lcm_perfect_square_l710_710191

-- Define the conditions and the final statement in Lean 4
theorem lcm_perfect_square (a b : ℕ) 
  (h: (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : 
  ∃ k : ℕ, lcm a b = k^2 :=
sorry

end lcm_perfect_square_l710_710191


namespace parallel_vectors_m_eq_neg3_l710_710490

theorem parallel_vectors_m_eq_neg3 :
  ∀ (m: ℝ),
  let a := (1, -2) in
  let b := (1 + m, 1 - m) in
  let parallel := (λ (a b: ℝ × ℝ), a.1 * b.2 - a.2 * b.1 = 0) in
  parallel a b →
  m = -3 :=
begin
  intros m a b parallel h,
  rw [prod.mk.eta] at *,

  sorry
end

end parallel_vectors_m_eq_neg3_l710_710490


namespace b_arithmetic_a_formula_l710_710002

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l710_710002


namespace division_identity_l710_710782

theorem division_identity : 45 / 0.05 = 900 :=
by
  sorry

end division_identity_l710_710782


namespace original_cost_price_is_correct_l710_710689

noncomputable def original_cost_price : ℝ :=
  let D_price := 400
  let profit_A_to_B := 0.20
  let profit_B_to_C := 0.25
  let profit_C_to_D := 0.30
  let price_multiplier_A_to_B := 1 + profit_A_to_B
  let price_multiplier_B_to_C := 1 + profit_B_to_C
  let price_multiplier_C_to_D := 1 + profit_C_to_D
  let final_multiplier := price_multiplier_A_to_B * price_multiplier_B_to_C * price_multiplier_C_to_D
  D_price / final_multiplier

theorem original_cost_price_is_correct :
  original_cost_price = 205.13 :=
by
  unfold original_cost_price
  have final_multiplier_value : 1.20 * 1.25 * 1.30 = 1.95 := by norm_num
  rw final_multiplier_value
  have price_correct : 400 / 1.95 = 205.13 := by norm_num
  exact price_correct

end original_cost_price_is_correct_l710_710689


namespace total_siblings_weight_l710_710388

variable (Antonio_weight : ℕ) (sister_diff : ℕ)
def sister_weight := Antonio_weight - sister_diff
def total_weight := Antonio_weight + sister_weight Antonio_weight sister_diff

theorem total_siblings_weight 
  (h_Antonio_weight : Antonio_weight = 50) 
  (h_sister_diff : sister_diff = 12) : 
  total_weight Antonio_weight sister_diff = 88 := 
by
  rw [total_weight, sister_weight, h_Antonio_weight, h_sister_diff]
  norm_num
  sorry

end total_siblings_weight_l710_710388


namespace b_arithmetic_a_formula_l710_710007

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l710_710007


namespace triangle_CDP_area_l710_710590

-- Definitions based on conditions
variables AB AD : ℝ
def M := AD / 2
def N := AB / 3
def Area_ABCD := AB * AD

-- Areas of triangles AND and DOC based on conditions
def Area_AND := (1 / 2) * N * AD
def OE := 18
def Area_DOC := (1 / 2) * AD * OE
def Area_BCON := Area_ABCD - Area_AND - Area_DOC

-- Given point P and BP bisects the area of BCON
-- Find Area of ∆CDP

def B := 52
def DC := AD
def PG := 13

noncomputable
def Area_CDP := (1 / 2) * DC * PG

-- The theorem to be proved
theorem triangle_CDP_area : AB = 84 → AD = 42 → Area_CDP = 546 := by
  assume h1 : AB = 84
  assume h2 : AD = 42
  sorry

end triangle_CDP_area_l710_710590


namespace range_of_k_l710_710823

variable (x k : ℝ)

def p : Prop := x ≥ k
def q : Prop := (2 - x) / (x + 1) < 0

theorem range_of_k :
  (∀ x, (p x k ↔ q x)) → k ∈ (Ioo 2 (⊤)) :=
by
  sorry

end range_of_k_l710_710823


namespace b_arithmetic_a_general_formula_l710_710011

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710011


namespace molecular_weight_CaOH2_l710_710427

def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

theorem molecular_weight_CaOH2 :
  (atomic_weight_Ca + 2 * atomic_weight_O + 2 * atomic_weight_H = 74.10) := 
by 
  sorry

end molecular_weight_CaOH2_l710_710427


namespace least_number_of_cubic_tiles_l710_710376

def room_length_cm := 624
def room_width_cm := 432
def room_height_cm := 356

def room_volume_cm3 := room_length_cm * room_width_cm * room_height_cm

def gcd_length_width_height := Nat.gcd (Nat.gcd room_length_cm room_width_cm) room_height_cm
def cubic_tile_side_length_cm := gcd_length_width_height
def cubic_tile_volume_cm3 := cubic_tile_side_length_cm ^ 3

def number_of_tiles := room_volume_cm3 / cubic_tile_volume_cm3

theorem least_number_of_cubic_tiles : 
  number_of_tiles = 1493952 := 
by 
  calc
  number_of_tiles = 1493952 : by sorry

end least_number_of_cubic_tiles_l710_710376


namespace distance_travelled_after_100_rotations_l710_710339

theorem distance_travelled_after_100_rotations
  (A B C : Point) (circle : ℝ)
  (h1 : B.center = circle) 
  (h2 : A.on_circumference circle)
  (h3 : C.on_circumference circle)
  (alpha : ℝ) 
  (h4 : 0 < alpha) 
  (h5 : alpha < π/3)
  (h6 : angle_ABC_eq_2alpha : ∠A B C = 2 * alpha) :
  total_distance_travelled A circle 100 = 22 * π * (1 + sin(alpha)) - 66 * alpha :=
sorry

end distance_travelled_after_100_rotations_l710_710339


namespace triangle_area_med_sum_l710_710834

-- Definitions based on the conditions
def triangle (A B C : Type) := Prop

def is_median (A B C M : Type) := Prop

def is_outside_triangle (X A B C : Type) := Prop 

def area (X A M : Type) := ℝ

-- The Lean theorem statement
theorem triangle_area_med_sum (A B C M N Q X : Type) 
  (h_triangle_ABC : triangle A B C) 
  (h_median_AM : is_median A B C M) 
  (h_median_BN : is_median B A C N)
  (h_median_CQ : is_median C A B Q) 
  (h_outside_X : is_outside_triangle X A B C) 
  : area X A M = area X B N + area X C Q := 
by sorry

end triangle_area_med_sum_l710_710834


namespace find_c5_l710_710635

noncomputable def c : ℕ → ℝ
| 0       := 1
| (n + 1) := (3 / 2) * c n + 2 * real.sqrt (9^n - (c n)^2)

def d : ℕ → ℝ
| 0       := 1 / 3^0
| (n + 1) := (1 / 2) * d n + (2 / 3) * real.sqrt (1 - (d n)^2)

theorem find_c5 (d5 : ℝ) (hd5 : d 5 = d5) : c 5 = 243 * d5 :=
sorry

end find_c5_l710_710635


namespace tournament_participant_count_l710_710907

theorem tournament_participant_count 
  (n : ℕ) 
  (matches : n * (n - 1) / 2 = 90 + (n - 10) * (n - 11)) : 
  n = 25 :=
sorry

end tournament_participant_count_l710_710907


namespace prove_option_d_l710_710821

-- Definitions of conditions
variables (a b : ℝ)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0)
variable (h_lt : a < b)

-- The theorem to be proved
theorem prove_option_d : a^3 < b^3 :=
sorry

end prove_option_d_l710_710821


namespace number_of_desks_l710_710725

theorem number_of_desks (accommodated_students_base6 : ℕ := 0) (seats_per_desk : ℕ := 3) : 
  let accommodated_students := 3 * 6^2 + 0 * 6^1 + 5 * 6^0 in
  let number_of_desks := if accommodated_students % seats_per_desk = 0 
                         then accommodated_students / seats_per_desk 
                         else accommodated_students / seats_per_desk + 1 in
  number_of_desks = 38 :=
by 
  sorry

end number_of_desks_l710_710725


namespace distinct_products_count_eq_380_l710_710558

noncomputable def set_T : finset ℕ :=
  {d | d ∣ 72000}.to_finset

def count_distinct_products (S : finset ℕ) : ℕ :=
  (S.product S).filter (λ p, p.fst < p.snd).card

theorem distinct_products_count_eq_380 :
  count_distinct_products set_T = 380 := by
  sorry

end distinct_products_count_eq_380_l710_710558


namespace range_of_omega_extremum_l710_710514

theorem range_of_omega_extremum (ω : ℝ) (hω : ω > 0) : 
  (∀ x ∈ Icc (-π / 6) (π / 3), Deriv (λ x, 2 * sin (ω * x)) x = 0 → x ∉ interior (Icc (-π / 6) (π / 3)))
  ↔ (3 / 2 < ω ∧ ω ≤ 3) :=
by
  sorry

end range_of_omega_extremum_l710_710514


namespace c_10_is_3_pow_89_l710_710175

noncomputable def sequence_c : ℕ → ℕ 
| 0         := 3 
| 1         := 6 
| (n + 2) := sequence_c (n + 1) * sequence_c n

theorem c_10_is_3_pow_89 : sequence_c 9 = 3 ^ 89 := 
sorry

end c_10_is_3_pow_89_l710_710175


namespace b_arithmetic_a_general_formula_l710_710009

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710009


namespace retailer_profit_percentage_l710_710734

def wholesale_price : ℝ := 99
def retail_price : ℝ := 132
def discount_percentage : ℝ := 0.10

def discount_amount (retail_price : ℝ) (discount_percentage : ℝ) : ℝ := retail_price * discount_percentage
def selling_price (retail_price : ℝ) (discount_amount : ℝ) : ℝ := retail_price - discount_amount
def profit (selling_price : ℝ) (wholesale_price : ℝ) : ℝ := selling_price - wholesale_price
def percentage_profit (profit : ℝ) (wholesale_price : ℝ) : ℝ := (profit / wholesale_price) * 100

theorem retailer_profit_percentage :
  percentage_profit (profit (selling_price retail_price (discount_amount retail_price discount_percentage)) wholesale_price) wholesale_price = 20 :=
  sorry

end retailer_profit_percentage_l710_710734


namespace b_arithmetic_a_general_formula_l710_710022

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710022


namespace bn_arithmetic_sequence_an_formula_l710_710028

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710028


namespace part1_arithmetic_sequence_part2_general_formula_l710_710067

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710067


namespace arithmetic_seq_a4_l710_710457

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions and the goal to prove
theorem arithmetic_seq_a4 (h₁ : is_arithmetic_sequence a d) (h₂ : a 2 + a 6 = 10) : 
  a 4 = 5 :=
by
  sorry

end arithmetic_seq_a4_l710_710457


namespace max_area_quadrilateral_12_l710_710534

theorem max_area_quadrilateral_12 (O : ℝ × ℝ) (d₁ d₂ : ℝ)
  (hO : O = (0, 0))
  (radius_O : ∀ P : ℝ × ℝ, (P.1 - O.1)^2 + (P.2 - O.2)^2 = 9)
  (perpendicular : d₁^2 + d₂^2 = 6)
  (foot_perpendicular : ∃ M : ℝ × ℝ, M = (1, sqrt 5) ∧ (O.1 - M.1)^2 + (O.2 - M.2)^2 = 6)
  (intersect_l1 : ∃ A C : ℝ × ℝ, (A = l₁ ∧ C = l₁) ∧ (A.1 - O.1)^2 + (A.2 - O.2)^2 = 9 ∧ (C.1 - O.1)^2 + (C.2 - O.2)^2 = 9)
  (intersect_l2 : ∃ B D : ℝ × ℝ, (B = l₂ ∧ D = l₂) ∧ (B.1 - O.1)^2 + (B.2 - O.2)^2 = 9 ∧ (D.1 - O.1)^2 + (D.2 - O.2)^2 = 9) :
  ∃ S : ℝ, S = 12 :=
sorry

end max_area_quadrilateral_12_l710_710534


namespace hyperbola_eccentricity_perpendicular_asymptotes_l710_710871

-- The problem statement with conditions and correct answer.
theorem hyperbola_eccentricity_perpendicular_asymptotes :
    ∀ b : ℝ, (b ≠ 0) → 
    (∀ x y : ℝ, ((x^2 / 144) - (y^2 / b^2) = 1) → 
    (b > 0) → 
    let a := 12 in
    let c := Real.sqrt (a^2 + b^2) in
    a = 12 ∧ b = 12 ∧ b ≠ 0 ∧ (b / a) * (-b / a) = -1 →
    c / a = Real.sqrt(2)) :=
begin
  intros b hneq x y heq hb,
  let a := 12,
  let c := Real.sqrt (a^2 + b^2),
  have ha : a = 12 := rfl,
  have hb_eq : b = 12, 
  { sorry },  -- This is given in the problem statement and not the steps.
  have hc : c = 12 * Real.sqrt(2), 
  { sorry },  -- Calculation from the conditions.
  show c / a = Real.sqrt 2,
  rw [hc, ha],
  norm_num,
end

end hyperbola_eccentricity_perpendicular_asymptotes_l710_710871


namespace bn_is_arithmetic_an_general_formula_l710_710107

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710107


namespace bn_is_arithmetic_an_general_formula_l710_710118

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710118


namespace extreme_values_for_a_eq_3_monotonic_intervals_l710_710479

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + (2 * a - 1) * x

-- Step 1: When a = 3
theorem extreme_values_for_a_eq_3 :
  (∃ x_max x_min : ℝ, x_max = -5 ∧ x_min = -1 ∧ 
    f x_max 3 = 25/3 ∧
    f x_min 3 = -7/3) :=
sorry

-- Step 2: Monotonic intervals for different values of a
theorem monotonic_intervals (a : ℝ) :
  (a < 1 → 
    (∀ x : ℝ, 
      (x < -1 → f' x a > 0) ∧
      (-1 < x ∧ x < -2 * a + 1 → f' x a < 0) ∧
      (x > -2 * a + 1 → f' x a > 0))) ∧
  (a = 1 →
    (∀ x : ℝ, f' x a ≥ 0)) ∧
  (a > 1 →
    (∀ x : ℝ, 
      (x < -2 * a + 1 → f' x a > 0) ∧
      (-2 * a + 1 < x ∧ x < -1 → f' x a < 0) ∧
      (x > -1 → f' x a > 0))) :=
sorry

end extreme_values_for_a_eq_3_monotonic_intervals_l710_710479


namespace sum_of_valid_m_l710_710896

theorem sum_of_valid_m : 
  (∑ m in Finset.filter 
    (λ m : ℤ, 
      (∃ x : ℤ, 6 * x - 5 ≥ m ∧ x < 5 / 2) ∧ 
      (y = m + 3 ∧ y ≥ 0)) 
    (Finset.Icc (-3) 1), m) = -5 := 
sorry

end sum_of_valid_m_l710_710896


namespace RelativelyPrimeProbability_l710_710294

def relatively_prime_probability_42 : Rat :=
  let n := 42
  let total := n
  let rel_prime_count := total - (21 + 14 + 6 - 7 - 3 - 2 + 1)
  let probability := (rel_prime_count : Rat) / total
  probability

theorem RelativelyPrimeProbability : relatively_prime_probability_42 = 2 / 7 :=
sorry

end RelativelyPrimeProbability_l710_710294


namespace eccentricity_of_ellipse_l710_710176

/-- Given an ellipse with equation x^2 / a^2 + y^2 / b^2 = 1 (a > b > 0), 
    foci F1 and F2, and point P on the ellipse such that ∠F1PF2 = π/3.
    Let the radii of the circumcircle and incircle of triangle F1PF2 be R and r respectively,
    and given R = 3r, prove that the eccentricity of the ellipse is 3/5.
-/
theorem eccentricity_of_ellipse
    (a b : ℝ) (h1 : a > b > 0)
    (F1 F2 P : ℝ × ℝ)
    (h2 : ∀ (x y : ℝ), (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2 / b)^2 = 1))
    (h3 : ∠ F1 P F2 = π / 3)
    (R r : ℝ) (h4 : R = 3 * r) :
    let c := sqrt (a^2 - b^2) in
    let e := c / a in
    e = 3 / 5 :=
sorry

end eccentricity_of_ellipse_l710_710176


namespace continuous_paths_count_l710_710497

-- Define the points in the figure
inductive Point : Type
| A | B | C | D | E | F | G

open Point

-- Define the segments connecting the points
def segment : Point → Point → Prop
| A, C | A, D | A, G | G, B | C, B | D, B | D, F | C, F | E, F | C, G | D, G := true
| _, _ := false

-- Define what it means for a path to be continuous and not revisit points
def continuous_path (path : List Point) : Prop :=
  ∀ p q, segment p q → (p ∈ path ∧ q ∈ path) ∧ (p ≠ q) ∧
  path.nodup ∧
  path.head = some A ∧
  path.last = some B

-- Prove that the number of such paths from A to B is 11
theorem continuous_paths_count : {p : List Point // continuous_path p}.card = 11 := sorry

end continuous_paths_count_l710_710497


namespace probability_at_least_one_three_equals_9_over_17_l710_710362

noncomputable def probability_at_least_one_three_given_sum_condition : ℚ :=
  let all_possible_tosses := (finset.range 7).bind (λ x4, 
    let possible_combinations := if x4 = 3 then [([1,1,1], 3)]
                                 else if x4 = 4 then [([1,1,2], 4), ([1,2,1], 4), ([2,1,1], 4)]
                                 else if x4 = 5 then [([1,1,3], 5), ([1,3,1], 5), ([3,1,1], 5),
                                                       ([1,2,2], 5), ([2,1,2], 5), ([2,2,1], 5)]
                                 else if x4 = 6 then [([1,2,3], 6), ([1,3,2], 6), ([2,1,3], 6),
                                                       ([2,3,1], 6), ([3,1,2], 6), ([3,2,1], 6),
                                                       ([2,2,2], 6)]
                                 else [] in
    finset.map (prod.map id id) (finset.powerset (finset.univ.filter (λ x, x ∈ possible_combinations))) )  in
  let total_outcomes := all_possible_tosses.card in
  let favorable_outcomes := all_possible_tosses.filter (λ toss, 3 ∈ toss.fst).card in
  favorable_outcomes / total_outcomes

theorem probability_at_least_one_three_equals_9_over_17 :
  probability_at_least_one_three_given_sum_condition = 9 / 17 :=
sorry

end probability_at_least_one_three_equals_9_over_17_l710_710362


namespace sequence_bn_arithmetic_and_an_formula_l710_710128

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710128


namespace symmetric_point_origin_l710_710244

theorem symmetric_point_origin (x y : ℤ) (hx : x = -2) (hy : y = 3) : 
    let symmetric_x := -x,
        symmetric_y := -y
    in (symmetric_x, symmetric_y) = (2, -3) :=
by {
  sorry
}

end symmetric_point_origin_l710_710244


namespace completing_the_square_l710_710673

theorem completing_the_square (x : ℝ) : 
  x^2 - 2 * x = 9 → (x - 1)^2 = 10 :=
by
  intro h
  sorry

end completing_the_square_l710_710673


namespace arithmetic_sequence_incorrect_option_l710_710527

theorem arithmetic_sequence_incorrect_option 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, S (n + 1) = S n + a (n + 1))
  (h2 : S 5 < S 6)
  (h3 : S 6 = S 7)
  (h4 : S 7 > S 8) :
  ¬ (S 9 > S 5) :=
begin
  sorry
end

end arithmetic_sequence_incorrect_option_l710_710527


namespace four_digit_number_impossible_l710_710404

theorem four_digit_number_impossible (X Y : ℕ) (hX : X ∈ {0, 1, 2, 3, 5, 7, 8, 9})
  (hY : Y ∈ {0, 1, 2, 3, 5, 7, 8, 9}) :
  ¬(∃ (a b c d : ℕ), {a, b, c, d} = {4, 6, X, Y} ∧ (1000*a + 100*b + 10*c + d = 46*(10*X + Y))) :=
sorry

end four_digit_number_impossible_l710_710404


namespace angle_line_plane_l710_710889

theorem angle_line_plane {l α : Type} (θ : ℝ) (h : θ = 150) : 
  ∃ φ : ℝ, φ = 60 := 
by
  -- This part would require the actual proof.
  sorry

end angle_line_plane_l710_710889


namespace sum_of_first_15_terms_is_largest_l710_710536

theorem sum_of_first_15_terms_is_largest
  (a : ℕ → ℝ)
  (s : ℕ → ℝ)
  (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : ∀ n, s n = n * a 1 + (n * (n - 1) * d) / 2)
  (h1: 13 * a 6 = 19 * (a 6 + 3 * d))
  (h2: a 1 > 0) : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≠ 15 → s 15 > s n :=
by
  sorry

end sum_of_first_15_terms_is_largest_l710_710536


namespace part1_arithmetic_sequence_part2_general_formula_l710_710065

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710065


namespace complex_solution_l710_710233

theorem complex_solution (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (Complex.mk a b)^2 = Complex.mk 3 4) :
  Complex.mk a b = Complex.mk 2 1 :=
sorry

end complex_solution_l710_710233


namespace problem_conditions_l710_710091

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710091


namespace part1_arithmetic_sequence_part2_general_formula_l710_710072

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710072


namespace parabola_hyperbola_focus_l710_710890

noncomputable def focus_left (p : ℝ) : ℝ × ℝ :=
  (-p / 2, 0)

theorem parabola_hyperbola_focus (p : ℝ) (hp : p > 0) : 
  focus_left p = (-2, 0) ↔ p = 4 :=
by 
  sorry

end parabola_hyperbola_focus_l710_710890


namespace min_sequence_6_moves_l710_710986

-- Define the initial conditions as assumptions
open Matrix

noncomputable def initial_4x4_grid : matrix (fin 4) (fin 4) bool :=
  λ _ _, tt  -- all counters are initially black side up

-- A move flips a 2x2 sub-square
def flip_2x2 (grid : matrix (fin 4) (fin 4) bool) (i j : fin 4) : matrix (fin 4) (fin 4) bool :=
  if i + 1 < 4 ∧ j + 1 < 4 then
    λ x y, if (x = i ∨ x = i + 1) ∧ (y = j ∨ y = j + 1) then ¬grid x y else grid x y
  else
    grid

-- Define the alternating pattern goal
def alternating_pattern : matrix (fin 4) (fin 4) bool :=
  λ i j, (i + j) % 2 = 0

-- Statement of the proof problem
theorem min_sequence_6_moves :
  ∃ moves : list (fin 4 × fin 4), moves.length = 6 ∧ 
    (moves.foldl (λ grid move, flip_2x2 grid move.1 move.2) initial_4x4_grid) = alternating_pattern :=
sorry

end min_sequence_6_moves_l710_710986


namespace min_8_minus_x_sq_geq_neg1_and_ctg_x_geq_neg1_total_length_l710_710788

noncomputable def interval_length (l : Real) (u : Real) : Real :=
  if l > u then 0 else u - l

theorem min_8_minus_x_sq_geq_neg1_and_ctg_x_geq_neg1_total_length :
  let lengths := [
    interval_length (-3) (-7 * Real.pi / 4),
    interval_length 0 (3 * Real.pi / 4)
  ]
  Real.sum lengths = 3 + Real.pi / 2 :=
by
  sorry

end min_8_minus_x_sq_geq_neg1_and_ctg_x_geq_neg1_total_length_l710_710788


namespace prob_relatively_prime_42_l710_710317

noncomputable def euler_totient (n : ℕ) : ℕ :=
  (List.range n).filter (λ i => Nat.gcd i n = 1).length

theorem prob_relatively_prime_42 : 
  (euler_totient 42 : ℚ) / 42 = 2 / 7 := 
by
  sorry

end prob_relatively_prime_42_l710_710317


namespace ellipse_chord_intersections_l710_710202

theorem ellipse_chord_intersections (n : ℕ) (h_n : n = 8) 
  (h_no_three_chords_intersect : ∀ (chords : set (set (ℝ × ℝ))), 
    (∀ chord ∈ chords, ∃ (p1 p2 : ℝ × ℝ), chord = {p1, p2}) → 
    (∀ p : ℝ × ℝ, p ∈ (⋂ c ∈ chords, c) → p.fst ∉ {p.snd | ∃ (c ∈ chords), c = chord})) :
  ∃ k : ℕ, k = 70 :=
by
  sorry

end ellipse_chord_intersections_l710_710202


namespace RelativelyPrimeProbability_l710_710295

def relatively_prime_probability_42 : Rat :=
  let n := 42
  let total := n
  let rel_prime_count := total - (21 + 14 + 6 - 7 - 3 - 2 + 1)
  let probability := (rel_prime_count : Rat) / total
  probability

theorem RelativelyPrimeProbability : relatively_prime_probability_42 = 2 / 7 :=
sorry

end RelativelyPrimeProbability_l710_710295


namespace problem_statement_l710_710163

-- Define the universal set U, set A, and set B
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

-- Define the complement of B in U
def C_U (B : Set ℕ) : Set ℕ := U \ B

-- Define the intersection of A and C_U(B)
def intersection : Set ℕ := A ∩ C_U(B)

-- State the theorem
theorem problem_statement : intersection = {2} := by
  sorry

end problem_statement_l710_710163


namespace percentage_increase_is_50_l710_710348

def initial : ℝ := 110
def final : ℝ := 165

theorem percentage_increase_is_50 :
  ((final - initial) / initial) * 100 = 50 := by
  sorry

end percentage_increase_is_50_l710_710348


namespace hall_area_l710_710733

theorem hall_area (L W : ℝ) 
  (h1 : W = (1/2) * L)
  (h2 : L - W = 8) : 
  L * W = 128 := 
  sorry

end hall_area_l710_710733


namespace rationalize_denominator_l710_710975

theorem rationalize_denominator :
  (1 / (Real.cbrt 3 - 2)) = -(Real.cbrt 9 + 2 * Real.cbrt 3 + 4) / 5 :=
by
  sorry

end rationalize_denominator_l710_710975


namespace exists_P_with_property_l710_710948

variable (S : Set (ℝ × ℝ)) (n : ℕ)
variable (h_card : S.card = n)
variable (h_collinear : ∀ (P1 P2 P3 : ℝ × ℝ), P1 ∈ S → P2 ∈ S → P3 ∈ S → ¬are_collinear P1 P2 P3)

def are_collinear (P1 P2 P3 : ℝ × ℝ) : Prop :=
  (P2.snd - P1.snd) * (P3.fst - P1.fst) = (P3.snd - P1.snd) * (P2.fst - P1.fst)

theorem exists_P_with_property :
  ∃ (P : Set (ℝ × ℝ)), P.card = 2 * n - 5 ∧ 
    ∀ (T : Set (ℝ × ℝ)), T ⊆ S → T.card = 3 → 
    ∃ (p : ℝ × ℝ), p ∈ T → p ∈ P → p ∈ interior (convex_hull ℝ T) := by
  sorry

end exists_P_with_property_l710_710948


namespace simplify_and_evaluate_expression_l710_710595

theorem simplify_and_evaluate_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 2) :
  x * y + (3 * x * y - 4 * x^2) - 2 * (x * y - 2 * x^2) = -4 :=
by 
  simp [h1, h2]
  sorry

end simplify_and_evaluate_expression_l710_710595


namespace find_c_l710_710474

theorem find_c (a b c : ℝ) : 
  (a * x^2 + b * x - 5) * (a * x^2 + b * x + 25) + c = (a * x^2 + b * x + 10)^2 → 
  c = 225 :=
by sorry

end find_c_l710_710474


namespace perfect_power_transfer_l710_710839

-- Given Conditions
variables {x y z : ℕ}

-- Definition of what it means to be a perfect seventh power
def is_perfect_seventh_power (n : ℕ) :=
  ∃ k : ℕ, n = k^7

-- The proof problem
theorem perfect_power_transfer 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h : is_perfect_seventh_power (x^3 * y^5 * z^6)) :
  is_perfect_seventh_power (x^5 * y^6 * z^3) := by
  sorry

end perfect_power_transfer_l710_710839


namespace b_arithmetic_sequence_general_formula_a_l710_710056

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710056


namespace max_binomial_coefficient_l710_710917

theorem max_binomial_coefficient (a b : ℕ) (n : ℕ) (sum_binom : (a + b)^n = 256) :
  binomial n (n / 2) = 70 := 
sorry

end max_binomial_coefficient_l710_710917


namespace math_problem_l710_710458

theorem math_problem 
  (a : Int) (b : Int) (c : Int)
  (h_a : a = -1)
  (h_b : b = 1)
  (h_c : c = 0) :
  a + c - b = -2 := 
by
  sorry

end math_problem_l710_710458


namespace calc_expression_three_digits_l710_710664

theorem calc_expression_three_digits :
  let x := (529 : ℝ) / (12 * (9 : ℝ)^(1/3) + 52 * (3 : ℝ)^(1/3) + 49)
  abs (x - 3.55) < 0.001 :=
by
  let A := 4
  let B := -4
  let C := 1
  let num := 529 : ℝ
  let denom := 12 * (9 : ℝ)^(1/3) + 52 * (3 : ℝ)^(1/3) + 49
  let rationalized := A * (9 : ℝ)^(1/3) + B * (3 : ℝ)^(1/3) + C
  let result := num / denom * rationalized
  calc abs (result - 3.55) < 0.001 := sorry

end calc_expression_three_digits_l710_710664


namespace distinguishable_dodecahedrons_l710_710656

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem distinguishable_dodecahedrons : 
  ∀ (number_of_colors : ℕ), number_of_colors = 12 →
  factorial (number_of_colors - 1) / 60 = 665280 :=
by
  assume (number_of_colors : ℕ)
  assume (h : number_of_colors = 12)
  sorry

end distinguishable_dodecahedrons_l710_710656


namespace goods_train_length_l710_710365

theorem goods_train_length
  (speed_kmph : ℕ) (speed_mps : ℕ) (platform_length : ℕ) (cross_time : ℕ) (train_length : ℕ) :
  speed_kmph = 72 → speed_mps = (72 * 5 / 18) → platform_length = 270 → cross_time = 26 →
  (speed_mps * cross_time - platform_length) = train_length → train_length = 250 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  rw h5
  sorry

end goods_train_length_l710_710365


namespace base_of_second_term_l710_710645

theorem base_of_second_term (e : ℕ) (base : ℝ) 
  (h1 : e = 35) 
  (h2 : (1/5)^e * base^18 = 1 / (2 * (10)^35)) : 
  base = 1/4 :=
by
  sorry

end base_of_second_term_l710_710645


namespace inequality_holds_for_all_real_l710_710438

open Real -- Open the real numbers namespace

theorem inequality_holds_for_all_real (x : ℝ) : 
  2^((sin x)^2) + 2^((cos x)^2) ≥ 2 * sqrt 2 :=
by
  sorry

end inequality_holds_for_all_real_l710_710438


namespace domain_y_eq_f_cos_x_sub_pi_over_3_range_of_a_l710_710482

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2)^2 - 2 * (Real.log x / Real.log (1/2)) + 1
noncomputable def g (x : ℝ) : ℝ := x^2 - a * x + 1

-- Part 1
theorem domain_y_eq_f_cos_x_sub_pi_over_3 :
  { x : ℝ | 2 * k * Real.pi - Real.pi / 6 < x ∧ x < 2 * k * Real.pi + 5 * Real.pi / 6, k ∈ ℤ } = 
  { x : ℝ | f (Real.cos (x - Real.pi / 3)) > 0 } :=
sorry

-- Part 2
theorem range_of_a :
  (∀ x1 ∈ Icc (1/8 : ℝ) 2, ∃! x0 ∈ Icc (-1 : ℝ) 2, f x1 = g x0) → (a ≤ -2 ∨ a > 5/2) :=
sorry

end domain_y_eq_f_cos_x_sub_pi_over_3_range_of_a_l710_710482


namespace sum_of_sequences_l710_710706

theorem sum_of_sequences :
  (1 + 11 + 21 + 31 + 41) + (9 + 19 + 29 + 39 + 49) = 250 := 
by 
  sorry

end sum_of_sequences_l710_710706


namespace map_coloring_theorem_l710_710349

-- Define the problem conditions
def regions : Nat := 26
def boy_short_of_one_paint : Prop := true
def no_black_and_white : Prop := true

-- Using the Four Color Theorem implicitly
theorem map_coloring_theorem 
  (regions : Nat)
  (boy_short_of_one_paint : Prop)
  (no_black_and_white : Prop) : 
  (number_of_colors_needed : Nat) 
  (has_colors : Nat) : 
  has_colors = 3 :=
by
  sorry

end map_coloring_theorem_l710_710349


namespace rectangle_A_plus_P_ne_162_l710_710372

theorem rectangle_A_plus_P_ne_162 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (A : ℕ) (P : ℕ) 
  (hA : A = a * b) (hP : P = 2 * a + 2 * b) : A + P ≠ 162 :=
by
  sorry

end rectangle_A_plus_P_ne_162_l710_710372


namespace problem_statement_l710_710648

theorem problem_statement :
  let s1 := (1 + 2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 2001 + 2002)
  let s2 := ∀ a b : ℤ, (odd a ∧ odd b) → odd ((a * a) * (a - b))
  let s3 := ∃ n : ℤ, (∑ i in (range (2002)), i + n)
  let s4 := ∃ a b : ℤ, (a + b) * (a - b) = 2002 
  (¬ ((s1 % 2 = 0) ∧ s2 ∧ s3 ∧ s4)) 
  :=
by
  sorry

end problem_statement_l710_710648


namespace complete_square_transform_l710_710681

theorem complete_square_transform (x : ℝ) : 
  x^2 - 2 * x = 9 ↔ (x - 1)^2 = 10 :=
by
  sorry

end complete_square_transform_l710_710681


namespace particle_paths_count_l710_710727

def is_valid_move (p q : ℕ × ℕ) : Prop :=
  q = (p.1 + 1, p.2) ∨ q = (p.1, p.2 + 1) ∨ q = (p.1 + 1, p.2 + 1)

def no_right_angle_turn (path : List (ℕ × ℕ)) : Prop :=
  ∀ (i : ℕ), i < path.length - 2 → ¬ ((path.get i.1, path.get i.2) ∧ (path.get (i + 2)).1 = path.get i.1 ∨ path.get (i + 2)).2 = path.get i.2

def is_valid_path (path : List (ℕ × ℕ)) : Prop :=
  path.head = (0,0) ∧ path.last = (5,5) ∧ (∀ i : ℕ, i < path.length - 1 → is_valid_move (path.get i) (path.get (i + 1))) ∧ no_right_angle_turn path

def count_paths : ℕ :=
  sorry -- This is the place where dynamic programming would go to count the paths

theorem particle_paths_count : count_paths = 252 :=
  sorry

end particle_paths_count_l710_710727


namespace curves_intersect_condition_l710_710323

noncomputable def curves_intersect_exactly_three_points (a : ℝ) : Prop :=
  ∃ x y : ℝ, 
    (x^2 + y^2 = a^2) ∧ (y = x^2 + a) ∧ 
    (y = a → x = 0) ∧ 
    ((2 * a + 1 < 0) → y = -(2 * a + 1) - 1)

theorem curves_intersect_condition (a : ℝ) : 
  curves_intersect_exactly_three_points a ↔ a < -1/2 :=
sorry

end curves_intersect_condition_l710_710323


namespace expression_evaluation_l710_710985

theorem expression_evaluation (a : ℝ) (h : a = Real.sqrt 2 - 3) : 
  (2 * a + Real.sqrt 3) * (2 * a - Real.sqrt 3) - 3 * a * (a - 2) + 3 = -7 :=
by
  sorry

end expression_evaluation_l710_710985


namespace bn_arithmetic_sequence_an_formula_l710_710027

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710027


namespace propositionA_necessary_but_not_sufficient_for_propositionB_l710_710955

-- Definitions for propositions and conditions
def propositionA (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0
def propositionB (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Theorem statement for the necessary but not sufficient condition
theorem propositionA_necessary_but_not_sufficient_for_propositionB (a : ℝ) :
  (propositionA a) → (¬ propositionB a) ∧ (propositionB a → propositionA a) :=
by
  sorry

end propositionA_necessary_but_not_sufficient_for_propositionB_l710_710955


namespace relatively_prime_probability_42_l710_710300

theorem relatively_prime_probability_42 : 
  (λ x, (x ≤ 42 ∧ x > 0 ∧ Nat.gcd x 42 = 1)) / (λ x, (x ≤ 42 ∧ x > 0)) = 2/7 :=
by 
  sorry

end relatively_prime_probability_42_l710_710300


namespace least_5_digit_divisible_by_12_15_18_l710_710692

theorem least_5_digit_divisible_by_12_15_18 : 
  ∃ n, n >= 10000 ∧ n < 100000 ∧ (180 ∣ n) ∧ n = 10080 :=
by
  -- Proof goes here
  sorry

end least_5_digit_divisible_by_12_15_18_l710_710692


namespace common_difference_of_AP_l710_710423

theorem common_difference_of_AP (a T_12 : ℝ) (d : ℝ) (n : ℕ) (h1 : a = 2) (h2 : T_12 = 90) (h3 : n = 12) 
(h4 : T_12 = a + (n - 1) * d) : d = 8 := 
by sorry

end common_difference_of_AP_l710_710423


namespace RelativelyPrimeProbability_l710_710296

def relatively_prime_probability_42 : Rat :=
  let n := 42
  let total := n
  let rel_prime_count := total - (21 + 14 + 6 - 7 - 3 - 2 + 1)
  let probability := (rel_prime_count : Rat) / total
  probability

theorem RelativelyPrimeProbability : relatively_prime_probability_42 = 2 / 7 :=
sorry

end RelativelyPrimeProbability_l710_710296


namespace probability_rel_prime_to_42_l710_710304

theorem probability_rel_prime_to_42 : 
  let n := 42 in
  let prime_factors := [2, 3, 7] in
  let relatively_prime_count := n * (1 - 1/prime_factors[0]) * (1 - 1/prime_factors[1]) * (1 - 1/prime_factors[2]) in
  let total_count := 42 in
  (relatively_prime_count / total_count) = 4 / 7 :=
by
  sorry

end probability_rel_prime_to_42_l710_710304


namespace siblings_total_weight_l710_710390

theorem siblings_total_weight (A_weight : ℕ) (S_diff : ℕ) (H : A_weight = 50) (H2 : S_diff = 12) :
  let S_weight := A_weight - S_diff in
  let total_weight := A_weight + S_weight in
  total_weight = 88 :=
by
  simp [H, H2]
  sorry

end siblings_total_weight_l710_710390


namespace axis_of_symmetry_shifted_cos_l710_710893

noncomputable def shifted_cos_axis_symmetry (x : ℝ) : Prop :=
  ∃ k : ℤ, x = k * (Real.pi / 2) - (Real.pi / 12)

theorem axis_of_symmetry_shifted_cos :
  shifted_cos_axis_symmetry x :=
sorry

end axis_of_symmetry_shifted_cos_l710_710893


namespace geometric_sequence_problem_l710_710829

noncomputable def a_n (n : ℕ) : ℕ :=
  if h : n > 0 then 2 * 3^(n-1) else 0

theorem geometric_sequence_problem :
  a_n 1 * nat.choose 6 0 - a_n 2 * nat.choose 6 1 + a_n 3 * nat.choose 6 2 - a_n 4 * nat.choose 6 3 +
  a_n 5 * nat.choose 6 4 - a_n 6 * nat.choose 6 5 + a_n 7 * nat.choose 6 6 = 128 :=
sorry

end geometric_sequence_problem_l710_710829


namespace area_bounded_by_arccos_0_0_l710_710755

theorem area_bounded_by_arccos_0_0 : 
  ∫ y in 0..1, real.cos y = 1 := 
sorry

end area_bounded_by_arccos_0_0_l710_710755


namespace lcm_is_perfect_square_l710_710193

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : ∃ k : ℕ, k^2 = Nat.lcm a b :=
by
  sorry

end lcm_is_perfect_square_l710_710193


namespace count_symmetric_in_punk_cd_for_sale_l710_710646

def is_symmetric (c : Char) : Prop :=
  c = 'U' ∨ c = 'O' ∨ c = 'A'

def count_symmetric_letters (s : String) : Nat :=
  s.toList.foldl (λ acc c => if is_symmetric c then acc + 1 else acc) 0

theorem count_symmetric_in_punk_cd_for_sale :
  count_symmetric_letters "PUNK CD FOR SALE" = 3 :=
by
  sorry

end count_symmetric_in_punk_cd_for_sale_l710_710646


namespace correct_choice_2point5_l710_710873

def set_M : Set ℝ := {x | -2 < x ∧ x < 3}

theorem correct_choice_2point5 : 2.5 ∈ set_M :=
by {
  -- sorry is added to close the proof for now
  sorry
}

end correct_choice_2point5_l710_710873


namespace sin_390_eq_half_l710_710779

theorem sin_390_eq_half : Float.sin (390 * Float.pi / 180) = 1 / 2 :=
  sorry

end sin_390_eq_half_l710_710779


namespace braden_total_amount_after_winning_l710_710396

noncomputable def initial_amount := 400
noncomputable def multiplier := 2

def total_amount_after_winning (initial: ℕ) (mult: ℕ) : ℕ := initial + (mult * initial)

theorem braden_total_amount_after_winning : total_amount_after_winning initial_amount multiplier = 1200 := by
  sorry

end braden_total_amount_after_winning_l710_710396


namespace find_number_l710_710719

theorem find_number (Number : ℝ) (h : Number / 5 = 30 / 600) : Number = 1 / 4 :=
by sorry

end find_number_l710_710719


namespace arithmetic_sequence_primes_l710_710604

theorem arithmetic_sequence_primes (a : ℕ) (d : ℕ) (primes_seq : ∀ n : ℕ, n < 15 → Nat.Prime (a + n * d))
  (distinct_primes : ∀ m n : ℕ, m < 15 → n < 15 → m ≠ n → a + m * d ≠ a + n * d) :
  d > 30000 := 
sorry

end arithmetic_sequence_primes_l710_710604


namespace complete_square_l710_710676

theorem complete_square (x : ℝ) : 
  (x ^ 2 - 2 * x = 9) -> ((x - 1) ^ 2 = 10) :=
by
  intro h
  rw [← add_zero (x ^ 2 - 2 * x), ← add_zero (10)]
  calc
    x ^ 2 - 2 * x = 9                   : by rw [h]
             ...  = (x ^ 2 - 2 * x + 1 - 1) : by rw [add_sub_cancel, add_zero]
             ...  = (x - 1) ^ 2 - 1     : by 
                           { rw [sub_eq_add_neg], exact add_sub_cancel _ _}
             ...  = 10 - 1              : by rw [h]
             ...  = 10                  : by rw (sub_sub_cancel)
 

end complete_square_l710_710676


namespace b_arithmetic_sequence_general_formula_a_l710_710058

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710058


namespace symmetry_axis_is_neg_pi_over_12_l710_710272

noncomputable def symmetry_axis_of_sine_function : Prop :=
  ∃ k : ℤ, ∀ x : ℝ, (3 * x + 3 * Real.pi / 4 = Real.pi / 2 + k * Real.pi) ↔ (x = - Real.pi / 12 + k * Real.pi / 3)

theorem symmetry_axis_is_neg_pi_over_12 : symmetry_axis_of_sine_function := sorry

end symmetry_axis_is_neg_pi_over_12_l710_710272


namespace sequence_term_formula_l710_710736

theorem sequence_term_formula (C : ℕ → ℝ) (h : ∀ n, (∑ i in Finset.range n, (1 / C (i + 1))) / n = 1 / (2 * n + 1)) :
  ∀ n, C n = 4 * n - 1 :=
by
  sorry

end sequence_term_formula_l710_710736


namespace rectangle_area_l710_710625

theorem rectangle_area (x y : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) (h3 : 1 ≤ y) (h4 : y ≤ 9)
  (h5 : (1000 * x + 100 * x + 10 * y + y).nat_digits 10 3 = (1000 * x + 100 * x + 10 * y + y).nat_digits 10 2)
  (h6 : ∃ k, 1000 * x + 100 * x + 10 * y + y = k^2) :
  x * y = 28 :=
begin
  sorry
end

end rectangle_area_l710_710625


namespace transport_cost_l710_710992

theorem transport_cost : 
  (cost_per_kg cost weight_in_grams : ℝ) 
  (weight_in_kg transport_cost : ℝ)
  (h1 : cost_per_kg = 18000)
  (h2 : weight_in_grams = 400)
  (h3 : weight_in_kg = weight_in_grams / 1000)
  (h4 : transport_cost = weight_in_kg * cost_per_kg) :
  transport_cost = 7200 :=
by
  sorry

end transport_cost_l710_710992


namespace part1_arithmetic_sequence_part2_general_formula_l710_710093

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710093


namespace arithmetic_sum_l710_710853

theorem arithmetic_sum :
  ∀ (a : ℕ → ℝ),
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →
  (∃ x : ℝ, ∃ y : ℝ, x^2 - 6 * x - 1 = 0 ∧ y^2 - 6 * y - 1 = 0 ∧ x = a 3 ∧ y = a 15) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by
  intros a h_arith_seq h_roots
  sorry

end arithmetic_sum_l710_710853


namespace horizontal_asymptote_exists_x_intercepts_are_roots_l710_710957

noncomputable def given_function (x : ℝ) : ℝ :=
  (15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5) / (5 * x^5 + 3 * x^3 + 9 * x^2 + 2 * x + 4)

theorem horizontal_asymptote_exists :
  ∃ L : ℝ, ∀ x : ℝ, (∃ M : ℝ, M > 0 ∧ (∀ x > M, abs (given_function x - L) < 1)) ∧ L = 0 := 
sorry

theorem x_intercepts_are_roots :
  ∀ y, y = 0 ↔ ∃ x : ℝ, x ≠ 0 ∧ 15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5 = 0 :=
sorry

end horizontal_asymptote_exists_x_intercepts_are_roots_l710_710957


namespace perpendicular_planes_k_value_l710_710851

theorem perpendicular_planes_k_value :
  let a := (1, 2, -2 : ℝ × ℝ × ℝ)
  let b := (-2, -4, k : ℝ × ℝ × ℝ)
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0) → k = -5 :=
by
  intro h
  sorry

end perpendicular_planes_k_value_l710_710851


namespace b_arithmetic_a_formula_l710_710000

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l710_710000


namespace part1_sequence_arithmetic_part2_general_formula_l710_710142

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710142


namespace tangent_points_arc_of_circle_l710_710971

noncomputable def inscribed_tangent_points (S : Circle) (A B : Point) (hAB : Chord S A B) : Set Point :=
  { M | ∃ S1 S2 : Circle, inscribed_in_segment S1 S A B ∧ inscribed_in_segment S2 S A B ∧ tangent S1 S2 M }

theorem tangent_points_arc_of_circle 
  (S : Circle) (A B : Point) (hAB : Chord S A B) :
  ∃ arc : Subset Point, arc_is_circle S arc ∧ ∀ M ∈ inscribed_tangent_points S A B hAB, M ∈ arc :=
sorry

end tangent_points_arc_of_circle_l710_710971


namespace percent_in_quarters_l710_710329

noncomputable def percent_quarters_value (Dimes Quarters : Nat) (value_dime value_quarter : ℕ) :=
  let total_value := (Dimes * value_dime) + (Quarters * value_quarter)
  let quarters_value := Quarters * value_quarter
  (quarters_value * 100) / total_value

theorem percent_in_quarters (Dimes Quarters value_dime value_quarter : ℕ) :
  Dimes = 40 → Quarters = 30 → value_dime = 10 → value_quarter = 25 →
  percent_quarters_value Dimes Quarters value_dime value_quarter = 65.22 := by
    intros hD hQ hv_d hv_q
    unfold percent_quarters_value
    simp [hD, hQ, hv_d, hv_q]
    sorry

end percent_in_quarters_l710_710329


namespace number_of_rational_terms_l710_710671

theorem number_of_rational_terms (x y : ℚ) :
  (∑ k in Finset.range 501,
    if ((k % 4 = 0) ∧ ((500 - k) % 2 = 0))
    then 1 else 0) = 126
:= sorry

end number_of_rational_terms_l710_710671


namespace jo_climb_10_stairs_l710_710935

def g : ℕ → ℕ
| 0       := 1
| 1       := 1
| 2       := 2
| 3       := 4
| (n + 4) := g (n + 3) + g (n + 2) + g (n + 1) + g n

theorem jo_climb_10_stairs :
  g 10 = 401 :=
by {
  -- Proof here
  sorry
}

end jo_climb_10_stairs_l710_710935


namespace find_k_l710_710476

theorem find_k (k : ℝ) : 
  (∀ α β : ℝ, (α * β = 15 ∧ α + β = -k ∧ (α + 3 + β + 3 = k)) → k = 3) :=
by 
  sorry

end find_k_l710_710476


namespace even_sum_of_digits_residue_l710_710943

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem even_sum_of_digits_residue (k : ℕ) (h : 2 ≤ k) (r : ℕ) (hr : r < k) :
  ∃ n : ℕ, sum_of_digits n % 2 = 0 ∧ n % k = r := 
sorry

end even_sum_of_digits_residue_l710_710943


namespace handkerchief_passing_methods_l710_710441

-- Definitions based on conditions
def Student := {A, B, C, D} -- Set of students
def passes (start end_ : Student) (n : ℕ) : Prop :=
  -- This is a placeholder definition to represent the passing process
  sorry

-- Statement of the problem
theorem handkerchief_passing_methods :
  ∃ methods : finset (list Student),
    methods.card = 60 ∧
    ∀ m ∈ methods, m.head = A ∧ m.last = A ∧ (list.length m) = 6 := -- 5 passes imply list length 6 (start and 5 passes)
sorry

end handkerchief_passing_methods_l710_710441


namespace range_of_m_f_gt_g_l710_710870

noncomputable def f (x : ℝ) : ℝ := real.exp x * real.sin x - real.cos x
noncomputable def g (x : ℝ) : ℝ := x * real.cos x - real.sqrt 2 * real.exp x

theorem range_of_m :
  ∀ (x1 : ℝ), x1 ∈ Icc 0 (real.pi / 2) →
  ∃ (x2 : ℝ), x2 ∈ Icc 0 (real.pi / 2) ∧
  f x1 + g x2 ≥ - 1 - real.sqrt 2 :=
sorry

theorem f_gt_g (x : ℝ) (hx : x > -1) : f x > g x :=
sorry

end range_of_m_f_gt_g_l710_710870


namespace T_C_not_solved_algorithmically_l710_710742

noncomputable def T_A := ∑ i in finset.range 101, i^2
noncomputable def T_B := ∑ i in finset.range (50 - 1), 1 / (i + 2)
noncomputable def T_C := ∑' i, i
noncomputable def T_D := ∑ i in finset.range 100, if i % 2 = 0 then i + 1 else -(i + 1)

theorem T_C_not_solved_algorithmically :
  ¬ computable T_C :=
sorry

end T_C_not_solved_algorithmically_l710_710742


namespace geometric_sequence_first_term_l710_710777

theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) 
  (h1 : r = 2) (h2 : a * r^5 = 48) : a = 3 / 2 := 
begin
  sorry
end

end geometric_sequence_first_term_l710_710777


namespace triangle_construction_possible_l710_710771

theorem triangle_construction_possible (r l_alpha k_alpha : ℝ) (h1 : r > 0) (h2 : l_alpha > 0) (h3 : k_alpha > 0) :
  l_alpha^2 < (4 * k_alpha^2 * r^2) / (k_alpha^2 + r^2) :=
sorry

end triangle_construction_possible_l710_710771


namespace minimum_production_quantity_l710_710274

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the revenue function given the selling price per unit
def revenue (x : ℝ) : ℝ := 25 * x

-- Define the interval for x
def x_range (x : ℝ) : Prop := 0 < x ∧ x < 240

-- State the minimum production quantity required to avoid a loss
theorem minimum_production_quantity (x : ℝ) (h : x_range x) : 150 <= x :=
by
  -- Sorry replaces the detailed proof steps
  sorry

end minimum_production_quantity_l710_710274


namespace correct_statements_sequence_l710_710865

def f (x : ℝ) : ℝ := |Real.cos x| * Real.sin x

theorem correct_statements_sequence :
  f(2014 * Real.pi / 3) = -Real.sqrt 3 / 4 ∧
  ¬(∀ x, f (x + Real.pi) = f x) ∧
  MonotoneOn f (Set.Icc (-Real.pi / 4) (Real.pi / 4)) ∧
  ¬ ∃ y, f (2 * y) = -f (2 * y - (Real.pi / 2)) :=
by 
  sorry

end correct_statements_sequence_l710_710865


namespace part1_arithmetic_sequence_part2_general_formula_l710_710075

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710075


namespace possible_values_of_a_l710_710868

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a + b * Real.cos x + c * Real.sin x

theorem possible_values_of_a 
    (a b c : ℝ) 
    (h1 : f a b c (Real.pi / 2) = 1) 
    (h2 : f a b c Real.pi = 1) 
    (h3 : ∀ x : ℝ, |f a b c x| ≤ 2) :
    4 - 3 * Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2 :=
by
  sorry

end possible_values_of_a_l710_710868


namespace tournament_players_l710_710908

theorem tournament_players (n : ℕ) :
  (∃ k : ℕ, k = n + 12 ∧
    -- Exactly one-third of the points earned by each player were earned against the twelve players with the least number of points.
    (2 * (1 / 3 * (n * (n - 1) / 2)) + 2 / 3 * 66 + 66 = (k * (k - 1)) / 2) ∧
    --- Solving the quadratic equation derived
    (n = 4)) → 
    k = 16 :=
by
  sorry

end tournament_players_l710_710908


namespace find_number_l710_710419

theorem find_number:
  ∃ x : ℝ, 0 < x ∧ 
          let y := swap_first_fifth_digits x in 
          y = 2501 * x ∧ 
          x = 0.000279972 := 
sorry

noncomputable def swap_first_fifth_digits (x : ℝ) : ℝ :=
  -- Function to swap the first and fifth digits after the decimal point of x
  sorry

end find_number_l710_710419


namespace find_acute_angle_l710_710877

def vector_a (α : ℝ) : ℝ × ℝ := (3/2, Real.sin α)
def vector_b (α : ℝ) : ℝ × ℝ := (Real.cos α, 1/3)

def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_acute_angle (α : ℝ) (h : parallel (vector_a α) (vector_b α)) : α = π / 4 :=
by
  -- Proof goes here
  sorry

end find_acute_angle_l710_710877


namespace part1_sequence_arithmetic_part2_general_formula_l710_710143

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710143


namespace b_arithmetic_a_general_formula_l710_710017

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710017


namespace selection_including_at_least_one_female_l710_710979

-- Define the number of male students, female students, and total representatives
def M : ℕ := 6
def F : ℕ := 3
def T : ℕ := 4

-- Define the problem statement
theorem selection_including_at_least_one_female :
  ∑ k in finset.range (M + 1), (nat.choose M T) - ∑ j in finset.range (F + 1), (nat.choose (M + F) T) = 111 := by
  sorry

end selection_including_at_least_one_female_l710_710979


namespace max_a_l710_710462

-- Define the function is odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function is increasing on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

-- Define the condition that f(x^2 + ax + a) ≤ f(-at^2 - t + 1) for x, t ∈ [1, 2]
def condition3 (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x t, (1 ≤ x ∧ x ≤ 2) → (1 ≤ t ∧ t ≤ 2) → 
  f (x^2 + a*x + a) ≤ f (-a*(t^2) - t + 1)

-- Prove that the maximum value of 'a' such that all conditions hold is -1
theorem max_a (f : ℝ → ℝ) : 
  is_odd f → increasing_on_nonneg f → condition3 f (-1) :=
begin
  sorry
end

end max_a_l710_710462


namespace variance_linear_transform_l710_710468

variable {X : Type} [RandomVariable X]
variable (D : X → ℝ)

-- Given condition
axiom D_X : D X = 3

-- To prove
theorem variance_linear_transform :
  D (3 * X + 2) = 27 := by
  sorry

end variance_linear_transform_l710_710468


namespace foreign_objects_total_l710_710749

theorem foreign_objects_total (burrs ticks : ℕ) (h1 : burrs = 12) (h2 : ticks = 6 * burrs) : burrs + ticks = 84 :=
by {
  subst h1,
  subst h2,
  simp,
}

end foreign_objects_total_l710_710749


namespace project_scheduling_count_l710_710359

-- Definitions of the constraints
def project_schedule := list char

def valid_project_schedule (s : project_schedule) : Prop :=
  ∃ n1 n2 n3 n4 n5 n6 : ℕ,
  s = ['A', 'B', 'C', 'D', 'E', 'F'] ∧ n1 < n2 ∧ n2 < n3 ∧ n3 = n4 - 1

-- The main theorem statement
theorem project_scheduling_count :
  ∃ (count : ℕ), count = 20 ∧
  ∀ (s : project_schedule), valid_project_schedule s → count = 20 :=
by sorry

end project_scheduling_count_l710_710359


namespace train_present_when_maria_arrives_l710_710740

noncomputable def probability_train_present : ℝ :=
  let train_arrival := (interval 0 120)
  let maria_arrival := (interval 0 120)
  let overlap_area := 3150
  let total_area := 14400
  (overlap_area : ℝ) / total_area

theorem train_present_when_maria_arrives :
  probability_train_present = 7 / 32 :=
by
  sorry

end train_present_when_maria_arrives_l710_710740


namespace triangle_area_is_72_l710_710929

noncomputable def area_of_triangle (AB m_a m_b : ℝ) : ℝ :=
  let m_c := 15 in -- Median from vertex C calculated from Stewart's Theorem and other steps
  let s_m := (m_a + m_b + m_c) / 2 in
  (4 / 3) * Real.sqrt (s_m * (s_m - m_a) * (s_m - m_b) * (s_m - m_c))

theorem triangle_area_is_72 (AB m_a m_b : ℝ) (hAB : AB = 10) (hm_a : m_a = 9) (hm_b : m_b = 12) :
  area_of_triangle AB m_a m_b = 72 :=
by {
  rw [hAB, hm_a, hm_b],
  sorry
}

end triangle_area_is_72_l710_710929


namespace bn_arithmetic_sequence_an_formula_l710_710026

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710026


namespace total_flour_used_l710_710686

def wheat_flour : ℝ := 0.2
def white_flour : ℝ := 0.1

theorem total_flour_used : wheat_flour + white_flour = 0.3 :=
by
  sorry

end total_flour_used_l710_710686


namespace part1_sequence_arithmetic_part2_general_formula_l710_710144

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710144


namespace ratio_sum_l710_710911

-- Define the problem setup
structure Rectangle :=
  (A B C D E F P Q : Type)
  (AB BC BE EF FC : ℝ)
  (hAB : AB = 8)
  (hBC : BC = 2)
  (hBE : BE = BC / 3)
  (hEF : EF = BC / 3)
  (hFC : FC = BC / 3)

-- Define the theorem to prove the sum of the ratio components
theorem ratio_sum (rect : Rectangle) : 
  let BP := 1, PQ := 1, QD := 1 in BP + PQ + QD = 3 :=
by
  sorry

end ratio_sum_l710_710911


namespace chromatic_number_le_max_degree_add_one_l710_710169

open GraphTheory

variables (G : simple_graph V) (D : ℕ) [G.is_vertex_labeled]

-- Define the maximum degree condition
def max_degree (D : ℕ) := ∀ v : V, G.degree v ≤ D

-- Define the chromatic number definition
def chromatic_number (G : simple_graph V) := finset.univ.card

theorem chromatic_number_le_max_degree_add_one (hG : max_degree G D) : chromatic_number G ≤ D + 1 := 
sorry

end chromatic_number_le_max_degree_add_one_l710_710169


namespace strawberries_eaten_l710_710579

-- Define the initial number of strawberries.
def initial_strawberries : ℝ := 78.0

-- Define the remaining number of strawberries.
def remaining_strawberries : ℝ := 36.0

-- The statement we want to prove.
theorem strawberries_eaten :
  initial_strawberries - remaining_strawberries = 42 := by
  calc
    initial_strawberries - remaining_strawberries = 78.0 - 36.0 : by rfl
    ... = 42 : by norm_num

end strawberries_eaten_l710_710579


namespace b_seq_arithmetic_a_seq_formula_l710_710155

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710155


namespace bn_is_arithmetic_an_general_formula_l710_710120

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710120


namespace part1_sequence_arithmetic_part2_general_formula_l710_710138

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710138


namespace statement_b_statement_d_l710_710684

-- Noncomputable to handle real number calculations
noncomputable section

-- Statement B: Given the conditions, prove P(B|A) = 2/3
theorem statement_b (P A B : Prop) (P_A : ℝ) (P_B : ℝ) (P_A_given_B : ℝ)
  (ha : P_A = 0.6)
  (hb : P_B = 0.8)
  (hc : P_A_given_B = 0.5) :
  P(A ∧ B) = P_B * P_A_given_B ∧
  P B | A = (P(A ∧ B)) / P(A) :=
begin
  sorry
end

-- Statement D: Given the conditions, prove values of c and k
theorem statement_d (c k : ℝ) (z y : ℝ → ℝ) (hx : ℝ)
  (hr : ∀ x, z = log y ∧ z = 4 * x + 0.3) :
  c = exp 0.3 ∧ k = 4 :=
begin
  sorry
end

end statement_b_statement_d_l710_710684


namespace midpoints_collinear_circle_center_l710_710586

theorem midpoints_collinear_circle_center {A B C D O K L : Point} 
  (ω ω₁ ω₂ : Circle) 
  (hᴏcirc : quadrilateral_circumscribed_about_circle ABCD ω)
  (hA_BO_CD : lines_intersect_at O A B C D)
  (h₁ : tangent_to_sides ω₁ B C A B C D K)
  (h₂ : tangent_to_sides ω₂ A D A B C D L) 
  (collinear_OKL : collinear O K L) : 
  collinear (midpoint B C) (midpoint A D) (center ω) := sorry

end midpoints_collinear_circle_center_l710_710586


namespace product_even_probability_l710_710235

theorem product_even_probability : 
  let S := {x // 4 ≤ x ∧ x ≤ 20 ∧ x ≠ 13}
  let total_choices := (S.card.choose 2)
  let even_product_choices := (S.filter (λ p, p.1 * p.2 % 2 = 0)).card
  total_choices = nat.choose 16 2 ∧ even_product_choices = 99 
  → even_product_choices / total_choices = 33 / 40 :=
by
  sorry

end product_even_probability_l710_710235


namespace four_digit_even_numbers_divisible_by_4_l710_710496

noncomputable def number_of_4_digit_even_numbers_divisible_by_4 : Nat :=
  500

theorem four_digit_even_numbers_divisible_by_4 : 
  (∃ count : Nat, count = number_of_4_digit_even_numbers_divisible_by_4) :=
sorry

end four_digit_even_numbers_divisible_by_4_l710_710496


namespace sheela_total_income_l710_710217

-- Define the monthly income as I
def monthly_income (I : Real) : Prop :=
  4500 = 0.28 * I

-- Define the annual income computed from monthly income
def annual_income (I : Real) : Real :=
  I * 12

-- Define the interest earned from savings account 
def interest_savings (principal : Real) (monthly_rate : Real) : Real :=
  principal * (monthly_rate * 12)

-- Define the interest earned from fixed deposit
def interest_fixed (principal : Real) (annual_rate : Real) : Real :=
  principal * annual_rate

-- Overall total income after one year calculation
def overall_total_income (annual_income : Real) (interest_savings : Real) (interest_fixed : Real) : Real :=
  annual_income + interest_savings + interest_fixed

-- Given conditions
variable (I : Real)
variable (principal_savings : Real := 4500)
variable (principal_fixed : Real := 3000)
variable (monthly_rate_savings : Real := 0.02)
variable (annual_rate_fixed : Real := 0.06)

-- Theorem statement to be proved
theorem sheela_total_income :
  monthly_income I →
  overall_total_income (annual_income I) 
                      (interest_savings principal_savings monthly_rate_savings)
                      (interest_fixed principal_fixed annual_rate_fixed)
  = 194117.16 :=
by
  sorry

end sheela_total_income_l710_710217


namespace units_digit_calculation_l710_710430

-- Define a function to compute the units digit of a number in base 10
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_calculation :
  units_digit (8 * 18 * 1988 - 8^3) = 0 := by
  sorry

end units_digit_calculation_l710_710430


namespace fraction_of_red_knights_magical_l710_710901

variable {knights : ℕ}
variable {red_knights : ℕ}
variable {blue_knights : ℕ}
variable {magical_knights : ℕ}
variable {magical_red_knights : ℕ}
variable {magical_blue_knights : ℕ}

axiom total_knights : knights > 0
axiom red_knights_fraction : red_knights = (3 * knights) / 8
axiom blue_knights_fraction : blue_knights = (5 * knights) / 8
axiom magical_knights_fraction : magical_knights = knights / 4
axiom magical_fraction_relation : 3 * magical_blue_knights = magical_red_knights

theorem fraction_of_red_knights_magical :
  (magical_red_knights : ℚ) / red_knights = 3 / 7 :=
by
  sorry

end fraction_of_red_knights_magical_l710_710901


namespace find_a2_l710_710484

noncomputable def a_sequence (k : ℕ+) (n : ℕ) : ℚ :=
  -(1 / 2 : ℚ) * n^2 + k * n

theorem find_a2
  (k : ℕ+)
  (max_S : ∀ n : ℕ, a_sequence k n ≤ 8)
  (max_reached : ∃ n : ℕ, a_sequence k n = 8) :
  a_sequence 4 2 - a_sequence 4 1 = 5 / 2 :=
by
  -- To be proved, insert appropriate steps here
  sorry

end find_a2_l710_710484


namespace range_of_x_l710_710508

theorem range_of_x (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x > -(1 / 2) :=
sorry

end range_of_x_l710_710508


namespace swapped_digit_number_l710_710723

theorem swapped_digit_number (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  10 * b + a = new_number :=
sorry

end swapped_digit_number_l710_710723


namespace inlet_fill_rate_correct_l710_710368

def leak_rate (tank_volume : ℕ) (empty_time : ℕ) : ℕ :=
  tank_volume / empty_time

def net_empty_rate (tank_volume : ℕ) (combined_time : ℕ) : ℕ :=
  tank_volume / combined_time

def inlet_fill_rate (leak_rate : ℕ) (net_rate : ℕ) : ℕ :=
  net_rate + leak_rate

theorem inlet_fill_rate_correct (tank_volume : ℕ) (leak_empty_time : ℕ) (combined_empty_time : ℕ) (expected_rate : ℕ) :
  let L := leak_rate tank_volume leak_empty_time in
  let net_rate := net_empty_rate tank_volume combined_empty_time in
  let F := inlet_fill_rate L net_rate in
  F / 60 = expected_rate :=
by
  sorry

-- Given data
example : inlet_fill_rate_correct 1440 3 12 10 :=
by
  sorry

end inlet_fill_rate_correct_l710_710368


namespace conversion_problem_l710_710417

noncomputable def conversion1 : ℚ :=
  35 * (1/1000)  -- to convert cubic decimeters to cubic meters

noncomputable def conversion2 : ℚ :=
  53 * (1/60)  -- to convert seconds to minutes

noncomputable def conversion3 : ℚ :=
  5 * (1/60)  -- to convert minutes to hours

noncomputable def conversion4 : ℚ :=
  1 * (1/100)  -- to convert square centimeters to square decimeters

noncomputable def conversion5 : ℚ :=
  450 * (1/1000)  -- to convert milliliters to liters

theorem conversion_problem : 
  (conversion1 = 7 / 200) ∧ 
  (conversion2 = 53 / 60) ∧ 
  (conversion3 = 1 / 12) ∧ 
  (conversion4 = 1 / 100) ∧ 
  (conversion5 = 9 / 20) :=
by
  sorry

end conversion_problem_l710_710417


namespace bn_arithmetic_sequence_an_formula_l710_710025

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710025


namespace select_best_player_l710_710440

theorem select_best_player : 
  (average_A = 9.6 ∧ variance_A = 0.25) ∧ 
  (average_B = 9.5 ∧ variance_B = 0.27) ∧ 
  (average_C = 9.5 ∧ variance_C = 0.30) ∧ 
  (average_D = 9.6 ∧ variance_D = 0.23) → 
  best_player = D := 
by 
  sorry

end select_best_player_l710_710440


namespace total_number_of_people_wearing_sunglasses_l710_710524

theorem total_number_of_people_wearing_sunglasses :
  let total_people := 3000
  let children_percentage := 0.4
  let women_sunglasses_percentage := 0.25
  let men_sunglasses_percentage := 0.15
  let children_sunglasses_percentage := 0.05
  let children := total_people * children_percentage
  let adults := total_people - children
  let women := adults / 2
  let men := adults / 2
  let women_wearing_sunglasses := women * women_sunglasses_percentage
  let men_wearing_sunglasses := men * men_sunglasses_percentage
  let children_wearing_sunglasses := children * children_sunglasses_percentage
  let total_wearing_sunglasses := women_wearing_sunglasses + men_wearing_sunglasses + children_wearing_sunglasses
  in total_wearing_sunglasses = 420 := by
  sorry

end total_number_of_people_wearing_sunglasses_l710_710524


namespace same_solution_count_for_inequalities_l710_710812

theorem same_solution_count_for_inequalities (n k : ℕ) :
  (∃ (x : Fin k → ℤ), (∑ i, |x i| ≤ n) ↔ 
   ∃ (y : Fin n → ℤ), (∑ j, |y j| ≤ k)) :=
by sorry

end same_solution_count_for_inequalities_l710_710812


namespace find_q_l710_710247

noncomputable def expr (a b c : ℝ) := a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2

noncomputable def lhs (a b c : ℝ) := (a - b) * (b - c) * (c - a)

theorem find_q (a b c : ℝ) : expr a b c = lhs a b c * 1 := by
  sorry

end find_q_l710_710247


namespace part1_arithmetic_sequence_part2_general_formula_l710_710070

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710070


namespace b_arithmetic_sequence_a_general_formula_l710_710050

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710050


namespace solve_inequality_find_extrema_min_find_extrema_max_l710_710822

theorem solve_inequality (x : ℝ) : log 2 (2 - x) ≤ log 2 (3 * x + 6) → (-1 ≤ x ∧ x < 2) :=
by {
  sorry
}

theorem find_extrema_min (x : ℝ) (hx : -1 ≤ x ∧ x < 2) : 
  (∀ x, y = (1/4)^(x - 1) - 4 * (1/2)^x + 2 → (y = 1 ↔ x = 1)) :=
by {
  sorry
}

theorem find_extrema_max (x : ℝ) (hx : -1 ≤ x ∧ x < 2) : 
  (∀ x, y = (1/4)^(x - 1) - 4 * (1/2)^x + 2 → (y = 10 ↔ x = -1)) :=
by {
  sorry
}

end solve_inequality_find_extrema_min_find_extrema_max_l710_710822


namespace smallest_two_digit_multiple_of_six_not_multiple_of_four_l710_710320

def two_digit_positive_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def multiple_of_six (n : ℕ) : Prop :=
  n % 6 = 0

def not_multiple_of_four (n : ℕ) : Prop :=
  n % 4 ≠ 0

theorem smallest_two_digit_multiple_of_six_not_multiple_of_four :
  ∃ n, two_digit_positive_integer n ∧ multiple_of_six n ∧ not_multiple_of_four n ∧
       ∀ m, (two_digit_positive_integer m ∧ multiple_of_six m ∧ not_multiple_of_four m) → n ≤ m :=
begin
  use 18,
  split,
  { -- Prove 18 is a two-digit positive integer
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Prove 18 is a multiple of six
    exact by norm_num },
  split,
  { -- Prove 18 is not a multiple of four
    exact by norm_num },
  { -- Prove 18 is the smallest such number
    intros m h,
    cases h with h1 h2,
    cases h2 with h2 h3,
    cases h1 with hm1 hm2,
    have : m ≥ 18 := sorry,
    exact this,
  }
end

end smallest_two_digit_multiple_of_six_not_multiple_of_four_l710_710320


namespace b_arithmetic_sequence_a_general_formula_l710_710042

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710042


namespace part1_arithmetic_sequence_part2_general_formula_l710_710106

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710106


namespace foreign_objects_total_l710_710750

theorem foreign_objects_total (burrs ticks : ℕ) (h1 : burrs = 12) (h2 : ticks = 6 * burrs) : burrs + ticks = 84 :=
by {
  subst h1,
  subst h2,
  simp,
}

end foreign_objects_total_l710_710750


namespace max_original_chess_pieces_l710_710960

theorem max_original_chess_pieces (m n M N : ℕ) (h1 : m ≤ 19) (h2 : n ≤ 19) (h3 : M ≤ 19) (h4 : N ≤ 19) (h5 : M * N = m * n + 45) (h6 : M = m ∨ N = n) : m * n ≤ 285 :=
by
  sorry

end max_original_chess_pieces_l710_710960


namespace relatively_prime_probability_l710_710313

theorem relatively_prime_probability (n : ℕ) (h : n = 42) :
  let phi := n * (1 - 1 / 2) * (1 - 1 / 3) * (1 - 1 / 7) in
  (phi / n) = 2 / 7 :=
by
  sorry

end relatively_prime_probability_l710_710313


namespace symmetric_point_origin_l710_710242

theorem symmetric_point_origin (x y : ℝ) (h : x = -2 ∧ y = 3) : (-x, -y) = (2, -3) := by
  cases h with
  | intro h_x h_y => 
    rw [h_x, h_y]
    simp
    sorry

end symmetric_point_origin_l710_710242


namespace triangle_area_correct_l710_710517

noncomputable def triangle_area (A B C : ℝ) (a b c R : ℝ) :=
  if A = π / 3 ∧ b = 1 ∧ R = 1 then
    1 / 2 * (2 * R * sin A) * b * sin (π - A - (asin (b / (2 * R)))) = sqrt 3 / 2
  else
    false

theorem triangle_area_correct :
  triangle_area (π / 3) _ _ (2 * 1 * sin (π / 3)) 1 _ 1 :=
by
  sorry

end triangle_area_correct_l710_710517


namespace Lily_points_l710_710655

variable (x y z : ℕ) -- points for inner ring (x), middle ring (y), and outer ring (z)

-- Tom's score
axiom Tom_score : 3 * x + y + 2 * z = 46

-- John's score
axiom John_score : x + 3 * y + 2 * z = 34

-- Lily's score
def Lily_score : ℕ := 40

theorem Lily_points : ∀ (x y z : ℕ), 3 * x + y + 2 * z = 46 → x + 3 * y + 2 * z = 34 → Lily_score = 40 := by
  intros x y z Tom_score John_score
  sorry

end Lily_points_l710_710655


namespace geom_seq_a2_eq_8_l710_710850

-- A helper function to compute the sum of the first n terms of a geometric sequence
def geom_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

-- The main theorem to prove
theorem geom_seq_a2_eq_8 (a_1 : ℝ) (q : ℝ) (S_4 : ℝ) (h1 : q = 2) (h2 : S_4 = 60) (h3 : S_4 = geom_sum a_1 q 4) :
  a_1 * q = 8 :=
by
  sorry

end geom_seq_a2_eq_8_l710_710850


namespace sequence_bn_arithmetic_and_an_formula_l710_710122

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710122


namespace number_of_tiger_groups_is_3_l710_710709

theorem number_of_tiger_groups_is_3 :
  ∀ (tigers foxes : ℕ) (groups : ℕ) (animals_per_group : ℕ) (no_responses : ℕ),
  tigers = 30 →
  foxes = 30 →
  groups = 20 →
  animals_per_group = 3 →
  no_responses = 39 →
  (∃ n, n * 3 = (no_responses - foxes) ∧ n = 3) :=
by
  intros tigers foxes groups animals_per_group no_responses
  intro h_tigers
  intro h_foxes
  intro h_groups
  intro h_animals_per_group
  intro h_no_responses

  have : (no_responses - foxes) = 9, from sorry,
  have : 9 / 3 = 3, from sorry,

  existsi 3,
  split,
  {
    calc 3 * 3 = 9 : by norm_num
            ...  = (no_responses - foxes) : by sorry,
  },
  {
    show 3 = 3, from rfl,
  }

end number_of_tiger_groups_is_3_l710_710709


namespace angle_sum_lt_ninety_l710_710833

theorem angle_sum_lt_ninety {A B C O P : Type*} [Triangle A B C] [Orthocenter O A B C]
  [Altitude P A B C] (h_angle_cond : angle B C A ≥ angle A B C + 30) :
  angle A C B + angle C O P < 90 :=
sorry

end angle_sum_lt_ninety_l710_710833


namespace max_knights_statement_l710_710647

def knight_liar_table (people : Fin 10 → Prop) : Prop :=
  ∃ k l : Fin 10, people k ∧ ¬people l

theorem max_knights_statement :
  ∀ (people : Fin 10 → Prop),
  knight_liar_table people →
  ∃ (max_sayers : ℕ), max_sayers = 9 ∧ 
  ∀ (sayers : Fin 10 → Prop), 
  (∀ i, sayers i → people (i - 1) ∧ people (i + 1)) → 
  card (filter sayers (finset.univ)) ≤ max_sayers :=
by
  sorry

end max_knights_statement_l710_710647


namespace statement_B_is_false_l710_710775

def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

theorem statement_B_is_false (x y : ℝ) : 3 * (heartsuit x y) ≠ heartsuit (3 * x) y := by
  sorry

end statement_B_is_false_l710_710775


namespace distance_of_given_complex_l710_710610

noncomputable def complex_to_origin_distance : ℂ → ℝ
| z := complex.abs z

theorem distance_of_given_complex : complex_to_origin_distance (i / (1 + i)) = (Real.sqrt 2) / 2 :=
by
  sorry

end distance_of_given_complex_l710_710610


namespace count_solutions_eq_l710_710813

theorem count_solutions_eq (n k : ℕ) :
  ∃ f : (fin k → ℤ) → (fin n → ℤ),
  ∀ x : fin k → ℤ,
    (∑ i, |x i| ≤ n) →
    (∑ j, |(f x) j| ≤ k) :=
sorry

end count_solutions_eq_l710_710813


namespace range_of_x_l710_710841

theorem range_of_x {a x : ℝ} (h : ∀ x : ℝ, x^2 - 4 * a * x + 2 * a + 30 ≥ 0) :
  (x / (a + 3) = |a - 1| + 1) → x ∈ set.Ico (-9 / 4 : ℝ) 0 ∪ set.Ioo 0 15 :=
by
  sorry

end range_of_x_l710_710841


namespace possible_A_values_l710_710592

-- Defining A within the range of 0 to 9
def is_possible_A (A : ℕ) : Prop := A ≥ 5 ∧ A ≤ 9

-- The core statement of the proof problem
theorem possible_A_values : {A : ℕ | is_possible_A A}.toFinset.card = 5 := 
by
  sorry

end possible_A_values_l710_710592


namespace bn_arithmetic_sequence_an_formula_l710_710029

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710029


namespace prob_relatively_prime_42_l710_710315

noncomputable def euler_totient (n : ℕ) : ℕ :=
  (List.range n).filter (λ i => Nat.gcd i n = 1).length

theorem prob_relatively_prime_42 : 
  (euler_totient 42 : ℚ) / 42 = 2 / 7 := 
by
  sorry

end prob_relatively_prime_42_l710_710315


namespace bn_arithmetic_sequence_an_formula_l710_710030

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710030


namespace initial_money_eq_l710_710959

-- Definitions for the problem conditions
def spent_on_sweets : ℝ := 1.25
def spent_on_friends : ℝ := 2 * 1.20
def money_left : ℝ :=  4.85

-- Statement of the problem to prove
theorem initial_money_eq :
  spent_on_sweets + spent_on_friends + money_left = 8.50 := 
sorry

end initial_money_eq_l710_710959


namespace b_arithmetic_a_general_formula_l710_710018

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710018


namespace incenter_lines_pass_through_orthocenter_l710_710168

theorem incenter_lines_pass_through_orthocenter
  (A E F D B C : Point)
  (A1 B1 C1 : Point)
  (hA1 : is_incenter_of_triangle A1 A E F)
  (hB1 : is_incenter_of_triangle B1 B D F)
  (hC1 : is_incenter_of_triangle C1 C D E) :
  passes_through_orthocenter A1 D B1 E C1 F A1 B1 C1 := sorry

end incenter_lines_pass_through_orthocenter_l710_710168


namespace number_of_integer_pairs_l710_710874

theorem number_of_integer_pairs :
  ∃ (a b : ℕ), 
    (2 ≤ a / 5 ∧ 3 ≤ a / 5 ∧ 4 ≤ a / 5) ∧
    (a < 25) ∧
    (6 ≤ b ∧ b < 12) ∧
    (A ∩ B ∩ ℕ = {2, 3, 4}) ∧
    ((∃ count_a count_b, count_a = {a | 20 ≤ a ∧ a < 25}.card ∧ 
                          count_b = {b | 6 ≤ b ∧ b < 12}.card ∧ 
                          count_a * count_b = 30)) :=
begin
  sorry  -- proof goes here
end

end number_of_integer_pairs_l710_710874


namespace range_of_a_l710_710270

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) := by
  sorry

end range_of_a_l710_710270


namespace lcm_is_perfect_square_l710_710197

open Nat

theorem lcm_is_perfect_square (a b : ℕ) : 
  (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0 → ∃ k : ℕ, k^2 = lcm a b :=
by
  sorry

end lcm_is_perfect_square_l710_710197


namespace midpoint_translation_l710_710214

def translate_point (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoint_translation :
  let s1_p1 := (3, -2)
  let s1_p2 := (-7, 4)
  let s2_dx := 5
  let s2_dy := -2
  let mid_s1 := midpoint s1_p1 s1_p2
  let mid_s2 := translate_point mid_s1 s2_dx s2_dy
  mid_s2 = (3, -1) :=
by
  sorry

end midpoint_translation_l710_710214


namespace find_c_l710_710253

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem find_c (c : ℝ) : 
  let A := (2, 4)
  let B := (6, 8)
  let M := midpoint A B
  (M.1 + M.2 = c) ∧ (M = (4, 6)) → c = 10 := 
by
  intros
  sorry

end find_c_l710_710253


namespace max_sum_of_xyz_inf_rational_triples_l710_710708

theorem max_sum_of_xyz (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) 
(h_eq : 16 * x * y * z = (x + y)^2 * (x + z)^2) : 
  x + y + z ≤ 4 :=
sorry

theorem inf_rational_triples (x y z : ℚ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) 
(h_eq : 16 * x * y * z = (x + y)^2 * (x + z)^2) 
(h_sum : x + y + z = 4) : 
  ∃^(infinitely many) (x y z : ℚ), 16 * x * y * z = (x + y)^2 * (x + z)^2 ∧ x + y + z = 4 :=
sorry

end max_sum_of_xyz_inf_rational_triples_l710_710708


namespace tom_change_l710_710282

theorem tom_change :
  let SNES_value := 150
  let credit_percent := 0.80
  let amount_given := 80
  let game_value := 30
  let NES_sale_price := 160
  let credit_for_SNES := credit_percent * SNES_value
  let amount_to_pay_for_NES := NES_sale_price - credit_for_SNES
  let effective_amount_paid := amount_to_pay_for_NES - game_value
  let change_received := amount_given - effective_amount_paid
  change_received = 70 :=
by
  sorry

end tom_change_l710_710282


namespace lcm_perfect_square_l710_710190

-- Define the conditions and the final statement in Lean 4
theorem lcm_perfect_square (a b : ℕ) 
  (h: (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : 
  ∃ k : ℕ, lcm a b = k^2 :=
sorry

end lcm_perfect_square_l710_710190


namespace count_solutions_eq_l710_710814

theorem count_solutions_eq (n k : ℕ) :
  ∃ f : (fin k → ℤ) → (fin n → ℤ),
  ∀ x : fin k → ℤ,
    (∑ i, |x i| ≤ n) →
    (∑ j, |(f x) j| ≤ k) :=
sorry

end count_solutions_eq_l710_710814


namespace constant_term_poly_expr_l710_710287

-- Define the polynomial expression
def poly_expr (x : ℝ) : ℝ := (3 * x + 2 / x) ^ 8

-- Statement of the theorem
theorem constant_term_poly_expr : 
  let x := var in
  (polynomial.constant_coeff (C (3 * x) + C (2 / x))) = 90720 :=
begin
  sorry
end

end constant_term_poly_expr_l710_710287


namespace seat_1_is_abby_l710_710382

-- Definitions for conditions
def seated (seats : list string) : Prop :=
  seats = ["Abby", "Bret", "Carl", "Dana"]

def bret_sits_in_seat_2 (seats : list string) : Prop :=
  seats.nth 1 = some "Bret"

def bret_next_to_abby (seats : list string) : Prop :=
  (seats.nth 0 = some "Abby" ∧ seats.nth 1 = some "Bret") ∨
  (seats.nth 1 = some "Bret" ∧ seats.nth 2 = some "Abby")

def dana_not_between_bret_and_carl (seats : list string) : Prop :=
  ¬((seats.nth 1 = some "Bret" ∧ seats.nth 2 = some "Carl" ∧ seats.nth 2 = some "Dana") ∨
    (seats.nth 2 = some "Bret" ∧ seats.nth 1 = some "Carl" ∧ seats.nth 3 = some "Dana"))

-- Theorem statement
theorem seat_1_is_abby (seats : list string) :
  bret_sits_in_seat_2 seats →
  bret_next_to_abby seats →
  dana_not_between_bret_and_carl seats →
  seats.nth 0 = some "Abby" :=
begin
  sorry
end

end seat_1_is_abby_l710_710382


namespace total_students_course_l710_710696

theorem total_students_course 
  (T : ℕ)
  (H1 : (1 / 5 : ℚ) * T = (1 / 5) * T)
  (H2 : (1 / 4 : ℚ) * T = (1 / 4) * T)
  (H3 : (1 / 2 : ℚ) * T = (1 / 2) * T)
  (H4 : T = (1 / 5 : ℚ) * T + (1 / 4 : ℚ) * T + (1 / 2 : ℚ) * T + 30) : 
  T = 600 :=
sorry

end total_students_course_l710_710696


namespace complex_mod_sum_inv_is_3_over_8_l710_710953

noncomputable def complex_mod_sum_inv (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) : ℝ :=
  |(1/z) + (1/w)|

theorem complex_mod_sum_inv_is_3_over_8 (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  complex_mod_sum_inv z w hz hw hzw = 3 / 8 :=
sorry

end complex_mod_sum_inv_is_3_over_8_l710_710953


namespace probability_rel_prime_to_42_l710_710306

theorem probability_rel_prime_to_42 : 
  let n := 42 in
  let prime_factors := [2, 3, 7] in
  let relatively_prime_count := n * (1 - 1/prime_factors[0]) * (1 - 1/prime_factors[1]) * (1 - 1/prime_factors[2]) in
  let total_count := 42 in
  (relatively_prime_count / total_count) = 4 / 7 :=
by
  sorry

end probability_rel_prime_to_42_l710_710306


namespace product_of_roots_l710_710164

noncomputable def roots_product : Prop :=
  let f := (3 : ℝ) * X^3 - 9 * X^2 + 4 * X - 12
  ∃ (a b c : ℝ), (3*a^3 - 9*a^2 + 4*a - 12 = 0) ∧
                 (3*b^3 - 9*b^2 + 4*b - 12 = 0) ∧
                 (3*c^3 - 9*c^2 + 4*c - 12 = 0) ∧
                 (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧
                 a * b * c = 4

theorem product_of_roots : roots_product :=
  sorry

end product_of_roots_l710_710164


namespace greatest_divisor_of_sum_of_first_15_terms_arithmetic_sequence_l710_710291

theorem greatest_divisor_of_sum_of_first_15_terms_arithmetic_sequence
  (a d : ℕ) (h : ∀ n, n ≥ 1 → n ≤ 15 → a + n * d > 0):
  15 ∣ ∑ k in finset.range 15, (a + k * d) := 
by
  sorry

end greatest_divisor_of_sum_of_first_15_terms_arithmetic_sequence_l710_710291


namespace find_x_l710_710785

theorem find_x (x : ℝ) (h1 : -1 < x ∧ x ≤ 2)
    (h2 : sqrt (2 - x) + sqrt (2 + 2 * x) = sqrt ((x^4 + 1) / (x^2 + 1)) + (x + 3) / (x + 1)) :
    x = 1 :=
begin
  sorry
end

end find_x_l710_710785


namespace part1_arithmetic_sequence_part2_general_formula_l710_710071

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710071


namespace medal_allocation_correct_l710_710384

-- Noncomputable, because we're dealing with logical assertions rather than computations
noncomputable def medal := {Gold, Silver, Bronze}

def individuals := {Xiaoming, Xiaole, Xiaoqiang}

variable (X_M : medal)
variable (X_L : medal)
variable (X_Q : medal)

def teacher_guesses := {
  guess1 : X_M = Gold,
  guess2 : X_L ≠ Gold,
  guess3 : X_Q ≠ Bronze
}

def valid_medal_allocation :=
  X_L = Gold ∧ X_Q = Silver ∧ X_M = Bronze 

theorem medal_allocation_correct (h : valid_medal_allocation) : 
  exists guess ∈ teacher_guesses, guess ∧ ¬ (teacher_guesses - {guess}) :=
by {
  sorry
}

end medal_allocation_correct_l710_710384


namespace cross_section_area_of_truncated_pyramid_l710_710636

-- Given conditions
variables (a b : ℝ) (α : ℝ)
-- Constraints
variable (h : a > b ∧ b > 0 ∧ α > 0 ∧ α < Real.pi / 2)

-- Proposed theorem
theorem cross_section_area_of_truncated_pyramid (h : a > b ∧ b > 0 ∧ α > 0 ∧ α < Real.pi / 2) :
    ∃ area : ℝ, area = (7 * a + 3 * b) / (144 * Real.cos α) * Real.sqrt (3 * (a^2 + b^2 + 2 * a * b * Real.cos (2 * α))) :=
sorry

end cross_section_area_of_truncated_pyramid_l710_710636


namespace b_arithmetic_sequence_a_general_formula_l710_710049

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710049


namespace sequence_bn_arithmetic_and_an_formula_l710_710123

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710123


namespace largest_zigzag_number_count_four_digit_zigzag_l710_710732

-- Predicate that represents a zigzag number
def is_zigzag (n : ℕ) : Prop :=
  let digits := nat.digits 10 n
  ∃ distinct_digits : ∀ i, i < list.length digits → digits.nth i ≠ 0 ∧ ∀ j, i ≠ j → digits.nth i ≠ digits.nth j,
  ∃ no_inc_sequence : ∀ i, i + 2 < list.length digits → digits.nth i < digits.nth (i+1) ∧ digits.nth (i+1) < digits.nth (i+2) → false,
  ∃ no_dec_sequence : ∀ i, i + 2 < list.length digits → digits.nth i > digits.nth (i+1) ∧ digits.nth (i+1) > digits.nth (i+2) → false,
  true

-- Proof that the largest zigzag number is 978563412
theorem largest_zigzag_number :
  ∀ n : ℕ, is_zigzag n → n ≤ 978563412 :=
by sorry

-- Predicate that represents a four-digit zigzag number
def is_four_digit_zigzag (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ is_zigzag n

-- Proof that there are 1260 four-digit zigzag numbers
theorem count_four_digit_zigzag :
  fintype.card {n // is_four_digit_zigzag n} = 1260 :=
by sorry

end largest_zigzag_number_count_four_digit_zigzag_l710_710732


namespace probability_heads_and_three_l710_710881

open ProbabilityTheory

-- Define the sample space for the coin flips and the die roll
def coin_flip_space := {HH, HT, TH, TT}
def die_roll_space := {1, 2, 3, 4, 5, 6}

-- Define the event of interest: flipping two heads and rolling a 3
def event := { (HH, 3) }

-- Define the probability measure
def p : finset (string × ℕ) → ℚ := λ s,
  if s = { (HH, 3) } then 1 / 24 else 0

theorem probability_heads_and_three : 
  p { (HH, 3) } = 1 / 24 := 
by
  sorry

end probability_heads_and_three_l710_710881


namespace translate_statement_to_inequality_l710_710413

theorem translate_statement_to_inequality (y : ℝ) : (1/2) * y + 5 > 0 ↔ True := 
sorry

end translate_statement_to_inequality_l710_710413


namespace calculate_wall_length_l710_710353

theorem calculate_wall_length :
  let brick_length := 0.25 -- in meters
  let brick_width := 0.11 -- in meters
  let brick_height := 0.06 -- in meters
  let wall_width := 3.0 -- in meters
  let wall_height := 0.02 -- in meters
  let num_bricks := 72.72727272727273
  let brick_volume := brick_length * brick_width * brick_height
  let wall_volume := num_bricks * brick_volume
  let wall_length := wall_volume / (wall_width * wall_height)
  wall_length = 2 :=
begin
  sorry
end

end calculate_wall_length_l710_710353


namespace part1_arithmetic_sequence_part2_general_formula_l710_710102

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710102


namespace concrete_pillars_l710_710548

-- Definitions based on the conditions of the problem
def C_deck : ℕ := 1600
def C_anchor : ℕ := 700
def C_total : ℕ := 4800

-- Theorem to prove the concrete required for supporting pillars
theorem concrete_pillars : C_total - (C_deck + 2 * C_anchor) = 1800 :=
by sorry

end concrete_pillars_l710_710548


namespace complete_square_transform_l710_710680

theorem complete_square_transform (x : ℝ) : 
  x^2 - 2 * x = 9 ↔ (x - 1)^2 = 10 :=
by
  sorry

end complete_square_transform_l710_710680


namespace imaginary_part_of_complex_number_l710_710623

def imaginary_unit (i : ℂ) : Prop := i * i = -1

def complex_number (z : ℂ) (i : ℂ) : Prop := z = i * (1 - 3 * i)

theorem imaginary_part_of_complex_number (i z : ℂ) (h1 : imaginary_unit i) (h2 : complex_number z i) : z.im = 1 :=
by
  sorry

end imaginary_part_of_complex_number_l710_710623


namespace tetrahedron_area_sum_l710_710769

theorem tetrahedron_area_sum (m n p : ℕ) (a b c d : ℚ) :
  let s := 2 in
  let A := (Finset.univ : Finset (Fin n)).card in
  let area_of_face := (√3 / 4) * s^2 in
  let total_area := A * area_of_face in
  let (m, n, p) := (4, 3, 0) in
  m + n + p = 7 := 
by
  let vol := s * sqrt (s^2 - (s/2)^2),
  let x := sqrt (4 − vol^2),
  show m + n + p = 7,
  sorry

end tetrahedron_area_sum_l710_710769


namespace power_function_passing_point_f_monotonic_on_nonneg_g_even_function_range_of_m_l710_710466

noncomputable def f(x : ℝ) : ℝ := x ^ (1 / 2 : ℝ)

theorem power_function_passing_point :
  f(2) = Real.sqrt 2 := by
  sorry

theorem f_monotonic_on_nonneg : monotone_on f (Ici 0) := by
  sorry

def g(x : ℝ) : ℝ := if x ≥ 0 then f(x) else f(-x)

theorem g_even_function (x : ℝ) : g(x) = g(-x) := by
  sorry

theorem range_of_m : { m : ℝ // g(1 - m) ≤ Real.sqrt 5 } = set.Icc (-4) 6 := by
  sorry

end power_function_passing_point_f_monotonic_on_nonneg_g_even_function_range_of_m_l710_710466


namespace part1_sequence_arithmetic_part2_general_formula_l710_710135

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710135


namespace find_c_l710_710252

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem find_c (c : ℝ) : 
  let A := (2, 4)
  let B := (6, 8)
  let M := midpoint A B
  (M.1 + M.2 = c) ∧ (M = (4, 6)) → c = 10 := 
by
  intros
  sorry

end find_c_l710_710252


namespace b_arithmetic_a_general_formula_l710_710013

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710013


namespace factorize_expression_l710_710414

theorem factorize_expression (x y : ℝ) : 
  x * y^2 - 6 * x * y + 9 * x = x * (y - 3)^2 := 
by sorry

end factorize_expression_l710_710414


namespace two_person_subcommittees_l710_710357

theorem two_person_subcommittees (groupA groupB : Type) [Fintype groupA] [Fintype groupB] (hA : Fintype.card groupA = 5) (hB : Fintype.card groupB = 3) :
  (Fintype.card groupA) * (Fintype.card groupB) = 15 :=
by
  rw [hA, hB]
  norm_num
  sorry

end two_person_subcommittees_l710_710357


namespace petya_prevents_natural_sum_l710_710206

def fractions_sum_prevention (current_sum : ℚ) : Prop :=
  ∀ S : ℚ, ∀ n : ℕ, n ≥ 1 → ∀ k : ℕ, ∀ (f : fin (k + 1) → ℕ), 
  (S = current_sum + ∑ i in fin (k + 1), (1 / (f i) : ℚ)) → ¬ (S ∈ set.univ ℕ)

theorem petya_prevents_natural_sum : ∀ (current_sum : ℚ), fractions_sum_prevention current_sum :=
by 
  sorry

end petya_prevents_natural_sum_l710_710206


namespace expression_value_l710_710257

theorem expression_value (a b c d : ℝ) 
  (intersect1 : 4 = a * (2:ℝ)^2 + b * 2 + 1) 
  (intersect2 : 4 = (2:ℝ)^2 + c * 2 + d) 
  (hc : b + c = 1) : 
  4 * a + d = 1 := 
sorry

end expression_value_l710_710257


namespace clock_hands_apart_at_5_24_l710_710752

theorem clock_hands_apart_at_5_24 : 
  ∀ (hour minute : ℕ), hour = 5 → minute = 24 → 
  let minute_hand_position := 24 * 6 in
  let hour_hand_position := (5 * 30) + (24 * 0.5) in
  let difference := |minute_hand_position - hour_hand_position| in
  round (difference / 6) = 13 :=
by
  intros hour minute h1 h2
  simp [h1, h2]
  sorry

end clock_hands_apart_at_5_24_l710_710752


namespace log2_intersects_x_axis_l710_710251

theorem log2_intersects_x_axis :
  ∃ x : ℝ, x > 0 ∧ log 2 x = 0 ∧ (x = 1) :=
by
  sorry

end log2_intersects_x_axis_l710_710251


namespace devah_erases_895_dots_l710_710780

theorem devah_erases_895_dots :
  let number_of_dots := 1000
  ∃ erased_count, erased_count = number_of_dots - (number_of_dots - (Nat.range number_of_dots).count (λ n, ∀ b ∈ nat.digits 3 n, b ≠ 2)) :=
by {
  let number_of_dots := 1000;
  have h : (number_of_dots - (number_of_dots - (Nat.range number_of_dots).count (λ n, ∀ b ∈ nat.digits 3 n, b ≠ 2))) = 895,
  {
    sorry,
  },
  use (number_of_dots - (number_of_dots - (Nat.range number_of_dots).count (λ n, ∀ b ∈ nat.digits 3 n, b ≠ 2))),
  exact h,
}

end devah_erases_895_dots_l710_710780


namespace sin_half_pi_plus_A_cos_A_minus_B_l710_710845

-- Define the problem conditions
noncomputable def area_triangle (AB AC : ℝ) (A : ℝ) : ℝ := 
  1 / 2 * AB * AC * Classical.sin A

-- Questions
theorem sin_half_pi_plus_A 
  (A : ℝ) (AB AC : ℝ) 
  (h₁ : AB = 3) 
  (h₂ : AC = 4) 
  (h₃ : area_triangle AB AC A = 3 * Real.sqrt 3) : 
  Classical.sin (Real.pi / 2 + A) = 1 / 2 :=
sorry

theorem cos_A_minus_B 
  (A B : ℝ) (AB AC : ℝ) 
  (h₁ : AB = 3) 
  (h₂ : AC = 4) 
  (h₃ : area_triangle AB AC A = 3 * Real.sqrt 3) 
  (h₄ : Classical.cos A = 1 / 2) 
  (BC : ℝ) 
  (h₅ : BC = Real.sqrt 13) 
  (h₆ : Classical.sin B = 2 * Real.sqrt 39 / 13) 
  (h₇ : Classical.cos B = Real.sqrt 13 / 13): 
  Classical.cos (A - B) = 7 * Real.sqrt 13 / 26 :=
sorry

end sin_half_pi_plus_A_cos_A_minus_B_l710_710845


namespace num_pos_int_solutions_2a_plus_3b_eq_15_l710_710265

theorem num_pos_int_solutions_2a_plus_3b_eq_15 : 
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 2 * a + 3 * b = 15) ∧ 
  (∀ (a1 a2 b1 b2 : ℕ), 0 < a1 ∧ 0 < a2 ∧ 0 < b1 ∧ 0 < b2 ∧ 
  (2 * a1 + 3 * b1 = 15) ∧ (2 * a2 + 3 * b2 = 15) → 
  ((a1 = 3 ∧ b1 = 3 ∨ a1 = 6 ∧ b1 = 1) ∧ (a2 = 3 ∧ b2 = 3 ∨ a2 = 6 ∧ b2 = 1))) := 
  sorry

end num_pos_int_solutions_2a_plus_3b_eq_15_l710_710265


namespace problem_statement_l710_710445

def f (x : ℝ) : ℝ := x^2 + 1

theorem problem_statement : f(f(f(-1))) = 26 := by
  sorry

end problem_statement_l710_710445


namespace optimal_strategy_l710_710687

def ladder_rungs := 5
def bottom_rung := 1
def top_rung := ladder_rungs
def probability_of_winning (coin: Nat → Rat) (rung: Nat): Rat :=
  if rung = bottom_rung then 0 else if rung = top_rung then 1 else (coin (rung - 1) + coin (rung + 1)) / 2

def fair_coin (rung: Nat) : Rat :=
  match rung with
  | r if r = bottom_rung => 0
  | r if r = top_rung => 1
  | r => r * (1 / 6)

def biased_coin_heads (rung: Nat): Rat := 1
def biased_coin_tails (rung: Nat): Rat :=
  match rung with
  | r if r = bottom_rung => 0
  | r => fair_coin (rung - 1)

def combined_biased_coin (rung: Nat): Rat :=
  (biased_coin_heads rung + biased_coin_tails rung) / 2

def winning_probability (rung: Nat): Rat :=
  if rung = bottom_rung then (1 / 2) else if rung = top_rung then 1 else fair_coin rung

theorem optimal_strategy (rung: Nat): Prop :=
  rung = bottom_rung → winning_probability rung = (1 / 2)

-- To skip the proof:
example : optimal_strategy bottom_rung := sorry

end optimal_strategy_l710_710687


namespace b_arithmetic_a_formula_l710_710005

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l710_710005


namespace C1_equation_C2_equation_MA_MB_distance_l710_710535

-- Defining the parametric equations of curve C1
def C1_parametric (t : ℝ) : ℝ × ℝ := (1 + t, 1 + 2 * t)

-- Defining the polar equation of curve C2
def C2_polar (θ ρ : ℝ) : Prop := ρ * (1 - Real.sin θ) = 1

-- Given point M
def M : ℝ × ℝ := (0, -1)

-- Intersection points A and B
def A : ℝ × ℝ := (2 + Real.sqrt 3, 3 + 2 * Real.sqrt 3)
def B : ℝ × ℝ := (2 - Real.sqrt 3, 3 - 2 * Real.sqrt 3)

-- Prove that the general equation of C1 is 2x - y - 1 = 0
theorem C1_equation (x y : ℝ) (t : ℝ) (h : C1_parametric t = (x, y)) : 2 * x - y - 1 = 0 := sorry

-- Prove that the rectangular coordinate equation of C2 is x^2 = 2y + 1
theorem C2_equation (x y : ℝ) (ρ θ : ℝ) (h : C2_polar θ ρ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) :
    x^2 = 2 * y + 1 := sorry

-- Prove that |MA| * |MB| = 5
theorem MA_MB_distance : 
    Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2) * Real.sqrt ((B.1 - M.1)^2 + (B.2 - M.2)^2) = 5 := sorry

end C1_equation_C2_equation_MA_MB_distance_l710_710535


namespace concrete_pillars_correct_l710_710554

-- Definitions based on conditions
def concrete_for_roadway := 1600
def concrete_for_one_anchor := 700
def total_concrete_for_bridge := 4800

-- Total concrete for both anchors
def concrete_for_both_anchors := 2 * concrete_for_one_anchor

-- Total concrete needed for the roadway and anchors
def concrete_for_roadway_and_anchors := concrete_for_roadway + concrete_for_both_anchors

-- Concrete needed for the supporting pillars
def concrete_for_pillars := total_concrete_for_bridge - concrete_for_roadway_and_anchors

-- Proof problem statement, verify that the concrete for the supporting pillars is 1800 tons
theorem concrete_pillars_correct : concrete_for_pillars = 1800 := by
  sorry

end concrete_pillars_correct_l710_710554


namespace frogs_count_l710_710542

variables (Alex Brian Chris LeRoy Mike : Type) 

-- Definitions for the species
def toad (x : Type) : Prop := ∃ p : Prop, p -- Dummy definition for toads
def frog (x : Type) : Prop := ∃ p : Prop, ¬p -- Dummy definition for frogs

-- Conditions
axiom Alex_statement : (toad Alex) → (∃ x : ℕ, x = 3) ∧ (frog Alex) → (¬(∃ x : ℕ, x = 3))
axiom Brian_statement : (toad Brian) → (toad Mike) ∧ (frog Brian) → (frog Mike)
axiom Chris_statement : (toad Chris) → (toad LeRoy) ∧ (frog Chris) → (frog LeRoy)
axiom LeRoy_statement : (toad LeRoy) → (toad Chris) ∧ (frog LeRoy) → (frog Chris)
axiom Mike_statement : (toad Mike) → (∃ x : ℕ, x < 3) ∧ (frog Mike) → (¬(∃ x : ℕ, x < 3))

theorem frogs_count (total : ℕ) : total = 5 → 
  (∃ frog_count : ℕ, frog_count = 2) :=
by
  -- Leaving the proof as a sorry placeholder
  sorry

end frogs_count_l710_710542


namespace quadratic_roots_l710_710516

theorem quadratic_roots (c : ℝ) (h : ∀ x : ℝ, x = (-14 + sqrt 10) / 4 ∨ x = (-14 - sqrt 10) / 4 → 2 * x^2 + 14 * x + c = 0) : 
  c = 93 / 4 :=
by
  sorry

end quadratic_roots_l710_710516


namespace peter_ends_up_with_eleven_erasers_l710_710205

def eraser_problem : Nat :=
  let initial_erasers := 8
  let additional_erasers := 3
  let total_erasers := initial_erasers + additional_erasers
  total_erasers

theorem peter_ends_up_with_eleven_erasers :
  eraser_problem = 11 :=
by
  sorry

end peter_ends_up_with_eleven_erasers_l710_710205


namespace number_of_valid_integers_l710_710406

noncomputable def p (x : ℤ) : ℤ := x^6 - 52 * x^3 + 51

def P (x : ℤ) : Prop := p(x) < 0

def valid_integers : Set ℤ := {x | P x}

theorem number_of_valid_integers : (valid_integers.toFinset.card = 2) :=
by sorry

end number_of_valid_integers_l710_710406


namespace find_B_and_sides_l710_710459

open Real

noncomputable def triangle := Type*

-- Define two constants used in the question
def a : ℝ := 2
def b : ℝ := 2
def c : ℝ := 2

-- Define angle B
def angle_B : ℝ := 60 * (π / 180)

-- Definition of area
def area (a b c : ℝ) (angle_B : ℝ) : ℝ :=
  1 / 2 * a * c * sin angle_B

-- Given the sides a, b, c of the triangle are opposite
-- angles A, B, C respectively, and it satisfies the conditions
axiom sides_and_angle {A B C : ℝ} (a b c : ℝ) : 
  2 * b * cos C = 2 * a - c ∧ 
  1 / 2 * a * c * sin B = sqrt 3 ∧ 
  b = 2

-- We need to prove that B = 60 degrees
theorem find_B_and_sides {A B C : ℝ} : 
  sides_and_angle a b c → 
  angle_B = 60 * (π / 180) ∧ 
  a = 2 ∧ c = 2 := 
sorry

end find_B_and_sides_l710_710459


namespace b_arithmetic_sequence_a_general_formula_l710_710041

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710041


namespace sum_of_undefined_values_l710_710778

theorem sum_of_undefined_values : 
  let f := λ x : ℝ, 5 * x / (3 * x^2 - 9 * x + 6)
  in ∃ C D : ℝ, (3 * C^2 - 9 * C + 6 = 0) ∧ (3 * D^2 - 9 * D + 6 = 0) ∧ (C + D = 3) :=
sorry

end sum_of_undefined_values_l710_710778


namespace part1_part2_l710_710855

-- Defining the properties of ellipse C₁
def ellipse_C₁ (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Define the left and right vertices
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Point P on the ellipse, distinct from A and B
variable {P : ℝ × ℝ} (hP : P ≠ A ∧ P ≠ B ∧ ellipse_C₁ P.1 P.2)

-- Define the slopes k₁ and k₂
def k₁ (P : ℝ × ℝ) : ℝ := P.2 / (P.1 + 2)
def k₂ (P : ℝ × ℝ) : ℝ := P.2 / (P.1 - 2)

-- Given lambda and the trajectory condition
variable (λ : ℝ) (hλ : λ ≠ 0)
def trajectory_condition (Q : ℝ × ℝ) : Prop := (Q.2 ^ 2) / (Q.1 ^ 2 - 4) = λ * (k₁ P * k₂ P)

-- Prove the equation of curve C₂ for λ = 4
theorem part1 : λ = 4 → ∀ Q : ℝ × ℝ, (trajectory_condition λ Q) → ((Q.1)^2 + (Q.2)^2 = 4) := sorry

-- Given point M and lines intersecting C₂
def M : ℝ × ℝ := (1, 1/2)
def line_AM (y : ℝ) : ℝ := 6 * y - 2
def line_BM (y : ℝ) : ℝ := -2 * y + 2

-- Points E and F on C₂
variable {E F : ℝ × ℝ}
def on_line_AM (P : ℝ × ℝ) : Prop := P.1 = line_AM P.2
def on_line_BM (P : ℝ × ℝ) : Prop := P.1 = line_BM P.2
def on_curve_C₂ (P : ℝ × ℝ) : Prop := (P.1)^2 + (P.2)^2 = 4

variable (hE : on_line_AM E ∧ on_curve_C₂ E ∧ E ≠ A ∧ E ≠ B)
variable (hF : on_line_BM F ∧ on_curve_C₂ F ∧ F ≠ A ∧ F ≠ B)

-- Areas of triangles AMF and BME
def area_triangle (A B C : ℝ × ℝ) : ℝ := 0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

-- S₁ and S₂
def S₁ := area_triangle A M F
def S₂ := area_triangle B M E

-- Prove the range of S₁ / S₂ for λ ∈ [1, 3]
theorem part2 : ∀ λ : ℝ, 1 ≤ λ ∧ λ ≤ 3 → (5 ≤ (S₁ / S₂) ∧ (S₁ / S₂) ≤ 7) := sorry

end part1_part2_l710_710855


namespace shirt_selling_price_l710_710715

theorem shirt_selling_price (x : ℝ)
  (cost_price : x = 80)
  (initial_shirts_sold : ∃ s : ℕ, s = 30)
  (profit_per_shirt : ∃ p : ℝ, p = 50)
  (additional_shirts_per_dollar_decrease : ∃ a : ℕ, a = 2)
  (target_daily_profit : ∃ t : ℝ, t = 2000) :
  (x = 105 ∨ x = 120) := 
sorry

end shirt_selling_price_l710_710715


namespace Tim_gave_kittens_to_Jessica_l710_710279

def Tim_original_kittens : ℕ := 6
def kittens_given_to_Jessica := 3
def kittens_given_by_Sara : ℕ := 9 
def Tim_final_kittens : ℕ := 12

theorem Tim_gave_kittens_to_Jessica :
  (Tim_original_kittens + kittens_given_by_Sara - kittens_given_to_Jessica = Tim_final_kittens) :=
by sorry

end Tim_gave_kittens_to_Jessica_l710_710279


namespace plane_equation_thm_l710_710791

-- Define the points
def a : ℝ × ℝ × ℝ := (-2, 3, -1)
def b : ℝ × ℝ × ℝ := (2, 3, 1)
def c : ℝ × ℝ × ℝ := (4, 1, 0)

-- Define the vectors between points
def ab : ℝ × ℝ × ℝ := (2 - (-2), 3 - 3, 1 - (-1))
def ac : ℝ × ℝ × ℝ := (4 - (-2), 1 - 3, 0 - (-1))

-- Function to compute cross product of two 3D vectors
def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((v1.2 * v2.3 - v1.3 * v2.2), (v1.3 * v2.1 - v1.1 * v2.3), (v1.1 * v2.2 - v1.2 * v2.1))

-- Normal vector (using cross product)
def normal_vector : ℝ × ℝ × ℝ := cross_product ab ac

-- Equation of the plane condition
def plane_equation (x y z : ℝ) : Prop :=
  x + y - z - 2 = 0

-- Lean proof statement
theorem plane_equation_thm : ∃ (A B C D : ℝ), 
  (A * (-2) + B * 3 + C * (-1) + D = 0) ∧ 
  (A * 2 + B * 3 + C * 1 + D = 0) ∧ 
  (A * 4 + B * 1 + C * 0 + D = 0) ∧ 
  (A > 0) ∧ 
  (Real.gcd (Abs.A).nat_abs (Real.gcd (Abs.B).nat_abs (Real.gcd (Abs.C).nat_abs (Abs.D).nat_abs)) = 1) ∧ 
  plane_equation A B C D := 
by 
  -- The proof would follow from here, skipping with sorry
  sorry

end plane_equation_thm_l710_710791


namespace total_amount_spent_l710_710774

-- Definitions for the conditions
def cost_magazine : ℝ := 0.85
def cost_pencil : ℝ := 0.50
def coupon_discount : ℝ := 0.35

-- The main theorem to prove
theorem total_amount_spent : cost_magazine + cost_pencil - coupon_discount = 1.00 := by
  sorry

end total_amount_spent_l710_710774


namespace fixed_point_of_log_function_l710_710618

theorem fixed_point_of_log_function (a : ℝ) (h_a : a > 0 ∧ a ≠ 1) :
  (∃ x y : ℝ, y = log a (2 * x - 3) + 1 ∧ x = 2 ∧ y = 1) :=
begin
  use 2,
  use 1,
  split,
  { calc
      1 = log a 1 + 1 : by rw log_one
      ... = log a (2 * 2 - 3) + 1 : by norm_num },
  split,
  { refl },
  { refl }
end

end fixed_point_of_log_function_l710_710618


namespace interval_length_proof_l710_710786

open Real

noncomputable def cot (x : ℝ) := cos x / sin x

/-- Given the conditions on x, prove the total length of valid intervals is 4.57. -/
theorem interval_length_proof :
  let cond1 := (λ x : ℝ, 8 - x^2 ≥ -1)
  let cond2 := (λ x : ℝ, cot x ≥ -1)
  (∑ I in [0, 3.14 / 2].to_list, I.right_endpoint - I.left_endpoint) ≈ 4.57 :=
  sorry

end interval_length_proof_l710_710786


namespace quotient_is_four_l710_710797

theorem quotient_is_four (dividend : ℕ) (k : ℕ) (h1 : dividend = 16) (h2 : k = 4) : dividend / k = 4 :=
by
  sorry

end quotient_is_four_l710_710797


namespace fraction_spent_on_dvd_l710_710587

theorem fraction_spent_on_dvd (r l m d x : ℝ) (h1 : r = 200) (h2 : l = (1/4) * r) (h3 : m = r - l) (h4 : x = 50) (h5 : d = m - x) : d / r = 1 / 2 :=
by
  sorry

end fraction_spent_on_dvd_l710_710587


namespace salad_chopping_l710_710701

theorem salad_chopping (tom_rate : ℝ) (tammy_rate : ℝ) (total_salad : ℝ)
  (h1 : tom_rate = 2 / 3)
  (h2 : tammy_rate = 3 / 2)
  (h3 : total_salad = 65) :
  let tom_share := (tom_rate / (tom_rate + tammy_rate)) * total_salad
  let tammy_share := (tammy_rate / (tom_rate + tammy_rate)) * total_salad
  let difference := tammy_share - tom_share
  let percentage_difference := (difference / tom_share) * 100
  percentage_difference = 125 := 
by {
  -- sorry to skip the proof
  sorry
}

end salad_chopping_l710_710701


namespace angle_between_vectors_l710_710504

open Real EuclideanSpace

theorem angle_between_vectors (a b : EuclideanSpace ℝ 3) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ‖a + b‖ = 2 * ‖a - b‖) : 
  let θ := real.arccos (- (3 * (‖a‖^2 + ‖b‖^2) / (10 * ‖a‖ * ‖b‖))) in
  θ = real.arccos (- (3 * (‖a‖^2 + ‖b‖^2) / (10 * ‖a‖ * ‖b‖))) :=
by
  -- Proof goes here
  sorry

end angle_between_vectors_l710_710504


namespace prob_relatively_prime_42_l710_710318

noncomputable def euler_totient (n : ℕ) : ℕ :=
  (List.range n).filter (λ i => Nat.gcd i n = 1).length

theorem prob_relatively_prime_42 : 
  (euler_totient 42 : ℚ) / 42 = 2 / 7 := 
by
  sorry

end prob_relatively_prime_42_l710_710318


namespace sequence_bn_arithmetic_and_an_formula_l710_710125

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710125


namespace roger_collected_nickels_l710_710591

theorem roger_collected_nickels 
  (N : ℕ)
  (initial_pennies : ℕ := 42) 
  (initial_dimes : ℕ := 15)
  (donated_coins : ℕ := 66)
  (left_coins : ℕ := 27)
  (h_total_coins_initial : initial_pennies + N + initial_dimes - donated_coins = left_coins) :
  N = 36 := 
sorry

end roger_collected_nickels_l710_710591


namespace chickens_do_not_lay_eggs_l710_710964

theorem chickens_do_not_lay_eggs (total_chickens : ℕ) 
  (roosters : ℕ) (hens : ℕ) (hens_lay_eggs : ℕ) (hens_do_not_lay_eggs : ℕ) 
  (chickens_do_not_lay_eggs : ℕ) :
  total_chickens = 80 →
  roosters = total_chickens / 4 →
  hens = total_chickens - roosters →
  hens_lay_eggs = 3 * hens / 4 →
  hens_do_not_lay_eggs = hens - hens_lay_eggs →
  chickens_do_not_lay_eggs = hens_do_not_lay_eggs + roosters →
  chickens_do_not_lay_eggs = 35 :=
by
  intros h0 h1 h2 h3 h4 h5
  sorry

end chickens_do_not_lay_eggs_l710_710964


namespace square_areas_l710_710392

variables (a b : ℝ)

def is_perimeter_difference (a b : ℝ) : Prop :=
  4 * a - 4 * b = 12

def is_area_difference (a b : ℝ) : Prop :=
  a^2 - b^2 = 69

theorem square_areas (a b : ℝ) (h1 : is_perimeter_difference a b) (h2 : is_area_difference a b) :
  a^2 = 169 ∧ b^2 = 100 :=
by {
  sorry
}

end square_areas_l710_710392


namespace line_circle_intersect_or_tangent_l710_710628

theorem line_circle_intersect_or_tangent (k : ℝ) : 
  let circle_eq := λ x y : ℝ, x^2 + y^2 + 4*y + 3
  let line_eq := λ x y : ℝ, k*x - y - 1
  (∃ x y : ℝ, circle_eq x y = 0 ∧ line_eq x y = 0) :=
sorry

end line_circle_intersect_or_tangent_l710_710628


namespace time_to_cross_man_l710_710711

-- Define the conversion from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℕ := (speed_kmh * 1000) / 3600

-- Given conditions
def length_of_train : ℕ := 150
def speed_of_train_kmh : ℕ := 180

-- Calculate speed in m/s
def speed_of_train_ms : ℕ := kmh_to_ms speed_of_train_kmh

-- Proof problem statement
theorem time_to_cross_man : (length_of_train : ℕ) / (speed_of_train_ms : ℕ) = 3 := by
  sorry

end time_to_cross_man_l710_710711


namespace b_arithmetic_sequence_a_general_formula_l710_710043

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710043


namespace david_on_sixth_platform_l710_710662

theorem david_on_sixth_platform 
  (h₁ : walter_initial_fall = 4)
  (h₂ : walter_additional_fall = 3 * walter_initial_fall)
  (h₃ : total_fall = walter_initial_fall + walter_additional_fall)
  (h₄ : total_platforms = 8)
  (h₅ : total_height = total_fall)
  (h₆ : platform_height = total_height / total_platforms)
  (h₇ : david_fall_distance = walter_initial_fall)
  : (total_height - david_fall_distance) / platform_height = 6 := 
  by sorry

end david_on_sixth_platform_l710_710662


namespace part1_part2_l710_710781

def unitPrices (x : ℕ) (y : ℕ) : Prop :=
  (20 * x = 16 * (y + 20)) ∧ (x = y + 20)

def maxBoxes (a : ℕ) : Prop :=
  ∀ b, (100 * a + 80 * b ≤ 4600) → (a + b = 50)

theorem part1 (x : ℕ) :
  unitPrices x (x - 20) → x = 100 ∧ (x - 20 = 80) :=
by
  sorry

theorem part2 :
  maxBoxes 30 :=
by
  sorry

end part1_part2_l710_710781


namespace completing_the_square_l710_710674

theorem completing_the_square (x : ℝ) : 
  x^2 - 2 * x = 9 → (x - 1)^2 = 10 :=
by
  intro h
  sorry

end completing_the_square_l710_710674


namespace order_of_means_l710_710444

noncomputable theory

open_locale real

theorem order_of_means 
  (a b A G : ℝ)
  (ha : a = sin (real.pi / 3))
  (hb : b = cos (real.pi / 3))
  (hA : A = (a + b) / 2)
  (hG : G = real.sqrt (a * b)) :
  b < G ∧ G < A ∧ A < a :=
by {
  sorry
}

end order_of_means_l710_710444


namespace bn_arithmetic_sequence_an_formula_l710_710035

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710035


namespace find_min_area_l710_710538

variable (p q : ℝ)
variable (h_p_gt_zero : p > 0)
variable (h_q_gt_zero : q > 0)
variable (h_tangent : ∀ x : ℝ, (-p / 2 * x^2 + q)^2 + x^2 = 1)

noncomputable def area_enclosed (p q : ℝ) : ℝ :=
  2 * ∫ x in 0 ..(Real.sqrt (p^2 + 1) / p), (-p / 2 * x^2 + (p^2 + 1) / (2 * p))

theorem find_min_area :
    ∃ p q : ℝ, p > 0 ∧ q > 0 ∧ h_tangent p q ∧ area_enclosed p q = Real.sqrt 3 :=
sorry

end find_min_area_l710_710538


namespace part1_arithmetic_sequence_part2_general_formula_l710_710100

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710100


namespace bales_in_barn_now_l710_710651

-- Define the initial number of bales
def initial_bales : ℕ := 28

-- Define the number of bales added by Tim
def added_bales : ℕ := 26

-- Define the total number of bales
def total_bales : ℕ := initial_bales + added_bales

-- Theorem stating the total number of bales
theorem bales_in_barn_now : total_bales = 54 := by
  sorry

end bales_in_barn_now_l710_710651


namespace tangent_length_min_val_l710_710464

-- Definitions of circle and line
structure Point where
  x : ℝ
  y : ℝ

def on_line (P : Point) : Prop :=
  P.x + 2 * P.y = 3

def on_circle (P : Point) : Prop :=
  (P.x - 1 / 2) ^ 2 + (P.y - 1 / 4) ^ 2 = 1 / 2

-- Proving the tangent line length is equal to sqrt(6)/2
theorem tangent_length_min_val (P : Point) :
  on_line P →
  2^P.x + 4^P.y = 4 * Real.sqrt 2 →
  ∃ l : ℝ, l = Real.sqrt 6 / 2 :=
by
  assume h_line h_min_val
  exists (Real.sqrt 6 / 2)
  sorry

end tangent_length_min_val_l710_710464


namespace bn_arithmetic_sequence_an_formula_l710_710034

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710034


namespace binomial_theorem_logarithm_l710_710397

theorem binomial_theorem_logarithm :
  (∑ r in (finset.range 21), (nat.choose 20 r) * (Real.log 2)^(20 - r) * (Real.log 5)^r) = 1 :=
by
  sorry

end binomial_theorem_logarithm_l710_710397


namespace max_value_of_expression_l710_710266

theorem max_value_of_expression (a b c d : ℕ) (h : {a, b, c, d} = {2, 3, 5, 7}) :
  ∃ (ac bd ab cd : ℚ), ac * bd + ab * cd + ab + cd = 72.25 :=
begin
  sorry
end

end max_value_of_expression_l710_710266


namespace cos_double_angle_square_problem_l710_710393

theorem cos_double_angle_square_problem (a B : ℝ) (b c : ℝ) (θ : ℝ) (h_small_square : a^2 = 1) (h_large_square : B^2 = 25)
(h_sum : b + c = 4) (h_product : b * c = 2) (h_cos_theta : real.cos θ = (b / B)) :
  real.cos (2 * θ) = (4 * real.sqrt 2 - 19) / 25 := by
  -- Definitions and paragraphing conditions prove here.
  sorry

end cos_double_angle_square_problem_l710_710393


namespace total_cost_of_hotel_stay_l710_710642

-- Define the necessary conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- State the problem
theorem total_cost_of_hotel_stay :
  (cost_per_night_per_person * number_of_people * number_of_nights) = 360 := by
  sorry

end total_cost_of_hotel_stay_l710_710642


namespace minimum_distance_between_parabola_and_circle_l710_710566

noncomputable def min_distance_PQ : ℝ :=
  let P := λ y : ℝ, (y ^ 2, y)
  let distance := λ (p1 p2 : ℝ × ℝ), Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let C := (3, 0)
  Real.sqrt (((P 0).1 - C.1)^2 + ((P 0).2 - C.2)^2) - 1

theorem minimum_distance_between_parabola_and_circle (P Q : ℝ × ℝ) :
  (P.2 ^ 2 = P.1 ∧ ((Q.1 - 3) ^ 2 + Q.2 ^ 2 = 1)) → 
  distance P Q = min_distance_PQ :=
sorry

end minimum_distance_between_parabola_and_circle_l710_710566


namespace b_seq_arithmetic_a_seq_formula_l710_710157

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710157


namespace polar_to_cartesian_l710_710611

-- Given definitions
def r (p θ : ℝ) : ℝ := p * Real.sin (5 * θ)
def to_cartesian (x y p : ℝ) : Prop :=
  (x^2 + y^2)^3 = p * y * (5 * x^4 - 10 * x^2 * y^2 + y^4)

theorem polar_to_cartesian (p x y : ℝ) (h : 0 ≤ x ∧ 0 ≤ y) :
  to_cartesian x y p ↔ 
  x^6 - 5 * p * x^4 * y + 10 * p * x^2 * y^3 + y^6 + 3 * x^4 * y^2 - p * y^5 + 3 * x^2 * y^4 = 0 :=
by
  sorry

end polar_to_cartesian_l710_710611


namespace problem_statement_l710_710731

/-- A monochromatic polygon is defined such that all its vertices are colored the same. -/
noncomputable def monochromatic {α : Type*} (colors : α → Prop) (vertices : set α) : Prop :=
  ∃ c, ∀ v ∈ vertices, colors v = c

/-- Main theorem statement:
Given that every point on the plane is colored either red or blue,
we show that there exists either a monochromatic equilateral triangle of side length 2,
a monochromatic equilateral triangle of side length √3,
or a monochromatic rhombus of side length 1. -/
theorem problem_statement :
  (∀ p : ℝ × ℝ, p.1 ∈ real) → ∃ (triangle_or_rhombus : set (ℝ × ℝ)),
    (monochromatic red_or_blue triangle_or_rhombus ∧
     (is_equilateral_triangle triangle_or_rhombus ∧ side_length triangle_or_rhombus = 2) ∨
     (is_equilateral_triangle triangle_or_rhombus ∧ side_length triangle_or_rhombus = real.sqrt 3) ∨
     (is_rhombus triangle_or_rhombus ∧ side_length triangle_or_rhombus = 1)) :=
sorry

end problem_statement_l710_710731


namespace hyperbola_eccentricity_l710_710239

theorem hyperbola_eccentricity (k : ℝ) (e : ℝ) :
    (∀ x y : ℝ, k*x^2 - y^2 = 1 → asymptote_perpendicular_to_line 2x - y + 3 = 0) →
    e = sqrt 5 / 2 :=
by
  assume h
  sorry

end hyperbola_eccentricity_l710_710239


namespace altitude_and_median_eq_l710_710832

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

def slope (P1 P2 : Point) : ℝ :=
  (P2.y - P1.y) / (P2.x - P1.x)

def altitude_line (A B C : Point) : ℝ × ℝ × ℝ :=
  let k_BC := slope B C
  let k_h := -1 / k_BC
  let b := A.y - k_h * A.x
  (k_h, -1, b)

def midpoint (P1 P2 : Point) : Point :=
  Point.mk ((P1.x + P2.x) / 2) ((P1.y + P2.y) / 2)

def median_line (A B C : Point) : ℝ × ℝ × ℝ :=
  let E := midpoint B C
  let k_m := slope A E
  let b := A.y - k_m * A.x
  (-k_m, 1, b)

theorem altitude_and_median_eq (A B C : Point)
  (A_coord : A = {x := 4, y := 0})
  (B_coord : B = {x := 6, y := 7})
  (C_coord : C = {x := 0, y := 8}) :
  altitude_line A B C = (6, -1, 24) ∧
  median_line A B C = (-15 / 2, 1, 30) :=
by
  sorry

end altitude_and_median_eq_l710_710832


namespace illuminate_entire_plane_l710_710439

-- Definitions for the conditions given
structure Point2D := (x : ℝ) (y : ℝ)

def can_direct_spotlights (points : List Point2D) : Prop :=
  ∀ (point : Point2D), point ∈ points → (∃ (direction1 direction2 : ℝ → ℝ → Prop),
    (direction1 = north ∨ direction1 = south ∨ direction1 = east ∨ direction1 = west) ∧
    (direction2 = north ∨ direction2 = south ∨ direction2 = east ∨ direction2 = west) ∧
    illuminates_right_angle point direction1 direction2)

-- Theorem statement
theorem illuminate_entire_plane (points : List Point2D) (h : points.length = 4) : 
  ∃ (directions : List (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop)), 
    (∀ (point : Point2D), point ∈ points → 
      (point, directions) illuminates_entire_plane) :=
sorry

end illuminate_entire_plane_l710_710439


namespace sum_of_prime_factors_of_2_pow_10_minus_1_l710_710643

/-- Statement of the problem -/
theorem sum_of_prime_factors_of_2_pow_10_minus_1 : 
  (let n := 2^10 - 1;
       prime_factors := [3, 11, 31] in 
   prime_factors.sum) = 45 :=
by
  let n := 2^10 - 1; -- n = 1023
  let prime_factors := [3, 11, 31]; -- prime factors of 1023
  show prime_factors.sum = 45; -- needs to show that sum = 45
  { sorry }

end sum_of_prime_factors_of_2_pow_10_minus_1_l710_710643


namespace number_of_correct_statements_is_3_l710_710250

def positive_integers_and_negative_integers := {x : Int // x > 0 ∨ x < 0}
def fractions := {x : ℚ // x > 0 ∨ x < 0}
def rational_numbers := {x : ℚ // ∃ a b : ℤ, b ≠ 0 ∧ x = a / b}
def algebraic_expressions := "An algebraic expression is a monomial or a polynomial"
def zero_property := "0 is not a positive number but it is a non-negative number"

def conditions :=
  [positive_integers_and_negative_integers,
   fractions,
   rational_numbers,
   algebraic_expressions,
   zero_property]

theorem number_of_correct_statements_is_3 : (counts_correct_statements conditions) = 3 :=
by
  sorry

def counts_correct_statements (conditions : List String) : Nat :=
  conditions.filter (fun c => c.contains "correct").length

end number_of_correct_statements_is_3_l710_710250


namespace Fiona_cleaning_time_l710_710181

theorem Fiona_cleaning_time :
  (lilly_fiona_time: ℝ) (lilly_time: ℝ) (fiona_time: ℝ) (fiona_time_minutes: ℝ)
  (h1: lilly_fiona_time = 8)
  (h2: lilly_time = lilly_fiona_time / 4)
  (h3: fiona_time = lilly_fiona_time - lilly_time)
  (fiona_time_minutes = fiona_time * 60) :
  fiona_time_minutes = 360 :=
by 
  sorry

end Fiona_cleaning_time_l710_710181


namespace friends_meeting_games_only_l710_710346

theorem friends_meeting_games_only 
  (M P G MP MG PG MPG : ℕ) 
  (h1 : M + MP + MG + MPG = 10) 
  (h2 : P + MP + PG + MPG = 20) 
  (h3 : MP = 4) 
  (h4 : MG = 2) 
  (h5 : PG = 0) 
  (h6 : MPG = 2) 
  (h7 : M + P + G + MP + MG + PG + MPG = 31) : 
  G = 1 := 
by
  sorry

end friends_meeting_games_only_l710_710346


namespace cost_per_toy_l710_710326

def will_initial_amount : ℕ := 83
def amount_spent_on_game : ℕ := 47
def number_of_toys : ℕ := 9
def remaining_money : ℕ := will_initial_amount - amount_spent_on_game

theorem cost_per_toy : remaining_money / number_of_toys = 4 := 
by
  have remaining_money_calc : remaining_money = 36 := by rfl
  have cost_per_toy_calc : remaining_money / number_of_toys = 4 := by norm_num
  exact cost_per_toy_calc

end cost_per_toy_l710_710326


namespace interval_length_proof_l710_710787

open Real

noncomputable def cot (x : ℝ) := cos x / sin x

/-- Given the conditions on x, prove the total length of valid intervals is 4.57. -/
theorem interval_length_proof :
  let cond1 := (λ x : ℝ, 8 - x^2 ≥ -1)
  let cond2 := (λ x : ℝ, cot x ≥ -1)
  (∑ I in [0, 3.14 / 2].to_list, I.right_endpoint - I.left_endpoint) ≈ 4.57 :=
  sorry

end interval_length_proof_l710_710787


namespace prime_factorization_2006_expr_l710_710795

theorem prime_factorization_2006_expr :
  let a := 2006
  let b := 669
  let c := 1593
  (a^2 * (b + c) - b^2 * (c + a) + c^2 * (a - b)) =
  2 * 3 * 7 * 13 * 29 * 59 * 61 * 191 :=
by
  let a := 2006
  let b := 669
  let c := 1593
  have h1 : 2262 = b + c := by norm_num
  have h2 : 3599 = c + a := by norm_num
  have h3 : 1337 = a - b := by norm_num
  sorry

end prime_factorization_2006_expr_l710_710795


namespace simultaneous_choice_implies_consecutive_choice_l710_710219

-- Definitions of the axioms and related concepts
axiom axiom_of_simultaneous_choice : Prop
axiom axiom_of_consecutive_choice : Prop
axiom infinite_tree (T : Type) (root : T) (children : T → Set T) : Prop :=
  ∀ t : T, ∃ child : T, child ∈ children t

theorem simultaneous_choice_implies_consecutive_choice 
  (sim_choice : axiom_of_simultaneous_choice)
  (infinite_tree_axiom : ∀ (T : Type) (root : T) (children : T → Set T), infinite_tree T root children) :
  axiom_of_consecutive_choice :=
sorry

end simultaneous_choice_implies_consecutive_choice_l710_710219


namespace hexagonal_pyramid_dihedral_angle_l710_710905

/-- 
  In a regular hexagonal pyramid, given that a line is drawn through the centers
  of circles inscribed around a lateral face and the largest diagonal cross-section, 
  prove that the cosine of the dihedral angle at the base is √(3/13).
-/
def dihedral_angle_cosine : Prop :=
  let φ := dihedral_angle at the base of the regular hexagonal pyramid
  in cos(φ) = sqrt (3 / 13)

theorem hexagonal_pyramid_dihedral_angle :
  dihedral_angle_cosine := 
sorry

end hexagonal_pyramid_dihedral_angle_l710_710905


namespace compare_neg_fractions_l710_710400

theorem compare_neg_fractions : - (2 / 3 : ℝ) > - (3 / 4 : ℝ) :=
sorry

end compare_neg_fractions_l710_710400


namespace discriminant_quadratic_5x2_minus_9x_plus_4_l710_710668

theorem discriminant_quadratic_5x2_minus_9x_plus_4 :
  let a := 5
  let b := -9
  let c := 4
  in b^2 - 4 * a * c = 1 := 
by
  -- We directly plug in the values and calculate
  let a := 5
  let b := -9
  let c := 4
  show b^2 - 4 * a * c = 1, from sorry

end discriminant_quadratic_5x2_minus_9x_plus_4_l710_710668


namespace probability_even_or_greater_than_four_l710_710361

theorem probability_even_or_greater_than_four :
  let n := 6 in
  let m := 4 in
  let p := m / n in
  p = (2 : ℚ) / 3 :=
by
  sorry

end probability_even_or_greater_than_four_l710_710361


namespace chandra_monsters_l710_710765

def monsters_day_1 : Nat := 2
def monsters_day_2 : Nat := monsters_day_1 * 3
def monsters_day_3 : Nat := monsters_day_2 * 4
def monsters_day_4 : Nat := monsters_day_3 * 5
def monsters_day_5 : Nat := monsters_day_4 * 6

def total_monsters : Nat := monsters_day_1 + monsters_day_2 + monsters_day_3 + monsters_day_4 + monsters_day_5

theorem chandra_monsters : total_monsters = 872 :=
by
  unfold total_monsters
  unfold monsters_day_1
  unfold monsters_day_2
  unfold monsters_day_3
  unfold monsters_day_4
  unfold monsters_day_5
  sorry

end chandra_monsters_l710_710765


namespace robert_more_photos_than_claire_l710_710970

theorem robert_more_photos_than_claire
  (claire_photos : ℕ)
  (Lisa_photos : ℕ)
  (Robert_photos : ℕ)
  (Claire_takes_photos : claire_photos = 12)
  (Lisa_takes_photos : Lisa_photos = 3 * claire_photos)
  (Lisa_and_Robert_same_photos : Lisa_photos = Robert_photos) :
  Robert_photos - claire_photos = 24 := by
    sorry

end robert_more_photos_than_claire_l710_710970


namespace part1_sequence_arithmetic_part2_general_formula_l710_710146

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710146


namespace center_square_side_length_l710_710712

-- Given conditions
def total_area : ℝ := 400
def l_shaped_fraction : ℝ := 1 / 5
def num_l_shaped : ℝ := 4
def l_shaped_total_area : ℝ := num_l_shaped * l_shaped_fraction * total_area
def center_square_area : ℝ := total_area - l_shaped_total_area
def correct_side_length : ℝ := 4 * Real.sqrt 5

-- The theorem we need to prove
theorem center_square_side_length : Real.sqrt center_square_area = correct_side_length := by
  -- Proof goes here
  sorry

end center_square_side_length_l710_710712


namespace henry_books_count_l710_710493

theorem henry_books_count
  (initial_books : ℕ)
  (boxes : ℕ)
  (books_per_box : ℕ)
  (room_books : ℕ)
  (table_books : ℕ)
  (kitchen_books : ℕ)
  (picked_books : ℕ) :
  initial_books = 99 →
  boxes = 3 →
  books_per_box = 15 →
  room_books = 21 →
  table_books = 4 →
  kitchen_books = 18 →
  picked_books = 12 →
  initial_books - (boxes * books_per_box + room_books + table_books + kitchen_books) + picked_books = 23 :=
by
  intros initial_books_eq boxes_eq books_per_box_eq room_books_eq table_books_eq kitchen_books_eq picked_books_eq
  rw [initial_books_eq, boxes_eq, books_per_box_eq, room_books_eq, table_books_eq, kitchen_books_eq, picked_books_eq]
  norm_num
  sorry

end henry_books_count_l710_710493


namespace value_BP_l710_710932

-- Definition of the problem conditions
variables (A B C D E F G : Type) [HasCoords A] [HasCoords B] [HasCoords C] [HasCoords D]
          [HasCoords E] [HasCoords F] [HasCoords G]
variables (P : Type) [HasCoords P]

-- Triangle with concurrent lines
variables (AD BE CF : Line) (interp : AD.intersects BE = P) (interpf : AD.intersects CF = P)
          (trip_concurrency : ∀ AD BE CF AD.intersects BE = P ∧ AD.intersects CF = P)

-- Given ratios and perpendicularity
variables (ratio_AF_BF_AE_CE : AF/BF = AE/CE) 
          (AD_perp_BC : AD.is_perpendicular BC)

-- Similar triangles and segment constraints
variables (PC PG : ℝ) (segment_PE : PE = 1) (segment_EG : EG = 2) (parallel_line : Is_Parallel AB CG)

-- Prove BP = sqrt(3)
theorem value_BP : BP = sqrt(3) :=
by
  have h1 : CE = PC - PB,
  have h2 : EG = PG - PE,
  have h3 : PG = segment_PE + segment_EG,
  have h4 : BP = sqrt(PE * PG),
  sorry

end value_BP_l710_710932


namespace bn_is_arithmetic_an_general_formula_l710_710119

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710119


namespace range_of_decreasing_function_l710_710864

noncomputable def f (a x : ℝ) : ℝ := 2 * a * x^2 + 4 * (a - 3) * x + 5

theorem range_of_decreasing_function (a : ℝ) :
  (∀ x : ℝ, x < 3 → (deriv (f a) x) ≤ 0) ↔ 0 ≤ a ∧ a ≤ 3/4 := 
sorry

end range_of_decreasing_function_l710_710864


namespace limit_of_sequence_l710_710759

open Real

noncomputable def a (n : ℕ) : ℝ := (sqrt (n * (n + 2)) - sqrt (n ^ 2 - 2 * n + 3))

theorem limit_of_sequence :
  tendsto (λ n : ℕ, a n) at_top (𝓝 2) :=
sorry

end limit_of_sequence_l710_710759


namespace hyperbola_asymptote_equation_l710_710828

noncomputable def hyperbola_condition (a b c : ℝ) := 
  (a > 0) ∧ 
  (b > 0) ∧ 
  ( ∃ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1) ∧ 
  (b = (1/4) * 2 * c)

theorem hyperbola_asymptote_equation (a b c : ℝ) 
  (h : hyperbola_condition a b c) : 
  ∃ (k : ℝ), k = sqrt 3 ∧ (∀ (x y : ℝ), (a > 0 ∧ b > 0) → (b = (1/4) * 2 * c) → (x ± k * y) = 0) := 
sorry

end hyperbola_asymptote_equation_l710_710828


namespace find_m_l710_710455

theorem find_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = Real.sqrt 10 :=
sorry

end find_m_l710_710455


namespace B_salary_after_tax_l710_710658
-- Import the required libraries

-- Define the problem statement and conditions in Lean 4
theorem B_salary_after_tax :
  ∃ (x : ℝ), 
  let A_salary := 1.5 * x,
      total_salary_before_tax := A_salary + x,
      B_tax_deduction := 0.10 * x,
      B_salary_after_tax := x - B_tax_deduction
  in total_salary_before_tax = 570 ∧ B_salary_after_tax = 205.2 :=
by
  sorry

end B_salary_after_tax_l710_710658


namespace part1_sequence_arithmetic_part2_general_formula_l710_710141

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710141


namespace positive_reals_condition_l710_710420

theorem positive_reals_condition (a : ℝ) (h_pos : 0 < a) : a < 2 :=
by
  -- Problem conditions:
  -- There exists a positive integer n and n pairwise disjoint infinite sets A_i
  -- such that A_1 ∪ ... ∪ A_n = ℕ* and for any two numbers b > c in each A_i,
  -- b - c ≥ a^i.

  sorry

end positive_reals_condition_l710_710420


namespace arithmetic_sequence_nth_term_is_4020_l710_710249

noncomputable def sequence (x : ℝ) : ℕ → ℝ
| 0       = 3 * x - 2
| 1       = 7 * x - 15
| 2       = 4 * x + 3
| (n + 3) := sequence 2 + n * d -- We don't need the explicit value of d here

theorem arithmetic_sequence_nth_term_is_4020 :
  ∃ (x : ℝ) (n : ℕ), 
    (sequence x (n - 1) = 4020) → n = 851 :=
begin
  sorry
end

end arithmetic_sequence_nth_term_is_4020_l710_710249


namespace seq_max_val_occurs_l710_710634

def a : ℕ → ℕ
| 1       := 1
| (2 * n) := a n
| (2 * n + 1) := a (2 * n) + 1

theorem seq_max_val_occurs (n : ℕ) (h_n : n = 1989) :
  (∀ k, k ≤ n → a k ≤ 10) ∧ (∃ k, k ≤ n ∧ a k = 10 ∧ (∃ l, (l ≤ n ∧ a l = 10) → l = 1023 ∨ l = 1535 ∨ l = 1791 ∨ l = 1919 ∨ l = 1983)) :=
by
  sorry

end seq_max_val_occurs_l710_710634


namespace sequence_bn_arithmetic_and_an_formula_l710_710127

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710127


namespace same_solution_count_for_inequalities_l710_710811

theorem same_solution_count_for_inequalities (n k : ℕ) :
  (∃ (x : Fin k → ℤ), (∑ i, |x i| ≤ n) ↔ 
   ∃ (y : Fin n → ℤ), (∑ j, |y j| ≤ k)) :=
by sorry

end same_solution_count_for_inequalities_l710_710811


namespace problem_statement_l710_710583

theorem problem_statement (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) : y * (y - 1) ≤ x^2 := 
sorry

end problem_statement_l710_710583


namespace general_term_l710_710926

noncomputable def sequence_a (n : ℕ) : ℚ :=
if n = 1 then 2 
else 
  let a_n := sequence_a (n - 1) in
  (2 * a_n - 4) / (a_n + 6)

theorem general_term :
  ∀ (n : ℕ), n > 0 → sequence_a n = (4 - 2 * n) / n :=
  by
    intro n hn
    induction n using nat.strong_induction_on with n ih
    cases n
    case zero =>
      simp at hn
    case succ m =>
      cases m
      case zero =>
        simp [sequence_a]
      case succ k =>
        have h₁ : sequence_a (k + 2) = (2 * sequence_a (k + 1) - 4) / (sequence_a (k + 1) + 6) := by simp [sequence_a]
        have h₂ : sequence_a (k + 1) = (4 - 2 * (k + 1)) / (k + 1) := by apply ih; simp
        rw [h₂] at h₁
        have := (4 - 2 * (k + 1)) / (k + 1)
        sorry -- Proof steps are omitted.

end general_term_l710_710926


namespace hexagon_area_l710_710522

theorem hexagon_area (s : ℝ) (h : s = 3) : 
  let area_equilateral := (sqrt 3 / 4) * s^2 
  in 4 * area_equilateral = 9 * sqrt 3 :=
by
  rw [h]
  let area_equilateral := (sqrt 3 / 4) * 9
  have calc1 : (sqrt 3 / 4) * (3^2) = (sqrt 3 / 4) * 9 := by rw pow_two 3
  have calc2 : 4 * ((sqrt 3 / 4) * 9) = 9 * sqrt 3 := by ring
  exact calc2
  sorry

end hexagon_area_l710_710522


namespace range_of_m_l710_710925

axiom represents_ellipse (x y m : ℝ) : 
  (m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3) ^ 2) → m > 5

theorem range_of_m (m : ℝ) (h : ∃ x y : ℝ, represents_ellipse x y m) : m > 5 :=
  by { cases h with x hx, cases hx with y hy, exact hy }

end range_of_m_l710_710925


namespace households_neither_car_nor_bike_l710_710525

-- Define the given conditions
def total_households : ℕ := 90
def car_and_bike : ℕ := 18
def households_with_car : ℕ := 44
def bike_only : ℕ := 35

-- Prove the number of households with neither car nor bike
theorem households_neither_car_nor_bike :
  (total_households - ((households_with_car + bike_only) - car_and_bike)) = 11 :=
by
  sorry

end households_neither_car_nor_bike_l710_710525


namespace largest_n_polynomial_roots_exceed_2_l710_710809

theorem largest_n_polynomial_roots_exceed_2 :
  ∃ n : ℤ, (∀ a b c : ℝ, (a^3 - (n + 9) * a^2 + (2 * n^2 - 3 * n - 34) * a + 2 * (n - 4) * (n + 3) = 0) → a > 2 ∧ b > 2 ∧ c < 2) ∧
  (∀ m : ℤ, m > n → ¬ (∀ a b c : ℝ, (a^3 - (m + 9) * a^2 + (2 * m^2 - 3 * m - 34) * a + 2 * (m - 4) * (m + 3) = 0) → a > 2 ∧ b > 2 ∧ c < 2)) :=
begin
  use 8,
  sorry

end largest_n_polynomial_roots_exceed_2_l710_710809


namespace part1_sequence_arithmetic_part2_general_formula_l710_710139

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710139


namespace incorrect_conclusion_intersection_l710_710436

theorem incorrect_conclusion_intersection :
  ∀ (x : ℝ), (0 = -2 * x + 4) → (x = 2) :=
by
  intro x h
  sorry

end incorrect_conclusion_intersection_l710_710436


namespace sum_of_distances_is_correct_l710_710561

def parabola (x : ℝ) : ℝ := 2 * x^2

-- Given points on the intersection
def point1 : ℝ × ℝ := (-14, 392)
def point2 : ℝ × ℝ := (-1, 2)
def point3 : ℝ × ℝ := (6.5, 84.5)
def point4 : ℝ × ℝ := (8.5, 144.5)

def focus : ℝ × ℝ := (0, 1 / 8)

-- Function to calculate the Euclidean distance
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

-- Sum of distances from the focus to each point
def total_distance : ℝ :=
   distance focus point1 +
   distance focus point2 +
   distance focus point3 +
   distance focus point4

-- Proof statement
theorem sum_of_distances_is_correct :
  total_distance = 869.48 := by
  sorry

end sum_of_distances_is_correct_l710_710561


namespace find_value_of_expression_l710_710491

theorem find_value_of_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) : x^2 + 2*x + 3 = 4 := by
  sorry

end find_value_of_expression_l710_710491


namespace linear_function_m_eq_neg1_l710_710446

theorem linear_function_m_eq_neg1 (m : ℝ) (x : ℝ) :
  (∃ f : ℝ → ℝ, f = λ x, (m^2 - m) * x / (m^2 + 1) ∧ (∃ a b : ℝ, f = λ x, a * x + b)) → m = -1 :=
by
  sorry

end linear_function_m_eq_neg1_l710_710446


namespace value_of_f_neg2021_plus_f_2022_l710_710613

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 2 then log (x + 1) / log 2 else sorry -- Extend as needed for other x

lemma even_function (x : ℝ) : f (-x) = f x :=
sorry

lemma periodic_function (x : ℝ) : 0 ≤ x → f (x + 2) = f x :=
sorry

lemma special_interval (x : ℝ) : 0 ≤ x ∧ x < 2 → f x = log (x + 1) / log 2 :=
sorry

theorem value_of_f_neg2021_plus_f_2022 : f (-2021) + f 2022 = 1 :=
sorry

end value_of_f_neg2021_plus_f_2022_l710_710613


namespace sequence_bn_arithmetic_and_an_formula_l710_710129

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710129


namespace b_arithmetic_sequence_general_formula_a_l710_710055

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710055


namespace infinite_rel_prime_pairs_l710_710209

theorem infinite_rel_prime_pairs (x : ℤ) : 
  ∃∞ x, let a := 2 * x + 1, 
             b := 2 * x - 1 in 
             Nat.gcd a b = 1 
             ∧ a > 1 
             ∧ b > 1 
             ∧ (a^b + b^a) % (a + b) = 0 := 
by
  sorry

end infinite_rel_prime_pairs_l710_710209


namespace bn_arithmetic_sequence_an_formula_l710_710036

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710036


namespace distinct_sums_of_cyclic_triplets_l710_710341

open Function

def is_cyclic_triplet (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  f a = b ∧ f b = c ∧ f c = a

theorem distinct_sums_of_cyclic_triplets (f : ℝ → ℝ)
  (h_f : ∀ x, degree (polynomial.of_atomically_polynomial f x) = 3)
  {t : fin 8 → (ℝ × ℝ × ℝ)}
  (h_t : ∀ i, is_cyclic_triplet f (t i).1 (t i).2.1 (t i).2.2)
  (h_distinct : ∀ i j, i ≠ j → t i ≠ t j) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
  (t i).1 + (t i).2.1 + (t i).2.2 ≠ (t j).1 + (t j).2.1 + (t j).2.2 ∧
  (t j).1 + (t j).2.1 + (t j).2.2 ≠ (t k).1 + (t k).2.1 + (t k).2.2 ∧
  (t k).1 + (t k).2.1 + (t k).2.2 ≠ (t i).1 + (t i).2.1 + (t i).2.2 :=
by
  sorry

end distinct_sums_of_cyclic_triplets_l710_710341


namespace OE_perp_AB_AH_CE_BD_concurrent_l710_710657

/-- Two concentric circles with center O. Triangle ABC is circumscribed around the smaller circle.
    Points A, D, H, E lie on the larger circle. OD is perpendicular to AC, and OH is perpendicular
    to CB. Point F is the intersection of AC and DE. Points O, F, A, E are concyclic. -/
variables {O A B C D E F H: Point}
variable [TwoConcentricCircles O]
variable [CircumscribedTriangle ABC]
variable [OnLargerCircle A D H E]
variable (circumCond : OD ⊥ AC ∧ OH ⊥ CB ∧ Intersection AC DE F)
variable (concyclicOFAE : Concyclic O F A E)

/-- Prove that OE is perpendicular to AB. -/
theorem OE_perp_AB : OE ⊥ AB :=
sorry

/-- Prove that lines AH, CE, and BD are concurrent. -/
theorem AH_CE_BD_concurrent : Concurrent (AH, CE, BD) :=
sorry

end OE_perp_AB_AH_CE_BD_concurrent_l710_710657


namespace elizabeth_time_l710_710654

-- Defining the conditions
def tom_time_minutes : ℕ := 120
def time_ratio : ℕ := 4

-- Proving Elizabeth's time
theorem elizabeth_time : tom_time_minutes / time_ratio = 30 := 
by
  sorry

end elizabeth_time_l710_710654


namespace rectangle_ratio_l710_710259

noncomputable def ratio_of_length_to_width (w : ℝ) : ℝ :=
  40 / w

theorem rectangle_ratio (w : ℝ) 
  (hw1 : 35 * (w + 5) = 40 * w + 75) : 
  ratio_of_length_to_width w = 2 :=
by
  sorry

end rectangle_ratio_l710_710259


namespace algebraic_expression_at_2_sum_of_squares_of_roots_positive_integer_m_l710_710878

-- (1) Prove that the value of the given algebraic expression when x = 2 is -1
theorem algebraic_expression_at_2 : 
  (x : ℝ) (h : x = 2) : (x^2 - 4 * x + 3) = -1 := 
by {
  rw h,
  norm_num,
  sorry,
}

-- (2) Prove that if the value of the algebraic expression is 4, then the values of x satisfy 
--     the quadratic equation such that x1^2 + x2^2 = 18
theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) (h : x^2 - 4 * x + 3 = 4) :
  x₁^2 + x₂^2 = 18 :=
by {
  rw h,
  -- Further steps to verify the roots x1 and x2 are such that x1^2 + x2^2 = 18
  sorry,
}

-- (3) If n1 = m^2 - 1 and n2 = 4m^2 - 4m, and if n2 / n1 is an integer, then m = 3
theorem positive_integer_m (m : ℕ) (n₁ n₂ : ℝ) 
  (h₁: n₁ = (m + 2)^2 - 4 * (m + 2) + 3)
  (h₂: n₂ = (2 * m + 1)^2 - 4 * (2 * m + 1) + 3)
  (h₃ : (n₂ / n₁) ∈ ℤ) : 
  m = 3 :=
by {
  sorry,
}

end algebraic_expression_at_2_sum_of_squares_of_roots_positive_integer_m_l710_710878


namespace evaluate_expression_l710_710343

theorem evaluate_expression :
  2 * 7^(-1/3 : ℝ) + (1/2 : ℝ) * Real.log (1/64) / Real.log 2 = -3 := 
  sorry

end evaluate_expression_l710_710343


namespace proportional_relationships_l710_710442

-- Let l, v, t be real numbers indicating distance, velocity, and time respectively.
variables (l v t : ℝ)

-- Define the relationships according to the given formulas
def distance_formula := l = v * t
def velocity_formula := v = l / t
def time_formula := t = l / v

-- Definitions of proportionality
def directly_proportional (x y : ℝ) := ∃ k : ℝ, x = k * y
def inversely_proportional (x y : ℝ) := ∃ k : ℝ, x * y = k

-- The main theorem
theorem proportional_relationships (const_t const_v const_l : ℝ) :
  (distance_formula l v const_t → directly_proportional l v) ∧
  (distance_formula l const_v t → directly_proportional l t) ∧
  (velocity_formula const_l v t → inversely_proportional v t) :=
by
  sorry

end proportional_relationships_l710_710442


namespace integral_even_odd_l710_710503

open Real

theorem integral_even_odd (a : ℝ) :
  (∫ x in -a..a, x^2 + sin x) = 18 → a = 3 :=
by
  intros h
  -- We'll skip the proof
  sorry

end integral_even_odd_l710_710503


namespace incorrect_statement_exists_l710_710685

theorem incorrect_statement_exists :
  ∃ (s : String), s = "A" ∨ s = "B" ∨ s = "C" ∨ s = "D" ∧ incorrect s :=
by
  let A_incorrect : Bool := (90 - α < 180 - α)
  let B_incorrect : Bool := ¬(∀ θ₁ θ₂, θ₁ + θ₂ = 180 → θ₁ = θ₂)
  let C_incorrect : Bool := ¬(∀ a b c d, a = b ∧ b = c ∧ c = d → a = 90)
  let D_incorrect : Bool := ∃ l m, l ∥ m ∧ l ≠ m
  existsi "A"
  apply or.inl
  apply and.intro
  exact A_incorrect
  -- Proofs for B_incorrect, C_incorrect, D_incorrect would follow the same pattern
  sorry

-- Definitions and auxiliary results omitted for brevity.

end incorrect_statement_exists_l710_710685


namespace solve_for_k_l710_710475

theorem solve_for_k (x y k : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : (1 / 2)^(25 * x) * (1 / 81)^k = 1 / (18 ^ (25 * y))) :
  k = 25 * y / 2 :=
by
  sorry

end solve_for_k_l710_710475


namespace b_seq_arithmetic_a_seq_formula_l710_710154

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710154


namespace ff1_is_1_l710_710177

noncomputable def f (x : ℝ) := Real.log x - 2 * x + 3

theorem ff1_is_1 : f (f 1) = 1 := by
  sorry

end ff1_is_1_l710_710177


namespace sum_largest_odd_factor_bound_l710_710954

def largestOddFactor (x : ℕ) : ℕ :=
  if x = 0 then 0 else
  let rec oddFactor (y : ℕ) : ℕ :=
    if y % 2 = 1 then y else oddFactor (y / 2)
  oddFactor x

theorem sum_largest_odd_factor_bound (x : ℕ) (h : 0 < x) :
  |∑ n in Finset.range (x + 1), (largestOddFactor n) / n - (2 / 3) * x| < 1 := sorry

end sum_largest_odd_factor_bound_l710_710954


namespace carmen_candle_burn_time_l710_710764

theorem carmen_candle_burn_time 
  (burn_time_first_scenario : ℕ)
  (nights_per_candle : ℕ)
  (total_candles_second_scenario : ℕ)
  (total_nights_second_scenario : ℕ)
  (h1 : burn_time_first_scenario = 1)
  (h2 : nights_per_candle = 8)
  (h3 : total_candles_second_scenario = 6)
  (h4 : total_nights_second_scenario = 24) :
  (total_candles_second_scenario * nights_per_candle) / total_nights_second_scenario = 2 :=
by
  sorry

end carmen_candle_burn_time_l710_710764


namespace representation_1_requires_eight_terms_l710_710982

def in_arith_prog (n : ℕ) : ℕ := 2 + 3 * n

def is_representation_1 (terms : List ℕ) : Prop :=
  (terms.nodup ∧ (∀ t ∈ terms, ∃ n : ℕ, t = in_arith_prog n)) ∧ (terms.map (λ x => (1 : ℚ) / x)).sum = 1

theorem representation_1_requires_eight_terms (terms : List ℕ) :
  is_representation_1 terms → 8 ≤ terms.length :=
begin
  sorry
end

end representation_1_requires_eight_terms_l710_710982


namespace part1_arithmetic_sequence_part2_general_formula_l710_710094

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710094


namespace problem_conditions_l710_710087

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710087


namespace single_solution_of_quadratic_l710_710234

theorem single_solution_of_quadratic (b : ℝ) (hb : b ≠ 0) (h : (16^2 - 4 * b * 12) = 0) :
  ∃ x : ℝ, (bx^2 + 16x + 12 = 0) ∧ x = -(3 / 2) :=
begin
  sorry
end

end single_solution_of_quadratic_l710_710234


namespace find_x_l710_710433

theorem find_x (x : ℝ) :
  (x^2 - 7 * x + 12) / (x^2 - 9 * x + 20) = (x^2 - 4 * x - 21) / (x^2 - 5 * x - 24) -> x = 11 :=
by
  sorry

end find_x_l710_710433


namespace find_p_q_r_l710_710819

theorem find_p_q_r (p q r : ℤ) :
  (∀ x : ℤ, x^2 + p * x - 2 = 0 ↔ x ∈ { -2, 1 } ∨ x ∈ { -2, 5 }) →
  (∀ x : ℤ, x^2 + q * x + r = 0 ↔ x ∈ { -2, 5 }) →
  p = -1 ∧ q = -3 ∧ r = -10 :=
begin
  intros hA hB,
  -- proof goes here
  sorry,
end

end find_p_q_r_l710_710819


namespace area_ratio_of_triangles_l710_710976

open Real

def area_ratio (K : ℝ) : ℝ :=
  let area_AHB := 3 * K
  let area_ACE := 4 * K
  area_AHB / area_ACE

theorem area_ratio_of_triangles (K : ℝ) : area_ratio K = 3 / 4 :=
by
  unfold area_ratio
  simp
  sorry

end area_ratio_of_triangles_l710_710976


namespace second_consecutive_odd_integer_l710_710640

theorem second_consecutive_odd_integer (n : ℤ) : 
  (n - 2) + (n + 2) = 152 → n = 76 := 
by 
  sorry

end second_consecutive_odd_integer_l710_710640


namespace each_vowel_written_same_number_of_times_l710_710180

theorem each_vowel_written_same_number_of_times
  (vowels : ℕ)
  (total_alphabets : ℕ)
  (h_vowels_count : vowels = 5)
  (h_total_alphabets : total_alphabets = 15)
  (h_equal_distribution : ∀ (n : ℕ), total_alphabets = vowels * n → n = 3) : 
  ∃ (n : ℕ), n = 3 := 
by {
  existsi 3,
  simp,
  exact h_equal_distribution 3 (by simp [h_vowels_count, h_total_alphabets]),
}

end each_vowel_written_same_number_of_times_l710_710180


namespace slant_height_of_cone_l710_710637

theorem slant_height_of_cone (r : ℝ) (CSA : ℝ) (h_radius : r = 5) (h_CSA : CSA = 157.07963267948966) : 
  let π := Real.pi in
  r * l = CSA → l = 10 :=
by {
  sorry
}

end slant_height_of_cone_l710_710637


namespace inverse_proportion_point_l710_710872

theorem inverse_proportion_point (a : ℝ) (h : (a, 7) ∈ {p : ℝ × ℝ | ∃ x y, y = 14 / x ∧ p = (x, y)}) : a = 2 :=
by
  sorry

end inverse_proportion_point_l710_710872


namespace b_seq_arithmetic_a_seq_formula_l710_710161

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710161


namespace problem_statement_l710_710848

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f(-x) = f(x)
axiom f_periodic : ∀ x : ℝ, x ≥ 0 → f(x+2) = f(x)
axiom f_def : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f(x) = x^2 - x + 1

theorem problem_statement : f(-2014) + f(1) = 2 := 
sorry

end problem_statement_l710_710848


namespace valerie_light_bulbs_deficit_l710_710284

theorem valerie_light_bulbs_deficit :
  let small_price := 8.75
  let medium_price := 11.25
  let large_price := 15.50
  let xsmall_price := 6.10
  let budget := 120
  
  let lamp_A_cost := 2 * small_price
  let lamp_B_cost := 3 * medium_price
  let lamp_C_cost := large_price
  let lamp_D_cost := 4 * xsmall_price
  let lamp_E_cost := 2 * large_price
  let lamp_F_cost := small_price + medium_price

  let total_cost := lamp_A_cost + lamp_B_cost + lamp_C_cost + lamp_D_cost + lamp_E_cost + lamp_F_cost

  total_cost - budget = 22.15 :=
by
  sorry

end valerie_light_bulbs_deficit_l710_710284


namespace part1_part2_l710_710178

noncomputable def seq_a : ℕ → ℚ := λ n, (2 * n - 1) / (2 * n + 1)
noncomputable def prod_T : ℕ → ℚ := λ n, (List.prod (List.map seq_a (List.range n)))
noncomputable def sum_S : ℕ → ℚ := λ n, (List.sum (List.map (λ i, (prod_T i)^2) (List.range n)))

theorem part1 (n : ℕ) (h : n > 0) : 2 * prod_T n = 1 - seq_a n :=
sorry

theorem part2 (n : ℕ) (h : n > 0) : sum_S n < 1 / 4 :=
sorry

end part1_part2_l710_710178


namespace b_arithmetic_sequence_general_formula_a_l710_710059

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710059


namespace b_seq_arithmetic_a_seq_formula_l710_710159

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710159


namespace value_of_p_minus_2_q_minus_2_l710_710165

noncomputable def quadratic_roots (a b c : ℝ) : (ℝ × ℝ) :=
let D := b^2 - 4*a*c in
if h : D ≥ 0 then
    ((-b + real.sqrt D) / (2 * a), (-b - real.sqrt D) / (2 * a))
else
    (0, 0)  -- we are only interested in real roots

theorem value_of_p_minus_2_q_minus_2 :
  let (p, q) := quadratic_roots 3 5 (-7) in
  (p - 2) * (q - 2) = 5 :=
by
  -- The roots of the equation 3x^2 + 5x - 7 = 0 can be represented as p and q
  let (p, q) := quadratic_roots 3 5 (-7) in
  have sum_roots : p + q = -5 / 3 := 
    by
      sorry,  -- using Vieta's formulas for sum of roots
  have product_roots : p * q = -7 / 3 := 
    by
      sorry,  -- using Vieta's formulas for product of roots
  calc
    (p - 2) * (q - 2) = p * q - 2 * (p + q) + 4 : by ring
                   ... = -7 / 3 - 2 * (-5 / 3) + 4 : by rw [product_roots, sum_roots]
                   ... = -7 / 3 + 10 / 3 + 4 : by ring
                   ... = 3 / 3 + 4 : by ring
                   ... = 1 + 4 : by ring
                   ... = 5 : by ring

end value_of_p_minus_2_q_minus_2_l710_710165


namespace budget_allocation_l710_710354

theorem budget_allocation 
  (total_degrees : ℝ := 360)
  (total_budget : ℝ := 100)
  (degrees_basic_astrophysics : ℝ := 43.2)
  (percent_microphotonics : ℝ := 12)
  (percent_home_electronics : ℝ := 24)
  (percent_food_additives : ℝ := 15)
  (percent_industrial_lubricants : ℝ := 8) :
  ∃ percent_genetically_modified_microorganisms : ℝ,
  percent_genetically_modified_microorganisms = 29 :=
sorry

end budget_allocation_l710_710354


namespace sin_neg_4_div_3_pi_l710_710344

theorem sin_neg_4_div_3_pi : Real.sin (- (4 / 3) * Real.pi) = Real.sqrt 3 / 2 :=
by sorry

end sin_neg_4_div_3_pi_l710_710344


namespace sin_2013_eq_neg_sin_33_l710_710597

theorem sin_2013_eq_neg_sin_33 : sin (2013 * real.pi / 180) = -sin (33 * real.pi / 180) := by
  sorry

end sin_2013_eq_neg_sin_33_l710_710597


namespace prime_gt_three_square_minus_one_divisible_by_twentyfour_l710_710208

theorem prime_gt_three_square_minus_one_divisible_by_twentyfour (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) : 24 ∣ (p^2 - 1) :=
sorry

end prime_gt_three_square_minus_one_divisible_by_twentyfour_l710_710208


namespace min_sum_is_83_l710_710519

noncomputable def minSum (grid : ℕ → ℕ → ℕ) : ℕ :=
  ∑ i in Finset.range 5, (∑ j in Finset.range 7, grid i j)

lemma grid_conditions (grid : ℕ → ℕ → ℕ) :
  (∀ i j, (i < 5) ∧ (j < 7) → grid i j ∈ {1, 2, 3}) ∧
  (∀ i j, (i < 4) ∧ (j < 6) → {grid i j, grid (i+1) j, grid i (j+1), grid (i+1) (j+1)} = {1, 2, 3}) :=
  sorry

theorem min_sum_is_83 : 
  ∃ grid : ℕ → ℕ → ℕ, 
    (∀ i j, (i < 5) ∧ (j < 7) → grid i j ∈ {1, 2, 3}) ∧
    (∀ i j, (i < 4) ∧ (j < 6) → {grid i j, grid (i+1) j, grid i (j+1), grid (i+1) (j+1)} = {1, 2, 3}) ∧
    (minSum grid) = 83 :=
sorry

end min_sum_is_83_l710_710519


namespace relatively_prime_probability_l710_710312

theorem relatively_prime_probability (n : ℕ) (h : n = 42) :
  let phi := n * (1 - 1 / 2) * (1 - 1 / 3) * (1 - 1 / 7) in
  (phi / n) = 2 / 7 :=
by
  sorry

end relatively_prime_probability_l710_710312


namespace bn_is_arithmetic_an_general_formula_l710_710108

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710108


namespace function_has_exactly_one_zero_l710_710507

open Set

-- Conditions
def a_gt_3 (a : ℝ) : Prop := a > 3
def f (x a : ℝ) : ℝ := x^2 - a * x + 1

-- Theorem Statement
theorem function_has_exactly_one_zero (a : ℝ) (h : a_gt_3 a) :
  ∃! x ∈ Ioo 0 2, f x a = 0 := sorry

end function_has_exactly_one_zero_l710_710507


namespace sequence_bn_arithmetic_and_an_formula_l710_710131

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710131


namespace incenter_eq_angle_bisectors_intersection_l710_710258

-- Define the triangle type
structure Triangle (α : Type) := 
  (A B C : α)

-- Define a function that returns the incenter
def incenter {α : Type} [EuclideanSpace α] (T : Triangle α) : α :=
  intersection (angleBisector T.A T.B T.C) (angleBisector T.B T.C T.A)

-- Define a theorem stating that the incenter is the intersection of the angle bisectors
theorem incenter_eq_angle_bisectors_intersection (T : Triangle α) : 
  ∃ P : α, P = incenter T ↔ 
    (P ∈ angleBisector T.A T.B T.C ∧ 
     P ∈ angleBisector T.B T.C T.A ∧ 
     P ∈ angleBisector T.C T.A T.B) :=
begin
  sorry
end

end incenter_eq_angle_bisectors_intersection_l710_710258


namespace pencil_cost_is_11_l710_710188

-- Define the initial and remaining amounts
def initial_amount : ℤ := 15
def remaining_amount : ℤ := 4

-- Define the cost of the pencil
def cost_of_pencil : ℤ := initial_amount - remaining_amount

-- The statement we need to prove
theorem pencil_cost_is_11 : cost_of_pencil = 11 :=
by
  sorry

end pencil_cost_is_11_l710_710188


namespace john_gives_to_stud_owner_l710_710936

variable (initial_puppies : ℕ) (puppies_given_away : ℕ) (puppies_kept : ℕ) (price_per_puppy : ℕ) (profit : ℕ)

theorem john_gives_to_stud_owner
  (h1 : initial_puppies = 8)
  (h2 : puppies_given_away = initial_puppies / 2)
  (h3 : puppies_kept = 1)
  (h4 : price_per_puppy = 600)
  (h5 : profit = 1500) :
  let puppies_left_to_sell := initial_puppies - puppies_given_away - puppies_kept
  let total_sales := puppies_left_to_sell * price_per_puppy
  total_sales - profit = 300 :=
by
  intro puppies_left_to_sell
  intro total_sales
  sorry

end john_gives_to_stud_owner_l710_710936


namespace intercept_condition_l710_710894

theorem intercept_condition (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0) :
  (∃ x y : ℝ, a * x + b * y + c = 0 ∧ x = -c / a ∧ y = -c / b ∧ x = y) → (c = 0 ∨ a = b) :=
by
  sorry

end intercept_condition_l710_710894


namespace sequence_bn_arithmetic_and_an_formula_l710_710133

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710133


namespace relatively_prime_probability_42_l710_710299

theorem relatively_prime_probability_42 : 
  (λ x, (x ≤ 42 ∧ x > 0 ∧ Nat.gcd x 42 = 1)) / (λ x, (x ≤ 42 ∧ x > 0)) = 2/7 :=
by 
  sorry

end relatively_prime_probability_42_l710_710299


namespace problem_statement_l710_710967

def groups : ℕ → List ℕ
| 0 => []
| 1 => [1]
| 2 => [2, 4]
| 3 => [3, 5, 7]
| 4 => [6, 8, 10, 12]
| 5 => [9, 11, 13, 15, 17]
-- This is a simplified version. In reality, you would define the complete pattern.

def Sn (n : ℕ) : ℕ :=
  (groups n).sum

theorem problem_statement (n : ℕ) : (Sn (2 * n + 1)) / (2 * n + 1) - (Sn (2 * n)) / (2 * n) = 2 * n :=
by sorry

end problem_statement_l710_710967


namespace problem_conditions_l710_710084

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710084


namespace base4_addition_l710_710624

def base4_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n + 1 => (n % 10) + 4 * base4_to_base10 (n / 10)

def base10_to_base4 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n + 1 => (n % 4) + 10 * base10_to_base4 (n / 4)

theorem base4_addition :
  base10_to_base4 (base4_to_base10 234 + base4_to_base10 73) = 1203 := by
  sorry

end base4_addition_l710_710624


namespace max_intersection_points_l710_710356

theorem max_intersection_points (circle : Type) (line : Type) (triangle : Type) : 
  (∃ P1 P2 P3 : Type, 
    (∀ (C : circle) (L1 L2 : line) (T : triangle),
      (set.range (λ p : circle × line, ∃ x, p.1 = C ∧ p.2 = L1 ∨ p.1 = C ∧ p.2 = L2 ∧ P1 x) ∧ 
       set.range (λ l : line × line, ∃ x, l.1 = L1 ∧ l.2 = L2 ∧ P2 x) ∧ 
       set.range (λ t : circle × triangle, ∃ x, t.1 = C ∧ t.2 = T ∧ P3 x)) ∧ 
      4 + 1 + 6 + 6 = 17) :=
begin
  sorry
end

end max_intersection_points_l710_710356


namespace find_x_from_conditions_l710_710537

theorem find_x_from_conditions (x y : ℝ)
  (h1 : (6 : ℝ) = (1 / 2 : ℝ) * x)
  (h2 : y = (1 / 2 :ℝ) * 10)
  (h3 : x * y = 60) : x = 12 := by
  sorry

end find_x_from_conditions_l710_710537


namespace prob_relatively_prime_42_l710_710316

noncomputable def euler_totient (n : ℕ) : ℕ :=
  (List.range n).filter (λ i => Nat.gcd i n = 1).length

theorem prob_relatively_prime_42 : 
  (euler_totient 42 : ℚ) / 42 = 2 / 7 := 
by
  sorry

end prob_relatively_prime_42_l710_710316


namespace bob_guarantees_4_fish_l710_710374

theorem bob_guarantees_4_fish : 
  ∀ (coloring : Fin 20 → Bool),  ∃ (segments : Finset (Fin (20 × 20))), segments.card ≥ 4 ∧ 
  (∀ (segment ∈ segments), let ⟨i, j⟩ := segment in i ≠ j ∧ coloring i = coloring j) :=
by sorry

end bob_guarantees_4_fish_l710_710374


namespace digit_in_60th_position_l710_710246

def repeating_decimal := [4, 5, 3]

theorem digit_in_60th_position : 
  ∀ (n : ℕ), 
    n = 60 →
    let cycle_length := 3 in
    let digit_sequence := [2] ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ repeating_decimal ++ [4, 5]
    (digit_sequence[n - 2] = 5) :=
by
  intros n hn
  simp only [nat.succ_eq_add_one, add_tsub_cancel_left]
  sorry

end digit_in_60th_position_l710_710246


namespace constant_term_expansion_l710_710290

theorem constant_term_expansion (k: ℕ) (h1: 2 * k = 8) : 
  (choose 8 k * 3^(8-k) * 2^k : ℤ) = 90720 := by
  sorry

end constant_term_expansion_l710_710290


namespace correlation_coefficient_properties_l710_710682

theorem correlation_coefficient_properties (r : ℝ) (h1 : r ∈ Icc (-1 : ℝ) (1 : ℝ)) :
  |r| ≤ 1 ∧ (∀ r₁ r₂, |r₁| ≥ |r₂| → degree_of_correlation(r₁) ≥ degree_of_correlation(r₂)) :=
sorry

end correlation_coefficient_properties_l710_710682


namespace necessary_but_not_sufficient_condition_l710_710705

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∃ x, x > 2 ∧ ¬ (x > 3)) ∧ 
  (∀ x, x > 3 → x > 2) := by sorry

end necessary_but_not_sufficient_condition_l710_710705


namespace limit_of_sequence_l710_710758

open Real

noncomputable def a (n : ℕ) : ℝ := (sqrt (n * (n + 2)) - sqrt (n ^ 2 - 2 * n + 3))

theorem limit_of_sequence :
  tendsto (λ n : ℕ, a n) at_top (𝓝 2) :=
sorry

end limit_of_sequence_l710_710758


namespace solve_for_y_l710_710226

theorem solve_for_y (y : ℚ) (h : 1 / 3 + 1 / y = 7 / 9) : y = 9 / 4 :=
by
  sorry

end solve_for_y_l710_710226


namespace quarters_total_l710_710977

def initial_quarters : ℕ := 21
def additional_quarters : ℕ := 49
def total_quarters : ℕ := initial_quarters + additional_quarters

theorem quarters_total : total_quarters = 70 := by
  sorry

end quarters_total_l710_710977


namespace number_of_pupils_l710_710330

theorem number_of_pupils
  (pupil_mark_wrong : ℕ)
  (pupil_mark_correct : ℕ)
  (average_increase : ℚ)
  (n : ℕ)
  (h1 : pupil_mark_wrong = 73)
  (h2 : pupil_mark_correct = 45)
  (h3 : average_increase = 1/2)
  (h4 : 28 / n = average_increase) : n = 56 := 
sorry

end number_of_pupils_l710_710330


namespace lines_perpendicular_to_line_in_plane_l710_710843

theorem lines_perpendicular_to_line_in_plane (a : Set Point)
                                             (alpha : Set Point)
                                             (H1 : ¬ (∃ p : Point, ∀ q : Point, (q ∈ alpha) → (inner_product p q = 0)))
                                             : ∃ (l : Set Point), (∀ q : Point, q ∈ alpha → l ⊥ a) :=
by
  sorry

end lines_perpendicular_to_line_in_plane_l710_710843


namespace part1_part2_l710_710485

-- Part (1)
theorem part1 (m : ℝ) (A B : set ℝ) (hA : A = {x | -5 ≤ x ∧ x ≤ 2}) (hB : B = set.Icc (-2 * m + 1) (-m - 1)) (h_union : A ∪ B = A) : 2 < m ∧ m ≤ 3 :=
  by
    have h₁ : B ≠ ∅,
    sorry
    have h₂ : B ⊆ A,
    sorry

    -- Solve the inequalities here
    sorry

-- Part (2)
theorem part2 (m : ℝ) (A B : set ℝ) (hA : A = {x | -5 ≤ x ∧ x ≤ 2}) (hB : B = {x | -2 * m + 1 ≤ x ∧ x ≤ -m - 1}) (h_union : A ∪ B = A) : m ≤ 3 :=
  by
    have h₁ : B ⊆ A ∨ B = ∅,
    sorry

    -- Solve the inequalities here
    sorry

end part1_part2_l710_710485


namespace only_book_A_l710_710697

variable (numA numB numBoth numOnlyB x : ℕ)
variable (h1 : numA = 2 * numB)
variable (h2 : numBoth = 500)
variable (h3 : numBoth = 2 * numOnlyB)
variable (h4 : numB = numOnlyB + numBoth)
variable (h5 : x = numA - numBoth)

theorem only_book_A : 
  x = 1000 := 
by
  sorry

end only_book_A_l710_710697


namespace additional_investment_interest_rate_l710_710394

theorem additional_investment_interest_rate :
  let initial_investment := 2400
  let initial_rate := 0.05
  let additional_investment := 600
  let total_investment := initial_investment + additional_investment
  let desired_total_income := 0.06 * total_investment
  let income_from_initial := initial_rate * initial_investment
  let additional_income_needed := desired_total_income - income_from_initial
  let additional_rate := additional_income_needed / additional_investment
  additional_rate * 100 = 10 :=
by
  sorry

end additional_investment_interest_rate_l710_710394


namespace hyperbola_eccentricity_is_correct_l710_710443

variable {a b c : ℝ}
variable {F1 F2 M P : ℝ × ℝ}

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  e

theorem hyperbola_eccentricity_is_correct :
  ∀ a b : ℝ, a > 0 → b > 0 →
  let c := Real.sqrt (a^2 + b^2) in 
  let F1 : ℝ × ℝ := (-c, 0) in
  let F2 : ℝ × ℝ := (c, 0) in
  -- We assume existence of M and P with specified properties.
  ∃ M P : ℝ × ℝ, 
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧ -- P lies on the hyperbola.
  (distance P F1 = c) ∧            -- |PF1| = c
  (distance P F2 = c * Real.sqrt 3) -- |PF2| = c√3
  →
  hyperbola_eccentricity a b = Real.sqrt 3 + 1 := 
sorry

end hyperbola_eccentricity_is_correct_l710_710443


namespace b_arithmetic_sequence_general_formula_a_l710_710062

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710062


namespace prob_one_color_after_4_draws_prob_distribution_X_expectation_X_l710_710471

-- Definitions for the problem
def white_balls_initial : ℕ := 6
def yellow_balls_initial : ℕ := 4
def total_draws : ℕ := 4

-- Probabilities and expectations
def prob_one_color (w: ℕ) (y: ℕ) (draws: ℕ) : ℚ := (w/(w+y)) * ((w+1)/(w+y+1)) * ((w+2)/(w+y+2)) * ((w+3)/(w+y+3))

noncomputable def prob_X_equals (x : ℕ) (w: ℕ) (y: ℕ) (draws: ℕ) : ℚ :=
  match x with
  | 2  => (y/(w+y)) * ((y+1)/(w+y+1)) * ((y+2)/(w+y+2)) * ((y+3)/(w+y+3))
  | 4  => (w/(w+y)) * ((y+1)/(w+y+1)) * ((y+2)/(w+y+2)) * ((y+3)/(w+y+3))
  | 6  => 1 - ( (2/(w+y)) * prob_X_equals 4 w y draws +
                 (3/(w+y)) * prob_X_equals 8 w y draws + 
                 4 * prob_X_equals 2 w y draws )
  | 8  => (y/(w+y)) * ((w+1)/(w+y+1)) * ((w+2)/(w+y+2)) * ((w+3)/(w+y+3))
  | 10 => prob_one_color w y draws
  | _ => 0

noncomputable def expectation (w: ℕ) (y: ℕ) (draws: ℕ) : ℚ :=
  2 * prob_X_equals 2 w y draws +
  4 * prob_X_equals 4 w y draws +
  6 * prob_X_equals 6 w y draws +
  8 * prob_X_equals 8 w y draws +
  10 * prob_X_equals 10 w y draws

-- Theorems for the problem
theorem prob_one_color_after_4_draws : prob_one_color white_balls_initial yellow_balls_initial total_draws = 189 / 625 := sorry

theorem prob_distribution_X (x : ℕ) (w: ℕ) (y: ℕ) (draws: ℕ) :
  prob_X_equals x white_balls_initial yellow_balls_initial total_draws ∈ {21/250, 19/125, 131/625, 63/250, 189/625} := sorry

theorem expectation_X : expectation white_balls_initial yellow_balls_initial total_draws = 4421 / 625 := sorry

end prob_one_color_after_4_draws_prob_distribution_X_expectation_X_l710_710471


namespace b_seq_arithmetic_a_seq_formula_l710_710160

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710160


namespace surface_area_of_circumscribed_sphere_l710_710928

-- Define the points P, A, B, and C
variables {P A B C : Type} [Point P] [Point A] [Point B] [Point C]

-- Define the conditions
variable (AC_perpendicular_to_plane_PAB : ⟂ AC (plane_of P A B))
variable (AB_eq : AB = 6)
variable (AC_eq : AC = 6)
variable (BP_eq : BP = 2 * Real.sqrt 2)
variable (angle_ABP_eq : ∠A B P = 45)

-- Define the main theorem statement
theorem surface_area_of_circumscribed_sphere :
  (circumscribed_sphere_surface_area P A B C) = 76 * Real.pi :=
sorry

end surface_area_of_circumscribed_sphere_l710_710928


namespace probability_inside_octahedron_l710_710371

noncomputable def probability_of_octahedron : ℝ := 
  let cube_volume := 8
  let octahedron_volume := 4 / 3
  octahedron_volume / cube_volume

theorem probability_inside_octahedron :
  probability_of_octahedron = 1 / 6 :=
  by
    sorry

end probability_inside_octahedron_l710_710371


namespace problem_conditions_l710_710079

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710079


namespace intersect_segment_ineq_l710_710452

theorem intersect_segment_ineq (b : ℝ) :
  let A := (1 : ℝ, 0)
  let B := (-1 : ℝ, 0)
  let L := λ x y, 2 * x + y - b
  L 1 0 * L (-1) 0 ≤ 0 ↔ -2 ≤ b ∧ b ≤ 2 :=
by sorry

end intersect_segment_ineq_l710_710452


namespace concrete_for_supporting_pillars_l710_710550

-- Define the given conditions
def roadway_deck_concrete : ℕ := 1600
def one_anchor_concrete : ℕ := 700
def total_bridge_concrete : ℕ := 4800

-- State the theorem
theorem concrete_for_supporting_pillars :
  let total_anchors_concrete := 2 * one_anchor_concrete in
  let total_deck_and_anchors_concrete := roadway_deck_concrete + total_anchors_concrete in
  total_bridge_concrete - total_deck_and_anchors_concrete = 1800 :=
by
  sorry

end concrete_for_supporting_pillars_l710_710550


namespace mul_72518_9999_eq_725107482_l710_710693

theorem mul_72518_9999_eq_725107482 : 72518 * 9999 = 725107482 := by
  sorry

end mul_72518_9999_eq_725107482_l710_710693


namespace angle_equality_l710_710448

-- Define the conditions
variables {A B C D X K L T : Type*} [linear_ordered_field Type*] -- Define variables as necessary
variables (hABC : A ≠ B ∧ A ≠ C ∧ B ≠ C)
variables (hBCA : ∠ B C A = 90)
variables (hD : is_foot_of_altitude C A B D)
variables (hX : is_chosen_on_segment C D X)
variables (hBK : is_on_segment A X K ∧ BK = BC)
variables (hAL : is_on_segment B X L ∧ AL = AC)
variables (hConcyclic : concyclic_points K L T D)

-- Theorem statement
theorem angle_equality (hABC hBCA hD hX hBK hAL hConcyclic) :
  ∠ A C T = ∠ B C T :=
sorry -- Proof goes here

-- Definitions for clarity
def is_foot_of_altitude (C A B D : Type*) [linear_ordered_field Type*] : Prop := sorry
def is_chosen_on_segment (C D X : Type*) [linear_ordered_field Type*] : Prop := sorry
def is_on_segment (P Q R : Type*) [linear_ordered_field Type*] : Prop := sorry
def concyclic_points (K L T D : Type*) [linear_ordered_field Type*] : Prop := sorry

-- Ensure the Lean environment knows these definitions are non-computable 
noncomputable theory

end angle_equality_l710_710448


namespace func_translation_right_symm_yaxis_l710_710616

def f (x : ℝ) : ℝ := sorry

theorem func_translation_right_symm_yaxis (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x - 1) = e ^ (-x)) :
  ∀ x, f x = e ^ (-x - 1) := sorry

end func_translation_right_symm_yaxis_l710_710616


namespace factorize_expression_l710_710415

variable {X M N : ℕ}

theorem factorize_expression (x m n : ℕ) : x * m - x * n = x * (m - n) :=
sorry

end factorize_expression_l710_710415


namespace f_neg4_eq_1_over_16_l710_710862

def f : ℝ → ℝ
| x => if x < 3 then f (x + 2) else (1 / 2) ^ x

theorem f_neg4_eq_1_over_16 : f (-4) = 1 / 16 :=
by
  sorry

end f_neg4_eq_1_over_16_l710_710862


namespace matrix_multiplication_correct_l710_710767

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 1],
  ![4, -2]
]

def matrix2 : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![5, -1],
  ![2, 4]
]

def result : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![17, 1],
  ![16, -12]
]

theorem matrix_multiplication_correct :
  matrix1 ⬝ matrix2 = result := by
  sorry

end matrix_multiplication_correct_l710_710767


namespace deleted_test_mark_l710_710198

-- Definitions based on the conditions
def average_five_tests := 73
def average_four_tests := 76

-- Proving the deleted test mark
theorem deleted_test_mark :
  let total_five := 5 * average_five_tests,
      total_four := 4 * average_four_tests in
  total_five - total_four = 61 := by
  sorry

end deleted_test_mark_l710_710198


namespace convex_polygon_interior_angles_l710_710627

theorem convex_polygon_interior_angles (n : ℕ) 
  (h1 : 4 > 0) 
  (h2 : 170 < 180) 
  (h3 : ∑ i in Finset.range n, (170 - 4*(n-1-i)) = 180 * (n - 2)) :
  n = 12 := 
sorry

end convex_polygon_interior_angles_l710_710627


namespace person_who_visited_both_sites_l710_710607

def A_statement := "I haven't visited either site."
def B_statement := "I visited the Sanxingdui site with A."
def C_statement := "I visited the Jinsha site with B."
def D_statement := "The person who visited both sites is neither me nor B."

noncomputable def visited_both_sites (A B C D : Prop) : Prop :=
A ∨ B ∨ C ∨ D ∧ (¬A ∧ ¬B ∧ C ∧ ¬D)

theorem person_who_visited_both_sites
  (A_visited_neither : Prop) (B_visited_sanxingdui_with_A : Prop)
  (C_visited_jinsha_with_B : Prop) (D_not_me_nor_B : Prop)
  (A_true : A_visited_neither)
  (B_false : ¬ B_visited_sanxingdui_with_A)
  (C_true : C_visited_jinsha_with_B)
  (D_true : D_not_me_nor_B) :
  visited_both_sites false false true false :=
by {rw visited_both_sites, sorry}

end person_who_visited_both_sites_l710_710607


namespace complete_square_l710_710678

theorem complete_square (x : ℝ) : 
  (x ^ 2 - 2 * x = 9) -> ((x - 1) ^ 2 = 10) :=
by
  intro h
  rw [← add_zero (x ^ 2 - 2 * x), ← add_zero (10)]
  calc
    x ^ 2 - 2 * x = 9                   : by rw [h]
             ...  = (x ^ 2 - 2 * x + 1 - 1) : by rw [add_sub_cancel, add_zero]
             ...  = (x - 1) ^ 2 - 1     : by 
                           { rw [sub_eq_add_neg], exact add_sub_cancel _ _}
             ...  = 10 - 1              : by rw [h]
             ...  = 10                  : by rw (sub_sub_cancel)
 

end complete_square_l710_710678


namespace eccentricity_of_ellipse_l710_710513

theorem eccentricity_of_ellipse (a : ℝ) (e : ℝ) (P : ℝ × ℝ) (h_ellipse : P = (1, sqrt 6 / 3)) 
    (h_eq : (1 : ℝ) / a^2 + (sqrt 6 / 3)^2 = 1) : e = sqrt 6 / 3 := 
by 
  -- Proof is omitted as per instruction
  sorry

end eccentricity_of_ellipse_l710_710513


namespace roots_in_intervals_l710_710470

theorem roots_in_intervals (k : ℝ) :
  (∀ x ∈ set.Ioo (0 : ℝ) 1, 2 * x^2 - k * x + k - 3 = 0) ∧ 
  (∀ x ∈ set.Ioo (1 : ℝ) 2, 2 * x^2 - k * x + k - 3 = 0) ↔
  3 < k ∧ k < 5 :=
by
  sorry

end roots_in_intervals_l710_710470


namespace b_seq_arithmetic_a_seq_formula_l710_710149

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710149


namespace problem_conditions_l710_710090

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710090


namespace solution_proof_l710_710229

variable (x y z : ℝ)

-- Given system of equations
def equation1 := 6 / (3 * x + 4 * y) + 4 / (5 * x - 4 * z) = 7 / 12
def equation2 := 9 / (4 * y + 3 * z) - 4 / (3 * x + 4 * y) = 1 / 3
def equation3 := 2 / (5 * x - 4 * z) + 6 / (4 * y + 3 * z) = 1 / 2

theorem solution_proof : 
  equation1 4 3 2 ∧ equation2 4 3 2 ∧ equation3 4 3 2 := by
  sorry

end solution_proof_l710_710229


namespace number_of_sophomores_in_sample_l710_710717

theorem number_of_sophomores_in_sample :
  let total_students := 400 + 320 + 280 in
  let sophomores := 320 in
  let sample_size := 200 in
  sophomores * sample_size / total_students = 64 := 
by
  let total_students := 400 + 320 + 280
  let proportion_sophomores := 320 / total_students
  have : proportion_sophomores = 0.32 := rfl
  have : sample_size * proportion_sophomores = 64 := rfl
  exact rfl

end number_of_sophomores_in_sample_l710_710717


namespace binom_sum_eq_l710_710799

theorem binom_sum_eq :
  ∃ (j₁ j₂ : ℕ), (binomial 20 3 + binomial 20 4 = binomial 21 j₁) ∧ 
                (binomial 20 3 + binomial 20 4 = binomial 21 j₂) ∧ 
                (j₁ ≠ j₂) ∧ 
                (j₁ + j₂ = 21) := 
by 
  sorry

end binom_sum_eq_l710_710799


namespace fraction_value_l710_710644

theorem fraction_value : 
  (1 + 2 + 3 + 4 + 5) / (2 + 4 + 6 + 8 + 10) = 1 / 2 := 
by skip_proof sorry

end fraction_value_l710_710644


namespace find_number_l710_710700

theorem find_number (N : ℕ) (h1 : ∃ k : ℤ, N = 13 * k + 11) (h2 : ∃ m : ℤ, N = 17 * m + 9) : N = 89 := 
sorry

end find_number_l710_710700


namespace slopes_sum_l710_710609

def is_isosceles_trapezoid (P Q R S : ℤ × ℤ) : Prop :=
  ¬ (P.1 = Q.1 ∨ P.2 = Q.2 ∨ Q.1 = R.1 ∨ Q.2 = R.2 ∨ R.1 = S.1 ∨ R.2 = S.2) ∧
  (let PQ_slope := (Q.2 - P.2) / (Q.1 - P.1) in
   let RS_slope := (S.2 - R.2) / (S.1 - R.1) in
   PQ_slope = RS_slope)

def translation (P S : ℤ × ℤ) : ℤ × ℤ :=
  (S.1 - P.1, S.2 - P.2)

def slopes_of_parallel_sides (P Q R S : ℤ × ℤ) : ℤ :=
  if h : is_isosceles_trapezoid P Q R S then
    let PQ_slope := (Q.2 - P.2) / (Q.1 - P.1) in
    let PS := translation P S in
    let solutions := [ (1, 8), (-1, 8), (1, -8), (-1, -8), (8, 1), (-8, 1), (8, -1), (-8, -1),
                       (4, 7), (-4, 7), (4, -7), (-4, -7), (7, 4), (-7, 4), (7, -4), (-7, -4) ] in
    let possible_slopes := [1, -7/6, 7/6, -1, 11/6, -11/6] in
    (possible_slopes.map Int.abs).sum
  else 0

theorem slopes_sum {
  P Q R S : ℤ × ℤ
  (hP : P = (10, 50))
  (hS : S = (11, 58))
  (hQRS : ∀ Q R S, is_isosceles_trapezoid P Q R S)
}
: slopes_of_parallel_sides P Q R S = 40 := sorry

end slopes_sum_l710_710609


namespace henry_books_l710_710494

theorem henry_books (initial_books packed_boxes each_box room_books coffee_books kitchen_books taken_books : ℕ)
  (h1 : initial_books = 99)
  (h2 : packed_boxes = 3)
  (h3 : each_box = 15)
  (h4 : room_books = 21)
  (h5 : coffee_books = 4)
  (h6 : kitchen_books = 18)
  (h7 : taken_books = 12) :
  initial_books - (packed_boxes * each_box + room_books + coffee_books + kitchen_books) + taken_books = 23 :=
by
  sorry

end henry_books_l710_710494


namespace proof_problem_l710_710518

noncomputable def length_of_AB (B C : Point) : ℝ :=
  let AB := 78.43
  AB

noncomputable def ratio_BC_AB (B C : Point) : ℝ :=
  let ratio := 0.2
  ratio

theorem proof_problem (A B C : Point) 
  (H1 : angle A = 90) 
  (H2 : tan C = 5) 
  (H3 : dist A C = 80) : 
  length_of_AB B C = 78.43 ∧ ratio_BC_AB B C = 0.2 := 
  by
  sorry

end proof_problem_l710_710518


namespace angle_between_a_b_is_pi_over_3_l710_710601

variables (a b c : ℝ^3)

-- Define that the vectors are unit vectors
axiom a_unit : ‖a‖ = 1
axiom b_unit : ‖b‖ = 1
axiom c_unit : ‖c‖ = 1

-- Given the vector relationship
axiom a_eq_b_add_c : a = b + c

-- Prove the angle between a and b is π/3
theorem angle_between_a_b_is_pi_over_3 : real.angle a b = real.pi / 3 :=
by
  sorry

end angle_between_a_b_is_pi_over_3_l710_710601


namespace distance_from_X_to_base_l710_710724

open EuclideanGeometry

variables {A B C P Q S T X : Point}
variables {R : ℝ}

-- Conditions of the problem
def isosceles_triangle (A B C : Point) : Prop := 
∃ (M : Point), midpoint M A C ∧ M = midpoint B A C

def circle_tangent_at_midpoint (A B C : Point) (R : ℝ) : Prop :=
∃ (O : Point), 
  circle O R touches A B C ∧ midpoint O A C

def circle_intersects_sides (A B C P Q S T : Point) : Prop :=
∃ (O : Point),
  circle O R ∧ 
  intersects O A P ∧ intersects O A Q ∧ 
  intersects O B S ∧ intersects O B T

def circumcircles_intersect_at_points (P Q B S T X : Point) : Prop :=
∃ (O₁ O₂ : Point),
  O₁ ≠ O₂ ∧ 
  circumcircle P Q B = O₁ ∧ circumcircle S T B = O₂ ∧
  intersects O₁ O₂ X ∧ intersects O₁ O₂ B

/-- 
  Prove that the distance from point X to the base AC is R.
-/
theorem distance_from_X_to_base 
  (h1 : isosceles_triangle A B C)
  (h2 : circle_tangent_at_midpoint A B C R)
  (h3 : circle_intersects_sides A B C P Q S T)
  (h4 : circumcircles_intersect_at_points P Q B S T X) :
  distance_from_X_to_base X A C = R :=
sorry

end distance_from_X_to_base_l710_710724


namespace area_of_triangle_l710_710920

-- Definitions from the conditions
def li_to_meters (li : ℝ) : ℝ := li * 500
def side1 := 5
def side2 := 12
def hypotenuse := 13

-- The statement translates the mathematical equivalence
theorem area_of_triangle :
  let a := li_to_meters side1
  let b := li_to_meters side2
  let c := li_to_meters hypotenuse
  -- Confirming the triangle is right-angled
  a ^ 2 + b ^ 2 = c ^ 2 ->
  -- Calculating the area
  (1 / 2) * a * b = 7.5 * 10^6 :=  
begin
  -- Pythagorean theorem ensures this is a right-angled triangle
  intros,
  calc
    a ^ 2 + b ^ 2 = c ^ 2 : by sorry
    (1 / 2) * a * b = 7.5 * 10^6 : by sorry,
end

end area_of_triangle_l710_710920


namespace equivalent_slope_condition_l710_710456

variables {α : Type*} [linear_ordered_field α]
variables {A B C : α × α} (Γ : set (α × α))
variables (AB AC BC : (α × α) → (α × α) → bool)
variable (L : (α × α) → (α × α) → bool)

-- We define the slopes of the lines
noncomputable def slope : ((α × α) → (α × α) → α) :=
  sorry -- Definition of slope according to coordinates 

-- Define the conditions
axiom triangle_on_conic (h : Γ) : A ∈ h ∧ B ∈ h ∧ C ∈ h

axiom tangent_line_condition (l : (α × α) → (α × α) → α)
  (hA : A ∈ Γ) : L(A, l(A, A)) -- Tangent line through A

-- Define the equivalence relationship
theorem equivalent_slope_condition :
  (slope A B * slope A C = -1 ↔ slope B C * slope (L A A) = -1) :=
sorry

end equivalent_slope_condition_l710_710456


namespace distance_from_origin_to_line_l710_710856

open Real

theorem distance_from_origin_to_line
  (a b c : ℝ) (h : a + c - 2 * b = 0) :
  real.dist (0, 0) (1, -2) = real.sqrt 5 :=
by 
  sorry

end distance_from_origin_to_line_l710_710856


namespace largest_n_l710_710739

theorem largest_n (a b c n : ℕ) (h_digit_a : 0 ≤ a ∧ a ≤ 9)
  (h_digit_b : 0 ≤ b ∧ b ≤ 9) (h_digit_c : 0 ≤ c ∧ c ≤ 9)
  (h_a_even : a % 2 = 0) 
  (h_ab_div3 : (10 * a + b) % 3 = 0)
  (h_ab_not_div6 : ¬((10 * a + b) % 6 = 0))
  (h_n_div5 : n % 5 = 0)
  (h_n_not_div7 : ¬(n % 7 = 0))
  (h_n_val : n = 100 * a + 10 * b + c) :
  n = 870 := 
begin
  sorry,
end

end largest_n_l710_710739


namespace total_foreign_objects_l710_710747

-- Definitions based on the conditions
def burrs := 12
def ticks := 6 * burrs

-- Theorem to prove the total number of foreign objects
theorem total_foreign_objects : burrs + ticks = 84 :=
by
  sorry -- Proof omitted

end total_foreign_objects_l710_710747


namespace b_seq_arithmetic_a_seq_formula_l710_710162

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710162


namespace part1_arithmetic_sequence_part2_general_formula_l710_710103

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710103


namespace find_x_l710_710933

open Real

-- Define triangle and intersects
structure Triangle (A B C : Type) :=
(AB BC CA : ℝ)

structure Intersects (A1 A2 B1 B2 C1 C2 : Type) :=
(A1_A2 B1_B2 C1_C2 : ℝ)

-- Hypotheses
variables (A B C A1 A2 B1 B2 C1 C2 : Type)
variable (triangle_ABC : Triangle A B C)
variable (intersects : Intersects A1 A2 B1 B2 C1 C2)

noncomputable def problem_statement := 
triangle_ABC.AB = 3 ∧
triangle_ABC.BC = 4 ∧
triangle_ABC.CA = 5 ∧
intersects.A1_A2 = intersects.B1_B2 ∧
intersects.B1_B2 = intersects.C1_C2 ∧
(intersects.A1_A2 = x) ∧
(area_of_hexagon formed_by (A1,A2,B1,B2,C1,C2) = 4) →
x = (11 - sqrt 37) / 3

-- Theorem statement
theorem find_x (A B C A1 A2 B1 B2 C1 C2 : Type) 
(triangle_ABC : Triangle A B C)
(intersects : Intersects A1 A2 B1 B2 C1 C2) 
(x : ℝ) :
problem_statement tA tB tC A1 A2 B1 B2 C1 C2 triangle_ABC intersects x := sorry

end find_x_l710_710933


namespace canonical_equations_parametric_equations_distance_from_point_l710_710261

def line_equations := 
  ∀ x y z : ℝ, (3 * x - 4 * y - z - 1 = 0) ∧ (x + 2 * y + 4 * z - 5 = 0)

theorem canonical_equations (x y z : ℝ) (h : line_equations x y z) :
  (Exists (t : ℝ) (ht : x = 2.2 - 1.4 * t ∧ y = 1.4 - 1.3 * t ∧ z = t), 
  (x - 2.2) / (-1.4) = (y - 1.4) / (-1.3) = z / 1 ) :=
  sorry

theorem parametric_equations (x y z t : ℝ) (h : line_equations x y z) :
  x = 2.2 - 1.4 * t ∧ y = 1.4 - 1.3 * t ∧ z = t :=
  sorry

theorem distance_from_point (A : ℝ × ℝ × ℝ) 
  (line_direction : ℝ × ℝ × ℝ) (line_point : ℝ × ℝ × ℝ) :
  distance A line_direction line_point ≈ 2.915 :=
  sorry

end canonical_equations_parametric_equations_distance_from_point_l710_710261


namespace visit_cities_l710_710593

/-
Select 4 individuals from a group of 6 to visit Paris, London, Sydney, and Moscow,
with the requirement that each city is visited by one person, each individual visits
only one city, and among these 6 individuals, individuals A and B shall not visit Paris.
-/

noncomputable def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem visit_cities (total_individuals : ℕ) (chosen_individuals : ℕ) (cities : ℕ) (A B : ℕ) 
  (not_visit : ℕ) :
  total_individuals = 6 ∧ chosen_individuals = 4 ∧ cities = 4 ∧ not_visit = 1 ∧ A = 1 ∧ B = 2 →
  240 = (perm total_individuals chosen_individuals) - (perm (total_individuals - not_visit) (chosen_individuals - not_visit)) - (perm (total_individuals - not_visit) (chosen_individuals - not_visit)) :=
by
  intro h
  have h1 := perm 6 4
  have h2 := perm 5 3
  have sum := h1 - h2 - h2
  -- The next line ensures the proof compiles correctly, but the full proof isn't necessary.
  assume sorry

end visit_cities_l710_710593


namespace proving_smallest_n_l710_710166

noncomputable def smallest_n (n : ℕ) : Prop :=
  ∀ (x : ℕ → ℝ), (∀ i, x i ≥ 0) → (finset.sum (finset.range n) x = 1) → (finset.sum (finset.range n) (λ i, (x i)^2) ≤ 1/64) → n ≥ 64

theorem proving_smallest_n : ∀ n, smallest_n n :=
by sorry

end proving_smallest_n_l710_710166


namespace complete_square_l710_710677

theorem complete_square (x : ℝ) : 
  (x ^ 2 - 2 * x = 9) -> ((x - 1) ^ 2 = 10) :=
by
  intro h
  rw [← add_zero (x ^ 2 - 2 * x), ← add_zero (10)]
  calc
    x ^ 2 - 2 * x = 9                   : by rw [h]
             ...  = (x ^ 2 - 2 * x + 1 - 1) : by rw [add_sub_cancel, add_zero]
             ...  = (x - 1) ^ 2 - 1     : by 
                           { rw [sub_eq_add_neg], exact add_sub_cancel _ _}
             ...  = 10 - 1              : by rw [h]
             ...  = 10                  : by rw (sub_sub_cancel)
 

end complete_square_l710_710677


namespace b_arithmetic_sequence_a_general_formula_l710_710044

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710044


namespace bisect_LK_AB_l710_710260

noncomputable theory
open_locale classical

variables {Point Line Circle : Type} [MetricSpace Point]
variables (A B C D L K : Point)
variables (l : Line)
variables (circle : Circle)
variables (tangent_l : tangent l circle A)
variables (parallel_l_CD : ∀ {x : Point}, x ∈ l → ∃ {y : Point}, y ∈ CD ∧ dist x y = 0)
variables (B_on_l : B ∈ l)
variables (C_on_circle : C ∈ circle)
variables (D_on_circle : D ∈ circle)
variables (L_on_circle : L ∈ circle)
variables (K_on_circle : K ∈ circle)
variables (CB : Line)
variables (DB : Line)

-- Proving that the line LK bisects segment AB
theorem bisect_LK_AB :
  let LK := Line.mk L K,
      M  := LK.intersect l,
      N  := LK.intersect (Line.mk C D)
  in midpoint AB M :=
sorry

end bisect_LK_AB_l710_710260


namespace maria_savings_percentage_l710_710913

theorem maria_savings_percentage :
  let regular_price := 30
  let discount_prices := [30, 30 * 1/2, 30 * (1 - 0.3), 30 * (1 - 0.3)]
  let total_regular_price := 4 * regular_price
  let total_discounted_price := discount_prices.sum
  let savings := total_regular_price - total_discounted_price
  percentage_savings := (savings / total_regular_price) * 100
  percentage_savings = 27.5 :=
by
  sorry

end maria_savings_percentage_l710_710913


namespace solve_for_a_l710_710883

theorem solve_for_a (x a : ℝ) (h : 3 * x + 2 * a = 3) (hx : x = 5) : a = -6 :=
by
  sorry

end solve_for_a_l710_710883


namespace lathe_B_more_stable_l710_710659

noncomputable def mean (l : List ℝ) : ℝ := l.sum / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
mean (l.map (λ x => (x - mean l) ^ 2))

def data_set_A := [99, 100, 98, 100, 100, 103]
def data_set_B := [99, 100, 102, 99, 100, 100]

theorem lathe_B_more_stable :
  let μA := mean data_set_A
  let μB := mean data_set_B
  let σA := variance data_set_A
  let σB := variance data_set_B
  μA = 100 ∧ μB = 100 ∧ σA = 7 / 3 ∧ σB = 1 → σA > σB :=
by
  intro h
  rcases h with ⟨hμA, hμB, hσA, hσB⟩
  rw [hσA, hσB]
  exact div_pos (by norm_num) (by norm_num) 


end lathe_B_more_stable_l710_710659


namespace impossibility_vertical_99_l710_710608

-- Definitions for the problem
def is_horizontally_divisible_by_11 (table : Fin 100 → Fin 100 → ℕ) : Prop :=
  ∀ i : Fin 100, (Finset.range 100).sum (λ j, table i j * (-1) ^ j) % 11 = 0

def is_vertically_divisible_by_11 (table : Fin 100 → Fin 100 → ℕ) : Fin 100 → Prop :=
  λ j, (Finset.range 100).sum (λ i, table i j * (-1) ^ i) % 11 = 0

theorem impossibility_vertical_99
  (table : Fin 100 → Fin 100 → ℕ)
  (horiz_wrap : is_horizontally_divisible_by_11 table) :
  ¬ (∃ (S : Finset (Fin 100)), S.card = 99 ∧ ∀ j ∈ S, is_vertically_divisible_by_11 table j) :=
sorry

end impossibility_vertical_99_l710_710608


namespace cyclist_round_trip_time_l710_710200

-- Define the conditions of the problem
def distance1 := 12  -- first part of the trip in miles
def speed1 := 8      -- speed for the first part of the trip in mph
def distance2 := 24  -- second part of the trip in miles
def speed2 := 12     -- speed for the second part of the trip in mph
def return_speed := 9  -- speed for the return trip in mph

noncomputable def time1 := distance1 / speed1  -- time for the first part
noncomputable def time2 := distance2 / speed2  -- time for the second part
noncomputable def time_to_reach := time1 + time2  -- time to reach the destination

noncomputable def return_distance := distance1 + distance2  -- return trip distance
noncomputable def return_time := return_distance / return_speed -- time for the return trip

noncomputable def total_time := time_to_reach + return_time  -- total round trip time

-- The theorem stating the answer to the problem
theorem cyclist_round_trip_time : total_time = 7.5 := by
  sorry

end cyclist_round_trip_time_l710_710200


namespace fourth_root_squared_eq_81_l710_710891

theorem fourth_root_squared_eq_81 (y : ℝ) (h : (real.sqrt (real.sqrt y))^2 = 81) : y = 81 :=
sorry

end fourth_root_squared_eq_81_l710_710891


namespace average_speed_l710_710576

theorem average_speed (v : ℝ) (h1 : 0 < v) 
  (h2 : ∀ day : ℕ, (day ∈ [1, 2] → Louisa traveled day = 375 ∨ Louisa traveled day = 525)) 
  (h3 : (525 / v) = (375 / v) + 4) : v = 37.5 :=
by {
  -- Assume Louisa traveled same speed
  sorry
}

end average_speed_l710_710576


namespace quadratic_sequence_exists_l710_710745

theorem quadratic_sequence_exists (b c : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), 
  a 0 = b ∧ 
  a n = c ∧ 
  ∀ i, 1 ≤ i → i ≤ n → |a i - a (i - 1)| = i^2 :=
sorry

end quadratic_sequence_exists_l710_710745


namespace complete_square_transform_l710_710679

theorem complete_square_transform (x : ℝ) : 
  x^2 - 2 * x = 9 ↔ (x - 1)^2 = 10 :=
by
  sorry

end complete_square_transform_l710_710679


namespace count_distinct_product_divisors_l710_710559

-- Define the properties of 8000 and its divisors
def isDivisor (n d : ℕ) := d > 0 ∧ n % d = 0

def T := {d : ℕ | isDivisor 8000 d}

-- The main statement to prove
theorem count_distinct_product_divisors : 
    (∃ n : ℕ, n ∈ { m | ∃ a b : ℕ, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ m = a * b } ∧ n = 88) :=
by {
  sorry
}

end count_distinct_product_divisors_l710_710559


namespace simplify_sqrt_7_pow_6_l710_710596

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l710_710596


namespace bn_is_arithmetic_an_general_formula_l710_710110

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710110


namespace orchid_initial_count_l710_710649

-- Definitions for conditions
def bushes_today : Nat := 37
def bushes_tomorrow : Nat := 25
def workers : Nat := 35
def final_num_orchids : Nat := 109

-- Proof problem statement
theorem orchid_initial_count : 
  let total_bushes_planted := bushes_today + bushes_tomorrow in
  let initial_num_orchids := final_num_orchids - total_bushes_planted in
  initial_num_orchids = 47 := 
by
  sorry

end orchid_initial_count_l710_710649


namespace C_moves_on_circle_l710_710345

-- Definitions of conditions
variable (P Q : Type) [metric_space P] [add_comm_group P] [module ℂ P]
variable (A B C : ℂ)
variable (ω : ℂ)
variable (t : ℝ)

-- C moves continuously such that AB = BC = CA
variable (equal_speed : A t = P + A (t - 1) * complex.exp (complex.I * ω * t) ∧
                       B t = Q + B (t - 1) * complex.exp (complex.I * ω * t))

-- Prove C moves along a circular path
theorem C_moves_on_circle (equal_speed : AB = BC ∧ BC = CA) : 
  ∃ r : ℝ, ∀ t : ℝ, ∃ α : ℝ, C t = r * (α + t) :=
sorry

end C_moves_on_circle_l710_710345


namespace tan_A_equals_one_l710_710930

theorem tan_A_equals_one (A : ℝ) (h : (√3 * Real.cos A + Real.sin A) / (√3 * Real.sin A - Real.cos A) = Real.tan (-7 / 12 * Real.pi)) :
  Real.tan A = 1 :=
by
  sorry

end tan_A_equals_one_l710_710930


namespace sheela_overall_total_income_l710_710215

def monthly_income_in_rs (income: ℝ) (savings: ℝ) (percent: ℝ): Prop :=
  savings = percent * income

def overall_total_income_in_rs (monthly_income: ℝ) 
                              (savings_deposit: ℝ) (fd_deposit: ℝ) 
                              (savings_interest_rate_monthly: ℝ) 
                              (fd_interest_rate_annual: ℝ): ℝ :=
  let annual_income := monthly_income * 12
  let savings_interest := savings_deposit * (savings_interest_rate_monthly * 12)
  let fd_interest := fd_deposit * fd_interest_rate_annual
  annual_income + savings_interest + fd_interest

theorem sheela_overall_total_income:
  ∀ (monthly_income: ℝ)
    (savings_deposit: ℝ) (fd_deposit: ℝ)
    (savings_interest_rate_monthly: ℝ) (fd_interest_rate_annual: ℝ),
    (monthly_income_in_rs monthly_income savings_deposit 0.28)  →
    monthly_income = 16071.43 →
    savings_deposit = 4500 →
    fd_deposit = 3000 →
    savings_interest_rate_monthly = 0.02 →
    fd_interest_rate_annual = 0.06 →
    overall_total_income_in_rs monthly_income savings_deposit fd_deposit
                           savings_interest_rate_monthly fd_interest_rate_annual
    = 194117.16 := 
by
  intros
  sorry

end sheela_overall_total_income_l710_710215


namespace geometric_shape_is_sphere_l710_710802

-- Define the spherical coordinate system conditions
def spherical_coordinates (ρ θ φ r : ℝ) : Prop :=
  ρ = r

-- The theorem we want to prove
theorem geometric_shape_is_sphere (ρ θ φ r : ℝ) (h : spherical_coordinates ρ θ φ r) : ∀ (x y z : ℝ), (x^2 + y^2 + z^2 = r^2) :=
by
  sorry

end geometric_shape_is_sphere_l710_710802


namespace prob_event_a_and_b_l710_710334

-- Definitions based on the problem's conditions
def event_a (n : ℕ) : Prop := n % 10 = 0 ∧ 10 ≤ n ∧ n ≤ 99
def event_b (n : ℕ) : Prop := n % 5 = 0 ∧ 10 ≤ n ∧ n ≤ 99

-- Our main statement
theorem prob_event_a_and_b : 
  let favorable := {n | event_a n} in
  let total := (99 - 10 + 1) in
  (favorable.card.toReal / total.toReal) = 0.1 := 
by 
  sorry

end prob_event_a_and_b_l710_710334


namespace b_arithmetic_sequence_a_general_formula_l710_710047

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710047


namespace shirt_selling_price_l710_710716

theorem shirt_selling_price (x : ℝ)
  (cost_price : x = 80)
  (initial_shirts_sold : ∃ s : ℕ, s = 30)
  (profit_per_shirt : ∃ p : ℝ, p = 50)
  (additional_shirts_per_dollar_decrease : ∃ a : ℕ, a = 2)
  (target_daily_profit : ∃ t : ℝ, t = 2000) :
  (x = 105 ∨ x = 120) := 
sorry

end shirt_selling_price_l710_710716


namespace find_a_2002_l710_710950

noncomputable def a : ℕ → ℝ
| 0 => 2
| (n + 1) => (√3 * a n + 1) / (√3 - a n)

theorem find_a_2002 : a 2002 = 2 + 4 * √3 :=
by
  sorry

end find_a_2002_l710_710950


namespace parallel_perpendicular_implies_l710_710231

variables {Line : Type} {Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions
axiom distinct_lines : m ≠ n
axiom distinct_planes : α ≠ β

-- Parallel and Perpendicular relationships
axiom parallel : Line → Plane → Prop
axiom perpendicular : Line → Plane → Prop

-- Given conditions
axiom parallel_mn : parallel m n
axiom perpendicular_mα : perpendicular m α

-- Proof statement
theorem parallel_perpendicular_implies (h1 : parallel m n) (h2 : perpendicular m α) : perpendicular n α :=
sorry

end parallel_perpendicular_implies_l710_710231


namespace slope_angle_of_line_l710_710638

theorem slope_angle_of_line (α : ℝ) (h : 0 ≤ α ∧ α < 180) : 
  tan α = sqrt 3 → α = 60 :=
by
  sorry

end slope_angle_of_line_l710_710638


namespace smallest_munificence_of_monic_cubic_is_zero_l710_710405

noncomputable def monic_cubic_munificence_min (a b c : ℝ) : ℝ :=
  let p : ℝ → ℝ := λ x, x^3 + a * x^2 + b * x + c
  in max (|p (-1)|) (max (|p 0|) (|p 1|))

theorem smallest_munificence_of_monic_cubic_is_zero :
  ∃ (a b c : ℝ), monic_cubic_munificence_min a b c = 0 :=
sorry

end smallest_munificence_of_monic_cubic_is_zero_l710_710405


namespace circle_tangent_distance_l710_710815

theorem circle_tangent_distance (r : ℝ) (total_tangent_length : ℝ) (h_r : r = 11) (h_total_tangent_length : total_tangent_length = 120) :
  ∃ d : ℝ, d = 61 :=
by
  have tangent_length := total_tangent_length / 2
  have h_tangent_length : tangent_length = 60 := by
    rw [h_total_tangent_length, div_eq_mul_inv, mul_one, inv_of_eq_inv, inv_eq_inv_of_eq]
    -- specific calculations
  have radius_square : ℝ := r ^ 2
  have h_radius_square : radius_square = 121 := by
    rw [h_r, pow_two, mul_eq_mul_of_eq]
    -- specific calculations
  have tangent_square : ℝ := tangent_length ^ 2
  have h_tangent_square : tangent_square = 3600 := by
    rw [h_tangent_length, pow_two, mul_eq_mul_of_eq]
    -- specific calculations
  have distance_square : ℝ := radius_square + tangent_square
  have h_distance_square : distance_square = 3721 := by
    -- specific calculations
  have distance := sqrt distance_square
  have h_distance : distance = 61 := by
    rw [sqrt_eq_iff_sq_eq, eq_comm]
    -- specific calculations
  use distance
  exact h_distance
  -- ending by using already established distance

end circle_tangent_distance_l710_710815


namespace solve_X_Y_Z_sum_l710_710277

-- Definitions of the initial conditions and the row sum constraint
def initial_grid (X Y Z : ℕ) : Prop :=
  (1 + X + 3 = 9) ∧ (2 + Y + Z = 9)

-- The theorem we want to prove
theorem solve_X_Y_Z_sum (X Y Z : ℕ) (h : initial_grid X Y Z) : X + Y + Z = 12 :=
by
  -- include the constraints from the problem
  cases h with h1 h2
  sorry

end solve_X_Y_Z_sum_l710_710277


namespace main_inequality_l710_710942

noncomputable def b (c : ℝ) : ℝ := (1 + c) / (2 + c)

def f (c : ℝ) (x : ℝ) : ℝ := sorry

lemma f_continuous (c : ℝ) (h_c : 0 < c) : Continuous (f c) := sorry

lemma condition1 (c : ℝ) (h_c : 0 < c) (x : ℝ) (h_x : 0 ≤ x ∧ x ≤ 1/2) : 
  b c * f c (2 * x) = f c x := sorry

lemma condition2 (c : ℝ) (h_c : 0 < c) (x : ℝ) (h_x : 1/2 ≤ x ∧ x ≤ 1) : 
  f c x = b c + (1 - b c) * f c (2 * x - 1) := sorry

theorem main_inequality (c : ℝ) (h_c : 0 < c) : 
  ∀ x : ℝ, (0 < x ∧ x < 1) → (0 < f c x - x ∧ f c x - x < c) := sorry

end main_inequality_l710_710942


namespace sin_eq_cos_suff_but_not_nec_l710_710286

theorem sin_eq_cos_suff_but_not_nec (α : ℝ) : (cos (2 * α) = 0) → (sin α = cos α) :=
by
  sorry

end sin_eq_cos_suff_but_not_nec_l710_710286


namespace range_of_a_l710_710776

-- Define the operation * on reals
def custom_mul (x y : ℝ) : ℝ := x * (1 - y)

-- Define the condition for x * (x - a) > 0 to be a subset of the interval [-1, 1]
def in_interval (I : set ℝ) (a : ℝ) : Prop :=
  ∀ x, x * (x - a) > 0 → -1 ≤ x ∧ x ≤ 1

-- The theorem to be proved
theorem range_of_a :
  { a : ℝ | in_interval { x : ℝ | -1 ≤ x ∧ x ≤ 1 } a } = Icc (-2) 0 :=
sorry

end range_of_a_l710_710776


namespace part1_arithmetic_sequence_part2_general_formula_l710_710074

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710074


namespace value_of_fraction_l710_710505

open Real

theorem value_of_fraction (a : ℝ) (h : a^2 + a - 1 = 0) : (1 - a) / a + a / (1 + a) = 1 := 
by { sorry }

end value_of_fraction_l710_710505


namespace bn_is_arithmetic_an_general_formula_l710_710117

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710117


namespace tile_square_with_dominant_rectangles_l710_710810

def is_dominant_rectangle (r : ℝ × ℝ) : Prop :=
  ∃ x, x > 0 ∧ (r = (2 * x, x) ∨ r = (x, 2 * x))

theorem tile_square_with_dominant_rectangles (n : ℕ) (h : n ≥ 5) :
  ∃ rects : list (ℝ × ℝ), rects.length = n ∧ ∀ r ∈ rects, is_dominant_rectangle r ∧ /* a proof that they tile a square */
:= sorry

end tile_square_with_dominant_rectangles_l710_710810


namespace concrete_pillars_correct_l710_710555

-- Definitions based on conditions
def concrete_for_roadway := 1600
def concrete_for_one_anchor := 700
def total_concrete_for_bridge := 4800

-- Total concrete for both anchors
def concrete_for_both_anchors := 2 * concrete_for_one_anchor

-- Total concrete needed for the roadway and anchors
def concrete_for_roadway_and_anchors := concrete_for_roadway + concrete_for_both_anchors

-- Concrete needed for the supporting pillars
def concrete_for_pillars := total_concrete_for_bridge - concrete_for_roadway_and_anchors

-- Proof problem statement, verify that the concrete for the supporting pillars is 1800 tons
theorem concrete_pillars_correct : concrete_for_pillars = 1800 := by
  sorry

end concrete_pillars_correct_l710_710555


namespace other_factor_is_13_l710_710721

theorem other_factor_is_13 :
  ∀ w : ℕ, (w > 0) → (1452 * w = 2^4 * 3^3 * k) → (w = 468) → ∃ k, k = 13^1 :=
by
  intro w w_pos h_product h_min_w
  use 13
  exact h_min_w
  sorry

end other_factor_is_13_l710_710721


namespace pork_transportation_costs_l710_710221

theorem pork_transportation_costs :
  (∀ x : ℕ, 5 ≤ x ∧ x ≤ 9 → -800 * x + 17200 ≥ 10000 ∧ -800 * x + 17200 ≤ 13200) :=
by 
  assume x hx,
  have h_range : 5 ≤ x ∧ x ≤ 9 := hx,
  have h_min : -800 * x + 17200 ≥ 10000,
  { sorry }, -- Prove the minimum cost
  have h_max : -800 * x + 17200 ≤ 13200,
  { sorry }, -- Prove the maximum cost
  exact ⟨h_min, h_max⟩

end pork_transportation_costs_l710_710221


namespace find_the_courtyard_width_l710_710880

-- Define the problem conditions
def paving_stone_length : ℝ := 2
def paving_stone_width : ℝ := 2
def number_of_paving_stones : ℕ := 135
def courtyard_length : ℝ := 30

-- Define the area calculations
def area_of_one_paving_stone : ℝ := paving_stone_length * paving_stone_width
def total_area_covered_by_stones : ℝ := number_of_paving_stones * area_of_one_paving_stone

-- Define the unknown width of the courtyard
def courtyard_width : ℝ :=  total_area_covered_by_stones / courtyard_length

-- The theorem to prove
theorem find_the_courtyard_width : 
  courtyard_width = 18 :=
by
  -- Proof placeholder
  sorry

end find_the_courtyard_width_l710_710880


namespace area_of_30_60_90_triangle_l710_710520

theorem area_of_30_60_90_triangle (hypotenuse : ℝ) (h : hypotenuse = 9) : 
  ∃ A : ℝ, A = (10.125 * Real.sqrt 3) ∧ 
  (∃ a b : ℝ, a = hypotenuse / 2 ∧ b = a * Real.sqrt 3 ∧ A = 0.5 * a * b) :=
begin
  sorry
end

end area_of_30_60_90_triangle_l710_710520


namespace correct_fraction_simplification_l710_710743
-- Ensure the necessary library is imported.

-- Define the main theorem.
theorem correct_fraction_simplification (a b c : ℝ) :
  \(\frac{a - b}{2ab - b^2 - a^2} = \frac{1}{b - a}\) :=
by
  sorry

end correct_fraction_simplification_l710_710743


namespace b_arithmetic_a_general_formula_l710_710021

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710021


namespace problem_conditions_l710_710083

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710083


namespace median_salary_is_28000_l710_710741

noncomputable def num_employees : ℕ := 75

noncomputable def salaries : List ℕ := 
  list.repeat 18000 16 ++ list.repeat 28000 33 ++ list.repeat 55000 8 ++
  list.repeat 80000 12 ++ list.repeat 95000 5 ++ list.repeat 135000 1

def median_salary (l : List ℕ) : ℕ :=
  let sorted_l := list.sort (≤) l in
  let n := list.length sorted_l in
  sorted_l[(n / 2)] -- Since our list length is odd, this directly gives us the median.

theorem median_salary_is_28000 : median_salary salaries = 28000 :=
by
  sorry

end median_salary_is_28000_l710_710741


namespace attendance_percentage_near_tenth_l710_710521

theorem attendance_percentage_near_tenth (skilled_attendance : ℕ) (semi_skilled_attendance : ℕ) (unskilled_attendance : ℕ)
    (skilled_total : ℕ) (semi_skilled_total : ℕ) (unskilled_total : ℕ) :
    skilled_attendance = 94 → semi_skilled_attendance = 53 → unskilled_attendance = 45 →
    skilled_total = 100 → semi_skilled_total = 60 → unskilled_total = 50 →
    let total_attendance := skilled_attendance + semi_skilled_attendance + unskilled_attendance in
    let total_workers := skilled_total + semi_skilled_total + unskilled_total in
    Float.floor ((total_attendance.toFloat / total_workers.toFloat) * 100 * 10) / 10 = 91.4 :=
by
  intros h1 h2 h3 h4 h5 h6
  let total_attendance := 94 + 53 + 45
  let total_workers := 100 + 60 + 50
  have : total_attendance = 192 := by
    unfold total_attendance
    linarith
  have : total_workers = 210 := by
    unfold total_workers
    linarith
  have attendance_ratio := (192.toFloat / 210.toFloat) * 100
  have attendance_percentage := Float.floor (attendance_ratio * 10) / 10
  show attendance_percentage = 91.4
  sorry

end attendance_percentage_near_tenth_l710_710521


namespace donny_money_left_l710_710411

theorem donny_money_left (initial_amount kite_cost frisbee_cost : ℕ) (h1 : initial_amount = 78) (h2 : kite_cost = 8) (h3 : frisbee_cost = 9) :
  initial_amount - (kite_cost + frisbee_cost) = 61 :=
by
  rw [h1, h2, h3]
  exact Nat.sub_eq_of_eq_add (by norm_num)

end donny_money_left_l710_710411


namespace square_area_of_9_circles_grid_l710_710738

theorem square_area_of_9_circles_grid :
  ∀ (r : ℝ), r = 3 →
  ∃ (a : ℝ), a = (3 * (2 * r) * 3) → a * a = 324  :=
begin
  intros r hr,
  use 18,
  split,
  { calc 18 = 3 * (2 * r) * 3 : by sorry, }, -- Adjust to use exact steps from the conditions
  { sorry } -- To prove a * a = 324
end

end square_area_of_9_circles_grid_l710_710738


namespace bn_is_arithmetic_an_general_formula_l710_710114

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710114


namespace measure_angle_PTV_l710_710171

variable (R S T U V P : Type) 
variable [is_regular_pentagon : is_regular_pentagon R S T U V]
variable [is_equilateral_triangle : is_equilateral_triangle P R S]

theorem measure_angle_PTV (R S T U V P : Type) 
  [is_regular_pentagon : is_regular_pentagon R S T U V]
  [is_equilateral_triangle : is_equilateral_triangle P R S] :
  measure (angle P T V) = 6 :=
sorry

end measure_angle_PTV_l710_710171


namespace sequence_bn_arithmetic_and_an_formula_l710_710124

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710124


namespace sector_area_is_two_l710_710463

-- Define the given conditions: radius = 1 and arc length = 4
def radius : ℝ := 1
def arc_length : ℝ := 4

-- Define the formula for the area of a sector
def sector_area (l R : ℝ) : ℝ := (1 / 2) * l * R

-- The theorem to prove: the area of the sector is 2 given the conditions
theorem sector_area_is_two : sector_area arc_length radius = 2 :=
by
  sorry

end sector_area_is_two_l710_710463


namespace relatively_prime_probability_42_l710_710301

theorem relatively_prime_probability_42 : 
  (λ x, (x ≤ 42 ∧ x > 0 ∧ Nat.gcd x 42 = 1)) / (λ x, (x ≤ 42 ∧ x > 0)) = 2/7 :=
by 
  sorry

end relatively_prime_probability_42_l710_710301


namespace minimum_value_l710_710569

theorem minimum_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y = 1 / 2) (h5 : x ≤ y) (h6 : y ≤ z) :
  \(\frac{x + z}{xyz} \) ≥ 48 :=
sorry

end minimum_value_l710_710569


namespace range_of_fx_l710_710892

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f'' (x : ℝ) : ℝ := sorry

theorem range_of_fx (f : ℝ → ℝ) (f'' : ℝ → ℝ) 
  (h1 : ∀ x, f'' x - f x = 2 * x * real.exp x)
  (h2 : f 0 = 1) : 
  set.range (λ x, f'' x / f x) = set.Ioc 1 2 :=
sorry

end range_of_fx_l710_710892


namespace b_arithmetic_a_general_formula_l710_710020

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710020


namespace exists_root_in_interval_l710_710941

open Real

theorem exists_root_in_interval 
  (a b c r s : ℝ) 
  (ha : a ≠ 0) 
  (hr : a * r ^ 2 + b * r + c = 0) 
  (hs : -a * s ^ 2 + b * s + c = 0) : 
  ∃ t : ℝ, r < t ∧ t < s ∧ (a / 2) * t ^ 2 + b * t + c = 0 :=
by
  sorry

end exists_root_in_interval_l710_710941


namespace max_area_quadrilateral_l710_710532

theorem max_area_quadrilateral 
(hO : O = (0, 0)) 
(hradius : ∀ P, dist O P = 3 ↔ (P = O)) 
(hperpendicular : ∀ x y, x ≠ y → ⟪x, y⟫ = 0) 
(hM : M = (1, √5)) 
(hline1 : ∀ A C, A ≠ C → line_through A C = l₁ ↔ (A, C) ∈ circle(O, 3)) 
(hline2 : ∀ B D, B ≠ D → line_through B D = l₂ ↔ (B, D) ∈ circle(O, 3)) :
  max_area_quadrilateral O l₁ l₂ M = 12 := 
by sorry

end max_area_quadrilateral_l710_710532


namespace b_arithmetic_sequence_general_formula_a_l710_710053

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710053


namespace value_of_c_l710_710255

noncomputable def find_c (M : ℝ × ℝ) (l1 l2 : ℝ × ℝ) (f : ℝ × ℝ → ℝ) : ℝ :=
if f M = f l1 + f l2 then f M else 0

theorem value_of_c :
  let M := (4, 6)
  let l1 := (2, 4)
  let l2 := (6, 8)
  let f := λ p : ℝ × ℝ, p.1 + p.2
  in find_c M l1 l2 f = 10 :=
by {
  let M := (4, 6)
  let l1 := (2, 4)
  let l2 := (6, 8)
  let f := λ p : ℝ × ℝ, p.1 + p.2
  have hMidpoint : M = ((l1.1 + l2.1) / 2, (l1.2 + l2.2) / 2),
  { simp [l1, l2] },
  have hc : f M = 10,
  { simp [M, f] },
  show find_c M l1 l2 f = 10,
  { unfold find_c,
    simp [hc, l1, l2, f],
  }
  sorry
}

end value_of_c_l710_710255


namespace factor_x11_minus_x_l710_710672

theorem factor_x11_minus_x (R : Type*) [CommRing R] : 
  ∃ (p q r s : R[X]), x^11 - x = p * q * r * s :=
by
  sorry

end factor_x11_minus_x_l710_710672


namespace number_of_students_l710_710726

theorem number_of_students (n : ℕ) :
  n < 50 ∧ n % 6 = 4 ∧ n % 4 = 2 → n = 10 ∨ n = 22 ∨ n = 34 := by
  -- Proof goes here
  sorry

end number_of_students_l710_710726


namespace concrete_pillars_l710_710547

-- Definitions based on the conditions of the problem
def C_deck : ℕ := 1600
def C_anchor : ℕ := 700
def C_total : ℕ := 4800

-- Theorem to prove the concrete required for supporting pillars
theorem concrete_pillars : C_total - (C_deck + 2 * C_anchor) = 1800 :=
by sorry

end concrete_pillars_l710_710547


namespace b_seq_arithmetic_a_seq_formula_l710_710150

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710150


namespace math_proof_problem_l710_710835

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

-- Condition for a_3
def a_3_eq_5 : Prop := a_n 3 = 5

-- Definition of sum S_n for arithmetic sequence a_n
noncomputable def S_n (n : ℕ) : ℕ := n * (a_n 1 + a_n n) / 2

-- Condition for S_10
def S_10_eq_100 : Prop := S_n 10 = 100

-- General term formula proof statement
def general_term_formula_proof : Prop :=
  ∀ n ∈ ℕ, a_n n = 2 * n - 1

-- Definition of b_n
noncomputable def b_n (n : ℕ) : ℝ := 
  2^(a_n n) + a_n n * (sin (n * Real.pi / 2))^2

-- Sum T_n for sequence b_n
noncomputable def T_n (n : ℕ) : ℝ :=
  if even n then 
    (2 / 3) * 4^n + n^2 / 2 - n / 2 - 2 / 3
  else 
    (2 / 3) * 4^n + n^2 / 2 + n / 2 - 2 / 3

-- Sum of the first n terms of b_n proof statement
def sum_T_n_proof : Prop :=
  ∀ n ∈ ℕ, 
  T_n n = if even n then 
    (2 / 3) * 4^n + n^2 / 2 - n / 2 - 2 / 3
  else 
    (2 / 3) * 4^n + n^2 / 2 + n / 2 - 2 / 3

-- Combining all conditions and proof statements
theorem math_proof_problem:
  a_3_eq_5 ∧ S_10_eq_100 → general_term_formula_proof ∧ sum_T_n_proof :=
by
  sorry

end math_proof_problem_l710_710835


namespace sum_three_smallest_m_l710_710996

theorem sum_three_smallest_m :
  (∃ a m, 
    (a - 2 + a + a + 2) / 3 = 7 
    ∧ m % 4 = 3 
    ∧ m ≠ 5 ∧ m ≠ 7 ∧ m ≠ 9 
    ∧ (5 + 7 + 9 + m) % 4 = 0 
    ∧ m > 0) 
  → 3 + 11 + 15 = 29 :=
sorry

end sum_three_smallest_m_l710_710996


namespace difference_between_mean_and_median_l710_710803

noncomputable def mean (histogram : list (ℕ × ℕ)) : ℝ :=
  let total_students := (histogram.map Prod.snd).sum
  let sum_pages := (histogram.map (λ (p : ℕ × ℕ), ((p.fst + p.fst + 1) / 2.0 * p.snd : ℝ))).sum
  sum_pages / total_students

noncomputable def median (histogram : list (ℕ × ℕ)) : ℝ :=
  let cumulative := histogram.scanl (λ acc p, acc + p.snd) 0
  let total_students := cumulative.getLast!
  let half := total_students / 2
  let (first_half, second_half) := cumulative.zip histogram |>.span (λ p, p.fst < half)
  if first_half.empty then
    let current_class := second_half.head!.snd
    (current_class.fst + current_class.fst + 1) / 2.0
  else
    let current_class := second_half.head!.snd
    if first_half.getLast! = half then
      (current_class.fst + current_class.fst + 1) / 2.0
    else
      let next_class := second_half.tail!.head!.snd
      ((current_class.fst + current_class.fst + 1) / 2.0 + (next_class.fst + next_class.fst + 1) / 2.0) / 2.0

theorem difference_between_mean_and_median :
  let histogram := [(1, 4), (3, 2), (5, 6), (7, 3), (9, 5)]
  (mean histogram - median histogram).abs = 0.3 :=
by
  let histogram := [(1, 4), (3, 2), (5, 6), (7, 3), (9, 5)]
  sorry

end difference_between_mean_and_median_l710_710803


namespace prob_relatively_prime_42_l710_710314

noncomputable def euler_totient (n : ℕ) : ℕ :=
  (List.range n).filter (λ i => Nat.gcd i n = 1).length

theorem prob_relatively_prime_42 : 
  (euler_totient 42 : ℚ) / 42 = 2 / 7 := 
by
  sorry

end prob_relatively_prime_42_l710_710314


namespace solve_for_y_l710_710225

theorem solve_for_y (y : ℚ) (h : 1 / 3 + 1 / y = 7 / 9) : y = 9 / 4 :=
by
  sorry

end solve_for_y_l710_710225


namespace least_k_no_arithmetic_progression_2019_integers_l710_710425

theorem least_k_no_arithmetic_progression_2019_integers :
  ∃ (k : ℕ), (∀ (a r : ℝ), 
    ∀ (f : ℕ → ℝ), 
      (∀ (i : ℕ),  1 ≤ i ∧ i ≤ 2019 → f i = a + (i - 1) * r)
      → k ≥ 71 
      → ∃ (i_set : finset ℕ), 
          i_set.card = k 
          ∧ (∀ i ∈ i_set, ((a + (i - 1) * r) : ℝ).denom = 1))
      ∧ ((a,b,i,r) = ?)
  := sorry

end least_k_no_arithmetic_progression_2019_integers_l710_710425


namespace total_area_of_region_l710_710373

variable (a b c d : ℝ)
variable (ha : a > 0) (hb : b > 0) (hd : d > 0)

theorem total_area_of_region : (a + b) * d + (1 / 2) * Real.pi * c ^ 2 = (a + b) * d + (1 / 2) * Real.pi * c ^ 2 := by
  sorry

end total_area_of_region_l710_710373


namespace measure_angle_bac_l710_710544

-- Define the type representing angle measures in degrees.
def Degrees := ℝ

-- Define points and segments
variables (A B C Y X : Type)

-- Define distances between points
variables (dAX dXY dYB dBC : ℝ)

-- Define an angle measure function
noncomputable def measure_angle (p1 p2 p3 : Type) : Degrees := sorry

-- Define the given conditions
axiom ax_eq_xy : dAX = dXY
axiom xy_eq_yb : dXY = dYB
axiom yb_eq_bc : dYB = dBC
axiom angle_abc_right : measure_angle B A C = 90

-- Main theorem to prove
theorem measure_angle_bac : measure_angle B A C = 22.5 :=
sorry

end measure_angle_bac_l710_710544


namespace find_m_l710_710639

-- Define the polynomial expansion and specify the condition for the coefficients
theorem find_m (m : ℝ) : 
  (let poly := (m + x) * (1 + x)^4 in
   let sum_even_coeffs := m + 6 * m + 4 in -- sum of coefficients for even power terms
   sum_even_coeffs = 24) ↔ m = 2 := 
by {
  sorry
}

end find_m_l710_710639


namespace probability_rel_prime_to_42_l710_710308

theorem probability_rel_prime_to_42 : 
  let n := 42 in
  let prime_factors := [2, 3, 7] in
  let relatively_prime_count := n * (1 - 1/prime_factors[0]) * (1 - 1/prime_factors[1]) * (1 - 1/prime_factors[2]) in
  let total_count := 42 in
  (relatively_prime_count / total_count) = 4 / 7 :=
by
  sorry

end probability_rel_prime_to_42_l710_710308


namespace distance_between_points_l710_710669

theorem distance_between_points :
  let P1 := (-3, 5);
  let P2 := (4, -9);
  let dist := Real.sqrt ((P2.1 - P1.1)^2 + (P2.2 - P1.2)^2);
  dist = Real.sqrt 245 := 
by {
  let P1 := (-3 : ℤ, 5 : ℤ);
  let P2 := (4 : ℤ, -9 : ℤ);
  let dist := Real.sqrt ((P2.1 - P1.1)^2 + (P2.2 - P1.2)^2);
  exact sorry;
}

end distance_between_points_l710_710669


namespace ratio_of_times_gina_chooses_to_her_sister_l710_710816

theorem ratio_of_times_gina_chooses_to_her_sister (sister_shows : ℕ) (minutes_per_show : ℕ) (gina_minutes : ℕ) (ratio : ℕ × ℕ) :
  sister_shows = 24 →
  minutes_per_show = 50 →
  gina_minutes = 900 →
  ratio = (900 / Nat.gcd 900 1200, 1200 / Nat.gcd 900 1200) →
  ratio = (3, 4) :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_times_gina_chooses_to_her_sister_l710_710816


namespace correct_answers_l710_710846

noncomputable def f (x : ℝ) : ℝ := g((x + 1) / 2) + x

/-- Function \( g \) and its derivative \( g' \) are defined. -/
variable (g : ℝ → ℝ) (g' : ℝ → ℝ)

/-- Function \( f \) is even. -/
axiom f_even : ∀ x : ℝ, f x = f (-x)

/-- Function \( g' \) is odd around \( x + 1 \). -/
axiom g'_odd : ∀ x : ℝ, g'(-x + 1) = -g'(x + 1)

/-- For all \( x \), \( f(x) = g((x + 1) / 2) + x \). -/
axiom f_def : ∀ x : ℝ, f x = g ((x + 1) / 2) + x

/-- Correct answers for the conditions provided. -/
theorem correct_answers : 
  f' 1 = 1 ∧
  ¬ (g' (1/2) = 4) ∧
  g' (3/2) = 2 ∧
  g' 2 = 4 :=
sorry

end correct_answers_l710_710846


namespace find_p_q_r_sum_l710_710560

variable {E : Type*} [inner_product_space ℝ E] {a b c : E}
variables (p q r : ℝ)

-- Given conditions
def is_unit_vector (v : E) : Prop := ‖v‖ = 1
def has_magnitude_2 (v : E) : Prop := ‖v‖ = 2
def dot_product_zero (u v : E) : Prop := ⟪u, v⟫ = 0
def dot_product_four (u v : E) : Prop := ⟪u, v⟫ = 4
def given_expression (a b c : E) (p q r : ℝ) : Prop := a = p • (a × b) + q • (b × c) + r • (c × a)
def triple_product (a b c : E) : ℝ := ⟪a, b × c⟫

-- Given conditions as definitions
axiom ha : is_unit_vector a
axiom hb : has_magnitude_2 b
axiom hc : has_magnitude_2 c
axiom h_ab_dot_zero : dot_product_zero a b
axiom h_bc_dot_four : dot_product_four b c
axiom h_given_expr : given_expression a b c p q r
axiom h_triple_product : triple_product a b c = 2

-- The proof problem
theorem find_p_q_r_sum : p + q + r = 1 / 2 := by
  sorry

end find_p_q_r_sum_l710_710560


namespace minimal_T_n_l710_710451

open Classical

noncomputable def a_sequence {α : Type} [LinearOrderedField α] (a1 q : α) : ℕ → α
| 0     := a1
| (n+1) := a_sequence a1 q n * q

def T_sequence {α : Type} [LinearOrderedField α] (a : ℕ → α) : ℕ → α
| 0     := 1
| (n+1) := a (n+1) * T_sequence a n

theorem minimal_T_n (a1 q x : ℝ) (h0 : 0 < a1) (h1 : a1 < 1) (h2 : 1 < q) 
  (h3 : (a_sequence a1 q 50 - 1) * (a_sequence a1 q 49 - 1) < 0) 
  (h4 : ∀ n : ℕ, n > 0 → T_sequence (a_sequence a1 q) 49 ≤ T_sequence (a_sequence a1 q) n) :
  49 = x := by
  sorry

end minimal_T_n_l710_710451


namespace problem_proof_l710_710563

variable (Line Plane : Type)
variable (l m : Line) (α β : Plane)

-- Definitions of the relationships
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Given conditions
variable [twoDifferentLines : l ≠ m]
variable [twoDifferentPlanes : α ≠ β]

-- Proof goal
theorem problem_proof :
  (parallel l m) ∧ (perpendicular l α) → (perpendicular m α) :=
by
  sorry

end problem_proof_l710_710563


namespace toy_store_revenue_fraction_l710_710722

theorem toy_store_revenue_fraction (N D J : ℝ) 
  (h1 : J = N / 3) 
  (h2 : D = 3.75 * (N + J) / 2) : 
  (N / D) = 2 / 5 :=
by sorry

end toy_store_revenue_fraction_l710_710722


namespace area_of_pentagon_l710_710580

-- Define the vertices of the pentagon
structure Vertex where
  x : ℕ
  y : ℕ

-- Provide the vertices of the pentagon
def vertices : List Vertex := [
  ⟨0, 1⟩, 
  ⟨2, 5⟩, 
  ⟨6, 3⟩, 
  ⟨5, 0⟩, 
  ⟨1, 0⟩
]

-- Define the Pick's theorem calculation
theorem area_of_pentagon : 
  ∃ I B : ℕ, 
    I = 13 ∧ B = 10 ∧ 
    (I + B / 2 - 1 = 17) :=
  by
  -- Conditions based on problem statement
  have I : ℕ := 13
  have B : ℕ := 10
  use I, B
  -- Applying Pick's Theorem
  exact ⟨rfl, rfl, by norm_num⟩

end area_of_pentagon_l710_710580


namespace units_digit_calculation_l710_710429

-- Define a function to compute the units digit of a number in base 10
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_calculation :
  units_digit (8 * 18 * 1988 - 8^3) = 0 := by
  sorry

end units_digit_calculation_l710_710429


namespace solve_for_y_l710_710224

theorem solve_for_y (y : ℚ) (h : 1/3 + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end solve_for_y_l710_710224


namespace log_function_fixed_point_l710_710619

theorem log_function_fixed_point (a : ℝ) (h : 1 < a) :
  ∃ P : ℝ × ℝ, P = (2, 1) ∧ ∀ x : ℝ, x = 2 → ∀ y : ℝ, y = 1 → y = log a (2 * x - 3) + 1 := 
by
  sorry

end log_function_fixed_point_l710_710619


namespace b_arithmetic_sequence_a_general_formula_l710_710038

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710038


namespace number_of_valid_subsets_l710_710949

noncomputable theory

open Nat

theorem number_of_valid_subsets (p : ℕ) (hp : Nat.Prime p) (hp2 : Odd p) :
  let U := finset.range (2 * p + 1),
      number_valid_subsets := (finset.card (finset.filter 
        (λ (A : finset ℕ), A.card = p ∧ (A.sum id) % p = 0) (finset.powersetLen p U))) in
  number_valid_subsets = (1 / p : ℚ) * (nat.choose (2 * p) p + 2 * (p - 1)) :=
by sorry

end number_of_valid_subsets_l710_710949


namespace b_arithmetic_sequence_general_formula_a_l710_710052

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710052


namespace fixed_point_of_inverse_l710_710861

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1) + 4

theorem fixed_point_of_inverse (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  f a (5) = 1 :=
by
  unfold f
  sorry

end fixed_point_of_inverse_l710_710861


namespace usual_time_to_catch_bus_l710_710661

variables (S T T' : ℝ)

theorem usual_time_to_catch_bus
  (h1 : T' = (5 / 4) * T)
  (h2 : T' - T = 6) : T = 24 :=
sorry

end usual_time_to_catch_bus_l710_710661


namespace solve_for_y_l710_710223

theorem solve_for_y (y : ℚ) (h : 1/3 + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end solve_for_y_l710_710223


namespace circumradius_of_triangle_APQ_l710_710556

variable {A B C P Q: Type*}
variable [DecidableEq A]
variable [DecidableEq B]
variable [DecidableEq C]
variable [DecidableEq P]
variable [DecidableEq Q]
variable r1 r2 : ℝ

-- Given triangle ABC with angle BAC = 60 degrees, 
-- P is the intersection of angle bisector of ABC with side AC,
-- Q is the intersection of angle bisector of ACB with side AB,
-- r1 is the in-radius of triangle ABC,
-- r2 is the in-radius of triangle APQ

theorem circumradius_of_triangle_APQ (h₁ : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 - a * b = c^2) 
  (h₂ : ∃ (p q : ℝ), 0 < p ∧ 0 < q ∧ (p + q)^2 = 2(p^2 + q^2 - p * q)) :
  circumradius (triangle APQ) = 2 * (r1 - r2) :=
sorry

end circumradius_of_triangle_APQ_l710_710556


namespace surface_integral_of_upper_hemisphere_l710_710762

noncomputable def integral_surface (a : ℝ) : ℝ :=
  ∫∫ (S : Set (ℝ × ℝ × ℝ)), (λ (x y z : ℝ), (x * y^2 + z^2) + (y * z^2 + x^2) + (z * x^2 + y^2))  dS

def upper_hemisphere (a : ℝ) : Set (ℝ × ℝ × ℝ) :=
  { p | let ⟨x, y, z⟩ := p in x^2 + y^2 + z^2 = a^2 ∧ z ≥ 0 }

theorem surface_integral_of_upper_hemisphere (a : ℝ) :
  integral_surface (upper_hemisphere a) = (π * a^4 / 20) * (8 * a + 5) :=
sorry

end surface_integral_of_upper_hemisphere_l710_710762


namespace equal_triangle_areas_l710_710570
-- Import the necessary library to access geometric constructions and theorems

-- We start by stating our problem
theorem equal_triangle_areas (A B C P D E F : Type)
(H1: triangle A B C)
(H2: exterior_point P)
(H3: lies_on_line AP D)
(H4: lies_on_line BP E)
(H5: lies_on_line CP F)
(H6: sides_intersect D B C)
(H7: sides_intersect E C A)
(H8: sides_intersect F A B)
(H9: area_pbd = area_pce)
(H10: area_pce = area_paf)
: area_pbd = area_abc :=
sorry

end equal_triangle_areas_l710_710570


namespace b_arithmetic_sequence_general_formula_a_l710_710060

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710060


namespace necessary_but_not_sufficient_condition_l710_710179

def isNecessaryButNotSufficient (m : ℤ) : Prop :=
  let A := {1, m^2}
  let B := {2, 4}
  (A ∩ B = {4}) ↔ (m = 2 ∨ m = -2)

theorem necessary_but_not_sufficient_condition (m : ℤ) :
  isNecessaryButNotSufficient(m) → (m = 2) := sorry

end necessary_but_not_sufficient_condition_l710_710179


namespace b_arithmetic_sequence_a_general_formula_l710_710045

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710045


namespace ratio_of_segments_l710_710906

theorem ratio_of_segments
  (x y z u v : ℝ)
  (h_triangle : x^2 + y^2 = z^2)
  (h_ratio_legs : 4 * x = 3 * y)
  (h_u : u = x^2 / z)
  (h_v : v = y^2 / z) :
  u / v = 9 / 16 := 
  sorry

end ratio_of_segments_l710_710906


namespace inequality_solution_range_l710_710515

theorem inequality_solution_range (a : ℝ) :
  (∃ x ∈ Icc 1 5, x^2 + a * x - 2 > 0) → a > -23 / 5 := 
by 
  sorry

end inequality_solution_range_l710_710515


namespace concyclic_points_l710_710167

-- Define the geometric entities
variables {A B C O M D E S : Type}
(noncomputable def midpoint (B C : Type) : Type := M)
(noncomputable def reflection (A B C : Type) : Type := D)
(noncomputable def intersection (Γ : Type) (M D : Type) : Type := E)
(noncomputable def circumcenter (A D E : Type) : Type := S)

-- Definitions based on conditions
variable {Γ : Type} -- circumcircle of triangle ABC
variable (circumcenter Γ : Type := O)
variable {triangle_ABC : Type} -- triangle ABC

-- Midpoint M of BC
variable (midpoint_BC : midpoint B C)

-- Reflection D of A over BC
variable (reflection_A : reflection A B C)

-- Intersection E of Γ and MD
variable (intersection_MD : intersection Γ M D)

-- Circumcenter S of triangle ADE
variable (circumcenter_ADE : circumcenter A D E)

-- Prove that the points A, E, M, O, and S lie on the same circle
theorem concyclic_points :
  cyclic {A, E, M, O, S} :=
sorry

end concyclic_points_l710_710167


namespace b_arithmetic_a_general_formula_l710_710014

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710014


namespace solve_eigenvalue_problem_l710_710421

def eigenvalue_problem : Prop :=
  ∃ (k : ℝ), (k = 4.5 ∧ 
    ∃ (v : ℝ × ℝ), 
      v ≠ (0, 0) ∧ 
      let A := matrix ([[3, 6], [4, 2]]) in
      (A * ![v.1, v.2] = k • ![v.1, v.2]))

theorem solve_eigenvalue_problem : eigenvalue_problem :=
  sorry

end solve_eigenvalue_problem_l710_710421


namespace friends_not_going_to_movies_l710_710965

theorem friends_not_going_to_movies (total_friends : ℕ) (can_take_to_movies : ℕ) (h1 : total_friends = 25) (h2 : can_take_to_movies = 6) :
  total_friends - can_take_to_movies = 19 :=
by
  rw [h1, h2]
  norm_num

end friends_not_going_to_movies_l710_710965


namespace correct_statement_b_correct_statement_d_l710_710465

/- Given conditions for the function f(x) and its derivative -/
variables (f : ℝ → ℝ)
variables (f' : ℝ → ℝ)
variables (g : ℝ → ℝ)

axiom f_domain : ∀ x : ℝ, f x ∈ ℝ
axiom f'_domain : ∀ x : ℝ, f' x ∈ ℝ
axiom f_eq : ∀ x : ℝ, f(x) = -f(6 - x)
axiom f'_eq : ∀ x : ℝ, f'(x) = 2 - f'(4 - x)
axiom f'_3 : f'(3) = -1

/- Definition for g(x) -/
def g (x : ℝ) : ℝ := 2 * f(3 - x) - 1

/- Statement B: The function y = f'(2x + 4) - 1 is an odd function. -/
def B_statement (h : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, h x = f'(2 * x + 4) - 1 ∧ ∀ x : ℝ, h (-x) = -h x

/- Statement D: The sum of the first 2023 terms of the sequence {g'(n)} is -4050. -/
def D_statement (g' : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, g'(n.succ) = (2 * λ n, f'(3 - n.succ) - 1) ∧
  (finset.sum finset.range 2023 (λ i, g' i.succ)) = -4050

/- The final theorems to be proved -/
theorem correct_statement_b : B_statement (λ x, f'(2 * x + 4) - 1) :=
sorry -- Proof omitted

theorem correct_statement_d : D_statement (g' : ℕ → ℝ) :=
sorry -- Proof omitted

end correct_statement_b_correct_statement_d_l710_710465


namespace infinite_geometric_series_sum_l710_710567

theorem infinite_geometric_series_sum (p q : ℝ)
  (h : (∑' n : ℕ, p / q ^ (n + 1)) = 5) :
  (∑' n : ℕ, p / (p^2 + q) ^ (n + 1)) = 5 * (q - 1) / (25 * q^2 - 50 * q + 26) :=
sorry

end infinite_geometric_series_sum_l710_710567


namespace completing_the_square_l710_710675

theorem completing_the_square (x : ℝ) : 
  x^2 - 2 * x = 9 → (x - 1)^2 = 10 :=
by
  intro h
  sorry

end completing_the_square_l710_710675


namespace trapezium_in_five_chosen_l710_710232

def nonagon_vertices : list ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def are_consecutive (a b : ℕ) : Prop := (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 1 ∧ b = 9) ∨ (a = 9 ∧ b = 1)

def forms_trapezium (s : finset ℕ) : Prop := 
  ∃ a b c d ∈ s, are_consecutive a b ∧ are_consecutive c d ∧ (a ≠ b) ∧ (c ≠ d) ∧ ((a < c ∧ b < d) ∨ (c < a ∧ d < b))

theorem trapezium_in_five_chosen (S : finset ℕ) (h_size : S.card = 5) (h_sub : S ⊆ (finset.of_list nonagon_vertices)) :
  ∃ T : finset ℕ, T ⊆ S ∧ T.card = 4 ∧ forms_trapezium T :=
sorry

end trapezium_in_five_chosen_l710_710232


namespace no_correct_simultaneously_l710_710364

structure TreeRow where
  tree_at : Fin 10 → ℕ
  no_adjacent_same : ∀ i : Fin 9, tree_at i ≠ tree_at (i + 1)

def sasha_statement (tr : TreeRow) : Prop :=
  ∃ b m n,
    (b + m + n = 10) ∧
    (b > m) ∧ 
    (b > n) ∧
    (∀ i, tr.tree_at i < 3) ∧
    (∃ a, tr.tree_at a = 0 ∧ ∃ b, tr.tree_at b = 1 ∧ ∃ c, tr.tree_at c = 2)

def yasha_statement (tr : TreeRow) : Prop :=
  tr.tree_at 0 = 0 ∧ tr.tree_at 9 = 0 ∧ (∀ i, (i ≠ 0) ∧ (i ≠ 9) → tr.tree_at i ≠ 0)

def lesha_statement (tr : TreeRow) : Prop :=
  tr.no_adjacent_same

theorem no_correct_simultaneously :
  ¬(∃ tr : TreeRow, sasha_statement tr ∧ yasha_statement tr ∧ lesha_statement tr) :=
by
  sorry

end no_correct_simultaneously_l710_710364


namespace num_tickets_bought_l710_710358

-- Defining the cost and discount conditions
def ticket_cost : ℝ := 40
def discount_rate : ℝ := 0.05
def total_paid : ℝ := 476
def base_tickets : ℕ := 10

-- Definition to calculate the cost of the first 10 tickets
def cost_first_10_tickets : ℝ := base_tickets * ticket_cost
-- Definition of the discounted price for tickets exceeding 10
def discounted_ticket_cost : ℝ := ticket_cost * (1 - discount_rate)
-- Definition of the total cost for the tickets exceeding 10
def cost_discounted_tickets (num_tickets_exceeding_10 : ℕ) : ℝ := num_tickets_exceeding_10 * discounted_ticket_cost
-- Total amount spent on the tickets exceeding 10
def amount_spent_on_discounted_tickets : ℝ := total_paid - cost_first_10_tickets

-- Main theorem statement proving the total number of tickets Mr. Benson bought
theorem num_tickets_bought : ∃ x : ℕ, x = base_tickets + (amount_spent_on_discounted_tickets / discounted_ticket_cost) ∧ x = 12 := 
by
  sorry

end num_tickets_bought_l710_710358


namespace geometric_sequence_a5_l710_710998

theorem geometric_sequence_a5
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_ratio : ∀ n, a (n + 1) = 2 * a n)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 := 
sorry

end geometric_sequence_a5_l710_710998


namespace probability_rel_prime_to_42_l710_710305

theorem probability_rel_prime_to_42 : 
  let n := 42 in
  let prime_factors := [2, 3, 7] in
  let relatively_prime_count := n * (1 - 1/prime_factors[0]) * (1 - 1/prime_factors[1]) * (1 - 1/prime_factors[2]) in
  let total_count := 42 in
  (relatively_prime_count / total_count) = 4 / 7 :=
by
  sorry

end probability_rel_prime_to_42_l710_710305


namespace problem_1_problem_2_l710_710487

def E (m : ℝ) : set ℝ := {x | abs (x - 1) ≥ m}
def F : set ℝ := {x | 10 / (x + 6) > 1}

theorem problem_1 (m : ℝ) (h : m = 3) : 
  (E m ∩ F) = set.Icc (-6) (-2) := by
  sorry

theorem problem_2 (m : ℝ) (h : E m ∩ F = ∅) : 
  m ≥ 7 := by
  sorry

end problem_1_problem_2_l710_710487


namespace trigonometric_identity_l710_710502

open Real

theorem trigonometric_identity (α : ℝ) (h : α ∈ Set.Ioo (-π) (-π / 2)) : 
  sqrt ((1 + cos α) / (1 - cos α)) - sqrt ((1 - cos α) / (1 + cos α)) = 2 / tan α := 
by
  sorry

end trigonometric_identity_l710_710502


namespace b_arithmetic_sequence_a_general_formula_l710_710048

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710048


namespace trips_to_fill_tank_to_three_fourths_l710_710280

noncomputable def pi : ℝ := real.pi

def tank_length : ℝ := 300  -- cm
def tank_width : ℝ := 50    -- cm
def tank_height : ℝ := 36   -- cm

def bucket_diameter : ℝ := 30 -- cm
def bucket_radius : ℝ := bucket_diameter / 2
def bucket_height : ℝ := 48  -- cm

def bucket_volume : ℝ := pi * (bucket_radius ^ 2) * bucket_height

def tank_volume : ℝ := tank_length * tank_width * tank_height
def target_volume : ℝ := (3 / 4) * tank_volume

def effective_water_per_trip : ℝ := (9 / 10) * (4 / 5) * bucket_volume

def number_of_trips : ℝ := target_volume / effective_water_per_trip

theorem trips_to_fill_tank_to_three_fourths : ⌈number_of_trips⌉ = 17 := 
by sorry

end trips_to_fill_tank_to_three_fourths_l710_710280


namespace sequence_bn_arithmetic_and_an_formula_l710_710132

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710132


namespace intersection_reciprocal_sum_l710_710530

-- Definitions of curves C1 and C2
def curve_C1_param (t : ℝ) : ℝ × ℝ :=
  (1 + t, sqrt 3 * t)

def curve_C2_polar (θ : ℝ) : ℝ :=
  sqrt (3 / (2 - cos (2 * θ)))

-- Cartesian equations from conditions
def curve_C1_cartesian (x y : ℝ) : Prop :=
  sqrt 3 * x - y - sqrt 3 = 0

def curve_C2_cartesian (x y : ℝ) : Prop :=
  (x^2) / 3 + y^2 = 1

-- The theorem to prove
theorem intersection_reciprocal_sum (A B P : ℝ × ℝ) 
  (hA : curve_C1_cartesian A.1 A.2 ∧ curve_C2_cartesian A.1 A.2)
  (hB : curve_C1_cartesian B.1 B.2 ∧ curve_C2_cartesian B.1 B.2)
  (hP : P = (1, 0))
  : (1 / (dist P A)) + (1 / (dist P B)) = sqrt 21 / 2 := 
by 
  sorry

end intersection_reciprocal_sum_l710_710530


namespace b_arithmetic_sequence_general_formula_a_l710_710051

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710051


namespace bn_is_arithmetic_an_general_formula_l710_710109

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710109


namespace length_of_CE_l710_710918

theorem length_of_CE {A B C D E F : Type*} [euclidean_geometry A B C D E F]
  (h1 : right_angle A B E)
  (h2 : right_angle B C E)
  (h3 : right_angle C D E)
  (h4 : angle A E B = 45)
  (h5 : angle B E C = 45)
  (h6 : angle C E D = 45)
  (h7 : distance A E = 28)
  (h8 : is_square C D E F) :
  distance C E = 28 :=
by
  sorry

end length_of_CE_l710_710918


namespace overlapping_region_area_l710_710660

noncomputable def radius : ℝ := 10
noncomputable def angle : ℝ := π / 4
noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ := 0.5 * r * r * θ
noncomputable def triangle_area (r : ℝ) (θ : ℝ) : ℝ := 0.5 * r * r * real.sin θ
noncomputable def overlapping_area (r : ℝ) (θ : ℝ) : ℝ := 2 * (sector_area r θ - triangle_area r θ)

theorem overlapping_region_area : overlapping_area radius angle = 25 * π - 50 * real.sqrt 2 := 
by sorry

end overlapping_region_area_l710_710660


namespace jessica_journey_total_distance_l710_710546

theorem jessica_journey_total_distance
  (y : ℝ)
  (h1 : y = (y / 4) + 25 + (y / 4)) :
  y = 50 :=
by
  sorry

end jessica_journey_total_distance_l710_710546


namespace find_savings_l710_710695

theorem find_savings (income expenditure : ℕ) (ratio_income_expenditure : ℕ × ℕ) (income_value : income = 40000)
    (ratio_condition : ratio_income_expenditure = (8, 7)) :
    income - expenditure = 5000 :=
by
  sorry

end find_savings_l710_710695


namespace find_f_equals_n_l710_710940

open Function

noncomputable def f : ℕ+ → ℕ+ := fun n => n -- This is a stub definition

theorem find_f_equals_n (f : ℕ+ → ℕ+) (H : ∀ n : ℕ+, (n-1)^(2020 : ℕ) < (∏ l in Finset.range (2020 + 1), f^[l] n) ∧ (∏ l in Finset.range (2020 + 1), f^[l] n) < n^(2020 : ℕ) + n^(2019 : ℕ)) : 
∀ n : ℕ+, f n = n :=
sorry

end find_f_equals_n_l710_710940


namespace slope_range_l710_710453

-- Definitions for the coordinates of points A and B
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-2, 3⟩
def B : Point := ⟨3, 0⟩
def O : Point := ⟨0, 0⟩

-- Definition of slopes k_OA and k_OB
def slope (P Q : Point) : ℝ := if Q.x - P.x = 0 then 0 else (Q.y - P.y) / (Q.x - P.x)
def k_OA := slope O A
def k_OB := slope O B

-- The Lean theorem statement
theorem slope_range (k : ℝ) : 
  k = slope O A ∨ k = slope O B → k ≤ -3 / 2 ∨ k ≥ 0 := by
  sorry

end slope_range_l710_710453


namespace units_digit_proof_l710_710432

def units_digit (n : ℤ) : ℤ := n % 10

theorem units_digit_proof :
  ∀ (a b c : ℤ),
  a = 8 →
  b = 18 →
  c = 1988 →
  units_digit (a * b * c - a^3) = 0 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  -- Proof will go here
  sorry

end units_digit_proof_l710_710432


namespace total_foreign_objects_l710_710748

-- Definitions based on the conditions
def burrs := 12
def ticks := 6 * burrs

-- Theorem to prove the total number of foreign objects
theorem total_foreign_objects : burrs + ticks = 84 :=
by
  sorry -- Proof omitted

end total_foreign_objects_l710_710748


namespace general_formula_maximize_sum_l710_710921

-- Define conditions 
def geometric_sequence (a : ℕ → ℝ) :=
∀ n : ℕ, a n > 0 ∧ ∃ q ∈ Set.Ioo 0 1, ∀ n : ℕ, a (n + 1) = q * a n

def condition_holds (a : ℕ → ℝ) :=
a 1 * a 5 + 2 * a 3 * a 5 + a 2 * a 8 = 25 ∧ (a 3 * a 5) ^ (1 / 2) = 2

-- Prove general formula of the sequence a_n
theorem general_formula (a : ℕ → ℝ) (h_seq : geometric_sequence a) (h_cond : condition_holds a) :
  ∀ n : ℕ, a n = 2 ^ (5 - n) :=
sorry

-- Define b_n and S_n
def b (a : ℕ → ℝ) (n : ℕ) : ℝ := Real.logb 2 (a n)
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n * (9 - n)) / 2

-- Prove value of n that maximizes the given sum
theorem maximize_sum (a : ℕ → ℝ) (h_seq : geometric_sequence a) (h_cond : condition_holds a) :
  ∃ n : ℕ, (∑ i in Finset.range (n + 1), S a (i + 1) / (i + 1)) = (∑ i in Finset.range 9, S a (i + 1) / (i + 1)) →
  n = 8 ∨ n = 9 :=
sorry

end general_formula_maximize_sum_l710_710921


namespace Ravi_Prakash_finish_together_l710_710589

-- Definitions based on conditions
def Ravi_time := 24
def Prakash_time := 40

-- Main theorem statement
theorem Ravi_Prakash_finish_together :
  (1 / Ravi_time + 1 / Prakash_time) = 1 / 15 :=
by
  sorry

end Ravi_Prakash_finish_together_l710_710589


namespace compare_nsquare_pow2_pos_int_l710_710461

-- Proposition that captures the given properties of comparing n^2 and 2^n
theorem compare_nsquare_pow2_pos_int (n : ℕ) (hn : n > 0) : 
  (n = 1 → n^2 < 2^n) ∧
  (n = 2 → n^2 = 2^n) ∧
  (n = 3 → n^2 > 2^n) ∧
  (n = 4 → n^2 = 2^n) ∧
  (n ≥ 5 → n^2 < 2^n) :=
by
  sorry

end compare_nsquare_pow2_pos_int_l710_710461


namespace complex_sum_zero_l710_710275

theorem complex_sum_zero (i : ℂ) (h_i : i = complex.I) : 
  i + i^2 + i^3 + i^4 = 0 := sorry

end complex_sum_zero_l710_710275


namespace midpoint_of_coordinates_l710_710182

theorem midpoint_of_coordinates :
  let lisa := (2, 4)
  let john := (6, -2)
  let midpoint := ( (fst lisa + fst john) / 2, (snd lisa + snd john) / 2 )
  midpoint = (4, 1) :=
by
  let lisa := (2, 4)
  let john := (6, -2)
  let midpoint := ( (fst lisa + fst john) / 2, (snd lisa + snd john) / 2 )
  have h1 : fst lisa + fst john = 2 + 6 := rfl
  have h2 : snd lisa + snd john = 4 + -2 := rfl
  have h3 : (2 + 6) / 2 = 4 := rfl
  have h4 : (4 + -2) / 2 = 1 := rfl
  show ( (fst lisa + fst john) / 2, (snd lisa + snd john) / 2 ) = (4, 1) from
    by rw [h1, h2, h3, h4]
  done

end midpoint_of_coordinates_l710_710182


namespace kerosene_cost_l710_710332

/-- A dozen eggs cost as much as a pound of rice, a half-liter of kerosene costs as much as 8 eggs,
and each pound of rice costs $0.33. Prove that a liter of kerosene costs 44 cents. -/
theorem kerosene_cost :
  let egg_cost := 0.33 / 12
  let rice_cost := 0.33
  let half_liter_kerosene_cost := 8 * egg_cost
  let liter_kerosene_cost := half_liter_kerosene_cost * 2
  liter_kerosene_cost * 100 = 44 := 
by
  sorry

end kerosene_cost_l710_710332


namespace total_photos_l710_710183

-- Define the number of photos Claire has taken
def photos_by_Claire : ℕ := 8

-- Define the number of photos Lisa has taken
def photos_by_Lisa : ℕ := 3 * photos_by_Claire

-- Define the number of photos Robert has taken
def photos_by_Robert : ℕ := photos_by_Claire + 16

-- State the theorem we want to prove
theorem total_photos : photos_by_Lisa + photos_by_Robert = 48 :=
by
  sorry

end total_photos_l710_710183


namespace molly_candles_l710_710963

theorem molly_candles :
  ∀ (previous_candles current_age additional_candles : ℕ),
    previous_candles = 14 →
    current_age = 20 →
    additional_candles = current_age - previous_candles →
    additional_candles = 6 :=
begin
  intros previous_candles current_age additional_candles h1 h2 h3,
  rw [h1, h2] at h3,
  exact h3,
end

end molly_candles_l710_710963


namespace symmetry_about_line_l710_710869

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, Real.log (x - 2) / Real.log a

noncomputable def g (a : ℝ) : ℝ → ℝ :=
  λ x, a^(x - 2)

def point_A : (ℝ × ℝ) := (3, 0)
def point_B : (ℝ × ℝ) := (2, 1)

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def slope (A B : ℝ × ℝ) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

def perpendicular_slope (k : ℝ) : ℝ :=
  -1 / k

def perpendicular_bisector (A B : ℝ × ℝ) : ℝ → ℝ :=
  λ x, let m := midpoint A B
       in 1 * (x - m.1) + m.2

theorem symmetry_about_line (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1)
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : f a A.1 = A.2) (hB : g a B.1 = B.2)
  : A = point_A ∧ B = point_B →
    midpoint A B = (5 / 2, 1 / 2) ∧
    slope A B = -1 ∧
    perpendicular_bisector A B = λ x, x - 2 :=
sorry

end symmetry_about_line_l710_710869


namespace race_total_distance_l710_710903

theorem race_total_distance (D : ℝ) 
  (A_time : D / 20 = D / 25 + 1) 
  (beat_distance : D / 20 * 25 = D + 20) : 
  D = 80 :=
sorry

end race_total_distance_l710_710903


namespace relatively_prime_probability_l710_710310

theorem relatively_prime_probability (n : ℕ) (h : n = 42) :
  let phi := n * (1 - 1 / 2) * (1 - 1 / 3) * (1 - 1 / 7) in
  (phi / n) = 2 / 7 :=
by
  sorry

end relatively_prime_probability_l710_710310


namespace sum_of_other_endpoint_coordinates_l710_710577

theorem sum_of_other_endpoint_coordinates (x y : ℤ)
  (h1 : (6 + x) / 2 = 3)
  (h2 : (-1 + y) / 2 = 6) :
  x + y = 13 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l710_710577


namespace b_arithmetic_a_general_formula_l710_710012

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710012


namespace b_arithmetic_a_general_formula_l710_710015

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710015


namespace orthocentric_tetrahedron_edge_tangent_iff_l710_710584

structure Tetrahedron :=
(V : Type*)
(a b c d e f : V)
(is_orthocentric : Prop)
(has_edge_tangent_sphere : Prop)
(face_equilateral : Prop)
(edges_converging_equal : Prop)

variable (T : Tetrahedron)

noncomputable def edge_tangent_iff_equilateral_edges_converging_equal : Prop :=
T.has_edge_tangent_sphere ↔ (T.face_equilateral ∧ T.edges_converging_equal)

-- Now create the theorem statement
theorem orthocentric_tetrahedron_edge_tangent_iff :
  T.is_orthocentric →
  (∀ a d b e c f p r : ℝ, 
    a + d = b + e ∧ b + e = c + f ∧ a^2 + d^2 = b^2 + e^2 ∧ b^2 + e^2 = c^2 + f^2 ) → 
    edge_tangent_iff_equilateral_edges_converging_equal T := 
by
  intros
  unfold edge_tangent_iff_equilateral_edges_converging_equal
  sorry

end orthocentric_tetrahedron_edge_tangent_iff_l710_710584


namespace find_selling_price_l710_710713

-- Define the basic parameters
def cost := 80
def s0 := 30
def profit0 := 50
def desired_profit := 2000

-- Additional shirts sold per price reduction
def add_shirts (p : ℕ) := 2 * p

-- Number of shirts sold given selling price x
def num_shirts (x : ℕ) := 290 - 2 * x

-- Profit equation
def profit_equation (x : ℕ) := (x - cost) * num_shirts x = desired_profit

theorem find_selling_price (x : ℕ) :
  (x = 105 ∨ x = 120) ↔ profit_equation x := by
  sorry

end find_selling_price_l710_710713


namespace sum_of_grid_is_less_9_l710_710910

open Classical

noncomputable def sum_of_grid_less_9 (grid : Fin 9 → Fin 9 → ℝ)
  (h_abs_lt_1 : ∀ i j, abs (grid i j) < 1)
  (h_sum_2x2_eq_0 : ∀ i j, i < 8 → j < 8 → grid i j + grid i (j + 1) +
                                           grid (i + 1) j + grid (i + 1) (j + 1) = 0) :
  ℝ :=
  ∑ i in Finset.range 9, ∑ j in Finset.range 9, grid i j

theorem sum_of_grid_is_less_9 (grid : Fin 9 → Fin 9 → ℝ)
  (h_abs_lt_1 : ∀ i j, abs (grid i j) < 1)
  (h_sum_2x2_eq_0 : ∀ i j, i < 8 → j < 8 → grid i j + grid i (j + 1) +
                                           grid (i + 1) j + grid (i + 1) (j + 1) = 0) :
  sum_of_grid_less_9 grid h_abs_lt_1 h_sum_2x2_eq_0 < 9 := 
sorry

end sum_of_grid_is_less_9_l710_710910


namespace bn_arithmetic_sequence_an_formula_l710_710033

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710033


namespace central_angle_of_sector_is_one_l710_710852

-- Definitions based on the given conditions
def radius := ℝ -- real number for radius
def circumference (R : radius) := 3 * R
def arc_length (R : radius) := circumference R - 2 * R
def central_angle (R : radius) := arc_length R / R

-- The theorem we need to prove
theorem central_angle_of_sector_is_one (R : radius) (hCircumference : circumference R = 3 * R) :
  central_angle R = 1 :=
sorry

end central_angle_of_sector_is_one_l710_710852


namespace imaginary_part_of_conjugate_z_l710_710460

def i : ℂ := complex.I

theorem imaginary_part_of_conjugate_z (z : ℂ) (h : z * (1 + i) = complex.abs (1 + i)) :
  complex.im (conjugate z) = real.sqrt 2 / 2 :=
by sorry

end imaginary_part_of_conjugate_z_l710_710460


namespace mary_no_torn_cards_l710_710962

theorem mary_no_torn_cards
  (T : ℕ) -- number of Mary's initial torn baseball cards
  (initial_cards : ℕ := 18) -- initial baseball cards
  (fred_cards : ℕ := 26) -- baseball cards given by Fred
  (bought_cards : ℕ := 40) -- baseball cards bought
  (total_cards : ℕ := 84) -- total baseball cards Mary has now
  (h : initial_cards - T + fred_cards + bought_cards = total_cards)
  : T = 0 :=
by sorry

end mary_no_torn_cards_l710_710962


namespace problem_equivalence_l710_710683

section ProblemDefinitions

def odd_function_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def statement_A (f : ℝ → ℝ) : Prop :=
  (∀ x < 0, f x = -Real.log (-x)) →
  odd_function_condition f →
  ∀ x > 0, f x ≠ -Real.log x

def statement_B (a : ℝ) : Prop :=
  Real.logb a (1 / 2) < 1 →
  (0 < a ∧ a < 1 / 2) ∨ (1 < a)

def statement_C : Prop :=
  ∀ x, (Real.logb 2 (Real.sqrt (x-1)) = (1/2) * Real.logb 2 x)

def statement_D (x1 x2 : ℝ) : Prop :=
  (x1 + Real.log x1 = 2) →
  (Real.log (1 - x2) - x2 = 1) →
  x1 + x2 = 1

end ProblemDefinitions

structure MathProofProblem :=
  (A : ∀ f : ℝ → ℝ, statement_A f)
  (B : ∀ a : ℝ, statement_B a)
  (C : statement_C)
  (D : ∀ x1 x2 : ℝ, statement_D x1 x2)

theorem problem_equivalence : MathProofProblem :=
  { A := sorry,
    B := sorry,
    C := sorry,
    D := sorry }

end problem_equivalence_l710_710683


namespace at_least_two_elements_same_label_l710_710947

universe u

variable {A B : Type u}
variable [decidable_eq A] [decidable_eq B]
variable (a b : list ℕ)

def is_two_element_subset (s : list ℕ) : Prop :=
  s.length = 2

def f_inj (f : A ∪ B → ℕ) : Prop :=
  function.injective f

noncomputable def C := { s : list ℕ | is_two_element_subset s ∧ (s ⊆ a ∨ s ⊆ b) }

theorem at_least_two_elements_same_label
  (A B : set ℕ)
  (h : |A ∩ B| = 1)
  (n : ℕ)
  (hn : n ≥ 6)
  (f : A ∪ B → ℕ)
  (hf : f_inj f) :
  ∃ (x y z w : ℕ), {x, y} ∈ C ∧ {z, w} ∈ C ∧ x ≠ z ∧ y ≠ w ∧ |f x - f y| = |f z - f w| :=
begin
  sorry
end

end at_least_two_elements_same_label_l710_710947


namespace arcsin_arccos_eq_l710_710987

theorem arcsin_arccos_eq (x : ℝ) (hx₁ : -1 ≤ x ∧ x ≤ 1) (hx₂ : -1 ≤ x-1 ∧ x-1 ≤ 1) (hx₃ : -1 ≤ 1-x ∧ 1-x ≤ 1) :
  arcsin x + arcsin (x - 1) = arccos (1 - x) → x = 1 :=
by
  sorry

end arcsin_arccos_eq_l710_710987


namespace find_f_neg_one_l710_710849

-- Define the function f
def f (x : ℝ) : ℝ := 
  if h : x > 0 then 
    x^3 - 3 * x^2 
  else 
    0  -- this definition placeholder will be used only for x <= 0 conditions, not evaluated.

-- Define the odd function condition
axiom odd_function (x : ℝ) : f (-x) = -f (x)

-- Now, we need to state the goal to prove
theorem find_f_neg_one (f_odd: ∀ x : ℝ, f (-x) = -f x) : f (-1) = 2 := by
  -- Introduce lemma to use steps from the solution present in part b
  have f1_pos: f (1) = 1^3 - 3 * 1^2 := by
    -- as per the condition from problem
    simp [f]
  calc
    f (-1) = -f (1) := by apply f_odd
        ... = -(-2)  := by simp [f1_pos]
        ... = 2     := by linarith

end find_f_neg_one_l710_710849


namespace integral_f_eq_4_over_3_l710_710557

def f (x : ℝ) : ℝ :=
  if (0 ≤ x) ∧ (x ≤ 1) then x^2
  else if (1 < x) ∧ (x ≤ Real.exp 1) then 1 / x
  else 0

theorem integral_f_eq_4_over_3 :
  ∫ x in 0..Real.exp 1, f x = 4 / 3 :=
by
  sorry

end integral_f_eq_4_over_3_l710_710557


namespace min_8_minus_x_sq_geq_neg1_and_ctg_x_geq_neg1_total_length_l710_710789

noncomputable def interval_length (l : Real) (u : Real) : Real :=
  if l > u then 0 else u - l

theorem min_8_minus_x_sq_geq_neg1_and_ctg_x_geq_neg1_total_length :
  let lengths := [
    interval_length (-3) (-7 * Real.pi / 4),
    interval_length 0 (3 * Real.pi / 4)
  ]
  Real.sum lengths = 3 + Real.pi / 2 :=
by
  sorry

end min_8_minus_x_sq_geq_neg1_and_ctg_x_geq_neg1_total_length_l710_710789


namespace locus_of_point_M_l710_710355

theorem locus_of_point_M 
  {O A B A1 B1 M : Type*} 
  [CommRing O] [CommRing A] [CommRing B] [CommRing A1] [CommRing B1] [CommRing M] 
  (h_circle: is_circle_with_diameter O A B)
  (h_chords: is_chord O A A1 ∧ is_chord O B B1)
  (h_intersection: intersects_at A A1 B B1 M)
  (h_angle_condition: ∠AMB + ∠A1OB1 = 180) : 
  {M | is_on_arcs_of_equilateral_circles M A B} :=
begin
  sorry
end

end locus_of_point_M_l710_710355


namespace sequence_bn_arithmetic_and_an_formula_l710_710121

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710121


namespace value_of_c_l710_710254

noncomputable def find_c (M : ℝ × ℝ) (l1 l2 : ℝ × ℝ) (f : ℝ × ℝ → ℝ) : ℝ :=
if f M = f l1 + f l2 then f M else 0

theorem value_of_c :
  let M := (4, 6)
  let l1 := (2, 4)
  let l2 := (6, 8)
  let f := λ p : ℝ × ℝ, p.1 + p.2
  in find_c M l1 l2 f = 10 :=
by {
  let M := (4, 6)
  let l1 := (2, 4)
  let l2 := (6, 8)
  let f := λ p : ℝ × ℝ, p.1 + p.2
  have hMidpoint : M = ((l1.1 + l2.1) / 2, (l1.2 + l2.2) / 2),
  { simp [l1, l2] },
  have hc : f M = 10,
  { simp [M, f] },
  show find_c M l1 l2 f = 10,
  { unfold find_c,
    simp [hc, l1, l2, f],
  }
  sorry
}

end value_of_c_l710_710254


namespace max_area_quadrilateral_12_l710_710533

theorem max_area_quadrilateral_12 (O : ℝ × ℝ) (d₁ d₂ : ℝ)
  (hO : O = (0, 0))
  (radius_O : ∀ P : ℝ × ℝ, (P.1 - O.1)^2 + (P.2 - O.2)^2 = 9)
  (perpendicular : d₁^2 + d₂^2 = 6)
  (foot_perpendicular : ∃ M : ℝ × ℝ, M = (1, sqrt 5) ∧ (O.1 - M.1)^2 + (O.2 - M.2)^2 = 6)
  (intersect_l1 : ∃ A C : ℝ × ℝ, (A = l₁ ∧ C = l₁) ∧ (A.1 - O.1)^2 + (A.2 - O.2)^2 = 9 ∧ (C.1 - O.1)^2 + (C.2 - O.2)^2 = 9)
  (intersect_l2 : ∃ B D : ℝ × ℝ, (B = l₂ ∧ D = l₂) ∧ (B.1 - O.1)^2 + (B.2 - O.2)^2 = 9 ∧ (D.1 - O.1)^2 + (D.2 - O.2)^2 = 9) :
  ∃ S : ℝ, S = 12 :=
sorry

end max_area_quadrilateral_12_l710_710533


namespace sum_of_alternating_series_l710_710418

theorem sum_of_alternating_series :
  (∑ n in (finset.range 21).map (equiv.add_left (-10)), (-1)^n) = 1 := 
begin
  sorry
end

end sum_of_alternating_series_l710_710418


namespace alex_blueberry_pies_l710_710385

theorem alex_blueberry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ)
  (h_total : total_pies = 30)
  (h_ratio : apple_ratio = 2 ∧ blueberry_ratio = 3 ∧ cherry_ratio = 5) :
  let parts := apple_ratio + blueberry_ratio + cherry_ratio,
      pies_per_part := total_pies / parts,
      blueberry_pies := blueberry_ratio * pies_per_part
  in blueberry_pies = 9 :=
by
  sorry

end alex_blueberry_pies_l710_710385


namespace sum_of_solutions_eq_3pi_div_2_l710_710800

theorem sum_of_solutions_eq_3pi_div_2 : 
  (∑ x in {x | 0 ≤ x ∧ x ≤ 2 * π ∧ (1 / sin x + 1 / cos x = 2)}, x) = 3 * π / 2 :=
by {
  -- proof is skipped
  sorry
}

end sum_of_solutions_eq_3pi_div_2_l710_710800


namespace decagon_division_impossible_l710_710540

-- Define a polygon and the conditions on the triangles used to divide it.
structure Polygon where
  sides : ℕ

-- Define the properties of black and white triangles
def black_triangle_condition (p : Polygon) (n : ℕ) := 
  n % 3 = 0 ∧ n = p.sides + nonBoundarySides n

def white_triangle_condition (m : ℕ) := 
  m % 3 = 0

def nonBoundarySides (n : ℕ): ℕ := sorry

-- Define the impossibility condition for a decagon
theorem decagon_division_impossible :
  ∀ (n m : ℕ),
    let decagon := Polygon.mk 10
    black_triangle_condition decagon n →
    white_triangle_condition m →
    n - m = 10 → False := 
by
  intros n m decagon h_black h_white h_diff
  have h1 : n % 3 = 0 := h_black.1
  have h2 : m % 3 = 0 := h_white
  have h3 : 10 % 3 ≠ 0 := by norm_num
  have h4 : (n - m) % 3 = 0 := by
    rw [← h_diff]
    exact h3
  contradiction

end decagon_division_impossible_l710_710540


namespace b_arithmetic_sequence_general_formula_a_l710_710063

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710063


namespace lcm_is_perfect_square_l710_710195

open Nat

theorem lcm_is_perfect_square (a b : ℕ) : 
  (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0 → ∃ k : ℕ, k^2 = lcm a b :=
by
  sorry

end lcm_is_perfect_square_l710_710195


namespace students_after_third_stop_l710_710276

/-!
  We are given a problem concerning the number of students on a bus after several stops where students get on and off.
  We start with 10 students. At each stop, a certain number of students get off the bus and a certain number get on.
  Our goal is to prove the number of students on the bus after the third stop.
-/

def initial_students : ℕ := 10
def stop1_off : ℕ := 3
def stop1_on : ℕ := 4
def stop2_off : ℕ := 2
def stop2_on : ℕ := 5
def stop3_off : ℕ := 6
def stop3_on : ℕ := 3

theorem students_after_third_stop :
  let s1 := initial_students - stop1_off + stop1_on in
  let s2 := s1 - stop2_off + stop2_on in
  let s3 := s2 - stop3_off + stop3_on in
  s3 = 11 :=
by
  sorry

end students_after_third_stop_l710_710276


namespace perfect_cube_probability_l710_710377

theorem perfect_cube_probability (p q : ℕ) (h : Nat.coprime p q) 
  (p_dice : ℕ := 6) (r_tosses : ℕ := 6) :
  let prob := (p / q : ℚ)
  finset.filter (λ (prod_val : ℕ), is_cube prod_val) (finset.finrange (p_dice^r_tosses + 1)).card = 1 → 
  p + q = 46657 :=
by sorry

end perfect_cube_probability_l710_710377


namespace value_of_expression_l710_710322

theorem value_of_expression (x : ℤ) (h : x = 5) : x^5 - 10 * x = 3075 := by
  sorry

end value_of_expression_l710_710322


namespace chord_circle_identity_l710_710399

theorem chord_circle_identity
  (R : ℝ)
  (A B C D O : point)
  (circle : set point)
  (h1 : is_chord A B R circle)
  (h2 : AB = BC)
  (h3 : is_center O R circle)
  (h4 : segment CO intersects circle at D)
  (h5 : can_inscribe_square_using AB R circle) :
  CD = 4 * R * real.sin (real.pi / 10) :=
sorry

end chord_circle_identity_l710_710399


namespace log_function_fixed_point_l710_710620

theorem log_function_fixed_point (a : ℝ) (h : 1 < a) :
  ∃ P : ℝ × ℝ, P = (2, 1) ∧ ∀ x : ℝ, x = 2 → ∀ y : ℝ, y = 1 → y = log a (2 * x - 3) + 1 := 
by
  sorry

end log_function_fixed_point_l710_710620


namespace necessary_and_sufficient_condition_real_roots_l710_710263

theorem necessary_and_sufficient_condition_real_roots (a : ℝ) :
  ∀ x : ℝ, ∃ t : ℝ, t = 3 ^ (-(| x - 2 |)) ∧ 1 ≥ t ∧ t > 0 → (∃ t : ℝ, 0 < t ∧ t ≤ 1 ∧ t^2 - 4 * t - a = 0) ↔ -3 ≤ a ∧ a < 0 :=
begin
  sorry,
end

end necessary_and_sufficient_condition_real_roots_l710_710263


namespace problem_conditions_l710_710085

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710085


namespace rhombus_overlap_area_l710_710398

theorem rhombus_overlap_area 
(diagonal1_1 diagonal2_1 diagonal1_2 diagonal2_2 : ℝ) 
(h_diag1_1 : diagonal1_1 = 4) 
(h_diag2_1 : diagonal2_1 = 6) 
(h_diag1_2 : diagonal1_2 = 4) 
(h_diag2_2 : diagonal2_2 = 6) 
(rotated : ℝ → ℝ → Prop)
(h_rotated : ∀ x y, rotated x y ↔ x = y - 90 ∨ x = y + 90) :
  let s := sqrt (2^2 + 3^2)
  let ok := sqrt ((2 * s^2 * 17) / 100)
  let kh := ok / sqrt 2
  let tri_aob_area := 1/2 * 2 * 3 
  let tri_afk_area := 1/2 * 1.5 * kh in
  4 * (tri_aob_area - tri_afk_area) = 9.6 := sorry

end rhombus_overlap_area_l710_710398


namespace mean_of_numbers_l710_710262

def mean (s : List Float) : Float :=
  s.sum / s.length

theorem mean_of_numbers :
  mean [1, 22, 23, 24, 25, 26, 27, 2] = 18.75 :=
by
  sorry

end mean_of_numbers_l710_710262


namespace b_seq_arithmetic_a_seq_formula_l710_710153

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710153


namespace sum_y_coordinates_of_other_vertices_of_parallelogram_l710_710500

theorem sum_y_coordinates_of_other_vertices_of_parallelogram :
  let x1 := 4
  let y1 := 26
  let x2 := 12
  let y2 := -8
  let midpoint_y := (y1 + y2) / 2
  2 * midpoint_y = 18 := by
    sorry

end sum_y_coordinates_of_other_vertices_of_parallelogram_l710_710500


namespace length_of_AB_l710_710582

variables {A B P Q : ℝ}
variables (x y : ℝ)

-- Conditions
axiom h1 : A < P ∧ P < Q ∧ Q < B
axiom h2 : P - A = 3 * x
axiom h3 : B - P = 5 * x
axiom h4 : Q - A = 2 * y
axiom h5 : B - Q = 3 * y
axiom h6 : Q - P = 3

-- Theorem statement
theorem length_of_AB : B - A = 120 :=
by
  sorry

end length_of_AB_l710_710582


namespace sum_of_digits_l710_710380

open Nat

theorem sum_of_digits (N : ℕ) (h1 : (N * (N + 1)) / 2 = 2485) : sum (digits 10 70) = 7 :=
by
  -- The proof would go here, but for now, we use sorry to skip the actual proof steps.
  sorry

end sum_of_digits_l710_710380


namespace minimum_MN_when_cuboid_maximized_l710_710539

-- Define the conditions as premises
variables (D C B C1 A M N : Type) 
variables (DC C1C Sum : ℝ)
variable (CB : ℝ)
variable (AM MB : ℝ → ℝ)
variable (C1N : ℝ)
variable (V_cuboid : ℝ)

-- Assume the given premises are true, to be used in the theorem
axiom DC_plus_C1C_eq_8 : DC + C1C = 8
axiom CB_eq_4 : CB = 4
axiom AM_eq_MB : ∀ x, AM x = MB x
axiom C1N_eq_sqrt_5 : C1N = Real.sqrt 5
axiom V_cuboid_maximized : ∀ (x y z : ℝ), (x + y + z = 12) → (x = 4 ∧ y = 4 ∧ z = 4)

-- Define the theorem to be proven
theorem minimum_MN_when_cuboid_maximized : 
  V_cuboid = 4^3 → 
  let M : ℝ := 4 in 
  let N : ℝ := 4 in 
  let MN_min : ℝ := Real.sqrt (16 + 5) in 
  MN_min = Real.sqrt 21 :=
sorry

end minimum_MN_when_cuboid_maximized_l710_710539


namespace compute_expression_l710_710401

theorem compute_expression :
  24 * 42 + 58 * 24 + 12 * 24 = 2688 := by
  sorry

end compute_expression_l710_710401


namespace problem_equivalent_proof_l710_710831

-- Define sequences a_n and b_n with given properties
def a (n : ℕ) : ℚ :=
  if n = 1 then 1/2 else 2 * a (n - 1) + 1/2

def Sn (n : ℕ) : ℚ := (n : ℚ) * (n + 1)

def b (n : ℕ) : ℚ :=
  if n = 1 then Sn 1 else Sn n - Sn (n - 1)

-- Given conditions
def condition_a1 := a 1 = 1/2
def condition_a_recur (n : ℕ) := a (n + 1) = 2 * a n + 1/2

def condition_Sn (n : ℕ) := ∀ n : ℕ, Sn n = (↑n : ℚ) * (n + 1)

-- General formulas for a_n and b_n
def formula_a (n : ℕ) : ℚ :=
  2^(n-1) - 1/2

def formula_b (n : ℕ) : ℚ :=
  2 * (n : ℚ)

-- Sum of the first n terms of c_n
def c (n : ℕ) : ℚ := a n * b n

def sum_c (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, c (k + 1)

def Tn (n : ℕ) : ℚ :=
  (n - 1) * 2^(n + 1) + 2 - (1/2 : ℚ) * n * (n + 1)

-- Proof statement
theorem problem_equivalent_proof (n : ℕ) :
  (∀ m, m < n → a (m + 1) = formula_a (m + 1)) ∧
  (∀ m, m < n → b (m + 1) = formula_b (m + 1)) ∧
  (sum_c n = Tn n) :=
by sorry

end problem_equivalent_proof_l710_710831


namespace b_arithmetic_a_formula_l710_710003

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l710_710003


namespace infinite_rel_prime_pairs_l710_710210

theorem infinite_rel_prime_pairs (x : ℤ) : 
  ∃∞ x, let a := 2 * x + 1, 
             b := 2 * x - 1 in 
             Nat.gcd a b = 1 
             ∧ a > 1 
             ∧ b > 1 
             ∧ (a^b + b^a) % (a + b) = 0 := 
by
  sorry

end infinite_rel_prime_pairs_l710_710210


namespace final_integer_after_steps_l710_710991

theorem final_integer_after_steps :
  let initial := (2^10) * (5^10)
  let after_steps := (2^16) * (5^4)
  ∀ n : ℕ, n = 12 →
  ∀ num_pairs : ℕ, num_pairs = n / 2 →
  Seq.initial > 0 →
  initial →: (Seq.iterate (λ x, (x / 5) * 2) 12 initial) = after_steps :=
by
  sorry

end final_integer_after_steps_l710_710991


namespace part1_arithmetic_sequence_part2_general_formula_l710_710069

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710069


namespace part1_arithmetic_sequence_part2_general_formula_l710_710096

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710096


namespace count_valid_square_roots_l710_710857

noncomputable def is_valid_square_root (x : ℝ) : Prop :=
  x >= 0

theorem count_valid_square_roots (x : ℝ) (a b : ℝ) :
  let expr1 := sqrt 10,
      expr2 := sqrt (-x^2 - 1),
      expr3 := sqrt (a^2 + b^2),
      expr4 := sqrt 0,
      expr5 := sqrt 1
  in is_valid_square_root expr1 ∧ ¬ is_valid_square_root expr2 ∧ is_valid_square_root expr3 ∧ is_valid_square_root expr4 ∧ is_valid_square_root expr5 →
     (4 = 4) :=
by {
  sorry
}

end count_valid_square_roots_l710_710857


namespace constant_term_expansion_l710_710289

theorem constant_term_expansion (k: ℕ) (h1: 2 * k = 8) : 
  (choose 8 k * 3^(8-k) * 2^k : ℤ) = 90720 := by
  sorry

end constant_term_expansion_l710_710289


namespace range_of_a_l710_710825

/-- Given a fixed point A(a, 3) is outside the circle x^2 + y^2 - 2ax - 3y + a^2 + a = 0,
we want to show that the range of values for a is (0, 9/4). -/
theorem range_of_a (a : ℝ) :
  (∃ (A : ℝ × ℝ), A = (a, 3) ∧ ¬(∃ (x y : ℝ), x^2 + y^2 - 2*a*x - 3*y + a^2 + a = 0))
  ↔ (0 < a ∧ a < 9/4) :=
sorry

end range_of_a_l710_710825


namespace total_reduction_500_l710_710350

noncomputable def total_price_reduction (P : ℝ) (first_reduction_percent : ℝ) (second_reduction_percent : ℝ) : ℝ :=
  let first_reduction := P * first_reduction_percent / 100
  let intermediate_price := P - first_reduction
  let second_reduction := intermediate_price * second_reduction_percent / 100
  let final_price := intermediate_price - second_reduction
  P - final_price

theorem total_reduction_500 (P : ℝ) (first_reduction_percent : ℝ)  (second_reduction_percent: ℝ) (h₁ : P = 500) (h₂ : first_reduction_percent = 5) (h₃ : second_reduction_percent = 4):
  total_price_reduction P first_reduction_percent second_reduction_percent = 44 := 
by
  sorry

end total_reduction_500_l710_710350


namespace FI_squared_correct_l710_710912

noncomputable def FI_squared : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (4, 4)
  let D : ℝ × ℝ := (0, 4)
  let E : ℝ × ℝ := (3, 0)
  let H : ℝ × ℝ := (0, 1)
  let F : ℝ × ℝ := (4, 1)
  let G : ℝ × ℝ := (1, 4)
  let I : ℝ × ℝ := (3, 0)
  let J : ℝ × ℝ := (0, 1)
  let FI_squared := (4 - 3)^2 + (1 - 0)^2
  FI_squared

theorem FI_squared_correct : FI_squared = 2 :=
by
  sorry

end FI_squared_correct_l710_710912


namespace arctan_sum_of_roots_l710_710602

open Real

theorem arctan_sum_of_roots : 
  let x1 := (-sin (3 * π / 5) + sqrt (sin (3 * π / 5)^2 - 4 * cos (3 * π / 5))) / 2
  let x2 := (-sin (3 * π / 5) - sqrt (sin (3 * π / 5)^2 - 4 * cos (3 * π / 5))) / 2 in
  arctan x1 + arctan x2 = π / 5 :=
sorry

end arctan_sum_of_roots_l710_710602


namespace circles_ordered_by_radius_l710_710766

def circle_radii_ordered (rA rB rC : ℝ) : Prop :=
  rA < rC ∧ rC < rB

theorem circles_ordered_by_radius :
  let rA := 2
  let CB := 10 * Real.pi
  let AC := 16 * Real.pi
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  circle_radii_ordered rA rB rC :=
by
  intros
  let rA := 2
  let CB := 10 * Real.pi
  let AC := 16 * Real.pi
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  show circle_radii_ordered rA rB rC
  sorry

end circles_ordered_by_radius_l710_710766


namespace complex_numbers_sum_l710_710650

theorem complex_numbers_sum (a b c d e f : ℂ) 
  (h_b : b = 4)
  (h_e : e = -a - c)
  (h_sum : (a + b * complex.I) + (c + d * complex.I) + (e + f * complex.I) = 6 + 3 * complex.I) : 
  d + f = -1 := 
by 
  sorry

end complex_numbers_sum_l710_710650


namespace power_function_through_point_l710_710615

theorem power_function_through_point (m : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ m)
  (p : (2:ℝ, (1 / 4):ℝ) ∈ set_of (λ x : ℝ × ℝ, (x.snd = f x.fst))) :
  f = λ x : ℝ, x ^ (-2 : ℝ) :=
by
  sorry

end power_function_through_point_l710_710615


namespace problem_conditions_l710_710092

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710092


namespace initial_gap_l710_710884

theorem initial_gap
  (teena_speed : ℝ) (coe_speed : ℝ) (ahead_distance : ℝ) (time_minutes : ℝ)
  (h1 : teena_speed = 55)
  (h2 : coe_speed = 40)
  (h3 : ahead_distance = 15)
  (h4 : time_minutes = 90) :
  (teena_speed - coe_speed) * (time_minutes / 60) - ahead_distance = 7.5 :=
by
  simp [h1, h2, h3, h4]
  norm_num
  sorry

end initial_gap_l710_710884


namespace part1_sequence_arithmetic_part2_general_formula_l710_710148

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710148


namespace max_planes_of_symmetry_l710_710293

-- Definitions of the conditions
def three_pairwise_non_parallel_lines (l1 l2 l3 : affine_space ℝ ℝ) : Prop :=
  (¬∃ c : ℝ, l1 = l2 + c * l3) ∧
  (¬∃ c : ℝ, l2 = l1 + c * l3) ∧
  (¬∃ c : ℝ, l3 = l1 + c * l2)

-- Theorem statement
theorem max_planes_of_symmetry (l1 l2 l3 : affine_space ℝ ℝ)
  (h : three_pairwise_non_parallel_lines l1 l2 l3) :
  ∃ (n : ℕ), n = 9 :=
sorry

end max_planes_of_symmetry_l710_710293


namespace morgan_olivia_same_debt_l710_710186

theorem morgan_olivia_same_debt (t : ℝ) : 
  (200 * (1 + 0.12 * t) = 300 * (1 + 0.04 * t)) → 
  t = 25 / 3 :=
by
  sorry

end morgan_olivia_same_debt_l710_710186


namespace sequence_bound_l710_710770

-- Define the conditions and the sequence
def is_valid_sequence (seq : List ℕ) : Prop :=
  seq.length >= 2 ∧
  seq.head = 1 ∧
  seq.tail.head = 2 ∧
  ∀ i j, 0 ≤ i < seq.length → 0 ≤ j < seq.length → seq.nth_le i (nat.lt_of_lt_le i (le_of_lt seq.length)) + seq.nth_le j (nat.lt_of_lt_le j (le_of_lt seq.length)) ∉ set.of_list seq

-- The theorem to be proved
theorem sequence_bound (seq : List ℕ) (k : ℕ) :
  is_valid_sequence seq →
  (∀ n ∈ seq, n < k) →
  seq.length ≤ (k / 3) + 2 :=
by
  -- Skip the proof
  sorry

end sequence_bound_l710_710770


namespace proof_problem_l710_710574

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := 3 * Real.sin (ω * x + φ)

theorem proof_problem
  (ω : ℝ) (φ : ℝ)
  (h1 : ω > 0)
  (h2 : -π / 2 < φ ∧ φ < π / 2)
  (h_period : ∀ x, f x ω φ = f (x + π) ω φ)
  (h_sym : ∀ x, f x ω φ = f (4 * π / 3 - x) ω φ) :
  ( (f 0 ω φ ≠ 1 / 2)
    ∧ (f (5 * π / 12) ω φ = 0)
    ∧ (∀ x, x ∈ Icc (π / 12) (2 * π / 3) → ¬ (f x ω φ < 0))
    ∧ (∀ x, f x ω φ ≠ f (x + abs φ) 2 0) ) →
  true := sorry

end proof_problem_l710_710574


namespace sequence_bn_arithmetic_and_an_formula_l710_710130

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710130


namespace concrete_for_supporting_pillars_l710_710551

-- Define the given conditions
def roadway_deck_concrete : ℕ := 1600
def one_anchor_concrete : ℕ := 700
def total_bridge_concrete : ℕ := 4800

-- State the theorem
theorem concrete_for_supporting_pillars :
  let total_anchors_concrete := 2 * one_anchor_concrete in
  let total_deck_and_anchors_concrete := roadway_deck_concrete + total_anchors_concrete in
  total_bridge_concrete - total_deck_and_anchors_concrete = 1800 :=
by
  sorry

end concrete_for_supporting_pillars_l710_710551


namespace b_seq_arithmetic_a_seq_formula_l710_710156

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710156


namespace b_arithmetic_sequence_general_formula_a_l710_710061

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710061


namespace fractions_are_integers_l710_710571

theorem fractions_are_integers (a b c : ℤ) (h : ∃ k : ℤ, (a * b / c) + (a * c / b) + (b * c / a) = k) :
  ∃ k1 k2 k3 : ℤ, (a * b / c) = k1 ∧ (a * c / b) = k2 ∧ (b * c / a) = k3 :=
by
  sorry

end fractions_are_integers_l710_710571


namespace probability_of_multiple_of_42_is_zero_l710_710691

-- Given conditions
def factors_200 : Set ℕ := {1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200}
def multiple_of_42 (n : ℕ) : Prop := n % 42 = 0

-- Problem statement: the probability of selecting a multiple of 42 from the factors of 200 is 0.
theorem probability_of_multiple_of_42_is_zero : 
  ∀ (n : ℕ), n ∈ factors_200 → ¬ multiple_of_42 n := 
by
  sorry

end probability_of_multiple_of_42_is_zero_l710_710691


namespace max_quadratic_function_at_intersection_l710_710914

-- Define the quadratic function f(x)
noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x * (3 - x)

-- Define function g(x) = 2 * ln x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log x

-- Define the circle equation
def circle (x y r : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = r ^ 2

-- Define point P(x0, y0) conditions
def point_P (x0 y0 r : ℝ) : Prop := g x0 = y0 ∧ circle x0 y0 r

theorem max_quadratic_function_at_intersection (x0 y0 : ℝ) (hP: point_P x0 y0 1) :
  ∃ x, f x = 9 / 8 :=
by
  use 3 / 2
  sorry

end max_quadratic_function_at_intersection_l710_710914


namespace distance_travelled_downstream_l710_710337

theorem distance_travelled_downstream
  (boat_speed : ℝ) (current_speed : ℝ) (time_in_minutes : ℝ) (distance : ℝ)
  (h_boat_speed : boat_speed = 15)
  (h_current_speed : current_speed = 3)
  (h_time : time_in_minutes = 12)
  (h_distance : distance = 3.6) :
  distance = (boat_speed + current_speed) * (time_in_minutes / 60) * 60 / 60 := by
begin
  rw [h_boat_speed, h_current_speed, h_time, h_distance],
  norm_num,
  sorry
end

end distance_travelled_downstream_l710_710337


namespace polynomial_divisible_by_square_l710_710585

def f (x : ℝ) (a1 a2 a3 a4 : ℝ) : ℝ := x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4
def f' (x : ℝ) (a1 a2 a3 : ℝ) : ℝ := 4 * x^3 + 3 * a1 * x^2 + 2 * a2 * x + a3

theorem polynomial_divisible_by_square (x0 a1 a2 a3 a4 : ℝ) 
  (h1 : f x0 a1 a2 a3 a4 = 0) 
  (h2 : f' x0 a1 a2 a3 = 0) : 
  ∃ g : ℝ → ℝ, ∀ x : ℝ, f x a1 a2 a3 a4 = (x - x0)^2 * (g x) :=
sorry

end polynomial_divisible_by_square_l710_710585


namespace triangle_inequality_x_range_l710_710469

theorem triangle_inequality_x_range {x : ℝ} (h1 : 3 + 6 > x) (h2 : x + 3 > 6) :
  3 < x ∧ x < 9 :=
by 
  sorry

end triangle_inequality_x_range_l710_710469


namespace factorize_expression_l710_710784

theorem factorize_expression (m x : ℝ) : 
  m^3 * (x - 2) - m * (x - 2) = m * (x - 2) * (m + 1) * (m - 1) := 
by sorry

end factorize_expression_l710_710784


namespace b_arithmetic_a_general_formula_l710_710016

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710016


namespace chord_length_of_line_and_curve_C_l710_710267

-- Define the parametric form of the curve C
def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 + sqrt 10 * Real.cos θ, -1 + sqrt 10 * Real.sin θ)

-- Define the parametric form of the line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 1 + t)

-- Define the center of the curve C (circle)
def center_C : ℝ × ℝ := (2, -1)

-- Define the radius of the curve C (circle)
def radius_C : ℝ := sqrt 10

-- Define the standard form of the line l
def line_l_standard (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

-- Define the distance from the center of the circle to the line
def distance_from_center_to_line : ℝ :=
  let (x0, y0) := center_C in
  abs ((1 * x0) + (-2) * y0 + 1) / sqrt (1^2 + (-2)^2)

-- State the problem: the length of the chord formed by line l and curve C
theorem chord_length_of_line_and_curve_C :
  @eq ℝ (2 * sqrt (radius_C^2 - distance_from_center_to_line^2)) (2 * sqrt 5) :=
sorry

end chord_length_of_line_and_curve_C_l710_710267


namespace percent_problem_l710_710510

theorem percent_problem :
  ∀ (x : ℝ), 0.60 * 600 = 0.50 * x → x = 720 :=
by
  intros x h
  sorry

end percent_problem_l710_710510


namespace max_distance_car_motorcycle_l710_710703

-- Definition related to the positions of car and motorcycle
def car_position (t : ℝ) : ℝ := 40 * t
def motorcycle_position (t : ℝ) : ℝ := 16 * t^2 + 9
def distance (t : ℝ) : ℝ := abs (motorcycle_position t - car_position t)

-- Problem statement in Lean 4: 
theorem max_distance_car_motorcycle : 
  ∃ t ∈ set.Icc 0 2, ∀ t' ∈ set.Icc 0 2, distance t' ≤ distance t ∧ distance t = 25 :=
sorry

end max_distance_car_motorcycle_l710_710703


namespace calculate_result_l710_710761

theorem calculate_result :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = Real.cos (Real.pi / 4) :=
by
  sorry

end calculate_result_l710_710761


namespace leonid_painted_cells_l710_710939

theorem leonid_painted_cells (k l : ℕ) (hkl : k * l = 74) :
  ∃ (painted_cells : ℕ), painted_cells = ((2 * k + 1) * (2 * l + 1) - 74) ∧ (painted_cells = 373 ∨ painted_cells = 301) :=
by
  sorry

end leonid_painted_cells_l710_710939


namespace min_white_surface_area_is_five_over_ninety_six_l710_710403

noncomputable def fraction_white_surface_area (total_surface_area white_surface_area : ℕ) :=
  (white_surface_area : ℚ) / (total_surface_area : ℚ)

theorem min_white_surface_area_is_five_over_ninety_six :
  let total_surface_area := 96
  let white_surface_area := 5
  fraction_white_surface_area total_surface_area white_surface_area = 5 / 96 :=
by
  sorry

end min_white_surface_area_is_five_over_ninety_six_l710_710403


namespace cannot_end_with_only_piles_of_three_l710_710730

theorem cannot_end_with_only_piles_of_three (n : ℕ) :
  ∀ (seashells piles : ℕ), seashells = 637 - n ∧ piles = 1 + n → ¬(seashells = 3 * piles) :=
by
  intros seashells piles h
  dsimp only at h
  rw [h.left, h.right]
  have : 637 - n ≠ 3 * (n + 1) := sorry
  rw this
  tauto

end cannot_end_with_only_piles_of_three_l710_710730


namespace part1_sequence_arithmetic_part2_general_formula_l710_710145

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710145


namespace symmetric_point_origin_l710_710243

theorem symmetric_point_origin (x y : ℝ) (h : x = -2 ∧ y = 3) : (-x, -y) = (2, -3) := by
  cases h with
  | intro h_x h_y => 
    rw [h_x, h_y]
    simp
    sorry

end symmetric_point_origin_l710_710243


namespace survival_probability_estimation_l710_710273

theorem survival_probability_estimation :
  let survival_rates := [
    (5, 4),      -- (Total Transplanted, Number of Survivors)
    (50, 45),  
    (200, 188),  
    (500, 476),  
    (1000, 951), 
    (3000, 285)
  ]
in
  let calculated_rates := survival_rates.map (λ ⟨n, m⟩, (m : ℝ) / n)
in
  calculated_rates.all (λ r, |r - 0.95| ≤ 0.01) → 
  0.95 ≠ 0 := sorry

end survival_probability_estimation_l710_710273


namespace product_of_solutions_product_of_solutions_is_16_l710_710428

theorem product_of_solutions (y : ℝ) (h : y + 16 / y = 12) : y = 6 + 2 * Real.sqrt 5 ∨ y = 6 - 2 * Real.sqrt 5 :=
begin
  sorry
end

theorem product_of_solutions_is_16 (y1 y2 : ℝ) (h1 : y1 + 16 / y1 = 12) (h2 : y2 + 16 / y2 = 12) : y1 * y2 = 16 :=
begin
  sorry
end

end product_of_solutions_product_of_solutions_is_16_l710_710428


namespace part1_arithmetic_sequence_part2_general_formula_l710_710095

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710095


namespace overtakes_are_even_l710_710338

def eij (i j : ℕ) : ℕ := number_of_overtakes i j  -- Placeholder definition

def ki (i : ℕ) : ℕ := laps_completed i  -- Placeholder definition

theorem overtakes_are_even (car_count : ℕ) (different_start_points : ∀ i j, i ≠ j → start_point i ≠ start_point j)
  (max_two_side_by_side : ∀ t, at_most_two_side_by_side t) (return_to_start : ∀ i, ∃ t, position t i = start_point i)
  (n := 25) (p : car_count = n) :
  ∑ i in range car_count, ∑ j in range car_count, if i < j then eij i j else 0 + eij j i % 2 = 0 :=
by
  sorry

end overtakes_are_even_l710_710338


namespace g20_members_from_asia_l710_710606

theorem g20_members_from_asia :
  ∃ (a e am af oc : ℕ),  -- number of members from Asia, Europe, America, Africa, Oceania
    a > e ∧ e > am ∧  -- Asia has most members, followed by Europe, then America
    af = oc ∧          -- Africa and Oceania have equal members
    20 = a + e + am + af + oc ∧  -- Total number of G20 countries equals to 20
    a = am + 2 ∧          -- Asia's member number is the highest in consecutive numbers
    e = am + 1 ∧          -- Europe follows as consecutive number after America
    am = 5 ∧              -- America has 5 members
    a = 7.                -- Therefore, members from Asia is 7
Proof := by
  sorry

end g20_members_from_asia_l710_710606


namespace part1_arithmetic_sequence_part2_general_formula_l710_710099

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710099


namespace evaluate_f_double_composite_l710_710859

def f : ℝ → ℝ :=
  λ x, if x > 0 then Real.log x / Real.log 2 else 3^x + 1

theorem evaluate_f_double_composite :
  f (f (1 / 4)) = 10 / 9 := by
  sorry

end evaluate_f_double_composite_l710_710859


namespace sheela_overall_total_income_l710_710216

def monthly_income_in_rs (income: ℝ) (savings: ℝ) (percent: ℝ): Prop :=
  savings = percent * income

def overall_total_income_in_rs (monthly_income: ℝ) 
                              (savings_deposit: ℝ) (fd_deposit: ℝ) 
                              (savings_interest_rate_monthly: ℝ) 
                              (fd_interest_rate_annual: ℝ): ℝ :=
  let annual_income := monthly_income * 12
  let savings_interest := savings_deposit * (savings_interest_rate_monthly * 12)
  let fd_interest := fd_deposit * fd_interest_rate_annual
  annual_income + savings_interest + fd_interest

theorem sheela_overall_total_income:
  ∀ (monthly_income: ℝ)
    (savings_deposit: ℝ) (fd_deposit: ℝ)
    (savings_interest_rate_monthly: ℝ) (fd_interest_rate_annual: ℝ),
    (monthly_income_in_rs monthly_income savings_deposit 0.28)  →
    monthly_income = 16071.43 →
    savings_deposit = 4500 →
    fd_deposit = 3000 →
    savings_interest_rate_monthly = 0.02 →
    fd_interest_rate_annual = 0.06 →
    overall_total_income_in_rs monthly_income savings_deposit fd_deposit
                           savings_interest_rate_monthly fd_interest_rate_annual
    = 194117.16 := 
by
  intros
  sorry

end sheela_overall_total_income_l710_710216


namespace find_selling_price_l710_710714

-- Define the basic parameters
def cost := 80
def s0 := 30
def profit0 := 50
def desired_profit := 2000

-- Additional shirts sold per price reduction
def add_shirts (p : ℕ) := 2 * p

-- Number of shirts sold given selling price x
def num_shirts (x : ℕ) := 290 - 2 * x

-- Profit equation
def profit_equation (x : ℕ) := (x - cost) * num_shirts x = desired_profit

theorem find_selling_price (x : ℕ) :
  (x = 105 ∨ x = 120) ↔ profit_equation x := by
  sorry

end find_selling_price_l710_710714


namespace tangent_circle_exists_l710_710545

theorem tangent_circle_exists
  {A B C M N: Point}
  (h_triangle: is_triangle A B C)
  (h_M_on_AC: lies_on M (segment A C))
  (h_N_on_BC: lies_on N (segment B C))
  (h_length_condition: dist M N = dist A M + dist B N) :
  ∃ (O: Circle), ∀ (M' N': Point), 
  (lies_on M' (segment A C)) →
  (lies_on N' (segment B C)) →
  (dist M' N' = dist A M' + dist B N') →
  is_tangent_line (line_through M' N') O :=
sorry

end tangent_circle_exists_l710_710545


namespace probability_rel_prime_to_42_l710_710307

theorem probability_rel_prime_to_42 : 
  let n := 42 in
  let prime_factors := [2, 3, 7] in
  let relatively_prime_count := n * (1 - 1/prime_factors[0]) * (1 - 1/prime_factors[1]) * (1 - 1/prime_factors[2]) in
  let total_count := 42 in
  (relatively_prime_count / total_count) = 4 / 7 :=
by
  sorry

end probability_rel_prime_to_42_l710_710307


namespace count_consecutive_integers_l710_710641

theorem count_consecutive_integers : 
  ∃ n : ℕ, (∀ x : ℕ, (1 < x ∧ x < 111) → (x - 1) + x + (x + 1) < 333) ∧ n = 109 := 
  by
    sorry

end count_consecutive_integers_l710_710641


namespace sum_first_6_terms_l710_710895

-- Definitions based on conditions
def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

def a1_condition (a : ℕ → ℕ) : Prop :=
  a 1 = 2

-- Lean 4 statement for the proof problem
theorem sum_first_6_terms (a : ℕ → ℕ) (h_seq : sequence a) (h_a1 : a1_condition a) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 126 :=
by
  sorry

end sum_first_6_terms_l710_710895


namespace even_function_exists_l710_710573

def f (x m : ℝ) : ℝ := x^2 + m * x

theorem even_function_exists : ∃ m : ℝ, ∀ x : ℝ, f x m = f (-x) m :=
by
  use 0
  intros x
  unfold f
  simp

end even_function_exists_l710_710573


namespace weight_of_new_student_l710_710335

theorem weight_of_new_student (avg_weight_29 : ℕ → ℕ)
                              (avg_29 : avg_weight_29 29 = 28)
                              (avg_weight_30 : ℕ → ℕ)
                              (avg_30 : avg_weight_30 30 = 27.2) :
  ∃ (weight_new_student : ℕ), weight_new_student = 4 := by
  sorry

end weight_of_new_student_l710_710335


namespace find_unknown_number_l710_710707

theorem find_unknown_number :
  ∃ (x : ℝ), (786 * x) / 30 = 1938.8 → x = 74 :=
by 
  sorry

end find_unknown_number_l710_710707


namespace calculate_mr_hernandez_tax_l710_710187

def convert_income (usd_income : Float) (scale : Float) : Float :=
  usd_income * scale

def calculate_tax 
  (income : Float) 
  (brackets : List (Float × Float)) : Float :=
  let rec calc (remaining: Float) (brackets : List (Float × Float)) (acc : Float) :=
    match brackets with
    | [] => acc
    | (limit, rate) :: rest =>
      let taxable = Float.min limit remaining
      calc (remaining - taxable) rest (acc + taxable * rate)
  calc income brackets 0

def prorate_amount (amount : Float) (months_resident : Float) (total_months : Float) : Float :=
  amount * (months_resident / total_months)

def final_tax (income_usd : Float) (scale_factor : Float) (residency_months : Float) (total_months : Float)
  (standard_deduction : Float) (tax_credit : Float) (tax_brackets : List (Float × Float)) : Float :=
  let income = convert_income income_usd scale_factor
  let tax = calculate_tax income tax_brackets
  let prorated_tax = prorate_amount tax residency_months total_months
  prorated_tax - tax_credit

theorem calculate_mr_hernandez_tax :
  final_tax 120000 2.83 9 12 7000 700 [(15000, 0.02), (20000, 0.04), (35000, 0.06), (20000, 0.08), (Float.infinity, 0.10)] = 21620 :=
by
  sorry

end calculate_mr_hernandez_tax_l710_710187


namespace jasmine_percentage_new_solution_l710_710744

-- Define the initial conditions
def initial_volume : ℝ := 80
def initial_jasmine_percent : ℝ := 0.10
def added_jasmine : ℝ := 5
def added_water : ℝ := 15

-- Define the correct answer
theorem jasmine_percentage_new_solution :
  let initial_jasmine := initial_jasmine_percent * initial_volume
  let new_jasmine := initial_jasmine + added_jasmine
  let total_new_volume := initial_volume + added_jasmine + added_water
  (new_jasmine / total_new_volume) * 100 = 13 := 
by 
  sorry

end jasmine_percentage_new_solution_l710_710744


namespace simplify_expression_l710_710984

theorem simplify_expression :
  (1024 ^ (1/5) * 125 ^ (1/3)) = 20 :=
by
  have h1 : 1024 = 2 ^ 10 := by norm_num
  have h2 : 125 = 5 ^ 3 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end simplify_expression_l710_710984


namespace relatively_prime_probability_42_l710_710302

theorem relatively_prime_probability_42 : 
  (λ x, (x ≤ 42 ∧ x > 0 ∧ Nat.gcd x 42 = 1)) / (λ x, (x ≤ 42 ∧ x > 0)) = 2/7 :=
by 
  sorry

end relatively_prime_probability_42_l710_710302


namespace slopes_product_l710_710472

/-- 
Given an ellipse with equation x²/a² + y²/b² = 1 and an eccentricity of √2/2,
and a line passing through a point on the ellipse intersecting it at two points 
symmetric about the origin, we want to prove that the product of the slopes of the 
lines at these intersections is -1/2.
-/
theorem slopes_product : 
  ∀ (a b : ℝ) (h1 : a = √2 * b)
    (M : ℝ × ℝ)
    (hx : (M.1 / a) ^ 2 + (M.2 / b) ^ 2 = 1)
    (A B : ℝ × ℝ)
    (k1 k2 : ℝ)
    (h2 : A = (M.1, k1 * M.1))
    (h3 : B = (-M.1, k2 * (-M.1)))
    (h4 : M = (M.1, M.2))
    (h5 : A = (-B.1, -B.2))
  , k1 * k2 = -1 / 2 :=
by 
  sorry

end slopes_product_l710_710472


namespace probability_two_one_color_l710_710351

theorem probability_two_one_color (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (drawn_balls : ℕ)
  (h_total : total_balls = 19) (h_black : black_balls = 10) (h_white : white_balls = 9) (h_drawn : drawn_balls = 3) :
  let total_ways := (Nat.choose total_balls drawn_balls),
      ways_two_black_one_white := (Nat.choose black_balls 2) * (Nat.choose white_balls 1),
      ways_one_black_two_white := (Nat.choose black_balls 1) * (Nat.choose white_balls 2),
      favorable_ways := ways_two_black_one_white + ways_one_black_two_white
  in (favorable_ways / total_ways : ℚ) = 85 / 107 :=
by
  sorry

end probability_two_one_color_l710_710351


namespace maximize_probability_pass_l710_710328

/-- 
  Xiao Zhang participated in a test with ten multiple-choice questions.
  Each correct answer earns one point, each wrong answer deducts one point, and no answer earns zero points.
  His goal is to score at least 7 points to pass. Xiao Zhang is certain that he answered the first six questions correctly.
  Each of the remaining questions has a probability of \( \frac{1}{2} \) of being correct.
  Prove that to maximize his probability of passing (scoring at least 7 points), Xiao Zhang should attempt 1 or 3 more questions.
-/
theorem maximize_probability_pass
  (answers_correct_first_six : ℕ)
  (correct_points : ℕ)
  (wrong_points : ℕ)
  (remaining_questions : ℕ)
  (prob_correct : ℚ)
  (goal_points : ℕ) :
  answers_correct_first_six = 6 →
  correct_points = 1 →
  wrong_points = -1 →
  remaining_questions = 4 →
  prob_correct = 1 / 2 →
  goal_points = 7 →
  (maximize_attempts : ℕ) ∈ {7, 9} :=
by
  intro h1 h2 h3 h4 h5 h6
  -- skipping proof since only statement is required
  sorry

end maximize_probability_pass_l710_710328


namespace students_wear_other_colors_l710_710333

variable (TotalStudents : ℕ) (BluePercentage RedPercentage GreenPercentage : ℝ)

theorem students_wear_other_colors :
  TotalStudents = 700 →
  BluePercentage = 0.45 →
  RedPercentage = 0.23 →
  GreenPercentage = 0.15 →
  (TotalStudents * (1 - (BluePercentage + RedPercentage + GreenPercentage))).to_nat = 119 :=
by
  intros hTotal hBlue hRed hGreen
  sorry

end students_wear_other_colors_l710_710333


namespace find_CM_length_l710_710622

theorem find_CM_length (A B C M : Point) (hAB : dist A B = 9) (hBC : dist B C = 3)
  (h_right_angle : ∠ABC = 90) (hM_divides : dist A M / dist M B = 1 / 2) : 
  dist C M = Real.sqrt 33 := 
  sorry

end find_CM_length_l710_710622


namespace find_m_l710_710489

theorem find_m (m : ℝ) 
  (h : (1 : ℝ) * (-3 : ℝ) + (3 : ℝ) * ((3 : ℝ) + 2 * m) = 0) : 
  m = -1 :=
by sorry

end find_m_l710_710489


namespace green_more_than_blue_l710_710331

variable (B Y G : ℕ)

theorem green_more_than_blue
  (h_sum : B + Y + G = 126)
  (h_ratio : ∃ k : ℕ, B = 3 * k ∧ Y = 7 * k ∧ G = 8 * k) :
  G - B = 35 := by
  sorry

end green_more_than_blue_l710_710331


namespace b_arithmetic_sequence_general_formula_a_l710_710064

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710064


namespace alpha_beta_range_l710_710512

theorem alpha_beta_range (α β : ℝ) (h1 : - (π / 2) < α) (h2 : α < β) (h3 : β < π) : 
- 3 * (π / 2) < α - β ∧ α - β < 0 :=
by
  sorry

end alpha_beta_range_l710_710512


namespace proof_problem_l710_710842

theorem proof_problem 
  (a1 a2 b2 : ℚ)
  (ha1 : a1 = -9 + (8/3))
  (ha2 : a2 = -9 + 2 * (8/3))
  (hb2 : b2 = -3) :
  b2 * (a1 + a2) = 30 :=
by
  sorry

end proof_problem_l710_710842


namespace sheela_total_income_l710_710218

-- Define the monthly income as I
def monthly_income (I : Real) : Prop :=
  4500 = 0.28 * I

-- Define the annual income computed from monthly income
def annual_income (I : Real) : Real :=
  I * 12

-- Define the interest earned from savings account 
def interest_savings (principal : Real) (monthly_rate : Real) : Real :=
  principal * (monthly_rate * 12)

-- Define the interest earned from fixed deposit
def interest_fixed (principal : Real) (annual_rate : Real) : Real :=
  principal * annual_rate

-- Overall total income after one year calculation
def overall_total_income (annual_income : Real) (interest_savings : Real) (interest_fixed : Real) : Real :=
  annual_income + interest_savings + interest_fixed

-- Given conditions
variable (I : Real)
variable (principal_savings : Real := 4500)
variable (principal_fixed : Real := 3000)
variable (monthly_rate_savings : Real := 0.02)
variable (annual_rate_fixed : Real := 0.06)

-- Theorem statement to be proved
theorem sheela_total_income :
  monthly_income I →
  overall_total_income (annual_income I) 
                      (interest_savings principal_savings monthly_rate_savings)
                      (interest_fixed principal_fixed annual_rate_fixed)
  = 194117.16 :=
by
  sorry

end sheela_total_income_l710_710218


namespace b_arithmetic_sequence_a_general_formula_l710_710040

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710040


namespace problem_conditions_l710_710086

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710086


namespace cube_root_simplify_l710_710598

theorem cube_root_simplify (x : ℕ) (hx : x = 5488000) : (∛x = 280 * ∛2) :=
by
  rw hx
  -- Continue the proof here.
  sorry

end cube_root_simplify_l710_710598


namespace minimum_routes_A_C_l710_710904

namespace SettlementRoutes

-- Define three settlements A, B, and C
variable (A B C : Type)

-- Assume there are more than one roads connecting each settlement pair directly
variable (k m n : ℕ) -- k: roads between A and B, m: roads between B and C, n: roads between A and C

-- Conditions: Total paths including intermediate nodes
axiom h1 : k + m * n = 34
axiom h2 : m + k * n = 29

-- Theorem: Minimum number of routes connecting A and C is 26
theorem minimum_routes_A_C : ∃ n k m : ℕ, k + m * n = 34 ∧ m + k * n = 29 ∧ n + k * m = 26 := sorry

end SettlementRoutes

end minimum_routes_A_C_l710_710904


namespace count_good_numbers_up_to_2019_l710_710511

def is_good_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^3 + y^3 = z^n

theorem count_good_numbers_up_to_2019 : 
  (finset.range 2020).filter is_good_number = 1346 := 
sorry

end count_good_numbers_up_to_2019_l710_710511


namespace probability_green_given_not_red_l710_710900

theorem probability_green_given_not_red :
  let total_balls := 20
  let red_balls := 5
  let yellow_balls := 5
  let green_balls := 10
  let non_red_balls := total_balls - red_balls

  let probability_green_given_not_red := (green_balls : ℚ) / (non_red_balls : ℚ)

  probability_green_given_not_red = 2 / 3 :=
by
  sorry

end probability_green_given_not_red_l710_710900


namespace problem1_problem2_l710_710876

theorem problem1 (l1 l2 : Line) (l1_eq : l1.equation = fun x y => sqrt 3 * x - y + 1) 
                 (l2_eq : l2.equation = fun x y => sqrt 3 * x - y + 3) 
                 (is_perpendicular : ∀ (n : Line), n.isPerpendicular l1 ∧ n.isPerpendicular l2 → 
                    (Area (triangleFormedByAxes n) = 2 * sqrt 3)) :
  (∃ (b : ℝ), n.equation = fun x y => -sqrt 3 / 3 * x + b ∧ (b = 2 ∨ b = -2)) :=
sorry

theorem problem2 (l1 l2 m : Line) (l1_eq : l1.equation = fun x y => sqrt 3 * x - y + 1)
                 (l2_eq : l2.equation = fun x y => sqrt 3 * x - y + 3) 
                 (passes_through : m.containsPoint (sqrt 3, 4)) 
                 (segment_length : ∀ (p1 p2 : Point), p1.onLine l1 → p2.onLine l2 → 
                    distance p1 p2 = 2) :
  (m.equation = fun x y => x = sqrt 3 ∨ m.equation = fun x y => y = sqrt 3 / 3 * x + 3) :=
sorry

end problem1_problem2_l710_710876


namespace concrete_pillars_correct_l710_710553

-- Definitions based on conditions
def concrete_for_roadway := 1600
def concrete_for_one_anchor := 700
def total_concrete_for_bridge := 4800

-- Total concrete for both anchors
def concrete_for_both_anchors := 2 * concrete_for_one_anchor

-- Total concrete needed for the roadway and anchors
def concrete_for_roadway_and_anchors := concrete_for_roadway + concrete_for_both_anchors

-- Concrete needed for the supporting pillars
def concrete_for_pillars := total_concrete_for_bridge - concrete_for_roadway_and_anchors

-- Proof problem statement, verify that the concrete for the supporting pillars is 1800 tons
theorem concrete_pillars_correct : concrete_for_pillars = 1800 := by
  sorry

end concrete_pillars_correct_l710_710553


namespace circumcenter_on_perpendicular_from_B_l710_710969

/- Given a triangle ABC and a point M on side BC such that ∠ BAM = ∠ BCA, 
   prove that the center of the circumcircle of triangle ABC lies on the 
   line passing through point B and perpendicular to AM. -/
theorem circumcenter_on_perpendicular_from_B
  (A B C M : Point)
  (hM_on_BC : M ∈ Segment B C)
  (hAngle_eq : ∠ BAM = ∠ BCA) :
  let O := circumcenter A B C in
    on_line B (perpendicular_line_from_AM A B M) O :=
begin
  sorry
end

end circumcenter_on_perpendicular_from_B_l710_710969


namespace outlined_square_digit_l710_710278

-- Definition of three-digit numbers
def three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Definition of three-digit power of 3
def power_of_three_digit (x : ℕ) : Prop := ∃ m : ℕ, three_digit x ∧ x = 3^m

-- Definition of three-digit power of 7
def power_of_seven_digit (x : ℕ) : Prop := ∃ n : ℕ, three_digit x ∧ x = 7^n

-- The statement we need to prove
theorem outlined_square_digit :
  ∃ digit : ℕ, (∀ x : ℕ, power_of_three_digit x → String.toNat (String.singleton (toString x).data[1]) = digit) ∧
               (∀ y : ℕ, power_of_seven_digit y → String.toNat (String.singleton (toString y).data[1]) = digit) ∧
                digit = 4 :=
by
  sorry

end outlined_square_digit_l710_710278


namespace sum_ai_le_sum_bi_l710_710974

open BigOperators

variable {α : Type*} [LinearOrderedField α]

theorem sum_ai_le_sum_bi {n : ℕ} {a b : Fin n → α}
  (h1 : ∀ i, 0 < a i)
  (h2 : ∀ i, 0 < b i)
  (h3 : ∑ i, (a i)^2 / b i ≤ ∑ i, b i) :
  ∑ i, a i ≤ ∑ i, b i :=
sorry

end sum_ai_le_sum_bi_l710_710974


namespace part1_arithmetic_sequence_part2_general_formula_l710_710078

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710078


namespace sqrt_interval_l710_710292

theorem sqrt_interval {x : ℤ} (h1 : 3 < real.sqrt 11) (h2 : real.sqrt 11 < 4)
                      (h3 : 4 < real.sqrt 19) (h4 : real.sqrt 19 < 5) :
  x = 4 :=
by
  sorry

end sqrt_interval_l710_710292


namespace bn_arithmetic_sequence_an_formula_l710_710023

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710023


namespace coins_problem_l710_710367

theorem coins_problem :
  ∃ D : ℕ, 
    let A := 21 in
    let B := A - 9 in
    let C := B + 17 in
    (A + B + 5 = C + D) ∧ (D = 9) :=
by
  let A := 21
  let B := A - 9
  let C := B + 17
  exists.intro 9
  split
  · sorry
  · simp

end coins_problem_l710_710367


namespace xy_eq_119_imp_sum_values_l710_710887

theorem xy_eq_119_imp_sum_values (x y : ℕ) (hx : x > 0) (hy : y > 0)
(hx_lt_30 : x < 30) (hy_lt_30 : y < 30) (h : x + y + x * y = 119) :
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 := 
sorry

end xy_eq_119_imp_sum_values_l710_710887


namespace problem1_problem2_l710_710473

-- Definition of the ellipse and line
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4
def line (m x y : ℝ) : Prop := y = x + m

-- Problem 1: Prove m = ±√5 if the line intersects the ellipse at one point
theorem problem1 (m : ℝ) : 
  (∃ x y : ℝ, ellipse x y ∧ line m x y ∧ 
    (∀ x' y', ellipse x' y' ∧ line m x' y' → (x = x' ∧ y = y'))) ↔ 
  (m = sqrt 5 ∨ m = -sqrt 5) := 
sorry

-- Problem 2: Prove m = ±√(30)/4 if the line intersects the ellipse at two points and |PQ| equals the short axis length
theorem problem2 (m : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, x1 ≠ x2 ∧ ellipse x1 y1 ∧ line m x1 y1 ∧ ellipse x2 y2 ∧ line m x2 y2 ∧ 
    (sqrt((x1 - x2)^2 + (y1 - y2)^2) = 2)) ↔ 
  (m = sqrt 30 / 4 ∨ m = -sqrt 30 / 4) :=
sorry

end problem1_problem2_l710_710473


namespace relatively_prime_probability_l710_710309

theorem relatively_prime_probability (n : ℕ) (h : n = 42) :
  let phi := n * (1 - 1 / 2) * (1 - 1 / 3) * (1 - 1 / 7) in
  (phi / n) = 2 / 7 :=
by
  sorry

end relatively_prime_probability_l710_710309


namespace geo_prog_no_squares_l710_710827

theorem geo_prog_no_squares (t : ℕ → ℕ) (r : ℝ) (h_pos : ∀ n, t n > 0) (h_lt : ∀ n, t n < 150)
  (h_sum : t 0 + t 1 + t 2 + t 3 + t 4 + t 5 = 255)
  (h_geo_prog : ∀ n, t (n + 1) = ⌊(t n : ℝ) * r⌋) :
  (∑ n in Finset.range 6, if ∃ k : ℕ, t n = k * k then t n else 0) = 0 :=
by
  sorry

end geo_prog_no_squares_l710_710827


namespace solve_for_y_l710_710227

theorem solve_for_y (y : ℚ) (h : 1 / 3 + 1 / y = 7 / 9) : y = 9 / 4 :=
by
  sorry

end solve_for_y_l710_710227


namespace det_B2_l710_710882

theorem det_B2 (B : Matrix n n ℝ) (h : det B = 8) : det (B ^ 2) = 64 :=
by
  sorry

end det_B2_l710_710882


namespace tan_product_l710_710885

open Real

theorem tan_product (x y : ℝ) 
(h1 : sin x * sin y = 24 / 65) 
(h2 : cos x * cos y = 48 / 65) :
tan x * tan y = 1 / 2 :=
by
  sorry

end tan_product_l710_710885


namespace find_angle_MQP_l710_710751

-- Definitions used in conditions
variables (O : Type) [semicircle O]
variables (P Q M N S : O)
variables (angleK anglePMQ angleMQP : ℝ)

-- Given conditions
axiom diameter_MN : is_diameter M N
axiom angle_K : angleK = 20
axiom angle_PMQ : anglePMQ = 40

-- Proof goal
theorem find_angle_MQP :
  angleMQP = 35 :=
sorry

end find_angle_MQP_l710_710751


namespace f_monotonically_decreasing_intervals_l710_710614

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ x : ℝ, f(x - 1) = f(3 - x)
axiom cond2 : ∀ x : ℝ, f(x - 1) = f(x - 3)
axiom cond3 : ∀ x: ℝ, 1 ≤ x ∧ x ≤ 2 → f(x) = x^2

def is_monotonically_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f(x) ≥ f(y)

def intervals_decreasing (f : ℝ → ℝ) : set (set ℝ) :=
  { I : set ℝ | ∃ k : ℤ, I = set.Icc (2*k - 1 : ℝ) (2*k : ℝ) }

theorem f_monotonically_decreasing_intervals :
  ∀ I ∈ intervals_decreasing f, is_monotonically_decreasing_on f I :=
  sorry

end f_monotonically_decreasing_intervals_l710_710614


namespace solution_set_l710_710562

variable {R : Type*} [LinearOrderedField R] (f : R → R)

theorem solution_set (h₁ : ∀ x, 3 * f x + deriv f x < 0)
    (h₂ : f (log 2) = 1) :
  { x | f x > 8 * exp (-3 * x) } = set.Iio (log 2) :=
by
  sorry

end solution_set_l710_710562


namespace part1_arithmetic_sequence_part2_general_formula_l710_710076

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710076


namespace integer_solutions_exist_l710_710594

theorem integer_solutions_exist (m n : ℤ) :
  ∃ (w x y z : ℤ), 
  (w + x + 2 * y + 2 * z = m) ∧ 
  (2 * w - 2 * x + y - z = n) := sorry

end integer_solutions_exist_l710_710594


namespace lines_with_equal_intercepts_l710_710369

theorem lines_with_equal_intercepts (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (n : ℕ), n = 3 ∧ (∀ l : ℝ → ℝ, (l 1 = 2) → ((l 0 = l (-0)) ∨ (l (-0) = l 0))) :=
by
  sorry

end lines_with_equal_intercepts_l710_710369


namespace lines_form_pencil_through_fixed_point_l710_710626

noncomputable theory

open EuclideanGeometry

def parallel_lines_through_vertices (A B C : Point) (m : Line) (A1 B1 C1 : Point) : Prop :=
  parallel m (Line.mk A A1) ∧ parallel m (Line.mk B B1) ∧ parallel m (Line.mk C C1) ∧
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  A1 ∉ (triangle_sides A B C) ∧ B1 ∉ (triangle_sides A B C) ∧ C1 ∉ (triangle_sides A B C) ∧
  (dist A A1) + (dist B B1) = (dist C C1)

theorem lines_form_pencil_through_fixed_point (A B C D : Point) (m : Line) (A1 B1 C1 : Point)
  (h : parallel_lines_through_vertices A B C m A1 B1 C1) :
  ∃ D : Point, ∀ m : Line, parallel_lines_through_vertices A B C m A1 B1 C1 → D ∈ m :=
sorry

end lines_form_pencil_through_fixed_point_l710_710626


namespace vertical_asymptote_of_g_l710_710437

-- Definitions extracted from conditions in a)
def g (x k : ℝ) : ℝ := (x^2 + 3 * x + k) / (x^2 - 5 * x + 6)

theorem vertical_asymptote_of_g (k : ℝ) :
  (∃ c : ℝ, (c = 2 ∨ c = 3)  ∧ (g c k = 0)) →
  (∀ x : ℝ, g x k = 0 ∨ g x k = ∞) ↔ k = -10 := 
sorry

end vertical_asymptote_of_g_l710_710437


namespace part1_part2_l710_710480

-- Definition of the given function f(x) = (x + 1/x) * ln(x)
def f (x : ℝ) : ℝ := (x + 1/x) * Real.log x

-- Part (Ⅰ): Proof of monotonicity on (0, +∞)
theorem part1 : ∀ x : ℝ, 0 < x → strictly_mono_incr_on f { x : ℝ | 0 < x } := 
by
  intros x h
  sorry

-- Part (Ⅱ): Prove the range of m given the inequality for all x ∈ (0, +∞)
theorem part2 (m : ℝ) :
  (∀ x : ℝ, 0 < x → (2 * f x - m) / Real.exp (m * x) ≤ m) → m ≥ 2 / Real.exp 1 :=
by
  intros h
  sorry

end part1_part2_l710_710480


namespace solve_fish_tank_problem_l710_710605

def fish_tank_problem : Prop :=
  ∃ (first_tank_fish second_tank_fish third_tank_fish : ℕ),
  first_tank_fish = 7 + 8 ∧
  second_tank_fish = 2 * first_tank_fish ∧
  third_tank_fish = 10 ∧
  (third_tank_fish : ℚ) / second_tank_fish = 1 / 3

theorem solve_fish_tank_problem : fish_tank_problem :=
by
  sorry

end solve_fish_tank_problem_l710_710605


namespace decrypt_code_is_SUMAREMOS_l710_710236

-- Given conditions
noncomputable def code_words := ["⌑*⊗", "⨁⌳◉", "*⌑◉", "⊗⦼⨁"]
noncomputable def decrypted_words := ["AMO", "SUR", "REO", "MAS"]
noncomputable def code_to_decrypt := "⊗⦼⌑*⨁⌳⌑◉⊗"

-- Desired proof
theorem decrypt_code_is_SUMAREMOS : 
  decrypt code_words decrypted_words code_to_decrypt = "SUMAREMOS" := sorry

end decrypt_code_is_SUMAREMOS_l710_710236


namespace max_chocolates_l710_710283

theorem max_chocolates (b c k : ℕ) (h1 : b + c = 36) (h2 : c = k * b) (h3 : k > 0) : b ≤ 18 :=
sorry

end max_chocolates_l710_710283


namespace dot_product_a_b_l710_710447

-- Definitions for vectors a and b with given magnitudes
variables (a b : ℝ^3)
-- Given magnitudes of vectors a and b
axiom h₁ : ∥a∥ = 4
axiom h₂ : ∥b∥ = real.sqrt 3
-- Given dot product condition
axiom h₃ : (a + b) ⬝ (a - 2 • b) = 4

-- Goal: Prove that the dot product a ⋅ b is equal to 6
theorem dot_product_a_b : a ⬝ b = 6 := 
by sorry

end dot_product_a_b_l710_710447


namespace minimize_sum_areas_l710_710381

theorem minimize_sum_areas (x : ℝ) (h_wire_length : 0 < x ∧ x < 1) :
    let side_length := x / 4
    let square_area := (side_length ^ 2)
    let circle_radius := (1 - x) / (2 * Real.pi)
    let circle_area := Real.pi * (circle_radius ^ 2)
    let total_area := square_area + circle_area
    total_area = (x^2 / 16 + (1 - x)^2 / (4 * Real.pi)) -> 
    x = Real.pi / (Real.pi + 4) :=
by
  sorry

end minimize_sum_areas_l710_710381


namespace b_arithmetic_a_formula_l710_710001

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l710_710001


namespace exists_mutually_inscribed_pentagons_l710_710220

theorem exists_mutually_inscribed_pentagons : 
  ∀ (P : pentagon), ∃ (Q : pentagon), (Q.inscribed_in P) ∧ (P.inscribed_in Q) :=
by
  sorry

end exists_mutually_inscribed_pentagons_l710_710220


namespace sum_f_nonneg_l710_710172

def f (n k a x : ℕ) : ℤ := (⌊(n + k + x) / a⌋ : ℤ) - (⌊(n + x) / a⌋ : ℤ) - (⌊(k + x) / a⌋ : ℤ) + (⌊x / a⌋ : ℤ)

theorem sum_f_nonneg (n k a : ℕ) (hn : 0 ≤ n) (hk : 0 ≤ k) (ha : 0 < a) (m : ℕ) :
  (finset.range (m + 1)).sum (λ x, f n k a x) ≥ 0 :=
begin
  sorry
end

end sum_f_nonneg_l710_710172


namespace ratio_fraction_l710_710631

theorem ratio_fraction (A B C : ℕ) (h1 : 7 * B = 3 * A) (h2 : 6 * C = 5 * B) :
  (C : ℚ) / (A : ℚ) = 5 / 14 ∧ (A : ℚ) / (C : ℚ) = 14 / 5 :=
by
  sorry

end ratio_fraction_l710_710631


namespace checkerboard_cover_max_tiles_l710_710704

theorem checkerboard_cover_max_tiles :
  ∃ k m : ℕ, (15 * 36 = 49 * k + 25 * m) ∧
    ∀ k' m' : ℕ, (15 * 36 = 49 * k' + 25 * m') → (k + m) ≥ (k' + m') :=
begin
  sorry
end

end checkerboard_cover_max_tiles_l710_710704


namespace polynomial_expression_value_l710_710952

theorem polynomial_expression_value
  (p q r s : ℂ)
  (h1 : p + q + r + s = 0)
  (h2 : p*q + p*r + p*s + q*r + q*s + r*s = -1)
  (h3 : p*q*r + p*q*s + p*r*s + q*r*s = -1)
  (h4 : p*q*r*s = 2) :
  p*(q - r)^2 + q*(r - s)^2 + r*(s - p)^2 + s*(p - q)^2 = -6 :=
by sorry

end polynomial_expression_value_l710_710952


namespace regular_seven_pointed_star_angle_l710_710395

theorem regular_seven_pointed_star_angle :
  (∀ (a : ℝ) (r : ℝ), 
    (∀ (v : ℝ), v ∈ (set.range (λ i, ((2 * i * π) / 7) : ℕ → ℝ)) → 
    ∃ (x y : ℝ), 
      (x^2 + y^2 = r^2) ∧ 
      (x = r * cos v) ∧ 
      (y = r * sin v)) →
    a = 5 * π / 7) :=
sorry

end regular_seven_pointed_star_angle_l710_710395


namespace optimal_submission_l710_710230

open Real

def score (N T : ℕ) : ℝ :=
  2 / (0.5 * abs (N - T) + 1)

def occurrences : ℕ → ℕ
| 1 := 4
| 2 := 2
| 3 := 4
| 4 := 3
| 5 := 5
| 6 := 4
| 7 := 2
| 8 := 0
| 9 := 0
| 10 := 0
| 11 := 0
| 12 := 1
| 13 := 1
| 14 := 0
| 15 := 2
| _ := 0

theorem optimal_submission :
  (score 2 (occurrences 2) = 2 ∧ score 2 (occurrences 2) ≥ score 5 (occurrences 5)) ∨
  (score 5 (occurrences 5) = 2 ∧ score 5 (occurrences 5) ≥ score 2 (occurrences 2)) :=
sorry

end optimal_submission_l710_710230


namespace relatively_prime_probability_42_l710_710303

theorem relatively_prime_probability_42 : 
  (λ x, (x ≤ 42 ∧ x > 0 ∧ Nat.gcd x 42 = 1)) / (λ x, (x ≤ 42 ∧ x > 0)) = 2/7 :=
by 
  sorry

end relatively_prime_probability_42_l710_710303


namespace find_a_l710_710915

theorem find_a (a : ℝ) : 
  let curve := λ x : ℝ, x^3 - a * x
  let derivative := λ x : ℝ, 3 * x^2 - a
  let eq1 := derivative x = 1
  let x1 := Real.sqrt ((a + 1) / 3)
  let x2 := -Real.sqrt ((a + 1) / 3)
  let tangent1 := λ x : ℝ, 1 * (x - x1) + (x1^3 - a * x1)
  let tangent2 := λ x : ℝ, 1 * (x - x2) + (x2^3 - a * x2)
  let distance := |(tangent1 0) - (tangent2 0)|
  (distance = 8) → a = 5 := by 
    sorry 

end find_a_l710_710915


namespace polygon_area_is_5_5_l710_710666

def Point : Type := (ℝ × ℝ)

def vertices : List Point :=
  [(2, 1), (4, 3), (6, 1), (4, -2), (3, 4)]

def area_of_polygon (vertices : List Point) : ℝ :=
  1 / 2 * (Real.abs (
    (vertices[0].1 * vertices[1].2 + vertices[1].1 * vertices[2].2 + vertices[2].1 * vertices[3].2 + vertices[3].1 * vertices[4].2 + vertices[4].1 * vertices[0].2) 
    - 
    (vertices[0].2 * vertices[1].1 + vertices[1].2 * vertices[2].1 + vertices[2].2 * vertices[3].1 + vertices[3].2 * vertices[4].1 + vertices[4].2 * vertices[0].1)
  ))

theorem polygon_area_is_5_5 : 
  area_of_polygon vertices = 5.5 :=
by
  sorry

end polygon_area_is_5_5_l710_710666


namespace prob_hit_10_or_9_prob_hit_below_7_l710_710737

noncomputable def prob_hit_ring : Type := ℝ

variable {p10 p9 p8 p7 : prob_hit_ring}

def total_probability : prob_hit_ring := 1.0

-- Probabilities of hitting specific rings
axiom prob_hit_10 : p10 = 0.21
axiom prob_hit_9 : p9 = 0.23
axiom prob_hit_8 : p8 = 0.25
axiom prob_hit_7 : p7 = 0.28

-- Question 1: Probability of hitting either the 10-ring or 9-ring
theorem prob_hit_10_or_9 : p10 + p9 = 0.44 := by
  rw [prob_hit_10, prob_hit_9]
  norm_num

-- Question 2: Probability of hitting below 7-ring
theorem prob_hit_below_7 : total_probability - (p10 + p9 + p8 + p7) = 0.03 := by
  rw [prob_hit_10, prob_hit_9, prob_hit_8, prob_hit_7]
  norm_num

#check prob_hit_10_or_9
#check prob_hit_below_7

end prob_hit_10_or_9_prob_hit_below_7_l710_710737


namespace range_of_f_on_interval_l710_710866

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem range_of_f_on_interval :
  (set.range (λ x, f x) ∩ set.Icc 3 7) = set.Icc 3 7 :=
sorry

end range_of_f_on_interval_l710_710866


namespace b_arithmetic_a_formula_l710_710006

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l710_710006


namespace bn_arithmetic_sequence_an_formula_l710_710032

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710032


namespace arithmetic_mean_l710_710790

theorem arithmetic_mean (x y : ℝ) (h1 : x = Real.sqrt 2 - 1) (h2 : y = 1 / (Real.sqrt 2 - 1)) :
  (x + y) / 2 = Real.sqrt 2 := sorry

end arithmetic_mean_l710_710790


namespace treaty_signed_on_saturday_l710_710237

-- Define the start day and the total days until the treaty.
def start_day_of_week : Nat := 4 -- Thursday is the 4th day (0 = Sunday, ..., 6 = Saturday)
def days_until_treaty : Nat := 919

-- Calculate the final day of the week after 919 days since start_day_of_week.
def treaty_day_of_week : Nat := (start_day_of_week + days_until_treaty) % 7

-- The goal is to prove that the treaty was signed on a Saturday.
theorem treaty_signed_on_saturday : treaty_day_of_week = 6 :=
by
  -- Implement the proof steps
  sorry

end treaty_signed_on_saturday_l710_710237


namespace domain_of_composition_l710_710572

theorem domain_of_composition {f : ℝ → ℝ} (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≠ none) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → f (x^2) ≠ none :=
by
  sorry

end domain_of_composition_l710_710572


namespace ratio_AC_BC_l710_710923

variables {A B C D E O : Type} [MetricSpace A] [MetricSpace B] 
  [MetricSpace C] [MetricSpace D] 
  [MetricSpace E] [MetricSpace O]

-- Assume the triangle is isosceles
variable (iso_triangle : (AB = BC))

-- The medians AD and EC intersect at point O
variable (med1 : Median AD)
variable (med2 : Median EC)
variable (intersect_O : med1 ∩ med2 = O)

-- Ratio of radii of inscribed circles
def radius_ratio: ℝ := 2 / 3

-- Proof statement: ratio of AC to BC is 20/17
theorem ratio_AC_BC (h : radius_ratio): (AC / BC) = 20 / 17 := 
  sorry

end ratio_AC_BC_l710_710923


namespace rectangle_area_l710_710336

theorem rectangle_area
  (length_rectangle : ℕ)
  (side_square : ℕ)
  (area_square : ℕ)
  (radius_circle : ℕ)
  (breadth_rectangle : ℕ) :
  length_rectangle = 10 →
  area_square = 2025 →
  side_square * side_square = area_square →
  radius_circle = side_square →
  breadth_rectangle = (3 * radius_circle) / 5 →
  length_rectangle * breadth_rectangle = 270 :=
by
  intros h_length h_area h_side_square h_radius_circle h_breadth_rectangle
  rw [h_length, h_area, h_side_square, h_radius_circle, h_breadth_rectangle]
  have side_square_val : ℕ := 45
  rw [nat.mul_self_eq, ←nat.succ_pred_eq_of_pos, nat.succ_pred_eq_of_eq, nat.div_eq_iff_eq_mul_left] at h_side_square
  focus
    exact eq.refl 2025
  focus
    exact nat.zero_lt_succ 44
  simp_rw [side_square_val]
  norm_num [breadth_rectangle, radius_circle]
  norm_num
  simp
  exact eq.refl 270

end rectangle_area_l710_710336


namespace first_month_sale_l710_710366

def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029
def sale6 : ℕ := 4937
def average_sale : ℕ := 5600

theorem first_month_sale :
  let total_sales := average_sale * 6
  let known_sales := sale2 + sale3 + sale4 + sale5 + sale6
  let sale1 := total_sales - known_sales
  sale1 = 5266 :=
by
  sorry

end first_month_sale_l710_710366


namespace sum_of_a_values_l710_710958

theorem sum_of_a_values :
  ∀ (a b c : ℂ),
    (a + b + a * c = 5) →
    (b + c + a * b = 10) →
    (c + a + b * c = 15) →
    let S := {x : ℂ | ∃ (y z : ℂ), (x + y + x * z = 5) ∧ (y + z + x * y = 10) ∧ (z + x + y * z = 15)} in
    (∑ a in S, a) = -7 :=
by
  intros a b c h1 h2 h3 S,
  sorry

end sum_of_a_values_l710_710958


namespace b_arithmetic_sequence_a_general_formula_l710_710037

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710037


namespace sum_f_eq_1009_l710_710826

def f : ℝ → ℝ := sorry

axiom f_add_mul (x y : ℝ) : f (x + y) = f x * f y
axiom f_one : f 1 = 1

theorem sum_f_eq_1009 : 
  (∑ k in Finset.range 1009, (f k)^2 / (f (2 * k + 1))) = 1009 :=
sorry

end sum_f_eq_1009_l710_710826


namespace b_arithmetic_sequence_a_general_formula_l710_710039

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l710_710039


namespace side_length_of_square_perimeter_of_square_l710_710999

theorem side_length_of_square {d s: ℝ} (h: d = 2 * Real.sqrt 2): s = 2 :=
by
  sorry

theorem perimeter_of_square {s P: ℝ} (h: s = 2): P = 8 :=
by
  sorry

end side_length_of_square_perimeter_of_square_l710_710999


namespace total_apartments_l710_710973

theorem total_apartments (apt_initial : ℕ) (apt_reversed : ℕ) (entrances : ℕ) : 
    apt_initial = 636 → apt_reversed = 242 → entrances = 5 → 985 =
      entrances * ((apt_initial - apt_reversed) / 2) := by
  assume h1 : apt_initial = 636
  assume h2 : apt_reversed = 242
  assume h3 : entrances = 5
  calc
    985 = 5 * ((636 - 242) / 2) := by
      rw [h1, h2, h3]
      exact rfl
    ... = 5 * (394 / 2) := by
      rw [Nat.sub_eq]
    ... = 5 * 197 := by
      rw [Nat.div_eq]
    ... = 985 := rfl
  sorry

end total_apartments_l710_710973


namespace expected_steps_unit_interval_l710_710966

noncomputable def expected_steps_to_color_interval : ℝ := 
  -- Placeholder for the function calculating expected steps
  sorry 

theorem expected_steps_unit_interval : expected_steps_to_color_interval = 5 :=
  sorry

end expected_steps_unit_interval_l710_710966


namespace necessary_and_sufficient_l710_710817

theorem necessary_and_sufficient (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬ ((a > 0 ∧ b > 0 → ab < (a + b) / 2 ^ 2) 
  ∧ (ab < (a + b) / 2 ^ 2 → a > 0 ∧ b > 0)) := 
sorry

end necessary_and_sufficient_l710_710817


namespace evaluate_expression_l710_710783

noncomputable def expr : ℚ := (3 ^ 512 + 7 ^ 513) ^ 2 - (3 ^ 512 - 7 ^ 513) ^ 2
noncomputable def k : ℚ := 28 * 2.1 ^ 512

theorem evaluate_expression : expr = k * 10 ^ 513 :=
by
  sorry

end evaluate_expression_l710_710783


namespace siblings_total_weight_l710_710391

theorem siblings_total_weight (A_weight : ℕ) (S_diff : ℕ) (H : A_weight = 50) (H2 : S_diff = 12) :
  let S_weight := A_weight - S_diff in
  let total_weight := A_weight + S_weight in
  total_weight = 88 :=
by
  simp [H, H2]
  sorry

end siblings_total_weight_l710_710391


namespace mandy_15th_replacement_l710_710184

theorem mandy_15th_replacement :
  ∀ (n cycles month: ℕ), cycles = 12 -> month = 1 ->
  (n - 1) * 3 % cycles + month = 7 ↔ n = 15 :=
by
  assume 15 12 1
  intros h_cycle h_month
  sorry

end mandy_15th_replacement_l710_710184


namespace find_A_plus_B_l710_710410

theorem find_A_plus_B:
  ∃ A B : ℚ, 
    (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 → (Bx - 13) / (x^2 - 8 * x + 15) = A / (x - 3) + 4 / (x - 5)) →
    A + B = 22 / 5 :=
begin
  sorry
end

end find_A_plus_B_l710_710410


namespace locus_points_rhombus_AC_BD_l710_710793

theorem locus_points_rhombus_AC_BD (A B C D M : Point) (h1 : Inside M (Rhombus A B C D)) 
    (h2 : ∠A M D + ∠B M C = 180) : OnDiagonals M (AC ∪ BD) :=
sorry

end locus_points_rhombus_AC_BD_l710_710793


namespace remainder_2pow33_minus_1_div_9_l710_710633

theorem remainder_2pow33_minus_1_div_9 : (2^33 - 1) % 9 = 7 := 
  sorry

end remainder_2pow33_minus_1_div_9_l710_710633


namespace range_of_m_l710_710863

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x - (Real.pi / 3))

def p (x : ℝ) : Prop := x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)
def q (x : ℝ) (m : ℝ) : Prop := m - 3 < f x ∧ f x < m + 3

theorem range_of_m (m : ℝ) : (∀ x : ℝ, p x → q x m) ↔ (-1 < m ∧ m < 4) := 
by
  sorry

end range_of_m_l710_710863


namespace correct_eq_count_l710_710327

-- Define the correctness of each expression
def eq1 := (∀ x : ℤ, (-2 * x)^3 = 2 * x^3 = false)
def eq2 := (∀ a : ℤ, a^2 * a^3 = a^3 = false)
def eq3 := (∀ x : ℤ, (-x)^9 / (-x)^3 = x^6 = true)
def eq4 := (∀ a : ℤ, (-3 * a^2)^3 = -9 * a^6 = false)

-- Define the condition that there are exactly one correct equation
def num_correct_eqs := (1 = 1)

-- The theorem statement, proving the count of correct equations is 1
theorem correct_eq_count : eq1 → eq2 → eq3 → eq4 → num_correct_eqs :=
  by intros; sorry

end correct_eq_count_l710_710327


namespace average_of_remaining_two_numbers_l710_710995

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95) 
  (h_avg_ab : (a + b) / 2 = 3.8) 
  (h_avg_cd : (c + d) / 2 = 3.85) :
  ((e + f) / 2) = 4.2 := 
by 
  sorry

end average_of_remaining_two_numbers_l710_710995


namespace concrete_for_supporting_pillars_l710_710552

-- Define the given conditions
def roadway_deck_concrete : ℕ := 1600
def one_anchor_concrete : ℕ := 700
def total_bridge_concrete : ℕ := 4800

-- State the theorem
theorem concrete_for_supporting_pillars :
  let total_anchors_concrete := 2 * one_anchor_concrete in
  let total_deck_and_anchors_concrete := roadway_deck_concrete + total_anchors_concrete in
  total_bridge_concrete - total_deck_and_anchors_concrete = 1800 :=
by
  sorry

end concrete_for_supporting_pillars_l710_710552


namespace problem1_problem2_problem3_problem4_problem5_l710_710879

-- Problem 1
theorem problem1 (n : ℕ) : n ∣ 42 ∧ 7 ∣ n ∧ 2 ∣ n ∧ 3 ∣ n → n = 42 := 
sorry

-- Problem 2
theorem problem2 (m : ℕ) : (∀ d, d ∣ m → d ∣ 18) ∧ (∀ k, k ∣ 18 → k ∣ m) → m = 18 :=
sorry

-- Problem 3
theorem problem3 (t : ℕ) : (∀ k, k ∣ t → k = 1 ∨ k = t) → t = 1 :=
sorry

-- Problem 4
theorem problem4 (p1 p2 : ℕ) : nat.prime p1 ∧ nat.prime p2 ∧ p1 + p2 = 10 ∧ p1 * p2 = 21 → (p1 = 3 ∧ p2 = 7) ∨ (p1 = 7 ∧ p2 = 3) :=
sorry

-- Problem 5
theorem problem5 (p3 p4 : ℕ) : nat.prime p3 ∧ nat.prime p4 ∧ p3 + p4 = 20 ∧ p3 * p4 = 91 → (p3 = 13 ∧ p4 = 7) ∨ (p3 = 7 ∧ p4 = 13) :=
sorry

end problem1_problem2_problem3_problem4_problem5_l710_710879


namespace value_of_exponent_l710_710818

theorem value_of_exponent (a b : ℝ) (h1 : 3^a = 10) (h2 : 3^(2 * b) = 2) : 3^(a - 2 * b) = 5 := 
by sorry

end value_of_exponent_l710_710818


namespace triangles_with_two_colors_l710_710699

theorem triangles_with_two_colors {n : ℕ} 
  (h1 : ∀ (p : Finset ℝ) (hn : p.card = n) 
      (e : p → p → Prop), 
      (∀ (x y : p), e x y → e x y = red ∨ e x y = yellow ∨ e x y = green) /\
      (∀ (a b c : p), 
        (e a b = red ∨ e a b = yellow ∨ e a b = green) ∧ 
        (e b c = red ∨ e b c = yellow ∨ e b c = green) ∧ 
        (e a c = red ∨ e a c = yellow ∨ e a c = green) → 
        (e a b ≠ e b c ∨ e b c ≠ e a c ∨ e a b ≠ e a c))) :
  n < 13 := 
sorry

end triangles_with_two_colors_l710_710699


namespace sequence_term_l710_710830

noncomputable def geometric_sum (n : ℕ) : ℝ :=
  2 * (1 - (1 / 2) ^ n) / (1 - 1 / 2)

theorem sequence_term (m n : ℕ) (h : n < m) : 
  let Sn := geometric_sum n
  let Sn_plus_1 := geometric_sum (n + 1)
  Sn - Sn_plus_1 = -(1 / 2 ^ (n - 1)) := sorry

end sequence_term_l710_710830


namespace inequality_proof_l710_710174

variables {n : ℕ}
variable (x : Fin n → ℝ)
variable (a : Fin n → ℝ)
variable (p q : ℝ)

noncomputable def normalized_a_sum : Prop := ∑ i, a i = 1
noncomputable def pos_x : Prop := ∀ i, x i > 0
noncomputable def nonneg_a : Prop := ∀ i, a i ≥ 0
noncomputable def p_lt_q : Prop := 0 < p ∧ p < q
noncomputable def not_all_zero_a : Prop := ∑ i, a i > 0

theorem inequality_proof
  (h1 : normalized_a_sum a)
  (h2 : pos_x x)
  (h3 : nonneg_a a)
  (h4 : p_lt_q p q)
  (h5 : not_all_zero_a a) :
  (∑ i, a i * (x i)^p)^(1 / p) ≤ (∑ i, a i * (x i)^q)^(1 / q) :=
sorry

end inequality_proof_l710_710174


namespace product_of_roots_l710_710796

noncomputable def quadratic_has_product_of_roots (A B C : ℤ) : ℚ :=
  C / A

theorem product_of_roots (α β : ℚ) (h : 12 * α^2 + 28 * α - 320 = 0) (h2 : 12 * β^2 + 28 * β - 320 = 0) :
  quadratic_has_product_of_roots 12 28 (-320) = -80 / 3 :=
by
  -- Insert proof here
  sorry

end product_of_roots_l710_710796


namespace locus_centroid_MTT_l710_710710

/-- Let \( M, N, P \) be collinear points with \( N \) between \( M \) and \( P \).
Let \( r \) be the perpendicular bisector of \( NP \).
\( O \) lies on \( r \).
Let \( \omega \) be the circle centered at \( O \) and passing through \( N \).
The tangents from \( M \) to \( \omega \) intersect \( \omega \) at \( T \) and \( T' \).
The locus of the centroid of the triangle \( MTT' \) as \( O \) varies over \( r \) is given by
\( 3(m+p)(x^2 + y^2) + 2m(2m + p)x + m^2(m - p) = 0 \).
-/
theorem locus_centroid_MTT'_is_circle
  (M N P : ℝ)
  (h_collinear : N = (M + P) / 2)
  (r : set (ℝ × ℝ))
  (O : ℝ × ℝ)
  (h_O_on_r : O ∈ r)
  (ω : (ℝ × ℝ) → ℝ)
  (h_ω_center : ω O = sqrt ((O.1 - (M + P) / 2)^2 + O.2^2))
  (T T' : ℝ × ℝ)
  (h_tangents : ∀ (M ω : (ℝ × ℝ) → ℝ), is_tangent M (circle ω O) T ∧ is_tangent M (circle ω O) T') :
  let G := ((M + T.1 + T'.1) / 3, (T.2 + T'.2) / 3) in
  3 * (M + P) * (G.1^2 + G.2^2) + 2 * M * (2 * M + P) * G.1 + M^2 * (M - P) = 0 := sorry

end locus_centroid_MTT_l710_710710


namespace male_students_count_l710_710240

variable (M F : ℕ)
variable (average_all average_male average_female : ℕ)
variable (total_male total_female total_all : ℕ)

noncomputable def male_students (M F : ℕ) : ℕ := 8

theorem male_students_count:
  F = 32 -> average_all = 90 -> average_male = 82 -> average_female = 92 ->
  total_male = average_male * M -> total_female = average_female * F -> 
  total_all = average_all * (M + F) -> total_male + total_female = total_all ->
  M = male_students M F := 
by
  intros hF hAvgAll hAvgMale hAvgFemale hTotalMale hTotalFemale hTotalAll hEqTotal
  sorry

end male_students_count_l710_710240


namespace sequence_bn_arithmetic_and_an_formula_l710_710134

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710134


namespace find_angle_F_l710_710931

-- Define the angles of the triangle
variables (D E F : ℝ)

-- Define the conditions given in the problem
def angle_conditions (D E F : ℝ) : Prop :=
  (D = 3 * E) ∧ (E = 18) ∧ (D + E + F = 180)

-- The theorem to prove that angle F is 108 degrees
theorem find_angle_F (D E F : ℝ) (h : angle_conditions D E F) : 
  F = 108 :=
by
  -- The proof body is omitted
  sorry

end find_angle_F_l710_710931


namespace problem_statement_l710_710956

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f x + f 2

theorem problem_statement (f : ℝ → ℝ) 
  (h1 : odd_function f) 
  (h2 : satisfies_condition f) 
  (h3 : f 1 = f 1) : f 5 = C :=
begin
  sorry
end

end problem_statement_l710_710956


namespace max_area_quadrilateral_l710_710531

theorem max_area_quadrilateral 
(hO : O = (0, 0)) 
(hradius : ∀ P, dist O P = 3 ↔ (P = O)) 
(hperpendicular : ∀ x y, x ≠ y → ⟪x, y⟫ = 0) 
(hM : M = (1, √5)) 
(hline1 : ∀ A C, A ≠ C → line_through A C = l₁ ↔ (A, C) ∈ circle(O, 3)) 
(hline2 : ∀ B D, B ≠ D → line_through B D = l₂ ↔ (B, D) ∈ circle(O, 3)) :
  max_area_quadrilateral O l₁ l₂ M = 12 := 
by sorry

end max_area_quadrilateral_l710_710531


namespace distance_point_to_line_l710_710477

theorem distance_point_to_line :
  let x0 := 2
  let y0 := 4
  let z0 := 1
  let A := 1
  let B := 2
  let C := 2
  let D := 3
  let num := |A * x0 + B * y0 + C * z0 + D|
  let denom := Real.sqrt (A^2 + B^2 + C^2)
  (num / denom) = 5 :=
by
  let x0 := 2
  let y0 := 4
  let z0 := 1
  let A := 1
  let B := 2
  let C := 2
  let D := 3
  let num := abs (A * x0 + B * y0 + C * z0 + D)
  let denom := Real.sqrt (A^2 + B^2 + C^2)
  have num_eq : num = 15 := sorry
  have denom_eq : denom = 3 := sorry
  exact (num / denom).to_eq (15 / 3)
#align_elementary_linalg.distance_point_to_line distance_point_to_line

end distance_point_to_line_l710_710477


namespace max_value_of_f_l710_710806

def f (x : ℝ) : ℝ := min (min (4 * x + 1) (x + 2)) (-2 * x + 4)

theorem max_value_of_f : ∃ x : ℝ, f x = 8 / 3 :=
by sorry

end max_value_of_f_l710_710806


namespace RelativelyPrimeProbability_l710_710298

def relatively_prime_probability_42 : Rat :=
  let n := 42
  let total := n
  let rel_prime_count := total - (21 + 14 + 6 - 7 - 3 - 2 + 1)
  let probability := (rel_prime_count : Rat) / total
  probability

theorem RelativelyPrimeProbability : relatively_prime_probability_42 = 2 / 7 :=
sorry

end RelativelyPrimeProbability_l710_710298


namespace b_arithmetic_a_general_formula_l710_710019

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l710_710019


namespace triangle_BKL_is_equilateral_l710_710268

open EuclideanGeometry

noncomputable def IsoscelesCircumscribedEquilateral (A B C K L : Point) : Prop :=
  IsIsoscelesTriangle A B C ∧ Distance (circumcenter A B C) A = Distance A C ∧ 
  IsSquare A K L C → IsEquilateralTriangle B K L

variable {A B C K L : Point}

theorem triangle_BKL_is_equilateral
  (h_isosceles : IsIsoscelesTriangle A B C)
  (h_radius : Distance (circumcenter A B C) A = Distance A C)
  (h_square : IsSquare A K L C) : IsEquilateralTriangle B K L :=
by
  sorry

end triangle_BKL_is_equilateral_l710_710268


namespace camila_bikes_more_l710_710902

-- Definitions based on conditions
def camila_speed : ℝ := 15
def daniel_speed_initial : ℝ := 15
def daniel_speed_after_3hours : ℝ := 10
def biking_time : ℝ := 6
def time_before_decrease : ℝ := 3
def time_after_decrease : ℝ := biking_time - time_before_decrease

def distance_camila := camila_speed * biking_time
def distance_daniel := (daniel_speed_initial * time_before_decrease) + (daniel_speed_after_3hours * time_after_decrease)

-- The statement to prove: Camila has biked 15 more miles than Daniel
theorem camila_bikes_more : distance_camila - distance_daniel = 15 := 
by
  sorry

end camila_bikes_more_l710_710902


namespace part1_arithmetic_sequence_part2_general_formula_l710_710098

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710098


namespace line_equations_l710_710424

noncomputable def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

theorem line_equations (A : ℝ × ℝ) (m : ℝ) (θ : ℝ) : 
  A = (-1, real.sqrt 3) → m = real.sqrt 3 → θ = real.pi / 6 →
  (∃ l : ℝ × ℝ → Prop, (l A) ∧
    (l = λ P, P.1 + 1 = 0 ∨ l = λ P, P.1 - real.sqrt 3 * P.2 + 4 = 0)) :=
by {
  intros hA hm hθ,
  use λ P, P.1 + 1 = 0 ∨ P.1 - real.sqrt 3 * P.2 + 4 = 0,
  split,
  {
    left,
    rw hA,
    norm_num,
  },
  refl,
  sorry
}

end line_equations_l710_710424


namespace b_arithmetic_a_formula_l710_710004

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l710_710004


namespace b_seq_arithmetic_a_seq_formula_l710_710151

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710151


namespace henry_books_count_l710_710492

theorem henry_books_count
  (initial_books : ℕ)
  (boxes : ℕ)
  (books_per_box : ℕ)
  (room_books : ℕ)
  (table_books : ℕ)
  (kitchen_books : ℕ)
  (picked_books : ℕ) :
  initial_books = 99 →
  boxes = 3 →
  books_per_box = 15 →
  room_books = 21 →
  table_books = 4 →
  kitchen_books = 18 →
  picked_books = 12 →
  initial_books - (boxes * books_per_box + room_books + table_books + kitchen_books) + picked_books = 23 :=
by
  intros initial_books_eq boxes_eq books_per_box_eq room_books_eq table_books_eq kitchen_books_eq picked_books_eq
  rw [initial_books_eq, boxes_eq, books_per_box_eq, room_books_eq, table_books_eq, kitchen_books_eq, picked_books_eq]
  norm_num
  sorry

end henry_books_count_l710_710492


namespace part_a_rectangle_with_unique_squares_part_b_not_fill_aquarium_with_unique_cubes_l710_710690

/-- Part a -/
theorem part_a_rectangle_with_unique_squares :
  ∃ (squares : List ℕ) (rect : ℕ × ℕ), (∀ (x ∈ squares), ∃ (a : ℕ), a^2 = x) ∧ rect.1 * rect.2 = List.sum squares := 
    sorry

/-- Part b -/
theorem part_b_not_fill_aquarium_with_unique_cubes :
  ∀ (cubes : List ℕ), (∀ (x y ∈ cubes), x ≠ y) → (∀ (rect : ℕ), ∑ x in cubes, x^3 ≠ rect^3) :=
    sorry

end part_a_rectangle_with_unique_squares_part_b_not_fill_aquarium_with_unique_cubes_l710_710690


namespace part1_arithmetic_sequence_part2_general_formula_l710_710105

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710105


namespace find_Q_coordinates_l710_710581

def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
((a.1 + b.1) / 2, (a.2 + b.2) / 2)

theorem find_Q_coordinates (P Q R S : ℝ × ℝ)
(hP : P = (2, 1))
(hS : S = (14, 7))
(hR_mid : R = midpoint P Q)
(hS_mid : S = midpoint Q R) :
Q = (18, 9) :=
sorry

end find_Q_coordinates_l710_710581


namespace sequence_property_l710_710422

theorem sequence_property :
  ∃ (a_0 a_1 a_2 a_3 : ℕ),
    a_0 + a_1 + a_2 + a_3 = 4 ∧
    (a_0 = ([a_0, a_1, a_2, a_3].count 0)) ∧
    (a_1 = ([a_0, a_1, a_2, a_3].count 1)) ∧
    (a_2 = ([a_0, a_1, a_2, a_3].count 2)) ∧
    (a_3 = ([a_0, a_1, a_2, a_3].count 3)) :=
sorry

end sequence_property_l710_710422


namespace measure_angle_C_values_a_b_l710_710898

theorem measure_angle_C (A B C a b c : ℝ) (hcond1 : c = 2) 
  (hcond2 : real.sin A - real.sin C * (real.cos B + real.sqrt 3 / 3 * real.sin B) = 0) :
  C = π / 3 :=
sorry

theorem values_a_b (A B C a b c : ℝ) (hC : C = π / 3) (hcond1 : c = 2)
  (hcond4 : 1 / 2 * a * b * real.sin C = real.sqrt 3) :
  ab = 4 ∧ a^2 + b^2 = 8 → a = 2 ∧ b = 2 :=
sorry

end measure_angle_C_values_a_b_l710_710898


namespace trajectory_and_line_eq_l710_710529

/-- 
  Point A is given as (4, 0), and N is the orthogonal projection of a moving point M(x, y) on the y-axis. 
  Line MO is perpendicular to line NA.
  (1) Prove that the trajectory C of the moving point M satisfies y^2 = 4x.
  (2) When ∠MOA = π/6, prove that the equation of line NA is either 
      √3*x - y - 4*√3 = 0 or √3*x + y - 4*√3 = 0.
--/
theorem trajectory_and_line_eq (x y : ℝ) (h : (x, y) ≠ (0, 0)) (MO_perp_NA : ∀ x y, (x, y) • (4, -y) = 0)
    (angle_condition : ∀ (x y : ℝ), ∃ θ : ℝ, θ = π / 6) : 
  y^2 = 4*x ∧ ((√3 * x - y - 4 * √3 = 0) ∨ (√3 * x + y - 4 * √3 = 0)) :=
by
  sorry

end trajectory_and_line_eq_l710_710529


namespace maximum_volume_maximum_dot_product_l710_710824

variable {a b c : ℝ}
variable {m : ℝ × ℝ × ℝ} := (1, 3, Real.sqrt 6)
variable {n : ℝ × ℝ × ℝ} := (a, b, c)

theorem maximum_volume (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + b^2 + c^2 = 9) :
  a * b * c ≤ 3 * Real.sqrt 3 :=
sorry

theorem maximum_dot_product (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + b^2 + c^2 = 9) :
  let dot_prod := 1 * a + 3 * b + Real.sqrt 6 * c
  dot_prod ≤ 12 :=
sorry

end maximum_volume_maximum_dot_product_l710_710824


namespace problem_conditions_l710_710080

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710080


namespace bn_arithmetic_sequence_an_formula_l710_710024

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710024


namespace calculation_one_calculation_two_l710_710756

-- Statement for the first problem
theorem calculation_one : 6.42 - 2.8 + 3.58 = 7.2 :=
by sorry

-- Statement for the second problem
theorem calculation_two : 0.36 ÷ (0.4 * (6.1 - 4.6)) = 0.6 :=
by sorry

end calculation_one_calculation_two_l710_710756


namespace swap_numerators_fraction_sum_odd_denominator_l710_710347

theorem swap_numerators_fraction_sum_odd_denominator:
  ∃ (fractions: list (ℕ × ℕ)), 
    (∀ (f ∈ fractions), f.1 ∈ (finset.range 100).image (λ i, i + 1) ∧ f.2 ∈ (finset.range 100).image (λ i, i + 1)) ∧ 
    (finset.univ.sum (λ f, (f.1: ℚ) / f.2) = (a: ℚ) / 2 ∧ 
     ∀ (f₁ f₂: ℕ × ℕ), f₁ ∈ fractions ∧ f₂ ∈ fractions ∧ f₁ ≠ f₂ →
      is_irreducible (finset.univ.sum (λ f, (if f = f₁ then f₂.1 else if f = f₂ then f₁.1 else f.1 : ℚ) / f.2)) ∧
      odd (finset.univ.sum (λ f, (if f = f₁ then f₂.1 else if f = f₂ then f₁.1 else f.1 : ℚ) / f.2).den)) :=
by
  sorry

end swap_numerators_fraction_sum_odd_denominator_l710_710347


namespace rearrange_checkers_case_3_not_rearrange_checkers_case_8_l710_710201

section

def knight_move (p1 p2 : (ℕ × ℕ)) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  ((abs (x1 - x2) = 2 ∧ abs (y1 - y2) = 1) ∨ (abs (x1 - x2) = 1 ∧ abs (y1 - y2) = 2))

def king_move (p1 p2 : (ℕ × ℕ)) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  max (abs (x1 - x2)) (abs (y1 - y2)) = 1

theorem rearrange_checkers_case_3 : ∀ (checkers : list (ℕ × ℕ)),
  card checkers = 9 →
  (∀ p1 p2, p1 ≠ p2 → p1 ∈ checkers → p2 ∈ checkers → knight_move p1 p2 → king_move p1 p2) :=
by sorry

theorem not_rearrange_checkers_case_8 : ∀ (checkers : list (ℕ × ℕ)),
  card checkers = 64 →
  ¬(∀ p1 p2, p1 ≠ p2 → p1 ∈ checkers → p2 ∈ checkers → knight_move p1 p2 → king_move p1 p2) :=
by sorry

end

end rearrange_checkers_case_3_not_rearrange_checkers_case_8_l710_710201


namespace part1_sequence_arithmetic_part2_general_formula_l710_710136

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710136


namespace triangle_vertices_are_valid_l710_710772

variables {Point : Type*} [MetricSpace Point]

-- Define the points A and B that form side AB of the triangle
def A : Point := sorry
def B : Point := sorry

-- Define the condition for side length a
def side_length (a : ℝ) : Prop := dist A B = a

-- Define the midpoint M of the side AB
def M : Point := midpoint ℝ A B

-- Define the circle centered at M with radius sa
def circle (M : Point) (sa : ℝ) : set Point := {C | dist M C = sa}

-- Define the parallel lines at a distance ma from AB
def parallel_lines (A B : Point) (ma : ℝ) : Prop :=
  ∃ line : set Point, 
  (∀ p ∈ line, dist p (line_through_point A B) = ma)

-- Define the intersection points
def intersection_points (M : Point) (sa : ℝ) (ma : ℝ) : finset Point :=
{C | C ∈ circle M sa ∧ C ∈ parallel_lines A B ma}.to_finset

-- Main statement with conditions and the correct answer
theorem triangle_vertices_are_valid (a ma sa : ℝ) (ha : side_length a) :
  intersection_points M sa ma = fintype.of_list [(X : Point), (Z : Point), (S : Point), (R : Point)] (sorry) :=
begin
  sorry
end

end triangle_vertices_are_valid_l710_710772


namespace correct_calculated_value_l710_710499

theorem correct_calculated_value (N : ℕ) (h : N ≠ 0) :
  N * 16 = 2048 * (N / 128) := by 
  sorry

end correct_calculated_value_l710_710499


namespace A_cannot_win_when_k_eq_6_l710_710951

theorem A_cannot_win_when_k_eq_6 : 
  ∀ (grid : infinite_grid hexagon) (k : ℕ), k = 6 → 
  (∀ game_state : grid.state, alternative_turns game_state player_A player_B → 
  (∀ (A_move : grid.state → grid.state) (B_move : grid.state → grid.state), 
  A_move places_counters_and_adjacent → 
  B_move removes_counter → 
  ∀ turns : ℕ, 
  ¬ grid.contains_consecutive_counters k (apply_turns game_state turns))) := 
sorry

end A_cannot_win_when_k_eq_6_l710_710951


namespace prove_remainder_l710_710509

def problem_statement : Prop := (33333332 % 8 = 4)

theorem prove_remainder : problem_statement := 
by
  sorry

end prove_remainder_l710_710509


namespace relatively_prime_probability_l710_710311

theorem relatively_prime_probability (n : ℕ) (h : n = 42) :
  let phi := n * (1 - 1 / 2) * (1 - 1 / 3) * (1 - 1 / 7) in
  (phi / n) = 2 / 7 :=
by
  sorry

end relatively_prime_probability_l710_710311


namespace collinear_points_l710_710565

def finite_set_of_points (M : Finset (Fin₃ × Fin₃)) : Prop :=
  ∀ (p1 p2 : Fin₃ × Fin₃), p1 ∈ M → p2 ∈ M → p1 ≠ p2 → ∃ (p3 : Fin₃ × Fin₃), p3 ∈ M ∧ 
  ((p3.1 * (p2.2 - p1.2) = p3.2 * (p2.1 - p1.1)) → p3 = p1 ∨ p3 = p2)

theorem collinear_points (M : Finset (Fin₃ × Fin₃)) 
  (cond : finite_set_of_points M) : 
  ∃ (a b : Fin₃ × Fin₃), a ∈ M ∧ b ∈ M ∧ ∀ (p : Fin₃ × Fin₃), p ∈ M → 
  (p.1 - a.1) * (b.2 - a.2) = (p.2 - a.2) * (b.1 - a.1) := 
sorry

end collinear_points_l710_710565


namespace maximal_area_of_cross_section_l710_710375

-- Definition of point in 3D
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of the given conditions
def A : Point3D := ⟨4, 0, 0⟩
def B : Point3D := ⟨-2, 2 * real.sqrt 3, 0⟩
def C : Point3D := ⟨-2, -2 * real.sqrt 3, 0⟩
def prism_height_A := 2
def prism_height_B := 4
def prism_height_C := 3

-- Cut plane definition
def cut_plane (p : Point3D) : Prop := 5 * p.x - 3 * p.y + 2 * p.z = 30

-- Vertices of the prism after considering height
def A' : Point3D := ⟨A.x, A.y, prism_height_A⟩
def B' : Point3D := ⟨B.x, B.y, prism_height_B⟩
def C' : Point3D := ⟨C.x, C.y, prism_height_C⟩

-- Intersection points with the plane
def A'' : Point3D := ⟨A.x, A.y, 5⟩
def B'' : Point3D := ⟨B.x, B.y, 26.39⟩
def C'' : Point3D := ⟨C.x, C.y, 22.61⟩

-- Proof that the max area is 104.25
theorem maximal_area_of_cross_section : 
  ∃ (area : ℝ), 
  let A'' := ⟨4, 0, 5⟩;
      B'' := ⟨-2, 2 * real.sqrt 3, 26.39⟩;
      C'' := ⟨-2, -2 * real.sqrt 3, 22.61⟩ in
  5 * A''.x - 3 * A''.y + 2 * A''.z = 30 ∧
  5 * B''.x - 3 * B''.y + 2 * B''.z = 30 ∧
  5 * C''.x - 3 * C''.y + 2 * C''.z = 30 ∧
  area = 104.25 :=
by 
  -- Proof will be completed here
  sorry

end maximal_area_of_cross_section_l710_710375


namespace total_distance_12_hours_l710_710698

noncomputable def car_distance_traveled (hours : ℕ) : ℕ :=
  if hours = 0 then 0 
  else if hours = 1 then 45 
  else 45 + 2 * (hours - 1)

theorem total_distance_12_hours :
  ∑ i in (Finset.range 12).map Finset.Nat.cast, car_distance_traveled (i + 1) = 672 := 
sorry

end total_distance_12_hours_l710_710698


namespace part1_arithmetic_sequence_part2_general_formula_l710_710073

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710073


namespace rectangle_sides_l710_710993

noncomputable def sides_of_rectangle (a b : ℝ) : Prop :=
  a * b = 9 ∧ ∃ (θ : ℝ), θ = 120 ∧ a = real.sqrt (real.mul 3 (real.sqrt 3)) ∧ b = real.sqrt (real.pow 3 3/4)

theorem rectangle_sides : ∃ a b, sides_of_rectangle a b :=
  sorry

end rectangle_sides_l710_710993


namespace measure_of_angle_is_135_l710_710667

noncomputable def degree_measure_of_angle (x : ℝ) : Prop :=
  (x = 3 * (180 - x)) ∧ (2 * x + (180 - x) = 180) -- Combining all conditions

theorem measure_of_angle_is_135 (x : ℝ) (h : degree_measure_of_angle x) : x = 135 :=
by sorry

end measure_of_angle_is_135_l710_710667


namespace parallel_Q1Q2_iff_AB_CD_l710_710340

variable {A B C D P Q1 Q2 : Type} [ConvexQuadrilateral A B C D] [PointInsideQuadrilateral P A B C D]
variables [Angle_eq Q1 B C (Angle_eq A B P)] [Angle_eq Q1 C B (Angle_eq D C P)]
variables [Angle_eq Q2 A D (Angle_eq B A P)] [Angle_eq Q2 D A (Angle_eq C D P)]

theorem parallel_Q1Q2_iff_AB_CD :
  (Parallel Q1 Q2 A B) ↔ (Parallel Q1 Q2 C D) := 
  sorry

end parallel_Q1Q2_iff_AB_CD_l710_710340


namespace b_arithmetic_sequence_general_formula_a_l710_710054

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710054


namespace lcm_is_perfect_square_l710_710196

open Nat

theorem lcm_is_perfect_square (a b : ℕ) : 
  (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0 → ∃ k : ℕ, k^2 = lcm a b :=
by
  sorry

end lcm_is_perfect_square_l710_710196


namespace line_through_point_trisection_l710_710575

theorem line_through_point_trisection :
  ∃ (x y : ℝ), (x, y) = (2, 3) ∧
  ((∃ (a b : ℝ), (a, b) ∈ {(0, 3), (3, 0)} ∧ 3 * x + y - 9 = 0) ∨
  (3 * x + y - 9 = 0)) :=
sorry

end line_through_point_trisection_l710_710575


namespace tetrahedron_faces_equal_l710_710526

-- Define the tetrahedron type with vertices A, B, C, D
structure Tetrahedron (Point : Type) :=
  (A B C D : Point)

noncomputable def area {Point : Type} [MetricSpace Point] 
  (P Q R : Point) : ℝ := sorry  -- Definition for the area of a triangle, to be provided

-- Define that all faces have the same area
def equal_face_areas {Point : Type} [MetricSpace Point] (t : Tetrahedron Point) : Prop :=
  area t.A t.B t.C = area t.A t.B t.D ∧
  area t.A t.B t.C = area t.A t.C t.D ∧
  area t.A t.B t.C = area t.B t.C t.D 

-- Define that all faces of the tetrahedron are congruent
def faces_congruent {Point : Type} [MetricSpace Point] (t : Tetrahedron Point) : Prop :=
  sorry  -- A precise mathematical definition for congruent faces, to be provided

-- The theorem to be proved
theorem tetrahedron_faces_equal {Point : Type} [MetricSpace Point] (t : Tetrahedron Point) :
  equal_face_areas t → faces_congruent t :=
begin
  sorry
end

end tetrahedron_faces_equal_l710_710526


namespace infinite_rel_prime_divisible_pairs_l710_710211

theorem infinite_rel_prime_divisible_pairs :
  ∀ n ≥ 2, let a := 2^n - 1; let b := 2^n + 1 in 
    nat.coprime a b ∧ (a^b + b^a) % (a + b) = 0 := 
by
  intros n hn
  let a := 2^n - 1
  let b := 2^n + 1
  sorry

end infinite_rel_prime_divisible_pairs_l710_710211


namespace sin_2BPC_l710_710207

-- Definitions and conditions
variables (A B C D P : Type)
variables (AB BC CD : ℝ) (angle_APC angle_BPD angle_BPC : ℝ)
axiom points_equally_spaced : AB = BC ∧ BC = CD
axiom cos_angle_APC : cos angle_APC = 5 / 13
axiom cos_angle_BPD : cos angle_BPD = 12 / 13

-- Statement to be proven
theorem sin_2BPC : sin (2 * angle_BPC) = 18 / 169 :=
sorry

end sin_2BPC_l710_710207


namespace shekar_marks_math_l710_710980

theorem shekar_marks_math (M : ℕ) (science : ℕ) (social_studies : ℕ) (english : ℕ) 
(biology : ℕ) (average : ℕ) (num_subjects : ℕ) 
(h_science : science = 65)
(h_social : social_studies = 82)
(h_english : english = 67)
(h_biology : biology = 55)
(h_average : average = 69)
(h_num_subjects : num_subjects = 5) :
M + science + social_studies + english + biology = average * num_subjects →
M = 76 :=
by
  sorry

end shekar_marks_math_l710_710980


namespace min_value_l710_710630

noncomputable def normal_distribution (μ σ2 : ℝ) : Type := sorry

variable {X : ℝ → ℝ}

variable (P : Set ℝ → ℝ)

def normal_X : Prop := X ∼ normal_distribution 10 σ2
def p1 : Prop := P {x | x > 12} = m
def p2 : Prop := P {x | 8 ≤ x ∧ x ≤ 10} = n
def p3 : Prop := m + n = 1 / 2

theorem min_value : normal_X ∧ p1 ∧ p2 ∧ p3 →
  ∃ v, v = (6 + 4 * Real.sqrt 2) ∧ 
  ∀ m n, v ≤ (2 / m) + (1 / n) := 
sorry

end min_value_l710_710630


namespace bn_arithmetic_sequence_an_formula_l710_710031

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l710_710031


namespace ratio_of_female_officers_l710_710578

theorem ratio_of_female_officers (F : ℕ) (total_on_duty : ℕ) 
  (percentage_on_duty : ℚ) (female_on_duty : ℕ) (approx_female_officers : ℕ) :
  percentage_on_duty = 0.17 →
  total_on_duty = 170 →
  approx_female_officers = 500 →
  female_on_duty = (percentage_on_duty * approx_female_officers) →
  female_on_duty / total_on_duty = 1 / 2 :=
begin
  intros h1 h2 h3 h4,
  rw h1 at h4,
  rw h3 at h4,
  norm_num at h4,
  have h5 : female_on_duty = 85, norm_num,
  rw h2,
  simp [female_on_duty],
  sorry
end

end ratio_of_female_officers_l710_710578


namespace inequality_solution_range_l710_710807

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℤ, 6 - 3 * (x : ℝ) < 0 ∧ 2 * (x : ℝ) ≤ a) ∧
  (∃ x1 x2 x3 : ℤ, (x1 = 3 ∧ x2 = 4 ∧ x3 = 5) ∧
   (6 - 3 * (x1 : ℝ) < 0 ∧ 2 * (x1 : ℝ) ≤ a) ∧
   (6 - 3 * (x2 : ℝ) < 0 ∧ 2 * (x2 : ℝ) ≤ a) ∧
   (6 - 3 * (x3 : ℝ) < 0 ∧ 2 * (x3 : ℝ) ≤ a) ∧
   (∀ x : ℤ, (6 - 3 * (x : ℝ) < 0 ∧ 2 * (x : ℝ) ≤ a) → 
     (x = 3 ∨ x = 4 ∨ x = 5)))
  → 10 ≤ a ∧ a < 12 :=
sorry

end inequality_solution_range_l710_710807


namespace ellie_sam_in_photo_probability_l710_710412

-- Definitions of the conditions
def lap_time_ellie := 120 -- seconds
def lap_time_sam := 75 -- seconds
def start_time := 10 * 60 -- 10 minutes in seconds
def photo_duration := 60 -- 1 minute in seconds
def photo_section := 1 / 3 -- fraction of the track captured in the photo

-- The probability that both Ellie and Sam are in the photo section between 10 to 11 minutes
theorem ellie_sam_in_photo_probability :
  let ellie_time := start_time;
  let sam_time := start_time;
  let ellie_range := (ellie_time - (photo_section * lap_time_ellie / 2), ellie_time + (photo_section * lap_time_ellie / 2));
  let sam_range := (sam_time - (photo_section * lap_time_sam / 2), sam_time + (photo_section * lap_time_sam / 2));
  let overlap_start := max ellie_range.1 sam_range.1;
  let overlap_end := min ellie_range.2 sam_range.2;
  let overlap_duration := max 0 (overlap_end - overlap_start);
  let overlap_probability := overlap_duration / photo_duration;
  overlap_probability = 5 / 12 :=
by
  sorry

end ellie_sam_in_photo_probability_l710_710412


namespace lcm_is_perfect_square_l710_710192

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : ∃ k : ℕ, k^2 = Nat.lcm a b :=
by
  sorry

end lcm_is_perfect_square_l710_710192


namespace part1_arithmetic_sequence_part2_general_formula_l710_710101

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l710_710101


namespace bn_is_arithmetic_an_general_formula_l710_710113

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710113


namespace b_arithmetic_a_formula_l710_710008

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l710_710008


namespace triangle_area_base2t_height3t1_l710_710665

theorem triangle_area_base2t_height3t1 (t : ℝ) (h : t = 4) : 
  let base := 2 * t,
      height := 3 * t + 1,
      area := (1 / 2) * base * height in
  area = 52 := by
  let base := 2 * t
  let height := 3 * t + 1
  let area := (1 / 2) * base * height
  have h0 : base = 2 * t := rfl
  have h1 : height = 3 * t + 1 := rfl
  have h2 : area = (1 / 2) * (2 * t) * (3 * t + 1) := rfl
  rw [h] at *
  have h3 : 2 * 4 = 8 := by norm_num
  have h4 : 3 * 4 + 1 = 13 := by norm_num
  have h5 : (1 / 2) * 8 * 13 = 52 := by norm_num
  have result : area = 52 := by rw [h2, h3, h4, h5]
  exact result

end triangle_area_base2t_height3t1_l710_710665


namespace smallest_gcd_qr_l710_710886

theorem smallest_gcd_qr {p q r : ℕ} (hpq : Nat.gcd p q = 300) (hpr : Nat.gcd p r = 450) : 
  ∃ (g : ℕ), g = Nat.gcd q r ∧ g = 150 :=
by
  sorry

end smallest_gcd_qr_l710_710886


namespace max_path_length_of_fly_l710_710363

structure RectangularPrism :=
(length : ℝ)
(width : ℝ)
(height : ℝ)

def fly_trapped (start : ℝ × ℝ × ℝ) (corners : list (ℝ × ℝ × ℝ)) : Prop :=
start ∈ corners ∧ list.nodup corners ∧ corners.length = 8 

noncomputable def maximum_path_length (P : RectangularPrism) : ℝ :=
4 * real.sqrt 6 + 4 * real.sqrt 5

theorem max_path_length_of_fly (P : RectangularPrism)
  (h₁ : P.length = 2)
  (h₂ : P.width = 1)
  (h₃ : P.height = 1)
  (h₄ : ∃ (start : ℝ × ℝ × ℝ) (corners : list (ℝ × ℝ × ℝ)), fly_trapped start corners) :
  maximum_path_length P = 4 * real.sqrt 6 + 4 * real.sqrt 5 := 
sorry

end max_path_length_of_fly_l710_710363


namespace sequence_bn_arithmetic_and_an_formula_l710_710126

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l710_710126


namespace yellow_tiles_count_l710_710937

theorem yellow_tiles_count
  (total_tiles : ℕ)
  (yellow_tiles : ℕ)
  (blue_tiles : ℕ)
  (purple_tiles : ℕ)
  (white_tiles : ℕ)
  (h1 : total_tiles = 20)
  (h2 : blue_tiles = yellow_tiles + 1)
  (h3 : purple_tiles = 6)
  (h4 : white_tiles = 7)
  (h5 : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  yellow_tiles = 3 :=
by sorry

end yellow_tiles_count_l710_710937


namespace symmetric_point_proof_l710_710927

def Point3D := (ℝ × ℝ × ℝ)

def symmetric_point_yOz (p : Point3D) : Point3D :=
  let (x, y, z) := p
  (-x, y, z)

theorem symmetric_point_proof :
  symmetric_point_yOz (1, -2, 3) = (-1, -2, 3) :=
by
  sorry

end symmetric_point_proof_l710_710927


namespace total_apples_l710_710653

theorem total_apples (A B C : ℕ) (h1 : A + B = 11) (h2 : B + C = 18) (h3 : A + C = 19) : A + B + C = 24 :=  
by
  -- Skip the proof
  sorry

end total_apples_l710_710653


namespace sequence_equality_l710_710945
open Nat

def is_increasing_sequence (a : ℕ → ℕ) := ∀ n : ℕ, n ≥ 1 → a n < a (n + 1)

def sequence_property_1 (a : ℕ → ℕ) := ∀ n : ℕ, n ≥ 1 → a (2 * n) = a n + n

def sequence_property_2 (a : ℕ → ℕ) := ∀ n : ℕ, n ≥ 1 → Prime (a n) → Prime n

theorem sequence_equality (a : ℕ → ℕ) 
  (H_inc : is_increasing_sequence a)
  (H_prop1 : sequence_property_1 a)
  (H_prop2 : sequence_property_2 a) : 
  ∀ n : ℕ, n ≥ 1 → a n = n :=
begin
  sorry
end

end sequence_equality_l710_710945


namespace inequality_solution_l710_710988

theorem inequality_solution (x : ℝ) : 
  (∃ (y : ℝ), y = 1 / (3 ^ x) ∧ y * (y - 2) < 15) ↔ x > - (Real.log 5 / Real.log 3) :=
by 
    sorry

end inequality_solution_l710_710988


namespace cosine_not_equal_l710_710488

noncomputable def vector_between (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2, Q.3 - P.3)

noncomputable def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def cosine_of_angle (v w : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v w / (magnitude v * magnitude w)

def point_A : ℝ × ℝ × ℝ := (0, 1, 0)
def point_B : ℝ × ℝ × ℝ := (2, 2, 0)
def point_C : ℝ × ℝ × ℝ := (-1, 3, 1)

noncomputable def vector_AB := vector_between point_A point_B
noncomputable def vector_BC := vector_between point_B point_C

theorem cosine_not_equal :
  ¬ (cosine_of_angle vector_AB vector_BC = sqrt 55 / 11) :=
by {
  sorry
}

end cosine_not_equal_l710_710488


namespace quadratic_roots_l710_710467

theorem quadratic_roots (a c : ℝ) (h₀ : a ≠ 0)
  (h₁ : polynomial.eval (-1) (polynomial.C a * polynomial.X^2 + polynomial.C (-2 * a) * polynomial.X + polynomial.C c) = 0) :
  (polynomial.C a * polynomial.X^2 + polynomial.C (-2 * a) * polynomial.X + polynomial.C c).roots = { -1, 3 } :=
by
  sorry

end quadratic_roots_l710_710467


namespace not_monotonic_over_entire_real_line_l710_710663

noncomputable def counterexample_function_1 : ℝ → ℝ := abs

noncomputable def counterexample_function_2 : ℝ → ℝ 
| x := if x < 0 then x else x - 1

theorem not_monotonic_over_entire_real_line 
  (f : ℝ → ℝ) 
  (H : ∀ a : ℝ, ∃ b : ℝ, b > a ∧ (monotone_on f (set.Ioo a b) ∨ antitone_on f (set.Ioo a b))) : 
  ¬(monotone f ∨ antitone f) :=
begin
  -- the proof would go here
  -- however the statement of the theorem as requested
  -- covers the problem provided
  sorry
end

end not_monotonic_over_entire_real_line_l710_710663


namespace parabola_equation_through_P_l710_710798

def is_parabola_with_vertex_origin (f : ℝ → ℝ → Prop) : Prop :=
∀ (x y : ℝ), f x y ↔ (y^2 = -x ∨ x^2 = -8y)

theorem parabola_equation_through_P : 
  ∃ (f : ℝ → ℝ → Prop), 
    (∀ x y, f x y ↔ ((y = 0) ∧ (x = 0))) ∧
    (∃ x y, f x y ∧ y = 0 ∨  y ≠ 0) ∧ 
    (f (-4) (-2)) := 
begin
  sorry
end

end parabola_equation_through_P_l710_710798


namespace tan_product_identity_l710_710342

theorem tan_product_identity : (1 + Real.tan (Real.pi / 180 * 17)) * (1 + Real.tan (Real.pi / 180 * 28)) = 2 := by
  sorry

end tan_product_identity_l710_710342


namespace lcm_of_two_numbers_l710_710632

theorem lcm_of_two_numbers (a b HCF : ℕ) (h1 : a / b = 4 / 5) (h2 : nat.gcd a b = 4) : nat.lcm a b = 80 :=
by 
  sorry

end lcm_of_two_numbers_l710_710632


namespace bn_is_arithmetic_an_general_formula_l710_710115

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710115


namespace arithmetic_geometric_sequence_relation_l710_710836

theorem arithmetic_geometric_sequence_relation 
  (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (hA : ∀ n: ℕ, a (n + 1) - a n = a 1) 
  (hG : ∀ n: ℕ, b (n + 1) / b n = b 1) 
  (h1 : a 1 = b 1) 
  (h11 : a 11 = b 11) 
  (h_pos : 0 < a 1 ∧ 0 < a 11 ∧ 0 < b 11 ∧ 0 < b 1) :
  a 6 ≥ b 6 := sorry

end arithmetic_geometric_sequence_relation_l710_710836


namespace power_function_value_at_neg2_l710_710256

theorem power_function_value_at_neg2 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x : ℝ, f x = x^a)
  (h2 : f 2 = 1 / 4) 
  : f (-2) = 1 / 4 := by
  sorry

end power_function_value_at_neg2_l710_710256


namespace sector_ratio_l710_710204

-- Definitions from the conditions
variables (O : Type) (circle : O → Prop) (A B E F : O) 
variables {diameter : A ≠ B}
variables (on_same_side : ∀ E F ∈ circle O, E ≠ F → (∃ A B, A ≠ B ∧ diameter) ∧ 
                                      (∃ O, angle A O E = 60) ∧ (∃ O, angle F O B = 90))

-- The statement to prove the ratio of the area of sector EOF to the area of the circle is 1/12
theorem sector_ratio (circle_satisfy : circle O A ∧ circle O B ∧ circle O E ∧ circle O F) : 
  (EOF_area / circle_area) = 1 / 12 := 
sorry

end sector_ratio_l710_710204


namespace ratio_a5_b5_l710_710875

-- Definitions of arithmetic sequences
variable {a b : ℕ → ℕ} -- sequences a_n and b_n are functions ℕ → ℕ

-- Sum of the first n terms of sequences
def S (n : ℕ) : ℕ := (n * (a 1 + a n)) / 2
def T (n : ℕ) : ℕ := (n * (b 1 + b n)) / 2

-- Given ratio condition
axiom sum_ratio (n : ℕ) : (S n) / (T n) = (7 * n + 2) / (n + 3)

theorem ratio_a5_b5 : (a 5) / (b 5) = 65 / 12 := by
  -- Proof is to be completed
  sorry

end ratio_a5_b5_l710_710875


namespace three_power_variations_distinct_values_l710_710383

theorem three_power_variations_distinct_values :
  let orig_val := 3 ^ (3 ^ (3 ^ 3)) in
  let expr_list :=
    [3 ^ (3 ^ (3 ^ 3)),
     3 ^ ((3 ^ 3) ^ 3),
     ((3 ^ 3) ^ 3) ^ 3,
     (3 ^ (3 ^ 3)) ^ 3,
     (3 ^ 3) ^ (3 ^ 3)] in
  (orig_val = 3 ^ 27) → (list.length (list.nodup expr_list) = 3) :=
begin
  sorry
end

end three_power_variations_distinct_values_l710_710383


namespace median_of_donations_l710_710378

theorem median_of_donations : 
  let donations := [5, 10, 6, 6, 7, 8, 9];
  List.median donations = 7 := by
  sorry

end median_of_donations_l710_710378


namespace value_sum_l710_710753

def v (x : ℝ) : ℝ := -x + 2 * Real.sin (Real.pi * x / 2)

theorem value_sum :
  v (-3) + v (-1) + v (1) + v (3) = -4 :=
by
  sorry

end value_sum_l710_710753


namespace tan_sum_formula_l710_710801

theorem tan_sum_formula {A B : ℝ} (hA : A = 55) (hB : B = 65) (h1 : Real.tan (A + B) = Real.tan 120) 
    (h2 : Real.tan 120 = -Real.sqrt 3) :
    Real.tan 55 + Real.tan 65 - Real.sqrt 3 * Real.tan 55 * Real.tan 65 = -Real.sqrt 3 := 
by
  sorry

end tan_sum_formula_l710_710801


namespace RelativelyPrimeProbability_l710_710297

def relatively_prime_probability_42 : Rat :=
  let n := 42
  let total := n
  let rel_prime_count := total - (21 + 14 + 6 - 7 - 3 - 2 + 1)
  let probability := (rel_prime_count : Rat) / total
  probability

theorem RelativelyPrimeProbability : relatively_prime_probability_42 = 2 / 7 :=
sorry

end RelativelyPrimeProbability_l710_710297


namespace probability_A_not_first_B_not_last_l710_710760

-- Define the conditions
def athletes : Finset ℕ := {1, 2, 3, 4, 5, 6} -- Represents the 6 athletes
def team_size : ℕ := 4
def total_ways : ℕ := (athletes.card.choose team_size)
def ways_A_first : ℕ := (athletes.erase 1).card.choose 3
def ways_B_last : ℕ := (athletes.erase 2).card.choose 3
def overlap_A_first_B_last : ℕ := (athletes.erase 1).erase 2.card.choose 2

noncomputable def desirable_selections : ℕ :=
  total_ways - ways_A_first - ways_B_last + overlap_A_first_B_last

noncomputable def probability : ℚ :=
  desirable_selections / total_ways

theorem probability_A_not_first_B_not_last : probability = 7 / 10 :=
  by sorry

end probability_A_not_first_B_not_last_l710_710760


namespace interval_intersect_exists_l710_710564

noncomputable def T (a x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ a then x + (1 - a) else if a < x ∧ x ≤ 1 then x - a else 0

theorem interval_intersect_exists (a : ℝ) (h : 0 < a ∧ a < 1) (J : set ℝ) (hJ : ∀ x, J x → 0 < x ∧ x ≤ 1) :
  ∃ (n : ℕ) (hn : n > 0), (∃ x, J x ∧ T (nat.iterate (λ y, T a y) n x) x) :=
sorry

end interval_intersect_exists_l710_710564


namespace geometric_sequence_general_term_l710_710922

theorem geometric_sequence_general_term (a : ℕ → ℕ) (q : ℕ) (h_q : q = 4) (h_sum : a 0 + a 1 + a 2 = 21)
  (h_geo : ∀ n, a (n + 1) = a n * q) : ∀ n, a n = 4 ^ n :=
by {
  sorry
}

end geometric_sequence_general_term_l710_710922


namespace problem_conditions_l710_710089

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l710_710089


namespace incorrect_statement_C_l710_710325

theorem incorrect_statement_C :
  ∀ A B C D : Prop,
  (A ↔ ("The intersection line of a plane with two parallel planes is parallel")) →
  (B ↔ ("Two planes parallel to the same plane are parallel to each other")) →
  (C ↔ ("Two planes parallel to the same line are parallel to each other")) →
  (D ↔ ("If a line intersects one of two parallel planes, it must intersect the other")) →
  ¬ C :=
by
  intros A B C D hA hB hC hD,
  -- incorrect statement C is proven by example in the solution
  let example := "For example, a vertical flagpole is parallel to all vertical walls, but not all vertical walls are necessarily parallel to each other.",
  sorry

end incorrect_statement_C_l710_710325


namespace distance_upstream_is_14_l710_710271

variable (speed_boat : ℝ) (speed_stream : ℝ) (distance_downstream : ℝ)

def effective_speed_downstream := speed_boat + speed_stream
def effective_speed_upstream := speed_boat - speed_stream
def downstream_time := distance_downstream / effective_speed_downstream
def upstream_distance := effective_speed_upstream * downstream_time

theorem distance_upstream_is_14
  (h1 : speed_boat = 20) 
  (h2 : speed_stream = 6)
  (h3 : distance_downstream = 26)
  : upstream_distance speed_boat speed_stream distance_downstream = 14 :=
by
  unfold effective_speed_downstream effective_speed_upstream downstream_time upstream_distance
  rw [h1, h2, h3]
  norm_num
  sorry

end distance_upstream_is_14_l710_710271


namespace valid_schedule_count_l710_710498

theorem valid_schedule_count :
  ∃ (valid_schedules : Finset (Fin 8 → Option (Fin 4))),
    valid_schedules.card = 488 ∧
    (∀ (schedule : Fin 8 → Option (Fin 4)), schedule ∈ valid_schedules →
      (∀ i : Fin 7, schedule i ≠ none ∧ schedule (i + 1) ≠ schedule i) ∧
      schedule 4 = none) :=
sorry

end valid_schedule_count_l710_710498


namespace kimberly_loan_l710_710938

theorem kimberly_loan (t : ℕ) (h : 1.06 ^ t > 2) : t = 12 :=
begin
  sorry,
end

end kimberly_loan_l710_710938


namespace polynomials_equal_l710_710170

noncomputable def P : ℝ → ℝ := sorry -- assume P is a nonconstant polynomial
noncomputable def Q : ℝ → ℝ := sorry -- assume Q is a nonconstant polynomial

axiom floor_eq_for_all_y (y : ℝ) : ⌊P y⌋ = ⌊Q y⌋

theorem polynomials_equal (x : ℝ) : P x = Q x :=
by
  sorry

end polynomials_equal_l710_710170


namespace min_value_div_n_and_m_eq_2_5_l710_710248

noncomputable def a_n (n : ℕ) : ℝ := 2 * 3^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := 3^n - 1

theorem min_value_div_n_and_m_eq_2_5 :
  ∃ (n m : ℕ), n > 0 ∧ m > 0 ∧
  (log 3 (1/2 * a_n n * (S_n (4 * m) + 1)) = 9) ∧
  (1 / n + 4 / m = 2.5) :=
sorry

end min_value_div_n_and_m_eq_2_5_l710_710248


namespace maximize_tables_eqn_l710_710281

theorem maximize_tables_eqn :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 12 → 400 * x = 20 * (12 - x) * 4 :=
by
  sorry

end maximize_tables_eqn_l710_710281


namespace line_symmetric_eq_l710_710844

theorem line_symmetric_eq (l : ℝ → Prop) : 
  (∀ x y, (2 * x - 3 * y + 4 = 0) ↔ l (2 - x) y) →  
  (∀ x y, l x y ↔ (2 * x + 3 * y - 8 = 0)) :=
by
  sorry

end line_symmetric_eq_l710_710844


namespace b_arithmetic_sequence_general_formula_a_l710_710057

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l710_710057


namespace problem1_solution_l710_710600

theorem problem1_solution : ∀ x : ℝ, x^2 - 6 * x + 9 = (5 - 2 * x)^2 → (x = 8/3 ∨ x = 2) :=
sorry

end problem1_solution_l710_710600


namespace percentage_of_students_is_approximately_80_09_l710_710919

noncomputable def percentage_of_second_year_students
    (students_numeric_methods : ℕ)
    (students_automatic_control : ℕ)
    (students_both : ℕ)
    (total_students : ℕ) : ℚ :=
  let total_second_year_students := students_numeric_methods + students_automatic_control - students_both
  in (total_second_year_students * 100 : ℚ) / total_students

theorem percentage_of_students_is_approximately_80_09
    (students_numeric_methods : ℕ := 250)
    (students_automatic_control : ℕ := 423)
    (students_both : ℕ := 134)
    (total_students : ℕ := 673) :
  percentage_of_second_year_students students_numeric_methods students_automatic_control students_both total_students ≈ 80.09 := sorry

end percentage_of_students_is_approximately_80_09_l710_710919


namespace swim_time_against_current_l710_710728

theorem swim_time_against_current (
  (speed_still_water : ℝ) (speed_current : ℝ) (time_with_current : ℝ) : 
  speed_still_water = 16 → 
  speed_current = 8 → 
  time_with_current = 1.5 → 
  (let effective_speed_against_current := speed_still_water - speed_current,
       speed_with_current := speed_still_water + speed_current,
       distance := speed_with_current * time_with_current,
       time_against_current := distance / effective_speed_against_current
  in time_against_current = 4.5)) :=
begin
  intros h1 h2 h3,
  have h4 : effective_speed_against_current := speed_still_water - speed_current,
  have h5 : speed_with_current := speed_still_water + speed_current,
  have h6 : distance := speed_with_current * time_with_current,
  have h7 : time_against_current := distance / effective_speed_against_current,
  sorry,
end

end swim_time_against_current_l710_710728


namespace bn_is_arithmetic_an_general_formula_l710_710112

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l710_710112


namespace max_area_quadrilateral_l710_710670

theorem max_area_quadrilateral (a b c d : ℝ) (h_eq1 : a = 1) (h_eq2 : b = 4) (h_eq3 : c = 7) (h_eq4 : d = 8)
  (h_cyclic : (a * a + d * d = b * b + c * c)) : ∃ (A : ℝ), A = 18 :=
by
  have s := (a + b + c + d) / 2
  have area_max := (s - a) * (s - b) * (s - c) * (s - d)
  exists sqrt area_max
  sorry

end max_area_quadrilateral_l710_710670


namespace number_of_sandwiches_l710_710978

theorem number_of_sandwiches (choices_per_topping : ℕ) (number_of_toppings : ℕ) (choices_per_pattie : ℕ) 
  (h1 : choices_per_topping = 2)
  (h2 : number_of_toppings = 9)
  (h3 : choices_per_pattie = 2) : 
  choices_per_topping ^ number_of_toppings * choices_per_pattie = 1024 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end number_of_sandwiches_l710_710978


namespace smallest_number_of_blue_chips_l710_710899

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n ∈ finset.range (p - 1) \ {0}, p % n ≠ 0

theorem smallest_number_of_blue_chips : ∃ b : ℕ, b = 1 ∧ (∃ r p : ℕ, r + b = 49 ∧ r = b + p ∧ is_prime p) :=
by
  sorry

end smallest_number_of_blue_chips_l710_710899


namespace fixed_point_of_log_function_l710_710617

theorem fixed_point_of_log_function (a : ℝ) (h_a : a > 0 ∧ a ≠ 1) :
  (∃ x y : ℝ, y = log a (2 * x - 3) + 1 ∧ x = 2 ∧ y = 1) :=
begin
  use 2,
  use 1,
  split,
  { calc
      1 = log a 1 + 1 : by rw log_one
      ... = log a (2 * 2 - 3) + 1 : by norm_num },
  split,
  { refl },
  { refl }
end

end fixed_point_of_log_function_l710_710617


namespace b_seq_arithmetic_a_seq_formula_l710_710152

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l710_710152


namespace sum_b_n_eq_l710_710543

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {S : ℕ → ℝ}

def a_n (n : ℕ) [Fact (0 < n)] : ℝ :=
  (∑ i in Finset.range n, (i+1)/(n+1))

def b_n (n : ℕ) [Fact (0 < n)] : ℝ :=
  1 / (a_n n * a_n (n+1))

def S_n (n : ℕ) [Fact (0 < n)] : ℝ :=
  ∑ i in Finset.range n, b_n (i+1)

theorem sum_b_n_eq (n : ℕ) [Fact (0 < n)] : S_n n = (4 * n) / (n + 1) := 
sorry

end sum_b_n_eq_l710_710543


namespace range_of_function_l710_710269

theorem range_of_function :
  let f (x : ℝ) := 1 - x - 9 / x in
  (∀ y, ∃ x, f x = y) ↔ (y ∈ (Set.Iic (-5) ∪ Set.Ici 7)) :=
by
  sorry

end range_of_function_l710_710269


namespace measure_using_hourglasses_l710_710285

theorem measure_using_hourglasses (p q n : ℕ) (h_coprime : Nat.coprime p q) (h_p_gt_0 : p > 0) (h_q_gt_0 : q > 0) (h_n_ge_half_pq : n ≥ p * q / 2) : 
  ∃ (measure_possible : Prop), measure_possible :=
by
  let measure_possible := ∃ (k : ℕ), (0 ≤ k ∧ k ≤ n / p ∧ k * p ≡ n [MOD q])
  exact Exists.intro measure_possible sorry

end measure_using_hourglasses_l710_710285


namespace buyers_cake_and_muffin_l710_710718

theorem buyers_cake_and_muffin (total_buyers cake_buyers muffin_buyers neither_prob : ℕ) :
  total_buyers = 100 →
  cake_buyers = 50 →
  muffin_buyers = 40 →
  neither_prob = 26 →
  (cake_buyers + muffin_buyers - neither_prob) = 74 →
  90 - cake_buyers - muffin_buyers = neither_prob :=
by
  sorry

end buyers_cake_and_muffin_l710_710718


namespace part1_sequence_arithmetic_part2_general_formula_l710_710147

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l710_710147


namespace hyperbola_eccentricity_l710_710603

theorem hyperbola_eccentricity (P: ℝ × ℝ) (a b c: ℝ) (F1 F2: ℝ × ℝ) 
  (h1: a > 0) (h2: b > 0) 
  (h3: P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (h4: P.1^2 + P.2^2 = a^2 + b^2)
  (h5: ∠P F2 F1 = 2 * ∠P F1 F2)
  (h6: c^2 = a^2 + b^2):
  hyperbola_eccentricity = sqrt 3 + 1 := 
sorry

end hyperbola_eccentricity_l710_710603


namespace problem1_problem2_l710_710763

-- Problem 1
theorem problem1 (a : ℝ) : 2 * a + 3 * a - 4 * a = a :=
by sorry

-- Problem 2
theorem problem2 : 
  - (1 : ℝ) ^ 2022 + (27 / 4) * (- (1 / 3) - 1) / ((-3) ^ 2) + abs (-1) = -1 :=
by sorry

end problem1_problem2_l710_710763


namespace maximize_abs_sum_solution_problem_l710_710599

theorem maximize_abs_sum_solution :
ℤ → ℤ → Ennreal := sorry

theorem problem :
  (∃ (x y : ℤ), 6 * x^2 + 5 * x * y + y^2 = 6 * x + 2 * y + 7 ∧ 
  x = -8 ∧ y = 25 ∧ (maximize_abs_sum_solution x y = 33)) := sorry

end maximize_abs_sum_solution_problem_l710_710599


namespace smaller_octagon_area_ratio_l710_710944

theorem smaller_octagon_area_ratio (ABCDEFGH : Octagon) 
(midpoints : Midpoints ABCDEFGH) 
(smaller_octagon_IKQMOPIP : SmallerOctagon ABCDEFGH midpoints) : 
  ratio_areas smaller_octagon_IKQMOPIP ABCDEFGH = 1 / 8 := 
sorry

end smaller_octagon_area_ratio_l710_710944


namespace find_values_of_a_and_b_l710_710454

theorem find_values_of_a_and_b : 
  ∃ a b : ℝ, a = -3 ∧ b = 3 ∧ (∀ x : ℝ, x < -1 → a * x > b) ∧ (∀ y : ℝ, y^2 + 3 * y + b > 0) :=
by {
  use [-3, 3],
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  { sorry },  -- Proof for ∀ x : ℝ, x < -1 → -3 * x > 3
  { sorry }   -- Proof for ∀ y : ℝ, y^2 + 3 * y + 3 > 0
}

end find_values_of_a_and_b_l710_710454


namespace heptagon_triangulation_l710_710934

theorem heptagon_triangulation (heptagon_vertices: ℕ) (internal_points: ℕ) (no_three_collinear: Prop) 
  (heptagon_vertices = 7) (internal_points = 10) : 
  ∃ T : ℕ, T = 25 := 
by {
  sorry
}

end heptagon_triangulation_l710_710934


namespace exists_large_area_triangle_l710_710434

variable (A B C X Y : Point)
variable (h : ∀ P Q R : Point, 2 ≤ area P Q R)

theorem exists_large_area_triangle :
  ∃ P Q R : Point, P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ 3 ≤ area P Q R :=
sorry

end exists_large_area_triangle_l710_710434


namespace orangeade_price_per_glass_l710_710968

theorem orangeade_price_per_glass (O : ℝ) (W : ℝ) (P : ℝ) (price_1_day : ℝ) 
    (h1 : W = O) (h2 : price_1_day = 0.30) (revenue_equal : 2 * O * price_1_day = 3 * O * P) :
  P = 0.20 :=
by
  sorry

end orangeade_price_per_glass_l710_710968


namespace find_all_friendly_pairs_l710_710804

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n, n ∣ p → n = 1 ∨ n = p

def phi (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ x, nat.coprime x n).card

def f (n : ℕ) : ℕ := 
  nat.find (λ m, m > n ∧ ¬ nat.coprime m n)

def friendly_pair (n m : ℕ) : Prop := 
  f(n) = m ∧ phi(m) = n

theorem find_all_friendly_pairs : ∀ n m : ℕ, friendly_pair n m → (n = 2 ∧ m = 4) :=
by
  sorry

end find_all_friendly_pairs_l710_710804


namespace ram_total_distance_l710_710588

noncomputable def total_distance 
  (speed1 speed2 time1 total_time : ℝ) 
  (h_speed1 : speed1 = 20) 
  (h_speed2 : speed2 = 70)
  (h_time1 : time1 = 3.2)
  (h_total_time : total_time = 8) 
  : ℝ := 
  speed1 * time1 + speed2 * (total_time - time1)

theorem ram_total_distance
  (speed1 speed2 time1 total_time : ℝ)
  (h_speed1 : speed1 = 20)
  (h_speed2 : speed2 = 70)
  (h_time1 : time1 = 3.2)
  (h_total_time : total_time = 8)
  : total_distance speed1 speed2 time1 total_time h_speed1 h_speed2 h_time1 h_total_time = 400 :=
  sorry

end ram_total_distance_l710_710588


namespace length_of_symmetric_chord_l710_710483

noncomputable theory

-- Define the parabola y = -x^2 + 3
def parabola (x : ℝ) : ℝ :=
  -x^2 + 3

-- Given that points A and B are symmetric about the line x + y = 0
-- Define the symmetric condition
def symmetric (A B : ℝ × ℝ) : Prop :=
  A.1 + A.2 = 0 ∧ B.1 + B.2 = 0 ∧ A.1 = -B.1 ∧ A.2 = -B.2

-- Define line passing through points a and b.
def line_segment_length (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem length_of_symmetric_chord :
  ∀ A B : ℝ × ℝ, symmetric A B ∧ A.2 = parabola A.1 ∧ B.2 = parabola B.1 → line_segment_length A B = 3 * real.sqrt 2 :=
sorry

end length_of_symmetric_chord_l710_710483


namespace faith_change_and_payment_proof_l710_710416

noncomputable def calculate_change_and_payment (flour cake_stand baking_powder frosting_nozzles : ℚ)
                                                (discount_rate tax_rate : ℚ)
                                                (amount_paid : ℚ) :
                                                ℚ × ℚ :=
let before_discount := flour + cake_stand + baking_powder + frosting_nozzles
    in
let cake_stand_discounted := cake_stand * (1 - discount_rate)
    in
let total_after_discount := flour + cake_stand_discounted + baking_powder + frosting_nozzles
    in
let tax := total_after_discount * tax_rate
    in
let final_total := total_after_discount + tax
    in
let total_change := amount_paid - final_total
    in
(total_change, (if total_change < 0 then -total_change else 0))

theorem faith_change_and_payment_proof :
  calculate_change_and_payment 5.25 28.75 3.25 15.33 0.10 0.08 100 = (46.32, 0) :=
sorry

end faith_change_and_payment_proof_l710_710416
