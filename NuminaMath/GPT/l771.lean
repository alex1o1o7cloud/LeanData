import Mathlib
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.Combinatorics.CombinatorialIdentity
import Mathlib.Algebra.Field
import Mathlib.Algebra.Floor
import Mathlib.Algebra.Group.Definitions
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Vector
import Mathlib.Analysis.Calculus.Curve
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Configurations
import Mathlib.Combinatorics.SimpleGraph.Connectivity
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Probability.Mean
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Theory
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace laptop_repairs_l771_771336

noncomputable def phoneRepairCost : ℕ := 11
noncomputable def laptopRepairCost : ℕ := 15
noncomputable def computerRepairCost : ℕ := 18
noncomputable def phoneRepairs : ℕ := 5
noncomputable def computerRepairs : ℕ := 2
noncomputable def totalEarnings : ℕ := 121

theorem laptop_repairs :
  ∃ (L : ℕ), 
    (phoneRepairs * phoneRepairCost) + 
    (computerRepairs * computerRepairCost) + 
    (L * laptopRepairCost) = totalEarnings ∧
    L = 2 := 
begin
  sorry
end

end laptop_repairs_l771_771336


namespace tan_of_frac_sinx_cot_sinx_cos_simp_sinx_cosx_l771_771433

-- Part (1): Prove tan x = 2 given the condition
theorem tan_of_frac_sinx_cot_sinx:
  (∀ x: Real, ((sin x + cos x) / (sin x - cos x) = 3) → (tan x = 2)) :=
begin
  sorry
end

-- Part (2): Prove the simplified value of cos(6π - x) · sqrt((1 + sin x) / (1 - sin x))
theorem cos_simp_sinx_cosx:
  (∀ x: Real, ((sin x + cos x) / (sin x - cos x) = 3) → x ∈ set.Icc (π) (3 * π) → 
    (cos (6 * π - x) * sqrt ((1 + sin x) / (1 - sin x)) = -1 + (2 * sqrt (5)) / 5)) :=
begin
  sorry
end

end tan_of_frac_sinx_cot_sinx_cos_simp_sinx_cosx_l771_771433


namespace evaluate_neg_64_exp_4_over_3_l771_771374

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l771_771374


namespace range_of_m_is_empty_l771_771541

-- Define a circle in the Cartesian coordinate system
def circle (x y m : ℝ) := (x + 2)^2 + (y - m)^2 = 3

-- Define the conditions for chord AB and midpoint G
def chord_condition (x y m : ℝ) : Prop :=
  let G_O_length := Real.sqrt (x^2 + y^2) in
  ∃ (AB G_O_length_half: ℝ),
    AB = 2 * G_O_length_half ∧
    (2 * Real.sqrt (3 - (x + 2)^2 - (y - m)^2) = 2 * G_O_length)

-- Effective condition to be proved: range of m under the specified constraints
theorem range_of_m_is_empty :
  ∀ m, ¬ (∃ x y, circle x y m ∧ chord_condition x y m) := sorry

end range_of_m_is_empty_l771_771541


namespace evaluate_pow_l771_771361

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l771_771361


namespace length_CE_l771_771971

noncomputable def Point := Type

-- Assume the given points and the lines.
variables (A E C D B : Point)

-- Assume the given lengths as constants.
variables (AB CD AE : ℝ)
variables (hAB : AB = 4) (hCD : CD = 8) (hAE : AE = 5)

-- Define perpendicularity condition.
structure Perpendicular (x y : Point → Prop) : Prop :=
(perp : ∀ z, x z → y z → false)

-- Assume perpendicular relationships.
variables (p1 : Perpendicular (λ P, ∃ x, P = C ∧ x = D) (λ P, ∃ x, P = A ∧ x = E))
variables (p2 : Perpendicular (λ P, ∃ x, P = A ∧ x = B) (λ P, ∃ x, P = C ∧ x = E))

-- Define the length of segment CE and the goal theorem.
def CE : ℝ := sorry  -- This is just a placeholder for definition.

theorem length_CE : CE = 10 :=
by
  -- You would normally provide a proof here, but for the purposes of this problem, we just state the theorem.
  sorry

end length_CE_l771_771971


namespace sin_A_value_AC_value_l771_771474

variables {A B C : ℝ}
variables {S : ℝ}

/-- Given the area of a triangle and a dot product condition, prove sin(A) equals a given value. -/
theorem sin_A_value (h₁: 3 * (A * B) * (A * C) = 2 * S) (h₂: S = 1/2 * A * B * sin A) : sin A = (3 * sqrt 10) / 10 :=
sorry

/-- Given the values of C and dot product condition, prove AC equals a given value. -/
theorem AC_value (h₁: C = π / 4) (h₂: (A * B) * (A * C) = 16) (h₃: sin A = (3 * sqrt 10) / 10) : A * C = 8 :=
sorry

end sin_A_value_AC_value_l771_771474


namespace find_smallest_n_l771_771624

/-- 
Define the doubling sum function D(a, n)
-/
def doubling_sum (a : ℕ) (n : ℕ) : ℕ := a * (2^n - 1)

/--
Main theorem statement that proves the smallest n for the given conditions
-/
theorem find_smallest_n :
  ∃ (n : ℕ), (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 6 → ∃ (ai : ℕ), doubling_sum ai i = n) ∧ n = 9765 := 
sorry

end find_smallest_n_l771_771624


namespace transformed_data_properties_l771_771011

variable {n : ℕ} (x : Fin n → ℝ)
noncomputable def average := (∑ i, x i) / n
noncomputable def variance := (∑ i, (x i - average x) ^ 2) / n

theorem transformed_data_properties (x : Fin n → ℝ) :
  let y := fun i => 3 * x i + 5 in
  average y = 3 * average x + 5 ∧
  variance y = 9 * variance x :=
by
  let y := fun i => 3 * x i + 5
  sorry

end transformed_data_properties_l771_771011


namespace bench_press_after_injury_and_training_l771_771099

theorem bench_press_after_injury_and_training
  (p : ℕ) (h1 : p = 500) (h2 : p' : ℕ) (h3 : preduce : ℕ) (h4 : p' = p - preduce) 
  (h5 : preduce = 4 * p / 5) (h6 : q : ℕ) (h7 : q = 3 * p') : 
  q = 300 := by
  sorry

end bench_press_after_injury_and_training_l771_771099


namespace largest_prime_divisor_test_range_1000_1100_l771_771067

theorem largest_prime_divisor_test_range_1000_1100 (n : ℕ) (h1 : 1000 ≤ n ∧ n ≤ 1100) :
  ∃ p, Prime p ∧ p ≤ Int.sqrt 1100 ∧ ∀ q, Prime q ∧ q ≤ Int.sqrt 1100 → q ≤ p :=
begin
  sorry
end

end largest_prime_divisor_test_range_1000_1100_l771_771067


namespace ordered_6_tuples_count_l771_771265

theorem ordered_6_tuples_count (n : ℕ) : 
  let X := {x | 1 ≤ x ∧ x ≤ n} in
  ∃ A : finset (finset ℕ) × finset (finset ℕ) × finset (finset ℕ) × finset (finset ℕ) × finset (finset ℕ) × finset (finset ℕ),
  (∀ x ∈ X, ∑ (i : fin 6), if x ∈ A.1 ∪ A.2 ∪ A.3 ∪ A.4 ∪ A.5 ∪ A.6 then 1 else 0 = 0 ∨ 3 ∨ 6) →
  finset.card ((A.1 ∪ A.2 ∪ A.3 ∪ A.4 ∪ A.5 ∪ A.6).powerset) = 22^n :=
begin
  let X : finset ℕ := finset.range n,
  sorry
end

end ordered_6_tuples_count_l771_771265


namespace piece_exits_at_A2_l771_771937

structure Cell where
  row : Char -- 'A', 'B', 'C', 'D'
  col : Nat  -- 1, 2, 3, 4

-- Define the movement rules based on the conditions of the problem.
inductive Direction
| up | down | left | right

-- Define the movement based on the rules and how the direction changes.
def move_piece (current : Cell) (direction : Direction) : Cell :=
  match direction with
  | Direction.up => 
    if current.row == 'A' then current else ⟨Char.pred current.row, current.col⟩
  | Direction.down => 
    if current.row == 'D' then current else ⟨Char.succ current.row, current.col⟩
  | Direction.left => 
    if current.col == 1 then current else ⟨current.row, current.col - 1⟩
  | Direction.right => 
    if current.col == 4 then current else ⟨current.row, current.col + 1⟩

-- Initial position and direction of the piece
def initial_position : Cell := ⟨'C', 2⟩
def initial_direction : Direction := Direction.right

-- Translates the entire movement sequence into a proof problem for Lean
theorem piece_exits_at_A2 :
  ∃ (final_position : Cell) (final_direction : Direction),
    final_position = ⟨'A', 2⟩ 
    ∧ movement_sequence initial_position initial_direction outcome (move_piece (move_piece ...))  -- Placeholder for the correct sequence of moves.
    := sorry

end piece_exits_at_A2_l771_771937


namespace find_m_l771_771036

variables {m : ℝ}
def vec_a : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (2, 5)
def vec_c : ℝ × ℝ := (m, 3)
def vec_a_plus_c := (1 + m, 3 + m)
def vec_a_minus_b := (1 - 2, m - 5)

theorem find_m (h : (1 + m) * (m - 5) = -1 * (m + 3)) : m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 := 
sorry

end find_m_l771_771036


namespace sugar_amount_indeterminate_l771_771965

-- Define the variables and conditions
variable (cups_of_flour_needed : ℕ) (cups_of_sugar_needed : ℕ)
variable (cups_of_flour_put_in : ℕ) (cups_of_flour_to_add : ℕ)

-- Conditions
axiom H1 : cups_of_flour_needed = 8
axiom H2 : cups_of_flour_put_in = 4
axiom H3 : cups_of_flour_to_add = 4

-- Problem statement: Prove that the amount of sugar cannot be determined
theorem sugar_amount_indeterminate (h : cups_of_sugar_needed > 0) :
  cups_of_flour_needed = 8 → cups_of_flour_put_in = 4 → cups_of_flour_to_add = 4 → cups_of_sugar_needed > 0 :=
by
  intros
  sorry

end sugar_amount_indeterminate_l771_771965


namespace power_simplification_l771_771687

theorem power_simplification :
  (1 / ((-5) ^ 4) ^ 2) * (-5) ^ 9 = -5 :=
by 
  sorry

end power_simplification_l771_771687


namespace inverse_proposition_l771_771630

theorem inverse_proposition {T1 T2 : Triangle} : 
  (∀ (A B C : Angle), corresponding_angles T1 T2 ↔ congruent T1 T2) →
  (congruent T1 T2 ↔ ∀ (A B C : Angle), corresponding_angles T1 T2) :=
by
  sorry

end inverse_proposition_l771_771630


namespace min_max_calculation_l771_771569

theorem min_max_calculation
  (p q r s : ℝ)
  (h1 : p + q + r + s = 8)
  (h2 : p^2 + q^2 + r^2 + s^2 = 20) :
  -32 ≤ 5 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) ∧
  5 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) ≤ 12 :=
sorry

end min_max_calculation_l771_771569


namespace area_of_region_l771_771224

theorem area_of_region :
  (∃ (x y : ℝ), x^2 + y^2 + 6 * x - 12 * y - 10 = 0) →
  (area_of_shape (circle (-3) 6 (√55)) = 55 * π) :=
by
  intros h
  have h1 : (x + 3)^2 + (y - 6)^2 = 55 := -- completing the square step
    sorry -- proof required here
  exact sorry -- final proof step here

end area_of_region_l771_771224


namespace distance_BC_l771_771275

theorem distance_BC (AB AC CD DA: ℝ) (hAB: AB = 50) (hAC: AC = 40) (hCD: CD = 25) (hDA: DA = 35):
  BC = 10 ∨ BC = 90 :=
by
  sorry

end distance_BC_l771_771275


namespace total_monthly_sales_l771_771221

-- Definitions and conditions
def num_customers_per_month : ℕ := 500
def lettuce_per_customer : ℕ := 2
def price_per_lettuce : ℕ := 1
def tomatoes_per_customer : ℕ := 4
def price_per_tomato : ℕ := 1 / 2

-- Statement to prove
theorem total_monthly_sales : num_customers_per_month * (lettuce_per_customer * price_per_lettuce + tomatoes_per_customer * price_per_tomato) = 2000 := 
by 
  sorry

end total_monthly_sales_l771_771221


namespace largest_value_of_d_l771_771124

noncomputable def maximum_possible_value_of_d 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 17) : ℝ :=
  (5 + Real.sqrt 123) / 2

theorem largest_value_of_d 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 17) : 
  d ≤ maximum_possible_value_of_d a b c d h1 h2 :=
sorry

end largest_value_of_d_l771_771124


namespace solution_is_17_l771_771496

noncomputable def count_distinct_triangles : ℕ :=
    let t : Finset (ℕ × ℕ) := 
        { (a, b) | c = 10 ∧ ((a + b = 20 ∧ a ≥ b) ∨ ((2 * a = b + 10) ∧ a ≥ b) ∨ ((2 * b = a + 10) ∧ a ≥ b)) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a)  }.toFinset
    in t.card - 2      -- Account for duplications

theorem solution_is_17 : count_distinct_triangles = 17 :=
 by sorry

end solution_is_17_l771_771496


namespace chord_length_in_circle_l771_771940

theorem chord_length_in_circle 
  (radius : ℝ) 
  (chord_midpoint_perpendicular_radius : ℝ)
  (r_eq_10 : radius = 10)
  (cmp_eq_5 : chord_midpoint_perpendicular_radius = 5) : 
  ∃ (chord_length : ℝ), chord_length = 10 * Real.sqrt 3 := 
by 
  sorry

end chord_length_in_circle_l771_771940


namespace amy_remaining_money_l771_771322

-- Definitions based on conditions
def initial_money : ℕ := 100
def doll_cost : ℕ := 1
def number_of_dolls : ℕ := 3

-- The theorem we want to prove
theorem amy_remaining_money : initial_money - number_of_dolls * doll_cost = 97 :=
by 
  sorry

end amy_remaining_money_l771_771322


namespace ratios_equivalence_l771_771213

noncomputable def h (A B: ℝ) : ℝ := real.sqrt (A^2 - (B / 2)^2)

noncomputable def K1 (A B: ℝ) : ℝ := 1 / 2 * B * real.sqrt (A^2 - (B / 2)^2)

noncomputable def p (a: ℝ) : ℝ := 3 * a

noncomputable def k1 (a: ℝ) : ℝ := a^2 * real.sqrt 3 / 4

def compare_ratios (A B a: ℝ) (h p K1 k1: ℝ): Prop := 
  (A = a) →
  (h = real.sqrt (A^2 - (B / 2)^2)) →
  (K1 = 1 / 2 * B * real.sqrt (A^2 - (B / 2)^2)) →
  (p = 3 * a) →
  (k1 = a^2 * real.sqrt 3 / 4) →
  (h / p = K1 / k1) ↔ false

theorem ratios_equivalence : ∀ (A B a: ℝ), 
  (A = a) →
  ((A ≠ B) ∧ (A > 0) ∧ (B > 0) ∧ (a > 0)) →
  compare_ratios A B a (h A B) (p a) (K1 A B) (k1 a) := 
begin
  intros A B a A_eq_a basic_conditions,
  unfold compare_ratios,
  intros,
  sorry
end

end ratios_equivalence_l771_771213


namespace inequality_system_solution_l771_771610

theorem inequality_system_solution (x: ℝ) (h1: 5 * x - 2 < 3 * (x + 2)) (h2: (2 * x - 1) / 3 - (5 * x + 1) / 2 <= 1) : 
  -1 ≤ x ∧ x < 4 :=
sorry

end inequality_system_solution_l771_771610


namespace min_value_distinct_integers_l771_771123

noncomputable def omega : ℂ := exp (2 * π * complex.I / 4)

-- omega^4 = 1 and omega ≠ 1
lemma omega_property : omega ^ 4 = 1 ∧ omega ≠ 1 :=
begin
  split,
  { show (exp (2 * π * complex.I / 4)) ^ 4 = 1,
    rw [←exp_nat_mul, mul_comm, exp_zero, mul_div_cancel'_right],
    exact π_ne_zero },
  { intro h,
    have : exp (2 * π * complex.I / 4) = 1,
    { rw exp_eq_one_iff at h,
      rcases h with ⟨n, hn⟩,
      linarith [complex.I_ne_zero, pi_pos] },
    contradiction }
end

theorem min_value_distinct_integers (a b c d : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ∃ x : ℝ, x = |a + b * omega + c * omega^2 + d * omega^3| ∧ x ≥ sqrt 3 :=
sorry

end min_value_distinct_integers_l771_771123


namespace mean_of_remaining_three_numbers_l771_771168

variable {a b c : ℝ}

theorem mean_of_remaining_three_numbers (h1 : (a + b + c + 103) / 4 = 90) : (a + b + c) / 3 = 85.7 :=
by
  -- Sorry placeholder for the proof
  sorry

end mean_of_remaining_three_numbers_l771_771168


namespace problem1_problem2_l771_771459

def set_A (a : ℝ) : set ℝ := { x | (x-2)*(x-(3*a+1)) < 0 }
def set_B (a : ℝ) : set ℝ := { x | (x-2*a)/(x-(a^2+1)) < 0 }

/-- Problem 1: Prove that when a = 2, A ∩ B = (4, 5) --/
theorem problem1 (a : ℝ) (h : a = 2) : 
  (set_A a ∩ set_B a) = { x | 4 < x ∧ x < 5 } := by
  sorry

/-- Problem 2: Find the range of a such that B ⊆ A --/
theorem problem2 (a : ℝ) : 
  (∀ x, x ∈ set_B a -> x ∈ set_A a)
    ↔ (a = -1 ∨ (1 ≤ a ∧ a ≤ 3)) := by
  sorry

end problem1_problem2_l771_771459


namespace f_has_two_zeros_l771_771635

def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem f_has_two_zeros : ∃ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 := 
by
  sorry

end f_has_two_zeros_l771_771635


namespace find_x_plus_y_l771_771506

theorem find_x_plus_y (x y : ℚ) (h1 : 3 * x - 4 * y = 18) (h2 : x + 3 * y = -1) :
  x + y = 29 / 13 :=
sorry

end find_x_plus_y_l771_771506


namespace smallest_base_for_80_l771_771249

-- Define the problem in terms of inequalities
def smallest_base (n : ℕ) (d : ℕ) :=
  ∃ b : ℕ, b > 1 ∧ b <= (n^(1/d)) ∧ (n^(1/(d+1))) < (b + 1)

-- Assertion that the smallest whole number b such that 80 can be expressed in base b using only three digits
theorem smallest_base_for_80 : ∃ b, smallest_base 80 3 ∧ b = 5 :=
  sorry

end smallest_base_for_80_l771_771249


namespace sin_12x_eq_x_solutions_in_0_pi_l771_771634

theorem sin_12x_eq_x_solutions_in_0_pi :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.sin (12 * x)) → 
  set.countable {x ∈ set.Ico 0 Real.pi | f x = x} ∧
  set.to_finset {x ∈ set.Ico 0 Real.pi | f x = x}.card = 6 :=
by
  sorry

end sin_12x_eq_x_solutions_in_0_pi_l771_771634


namespace function_extremum_in_interval_l771_771173

theorem function_extremum_in_interval (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 1 → (x^3 + x^2 - a*x - 4).derivative = 3*x^2 + 2*x - a) →
  (3 * (-1)^2 + 2 * (-1) - a ≤ 0) ∧ (3 * 1^2 + 2 * 1 - a ≥ 0) →
  1 ≤ a ∧ a < 5 :=
sorry

end function_extremum_in_interval_l771_771173


namespace smallest_n_with_ten_trailing_zeros_l771_771698

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def legendre_formula (n p : ℕ) : ℕ :=
  if p = 0 then 0 else ∑ i in (finset.range (n.log p + 1)).image (λ i, p ^ i), n / i

def v_p (n p : ℕ) : ℕ :=
  legendre_formula n p 

def smallest_n_with_trailing_zeros (k : ℕ) : ℕ :=
  nat.find (λ n, v_p (factorial n) 5 ≥ k)

theorem smallest_n_with_ten_trailing_zeros : smallest_n_with_trailing_zeros 10 = 45 := by
  sorry

end smallest_n_with_ten_trailing_zeros_l771_771698


namespace NaCl_proof_l771_771816

-- Example Chemical Reaction statement for proof
def NH4Cl := Type
def NaOH := Type
def NaCl := Type
def NH3 := Type
def H2O := Type

-- Amounts in moles
constant n_NH4Cl : ℕ := 2
constant n_NaOH : ℕ := 2
constant n_NaCl : ℕ := 2

-- Balanced chemical reaction (ratios)
axiom balanced_reaction : (n_NH4Cl, n_NaOH) → n_NaCl

-- Proof problem statement
theorem NaCl_proof (n_NH4Cl : ℕ) (n_NaOH : ℕ) : n_NaCl = 2 := 
by 
-- Proof steps will be inserted here logically 
sorry

end NaCl_proof_l771_771816


namespace find_a_find_k_l771_771028

noncomputable def f (x : ℝ) (a : ℝ) := a * x^3 + 3 * x^2 - 6 * a * x - 11
noncomputable def g (x : ℝ) := 3 * x^2 + 6 * x + 12
noncomputable def m (x k : ℝ) := k * x + 9

-- (1) Prove: a = -2 given f'(-1) = 0
theorem find_a (a : ℝ) (h : deriv (f x a) = λ x, 3 * a * x^2 + 6 * x - 6 * a) (h' : deriv (f x a) (-1) = 0) : a = -2 := sorry

-- (2) Determine the value of k such that the line m is tangent to both curves f and g
theorem find_k (a : ℝ) (k : ℝ) (h : a = -2)
  (h_tangent_f : ∃ c, m c k = f c a ∧ deriv (f x a) c = k)
  (h_tangent_g : ∃ d, m d k = g d ∧ deriv g d = k) : k = 0 := sorry

end find_a_find_k_l771_771028


namespace pair_points_no_intersection_l771_771849

-- Definition of the problem
noncomputable def pairwise_non_intersecting (T : Set (ℝ × ℝ)) : Prop :=
∀ p q r s ∈ T, p ≠ q → r ≠ s → ¬ (segments_intersect p q r s)

-- Statement of the mathematical problem
theorem pair_points_no_intersection (T : Set (ℝ × ℝ)) (hT : ∃ n, T.card = 2 * n) : 
  ∃ pairs : Finset (Finset (ℝ × ℝ)), 
    (pairs.card = hT.some) ∧ 
    (∀ p ∈ pairs, p.card = 2) ∧
    pairwise_non_intersecting T := 
sorry

end pair_points_no_intersection_l771_771849


namespace quadratic_vertex_position_l771_771468

theorem quadratic_vertex_position (a p q m : ℝ) (ha : 0 < a) (hpq : p < q) (hA : p = a * (-1 - m)^2) (hB : q = a * (3 - m)^2) : m ≠ 2 :=
by
  sorry

end quadratic_vertex_position_l771_771468


namespace slope_of_line_l771_771026

theorem slope_of_line : 
  (∀ x y : ℝ, (y = (1/2) * x + 1) → ∃ m : ℝ, m = 1/2) :=
sorry

end slope_of_line_l771_771026


namespace horner_value_x_neg2_l771_771681

noncomputable def horner (x : ℝ) : ℝ :=
  (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 0.3) * x + 2

theorem horner_value_x_neg2 : horner (-2) = -40 :=
by
  sorry

end horner_value_x_neg2_l771_771681


namespace pair_points_no_intersection_l771_771855

noncomputable theory
open_locale classical

universe u

def no_intersecting_pairs (T : set (ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 p3 p4 : ℝ × ℝ), p1 ∈ T → p2 ∈ T → p3 ∈ T → p4 ∈ T → 
  p1 ≠ p2 → p3 ≠ p4 →
  ¬((p1, p2), (p3, p4) : set (ℝ × ℝ) × set (ℝ × ℝ)).pairwise_disjoint segments

theorem pair_points_no_intersection (T : set (ℝ × ℝ)) (hT : ∃ (n : ℕ), T.card = 2 * n) :
  ∃ P : set (ℝ × ℝ) × set (ℝ × ℝ), no_intersecting_pairs T :=
sorry

end pair_points_no_intersection_l771_771855


namespace ramanujan_picked_l771_771079

noncomputable def ramanujan_number (r h : ℂ) := r * h = 48 - 16 * complex.I

theorem ramanujan_picked {r : ℂ} (h : ℂ) (hr : h = 6 + 2 * complex.I) :
  ramanujan_number r h → r = 6.4 - 4.8 * complex.I :=
begin
  intro H,
  rw [ramanujan_number, hr] at H,
  -- The following line is the core part of the challenge,
  -- showing how to derive that r = 6.4 - 4.8i
  sorry
end

end ramanujan_picked_l771_771079


namespace tan_diff_alpha_l771_771865

open Real

theorem tan_diff_alpha (α : ℝ) (h1 : cos (π + α) = 3 / 5) (h2 : α ∈ Ioo (π / 2) π) :
  tan (π / 4 - α) = -7 := by
  sorry

end tan_diff_alpha_l771_771865


namespace floor_sum_arith_prog_l771_771778

theorem floor_sum_arith_prog :
  ∑ k in (Finset.range 142), (⟦2 + k * 0.7⟧) = 7242 :=
by
  sorry

end floor_sum_arith_prog_l771_771778


namespace IsoTriConcycLineParallel_l771_771538

theorem IsoTriConcycLineParallel
  {A B C D I E : Type}
  (h_iso : AB = AC)
  (h_incenter : incenter I A B C)
  (h_concyc : concyclic {I, B, C, D})
  (h_parallel : parallel (line C E) (line B D))
  (h_intersect : ∃ E, E ∈ extension (line A D) ∧ ∃ k : Real, E = k * (C - B)) :
  CD^2 = BD * CE :=
sorry

end IsoTriConcycLineParallel_l771_771538


namespace extreme_points_inequality_l771_771627

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * Real.log (1 + x)

-- Given m > 0 and f(x) has extreme points x1 and x2 such that x1 < x2
theorem extreme_points_inequality {m x1 x2 : ℝ} (h_m : m > 0)
    (h_extreme1 : x1 = (-1 - Real.sqrt (1 - 2 * m)) / 2)
    (h_extreme2 : x2 = (-1 + Real.sqrt (1 - 2 * m)) / 2)
    (h_order : x1 < x2) :
    2 * f x2 m > -x1 + 2 * x1 * Real.log 2 := sorry

end extreme_points_inequality_l771_771627


namespace part1_l771_771952

theorem part1 (a b c t m n : ℝ) (h1 : a > 0) (h2 : m = n) (h3 : t = (3 + (t + 1)) / 2) : t = 4 :=
sorry

end part1_l771_771952


namespace distance_between_points_l771_771238

-- Define the two points.
def point1 : ℝ × ℝ := (-3, 5)
def point2 : ℝ × ℝ := (4, -9)

-- Define the distance formula.
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- State the theorem.
theorem distance_between_points :
  distance point1 point2 = real.sqrt 245 :=
by
  -- Placeholder for the proof.
  sorry

end distance_between_points_l771_771238


namespace hyperbola_eccentricity_sqrt_five_l771_771488

/-- Given a hyperbola with the equation x^2/a^2 - y^2/b^2 = 1 where a > 0 and b > 0,
and its focus lies symmetrically with respect to the asymptote lines and on the hyperbola,
proves that the eccentricity of the hyperbola is sqrt(5). -/
theorem hyperbola_eccentricity_sqrt_five 
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) 
  (c : ℝ) (h_focus : c^2 = 5 * a^2) : 
  (c / a = Real.sqrt 5) := sorry

end hyperbola_eccentricity_sqrt_five_l771_771488


namespace prob_both_A_B_prob_exactly_one_l771_771677

def prob_A : ℝ := 0.8
def prob_not_B : ℝ := 0.1
def prob_B : ℝ := 1 - prob_not_B

lemma prob_independent (a b : Prop) : Prop := -- Placeholder for actual independence definition
sorry

-- Given conditions
variables (P_A : ℝ := prob_A) (P_not_B : ℝ := prob_not_B) (P_B : ℝ := prob_B) (indep : ∀ A B, prob_independent A B)

-- Questions translated to Lean statements
theorem prob_both_A_B : P_A * P_B = 0.72 := sorry

theorem prob_exactly_one : (P_A * P_not_B) + ((1 - P_A) * P_B) = 0.26 := sorry

end prob_both_A_B_prob_exactly_one_l771_771677


namespace num_integers_with_digit_sum_seventeen_l771_771896

-- Define a function to calculate the sum of the digits of an integer
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (λ acc d, acc + d) 0

-- Define the main proposition
theorem num_integers_with_digit_sum_seventeen : 
  (finset.filter (λ n, sum_of_digits n = 17) (finset.Icc 400 600)).card = 13 := 
sorry

end num_integers_with_digit_sum_seventeen_l771_771896


namespace find_l_eq_find_l1_y_intercept_l771_771020

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨-1, 2⟩
def C : Point := ⟨5, 4⟩

def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def D : Point := midpoint B C

def line_eq (p1 p2 : Point) : (ℝ × ℝ × ℝ) :=
  let a := p2.y - p1.y
  let b := p1.x - p2.x
  let c := p2.x * p1.y - p1.x * p2.y
  (a, b, c)

def l : ℝ × ℝ × ℝ := line_eq A D

def is_parallel (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  l1.1 * l2.2 = l1.2 * l2.1

def y_intercept (l : ℝ × ℝ × ℝ) : ℝ :=
  -l.3 / l.2

theorem find_l_eq : l = (1, -1, 1) := by
  sorry

theorem find_l1_y_intercept : 
  (∃ l1, l1.1 * B.x + l1.2 * B.y + l1.3 = 0 ∧ is_parallel l l1 ∧ y_intercept l1 = 3) := by
  sorry

end find_l_eq_find_l1_y_intercept_l771_771020


namespace area_of_circle_segment_l771_771223

-- Definitions for the conditions in the problem
def circle_eq (x y : ℝ) : Prop := x^2 - 10 * x + y^2 = 9
def line_eq (x y : ℝ) : Prop := y = x - 5

-- The area of the portion of the circle that lies above the x-axis and to the left of the line y = x - 5
theorem area_of_circle_segment :
  let area_of_circle := 34 * Real.pi
  let portion_fraction := 1 / 8
  portion_fraction * area_of_circle = 4.25 * Real.pi :=
by
  sorry

end area_of_circle_segment_l771_771223


namespace distance_between_points_l771_771861

theorem distance_between_points :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (4, 4)
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5 := by
  sorry

end distance_between_points_l771_771861


namespace fred_earnings_l771_771966
noncomputable def start := 111
noncomputable def now := 115
noncomputable def earnings := now - start

theorem fred_earnings : earnings = 4 :=
by
  sorry

end fred_earnings_l771_771966


namespace mary_income_percentage_more_than_tim_l771_771996

variables (J T M : ℝ)
-- Define the conditions
def condition1 := T = 0.5 * J -- Tim's income is 50% less than Juan's
def condition2 := M = 0.8 * J -- Mary's income is 80% of Juan's

-- Define the theorem stating the question and the correct answer
theorem mary_income_percentage_more_than_tim (J T M : ℝ) 
  (h1 : T = 0.5 * J) 
  (h2 : M = 0.8 * J) : 
  (M - T) / T * 100 = 60 := 
  by sorry

end mary_income_percentage_more_than_tim_l771_771996


namespace pair_points_no_intersection_l771_771847

-- Definition of the problem
noncomputable def pairwise_non_intersecting (T : Set (ℝ × ℝ)) : Prop :=
∀ p q r s ∈ T, p ≠ q → r ≠ s → ¬ (segments_intersect p q r s)

-- Statement of the mathematical problem
theorem pair_points_no_intersection (T : Set (ℝ × ℝ)) (hT : ∃ n, T.card = 2 * n) : 
  ∃ pairs : Finset (Finset (ℝ × ℝ)), 
    (pairs.card = hT.some) ∧ 
    (∀ p ∈ pairs, p.card = 2) ∧
    pairwise_non_intersecting T := 
sorry

end pair_points_no_intersection_l771_771847


namespace smallest_solution_l771_771819

theorem smallest_solution (x : ℝ) : (1 / (x - 3) + 1 / (x - 5) = 5 / (x - 4)) → x = 4 - (Real.sqrt 15) / 3 :=
by
  sorry

end smallest_solution_l771_771819


namespace find_a_of_polar_to_cartesian_and_geometric_condition_l771_771546

/-- Given a polar curve and parametric line, finding the correct value of a based on geometric
conditions. -/
theorem find_a_of_polar_to_cartesian_and_geometric_condition
  (a : ℝ) (h₀ : a > 0)
  (C : ℝ → ℝ → Prop)
  (hC : ∀ (ρ θ : ℝ), C (ρ * Real.cos θ) (ρ * Real.sin θ) ↔ ρ * (Real.sin θ)^2 = 2 * a * (Real.cos θ))
  (l : ℝ → ℝ → Prop)
  (hl : ∀ t, (l (-2 + (Real.sqrt 2 / 2) * t) (-4 + (Real.sqrt 2 / 2) * t))
    ↔ (l (-2 + (Real.sqrt 2 / 2) * t) (-4 + (Real.sqrt 2 / 2) * t)))
  (P : ℝ × ℝ)
  (hP : P = (-2, -4))
  (M N : ℝ × ℝ)
  (hM : M ∈ {p | ∃ t, l p.1 p.2 ∧ ∃ t₁ t₂, l p.1 p.2 ∧ C p.1 p.2})
  (hN : N ∈ {p | ∃ t, l p.1 p.2 ∧ ∃ t₁ t₂, l p.1 p.2 ∧ C p.1 p.2})
  (geom_seq : (λ P M N : ℝ × ℝ, dist P M, dist M N, dist P N) (dist P M) (dist M N) (dist P N)) :
  (C x y ↔ y^2 = 2 * a * x) ∧ (l x y ↔ x - y - 2 = 0) ∧ (a = 1) := 
sorry

end find_a_of_polar_to_cartesian_and_geometric_condition_l771_771546


namespace hyperbola_ratio_l771_771843

theorem hyperbola_ratio (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0)
  (h_eq : a^2 - b^2 = 1)
  (h_ecc : 2 = c / a)
  (h_focus : c = 1) :
  a / b = Real.sqrt 3 / 3 := by
  have ha : a = 1 / 2 := sorry
  have hc : c = 1 := h_focus
  have hb : b = Real.sqrt 3 / 2 := sorry
  exact sorry

end hyperbola_ratio_l771_771843


namespace gray_region_area_correct_l771_771338

noncomputable theory

-- Definitions of the conditions of the problem
def center_C : (ℝ × ℝ) := (3, 5)
def radius_C : ℝ := 5
def center_D : (ℝ × ℝ) := (13, 5)
def radius_D : ℝ := 5

-- Definition of the question and expected answer
def gray_region_area : ℝ := 50 - 12.5 * Real.pi

-- Equation to be proved
theorem gray_region_area_correct :
  ∃ (area : ℝ), area = gray_region_area :=
by
  use 50 - 12.5 * Real.pi
  sorry

end gray_region_area_correct_l771_771338


namespace find_k_l771_771023

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + d * (n - 1)

def sum_arithmetic_seq (a1 d n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem find_k (k : ℕ) (h_pos : k > 0) : 
  let a_n := arithmetic_seq 4 2 (k + 5),
      S_n := sum_arithmetic_seq 4 2 k in
  S_n - a_n = 44 → k = 7 :=
by
  intro h
  sorry

end find_k_l771_771023


namespace angle_AKT_eq_angle_CAM_l771_771145

open EuclideanGeometry

variables {A B C D K T M : Point}
variables [IsoscelesTriangle A B C] [Between B C A]
variables {circumcircle_BCD : Set Point} [Circumcircle BCD circumcircle_BCD]
variables {line_parallel_BC_through_A : Line} [Parallel (Line.mk A B) (Line.mk A C)]
variables (onArc_K : OnCircleArc circumcircle_BCD C D K)
variables (midpoint_M : Midpoint D T M)

theorem angle_AKT_eq_angle_CAM :
  ∠ A K T = ∠ C A M :=
sorry

end angle_AKT_eq_angle_CAM_l771_771145


namespace polyhedron_with_special_projections_l771_771094

open_locale classical

noncomputable theory

def exists_special_projections_polyhedron : Prop :=
  ∃ (P : set (set ℝ^3)),
  ∃ perpendicular_planes : set (set ℝ^2),
  perpendicular_planes.card = 3 ∧
  (∃ (triangle_projection : set ℝ^2), triangle_projection ∈ perpendicular_planes ∧ triangle_projection.shape = "triangle") ∧
  (∃ (quadrilateral_projection : set ℝ^2), quadrilateral_projection ∈ perpendicular_planes ∧ quadrilateral_projection.shape = "quadrilateral") ∧
  (∃ (pentagon_projection : set ℝ^2), pentagon_projection ∈ perpendicular_planes ∧ pentagon_projection.shape = "pentagon")

theorem polyhedron_with_special_projections : exists_special_projections_polyhedron :=
sorry

end polyhedron_with_special_projections_l771_771094


namespace polygon_sides_l771_771520

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 720) : n = 6 :=
sorry

end polygon_sides_l771_771520


namespace binary_1101_is_13_l771_771344

-- defining the binary number as a list of bits
def binaryNumber : List ℕ := [1, 1, 0, 1]

-- function to convert a list of bits representing a binary number to decimal
def binaryToDecimal (bits : List ℕ) : ℕ :=
  bits.foldr (λ (bit : ℕ) (acc : ℕ × ℕ), (acc.1 + bit * acc.2, acc.2 * 2)) (0, 1) |>.fst

theorem binary_1101_is_13 : binaryToDecimal binaryNumber = 13 := by
  sorry

end binary_1101_is_13_l771_771344


namespace sum_of_roots_of_quadratic_eq_l771_771505

theorem sum_of_roots_of_quadratic_eq (x : ℝ) :
  (x + 3) * (x - 4) = 18 → (∃ a b : ℝ, x ^ 2 + a * x + b = 0) ∧ (a = -1) ∧ (b = -30) :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l771_771505


namespace best_shooter_l771_771830

noncomputable def avg_A : ℝ := 9
noncomputable def avg_B : ℝ := 8
noncomputable def avg_C : ℝ := 9
noncomputable def avg_D : ℝ := 9

noncomputable def var_A : ℝ := 1.2
noncomputable def var_B : ℝ := 0.4
noncomputable def var_C : ℝ := 1.8
noncomputable def var_D : ℝ := 0.4

theorem best_shooter :
  (avg_A = 9 ∧ var_A = 1.2) →
  (avg_B = 8 ∧ var_B = 0.4) →
  (avg_C = 9 ∧ var_C = 1.8) →
  (avg_D = 9 ∧ var_D = 0.4) →
  avg_D = 9 ∧ var_D = 0.4 :=
by {
  sorry
}

end best_shooter_l771_771830


namespace evaluate_pow_l771_771360

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l771_771360


namespace option_A_option_B_l771_771874

variable (A B : Prop)
variable [ProbabilityMeasure ℙ₀]
variable [ProbabilityMeasure ℙ₁]

axiom PA : ℙ₀(A) = 1 / 3
axiom PB : ℙ₁(B) = 1 / 6

-- Option A
axiom PAB : ℙ₁(B | ¬A) = 1 / 9

theorem option_A :
  (ℙ₀(¬A) * ℙ₁(B)) = 1 / 9 :=
by sorry

-- Option B
axiom A_and_B_indep : IndependentEvents A B

theorem option_B :
  ℙ₀(A) + ℙ₁(B) - ℙ₀(A ∧ B) = 4 / 9 :=
by sorry

end option_A_option_B_l771_771874


namespace evaluate_neg_64_exp_4_over_3_l771_771367

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l771_771367


namespace find_two_digit_numbers_l771_771803

theorem find_two_digit_numbers :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → (10 * a + b = 3 * a * b) → (10 * a + b = 15 ∨ 10 * a + b = 24) :=
by
  intros
  sorry

end find_two_digit_numbers_l771_771803


namespace max_product_geom_seq_l771_771781

-- Definition of geometric sequence terms based on the given solution.
def geom_seq (a₁ : ℕ) (q : ℕ) (n : ℕ) : ℕ := a₁ * q ^ (n - 1)

-- Conditions given in the problem.
axiom cond1 : geom_seq 8 2 1 + geom_seq 8 2 3 = 10
axiom cond2 : geom_seq 8 2 2 + geom_seq 8 2 4 = 5

-- The maximum product of the first n terms in the geometric sequence.
def prod_geom_seq (a₁ : ℕ) (q : ℕ) (n : ℕ) : ℕ := 
  (List.range n).map (λ i, geom_seq a₁ q (i + 1)).foldl (*) 1

theorem max_product_geom_seq : ∃ n, prod_geom_seq 8 2 n = 64 :=
by
  use 4
  have h₁ : geom_seq 8 2 1 = 8 := by sorry
  have h₂ : geom_seq 8 2 2 = 4 := by sorry
  have h₃ : geom_seq 8 2 3 = 2 := by sorry
  have h₄ : geom_seq 8 2 4 = 1 := by sorry
  have prod_bound : (geom_seq 8 2 1) * (geom_seq 8 2 2) * (geom_seq 8 2 3) * (geom_seq 8 2 4) = 64 := by
    calc
      (geom_seq 8 2 1) * (geom_seq 8 2 2) * (geom_seq 8 2 3) * (geom_seq 8 2 4) =
      8 * 4 * 2 * 1 : by rw [h₁, h₂, h₃, h₄]
      ... = 64 : by norm_num
  exact prod_bound

end max_product_geom_seq_l771_771781


namespace cartesian_equation_of_curve_rectangular_equation_of_line_polar_coordinates_of_intersection_points_l771_771955

section

variable {α : Type} [Real]

def param_curve_C (α : Real) : Real × Real :=
  (cos α, 1 + sin α)

def polar_line_l (ρ θ : Real) : Prop :=
  (sqrt 2) * ρ * cos (θ + π / 4) = -2

theorem cartesian_equation_of_curve :
  ∀ x y, (∃ α, (x = cos α) ∧ (y = 1 + sin α)) ↔ (x^2 + (y-1)^2 = 1) :=
by
  -- Proof omitted
  sorry

theorem rectangular_equation_of_line :
  ∀ x y, polar_line_l (sqrt(x^2 + y^2)) (arctan (y / x)) ↔ (x - y + 2 = 0) :=
by
  -- Proof omitted
  sorry

theorem polar_coordinates_of_intersection_points :
  ∃ (ρ1 θ1 ρ2 θ2 : Real), (ρ1 ≥ 0 ∧ 0 ≤ θ1 ∧ θ1 < 2 * π) ∧ 
                            (ρ2 ≥ 0 ∧ 0 ≤ θ2 ∧ θ2 < 2 * π) ∧ 
                            (ρ1 = sqrt 2) ∧ (θ1 = 3 * π / 4) ∧ 
                            (ρ2 = 2) ∧ (θ2 = π / 2) ∧ 
                            ∀ x y, (x, y) = (-1, 1) ∨ (x, y) = (0, 2)
                            ↔ (ρ1, θ1) = (sqrt 2, 3 * π / 4) ∧ (ρ2, θ2) = (2, π / 2) :=
by
  -- Proof omitted
  sorry

end

end cartesian_equation_of_curve_rectangular_equation_of_line_polar_coordinates_of_intersection_points_l771_771955


namespace solve_for_x_l771_771604

theorem solve_for_x (x : ℝ) :
  (x - 2)^6 + (x - 6)^6 = 64 → x = 3 ∨ x = 5 :=
by
  intros h
  sorry

end solve_for_x_l771_771604


namespace kristin_bell_peppers_count_l771_771550

variables (jaylen_carrots : ℕ) (jaylen_cucumbers : ℕ) (jaylen_bell_peppers : ℕ) (kristin_bell_peppers : ℕ)
          (jaylen_green_beans : ℕ) (kristin_green_beans : ℕ) (jaylen_total_vegetables : ℕ)

def jaylen_has_5_carrots : jaylen_carrots = 5 := sorry
def jaylen_has_2_cucumbers : jaylen_cucumbers = 2 := sorry
def jaylen_has_twice_as_many_bell_peppers_as_kristin : jaylen_bell_peppers = 2 * kristin_bell_peppers := sorry
def jaylen_has_3_less_than_half_kristin_green_beans : jaylen_green_beans = kristin_green_beans / 2 - 3 := sorry
def kristin_has_20_green_beans : kristin_green_beans = 20 := sorry
def jaylen_has_18_vegetables_in_total : jaylen_total_vegetables = 18 := sorry
def total_vegetables : jaylen_carrots + jaylen_cucumbers + jaylen_green_beans + jaylen_bell_peppers = jaylen_total_vegetables := sorry

theorem kristin_bell_peppers_count : kristin_bell_peppers = 2 := 
by
assume h : 
  jaylen_carrots = 5 ∧ 
  jaylen_cucumbers = 2 ∧ 
  jaylen_bell_peppers = 2 * kristin_bell_peppers ∧ 
  jaylen_green_beans = kristin_green_beans / 2 - 3 ∧ 
  kristin_green_beans = 20 ∧ 
  jaylen_total_vegetables = 18 ∧
  jaylen_carrots + jaylen_cucumbers + jaylen_green_beans + jaylen_bell_peppers = jaylen_total_vegetables,
show kristin_bell_peppers = 2, from
sorry

end kristin_bell_peppers_count_l771_771550


namespace slope_of_l_l771_771954

noncomputable def C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 4 * Real.sin θ)
noncomputable def l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

theorem slope_of_l
  (α θ₁ θ₂ t₁ t₂ : ℝ)
  (h_midpoint : (C θ₁).fst + (C θ₂).fst = 1 + (t₁ + t₂) * Real.cos α ∧ 
                (C θ₁).snd + (C θ₂).snd = 2 + (t₁ + t₂) * Real.sin α) :
  Real.tan α = -2 :=
by
  sorry

end slope_of_l_l771_771954


namespace find_m_value_l771_771866

theorem find_m_value
  (x y : ℤ)
  (h1 : x = 2)
  (h2 : y = m)
  (h3 : 3 * x + 2 * y = 10) : 
  m = 2 :=
by
  sorry

end find_m_value_l771_771866


namespace sum_of_zeros_l771_771035

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x then 2^(x - 2) - 1 else x + 2

noncomputable def g (x : ℝ) : ℝ :=
if 0 ≤ x then x^2 - 2 * x else 1 / x

noncomputable def h (x : ℝ) : ℝ :=
f (g x)

theorem sum_of_zeros : (1 + sqrt 3) + (-1 / 2) = 1/2 + sqrt 3 := by
  sorry

end sum_of_zeros_l771_771035


namespace find_m_l771_771734

theorem find_m (m : ℚ) : 
  (∃ m, (∀ x y z : ℚ, ((x, y) = (2, 9) ∨ (x, y) = (15, m) ∨ (x, y) = (35, 4)) ∧ 
  (∀ a b c d e f : ℚ, ((a, b) = (2, 9) ∨ (a, b) = (15, m) ∨ (a, b) = (35, 4)) → 
  ((b - d) / (a - c) = (f - d) / (e - c))) → m = 232 / 33)) :=
sorry

end find_m_l771_771734


namespace infinite_series_convergence_l771_771559

theorem infinite_series_convergence (a : ℕ → ℕ)
  (strictly_increasing : ∀ {i j : ℕ}, i < j → a i < a j)
  (positive_integers : ∀ {n : ℕ}, 0 < a n)
  (gcd_condition : ∀ {i j : ℕ}, i ≠ j → Nat.gcd (a i) (a j) = 1)
  (difference_condition : ∀ {i : ℕ}, a (i + 2) - a (i + 1) > a (i + 1) - a i):
  Summable (λ i, 1 / (a i : ℝ)) := 
  sorry

end infinite_series_convergence_l771_771559


namespace compound_interest_comparison_l771_771928

theorem compound_interest_comparison :
  (1 + 0.04) < (1 + 0.04 / 12) ^ 12 := sorry

end compound_interest_comparison_l771_771928


namespace monotonic_decreasing_interval_l771_771182

noncomputable def func (x : ℝ) : ℝ :=
  x * Real.log x

noncomputable def derivative (x : ℝ) : ℝ :=
  Real.log x + 1

theorem monotonic_decreasing_interval :
  { x : ℝ | 0 < x ∧ x < Real.exp (-1) } ⊆ { x : ℝ | derivative x < 0 } :=
by
  sorry

end monotonic_decreasing_interval_l771_771182


namespace number_with_coprime_property_l771_771134

theorem number_with_coprime_property (n : ℕ) (a : Fin n → ℤ) : 
  let b := (∏ i, a i) + 1 in
  b > 1 ∧ ∀ i, Nat.gcd (a i).natAbs b.natAbs = 1 := by
  sorry

end number_with_coprime_property_l771_771134


namespace ellipse_line_slope_intersection_l771_771625

theorem ellipse_line_slope_intersection (m n : ℝ) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1 → x + y = 1) →
  (∀ x1 y1 x2 y2 : ℝ, (mx1^2 + ny1^2 = 1) ∧ (mx2^2 + ny2^2 = 1) →
  (x1 + y1 = 1) ∧ (x2 + y2 = 1) →
  let x0 := (x1 + x2) / 2,
      y0 := (y1 + y2) / 2 in
  (y0 / x0 = sqrt 2 / 2)) →
  (m / n = sqrt 2 / 2) :=
sorry

end ellipse_line_slope_intersection_l771_771625


namespace math_problem_l771_771838

theorem math_problem
  (a b c t : ℝ)
  (h1 : 0 ≤ a)
  (h2 : 0 ≤ b)
  (h3 : 0 ≤ c)
  (h4 : 1 ≤ t)
  (h5 : a + b + c = 1/2)
  (h6 : Real.sqrt (a + 1/2 * (b - c)^2) + Real.sqrt b + Real.sqrt c = Real.sqrt (6 * t) / 2) :
  a^(2 * t) + b^(2 * t) + c^(2 * t) = 1 / 12 := 
begin
  sorry
end

end math_problem_l771_771838


namespace brunch_combinations_l771_771679

def bread_types := 3
def fruit_types := 4
def drink_types := 3
def fruit_combinations := Nat.choose fruit_types 2 -- Combination of choosing 2 out of 4 fruits

theorem brunch_combinations : bread_types * fruit_combinations * drink_types = 54 := 
by
  -- number of bread choices
  have h_bread : bread_types = 3 := rfl

  -- number of drink choices
  have h_drink : drink_types = 3 := rfl

  -- number of ways to choose 2 out of 4 fruits
  have h_fruit_combinations : fruit_combinations = Nat.choose 4 2 := rfl
  have h_fruit_combinations_value : Nat.choose 4 2 = 6 := by norm_num

  -- total combinations
  calc
    3 * 6 * 3 = 54 := by norm_num

end brunch_combinations_l771_771679


namespace sector_area_is_half_sin_one_l771_771470

-- Definitions
def perimeter (r θ : ℝ) : ℝ := 2 * r + r * θ
def sector_area (r θ : ℝ) : ℝ := 1 / 2 * r^2 * θ

-- Variables and Hypothesis
variables (r θ : ℝ)
hypothesis_perimeter : perimeter r θ = 3
hypothesis_angle : θ = 1

-- Theorem to prove
theorem sector_area_is_half_sin_one :
  sector_area r θ = 1 / 2 * real.sin 1 :=
by
  have h1 : r = 1 := by
    have key := hypothesis_perimeter
    sorry
  have h2 := hypothesis_angle
  sorry

end sector_area_is_half_sin_one_l771_771470


namespace magnitude_of_sum_l771_771166

variables (a b : ℝ × ℝ)
def angle (v w : ℝ × ℝ) : ℝ := 
  real.arccos ((v.1 * w.1 + v.2 * w.2) / (real.sqrt(v.1 ^ 2 + v.2 ^ 2) * real.sqrt(w.1 ^ 2 + w.2 ^ 2)))

def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := 
  (v.1 + w.1, v.2 + w.2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := 
  (k * v.1, k * v.2)

-- Given conditions: a = (2, 0) and |b| = 1, and the angle between them is 60°
axiom a_def : a = (2, 0)
axiom b_mag : vector_magnitude b = 1
axiom angle_a_b : angle a b = real.pi / 3

theorem magnitude_of_sum : vector_magnitude (vector_add a (scalar_mult 2 b)) = 2 * real.sqrt 3 := 
  sorry

end magnitude_of_sum_l771_771166


namespace find_f_2000_l771_771821

variable (f : ℕ → ℕ)
variable (x : ℕ)

axiom initial_condition : f 0 = 1
axiom recurrence_relation : ∀ x, f (x + 2) = f x + 4 * x + 2

theorem find_f_2000 : f 2000 = 3998001 :=
by
  sorry

end find_f_2000_l771_771821


namespace collinear_vectors_l771_771579

theorem collinear_vectors :
  ∃ x : ℝ, (collinear ({ x, 1 }, { 4, x }))
    ∧ ∀ y : ℝ, collinear ({ y, 1 }, { 4, y }) → (y = 2 ∨ y = -2) :=
by
  sorry

end collinear_vectors_l771_771579


namespace proof_problem_l771_771001

variables (a b c : Line) (alpha beta gamma : Plane)

-- Define perpendicular relationship between line and plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define parallel relationship between lines
def parallel_lines (l1 l2 : Line) : Prop := sorry

-- Define parallel relationship between planes
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Main theorem statement
theorem proof_problem 
  (h1 : perp_line_plane a alpha) 
  (h2 : perp_line_plane b beta) 
  (h3 : parallel_planes alpha beta) : 
  parallel_lines a b :=
sorry

end proof_problem_l771_771001


namespace place_circle_no_overlap_l771_771147

theorem place_circle_no_overlap 
    (rect_width rect_height : ℝ) (num_squares : ℤ) (square_size square_diameter : ℝ)
    (h_rect_dims : rect_width = 20 ∧ rect_height = 25)
    (h_num_squares : num_squares = 120)
    (h_square_size : square_size = 1)
    (h_circle_diameter : square_diameter = 1) : 
  ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ rect_width ∧ 0 ≤ y ∧ y ≤ rect_height ∧ 
    ∀ (square_x square_y : ℝ), 
      0 ≤ square_x ∧ square_x ≤ rect_width - square_size ∧ 
      0 ≤ square_y ∧ square_y ≤ rect_height - square_size → 
      (x - square_x)^2 + (y - square_y)^2 ≥ (square_diameter / 2)^2 :=
sorry

end place_circle_no_overlap_l771_771147


namespace semifinalists_not_advance_l771_771076

theorem semifinalists_not_advance
  (s : ℕ) (medals : ℕ) (groups : ℕ)
  (h_s : s = 8)
  (h_medals : medals = 3)
  (h_groups : groups = 56) :
  ∑ n in Ico 1 s.succ, if (∑ k in Ico 1 n, k = groups) && (medals = 3) then some 0 else none = some 0 :=
by
  sorry

end semifinalists_not_advance_l771_771076


namespace inscribable_polygons_false_l771_771595

noncomputable def inscribable_polygons (n : ℕ) (h : n ≥ 6) : Prop :=
  ¬ (∃ (inscribe : (n-1)-gon ⊂ n-gon), ∀ (i : ℕ), i < n-1 → 
    (inscribe.B (i+1) ⊆ inscribe.A (i+2) ∧ i ≠ n-2 ∧ inscribe.B 0 ⊆ inscribe.A 1))

theorem inscribable_polygons_false (n : ℕ) (h : n ≥ 6) : inscribable_polygons n h :=
sorry

end inscribable_polygons_false_l771_771595


namespace power_evaluation_l771_771386

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l771_771386


namespace arithmetic_seq_sum_l771_771946

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h₁ : ∀ n k : ℕ, a (n + k) = a n + k * d) 
  (h₂ : a 5 + a 6 + a 7 + a 8 = 20) : a 1 + a 12 = 10 := 
by 
  sorry

end arithmetic_seq_sum_l771_771946


namespace total_percentage_increase_l771_771552

def initial_salary : Float := 60
def first_raise (s : Float) : Float := s + 0.10 * s
def second_raise (s : Float) : Float := s + 0.15 * s
def deduction (s : Float) : Float := s - 0.05 * s
def promotion_raise (s : Float) : Float := s + 0.20 * s
def final_salary (s : Float) : Float := promotion_raise (deduction (second_raise (first_raise s)))

theorem total_percentage_increase :
  final_salary initial_salary = initial_salary * 1.4421 :=
by
  sorry

end total_percentage_increase_l771_771552


namespace cherries_in_mix_l771_771298

theorem cherries_in_mix (total_fruit : ℕ) (blueberries : ℕ) (raspberries : ℕ) (cherries : ℕ) 
  (H1 : total_fruit = 300)
  (H2: raspberries = 3 * blueberries)
  (H3: cherries = 5 * blueberries)
  (H4: total_fruit = blueberries + raspberries + cherries) : cherries = 167 :=
by
  sorry

end cherries_in_mix_l771_771298


namespace group_time_l771_771503

theorem group_time (total_students number_groups time_per_student : ℕ) (h1 : total_students = 18) (h2 : number_groups = 3) (h3 : time_per_student = 4) :
  let students_per_group := total_students / number_groups in
  let time_per_group := students_per_group * time_per_student in
  time_per_group = 24 :=
by
  let students_per_group := total_students / number_groups
  let time_per_group := students_per_group * time_per_student
  have students_per_group_calc : students_per_group = 6, by sorry
  have time_per_group_calc : time_per_group = 24, by sorry
  exact time_per_group_calc

end group_time_l771_771503


namespace final_bathtub_water_l771_771963

-- Define the conditions and rates
def faucet1_rate : ℕ := 40  -- ml/min
def faucet2_rate : ℕ := 60  -- ml/min
def evaporation_rate_initial : ℕ := 200  -- ml/hour
def evaporation_increment : ℕ := 50  -- ml/hour
def dumping_water : ℕ := 15000  -- ml (15 liters)

-- Problem duration
def duration : ℕ := 9  -- hours
def initial_duration : ℕ := 4  -- hours
def remaining_duration := duration - initial_duration  -- hours

-- Total water input & calculations
def total_water_input : ℕ := (faucet1_rate * 60 * duration) + (faucet2_rate * 60 * duration)  -- total ml
def total_initial_evaporation : ℕ := evaporation_rate_initial * initial_duration  -- ml
def total_remaining_evaporation : ℕ := (List.range remaining_duration).sum (λ i, evaporation_rate_initial + (i+1) * evaporation_increment)  -- ml
def total_evaporation : ℕ := total_initial_evaporation + total_remaining_evaporation  -- ml

-- Problem's expected answer
def final_water_volume (input_water evaporation water_dumped : ℕ) : ℕ :=
  input_water - evaporation - water_dumped

-- Lean statement
theorem final_bathtub_water (f1 f2 evap_init evap_inc dump dur : ℕ) :
  final_water_volume ((f1 * 60 * dur) + (f2 * 60 * dur))
      (evap_init * initial_duration + (List.range remaining_duration).sum (λ i, evap_init + (i + 1) * evap_inc)) dump = 36450 :=
by
  -- Input values based on the problem statement
  let f1 := faucet1_rate
  let f2 := faucet2_rate
  let evap_init := evaporation_rate_initial
  let evap_inc := evaporation_increment
  let dump := dumping_water
  let dur := duration
  exact sorry

end final_bathtub_water_l771_771963


namespace odd_terms_in_binomial_expansion_l771_771510

-- Define the problem conditions and the statement
theorem odd_terms_in_binomial_expansion (a b : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) : 
  let terms := [(nat.choose 8 0) * (a^8),
                (nat.choose 8 1) * (a^7 * b),
                (nat.choose 8 2) * (a^6 * b^2),
                (nat.choose 8 3) * (a^5 * b^3),
                (nat.choose 8 4) * (a^4 * b^4),
                (nat.choose 8 5) * (a^3 * b^5),
                (nat.choose 8 6) * (a^2 * b^6),
                (nat.choose 8 7) * (a * b^7),
                (nat.choose 8 8) * (b^8)] in
  (terms.filter (λ term, term % 2 = 1)).length = 2 :=
sorry

end odd_terms_in_binomial_expansion_l771_771510


namespace determine_square_in_10_operations_l771_771583

noncomputable def minimum_operations_to_determine_square {A B C D : Type}
  (measure_distance : A → A → ℝ) 
  (compare_numbers : ℝ → ℝ → bool) 
  (pAB : measure_distance A B)
  (pBC : measure_distance B C)
  (pCD : measure_distance C D)
  (pDA : measure_distance D A)
  (pAC : measure_distance A C)
  (pBD : measure_distance B D)
  (e1 : compare_numbers pAB pBC)
  (e2 : compare_numbers pBC pCD)
  (e3 : compare_numbers pCD pDA)
  (e4 : compare_numbers pAC pBD) : ℕ :=
4 + 2 + 3 + 1

theorem determine_square_in_10_operations
  (A B C D : Type)
  (measure_distance : A → A → ℝ) 
  (compare_numbers : ℝ → ℝ → bool) 
  (pAB : measure_distance A B)
  (pBC : measure_distance B C)
  (pCD : measure_distance C D)
  (pDA : measure_distance D A)
  (pAC : measure_distance A C)
  (pBD : measure_distance B D)
  (e1 : compare_numbers pAB pBC)
  (e2 : compare_numbers pBC pCD)
  (e3 : compare_numbers pCD pDA)
  (e4 : compare_numbers pAC pBD) :
  minimum_operations_to_determine_square measure_distance compare_numbers pAB pBC pCD pDA pAC pBD e1 e2 e3 e4 = 10 :=
by
  sorry

end determine_square_in_10_operations_l771_771583


namespace collinear_points_y_l771_771455

theorem collinear_points_y (y : ℝ) 
  (h : ∃ (A B C : ℝ × ℝ), A = (4, 8) ∧ B = (2, 4) ∧ C = (3, y) 
  ∧ (∀ (P Q R : ℝ × ℝ), P = A ∧ Q = B ∧ R = C → collinear P Q R)) : y = 6 :=
by
  sorry

end collinear_points_y_l771_771455


namespace sum_of_consecutive_odd_integers_l771_771709

theorem sum_of_consecutive_odd_integers (n : ℕ) (h : (∑ i in finset.range n, 2 * i + 1) = 169) : n = 13 :=
sorry

end sum_of_consecutive_odd_integers_l771_771709


namespace purchase_price_is_60_l771_771738

variable (P S D : ℝ)
variable (GP : ℝ := 4)

theorem purchase_price_is_60
  (h1 : S = P + 0.25 * S)
  (h2 : D = 0.80 * S)
  (h3 : GP = D - P) :
  P = 60 :=
by
  sorry

end purchase_price_is_60_l771_771738


namespace probability_heart_then_club_l771_771657

theorem probability_heart_then_club :
  let P_heart := 13 / 52
  let P_club_given_heart := 13 / 51
  P_heart * P_club_given_heart = 13 / 204 := 
by
  let P_heart := (13 : ℚ) / 52
  let P_club_given_heart := (13 : ℚ) / 51
  have h : P_heart * P_club_given_heart = 13 / 204 := by
    calc
      P_heart * P_club_given_heart
        = (13 / 52) * (13 / 51) : rfl
    ... = (13 * 13) / (52 * 51) : by rw [mul_div_mul_comm]
    ... = 169 / 2652 : rfl
    ... = 13 / 204 : by norm_num
  exact h

end probability_heart_then_club_l771_771657


namespace pqrsum_l771_771128

-- Given constants and conditions:
variables {p q r : ℝ} -- p, q, r are real numbers
axiom Hpq : p < q -- given condition p < q
axiom Hineq : ∀ x : ℝ, (x > 5 ∨ 7 ≤ x ∧ x ≤ 15) ↔ ( (x - p) * (x - q) / (x - r) ≥ 0) -- given inequality condition

-- Values from the solution:
axiom Hp : p = 7
axiom Hq : q = 15
axiom Hr : r = 5

-- Proof statement:
theorem pqrsum : p + 2 * q + 3 * r = 52 :=
sorry 

end pqrsum_l771_771128


namespace rotate_disks_to_positive_sum_l771_771674

variable {n : ℕ}
variable (a b : Fin n → ℝ)

-- The conditions
def conditions (n : ℕ) (a b : Fin n → ℝ) : Prop :=
  (0 < ∑ i, a i) ∧ (0 < ∑ i, b i)

-- The function to calculate the sum of products for a given rotation k
def S (k : ℕ) : ℝ :=
  ∑ i, a i * b ((i + k) % n)

-- The theorem statement
theorem rotate_disks_to_positive_sum (n : ℕ) (a b : Fin n → ℝ) 
  (h₀ : conditions n a b) : ∃ k : Fin n, 0 < S a b k :=
by
  sorry

end rotate_disks_to_positive_sum_l771_771674


namespace magnitude_calc_l771_771917

open Real

variables {e₁ e₂ : EuclideanSpace ℝ (Fin 3)}

noncomputable def is_unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∥v∥ = 1

noncomputable def orthogonal (v w : EuclideanSpace ℝ (Fin 3)) : Prop :=
  inner v w = 0

theorem magnitude_calc (he₁ : is_unit_vector e₁) (he₂ : is_unit_vector e₂)
  (h_ortho : orthogonal (2 • e₁ + e₂) (-2 • e₁ + 3 • e₂)) :
  ∥e₁ + 2 • e₂∥ = sqrt 6 :=
sorry

end magnitude_calc_l771_771917


namespace range_of_k_l771_771487

noncomputable def f (x : ℝ) : ℝ := (exp 2 * x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := (exp 2 * x) / (exp x)

theorem range_of_k (x1 x2 : ℝ) (k : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 1 ≤ k) : 
    (g x1 / k ≤ f x2 / (k + 1)) :=
sorry

end range_of_k_l771_771487


namespace obtain_a6_l771_771962

theorem obtain_a6 (a : ℚ) (h1 : a^4) (h2 : a^6 - 1) : ∃ x, x = a^6 := by
  sorry

end obtain_a6_l771_771962


namespace evaluate_pow_l771_771366

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l771_771366


namespace tangent_slope_l771_771061

theorem tangent_slope (x : ℝ) (h : (deriv (λ x : ℝ, x^2) x = 4)) : x = 2 :=
by {
  intros,
  calc x = 2 : sorry
}

end tangent_slope_l771_771061


namespace sum_of_possible_integers_eq_36_l771_771869

theorem sum_of_possible_integers_eq_36 (m : ℤ) (h1 : 0 < 5 * m) (h2 : 5 * m < 45) : 
  (Finset.sum (Finset.filter (λ x, 0 < x ∧ x < 9) (Finset.range 9)) id) = 36 :=
by
  sorry

end sum_of_possible_integers_eq_36_l771_771869


namespace tangents_parallel_to_common_tangents_l771_771216

-- Definitions that directly appear in the conditions problem
variables {A B C D : Point}
variable (circle1 : Circle)
variable (circle2 : Circle)
variable (circle3 : Circle)
  
-- Conditions provided in the problem.
variables (h1 : circle1 ∋ A ∧ circle1 ∋ B)
variables (h2 : circle2 ∋ A ∧ circle2 ∋ B)
variables (h3 : circle3.touches circle1)
variables (h4 : circle3.touches circle2)
variables (h5 : circle3 ∣ AB = {C, D})

-- Statement to be proved
theorem tangents_parallel_to_common_tangents :
  ∀ (tang_C tang_D : Line), 
    (tang_C.is_tangent_to circle3 ∧ tang_C ∋ C) ∧
    (tang_D.is_tangent_to circle3 ∧ tang_D ∋ D) → 
    tang_C ∥ common_tangent_of circle1 circle2 ∧
    tang_D ∥ common_tangent_of circle1 circle2 :=
sorry

end tangents_parallel_to_common_tangents_l771_771216


namespace find_angle_A_find_value_of_b_l771_771547

noncomputable def area_of_triangle (a b c : ℝ) (A : ℝ) : ℝ :=
  0.5 * b * c * Real.sin(A)

theorem find_angle_A (a b c A S : ℝ) (h1 : 2 * Real.sqrt 3 * S = b * c * Real.cos(A)) 
  (h2 : c = 2) : A = Real.pi / 6 :=
by
  sorry

theorem find_value_of_b (a b c A B C S : ℝ) 
  (h1 : 2 * Real.sqrt 3 * S = b * c * Real.cos(A))
  (h2 : c = 2) 
  (h3 : a^2 + b^2 - c^2 = 6/5 * a * b) 
  : b = (3 + 4 * Real.sqrt 3) / 4 :=
by
  sorry

end find_angle_A_find_value_of_b_l771_771547


namespace sqrt_of_9_eq_pm_3_l771_771644

theorem sqrt_of_9_eq_pm_3 : (∃ x : ℤ, x * x = 9) → (∃ x : ℤ, x = 3 ∨ x = -3) :=
by
  sorry

end sqrt_of_9_eq_pm_3_l771_771644


namespace concyclic_X_P_Q_Y_l771_771082

-- Definitions for points and segments in a given quadrilateral ABCD
section
variables {A B C D E F P Q X Y : Type*}
variables [euclidean_geometry.point A B C D E F P Q X Y]
variables {AB AD CB CD : ℝ} [AB = AD] [CB = CD]
variables (angle_ABC : ∠ B A C = π/2)
variables {segment : Type*} [euclidean_geometry.segment_AB segment B]
variables [euclidean_geometry.segment_AD segment D]

-- Definitions for the placement of points on segments and the given ratio conditions
variables {E_on_AB : E ∈ segment AB}
variables {F_on_AD : F ∈ segment AD}
variables {P_on_EF : P ∈ segment EF}
variables {Q_on_EF : Q ∈ segment EF}
variables {P_between_E_and_Q : euclidean_geometry.between E P Q}
variables {ratio_condition : ∀ (AE EP AF FQ : ℝ), AE / EP = AF / FQ}

-- Definitions for perpendicular conditions
variables {BX_perp_CP : euclidean_geometry.perpendicular (segment B X) (segment CP)}
variables {DY_perp_CQ : euclidean_geometry.perpendicular (segment D Y) (segment CQ)}

theorem concyclic_X_P_Q_Y : euclidean_geometry.concyclic {X, P, Q, Y} := sorry
end

end concyclic_X_P_Q_Y_l771_771082


namespace minimum_value_expression_l771_771000

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∃ x : ℝ, x = 9 ∧ ∀ y : ℝ, y = (1 / a^2 - 1) * (1 / b^2 - 1) → x ≤ y) :=
sorry

end minimum_value_expression_l771_771000


namespace grover_total_profit_l771_771044

theorem grover_total_profit :
  let boxes := 3
  let masks_per_box := 20
  let price_per_mask := 0.50
  let cost := 15
  let total_masks := boxes * masks_per_box
  let total_revenue := total_masks * price_per_mask
  let total_profit := total_revenue - cost
  total_profit = 15 := by
sorry

end grover_total_profit_l771_771044


namespace average_speed_l771_771704

theorem average_speed (x : ℝ) (h1 : x > 0) :
  let time1 := x / 40,
      time2 := 2 * x / 20,
      total_time := time1 + time2,
      total_distance := 2 * x in
  total_distance / total_time = 16 :=
by
  sorry

end average_speed_l771_771704


namespace trajectory_of_P_is_ray_l771_771456

-- Define the points M and N
structure Point where
  x : ℝ
  y : ℝ

def M : Point := {x := -2, y := 0}
def N : Point := {x := 2, y := 0}

-- Distance function between two points
def distance (P1 P2 : Point) : ℝ :=
  real.sqrt ((P1.x - P2.x)^2 + (P1.y - P2.y)^2)

-- Condition: |PM| - |PN| = 4
def condition (P : Point) : Prop :=
  (distance P M - distance P N) = 4

-- Trajectory of point P is a ray extending to the right from the midpoint of MN
noncomputable def trajectory : set Point :=
  {P | P.x >= 0 ∧ P.y = 0}

theorem trajectory_of_P_is_ray (P : Point) (h : condition P) :
  P ∈ trajectory :=
  sorry

end trajectory_of_P_is_ray_l771_771456


namespace two_digit_numbers_equal_three_times_product_of_digits_l771_771804

theorem two_digit_numbers_equal_three_times_product_of_digits :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 3 * a * b} = {15, 24} :=
by
  sorry

end two_digit_numbers_equal_three_times_product_of_digits_l771_771804


namespace consecutive_sum_36_unique_l771_771500

def is_consecutive_sum (a b n : ℕ) :=
  (0 < n) ∧ ((n ≥ 2) ∧ (b = a + n - 1) ∧ (2 * a + n - 1) * n = 72)

theorem consecutive_sum_36_unique :
  ∃! n, ∃ a b, is_consecutive_sum a b n :=
by
  sorry

end consecutive_sum_36_unique_l771_771500


namespace count_3_letter_words_l771_771495

theorem count_3_letter_words : 
  ∃ w : Finset (List Char), w.card = 9 ∧ 
    (∀ x : List Char, x ∈ w → x.length = 3 ∧ List.getLast x (by simp) = 'E' ∧ 'A' ∈ x) :=
by
  sorry

end count_3_letter_words_l771_771495


namespace fraction_of_EF_l771_771591

variables (E F G H : Point)
          (length : Point → Point → ℝ)
          (hEF : F ∈ segment G H)
          (hGE : length G E = 3 * length E H)
          (hGF : length G F = 5 * length F H)

theorem fraction_of_EF : length E F / length G H = 1 / 12 :=
by
  -- the proof goes here
  sorry

end fraction_of_EF_l771_771591


namespace find_speed_ratio_l771_771335

-- Define the problem setup
variables (A B C D : Point)
variables (k : ℝ)
variables (speedA speedB : ℝ) (travel_time_A_to_B travel_time_B_to_A : ℝ)

-- Given conditions
def condition1 := speedB = k * speedA
def condition2 : k > 1 := sorry
def condition3 := midpoint_between AB D
def condition4 := (C : Point) and B meets A at C on return with same speed

-- Given ratio CD/AD = 1/2
def condition5 := dist C D = (1 / 2 : ℝ) * dist A D

-- Target condition
def target := k = 2

-- Main theorem to prove
theorem find_speed_ratio 
    (A B C D : Point)
    (k : ℝ)
    (speedA speedB : ℝ)
    (travel_time_A_to_B travel_time_B_to_A : ℝ)
    (h1 : speedB = k * speedA)
    (h2 : k > 1)
    (h3 : midpoint_between AB D)
    (h4 : (C : Point) and B meets A at C on return with same speed)
    (h5 : dist C D = (1 / 2 : ℝ) * dist A D) :
    k = 2 :=
begin
    sorry,
end

end find_speed_ratio_l771_771335


namespace monotonic_intervals_a_b_half_range_of_m_l771_771881

-- Definitions for Part 1
def f1 (x : ℝ) : ℝ := real.log x - (1 / 4) * x^2 - (1 / 2) * x
def f1_deriv (x : ℝ) : ℝ := -(x + 2) * (x - 1) / (2 * x)

-- Theorem for Part 1
theorem monotonic_intervals_a_b_half : 
  (∀ x : ℝ, 0 < x ∧ x < 1 → f1_deriv x > 0) ∧ 
  (∀ x : ℝ, x > 1 → f1_deriv x < 0) :=
sorry

-- Definitions for Part 2
def f2 (x : ℝ) : ℝ := real.log x + x
def g (x : ℝ) : ℝ := 1 + real.log x / x
def g_deriv (x : ℝ) : ℝ := (1 - real.log x) / x^2

-- Theorem for Part 2
theorem range_of_m :
  g 1 = 1 ∧ g (exp 2) = 1 + 2 / (exp 2) ∧ g (exp 1) = 1 + 1 / (exp 1) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ real.exp 1 → g_deriv x > 0) ∧ 
  (∀ x : ℝ, real.exp 1 ≤ x ∧ x ≤ real.exp 2 → g_deriv x < 0) ∧ 
  (∀ m : ℝ, m = 1 + 1 / (real.exp 1) ∨ (1 ≤ m ∧ m < 1 + 2 / (real.exp 2))) :=
sorry

end monotonic_intervals_a_b_half_range_of_m_l771_771881


namespace probability_of_event_A_l771_771281

def event_A (m n : ℕ) : Prop :=
  m + n ≤ 6 ∧ m ≥ 1 ∧ n ≥ 1 ∧ m ≤ 6 ∧ n ≤ 6

theorem probability_of_event_A : 
  (finset.univ.filter (λ mn, event_A mn.1 mn.2)).card / finset.univ.card = 5 / 12 :=
sorry

end probability_of_event_A_l771_771281


namespace eval_expression_correct_l771_771400

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l771_771400


namespace range_of_m_l771_771818

theorem range_of_m (m : ℝ) : (-6 < m ∧ m < 2) ↔ ∃ x : ℝ, |x - m| + |x + 2| < 4 :=
by sorry

end range_of_m_l771_771818


namespace intersection_points_range_l771_771059

def f (x : ℝ) : ℝ := 1/3 * x^3 - 4 * x + 4

theorem intersection_points_range (b : ℝ) : 
  (∃ x1 x2 x3 : ℝ, f x1 = b ∧ f x2 = b ∧ f x3 = b ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ 
  b ∈ set.Ioo (-(4 / 3)) (28 / 3) :=
by sorry

end intersection_points_range_l771_771059


namespace fraction_expression_l771_771340

theorem fraction_expression : (1 / 3) ^ 3 * (1 / 8) = 1 / 216 :=
by
  sorry

end fraction_expression_l771_771340


namespace yuna_average_score_l771_771702

theorem yuna_average_score (avg_may_june : ℕ) (score_july : ℕ) (h1 : avg_may_june = 84) (h2 : score_july = 96) :
  (avg_may_june * 2 + score_july) / 3 = 88 := by
  sorry

end yuna_average_score_l771_771702


namespace sum_product_difference_l771_771158

theorem sum_product_difference :
  ∃ (a b : ℕ), a ∈ Finset.range (41) ∧ b ∈ Finset.range (41) ∧ a ≠ b ∧
               (∑ i in (Finset.range (41)).erase a ∪ {a} ∪ {b}, i) = 820 ∧
               ( ∑ i in (Finset.range (41)).erase a ∪ (Finset.range (41)).erase b, i ) + 2 = a * b ∧
               abs (a - b) = 50 :=
by
  sorry

end sum_product_difference_l771_771158


namespace leadership_choices_l771_771761

/--
Given a tribe with 22 members where the tribal leadership hierarchy consists of 
one chief, 3 supporting chiefs (supporting chief A, B, and C), and each supporting chief 
has 3 inferior officers, prove that the number of different ways to choose the leadership 
of the tribe is 22308038400.
-/
theorem leadership_choices (n : ℕ) (h : n = 22) : 
  (choose n 1) * (choose (n - 1) 3) * (factorial 3) * 
  (choose (n - 4) 3) * (choose (n - 7) 3) * 
  (choose (n - 10) 3) = 22308038400 :=
by
  sorry

end leadership_choices_l771_771761


namespace solution_set_inequality_l771_771424

theorem solution_set_inequality (x : ℝ) :
  (|x + 3| - |x - 3| > 3) ↔ (x > 3 / 2) := 
sorry

end solution_set_inequality_l771_771424


namespace distance_between_points_l771_771234

def point : Type := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  Math.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem distance_between_points :
  distance (-3, 5 : point) (4, -9 : point) = Math.sqrt 245 := 
sorry

end distance_between_points_l771_771234


namespace largest_prime_divisor_to_test_in_range_1000_1100_l771_771065

theorem largest_prime_divisor_to_test_in_range_1000_1100 : 
  ∀ n : ℕ, (1000 ≤ n ∧ n ≤ 1100) → (∀ p : ℕ, p.prime → p ≤ 31 → ¬ p ∣ n) → (31 = Nat.max (List.filter Prime (List.range (Nat.floor (Real.sqrt 1100))))) :=
by
  intros n h_bounds h_prime_test
  sorry

end largest_prime_divisor_to_test_in_range_1000_1100_l771_771065


namespace find_missing_x_coordinate_l771_771078

theorem find_missing_x_coordinate (x : ℝ) : 
  (vertices : list (ℝ × ℝ)) → 
  (vertices = [(-8, 1), (x, 1), (x, -7), (-8, -7)]) → 
  (area : ℝ) → 
  area = 72 → 
  x = 1 :=
by
  intro vertices hverts area hare
  rw hverts at hare
  sorry

end find_missing_x_coordinate_l771_771078


namespace find_m_for_perfect_square_trinomial_l771_771914

theorem find_m_for_perfect_square_trinomial :
  ∃ m : ℤ, (∀ (x y : ℝ), (9 * x^2 + m * x * y + 16 * y^2 = (3 * x + 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (3 * x - 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (-3 * x + 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (-3 * x - 4 * y)^2)) ↔ 
          (m = 24 ∨ m = -24) := 
by
  sorry

end find_m_for_perfect_square_trinomial_l771_771914


namespace snail_minute_hand_meeting_times_l771_771745

theorem snail_minute_hand_meeting_times :
  ∀ (starting_time : ℕ) (full_circle_time : ℕ) (minute_hand_circle_time : ℕ),
    starting_time = 0 →
    full_circle_time = 120 →
    minute_hand_circle_time = 60 →
    ∃ (meeting_times : list ℕ), meeting_times = [40, 80] →
      (([starting_time + (full_circle_time / 3 * n) % full_circle_time | n in [1, 2]]) = [40, 80]) :=
by
  intros
  -- Skip the proof for now, just ensure the theorem statement is valid as shown.
  sorry

end snail_minute_hand_meeting_times_l771_771745


namespace triangle_rectangle_area_l771_771656

theorem triangle_rectangle_area (DE EF DF : ℕ) (h1 : DE = 15) (h2 : EF = 39) (h3 : DF = 36)
  (Area_DEF : ℕ) (h4 : Area_DEF = 270)
  (Area_WXYZ : ℕ → ℕ) (h5 : ∀ ω, Area_WXYZ ω = 39 * ω - ((60 / 169) * ω^2))
: ∃ p q : ℕ, p + q = 229 :=
by
  use 60
  use 169
  norm_num
  exact rfl

end triangle_rectangle_area_l771_771656


namespace min_black_cells_in_grid_l771_771284

-- Define the grid dimensions
def grid_width : ℕ := 12
def grid_height : ℕ := 12

-- Define the conditions for the subgrids
def subgrid1_width : ℕ := 3
def subgrid1_height : ℕ := 4
def subgrid2_width : ℕ := 4
def subgrid2_height : ℕ := 3

-- Define the minimum number of black cells required
def min_black_cells : ℕ := 12

-- State the theorem
theorem min_black_cells_in_grid :
  ∀ (grid : fin grid_height → fin grid_width → bool),
  (∀ (x : fin (grid_height / subgrid1_height)) (y : fin (grid_width / subgrid1_width)),
    ∃ i j, ((x.val * subgrid1_height + i) < grid_height) ∧ ((y.val * subgrid1_width + j) < grid_width) ∧ 
            (grid ⟨x.val * subgrid1_height + i, sorry⟩ ⟨y.val * subgrid1_width + j, sorry⟩ = tt)) ∧
  (∀ (x : fin (grid_height / subgrid2_height)) (y : fin (grid_width / subgrid2_width)),
    ∃ i j, ((x.val * subgrid2_height + i) < grid_height) ∧ ((y.val * subgrid2_width + j) < grid_width) ∧ 
            (grid ⟨x.val * subgrid2_height + i, sorry⟩ ⟨y.val * subgrid2_width + j, sorry⟩ = tt)) →
  (∃ (black_cells : finset (fin grid_height × fin grid_width)), 
    black_cells.card = min_black_cells ∧
    ∀ x y, (∃ i j, ((x.val * subgrid1_height + i) < grid_height) ∧ ((y.val * subgrid1_width + j) < grid_width) ∧ 
             (⟨x.val * subgrid1_height + i, sorry⟩, ⟨y.val * subgrid1_width + j, sorry⟩) ∈ black_cells) ∨ 
            (∃ i j, ((x.val * subgrid2_height + i) < grid_height) ∧ ((y.val * subgrid2_width + j) < grid_width) ∧ 
             (⟨x.val * subgrid2_height + i, sorry⟩, ⟨y.val * subgrid2_width + j, sorry⟩) ∈ black_cells)) := sorry

end min_black_cells_in_grid_l771_771284


namespace smallest_geometric_third_term_l771_771744

theorem smallest_geometric_third_term
  (d : ℝ)
  (h1 : (10 + d) ^ 2 = 7 * (29 + 2 * d)) :
  ∃ d, (d = -3 + 8 * Real.sqrt 7 ∨ d = -3 - 8 * Real.sqrt 7) ∧ 
  let third_term := 29 + 2 * d in
  third_term = 23 + 16 * Real.sqrt 7 :=
begin
  sorry
end

end smallest_geometric_third_term_l771_771744


namespace ratio_proof_l771_771915

theorem ratio_proof (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) : (a + b) / (b + c) = 4 / 15 := by
  sorry

end ratio_proof_l771_771915


namespace power_equivalence_l771_771047

theorem power_equivalence (m : ℕ) : 16^6 = 4^m → m = 12 :=
by
  sorry

end power_equivalence_l771_771047


namespace circle_center_l771_771808

theorem circle_center (x y : ℝ) :
  4 * x^2 - 16 * x + 4 * y^2 + 8 * y - 12 = 0 →
  (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = 8 ∧ h = 2 ∧ k = -1) :=
sorry

end circle_center_l771_771808


namespace distinct_factors_count_l771_771497

theorem distinct_factors_count :
  let n := 4^4 * 5^5 * 7^3 in
  (∃ a b c : ℕ, n = 2^8 * 5^5 * 7^3 ∧ 
                 a ∈ finset.range (9) ∧ 
                 b ∈ finset.range (6) ∧ 
                 c ∈ finset.range (4)) → 9 * 6 * 4 = 216 :=
by
  sorry

end distinct_factors_count_l771_771497


namespace bottles_remaining_l771_771746

def initial_small := 6000
def initial_big := 14000
def percent_sold_small := 0.20
def percent_sold_big := 0.23

theorem bottles_remaining :
  let sold_small := initial_small * percent_sold_small in
  let sold_big := initial_big * percent_sold_big in
  let remaining_small := initial_small - sold_small in
  let remaining_big := initial_big - sold_big in
  let total_remaining := remaining_small + remaining_big in
  total_remaining = 15580 :=
by
  sorry

end bottles_remaining_l771_771746


namespace partition_grid_l771_771969

theorem partition_grid (n : ℕ) (hn : n > 0) :
  let ways := {p : fin (n+1) → fin (n+1) // 
                ∀ i1 i2 j1 j2, p i1 = j1 → p i2 = j2 → i1 ≠ i2 → j1 ≠ j2} in
  ways.fintype.card = (n + 1)! :=
by sorry

end partition_grid_l771_771969


namespace solution_set_of_inequality_l771_771925

def f (x : ℝ) : ℝ := 4 - x ^ 2 - 2 * log (x ^ 2 + 1) / log 5

theorem solution_set_of_inequality :
  {x : ℝ | f (log x) + 3 * f (log (1 / x)) + 8 ≤ 0} = {x : ℝ | 0 < x ∧ (x ≤ exp (-2) ∨ exp 2 ≤ x)} :=
begin
  sorry
end

end solution_set_of_inequality_l771_771925


namespace circles_externally_tangent_l771_771350

theorem circles_externally_tangent :
  let C1x := -3
  let C1y := 2
  let r1 := 2
  let C2x := 3
  let C2y := -6
  let r2 := 8
  let d := Real.sqrt ((C2x - C1x)^2 + (C2y - C1y)^2)
  (d = r1 + r2) → 
  ((x + 3)^2 + (y - 2)^2 = 4) → ((x - 3)^2 + (y + 6)^2 = 64) → 
  ∃ (P : ℝ × ℝ), (P.1 + 3)^2 + (P.2 - 2)^2 = 4 ∧ (P.1 - 3)^2 + (P.2 + 6)^2 = 64 :=
by
  intros
  sorry

end circles_externally_tangent_l771_771350


namespace max_value_of_fraction_l771_771886

theorem max_value_of_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∃ (z : ℝ), z = (x + y) / (x - y) ∧ 
  ∀ (a b : ℝ) (ha : -3 ≤ a ∧ a ≤ -1) (hb : 1 ≤ b ∧ b ≤ 3), (a + b) / (a - b) ≤ z) :=
begin
  use (1/2),
  intros a b ha hb,
  sorry
end

end max_value_of_fraction_l771_771886


namespace max_correct_answers_l771_771739

theorem max_correct_answers (c w b : ℕ) (h1 : c + w + b = 25) (h2 : 6 * c - 3 * w = 60) : c ≤ 15 :=
by {
  sorry
}

end max_correct_answers_l771_771739


namespace ratio_of_areas_is_one_l771_771084

noncomputable def right_triangle : Type :=
{ A B C : Type |
  hypotenuse : ℝ, 
  legs : ℝ × ℝ,
  angleB : ℝ,
  h_AB : ℝ,
  h_BC : ℝ,
  midpointD : ℝ × ℝ,
  midpointE : ℝ × ℝ,
  intersectionX : Prop }

theorem ratio_of_areas_is_one (A B C : Type) (AB BC : ℝ) (angleB ninety_degrees : ℝ) (D E : Type) 
  (m_A_mid_AB m_B_mid_AC : Prop)
  (AC : ℝ) (intersection_CD_BE_at_X : Prop) : 
  (ratio (area A E X D) (area B X C) = 1) :=
by
  sorry

end ratio_of_areas_is_one_l771_771084


namespace john_new_bench_press_l771_771101

theorem john_new_bench_press (initial_weight : ℕ) (decrease_percent : ℕ) (retain_percent : ℕ) (training_factor : ℕ) (final_weight : ℕ) 
  (h1 : initial_weight = 500)
  (h2 : decrease_percent = 80)
  (h3 : retain_percent = 20)
  (h4 : training_factor = 3)
  (h5 : final_weight = initial_weight * retain_percent / 100 * training_factor) : 
  final_weight = 300 := 
by sorry

end john_new_bench_press_l771_771101


namespace eval_neg_pow_l771_771384

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l771_771384


namespace vector_dot_product_l771_771432

-- Define vectors and their magnitudes
variables {α : Type*} [inner_product_space ℝ α]
variables (a b c : α)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1)
variables (cond : 3 • a + 4 • b + 5 • c = 0)

-- Statement to prove
theorem vector_dot_product : 
  a ⬝ (b + c) = - (3 / 5) :=
sorry

end vector_dot_product_l771_771432


namespace intersection_A_B_l771_771889

section
  def A : Set ℤ := {-2, 0, 1}
  def B : Set ℤ := {x | x^2 > 1}
  theorem intersection_A_B : A ∩ B = {-2} := 
  by
    sorry
end

end intersection_A_B_l771_771889


namespace math_problem_l771_771512

theorem math_problem 
  (a : ℤ) 
  (h_a : a = -1) 
  (b : ℚ) 
  (h_b : b = 0) 
  (c : ℕ) 
  (h_c : c = 1)
  : a^2024 + 2023 * b - c^2023 = 0 := by
  sorry

end math_problem_l771_771512


namespace proportional_function_ratio_l771_771176

-- Let k be a constant, and y = kx be a proportional function.
-- We know that f(1) = 3 and f(a) = b where b ≠ 0.
-- We want to prove that a / b = 1 / 3.

theorem proportional_function_ratio (a b k : ℝ) :
  (∀ x, x = 1 → k * x = 3) →
  (∀ x, x = a → k * x = b) →
  b ≠ 0 →
  a / b = 1 / 3 :=
by
  intros h1 h2 h3
  -- the proof will follow but is not required here
  sorry

end proportional_function_ratio_l771_771176


namespace evaluate_pow_l771_771365

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l771_771365


namespace mean_value_theorem_relation_l771_771481

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  2 * real.log x - (1 / 2) * a * x^2 + (2 - a) * x

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  (2 / x) - a * x + (2 - a)

lemma monotonicity (a : ℝ) :
  ((∀ x, 0 < x → 0 ≤ a → f_prime x a > 0) ∧ (∀ x, 0 < x → a > 0 → (f_prime x a > 0 ↔ x < 2 / a) ∧ (f_prime x a < 0 ↔ x > 2 / a))) :=
by sorry

theorem mean_value_theorem_relation (a x1 x2 x0 : ℝ) (hpos_a : 0 < a) (hpos_x1 : 0 < x1) (hpos_x2 : 0 < x2) (hlt : x1 < x2) :
  (f x2 a - f x1 a = f_prime x0 a * (x2 - x1)) →
  f_prime ((x1 + x2) / 2) a < f_prime x0 a :=
by sorry

end mean_value_theorem_relation_l771_771481


namespace GCF_of_LCMs_l771_771130

def GCF : ℕ → ℕ → ℕ := Nat.gcd
def LCM : ℕ → ℕ → ℕ := Nat.lcm

theorem GCF_of_LCMs :
  GCF (LCM 9 21) (LCM 10 15) = 3 :=
by
  sorry

end GCF_of_LCMs_l771_771130


namespace abs_pi_sub_abs_pi_sub_3_eq_3_l771_771779

theorem abs_pi_sub_abs_pi_sub_3_eq_3 : abs(π - abs(π - 3)) = 3 :=
by
  -- The proof would be added here
  sorry

end abs_pi_sub_abs_pi_sub_3_eq_3_l771_771779


namespace max_value_a_l771_771058

theorem max_value_a (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 :=
by
  intro h
  have key_inequality : |a - 2| ≥ a :=
    calc
      |a - 2| = |a - 2| : rfl
      _ ≥ (λ x, |x - 2| + |x - a|) a : by { apply h, use a } -- Apply given condition at x = a
      _ ≥ a : by { exact h a }
  sorry

end max_value_a_l771_771058


namespace T_has_desired_roots_l771_771572

noncomputable def construct_polynomial_T (P Q R : Polynomial ℝ) (S : Polynomial ℝ)
  (hS : S = P.comp (X ^ 3) + X * Q.comp (X ^ 3) + (X ^ 2) * R.comp (X ^ 3))
  (h_roots : ∃ (x : ℕ → ℝ), ∀ i, i < root_count S → S.eval (x i) = 0 ∧ ∀ i j, i ≠ j → x i ≠ x j) :
  Polynomial ℝ :=
  P.comp (X ^ 3) ^ 3 + X * Q.comp (X ^ 3) ^ 3 + (X ^ 2) * R.comp (X ^ 3) ^ 3 - 3 * P.comp (X ^ 3) * Q.comp (X ^ 3) * R.comp (X ^ 3)

theorem T_has_desired_roots (P Q R : Polynomial ℝ) (S : Polynomial ℝ)
  (hS : S = P.comp (X ^ 3) + X * Q.comp (X ^ 3) + (X ^ 2) * R.comp (X ^ 3))
  (h_roots : ∃ (x : ℕ → ℝ), ∀ i, i < root_count S → S.eval (x i) = 0 ∧ ∀ i j, i ≠ j → x i ≠ x j) :
  ∃ (x : ℕ → ℝ), ∀ i, i < root_count S → (construct_polynomial_T P Q R S hS h_roots).eval (x i ^ 3) = 0 := 
sorry

end T_has_desired_roots_l771_771572


namespace cost_per_pound_of_sausages_l771_771549

/-- Jake buys 2-pound packages of sausages. He buys 3 packages. He pays $24. 
To find the cost per pound of sausages. --/
theorem cost_per_pound_of_sausages 
  (pkg_weight : ℕ) 
  (num_pkg : ℕ) 
  (total_cost : ℕ) 
  (cost_per_pound : ℕ) 
  (h_pkg_weight : pkg_weight = 2) 
  (h_num_pkg : num_pkg = 3) 
  (h_total_cost : total_cost = 24) 
  (h_total_weight : num_pkg * pkg_weight = 6) :
  total_cost / (num_pkg * pkg_weight) = cost_per_pound :=
sorry

end cost_per_pound_of_sausages_l771_771549


namespace num_integers_with_digit_sum_seventeen_l771_771897

-- Define a function to calculate the sum of the digits of an integer
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (λ acc d, acc + d) 0

-- Define the main proposition
theorem num_integers_with_digit_sum_seventeen : 
  (finset.filter (λ n, sum_of_digits n = 17) (finset.Icc 400 600)).card = 13 := 
sorry

end num_integers_with_digit_sum_seventeen_l771_771897


namespace distance_between_skew_lines_l771_771953

noncomputable def distance_skew_lines (a : ℝ) : ℝ :=
  let A := (-2 * a, -2 * a, 0)
  let B := (2 * a, -2 * a, 0)
  let C := (2 * a, 2 * a, 0)
  let D := (-2 * a, 2 * a, 0)
  let S := (0, 0, 8 * a)
  let M := ((-2 * a + 0) / 2, (-2 * a + 0) / 2, (0 + 8 * a) / 2)
  let N := ((2 * a + 0) / 2, (2 * a + 0) / 2, (0 + 8 * a) / 2)
  let BN := (N.1 - B.1, N.2 - B.2, N.3 - B.3)
  let DM := (M.1 - D.1, M.2 - D.2, M.3 - D.3)
  let n  := (BN.2 * DM.3 - BN.3 * DM.2, BN.3 * DM.1 - BN.1 * DM.3, BN.1 * DM.2 - BN.2 * DM.1)
  let MN := (N.1 - M.1, N.2 - M.2, N.3 - M.3)
  let dot_prod := MN.1 * n.1 + MN.2 * n.2 + MN.3 * n.3
  let mag_n := sqrt (n.1 * n.1 + n.2 * n.2 + n.3 * n.3)
  (abs dot_prod) / mag_n

theorem distance_between_skew_lines (a : ℝ) :
  distance_skew_lines a = (4 * a * sqrt 10) / 5 := sorry

end distance_between_skew_lines_l771_771953


namespace a_2003_value_l771_771037

def sequence (n : ℕ) : ℤ :=
  if n = 1 then 3
  else if n = 2 then 5
  else sequence (n - 1) - sequence (n - 2)

theorem a_2003_value : sequence 2003 = -5 :=
  sorry

end a_2003_value_l771_771037


namespace blocks_to_beach_l771_771548

theorem blocks_to_beach (melt_time_in_minutes : ℕ) (block_length_in_miles : ℚ) (speed_in_miles_per_hour : ℚ)
  (h1 : melt_time_in_minutes = 10)
  (h2 : block_length_in_miles = 1 / 8)
  (h3 : speed_in_miles_per_hour = 12) :
  let melt_time_in_hours := melt_time_in_minutes / 60
      distance_in_miles := speed_in_miles_per_hour * melt_time_in_hours
      blocks := distance_in_miles / block_length_in_miles
  in blocks = 16 :=
by
  let melt_time_in_hours := melt_time_in_minutes / 60
  let distance_in_miles := speed_in_miles_per_hour * melt_time_in_hours
  let blocks := distance_in_miles / block_length_in_miles
  -- Proof steps would go here.
  have : blocks = 16 := sorry
  exact this

end blocks_to_beach_l771_771548


namespace xiaoming_correct_probability_l771_771790

theorem xiaoming_correct_probability :
  let letters := { "e", "g₁", "g₂" }
  let correct_arrangement := [ "e", "g₁", "g₂" ]
  let all_arrangements := { ["e", "g₁", "g₂"], ["g₁", "e", "g₂"], ["g₁", "g₂", "e"] }
  ∃ prob : ℚ, prob = 1 / 3 ∧ prob = (1 : ℚ) / (all_arrangements.to_finset.card: ℚ)
:= 
sorry

end xiaoming_correct_probability_l771_771790


namespace seven_books_cost_l771_771205

-- Given condition: Three identical books cost $45
def three_books_cost (cost_per_book : ℤ) := 3 * cost_per_book = 45

-- Question to prove: The cost of seven identical books is $105
theorem seven_books_cost (cost_per_book : ℤ) (h : three_books_cost cost_per_book) : 7 * cost_per_book = 105 := 
sorry

end seven_books_cost_l771_771205


namespace inequality_solution_l771_771606

theorem inequality_solution (x : ℝ) (h : x > -4/3) : 2 - 1 / (3 * x + 4) < 5 :=
sorry

end inequality_solution_l771_771606


namespace percentage_of_solution_A_l771_771139

variables (P : ℝ) -- The percentage of liquid X in solution A (in decimal form)
variables (wA wB : ℝ) -- Weights of solutions A and B respectively
variables (pB : ℝ) -- Percentage of liquid X in solution B (in decimal form)
variables (p_result : ℝ) -- Percentage of liquid X in the resulting solution (in decimal form)
variables (w_result : ℝ) -- Total weight of the resulting solution

-- Conditions given in the problem
def condition1 : Prop := P * wA + pB * wB = p_result * w_result
def solutionA_weight : Prop := wA = 300
def solutionB_weight : Prop := wB = 700
def percentageB : Prop := pB = 0.018
def resulting_solution_weight : Prop := w_result = 1000
def resulting_solution_percentage : Prop := p_result = 0.015

-- We need to prove that given the conditions, P = 0.008.
theorem percentage_of_solution_A
  (h1 : condition1)
  (h2 : solutionA_weight)
  (h3 : solutionB_weight)
  (h4 : percentageB)
  (h5 : resulting_solution_weight)
  (h6 : resulting_solution_percentage) :
  P = 0.008 :=
sorry

end percentage_of_solution_A_l771_771139


namespace minimum_number_of_distinct_complex_solutions_l771_771117

open Polynomial

theorem minimum_number_of_distinct_complex_solutions 
  (P Q R : Polynomial ℂ)
  (hP_deg : P.degree = 4)
  (hQ_deg : Q.degree = 4)
  (hR_deg : R.degree = 8)
  (hP_const : P.coeff 0 = 1)
  (hQ_const : Q.coeff 0 = 3)
  (hR_const : R.coeff 0 = 5) :
  (P * Q - (R + 1)).roots.to_finset.card = 1 := 
sorry

end minimum_number_of_distinct_complex_solutions_l771_771117


namespace probability_of_first_hearts_and_second_clubs_l771_771670

noncomputable def probability_first_hearts_second_clubs : ℚ :=
  let total_cards := 52
  let hearts_count := 13
  let clubs_count := 13
  let probability_first_hearts := hearts_count / total_cards
  let probability_second_clubs_given_first_hearts := clubs_count / (total_cards - 1)
  probability_first_hearts * probability_second_clubs_given_first_hearts

theorem probability_of_first_hearts_and_second_clubs :
  probability_first_hearts_second_clubs = 13 / 204 :=
by
  sorry

end probability_of_first_hearts_and_second_clubs_l771_771670


namespace eval_neg_pow_l771_771380

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l771_771380


namespace math_problem_l771_771217

variables {A B : Type} [Fintype A] [Fintype B]
          (p1 p2 : ℝ) (h1 : 1/2 < p1) (h2 : p1 < p2) (h3 : p2 < 1)
          (nA : ℕ) (hA : nA = 3) (nB : ℕ) (hB : nB = 3)

noncomputable def E_X : ℝ := nA * p1
noncomputable def E_Y : ℝ := nB * p2

noncomputable def D_X : ℝ := nA * p1 * (1 - p1)
noncomputable def D_Y : ℝ := nB * p2 * (1 - p2)

theorem math_problem :
  E_X p1 nA = 3 * p1 →
  E_Y p2 nB = 3 * p2 →
  D_X p1 nA = 3 * p1 * (1 - p1) →
  D_Y p2 nB = 3 * p2 * (1 - p2) →
  E_X p1 nA < E_Y p2 nB ∧ D_X p1 nA > D_Y p2 nB :=
by
  sorry

end math_problem_l771_771217


namespace prob_diff_colors_sum_ge_4_l771_771939

-- Definition of the number of balls and their corresponding labels and colors
constant red_labels : Finset ℕ := {1, 2, 3}
constant blue_labels : Finset ℕ := {1, 2}

-- Definition of the probability of the given event
theorem prob_diff_colors_sum_ge_4 : (probability : ℚ) = 3 / 10 := by
  sorry

end prob_diff_colors_sum_ge_4_l771_771939


namespace area_difference_calculation_l771_771718

noncomputable def area_difference_rectangle_to_circle (length width : ℝ) (r : ℝ) : ℝ :=
  let original_area := length * width
  let circular_area := π * r ^ 2
  circular_area - original_area

theorem area_difference_calculation : 
  area_difference_rectangle_to_circle 60 8 (68 / π) ≈ 992 := sorry

end area_difference_calculation_l771_771718


namespace pair_points_no_intersection_l771_771848

-- Definition of the problem
noncomputable def pairwise_non_intersecting (T : Set (ℝ × ℝ)) : Prop :=
∀ p q r s ∈ T, p ≠ q → r ≠ s → ¬ (segments_intersect p q r s)

-- Statement of the mathematical problem
theorem pair_points_no_intersection (T : Set (ℝ × ℝ)) (hT : ∃ n, T.card = 2 * n) : 
  ∃ pairs : Finset (Finset (ℝ × ℝ)), 
    (pairs.card = hT.some) ∧ 
    (∀ p ∈ pairs, p.card = 2) ∧
    pairwise_non_intersecting T := 
sorry

end pair_points_no_intersection_l771_771848


namespace part_a_part_b_l771_771685

variable (p : ℕ → ℕ → ℝ) (A : ℕ → ℝ)

noncomputable def inequality_pmk (m k : ℕ) : Prop :=
  (2 / 3) * p m k ≥ p m (k + 2)

noncomputable def limit_exists_and_gt_zero (A0 : ℝ) : Prop :=
  ∃ A0, (A0 = limit (λ m, p m 0)) ∧ (A0 ≥ 1 / 3)

theorem part_a (m k : ℕ) : inequality_pmk p m k :=
by
sorry

theorem part_b : limit_exists_and_gt_zero (A 0) :=
by
sorry

end part_a_part_b_l771_771685


namespace largest_prime_value_of_f_l771_771341

def f (x : ℝ) : ℝ := 5 * x^4 - 12 * x^3 + 30 * x^2 - 12 * x + 5

theorem largest_prime_value_of_f :
  ∀ (x_1 : ℝ) (p : ℕ), (f x_1 = p) → (0 ≤ x_1) → (prime p) → (p ≤ 5) :=
by
  sorry

end largest_prime_value_of_f_l771_771341


namespace complex_conjugate_l771_771809

open Complex

theorem complex_conjugate (z : ℂ) (h : z = (-3 + I) / (2 + I)) : conj z = -1 - I :=
sorry

end complex_conjugate_l771_771809


namespace range_of_a_l771_771988

variable {a : ℝ}

def f (x : ℝ) : ℝ := Real.exp x + a * x

theorem range_of_a (h : ∀ x > 0, 0 ≤ Real.exp x + a) : a ≥ -1 :=
by sorry

end range_of_a_l771_771988


namespace lattice_points_on_curve_l771_771905

def is_lattice_point (x y : ℤ) : Prop :=
  x^2 - y^2 = 65

def lattice_points_count : ℕ :=
  { p : ℤ × ℤ // is_lattice_point p.1 p.2 }.card

theorem lattice_points_on_curve :
  lattice_points_count = 8 :=
sorry

end lattice_points_on_curve_l771_771905


namespace simplify_fraction_l771_771602

theorem simplify_fraction (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (15 * x^2 * y^4 * z^2) / (9 * x * y^3 * z) = 10 := 
by
  sorry

end simplify_fraction_l771_771602


namespace yola_past_weight_l771_771683

variable (W Y Y_past : ℕ)

-- Conditions
def condition1 : Prop := W = Y + 30
def condition2 : Prop := W = Y_past + 80
def condition3 : Prop := Y = 220

-- Theorem statement
theorem yola_past_weight : condition1 W Y → condition2 W Y_past → condition3 Y → Y_past = 170 :=
by
  intros h_condition1 h_condition2 h_condition3
  -- Placeholder for the proof, not required in the solution
  sorry

end yola_past_weight_l771_771683


namespace proof_speed_of_goods_train_l771_771267

def speed_of_goods_train (Vm_kmph Vg_kmph Vr_mps distance length time : ℝ) : Prop :=
  Vm_kmph = 50 ∧ 
  distance = 280 ∧ 
  time = 9 ∧ 
  Vr_mps = distance / time ∧ 
  Vm_kmph * (1000 / 3600) = Vm ∧ 
  Vg = Vr_mps - Vm ∧ 
  Vg_kmph = Vg * (3600 / 1000) → 
  abs (Vg_kmph - 61.99) < 0.01

-- Statement (without proof)
theorem proof_speed_of_goods_train : speed_of_goods_train 50 _ _ 280 _ 9 := sorry

end proof_speed_of_goods_train_l771_771267


namespace integral_binomial_expansion_l771_771088

def binomialTermCoeff : ℤ := 10
def integralFrom1ToA (a : ℤ) : ℝ := ∫ x in 1..a, x⁻¹

theorem integral_binomial_expansion :
  integralFrom1ToA binomialTermCoeff = Real.log 10 := by
  sorry

end integral_binomial_expansion_l771_771088


namespace area_of_triangle_formed_by_centers_of_hexagons_at_second_level_is_48sqrt3_l771_771780

def hexagon_side_length : ℕ := 2
def surrounding_hexagons : ℕ := 6
def smaller_hexagon_side_length : ℕ := 1

theorem area_of_triangle_formed_by_centers_of_hexagons_at_second_level_is_48sqrt3 :
  ∃ (a : ℝ), a = 48 * real.sqrt 3 := 
sorry

end area_of_triangle_formed_by_centers_of_hexagons_at_second_level_is_48sqrt3_l771_771780


namespace probability_of_first_hearts_and_second_clubs_l771_771672

noncomputable def probability_first_hearts_second_clubs : ℚ :=
  let total_cards := 52
  let hearts_count := 13
  let clubs_count := 13
  let probability_first_hearts := hearts_count / total_cards
  let probability_second_clubs_given_first_hearts := clubs_count / (total_cards - 1)
  probability_first_hearts * probability_second_clubs_given_first_hearts

theorem probability_of_first_hearts_and_second_clubs :
  probability_first_hearts_second_clubs = 13 / 204 :=
by
  sorry

end probability_of_first_hearts_and_second_clubs_l771_771672


namespace evaluate_pow_l771_771359

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l771_771359


namespace general_formula_an_sum_Tn_l771_771887

variable (n : ℕ) (S : ℕ → ℕ)
variable (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)

-- Condition: The sum of the first n terms of the sequence {a_n} is given by S_n = 2^(n + 2) - 4
axiom Sn_def : ∀ n, S n = 2^(n + 2) - 4

-- To prove: The general formula for the sequence {a_n} is a_n = 2^(n + 1)
theorem general_formula_an (h1 : ∀ n, S n = 2^(n + 2) - 4) (n : ℕ) : a n = 2^(n + 1) :=
  sorry

-- Condition: b_n = a_n * log_2 a_n
axiom bn_def : ∀ n, b n = (a n) * (Int.ofNat ((nat.log 2).toNat (a n)))

-- To prove: The sum of the first n terms of the sequence {b_n} is T_n = n * 2^(n + 2)
theorem sum_Tn (h2 : ∀ n, b n = a n * (Int.ofNat ((nat.log 2).toNat (a n)))) (n : ℕ) : T n = n * 2^(n + 2) :=
  sorry

end general_formula_an_sum_Tn_l771_771887


namespace anagram_count_Abracadabra_l771_771895

theorem anagram_count_Abracadabra : 
  let n := 11
  let f_A := 5
  let f_B := 2
  let f_R := 2
  let f_C := 1
  let f_D := 1
  ∃ k : ℕ, 
  k = (Nat.factorial n) / ((Nat.factorial f_A) * (Nat.factorial f_B) * (Nat.factorial f_R) * (Nat.factorial f_C) * (Nat.factorial f_D)) ∧
  k = 83160 :=
begin
  sorry
end

end anagram_count_Abracadabra_l771_771895


namespace projection_correct_l771_771893

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-2, 4)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude_squared (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dp := dot_product u v
  let ms := magnitude_squared v
  (dp / ms * v.1, dp / ms * v.2)

theorem projection_correct :
  projection vector_a vector_b = (-3 / 5, 6 / 5) :=
  sorry

end projection_correct_l771_771893


namespace relationship_in_size_l771_771434

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 2.1
noncomputable def c : ℝ := Real.log (1.5) / Real.log (2)

theorem relationship_in_size : b > a ∧ a > c := by
  sorry

end relationship_in_size_l771_771434


namespace drawing_black_ball_impossible_l771_771072

theorem drawing_black_ball_impossible (h1 : ∀ x, x ∈ {red_ball_1, red_ball_2} → x ≠ black_ball) : 
  event_occurs = false :=
by
  -- Given conditions:
  -- h1: The bag only contains red balls, any ball drawn from the bag cannot be a black ball.

  -- We need to show that the event of drawing a black ball is impossible.
  -- More formally, we need to show that the probability of drawing a black ball is 0.
  sorry

end drawing_black_ball_impossible_l771_771072


namespace max_area_PQR_max_area_incenter_triangle_l771_771537

/-- Given conditions -/
def unit_area_equilateral_triangle (ABC : Triangle) : Prop := 
  (ABC.is_equilateral) ∧ (ABC.area = 1)

def external_equilateral_triangles (ABC APB BQC CRA : Triangle) : Prop := 
  (APB.is_equilateral ∧ BQC.is_equilateral ∧ CRA.is_equilateral) ∧ 
  (∠APB = 60 ∧ ∠BQC = 60 ∧ ∠CRA = 60)

/-- The proof problem statements -/
theorem max_area_PQR (ABC APB BQC CRA P Q R : Triangle) 
  (h1 : unit_area_equilateral_triangle ABC)
  (h2 : external_equilateral_triangles ABC APB BQC CRA) : 
  area (Triangle.mk P Q R) = 4 * sqrt 3 :=
sorry

theorem max_area_incenter_triangle (ABC APB BQC CRA : Triangle)
  (M N O : Point) 
  (h1 : unit_area_equilateral_triangle ABC)
  (h2 : external_equilateral_triangles ABC APB BQC CRA)
  (h3 : M = incenter APB)
  (h4 : N = incenter BQC)
  (h5 : O = incenter CRA) : 
  area (Triangle.mk M N O) = 1 :=
sorry

end max_area_PQR_max_area_incenter_triangle_l771_771537


namespace sufficient_condition_not_monotonic_l771_771483

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 4 * a * x - Real.log x

def sufficient_not_monotonic (a : ℝ) : Prop :=
  (a > 1 / 6) ∨ (a < -1 / 2)

theorem sufficient_condition_not_monotonic (a : ℝ) :
  sufficient_not_monotonic a → ¬(∀ x y : ℝ, 1 < x ∧ x < 3 ∧ 1 < y ∧ y < 3 ∧ x ≠ y → ((f a x - f a y) / (x - y) ≥ 0 ∨ (f a y - f a x) / (y - x) ≥ 0)) :=
by
  sorry

end sufficient_condition_not_monotonic_l771_771483


namespace solve_f_x_eq_2_l771_771027

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 ^ x else -x

theorem solve_f_x_eq_2 (x : ℝ) : f x = 2 → x = 1 := by
  sorry

end solve_f_x_eq_2_l771_771027


namespace tire_price_l771_771632

theorem tire_price (x : ℝ) (h1 : 2 * x + 5 = 185) : x = 90 :=
by
  sorry

end tire_price_l771_771632


namespace cos_ACB_in_terms_of_cos_CAD_cos_CBD_and_theta_l771_771540

-- Representing the conditions
variables (A B C D : ℝ) -- points in space, represented as real numbers for simplicity
variables (θ x y : ℝ) -- angles and their cosines

-- Conditions in the problem
-- In the tetrahedron ABCD
def angle_ADB := 90 -- angle ADB is 90 degrees
def angle_BDC := 90 -- angle BDC is 90 degrees
def angle_ADC := θ -- angle ADC is θ
def cos_CAD := x    -- x = cos(angle CAD)
def cos_CBD := y    -- y = cos(angle CBD)

-- Proving the resulting expression
theorem cos_ACB_in_terms_of_cos_CAD_cos_CBD_and_theta :
  cos (angle A C B) = sin^2 θ :=
by
  sorry

end cos_ACB_in_terms_of_cos_CAD_cos_CBD_and_theta_l771_771540


namespace find_length_AD_l771_771959

-- Given data and conditions
def triangle_ABC (A B C D : Type) : Prop := sorry
def angle_bisector_AD (A B C D : Type) : Prop := sorry
def length_BD : ℝ := 40
def length_BC : ℝ := 45
def length_AC : ℝ := 36

-- Prove that AD = 320 units
theorem find_length_AD (A B C D : Type)
  (h1 : triangle_ABC A B C D)
  (h2 : angle_bisector_AD A B C D)
  (h3 : length_BD = 40)
  (h4 : length_BC = 45)
  (h5 : length_AC = 36) :
  ∃ x : ℝ, x = 320 :=
sorry

end find_length_AD_l771_771959


namespace common_tangent_a_l771_771029

noncomputable def f (x : ℝ) := x^2 - 1
noncomputable def g (x : ℝ) (a : ℝ) := a * Real.log x

theorem common_tangent_a :
  (∀ (a : ℝ) (x : ℝ), a ≠ 0 → Real.log x ≠ 0 → x^2 - 1 = 0 → x * a * Real.log x = 0) →
  f 1 = 0 → g 1 2 = 0 → (∃ x, f'(x) = g'(x)) :=
by
  sorry

end common_tangent_a_l771_771029


namespace kim_change_l771_771104

def meal_cost := 10
def drink_cost := 2.5
def tip_percentage := 0.2
def total_payment := 20

theorem kim_change : total_payment - (meal_cost + drink_cost + (meal_cost + drink_cost) * tip_percentage) = 5 := 
by 
  have total_cost := meal_cost + drink_cost 
  have tip := total_cost * tip_percentage 
  have total_amount := total_cost + tip 
  have change := total_payment - total_amount 
  exact change

end kim_change_l771_771104


namespace g_even_l771_771960

def g (x : ℝ) : ℝ := ( (3^x - 1) / (3^x + 1) ) ^ 2

theorem g_even : ∀ (x : ℝ), g (-x) = g x :=
by
  sorry

end g_even_l771_771960


namespace range_of_a_l771_771442

variable {x a : ℝ}

def p (x : ℝ) := x^2 - 8 * x - 20 > 0
def q (a : ℝ) (x : ℝ) := x^2 - 2 * x + 1 - a^2 > 0

theorem range_of_a (h₀ : ∀ x, p x → q a x) (h₁ : a > 0) : 0 < a ∧ a ≤ 3 := 
by 
  sorry

end range_of_a_l771_771442


namespace lattice_points_on_hyperbola_l771_771910

theorem lattice_points_on_hyperbola :
  ∃ (s : Finset (ℤ × ℤ)), 
  (∀ p ∈ s, let (x, y) := p in x^2 - y^2 = 65) ∧ 
  (4 : Finset.card s) :=
begin
  sorry
end

end lattice_points_on_hyperbola_l771_771910


namespace polar_equation_of_circle_max_op_oq_l771_771956

-- Define the parametric equations of the circle.
def parametric_circle (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos φ, Real.sin φ)

-- Problem (I): Proving the polar equation of the circle.
theorem polar_equation_of_circle :
  ∀ θ : ℝ, ∃ ρ : ℝ, (ρ = 2 * Real.cos θ ∧ (1 + Real.cos θ)^2 + (Real.sin θ)^2 = 1) :=
by
  sorry

-- Problem (II): Finding the maximum value of |OP| + |OQ| when ∠POQ = π/3.
theorem max_op_oq :
  ∀ θ : ℝ, 
  θ > -π/2 ∧ θ < π/6 →
  ∃ max_value : ℝ, (max_value = 2 * Real.sqrt 3 ∧ ∀ P Q : ℝ × ℝ, 
  (parametric_circle θ = P) ∧ 
  (parametric_circle (θ + π/3) = Q) → 
  (P.1^2 + P.2^2 = 1 ∧ Q.1^2 + Q.2^2 = 1 ∧
   (Real.sqrt (P.1^2 + P.2^2) + Real.sqrt (Q.1^2 + Q.2^2)) ≤ 2 * Real.sqrt 3)) :=
by
  sorry

end polar_equation_of_circle_max_op_oq_l771_771956


namespace cone_volume_arith_prog_l771_771788

variable {r h s : ℝ}
variable (d : ℝ)

-- Definition of arithmetic progression relation
def arithmetic_progression := s = (r + h) / 2

-- Volume of a cone in terms of r and h
def volume_cone := (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_arith_prog (h d : ℝ) (h_arith : arithmetic_progression d) : 
  volume_cone r h = (1 / 3) * Real.pi * (r^3 + 2 * d * r^2) := 
by
  sorry -- Proof omitted

end cone_volume_arith_prog_l771_771788


namespace evaluate_neg_64_exp_4_over_3_l771_771369

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l771_771369


namespace ratio_floor_to_total_l771_771584

noncomputable def students_in_gym : ℕ := 26
noncomputable def students_on_bleachers : ℝ := 4.3
noncomputable def students_on_floor : ℝ := 10.6
noncomputable def students_on_chairs_benches : ℝ := 11.1

theorem ratio_floor_to_total (A B C : ℝ) (students_in_gym : ℕ) :
  round(A) + round(B) + round(C) = students_in_gym →
  students_on_floor = B →
  round(B) / students_in_gym = 11 / 26 :=
by {
  intros h hB,
  sorry
}

-- Definitions
def A := students_on_bleachers
def B := students_on_floor
def C := students_on_chairs_benches

-- Invoke the theorem with specific values
example : ratio_floor_to_total A B C students_in_gym :=
by {
  -- Assuming A = 4.3, B = 10.6, C = 11.1, students_in_gym = 26
  have : round(A) + round(B) + round(C) = students_in_gym := by {
    norm_num [A, B, C, students_in_gym]
  },
  exact_mode,
  exact this
}

end ratio_floor_to_total_l771_771584


namespace sum_even_divisors_210_l771_771699

theorem sum_even_divisors_210 : 
  let even_divisors := {d ∈ {n | n ∣ 210} | ∃ k, 2 * k = d}
  ∑ d in even_divisors, d = 384 :=
by
  sorry

end sum_even_divisors_210_l771_771699


namespace coronavirus_scientific_notation_l771_771170

def new_coronavirus_diameter : ℝ := 0.00000003

theorem coronavirus_scientific_notation : new_coronavirus_diameter = 3 * 10^(-8) :=
sorry

end coronavirus_scientific_notation_l771_771170


namespace inequality_proof_l771_771439

variables {R : Type*} [OrderedRing R] [LinearOrderedField R]

theorem inequality_proof (a : ℕ → R) (m n : ℕ)
  (hnm : n - m ≥ 3) (ha_pos : ∀ i, 1 ≤ i ∧ i ≤ n - m → 0 < a i) :
  ∏ i in finset.range (n - m), (a i ^ n - a i ^ m + n - m) ≥ (∑ i in finset.range (n - m), a i) ^ (n - m) :=
begin
  sorry
end

end inequality_proof_l771_771439


namespace part1_intersection_part2_subset_l771_771923

variable {α : Type*} (A B : Set α) (m : ℝ)

def setA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 6 }

def setB (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 3 * m - 1 }

theorem part1_intersection (m : ℝ) (h : m = 2) : 
  A = setA ∧ B = setB m → A ∩ B = { x | 3 ≤ x ∧ x ≤ 5 } :=
sorry

theorem part2_subset (h : B ⊆ A) : m ≤ 7 / 3 :=
sorry

end part1_intersection_part2_subset_l771_771923


namespace odd_function_evaluation_l771_771985

theorem odd_function_evaluation (f : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) (h : f (-3) = -2) : f 3 + f 0 = 2 :=
by 
  sorry

end odd_function_evaluation_l771_771985


namespace largest_integer_solution_l771_771691

theorem largest_integer_solution (x : ℤ) : 
  x < (92 / 21 : ℝ) → ∀ y : ℤ, y < (92 / 21 : ℝ) → y ≤ x :=
by
  sorry

end largest_integer_solution_l771_771691


namespace last_digit_2008_pow_2005_l771_771692

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_2008_pow_2005 : last_digit (2008 ^ 2005) = 8 :=
by
  sorry

end last_digit_2008_pow_2005_l771_771692


namespace tangent_lines_count_l771_771480

def f (x : ℝ) : ℝ := x^3

theorem tangent_lines_count :
  (∃ x : ℝ, deriv f x = 3) ∧ 
  (∃ y : ℝ, deriv f y = 3 ∧ y ≠ x) := 
by
  -- Since f(x) = x^3, its derivative is f'(x) = 3x^2
  -- We need to solve 3x^2 = 3
  -- Therefore, x^2 = 1 and x = ±1
  -- Thus, there are two tangent lines
  sorry

end tangent_lines_count_l771_771480


namespace number_of_valid_x0_l771_771980

noncomputable def sequence (x : ℝ) :=
  if 3 * x < 1 then 3 * x
  else if 3 * x < 2 then 3 * x - 1
  else 3 * x - 2

def isPeriodic (x : ℝ) : Prop :=
  ∀ n : ℕ, sequence^[n] x = sequence^[n + 4] x

theorem number_of_valid_x0 : ∃ (count : ℕ), count = 27 ∧ 
  (∀ (x0 : ℝ), 0 ≤ x0 ∧ x0 < 1 → isPeriodic x0 ↔ x0 ∈ finset.range 27.map (λ k, k / 81)) :=
sorry

end number_of_valid_x0_l771_771980


namespace pos_difference_of_b_l771_771135

def f (n : ℤ) : ℤ :=
  if n < 0 then n ^ 2 - 4 else 2 * n - 24

theorem pos_difference_of_b (b : ℤ) : 
  (f (-3) + f (3) + f b = 2) → 
  abs (((39 : ℤ)/2 : ℚ) + (Real.sqrt 19) - b) = (Real.sqrt 19 + (39 / 2 : ℚ)) :=
begin
  sorry,
end

end pos_difference_of_b_l771_771135


namespace johns_total_monthly_payment_l771_771724

-- Define the various costs
def base_cost : ℝ := 25
def text_msg_cost : ℝ := 0.08
def extra_minute_cost : ℝ := 0.15

-- Define John's usage
def text_messages_sent : ℝ := 250
def hours_talked : ℝ := 50.75
def included_hours : ℝ := 50

-- Calculate costs
def text_cost := text_messages_sent * text_msg_cost
def extra_time := hours_talked - included_hours
def extra_minutes := if extra_time > 0 then extra_time * 60 else 0
def extra_minutes_cost := extra_minutes * extra_minute_cost

-- Calculate total cost
def total_cost : ℝ := base_cost + text_cost + extra_minutes_cost

-- Statement to prove
theorem johns_total_monthly_payment : total_cost = 51.75 := by
  sorry

end johns_total_monthly_payment_l771_771724


namespace basketball_probability_l771_771675

variable (A B : Event)
variable (P : Probability)
variable (P_A : P A = 0.8)
variable (P_B' : P (¬ B) = 0.1)
variable (ind : independent A B)

theorem basketball_probability :
  (P (A ∩ B) = 0.72) ∧ 
  (P (A ∩ (¬ B)) + P ((¬ A) ∩ B) = 0.26) :=
by
  sorry

end basketball_probability_l771_771675


namespace value_of_2alpha_minus_beta_l771_771863

theorem value_of_2alpha_minus_beta (a β : ℝ) (h1 : 3 * Real.sin a - Real.cos a = 0) 
    (h2 : 7 * Real.sin β + Real.cos β = 0) (h3 : 0 < a ∧ a < Real.pi / 2) 
    (h4 : Real.pi / 2 < β ∧ β < Real.pi) : 
    2 * a - β = -3 * Real.pi / 4 := 
sorry

end value_of_2alpha_minus_beta_l771_771863


namespace expected_value_unfair_die_l771_771831

theorem expected_value_unfair_die :
  let p8 := 3 / 8
  let p1_7 := (1 - p8) / 7
  let E := p1_7 * (1 + 2 + 3 + 4 + 5 + 6 + 7) + p8 * 8
  E = 5.5 := by
  sorry

end expected_value_unfair_die_l771_771831


namespace remainder_when_divided_by_n_l771_771126

variables (n : ℕ) (a b c : ℤ)

def is_invertible_mod (m x : ℤ) : Prop := ∃ y, (x * y) % m = 1

#check is_invertible_mod

theorem remainder_when_divided_by_n 
  (hn : n > 0)
  (hinv_a : is_invertible_mod n a)
  (hinv_b : is_invertible_mod n b)
  (hc : true) -- hc stands for "c is a constant integer"
  (h : a ≡ b⁻¹ + c [MOD n]) :
  ((a - c) * b) % n = 1 :=
by sorry

end remainder_when_divided_by_n_l771_771126


namespace rope_cutting_l771_771309

theorem rope_cutting (L total_length piece_length n remaining_length : ℕ) (h1 : total_length = 20) (h2 : piece_length = 3.8) (h3 : total_length = piece_length * n + remaining_length) (h4 : total_length < piece_length * (n + 1)) :
  n = 5 ∧ remaining_length = 1 :=
sorry

end rope_cutting_l771_771309


namespace triangle_ratio_l771_771077

-- Define the setting and objects
variable {A B C D : Type}
variable [is_point A] [is_point B] [is_point C] [is_point D]
variable {a : ℝ} (h_eq_a: distance A B = a) (h_eq_bc: distance B C = a)

-- Definitions based on conditions
def equilateral_triangle (A B C : Type) [is_point A] [is_point B] [is_point C] :=
  distance A B = distance B C ∧ distance B C = distance C A

def isosceles_triangle_with_equal_sides (B C D : Type) [is_point B] [is_point C] [is_point D] :=
  distance B C = distance B D

-- Theorem to prove the question equals the answer
theorem triangle_ratio (h_equilateral : equilateral_triangle A B C)
    (h_isosceles: isosceles_triangle_with_equal_sides B C D) :
    distance A D / distance B C = 1 :=
  sorry

end triangle_ratio_l771_771077


namespace monotonicity_of_f_range_of_a_l771_771880

open Real

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * log x + a * x^2 + 1

theorem monotonicity_of_f (a x : ℝ) (h1 : 0 < x) : 
  if a ≥ 1 then (∀ x1 x2, 0 < x1 ∧ 0 < x2 ∧ x1 < x2 → f a x1 < f a x2)
  else if a ≤ 0 then (∀ x1 x2, 0 < x1 ∧ 0 < x2 ∧ x1 < x2 → f a x1 > f a x2)
  else let c := sqrt ((1 - a) / (2 * a)) in 
  (∀ x1 x2, 0 < x1 ∧ x1 < c → 0 < x2 ∧ x2 < c → f a x1 < f a x2) ∧ 
  (∀ x1 x2, c < x1 → c < x2 ∧ x1 < x2 → f a x1 > f a x2) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, 0 < x1 ∧ 0 < x2 ∧ x1 > x2 → (f a x1 - f a x2) / (x1 - x2) ≥ 2)
  → a ≥ (sqrt 3 + 1) / 2 := sorry

end monotonicity_of_f_range_of_a_l771_771880


namespace ladder_length_l771_771733

variable (x y : ℝ)

theorem ladder_length :
  (x^2 = 15^2 + y^2) ∧ (x^2 = 24^2 + (y - 13)^2) → x = 25 := by
  sorry

end ladder_length_l771_771733


namespace sin_cos_sum_l771_771979

-- Let theta be an angle in the second quadrant
variables (θ : ℝ)
-- Given the condition tan(θ + π / 4) = 1 / 2
variable (h1 : Real.tan (θ + Real.pi / 4) = 1 / 2)
-- Given θ is in the second quadrant
variable (h2 : θ ∈ Set.Ioc (Real.pi / 2) Real.pi)

-- Prove sin θ + cos θ = - sqrt(10) / 5
theorem sin_cos_sum (h1 : Real.tan (θ + Real.pi / 4) = 1 / 2) (h2 : θ ∈ Set.Ioc (Real.pi / 2) Real.pi) :
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_cos_sum_l771_771979


namespace johnson_potatoes_l771_771553

/-- Given that Johnson has a sack of 300 potatoes, 
    gives some to Gina, twice that amount to Tom, and 
    one-third of the amount given to Tom to Anne,
    and has 47 potatoes left, we prove that 
    Johnson gave Gina 69 potatoes. -/
theorem johnson_potatoes : 
  ∃ G : ℕ, 
  ∀ (Gina Tom Anne total : ℕ), 
    total = 300 ∧ 
    total - (Gina + Tom + Anne) = 47 ∧ 
    Tom = 2 * Gina ∧ 
    Anne = (1 / 3 : ℚ) * Tom ∧ 
    (Gina + Tom + (Anne : ℕ)) = (11 / 3 : ℚ) * Gina ∧ 
    (Gina + Tom + Anne) = 253 
    ∧ total = Gina + Tom + Anne + 47 
    → Gina = 69 := sorry


end johnson_potatoes_l771_771553


namespace distance_between_midpoints_eq_one_l771_771560

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

variables (A A' B B' C C' D D' M N : ℝ × ℝ × ℝ) 
          (hA : A = (0,0,0)) (hB : B = (2,0,0)) (hC : C = (2,1,0)) (hD : D = (0,1,0))
          (hA' : A' = (0,0,12)) (hB' : B' = (2,0,10)) (hC' : C' = (2,1,20)) (hD' : D' = (0,1,24))
          (hM : M = ((A'.1 + C'.1)/2, (A'.2 + C'.2)/2, (A'.3 + C'.3)/2))
          (hN : N = ((B'.1 + D'.1)/2, (B'.2 + D'.2)/2, (B'.3 + D'.3)/2))

theorem distance_between_midpoints_eq_one :
  distance M N = 1 :=
sorry

end distance_between_midpoints_eq_one_l771_771560


namespace intersect_at_single_point_l771_771711

-- Definitions of points and lines
variables {A B C D P Q R P1 Q1 R1 : Point}
variable {l : Line}

-- Conditions from the problem
def quadrilateral (A B C D : Point) : Prop := 
  true -- Assuming the existence of a quadrilateral defined by points A, B, C, and D.

def intersection (P : Point) (l₁ l₂ : Line) : Prop := 
  true -- Assuming P is the intersection point of lines l₁ and l₂.

def midpoint (P₁ Q₁ R₁ : Point) (l : Line) : Prop := 
  true -- Assuming P1, Q1, R1 are midpoints of segments cut by respective line intersections on line l.

-- Problem statement in Lean 4
theorem intersect_at_single_point (h1 : quadrilateral A B C D)
                                  (h2 : intersection P (line_through A B) (line_through C D))
                                  (h3 : intersection Q (line_through A C) (line_through B D))
                                  (h4 : intersection R (line_through B C) (line_through A D))
                                  (h5 : midpoint P1 Q1 R1 l):
  intersects_at_single_point (line_through P P1) (line_through Q Q1) (line_through R R1) :=
sorry

end intersect_at_single_point_l771_771711


namespace antonym_word_is_rarely_l771_771314

theorem antonym_word_is_rarely : 
  (∃ word : String, word = "antonym of 26" → word = "rarely") :=
begin
  sorry
end

end antonym_word_is_rarely_l771_771314


namespace integral_of_semicircle_l771_771413

theorem integral_of_semicircle :
  ∫ x in -real.sqrt 2..real.sqrt 2, real.sqrt (2 - x^2) = real.pi :=
by
  sorry

end integral_of_semicircle_l771_771413


namespace enclosed_area_is_correct_l771_771063

noncomputable def sum_of_coefficients (x : ℝ) (n : ℕ) := (x + (2 / x)) ^ n
noncomputable def constant_term (x : ℝ) (n : ℕ) (a : ℝ) := (∑ k in finset.range(n), (nat.choose n k : ℝ) * x ^ (n - k) * (2 / x) ^ k)

theorem enclosed_area_is_correct (n : ℕ) (a : ℝ) 
    (h1 : sum_of_coefficients 1 n = 2 ^ n)
    (h2 : constant_term 2 n a = (nat.choose n (n / 2) * 2 ^ (n / 2))) 
    (h3 : (2:ℝ) ≤ (n / 2) ≤ 4) :
    let line_area := (1 / 2) * 6 * (a / 6) * 6
    let curve_area := 2 * 6
    in (curve_area - line_area) = 32 / 3 := sorry

end enclosed_area_is_correct_l771_771063


namespace limit_evaluation_l771_771004

variables {α : Type*} [NormedField α] [NormedSpace ℝ α] [CompleteSpace α]

def f : α → α := sorry  -- Define f with appropriate type
def x₀ : α := sorry    -- The point x₀ in type α
def a : α := sorry     -- The value a in type α

axiom deriv_f_at_x₀ : deriv f x₀ = a

theorem limit_evaluation :
  ∀ (Δx : α), 
  filter.tendsto (λ Δx, (f (x₀ + Δx) - f (x₀ - 3 * Δx)) / (2 * Δx)) (nhds 0) (nhds (2 * a)) :=
begin
  -- The condition provided
  have h_deriv := deriv_f_at_x₀,
  sorry  -- Proof not included
end

end limit_evaluation_l771_771004


namespace value_of_f_at_pi_over_12_l771_771820

def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.cos x) ^ 4

theorem value_of_f_at_pi_over_12 : f (π / 12) = - (Real.sqrt 3 / 2) := by
  sorry

end value_of_f_at_pi_over_12_l771_771820


namespace train_crossing_time_l771_771269

def kmh_to_ms (speed_kmh : ℕ) : ℚ :=
  (speed_kmh * 1000) / 3600

def relative_speed (train_speed man_speed : ℕ) : ℚ :=
  kmh_to_ms train_speed - kmh_to_ms man_speed

theorem train_crossing_time
  (train_length : ℕ)
  (train_speed : ℕ)
  (man_speed : ℕ)
  (direction : Prop) -- true if in the same direction, false otherwise
  (correct_relative_speed : relative_speed train_speed man_speed = 15)
  (correct_length : train_length = 450) :
  train_length / relative_speed train_speed man_speed = 30 := 
by
  rw [correct_length, correct_relative_speed]
  norm_num
  rw [div_eq_mul_inv, mul_inv_cancel]
  norm_num
  
-- The proof is omitted with sorry. 
-- The steps include demonstrating correct relative_speed and train_length.

end train_crossing_time_l771_771269


namespace evaluate_neg_sixtyfour_exp_four_thirds_l771_771405

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l771_771405


namespace no_xy_term_implies_k_eq_4_l771_771060

theorem no_xy_term_implies_k_eq_4 (k : ℝ) :
  (∀ x y : ℝ, (x + 2 * y) * (2 * x - k * y - 1) = 2 * x^2 + (4 - k) * x * y - x - 2 * k * y^2 - 2 * y) →
  ((4 - k) = 0) →
  k = 4 := 
by
  intros h1 h2
  sorry

end no_xy_term_implies_k_eq_4_l771_771060


namespace admissions_price_l771_771163

theorem admissions_price (A : ℝ) : 
  (∃ A, 
     let total_tickets     := 1500 in
     let student_price     := 6 in
     let adult_tickets     := 1500 - 300 in
     let student_tickets   := 300 in
     let total_collected   := 16200 in
     let student_revenue   := student_tickets * student_price in
     let adult_revenue     := A * adult_tickets in
     total_collected = student_revenue + adult_revenue 
  ) → A = 12 :=
by
  sorry

end admissions_price_l771_771163


namespace plane_isometry_three_reflections_l771_771150

def isometry (G : Type _) (point : Type _) := ∀ (A B : point), dist (G A) (G B) = dist A B

theorem plane_isometry_three_reflections
  (G : point → point)
  (h_iso : isometry G point) : 
  ∃ (S₁ S₂ S₃ : point → point), (∃ (line : Type _), reflection line S₁) ∧ (∃ (line : Type _), reflection line S₂) ∧ (∃ (line : Type _), reflection line S₃) ∧ G = S₁ ∘ S₂ ∘ S₃ := 
sorry

end plane_isometry_three_reflections_l771_771150


namespace equilateral_triangle_limit_sum_l771_771762

variables (b : ℝ)

def perimeter_first_triangle (b : ℝ) : ℝ := 
  3 * b

noncomputable def geometric_series_sum (a r : ℝ) : ℝ :=
  a / (1 - r)

theorem equilateral_triangle_limit_sum (b : ℝ) :
  let S := geometric_series_sum (perimeter_first_triangle b) (1/3)
  in S = 9 * b / 2 :=
by
  sorry

end equilateral_triangle_limit_sum_l771_771762


namespace rate_of_current_equals_expected_l771_771643

-- Define the speed of the boat in still water
def boatSpeedStill: ℝ := 21

-- Define the distance travelled downstream
def distanceDownstream: ℝ := 6.283333333333333

-- Define the time in hours
def timeInHours: ℝ := 13 / 60

-- Define the variable for the rate of the current
variable (C: ℝ)

-- The expected rate of the current
def expectedCurrentRate: ℝ := 8

-- Define the proof problem
theorem rate_of_current_equals_expected :
  distanceDownstream = (boatSpeedStill + C) * timeInHours → 
  C = expectedCurrentRate :=
by
  sorry

end rate_of_current_equals_expected_l771_771643


namespace find_OJ_l771_771981

noncomputable def right_triangle_distance (O I J : Point) (R r : Real) (right_angle : ∠C = 90) : Real :=
  sorry

theorem find_OJ (O I J : Point) (R r : Real) (right_triangle : ∠C = 90) 
  (JO_symmetric_to_I : J = midpoint(I, symmetric_vertex(C))) : 
  right_triangle_distance(O, I, J, R, r) = R - 2 * r :=
by
  sorry

end find_OJ_l771_771981


namespace sin_cos_equation_range_l771_771828

theorem sin_cos_equation_range (x : ℝ) (h : x ∈ Set.Icc 0 (Real.pi / 2)) :
  ∃ k : ℝ, (∃ x ∈ Set.Icc 0 (Real.pi / 2), sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x) = k + 1) ↔ k ∈ Set.Icc (-2) 1 :=
sorry

end sin_cos_equation_range_l771_771828


namespace exists_pos_k_composite_l771_771597

theorem exists_pos_k_composite (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, k > 0 ∧ ∀ (n : ℕ), n > 0 → ¬ (nat.prime (k * 2^n + 1)) :=
begin
  sorry
end

end exists_pos_k_composite_l771_771597


namespace area_outside_circles_in_equilateral_triangle_l771_771093

theorem area_outside_circles_in_equilateral_triangle (a : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ 
                ∀ (x : ℝ), 
                x = a / (2 * (Real.sqrt 3 + 1)) ∧ 
                r = x ∧ 
                let area_triangle := (a^2 * Real.sqrt 3) / 4,
                    area_circle := π * (a * (Real.sqrt 3 - 1) / 4)^2,
                    total_area_circles := 3 * area_circle in
                  let area_outside := area_triangle - total_area_circles in
                    area_outside = a^2 * (2 * Real.sqrt 3 - 6 * π + 3 * π * Real.sqrt 3) / 8) :=
sorry

end area_outside_circles_in_equilateral_triangle_l771_771093


namespace lattice_points_on_curve_l771_771906

def is_lattice_point (x y : ℤ) : Prop :=
  x^2 - y^2 = 65

def lattice_points_count : ℕ :=
  { p : ℤ × ℤ // is_lattice_point p.1 p.2 }.card

theorem lattice_points_on_curve :
  lattice_points_count = 8 :=
sorry

end lattice_points_on_curve_l771_771906


namespace student_ratio_l771_771581

-- Define the variables and conditions
variables (S F T : ℕ)
constants (h1 : S = 40) 
          (h2 : F = 4 * S) 
          (h3 : T = 2 * F) 
          (h4 : F + S + T = 520)

-- The goal is to show that the ratio T:F is 2:1
theorem student_ratio (h1 : S = 40) (h2 : F = 4 * S) (h3 : T = 2 * F) (h4 : F + S + T = 520) :
  T / F = 2 := 
by 
  sorry

end student_ratio_l771_771581


namespace derivative_at_one_is_three_l771_771016

-- Definition of the function
def f (x : ℝ) := (x - 1)^2 + 3 * (x - 1)

-- The statement of the problem
theorem derivative_at_one_is_three : deriv f 1 = 3 := 
  sorry

end derivative_at_one_is_three_l771_771016


namespace books_cost_l771_771208

theorem books_cost (total_cost_three_books cost_seven_books : ℕ) 
  (h₁ : total_cost_three_books = 45)
  (h₂ : cost_seven_books = 7 * (total_cost_three_books / 3)) : 
  cost_seven_books = 105 :=
  sorry

end books_cost_l771_771208


namespace tan_double_angle_of_second_quadrant_l771_771022

-- Problem statement
theorem tan_double_angle_of_second_quadrant (α : ℝ) (h1 : α > π / 2 ∧ α < π) (h2 : sin α = 4 / 5) : tan (2 * α) = -24 / 7 := 
sorry

end tan_double_angle_of_second_quadrant_l771_771022


namespace cookies_remaining_percentage_l771_771199

theorem cookies_remaining_percentage: 
  ∀ (total initial_remaining eduardo_remaining final_remaining: ℕ),
  total = 600 → 
  initial_remaining = total - (2 * total / 5) → 
  eduardo_remaining = initial_remaining - (3 * initial_remaining / 5) → 
  final_remaining = eduardo_remaining → 
  (final_remaining * 100) / total = 24 := 
by
  intros total initial_remaining eduardo_remaining final_remaining h_total h_initial_remaining h_eduardo_remaining h_final_remaining
  sorry

end cookies_remaining_percentage_l771_771199


namespace E_plays_2_games_l771_771647

-- Definitions for the students and the number of games they played
def students := ["A", "B", "C", "D", "E"]
def games_played_by (S : String) : Nat :=
  if S = "A" then 4 else
  if S = "B" then 3 else
  if S = "C" then 2 else 
  if S = "D" then 1 else
  2  -- this is the number of games we need to prove for student E 

-- Theorem stating the number of games played by E
theorem E_plays_2_games : games_played_by "E" = 2 :=
  sorry

end E_plays_2_games_l771_771647


namespace pair_points_no_intersection_l771_771854

noncomputable theory
open_locale classical

universe u

def no_intersecting_pairs (T : set (ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 p3 p4 : ℝ × ℝ), p1 ∈ T → p2 ∈ T → p3 ∈ T → p4 ∈ T → 
  p1 ≠ p2 → p3 ≠ p4 →
  ¬((p1, p2), (p3, p4) : set (ℝ × ℝ) × set (ℝ × ℝ)).pairwise_disjoint segments

theorem pair_points_no_intersection (T : set (ℝ × ℝ)) (hT : ∃ (n : ℕ), T.card = 2 * n) :
  ∃ P : set (ℝ × ℝ) × set (ℝ × ℝ), no_intersecting_pairs T :=
sorry

end pair_points_no_intersection_l771_771854


namespace P_has_no_integer_roots_l771_771132

-- Define that a polynomial P(x) has integer coefficients
def has_integer_coeffs (P : Polynomial ℤ) : Prop :=
  ∀ (n : ℤ), P.coeff n ∈ ℤ

-- Define the polynomial Q(x)
def Q (P : Polynomial ℤ) : Polynomial ℤ :=
  P + Polynomial.C 12

-- Define that a polynomial has at least k distinct integer roots
def has_k_distinct_integer_roots (P : Polynomial ℤ) (k : ℕ) : Prop :=
  ∃ (roots : Finset ℤ), roots.card = k ∧ ∀ r ∈ roots, P.eval r = 0

-- Main theorem statement
theorem P_has_no_integer_roots (P : Polynomial ℤ) :
  has_integer_coeffs P →
  (∃ (roots : Finset ℤ), roots.card ≥ 6 ∧ ∀ r ∈ roots, (Q P).eval r = 0) →
  ¬ ∃ r : ℤ, P.eval r = 0 :=
by
  sorry

end P_has_no_integer_roots_l771_771132


namespace part1_part2_l771_771484

noncomputable def f (x a : ℝ) : ℝ := x * real.exp x - a * x^2 - x
noncomputable def g (x a : ℝ) : ℝ := x * real.exp x + a * x^2

theorem part1 (x : ℝ) (h : x > 0 ∨ x < -1 ∨ -1 < x < 0 ∨ -1 < x < 0 ∧ (2 * x)) :
  let a:ℝ := 1/2 in 
  (∀ x,   f'(x) = (x+1) * (real.exp x - 1) ∧   
  (∀ x,  (f'(x)= 0 ) ->(  (x > 0 ∨ x < -1) ∨ -1 < x < 0) )   :=
sorry

theorem part2 (a : ℝ) : (∀ x, x ≥ 1 → f(x, a) ≥ g(x, a)) ↔ a ≤ -1 :=
sorry

end part1_part2_l771_771484


namespace prove_circle_center_and_chord_length_l771_771025

noncomputable def circle_center_and_chord_length : Prop :=
  let eq_circle := ∀ x y : ℝ, x^2 + y^2 - 6*x - 8*y = 0
  let point_on_chord := (3, 5)
  let center := (3, 4)
  let radius := 5
  let chord_length := 4 * Real.sqrt 6
  eq_circle (3, 4) = 
    and (dist (3, 5) (3, 4) = 1)
    and (chord_length = 2 * Real.sqrt ((radius)^2 - (dist (3, 5) (3, 4))^2))

theorem prove_circle_center_and_chord_length : circle_center_and_chord_length := 
  sorry

end prove_circle_center_and_chord_length_l771_771025


namespace sequence_formula_no_arithmetic_progression_l771_771120

theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = q * a n) 
  (h_positive : ∀ n, a n > 0) 
  (h_sum : S 3 = 7) 
  (h_third_term : a 3 = 4) 
  (h_sum_def : ∀ n, S n = ∑ i in finset.range n, a (i + 1)) : 
  a n = 2 ^ (n - 1) :=
by
  sorry

theorem no_arithmetic_progression (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = q * a n) 
  (h_positive : ∀ n, a n > 0) 
  (h_sum : S 3 = 7) 
  (h_third_term : a 3 = 4) 
  (h_sum_def : ∀ n, S n = ∑ i in finset.range n, a (i + 1)) 
  (h_S_formula : ∀ n, S n = 2 ^ n - 1) : 
  ¬ ∃ (m n p : ℕ), m < n ∧ n < p ∧ (2 * S n = S m + S p) :=
by
  sorry

end sequence_formula_no_arithmetic_progression_l771_771120


namespace fred_gave_cards_l771_771140

theorem fred_gave_cards (initial_cards : ℕ) (torn_cards : ℕ) 
  (bought_cards : ℕ) (total_cards : ℕ) (fred_cards : ℕ) : 
  initial_cards = 18 → torn_cards = 8 → bought_cards = 40 → total_cards = 84 →
  fred_cards = total_cards - (initial_cards - torn_cards + bought_cards) →
  fred_cards = 34 :=
by
  intros h_initial h_torn h_bought h_total h_fred
  sorry

end fred_gave_cards_l771_771140


namespace razorback_shop_jersey_revenue_l771_771165

theorem razorback_shop_jersey_revenue :
  let price_per_tshirt := 67
  let price_per_jersey := 165
  let tshirts_sold := 74
  let jerseys_sold := 156
  jerseys_sold * price_per_jersey = 25740 := by
  sorry

end razorback_shop_jersey_revenue_l771_771165


namespace polynomial_coefficients_l771_771113

theorem polynomial_coefficients (x a : ℝ) (n : ℕ) (A : ℕ → ℝ) :
  ((x + a)^n = ∑ i in finset.range (n + 1), A i * x^i) →
  ∀ m : ℕ, m ≤ n → A m = (n.factorial / ((n - m).factorial * m.factorial)) * a^(n - m) :=
by
  sorry

end polynomial_coefficients_l771_771113


namespace trigonometric_identity_l771_771266

theorem trigonometric_identity (t : ℝ) : 
  ∃ (k : ℤ), t = (π / 8) * (2 * k + 1) ↔ 
  2 * (cos (2 * t))^6 - (cos (2 * t))^4 + 1.5 * (sin (4 * t))^2 - 3 * (sin (2 * t))^2 = 0 :=
by
  sorry

end trigonometric_identity_l771_771266


namespace power_evaluation_l771_771390

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l771_771390


namespace part1_part2_part3_l771_771137

section Part1
variable (f : ℝ → ℝ) (a b : ℝ)
def is_mean_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x_0 ∈ Set.Ioo a b, f x_0 = (f b - f a) / (b - a)

theorem part1
  (f := λ x : ℝ => x^2)
  (a := 1)
  (b := 2) :
  is_mean_function f a b ∧ ∃ x_0, x_0 = Real.sqrt 3 := by
  sorry
end Part1

section Part2
variable (f : ℝ → ℝ) (m : ℝ)
def is_mean_function_for_m (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x_0 ∈ Set.Ioo a b, f x_0 = (f b - f a) / (b - a)

theorem part2
  (f := λ x : ℝ => -2^(2*x-1) + m * 2^(x-1) - 12)
  (a := 1)
  (b := 3) :
  is_mean_function_for_m f a b → 
  m ∈ Set.union
       (Set.Iio 2) 
       (Set.Ici (2 * Real.sqrt 3 + 6)) := by
  sorry
end Part2

section Part3
variable (f : ℝ → ℝ) (a : ℝ)
def is_mean_function_with_mean_point (f : ℝ → ℝ) (a b mean_point : ℝ) : Prop :=
  ∃ x_0 ∈ Set.Ioo a b, f x_0 = (f b - f a) / (b - a) ∧ x_0 = mean_point

def H (f : ℝ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range (2^n)).sum (λ i => f (i / 2^n))

def G (f : ℝ → ℝ) (m : ℕ) : ℝ :=
  (Finset.range m).sum (λ i => abs (f (i / m) - f ((i+1) / m)))

theorem part3
  (f := λ x : ℝ => (x^2 + a) / (2*(x^2 - 2*x + 2)))
  (a := 0)
  (mean_point := 2/3)
  (n : ℕ) :
  is_mean_function_with_mean_point f (-2) 2 mean_point →
  (H f n) * (G f (2^n)) > 2023 →
  n ≥ 15 := by
  sorry
end Part3

end part1_part2_part3_l771_771137


namespace sum_of_possible_values_of_d_l771_771726

theorem sum_of_possible_values_of_d : 
  ∀ (n : ℕ), (16^4 ≤ n ∧ n < 16^5) → 
  (∑ (d : ℕ) in {(17, true), (18, true), (19, true), (20, true)}, d.1) = 74 :=
by
  intros n hn
  have h16 := (16 : ℕ)
  have h2 := (2 : ℕ)
  change 16^4 ≤ n ∧ n < 16^5 at hn
  sorry

end sum_of_possible_values_of_d_l771_771726


namespace residue_of_sequence_modulo_2011_l771_771118

theorem residue_of_sequence_modulo_2011 :
  let T : ℤ := (∑ i in Finset.range 1006, if i % 2 = 0 then (2 + 2 * i) else -(3 + 2 * (i - 1)))
  in T % 2011 = 1006 :=
by
  sorry

end residue_of_sequence_modulo_2011_l771_771118


namespace calculate_expression_solve_linear_system_l771_771811

-- Part 1
theorem calculate_expression :
  -real.sqrt 16 / abs (-2) + real.cbrt 27 = 1 :=
by sorry

-- Part 2
theorem solve_linear_system (x y : ℝ) (h1 : 2 * x - y = 7) (h2 : 3 * x + 2 * y = 0) :
  x = 2 ∧ y = -3 :=
by sorry

end calculate_expression_solve_linear_system_l771_771811


namespace statement_1_statement_2_statement_3_correct_statements_l771_771321

-- Definitions of propositions p and q
def proposition_p : Prop := ∃ x0 : ℝ, Real.tan x0 = 2
def proposition_q : Prop := ∀ x : ℝ, x^2 - x + 1/2 > 0

-- statement 1: (proposition_p ∧ ¬ proposition_q) is false
theorem statement_1 : proposition_p ∧ ¬ proposition_q → False := 
by {
  intro h,
  cases h with hp hq,
  apply hq,
  intro x,
  have hx : (x - 1/2)^2 + 1/4 > 0,
  { exact sq_nonneg (x - 1/2) + 1/4 },
  linarith,
}

-- Definitions for the lines and conditions for perpendicularity
def l1 (a : ℝ) (x : ℝ) (y : ℝ) : Prop := a * x + 3 * y - 1 = 0
def l2 (b : ℝ) (x : ℝ) (y : ℝ) : Prop := x + b * y + 1 = 0
def lines_perpendicular (a b : ℝ) : Prop := a + 3 * b = 0

-- statement 2: correct condition for l1 and l2 to be perpendicular
theorem statement_2 (a b : ℝ) : (a + 3 * b = 0) ↔ (a/b = -3) := 
by {
  split;
  intro h,
  -- Forward direction
  { field_simp at h, linarith, },
  -- Backward direction
  { field_simp, linarith, },
}

-- Definitions for the third problem
def ab_ge_2 (a b : ℝ) : Prop := a * b ≥ 2
def a2_b2_gt_4 (a b : ℝ) : Prop := a^2 + b^2 > 4

-- statement 3: negation is correctly identified
theorem statement_3 (a b : ℝ) : ab_ge_2 a b → a2_b2_gt_4 a b → (¬ (ab_ge_2 a b) → ¬ (a2_b2_gt_4 a b)) := 
by {
  intros h1 h2 h3,
  apply h3,
  intro h4,
  linarith,
}

-- Now to state the main theorem:
theorem correct_statements : statement_1 ∨ statement_2 ∨ statement_3 := 
by {
  left,
  exact statement_1,
  right,
  exact statement_2,
  right,
  exact statement_3,
}

end statement_1_statement_2_statement_3_correct_statements_l771_771321


namespace unique_n_for_solutions_l771_771127

theorem unique_n_for_solutions :
  ∃! (n : ℕ), (∀ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (3 * x + 3 * y + 2 * z = n)) → 
  ((∃ (s : ℕ), s = 10) ∧ (n = 17)) :=
sorry

end unique_n_for_solutions_l771_771127


namespace gab_score_ratio_l771_771938

theorem gab_score_ratio (S G C O : ℕ) (h1 : S = 20) (h2 : C = 2 * G) (h3 : O = 85) (h4 : S + G + C = O + 55) :
  G / S = 2 := 
by 
  sorry

end gab_score_ratio_l771_771938


namespace dice_probability_l771_771751

noncomputable def total_events : ℕ := 36

def favorable_events : ℕ :=
  let faces := [1, 2, 3, 4, 5, 6] in
  let possible_pairs := (faces.product faces) in
  possible_pairs.count (λ (pair : ℕ × ℕ), (pair.1 + pair.2) % 4 = 2)

theorem dice_probability : (favorable_events : ℝ) / (total_events : ℝ) = 1 / 4 :=
sorry

end dice_probability_l771_771751


namespace witch_votes_is_seven_l771_771543

-- Definitions
def votes_for_witch (W : ℕ) : ℕ := W
def votes_for_unicorn (W : ℕ) : ℕ := 3 * W
def votes_for_dragon (W : ℕ) : ℕ := W + 25
def total_votes (W : ℕ) : ℕ := votes_for_witch W + votes_for_unicorn W + votes_for_dragon W

-- Proof Statement
theorem witch_votes_is_seven (W : ℕ) (h1 : total_votes W = 60) : W = 7 :=
by
  sorry

end witch_votes_is_seven_l771_771543


namespace vector_dot_product_identity_l771_771493

-- Define the vectors a, b, and c in ℝ²
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (-3, 1)

-- Define vector addition and dot product in ℝ²
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove that c · (a + b) = 9
theorem vector_dot_product_identity : dot_product c (vector_add a b) = 9 := 
by 
sorry

end vector_dot_product_identity_l771_771493


namespace lattice_points_on_hyperbola_l771_771912

-- The statement to be proven
theorem lattice_points_on_hyperbola : 
  {p : ℤ × ℤ | p.1 ^ 2 - p.2 ^ 2 = 65}.finite.toFinset.card = 4 :=
by
  sorry

end lattice_points_on_hyperbola_l771_771912


namespace gcd_gx_x_l771_771006

theorem gcd_gx_x (x : ℤ) (hx : 34560 ∣ x) :
  Int.gcd ((3 * x + 4) * (8 * x + 5) * (15 * x + 11) * (x + 17)) x = 20 := 
by
  sorry

end gcd_gx_x_l771_771006


namespace car_P_distance_is_3v_l771_771774

-- Assume speed v.
variables (v : ℝ)

-- Define Car M's conditions.
def car_M_distance := 3 * v

-- Define Car N's conditions.
def car_N_speed := 3 * v
def car_N_distance := 2 * car_N_speed

-- Define Car P's conditions.
def car_P_start := 1.5
def car_P_speed := 2 * v
def car_P_time := 1.5
def car_P_distance := car_P_speed * car_P_time

-- Prove the distance covered by Car P is 3v.
theorem car_P_distance_is_3v : car_P_distance v = 3 * v := by
  sorry

end car_P_distance_is_3v_l771_771774


namespace twenty_four_multiples_of_4_l771_771194

theorem twenty_four_multiples_of_4 {n : ℕ} : (n = 104) ↔ (∃ k : ℕ, k = 24 ∧ ∀ m : ℕ, (12 ≤ m ∧ m ≤ n) → ∃ t : ℕ, m = 12 + 4 * t ∧ 1 ≤ t ∧ t ≤ 24) := 
by
  sorry

end twenty_four_multiples_of_4_l771_771194


namespace odd_terms_in_binomial_expansion_l771_771511

-- Define the problem conditions and the statement
theorem odd_terms_in_binomial_expansion (a b : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) : 
  let terms := [(nat.choose 8 0) * (a^8),
                (nat.choose 8 1) * (a^7 * b),
                (nat.choose 8 2) * (a^6 * b^2),
                (nat.choose 8 3) * (a^5 * b^3),
                (nat.choose 8 4) * (a^4 * b^4),
                (nat.choose 8 5) * (a^3 * b^5),
                (nat.choose 8 6) * (a^2 * b^6),
                (nat.choose 8 7) * (a * b^7),
                (nat.choose 8 8) * (b^8)] in
  (terms.filter (λ term, term % 2 = 1)).length = 2 :=
sorry

end odd_terms_in_binomial_expansion_l771_771511


namespace find_cos_E_floor_l771_771075

theorem find_cos_E_floor (EF GH EH FG : ℝ) (E G : ℝ) 
  (h1 : EF = 200) 
  (h2 : GH = 200) 
  (h3 : EH ≠ FG) 
  (h4 : EF + GH + EH + FG = 800) 
  (h5 : E = G) : 
  (⌊1000 * Real.cos E⌋ = 1000) := 
by 
  sorry

end find_cos_E_floor_l771_771075


namespace sum_of_xi_l771_771441

theorem sum_of_xi {x1 x2 x3 x4 : ℝ} (h1: (x1 - 3) * Real.sin (π * x1) = 1)
  (h2: (x2 - 3) * Real.sin (π * x2) = 1)
  (h3: (x3 - 3) * Real.sin (π * x3) = 1)
  (h4: (x4 - 3) * Real.sin (π * x4) = 1)
  (hx1 : x1 > 0) (hx2: x2 > 0) (hx3 : x3 > 0) (hx4: x4 > 0) :
  x1 + x2 + x3 + x4 = 12 :=
by
  sorry

end sum_of_xi_l771_771441


namespace count_integers_with_digit_sum_17_l771_771900

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + (n / 100)

theorem count_integers_with_digit_sum_17 : 
  let count := (finset.range 201).filter (λ x, sum_of_digits (x + 400) = 17)
  count.card = 13 :=
by
  sorry

end count_integers_with_digit_sum_17_l771_771900


namespace friends_pay_6_22_l771_771822

noncomputable def cost_per_friend : ℕ :=
  let hamburgers := 5 * 3
  let fries := 4 * 120 / 100
  let soda := 5 * 50 / 100
  let spaghetti := 270 / 100
  let milkshakes := 3 * 250 / 100
  let nuggets := 2 * 350 / 100
  let total_bill := hamburgers + fries + soda + spaghetti + milkshakes + nuggets
  let discount := total_bill * 10 / 100
  let discounted_bill := total_bill - discount
  let birthday_friend := discounted_bill * 30 / 100
  let remaining_amount := discounted_bill - birthday_friend
  remaining_amount / 4

theorem friends_pay_6_22 : cost_per_friend = 622 / 100 :=
by
  sorry

end friends_pay_6_22_l771_771822


namespace bus_distance_720_l771_771297

theorem bus_distance_720 (total_distance : ℕ) (bus_distance : ℕ) (plane_fraction : ℚ) (train_bus_ratio : ℚ) :
  total_distance = 1800 →
  plane_fraction = 1 / 3 →
  train_bus_ratio = 2 / 3 →
  bus_distance = 720 →
  bus_distance + (train_bus_ratio * bus_distance).natAbs + (plane_fraction * total_distance).natAbs = total_distance :=
by
  intros h_total h_plane_frac h_train_bus_ratio h_bus
  rw [h_total, h_plane_frac, h_train_bus_ratio, h_bus]
  norm_num
  sorry

end bus_distance_720_l771_771297


namespace digit_in_hundredths_place_of_fraction_l771_771226

theorem digit_in_hundredths_place_of_fraction (n d : ℕ) (h : d = 8) (h' : n = 5) :
  (decDigitHundredthsPlace (n / d) = 2) :=
sorry

def decDigitHundredthsPlace (r : ℚ) : ℕ := 
  -- Fill in this definition appropriately to extract the hundredths digit.
  sorry

end digit_in_hundredths_place_of_fraction_l771_771226


namespace binomial_distribution_equivalence_l771_771989

open ProbabilityTheory

-- Define the binomial distribution with parameters n and p
def binomial (n : ℕ) (p : ℝ) : Measure ℕ :=
  Measure.dirac' (Nat' n) * Bernoulli (real.sqrt p)

-- Define the random variables X and Y
def X (p : ℝ) : Measure ℕ := binomial 2 p
def Y (p : ℝ) : Measure ℕ := binomial 3 p

-- State the theorem
theorem binomial_distribution_equivalence (p : ℝ) (h : ∫ (x : ℕ) in X p, if x ≥ 1 then 1 else 0 = 5 / 9) :
  ∫ (y : ℕ) in Y p, if y = 2 then 1 else 0 = 2 / 9 :=
sorry

end binomial_distribution_equivalence_l771_771989


namespace root_properties_of_polynomial_l771_771570

variables {r s t : ℝ}

def polynomial (x : ℝ) : ℝ := 6 * x^3 + 4 * x^2 + 1500 * x + 3000

theorem root_properties_of_polynomial :
  (∀ x : ℝ, polynomial x = 0 → (x = r ∨ x = s ∨ x = t)) →
  (r + s + t = -2 / 3) →
  (r * s + r * t + s * t = 250) →
  (r * s * t = -500) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = -5992 / 27 :=
by
  sorry

end root_properties_of_polynomial_l771_771570


namespace bicycle_cost_price_l771_771743

-- Definitions of conditions
def profit_22_5_percent (x : ℝ) : ℝ := 1.225 * x
def loss_14_3_percent (x : ℝ) : ℝ := 0.857 * x
def profit_32_4_percent (x : ℝ) : ℝ := 1.324 * x
def loss_7_8_percent (x : ℝ) : ℝ := 0.922 * x
def discount_5_percent (x : ℝ) : ℝ := 0.95 * x
def tax_6_percent (x : ℝ) : ℝ := 1.06 * x

theorem bicycle_cost_price (CP_A : ℝ) (TP_E : ℝ) (h : TP_E = 295.88) : 
  CP_A = 295.88 / 1.29058890594 :=
by
  sorry

end bicycle_cost_price_l771_771743


namespace jogger_distance_ahead_l771_771295

noncomputable def jogger_speed_kmph : ℤ := 9
noncomputable def train_speed_kmph : ℤ := 45
noncomputable def train_length_m : ℤ := 120
noncomputable def time_to_pass_seconds : ℤ := 38

theorem jogger_distance_ahead
  (jogger_speed_kmph : ℤ)
  (train_speed_kmph : ℤ)
  (train_length_m : ℤ)
  (time_to_pass_seconds : ℤ) :
  jogger_speed_kmph = 9 →
  train_speed_kmph = 45 →
  train_length_m = 120 →
  time_to_pass_seconds = 38 →
  ∃ distance_ahead : ℤ, distance_ahead = 260 :=
by 
  -- the proof would go here
  sorry  

end jogger_distance_ahead_l771_771295


namespace find_angle_x_l771_771544

-- Definitions for the angles
def angle_ABC : ℝ := 70
def angle_BAC : ℝ := 55
def angle_CED : ℝ := 90

-- Goal to prove:
theorem find_angle_x (ABC BAC CED : ℝ) (h1 : ABC = 70) (h2 : BAC = 55) (h3 : CED = 90) : 
  let BCA := 180 - ABC - BAC in
  let x := 180 - CED in
  x = 90 :=
by
  rw [h1, h2, h3]
  let BCA := 180 - 70 - 55
  let x := 180 - 90
  have h4 : BCA = 55 := by norm_num
  have h5 : x = 90 := by norm_num
  exact h5

end find_angle_x_l771_771544


namespace domain_of_f_l771_771171

noncomputable def f (x : ℝ) : ℝ := sqrt ((1 - x) / (3 + x))

theorem domain_of_f :
  {x : ℝ | 0 ≤ (1 - x) / (3 + x)} = {x : ℝ | -3 < x ∧ x ≤ 1} :=
by
  sorry

end domain_of_f_l771_771171


namespace smallest_base_for_80_l771_771250

-- Define the problem in terms of inequalities
def smallest_base (n : ℕ) (d : ℕ) :=
  ∃ b : ℕ, b > 1 ∧ b <= (n^(1/d)) ∧ (n^(1/(d+1))) < (b + 1)

-- Assertion that the smallest whole number b such that 80 can be expressed in base b using only three digits
theorem smallest_base_for_80 : ∃ b, smallest_base 80 3 ∧ b = 5 :=
  sorry

end smallest_base_for_80_l771_771250


namespace prove_inequality_1_prove_inequality_2_prove_inequality_3_l771_771477

theorem prove_inequality_1 (a b c d : ℝ) (h1 : ab > 0) (h2 : bc > ad) : (c / a) > (d / b) :=
by sorry

theorem prove_inequality_2 (a b c d : ℝ) (h1 : ab > 0) (h2 : (c / a) > (d / b)) : bc > ad :=
by sorry

theorem prove_inequality_3 (a b c d : ℝ) (h1 : bc > ad) (h2 : (c / a) > (d / b)) : ab > 0 :=
by sorry

end prove_inequality_1_prove_inequality_2_prove_inequality_3_l771_771477


namespace eval_expression_correct_l771_771402

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l771_771402


namespace alpha_plus_beta_l771_771196

theorem alpha_plus_beta :
  (∃ α β : ℝ, 
    (∀ x : ℝ, x ≠ -β ∧ x ≠ 45 → (x - α) / (x + β) = (x^2 - 90 * x + 1980) / (x^2 + 70 * x - 3570))
  ) → (∃ α β : ℝ, α + β = 123) :=
by {
  sorry
}

end alpha_plus_beta_l771_771196


namespace proof_set_intersection_l771_771491

noncomputable def U := ℝ
noncomputable def M := {x : ℝ | 0 ≤ x ∧ x < 5}
noncomputable def N := {x : ℝ | x ≥ 2}
noncomputable def compl_U_N := {x : ℝ | x < 2}
noncomputable def intersection := { x : ℝ | 0 ≤ x ∧ x < 2 }

theorem proof_set_intersection : ((compl_U_N ∩ M) = {x : ℝ | 0 ≤ x ∧ x < 2}) :=
by
  sorry

end proof_set_intersection_l771_771491


namespace symmetric_circles_intersecting_line_l771_771891

noncomputable def circle_C_standard_eq : Prop :=
  (∃ (x y : ℝ), ((x+2)^2 + (y-1)^2 = 4))

noncomputable def line_l_eq : Prop :=
  (∃ (k : ℝ), (k = (real.sqrt(3) / 3) ∨ k = -(real.sqrt(3) / 3)) ∧
  (∀ (x y : ℝ), (y = k*x + 1)))

theorem symmetric_circles_intersecting_line:
  (∃ (x y : ℝ), (x-1)^2 + (y+2)^2 = 4) ∧
  (∃ (x y : ℝ), ((x = y) ∨ (x = -y))) ∧
  (∃ (x y : ℝ), (∃ (A B : ℝ), (|A - B| = 2 * real.sqrt(3))) ∧
  (∃ (C : ℝ), ((x + 2) ^ 2 + (y - 1) ^ 2 = 4) ∧ (C = -A/B))) →
  circle_C_standard_eq ∧ line_l_eq :=
by
  sorry

end symmetric_circles_intersecting_line_l771_771891


namespace evaluate_pow_l771_771358

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l771_771358


namespace transplant_trees_l771_771305

def tree_arrangement_exists : Prop :=
  ∃ (initial_arrangement : fin 22 → Finset (ℕ × ℕ)),
  ∃ (new_arrangement : fin 22 → Finset (ℕ × ℕ)),
  (transplanted: fin 6 → Finset (ℕ × ℕ)),
  (initial_arrangement ∩ transplanted = ∅) ∧ 
  (initial_arrangement ∪ transplanted = new_arrangement) ∧ 
  ((∑ row in rows(new_arrangement), length(row) = 20)

theorem transplant_trees : tree_arrangement_exists :=
sorry

end transplant_trees_l771_771305


namespace inequality_proof_l771_771151

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ab * (a + b) + bc * (b + c) + ac * (a + c) ≥ 6 * abc := 
sorry

end inequality_proof_l771_771151


namespace mark_total_payment_l771_771995

def total_cost (work_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  work_hours * hourly_rate + part_cost

theorem mark_total_payment :
  total_cost 2 75 150 = 300 :=
by
  -- Proof omitted, sorry used to skip the proof
  sorry

end mark_total_payment_l771_771995


namespace math_problem_l771_771513

theorem math_problem (r p q : ℝ) (h₁ : r > 0) (h₂ : pq ≠ 0) (h₃ : pr > qr) : 
  ¬(-p > -q) ∧ ¬(-p > q) ∧ ¬(1 > -q / p) ∧ ¬(1 < q / p) :=
by
  sorry

end math_problem_l771_771513


namespace flu_infection_equation_l771_771301

theorem flu_infection_equation (x : ℕ) (h : 1 + x + x^2 = 36) : 1 + x + x^2 = 36 :=
by
  sorry

end flu_infection_equation_l771_771301


namespace correct_choice_of_f_l771_771014

def f1 (x : ℝ) : ℝ := (x - 1)^2 + 3 * (x - 1)
def f2 (x : ℝ) : ℝ := 2 * (x - 1)
def f3 (x : ℝ) : ℝ := 2 * (x - 1)^2
def f4 (x : ℝ) : ℝ := x - 1

theorem correct_choice_of_f (h : (deriv f1 1 = 3) ∧ (deriv f2 1 ≠ 3) ∧ (deriv f3 1 ≠ 3) ∧ (deriv f4 1 ≠ 3)) : 
  ∀ f, (f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4) → (deriv f 1 = 3 → f = f1) :=
by sorry

end correct_choice_of_f_l771_771014


namespace solution_correct_l771_771416

noncomputable def recurrence_rel (n : ℕ) : ℕ × ℕ → ℕ × ℕ 
| (a, b) := (29 * a - b, a)

def initial_pair : ℕ × ℕ := (0, 29)

theorem solution_correct :
  ∀ (a b : ℕ), 
  (∃ n : ℕ, (a, b) = Nat.iterate recurrence_rel n initial_pair) ↔ (a^2 + b^2 = 841 * (a * b + 1)) ∧ (a ≥ 0) ∧ (b ≥ 0) := 
sorry

end solution_correct_l771_771416


namespace probability_of_rolling_one_five_times_and_two_once_in_seven_rolls_l771_771921

noncomputable def probability_roll_event : ℚ :=
  let p_one := 1 / 6 in
  let p_two := 1 / 6 in
  let p_other := 2 / 3 in
  let comb := Nat.choose 7 5 * Nat.choose 2 1 in
  comb * p_one^5 * p_two * p_other

theorem probability_of_rolling_one_five_times_and_two_once_in_seven_rolls :
  probability_roll_event = 1 / 417 := 
sorry

end probability_of_rolling_one_five_times_and_two_once_in_seven_rolls_l771_771921


namespace initial_men_count_l771_771924

theorem initial_men_count (x : ℕ) 
  (h1 : ∀ t : ℕ, t = 25 * x) 
  (h2 : ∀ t : ℕ, t = 12 * 75) : 
  x = 36 := 
by
  sorry

end initial_men_count_l771_771924


namespace find_e_l771_771188

theorem find_e (d e f : ℝ) (h1 : f = 5)
  (h2 : -d / 3 = -f)
  (h3 : -f = 1 + d + e + f) :
  e = -26 := 
by
  sorry

end find_e_l771_771188


namespace new_height_of_water_l771_771728

theorem new_height_of_water 
  (r₁ : ℝ) (h₁ : ℝ) (r₂ : ℝ)
  (H₁ : r₁ = 8) (H₂ : h₁ = 24) (H₃ : r₂ = 16) : 
  let V₁ := (1/3) * π * r₁^2 * h₁,
      h₂ := V₁ / (π * r₂^2)
  in h₂ = 2 := 
by
  sorry

end new_height_of_water_l771_771728


namespace eval_neg_pow_l771_771378

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l771_771378


namespace cafeteria_pies_l771_771278

theorem cafeteria_pies (initial_apples handed_out_apples apples_per_pie : ℕ)
  (h1 : initial_apples = 96)
  (h2 : handed_out_apples = 42)
  (h3 : apples_per_pie = 6) :
  (initial_apples - handed_out_apples) / apples_per_pie = 9 := by
  sorry

end cafeteria_pies_l771_771278


namespace standard_equation_of_ellipse_distance_ab_intersect_l771_771872

-- Definitions for the conditions
def center_at_origin (C : Type) := @eq (Prod ℝ ℝ) (0, 0) ⟨0, 0⟩
def focus_on_x_axis (F : Type) := ∃ cx : ℝ, @eq (Prod ℝ ℝ) (cx, 0) ⟨cx, 0⟩
def max_distance_to_focus (C : Type) (F : Type) (d : ℝ) := d = 3
def eccentricity (C : Type) (e : ℝ) := e = 1 / 2

-- The standard equation of the ellipse
theorem standard_equation_of_ellipse : 
  ∃ (a b : ℝ), a = 2 ∧ b = sqrt (4 - 1) ∧ ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1 :=
sorry

-- The distance |AB| when line intersects the ellipse
theorem distance_ab_intersect (Cx Cy : ℝ) (F1x F1y : ℝ) (m : ℝ) : 
  F1x = -1 ∧ F1y = 0 ∧ m = 1 ∧ ∀ x1 y1 x2 y2 : ℝ, 
((7 * (x1 ^ 2)) + (8 * x1) - 8 = 0) ∧ 
  (x2 = (-8) / 7) ∧ 
  (y1 = x1 + 1) ∧
  (y2 = x1 + 1) ∧   
  (abs (x1 - x2)) = 24 / 7 :=
sorry

end standard_equation_of_ellipse_distance_ab_intersect_l771_771872


namespace hexagon_side_squares_sum_l771_771326

variables {P Q R P' Q' R' A B C D E F : Type}
variables (a1 a2 a3 b1 b2 b3 : ℝ)
variables (h_eq_triangles : congruent (triangle P Q R) (triangle P' Q' R'))
variables (h_sides : 
  AB = a1 ∧ BC = b1 ∧ CD = a2 ∧ 
  DE = b2 ∧ EF = a3 ∧ FA = b3)
  
theorem hexagon_side_squares_sum :
  a1^2 + a2^2 + a3^2 = b1^2 + b2^2 + b3^2 :=
sorry

end hexagon_side_squares_sum_l771_771326


namespace solution_set_f_lt_exp_l771_771012

noncomputable def f : ℝ → ℝ := sorry

variable (f' : ∀ x : ℝ, has_deriv_at f (f' x) x)
variable (f'_ineq : ∀ x : ℝ, f' x < f x)
variable (f1 : f 1 = 1)

theorem solution_set_f_lt_exp (x : ℝ) :
  (f x < exp (x - 1)) ↔ (x > 1) :=
by
  sorry

end solution_set_f_lt_exp_l771_771012


namespace lambda_6_ge_sqrt3_l771_771436

-- Define the points and the distance function in the plane
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the ratio λ₆ for a set of six points
def lambda_6 (points : fin 6 → (ℝ × ℝ)) : ℝ :=
  let dists := {dist | i j : fin 6, i ≠ j, let dist := distance (points i) (points j), dist}
  (dists.max' ⟨sorry⟩) / (dists.min' ⟨sorry⟩)

-- Theorem statement
theorem lambda_6_ge_sqrt3 (points : fin 6 → (ℝ × ℝ)) : 
  lambda_6 points ≥ real.sqrt 3 := 
sorry

end lambda_6_ge_sqrt3_l771_771436


namespace inequalities_correct_l771_771463

theorem inequalities_correct (a b : ℝ) (h : a * b > 0) :
  |b| > |a| ∧ |a + b| < |b| := sorry

end inequalities_correct_l771_771463


namespace unique_function_satisfies_condition_l771_771423

theorem unique_function_satisfies_condition :
  ∃! f : ℝ → ℝ, ∀ x y z : ℝ, f (x * Real.sin y) + f (x * Real.sin z) -
    f x * f (Real.sin y * Real.sin z) + Real.sin (Real.pi * x) ≥ 1 := sorry

end unique_function_satisfies_condition_l771_771423


namespace lattice_points_on_curve_l771_771907

def is_lattice_point (x y : ℤ) : Prop :=
  x^2 - y^2 = 65

def lattice_points_count : ℕ :=
  { p : ℤ × ℤ // is_lattice_point p.1 p.2 }.card

theorem lattice_points_on_curve :
  lattice_points_count = 8 :=
sorry

end lattice_points_on_curve_l771_771907


namespace triangle_areas_l771_771116

theorem triangle_areas (A B C P G1 G2 G3 : Point) (P_incenter : incenter A B C P)
  (G1_centroid : centroid P B C G1) (G2_centroid : centroid P C A G2) (G3_centroid : centroid P A B G3) 
  (area_ABC : area A B C = 48) : area G1 G2 G3 = 48 * (1 / 9) :=
by
  sorry

end triangle_areas_l771_771116


namespace exists_positive_b_l771_771968

theorem exists_positive_b (m p : ℕ) (hm : 0 < m) (hp : Prime p)
  (h1 : m^2 ≡ 2 [MOD p])
  (ha : ∃ a : ℕ, 0 < a ∧ a^2 ≡ 2 - m [MOD p]) :
  ∃ b : ℕ, 0 < b ∧ b^2 ≡ m + 2 [MOD p] := 
  sorry

end exists_positive_b_l771_771968


namespace kim_change_is_5_l771_771107

variable (meal_cost : ℝ) (drink_cost : ℝ) (tip_percent : ℝ) (payment : ℝ)

theorem kim_change_is_5 (h_meal_cost : meal_cost = 10)
                        (h_drink_cost : drink_cost = 2.5)
                        (h_tip_percent : tip_percent = 0.20)
                        (h_payment : payment = 20) :
  let total_cost := meal_cost + drink_cost
      tip_amount := tip_percent * total_cost
      total_with_tip := total_cost + tip_amount
      change := payment - total_with_tip
  in change = 5 := 
by
  sorry

end kim_change_is_5_l771_771107


namespace y_function_chord_length_constant_l771_771767

-- Function that describes y in terms of x
def y (x : ℝ) : ℝ := -x - 1

-- Prove y is equal to -x - 1 and define the range of x
theorem y_function : ∀ x : ℝ, y x = -x - 1 := 
by
  intros x
  simp [y]
  sorry

-- Condition involving P, Q, and the chord length constraints
def P (x : ℝ) : ℝ × ℝ := (x, y x)
def Q (t : ℝ) : ℝ × ℝ := (0, t)

-- The chord length remains constant, proving the range of t
theorem chord_length_constant (l : ℝ) (t : ℝ) :
  (∀ (x : ℝ), chord_length (circle_from_diameter (P x) (Q t)) l) →
  t = 0 ∧ x ≠ 0 :=
by
  intros x h
  simp
  sorry

end y_function_chord_length_constant_l771_771767


namespace max_value_is_one_sixteenth_l771_771244

noncomputable def max_value_expression (t : ℝ) : ℝ :=
  (3^t - 4 * t) * t / 9^t

theorem max_value_is_one_sixteenth : 
  ∃ t : ℝ, max_value_expression t = 1 / 16 :=
sorry

end max_value_is_one_sixteenth_l771_771244


namespace arithmetic_sequence_difference_l771_771771

theorem arithmetic_sequence_difference :
  let seq1 := list.range' 2001 93
  let seq2 := list.range' 201 93
  (seq1.sum - seq2.sum) = 167400 :=
by
  sorry

end arithmetic_sequence_difference_l771_771771


namespace solve_y_such_that_l771_771605

theorem solve_y_such_that (y : ℝ) (h : 16^(y+1) / 8^(y+1) = 128^(y-2)) : y = 3 :=
sorry

end solve_y_such_that_l771_771605


namespace monotonic_decreasing_interval_l771_771183

noncomputable def func (x : ℝ) : ℝ :=
  x * Real.log x

noncomputable def derivative (x : ℝ) : ℝ :=
  Real.log x + 1

theorem monotonic_decreasing_interval :
  { x : ℝ | 0 < x ∧ x < Real.exp (-1) } ⊆ { x : ℝ | derivative x < 0 } :=
by
  sorry

end monotonic_decreasing_interval_l771_771183


namespace functions_odd_or_not_l771_771261

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem functions_odd_or_not :
  (¬ is_odd (λ x, |x|)) ∧
  is_odd (λ x, x + 1/x) ∧
  is_odd (λ x, x^3 + 2*x) ∧
  (¬ is_odd (λ x, x^2 + x + 1)) :=
by sorry

end functions_odd_or_not_l771_771261


namespace Markus_bags_count_l771_771993

-- Definitions of the conditions
def Mara_bags : ℕ := 12
def Mara_marbles_per_bag : ℕ := 2
def Markus_marbles_per_bag : ℕ := 13
def marbles_difference : ℕ := 2

-- Derived conditions
def Mara_total_marbles : ℕ := Mara_bags * Mara_marbles_per_bag
def Markus_total_marbles : ℕ := Mara_total_marbles + marbles_difference

-- Statement to prove
theorem Markus_bags_count : Markus_total_marbles / Markus_marbles_per_bag = 2 :=
by
  -- Skip the proof, leaving it as a task for the prover
  sorry

end Markus_bags_count_l771_771993


namespace collinear_points_y_value_l771_771452

theorem collinear_points_y_value : ∀ y : ℝ, 
    points_collinear (point.mk 4 8) (point.mk 2 4) (point.mk 3 y) → y = 6 :=
begin
  intro y,
  intro h_collinear,
  sorry
end

end collinear_points_y_value_l771_771452


namespace highest_coeff_bound_no_roots_l771_771152

variable {R : Type*} [LinearOrderedField R]

noncomputable def polynomial (a : ℕ → R) (n : ℕ) : R → R :=
  λ x, ∑ i in Finset.range (n + 1), a i * x ^ i

theorem highest_coeff_bound_no_roots {R : Type*} [LinearOrderedField R] (a : ℕ → R) (n : ℕ) (hmax : ∀ i < n, |a n| ≥ |a i|) :
  ∀ x : R, |x| > 2 → polynomial a n x ≠ 0 :=
sorry

end highest_coeff_bound_no_roots_l771_771152


namespace circumcenter_reflection_l771_771783

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def reflection (P L : Point) : Point := sorry

variables (A B C : Point)
variables (A' B' C' : Point)
variables (O : Point)

theorem circumcenter_reflection (hA' : A' = reflection O A) (hB' : B' = reflection O B) (hC' : C' = reflection O C) :
  circumcenter A B C = O ∧ (A - B) ∥ (A' - B') ∧ (B - C) ∥ (B' - C') ∧ (C - A) ∥ (C' - A') :=
by
  sorry

end circumcenter_reflection_l771_771783


namespace distance_between_points_l771_771233

def point : Type := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  Math.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem distance_between_points :
  distance (-3, 5 : point) (4, -9 : point) = Math.sqrt 245 := 
sorry

end distance_between_points_l771_771233


namespace total_problems_l771_771748

theorem total_problems (x : ℕ) (h1 : 6 * 2 = 3 * x) : 6 + x = 15 :=
by
  have h2 : 2 * x = 18 := by linarith
  have h3 : x = 9 := by linarith
  rw h3
  rfl

end total_problems_l771_771748


namespace smallest_m_l771_771133

noncomputable def S : set ℂ := {z : ℂ | ∃ (x y : ℝ), z = x + y * complex.I ∧ (1/2 : ℝ) ≤ x ∧ x ≤ (2/3 : ℝ)}

def satisfies_condition (n : ℕ) : Prop :=
  ∃ z ∈ S, z ^ n = 1

theorem smallest_m (m : ℕ) (h_m : ∀ n ≥ m, satisfies_condition n) : m = 24 :=
  by
    sorry

end smallest_m_l771_771133


namespace min_dot_product_l771_771461

open Real

-- Definitions of points and vectors
def OA : ℝ × ℝ × ℝ := (1, 2, 3)
def OB : ℝ × ℝ × ℝ := (2, 1, 2)
def OC : ℝ × ℝ × ℝ := (1, 1, 2)

-- Definition of point M moving along the line OC
def M (λ : ℝ) : ℝ × ℝ × ℝ := (λ, λ, 2*λ)

-- Definitions of vectors MA and MB
def MA (λ : ℝ) : ℝ × ℝ × ℝ := (1 - λ, 2 - λ, 3 - 2*λ)
def MB (λ : ℝ) : ℝ × ℝ × ℝ := (2 - λ, 1 - λ, 2 - 2*λ)

-- Dot product function
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- The theorem to prove
theorem min_dot_product :
  ∃ λ : ℝ, dot_product (MA λ) (MB λ) = - (2 / 3) :=
by
  sorry

end min_dot_product_l771_771461


namespace power_evaluation_l771_771393

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l771_771393


namespace height_of_A_l771_771148

open Real

theorem height_of_A (a : ℝ) (α β : ℝ) (hαβ : α < β) :
  tan β ≠ 0 → tan α ≠ 0 →
  let x := a * sin α * sin β / sin (β - α) in
  x = a * sin α * sin β / sin (β - α) :=
by
  intros h1 h2
  let x := a * sin α * sin β / sin (β - α)
  sorry

end height_of_A_l771_771148


namespace intersection_points_circle_l771_771827

-- Defining the two lines based on the parameter u
def line1 (u : ℝ) (x y : ℝ) : Prop := 2 * u * x - 3 * y - 2 * u = 0
def line2 (u : ℝ) (x y : ℝ) : Prop := x - 3 * u * y + 2 = 0

-- Proof statement that shows the intersection points lie on a circle
theorem intersection_points_circle (u x y : ℝ) :
  line1 u x y → line2 u x y → (x - 1)^2 + y^2 = 1 :=
by {
  -- This completes the proof statement, but leaves implementation as exercise
  sorry
}

end intersection_points_circle_l771_771827


namespace count_integers_with_sum_digits_17_l771_771903

-- Define the sum of digits of a number
def sum_digits (n : ℕ) : ℕ :=
  (n / 100) + (n % 100) / 10 + (n % 10)

-- Define the condition that integer is between 400 and 600
def in_range (n : ℕ) : Prop :=
  400 ≤ n ∧ n ≤ 600

-- Define the main theorem
theorem count_integers_with_sum_digits_17 : ∃ k = 13, ∃ (n : ℕ), in_range n ∧ sum_digits n = 17 := by
  sorry

end count_integers_with_sum_digits_17_l771_771903


namespace proof_1_proof_2_l771_771603

noncomputable def problem_1 : Prop :=
  sqrt((3 - Real.pi) ^ 2) + (0.008)^(1/3) + (0.25)^(1/2) * (2 ^ (1/2)) ^ 4 - Real.exp (Real.log Real.pi) = - 4 / 5

noncomputable def problem_2 (x : ℝ) (h : x^2 + 1 ≥ 0) : Prop :=
  Real.log (sqrt(x^2 + 1) + x) + Real.log (sqrt(x^2 + 1) - x) + (log 10 2)^2 + (1 + log 10 2) * log 10 5 = 1

theorem proof_1 : problem_1 :=
  sorry

theorem proof_2 (x : ℝ) (h : x^2 + 1 ≥ 0) : problem_2 x h :=
  sorry

end proof_1_proof_2_l771_771603


namespace magic_square_4x4_is_valid_l771_771765

def is_magic_square_4x4 (M : Matrix (Fin 4) (Fin 4) ℕ) (s : ℕ) : Prop :=
  (∀ i, ∑ j in Finset.univ, M i j = s) ∧  -- sum of each row
  (∀ j, ∑ i in Finset.univ, M i j = s) ∧  -- sum of each column
  (∑ i in Finset.univ, M i i = s) ∧       -- sum of the main diagonal
  (∑ i in Finset.univ, M i (3 - i) = s)   -- sum of the secondary diagonal

def M : Matrix (Fin 4) (Fin 4) ℕ :=
  ![![1, 15, 14, 4], ![12, 6, 7, 9], ![8, 10, 11, 5], ![13, 3, 2, 16]]

theorem magic_square_4x4_is_valid : is_magic_square_4x4 M 34 :=
  sorry

end magic_square_4x4_is_valid_l771_771765


namespace find_f_neg_2017_l771_771986

def f : ℝ → ℝ
| x := if h : 0 ≤ x ∧ x ≤ 2 then x * (2 - x) else -f(x + 2)

theorem find_f_neg_2017 : f (-2017) = -1 := by
  sorry

end find_f_neg_2017_l771_771986


namespace combined_molecular_weight_l771_771225

def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45
def atomic_weight_S : ℝ := 32.07
def atomic_weight_F : ℝ := 19.00

def molecular_weight_CCl4 : ℝ := atomic_weight_C + 4 * atomic_weight_Cl
def molecular_weight_SF6 : ℝ := atomic_weight_S + 6 * atomic_weight_F

def weight_moles_CCl4 (moles : ℝ) : ℝ := moles * molecular_weight_CCl4
def weight_moles_SF6 (moles : ℝ) : ℝ := moles * molecular_weight_SF6

theorem combined_molecular_weight : weight_moles_CCl4 9 + weight_moles_SF6 5 = 2114.64 := by
  sorry

end combined_molecular_weight_l771_771225


namespace count_integers_with_digit_sum_17_l771_771901

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + (n / 100)

theorem count_integers_with_digit_sum_17 : 
  let count := (finset.range 201).filter (λ x, sum_of_digits (x + 400) = 17)
  count.card = 13 :=
by
  sorry

end count_integers_with_digit_sum_17_l771_771901


namespace eval_expression_correct_l771_771394

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l771_771394


namespace number_of_ways_correct_l771_771339

-- Definition used for the problem
def number_of_ways : Nat :=
  -- sorry is used here to ignore the function body, since we focus on statement
  sorry

-- Statement to be proved
theorem number_of_ways_correct : 
  number_of_ways = 114 := sorry

end number_of_ways_correct_l771_771339


namespace verify_fruit_order_l771_771654

-- Define the weights of each type of fruit.
def weight_apples := 4
def weight_oranges := 2
def weight_grapes := 4
def weight_strawberries := 3
def weight_bananas := 1
def weight_pineapples := 3

-- Define the prices per kilogram for each type of fruit.
def price_apple := 2
def price_orange := 3
def price_grape := 2.5
def price_strawberry := 4
def price_banana := 1.5
def price_pineapple := 3.5

-- Calculate the percentages of each type of fruit.
def percentage_apples := (weight_apples / 20) * 100
def percentage_oranges := (weight_oranges / 20) * 100
def percentage_grapes := (weight_grapes / 20) * 100
def percentage_strawberries := (weight_strawberries / 20) * 100
def percentage_bananas := (weight_bananas / 20) * 100
def percentage_pineapples := (weight_pineapples / 20) * 100

-- Calculate the total cost for each type of fruit.
def cost_apples := weight_apples * price_apple
def cost_oranges := weight_oranges * price_orange
def cost_grapes := weight_grapes * price_grape
def cost_strawberries := weight_strawberries * price_strawberry
def cost_bananas := weight_bananas * price_banana
def cost_pineapples := weight_pineapples * price_pineapple

-- Calculate the total cost of all the fruits.
def total_cost := cost_apples + cost_oranges + cost_grapes + cost_strawberries + cost_bananas + cost_pineapples

theorem verify_fruit_order :
  percentage_apples = 20 ∧
  percentage_oranges = 10 ∧
  percentage_grapes = 20 ∧
  percentage_strawberries = 15 ∧
  percentage_bananas = 5 ∧
  percentage_pineapples = 15 ∧
  total_cost = 48 :=
by
  sorry

end verify_fruit_order_l771_771654


namespace sum_of_distinct_squares_l771_771611

theorem sum_of_distinct_squares:
  ∀ (a b c : ℕ),
  a + b + c = 23 ∧ Nat.gcd a b + Nat.gcd b c + Nat.gcd c a = 9 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 + c^2 = 179 ∨ a^2 + b^2 + c^2 = 259 →
  a^2 + b^2 + c^2 = 438 :=
by
  sorry

end sum_of_distinct_squares_l771_771611


namespace evaluate_neg_sixtyfour_exp_four_thirds_l771_771411

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l771_771411


namespace passengers_from_other_continents_l771_771514

theorem passengers_from_other_continents :
  (∀ (n NA EU AF AS : ℕ),
     NA = n / 4 →
     EU = n / 8 →
     AF = n / 12 →
     AS = n / 6 →
     96 = n →
     n - (NA + EU + AF + AS) = 36) :=
by
  sorry

end passengers_from_other_continents_l771_771514


namespace integer_solution_system_l771_771415

theorem integer_solution_system (n : ℕ) (H : n ≥ 2) : 
  ∃ (x : ℕ → ℤ), (
    ∀ i : ℕ, x ((i % n) + 1)^2 + x (((i + 1) % n) + 1)^2 + 50 = 16 * x ((i % n) + 1) + 12 * x (((i + 1) % n) + 1)
  ) ↔ n % 3 = 0 :=
by
  sorry

end integer_solution_system_l771_771415


namespace amount_john_paid_l771_771098

-- Define the conditions.
def total_candy_bars : ℕ := 20
def candy_bars_paid_by_dave : ℕ := 6
def cost_per_candy_bar : ℝ := 1.50

-- This is the Lean 4 statement of our math proof problem.
theorem amount_john_paid : 
  let candy_bars_paid_by_john := total_candy_bars - candy_bars_paid_by_dave in
  let total_cost := candy_bars_paid_by_john * cost_per_candy_bar in
  total_cost = 21 :=
by
  -- Proof is omitted.
  sorry

end amount_john_paid_l771_771098


namespace color_rotation_l771_771713

theorem color_rotation
  (k : ℕ)
  (P : Fin (2 * k) → Bool)
  (Q : Fin (2 * k) → Bool)
  (hP_color_balance : (∑ i, if P i then (1:ℤ) else (-1)) = 0)
  (hQ_color_balance : (∑ i, if Q i then (1:ℤ) else (-1)) = 0):
  ∃ j, (∑ i, if P i = Q ((i + j) % (2 * k)) then (1:ℤ) else (-1)) ≤ -k ∨ 
       (∑ i, if P i = Q ((i + j) % (2 * k)) then (1:ℤ) else (-1)) ≥ k :=
by
  sorry

end color_rotation_l771_771713


namespace russian_pairing_probability_l771_771616

noncomputable theory

def total_players : ℕ := 10

def russian_players : ℕ := 4

def probability_all_russians_only_play_russians : ℚ :=
  1 / 21

theorem russian_pairing_probability :
  total_players = 10 →
  russian_players = 4 →
  -- Prove that the probability that all Russian players are paired only with other Russians is 1/21.
  (∃ (p : ℚ), p = 1 / 21) :=
by {
  intros _ _,
  use 1 / 21,
  sorry
}

end russian_pairing_probability_l771_771616


namespace area_ratio_of_squares_l771_771162

theorem area_ratio_of_squares (y : ℝ) (hy : y > 0) : 
  let areaC := y^2
      areaD := (5 * y) ^ 2
  in areaC / areaD = 1 / 25 := by
  sorry

end area_ratio_of_squares_l771_771162


namespace assign_grades_l771_771586

def is_not_first_grader : Type := sorry
def one_year_older (misha dima : Type) : Prop := sorry
def different_streets (vasya ivanov : Type) : Prop := sorry
def neighbors (boris orlov : Type) : Prop := sorry
def met_one_year_ago (krylov petrov : Type) : Prop := sorry
def gave_textbook_last_year (vasya boris : Type) : Prop := sorry

theorem assign_grades 
  (name : Type) 
  (surname : Type) 
  (grade : Type) 
  (Dima Misha Boris Vasya : name) 
  (Ivanov Krylov Petrov Orlov : surname)
  (first second third fourth : grade)
  (h1 : ¬is_not_first_grader Boris)
  (h2 : different_streets Vasya Ivanov)
  (h3 : one_year_older Misha Dima)
  (h4 : neighbors Boris Orlov)
  (h5 : met_one_year_ago Krylov Petrov)
  (h6 : gave_textbook_last_year Vasya Boris) : 
  (Dima, Ivanov, first) ∧
  (Misha, Krylov, second) ∧
  (Boris, Petrov, third) ∧
  (Vasya, Orlov, fourth) :=
sorry

end assign_grades_l771_771586


namespace calculate_area_of_region_with_segments_of_unequal_lengths_l771_771941

theorem calculate_area_of_region_with_segments_of_unequal_lengths :
  ∀ (r a d : ℝ) (m n d_prime : ℕ),
    r = 50 → a = 84 → d = 24 →
    ∃ (area : ℝ), area = m * π - n * real.sqrt d_prime ∧ 
    (∀ d_prime_not_square (hp : d_prime_not_square ∉ nat.squarefree), 
      d_prime_not_square < 0) → m + n + d_prime = 2503 :=
by
  intro r a d m n d_prime hr ha hd
  use m * π - n * real.sqrt d_prime
  split
  { sorry }
  { intros d_prime_not_square hp
    sorry }

end calculate_area_of_region_with_segments_of_unequal_lengths_l771_771941


namespace prob_digit7_in_8_over_11_l771_771293

theorem prob_digit7_in_8_over_11 : 
  (∃ (x : ℚ), x = 8 / 11 ∧ repeating_decimal x = [7, 2] ∧ prob_digit_in_list 7 [7, 2] = 1 / 2) :=
sorry

end prob_digit7_in_8_over_11_l771_771293


namespace isosceles_triangle_construction_l771_771343

noncomputable def isosceles_triangle_construction_impossible 
  (hb lb : ℝ) : Prop :=
  ∀ (α β : ℝ), 
  3 * β ≠ α

theorem isosceles_triangle_construction : 
  ∃ (hb lb : ℝ), isosceles_triangle_construction_impossible hb lb :=
sorry

end isosceles_triangle_construction_l771_771343


namespace distance_between_points_l771_771232

def point : Type := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  Math.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem distance_between_points :
  distance (-3, 5 : point) (4, -9 : point) = Math.sqrt 245 := 
sorry

end distance_between_points_l771_771232


namespace correct_proposition_l771_771319

variables (L₁ L₂ : Line) (P : Plane)

-- Representing the intersecting lines condition.
def intersecting_lines (L₁ L₂ : Line) : Prop :=
  ∃ P, P ⊂ L₁ ∧ P ⊂ L₂

-- Statement of proposition A in Lean
def proposition_A : Prop :=
  ∀ (L₁ L₂ : Line), intersecting_lines L₁ L₂ → ∃! P, Plane_contains_P P L₁ ∧ Plane_contains_P P L₂

-- Theorem to be proved
theorem correct_proposition : proposition_A :=
sorry

end correct_proposition_l771_771319


namespace range_of_a_for_decreasing_function_l771_771034

theorem range_of_a_for_decreasing_function:
  (∀ x : ℝ, x ∈ set.Ici (-1) → (deriv (λ x, -x^2 - 2*(a - 1)*x + 5) x ≤ 0)) → a ≤ 2 :=
begin
  sorry
end

end range_of_a_for_decreasing_function_l771_771034


namespace max_y_value_l771_771787

noncomputable def y (x : ℝ) : ℝ :=
  sin (x + π/4) + sin (x + π/3) * cos (x + π/6)

theorem max_y_value :
  ∃ x, (-π/4 ≤ x ∧ x ≤ π/12) ∧ y x = 1/2 + 0.96592 := sorry

end max_y_value_l771_771787


namespace probability_A_selected_l771_771245

theorem probability_A_selected (group_size : ℕ) (choose_count : ℕ) (basic_events : ℕ) (A_selected_events : ℕ)
  (h1 : group_size = 5) (h2 : choose_count = 2) (h3 : basic_events = 10) (h4 : A_selected_events = 4) :
  (A_selected_events : ℝ) / (basic_events : ℝ) = 0.4 :=
begin
  -- Provided conditions 
  have h_group_size : group_size = 5 := h1,
  have h_choose_count : choose_count = 2 := h2,
  have h_basic_events : basic_events = 10 := h3,
  have h_A_selected_events : A_selected_events = 4 := h4,
  sorry
end

end probability_A_selected_l771_771245


namespace range_of_a_l771_771518

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 2^(2*x) + 2^x * a + a + 1 = 0) ↔ a ∈ Iic (2 - 2*sqrt 2) := 
sorry

end range_of_a_l771_771518


namespace sequence_inequality_l771_771449

theorem sequence_inequality {m : ℝ} (a b : ℕ → ℝ) (n : ℕ) (hn : n > 0)
  (h_a1 : a 1 = 1)
  (h_a_recur : ∀ n : ℕ, a (n + 1) = a n + 2 * n + 1)
  (h_b : ∀ n : ℕ, b n = a n - 1)
  (h_ineq : ∑ i in Finset.range n, 1 / b (i + 1) < m) : m ≥ 3 / 4 :=
sorry

end sequence_inequality_l771_771449


namespace arithmetic_sequence_a5_l771_771867

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) :=
  ∃ a_1 d, ∀ n, a (n + 1) = a_1 + n * d

theorem arithmetic_sequence_a5 (a : ℕ → α) (h_seq : is_arithmetic_sequence a) (h_cond : a 1 + a 7 = 12) :
  a 4 = 6 :=
by
  sorry

end arithmetic_sequence_a5_l771_771867


namespace magnitude_of_c_l771_771489

noncomputable def magnitude (vec : ℝ × ℝ) : ℝ :=
  real.sqrt (vec.1 ^ 2 + vec.2 ^ 2)

theorem magnitude_of_c (k : ℝ) (h : (1, 4) = (-2, k) ∨ (1, 4) = (2, -k)) :
  magnitude (-2, k) = 2 * real.sqrt 17 :=
by
  unfold magnitude
  sorry

end magnitude_of_c_l771_771489


namespace museum_group_time_l771_771501

theorem museum_group_time :
  ∀ (total_students groups : ℕ) (time_per_student : ℕ),
  total_students = 18 →
  groups = 3 →
  time_per_student = 4 →
  (total_students / groups) * time_per_student = 24 := by
  intros total_students groups time_per_student h1 h2 h3
  rw [h1, h2, h3]
  exact Nat.mul_div_cancel 24 (by norm_num)  -- Here, you can use some tactics to handle arithmetic operations
  sorry  -- This is your placeholder where you would normally complete the proof


end museum_group_time_l771_771501


namespace balloon_volume_safety_l771_771719

theorem balloon_volume_safety (p V : ℝ) (h_prop : p = 90 / V) (h_burst : p ≤ 150) : 0.6 ≤ V :=
by {
  sorry
}

end balloon_volume_safety_l771_771719


namespace max_square_inequality_l771_771458

theorem max_square_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h₁ : (∑ i in range n, a i) = 0) : 
  max (λ i, (a i)^2) (range n) ≤ (n / 3) * (∑ i in range (n-1), (a i - a (i+1))^2) :=
by
  sorry

end max_square_inequality_l771_771458


namespace union_complement_l771_771040

universe u

namespace set_proof

open set

-- Define the universal set U
def U : set ℕ := { x | x < 4 }

-- Set A as given
def A : set ℕ := { 0, 1, 2 }

-- Set B as given
def B : set ℕ := { 2, 3 }

-- Prove B ∪ (U \ A) = { 2, 3 }
theorem union_complement :
  B ∪ (U \ A) = { 2, 3 } :=
by
  sorry

end set_proof

end union_complement_l771_771040


namespace leak_empty_tank_time_l771_771590

-- Define the rates and the problem
theorem leak_empty_tank_time :
  let A := (1 : ℝ) / 4  -- Pipe A's filling rate without leak
  let combined_rate := (1 : ℝ) / 6  -- Combined rate with leak
  let L := A - combined_rate  -- Leak rate calculation
  L = (1 : ℝ) / 12 →  -- Correct leak rate
  (1 : ℝ) / L = 12 := -- Time for the leak alone to empty the tank should be 12 hours
by
  intro A combined_rate L hL
  rw hL
  field_simp
  norm_num

end leak_empty_tank_time_l771_771590


namespace collinear_points_y_l771_771454

theorem collinear_points_y (y : ℝ) 
  (h : ∃ (A B C : ℝ × ℝ), A = (4, 8) ∧ B = (2, 4) ∧ C = (3, y) 
  ∧ (∀ (P Q R : ℝ × ℝ), P = A ∧ Q = B ∧ R = C → collinear P Q R)) : y = 6 :=
by
  sorry

end collinear_points_y_l771_771454


namespace area_quadrilateral_EFGH_l771_771539

-- Define the lengths of the sides of the quadrilateral EFGH and the right angle condition at ∠EFG.
def EF : ℝ := 5
def FG : ℝ := 12
def GH : ℝ := 5
def HE : ℝ := 13
def right_angle (θ : ℝ) : Prop := θ = 90

-- Define the points and the right angle at EFG
axiom E : Type
axiom F : E
axiom G : E
axiom H : E
axiom length_EF : dist F G = EF
axiom length_FG : dist G H = FG
axiom length_GH : dist H E = GH
axiom length_HE : dist E F = HE
axiom angle_EFG : right_angle (angle F E G)

-- State the theorem
theorem area_quadrilateral_EFGH : area (EFGH) = 30 :=
by {
  sorry
}

end area_quadrilateral_EFGH_l771_771539


namespace min_fraction_value_l771_771437

noncomputable def min_value_fraction (x y : ℝ) (hx: x > 0) (hy: y > 0) (h: 2 * x + y = 2) : Prop :=
  (frac 1 x + frac 1 y = frac 3 2 + sqrt 2)

theorem min_fraction_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  frac 1 x + frac 1 y >= frac 3 2 + sqrt 2 :=
sorry

end min_fraction_value_l771_771437


namespace candy_bar_calories_l771_771648

theorem candy_bar_calories (calories : ℕ) (bars : ℕ) (dozen : ℕ) (total_calories : ℕ) 
  (H1 : total_calories = 2016) (H2 : bars = 42) (H3 : dozen = 12) 
  (H4 : total_calories = bars * calories) : 
  calories / dozen = 4 := 
by 
  sorry

end candy_bar_calories_l771_771648


namespace smallest_n_l771_771568

def count_trailing_zeros (n : ℕ) : ℕ :=
  let rec count_helper (x : ℕ) (d : ℕ) : ℕ :=
    if d > x then 0 else x / d + count_helper x (d * 5)
  in count_helper n 5

theorem smallest_n : ∃ n : ℕ, (n > 5 ∧
  (count_trailing_zeros (2 * n) = 2 * count_trailing_zeros n + 1) ∧
  ∀ m : ℕ, (m > 5 ∧ (count_trailing_zeros (2 * m) = 2 * count_trailing_zeros m + 1) → m ≥ n)) :=
by {
  let k := count_trailing_zeros 15,
  have h1 := (count_trailing_zeros (2 * 15) = 2 * k + 1),
  have h2 : ∀ m, m > 5 ∧ count_trailing_zeros (2 * m) = 2 * count_trailing_zeros m + 1 → m ≥ 15, 
  { intros m hm,
    sorry },
  existsi 15,
  exact ⟨hgt, h1, h2⟩
}

end smallest_n_l771_771568


namespace non_intersecting_pairs_exists_l771_771850

open Set

theorem non_intersecting_pairs_exists (T : Set (ℝ × ℝ)) (hT : ∃ n, T.card = 2 * n) :
  ∃ P : Set (Set (ℝ × ℝ)), (∀ p ∈ P, p.card = 2) ∧ (P.pairwise Disjoint ∧ P ⊆ T):  sorry

end non_intersecting_pairs_exists_l771_771850


namespace transformation_parameters_l771_771628

-- Define the function f and h with the given conditions
variable (f : ℝ → ℝ)

theorem transformation_parameters (a d c : ℝ) :
  (∀ x, h(x) = a * f(c * x) + d) →
  (∀ x, h(x) = -f(3 * x) + 3) → 
  (a = -1) ∧ (d = 3) ∧ (c = 3) := 
by
  sorry

end transformation_parameters_l771_771628


namespace range_of_x_l771_771446

variable {f : ℝ → ℝ}
variable (hf1 : ∀ x : ℝ, has_deriv_at f (derivative f x) x)
variable (hf2 : ∀ x : ℝ, derivative f x > - f x)

theorem range_of_x (h : f (Real.log 3) = 1/3) : 
  {x : ℝ | f x > 1 / Real.exp x} = Set.Ioi (Real.log 3) := 
by 
  sorry

end range_of_x_l771_771446


namespace distance_between_points_l771_771230

theorem distance_between_points :
  let x1 := -3
  let y1 := 5
  let x2 := 4
  let y2 := -9
  sqrt ((x2 - x1)^2 + (y2 - y1)^2) = sqrt 245 :=
by
  sorry

end distance_between_points_l771_771230


namespace card_probability_l771_771663

theorem card_probability :
  let hearts := 13
  let clubs := 13
  let total_cards := 52
  let first_card_is_heart := (hearts.to_rat / total_cards.to_rat)
  let second_card_is_club_given_first_is_heart := (clubs.to_rat / (total_cards - 1).to_rat)
  first_card_is_heart * second_card_is_club_given_first_is_heart = (13.to_rat / 204.to_rat) := by
  sorry

end card_probability_l771_771663


namespace inequality_holds_for_all_x_l771_771426

theorem inequality_holds_for_all_x : 
  ∀ (a : ℝ), (∀ (x : ℝ), |x| ≤ 1 → x^2 - (a + 1) * x + a + 1 > 0) ↔ a < -1 := 
sorry

end inequality_holds_for_all_x_l771_771426


namespace sin_cos_eq_l771_771982

theorem sin_cos_eq (a : ℝ) :
    (∀ x ∈ set.Icc (0:ℝ) (2 * Real.pi), ∀ x1 x2 x3,
    (sin x + Real.sqrt 3 * cos x = a) → (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)) ↔ (a = Real.sqrt 3) :=
by
  sorry

end sin_cos_eq_l771_771982


namespace problem_solution_l771_771525

def sum_of_arithmetic_series (a l n : ℕ) : ℕ :=
  n * (a + l) / 2

def count_even_integers_in_range (a l : ℕ) : ℕ :=
  (l - a) / 2 + 1

def product_of_odd_integers_in_range (a l : ℕ) : ℕ :=
  (a + 1).stepFun (λ n, 2*n - 1) * (l + 1) / 2

theorem problem_solution :
  let x := sum_of_arithmetic_series 20 30 11 in
  let y := count_even_integers_in_range 20 30 in
  let z := (21 * 23 * 25 * 27 * 29) in
  x + y + z = 4807286 :=
by
  sorry

end problem_solution_l771_771525


namespace math_problem_proof_l771_771352

--- Conditions
variable (a b : ℝ)
ax2_bx_1_gt_0 : ∀ x : ℝ, 1 < x ∧ x < 2 → a * x ^ 2 + b * x - 1 > 0

--- Questions and Answers
noncomputable def find_ab : Prop :=
a = -1 / 2 ∧ b = 3 / 2

noncomputable def solve_inequality : Set ℝ :=
{x | (2 / 3) < x ∧ x < 2}

--- Lean statement
theorem math_problem_proof : 
find_ab a b 
∧ (∀ x : ℝ, ((a * x + 1) / (b * x - 1) > 0) ↔ x ∈ solve_inequality) :=
by sorry

end math_problem_proof_l771_771352


namespace eval_expression_correct_l771_771395

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l771_771395


namespace museum_group_time_l771_771502

theorem museum_group_time :
  ∀ (total_students groups : ℕ) (time_per_student : ℕ),
  total_students = 18 →
  groups = 3 →
  time_per_student = 4 →
  (total_students / groups) * time_per_student = 24 := by
  intros total_students groups time_per_student h1 h2 h3
  rw [h1, h2, h3]
  exact Nat.mul_div_cancel 24 (by norm_num)  -- Here, you can use some tactics to handle arithmetic operations
  sorry  -- This is your placeholder where you would normally complete the proof


end museum_group_time_l771_771502


namespace range_of_a_l771_771717

noncomputable def f (x a : ℝ) : ℝ := x^3 - a*x - 1
noncomputable def df (x a : ℝ) : ℝ := 3*x^2 - a

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-1 : ℝ) 1, df x a ≤ 0
def q (a : ℝ) : Prop := ∃ x_0 : ℝ, x_0^2 + a*x_0 + 1 ≤ 0

theorem range_of_a : {a : ℝ | (p a ∨ q a) ∧ ¬ (p a ∧ q a)} = Set.Icc (-∞) (-2) ∪ Set.Ico (2 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_a_l771_771717


namespace quadratic_inequality_false_iff_l771_771931

open Real

theorem quadratic_inequality_false_iff (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
sorry

end quadratic_inequality_false_iff_l771_771931


namespace flour_price_increase_l771_771653

theorem flour_price_increase (x : ℝ) (hx : x > 0) :
  (9600 / (1.5 * x) - 6000 / x = 0.4) :=
by 
  sorry

end flour_price_increase_l771_771653


namespace sum_of_four_consecutive_even_integers_l771_771639

theorem sum_of_four_consecutive_even_integers (x : ℕ) (hx : x > 4) :
  (x - 4) * (x - 2) * x * (x + 2) = 48 * (4 * x) → (x - 4) + (x - 2) + x + (x + 2) = 28 := by
{
  sorry
}

end sum_of_four_consecutive_even_integers_l771_771639


namespace radius_range_result_l771_771290

def ellipse : set (ℝ × ℝ) := {p | (p.1^2 / 9) + (p.2^2 / 4) = 1}

def foci : set (ℝ × ℝ) := {p | p = (sqrt 5, 0) ∨ p = (-sqrt 5, 0)}

def circle (r : ℝ) (center : ℝ × ℝ) : set (ℝ × ℝ) := 
{p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2}

theorem radius_range : ∀ (r : ℝ), r ∈ set.Icc (sqrt 5) 9 → 
  ∃ c : ℝ × ℝ, (circle r c).inter ellipse = (foci : set (ℝ × ℝ)) :=
sorry

theorem result : 
  let a := sqrt 5 in
  let b := 9 in
  a + b = sqrt 5 + 9 :=
by
  simp [sqrt, add_comm]
  norm_num
  exact rfl

end radius_range_result_l771_771290


namespace problem1_problem2_problem3_problem3_min_value_case1_problem3_min_value_case2_l771_771562

-- Define the set M_a
def M_a (a : ℝ) (f : ℝ → ℝ) := ∀ x, f (x + a) > f x

-- Problem 1: f(x) = 2^x - x^2
def f (x : ℝ) : ℝ := 2^x - x^2
theorem problem1 : ¬ M_a 1 f :=
sorry

-- Problem 2: g(x) = x^3 - 1/4*x + 3
def g (x : ℝ) : ℝ := x^3 - (1/4)*x + 3
theorem problem2 (a : ℝ) : M_a a g ↔ a > 1 :=
sorry

-- Problem 3: h(x) = log_3(x + k/x), x ∈ [1, +∞)
noncomputable def h (k : ℝ) (x : ℝ) : ℝ := log 3 (x + k / x)
theorem problem3 (k : ℝ) : M_a 2 (h k) ↔ -1 < k ∧ k < 3 :=
sorry

theorem problem3_min_value_case1 (k : ℝ) (hk : -1 < k ∧ k < 1) : ∃ x, 1 ≤ x ∧ h k x = log 3 (1 + k) :=
sorry

theorem problem3_min_value_case2 (k : ℝ) (hk : 1 ≤ k ∧ k < 3) : ∃ x, 1 ≤ x ∧ h k x = log 3 (2 * sqrt k) :=
sorry

end problem1_problem2_problem3_problem3_min_value_case1_problem3_min_value_case2_l771_771562


namespace abc_sum_l771_771172

theorem abc_sum
  (a b c : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 17 * x + 70 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 - 19 * x + 84 = (x - b) * (x - c)) :
  a + b + c = 29 := by
  sorry

end abc_sum_l771_771172


namespace range_of_a_l771_771984

noncomputable def f (x a : ℝ) : ℝ := x * abs (x - a)

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), 3 ≤ x1 ∧ 3 ≤ x2 ∧ x1 ≠ x2 → (x1 - x2) * (f x1 a - f x2 a) > 0) → a ≤ 3 :=
by sorry

end range_of_a_l771_771984


namespace slopeRange_l771_771844

noncomputable def lineIntersectsCircleAtTwoPoints (l : ℝ → ℝ) := 
  (∃ p₁ p₂ : ℝ × ℝ, 
    p₁ ≠ p₂ ∧ 
    p₁.1^2 + p₁.2^2 = 2 * p₁.1 ∧ 
    p₂.1^2 + p₂.2^2 = 2 * p₂.1 ∧ 
    p₁.2 = l p₁.1 ∧
    p₂.2 = l p₂.1)

theorem slopeRange (k : ℝ) :
  (∀ x, (x + 2) * k) = l →
  lineIntersectsCircleAtTwoPoints l →
  - real.sqrt 2 / 4 < k ∧ k < real.sqrt 2 / 4 :=
sorry

end slopeRange_l771_771844


namespace inequality_induction_step_l771_771219

theorem inequality_induction_step (k : ℕ) (h : k ≥ 2) :
  (∑ i in finset.range (k + 1) \ {0} | i ≥ k + 1 ∧ i ≤ 2 * k, (i : ℝ)⁻¹) -
  (∑ i in finset.range (k + 2) \ {0} | i ≥ k + 2 ∧ i ≤ 2 * k + 2, (i : ℝ)⁻¹) =
  (1 : ℝ) / (2 * k + 1) + (1 : ℝ) / (2 * (k + 1)) - (1 : ℝ) / (k + 1) := sorry

end inequality_induction_step_l771_771219


namespace evaluate_neg_sixtyfour_exp_four_thirds_l771_771407

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l771_771407


namespace calculate_rows_l771_771142

-- Definitions based on conditions
def totalPecanPies : ℕ := 16
def totalApplePies : ℕ := 14
def piesPerRow : ℕ := 5

-- The goal is to prove the total rows of pies
theorem calculate_rows : (totalPecanPies + totalApplePies) / piesPerRow = 6 := by
  sorry

end calculate_rows_l771_771142


namespace evaluate_pow_l771_771362

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l771_771362


namespace prob_selection_and_count_prob_one_female_selected_stability_comparison_l771_771291

-- Problem context definitions
def total_students := 30 + 20 : ℕ
def group_size := 5 : ℕ
def male_students := 30 : ℕ
def female_students := 20 : ℕ
def stratified_sampling_value := (5 : ℝ) / (50 : ℝ)
def group_males := 3 : ℕ
def group_females := 2 : ℕ
def first_student_data := [68, 70, 71, 72, 74]
def second_student_data := [69, 70, 70, 72, 74]

-- Question (1)
theorem prob_selection_and_count : 
  stratified_sampling_value = 1 / 10 ∧ group_males = 3 ∧ group_females = 2 :=
sorry

-- Question (2)
theorem prob_one_female_selected :
  let total_basic_events := group_size * (group_size - 1)
  let favorable_events := group_males * group_females + group_females * group_males
  (favorable_events / total_basic_events : ℝ) = 3 / 5 :=
sorry

-- Variance calculation helper functions
noncomputable def mean (data : list ℕ) : ℝ := (list.sum data : ℝ) / (data.length : ℝ)

noncomputable def variance (data : list ℕ) : ℝ := 
  let m := mean data in
  (list.sum (list.map (λ x, ((x : ℝ) - m) ^ 2) data)) / (data.length : ℝ)

-- Question (3)
theorem stability_comparison :
  variance first_student_data < variance second_student_data :=
sorry

end prob_selection_and_count_prob_one_female_selected_stability_comparison_l771_771291


namespace pedestrian_walking_time_in_interval_l771_771740

noncomputable def bus_departure_interval : ℕ := 5  -- Condition 1: Buses depart every 5 minutes
noncomputable def buses_same_direction : ℕ := 11  -- Condition 2: 11 buses passed him going the same direction
noncomputable def buses_opposite_direction : ℕ := 13  -- Condition 3: 13 buses came from opposite direction
noncomputable def bus_speed_factor : ℕ := 8  -- Condition 4: Bus speed is 8 times the pedestrian's speed
noncomputable def min_walking_time : ℚ := 57 + 1 / 7 -- Correct Answer: Minimum walking time
noncomputable def max_walking_time : ℚ := 62 + 2 / 9 -- Correct Answer: Maximum walking time

theorem pedestrian_walking_time_in_interval (t : ℚ)
  (h1 : bus_departure_interval = 5)
  (h2 : buses_same_direction = 11)
  (h3 : buses_opposite_direction = 13)
  (h4 : bus_speed_factor = 8) :
  min_walking_time ≤ t ∧ t ≤ max_walking_time :=
sorry

end pedestrian_walking_time_in_interval_l771_771740


namespace probability_sum_of_three_dice_is_9_l771_771524

def sum_of_three_dice_is_9 : Prop :=
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 9)

theorem probability_sum_of_three_dice_is_9 : 
  (∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 → a + b + c = 9 → sum_of_three_dice_is_9) ∧ 
  (1 / 216 = 25 / 216) := 
by
  sorry

end probability_sum_of_three_dice_is_9_l771_771524


namespace bowling_ball_weight_l771_771823

theorem bowling_ball_weight (b c : ℝ) (h1 : 5 * b = 3 * c) (h2 : 2 * c = 56) : b = 16.8 := by
  sorry

end bowling_ball_weight_l771_771823


namespace evaluate_neg_sixtyfour_exp_four_thirds_l771_771410

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l771_771410


namespace klinker_age_l771_771580

theorem klinker_age (K D : ℕ) (h1 : D = 10) (h2 : K + 15 = 2 * (D + 15)) : K = 35 :=
by
  sorry

end klinker_age_l771_771580


namespace inverse_of_h_l771_771975

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x^2 - 2
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem inverse_of_h (x y : ℝ) :
  (h y = x ↔ y = sqrt ((x + 3) / 12) ∨ y = -sqrt ((x + 3) / 12)) :=
by
  sorry

end inverse_of_h_l771_771975


namespace wine_distribution_l771_771837

theorem wine_distribution (m n k s : ℕ) (h : Nat.gcd m (Nat.gcd n k) = 1) (h_s : s < m + n + k) :
  ∃ g : ℕ, g = s := by
  sorry

end wine_distribution_l771_771837


namespace cannot_equal_120_l771_771306

def positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

theorem cannot_equal_120 (a b : ℕ) (ha : positive_even a) (hb : positive_even b) :
  let A := a * b
  let P' := 2 * (a + b) + 6
  A + P' ≠ 120 :=
sorry

end cannot_equal_120_l771_771306


namespace distance_between_points_l771_771228

theorem distance_between_points :
  let x1 := -3
  let y1 := 5
  let x2 := 4
  let y2 := -9
  sqrt ((x2 - x1)^2 + (y2 - y1)^2) = sqrt 245 :=
by
  sorry

end distance_between_points_l771_771228


namespace kishore_savings_l771_771706

noncomputable def total_expenses : ℝ :=
  5000 + 1500 + 4500 + 2500 + 2000 + 5200

def percentage_saved : ℝ := 0.10

theorem kishore_savings (salary : ℝ) :
  (total_expenses + percentage_saved * salary) = salary → 
  (percentage_saved * salary = 2077.78) :=
by
  intros h
  rw [← h]
  sorry

end kishore_savings_l771_771706


namespace problem_even_and_decreasing_l771_771836

def f (x : ℝ) : ℝ := (1 / 2) ^ |x|

-- Definitions for the properties we need to prove
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → f x > f y

theorem problem_even_and_decreasing : is_even_function f ∧ is_decreasing_on_pos f :=
by
  sorry

end problem_even_and_decreasing_l771_771836


namespace patrick_race_time_l771_771146

variable (t_p t_m t_a : ℕ)

theorem patrick_race_time :
  (t_m = t_p + 12) → (t_a = 36) → (t_a = t_m / 2) → t_p = 60 :=
by
intros h1 h2 h3
-- sorry

end patrick_race_time_l771_771146


namespace monotonicity_of_f_when_a_eq_1_range_of_a_for_two_zeros_l771_771485

section part1

variable (x : ℝ) (a : ℝ)
noncomputable def f (x : ℝ) := Real.log x - a * (x - 1)

theorem monotonicity_of_f_when_a_eq_1 :
  (∀ x ∈ Ioo 0 1, (Real.log x - x + 1) > 0) ∧ (∀ x ∈ Ioi 1, (Real.log x - x + 1) < 0) :=
sorry

end part1

section part2

noncomputable def g (x : ℝ) := (Real.log x) / (x - 1)

theorem range_of_a_for_two_zeros (a : ℝ) :
  (∀ a, (g x > a) ↔ (a ∈ Set.Ioo 0 1 )) ∧ (∀ a ∈ Set.Ioi 1, g x > a) ↔ true :=
sorry

end part2

end monotonicity_of_f_when_a_eq_1_range_of_a_for_two_zeros_l771_771485


namespace probability_heart_then_club_l771_771665

noncomputable def numHearts : ℕ := 13
noncomputable def numClubs : ℕ := 13
noncomputable def totalCards (n : ℕ) : ℕ := 52 - n

noncomputable def probabilityFirstHeart : ℚ := numHearts / totalCards 0
noncomputable def probabilitySecondClubGivenFirstHeart : ℚ := numClubs / totalCards 1

theorem probability_heart_then_club :
  (probabilityFirstHeart * probabilitySecondClubGivenFirstHeart) = 13 / 204 :=
by
  sorry

end probability_heart_then_club_l771_771665


namespace sum_composite_l771_771826

theorem sum_composite (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 34 * a = 43 * b) : ∃ d : ℕ, d > 1 ∧ d < a + b ∧ d ∣ (a + b) :=
by
  sorry

end sum_composite_l771_771826


namespace water_leaving_rate_l771_771177

-- Definitions: Volume of water and time taken
def volume_of_water : ℕ := 300
def time_taken : ℕ := 25

-- Theorem statement: Rate of water leaving the tank
theorem water_leaving_rate : (volume_of_water / time_taken) = 12 := 
by sorry

end water_leaving_rate_l771_771177


namespace sum_base8_l771_771425

theorem sum_base8 (a b c : ℕ) (h₁ : a = 7*8^2 + 7*8 + 7)
                           (h₂ : b = 7*8 + 7)
                           (h₃ : c = 7) :
  a + b + c = 1*8^3 + 1*8^2 + 0*8 + 5 :=
by
  sorry

end sum_base8_l771_771425


namespace range_of_k_l771_771085

theorem range_of_k 
  (hC : ∀ x y : ℝ, x^2 + y^2 - 4 * x = 0 → ∃ C : ℝ × ℝ, C = (2, 0) ∧ ∀ R : ℝ, R = 2) 
  (hP : ∃ P : ℝ × ℝ, ∃ k : ℝ, ∀ x : ℝ, P = (x, k * (x + 1))) 
  (h_tangents_perpendicular : ∀ A B : ℝ × ℝ, P ≠ A ∧ P ≠ B ∧ IsTangent P A ∧ IsTangent P B ∧ ⟪A - P, B - P⟫ = 0) : 
  k ∈ [-2 * Real.sqrt 2, 2 * Real.sqrt 2] :=
sorry

end range_of_k_l771_771085


namespace number_of_male_animals_l771_771070

theorem number_of_male_animals (Horses : ℕ) (Sheep : ℕ) (Chickens : ℕ) (Cows : ℕ) (Pigs : ℕ) 
(Brian_horses_perc : ℝ) (Brian_sheep_perc : ℝ) (Brian_chickens_perc : ℝ) 
(Jeremy_gift_goats : ℕ) 
(Caroline_cows_perc : ℝ) (Caroline_pigs_perc : ℝ)
(Brian_horses_rounding_fn : ℕ → ℕ) (Brian_sheep_rounding_fn : ℕ → ℕ) (Brian_chickens_rounding_fn : ℕ → ℕ)
(Caroline_cows_rounding_fn : ℕ → ℕ) (Caroline_pigs_rounding_fn : ℕ → ℕ)
(Horses_initial := 100)
(Sheep_initial := 29)
(Chickens_initial := 9)
(Cows_initial := 15)
(Pigs_initial := 18)
(Brian_horses := Brian_horses_perc * Horses_initial)
(Brian_sheep := Brian_sheep_perc * Sheep_initial)
(Brian_chickens := Brian_chickens_perc * Chickens_initial)
(Caroline_cows := Caroline_cows_perc * Cows_initial)
(Caroline_pigs := Caroline_pigs_perc * Pigs_initial)
(total_animals := Horses + Sheep + Chickens + Cows + Pigs + Jeremy_gift_goats):
  Brian_horses_perc = 0.4 → Brian_sheep_perc = 0.5 → Brian_chickens_perc = 0.6 → 
  Jeremy_gift_goats = 37 →
  Caroline_cows_perc = 0.3 → Caroline_pigs_perc = 0.2 →
  Horses = 60 → Sheep = 15 → Chickens = 4 → Cows = 11 → Pigs = 15 →
  Horses = Horses_initial - Brian_horses_rounding_fn Brian_horses →
  Sheep = Sheep_initial - Brian_sheep_rounding_fn Brian_sheep →
  Chickens = Chickens_initial - Brian_chickens_rounding_fn Brian_chickens →
  Cows = Cows_initial - Caroline_cows_rounding_fn Caroline_cows →
  Pigs = Pigs_initial - Caroline_pigs_rounding_fn Caroline_pigs →
  total_animals = 142 →
  (total_animals / 2) = 71 :=
sorry

end number_of_male_animals_l771_771070


namespace RecipeCallForFlour_l771_771551

-- Definitions based on the conditions
def FlourAlready := 4
def FlourToAdd := 4
def TotalFlour := FlourAlready + FlourToAdd

-- Theorem statement
theorem RecipeCallForFlour : TotalFlour = 8 :=
by
  simp [TotalFlour, FlourAlready, FlourToAdd]
  sorry

end RecipeCallForFlour_l771_771551


namespace explicit_form_l771_771486

noncomputable def f : ℝ → ℝ :=
sorry

axiom functional_eq : ∀ x y : ℝ, f (f x^2 + f y) = x * f x + y

theorem explicit_form (f:ℝ → ℝ) (condition: ∀ x y : ℝ, f (f x^2 + f y) = x * f x + y) :
  (∀ x, f x = x) ∨ (∀ x, f x = -x) :=
begin
  sorry
end

end explicit_form_l771_771486


namespace square_of_rational_l771_771714

theorem square_of_rational (b : ℚ) : b^2 = b * b :=
sorry

end square_of_rational_l771_771714


namespace coffee_grinder_assembly_time_l771_771598

-- Variables for the assembly rates
variables (h r : ℝ)

-- Definitions of conditions
def condition1 : Prop := h / 4 = r
def condition2 : Prop := r / 4 = h
def condition3 : Prop := ∀ start_time end_time net_added, 
  start_time = 9 ∧ end_time = 12 ∧ net_added = 27 → 3 * 3/4 * h = net_added
def condition4 : Prop := ∀ start_time end_time net_added, 
  start_time = 13 ∧ end_time = 19 ∧ net_added = 120 → 6 * 3/4 * r = net_added

-- Theorem statement
theorem coffee_grinder_assembly_time
  (h r : ℝ)
  (c1 : condition1 h r)
  (c2 : condition2 h r)
  (c3 : condition3 h)
  (c4 : condition4 r) :
  h = 12 ∧ r = 80 / 3 :=
sorry

end coffee_grinder_assembly_time_l771_771598


namespace cone_lateral_surface_angle_l771_771292

theorem cone_lateral_surface_angle (x n : ℝ) (h₁ : x > 0)
  (h₂ : 2 * π * (1 / 2) * x = (n * π * x) / 180) : n = 180 :=
by
  have h₃ : 2 * π * (1 / 2) * x = (n * π * x) / 180 := h₂
  linarith

end cone_lateral_surface_angle_l771_771292


namespace tree_heights_l771_771936

theorem tree_heights :
  let elm_height := 12.25 
  let oak_height := 18.5 
  let pine_height := 18.75
  oak_height - elm_height = 6.25 ∧ pine_height - elm_height = 6.5 :=
by {
  unfold elm_height oak_height pine_height,
  norm_num,
  exact ⟨rfl, rfl⟩,
}

end tree_heights_l771_771936


namespace cantaloupe_count_l771_771431

theorem cantaloupe_count (fred_grew : ℕ) (tim_grew : ℕ) (h1 : fred_grew = 38) (h2 : tim_grew = 44) : fred_grew + tim_grew = 82 := 
by
  rw [h1, h2]
  norm_num

end cantaloupe_count_l771_771431


namespace minimum_product_xyz_l771_771977

noncomputable def minimalProduct (x y z : ℝ) : ℝ :=
  3 * x^2 * (1 - 4 * x)

theorem minimum_product_xyz :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  z = 3 * x →
  x ≤ y ∧ y ≤ z →
  minimalProduct x y z = (9 / 343) :=
by
  intros x y z x_pos y_pos z_pos sum_eq1 z_eq3x inequalities
  sorry

end minimum_product_xyz_l771_771977


namespace whales_next_year_l771_771193

theorem whales_next_year (w_last_year : ℕ) (w_this_year : ℕ) (w_next_year : ℕ) :
  w_last_year = 4000 →
  w_this_year = 2 * w_last_year →
  w_next_year = w_this_year + 800 →
  w_next_year = 8800 :=
by
  intros w_last_year_eq w_this_year_eq w_next_year_eq
  rw [w_last_year_eq, w_this_year_eq] at w_next_year_eq
  simp at w_next_year_eq
  exact w_next_year_eq

end whales_next_year_l771_771193


namespace prime_odd_sum_l771_771868

theorem prime_odd_sum (a b : ℕ) (h1 : Prime a) (h2 : Odd b) (h3 : a^2 + b = 2001) : a + b = 1999 :=
sorry

end prime_odd_sum_l771_771868


namespace annual_interest_rate_l771_771615

theorem annual_interest_rate (initial_amount final_amount : ℝ) 
  (h_initial : initial_amount = 90) 
  (h_final : final_amount = 99) : 
  ((final_amount - initial_amount) / initial_amount) * 100 = 10 :=
by {
  sorry
}

end annual_interest_rate_l771_771615


namespace find_relationship_arith_sum_geometric_seq_l771_771571

variable {C : ℕ → ℝ} (C1 d : ℝ) (h_d : d ≠ 0)

-- Assume the sequence {C_n} is an arithmetic sequence
def arithmetic_seq (C : ℕ → ℝ) (C1 d : ℝ) : Prop :=
  ∀ n : ℕ, C n = C1 + n * d

-- Define the sum of the first n terms of an arithmetic sequence
def S (n : ℕ) : ℝ := n * C1 + (n * (n - 1) / 2) * d

-- {C_n} is a sum-geometric sequence
def sum_geometric_seq (C : ℕ → ℝ) : Prop :=
  ∃ x ≠ 0, ∀ n : ℕ, (n > 0) → S (2 * n) / S n = x

-- The goal is to prove that d = 2 * C1
theorem find_relationship_arith_sum_geometric_seq :
  arithmetic_seq C C1 d →
  sum_geometric_seq C →
  d = 2 * C1 := 
by
  intros h_arith h_sum_geom
  sorry

end find_relationship_arith_sum_geometric_seq_l771_771571


namespace minimize_square_sum_l771_771095

theorem minimize_square_sum (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : 
  ∃ x y z, (x + 2 * y + 3 * z = 1) ∧ (x^2 + y^2 + z^2 ≥ 0) ∧ ((x^2 + y^2 + z^2) = 1 / 14) :=
sorry

end minimize_square_sum_l771_771095


namespace limit_of_volumes_l771_771198

noncomputable def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def radii (n : ℕ) : ℝ := (1 / 2)^n

def volumes (n : ℕ) : ℝ := volume (radii n)

theorem limit_of_volumes : 
  tendsto (λ n, ∑ i in Finset.range n, volumes i) atTop (𝓝 (32 / 21 * Real.pi)) :=
by sorry

end limit_of_volumes_l771_771198


namespace eval_expression_correct_l771_771399

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l771_771399


namespace decreasing_function_iff_m_eq_2_l771_771189

theorem decreasing_function_iff_m_eq_2 
    (m : ℝ) : 
    (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(-5*m - 3) < (m^2 - m - 1) * (x + 1)^(-5*m - 3)) ↔ m = 2 := 
sorry

end decreasing_function_iff_m_eq_2_l771_771189


namespace minimal_cost_to_form_2009_l771_771682

-- define the denominations
constant denom : set ℕ := {1, 2, 5, 10}

-- define the target number
constant target_num : ℕ := 2009

-- define the optimal cost
constant min_cost : ℕ := 23

-- define condition that a number can be formed using given denominations with a certain cost
def formable_with_cost (n cost : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ → ℕ), 
    (∀ x y, x ∈ denom → y ∈ denom → f x y ∈ denom ∧ f (f x y) (f x y) = n ∧ cost = sum (denom.to_list))

-- state the theorem
theorem minimal_cost_to_form_2009 : 
  formable_with_cost target_num min_cost :=
sorry

end minimal_cost_to_form_2009_l771_771682


namespace symmetric_point_min_value_l771_771930

theorem symmetric_point_min_value (a b : ℝ) 
  (h1 : a > 0 ∧ b > 0) 
  (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ - 2 = 0 ∧ 2 * x₀ + y₀ + 3 = 0 ∧ 
        a + b = x₀ + y₀ ∧ ∃ k, k = (y₀ - b) / (x₀ - a) ∧ y₀ = k * x₀ + 2 - k * (a + k * b))
   : ∃ α β, a = β / α ∧  b = 2 * β / α ∧ (1 / a + 8 / b) = 25 / 9 :=
sorry

end symmetric_point_min_value_l771_771930


namespace imaginary_part_of_z_l771_771629

def z : ℂ := 1 - 2 * complex.i

theorem imaginary_part_of_z : complex.im z = -2 := sorry

end imaginary_part_of_z_l771_771629


namespace find_first_number_l771_771164

theorem find_first_number (A B LCM HCF : ℕ) (h_LCM : LCM = 2310) (h_HCF : HCF = 30) (h_B : B = 150) :
  A * B = LCM * HCF → A = 462 :=
by {
  intro h_product,
  rw [h_B, h_LCM, h_HCF] at h_product,
  have h1 : A * 150 = 2310 * 30, by exact h_product,
  have h2 : A = (2310 * 30) / 150, by exact (nat.mul_div_cancel' (nat.gcd_dvd_mul_mul_div_right h_product)).symm,
  linarith
}

end find_first_number_l771_771164


namespace remainder_25197629_mod_4_l771_771515

theorem remainder_25197629_mod_4 : 25197629 % 4 = 1 := by
  sorry

end remainder_25197629_mod_4_l771_771515


namespace intersection_of_A_and_B_l771_771862

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_A_and_B : (A ∩ B) = {0, 1} := by
  sorry

end intersection_of_A_and_B_l771_771862


namespace probability_heart_then_club_l771_771667

noncomputable def numHearts : ℕ := 13
noncomputable def numClubs : ℕ := 13
noncomputable def totalCards (n : ℕ) : ℕ := 52 - n

noncomputable def probabilityFirstHeart : ℚ := numHearts / totalCards 0
noncomputable def probabilitySecondClubGivenFirstHeart : ℚ := numClubs / totalCards 1

theorem probability_heart_then_club :
  (probabilityFirstHeart * probabilitySecondClubGivenFirstHeart) = 13 / 204 :=
by
  sorry

end probability_heart_then_club_l771_771667


namespace Dima_claim_false_l771_771789

theorem Dima_claim_false (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a*x^2 + b*x + c = 0) → ∃ α β, α < 0 ∧ β < 0 ∧ (α + β = -b/a) ∧ (α*β = c/a)) :
  ¬ ∃ α' β', α' > 0 ∧ β' > 0 ∧ (α' + β' = -c/b) ∧ (α'*β' = a/b) :=
sorry

end Dima_claim_false_l771_771789


namespace card_probability_l771_771664

theorem card_probability :
  let hearts := 13
  let clubs := 13
  let total_cards := 52
  let first_card_is_heart := (hearts.to_rat / total_cards.to_rat)
  let second_card_is_club_given_first_is_heart := (clubs.to_rat / (total_cards - 1).to_rat)
  first_card_is_heart * second_card_is_club_given_first_is_heart = (13.to_rat / 204.to_rat) := by
  sorry

end card_probability_l771_771664


namespace number_of_n_such_that_21n_is_perfect_square_l771_771817

theorem number_of_n_such_that_21n_is_perfect_square :
  {n : ℕ | n ≤ 1000 ∧ (∃ m : ℕ, 21 * n = m^2)}.card = 6 := sorry

end number_of_n_such_that_21n_is_perfect_square_l771_771817


namespace probability_of_A9B8C_l771_771951

theorem probability_of_A9B8C :
  (let vowels := ['A', 'E', 'I', 'O', 'U'] in
   let digits := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] in
   let non_vowels := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 
                      'K', 'L', 'M', 'N', 'P', 'Q', 'R',
                      'S', 'T', 'V', 'W', 'X', 'Y', 'Z'] in
   let total := 5 * 10 * 21 * 20 * 9 in
   (1 : ℝ) / total = 1 / 189000) :=
sorry

end probability_of_A9B8C_l771_771951


namespace radius_of_spheres_l771_771447

-- Define the regular tetrahedron with edge length a
variable (a : ℝ)
-- Define the radius r of the spheres
variable (r : ℝ)

-- Define the conditions
-- Regular tetrahedron height
def height_tetrahedron (a : ℝ) : ℝ := (Real.sqrt 6 * a) / 3

-- Height of the inner tetrahedron formed by sphere centers
def height_inner_tetrahedron (r : ℝ) : ℝ := (Real.sqrt 6 * 2 * r) / 3

-- The combined vertical distance involving both spheres and inner tetrahedron height
def total_vertical_distance (r : ℝ) : ℝ := 4 * r + height_inner_tetrahedron r

-- The height of the larger tetrahedron must match the combined vertical distance consideration
theorem radius_of_spheres 
  (h₁ : total_vertical_distance r = height_tetrahedron a) :
  r = ((Real.sqrt 6 - 1) * a) / 10 :=
sorry

end radius_of_spheres_l771_771447


namespace total_legs_l771_771619

def animals_legs (dogs : Nat) (birds : Nat) (insects : Nat) : Nat :=
  (dogs * 4) + (birds * 2) + (insects * 6)

theorem total_legs :
  animals_legs 3 2 2 = 22 := by
  sorry

end total_legs_l771_771619


namespace sum_of_squares_identity_l771_771703

noncomputable def sum_of_squares_of_chords (n : ℕ) : ℝ :=
∑ 1 ≤ i < j ≤ n, (2 * sin(Real.pi * (i - j) / n))^2

theorem sum_of_squares_identity (n : ℕ) (h : 2 ≤ n) :
  ∑ 1 ≤ i < j ≤ n, (2 * sin(Real.pi * (i - j) / n))^2 = n^2 * sqrt(2 - 2 * cos(2 * Real.pi / n)) :=
sorry

end sum_of_squares_identity_l771_771703


namespace find_value_of_A_l771_771641

theorem find_value_of_A (A ω φ c : ℝ)
  (a : ℕ+ → ℝ)
  (h_seq : ∀ n : ℕ+, a n * a (n + 1) * a (n + 2) = a n + a (n + 1) + a (n + 2))
  (h_neq : ∀ n : ℕ+, a n * a (n + 1) ≠ 1)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 2)
  (h_form : ∀ n : ℕ+, a n = A * Real.sin (ω * n + φ) + c)
  (h_ω_gt_0 : ω > 0)
  (h_phi_lt_pi_div_2 : |φ| < Real.pi / 2) :
  A = -2 * Real.sqrt 3 / 3 := 
sorry

end find_value_of_A_l771_771641


namespace line_always_passes_through_fixed_point_l771_771180

theorem line_always_passes_through_fixed_point :
  ∀ (a : ℝ), ∃ (x y : ℝ), (a - 1) * x - y + 2 * a + 1 = 0 ∧ x = -2 ∧ y = 3 :=
by
  intro a
  use [-2, 3]
  simp
  sorry

end line_always_passes_through_fixed_point_l771_771180


namespace infinite_series_evaluation_l771_771772
noncomputable def infinite_series_sum : ℝ :=
  ∑' n, (n : ℝ) / (↑(n + 1)! * 3^n)

theorem infinite_series_evaluation : infinite_series_sum = 0.7388 :=
by
  sorry

end infinite_series_evaluation_l771_771772


namespace evaluate_pow_l771_771364

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l771_771364


namespace arrangement_problem_l771_771308

open Nat

def arrangements (n m : ℕ) : ℕ :=
  (n+m-2).choose (m-1) * (n-1).permute (m-1)

theorem arrangement_problem :
  arrangements 5 2 = 960 := 
by
  sorry

end arrangement_problem_l771_771308


namespace triangle_BCD_area_l771_771970

open Real -- opening the real number space for necessary definitions

/-- Given a tetrahedron ABCD with edges mutually perpendicular at A
    and lengths AB = a, AC = b, AD = c.
    Let the areas of triangles ABC, ACD, and ADB be s, t, and u respectively.
    Prove that the area of triangle BCD is √(s^2 + t^2 + u^2). -/
theorem triangle_BCD_area (a b c s t u : ℝ)
  (h1 : s = (1 / 2) * a * b)
  (h2 : t = (1 / 2) * b * c)
  (h3 : u = (1 / 2) * a * c) :
  let area_BCD := Real.sqrt (s^2 + t^2 + u^2) in
  True :=
sorry

end triangle_BCD_area_l771_771970


namespace projection_correct_l771_771892

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-2, 4)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude_squared (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dp := dot_product u v
  let ms := magnitude_squared v
  (dp / ms * v.1, dp / ms * v.2)

theorem projection_correct :
  projection vector_a vector_b = (-3 / 5, 6 / 5) :=
  sorry

end projection_correct_l771_771892


namespace n_not_both_perfect_squares_l771_771594

open Int

theorem n_not_both_perfect_squares (n x y : ℤ) (h1 : n > 0) :
  ¬ ((n + 1 = x^2) ∧ (4 * n + 1 = y^2)) :=
by {
  -- Problem restated in Lean, proof not required
  sorry
}

end n_not_both_perfect_squares_l771_771594


namespace solve_system_l771_771161

theorem solve_system :
  ∀ {x y : ℝ}, x ≠ 0 → y ≠ 0 →
    (x ^ x = y ∧ x ^ y = y ^ x) →
    (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 2 ∧ y = 4) := by
  intro x y hx0 hy0 h
  sorry

end solve_system_l771_771161


namespace part_a_part_b_part_c_part_d_l771_771651

-- Part (a)
theorem part_a (n m : ℕ) (hn : n ≥ 2) (hm : m ≥ 2) (c : ℕ × ℕ) :
  (c.1 < n → c.2 < m → ∃ adj : ℕ × ℕ, (adj.1 < n) ∧ (adj.2 < m) ∧ (abs (adj.1 - c.1) + abs (adj.2 - c.2) = 1)) :=
sorry

-- Part (b)
theorem part_b : ∃ shape : list (ℕ × ℕ),
  (∀ i j, i < 4 → j < 4 → (shape i j) \∈ ∧ (∀ c1 c2 c3 c4, {c1, c2, c3, c4} ⊆ shape → connected shape \ {c1, c2, c3, c4})) ∧
  (connected shape) :=
sorry

-- Part (c)
theorem part_c : 
  ∃ (shapes : array (array (option ℕ)) (array (option ℕ)), 
  (∀ i, shapes(i) ≠ none → ∃ n m, shifted shapes(i) n m) → ∀ c, (shape_shifts.shifts c) (disconnected shapes) :=
sorry

-- Part (d)
theorem part_d (shape : list (ℕ × ℕ)) (hsize: shape.length = 533):
  (∃ c1 c2, cut shape (c1::c2::nil) ) → disconnected shape :=
sorry

end part_a_part_b_part_c_part_d_l771_771651


namespace solve_problem_l771_771871

theorem solve_problem (x y : ℝ) (h1 : 3^x = 2) (h2 : 12^y = 2) : (1/x) - (1/y) = -2 := 
by 
  sorry

end solve_problem_l771_771871


namespace min_value_of_a_plus_b_l771_771457

theorem min_value_of_a_plus_b 
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_eq : 1 / a + 2 / b = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_of_a_plus_b_l771_771457


namespace factorization_l771_771794

theorem factorization (m : ℤ) : m^2 + 3 * m = m * (m + 3) :=
by sorry

end factorization_l771_771794


namespace sandy_correct_sums_l771_771157

/-- 
Sandy gets 3 marks for each correct sum and loses 2 marks for each incorrect sum.
Sandy attempts 50 sums and obtains 100 marks within a 45-minute time constraint.
If Sandy receives a 1-mark penalty for each sum not completed within the time limit,
prove that the number of correct sums Sandy got is 25.
-/
theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 50) (h2 : 3 * c - 2 * i - (50 - c) = 100) : c = 25 :=
by
  sorry

end sandy_correct_sums_l771_771157


namespace correct_choice_of_f_l771_771013

def f1 (x : ℝ) : ℝ := (x - 1)^2 + 3 * (x - 1)
def f2 (x : ℝ) : ℝ := 2 * (x - 1)
def f3 (x : ℝ) : ℝ := 2 * (x - 1)^2
def f4 (x : ℝ) : ℝ := x - 1

theorem correct_choice_of_f (h : (deriv f1 1 = 3) ∧ (deriv f2 1 ≠ 3) ∧ (deriv f3 1 ≠ 3) ∧ (deriv f4 1 ≠ 3)) : 
  ∀ f, (f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4) → (deriv f 1 = 3 → f = f1) :=
by sorry

end correct_choice_of_f_l771_771013


namespace const_sum_dist_to_planes_l771_771558

variables {A B C D O M : Point}

-- Define the tetrahedron ABCD
def tetrahedron (A B C D : Point) : Prop := 
  -- Assume a non-degenerate tetrahedron
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A) ∧ (A ≠ C) ∧ (B ≠ D)

-- Define point O being equidistant from all vertices
def equidistant_from_vertices (O A B C D : Point) : Prop := 
  dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O D

-- Define that O's distances to the faces (planes) are equal
def equidistant_from_faces (O A B C D : Point) : Prop :=
  let plane_BCD := plane B C D,
      plane_ACD := plane A C D,
      plane_ABD := plane A B D,
      plane_ABC := plane A B C in
  dist_point_plane O plane_BCD = dist_point_plane O plane_ACD ∧ 
  dist_point_plane O plane_ACD = dist_point_plane O plane_ABD ∧ 
  dist_point_plane O plane_ABD = dist_point_plane O plane_ABC

-- Proving the sum of distances from M to the plane faces is constant
theorem const_sum_dist_to_planes 
  (A B C D O M : Point) 
  (tet : tetrahedron A B C D) 
  (equi_vertices : equidistant_from_vertices O A B C D) 
  (equi_faces : equidistant_from_faces O A B C D) 
  (M_in_tetrahedron : M ∈ interior (tetrahedron A B C D)) :
  ∃ k : ℝ, ∀ M ∈ interior (tetrahedron A B C D), 
    (dist_point_plane M (plane B C D)) + 
    (dist_point_plane M (plane A C D)) + 
    (dist_point_plane M (plane A B D)) + 
    (dist_point_plane M (plane A B C)) = k :=
sorry

end const_sum_dist_to_planes_l771_771558


namespace group_time_l771_771504

theorem group_time (total_students number_groups time_per_student : ℕ) (h1 : total_students = 18) (h2 : number_groups = 3) (h3 : time_per_student = 4) :
  let students_per_group := total_students / number_groups in
  let time_per_group := students_per_group * time_per_student in
  time_per_group = 24 :=
by
  let students_per_group := total_students / number_groups
  let time_per_group := students_per_group * time_per_student
  have students_per_group_calc : students_per_group = 6, by sorry
  have time_per_group_calc : time_per_group = 24, by sorry
  exact time_per_group_calc

end group_time_l771_771504


namespace distance_between_points_l771_771240

-- Define the coordinates of the points
def P1 := (-3 : ℝ, 5 : ℝ)
def P2 := (4 : ℝ, -9 : ℝ)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem to be proved
theorem distance_between_points :
  distance P1 P2 = real.sqrt 245 :=
by sorry

end distance_between_points_l771_771240


namespace unique_polynomial_exists_l771_771498

noncomputable def omega : ℂ :=
  (-1 + Complex.I * Real.sqrt 3) / 2

theorem unique_polynomial_exists :
  ∃! (Q : ℂ[X]),
    (∃ (a b c : ℝ), Q = X^4 + a * X^3 + b * X^2 + c * X + 12) ∧
    (∀ r : ℂ, Q.eval r = 0 → Q.eval (omega * r) = 0) :=
sorry

end unique_polynomial_exists_l771_771498


namespace abs_sum_l771_771978

theorem abs_sum (y A B C : ℝ) (hy1 : y = 1 + sqrt 3 / (1 + sqrt 3 / (1 + sqrt 3 / (1 + sqrt 3 / (y)))))
    (hy2 : y^2 - y = sqrt 3) 
    (h_fraction_expr: ∃ (A B C : ℤ), 1 / ((y + 1) * (y - 2)) = (A + sqrt B) / C)
    (h_correct_form:  (1 / ((y + 1) * (y - 2)) = -sqrt 3 - 2) ∧ B = 3 ∧ A = -2 ∧ C = -1) :
    |A| + |B| + |C| = 6 := 
begin
  sorry
end

end abs_sum_l771_771978


namespace volume_of_S_l771_771642

def S (x y z : ℝ) : Prop := |x| + |y| ≤ 2 ∧ |x| + |z| ≤ 2 ∧ |y| + |z| ≤ 2

theorem volume_of_S : volume {p : ℝ × ℝ × ℝ | S p.1 p.2 p.3} = 8 / 3 :=
sorry

end volume_of_S_l771_771642


namespace ellipse_standard_eq_with_given_conditions_l771_771017

theorem ellipse_standard_eq_with_given_conditions : 
  let c := sqrt 5
  let a := 5
  let b := sqrt 20
  ∃ e, e = (1 / sqrt 5) ∧
    (  a * e = c  ) ∧ 
    (b * (sqrt 5) = 5) → 
     ( (x^2 / a^2 ) + (y^2 / b^2)  = 1)  :=
by
  sorry

end ellipse_standard_eq_with_given_conditions_l771_771017


namespace find_m_if_a_b_parallel_l771_771492

theorem find_m_if_a_b_parallel :
  ∃ m : ℝ, (∃ a : ℝ × ℝ, a = (-2, 1)) ∧ (∃ b : ℝ × ℝ, b = (1, m)) ∧ (m * -2 = 1) ∧ (m = -1 / 2) :=
by
  sorry

end find_m_if_a_b_parallel_l771_771492


namespace eval_neg_pow_l771_771379

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l771_771379


namespace log_15_20_eq_l771_771833

theorem log_15_20_eq (a b : ℝ) (h1 : 2^a = 3) (h2 : log 3 5 = b) :
    log (15 : ℝ) 20 = (2 + a * b) / (a + a * b) :=
sorry

end log_15_20_eq_l771_771833


namespace number_of_football_players_l771_771532

theorem number_of_football_players
  (cricket_players : ℕ)
  (hockey_players : ℕ)
  (softball_players : ℕ)
  (total_players : ℕ) :
  cricket_players = 22 →
  hockey_players = 15 →
  softball_players = 19 →
  total_players = 77 →
  total_players - (cricket_players + hockey_players + softball_players) = 21 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_football_players_l771_771532


namespace power_simplification_l771_771686

theorem power_simplification :
  (1 / ((-5) ^ 4) ^ 2) * (-5) ^ 9 = -5 :=
by 
  sorry

end power_simplification_l771_771686


namespace find_b_l771_771010

theorem find_b (a b c A B C : ℝ) (h1 : A + C = 2 * B) 
  (h2 : sqrt 3 * a^2 + sqrt 3 * c^2 - 2 * a * c * sin B = 9 * sqrt 3) 
  (h3 : A + B + C = π) : b = 3 :=
sorry

end find_b_l771_771010


namespace prob_line_intersects_circle_l771_771860

noncomputable def intersection_probability_of_line_and_circle : ℝ :=
  let C := λ x y : ℝ, x^2 + y^2 = 1
  let l := λ k x : ℝ, y = k * (x + 2)
  let interval := Set.Icc (-1:ℝ) (1:ℝ)
  (λ P, ∀ (k : ℝ), k ∈ interval → by 
    have h1: k ∈ Set.Ioo (- Real.sqrt(3) / 3) (Real.sqrt(3) / 3) := sorry
    have h2: P = (Real.sqrt(3) / 3) := sorry
    exact h2) sorry

theorem prob_line_intersects_circle :
  intersection_probability_of_line_and_circle = Real.sqrt(3) / 3 :=
sorry

end prob_line_intersects_circle_l771_771860


namespace power_evaluation_l771_771388

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l771_771388


namespace monotonicity_f_parity_f_max_value_f_min_value_f_l771_771032

noncomputable def f (x : ℝ) : ℝ := x / (x^2 - 4)

-- Monotonicity Proof
theorem monotonicity_f : ∀ {x1 x2 : ℝ}, 2 < x1 → 2 < x2 → x1 < x2 → f x1 > f x2 :=
sorry

-- Parity Proof
theorem parity_f : ∀ x : ℝ, f (-x) = -f x :=
sorry

-- Maximum Value Proof
theorem max_value_f : ∀ {x : ℝ}, x = -6 → f x = -3/16 :=
sorry

-- Minimum Value Proof
theorem min_value_f : ∀ {x : ℝ}, x = -3 → f x = -3/5 :=
sorry

end monotonicity_f_parity_f_max_value_f_min_value_f_l771_771032


namespace alice_cannot_win_l771_771710

-- Definitions based on game rules provided
structure GridCell (n : ℕ) :=
(coords : fin n → ℤ)
(nonneg_coords : ∀ i, coords i ≥ 0)

def adjacent (c1 c2 : GridCell n) : Prop :=
∃ i, (∃ j, j ≠ i ∧ c1.coords j = c2.coords j) ∧ (c1.coords i = c2.coords i + 1 ∨ c1.coords i = c2.coords i - 1)

structure GameState (n : ℕ) :=
(white_pieces : set (GridCell n))
(black_piece : GridCell n)

def valid_move (gs : GameState n) (pos : GridCell n) : Prop :=
pos ∈ gs.white_pieces ∧
(∃ pos₁ pos₂ : GridCell n,
 adj1 : adjacent pos pos₁ ∧ pos₁ ∉ gs.white_pieces ∧
 adj2 : adjacent pos pos₂ ∧ pos₂ ∉ gs.white_pieces ∧
 pos₁.coords.snd > pos.coords.snd ∧ pos₂.coords.snd > pos.coords.snd)

-- Lean statement for the problem
theorem alice_cannot_win (n : ℕ) (initial_white : GridCell n) :
  ∀ gs : GameState n,
  gs.white_pieces = {initial_white} →
  gs.black_piece.coords = λ i, 0 →
  (∃ move_sequence : ℕ → fin n → ℤ, 
      ∀ (k : ℕ), ∃ (next_pos : GridCell n),
         adjacent gs.black_piece next_pos ∧
         next_pos ∉ gs.white_pieces) →
  (∃ K : ℕ, within_hypercube (2 * K) gs.white_pieces) :=
sorry

end alice_cannot_win_l771_771710


namespace power_evaluation_l771_771385

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l771_771385


namespace minimum_area_of_square_l771_771055

noncomputable def min_area_square (t : ℝ) : ℝ :=
if t ≤ 1 then
  if t = -1 then
    80
  else if t = -7 then
    80
  else 
    sorry
else 
  if t = 3 then
    80
  else if t = 9 then
    80
  else 
    sorry

theorem minimum_area_of_square :
  (∃ t : ℝ, (y = 2 * x - 17) ∧ (y = x^2) ∧ min_area_square t = 80) :=
begin
  -- Proof proceeds here
  sorry
end

end minimum_area_of_square_l771_771055


namespace intersection_sum_zero_l771_771947

-- Given Points
def A : Point := ⟨0, 8⟩
def B : Point := ⟨0, 0⟩
def C : Point := ⟨10, 0⟩

-- Midpoints
def G : Point := midpoint A B
def H : Point := midpoint A C

-- The Polynomial defintion
def Line := { p : Point | }

-- Intersection
def Inter := intersection AG BH


theorem intersection_sum_zero :
  let G := midpoint A B,
  let H := midpoint A C,
  let I := intersection (line_through A G) (line_through B H)
in I.x + I.y = 0 := sorry

end intersection_sum_zero_l771_771947


namespace find_angle_C_l771_771536

variable {a b : ℝ}
variable {S : ℝ}
variable {C : ℝ}

-- Given conditions
def triangle_ABC_is_acute (a b C : ℝ) : Prop := 
  a = 3 ∧ b = 4 ∧ (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3) ∧ (C < π/2)

-- The proof problem: prove that C = π/3
theorem find_angle_C (h : triangle_ABC_is_acute a b C) : C = π/3 := 
  sorry

end find_angle_C_l771_771536


namespace hours_in_one_year_l771_771649

/-- Given that there are 24 hours in a day and 365 days in a year,
    prove that there are 8760 hours in one year. -/
theorem hours_in_one_year (hours_per_day : ℕ) (days_per_year : ℕ) (hours_value : ℕ := 8760) : hours_per_day = 24 → days_per_year = 365 → hours_per_day * days_per_year = hours_value :=
by
  intros
  sorry

end hours_in_one_year_l771_771649


namespace card_probability_l771_771662

theorem card_probability :
  let hearts := 13
  let clubs := 13
  let total_cards := 52
  let first_card_is_heart := (hearts.to_rat / total_cards.to_rat)
  let second_card_is_club_given_first_is_heart := (clubs.to_rat / (total_cards - 1).to_rat)
  first_card_is_heart * second_card_is_club_given_first_is_heart = (13.to_rat / 204.to_rat) := by
  sorry

end card_probability_l771_771662


namespace find_M_l771_771516

theorem find_M : 
  let S := (981 + 983 + 985 + 987 + 989 + 991 + 993 + 995 + 997 + 999)
  let Target := 5100 - M
  S = Target → M = 4800 :=
by
  sorry

end find_M_l771_771516


namespace jacob_total_bill_l771_771287

def base_cost : ℝ := 25
def included_hours : ℕ := 25
def cost_per_text : ℝ := 0.08
def cost_per_extra_minute : ℝ := 0.13
def jacob_texts : ℕ := 150
def jacob_hours : ℕ := 31

theorem jacob_total_bill : 
  let extra_minutes := (jacob_hours - included_hours) * 60
  let total_cost := base_cost + jacob_texts * cost_per_text + extra_minutes * cost_per_extra_minute
  total_cost = 83.80 := 
by 
  -- Placeholder for proof
  sorry

end jacob_total_bill_l771_771287


namespace bases_form_cyclic_quadrilateral_radius_of_new_quadrilateral_l771_771154

-- Define the set up for the problem
variables {R d : ℝ} (ABCD : Type) [cyclic_quadrilateral ABCD]
variables {P : ABCD} [intersection_of_diagonals_perpendicular ABCD P]

-- Define the points where the perpendiculars drop
variables (K L M N : ABCD)
variables (base_of_perpendicular K P L) (base_of_perpendicular L P M)
variables (base_of_perpendicular M P N) (base_of_perpendicular N P K)

-- Prove the bases of the perpendiculars form a cyclic quadrilateral
theorem bases_form_cyclic_quadrilateral :
  cyclic_quadrilateral (K, L, M, N) := sorry

-- Given the radius of the original circle and the distance to the intersection
parameters (R d : ℝ)

-- Radius of the inscribed circle in the new cyclic quadrilateral KLMN
theorem radius_of_new_quadrilateral (h : R > d) :
  radius_of_circumcircle (K, L, M, N) = (R * R - d * d) / (2 * R) := sorry

end bases_form_cyclic_quadrilateral_radius_of_new_quadrilateral_l771_771154


namespace knights_probability_sum_l771_771214

/--
There are 25 knights seated around a circular table. We choose 3 knights at random.
Let P be the probability that at least two of the three knights are sitting next to each other.
Prove that if P is written as a fraction in lowest terms, the sum of the numerator and denominator is 57.
-/
theorem knights_probability_sum :
  let total_knights := 25
  let chosen_knights := 3
  let P := 1 - (22 * 21 * 20) / (22 * 23 * 24)
  (P.denom + P.num) = 57 := 
sorry

end knights_probability_sum_l771_771214


namespace find_g_zero_abs_l771_771125

noncomputable def g : ℝ → ℝ := sorry

axiom h_g_third_degree_polynomial : ∀ x, |g 1| = 18 ∧ |g 2| = 18 ∧ |g 3| = 18 ∧ |g 4| = 18 ∧ |g 5| = 18 ∧ |g 6| = 18

theorem find_g_zero_abs : |g 0| = 162 :=
by
  sorry

end find_g_zero_abs_l771_771125


namespace distance_between_points_l771_771241

-- Define the coordinates of the points
def P1 := (-3 : ℝ, 5 : ℝ)
def P2 := (4 : ℝ, -9 : ℝ)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem to be proved
theorem distance_between_points :
  distance P1 P2 = real.sqrt 245 :=
by sorry

end distance_between_points_l771_771241


namespace exists_m_F_gt_G_l771_771684

def is_loyal (B : Finset ℕ) : Prop :=
  ∃ (i j : ℕ), i ≤ j ∧ B = Finset.range (j - i + 1) + i

def loyal_sets (n : ℕ) : Finset (Finset ℕ) :=
  Finset.univ.filter is_loyal

def A_f (A : Finset ℕ) : ℕ :=
  if h : A.card ≤ 1 then 0 else
    (A.to_list.sort (· ≤ ·)).tails.pmap (λ a ha, list.head' ha - list.head' A.to_list)
      (λ _ h, list.tail_ne_nil_of_ne_nil h) |>.maximum'.get_or_else 0

def A_g (A : Finset ℕ) : ℕ :=
  loyal_sets A.to_list.length |>.pmap (λ B hB, B.card) sorry |>.maximum'.get_or_else 0

def F (n : ℕ) : ℕ :=
  (Finset.powerset (Finset.range n)).sum (λ A, A_f A)

def G (n : ℕ) : ℕ :=
  (Finset.powerset (Finset.range n)).sum (λ A, A_g A)

theorem exists_m_F_gt_G : ∃ m : ℕ, ∀ n > m, F n > G n := 
  sorry

end exists_m_F_gt_G_l771_771684


namespace final_color_after_2019_applications_l771_771272

def f (n : ℕ) : ℕ :=
  if n ≤ 19 then n + 4 else |129 - 2 * n|

def apply_f_n_times (n : ℕ) (initial : ℕ) : ℕ :=
  Nat.iterate f n initial

theorem final_color_after_2019_applications :
  apply_f_n_times 2019 5 = 75 :=
sorry

end final_color_after_2019_applications_l771_771272


namespace largest_prime_divisor_to_test_in_range_1000_1100_l771_771064

theorem largest_prime_divisor_to_test_in_range_1000_1100 : 
  ∀ n : ℕ, (1000 ≤ n ∧ n ≤ 1100) → (∀ p : ℕ, p.prime → p ≤ 31 → ¬ p ∣ n) → (31 = Nat.max (List.filter Prime (List.range (Nat.floor (Real.sqrt 1100))))) :=
by
  intros n h_bounds h_prime_test
  sorry

end largest_prime_divisor_to_test_in_range_1000_1100_l771_771064


namespace pair_points_no_intersection_l771_771853

noncomputable theory
open_locale classical

universe u

def no_intersecting_pairs (T : set (ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 p3 p4 : ℝ × ℝ), p1 ∈ T → p2 ∈ T → p3 ∈ T → p4 ∈ T → 
  p1 ≠ p2 → p3 ≠ p4 →
  ¬((p1, p2), (p3, p4) : set (ℝ × ℝ) × set (ℝ × ℝ)).pairwise_disjoint segments

theorem pair_points_no_intersection (T : set (ℝ × ℝ)) (hT : ∃ (n : ℕ), T.card = 2 * n) :
  ∃ P : set (ℝ × ℝ) × set (ℝ × ℝ), no_intersecting_pairs T :=
sorry

end pair_points_no_intersection_l771_771853


namespace inequality_proof_l771_771008

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 1 / b^3 - 1) * (b^3 + 1 / c^3 - 1) * (c^3 + 1 / a^3 - 1) ≤ (a * b * c + 1 / (a * b * c) - 1)^3 :=
by
  sorry

end inequality_proof_l771_771008


namespace buffons_needle_l771_771617

open Real
open Interval
open MeasureTheory
open ProbabilityTheory

noncomputable def probability_intersection (a : ℝ) (h_a_pos : 0 < a) : ℝ :=
 measure_theory.measure_space volume (set_of (λ (p : ℝ × ℝ), p.snd ≤ a * sin p.fst)) (0, a) * (0, π)

theorem buffons_needle : ∀ (a : ℝ) (h_a_pos : 0 < a), probability_intersection a h_a_pos = 2 / π := 
by
  sorry

end buffons_needle_l771_771617


namespace length_of_FG_l771_771655

theorem length_of_FG (A B C F G : Point)
  (hAB : distance A B = 13)
  (hAC : distance A C = 14)
  (hBC : distance B C = 15)
  (hF : F ∈ line AB)
  (hG : G ∈ line AC)
  (h_parallel : parallel (line FG) (line BC))
  (h_centroid : contains_centroid (triangle A B C) (line FG)) :
  ∃ p q : ℕ, gcd p q = 1 ∧ FG.length = p / q ∧ p + q = 17 := sorry

end length_of_FG_l771_771655


namespace triangle_inequality_violation_l771_771222

theorem triangle_inequality_violation (a b c : ℝ) (ha : a = 1) (hb : b = 2) (hc : c = 7) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  rw [ha, hb, hc]
  simp
  sorry

end triangle_inequality_violation_l771_771222


namespace count_valid_four_digit_numbers_l771_771220

theorem count_valid_four_digit_numbers :
  let digits := {1, 2, 3, 4}
  in (finset.filter (λ (abcd : fin 4 → ℕ), 
      abcd 0 ≠ abcd 1 ∧ abcd 0 ≠ abcd 2 ∧ abcd 0 ≠ abcd 3 ∧
      abcd 1 ≠ abcd 2 ∧ abcd 1 ≠ abcd 3 ∧ abcd 2 ≠ abcd 3 ∧
      abcd 1 > abcd 0 ∧ abcd 1 > abcd 3 ∧
      abcd 2 > abcd 0 ∧ abcd 2 > abcd 3) 
      (finset.pi set.univ (λ _, digits))).card = 8 :=
by
  sorry

end count_valid_four_digit_numbers_l771_771220


namespace evaluate_neg_64_exp_4_over_3_l771_771371

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l771_771371


namespace cos_double_angle_l771_771840

open Real

theorem cos_double_angle (α : Real) (h : tan α = 3) : cos (2 * α) = -4/5 :=
  sorry

end cos_double_angle_l771_771840


namespace prob_both_A_B_prob_exactly_one_l771_771678

def prob_A : ℝ := 0.8
def prob_not_B : ℝ := 0.1
def prob_B : ℝ := 1 - prob_not_B

lemma prob_independent (a b : Prop) : Prop := -- Placeholder for actual independence definition
sorry

-- Given conditions
variables (P_A : ℝ := prob_A) (P_not_B : ℝ := prob_not_B) (P_B : ℝ := prob_B) (indep : ∀ A B, prob_independent A B)

-- Questions translated to Lean statements
theorem prob_both_A_B : P_A * P_B = 0.72 := sorry

theorem prob_exactly_one : (P_A * P_not_B) + ((1 - P_A) * P_B) = 0.26 := sorry

end prob_both_A_B_prob_exactly_one_l771_771678


namespace sample_variance_l771_771646

theorem sample_variance (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) :
  (1 / 5) * ((a - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
by
  sorry

end sample_variance_l771_771646


namespace cost_price_of_A_l771_771268

-- Assume the cost price of the bicycle for A which we need to prove
def CP_A : ℝ := 144

-- Given conditions
def profit_A_to_B (CP_A : ℝ) := 1.25 * CP_A
def profit_B_to_C (CP_B : ℝ) := 1.25 * CP_B
def SP_C := 225

-- Proof statement
theorem cost_price_of_A : 
  profit_B_to_C (profit_A_to_B CP_A) = SP_C :=
by
  sorry

end cost_price_of_A_l771_771268


namespace product_of_sum_divisors_even_l771_771427

def sum_divisors (n : ℕ) : ℕ :=
  ∑ i in (Finset.range (n + 1)).filter (n % · = 0), i

theorem product_of_sum_divisors_even (n : ℕ) (h : n > 1) : Even (sum_divisors (n - 1) * sum_divisors n * sum_divisors (n + 1)) :=
sorry

end product_of_sum_divisors_even_l771_771427


namespace set_A_roster_l771_771038

def is_nat_not_greater_than_4 (x : ℕ) : Prop := x ≤ 4

def A : Set ℕ := {x | is_nat_not_greater_than_4 x}

theorem set_A_roster : A = {0, 1, 2, 3, 4} := by
  sorry

end set_A_roster_l771_771038


namespace value_of_f_log2_12_l771_771136

noncomputable def f : ℝ → ℝ :=
sorry

theorem value_of_f_log2_12 (f_periodic : ∀ x : ℝ, f (x + 1) = f x)
                          (f_definition : ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = 2^x) :
  f (real.log 12 / real.log 2) = 3 / 2 :=
by sorry

end value_of_f_log2_12_l771_771136


namespace abc_sum_leq_three_l771_771009

open Real

theorem abc_sum_leq_three {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 + a * b * c = 4) :
  a + b + c ≤ 3 :=
sorry

end abc_sum_leq_three_l771_771009


namespace find_intersection_l771_771577

def A : set ℝ := { x | 1 < x ∧ x < 4 }
def B : set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }
def CR_B : set ℝ := { x | x ∈ (-∞, -1) ∪ (3, ∞) }

theorem find_intersection : A ∩ CR_B = { x | 3 < x ∧ x < 4 } := by
  sorry

end find_intersection_l771_771577


namespace parametric_eq_C1_l771_771289

noncomputable def C := { p : ℝ × ℝ // p.1^2 + p.2^2 = 1 }
noncomputable def transformation : ℝ × ℝ → ℝ × ℝ := λ p, (2 * p.1, sqrt 2 * p.2)
noncomputable def C1 := { p : ℝ × ℝ // (p.1 / 2)^2 + (p.2 / sqrt 2)^2 = 1 }
noncomputable def polar_l : ℝ × ℝ → Prop := λ ρθ, ρθ.1 * Math.cos (ρθ.2 + π / 3) = 1 / 2
noncomputable def M : ℝ × ℝ := (1, 0)
noncomputable def l (x y : ℝ) : Prop := x - sqrt 3 * y - 1 = 0

theorem parametric_eq_C1 :
    (∀ α : ℝ, α ∈ ℝ → (2 * Math.cos α, sqrt 2 * Math.sin α) ∈ C1) ∧
    (∀ x y : ℝ, l x y ↔ x - sqrt 3 * y - 1 = 0) ∧
    (∀ A B : ℝ × ℝ, A ∈ C1 → B ∈ C1 → point_on_l A → point_on_l B → 
      let MA := (dist M A), MB := (dist M B), AB := (dist A B) in
      MA * MB = 12 / 5 ∧ AB = 12 * sqrt 2 / 5) 
    := by sorry

end parametric_eq_C1_l771_771289


namespace leaves_remaining_l771_771156

theorem leaves_remaining : 
  (initial_leaves lost_leaves broken_leaves : ℕ) 
  (h_initial : initial_leaves = 89)
  (h_lost : lost_leaves = 24)
  (h_broken : broken_leaves = 43) : 
  initial_leaves - lost_leaves - broken_leaves = 22 := 
by
  -- Initial number of leaves is 89
  have h1 : initial_leaves = 89 := h_initial
  -- Leaves lost
  have h2 : lost_leaves = 24 := h_lost
  -- More leaves broken
  have h3 : broken_leaves = 43 := h_broken
  -- Total leaves remaining
  calc
    initial_leaves - lost_leaves - broken_leaves
    = 89 - 24 - 43 : by rw [h1, h2, h3]
    = 22          : by norm_num

end leaves_remaining_l771_771156


namespace maximum_value_of_f_l771_771181

noncomputable def f (x : ℝ) : ℝ := log (3 * x) - 3 * x

theorem maximum_value_of_f :
  ∃ x ∈ Icc 0 e, (∀ y ∈ Icc 0 e, f(x) >= f(y)) ∧ f(x) = -log 3 - 1 :=
by
  -- Note the interval (0, e] is represented as Icc 0 e in Lean inclusive on both ends
  sorry

end maximum_value_of_f_l771_771181


namespace determine_friends_l771_771589

inductive Grade
| first
| second
| third
| fourth

inductive Name
| Petya
| Kolya
| Alyosha
| Misha
| Dima
| Borya
| Vasya

inductive Surname
| Ivanov
| Krylov
| Petrov
| Orlov

structure Friend :=
  (name : Name)
  (surname : Surname)
  (grade : Grade)

def friends : List Friend :=
  [ {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first},
    {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second},
    {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third},
    {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ]

theorem determine_friends : ∃ l : List Friend, 
  {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first} ∈ l ∧
  {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second} ∈ l ∧
  {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third} ∈ l ∧
  {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ∈ l :=
by 
  use friends
  repeat { simp [friends] }


end determine_friends_l771_771589


namespace probability_heart_then_club_l771_771658

theorem probability_heart_then_club :
  let P_heart := 13 / 52
  let P_club_given_heart := 13 / 51
  P_heart * P_club_given_heart = 13 / 204 := 
by
  let P_heart := (13 : ℚ) / 52
  let P_club_given_heart := (13 : ℚ) / 51
  have h : P_heart * P_club_given_heart = 13 / 204 := by
    calc
      P_heart * P_club_given_heart
        = (13 / 52) * (13 / 51) : rfl
    ... = (13 * 13) / (52 * 51) : by rw [mul_div_mul_comm]
    ... = 169 / 2652 : rfl
    ... = 13 / 204 : by norm_num
  exact h

end probability_heart_then_club_l771_771658


namespace find_S3_l771_771472

noncomputable def a (n : ℕ) : ℝ := sorry  -- Define the geometric sequence
noncomputable def q : ℝ := sorry           -- Define the common ratio

-- Conditions
axiom geometric_sequence (n : ℕ) : a (n + 1) = a n * q
axiom common_ratio_ne_one : q ≠ 1
axiom a2_eq_2 : a 2 = 2
axiom arithmetic_sequence : 16 * a 1 + 2 * a 7 = 18 * a 4

-- Sum of the first three terms
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

theorem find_S3 : S 3 = 7 :=
by sorry

end find_S3_l771_771472


namespace lattice_points_on_hyperbola_l771_771913

-- The statement to be proven
theorem lattice_points_on_hyperbola : 
  {p : ℤ × ℤ | p.1 ^ 2 - p.2 ^ 2 = 65}.finite.toFinset.card = 4 :=
by
  sorry

end lattice_points_on_hyperbola_l771_771913


namespace distance_between_points_l771_771229

theorem distance_between_points :
  let x1 := -3
  let y1 := 5
  let x2 := 4
  let y2 := -9
  sqrt ((x2 - x1)^2 + (y2 - y1)^2) = sqrt 245 :=
by
  sorry

end distance_between_points_l771_771229


namespace polynomial_no_two_positive_roots_l771_771475

theorem polynomial_no_two_positive_roots (n : ℕ) (a : Fin (n+1) → ℝ) (h : ∀ i, 0 ≤ a i) :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → 
  ((∑ i in Fin.range (n+1), a i * x₁^(n - i)) = 0) →
  ((∑ i in Fin.range (n+1), a i * x₂^(n - i)) = 0) → x₁ = x₂ :=
by
  sorry

end polynomial_no_two_positive_roots_l771_771475


namespace raccoon_hid_nuts_l771_771942

theorem raccoon_hid_nuts :
  ∃ (r p : ℕ), r + p = 25 ∧ (p = r - 3) ∧ 5 * r = 6 * p ∧ 5 * r = 70 :=
by
  sorry

end raccoon_hid_nuts_l771_771942


namespace distance_between_points_l771_771239

-- Define the coordinates of the points
def P1 := (-3 : ℝ, 5 : ℝ)
def P2 := (4 : ℝ, -9 : ℝ)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem to be proved
theorem distance_between_points :
  distance P1 P2 = real.sqrt 245 :=
by sorry

end distance_between_points_l771_771239


namespace exist_functions_iff_n_even_l771_771430

theorem exist_functions_iff_n_even (n : ℕ) (h : 0 < n) :
  (∃ (f g : Fin n → Fin n), ∀ i : Fin n, (f (g i) = i ∧ g (f i) ≠ i) ∨ (g (f i) = i ∧ f (g i) ≠ i)) ↔ Even n :=
by
  sorry

end exist_functions_iff_n_even_l771_771430


namespace min_correct_answers_l771_771533

-- Problem parameters
def total_questions := 20
def points_per_correct := 10
def points_per_incorrect := -3
def required_points := 70

-- Lean statement
theorem min_correct_answers (x : ℕ) (h1 : x ≤ total_questions) :
  10 * x + (-3) * (total_questions - x) ≥ required_points ↔ x ≥ 10 :=
begin
  sorry
end

end min_correct_answers_l771_771533


namespace circumference_tank_B_l771_771614

-- Defining the conditions
def height_A := 7
def circumference_A := 8
def height_B := 8
def volume_proportion := 0.56

-- Problem statement
theorem circumference_tank_B :
  2 * Real.pi * (cbrt( (7 * (8 / (2 * Real.pi))^2 * volume_proportion) / (8 * (height_B))) / Real.pi ) = 10 :=
by
  sorry

end circumference_tank_B_l771_771614


namespace domain_of_function_l771_771623

noncomputable def function_domain (x : ℝ) : Prop := x ≥ -1 ∧ x ≠ 1

theorem domain_of_function :
  (∀ x : ℝ, function_domain x → ∃ y : ℝ, y = (sqrt (x + 1)) / (x - 1)) :=
begin
  sorry
end

end domain_of_function_l771_771623


namespace charlie_additional_metal_and_best_price_l771_771263

-- Define constants for known values
def total_metal_needed : ℝ := 635
def metal_in_storage : ℝ := 276
def aluminum_percentage : ℝ := 0.60
def steel_percentage : ℝ := 0.40

-- Define supplier prices
def price_aluminum_A : ℝ := 1.30
def price_steel_A : ℝ := 0.90
def price_aluminum_B : ℝ := 1.10
def price_steel_B : ℝ := 1.00
def price_aluminum_C : ℝ := 1.25
def price_steel_C : ℝ := 0.95

-- Define the problem statement
theorem charlie_additional_metal_and_best_price :
  let additional_metal := total_metal_needed - metal_in_storage,
      aluminum_needed := aluminum_percentage * additional_metal,
      steel_needed := steel_percentage * additional_metal,
      cost_A := aluminum_needed * price_aluminum_A + steel_needed * price_steel_A,
      cost_B := aluminum_needed * price_aluminum_B + steel_needed * price_steel_B,
      cost_C := aluminum_needed * price_aluminum_C + steel_needed * price_steel_C
  in additional_metal = 359 ∧ cost_B = 380.54 ∧ cost_B < cost_A ∧ cost_B < cost_C :=
by
  -- Proof is omitted as per the instruction
  sorry

end charlie_additional_metal_and_best_price_l771_771263


namespace smallest_enclosing_sphere_radius_l771_771356

theorem smallest_enclosing_sphere_radius :
  let r := 2
  let d := 4 * Real.sqrt 3
  let total_diameter := d + 2*r
  let radius_enclosing_sphere := total_diameter / 2
  radius_enclosing_sphere = 2 + 2 * Real.sqrt 3 := by
  -- Define the radius of the smaller spheres
  let r : ℝ := 2
  -- Space diagonal of the cube which is 4√3 where 4 is the side length
  let d : ℝ := 4 * Real.sqrt 3
  -- Total diameter of the sphere containing the cube (space diagonal + 2 radius of one sphere)
  let total_diameter : ℝ := d + 2 * r
  -- Radius of the enclosing sphere
  let radius_enclosing_sphere : ℝ := total_diameter / 2
  -- We need to prove that this radius equals 2 + 2√3
  sorry

end smallest_enclosing_sphere_radius_l771_771356


namespace arithmetic_sequence_proof_l771_771859

open Nat

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 2 ∧ (a 2) ^ 2 = (a 1) * (a 5)

def general_formula (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (d = 0 ∧ ∀ n, a n = 2) ∨ (d = 4 ∧ ∀ n, a n = 4 * n - 2)

def sum_seq (a : ℕ → ℤ) (S_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ((∀ n, a n = 2) ∧ (∀ n, S_n n = 2 * n)) ∨ ((∀ n, a n = 4 * n - 2) ∧ (∀ n, S_n n = 4 * n^2 - 2 * n))

theorem arithmetic_sequence_proof :
  ∃ a : ℕ → ℤ, ∃ d : ℤ, arithmetic_seq a d ∧ general_formula a d ∧ ∃ S_n : ℕ → ℤ, sum_seq a S_n d := by
  sorry

end arithmetic_sequence_proof_l771_771859


namespace decreasing_interval_implies_range_of_a_l771_771057

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem decreasing_interval_implies_range_of_a (a : ℝ)
  (h : ∀ x y : ℝ, x ≤ y → y ≤ 4 → f a x ≥ f a y) : a ≤ -3 :=
by
  sorry

end decreasing_interval_implies_range_of_a_l771_771057


namespace lcm_18_45_l771_771693

theorem lcm_18_45 : Int.lcm 18 45 = 90 :=
by
  -- Prime factorizations
  have h1 : Nat.factors 18 = [2, 3, 3] := by sorry
  have h2 : Nat.factors 45 = [3, 3, 5] := by sorry
  
  -- Calculate LCM
  rw [←Int.lcm_def, Nat.factors_mul, List.perm.ext']
  apply List.Permutation.sublist
  sorry

end lcm_18_45_l771_771693


namespace odd_terms_in_expansion_of_a_plus_b_to_8_l771_771508

theorem odd_terms_in_expansion_of_a_plus_b_to_8 (a b : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) :
  let terms := list.map (fun k => binom 8 k * a^k * b^(8 - k)) (list.range 9) in
  list.countp (fun x => x % 2 = 1) terms = 2 :=
sorry

end odd_terms_in_expansion_of_a_plus_b_to_8_l771_771508


namespace rationalize_denominator_l771_771155

theorem rationalize_denominator : (1 / (Real.sqrt 3 + 1)) = ((Real.sqrt 3 - 1) / 2) :=
by
  sorry

end rationalize_denominator_l771_771155


namespace carolyn_sum_of_removals_l771_771337

-- Represent the initial conditions of the problem
def initial_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def has_divisors (n : ℕ) (lst : List ℕ) : Prop :=
  ∃ d ∈ lst, 1 < d ∧ d < n ∧ n % d = 0

-- Define the game's conditions
def carolyn_remove (lst : List ℕ) (n : ℕ) : Prop :=
  n ∈ lst ∧ has_divisors n lst

def paul_remove (lst : List ℕ) (n : ℕ) : List ℕ :=
  lst.filter (λ m, ¬ m ∣ n)

-- Create the equivalent proof statement
theorem carolyn_sum_of_removals : 
  ∀ (lst : List ℕ), 
    (initial_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) → 
    (carolyn_remove lst 4) → 
    ∑ i in [4, 6, 8], i = 18 := by
  sorry

end carolyn_sum_of_removals_l771_771337


namespace scientific_notation_l771_771934

theorem scientific_notation :
  686530000 = 6.8653 * 10^8 :=
sorry

end scientific_notation_l771_771934


namespace savings_on_each_apple_l771_771626

-- Define the costs in cents for clarity
constant cost_first_store : ℕ
constant apples_first_store : ℕ
constant cost_second_store : ℕ
constant apples_second_store : ℕ

noncomputable def cost_per_apple_first_store : ℚ :=
(cost_first_store : ℚ) / (apples_first_store : ℚ)

noncomputable def cost_per_apple_second_store : ℚ :=
(cost_second_store : ℚ) / (apples_second_store : ℚ)

-- Save in cents calculation
noncomputable def savings_per_apple : ℚ :=
cost_per_apple_first_store - cost_per_apple_second_store

noncomputable def savings_per_apple_cents : ℚ :=
savings_per_apple * 100

-- Given conditions
def conditions : Prop :=
cost_first_store = 300 ∧
apples_first_store = 6 ∧
cost_second_store = 400 ∧
apples_second_store = 10

-- Main theorem statement
theorem savings_on_each_apple : conditions -> savings_per_apple_cents = 10 :=
begin
  sorry
end

end savings_on_each_apple_l771_771626


namespace probability_useful_parts_l771_771313

noncomputable def probability_three_parts_useful (pipe_length : ℝ) (min_length : ℝ) : ℝ :=
  let total_area := (pipe_length * pipe_length) / 2
  let feasible_area := ((pipe_length - min_length) * (pipe_length - min_length)) / 2
  feasible_area / total_area

theorem probability_useful_parts :
  probability_three_parts_useful 300 75 = 1 / 16 :=
by
  sorry

end probability_useful_parts_l771_771313


namespace polygons_sides_l771_771645

theorem polygons_sides 
  (n1 n2 : ℕ)
  (h1 : n1 * (n1 - 3) / 2 + n2 * (n2 - 3) / 2 = 158)
  (h2 : 180 * (n1 + n2 - 4) = 4320) :
  (n1 = 16 ∧ n2 = 12) ∨ (n1 = 12 ∧ n2 = 16) :=
sorry

end polygons_sides_l771_771645


namespace exp_log_properties_l771_771791

variable (a b m M N : ℝ)
variable (h_a_pos : a > 0) (h_a_ne_zero : a ≠ 0)
variable (h_b_pos : b > 0) (h_b_ne_one : b ≠ 1)
variable (h_m_ne_zero : m ≠ 0) (h_M_pos : M > 0) (h_N_pos : N > 0)

-- The domain and range of the exponential function
def exp_domain_range : Prop :=
  ∀ (x : ℝ), (0 < Real.exp x) ∧ (∃ y : ℝ, Real.exp y = x)

-- The domain and range of the logarithmic function
def log_domain_range : Prop :=
  ∀ (x : ℝ), (0 < x → ∃ y : ℝ, Real.log x = y)

-- Logarithmic properties given conditions
def logarithmic_properties : Prop :=
  ∀ (M N : ℝ), M > 0 → N > 0 →
    Real.log (a) (M * N) = Real.log (a) M + Real.log (a) N ∧
    a ^ (Real.log (a) N) = N ∧
    Real.log (a^m) (b^n) = (n / m) * Real.log (a) b

theorem exp_log_properties :
  exp_domain_range a ∧
  log_domain_range a ∧
  logarithmic_properties a b m M N :=
by
  sorry

end exp_log_properties_l771_771791


namespace problem_statement_l771_771018

variable {f : ℝ → ℝ}

-- Conditions: f is differentiable and f'(x) - f(x) > 0 for x > 0
variable (h_diff : ∀ x > 0, differentiable_at ℝ f x)
variable (h_ineq : ∀ x > 0, deriv f x - f x > 0)

theorem problem_statement : f 3 > real.exp 1 * f 2 :=
sorry

end problem_statement_l771_771018


namespace tangent_line_equation_l771_771465

-- Define f(x) based on the given conditions
def f (x : ℝ) : ℝ :=
  if x < 0 then
    Real.log (-x) + 3 * x
  else
    Real.log x - 3 * x

-- Define the even property of the function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the problem in Lean 4
theorem tangent_line_equation :
  even_function f →
  f 1 = -3 →
  deriv f 1 = -2 →
  ∃ m b : ℝ, (m = -2 ∧ b = -1 ∧ (∀ x y : ℝ, y - (-3) = m * (x - 1) ↔ y = m * x + b)) :=
by
  intros h_even h_f1 h_deriv
  use -2
  use -1
  split
  sorry
  split
  sorry
  intros x y
  split
  sorry
  sorry

end tangent_line_equation_l771_771465


namespace f_is_odd_f_is_decreasing_l771_771987

variable (f : ℝ → ℝ)

-- Conditions
axiom cond1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
axiom cond2 : ∀ x : ℝ, x > 0 → f(x) < 0

-- Proof problems
theorem f_is_odd : ∀ x : ℝ, f(-x) = -f(x) :=
by 
  -- Proof to be provided here
  sorry

theorem f_is_decreasing : ∀ x_1 x_2 : ℝ, x_1 < x_2 → f(x_2) < f(x_1) :=
by 
  -- Proof to be provided here
  sorry

end f_is_odd_f_is_decreasing_l771_771987


namespace ratio_of_sums_equiv_seven_eighths_l771_771052

variable (p q r u v w : ℝ)
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
variable (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
variable (h1 : p^2 + q^2 + r^2 = 49)
variable (h2 : u^2 + v^2 + w^2 = 64)
variable (h3 : p * u + q * v + r * w = 56)

theorem ratio_of_sums_equiv_seven_eighths :
  (p + q + r) / (u + v + w) = 7 / 8 :=
by
  sorry

end ratio_of_sums_equiv_seven_eighths_l771_771052


namespace evaluate_neg_64_exp_4_over_3_l771_771370

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l771_771370


namespace two_pow_n_not_div_n_fact_two_pow_n_minus_one_div_n_fact_infinite_l771_771601

theorem two_pow_n_not_div_n_fact (n : ℕ) : ¬ (2^n ∣ n!) :=
begin
  assume h,
  sorry -- This should prove that 2^n does not divide n!
end

theorem two_pow_n_minus_one_div_n_fact_infinite : 
  ∃ᵢ (n : ℕ), 2^(n-1) ∣ n! :=
begin
  use 2^m, -- Considering the form of n=2^m for any natural number m
  sorry -- This should prove that 2^(n-1) divides n! for infinitely many n
end

end two_pow_n_not_div_n_fact_two_pow_n_minus_one_div_n_fact_infinite_l771_771601


namespace sum_sequence_l771_771414

theorem sum_sequence : 
  ∑ k in range 9, (k + 4) * (1 - (1 / (k + 2))) = 63 := by
  sorry

end sum_sequence_l771_771414


namespace find_angle_C_find_max_area_l771_771528

-- Define a triangle with sides opposite to angles A, B, and C.
variable (A B C a b c : ℝ)

theorem find_angle_C (h1 : 4 * sin (A + B) / 2 ^ 2 - cos (2 * C) = 7 / 2)
                     (h2 : c = sqrt 7)
                     (h3 : True) :  -- Placeholder for the triangle ABC condition
  C = π / 3 :=
sorry

theorem find_max_area (h1 : 4 * sin (A + B) / 2 ^ 2 - cos (2 * C) = 7 / 2)
                      (h2 : c = sqrt 7)
                      (h3 : True) :  -- Placeholder for the triangle ABC condition
  let area := (1 / 2) * a * b * sin C in
  area ≤ 7 * sqrt 3 / 4 :=
sorry


end find_angle_C_find_max_area_l771_771528


namespace calc_expression_l771_771129

def r (θ : ℚ) : ℚ := 1 / (1 + θ)
def s (θ : ℚ) : ℚ := θ + 1

theorem calc_expression : s (r (s (r (s (r 2))))) = 24 / 17 :=
by 
  sorry

end calc_expression_l771_771129


namespace find_k_l771_771712

noncomputable def k_values : Set ℝ := { k | let a := (2, 3)
                                           let b := (3, -2)
                                           let u := (λ k, (2 * k + 3, 3 * k - 2))
                                           let v := (λ k, (2 + 3 * k, 3 - 2 * k))
                                           let dot_product := (u k).fst * (v k).fst + (u k).snd * (v k).snd
                                           let magnitude_u := Real.sqrt ((u k).fst ^ 2 + (u k).snd ^ 2)
                                           let magnitude_v := Real.sqrt ((v k).fst ^ 2 + (v k).snd ^ 2)
                                           dot_product = 1 / 2 * (magnitude_u * magnitude_v)
                                         } 

theorem find_k : k_values = {2 + Real.sqrt 3, 2 - Real.sqrt 3} := 
  sorry

end find_k_l771_771712


namespace arithmetic_mean_is_correct_l771_771535

variable (n : ℕ) (h : n > 2)

def num1 : ℝ := 1 - (1 / n)
def num2 : ℝ := 1 - (1 / n)

def num_list : List ℝ :=
  num1 :: num2 :: (List.replicate (n - 2) 1)

def arithmetic_mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem arithmetic_mean_is_correct :
  arithmetic_mean n num_list = 1 - (2 / n^2) :=
by
  sorry

end arithmetic_mean_is_correct_l771_771535


namespace max_table_sum_l771_771776

/- Conditions -/
def chosen_numbers : List ℕ := [3, 5, 7, 11, 17, 19]
def sum_is_62 (l : List ℕ) : Prop := l.sum = 62
def table_sum (a b c d e f : ℕ) : Prop := (a + b + c) * (d + e + f) = 961

/- Statement -/
theorem max_table_sum :
  ∃ (a b c d e f : ℕ), 
  {a, b, c, d, e, f} ⊆ chosen_numbers ∧
  sum_is_62 [a, b, c, d, e, f] ∧ 
  table_sum a b c d e f :=
by 
  sorry

end max_table_sum_l771_771776


namespace probability_heart_then_club_l771_771668

noncomputable def numHearts : ℕ := 13
noncomputable def numClubs : ℕ := 13
noncomputable def totalCards (n : ℕ) : ℕ := 52 - n

noncomputable def probabilityFirstHeart : ℚ := numHearts / totalCards 0
noncomputable def probabilitySecondClubGivenFirstHeart : ℚ := numClubs / totalCards 1

theorem probability_heart_then_club :
  (probabilityFirstHeart * probabilitySecondClubGivenFirstHeart) = 13 / 204 :=
by
  sorry

end probability_heart_then_club_l771_771668


namespace tangent_dihedral_angle_l771_771273

noncomputable def sqrt2 : ℝ := real.sqrt 2

theorem tangent_dihedral_angle (A B C A1 B1 C1 E : Type)
  (h1 : AB ⊥ plane (BB1, C1, C))
  (h2 : E ∈ segment C C1)
  (h3 : EA ⊥ EB1)
  (h4 : ∥A - B∥ = sqrt2)
  (h5 : ∥B - B1∥ = 2)
  (h6 : ∥B - C∥ = 1)
  (h7 : angle B C C1 = π / 3)
  : tan (dihedral_angle (plane (A, EB1)) (plane A1)) = sqrt2 / 2 := 
  sorry

end tangent_dihedral_angle_l771_771273


namespace determine_mu_l771_771990

open ProbabilityTheory

noncomputable def random_variable : Type := ℝ
def normal_distribution (μ σ : ℝ) : Measure random_variable := gaussian μ σ

theorem determine_mu (μ σ : ℝ) (ξ : random_variable)
  (hξ : ξ ∼ normal_distribution μ σ)
  (hprob : P(λ x, x > 4, ξ) = 0.5) :
  μ = 4 := sorry

end determine_mu_l771_771990


namespace proof_problem_l771_771640

-- Definitions of propositions p and q
def p (a b : ℝ) : Prop := a < b → ∀ c : ℝ, c ≠ 0 → a * c^2 < b * c^2
def q : Prop := ∃ x₀ > 0, x₀ - 1 + Real.log x₀ = 0

-- Conditions for the problem
variable (a b : ℝ)
variable (p_false : ¬ p a b)
variable (q_true : q)

-- Proving which compound proposition is true
theorem proof_problem : (¬ p a b) ∧ q := by
  exact ⟨p_false, q_true⟩

end proof_problem_l771_771640


namespace cubic_root_identity_l771_771565

theorem cubic_root_identity (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a * b + a * c + b * c = -3)
  (h3 : a * b * c = -2) : 
  a * (b + c) ^ 2 + b * (c + a) ^ 2 + c * (a + b) ^ 2 = -6 := 
by
  sorry

end cubic_root_identity_l771_771565


namespace factorize_expression_l771_771798

theorem factorize_expression (m : ℝ) : m^2 + 3 * m = m * (m + 3) :=
by
  sorry

end factorize_expression_l771_771798


namespace maximize_L_l771_771883

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + 8 * x + 3

theorem maximize_L :
  ∃ (L : ℝ) (a < 0),
    (∀ x, 0 ≤ x ∧ x ≤ L → |f a x| ≤ 5) ∧
    (∀ b < 0, (∀ y, 0 ≤ y ∧ y ≤ L → |f b y| ≤ 5) → L ≤ (√5 + 1) / 2) ∧ 
    (∃ a, a = -8 ∧ L = (√5 + 1) / 2) := sorry

end maximize_L_l771_771883


namespace eight_packets_weight_l771_771311

variable (weight_per_can : ℝ)
variable (weight_per_packet : ℝ)

-- Conditions
axiom h1 : weight_per_can = 1
axiom h2 : 3 * weight_per_can = 8 * weight_per_packet
axiom h3 : weight_per_packet = 6 * weight_per_can

-- Question to be proved: 8 packets weigh 12 kg
theorem eight_packets_weight : 8 * weight_per_packet = 12 :=
by 
  -- Proof would go here
  sorry

end eight_packets_weight_l771_771311


namespace local_extrema_f_l771_771003

noncomputable def e := Real.exp 1

def f (k : ℕ) (x : ℝ) : ℝ := (Real.exp x - 1) * (x - 1)^k

theorem local_extrema_f :
  (∀ k : ℕ, k = 1 → ¬ (IsLocalMin f k 1 ∧ ¬IsLocalMax f k 1)) ∧
  (∀ k : ℕ, k = 2 → IsLocalMin (f k) 1) :=
by {
  sorry
}

end local_extrema_f_l771_771003


namespace miles_per_dollar_l771_771346

def car_mpg : ℝ := 32
def gas_cost_per_gallon : ℝ := 4

theorem miles_per_dollar (X : ℝ) : 
  (X / gas_cost_per_gallon) * car_mpg = 8 * X :=
by
  sorry

end miles_per_dollar_l771_771346


namespace at_least_one_ge_2017_l771_771274

-- Defining the sequences and their properties
def sequences (a : ℕ → ℤ) (b : ℕ → ℕ) :=
  (a 0 = 0) ∧ (a 1 = 1) ∧ ∀ n ≥ 1, 
  a (n+1) = if (b (n-1) = 1) then (a n * b n + a (n-1)) else (a n * b n - a (n-1))

-- Statement of the theorem to be proven
theorem at_least_one_ge_2017 (a : ℕ → ℤ) (b : ℕ → ℤ) (hb : ∀ n, b n > 0) (hseq : sequences a b) :
  a 2017 ≥ 2017 ∨ a 2018 ≥ 2017 :=
  sorry

end at_least_one_ge_2017_l771_771274


namespace slope_of_tangent_line_at_A_l771_771024

theorem slope_of_tangent_line_at_A :
  ∀ (x : ℝ), (f : ℝ → ℝ) (hf : f x = 2 * x^2), (P : ℝ × ℝ) (P1 : P = (1, 2)) (f' : ℝ → ℝ) (hf' : f' x = 4 * x),
  f' 1 = 4 :=
begin
  sorry
end

end slope_of_tangent_line_at_A_l771_771024


namespace equivalent_curves_and_minimum_distance_l771_771264

noncomputable def curve_C1 (θ : ℝ) : ℝ := (2 * θ - 3.14 / 2)
def standard_eq_C1 (x y : ℝ) : Prop := x - y + 2 = 0

noncomputable def curve_C2 (θ : ℝ) : ℝ × ℝ := (⟨(sqrt 2 / 2) + cos θ, (sqrt 2 / 2) + sin θ⟩)
def param_eq_C2 (x y : ℝ) : Prop := (∃ θ, x = (sqrt 2 / 2 + cos θ) ∧ y = (sqrt 2 / 2 + sin θ))

def distance (M N : ℝ × ℝ) : ℝ := Real.sqrt ((M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2)
def min_distance : ℝ := sqrt 2 - 1

theorem equivalent_curves_and_minimum_distance :
  (∀ (x y ρ θ : ℝ), 
    (curve_C1 θ = curve_C2 θ.2) → standard_eq_C1 x y ∧ param_eq_C2 x y ) ∧
  (∀ (M N : ℝ × ℝ), 
    M ∈ curve_C1 → N ∈ curve_C2 → 
    distance M N = min_distance) :=
sorry

end equivalent_curves_and_minimum_distance_l771_771264


namespace evaluate_neg_sixtyfour_exp_four_thirds_l771_771403

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l771_771403


namespace parabola_equation_conditions_l771_771428

def focus_on_x_axis (focus : ℝ × ℝ) := (∃ x : ℝ, focus = (x, 0))
def foot_of_perpendicular (line : ℝ × ℝ → Prop) (focus : ℝ × ℝ) :=
  (∃ point : ℝ × ℝ, point = (2, 1) ∧ line focus ∧ line point ∧ line (0, 0))

theorem parabola_equation_conditions (focus : ℝ × ℝ) (line : ℝ × ℝ → Prop) :
  focus_on_x_axis focus →
  foot_of_perpendicular line focus →
  ∃ a : ℝ, ∀ x y : ℝ, y^2 = a * x ↔ y^2 = 10 * x :=
by
  intros h1 h2
  use 10
  sorry

end parabola_equation_conditions_l771_771428


namespace expr_of_odd_func_l771_771835

theorem expr_of_odd_func (f : ℝ → ℝ) (f_odd : ∀ x, f(-x) = -f(x))
    (f_cond : ∀ x, x < 0 → f(x) = (cos x + sin (2 * x))) :
  ∀ x, x > 0 → f(x) = -cos x + sin (2 * x) :=
by
  intro x pos
  have : f(-x) = cos (-x) + sin (2 * -x) := f_cond (-x) (neg_neg.mp pos)
  rw [cos_neg, sin_neg, ←neg_mul_eq_neg_mul] at this
  have h : f x = -f (-x) := f_odd x
  rw [this, h]
  ring

end expr_of_odd_func_l771_771835


namespace domain_and_range_of_h_l771_771566

theorem domain_and_range_of_h (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ [-1, 3], f x ∈ [0, 2]) :
  (∀ x ∈ [0, 4], (2 - f (x - 1)) ∈ [0, 2]) ∧ [0 <= 0, 4 <= 4] := 
by
  sorry

end domain_and_range_of_h_l771_771566


namespace last_digit_octal_conversion_l771_771345

theorem last_digit_octal_conversion (n : ℕ) (hn : n = 2016) : 
  (n % 8 = 0) :=
by
  rw hn
  -- The octal conversion steps correspond to modulus operations at each step
  have h1 : 2016 % 8 = 0, by norm_num
  exact h1

end last_digit_octal_conversion_l771_771345


namespace eq_has_exactly_one_real_root_l771_771753

theorem eq_has_exactly_one_real_root : ∀ x : ℝ, 2007 * x^3 + 2006 * x^2 + 2005 * x = 0 ↔ x = 0 :=
by
sorry

end eq_has_exactly_one_real_root_l771_771753


namespace sum_of_exponents_of_square_root_of_largest_perfect_square_in_15_factorial_l771_771253

-- Definitions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def largest_perfect_square_factors (n : ℕ) : List (ℕ × ℕ) :=
  [(2, 10), (3, 6), (5, 2), (7, 2)]  -- precomputed prime factors and their adjusted exponents for 15!

def sum_of_exponents_of_square_root (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc prime_exp_pair, acc + prime_exp_pair.snd / 2) 0

-- Statement to prove
theorem sum_of_exponents_of_square_root_of_largest_perfect_square_in_15_factorial :
  sum_of_exponents_of_square_root (largest_perfect_square_factors 15) = 10 :=
by
  sorry

end sum_of_exponents_of_square_root_of_largest_perfect_square_in_15_factorial_l771_771253


namespace value_of_m_l771_771636

theorem value_of_m
  (m : ℤ)
  (h1 : ∃ p : ℕ → ℝ, p 4 = 1/3 ∧ p 1 = -(m + 4) ∧ p 0 = -11 ∧ (∀ (n : ℕ), (n ≠ 4 ∧ n ≠ 1 ∧ n ≠ 0) → p n = 0) ∧ 1 ≤ p 4 + p 1 + p 0) :
  m = 4 :=
  sorry

end value_of_m_l771_771636


namespace inscribed_quadrilateral_of_tangency_points_l771_771943

theorem inscribed_quadrilateral_of_tangency_points
  {A B C D P Q R P' Q' R' : Point}
  (incircle_ABC : Incircle ABC)
  (incircle_ACD : Incircle ACD)
  (tangent_points_ABC : TangentPoints ABC incircle_ABC P Q R)
  (tangent_points_ACD : TangentPoints ACD incircle_ACD P' Q' R')
  : InscribedQuadrilateral P Q R P' Q' R' :=
  sorry

end inscribed_quadrilateral_of_tangency_points_l771_771943


namespace sqrt_inequality_l771_771462

variable {a b : ℝ}

theorem sqrt_inequality (h₁ : a > b) (h₂ : b > 0) : 
  (sqrt a - sqrt b) < (sqrt (a - b)) :=
sorry

end sqrt_inequality_l771_771462


namespace solution_set_of_inequality_l771_771191

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 + 4 * x - 5 > 0 ↔ (x < -5 ∨ x > 1) :=
sorry

end solution_set_of_inequality_l771_771191


namespace find_angle_between_lateral_edge_and_base_plane_l771_771209

/--  Define the problem context including a regular triangular pyramid and conditions provided. -/
structure RegularTriangularPyramid :=
(base_side : ℝ)
(lateral_edge : ℝ)
(PA_angle_plane : ℝ)

def target_angle (pyramid : RegularTriangularPyramid) : Prop :=
  pyramid.PA_angle_plane = arcsin (sqrt 2 / 3)

theorem find_angle_between_lateral_edge_and_base_plane (pyramid : RegularTriangularPyramid)
  (h1 : pyramid.PA_angle_plane = arcsin (sqrt 2 / 3))
  (h2 : pyramid.base_side > 0)
  (h3 : pyramid.lateral_edge > 0) :
  target_angle pyramid :=
sorry

end find_angle_between_lateral_edge_and_base_plane_l771_771209


namespace logistics_personnel_in_sample_l771_771741

theorem logistics_personnel_in_sample
  (total_staff : ℕ)
  (teachers : ℕ)
  (admin_personnel : ℕ)
  (logistics_personnel : ℕ)
  (sample_size : ℕ)
  (staff_ratio : total_staff = teachers + admin_personnel + logistics_personnel)
  (teachers_ratio : teachers  = 120)
  (admin_ratio : admin_personnel = 16)
  (logistics_ratio : logistics_personnel  = 24)
  (sample_ratio : sample_size  = 20) :
  (∃ x : ℕ, x = 3) :=
by
  have h : (logistics_personnel : ℝ) / total_staff = 24 / 160 := 
    by rw [logistics_ratio, show total_staff = 120 + 16 + 24 from staff_ratio];
  have x_calc : (logistics_personnel : ℝ) / total_staff * sample_size = 3 := 
    by rw [h, show sample_size = 20 from sample_ratio];
  exact ⟨3, by exact_mod_cast x_calc⟩
sorry

end logistics_personnel_in_sample_l771_771741


namespace prove_absolute_difference_of_C_and_D_l771_771806

/-- Define all variables and conditions in a Lean 4 statement. -/
def C : ℕ := 2
def D : ℕ := 3

def proof_problem (C D : ℕ) := 
  (C, D) ∈ (finset.range 4).product (finset.range 4) -- C and D are single-digit numbers in base 4

/-- Stating the proof problem in Lean 4. -/
theorem prove_absolute_difference_of_C_and_D : 
  ∀ (C D : ℕ), proof_problem C D → abs (C - D) = 1 :=
by 
  intros C D h
  sorry

end prove_absolute_difference_of_C_and_D_l771_771806


namespace small_drinking_glasses_count_l771_771097

theorem small_drinking_glasses_count :
  ∀ (large_jelly_beans_per_large_glass small_jelly_beans_per_small_glass total_jelly_beans : ℕ),
  (large_jelly_beans_per_large_glass = 50) →
  (small_jelly_beans_per_small_glass = large_jelly_beans_per_large_glass / 2) →
  (5 * large_jelly_beans_per_large_glass + n * small_jelly_beans_per_small_glass = total_jelly_beans) →
  (total_jelly_beans = 325) →
  n = 3 := by
  sorry

end small_drinking_glasses_count_l771_771097


namespace non_basalt_rocks_total_eq_l771_771195

def total_rocks_in_box_A : ℕ := 57
def basalt_rocks_in_box_A : ℕ := 25

def total_rocks_in_box_B : ℕ := 49
def basalt_rocks_in_box_B : ℕ := 19

def non_basalt_rocks_in_box_A : ℕ := total_rocks_in_box_A - basalt_rocks_in_box_A
def non_basalt_rocks_in_box_B : ℕ := total_rocks_in_box_B - basalt_rocks_in_box_B

def total_non_basalt_rocks : ℕ := non_basalt_rocks_in_box_A + non_basalt_rocks_in_box_B

theorem non_basalt_rocks_total_eq : total_non_basalt_rocks = 62 := by
  -- proof goes here
  sorry

end non_basalt_rocks_total_eq_l771_771195


namespace determine_friends_l771_771588

inductive Grade
| first
| second
| third
| fourth

inductive Name
| Petya
| Kolya
| Alyosha
| Misha
| Dima
| Borya
| Vasya

inductive Surname
| Ivanov
| Krylov
| Petrov
| Orlov

structure Friend :=
  (name : Name)
  (surname : Surname)
  (grade : Grade)

def friends : List Friend :=
  [ {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first},
    {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second},
    {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third},
    {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ]

theorem determine_friends : ∃ l : List Friend, 
  {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first} ∈ l ∧
  {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second} ∈ l ∧
  {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third} ∈ l ∧
  {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ∈ l :=
by 
  use friends
  repeat { simp [friends] }


end determine_friends_l771_771588


namespace smallest_base_l771_771252

theorem smallest_base (b : ℕ) : (b^2 ≤ 80 ∧ 80 < b^3) → b = 5 := by
  sorry

end smallest_base_l771_771252


namespace triangle_angle_measure_l771_771109

theorem triangle_angle_measure (A B C : ℝ) (AB AD CD AC BC : ℝ) (D : Point)
    (h1: AB + AD = CD)
    (h2: AC + AD = BC)
    (bisector_AD : isBisector AD ∧ D ∈ BC):
  A = 180 - 3 * C ∧ B = 2 * C ∧ C = C := 
  sorry

end triangle_angle_measure_l771_771109


namespace min_number_of_bags_l771_771576

theorem min_number_of_bags (a b : ℕ) : 
  ∃ K : ℕ, K = a + b - Nat.gcd a b :=
by
  sorry

end min_number_of_bags_l771_771576


namespace number_of_intersections_is_four_l771_771349

def LineA (x y : ℝ) : Prop := 3 * x - 2 * y + 4 = 0
def LineB (x y : ℝ) : Prop := 6 * x + 4 * y - 12 = 0
def LineC (x y : ℝ) : Prop := x - y + 1 = 0
def LineD (x y : ℝ) : Prop := y - 2 = 0

def is_intersection (L1 L2 : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := L1 p.1 p.2 ∧ L2 p.1 p.2

theorem number_of_intersections_is_four :
  (∃ p1 : ℝ × ℝ, is_intersection LineA LineB p1) ∧
  (∃ p2 : ℝ × ℝ, is_intersection LineC LineD p2) ∧
  (∃ p3 : ℝ × ℝ, is_intersection LineA LineD p3) ∧
  (∃ p4 : ℝ × ℝ, is_intersection LineB LineD p4) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :=
by
  sorry

end number_of_intersections_is_four_l771_771349


namespace angles_on_x_axis_l771_771190

theorem angles_on_x_axis (α : ℝ) : 
  (∃ k : ℤ, α = 2 * k * Real.pi) ∨ (∃ k : ℤ, α = (2 * k + 1) * Real.pi) ↔ 
  ∃ k : ℤ, α = k * Real.pi :=
by
  sorry

end angles_on_x_axis_l771_771190


namespace option_A_option_D_l771_771974

variable {a : ℕ → ℤ} -- The arithmetic sequence
variable {S : ℕ → ℤ} -- Sum of the first n terms
variable {a1 d : ℤ} -- First term and common difference

-- Conditions for arithmetic sequence
axiom a_n (n : ℕ) : a n = a1 + ↑(n-1) * d
axiom S_n (n : ℕ) : S n = n * a1 + (n * (n - 1) / 2) * d
axiom condition : a 4 + 2 * a 8 = a 6

theorem option_A : a 7 = 0 :=
by
  -- Proof to be done
  sorry

theorem option_D : S 13 = 0 :=
by
  -- Proof to be done
  sorry

end option_A_option_D_l771_771974


namespace combined_weight_l771_771330

variable (X : ℕ)
constant b r : ℕ
constant f : ℕ
constant barney_weight : ℕ
constant regular_dinosaur_weight : ℕ

axiom A1 : barney_weight = 1500 + 5 * regular_dinosaur_weight
axiom A2 : regular_dinosaur_weight = 800
axiom A3 : f = X

theorem combined_weight
  (A1 : barney_weight = 1500 + 5 * regular_dinosaur_weight)
  (A2 : regular_dinosaur_weight = 800)
  (A3 : f = X) :
  barney_weight + 5 * regular_dinosaur_weight + f = 9500 + X :=
by sorry

end combined_weight_l771_771330


namespace ways_to_draw_6_balls_with_score_less_than_8_l771_771531

theorem ways_to_draw_6_balls_with_score_less_than_8 :
  let red_balls := 5
      black_balls := 7
      total_balls := red_balls + black_balls
      score (red : ℕ) (black : ℕ) := red * 2 + black
      total_draws := 6 in
  (∑ red in finset.range(total_draws + 1), 
    if score red (total_draws - red) < 8 then
      nat.choose red_balls red * nat.choose black_balls (total_draws - red)
    else 0) = 112 :=
by
  sorry

end ways_to_draw_6_balls_with_score_less_than_8_l771_771531


namespace line_through_center_of_circle_l771_771878

theorem line_through_center_of_circle 
    (x y : ℝ) 
    (h : x^2 + y^2 - 4*x + 6*y = 0) : 
    3*x + 2*y = 0 :=
sorry

end line_through_center_of_circle_l771_771878


namespace option_b_is_correct_l771_771318

theorem option_b_is_correct :
  ∃ (f : ℝ → ℝ), (∀ x : ℝ, f x = - 1 / x)
  ∧ (∀ x : ℝ, f (-x) = - f (x))
  ∧ (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)
:=
begin
  use (λ x : ℝ, - 1 / x),
  split,
  { intro x,
    refl, },
  split,
  { intros x,
    simp, },
  { intros x y hx hy hxy,
    simp [hx, hy, hxy], },
end

end option_b_is_correct_l771_771318


namespace exercise_l771_771918

noncomputable def g (x : ℝ) : ℝ := x^3
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem exercise : f (g 3) = 1457 := by
  sorry

end exercise_l771_771918


namespace files_deleted_l771_771785

theorem files_deleted (original_files remaining_files : ℕ) (h1 : original_files = 21) (h2 : remaining_files = 7) : 
original_files - remaining_files = 14 :=
by
  rw [h1, h2]
  sorry

end files_deleted_l771_771785


namespace bench_press_after_injury_and_training_l771_771100

theorem bench_press_after_injury_and_training
  (p : ℕ) (h1 : p = 500) (h2 : p' : ℕ) (h3 : preduce : ℕ) (h4 : p' = p - preduce) 
  (h5 : preduce = 4 * p / 5) (h6 : q : ℕ) (h7 : q = 3 * p') : 
  q = 300 := by
  sorry

end bench_press_after_injury_and_training_l771_771100


namespace total_selling_price_l771_771736

def selling_price_A (purchase_price_A : ℝ) : ℝ :=
  purchase_price_A - (0.15 * purchase_price_A)

def selling_price_B (purchase_price_B : ℝ) : ℝ :=
  purchase_price_B + (0.10 * purchase_price_B)

def selling_price_C (purchase_price_C : ℝ) : ℝ :=
  purchase_price_C - (0.05 * purchase_price_C)

theorem total_selling_price 
  (purchase_price_A : ℝ)
  (purchase_price_B : ℝ)
  (purchase_price_C : ℝ)
  (loss_A : ℝ := 0.15)
  (gain_B : ℝ := 0.10)
  (loss_C : ℝ := 0.05)
  (total_price := selling_price_A purchase_price_A + selling_price_B purchase_price_B + selling_price_C purchase_price_C) :
  purchase_price_A = 1400 → purchase_price_B = 2500 → purchase_price_C = 3200 →
  total_price = 6980 :=
by sorry

end total_selling_price_l771_771736


namespace smallest_base_l771_771251

theorem smallest_base (b : ℕ) : (b^2 ≤ 80 ∧ 80 < b^3) → b = 5 := by
  sorry

end smallest_base_l771_771251


namespace gardening_club_interest_cents_part_l771_771735

noncomputable def compoundInterest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem gardening_club_interest_cents_part :
  ∀ (P A : ℝ) (r : ℝ) (n t : ℕ), 
     A = compoundInterest P r n t →
     A = 367.20 → r = 0.04 → n = 2 → t = 1 →
     let interest := A - P in 
     let cents := (interest * 100) % 100 in
     cents = 20 :=
by sorry

end gardening_club_interest_cents_part_l771_771735


namespace evaluate_neg_sixtyfour_exp_four_thirds_l771_771408

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l771_771408


namespace collinear_points_y_value_l771_771453

theorem collinear_points_y_value : ∀ y : ℝ, 
    points_collinear (point.mk 4 8) (point.mk 2 4) (point.mk 3 y) → y = 6 :=
begin
  intro y,
  intro h_collinear,
  sorry
end

end collinear_points_y_value_l771_771453


namespace sum_of_permuted_numbers_not_all_ones_l771_771069

theorem sum_of_permuted_numbers_not_all_ones (x y : ℕ) (hx : ∀ d ∈ (Nat.digits 10 x), d ≠ 0)
  (hy : Nat.digits 10 y = List.perm (Nat.digits 10 x)) :
  ∃ d ∈ (Nat.digits 10 (x + y)), d ≠ 1 :=
by
  sorry

end sum_of_permuted_numbers_not_all_ones_l771_771069


namespace calculate_number_of_committees_l771_771073

def number_of_committees 
  (num_senators : ℕ)
  (num_aides_per_senator : ℕ)
  (committees_of_5_senators : ℕ)
  (committees_of_4_senators_4_aides : ℕ)
  (committees_of_2_senators_12_aides : ℕ)
  (senators_per_committee_type_a : ℕ)
  (aides_per_committee_type_a : ℕ)
  (senators_per_committee_type_b : ℕ)
  (aides_per_committee_type_b : ℕ)
  (senators_per_committee_type_c : ℕ)
  (aides_per_committee_type_c : ℕ)
  (senator_committees : ℕ)
  (aide_committees : ℕ)
  (total_points_from_senators : ℕ)
  (total_points_from_aides : ℕ)
  (total_points : ℕ)
  (points_per_committee : ℕ)
:= sorry

theorem calculate_number_of_committees 
  (num_senators : ℕ) 
  (aides_per_senator : ℕ) 
  (committees_of_5_senators : ℕ) 
  (committees_of_4_senators_4_aides : ℕ) 
  (committees_of_2_senators_12_aides : ℕ) 
  (senator_committees : ℕ) 
  (aide_committees : ℕ) 
  (total_points_from_senators : 100 * 5) 
  (total_points_from_aides :  100 * 4 * 3 / 4) 
  (total_points : 800) 
  (points_per_committee : 5) 
  : (num_senators = 100) 
  → (aides_per_senator = 4) 
  → (committees_of_5_senators = 5) 
  → (committees_of_4_senators_4_aides = 5) 
  → (committees_of_2_senators_12_aides = 5) 
  → (senator_committees = 500) 
  → (aide_committees = 3 / 4)
  → (total_points_from_senators = 500)
  → (total_points_from_aides = 300)
  → (total_points = 800)
  → (points_per_committee = 5)
  → ∃ (total_committees : ℕ), total_committees = 160 
:= sorry

end calculate_number_of_committees_l771_771073


namespace has_inverse_a_has_inverse_c_has_inverse_d_has_inverse_f_has_inverse_g_has_inverse_h_l771_771786

noncomputable def a (x : ℝ) : ℝ := real.sqrt (3 - x)
def a_domain : set ℝ := {x | x ≤ 3}

noncomputable def b (x : ℝ) : ℝ := x^3 - 3 * x
def b_domain : set ℝ := set.univ

noncomputable def c (x : ℝ) : ℝ := x + 2 / x
def c_domain : set ℝ := {x | 0 < x}

noncomputable def d (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 10
def d_domain : set ℝ := {x | 0 ≤ x}

noncomputable def e (x : ℝ) : ℝ := abs (x - 3) + abs (x + 4)
def e_domain : set ℝ := set.univ

noncomputable def f (x : ℝ) : ℝ := 2^x + 5^x
def f_domain : set ℝ := set.univ

noncomputable def g (x : ℝ) : ℝ := x - 2 / x
def g_domain : set ℝ := {x | 0 < x}

noncomputable def h (x : ℝ) : ℝ := x / 3
def h_domain : set ℝ := {x | -3 ≤ x ∧ x < 8}

theorem has_inverse_a : ∀ x ∈ a_domain, ∃ y, a y = x := sorry
theorem has_inverse_c : ∀ x ∈ c_domain, ∃ y, c y = x := sorry
theorem has_inverse_d : ∀ x ∈ d_domain, ∃ y, d y = x := sorry
theorem has_inverse_f : ∀ x ∈ f_domain, ∃ y, f y = x := sorry
theorem has_inverse_g : ∀ x ∈ g_domain, ∃ y, g y = x := sorry
theorem has_inverse_h : ∀ x ∈ h_domain, ∃ y, h y = x := sorry

end has_inverse_a_has_inverse_c_has_inverse_d_has_inverse_f_has_inverse_g_has_inverse_h_l771_771786


namespace work_together_days_l771_771285

def work_rate_A : ℝ := 1 / 4
def work_rate_B : ℝ := 1 / 10
def work_done_B_alone_in_3_days : ℝ := work_rate_B * 3

theorem work_together_days :
  ∃ x : ℝ, (work_rate_A + work_rate_B) * x + work_done_B_alone_in_3_days = 1 → x = 2 :=
by {
  sorry
}

end work_together_days_l771_771285


namespace odd_functions_B_and_C_l771_771259

def f_A (x : ℝ) : ℝ := abs x
def f_B (x : ℝ) : ℝ := x + 1 / x
def f_C (x : ℝ) : ℝ := x^3 + 2 * x
def f_D (x : ℝ) : ℝ := x^2 + x + 1

theorem odd_functions_B_and_C :
  (∀ x : ℝ, f_B (-x) = -f_B x) ∧ (∀ x : ℝ, f_C (-x) = -f_C x) :=
by
  sorry

end odd_functions_B_and_C_l771_771259


namespace distance_between_points_l771_771237

-- Define the two points.
def point1 : ℝ × ℝ := (-3, 5)
def point2 : ℝ × ℝ := (4, -9)

-- Define the distance formula.
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- State the theorem.
theorem distance_between_points :
  distance point1 point2 = real.sqrt 245 :=
by
  -- Placeholder for the proof.
  sorry

end distance_between_points_l771_771237


namespace eval_neg_pow_l771_771381

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l771_771381


namespace min_n_for_sum_greater_than_62_l771_771567

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_ne_zero (x : ℝ) : g(x) ≠ 0
axiom f_prime_g_gt_f_g_prime (x : ℝ) : (deriv f x) * g(x) > f(x) * (deriv g x)

noncomputable def a : ℝ := 2

axiom f_def (x : ℝ) : f(x) = a^x * g(x)

axiom condition_at_1_and_minus_1 : (f 1) / (g 1) + (f (-1)) / (g (-1)) = 5 / 2

def sequence_term (n : ℕ) : ℝ := f(n) / g(n)

def sequence_sum (n : ℕ) : ℝ := (finset.range n).sum (λ i, sequence_term i)

theorem min_n_for_sum_greater_than_62 : ∃ (n : ℕ), sequence_sum n > 62 ∧ n = 6 := 
by {
  existsi 6,
  split,
  { -- Prove that the sum is greater than 62 for n = 6
    sorry
  },
  { -- Prove that 6 is the minimum value satisfying the condition
    sorry
  }
}

end min_n_for_sum_greater_than_62_l771_771567


namespace find_m_b_l771_771256

theorem find_m_b (m b : ℚ) :
  (3 * m - 14 = 2) ∧ (m ^ 2 - 6 * m + 15 = b) →
  m = 16 / 3 ∧ b = 103 / 9 := by
  intro h
  rcases h with ⟨h1, h2⟩
  -- proof steps here
  sorry

end find_m_b_l771_771256


namespace quadratic_point_comparison_l771_771522

theorem quadratic_point_comparison (c y1 y2 y3 : ℝ) 
  (h1 : y1 = -(-2:ℝ)^2 + c)
  (h2 : y2 = -(1:ℝ)^2 + c)
  (h3 : y3 = -(3:ℝ)^2 + c) : y2 > y1 ∧ y1 > y3 := 
by
  sorry

end quadratic_point_comparison_l771_771522


namespace dessert_menus_count_l771_771729

noncomputable def dessert_menus_possible : Nat := 20480

theorem dessert_menus_count :
  let desserts := ["cake", "pie", "ice cream", "pudding", "cookies"]
  let days := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
  let fixed_choices := [("Monday", "pie"), ("Friday", "cake")]
  let constraints := ∀ i j, i ≠ j ∧ i > 0 ∧ i < 7  → fixed_choices[i].snd ≠ fixed_choices[i-1].snd
  let menu_combinations := 5 * (4 ^ 6)
  menu_combinations = dessert_menus_possible := by
  sorry

end dessert_menus_count_l771_771729


namespace find_x_l771_771920

-- Definitions from the conditions
def isPositiveMultipleOf7 (x : ℕ) : Prop := ∃ k : ℕ, x = 7 * k ∧ x > 0
def xSquaredGreaterThan150 (x : ℕ) : Prop := x^2 > 150
def xLessThan40 (x : ℕ) : Prop := x < 40

-- Main problem statement
theorem find_x (x : ℕ) (h1 : isPositiveMultipleOf7 x) (h2 : xSquaredGreaterThan150 x) (h3 : xLessThan40 x) : x = 14 :=
sorry

end find_x_l771_771920


namespace count_integers_with_digit_sum_17_l771_771899

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + (n / 100)

theorem count_integers_with_digit_sum_17 : 
  let count := (finset.range 201).filter (λ x, sum_of_digits (x + 400) = 17)
  count.card = 13 :=
by
  sorry

end count_integers_with_digit_sum_17_l771_771899


namespace assign_grades_l771_771587

def is_not_first_grader : Type := sorry
def one_year_older (misha dima : Type) : Prop := sorry
def different_streets (vasya ivanov : Type) : Prop := sorry
def neighbors (boris orlov : Type) : Prop := sorry
def met_one_year_ago (krylov petrov : Type) : Prop := sorry
def gave_textbook_last_year (vasya boris : Type) : Prop := sorry

theorem assign_grades 
  (name : Type) 
  (surname : Type) 
  (grade : Type) 
  (Dima Misha Boris Vasya : name) 
  (Ivanov Krylov Petrov Orlov : surname)
  (first second third fourth : grade)
  (h1 : ¬is_not_first_grader Boris)
  (h2 : different_streets Vasya Ivanov)
  (h3 : one_year_older Misha Dima)
  (h4 : neighbors Boris Orlov)
  (h5 : met_one_year_ago Krylov Petrov)
  (h6 : gave_textbook_last_year Vasya Boris) : 
  (Dima, Ivanov, first) ∧
  (Misha, Krylov, second) ∧
  (Boris, Petrov, third) ∧
  (Vasya, Orlov, fourth) :=
sorry

end assign_grades_l771_771587


namespace car_speed_second_third_l771_771286

theorem car_speed_second_third (D : ℝ) (t1 t3 : ℝ) (avg_speed : ℝ) : 
    let V := 30 in
    D > 0 ∧ t1 = D / (3 * 80) ∧ t3 = D / (3 * 48) ∧ avg_speed = 45 → 
    avg_speed = D / (t1 + (D / (3 * V)) + t3) :=
by
  intros h1 h2 h3 h4
  have h5 : 45 = D / (D / (3 * 80) + D / (3 * 30) + D / (3 * 48)), by sorry
  rw h5
  sorry

end car_speed_second_third_l771_771286


namespace plane_equation_correct_l771_771421

-- Definition of points and planes
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def plane_eq (A B C D : ℝ) (P : Point3D) : Prop :=
  A * P.x + B * P.y + C * P.z + D = 0

-- Given points
def P1 : Point3D := ⟨2, -1, 0⟩
def P2 : Point3D := ⟨0, 3, 1⟩

-- Given normal vector to the existing plane
def normal_vec_plane1 : Point3D := ⟨2, -1, 4⟩

-- Equation of the required plane
noncomputable def required_plane_eq : Prop :=
  let normal_vec_required_plane := ⟨17, 10, -6⟩
  let D := -24
  plane_eq normal_vec_required_plane.x normal_vec_required_plane.y normal_vec_required_plane.z D P1 ∧
  plane_eq normal_vec_required_plane.x normal_vec_required_plane.y normal_vec_required_plane.z D P2

theorem plane_equation_correct : required_plane_eq := by 
  sorry  -- Proof goes here, but is not required as per instructions

end plane_equation_correct_l771_771421


namespace symmetric_point_coordinates_l771_771542

def point_symmetric_to_x_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (x, -y, -z)

theorem symmetric_point_coordinates :
  point_symmetric_to_x_axis (-2, 1, 4) = (-2, -1, -4) := by
  sorry

end symmetric_point_coordinates_l771_771542


namespace paco_initial_cookies_l771_771585

theorem paco_initial_cookies (cookies_ate : ℕ) (cookies_left : ℕ) (cookies_initial : ℕ) 
  (h1 : cookies_ate = 15) (h2 : cookies_left = 78) :
  cookies_initial = cookies_ate + cookies_left → cookies_initial = 93 :=
by
  sorry

end paco_initial_cookies_l771_771585


namespace find_principal_amount_l771_771707

variable (x y : ℝ)

-- conditions given in the problem
def simple_interest_condition : Prop :=
  600 = (x * y * 2) / 100

def compound_interest_condition : Prop :=
  615 = x * ((1 + y / 100)^2 - 1)

-- target statement to be proven
theorem find_principal_amount (h1 : simple_interest_condition x y) (h2 : compound_interest_condition x y) :
  x = 285.7142857 :=
  sorry

end find_principal_amount_l771_771707


namespace greater_difference_implication_l771_771530

variable (a b c d n : ℝ)

def K_squared (a b c d n : ℝ) : ℝ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem greater_difference_implication
  (h : (a * d - b * c) > 0):
  (K_squared a b c d n > 0) :=
by
  sorry

end greater_difference_implication_l771_771530


namespace lattice_points_on_hyperbola_l771_771908

theorem lattice_points_on_hyperbola :
  ∃ (s : Finset (ℤ × ℤ)), 
  (∀ p ∈ s, let (x, y) := p in x^2 - y^2 = 65) ∧ 
  (4 : Finset.card s) :=
begin
  sorry
end

end lattice_points_on_hyperbola_l771_771908


namespace sequence_geometric_and_sum_l771_771448

variables {S : ℕ → ℝ} (a1 : S 1 = 1)
variable (n : ℕ)
def a := (S (n+1) - 2 * S n, S n)
def b := (2, n)

/-- Prove that the sequence {S n / n} is a geometric sequence 
with first term 1 and common ratio 2, and find the sum of the first 
n terms of the sequence {S n} -/
theorem sequence_geometric_and_sum {S : ℕ → ℝ} (a1 : S 1 = 1)
  (n : ℕ)
  (parallel : ∀ n, n * (S (n + 1) - 2 * S n) = 2 * S n) :
  ∃ r : ℝ, r = 2 ∧ ∃ T : ℕ → ℝ, T n = (n-1)*2^n + 1 :=
by
  sorry

end sequence_geometric_and_sum_l771_771448


namespace express_as_difference_of_increasing_functions_l771_771793

def f (x : ℝ) : ℝ := 2 * x ^ 2 + 5 * x - 3
def g (x : ℝ) : ℝ := x ^ 3 + 2 * x ^ 2 + 5 * x - 3
def h (x : ℝ) : ℝ := x ^ 3

theorem express_as_difference_of_increasing_functions :
  ∃ (g h : ℝ → ℝ), (∀ x y : ℝ, x < y → g x < g y) ∧ (∀ x y : ℝ, x < y → h x < h y) ∧ (∀ x : ℝ, f x = g x - h x) :=
by
  use [g, h]
  sorry

end express_as_difference_of_increasing_functions_l771_771793


namespace eval_expression_correct_l771_771397

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l771_771397


namespace min_value_expression_l771_771121

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    let a := 2
    let b := 3
    let term1 := 2*x + 1/(3*y)
    let term2 := 3*y + 1/(2*x)
    (term1 * (term1 - 2023) + term2 * (term2 - 2023)) = -2050529.5 :=
sorry

end min_value_expression_l771_771121


namespace cookies_remaining_percentage_l771_771200

theorem cookies_remaining_percentage: 
  ∀ (total initial_remaining eduardo_remaining final_remaining: ℕ),
  total = 600 → 
  initial_remaining = total - (2 * total / 5) → 
  eduardo_remaining = initial_remaining - (3 * initial_remaining / 5) → 
  final_remaining = eduardo_remaining → 
  (final_remaining * 100) / total = 24 := 
by
  intros total initial_remaining eduardo_remaining final_remaining h_total h_initial_remaining h_eduardo_remaining h_final_remaining
  sorry

end cookies_remaining_percentage_l771_771200


namespace length_of_PR_square_l771_771948

noncomputable def square_side := 20

def triangle_area (x : ℝ) : ℝ := 2 * x^2 + (square_side - x)^2

theorem length_of_PR_square {x s : ℝ} (h1 : s = square_side) (h2 : triangle_area x = 400) : 
    PR = s - x := 
sorry

end length_of_PR_square_l771_771948


namespace hexagon_area_l771_771187

-- Definitions of the conditions
def DEF_perimeter := 42
def circumcircle_radius := 10
def area_of_hexagon_DE'F'D'E'F := 210

-- The theorem statement
theorem hexagon_area (DEF_perimeter : ℕ) (circumcircle_radius : ℕ) : Prop :=
  DEF_perimeter = 42 → circumcircle_radius = 10 → 
  area_of_hexagon_DE'F'D'E'F = 210

-- Example invocation of the theorem, proof omitted.
example : hexagon_area DEF_perimeter circumcircle_radius :=
by {
  sorry
}

end hexagon_area_l771_771187


namespace one_empty_box_methods_l771_771282

theorem one_empty_box_methods : 
  (number_of_methods 4 3) = 144 :=
sorry

end one_empty_box_methods_l771_771282


namespace find_g_inv_f_of_10_l771_771050

noncomputable def f_inv : ℝ → ℝ := sorry -- Definition of f's inverse function
noncomputable def g : ℝ → ℝ := sorry -- Definition of function g
noncomputable def f : ℝ → ℝ := sorry -- Definition of function f
noncomputable def g_inv : ℝ → ℝ := sorry -- Definition of g's inverse function

-- Condition that captures f's inverse in terms of g
def condition (x : ℝ) : Prop := f_inv(g(x)) = x^2 - 4
-- Condition stating that g has an inverse
def g_has_inverse : Prop := ∀ y, ∃ x, g x = y

theorem find_g_inv_f_of_10 :
  (condition 10) → (g_has_inverse) → g_inv(f 10) = real.sqrt 14 :=
by
  intros,
  sorry -- skip the proof

end find_g_inv_f_of_10_l771_771050


namespace ambika_candles_count_l771_771325

-- Definitions
def Aniyah_candles (A : ℕ) : ℕ := 6 * A
def combined_candles (A : ℕ) : ℕ := A + Aniyah_candles A

-- Problem Statement:
theorem ambika_candles_count : ∃ A : ℕ, combined_candles A = 28 ∧ A = 4 :=
by
  sorry

end ambika_candles_count_l771_771325


namespace factorization_l771_771796

theorem factorization (m : ℤ) : m^2 + 3 * m = m * (m + 3) :=
by sorry

end factorization_l771_771796


namespace tangent_slope_through_origin_l771_771521

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^a + 1

theorem tangent_slope_through_origin (a : ℝ) (h : curve a 1 = 2) 
  (tangent_passing_through_origin : ∀ y, (y - 2 = a * (1 - 0)) → y = 0): a = 2 := 
sorry

end tangent_slope_through_origin_l771_771521


namespace total_amount_paid_correct_l771_771324

def price_per_kg_grapes := 74
def quantity_grapes := 6
def price_per_kg_mangoes := 59
def quantity_mangoes := 9

def cost_grapes := price_per_kg_grapes * quantity_grapes
def cost_mangoes := price_per_kg_mangoes * quantity_mangoes
def total_amount_paid := cost_grapes + cost_mangoes

theorem total_amount_paid_correct : total_amount_paid = 975 := by
  unfold total_amount_paid cost_grapes cost_mangoes
  simp [price_per_kg_grapes, quantity_grapes, price_per_kg_mangoes, quantity_mangoes]
  sorry

end total_amount_paid_correct_l771_771324


namespace range_and_increasing_intervals_l771_771482

-- Definitions for given conditions
def f (ω x : ℝ) : ℝ := 2 * (Real.sin (ω * x - Real.pi / 6)) - 1

-- Main Theorem
theorem range_and_increasing_intervals (ω : ℝ) (hω : ω > 0) :
  (Set.range (f ω)) = Set.Icc (-3 : ℝ) 1 ∧
  (∀ k : ℤ, ∃ (a b : ℝ), y = -1 intersects f ω x at two adjacent points (a, b) 
  ∧ (b - a = Real.pi / 2)
  ∧ ω = 2
  ∧ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3) 
  \text{are intervals where } f 2 \text{ is increasing} := by
    sorry

end range_and_increasing_intervals_l771_771482


namespace proof_4_minus_a_l771_771048

theorem proof_4_minus_a :
  ∀ (a b : ℚ),
    (5 + a = 7 - b) →
    (3 + b = 8 + a) →
    4 - a = 11 / 2 :=
by
  intros a b h1 h2
  sorry

end proof_4_minus_a_l771_771048


namespace non_integer_sum_l771_771573

theorem non_integer_sum (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ (M : ℕ), ∀ n, n > M → ¬ (⌊(k + 1 / 2)^n⌋ + ⌊(l + 1 / 2)^n⌋ ∈ ℤ) :=
sorry

end non_integer_sum_l771_771573


namespace no_valid_schedule_l771_771046

theorem no_valid_schedule :
  ∀ (days : ℕ) (courses : ℕ) (total_periods : ℕ),
  total_periods = 7 →
  courses = 3 →
  (∃ (schedule : list ℕ), 
    length schedule = courses ∧ 
    ∀ (i : ℕ), i < courses - 1 → schedule.nth i.succ ≠ schedule.nth i + 1) →
  0 := 
by
  intros days courses total_periods ht hc hs
  sorry

end no_valid_schedule_l771_771046


namespace perpendicular_planes_and_lines_l771_771973

variables (α β γ m n : Type)
variables [plane α] [plane β] [plane γ]
variables [line m] [line n]

theorem perpendicular_planes_and_lines
  (hne_alpha : perpendicular n α)
  (hne_beta : perpendicular n β)
  (hme_alpha : perpendicular m α) :
  perpendicular m β :=
  sorry

end perpendicular_planes_and_lines_l771_771973


namespace valid_three_digit_palindromes_l771_771277

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def valid_three_digit_numbers : Finset ℕ :=
  {102, 103, 122}

theorem valid_three_digit_palindromes (n : ℕ) :
  100 ≤ n ∧ n ≤ 316 ∧ n = 102 ∧ is_palindrome (n^2) → n ∈ valid_three_digit_numbers :=
sorry

end valid_three_digit_palindromes_l771_771277


namespace milk_production_l771_771612

theorem milk_production (a b c d e : ℝ) (h1 : a ≠ 0) (h2 : c ≠ 0) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h6 : d > 0) (h7 : e > 0) :
  (d * 0.9 * b * e) / (a * c) = 0.9 * (d * b * e) / (a * c) :=
begin
  sorry
end

end milk_production_l771_771612


namespace flu_infection_l771_771303

theorem flu_infection (x : ℕ) (H : 1 + x + x^2 = 36) : True :=
begin
  sorry
end

end flu_infection_l771_771303


namespace transport_cost_expression_and_min_cost_l771_771211

noncomputable def total_transport_cost (x : ℕ) (a : ℕ) : ℕ :=
if 2 ≤ a ∧ a ≤ 6 then (5 - a) * x + 23200 else 0

theorem transport_cost_expression_and_min_cost :
  ∀ x : ℕ, ∀ a : ℕ,
  (100 ≤ x ∧ x ≤ 800) →
  (2 ≤ a ∧ a ≤ 6) →
  (total_transport_cost x a = 5 * x + 23200) ∧ 
  (a = 6 → total_transport_cost 800 a = 22400) :=
by
  intros
  -- Provide the detailed proof here.
  sorry

end transport_cost_expression_and_min_cost_l771_771211


namespace factorize_expression_l771_771799

theorem factorize_expression (m : ℝ) : m^2 + 3 * m = m * (m + 3) :=
by
  sorry

end factorize_expression_l771_771799


namespace negation_of_universal_l771_771186

theorem negation_of_universal:
  ¬(∀ x : ℕ, x^2 > 1) ↔ ∃ x : ℕ, x^2 ≤ 1 :=
by sorry

end negation_of_universal_l771_771186


namespace avg_gas_mileage_round_trip_l771_771732

def avg_gas_mileage (d1 d2 m1 m2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let total_gallons := d1 / m1 + d2 / m2
  total_distance / total_gallons

theorem avg_gas_mileage_round_trip :
  avg_gas_mileage 150 180 25 15 = 18 :=
by
  sorry

end avg_gas_mileage_round_trip_l771_771732


namespace MQ_parallel_AL_l771_771673

-- Definitions based on the conditions
variables (P L A B C E M Q : Point)
variables (w1 w2 : Circle)
variables (line_EL : Line)

-- Given conditions
axiom touching_circles_at_L : w1.touching w2 L
axiom w1_touches_AB_at_E : touches_ray w1 A B E
axiom w2_touches_AC_at_M : touches_ray w2 A C M
axiom line_EL_meets_w2_at_Q : meets_line_twice line_EL w2 L Q

-- The property to prove
theorem MQ_parallel_AL : parallel (line_through M Q) (line_through A L) :=
begin
  sorry
end

end MQ_parallel_AL_l771_771673


namespace lcm_18_45_l771_771695

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end lcm_18_45_l771_771695


namespace determine_multiple_l771_771991

-- Define the conditions
def regular_hours : ℝ := 7.5
def excess_hours (total_hours : ℝ) : ℝ := total_hours - regular_hours
def regular_rate : ℝ := 4.5
def total_hours : ℝ := 10.5
def total_earnings : ℝ := 67.5

-- Define the earnings calculation for regular and excess hours
def earnings_regular_hours : ℝ := regular_hours * regular_rate
def earnings_excess_hours : ℝ := total_earnings - earnings_regular_hours
def excess_rate (total_hours : ℝ) : ℝ := earnings_excess_hours / excess_hours total_hours

-- Define the multiple to be proven
def multiple : ℝ := excess_rate total_hours / regular_rate

-- Theorem statement
theorem determine_multiple : multiple = 2.5 :=
by
  sorry

end determine_multiple_l771_771991


namespace area_triangle_STU_l771_771312

-- Definition of the points and lengths based on given conditions
def W : Point := ⟨0, 0, 0⟩
def X : Point := ⟨4, 0, 0⟩
def Y : Point := ⟨4, 4, 0⟩
def Z : Point := ⟨0, 4, 0⟩
def A : Point := ⟨2, 2, 8⟩

def S : Point := Point.between W A (1/4 : ℝ)
def T : Point := Point.between X A (1/2 : ℝ)
def U : Point := Point.between Y A (3/4 : ℝ)

-- Statement of the problem: The area of triangle STU is 7.5 square centimeters
theorem area_triangle_STU : area (triangle S T U) = 7.5 := by
  sorry

end area_triangle_STU_l771_771312


namespace distance_between_points_l771_771236

-- Define the two points.
def point1 : ℝ × ℝ := (-3, 5)
def point2 : ℝ × ℝ := (4, -9)

-- Define the distance formula.
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- State the theorem.
theorem distance_between_points :
  distance point1 point2 = real.sqrt 245 :=
by
  -- Placeholder for the proof.
  sorry

end distance_between_points_l771_771236


namespace count_integers_with_sum_digits_17_l771_771904

-- Define the sum of digits of a number
def sum_digits (n : ℕ) : ℕ :=
  (n / 100) + (n % 100) / 10 + (n % 10)

-- Define the condition that integer is between 400 and 600
def in_range (n : ℕ) : Prop :=
  400 ≤ n ∧ n ≤ 600

-- Define the main theorem
theorem count_integers_with_sum_digits_17 : ∃ k = 13, ∃ (n : ℕ), in_range n ∧ sum_digits n = 17 := by
  sorry

end count_integers_with_sum_digits_17_l771_771904


namespace min_sum_nonneg_l771_771824

theorem min_sum_nonneg (n : ℕ) (hn : 0 < n) (x : ℕ → ℝ) :
  (∑ i in Finset.range n, ∑ j in Finset.range n, min i.succ j.succ * x i * x j) ≥ 0 :=
by sorry

end min_sum_nonneg_l771_771824


namespace fraction_of_tank_used_is_5_12_l771_771329

-- Define the conditions given in the problem
def speed : ℝ := 50 -- miles per hour
def consumption_rate : ℝ := 1 / 30 -- gallons per mile
def initial_gas : ℝ := 20 -- gallons
def travel_time : ℝ := 5 -- hours

-- Define the distance traveled and gallons used
def distance_traveled : ℝ := speed * travel_time
def gallons_used : ℝ := distance_traveled * consumption_rate

-- Define the fraction of the tank used
def fraction_of_tank_used : ℝ := gallons_used / initial_gas

-- State the theorem to be proven
theorem fraction_of_tank_used_is_5_12 : fraction_of_tank_used = 5 / 12 :=
by
  sorry

end fraction_of_tank_used_is_5_12_l771_771329


namespace cosine_angle_between_diagonals_l771_771300

open RealMatrix

def v1 : ℝ ^ 3 := ![3, 2, 1]
def v2 : ℝ ^ 3 := ![2, -2, -2]

noncomputable def dot_product (u v : ℝ ^ 3) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1) + (u 2) * (v 2)

noncomputable def norm (u : ℝ ^ 3) : ℝ :=
  Real.sqrt ((u 0) ^ 2 + (u 1) ^ 2 + (u 2) ^ 2)

def diagonal1 : ℝ ^ 3 := ![v1 0 + v2 0, v1 1 + v2 1, v1 2 + v2 2]
def diagonal2 : ℝ ^ 3 := ![v2 0 - v1 0, v2 1 - v1 1, v2 2 - v1 2]

noncomputable def cosine_phi : ℝ :=
  dot_product diagonal1 diagonal2 / (norm diagonal1 * norm diagonal2)

theorem cosine_angle_between_diagonals :
  cosine_phi = -1 / 13 :=
by
  sorry

end cosine_angle_between_diagonals_l771_771300


namespace real_tax_revenue_decrease_approx_31_percent_l771_771327

noncomputable def percentage_reduction_in_real_tax_revenue : ℝ :=
let initial_nominal_value : ℝ := 100000
let real_value : ℝ := 100000
let first_year_interest_rate : ℝ := 0.25
let first_year_interest : ℝ := initial_nominal_value * first_year_interest_rate
let first_year_tax_rate : ℝ := 0.20
let first_year_tax : ℝ := first_year_interest * first_year_tax_rate
let first_year_interest_after_tax : ℝ := first_year_interest - first_year_tax
let first_year_real_tax : ℝ := first_year_tax / (1 + first_year_interest_rate)

let nominal_value_second_year : ℝ := initial_nominal_value * (1 + first_year_interest_rate)
let second_year_interest_rate : ℝ := 0.16
let second_year_interest : ℝ := nominal_value_second_year * second_year_interest_rate
let second_year_tax : ℝ := second_year_interest * first_year_tax_rate
let second_year_real_tax : ℝ := second_year_tax / ((1 + first_year_interest_rate) * (1 + second_year_interest_rate))

let reduction_in_real_tax : ℝ := first_year_real_tax - second_year_real_tax
let percentage_reduction : ℝ := (reduction_in_real_tax / first_year_real_tax) * 100 in
Real.floor (percentage_reduction + 0.5)

theorem real_tax_revenue_decrease_approx_31_percent:
  percentage_reduction_in_real_tax_revenue ≈ 31 :=
sorry

end real_tax_revenue_decrease_approx_31_percent_l771_771327


namespace arithmetic_sequence_properties_l771_771858

def arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties (a : ℕ → ℤ) (d : ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a4 : a 4 = 10)
  (h_S6_S3_plus_39 : (∑ i in finset.range 6, a i) = (∑ i in finset.range 3, a i) + 39) :
  a 1 = 1 ∧ (∀ n : ℕ, a n = 3 * n - 2) :=
by
  sorry

end arithmetic_sequence_properties_l771_771858


namespace jersey_cost_l771_771103

theorem jersey_cost (
  (long_sleeves : ℕ) (striped : ℕ) (striped_cost : ℕ) (total_spent : ℕ) (x : ℕ) :
  long_sleeves = 4 ∧ striped = 2 ∧ striped_cost = 10 ∧ total_spent = 80 ∧ 4 * x + 2 * striped_cost = total_spent) :
  x = 15 :=
by
  sorry

end jersey_cost_l771_771103


namespace highest_value_C_n_l771_771429

open Function

theorem highest_value_C_n (n : ℕ) (hpos : 0 < n) (a : ℕ → ℤ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  (n + 1) * (Finset.range n).sum (λ i, (a i) ^ 2) 
  - ((Finset.range n).sum (λ i, a i)) ^ 2 ≥ n * (n - 1) :=
by
  sorry

end highest_value_C_n_l771_771429


namespace find_K_values_l771_771062

theorem find_K_values (K M : ℕ) (h1 : (K * (K + 1)) / 2 = M^2) (h2 : M < 200) (h3 : K > M) :
  K = 8 ∨ K = 49 :=
sorry

end find_K_values_l771_771062


namespace flu_infection_equation_l771_771302

theorem flu_infection_equation (x : ℕ) (h : 1 + x + x^2 = 36) : 1 + x + x^2 = 36 :=
by
  sorry

end flu_infection_equation_l771_771302


namespace A_is_odd_l771_771112

theorem A_is_odd (m n : ℕ) (h : ∃ A : ℕ, A = (m + 3)^n + 1 / 3 / m) : odd (A) :=
sorry

end A_is_odd_l771_771112


namespace algebraic_notation_equivalence_l771_771792

-- Define the variables
variables (x y : ℤ)

-- Define "three times x"
def three_times_x (x : ℤ) := 3 * x

-- Define "the cube of y"
def cube_of_y (y : ℤ) := y ^ 3

-- Define the target expression
def target_expression (x y : ℤ) := three_times_x x - cube_of_y y

-- Theorem stating the equivalence of the target expression to 3x - y^3
theorem algebraic_notation_equivalence : target_expression x y = 3 * x - y ^ 3 := by
  sorry

end algebraic_notation_equivalence_l771_771792


namespace transformed_mean_variance_l771_771929

variables {n : ℕ} {x : Finₓ n → ℝ} {x' : Finₓ n → ℝ}
noncomputable def mean (x : Finₓ n → ℝ) : ℝ := (∑ i, x i) / n

noncomputable def variance (x : Finₓ n → ℝ) : ℝ :=
  (∑ i, (x i - mean x)^2) / n

theorem transformed_mean_variance
  (hx : mean x = x.mean)
  (hs : variance x = s^2) :
  mean (fun i => 3 * x i + 5) = 3 * x.mean + 5 ∧
  variance (fun i => 3 * x i + 5) = 9 * s^2 :=
by
  sorry

end transformed_mean_variance_l771_771929


namespace sum_of_sequence_l771_771002

theorem sum_of_sequence (n : ℕ) (h_pos : 0 < n) :
  let a_n := λ n : ℕ, (Nat.choose (n + 2) n) in
  let sequence := λ (i : ℕ), 1 / a_n i in
  ∑ i in Finset.range n, sequence i = (n : ℚ) / (n + 2) := by
  sorry

end sum_of_sequence_l771_771002


namespace length_of_extended_crease_l771_771307

def width := 7  -- width of the rectangular piece of paper
def extension := 2  -- length of the additional flap
variable θ : ℝ  -- defining the angle theta
def secθ := Real.sec θ  -- sec(theta)
def sinθ := Real.sin θ  -- sin(theta)

theorem length_of_extended_crease (θ : ℝ) : 
  (7 * secθ) + (2 * sinθ) = 7 * Real.sec θ + 2 * Real.sin θ :=
sorry

end length_of_extended_crease_l771_771307


namespace power_evaluation_l771_771387

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l771_771387


namespace functional_equation_solution_l771_771800

noncomputable def f (x : ℚ) : ℚ := sorry

theorem functional_equation_solution (f : ℚ → ℚ) (f_pos_rat : ∀ x : ℚ, 0 < x → 0 < f x) :
  (∀ x y : ℚ, 0 < x → 0 < y → f x + f y + 2 * x * y * f (x * y) = f (x * y) / f (x + y)) →
  (∀ x : ℚ, 0 < x → f x = 1 / x ^ 2) :=
by
  sorry

end functional_equation_solution_l771_771800


namespace evaluate_neg_sixtyfour_exp_four_thirds_l771_771406

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l771_771406


namespace sum_of_smallest_five_angles_l771_771342

noncomputable def Q (x : ℂ) : ℂ := ((x^21 - 1) * (x^19 - 1)) / (x - 1)^2

theorem sum_of_smallest_five_angles :
  (let β1 := 1/21;
       β2 := 1/19;
       β3 := 2/21;
       β4 := 2/19;
       β5 := 3/21 
   in β1 + β2 + β3 + β4 + β5 = 177 / 399) :=
by {
    let β1 := 1 / 21;
    let β2 := 1 / 19;
    let β3 := 2 / 21;
    let β4 := 2 / 19;
    let β5 := 3 / 21;
    show β1 + β2 + β3 + β4 + β5 = 177 / 399,
    sorry
}

end sum_of_smallest_five_angles_l771_771342


namespace sin_theta_correct_l771_771563

noncomputable def sin_theta : ℝ :=
  let d := (4, 5, 7)
  let n := (3, -4, 5)
  let d_dot_n := 4 * 3 + 5 * (-4) + 7 * 5
  let norm_d := Real.sqrt (4^2 + 5^2 + 7^2)
  let norm_n := Real.sqrt (3^2 + (-4)^2 + 5^2)
  let cos_theta := d_dot_n / (norm_d * norm_n)
  cos_theta

theorem sin_theta_correct :
  sin_theta = 27 / Real.sqrt 4500 :=
by
  sorry

end sin_theta_correct_l771_771563


namespace count_quadratic_functions_l771_771490

theorem count_quadratic_functions :
  let S := {0, 2, 4, 6, 8} in
  let a_values := {2, 4, 6, 8} in
  let b_values := S in
  let c_values := S in
  ∃ (a b c : ℕ), a ∈ a_values ∧ b ∈ b_values ∧ c ∈ c_values ∧ a ≠ 0 →
  (4 * 5 * 5 = 100) :=
by
  intros
  sorry

end count_quadratic_functions_l771_771490


namespace f_of_90_l771_771294

def f (n : ℕ) : ℕ :=
  if n ≥ 1000 then
    n - 3
  else
    f (f (n + 7))

theorem f_of_90 : f 90 = 999 :=
  sorry

end f_of_90_l771_771294


namespace locus_area_problem_l771_771967

noncomputable def problem_statement : ℕ :=
sorry -- insert a precise Lean representation of the problem statement here.

theorem locus_area_problem (
  (Γ₁ : Type) [metric_space Γ₁] [normed_group Γ₁] (O₁ : Γ₁) (r₁ : ℝ) (P : Γ₁) (HPO₁ : dist P O₁ = r₁),
  (Γ₂ : Type) [metric_space Γ₂] [normed_group Γ₂] (O₂ : Γ₂) (r₂ : ℝ) (Q : Γ₂) (HQO₂ : dist Q O₂ = r₂),
  (OO₂ : dist O₁ O₂ = 2),
  (Ω : Type) [metric_space Ω] [normed_group Ω] (rΩ : ℝ) (HO₁P : dist P O₁ = rΩ) (HQO₂Ω : dist Q O₂ = rΩ)
) : ∃ (p q : ℕ), p + q = 16909 :=
sorry

end locus_area_problem_l771_771967


namespace sum_a1_to_a5_max_sum_S_n_l771_771870

-- Arithmetic sequence definition
noncomputable def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Geometric sequence definition for three terms
def geometric_three (a1 a2 a5 : ℝ) : Prop :=
(a1 + (a2 - a1))^2 = a1 * (a1 + 4 * (a2 - a1))

-- Definitions based on given conditions
constant a : ℕ → ℝ
constant d : ℝ 

axiom h_arith_seq : arithmetic_seq a d
axiom h_geo_seq : geometric_three (a 0) (a 1) (a 4)
axiom h_sum_a3_a4 : a 2 + a 3 = 12

-- Definition for modified sequence b_n
noncomputable def b (n : ℕ) : ℝ := 10 - a n

-- Definition for sum of the first n terms of sequence b
noncomputable def S (n : ℕ) : ℝ :=
∑ i in finset.range n, b i

-- Axiom stating b_1 ≠ b_2
axiom h_b1_ne_b2 : b 0 ≠ b 1

-- Statement (1): Prove the sum a1 + a2 + a3 + a4 + a5
theorem sum_a1_to_a5 : a 0 + a 1 + a 2 + a 3 + a 4 = 25 := sorry

-- Statement (2): Prove the maximum S_n is 25 when n = 5
theorem max_sum_S_n : ∃ n, S n = 25 ∧ n = 5 := sorry

end sum_a1_to_a5_max_sum_S_n_l771_771870


namespace expected_heads_of_alice_l771_771757

noncomputable def expected_heads_alice (X Y : ℕ → ℝ) (n : ℕ) :=
  \[ \mathbb{E}[X \mid X \geq Y] = 20 \cdot \frac{2^{38} + \binom{39}{19}}{2^{39} + \binom{39}{19}} \]

theorem expected_heads_of_alice (n : ℕ) (X Y : ℕ → ℝ) :
  ( ∀ i : ℕ, X i = (0:ℝ) ∨ X i = 1) →
  ( ∀ i : ℕ, Y i = (0:ℝ) ∨ Y i = 1) →
  ( ∀ i, X(i) = X(i % n)) →
  ( ∀ i, Y(i) = Y(i % n)) →
  @expected_heads_alice X Y n =
    20 * (2^38 + Mathlib.Combinatorics.Binom.binom(39, 19)) /
    (2^39 + Mathlib.Combinatorics.Binom.binom(39, 19)) := 
sorry

end expected_heads_of_alice_l771_771757


namespace domain_of_f_l771_771622

noncomputable def f (x : ℝ) := Real.sqrt (x - 1) + (1 / (x - 2))

theorem domain_of_f : { x : ℝ | x ≥ 1 ∧ x ≠ 2 } = { x : ℝ | ∃ (y : ℝ), f x = y } :=
sorry

end domain_of_f_l771_771622


namespace cosine_sum_equals_segment_length_l771_771972

variable {A B C M O : Point}

-- Definition of a triangle inscribed in a unit circle
def is_triangle_inscribed (A B C O: Point) : Prop :=
  dist O A = 1 ∧ dist O B = 1 ∧ dist O C = 1

-- Definition of the orthocenter
def is_orthocenter (M A B C : Point) : Prop :=
  collinear A B C ∧ collinear A B M ∧ collinear B C M ∧ collinear C A M

-- Definition to get the distance between points
def dist (A B : Point) : ℝ := 
  sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

theorem cosine_sum_equals_segment_length
  (A B C M O : Point)
  (h_inscribed: is_triangle_inscribed A B C O)
  (h_orthocenter: is_orthocenter M A B C) :
  cos (angle M O A) + cos (angle M O B) + cos (angle M O C) = dist M O :=
sorry

end cosine_sum_equals_segment_length_l771_771972


namespace min_value_of_f_l771_771443

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / (x - 1)

theorem min_value_of_f : 
  ∃ x : ℝ, (1 < x) ∧ (∀ y : ℝ, 1 < y → f y ≥ 2 + 2 * Real.sqrt 2) :=
begin
  use 1 + Real.sqrt 2 / 2,
  split,
  { linarith [Real.sqrt 2_pos], },
  { intros y hy,
    have h1 : y = 1 + Real.sqrt 2 / 2 → f y = 2 + 2 * Real.sqrt 2,
    { intro hy_eq,
      rw [hy_eq, f],
      field_simp [ne_of_gt],
      ring_nf,
      rcases @Real.sqrt_pos 2,
   },
    have h2 : 1 < y → f y ≥ 2 + 2 * Real.sqrt 2,
      sorry }
end

end min_value_of_f_l771_771443


namespace power_evaluation_l771_771389

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l771_771389


namespace largest_prime_divisor_test_range_1000_1100_l771_771066

theorem largest_prime_divisor_test_range_1000_1100 (n : ℕ) (h1 : 1000 ≤ n ∧ n ≤ 1100) :
  ∃ p, Prime p ∧ p ≤ Int.sqrt 1100 ∧ ∀ q, Prime q ∧ q ≤ Int.sqrt 1100 → q ≤ p :=
begin
  sorry
end

end largest_prime_divisor_test_range_1000_1100_l771_771066


namespace all_meet_standard_l771_771875

variables (A B C : Event) (P : ProbabilitySpace)

axiom P_A : P (A) = 0.8
axiom P_B : P (B) = 0.6
axiom P_C : P (C) = 0.5
axiom indep_ABC : IndepEvents [A, B, C] P

theorem all_meet_standard : P (A ∩ B ∩ C) = 0.24 := 
by {
  sorry
}

end all_meet_standard_l771_771875


namespace collinear_O1_O_O2_l771_771144

variables {A B C D O O1 O2 : Type*}
variables [EuclideanGeometry A B C D]
variables [trapezoid ABCD A B C D]

-- The conditions
axiom base_parallel : Parallel AB CD
axiom square_on_AB : is_square_on_base AB
axiom square_on_CD : is_square_on_base CD
axiom intersect_point_O : Intersect AC BD O
axiom center_square_AB : center_of_square AB O1
axiom center_square_CD : center_of_square CD O2

theorem collinear_O1_O_O2 : Collinear O1 O O2 :=
sorry

end collinear_O1_O_O2_l771_771144


namespace probability_of_first_hearts_and_second_clubs_l771_771669

noncomputable def probability_first_hearts_second_clubs : ℚ :=
  let total_cards := 52
  let hearts_count := 13
  let clubs_count := 13
  let probability_first_hearts := hearts_count / total_cards
  let probability_second_clubs_given_first_hearts := clubs_count / (total_cards - 1)
  probability_first_hearts * probability_second_clubs_given_first_hearts

theorem probability_of_first_hearts_and_second_clubs :
  probability_first_hearts_second_clubs = 13 / 204 :=
by
  sorry

end probability_of_first_hearts_and_second_clubs_l771_771669


namespace largest_interior_angle_l771_771631

theorem largest_interior_angle (x : ℝ) (h₀ : 50 + 55 + x = 180) : 
  max 50 (max 55 x) = 75 := by
  sorry

end largest_interior_angle_l771_771631


namespace y_equals_6_l771_771110

def point (α : Type) := (x y : α)

variables (A B C : point ℝ)
variable (x y : ℕ)

-- Conditions
-- 1. Geometry of the points
def A := point.mk 43 (43 * Real.sqrt 3)
def B := point.mk 86 0
def C := point.mk 0 0

-- 2. Boy swims from A to B directly
def direct_swim := point.mk (A.x - ↑x) (A.y - (x * Real.sqrt 3))

-- 3. Turn westward swim
def westward_swim := point.mk (A.x - ↑x - ↑y) (A.y - (x * Real.sqrt 3))

-- 4. Given equation
noncomputable def x_solution (x : ℕ) (y : ℕ) : Prop := ↑x * (344 - 3 * ↑x) = (2 * ↑y + ↑x)^2

theorem y_equals_6 : ∃ x, x_solution x 6 :=
by
  sorry

end y_equals_6_l771_771110


namespace term_of_sequence_l771_771888

def S (n : ℕ) : ℚ := n^2 + 2/3

def a (n : ℕ) : ℚ :=
  if n = 1 then 5/3
  else 2 * n - 1

theorem term_of_sequence (n : ℕ) : a n = 
  if n = 1 then S n 
  else S n - S (n - 1) :=
by
  sorry

end term_of_sequence_l771_771888


namespace integral_half_circle_l771_771638

theorem integral_half_circle :
  ∫ x in (-3 : ℝ)..3, sqrt (9 - x^2) = (9 * Real.pi / 2) :=
by
  sorry

end integral_half_circle_l771_771638


namespace total_amount_divided_l771_771725

theorem total_amount_divided (P1 : ℝ) (r1 : ℝ) (r2 : ℝ) (interest : ℝ) (T : ℝ) :
  P1 = 1550 →
  r1 = 0.03 →
  r2 = 0.05 →
  interest = 144 →
  (P1 * r1 + (T - P1) * r2 = interest) → T = 3500 :=
by
  intros hP1 hr1 hr2 hint htotal
  sorry

end total_amount_divided_l771_771725


namespace delivery_fee_is_twenty_l771_771149

-- Given conditions in the problem
def sandwich_price : ℝ := 5
def sandwich_quantity : ℕ := 18
def tip_percentage : ℝ := 0.10
def total_received : ℝ := 121

-- Define the cost of sandwiches
def cost_of_sandwiches : ℝ := sandwich_quantity * sandwich_price

-- Define the delivery fee
noncomputable def delivery_fee : ℝ :=
  sorry

-- Define the tip
noncomputable def tip (delivery_fee : ℝ) : ℝ :=
  tip_percentage * (cost_of_sandwiches + delivery_fee)

-- Define the total amount Preston received
def total (delivery_fee : ℝ) : ℝ :=
  cost_of_sandwiches + delivery_fee + tip delivery_fee

-- The theorem to prove
theorem delivery_fee_is_twenty :
  ∃ D : ℝ, total D = total_received ∧ D = 20 :=
sorry

end delivery_fee_is_twenty_l771_771149


namespace regular_even_gon_divisible_into_rhombuses_l771_771593

/--
Prove that any regular 2n-gon can be divided into rhombuses.
-/
theorem regular_even_gon_divisible_into_rhombuses (n : ℕ) (h : 2 ≤ n) :
  ∃ (division : (ℕ × ℕ) → Prop), (is_regular_2n_gon n) ∧ (divides_into_rhombuses n) :=
sorry

end regular_even_gon_divisible_into_rhombuses_l771_771593


namespace find_a_b_extremum_l771_771174

def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

theorem find_a_b_extremum (a b : ℝ) :
  (∀ x, deriv (f x a b) x = 3 * x^2 - 2 * a * x - b) →
  (deriv (f 1 a b) 1 = 0) →
  (f 1 a b = 10) →
  (a = -4 ∧ b = 11) :=
by
  sorry

end find_a_b_extremum_l771_771174


namespace nancy_homework_problems_l771_771143

theorem nancy_homework_problems (finished : ℕ) (pages_left : ℕ) (problems_per_page : ℕ) : 
  finished = 47 → 
  pages_left = 6 → 
  problems_per_page = 9 → 
  (finished + pages_left * problems_per_page) = 101 :=
by
  intros h_finished h_pages_left h_problems_per_page
  rw [h_finished, h_pages_left, h_problems_per_page]
  norm_num
  sorry

end nancy_homework_problems_l771_771143


namespace two_points_determine_straight_line_l771_771255

-- Let's define two points in a 2D Euclidean space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Let's assert that given two unique points, there exists a unique straight line that passes through both
theorem two_points_determine_straight_line (A B : Point) (h : A ≠ B) : ∃! l : ℝ × ℝ, ∀ p : ℝ × ℝ, p ∈ l ↔
  ∃ k : ℝ, (p.1 = A.x + k * (B.x - A.x) ∧ p.2 = A.y + k * (B.y - A.y)) :=
sorry

end two_points_determine_straight_line_l771_771255


namespace probability_3_heads_is_40_243_l771_771731

noncomputable def probability_of_heads (n k : ℕ) (r : ℚ) : ℚ :=
(n.choose k) * r^k * (1 - r)^(n - k)

theorem probability_3_heads_is_40_243 (r : ℚ) (hr : r = 1 / 3) :
  let p_3_heads := probability_of_heads 5 3 r in
  ∃ (i j : ℕ), p_3_heads = i / j ∧ i + j = 283 :=
by
  sorry

end probability_3_heads_is_40_243_l771_771731


namespace betty_grows_average_parsnips_l771_771332

def betty_parsnip_harvest 
    (box_capacity : ℕ) 
    (full_fraction : ℚ) 
    (half_full_fraction : ℚ) 
    (average_boxes : ℕ) : ℕ :=
  let full_boxes := (full_fraction * average_boxes) in
  let half_full_boxes := (half_full_fraction * average_boxes) in
  let parsnips_in_full_boxes := full_boxes * box_capacity in
  let parsnips_in_half_full_boxes := half_full_boxes * (box_capacity / 2) in
  parsnips_in_full_boxes + parsnips_in_half_full_boxes

theorem betty_grows_average_parsnips
    (box_capacity : ℕ := 20) 
    (full_fraction : ℚ := 3/4) 
    (half_full_fraction : ℚ := 1/4) 
    (average_boxes : ℕ := 20) : 
  betty_parsnip_harvest box_capacity full_fraction half_full_fraction average_boxes = 350 :=
by 
  sorry

end betty_grows_average_parsnips_l771_771332


namespace packs_of_noodles_l771_771992

theorem packs_of_noodles (total_packs cookies_packs : ℕ) (h1 : total_packs = 28) (h2 : cookies_packs = 12) :
  total_packs - cookies_packs = 16 :=
by {
  -- Statement of what the theorem needs to prove
  rw [h1, h2],
  sorry
}

end packs_of_noodles_l771_771992


namespace arg_cube_div_eq_pi_l771_771007

noncomputable def arg_cube_div (z1 z2 : ℂ) (h1 : abs z1 = 3) (h2 : abs z2 = 5) (h3 : abs (z1 + z2) = 7) : ℝ :=
  if h1' : abs z1 = 3 ∧ abs z2 = 5 ∧ abs (z1 + z2) = 7 then
    sorry

theorem arg_cube_div_eq_pi (z1 z2 : ℂ) (h1 : abs z1 = 3) (h2 : abs z2 = 5) (h3 : abs (z1 + z2) = 7) :
  arg_cube_div z1 z2 h1 h2 h3 = π :=
  sorry

end arg_cube_div_eq_pi_l771_771007


namespace range_of_a_derivative_midpoint_l771_771884

noncomputable def f (x a : ℝ) : ℝ := a * (x + 1) ^ 2 + x * Real.exp x
noncomputable def f' (x a : ℝ) : ℝ := (x + 1) * Real.exp x + 2 * a * (x + 1)

theorem range_of_a (a : ℝ) :
  (∃ x, ∃ c, f' c a = 0 ∧ (∃ x1, ∃ x2, c ≠ x1 ∧ c ≠ x2) → a ∈ (-∞, -1 / (2 * Real.exp 1)) ∪ (-1 / (2 * Real.exp 1), 0)) :=
sorry

theorem derivative_midpoint (a x1 x2 : ℝ) (h_a_pos : a > 0) (h_zero_points : f x1 a = 0 ∧ f x2 a = 0) :
  f' ((x1 + x2) / 2) a < 0 :=
sorry

end range_of_a_derivative_midpoint_l771_771884


namespace remaining_area_l771_771768

-- Given a regular hexagon and a rhombus composed of two equilateral triangles.
-- Hexagon area is 135 square centimeters.

variable (hexagon_area : ℝ) (rhombus_area : ℝ)
variable (is_regular_hexagon : Prop) (is_composed_of_two_equilateral_triangles : Prop)

-- The conditions
def correct_hexagon_area := hexagon_area = 135
def rhombus_is_composed := is_composed_of_two_equilateral_triangles = true
def hexagon_is_regular := is_regular_hexagon = true

-- Goal: Remaining area after cutting out the rhombus should be 75 square centimeters
theorem remaining_area : 
  correct_hexagon_area hexagon_area →
  hexagon_is_regular is_regular_hexagon →
  rhombus_is_composed is_composed_of_two_equilateral_triangles →
  hexagon_area - rhombus_area = 75 :=
by
  sorry

end remaining_area_l771_771768


namespace find_number_l771_771526

theorem find_number (x n : ℤ) (h1 : |x| = 9 * x - n) (h2 : x = 2) : n = 16 := by 
  sorry

end find_number_l771_771526


namespace sum_largest_next_largest_l771_771197

theorem sum_largest_next_largest (numbers : List ℕ) (h : numbers = [10, 11, 12, 13, 14]) :
  (List.maximum numbers).getOrElse 0 + (List.maximum (numbers.erase (List.maximum numbers).getOrElse 0)).getOrElse 0 = 27 :=
by
  sorry

end sum_largest_next_largest_l771_771197


namespace value_of_f_3_div_2_l771_771478

noncomputable def f : ℝ → ℝ
| x => if 0 < x ∧ x < 1
       then 1 - 4 * x
       else if 1 ≤ x
       then 2 * f (x / 2)
       else 0  -- This case is actually not needed since x > 0 is given

theorem value_of_f_3_div_2 :
  f (3 / 2) = -4 := by
  sorry

end value_of_f_3_div_2_l771_771478


namespace appropriate_grouping_43_neg78_27_neg52_l771_771700

theorem appropriate_grouping_43_neg78_27_neg52 :
  let a := 43
  let b := -78
  let c := 27
  let d := -52
  (a + c) + (b + d) = -60 :=
by
  let a := 43
  let b := -78
  let c := 27
  let d := -52
  sorry

end appropriate_grouping_43_neg78_27_neg52_l771_771700


namespace geometric_sequence_ratio_l771_771842

theorem geometric_sequence_ratio
  (a₁ : ℝ) (q : ℝ) (hq : q ≠ 1)
  (S : ℕ → ℝ)
  (hS₃ : S 3 = a₁ * (1 - q^3) / (1 - q))
  (hS₆ : S 6 = a₁ * (1 - q^6) / (1 - q))
  (hS₃_val : S 3 = 2)
  (hS₆_val : S 6 = 18) :
  S 10 / S 5 = 1 + 2^(1/3) + 2^(2/3) :=
sorry

end geometric_sequence_ratio_l771_771842


namespace sum_ratios_l771_771864

variable (a b d : ℕ)

def A_n (a b d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def arithmetic_sum (a n d : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem sum_ratios (k : ℕ) (h1 : 2 * (a + d) = 7 * k) (h2 : 4 * (a + 3 * d) = 6 * k) :
  arithmetic_sum a 7 d / arithmetic_sum a 3 d = 2 / 1 :=
by
  sorry

end sum_ratios_l771_771864


namespace emily_has_28_beads_l771_771357

def beads_per_necklace : ℕ := 7
def necklaces : ℕ := 4

def total_beads : ℕ := necklaces * beads_per_necklace

theorem emily_has_28_beads : total_beads = 28 := by
  sorry

end emily_has_28_beads_l771_771357


namespace seven_books_cost_l771_771204

-- Given condition: Three identical books cost $45
def three_books_cost (cost_per_book : ℤ) := 3 * cost_per_book = 45

-- Question to prove: The cost of seven identical books is $105
theorem seven_books_cost (cost_per_book : ℤ) (h : three_books_cost cost_per_book) : 7 * cost_per_book = 105 := 
sorry

end seven_books_cost_l771_771204


namespace no_such_sequence_exists_l771_771574

theorem no_such_sequence_exists (n : ℕ) (h : n ≥ 2) (a : fin n → ℕ) (h_pos : ∀ i, 0 < a i) (h_neq : ∃ i j, i ≠ j ∧ a i ≠ a j) :
  ¬ (∀ i j, ∃ k1 k2 ..., k_m, (a i + a j) / 2 = real.sqrt (a k1 * ... * a k_m)) := 
sorry

end no_such_sequence_exists_l771_771574


namespace sum_b_n_formula_l771_771846

noncomputable def a_n (n : ℕ) : ℕ :=
if n = 1 then 4 else n * 2^(n + 1)

noncomputable def b_n (n : ℕ) : ℤ :=
((-1 : ℤ)^n * a_n n) / 2

noncomputable def S_n (n : ℕ) : ℤ :=
∑ k in Finset.range (n + 1), b_n k

theorem sum_b_n_formula (n : ℕ) : 
  S_n n = -(((3 * n + 1) * (-2)^(n + 1) + 2) / 9) :=
sorry

end sum_b_n_formula_l771_771846


namespace x_less_than_2_necessary_not_sufficient_x_less_than_2_is_necessary_not_sufficient_l771_771354

theorem x_less_than_2_necessary_not_sufficient (x : ℝ) :
  (x^2 - 3 * x + 2 < 0) ↔ (1 < x ∧ x < 2) := sorry

theorem x_less_than_2_is_necessary_not_sufficient : 
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x < 2) ∧ 
  (¬ ∀ x : ℝ, x < 2 → x^2 - 3*x + 2 < 0) := sorry

end x_less_than_2_necessary_not_sufficient_x_less_than_2_is_necessary_not_sufficient_l771_771354


namespace only_quadratic_function_must_be_C_l771_771258

theorem only_quadratic_function_must_be_C :
  ∀ (A B C D : Type) (fA : ℝ → ℝ) (fB : ℝ → ℝ) (fC : ℝ → ℝ) (fD : ℝ → ℝ),
    (fA = λ x, 2 * x - 5) →
    (fB = λ x, ∃ a b c, fB x = a * x^2 + b * x + c) →
    (fC = λ t, t^2 / 2) →
    (fD = λ x, x^2 + 1 / x) →
    (∃ c, fC = λ t, c * t^2) ∧ (¬ ∃ a b c, fA = λ x, a * x^2 + b * x + c) ∧ 
    (¬ ∃ a b c, fD = λ x, a * x^2 + b * x + c) :=
by {
  -- The actual proof should distinguish the quadratic nature of fC vs others being not strictly quadratic.
  sorry
}

end only_quadratic_function_must_be_C_l771_771258


namespace power_mod_seven_pow_eight_mod_100_l771_771246

theorem power_mod {a b m : ℕ} (h: a ≡ b [MOD m]) (n : ℕ) : (a ^ n) ≡ (b ^ n) [MOD m] :=
by sorry

theorem seven_pow_eight_mod_100 : (7 ^ 8) % 100 = 1 :=
by {
  have h1 : (7 ^ 4) % 100 = 1,
  {
    calc (7 ^ 4) % 100
         = (7 * 7 * 7 * 7) % 100 : by rw pow_succ
     ... = 49 * 49 % 100 : by rw mul_assoc
     ... = 2401 % 100 : sorry
     ... = 1 : sorry,
  },
  calc (7 ^ 8) % 100
       = (7 ^ (4 * 2)) % 100 : by rw pow_mul
   ... = (1 ^ 2) % 100 : by rw h1
   ... = 1 : sorry
}

end power_mod_seven_pow_eight_mod_100_l771_771246


namespace distance_between_points_l771_771227

theorem distance_between_points :
  let x1 := -3
  let y1 := 5
  let x2 := 4
  let y2 := -9
  sqrt ((x2 - x1)^2 + (y2 - y1)^2) = sqrt 245 :=
by
  sorry

end distance_between_points_l771_771227


namespace problem1_problem2_l771_771435

open Real Complex

noncomputable def z (m : ℝ) : ℂ :=
  ((m * (m - 1)) / (m + 1)) + (m ^ 2 + 2 * m - 3) * Complex.I

theorem problem1 (m : ℝ) (h : z m.im = 0) : m = 0 := by
  sorry

theorem problem2 (m : ℝ) (h : (z m).re + (z m).im + 3 = 0) : m = 0 ∨ m = -2 + Real.sqrt 3 ∨ m = -2 - Real.sqrt 3 := by
  sorry

end problem1_problem2_l771_771435


namespace cos_beta_unit_circle_rotation_l771_771021

theorem cos_beta_unit_circle_rotation 
  (alpha beta : ℝ)
  (h1 : (0, 0) → α)
  (h2 : (1, 0) → α)
  (h3 : ∃ P: ℝ × ℝ, P = (-3/5, -4/5) ∧ P ∈ { Q : ℝ × ℝ | Q.1 ^ 2 + Q.2 ^ 2 = 1 })
  (h4 : β = α + π / 2) :
  cos β = 4 / 5 :=
sorry

end cos_beta_unit_circle_rotation_l771_771021


namespace inequality_proof_l771_771440

variable (m : ℕ) (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1)

theorem inequality_proof :
    (m > 0) →
    (x^m / ((1 + y) * (1 + z)) + y^m / ((1 + x) * (1 + z)) + z^m / ((1 + x) * (1 + y)) >= 3/4) :=
by
  intro hm_pos
  -- Proof skipped
  sorry

end inequality_proof_l771_771440


namespace area_ratio_triangle_l771_771115

variables {A B C P : Type} 
variables {PA PB PC : A → B → C → P → Vector}
variables (a b c p : Vector)

/- Given Conditions -/
def condition1 : PA A P + 3 * PB B P + 2 * PC C P = 0 := sorry

theorem area_ratio_triangle (h : condition1 PA PB PC A B C P) :
  area A B C / area A P B = 3 := 
sorry

end area_ratio_triangle_l771_771115


namespace probability_heart_then_club_l771_771659

theorem probability_heart_then_club :
  let P_heart := 13 / 52
  let P_club_given_heart := 13 / 51
  P_heart * P_club_given_heart = 13 / 204 := 
by
  let P_heart := (13 : ℚ) / 52
  let P_club_given_heart := (13 : ℚ) / 51
  have h : P_heart * P_club_given_heart = 13 / 204 := by
    calc
      P_heart * P_club_given_heart
        = (13 / 52) * (13 / 51) : rfl
    ... = (13 * 13) / (52 * 51) : by rw [mul_div_mul_comm]
    ... = 169 / 2652 : rfl
    ... = 13 / 204 : by norm_num
  exact h

end probability_heart_then_club_l771_771659


namespace kim_change_is_5_l771_771106

variable (meal_cost : ℝ) (drink_cost : ℝ) (tip_percent : ℝ) (payment : ℝ)

theorem kim_change_is_5 (h_meal_cost : meal_cost = 10)
                        (h_drink_cost : drink_cost = 2.5)
                        (h_tip_percent : tip_percent = 0.20)
                        (h_payment : payment = 20) :
  let total_cost := meal_cost + drink_cost
      tip_amount := tip_percent * total_cost
      total_with_tip := total_cost + tip_amount
      change := payment - total_with_tip
  in change = 5 := 
by
  sorry

end kim_change_is_5_l771_771106


namespace sum_even_translation_odd_l771_771927

-- Given definitions: even function and translation resulting in an odd function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (x) = f (-x)
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (x) = -f (-x)
def translated (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop := ∀ x, g x = f (x - 1)

-- Define the specific problem conditions
variable (f : ℝ → ℝ)
hypothesis (h₁ : is_even_function f)
hypothesis (h₂ : is_odd_function (λ x, f (x-1)))

-- The theorem statement
theorem sum_even_translation_odd :
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 = 0 :=
sorry

end sum_even_translation_odd_l771_771927


namespace cos_sum_eq_neg_ratio_l771_771916

theorem cos_sum_eq_neg_ratio (γ δ : ℝ) 
  (hγ: Complex.exp (Complex.I * γ) = 4 / 5 + 3 / 5 * Complex.I) 
  (hδ: Complex.exp (Complex.I * δ) = -5 / 13 + 12 / 13 * Complex.I) :
  Real.cos (γ + δ) = -56 / 65 :=
  sorry

end cos_sum_eq_neg_ratio_l771_771916


namespace power_simplification_l771_771688

theorem power_simplification :
  (1 / ((-5) ^ 4) ^ 2) * (-5) ^ 9 = -5 :=
by 
  sorry

end power_simplification_l771_771688


namespace num_colorings_l771_771650

def color := {red, blue, green}

def adjacency (i j : Fin 3 × Fin 3) : Prop :=
  (i.1 = j.1 ∧ (i.2 = j.2 + 1 ∨ i.2 = j.2 - 1)) ∨
  (i.2 = j.2 ∧ (i.1 = j.1 + 1 ∨ i.1 = j.1 - 1))

def valid_coloring (grid : Fin 3 × Fin 3 → color) : Prop :=
  ∀ i j, adjacency i j → grid i ≠ grid j

noncomputable def count_valid_colorings : Nat :=
  Finset.card {c : Fin 3 × Fin 3 → color | valid_coloring c}

theorem num_colorings : count_valid_colorings = 3 :=
sorry

end num_colorings_l771_771650


namespace aleesia_lost_each_week_l771_771755

-- Definitions based on conditions
variable (A : ℝ)
variable (total_lost_by_friends : ℝ := 35)
variable (total_lost_by_alexei : ℝ := 2.5 * 8)
variable (total_lost_by_aleesia : ℝ := total_lost_by_friends - total_lost_by_alexei)
variable (weeks_aleesia : ℝ := 10)

-- Theorem statement based on the problem
theorem aleesia_lost_each_week :
  A = total_lost_by_aleesia / weeks_aleesia :=
begin
  sorry
end

end aleesia_lost_each_week_l771_771755


namespace range_of_k_l771_771450

-- Define the sequence a_n
def a : ℕ → ℕ
| 0       := 2
| (n + 1) := 3 * (a n) + 2

-- Define the property of k
def property_k (k : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → k * (a n + 1) ≥ 2 * n - 3

-- State the theorem for the range of k
theorem range_of_k : ∃ k : ℝ, k = 1 / 9 :=
begin
  use (1 / 9),
  sorry -- Proof not required
end

end range_of_k_l771_771450


namespace identify_7_real_coins_l771_771096

theorem identify_7_real_coins (coins : Fin 63 → ℝ) (fakes : Finset (Fin 63)) (h_fakes_count : fakes.card = 7) (real_weight fake_weight : ℝ)
  (h_weights : ∀ i, i ∉ fakes → coins i = real_weight) (h_fake_weights : ∀ i, i ∈ fakes → coins i = fake_weight) (h_lighter : fake_weight < real_weight) :
  ∃ real_coins : Finset (Fin 63), real_coins.card = 7 ∧ (∀ i, i ∈ real_coins → coins i = real_weight) :=
sorry

end identify_7_real_coins_l771_771096


namespace gum_pieces_bought_correct_l771_771315

-- Define initial number of gum pieces
def initial_gum_pieces : ℕ := 10

-- Define number of friends Adrianna gave gum to
def friends_given_gum : ℕ := 11

-- Define the number of pieces Adrianna has left
def remaining_gum_pieces : ℕ := 2

-- Define a function to calculate the number of gum pieces Adrianna bought at the store
def gum_pieces_bought (initial_gum : ℕ) (given_gum : ℕ) (remaining_gum : ℕ) : ℕ :=
  (given_gum + remaining_gum) - initial_gum

-- Now state the theorem to prove the number of pieces bought is 3
theorem gum_pieces_bought_correct : 
  gum_pieces_bought initial_gum_pieces friends_given_gum remaining_gum_pieces = 3 :=
by
  sorry

end gum_pieces_bought_correct_l771_771315


namespace banana_bread_proof_l771_771600

theorem banana_bread_proof (total_loaves : ℕ) (total_bananas : ℕ) (h : total_loaves = 99 ∧ total_bananas = 33) :
  total_loaves / total_bananas = 3 :=
by
  cases h with
  | intro h_loaves h_bananas =>
    rw [h_loaves, h_bananas]
    norm_num
    exact rfl

end banana_bread_proof_l771_771600


namespace adult_ticket_cost_l771_771210

variable (A : ℝ)

theorem adult_ticket_cost :
  (20 * 6) + (12 * A) = 216 → A = 8 :=
by
  intro h
  sorry

end adult_ticket_cost_l771_771210


namespace probability_of_first_hearts_and_second_clubs_l771_771671

noncomputable def probability_first_hearts_second_clubs : ℚ :=
  let total_cards := 52
  let hearts_count := 13
  let clubs_count := 13
  let probability_first_hearts := hearts_count / total_cards
  let probability_second_clubs_given_first_hearts := clubs_count / (total_cards - 1)
  probability_first_hearts * probability_second_clubs_given_first_hearts

theorem probability_of_first_hearts_and_second_clubs :
  probability_first_hearts_second_clubs = 13 / 204 :=
by
  sorry

end probability_of_first_hearts_and_second_clubs_l771_771671


namespace evaluate_neg_64_exp_4_over_3_l771_771368

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l771_771368


namespace island_connectivity_after_years_l771_771444

noncomputable def ferry_network (n : ℕ) (h : n ≥ 3) : Prop :=
  ∀ G : SimpleGraph (Fin n),
    (∀ s t : Finset (Fin n), s ∪ t = Finset.univ → s ≠ ∅ → t ≠ ∅ → ∃ (x : Fin n), G.exists_edge (x, G.neighbors x)) →
    (∃ x : Fin n, ∀ y : Fin n, y ≠ x → x ∈ G.neighbors y)

theorem island_connectivity_after_years (n : ℕ) (h : n ≥ 3) :
  ferry_network n h :=
begin
  sorry
end

end island_connectivity_after_years_l771_771444


namespace solve_for_x_l771_771257

theorem solve_for_x (x : ℝ) (h : (sqrt x)^3 = 216) : x = 36 :=
by
    -- proof is omitted
    sorry

end solve_for_x_l771_771257


namespace poly_product_even_but_not_all_div_by_4_l771_771153

noncomputable def poly (R : Type*) := R[X]

variables {R : Type*} [CommRing R] (A B : poly ℤ)

theorem poly_product_even_but_not_all_div_by_4
  (h1 : ∀ k, ((A * B).coeff k % 2 = 0))
  (h2 : ∃ k, ((A * B).coeff k % 4 ≠ 0)) :
  (∀ i, (A.coeff i % 2 = 0) ∧ (∃ j, B.coeff j % 2 ≠ 0)) ∨
  ((∃ i, A.coeff i % 2 ≠ 0) ∧ ∀ j, B.coeff j % 2 = 0) :=
sorry

end poly_product_even_but_not_all_div_by_4_l771_771153


namespace count_valid_n_in_range_l771_771499

theorem count_valid_n_in_range : 
  (set.count {n : ℕ | 1000 < n^2 ∧ n^2 < 2000}) = 13 :=
by 
  sorry

end count_valid_n_in_range_l771_771499


namespace cubes_end_same_digits_l771_771056

theorem cubes_end_same_digits (a b : ℕ) (h : a % 1000 = b % 1000) : (a^3) % 1000 = (b^3) % 1000 := by
  sorry

end cubes_end_same_digits_l771_771056


namespace sets_count_eq_seven_l771_771159

theorem sets_count_eq_seven : 
  let U := {1, 2, 3, 4, 5}
  let subsets_of_U := {A | A ⊆ U}
  let cond := {A | {1, 2} ⊆ A ∧ A ⊂ U}
  cond.card = 7 := 
sorry

end sets_count_eq_seven_l771_771159


namespace max_value_expression_l771_771814

theorem max_value_expression : 
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 18 → sqrt (35 - x) + sqrt x + sqrt (18 - x) ≤ sqrt 35 + sqrt 18 := 
by 
  intro x
  intro hx
  sorry

end max_value_expression_l771_771814


namespace total_fruits_sold_fruit_vendor_sold_total_fruits_l771_771582

def dozens_to_fruits (dozens : ℝ) : ℕ :=
  (dozens * 12).floor.to_nat

theorem total_fruits_sold (
  morning_lemons : ℝ := 3.5, afternoon_lemons : ℝ := 6.15, returns_lemons : ℝ := 0.75,
  morning_avocados : ℝ := 6.25, afternoon_avocados : ℝ := 4, returns_avocados : ℝ := 1.25,
  morning_oranges : ℝ := 5.5, afternoon_oranges : ℝ := 7.6,
  morning_apples : ℝ := 4.87, afternoon_apples : ℝ := 2.98,
  morning_bananas : ℝ := 2.68, afternoon_bananas : ℝ := 3
) : ℕ :=
  let total_lemons := morning_lemons + afternoon_lemons - returns_lemons
  let total_avocados := morning_avocados + afternoon_avocados - returns_avocados
  let total_oranges := morning_oranges + afternoon_oranges
  let total_apples := morning_apples + afternoon_apples
  let total_bananas := morning_bananas + afternoon_bananas
  let total_fruits := (dozens_to_fruits total_lemons) +
                      (dozens_to_fruits total_avocados) +
                      (dozens_to_fruits total_oranges) +
                      (dozens_to_fruits total_apples) +
                      (dozens_to_fruits total_bananas)
  total_fruits

theorem fruit_vendor_sold_total_fruits :
  total_fruits_sold = 533 := 
by
  -- Proof steps will go here
  sorry

end total_fruits_sold_fruit_vendor_sold_total_fruits_l771_771582


namespace rem_frac_eq_l771_771770

theorem rem_frac_eq :
  (x y : ℚ) (hx : x = -5/6) (hy : y = 3/4) :
  rat.floor (x / y) = -2 → 
  x - y * (rat.floor (x / y)) = 2/3 :=
by
  intros
  sorry

end rem_frac_eq_l771_771770


namespace vertex_closest_point_l771_771523

theorem vertex_closest_point (a : ℝ) (x y : ℝ) :
  (x^2 = 2 * y) ∧ (y ≥ 0) ∧ ((y^2 + 2 * (1 - a) * y + a^2) ≤ 0) → a ≤ 1 :=
by 
  sorry

end vertex_closest_point_l771_771523


namespace first_team_more_points_l771_771545

/-
Conditions:
  - Beth scored 12 points.
  - Jan scored 10 points.
  - Judy scored 8 points.
  - Angel scored 11 points.
Question:
  - How many more points did the first team get than the second team?
Prove that the first team scored 3 points more than the second team.
-/

theorem first_team_more_points
  (Beth_score : ℕ)
  (Jan_score : ℕ)
  (Judy_score : ℕ)
  (Angel_score : ℕ)
  (First_team_total : ℕ := Beth_score + Jan_score)
  (Second_team_total : ℕ := Judy_score + Angel_score)
  (Beth_score_val : Beth_score = 12)
  (Jan_score_val : Jan_score = 10)
  (Judy_score_val : Judy_score = 8)
  (Angel_score_val : Angel_score = 11)
  : First_team_total - Second_team_total = 3 := by
  sorry

end first_team_more_points_l771_771545


namespace possible_fourth_face_l771_771922

noncomputable def is_right_angled_triangle : Type := sorry
noncomputable def is_acute_angled_triangle : Type := sorry
noncomputable def is_obtuse_angled_triangle : Type := sorry
noncomputable def is_isosceles_triangle : Type := sorry
noncomputable def is_isosceles_right_angled_triangle : Type := sorry

def tetrahedron (faces : List (Type)) : Prop :=
  faces.length = 4 ∧ faces.count is_right_angled_triangle = 3

theorem possible_fourth_face (faces : List Type) :
  tetrahedron faces →
  ∃ fourth_face : Type, fourth_face ∈ faces ∧
    (fourth_face = is_right_angled_triangle ∨
     fourth_face = is_acute_angled_triangle ∨
     fourth_face = is_isosceles_triangle ∨
     fourth_face = is_isosceles_right_angled_triangle) ∧
    ¬ (fourth_face = is_obtuse_angled_triangle) :=
by
  sorry

end possible_fourth_face_l771_771922


namespace triangle_sin_C_triangle_area_l771_771527

noncomputable def sin_C (A B: ℝ) (cos_A: ℝ) (b: ℝ) : ℝ :=
  sin (2 * π / 3 - A)

theorem triangle_sin_C (A : ℝ) (B: ℝ := π / 3) (cos_A: ℝ := 4 / 5) (b: ℝ := √3) :
  sin_C A B cos_A b = (3 + 4 * √3) / 10 :=
by 
  sorry

noncomputable def area_triangle (a b: ℝ) (sin_C: ℝ) : ℝ :=
  1 / 2 * a * b * sin_C

theorem triangle_area (A : ℝ) (B: ℝ := π / 3) (cos_A: ℝ := 4 / 5) (b: ℝ := √3) (sin_C := (3 + 4 * √3) / 10) :
  area_triangle (6 / 5) b sin_C = (36 + 9 * √3) / 50 :=
by 
  sorry

end triangle_sin_C_triangle_area_l771_771527


namespace cosine_value_of_skew_lines_l771_771857

-- Define the triangular prism with equal side and base edges
structure TriangularPrism (A B C A1 B1 C1 : Type*) :=
  (side_length : ℝ)
  (base_length : ℝ)
  (side_edges_equal : ∀ (x y : Type*), x ≠ y → dist x y = side_length)
  (base_edges_equal : ∀ (x : Type*), x ≠ A → x ≠ B → x ≠ C → dist x A = base_length ∧ dist x B = base_length ∧ dist x C = base_length)

-- Define the midpoint condition of the projection of A1 on BC
structure MidpointProjection (A B C A1 : Type*) :=
  (midpoint : Type*)
  (is_midpoint : dist midpoint B = dist midpoint C)
  (projection_condition : ∀ (x : Type*), dist x midpoint = dist x A1 / 2)

-- Define the problem in Lean
theorem cosine_value_of_skew_lines {A B C A1 B1 C1 : Type*} (prism : TriangularPrism A B C A1 B1 C1)
  (midpoint_projection : MidpointProjection A B C A1) :
  ∃ (cos_value : ℝ), cos_value = 3 / 4 :=
  sorry

end cosine_value_of_skew_lines_l771_771857


namespace cleaning_task_sequences_correct_l771_771071

section ChemistryClass

-- Total number of students
def total_students : ℕ := 15

-- Number of classes in a week
def classes_per_week : ℕ := 5

-- Calculate the number of valid sequences of task assignments
def num_valid_sequences : ℕ := total_students * (total_students - 1) * (total_students - 2) * (total_students - 3) * (total_students - 4)

theorem cleaning_task_sequences_correct :
  num_valid_sequences = 360360 :=
by
  unfold num_valid_sequences
  norm_num
  sorry

end ChemistryClass

end cleaning_task_sequences_correct_l771_771071


namespace determine_constants_l771_771175

noncomputable def f : ℝ → ℝ := sorry -- Assume f is some arbitrary real-valued function

theorem determine_constants (h : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, h(x) = a * f(b * x) + c) ∧
  (∀ y, y = f(b * y / b)) ∧  -- Represents the property of the function transformation
  (b = 1 / 4) ∧
  (a = 1 / 3) ∧
  (c = 1) :=
begin
  split,
  sorry, -- Proof needed to show h(x) = a * f(b * x) + c
  split,
  sorry, -- To show the function property after transformation
  split,
  exact (1 / 4), -- b
  split,
  exact (1 / 3), -- a
  exact 1, -- c
end

end determine_constants_l771_771175


namespace angle_sum_and_relation_l771_771215

variable {A B : ℝ}

theorem angle_sum_and_relation (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end angle_sum_and_relation_l771_771215


namespace largest_3_digit_divisible_by_sum_of_digits_and_11_l771_771422

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem largest_3_digit_divisible_by_sum_of_digits_and_11 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
            (sum_of_digits n ∣ n) ∧ (11 ∣ sum_of_digits n) ∧
            ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (sum_of_digits m ∣ m) ∧ (11 ∣ sum_of_digits m) → m ≤ n :=
  ⟨990, by {
    -- Proof that 990 meets the conditions
    sorry, 
    -- Proof that no larger three-digit number meets the conditions
    sorry 
  }⟩

end largest_3_digit_divisible_by_sum_of_digits_and_11_l771_771422


namespace gibbs_free_energy_change_l771_771218

-- Given Gibbs free energies of formation
def Gf_NaOH := -381.1 -- kJ/mol
def Gf_Na2O := -378 -- kJ/mol
def Gf_H2O := -237 -- kJ/mol

-- Proof statement for the Gibbs free energy change at 298 K
theorem gibbs_free_energy_change : 
  2 * Gf_NaOH - (Gf_Na2O + Gf_H2O) = -147.2 := 
sorry

end gibbs_free_energy_change_l771_771218


namespace distance_between_peaks_correct_l771_771618

noncomputable def distance_between_peaks 
    (α β γ δ ε : ℝ) 
    (hα : α = 6 + 50/60 + 33/3600)
    (hβ : β = 7 + 25/60 + 52/3600)
    (hγ : γ = 5 + 24/60 + 52/3600)
    (hδ : δ = 5 + 55/60 + 36/3600)
    (hε : ε = 31 + 4/60 + 34/3600)
    (h_unit_conversion : ∀ θ : ℝ, θ * π / 180 = θ * 3.141592653589793 / 180)
    : ℝ :=
    let α_rad := α * π / 180,
    β_rad := β * π / 180,
    γ_rad := γ * π / 180,
    δ_rad := δ * π / 180,
    ε_rad := ε * π / 180,
    MA := 200 * cos α_rad * sin β_rad / sin (β_rad - α_rad),
    M1A := 200 * cos γ_rad * sin δ_rad / sin (δ_rad - γ_rad),
    MM1 := (MA^2 + M1A^2 - 2 * MA * M1A * cos ε_rad).sqrt,
    OM := MA * tan α_rad,
    O1M1 := M1A * tan γ_rad
in (MM1^2 + (OM - O1M1)^2).sqrt

theorem distance_between_peaks_correct :
    distance_between_peaks (6 + 50 / 60 + 33 / 3600) (7 + 25 / 60 + 52 / 3600) (5 + 24 / 60 + 52 / 3600) (5 + 55 / 60 + 36 / 3600) (31 + 4 / 60 + 34 / 3600) 
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (fun _ => by norm_cast; simp) = 1303 :=
sorry

end distance_between_peaks_correct_l771_771618


namespace increasing_infinite_sequence_l771_771320

-- Definitions of the sequences
def seq1 : ℕ → ℕ := λ n, if n < 20 then n + 1 else 0
def seq2 : ℕ → ℤ := λ n, -n
def seq3 : ℕ → ℕ
| 0 := 1
| 1 := 2
| 2 := 3
| 3 := 2
| 4 := 5
| n := n -- Assuming the nth term follows some pattern we can't capture directly
def seq4 : ℕ → ℤ := λ n, n - 1

-- Theorem proving that Sequence 4 is the only sequence that is both increasing and infinite
theorem increasing_infinite_sequence :
  (∀ (n : ℕ), seq1 n < seq1 (n + 1)) ∨ (∀ (n : ℕ), seq2 n < seq2 (n + 1)) ∨
  (∀ (n : ℕ), seq3 n < seq3 (n + 1)) ∨ (∀ (n : ℕ), seq4 n < seq4 (n + 1)) →
  (¬ (∀ (n : ℕ), seq1 n < seq1 (n + 1))) ∧ (¬ (∀ (n : ℕ), seq2 n < seq2 (n + 1))) ∧
  (¬ (∀ (n : ℕ), seq3 n < seq3 (n + 1))) ∧ (∀ (n : ℕ), seq4 n < seq4 (n + 1)) := 
sorry

end increasing_infinite_sequence_l771_771320


namespace range_of_real_a_l771_771877

theorem range_of_real_a (a : ℝ) : 
  (∃ x : ℝ, e^(2 * x) + e^x - a = 0) ↔ a ∈ set.Ioi 0 :=
begin
  sorry
end

end range_of_real_a_l771_771877


namespace fraction_phone_numbers_begin_with_9_end_with_0_l771_771769

theorem fraction_phone_numbers_begin_with_9_end_with_0 :
  (∑ d in Finset.Icc 9 9, ∑ n in Finset.range (10^6), 1) * 1 / 
  (∑ d in Finset.Icc 2 9, ∑ n in Finset.range (10^7), 1) = 1 / 80 :=
sorry

end fraction_phone_numbers_begin_with_9_end_with_0_l771_771769


namespace find_y_l771_771054

theorem find_y (x y : ℝ) (h₁ : 1.5 * x = 0.3 * y) (h₂ : x = 20) : y = 100 :=
sorry

end find_y_l771_771054


namespace eval_expression_correct_l771_771396

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l771_771396


namespace percentage_increase_correct_l771_771999

-- Define constants based on the conditions provided
def old_camera_cost : ℝ := 4000
def lens_discount : ℝ := 200
def lens_cost_before_discount : ℝ := 400
def total_paid : ℝ := 5400

-- Define the correct answer percentage increase
def percentage_increase : ℝ := 30

-- Calculate the lens cost after discount
def lens_cost_after_discount : ℝ := lens_cost_before_discount - lens_discount

-- Calculate the new camera cost
def new_camera_cost : ℝ := total_paid - lens_cost_after_discount

-- Calculate the increase in cost
def increase_in_cost : ℝ := new_camera_cost - old_camera_cost

-- Calculate the percentage increase in cost
def calculated_percentage_increase : ℝ := (increase_in_cost / old_camera_cost) * 100

-- The theorem stating the mathematically equivalent proof problem
theorem percentage_increase_correct :
  calculated_percentage_increase = percentage_increase := 
by 
  sorry

end percentage_increase_correct_l771_771999


namespace standard_deviation_is_one_l771_771620

def mean : ℝ := 10.5
def value : ℝ := 8.5

theorem standard_deviation_is_one (σ : ℝ) (h : value = mean - 2 * σ) : σ = 1 :=
by {
  sorry
}

end standard_deviation_is_one_l771_771620


namespace period_of_f_find_AC_and_area_l771_771041

-- Conditions:
def vec_a (x : ℝ) : ℝ × ℝ := (1/2, 1/2 * Real.sin x + (√3)/2 * Real.cos x)
def vec_b (x y : ℝ) : ℝ × ℝ := (1, y)
def parallel (x : ℝ) : Prop := ∃ y, vec_b x y = vec_a x

-- Question 1:
def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x, f (x + T) = f x

-- Answer 1:
theorem period_of_f : smallest_positive_period (λ x, 2 * Real.sin (x + π/3)) (2 * π) :=
sorry

-- Conditions for Question 2:
variables {A B C : ℝ}
variable (BC : ℝ)
variable (sin_B : ℝ)
def acute_triangle (A B C : ℝ) : Prop := A < π/2 ∧ B < π/2 ∧ C < π/2

-- Function value given:
def func_value (A : ℝ) : Prop := 2 * Real.sin (A) = √3

-- Law of Sines:
def law_of_sines (a b c A B : ℝ) : Prop := a / Real.sin A = b / Real.sin B

-- Answer 2:
theorem find_AC_and_area (h1 : acute_triangle A B C) (h2 : func_value (A - π/3)) 
  (BC : ℝ) (sin_B : ℝ) : 
  BC = √7 ∧ sin_B = √21 / 7 ∧
  AC = 2 ∧ 
  (1/2) * 3 * 2 * (√3) / 2 = (3 * √3) / 2 :=
sorry

end period_of_f_find_AC_and_area_l771_771041


namespace scientific_notation_of_10760000_l771_771754

theorem scientific_notation_of_10760000 : 
  (10760000 : ℝ) = 1.076 * 10^7 := 
sorry

end scientific_notation_of_10760000_l771_771754


namespace f_inv_sum_l771_771554

def f (x : ℝ) : ℝ :=
if x < 20 then x + 3 else 2 * x - 2

noncomputable def f_inv (y : ℝ) : ℝ :=
if y = 7 then 4 else if y = 46 then 24 else 0

theorem f_inv_sum : f_inv 7 + f_inv 46 = 28 :=
by
  have h1 : f_inv 7 = 4 := by rfl
  have h2 : f_inv 46 = 24 := by rfl
  calc
    f_inv 7 + f_inv 46 = 4 + 24 := by rw [h1, h2]
                    ... = 28 := by rfl

end f_inv_sum_l771_771554


namespace lattice_points_on_hyperbola_l771_771911

-- The statement to be proven
theorem lattice_points_on_hyperbola : 
  {p : ℤ × ℤ | p.1 ^ 2 - p.2 ^ 2 = 65}.finite.toFinset.card = 4 :=
by
  sorry

end lattice_points_on_hyperbola_l771_771911


namespace evaluate_pow_l771_771363

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l771_771363


namespace area_of_square_field_l771_771689

theorem area_of_square_field (d : ℝ) (s : ℝ) (A : ℝ) (h_d : d = 28) (h_relation : d = s * Real.sqrt 2) (h_area : A = s^2) :
  A = 391.922 :=
by sorry

end area_of_square_field_l771_771689


namespace prime_factors_absolute_difference_squares_l771_771613

theorem prime_factors_absolute_difference_squares 
  (A B C : ℕ) 
  (hA : A ≠ 0) 
  (hB : B ≠ C) :
  ∃ k : ℕ, k * (A - C) = |(100A + 10B + C)^2 - (100C + 10B + A)^2| :=
by
  sorry

end prime_factors_absolute_difference_squares_l771_771613


namespace tangent_line_eq_at_1_0_curve_intersection_compare_f_l771_771479

open Real

-- Definition of function f(x) = e^x
def f (x : ℝ) : ℝ := exp x

-- Definition of inverse function g(x) = ln x
def g (x : ℝ) : ℝ := log x

-- Definition of h(x) = e^x - (1/2)x^2 - x - 1
def h (x : ℝ) : ℝ := exp x - (1/2) * x^2 - x - 1

-- Definition of comparison functions
def compare_func (a b : ℝ) (ha : a < b) : Prop :=
  (f a + f b) / 2 > (f b - f a) / (b - a)

-- Problem (1): Equation of the tangent line to the inverse function at point (1, 0).
theorem tangent_line_eq_at_1_0:
  let k := 1 in -- Slope of the tangent line
  ∀ (x y: ℝ), (x = 1) → (y = g 1) → (x - y - 1 = 0) :=
by
  sorry

-- Problem (2): The curve y = f(x) intersects y = 1/2 x^2 + x + 1 at exactly one point.
theorem curve_intersection (x : ℝ):
  h 0 = 0 ∧ ∀ y, y ≠ 0 → h y ≠ 0 :=
by
  sorry

-- Problem (3): Comparison of (f(a) + f(b)) / 2 and (f(b) - f(a)) / (b - a) for a < b
theorem compare_f (a b : ℝ) (ha : a < b) :
  compare_func a b ha :=
by
  sorry

end tangent_line_eq_at_1_0_curve_intersection_compare_f_l771_771479


namespace find_equation_of_tangent_line_perpendicular_l771_771420

noncomputable def tangent_line_perpendicular_to_curve (a b : ℝ) : Prop :=
  (∃ (P : ℝ × ℝ), P = (-1, -3) ∧ 2 * P.1 - 6 * P.2 + 1 = 0 ∧ P.2 = P.1^3 + 5 * P.1^2 - 5) ∧
  (-3) = 3 * (-1)^2 + 6 * (-1)

theorem find_equation_of_tangent_line_perpendicular :
  tangent_line_perpendicular_to_curve (-1) (-3) →
  ∀ x y : ℝ, 3 * x + y + 6 = 0 :=
by
  sorry

end find_equation_of_tangent_line_perpendicular_l771_771420


namespace john_new_bench_press_l771_771102

theorem john_new_bench_press (initial_weight : ℕ) (decrease_percent : ℕ) (retain_percent : ℕ) (training_factor : ℕ) (final_weight : ℕ) 
  (h1 : initial_weight = 500)
  (h2 : decrease_percent = 80)
  (h3 : retain_percent = 20)
  (h4 : training_factor = 3)
  (h5 : final_weight = initial_weight * retain_percent / 100 * training_factor) : 
  final_weight = 300 := 
by sorry

end john_new_bench_press_l771_771102


namespace translate_sin_function_l771_771212

theorem translate_sin_function :
  ∀ x : ℝ, 
  let f : ℝ → ℝ := λ x, 2 * Real.sin ((x / 3) + (Real.pi / 6))
  let g : ℝ → ℝ := λ x, 2 * Real.sin ((x / 3) + (Real.pi / 4)) + 3
  in g x = (2 * Real.sin ((x + (Real.pi / 4)) / 3 + (Real.pi / 6)) + 3) :=
by {
  sorry
}

end translate_sin_function_l771_771212


namespace gcd_153_119_l771_771680

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_l771_771680


namespace part_a_part_b_part_c_l771_771845

structure Pentagon :=
(vertices : Fin 5 → Point)

def is_legal_point (p : Point) (pent : Pentagon) : Prop :=
  p ∈ finset.univ.image pent.vertices ∨
  ∃ (p1 p2 : Point), is_legal_point p1 pent ∧ is_legal_point p2 pent ∧ p = intersection_point p1 p2

inductive LegalSegment (pent : Pentagon)
| mk (p1 p2 : Point) (hp1 : is_legal_point p1 pent) (hp2 : is_legal_point p2 pent) : LegalSegment

def is_legal_triangulation (pent : Pentagon) (triangles : Finset (Triangle)) : Prop :=
  ∀ (t ∈ triangles), ∃ (p1 p2 p3 : Point), LegalSegment pent.mk p1 p2 ∧ 
  LegalSegment pent.mk p2 p3 ∧ LegalSegment pent.mk p1 p3 ∧ 
  t = Triangle.mk p1 p2 p3

theorem part_a (pent : Pentagon) : 
  ∃ (triangles : Finset (Triangle)), is_legal_triangulation pent triangles ∧ triangles.card = 7 :=
  sorry

theorem part_b (pent : Pentagon) (k : ℕ) (hk : k % 2 = 1 ∧ k > 1) : 
  ∃ (triangles : Finset (Triangle)), is_legal_triangulation pent triangles ∧ triangles.card = k :=
  sorry

theorem part_c (pent : Pentagon) (k : ℕ) (hk : k % 2 = 0) : 
  ¬ (∃ (triangles : Finset (Triangle)), is_legal_triangulation pent triangles ∧ triangles.card = k) :=
  sorry

end part_a_part_b_part_c_l771_771845


namespace problem_statement_l771_771473

noncomputable def A := 5 * Real.pi / 12
noncomputable def B := Real.pi / 3
noncomputable def C := Real.pi / 4
noncomputable def b := Real.sqrt 3
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3

theorem problem_statement :
  (Set.Icc (-2 : ℝ) 2 = Set.image f Set.univ) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∃ (area : ℝ), area = (3 + Real.sqrt 3) / 4)
:= sorry

end problem_statement_l771_771473


namespace unique_solution_of_equations_l771_771832

def equation1 (x y : ℝ) : Prop :=
  y = x + sqrt (x + sqrt (x + sqrt (x + ... + sqrt (x + sqrt y))))

def equation2 (x y : ℝ) : Prop :=
  x + y = 6

theorem unique_solution_of_equations (x y : ℝ) :
  equation1 x y → equation2 x y → (x, y) = (2, 4) :=
by
  sorry

end unique_solution_of_equations_l771_771832


namespace statements_l771_771169

open Real EuclideanGeometry

def curve_C (a : ℝ) (h : a > 1) : Set (ℝ × ℝ) := 
  { P | (dist P (-1, 0)) * (dist P (1, 0)) = a^2 }

theorem statements (a : ℝ) (h : a > 1) :
  ¬(0, 0) ∈ curve_C a h ∧ 
  (∀ P ∈ curve_C a h, (-P.1, -P.2) ∈ curve_C a h) ∧
  (∀ P ∈ curve_C a h, (euclidean_area (triangle_mk (P, (-1, 0), (1, 0)))) ≤ (1 / 2) * a^2) ∧
  (∀ P ∈ curve_C a h, (dist P (-1, 0)) + (dist P (1, 0)) + 2 ≥ 2 * a + 2) :=
by
  sorry

end statements_l771_771169


namespace scientific_notation_l771_771935

theorem scientific_notation :
  686530000 = 6.8653 * 10^8 :=
sorry

end scientific_notation_l771_771935


namespace value_of_p_h_3_l771_771919

-- Define the functions h and p
def h (x : ℝ) : ℝ := 4 * x + 5
def p (x : ℝ) : ℝ := 6 * x - 11

-- Statement to prove
theorem value_of_p_h_3 : p (h 3) = 91 := sorry

end value_of_p_h_3_l771_771919


namespace fishing_trip_costs_l771_771759

/-- Given the following conditions:
  - Alice paid $135.
  - Bob paid $165.
  - Chris paid $225.
  - All costs are split equally.
  Prove that Alice should give Chris $x$ dollars, Bob should give Chris $y$ dollars,
    and calculate \( x - y = 30 \).
-/
theorem fishing_trip_costs :
  let Alice_paid := 135,
      Bob_paid := 165,
      Chris_paid := 225,
      total_cost := Alice_paid + Bob_paid + Chris_paid,
      avg_cost := total_cost / 3,
      Alice_share := avg_cost - Alice_paid,
      Bob_share := avg_cost - Bob_paid
  in Alice_share - Bob_share = 30 := 
by {
  sorry
}

end fishing_trip_costs_l771_771759


namespace books_cost_l771_771207

theorem books_cost (total_cost_three_books cost_seven_books : ℕ) 
  (h₁ : total_cost_three_books = 45)
  (h₂ : cost_seven_books = 7 * (total_cost_three_books / 3)) : 
  cost_seven_books = 105 :=
  sorry

end books_cost_l771_771207


namespace card_probability_l771_771661

theorem card_probability :
  let hearts := 13
  let clubs := 13
  let total_cards := 52
  let first_card_is_heart := (hearts.to_rat / total_cards.to_rat)
  let second_card_is_club_given_first_is_heart := (clubs.to_rat / (total_cards - 1).to_rat)
  first_card_is_heart * second_card_is_club_given_first_is_heart = (13.to_rat / 204.to_rat) := by
  sorry

end card_probability_l771_771661


namespace general_formula_an_bounds_Mn_l771_771578

variable {n : ℕ}

-- Define the sequence Sn
def S : ℕ → ℚ := λ n => n * (4 * n - 3) - 2 * n * (n - 1)

-- Define the sequence an based on Sn
def a : ℕ → ℚ := λ n =>
  if n = 0 then 0 else S n - S (n - 1)

-- Define the sequence Mn and the bounds to prove
def M : ℕ → ℚ := λ n => (1 / 4) * (1 - (1 / (4 * n + 1)))

-- Theorem: General formula for the sequence {a_n}
theorem general_formula_an (n : ℕ) (hn : 1 ≤ n) : a n = 4 * n - 3 :=
  sorry

-- Theorem: Bounds for the sequence {M_n}
theorem bounds_Mn (n : ℕ) (hn : 1 ≤ n) : (1 / 5 : ℚ) ≤ M n ∧ M n < (1 / 4) :=
  sorry

end general_formula_an_bounds_Mn_l771_771578


namespace complement_of_intersection_l771_771138

def S : Set ℝ := {-2, -1, 0, 1, 2}
def T : Set ℝ := {x | x + 1 ≤ 2}
def complement (A B : Set ℝ) : Set ℝ := {x ∈ B | x ∉ A}

theorem complement_of_intersection :
  complement (S ∩ T) S = {2} :=
by
  sorry

end complement_of_intersection_l771_771138


namespace no_consecutive_days_played_l771_771715

theorem no_consecutive_days_played (john_interval mary_interval : ℕ) :
  john_interval = 16 ∧ mary_interval = 25 → 
  ¬ ∃ (n : ℕ), (n * john_interval + 1 = m * mary_interval ∨ n * john_interval = m * mary_interval + 1) :=
by
  sorry

end no_consecutive_days_played_l771_771715


namespace construct_convex_quadrilateral_with_perpendicular_diagonals_l771_771856

noncomputable def distances_to_vertices (P : Point) (N : Square) : List ℝ := 
[sorry, sorry, sorry, sorry]

theorem construct_convex_quadrilateral_with_perpendicular_diagonals
  (N : Square) (P : Point)
  (hP₁ : ¬ (P ∈ N.edges))
  (dists : List ℝ := distances_to_vertices P N) :
  ∃ quad : Quadrilateral, (convex quad ∧ diagonals_perpendicular quad) := sorry

end construct_convex_quadrilateral_with_perpendicular_diagonals_l771_771856


namespace sum_dist_geq_nR_l771_771592

open EuclideanGeometry

-- Define the points and vectors
variables (O X : Point3d) (n : ℕ) (R : ℝ)
variables (A : Fin n → Point3d)
variable (h_circle : ∀ i, dist O (A i) = R)
variable (h_sum_zero : (Finset.univ.sum (λ i, vectorBetween O (A i))) = vectorZero)

-- The theorem statement
theorem sum_dist_geq_nR : ∑ i, dist X (A i) ≥ n * R :=
sorry

end sum_dist_geq_nR_l771_771592


namespace line_circle_relationship_l771_771051

theorem line_circle_relationship (m : ℝ) :
  (∃ x y : ℝ, (mx + y - m - 1 = 0) ∧ (x^2 + y^2 = 2)) ∨ 
  (∃ x : ℝ, (x - 1)^2 + (m*(x - 1) + (1 - 1))^2 = 2) :=
by
  sorry

end line_circle_relationship_l771_771051


namespace OBEC_area_correct_l771_771296

-- Define points and lines based on the given conditions.
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 6, y := 0} -- Intersection with x-axis for line with slope -2
def B : Point := {x := 0, y := 12} -- Intersection with y-axis for line with slope -2
def C : Point := {x := 8, y := 0} -- Intersection with x-axis for second line
def D : Point := {x := 0, y := 8} -- Intersection with y-axis for second line (derived)
def E : Point := {x := 4, y := 4}

def OBEC_area (O B E C : Point) : ℝ :=
  let oxy_area := 4 * 4 -- Area of square OEXY when side length is 4
  let xce_area := (4 * 4) / 2 -- Area of triangle XCE
  let yeb_area := (8 * 4) / 2 -- Area of triangle YEB
  oxy_area + xce_area + yeb_area

theorem OBEC_area_correct : OBEC_area ⟨0,0⟩ B E C = 40 := by
  sorry

end OBEC_area_correct_l771_771296


namespace books_cost_l771_771206

theorem books_cost (total_cost_three_books cost_seven_books : ℕ) 
  (h₁ : total_cost_three_books = 45)
  (h₂ : cost_seven_books = 7 * (total_cost_three_books / 3)) : 
  cost_seven_books = 105 :=
  sorry

end books_cost_l771_771206


namespace cost_of_first_type_of_rice_l771_771283

theorem cost_of_first_type_of_rice:
  ∀ (x : ℝ),
  let total_weight := 8 + 4,
      average_price := 18,
      total_cost := total_weight * average_price in
  8 * x + 4 * 22 = total_cost → x = 16 :=
by
  intro x
  let total_weight := 8 + 4
  let average_price := 18
  let total_cost := total_weight * average_price
  intro h
  sorry

end cost_of_first_type_of_rice_l771_771283


namespace distance_between_points_l771_771242

-- Define the coordinates of the points
def P1 := (-3 : ℝ, 5 : ℝ)
def P2 := (4 : ℝ, -9 : ℝ)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The theorem to be proved
theorem distance_between_points :
  distance P1 P2 = real.sqrt 245 :=
by sorry

end distance_between_points_l771_771242


namespace midpoint_proof_l771_771957

variable {Point : Type}
variable [Add Point] [Div Point Point] [OfNat Point 2] [OfNat Point (-1)] [OfNat Point 5]
variable [OfNat Point 0] [OfNat Point 1] [OfNat Point 2]  

def midpoint3D (A B : Point × Point × Point) : Point × Point × Point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

theorem midpoint_proof : 
  midpoint3D (-1, 0, 1) (5, 2, 1) = (2, 1, 1) :=
by
  sorry

end midpoint_proof_l771_771957


namespace evaluate_neg_sixtyfour_exp_four_thirds_l771_771409

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l771_771409


namespace smallest_side_length_of_square_l771_771747

theorem smallest_side_length_of_square : 
  ∃ s : ℕ, 
  (s ^ 2 ≥ 21) 
  ∧ (∀ t : ℕ, t ^ 2 ≥ 21 → s ≤ t) := 
begin
  use 5,
  split,
  { norm_num, },
  { intros t ht,
    interval_cases t; norm_num at ht; contradiction }
end

end smallest_side_length_of_square_l771_771747


namespace derivative_at_one_is_three_l771_771015

-- Definition of the function
def f (x : ℝ) := (x - 1)^2 + 3 * (x - 1)

-- The statement of the problem
theorem derivative_at_one_is_three : deriv f 1 = 3 := 
  sorry

end derivative_at_one_is_three_l771_771015


namespace compounded_interest_approx_l771_771045

variable (K : Real) (p : Real) (n : Nat)

noncomputable def final_capital_approximation : Real :=
  K * (1 + n * (p / 100) + (n * (n - 1) / 2) * (p / 100) ^ 2 + (n * (n - 1) * (n - 2) / 6) * (p / 100) ^ 3)

theorem compounded_interest_approx : 
  K * (1 + (p / 100)) ^ n ≈ final_capital_approximation K p n :=
sorry

end compounded_interest_approx_l771_771045


namespace factorize_expression_l771_771797

theorem factorize_expression (m : ℝ) : m^2 + 3 * m = m * (m + 3) :=
by
  sorry

end factorize_expression_l771_771797


namespace thief_overtaken_distance_l771_771749

def distance_thief_runs_before_overtaken (initial_distance : ℝ) 
(speed_thief : ℝ) (speed_policeman : ℝ) : ℝ :=
  let relative_speed := speed_policeman - speed_thief
  let relative_speed_mps := relative_speed * 1000 / 3600
  let time_to_catch_up := initial_distance / relative_speed_mps
  let speed_thief_mps := speed_thief * 1000 / 3600
  speed_thief_mps * time_to_catch_up

theorem thief_overtaken_distance :
  distance_thief_runs_before_overtaken 300 12 16 = 900 := 
sorry

end thief_overtaken_distance_l771_771749


namespace eq_solutions_num_l771_771348

noncomputable def num_solutions (a : ℝ) : ℕ :=
  if a = sqrt 3 ∨ a = 3 then 1
  else if a ∈ set.Icc (sqrt 6) 3 then 2
  else if a ∈ set.Ioo (sqrt 3) (sqrt 6) then 1
  else 0

theorem eq_solutions_num (a : ℝ) :
  num_solutions a = 
    if a = sqrt 3 ∨ a = 3 then 1
    else if a ∈ set.Icc (sqrt 6) 3 then 2
    else if a ∈ set.Ioo (sqrt 3) (sqrt 6) then 1
    else 0 := 
  sorry

end eq_solutions_num_l771_771348


namespace kim_change_l771_771105

def meal_cost := 10
def drink_cost := 2.5
def tip_percentage := 0.2
def total_payment := 20

theorem kim_change : total_payment - (meal_cost + drink_cost + (meal_cost + drink_cost) * tip_percentage) = 5 := 
by 
  have total_cost := meal_cost + drink_cost 
  have tip := total_cost * tip_percentage 
  have total_amount := total_cost + tip 
  have change := total_payment - total_amount 
  exact change

end kim_change_l771_771105


namespace BE_eq_16_div_3_l771_771090

-- Definitions and conditions based on problem statement
variable (A B C D E F : Point)
variable (AF DF CF BE AC : ℝ)
variable (h₁ : ∠ BAC = 90°)
variable (h₂ : ∠ BCD = 90°)
variable (AF : ℕ) (DF : ℕ) (CF : ℕ) (hAF : AF = 4) (hDF : DF = 6) (hCF : CF = 8)
variable (hDF_perp : DF ⊥ AC) (hBE_perp : BE ⊥ AC) (hE_F_on_AC : E ∈ AC ∧ F ∈ AC)

-- Prove statement
theorem BE_eq_16_div_3 : BE = 16 / 3 :=
by
  sorry

end BE_eq_16_div_3_l771_771090


namespace minimum_value_condition_l771_771926

-- Define the function y = x^3 - 2ax + a
noncomputable def f (a x : ℝ) : ℝ := x^3 - 2 * a * x + a

-- Define its derivative
noncomputable def f' (a x : ℝ) : ℝ := 3 * x^2 - 2 * a

-- Define the lean theorem statement
theorem minimum_value_condition (a : ℝ) : 
  (∃ x y : ℝ, 0 < x ∧ x < 1 ∧ y = f a x ∧ (∀ z : ℝ, 0 < z ∧ z < 1 → f a z ≥ y)) ∧
  ¬(∃ x y : ℝ, 0 < x ∧ x < 1 ∧ y = f a x ∧ (∀ z : ℝ, 0 < z ∧ z < 1 → f a z < y)) 
  ↔ 0 < a ∧ a < 3 / 2 :=
sorry

end minimum_value_condition_l771_771926


namespace coffee_on_Thursday_coffee_on_Friday_average_coffee_l771_771633

noncomputable def coffee_consumption (k h : ℝ) : ℝ := k / h

theorem coffee_on_Thursday : coffee_consumption 24 4 = 6 :=
by sorry

theorem coffee_on_Friday : coffee_consumption 24 10 = 2.4 :=
by sorry

theorem average_coffee : 
  (coffee_consumption 24 8 + coffee_consumption 24 4 + coffee_consumption 24 10) / 3 = 3.8 :=
by sorry

end coffee_on_Thursday_coffee_on_Friday_average_coffee_l771_771633


namespace evaluate_neg_sixtyfour_exp_four_thirds_l771_771404

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l771_771404


namespace min_value_of_expression_l771_771019

theorem min_value_of_expression {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 1/a + 2/b = 3) :
  (a + 1) * (b + 2) = 50/9 :=
sorry

end min_value_of_expression_l771_771019


namespace find_center_and_radius_sum_l771_771561

-- Define the given equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 16 * x + y^2 + 10 * y = -75

-- Define the center of the circle
def center (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → (x = a) ∧ (y = b)

-- Define the radius of the circle
def radius (r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → (x^2 - 16 * x + y^2 + 10 * y = r^2)

-- Main theorem to prove a + b + r = 3 + sqrt 14
theorem find_center_and_radius_sum (a b r : ℝ) (h_cen : center a b) (h_rad : radius r) : 
  a + b + r = 3 + Real.sqrt 14 :=
  sorry

end find_center_and_radius_sum_l771_771561


namespace find_integer_pairs_l771_771417

noncomputable def modulusConditionSatisfied (a b : ℤ) : Prop :=
    ∀ r : ℂ, (r^2 + a * r + b = 0) → |r| < 2

theorem find_integer_pairs (a b : ℤ) :
  (∃ P : Polynomial ℤ, (Polynomial.monic (X^2 + a * X + b) * P) ∧ 
    (∀ coeff : ℤ, coeff ∈ (Polynomial.coeffs (X^2 + a * X + b) * P) → coeff = 1 ∨ coeff = -1))
  ↔ modulusConditionSatisfied a b :=
sorry

end find_integer_pairs_l771_771417


namespace perpendicular_DE_CF_ratio_CF_PC_EP_l771_771111

variable {a : ℝ} (A B C D E F P: ℝ²)
variable {side_length : ℝ}
variable (E_is_midpoint_AB : E = ((2 * a + 2 * a) / 2, (0 + 2 * a) / 2))
variable (F_is_midpoint_AD : F = ((0 + 2 * a) / 2, (0 + 0) / 2))
variable (intersection_P : P = (function_to_find_intersection C F D E))

theorem perpendicular_DE_CF 
  (square_ABCD : (A = (2 * a, 0)) ∧ (B = (2 * a, 2 * a)) ∧ (C = (0, 2 * a)) ∧ (D = (0, 0)))
  (midpoints_EF : E = (2 * a, a) ∧ F = (a, 0)):
  slope D E * slope C F = -1 
:= sorry

theorem ratio_CF_PC_EP 
  (square_side_length : side_length = 2 * a)
  (midpoints_positions : E = (2 * a, a) ∧ F = (a, 0))
  (P_intersection : P = intersection_function C F D E)
  (lengths_CF_PC_EP : dist C F = 4 * x ∧ dist P C = 3 * x ∧ dist E P = 3 * x):
  4 / 4 = 4 / 3 ∧ 3 / 3 = 1 := sorry

end perpendicular_DE_CF_ratio_CF_PC_EP_l771_771111


namespace AM_eq_diameter_of_incircle_l771_771131

variables {A B C K O M : Type} [Point]
variables (r : ℝ) [incircle_center K A B C] [circumcircle_center O A B C]
variables [circle_passing_through_and_tangent_to K A O M]

theorem AM_eq_diameter_of_incircle (h : intersects_at OA M) : AM = 2 * r := by
  sorry

end AM_eq_diameter_of_incircle_l771_771131


namespace lcm_18_45_l771_771696

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end lcm_18_45_l771_771696


namespace problem1_problem2_l771_771529

-- Conditions for the triangle
variables {A B C : ℝ} {a b c : ℝ}
variable h1 : a + b + c = 16

-- Problem 1 conditions
variable ha : a = 4
variable hb : b = 5

-- Problem 1 target
theorem problem1 : c = 16 - (a + b) → cos C = -1 / 5 := by
  intro h2
  have hc : c = 16 - (a + b) := h2 
  sorry

-- Problem 2 conditions
variable hsina : sin A + sin B = 3 * sin C
variable area : 18 * sin C = 1 / 2 * a * b * sin C

-- Problem 2 target
theorem problem2 : a = 6 ∧ b = 6 := by
  sorry

end problem1_problem2_l771_771529


namespace customers_non_holiday_l771_771720

theorem customers_non_holiday (h : ∀ n, 2 * n = 350) (H : ∃ h : ℕ, h * 8 = 2800) : (2800 / 8 / 2 = 175) :=
by sorry

end customers_non_holiday_l771_771720


namespace polynomial_division_result_l771_771247

def polynomial : ℤ[X] := 3 * X^3 - 2 * X^2 - 23 * X + 60
def divisor : ℤ[X] := X - 6
def remainder : ℤ := -378

theorem polynomial_division_result :
  polynomial % divisor = C remainder := sorry

end polynomial_division_result_l771_771247


namespace evaluate_neg_64_exp_4_over_3_l771_771373

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l771_771373


namespace car_efficiency_pre_modification_l771_771723

def pre_mod_efficiency := 32

variable (x : ℝ)
variable (fuel_efficiency_before : ℝ)
variable (fuel_efficiency_after : ℝ)
variable (miles_more : ℝ)
variable (tank_capacity : ℝ)

-- Conditions
def condition_1 := fuel_efficiency_after = fuel_efficiency_before / 0.8
def condition_2 := tank_capacity = 12
def condition_3 := 12 * fuel_efficiency_after - 12 * fuel_efficiency_before = miles_more

-- Factoring in the conditions
axiom h1 : condition_1
axiom h2 : condition_2
axiom h3 : condition_3

theorem car_efficiency_pre_modification : 
  fuel_efficiency_before = pre_mod_efficiency :=
by
  sorry

end car_efficiency_pre_modification_l771_771723


namespace angle_B_value_l771_771114

variable {ABC : Type*} [triangle ABC] [right_triangle ABC]
variable (O H : point)
variable [is_circumcenter O ABC] [is_orthocenter H ABC]
variable (B C : point)
variable (angle_B angle_C : angle)
variable (BO BH : dist)
variable [eq_angle angle_C (90 : ℝ)]
variable [eq_angle (dist BO) (dist BH)]

theorem angle_B_value (ABC : Type*) [triangle ABC] [right_triangle ABC]
  (O H : point) 
  [is_circumcenter O ABC] [is_orthocenter H ABC]
  (B C : point) (angle_B angle_C : angle)
  (BO BH : dist)
  [eq_angle angle_C (90 : ℝ)]
  [eq_angle (dist BO) (dist BH)] : 
  eq_angle angle_B (60 : ℝ) := 
sorry

end angle_B_value_l771_771114


namespace longest_side_of_polygonal_region_l771_771279

theorem longest_side_of_polygonal_region :
  (∃ (x y : ℝ), x + 2 * y ≤ 6 ∧ 3 * x + y ≥ 3 ∧ x ≤ 4 ∧ y ≥ 0) →
  (x + 2 * y ≤ 6 ∧ 3 * x + y ≥ 3 ∧ x ≤ 4 ∧ y ≥ 0 → 
  ∃ (length : ℝ), length = 2 * sqrt 5) := by
  intros
  sorry

end longest_side_of_polygonal_region_l771_771279


namespace factorization_l771_771795

theorem factorization (m : ℤ) : m^2 + 3 * m = m * (m + 3) :=
by sorry

end factorization_l771_771795


namespace area_of_square_WXYZ_l771_771167

theorem area_of_square_WXYZ : 
  (∀ (A B C D X W Y Z : ℝ), 
  (A - B) ^ 2 + (B - C) ^ 2 + (C - D) ^ 2 + (D - A) ^ 2 = 256 → 
  (X - W) ^ 2 + (W - Y) ^ 2 + (Y - Z) ^ 2 + (Z - X) ^ 2 = 16) →
  ∀ (A B C D X W Y Z : ℝ), 
  (√ ([|X - A|^2 + 2^2] * [|X - A|^2 + 6^2]) + 
   √ ([|W - B|^2 + 2^2] * [|W - B|^2 + 6^2]) + 
   √ ([|Y - C|^2 + 2^2] * [|Y - C|^2 + 6^2]) + 
   √ ([|Z - D|^2 + 2^2] * [|Z - D|^2 + 6^2]) = √ 40) := 
sorry

end area_of_square_WXYZ_l771_771167


namespace heather_distance_17_09_l771_771271

theorem heather_distance_17_09 :
  ∀ (H S : ℝ) (distance_between : ℝ) (heather_start_delay : ℝ),
  H = 5 →
  S = H + 1 →
  distance_between = 40 →
  heather_start_delay = 0.4 →
  (let t := (distance_between - (S * heather_start_delay)) / (H + S) in
    H * t ≈ 17.09) :=
by
  intros H S distance_between heather_start_delay H_eq S_eq dist_eq delay_eq
  have H_annot := by rw [H_eq at *]
  have S_annot := by rw [H_eq at *; S_eq at *]
  have dist_annot := by rw [dist_eq at *]
  have delay_annot := by rw [delay_eq at *]
  have t_def : ℝ := (distance_between - (S * heather_start_delay)) / (H + S)
  sorry

end heather_distance_17_09_l771_771271


namespace compute_binom_10_3_l771_771777

theorem compute_binom_10_3 : binom 10 3 = 120 := by
  sorry

end compute_binom_10_3_l771_771777


namespace rectangle_tangent_circles_l771_771557

noncomputable def rectangle_properties (r1 r2 : ℝ) (r1_lt_r2 : r1 < r2) (a b : ℝ) (rectangle : a > b) : 
  Prop := 
  let a_def := a = r1 + r2 + 2 * real.sqrt (r1 * r2) in
  let b_def := b = 2 * r1 * (r2 + real.sqrt (r1 * r2)) / (r2 - r1) in
  a_def ∧ b_def

theorem rectangle_tangent_circles (r1 r2 : ℝ) (r1_lt_r2 : r1 < r2) (a b : ℝ) (rectangle : a > b) :
  rectangle_properties r1 r2 r1_lt_r2 a b rectangle :=
begin
  sorry
end

end rectangle_tangent_circles_l771_771557


namespace cost_price_of_sports_suits_l771_771288

theorem cost_price_of_sports_suits :
  ∃ x y : ℝ, 
  (x > 0) ∧ 
  (y > 0) ∧ 
  (8100 / x = 9000 / y) ∧ 
  (x + y = 380) ∧ 
  (x = 180) ∧ 
  (y = 200) :=
by
  have h1 : ∀ x y : ℝ, (x > 0) → (y > 0) → (8100 / x = 9000 / y) → (x + y = 380) → x = 180 → y = 200 := sorry
  existsi (180 : ℝ)
  existsi (200 : ℝ)
  split
  · exact sorry 
  split
  · exact sorry 
  split
  · exact sorry 
  split
  · exact sorry 
  · exact sorry      

end cost_price_of_sports_suits_l771_771288


namespace calculate_inv_dist_sum_l771_771945

noncomputable def polar_eq_C (θ : ℝ) : ℝ := 2 * real.sqrt 2 * real.sin (θ + real.pi / 4)

def line_param_eq (t : ℝ) : ℝ × ℝ := (t * real.cos (real.pi / 3), 1 + t * real.sin (real.pi / 3))

def cart_eq_C (x y : ℝ) : Prop := x^2 + y^2 = 2 * x + 2 * y

theorem calculate_inv_dist_sum :
  let t₁ t₂ : ℝ := by {
    let eq := λ t : ℝ, t^2 - t - 1 = 0,
    let roots := quadratic_roots eq,
    exact (roots.1, roots.2)
  }
  let PM := by {
    let (x, y) := line_param_eq t₁,
    exact real.sqrt (x^2 + (y - 1)^2)
  }
  let PN := by {
    let (x, y) := line_param_eq t₂,
    exact real.sqrt (x^2 + (y - 1)^2)
  }
  (1 / |PM|) + (1 / |PN|) = real.sqrt 5 :=
begin
  sorry
end

end calculate_inv_dist_sum_l771_771945


namespace grasshopper_reaches_pit_l771_771080

-- Define the basic conditions of the problem
def meadow_square (a : ℝ) : Prop := a > 0

def circular_pit (r : ℝ) : Prop := r > 0

def grasshopper_jump (x y : ℝ) (v : ℝ) : Prop :=
    -- Assuming the grasshopper can jump to a vertex (v) and moves half the distance
    (x ≠ v ∨ y ≠ v) → ((x + (v - x) / 2), (y + (v - y) / 2))

-- This is the main theorem to be proven
theorem grasshopper_reaches_pit (a r : ℝ) (hole : ℝ × ℝ) (x y : ℝ):
    meadow_square a →
    circular_pit r →
    (exists n: ℕ, hole ∈ set.Icc (x, y) ((x + a / (2^n)), (y + a / (2^n)))) → 
    ∃ v : ℝ, grasshopper_jump x y v → hole ∈ set.Icc (x, y) (x + a, y + a) :=
by 
  sorry

end grasshopper_reaches_pit_l771_771080


namespace value_of_a_minus_b_l771_771331

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the invertible function f

theorem value_of_a_minus_b (a b : ℝ) (hf_inv : Function.Injective f)
  (hfa : f a = b) (hfb : f b = 6) (ha1 : f 3 = 1) (hb1 : f 1 = 6) : a - b = 2 :=
sorry

end value_of_a_minus_b_l771_771331


namespace well_defined_interval_l771_771353

def is_well_defined (x : ℝ) : Prop :=
  (5 - x > 0) ∧ (x ≠ 2)

theorem well_defined_interval : 
  ∀ x : ℝ, (is_well_defined x) ↔ (x < 5 ∧ x ≠ 2) :=
by 
  sorry

end well_defined_interval_l771_771353


namespace evaluate_neg_64_exp_4_over_3_l771_771375

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l771_771375


namespace complex_number_in_third_quadrant_l771_771087

def quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else if z.re > 0 ∧ z.im < 0 then 4
  else 0 -- Edge case for the origin or axes (not normally needed here)

theorem complex_number_in_third_quadrant : quadrant (-5 - 2 * Complex.I) = 3 :=
by
  sorry

end complex_number_in_third_quadrant_l771_771087


namespace domain_of_k_l771_771690

noncomputable def k (x : ℝ) : ℝ := (1 / (x + 9)) + (1 / (x^2 + 9)) + (1 / (x^3 + 9))

theorem domain_of_k : 
  ∀ x : ℝ, x ∈ (set.Ioo (-∞ : ℝ) (-9) ∪ set.Ioo (-9) (-real.cbrt 9) ∪ set.Ioo (-real.cbrt 9) ∞) ↔ x ∈ {(y : ℝ) | y ≠ -9 ∧ y ≠ -real.cbrt 9} :=
by 
  sorry

end domain_of_k_l771_771690


namespace trapezoid_midpoints_distance_l771_771958

theorem trapezoid_midpoints_distance
  (ABCD : Type) [IsTrapezoid ABCD]
  (BC AD : ℝ)
  (angleA angleD : ℝ)
  (BC_mid AD_mid : Point)
  (BC_parallel_AD : BC ∥ AD)
  (BC_eq : BC = 1000)
  (AD_eq : AD = 2008)
  (angleA_eq : angleA = 37)
  (angleD_eq : angleD = 53)
  (BC_midpoint : IsMidpoint BC_mid BC)
  (AD_midpoint : IsMidpoint AD_mid AD) :
  distance BC_mid AD_mid = 504 := sorry

end trapezoid_midpoints_distance_l771_771958


namespace melanies_mother_gave_l771_771997

-- Define initial dimes, dad's contribution, and total dimes now
def initial_dimes : ℕ := 7
def dad_dimes : ℕ := 8
def total_dimes : ℕ := 19

-- Define the number of dimes the mother gave
def mother_dimes := total_dimes - (initial_dimes + dad_dimes)

-- Proof statement
theorem melanies_mother_gave : mother_dimes = 4 := by
  sorry

end melanies_mother_gave_l771_771997


namespace tina_sold_books_to_4_customers_l771_771652

theorem tina_sold_books_to_4_customers :
  (∀ (price_per_book cost_per_book total_profit books_per_customers : ℕ), 
  price_per_book = 20 ∧ cost_per_book = 5 ∧ total_profit = 120 ∧ books_per_customers = 2 →
  (total_profit / (price_per_book - cost_per_book)) / books_per_customers = 4) :=
begin
  intros price_per_book cost_per_book total_profit books_per_customers h,
  cases h with h1 h_rem,
  cases h_rem with h2 h_rem,
  cases h_rem with h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
end

end tina_sold_books_to_4_customers_l771_771652


namespace non_intersecting_pairs_exists_l771_771852

open Set

theorem non_intersecting_pairs_exists (T : Set (ℝ × ℝ)) (hT : ∃ n, T.card = 2 * n) :
  ∃ P : Set (Set (ℝ × ℝ)), (∀ p ∈ P, p.card = 2) ∧ (P.pairwise Disjoint ∧ P ⊆ T):  sorry

end non_intersecting_pairs_exists_l771_771852


namespace count_integers_with_sum_digits_17_l771_771902

-- Define the sum of digits of a number
def sum_digits (n : ℕ) : ℕ :=
  (n / 100) + (n % 100) / 10 + (n % 10)

-- Define the condition that integer is between 400 and 600
def in_range (n : ℕ) : Prop :=
  400 ≤ n ∧ n ≤ 600

-- Define the main theorem
theorem count_integers_with_sum_digits_17 : ∃ k = 13, ∃ (n : ℕ), in_range n ∧ sum_digits n = 17 := by
  sorry

end count_integers_with_sum_digits_17_l771_771902


namespace cookies_remaining_percentage_l771_771202

theorem cookies_remaining_percentage (c : ℕ) (p_n p_e : ℚ) (h_c : c = 600) (h_nicole : p_n = 2/5) (h_eduardo : p_e = 3/5) :
  ∃ p_r : ℚ, p_r = 24 :=
by
  have h_nicole_cookies : ℚ := p_n * c
  have h_remaining_after_nicole : ℚ := c - h_nicole_cookies
  have h_eduardo_cookies : ℚ := p_e * h_remaining_after_nicole
  have h_remaining_after_eduardo : ℚ := h_remaining_after_nicole - h_eduardo_cookies
  have h_percentage_remaining : ℚ := (h_remaining_after_eduardo / c) * 100
  use h_percentage_remaining
  field_simp [h_c, h_nicole, h_eduardo]
  norm_num
  sorry

end cookies_remaining_percentage_l771_771202


namespace divisible_by_2_is_necessary_not_sufficient_for_6_l771_771834

theorem divisible_by_2_is_necessary_not_sufficient_for_6 (n : ℤ) :
  (∃ k1 : ℤ, n = 2 * k1) → (∃ k2 : ℤ, n = 6 * k2) ↔ (∃ k2 : ℤ, n = 6 * k2) → (∃ k1 : ℤ, n = 2 * k1) :=
by 
  (intro h₁ h₂ ⟨k, hk⟩; rw [hk, mul_assoc]; exact ⟨3 * k, rfl⟩;
   intro h₁; cases h₁ with k1 hk1; use k1;
   rcases int.mod_eq_zero_or_ne_zero_of_eq hk1.symm with (hk | hk);
   exact ⟨⟨k1, hk1⟩, h⟩;
   right; exact ⟨n/2, int.div_mul_cancel hk⟩)

end divisible_by_2_is_necessary_not_sufficient_for_6_l771_771834


namespace problem_equations_and_slope_l771_771873

theorem problem_equations_and_slope (
    a : ℝ
    (l : ℝ × ℝ → ℝ) (l' : ℝ × ℝ → ℝ)
    (center_C : ℝ × ℝ) (M : ℝ × ℝ)
    (N : ℝ × ℝ)
    (P Q : ℝ × ℝ)) :
    (l' = λ p, p.1 - 2 * p.2) →
    (l = λ p, 4 * p.1 + a * p.2 - 5) →
    ( ∀ x, l (2, 1 + x) = l (center_C.1 + x, center_C.2 + x)) →
    ( ∀ x, center_C = (-1, -1) ∨ center_C = (0,0)) → 
    (
    P ∈ (λ p, p.1^2 + p.2^2 = 2) ∧ 
    Q ∈ (λ p, p.1^2 + p.2^2 = 2) →
    (
        let slope_MP := (M.2 - P.2) / (M.1 - P.1),
            slope_MQ := (M.2 - Q.2) / (M.1 - Q.1),
            slope_PQ := (P.2 - Q.2) / (P.1 - Q.1)
        in
        slope_MP + slope_MQ = 0 → slope_PQ = 1)) :=
sorry

end problem_equations_and_slope_l771_771873


namespace students_with_both_dog_and_cat_l771_771328

def JeffersonHigh: Type := {s // s < 50}

variable (D C: Set JeffersonHigh) 

axiom numberOfStudents : D ∪ C = λ _ => True

axiom hasDog : |D| = 35
axiom hasCat : |C| = 42
axiom totalStudents : |D ∪ C| = 50

theorem students_with_both_dog_and_cat : |D ∩ C| = 27 := 
by
  sorry

end students_with_both_dog_and_cat_l771_771328


namespace value_of_f_at_4_l771_771049

noncomputable def f (x : ℝ) (c : ℝ) (d : ℝ) : ℝ :=
  c * x ^ 2 + d * x + 3

theorem value_of_f_at_4 :
  (∃ c d : ℝ, f 1 c d = 3 ∧ f 2 c d = 5) → f 4 1 (-1) = 15 :=
by
  sorry

end value_of_f_at_4_l771_771049


namespace problem1_problem2a_problem2b_l771_771333

-- Mathematical translation of (1)
theorem problem1: 
  27^(2/3) - 2^(Real.log 3 / Real.log 4) * Real.log2 (1/8) + Real.log2 3 * Real.log (4/3) = 11 + 3 * Real.sqrt 3 :=
by
  sorry

-- Definitions and mathematical translation of (2)
def cos_alpha : ℝ := 5 / 13
def is_in_fourth_quadrant : Prop := true  -- Placeholder for quadrant condition

theorem problem2a 
  (h1 : cos_alpha = 5 / 13)
  (h2 : is_in_fourth_quadrant) : 
  Real.sin (angle α) = -12 / 13 :=
by
  sorry

theorem problem2b 
  (h1 : cos_alpha = 5 / 13)
  (h2 : is_in_fourth_quadrant) : 
  Real.tan (angle α) = -12 / 5 :=
by
  sorry

end problem1_problem2a_problem2b_l771_771333


namespace shortest_chord_length_correct_points_concyclic_Q_always_on_line_x_eq_2_l771_771469

-- Variables and Definitions
variables {P Q A B : ℝ × ℝ}
def circle_C := {P | P.1^2 + P.2^2 + 4 * P.1 = 0}
def point_P := (-1 : ℝ, 0 : ℝ)
def tangent_line_A (A : ℝ × ℝ) := (A.1 + 2) * {x | x.1 = A.1} + B.2 + 4 * {y | y.1 = 0}
def tangent_line_B (B : ℝ × ℝ) := (B.1 + 2) * {x | x.1 = B.1} + B.2 + 4 * {y | y.1 = 0}
def mid_point (A B : ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Lean 4 Statements to Prove
theorem shortest_chord_length_correct (l : ℝ → ℝ) : 
  let A B := find_intersections l circle_C in
  |A B| ≥ 2 * real.sqrt (4 - (distance (mid_point point_P (-2 : ℝ, 0)) ^ 2)) := sorry

theorem points_concyclic (Q A B : ℝ × ℝ) : 
  let C := (-2 : ℝ, 0) in
  are_concyclic [Q, A, B, C] := sorry

theorem Q_always_on_line_x_eq_2 (l1 l2 : ℝ → ℝ) (Q : ℝ × ℝ) : 
  let A B := find_intersections tangent_line_A l1, tangent_line_B l2 in
  let Q := line_intersection l1 l2 in 
  Q.1 = 2 := sorry

end shortest_chord_length_correct_points_concyclic_Q_always_on_line_x_eq_2_l771_771469


namespace power_evaluation_l771_771391

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l771_771391


namespace derivative_at_pi_over_4_l771_771031

open Real -- Opening the module for real numbers and related functions

-- Define the function f
def f (x : ℝ) : ℝ := cos x * (sin x - cos x)

-- Define the problem statement that we seek to prove
theorem derivative_at_pi_over_4 : deriv f (π / 4) = 1 :=
by 
  -- Proof is to be filled in. For now, we use "sorry" to indicate the proof.
  sorry

end derivative_at_pi_over_4_l771_771031


namespace ineq_10_3_minus_9_5_l771_771983

variable {a b c : ℝ}

/-- Given \(a, b, c\) are positive real numbers and \(a + b + c = 1\), prove \(10(a^3 + b^3 + c^3) - 9(a^5 + b^5 + c^5) \geq 1\). -/
theorem ineq_10_3_minus_9_5 (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := 
sorry

end ineq_10_3_minus_9_5_l771_771983


namespace find_principal_amount_l771_771766

-- Define the given conditions
def interest_rate1 : ℝ := 0.08
def interest_rate2 : ℝ := 0.10
def interest_rate3 : ℝ := 0.12
def period1 : ℝ := 4
def period2 : ℝ := 6
def period3 : ℝ := 5
def total_interest_paid : ℝ := 12160

-- Goal is to find the principal amount P
theorem find_principal_amount (P : ℝ) :
  total_interest_paid = P * (interest_rate1 * period1 + interest_rate2 * period2 + interest_rate3 * period3) →
  P = 8000 :=
by
  sorry

end find_principal_amount_l771_771766


namespace younger_brother_age_l771_771141

-- Define the ages of Michael, his older brother, and his younger brother.
variables (M O Y : ℕ)

-- Define the conditions.
def condition1 : Prop := O = 2 * (M - 1) + 1
def condition2 : Prop := Y = O / 3
def condition3 : Prop := M + O + Y = 28

-- The statement to be proved.
theorem younger_brother_age (h1 : condition1) (h2 : condition2) (h3 : condition3) : Y = 5 :=
sorry

end younger_brother_age_l771_771141


namespace number_of_fridays_l771_771964

theorem number_of_fridays (jan_1_sat : true) (is_non_leap_year : true) : ∃ (n : ℕ), n = 52 :=
by
  -- Conditions: January 1st is Saturday and it is a non-leap year.
  -- We are given that January 1st is a Saturday.
  have jan_1_sat_condition : true := jan_1_sat
  -- We are given that the year is a non-leap year (365 days).
  have non_leap_condition : true := is_non_leap_year
  -- Therefore, there are 52 Fridays in the year.
  use 52
  done

end number_of_fridays_l771_771964


namespace average_speed_l771_771708

-- Definitions
variable (total_distance : ℝ := 400)
variable (distance1 : ℝ := 100)
variable (speed1 : ℝ := 20)
variable (distance2 : ℝ := 300)
variable (speed2 : ℝ := 15)

-- Theorem
theorem average_speed : 
  let time1 := distance1 / speed1,
      time2 := distance2 / speed2,
      total_time := time1 + time2,
      avg_speed := total_distance / total_time in
  avg_speed = 16 := 
sorry

end average_speed_l771_771708


namespace evaluate_log_expression_l771_771412

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

-- Given conditions
def a (x y : ℝ) : ℝ := log_base (y^3) (x^2)
def b (x y : ℝ) : ℝ := log_base (x^4) (y^3)
def c (x y : ℝ) : ℝ := log_base (y^5) (x^4)
def d (x y : ℝ) : ℝ := log_base (x^2) (y^5)

-- Main theorem
theorem evaluate_log_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  a x y * b x y * c x y * d x y = log_base y x :=
by
  sorry

end evaluate_log_expression_l771_771412


namespace find_m_l771_771933

theorem find_m (m : ℝ) : 
  (∃ α β : ℝ, (α + β = 2 * (m + 1)) ∧ (α * β = m + 4) ∧ ((1 / α) + (1 / β) = 1)) → m = 2 :=
by
  sorry

end find_m_l771_771933


namespace clubsuit_eval_l771_771564

def clubsuit (a b : ℚ) : ℚ := a + b / a

theorem clubsuit_eval : (clubsuit 4 (clubsuit 2 3)) \clubsuit 5 = 1841 / 312 :=
by 
  sorry

end clubsuit_eval_l771_771564


namespace sum_of_distances_from_circumcenter_to_sides_l771_771890

theorem sum_of_distances_from_circumcenter_to_sides :
  let r1 := 3
  let r2 := 5
  let r3 := 7
  let a := r1 + r2
  let b := r1 + r3
  let c := r2 + r3
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r_incircle := area / s
  r_incircle = Real.sqrt 7 →
  let sum_distances := (7 / 4) + (7 / (3 * Real.sqrt 6)) + (7 / (Real.sqrt 30))
  sum_distances = (7 / 4) + (7 / (3 * Real.sqrt 6)) + (7 / (Real.sqrt 30))
:= sorry

end sum_of_distances_from_circumcenter_to_sides_l771_771890


namespace total_marbles_l771_771716

theorem total_marbles (marbles_per_row_8 : ℕ) (rows_of_9 : ℕ) (marbles_per_row_1 : ℕ) (rows_of_4 : ℕ) 
  (h1 : marbles_per_row_8 = 9) 
  (h2 : rows_of_9 = 8) 
  (h3 : marbles_per_row_1 = 4) 
  (h4 : rows_of_4 = 1) : 
  (marbles_per_row_8 * rows_of_9 + marbles_per_row_1 * rows_of_4) = 76 :=
by
  sorry

end total_marbles_l771_771716


namespace domain_of_f_l771_771347

noncomputable def f : ℝ → ℝ := sorry

theorem domain_of_f :
  (∀ x : ℝ, x ≠ 0 → f(x) + f(1/x) = 1 + x^2) →
  (f(2) = 0) →
  (∀ x : ℝ, x ≠ 0) :=
sorry

end domain_of_f_l771_771347


namespace line_intersects_circle_l771_771637

def line_eq (m x y : ℝ) : Prop :=
  m * (x - 1) - y + 3 = 0

def circle_eq (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 5

theorem line_intersects_circle (m : ℝ) :
  ∃ x y, line_eq m x y ∧ circle_eq x y :=
by
  use 1, 3
  split
  · show line_eq m 1 3
    simp [line_eq]
  · show circle_eq 1 3
    simp [circle_eq]
  sorry

end line_intersects_circle_l771_771637


namespace eval_expression_correct_l771_771398

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l771_771398


namespace unknown_person_rabbits_purchased_on_first_day_l771_771764

theorem unknown_person_rabbits_purchased_on_first_day
  (n : ℕ)                                -- total number of rabbits sold each day
  (x : ℕ)                                -- number of rabbits bought by person X on the first day
  (h1 : 0.6 * n - x = 7)                 -- condition: 7 rabbits left after person X's purchase on first day
  (h2 : 0.25 * n = n / 4)                -- equivalence: 25% is the same as 1/4
  (h3 : 0.2 * n = n * 1 / 5)             -- equivalence: 20% is the same as 1/5
  (h4 : 0.5 * n = n / 2)                 -- equivalence: 50% is the same as 1/2
  (h5 : 0.75 * n = n * 3 / 4)            -- equivalence: 75% is the same as 3/4
  (h6 : 0.3 * n = 3 * n / 10)            -- equivalence: 30% is the same as 3/10
  (h7 : 2 * x = 0.5 * n)                 -- condition: person X bought twice the number of rabbits on the second day
  : x = 5 :=
sorry

end unknown_person_rabbits_purchased_on_first_day_l771_771764


namespace power_evaluation_l771_771392

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l771_771392


namespace intersection_M_N_l771_771039

def I : Set ℤ := {0, -1, -2, -3, -4}
def M : Set ℤ := {0, -1, -2}
def N : Set ℤ := {0, -3, -4}

theorem intersection_M_N : M ∩ N = {0} := 
by 
  sorry

end intersection_M_N_l771_771039


namespace polynomial_degree_at_least_n_l771_771555

variable (t : ℝ) (f : ℝ → ℝ) (n : ℕ)

-- Conditions
axiom t_ge_3 : t ≥ 3
axiom f_abs_diff_bound : ∀ k : ℕ, k ≤ n → |f k - t^k| < 1

-- Goal
theorem polynomial_degree_at_least_n (t : ℝ) (f : ℝ → ℝ) (n : ℕ) [t_ge_3 : t ≥ 3] [∀ k : ℕ, k ≤ n → |f k - t^k| < 1] : 
  polynomial.degree f ≥ n :=
sorry

end polynomial_degree_at_least_n_l771_771555


namespace problem_solution_l771_771701

theorem problem_solution (m : ℝ) (hm : m > 0) (x α : ℝ) (k : ℤ) :
  (∀ α, (α = (π / 4) + 2 * k * π) -> true) ∧
  (x ∈ set.Icc 0 (2 * π) → ((π / 2 < x ∧ x < π) ∨ (3 * π / 2 < x ∧ x < 2 * π))) ∧
  (∀ z, (45 + z * 90) ∈ M → (90 + z * 45) ∈ N) ∧
  (α ∈ set.Ioo (π) ((3 * π) / 2) → ((α / 2) ∈ set.Ioo (π / 2) ((3 * π / 4)) ∨ ((2 * α) ∈ set.Ioo (0) (π)))) →
  true := sorry

end problem_solution_l771_771701


namespace exists_N_a_sum_exceed_100N_l771_771043

noncomputable theory
open Classical

def sum_blackboard_sequence_exceeds (N a : ℕ) : Prop :=
  a < N ∧
  (let seq := Nat.unfoldr (λ x, if x = 0 then none else some (N % x, N % x)) a in
   seq.sum > 100 * N)

theorem exists_N_a_sum_exceed_100N : ∃ (N a : ℕ), sum_blackboard_sequence_exceeds N a := 
sorry

end exists_N_a_sum_exceed_100N_l771_771043


namespace relationship_among_abc_l771_771005

-- Define the function f and the given conditions
variable {f : ℝ → ℝ}
variable (even_f : ∀ x : ℝ, f (-x) = f x)
variable (incr_f : ∀ ⦃x y : ℝ⦄, 0 < x → x < y → f x < f y)

-- Define the specific values a, b, c
def a : ℝ := f (sqrt 3)
def b : ℝ := f (log 3 (1/2))
def c : ℝ := f (4/3)

-- The theorem statement
theorem relationship_among_abc : b < c < a := sorry

end relationship_among_abc_l771_771005


namespace lateral_area_of_given_cone_l771_771445

-- Define the conditions
def cone_diameter : ℝ := 6
def slant_height : ℝ := 6
def radius : ℝ := cone_diameter / 2

-- Define the lateral area formula for a cone
def lateral_area (r h : ℝ) : ℝ := π * r * h

-- The main statement to be proved
theorem lateral_area_of_given_cone : lateral_area radius slant_height = 18 * π :=
by sorry

end lateral_area_of_given_cone_l771_771445


namespace max_min_f_in_disk_l771_771813

def f (x y : ℝ) : ℝ := 2 * x^2 - 2 * y^2

theorem max_min_f_in_disk :
  ∀ (x y : ℝ), x^2 + y^2 ≤ 9 → (-18 ≤ f x y ∧ f x y ≤ 18) ∧
    (∃ x y, x^2 + y^2 = 9 ∧ f x y = 18) ∧
    (∃ x y, x^2 + y^2 = 9 ∧ f x y = -18) :=
begin
  sorry
end

end max_min_f_in_disk_l771_771813


namespace monotonic_decreasing_interval_l771_771184

noncomputable def y (x : ℝ) : ℝ := x * log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < real.exp (-1) → deriv y x < 0 :=
by
  intros x h
  have h1 : deriv y x = log x + 1 := sorry
  have h2 : log x + 1 < 0 := sorry
  rw [h1]
  exact h2

end monotonic_decreasing_interval_l771_771184


namespace probability_of_convex_quadrilateral_l771_771355

open Finset

-- Define the number of points on the circle
def num_points : ℕ := 8

-- Define the number of ways to choose 2 points out of num_points
def num_chords : ℕ := choose num_points 2

-- Define the number of ways to choose 4 chords from num_chords
def num_ways_to_choose_4_chords : ℕ := choose num_chords 4

-- Define the number of ways to choose 4 points out of num_points, each forming a convex quadrilateral
def num_ways_to_form_convex_quad : ℕ := choose num_points 4

-- Define the probability calculation
def probability : ℚ := num_ways_to_form_convex_quad / num_ways_to_choose_4_chords

-- Main theorem to prove
theorem probability_of_convex_quadrilateral : probability = 2 / 585 := by sorry

end probability_of_convex_quadrilateral_l771_771355


namespace evaluate_neg_64_exp_4_over_3_l771_771372

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l771_771372


namespace geometric_sequence_150th_term_l771_771419

-- Given conditions
def a1 : ℤ := 5
def a2 : ℤ := -10

-- Computation of common ratio
def r : ℤ := a2 / a1

-- Definition of the n-th term in geometric sequence
def nth_term (n : ℕ) : ℤ :=
  a1 * r^(n-1)

-- Statement to prove
theorem geometric_sequence_150th_term :
  nth_term 150 = -5 * 2^149 :=
by
  sorry

end geometric_sequence_150th_term_l771_771419


namespace area_of_intersecting_plane_l771_771763

noncomputable def midpoint (p1 : ℝ × ℝ × ℝ) (p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

noncomputable def calculate_area (A B C : ℝ × ℝ × ℝ) : ℝ :=
let a := (B.1 - A.1, B.2 - A.2, B.3 - A.3) in
let b := (C.1 - A.1, C.2 - A.2, C.3 - A.3) in
0.5 * ((a.2 * b.3 - a.3 * b.2) ^ 2 + (a.3 * b.1 - a.1 * b.3) ^ 2 + (a.1 * b.2 - a.2 * b.1) ^ 2) ^ 0.5

theorem area_of_intersecting_plane : 
  let A := (0.0, 0.0, 0.0) in
  let B := (6.0, 0.0, 0.0) in
  let C := (3.0, 3.0 * Real.sqrt 3, 0.0) in
  let P := (3.0, Real.sqrt 3, 3.0 * Real.sqrt 6) in
  let M := midpoint P A in
  let N := midpoint A B in
  let Q := midpoint B C in
  calculate_area M N Q = "expected exact value" :=
by
  -- Compute all necessary vectors and plane equations
  -- Derive intersections and area calculations
  -- Validate computed areas using the correct formula
  sorry

end area_of_intersecting_plane_l771_771763


namespace mica_total_cost_l771_771998

def pasta_cost (quantity price_per_kg : ℝ) : ℝ :=
  quantity * price_per_kg

def ground_beef_cost (quantity price_per_kg : ℝ) : ℝ :=
  quantity * price_per_kg

def pasta_sauce_cost (quantity price_per_jar : ℝ) : ℝ :=
  quantity * price_per_jar

def quesadilla_cost (price : ℝ) : ℝ :=
  price

def total_cost : ℝ :=
  let pasta := pasta_cost 2 1.5
  let beef := ground_beef_cost (1/4) 8
  let sauce := pasta_sauce_cost 2 2
  let quesadilla := quesadilla_cost 6
  pasta + beef + sauce + quesadilla

theorem mica_total_cost : total_cost = 15 := by
  sorry

end mica_total_cost_l771_771998


namespace distance_between_points_l771_771231

def point : Type := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  Math.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem distance_between_points :
  distance (-3, 5 : point) (4, -9 : point) = Math.sqrt 245 := 
sorry

end distance_between_points_l771_771231


namespace eval_neg_pow_l771_771377

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l771_771377


namespace no_figure_with_center_and_one_axis_of_symmetry_l771_771773

noncomputable def has_center_of_symmetry (figure : Type) (O : point) := 
  ∀ P : point, ∃ P' : point, midpoint O P P' ∧ is_on_figure figure P ∧ is_on_figure figure P’

noncomputable def has_exactly_one_axis_of_symmetry (figure : Type) :=
  ∃! l : line, is_axis_of_symmetry figure l

theorem no_figure_with_center_and_one_axis_of_symmetry (figure : Type) 
  (O : point) (l : line) (H1 : has_center_of_symmetry figure O) 
  (H2 : has_exactly_one_axis_of_symmetry figure) : 
  false :=
sorry

end no_figure_with_center_and_one_axis_of_symmetry_l771_771773


namespace remainder_3_pow_2040_mod_11_l771_771697

theorem remainder_3_pow_2040_mod_11 : (3 ^ 2040) % 11 = 1 := by
  have h1 : 3 % 11 = 3 := by norm_num
  have h2 : (3 ^ 2) % 11 = 9 := by norm_num
  have h3 : (3 ^ 3) % 11 = 5 := by norm_num
  have h4 : (3 ^ 4) % 11 = 4 := by norm_num
  have h5 : (3 ^ 5) % 11 = 1 := by norm_num
  have h_mod : 2040 % 5 = 0 := by norm_num
  sorry

end remainder_3_pow_2040_mod_11_l771_771697


namespace rectangle_area_l771_771179

theorem rectangle_area (b : ℕ) (area_square : ℕ) (length_ratio : ℚ) : area_square = 2025 → length_ratio = 2/5 → ∃ area_rectangle : ℕ, area_rectangle = 18 * b :=
by
  assume h1 : area_square = 2025,
  assume h2 : length_ratio = 2/5,
  let side := Nat.sqrt area_square,
  have r : side = 45 := by sorry,
  let radius := side,
  let length := length_ratio * radius,
  have l : length = 18 := by sorry,
  let area_rectangle := length * b,
  use area_rectangle
  exact rfl

end rectangle_area_l771_771179


namespace mass_of_eight_moles_of_CaO_l771_771334

theorem mass_of_eight_moles_of_CaO : 
  (atomic_mass_Ca : ℝ) (atomic_mass_O : ℝ) (num_moles : ℝ) 
  (h₁ : atomic_mass_Ca = 42) (h₂ : atomic_mass_O = 16) (h₃ : num_moles = 8) : 
  num_moles * (atomic_mass_Ca + atomic_mass_O) = 464 := 
by 
  have molar_mass_CaO : ℝ := atomic_mass_Ca + atomic_mass_O
  have mass : ℝ := num_moles * molar_mass_CaO
  rw [h₁, h₂]
  rw [h₃]
  -- This ultimately simplifies to
  -- 8 * (42 + 16) = 464
  sorry

end mass_of_eight_moles_of_CaO_l771_771334


namespace sum_of_roots_of_cubic_l771_771782

theorem sum_of_roots_of_cubic :
  let p := Polynomial.Cubic 3 (-9) (-48) 8 in
  Polynomial.roots_sum p = 3 :=
by sorry

end sum_of_roots_of_cubic_l771_771782


namespace eval_neg_pow_l771_771382

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l771_771382


namespace sine_function_omega_range_l771_771882

theorem sine_function_omega_range (A ω : ℝ) (hA : A > 0) (hω : ω > 0) :
  (∀ x ∈ set.Icc 0 1, y = A * real.sin (ω * x - π / 3) → y_max_min_count y = (50, 50)) → 
  ω ∈ set.Icc (599 * π / 6) (605 * π / 6) :=
sorry

end sine_function_omega_range_l771_771882


namespace BD_tangent_circumcircle_ADZ_l771_771556

-- Define the triangle ABC with the given properties
variables (A B C D E X Y Z : Type) [real_point A] [real_point B] [real_point C] [real_point D] [real_point E] [real_point X] [real_point Y] [real_point Z]
variables [circle ω A B C]
variables [is_tangent A ω BC D]
variables [reflection_in_line A BC E]
variables [foot_of_perpendicular A BE X]
variables [midpoint_segment AX Y]
variables [intersects_again BY ω Z]

-- Define the proposition we want to prove
theorem BD_tangent_circumcircle_ADZ : tangent BD (circumcircle A D Z) :=
sorry

end BD_tangent_circumcircle_ADZ_l771_771556


namespace non_intersecting_pairs_exists_l771_771851

open Set

theorem non_intersecting_pairs_exists (T : Set (ℝ × ℝ)) (hT : ∃ n, T.card = 2 * n) :
  ∃ P : Set (Set (ℝ × ℝ)), (∀ p ∈ P, p.card = 2) ∧ (P.pairwise Disjoint ∧ P ⊆ T):  sorry

end non_intersecting_pairs_exists_l771_771851


namespace max_number_of_negative_integers_l771_771316

-- Define the context where Alice chooses eight integers
def maxNegatives (l : List Int) : Prop :=
  l.length = 8 ∧ (l.prod % 2 = 0) ∧ (l.count (· < 0) = 8)

-- We need to prove that there exists such a list of integers
theorem max_number_of_negative_integers (l : List Int) (h_len : l.length = 8) (h_even : l.prod % 2 = 0) :
  ∃ l, maxNegatives l := by
  sorry

end max_number_of_negative_integers_l771_771316


namespace shortest_distance_from_origin_l771_771248

noncomputable def shortest_distance_to_circle (x y : ℝ) : ℝ :=
  x^2 + 6 * x + y^2 - 8 * y + 18

theorem shortest_distance_from_origin :
  ∃ (d : ℝ), d = 5 - Real.sqrt 7 ∧ ∀ (x y : ℝ), shortest_distance_to_circle x y = 0 →
    (Real.sqrt ((x - 0)^2 + (y - 0)^2) - Real.sqrt ((x + 3)^2 + (y - 4)^2)) = d := sorry

end shortest_distance_from_origin_l771_771248


namespace max_value_f_no_real_roots_f_l771_771254

def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 10

theorem max_value_f : ∃ x : ℝ, (f x = -2) ∧ (∀ y : ℝ, f y ≤ f x) := 
by 
  use 2
  split
  { simp [f], }
  { sorry }  -- To be completed

theorem no_real_roots_f : ¬ ∃ x : ℝ, f x = 0 := 
by 
  intro h
  cases h with x hx
  have : x^2 - 4 * x + 5 = 0 := 
  by 
    have := calc 
      0 = f x : hx
        ... = -2 * x^2 + 8 * x - 10 : by simp [f]
    linarith
  let d := (-4)^2 - 4 * 1 * 5 
  show false, from by 
    simp [-sub_eq_add_neg, pow_two, mul_assoc, -add_comm, -mul_comm, not_le]
    linarith
  sorry  -- To be completed

end max_value_f_no_real_roots_f_l771_771254


namespace find_f_minus_one_l771_771476

theorem find_f_minus_one (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : a^3 = 8) :
  (a^(-1) = 1/2) :=
sorry

end find_f_minus_one_l771_771476


namespace eval_neg_pow_l771_771376

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l771_771376


namespace box_dimensions_l771_771721

theorem box_dimensions (a b c : ℕ) (h_div1 : ∃ x y z : ℕ, a = x * ∛2 ∧ b = y * ∛2 ∧ c = z * ∛2) :
  (a * b * c = 125) → (a = 2 ∧ b = 5 ∧ c = 6) ∨ (a = 2 ∧ b = 5 ∧ c = 3) :=
by
  sorry

end box_dimensions_l771_771721


namespace problem_statement_l771_771317

-- Define the given functions
def f1 (x : ℝ) := 2^x - 2^(-x)
def f2 (x : ℝ) := x^2 - 1
def f3 (x : ℝ) := Real.log (abs x) / Real.log (1/2)
def f4 (x : ℝ) := x * Real.sin x

-- State the condition that f2 is even
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- State the condition that f2 is monotonically increasing on (0, +∞)
def is_monotonically_increasing_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

-- Define the set (0, +∞)
def positives := { x : ℝ | 0 < x }

-- Proof statement
theorem problem_statement :
  (is_even f1 → is_monotonically_increasing_on f1 positives → False) →
  (is_even f2 ∧ is_monotonically_increasing_on f2 positives) →
  (is_even f3 → is_monotonically_increasing_on f3 positives → False) →
  (is_even f4 → is_monotonically_increasing_on f4 positives → False) :=
by sorry

end problem_statement_l771_771317


namespace time_for_a_alone_l771_771705

theorem time_for_a_alone
  (b_work_time : ℕ := 20)
  (c_work_time : ℕ := 45)
  (together_work_time : ℕ := 72 / 10) :
  ∃ (a_work_time : ℕ), a_work_time = 15 :=
by
  sorry

end time_for_a_alone_l771_771705


namespace people_didnt_show_up_l771_771727

theorem people_didnt_show_up (invited people_per_table num_tables seated : ℕ) (h1 : invited = 18) 
  (h2 : people_per_table = 3) (h3 : num_tables = 2) (showed_up : ℕ) (h4 : showed_up = people_per_table * num_tables) : invited - showed_up = 12 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end people_didnt_show_up_l771_771727


namespace triangle_inequality_l771_771438

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a^2 + 2 * b * c) / (b^2 + c^2) + (b^2 + 2 * a * c) / (c^2 + a^2) + (c^2 + 2 * a * b) / (a^2 + b^2) > 3 :=
by
  sorry

end triangle_inequality_l771_771438


namespace seven_books_cost_l771_771203

-- Given condition: Three identical books cost $45
def three_books_cost (cost_per_book : ℤ) := 3 * cost_per_book = 45

-- Question to prove: The cost of seven identical books is $105
theorem seven_books_cost (cost_per_book : ℤ) (h : three_books_cost cost_per_book) : 7 * cost_per_book = 105 := 
sorry

end seven_books_cost_l771_771203


namespace find_two_digit_numbers_l771_771802

theorem find_two_digit_numbers :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → (10 * a + b = 3 * a * b) → (10 * a + b = 15 ∨ 10 * a + b = 24) :=
by
  intros
  sorry

end find_two_digit_numbers_l771_771802


namespace find_a_and_roots_of_abs_quadratic_eqn_l771_771876

theorem find_a_and_roots_of_abs_quadratic_eqn :
  ∃ (a : ℝ) (r1 r2 r3 : ℝ),
    (|x^2 + a * x| = 4) ∧
    (set.has_size {r1, r2, r3} 3) ∧
    (a = 4 ∧ {r1, r2, r3} = {-2, -2 + sqrt 2, -2 - sqrt 2}) ∨
    (a = -4 ∧ {r1, r2, r3} = {2, 2 + sqrt 2, 2 - sqrt 2}) :=
sorry

end find_a_and_roots_of_abs_quadratic_eqn_l771_771876


namespace functions_odd_or_not_l771_771262

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem functions_odd_or_not :
  (¬ is_odd (λ x, |x|)) ∧
  is_odd (λ x, x + 1/x) ∧
  is_odd (λ x, x^3 + 2*x) ∧
  (¬ is_odd (λ x, x^2 + x + 1)) :=
by sorry

end functions_odd_or_not_l771_771262


namespace Lana_muffin_goal_l771_771108

theorem Lana_muffin_goal (total_muffins : ℕ) 
  (early_percent late_percent afternoon_percent spoiled_percent : ℕ) 
  (early_sold late_sold afternoon_sold spoiled_sold : ℕ) :
  total_muffins = 120 →
  early_percent = 25 → 
  late_percent = 15 → 
  afternoon_percent = 10 → 
  spoiled_percent = 20 → 
  early_sold = total_muffins * early_percent / 100 →
  late_sold = total_muffins * late_percent / 100 →
  afternoon_sold = total_muffins * afternoon_percent / 100 →
  spoiled_sold = (total_muffins - (early_sold + late_sold + afternoon_sold)) * spoiled_percent / 100 →
  total_muffins - (early_sold + late_sold + afternoon_sold + spoiled_sold) = 48 :=
begin
  sorry
end

end Lana_muffin_goal_l771_771108


namespace lcm_18_45_l771_771694

theorem lcm_18_45 : Int.lcm 18 45 = 90 :=
by
  -- Prime factorizations
  have h1 : Nat.factors 18 = [2, 3, 3] := by sorry
  have h2 : Nat.factors 45 = [3, 3, 5] := by sorry
  
  -- Calculate LCM
  rw [←Int.lcm_def, Nat.factors_mul, List.perm.ext']
  apply List.Permutation.sublist
  sorry

end lcm_18_45_l771_771694


namespace machine_shirts_today_l771_771323

theorem machine_shirts_today :
  ∀ (rate_per_minute minutes_today : ℕ), rate_per_minute = 6 → minutes_today = 12 → rate_per_minute * minutes_today = 72 :=
by
  intros rate_per_minute minutes_today hrate hminutes
  rw [hrate, hminutes]
  exact Nat.mul_eq_one (6 * 12) sorry

end machine_shirts_today_l771_771323


namespace population_growth_proof_l771_771068

noncomputable def population_growth (P0 : ℕ) (P200 : ℕ) (t : ℕ) (x : ℝ) : Prop :=
  P200 = P0 * (1 + 1 / x)^t

theorem population_growth_proof :
  population_growth 6 1000000 200 16 :=
by
  -- Proof goes here
  sorry

end population_growth_proof_l771_771068


namespace log_property_l771_771507

theorem log_property (m y : ℝ)
  (h : log m y * log 7 m = 4) : y = 2401 := 
sorry

end log_property_l771_771507


namespace lcm_12_21_30_l771_771812

theorem lcm_12_21_30 : Nat.lcm (Nat.lcm 12 21) 30 = 420 := by
  sorry

end lcm_12_21_30_l771_771812


namespace basketball_probability_l771_771676

variable (A B : Event)
variable (P : Probability)
variable (P_A : P A = 0.8)
variable (P_B' : P (¬ B) = 0.1)
variable (ind : independent A B)

theorem basketball_probability :
  (P (A ∩ B) = 0.72) ∧ 
  (P (A ∩ (¬ B)) + P ((¬ A) ∩ B) = 0.26) :=
by
  sorry

end basketball_probability_l771_771676


namespace cricket_target_runs_l771_771949

theorem cricket_target_runs:
  let run_rate_first = 3.8
  let overs_first = 10
  let run_rate_rest = 6.1
  let overs_rest = 40
  (run_rate_first * overs_first + run_rate_rest * overs_rest) = 282 :=
by
  sorry

end cricket_target_runs_l771_771949


namespace monotonic_decreasing_interval_l771_771185

noncomputable def y (x : ℝ) : ℝ := x * log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < real.exp (-1) → deriv y x < 0 :=
by
  intros x h
  have h1 : deriv y x = log x + 1 := sorry
  have h2 : log x + 1 < 0 := sorry
  rw [h1]
  exact h2

end monotonic_decreasing_interval_l771_771185


namespace total_gain_is_19200_l771_771752

noncomputable def total_annual_gain_of_partnership (x : ℝ) (A_share : ℝ) (B_investment_after : ℕ) (C_investment_after : ℕ) : ℝ :=
  let A_investment_time := 12
  let B_investment_time := 12 - B_investment_after
  let C_investment_time := 12 - C_investment_after
  let proportional_sum := x * A_investment_time + 2 * x * B_investment_time + 3 * x * C_investment_time
  let individual_proportion := proportional_sum / A_investment_time
  3 * A_share

theorem total_gain_is_19200 (x A_share : ℝ) (B_investment_after C_investment_after : ℕ) :
  A_share = 6400 →
  B_investment_after = 6 →
  C_investment_after = 8 →
  total_annual_gain_of_partnership x A_share B_investment_after C_investment_after = 19200 :=
by
  intros hA hB hC
  have x_pos : x > 0 := by sorry   -- Additional assumptions if required
  have A_share_pos : A_share > 0 := by sorry -- Additional assumptions if required
  sorry

end total_gain_is_19200_l771_771752


namespace odd_functions_B_and_C_l771_771260

def f_A (x : ℝ) : ℝ := abs x
def f_B (x : ℝ) : ℝ := x + 1 / x
def f_C (x : ℝ) : ℝ := x^3 + 2 * x
def f_D (x : ℝ) : ℝ := x^2 + x + 1

theorem odd_functions_B_and_C :
  (∀ x : ℝ, f_B (-x) = -f_B x) ∧ (∀ x : ℝ, f_C (-x) = -f_C x) :=
by
  sorry

end odd_functions_B_and_C_l771_771260


namespace largest_prime_factor_720_l771_771243

def largest_prime_factor (n : ℕ) : ℕ :=
  if n = 1 then 1
  else Nat.factorization n |>.max'.get (Nat.factorization n).nonempty_of_ne_zero n

theorem largest_prime_factor_720 : largest_prime_factor 720 = 5 := by
  sorry

end largest_prime_factor_720_l771_771243


namespace geometric_sequence_problem_l771_771950

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_problem 
  (h : geometric_sequence a) 
  (h_a20_a60_roots : ∀ x ∈ ({a 20, a 60} : set ℝ), x^2 - 10 * x + 16 = 0) :
  (a 30 * a 40 * a 50) / 2 = 32 :=
sorry

end geometric_sequence_problem_l771_771950


namespace parallelogram_area_l771_771807

theorem parallelogram_area (base : ℝ) (height_mm : ℝ) (inch_to_mm : ℝ) : 
  inch_to_mm = 25.4 → 
  base = 18 → 
  height_mm = 25.4 → 
  (base * (height_mm / inch_to_mm) = 18) :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end parallelogram_area_l771_771807


namespace inequality_solution_l771_771607

theorem inequality_solution (x : ℝ) (h : x > -4/3) : 2 - 1 / (3 * x + 4) < 5 :=
sorry

end inequality_solution_l771_771607


namespace max_omega_value_l771_771033

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x + φ)

theorem max_omega_value 
  (ω : ℝ) 
  (φ : ℝ) 
  (hω : 0 < ω) 
  (hφ : |φ| ≤ Real.pi / 2)
  (h_zero : f ω φ (-Real.pi / 4) = 0)
  (h_sym : f ω φ (Real.pi / 4) = f ω φ (-Real.pi / 4))
  (h_monotonic : ∀ x₁ x₂, (Real.pi / 18) < x₁ → x₁ < x₂ → x₂ < (5 * Real.pi / 36) → f ω φ x₁ < f ω φ x₂) :
  ω = 9 :=
  sorry

end max_omega_value_l771_771033


namespace sum_cubes_le_square_sum_l771_771310

variable (n : ℕ)
variable (a : ℕ → ℕ)

-- Conditions:
-- 1. Initial term a_0 = 0
def a_0_condition : Prop := a 0 = 0

-- 2. Sequence progression condition
def sequence_condition : Prop := ∀ k : ℕ, k < n → 0 ≤ a (k + 1) - a k ∧ a (k + 1) - a k ≤ 1

-- The theorem to prove
theorem sum_cubes_le_square_sum 
  (a_0_cond : a_0_condition a) 
  (seq_cond : sequence_condition n a) :
  ∑ k in Finset.range n, (a k.succ)^3 ≤ (∑ k in Finset.range n, a k.succ)^2 := 
sorry

end sum_cubes_le_square_sum_l771_771310


namespace find_m_for_divisibility_l771_771517

def sum_digits (d : list ℕ) : ℕ := d.sum

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

theorem find_m_for_divisibility :
  ∃ m : ℕ, sum_digits [9, 7, 3, m, 2, 1, 5, 8] = 36 ∧ is_divisible_by_9 (sum_digits [9, 7, 3, m, 2, 1, 5, 8]) :=
by {
  use 1,
  sorry
}

end find_m_for_divisibility_l771_771517


namespace eval_expression_correct_l771_771401

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l771_771401


namespace cookies_remaining_percentage_l771_771201

theorem cookies_remaining_percentage (c : ℕ) (p_n p_e : ℚ) (h_c : c = 600) (h_nicole : p_n = 2/5) (h_eduardo : p_e = 3/5) :
  ∃ p_r : ℚ, p_r = 24 :=
by
  have h_nicole_cookies : ℚ := p_n * c
  have h_remaining_after_nicole : ℚ := c - h_nicole_cookies
  have h_eduardo_cookies : ℚ := p_e * h_remaining_after_nicole
  have h_remaining_after_eduardo : ℚ := h_remaining_after_nicole - h_eduardo_cookies
  have h_percentage_remaining : ℚ := (h_remaining_after_eduardo / c) * 100
  use h_percentage_remaining
  field_simp [h_c, h_nicole, h_eduardo]
  norm_num
  sorry

end cookies_remaining_percentage_l771_771201


namespace probability_heart_then_club_l771_771666

noncomputable def numHearts : ℕ := 13
noncomputable def numClubs : ℕ := 13
noncomputable def totalCards (n : ℕ) : ℕ := 52 - n

noncomputable def probabilityFirstHeart : ℚ := numHearts / totalCards 0
noncomputable def probabilitySecondClubGivenFirstHeart : ℚ := numClubs / totalCards 1

theorem probability_heart_then_club :
  (probabilityFirstHeart * probabilitySecondClubGivenFirstHeart) = 13 / 204 :=
by
  sorry

end probability_heart_then_club_l771_771666


namespace prove_sum_l771_771825

-- Given conditions:
def positive_integers (c d : ℕ) : Prop :=
  1 ≤ c ∧ 1 ≤ d

def product_of_logs (c d : ℕ) : Prop :=
  ∏ i in finset.range (d - c), real.log_base (c + ↑i) (c + ↑i + 1) = 3

def product_length (c d : ℕ) : Prop :=
  d - c = 839

-- Proof problem:
theorem prove_sum (c d : ℕ) (h1 : positive_integers c d) (h2 : product_of_logs c d) (h3: product_length c d) : c + d = 1010 := 
sorry

end prove_sum_l771_771825


namespace ratio_of_place_values_l771_771089

-- Definitions based on conditions
def place_value_tens_digit : ℝ := 10
def place_value_hundredths_digit : ℝ := 0.01

-- Statement to prove
theorem ratio_of_place_values :
  (place_value_tens_digit / place_value_hundredths_digit) = 1000 :=
by
  sorry

end ratio_of_place_values_l771_771089


namespace loan_difference_is_979_l771_771756

noncomputable def compounded_interest (P r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

noncomputable def loan_difference (P : ℝ) : ℝ :=
  let compounded_7_years := compounded_interest P 0.08 12 7
  let half_payment := compounded_7_years / 2
  let remaining_balance := compounded_interest half_payment 0.08 12 8
  let total_compounded := half_payment + remaining_balance
  let total_simple := simple_interest P 0.10 15
  abs (total_compounded - total_simple)

theorem loan_difference_is_979 : loan_difference 15000 = 979 := sorry

end loan_difference_is_979_l771_771756


namespace fraction_sum_is_correct_fraction_sum_percentage_l771_771280

def fraction_sum : ℚ :=
  (4/20) + (8/200) + (12/2000)

theorem fraction_sum_is_correct :
  fraction_sum = 123/500 := by
  sorry

theorem fraction_sum_percentage :
  (fraction_sum * 100 : ℚ) = 24.6 :=
  by
  calc 
    fraction_sum * 100 = (123/500) * 100 : by rw [fraction_sum_is_correct]
                  ... = 24.6            : by norm_num

end fraction_sum_is_correct_fraction_sum_percentage_l771_771280


namespace riverside_high_badges_l771_771829

/-- Given the conditions on the sums of consecutive prime badge numbers of the debate team members,
prove that Giselle's badge number is 1014, given that the current year is 2025.
-/
theorem riverside_high_badges (p1 p2 p3 p4 : ℕ) (hp1 : Prime p1) (hp2 : Prime p2) (hp3 : Prime p3) (hp4 : Prime p4)
    (hconsec : p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 = p3 + 6)
    (h1 : ∃ x, p1 + p3 = x) (h2 : ∃ y, p1 + p2 = y) (h3 : ∃ z, p2 + p3 = z ∧ z ≤ 31) 
    (h4 : p3 + p4 = 2025) : p4 = 1014 :=
by sorry

end riverside_high_badges_l771_771829


namespace find_alpha_minus_beta_find_cos_2alpha_minus_beta_l771_771944

-- Definitions and assumptions
variables (α β : ℝ)
axiom sin_alpha : Real.sin α = (Real.sqrt 5) / 5
axiom sin_beta : Real.sin β = (3 * Real.sqrt 10) / 10
axiom alpha_acute : 0 < α ∧ α < Real.pi / 2
axiom beta_acute : 0 < β ∧ β < Real.pi / 2

-- Statement to prove α - β = -π/4
theorem find_alpha_minus_beta : α - β = -Real.pi / 4 :=
sorry

-- Given α - β = -π/4, statement to prove cos(2α - β) = 3√10 / 10
theorem find_cos_2alpha_minus_beta (h : α - β = -Real.pi / 4) : Real.cos (2 * α - β) = (3 * Real.sqrt 10) / 10 :=
sorry

end find_alpha_minus_beta_find_cos_2alpha_minus_beta_l771_771944


namespace gcd_polynomial_l771_771464

theorem gcd_polynomial (b : ℤ) (h1 : ∃ k : ℤ, b = 7 * k ∧ k % 2 = 1) : 
  Int.gcd (3 * b ^ 2 + 34 * b + 76) (b + 16) = 7 := 
sorry

end gcd_polynomial_l771_771464


namespace general_term_a_sum_terms_c_l771_771471

-- Defining the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := n

-- Given conditions
axiom a_1_eq : a 1 = 1
axiom a_geom : ∀ n : ℕ, n ≠ 0 ∧ n ≠ 1 ∧ n ≠ 3 ∧ n ≠ 7 → a (n+1)^2 = a 1 * a (n+7)

-- Defining the sequence {b_n}
axiom sum_b : ∀ n : ℕ, (finset.range (n + 1)).sum (λ i, (a (i + 1)) * b (i + 1)) = 2^(n + 1)

-- Defining the sequence {c_n}
def c (n : ℕ) : ℚ := b (n+1) / 2^(n+1)

-- Prove the general term formula for {a_n}
theorem general_term_a : ∀ n : ℕ, ¬(a n = n) := sorry

-- Prove the sum of the first n terms of {c_nc_{n+1}}
theorem sum_terms_c : ∀ n : ℕ, (finset.range (n+1)).sum (λ i, c i * c (i+1)) = n / (2*(n+2)) := sorry

end general_term_a_sum_terms_c_l771_771471


namespace cartesian_eq_1_cartesian_eq_2_cartesian_eq_3_l771_771784

-- Definitions for converting polar coordinates to Cartesian coordinates
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * θ.cos, ρ * θ.sin)

-- Theorem for the first equation
theorem cartesian_eq_1 (ρ θ : ℝ) (h : ρ = θ.cos + 2 * θ.sin) :
  (polar_to_cartesian ρ θ).1^2 + (polar_to_cartesian ρ θ).2^2 = (polar_to_cartesian ρ θ).1 + 2 * (polar_to_cartesian ρ θ).2 :=
sorry

-- Theorem for the second equation
theorem cartesian_eq_2 (ρ θ : ℝ) (h : ρ = 1 + θ.sin) :
  (polar_to_cartesian ρ θ).1^2 + (polar_to_cartesian ρ θ).2^2 = ((polar_to_cartesian ρ θ).1^2 + (polar_to_cartesian ρ θ).2^2).sqrt + (polar_to_cartesian ρ θ).2 :=
sorry

-- Theorem for the third equation
theorem cartesian_eq_3 (ρ θ : ℝ) (h : ρ^3 * θ.sin * (2 * θ).cos = ρ^2 * (2 * θ).cos - ρ * θ.sin + 1) :
  (polar_to_cartesian ρ θ).2 = 1 ∨ (polar_to_cartesian ρ θ).1 = (polar_to_cartesian ρ θ).2 ∨ (polar_to_cartesian ρ θ).1 = -(polar_to_cartesian ρ θ).2 :=
sorry

end cartesian_eq_1_cartesian_eq_2_cartesian_eq_3_l771_771784


namespace seating_arrangement_l771_771758

/-- Alice refuses to sit next to either Bob or Carla.
    Derek refuses to sit next to Eric or Paul.
    Paul also refuses to sit next to Carla.
    Prove that there are exactly 72 ways for Alice, Bob, Carla, Derek, Eric and Paul
    to sit in a row of 6 chairs under these conditions. -/
theorem seating_arrangement :
  let persons := ["Alice", "Bob", "Carla", "Derek", "Eric", "Paul"] in
  let conditions (arrangement : List String) : Prop :=
    -- Conditions for Alice
    ¬ ((arrangement.index_of "Alice" - arrangement.index_of "Bob").abs = 1) ∧
    ¬ ((arrangement.index_of "Alice" - arrangement.index_of "Carla").abs = 1) ∧
    -- Conditions for Derek
    ¬ ((arrangement.index_of "Derek" - arrangement.index_of "Eric").abs = 1) ∧
    ¬ ((arrangement.index_of "Derek" - arrangement.index_of "Paul").abs = 1) ∧
    -- Conditions for Paul
    ¬ ((arrangement.index_of "Paul" - arrangement.index_of "Carla").abs = 1) in
  (List.permutations persons).count conditions = 72 :=
sorry

end seating_arrangement_l771_771758


namespace right_triangle_sum_of_legs_eq_sum_of_diameters_l771_771596

variables {R r a b c : ℝ}
variable (h_right_triangle : c = Math.sqrt (a^2 + b^2))
variable (h_circum_radius : c = 2 * R)
variable (h_in_radius : r = (a + b - c) / 2)

theorem right_triangle_sum_of_legs_eq_sum_of_diameters :
  2 * R + 2 * r = a + b :=
by
  have h1 : c = a + b - 2 * r, from sorry,
  rw [h_circum_radius] at h1,
  linarith

end right_triangle_sum_of_legs_eq_sum_of_diameters_l771_771596


namespace fifth_term_expansion_l771_771160

theorem fifth_term_expansion (a x : ℝ) : 
  let term := Nat.choose 7 4 * ((a / x^2) ^ (7 - 4)) * ((-x / a^3) ^ 4)
  in term = 35 / (x^2 * a^9) :=
by sorry

end fifth_term_expansion_l771_771160


namespace A_strictly_decreasing_H_strictly_decreasing_l771_771976

noncomputable def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)
noncomputable def harmonic_mean (a b : ℝ) : ℝ := (2 * a * b) / (a + b)
noncomputable def power_mean_2 (a b : ℝ) : ℝ := Real.sqrt ((a^2 + b^2) / 2)

def decreasing_seq (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → f n > f (n + 1)

variables (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)

-- Definitions for the sequence terms
noncomputable def A : ℕ → ℝ
| 1 := arithmetic_mean x y
| (n+2) := arithmetic_mean (power_mean_2 (A (n+1)) (H (n+1))) (H (n+1))

noncomputable def G : ℕ → ℝ
| 1 := geometric_mean x y
| (n+2) := geometric_mean (power_mean_2 (A (n+1)) (H (n+1))) (H (n+1))

noncomputable def H : ℕ → ℝ
| 1 := harmonic_mean x y
| (n+2) := harmonic_mean (power_mean_2 (A (n+1)) (H (n+1))) (H (n+1))

noncomputable def P : ℕ → ℝ
| 1 := power_mean_2 x y
| (n+2) := power_mean_2 (P (n+1)) (H (n+1))

theorem A_strictly_decreasing : decreasing_seq (A x y hx hy hxy) :=
sorry

theorem H_strictly_decreasing : decreasing_seq (H x y hx hy hxy) :=
sorry

end A_strictly_decreasing_H_strictly_decreasing_l771_771976


namespace distance_between_points_l771_771235

-- Define the two points.
def point1 : ℝ × ℝ := (-3, 5)
def point2 : ℝ × ℝ := (4, -9)

-- Define the distance formula.
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- State the theorem.
theorem distance_between_points :
  distance point1 point2 = real.sqrt 245 :=
by
  -- Placeholder for the proof.
  sorry

end distance_between_points_l771_771235


namespace increased_expenses_percent_l771_771737

theorem increased_expenses_percent (S : ℝ) (hS : S = 6250) (initial_save_percent : ℝ) (final_savings : ℝ) 
  (initial_save_percent_def : initial_save_percent = 20) 
  (final_savings_def : final_savings = 250) : 
  (initial_save_percent / 100 * S - final_savings) / (S - initial_save_percent / 100 * S) * 100 = 20 := by
  sorry

end increased_expenses_percent_l771_771737


namespace remaining_leaves_l771_771750

def initial_leaves := 1000
def first_week_shed := (2 / 5 : ℚ) * initial_leaves
def leaves_after_first_week := initial_leaves - first_week_shed
def second_week_shed := (40 / 100 : ℚ) * leaves_after_first_week
def leaves_after_second_week := leaves_after_first_week - second_week_shed
def third_week_shed := (3 / 4 : ℚ) * second_week_shed
def leaves_after_third_week := leaves_after_second_week - third_week_shed

theorem remaining_leaves (initial_leaves first_week_shed leaves_after_first_week second_week_shed leaves_after_second_week third_week_shed leaves_after_third_week: ℚ) : 
  leaves_after_third_week = 180 := by
  sorry

end remaining_leaves_l771_771750


namespace flu_infection_l771_771304

theorem flu_infection (x : ℕ) (H : 1 + x + x^2 = 36) : True :=
begin
  sorry
end

end flu_infection_l771_771304


namespace solve_inequality_l771_771609

noncomputable def solution_set : Set ℝ := {x | x < -4/3 ∨ x > -13/9}

theorem solve_inequality (x : ℝ) : 
  2 - 1 / (3 * x + 4) < 5 → x ∈ solution_set :=
by
  sorry

end solve_inequality_l771_771609


namespace clock_correct_time_fraction_l771_771730

theorem clock_correct_time_fraction :
  let hours := 24
  let incorrect_hours := 6
  let correct_hours_fraction := (hours - incorrect_hours) / hours
  let minutes_per_hour := 60
  let incorrect_minutes_per_hour := 15
  let correct_minutes_fraction := (minutes_per_hour - incorrect_minutes_per_hour) / minutes_per_hour
  correct_hours_fraction * correct_minutes_fraction = (9 / 16) :=
by 
  sorry

end clock_correct_time_fraction_l771_771730


namespace max_BP_squared_l771_771575

-- Given definitions from the conditions
def diameter (A B : Point) (ω : Circle) : Prop := is_diameter A B
def extended_line (A B C : Point) : Prop := on_line A B C
def tangent (C T : Point) (ω : Circle) : Prop := is_tangent C T ω
def perpendicular_foot (A P C T : Point) : Prop := is_perpendicular_foot A P C T
def AB_length (A B : Point) : ℝ := 20

-- Math proof problem
theorem max_BP_squared (A B C T P : Point) (ω : Circle)
  (h1 : diameter A B ω)
  (h2 : extended_line A B C)
  (h3 : tangent C T ω)
  (h4 : perpendicular_foot A P C T)
  (h5 : AB_length A B = 20) :
  let n := max_BP A B C T P ω
  n^2 = 405 :=
  sorry

-- Helper function to capture the max length
noncomputable def max_BP (A B C T P : Point) (ω : Circle) : ℝ :=
  -- This is just a placeholder to demonstrate the Lean structure
  20 * (√20 + 1 / √20)

end max_BP_squared_l771_771575


namespace num_integers_with_digit_sum_seventeen_l771_771898

-- Define a function to calculate the sum of the digits of an integer
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (λ acc d, acc + d) 0

-- Define the main proposition
theorem num_integers_with_digit_sum_seventeen : 
  (finset.filter (λ n, sum_of_digits n = 17) (finset.Icc 400 600)).card = 13 := 
sorry

end num_integers_with_digit_sum_seventeen_l771_771898


namespace polynomial_degree_rat_coeffs_l771_771299

theorem polynomial_degree_rat_coeffs (P : Polynomial ℚ)
  (roots : ∀ n : ℕ, n > 0 ∧ n ≤ 500 → P.eval (n + real.sqrt (2 * n + 1)) = 0) :
  P.degree.to_nat = 1000 :=
sorry

end polynomial_degree_rat_coeffs_l771_771299


namespace ratio_of_areas_l771_771083

variables (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables (AB DC CB AD: ℝ)
variables (x : ℝ)

def is_rectangle (A B C D : Type) : Prop := sorry
def bisects (ED : Type) (E D C : Type) : Prop := sorry
def area_of_triangle (DEF: Type) : ℝ := sorry
def area_of_rectangle (ABCD: Type) : ℝ := sorry

theorem ratio_of_areas (h1 : is_rectangle A B C D) (h2 : DC = 3 * CB)
  (h3 : ∃ E F : Type, E ∈ segment A B ∧ F ∈ segment A B ∧ bisects (E D) D C) :
  area_of_triangle E D F / area_of_rectangle A B C D = 1 / (4 * sqrt 2) :=
sorry

end ratio_of_areas_l771_771083


namespace gain_percent_l771_771270

theorem gain_percent (C S : ℝ) (h : 50 * C = 32 * S) : (S - C) / C * 100 = 56.25 := by
  have h₀ : S = 50 * C / 32 := by sorry
  have h₁ : (S - C) = 9 * C / 16 := by sorry
  have h₂ : (S - C) / C = 9 / 16 := by sorry
  calc
    (S - C) / C * 100
        = (9 / 16) * 100 : by sorry
    ... = 56.25 : by norm_num

end gain_percent_l771_771270


namespace geometric_relationship_l771_771091

-- Definitions of geometric entities and properties involved
variables {A B C O E F : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O] [MetricSpace E] [MetricSpace F]

-- Geometric specifics of the problem
variable (triangle : Triangle)
variable (hypotenuse : Hypotenuse)
variable (midpoint : Midpoint)
variable (perpendicular : Perpendicular)

-- Conditions as attributes of the triangle and points
axiom right_angle : angle A B C = 90
axiom O_midpoint : is_midpoint O B C
axiom E_perpendicular : is_perpendicular O E B
axiom F_perpendicular : is_perpendicular O F C

-- The proof statement
theorem geometric_relationship (triangle : Triangle A B C) (hypotenuse : Hypotenuse B C)
  (midpoint : Midpoint O B C) (perpendicular : Perpendicular O E B) (perpendicular' : Perpendicular O F C) :
  (distance O E) * (distance O F) = (distance A O) ^ 2 :=
sorry

end geometric_relationship_l771_771091


namespace tagged_fish_in_second_catch_l771_771074

def fish_catch_problem (N T : ℕ) (N_approx : N = 1000) 
  (tagged_first_catch : 40) (second_catch_total : 50) : Prop :=
  (40 / N : ℚ) = (T / 50 : ℚ)

theorem tagged_fish_in_second_catch : ∃ (T : ℕ), fish_catch_problem 1000 T  (by rfl) 40 50 ∧ T = 2 :=
begin
  use 2,
  have h1 : (40 / 1000 : ℚ) = (2 / 50 : ℚ),
  { norm_num },
  have h2 : fish_catch_problem 1000 2 (by rfl) 40 50,
  { unfold fish_catch_problem,
    norm_num },
  exact ⟨h2, rfl⟩,
end

end tagged_fish_in_second_catch_l771_771074


namespace percentage_without_conditions_l771_771081

theorem percentage_without_conditions (total_teachers : ℕ) (hp_teachers : ℕ) (ht_teachers : ℕ) (both_teachers : ℕ) (H1 : total_teachers = 150) (H2 : hp_teachers = 90) (H3 : ht_teachers = 50) (H4 : both_teachers = 30) :
  (total_teachers - (hp_teachers + ht_teachers - both_teachers)) * 100 / total_teachers = 26.67 :=
by
  have neither_teachers := total_teachers - (hp_teachers + ht_teachers - both_teachers)
  have percent_without_conditions := (neither_teachers * 100 : ℚ) / total_teachers
  have approx_eq : abs (percent_without_conditions - 26.67) < 0.01 := sorry
  exact sorry

end percentage_without_conditions_l771_771081


namespace cost_price_of_apple_l771_771742

theorem cost_price_of_apple (SP : ℝ) (h1 : SP = 15) (h2 : ∃ CP, SP = CP - (1/6) * CP) : ∃ CP, CP = 18 :=
by
  have h3 : ∀ CP, 15 = (5/6) * CP → CP = 18
  · intro CP h4
    have h5 : (5/6) * CP = 18
    · rw [h4]
    rw [← mul_left_inj' (ne_of_gt (by norm_num : (5/6 : ℝ) > 0))]
    exact h5
  cases' h2 with CP h6
  use CP
  exact (h3 CP h6)

end cost_price_of_apple_l771_771742


namespace total_running_duration_l771_771722

-- Conditions
def speed1 := 15 -- speed during the first part in mph
def time1 := 3 -- time during the first part in hours
def speed2 := 19 -- speed during the second part in mph
def distance2 := 190 -- distance during the second part in miles

-- Initialize
def distance1 := speed1 * time1 -- distance covered in the first part in miles

def time2 := distance2 / speed2 -- time to cover the distance in the second part in hours

-- Total duration
def total_duration := time1 + time2

-- Proof statement
theorem total_running_duration : total_duration = 13 :=
by
  sorry

end total_running_duration_l771_771722


namespace sam_cycled_distance_l771_771994

theorem sam_cycled_distance (h₁ : Marguerite_distance = 150) (h₂ : Marguerite_time = 3.5) (h₃ : Sam_time = 4) :
  let marguerite_speed := Marguerite_distance / Marguerite_time in
  let sam_distance := marguerite_speed * Sam_time in
  sam_distance = 171.428 := 
by
  let marguerite_speed := h₁ / h₂
  let sam_distance := marguerite_speed * h₃
  have h₄ : marguerite_speed = 42.857 := sorry
  have h₅ : sam_distance = 171.428 := sorry
  exact h₅

end sam_cycled_distance_l771_771994


namespace solve_inequality_l771_771608

noncomputable def solution_set : Set ℝ := {x | x < -4/3 ∨ x > -13/9}

theorem solve_inequality (x : ℝ) : 
  2 - 1 / (3 * x + 4) < 5 → x ∈ solution_set :=
by
  sorry

end solve_inequality_l771_771608


namespace cubic_polynomial_unique_l771_771810

-- Define the polynomial q(x)
def q (x : ℝ) : ℝ := -x^3 + 4*x^2 - 7*x - 4

-- State the conditions
theorem cubic_polynomial_unique :
  q 1 = -8 ∧
  q 2 = -10 ∧
  q 3 = -16 ∧
  q 4 = -32 :=
by
  -- Expand the function definition for the given inputs.
  -- Add these expansions in the proof part.
  sorry

end cubic_polynomial_unique_l771_771810


namespace circle_equation_problem_l771_771466

theorem circle_equation_problem :
  ∃ C2 : ℝ × ℝ → ℝ,
    (∀ p : ℝ × ℝ, C2 p = (p.1-4)^2 + (p.2-4)^2 - 16 
                ∨ C2 p = (p.1+2)^2 + (p.2+2)^2 - 4 
                ∨ C2 p = (p.1 - 2*sqrt 2)^2 + (p.2 + 2*sqrt 2)^2 - 8 
                ∨ C2 p = (p.1 + 2*sqrt 2)^2 + (p.2 - 2*sqrt 2)^2 - 8) :=
begin
  sorry
end

end circle_equation_problem_l771_771466


namespace range_a_l771_771841

theorem range_a (f : ℝ → ℝ)
  (diff : ∀ x, differentiable ℝ f)
  (h1 : ∀ x, f(x) - f(-x) = 2 * x)
  (h2 : ∀ x : ℝ, 0 < x → deriv f x > 1) :
  (∀ a, f(a) - f(1-a) ≥ 2*a - 1) → ∀ a, a ≥ 1/2 :=
by sorry

end range_a_l771_771841


namespace two_digit_numbers_equal_three_times_product_of_digits_l771_771805

theorem two_digit_numbers_equal_three_times_product_of_digits :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 3 * a * b} = {15, 24} :=
by
  sorry

end two_digit_numbers_equal_three_times_product_of_digits_l771_771805


namespace monotonically_decreasing_interval_area_of_triangle_ABC_l771_771894

namespace Problem

-- Define the vectors
def vec_a (x : Real) : Real × Real := (Real.sin x, -1)
def vec_b (x : Real) : Real × Real := (Real.sqrt 3 * Real.cos x, -0.5)

-- Define the function f(x)
def f (x : Real) := let a := vec_a x; let b := vec_b x; (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 - 2

-- Problem 1: Monotonically decreasing interval
theorem monotonically_decreasing_interval (x : Real) (h : 0 ≤ x ∧ x ≤ Real.pi) :
  x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 6) → ∀ y ∈ Set.Icc 0 x, f y ≤ f x := sorry

-- Problem 2: Area of triangle ABC
-- Given conditions
constant A : Real
axiom h_A_acute : 0 < A ∧ A < Real.pi / 2
constant a : Real := 2 * Real.sqrt 3
constant c : Real := 4
axiom A_acute : A ∈ Set.Icc 0 (Real.pi / 2)

theorem area_of_triangle_ABC (h_fA : f A = 1) (h_A_acute : 0 < A ∧ A < Real.pi / 2) (a_eq : a = 2 * Real.sqrt 3) (c_eq : c = 4) :
  let b := c / 2
  let area := (a * b * Real.sin A) / 2
  area = 2 * Real.sqrt 3 := sorry

end Problem

end monotonically_decreasing_interval_area_of_triangle_ABC_l771_771894


namespace inequality_solution_l771_771418

theorem inequality_solution {x : ℝ} :
  {x | (2 * x - 8) * (x - 4) / x ≥ 0} = {x | x < 0} ∪ {x | x > 0} :=
by
  sorry

end inequality_solution_l771_771418


namespace root_interval_range_l771_771932

def f (a x : ℝ) := 2^x + a^(2*x) - 2*a

theorem root_interval_range (a : ℝ) (h₀ : ∃ x ∈ (0, 1), f a x = 0) : 
  (2a - 1) > 0 :=
  sorry

end root_interval_range_l771_771932


namespace problem_statement_l771_771119

def T : Finset ℕ := {n | ∃ (j k : ℕ), 0 ≤ j ∧ j < k ∧ k ≤ 19 ∧ n = 2^j + 2^k ∧ n < 2^20}.toFinset

def count_divisible_by_5 (s : Finset ℕ) : ℕ := s.filter (λ x, x % 5 = 0).card

/-- Set T contains integers between 1 and 2^20 whose binary expansions have exactly two 1's.
    The probability of randomly choosing a number from T that is divisible by 5 is p/q,
    where p and q are relatively prime. We aim to show that p + q = 199. -/
theorem problem_statement : (count_divisible_by_5 T : ℚ) / T.card = 9 / 190 ∧ Nat.gcd 9 190 = 1 → 9 + 190 = 199 :=
by sorry

end problem_statement_l771_771119


namespace smallest_n_and_ratio_l771_771351

theorem smallest_n_and_ratio (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a + b * complex.I)^n = - (a - b * complex.I)^n → n = 4 ∧ b / a = 1 :=
by
  sorry

end smallest_n_and_ratio_l771_771351


namespace odd_terms_in_expansion_of_a_plus_b_to_8_l771_771509

theorem odd_terms_in_expansion_of_a_plus_b_to_8 (a b : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) :
  let terms := list.map (fun k => binom 8 k * a^k * b^(8 - k)) (list.range 9) in
  list.countp (fun x => x % 2 = 1) terms = 2 :=
sorry

end odd_terms_in_expansion_of_a_plus_b_to_8_l771_771509


namespace chess_tournament_l771_771621

theorem chess_tournament (m p k n : ℕ) 
  (h1 : m * 9 = p * 6) 
  (h2 : m * n = k * 8) 
  (h3 : p * 2 = k * 6) : 
  n = 4 := 
by 
  sorry

end chess_tournament_l771_771621


namespace translated_coordinates_of_B_l771_771086

-- Definitions and conditions
def pointA : ℝ × ℝ := (-2, 3)

def translate_right (x : ℝ) (units : ℝ) : ℝ := x + units
def translate_down (y : ℝ) (units : ℝ) : ℝ := y - units

-- Theorem statement
theorem translated_coordinates_of_B :
  let Bx := translate_right (-2) 3
  let By := translate_down 3 5
  (Bx, By) = (1, -2) :=
by
  -- This is where the proof would go, but we're using sorry to skip the proof steps.
  sorry

end translated_coordinates_of_B_l771_771086


namespace combinations_with_repetition_l771_771815

theorem combinations_with_repetition (n k: ℕ) : 
  (∑ (x : Fin (n+k-1+1)), function.injective x) = nat.choose (n + k - 1) k := 
begin
  sorry
end

end combinations_with_repetition_l771_771815


namespace eval_neg_pow_l771_771383

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l771_771383


namespace inequality_proof_l771_771839

open Real

variable {n : ℕ}
variable {x : Fin n → ℝ}

theorem inequality_proof (h1 : ∀ i, 0 < x i) (h2 : 2 ≤ n) (h3 : ∑ i, x i = 1) :
    ∑ i, x i / sqrt (1 - x i) ≥ (∑ i, sqrt (x i)) / sqrt (n - 1) := 
sorry

end inequality_proof_l771_771839


namespace cut_tromino_reassemble_to_square_l771_771599

-- Define the T-shaped tromino composed of three squares of equal size.
structure TShapedTromino where
  square_size : ℕ
  nonempty : square_size > 0

-- Definition of what it means to be reassembled into a square
def can_reassemble_into_square (parts : List (TShapedTromino)) : Prop :=
  ∃ (side : ℕ), (parts : List (λ (tromino : T-shaped Tromino), tromino.square_size)).sum = side * side

-- The main theorem statement
theorem cut_tromino_reassemble_to_square (t : TShapedTromino) : 
  ∃ parts : List TShapedTromino, can_reassemble_into_square parts :=
sorry

end cut_tromino_reassemble_to_square_l771_771599


namespace common_divisor_greater_than_1_l771_771451
open Nat

theorem common_divisor_greater_than_1 (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_ab : (a + b) ∣ (a * b)) (h_bc : (b + c) ∣ (b * c)) (h_ca : (c + a) ∣ (c * a)) :
    ∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ b ∧ k ∣ c := 
by
  sorry

end common_divisor_greater_than_1_l771_771451


namespace probability_heart_then_club_l771_771660

theorem probability_heart_then_club :
  let P_heart := 13 / 52
  let P_club_given_heart := 13 / 51
  P_heart * P_club_given_heart = 13 / 204 := 
by
  let P_heart := (13 : ℚ) / 52
  let P_club_given_heart := (13 : ℚ) / 51
  have h : P_heart * P_club_given_heart = 13 / 204 := by
    calc
      P_heart * P_club_given_heart
        = (13 / 52) * (13 / 51) : rfl
    ... = (13 * 13) / (52 * 51) : by rw [mul_div_mul_comm]
    ... = 169 / 2652 : rfl
    ... = 13 / 204 : by norm_num
  exact h

end probability_heart_then_club_l771_771660


namespace determine_m_if_root_exists_l771_771519

def fractional_equation_has_root (x m : ℝ) : Prop :=
  (3 / (x - 4) + (x + m) / (4 - x) = 1)

theorem determine_m_if_root_exists (x : ℝ) (h : fractional_equation_has_root x m) : m = -1 :=
sorry

end determine_m_if_root_exists_l771_771519


namespace dogs_with_three_legs_l771_771775

theorem dogs_with_three_legs 
  (DogsTotal : ℕ)
  (NailsTrimmed : ℕ)
  (NailsPerPaw : ℕ)
  (PawsPerFourLeggedDog : ℕ)
  (DogsTotal = 11) 
  (NailsTrimmed = 164) 
  (NailsPerPaw = 4) 
  (PawsPerFourLeggedDog = 4) :
  ∃ DogsWithThreeLegs : ℕ, DogsWithThreeLegs = 3 :=
by {
  sorry,
}

end dogs_with_three_legs_l771_771775


namespace find_f_of_2023_l771_771467

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_2023 : ∀ (f : ℝ → ℝ),
  (∀ x, f (-x) = -f x) ∧
  (∀ x, f (x + 4) = f x) ∧
  (∀ x ∈ set.Ioo 0 2, f x = 2^x + real.logb 3 (x + 2)) →
  f 2023 = -3 :=
sorry

end find_f_of_2023_l771_771467


namespace tram_speed_l771_771276

variables (V : ℝ)

theorem tram_speed (h : (V + 5) / (V - 5) = 600 / 225) : V = 11 :=
sorry

end tram_speed_l771_771276


namespace only_n_divides_2_n_minus_1_l771_771801

theorem only_n_divides_2_n_minus_1 :
  ∀ n : ℕ, n ≥ 1 → (n ∣ (2^n - 1)) → n = 1 :=
by
  sorry

end only_n_divides_2_n_minus_1_l771_771801


namespace subway_train_speed_l771_771192

open Nat

-- Define the speed function
def speed (s : ℕ) : ℕ := s^2 + 2*s

-- Define the theorem to be proved
theorem subway_train_speed (t : ℕ) (ht : 0 ≤ t ∧ t ≤ 7) (h_speed : speed 7 - speed t = 28) : t = 5 :=
by
  sorry

end subway_train_speed_l771_771192


namespace variance_of_Y_l771_771885

noncomputable def E_X : ℝ := 2
noncomputable def D_X : ℝ := 4 / 3
noncomputable def X := binomial 6 (1/3)
def Y : (ℝ → ℝ) := λ X => 3 * X + 1

theorem variance_of_Y : (variance Y) = 12 := by 
  sorry

end variance_of_Y_l771_771885


namespace lattice_points_on_hyperbola_l771_771909

theorem lattice_points_on_hyperbola :
  ∃ (s : Finset (ℤ × ℤ)), 
  (∀ p ∈ s, let (x, y) := p in x^2 - y^2 = 65) ∧ 
  (4 : Finset.card s) :=
begin
  sorry
end

end lattice_points_on_hyperbola_l771_771909


namespace problem_statement_l771_771460

-- Definition of the given condition: Sum of the first n terms
def S (n : ℕ) : ℕ := n^2 - 4 * n + 4

-- Problem statement:
theorem problem_statement :
  (∀ n ≥ 1, a_n = (if n = 1 then 1 else 2 * n - 5)) → 
  (let c (n : ℕ) := if n = 1 then -3 else 1 - 4 / (2 * n - 5) in
    (∑ k in range up_to 2n, (c k) * (c (k + 1)) < 0 ) = 3) →
  (let T (n : ℕ) := ∑ i in range n, (if i = 1 then 1 else 1 / (2 * i - 5)) in
    ∀ n, T(2 * n + 1) - T n ≤ 23 / 15) := sorry

end problem_statement_l771_771460


namespace parrot_silent_condition_l771_771534

-- Definitions based on conditions
def repeats_every_word_its_hears (parrot : Type) : Prop := 
  ∀ (word : String), parrot.hear word → parrot.repeat word

def is_silent (parrot : Type) : Prop := 
  ∀ (word : String), ¬parrot.speaks_any_word word

-- The proof statement
theorem parrot_silent_condition (parrot : Type) (h1 : repeats_every_word_its_hears parrot) (h2 : is_silent parrot) : 
  (∀ (word : String), ¬parrot.hear word) ∨ parrot.is_deaf :=
sorry

end parrot_silent_condition_l771_771534


namespace relationship_among_abc_l771_771122

noncomputable def a : ℝ := classical.some (exists_eq_eq (λ x, 2^x + x) 1)
noncomputable def b : ℝ := classical.some (exists_eq_eq (λ x, 2^x + x) 2)
noncomputable def c : ℝ := classical.some (exists_eq_eq (λ x, 3^x + x) 2)

theorem relationship_among_abc : a < c < b := sorry

end relationship_among_abc_l771_771122


namespace vectors_parallel_l771_771042

theorem vectors_parallel (x : ℝ) :
    ∀ (a b : ℝ × ℝ × ℝ),
    a = (2, -1, 3) →
    b = (x, 2, -6) →
    (∃ k : ℝ, b = (k * 2, k * -1, k * 3)) →
    x = -4 :=
by
  intro a b ha hb hab
  sorry

end vectors_parallel_l771_771042


namespace tangent_line_eqn_at_1_when_a_neg_one_unique_zero_of_g_implies_a_is_one_range_of_m_under_conditions_l771_771030

section
  variable {f g : ℝ → ℝ}
  variable {a : ℝ}

  -- Define f and g
  def f (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + a*x^2 + 2
  def g (x : ℝ) : ℝ := f x - x - 2
  
  -- Problem (I)
  theorem tangent_line_eqn_at_1_when_a_neg_one :
    (a = -1) → (∀ x, f x = (x^2 - 2*x) * Real.log x - x^2 + 2) → 
    (∃ m b: ℝ, m = - (f' 1) ∧ b = f 1 + 1 * (f' 1) ∧ 3 * 1 + b - 4 = 0) :=
  sorry

  -- Problem (II)
  theorem unique_zero_of_g_implies_a_is_one :
    (a > 0) → (∃ x₀ : ℝ, g x₀ = 0) → (∀ x, g x = 0 ↔ x = 1) → a = 1 :=
  sorry

  -- Problem (III)
  theorem range_of_m_under_conditions :
    (0 < a) → (a = 1) → (∀ x, e^(-2) < x ∧ x < e → g x ≤ m) → 
    (2 * e^2 - 3 * e < m ∧ ∀ k : ℝ, k > (2 * e^2 - 3 * e)) :=
  sorry
end

end tangent_line_eqn_at_1_when_a_neg_one_unique_zero_of_g_implies_a_is_one_range_of_m_under_conditions_l771_771030


namespace container_volume_ratio_l771_771760

variable (A B C : ℝ)

theorem container_volume_ratio (h1 : (4 / 5) * A = (3 / 5) * B) (h2 : (3 / 5) * B = (3 / 4) * C) :
  A / C = 15 / 16 :=
sorry

end container_volume_ratio_l771_771760


namespace part1_part2_part3_l771_771879

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * a * x^3 - exp x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a * x^2 - exp x

-- If f is monotonically increasing in [1/2, 1], then a > e.
theorem part1 (a : ℝ) (hmono : ∀ x ∈ Icc (1 / 2 : ℝ) 1, 0 < f' a x) : a > Real.exp 1 :=
sorry

-- If f' has two extreme points x1 and x2 (x1 < x2), then a > e / 2
theorem part2 (a : ℝ) (x1 x2 : ℝ) (hx : x1 < x2) 
  (hfx1 : f' a x1 = 0) (hfx2 : f' a x2 = 0) : a > Real.exp 1 / 2 :=
sorry

-- Under the conditions of part2, prove -e / 2 < f'(x1) < -1
theorem part3 (a : ℝ) (x1 x2 : ℝ) (hx : x1 < x2) 
  (hfx1 : f' a x1 = 0) (hfx2 : f' a x2 = 0) 
  (hx_range : 0 < x1 ∧ x1 < 1) : -Real.exp 1 / 2 < f' a x1 ∧ f' a x1 < -1 :=
sorry

end part1_part2_part3_l771_771879


namespace totalPizzaEaten_l771_771178

-- Define the conditions
def rachelAte : ℕ := 598
def bellaAte : ℕ := 354

-- State the theorem
theorem totalPizzaEaten : rachelAte + bellaAte = 952 :=
by
  -- Proof omitted
  sorry

end totalPizzaEaten_l771_771178


namespace ratio_of_sums_equiv_seven_eighths_l771_771053

variable (p q r u v w : ℝ)
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
variable (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
variable (h1 : p^2 + q^2 + r^2 = 49)
variable (h2 : u^2 + v^2 + w^2 = 64)
variable (h3 : p * u + q * v + r * w = 56)

theorem ratio_of_sums_equiv_seven_eighths :
  (p + q + r) / (u + v + w) = 7 / 8 :=
by
  sorry

end ratio_of_sums_equiv_seven_eighths_l771_771053


namespace purchase_46_pots_min_cost_369_l771_771092

variable (x y m : ℕ)

theorem purchase_46_pots (hx : x + y = 46)
    (hc : 9 * x + 6 * y = 390)
    (hgc : x ≥ 2 * y) : 
    x = 38 ∧ y = 8 := 
sorry

theorem min_cost_369 (min_m : m ≥ 31)
    (hm : m ≥ ⌈92 / 3⌉) 
    (hx : m = 31) : 
    (3 * m + 276) = 369 := 
by
  sorry

-- Definition of ceiling, necessary for the second theorem
noncomputable def ceil (a : ℚ) : ℕ :=
  if h : a < 0 then 0 else (Nat.ceil a).toNat

end purchase_46_pots_min_cost_369_l771_771092


namespace tangent_planes_through_centers_l771_771961

theorem tangent_planes_through_centers (R1 R2 R3 R4 R5 : ℝ) (C1 C2 C3 C4 C5 : ℝ × ℝ × ℝ) 
  (h1 : R1 > 0) 
  (h2 : R2 > 0) 
  (h3 : R3 > 0) 
  (h4 : R4 > 0) 
  (h5 : R5 > 0) : 
  ∃ (P1 P2 P3 P4 P5 : set (ℝ × ℝ × ℝ)), 
    (∀ i j, i ≠ j → tangent_plane_through_center P1 C1 C2 ∧
            tangent_plane_through_center P2 C2 C3 ∧
            tangent_plane_through_center P3 C3 C4 ∧
            tangent_plane_through_center P4 C4 C5 ∧
            tangent_plane_through_center P5 C5 C1) := sorry

def tangent_plane_through_center (P : set (ℝ × ℝ × ℝ)) (C1 C2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (n : ℝ × ℝ × ℝ), ∀ (x : ℝ × ℝ × ℝ), (x ∈ P ↔ inner n x = 1)

end tangent_planes_through_centers_l771_771961


namespace car_travel_distance_l771_771494

theorem car_travel_distance 
  (v_train : ℝ) (h_train_speed : v_train = 90) 
  (v_car : ℝ) (h_car_speed : v_car = (2 / 3) * v_train) 
  (t : ℝ) (h_time : t = 0.5) :
  ∃ d : ℝ, d = v_car * t ∧ d = 30 := 
sorry

end car_travel_distance_l771_771494
