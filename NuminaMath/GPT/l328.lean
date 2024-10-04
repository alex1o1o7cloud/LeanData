import Complex
import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Geometry
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupWithZero
import Mathlib.Algebra.GroupWithZero.Basic
import Mathlib.Algebra.Log
import Mathlib.Algebra.Matrix.Determinant
import Mathlib.Algebra.Mod.Basic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Factorial
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Perm
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.Squarefree
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.MeasureTheory.measurableSpace
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic

namespace no_valid_ratio_isosceles_trapezoid_l328_328993

theorem no_valid_ratio_isosceles_trapezoid (a b : ℝ) :
  let CE := b in  -- The altitude equals the larger base
  let AC := a in  -- The smaller base equals the diagonal
  let AB := a in  -- The smaller base
  let DE := (b - a) / 2 in -- Length of DE, derived from the isosceles property
  let AE := a + b / 2 in -- Length of AE
  let pythagorean_theorem := a^2 = (b/2 + a/2)^2 + b^2 in
  ¬ ∃ x : ℝ, 3 * x^2 + 2 * x + 1 = 0 :=  -- No real solution to the quadratic equation
sorry

end no_valid_ratio_isosceles_trapezoid_l328_328993


namespace hyperbola_properties_l328_328737

theorem hyperbola_properties :
  (∀ x : ℝ, f x = x + (1 / x)) →
  vertices (f x) = { (1 / (2 ^ (1 / 4)), (sqrt 2 + 1) / (2 ^ (1 / 4))) , (-1 / (2 ^ (1 / 4)), -(sqrt 2 + 1) / (2 ^ (1 / 4))) } ∧
  eccentricity (f x) = sqrt (4 - 2 * sqrt 2) ∧
  foci (f x) = { (sqrt (2 + 2 * sqrt 2) / (sqrt 2 + 1), sqrt (2 + 2 * sqrt 2)), (-sqrt (2 + 2 * sqrt 2) / (sqrt 2 + 1), -sqrt (2 + 2 * sqrt 2)) } :=
by
  sorry

end hyperbola_properties_l328_328737


namespace product_of_dodecagon_points_is_one_l328_328707

-- Define the regular dodecagon with specific points
structure RegularDodecagon :=
  (Q : ℕ → Complex)
  (center_origin : ∀ n, |Q n| = 1) -- Points lie on the unit circle
  (Q1_pos : Q 1 = Complex.ofReal 1)
  (Q7_pos : Q 7 = Complex.ofReal (-1))
  (dodecagon : ∀ k, Q k = Complex.exp (2 * k * π * Complex.I / 12))

theorem product_of_dodecagon_points_is_one (dodecagon : RegularDodecagon) : 
  ∏ k in finset.range 12, (dodecagon.Q k) = 1 :=
by
  sorry

end product_of_dodecagon_points_is_one_l328_328707


namespace hotel_loss_l328_328131

variable (operations_expenses : ℝ)
variable (fraction_payment : ℝ)

theorem hotel_loss :
  operations_expenses = 100 →
  fraction_payment = 3 / 4 →
  let total_payment := fraction_payment * operations_expenses in
  let loss := operations_expenses - total_payment in
  loss = 25 :=
by
  intros h₁ h₂
  have tstp : total_payment = 75 := by
    rw [h₁, h₂]
    norm_num
  have lss : loss = 25 := by
    rw [h₁, tstp]
    norm_num
  exact lss

end hotel_loss_l328_328131


namespace P_is_linear_l328_328575

variable {a : ℕ → ℝ}

-- Define the sequence condition
axiom seq_cond (i : ℕ) : 1 ≤ i → a (i - 1) + a (i + 1) = 2 * a i

-- Define the polynomial P
def P (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in finset.range (n + 1), a k * nat.choose n k * x^k * (1-x)^(n-k)

-- Theorem: P(x) is a first-degree polynomial in x
theorem P_is_linear (n : ℕ) (x : ℝ) :
  P a n x = a 0 + (a 1 - a 0) * n * x := by
  sorry

end P_is_linear_l328_328575


namespace collinear_vectors_result_l328_328965

noncomputable def collinear_vectors :=
  ∀ (t : ℝ), (1 : ℝ, t) ∥ (t, 9) → (t = 3 ∨ t = -3)

-- Prove the statement (proof is omitted with sorry)
theorem collinear_vectors_result : collinear_vectors :=
by
  sorry

end collinear_vectors_result_l328_328965


namespace part_a_part_b_part_c_l328_328572

section
variable {R : Type*} 
variable (f : R → R)
variable (domain : set R)
variable [ordered_semiring R]

-- Conditions from the given problem.
axiom f_condition_1 : ∀ x : R, x > 1 → f x < 0
axiom f_condition_2 : f (1 / 2) = 1
axiom f_property : ∀ (x y : R), x ∈ domain → y ∈ domain → f (x * y) = f x + f y

-- Part (a): Prove that f(1/x) = -f(x)
theorem part_a (x : R) (hx : x > 0): f (1 / x) = - f x :=
sorry

-- Part (b): Prove that f is a decreasing function
theorem part_b (x₁ x₂ : R) (hx₁ : x₁ ∈ domain) (hx₂ : x₂ ∈ domain) (h : x₁ < x₂) : f x₂ < f x₁ :=
sorry

-- Part (c): Determine the solution set for the inequality f(2) + f(5 - x) ≥ -2
theorem part_c (x : R) (hx : x ∈ domain) : (x ≥ 3 ∧ x < 5) ↔ f 2 + f (5 - x) ≥ -2 :=
sorry

end

end part_a_part_b_part_c_l328_328572


namespace monotonic_decreasing_interval_l328_328419

open Real

noncomputable def f (x : ℝ) : ℝ := log (1/2) (x^2 - 2 * x - 3)

theorem monotonic_decreasing_interval : 
  ∀ x, x ∈ set.Ioi 3 → strict_mono (λ x, -f x) :=
begin
  sorry
end

end monotonic_decreasing_interval_l328_328419


namespace inverse_g_l328_328541

variable {X : Type} [invX : Invertible X] -- Assuming we have a type X that can support invertible functions

noncomputable def g (x : X) (a b c : X → X) [Invertible a] [Invertible b] [Invertible c] : X :=
  (b ∘ a ∘ c) x

theorem inverse_g (a b c : X → X) [Invertible a] [Invertible b] [Invertible c] :
  Inverse (g _ a b c) = (Inverse c) ∘ (Inverse a) ∘ (Inverse b) :=
by
  sorry

end inverse_g_l328_328541


namespace other_endpoint_product_l328_328423

theorem other_endpoint_product :
  ∀ (x y : ℤ), 
    (3 = (x + 7) / 2) → 
    (-5 = (y - 1) / 2) → 
    x * y = 9 :=
by
  intro x y h1 h2
  sorry

end other_endpoint_product_l328_328423


namespace simplify_expression_l328_328394

theorem simplify_expression (x : ℝ) :
  x - 3 * (1 + x) + 4 * (1 - x)^2 - 5 * (1 + 3 * x) = 4 * x^2 - 25 * x - 4 := by
  sorry

end simplify_expression_l328_328394


namespace number_of_pounds_colombian_beans_l328_328852

def cost_per_pound_colombian : ℝ := 5.50
def cost_per_pound_peruvian : ℝ := 4.25
def total_weight : ℝ := 40
def desired_cost_per_pound : ℝ := 4.60
noncomputable def amount_colombian_beans (C : ℝ) : Prop := 
  let P := total_weight - C
  cost_per_pound_colombian * C + cost_per_pound_peruvian * P = desired_cost_per_pound * total_weight

theorem number_of_pounds_colombian_beans : ∃ C, amount_colombian_beans C ∧ C = 11.2 :=
sorry

end number_of_pounds_colombian_beans_l328_328852


namespace prob_both_questions_correct_l328_328987

variable (P : String → ℝ)

-- Definitions for the given problem conditions
def P_A : ℝ := P "first question"
def P_B : ℝ := P "second question"
def P_A'B' : ℝ := P "neither question"

-- Given values for the problem
axiom P_A_given : P_A = 0.63
axiom P_B_given : P_B = 0.49
axiom P_A'B'_given : P_A'B' = 0.20

-- The theorem that needs to be proved
theorem prob_both_questions_correct : (P "first and second question") = 0.32 :=
by
  have P_A_union_B := 1 - P_A'B'
  have P_A_and_B := P_A + P_B - P_A_union_B
  exact sorry

#eval prob_both_questions_correct P

end prob_both_questions_correct_l328_328987


namespace count_even_odd_divisors_9_l328_328272

theorem count_even_odd_divisors_9! : 
  let n := factorial 9
  ∃ even_divisors odd_divisors : ℕ, 
  (prime.factorization n = {2: 7, 3: 4, 5: 1, 7: 1}) ∧ 
  even_divisors = 140 ∧
  odd_divisors = 20 :=
by
  let n := factorial 9
  have prime_factors : n.prime.factorization = {2: 7, 3: 4, 5: 1, 7: 1} := sorry
  have even_divisor_count : even_divisors = count_even_divisors n := sorry -- assuming count_even_divisors is defined
  have odd_divisor_count : odd_divisors = count_odd_divisors n := sorry -- assuming count_odd_divisors is defined
  use even_divisors
  use odd_divisors
  split
  . exact prime_factors
  split
  . exact even_divisor_count
  . exact odd_divisor_count
  exact ⟨even_divisors, odd_divisors, prime_factors, even_divisor_count, odd_divisor_count⟩
  
-- Helper functions would be required to complete the proof (like count_even_divisors and count_odd_divisors).
-- Skipping these for this example.

end count_even_odd_divisors_9_l328_328272


namespace expected_americans_with_allergies_l328_328695

theorem expected_americans_with_allergies (prob : ℚ) (sample_size : ℕ) (h_prob : prob = 1/5) (h_sample_size : sample_size = 250) :
  sample_size * prob = 50 := by
  rw [h_prob, h_sample_size]
  norm_num

#print expected_americans_with_allergies

end expected_americans_with_allergies_l328_328695


namespace lcm_gcd_12_18_l328_328197

open Int

theorem lcm_gcd_12_18 : 
    lcm 12 18 = 36 ∧ gcd 12 18 = 6 := by
  sorry

end lcm_gcd_12_18_l328_328197


namespace find_ST_l328_328722

theorem find_ST (SP ST S : ℝ) (h₀ : SP = 10) (h₁ : cos S = 0.5) : ST = 20 :=
by
  -- We skip the proof details here
  sorry

end find_ST_l328_328722


namespace transformed_cubic_polynomial_l328_328622

theorem transformed_cubic_polynomial (x z : ℂ) 
    (h1 : z = x + x⁻¹) (h2 : x^3 - 3 * x^2 + x + 2 = 0) : 
    x^2 * (z^2 - z - 1) + 3 = 0 :=
sorry

end transformed_cubic_polynomial_l328_328622


namespace a2_value_a3_value_reciprocal_is_arithmetic_inequality_lambda_l328_328962

variable (a_seq b_seq : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (a_1 : ℝ)
variable (condition : ∀ n, a_seq n + b_seq n = 1 ∧ b_seq (n + 1) = b_seq n / (1 - (a_seq n)^2))
variable (a_seq_formula : ∀ n, a_seq n = 1 / (n + 3))
variable (b_seq_formula : ∀ n, b_seq n = (n + 2) / (n + 3))
variable (S_formula : ∀ n, S n = ∑ k in finset.range n, a_seq k * a_seq (k + 1))

theorem a2_value : a_seq 2 = 1 / 5 := sorry

theorem a3_value : a_seq 3 = 1 / 6 := sorry

theorem reciprocal_is_arithmetic : ∀ n, (1 / a_seq n) = n + 3 := sorry

theorem inequality_lambda : ∀ (λ : ℝ), (∀ n, 4 * λ * (S n) < b_seq n) ↔ λ ≤ 1 := sorry

end a2_value_a3_value_reciprocal_is_arithmetic_inequality_lambda_l328_328962


namespace sum_of_variables_l328_328024

theorem sum_of_variables (x y z w : ℤ) 
(h1 : x - y + z = 7) 
(h2 : y - z + w = 8) 
(h3 : z - w + x = 4) 
(h4 : w - x + y = 3) : 
x + y + z + w = 11 := 
sorry

end sum_of_variables_l328_328024


namespace eval_polynomial_positive_root_l328_328551

theorem eval_polynomial_positive_root : 
  ∃ x : ℝ, (x^2 - 3 * x - 10 = 0 ∧ 0 < x ∧ (x^3 - 3 * x^2 - 9 * x + 7 = 12)) :=
sorry

end eval_polynomial_positive_root_l328_328551


namespace count_valid_two_digit_prime_sum_even_l328_328613

def is_prime_digit (n : Nat) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_two_digit_integer (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def sum_of_digits_even (n : Nat) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  (d1 + d2) % 2 = 0

def valid_two_digit_prime_sum_even (n : Nat) : Prop :=
  is_two_digit_integer n ∧ is_prime_digit (n / 10) ∧ is_prime_digit (n % 10) ∧ sum_of_digits_even n

theorem count_valid_two_digit_prime_sum_even : 
  ∃ n, n = 10 ∧ (Set.count valid_two_digit_prime_sum_even {n | is_two_digit_integer n}.toFinset) = n := 
by
  sorry

end count_valid_two_digit_prime_sum_even_l328_328613


namespace rhombus_PQ_min_length_l328_328666

namespace RhombusProof

-- Define the rhombus and given conditions
structure Rhombus (A B C D : Type) extends Quadrilateral A B C D :=
  (diagonal_AC : ℝ := 18)
  (diagonal_BD : ℝ := 24)
  (midpoint_E : ∀ E : Type, (segment_AC AC) / 2 = (segment_BD BD) / 2)

-- Define point N on segment AB
def PointOnSegmentAB (A B N : Type) := sorry

-- Define perpendiculars from N to AC and BD to get points P and Q
def PerpendicularFoot (N AC BD P Q : Type) : Prop :=
  ⟦NP ⊥ AC ∧ NQ ⊥ BD⟧

-- Statement to be proved
theorem rhombus_PQ_min_length (A B C D N P Q: Type) [Rhombus A B C D] :
  PointOnSegmentAB A B N →
  PerpendicularFoot N AC BD P Q →
  ℝ := 4 :=
sorry

end RhombusProof

end rhombus_PQ_min_length_l328_328666


namespace cost_price_l328_328518

theorem cost_price (C : ℝ) (M : ℝ) : 
  (0.90 * M = 216.67) → 
  (1.30 * C = 216.67) → 
  C = 166.67 := by
  -- Assume conditions
  intro h1 h2
  -- Solve for C
  have h3: C = 216.67 / 1.30 := by sorry 
  exact h3

end cost_price_l328_328518


namespace episode_duration_is_half_l328_328660

noncomputable def episode_duration : ℕ → ℕ → ℕ → ℚ
| 9, 22, 112 => 112 / ((9 * 22) + (22 + 4))

theorem episode_duration_is_half :
  episode_duration 9 22 112 = 0.5 := by
  sorry

end episode_duration_is_half_l328_328660


namespace insects_remaining_l328_328710

-- Define the initial counts of spiders, ants, and ladybugs
def spiders : ℕ := 3
def ants : ℕ := 12
def ladybugs : ℕ := 8

-- Define the number of ladybugs that flew away
def ladybugs_flew_away : ℕ := 2

-- Prove the total number of remaining insects in the playground
theorem insects_remaining : (spiders + ants + ladybugs - ladybugs_flew_away) = 21 := by
  -- Expand the definitions and compute the result
  sorry

end insects_remaining_l328_328710


namespace final_result_l328_328543

noncomputable def double_factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * double_factorial (n - 2)

noncomputable def sum_expression : ℚ :=
  ∑ i in finset.range 1005, (double_factorial (2 * i + 1) : ℚ) / double_factorial (2 * (i + 1))

noncomputable def denominator_condition (n : ℕ) : ℕ :=
  if n = 0 then 0 else ↑(nat.factors n).count 2

noncomputable def find_ab_div_10 : ℚ :=
  let sum_result : ℚ := sum_expression in
  let den := sum_result.denom in
  let a := denominator_condition den in
  let b := den / (2^a) in
  (a * b) / 10

theorem final_result : find_ab_div_10 = 100.5 := sorry

end final_result_l328_328543


namespace combined_area_of_triangles_l328_328054

noncomputable def area_of_rectangle (length width : ℝ) : ℝ :=
  length * width

noncomputable def first_triangle_area (x : ℝ) : ℝ :=
  5 * x

noncomputable def second_triangle_area (base height : ℝ) : ℝ :=
  (base * height) / 2

theorem combined_area_of_triangles (length width x base height : ℝ)
  (h1 : area_of_rectangle length width / first_triangle_area x = 2 / 5)
  (h2 : base + height = 20)
  (h3 : second_triangle_area base height / first_triangle_area x = 3 / 5)
  (length_value : length = 6)
  (width_value : width = 4)
  (base_value : base = 8) :
  first_triangle_area x + second_triangle_area base height = 108 := 
by
  sorry

end combined_area_of_triangles_l328_328054


namespace product_of_roots_in_range_l328_328604

noncomputable def f (x : ℝ) : ℝ := abs (abs (x - 1) - 1)

theorem product_of_roots_in_range (m : ℝ) (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∃ x1 x2 x3 x4 : ℝ, 
        f x1 = m ∧ 
        f x2 = m ∧ 
        f x3 = m ∧ 
        f x4 = m ∧ 
        x1 ≠ x2 ∧ 
        x1 ≠ x3 ∧ 
        x1 ≠ x4 ∧ 
        x2 ≠ x3 ∧ 
        x2 ≠ x4 ∧ 
        x3 ≠ x4) :
  ∃ p : ℝ, p = (m * (2 - m) * (m + 2) * (-m)) ∧ -3 < p ∧ p < 0 :=
sorry

end product_of_roots_in_range_l328_328604


namespace y_value_l328_328868

theorem y_value (y : ℕ) : 8^3 + 8^3 + 8^3 + 8^3 = 2^y → y = 11 := 
by 
  sorry

end y_value_l328_328868


namespace derivative_at_1_l328_328043

def f (x : ℝ) : ℝ := x * Real.log x

theorem derivative_at_1 : HasDerivAt f 1 1 := by
  sorry

end derivative_at_1_l328_328043


namespace problem_statement_l328_328342

variable {α : Type*} [field α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0     := a₁
| (n+1) := geometric_sequence a₁ q n * q

def Sn (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * n else a₁ * (1 - q ^ n) / (1 - q)

theorem problem_statement (a₁ q : α) (n : ℕ) (h₀ : q ≠ 1)
  (h₁ : geometric_sequence a₁ q 4 - geometric_sequence a₁ q 2 = 12)
  (h₂ : geometric_sequence a₁ q 5 - geometric_sequence a₁ q 3 = 24) :
  Sn a₁ q n / geometric_sequence a₁ q (n - 1) = 2 - 2 ^ (1 - n) :=
sorry

end problem_statement_l328_328342


namespace ellipse_chord_line_eqn_l328_328630

theorem ellipse_chord_line_eqn :
  (∀ x y : ℝ, x^2 / 36 + y^2 / 9 = 1) ∧
  (∃ A B : ℝ × ℝ, (A.1 + B.1) / 2 = 4 ∧ (A.2 + B.2) / 2 = 2) →
  ∃ k : ℝ, (k = -1 / 2) →
  (∀ x y : ℝ, y - 2 = k * (x - 4)) →
  ∀ x y : ℝ, x + 2y - 8 = 0 := 
by sorry

end ellipse_chord_line_eqn_l328_328630


namespace power_sums_divisibility_l328_328372

open Polynomial

theorem power_sums_divisibility {n : ℕ} {a : Fin n → ℤ} {p : ℕ} (hp : p > 0)
  (h_coeffs : ∀ i, p^(i + 1) ∣ a i)
  (h_roots : ∀ x : Fin n → ℤ, IsRoot (λ x : Fin n → ℤ, ∑ i in Finset.range n, a i * x^(n - i))) :
  ∀ k : ℕ, (k > 0) → p^k ∣ ∑ i in Finset.range n, (roots . val) i ^ k :=
by
  sorry

end power_sums_divisibility_l328_328372


namespace sum_of_solutions_quadratic_l328_328888

theorem sum_of_solutions_quadratic :
  ∀ x : ℝ, -18 * x * x + 54 * x - 72 = 0 → (sum_of_solutions (-18) 54 (-72)) = 3 := by
  sorry

end sum_of_solutions_quadratic_l328_328888


namespace great_grandchildren_l328_328792

theorem great_grandchildren (n age grandchildren : ℕ)
  (age_eq : age = 91)
  (grandchildren_eq : grandchildren = 11)
  (concatenation_condition : 11 * n * 91 = n * 1001) :
  n = 1 :=
by
  rw [age_eq, grandchildren_eq] at concatenation_condition
  have h : 11 * 91 = 1001, from rfl
  rw [h] at concatenation_condition
  exact (mul_left_inj' (by norm_num : 1001 ≠ 0)).1 concatenation_condition

end great_grandchildren_l328_328792


namespace exists_cities_with_degrees_less_than_k_minus_1_l328_328306

-- Definitions and conditions
variables (n k : ℕ) (deg : ℕ → ℕ) (connected : ℕ → ℕ → Prop)

-- Conditions
-- 1. There are n cities
-- 2. Each pair of cities is connected by at most one road
-- 3. The degree of a city is defined
-- 4. 2 ≤ k ≤ n

def degree (city : ℕ) : ℕ := deg city
axiom connected_by_at_most_one_road : ∀ i j, connected i j → i ≠ j

theorem exists_cities_with_degrees_less_than_k_minus_1 :
  2 ≤ k → k ≤ n → ∃ cities : fin k → ℕ, ∀ (i j : fin k), i ≠ j → abs (degree (cities i) - degree (cities j)) < k - 1 := 
by
  sorry

end exists_cities_with_degrees_less_than_k_minus_1_l328_328306


namespace intersection_S_T_l328_328685

-- Definition of Set S
def S : Set ℝ := {y | ∃ x : ℝ, y = 3^x}

-- Definition of Set T
def T : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Proof statement
theorem intersection_S_T :
  S ∩ T = { y : ℝ | y ∈ Icc 1 ⊤ } :=
sorry

end intersection_S_T_l328_328685


namespace sum_of_roots_eq_20_l328_328432

def square_geometry (ABCD : ℝ × ℝ × ℝ × ℝ) (L : ℝ) : Prop :=
  -- Here we define the properties of the square ABCD with side length L
  ∃ A B C D : (ℝ × ℝ), 
    (A.2 = B.2 ∧ A.1 ≠ B.1 ∧ B.1 = C.1 ∧ C.2 ≠ D.2 ∧ D.1 = A.1 ∧ A.2 ≠ D.2) ∧
    -- Diagonal AD is parallel to x-axis
    (A.1 ≠ D.1 ∧ A.2 = D.2) ∧
    -- Side lengths are equal to L
    (abs (B.2 - A.2) = L ∧ abs (C.1 - D.1) = L)

def parabola_properties (a b : ℝ) : Prop :=
  ∃ x : ℝ, ∃ y : ℝ, y = (1/5) * x^2 + a * x + b

def vertex_on_segment (a b L : ℝ) : Prop :=
  ∃ x : ℝ, ∃ y : ℝ, 
    (y = (1/5) * x^2 + a * x + b) ∧
    (x, y) ∈ {p : ℝ × ℝ | ∃ t ∈ set.Icc 0 1, (t * (L, 0) + (1 - t) * (0, 0)) = p }

theorem sum_of_roots_eq_20 :
  ∀ a b L : ℝ, 
    square_geometry (0, 0, L, L) L → 
    parabola_properties a b → 
    vertex_on_segment a b L →
    ((-a / (1 / 5)) + (10)) = 20 :=
sorry

end sum_of_roots_eq_20_l328_328432


namespace twin_brothers_age_l328_328090

theorem twin_brothers_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 17) : x = 8 := 
  sorry

end twin_brothers_age_l328_328090


namespace intersection_distance_squared_l328_328729

noncomputable def circle_center1 : ℝ × ℝ := (1, -2)
def circle_radius1 : ℝ := 3
noncomputable def circle_center2 : ℝ × ℝ := (1, 4)
def circle_radius2 : ℝ := Real.sqrt 13

theorem intersection_distance_squared :
  let C := (4/3, 1)
  let D := (2/3, 1)
  (CD : ℝ) := Real.dist C D
  (CD^2 = 4 / 9) :=
by
  let C := (1 + 1/3, -2 + 10/3)
  let D := (1 - 1/3, -2 + 10/3)
  let CD := Real.dist C D
  have H1 : (C.1 - D.1) = (2 / 3)
  have H2 : (C.2) = (D.2)
  have H3 : (H1 ^ 2 + H2 ^ 2) = 4 / 9
  sorry

end intersection_distance_squared_l328_328729


namespace largest_integral_x_l328_328558

theorem largest_integral_x (x : ℤ) : (1 / 4 : ℚ) < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 7 / 9 ↔ x = 4 :=
by 
  sorry

end largest_integral_x_l328_328558


namespace inequality_proof_l328_328389

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + 8*y + 2*z) * (x + 2*y + z) * (x + 4*y + 4*z) ≥ 256 * x * y * z :=
by
  -- Proof goes here
  sorry

end inequality_proof_l328_328389


namespace image_of_orthocenter_under_parallel_projection_l328_328927

open EuclideanGeometry

-- Define the context
variables {A B C O : Point} -- Vertices and circumcenter of triangle ABC
variables {A1 B1 C1 O1 : Point} -- Images of the respective points under parallel projection

-- Define some points in between
variables {M1 : Point} (hm1 : is_midpoint A1 B1 M1) -- M1 is midpoint of A1B1
variables {N1 : Point} (hn1 : is_midpoint A1 C1 N1) -- N1 is midpoint of A1C1

-- Parallelism preservation under projection
variables (hp1 : parallel (line_through C1 H1) (line_through O1 M1))
variables (hp2 : parallel (line_through B1 H1) (line_through O1 N1))

-- Orthocenter image definition
def H1 : Point := intersection_of_two_lines
  ⟨C1, parallel_line_through O1 M1, hp1⟩
  ⟨B1, parallel_line_through O1 N1, hp2⟩

theorem image_of_orthocenter_under_parallel_projection 
  (H : Point)
  (orthocenter_of_ABC : orthocenter A B C H) :
  image_of_orthocenter A1 B1 C1 O1 M1 N1 H1 :=
sorry

end image_of_orthocenter_under_parallel_projection_l328_328927


namespace least_possible_value_expression_l328_328464

theorem least_possible_value_expression :
  ∃ min_value : ℝ, ∀ x : ℝ, ((x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019) ≥ min_value ∧ min_value = 2018 :=
by
  sorry

end least_possible_value_expression_l328_328464


namespace polygon_sides_eq_seven_l328_328742

theorem polygon_sides_eq_seven (n : ℕ) (h : 2 * n - (n * (n - 3)) / 2 = 0) : n = 7 :=
by sorry

end polygon_sides_eq_seven_l328_328742


namespace hotel_loss_l328_328120
  
  -- Conditions
  def operations_expenses : ℝ := 100
  def total_payments : ℝ := (3 / 4) * operations_expenses
  
  -- Theorem to prove
  theorem hotel_loss : operations_expenses - total_payments = 25 :=
  by
    sorry
  
end hotel_loss_l328_328120


namespace point_in_fourth_quadrant_l328_328211

open Complex

def z : ℂ := (1 + I) * (1 - 2 * I)

theorem point_in_fourth_quadrant (z := (1 + I) * (1 - 2 * I)) : 0 < z.re ∧ z.im < 0 :=
by
  -- steps to manually check conditions
  have z_val : z = 3 - I := by
    calc
      (1 + I) * (1 - 2 * I)
        = 1 * 1 + 1 * (-2 * I) + I * 1 + I * (-2 * I) : by ring
    ... = 1 - 2 * I + I - 2 * -1 : by norm_num
    ... = 1 - 2 * I + I + 2 : by norm_num
    ... = 3 - I : by ring
  -- substitute obtained value of z and prove the condition
  rw [z_val]
  -- split into two conditions and prove individually
  split
  -- prove real part > 0
  norm_num
  -- prove imaginary part < 0
  norm_num
  done

end point_in_fourth_quadrant_l328_328211


namespace share_of_y_is_210_l328_328390

theorem share_of_y_is_210 (total_amount : ℕ) (ratio_x ratio_y ratio_z : ℕ)
    (h_total_amount : total_amount = 690)
    (h_ratio_x : ratio_x = 5)
    (h_ratio_y : ratio_y = 7)
    (h_ratio_z : ratio_z = 11) :
    let total_ratio := ratio_x + ratio_y + ratio_z in
    let share_of_y := (total_amount * ratio_y) / total_ratio in
    share_of_y = 210 := 
by
  sorry

end share_of_y_is_210_l328_328390


namespace factorial_double_factorial_identity_l328_328701

-- Defining the double factorial for odd numbers
def double_factorial : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := (n+2) * double_factorial n

-- The theorem statement
theorem factorial_double_factorial_identity (n : ℕ) : 
  (2 * n).factorial / n.factorial = 2^n * double_factorial (2 * n - 1) :=
by
  sorry

end factorial_double_factorial_identity_l328_328701


namespace min_value_of_reciprocal_sum_l328_328972

variable (m n : ℝ)
variable (a : ℝ × ℝ := (m, 1))
variable (b : ℝ × ℝ := (4 - n, 2))

theorem min_value_of_reciprocal_sum
  (h1 : m > 0) (h2 : n > 0)
  (h3 : a.1 * b.2 = a.2 * b.1) :
  (1/m + 8/n) = 9/2 :=
sorry

end min_value_of_reciprocal_sum_l328_328972


namespace geometric_sum_ratio_l328_328350

-- Definitions and Conditions
variables {a : ℕ → ℕ}
variable q : ℕ
variable n : ℕ

-- Condition a₅ - a₃ = 12
axiom h1 : a 5 - a 3 = 12

-- Condition a₆ - a₄ = 24
axiom h2 : a 6 - a 4 = 24

-- The goal to prove: Sₙ / aₙ = 2 - 2⁽¹⁻ⁿ⁾, where Sₙ is the sum of first n terms and aₙ is the nth term
theorem geometric_sum_ratio (S : ℕ → ℕ) (a_n : ℕ) : 
  (S n) / (a n) = 2 - 2^(1 - n) := 
sorry

end geometric_sum_ratio_l328_328350


namespace slices_per_pizza_l328_328456

theorem slices_per_pizza (total_pizzas : ℕ) (total_slices : ℕ) (h : total_pizzas = 21) (hs : total_slices = 168) : total_slices / total_pizzas = 8 :=
by 
  rw [h, hs]
  norm_num
  sorry

end slices_per_pizza_l328_328456


namespace find_a_l328_328205

theorem find_a (a : ℝ) : 
  (∀ (x : ℝ), 
    let expr := (1 + a * x)^5 * (1 - 2 * x)^4 in 
    ∃ c : ℝ, c * x^2 ∈ expr := 
        -(16 * (x ^ 2)) → a = 2 :=
by 
  sorry

end find_a_l328_328205


namespace hotel_loss_l328_328122

theorem hotel_loss (operations_expenses : ℝ) (payment_fraction : ℝ) (total_payment : ℝ) (loss : ℝ) 
  (hOpExp : operations_expenses = 100) 
  (hPayFr : payment_fraction = 3 / 4)
  (hTotalPay : total_payment = payment_fraction * operations_expenses) 
  (hLossCalc : loss = operations_expenses - total_payment) : 
  loss = 25 := 
by 
  sorry

end hotel_loss_l328_328122


namespace part_I_part_II_l328_328925

-- Defining the arithmetic sequence {a_n}
def a (n : ℕ) := 1 + (n - 1) * 2

-- Part (I): Prove that a_4 = 7 when n = 25
theorem part_I : ∃ m : ℕ, a m = 2 * 25 - 1 ∧ (1 : ℚ)^2⁻¹, (a 4)^2⁻¹, (a m)^2⁻¹ form_geom_seq :=
begin
    sorry -- placeholder for the proof
end

-- Part (II): Prove that for any n, 
-- 1 / a(n), 1 / a(n+1), and 1 / a(n+2) do not form an arithmetic sequence.
theorem part_II (n : ℕ) (hn : n > 0) : ¬is_arithmetic_seq ((1 : ℚ) / a n) ((1 : ℚ) / a (n + 1)) ((1 : ℚ) / a (n + 2)) :=
begin
  by_contradiction h,
  sorry -- placeholder for the proof
end

end part_I_part_II_l328_328925


namespace sum_difference_l328_328482

-- Definitions of the table and placement conditions
def is_valid_table (table : ℕ → ℕ → ℕ) (n : ℕ) :=
  ∀ i j, 1 ≤ table i j ∧ table i j ≤ n^2 

def placement_conditions (table : ℕ → ℕ → ℕ) (n : ℕ) :=
  ∃ i j, table i j = 1 ∧
  (∀ k, k ≥ 1 ∧ k < n^2 → ∃ r, ∃ c,
    table r c = k + 1 ∧ r = c ∧ c = table (r - 1) c)

-- The main theorem statement
theorem sum_difference (n : ℕ) (table : ℕ → ℕ → ℕ) 
  (h_valid_table : is_valid_table table n)
  (h_placement_conditions : placement_conditions table n) :
  let row_with_1   := some_row_containing_1 table n,
      col_with_n2 := some_col_containing_n2 table n,
      sum_row_with_1 := (∑ j, table row_with_1 j),
      sum_col_with_n2 := (∑ i, table i col_with_n2)
  in sum_row_with_1 - sum_col_with_n2 = n^2 - n :=
   sorry
 
-- Placeholder functions to extract specific rows and columns
-- Definitions are not provided here as they are more complex and specific to the problem solution mechanics
noncomputable def some_row_containing_1 (table : ℕ → ℕ → ℕ) (n : ℕ) : ℕ := sorry
noncomputable def some_col_containing_n2 (table : ℕ → ℕ → ℕ) (n : ℕ) : ℕ := sorry

end sum_difference_l328_328482


namespace tangent_line_eq_intervals_extreme_values_l328_328256

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Define the derivative f'(x)
def f' (x : ℝ) : ℝ := 3 * x^2 - 3

-- Part (1): Prove the tangent line equation at x=0 is y = -3x
theorem tangent_line_eq (x : ℝ) (h : x = 0) : ∀ y : ℝ, y = f'(0) * x :=
by
  sorry

-- Part (2): Prove the intervals of monotonicity and extreme values
theorem intervals_extreme_values : 
  (∀ x : ℝ, (x > 1 ∨ x < -1) → f'(x) > 0) ∧
  (∀ x : ℝ, (-1 < x ∧ x < 1) → f'(x) < 0) ∧
  max f (-1) = 2 ∧ min f (1) = -2 :=
by
  sorry

end tangent_line_eq_intervals_extreme_values_l328_328256


namespace calculate_area_BDF_l328_328385

noncomputable theory
open_locale vector_space

variables {A B C D E F : ℝ × ℝ × ℝ}
variables (AB BC CD DE EF FA : ℝ) 
variables (angle_ABC angle_CDE angle_EFA : ℝ)
variables (perpendicular_plane : Prop)

-- Given Conditions
def conditions : Prop :=
  AB = 3 ∧ BC = 3 ∧ CD = 3 ∧ DE = 3 ∧ EF = 3 ∧ FA = 3 ∧
  angle_ABC = 120 ∧ angle_CDE = 120 ∧ angle_EFA = 120 ∧
  perpendicular_plane

-- Using vector space operations to define points' arrangement and plane conditions
def vector_BD : ℝ × ℝ × ℝ := (B.1 - D.1, B.2 - D.2, B.3 - D.3)
def vector_BF : ℝ × ℝ × ℝ := (B.1 - F.1, B.2 - F.2, B.3 - F.3)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def area_triangle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  1 / 2 * magnitude (cross_product v1 v2)

-- Theorem Statement
theorem calculate_area_BDF :
  conditions ∧ perpendicular_plane → ∃ area, area_triangle (vector_BD B D) (vector_BF B F) = area :=
sorry

end calculate_area_BDF_l328_328385


namespace largest_average_l328_328771

def average (a b : ℕ) : ℝ := (a + b : ℝ) / 2

def multiples (k n : ℕ) : List ℕ := List.filter (λ x => x % k = 0) (List.range (n+1))

def set_avg (k n : ℕ) : ℝ :=
  let m := multiples k n
  if h : m ≠ [] then
    average (List.head m h) (List.last m h)
  else
    0

theorem largest_average {n : ℕ} (h : n = 151) :
  set_avg 6 n > set_avg 2 n ∧
  set_avg 6 n > set_avg 3 n ∧
  set_avg 6 n > set_avg 4 n ∧
  set_avg 6 n > set_avg 5 n :=
by
  sorry

end largest_average_l328_328771


namespace exercise_time_l328_328332

theorem exercise_time :
  let time_monday := 6 / 2 in
  let time_wednesday := 6 / 3 in
  let time_friday := 6 / 6 in
  let total_time := time_monday + time_wednesday + time_friday in
  total_time = 6 :=
by
  sorry

end exercise_time_l328_328332


namespace sum_of_exponents_is_28_l328_328754

theorem sum_of_exponents_is_28 (s : ℕ) (m : fin s → ℕ) (b : fin s → ℤ)
  (hm : strict_mono (m ∘ subtype.val)) (hb : ∀ i, b i = 1 ∨ b i = -1)
  (hbm : ∑ i, b i * 3 ^ m i = 2022) : (∑ i, m i) = 28 :=
sorry

end sum_of_exponents_is_28_l328_328754


namespace area_of_sector_l328_328406

theorem area_of_sector (r l : ℝ) (h_r : r = 5) (h_l : l = 3.5) : 
  (l / (2 * Real.pi * r)) * (Real.pi * r ^ 2) = 8.75 :=
by
  have h_r : r = 5 := by rfl
  have h_l : l = 3.5 := by rfl
  rw [h_r, h_l]
  calc
    (3.5 / (2 * Real.pi * 5)) * (Real.pi * 5 ^ 2) 
      = 0.35 * 25 : by rw [Real.mul_div_cancel_left,
                           Real.mul_comm,
                           Real.pi_mul_eq_mul,
                           Real.mul_div_assoc,
                           Real.pi_mul_real,
                           Real.div_eq_mul_inv]
      = 8.75 : by norm_num

end area_of_sector_l328_328406


namespace parametric_equations_solution_l328_328219

theorem parametric_equations_solution (t₁ t₂ : ℝ) : 
  (1 = 1 + 2 * t₁ ∧ 2 = 2 - 3 * t₁) ∧
  (-1 = 1 + 2 * t₂ ∧ 5 = 2 - 3 * t₂) ↔
  (t₁ = 0 ∧ t₂ = -1) :=
by
  sorry

end parametric_equations_solution_l328_328219


namespace lions_at_sanctuary_l328_328166

variable (L C : ℕ)

noncomputable def is_solution :=
  C = 1 / 2 * (L + 14) ∧
  L + 14 + C = 39 ∧
  L = 12

theorem lions_at_sanctuary : is_solution L C :=
sorry

end lions_at_sanctuary_l328_328166


namespace april_earned_money_l328_328843

theorem april_earned_money (price_per_rose : ℕ) (initial_roses : ℕ) (final_roses : ℕ) :
  price_per_rose = 7 →
  initial_roses = 9 →
  final_roses = 4 →
  price_per_rose * (initial_roses - final_roses) = 35 :=
by
  intros h_price h_initial h_final
  rw [h_price, h_initial, h_final]
  sorry

end april_earned_money_l328_328843


namespace sum_of_cubes_and_squares_mod_5_l328_328529

theorem sum_of_cubes_and_squares_mod_5 :
  (∑ i in Finset.range 50, (i + 1) ^ 3 + (i + 1) ^ 2) % 5 = 0 :=
by
  sorry

end sum_of_cubes_and_squares_mod_5_l328_328529


namespace prove_P_on_AC_l328_328730

noncomputable def problem_condition (A B C D P : Type*) [EuclideanGeometry ℝ] : Prop :=
  let ∠ := angle in 
  let ∠BAC := 30 in
  let ∠CAD := 30 in
  let ∠ACD := 40 in
  let ⟨∠ACB, _⟩ := 120 in
  ∠D := 40 in
  ∠PDA := 40 in
  ∠PBA := 10 in
  true -- This will encapsulate our conditions. 'true' is just a place-holder.

theorem prove_P_on_AC (A B C D P : Type*) [EuclideanGeometry ℝ] :
  problem_condition A B C D P →
  lies_on P A C :=
begin
  sorry,
end

end prove_P_on_AC_l328_328730


namespace base11_base14_subtraction_l328_328875

def convert_base11_to_base10 (d2 d1 d0 : ℕ) : ℕ :=
  d2 * 11^2 + d1 * 11^1 + d0 * 11^0

def convert_base14_to_base10 (d2 d1 d0 : ℕ) : ℕ :=
  d2 * 14^2 + d1 * 14^1 + d0 * 14^0

theorem base11_base14_subtraction :
  convert_base11_to_base10 3 7 3 - convert_base14_to_base10 4 14 5 = -542 :=
by
  sorry

end base11_base14_subtraction_l328_328875


namespace guests_and_gifts_l328_328525

-- Define guests
inductive Guest
| Bear
| Lynx
| Squirrel
| Mouse
| Wolf
| Sheep

open Guest

-- Define gifts
inductive Gift
| Candlestick
| Plate
| Needle
| Ring

open Gift

-- Define invitations and gifts received
def invitation_1 := [Bear, Lynx, Squirrel]
def gifts_1 := [Candlestick, Plate]

def invitation_2 := [Lynx, Squirrel, Mouse, Wolf]
def gifts_2 := [Candlestick, Needle]

def invitation_3 := [Wolf, Mouse, Sheep]
def gifts_3 := [Needle, Ring]

def invitation_4 := [Sheep, Bear, Wolf, Squirrel]
def gifts_4 := [Ring, Plate]

-- Define a function to map guests to their gifts
def gift_from_guest (guest : Guest) : Option Gift :=
match guest with
| Bear     => some Plate
| Lynx     => some Candlestick
| Mouse    => some Needle
| Sheep    => some Ring
| _        => none

-- Define the proposition to be proven
theorem guests_and_gifts :
    gift_from_guest Bear = some Plate ∧
    gift_from_guest Lynx = some Candlestick ∧
    gift_from_guest Mouse = some Needle ∧
    gift_from_guest Sheep = some Ring ∧
    gift_from_guest Wolf = none ∧
    gift_from_guest Squirrel = none :=
by sorry

end guests_and_gifts_l328_328525


namespace laborers_monthly_income_l328_328475

noncomputable def monthly_income (I : ℝ) (D : ℝ) : Prop :=
  (6 * I = 480 - D) ∧ (4 * I = 240 + D + 30)

theorem laborers_monthly_income : ∃ I, ∃ D, monthly_income I D ∧ I = 75 :=
by
  use 75
  use 30
  unfold monthly_income
  split
  { rw mul_comm, norm_num, refl }
  { rw mul_comm, norm_num, refl }

end laborers_monthly_income_l328_328475


namespace base_subtraction_proof_l328_328528

def convert_base8_to_base10 (n : Nat) : Nat :=
  5 * 8^4 + 4 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1

def convert_base9_to_base10 (n : Nat) : Nat :=
  4 * 9^3 + 3 * 9^2 + 2 * 9^1 + 1

theorem base_subtraction_proof :
  convert_base8_to_base10 54321 - convert_base9_to_base10 4321 = 19559 :=
by
  sorry

end base_subtraction_proof_l328_328528


namespace length_of_platform_l328_328801

-- Define the conditions and the problem we want to prove
theorem length_of_platform
  (length_train : ℕ)
  (time_pole : ℕ)
  (time_platform : ℕ)
  (H_train : length_train = 450)
  (H_pole : time_pole = 24)
  (H_platform : time_platform = 56) :
  (length_train * (time_platform : ℕ) / (time_pole : ℕ) - length_train = 600) :=
by
  rw [H_train, H_pole, H_platform]
  -- length_train is 450, time_pole is 24, time_platform is 56
  norm_num
  -- Prove the resulting algebraic equation
  exact sorry

end length_of_platform_l328_328801


namespace area_of_quadrilateral_l328_328503

-- Coordinates of the points
def A := (0, 0)
def B := (0, 2)
def C := (3, 2)
def D := (5, 0)

theorem area_of_quadrilateral : 
  let triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := 
    (1 / 2 : ℝ) * (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) in
  triangle_area A B C + triangle_area A C D = 8 := sorry

end area_of_quadrilateral_l328_328503


namespace harriet_return_speed_l328_328775

/-- Harriet's trip details: 
  - speed from A-ville to B-town is 100 km/h
  - the entire trip took 5 hours
  - time to drive from A-ville to B-town is 180 minutes (3 hours) 
  Prove the speed while driving back to A-ville is 150 km/h
--/
theorem harriet_return_speed:
  ∀ (t₁ t₂ : ℝ),
  (t₁ = 3) ∧ 
  (100 * t₁ = d) ∧ 
  (t₁ + t₂ = 5) ∧ 
  (t₂ = 2) →
  (d / t₂ = 150) :=
by
  intros t₁ t₂ h
  sorry

end harriet_return_speed_l328_328775


namespace infinite_positives_l328_328896

-- Define the quadratic equation
def quadratic_eqn (a x : ℝ) : ℝ :=
  a^2 * x^2 + a * x + 1 - 7 * a^2

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ :=
  (a)^2 - 4 * (a^2 * (1 - 7 * a^2))

-- The main statement
theorem infinite_positives (a : ℝ) :
  (∀ x : ℝ, quadratic_eqn a x = 0) →
  (discriminant a > 0) →
  ∃ (y : ℕ), y > 2 := sorry

end infinite_positives_l328_328896


namespace correct_calculation_result_l328_328616

theorem correct_calculation_result :
  ∃ x : ℕ, (6 * x = 96) ∧ (x / 8 = 2) :=
by {
  use 16,
  split,
  sorry,  -- 6 * 16 = 96
  sorry   -- 16 / 8 = 2
}

end correct_calculation_result_l328_328616


namespace arithmetic_sequence_iff_condition_l328_328920

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def condition (a : ℕ → ℝ) := 
  ∀ n : ℕ, ∑ i in finset.range n, (1 / (a i * a (i + 1))) = n / (a 0 * a (n + 1))

theorem arithmetic_sequence_iff_condition (a : ℕ → ℝ) (h : ∀ n, a n ≠ 0) :
  (arithmetic_sequence a) ↔ (condition a) :=
by 
  sorry

end arithmetic_sequence_iff_condition_l328_328920


namespace distinguishable_dodecahedron_colorings_l328_328452

theorem distinguishable_dodecahedron_colorings : 
    ∃ n : ℕ, n = 7983360 ∧ (∀ (dodecahedron : Type) 
    (face_colors : Fin 12 → Fin 12) 
    (rotations : Fin 5 → Perm (Fin 12)),
        counting_distinguishable_colorings dodecahedron face_colors rotations = n) :=
sorry

end distinguishable_dodecahedron_colorings_l328_328452


namespace negation_of_universal_proposition_l328_328420

theorem negation_of_universal_proposition :
  (∃ x : ℤ, x % 5 = 0 ∧ ¬ (x % 2 = 1)) ↔ ¬ (∀ x : ℤ, x % 5 = 0 → (x % 2 = 1)) :=
by sorry

end negation_of_universal_proposition_l328_328420


namespace professor_oscar_review_questions_l328_328838

-- Define the problem conditions.
def students_per_class : ℕ := 35
def questions_per_exam : ℕ := 10
def number_of_classes : ℕ := 5

-- Define the number of questions that must be reviewed.
def total_questions_to_review : ℕ := 1750

-- The theorem to be proved.
theorem professor_oscar_review_questions :
  students_per_class * questions_per_exam * number_of_classes = total_questions_to_review :=
by
  -- Here we write 'sorry' since we are not providing the full proof.
  sorry

end professor_oscar_review_questions_l328_328838


namespace parallelogram_rectangle_and_lines_l328_328646

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

def line_equation (p : ℝ × ℝ) (m : ℝ) : ℝ × ℝ → Prop :=
  λ q, q.2 = m * (q.1 - p.1) + p.2

def angle_bisector_slope (m1 m2 : ℝ) : ℝ :=
  (sqrt (1 + m1^2) + sqrt (1 + m2^2)) / (1 + m1 * m2)

theorem parallelogram_rectangle_and_lines (A B C : ℝ × ℝ) (D : ℝ × ℝ) :
    A = (0,0) →
    B = (3, Real.sqrt 3) →
    C = (4,0) →
    D = (1, -Real.sqrt 3) ∧
    slope A B * slope B C = -1 ∧
    ∀ q, line_equation C (slope A B) q ↔ q.1 - sqrt 3 * q.2 - 4 = 0 ∧
    ∀ q, line_equation B (2 + sqrt 3) q ↔ (2 + sqrt 3) * q.1 - q.2 - 6 - 2 * sqrt 3 = 0 :=
by
  intro hA hB hC
  sorry

end parallelogram_rectangle_and_lines_l328_328646


namespace range_of_a_l328_328292

theorem range_of_a (a : ℝ) (x : ℝ) (h : sqrt 3 * sin x + cos x = 2 * a - 1) : -1 / 2 ≤ a ∧ a ≤ 3 / 2 :=
by sorry

end range_of_a_l328_328292


namespace distance_light_travels_500_years_l328_328412

def distance_light_travels_one_year : ℝ := 5.87e12
def years : ℕ := 500

theorem distance_light_travels_500_years :
  distance_light_travels_one_year * years = 2.935e15 := 
sorry

end distance_light_travels_500_years_l328_328412


namespace triangle_area_is_correct_l328_328882

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_is_correct : 
  area_of_triangle (1, 3) (5, -2) (8, 6) = 23.5 := 
by
  sorry

end triangle_area_is_correct_l328_328882


namespace part_I_part_II_l328_328596

def f (x : ℝ) (m : ℝ) : ℝ := abs (x - m)

theorem part_I (m x : ℝ) : f (-x) m + f (1/x) m ≥ 2 :=
sorry

theorem part_II (m : ℝ) (a b c x : ℝ) (h_m : m = 1) (h_abc : a + b + c = 2/7) :
  f (log x / log 2) m + f (2 + log x / log 2) m > sqrt a + 2 * sqrt b + 3 * sqrt c ∧ 
  ((x > 2) ∨ (0 < x ∧ x < 1/2)) :=
sorry

end part_I_part_II_l328_328596


namespace sandy_attempts_sums_l328_328713

theorem sandy_attempts_sums (correct_sums : ℕ) (total_marks : ℕ)
  (marks_per_correct : ℕ) (marks_lost_per_incorrect : ℕ) (sums_attempted : ℕ) :
  correct_sums = 25 ∧ total_marks = 65 ∧ marks_per_correct = 3 ∧ marks_lost_per_incorrect = 2 ∧ 
  (sums_attempted = correct_sums + (total_marks - (correct_sums * marks_per_correct)) / marks_lost_per_incorrect) → 
  sums_attempted = 30 :=
by
  intro h
  cases h with h_sum_correct h_rest
  cases h_rest with h_total_marks h_more
  cases h_more with h_marks_correct h_more2
  cases h_more2 with h_marks_lost h_calc
  sorry

end sandy_attempts_sums_l328_328713


namespace count_valid_integers_l328_328173

def D (n : ℕ) : ℕ :=
  (nat.bits n).zip (nat.bits n).drop 1).count (λ x => x.1 ≠ x.2)

def valid_integers : ℕ := (list.range' 1 64).count (λ n, D(n) = 3)

theorem count_valid_integers : valid_integers = 5 :=
sorry

end count_valid_integers_l328_328173


namespace geometric_sequence_sum_l328_328359

def geometric_sequence_props (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  (∀ n, a n = a 1 * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (1 - 2^n) / (1 - 2)) ∧ 
  (a 5 - a 3 = 12) ∧ 
  (a 6 - a 4 = 24)

theorem geometric_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : geometric_sequence_props a S) :
  ∀ n, S n / a n = 2 - 2^(1 - n) :=
by
  sorry

end geometric_sequence_sum_l328_328359


namespace distribute_stickers_10_5_l328_328271

open Finset

def distribute_stickers (stickers sheets : ℕ) : ℕ :=
  (powerset (multiset.repeat 1 sheets)).filter (λ s, s.sum = stickers).card

-- Given the conditions: 10 stickers and 5 sheets
theorem distribute_stickers_10_5 : distribute_stickers 10 5 = 25 :=
by 
  have : multiset.sum (multiset.repeat 1 5) = 5 := by rw [multiset.sum_repeat, mul_one]
  
  -- Now we create the multiset of stickers and check all splits where the sum is 10
  sorry

end distribute_stickers_10_5_l328_328271


namespace pyramid_volume_correct_l328_328560

/-- Define a structure for regular triangular pyramid geometrical parameters. --/
structure RegularTriangularPyramid :=
  (height_midpoint_to_face : ℝ)
  (height_midpoint_to_edge : ℝ)
  (volume : ℝ)

/-- Define the given problem conditions and the target volume. --/
def pyramid_conditions : RegularTriangularPyramid := {
  height_midpoint_to_face := 2,
  height_midpoint_to_edge := Real.sqrt 12,
  volume := 374.12
}

/-- Theorem: The volume of the regular triangular pyramid with the given conditions. --/
theorem pyramid_volume_correct : pyramid_conditions.volume ≈ 374.12 :=
sorry

end pyramid_volume_correct_l328_328560


namespace common_difference_arithmetic_seq_l328_328749

theorem common_difference_arithmetic_seq (a1 d : ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n * a1 + n * (n - 1) / 2 * d) : 
  (S 5 / 5 - S 2 / 2 = 3) → d = 2 :=
by
  intros h1
  sorry

end common_difference_arithmetic_seq_l328_328749


namespace range_of_slope_angle_l328_328632

theorem range_of_slope_angle (l : ℝ → ℝ) (theta : ℝ) 
    (h_line_eqn : ∀ x y, l x = y ↔ x - y * Real.sin theta + 2 = 0) : 
    ∃ α : ℝ, α ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) :=
sorry

end range_of_slope_angle_l328_328632


namespace max_value_of_k_proof_l328_328620

noncomputable def maximum_value_of_k (x y k : ℝ) (h1: x > 0) (h2: y > 0) (h3: k > 0) 
(h4: 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) : Prop :=
  k = (-1 + Real.sqrt 17) / 2

-- This is the statement that needs to be proven:
theorem max_value_of_k_proof (x y k : ℝ) (h1: x > 0) (h2: y > 0) (h3: k > 0) 
(h4: 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) : maximum_value_of_k x y k h1 h2 h3 h4 :=
sorry

end max_value_of_k_proof_l328_328620


namespace expected_rank_of_winner_in_tournament_l328_328790

noncomputable def expected_winner_rank (n : ℕ) (p : ℝ) : ℝ :=
  2^n - 2^n * p + p

theorem expected_rank_of_winner_in_tournament :
  expected_winner_rank 8 (3/5) = 103 := by
    sorry

end expected_rank_of_winner_in_tournament_l328_328790


namespace acute_triangle_probability_correct_l328_328870

noncomputable def acute_triangle_probability : ℝ :=
  let l_cube_vol := 1
  let quarter_cone_vol := (1/4) * (1/3) * Real.pi * (1^2) * 1
  let total_unfavorable_vol := 3 * quarter_cone_vol
  let favorable_vol := l_cube_vol - total_unfavorable_vol
  favorable_vol / l_cube_vol

theorem acute_triangle_probability_correct : abs (acute_triangle_probability - 0.2146) < 0.0001 :=
  sorry

end acute_triangle_probability_correct_l328_328870


namespace number_of_sheep_l328_328178

-- Definitions based on conditions
def cows : ℕ := 4
def chickens : ℕ := 7
def bushels_for_chickens : ℕ := 3
def total_bushels/day : ℕ := 35
def bushels_per_sheep / day : ℕ := 2

-- Proof problem
theorem number_of_sheep:
  ∃ S : ℕ, (bushels_for_chickens + 2 * S = total_bushels/day) → S = 16 :=
sorry

end number_of_sheep_l328_328178


namespace problem_l328_328621

theorem problem (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, x^5 = a_0 + a_1 * (1 - x) + a_2 * (1 - x)^2 + a_3 * (1 - x)^3 + a_4 * (1 - x)^4 + a_5 * (1 - x)^5) →
  a_3 = -10 ∧ a_1 + a_3 + a_5 = -16 :=
by 
  sorry

end problem_l328_328621


namespace intersection_always_exists_minimum_chord_length_and_equation_l328_328929

open Real

noncomputable def circle_eq (x y : ℝ) : Prop :=
  4 * x^2 + 4 * y^2 - 4 * x - 8 * y - 11 = 0

noncomputable def line_eq (m x y : ℝ) : Prop :=
  (m - 1) * x + m * y = m + 1

theorem intersection_always_exists :
  ∀ (m : ℝ), ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
by
  sorry

theorem minimum_chord_length_and_equation :
  ∃ (k : ℝ) (x y : ℝ), k = sqrt 3 ∧ (3 * x - 2 * y + 7 = 0) ∧
    ∀ m, ∃ (xp yp : ℝ), line_eq m xp yp ∧ ∃ (l1 l2 : ℝ), line_eq m l1 l2 ∧ 
    (circle_eq xp yp ∧ circle_eq l1 l2)  :=
by
  sorry

end intersection_always_exists_minimum_chord_length_and_equation_l328_328929


namespace not_arithmetic_sequence_area_of_triangle_l328_328321

-- Define the geometric setup and given conditions
variables {A B C : ℝ} -- angles in triangle ABC
variables {a b c : ℝ} -- sides opposite to angles A, B, C respectively

-- Given conditions
axiom sine_condition : 4 * c * Real.sin C = (b + a) * (Real.sin B - Real.sin A)
axiom sides_condition : a^2 = 5 * c^2
axiom side_b : b = 3 * c
axiom perimeter : a + b + c = 4 + Real.sqrt 5

-- Prove that a, b, c cannot form an arithmetic sequence
theorem not_arithmetic_sequence : ¬(b = (a + c) / 2) := sorry

-- Prove the area of the triangle with given values
theorem area_of_triangle : 
  let cosA := (b^2 + c^2 - a^2) / (2 * b * c),
      sinA := Real.sqrt(1 - cosA^2)
  in (1 / 2) * b * c * sinA = Real.sqrt 11 / 4 := sorry

end not_arithmetic_sequence_area_of_triangle_l328_328321


namespace condition_pq_2m_l328_328245

-- Define arithmetic sequence and its properties
variables {n m p q : ℕ} (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Prove the statement: \(p+q=2m\) is a necessary but not sufficient condition for \(a_p + a_q = 2a_m\)
theorem condition_pq_2m (h : is_arithmetic_sequence a) (hp : 0 < p) (hq : 0 < q) (hm : 0 < m) :
  (p + q = 2 * m → a p + a q = 2 * a m) ∧ ¬(a p + a q = 2 * a m → p + q = 2 * m) :=
begin
  sorry
end

end condition_pq_2m_l328_328245


namespace tangent_slope_angle_range_l328_328933

theorem tangent_slope_angle_range
  (m n : ℝ) (h_m_pos : 0 < m) (h_n_pos : 0 < n)
  (h_prod : m * n = sqrt 3 / 2)
  (f : ℝ → ℝ) (h_f : ∀ x, f x = (1 / 3) * x^3 + n^2 * x) :
  ∃ θ, θ ∈ set.Ico (π / 3) (π / 2) ∧ (∀ x : ℝ, θ = real.arctan (f' m)) :=
sorry

end tangent_slope_angle_range_l328_328933


namespace find_tan_F_l328_328656

theorem find_tan_F (D E F : ℝ) (h_sum_angles : D + E + F = 180) 
  (h1 : Real.cot D * Real.cot F = 1 / 3) 
  (h2 : Real.cot E * Real.cot F = 1 / 12) : 
  Real.tan F = 35.579 :=
by
  sorry

end find_tan_F_l328_328656


namespace find_y_l328_328094

theorem find_y (x y : ℕ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : ∃ q : ℕ, x = q * y + 9) (h₃ : x / y = 96 + 3 / 20) : y = 60 :=
sorry

end find_y_l328_328094


namespace monotonic_f_range_of_a_l328_328601

def f (x : ℝ) : ℝ := x^2 + 2 / x

def g (x a : ℝ) : ℝ := (x^2 / (x^2 + 2 * x + 1)) + ((4 * x + 10) / (9 * x + 9)) - a

theorem monotonic_f : 
  ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f x1 > f x2 := 
by
  sorry

theorem range_of_a (a : ℝ) : 
  (∀ x1 : ℝ, 0 ≤ x1 → x1 ≤ 1 → 
    ∃ x2 : ℝ, (2 / 3) ≤ x2 → x2 ≤ 2 → g x1 a = f x2) → 
  -35 / 9 ≤ a ∧ a ≤ -2 :=
by
  sorry

end monotonic_f_range_of_a_l328_328601


namespace determine_k_for_intersection_l328_328189

theorem determine_k_for_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 3 = 2 * x + 5) ∧ 
  (∀ x₁ x₂ : ℝ, (k * x₁^2 + 2 * x₁ + 3 = 2 * x₁ + 5) ∧ 
                (k * x₂^2 + 2 * x₂ + 3 = 2 * x₂ + 5) → 
              x₁ = x₂) ↔ k = -1/2 :=
by
  sorry

end determine_k_for_intersection_l328_328189


namespace pascal_current_speed_l328_328382

variable (v : ℝ)
variable (h₁ : v > 0) -- current speed is positive

-- Conditions
variable (h₂ : 96 / (v - 4) = 96 / (1.5 * v) + 16)

-- Proving the speed
theorem pascal_current_speed (h₁ : v > 0) (h₂ : 96 / (v - 4) = 96 / (1.5 * v) + 16) : v = 8 :=
sorry

end pascal_current_speed_l328_328382


namespace initial_investment_amount_l328_328286

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem initial_investment_amount :
  let r := 0.08
  let n := 1
  let t := 18
  let A := 40000
  (A / (1 + r / n) ^ (n * t)) ≈ 10009.45 :=
by {
  sorry
}

end initial_investment_amount_l328_328286


namespace range_of_m_l328_328671

theorem range_of_m (m : ℝ) : 
  (¬(∃ (x : ℝ), x^2 + m * x + 1 = 0) ∧ (∀ (x : ℝ), 4 * x^2 + 4 * (m - 2) * x + 1 > 0)) → (1 < m ∧ m ≤ 2) := 
by 
  intro h
  cases h with h_np h_q
  sorry

end range_of_m_l328_328671


namespace function_range_l328_328438

noncomputable def f (x : ℝ) := Real.cos (x - (Real.pi / 3))

theorem function_range : ∀ x ∈ Icc 0 (Real.pi / 2), f x ∈ Icc (1 / 2) 1 :=
by
  intro x hx
  -- Therefore, use sorry as proof is not required.
  sorry

end function_range_l328_328438


namespace mary_turnips_grown_l328_328009

variable (sally_turnips : ℕ)
variable (total_turnips : ℕ)
variable (mary_turnips : ℕ)

theorem mary_turnips_grown (h_sally : sally_turnips = 113)
                          (h_total : total_turnips = 242) :
                          mary_turnips = total_turnips - sally_turnips := by
  sorry

end mary_turnips_grown_l328_328009


namespace count_whole_numbers_in_interval_l328_328978

-- Define the bounds of the interval
def lower_bound : Real := Real.sqrt 3
def upper_bound : Real := 3 * Real.pi

-- Prove the number of whole numbers in the interval is 8
theorem count_whole_numbers_in_interval :
  let numbers_in_interval := {n : ℕ | n > lower_bound ∧ n < upper_bound}
  numbers_in_interval.to_finset.card = 8 :=
by
  sorry  -- proof omitted

end count_whole_numbers_in_interval_l328_328978


namespace nested_sum_divisors_l328_328540

-- Define the sum of divisors function excluding n itself and prime divisors
def sum_divisors_excluding_primes (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d ≠ n ∧ ¬Nat.prime d) (Finset.divisors n)).sum

theorem nested_sum_divisors (n : ℕ) :
  sum_divisors_excluding_primes (sum_divisors_excluding_primes (sum_divisors_excluding_primes 10)) = 0 :=
by
  sorry

end nested_sum_divisors_l328_328540


namespace count_whole_numbers_in_interval_l328_328979

-- Define the bounds of the interval
def lower_bound : Real := Real.sqrt 3
def upper_bound : Real := 3 * Real.pi

-- Prove the number of whole numbers in the interval is 8
theorem count_whole_numbers_in_interval :
  let numbers_in_interval := {n : ℕ | n > lower_bound ∧ n < upper_bound}
  numbers_in_interval.to_finset.card = 8 :=
by
  sorry  -- proof omitted

end count_whole_numbers_in_interval_l328_328979


namespace coefficient_x3y3_in_polynomials_l328_328766

/--

Given the polynomials \( (x + y)^6 \) and \( \left(z + \frac{1}{z}\right)^8 \),
prove that the coefficient of \( x^3 y^3 \) in \( (x + y)^6 \left(z + \frac{1}{z}\right)^8 \) is \( 1400 \).

-/
theorem coefficient_x3y3_in_polynomials :
  coeff (monomial 3 1 x * monomial 3 1 y) ((x + y)^6 * (z + (1/z))^8) = 1400 :=
sorry

end coefficient_x3y3_in_polynomials_l328_328766


namespace max_f_l328_328683

noncomputable def f'' (x : ℝ) : ℝ :=
  - (1 / 4) * x ^ (-3 / 2) + 1 / (x ^ 2)

theorem max_f'' : ∃ x : ℝ, f'' x = 1 / 16 ∧ ∀ y : ℝ, f'' y ≤ 1 / 16 := 
  sorry

end max_f_l328_328683


namespace product_of_two_numbers_l328_328411

theorem product_of_two_numbers (x y : ℝ) (h_diff : x - y = 12) (h_sum_of_squares : x^2 + y^2 = 245) : x * y = 50.30 :=
sorry

end product_of_two_numbers_l328_328411


namespace largest_expression_l328_328370

theorem largest_expression (y : ℝ) (hy : y = 2 * 10 ^ (-1000)) :
  (∀ x ∈ {4 + y, 4 - y, 2 * y, 4 / y, y / 4}, x ≤ 4 / y) ∧ 
  (4 / y ∉ {4 + y, 4 - y, 2 * y, y / 4} → true) :=
by
  sorry

end largest_expression_l328_328370


namespace g_divisors_count_l328_328203

def g (n : ℕ) : ℕ := 2^n * 3^n

theorem g_divisors_count (n : ℕ) : 
  g(20) = 2^20 * 3^20 → 
  (n + 1) * (n + 1) = 441 := 
by
  intro h
  simp [h]
  exact 21 * 21

end g_divisors_count_l328_328203


namespace ensure_object_falls_within_interval_object_passes_through_P_l328_328793

-- Define the conditions
def passes_through_A (a : ℝ) (c : ℝ) := (0, 9) ∈ (λ x, a * x^2 + c)
def valid_trajectory (a : ℝ) := ∃ c, passes_through_A a c ∧ a < 0
def interval_D (a : ℝ) := -9/49 < a ∧ a < -1/4
def passes_through_P (a : ℝ) (c: ℝ) := (2, 8.1) ∈ (λ x, a * x^2 + c)
def point_P_value_a := (-9 / 40 : ℝ)

theorem ensure_object_falls_within_interval (a : ℝ) :
  valid_trajectory a → interval_D a :=
by sorry

theorem object_passes_through_P (a : ℝ) (c : ℝ) :
  passes_through_A a c → passes_through_P a c → interval_D point_P_value_a :=
by sorry

end ensure_object_falls_within_interval_object_passes_through_P_l328_328793


namespace range_of_a_l328_328208

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x : ℝ, if x > 1 then log a x else (a - 2) * x - 1

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ a ∈ (Set.Ioc 2 3) :=
by
  sorry

end range_of_a_l328_328208


namespace general_term_and_sum_sum_of_T_n_l328_328317

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {b : ℕ → ℝ} {T : ℕ → ℝ}

-- Conditions
def condition1 := S 5 = 25
def condition2 := a 10 = 19

-- General term formula and sum of first n terms
def general_term_a (n : ℕ) := 2 * n - 1
def sum_S (n : ℕ) := n^2

-- Sequence b_n and its sum T_n
def sequence_b (n : ℕ) := 1 / (a n * a (n + 1))
def sum_T (n : ℕ) := n / (2 * n + 1)

-- Proving the first part of the problem
theorem general_term_and_sum 
  (h1 : condition1) 
  (h2 : condition2) :
  (∀ n, a n = general_term_a n) ∧ (∀ n, S n = sum_S n) := 
sorry

-- Proving the second part of the problem
theorem sum_of_T_n
  (h1 : condition1) 
  (h2 : condition2)
  (h3 : ∀ n, a n = general_term_a n) :
  ∀ n, T n = sum_T n := 
sorry

end general_term_and_sum_sum_of_T_n_l328_328317


namespace number_of_arrangements_with_one_between_A_and_B_l328_328442

theorem number_of_arrangements_with_one_between_A_and_B :
  let people := ["A", "B", "C", "D", "E"] in
  let total_people := people.length in
  list.arrangements people total_people = 36 := sorry

end number_of_arrangements_with_one_between_A_and_B_l328_328442


namespace each_person_bid_count_l328_328160

-- Define the conditions and initial values
noncomputable def auctioneer_price_increase : ℕ := 5
noncomputable def initial_price : ℕ := 15
noncomputable def final_price : ℕ := 65
noncomputable def number_of_bidders : ℕ := 2

-- Define the proof statement
theorem each_person_bid_count : 
  (final_price - initial_price) / auctioneer_price_increase / number_of_bidders = 5 :=
by sorry

end each_person_bid_count_l328_328160


namespace jay_savings_after_expense_l328_328324

noncomputable def weekly_savings_goal_A : ℕ → ℕ
| 0       := 20
| (n + 1) := weekly_savings_goal_A n + 10

def total_savings_goal_A (weeks : ℕ) : ℕ :=
(0..weeks).sum weekly_savings_goal_A

noncomputable def weekly_savings_goal_B (week: ℕ) : ℕ :=
(500 * (5 + week)) / 100

def total_savings_goal_B (weeks : ℕ) : ℕ :=
(0..weeks).sum weekly_savings_goal_B

def total_savings (weeks : ℕ) : ℕ :=
total_savings_goal_A weeks + total_savings_goal_B weeks

theorem jay_savings_after_expense (weeks : ℕ) (expense : ℕ) :
  weeks = 4 ∧ expense = 75 → 
  total_savings weeks - expense = 195 := 
by
  intros
  sorry

end jay_savings_after_expense_l328_328324


namespace coefficient_x3y3_in_expansion_l328_328765

theorem coefficient_x3y3_in_expansion :
  (coeff_of_x3y3_in (x + y)^6) * (constant_term_in (z + 1/z)^8) = 1400 := by
sorry

end coefficient_x3y3_in_expansion_l328_328765


namespace map_distance_in_cm_l328_328693

theorem map_distance_in_cm(
  h1 : 1.5 / 24 = x / 302.3622047244094,
  h2 : 1 * 2.54 = 2.54
) : x * 2.54 = 48
:= 
by
  sorry

end map_distance_in_cm_l328_328693


namespace correlation_comparison_l328_328817

variable (x y U V : List ℝ)
variable (r1 r2 : ℝ)

-- Conditions
def data_xy := [(1,3), (2,5.3), (3,6.9), (4,9.1), (5,10.8)]
def data_UV := [(1,12.7), (2,10.2), (3,7), (4,3.6), (5,1)]

-- Definitions
def correlation_xy := r1
def correlation_UV := r2

theorem correlation_comparison (h1 : correlation_xy > 0) (h2 : correlation_UV < 0) : r2 < 0 < r1 := by
  sorry

end correlation_comparison_l328_328817


namespace solution_l328_328235

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 0 then a ^ x else log a x

noncomputable def problem_statement : Prop :=
  let a := (cos (420 * Real.pi / 180)) in
  let val1 := f (1 / 4) a in
  let val2 := f (Real.log 2 (1 / 6)) a in
  val1 + val2 = 8

theorem solution : problem_statement :=
  by
    sorry

end solution_l328_328235


namespace circumcenter_triangle_similarity_l328_328522

/-- 
Given a triangle ABC with points P, Q, S on sides AB, BC, CA respectively,
and O1, O2, O3 as the circumcenters of triangles APS, BQP, CSQ,
prove that the triangle formed by circumcenters O1O2O3 is similar to triangle ABC.
-/
theorem circumcenter_triangle_similarity (A B C P Q S O1 O2 O3 : Point)
  (hP : IsOnLine P A B) 
  (hQ : IsOnLine Q B C) 
  (hS : IsOnLine S C A)  
  (hO1 : IsCircumcenter O1 A P S) 
  (hO2 : IsCircumcenter O2 B Q P) 
  (hO3 : IsCircumcenter O3 C S Q) : 
  Similar (Triangle.mk O1 O2 O3) (Triangle.mk A B C) := 
sorry

end circumcenter_triangle_similarity_l328_328522


namespace Problem1_coefficient_of_x4_term_Problem2_binomial_identity_l328_328906

-- Problem (1)
def coefficient_of_x4_in_g : Nat := 56

theorem Problem1_coefficient_of_x4_term (x : ℚ) : 
    let f : (n : Nat) → ℚ := fun n => (1 + x) ^ n
    let g := f 4 + 2 * f 5 + 3 * f 6
    coefficient_of_x4_term : ∑ (k : ℕ) in (Finset.range 7).filter (fun k => k = 4), g = coefficient_of_x4_in_g :=
  sorry

-- Problem (2)
theorem Problem2_binomial_identity (m n : Nat) (h : m > 0) : 
    ∑ k in Finset.range n + 1, (k : ℚ) * (Nat.choose (m + k) k) = (m + 2) * n / (m + 3) * (Nat.choose (m + n + 1) (m + 2)) :=
  sorry

end Problem1_coefficient_of_x4_term_Problem2_binomial_identity_l328_328906


namespace intersection_eq_l328_328222

variable (A : Set ℤ) (B : Set ℤ)

def A_def := A = {-1, 0, 1, 2}
def B_def := B = {x | -1 < x ∧ x < 2}

theorem intersection_eq : A ∩ B = {0, 1} :=
by
  have A_def : A = {-1, 0, 1, 2} := sorry
  have B_def : B = {x | -1 < x ∧ x < 2} := sorry
  sorry

end intersection_eq_l328_328222


namespace hotel_loss_l328_328119
  
  -- Conditions
  def operations_expenses : ℝ := 100
  def total_payments : ℝ := (3 / 4) * operations_expenses
  
  -- Theorem to prove
  theorem hotel_loss : operations_expenses - total_payments = 25 :=
  by
    sorry
  
end hotel_loss_l328_328119


namespace complement_U_A_l328_328999

open Set

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {x | x^2 < 3}

theorem complement_U_A :
  (U \ A) = {-2, 2} :=
sorry

end complement_U_A_l328_328999


namespace starting_player_wins_l328_328479

structure Table :=
  (width : ℝ)
  (height : ℝ)
  (non_negative_size : width > 0 ∧ height > 0)

structure Coin :=
  (radius : ℝ)
  (non_negative_radius : radius > 0)

structure Position :=
  (x : ℝ)
  (y : ℝ)

def center_reflection (pos : Position) : Position :=
  ⟨-pos.x, -pos.y⟩

axiom non_overlapping {t : Table} {c : Coin} (positions : list Position) : Prop :=
  ∀ i j, i ≠ j → (positions.nth i).distance (positions.nth j) ≥ 2 * c.radius

def valid_move_sequence (t : Table) (c : Coin) (positions : list Position) : Prop :=
  ∀ pos ∈ positions, abs pos.x ≤ t.width / 2 ∧ abs pos.y ≤ t.height / 2 ∧
  positions.length ≤ (t.width * t.height / (π * c.radius^2))

theorem starting_player_wins
  (t : Table)
  (c : Coin)
  (valid_seq : valid_move_sequence t c)
  (non_overlap : non_overlapping valid_seq)
  : ∃ first_player_wins : bool, first_player_wins = tt :=
begin
  sorry -- proof to be filled in
end

end starting_player_wins_l328_328479


namespace perpendicular_vect_collinear_vect_l328_328969

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (3, x)
def b : ℝ × ℝ := (-2, 2)

-- Problem 1: Perpendicular vectors a and b
-- If a • b = 0, show that x = 3
theorem perpendicular_vect (x : ℝ) (h : a x.1 * b.1 + a x.2 * b.2 = 0) : x = 3 := sorry

-- Problem 2: Collinear vectors (b - a) and (3a + 2b)
-- If b - a is collinear with 3a + 2b, show that x = -3
theorem collinear_vect (x : ℝ) 
    (h : ∃ k : ℝ, (b.1 - (a x.1), b.2 - (a x.2)) = k • (3 * a x.1 + 2 * b.1, 3 * a x.2 + 2 * b.2)) : x = -3 := sorry

end perpendicular_vect_collinear_vect_l328_328969


namespace number_of_teams_l328_328416

def girls : Nat := 3
def boys : Nat := 5
def team_girls : Nat := 2
def team_boys : Nat := 2

theorem number_of_teams : choose girls team_girls * choose boys team_boys = 30 := by
  sorry

end number_of_teams_l328_328416


namespace sigma_power_of_two_l328_328087

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ k : ℕ, p = 2^k - 1

def product_of_distinct_mersenne_primes (n : ℕ) : Prop :=
  ∃ (ps : List ℕ), (∀ p ∈ ps, is_mersenne_prime p) ∧ (ps.nodup ∧ ps.prod = n)

theorem sigma_power_of_two (n : ℕ) (h : product_of_distinct_mersenne_primes n) :
  ∃ m : ℕ, sigma n = 2^m := sorry

end sigma_power_of_two_l328_328087


namespace binary_ternary_conversion_l328_328994

theorem binary_ternary_conversion (a b : ℕ) (h_b : b = 0 ∨ b = 1) (h_a : a = 0 ∨ a = 1 ∨ a = 2)
  (h_eq : 8 + 2 * b + 1 = 9 * a + 2) : 2 * a + b = 3 :=
by
  sorry

end binary_ternary_conversion_l328_328994


namespace sum_of_angles_l328_328592

theorem sum_of_angles (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2: 0 < β ∧ β < π / 2) 
  (h3 : Real.sin α = (√10) / 10) (h4 : Real.cos β = (2 * √5) / 5) : α + β = π / 4 := 
by
  sorry

end sum_of_angles_l328_328592


namespace find_y_in_exponent_equation_l328_328866

theorem find_y_in_exponent_equation :
  ∃ y : ℕ, 8^3 + 8^3 + 8^3 + 8^3 = 2^y ∧ y = 11 :=
begin
  use 11,
  split,
  { have h1 : 8 = 2^3 := by norm_num,
    have h2 : 8^3 = (2^3)^3 := by congr,
    have h3 : (2^3)^3 = 2^(3 * 3) := by rw [←pow_mul],
    rw [h2, h3, pow_mul],
    norm_num,
  },
  { refl },
end

end find_y_in_exponent_equation_l328_328866


namespace arithmetic_sequences_ratio_l328_328589

theorem arithmetic_sequences_ratio
  (a b : ℕ → ℕ)
  (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = (n * (2 * (a 1) + (n - 1) * (a 2 - a 1))) / 2)
  (h2 : ∀ n, T n = (n * (2 * (b 1) + (n - 1) * (b 2 - b 1))) / 2)
  (h3 : ∀ n, (S n) / (T n) = (2 * n + 2) / (n + 3)) :
  (a 10) / (b 9) = 2 := sorry

end arithmetic_sequences_ratio_l328_328589


namespace a_n_values_l328_328577

noncomputable def a : ℕ → ℕ := sorry
noncomputable def S : ℕ → ℕ := sorry

axiom Sn_property (n : ℕ) (hn : n > 0) : S n = 2 * (a n) - n

theorem a_n_values : a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 7 ∧ ∀ n : ℕ, n > 0 → a n = 2^n - 1 := 
by sorry

end a_n_values_l328_328577


namespace f_2010_eq_0_l328_328240

theorem f_2010_eq_0 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x) (h2 : ∀ x : ℝ, f (x + 2) = f x) : 
  f 2010 = 0 :=
by sorry

end f_2010_eq_0_l328_328240


namespace floor_eq_correct_l328_328554

theorem floor_eq_correct (y : ℝ) (h : ⌊y⌋ + y = 17 / 4) : y = 9 / 4 :=
sorry

end floor_eq_correct_l328_328554


namespace AN_divides_bisector_of_C_in_half_l328_328844

-- Definitions based on the problem's conditions
variables (A B C N : Point)
variables (h1 : right_angle A B C)
variables (h2 : is_midpoint_of_semicircle N C B)

-- Theorem statement
theorem AN_divides_bisector_of_C_in_half : ∃ T, bisects_angle A N T C ∧ divides_interval N C T h1 h2 := 
sorry

end AN_divides_bisector_of_C_in_half_l328_328844


namespace area_of_square_is_121_square_cm_l328_328165

-- Define the entities involved
variables (A B C D E F G H I J K B' N B'')

-- Define the properties and conditions
def is_square (a b c d : Point) : Prop := 
  -- Assuming an appropriate definition for square
  sorry

def right_isosceles_triangle (a b d : Point) : Prop :=
  -- Assuming an appropriate definition for right isosceles triangle
  sorry

def translation (a b c : Point) (d : ℝ) : Point :=
  -- Assuming an appropriate definition for translation
  sorry

def area_of_parallelogram (a b c d : Point) : ℝ :=
  -- Assuming an appropriate definition for the area of a parallelogram
  sorry

-- Given conditions based on the problem
axiom square_ABCD : is_square A B C D
axiom triangle_ABD : right_isosceles_triangle A B D
axiom translation_ABD_to_EFG : translation A E (translation B F (translation D G 3))
axiom translation_EFG_to_HIJ : translation E H (translation F I (translation G J 5))
axiom equal_overlapping_area : 
  area_of_parallelogram B K N B' = area_of_parallelogram B O J L

-- The main statement to be proved
theorem area_of_square_is_121_square_cm :
  ∃ (s : ℝ), s = 11 → s ^ 2 = 121 :=
begin
  sorry
end

end area_of_square_is_121_square_cm_l328_328165


namespace markup_constant_relationship_l328_328302

variable (C S : ℝ) (k : ℝ)
variable (fractional_markup : k * S = 0.25 * C)
variable (relation : S = C + k * S)

theorem markup_constant_relationship (fractional_markup : k * S = 0.25 * C) (relation : S = C + k * S) :
  k = 1 / 5 :=
by
  sorry

end markup_constant_relationship_l328_328302


namespace exist_ten_distinct_nat_numbers_mean_gcd_six_times_not_exist_ten_distinct_nat_numbers_mean_gcd_five_times_l328_328191

-- Case (a)
theorem exist_ten_distinct_nat_numbers_mean_gcd_six_times : 
  ∃ (numbers : Fin 10 → ℕ), 
    (∀ i j, i ≠ j → numbers i ≠ numbers j) ∧ 
    (arithmeticMean numbers = 6 * gcd_of_list (numbers.toList)) :=
sorry

-- Case (b)
theorem not_exist_ten_distinct_nat_numbers_mean_gcd_five_times : 
  ¬ ∃ (numbers : Fin 10 → ℕ), 
    (∀ i j, i ≠ j → numbers i ≠ numbers j) ∧ 
    (arithmeticMean numbers = 5 * gcd_of_list (numbers.toList)) :=
sorry

-- Helper definitions
def arithmeticMean (numbers : Fin 10 → ℕ) : ℕ :=
  (numbers.toList.sum) / 10

def gcd_of_list (l : List ℕ) : ℕ :=
  l.foldr gcd 0

end exist_ten_distinct_nat_numbers_mean_gcd_six_times_not_exist_ten_distinct_nat_numbers_mean_gcd_five_times_l328_328191


namespace proof_problem_l328_328958

def f (x : ℝ) : ℝ := if h : (0 < x ∧ x < 1) then 2^x else 
                     if h : (-1 < x ∧ x < 0) then -(2^(-x)) else 
                     f (x % 2)

lemma f_periodic : ∀ x : ℝ, f (x + 2) = f x := sorry
lemma f_odd : ∀ x : ℝ, f (-x) = -f x := sorry
lemma f_def_0_1 : ∀ x : ℝ, 0 < x → x < 1 → f x = 2^x := sorry

theorem proof_problem (L : ℝ) (h_log : L = real.logb (1/2) 23) : 
  f L = - (23 / 16) := 
by {
 have hL : L = -(real.log 23 / real.log 2) := by simp [h_log, real.logb, real.log_div, real.log_inv], 
 sorry
}

end proof_problem_l328_328958


namespace parallel_lines_when_m_is_neg7_l328_328964

-- Given two lines l1 and l2 defined as:
def l1 (m : ℤ) (x y : ℤ) := (3 + m) * x + 4 * y = 5 - 3 * m
def l2 (m : ℤ) (x y : ℤ) := 2 * x + (5 + m) * y = 8

-- The proof problem to show that l1 is parallel to l2 when m = -7
theorem parallel_lines_when_m_is_neg7 :
  ∃ m : ℤ, (∀ x y : ℤ, l1 m x y → l2 m x y) → m = -7 := 
sorry

end parallel_lines_when_m_is_neg7_l328_328964


namespace percentage_increase_in_second_year_l328_328744

-- Defining the conditions for the population increase problem
def initial_population := 1200
def final_population := 1950
def first_year_increase := 0.25

-- The final population after the first year's increase
def population_after_first_year := initial_population * (1 + first_year_increase)

-- The proof statement that the percentage increase in the second year is 30%
theorem percentage_increase_in_second_year : 
  (∃ P : ℝ, (population_after_first_year * (1 + P / 100) = final_population) ∧ P = 30) :=
by
  use 30
  calc
    population_after_first_year * (1 + 30 / 100)
          = population_after_first_year * 1.3 : by norm_num
      ... = 1500 * 1.3 : by rw [population_after_first_year, show initial_population * (1 + first_year_increase) = 1500, from by norm_num]
      ... = final_population : by norm_num
  done

end percentage_increase_in_second_year_l328_328744


namespace parabola_behavior_l328_328259

theorem parabola_behavior (x : ℝ) (h : x < 0) : ∃ y, y = 2*x^2 - 1 ∧ ∀ x1 x2 : ℝ, x1 < x2 ∧ x1 < 0 ∧ x2 < 0 → (2*x1^2 - 1) > (2*x2^2 - 1) :=
by
  sorry

end parabola_behavior_l328_328259


namespace ladder_slide_distance_approximation_l328_328799

noncomputable def ladderSlide (ladderLength initialFootDistance slipDistance : ℝ) : ℝ := 
  let initialHeight := Real.sqrt (ladderLength^2 - initialFootDistance^2)
  let newHeight := initialHeight - slipDistance
  let newFootDistance := Real.sqrt (ladderLength^2 - newHeight^2)
  newFootDistance - initialFootDistance

theorem ladder_slide_distance_approximation :
  ladderSlide 30 11 5 ≈ 3.7 :=
sorry

end ladder_slide_distance_approximation_l328_328799


namespace triangle_PQR_equilateral_centroid_PQR_on_AB_l328_328000

-- Definitions of points and conditions
variables {A B M C D E P Q R : Type} [Point A] [Point B] [Point M] [Point C] [Point D] [Point E] [Point P] [Point Q] [Point R]
variables {AB : Line A B} {MC : Line M C} {MD : Line M D} {ME : Line M E}
variables (M_on_AB : M ∈ AB) (M_neq_A : M ≠ A) (M_neq_B : M ≠ B)
variables (C_pos : C ∉ AB) (D_pos : D ∉ AB) (E_pos : E ∉ AB)
variables (tri_ABC : EquilateralTriangle A B C) (tri_AMD : EquilateralTriangle A M D) (tri_MBE : EquilateralTriangle M B E)
variables (medians_ABC : MediansIntersections ABC P) (medians_AMD : MediansIntersections AMD Q) (medians_MBE : MediansIntersections MBE R)
variables (medians_PQR : MediansIntersections PQR R')

-- Statements to prove
theorem triangle_PQR_equilateral (h1: EquilateralTriangle A B C) 
                                  (h2: EquilateralTriangle A M D) 
                                  (h3: EquilateralTriangle M B E) 
                                  (hmed_ABC: MediansIntersections ABC P) 
                                  (hmed_AMD: MediansIntersections AMD Q) 
                                  (hmed_MBE: MediansIntersections MBE R) :
  EquilateralTriangle P Q R := sorry

theorem centroid_PQR_on_AB (h1: EquilateralTriangle A B C) 
                           (h2: EquilateralTriangle A M D) 
                           (h3: EquilateralTriangle M B E) 
                           (hmed_ABC: MediansIntersections ABC P) 
                           (hmed_AMD: MediansIntersections AMD Q) 
                           (hmed_MBE: MediansIntersections MBE R) 
                           (h_tri_PQR: EquilateralTriangle P Q R) :
  CentroidOnSegment P Q R A B := sorry

end triangle_PQR_equilateral_centroid_PQR_on_AB_l328_328000


namespace problem_solution_l328_328080

-- Definitions based on the conditions
def forward_and_backward_opposite : Prop := 
  ∀d : ℝ, d > 0 → (forward d = - backward d)

def income_and_expenditure_opposite : Prop :=
  ∀in_exp : ℝ × ℝ, (in_exp.1 > 0 ∧ in_exp.2 > 0) → (income in_exp.1 = - expenditure in_exp.2)

def east_and_north_not_opposite : Prop :=
  ∀d1 d2 : ℝ, (d1 > 0 ∧ d2 > 0) → (east d1 ≠ - north d2)

def exceeding_and_falling_short_opposite : Prop :=
  ∀e : ℝ, ∀f : ℝ, (e > 0 ∧ f > 0) → (exceed e = - fall_short f)

-- Theorem stating the solution
theorem problem_solution : east_and_north_not_opposite :=
begin
  sorry
end

end problem_solution_l328_328080


namespace total_questions_reviewed_l328_328834

theorem total_questions_reviewed (questions_per_student : ℕ) (students_per_class : ℕ) (number_of_classes : ℕ) :
  questions_per_student = 10 → students_per_class = 35 → number_of_classes = 5 →
  questions_per_student * students_per_class * number_of_classes = 1750 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  norm_num
  sorry

end total_questions_reviewed_l328_328834


namespace stable_marriage_exists_l328_328444

-- Define boys and girls
variables {n : ℕ}
variables (B : fin n → Type)
variables (G : fin n → Type)

-- Define preferences as functions mapping each individual to a strict order on the other gender
variables (boy_prefers : Π (b : fin n), list (fin n)) -- Each boy ranks the girls
variables (girl_prefers : Π (g : fin n), list (fin n)) -- Each girl ranks the boys

-- Stable pairing proof
theorem stable_marriage_exists (B : fin n → Type) (G : fin n → Type)
  (boy_prefers : Π (b : fin n), list (fin n))
  (girl_prefers : Π (g : fin n), list (fin n)) :
  ∃ (pairing : fin n → fin n), (∀ b1 b2 g1 g2 : fin n, 
    (boy_prefers b1).index_of g2 < (boy_prefers b1).index_of g1 →
    (girl_prefers g2).index_of b1 < (girl_prefers g2).index_of b2 →
    pairing b1 = g1 → pairing b2 = g2 → false) :=
sorry

end stable_marriage_exists_l328_328444


namespace limit_cot_2x_l328_328559

open Real

theorem limit_cot_2x : tendsto (fun x => x * (cot (2 * x))) (𝓝 0) (𝓝 (1/2)) :=
sorry

end limit_cot_2x_l328_328559


namespace student_average_age_l328_328407

-- Define known quantities given in the problem
def numberOfStudents : ℕ := 30
def teacherAge : ℕ := 46
def combinedAverageAge : ℕ := 16
def totalPeople : ℕ := 31

-- Define the proof statement
theorem student_average_age :
  let totalAgeOf31People := totalPeople * combinedAverageAge in
  let totalAgeOfStudents := totalAgeOf31People - teacherAge in
  let averageAgeOfStudents := totalAgeOfStudents / numberOfStudents in
  averageAgeOfStudents = 15 :=
by
  -- Calculation steps included in the theorem for clarity
  let totalAgeOf31People := 31 * 16
  let totalAgeOfStudents := totalAgeOf31People - 46
  let averageAgeOfStudents := totalAgeOfStudents / 30
  show averageAgeOfStudents = 15
  from sorry

end student_average_age_l328_328407


namespace conic_section_is_parabola_l328_328774

def conic_section_type (x y : ℝ) : Prop := 
  | (2 * y + 3) = real.sqrt ((x - 3) ^ 2 + (2 * y) ^ 2)

theorem conic_section_is_parabola (x y : ℝ) : conic_section_type x y → "P" :=
sorry

end conic_section_is_parabola_l328_328774


namespace probability_white_marbles_l328_328549

variable {a b x y: ℕ}
variable {total_marbles: ℕ} (h_total: total_marbles = 30)
variable (h_prob_black: ((x: ℤ)/(a: ℤ)) * ((y: ℤ)/(b: ℤ)) = (1/2))
variable (h_x_eqn: x = 3 * y)
variable (h_a_b_eqn: a + b = total_marbles)

theorem probability_white_marbles : 
  (a: ℤ) > 0 → (b: ℤ) > 0 → 
  0 < y ∧ 3 * y ≤ a ∧ 0 < y ∧ y ≤ b → 
  ( (a - 3 * y): ℤ / a * (b - y): ℤ / b = (1/3) ) :=
sorry

end probability_white_marbles_l328_328549


namespace square_geometry_l328_328001

/-- Given a square ABCD and a point E on side CD, let the angle bisector
of ∠BAE intersect side BC at point F. Prove that AE = ED + BF. -/
theorem square_geometry 
  (A B C D E F : Point)
  (is_square : is_square A B C D)
  (E_on_CD : E ∈ line_segment C D)
  (F_on_bisector : ∃ F, is_angle_bisector (line A E) (line A B) (line_segment B C)) : 
  dist A E = dist E D + dist B F :=
sorry

end square_geometry_l328_328001


namespace arithmetic_sequence_formula_l328_328582

theorem arithmetic_sequence_formula {a_n : ℕ → ℤ} (x : ℤ)
    (h₀ : a_n 1 = x - 1)
    (h₁ : a_n 2 = x + 1)
    (h₂ : a_n 3 = 2x + 3) :
    ∃ f : ℕ → ℤ, ∀ n : ℕ, a_n n = f n ∧ f n = 2n - 3 :=
by
sory

end arithmetic_sequence_formula_l328_328582


namespace find_p_q_r_l328_328667

variables {R : Type*} [Field R] [Module R (EuclideanSpace R (Fin 3))]
variables (a b c : EuclideanSpace R (Fin 3))
variables (p q r : R)

noncomputable def magnitude (v : EuclideanSpace R (Fin 3)) : R :=
(∑ i, (v i) ^ 2) ^ (1/2 : R)

-- Given conditions
def are_mutually_orthogonal (u v w : EuclideanSpace R (Fin 3)) : Prop :=
(u ⬝ v = 0) ∧ (v ⬝ w = 0) ∧ (u ⬝ w = 0)

def given_conditions : Prop :=
  are_mutually_orthogonal a b c ∧
  magnitude a = 2 ∧
  magnitude b = 3 ∧
  magnitude c = 4 ∧
  a = p • (a ×ₑ b) + q • (b ×ₑ c) + r • (c ×ₑ a) ∧
  a ⬝ (b ×ₑ c) = 24

-- Proof statement
theorem find_p_q_r (h : given_conditions a b c p q r) : p + q + r = 1/6 := 
sorry

end find_p_q_r_l328_328667


namespace sigma_algebra_inequality_l328_328096

noncomputable section

open MeasureTheory

-- defining the Bernoulli random variable property
variables {Ω : Type*} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}

-- Assuming xi_n are independent Bernoulli random variables
variables (xi : ℕ → Ω → ℤ) (X : ℕ → Ω → ℤ)
variable (ℙ : MeasureTheory.ProbabilityMeasure Ω)
variables [∀ n, MeasureTheory.Independent_ (σ (xi n)) (ℙ)]
variables [∀ n, MeasureTheory.isProbabilityMeasure (σ (xi n)) ℙ]

-- Definitions from the problem
def is_bernoulli (xi : Ω → ℤ) : Prop :=
  (ℙ (λ ω, xi ω = -1) = 1/2) ∧ (ℙ (λ ω, xi ω = 1) = 1/2)

def X_n (n : ℕ) (ω : Ω) := ∏ i in finset.range (n + 1), xi i ω

-- Definitions of sigma algebras
def G : MeasurableSpace Ω := MeasurableSpace.generateFrom { s | ∃ n, s = {ω | xi n ω ∈ {-1, 1} }}
def E_n (n : ℕ) : MeasurableSpace Ω := MeasurableSpace.generateFrom { s | ∃ k ≥ n, s = {ω | X k ω ∈ {-1, 1} }}

-- Question: Proving the inequality of sigma algebras
theorem sigma_algebra_inequality (h_bernoulli : ∀ n, is_bernoulli (xi n)) :
  (⋂ n, MeasurableSpace.generateFrom { G, E_n n }) ≠ MeasurableSpace.generateFrom { G, ⋂ n, E_n n } :=
sorry

end sigma_algebra_inequality_l328_328096


namespace cos_double_angle_l328_328623

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7 / 25 := 
sorry

end cos_double_angle_l328_328623


namespace volunteer_distribution_l328_328483

theorem volunteer_distribution :
  let num_volunteers := 5
  let num_schools := 3
  (∀ s, 1 ≤ s) → -- Each school receives at least one volunteer
  (∃ d, d = 150) :=
by
  let num_volunteers := 5
  let num_schools := 3
  let condition := λ (s : ℕ), 1 ≤ s
  have d := 150
  use d
  sorry

end volunteer_distribution_l328_328483


namespace sum_of_y_values_is_120_l328_328923

theorem sum_of_y_values_is_120 (y1 y2 y3 y4 y5 : ℝ) :
  let x_values := [3, 5, 7, 12, 13]
      y_values := [y1, y2, y3, y4, y5]
      regression_line := λ x, (1/2 : ℝ) * x + 20
      x_bar := (3 + 5 + 7 + 12 + 13) / 5
      y_bar := (1/2) * x_bar + 20
  in y_bar = (y1 + y2 + y3 + y4 + y5) / 5 →
     y1 + y2 + y3 + y4 + y5 = 120 := by
  sorry

end sum_of_y_values_is_120_l328_328923


namespace div_by_5_l328_328714

theorem div_by_5 (n : ℕ) (hn : 0 < n) : (2^(4*n+1) + 3) % 5 = 0 := 
by sorry

end div_by_5_l328_328714


namespace sphere_diameter_form_l328_328142

-- Given conditions
def radius_small_sphere := 6
def volume_small_sphere : ℝ := (4 / 3) * Real.pi * (radius_small_sphere : ℝ) ^ 3
def volume_large_sphere : ℝ := 3 * volume_small_sphere
def radius_large_sphere := Real.cbrt (volume_large_sphere * (3 / (4 * Real.pi)))
def diameter_large_sphere := 2 * radius_large_sphere

-- Proving the form a*real_cbrt(b) with a=12 and b=3, and computing a + b
theorem sphere_diameter_form :
  ∃ (a b : ℝ), ∃ (a_int b_int : ℤ), 0 < a_int ∧ 0 < b_int ∧
  a_int = 12 ∧ b_int = 3 ∧
  volume_large_sphere = 3 * volume_small_sphere ∧
  diameter_large_sphere = a * Real.cbrt b ∧ a + b = 15 := by
sorry

end sphere_diameter_form_l328_328142


namespace sqrt_inequality_convex_quadrilateral_l328_328305

variable {A B C D E : Type}
variable {F_1 F_2 F : ℝ}
variable [ConvexQuadrilateral A B C D]
variable [Area_Triangle_ABE : A B E → F_1]
variable [Area_Triangle_CDE : C D E → F_2]
variable [Area_Quadrilateral_ABCD : A B C D → F]

theorem sqrt_inequality_convex_quadrilateral
    (convex : ConvexQuadrilateral A B C D)
    (intersection_point : E = DiagonalIntersection A C B D)
    (area_ABE : Area_Triangle A B E = F_1)
    (area_CDE : Area_Triangle C D E = F_2)
    (total_area : Area_Quadrilateral A B C D = F) :
    sqrt F_1 + sqrt F_2 ≤ sqrt F ∧ (sqrt F_1 + sqrt F_2 = sqrt F ↔ AD || BC) :=
by {
    sorry
}

end sqrt_inequality_convex_quadrilateral_l328_328305


namespace rare_numbers_divisors_of_24_l328_328183

open Nat

def is_rare (n : ℕ) : Prop :=
  ∀ (ps : List ℕ), ps.length = n → (∀ p ∈ ps, Prime p ∧ p > 3) → n ∣ (ps.map (λ p, p * p)).sum

theorem rare_numbers_divisors_of_24 (n : ℕ) : is_rare n ↔ n ∣ 24 := 
sorry

end rare_numbers_divisors_of_24_l328_328183


namespace each_person_bids_five_times_l328_328156

noncomputable def auction_bidding : Prop :=
  let initial_price := 15
  let final_price := 65
  let price_increase_per_bid := 5
  let number_of_bidders := 2
  let total_increase := final_price - initial_price
  let total_bids := total_increase / price_increase_per_bid
  total_bids / number_of_bidders = 5

theorem each_person_bids_five_times : auction_bidding :=
by
  -- The proof will be filled in here.
  sorry

end each_person_bids_five_times_l328_328156


namespace find_smallest_b_l328_328027

noncomputable def smallest_b (g : ℝ → ℝ) (h_periodic : ∀ x, g (x - 30) = g x) : ℝ :=
  let b := 120
  in if ∀ x, g ((x - b) / 4) = g (x / 4) then b else 0

theorem find_smallest_b (g : ℝ → ℝ) (h_periodic : ∀ x, g (x - 30) = g x) :
  smallest_b g h_periodic = 120 :=
by
  sorry

end find_smallest_b_l328_328027


namespace triangle_OMN_area_l328_328260

noncomputable def rho (theta : ℝ) : ℝ := 4 * Real.cos theta + 2 * Real.sin theta

theorem triangle_OMN_area :
  let l1 (x y : ℝ) := y = (Real.sqrt 3 / 3) * x
  let l2 (x y : ℝ) := y = Real.sqrt 3 * x
  let C (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 5
  let OM := 2 * Real.sqrt 3 + 1
  let ON := 2 + Real.sqrt 3
  let angle_MON := Real.pi / 6
  let area_OMN := (1 / 2) * OM * ON * Real.sin angle_MON
  (4 * (Real.sqrt 3 + 2) + 5 * Real.sqrt 3 = 8 + 5 * Real.sqrt 3) → 
  area_OMN = (8 + 5 * Real.sqrt 3) / 4 :=
sorry

end triangle_OMN_area_l328_328260


namespace find_a_l328_328255

def f (x a : ℝ) : ℝ := Real.logb 3 (x^2 - a)

theorem find_a (a : ℝ) (h : f 2 a = 1) : a = 1 :=
by
  sorry

end find_a_l328_328255


namespace number_of_spoons_bought_l328_328169

-- Definitions for the problem conditions
def plates_bought := 9
def cost_per_plate := 2
def total_cost := 24
def cost_per_spoon := 1.5

-- Main statement to prove
theorem number_of_spoons_bought : 
  9 * 2 + n * 1.5 = 24 -> ∃ n : ℕ, n = 4 :=
begin
  -- sorry to skip the proof for now
  sorry
end

end number_of_spoons_bought_l328_328169


namespace standard_eq_ellipse_trajectory_eq_midpoint_M_max_area_triangle_ABC_l328_328583

open Real

-- Conditions provided
def ellipse_center := (0, 0)
def left_focus := (-sqrt 3, 0)
def right_vertex := (2, 0)
def point_A := (1, 1/2)
-- Parameters found: a = 2, c = sqrt(3), b = 1

-- Standard equation of the ellipse
theorem standard_eq_ellipse :
  ∀ (x y : ℝ), (x, y) ∈ set_of (λ p : ℝ × ℝ, 
    (p.1 ^ 2) / 4 + p.2 ^ 2 = 1) :=
sorry

-- Trajectory equation of the midpoint M
theorem trajectory_eq_midpoint_M :
  ∀ (x y : ℝ), x = (2 * x - 1) ∧ y = (2 * y - 1/2) →
  (x - 1/2) ^ 2 + 4 * (y - 1/4) ^ 2 = 1 :=
sorry

-- Maximum area of the triangle ABC
theorem max_area_triangle_ABC :
  ∀ (B C : ℝ × ℝ),
  let k := B.2 / B.1 in 
  B = (2 / sqrt (4 * k ^ 2 + 1), 2 * k / sqrt (4 * k ^ 2 + 1)) ∨
  B = (-2 / sqrt (4 * k ^ 2 + 1), -2 * k / sqrt (4 * k ^ 2 + 1)) →
  let area := abs (2 * k - 1) / sqrt (1 + 4 * k ^ 2) in 
  area ≤ sqrt(2) :=
sorry

end standard_eq_ellipse_trajectory_eq_midpoint_M_max_area_triangle_ABC_l328_328583


namespace leo_assignment_third_part_time_l328_328338

-- Define all the conditions as variables
def first_part_time : ℕ := 25
def first_break : ℕ := 10
def second_part_time : ℕ := 2 * first_part_time
def second_break : ℕ := 15
def total_time : ℕ := 150

-- The calculated total time of the first two parts and breaks
def time_spent_on_first_two_parts_and_breaks : ℕ :=
  first_part_time + first_break + second_part_time + second_break

-- The remaining time for the third part of the assignment
def third_part_time : ℕ :=
  total_time - time_spent_on_first_two_parts_and_breaks

-- The theorem to prove that the time Leo took to finish the third part is 50 minutes
theorem leo_assignment_third_part_time : third_part_time = 50 := by
  sorry

end leo_assignment_third_part_time_l328_328338


namespace gray_region_correct_b_l328_328143

-- Define the basic conditions
def square_side_length : ℝ := 3
def small_square_side_length : ℝ := 1

-- Define the triangles resulting from cutting a square
def triangle_area : ℝ := 0.5 * square_side_length * square_side_length

-- Define the gray region area for the second figure (b)
def gray_region_area_b : ℝ := 0.25

-- Lean statement to prove the area of the gray region
theorem gray_region_correct_b : gray_region_area_b = 0.25 := by
  -- Proof is omitted
  sorry

end gray_region_correct_b_l328_328143


namespace inequality_abc_l328_328392

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b - c) * (b + c - a) * (c + a - b) ≤ a * b * c := 
sorry

end inequality_abc_l328_328392


namespace circle_tangent_unique_point_l328_328584

theorem circle_tangent_unique_point (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → (x+4)^2 + (y-a)^2 = 25 → false) →
  (a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 ∨ a = 0) :=
by
  sorry

end circle_tangent_unique_point_l328_328584


namespace digit_in_607th_place_l328_328470

theorem digit_in_607th_place : 
  let decimal_rep := "368421052631578947".to_list in
  (decimal_rep.nth ((607 % 18) - 1)) = some '1' := 
by 
  let cycle_length := 18
  let repeating_decimals := "368421052631578947".to_list
  have h1 : 607 % cycle_length = 13 := by norm_num
  show repeating_decimals.nth (13 - 1) = some '1' from sorry

end digit_in_607th_place_l328_328470


namespace solve_quartic_eq_l328_328556

theorem solve_quartic_eq : 
  {x : ℂ | x^4 + 81 = 0} = {3 + 3i, -3 - 3i, -3 + 3i, 3 - 3i} :=
sorry

end solve_quartic_eq_l328_328556


namespace two_absent_one_present_probability_l328_328638

-- Define the probabilities
def probability_absent_normal : ℚ := 1 / 15

-- Given that the absence rate on Monday increases by 10%
def monday_increase_factor : ℚ := 1.1

-- Calculate the probability of being absent on Monday
def probability_absent_monday : ℚ := probability_absent_normal * monday_increase_factor

-- Calculate the probability of being present on Monday
def probability_present_monday : ℚ := 1 - probability_absent_monday

-- Define the probability that exactly two students are absent and one present
def probability_two_absent_one_present : ℚ :=
  3 * (probability_absent_monday ^ 2) * probability_present_monday

-- Convert the probability to a percentage and round to the nearest tenth
def probability_as_percent : ℚ := round (probability_two_absent_one_present * 100 * 10) / 10

theorem two_absent_one_present_probability : probability_as_percent = 1.5 := by sorry

end two_absent_one_present_probability_l328_328638


namespace dress_designs_possible_l328_328115

theorem dress_designs_possible (colors patterns fabric_types : Nat) (color_choices : colors = 5) (pattern_choices : patterns = 6) (fabric_type_choices : fabric_types = 2) : 
  colors * patterns * fabric_types = 60 := by 
  sorry

end dress_designs_possible_l328_328115


namespace irrational_R_over_r_l328_328663

noncomputable def circumradius (a b c K : ℝ) : ℝ := (a * b * c) / (4 * K)
noncomputable def inradius (K s : ℝ) : ℝ := K / s
noncomputable def R_over_r (a b c K s : ℝ) : ℝ := (a * b * c * s) / (4 * K^2)

theorem irrational_R_over_r {a b c K s : ℝ} (h_lattice_points : K ∈ ℤ ∨ K ∈ (ℤ.cast ℝ / 2))
  (h_sqrt_n : ∃ (n : ℕ), a = Real.sqrt n ∧ Nat.square_free n) :
  Irrational (R_over_r a b c K s) :=
sorry

end irrational_R_over_r_l328_328663


namespace intersection_condition_l328_328461

noncomputable def line (k x : ℝ) : ℝ := k * x - 1

noncomputable def discriminant (k : ℝ) : ℝ := (2 * k) ^ 2 - 4 * (1 - k ^ 2) * (-5)

theorem intersection_condition (k : ℝ) : (x : ℝ) → (y : ℝ) → (k = ±(Real.sqrt 5 / 2)) 
(∃! p, p ∈ {p : ℝ × ℝ | p.1^2 - p.2^2 = 4 ∧ p.2 = line k p.1}) :=
by
  sorry

end intersection_condition_l328_328461


namespace scientific_notation_correct_l328_328164

theorem scientific_notation_correct :
  ∀ (n : ℕ), n = 239000000 → (∃ a : ℝ, ∃ b : ℤ, n = a * 10 ^ b ∧ a = 2.39 ∧ b = 8) :=
by
  intro n
  assume h : n = 239000000
  use 2.39
  use 8
  split
  { 
    sorry
  }
  split
  {
    sorry
  }
  {
    sorry
  }

end scientific_notation_correct_l328_328164


namespace factor_expression_l328_328553

variable (b : ℤ)

theorem factor_expression : 280 * b^2 + 56 * b = 56 * b * (5 * b + 1) :=
by
  sorry

end factor_expression_l328_328553


namespace find_unknown_number_l328_328287

theorem find_unknown_number (x n : ℚ) (h1 : n + 7/x = 6 - 5/x) (h2 : x = 12) : n = 5 :=
by
  sorry

end find_unknown_number_l328_328287


namespace general_formula_sum_of_terms_l328_328214

variable {n : ℕ+}
variable {S : ℕ+ → ℚ}
variable {a : ℕ+ → ℚ}
variable {T : ℕ+ → ℚ}

-- Conditions
axiom h1 : ∀ n, S n = 1 - a n
axiom h2 : ∀ n, a n = 1 / 2^n

-- Proof Problem 1: Prove a_n = 1 / 2^n
theorem general_formula (n : ℕ+) : a n = 1 / 2^n :=
by sorry

-- Proof Problem 2: Prove T_n = (2^(n+1) - n - 2) / 2^n
theorem sum_of_terms (n : ℕ+) : T n = (2^(n+1) - n - 2) / 2^n :=
by sorry

end general_formula_sum_of_terms_l328_328214


namespace number_of_acceptable_outfits_l328_328980

theorem number_of_acceptable_outfits :
  ∃ (shirts pants hats : ℕ) (common_colors : ℕ), 
  shirts = 7 ∧ pants = 5 ∧ hats = 7 ∧ common_colors = 4 →
  let total_outfits := shirts * pants * hats,
      restricted_outfits := common_colors * 1 * (shirts - common_colors + 2) * (hats - common_colors + 2) in
  total_outfits - restricted_outfits = 229 :=
by sorry

end number_of_acceptable_outfits_l328_328980


namespace solve_inequality_correct_l328_328720

noncomputable def solve_inequality (a : ℝ) : set ℝ :=
if h : a = 0 then
  { x : ℝ | 1 < x}
else if h : a = 1 then
  ∅
else if a < 0 then
  { x : ℝ | x < 1 / a } ∪ { x : ℝ | 1 < x }
else if 0 < a ∧ a < 1 then
  { x : ℝ | 1 < x ∧ x < 1 / a }
else -- a > 1
  { x : ℝ | 1 / a < x ∧ x < 1 }

theorem solve_inequality_correct (a : ℝ) :
  solve_inequality a = 
    if a = 0 then { x | 1 < x }
    else if a < 0 then { x | x < 1 / a } ∪ { x | 1 < x }
    else if 0 < a ∧ a < 1 then { x | 1 < x ∧ x < 1 / a }
    else if a > 1 then { x | 1 / a < x ∧ x < 1 }
    else ∅ :=
sorry

end solve_inequality_correct_l328_328720


namespace find_numbers_l328_328692

theorem find_numbers (x y z t : ℕ) 
  (h1 : x + t = 37) 
  (h2 : y + z = 36) 
  (h3 : x + z = 2 * y) 
  (h4 : y * t = z * z) : 
  x = 12 ∧ y = 16 ∧ z = 20 ∧ t = 25 :=
by
  sorry

end find_numbers_l328_328692


namespace distances_product_equal_l328_328798

theorem distances_product_equal {n : ℕ} (A : Fin (2 * n) → Point) (M : Point) (p : Fin (2 * n) → ℝ) (h : ∀ i, p i = distance_from_point_to_side M (A i) (A ((i + 1) % (2 * n)))) :
  ∏ i in (range (2 * n)).filter (λ k, odd k), p i = ∏ i in (range (2 * n)).filter (λ k, even k), p i := 
sorry

end distances_product_equal_l328_328798


namespace trigonometric_identity_l328_328797

noncomputable def cos_sequence_sum (n : ℕ) : ℝ :=
  (∑ j in finset.range n, (-1) ^ (j + 1) * (Real.cos (j * Real.pi / n)) ^ n)

theorem trigonometric_identity (n : ℕ) (h : n ≥ 1) :
  cos_sequence_sum n = n / 2 ^ (n - 1) :=
sorry

end trigonometric_identity_l328_328797


namespace final_total_cost_is_12_70_l328_328467

-- Definitions and conditions
def sandwich_count : ℕ := 2
def sandwich_cost_per_unit : ℝ := 2.45

def soda_count : ℕ := 4
def soda_cost_per_unit : ℝ := 0.87

def chips_count : ℕ := 3
def chips_cost_per_unit : ℝ := 1.29

def sandwich_discount : ℝ := 0.10
def sales_tax : ℝ := 0.08

-- Final price after discount and tax
noncomputable def total_cost : ℝ :=
  let sandwiches_total := sandwich_count * sandwich_cost_per_unit
  let discounted_sandwiches := sandwiches_total * (1 - sandwich_discount)
  let sodas_total := soda_count * soda_cost_per_unit
  let chips_total := chips_count * chips_cost_per_unit
  let subtotal := discounted_sandwiches + sodas_total + chips_total
  let final_total := subtotal * (1 + sales_tax)
  final_total

theorem final_total_cost_is_12_70 : total_cost = 12.70 :=
by 
  sorry

end final_total_cost_is_12_70_l328_328467


namespace thermal_engine_efficiency_l328_328077

noncomputable def P1 (P0 ω t : ℝ) : ℝ :=
  P0 * (Real.sin (ω * t) / (100 + Real.sin (t ^ 2)))

noncomputable def P2 (P0 ω t : ℝ) : ℝ :=
  3 * P0 * (Real.sin (2 * ω * t) / (100 + Real.sin ((2 * t) ^ 2)))

noncomputable def A_plus (P0 ω : ℝ) : ℝ :=
  ∫ t in 0..(π / ω), P1 P0 ω t

noncomputable def A_minus (P0 ω : ℝ) : ℝ :=
  ∫ t in 0..(π / (2 * ω)), P2 P0 ω t

theorem thermal_engine_efficiency (P0 ω : ℝ) :
  (∫ t in 0..(π / ω), P1 P0 ω t) - (∫ t in 0..(π / (2 * ω)), P2 P0 ω t) = 
  (1 / 3) * (∫ t in 0..(π / ω), P1 P0 ω t) :=
by
  sorry

end thermal_engine_efficiency_l328_328077


namespace quadrilateral_with_equal_angles_is_parallelogram_l328_328082

axiom Quadrilateral (a b c d : Type) : Prop
axiom Parallelogram (a b c d : Type) : Prop
axiom equal_angles (a b c d : Type) : Prop

theorem quadrilateral_with_equal_angles_is_parallelogram 
  (a b c d : Type) 
  (q : Quadrilateral a b c d)
  (h : equal_angles a b c d) : Parallelogram a b c d := 
sorry

end quadrilateral_with_equal_angles_is_parallelogram_l328_328082


namespace tangent_line_of_circle_l328_328320
-- Import the required libraries

-- Define the given condition of the circle in polar coordinates
def polar_circle (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta

-- Define the property of the tangent line in polar coordinates
def tangent_line (rho theta : ℝ) : Prop :=
  rho * Real.cos theta = 4

-- State the theorem to be proven
theorem tangent_line_of_circle (rho theta : ℝ) (h : polar_circle rho theta) :
  tangent_line rho theta :=
sorry

end tangent_line_of_circle_l328_328320


namespace each_person_bid_count_l328_328158

-- Define the conditions and initial values
noncomputable def auctioneer_price_increase : ℕ := 5
noncomputable def initial_price : ℕ := 15
noncomputable def final_price : ℕ := 65
noncomputable def number_of_bidders : ℕ := 2

-- Define the proof statement
theorem each_person_bid_count : 
  (final_price - initial_price) / auctioneer_price_increase / number_of_bidders = 5 :=
by sorry

end each_person_bid_count_l328_328158


namespace xiao_ying_pe_grade_l328_328112

-- Definitions for the conditions
def activity_weight := 0.3
def theory_weight := 0.2
def skills_weight := 0.5

def activity_score := 90
def theory_score := 80
def skills_score := 94

-- The calculation of the physical education grade
def calc_pe_grade (activity_weight activity_score theory_weight theory_score skills_weight skills_score : ℝ) : ℝ :=
  activity_weight * activity_score + theory_weight * theory_score + skills_weight * skills_score

-- The main theorem
theorem xiao_ying_pe_grade :
  calc_pe_grade activity_weight activity_score theory_weight theory_score skills_weight skills_score = 90 :=
by
  -- Proof should go here, but we'll use sorry to indicate it's not yet implemented
  sorry

end xiao_ying_pe_grade_l328_328112


namespace coin_toss_sequences_count_l328_328975

theorem coin_toss_sequences_count :
  ∃ (n : ℕ), 
  (∃ (HH HT TH TT : ℕ), 
   HH = 3 ∧ HT = 4 ∧ TH = 5 ∧ TT = 6 ∧ 
   (∃ (seqs : list (list char)), 
    seqs.length = 18 ∧ 
    -- Count the subsequences 
    seqs.count_substring "HH" = HH ∧ 
    seqs.count_substring "HT" = HT ∧ 
    seqs.count_substring "TH" = TH ∧ 
    seqs.count_substring "TT" = TT)
  ) ∧ n = 4200 :=
begin
  sorry
end

end coin_toss_sequences_count_l328_328975


namespace applicant_overall_score_l328_328808

-- Definitions for the conditions
def writtenTestScore : ℝ := 80
def interviewScore : ℝ := 60
def weightWrittenTest : ℝ := 0.6
def weightInterview : ℝ := 0.4

-- Theorem statement
theorem applicant_overall_score : 
  (writtenTestScore * weightWrittenTest) + (interviewScore * weightInterview) = 72 := 
by
  sorry

end applicant_overall_score_l328_328808


namespace t_plus_inv_t_eq_three_l328_328908

theorem t_plus_inv_t_eq_three {t : ℝ} (h : t^2 - 3 * t + 1 = 0) (hnz : t ≠ 0) : t + 1 / t = 3 :=
sorry

end t_plus_inv_t_eq_three_l328_328908


namespace chocolate_syrup_amount_l328_328534

noncomputable def total_glasses (total_ounces : ℕ) (glass_size : ℕ) : ℕ :=
  total_ounces / glass_size

noncomputable def total_chocolate_syrup (glasses : ℕ) (syrup_per_glass : ℕ) : ℕ :=
  glasses * syrup_per_glass

theorem chocolate_syrup_amount :
  let
    milk_per_glass := 6.5
    syrup_per_glass := 1.5
    milk_total := 130.0
    chocolate_milk_total := 160
    glass_size := 8
    glasses := total_glasses chocolate_milk_total glass_size
  in
  total_chocolate_syrup glasses syrup_per_glass = 30 := by
  sorry

end chocolate_syrup_amount_l328_328534


namespace manufacturing_firm_min_workers_l328_328496

noncomputable def min_workers_to_profit (maintenance_cost : ℕ) (wage_per_hour : ℕ) (widgets_per_hour : ℕ) (price_per_widget : ℕ) (workday_hours : ℕ) (required_profit: ℕ): ℕ :=
  let daily_cost := maintenance_cost + workday_hours * wage_per_hour in
  let revenue_per_worker := widgets_per_hour * price_per_widget in
  let total_daily_revenue := workday_hours * n * revenue_per_worker in
  have h : total_daily_revenue > daily_cost, from sorry
  21

theorem manufacturing_firm_min_workers :
  min_workers_to_profit 800 20 4 4 10 0 = 21 := 
  by {
    unfold min_workers_to_profit,
    dsimp,
    -- The proof can be written here
    sorry
  }

end manufacturing_firm_min_workers_l328_328496


namespace professor_oscar_review_questions_l328_328839

-- Define the problem conditions.
def students_per_class : ℕ := 35
def questions_per_exam : ℕ := 10
def number_of_classes : ℕ := 5

-- Define the number of questions that must be reviewed.
def total_questions_to_review : ℕ := 1750

-- The theorem to be proved.
theorem professor_oscar_review_questions :
  students_per_class * questions_per_exam * number_of_classes = total_questions_to_review :=
by
  -- Here we write 'sorry' since we are not providing the full proof.
  sorry

end professor_oscar_review_questions_l328_328839


namespace shaded_area_of_squares_l328_328819

theorem shaded_area_of_squares :
  let s_s := 4
  let s_L := 9
  let area_L := s_L * s_L
  let area_s := s_s * s_s
  area_L - area_s = 65 := sorry

end shaded_area_of_squares_l328_328819


namespace total_tin_is_117_5_l328_328796

def weight_alloy_A : ℝ := 100
def ratio_lead_tin_A : ℝ := 5 / 3
def weight_alloy_B : ℝ := 200
def ratio_tin_copper_B : ℝ := 2 / 3

theorem total_tin_is_117_5 :
  let tin_weight_A := (3 / (5 + 3)) * weight_alloy_A in
  let tin_weight_B := (2 / (2 + 3)) * weight_alloy_B in
  tin_weight_A + tin_weight_B = 117.5 :=
by
  sorry

end total_tin_is_117_5_l328_328796


namespace max_strip_length_is_correct_l328_328823

-- Definitions matching the given conditions
def diameter_base : ℝ := 20
def slant_height : ℝ := 20
def strip_width : ℝ := 2

-- Computation related definitions
def radius_base : ℝ := diameter_base / 2
def circumference_base : ℝ := 2 * Real.pi * radius_base
def semicircle_radius : ℝ := slant_height
def maximum_strip_length : ℝ := 2 * (Real.sqrt (semicircle_radius^2 - strip_width^2))

-- The theorem to prove
theorem max_strip_length_is_correct : maximum_strip_length ≈ 39.7 :=
by sorry

end max_strip_length_is_correct_l328_328823


namespace total_distance_l328_328526

-- Definitions for the given problem conditions
def Beka_distance : ℕ := 873
def Jackson_distance : ℕ := 563
def Maria_distance : ℕ := 786

-- Theorem that needs to be proved
theorem total_distance : Beka_distance + Jackson_distance + Maria_distance = 2222 := by
  sorry

end total_distance_l328_328526


namespace geometric_seq_ratio_l328_328360

theorem geometric_seq_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 5 - a 3 = 12) 
  (h2 : a 6 - a 4 = 24) 
  (ha : ∃ a₁ : ℝ, (∀ n : ℕ, a n = a₁ * q ^ (n - 1)))
  (hS : ∀ n : ℕ, S n = a₁ * (1 - q ^ n) / (1 - q)) :
  ∀ n : ℕ, S n / a n = 2 - 2 ^ (1 - n) :=
sorry

end geometric_seq_ratio_l328_328360


namespace no_values_of_expression_l328_328427

theorem no_values_of_expression (x : ℝ) (h : x^2 - 4 * x + 4 < 0) :
  ¬ ∃ y, y = x^2 + 4 * x + 5 :=
by
  sorry

end no_values_of_expression_l328_328427


namespace problem_statement_l328_328340

variable {α : Type*} [field α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0     := a₁
| (n+1) := geometric_sequence a₁ q n * q

def Sn (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * n else a₁ * (1 - q ^ n) / (1 - q)

theorem problem_statement (a₁ q : α) (n : ℕ) (h₀ : q ≠ 1)
  (h₁ : geometric_sequence a₁ q 4 - geometric_sequence a₁ q 2 = 12)
  (h₂ : geometric_sequence a₁ q 5 - geometric_sequence a₁ q 3 = 24) :
  Sn a₁ q n / geometric_sequence a₁ q (n - 1) = 2 - 2 ^ (1 - n) :=
sorry

end problem_statement_l328_328340


namespace total_bowling_balls_l328_328035

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l328_328035


namespace intersection_A_B_l328_328223

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < 2 - x ∧ 2 - x < 3 }

theorem intersection_A_B :
  A ∩ B = {0, 1} := sorry

end intersection_A_B_l328_328223


namespace cost_of_basic_calculator_l328_328135

variable (B S G : ℕ)

theorem cost_of_basic_calculator 
  (h₁ : S = 2 * B)
  (h₂ : G = 3 * S)
  (h₃ : B + S + G = 72) : 
  B = 8 :=
by
  sorry

end cost_of_basic_calculator_l328_328135


namespace ratio_of_B_to_A_investment_l328_328512

-- Definitions
variable {x m : ℝ}
variable {total_profit a_share b_share c_share : ℝ}
variable (investment_ratio : ℝ)

-- Conditions
def conditions :=
  total_profit = 18300 ∧
  a_share = 6100 ∧
  let b_profit := (mx * (6/12)) in
  let c_profit := (3x * (4/12)) in
  let a_profit := (x * 12/12) in
  (a_profit / (a_profit + b_profit + c_profit)) = (a_share / total_profit)

-- Proposition: Prove that the ratio of B's investment to A's investment is 3:1
theorem ratio_of_B_to_A_investment (h : conditions) : investment_ratio = 3 :=
  sorry

end ratio_of_B_to_A_investment_l328_328512


namespace cot_neg_45_l328_328878

theorem cot_neg_45 (cot_def : ∀ θ : ℝ, Real.cot θ = 1 / Real.tan θ)
  (tan_neg : ∀ θ : ℝ, Real.tan (-θ) = -Real.tan θ)
  (tan_45 : Real.tan (Real.pi / 4) = 1) :
  Real.cot (-Real.pi / 4) = -1 :=
by
  -- proof goes here
  sorry

end cot_neg_45_l328_328878


namespace surface_dots_sum_l328_328104

-- Define the sum of dots on opposite faces of a standard die
axiom sum_opposite_faces (x y : ℕ) : x + y = 7

-- Define the large cube dimensions
def large_cube_dimension : ℕ := 3

-- Define the total number of small cubes
def num_small_cubes : ℕ := large_cube_dimension ^ 3

-- Calculate the number of faces on the surface of the large cube
def num_surface_faces : ℕ := 6 * large_cube_dimension ^ 2

-- Given the sum of opposite faces, compute the total number of dots on the surface
theorem surface_dots_sum : num_surface_faces / 2 * 7 = 189 := by
  sorry

end surface_dots_sum_l328_328104


namespace number_of_extra_postages_l328_328727

def length_to_height_ratio (length : ℕ) (height : ℕ) : ℚ := length / height

def requires_extra_postage (ratio : ℚ) : Bool :=
  ratio < 1.2 ∨ ratio > 2.8

def envelope_a_requires_postage := requires_extra_postage (length_to_height_ratio 7 5)
def envelope_b_requires_postage := requires_extra_postage (length_to_height_ratio 10 4)
def envelope_c_requires_postage := requires_extra_postage (length_to_height_ratio 8 8)
def envelope_d_requires_postage := requires_extra_postage (length_to_height_ratio 14 5)

def number_of_envelopes_requiring_postage : ℕ :=
  [envelope_a_requires_postage, envelope_b_requires_postage, envelope_c_requires_postage, envelope_d_requires_postage].count (fun x => x = true)

theorem number_of_extra_postages : number_of_envelopes_requiring_postage = 1 := sorry

end number_of_extra_postages_l328_328727


namespace problem_equivalent_l328_328811

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (2 * a - x) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem problem_equivalent :
  has_period (λ x, real.sin (2 * x - real.pi / 6)) real.pi ∧
  symmetric_about (λ x, real.sin (2 * x - real.pi / 6)) (real.pi / 3) ∧
  increasing_on_interval (λ x, real.sin (2 * x - real.pi / 6)) (-real.pi / 6) (real.pi / 3) :=
sorry

end problem_equivalent_l328_328811


namespace point_P_movement_l328_328659

theorem point_P_movement (P : ℤ) (h : P = -4) : P - 2 = -6 :=
by {
  rw h,
  exact rfl,
}

end point_P_movement_l328_328659


namespace max_sales_increase_year_l328_328154

def sales1994 := 30
def sales1995 := 36
def sales1996 := 45
def sales1997 := 50
def sales1998 := 65
def sales1999 := 70
def sales2000 := 88
def sales2001 := 90
def sales2002 := 85
def sales2003 := 75

theorem max_sales_increase_year:
  ∀ year, (year > 1994 ∧ year ≤ 2003) →
  ((sales1995 - sales1994 = 6) ∧ (sales1996 - sales1995 = 9) ∧
   (sales1997 - sales1996 = 5) ∧ (sales1998 - sales1997 = 15) ∧
   (sales1999 - sales1998 = 5) ∧ (sales2000 - sales1999 = 18) ∧
   (sales2001 - sales2000 = 2) ∧ (sales2002 - sales2001 = -5) ∧
   (sales2003 - sales2002 = -10) →
   year = 2000) := by sorry

end max_sales_increase_year_l328_328154


namespace hotel_loss_l328_328126

theorem hotel_loss :
  (ops_expenses : ℝ) (payment_frac : ℝ) (total_received : ℝ) (loss : ℝ)
  (h_ops_expenses : ops_expenses = 100)
  (h_payment_frac : payment_frac = 3 / 4)
  (h_total_received : total_received = payment_frac * ops_expenses)
  (h_loss : loss = ops_expenses - total_received) :
  loss = 25 :=
by
  sorry

end hotel_loss_l328_328126


namespace find_a_given_condition_l328_328938

theorem find_a_given_condition (a : ℝ) (i : ℂ) (h1 : i = complex.I) (h2 : ∣((a + i) / i)∣ = 2) : 
  a = real.sqrt 3 ∨ a = -real.sqrt 3 :=
by
  sorry

end find_a_given_condition_l328_328938


namespace arithmetic_sequence_sum_l328_328648

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_sum
  (h1 : a 1 = 2)
  (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l328_328648


namespace Jonathan_total_exercise_time_l328_328333

theorem Jonathan_total_exercise_time :
  (let monday_speed := 2
     let wednesday_speed := 3
     let friday_speed := 6
     let distance := 6
     let monday_time := distance / monday_speed
     let wednesday_time := distance / wednesday_speed
     let friday_time := distance / friday_speed
   in monday_time + wednesday_time + friday_time = 6)
:= sorry

end Jonathan_total_exercise_time_l328_328333


namespace count_numbers_with_odd_tens_digit_of_cube_l328_328895

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def cubes_tens_digit_odd (n : ℕ) : Prop :=
  (tens_digit (n^3)) % 2 = 1

theorem count_numbers_with_odd_tens_digit_of_cube :
  (finset.range 150).filter cubes_tens_digit_odd).card = 30 := 
sorry

end count_numbers_with_odd_tens_digit_of_cube_l328_328895


namespace inequality_solution_l328_328721

theorem inequality_solution (x : ℝ) :
  ((2 / (x - 1)) - (3 / (x - 3)) + (2 / (x - 4)) - (2 / (x - 5)) < (1 / 15)) ↔
  (x < -1 ∨ (1 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (7 < x ∧ x < 8)) :=
by
  sorry

end inequality_solution_l328_328721


namespace each_person_bids_five_times_l328_328157

noncomputable def auction_bidding : Prop :=
  let initial_price := 15
  let final_price := 65
  let price_increase_per_bid := 5
  let number_of_bidders := 2
  let total_increase := final_price - initial_price
  let total_bids := total_increase / price_increase_per_bid
  total_bids / number_of_bidders = 5

theorem each_person_bids_five_times : auction_bidding :=
by
  -- The proof will be filled in here.
  sorry

end each_person_bids_five_times_l328_328157


namespace inversion_sphere_to_sphere_l328_328705

-- Define the necessary geometric properties and structures
variable (O : Point) (S : Sphere) (r : ℝ)

-- Condition 1: Sphere does not contain the center of inversion
def sphere_not_contain_center (S : Sphere) (O : Point) : Prop :=
  ¬ O ∈ S

-- Condition 2: Define inversion properties
def inversion_property (A A* B B* X X* O: Point) (r: ℝ) : Prop :=
  (dist O A * dist O A* = r^2) ∧
  (dist O B * dist O B* = r^2) ∧
  (dist O X * dist O X* = r^2)

-- Problem statement
theorem inversion_sphere_to_sphere (O : Point) (S : Sphere) (r : ℝ)
  (h1: sphere_not_contain_center S O)
  (inversion_property: ∀ (A A* B B* X X* : Point), inversion_property A A* B B* X X* O r) :
  ∃ S' : Sphere, ∀ (X : Point), X ∈ S ↔ X* ∈ S' :=
sorry

end inversion_sphere_to_sphere_l328_328705


namespace general_formula_sum_bn_l328_328246

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {b : ℕ → ℕ}

-- Condition 1
axiom a_n_arith : ∀ n, a n + 1 = a (n + 1)
-- Condition (1).1 -- We assume one of the conditions to complete the problem, e.g., (1)
axiom a_3_plus_a_15 : a 3 + a 15 = 20

-- Given $S_{n}$ is the sum of the first $n$ terms
axiom sum_S : ∀ n, S (n + 1) = S n + a n + 1

-- Given $b_n = a_n - 1$
def b (n : ℕ) : ℕ := a n - 1

noncomputable def T (n : ℕ) : ℕ := (Finset.sum (Finset.range n) (λ k, 2^(k+1) * k))

theorem general_formula : (∀ n, a n = n + 1) :=
by
  sorry

theorem sum_bn : ∀ n, T n = 2 + (n-1)*2^(n+1) :=
by
  sorry

end general_formula_sum_bn_l328_328246


namespace tan_of_internal_angle_l328_328588

theorem tan_of_internal_angle (α : ℝ) (h1 : α < π) (h2 : α > 0) (h3 : cos α = -3/5) : tan α = -4/3 :=
by
  sorry

end tan_of_internal_angle_l328_328588


namespace fraction_comparison_l328_328437

theorem fraction_comparison : (9 / 16) > (5 / 9) :=
by {
  sorry -- the detailed proof is not required for this task
}

end fraction_comparison_l328_328437


namespace problem_statement_l328_328970

variables (a b : ℝ × ℝ)

def vector_a := (1, -1)
def vector_b (x : ℝ) := (x, 2)

def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

def vector_magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem problem_statement :
  is_perpendicular (1, -1) (x, 2) → vector_magnitude (vector_add (1, -1) (x, 2)) = real.sqrt 10 := 
by 
  sorry

end problem_statement_l328_328970


namespace prob_two_hits_in_three_shots_l328_328136

theorem prob_two_hits_in_three_shots (p_hit : ℝ) (p_miss : ℝ) (n : ℕ) (k : ℕ) (h : 0 < p_hit ∧ p_hit < 1 ∧ p_miss = 1 - p_hit ∧ n = 3 ∧ k = 2) : 
  (nat.choose n k : ℝ) * (p_hit ^ k) * (p_miss ^ (n - k)) = 54 / 125 :=
by
  sorry

end prob_two_hits_in_three_shots_l328_328136


namespace binomial_30_3_l328_328853

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l328_328853


namespace hexagon_FX_length_l328_328717

theorem hexagon_FX_length (A B C D E F X : Point) (length_AB : Real) (h1 : is_regular_hexagon A B C D E F) (h2 : dist A B = 3) (h3 : collinear A B X) (h4 : dist A X = 4 * dist A B) :
  dist F X = 3 * sqrt 20.75 :=
by sorry

end hexagon_FX_length_l328_328717


namespace cubic_sum_l328_328991

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 :=
  sorry

end cubic_sum_l328_328991


namespace second_number_is_correct_l328_328782

theorem second_number_is_correct (A B C : ℝ) 
  (h1 : A + B + C = 157.5)
  (h2 : A / B = 14 / 17)
  (h3 : B / C = 2 / 3)
  (h4 : A - C = 12.75) : 
  B = 18.75 := 
sorry

end second_number_is_correct_l328_328782


namespace remainder_T_2023_mod_14_l328_328894

-- Define the sequence T(n) with the given constraints.
def T : ℕ → ℕ
| 0       := 1
| 1       := 2
| 2       := 4
| n + 1   := (T n + T (n - 1)) % 14

theorem remainder_T_2023_mod_14 : T 2023 % 14 = 8 := by
  sorry

end remainder_T_2023_mod_14_l328_328894


namespace expected_difference_l328_328527

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7
def is_composite (n : ℕ) : Prop := n = 4 ∨ n = 6 ∨ n = 8

def roll_die : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def probability_eat_sweetened : ℚ := 4 / 7
def probability_eat_unsweetened : ℚ := 3 / 7
def days_in_leap_year : ℕ := 366

def expected_days_unsweetened : ℚ := probability_eat_unsweetened * days_in_leap_year
def expected_days_sweetened : ℚ := probability_eat_sweetened * days_in_leap_year

theorem expected_difference :
  expected_days_sweetened - expected_days_unsweetened = 52.28 := by
  sorry

end expected_difference_l328_328527


namespace sqrt_of_square_of_neg_four_l328_328718

theorem sqrt_of_square_of_neg_four : sqrt ((-4)^2) = 4 :=
by
  sorry

end sqrt_of_square_of_neg_four_l328_328718


namespace sad_girls_count_l328_328691

-- Given definitions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 22
def girls : ℕ := 38
def happy_boys : ℕ := 6
def boys_neither_happy_nor_sad : ℕ := 10

-- Intermediate definitions
def sad_boys : ℕ := boys - happy_boys - boys_neither_happy_nor_sad
def sad_girls : ℕ := sad_children - sad_boys

-- Theorem to prove that the number of sad girls is 4
theorem sad_girls_count : sad_girls = 4 := by
  sorry

end sad_girls_count_l328_328691


namespace f_72_eq_2p_plus_3q_l328_328942

variable {α : Type*} [CommMonoid α] (f : α → ℤ)
variable (q p : ℤ)

-- Conditions
axiom f_mul : ∀ (a b : α), f (a * b) = f a + f b
axiom f_2 : f 2 = q
axiom f_3 : f 3 = p

-- Theorem statement
theorem f_72_eq_2p_plus_3q : f 72 = 2 * p + 3 * q :=
sorry

end f_72_eq_2p_plus_3q_l328_328942


namespace equal_number_of_colored_triangles_l328_328809

theorem equal_number_of_colored_triangles
    (n : ℕ) (h_even : Even n) (h_ge : n ≥ 4)
    (good_diag : Π (x y : Fin n), Prop)
    (good_diag_def :
      ∀ (x y : Fin n), good_diag x y ↔ ((y.val - x.val + 1) % 2 = 1))
    (triangulation : List (Fin n × Fin n × Fin n))
    (tri_def : ∀ t ∈ triangulation, t.1 ≠ t.2 ∧ t.2 ≠ t.3 ∧ t.1 ≠ t.3)
    (non_intersecting_diagonals :
      ∀ (t₁ t₂ ∈ triangulation) (d₁ d₂ ∈ List.map (λ t => [t.1, t.2, t.3]) triangulation),
        d₁ ≠ d₂ →
        ∄ (d : Fin n × Fin n), d ∈ d₁ ∧ d ∈ d₂)
    (good_diagonals_count : good_diagonals_count = n / 2 - 1)
    (triangle_coloring : Fin n → Fin n → Fin n → Fin 2)
    (triangle_coloring_def :
      ∀ (x y z : Fin n),
        triangle_coloring x y z ≠ triangle_coloring y z x ∧
        triangle_coloring y z x ≠ triangle_coloring z x y ∧
        triangle_coloring z x y ≠ triangle_coloring x y z) :
    ∃ (c : Fin 2),
    count (triangle_coloring • triangulation) c = count (triangle_coloring • triangulation) (Fin.succ c) :=
sorry

end equal_number_of_colored_triangles_l328_328809


namespace t_plus_reciprocal_l328_328914

theorem t_plus_reciprocal (t : ℝ) (h : t^2 - 3 * t + 1 = 0) (ht : t ≠ 0) : t + 1/t = 3 :=
by sorry

end t_plus_reciprocal_l328_328914


namespace roots_reciprocal_l328_328213

theorem roots_reciprocal (x1 x2 : ℝ) (h1 : x1^2 - 4 * x1 - 2 = 0) (h2 : x2^2 - 4 * x2 - 2 = 0) (h3 : x1 ≠ x2) :
  (1 / x1) + (1 / x2) = -2 := 
sorry

end roots_reciprocal_l328_328213


namespace part_a_l328_328647

theorem part_a
  (ABC : Type*)
  [is_triangle ABC]
  (A B C : ABC)
  (H : ABC)
  (AH BM : ℝ)
  (acute_angled : is_acute ABC)
  (H_on_BC : lies_on H B C)
  (AH_longest : is_longest_altitude A H B C)
  (M : ABC)
  (midpoint_M : is_midpoint M A C)
  (AH_le_BM : AH ≤ BM) :
  measure_angle ABC ≤ 60 := sorry

end part_a_l328_328647


namespace find_smallest_x_l328_328983

theorem find_smallest_x (y : ℤ) (x : ℤ) (hy : 0.8 = y / (200 + x)) (hx_pos : x > 0) (hy_pos : y > 0) : x = 0 :=
sorry

end find_smallest_x_l328_328983


namespace total_questions_reviewed_l328_328832

-- Defining the conditions
def questions_per_student : Nat := 10
def students_per_class : Nat := 35
def num_classes : Nat := 5

-- Define the total number of questions that Professor Oscar must review.
def total_questions : Nat := questions_per_student * students_per_class * num_classes

-- Theorem statement to be proved
theorem total_questions_reviewed : total_questions = 1750 := by
  sorry

end total_questions_reviewed_l328_328832


namespace nat_km_on_monday_l328_328379

theorem nat_km_on_monday :
  ∃ M : ℕ, 
    let T := 50 in
    let W := 0.5 * T in
    let Th := M + W in
    M + T + W + Th = 180 ∧ M = 40 := by
  sorry

end nat_km_on_monday_l328_328379


namespace perimeter_of_rhombus_with_given_diagonals_l328_328732

noncomputable def rhombus_perimeter (d1 d2 : ℝ) : ℝ :=
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side

theorem perimeter_of_rhombus_with_given_diagonals :
  rhombus_perimeter 10 24 = 52 :=
by
  -- Assuming the necessary calculations established in prior steps are correct
  sorry

end perimeter_of_rhombus_with_given_diagonals_l328_328732


namespace intersection_M_N_l328_328262

-- Definitions for M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | 3 ^ x < 1 }

-- The theorem to prove
theorem intersection_M_N :
  M ∩ N = { x | x < 0 } :=
by simp [M, N]; sorry

end intersection_M_N_l328_328262


namespace part1_part2_i_part2_ii_l328_328250

noncomputable def f (x : ℝ) (a : ℝ) := x + a / x

theorem part1 (a : ℝ) (h : 0 < a) :
  (∀ x, 0 < x ∧ x < Real.sqrt a → f x a < (λ (x : ℝ), x) x) ∧ 
  (∀ x, Real.sqrt a < x → (λ (x : ℝ), x) x < f x a) :=
  sorry

theorem part2_i :
  let f (x : ℝ) := 2^x + 1 / 2^x - 2 in
  ∃ x : ℝ, f x = 0 :=
  sorry

theorem part2_ii :
  ∀ m, (∀ x : ℝ, f (4^x) 1 ≥ m * f (2^x) 1 - 6) ↔ m ≤ 4 :=
  sorry

end part1_part2_i_part2_ii_l328_328250


namespace f_monotonic_increasing_g_a_lt_g_a1_l328_328931

def f (x : ℝ) : ℝ := x - 4/x

def g (x : ℝ) : ℝ :=
if x < 1 then -2*x - 1 else f x

theorem f_monotonic_increasing (x1 x2 : ℝ) (hx1 : 1 ≤ x1) (hx2 : x1 < x2) : f x1 < f x2 := 
sorry

theorem g_a_lt_g_a1 (a : ℝ) : (g a < g (a + 1)) ↔ (1/3 < a) := 
sorry

end f_monotonic_increasing_g_a_lt_g_a1_l328_328931


namespace geometric_sequence_sum_l328_328356

def geometric_sequence_props (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  (∀ n, a n = a 1 * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (1 - 2^n) / (1 - 2)) ∧ 
  (a 5 - a 3 = 12) ∧ 
  (a 6 - a 4 = 24)

theorem geometric_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : geometric_sequence_props a S) :
  ∀ n, S n / a n = 2 - 2^(1 - n) :=
by
  sorry

end geometric_sequence_sum_l328_328356


namespace number_of_rolls_l328_328492

theorem number_of_rolls (p : ℚ) (h : p = 1 / 9) : (2 : ℕ) = 2 :=
by 
  have h1 : 2 = 2 := rfl
  exact h1

end number_of_rolls_l328_328492


namespace sequence_general_formula_l328_328747

theorem sequence_general_formula :
  ∀ n : ℕ, n > 0 → let a_n := (if n % 2 = 1 then 1 else -1 : ℚ) * (2 * n + 1) / (n * (n + 1)) in
  match n with
  | 0 => false
  | _ + 1 => a_n = (if (n + 1) % 2 = 1 then 1 else -1 : ℚ) * (2 * (n + 1) + 1) / ((n + 1) * (n + 1 + 1))
sorry

end sequence_general_formula_l328_328747


namespace distance_between_ships_l328_328455

noncomputable def tan (x : ℝ) : ℝ := Real.tan x

theorem distance_between_ships
  (h : ℝ) (α : ℝ) (β : ℝ)
  (h_eq : h = 120)
  (α_eq : α = Real.pi * 25 / 180)
  (β_eq : β = Real.pi * 60 / 180) :
  let d₁ := h / tan α
  let d₂ := h / tan β
  let distance := d₁ + d₂
  distance ≈ 326.7 :=
by
  sorry

end distance_between_ships_l328_328455


namespace find_value_l328_328520

-- Defining the function f and conditions
def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
else if x < 0 then -f(-x)
else f(x - 2)

-- Conditions
axiom odd_f : ∀ x : ℝ, f(-x) = -f(x)
axiom periodic_f : ∀ x : ℝ, f(x + 2) = f(x)
axiom defined_interval : ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f(x) = 2 * x * (1 - x)

-- Proposition to prove
theorem find_value : f (-5/2) = -1/2 :=
by
  sorry

end find_value_l328_328520


namespace mean_median_mode_solution_l328_328418

theorem mean_median_mode_solution :
  ∃ (y x : ℝ), 
    (7 * y = 580 + x) ∧
    (x ≥ y) ∧
    (∃ (data : list ℝ), 
        data = [70, 110, x, 40, y, 50, 210, y, 100] ∧ 
        (mode data = y) ∧ 
        (mean data = y) ∧ 
        (median data = y)) → 
    y = 290 / 3 :=
begin
  sorry
end

end mean_median_mode_solution_l328_328418


namespace geometric_sequence_sum_l328_328355

def geometric_sequence_props (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  (∀ n, a n = a 1 * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (1 - 2^n) / (1 - 2)) ∧ 
  (a 5 - a 3 = 12) ∧ 
  (a 6 - a 4 = 24)

theorem geometric_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : geometric_sequence_props a S) :
  ∀ n, S n / a n = 2 - 2^(1 - n) :=
by
  sorry

end geometric_sequence_sum_l328_328355


namespace equation_has_three_real_roots_l328_328614

noncomputable def f (x : ℝ) : ℝ := 2^x - x^2 - 1

theorem equation_has_three_real_roots : ∃! (x : ℝ), f x = 0 :=
by sorry

end equation_has_three_real_roots_l328_328614


namespace sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth_l328_328227

noncomputable def sin_pi_div_two_plus_2alpha (α : ℝ) : ℝ :=
  Real.sin ((Real.pi / 2) + 2 * α)

def cos_alpha (α : ℝ) := Real.cos α = - (Real.sqrt 2) / 3

theorem sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth (α : ℝ) (h : cos_alpha α) :
  sin_pi_div_two_plus_2alpha α = -5 / 9 :=
sorry

end sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth_l328_328227


namespace geometric_seq_ratio_l328_328362

theorem geometric_seq_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 5 - a 3 = 12) 
  (h2 : a 6 - a 4 = 24) 
  (ha : ∃ a₁ : ℝ, (∀ n : ℕ, a n = a₁ * q ^ (n - 1)))
  (hS : ∀ n : ℕ, S n = a₁ * (1 - q ^ n) / (1 - q)) :
  ∀ n : ℕ, S n / a n = 2 - 2 ^ (1 - n) :=
sorry

end geometric_seq_ratio_l328_328362


namespace determine_all_functions_l328_328182

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

theorem determine_all_functions (f : ℝ → ℝ) (h : functional_equation f) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end determine_all_functions_l328_328182


namespace sum_induction_proof_l328_328761

theorem sum_induction_proof (n : ℕ) (hn : n > 0) :
  (finset.sum (finset.range n) (λ k, 1 / (2 * k * (2 * k + 2))) = (n / (4 * (n + 1)))) := sorry

end sum_induction_proof_l328_328761


namespace line_AZ_passes_through_diametric_opposite_of_X_l328_328013

open Classical

variables {A B C I I_A X Z : Type} [Incircle A B C I] [ExcircleOppositeToAngle A B C I_A]

theorem line_AZ_passes_through_diametric_opposite_of_X
  (tangency_X : TangencyPoint I B C X)
  (tangency_Z : TangencyPoint I_A B C Z)
  (homothety : Homothety A (circle I) (circle I_A))
  : PassesThrough (line_through A Z) (diametrically_opposite_point X (circle I)) :=
sorry

end line_AZ_passes_through_diametric_opposite_of_X_l328_328013


namespace base_b_number_not_divisible_by_5_l328_328901

-- We state the mathematical problem in Lean 4 as a theorem.
theorem base_b_number_not_divisible_by_5 (b : ℕ) (hb : b = 12) : 
  ¬ ((3 * b^2 * (b - 1) + 1) % 5 = 0) := 
by sorry

end base_b_number_not_divisible_by_5_l328_328901


namespace cheat_buying_percentage_l328_328818

-- Definitions for the problem
def profit_margin := 0.5
def cheat_selling := 0.2

-- Prove that the cheating percentage while buying is 20%
theorem cheat_buying_percentage : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ x = 0.2 := by
  sorry

end cheat_buying_percentage_l328_328818


namespace probability_two_points_one_unit_apart_l328_328723

theorem probability_two_points_one_unit_apart :
  let total_points := 10
  let total_ways := (total_points * (total_points - 1)) / 2
  let favorable_horizontal_pairs := 8
  let favorable_vertical_pairs := 5
  let favorable_pairs := favorable_horizontal_pairs + favorable_vertical_pairs
  let probability := (favorable_pairs : ℚ) / total_ways
  probability = 13 / 45 :=
by
  sorry

end probability_two_points_one_unit_apart_l328_328723


namespace angle_OMB_right_l328_328917

variables {A B C K N M O : Type*} [circle A B C K N O] 

noncomputable def points_on_circle (A B C K N : O) : Prop := sorry

noncomputable def intersects_at (circABC circKBN : O) (B M : O) : Prop := sorry

theorem angle_OMB_right :
  ∀ (A B C K N M O : Type*) [h₁ : circle A B C K N O] 
    [h₂: points_on_circle A B C K N] [h₃: intersects_at circABC circKBN B M],
  angle O M B = 90 :=
by
  sorry

end angle_OMB_right_l328_328917


namespace tangent_identity_l328_328477

variables (P A B C Q : Type) [Group P] [AffineSpace P] {α β γ : ℝ} {a b c : ℝ}

-- Define points on the line and Q not on the line
variables (h₁ : ∃ (P A B C : P), collinear P A B C) (h₂ : Q ∉ line P A)

-- Define angles α, β, γ and segments a, b, c
variables (hα : α = angle P Q A) (hβ : β = angle P Q B) (hγ : γ = angle P Q C)
variables (ha : a = dist P A) (hb : b = dist P B) (hc : c = dist P C)

theorem tangent_identity
: ∀ {a b c α β γ : ℝ}, 
  ( 1 / a * tan α * (tan β - tan γ) 
  + 1 / b * tan β * (tan γ - tan α) 
  + 1 / c * tan γ * (tan α - tan β) = 0) 
sorry

end tangent_identity_l328_328477


namespace finite_regular_polygon_l328_328194

open Set

-- Define the necessary conditions and theorems in Lean.
def is_axis_of_symmetry (S : Set (ℝ × ℝ)) (l : ℝ × ℝ → ℝ × ℝ) : Prop :=
  ∀ P ∈ S, l P ∈ S

def perpendicular_bisector (A B : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ := sorry

theorem finite_regular_polygon {S : Set (ℝ × ℝ)} (h1 : Finite S)
  (h2 : 3 ≤ S.card)
  (h3 : ∀ A B ∈ S, A ≠ B → is_axis_of_symmetry S (perpendicular_bisector A B)) :
  ∃ (k : ℕ) (h : k > 2), ∃ (T : Set (ℝ × ℝ)), is_regular_polygon T k ∧ S = T :=
sorry

end finite_regular_polygon_l328_328194


namespace triangle_probability_l328_328403

def sticks : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def forms_triangle (a b c : ℕ) : Prop :=
  (a < b + c) ∧ (b < c + a) ∧ (c < a + b)

def valid_combinations : List (ℕ × ℕ × ℕ) :=
  List.filter (λ (abc : ℕ × ℕ × ℕ), forms_triangle abc.1 abc.2 abc.3)
    [ (2, 3, 5), (2, 3, 7), (2, 3, 11), (2, 3, 13), (2, 3, 17), (2, 3, 19), (2, 3, 23), (2, 3, 29),
      (2, 5, 7), (2, 5, 11), (2, 5, 13), (2, 5, 17), (2, 5, 19), (2, 5, 23), (2, 5, 29),
      (2, 7, 11), (2, 7, 13), (2, 7, 17), (2, 7, 19), (2, 7, 23), (2, 7, 29),
      (2, 11, 13), (2, 11, 17), (2, 11, 19), (2, 11, 23), (2, 11, 29),
      (2, 13, 17), (2, 13, 19), (2, 13, 23), (2, 13, 29),
      (2, 17, 19), (2, 17, 23), (2, 17, 29),
      (2, 19, 23), (2, 19, 29),
      (2, 23, 29),
      (3, 5, 7), (3, 5, 11), (3, 5, 13), (3, 5, 17), (3, 5, 19), (3, 5, 23), (3, 5, 29),
      (3, 7, 11), (3, 7, 13), (3, 7, 17), (3, 7, 19), (3, 7, 23), (3, 7, 29),
      (3, 11, 13), (3, 11, 17), (3, 11, 19), (3, 11, 23), (3, 11, 29),
      (3, 13, 17), (3, 13, 19), (3, 13, 23), (3, 13, 29),
      (3, 17, 19), (3, 17, 23), (3, 17, 29),
      (3, 19, 23), (3, 19, 29),
      (3, 23, 29),
      (5, 7, 11), (5, 7, 13), (5, 7, 17), (5, 7, 19), (5, 7, 23), (5, 7, 29),
      (5, 11, 13), (5, 11, 17), (5, 11, 19), (5, 11, 23), (5, 11, 29),
      (5, 13, 17), (5, 13, 19), (5, 13, 23), (5, 13, 29),
      (5, 17, 19), (5, 17, 23), (5, 17, 29),
      (5, 19, 23), (5, 19, 29),
      (5, 23, 29),
      (7, 11, 13), (7, 11, 17), (7, 11, 19), (7, 11, 23), (7, 11, 29),
      (7, 13, 17), (7, 13, 19), (7, 13, 23), (7, 13, 29),
      (7, 17, 19), (7, 17, 23), (7, 17, 29),
      (7, 19, 23), (7, 19, 29),
      (7, 23, 29),
      (11, 13, 17), (11, 13, 19), (11, 13, 23), (11, 13, 29),
      (11, 17, 19), (11, 17, 23), (11, 17, 29),
      (11, 19, 23), (11, 19, 29),
      (11, 23, 29),
      (13, 17, 19), (13, 17, 23), (13, 17, 29),
      (13, 19, 23), (13, 19, 29),
      (13, 23, 29),
      (17, 19, 23), (17, 19, 29),
      (17, 23, 29),
      (19, 23, 29) ]

def probability_of_triangle : ℚ :=
  (valid_combinations.length : ℚ) / (Nat.choose sticks.length 3 : ℚ)

theorem triangle_probability : probability_of_triangle = 2 / 5 :=
by
  sorry

end triangle_probability_l328_328403


namespace symmetric_point_of_M_neg2_3_l328_328731

-- Conditions
def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, -M.2)

-- Main statement
theorem symmetric_point_of_M_neg2_3 :
  symmetric_point (-2, 3) = (2, -3) := 
by
  -- Proof goes here
  sorry

end symmetric_point_of_M_neg2_3_l328_328731


namespace original_lettuce_cost_l328_328617

theorem original_lettuce_cost
  (original_cost: ℝ) (tomatoes_original: ℝ) (tomatoes_new: ℝ) (celery_original: ℝ) (celery_new: ℝ) (lettuce_new: ℝ)
  (delivery_tip: ℝ) (new_bill: ℝ)
  (H1: original_cost = 25)
  (H2: tomatoes_original = 0.99) (H3: tomatoes_new = 2.20)
  (H4: celery_original = 1.96) (H5: celery_new = 2.00)
  (H6: lettuce_new = 1.75)
  (H7: delivery_tip = 8.00)
  (H8: new_bill = 35) :
  ∃ (lettuce_original: ℝ), lettuce_original = 1.00 :=
by
  let tomatoes_diff := tomatoes_new - tomatoes_original
  let celery_diff := celery_new - celery_original
  let new_cost_without_lettuce := original_cost + tomatoes_diff + celery_diff
  let new_cost_excl_delivery := new_bill - delivery_tip
  have lettuce_diff := new_cost_excl_delivery - new_cost_without_lettuce
  let lettuce_original := lettuce_new - lettuce_diff
  exists lettuce_original
  sorry

end original_lettuce_cost_l328_328617


namespace range_of_a_l328_328367

noncomputable def f (x : ℝ) : ℝ := (2 * x^2) / (x + 1)

noncomputable def g (a x : ℝ) : ℝ := a * x + 5 - 2 * a

theorem range_of_a (a : ℝ) (h : a > 0) : 
  (∀ x1 ∈ Icc (0 : ℝ) 1, ∃ x0 ∈ Icc (0 : ℝ) 1, g a x0 = f x1) ↔ (5 / 2 ≤ a ∧ a ≤ 4) :=
sorry

end range_of_a_l328_328367


namespace exists_n_element_subset_with_distinct_remainders_l328_328787

theorem exists_n_element_subset_with_distinct_remainders (N : ℕ) :
  ∀ (A : Finset ℕ), (A.card = N) →
    ∃ (B : Finset ℕ), (B ⊆ Finset.range (N^2)) ∧ (B.card = N) ∧
      (Finset.card ((A.product B).image (λ (ab : ℕ × ℕ), (ab.fst + ab.snd) % (N^2))) ≥ N^2 / 2) :=
by
  sorry

end exists_n_element_subset_with_distinct_remainders_l328_328787


namespace part1_part2_l328_328254

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + 4 * a * x + 2 * a + 6

-- State the conditions for Part 1
theorem part1 (a : ℝ) :
  (∀ x, f x a ≥ 0) ∧ (range (λ x, f x a) = set.Ici 0) → (a = -1 ∨ a = 3 / 2) :=
sorry

-- Define the function g
def g (a : ℝ) : ℝ := 2 - a * |a + 3|

-- State the conditions for Part 2
theorem part2 (h1 : ∀ x a, f x a ≥ 0) (a : ℝ) :
  -1 ≤ a ∧ a ≤ 3 / 2 → range g = set.Icc (-19 / 4) 4 :=
sorry

end part1_part2_l328_328254


namespace total_black_balls_l328_328997

-- Conditions
def number_of_white_balls (B : ℕ) : ℕ := 6 * B

def total_balls (B : ℕ) : ℕ := B + number_of_white_balls B

-- Theorem to prove
theorem total_black_balls (h : total_balls B = 56) : B = 8 :=
by
  sorry

end total_black_balls_l328_328997


namespace B_subset_A_iff_a_range_l328_328675

variable (a : ℝ)
def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x - 4 ≤ 0}

theorem B_subset_A_iff_a_range :
  B a ⊆ A ↔ 0 ≤ a ∧ a < 3 :=
by
  sorry

end B_subset_A_iff_a_range_l328_328675


namespace even_sigma_minus_d_l328_328716

open Nat

def sumOfDivisors (n : ℕ) : ℕ := ∑ i in (finset.range (n + 1)).filter (λ d, n % d = 0), d
def numOfDivisors (n : ℕ) : ℕ := (finset.range (n + 1)).filter (λ d, n % d = 0).card

def is_odd (n : ℕ) : Prop := n % 2 = 1

def largestOddDivisor (n : ℕ) : ℕ := nat.find_greatest (λ d, d ∣ n ∧ is_odd d) n

theorem even_sigma_minus_d {n m : ℕ} (hn : 0 < n) (hm : 0 < m) (h : m = largestOddDivisor n) :
  (sumOfDivisors n - numOfDivisors m) % 2 = 0 :=
sorry

end even_sigma_minus_d_l328_328716


namespace compute_product_fraction_l328_328171

theorem compute_product_fraction :
  ( ((3 : ℚ)^4 - 1) / ((3 : ℚ)^4 + 1) *
    ((4 : ℚ)^4 - 1) / ((4 : ℚ)^4 + 1) * 
    ((5 : ℚ)^4 - 1) / ((5 : ℚ)^4 + 1) *
    ((6 : ℚ)^4 - 1) / ((6 : ℚ)^4 + 1) *
    ((7 : ℚ)^4 - 1) / ((7 : ℚ)^4 + 1)
  ) = (25 / 210) := 
  sorry

end compute_product_fraction_l328_328171


namespace sum_of_distances_A_to_B_C_l328_328322

variable (A B C D : Type) [AddGroup A] [LinearOrderedAddCommGroup A] [MetricSpace A] [AddCommMonoid A]
variable (f g : A → A)
variable (AB AC BC AD DC : ℝ)

-- Assuming A, B, C, and D are points in ℝ
-- Defining distances to be non-negative real numbers
-- All points B, C, and D are uniquely determined with those distances from A
-- Each of those points must also satisfy triangle inequalities individually

-- The conditions
axiom triangle_inequality : 
  ∀ (x y z : A), dist x y + dist y z ≥ dist x z

axiom point_in_triangle (A B C : A) (D : A) :
  dist A D + dist D C ≤ dist A B + dist B C 

-- The statement to be proven
theorem sum_of_distances_A_to_B_C :
  dist A D + dist D C ≤ dist A B + dist B C :=
begin
  -- Place your proof here
  sorry
end

end sum_of_distances_A_to_B_C_l328_328322


namespace each_person_bid_count_l328_328159

-- Define the conditions and initial values
noncomputable def auctioneer_price_increase : ℕ := 5
noncomputable def initial_price : ℕ := 15
noncomputable def final_price : ℕ := 65
noncomputable def number_of_bidders : ℕ := 2

-- Define the proof statement
theorem each_person_bid_count : 
  (final_price - initial_price) / auctioneer_price_increase / number_of_bidders = 5 :=
by sorry

end each_person_bid_count_l328_328159


namespace construct_line_segment_equal_to_sphere_radius_l328_328580

theorem construct_line_segment_equal_to_sphere_radius 
  (sphere : Type) 
  (radius : ℝ)
  (construct_circle : Π (A : point3d), circle3d sphere A radius)
  (compass_straightedge_construct : Π (B C D: point3d), is_on_sphere B sphere ∧ is_on_sphere C sphere ∧ is_on_sphere D sphere → triangle_plane B C D → circumcircle_plane (triangle_plane B C D))
  (output : Type)
  (desired_segment : output)
  (segment_length : desired_segment = radius) :
  ∃ segment, segment = radius :=
by
  sorry

end construct_line_segment_equal_to_sphere_radius_l328_328580


namespace periodic_odd_function_property_l328_328587

-- Define the function f with the given properties
noncomputable def f : ℝ → ℝ :=
  λ x, if x ∈ Ioc 0 1 then sin (Real.pi * x) else
  if x < 0 then -f (-x) else -- odd function property
  if x ≥ 2 then f (x % 2) else 
  0 -- simplification, assuming the function is zero otherwise for simplicity.

theorem periodic_odd_function_property :
  (f (-5/2) + f 1 + f 2 = -1) :=
by sorry

end periodic_odd_function_property_l328_328587


namespace painted_rooms_l328_328499

/-- Given that there are a total of 11 rooms to paint, each room takes 7 hours to paint,
and the painter has 63 hours of work left to paint the remaining rooms,
prove that the painter has already painted 2 rooms. -/
theorem painted_rooms (total_rooms : ℕ) (hours_per_room : ℕ) (hours_left : ℕ) 
  (h_total_rooms : total_rooms = 11) (h_hours_per_room : hours_per_room = 7) 
  (h_hours_left : hours_left = 63) : 
  (total_rooms - hours_left / hours_per_room) = 2 := 
by
  sorry

end painted_rooms_l328_328499


namespace number_of_paths_l328_328641

def total_steps : ℕ := 13

def right_steps : ℕ := 7

def up_steps : ℕ := 6

def consecutive_right_steps_required : Prop := true

def paths_with_conditions := (total_steps = 13) ∧ (right_steps = 7) ∧ (up_steps = 6) ∧ consecutive_right_steps_required

theorem number_of_paths (p : paths_with_conditions): (nat.choose 12 6) = 924 :=
sorry

end number_of_paths_l328_328641


namespace initial_population_is_9250_l328_328312

noncomputable def initial_population : ℝ :=
  let final_population := 6514
  let factor := (1.08 * 0.85 * (1.02)^5 * 0.95 * 0.9)
  final_population / factor

theorem initial_population_is_9250 : initial_population = 9250 := by
  sorry

end initial_population_is_9250_l328_328312


namespace dan_spent_more_on_chocolates_l328_328177

def price_candy_bar : ℝ := 4
def number_of_candy_bars : ℕ := 5
def candy_discount : ℝ := 0.20
def discount_threshold : ℕ := 3
def price_chocolate : ℝ := 6
def number_of_chocolates : ℕ := 4
def chocolate_tax_rate : ℝ := 0.05

def candy_cost_total : ℝ :=
  let cost_without_discount := number_of_candy_bars * price_candy_bar
  if number_of_candy_bars >= discount_threshold
  then cost_without_discount * (1 - candy_discount)
  else cost_without_discount

def chocolate_cost_total : ℝ :=
  let cost_without_tax := number_of_chocolates * price_chocolate
  cost_without_tax * (1 + chocolate_tax_rate)

def difference_in_spending : ℝ :=
  chocolate_cost_total - candy_cost_total

theorem dan_spent_more_on_chocolates :
  difference_in_spending = 9.20 :=
by
  sorry

end dan_spent_more_on_chocolates_l328_328177


namespace correct_total_amount_l328_328408

variable (y z w : ℕ)

-- Define the errors for each type of coin miscount
def quarter_as_dime_error := 15 * y
def nickel_as_penny_error := 4 * z
def dime_as_quarter_error := -15 * w

-- Define the total correction amount
def total_correction := quarter_as_dime_error y + nickel_as_penny_error z + dime_as_quarter_error w

theorem correct_total_amount :
  total_correction y z w = 15 * y + 4 * z - 15 * w :=
sorry

end correct_total_amount_l328_328408


namespace linear_function_difference_l328_328677

theorem linear_function_difference (g : ℝ → ℝ) (linearity : ∀ x y, g(x + y) = g(x) + g(y)) (h : g(5) - g(0) = 10) : g(15) - g(5) = 20 :=
sorry

end linear_function_difference_l328_328677


namespace solve_rational_equation_l328_328396

theorem solve_rational_equation : 
  ∀ x : ℝ, x ≠ 1 -> (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) → 
  (x = 6 ∨ x = -2) :=
by
  intro x
  intro h
  intro h_eq
  sorry

end solve_rational_equation_l328_328396


namespace find_t_l328_328877

variable (t : ℚ)

def point_on_line (p1 p2 p3 : ℚ × ℚ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_t (t : ℚ) : point_on_line (3, 0) (0, 7) (t, 8) → t = -3 / 7 := by
  sorry

end find_t_l328_328877


namespace circle_area_and_diameter_l328_328409

theorem circle_area_and_diameter (C : ℝ) (hC : C = 18 * real.pi) :
  (∃ A d : ℝ, A = 81 * real.pi ∧ d = 18) :=
sorry

end circle_area_and_diameter_l328_328409


namespace bob_needs_to_improve_l328_328857

noncomputable def bob_average := 5.5 / 6
noncomputable def bob_total := 10 * 60 + 40
noncomputable def sister_average_one := 11
noncomputable def sister_average_two := 10
noncomputable def sister_total := 5 * 60 + 20

def required_improvement (bob_total : ℕ) (sister_total : ℕ) : ℚ :=
  ((bob_total - sister_total) / bob_total.to_rat) * 100

theorem bob_needs_to_improve : 
  required_improvement bob_total sister_total ≈ 48.98 :=
sorry

end bob_needs_to_improve_l328_328857


namespace find_m_eq_5_over_6_l328_328201

theorem find_m_eq_5_over_6 (x : ℝ) (h : sin (x / 6) ≠ 0 ∧ sin x ≠ 0) :
  (cot (x / 6) - cot x = (sin (5 * x / 6) / (sin (x / 6) * sin x))) :=
by
  sorry

end find_m_eq_5_over_6_l328_328201


namespace greatest_a_l328_328884

theorem greatest_a (a b : ℝ) : a^2 - 12 * a + 35 ≤ 0 → (b^2 - 12 * b + 35 ≤ 0 → b ≤ 7) :=
by
  intro h₁ h₂
  have h_upper : a ≤ 7 := by nlinarith
  have h_lower : 5 ≤ a := by nlinarith
  simp only [true_and]
  assumption

end greatest_a_l328_328884


namespace ajay_saves_each_month_l328_328513

theorem ajay_saves_each_month :
  (monthly_income percentage_household percentage_clothes percentage_medicines percentage_saved amount_saved : ℝ)
  (H1 : monthly_income = 40000)
  (H2 : percentage_household = 45 / 100)
  (H3 : percentage_clothes = 25 / 100)
  (H4 : percentage_medicines = 7.5 / 100)
  (H5 : percentage_saved = 1 - (percentage_household + percentage_clothes + percentage_medicines))
  (H6 : amount_saved = monthly_income * percentage_saved) :
  amount_saved = 9000 := by
  sorry

end ajay_saves_each_month_l328_328513


namespace num_valid_colorings_eq_six_l328_328977

def color (n : ℤ) : Bool := sorry

def valid_coloring (color : ℤ → Bool) : Prop :=
  (∀ n, color n = color (n + 7)) ∧
  ¬∃ k, color k = color (k + 1) ∧ color k = color (2 * k)

theorem num_valid_colorings_eq_six : 
  ∃ (colors : ℤ → Bool), valid_coloring colors ∧ (colors.enum.count (valid_coloring colors) = 6) :=
sorry

end num_valid_colorings_eq_six_l328_328977


namespace part_a_towers_part_b_towers_part_c_impossible_8_towers_l328_328794

-- Part (a)
theorem part_a_towers (pieces : list ℕ) (h_piece1 : pieces = [2, 3, 4])
(h_piece2 : pieces.length = 3) :
∃ towers : list (list ℕ), (∀ t ∈ towers, t.sum = 10) ∧
towers = [[4, 4, 2], [4, 2, 4], [2, 4, 4], [4, 3, 3], [3, 4, 3], [3, 3, 4]] :=
by sorry

-- Part (b)
theorem part_b_towers (total_pieces : ℕ) (type_pieces : ℕ)
(h_total : total_pieces = 27) (h_type : type_pieces = 9)
(piece_types : list ℕ) (h_piece_type : piece_types = [2, 3, 4]) :
∃ towers : list (list ℕ), (∀ t ∈ towers, t.sum = 10) ∧
towers.length = 7 ∧
(towers.count_in [4, 4, 2] = 4 ∧ towers.count_in [4, 3, 3] = 1 ∧ towers.count_in [3, 3, 2, 2] = 2) :=
by sorry

-- Part (c)
theorem part_c_impossible_8_towers (total_pieces : ℕ) (type_pieces : ℕ)
(h_total : total_pieces = 27) (h_type : type_pieces = 9)
(piece_types : list ℕ) (h_piece_type : piece_types = [2, 3, 4]) :
¬ ∃ towers : list (list ℕ), (∀ t ∈ towers, t.sum = 10) ∧
towers.length = 8 :=
by sorry

end part_a_towers_part_b_towers_part_c_impossible_8_towers_l328_328794


namespace ratio_AB_BC_l328_328386

theorem ratio_AB_BC (A B C : Point) (r : ℝ) (h_circle : on_circle A r) 
    (h_circle_B : on_circle B r) (h_circle_C : on_circle C r)
    (h_AB_AC : dist A B = dist A C) (h_AB_gt_r : dist A B > r)
    (h_arc_BC : arc_length B C < 2 * π * r ∧ arc_length B C = 3 * r / 2) :
    dist A B / dist B C = 4 / 3 * sin (3 / 4) :=
sorry

end ratio_AB_BC_l328_328386


namespace hotel_loss_l328_328121
  
  -- Conditions
  def operations_expenses : ℝ := 100
  def total_payments : ℝ := (3 / 4) * operations_expenses
  
  -- Theorem to prove
  theorem hotel_loss : operations_expenses - total_payments = 25 :=
  by
    sorry
  
end hotel_loss_l328_328121


namespace t_plus_reciprocal_l328_328916

theorem t_plus_reciprocal (t : ℝ) (h : t^2 - 3 * t + 1 = 0) (ht : t ≠ 0) : t + 1/t = 3 :=
by sorry

end t_plus_reciprocal_l328_328916


namespace find_b_in_triangle_l328_328637

theorem find_b_in_triangle
  (a b c A B C : ℝ)
  (cos_A : ℝ) (cos_C : ℝ)
  (ha : a = 1)
  (hcos_A : cos_A = 4 / 5)
  (hcos_C : cos_C = 5 / 13) :
  b = 21 / 13 :=
by
  sorry

end find_b_in_triangle_l328_328637


namespace brittany_money_times_brooke_l328_328515

theorem brittany_money_times_brooke 
  (kent_money : ℕ) (brooke_money : ℕ) (brittany_money : ℕ) (alison_money : ℕ)
  (h1 : kent_money = 1000)
  (h2 : brooke_money = 2 * kent_money)
  (h3 : alison_money = 4000)
  (h4 : alison_money = brittany_money / 2) :
  brittany_money = 4 * brooke_money :=
by
  sorry

end brittany_money_times_brooke_l328_328515


namespace irreducible_polynomial_l328_328095

theorem irreducible_polynomial {n : ℕ} (a : Fin n → ℤ) (h_distinct : Function.Injective a) :
  ¬ ∃ (f g : Polynomial ℤ), f.degree > 0 ∧ g.degree > 0 ∧
    (Polynomial.prod (Finset.univ.image (λ i : Fin n, Polynomial.X - Polynomial.C (a i))) - 1) = f * g := by
  sorry

end irreducible_polynomial_l328_328095


namespace first_term_exceeding_5000_is_8192_l328_328046

noncomputable def sequence : ℕ → ℕ
| 0 := 1
| n+1 := ∑ i in Finset.range (n+1), sequence i

theorem first_term_exceeding_5000_is_8192 :
  ∃ n, sequence n > 5000 ∧ sequence n = 8192 :=
by {
  -- Proof skipped
  sorry
}

end first_term_exceeding_5000_is_8192_l328_328046


namespace compute_x_l328_328664

/-- 
Let ABC be a triangle. 
Points D, E, and F are on BC, CA, and AB, respectively. 
Given that AE/AC = CD/CB = BF/BA = x for some x with 1/2 < x < 1. 
Segments AD, BE, and CF divide the triangle into 7 non-overlapping regions: 
4 triangles and 3 quadrilaterals. 
The total area of the 4 triangles equals the total area of the 3 quadrilaterals. 
Compute the value of x.
-/
theorem compute_x (x : ℝ) (h1 : 1 / 2 < x) (h2 : x < 1)
  (h3 : (∃ (triangleArea quadrilateralArea : ℝ), 
          let A := triangleArea + 3 * x
          let B := quadrilateralArea
          A = B))
  : x = (11 - Real.sqrt 37) / 6 := 
sorry

end compute_x_l328_328664


namespace vision_test_estimate_l328_328405

variables (total_students : ℕ) (sample_size : ℕ) (poor_vision_sample : ℕ)
variables (percentage_poor_vision : ℚ) (total_poor_vision_estimate : ℕ)

def calculate_percentage_poor_vision (sample_size poor_vision_sample : ℕ) : ℚ :=
  poor_vision_sample / sample_size

def estimate_total_poor_vision (total_students : ℕ) (percentage_poor_vision : ℚ) : ℕ :=
  (total_students * percentage_poor_vision).to_nat

theorem vision_test_estimate
  (h1 : total_students = 30000)
  (h2 : sample_size = 500)
  (h3 : poor_vision_sample = 100)
  (h4 : percentage_poor_vision = 0.2) :
  estimate_total_poor_vision total_students percentage_poor_vision = 6000 :=
by {
  sorry
}

end vision_test_estimate_l328_328405


namespace max_gold_coins_l328_328088

theorem max_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 110) : n ≤ 107 :=
by
  sorry

end max_gold_coins_l328_328088


namespace simplify_and_evaluate_l328_328018

theorem simplify_and_evaluate : 
  ∀ (x y : ℚ), x = 1 / 2 → y = 2 / 3 →
  ((x - 2 * y)^2 + (x - 2 * y) * (x + 2 * y) - 3 * x * (2 * x - y)) / (2 * x) = -4 / 3 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_l328_328018


namespace count_values_satisfying_condition_l328_328736

-- Function f is such that f(x) = 4 at x = -2, 2, and 4
def f : ℝ → ℝ
| -2 := 4
| 2  := 4
| 4  := 4
| x  := sorry  -- Placeholder for other values, as the main concern is f(x) = 4 at specific points

-- Prove there are exactly 2 values of x such that f(f(x)) = 4
theorem count_values_satisfying_condition : 
  ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (f(f(x1)) = 4 ∧ f(f(x2)) = 4) :=
sorry

end count_values_satisfying_condition_l328_328736


namespace three_digit_values_satisfying_condition_l328_328365

-- Define the sum of digits function for a given positive integer x
def club (x : ℕ) : ℕ := (String.mk x.digits).foldr (λ c sum => sum + (c.toNat - '0'.toNat)) 0

-- Define the range of three-digit integers
def is_three_digit (x : ℕ) : Prop := 100 ≤ x ∧ x ≤ 999

-- Define the main theorem statement
theorem three_digit_values_satisfying_condition : {x : ℕ // is_three_digit x ∧ club (club x) = 4}.toFinset.card = 51 := by
  sorry

end three_digit_values_satisfying_condition_l328_328365


namespace problem1_problem2_problem3_l328_328376

-- Definitions and conditions
def a (α : Real) : Vector ℝ := (4 * Real.cos α, Real.sin α)
def b (β : Real) : Vector ℝ := (Real.sin β, 4 * Real.cos β)
def c (β : Real) : Vector ℝ := (Real.cos β, -4 * Real.sin β)

-- Problem 1: Perpendicularity and tangent value
theorem problem1 (α β : Real) (h : a α ⬝ (b β - 2 • c β) = 0) : Real.tan (α + β) = 2 :=
sorry

-- Problem 2: Maximum value of vector sum magnitude
theorem problem2 (β : Real) : ∃ (m : Real), m = |b β + c β| ∧ m = 4 * Real.sqrt 2 :=
sorry

-- Problem 3: Collinearity given a tangent product condition
theorem problem3 (α β : Real) (h : Real.tan α * Real.tan β = 16) : a α ∥ b β :=
sorry

end problem1_problem2_problem3_l328_328376


namespace smallest_n_divisible_l328_328869

theorem smallest_n_divisible (n : ℕ) : (15 * n - 3) % 11 = 0 ↔ n = 9 := by
  sorry

end smallest_n_divisible_l328_328869


namespace carrie_profit_l328_328168

def weekday_rate := 35
def weekend_rate := 45
def hours_per_day := 4
def total_days := 8
def cost_of_supplies := 180
def discount := 0.10
def sales_tax_rate := 0.07
def weekdays := 5
def weekends := 3

def total_weekday_hours := weekdays * hours_per_day
def total_weekend_hours := weekends * hours_per_day
def weekday_earnings := total_weekday_hours * weekday_rate
def weekend_earnings := total_weekend_hours * weekend_rate
def total_earnings := weekday_earnings + weekend_earnings
def discounted_supplies := cost_of_supplies * (1 - discount)
def sales_tax := total_earnings * sales_tax_rate
def profit := total_earnings - discounted_supplies - sales_tax

theorem carrie_profit : profit = 991.20 := by sorry

end carrie_profit_l328_328168


namespace axis_of_symmetry_parabola_l328_328631

theorem axis_of_symmetry_parabola (a b c : ℝ) :
  (∀ x : ℝ, y = a * x ^ 2 + b * x + c) →
  (y = 0 → x = -1 ∨ x = 2) →
  axis_of_symmetry = (λ x : ℝ, x = 1 / 2) :=
sorry

end axis_of_symmetry_parabola_l328_328631


namespace complex_point_correspondence_l328_328247

theorem complex_point_correspondence :
  let z := 2 / (-1 - complex.I) in
  let z_conj := complex.conj z in
  let result := complex.I * z_conj in
  result = 1 - complex.I :=
by
  sorry

end complex_point_correspondence_l328_328247


namespace part_I_max_value_part_II_range_b_l328_328905

-- Definition of the function f(x)
def f (x : ℝ) : ℝ :=
if h : x > 0 then 1 - x^2 * Real.log x
else Real.exp (-x - 2)

-- Part (I): Prove the maximum value of f(x) when x > 0
theorem part_I_max_value {x : ℝ} (hx : x > 0) : 
  (∃ y, ∀ z, f z ≤ f y) ∧ f (1 / Real.sqrt Real.exp 1) = 1 + 1 / (2 * Real.exp 1) :=
by
  sorry

-- Part (II): Prove the range of b such that f(x) + ax^2 + bx = 0 has three distinct real roots
theorem part_II_range_b (a : ℝ) (ha : a ≥ 0) : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ x1 ≠ 0 ∧ x2 ≠ 0 ∧ x3 ≠ 0 ∧ f x1 + a * x1^2 + b * x1 = 0 ∧ f x2 + a * x2^2 + b * x2 = 0 ∧ f x3 + a * x3^2 + b * x3 = 0) ↔ b ∈ Icc (1 / Real.exp 1) - ∞ ∪ Ioi (2 * Real.sqrt 2) :=
by
  sorry

end part_I_max_value_part_II_range_b_l328_328905


namespace largest_common_divisor_525_385_l328_328071

theorem largest_common_divisor_525_385 : 
  ∀ (factors525 factors385 : set ℕ), 
    factors525 = {1, 3, 5, 7, 15, 21, 25, 35, 75, 105, 175, 525} → 
    factors385 = {1, 5, 7, 35, 55, 77, 385} → 
    ∃ (d : ℕ), d ∈ factors525 ∧ d ∈ factors385 ∧
      ∀ (d' : ℕ), d' ∈ factors525 ∧ d' ∈ factors385 → d' ≤ d := 
begin
  intros,
  sorry
end

end largest_common_divisor_525_385_l328_328071


namespace correct_number_of_true_statements_l328_328950

-- Definitions based on conditions and statements
def even_numbers_except_two form_a_set : Prop := True  -- Statement ①
def tall_students_first_year_not_form_a_set : Prop := True  -- Statement ② (negated form because "tall" is not defined)
def unequal_sets_are_equal : Prop := False  -- Statement ③
def events_rio_2016_form_a_set : Prop := True  -- Statement ④

-- Main theorem statement
theorem correct_number_of_true_statements : even_numbers_except_two form_a_set ∧ ¬tall_students_first_year_not_form_a_set ∧ ¬unequal_sets_are_equal ∧ events_rio_2016_form_a_set → 2 = 2 :=
by sorry

end correct_number_of_true_statements_l328_328950


namespace contradiction_proof_start_l328_328469

theorem contradiction_proof_start (a b c : ℕ) : 
  ¬ (∃ x ∈ {a, b, c}, x % 2 = 0) ↔ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) :=
sorry

end contradiction_proof_start_l328_328469


namespace triangle_probability_l328_328402

def sticks : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def forms_triangle (a b c : ℕ) : Prop :=
  (a < b + c) ∧ (b < c + a) ∧ (c < a + b)

def valid_combinations : List (ℕ × ℕ × ℕ) :=
  List.filter (λ (abc : ℕ × ℕ × ℕ), forms_triangle abc.1 abc.2 abc.3)
    [ (2, 3, 5), (2, 3, 7), (2, 3, 11), (2, 3, 13), (2, 3, 17), (2, 3, 19), (2, 3, 23), (2, 3, 29),
      (2, 5, 7), (2, 5, 11), (2, 5, 13), (2, 5, 17), (2, 5, 19), (2, 5, 23), (2, 5, 29),
      (2, 7, 11), (2, 7, 13), (2, 7, 17), (2, 7, 19), (2, 7, 23), (2, 7, 29),
      (2, 11, 13), (2, 11, 17), (2, 11, 19), (2, 11, 23), (2, 11, 29),
      (2, 13, 17), (2, 13, 19), (2, 13, 23), (2, 13, 29),
      (2, 17, 19), (2, 17, 23), (2, 17, 29),
      (2, 19, 23), (2, 19, 29),
      (2, 23, 29),
      (3, 5, 7), (3, 5, 11), (3, 5, 13), (3, 5, 17), (3, 5, 19), (3, 5, 23), (3, 5, 29),
      (3, 7, 11), (3, 7, 13), (3, 7, 17), (3, 7, 19), (3, 7, 23), (3, 7, 29),
      (3, 11, 13), (3, 11, 17), (3, 11, 19), (3, 11, 23), (3, 11, 29),
      (3, 13, 17), (3, 13, 19), (3, 13, 23), (3, 13, 29),
      (3, 17, 19), (3, 17, 23), (3, 17, 29),
      (3, 19, 23), (3, 19, 29),
      (3, 23, 29),
      (5, 7, 11), (5, 7, 13), (5, 7, 17), (5, 7, 19), (5, 7, 23), (5, 7, 29),
      (5, 11, 13), (5, 11, 17), (5, 11, 19), (5, 11, 23), (5, 11, 29),
      (5, 13, 17), (5, 13, 19), (5, 13, 23), (5, 13, 29),
      (5, 17, 19), (5, 17, 23), (5, 17, 29),
      (5, 19, 23), (5, 19, 29),
      (5, 23, 29),
      (7, 11, 13), (7, 11, 17), (7, 11, 19), (7, 11, 23), (7, 11, 29),
      (7, 13, 17), (7, 13, 19), (7, 13, 23), (7, 13, 29),
      (7, 17, 19), (7, 17, 23), (7, 17, 29),
      (7, 19, 23), (7, 19, 29),
      (7, 23, 29),
      (11, 13, 17), (11, 13, 19), (11, 13, 23), (11, 13, 29),
      (11, 17, 19), (11, 17, 23), (11, 17, 29),
      (11, 19, 23), (11, 19, 29),
      (11, 23, 29),
      (13, 17, 19), (13, 17, 23), (13, 17, 29),
      (13, 19, 23), (13, 19, 29),
      (13, 23, 29),
      (17, 19, 23), (17, 19, 29),
      (17, 23, 29),
      (19, 23, 29) ]

def probability_of_triangle : ℚ :=
  (valid_combinations.length : ℚ) / (Nat.choose sticks.length 3 : ℚ)

theorem triangle_probability : probability_of_triangle = 2 / 5 :=
by
  sorry

end triangle_probability_l328_328402


namespace area_BDFC_l328_328098

-- Define the problem conditions
variable (AB AC CD : ℝ) (B D C F : ℝ)

-- Given conditions based on the problem statement
axiom angle_BAC : ∠BAC = 90
axiom AB_length : AB = 2
axiom AC_length : AC = 1
axiom BC_length : BC = Real.sqrt (AB^2 + AC^2)
axiom CD_length : CD = 2
axiom BD_length : (BC + CD) = Real.sqrt(5) + 2
axiom BE_midpoint : BE = AB / 2
axiom EF_perpendicular_AC : EF ⊥ AC

-- State the goal: the area of quadrilateral BDFC
theorem area_BDFC : Area_quadrilateral BDFC = Real.sqrt(5) := by
  sorry

end area_BDFC_l328_328098


namespace circles_common_point_l328_328059

theorem circles_common_point {n : ℕ} (hn : n ≥ 5) (circles : Fin n → Set Point)
  (hcommon : ∀ (a b c : Fin n), (circles a ∩ circles b ∩ circles c).Nonempty) :
  ∃ p : Point, ∀ i : Fin n, p ∈ circles i :=
sorry

end circles_common_point_l328_328059


namespace limit_of_x3_minus_8_div_x_minus_2_l328_328193

noncomputable def limit_equiv : Prop :=
  ∀ f : ℝ → ℝ, f = (λ x, (x^3 - 8) / (x - 2)) → tendsto f (nhds 2) (nhds 12)

-- theorem proving limit of the function as x approaches 2 is 12:
theorem limit_of_x3_minus_8_div_x_minus_2 : limit_equiv := sorry

end limit_of_x3_minus_8_div_x_minus_2_l328_328193


namespace seq_inequality_l328_328607

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 6 ∧ ∀ n, n > 3 → a n = 3 * a (n - 1) - a (n - 2) - 2 * a (n - 3)

theorem seq_inequality (a : ℕ → ℕ) (h : seq a) : ∀ n, n > 3 → a n > 3 * 2 ^ (n - 2) :=
  sorry

end seq_inequality_l328_328607


namespace geometric_seq_ratio_l328_328363

theorem geometric_seq_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 5 - a 3 = 12) 
  (h2 : a 6 - a 4 = 24) 
  (ha : ∃ a₁ : ℝ, (∀ n : ℕ, a n = a₁ * q ^ (n - 1)))
  (hS : ∀ n : ℕ, S n = a₁ * (1 - q ^ n) / (1 - q)) :
  ∀ n : ℕ, S n / a n = 2 - 2 ^ (1 - n) :=
sorry

end geometric_seq_ratio_l328_328363


namespace det_A_zero_l328_328778

theorem det_A_zero
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h1 : a11 = Real.sin (x1 - y1)) (h2 : a12 = Real.sin (x1 - y2)) (h3 : a13 = Real.sin (x1 - y3))
  (h4 : a21 = Real.sin (x2 - y1)) (h5 : a22 = Real.sin (x2 - y2)) (h6 : a23 = Real.sin (x2 - y3))
  (h7 : a31 = Real.sin (x3 - y1)) (h8 : a32 = Real.sin (x3 - y2)) (h9 : a33 = Real.sin (x3 - y3)) :
  (Matrix.det ![![a11, a12, a13], ![a21, a22, a23], ![a31, a32, a33]]) = 0 := sorry

end det_A_zero_l328_328778


namespace prob_not_A_and_not_B_l328_328780

variable (Ω : Type) [probSpace : MeasureTheory.ProbabilitySpace Ω]

-- Definitions of events A and B
variable (A B : Set Ω) 

-- Given conditions
axiom prob_A : probabilisticMeasure.A xi (A) = 3/4
axiom prob_B : probabilisticMeasure.A xi (B) = 1/2
axiom prob_A_and_B : probabilisticMeasure.Intersection (A ∩ B) = 3/8

-- Statement to be proved
theorem prob_not_A_and_not_B : probabilisticMeasure.Intersection (Aᶜ ∩ Bᶜ) = 1/8 := by
  sorry

end prob_not_A_and_not_B_l328_328780


namespace solution_set_l328_328422

variable (f : ℝ → ℝ)

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = - (f x)
axiom domain_R (f : ℝ → ℝ) : ∀ x, x ∈ Set.univ
axiom f_positive (x : ℝ) : 0 < x → f x = log_base 3 x

theorem solution_set (f : ℝ → ℝ) (odd_function f) (domain_R f) (f_positive) :
  {x : ℝ | f x ≥ 0} = {x : ℝ | -1 ≤ x ∧ x < 0} ∪ {x : ℝ | 1 ≤ x} :=
by
  sorry

end solution_set_l328_328422


namespace smallest_sum_p_q_l328_328988

theorem smallest_sum_p_q (p q : ℕ) (h1: p > 0) (h2: q > 0) (h3 : (∃ k1 k2 : ℕ, 7 ^ (p + 4) * 5 ^ q * 2 ^ 3 = (k1 * 7 *  k2 * 5 * (2 * 3))) ^ 3) :
  p + q = 5 :=
by
  -- Proof goes here
  sorry

end smallest_sum_p_q_l328_328988


namespace student_chosen_number_l328_328144

theorem student_chosen_number : 
  ∃ x : ℝ, sqrt (2 * x^2 - 138) = 9 ∧ x = sqrt 109.5 :=
by 
  -- the proof starts here
  sorry

end student_chosen_number_l328_328144


namespace domain_eq_l328_328291

theorem domain_eq (f : ℝ → ℝ) : 
  (∀ x : ℝ, -1 ≤ 3 - 2 * x ∧ 3 - 2 * x ≤ 2) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 :=
by sorry

end domain_eq_l328_328291


namespace distinct_lines_in_4x4_grid_l328_328974

open Nat

theorem distinct_lines_in_4x4_grid : 
  let n := 16
  let total_pairs := choose n 2
  let overcount_correction := 8
  let additional_lines := 4
  total_pairs - 2 * overcount_correction + additional_lines = 108 := 
by
  sorry

end distinct_lines_in_4x4_grid_l328_328974


namespace birds_flew_up_l328_328105

theorem birds_flew_up (initial_birds new_birds total_birds : ℕ) 
    (h_initial : initial_birds = 29) 
    (h_total : total_birds = 42) : 
    new_birds = total_birds - initial_birds := 
by 
    sorry

end birds_flew_up_l328_328105


namespace hotel_loss_l328_328129

theorem hotel_loss :
  (ops_expenses : ℝ) (payment_frac : ℝ) (total_received : ℝ) (loss : ℝ)
  (h_ops_expenses : ops_expenses = 100)
  (h_payment_frac : payment_frac = 3 / 4)
  (h_total_received : total_received = payment_frac * ops_expenses)
  (h_loss : loss = ops_expenses - total_received) :
  loss = 25 :=
by
  sorry

end hotel_loss_l328_328129


namespace tangent_at_point_l328_328598

def f (x : ℝ) := (2 * x - 1) * real.log x + (x ^ 2) / 2
def tangent_line_eq (x y : ℝ) := 4 * x - 2 * y - 3 = 0

theorem tangent_at_point:
  tangent_line_eq 1 (f 1) :=
sorry

end tangent_at_point_l328_328598


namespace true_proposition_is_d_l328_328517

/--
Among the following propositions, the true proposition is:
1. The sum of two acute angles is always an obtuse angle.
2. Equal angles are vertical angles.
3. Numbers with square roots are always irrational numbers.
4. The perpendicular segment is the shortest.

We need to prove that the fourth proposition is true.
-/
theorem true_proposition_is_d :
  (∀ a b : ℝ, a < 90 ∧ b < 90 → a + b < 180) →    -- Proposition A
  (∀ θ φ : ℝ, θ = φ → (∃ x : ℝ, (θ = 180 - φ) ∧ (θ ≠ φ + x))) →  -- Proposition B
  (∀ n : ℝ, ∃ k : ℕ, (n = k * k) → irrational n) →  -- Proposition C
  (∀ p l : Set Point, ∃ m : Point, m ∈ l ∧ perpendicular p m = shortest p l) →  -- Proposition D (which we need to prove as true)
  True :=
by
  sorry

end true_proposition_is_d_l328_328517


namespace overall_support_percentage_l328_328510

def men_support_percentage : ℝ := 0.75
def women_support_percentage : ℝ := 0.70
def number_of_men : ℕ := 200
def number_of_women : ℕ := 800

theorem overall_support_percentage :
  ((men_support_percentage * ↑number_of_men + women_support_percentage * ↑number_of_women) / (↑number_of_men + ↑number_of_women) * 100) = 71 := 
by 
sorry

end overall_support_percentage_l328_328510


namespace jack_initial_flower_pattern_plates_l328_328323

theorem jack_initial_flower_pattern_plates:
  ∀ (F C P : ℕ), 
  (C = 8) → 
  (P = 2 * C) → 
  (F - 1 + C + P = 27) → 
  (F = 4) :=
by
  intros F C P
  intro hC
  intro hP
  intro hTotal
  rw hC at hP
  rw hC at hTotal
  rw hP at hTotal
  linarith

end jack_initial_flower_pattern_plates_l328_328323


namespace scalar_triple_product_symmetry_scalar_triple_product_sum_l328_328089

-- Assuming S is a scalar triple product function.
def S (A B C : Point) : ℝ := sorry

-- Problem (a)
theorem scalar_triple_product_symmetry (A B C : Point) :
  S(A, B, C) = -S(B, A, C) ∧ S(A, B, C) = S(B, C, A) := sorry

-- Problem (b)
theorem scalar_triple_product_sum (A B C D : Point) :
  S(A, B, C) = S(D, A, B) + S(D, B, C) + S(D, C, A) := sorry

end scalar_triple_product_symmetry_scalar_triple_product_sum_l328_328089


namespace find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square_l328_328462

theorem find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square :
  ∃ n : ℕ, (4^n + 5^n) = k^2 ↔ n = 1 :=
by
  sorry

end find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square_l328_328462


namespace value_of_a_19_l328_328586

noncomputable def sequence (n : ℕ) : ℤ :=
  if n = 1 then 3
  else if n = 2 then 6
  else sequence (n - 1) - sequence (n - 2)

theorem value_of_a_19 : sequence 19 = 3 :=
sorry

end value_of_a_19_l328_328586


namespace remainder_division_l328_328769

theorem remainder_division :
  let a := 2 ^ 210 + 210
  let b := 2 ^ 105 + 2 ^ 52 + 3
  a % b = 210 :=
by {
  let a := 2 ^ 210 + 210,
  let b := 2 ^ 105 + 2 ^ 52 + 3,
  sorry
}

end remainder_division_l328_328769


namespace problem_1_problem_2_l328_328892

-- Conditions as definitions
def is_bounded (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∃ M > 0, ∀ x ∈ D, |f x| ≤ M

def test_function (a x : ℝ) : ℝ :=
  1 + a * (1 / 2) ^ x + (1 / 4) ^ x

-- Problem (1)
theorem problem_1 : ¬ is_bounded (test_function 1) (Set.Iio 0) := by
  sorry

-- Problem (2)
theorem problem_2 (hb : is_bounded (test_function a) (Set.Ici 0)) : a ∈ Set.Icc (-5 : ℝ) 1 := by
  sorry

end problem_1_problem_2_l328_328892


namespace cos_of_angle_in_third_quadrant_l328_328628

theorem cos_of_angle_in_third_quadrant (B : ℝ) (h1 : -1 ≤ sin B ∧ sin B ≤ 1) (h2 : sin B = -5 / 13) (h3 : 3 * π / 2 ≤ B ∧ B ≤ 2 * π) :
  cos B = -12 / 13 :=
by
  sorry

end cos_of_angle_in_third_quadrant_l328_328628


namespace lowest_price_l328_328473

theorem lowest_price (
  cost_per_component : ℕ := 80,
  shipping_cost_per_unit : ℕ := 3,
  fixed_monthly_costs : ℕ := 16500,
  num_components_per_month : ℕ := 150
) : 
  let total_cost := (cost_per_component + shipping_cost_per_unit) * num_components_per_month + fixed_monthly_costs in
  let lowest_price_per_component := total_cost / num_components_per_month in
  lowest_price_per_component = 193 :=
by 
  sorry

end lowest_price_l328_328473


namespace sum_of_remainders_gt_2n_l328_328446

open Nat

theorem sum_of_remainders_gt_2n (n : ℕ) (h : n > 1970) : 
  (∑ k in Finset.range (n + 1) \ {0, 1}, 2^n % k) > 2 * n :=
by
  sorry

end sum_of_remainders_gt_2n_l328_328446


namespace all_points_on_single_quadratic_l328_328751

theorem all_points_on_single_quadratic (points : Fin 100 → (ℝ × ℝ)) :
  (∀ (p1 p2 p3 p4 : Fin 100),
    ∃ a b c : ℝ, 
      ∀ (i : Fin 100), 
        (i = p1 ∨ i = p2 ∨ i = p3 ∨ i = p4) →
          (points i).snd = a * (points i).fst ^ 2 + b * (points i).fst + c) → 
  ∃ a b c : ℝ, ∀ i : Fin 100, (points i).snd = a * (points i).fst ^ 2 + b * (points i).fst + c :=
by 
  sorry

end all_points_on_single_quadratic_l328_328751


namespace correct_statement_C_l328_328773

-- Definitions based on the conditions
def is_monomial (expr : Expr) : Prop :=
  -- Definition of a monomial (single term algebraic expression)
  sorry

def coefficient (expr : Expr) : ℚ :=
  -- Function to extract the coefficient of the given expression
  sorry

def degree (expr : Expr) : ℕ :=
  -- Function to extract the degree of the given expression
  sorry

-- Specific expressions
def expr_A : Expr := 4
def expr_B : Expr := - (1/2) * x * y
def expr_C : Expr := (1/3) * x^2 * y
def expr_D : Expr := π * r^2

-- Prove the statement C
theorem correct_statement_C : coefficient expr_C = 1/3 := by sorry

end correct_statement_C_l328_328773


namespace problem_statement_l328_328344

variable {α : Type*} [field α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0     := a₁
| (n+1) := geometric_sequence a₁ q n * q

def Sn (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * n else a₁ * (1 - q ^ n) / (1 - q)

theorem problem_statement (a₁ q : α) (n : ℕ) (h₀ : q ≠ 1)
  (h₁ : geometric_sequence a₁ q 4 - geometric_sequence a₁ q 2 = 12)
  (h₂ : geometric_sequence a₁ q 5 - geometric_sequence a₁ q 3 = 24) :
  Sn a₁ q n / geometric_sequence a₁ q (n - 1) = 2 - 2 ^ (1 - n) :=
sorry

end problem_statement_l328_328344


namespace diameter_of_circle_l328_328639

-- Definitions and conditions
variables {O A B C D P : Type*}
variables {a b : ℝ}
variables {circle : set ℝ} -- Defining the circle as a set of real numbers for simplicity

-- The diameter of the circle (needed for the goal)
variable (AB : ℝ)

-- We state the conditions
-- 1. AB is the diameter of the circle
-- 2. AD and BC are tangents to the circle with lengths 2a and 2b respectively
-- 3. P is the intersection of lines AC and BD inside the circle
def conditions (h1 : ∃ O, AB / 2 = O) 
               (h2 : ∃ A D, ∥A - D∥ = 2a)
               (h3 : ∃ B C, ∥B - C∥ = 2b)
               (h4 : ∃ P, P ∈ circle ∧ P = lineIntersect A C B D) : Prop :=
  true -- Placeholder for all the conditions together

-- The goal to prove
theorem diameter_of_circle {O A B C D P : Type*} 
                            {a b : ℝ} {circle : set ℝ}
                            (h1 : ∃ O, AB / 2 = O)
                            (h2 : ∃ A D, ∥A - D∥ = 2a)
                            (h3 : ∃ B C, ∥B - C∥ = 2b)
                            (h4 : ∃ P, P ∈ circle ∧ P = lineIntersect A C B D) : 
                            AB = 2 * (a + b) := by
sorry

end diameter_of_circle_l328_328639


namespace find_d_l328_328926

noncomputable def a_n : ℕ → ℝ := 
sorry   -- Define the arithmetic sequence

theorem find_d : 
  ∃ d < 0, 
    let a_1 := a_n 0 in
    let a_2 := a_1 + d in
    let a_9 := a_1 + 8 * d in
    (3 * Real.sqrt 5)^2 = (-a_2) * a_9 ∧ 
    10 * (a_1 + 4.5 * d) = 20 ∧ 
    d = (-39)/4 := 
sorry

end find_d_l328_328926


namespace find_perpendicular_line_to_l_l328_328957

-- Define the original line l
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the line l' we need to find
def perpendicular_line (x y : ℝ) (m : ℝ) : Prop := 4 * x - 3 * y + m = 0

-- The area of the triangle formed by line l' with coordinate axes is 4
def triangle_area_condition (m : ℝ) : Prop := (1 / 2) * (abs (m / 3)) * (abs (-m / 4)) = 4

theorem find_perpendicular_line_to_l :
  ∃ m : ℝ, triangle_area_condition m ∧ (∀ x y : ℝ, perpendicular_line x y m ↔ 4 * x - 3 * y = ±4 * sqrt 6) := sorry

end find_perpendicular_line_to_l_l328_328957


namespace balls_into_boxes_l328_328275

theorem balls_into_boxes (n k : ℕ) (hn : n = 7) (hk : k = 3) :
  (nat.choose (n + k - 1) (k - 1)) = 36 :=
by
  sorry

end balls_into_boxes_l328_328275


namespace parallel_line_eq_l328_328248

theorem parallel_line_eq (x y : ℝ) (hx : 3 * x + 4 * y + 1 = 0) (hpt : (1, 2)) : 
  ∃ c : ℝ, 3 * 1 + 4 * 2 + c = 0 ∧ c = -11 :=
sorry

end parallel_line_eq_l328_328248


namespace minimum_pound_requirement_l328_328845

theorem minimum_pound_requirement
  (cost_per_pound : ℕ) (baxter_spent : ℕ) (over_minimum : ℕ) (x : ℕ)
  (h1 : cost_per_pound = 3)
  (h2 : baxter_spent = 105)
  (h3 : over_minimum = 20) :
  3 * (x + 20) = 105 → x = 15 :=
begin
  intros h,
  calc
  3 * (x + 20) = 105 : h
              ... = 3 * x + 60 : by rw [mul_add, h1, h3]
              ... = 3 * x + 60 : by sorry
end

end minimum_pound_requirement_l328_328845


namespace symmetric_point_origin_l328_328651

theorem symmetric_point_origin :
  ∀ (x y : ℝ), (x, y) = (-2, 3) → (-x, -y) = (2, -3) :=
by
  intros x y h
  cases h
  exacts (rfl, rfl)

end symmetric_point_origin_l328_328651


namespace sequence_a_formula_sequence_T_formula_l328_328922

def sequence_S (n : ℕ) : ℕ := n^2 - 4 * n + 4

def sequence_a (n : ℕ) : ℕ :=
match n with
| 0 => 1
| n+1 => if n = 0 then 1 else 2 * (n + 1) - 5

def sequence_b (n : ℕ) : ℕ :=
match n with
| 0 => 1
| n+1 => if n = 0 then 1 else (sequence_a (n + 1) + 5) / 2

def sequence_T (n : ℕ) : ℕ :=
2 * 1 + ∑ i in list.range n, 2^(i + 1) * sequence_b (i + 1)

theorem sequence_a_formula :
  ∀ n : ℕ, sequence_a n = if n = 0 then 1 else 2 * n - 5 :=
sorry

theorem sequence_T_formula :
  ∀ n : ℕ, sequence_T n = (n-1) * 2^(n+1) + 2 :=
sorry

end sequence_a_formula_sequence_T_formula_l328_328922


namespace proof_theorem_l328_328016

noncomputable def proof_problem : Prop :=
  sqrt (sqrt[3] (sqrt (1 / 8192))) = 2 ^ (- 13 / 12)

theorem proof_theorem : proof_problem := by
  sorry

end proof_theorem_l328_328016


namespace max_value_quadratic_on_interval_l328_328052

open Real

noncomputable def quadratic_fn (x : ℝ) : ℝ :=
  -4 * x^2 + 8 * x + 3

theorem max_value_quadratic_on_interval :
  ∃ x ∈ (Ioo 0 3), ∀ y ∈ (Ioo 0 3), quadratic_fn y ≤ quadratic_fn x := sorry

end max_value_quadratic_on_interval_l328_328052


namespace marked_price_percentage_l328_328822

variable (L P M S : ℝ)

-- Conditions
def original_list_price := 100               -- L = 100
def purchase_price := 70                     -- P = 70
def required_profit_price := 91              -- S = 91
def final_selling_price (M : ℝ) := 0.85 * M  -- S = 0.85M

-- Question: What percentage of the original list price should the marked price be?
theorem marked_price_percentage :
  L = original_list_price →
  P = purchase_price →
  S = required_profit_price →
  final_selling_price M = S →
  M = 107.06 := sorry

end marked_price_percentage_l328_328822


namespace find_rotation_center_l328_328047

noncomputable def f (z : ℂ) : ℂ :=
  ((-1 - complex.i * real.sqrt 3) * z + (2 * real.sqrt 3 - 18 * complex.i)) / 2

theorem find_rotation_center :
  ∃ (d : ℂ), f d = d ∧ d = -real.sqrt 3 + 4 * complex.i :=
by
  sorry

end find_rotation_center_l328_328047


namespace find_income_on_first_day_l328_328485

theorem find_income_on_first_day (income_day_2 income_day_3 income_day_4 income_day_5 : ℕ) (avg_income : ℕ) : 
  income_day_2 = 150 → income_day_3 = 750 → income_day_4 = 200 → income_day_5 = 600 → avg_income = 400 →
  let income_sum := 300 + income_day_2 + income_day_3 + income_day_4 + income_day_5 in
  (income_sum / 5 = avg_income) →
  300 = 300 :=
by
  intros h2 h3 h4 h5 h_avg h_sum
  rw [h2, h3, h4, h5] at h_sum
  norm_num at h_sum
  subst h_sum
  exact rfl

end find_income_on_first_day_l328_328485


namespace find_x_range_l328_328242

variable {f : ℝ → ℝ}
hypothesis odd_f : ∀ x, f(-x) = -f(x)
hypothesis decreasing_f : ∀ x y, x < y → f(x) > f(y)
hypothesis f_domain : ∀ x, -3 ≤ x → x ≤ 3 → f(x) ∈ set.Icc (-3 : ℝ) (3 : ℝ)

theorem find_x_range (h : ∀ x, -3 ≤ x → x ≤ 3 → f(x^2 - 2 * x) + f(x - 2) < 0) : 
  ∀ x, 2 < x → x ≤ 3 → h x (-3) 3 → x ∈ set.Ioc (2 : ℝ) (3 : ℝ) := 
sorry
 
end find_x_range_l328_328242


namespace prime_1021_n_unique_l328_328562

theorem prime_1021_n_unique :
  ∃! (n : ℕ), n ≥ 2 ∧ Prime (n^3 + 2 * n + 1) :=
sorry

end prime_1021_n_unique_l328_328562


namespace max_population_l328_328303

theorem max_population (max_teeth : ℕ) (h : max_teeth = 32) : 
  ∃ max_population : ℕ, max_population = 2 ^ max_teeth :=
by
  use 2 ^ max_teeth
  rw h
  sorry

end max_population_l328_328303


namespace wendy_percentage_accounting_l328_328457

noncomputable def years_as_accountant : ℕ := 25
noncomputable def years_as_manager : ℕ := 15
noncomputable def total_lifespan : ℕ := 80

def total_years_in_accounting : ℕ := years_as_accountant + years_as_manager

def percentage_of_life_in_accounting : ℝ := (total_years_in_accounting / total_lifespan) * 100

theorem wendy_percentage_accounting : percentage_of_life_in_accounting = 50 := by
  unfold total_years_in_accounting
  unfold percentage_of_life_in_accounting
  sorry

end wendy_percentage_accounting_l328_328457


namespace area_of_triangle_abd_l328_328454

/-- Two right triangles share a side as follows: 
  Triangle ABC has a right angle at A with AB = 8 units and AC = 15 units.
  A point D lies on side BC such that BD = 12 units and DC = 5 units,
  forming a right triangle ABD. This Lean statement shows that
  the area of triangle ABD is (720/17) square units. -/
theorem area_of_triangle_abd :
  ∃ (A B C D : Type) (AB AC BD DC : ℝ),
    (A B C D are_points) ∧
    (right_triangle A B C) ∧
    (right_triangle A B D) ∧
    (AB = 8) ∧
    (AC = 15) ∧
    (BD = 12) ∧
    (DC = 5) ∧
    (BC = 17) ∧
    (area A B C = 60) ∧
    (area A B D = 720 / 17) :=
by
  sorry

end area_of_triangle_abd_l328_328454


namespace non_intersecting_matching_l328_328099

theorem non_intersecting_matching (n : ℕ) (red blue : Fin n → ℝ × ℝ)
  (h_general_position : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ¬ collinear {red i, red j, red k} ∧ ¬ collinear {blue i, blue j, blue k} ∧ ¬ collinear {red i, blue j, red k}) :
  ∃ (f : Fin n → Fin n), (∀ (i j : Fin n), i ≠ j → ¬ segments_intersect (red i, blue (f i)) (red j, blue (f j))) :=
sorry

where
  collinear (pts : Set (ℝ × ℝ)) : Prop :=
    ∃ (a b c : ℝ), ∀ (p : pts), a * (p.1 - b) + c = 0

  segments_intersect (s1 s2 : ℝ × ℝ × ℝ × ℝ) : Prop :=
    ∃ (p : ℝ × ℝ), on_segment s1 p ∧ on_segment s2 p

  on_segment (s : ℝ × ℝ × ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
    let (x1, y1, x2, y2) := s in
    (min x1 x2 ≤ p.1 ∧ p.1 ≤ max x1 x2) ∧ (min y1 y2 ≤ p.2 ∧ p.2 ≤ max y1 y2) ∧
    (x1 - x2) * (p.2 - y2) = (y1 - y2) * (p.1 - x2)

end non_intersecting_matching_l328_328099


namespace range_of_m_l328_328907

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Icc (-1 : ℝ) 2, ∃ x0 ∈ Icc (-1 : ℝ) 2, (m * x1 + 2) = (x0^2 - 2 * x0)) ↔ (-1 ≤ m ∧ m ≤ 1/2) :=
sorry

end range_of_m_l328_328907


namespace rotation_image_of_D_l328_328963

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

theorem rotation_image_of_D :
  rotate_90_clockwise (-3, 2) = (2, 3) :=
by
  sorry

end rotation_image_of_D_l328_328963


namespace area_of_side_face_of_box_l328_328091

theorem area_of_side_face_of_box:
  ∃ (l w h : ℝ), (w * h = (1/2) * (l * w)) ∧
                 (l * w = 1.5 * (l * h)) ∧
                 (l * w * h = 3000) ∧
                 ((l * h) = 200) :=
sorry

end area_of_side_face_of_box_l328_328091


namespace regina_final_earnings_l328_328706

-- Define the number of animals Regina has
def cows := 20
def pigs := 4 * cows
def goats := pigs / 2
def chickens := 2 * cows
def rabbits := 30

-- Define sale prices for each animal
def cow_price := 800
def pig_price := 400
def goat_price := 600
def chicken_price := 50
def rabbit_price := 25

-- Define annual earnings from animal products
def cow_milk_income := 500
def rabbit_meat_income := 10

-- Define annual farm maintenance and animal feed costs
def maintenance_cost := 10000

-- Define a calculation for the final earnings
def final_earnings : ℕ :=
  let cow_income := cows * cow_price
  let pig_income := pigs * pig_price
  let goat_income := goats * goat_price
  let chicken_income := chickens * chicken_price
  let rabbit_income := rabbits * rabbit_price
  let total_animal_sale_income := cow_income + pig_income + goat_income + chicken_income + rabbit_income

  let cow_milk_earning := cows * cow_milk_income
  let rabbit_meat_earning := rabbits * rabbit_meat_income
  let total_annual_income := cow_milk_earning + rabbit_meat_earning

  let total_income := total_animal_sale_income + total_annual_income
  let final_income := total_income - maintenance_cost

  final_income

-- Prove that the final earnings is as calculated
theorem regina_final_earnings : final_earnings = 75050 := by
  sorry

end regina_final_earnings_l328_328706


namespace orthographic_projection_of_two_skew_lines_cannot_be_two_points_l328_328658

-- Definitions of skew lines and orthographic projections
structure Line (R : Type*) [Field R] := 
(point : R × R × R) 
(direction : R × R × R)

def orthographic_projection (plane_normal : ℝ × ℝ × ℝ) (l : Line ℝ) : Line ℝ :=
  -- Implementation of orthographic projection (details not expanded for brevity, use sorry)
  sorry

def are_skew {R : Type*} [Field R] (l1 l2 : Line R) : Prop :=
  ¬(l1.direction = l2.direction) ∧ ¬(∃ p : R × R × R, (p ∈ l1) ∧ (p ∈ l2))

-- Proving the statement
theorem orthographic_projection_of_two_skew_lines_cannot_be_two_points 
  (plane_normal : ℝ × ℝ × ℝ) (l1 l2 : Line ℝ) 
  (h_skew : are_skew l1 l2) : 
  orthographic_projection plane_normal l1 ≠ orthographic_projection plane_normal l2 :=
sorry

end orthographic_projection_of_two_skew_lines_cannot_be_two_points_l328_328658


namespace super_prime_looking_count_l328_328531

-- Define the concept of a super prime-looking number
def is_super_prime_looking (n : ℕ) : Prop :=
  ¬Prime n ∧ 1 < n ∧ ¬(2 ∣ n) ∧ ¬(3 ∣ n) ∧ ¬(5 ∣ n) ∧ ¬(7 ∣ n)

-- Given condition: number of prime numbers less than 1500
def count_primes_less_than_1500 : ℕ := 234

-- Prove that the number of super prime-looking numbers less than 1500 is 207
theorem super_prime_looking_count :
  ∃ n, (∀ k, k < 1500 → is_super_prime_looking k ↔ k ∈ n) ∧ n.card = 207 := 
sorry

end super_prime_looking_count_l328_328531


namespace value_of_t_plus_one_over_t_l328_328911

theorem value_of_t_plus_one_over_t
  (t : ℝ)
  (h1 : t^2 - 3 * t + 1 = 0)
  (h2 : t ≠ 0) :
  t + 1 / t = 3 :=
by
  sorry

end value_of_t_plus_one_over_t_l328_328911


namespace inequality_proof_l328_328702

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + a + 1) * (b^2 + b + 1) * (c^2 + c + 1)) / (a * b * c) ≥ 27 :=
by
  sorry

end inequality_proof_l328_328702


namespace find_a_l328_328595

def f (x a : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x * Real.cos x + (Real.cos x) ^ 2 + a

theorem find_a (a : ℝ) : (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3), 
                             Real.sqrt 3 * Real.sin x * Real.cos x + (Real.cos x) ^ 2 + a 
                             ≤ (Real.sin (2 * x + Real.pi / 6) + a + 1/2) ∧
                             Real.sqrt 3 * Real.sin x * Real.cos x + (Real.cos x) ^ 2 + a
                             ≥ (Real.sin (2 * x + Real.pi / 6) + a + 1/2)) →
                           ((Real.sin (2 * (Real.pi / 3) + Real.pi / 6) + a + 1/2) + 
                           (Real.sin (-2 * (Real.pi / 6) + Real.pi / 6) + a + 1/2) = 3 / 2) → 
                           a = 0 :=
by
  -- sorry is used to indicate where the proof would go
  sorry

end find_a_l328_328595


namespace solve_for_x_l328_328397

-- Mathematical problem given the condition and required solution
theorem solve_for_x (x : ℝ) (h : real.cbrt (5 + 2/x) = -3) : x = -1/16 := 
by
  sorry

end solve_for_x_l328_328397


namespace geometric_sequence_ratio_l328_328349

variable {a : ℕ → ℕ}
variables (S : ℕ → ℕ) (n : ℕ)
def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, ∀ n : ℕ, a n = a 1 * q^n

variables (h1 : a 5 - a 3 = 12) (h2 : a 6 - a 4 = 24)

theorem geometric_sequence_ratio :
  is_geometric_sequence a →
  ∃ S : ℕ → ℕ, (S n = ∑ i in range n, a i) ∧
  (S n / a n = 2 - 2^(1 - n)) :=
by
  sorry

end geometric_sequence_ratio_l328_328349


namespace interval_length_difference_l328_328544

theorem interval_length_difference (a b : ℝ) (h₁ : a < b)
  (h₂ : ∀ x : ℝ, a ≤ x ∧ x ≤ b → 0 ≤ Real.log2 x ∧ Real.log2 x ≤ 2) :
  (b - a) = 3 :=
begin
  sorry
end

end interval_length_difference_l328_328544


namespace inverse_of_given_matrix_is_expected_l328_328885

open Matrix

def given_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5, -3], ![-2, 1]]

def expected_inverse : Matrix (Fin 2) (Fin 2) ℚ :=
  ![[-1, -3], ![-2, -5]]

theorem inverse_of_given_matrix_is_expected :
  (det given_matrix ≠ 0) → inverse given_matrix = expected_inverse := by
  sorry

end inverse_of_given_matrix_is_expected_l328_328885


namespace oranges_in_basket_l328_328023

def initial_oranges (taken remaining : ℕ) : ℕ := taken + remaining

theorem oranges_in_basket (taken remaining : ℕ) (ht : taken = 5) (hr : remaining = 3) :
  initial_oranges taken remaining = 8 :=
by
  rw [ht, hr]
  simp [initial_oranges]
  sorry

end oranges_in_basket_l328_328023


namespace area_of_quadrilateral_ABCM_l328_328539

-- Define the polygon properties
def Polygon16 (A B C D E F G H I J K L M N O P : Point) : Prop :=
  side_len AB = 5 ∧ side_len BC = 5 ∧ side_len CD = 5 ∧ side_len DE = 5 ∧
  side_len EF = 5 ∧ side_len FG = 5 ∧ side_len GH = 5 ∧ side_len HI = 5 ∧
  side_len IJ = 5 ∧ side_len JK = 5 ∧ side_len KL = 5 ∧ side_len LM = 5 ∧
  side_len MN = 5 ∧ side_len NO = 5 ∧ side_len OP = 5 ∧ side_len PA = 5 ∧
  angle_right (A, B, C) ∧ angle_right (B, C, D) ∧ angle_right (C, D, E) ∧ 
  angle_right (D, E, F) ∧ angle_right (E, F, G) ∧ angle_right (F, G, H) ∧
  angle_right (G, H, I) ∧ angle_right (H, I, J) ∧ angle_right (I, J, K) ∧ 
  angle_right (J, K, L) ∧ angle_right (K, L, M) ∧ angle_right (L, M, N) ∧
  angle_right (M, N, O) ∧ angle_right (N, O, P) ∧ angle_right (O, P, A)

-- Define the intersection and area proof
theorem area_of_quadrilateral_ABCM 
  (A B C D E F G H I J K L M N O P : Point)
  (hPolygon : Polygon16 A B C D E F G H I J K L M N O P)
  (hIntersect: Intersect (AI) (DL) M) :
  area (Quadrilateral A B C M) = 100 := 
sorry

end area_of_quadrilateral_ABCM_l328_328539


namespace calculate_original_eggs_l328_328309

noncomputable def original_eggs (goslings_first_year : Nat) (goslings_six_months : Nat) (goslings_three_months : Nat) (hatched_goslings : Nat) (goose_eggs : Nat) : Nat :=
  let survived_first_year := 270 / (2/5)
  let before_six_months := survived_first_year / (5/6)
  let before_three_months := before_six_months / (7/8)
  let hatched := before_three_months / (3/4)
  let rounded_hatched := Int.ceil hatched
  let eggs_laid := rounded_hatched / (4/5)
  Int.ceil eggs_laid

theorem calculate_original_eggs : original_eggs = 1934 :=
  sorry

end calculate_original_eggs_l328_328309


namespace geometric_sequence_ratio_l328_328347

variable {a : ℕ → ℕ}
variables (S : ℕ → ℕ) (n : ℕ)
def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, ∀ n : ℕ, a n = a 1 * q^n

variables (h1 : a 5 - a 3 = 12) (h2 : a 6 - a 4 = 24)

theorem geometric_sequence_ratio :
  is_geometric_sequence a →
  ∃ S : ℕ → ℕ, (S n = ∑ i in range n, a i) ∧
  (S n / a n = 2 - 2^(1 - n)) :=
by
  sorry

end geometric_sequence_ratio_l328_328347


namespace circle_tangency_problem_l328_328535

theorem circle_tangency_problem (r : ℝ) (p q : ℕ) :
  (p = 3) ∧ (q = 256) ∧ (p + q = 259) ↔ 
  let radius_C := 4,
      radius_D := 2 * r,
      radius_E := r,
      distance_CE := radius_C - radius_E,
      distance_CF := sqrt (4^2 - (4 - r)^2),
      distance_CD := radius_C - radius_D,
      distance_DF := radius_C - 2 * r + sqrt (8 * r - r^2),
      distance_DE := 4 * r
  in 
      (4 - 2 * r + sqrt (8 * r - r^2))^2 + r^2 = (4 * r)^2 ∧ 
      radius_D = (sqrt p) - q := sorry

end circle_tangency_problem_l328_328535


namespace rational_shift_polynomials_l328_328665

-- Define the problem statement
theorem rational_shift_polynomials (f g : Polynomial ℤ) :
  (∃ᶠ p in (Filter.atTop : Filter ℕ), ∃ m_p : ℤ,
    ∀ a : ℤ, Polynomial.eval a f % p = Polynomial.eval (a + m_p) g % p) →
  ∃ r : ℚ, ∀ x : ℚ, Polynomial.eval x f = Polynomial.eval (x + r) g :=
  sorry

end rational_shift_polynomials_l328_328665


namespace membership_percentage_change_l328_328828

theorem membership_percentage_change :
  let initial_membership := 100.0
  let first_fall_membership := initial_membership * 1.04
  let first_spring_membership := first_fall_membership * 0.95
  let second_fall_membership := first_spring_membership * 1.07
  let second_spring_membership := second_fall_membership * 0.97
  let third_fall_membership := second_spring_membership * 1.05
  let third_spring_membership := third_fall_membership * 0.81
  let final_membership := third_spring_membership
  let total_percentage_change := ((final_membership - initial_membership) / initial_membership) * 100.0
  total_percentage_change = -12.79 :=
by
  sorry

end membership_percentage_change_l328_328828


namespace binom_30_3_eq_4060_l328_328856

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := 
by sorry

end binom_30_3_eq_4060_l328_328856


namespace train_crossing_pole_time_l328_328825

noncomputable def time_to_cross_pole (speed_kmh : ℕ) (length_m : ℕ) : ℕ :=
  let speed_ms := (speed_kmh * 1000) / 3600 in
  length_m / speed_ms

theorem train_crossing_pole_time :
  time_to_cross_pole 180 900 = 18 :=
by
  sorry

end train_crossing_pole_time_l328_328825


namespace sasha_sequence_eventually_five_to_100_l328_328741

theorem sasha_sequence_eventually_five_to_100 :
  ∃ (n : ℕ), 
  (5 ^ 100) = initial_value + n * (3 ^ 100) - m * (2 ^ 100) ∧ 
  (initial_value + n * (3 ^ 100) - m * (2 ^ 100) > 0) :=
by
  let initial_value := 1
  let threshold := 2 ^ 100
  let increment := 3 ^ 100
  let decrement := 2 ^ 100
  sorry

end sasha_sequence_eventually_five_to_100_l328_328741


namespace bug_paths_count_l328_328111

theorem bug_paths_count :
  let A := point (0, 0)
  let B := point (n, n)  -- Assumed target point indices for clarity representation
  paths_count A B = 3360 :=
sorry

end bug_paths_count_l328_328111


namespace min_value_expression_l328_328206

theorem min_value_expression (x : ℝ) (hx : 0 < x ∧ x < 1) :
    x = sqrt 2 - 1 ↔ (∀ y, 0 < y ∧ y < 1 → (1 / x + 2 / (1 - x)) ≤ (1 / y + 2 / (1 - y))) := sorry

end min_value_expression_l328_328206


namespace polynomial_gcd_product_l328_328890

-- Definitions
def polynomial_gcd (P : polynomial ℤ) : ℤ := P.coeff_gcd

-- Conditions
variables (P Q : polynomial ℤ)
hypothesis hP : P ≠ 0
hypothesis hQ : Q ≠ 0

-- The theorem to be proved
theorem polynomial_gcd_product (P Q : polynomial ℤ) 
    (hP : P ≠ 0) (hQ : Q ≠ 0) : polynomial_gcd (P * Q) = polynomial_gcd P * polynomial_gcd Q := 
sorry

end polynomial_gcd_product_l328_328890


namespace remaining_half_speed_l328_328696

-- Define the given conditions
def total_time : ℕ := 11
def first_half_distance : ℕ := 150
def first_half_speed : ℕ := 30
def total_distance : ℕ := 300

-- Prove the speed for the remaining half of the distance
theorem remaining_half_speed :
  ∃ v : ℕ, v = 25 ∧
  (total_distance = 2 * first_half_distance) ∧
  (first_half_distance / first_half_speed = 5) ∧
  (total_time = 5 + (first_half_distance / v)) :=
by
  -- Proof omitted
  sorry

end remaining_half_speed_l328_328696


namespace simplify_expression_one_simplify_expression_two_l328_328019

-- Problem (1) Statement
theorem simplify_expression_one (α : ℝ) :
  (cos (α - π / 2) / sin (5 / 2 * π + α) * sin (α - π) * cos (2 * π - α)) = -sin(α) ^ 2 := 
sorry

-- Problem (2) Statement
theorem simplify_expression_two :
  (sqrt (1 - 2 * sin (20 * π / 180) * cos (200 * π / 180)) / 
   (cos (160 * π / 180) - sqrt (1 - cos (20 * π / 180) ^ 2))) = -1 :=
sorry

end simplify_expression_one_simplify_expression_two_l328_328019


namespace problem_statement_l328_328343

variable {α : Type*} [field α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0     := a₁
| (n+1) := geometric_sequence a₁ q n * q

def Sn (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * n else a₁ * (1 - q ^ n) / (1 - q)

theorem problem_statement (a₁ q : α) (n : ℕ) (h₀ : q ≠ 1)
  (h₁ : geometric_sequence a₁ q 4 - geometric_sequence a₁ q 2 = 12)
  (h₂ : geometric_sequence a₁ q 5 - geometric_sequence a₁ q 3 = 24) :
  Sn a₁ q n / geometric_sequence a₁ q (n - 1) = 2 - 2 ^ (1 - n) :=
sorry

end problem_statement_l328_328343


namespace expected_value_of_Xi_l328_328752

-- Definitions of the conditions given in the problem
def balls : List ℕ := [1, 1, 1, 2, 2, 2, 3, 3]

-- Define the function ξ which computes the product of the numbers on two drawn balls
def ξ (x y : ℕ) := x * y

-- Define the expected value calculation
def expected_value (l : List ℕ) : ℚ :=
  let p : ℚ := 1 / 8
  let all_combinations := do
    x ← l,
    y ← l,
    return ξ x y
  (all_combinations.sum * p * p : ℚ)

-- Prove that the expected value is equal to 225/64
theorem expected_value_of_Xi :
  expected_value balls = 225 / 64 :=
by
  sorry

end expected_value_of_Xi_l328_328752


namespace complement_union_l328_328264

theorem complement_union (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  U \ (A ∪ B) = {4} :=
by
  sorry

end complement_union_l328_328264


namespace paper_length_is_correct_l328_328502

/-- A piece of paper 3 cm wide is wrapped around a cardboard tube to form a paper roll for a drawing machine.
The roll is wrapped 450 times and the final diameter is 8 cm, starting from a diameter of 1 cm. -/
def paper_roll_length : ℝ :=
  let initial_diameter := 1 -- cm
  let final_diameter := 8 -- cm
  let wraps := 450
  let diameter_increase := (final_diameter - initial_diameter) / wraps
  let total_diameters := wraps * (initial_diameter + final_diameter) / 2
  let total_circumference := total_diameters * Real.pi
  total_circumference

theorem paper_length_is_correct : paper_roll_length = 2025 * Real.pi := by
  sorry

end paper_length_is_correct_l328_328502


namespace exercise_time_l328_328331

theorem exercise_time :
  let time_monday := 6 / 2 in
  let time_wednesday := 6 / 3 in
  let time_friday := 6 / 6 in
  let total_time := time_monday + time_wednesday + time_friday in
  total_time = 6 :=
by
  sorry

end exercise_time_l328_328331


namespace lesser_of_two_numbers_l328_328436

theorem lesser_of_two_numbers (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 6) : y = 22 :=
by
  sorry

end lesser_of_two_numbers_l328_328436


namespace pascal_triangle_row_contains_prime_l328_328274

-- Define the necessary properties and the main theorem
theorem pascal_triangle_row_contains_prime (p : ℕ) (h : Nat.Prime p) : 
  ∃! (n : ℕ), (∀ k, k <= n → Nat.choose n k = p → n = p) := 
begin
  existsi p,
  split,
  { intros k h_le h_eq,
    have h_k1 : k = 1 ∨ k = n - 1,
    { by_contradiction,
      -- Use properties of binomial coefficients and primes
      sorry },
    cases h_k1,
    { exact eq.symm h_k1 },
    { -- some similar argument for k = n - 1
      sorry  }
  },
  { intros n h_n,
    -- Showing that only n = p can satisfy the properties
    sorry }
end

end pascal_triangle_row_contains_prime_l328_328274


namespace projections_possibilities_l328_328936

-- Define the conditions: a and b are non-perpendicular skew lines, and α is a plane
variables {a b : Line} (α : Plane)

-- Non-perpendicular skew lines definition (external knowledge required for proper setup if not inbuilt)
def non_perpendicular_skew_lines (a b : Line) : Prop := sorry

-- Projections definition (external knowledge required for proper setup if not inbuilt)
def projections (a : Line) (α : Plane) : Line := sorry

-- The projections result in new conditions
def projected_parallel (a b : Line) (α : Plane) : Prop := sorry
def projected_perpendicular (a b : Line) (α : Plane) : Prop := sorry
def projected_same_line (a b : Line) (α : Plane) : Prop := sorry
def projected_line_and_point (a b : Line) (α : Plane) : Prop := sorry

-- Given the given conditions
variables (ha : non_perpendicular_skew_lines a b)

-- Prove the resultant conditions where the projections satisfy any 3 of the listed possibilities: parallel, perpendicular, line and point.
theorem projections_possibilities :
    (projected_parallel a b α ∨ projected_perpendicular a b α ∨ projected_line_and_point a b α) ∧
    ¬ projected_same_line a b α := sorry

end projections_possibilities_l328_328936


namespace angle_BPE_is_50_l328_328484

/-- Given an isosceles triangle ABC with AB = BC and ∠ABC = 60°, and points D and E such that ∠DCA = 30° and ∠EAC = 40°, if P is the intersection of DC and AE, then ∠BPE = 50°. -/
theorem angle_BPE_is_50 (A B C D E P : Point) -- Assume 'Point' is previously defined
  (isosceles_ABC : is_isosceles_triangle A B C)
  (AB_eq_BC : AB = BC)
  (angle_ABC_60 : ∠ B A C = 60)
  (D_on_AB : lies_on D A B)
  (angle_DCA_30 : ∠ D C A = 30)
  (E_on_BC : lies_on E B C)
  (angle_EAC_40 : ∠ E A C = 40)
  (P_intersection_DC_AE : intersection P D C A E)
  : ∠ B P E = 50 := sorry

end angle_BPE_is_50_l328_328484


namespace theater_total_bills_l328_328002

theorem theater_total_bills (tickets : ℕ) (price : ℕ) (x : ℕ) (number_of_5_bills : ℕ) (number_of_10_bills : ℕ) (number_of_20_bills : ℕ) :
  tickets = 300 →
  price = 40 →
  number_of_20_bills = x →
  number_of_10_bills = 2 * x →
  number_of_5_bills = 2 * x + 20 →
  20 * x + 10 * (2 * x) + 5 * (2 * x + 20) = tickets * price →
  number_of_5_bills + number_of_10_bills + number_of_20_bills = 1210 := by
    intro h_tickets h_price h_20_bills h_10_bills h_5_bills h_total
    sorry

end theater_total_bills_l328_328002


namespace solution_set_abs_le_one_inteval_l328_328433

theorem solution_set_abs_le_one_inteval (x : ℝ) : |x| ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

end solution_set_abs_le_one_inteval_l328_328433


namespace tricia_age_l328_328067

variable Tricia Amilia Yorick Eugene Khloe Rupert Vincent Selena : ℕ

-- Given conditions
axiom cond1 : Tricia = Amilia / 3
axiom cond2 : Amilia = Yorick / 4
axiom cond3 : Yorick = 2 * Eugene
axiom cond4 : Khloe = Eugene / 3
axiom cond5 : Rupert = Khloe + 10
axiom cond6 : Rupert = Vincent - 2
axiom cond7 : Vincent = 22
axiom cond8 : Yorick = Selena + 5
axiom cond9 : Selena = Amilia + 3

-- We need to prove that Tricia is 17 years old
theorem tricia_age : Tricia = 17 := 
  sorry

end tricia_age_l328_328067


namespace total_questions_reviewed_l328_328833

-- Defining the conditions
def questions_per_student : Nat := 10
def students_per_class : Nat := 35
def num_classes : Nat := 5

-- Define the total number of questions that Professor Oscar must review.
def total_questions : Nat := questions_per_student * students_per_class * num_classes

-- Theorem statement to be proved
theorem total_questions_reviewed : total_questions = 1750 := by
  sorry

end total_questions_reviewed_l328_328833


namespace zongzi_cost_prices_l328_328725

theorem zongzi_cost_prices (a : ℕ) (n : ℕ)
  (h1 : n * a = 8000)
  (h2 : n * (a - 10) = 6000)
  : a = 40 ∧ a - 10 = 30 :=
by
  sorry

end zongzi_cost_prices_l328_328725


namespace hotel_loss_l328_328118
  
  -- Conditions
  def operations_expenses : ℝ := 100
  def total_payments : ℝ := (3 / 4) * operations_expenses
  
  -- Theorem to prove
  theorem hotel_loss : operations_expenses - total_payments = 25 :=
  by
    sorry
  
end hotel_loss_l328_328118


namespace infinitely_many_m_n_l328_328006

theorem infinitely_many_m_n :
  ∃ᶠ (m n : ℕ) in at_top, (1 < m ∧ m < n) ∧
  (Nat.gcd m n > n.sqrt / 999) ∧
  (Nat.gcd m (n + 1) > n.sqrt / 999) ∧
  (Nat.gcd (m + 1) n > n.sqrt / 999) ∧
  (Nat.gcd (m + 1) (n + 1) > n.sqrt / 999) :=
sorry

end infinitely_many_m_n_l328_328006


namespace geometric_sequence_ratio_l328_328348

variable {a : ℕ → ℕ}
variables (S : ℕ → ℕ) (n : ℕ)
def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, ∀ n : ℕ, a n = a 1 * q^n

variables (h1 : a 5 - a 3 = 12) (h2 : a 6 - a 4 = 24)

theorem geometric_sequence_ratio :
  is_geometric_sequence a →
  ∃ S : ℕ → ℕ, (S n = ∑ i in range n, a i) ∧
  (S n / a n = 2 - 2^(1 - n)) :=
by
  sorry

end geometric_sequence_ratio_l328_328348


namespace projection_magnitude_correct_l328_328686

variables (a b : ℝ)
variables (a_vec b_vec : ℝ → ℝ → ℝ)

def vector_magnitude (v : ℝ → ℝ → ℝ) (norm : ℝ) : Prop :=
  |norm| = sqrt ((v a a_vec)^2 + (v b b_vec)^2)

noncomputable def dot_product (v w : ℝ → ℝ → ℝ) (dp : ℝ) : Prop :=
  dp = (v a a_vec) * (w a b_vec) + (v b a_vec) * (w b b_vec)

noncomputable def projection_magnitude (v w : ℝ → ℝ → ℝ) (proj : ℝ) : Prop :=
  proj = abs ((v a a_vec) * (w a b_vec) + (v b a_vec) * (w b b_vec)) / sqrt ((w a b_vec)^2 + (w b b_vec)^2)

theorem projection_magnitude_correct :
  vector_magnitude a_vec 5 →
  vector_magnitude b_vec 8 →
  dot_product a_vec b_vec 20 →
  projection_magnitude a_vec b_vec 2.5 :=
sorry

end projection_magnitude_correct_l328_328686


namespace exp_sum_is_neg_one_l328_328170

noncomputable def sumExpExpressions : ℂ :=
  (Complex.exp (Real.pi * Complex.I / 7) +
   Complex.exp (2 * Real.pi * Complex.I / 7) +
   Complex.exp (3 * Real.pi * Complex.I / 7) +
   Complex.exp (4 * Real.pi * Complex.I / 7) +
   Complex.exp (5 * Real.pi * Complex.I / 7) +
   Complex.exp (6 * Real.pi * Complex.I / 7) +
   Complex.exp (2 * Real.pi * Complex.I / 9) +
   Complex.exp (4 * Real.pi * Complex.I / 9) +
   Complex.exp (6 * Real.pi * Complex.I / 9) +
   Complex.exp (8 * Real.pi * Complex.I / 9) +
   Complex.exp (10 * Real.pi * Complex.I / 9) +
   Complex.exp (12 * Real.pi * Complex.I / 9) +
   Complex.exp (14 * Real.pi * Complex.I / 9) +
   Complex.exp (16 * Real.pi * Complex.I / 9))

theorem exp_sum_is_neg_one : sumExpExpressions = -1 := by
  sorry

end exp_sum_is_neg_one_l328_328170


namespace complex_division_proof_l328_328040

noncomputable def complex_div_eq : Prop := (3 + Complex.i) / (1 + Complex.i) = 2 - Complex.i

theorem complex_division_proof : complex_div_eq := by
  sorry

end complex_division_proof_l328_328040


namespace persimmons_in_Jungkook_house_l328_328440

-- Define the number of boxes and the number of persimmons per box
def num_boxes : ℕ := 4
def persimmons_per_box : ℕ := 5

-- Define the total number of persimmons calculation
def total_persimmons (boxes : ℕ) (per_box : ℕ) : ℕ := boxes * per_box

-- The main theorem statement proving the total number of persimmons
theorem persimmons_in_Jungkook_house : total_persimmons num_boxes persimmons_per_box = 20 := 
by 
  -- We should prove this, but we use 'sorry' to skip proof in this example.
  sorry

end persimmons_in_Jungkook_house_l328_328440


namespace sequence_count_l328_328216

theorem sequence_count :
  let s : list ℕ := [1, 1, 1, 2, 2, 3] in
  s.permutations.length = 60 :=
by
  sorry

end sequence_count_l328_328216


namespace expression_range_l328_328244

variables (a b c : EuclideanSpace ℝ (Fin 2))

-- Given conditions
axiom a_norm : ‖a‖ = 1
axiom b_norm : ‖b‖ = 2
axiom c_norm : ‖c‖ = 1
axiom a_dot_b : inner a b = 1

noncomputable def expression_value : ℝ :=
  ‖c + (a / 2)‖ + (1 / 2) * ‖c - b‖

theorem expression_range :
  3.sqrt ≤ expression_value a b c ∧ expression_value a b c ≤ 7.sqrt :=
sorry

end expression_range_l328_328244


namespace lim_sum_D_R_l328_328889

noncomputable def D_R (R : ℝ) : set (ℤ × ℤ) :=
  { p | 0 < p.1^2 + p.2^2 ∧ p.1^2 + p.2^2 < R }

theorem lim_sum_D_R :
  ∀ R : ℝ, R > 1 → (tendsto (λ R, ∑ p in D_R R, (↑(-1) ^ (p.1 + p.2) / (↑(p.1^2 + p.2^2) : ℝ)))
    at_top (𝓝 (-π * log 2))) :=
begin
  sorry
end

end lim_sum_D_R_l328_328889


namespace gcd_triples_l328_328184

theorem gcd_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  gcd a 20 = b ∧ gcd b 15 = c ∧ gcd a c = 5 ↔
  ∃ t : ℕ, t > 0 ∧ 
    ((a = 20 * t ∧ b = 20 ∧ c = 5) ∨ 
     (a = 20 * t - 10 ∧ b = 10 ∧ c = 5) ∨ 
     (a = 10 * t - 5 ∧ b = 5 ∧ c = 5)) :=
by
  sorry

end gcd_triples_l328_328184


namespace total_exercise_time_l328_328328

def distance(monday: Nat, monday_speed: Nat, wednesday: Nat, wednesday_speed: Nat, friday: Nat, friday_speed: Nat) : Nat :=
  (monday / monday_speed) + (wednesday / wednesday_speed) + (friday / friday_speed)

theorem total_exercise_time :
  distance 6 2 6 3 6 6 = 6 := by
  sorry

end total_exercise_time_l328_328328


namespace polygon_sides_l328_328295

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1440) : n = 10 := sorry

end polygon_sides_l328_328295


namespace arc_ratio_l328_328746

theorem arc_ratio (h₁ : ∀ x y : ℝ, x - sqrt 3 * y - 2 = 0 → (x - 1)^2 + y^2 = 1) :
  (arc_ratio_in_circle_divided_by_line ((x - 1)^2 + y^2 = 1) (x - sqrt 3 * y - 2 = 0)) = 1 / 2 := by
sory

end arc_ratio_l328_328746


namespace value_of_A_l328_328548

theorem value_of_A (G F L: ℤ) (H1 : G = 15) (H2 : F + L + 15 = 50) (H3 : F + L + 37 + 15 = 65) (H4 : F + ((58 - F - L) / 2) + ((58 - F - L) / 2) + L = 58) : 
  37 = 37 := 
by 
  sorry

end value_of_A_l328_328548


namespace smallest_number_formed_l328_328072

def odd_numbers_less_than_10 := {1, 3, 5, 7, 9}

theorem smallest_number_formed (s : Set ℕ) (h : s = odd_numbers_less_than_10) :
  (∀ (n : ℕ), (∀ m ∈ s, (m < 10 ∧ odd m)) → smallest_number_formed_by s = 13579) :=
by sorry

end smallest_number_formed_l328_328072


namespace polynomial_has_no_more_than_n_positive_roots_l328_328704

theorem polynomial_has_no_more_than_n_positive_roots 
  (n : ℕ) 
  (a : Fin n → ℝ) 
  (k : Fin n → ℕ) 
  (P : ℝ → ℝ := λ x, ∑ i in Finset.range n, a i * x^k i) :
  ∃ m ≤ n, (count_roots P > m) = false :=
  sorry

end polynomial_has_no_more_than_n_positive_roots_l328_328704


namespace man_l328_328146

noncomputable def man's_speed (length_train : ℝ) (speed_train_kmph : ℝ) (time_to_pass : ℝ) : ℝ :=
  let speed_train_mps := speed_train_kmph * 1000 / 3600 in
  let relative_speed_mps := length_train / time_to_pass in
  let relative_speed_kmph := relative_speed_mps * 3.6 in
  speed_train_kmph - relative_speed_kmph

theorem man's_speed_correct :
  man's_speed 150 62 9.99920006399488 ≈ 7.9997120057596 := 
by
  -- We skip the proof here using 'sorry'
  sorry

end man_l328_328146


namespace interval_of_monotonic_decrease_l328_328048

open set

-- Conditions
def y (x : ℝ) := log (2 - 2 * x)

-- The main statement to be proved
theorem interval_of_monotonic_decrease : ∀ x, y x = log (2 - 2 * x) → Ioo (-∞) 1 ⊆ {x | ∀ y, y x < y (x - x)} :=
sorry

end interval_of_monotonic_decrease_l328_328048


namespace parabola_area_l328_328959

theorem parabola_area 
    (Γ : ∀ x y : ℝ, y^2 = 8 * x)
    (K : (ℝ × ℝ)) (hK : K = (-2, 0))
    (F : (ℝ × ℝ)) (hF : F = (2, 0))
    (P : ℝ × ℝ)
    (hP : ∃ x y : ℝ, P = (x, y) ∧ y^2 = 8 * x) :
    (let PK := dist P K
     let PF := dist P F
     in PK = real.sqrt 2 * PF) →
    let FK := dist F K
    let y := (P.snd)
    in (1 / 2) * FK * abs y = 8 :=
sorry

end parabola_area_l328_328959


namespace bathtub_capacity_l328_328981

theorem bathtub_capacity
  (water_out_6_min : ℤ)
  (time_to_fill_min : ℤ)
  (drain_rate_per_min : ℤ) :
  (21 : ℤ) / 6 = water_out_6_min →
  (22 : ℤ) + 0.5 = time_to_fill_min →
  (0.3 : ℤ) = drain_rate_per_min →
  let tap_flow_rate := 3.5 in
  let fill_time := 22.5 in
  let total_leaked := drain_rate_per_min * fill_time in
  let total_tap_water := tap_flow_rate * fill_time in
  let bathtub_capacity := total_tap_water - total_leaked in
  bathtub_capacity = 72
by
  intros h_water_out h_fill_time h_drain_rate;
  sorry

end bathtub_capacity_l328_328981


namespace right_triangle_AB_length_l328_328299

theorem right_triangle_AB_length 
  (A B C : Type) [right_triangle A B C]
  (angle_A : ∠A = 90)
  (tan_B : tan B = 5 / 12)
  (AC : dist A C = 39) :
  dist A B = 15 :=
by
  sorry

end right_triangle_AB_length_l328_328299


namespace WXYZ_is_parallelogram_l328_328680

variables {A B C D W X Y Z : Point}
variables {T1 : Triangle A W Z} {T2 : Triangle B X W} {T3 : Triangle C Y X} {T4 : Triangle D Z Y}
variables (hABCD : Parallelogram A B C D)
variables (hIncenters : Parallelogram (incenter T1) (incenter T2) (incenter T3) (incenter T4))

theorem WXYZ_is_parallelogram (hW : W ∈ Line A B) (hX : X ∈ Line B C) (hY : Y ∈ Line C D) (hZ : Z ∈ Line D A) :
  Parallelogram W X Y Z := by
  sorry

end WXYZ_is_parallelogram_l328_328680


namespace trains_crossing_time_l328_328069

noncomputable def time_to_cross_opposite_directions (L : ℝ) : ℝ :=
  2 * L / (100 * 5 / 18)

theorem trains_crossing_time :
  ∀ (L : ℝ),
    L = (55 * (20 * 5 / 18)) / 2 →
    time_to_cross_opposite_directions L ≈ 11.01 :=
by
  intros,
  sorry

end trains_crossing_time_l328_328069


namespace min_cells_to_remove_tiling_202x202_l328_328107

def T_tetromino : Type := -- Definition of T-tetromino, here we are assuming it's implicitly defined/modeled

def is_tiled_with_T_tetrominoes (grid : list (list ℕ)) : Prop :=
  -- Predicate stating if the grid is tiled with T-tetrominoes
  sorry

def cells_to_remove (grid : list (list ℕ)) : ℕ :=
  -- Function to determine number of cells needed to remove
  sorry

theorem min_cells_to_remove_tiling_202x202 :
  let grid := list.replicate 202 (list.replicate 202 1) in
  cells_to_remove grid = 4 :=
by sorry

end min_cells_to_remove_tiling_202x202_l328_328107


namespace rectangle_area_is_140_l328_328049

def side_of_square (area : ℕ) : ℕ := Nat.sqrt area

def radius_of_circle (side : ℕ) : ℕ := side

def length_of_rectangle (radius : ℕ) : ℕ := (2 * radius) / 5

def area_of_rectangle (length breadth : ℕ) : ℕ := length * breadth

theorem rectangle_area_is_140 (breadth : ℕ) 
  (area_sq : ℕ) (H_area : area_sq = 1225) 
  (H_breath : breadth = 10) : 
  let side_sq := side_of_square area_sq
      radius := radius_of_circle side_sq
      length_rect := length_of_rectangle radius
  in area_of_rectangle length_rect breadth = 140 := by
  sorry

end rectangle_area_is_140_l328_328049


namespace bc_length_l328_328007

-- Definitions based on conditions
def H : ℝ × ℝ := (-11, 5)
def O : ℝ × ℝ := (0, 5)
def M : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-11, 0)
def HO := 11
def OM := 5

-- Showing that the length of BC is 28 given the conditions
theorem bc_length (H : ℝ × ℝ) (O : ℝ × ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) (HO : ℝ) (OM : ℝ) :
  H = (-11, 5) → 
  O = (0, 5) → 
  M = (0, 0) → 
  F = (-11, 0) → 
  HO = 11 → 
  OM = 5 → 
  let e := 14 in
  2 * e = 28 :=
by
  intros h_eq o_eq m_eq f_eq ho_eq om_eq e,
  simp [e],
  linarith

end bc_length_l328_328007


namespace option_A_option_B_option_C_option_D_l328_328258

-- Definitions related to the parabola and the geometric setup
def parabola (p : ℝ) : set (ℝ × ℝ) := {xy | xy.2 ^ 2 = 2 * p * xy.1}
def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)
def directrix (p : ℝ) : set (ℝ × ℝ) := {xy | xy.1 = -p / 2}

-- Points A and B on the parabola
variables {A B M : ℝ × ℝ}
variables {p : ℝ} (hp : 0 < p)

-- Conditions and Propositions
axiom point_on_parabola (hA : A ∈ parabola p) (hB : B ∈ parabola p) : true

-- Prove option A
theorem option_A (hAM : dist A M = dist A (focus p)) (hMF : dist M (focus p) = M.1 + p / 2): 
  perp (A - M) (directrix p) := sorry

-- Prove option B
theorem option_B (hAM : dist A M = dist A (focus p)) (hMF : dist M (focus p) = dist A (focus p)): 
  ¬ (dist (focus p) B = 2 * dist (focus p) B) := sorry

-- Prove option C
theorem option_C (hMA : perp (A - M) (B - M)) (hA : A ∈ parabola p) (hB : B ∈ parabola p): 
  let y_coords := [A.2, M.2, B.2] in 
  ∃ d, y_coords = [A.2, A.2 + d, A.2 + 2*d] := sorry

-- Prove option D
theorem option_D (hMA : perp (A - M) (B - M)) (hA : A ∈ parabola p) (hB : B ∈ parabola p):
  (dist A M) * (dist B M) ≥ 2 * (dist A (focus p)) * (dist B (focus p)) := sorry

end option_A_option_B_option_C_option_D_l328_328258


namespace bids_per_person_l328_328161

theorem bids_per_person (initial_price final_price price_increase_per_bid : ℕ) (num_people : ℕ)
  (h1 : initial_price = 15) (h2 : final_price = 65) (h3 : price_increase_per_bid = 5) (h4 : num_people = 2) :
  (final_price - initial_price) / price_increase_per_bid / num_people = 5 :=
  sorry

end bids_per_person_l328_328161


namespace surface_area_pyramid_correct_l328_328668

-- Definitions based on the conditions
structure Triangle :=
(A B C : Point) -- Points in the plane forming triangle ABC

structure Pyramid :=
(A B C D : Point) -- Points such that D is outside the plane of triangle ABC

def edge_lengths_valid (P : Pyramid) (lengths : Set ℝ) : Prop :=
  -- every edge length of the pyramid is either 25 or 34
  (dist P.A P.B ∈ lengths) ∧ (dist P.B P.C ∈ lengths) ∧ 
  (dist P.C P.A ∈ lengths) ∧ (dist P.A P.D ∈ lengths) ∧ 
  (dist P.B P.D ∈ lengths) ∧ (dist P.C P.D ∈ lengths)

def no_face_equilateral (P : Pyramid) : Prop :=
  -- no face of the pyramid is equilateral
  ¬(dist P.A P.B = dist P.B P.C ∧ dist P.B P.C = dist P.C P.A) ∧
  ¬(dist P.A P.B = dist P.B P.D ∧ dist P.B P.D = dist P.D P.A) ∧
  ¬(dist P.A P.C = dist P.C P.D ∧ dist P.C P.D = dist P.D P.A) ∧
  ¬(dist P.B P.C = dist P.C P.D ∧ dist P.C P.D = dist P.D P.B)

def surface_area (P : Pyramid) : ℝ :=
  -- this is where the surface area function would be defined.
  sorry

-- Formulate the theorem
theorem surface_area_pyramid_correct (P : Pyramid) (lengths : Set ℝ) :
  edge_lengths_valid P lengths ∧ no_face_equilateral P → 
  surface_area P = 17 * sqrt 336 + 37.5 * sqrt 999.75 :=
by
  -- the assumption is that the lengths set is {25, 34}
  assume h : edge_lengths_valid P {25, 34} ∧ no_face_equilateral P
  sorry

end surface_area_pyramid_correct_l328_328668


namespace correct_statement_among_given_l328_328083

theorem correct_statement_among_given (
  (cond_A : ∀ Q : Type, ∀ (q : quadrilateral Q), (one_pair_parallel_sides q → one_pair_equal_sides q → parallelogram q)),
  (cond_B : ∀ P : Type, ∀ (p : parallelogram P), (complementary_diagonals p)),
  (cond_C : ∀ Q : Type, ∀ (q : quadrilateral Q), (two_pairs_equal_angles q → parallelogram q)),
  (cond_D : ∀ P : Type, ∀ (p : parallelogram P), (diagonals_bisect_opposite_angles p))
) : 
  ∃ S, S = cond_C := sorry

end correct_statement_among_given_l328_328083


namespace tangent_to_circle_l328_328813

theorem tangent_to_circle
  (O : Type) [metric_space O] [normed_group O] [normed_space ℝ O]
  (P B C Q D A : O)
  (PB PC PQ PD BD AD CP : ℝ)
  (h1 : PB < PC)
  (h2 : PQ < PD)
  (h3 : BD^2 = AD * CP)
  (h_circle_O : is_circle O)
  (h_through_P : line_through P [B, C])
  (h_through_PO : line_through PO [Q, D])
  (h_perpendicular : is_perpendicular (line_through Q [B, C]) (line_through Q [A]))
  : is_tangent PA O :=
  sorry

end tangent_to_circle_l328_328813


namespace alexander_pencils_total_l328_328152

variables (num_initial_galleries : ℕ)
          (num_initial_pictures : ℕ)
          (num_new_galleries : ℕ)
          (pictures_per_new_gallery : ℕ)
          (pencils_per_picture : ℕ)
          (pencils_per_exhibition : ℕ)

-- Conditions from the problem statement
def alexander_total_pencils 
    (num_initial_galleries : ℕ)
    (num_initial_pictures : ℕ)
    (num_new_galleries : ℕ)
    (pictures_per_new_gallery : ℕ)
    (pencils_per_picture : ℕ)
    (pencils_per_exhibition : ℕ) : ℕ :=
  let num_initial_exhibition := num_initial_galleries * num_initial_pictures in
  let num_new_exhibition := num_new_galleries * pictures_per_new_gallery in
  let total_pictures := num_initial_pictures + num_new_exhibition in
  let pencils_for_drawing := total_pictures * pencils_per_picture in
  let total_exhibitions := num_initial_galleries + num_new_galleries in
  let pencils_for_signing := total_exhibitions * pencils_per_exhibition in
  pencils_for_drawing + pencils_for_signing

theorem alexander_pencils_total 
  : alexander_total_pencils 1 9 5 2 4 2 = 88 := by
  sorry

end alexander_pencils_total_l328_328152


namespace problem_l328_328654

noncomputable def LM_length (A B C K L M : ℝ) (angle_B angle_A : ℝ)
    (AK BL MC : ℝ) (KL_eq_KM : KL = KM) (L_on_BM : L ∈ Segment B M) : Prop :=
  (LM = λ _ => 14)

-- Define the conditions
def conditions (A B C K L M : Point)
  (angle_B_eq : angle B = 30) (angle_A_eq : angle A = 90)
  (AK_eq : segment_length A K = 4) (BL_eq : segment_length B L = 31)
  (MC_eq : segment_length M C = 3) (KL_eq_KM : segment_length K L = segment_length K M)
  (L_on_BM : L ∈ Segment B M) : Prop :=
  angle_B_eq ∧ angle_A_eq ∧ AK_eq ∧ BL_eq ∧ MC_eq ∧ KL_eq_KM ∧ L_on_BM

-- The proof goal
theorem problem (A B C K L M : Point)
  (angle_B_eq : angle B = 30) (angle_A_eq : angle A = 90)
  (AK_eq : segment_length A K = 4) (BL_eq : segment_length B L = 31)
  (MC_eq : segment_length M C = 3) (KL_eq_KM : segment_length K L = segment_length K M)
  (L_on_BM : L ∈ Segment B M) : LM_length A B C K L M 30 90 4 31 3 KL_eq_KM L_on_BM :=
by
  apply LM
  sorry

end problem_l328_328654


namespace maximize_sum_of_arithmetic_sequence_l328_328605

theorem maximize_sum_of_arithmetic_sequence (a : ℕ → ℤ) (h : ∀ n, a n = 26 - 2 * n) :
  ∃ n, n = 12 ∨ n = 13 ∧ (∑ i in range n, a i) = max (∑ i in range 12, a i) (∑ i in range 13, a i) := by
  sorry

end maximize_sum_of_arithmetic_sequence_l328_328605


namespace percentage_of_life_in_accounting_jobs_l328_328460

-- Define the conditions
def years_as_accountant : ℕ := 25
def years_as_manager : ℕ := 15
def lifespan : ℕ := 80

-- Define the proof problem statement
theorem percentage_of_life_in_accounting_jobs :
  (years_as_accountant + years_as_manager) / lifespan * 100 = 50 := 
by sorry

end percentage_of_life_in_accounting_jobs_l328_328460


namespace aaron_jogging_speed_l328_328830

theorem aaron_jogging_speed :
  ∃ v : ℝ, v = 4 / 3 ∧ ((3 / v) + (3 / 4) = 3) :=
begin
  use 4 / 3,
  split,
  { refl },
  { field_simp,
    linarith },
end

end aaron_jogging_speed_l328_328830


namespace min_value_of_x1_x2_squared_l328_328100

open Real

theorem min_value_of_x1_x2_squared :
  ∀ (m : ℝ), m ≤ -3 / 2 →
  let x1 := (2 * m + sqrt ((-8) * m - 12)) / 2,
      x2 := (2 * m - sqrt ((-8) * m - 12)) / 2 in
  x1^2 + x2^2 = 9 / 2 :=
by
  intros m hm
  let x1 := (2 * m + sqrt ((-8) * m - 12)) / 2
  let x2 := (2 * m - sqrt ((-8) * m - 12)) / 2
  calc
    x1^2 + x2^2 = (2 * m + sqrt ((-8) * m - 12)) ^ 2 / 4 + (2 * m - sqrt ((-8) * m - 12)) ^ 2 / 4 : sorry
                  ... = (4 * m^2 + 4 * m * sqrt ((-8) * m - 12) + (-8) * m - 12) / 4 + (4 * m^2 - 4 * m * sqrt ((-8) * m - 12) + (-8) * m - 12) / 4 : sorry
                  ... = 9 / 2 : sorry

end min_value_of_x1_x2_squared_l328_328100


namespace no_prime_solution_l328_328005

open Nat

theorem no_prime_solution (p q r s t : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s) (ht : Prime t) : 
  p^2 + q^2 ≠ r^2 + s^2 + t^2 := 
by sorry

end no_prime_solution_l328_328005


namespace total_bowling_balls_is_66_l328_328037

-- Define the given conditions
def red_bowling_balls := 30
def difference_green_red := 6
def green_bowling_balls := red_bowling_balls + difference_green_red

-- The statement to prove
theorem total_bowling_balls_is_66 :
  red_bowling_balls + green_bowling_balls = 66 := by
  sorry

end total_bowling_balls_is_66_l328_328037


namespace relay_orders_l328_328642

-- Definitions based on the conditions
def num_team_members : ℕ := 5
def jordan_last_lap : bool := true

-- Theorem statement
theorem relay_orders (h1 : num_team_members = 5) (h2 : jordan_last_lap = true) : 
  ∃ orders : ℕ, orders = 24 := by
  sorry

end relay_orders_l328_328642


namespace six_digit_numbers_without_repetition_count_sum_of_six_digit_numbers_without_repetition_l328_328762

open Finset

-- Define statements for the problems.

theorem six_digit_numbers_without_repetition_count :
  (∑ n in (finset.perm (finset.range 6)).filter (λ l, l.head ≠ 0), 1) = 600 := sorry

theorem sum_of_six_digit_numbers_without_repetition :
  (∑ n in (finset.perm (finset.range 6)).filter (λ l, l.head ≠ 0), n.foldl (λ acc d, 10 * acc + d) 0) = 19000000 := sorry

end six_digit_numbers_without_repetition_count_sum_of_six_digit_numbers_without_repetition_l328_328762


namespace right_triangle_hypotenuse_l328_328029

theorem right_triangle_hypotenuse (A : ℝ) (h height : ℝ) :
  A = 320 ∧ height = 16 →
  ∃ c : ℝ, c = 4 * Real.sqrt 116 :=
by
  intro h
  sorry

end right_triangle_hypotenuse_l328_328029


namespace wait_time_at_construction_site_l328_328521

def school_start : Nat := 8 * 60 -- School starts at 8:00 AM
def normal_travel_time : Nat := 30 -- 30 minutes normally to school
def red_lights : Nat := 4 -- 4 red lights
def red_light_delay : Nat := 3 -- 3 minutes each red light
def total_red_light_delay : Nat := red_lights * red_light_delay -- Total red light delay in minutes
def departure_time : Nat := 7 * 60 + 15 -- 7:15 AM in minutes
def late_minutes : Nat := 7 -- Andy was 7 minutes late

-- Total time to reach school
def total_time_to_school (extra_delay : Nat) : Nat :=
  normal_travel_time + total_red_light_delay + extra_delay

-- Expected arrival time without extra delay
def expected_arrival_without_extra_delay : Nat :=
  departure_time + normal_travel_time + total_red_light_delay

-- Actual arrival time
def actual_arrival_time : Nat :=
  school_start + late_minutes

-- Time waited at construction site
def construction_delay : Nat := actual_arrival_time - expected_arrival_without_extra_delay

theorem wait_time_at_construction_site : construction_delay = 10 := by
  calc
    construction_delay = actual_arrival_time - expected_arrival_without_extra_delay : by rfl
    ... = (school_start + late_minutes) - (departure_time + normal_travel_time + total_red_light_delay) : by rfl
    ... = ((8 * 60) + 7) - ((7 * 60 + 15) + 30 + (4 * 3)) : by rfl
    ... = (480 + 7) - (435 + 30 + 12)                          : by rfl
    ... = 487 - 477                                           : by rfl
    ... = 10                                                  : by rfl

end wait_time_at_construction_site_l328_328521


namespace remaining_insects_is_twenty_one_l328_328712

-- Define the initial counts of each type of insect
def spiders := 3
def ants := 12
def ladybugs := 8

-- Define the number of ladybugs that flew away
def ladybugs_flew_away := 2

-- Define the total initial number of insects
def total_insects_initial := spiders + ants + ladybugs

-- Define the total number of insects that remain after some ladybugs fly away
def total_insects_remaining := total_insects_initial - ladybugs_flew_away

-- Theorem statement: proving that the number of insects remaining is 21
theorem remaining_insects_is_twenty_one : total_insects_remaining = 21 := sorry

end remaining_insects_is_twenty_one_l328_328712


namespace total_bowling_balls_is_66_l328_328039

-- Define the given conditions
def red_bowling_balls := 30
def difference_green_red := 6
def green_bowling_balls := red_bowling_balls + difference_green_red

-- The statement to prove
theorem total_bowling_balls_is_66 :
  red_bowling_balls + green_bowling_balls = 66 := by
  sorry

end total_bowling_balls_is_66_l328_328039


namespace perpendicular_lines_a_plus_b_eq_4_l328_328633

theorem perpendicular_lines_a_plus_b_eq_4 (a b : ℝ) :
  (∃ x y : ℝ, x + (a - 4) * y + 1 = 0 ∧ bx + y - 2 = 0 ∧
  let slope1 := -1 / (a - 4) in let slope2 := -b in slope1 * slope2 = -1) → a + b = 4 :=
by
  sorry

end perpendicular_lines_a_plus_b_eq_4_l328_328633


namespace rob_final_value_in_euros_l328_328008

noncomputable def initial_value_in_usd : ℝ := 
  (7 * 0.25) + (3 * 0.10) + (5 * 0.05) + (12 * 0.01) + (3 * 0.50) + (2 * 1.00)

noncomputable def value_after_losing_coins : ℝ := 
  (6 * 0.25) + (2 * 0.10) + (4 * 0.05) + (11 * 0.01) + (2 * 0.50) + (1 * 1.00)

noncomputable def value_after_first_exchange : ℝ :=
  (6 * 0.25) + (4 * 0.10) + (1 * 0.05) + (11 * 0.01) + (2 * 0.50) + (1 * 1.00)

noncomputable def value_after_second_exchange : ℝ :=
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (11 * 0.01) + (1 * 0.50) + (1 * 1.00)

noncomputable def value_after_third_exchange : ℝ :=
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (61 * 0.01) + (1 * 0.50)

noncomputable def final_value_in_usd : ℝ := 
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (61 * 0.01) + (1 * 0.50)

noncomputable def exchange_rate_usd_to_eur : ℝ := 0.85

noncomputable def final_value_in_eur : ℝ :=
  final_value_in_usd * exchange_rate_usd_to_eur

theorem rob_final_value_in_euros : final_value_in_eur = 2.9835 := by
  sorry

end rob_final_value_in_euros_l328_328008


namespace total_questions_reviewed_l328_328836

theorem total_questions_reviewed (questions_per_student : ℕ) (students_per_class : ℕ) (number_of_classes : ℕ) :
  questions_per_student = 10 → students_per_class = 35 → number_of_classes = 5 →
  questions_per_student * students_per_class * number_of_classes = 1750 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  norm_num
  sorry

end total_questions_reviewed_l328_328836


namespace cot_neg_45_l328_328879

theorem cot_neg_45 (cot_def : ∀ θ : ℝ, Real.cot θ = 1 / Real.tan θ)
  (tan_neg : ∀ θ : ℝ, Real.tan (-θ) = -Real.tan θ)
  (tan_45 : Real.tan (Real.pi / 4) = 1) :
  Real.cot (-Real.pi / 4) = -1 :=
by
  -- proof goes here
  sorry

end cot_neg_45_l328_328879


namespace log_exp_calculation_l328_328530

theorem log_exp_calculation :
  ( (log 32 / log 10 - log 4 / log 10) / (log 2 / log 10) + 27^(2 / 3) ) = 12 :=
by
  sorry

end log_exp_calculation_l328_328530


namespace sum_harmonic_numbers_2023_l328_328992

def is_harmonic (n : ℕ) : Prop :=
  ∃ k : ℕ, (k % 2 = 1) ∧ n = (k + 2)^2 - k^2 ∧ n % 8 = 0

def sum_of_harmonic_numbers_up_to (N : ℕ) : ℕ :=
  ∑ n in Finset.filter (λ n, is_harmonic n ∧ n ≤ N) (Finset.range (N + 1)), n

theorem sum_harmonic_numbers_2023 : sum_of_harmonic_numbers_up_to 2023 = 255024 := 
by {
  sorry
}

end sum_harmonic_numbers_2023_l328_328992


namespace inequality_correct_l328_328280

variable {a b : ℝ}

theorem inequality_correct (h₁ : a < 1) (h₂ : b > 1) : ab < a + b :=
sorry

end inequality_correct_l328_328280


namespace joan_gave_sam_43_seashells_l328_328326

def joan_original_seashells : ℕ := 70
def joan_seashells_left : ℕ := 27
def seashells_given_to_sam : ℕ := 43

theorem joan_gave_sam_43_seashells :
  joan_original_seashells - joan_seashells_left = seashells_given_to_sam :=
by
  sorry

end joan_gave_sam_43_seashells_l328_328326


namespace range_of_a_l328_328930

noncomputable def f (x : ℝ) : ℝ := -x^2 - 2 * x + 4
noncomputable def g (a x : ℝ) : ℝ := log a x

theorem range_of_a (a : ℝ) 
  (h₀ : 0 < a ∧ a ≠ 1)
  (h₁ : ∀ x₂ ∈ set.Icc 3 5, ∃ x₁ ∈ set.Icc (-(3:ℝ)/2) 1, f x₁ < g a x₂) :
  1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l328_328930


namespace find_value_of_a_l328_328207

theorem find_value_of_a (n : ℕ) (a : ℕ) : 
  ((∑ i in finset.range (5 * n), 2 ^ (i + 1)) + a) % 31 = 3 → a = 4 :=
sorry

end find_value_of_a_l328_328207


namespace total_exercise_time_l328_328327

def distance(monday: Nat, monday_speed: Nat, wednesday: Nat, wednesday_speed: Nat, friday: Nat, friday_speed: Nat) : Nat :=
  (monday / monday_speed) + (wednesday / wednesday_speed) + (friday / friday_speed)

theorem total_exercise_time :
  distance 6 2 6 3 6 6 = 6 := by
  sorry

end total_exercise_time_l328_328327


namespace negation_prop_p_l328_328375

def prop_p (x : ℝ) : Prop := x > 2 → 2 ^ x - 3 > 0

theorem negation_prop_p : (¬ (∀ x > 2, 2 ^ x - 3 > 0)) ↔ (∃ x₀ > 2, 2 ^ x₀ - 3 ≤ 0) :=
by
  sorry

end negation_prop_p_l328_328375


namespace complement_of_angle_l328_328278

theorem complement_of_angle (α : ℝ) (h : α = 23 + 36 / 60) : 180 - α = 156.4 := 
by
  sorry

end complement_of_angle_l328_328278


namespace money_left_correct_l328_328474

-- Define the initial amount of money John had
def initial_money : ℝ := 10.50

-- Define the amount spent on sweets
def sweets_cost : ℝ := 2.25

-- Define the amount John gave to each friend
def gift_per_friend : ℝ := 2.20

-- Define the total number of friends
def number_of_friends : ℕ := 2

-- Calculate the total gifts given to friends
def total_gifts := gift_per_friend * (number_of_friends : ℝ)

-- Calculate the total amount spent
def total_spent := sweets_cost + total_gifts

-- Define the amount of money left
def money_left := initial_money - total_spent

-- The theorem statement
theorem money_left_correct : money_left = 3.85 := 
by 
  sorry

end money_left_correct_l328_328474


namespace number_of_factors_m_l328_328679

noncomputable def m : ℕ := 2^5 * 3^4 * 4^5 * 6^6

theorem number_of_factors_m : (finset.range (21 + 1)).card * (finset.range (10 + 1)).card = 242 :=
by
  sorry

end number_of_factors_m_l328_328679


namespace combined_weight_l328_328624

theorem combined_weight (S R : ℕ) (h1 : S = 71) (h2 : S - 5 = 2 * R) : S + R = 104 := by
  sorry

end combined_weight_l328_328624


namespace t_50_mod_6_l328_328739

def t : ℕ → ℕ
| 0       := 3
| (n+1)   := 3 ^ (t n)

theorem t_50_mod_6 : t 50 % 6 = 3 := 
sorry

end t_50_mod_6_l328_328739


namespace correct_statement_among_given_l328_328084

theorem correct_statement_among_given (
  (cond_A : ∀ Q : Type, ∀ (q : quadrilateral Q), (one_pair_parallel_sides q → one_pair_equal_sides q → parallelogram q)),
  (cond_B : ∀ P : Type, ∀ (p : parallelogram P), (complementary_diagonals p)),
  (cond_C : ∀ Q : Type, ∀ (q : quadrilateral Q), (two_pairs_equal_angles q → parallelogram q)),
  (cond_D : ∀ P : Type, ∀ (p : parallelogram P), (diagonals_bisect_opposite_angles p))
) : 
  ∃ S, S = cond_C := sorry

end correct_statement_among_given_l328_328084


namespace elena_bouquet_petals_l328_328872

def num_petals (count : ℕ) (petals_per_flower : ℕ) : ℕ :=
  count * petals_per_flower

theorem elena_bouquet_petals :
  let num_lilies := 4
  let lilies_petal_count := num_petals num_lilies 6
  
  let num_tulips := 2
  let tulips_petal_count := num_petals num_tulips 3

  let num_roses := 2
  let roses_petal_count := num_petals num_roses 5
  
  let num_daisies := 1
  let daisies_petal_count := num_petals num_daisies 12
  
  lilies_petal_count + tulips_petal_count + roses_petal_count + daisies_petal_count = 52 := by
  sorry

end elena_bouquet_petals_l328_328872


namespace log_x3y2_eq_2_l328_328279

theorem log_x3y2_eq_2 (x y : ℝ) (h1 : log (x^2 * y^5) = 2) (h2 : log (x^3 * y^2) = 2) :
  log (x^3 * y^2) = 2 :=
sorry

end log_x3y2_eq_2_l328_328279


namespace find_y_in_exponent_equation_l328_328865

theorem find_y_in_exponent_equation :
  ∃ y : ℕ, 8^3 + 8^3 + 8^3 + 8^3 = 2^y ∧ y = 11 :=
begin
  use 11,
  split,
  { have h1 : 8 = 2^3 := by norm_num,
    have h2 : 8^3 = (2^3)^3 := by congr,
    have h3 : (2^3)^3 = 2^(3 * 3) := by rw [←pow_mul],
    rw [h2, h3, pow_mul],
    norm_num,
  },
  { refl },
end

end find_y_in_exponent_equation_l328_328865


namespace quadratic_inequality_solution_l328_328022

theorem quadratic_inequality_solution :
  ∀ (x : ℝ), -3 * x^2 + 5 * x + 4 < 0 ↔ x ∈ set.Ioo (-4 / 3) 1 :=
by
  sorry

end quadratic_inequality_solution_l328_328022


namespace number_of_quintuplets_l328_328846

variables (x y z w : ℝ)
variables (total_babies : ℝ)

noncomputable def condition1 := (y = 3 * z)
noncomputable def condition2 := (x = 2 * y)
noncomputable def condition3 := (w = (1/2) * z)
noncomputable def total_condition := (2 * x + 3 * y + 4 * z + 5 * w = 1500)

theorem number_of_quintuplets (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : total_condition) : 
  5 * w = 147 :=
by
  sorry

end number_of_quintuplets_l328_328846


namespace dobrynya_killed_zmey_l328_328319

-- Define the statements of the three heroes
def ilya_statements := ["I did not kill Zmey Gorynych", "I traveled to foreign lands", "Zmey Gorynych was killed by Alyosha Popovich"]
def dobrynya_statements := ["Zmey Gorynych was killed by Alyosha Popovich", "If I had killed him, I would not have admitted it", "There is still a lot of evil left"]
def alyosha_statements := ["I did not kill Zmey Gorynych", "I have long been looking for a heroic deed to perform", "Ilya Muromets indeed traveled to foreign lands"]

-- Define the possible killers
def killer := ["Ilya Muromets", "Dobrynya Nikitich", "Alyosha Popovich"]

-- The proof goal is to show that given conditions, Dobrynya Nikitich is the one who killed Zmey Gorynych
theorem dobrynya_killed_zmey : 
  (∀ (n : ℕ), (0 ≤ n ∧ n < 3) → 
    (ilya_statements n ≠ dobrynya_statements n ∨ ilya_statements n ≠ alyosha_statements n) ∧ 
    (dobrynya_statements n = "Zmey Gorynych was killed by Alyosha Popovich" → false) → 
    (∃ k, k < 3 ∧ killer k = "Dobrynya Nikitich")) := 
by 
  sorry

end dobrynya_killed_zmey_l328_328319


namespace count_valid_pairs_l328_328703

-- Define the given parameters and conditions
variables (X : Set) (n : ℕ)
hypothesis (h : |X|=n)

-- Define the pairs (A, B) where A ⊆ B ⊆ X
def valid_pairs (A B : Set) : Prop := A ⊆ B ∧ B ⊆ X

-- Statement: Prove that the number of such pairs is 3^n - 2^n
theorem count_valid_pairs : ∃ N, N = (3^n - 2^n) :=
by sorry

end count_valid_pairs_l328_328703


namespace sort_three_numbers_l328_328538

theorem sort_three_numbers (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  let (a', b') := if a < b then (b, a) else (a, b) in
  let (a'', c') := if a' < c then (c, a') else (a', c) in
  let (b'', c'') := if b' < c' then (c', b') else (b', c') in
  (a'' ≥ b'' ∧ b'' ≥ c'') :=
by
  sorry

end sort_three_numbers_l328_328538


namespace tan_half_angle_identity_l328_328283

variable {a b : ℝ}

theorem tan_half_angle_identity (h : 6 * (cos a + cos b) + 3 * (sin a + sin b) + 5 * (cos a * cos b + 1) = 0) :
  ∃ t : ℝ, (t = 5 ∨ t = -5) ∧ t = (tan (a / 2)) * (tan (b / 2)) := by 
  sorry

end tan_half_angle_identity_l328_328283


namespace volume_of_rectangular_prism_l328_328505

-- Define the conditions
def side_of_square : ℕ := 35
def area_of_square : ℕ := 1225
def radius_of_sphere : ℕ := side_of_square
def length_of_prism : ℕ := (2 * radius_of_sphere) / 5
def width_of_prism : ℕ := 10
variable (h : ℕ) -- height of the prism

-- The theorem to prove
theorem volume_of_rectangular_prism :
  area_of_square = side_of_square * side_of_square →
  length_of_prism = (2 * radius_of_sphere) / 5 →
  radius_of_sphere = side_of_square →
  volume_of_prism = (length_of_prism * width_of_prism * h)
  → volume_of_prism = 140 * h :=
by sorry

end volume_of_rectangular_prism_l328_328505


namespace wool_used_for_sweater_correct_l328_328873

def wool_for_scarf := 3
def aaron_scarves := 10
def aaron_sweaters := 5
def enid_sweaters := 8
def total_wool_used := 82

noncomputable def wool_for_sweater : ℕ :=
  if (aaron_scarves * wool_for_scarf + aaron_sweaters * x + enid_sweaters * x = total_wool_used) 
  then x
  else 0

theorem wool_used_for_sweater_correct : wool_for_sweater = 4 :=
by
  -- Definition and proof go here
  sorry

end wool_used_for_sweater_correct_l328_328873


namespace trains_cross_time_l328_328093

noncomputable def timeToCrossEachOther (L : ℝ) (T1 : ℝ) (T2 : ℝ) : ℝ :=
  let V1 := L / T1
  let V2 := L / T2
  let Vr := V1 + V2
  let totalDistance := L + L
  totalDistance / Vr

theorem trains_cross_time (L T1 T2 : ℝ) (hL : L = 120) (hT1 : T1 = 10) (hT2 : T2 = 15) :
  timeToCrossEachOther L T1 T2 = 12 :=
by
  simp [timeToCrossEachOther, hL, hT1, hT2]
  sorry

end trains_cross_time_l328_328093


namespace equation_of_parallel_line_l328_328414

theorem equation_of_parallel_line (c : ℕ) :
  (∃ c, x + 2 * y + c = 0) ∧ (1 + 2 * 1 + c = 0) -> x + 2 * y - 3 = 0 :=
by 
  sorry

end equation_of_parallel_line_l328_328414


namespace total_bowling_balls_l328_328036

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l328_328036


namespace total_spending_l328_328874

-- Define the conditions
def notebooks_price : ℕ := 4 * 2
def magazines_price : ℕ := 3 * 6
def pens_price : ℝ := 5 * (1.50 * 0.75)
def books_price : ℕ := 2 * 12
def total_cost_before_discount : ℝ := notebooks_price + magazines_price + pens_price + books_price
def membership_discount : ℝ := if total_cost_before_discount ≥ 50 then 10 else 0
def final_cost : ℝ := total_cost_before_discount - membership_discount

-- Statement of the problem
theorem total_spending :
  final_cost = 45.625 := by
  sorry

end total_spending_l328_328874


namespace tan_neg_five_pi_over_four_l328_328167

theorem tan_neg_five_pi_over_four : Real.tan (-5 * Real.pi / 4) = -1 :=
  sorry

end tan_neg_five_pi_over_four_l328_328167


namespace total_money_given_l328_328137

noncomputable def total_amount (a b c d : ℝ) (T : ℝ) : Prop :=
  (a / b = 5 / 9) ∧ (b / c = 9 / 6) ∧ (c / d = 6 / 5) ∧ (a + c = 7022.222222222222) ∧ (T = a + b + c + d)

theorem total_money_given (a b c d T : ℝ) (x : ℝ)
  (h1 : a = 5 * x)
  (h2 : b = 9 * x)
  (h3 : c = 6 * x)
  (h4 : d = 5 * x)
  (h5 : a + c = 7022.222222222222) :
  T = 15959.60 :=
begin
  have h_total : T = a + b + c + d, by sorry,
  have h_combined : a + c = 11 * x, by sorry,
  have h_x : x = 7022.222222222222 / 11, by sorry,
  have h_T : T = 25 * x, by sorry,
  rw h_x at h_T,
  norm_num at h_T,
  exact h_T,
end

end total_money_given_l328_328137


namespace vector_problem_l328_328268

noncomputable def a : ℝ × ℝ × ℝ := (1, 2, real.sqrt 3)
noncomputable def b : ℝ × ℝ × ℝ := (-1, real.sqrt 3, 0)

noncomputable def dot_product (v₁ v₂ : ℝ × ℝ × ℝ) : ℝ :=
v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

theorem vector_problem :
  dot_product a b + magnitude b = 2 * real.sqrt 3 + 1 :=
by sorry

end vector_problem_l328_328268


namespace minimum_F_zero_range_k_l328_328252

open Real

theorem minimum_F_zero_range_k (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (|exp x - (1 + ln x + k * x) / x| = 0)) → k ∈ Icc 1 (real_top) :=
by
  sorry

end minimum_F_zero_range_k_l328_328252


namespace delicious_numbers_less_than_99_l328_328786

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def has_exactly_4_factors (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ (n = p^3 ∨ n = p * q)

def is_delicious (n : ℕ) : Prop :=
  has_exactly_4_factors n ∧ 2 ∣ n

theorem delicious_numbers_less_than_99 : 
  ∑ n in finset.range 99, if is_delicious n then 1 else 0 = 15 :=
sorry

end delicious_numbers_less_than_99_l328_328786


namespace correct_statements_l328_328952

def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x + Real.pi / 6)

theorem correct_statements :
  (∀ x1 x2 : ℝ, (|f x1| = 3 ∧ |f x2| = 3 ∧ x1 ≠ x2) → ∃ k : ℤ, x1 - x2 = (k : ℝ) * (Real.pi / 2)) ∧
  (f (Real.pi / 6) = 0) ∧
  (∃ x : ℝ, f (Real.pi / 3) ≠ x) ∧
  (¬ (f  x = 3 * Real.sin (2 * x - Real.pi / 3))) ∧
  (∀ x ∈ Set.Icc (- Real.pi / 3) (- Real.pi / 6), MonotoneOn f (Set.Icc x x))
:= by sorry

end correct_statements_l328_328952


namespace bisector_through_fixed_point_l328_328791

theorem bisector_through_fixed_point
  (C C' : Circle)
  (A B : Point)
  (hAB : C ∩ C' = {A, B})
  (D : Line)
  (hD : A ∈ D)
  (P P' : Point)
  (hP : P ∈ C ∧ P ∈ D)
  (hP' : P' ∈ C' ∧ P' ∈ D) :
  ∃ M : Point, (is_midpoint M (line_through B D ∩ C) (line_through B D ∩ C')) ∧
    (is_perpendicular_bisector M P P') := sorry

end bisector_through_fixed_point_l328_328791


namespace tara_had_more_l328_328689

theorem tara_had_more (M T X : ℕ) (h1 : T = 15) (h2 : M + T = 26) (h3 : T = M + X) : X = 4 :=
by 
  sorry

end tara_had_more_l328_328689


namespace number_of_valid_six_digit_house_numbers_l328_328863

-- Define the set of two-digit primes less than 60
def two_digit_primes : List ℕ := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

-- Define a predicate checking if a number is a two-digit prime less than 60
def is_valid_prime (n : ℕ) : Prop :=
  n ∈ two_digit_primes

-- Define the function to count distinct valid primes forming ABCDEF
def count_valid_house_numbers : ℕ :=
  let primes_count := two_digit_primes.length
  primes_count * (primes_count - 1) * (primes_count - 2)

-- State the main theorem
theorem number_of_valid_six_digit_house_numbers : count_valid_house_numbers = 1716 := by
  -- Showing the count of valid house numbers forms 1716
  sorry

end number_of_valid_six_digit_house_numbers_l328_328863


namespace most_likely_hits_8_l328_328424

noncomputable def most_likely_hits (p : ℝ) (n : ℕ) : ℕ := (p * n).to_nat

theorem most_likely_hits_8 :
  most_likely_hits 0.8 10 = 8 := by
  sorry

end most_likely_hits_8_l328_328424


namespace total_exercise_time_l328_328329

def distance(monday: Nat, monday_speed: Nat, wednesday: Nat, wednesday_speed: Nat, friday: Nat, friday_speed: Nat) : Nat :=
  (monday / monday_speed) + (wednesday / wednesday_speed) + (friday / friday_speed)

theorem total_exercise_time :
  distance 6 2 6 3 6 6 = 6 := by
  sorry

end total_exercise_time_l328_328329


namespace green_area_growth_l328_328042

theorem green_area_growth (x : ℕ) (hx : 0 < x) : 
  ∃ y : ℕ, y = 1000 * (1.04 ^ x) := by
  sorry

end green_area_growth_l328_328042


namespace inverse_proportion_example_l328_328770

def is_inverse_proportion (f : ℝ → ℝ) :=
  ∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → (f x) * x = c

theorem inverse_proportion_example :
  is_inverse_proportion (λ x : ℝ, -2 / x) ∧
  ¬ is_inverse_proportion (λ x : ℝ, -x / 2) ∧
  ¬ is_inverse_proportion (λ x : ℝ, -2 * x^2) ∧
  ¬ is_inverse_proportion (λ x : ℝ, -2 * x + 1) :=
by
  sorry

end inverse_proportion_example_l328_328770


namespace binomial_30_3_l328_328854

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l328_328854


namespace total_bowling_balls_l328_328033

theorem total_bowling_balls (red_balls : ℕ) (green_balls : ℕ) (h1 : red_balls = 30) (h2 : green_balls = red_balls + 6) : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l328_328033


namespace max_trig_expr_l328_328202

noncomputable def trig_expr (A B C : ℝ) : ℝ :=
  sin A ^ 2 * cos B ^ 2 + sin B ^ 2 * cos C ^ 2 + sin C ^ 2 * cos A ^ 2

theorem max_trig_expr (A B C : ℝ) : trig_expr A B C ≤ 1 :=
sorry

end max_trig_expr_l328_328202


namespace f_le_g_l328_328567

noncomputable def f (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / (i + 1 : ℝ) ^ 2

noncomputable def g (n : ℕ) : ℝ := 1 / 2 * (3 - 1 / (n : ℝ) ^ 2)

theorem f_le_g (n : ℕ) (h : n > 0) : f n ≤ g n :=
sorry

end f_le_g_l328_328567


namespace simplify_expression_l328_328395

theorem simplify_expression : 
  2 + (1 / (2 + Real.sqrt 5)) - (1 / (2 - Real.sqrt 5)) = 2 - 2 * Real.sqrt 5 := 
by 
  sorry

end simplify_expression_l328_328395


namespace simplify_evaluate_expression_l328_328017

theorem simplify_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6)) = Real.sqrt 2 :=
by
  sorry

end simplify_evaluate_expression_l328_328017


namespace find_x_for_hx_eq_x_l328_328368

-- Given: Definition of h with the condition h(6x - 3) = 4x + 9
def h (y : ℝ) := (2 * (y + 3) / 3) + 9

-- State the problem: Find x such that h(x) = x
theorem find_x_for_hx_eq_x : ∃ x : ℝ, h x = x :=
begin
  use 33,
  dsimp [h],
  linarith,
end

end find_x_for_hx_eq_x_l328_328368


namespace distinct_triangles_3x3_grid_l328_328611

open Finset
open Fin

noncomputable def count_distinct_triangles : ℕ :=
  let points := univ.product univ in
  let total_combinations := (points.choose 3).card in
  let rows_cols_diagonals := 3 + 3 + 2 in
  total_combinations - rows_cols_diagonals

theorem distinct_triangles_3x3_grid : count_distinct_triangles = 76 := by
  sorry

end distinct_triangles_3x3_grid_l328_328611


namespace surface_is_plane_l328_328891

-- Define cylindrical coordinates
structure CylindricalCoordinate where
  r : ℝ
  θ : ℝ
  z : ℝ

-- Define the property for a constant θ
def isConstantTheta (c : ℝ) (coord : CylindricalCoordinate) : Prop :=
  coord.θ = c

-- Define the plane in cylindrical coordinates
def isPlane (c : ℝ) (coord : CylindricalCoordinate) : Prop :=
  isConstantTheta c coord

-- Theorem: The surface described by θ = c in cylindrical coordinates is a plane.
theorem surface_is_plane (c : ℝ) (coord : CylindricalCoordinate) :
    isPlane c coord ↔ isConstantTheta c coord := sorry

end surface_is_plane_l328_328891


namespace eval_complex_powers_l328_328550

noncomputable def i_pow (n : ℕ) : ℂ :=
  if n % 4 = 0 then 1
  else if n % 4 = 1 then complex.I
  else if n % 4 = 2 then -1
  else -complex.I

theorem eval_complex_powers : i_pow 14 + i_pow 19 + i_pow 24 + i_pow 29 + i_pow 34 = -1 :=
by
  -- The proof steps are omitted
  sorry

end eval_complex_powers_l328_328550


namespace probability_not_six_l328_328634

theorem probability_not_six (odds_six : ℚ) (odds_not_six : ℚ) (h_odds : odds_six / odds_not_six = 2 / 5) : 
  (1 - odds_six / (odds_six + odds_not_six)) = 5 / 7 :=
by
  have h_sum : odds_six + odds_not_six = 7 * (odds_six / (odds_six / 7 + odds_six / 7)),
  {
    sorry -- Placeholder for proof that the sum of the odds equals 7 times the combined probability
  }
  have h_prob_six : odds_six / (odds_six + odds_not_six) = 2 / 7,
  {
    -- Placeholder
    sorry 
  }
  have h_prob_not_six : 1 - (odds_six / (odds_six + odds_not_six)) = 5 / 7,
  {
    -- Placeholder
    sorry 
  }
  exact h_prob_not_six

end probability_not_six_l328_328634


namespace odd_function_expression_l328_328218

theorem odd_function_expression (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 < x → f x = x^2 + |x| - 1) : 
  ∀ x, x < 0 → f x = -x^2 + x + 1 :=
by
  sorry

end odd_function_expression_l328_328218


namespace percent_matches_won_l328_328426

theorem percent_matches_won (ratio_won_lost : ℕ) (ratio_lost_won : ℕ) (games_played : ℕ)
  (h_ratio : ratio_won_lost / ratio_lost_won = 7 / 5)
  (h_games : games_played ≥ 48) :
  let wins := ratio_won_lost * (games_played / (ratio_won_lost + ratio_lost_won)),
      total := ratio_won_lost + ratio_lost_won,
      percent := ((wins * 100) / games_played).to_nat in
  percent = 58 :=
by sorry

end percent_matches_won_l328_328426


namespace intersection_A_notB_l328_328263

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A according to the given condition
def A : Set ℝ := { x | |x - 1| > 1 }

-- Define set B according to the given condition
def B : Set ℝ := { x | (x - 1) * (x - 4) > 0 }

-- Define the complement of set B in U
def notB : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

-- Lean statement to prove A ∩ notB = { x | 2 < x ∧ x ≤ 4 }
theorem intersection_A_notB :
  A ∩ notB = { x | 2 < x ∧ x ≤ 4 } :=
sorry

end intersection_A_notB_l328_328263


namespace chocolates_remaining_l328_328610

theorem chocolates_remaining 
  (total_chocolates : ℕ)
  (ate_day1 : ℕ) (ate_day2 : ℕ) (ate_day3 : ℕ) (ate_day4 : ℕ) (ate_day5 : ℕ) (remaining_chocolates : ℕ) 
  (h_total : total_chocolates = 48)
  (h_day1 : ate_day1 = 6) 
  (h_day2 : ate_day2 = 2 * ate_day1 + 2) 
  (h_day3 : ate_day3 = ate_day1 - 3) 
  (h_day4 : ate_day4 = 2 * ate_day3 + 1) 
  (h_day5 : ate_day5 = ate_day2 / 2) 
  (h_rem : remaining_chocolates = total_chocolates - (ate_day1 + ate_day2 + ate_day3 + ate_day4 + ate_day5)) :
  remaining_chocolates = 14 :=
sorry

end chocolates_remaining_l328_328610


namespace not_both_conditions_l328_328086

noncomputable def pairs : List (ℚ × ℚ) := [
  (-6, -4), 
  (3, 8), 
  (-3/2, -16), 
  (2, 12), 
  (4/3, 18)
]

def product_is_24 (p : ℚ × ℚ) : Prop :=
  p.1 * p.2 = 24

def sum_is_greater_than_zero (p : ℚ × ℚ) : Prop :=
  p.1 + p.2 > 0

theorem not_both_conditions (p : ℚ × ℚ) (h : p = (-6, -4) ∨ p = (-3/2, -16)) :
  ¬(product_is_24 p ∧ sum_is_greater_than_zero p) :=
by
  intro h_cond
  cases h with h1 h2
  · rw h1 at h_cond
    simp at h_cond
    contradiction
  · rw h2 at h_cond
    simp at h_cond
    contradiction

end not_both_conditions_l328_328086


namespace product_ab_zero_l328_328585

theorem product_ab_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end product_ab_zero_l328_328585


namespace major_airlines_wifi_l328_328398

-- Definitions based on conditions
def percentage (x : ℝ) := 0 ≤ x ∧ x ≤ 100

variables (W S B : ℝ)

-- Assume the conditions
axiom H1 : S = 70
axiom H2 : B = 45
axiom H3 : B ≤ S

-- The final proof problem that W = 45
theorem major_airlines_wifi : W = B :=
by
  sorry

end major_airlines_wifi_l328_328398


namespace remaining_solid_edges_l328_328820

-- A cube structure is defined by its side length
structure Cube where
  side_length : ℕ

-- Conditions based on the problem
def initial_cube : Cube := {side_length := 4}
def removed_cubes := List.replicate 8 {side_length := 2}

-- Main theorem stating that removing smaller cubes results in a solid with 24 edges
theorem remaining_solid_edges (c : Cube) (removals : List Cube) (h1 : c = initial_cube) (h2 : removals = removed_cubes) :
  let edges_after_removal := 24
  edges_after_removal = 24 :=
by
  sorry

end remaining_solid_edges_l328_328820


namespace polygon_diagonal_perimeter_inequality_l328_328371

theorem polygon_diagonal_perimeter_inequality
  (n p d : ℕ) (h_n : n > 3) 
  (h_diagonal_sum : d = sum_of_diagonals n)
  (h_perimeter : p = perimeter_of_polygon n) :
  n - 3 < (2 * d) / p ∧ (2 * d) / p < (⌊ n / 2 ⌋ * ⌊ (n + 1) / 2 ⌋ - 2) := sorry

end polygon_diagonal_perimeter_inequality_l328_328371


namespace slices_in_large_pizza_l328_328690

theorem slices_in_large_pizza (pizzas : ℕ) (games : ℕ) (goals_per_game : ℕ) (slices_per_goal : goals_per_game * games = pizzas * 12) :
  (games = 8) → (goals_per_game = 9) → (pizzas = 6) → 
  slices_per_goal := 12 :=
begin
  assume h_games h_goals_per_game h_pizzas,
  rw [h_games, h_goals_per_game, h_pizzas] at slices_per_goal,
  exact slices_per_goal,
end

end slices_in_large_pizza_l328_328690


namespace clarence_oranges_left_l328_328536

-- Definitions based on the conditions in the problem
def initial_oranges : ℕ := 5
def oranges_from_joyce : ℕ := 3
def total_oranges_after_joyce : ℕ := initial_oranges + oranges_from_joyce
def oranges_given_to_bob : ℕ := total_oranges_after_joyce / 2
def oranges_left : ℕ := total_oranges_after_joyce - oranges_given_to_bob

-- Proof statement that needs to be proven
theorem clarence_oranges_left : oranges_left = 4 :=
by
  sorry

end clarence_oranges_left_l328_328536


namespace find_range_a_l328_328602

noncomputable def f (a x : ℝ) : ℝ := abs (2 * x * a + abs (x - 1))

theorem find_range_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 5) ↔ a ≥ 6 :=
by
  sorry

end find_range_a_l328_328602


namespace cos_of_angle_in_third_quadrant_l328_328627

theorem cos_of_angle_in_third_quadrant (B : ℝ) (h1 : -1 ≤ sin B ∧ sin B ≤ 1) (h2 : sin B = -5 / 13) (h3 : 3 * π / 2 ≤ B ∧ B ≤ 2 * π) :
  cos B = -12 / 13 :=
by
  sorry

end cos_of_angle_in_third_quadrant_l328_328627


namespace gabrielle_peaches_l328_328687

theorem gabrielle_peaches (B G : ℕ) 
  (h1 : 16 = 2 * B + 6)
  (h2 : B = G / 3) :
  G = 15 :=
by
  sorry

end gabrielle_peaches_l328_328687


namespace range_of_m_l328_328220

open Real

axiom sqrt_three_sin_plus_cos (m : ℝ) : (∀ x : ℝ, sqrt 3 * sin x + cos x > m) ↔ m < -2
axiom x_squared_plus_mx_plus_one (m : ℝ) : (∃ x : ℝ, x^2 + m * x + 1 ≤ 0) ↔ (m ≤ -2 ∨ m ≥ 2)

theorem range_of_m (m : ℝ) (p : (∀ x : ℝ, sqrt 3 * sin x + cos x > m)) (q : (∃ x : ℝ, x^2 + m * x + 1 ≤ 0)) 
  (h1 : p ∨ q) (h2 : ¬ (p ∧ q)) : m = -2 ∨ m ≥ 2 :=
by
  sorry

end range_of_m_l328_328220


namespace hourly_wage_increase_l328_328501

variables (W W' H H' : ℝ)

theorem hourly_wage_increase :
  H' = (2/3) * H →
  W * H = W' * H' →
  W' = (3/2) * W :=
by
  intros h_eq income_eq
  rw [h_eq] at income_eq
  sorry

end hourly_wage_increase_l328_328501


namespace problem1_problem2_problem3_l328_328919

-- Problem (1)
theorem problem1 (a c : ℝ) (b : ℝ) (h₁ : a = 1) (h₂ : c = 1) (h₃ : ∀ x : ℝ, ¬(a * x^2 + b * x + c ≤ 0)) :
  -2 < b ∧ b < 2 :=
sorry

-- Problem (2)
theorem problem2 (a c : ℝ) (b : ℝ) (M : set ℝ) (h₁ : a > 0) (h₂ : M = {x : ℝ | -1 ≤ x ∧ x ≤ 3}) :
  {x : ℝ | -cx^2 - bx - b > cx + 4a} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 1 / 3} :=
sorry

-- Problem (3)
theorem problem3 (a b c : ℝ) (h₁ : a > b) (h₂ : b > c) (h₃ : ∃ t : ℝ, ∀ x : ℝ, x ≠ t → ¬ (a * x^2 + b * x + c ≤ 0)) :
  (a + 4 * c) / b = 2 :=
sorry

end problem1_problem2_problem3_l328_328919


namespace circle_line_intersection_solutions_l328_328557

theorem circle_line_intersection_solutions :
  (∃ x y : ℝ, (x^2 + y^2 = 1) ∧ (2 * x + 2 * y - 1 - sqrt 3 = 0)) →
  (∃ α β : ℝ, (sin α = sqrt 3 / 2) ∧ (sin (2 * β) = 1 / 2) ∧ 
               (sin β = 1 / 2) ∧ (cos (2 * α) = sqrt 3 / 2) ∨
               (sin α = 1 / 2) ∧ (sin (2 * β) = sqrt 3 / 2) ∧ 
               (sin β = sqrt 3 / 2) ∧ (cos (2 * α) = 1 / 2)) →
  (∃ n k : ℤ, α = (-1)^n * π / 6 + π * n ∧ β = π / 3 + 2 * π * k) :=
by
  intro h1
  intro h2
  sorry

end circle_line_intersection_solutions_l328_328557


namespace find_speed_of_first_train_l328_328759

noncomputable def speed_of_first_train : ℝ :=
  let length_train_1 := 121 / 1000 -- in kilometers
  let length_train_2 := 165 / 1000 -- in kilometers
  let speed_train_2 := 65 -- in kmph
  let time_seconds := 7.100121645440779
  let time_hours := time_seconds / 3600 -- converting seconds to hours
  let total_distance := length_train_1 + length_train_2
  let relative_speed := total_distance / time_hours
  relative_speed - speed_train_2

theorem find_speed_of_first_train :
  speed_of_first_train = 80.008 :=
begin
  sorry
end

end find_speed_of_first_train_l328_328759


namespace scientific_notation_2600000_l328_328151

theorem scientific_notation_2600000 : ∃ (c : ℝ) (n : ℤ), 1 ≤ c ∧ c < 10 ∧ 2600000 = c * 10 ^ n ∧ c = 2.6 ∧ n = 6 :=
by
  use 2.6, 6
  split; try {norm_num}
  exact ⟨by norm_num, by norm_num, by norm_num⟩

end scientific_notation_2600000_l328_328151


namespace problem_solution_l328_328200

-- Define the function g(n) as specified in the problem
def g (n : ℕ) : ℝ := ∑' k in (Set.Ici 3), (1 : ℝ) / (k : ℝ)^n

-- Statement of the problem to prove
theorem problem_solution : (∑' n in (Set.Ici 2), g n) = 1 / 2 :=
by
  sorry

end problem_solution_l328_328200


namespace min_value_zero_of_f_l328_328571

noncomputable def f (x : ℝ) (a : ℝ) := x^2 - a * x - a^2 * (Real.log x)

theorem min_value_zero_of_f (a : ℝ) (x : ℝ) (h : f x a = 0) : a = 1 ∨ a = -2 * Real.exp(3/4) :=
sorry

end min_value_zero_of_f_l328_328571


namespace timothy_tea_cups_l328_328064

theorem timothy_tea_cups (t : ℕ) (h : 6 * t + 60 = 120) : t + 12 = 22 :=
by
  sorry

end timothy_tea_cups_l328_328064


namespace equal_mass_piles_l328_328564

theorem equal_mass_piles (n : ℕ) (hn : n > 3) (hn_mod : n % 3 = 0 ∨ n % 3 = 2) : 
  ∃ A B C : Finset ℕ, A ∪ B ∪ C = {i | i ∈ Finset.range (n + 1)} ∧
  Disjoint A B ∧ Disjoint A C ∧ Disjoint B C ∧
  A.sum id = B.sum id ∧ B.sum id = C.sum id :=
sorry

end equal_mass_piles_l328_328564


namespace team_team_count_correct_l328_328204

/-- Number of ways to select a team of three students from 20,
    one for each subject: math, Russian language, and informatics. -/
def ways_to_form_team (n : ℕ) : ℕ :=
  if n ≥ 3 then n * (n - 1) * (n - 2) else 0

theorem team_team_count_correct : ways_to_form_team 20 = 6840 :=
by sorry

end team_team_count_correct_l328_328204


namespace circle_prime_quotients_l328_328373

theorem circle_prime_quotients (m : ℕ) (h_pos : 0 < m) : 
  (∃ (a : Fin m → ℕ), 
    (∀ i : Fin m, 0 < a i) ∧ 
    (∀ i : Fin m, Prime (if a i < a ((i + 1) % m) then a ((i + 1) % m) / a i else a i / a ((i + 1) % m)))) ↔ 
  Even m := 
by 
  sorry

end circle_prime_quotients_l328_328373


namespace product_of_real_parts_l328_328175

noncomputable def complex_solutions_product : ℂ :=
  4 - complex.abs (√17) * (complex.cos ((1 / 2) * real.arctan (1 / 4))) ^ 2

theorem product_of_real_parts (x : ℂ) (hx : x^2 + 4 * x = complex.I) :
  ∀ x₁ x₂ : ℂ, 
    (x₁^2 + 4 * x₁ = complex.I ∧ x₂^2 + 4 * x₂ = complex.I) →
    x₁.re * x₂.re = 4 - complex.abs (√17) * (complex.cos ((1 / 2) * real.arctan (1 / 4))) ^ 2 :=
begin
  sorry
end

end product_of_real_parts_l328_328175


namespace find_initial_cards_l328_328848

theorem find_initial_cards (B : ℕ) :
  let Tim_initial := 20
  let Sarah_initial := 15
  let Tim_after_give_to_Sarah := Tim_initial - 5
  let Sarah_after_give_to_Sarah := Sarah_initial + 5
  let Tim_after_receive_from_Sarah := Tim_after_give_to_Sarah + 2
  let Sarah_after_receive_from_Sarah := Sarah_after_give_to_Sarah - 2
  let Tim_after_exchange_with_Ben := Tim_after_receive_from_Sarah - 3
  let Ben_after_exchange := B + 13
  let Ben_after_all_transactions := 3 * Tim_after_exchange_with_Ben
  Ben_after_exchange = Ben_after_all_transactions -> B = 29 := by
  sorry

end find_initial_cards_l328_328848


namespace slope_of_line_l328_328728

open Classical

-- Define the circle
def circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line passing through point (-1, -2) with unknown slope m
def line (x y m : ℝ) : Prop := y + 2 = m * (x + 1)

-- Condition that the line l intercepts the circle
def line_intercepts_circle (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle x y ∧ line x y m

-- Theorem to be proven
theorem slope_of_line (m : ℝ) :
  (line_intercepts_circle m) → (m = 1 ∨ m = -1) :=
by
  sorry

end slope_of_line_l328_328728


namespace expression_evaluation_l328_328101

theorem expression_evaluation :
  (0.8 ^ 3) - ((0.5 ^ 3) / (0.8 ^ 2)) + 0.40 + (0.5 ^ 2) = 0.9666875 := 
by 
  sorry

end expression_evaluation_l328_328101


namespace range_of_x_l328_328573

noncomputable def g : ℝ → ℝ := λ x, 2 ^ x + 2 ^ (-x) + |x|

theorem range_of_x (x : ℝ) : g (2 * x - 1) < g 3 ↔ -1 < x ∧ x < 2 := by
  sorry

end range_of_x_l328_328573


namespace probability_of_tulip_l328_328552

theorem probability_of_tulip (roses tulips daisies lilies : ℕ) (total : ℕ) (h_total : total = 3 + 2 + 4 + 6) (h_tulips : tulips = 2) :
  tulips / total = 2 / 15 :=
by {
  rw [h_total, h_tulips],
  norm_num,
}

end probability_of_tulip_l328_328552


namespace three_distinct_prime_products_sum_59_l328_328078

theorem three_distinct_prime_products_sum_59 :
  ∃! (S : Finset (Finset ℕ)), (∀ s ∈ S, (∑ p in s, p = 59) ∧
                                        s.card = 3 ∧
                                        ∀ p ∈ s, Prime p) ∧
                            S.card = 3 := sorry

end three_distinct_prime_products_sum_59_l328_328078


namespace vector_properties_l328_328608

noncomputable def a : ℝ × ℝ := (3, -4)
noncomputable def b (x : ℝ) : ℝ × ℝ := (2, x)
noncomputable def c (y : ℝ) : ℝ × ℝ := (2, y)

def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem vector_properties :
  let x := -8 / 3,
      y := 3 / 2,
      b := b x,
      c := c y in
  parallel a b ∧ perpendicular a c ∧ 
  b = (2, -8 / 3) ∧ c = (2, 3 / 2) ∧ 
  let θ := real.arccos ((2 * 2 + (-8 / 3) * (3 / 2)) / (real.sqrt (2^2 + (-8 / 3)^2) * real.sqrt (2^2 + (3 / 2)^2))) in
  θ = real.pi / 2 :=
by
  sorry

end vector_properties_l328_328608


namespace total_bowling_balls_l328_328031

theorem total_bowling_balls (red_balls : ℕ) (green_balls : ℕ) (h1 : red_balls = 30) (h2 : green_balls = red_balls + 6) : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l328_328031


namespace scientific_notation_of_2600000_l328_328149

theorem scientific_notation_of_2600000 :
    ∃ (a : ℝ) (b : ℤ), (2600000 : ℝ) = a * 10^b ∧ a = 2.6 ∧ b = 6 :=
begin
    sorry
end

end scientific_notation_of_2600000_l328_328149


namespace applicant_overall_score_is_72_l328_328806

-- Define conditions as variables
variables (written_score : ℕ) (interview_score : ℕ) (written_weight : ℝ) (interview_weight : ℝ)

-- Define correct answer for the overall score calculation
def overall_score (written_score interview_score : ℕ) (written_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

-- We state the main theorem to prove the overall score is 72 given the conditions
theorem applicant_overall_score_is_72 :
  written_score = 80 → interview_score = 60 → written_weight = 0.6 → interview_weight = 0.4 →
  overall_score written_score interview_score written_weight interview_weight = 72 :=
by
  intros h_written_score h_interview_score h_written_weight h_interview_weight
  rw [h_written_score, h_interview_score, h_written_weight, h_interview_weight]
  norm_num
  rw [nat.cast_mul]
  sorry

end applicant_overall_score_is_72_l328_328806


namespace tom_remaining_balloons_l328_328450

theorem tom_remaining_balloons (initial_balloons : ℕ) (balloons_given : ℕ) (balloons_remaining : ℕ) 
  (h1 : initial_balloons = 30) (h2 : balloons_given = 16) : balloons_remaining = 14 := 
by
  sorry

end tom_remaining_balloons_l328_328450


namespace students_surveyed_l328_328108

theorem students_surveyed (S : ℕ)
  (h1 : (2/3 : ℝ) * 6 + (1/3 : ℝ) * 4 = 16/3)
  (h2 : S * (16/3 : ℝ) = 320) :
  S = 60 :=
sorry

end students_surveyed_l328_328108


namespace sum_integers_75_through_85_l328_328073

theorem sum_integers_75_through_85 : ∑ k in Finset.Icc 75 85, k = 880 := by
  sorry

end sum_integers_75_through_85_l328_328073


namespace find_N_l328_328435

theorem find_N (N : ℝ) (h : sqrt (0.05 * N) * sqrt 5 = 0.25000000000000006) : N = 0.25 := by
  sorry

end find_N_l328_328435


namespace vector_difference_parallelogram_l328_328967

noncomputable theory

open_locale classical

-- Define the vector types and the two given vectors
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def a : Vector2D := ⟨-1, -3⟩
def b : ℝ → Vector2D := λ t, ⟨2, t⟩

-- Define the condition that vectors are parallel
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

-- Theorem to prove
theorem vector_difference_parallelogram (t : ℝ) (h : parallel a (b t)) :
  a.x - (b t).x = -3 ∧ a.y - (b t).y = -9 :=
by sorry

end vector_difference_parallelogram_l328_328967


namespace determine_N_l328_328277

theorem determine_N (N : ℕ) : 995 + 997 + 999 + 1001 + 1003 = 5100 - N → N = 100 := by
  sorry

end determine_N_l328_328277


namespace range_of_f_range_of_a_l328_328249

-- Define the function f as given in the problem
def f (x : ℝ) : ℝ := 2 * (Real.sin (x + Real.pi / 4)) ^ 2 - Real.sqrt 3 * Real.cos (2 * x)

-- Condition: the domain of x is [π/4, π/2]
def domain (x : ℝ) := (Real.pi / 4) ≤ x ∧ x ≤ (Real.pi / 2)

-- Theorem 1: Prove the range of f(x)
theorem range_of_f :
  ∀ x, domain x → 2 ≤ f x ∧ f x ≤ 3 :=
sorry

-- Define the condition for the function y = f(x) - a to have two zeros
def has_two_zeros (a : ℝ) : Prop :=
  ∃ x1 x2, domain x1 ∧ domain x2 ∧ x1 ≠ x2 ∧ f x1 - a = 0 ∧ f x2 - a = 0

-- Theorem 2: Prove the range of a
theorem range_of_a : 
  ∀ a, has_two_zeros a → Real.sqrt 3 + 1 ≤ a ∧ a < 3 :=
sorry

end range_of_f_range_of_a_l328_328249


namespace find_vertex_P_l328_328657

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def midpoint (A B : Point3D) : Point3D :=
  { x := (A.x + B.x) / 2,
    y := (A.y + B.y) / 2,
    z := (A.z + B.z) / 2 }

theorem find_vertex_P (Q R N O : Point3D)
  (hmn : midpoint Q R = { x := 2, y := 3, z := 1 })
  (hnp : midpoint P R = { x := -1, y := 2, z := -3 })
  (hop : midpoint P Q = { x := 3, y := 0, z := 5 }) :
  P = { x := 1, y := 5, z := -4.5 } :=
sorry

end find_vertex_P_l328_328657


namespace bids_per_person_l328_328162

theorem bids_per_person (initial_price final_price price_increase_per_bid : ℕ) (num_people : ℕ)
  (h1 : initial_price = 15) (h2 : final_price = 65) (h3 : price_increase_per_bid = 5) (h4 : num_people = 2) :
  (final_price - initial_price) / price_increase_per_bid / num_people = 5 :=
  sorry

end bids_per_person_l328_328162


namespace evaluate_expression_l328_328179

def binary_operation (X Y : ℝ) : ℝ := (X + Y) / 4

theorem evaluate_expression : binary_operation (binary_operation 3 9) 6 = 2.25 :=
by 
  sorry

end evaluate_expression_l328_328179


namespace choose_team_captains_l328_328824

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem choose_team_captains :
  let total_members := 15
  let shortlisted := 5
  let regular := total_members - shortlisted
  binom total_members 4 - binom regular 4 = 1155 :=
by
  sorry

end choose_team_captains_l328_328824


namespace find_theta_l328_328740

-- Definitions of the conditions
variables {α β θ : ℝ}

-- Main Theorem
theorem find_theta (h_alpha : 0 < α ∧ α < 60) 
    (h_beta : 0 < β ∧ β < 60) 
    (h_theta : 0 < θ ∧ θ < 60) 
    (h_alpha_beta_theta : α + β = 2 * θ)
    (h_trig_identity : sin α * sin β * sin θ = sin (60 - α) * sin (60 - β) * sin (60 - θ)) : 
    θ = 30 :=
by
  sorry

end find_theta_l328_328740


namespace GS_perpendicular_BC_l328_328097

noncomputable def triangle (A B C: Type) :=
  ∃ α β γ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ 
  acute α ∧ acute β ∧ acute γ

noncomputable def centroid (A B C G: Type) : Prop :=
  is_centroid A B C G

noncomputable def midpoint (B C M: Type) : Prop :=
  is_midpoint B C M

noncomputable def circle (G M Ω: Type) : Prop :=
  is_center G Ω ∧ radius Ω = distance G M
  
noncomputable def intersection (Ω BC N M: Type) : Prop :=
  intersects Ω BC N ∧ N ≠ M

noncomputable def symmetric_point (A N S: Type) : Prop :=
  symmetric A N S ∧ distance A N = distance N S 

theorem GS_perpendicular_BC 
  {A B C G M Ω N S: Type}
  (h1 : triangle A B C)
  (h2 : centroid A B C G)
  (h3 : midpoint B C M)
  (h4 : circle G M Ω)
  (h5: intersection Ω BC N M)
  (h6 : symmetric_point A N S) :
  perpendicular (line G S) (line B C) :=
sorry

end GS_perpendicular_BC_l328_328097


namespace xiaoli_estimate_larger_l328_328644

variable (x y z w : ℝ)
variable (hxy : x > y) (hy0 : y > 0) (hz1 : z > 1) (hw0 : w > 0)

theorem xiaoli_estimate_larger : (x + w) - (y - w) * z > x - y * z :=
by sorry

end xiaoli_estimate_larger_l328_328644


namespace area_of_triangle_l328_328417

theorem area_of_triangle (h : ℝ) (a : ℝ) (b : ℝ) (hypotenuse : h = 13) (side_a : a = 5) (right_triangle : a^2 + b^2 = h^2) : 
  ∃ (area : ℝ), area = 30 := 
by
  sorry

end area_of_triangle_l328_328417


namespace hotel_loss_l328_328133

variable (operations_expenses : ℝ)
variable (fraction_payment : ℝ)

theorem hotel_loss :
  operations_expenses = 100 →
  fraction_payment = 3 / 4 →
  let total_payment := fraction_payment * operations_expenses in
  let loss := operations_expenses - total_payment in
  loss = 25 :=
by
  intros h₁ h₂
  have tstp : total_payment = 75 := by
    rw [h₁, h₂]
    norm_num
  have lss : loss = 25 := by
    rw [h₁, tstp]
    norm_num
  exact lss

end hotel_loss_l328_328133


namespace insects_remaining_l328_328709

-- Define the initial counts of spiders, ants, and ladybugs
def spiders : ℕ := 3
def ants : ℕ := 12
def ladybugs : ℕ := 8

-- Define the number of ladybugs that flew away
def ladybugs_flew_away : ℕ := 2

-- Prove the total number of remaining insects in the playground
theorem insects_remaining : (spiders + ants + ladybugs - ladybugs_flew_away) = 21 := by
  -- Expand the definitions and compute the result
  sorry

end insects_remaining_l328_328709


namespace largest_hexagon_in_square_l328_328924

theorem largest_hexagon_in_square {s : ℝ} (h : 0 < s):
  ∃ hexagon, is_regular_hexagon hexagon ∧ inscribed hexagon (square s) ∧ angle_at_center hexagon (square s) = 60 :=
sorry

end largest_hexagon_in_square_l328_328924


namespace range_of_m_l328_328198

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x-3| + |x+4| ≥ |2*m-1|) ↔ (-3 ≤ m ∧ m ≤ 4) := by
  sorry

end range_of_m_l328_328198


namespace ratio_of_areas_l328_328504

noncomputable def area_of_paper (width length : ℝ) : ℝ :=
  width * length

noncomputable def area_of_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

theorem ratio_of_areas (w : ℝ) (A B : ℝ) (hA : A = 2 * w^2)
  (hB : B = (sqrt 2 * w^2) / 4) :
  B / A = sqrt 2 / 8 :=
by sorry

end ratio_of_areas_l328_328504


namespace coefficient_a9_of_polynomial_l328_328294

theorem coefficient_a9_of_polynomial (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a_0 + 
    a_1 * (x + 1) + 
    a_2 * (x + 1)^2 + 
    a_3 * (x + 1)^3 + 
    a_4 * (x + 1)^4 + 
    a_5 * (x + 1)^5 + 
    a_6 * (x + 1)^6 + 
    a_7 * (x + 1)^7 + 
    a_8 * (x + 1)^8 + 
    a_9 * (x + 1)^9 + 
    a_10 * (x + 1)^10) 
  → a_9 = -10 :=
by
  intro h
  sorry

end coefficient_a9_of_polynomial_l328_328294


namespace each_person_bids_five_times_l328_328155

noncomputable def auction_bidding : Prop :=
  let initial_price := 15
  let final_price := 65
  let price_increase_per_bid := 5
  let number_of_bidders := 2
  let total_increase := final_price - initial_price
  let total_bids := total_increase / price_increase_per_bid
  total_bids / number_of_bidders = 5

theorem each_person_bids_five_times : auction_bidding :=
by
  -- The proof will be filled in here.
  sorry

end each_person_bids_five_times_l328_328155


namespace applicant_overall_score_is_72_l328_328805

-- Define conditions as variables
variables (written_score : ℕ) (interview_score : ℕ) (written_weight : ℝ) (interview_weight : ℝ)

-- Define correct answer for the overall score calculation
def overall_score (written_score interview_score : ℕ) (written_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

-- We state the main theorem to prove the overall score is 72 given the conditions
theorem applicant_overall_score_is_72 :
  written_score = 80 → interview_score = 60 → written_weight = 0.6 → interview_weight = 0.4 →
  overall_score written_score interview_score written_weight interview_weight = 72 :=
by
  intros h_written_score h_interview_score h_written_weight h_interview_weight
  rw [h_written_score, h_interview_score, h_written_weight, h_interview_weight]
  norm_num
  rw [nat.cast_mul]
  sorry

end applicant_overall_score_is_72_l328_328805


namespace solve_for_x_l328_328021

theorem solve_for_x (x : ℚ) : 
  5*x + 9*x = 450 - 10*(x - 5) -> x = 125/6 :=
by
  sorry

end solve_for_x_l328_328021


namespace parallelogram_area_l328_328308

theorem parallelogram_area (angle_B angle_D : ℝ)
                           (AB BC : ℝ) 
                           (h_angle_B : angle_B = 100) 
                           (h_angle_D : angle_D = 100)
                           (h_AB : AB = 18) 
                           (h_BC : BC = 10) :
  ∃ (area: ℝ), area = 180 * Real.sin 10 :=
begin
  use 180 * Real.sin 10,
  sorry
end

end parallelogram_area_l328_328308


namespace max_min_difference_l328_328603

noncomputable def f (x : ℝ) : ℝ := 4 * Real.pi * Real.arcsin x - (Real.arccos (-x))^2

theorem max_min_difference : 
  ∃ M m : ℝ, (∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x ≤ M) ∧ (∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), m ≤ f x) ∧
  (M - m = 3 * Real.pi ^ 2) := sorry

end max_min_difference_l328_328603


namespace farmer_pigs_chickens_l328_328116

-- Defining the problem in Lean 4

theorem farmer_pigs_chickens (p ch : ℕ) (h₁ : 30 * p + 24 * ch = 1200) (h₂ : p > 0) (h₃ : ch > 0) : 
  (p = 4) ∧ (ch = 45) :=
by sorry

end farmer_pigs_chickens_l328_328116


namespace max_ab_l328_328304

theorem max_ab (a b c : ℝ) (h1 : 3 * a + b = 1) (h2 : 0 ≤ a) (h3 : a < 1) (h4 : 0 ≤ b) 
(h5 : b < 1) (h6 : 0 ≤ c) (h7 : c < 1) (h8 : a + b + c = 1) : 
  ab ≤ 1 / 12 := by
  sorry

end max_ab_l328_328304


namespace find_fff_l328_328597

def f (x : ℚ) : ℚ :=
  if x ≥ 2 then x + 2 else x * x

theorem find_fff : f (f (3/2)) = 17/4 := by
  sorry

end find_fff_l328_328597


namespace tom_remaining_balloons_l328_328448

def original_balloons : ℕ := 30
def given_balloons : ℕ := 16
def remaining_balloons (original_balloons given_balloons : ℕ) : ℕ := original_balloons - given_balloons

theorem tom_remaining_balloons : remaining_balloons original_balloons given_balloons = 14 :=
by
  -- proof omitted for clarity
  sorry

end tom_remaining_balloons_l328_328448


namespace engine_efficiency_l328_328074

def P1 (P0 ω t : Real) : Real := P0 * (Real.sin (ω * t)) / (100 + (Real.sin (t^2)))
def P2 (P0 ω t : Real) : Real := 3 * P0 * (Real.sin (2 * ω * t)) / (100 + (Real.sin (2 * t)^2))

theorem engine_efficiency (P0 ω : Real) (A+ A- Q+ η : Real) 
  (hA- : A- = (2 / 3) * A+)
  (hQ+ : Q+ = A+)
  (hη : η = (A+ - A-) / Q+):
  η = 1 / 3 := 
by
  sorry

end engine_efficiency_l328_328074


namespace problem_l328_328251

noncomputable def f : ℝ → ℝ :=
λ x, if x < 1 then 3^x + 2 else log (x + 2) / log 3

theorem problem : f 7 + f 0 = 5 :=
by {
  have h1 : f 7 = log (7 + 2) / log 3, by rw f; simp [if_neg (by norm_num)],
  have h2 : f 0 = 3^0 + 2, by rw f; simp [if_pos (by norm_num)],
  rw [h1, h2],
  norm_num,
  sorry
}

end problem_l328_328251


namespace triangle_angles_l328_328185

-- Set up the conditions of the triangle
variable (A B C : Type) [PlaneGeometry A B C]

-- Definitions of the known properties
variable (median angleBisector altitude : Line B)
variable (angleB : Angle B = 4 * alpha)

-- The goal: to prove the specific angles of the triangle
theorem triangle_angles (A B C : Triangle)
  (median, angleBisector, altitude : Line B)
  (angle_ABC_divided : angleABC B = 4 * alpha) :
  angleABC B = 90 ∧ angleBCA B = 22.5 ∧ angleBAC B = 67.5 :=
sorry

end triangle_angles_l328_328185


namespace minimize_modulus_z_purely_imaginary_u_l328_328233

variables {z : ℂ} (a b : ℝ)

/-- Given that z is a complex number and z + 1/z is real, and also that |z + 2 - i| is minimized
when z = -2√5/5 + √5/5i with the minimum value √5 - 1 --/
theorem minimize_modulus_z
  (hz_real : z + 1/z ∈ ℝ)
  (hz_val : z = - (2 * (sqrt 5) / 5) + (sqrt 5) / 5 * complex.I) :
  |z + 2 - complex.I| = sqrt 5 - 1 :=
sorry

/-- Given z = a + bi (a^2 + b^2 = 1), prove that u = (1 - z) / (1 + z) is purely imaginary --/
theorem purely_imaginary_u
  (h : z = a + b * complex.I)
  (unit_circle : a^2 + b^2 = 1) :
  let u := (1 - z) / (1 + z) in
  ∃ (y : ℝ), u = y * complex.I :=
sorry

end minimize_modulus_z_purely_imaginary_u_l328_328233


namespace magnitude_correct_l328_328266

def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (1, x - 1)
def perp_vectors (x : ℝ) : Prop := 
  let a := vector_a x 
  let b := vector_b x
  (a.1 - 2 * b.1, a.2 - 2 * b.2) ⬝ a = 0

theorem magnitude_correct {x : ℝ} (h : perp_vectors x) : 
  let a := vector_a x 
  let b := vector_b x
  real.sqrt ((a.1 - 2 * b.1) ^ 2 + (a.2 - 2 * b.2) ^ 2) = real.sqrt 2 :=
sorry

end magnitude_correct_l328_328266


namespace cos_of_angle_in_third_quadrant_l328_328625

theorem cos_of_angle_in_third_quadrant (B : ℝ) (h1 : π < B ∧ B < 3 * π / 2) (h2 : Real.sin B = -5 / 13) : Real.cos B = -12 / 13 := 
by 
  sorry

end cos_of_angle_in_third_quadrant_l328_328625


namespace time_to_cross_pole_l328_328511

def train_length := 3000 -- in meters
def train_speed_kmh := 90 -- in kilometers per hour

noncomputable def train_speed_mps : ℝ := train_speed_kmh * (1000 / 3600) -- converting speed to meters per second

theorem time_to_cross_pole : (train_length : ℝ) / train_speed_mps = 120 := 
by
  -- Placeholder for the actual proof
  sorry

end time_to_cross_pole_l328_328511


namespace t_plus_reciprocal_l328_328915

theorem t_plus_reciprocal (t : ℝ) (h : t^2 - 3 * t + 1 = 0) (ht : t ≠ 0) : t + 1/t = 3 :=
by sorry

end t_plus_reciprocal_l328_328915


namespace age_twice_in_years_l328_328065

theorem age_twice_in_years (x : ℕ) : (40 + x = 2 * (12 + x)) → x = 16 :=
by {
  sorry
}

end age_twice_in_years_l328_328065


namespace circle_equation_and_max_area_point_exists_l328_328314

theorem circle_equation_and_max_area_point_exists :
  (∃ C : ℝ → ℝ → Prop, 
    (∀ x y, C x y ↔ (x - 2)^2 + y^2 = 4)) ∧
  (∃ M : ℝ × ℝ, 
    ∃ A B : ℝ × ℝ, 
    M = (1/2, sqrt 7 / 2) ∧ 
    (∀ x y, ((x, y) ∈ circle (0, 0) 1) → ∀ m n, ((m, n) = M) → m * x + n * y = 1) ∧ 
    A ≠ B ∧ 
    (area_of_triangle (0, 0) A B = 1/2)) :=
by sorry

end circle_equation_and_max_area_point_exists_l328_328314


namespace find_m_value_l328_328568

theorem find_m_value (f : ℝ → ℝ) (h1 : ∀ x, f ((x / 2) - 1) = 2 * x + 3) (h2 : f m = 6) : m = -(1 / 4) :=
sorry

end find_m_value_l328_328568


namespace octopus_dressing_orders_l328_328816

/-- A robotic octopus has four legs, and each leg needs to wear a glove before it can wear a boot.
    Additionally, it has two tentacles that require one bracelet each before putting anything on the legs.
    The total number of valid dressing orders is 1,286,400. -/
theorem octopus_dressing_orders : 
  ∃ (n : ℕ), n = 1286400 :=
by
  sorry

end octopus_dressing_orders_l328_328816


namespace space_diagonal_integer_iff_ab_even_l328_328748

theorem space_diagonal_integer_iff_ab_even (a b : ℤ) :
    (∃ c : ℤ, c > 0 ∧ ∃ d : ℤ, d^2 = a^2 + b^2 + c^2) ↔ even (a * b) :=
sorry

end space_diagonal_integer_iff_ab_even_l328_328748


namespace trajectory_of_M_l328_328918

variables {x y : ℝ}

def point_A := (x, 0)
def point_B := (0, y)
def point_M := (3 / 5 * x, 2 / 5 * y)

axiom length_AB : real.sqrt (x^2 + y^2) = 5

theorem trajectory_of_M : (point_M.1^2) / 9 + (point_M.2^2) / 4 = 1 :=
by
  unfold point_M
  have hx : x^2 / 25 = (3/5 * x)^2 / (3/5)^2 := by sorry
  have hy : y^2 / 25 = (2/5 * y)^2 / (2/5)^2 := by sorry
  have hxy : x^2 / 25 + y^2 / 25 = 1 := by sorry
  exact hx.trans $ add_eq_one_of_eq_sub hx hy hxy
  sorry

end trajectory_of_M_l328_328918


namespace bids_per_person_l328_328163

theorem bids_per_person (initial_price final_price price_increase_per_bid : ℕ) (num_people : ℕ)
  (h1 : initial_price = 15) (h2 : final_price = 65) (h3 : price_increase_per_bid = 5) (h4 : num_people = 2) :
  (final_price - initial_price) / price_increase_per_bid / num_people = 5 :=
  sorry

end bids_per_person_l328_328163


namespace longest_side_of_triangle_l328_328147

-- Definitions of the problem conditions
def triangle (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b
def tangent_point_division (a b c : ℝ) : Prop := a = b + c
def incircle_radius (r : ℝ) (s : ℝ) (A : ℝ) : Prop := r = A / s

-- Given conditions
def given_conditions (a b c : ℝ) (r : ℝ) : Prop :=
  tangent_point_division a 9 5 ∧ incircle_radius r ((a + b + c) / 2) (sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)))

-- The statement to prove
theorem longest_side_of_triangle (a b c : ℝ) (r : ℝ) :
  given_conditions a b c r → (c = 18 ∨ b = 18 ∨ a = 18) :=
sorry

end longest_side_of_triangle_l328_328147


namespace lcm_sum_of_numbers_in_ratio_l328_328404

theorem lcm_sum_of_numbers_in_ratio (x : ℤ) (h1 : Nat.lcm (2 * x) (3 * x) = 48) : 
  (2 * x) + (3 * x) = 40 := 
by sorry

end lcm_sum_of_numbers_in_ratio_l328_328404


namespace sum_of_abs_a_l328_328921

variable {n : ℕ}

def S (n : ℕ) : ℕ := n^2 - 6 * n
def a (n : ℕ) : ℤ := 2 * n - 7

theorem sum_of_abs_a (n : ℕ) : 
  (T n = if n ≤ 3 then 6 * n - n^2 else n^2 - 6 * n + 18) :=
  sorry

end sum_of_abs_a_l328_328921


namespace number_of_valid_lattice_points_l328_328134

/-- A lattice point is a point in the plane with integer coordinates. -/
def is_lattice_point (p : ℤ × ℤ) := p.1 ∈ ℤ ∧ p.2 ∈ ℤ

/-- A point on the line segment with endpoints (5, 23) and (73, 431) where y is even -/
def is_valid_lattice_point (p : ℤ × ℤ) : Prop :=
  let t := (p.1 - 5)
  p.2 = 23 + 6 * t ∧ p.1 ≥ 5 ∧ p.1 ≤ 73 ∧ p.2 % 2 = 0

theorem number_of_valid_lattice_points : {p : ℤ × ℤ | is_valid_lattice_point p}.finite.to_finset.card = 34 :=
sorry

end number_of_valid_lattice_points_l328_328134


namespace smallest_range_is_D_l328_328594

def setA : List ℕ := [13, 15, 11, 12, 15, 11, 15]
def setB : List ℕ := [6, 9, 8, 7, 9, 9, 8, 5, 4]
def setC : List ℕ := [5, 4, 5, 7, 1, 7, 8, 7, 4]
def setD : List ℕ := [17, 11, 10, 9, 5, 4, 4, 3]

def range (s : List ℕ) : ℕ := s.maximum? - s.minimum?

theorem smallest_range_is_D :
  range setD < range setA ∧ range setD < range setB ∧ range setD < range setC :=
by
  -- Proof omitted.
  sorry

end smallest_range_is_D_l328_328594


namespace option_a_option_b_option_d_option_c_false_l328_328569

variable (m n : ℝ)
variable (h1 : m > n)
variable (h2 : n > 0)

theorem option_a : sqrt(m^2 + m) > sqrt(n^2 + n) :=
sorry

theorem option_b : m - n > sin m - sin n :=
sorry

theorem option_d : (e^m - e^n) / (m + n) > m - n :=
sorry

theorem option_c_false : |log m| ≤ |log n| :=
sorry

end option_a_option_b_option_d_option_c_false_l328_328569


namespace unique_zero_location_l328_328591

theorem unique_zero_location (f : ℝ → ℝ) (h : ∃! x, f x = 0 ∧ 1 < x ∧ x < 3) :
  ¬ (∃ x, 2 < x ∧ x < 5 ∧ f x = 0) :=
sorry

end unique_zero_location_l328_328591


namespace conclusion1_conclusion2_conclusion3_l328_328285

-- Define the Δ operation
def delta (m n : ℚ) : ℚ := (m + n) / (1 + m * n)

-- 1. Proof that (-2^2) Δ 4 = 0
theorem conclusion1 : delta (-4) 4 = 0 := sorry

-- 2. Proof that (1/3) Δ (1/4) = 3 Δ 4
theorem conclusion2 : delta (1/3) (1/4) = delta 3 4 := sorry

-- 3. Proof that (-m) Δ n = m Δ (-n)
theorem conclusion3 (m n : ℚ) : delta (-m) n = delta m (-n) := sorry

end conclusion1_conclusion2_conclusion3_l328_328285


namespace integral_value_l328_328290

noncomputable def coefficient_of_second_term (a : ℝ) : ℝ :=
  3 * a^2 * (- (Real.sqrt 3) / 6)

noncomputable def integral (a : ℝ) : ℝ :=
  ∫ x in -2..a, x^2

theorem integral_value (a: ℝ) (h: coefficient_of_second_term a = - (Real.sqrt 3) / 2) :
  integral a = 3 ∨ integral a = 7 / 3 := by
  sorry

end integral_value_l328_328290


namespace necklace_amethyst_beads_l328_328800

theorem necklace_amethyst_beads:
  ∀ (A : ℕ),
    let amber := 2 * A in
    let turquoise := 19 in
    A + amber + turquoise = 40 →
    A = 7 :=
by
  sorry

end necklace_amethyst_beads_l328_328800


namespace sin_squared_roots_poly_cotg_squared_roots_poly_l328_328779

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sin_squared_roots_poly (n : ℕ) (n_pos : 0 < n) :
  ∀ k, 1 ≤ k ∧ k ≤ n → 
  is_root (λ x, (binomial_coefficient (2*n+1) 1) * (1 - x)^n 
                  - (binomial_coefficient (2*n+1) 3) * (1 - x)^(n-1) * x 
                  + (binomial_coefficient (2*n+1) 5) * (1 - x)^(n-2) * x^2 
                  -  ... + (-1)^n * x^n)
  (Real.sin k * (Real.pi / (2*n+1))^2) := sorry

theorem cotg_squared_roots_poly (n : ℕ) (n_pos : 0 < n) :
  ∀ k, 1 ≤ k ∧ k ≤ n →
  is_root (λ x, (binomial_coefficient (2*n+1) 1) * x^n 
                  - (binomial_coefficient (2*n+1) 3) * x^(n-1) 
                  + (binomial_coefficient (2*n+1) 5) * x^(n-2) 
                  - ... + (-1)^n)
  (Real.cot k * (Real.pi / (2*n+1))^2) := sorry

end sin_squared_roots_poly_cotg_squared_roots_poly_l328_328779


namespace valid_outfits_l328_328615

-- Let's define the conditions first:
variable (shirts colors pairs : ℕ)

-- Suppose we have the following constraints according to the given problem:
def totalShirts : ℕ := 6
def totalPants : ℕ := 6
def totalHats : ℕ := 6
def totalShoes : ℕ := 6
def numOfColors : ℕ := 6

-- We refuse to wear an outfit in which all 4 items are the same color, or in which the shoes match the color of any other item.
theorem valid_outfits : 
  (totalShirts * totalPants * totalHats * (totalShoes - 1) + (totalShirts * 5 - totalShoes)) = 1104 :=
by sorry

end valid_outfits_l328_328615


namespace number_of_correct_propositions_is_1_l328_328593

-- Define the propositions
def prop1 : Prop := ∀ (solid : Type), (solid.has_two_parallel_faces ∧ solid.all_other_faces_parallelograms) → solid.is_prism
def prop2 : Prop := ∀ (prism : Type), (prism.has_two_lateral_faces_perpendicular_to_base) → prism.is_right_prism
def prop3 : Prop := ∀ (slant_prism : Type), ¬(slant_prism.can_obtain_rectangle_by_cutting_along_lateral_edges)
def prop4 : Prop := ∀ (quad_prism : Type), (quad_prism.lateral_faces_congruent_rectangles) → quad_prism.is_regular

-- Problem statement: Prove that the number of correct propositions is 1
theorem number_of_correct_propositions_is_1 :
  (prop1 = false) ∧ (prop2 = false) ∧ (prop3 = true) ∧ (prop4 = false) → (number_of_correct_propositions = 1) := by
  sorry

end number_of_correct_propositions_is_1_l328_328593


namespace laps_needed_l328_328493

theorem laps_needed (r1 r2 : ℕ) (laps1 : ℕ) (h1 : r1 = 30) (h2 : r2 = 10) (h3 : laps1 = 40) : 
  (r1 * laps1) / r2 = 120 := by
  sorry

end laps_needed_l328_328493


namespace articles_produced_l328_328282

variable (x y z : ℝ)

-- Condition 1: x men, x hours/day, x days produce x articles
def initial_production (x : ℝ) : ℝ :=
  x

-- Condition 2: Productivity per man decreases by a factor of z for each additional worker
def productivity_factor (x z : ℝ) : ℝ :=
  x / z

-- Prove that the number of articles produced by y men, y hours/day, y days is y³ / (x * z)
theorem articles_produced (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (y * y * y) / (x * z) = y^3 / (x * z) :=
by
  sorry

end articles_produced_l328_328282


namespace find_y_l328_328986

theorem find_y (x y : ℝ) (h1 : 0.5 * x = 0.25 * y - 30) (h2 : x = 690) : y = 1500 :=
by
  sorry

end find_y_l328_328986


namespace inequality_proof_l328_328700

-- Definition of ω(n) - number of distinct prime factors
def ω(n : ℕ) : ℕ :=
  (nat.factors n).to_finset.card

-- Definition of Ω(n) - total number of prime factors (counting multiplicities)
def Ω(n : ℕ) : ℕ :=
  (nat.factors n).length

-- Definition of τ(n) - number of divisors
def τ(n : ℕ) : ℕ :=
  nat.divisors n |>.length

-- Statement of the theorem
theorem inequality_proof (n : ℕ) :
  ∑ m in finset.range (n+1), 5 ^ ω m
  ≤ ∑ k in finset.range (n+1), (n / k) * (τ k)^2
  ∧ ∑ k in finset.range (n+1), (n / k) * (τ k)^2
  ≤ ∑ m in finset.range (n+1), 5 ^ Ω m :=
sorry

end inequality_proof_l328_328700


namespace missing_weights_l328_328431

theorem missing_weights :
  ∃ (n k : ℕ), (n > 10) ∧ (606060 % 8 = 4) ∧ (606060 % 9 = 0) ∧ 
  (5 * k + 24 * k + 43 * k = 606060 + 72 * n) :=
sorry

end missing_weights_l328_328431


namespace perfect_squares_suitable_factorials_suitable_no_squares_l328_328509

-- Define what it means for a set to be suitable in Lean
def is_suitable (A : set ℕ) : Prop :=
∀ n > 0, ∀ p q : ℕ, p.prime → q.prime → (n - p) ∈ A → (n - q) ∈ A → p = q

-- Theorem 1: The set of perfect squares is suitable
theorem perfect_squares_suitable : is_suitable {n : ℕ | ∃ k : ℕ, n = k * k} :=
sorry

-- Theorem 2: An infinite suitable set containing no perfect squares
theorem factorials_suitable_no_squares : ∃ (A : set ℕ), infinite A ∧ is_suitable A ∧ (∀ n ∈ A, ∃ k ≥ 2, n = k!) ∧ ∀ n ∈ A, ¬ ∃ m, n = m * m :=
sorry

end perfect_squares_suitable_factorials_suitable_no_squares_l328_328509


namespace simplify_expression_l328_328393

theorem simplify_expression (x : ℝ) :
  (5 / (4 * x^(-4)) * (4 * x^3) / 3) / (x / 2) = 10 * x^6 / 3 :=
by sorry

end simplify_expression_l328_328393


namespace f_7_5_l328_328670

-- Defining the odd function property
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f ( -x ) = - f x 

-- Definitions given in the problem conditions
noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then x else
if 1 < x ∧ x ≤ 2 then -(f (x-2)) else
- f ( x -2 )

-- Conditions as given
axiom f_is_odd : is_odd f 
axiom f_periodicity : ∀ x, f (x+2) = -f x 

-- Theorem to prove that f(7.5) = -0.5
theorem f_7_5 : f 7.5 = -0.5 := sorry

end f_7_5_l328_328670


namespace shot_radius_l328_328976

theorem shot_radius (r_original : ℝ) (n : ℕ) (V_original V_shot : ℝ)
    (h1 : r_original = 7)
    (h2 : n = 343)
    (h3 : V_original = (4 / 3) * real.pi * r_original ^ 3)
    (h4 : V_shot = (4 / 3) * real.pi * r_shot ^ 3)
    (h5 : V_original = n * V_shot) :
    r_shot = real.cbrt (1 / 3) :=
by
  sorry

end shot_radius_l328_328976


namespace perimeter_of_garden_l328_328138

-- Definitions based on conditions
def length : ℕ := 150
def breadth : ℕ := 150
def is_square (l b : ℕ) := l = b

-- Theorem statement proving the perimeter given conditions
theorem perimeter_of_garden : is_square length breadth → 4 * length = 600 :=
by
  intro h
  rw [h]
  norm_num
  sorry

end perimeter_of_garden_l328_328138


namespace continuous_function_identity_l328_328555

/- Define the problem statement -/
theorem continuous_function_identity 
  (f : ℝ → ℝ) 
  (h_cont : ∀ x, ContinuousAt f x) 
  (h_func_eq : ∀ x y, 3 * f (x + y) = f x * f y) 
  (h_val : f 1 = 12) :
  ∀ x, f x = 3 * 4^x := 
sorry

end continuous_function_identity_l328_328555


namespace algebraic_expression_value_l328_328296

theorem algebraic_expression_value (x : ℝ) 
  (h : 2 * x^2 + 3 * x + 7 = 8) : 
  4 * x^2 + 6 * x - 9 = -7 := 
by 
  sorry

end algebraic_expression_value_l328_328296


namespace min_rectilinear_distance_l328_328645

noncomputable def rectilinear_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

theorem min_rectilinear_distance : ∀ (M : ℝ × ℝ), (M.1 - M.2 + 4 = 0) → rectilinear_distance (1, 1) M ≥ 4 :=
by
  intro M hM
  -- We only need the statement, not the proof
  sorry

end min_rectilinear_distance_l328_328645


namespace find_angle_B_l328_328655

theorem find_angle_B (A B C : ℝ) (a b c : ℝ)
  (hAngleA : A = 120) (ha : a = 2) (hb : b = 2 * Real.sqrt 3 / 3) : B = 30 :=
sorry

end find_angle_B_l328_328655


namespace circle_definition_l328_328772

theorem circle_definition (P : Type) [metric_space P] (center : P) (radius : ℝ) :
  set_of (λ p : P, dist p center = radius) = {p | dist p center = radius} :=
by
  sorry

end circle_definition_l328_328772


namespace largest_possible_d_plus_r_l328_328745

theorem largest_possible_d_plus_r :
  ∃ d r : ℕ, 0 < d ∧ 468 % d = r ∧ 636 % d = r ∧ 867 % d = r ∧ d + r = 27 := by
  sorry

end largest_possible_d_plus_r_l328_328745


namespace sin_cos_over_add_sin_cos_eq_one_seventh_l328_328955

variables (a θ : ℝ) (A : ℝ × ℝ)
hypothesis h_a_pos : a > 0
hypothesis h_a_ne_one : a ≠ 1
hypothesis h_A_coords : A = (3, 4)
hypothesis h_A_on_graph : ∃ x, x = 3 ∧ (a ^ (x - 3) + x = 4)
hypothesis h_tan_theta : Real.tan θ = 4 / 3

theorem sin_cos_over_add_sin_cos_eq_one_seventh :
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 / 7 := by
  sorry

end sin_cos_over_add_sin_cos_eq_one_seventh_l328_328955


namespace solve_for_x_l328_328990

theorem solve_for_x (x : ℝ) (h : 40 / x - 1 = 19) : x = 2 :=
by {
  sorry
}

end solve_for_x_l328_328990


namespace speed_of_B_l328_328508

theorem speed_of_B 
  (A_speed : ℝ)
  (t1 : ℝ)
  (t2 : ℝ)
  (d1 := A_speed * t1)
  (d2 := A_speed * t2)
  (total_distance := d1 + d2)
  (B_speed := total_distance / t2) :
  A_speed = 7 → 
  t1 = 0.5 → 
  t2 = 1.8 →
  B_speed = 8.944 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  exact sorry

end speed_of_B_l328_328508


namespace log_function_pass_through_point_l328_328956

theorem log_function_pass_through_point (a m n : ℝ) : 
  (∀ x ∈ set.Ioc (-1) 1, n = -2) → 
  (∀ x ∈ set.Ioc (-1) 1, x + m = 1) → 
  m * n = -4 := 
by 
  sorry

end log_function_pass_through_point_l328_328956


namespace sum_of_T_l328_328366

def is_valid_abcd (a b c d : ℕ) : Prop :=
  (0 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (0 ≤ d ∧ d ≤ 9) ∧
  (∀ i j k l, (i, j, k, l) ∈ {(a, b, c, d)} → 
    (i = j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l) ∨ 
    (i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l))

def T : Set ℝ := { x : ℝ | ∃ a b c d : ℕ, is_valid_abcd a b c d ∧ x = (a * 1000 + b * 100 + c * 10 + d) / 9999 }

theorem sum_of_T : ∑ x in T, x = 504 :=
by
  sorry

end sum_of_T_l328_328366


namespace find_k_l328_328789

theorem find_k (k t : ℝ) (h1 : t = 5) (h2 : (1/2) * (t^2) / ((k-1) * (k+1)) = 10) : 
  k = 3/2 := 
  sorry

end find_k_l328_328789


namespace geometric_seq_ratio_l328_328364

theorem geometric_seq_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 5 - a 3 = 12) 
  (h2 : a 6 - a 4 = 24) 
  (ha : ∃ a₁ : ℝ, (∀ n : ℕ, a n = a₁ * q ^ (n - 1)))
  (hS : ∀ n : ℕ, S n = a₁ * (1 - q ^ n) / (1 - q)) :
  ∀ n : ℕ, S n / a n = 2 - 2 ^ (1 - n) :=
sorry

end geometric_seq_ratio_l328_328364


namespace trig_identity_third_quadrant_l328_328619

theorem trig_identity_third_quadrant (α : ℝ) (h1 : sin α < 0) (h2 : cos α < 0) :
  (cos α / (sqrt (1 - sin α ^ 2))) + (sin α / (sqrt (1 - cos α ^ 2))) = -2 := 
sorry

end trig_identity_third_quadrant_l328_328619


namespace min_fraction_sum_l328_328307

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ n, a n = a 1 * q ^ (n - 1)

theorem min_fraction_sum {a : ℕ → ℝ} {m n : ℕ} (h_pos : ∀ k, a k > 0)
    (h_geom : geometric_sequence a q) (h_sqrt : sqrt (a m * a n) = 4 * a 1)
    (h_a6 : a 6 = a 5 + 2 * a 4) :
    m + n = 6 → (1 / m : ℝ) + (4 / n) = 3 / 2 :=
by {
  sorry
}

end min_fraction_sum_l328_328307


namespace geometric_sequence_ratio_l328_328346

variable {a : ℕ → ℕ}
variables (S : ℕ → ℕ) (n : ℕ)
def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, ∀ n : ℕ, a n = a 1 * q^n

variables (h1 : a 5 - a 3 = 12) (h2 : a 6 - a 4 = 24)

theorem geometric_sequence_ratio :
  is_geometric_sequence a →
  ∃ S : ℕ → ℕ, (S n = ∑ i in range n, a i) ∧
  (S n / a n = 2 - 2^(1 - n)) :=
by
  sorry

end geometric_sequence_ratio_l328_328346


namespace vertex_of_parabola_l328_328413

-- Define the equation of the parabola
def parabola_eq (y x : ℝ) : Prop :=
  y^2 - 4*y + 2*x + 8 = 0

-- Prove the vertex of the parabola is (-2, 2)
theorem vertex_of_parabola : ∃ (h k : ℝ), 
  (∀ (y x : ℝ), parabola_eq y x ↔ (x = -0.5 * (y - k)^2 + h)) ∧ 
  h = -2 ∧ 
  k = 2 :=
begin
  sorry
end

end vertex_of_parabola_l328_328413


namespace pebble_sequence_10th_image_l328_328842

def a : ℕ → ℕ
| 1       := 1
| 2       := 5
| 3       := 12
| 4       := 22
| (n + 1) := a n + 3 (n + 1) - 2

theorem pebble_sequence_10th_image :
  a 10 = 145 :=
sorry

end pebble_sequence_10th_image_l328_328842


namespace runs_in_last_match_l328_328495

-- Definitions based on the conditions
def initial_bowling_average : ℝ := 12.4
def wickets_last_match : ℕ := 7
def decrease_average : ℝ := 0.4
def new_average : ℝ := initial_bowling_average - decrease_average
def approximate_wickets_before : ℕ := 145

-- The Lean statement of the problem
theorem runs_in_last_match (R : ℝ) :
  ((initial_bowling_average * approximate_wickets_before + R) / 
   (approximate_wickets_before + wickets_last_match) = new_average) →
   R = 28 :=
by
  sorry

end runs_in_last_match_l328_328495


namespace not_concave_functions_l328_328947

noncomputable def f1 (x : ℝ) : ℝ := sin x + cos x
noncomputable def f2 (x : ℝ) : ℝ := log x - 2 * x
noncomputable def f3 (x : ℝ) : ℝ := -x^3 + 2 * x - 1
noncomputable def f4 (x : ℝ) : ℝ := x * exp x

def f1_second_derivative_is_negative_on_interval : Prop :=
  ∀ (x : ℝ), 0 < x ∧ x < (π / 2) → (derivative (derivative f1 x)) < 0

def f2_second_derivative_is_negative_on_interval : Prop :=
  ∀ (x : ℝ), 0 < x ∧ x < (π / 2) → (derivative (derivative f2 x)) < 0

def f3_second_derivative_is_negative_on_interval : Prop :=
  ∀ (x : ℝ), 0 < x ∧ x < (π / 2) → (derivative (derivative f3 x)) < 0

def f4_second_derivative_is_positive_on_interval : Prop :=
  ∀ (x : ℝ), 0 < x ∧ x < (π / 2) → (derivative (derivative f4 x)) > 0

theorem not_concave_functions :
  ¬(f4_second_derivative_is_positive_on_interval → f1_second_derivative_is_negative_on_interval ∧ 
   f2_second_derivative_is_negative_on_interval ∧ f3_second_derivative_is_negative_on_interval) :=
sorry

end not_concave_functions_l328_328947


namespace A_inter_B_empty_l328_328261

def setA : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def setB : Set ℝ := {x | Real.log x / Real.log 4 > 1/2}

theorem A_inter_B_empty : setA ∩ setB = ∅ := by
  sorry

end A_inter_B_empty_l328_328261


namespace triangle_max_area_proof_l328_328757

noncomputable def max_triangle_area (D E F : Type*) [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (dE : D → E → ℝ) (fEqD : ℝ) (ratioEF_DF : ℝ × ℝ) : ℝ := 1584.375

theorem triangle_max_area_proof (D E F : Type*) [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (dE : D → E → ℝ) (dED : dE D E = 15) (ratioEF_DF : (ℝ × ℝ) := (25, 24)) :
  ∃ (x : ℝ), max_triangle_area D E F dE dED ratioEF_DF = 1584.375 :=
begin
  sorry
end

end triangle_max_area_proof_l328_328757


namespace solution_of_system_l328_328434

theorem solution_of_system : ∃ x y : ℝ, (2 * x + y = 2) ∧ (x - y = 1) ∧ (x = 1) ∧ (y = 0) := 
by
  sorry

end solution_of_system_l328_328434


namespace election_votes_l328_328092

theorem election_votes (V : ℕ) 
  (h1 : 0.60 * V = 0.40 * V + 900) : 
  V = 4500 :=
by
  -- Applying the given majority condition
  -- Calculating V given the majority vote difference 900
  sorry

end election_votes_l328_328092


namespace sequences_satisfy_conditions_l328_328195

noncomputable def find_sequences : (ℕ → ℝ) × (ℕ → ℝ) :=
  let b : ℕ → ℝ := λ n, 0
  let c : ℕ → ℝ := λ n, 0
  (b, c)

theorem sequences_satisfy_conditions (b : ℕ → ℝ) (c : ℕ → ℝ) : 
  (∀ n, b n ≤ c n) →
  (∀ n, ∀ x : ℝ, x^2 + b n * x + c n = 0 → x = b (n + 1) ∨ x = c (n + 1)) →
  (b = λ n, 0) ∧ (c = λ n, 0) :=
begin
  assume hb hc,
  sorry
end

end sequences_satisfy_conditions_l328_328195


namespace arrangement_count_l328_328643

-- Definitions based on the conditions
def digit_set : Multiset ℕ := {6, 0, 0, 6, 3}
def valid_positions : Finset (Fin 5) := {1, 2, 3}

-- Main problem statement
theorem arrangement_count : Multiset.countP (λ n, ¬(n = 0)) 
  (Multiset.filter (λ n : ℕ, n ≠ 0) digit_set).card = 9 := 
  sorry

end arrangement_count_l328_328643


namespace total_voters_l328_328311

theorem total_voters (x : ℝ)
  (h1 : 0.35 * x + 80 = (0.35 * x + 80) + 0.65 * x - (0.65 * x - 0.45 * (x + 80)))
  (h2 : 0.45 * (x + 80) = 0.65 * x) : 
  x + 80 = 260 := by
  -- We'll provide the proof here
  sorry

end total_voters_l328_328311


namespace original_population_l328_328145

theorem original_population (n : ℕ) (h1 : n + 1500 * 85 / 100 = n - 45) : n = 8800 := 
by
  sorry

end original_population_l328_328145


namespace hotel_loss_l328_328128

theorem hotel_loss :
  (ops_expenses : ℝ) (payment_frac : ℝ) (total_received : ℝ) (loss : ℝ)
  (h_ops_expenses : ops_expenses = 100)
  (h_payment_frac : payment_frac = 3 / 4)
  (h_total_received : total_received = payment_frac * ops_expenses)
  (h_loss : loss = ops_expenses - total_received) :
  loss = 25 :=
by
  sorry

end hotel_loss_l328_328128


namespace hotel_loss_l328_328130

variable (operations_expenses : ℝ)
variable (fraction_payment : ℝ)

theorem hotel_loss :
  operations_expenses = 100 →
  fraction_payment = 3 / 4 →
  let total_payment := fraction_payment * operations_expenses in
  let loss := operations_expenses - total_payment in
  loss = 25 :=
by
  intros h₁ h₂
  have tstp : total_payment = 75 := by
    rw [h₁, h₂]
    norm_num
  have lss : loss = 25 := by
    rw [h₁, tstp]
    norm_num
  exact lss

end hotel_loss_l328_328130


namespace expression_equals_neg_one_l328_328859

theorem expression_equals_neg_one (b y : ℝ) (hb : b ≠ 0) (h₁ : y ≠ b) (h₂ : y ≠ -b) :
  ( (b / (b + y) + y / (b - y)) / (y / (b + y) - b / (b - y)) ) = -1 :=
sorry

end expression_equals_neg_one_l328_328859


namespace Q_subset_P_l328_328265

-- Definitions from conditions
def P : Set ℝ := {x | x^2 ≠ 4}
def Q (a : ℝ) : Set ℝ := {x | a * x = 4}

-- The statement to be proven in Lean
theorem Q_subset_P (a : ℝ) : Q a ⊆ P ↔ a ∈ {0, 2, -2} :=
by
  sorry

end Q_subset_P_l328_328265


namespace factor_z4_minus_81_l328_328876

theorem factor_z4_minus_81 :
  (z^4 - 81) = (z - 3) * (z + 3) * (z^2 + 9) :=
by
  sorry

end factor_z4_minus_81_l328_328876


namespace fraction_of_primes_is_prime_l328_328374

theorem fraction_of_primes_is_prime
  (p q r : ℕ) 
  (hp : Nat.Prime p)
  (hq : Nat.Prime q)
  (hr : Nat.Prime r)
  (h : ∃ k : ℕ, p * q * r = k * (p + q + r)) :
  Nat.Prime (p * q * r / (p + q + r)) := 
sorry

end fraction_of_primes_is_prime_l328_328374


namespace height_of_isosceles_trapezoid_l328_328044

-- Definitions based on the conditions
def is_isosceles_trapezoid (ABCD : Type) (A B C D : ABCD) := sorry -- define isosceles trapezoid
def diagonal_length (A C : Point) : Real := sorry -- length of AC is 10
def angle_with_base (A B C : Point) : Real := sorry -- angle BAC is 60 degrees

-- Main statement to prove
theorem height_of_isosceles_trapezoid
  (ABCD : Type) [isosceles_trapezoid : is_isosceles_trapezoid ABCD A B C D]
  (A B C : Point)
  (h_diagonal: diagonal_length A C = 10)
  (h_angle: angle_with_base A B C = 60) :
  ∃ h : ℝ, h = 5 * (Real.sqrt 3) :=
sorry

end height_of_isosceles_trapezoid_l328_328044


namespace remaining_insects_is_twenty_one_l328_328711

-- Define the initial counts of each type of insect
def spiders := 3
def ants := 12
def ladybugs := 8

-- Define the number of ladybugs that flew away
def ladybugs_flew_away := 2

-- Define the total initial number of insects
def total_insects_initial := spiders + ants + ladybugs

-- Define the total number of insects that remain after some ladybugs fly away
def total_insects_remaining := total_insects_initial - ladybugs_flew_away

-- Theorem statement: proving that the number of insects remaining is 21
theorem remaining_insects_is_twenty_one : total_insects_remaining = 21 := sorry

end remaining_insects_is_twenty_one_l328_328711


namespace pears_thrown_away_on_first_day_l328_328827

theorem pears_thrown_away_on_first_day (x : ℝ) (P : ℝ) 
  (h1 : P > 0)
  (h2 : 0.8 * P = P * 0.8)
  (total_thrown_percentage : (x / 100) * 0.2 * P + 0.2 * (1 - x / 100) * 0.2 * P = 0.12 * P ) : 
  x = 50 :=
by
  sorry

end pears_thrown_away_on_first_day_l328_328827


namespace minimum_area_of_triangle_l328_328939

def parabola_focus : Prop :=
  ∃ F : ℝ × ℝ, F = (1, 0)

def on_parabola (A B : ℝ × ℝ) : Prop :=
  (A.2 ^ 2 = 4 * A.1 ∧ B.2 ^ 2 = 4 * B.1) ∧ (A.2 * B.2 < 0)

def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = -4

noncomputable def area (A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 - B.1 * A.2)

theorem minimum_area_of_triangle
  (A B : ℝ × ℝ)
  (h_focus : parabola_focus)
  (h_on_parabola : on_parabola A B)
  (h_dot : dot_product_condition A B) :
  ∃ C : ℝ, C = 4 * Real.sqrt 2 ∧ area A B = C :=
by
  sorry

end minimum_area_of_triangle_l328_328939


namespace ratio_perimeter_width_l328_328425

-- Define the conditions as constants
constant (A : ℝ) (l : ℝ) (w : ℝ) (P : ℝ)
constant (hA : A = 150)
constant (hl : l = 15)

-- Define supplementary hypotheses based on given information
noncomputable def width : ℝ := A / l
noncomputable def perimeter : ℝ := 2 * (l + width)

-- Statement of the theorem
theorem ratio_perimeter_width : (P / w = 5) :=
by
  -- Definitions based on conditions
  let w := width
  let P := perimeter
  -- Verify definitions, ensuring Lean accepts them
  have hw: A = l * w := by
    sorry  -- prove or assume A = l * w

  have hP : P = 2 * (l + w) := by
    sorry  -- prove or assume P = 2 * (l + w)

  -- Main proof statement
  sorry -- prove the ratio P / w = 5 (this is where the actual Lean proof would go)

end ratio_perimeter_width_l328_328425


namespace weight_in_pounds_l328_328753

theorem weight_in_pounds (weight_kg : ℝ) (conversion_factor : ℝ) : 
  weight_kg = 300 → conversion_factor = 0.454 → 
  round (weight_kg / conversion_factor) = 661 :=
begin
  -- Assume conditions from the problem
  intros h_weight_kg h_conversion_factor,
  -- Skip the proof for now
  sorry
end

end weight_in_pounds_l328_328753


namespace perpendicular_lines_k_value_l328_328629

variable (k : ℝ)

def line1_perpendicular_to_line2 : Prop :=
  let l1 := (k - 3) * x + (k + 4) * y + 1 = 0
  let l2 := (k + 1) * x + 2 * (k - 3) * y + 3 = 0
  ∃ (mx my : ℝ), (y = mx * x + b -- where b is some constant) ∧ (line equality) ∧ 
                  (slope1 * slope 2(so∃ k), mx.my=0 and rearrange left sides if necessary) 

theorem perpendicular_lines_k_value (k : ℝ) (h : line1_perpendicular_to_line2 k) : 
  k = 3 ∨ k = -3 := 
 sorry

end perpendicular_lines_k_value_l328_328629


namespace exercise_time_l328_328330

theorem exercise_time :
  let time_monday := 6 / 2 in
  let time_wednesday := 6 / 3 in
  let time_friday := 6 / 6 in
  let total_time := time_monday + time_wednesday + time_friday in
  total_time = 6 :=
by
  sorry

end exercise_time_l328_328330


namespace factorial_sum_not_1990_end_l328_328532

theorem factorial_sum_not_1990_end (m n : ℕ) : (m! + n!) % 100 ≠ 90 := 
by sorry

end factorial_sum_not_1990_end_l328_328532


namespace solution_parking_problem_l328_328500

def parking_problem : Prop :=
  ∃ (x : ℕ), 
    let spaces1 := x in
    let spaces2 := x + 8 in
    let spaces3 := x + 20 in
    let spaces4 := x + 11 in
    let total_spaces := spaces1 + spaces2 + spaces3 + spaces4 in
    total_spaces - 100 = 299 ∧ spaces1 = 40

theorem solution_parking_problem : parking_problem :=
  sorry

end solution_parking_problem_l328_328500


namespace professor_oscar_review_questions_l328_328837

-- Define the problem conditions.
def students_per_class : ℕ := 35
def questions_per_exam : ℕ := 10
def number_of_classes : ℕ := 5

-- Define the number of questions that must be reviewed.
def total_questions_to_review : ℕ := 1750

-- The theorem to be proved.
theorem professor_oscar_review_questions :
  students_per_class * questions_per_exam * number_of_classes = total_questions_to_review :=
by
  -- Here we write 'sorry' since we are not providing the full proof.
  sorry

end professor_oscar_review_questions_l328_328837


namespace complex_number_powers_l328_328674

theorem complex_number_powers (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^97 + z^98 + z^99 + z^100 + z^101 = -1 :=
sorry

end complex_number_powers_l328_328674


namespace percentage_of_towns_correct_l328_328497

def percentage_of_towns_with_fewer_than_50000_residents (p1 p2 p3 : ℝ) : ℝ :=
  p1 + p2

theorem percentage_of_towns_correct (p1 p2 p3 : ℝ) (h1 : p1 = 0.15) (h2 : p2 = 0.30) (h3 : p3 = 0.55) :
  percentage_of_towns_with_fewer_than_50000_residents p1 p2 p3 = 0.45 :=
by 
  sorry

end percentage_of_towns_correct_l328_328497


namespace geometric_sum_ratio_l328_328354

-- Definitions and Conditions
variables {a : ℕ → ℕ}
variable q : ℕ
variable n : ℕ

-- Condition a₅ - a₃ = 12
axiom h1 : a 5 - a 3 = 12

-- Condition a₆ - a₄ = 24
axiom h2 : a 6 - a 4 = 24

-- The goal to prove: Sₙ / aₙ = 2 - 2⁽¹⁻ⁿ⁾, where Sₙ is the sum of first n terms and aₙ is the nth term
theorem geometric_sum_ratio (S : ℕ → ℕ) (a_n : ℕ) : 
  (S n) / (a n) = 2 - 2^(1 - n) := 
sorry

end geometric_sum_ratio_l328_328354


namespace Jonathan_total_exercise_time_l328_328334

theorem Jonathan_total_exercise_time :
  (let monday_speed := 2
     let wednesday_speed := 3
     let friday_speed := 6
     let distance := 6
     let monday_time := distance / monday_speed
     let wednesday_time := distance / wednesday_speed
     let friday_time := distance / friday_speed
   in monday_time + wednesday_time + friday_time = 6)
:= sorry

end Jonathan_total_exercise_time_l328_328334


namespace ball_distribution_l328_328699

/-- Given 15 balls numbered from 1 to 15, conditions on distribution into three plates A, B, and C -/
theorem ball_distribution :
  ∃ (A B C : Finset ℕ), 
    (∀ n, n ∈ A → n ∈ (Finset.range 16)), 
    (∀ n, n ∈ B → n ∈ (Finset.range 16)),
    (∀ n, n ∈ C → n ∈ (Finset.range 16)),
    A.card ≥ 4 ∧ B.card ≥ 4 ∧ C.card ≥ 4 ∧
    (A.sum id : ℚ) / A.card = 3 ∧
    (B.sum id : ℚ) / B.card = 8 ∧
    (C.sum id : ℚ) / C.card = 13 ∧
    (A ∪ B ∪ C = Finset.range 16) ∧ (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
    (A.card + B.card + C.card = 15) ∧
    ∃ (possible_set_A : Finset ℕ), 
        possible_set_A = {1, 2, 4, 5} ∧
        possible_set_A ⊆ A ∧ (
        B.card = 7 ∨ 
        B.card = 5) :=
begin
  sorry
end

end ball_distribution_l328_328699


namespace slope_angle_l328_328050

theorem slope_angle (A B : ℝ × ℝ) (θ : ℝ) (hA : A = (-1, 3)) (hB : B = (1, 1)) (hθ : θ ∈ Set.Ico 0 Real.pi)
  (hslope : Real.tan θ = (B.2 - A.2) / (B.1 - A.1)) :
  θ = (3 / 4) * Real.pi :=
by
  cases hA
  cases hB
  simp at hslope
  sorry

end slope_angle_l328_328050


namespace find_eccentricity_l328_328045

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (d : ℝ) := 
  d = b / 2 → (2 : ℝ) = 2

theorem find_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  hyperbola_eccentricity a b h1 h2 (b / 2) :=
begin
  sorry
end

end find_eccentricity_l328_328045


namespace range_of_m_l328_328858

noncomputable def set_M (m : ℝ) : Set ℝ := {x | x < m}
noncomputable def set_N : Set ℝ := {y | ∃ (x : ℝ), y = Real.log x / Real.log 2 - 1 ∧ 4 ≤ x}

theorem range_of_m (m : ℝ) : set_M m ∩ set_N = ∅ → m < 1 
:= by
  sorry

end range_of_m_l328_328858


namespace courtyard_length_is_18_l328_328114

-- Define the given conditions as assumptions.

def courtyard_width : ℝ := 16   -- The width of the courtyard is 16 meters

def brick_length : ℝ := 0.20    -- The length of the brick is 0.20 meters
def brick_width : ℝ := 0.10     -- The width of the brick is 0.10 meters
def number_of_bricks : ℝ := 14400 -- The total number of bricks required is 14400

-- Define the area of one brick.
def brick_area : ℝ := brick_length * brick_width

-- Define the total area covered by all bricks.
def total_area : ℝ := number_of_bricks * brick_area

-- Define the length of the courtyard.
def courtyard_length (width : ℝ) (area : ℝ) : ℝ := area / width

-- The theorem to prove that the courtyard is 18 meters long.
theorem courtyard_length_is_18 :
  courtyard_length courtyard_width total_area = 18 :=
by
  -- The proof will be added here.
  sorry

end courtyard_length_is_18_l328_328114


namespace chord_intersection_probability_l328_328561

theorem chord_intersection_probability
  (points : Finset Point)
  (hp : points.card = 2000)
  (A B C D E : Point)
  (hA : A ∈ points)
  (hB : B ∈ points)
  (hC : C ∈ points)
  (hD : D ∈ points)
  (hE : E ∈ points)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  : probability_chord_intersection := by
    sorry

end chord_intersection_probability_l328_328561


namespace squares_in_21st_figure_l328_328507

theorem squares_in_21st_figure : ∀ n : ℕ, (S n = n^2 + (n - 1)^2) → S 21 = 841 :=
by
  intros n h
  sorry

end squares_in_21st_figure_l328_328507


namespace ellipse_standard_eq_hyperbola_eq_l328_328103

-- Define the conditions for the ellipse problem
def ellipse_eccentricity : ℝ := real.sqrt 2 / 2
def ellipse_latus_rectum : ℝ := 8

-- Statement for the ellipse equation problem
theorem ellipse_standard_eq (a b c : ℝ) (h1 : c / a = ellipse_eccentricity) (h2 : a^2 / c = ellipse_latus_rectum) :
  (a = 4 * real.sqrt 2) → (b = 4) → (c = 4) → (x y : ℝ), (x^2 / 32) + (y^2 / 16) = 1 :=
sorry

-- Define the conditions for the hyperbola problem
def hyperbola_m : ℝ := -4
def point_M : ℝ × ℝ := (2, -2)

-- Statement for the hyperbola equation problem
theorem hyperbola_eq (x y m : ℝ) (h_asymptotes : x^2 - 2*y^2 = m) (h_pass_through : (2^2 - 2*(-2)^2 = m)) :
  (m = hyperbola_m) → (y ^ 2 / 2) - (x ^ 2 / 4) = 1 :=
sorry

end ellipse_standard_eq_hyperbola_eq_l328_328103


namespace complex_eq_z_l328_328212

noncomputable def z := Complex

theorem complex_eq_z (z : Complex) (i : Complex.Im) (hz: (z - 2) * (1 + i) = 1 - i) :
  z = 2 - i :=
sorry

end complex_eq_z_l328_328212


namespace binom_30_3_eq_4060_l328_328855

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := 
by sorry

end binom_30_3_eq_4060_l328_328855


namespace geometric_sequence_common_ratio_l328_328117

theorem geometric_sequence_common_ratio : 
  ∀ a1 a2 a3 a4 a5 : ℤ, 
  a1 = 32 → 
  a2 = -48 → 
  a3 = 72 → 
  a4 = -108 → 
  a5 = 162 → 
  ∃ r : ℚ, 
  r = -3/2 ∧ 
  a2 = a1 * r ∧ 
  a3 = a2 * r ∧ 
  a4 = a3 * r ∧ 
  a5 = a4 * r :=
by 
  intros a1 a2 a3 a4 a5 h1 h2 h3 h4 h5
  use -3/2
  simp [h1, h2, h3, h4, h5]
  split; { linarith }

end geometric_sequence_common_ratio_l328_328117


namespace minimum_value_of_z_l328_328768

theorem minimum_value_of_z :
  ∀ (x y : ℝ), ∃ z : ℝ, z = 2*x^2 + 3*y^2 + 8*x - 6*y + 35 ∧ z ≥ 24 := by
  sorry

end minimum_value_of_z_l328_328768


namespace abs_minus_four_minus_two_l328_328851

theorem abs_minus_four_minus_two : | -4 - 2 | = 6 := 
by
  sorry

end abs_minus_four_minus_two_l328_328851


namespace range_of_m_intersection_distance_perpendicular_condition_l328_328948

-- (1) Given the circle equation, prove the range of values for m
theorem range_of_m (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0 → m < 5) :=
sorry

-- (2) Given the circle equation, line equation and distance condition, prove the value of m
theorem intersection_distance (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0) ∧
  (∀ x y : ℝ, x + 2*y - 4 = 0) ∧
  (∀ M N : ℝ, M - N = 4*sqrt(5)/5) →
  m = 4 :=
sorry

-- (3) Given the circle equation, line equation and perpendicular condition, prove the value of m
theorem perpendicular_condition (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0) ∧
  (∀ x y : ℝ, x + 2*y - 4 = 0) ∧
  (∀ O M N : ℝ, O = 0 ∧ (M * N = 0)) → 
  m = 8/5 :=
sorry

end range_of_m_intersection_distance_perpendicular_condition_l328_328948


namespace correct_exponentiation_l328_328471

theorem correct_exponentiation (x : ℝ) : x^2 * x^3 = x^5 :=
by sorry

end correct_exponentiation_l328_328471


namespace coefficient_x3y3_in_polynomials_l328_328767

/--

Given the polynomials \( (x + y)^6 \) and \( \left(z + \frac{1}{z}\right)^8 \),
prove that the coefficient of \( x^3 y^3 \) in \( (x + y)^6 \left(z + \frac{1}{z}\right)^8 \) is \( 1400 \).

-/
theorem coefficient_x3y3_in_polynomials :
  coeff (monomial 3 1 x * monomial 3 1 y) ((x + y)^6 * (z + (1/z))^8) = 1400 :=
sorry

end coefficient_x3y3_in_polynomials_l328_328767


namespace maximize_reciprocal_sum_at_perpendicular_l328_328003

noncomputable def maximized_sum_of_reciprocals_condition (O A B P Q R : Point) (OA OB : Line) (h1 : P ∈ line_segment O B) (h2 : Q ∈ OA) (h3 : R ∈ OB) (h4 : collinear {Q, P, R}) : Prop :=
∃(CD : Real),
CD > 0 ∧ 
(forall (PR PQ : Real), 1 / PR + 1/ PQ = 1 / CD) ∧
(∀ (angle_QRP : Real), angle_QRP = 90 → 
(1 / distance P Q + 1 / distance P R = 1 / min distance Q R CD))

theorem maximize_reciprocal_sum_at_perpendicular (O A B P Q R : Point) (OA OB : Line) (h1 : P ∈ line_segment O B) (h2 : Q ∈ OA) (h3 : R ∈ OB) (h4 : collinear {Q, P, R}) : 
    (∃ (angle_QRP : Real), angle_QRP = 90 → maximized_sum_of_reciprocals_condition O A B P Q R OA OB h1 h2 h3 h4) :=
    sorry

end maximize_reciprocal_sum_at_perpendicular_l328_328003


namespace revenue_from_full_price_tickets_l328_328113

-- Definitions of the conditions
def total_tickets (f h : ℕ) : Prop := f + h = 180
def total_revenue (f h p : ℕ) : Prop := f * p + h * (p / 2) = 2750

-- Theorem statement
theorem revenue_from_full_price_tickets (f h p : ℕ) 
  (h_total_tickets : total_tickets f h) 
  (h_total_revenue : total_revenue f h p) : 
  f * p = 1000 :=
  sorry

end revenue_from_full_price_tickets_l328_328113


namespace crabapple_sequences_l328_328378

theorem crabapple_sequences (students : ℕ) (days : ℕ) (choices : ℕ) 
  (h_students : students = 15) (h_days : days = 5) (h_choices : choices = 15^5) : 
  choices = 759375 :=
by 
  unfold students days choices at *
  rw [h_students, h_days]
  sorry

end crabapple_sequences_l328_328378


namespace students_chocolate_milk_l328_328897

-- Definitions based on the problem conditions
def students_strawberry_milk : ℕ := 15
def students_regular_milk : ℕ := 3
def total_milks_taken : ℕ := 20

-- The proof goal
theorem students_chocolate_milk : total_milks_taken - (students_strawberry_milk + students_regular_milk) = 2 := by
  -- The proof steps will go here (not required as per instructions)
  sorry

end students_chocolate_milk_l328_328897


namespace geometric_sequence_sum_l328_328357

def geometric_sequence_props (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  (∀ n, a n = a 1 * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (1 - 2^n) / (1 - 2)) ∧ 
  (a 5 - a 3 = 12) ∧ 
  (a 6 - a 4 = 24)

theorem geometric_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : geometric_sequence_props a S) :
  ∀ n, S n / a n = 2 - 2^(1 - n) :=
by
  sorry

end geometric_sequence_sum_l328_328357


namespace tan_alpha_add_pi_over_4_l328_328316

open Real

theorem tan_alpha_add_pi_over_4 
  (α : ℝ)
  (h1 : tan α = sqrt 3) : 
  tan (α + π / 4) = -2 - sqrt 3 :=
by
  sorry

end tan_alpha_add_pi_over_4_l328_328316


namespace paint_cost_is_correct_l328_328298

-- Define the prices and coverage rates of the paints
def price_A : ℝ := 3.20
def coverage_A : ℝ := 60
def price_B : ℝ := 5.50
def coverage_B : ℝ := 55
def price_C : ℝ := 4.00
def coverage_C : ℝ := 50

-- Define the edge lengths of the cuboid
def length_x : ℝ := 12
def length_y : ℝ := 15
def length_z : ℝ := 20

-- Define the areas of the faces of the cuboid
def area_largest_faces : ℝ := 2 * (length_y * length_z)
def area_middle_faces : ℝ := 2 * (length_x * length_z)
def area_smallest_faces : ℝ := 2 * (length_x * length_y)

-- Define the required quarts of paint
def quarts_A : ℝ := area_largest_faces / coverage_A
def quarts_B : ℝ := area_middle_faces / coverage_B
def quarts_C : ℝ := area_smallest_faces / coverage_C

-- Define the costs of the paints
def cost_A : ℝ := (quarts_A.ceil) * price_A
def cost_B : ℝ := (quarts_B.ceil) * price_B
def cost_C : ℝ := (quarts_C.ceil) * price_C

-- Define the total cost to paint the cuboid
def total_cost : ℝ := cost_A + cost_B + cost_C

-- The statement to prove
theorem paint_cost_is_correct : total_cost = 113.50 := by
  sorry

end paint_cost_is_correct_l328_328298


namespace additional_lollipops_needed_l328_328640

theorem additional_lollipops_needed
  (kids : ℕ) (initial_lollipops : ℕ) (min_lollipops : ℕ) (max_lollipops : ℕ)
  (total_kid_with_lollipops : ∀ k, ∃ n, min_lollipops ≤ n ∧ n ≤ max_lollipops ∧ k = n ∨ k = n + 1 )
  (divisible_by_kids : (min_lollipops + max_lollipops) % kids = 0)
  (min_lollipops_eq : min_lollipops = 42)
  (kids_eq : kids = 42)
  (initial_lollipops_eq : initial_lollipops = 650)
  : ∃ additional_lollipops, (n : ℕ) = 42 → additional_lollipops = 1975 := 
by sorry

end additional_lollipops_needed_l328_328640


namespace Q_ratio_one_l328_328172

noncomputable def P (x : ℝ) : ℝ := x^2013 + 19 * x^2012 + 1

def distinctRoots (f : ℝ → ℝ) : Prop := 
  ∃ r : list ℝ, r.length = 2013 ∧ (∀ (i j : ℕ), i ≠ j → r[i] ≠ r[j]) ∧ 
  ∀ (x : ℝ), f x = 0 → (∃ (i : ℕ), i < 2013 ∧ x = r[i])

theorem Q_ratio_one :
  (distinctRoots P) →
  (∀ j, j ∈ (list.range 2013) →
  Q (r[j] + 1 / r[j]) = 0) →
  Q(1) / Q(-1) = 1 :=
begin
  sorry
end

end Q_ratio_one_l328_328172


namespace max_volume_triang_box_l328_328826

theorem max_volume_triang_box (a h : ℝ) 
    (h1 : a + 2*real.sqrt 3*h = 30) :
  (∃ (V : ℝ), (V = (real.sqrt 3 * a^2 * h) / 4) ∧ 
    V ≤ 500) :=
begin
  sorry,
end

end max_volume_triang_box_l328_328826


namespace recursive_relation_f_l328_328984

def f (n : ℕ) : ℕ :=
  ∑ i in finset.range (2 * n + 1), i^2

theorem recursive_relation_f (k : ℕ) : 
    f (k + 1) = f k + (2 * k + 1)^2 + (2 * k + 2)^2 :=
sorry

end recursive_relation_f_l328_328984


namespace cos_two_theta_range_f_l328_328270

-- Definition of vectors and the function f
def a (x : ℝ) : ℝ × ℝ := (1, Real.cos (2 * x))
def b (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), -Real.sqrt 3)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Proof statements
theorem cos_two_theta (θ : ℝ) (h : f ((θ / 2) + (2 * Real.pi / 3)) = 6 / 5) : Real.cos (2 * θ) = 7 / 25 := sorry

theorem range_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) : -Real.sqrt 3 ≤ f x ∧ f x ≤ 2 := sorry

end cos_two_theta_range_f_l328_328270


namespace fx_inequality_l328_328949

def even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def f_sign (x : ℝ) (f : ℝ → ℝ) : ℝ :=
  if -3 < x ∧ x < -1 then 1
  else if -1 < x ∧ x < 1 then -1
  else if 1 < x ∧ x < 3 then 1
  else 0

def x_cube_sign (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x < 0 then -1
  else 0

theorem fx_inequality (f : ℝ → ℝ) (hf_even : even f) :
  ∀ x, (-3 <= x ∧ x <= 3) → (x^3 * f x < 0) ↔ (x ∈ set.Icc (-3) (-1) ∪ set.Icc 0 1) :=
by
  sorry

end fx_inequality_l328_328949


namespace problem_solution_l328_328238

variables {a b c x y z : ℝ}

-- Given conditions
def conditions := a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  a^2 + b^2 + c^2 = 25 ∧ x^2 + y^2 + z^2 = 36 ∧ a * x + b * y + c * z = 30

-- Statement to prove
theorem problem_solution (h : conditions) : 
  (a + b + c) / (x + y + z) = 5 / 6 := 
sorry

end problem_solution_l328_328238


namespace avg_speed_yx_30_l328_328784

-- Define the conditions
def distance : ℝ := sorry -- We do not need the exact distance for the proof
def speed_xy : ℝ := 60
def avg_speed_total : ℝ := 40

-- Define the time taken to travel based on given speeds
noncomputable def time_xy := distance / speed_xy

-- Declare the unknown average speed from y to x
variable (speed_yx : ℝ)

-- Define the total time for the round trip
noncomputable def time_yx := distance / speed_yx
noncomputable def total_time := time_xy + time_yx

-- Calculate average speed formula for the entire journey
noncomputable def avg_speed := (2 * distance) / total_time

-- The theorem stating that the average speed on the return journey is 30 km/hr
theorem avg_speed_yx_30 : avg_speed_total = 40 → avg_speed = 40 → speed_yx = 30 := by
  intro h1 h2
  sorry

end avg_speed_yx_30_l328_328784


namespace polynomial_root_multiplicity_l328_328565

theorem polynomial_root_multiplicity (A B n : ℤ) (h1 : A + B + 1 = 0) (h2 : (n + 1) * A + n * B = 0) :
  A = n ∧ B = -(n + 1) :=
sorry

end polynomial_root_multiplicity_l328_328565


namespace simplify_tangent_expression_l328_328015

open Real

theorem simplify_tangent_expression :
  (tan (π / 6) + tan (2 * π / 9) + tan (5 * π / 18) + tan (π / 3)) / cos (π / 9) = 8 * sqrt 3 / 3 :=
by sorry

end simplify_tangent_expression_l328_328015


namespace percentage_of_life_in_accounting_jobs_l328_328459

-- Define the conditions
def years_as_accountant : ℕ := 25
def years_as_manager : ℕ := 15
def lifespan : ℕ := 80

-- Define the proof problem statement
theorem percentage_of_life_in_accounting_jobs :
  (years_as_accountant + years_as_manager) / lifespan * 100 = 50 := 
by sorry

end percentage_of_life_in_accounting_jobs_l328_328459


namespace marble_probability_l328_328871

-- Define the conditions
noncomputable def total_marbles : ℕ := 25
noncomputable def prob_both_black : ℚ := 27 / 50

-- Define the problem
theorem marble_probability (m n : ℕ) (hmn : Nat.gcd m n = 1) : 
  (m + n = 26) :=
by
  have prob_white := (1/25 : ℚ)
  have h_mn : (m : ℚ) / n = prob_white
  sorry -- Proof will go here

end marble_probability_l328_328871


namespace no_valid_A_for_quadratic_equation_solution_l328_328861

theorem no_valid_A_for_quadratic_equation_solution :
  ∀ A : ℕ, (1 ≤ A ∧ A ≤ 9) → ¬ ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ p * q = Nat.factorial A ∧ p + q = 10 + A :=
begin
  intros A hA,
  cases hA with hA1 hA2,
  intro h, cases h with p hp, cases hp with q hq, cases hq with hp_pos hq,
  cases hq with hq_pos h_sum_prod,
  cases h_sum_prod with h_prod h_sum,
  sorry
end

end no_valid_A_for_quadratic_equation_solution_l328_328861


namespace problem_statement_l328_328609

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (-1, 1)

-- Define vector addition
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

-- Define dot product
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define perpendicular condition
def perp (u v : ℝ × ℝ) : Prop := dot_prod u v = 0

theorem problem_statement : perp (vec_add a b) a :=
by
  sorry

end problem_statement_l328_328609


namespace prob_level_A_correct_prob_level_B_correct_prob_exactly_4_correct_prob_exactly_5_correct_l328_328487

-- Define the probabilities of answering correctly and incorrectly
def prob_correct : ℚ := 2/3
def prob_incorrect : ℚ := 1/3

-- Probability of being rated level A
def prob_level_A : ℚ := prob_correct^4 + prob_incorrect * prob_correct^4

-- Probability of being rated level B
def prob_level_C : ℚ := prob_incorrect^3 + prob_correct * prob_incorrect^3 + 
                           (prob_correct^2) * prob_incorrect^3 + 
                           prob_incorrect * prob_correct * prob_incorrect^3

def prob_level_B : ℚ := 1 - prob_level_A - prob_level_C

-- Probability of finishing exactly 4 questions
def prob_exactly_4 : ℚ := prob_correct^4 + prob_correct * prob_incorrect^3

-- Probability of finishing exactly 5 questions
def prob_exactly_5 : ℚ := 1 - prob_incorrect^3 - prob_exactly_4

-- Theorems to prove the calculated probabilities
theorem prob_level_A_correct : prob_level_A = 64/243 := 
by sorry

theorem prob_level_B_correct : prob_level_B = 158/243 := 
by sorry

theorem prob_exactly_4_correct : prob_exactly_4 = 2/9 := 
by sorry

theorem prob_exactly_5_correct : prob_exactly_5 = 20/27 := 
by sorry

end prob_level_A_correct_prob_level_B_correct_prob_exactly_4_correct_prob_exactly_5_correct_l328_328487


namespace point_set_condition_l328_328025

open Real EuclideanGeometry Set

variable {P B : Point} {r : ℝ}

-- Assume P, B, and r are given such that:
-- P is the center of circle Γ with radius r
-- B is a point on circle Γ
-- B is the center of circle Δ with radius r/2

def midpoint (A B : Point) : Point := (A + B) / 2

theorem point_set_condition :
  ∀ (A : Point),
  (dist A B ≤ min (dist A P + r) (dist A B + r/2))
  ↔ (A ∈ segment P (midpoint P B)) :=
by
  sorry

end point_set_condition_l328_328025


namespace caloprian_lifespan_proof_l328_328841

open Real

noncomputable def timeDilation (delta_t : ℝ) (v : ℝ) (c : ℝ) : ℝ :=
  delta_t * sqrt (1 - (v ^ 2) / (c ^ 2))

noncomputable def caloprianMinLifeSpan (d : ℝ) (v : ℝ) (c : ℝ) : ℝ :=
  let earth_time := (d / v) * 2
  timeDilation earth_time v c

theorem caloprian_lifespan_proof :
  caloprianMinLifeSpan 30 0.3 1 = 20 * sqrt 91 :=
sorry

end caloprian_lifespan_proof_l328_328841


namespace round_repeating_637_to_thousandth_l328_328708

noncomputable def repeatingDecimal := (37 + (637 / 999))
def roundedToThousandth := 37.638

theorem round_repeating_637_to_thousandth :
  Float.round (repeatingDecimal * 1000) / 1000 = roundedToThousandth :=
by
  -- Proof goes here
  sorry

end round_repeating_637_to_thousandth_l328_328708


namespace hotel_loss_l328_328123

theorem hotel_loss (operations_expenses : ℝ) (payment_fraction : ℝ) (total_payment : ℝ) (loss : ℝ) 
  (hOpExp : operations_expenses = 100) 
  (hPayFr : payment_fraction = 3 / 4)
  (hTotalPay : total_payment = payment_fraction * operations_expenses) 
  (hLossCalc : loss = operations_expenses - total_payment) : 
  loss = 25 := 
by 
  sorry

end hotel_loss_l328_328123


namespace intersection_A_B_l328_328224

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < 2 - x ∧ 2 - x < 3 }

theorem intersection_A_B :
  A ∩ B = {0, 1} := sorry

end intersection_A_B_l328_328224


namespace find_a_plus_b_l328_328937

theorem find_a_plus_b (a b : ℝ) (i : ℂ) (h1 : i = complex.I) (h2 : 2 / (1 - i) = a + b * i) : a + b = 2 := by
  sorry

end find_a_plus_b_l328_328937


namespace find_x_plus_y_l328_328284

theorem find_x_plus_y (x y : ℚ) (h1 : |x| + x + y = 12) (h2 : x + |y| - y = 10) : x + y = 26/5 :=
sorry

end find_x_plus_y_l328_328284


namespace quadratic_root_u_value_l328_328900

theorem quadratic_root_u_value (u : ℝ) :
  (∃ x : ℝ, x = (-25 - real.sqrt 469) / 12 ∧ 6 * x^2 + 25 * x + u = 0) → u = 6.5 :=
by
  sorry

end quadratic_root_u_value_l328_328900


namespace vector_problem_l328_328269

theorem vector_problem
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) (k : ℝ)
  (ha : a = (1, 1, 0))
  (hb : b = (-1, 0, 2))
  (hp : (k * a.1 + b.1, k * a.2 + b.2, k * a.3 + b.3) ∙ (2 * a.1 - b.1, 2 * a.2 - b.2, 2 * a.3 - b.3) = 0) :
  k = 7 / 5 :=
sorry

end vector_problem_l328_328269


namespace value_of_t_plus_one_over_t_l328_328912

theorem value_of_t_plus_one_over_t
  (t : ℝ)
  (h1 : t^2 - 3 * t + 1 = 0)
  (h2 : t ≠ 0) :
  t + 1 / t = 3 :=
by
  sorry

end value_of_t_plus_one_over_t_l328_328912


namespace total_volume_is_correct_l328_328468

def box_edge_length : ℕ := 4
def number_of_boxes : ℕ := 3
def volume_of_one_box : ℕ := box_edge_length ^ 3
def total_volume : ℕ := volume_of_one_box * number_of_boxes

theorem total_volume_is_correct : total_volume = 192 :=
by
  unfold box_edge_length number_of_boxes volume_of_one_box total_volume
  sorry

end total_volume_is_correct_l328_328468


namespace tom_remaining_balloons_l328_328449

def original_balloons : ℕ := 30
def given_balloons : ℕ := 16
def remaining_balloons (original_balloons given_balloons : ℕ) : ℕ := original_balloons - given_balloons

theorem tom_remaining_balloons : remaining_balloons original_balloons given_balloons = 14 :=
by
  -- proof omitted for clarity
  sorry

end tom_remaining_balloons_l328_328449


namespace exactly_one_even_contradiction_assumption_l328_328388

variable (a b c : ℕ)

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
  (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0)

def conclusion (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
  (a % 2 = 0 ∧ b % 2 = 0) ∨
  (b % 2 = 0 ∧ c % 2 = 0) ∨
  (c % 2 = 0 ∧ a % 2 = 0)

theorem exactly_one_even_contradiction_assumption :
    exactly_one_even a b c ↔ ¬ conclusion a b c :=
by
  sorry

end exactly_one_even_contradiction_assumption_l328_328388


namespace missing_weights_l328_328430

theorem missing_weights :
  ∃ (n k : ℕ), (n > 10) ∧ (606060 % 8 = 4) ∧ (606060 % 9 = 0) ∧ 
  (5 * k + 24 * k + 43 * k = 606060 + 72 * n) :=
sorry

end missing_weights_l328_328430


namespace tangent_divides_second_side_l328_328498

theorem tangent_divides_second_side
  (a : ℕ) (h_nonagon : a = 9) 
  (h_sides : ∀ i : fin a, (∃ n : ℕ, 1 ≤ n ∧ (i = 0 ∨ i = 2 → n = 1))) :
  ∃ s1 : ℝ, ∃ s2 : ℝ, s1 = 1 / 2 ∧ s2 = 1 / 2 ∧ (s1 + s2 = 1) :=
by
  sorry

end tangent_divides_second_side_l328_328498


namespace find_m_l328_328217

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d 

noncomputable def sum_first_n_terms (a S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

theorem find_m {a S : ℕ → ℤ} (d : ℤ) (m : ℕ) 
  (h1 : arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : a 1 = 1)
  (h4 : S 3 = a 5)
  (h5 : a m = 2011) :
  m = 1006 :=
sorry

end find_m_l328_328217


namespace find_a_b_l328_328899

-- Define that the roots of the corresponding equality yield the specific conditions.
theorem find_a_b (a b : ℝ) :
    (∀ x : ℝ, x^2 + (a + 1) * x + ab > 0 ↔ (x < -1 ∨ x > 4)) →
    a = -4 ∧ b = 1 := 
by
    sorry

end find_a_b_l328_328899


namespace quadrilateral_with_equal_angles_is_parallelogram_l328_328081

axiom Quadrilateral (a b c d : Type) : Prop
axiom Parallelogram (a b c d : Type) : Prop
axiom equal_angles (a b c d : Type) : Prop

theorem quadrilateral_with_equal_angles_is_parallelogram 
  (a b c d : Type) 
  (q : Quadrilateral a b c d)
  (h : equal_angles a b c d) : Parallelogram a b c d := 
sorry

end quadrilateral_with_equal_angles_is_parallelogram_l328_328081


namespace max_toys_l328_328903

theorem max_toys (saved_money : ℕ) (allowance : ℕ) (small_toy_cost : ℕ) (puzzle_cost : ℕ) (lego_set_cost : ℕ) 
                 (total_money : ℕ) (promotion : ℕ) : 
  saved_money = 3 → 
  allowance = 37 → 
  small_toy_cost = 8 → 
  puzzle_cost = 12 → 
  lego_set_cost = 20 → 
  total_money = saved_money + allowance →
  promotion = 3 →
  total_money = 40 →
  (max_toys : ℕ) = 6 :=
by {
  sorry
}

end max_toys_l328_328903


namespace line_perpendicular_to_plane_proof_l328_328941

noncomputable def line_perpendicular_to_plane : Prop :=
  ∃ (a : ℝ^3) (u : ℝ^3),
    a = ⟨1, -1, 2⟩ ∧
    u = ⟨-2, 2, -4⟩ ∧
    (∃ k : ℝ, u = k • a)

theorem line_perpendicular_to_plane_proof : line_perpendicular_to_plane :=
  sorry

end line_perpendicular_to_plane_proof_l328_328941


namespace smallest_sum_p_q_l328_328989

theorem smallest_sum_p_q (p q : ℕ) (h1: p > 0) (h2: q > 0) (h3 : (∃ k1 k2 : ℕ, 7 ^ (p + 4) * 5 ^ q * 2 ^ 3 = (k1 * 7 *  k2 * 5 * (2 * 3))) ^ 3) :
  p + q = 5 :=
by
  -- Proof goes here
  sorry

end smallest_sum_p_q_l328_328989


namespace hotel_loss_l328_328125

theorem hotel_loss (operations_expenses : ℝ) (payment_fraction : ℝ) (total_payment : ℝ) (loss : ℝ) 
  (hOpExp : operations_expenses = 100) 
  (hPayFr : payment_fraction = 3 / 4)
  (hTotalPay : total_payment = payment_fraction * operations_expenses) 
  (hLossCalc : loss = operations_expenses - total_payment) : 
  loss = 25 := 
by 
  sorry

end hotel_loss_l328_328125


namespace chord_length_of_intersection_l328_328051

-- Define the circle and the line
def circle_eq (x y : ℝ) := x^2 + y^2 = 3
def line_eq (x y : ℝ) := y = -x + 2

-- Define the distance from the center to the line
def distance_to_line_from_center := (λ (x y : ℝ), (abs (0 + 0 - 2)) / (real.sqrt 2))

-- State the theorem
theorem chord_length_of_intersection :
  ∀ (A B : ℝ × ℝ), 
  (circle_eq A.fst A.snd) ∧ (circle_eq B.fst B.snd) ∧
  (line_eq A.fst A.snd) ∧ (line_eq B.fst B.snd) →
  real.dist A B = 2 :=
by
  sorry

end chord_length_of_intersection_l328_328051


namespace regular_ticket_cost_l328_328494

theorem regular_ticket_cost
    (adults : ℕ) (children : ℕ) (cash_given : ℕ) (change_received : ℕ) (adult_cost : ℕ) (child_cost : ℕ) :
    adults = 2 →
    children = 3 →
    cash_given = 40 →
    change_received = 1 →
    child_cost = adult_cost - 2 →
    2 * adult_cost + 3 * child_cost = cash_given - change_received →
    adult_cost = 9 :=
by
  intros h_adults h_children h_cash_given h_change_received h_child_cost h_sum
  sorry

end regular_ticket_cost_l328_328494


namespace original_difference_in_books_l328_328804

theorem original_difference_in_books 
  (x y : ℕ) 
  (h1 : x + y = 5000) 
  (h2 : (1 / 2 : ℚ) * (x - 400) - (y + 400) = 400) : 
  x - y = 3000 := 
by 
  -- Placeholder for the proof
  sorry

end original_difference_in_books_l328_328804


namespace weight_of_10_moles_ascorbic_acid_l328_328276

theorem weight_of_10_moles_ascorbic_acid (atomic_weight_C : ℝ) (atomic_weight_H : ℝ) (atomic_weight_O : ℝ)
    (hC : atomic_weight_C = 12.01) (hH : atomic_weight_H = 1.008) (hO : atomic_weight_O = 16.00) :
    10 * (6 * atomic_weight_C + 8 * atomic_weight_H + 6 * atomic_weight_O) = 1761.24 :=
by
  rw [hC, hH, hO]
  norm_num
  sorry

end weight_of_10_moles_ascorbic_acid_l328_328276


namespace distance_against_current_l328_328802

theorem distance_against_current (V_b V_c : ℝ) (h1 : V_b + V_c = 2) (h2 : V_b = 1.5) : 
  (V_b - V_c) * 3 = 3 := by
  sorry

end distance_against_current_l328_328802


namespace prove_increased_radius_l328_328192

noncomputable def original_radius : ℝ := 16
noncomputable def distance_trip : ℝ := 520
noncomputable def odometer_return : ℝ := 500
noncomputable def one_mile_in_inches : ℝ := 63360

def correct_increased_radius (r r' : ℝ) : Prop :=
  ∃ Δr, Δr = r' - r ∧ Δr = 0.4

theorem prove_increased_radius :
  correct_increased_radius original_radius 
    (520 * (2 * (real.pi * 16) / one_mile_in_inches) / 500) := by
  sorry

end prove_increased_radius_l328_328192


namespace probability_two_dice_same_l328_328698

def fair_dice_probability (dice : ℕ) (sides : ℕ) : ℚ :=
  1 - ((sides.factorial / (sides - dice).factorial) / sides^dice)

theorem probability_two_dice_same (dice : ℕ) (sides : ℕ) (h1 : dice = 5) (h2 : sides = 10) :
  fair_dice_probability dice sides = 1744 / 2500 := by
  sorry

end probability_two_dice_same_l328_328698


namespace hotel_loss_l328_328127

theorem hotel_loss :
  (ops_expenses : ℝ) (payment_frac : ℝ) (total_received : ℝ) (loss : ℝ)
  (h_ops_expenses : ops_expenses = 100)
  (h_payment_frac : payment_frac = 3 / 4)
  (h_total_received : total_received = payment_frac * ops_expenses)
  (h_loss : loss = ops_expenses - total_received) :
  loss = 25 :=
by
  sorry

end hotel_loss_l328_328127


namespace triangle_formation_probability_l328_328400

theorem triangle_formation_probability : 
  let sticks := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] in
  let valid_triplets := 
    { (a, b, c) | a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧ a ≠ b ∧ b ≠ c ∧ a < b ∧ b < c ∧ a + b > c } in
  let total_triplets := finset.card (finset.filter (λ x, true) (finset.powerset_len 3 (finset.filter (λ x, true) sticks))) in
  let valid_triplets_count := finset.card valid_triplets in
  (valid_triplets_count : ℚ) / (total_triplets : ℚ) = 7 / 30 :=
by {
  sorry
}

end triangle_formation_probability_l328_328400


namespace chickens_and_sheep_are_ten_l328_328310

noncomputable def chickens_and_sheep_problem (C S : ℕ) : Prop :=
  (C + 4 * S = 2 * C) ∧ (2 * C + 4 * (S - 4) = 16 * (S - 4)) → (S + 2 = 10)

theorem chickens_and_sheep_are_ten (C S : ℕ) : chickens_and_sheep_problem C S :=
sorry

end chickens_and_sheep_are_ten_l328_328310


namespace flipping_teacups_l328_328058

theorem flipping_teacups (n : ℕ) (h : n = 12) : ∀ i : ℕ, (i < 13) → (i = 1 ∨ i = 4 ∨ i = 9 → cup_up_after_flips i) ∧ (i ≠ 1 ∧ i ≠ 4 ∧ i ≠ 9 → ¬ cup_up_after_flips i) :=
by
  sorry

end flipping_teacups_l328_328058


namespace alpha_beta_square_inequality_l328_328985

theorem alpha_beta_square_inequality
  (α β : ℝ)
  (h1 : α ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : β ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) :
  α^2 > β^2 :=
by
  sorry

end alpha_beta_square_inequality_l328_328985


namespace determine_x_of_ohara_triple_l328_328062

def is_ohara_triple (a b x : ℕ) : Prop :=
  real.sqrt a + real.sqrt b = x

theorem determine_x_of_ohara_triple :
  is_ohara_triple 49 16 x → x = 11 := by
  sorry

end determine_x_of_ohara_triple_l328_328062


namespace distances_equal_sqrt2_count_l328_328750

def Point : Type := (ℝ × ℝ × ℝ)

def distance (p1 p2 : Point) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  (Math.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2))

def vertices : List Point :=
  [ (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 3), (1, 0, 3), (1, 1, 3), (0, 1, 3) ]

def intermediatePoints : List Point :=
  [ (0, 0, 1), (0, 0, 2), (1, 0, 1), (1, 0, 2), 
    (1, 1, 1), (1, 1, 2), (0, 1, 1), (0, 1, 2) ]

theorem distances_equal_sqrt2_count : finset.card 
  (finset.filter (λ d, d = sqrt 2) 
    (finset.image2 distance 
      (finset.from_list vertices) 
      (finset.from_list (vertices ++ intermediatePoints)))) = 32 := 
by
  sorry

end distances_equal_sqrt2_count_l328_328750


namespace probability_even_sum_l328_328068

theorem probability_even_sum : 
  (let total_outcomes := 20 * 19 in -- Total number of ways to choose 2 out of 20
   let favorable_even_even := 10 * 9 in -- Both numbers are even
   let favorable_odd_odd := 10 * 9 in -- Both numbers are odd
   let total_favorable := favorable_even_even + favorable_odd_odd in -- Total favorable outcomes
   total_favorable / total_outcomes = 9 / 19) :=
by 
  sorry

end probability_even_sum_l328_328068


namespace intersection_points_l328_328196

noncomputable def line1 (x y : ℝ) : Prop := 3 * x - 2 * y = 12
noncomputable def line2 (x y : ℝ) : Prop := 2 * x + 4 * y = 8
noncomputable def line3 (x y : ℝ) : Prop := -5 * x + 15 * y = 30
noncomputable def line4 (x : ℝ) : Prop := x = -3

theorem intersection_points : 
  (∃ (x y : ℝ), line1 x y ∧ line2 x y) ∧ 
  (∃ (x y : ℝ), line1 x y ∧ x = -3 ∧ y = -10.5) ∧ 
  ¬(∃ (x y : ℝ), line2 x y ∧ line3 x y) ∧
  ∃ (x y : ℝ), line4 x ∧ y = -10.5 :=
  sorry

end intersection_points_l328_328196


namespace symmetric_5_points_l328_328443

theorem symmetric_5_points :
  ∀ (points : Fin 5 → ℝ × ℝ), ∃ (shifted_points : Fin 5 → ℝ × ℝ) (sym_line : ℝ × ℝ × ℝ),
  (∀ i j : Fin 5, dist (shifted_points i) (shifted_points j) = dist (points i) (points j)) ∧ 
  (∀ i, reflected_point (shifted_points i) sym_line ∈ set_of_points shifted_points) :=
sorry

-- Definitions of dist and reflected_point based on geometric principles
def dist (p q : ℝ × ℝ) : ℝ :=
  ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2) ^ 0.5

def reflected_point (p : ℝ × ℝ) (line : ℝ × ℝ × ℝ) : ℝ × ℝ :=
  let (A, B, C) := line in
  let d := A * A + B * B in
  let x' := (B * B * p.1 - A * A * p.1 + 2 * A * B * p.2 + 2 * A * C) / d in
  let y' := (A * A * p.2 - B * B * p.2 + 2 * A * B * p.1 + 2 * B * C) / d in
  (x', y')

def set_of_points (points : Fin 5 → ℝ × ℝ) : set (ℝ × ℝ) :=
  { p | ∃ i, points i = p }

end symmetric_5_points_l328_328443


namespace coeff_x4_in_binomial_expansion_l328_328481

theorem coeff_x4_in_binomial_expansion : (coeff (expandBinomial x - 2 / x) ^ 6 x^4 = -12) := sorry

end coeff_x4_in_binomial_expansion_l328_328481


namespace vector_difference_parallelogram_l328_328966

noncomputable theory

open_locale classical

-- Define the vector types and the two given vectors
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def a : Vector2D := ⟨-1, -3⟩
def b : ℝ → Vector2D := λ t, ⟨2, t⟩

-- Define the condition that vectors are parallel
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

-- Theorem to prove
theorem vector_difference_parallelogram (t : ℝ) (h : parallel a (b t)) :
  a.x - (b t).x = -3 ∧ a.y - (b t).y = -9 :=
by sorry

end vector_difference_parallelogram_l328_328966


namespace problem_l328_328061

theorem problem (a b c d e : ℝ) (h0 : a ≠ 0)
  (h1 : 625 * a + 125 * b + 25 * c + 5 * d + e = 0)
  (h2 : -81 * a + 27 * b - 9 * c + 3 * d + e = 0)
  (h3 : 16 * a + 8 * b + 4 * c + 2 * d + e = 0) :
  (b + c + d) / a = -6 :=
by
  sorry

end problem_l328_328061


namespace question_1_question_2_l328_328968

noncomputable def vector_a : ℝ × ℝ × ℝ := (1, 1, 0)
noncomputable def vector_b : ℝ × ℝ × ℝ := (-1, 0, 2)
noncomputable def vector_ka_plus_b (k : ℝ) : ℝ × ℝ × ℝ := (k - 1, k, 2)
noncomputable def vector_2a_minus_b : ℝ × ℝ × ℝ := (3, 2, -2)

-- Question (Ⅰ) Statement: Prove that the value of k for which the vector k*a + b is parallel to the vector 2*a - b is k = -2.
theorem question_1 (k : ℝ) : (vector_ka_plus_b k = ((vector_ka_plus_b k).1 / (vector_2a_minus_b).1) • vector_2a_minus_b) ↔ k = -2 := 
by
  sorry

noncomputable def normal_vector_of_plane : ℝ × ℝ × ℝ := (2, -2, 1)
noncomputable def unit_normal_vector_1 : ℝ × ℝ × ℝ := (2/3, -2/3, 1/3)
noncomputable def unit_normal_vector_2 : ℝ × ℝ × ℝ := (-2/3, 2/3, -1/3)

-- Question (Ⅱ) Statement: Prove that the unit normal vector of the plane determined by a and b is (2/3, -2/3, 1/3) or (-2/3, 2/3, -1/3).
theorem question_2 (n : ℝ × ℝ × ℝ) : 
  (normal_vector_of_plane = n ∧ ∥n∥ = 1) ↔ (n = unit_normal_vector_1 ∨ n = unit_normal_vector_2) :=
by
  sorry

end question_1_question_2_l328_328968


namespace ellipse_through_points_hyperbola_through_point_l328_328102

theorem ellipse_through_points (m n : ℝ) (h_m_pos : m > 0) (h_n_pos : n > 0) (h_m_ne_n : m ≠ n)
    (h1 : 2 * m + n = 1) (h2 : m + (3 / 2) * n = 1)
    (x y : ℝ) (ell_eq : m * x^2 + n * y^2 = 1 ) : 
    m = 1 / 4 ∧ n = 1 / 2 → ell_eq = (x^2 / 4 + y^2 / 2 = 1) := 
by
  sorry

theorem hyperbola_through_point (λ : ℝ) (x y : ℝ) (h_point : (x, y) = (3, -2))
    (hyp_eq : y^2 / 4 - x^2 / 3 = λ):
    λ = -2 → hyp_eq = x^2 / 6 - y^2 / 8 = 1 := 
by
  sorry

end ellipse_through_points_hyperbola_through_point_l328_328102


namespace problem_solution_l328_328932

def positive (n : ℕ) : Prop := n > 0
def pairwise_coprime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1
def divides (m n : ℕ) : Prop := ∃ k, n = k * m

theorem problem_solution (a b c : ℕ) :
  positive a → positive b → positive c →
  pairwise_coprime a b c →
  divides (a^2) (b^3 + c^3) →
  divides (b^2) (a^3 + c^3) →
  divides (c^2) (a^3 + b^3) →
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 1) ∨ 
  (a = 3 ∧ b = 1 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 1 ∧ c = 3) ∨ 
  (a = 1 ∧ b = 3 ∧ c = 2) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 3) := by
  sorry

end problem_solution_l328_328932


namespace find_f_prime_at_2_l328_328239

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x * (derivative f 2) + Real.log x

theorem find_f_prime_at_2 : (derivative f 2) = -9 / 4 :=
by
  sorry

end find_f_prime_at_2_l328_328239


namespace problem_statement_l328_328973

theorem problem_statement
  (a b c : ℝ)
  (h1 : abc > 0)
  (h2 : a + b + c = 0)
  : let m := (|a + b| / c) + (2 * |b + c| / a) + (3 * |c + a| / b) in
    (∃ x y : ℝ, set.countable {m} ∧ set.min {m} = y ∧ x + y = -1) :=
sorry

end problem_statement_l328_328973


namespace magnitude_of_Z_l328_328946

-- Definition of the condition
def complex_eq (Z : ℂ) : Prop := (1 - I) * Z = 1 + I

-- The proof statement
theorem magnitude_of_Z (Z : ℂ) (h : complex_eq Z) : |Z| = 1 :=
sorry

end magnitude_of_Z_l328_328946


namespace least_value_b_times_a_l328_328781

def is_valid_ab (ab : ℕ) : Prop :=
  let d₁ := ab / 10 in
  let d₂ := ab % 10 in
  (1100 + 10 * d₁ + d₂) % 115 = 0

theorem least_value_b_times_a :
  ∃ a b : ℕ, (10 ≤ 10 * a + b ∧ 10 * a + b < 100) ∧ is_valid_ab (10 * a + b) ∧ (b * a = 0) :=
begin
  sorry
end

end least_value_b_times_a_l328_328781


namespace number_of_real_values_number_of_solutions_l328_328421

theorem number_of_real_values (a b c : ℝ) (h₀ : a ≠ 0) :
  3 ^ (a * x^2 + b * x + c) = 1 ↔ a * x^2 + b * x + c = 0 :=
by sorry

theorem number_of_solutions :
  (∃ f : ℝ → ℝ, f = λ x, 3 ^ (3 * x ^ 2 - 8 * x + 2) ∧ ∃ s : ℝ, s = 3 ^ (3 * x ^ 2 - 8 * x + 2)) ↔ 2 :=
by sorry

end number_of_real_values_number_of_solutions_l328_328421


namespace original_curve_parametric_l328_328653

-- Define the stretching transformation
def transform (x y : ℝ) : ℝ × ℝ := (3 * x, y)

-- Define the equation x'^2 + 9y'^2 = 9
def transformed_curve (x' y' : ℝ) : Prop := x'^2 + 9 * y'^2 = 9

-- Define the parametric form in terms of θ
def parametric_curve (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- The statement of the problem
theorem original_curve_parametric (x y θ : ℝ) :
  (transform x y).fst = 3 * x ∧ (transform x y).snd = y →
  transformed_curve (transform x y).fst (transform x y).snd →
  parametric_curve θ = (x, y) :=
by
  sorry

end original_curve_parametric_l328_328653


namespace find_coordinates_of_P_l328_328315

/-- Let the curve C be defined by the equation y = x^3 - 10x + 3 and point P lies on this curve in the second quadrant.
We are given that the slope of the tangent line to the curve at point P is 2. We need to find the coordinates of P.
--/
theorem find_coordinates_of_P :
  ∃ (x y : ℝ), (y = x ^ 3 - 10 * x + 3) ∧ (3 * x ^ 2 - 10 = 2) ∧ (x < 0) ∧ (x = -2) ∧ (y = 15) :=
by
  sorry

end find_coordinates_of_P_l328_328315


namespace king_paid_total_l328_328812

def king_payment (crown_cost tip_percentage : ℝ) : ℝ :=
  crown_cost + crown_cost * (tip_percentage / 100)

theorem king_paid_total (h1 : crown_cost = 20000) (h2 : tip_percentage = 10) :
  king_payment crown_cost tip_percentage = 22000 := by
  sorry

end king_paid_total_l328_328812


namespace complex_multiplication_l328_328229

theorem complex_multiplication :
  ∀ (i : ℂ), i^2 = -1 → (1 - i) * i = 1 + i :=
by
  sorry

end complex_multiplication_l328_328229


namespace total_length_of_matches_is_700_l328_328618

theorem total_length_of_matches_is_700
  (a : ℕ) (d : ℕ) (n : ℕ) (h1 : a = 4) (h2 : d = 2) (h3 : n = 25) : 
  (∑ i in finset.range n, a + i * d) = 700 :=
by
  sorry

end total_length_of_matches_is_700_l328_328618


namespace max_points_on_circles_l328_328755

theorem max_points_on_circles (h1 : ∀ (l : Line) (c : Circle), ∃ (p1 p2 : Point), p1 ≠ p2 ∧ l ∩ c = {p1, p2})
                               (h2 : ∃ (c1 c2 c3 : Circle), coplanar [c1, c2, c3]) : 
  ∃ l : Line, ∀ c ∈ {c1, c2, c3}, ∃ (p1 p2 : Point), p1 ≠ p2 ∧ l ∩ c = {p1, p2} :=
by
  sorry

end max_points_on_circles_l328_328755


namespace geometric_seq_ratio_l328_328361

theorem geometric_seq_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 5 - a 3 = 12) 
  (h2 : a 6 - a 4 = 24) 
  (ha : ∃ a₁ : ℝ, (∀ n : ℕ, a n = a₁ * q ^ (n - 1)))
  (hS : ∀ n : ℕ, S n = a₁ * (1 - q ^ n) / (1 - q)) :
  ∀ n : ℕ, S n / a n = 2 - 2 ^ (1 - n) :=
sorry

end geometric_seq_ratio_l328_328361


namespace probability_prime_sum_cube_rolls_l328_328491

def fair_cube_sides : Finset ℕ := {1, 2, 3, 4, 5, 6}

def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_sums : Finset ℕ := {3, 5, 7, 11}

def outcomes : Finset (ℕ × ℕ) := 
  Finset.product fair_cube_sides fair_cube_sides

def prime_sum_outcomes : Finset (ℕ × ℕ) := 
  outcomes.filter (λ (p : ℕ × ℕ), is_prime (p.1 + p.2))

def r : ℚ := (prime_sum_outcomes.card : ℚ) / (outcomes.card : ℚ)

theorem probability_prime_sum_cube_rolls : r = 7 / 18 := 
by sorry

end probability_prime_sum_cube_rolls_l328_328491


namespace correct_statements_incorrect_statements_l328_328369

-- Definitions by conditions
variables (m n : Type) [line m] [line n] (α β γ : Type) [plane α] [plane β] [plane γ]
variable [different_lines m n]
variable [different_planes α β γ]

-- Statements
def statement1 : Prop := (m ∥ n ∧ n ∥ α) → (m ∥ α ∨ m ⊆ α)
def statement2 : Prop := (m ∥ α ∧ n ∥ α ∧ m ⊆ β ∧ n ⊆ β) → (α ∥ β)
def statement3 : Prop := (α ⊥ γ ∧ β ⊥ γ) → (α ∥ β)
def statement4 : Prop := (α ∥ β ∧ β ∥ γ ∧ m ⊥ α) → (m ⊥ γ)

-- Proof targets
theorem correct_statements : statement1 m n α ∧ statement4 m α β γ :=
by {
  sorry, -- Proof not required per instructions
}

theorem incorrect_statements : ¬ statement2 m n α β ∧ ¬ statement3 α β γ :=
by {
  sorry, -- Proof not required per instructions
}

end correct_statements_incorrect_statements_l328_328369


namespace intersection_eq_l328_328221

variable (A : Set ℤ) (B : Set ℤ)

def A_def := A = {-1, 0, 1, 2}
def B_def := B = {x | -1 < x ∧ x < 2}

theorem intersection_eq : A ∩ B = {0, 1} :=
by
  have A_def : A = {-1, 0, 1, 2} := sorry
  have B_def : B = {x | -1 < x ∧ x < 2} := sorry
  sorry

end intersection_eq_l328_328221


namespace Jovana_shells_l328_328478

theorem Jovana_shells (initial_shells : ℕ) (added_shells : ℕ) (total_shells : ℕ) :
  initial_shells = 5 → added_shells = 12 → total_shells = initial_shells + added_shells → total_shells = 17 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end Jovana_shells_l328_328478


namespace CD_bisects_angle_EDF_l328_328523

noncomputable def Circle (O : Point) (r : ℝ) := { P : Point // dist O P = r }

variables (O P A B C D E F : Point)
variables (r : ℝ)
variables (hP_ext : ¬ Point ∈ Circle O r) -- P is an external point
variables (h_tangent_PA : ∀ (P A : Point), tangent_to_circle O r P A) -- PA is tangent to Circle O
variables (h_tangent_PB : ∀ (P B : Point), tangent_to_circle O r P B) -- PB is tangent to Circle O
variables (h_on_circle_C : C ∈ Circle O r) -- C is on Circle O
variables (h_perp_CD_AB : perpendicular CD AB) -- CD ⊥ AB at D
variables (h_tangent_E : tangent_to_circle O r E C) -- Tangent to Circle O at C intersects PA at E
variables (h_tangent_F : tangent_to_circle O r F C) -- Tangent to Circle O at C intersects PB at F

theorem CD_bisects_angle_EDF : angle_bisector CD (angle EDF) :=
sorry

end CD_bisects_angle_EDF_l328_328523


namespace adding_cell_can_create_four_axes_symmetry_l328_328381

theorem adding_cell_can_create_four_axes_symmetry 
  (initial_no_symmetry : Prop) 
  (possible_to_add_one_cell : initial_no_symmetry → ∃ (resulting_fig_has_four_symmetry_axes : Prop), resulting_fig_has_four_symmetry_axes) : 
  Prop :=
  possible_to_add_one_cell initial_no_symmetry

end adding_cell_can_create_four_axes_symmetry_l328_328381


namespace geometric_sequence_ratio_l328_328345

variable {a : ℕ → ℕ}
variables (S : ℕ → ℕ) (n : ℕ)
def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, ∀ n : ℕ, a n = a 1 * q^n

variables (h1 : a 5 - a 3 = 12) (h2 : a 6 - a 4 = 24)

theorem geometric_sequence_ratio :
  is_geometric_sequence a →
  ∃ S : ℕ → ℕ, (S n = ∑ i in range n, a i) ∧
  (S n / a n = 2 - 2^(1 - n)) :=
by
  sorry

end geometric_sequence_ratio_l328_328345


namespace num_true_props_l328_328516

-- Definitions from conditions
def is_arithmetic (a : ℕ → ℤ) := ∃ d, ∀ n, a (n + 1) = a n + d

def is_geometric (a : ℕ → ℤ) (ratio : ℤ) := ∀ n, a (n + 1) = ratio * a n

def geom_mean (x y : ℤ) := ±(R * (x * y) ^ (1 / 2 : ℚ))

-- Propositions
def prop1 (a : ℕ → ℤ) (p q r : ℕ) := is_arithmetic a ∧ p + q = r → a p + a q = a r

def prop2 (a : ℕ → ℤ) := (∀ n, a (n + 1) = 2 * a n) → is_geometric a 2

def prop3 := geom_mean 2 8 = 4 ∨ geom_mean 2 8 = -4

def prop4 (f : ℕ → ℤ) := (∃ d, ∀ n, f (n + 1) = f n + d) → ∃ m,  b , f n = m * n + b

theorem num_true_props : (∀ a p q r, ¬prop1 a p q r) ∧ (∀ a, ¬prop2 a) ∧ prop3 ∧ (∀ f, ¬prop4 f) → 1 = 1 := sorry

end num_true_props_l328_328516


namespace y_alone_days_l328_328785

-- Definitions of the conditions
variable (x y z : ℝ) -- Work rates
variable (work : ℝ)
variable (days : ℝ)

-- The conditions
def condition1 : Prop := x = 3 * y
def condition2 : Prop := z = y / 2
def condition3 : Prop := (x + y + z) * 20 = work

-- The statement to be proved
theorem y_alone_days : condition1 x y z ∧ condition2 x y z ∧ condition3 x y z work →
    (work = y * 90) :=
begin
    sorry
end

end y_alone_days_l328_328785


namespace sphere_surface_area_proof_l328_328139

noncomputable def rectangular_solid_diagonal : ℝ :=
  real.sqrt (6^2 + 8^2 + 10^2)

noncomputable def sphere_radius : ℝ :=
  rectangular_solid_diagonal / 2

noncomputable def sphere_surface_area : ℝ :=
  4 * real.pi * sphere_radius^2

theorem sphere_surface_area_proof :
  sphere_surface_area = 200 * real.pi :=
by
  unfold rectangular_solid_diagonal
  unfold sphere_radius
  unfold sphere_surface_area
  sorry -- proof omitted

end sphere_surface_area_proof_l328_328139


namespace _l328_328581

noncomputable def arithmetic_sequence_prop (a S : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 0 ∧ a 3 + S 5 = 42 ∧ 
  (a 1, a 4, a 13 ∧ a n = a 1 + (n - 1) * d ∧ 
  (a 1 + 3 * d)^2 = a 1 * (a 1 + 12 * d))

noncomputable def geometric_sequence_prop (a : ℕ → ℕ) : Prop :=
  ∃ a1 d : ℕ, a 1 = a1 ∧ d ≠ 0 ∧ ∀ n : ℕ, a n = a 1 + (n - 1) * d

noncomputable def arithmetic_sequence_term (a : ℕ → ℕ) : ℕ → ℕ :=
  λ n, 2 * n + 1

def k_th_term_sum (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in finset.range n, b (i + 1)

noncomputable def b_sequence_prop (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = 1 / (arithmetic_sequence_term a (n - 1) * arithmetic_sequence_term a n)

noncomputable theorem sequence_proof (a S b T : ℕ → ℕ) (h1 : ∀ n, a n = 2 * n + 1)
  (h2 : b_sequence_prop a b) :
  (∀ n, b 1 + b 2 + ... + b n = n / (2 * n + 1)) :=
begin
    sorry
end

end _l328_328581


namespace solution_to_system_of_equations_l328_328289

def augmented_matrix_system_solution (x y : ℝ) : Prop :=
  (x + 3 * y = 5) ∧ (2 * x + 4 * y = 6)

theorem solution_to_system_of_equations :
  ∃! (x y : ℝ), augmented_matrix_system_solution x y ∧ x = -1 ∧ y = 2 :=
by {
  sorry
}

end solution_to_system_of_equations_l328_328289


namespace O_l328_328313

-- Define the basic geometric setup
variables {O A B G A' B' O' : Point}

-- Assume that G is on the line segment AB
axiom G_on_AB : G ∈ segment A B

-- Define that AA' is perpendicular to AB and AG = AA'
axiom AA'_perp_AB : is_perpendicular (line_through A A') (line_through A B)
axiom AA'_eq_AG : distance A A' = distance A G

-- Define that BB' is perpendicular to AB and BG = BB'
axiom BB'_perp_AB : is_perpendicular (line_through B B') (line_through A B)
axiom BB'_eq_BG : distance B B' = distance B G

-- Define that AA' and BB' are on the same side of AB
axiom same_side_AA'_BB' : same_side (line_through A B) A' B'

-- Define that O' is the midpoint of A'B'
axiom O'_midpoint_A'B' : midpoint O' A' B'

-- Prove that O' remains stationary as G moves from A to B
theorem O'_remains_stationary :
  ∀ G, G ∈ segment A B → ∀ O', midpoint O' A' B' → ∀ A' B',
  is_perpendicular (line_through A A') (line_through A B) →
  is_perpendicular (line_through B B') (line_through A B) →
  distance A A' = distance A G → distance B B' = distance B G →
  same_side (line_through A B) A' B' → 
  (O' = midpoint_of_segment (A', B')) :=
begin
  sorry
end

end O_l328_328313


namespace maxwell_distance_traveled_l328_328688

theorem maxwell_distance_traveled
  (dist_MB : ℕ)
  (speed_M : ℕ)
  (speed_B : ℕ)
  (speed_A : ℕ)
  (eq_dist_A : ℕ) :
  dist_MB = 36 →
  speed_M = 3 →
  speed_B = 6 →
  speed_A = 9 →
  eq_dist_A = dist_MB / 2 →
  ∃ x : ℕ, (x / speed_M = (dist_MB - x) / speed_B) ∧
           (x / speed_M = (eq_dist_A - x) / speed_A) ∧
           x = 12 :=
by {
  intros,
  use 12,
  sorry
}

end maxwell_distance_traveled_l328_328688


namespace problem1_problem2_l328_328300

noncomputable def a (b c : ℝ) (A : ℝ) : ℝ :=
  real.sqrt (b^2 + c^2 - 2 * b * c * real.cos A)

noncomputable def sin_B (b a : ℝ) (A : ℝ) : ℝ :=
  b / (2 * real.sqrt (7))

noncomputable def sin_C (c a : ℝ) (A : ℝ) : ℝ :=
  c / (2 * real.sqrt (7))

theorem problem1 (b c : ℝ) (A : ℝ) (hA : A = real.pi / 3) (hb : b = 5) (hc : c = 4) :
  a b c A = real.sqrt 21 :=
by {
  sorry
}

theorem problem2 (b c a : ℝ) (A : ℝ) (hA : A = real.pi / 3) (hb : b = 5) (hc : c = 4) (ha : a = real.sqrt 21) :
  sin_B b a A * sin_C c a Α = 5 / 7 :=
by {
  sorry
}

end problem1_problem2_l328_328300


namespace angle_MKF_45_l328_328236

-- Definitions based on the conditions
variables {p : ℝ} (hp : 0 < p)
def parabola_point (p : ℝ) := (p / 2, p)
def focus_point (p : ℝ) := (p / 2, 0) -- the focus of y^2 = 2px is (p/2, 0)
def directrix_intersection (p : ℝ) := (-p / 2, 0)

-- Prove that the angle ∠MKF is 45 degrees
theorem angle_MKF_45 (hp : 0 < p) :
  let M := parabola_point p, F := focus_point p, K := directrix_intersection p in
  ∠ M K F = 45 :=
sorry

end angle_MKF_45_l328_328236


namespace statement_a_is_incorrect_l328_328085

-- Definitions based on the problem conditions
def term_deg (t : ℚ[X]) : ℕ :=
  t.degree.to_nat

def is_polynomial (f : ℚ[X]) : Prop :=
  true  -- Any expression in ℚ[X] is a polynomial

def polynomial_terms (f : ℚ[X]) : Multiset ℚ[X] :=
  f.support.map (λ n => monomial n (f.coeff n))

-- The incorrect statement that we need to prove
theorem statement_a_is_incorrect : 
  ¬ (term_deg (C (2 * real.pi) * X) = 3 ∧ term_deg (C real.pi * X^2) = 3 ∧ is_polynomial ((C (2 * real.pi) * X) + (C real.pi * X^2))) := 
by
  sorry

end statement_a_is_incorrect_l328_328085


namespace solve_for_x_l328_328020

theorem solve_for_x (x : ℝ) (h : 3^(3 * x) = 27^(1/3)) : x = 1 / 3 :=
by
  sorry

end solve_for_x_l328_328020


namespace domain_inequality_l328_328995

theorem domain_inequality (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - (1 - m) * x + m ≥ 0) ↔ (m ≥ 1/3) :=
by
  sorry

end domain_inequality_l328_328995


namespace f_zero_f_neg_one_f_odd_l328_328241

def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom domain_of_f : ∀ x, x ∈ ℝ
axiom functional_eq : ∀ x y : ℝ, f (x * y) = x * f y + y * f x

-- The proof goals
theorem f_zero : f 0 = 0 := sorry
theorem f_neg_one : f (-1) = 0 := sorry
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := sorry

end f_zero_f_neg_one_f_odd_l328_328241


namespace geometric_sum_ratio_l328_328353

-- Definitions and Conditions
variables {a : ℕ → ℕ}
variable q : ℕ
variable n : ℕ

-- Condition a₅ - a₃ = 12
axiom h1 : a 5 - a 3 = 12

-- Condition a₆ - a₄ = 24
axiom h2 : a 6 - a 4 = 24

-- The goal to prove: Sₙ / aₙ = 2 - 2⁽¹⁻ⁿ⁾, where Sₙ is the sum of first n terms and aₙ is the nth term
theorem geometric_sum_ratio (S : ℕ → ℕ) (a_n : ℕ) : 
  (S n) / (a n) = 2 - 2^(1 - n) := 
sorry

end geometric_sum_ratio_l328_328353


namespace harmonic_series_not_integer_l328_328715

theorem harmonic_series_not_integer (n : ℕ) (h : n > 1) : ¬ ∃ k : ℤ, (1/(1 : ℚ) + 1/(2 : ℚ) + 1/(3 : ℚ) + ... + 1/(n : ℚ)) = k := 
sorry

end harmonic_series_not_integer_l328_328715


namespace theater_ticket_sales_l328_328063

theorem theater_ticket_sales (A K : ℕ) (h1 : A + K = 275) (h2 :  12 * A + 5 * K = 2150) : K = 164 := by
  sorry

end theater_ticket_sales_l328_328063


namespace find_pairs_divisible_by_7_l328_328881

theorem find_pairs_divisible_by_7 :
  ∃ (x y : ℕ), (x, y) = (6, 5) ∧ (1000 + 100 * x + 10 * y + 2) % 7 = 0
  ∧ (1000 * x + 120 + y) % 7 = 0 :=
by
  exists 6, 5
  split
  { refl }
  { split
    { sorry }
    { sorry } }

end find_pairs_divisible_by_7_l328_328881


namespace compare_sqrt_terms_l328_328537

/-- Compare the sizes of 5 * sqrt 2 and 3 * sqrt 3 -/
theorem compare_sqrt_terms : 5 * Real.sqrt 2 > 3 * Real.sqrt 3 := 
by sorry

end compare_sqrt_terms_l328_328537


namespace find_c_find_cos_2B_l328_328636

-- Definitions and conditions extracted from Question 1
variables {A B C : ℝ} -- Angles of triangle
variables {a b c : ℝ} -- Sides opposite to angles A, B, C respectively
variable (triangle_perimeter : a + b + c = 8)
variable (sin_A_eq : 2 * sin A = 3 * sin B)
variable (condition_eq : b * sin A = (3 * b - c) * sin B)

-- Proof problem for Question 1
theorem find_c (h1 : triangle_perimeter) (h2 : sin_A_eq) (h3 : condition_eq) : c = 3 := 
sorry

-- Definitions and conditions extracted from Question 2
variables {is_isosceles : a = b ∨ a = c ∨ b = c}

-- Proof problem for Question 2
theorem find_cos_2B (h4 : is_isosceles) (h3 : condition_eq) : cos (2 * B) = 17 / 81 := 
sorry

end find_c_find_cos_2B_l328_328636


namespace twenty_fifth_subset_l328_328579

variable {n : ℕ} (M : Finset (Fin (n+1)))

def subset_index (m : ℕ) (indices : Finset ℕ) : Finset ℕ :=
  indices.filter (λ i, 2^(i-1) ∈ (Finset.range (2^m.succ)))

theorem twenty_fifth_subset (M : Finset (Fin (n+1))) (h : n ≥ 5):
  subset_index M 25 = {1, 4, 5} :=
by
  sorry

end twenty_fifth_subset_l328_328579


namespace mat_inv_int_entries_l328_328339

open scoped Matrix

theorem mat_inv_int_entries (A B : Matrix (Fin 2) (Fin 2) ℤ) 
  (hA : A.det ∈ {-1, 1})
  (h1 : (A + B).det ∈ {-1, 1})
  (h2 : (A + 2 • B).det ∈ {-1, 1})
  (h3 : (A + 3 • B).det ∈ {-1, 1})
  (h4 : (A + 4 • B).det ∈ {-1, 1}) :
  ∃ (C : Matrix (Fin 2) (Fin 2) ℤ), (C.det = 1 ∨ C.det = -1) ∧ C = (A + 5 • B) :=
sorry

end mat_inv_int_entries_l328_328339


namespace intersection_point_of_lines_l328_328862

theorem intersection_point_of_lines :
  ∃ (x y : ℚ), 
    (3 * y = -2 * x + 6) ∧ 
    (-2 * y = 7 * x + 4) ∧ 
    x = -24 / 17 ∧ 
    y = 50 / 17 :=
by
  sorry

end intersection_point_of_lines_l328_328862


namespace find_hyperbola_equation_l328_328257

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def circle_center : (ℝ × ℝ) := (3, 0)
noncomputable def circle_radius : ℝ := 2

noncomputable def hyperbola_properties (a b : ℝ) (x y : ℝ) : Prop :=
  let c := 3 in
  (a^2 + b^2 = 9) ∧
  ((∃ (x y : ℝ), (x, y) = circle_center ∧ abs (3 * b) / real.sqrt (a^2 + b^2) = 2))

theorem find_hyperbola_equation :
  ∃ (a b : ℝ), hyperbola a b x y ∧ hyperbola_properties a b x y ∧ (x^2 / 5 - y^2 / 4 = 1) :=
begin
  sorry
end

end find_hyperbola_equation_l328_328257


namespace harry_blue_weights_l328_328847

theorem harry_blue_weights (B : ℕ) 
  (h1 : 2 * B + 17 = 25) : B = 4 :=
by {
  -- proof code here
  sorry
}

end harry_blue_weights_l328_328847


namespace sum_of_first_200_numbers_is_999_l328_328060

def sequence (a : ℕ → ℤ) : Prop :=
  a 0 = 6 ∧
  a 1 = 3 ∧
  (∀ n ≥ 1, a (n + 1) = a (n - 1) + a n - 5)

noncomputable def sum_first_200 (a : ℕ → ℤ) : ℤ :=
  ∑ i in finRange 200, a i

theorem sum_of_first_200_numbers_is_999 (a : ℕ → ℤ) (h : sequence a) :
  sum_first_200 a = 999 :=
sorry

end sum_of_first_200_numbers_is_999_l328_328060


namespace number_of_divisors_of_720_l328_328661

theorem number_of_divisors_of_720 : 
  let n := 720
  let prime_factorization := [(2, 4), (3, 2), (5, 1)] 
  let num_divisors := (4 + 1) * (2 + 1) * (1 + 1)
  n = 2^4 * 3^2 * 5^1 →
  num_divisors = 30 := 
by
  -- Placeholder for the proof
  sorry

end number_of_divisors_of_720_l328_328661


namespace student_2005_says_1_l328_328297

def pattern : List ℕ := [1, 2, 3, 4, 3, 2]

def nth_number_in_pattern (n : ℕ) : ℕ :=
  List.nthLe pattern (n % 6) sorry  -- The index is (n-1) % 6 because Lean indices start at 0

theorem student_2005_says_1 : nth_number_in_pattern 2005 = 1 := 
  by
  -- The proof goes here
  sorry

end student_2005_says_1_l328_328297


namespace cost_fill_canC_and_canA_l328_328453

def radius_b : ℝ := sorry  -- Placeholder for the radius of can B
def height_b : ℝ := sorry  -- Placeholder for the height of can B
def cost_half_b : ℝ := 4   -- Cost to fill half of can B

def volume_b : ℝ := π * radius_b^2 * height_b
def volume_c : ℝ := π * (2 * radius_b)^2 * (height_b / 2)
def volume_a : ℝ := π * (3 * radius_b)^2 * (height_b / 3)

def cost_b : ℝ := cost_half_b * 2
def cost_c : ℝ := (volume_c / volume_b) * cost_b
def cost_a : ℝ := (volume_a / volume_b) * cost_b
def total_cost : ℝ := cost_c + cost_a

theorem cost_fill_canC_and_canA : total_cost = 40 := by
  sorry

end cost_fill_canC_and_canA_l328_328453


namespace sum_b_l328_328141

noncomputable def b : ℕ → ℝ
| 1     := 2
| 2     := 3
| (k+3) := (1 / 2) * b (k + 2) + (1 / 3) * b (k + 1)

theorem sum_b : (∑' n, b n) = 30 :=
sorry

end sum_b_l328_328141


namespace circumcircle_common_point_l328_328576

theorem circumcircle_common_point
  (A B C D P Q : Point)
  (parallelogram : Parallelogram A B C D)
  (h1 : SideLength A B < SideLength B C)
  (h2 : OnSameLine P B C)
  (h3 : OnSameLine Q C D)
  (h4 : Distance C P = Distance C Q) :
  ∃ G, G ≠ A ∧ (CircleThrough A P Q).contains G ∧ (CircleThrough A Q P).contains G :=
sorry  -- Proof will be inserted here later

end circumcircle_common_point_l328_328576


namespace integral_f_l328_328209

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 else cos x - 1

theorem integral_f : ∫ x in (-1 : ℝ)..(real.pi / 2), f x = (4/3 : ℝ) - real.pi / 2 :=
by
  sorry

end integral_f_l328_328209


namespace dice_sum_three_probability_l328_328079

theorem dice_sum_three_probability : 
    let outcomes := (1, 2) :: (2, 1) :: []
    let total_outcomes := 6 * 6
    let favorable_outcomes := 2
    let probability := (favorable_outcomes : ℚ) / total_outcomes
    probability = 1 / 18 :=
begin
  sorry
end

end dice_sum_three_probability_l328_328079


namespace trains_meet_after_time_l328_328760

/-- Given the lengths of two trains, the initial distance between them, and their speeds,
prove that they will meet after approximately 2.576 seconds. --/
theorem trains_meet_after_time 
  (length_train1 : ℝ) (length_train2 : ℝ) (initial_distance : ℝ)
  (speed_train1_kmph : ℝ) (speed_train2_mps : ℝ) :
  length_train1 = 87.5 →
  length_train2 = 94.3 →
  initial_distance = 273.2 →
  speed_train1_kmph = 65 →
  speed_train2_mps = 88 →
  abs ((initial_distance / ((speed_train1_kmph * 1000 / 3600) + speed_train2_mps)) - 2.576) < 0.001 := by
  sorry

end trains_meet_after_time_l328_328760


namespace sum_squares_second_15_l328_328783

theorem sum_squares_second_15 :
  (∑ i in Finset.range 15, (16 + i) ^ 2) = 8185 :=
by
  have h1 : (∑ i in Finset.range 15, (1 + i) ^ 2) = 1270 := by sorry
  -- Use that the sum of squares of the first 15 positive integers is 1270
  have h2 : (∑ i in Finset.range 30, (1 + i) ^ 2) = 9455 := by sorry
  -- You would compute the total sum of squares from 1 to 30 (1 to 16 encoded differently)
  have h3 : (∑ i in Finset.range 15, (16 + i) ^ 2) = (∑ i in Finset.range 30, (1 + i) ^ 2) - (∑ i in Finset.range 15, (1 + i) ^ 2) := by sorry
  -- Deduction of two parts
  exact calc
    (∑ i in Finset.range 15, (16 + i) ^ 2)
        = (∑ i in Finset.range 30, (1 + i) ^ 2) - (∑ i in Finset.range 15, (1 + i) ^ 2) : by sorry
    ... = 9455 - 1270 : by sorry
    ... = 8185 : by sorry

end sum_squares_second_15_l328_328783


namespace math_problem_proof_l328_328578

def sum_reciprocal_S (a : ℕ → ℕ) (S : ℕ → ℕ) (P : ℕ → ℕ × ℕ) (n : ℕ) : Prop :=
  (∀ n : ℕ, P n = (a n, a (n + 1))) →
  (∀ n : ℕ, P n.1 - P n.2 + 1 = 0) →
  a 1 = 1 →
  (∀ n : ℕ, S n = (finset.range n).sum (λ k, a (k + 1))) →
  (finset.range n).sum (λ k, 1 / S (k + 1)) = 2 * n / (n + 1)

theorem math_problem_proof : ∀ (a : ℕ → ℕ) (S : ℕ → ℕ) (P : ℕ → ℕ × ℕ), sum_reciprocal_S a S P :=
by sorry

end math_problem_proof_l328_328578


namespace triangle_formation_probability_l328_328401

theorem triangle_formation_probability : 
  let sticks := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] in
  let valid_triplets := 
    { (a, b, c) | a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧ a ≠ b ∧ b ≠ c ∧ a < b ∧ b < c ∧ a + b > c } in
  let total_triplets := finset.card (finset.filter (λ x, true) (finset.powerset_len 3 (finset.filter (λ x, true) sticks))) in
  let valid_triplets_count := finset.card valid_triplets in
  (valid_triplets_count : ℚ) / (total_triplets : ℚ) = 7 / 30 :=
by {
  sorry
}

end triangle_formation_probability_l328_328401


namespace min_value_of_Sn_minus_8an_l328_328215

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

axiom a_4 : a 4 = 7
axiom Sn_relation : ∀ n, 4 * S n = n * (a n + a (n + 1))

theorem min_value_of_Sn_minus_8an : 
  ∃ n, S n - 8 * a n = -56 :=
sorry

end min_value_of_Sn_minus_8an_l328_328215


namespace engine_efficiency_l328_328075

def P1 (P0 ω t : Real) : Real := P0 * (Real.sin (ω * t)) / (100 + (Real.sin (t^2)))
def P2 (P0 ω t : Real) : Real := 3 * P0 * (Real.sin (2 * ω * t)) / (100 + (Real.sin (2 * t)^2))

theorem engine_efficiency (P0 ω : Real) (A+ A- Q+ η : Real) 
  (hA- : A- = (2 / 3) * A+)
  (hQ+ : Q+ = A+)
  (hη : η = (A+ - A-) / Q+):
  η = 1 / 3 := 
by
  sorry

end engine_efficiency_l328_328075


namespace original_marketing_pct_correct_l328_328301

-- Define the initial and final percentages of finance specialization students
def initial_finance_pct := 0.88
def final_finance_pct := 0.90

-- Define the final percentage of marketing specialization students
def final_marketing_pct := 0.43333333333333335

-- Define the original percentage of marketing specialization students
def original_marketing_pct := 0.45333333333333335

-- The Lean statement to prove the original percentage of marketing students
theorem original_marketing_pct_correct :
  initial_finance_pct + (final_marketing_pct - initial_finance_pct) = original_marketing_pct := 
sorry

end original_marketing_pct_correct_l328_328301


namespace abs_inequality_solution_set_l328_328056

theorem abs_inequality_solution_set (x : ℝ) :
  |x| + |x - 1| < 2 ↔ - (1 / 2) < x ∧ x < (3 / 2) :=
by
  sorry

end abs_inequality_solution_set_l328_328056


namespace cement_and_gravel_needed_l328_328489

theorem cement_and_gravel_needed (total_concrete: ℕ) (ratio_cement ratio_sand ratio_gravel: ℕ) 
    (h_ratio: ratio_cement = 2 ∧ ratio_sand = 4 ∧ ratio_gravel = 5) (h_total: total_concrete = 121) :
    let total_parts := ratio_cement + ratio_sand + ratio_gravel in
    let cement_needed := total_concrete * ratio_cement / total_parts in
    let gravel_needed := total_concrete * ratio_gravel / total_parts in
    cement_needed = 22 ∧ gravel_needed = 55 :=
by
  sorry

end cement_and_gravel_needed_l328_328489


namespace building_floor_metrics_l328_328803

theorem building_floor_metrics (
  b l : ℝ
  (h1 : l = 3 * b)
  (paint_costs : ℝ) 
  (h2 : 3 * (b * l) + 4 * (b * l) + 5 * (b * l) = 3160)
) : 
  (|b - 9.37| < 0.01) ∧ (|l - 28.11| < 0.01) ∧ (|3 * (b * l) - 790| < 0.01) :=
by
  sorry

end building_floor_metrics_l328_328803


namespace at_least_one_red_ball_l328_328439

def total_balls : ℕ := 10
def red_balls : ℕ := 8
def black_balls : ℕ := 2
def drawn_balls : ℕ := 3

theorem at_least_one_red_ball : 
  (drawn_balls = 3) → 
  (red_balls + black_balls = total_balls) → 
  (red_balls = 8) →
  (black_balls = 2) →
  (∃ r b, r + b = 3 ∧ r ≥ 1 ∧ r ≤ red_balls ∧ b ≤ black_balls) :=
by {
  intros,
  sorry
}

end at_least_one_red_ball_l328_328439


namespace Kaleb_got_rid_of_7_shirts_l328_328336

theorem Kaleb_got_rid_of_7_shirts (initial_shirts : ℕ) (remaining_shirts : ℕ) 
    (h1 : initial_shirts = 17) (h2 : remaining_shirts = 10) : initial_shirts - remaining_shirts = 7 := 
by
  sorry

end Kaleb_got_rid_of_7_shirts_l328_328336


namespace symmetry_axis_one_l328_328734

-- Define the function and the equation of the symmetry axis
def f (x : ℝ) : ℝ := sin (4 * x - (Real.pi / 3))

-- The proof problem: prove that the equation of one of the symmetry axes is x = 11 * pi / 24
theorem symmetry_axis_one (x : ℝ) : (∃ k : ℤ, x = k * Real.pi / 4 + 5 * Real.pi / 24) ∧ 
  (x = 11 * Real.pi / 24) -> is_symmetry_axis (f x) :=
sorry

end symmetry_axis_one_l328_328734


namespace screamers_lineup_count_l328_328726

theorem screamers_lineup_count :
  let players := 12
  let lineup_size := 5
  let bob_yogi_group := 2  -- Represents Bob and Yogi group.
  (choose (players - bob_yogi_group) lineup_size) + 2 * (choose (players - lineup_size) (lineup_size - 1)) = 672 :=
by
  sorry

end screamers_lineup_count_l328_328726


namespace necessary_but_not_sufficient_l328_328935

theorem necessary_but_not_sufficient (a b : ℝ) : a^2 > b^2 → a > b > 0 := by
  sorry

end necessary_but_not_sufficient_l328_328935


namespace sum_first_5_terms_l328_328944

-- Definitions of arithmetic sequence and sum of the first n terms
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a (i + 1)

-- Given conditions based on the problem statement
def problem_conditions (a : ℕ → ℝ) : Prop :=
  arithmetic_sequence a ∧
  (∃ b, polynomial.has_roots (polynomial.X^2 - polynomial.X - 2 : polynomial ℝ) (b, a 4) ∧ a 2 = b)

-- Statement of the proof problem
theorem sum_first_5_terms
  (a : ℕ → ℝ)
  (h : problem_conditions a) :
  sum_first_n_terms a 5 = 5 / 2 :=
sorry

end sum_first_5_terms_l328_328944


namespace work_completion_days_l328_328472

theorem work_completion_days (a b : Type) (can_complete_together_in : a → b → Nat → Prop)
  (days_b_alone : b → Nat → Prop) (h1 : can_complete_together_in a b 8) (h2 : days_b_alone b 24) :
  ∃ x : ℕ, can_complete_together_in a b x ∧ x = 12 :=
sorry

end work_completion_days_l328_328472


namespace intersect_and_missing_vertex_l328_328384

-- Define the problem parameters
def point1 : ℝ × ℝ := (2, -4)
def point2 : ℝ × ℝ := (10, 8)
def vertex1 : ℝ × ℝ := (5, -3)
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
def missing_vertex_coords (v : ℝ × ℝ) : ℝ × ℝ := (7, 7)

-- Define the intersection point calculation
def intersection_point := midpoint point1 point2

-- Define the proof that the diagonals intersect at the correct point and find the missing vertex
theorem intersect_and_missing_vertex (v3 : ℝ × ℝ) :
  intersection_point = (6, 2) ∧ 
  missing_vertex_coords v3 = (7, 7) :=
sorry

end intersect_and_missing_vertex_l328_328384


namespace geometric_sequence_sum_l328_328358

def geometric_sequence_props (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  (∀ n, a n = a 1 * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (1 - 2^n) / (1 - 2)) ∧ 
  (a 5 - a 3 = 12) ∧ 
  (a 6 - a 4 = 24)

theorem geometric_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : geometric_sequence_props a S) :
  ∀ n, S n / a n = 2 - 2^(1 - n) :=
by
  sorry

end geometric_sequence_sum_l328_328358


namespace cos_of_angle_in_third_quadrant_l328_328626

theorem cos_of_angle_in_third_quadrant (B : ℝ) (h1 : π < B ∧ B < 3 * π / 2) (h2 : Real.sin B = -5 / 13) : Real.cos B = -12 / 13 := 
by 
  sorry

end cos_of_angle_in_third_quadrant_l328_328626


namespace four_digit_numbers_count_distinct_four_digit_odd_numbers_count_l328_328070

-- Definitions for the problem conditions
def digits : List ℕ := [0, 1, 2, 3, 4, 5]
def odd_digits : List ℕ := [1, 3, 5]

-- Theorem statements
theorem four_digit_numbers_count : 
  (∃ d1 d2 d3 d4 : ℕ, 
      d1 ∈ digits ∧ d1 ≠ 0 ∧
      d2 ∈ digits ∧
      d3 ∈ digits ∧
      d4 ∈ digits) → 
  1080 :=
sorry

theorem distinct_four_digit_odd_numbers_count :
  (∃ d1 d2 d3 d4 : ℕ, 
      d1 ∈ digits ∧ d1 ≠ 0 ∧
      d2 ∈ digits ∧ d3 ∈ digits ∧ 
      d4 ∈ odd_digits ∧ 
      List.nodup ([d1, d2, d3, d4])) → 
  144 :=
sorry

end four_digit_numbers_count_distinct_four_digit_odd_numbers_count_l328_328070


namespace novels_in_shipment_l328_328109

theorem novels_in_shipment (N : ℕ) (H1: 225 = (3/4:ℚ) * N) : N = 300 := 
by
  sorry

end novels_in_shipment_l328_328109


namespace inequality_solution_l328_328176

theorem inequality_solution :
  {x : ℝ | (x^3 - 4 * x) / (x^2 - 4 * x + 3) > 0} =
    set.Ioo (-∞) (-2) ∪ set.Ioo (-2) 0 ∪ set.Ioo 1 2 ∪ set.Ioo 3 ∞ :=
by {
  sorry
}

end inequality_solution_l328_328176


namespace trapezoid_ratio_l328_328066

noncomputable theory

-- Definitions based on the given conditions
variables {EF FG GH HE : ℕ}
variable {Q : (fin 2) → ℝ} -- point Q has coordinates in 2D space
variables {x : ℝ} -- let EQ = x
variables {p q : ℕ} -- relatively prime positive integers p and q

-- Assume the conditions
def trapezoid_conditions : Prop :=
  EF = 86 ∧ FG = 38 ∧ GH = 23 ∧ HE = 66 ∧ (∃ Q : (fin 2) → ℝ, are_parallel EF GH ∧ tangent_to Q FG ∧ tangent_to Q HE)

-- Prove the ratio EQ:QF and p + q = 2879
theorem trapezoid_ratio : trapezoid_conditions → ∃ p q : ℕ, (p / q) = (2832 / 47) ∧ p + q = 2879 :=
by
  intros h
  sorry

end trapezoid_ratio_l328_328066


namespace Jonathan_total_exercise_time_l328_328335

theorem Jonathan_total_exercise_time :
  (let monday_speed := 2
     let wednesday_speed := 3
     let friday_speed := 6
     let distance := 6
     let monday_time := distance / monday_speed
     let wednesday_time := distance / wednesday_speed
     let friday_time := distance / friday_speed
   in monday_time + wednesday_time + friday_time = 6)
:= sorry

end Jonathan_total_exercise_time_l328_328335


namespace incorrect_tetrahedron_section_l328_328795

theorem incorrect_tetrahedron_section 
  (tetrahedron : Type) 
  (section : Set (tetrahedron))
  (polygon : ∀ t : tetrahedron, (section t) → Prop)
  (four_sided_polygon : ∃ t : tetrahedron, ∃ p : (section t), polygon p)
  (planar_intersection : ∀ (line1 line2 : (tetrahedron → Prop)), ∃ unique_point : tetrahedron, line1 unique_point ∧ line2 unique_point → Prop) :
  ∃ (t : tetrahedron) (p : (section t)), ¬(planar_intersection (λ _ , p) t) :=
sorry

end incorrect_tetrahedron_section_l328_328795


namespace benton_school_earnings_l328_328902

noncomputable def total_earnings_for_students 
  (students_adams : ℕ) (days_adams : ℕ)
  (students_benton : ℕ) (days_benton : ℕ)
  (students_camden : ℕ) (days_camden : ℕ)
  (total_amount : ℝ) :=
  let total_student_days := (students_adams * days_adams) + (students_benton * days_benton) + (students_camden * days_camden) in
  let daily_wage := total_amount / total_student_days in
  daily_wage * (students_benton * days_benton)

theorem benton_school_earnings : total_earnings_for_students 4 4 5 6 6 7 780 = 266 := 
by sorry

end benton_school_earnings_l328_328902


namespace bushes_for_60_zucchinis_l328_328380

/-- 
Given:
1. Each blueberry bush yields twelve containers of blueberries.
2. Four containers of blueberries can be traded for three pumpkins.
3. Six pumpkins can be traded for five zucchinis.

Prove that eight bushes are needed to harvest 60 zucchinis.
-/
theorem bushes_for_60_zucchinis (bush_to_containers : ℕ) (containers_to_pumpkins : ℕ) (pumpkins_to_zucchinis : ℕ) :
  (bush_to_containers = 12) → (containers_to_pumpkins = 4) → (pumpkins_to_zucchinis = 6) →
  ∃ bushes_needed, bushes_needed = 8 ∧ (60 * pumpkins_to_zucchinis / 5 * containers_to_pumpkins / 3 / bush_to_containers) = bushes_needed :=
by
  intros h1 h2 h3
  sorry

end bushes_for_60_zucchinis_l328_328380


namespace minimum_value_l328_328570

noncomputable def f (a b c x : ℝ) : ℝ := real.sqrt (x^2 + a) + real.sqrt ((c - x)^2 + b)

theorem minimum_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x, f a b c x = real.sqrt (c^2 + (real.sqrt a + real.sqrt b)^2) := 
sorry

end minimum_value_l328_328570


namespace length_of_line_segment_AB_l328_328652

theorem length_of_line_segment_AB :
  ∃ A B : ℝ × ℝ, 
  (∃ θ : ℝ, A = (4 * Float.sin θ, θ) ∧ A.1 * Float.cos θ = 1) ∧ 
  (∃ θ' : ℝ, B = (4 * Float.sin θ', θ') ∧ B.1 * Float.cos θ' = 1) ∧ 
  dist A B = 2 * Real.sqrt 3 := sorry

end length_of_line_segment_AB_l328_328652


namespace milk_mixture_l328_328612

theorem milk_mixture (x : ℝ) : 
  (2.4 + 0.1 * x) / (8 + x) = 0.2 → x = 8 :=
by
  sorry

end milk_mixture_l328_328612


namespace right_triangle_second_arm_square_l328_328293

theorem right_triangle_second_arm_square :
  ∀ (k : ℤ) (a : ℤ) (c : ℤ) (b : ℤ),
  a = 2 * k + 1 → 
  c = 2 * k + 3 → 
  a^2 + b^2 = c^2 → 
  b^2 ≠ a * c ∧ b^2 ≠ (c / a) ∧ b^2 ≠ (a + c) ∧ b^2 ≠ (c - a) :=
by sorry

end right_triangle_second_arm_square_l328_328293


namespace scientific_notation_2600000_l328_328150

theorem scientific_notation_2600000 : ∃ (c : ℝ) (n : ℤ), 1 ≤ c ∧ c < 10 ∧ 2600000 = c * 10 ^ n ∧ c = 2.6 ∧ n = 6 :=
by
  use 2.6, 6
  split; try {norm_num}
  exact ⟨by norm_num, by norm_num, by norm_num⟩

end scientific_notation_2600000_l328_328150


namespace five_digit_even_number_count_five_digit_even_number_units_digit_sum_l328_328012

open Finset

theorem five_digit_even_number_count :
  (∑ s in (finset.range 10).powerset.filter (λ s, s.card = 5), 
     if s ∩ (finset.of_list [0, 2, 4, 6, 8]) ≠ ∅ then 1 else 0) = 
  (nat.factorial 9 / nat.factorial 5) + 4 * (nat.factorial 8 / nat.factorial 4) :=
sorry

theorem five_digit_even_number_units_digit_sum :
  (∑ s in (finset.range 10).powerset.filter (λ s, s.card = 5), 
     if s ∩ (finset.of_list [0, 2, 4, 6, 8]) ≠ ∅ then 
     ∑ d in s ∩ (finset.of_list [2, 4, 6, 8]), d * (nat.factorial 8 / nat.factorial 4) else 0) = 
  (2 + 4 + 6 + 8) * (nat.factorial 8 / nat.factorial 4) :=
sorry

end five_digit_even_number_count_five_digit_even_number_units_digit_sum_l328_328012


namespace tom_remaining_balloons_l328_328451

theorem tom_remaining_balloons (initial_balloons : ℕ) (balloons_given : ℕ) (balloons_remaining : ℕ) 
  (h1 : initial_balloons = 30) (h2 : balloons_given = 16) : balloons_remaining = 14 := 
by
  sorry

end tom_remaining_balloons_l328_328451


namespace part_one_part_two_l328_328210

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem part_one :
  ∀ x m : ℕ, f x ≤ -m^2 + 6 * m → 1 ≤ m ∧ m ≤ 5 := 
by
  sorry

theorem part_two (a b c : ℝ) (h : 3 * a + 4 * b + 5 * c = 1) :
  (a^2 + b^2 + c^2) ≥ (1 / 50) :=
by
  sorry

end part_one_part_two_l328_328210


namespace PS_bisector_of_angle_APQ_l328_328743

open EuclideanGeometry

-- Define the problem
theorem PS_bisector_of_angle_APQ (A B C D E M P Q S : Point) : 
  (regular_pentagon A B C D E) →
  (center M A B C D E) →
  (P ∈ interior_segment(D, M)) →
  (Q ∈ (circumscribed_circle(triangle(A, B, P))) ∩ (line_segment(A, E))) →
  (Q ≠ A) →
  (S = perpendicular_intersection(P, line(C, D), line(A, E))) →
  is_angle_bisector(line(P, S), angle(A, P, Q)) :=
sorry

end PS_bisector_of_angle_APQ_l328_328743


namespace find_k_l328_328864

theorem find_k (k : ℝ) : 
  let a := 6
  let b := 25
  let root := (-25 - Real.sqrt 369) / 12
  6 * root^2 + 25 * root + k = 0 → k = 32 / 3 :=
sorry

end find_k_l328_328864


namespace tan_double_angle_l328_328945

-- Define the basic elements
def point (x y : ℝ) := (x, y)

-- Conditions given in the problem
def α_tan_through_point (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P in -y / x

def given_point : ℝ × ℝ := (1, 2)

noncomputable def double_angle_tan (tan_α : ℝ) : ℝ :=
  (2 * tan_α) / (1 - tan_α^2)

-- Statement to be proved
theorem tan_double_angle : 
  double_angle_tan (α_tan_through_point given_point) = 4 / 3 :=
by
  sorry

end tan_double_angle_l328_328945


namespace infinitely_many_n_even_floor_l328_328226

theorem infinitely_many_n_even_floor (α : ℝ) (h : α > 0) :
  ∃ᶠ n : ℕ in at_top, even (⌊n ^ 2 * α⌋₊) :=
by sorry

end infinitely_many_n_even_floor_l328_328226


namespace max_non_multiple_subset_l328_328153

-- Define the sequence of natural numbers
def sequence (n : ℕ) : ℕ := 2 * n - 1

-- Define the conditions of the problem
def is_valid_selection (S : set ℕ) : Prop :=
  ∀ x y ∈ S, x ≠ y → ¬(x % y = 0)

-- State the theorem
theorem max_non_multiple_subset :
  ∃ S : set ℕ, (∀ x ∈ S, x ∈ (set.range sequence ∩ set.Icc 1 199)) ∧ is_valid_selection S ∧ S.card = 67 :=
sorry

end max_non_multiple_subset_l328_328153


namespace find_linear_function_l328_328232

theorem find_linear_function (a m : ℝ) : 
  (∀ x y : ℝ, (x, y) = (-2, -3) ∨ (x, y) = (-1, -1) ∨ (x, y) = (0, m) ∨ (x, y) = (1, 3) ∨ (x, y) = (a, 5) → 
  y = 2 * x + 1) → 
  (m = 1 ∧ a = 2) :=
by
  sorry

end find_linear_function_l328_328232


namespace thermal_engine_efficiency_l328_328076

noncomputable def P1 (P0 ω t : ℝ) : ℝ :=
  P0 * (Real.sin (ω * t) / (100 + Real.sin (t ^ 2)))

noncomputable def P2 (P0 ω t : ℝ) : ℝ :=
  3 * P0 * (Real.sin (2 * ω * t) / (100 + Real.sin ((2 * t) ^ 2)))

noncomputable def A_plus (P0 ω : ℝ) : ℝ :=
  ∫ t in 0..(π / ω), P1 P0 ω t

noncomputable def A_minus (P0 ω : ℝ) : ℝ :=
  ∫ t in 0..(π / (2 * ω)), P2 P0 ω t

theorem thermal_engine_efficiency (P0 ω : ℝ) :
  (∫ t in 0..(π / ω), P1 P0 ω t) - (∫ t in 0..(π / (2 * ω)), P2 P0 ω t) = 
  (1 / 3) * (∫ t in 0..(π / ω), P1 P0 ω t) :=
by
  sorry

end thermal_engine_efficiency_l328_328076


namespace geometric_sequence_a7_value_l328_328590

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_a7_value (a : ℕ → ℝ) (r : ℝ) (S : ℕ → ℝ) :
  (∀ n, 0 < a n) →
  (geometric_sequence a r) →
  (S 4 = 3 * S 2) →
  (a 3 = 2) →
  (S n = a 1 + a 1 * r + a 1 * r^2 + a 1 * r^3) →
  a 7 = 8 :=
by
  sorry

end geometric_sequence_a7_value_l328_328590


namespace function_identity_l328_328880

theorem function_identity (f : ℕ → ℕ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n)
  (h3 : ∀ n : ℕ, f (n + 1) > f n) : 
  ∀ n : ℕ, f n = n :=
sorry

end function_identity_l328_328880


namespace max_product_abs_diff_l328_328893

theorem max_product_abs_diff {a : ℝ} {b : ℝ} {c : ℝ} {d : ℝ} {e : ℝ}
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : 0 ≤ b ∧ b ≤ 1)
  (h3 : 0 ≤ c ∧ c ≤ 1)
  (h4 : 0 ≤ d ∧ d ≤ 1)
  (h5 : 0 ≤ e ∧ e ≤ 1) :
  (\prod i j in Finset.range 5 \pair (i < j), |a.list_nth i - a.list_nth j| 
   = \frac{3 \sqrt{21}}{38416} := 
sorry

end max_product_abs_diff_l328_328893


namespace total_distance_traveled_is_correct_l328_328814

-- Definitions of given conditions
def Vm : ℕ := 8
def Vr : ℕ := 2
def round_trip_time : ℝ := 1

-- Definitions needed for intermediate calculations (speed computations)
def upstream_speed (Vm Vr : ℕ) : ℕ := Vm - Vr
def downstream_speed (Vm Vr : ℕ) : ℕ := Vm + Vr

-- The equation representing the total time for the round trip
def time_equation (D : ℝ) (Vm Vr : ℕ) : Prop :=
  D / upstream_speed Vm Vr + D / downstream_speed Vm Vr = round_trip_time

-- Prove that the total distance traveled by the man is 7.5 km
theorem total_distance_traveled_is_correct : ∃ D : ℝ, D / upstream_speed Vm Vr + D / downstream_speed Vm Vr = round_trip_time ∧ 2 * D = 7.5 :=
by
  sorry

end total_distance_traveled_is_correct_l328_328814


namespace range_of_a_l328_328600

-- Definitions based on given conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - x
def g (x : ℝ) : ℝ := Real.log x
def h (a : ℝ) (x : ℝ) : ℝ := f a x - g x

-- Definition of k based on the transformation used in the solution
def k (x : ℝ) : ℝ := (Real.log x + x) / (x ^ 2)

theorem range_of_a (a : ℝ) (ha : a ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h a x1 = 0 ∧ h a x2 = 0) ↔ 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l328_328600


namespace complex_modulus_l328_328230

theorem complex_modulus (a : ℝ) :
  let i := Complex.i in
  let Z := i * (3 - a * i) in
  Complex.abs Z = 5 → a = 4 ∨ a = -4 :=
by
  intro h
  sorry

end complex_modulus_l328_328230


namespace wendy_percentage_accounting_l328_328458

noncomputable def years_as_accountant : ℕ := 25
noncomputable def years_as_manager : ℕ := 15
noncomputable def total_lifespan : ℕ := 80

def total_years_in_accounting : ℕ := years_as_accountant + years_as_manager

def percentage_of_life_in_accounting : ℝ := (total_years_in_accounting / total_lifespan) * 100

theorem wendy_percentage_accounting : percentage_of_life_in_accounting = 50 := by
  unfold total_years_in_accounting
  unfold percentage_of_life_in_accounting
  sorry

end wendy_percentage_accounting_l328_328458


namespace sin_cos_identity_second_quadrant_l328_328682

open Real

theorem sin_cos_identity_second_quadrant (α : ℝ) (hcos : cos α < 0) (hsin : sin α > 0) :
  (sin α / cos α) * sqrt ((1 / (sin α)^2) - 1) = -1 :=
sorry

end sin_cos_identity_second_quadrant_l328_328682


namespace find_m_l328_328253

noncomputable def f (x a : ℝ) : ℝ := (2 / 3) * x^3 - 2 * a * x^2 - 3 * x

theorem find_m (a m : ℝ) (h_tangent: ∀ (P : ℝ × ℝ), P = (1, m) → (∃ b, 3 * P.1 - P.2 + b = 0)) (h_deriv: ∀ x, deriv (fun y => (2 / 3) * y^3 - 2 * a * y^2 - 3 * y) 1 = 3) : 
  m = -1 / 3 := 
sorry

end find_m_l328_328253


namespace determine_y_l328_328187

theorem determine_y (y : ℝ) (y_nonzero : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 := 
sorry

end determine_y_l328_328187


namespace least_number_of_tablets_l328_328110

theorem least_number_of_tablets (tablets_A : ℕ) (tablets_B : ℕ) (hA : tablets_A = 10) (hB : tablets_B = 13) :
  ∃ n, ((tablets_A ≤ 10 → n ≥ tablets_A + 2) ∧ (tablets_B ≤ 13 → n ≥ tablets_B + 2)) ∧ n = 12 :=
by
  sorry

end least_number_of_tablets_l328_328110


namespace min_lambda_for_inequality_l328_328951

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * Real.log x - x^2
noncomputable def g (x : ℝ) (lambda : ℝ) : ℝ :=
  (lambda - 1) * x^2 + 2 * (lambda - 1) * x - 2

theorem min_lambda_for_inequality (h : ∀ x : ℝ, 0 < x → f x 2 ≤ g x 2)
: (2 : ℕ) :=
begin
  sorry,
end

end min_lambda_for_inequality_l328_328951


namespace tan_alpha_value_l328_328904

variable (α : Real)
variable (h1 : Real.sin α = 4/5)
variable (h2 : π/2 < α ∧ α < π)

theorem tan_alpha_value : Real.tan α = -4/3 := by
  sorry

end tan_alpha_value_l328_328904


namespace number_of_non_prime_counterexamples_l328_328886

def sum_of_digits (n : Nat) : Nat :=
  n.digits.sum

def none_zero_digits (n : Nat) : Prop :=
  ∀ d ∈ n.digits, d ≠ 0

def is_prime (n : Nat) : Prop :=
  sorrry -- Needs real implementation or imported to check if a number is prime

theorem number_of_non_prime_counterexamples : 
  (Finset.filter (λ n, sum_of_digits n = 5 ∧ none_zero_digits n ∧ ¬ is_prime n) (@Finset.range Nat _ 100000)).card = 7 :=
sorry

end number_of_non_prime_counterexamples_l328_328886


namespace no_adj_yellow_permutations_l328_328545

theorem no_adj_yellow_permutations :
  let n1 := 3 -- green balls
      n2 := 2 -- red balls
      n3 := 2 -- white balls
      n4 := 3 -- yellow balls
  in
  let total_permutations := (n1 + n2 + n3 + n4)! / (n1! * n2! * n3! * n4!) in
  let non_adj_yellow_permutations :=
    ((n1 + n2 + n3)! / (n1! * n2! * n3!)) * (Nat.choose (n1 + n2 + n3 + 1) n4)
  in
  non_adj_yellow_permutations = 11760 := by
  sorry

end no_adj_yellow_permutations_l328_328545


namespace spadesuit_problem_l328_328563

def spadesuit (x y : ℝ) : ℝ := x - (1 / y)

theorem spadesuit_problem :
  spadesuit 3 (spadesuit 4 3) = 30 / 11 :=
by
  sorry

end spadesuit_problem_l328_328563


namespace cyc_sum_ge_zero_l328_328681

noncomputable theory
open_locale classical

variables {x y z : ℝ}
hypothesis h_x_pos : 0 < x
hypothesis h_y_pos : 0 < y
hypothesis h_z_pos : 0 < z
hypothesis h_xyz_ge_1 : x * y * z ≥ 1

theorem cyc_sum_ge_zero : 
(∑ cyc in [⟦⟨x, y, z⟩, ⟨y, z, x⟩, ⟨z, x, y⟩⟧], (cyc.1^5 - cyc.1^2) / (cyc.1^5 + cyc.2^2 + cyc.3^2)) ≥ 0 :=
sorry

end cyc_sum_ge_zero_l328_328681


namespace dan_gave_sally_14_cards_l328_328010

theorem dan_gave_sally_14_cards :
  (∃ D : ℕ, 27 - 27 + D + 20 = 34) → ∃ D : ℕ, D = 14 :=
by
  intro h1
  cases h1 with D hD
  existsi (14 : ℕ)
  simp at hD
  exact hD

end dan_gave_sally_14_cards_l328_328010


namespace dk_is_odd_l328_328678

def NTypePermutations (k : ℕ) (x : Fin (3 * k + 1) → ℕ) : Prop :=
  (∀ i j : Fin (k + 1), i < j → x i < x j) ∧
  (∀ i j : Fin (k + 1), i < j → x (k + 1 + i) > x (k + 1 + j)) ∧
  (∀ i j : Fin (k + 1), i < j → x (2 * k + 1 + i) < x (2 * k + 1 + j))

def countNTypePermutations (k : ℕ) : ℕ :=
  sorry -- This would be the count of all N-type permutations, use advanced combinatorics or algorithms

theorem dk_is_odd (k : ℕ) (h : 0 < k) : ∃ d : ℕ, countNTypePermutations k = 2 * d + 1 :=
  sorry

end dk_is_odd_l328_328678


namespace t_plus_inv_t_eq_three_l328_328909

theorem t_plus_inv_t_eq_three {t : ℝ} (h : t^2 - 3 * t + 1 = 0) (hnz : t ≠ 0) : t + 1 / t = 3 :=
sorry

end t_plus_inv_t_eq_three_l328_328909


namespace smallest_E_of_positive_reals_l328_328387

noncomputable def E (a b c : ℝ) : ℝ :=
  (a^3) / (1 - a^2) + (b^3) / (1 - b^2) + (c^3) / (1 - c^2)

theorem smallest_E_of_positive_reals (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) : 
  E a b c = 1 / 8 := 
sorry

end smallest_E_of_positive_reals_l328_328387


namespace common_rational_root_l328_328186

def polynomial1 (x : ℚ) (a b c : ℚ) : Prop :=
  45 * x^4 + a * x^3 + b * x^2 + c * x + 8 = 0

def polynomial2 (x : ℚ) (d e f g : ℚ) : Prop :=
  8 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 45 = 0

theorem common_rational_root (a b c d e f g : ℚ) : 
  (∃ k : ℚ, k = -⅓ ∧ k < 0 ∧ (polynomial1 k a b c) ∧ (polynomial2 k d e f g)) :=
  sorry

end common_rational_root_l328_328186


namespace perimeter_ratio_l328_328026

def original_paper : ℕ × ℕ := (12, 8)
def folded_paper : ℕ × ℕ := (original_paper.1, original_paper.2 / 2)
def small_rectangle : ℕ × ℕ := (folded_paper.1 / 2, folded_paper.2)

def perimeter (rect : ℕ × ℕ) : ℕ :=
  2 * (rect.1 + rect.2)

theorem perimeter_ratio :
  perimeter small_rectangle = 1 / 2 * perimeter original_paper :=
by
  sorry

end perimeter_ratio_l328_328026


namespace max_lateral_surface_area_l328_328243

theorem max_lateral_surface_area (l w : ℝ) (h : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : 2 * (l + w) = 36) 
  (h2 : h = w) 
  (h3 : 2 * real.pi * r = l) 
  (h4 : A = 2 * real.pi * r * h) :
  A ≤ 81 :=
by
  sorry

end max_lateral_surface_area_l328_328243


namespace angle_a_b_is_90_degrees_l328_328934

open Real

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

noncomputable def angle_between (x y : V) : ℝ :=
  real.arccos ((inner x y) / (norm x * norm y))

theorem angle_a_b_is_90_degrees (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) 
  (h₄ : ∥a + b + c∥ = ∥a - b - c∥) : angle_between a b = π / 2 :=
by
  sorry

end angle_a_b_is_90_degrees_l328_328934


namespace parabola_min_distance_a_l328_328940

noncomputable def directrix_distance (P : Real × Real) (a : Real) : Real :=
abs (P.2 + 1 / (4 * a))

noncomputable def distance (P Q : Real × Real) : Real :=
Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem parabola_min_distance_a (a : Real) :
  (∀ (P : Real × Real), P.2 = a * P.1^2 → 
    distance P (2, 0) + directrix_distance P a = Real.sqrt 5) ↔ 
    a = 1 / 4 ∨ a = -1 / 4 :=
by
  sorry

end parabola_min_distance_a_l328_328940


namespace problem_statement_l328_328898

theorem problem_statement : 
  (∀ (base : ℤ) (exp : ℕ), (-3) = base ∧ 2 = exp → (base ^ exp ≠ -9)) :=
by
  sorry

end problem_statement_l328_328898


namespace value_of_t_plus_one_over_t_l328_328913

theorem value_of_t_plus_one_over_t
  (t : ℝ)
  (h1 : t^2 - 3 * t + 1 = 0)
  (h2 : t ≠ 0) :
  t + 1 / t = 3 :=
by
  sorry

end value_of_t_plus_one_over_t_l328_328913


namespace range_of_t_l328_328599

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem range_of_t (t : ℝ) :  
  (∀ a > 0, ∀ x₀ y₀, 
    (a - a * Real.log x₀) / x₀^2 = 1 / 2 ∧ 
    y₀ = (a * Real.log x₀) / x₀ ∧ 
    x₀ = 2 * y₀ ∧ 
    a = Real.exp 1 ∧ 
    f (f x) = t -> t = 0) :=
by
  sorry

end range_of_t_l328_328599


namespace joan_spent_on_toys_l328_328325

theorem joan_spent_on_toys :
  let toy_cars := 14.88
  let toy_trucks := 5.86
  toy_cars + toy_trucks = 20.74 :=
by
  let toy_cars := 14.88
  let toy_trucks := 5.86
  sorry

end joan_spent_on_toys_l328_328325


namespace stick_cut_probability_l328_328829

theorem stick_cut_probability :
  let stick_length := 2
  let special_mark := 0.6
  ∀ (C : ℝ), 0 ≤ C ∧ C ≤ 1.4 →
    (2 - C ≥ 3 * C → set.prob (λ C, 0 ≤ C ∧ C ≤ 1.4) (λ C, 0 ≤ C ∧ C ≤ 0.5)) = (5 / 14) :=
by
  sorry

end stick_cut_probability_l328_328829


namespace football_team_practice_hours_l328_328810

theorem football_team_practice_hours (daily_hours : ℕ) (missed_days : ℕ) (week_days : ℕ) (total_hours : ℕ) :
  daily_hours = 5 ∧ missed_days = 1 ∧ week_days = 7 → total_hours = 30 :=
by
  intros h
  cases h with h1 h'
  cases h' with h2 h3
  rw [h1, h2, h3]
  sorry

end football_team_practice_hours_l328_328810


namespace find_xy_l328_328546

theorem find_xy (x y : ℝ) :
  (4 * x^2 + 3 * y^2 + 8 * x - 6 * y + 30 = 50) →
  (x = -1 ∧ y = 1 + real.sqrt (29 / 3)) ∨ (x = -1 ∧ y = 1 - real.sqrt (29 / 3)) :=
sorry

end find_xy_l328_328546


namespace smaller_cube_surface_area_l328_328821

theorem smaller_cube_surface_area (edge_length : ℝ) (h : edge_length = 12) :
  let sphere_diameter := edge_length
  let smaller_cube_side := sphere_diameter / Real.sqrt 3
  let surface_area := 6 * smaller_cube_side ^ 2
  surface_area = 288 := by
  sorry

end smaller_cube_surface_area_l328_328821


namespace find_phi_l328_328953

noncomputable def f (φ : ℝ) (x : ℝ) :=
  real.sin (real.sqrt 3 * x + φ)

noncomputable def f' (φ : ℝ) (x : ℝ) :=
  real.sqrt 3 * real.cos (real.sqrt 3 * x + φ)

noncomputable def g (φ : ℝ) (x : ℝ) :=
  f φ x + f' φ x

theorem find_phi (φ : ℝ) (h1 : 0 < φ) (h2 : φ < real.pi) (h3 : ∀ x : ℝ, g φ x = - g φ (-x)) :
  φ = 2 * real.pi / 3 :=
by
  sorry

end find_phi_l328_328953


namespace scientific_notation_of_2600000_l328_328148

theorem scientific_notation_of_2600000 :
    ∃ (a : ℝ) (b : ℤ), (2600000 : ℝ) = a * 10^b ∧ a = 2.6 ∧ b = 6 :=
begin
    sorry
end

end scientific_notation_of_2600000_l328_328148


namespace min_nsatisfying_condition_l328_328684

open Set

noncomputable def is_partition (A B : Set ℕ) (I : Set ℕ) : Prop :=
  A ⊆ I ∧ B ⊆ I ∧ A ∩ B = ∅ ∧ A ∪ B = I ∧ A ≠ ∅ ∧ B ≠ ∅

def exists_square_sum (A : Set ℕ) : Prop :=
  ∃ x y ∈ A, (x + y).is_square

theorem min_nsatisfying_condition (n : ℕ) (h : n ≥ 3) :
  (∀ A B : Set ℕ, is_partition A B (Finset.range 1 n).to_set →
    (exists_square_sum A ∨ exists_square_sum B)) → n ≥ 15 :=
sorry

end min_nsatisfying_condition_l328_328684


namespace shadowed_area_l328_328524

theorem shadowed_area (ABCD_area : ℝ)
                        (smaller_square_area : ℝ)
                        (larger_square_area : ℝ)
                        (overlap_area : ℝ) :
    ABCD_area = 196 ∧
    larger_square_area = 4 * smaller_square_area ∧
    overlap_area = 1 →
    (3 * sqrt(smaller_square_area))^2 + (3 * sqrt(smaller_square_area)) - 1 ∧
    ((sqrt(smaller_square_area) * 2 - overlap_area) * sqrt(smaller_square_area) - (overlap_area))^2 = 72 :=
begin
  sorry
end

end shadowed_area_l328_328524


namespace sum_f_eq_l328_328676

open Finset

/-- Define the set X_n -/
def X_n (n : ℕ) : Finset ℕ := range (n + 1) \ {0}

/-- Define the smallest element function f -/
def f (A : Finset ℕ) : ℕ := A.min' (by 
  -- Proof that A is non-empty because A is a subset of {1, ..., n} and thus has a minimal element.
  sorry)

theorem sum_f_eq (n : ℕ) : ∑ A in (powerset (X_n n)), f A = 2 ^ (n + 1) - 2 - n := by
  sorry

end sum_f_eq_l328_328676


namespace weight_of_mixture_is_112_5_l328_328486

noncomputable def weight_of_mixture (W : ℝ) : Prop :=
  (5 / 14) * W + (3 / 10) * W + (2 / 9) * W + (1 / 7) * W + 2.5 = W

theorem weight_of_mixture_is_112_5 : ∃ W : ℝ, weight_of_mixture W ∧ W = 112.5 :=
by {
  use 112.5,
  sorry
}

end weight_of_mixture_is_112_5_l328_328486


namespace problem_l328_328669

noncomputable def infinite_series_1 : ℚ :=
  ∑' n, (n^2 : ℚ) / (2^(n / 2 + 2))

noncomputable def infinite_series_2 : ℚ :=
  ∑' n, (n^2 : ℚ) / (3^(n / 2 + 1))

def rel_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem problem (a b : ℕ) (h_rel_prime : rel_prime a b) :
    ((a : ℚ) / (b : ℚ) = infinite_series_1 + infinite_series_2) → a + b = 97 := by
  sorry

end problem_l328_328669


namespace sample_mean_estimate_population_mean_l328_328998

variable (x̄ μ : ℝ)
variable (h_sample_mean : x̄ = x̄) -- represents the sample mean
variable (h_population_mean : μ = μ) -- represents the population mean
variable (h_estimation : ∀ x̄ μ, x̄ = x̄ → μ = μ → ∃ estimate, estimate = x̄)

theorem sample_mean_estimate_population_mean :
  ∃ estimate, estimate = x̄ := 
by
  sorry

end sample_mean_estimate_population_mean_l328_328998


namespace isosceles_triangle_midpoints_inequality_l328_328928

theorem isosceles_triangle_midpoints_inequality
  (A B C M P Q : ℝ) -- We assume everything is in the real number line for simplicity
  (isosceles_triangle : isosceles A B C)
  (midpoint_M : M = (A + C) / 2)
  (midpoint_P : P = (A + M) / 2)
  (Q_on_AB : Q ∈ line_segment A B)
  (ratio_AQ_BQ : AQ = 3 * BQ) :
  BP + MQ > AC :=
sorry

end isosceles_triangle_midpoints_inequality_l328_328928


namespace gravitational_force_at_300000_l328_328738

-- Definitions and premises
def gravitational_force (d : ℝ) : ℝ := sorry

axiom inverse_square_law (d : ℝ) (f : ℝ) (k : ℝ) : f * d^2 = k

axiom surface_force : gravitational_force 5000 = 800

-- Goal: Prove the gravitational force at 300,000 miles
theorem gravitational_force_at_300000 : gravitational_force 300000 = 1 / 45 := sorry

end gravitational_force_at_300000_l328_328738


namespace largest_integer_less_than_log_sum_l328_328463

theorem largest_integer_less_than_log_sum :
  let s := (finset.range 2023).sum (λ k, real.log (k + 2) / real.log 2 - real.log (k + 1) / real.log 2)
  ∃ n : ℤ, n < s ∧ (s < n + 1) := sorry

end largest_integer_less_than_log_sum_l328_328463


namespace number_of_adult_tickets_l328_328053

-- Let's define our conditions and the theorem to prove.
theorem number_of_adult_tickets (A C : ℕ) (h₁ : A + C = 522) (h₂ : 15 * A + 8 * C = 5086) : A = 131 :=
by
  sorry

end number_of_adult_tickets_l328_328053


namespace applicant_overall_score_l328_328807

-- Definitions for the conditions
def writtenTestScore : ℝ := 80
def interviewScore : ℝ := 60
def weightWrittenTest : ℝ := 0.6
def weightInterview : ℝ := 0.4

-- Theorem statement
theorem applicant_overall_score : 
  (writtenTestScore * weightWrittenTest) + (interviewScore * weightInterview) = 72 := 
by
  sorry

end applicant_overall_score_l328_328807


namespace equality_of_x_and_y_l328_328237

theorem equality_of_x_and_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x^(y^x) = y^(x^y)) : x = y :=
sorry

end equality_of_x_and_y_l328_328237


namespace sin_alpha_value_l328_328234

noncomputable def α : ℝ := sorry  -- α should be inferred later in the proof

theorem sin_alpha_value
  (h1 : 0 < α ∧ α < π / 2) -- α is an acute angle
  (h2 : cos (α + π / 4) = 3 / 5) : sin α = sqrt 2 / 10 :=
sorry

end sin_alpha_value_l328_328234


namespace tile_large_rectangles_l328_328533

theorem tile_large_rectangles (m n : ℕ) (hm : m > 840) (hn : n > 840) : 
  ∃ (N : ℕ), ∀ (m n : ℕ), (m > N) → (n > N) → (tiled_with_4x6_and_5x7 m n) :=
  sorry

-- Definition of tileability using 4x6 and 5x7 tiles
def tiled_with_4x6_and_5x7 (m n : ℕ) : Prop :=
  ∃ (k l : ℕ), (4 * k + 5 * l = m) ∧ (6 * k + 7 * l = n)

end tile_large_rectangles_l328_328533


namespace number_of_passed_candidates_l328_328030

variables (P F : ℕ) (h1 : P + F = 100)
          (h2 : P * 70 + F * 20 = 100 * 50)
          (h3 : ∀ p, p = P → 70 * p = 70 * P)
          (h4 : ∀ f, f = F → 20 * f = 20 * F)

theorem number_of_passed_candidates (P F : ℕ) (h1 : P + F = 100) 
                                    (h2 : P * 70 + F * 20 = 100 * 50) 
                                    (h3 : ∀ p, p = P → 70 * p = 70 * P) 
                                    (h4 : ∀ f, f = F → 20 * f = 20 * F) : 
  P = 60 :=
sorry

end number_of_passed_candidates_l328_328030


namespace geometric_sum_ratio_l328_328352

-- Definitions and Conditions
variables {a : ℕ → ℕ}
variable q : ℕ
variable n : ℕ

-- Condition a₅ - a₃ = 12
axiom h1 : a 5 - a 3 = 12

-- Condition a₆ - a₄ = 24
axiom h2 : a 6 - a 4 = 24

-- The goal to prove: Sₙ / aₙ = 2 - 2⁽¹⁻ⁿ⁾, where Sₙ is the sum of first n terms and aₙ is the nth term
theorem geometric_sum_ratio (S : ℕ → ℕ) (a_n : ℕ) : 
  (S n) / (a n) = 2 - 2^(1 - n) := 
sorry

end geometric_sum_ratio_l328_328352


namespace smallest_integer_four_consecutive_product_1680_l328_328566

theorem smallest_integer_four_consecutive_product_1680 : ∃ n : ℕ, n * (n+1) * (n+2) * (n+3) = 1680 ∧ n = 5 :=
by
  have integral_value : 1680 = 2^4 * 3 * 5 * 7 := by norm_num
  have eq_five : nat.sqrt (1680 / (2^4 * 3)) = 5 := by norm_num
  use 5
  split
  · norm_num
  sorry

end smallest_integer_four_consecutive_product_1680_l328_328566


namespace more_than_10_weights_missing_l328_328428

/-- 
Given weights of 5, 24, and 43 grams with an equal number of each type
and that the total remaining mass is 606060...60 grams,
prove that more than 10 weights are missing.
-/
theorem more_than_10_weights_missing (total_mass : ℕ) (n : ℕ) (k : ℕ) 
  (total_mass_eq : total_mass = k * (5 + 24 + 43))
  (total_mass_mod : total_mass % 72 ≠ 0) :
  k < n - 10 :=
sorry

end more_than_10_weights_missing_l328_328428


namespace number_of_ways_to_assign_roles_l328_328815

theorem number_of_ways_to_assign_roles : 
  let male_roles := 3
  let female_roles := 2
  let either_gender_roles := 1
  let men := 4
  let women := 5
  let total_roles := male_roles + female_roles + either_gender_roles
  let ways_to_assign_males := men * (men-1) * (men-2)
  let ways_to_assign_females := women * (women-1)
  let remaining_actors := men + women - male_roles - female_roles
  let ways_to_assign_either_gender := remaining_actors
  let total_ways := ways_to_assign_males * ways_to_assign_females * ways_to_assign_either_gender

  total_ways = 1920 :=
by
  sorry

end number_of_ways_to_assign_roles_l328_328815


namespace total_bowling_balls_is_66_l328_328038

-- Define the given conditions
def red_bowling_balls := 30
def difference_green_red := 6
def green_bowling_balls := red_bowling_balls + difference_green_red

-- The statement to prove
theorem total_bowling_balls_is_66 :
  red_bowling_balls + green_bowling_balls = 66 := by
  sorry

end total_bowling_balls_is_66_l328_328038


namespace parallel_lines_range_of_distances_l328_328758

-- Define the points P and Q
def P : ℝ × ℝ := (-1, 2)
def Q : ℝ × ℝ := (2, -3)

-- Define a function to calculate the distance between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- The range of distances function to consider the maximum distance PQ when perpendicular
def range_of_distances (p q : ℝ × ℝ) : set ℝ :=
  {x : ℝ | 0 < x ∧ x ≤ distance p q}

-- The proof statement to be verified
theorem parallel_lines_range_of_distances :
  range_of_distances P Q = {x : ℝ | 0 < x ∧ x ≤ Real.sqrt 34} :=
by
  sorry

end parallel_lines_range_of_distances_l328_328758


namespace collinear_incenter_circumcenter_intersect_point_l328_328447

-- Definitions of the geometric entities and their properties
noncomputable def Triangle : Type := sorry
noncomputable def Point : Type := sorry
noncomputable def Circle (center : Point) (radius : ℝ) : Type := sorry
noncomputable def incenter (T : Triangle) : Point := sorry
noncomputable def circumcenter (T : Triangle) : Point := sorry
noncomputable def tangent_to_two_sides (c : Circle) (T : Triangle) : Prop := sorry
noncomputable def inside (c : Circle) (T : Triangle) : Prop := sorry
noncomputable def intersect (c1 c2 : Circle) (P : Point) : Prop := sorry

axiom K_is_intersection_point (c1 c2 c3 : Circle) : 
  ∀ (P : Point), intersect c1 c2 P ∧ intersect c2 c3 P ∧ intersect c3 c1 P → P = K

-- Given conditions on the problem
variables (T : Triangle) 
          (O O1 O2 O3 I K : Point) 
          (c1 : Circle O1 5) 
          (c2 : Circle O2 5) 
          (c3 : Circle O3 5)

axiom congruent_circles : ∀ {O1 O2 O3 : Point}, c1 = Circle O1 5 ∧ c2 = Circle O2 5 ∧ c3 = Circle O3 5
axiom circles_position : ∀ {T : Triangle} (c : Circle), tangent_to_two_sides c T ∧ inside c T
axiom incenter_def : I = incenter T
axiom circumcenter_def : O = circumcenter T
axiom K_intersect : intersect c1 c2 K ∧ intersect c2 c3 K ∧ intersect c3 c1 K

-- The statement to prove
theorem collinear_incenter_circumcenter_intersect_point : 
  I = incenter T → O = circumcenter T → 
  (∀ {O1 O2 O3 : Point}, c1 = Circle O1 5 ∧ c2 = Circle O2 5 ∧ c3 = Circle O3 5) → 
  (∀ {T : Triangle} {c1 c2 c3 : Circle}, tangent_to_two_sides c1 T ∧ inside c1 T ∧ 
                                         tangent_to_two_sides c2 T ∧ inside c2 T ∧ 
                                         tangent_to_two_sides c3 T ∧ inside c3 T) → 
  (intersect c1 c2 K ∧ intersect c2 c3 K ∧ intersect c3 c1 K) → 
  collinear {I, O, K} :=
begin
  -- proof to be filled
  sorry
end

end collinear_incenter_circumcenter_intersect_point_l328_328447


namespace vector_dot_product_l328_328267

variable {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem vector_dot_product (h1 : a + 2 • b = 0) 
    (h2 : (a + b) ⬝ a = 2) : 
    a ⬝ b = -2 :=
by
  sorry

end vector_dot_product_l328_328267


namespace phillip_remaining_amount_l328_328383

-- Define the initial amount of money
def initial_amount : ℕ := 95

-- Define the amounts spent on various items
def amount_spent_on_oranges : ℕ := 14
def amount_spent_on_apples : ℕ := 25
def amount_spent_on_candy : ℕ := 6

-- Calculate the total amount spent
def total_spent : ℕ := amount_spent_on_oranges + amount_spent_on_apples + amount_spent_on_candy

-- Calculate the remaining amount of money
def remaining_amount : ℕ := initial_amount - total_spent

-- Statement to be proved
theorem phillip_remaining_amount : remaining_amount = 50 :=
by
  sorry

end phillip_remaining_amount_l328_328383


namespace Problem_Accessible_Functions_D_l328_328225

def accessible_functions (f g : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), x > 0 → |f x - g x| < 1

theorem Problem_Accessible_Functions_D :
  accessible_functions (λ x, x + 2/x) (λ x, log x + 2) :=
by
  sorry

end Problem_Accessible_Functions_D_l328_328225


namespace hotel_r_greater_than_g_l328_328476

variables (P R G : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := P = 0.30 * R
def condition2 : Prop := P = 0.90 * G

-- Define the mathematical statement to prove
theorem hotel_r_greater_than_g 
  (h1 : condition1)
  (h2 : condition2) :
  ((R - G) / G) = 1.70 :=
sorry

end hotel_r_greater_than_g_l328_328476


namespace symmetric_line_equation_l328_328415

theorem symmetric_line_equation 
  (L : ℝ → ℝ → Prop)
  (H : ∀ x y, L x y ↔ x - 2 * y + 1 = 0) : 
  ∃ L' : ℝ → ℝ → Prop, 
    (∀ x y, L' x y ↔ x + 2 * y - 3 = 0) ∧ 
    ( ∀ x y, L (2 - x) y ↔ L' x y ) := 
sorry

end symmetric_line_equation_l328_328415


namespace sum_of_y_coordinates_l328_328488

noncomputable def circle_y_coordinate_sum : ℝ :=
  let d := (|3 - y| / real.sqrt 2)
  let r := real.sqrt (1 + y^2)
  (∀ y, circle_passes_through (2,0) (4,0) ∧ tangent_to_line_y_eq_x ((3, y)) → (r = d)) → 
  (sum (s : set ℝ) := -6)

-- Circle passes through point (a, b) and (c, d)
def circle_passes_through (point1 point2 : ℝ × ℝ) : Prop := sorry 

-- Line y = x is tangent to the circle at point (3, y)
def tangent_to_line_y_eq_x (center : ℝ × ℝ) : Prop := sorry

theorem sum_of_y_coordinates : circle_y_coordinate_sum = -6 := sorry

end sum_of_y_coordinates_l328_328488


namespace restaurant_pizzas_more_than_hotdogs_l328_328506

theorem restaurant_pizzas_more_than_hotdogs
  (H P : ℕ) 
  (h1 : H = 60)
  (h2 : 30 * (P + H) = 4800) :
  P - H = 40 :=
by
  sorry

end restaurant_pizzas_more_than_hotdogs_l328_328506


namespace find_sum_of_products_of_roots_l328_328672

-- Constants representing the roots of the cubic equation
variables {p q r : ℝ}

-- The cubic equation 6x^3 - 4x^2 + 15x - 10 = 0 has roots p, q, r
def cubic_eq_roots : Prop := 
  ∀ (x : ℝ), 6 * x^3 - 4 * x^2 + 15 * x - 10 = 0 → (x = p ∨ x = q ∨ x = r)

-- Using Vieta's formulas, we express the sum of the products of the roots taken two at a time
def viexperience_formula_two_roots : Prop :=
  pq + qr + rp = (15 / 6)

-- The main statement we want to prove
theorem find_sum_of_products_of_roots (h : cubic_eq_roots) : pq + qr + rp = 5 / 2 := 
by 
  -- Applying Vieta's formula for the sum of the products of the roots taken two at a time 
  sorry

end find_sum_of_products_of_roots_l328_328672


namespace squirrel_spiral_distance_l328_328776

/-- The squirrel runs up a cylindrical post in a perfect spiral path, making one circuit for each rise of 4 feet.
Given the post is 16 feet tall and 3 feet in circumference, the total distance traveled by the squirrel is 20 feet. -/
theorem squirrel_spiral_distance :
  let height : ℝ := 16
  let circumference : ℝ := 3
  let rise_per_circuit : ℝ := 4
  let number_of_circuits := height / rise_per_circuit
  let distance_per_circuit := (circumference^2 + rise_per_circuit^2).sqrt
  number_of_circuits * distance_per_circuit = 20 := by
  sorry

end squirrel_spiral_distance_l328_328776


namespace problem_statement_l328_328341

variable {α : Type*} [field α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0     := a₁
| (n+1) := geometric_sequence a₁ q n * q

def Sn (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * n else a₁ * (1 - q ^ n) / (1 - q)

theorem problem_statement (a₁ q : α) (n : ℕ) (h₀ : q ≠ 1)
  (h₁ : geometric_sequence a₁ q 4 - geometric_sequence a₁ q 2 = 12)
  (h₂ : geometric_sequence a₁ q 5 - geometric_sequence a₁ q 3 = 24) :
  Sn a₁ q n / geometric_sequence a₁ q (n - 1) = 2 - 2 ^ (1 - n) :=
sorry

end problem_statement_l328_328341


namespace derivative_of_sine_exp_cos_l328_328410

theorem derivative_of_sine_exp_cos (x : ℝ) :
  (deriv (λ x : ℝ, sin x + exp x * cos x)) x = (1 + exp x) * cos x - exp x * sin x :=
by
  sorry

end derivative_of_sine_exp_cos_l328_328410


namespace water_remaining_after_pourings_l328_328490

theorem water_remaining_after_pourings :
  ∀ n, (1 * (3 / 4 * 4 / 5 * 5 / 6 * ⋯ * (n + 2) / (n + 3))) = 1 / 10 ↔ n = 27 := sorry

end water_remaining_after_pourings_l328_328490


namespace true_statements_l328_328228

def line := Type
def plane := Type

variables (a b c : line) (γ : plane)

theorem true_statements (h1 : a ∥ b ∧ b ∥ c → a ∥ c)
                        (h2 : a ⊥ b ∧ b ⊥ c → a ⊥ c)
                        (h3 : a ∥ γ ∧ b ∥ γ → a ⊥ b)
                        (h4 : a ⊥ γ ∧ b ⊥ γ → a ∥ b) :
                        {1, 4} = {i | (i = 1 ∧ (a ∥ b ∧ b ∥ c → a ∥ c)) ∨
                                   (i = 2 ∧ (a ⊥ b ∧ b ⊥ c → a ⊥ c)) ∨
                                   (i = 3 ∧ (a ∥ γ ∧ b ∥ γ → a ⊥ b)) ∨
                                   (i = 4 ∧ (a ⊥ γ ∧ b ⊥ γ → a ∥ b))} :=
begin
  sorry
end

end true_statements_l328_328228


namespace Alchemerion_is_3_times_older_than_his_son_l328_328840

-- Definitions of Alchemerion's age, his father's age and the sum condition
def Alchemerion_age : ℕ := 360
def Father_age (A : ℕ) := 2 * A + 40
def age_sum (A S F : ℕ) := A + S + F

-- Main theorem statement
theorem Alchemerion_is_3_times_older_than_his_son (S : ℕ) (h1 : Alchemerion_age = 360)
    (h2 : Father_age Alchemerion_age = 2 * Alchemerion_age + 40)
    (h3 : age_sum Alchemerion_age S (Father_age Alchemerion_age) = 1240) :
    Alchemerion_age / S = 3 :=
sorry

end Alchemerion_is_3_times_older_than_his_son_l328_328840


namespace ticket_distribution_l328_328190

-- Defining the problem conditions
def distribution_condition (distribution : list (list ℕ)) : Prop :=
  (∀ tickets, tickets.length = 1 ∨ tickets.length = 2) ∧
  (∀ tickets, tickets.length = 2 → tickets.head! + 1 = tickets.tail!.head!)

-- Defining a valid distribution
def valid_distribution (distr : list (list ℕ)) : Prop :=
  distr.length = 4 ∧ distribution_condition distr ∧ (∑ x in distr, x.length) = 5

-- Main theorem to prove
theorem ticket_distribution : ∃ distr : list (list ℕ), valid_distribution distr ∧ 
  (list.permutations [1,2,3,4,5]).length = 96 :=
sorry

end ticket_distribution_l328_328190


namespace length_CDtoEqual_2xMedianAM_l328_328318

-- Defining the pentagon and its properties
structure Pentagon (A B C D E : Type) :=
  (convex : Prop)
  (AE_eq_AD : A → E → D → Prop)
  (AB_eq_AC : A → B → C → Prop)
  (angle_CAD_eq_angle_AEB_plus_angle_ABE : (A → C → D → Prop) × (A → E → B → Prop) → (A → B → E → Prop) → Prop)

-- Defining point translations and lengths
variable {A B C D E M : Type}

-- Define the median AM
def Median_AM (A B E M : Type) := (A → B → E → M → Prop)

-- Define the length of segments and relationship
def Length (A B : Type) := ℝ

-- Define provided conditions
variables (h1 : A → E → D → Prop) (h2 : A → B → C → Prop) (h3 : (A → C → D → Prop) × (A → E → B → Prop) → (A → B → E → Prop) → Prop)

-- Define the main proof statement that needs to be proved
theorem length_CDtoEqual_2xMedianAM :
  ∀ {A B E M : Type} [h1 : Pentagon.convex A B C D E] [Pentagon.AE_eq_AD A E D]
  [Pentagon.AB_eq_AC A B C] [Pentagon.angle_CAD_eq_angle_AEB_plus_angle_ABE (A → C → D) (A → E → B) (A → B → E)]
  (AM : Median_AM A B E M) (CD : Length C D) (AM_length : Length A M),
  CD = 2 * AM_length :=
by sorry

end length_CDtoEqual_2xMedianAM_l328_328318


namespace num_primes_no_nonprime_subnumbers_l328_328028

/-- 
There are exactly 9 primes less than 1,000,000,000 that have no non-prime subnumbers, 
where subnumbers are defined as any contiguous subsequence of the digits of the number, 
and valid primes must start with one of the digits 2, 3, 5, or 7.
-/
theorem num_primes_no_nonprime_subnumbers :
  (∃ (primes : Finset ℕ), 
    (∀ p ∈ primes, 
       p < 1000000000 ∧ 
       Nat.Prime p ∧
       (∀ (subnumber : ℕ), subnumber ∈ subnumbers p → Nat.Prime subnumber)) ∧ 
    primes.card = 9) :=
sorry

/-- Generates the set of subnumbers from a given number -/
def subnumbers (n : ℕ) : Finset ℕ := 
  -- Placeholder for the actual implementation
  sorry

end num_primes_no_nonprime_subnumbers_l328_328028


namespace range_of_a_l328_328996

noncomputable def f (a x : ℝ) : ℝ := 
  (4 - a) * ((x^2 - 2*x - 2) * Real.exp x - a * x^3 + 12 * a * x)

theorem range_of_a (a : ℝ) (h : ∃ x, (x = 2) ∧ (4 - a) * ((x^2 - 2*x - 2) * Real.exp x - a * x^3 + 12 * a * x) = 0 ∧ ((differentiable ℝ (λ x, f a x)) ∧ (critical_point (f a) 2)) : a ∈ Ioo 4 (1/3 * Real.exp 3) :=
sorry

end range_of_a_l328_328996


namespace correct_conclusions_l328_328971

def vector_a : ℝ × ℝ × ℝ := (4, -2, -4)
def vector_b : ℝ × ℝ × ℝ := (6, -3, 2)
def vector_sum : ℝ × ℝ × ℝ := (10, -5, -2)
def vector_a_magnitude : ℝ := 6

theorem correct_conclusions :
  (vector_a.1 + vector_b.1 = vector_sum.1 ∧ vector_a.2 + vector_b.2 = vector_sum.2 ∧ vector_a.3 + vector_b.3 = vector_sum.3) ∧
  (Real.sqrt ((vector_a.1)^2 + (vector_a.2)^2 + (vector_a.3)^2) = vector_a_magnitude) :=
by
  sorry

end correct_conclusions_l328_328971


namespace algebra_expression_value_l328_328281

theorem algebra_expression_value (a : ℝ) (h : 3 * a ^ 2 + 2 * a - 1 = 0) : 3 * a ^ 2 + 2 * a - 2019 = -2018 := 
by 
  -- Proof goes here
  sorry

end algebra_expression_value_l328_328281


namespace domain_function_1_domain_function_2_domain_function_3_l328_328883

-- Define the conditions and the required domain equivalence in Lean 4
-- Problem (1)
theorem domain_function_1 (x : ℝ): x + 2 ≠ 0 ∧ x + 5 ≥ 0 ↔ x ≥ -5 ∧ x ≠ -2 := 
sorry

-- Problem (2)
theorem domain_function_2 (x : ℝ): x^2 - 4 ≥ 0 ∧ 4 - x^2 ≥ 0 ∧ x^2 - 9 ≠ 0 ↔ (x = 2 ∨ x = -2) :=
sorry

-- Problem (3)
theorem domain_function_3 (x : ℝ): x - 5 ≥ 0 ∧ |x| ≠ 7 ↔ x ≥ 5 ∧ x ≠ 7 :=
sorry

end domain_function_1_domain_function_2_domain_function_3_l328_328883


namespace geometric_sum_ratio_l328_328351

-- Definitions and Conditions
variables {a : ℕ → ℕ}
variable q : ℕ
variable n : ℕ

-- Condition a₅ - a₃ = 12
axiom h1 : a 5 - a 3 = 12

-- Condition a₆ - a₄ = 24
axiom h2 : a 6 - a 4 = 24

-- The goal to prove: Sₙ / aₙ = 2 - 2⁽¹⁻ⁿ⁾, where Sₙ is the sum of first n terms and aₙ is the nth term
theorem geometric_sum_ratio (S : ℕ → ℕ) (a_n : ℕ) : 
  (S n) / (a n) = 2 - 2^(1 - n) := 
sorry

end geometric_sum_ratio_l328_328351


namespace log_sequence_geometric_a_geometric_sequence_sum_b_sequence_l328_328961

noncomputable def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℝ := let an := a n in an / ((an + 1) * (a (n+1) + 1))

theorem log_sequence_geometric (n : ℕ) : log 2 (a (n + 1)) = log 2 (a n) + log 2 2 := 
by sorry

theorem a_geometric_sequence : a 2 * a 6 = 64 := 
by sorry

theorem sum_b_sequence (n : ℕ) : (finset.range n).sum b = (1 / 2) - 1 / (2^n + 1) := 
by sorry

end log_sequence_geometric_a_geometric_sequence_sum_b_sequence_l328_328961


namespace smallest_k_l328_328180

noncomputable def sequence_a : ℕ → ℝ
| 0     := 1
| 1     := real.root 17 3
| (n+2) := sequence_a (n+1) * (sequence_a n)^3

def is_integer_product : ℕ → Prop
| 0     := false
| k     := (real.logb 3 (sequence_a 1 * sequence_a 2 * .... * sequence_a k)) % 17 = 0

theorem smallest_k (k : ℕ) (h : k = 14) : is_integer_product k :=
by sorry

end smallest_k_l328_328180


namespace gcd_power_diff_l328_328850

theorem gcd_power_diff (n m : ℕ) (h₁ : n = 2025) (h₂ : m = 2007) :
  (Nat.gcd (2^n - 1) (2^m - 1)) = 2^18 - 1 :=
by
  sorry

end gcd_power_diff_l328_328850


namespace complex_exponential_to_rectangular_form_l328_328542

theorem complex_exponential_to_rectangular_form :
  Real.sqrt 2 * Complex.exp (13 * Real.pi * Complex.I / 4) = -1 - Complex.I := by
  -- Proof will go here
  sorry

end complex_exponential_to_rectangular_form_l328_328542


namespace game_winner_l328_328377

-- Definition of the conditions as given in the problem statement
def Monica_wins (n k : ℕ) : Prop := 
  ∀ (m : ℕ), ∃ (points : list (ℝ × ℝ)), points.length = n ∧
  ∀ i, ∃ j, dist (points.nth_le i sorry) (points.nth_le j sorry) = m 

-- Definition encapsulating the winning strategy
def Monica_has_winning_strategy (n k : ℕ) : Prop := n ≤ k

-- Main theorem stating the winning strategy equivalence
theorem game_winner (n k : ℕ) : Monica_has_winning_strategy n k ↔ Monica_wins n k := 
sorry

end game_winner_l328_328377


namespace Vasya_prevents_all_cords_burning_l328_328441

-- Definitions and assumptions based directly on the problem's given conditions.
variable (cords : ℕ) (matches : ℕ)
variables (arranges_in_square : cords = 40) (num_matches : matches = 12)

-- Statement to be proved:
theorem Vasya_prevents_all_cords_burning (cords = 40) (matches = 12): 
  ∃ arrangement, ¬ (∀ (burn : ℕ → Prop), (burn 40 → ∃ (fuse_placement : ℕ → ℕ), @burn cords fuse_placement 40)) :=
sorry

end Vasya_prevents_all_cords_burning_l328_328441


namespace number_of_factors_of_N_l328_328860

theorem number_of_factors_of_N : 
  let N := (2^4) * (3^3) * (5^2) * (7^1) in
  ∃ (count : ℕ), count = 120 ∧ (∀ d, d ∣ N → isNat d → (d ∈ (finset.range (N + 1).filter (λ n, n ∣ N)).card = count)) :=
by
  sorry

end number_of_factors_of_N_l328_328860


namespace min_value_of_fraction_l328_328960

theorem min_value_of_fraction (a b c : ℝ) (h_deriv : 2 * a > 0)
  (h_nonneg : ∀ x : ℝ, a * x * x + b * x + c ≥ 0) :
  (1 + (a + c) / b) ≥ 2 :=
by
  let f := λ x : ℝ, a * x ^ 2 + b * x + c
  let f_deriv := λ x : ℝ, 2 * a * x + b
  let f_sec_deriv := 2 * a
  have h_ac : 4 * a * c ≥ b ^ 2 := sorry
  have h_frac : (f 1) / f_sec_deriv ≥ 2 := sorry
  exact h_frac

end min_value_of_fraction_l328_328960


namespace stock_percent_change_l328_328391

theorem stock_percent_change (y : ℝ) : 
  let value_after_day1 := 0.85 * y
  let value_after_day2 := 1.25 * value_after_day1
  (value_after_day2 - y) / y * 100 = 6.25 := by
  sorry

end stock_percent_change_l328_328391


namespace total_questions_reviewed_l328_328831

-- Defining the conditions
def questions_per_student : Nat := 10
def students_per_class : Nat := 35
def num_classes : Nat := 5

-- Define the total number of questions that Professor Oscar must review.
def total_questions : Nat := questions_per_student * students_per_class * num_classes

-- Theorem statement to be proved
theorem total_questions_reviewed : total_questions = 1750 := by
  sorry

end total_questions_reviewed_l328_328831


namespace sandy_initial_carrots_l328_328011

-- Defining the conditions
def sam_took : ℕ := 3
def sandy_left : ℕ := 3

-- The statement to be proven
theorem sandy_initial_carrots :
  (sandy_left + sam_took = 6) :=
by
  sorry

end sandy_initial_carrots_l328_328011


namespace derivative_of_f_tangent_line_at_point_l328_328954

noncomputable def f (k : ℝ) (x : ℝ) := k * (x - 1) * Real.exp(x) + x^2

theorem derivative_of_f (k : ℝ) : 
  (Real.deriv (λ x => f k x)) = (λ x => k * x * Real.exp(x) + 2 * x) :=
by
  sorry

theorem tangent_line_at_point (k : ℝ) (h : k = -1 / Real.exp(1)) (y : ℝ) :
  ∃ m b, (∀ x, y = m * x + b) ∧ m = 1 ∧ b = 0 :=
by
  sorry

end derivative_of_f_tangent_line_at_point_l328_328954


namespace part_a_part_b_l328_328174

-- Definitions based on conditions
variables {ABC : Type} [non_isosceles_triangle ABC] [acute_angled_triangle ABC]
variables (O : point_circumcenter ABC) (M : point_orthocenter ABC)
variables (A B C D F G K E : point)
variables (midpoint_AB : F = midpoint A B)
variables (foot_altitude_A : D = foot_of_altitude A)
variables (ray_from_F_M : ray F M G)

-- Prove that the points A, F, D, and G are concyclic
theorem part_a : concyclic_points [A, F, D, G] :=
sorry

-- Additional conditions to define K and E
variables (circle_center_K : K = circle_center [A, F, D, G])
variables (midpoint_CM : E = midpoint C M)

-- Prove that EK = OK
theorem part_b (K_center : K = circle_center [A, F, D, G]) : distance E K = distance O K :=
sorry

end part_a_part_b_l328_328174


namespace original_number_fraction_l328_328694

theorem original_number_fraction (x : ℚ) (h : 1 + 1/x = 9/4) : x = 4/5 := by
  sorry

end original_number_fraction_l328_328694


namespace smallest_four_digit_palindrome_divisible_by_seven_is_2112_l328_328465

def is_palindrome (n : ℕ) : Prop :=
  let n_str := n.toString
  n_str = n_str.reverse

noncomputable def smallest_palindrome_divisible_by_seven : ℕ :=
  let candidates := [n | n <- list.range 10000, is_palindrome n, n ≥ 1000]
  list.find (λ n => n % 7 = 0) candidates

theorem smallest_four_digit_palindrome_divisible_by_seven_is_2112 :
  smallest_palindrome_divisible_by_seven = 2112 := by
    sorry

end smallest_four_digit_palindrome_divisible_by_seven_is_2112_l328_328465


namespace frank_floor_sixteen_l328_328181

variable (F D C : ℕ)

-- Conditions
def dennis_floor := 6
def dennis_above_charlie := D = C + 2
def charlie_quarter_frank := C = F / 4

-- Question re-phrased: Prove that Frank lives on the 16th floor.
theorem frank_floor_sixteen :
  dennis_floor = 6 → 
  dennis_above_charlie →
  charlie_quarter_frank →
  F = 16 :=
by
  intros
  sorry

end frank_floor_sixteen_l328_328181


namespace num_correct_statements_l328_328735

def Q := {x : ℚ | true}
def R := {x : ℝ | true}
def N₊ := {x : ℕ | x > 0}
def Z := {x : ℤ | true}
def empty_set := {x : Set ℕ | false}

lemma Q_contains_one_half : (1 / 2 : ℚ) ∈ Q := by trivial

lemma R_contains_sqrt_two : (real.sqrt 2) ∈ R := by exact real.sqrt_nonneg 2

lemma N₊_contains_zero : ¬ (0 : ℕ) ∈ N₊ := by simp [N₊]

lemma Z_contains_pi : ¬ (real.pi : ℤ) ∈ Z := by simp [Z]

lemma empty_set_contains_empty : ¬ (Set.empty ℕ) ∈ ({0} : Set ℕ) := by simp

theorem num_correct_statements : 1 = 1 :=
by
  have h1 : (1 / 2 : ℚ) ∈ Q := Q_contains_one_half
  have h2 : (real.sqrt 2) ∈ R := R_contains_sqrt_two
  have h3 : ¬ (0 : ℕ) ∈ N₊ := N₊_contains_zero
  have h4 : ¬ (real.pi : ℤ) ∈ Z := Z_contains_pi
  have h5 : ¬ (Set.empty ℕ) ∈ ({0} : Set ℕ) := empty_set_contains_empty
  sorry

end num_correct_statements_l328_328735


namespace overhead_cost_reduction_is_five_percent_l328_328041

variable (x : ℝ) -- Initial factor for costs

-- Initial costs
def initial_raw_material_cost := 4 * x
def initial_labor_cost := 3 * x
def initial_overhead_cost := 2 * x
def total_initial_cost := initial_raw_material_cost + initial_labor_cost + initial_overhead_cost

-- Next year's costs
def next_year_raw_material_cost := initial_raw_material_cost * 1.10
def next_year_labor_cost := initial_labor_cost * 1.08

-- Increase in total cost
def total_next_year_cost := total_initial_cost * 1.06

-- Overhead cost in next year
def next_year_overhead_cost := total_next_year_cost - next_year_raw_material_cost - next_year_labor_cost

-- Percentage reduction calculation
def reduction := (initial_overhead_cost - next_year_overhead_cost) / initial_overhead_cost * 100

-- Theorem to prove the percentage reduction in overhead cost is 5%.
theorem overhead_cost_reduction_is_five_percent : reduction x = 5 := sorry

end overhead_cost_reduction_is_five_percent_l328_328041


namespace cannot_form_last_larger_figure_l328_328697

-- Definitions of conditions
def is_rhombus_divided_into_two_triangles (shape : Type) :=
  shape = "rhombus" ∧ shape.contains_white_and_gray_triangl_angles

def can_be_rotated_not_flipped (shape : Type) := 
  shape.rotation ∈ {0, 90, 180, 270} ∧ ¬shape.flipped

-- Main problem statement
theorem cannot_form_last_larger_figure (shape : Type) (larger_figure : Type) :
  is_rhombus_divided_into_two_triangles shape →
  can_be_rotated_not_flipped shape →
  larger_figure ≠ "last_larger_figure" :=
begin
  intros h_shape h_rotation,
  sorry
end

end cannot_form_last_larger_figure_l328_328697


namespace total_bowling_balls_l328_328034

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l328_328034


namespace hotel_loss_l328_328132

variable (operations_expenses : ℝ)
variable (fraction_payment : ℝ)

theorem hotel_loss :
  operations_expenses = 100 →
  fraction_payment = 3 / 4 →
  let total_payment := fraction_payment * operations_expenses in
  let loss := operations_expenses - total_payment in
  loss = 25 :=
by
  intros h₁ h₂
  have tstp : total_payment = 75 := by
    rw [h₁, h₂]
    norm_num
  have lss : loss = 25 := by
    rw [h₁, tstp]
    norm_num
  exact lss

end hotel_loss_l328_328132


namespace total_bowling_balls_l328_328032

theorem total_bowling_balls (red_balls : ℕ) (green_balls : ℕ) (h1 : red_balls = 30) (h2 : green_balls = red_balls + 6) : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l328_328032


namespace OP_value_l328_328199

-- Define the points O, A, B, C, D and the distances
def O : ℝ := 0
def A : ℝ := 1
def B : ℝ := 3
def C : ℝ := 5
def D : ℝ := 7

-- Define the location of P
axiom P : ℝ
axiom P_condition : B ≤ P ∧ P ≤ C ∧ abs (A - P) / abs (P - D) = 2 * abs (B - P) / abs (P - C)

-- State the theorem
theorem OP_value : O + P = 1 + 4 * (real.sqrt 3) :=
by
  sorry

end OP_value_l328_328199


namespace t_plus_inv_t_eq_three_l328_328910

theorem t_plus_inv_t_eq_three {t : ℝ} (h : t^2 - 3 * t + 1 = 0) (hnz : t ≠ 0) : t + 1 / t = 3 :=
sorry

end t_plus_inv_t_eq_three_l328_328910


namespace ivan_returns_alive_ivan_takes_princesses_l328_328337

-- Definitions of the scenario
def five_girls : Type := Fin 5 -- Representation of the five girls

def is_tsar_daughter (g : five_girls) : Prop :=
  ∃ (n : ℕ), n < 3 ∧ g = ⟨n, by linarith⟩  -- Indicating three Tsar’s daughters

def is_koschei_daughter (g : five_girls) : Prop :=
  ∃ (k : ℕ), k ≥ 3 ∧ k < 5 ∧ g = ⟨k, by linarith⟩  -- Indicating two Koschei’s daughters

-- Function simulating the recognition pattern, output the names as princesses by girls
def names_princesses (girl : five_girls) : set five_girls := sorry -- To be defined according to the note passed by Ivan

-- Encoding Ivan's guess criteria
def ivan_guess_tsar_daughters_correct (guess : set five_girls) : Prop :=
  guess = {g | ∃ n, n < 3 ∧ g = ⟨n, by linarith⟩}

-- Logical statements for Part a) and Part b)
theorem ivan_returns_alive : ∃ guess, ivan_guess_tsar_daughters_correct guess := sorry

theorem ivan_takes_princesses : ∃ guess (eldest middle youngest : five_girls), 
  is_tsar_daughter eldest ∧ is_tsar_daughter middle ∧ is_tsar_daughter youngest ∧
  eldest ≠ middle ∧ middle ≠ youngest ∧ eldest ≠ youngest ∧
  ∀ g, g ∈ guess → (g = eldest ∨ g = middle ∨ g = youngest) := sorry

end ivan_returns_alive_ivan_takes_princesses_l328_328337


namespace smallest_k_no_real_roots_l328_328763

theorem smallest_k_no_real_roots :
  let eqn := (λ x : ℝ, (List.prod (List.map (λ n, x - n) (List.iota 2016))) = (List.prod (List.map (λ n, x - n) (List.iota 2016))));
  ∃ k ≥ 1, (k = 2016) ∧ ∀ f g : ℝ → ℝ,
  (f = (λ x, List.prod (List.map (λ n, x - (2 * n + 1)) (List.range 1008))) ∧
   g = (λ x, List.prod (List.map (λ n, x - (2 * n + 2)) (List.range 1008))) ∧
   ∀ x : ℝ, f x ≠ g x →  f x > g x) :=
by
  sorry

end smallest_k_no_real_roots_l328_328763


namespace problem_part1_problem_part2_l328_328480

def U : Set ℕ := {x | 0 < x ∧ x < 9}

def S : Set ℕ := {1, 3, 5}

def T : Set ℕ := {3, 6}

theorem problem_part1 : S ∩ T = {3} := by
  sorry

theorem problem_part2 : U \ (S ∪ T) = {2, 4, 7, 8} := by
  sorry

end problem_part1_problem_part2_l328_328480


namespace sum_of_digits_10_to_2008_minus_2008_l328_328466

theorem sum_of_digits_10_to_2008_minus_2008 : 
  sum_of_digits (10 ^ 2008 - 2008) = 18063 := by sorry

end sum_of_digits_10_to_2008_minus_2008_l328_328466


namespace find_x_l328_328649

def angles_triangle (a b c : ℝ) := a + b + c = 180

variables (A B C D X Y Z : Type) 
variables (angle_AXB angle_AXC angle_BXD angle_CYD : ℝ)

theorem find_x 
  (h1 : angle_AXB = 180)
  (h2 : angle_AXC = 80)
  (h3 : angle_BXD = 40)
  (h4 : angle_CYD = 110):
  let x : ℝ := 50 in
  x = 50 :=
by
  sorry

end find_x_l328_328649


namespace adam_walks_distance_l328_328733

/-- The side length of the smallest squares is 20 cm. --/
def smallest_square_side : ℕ := 20

/-- The side length of the middle-sized square is 2 times the smallest square. --/
def middle_square_side : ℕ := 2 * smallest_square_side

/-- The side length of the largest square is 3 times the smallest square. --/
def largest_square_side : ℕ := 3 * smallest_square_side

/-- The number of smallest squares Adam encounters. --/
def num_smallest_squares : ℕ := 5

/-- The number of middle-sized squares Adam encounters. --/
def num_middle_squares : ℕ := 5

/-- The number of largest squares Adam encounters. --/
def num_largest_squares : ℕ := 2

/-- The total distance Adam walks from P to Q. --/
def total_distance : ℕ :=
  num_smallest_squares * smallest_square_side +
  num_middle_squares * middle_square_side +
  num_largest_squares * largest_square_side

/-- Proof that the total distance Adam walks is 420 cm. --/
theorem adam_walks_distance : total_distance = 420 := by
  sorry

end adam_walks_distance_l328_328733


namespace regression_line_zero_corr_l328_328635

-- Definitions based on conditions
variables {X Y : Type}
variables [LinearOrder X] [LinearOrder Y]
variables {f : X → Y}  -- representing the regression line

-- Condition: Regression coefficient b = 0
def regression_coefficient_zero (b : ℝ) : Prop := b = 0

-- Definition of correlation coefficient; here symbolically represented since full derivation requires in-depth statistics definitions
def correlation_coefficient (r : ℝ) : ℝ := r

-- The mathematical goal to prove
theorem regression_line_zero_corr {b r : ℝ} 
  (hb : regression_coefficient_zero b) : correlation_coefficient r = 0 := 
by
  sorry

end regression_line_zero_corr_l328_328635


namespace tan_monotonic_increasing_l328_328014

noncomputable def tan_derivative (x : ℝ) : ℝ := 1 / (Real.cos x)^2

theorem tan_monotonic_increasing :
  ∀ x : ℝ, (-Real.pi/2 < x ∧ x < Real.pi/2) → 
  (∃ δ > 0, ∀ y : ℝ, (x - δ < y ∧ y < x + δ) → f(y) > f(x)) :=
by
  intro x hx
  have hcos : ∀ y : ℝ, (-Real.pi/2 < y ∧ y < Real.pi/2) → Real.cos y > 0 := sorry
  have hderiv : ∀ y : ℝ, (tan_derivative y > 0) := sorry
  show ∃ δ > 0, ∀ y : ℝ, (x - δ < y ∧ y < x + δ) → f y > f x from sorry

end tan_monotonic_increasing_l328_328014


namespace no_1_5_percent_solution_possible_l328_328445

theorem no_1_5_percent_solution_possible 
    (VesselA_initial_volume : ℕ) (VesselA_initial_salt_concentration : ℚ)
    (VesselB_initial_volume : ℕ) (VesselB_initial_salt_concentration : ℚ) :
    VesselA_initial_volume = 1 →
    VesselA_initial_salt_concentration = 0 →
    VesselB_initial_volume = 1 →
    VesselB_initial_salt_concentration = 0.02 →
    ∀ (transfers : list (ℚ × ℚ)), -- list of transfer operations (amount from A to B and from B to A)
    let total_salt := VesselA_initial_volume * VesselA_initial_salt_concentration +
                      VesselB_initial_volume * VesselB_initial_salt_concentration in
    let final_volume := VesselA_initial_volume + VesselB_initial_volume in
    ∀ concentration : ℚ,
    (VesselA_initial_volume + VesselB_initial_volume) = final_volume →
    final_volume = 2 →
    total_salt / final_volume ≤ 0.01 →
    concentration = 0.015 →
    ¬(∃ transfers, concentration = 0.015)
:=
by
  intros _ _ _ _ _ _ _ _ _ _
  sorry

end no_1_5_percent_solution_possible_l328_328445


namespace cookies_in_one_row_l328_328662

theorem cookies_in_one_row
  (num_trays : ℕ) (rows_per_tray : ℕ) (total_cookies : ℕ)
  (h_trays : num_trays = 4) (h_rows : rows_per_tray = 5) (h_cookies : total_cookies = 120) :
  total_cookies / (num_trays * rows_per_tray) = 6 := by
  sorry

end cookies_in_one_row_l328_328662


namespace solve_problem_l328_328106

-- Define the variables and conditions
def problem_statement : Prop :=
  ∃ x : ℕ, 865 * 48 = 240 * x ∧ x = 173

-- Statement to prove
theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l328_328106


namespace y_value_l328_328867

theorem y_value (y : ℕ) : 8^3 + 8^3 + 8^3 + 8^3 = 2^y → y = 11 := 
by 
  sorry

end y_value_l328_328867


namespace problem_l328_328982

theorem problem (x : ℝ) (a_0 a_1 a_2 a_3 a_4 : ℝ) (hx : (2*x + real.sqrt 3)^4 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4) :
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 1 :=
sorry

end problem_l328_328982


namespace total_questions_reviewed_l328_328835

theorem total_questions_reviewed (questions_per_student : ℕ) (students_per_class : ℕ) (number_of_classes : ℕ) :
  questions_per_student = 10 → students_per_class = 35 → number_of_classes = 5 →
  questions_per_student * students_per_class * number_of_classes = 1750 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  norm_num
  sorry

end total_questions_reviewed_l328_328835


namespace find_minutes_before_hours_l328_328273

theorem find_minutes_before_hours (H : ℕ) (M : ℕ) : 
  let θ_minute := 6 * M,
      θ_hour := 30 * H + 0.5 * M,
      m := (480 / 6.5) in
  H = 15 ∧ M = 60 - m → m = 21 + 9 / 11 :=
by 
  sorry

end find_minutes_before_hours_l328_328273


namespace reflection_line_eq_l328_328756

-- Define the vertices of the original triangle
def A := (3, 2)
def B := (8, 7)
def C := (6, -4)

-- Define the vertices of the reflected triangle
def A' := (-5, 2)
def B' := (-10, 7)
def C' := (-8, -4)

-- Definition of the problem to prove the line of reflection
theorem reflection_line_eq :
  ∃ L : ℝ , ∀ (p q : ℝ × ℝ), (p = A ∧ q = A' ∨ p = B ∧ q = B' ∨ p = C ∧ q = C') →
  p.2 = q.2 ∧ p.1 - q.1 = 2 * L := 
sorry

end reflection_line_eq_l328_328756


namespace curve_cartesian_equation_mu_range_l328_328606

theorem curve_cartesian_equation :
  ∀ (θ : ℝ), (let ρ := 4 * cos (θ - (π / 6)) in
    ∃ (C : Type*) [metric_space C] [normed_group C] [normed_space ℝ C]
    (x y : ℝ), 
      (x = ρ * cos θ ∧ y = ρ * sin θ) →
      (x - sqrt 3) ^ 2 + (y - 1) ^ 2 = 4) :=
by
  sorry

theorem mu_range (t : ℝ) (ht : t ∈ set.Icc (-2 : ℝ) 2) :
  let x := sqrt 3 - (sqrt 3 / 2) * t in
  let y := 1 + (1 / 2) * t in
  (sqrt 3 * x + y) ∈ set.Icc (2 : ℝ) 6 :=
by
  sorry

end curve_cartesian_equation_mu_range_l328_328606


namespace hcf_of_two_numbers_l328_328288

noncomputable theory

open Nat

theorem hcf_of_two_numbers (A B : ℕ) 
  (h_lcm : lcm A B = 560) 
  (h_prod : A * B = 42000) : gcd A B = 75 :=
by
  sorry

end hcf_of_two_numbers_l328_328288


namespace coefficient_x3y3_in_expansion_l328_328764

theorem coefficient_x3y3_in_expansion :
  (coeff_of_x3y3_in (x + y)^6) * (constant_term_in (z + 1/z)^8) = 1400 := by
sorry

end coefficient_x3y3_in_expansion_l328_328764


namespace part_a_part_b_l328_328777

-- Part (a): Is it possible to place 4 points with each point connected to 3 others without intersection?
theorem part_a : ∃ (vertices : fin 4 → ℝ × ℝ) (edges : list (fin 4 × fin 4)),
  (∀ v, list.count (list.map prod.snd (list.filter (λ e, e.1 = v) edges)) = 3) ∧
  (∀ e1 e2, e1 ≠ e2 → disjoint_segments e1 e2) :=
sorry

-- Part (b): Is it possible to place 6 points with each point connected to 4 others without intersection (no)?
theorem part_b : ¬ ∃ (vertices : fin 6 → ℝ × ℝ) (edges : list (fin 6 × fin 6)),
  (∀ v, list.count (list.map prod.snd (list.filter (λ e, e.1 = v) edges)) = 4) ∧
  (∀ e1 e2, e1 ≠ e2 → disjoint_segments e1 e2) :=
sorry

-- Assume a function that checks if segments are disjoint.
def disjoint_segments : (ℝ × ℝ) × (ℝ × ℝ) → (ℝ × ℝ) × (ℝ × ℝ) → Prop := 
sorry

end part_a_part_b_l328_328777


namespace hotel_loss_l328_328124

theorem hotel_loss (operations_expenses : ℝ) (payment_fraction : ℝ) (total_payment : ℝ) (loss : ℝ) 
  (hOpExp : operations_expenses = 100) 
  (hPayFr : payment_fraction = 3 / 4)
  (hTotalPay : total_payment = payment_fraction * operations_expenses) 
  (hLossCalc : loss = operations_expenses - total_payment) : 
  loss = 25 := 
by 
  sorry

end hotel_loss_l328_328124


namespace napkin_placements_l328_328140

theorem napkin_placements (n : ℕ) (hn : n ≥ 2) : 
  ∃ f : ℕ → ℕ, (f n = 2^n + 2 * (-1)^n) :=
by {
  use λ n, 2^n + 2 * (-1) ^ n,
  sorry
}

end napkin_placements_l328_328140


namespace minimum_g_value_l328_328724

open Real

noncomputable def g (P Q R S X : Point3D) : ℝ :=
  dist P X + dist Q X + dist R X + dist S X

theorem minimum_g_value (P Q R S : Point3D) :
  dist P R = 26 → dist Q S = 26 →
  dist P S = 34 → dist Q R = 34 →
  dist P Q = 50 → dist R S = 50 →
  ∃ X : Point3D, g P Q R S X = 2 * sqrt 2642 :=
begin
  intros h1 h2 h3 h4 h5 h6,
  -- Proof steps go here
  sorry
end

end minimum_g_value_l328_328724


namespace minimize_expression_is_correct_l328_328188

noncomputable
def minimize_expression : ℝ :=
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (14, 46)
  let dist (P Q : ℝ × ℝ) := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let expression (t : ℝ) := dist (t, t^2) A + dist (t, t^2) B
  classical.some (exists_min 'x, ∀ t, expression t ≥ expression x)

theorem minimize_expression_is_correct : minimize_expression = 7 / 2 :=
sorry

end minimize_expression_is_correct_l328_328188


namespace solve_for_x_l328_328719

theorem solve_for_x (x : ℝ) : (√(3 * x + 7) - √(2 * x - 1) + 2 = 0) → (x = 4) :=
by
  -- The proof is omitted
  sorry

end solve_for_x_l328_328719


namespace not_perfect_square_l328_328004

theorem not_perfect_square (n : ℤ) (hn : n > 4) : ¬ (∃ k : ℕ, n^2 - 3*n = k^2) :=
sorry

end not_perfect_square_l328_328004


namespace find_angle_STP_l328_328650

-- Declare the conditions as hypotheses in Lean
variables (m n : Line) (T : Point n) (P : Point m) (PT : Line) 
variable (S : Point)
variables (angle_SPT angle_TPS : ℝ)

-- Specify the given conditions
hypothesis h1 : m ∥ n
hypothesis h2 : T ∈ n
hypothesis h3 : P ∈ m
hypothesis h4 : PT ⟂ n
hypothesis h5 : angle_SPT = 150
hypothesis h6 : angle_TPS = 20

-- Prove the required question
theorem find_angle_STP : angle_STP = 10 :=
  sorry

end find_angle_STP_l328_328650


namespace correct_height_sum_alices_method_l328_328514

def roundToNearestTen (n : ℤ) : ℤ :=
  if n % 10 = 0 then n
  else n - n % 10 + if n % 10 ≥ 5 then 10 else 0

theorem correct_height_sum (h1 h2 : ℤ)
  (h1_val : h1 = 53) (h2_val : h2 = 78) : 
  roundToNearestTen (h1 + h2) = 130 :=
by
  rw [h1_val, h2_val]
  -- Perform the precise calculation
  exact Modulo.Arith.add_mod_right 131 10
  sorry

theorem alices_method (h1 h2 : ℤ)
  (h1_val : h1 = 53) (h2_val : h2 = 78) : 
  roundToNearestTen (h1 + roundToNearestTen h2) = 130 :=
by
  rw [h1_val, h2_val]
  -- Perform rounding the second height first
  have h2_rounded : roundToNearestTen 78 = 80 := by
    -- Prove that rounding 78 gives 80
    sorry
  rw h2_rounded
  -- Prove that the sum also results in a rounding to 130
  exact Modulo.Arith.add_mod_right 133 10
  sorry

end correct_height_sum_alices_method_l328_328514


namespace problem_solution_l328_328574

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x < 2 then e^x - 1
  else if x < 0 then f (-x)
  else f (x % 2)

theorem problem_solution : f(2016) + f(-2017) = 1 - e := 
sorry

end problem_solution_l328_328574


namespace water_height_in_tank_l328_328519

noncomputable def cone_radius := 10 -- in cm
noncomputable def cone_height := 15 -- in cm
noncomputable def tank_width := 20 -- in cm
noncomputable def tank_length := 30 -- in cm
noncomputable def cone_volume := (1/3:ℝ) * Real.pi * (cone_radius^2) * cone_height
noncomputable def tank_volume (h:ℝ) := tank_width * tank_length * h

theorem water_height_in_tank : ∃ h : ℝ, tank_volume h = cone_volume ∧ h = 5 * Real.pi / 6 := 
by 
  sorry

end water_height_in_tank_l328_328519


namespace more_than_10_weights_missing_l328_328429

/-- 
Given weights of 5, 24, and 43 grams with an equal number of each type
and that the total remaining mass is 606060...60 grams,
prove that more than 10 weights are missing.
-/
theorem more_than_10_weights_missing (total_mass : ℕ) (n : ℕ) (k : ℕ) 
  (total_mass_eq : total_mass = k * (5 + 24 + 43))
  (total_mass_mod : total_mass % 72 ≠ 0) :
  k < n - 10 :=
sorry

end more_than_10_weights_missing_l328_328429


namespace adam_speed_is_correct_l328_328849

variables {d : ℝ} -- d denotes one-third distance to the peak
variable betty_time_to_meet : ℝ

-- Conditions given in the problem
def betty_time_first_segment (d : ℝ) : ℝ := d / 3
def betty_time_second_segment (d : ℝ) : ℝ := 2 * d / 3
def betty_time_descent_to_meet (d : ℝ) : ℝ := (d / 3) / 2.5
def betty_total_time_to_meet (d : ℝ) : ℝ := betty_time_first_segment d + betty_time_second_segment d + betty_time_descent_to_meet d
def adam_distance_to_meet (d : ℝ) : ℝ := 2 * d / 3

-- Betty and Adam meet at one-third mark from the peak
def betty_and_adam_meet (d : ℝ) : Prop :=
  betty_time_to_meet = betty_total_time_to_meet d

-- Adam's average speed
def adam_average_speed (d : ℝ) (t : ℝ) : ℝ := (2 * d / 3) / t

noncomputable def adam_speed_proof (d : ℝ) (t : ℝ) : Prop :=
  adam_average_speed d t = 10 / 17

theorem adam_speed_is_correct (d : ℝ) (bt : ℝ) (ht : betty_and_adam_meet d) : adam_speed_proof d bt :=
by
  sorry

end adam_speed_is_correct_l328_328849


namespace initial_welders_count_l328_328399

theorem initial_welders_count (W : ℕ) (h1: (1 + 16 * (W - 9) / W = 8)) : W = 16 :=
by {
  sorry
}

end initial_welders_count_l328_328399


namespace tree_circumference_inequality_l328_328547

theorem tree_circumference_inequality (x : ℝ) : 
  (∀ t : ℝ, t = 10 + 3 * x ∧ t > 90 → x > 80 / 3) :=
by
  intro t ht
  obtain ⟨h_t_eq, h_t_gt_90⟩ := ht
  linarith

end tree_circumference_inequality_l328_328547


namespace wheel_speed_l328_328673

theorem wheel_speed (r : ℝ) (c : ℝ) (ts tf : ℝ) 
  (h₁ : c = 13) 
  (h₂ : r * ts = c / 5280) 
  (h₃ : (r + 6) * (tf - 1/3 / 3600) = c / 5280) 
  (h₄ : tf = ts - 1 / 10800) :
  r = 12 :=
  sorry

end wheel_speed_l328_328673


namespace right_triangle_hypotenuse_l328_328055

noncomputable def hypotenuse_length (y : ℝ) : ℝ := 
  let shorter_leg := 0.5 * y - 3
  let h_squared := y^2 + shorter_leg^2
  real.sqrt h_squared

theorem right_triangle_hypotenuse : 
  ∃ (y : ℝ), ((0.5 * y - 3) * y / 2 = 84) ∧
              (hypotenuse_length y ≈ 22.96) :=
sorry

end right_triangle_hypotenuse_l328_328055


namespace sin_cos_pi_div_12_l328_328057

theorem sin_cos_pi_div_12 :
  sin (π / 12) * cos (π / 12) = 1 / 4 :=
by
  sorry

end sin_cos_pi_div_12_l328_328057


namespace arithmetic_progression_first_three_terms_l328_328887

theorem arithmetic_progression_first_three_terms 
  (S_n : ℤ) (d a_1 a_2 a_3 a_5 : ℤ)
  (h1 : S_n = 112) 
  (h2 : (a_1 + d) * d = 30)
  (h3 : (a_1 + 2 * d) + (a_1 + 4 * d) = 32) 
  (h4 : ∀ (n : ℕ), S_n = (n * (2 * a_1 + (n - 1) * d)) / 2) : 
  ((a_1 = 7 ∧ a_2 = 10 ∧ a_3 = 13) ∨ (a_1 = 1 ∧ a_2 = 6 ∧ a_3 = 11)) :=
sorry

end arithmetic_progression_first_three_terms_l328_328887


namespace max_area_triangle_abp_l328_328943

noncomputable def curve_c_polar (ρ : ℝ) (θ : ℝ) : Prop := ρ = sqrt 3
noncomputable def line_l_parametric (x y t : ℝ) : Prop := (x = 1 + (sqrt 2 / 2) * t) ∧ (y = (sqrt 2 / 2) * t)
noncomputable def curve_m_parametric (x y θ : ℝ) : Prop := (x = cos θ) ∧ (y = sqrt 3 * sin θ)

theorem max_area_triangle_abp :
  let x := ∀ (t : ℝ), 1 + (sqrt 2 / 2) * t in
  let y := ∀ (t : ℝ), (sqrt 2 / 2) * t in
  let c := ∀ (ρ θ : ℝ), ρ = sqrt 3 in
  (∀ (x y : ℝ), c (sqrt (x^2 + y^2)) = sqrt 3 → x^2 + y^2 = 3) ∧
  (∃ (t : ℝ), y t = x t - 1) ∧
  (∃ (t1 t2 : ℝ), t1 + t2 = -sqrt 2 ∧ t1 * t2 = -2) ∧
  (∀ (θ : ℝ), abs (cos θ - sqrt 3 * sin θ - 1) / sqrt 2 ≤ 3 * sqrt 2 / 2) →
  ∃ (area : ℝ), area = 3 * sqrt 5 / 2 :=
by
  sorry

end max_area_triangle_abp_l328_328943


namespace range_of_m_l328_328231

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, mx^2 + 2 * m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 1) := by
  sorry

end range_of_m_l328_328231


namespace svetlana_triplet_difference_invariant_l328_328788

/-- Svetlana's transformation rule on a triplet of numbers replaces each number by the sum of the other two. 
Prove that after 1580 steps, given an initial triplet {80, 71, 20}, the difference between the largest 
and smallest number remains 60. -/
theorem svetlana_triplet_difference_invariant (x a b : ℕ) (n : ℕ) 
  (initial_triplet : x = 20 ∧ a = 71 - 20 ∧ b = 80 - 20)
  (transformation_rule : ∀ (x y z : ℕ), (x', y', z') = (y + z, x + z, x + y)) : 
  n = 1580 → ∃ d : ℕ, d = 60 ∧ ∀ (x y z : ℕ), abs (max (max x y) z - min (min x y) z) = d := 
by
  intros
  subst_vars
  sorry

end svetlana_triplet_difference_invariant_l328_328788
