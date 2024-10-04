import Mathlib
import Mathlib.Algebra.EuclideanDomain.Basic
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupWithZero.Power
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Power
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.Monotone
import Mathlib.Analysis.SpecialFunctions.ExpLog.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Probability--
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Log
import Mathlib.Geometry.Convex
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Algebra.Order.Basic
import Mathlib.Topology.ProbabilityTheory
import Probability.Basic
import Real

namespace binary_to_octal_101110_l12_12153

theorem binary_to_octal_101110 : 
  ∀ (binary_to_octal : ℕ → ℕ), 
  binary_to_octal 0b101110 = 0o56 :=
by
  sorry

end binary_to_octal_101110_l12_12153


namespace number_of_kg_of_mangoes_l12_12055

variable {m : ℕ}
def cost_apples := 8 * 70
def cost_mangoes (m : ℕ) := 75 * m
def total_cost := 1235

theorem number_of_kg_of_mangoes (h : cost_apples + cost_mangoes m = total_cost) : m = 9 :=
by
  sorry

end number_of_kg_of_mangoes_l12_12055


namespace monotonicity_tangent_points_l12_12278

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12278


namespace largest_prime_factor_5040_l12_12959

theorem largest_prime_factor_5040 : ∃ p, Nat.Prime p ∧ p ∣ 5040 ∧ (∀ q, Nat.Prime q ∧ q ∣ 5040 → q ≤ p) := by
  use 7
  constructor
  · exact Nat.prime_7
  constructor
  · exact dvd.intro (2^4 * 3^2 * 5) rfl
  · intros q hq
    cases hq with hq1 hq2
    exact Nat.le_of_dvd (Nat.pos_of_ne_zero (λ hq3, by linarith)) hq2
  sorry

end largest_prime_factor_5040_l12_12959


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12304

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12304


namespace StatementA_StatementB_StatementC_StatementD_l12_12232

noncomputable theory
open Classical

-- Define necessary constructs from the problem statement
variables (p x1 x2 y1 y2 : ℝ) (F M N : ℝ × ℝ) (OMN : ℝ)

-- Given conditions
def parabola (x y p : ℝ) : Prop := x^2 = 2 * p * y
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def line_through_focus_with_slope (F : ℝ × ℝ) (slope : ℝ) : ℝ × ℝ → Prop :=
  λ P, ∃ x, P = (x, F.2 + slope * (x - F.1))

def intersects_parabola_at_points (l : ℝ × ℝ → Prop) (points : list (ℝ × ℝ)) : Prop :=
  ∀ P ∈ points, l P ∧ parabola P.1 P.2 p

-- Proof statements as assertions
theorem StatementA : (parabola x1 y1 2) → (parabola x2 y2 2) → (x1 ≠ 0) → 
                     (focus 2 = F) → line_through_focus_with_slope F (sqrt 3) (x1, y1)  →
                     x1 * x2 = -4 := sorry

theorem StatementB : (y1 * y2 = 1) → (parabola x1 y1 p) → 
                     (parabola x2 y2 p) → 
                     (x1 ≠ 0) → 
                     (focus p = F) → 
                     line_through_focus_with_slope F (sqrt 3) (x1, y1) →
                     p = 2 := sorry

theorem StatementC : (parabola x1 y1 2) → (parabola x2 y2 2) → 
                     (x1 ≠ 0) → 
                     (focus 2 = F) → 
                     line_through_focus_with_slope F (sqrt 3) (x1, y1) → 
                     S_OMN ≠ (4 / 3) * sqrt 3 := sorry

theorem StatementD : (parabola x1 y1 2) → (parabola x2 y2 2) → 
                     (x1 ≠ 0) → 
                     (focus 2 = F) → 
                     line_through_focus_with_slope F (sqrt 3) (x1, y1) →
                     dist (x1, y1) F = 8 + 4 * sqrt 3 := sorry

end StatementA_StatementB_StatementC_StatementD_l12_12232


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12296

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12296


namespace negation_is_false_l12_12026

-- Define the original proposition
def original_proposition : Prop := ∀ x, 0 < x ∧ x < π / 2 → sin x < 1

-- Define the negation of the original proposition
def negated_proposition : Prop := ¬original_proposition

-- Given that the original proposition is true, prove that the negation is false
theorem negation_is_false (h : original_proposition) : negated_proposition = false := 
by 
  sorry

end negation_is_false_l12_12026


namespace genevieve_errors_fixed_l12_12196

theorem genevieve_errors_fixed (total_lines : ℕ) (lines_per_debug : ℕ) (errors_per_debug : ℕ)
  (h_total_lines : total_lines = 4300)
  (h_lines_per_debug : lines_per_debug = 100)
  (h_errors_per_debug : errors_per_debug = 3) :
  (total_lines / lines_per_debug) * errors_per_debug = 129 :=
by
  -- Placeholder proof to indicate the theorem should be true
  sorry

end genevieve_errors_fixed_l12_12196


namespace tangent_line_at_zero_l12_12405

def f (x : ℝ) : ℝ := x^2 * Real.exp x + 2 * x + 1

theorem tangent_line_at_zero :
  ∃ (m b : ℝ), (m = 2) ∧ (b = 1) ∧ (∀ x, (2 * x - (m * x + b) + 1 = 0)) :=
sorry

end tangent_line_at_zero_l12_12405


namespace polynomial_min_degree_l12_12003

theorem polynomial_min_degree :
  ∀ (P : Polynomial ℚ), P ≠ 0 →
  (2 - Real.sqrt 5) ∈ P.root_set ℚ → (-2 - Real.sqrt 5) ∈ P.root_set ℚ →
  (3 + Real.sqrt 11) ∈ P.root_set ℚ → (3 - Real.sqrt 11) ∈ P.root_set ℚ →
  P.degree ≥ 6 :=
by
  sorry

end polynomial_min_degree_l12_12003


namespace monotonicity_and_tangent_intersection_l12_12255

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12255


namespace ellipse_focal_length_l12_12665

theorem ellipse_focal_length (a b : ℝ) (h_a : a^2 = 25) (h_b : b^2 = 16) :
  2 * real.sqrt (a^2 - b^2) = 6 :=
by
  sorry

end ellipse_focal_length_l12_12665


namespace monotonicity_of_f_tangent_intersection_points_l12_12326

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12326


namespace compute_alpha_l12_12846

-- Declaring that we are using complex numbers
variable (α β : ℂ)

-- Conditions given in the problem
-- Condition 1 and 2: α, β are complex numbers (implicitly handled by type declaration)
-- Condition 3: α + β is a positive real number
def alpha_plus_beta_is_positive_real (α β : ℂ) : Prop := ∃ r : ℝ, r > 0 ∧ α + β = ↑r

-- Condition 4: i(α - 3β) is a positive real number
def i_times_alpha_minus_3beta_is_positive_real (α β : ℂ) : Prop := 
  ∃ r : ℝ, r > 0 ∧ i * (α - 3 * β) = ↑r

-- Condition 5: β = 4 + 3i
def beta_value : ℂ := 4 + 3i

-- The theorem to compute α
theorem compute_alpha (α : ℂ) (β : ℂ) (h1 : alpha_plus_beta_is_positive_real α β) 
  (h2 : i_times_alpha_minus_3beta_is_positive_real α β) 
  (h3 : β = beta_value) : α = 12 - 3i := 
sorry

end compute_alpha_l12_12846


namespace monotonicity_of_f_tangent_intersection_points_l12_12321

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12321


namespace divide_books_into_portions_l12_12931

theorem divide_books_into_portions (n : ℕ) (hn : n = 6) :
  (nat.choose 6 2 * nat.choose 4 2) / 6 = 15 := sorry

end divide_books_into_portions_l12_12931


namespace parabola_integer_points_l12_12915

-- Definitions based on conditions
def parabola (focus : ℝ × ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ (x y : ℝ), (x, y) ∈ {f : ℝ × ℝ | f = (x, y) ∧ y = a*x^2 + b*x + c}

-- Proposition based on the problem requirement
def parabola_has_focus_and_points : Prop :=
  parabola (2, 2) (6, 5) (2, -1)

noncomputable def num_integer_points (Q : set (ℝ × ℝ)) : ℕ :=
  {p : ℤ × ℤ | Q (p.1, p.2) ∧ |3 * p.1 + 4 * p.2| ≤ 800}.to_finset.card

-- Final statement to prove
theorem parabola_integer_points : parabola_has_focus_and_points →
  num_integer_points { p : ℝ × ℝ | parabola_has_focus_and_points p } = 36 :=
by
  sorry

end parabola_integer_points_l12_12915


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12339

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12339


namespace sum_of_sequence_l12_12015

theorem sum_of_sequence (avg : ℕ → ℕ → ℕ) (n : ℕ) (total_sum : ℕ) 
  (condition : avg 16 272 = 17) : 
  total_sum = 272 := 
by 
  sorry

end sum_of_sequence_l12_12015


namespace max_product_not_less_than_993_squared_l12_12565

theorem max_product_not_less_than_993_squared :
  ∀ (a : Fin 1985 → ℕ), 
    (∀ i, ∃ j, a j = i + 1) →  -- representation of permutation
    (∃ i : Fin 1985, i * (a i) ≥ 993 * 993) :=
by
  intros a h
  sorry

end max_product_not_less_than_993_squared_l12_12565


namespace math_proof_problem_l12_12236

-- Define the initial conditions and given points
def M : ℝ × ℝ := (2/3, 2 * (Real.sqrt 6) / 3)

-- Such that M is also on the parabola and hyperbola
def parabola_eq (y : ℝ) (x : ℝ) := y^2 = 4 * x
def hyperbola_eq (x : ℝ) (y : ℝ) := x^2 / (1/9) - y^2 / (8/9) = 1

-- Define the coordinates of the focus of the parabola
def parabola_focus : ℝ × ℝ := (1, 0)

-- Define the coordinates of the foci of the hyperbola
def hyperbola_focus_1 : ℝ × ℝ := (-1, 0)
def hyperbola_focus_2 : ℝ × ℝ := (1, 0)

-- Define the eccentricity of the hyperbola
def hyperbola_eccentricity := 3

-- Theorem to prove the correct answers according to given conditions
theorem math_proof_problem 
    (h_parabola : parabola_eq M.2 M.1)
    (h_hyperbola : hyperbola_eq M.1 M.2)
    (h_vertex_parabola : parabola_eq 0 0) 
    (h_axis_parabola_focus : parabola_focus = (1, 0))
    : 
    parabola_eq = (λ y x, y^2 = 4 * x) ∧ 
    hyperbola_eq = (λ x y, x^2 / (1/9) - y^2 / (8/9) = 1) ∧
    hyperbola_eccentricity = 3 :=
by
  sorry

end math_proof_problem_l12_12236


namespace algebraic_expression_value_l12_12753

variable (x : ℝ)

theorem algebraic_expression_value (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 2 = 4 :=
by
  -- This is where the detailed proof would go, but we are skipping it with sorry.
  sorry

end algebraic_expression_value_l12_12753


namespace horner_method_correct_l12_12592

def polynomial_value_at_2 : Prop :=
  let f := λ x : ℕ, 7 * x^3 + 3 * x^2 - 5 * x + 11
  f 2 = 69

theorem horner_method_correct : polynomial_value_at_2 :=
by
  sorry

end horner_method_correct_l12_12592


namespace product_of_all_possible_N_l12_12137

theorem product_of_all_possible_N (A B N : ℝ) 
  (h1 : A = B + N)
  (h2 : A - 4 = B + N - 4)
  (h3 : B + 5 = B + 5)
  (h4 : |((B + N - 4) - (B + 5))| = 1) :
  ∃ N₁ N₂ : ℝ, (|N₁ - 9| = 1 ∧ |N₂ - 9| = 1) ∧ N₁ * N₂ = 80 :=
by {
  -- We know the absolute value equation leads to two solutions
  -- hence we will consider N₁ and N₂ such that |N - 9| = 1
  -- which eventually yields N = 10 and N = 8, making their product 80.
  sorry
}

end product_of_all_possible_N_l12_12137


namespace suff_not_necess_cond_lnotP_lnotq_l12_12197

variable {x : ℝ}

def P := |5 * x - 2| > 3
def q := (1 / (x^2 + 4 * x - 5)) > 0

theorem suff_not_necess_cond_lnotP_lnotq (hP: ¬ P) (hq: ¬ q):
  ¬ P → (¬ q → (¬ P → ¬ q)) :=
by
  sorry

end suff_not_necess_cond_lnotP_lnotq_l12_12197


namespace sum_of_coeffs_a_plus_b_pow_7_l12_12037

theorem sum_of_coeffs_a_plus_b_pow_7 :
  (∑ k in Finset.range (7 + 1), Nat.choose 7 k) = 128 :=
by
  sorry

end sum_of_coeffs_a_plus_b_pow_7_l12_12037


namespace sphere_surface_area_l12_12925

-- Definitions for problem conditions
def surface_area_cube := 24 -- cm^2

def vertices_on_sphere (side_length_cube : ℝ) (radius_sphere : ℝ) : Prop :=
  let diagonal := 2 * radius_sphere in
  let cube_side := side_length_cube in
  diagonal^2 = cube_side^2 + cube_side^2 + cube_side^2

-- Given surface area of the cube we get the side length
def side_length_cube := sqrt(surface_area_cube / 6) -- Since surface area = 6 * side^2

-- Problem statement in Lean 4
theorem sphere_surface_area : vertices_on_sphere side_length_cube (sqrt 3) →
  ∃ (S : ℝ), S = 12 * Real.pi :=
by
  sorry

end sphere_surface_area_l12_12925


namespace one_third_of_1206_is_100_5_percent_of_400_l12_12873

theorem one_third_of_1206_is_100_5_percent_of_400 (n m : ℕ) (f : ℝ) :
  n = 1206 → m = 400 → f = 1 / 3 → (n * f) / m * 100 = 100.5 :=
by
  intros h_n h_m h_f
  rw [h_n, h_m, h_f]
  sorry

end one_third_of_1206_is_100_5_percent_of_400_l12_12873


namespace sum_of_first_15_terms_l12_12070

variables (a11 a7 : ℕ → ℝ) (a1 d : ℝ)

-- Conditions
axiom h1 : a11 = 5.25
axiom h2 : a7 = 3.25

-- Definition of the nth term of an AP
def a (n : ℕ) : ℝ := a1 + (n - 1) * d

-- Sum of the first n terms of the AP
def S (n : ℕ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

-- The theorem to prove
theorem sum_of_first_15_terms
  (h1 : a 11 = 5.25)
  (h2 : a 7 = 3.25)
  (a1 : ℝ)
  (d : ℝ) :
  S 15 = 30 :=
sorry

end sum_of_first_15_terms_l12_12070


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12317

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12317


namespace sequence_no_three_primes_l12_12186

noncomputable def x_n (a b n : ℕ) : ℚ :=
(a^n - 1) / (b^n - 1)

theorem sequence_no_three_primes (a b : ℕ) (h1 : a > b) (h2 : b > 1) :
  ¬ ∃ n, Nat.Prime (x_n a b n) ∧ Nat.Prime (x_n a b (n + 1)) ∧ Nat.Prime (x_n a b (n + 2)) :=
sorry

end sequence_no_three_primes_l12_12186


namespace max_theta_condition_l12_12035

variables (Mg L μ : ℝ) (g θ : ℝ)

-- Conditions
def normal_force (m g : ℝ) : ℝ := m * g / 2

def frictional_force (μ Fₙ : ℝ) : ℝ := μ * Fₙ

def torque_gravity (L Mg θ : ℝ) : ℝ := (L / 2) * (Mg / 2) * sin (θ / 2)

def torque_friction (L F θ : ℝ) : ℝ := L * F * cos (θ / 2)

-- Maximum θ ensures balance of torques
theorem max_theta_condition (h1 : normal_force (Mg / 2) g = Mg * g / 2)
    (h2 : frictional_force μ (Mg * g / 2) = μ * (Mg * g / 2))
    (h3 : torque_gravity L Mg θ = (L / 2) * (Mg / 2) * sin (θ / 2))
    (h4 : torque_friction L (μ * (Mg * g / 2)) θ = L * (μ * (Mg * g / 2)) * cos (θ / 2))
    : tan (θ / 2) = 2 * μ := 
sorry

end max_theta_condition_l12_12035


namespace monotonicity_f_tangent_points_l12_12365

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12365


namespace simplify_expression_l12_12672

theorem simplify_expression : (2^4 * 2^4 * 2^4) = 2^12 :=
by
  sorry

end simplify_expression_l12_12672


namespace seq_2010_eq_2_l12_12445

def seq : ℕ → ℕ
| 0       := 1
| 1       := 1
| (k + 2) := seq (k + 1) + 1 - 4 * (nat.floor ((k + 1) / 4) - nat.floor (k / 4))

theorem seq_2010_eq_2 : seq 2010 = 2 :=
sorry

end seq_2010_eq_2_l12_12445


namespace monotonicity_tangent_intersection_points_l12_12380

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12380


namespace part1_monotonicity_part2_tangent_intersection_l12_12263

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12263


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12301

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12301


namespace monotonicity_tangent_intersection_points_l12_12381

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12381


namespace concurrent_ceva_points_l12_12128

theorem concurrent_ceva_points
  (A B C H O D E F : Point)
  (h_tri_acute : acute_angled_triangle A B C)
  (orthocenter_H : orthocenter A B C H)
  (circumcenter_O : circumcenter A B C O)
  (D_on_BC : lies_on D (line B C))
  (E_on_CA : lies_on E (line C A))
  (F_on_AB : lies_on F (line A B)):
  (distance O D + distance D H = distance O E + distance E H) ∧
  (distance O E + distance E H = distance O F + distance F H) ∧
  (distance O F + distance F H = distance O D + distance D H) ∧
  concurrent (line A D) (line B E) (line C F) :=
sorry

end concurrent_ceva_points_l12_12128


namespace systematic_sampling_correct_l12_12729

open Finset

theorem systematic_sampling_correct : 
  ∃ (s : Finset ℕ), (s = {3, 13, 23, 33, 43}) ∧ (∀ k ∈ s, k ∈ range 51) ∧
    (∃ d : ℕ, d > 0 ∧ ∀ x y ∈ s, x ≠ y → |x - y| = d) :=
sorry

end systematic_sampling_correct_l12_12729


namespace min_value_of_fraction_l12_12408

theorem min_value_of_fraction (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (4 / (a + 2) + 1 / (b + 1)) = 9 / 4 :=
sorry

end min_value_of_fraction_l12_12408


namespace find_alpha_l12_12841

noncomputable def α_and_β_are_complex_numbers (α β : ℂ) : Prop := true
noncomputable def α_plus_β_is_positive_real (α β : ℂ) : Prop := ∃ r : ℝ, r > 0 ∧ (α + β) = r
noncomputable def i_α_minus_3β_is_positive_real (α β : ℂ) : Prop := ∃ r : ℝ, r > 0 ∧ (i * (α - 3 * β)) = r
def beta_value : ℂ := 4 + 3 * complex.i

theorem find_alpha (α : ℂ) (β : ℂ := beta_value) 
  (h1 : α_and_β_are_complex_numbers α β) 
  (h2 : α_plus_β_is_positive_real α β)
  (h3 : i_α_minus_3β_is_positive_real α β) 
  (h4 : β = beta_value) : α = 3 - 3 * complex.i := 
by
  sorry

end find_alpha_l12_12841


namespace solution_set_f_lt_0_l12_12769

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_positive : ∀ x : ℝ, x > 0 → f x = Real.log x

theorem solution_set_f_lt_0 : { x : ℝ | f x < 0 } = set.Iio (-1) ∪ set.Ioo 0 1 := by
  sorry

end solution_set_f_lt_0_l12_12769


namespace fraction_spent_is_one_third_l12_12079

-- Define the initial conditions and money variables
def initial_money := 32
def cost_bread := 3
def cost_candy := 2
def remaining_money_after_all := 18

-- Define the calculation for the money left after buying bread and candy bar
def money_left_after_bread_candy := initial_money - cost_bread - cost_candy

-- Define the calculation for the money spent on turkey
def money_spent_on_turkey := money_left_after_bread_candy - remaining_money_after_all

-- The fraction of the remaining money spent on the Turkey
noncomputable def fraction_spent_on_turkey := (money_spent_on_turkey : ℚ) / money_left_after_bread_candy

-- State the theorem that verifies the fraction spent on turkey is 1/3
theorem fraction_spent_is_one_third : fraction_spent_on_turkey = 1 / 3 := by
  sorry

end fraction_spent_is_one_third_l12_12079


namespace average_of_last_5_numbers_is_correct_l12_12541

-- Given conditions:
-- avg_total: average of 9 numbers
-- avg_first_5: average of first 5 numbers
-- num_5: the 5th number
def avg_total := 104
def avg_first_5 := 99
def num_5 := 59

-- We should prove:
-- avg_last_5: the average of the last 5 numbers must be 111.8
theorem average_of_last_5_numbers_is_correct :
  let S1 := avg_first_5 * 5 in
  let S_total := avg_total * 9 in
  let S2 := S_total - S1 + num_5 * 2 in
  S2 / 5 = 111.8 :=
by
  sorry

end average_of_last_5_numbers_is_correct_l12_12541


namespace integral_value_l12_12088

noncomputable def definite_integral := ∫ x in 0..π, 2^4 * (Real.sin (x / 2))^4 * (Real.cos (x / 2))^4

theorem integral_value : definite_integral = (3 * Real.pi) / 8 := 
  sorry

end integral_value_l12_12088


namespace points_concyclic_l12_12854

theorem points_concyclic 
  (A B C P Q C1 B1 : Point)
  (h_acute : triangle.is_acute ABC)
  (h_PQ_on_BC : line.contains_points (line.through B C) P ∧ line.contains_points (line.through B C) Q)
  (h_quadrilateral_APBC1_cyclic : cyclic_quadrilateral A P B C1)
  (h_QC1_parallel_CA : line.parallel (line.through Q C1) (line.through C A))
  (h_C1_Q_opposite_AB : line.opposite_sides (line.through A B) C1 Q)
  (h_quadrilateral_APCB1_cyclic : cyclic_quadrilateral A P C B1)
  (h_QB1_parallel_BA : line.parallel (line.through Q B1) (line.through B A))
  (h_B1_Q_opposite_AC : line.opposite_sides (line.through A C) B1 Q) : 
  concyclic B1 C1 P Q :=
sorry

end points_concyclic_l12_12854


namespace ice_cream_stack_orders_l12_12526

theorem ice_cream_stack_orders : 
  let flavors := ["vanilla", "chocolate", "strawberry", "cherry"] in 
  let ways_to_stack (l : List String) : Nat := l.permutations.length in 
  ways_to_stack flavors = 24 := 
by 
  sorry

end ice_cream_stack_orders_l12_12526


namespace steve_book_earning_l12_12002

theorem steve_book_earning
  (total_copies : ℕ)
  (advance_copies : ℕ)
  (total_kept : ℝ)
  (agent_cut_percentage : ℝ)
  (copies : ℕ)
  (money_kept : ℝ)
  (x : ℝ)
  (h1 : total_copies = 1000000)
  (h2 : advance_copies = 100000)
  (h3 : total_kept = 1620000)
  (h4 : agent_cut_percentage = 0.10)
  (h5 : copies = total_copies - advance_copies)
  (h6 : money_kept = copies * (1 - agent_cut_percentage) * x)
  (h7 : money_kept = total_kept) :
  x = 2 := 
by 
  sorry

end steve_book_earning_l12_12002


namespace monotonicity_of_f_tangent_intersection_points_l12_12329

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12329


namespace minimum_recolored_edges_l12_12584

theorem minimum_recolored_edges (n : ℕ) (h : 3 ≤ n) (E : Finset (Finset ℕ)) 
  (hE : ∀ e ∈ E, ∃ (c : Fin 3), e = {c}) :
  ∃ (k : ℕ), k = n - 1 ∧
    ∀ (recolor : Finset (Finset ℕ) → Finset (Finset ℕ)),
    (∀ (e ∈ E) (c : Fin 3), recolor (insert e E) = insert {c} (recolor E)) →
    (∃ C, ∃ T ⊆ E, T.card = n - 1 ∧ connected_subgraph T {C}) :=
sorry

end minimum_recolored_edges_l12_12584


namespace monotonicity_and_tangent_intersection_l12_12251

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12251


namespace part1_part2_l12_12199

variable (a b : ℝ × ℝ × ℝ)

-- Part (1)
theorem part1 (h₁ : a = (2, -1, -4)) (h₂ : b = (-1, k, 2))
  (h₃ : let v1 := (a.1 - b.1, a.2 - b.2, a.3 - b.3),
               v2 := (a.1 + b.1, a.2 + b.2, a.3 + b.3) in
             v1.1 / v2.1 = v1.2 / v2.2 ∧ v1.2 / v2.2 = v1.3 / v2.3) :
  k = 1/2 := sorry

-- Part (2)
theorem part2 (h₁ : a = (2, -1, -4)) (h₂ : b = (-1, k, 2))
  (h₃ : let v1 := (a.1 + 3 * b.1, a.2 + 3 * b.2, a.3 + 3 * b.3),
               v2 := (a.1 + b.1, a.2 + b.2, a.3 + b.3) in
             v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0) :
  k = -2/3 ∨ k = 2 := sorry

end part1_part2_l12_12199


namespace monotonicity_of_f_tangent_intersection_points_l12_12325

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12325


namespace difference_between_place_and_face_value_l12_12819

def numeral : Nat := 856973

def digit_of_interest : Nat := 7

def place_value : Nat := 7 * 10

def face_value : Nat := 7

theorem difference_between_place_and_face_value : place_value - face_value = 63 :=
by
  sorry

end difference_between_place_and_face_value_l12_12819


namespace all_elements_are_integers_l12_12666

noncomputable def seq (n : ℕ) : ℕ := 
  if h : n = 0 then 0 
  else @Nat.rec (fun _ => ℕ) (1 : ℕ) 
              (fun m am => 2 * am + Nat.sqrt (3 * am * am + 1)) 
              (n - 1) h

theorem all_elements_are_integers : ∀ (n : ℕ), ∃ k : ℕ, seq n = k :=
by
  intro n
  induction n with
  | zero => 
      use 0
      exact rfl
  | succ n ih =>
      cases ih with
      | intro k hk =>
          use 2 * k + Nat.sqrt (3 * k * k + 1)
          rw [seq, if_neg (Nat.succ_ne_zero n), Nat.rec]
          exact hk

end all_elements_are_integers_l12_12666


namespace infinite_product_eq_sqrt27_l12_12701

theorem infinite_product_eq_sqrt27 : 
  (∏ n : ℕ in (range (nat.succ n)).erase 0, (3^n)^(1 / (3^n : ℝ))) = real.sqrt (real.sqrt 27) := 
sorry

end infinite_product_eq_sqrt27_l12_12701


namespace value_of_polynomial_l12_12606

theorem value_of_polynomial :
  98^3 + 3 * (98^2) + 3 * 98 + 1 = 970299 :=
by sorry

end value_of_polynomial_l12_12606


namespace product_eq_44_implies_n_eq_5_l12_12201

theorem product_eq_44_implies_n_eq_5 (n : ℕ) 
  (h : nat.floor (∏ k in finset.range(n) + 1, ((2 * k + 1)^4 + (2 * k + 1)^2 + 1) / (4 * k^4 + 2 * k^2 + 1)) = 44) 
  : n = 5 := 
by 
  sorry

end product_eq_44_implies_n_eq_5_l12_12201


namespace sin_2y_computation_l12_12825

theorem sin_2y_computation (x y : ℝ) (hx : sin x = (3/2) * sin y - (2/3) * cos y) (hx2 : cos x = (3/2) * cos y - (2/3) * sin y) :
  sin (2 * y) = 25 / 72 :=
  sorry

end sin_2y_computation_l12_12825


namespace retailer_profit_percentage_l12_12973

theorem retailer_profit_percentage
  (cost_price : ℝ)
  (marked_percent : ℝ)
  (discount_percent : ℝ)
  (selling_price : ℝ)
  (marked_price : ℝ)
  (profit_percent : ℝ) :
  marked_percent = 60 →
  discount_percent = 25 →
  marked_price = cost_price * (1 + marked_percent / 100) →
  selling_price = marked_price * (1 - discount_percent / 100) →
  profit_percent = ((selling_price - cost_price) / cost_price) * 100 →
  profit_percent = 20 :=
by
  sorry

end retailer_profit_percentage_l12_12973


namespace problems_left_to_grade_l12_12127

theorem problems_left_to_grade 
  (problems_per_worksheet : ℕ)
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (h1 : problems_per_worksheet = 2)
  (h2 : total_worksheets = 14)
  (h3 : graded_worksheets = 7) : 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 14 :=
by 
  sorry

end problems_left_to_grade_l12_12127


namespace smallest_m_geq_l12_12719

noncomputable def smallest_m := 2

theorem smallest_m_geq :
  ∀ (a b c d e : ℝ), 0 < a → 0 < b → 0 < c → 0 < d → 0 < e →
  sqrt (a / (b + c + d + e)) +
  sqrt (b / (a + c + d + e)) +
  sqrt (c / (a + b + d + e)) +
  sqrt (d / (a + b + c + e)) +
  sqrt (e / (a + b + c + d)) ≥ smallest_m :=
by
  sorry

end smallest_m_geq_l12_12719


namespace monotonicity_and_tangent_intersections_l12_12349

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12349


namespace max_distance_point_ellipse_to_line_l12_12560

theorem max_distance_point_ellipse_to_line :
  ∀ (P : ℝ × ℝ), (7 * P.1^2 + 4 * P.2^2 = 28) →
  ∃ d : ℝ, ∀ Q, (Q = P) → (Q.1 ≠ 0 ∨ Q.2 ≠ 0) →
  d = (abs (6 * P.1 - 2 * sqrt 7 * P.2 - 16) / sqrt (3^2 + (-2)^2)) →
  d ≤ (24 / 13) * sqrt 13 := 
by
  intros P h1
  use (24 / 13) * sqrt 13
  sorry

end max_distance_point_ellipse_to_line_l12_12560


namespace ratio_AN_n_NC_eq_34_19_l12_12797

-- Define the basic conditions for the problem
variables {A B C N : Point} (AB BC AC : ℝ)
variable (h_tri : triangle ABC)

-- Express the given lengths
axiom AB_eq_10 : AB = 10
axiom BC_eq_11 : BC = 11
axiom AC_eq_16 : AC = 16

-- Assume the condition that the incircles of the sub-triangles have equal radii
axiom incircles_equal_radii : 
  ∀ {A B C N : Point} (h_tri : triangle ABC) (AN NC : ℝ),
  N ∈ line_segment A C →
  incircle_of_triangle A B N = incircle_of_triangle B C N

-- Define the goal to prove the required ratio
theorem ratio_AN_n_NC_eq_34_19 : 
  ∀ {A B C N : Point} (h_tri : triangle ABC) (H : N ∈ line_segment A C),
  (AB = 10) ∧ (BC = 11) ∧ (AC = 16) ∧ 
  (incircle_of_triangle A B N = incircle_of_triangle B C N) →
  (AN / NC = 34 / 19) :=
begin
  sorry
end

end ratio_AN_n_NC_eq_34_19_l12_12797


namespace scoops_per_carton_l12_12100

-- Definitions for scoops required by everyone
def ethan_vanilla := 1
def ethan_chocolate := 1
def lucas_danny_connor_chocolate_each := 2
def lucas_danny_connor := 3
def olivia_vanilla := 1
def olivia_strawberry := 1
def shannon_vanilla := 2 * olivia_vanilla
def shannon_strawberry := 2 * olivia_strawberry

-- Definitions for total scoops taken
def total_vanilla_taken := ethan_vanilla + olivia_vanilla + shannon_vanilla
def total_chocolate_taken := ethan_chocolate + (lucas_danny_connor_chocolate_each * lucas_danny_connor)
def total_strawberry_taken := olivia_strawberry + shannon_strawberry
def total_scoops_taken := total_vanilla_taken + total_chocolate_taken + total_strawberry_taken

-- Definitions for remaining scoops and original total scoops
def remaining_scoops := 16
def original_scoops := total_scoops_taken + remaining_scoops

-- Definition for number of cartons
def total_cartons := 3

-- Proof goal: scoops per carton
theorem scoops_per_carton : original_scoops / total_cartons = 10 := 
by
  -- Add your proof steps here
  sorry

end scoops_per_carton_l12_12100


namespace union_M_N_eq_N_l12_12496

def M : Set ℝ := {x : ℝ | x^2 - x < 0}
def N : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}

theorem union_M_N_eq_N : M ∪ N = N := by
  sorry

end union_M_N_eq_N_l12_12496


namespace negation_proposition_l12_12771

open Classical

variable (x : ℝ)

def proposition (x : ℝ) : Prop := ∀ x > 1, Real.log x / Real.log 2 > 0

theorem negation_proposition (h : ¬ proposition x) : 
  ∃ x > 1, Real.log x / Real.log 2 ≤ 0 := by
  sorry

end negation_proposition_l12_12771


namespace expression_undefined_at_13_l12_12692

theorem expression_undefined_at_13 :
  ∃ x : ℕ, x^2 - 26 * x + 169 = 0 ∧ x = 13 :=
by
  existsi 13
  split
  { calc
      13^2 - 26 * 13 + 169 = 169 - 338 + 169 : by norm_num
                         ... = 0 : by norm_num }
  { refl }

end expression_undefined_at_13_l12_12692


namespace find_number_l12_12072

theorem find_number (x : ℤ) : (((4 * x) - 28) / 7 + 12 = 36) → x = 49 :=
begin
  sorry
end

end find_number_l12_12072


namespace fox_can_determine_ages_l12_12561

theorem fox_can_determine_ages (ages : Finset ℕ) (fox_age : ℕ) 
  (product_condition : ∏ i in ages, i = 2450)
  (sum_condition : ∑ i in ages, i = 2 * fox_age) 
  (youngest_is_unique : ∀ a b c, a ∈ ages → b ∈ ages → c ∈ ages → a ≠ b ∧ b ≠ c ∧ c ≠ a → (a ≤ b ∧ a ≤ c) → a = 5): 
  ages = {5, 10, 49} :=
by
  sorry

end fox_can_determine_ages_l12_12561


namespace monotonicity_of_f_tangent_intersection_l12_12290

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12290


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12346

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12346


namespace total_pokemon_cards_l12_12863

-- Definitions based on the problem statement

def dozen_to_cards (dozen : ℝ) : ℝ :=
  dozen * 12

def melanie_cards : ℝ :=
  dozen_to_cards 7.5

def benny_cards : ℝ :=
  dozen_to_cards 9

def sandy_cards : ℝ :=
  dozen_to_cards 5.2

def jessica_cards : ℝ :=
  dozen_to_cards 12.8

def total_cards : ℝ :=
  melanie_cards + benny_cards + sandy_cards + jessica_cards

theorem total_pokemon_cards : total_cards = 414 := 
  by sorry

end total_pokemon_cards_l12_12863


namespace parallelogram_with_equal_diagonals_is_rectangle_l12_12074

-- Define what it means for a quadrilateral to be a parallelogram.
def is_parallelogram (A B C D : Type*) [AddGroup A] (a b c d : A) : Prop :=
  (A + D = B + C) ∧ (A + B = C + D)

-- Define what it means for a quadrilateral to be a rectangle.
def is_rectangle (A B C D : Type*) [AddGroup A] (a b c d : A) : Prop :=
  (is_parallelogram A B C D a b c d) ∧ (abs (A - C) = abs (B - D))

-- Theorem to prove: A parallelogram with equal diagonals is a rectangle.
theorem parallelogram_with_equal_diagonals_is_rectangle (A B C D : Type*) [AddGroup A] (a b c d : A) :
  is_parallelogram A B C D a b c d ∧ abs (A - C) = abs (B - D) → is_rectangle A B C D a b c d :=
by
  sorry

end parallelogram_with_equal_diagonals_is_rectangle_l12_12074


namespace monotonicity_tangent_points_l12_12280

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12280


namespace meaningful_fraction_implies_neq_neg4_l12_12054

theorem meaningful_fraction_implies_neq_neg4 (x : ℝ) : (x + 4 ≠ 0) ↔ (x ≠ -4) := 
by
  sorry

end meaningful_fraction_implies_neq_neg4_l12_12054


namespace inner_diameter_is_correct_l12_12585

-- Given conditions:
def mass : ℝ := 142 -- in grams
def outer_diameter : ℝ := 5 -- in cm
def density : ℝ := 7.9 -- in g/cm^3
def π : ℝ := 3.14

-- Calculate outer radius
def outer_radius : ℝ := outer_diameter / 2

-- Volume of a sphere
def volume (r : ℝ) : ℝ := (4 / 3) * π * r^3

-- Formula representing the problem
def inner_radius_eq (r : ℝ) : Prop :=
  density * (volume outer_radius - volume r) = mass

-- Inner diameter is twice the inner radius
def inner_diameter (r : ℝ) : ℝ := 2 * r

-- The problem statement:
theorem inner_diameter_is_correct (r : ℝ) (h : inner_radius_eq r) : inner_diameter r = 4.5 :=
  sorry

end inner_diameter_is_correct_l12_12585


namespace smallest_three_digit_solution_l12_12603

theorem smallest_three_digit_solution (n : ℕ) : 
  75 * n ≡ 225 [MOD 345] → 100 ≤ n ∧ n ≤ 999 → n = 118 :=
by
  intros h1 h2
  sorry

end smallest_three_digit_solution_l12_12603


namespace liquid_level_ratio_l12_12591

structure Cone :=
  (radius: ℝ)
  (initial_height: ℝ)
  (marble_radius: ℝ)

def volume (c: Cone) : ℝ := (1/3) * π * c.radius^2 * c.initial_height

def marble_volume (r: ℝ) : ℝ := (4/3) * π * r^3

noncomputable def new_height (c: Cone) : ℝ :=
  let initial_volume := volume c
  let marble_vol := marble_volume c.marble_radius
  let new_vol := initial_volume + marble_vol
  (3 * new_vol) / (π * c.radius^2)

theorem liquid_level_ratio :
  let narrow_cone := Cone.mk 4 4 * wide_cone.initial_height 1
  let wide_cone := Cone.mk 8 wide_cone.initial_height 2
  narrow_cone.initial_height = 4 * wide_cone.initial_height →
  (new_height narrow_cone - narrow_cone.initial_height) /
  (new_height wide_cone - wide_cone.initial_height) = 1 / 2 :=
by 
  intro h
  sorry

end liquid_level_ratio_l12_12591


namespace fraction_power_multiply_l12_12678

theorem fraction_power_multiply :
  ((1 : ℚ) / 3)^4 * ((1 : ℚ) / 5) = (1 / 405 : ℚ) :=
by sorry

end fraction_power_multiply_l12_12678


namespace triangles_not_necessarily_congruent_l12_12756

theorem triangles_not_necessarily_congruent 
  {α β γ : Type} [metric_space α] [metric_space β]
  (T1 T2 : triangle α) (a1 a2 b1 b2 : ℝ)
  (A1 A2 : angle β) 
  (h1 : T1.side1 = a1) (h2 : T1.side2 = b1)
  (h3 : T1.angle = A1)
  (h4 : T2.side1 = a2) (h5 : T2.side2 = b2)
  (h6 : T2.angle = A2)
  (h7 : a1 = a2) (h8 : b1 = b2) (h9 : A1 = A2) 
: ¬(T1 ≅ T2) := sorry

end triangles_not_necessarily_congruent_l12_12756


namespace determine_GH_l12_12577

-- Define a structure for a Tetrahedron with edge lengths as given conditions
structure Tetrahedron :=
  (EF FG EH FH EG GH : ℕ)

-- Instantiate the Tetrahedron with the given edge lengths
def tetrahedron_EFGH := Tetrahedron.mk 42 14 37 19 28 14

-- State the theorem
theorem determine_GH (t : Tetrahedron) (hEF : t.EF = 42) :
  t.GH = 14 :=
sorry

end determine_GH_l12_12577


namespace existence_of_ab_l12_12875

theorem existence_of_ab (n : ℕ) (hn : 0 < n) : ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ n ∣ (4 * a^2 + 9 * b^2 - 1) :=
by 
  sorry

end existence_of_ab_l12_12875


namespace q_div_p_eq_225_l12_12170

-- Definitions
def num_cards : ℕ := 50
def num_values : ℕ := 10
def cards_per_value : ℕ := 5
def draw_cards : ℕ := 5

-- Conditions
def all_five_same (p : ℚ) : Prop :=
  p = 10 / ((num_cards.choose draw_cards) : ℚ)

def four_one_diff (q : ℚ) : Prop :=
  q = 2250 / ((num_cards.choose draw_cards) : ℚ)

-- Problem Statement
theorem q_div_p_eq_225 (p q : ℚ) 
  (hp : all_five_same p) 
  (hq : four_one_diff q) : 
  q / p = 225 :=
sorry

end q_div_p_eq_225_l12_12170


namespace triangle_ratio_l12_12470

theorem triangle_ratio 
  (A B C D E T F : Type) 
  (a b c d e t f : Point)
  (AD DB AE EC : ℝ)
  (AD_eq : AD = 2) 
  (DB_eq : DB = 2) 
  (AE_eq : AE = 3) 
  (EC_eq : EC = 3) 
  (AD_on_AB : on_line A B D)
  (AE_on_AC : on_line A C E)
  (AT_bisects_DE : bisects A T F D E) :
  (dist A F) / (dist A T) = 3 / 5 := 
sorry

end triangle_ratio_l12_12470


namespace monotonicity_and_tangent_intersection_l12_12252

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12252


namespace ribbon_length_difference_l12_12022

theorem ribbon_length_difference (S : ℝ) : 
  let Seojun_ribbon := S 
  let Siwon_ribbon := S + 8.8 
  let Seojun_new := Seojun_ribbon - 4.3
  let Siwon_new := Siwon_ribbon + 4.3 
  Siwon_new - Seojun_new = 17.4 :=
by
  -- Definition of original ribbon lengths
  let Seojun_ribbon := S
  let Siwon_ribbon := S + 8.8
  -- Seojun cuts and gives 4.3 meters to Siwon
  let Seojun_new := Seojun_ribbon - 4.3
  let Siwon_new := Siwon_ribbon + 4.3
  -- Compute the difference
  have h1 : Siwon_new - Seojun_new = (S + 8.8 + 4.3) - (S - 4.3) := by sorry
  -- Prove the final answer
  have h2 : Siwon_new - Seojun_new = 17.4 := by sorry

  exact h2

end ribbon_length_difference_l12_12022


namespace n_pow_19_minus_n_pow_7_div_30_l12_12876

theorem n_pow_19_minus_n_pow_7_div_30 (n : ℕ) (hn : 0 < n) : 30 ∣ (n^19 - n^7) :=
sorry

end n_pow_19_minus_n_pow_7_div_30_l12_12876


namespace sum_of_positive_k_l12_12906

theorem sum_of_positive_k (k : ℤ) (x : ℤ) (α β : ℤ) (h_eq : x^2 - k * x - 18 = 0)
  (h_vieta_sum : α + β = k) (h_vieta_product : α * β = -18) :
  ∑ k in { k | (∃ (α β : ℤ), α + β = k ∧ α * β = -18) ∧ k > 0 }, k = 54 :=
by
  sorry

end sum_of_positive_k_l12_12906


namespace knight_minimum_moves_l12_12868

theorem knight_minimum_moves :
  let step1 := (2, 1)
  let step2 := (1, 2)
  let knight_moves := [(step1, step2), (step1, (-step2)), ((-step1), step2), ((-step1), (-step2)),
                       (step2, step1), (step2, (-step1)), ((-step2), step1), ((-step2), (-step1))]
  
  ∃ sequence : List (Int × Int), 
    List.foldl (fun (pos : Int × Int) (move : Int × Int) =>
                 (pos.fst + move.fst, pos.snd + move.snd)) (0, 0) sequence = (1991, 1991) ∧
    List.length sequence = 1328 :=
sorry

end knight_minimum_moves_l12_12868


namespace range_of_m_l12_12514

theorem range_of_m (a : ℝ) (h : a ≠ 0) (x1 x2 y1 y2 : ℝ) (m : ℝ)
  (hx1 : -2 < x1 ∧ x1 < 0) (hx2 : m < x2 ∧ x2 < m + 1)
  (h_on_parabola_A : y1 = a * x1^2 - 2 * a * x1 - 3)
  (h_on_parabola_B : y2 = a * x2^2 - 2 * a * x2 - 3)
  (h_diff_y : y1 ≠ y2) :
  (0 < m ∧ m ≤ 1) ∨ m ≥ 4 :=
sorry

end range_of_m_l12_12514


namespace triangle_is_isosceles_l12_12102

-- Define the conditions of the problem
variables {A B C M : Point}
variable (circle : Circle A B)
variable [finite_measure_oriented_flat_eq_triangulated EuclideanPlane Point]

-- Assume the conditions
def midpoint_condition : Prop := midpoint A C M
def circle_condition : Prop := circle.contains M

-- Main theorem statement
theorem triangle_is_isosceles (h1 : midpoint_condition A C M)
  (h2 : circle_condition circle M) :
  isosceles_triangle A B C :=
sorry

end triangle_is_isosceles_l12_12102


namespace slope_angle_range_l12_12744

theorem slope_angle_range (α : Real) (x : Real) (h₁ : x ≠ 0) (h₂ : Real.tan α = (1 / 2) * (x + 1 / x)) :
  ([Real.pi / 4, Real.pi / 2) ∪ (Real.pi / 2, 3 * Real.pi / 4]) :=
sorry

end slope_angle_range_l12_12744


namespace chemistry_marks_l12_12683

theorem chemistry_marks (marks_english : ℕ) (marks_math : ℕ) (marks_physics : ℕ) 
                        (marks_biology : ℕ) (average_marks : ℚ) (marks_chemistry : ℕ) 
                        (h_english : marks_english = 70) 
                        (h_math : marks_math = 60) 
                        (h_physics : marks_physics = 78) 
                        (h_biology : marks_biology = 65) 
                        (h_average : average_marks = 66.6) 
                        (h_total: average_marks * 5 = marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) : 
  marks_chemistry = 60 :=
by sorry

end chemistry_marks_l12_12683


namespace monotonicity_and_tangent_intersections_l12_12357

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12357


namespace range_of_x_for_f_sum_leq_2_l12_12226

open Function

noncomputable def f : ℝ → ℝ := sorry

axiom f_monotone : StrictMono f
axiom f_func_eqn : ∀ x y, f x + f y = f (x * y)
axiom f_three : f 3 = 1

theorem range_of_x_for_f_sum_leq_2 :
  ∀ x, 0 < x → 8 < x → f x + f (x - 8) ≤ 2 ↔ x ≤ 9 :=
begin
  sorry
end

end range_of_x_for_f_sum_leq_2_l12_12226


namespace prove_conditions_l12_12233

variables {f : ℤ → ℤ} {g : ℤ → ℤ} {a λ : ℤ}

-- Given conditions for the function f
def satisfies_f (f : ℤ → ℤ) (a : ℤ) : Prop :=
  (∀ x, f(x + 1) = x + 3 * a) ∧ (f a = 3)

-- Given conditions for the function g and monotonicity on (0,2)
def satisfies_g (g : ℤ → ℤ) (λ : ℤ) : Prop :=
  (λ < 0) ∧ (∀ x, 0 < x → x < 2 → g x = x * f x + λ * f x + 1) ∧
    (∀ x, 0 < x → x < 2 → (g x > g (x - 1) ∨ g x < g (x + 1)))

-- Main theorem to prove conditions imply the solutions
theorem prove_conditions (f : ℤ → ℤ) (g : ℤ → ℤ) (a λ : ℤ) :
  (satisfies_f f a) →
  (satisfies_g g λ) →
  (∀ x, f x = x + 2) ∧ (λ ≤ -6 ∨ (-2 ≤ λ ∧ λ < 0)) :=
by
  sorry

end prove_conditions_l12_12233


namespace max_a_value_l12_12407

theorem max_a_value (a : ℝ) (h : ∀ x : ℝ, |x - a| + |x - 3| ≥ 2 * a) : a ≤ 1 :=
sorry

end max_a_value_l12_12407


namespace intervals_of_monotonicity_arithmetic_mean_geometric_mean_l12_12859

theorem intervals_of_monotonicity (x : ℝ) (h1 : x > 0) :
  (0 < x ∧ x < 1 / Real.exp 1 → (x.ln + 1 < 0)) ∧
  (x > 1 / Real.exp 1 → x.ln + 1 > 0) := 
sorry

theorem arithmetic_mean_geometric_mean (a b: ℝ) (h1: a > 0) (h2: b > 0) :
  (f a + f b) / 2 > f ((a + b) / 2) := 
let f (x : ℝ) := x * Real.log x in 
sorry

end intervals_of_monotonicity_arithmetic_mean_geometric_mean_l12_12859


namespace total_cost_with_tax_l12_12016

/-- 
The statement of the math proof problem based on the given problem and solution:
Given the conditions:
- price per card from the first box = $1.25
- price per card from the second box = $1.75
- quantity of cards bought from the first box = 8
- quantity of cards bought from the second box = 12
- sales tax rate = 7%

Prove that the total cost including tax is $33.17.
-/
theorem total_cost_with_tax 
  (price_first_box : ℝ) (price_second_box : ℝ)
  (qty_first_box : ℕ) (qty_second_box : ℕ)
  (tax_rate : ℝ) :
  price_first_box = 1.25 ∧
  price_second_box = 1.75 ∧
  qty_first_box = 8 ∧
  qty_second_box = 12 ∧
  tax_rate = 0.07 →
  let cost_first_box := price_first_box * qty_first_box in
  let cost_second_box := price_second_box * qty_second_box in
  let subtotal := cost_first_box + cost_second_box in
  let total_cost := subtotal + (subtotal * tax_rate) in
  total_cost = 33.17 :=
by sorry

end total_cost_with_tax_l12_12016


namespace sale_price_of_lawn_chair_l12_12095

def original_price : ℝ := 72.95
def discount_percentage : ℝ := 0.1782

theorem sale_price_of_lawn_chair : original_price - (original_price * discount_percentage) ≈ 59.95 :=
by  sorry

end sale_price_of_lawn_chair_l12_12095


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12393

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12393


namespace find_incorrect_number_l12_12540

theorem find_incorrect_number (X : ℕ) : 
  (460 - X + 75 = 510) → 
  X = 25 := by
  assume h : 460 - X + 75 = 510,
  sorry

end find_incorrect_number_l12_12540


namespace intersection_complement_M_N_l12_12743

def M : Set ℝ := { x | x > 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }
def complement_M : Set ℝ := { x | x ≤ 1 }

theorem intersection_complement_M_N :
  (complement_M ∩ N) = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_complement_M_N_l12_12743


namespace sum_of_coefficients_correct_l12_12722

-- Define the polynomial
def polynomial (x y : ℤ) : ℤ := (x + 3 * y) ^ 17

-- Define the sum of coefficients by substituting x = 1 and y = 1
def sum_of_coefficients : ℤ := polynomial 1 1

-- Statement of the mathematical proof problem
theorem sum_of_coefficients_correct :
  sum_of_coefficients = 17179869184 :=
by
  -- proof will be provided here
  sorry

end sum_of_coefficients_correct_l12_12722


namespace conditions_implies_equilateral_l12_12823

variable (ABC : Type) [Triangle ABC]
variable {A B C : Point ABC → Point ABC → Point ABC}

def equilateral (ABC : Triangle) : Prop :=
  side ABC A B = side ABC B C ∧ side ABC B C = side ABC C A

def equal_angles (ABC : Triangle) : Prop :=
  angle ABC A = angle ABC B ∧ angle ABC B = angle ABC C

def two_angles_60 (ABC : Triangle) : Prop :=
  (angle ABC A = 60 ∧ angle ABC B = 60) ∨ (angle ABC B = 60 ∧ angle ABC C = 60) ∨ (angle ABC A = 60 ∧ angle ABC C = 60)

def isosceles_with_one_60 (ABC : Triangle) : Prop :=
  (side ABC A B = side ABC B C ∧ angle ABC A = 60) ∨ (side ABC B C = side ABC C A ∧ angle ABC B = 60) ∨ (side ABC C A = side ABC A B ∧ angle ABC C = 60)

theorem conditions_implies_equilateral :
  (side ABC A B = side ABC B C ∧ side ABC B C = side ABC C A ∨
  angle ABC A = angle ABC B ∧ angle ABC B = angle ABC C ∨
  (angle ABC A = 60 ∧ angle ABC B = 60) ∨ (angle ABC B = 60 ∧ angle ABC C = 60) ∨
  (side ABC A B = side ABC B C ∧ angle ABC A = 60) ∨ (side ABC B C = side ABC C A ∧ angle ABC B = 60) ∨ (side ABC C A = side ABC A B ∧ angle ABC C = 60))
  → (equilateral ABC ∧ equal_angles ABC ∧ two_angles_60 ABC ∧ isosceles_with_one_60 ABC) :=
by sorry

end conditions_implies_equilateral_l12_12823


namespace yankees_mets_ratio_l12_12799

-- Given conditions
def num_mets_fans : ℕ := 104
def total_fans : ℕ := 390
def ratio_mets_to_redsox : ℚ := 4 / 5

-- Definitions
def num_redsox_fans (M : ℕ) := (5 / 4) * M
def num_yankees_fans (Y M B : ℕ) := (total_fans - M - B)

-- Theorem statement
theorem yankees_mets_ratio (Y M B : ℕ)
  (h1 : M = num_mets_fans)
  (h2 : Y + M + B = total_fans)
  (h3 : (M : ℚ) / (B : ℚ) = ratio_mets_to_redsox) :
  (Y : ℚ) / (M : ℚ) = 3 / 2 :=
sorry

end yankees_mets_ratio_l12_12799


namespace red_pair_expectation_l12_12017

theorem red_pair_expectation (deck : Finite 40) (cards : deck.finset.card → bool) (circle : (Finset (Fin 40) → Prop )) :
  (∀ i, cards (⟨i, sorry⟩) = tt ∨ cards (⟨i, sorry⟩) = ff) → -- Every card is either red (true) or black (false)
  (Finset.card (deck.finset.filter (λ i, cards (⟨i, sorry⟩))) = 20) → -- 20 red cards in the deck
  (Finset.card (deck.finset.filter (λ i, ¬ cards (⟨i, sorry⟩))) = 20) → -- 20 black cards in the deck
  (expected_value (λ i, if cards i ∧ cards (circle.next i) then 1 else 0) = 380 / 39) := sorry

end red_pair_expectation_l12_12017


namespace number_of_incorrect_propositions_is_3_l12_12240

-- Definitions of the propositions as given in the problem
def proposition1 :=
  ∀ (f : ℝ → ℝ) (x₀ : ℝ), (f' x₀ = 0 → x₀ is_extremum f)

def proposition2 (a b : ℝ) :=
  (angle_between a b > π/2) → (a ⋅ b < 0)

def proposition3 :=
  ∀ (x : ℝ), (1/(x - 1) > 0 → 1/(x - 1) ≤ 0)

def proposition4 :=
  ∀ (x : ℝ), (¬∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)

-- The theorem that states the number of incorrect propositions is 3
theorem number_of_incorrect_propositions_is_3 :
  (¬ proposition1) ∧ (¬ proposition2) ∧ (¬ proposition3) ∧ proposition4 ∧ 3 = 3 :=
by sorry

end number_of_incorrect_propositions_is_3_l12_12240


namespace hyperbola_foci_x_axis_range_l12_12793

theorem hyperbola_foci_x_axis_range (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (m + 2)) - (y^2 / (m - 1)) = 1) →
  (1 < m) ↔ 
  (∀ x y : ℝ, (m + 2 > 0) ∧ (m - 1 > 0)) :=
sorry

end hyperbola_foci_x_axis_range_l12_12793


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12320

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12320


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12298

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12298


namespace average_age_of_team_l12_12543

theorem average_age_of_team (A : ℝ) : 
    (11 * A =
         9 * (A - 1) + 53) → 
    A = 31 := 
by 
  sorry

end average_age_of_team_l12_12543


namespace transform_function_l12_12553

def original_function (x : ℝ) : ℝ := 2^(x - 1) + 3

def transformed_function (x : ℝ) : ℝ := 2^x + 1

theorem transform_function :
  ∀ (x y : ℝ), y = original_function (x - 1) - 2 → y = transformed_function x :=
by
  intro x y
  intro h
  sorry

end transform_function_l12_12553


namespace problem1_problem2_l12_12725

theorem problem1 (a b : ℝ) (h : a ≠ b) (M : ℝ)
  (ineq : M * |a - b| ≤ |2 * a + b| + |a + 2 * b|) : M ≤ 1 := by
  sorry

theorem problem2 (x : ℝ) : |x - 1| < 2 * x + 1 ↔ 0 < x := by
  split
  · intro h
    cases lt_or_ge x 1 with
    | inl hxlt =>
      have h1 : 1 - x < 2 * x + 1 := by linarith
      linarith
    | inr hxge =>
      have h2 : x - 1 < 2 * x + 1 := by linarith
      linarith
  · intro hxpos
    cases lt_or_ge x 1 with
    | inl hxlt =>
      have h1 : 1 - x < 2 * x + 1 := by linarith
      linarith
    | inr hxge =>
      have h2 : x - 1 < 2 * x + 1 := by linarith
      linarith

end problem1_problem2_l12_12725


namespace genevieve_errors_fixed_l12_12195

theorem genevieve_errors_fixed (total_lines : ℕ) (lines_per_debug : ℕ) (errors_per_debug : ℕ)
  (h_total_lines : total_lines = 4300)
  (h_lines_per_debug : lines_per_debug = 100)
  (h_errors_per_debug : errors_per_debug = 3) :
  (total_lines / lines_per_debug) * errors_per_debug = 129 :=
by
  -- Placeholder proof to indicate the theorem should be true
  sorry

end genevieve_errors_fixed_l12_12195


namespace find_a_l12_12200

theorem find_a
  (a : ℝ)
  (ha : a > 0)
  (hcoef : let expr := (ax - 1)^4 * (x + 2) in expr.coeff x^2 = 1) :
  a = (2 + Real.sqrt 10) / 6 := by
  sorry

end find_a_l12_12200


namespace equal_scores_l12_12806

noncomputable def archery_competition 
  (score : Fin 30 → ℕ) 
  (arrows_hit_zone1 : Fin 30 → ℕ) 
  (arrows_hit_zone2 : Fin 30 → ℕ) 
  (arrows_missed : Fin 30 → ℕ) : Prop :=
  (∀ i : Fin 30, arrows_hit_zone1 i + arrows_hit_zone2 i + arrows_missed i = 16) ∧
  (∀ i : Fin 30, arrows_hit_zone1 i = arrows_missed i) ∧
  (∑ i : Fin 30, arrows_hit_zone2 i > 240) ∧
  (∀ i : Fin 30, score i = 2 * arrows_hit_zone1 i + arrows_hit_zone2 i) ∧
  (∃ i j : Fin 30, i ≠ j ∧ score i = score j)

theorem equal_scores 
  (score : Fin 30 → ℕ) 
  (arrows_hit_zone1 : Fin 30 → ℕ) 
  (arrows_hit_zone2 : Fin 30 → ℕ) 
  (arrows_missed : Fin 30 → ℕ) :
  archery_competition score arrows_hit_zone1 arrows_hit_zone2 arrows_missed :=
begin
  sorry
end

end equal_scores_l12_12806


namespace sum_of_super_cool_triangle_areas_l12_12651

theorem sum_of_super_cool_triangle_areas :
  let is_super_cool_triangle (a b : ℕ) : Prop := (a * b) / 2 = a + b
  ∃ (areas : Finset ℕ), 
    (∀ a b : ℕ, is_super_cool_triangle a b → ((a-2) * (b-2) = 4) → (a = 3 ∧ b = 6 ∨ a = 6 ∧ b = 3 ∨ a = 4 ∧ b = 4) ∧ areas = {9, 8})
    ∧ areas.sum = 17 :=
by
  let is_super_cool_triangle (a b : ℕ) : Prop := (a * b) / 2 = a + b
  let areas := {9, 8}
  have h_conditions : ∀ a b : ℕ, is_super_cool_triangle a b → ((a-2) * (b-2) = 4) → (a = 3 ∧ b = 6 ∨ a = 6 ∧ b = 3 ∨ a = 4 ∧ b = 4)
  sorry
  have h_sum : areas.sum = 17
  sorry
  exact ⟨areas, h_conditions, h_sum⟩

end sum_of_super_cool_triangle_areas_l12_12651


namespace alpha_eq_gamma_or_tau_l12_12855

theorem alpha_eq_gamma_or_tau 
  (α β γ τ : ℝ) 
  (hα : 0 < α) 
  (hβ : 0 < β) 
  (hγ : 0 < γ) 
  (hτ : 0 < τ) 
  (h : ∀ x : ℝ, sin (α * x) + sin (β * x) = sin (γ * x) + sin (τ * x)) :
  α = γ ∨ α = τ := 
by
  sorry

end alpha_eq_gamma_or_tau_l12_12855


namespace share_of_a_l12_12082

-- Define the initial investments, withdrawals, advances, and total profit.
def initial_investment_a : ℕ := 3000
def initial_investment_b : ℕ := 4000
def withdrawal_a : ℕ := 1000
def advance_b : ℕ := 1000
def total_profit : ℕ := 756

-- Calculate the share of A's profit given the conditions.
theorem share_of_a (share_a : ℕ) :
  (let investment_months_a := (initial_investment_a * 8) + ((initial_investment_a - withdrawal_a) * 4),
       investment_months_b := (initial_investment_b * 8) + ((initial_investment_b + advance_b) * 4),
       ratio_a_b := (investment_months_a, investment_months_b),
       total_ratio_parts := ratio_a_b.1 + ratio_a_b.2,
       value_per_part := total_profit / total_ratio_parts in
    share_a = value_per_part * ratio_a_b.1) → share_a = 288 :=
sorry

end share_of_a_l12_12082


namespace monotonicity_and_tangent_intersections_l12_12358

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12358


namespace monotonicity_tangent_intersection_points_l12_12373

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12373


namespace watched_movies_count_l12_12929

theorem watched_movies_count {M : ℕ} (total_books total_movies read_books : ℕ) 
  (h1 : total_books = 15) (h2 : total_movies = 14) (h3 : read_books = 11) 
  (h4 : read_books = M + 1) : M = 10 :=
by
  sorry

end watched_movies_count_l12_12929


namespace domain_of_ln_l12_12550

-- We define the function f and state the condition for it to be defined
def f (x : ℝ) : ℝ := Real.log (x - 3)

-- The domain of the function f is the set of x for which the argument of the log is positive
def is_defined (x : ℝ) : Prop := x > 3

-- Now we state the theorem which specifies the domain of f
theorem domain_of_ln : {x : ℝ | is_defined x} = {x : ℝ | 3 < x} :=
by
  sorry

end domain_of_ln_l12_12550


namespace roots_negative_reciprocal_l12_12737

theorem roots_negative_reciprocal (a b c : ℝ) (α β : ℝ) (h_eq : a * α ^ 2 + b * α + c = 0)
  (h_roots : α * β = -1) : c = -a :=
sorry

end roots_negative_reciprocal_l12_12737


namespace gcd_and_operations_l12_12947

-- Define the numbers used
def a : ℤ := 56
def b : ℤ := 264

-- Define the Euclidean algorithm
def euclidean_algorithm (m n : ℤ) : ℕ := 
  if n = 0 then 0 else 1 + euclidean_algorithm n (m % n)

-- Calculate GCD using Euclidean algorithm (We skip the actual proof here)
def gcd (m n : ℤ) : ℤ := 
  if n = 0 then m else gcd n (m % n)

-- Now state the theorem
theorem gcd_and_operations (gcd(a, b) = 8 ∧ euclidean_algorithm(a, b) = 4) : ∀ a b, gcd(a, b) = 8 ∧ euclidean_algorithm(a, b) = 4 := by
  sorry

end gcd_and_operations_l12_12947


namespace linear_inequality_solution_l12_12202

theorem linear_inequality_solution {x y m n : ℤ} 
  (h_table: (∀ x, if x = -2 then y = 3 
                else if x = -1 then y = 2 
                else if x = 0 then y = 1 
                else if x = 1 then y = 0 
                else if x = 2 then y = -1 
                else if x = 3 then y = -2 
                else true)) 
  (h_eq: m * x - n = y) : 
  x ≥ -1 :=
sorry

end linear_inequality_solution_l12_12202


namespace find_alpha_l12_12840

noncomputable def α_and_β_are_complex_numbers (α β : ℂ) : Prop := true
noncomputable def α_plus_β_is_positive_real (α β : ℂ) : Prop := ∃ r : ℝ, r > 0 ∧ (α + β) = r
noncomputable def i_α_minus_3β_is_positive_real (α β : ℂ) : Prop := ∃ r : ℝ, r > 0 ∧ (i * (α - 3 * β)) = r
def beta_value : ℂ := 4 + 3 * complex.i

theorem find_alpha (α : ℂ) (β : ℂ := beta_value) 
  (h1 : α_and_β_are_complex_numbers α β) 
  (h2 : α_plus_β_is_positive_real α β)
  (h3 : i_α_minus_3β_is_positive_real α β) 
  (h4 : β = beta_value) : α = 3 - 3 * complex.i := 
by
  sorry

end find_alpha_l12_12840


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12299

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12299


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12343

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12343


namespace find_k_l12_12419

theorem find_k {k : ℚ} (h : (3 : ℚ)^3 + 7 * (3 : ℚ)^2 + k * (3 : ℚ) + 23 = 0) : k = -113 / 3 :=
by
  sorry

end find_k_l12_12419


namespace max_programs_l12_12927

theorem max_programs {P  : Fin 12 → Set (Fin 6)} (H₁ : ∀ (i : Fin 6), P i ≠ ∅)
  (H₂ : ∀ i j, i ≠ j → (P i ∩ P j).card ≤ 2) : ∃ (n : ℕ), n ≤ 4 :=
sorry

end max_programs_l12_12927


namespace monotonicity_of_f_tangent_intersection_points_l12_12324

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12324


namespace smallest_degree_polynomial_l12_12006

noncomputable def polynomial_with_roots (a b c d : ℚ) : ℚ[X] :=
  (X - (a - real.sqrt b)) * (X - (a + real.sqrt b)) *
  (X - (-a - real.sqrt b)) * (X - (-a + real.sqrt b)) *
  (X - (c + real.sqrt d)) * (X - (c - real.sqrt d))

theorem smallest_degree_polynomial :
  ∃ p : ℚ[X], 
    (polynomial_with_roots 2 5 3 11) = p ∧
    p.degree = 6 :=
by
  sorry

end smallest_degree_polynomial_l12_12006


namespace minimize_MN_length_l12_12870

-- Definitions for the conditions
variables (A B C X M N : Point) (hypotenuse : Line) (right_triangle: RightTriangle ABC)

-- Conditions specified in the problem
def X_on_hypotenuse : Prop := X ∈ hypotenuse
def M_projection_AC : Prop := Proj M X AC
def N_projection_BC : Prop := Proj N X BC
def minimized_length_MN (X : Point) (MN : Segment) : Prop := MN.length = MHypotenuse X

-- Statement for Part a)
theorem minimize_MN_length : 
  ∀ (X : Point), 
    X_on_hypotenuse X → 
    M_projection_AC M X AC → 
    N_projection_BC N X BC → 
    minimized_length_MN MN X :=
sorry

end minimize_MN_length_l12_12870


namespace find_triangle_base_l12_12039

theorem find_triangle_base (left_side : ℝ) (right_side : ℝ) (base : ℝ) 
  (h_left : left_side = 12) 
  (h_right : right_side = left_side + 2)
  (h_sum : left_side + right_side + base = 50) :
  base = 24 := 
sorry

end find_triangle_base_l12_12039


namespace find_savings_l12_12085

-- Definitions and conditions from the problem
def income : ℕ := 36000
def ratio_income_to_expenditure : ℚ := 9 / 8
def expenditure : ℚ := 36000 * (8 / 9)
def savings : ℚ := income - expenditure

-- The theorem statement to prove
theorem find_savings : savings = 4000 := by
  sorry

end find_savings_l12_12085


namespace clock_hand_swap_times_l12_12604

noncomputable def time_between_2_and_3 : ℚ := (2 * 143 + 370) / 143
noncomputable def time_between_6_and_7 : ℚ := (6 * 143 + 84) / 143

theorem clock_hand_swap_times :
  time_between_2_and_3 = 2 + 31 * 7 / 143 ∧
  time_between_6_and_7 = 6 + 12 * 84 / 143 :=
by
  -- Math proof will go here
  sorry

end clock_hand_swap_times_l12_12604


namespace parallelogram_count_l12_12803

theorem parallelogram_count (m n : ℕ) : 
  ∃ p : ℕ, p = (m.choose 2) * (n.choose 2) :=
by
  sorry

end parallelogram_count_l12_12803


namespace slope_angle_of_line_l12_12920

noncomputable def line_equation (x y : ℝ) (a : ℝ) : Prop :=
  sqrt 3 * x + 3 * y + a = 0

theorem slope_angle_of_line (a : ℝ) : ∃ (α : ℝ), (0 <= α ∧ α < 180) ∧ α = 150 :=
begin
  use 150,
  split,
  { split,
    { norm_num,
      linarith, },
    { norm_num,
      linarith, }, },
  sorry,
end

end slope_angle_of_line_l12_12920


namespace can_write_one_l12_12027

theorem can_write_one (a b : ℝ) (h_a : a = Real.sqrt 2) (h_b : b = Real.sqrt 5) : 
  ∃ c ∈ set.univ, c = 1 := 
by
  sorry

end can_write_one_l12_12027


namespace triangles_congruent_l12_12754

-- Definitions
structure Triangle (α : Type) :=
(A B C : α)

structure Congruent (α : Type) (t1 t2 : Triangle α) :=
(side1 : α) (side2 : α) (angle : α)
(equal_side1 : t1.A = t2.A ∧ t1.B = t2.B)
(equal_side2 : t1.B = t2.B ∧ t1.C = t2.C)
(equal_angle : ∠(t1.A t1.B t1.C) = ∠(t2.A t2.B t2.C))

-- Conditions
variables {α : Type} [Nonempty α]
variables (t1 t2 : Triangle α)
variables (side1 side2 : α) (angle : α)

-- Problem statement
theorem triangles_congruent (h1 : t1.A = t2.A) (h2 : t1.B = t2.B) (h3 : t1.C = t2.C) 
  (h_angle : ∠(t1.A t1.B t1.C) = ∠(t2.A t2.B t2.C)) :
  Congruent α t1 t2 :=
by
  sorry

end triangles_congruent_l12_12754


namespace difference_is_12_l12_12975

-- Define the values based on the given problem
def eighty_percent_of_40 := 0.80 * 40
def four_fifths_of_25 := (4 / 5) * 25

-- Prove that the difference between eighty_percent_of_40 and four_fifths_of_25 is 12
theorem difference_is_12 : eighty_percent_of_40 - four_fifths_of_25 = 12 :=
by
  -- Skipping the actual proof
  sorry

end difference_is_12_l12_12975


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12311

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12311


namespace monotonicity_f_tangent_points_l12_12367

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12367


namespace find_function_l12_12708

noncomputable def solution_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = x + f y → ∃ c : ℝ, ∀ x : ℝ, f x = x + c

theorem find_function (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = x + f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end find_function_l12_12708


namespace sum_integer_solutions_abs_lt_abs_sub_lt_ten_l12_12066

theorem sum_integer_solutions_abs_lt_abs_sub_lt_ten :
  (∑ n in Finset.filter (λ n : ℤ, |n| < |n - 4| ∧ |n - 4| < 10) (Finset.Ico (-6) 2), n) = -20 :=
  sorry

end sum_integer_solutions_abs_lt_abs_sub_lt_ten_l12_12066


namespace triangles_congruent_l12_12755

-- Definitions
structure Triangle (α : Type) :=
(A B C : α)

structure Congruent (α : Type) (t1 t2 : Triangle α) :=
(side1 : α) (side2 : α) (angle : α)
(equal_side1 : t1.A = t2.A ∧ t1.B = t2.B)
(equal_side2 : t1.B = t2.B ∧ t1.C = t2.C)
(equal_angle : ∠(t1.A t1.B t1.C) = ∠(t2.A t2.B t2.C))

-- Conditions
variables {α : Type} [Nonempty α]
variables (t1 t2 : Triangle α)
variables (side1 side2 : α) (angle : α)

-- Problem statement
theorem triangles_congruent (h1 : t1.A = t2.A) (h2 : t1.B = t2.B) (h3 : t1.C = t2.C) 
  (h_angle : ∠(t1.A t1.B t1.C) = ∠(t2.A t2.B t2.C)) :
  Congruent α t1 t2 :=
by
  sorry

end triangles_congruent_l12_12755


namespace monthly_bill_relationship_l12_12632

-- Conditions
def monthly_rental_fee : ℝ := 10
def cost_per_call : ℝ := 0.2
def number_of_calls (x : ℕ) : ℝ := x

-- Relationship to be proven
theorem monthly_bill_relationship (x : ℕ) : 
  let y := monthly_rental_fee + cost_per_call * number_of_calls x in
  y = 10 + 0.2 * x :=
by
  sorry

end monthly_bill_relationship_l12_12632


namespace cylinder_height_in_sphere_l12_12647

-- Definitions based on conditions
def radius_sphere : ℝ := 7
def radius_cylinder : ℝ := 3

-- Mathematical equivalent proof problem
theorem cylinder_height_in_sphere : 
  ∃ h : ℝ, (radius_sphere)^2 = (radius_cylinder)^2 + (h / 2)^2 ∧ h = 4 * real.sqrt 10 :=
begin
  sorry
end

end cylinder_height_in_sphere_l12_12647


namespace min_value_function_l12_12746

theorem min_value_function 
  (x : ℝ) (hx : x > -1) : 
  ∃ y, y = (∃ t > 0, t = x + 1 ∧ y = (t + 4 / t) + 5) ∧ y = 9 := 
sorry

end min_value_function_l12_12746


namespace EQ_EP_equal_l12_12058

-- Definitions of points and the circles
variables (G1 G2 : Circle) (M N A B C D E P Q : Point)
variable (AB CD : Line)

-- Conditions
axiom circles_intersect : ∃ M N, M ∈ G1 ∧ M ∈ G2 ∧ N ∈ G1 ∧ N ∈ G2
axiom tangent_lines : AB.tangent_to G1 A ∧ AB.tangent_to G2 B
axiom M_closer_to_AB : (distance M AB) < (distance N AB)
axiom CD_parallel_to_AB : CD ∥ AB
axiom CD_passing_through_M : M ∈ CD ∧ C ∈ G1 ∧ D ∈ G2
axiom E_definition : E = Line.intersection (line AC) (line BD)
axiom P_definition : P = Line.intersection (line AN) (CD)
axiom Q_definition : Q = Line.intersection (line BN) (CD)

-- The theorem
theorem EQ_EP_equal : EP = EQ :=
sorry

end EQ_EP_equal_l12_12058


namespace minimal_square_area_l12_12941

theorem minimal_square_area (a b c d: ℕ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 4) (h4 : d = 3) :
  ∃ (s: ℕ), (a ≤ s ∧ b ≤ s ∧ c ≤ s ∧ d ≤ s ∧ s^2 = 36) :=
begin
  -- Given rectangle dimensions
  let A := (2, 5),
  let B := (4, 3),
  -- The smallest side length of the square
  have H : 6 = max (max 5 3) (2 + 4),
  { refl },
  existsi 6,
  split,
  { exact nat.le_succ 2},
  split,
  { exact nat.le_of_lt_succ (lt_add_of_pos_left 5 nat.succ_pos')},
  split,
  { refl},
  split,
  { exact nat.le_of_lt_succ (lt_add_of_pos_left 4 nat.succ_pos')},
  { exact pow_two },
end

end minimal_square_area_l12_12941


namespace stratified_sampling_female_athletes_l12_12125

-- Conditions
def total_male_athletes : ℕ := 56
def total_female_athletes : ℕ := 42
def total_sample_size : ℕ := 28

-- Proportion of female athletes in stratified sampling
def proportion_female_athletes : ℚ := total_female_athletes / (total_male_athletes + total_female_athletes)
def expected_female_athletes_in_sample : ℕ := (proportion_female_athletes * total_sample_size).natAbs

-- The proof statement
theorem stratified_sampling_female_athletes :
  expected_female_athletes_in_sample = 12 :=
sorry

end stratified_sampling_female_athletes_l12_12125


namespace Marco_strawberries_weigh_l12_12503

/-- 
Marco and his dad went strawberry picking and collected 36 pounds of strawberries.
On the way back, Marco's dad lost 8 pounds of strawberries. Now, his dad's strawberries 
weighed 16 pounds. Prove that Marco's strawberries weighed 12 pounds.
-/
theorem Marco_strawberries_weigh :
  ∀ (total_weight dad_lost dad_now : ℕ), total_weight = 36 ∧ dad_lost = 8 ∧ dad_now = 16 →
  ∃ (marco_strawberries : ℕ), marco_strawberries = 12 :=
by
  intros total_weight dad_lost dad_now h,
  obtain ⟨ht, hd, hn⟩ := h,
  let dad_original := dad_now + dad_lost,
  have htotal_dad : dad_original = 24 := by { rw [hd, hn], rfl },
  let marco_strawberries := total_weight - dad_original,
  have hmarco : marco_strawberries = 12 := by { rw [htotal_dad, ht], exact rfl },
  use marco_strawberries,
  exact hmarco

end Marco_strawberries_weigh_l12_12503


namespace distance_to_origin_l12_12549

theorem distance_to_origin (a : ℝ) (h: |a| = 5) : 3 - a = -2 ∨ 3 - a = 8 :=
sorry

end distance_to_origin_l12_12549


namespace product_of_symmetric_complex_l12_12220

-- Definitions based on the conditions.
def z1 : ℂ := 2 + complex.I
def z2 : ℂ := -2 + complex.I

-- Main statement to prove.
theorem product_of_symmetric_complex (z1 z2 : ℂ) (h1 : z1 = 2 + complex.I) (h2 : z2 = -2 + complex.I) : 
  z1 * z2 = -5 :=
by
  rw [h1, h2]
  sorry

end product_of_symmetric_complex_l12_12220


namespace monotonicity_of_f_tangent_intersection_l12_12289

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12289


namespace min_queries_find_white_balls_l12_12583

theorem min_queries_find_white_balls :
  ∀ (boxes : Fin 15 → bool) (W : Nat),
  (∀ i, boxes i = tt) ∨ (∃ i, boxes i = tt)  ∧ countp boxes (λ b, b = tt) ≤ 12 →
  ∃ n, n = 91 :=
by
  sorry

end min_queries_find_white_balls_l12_12583


namespace solution_set_leq_2_l12_12204

theorem solution_set_leq_2 (x y m n : ℤ)
  (h1 : m * 0 - n = 1)
  (h2 : m * 1 - n = 0)
  (h3 : y = m * x - n) :
  x ≥ -1 ↔ m * x - n ≤ 2 :=
by {
  sorry
}

end solution_set_leq_2_l12_12204


namespace new_number_formed_l12_12792

theorem new_number_formed (h t u : ℕ) (Hh : h < 10) (Ht : t < 10) (Hu : u < 10) :
  let original_number := 100 * h + 10 * t + u
  let new_number := 2000 + 10 * original_number
  new_number = 1000 * (h + 2) + 100 * t + 10 * u :=
by
  -- Proof would go here
  sorry

end new_number_formed_l12_12792


namespace min_employees_needed_l12_12106

/-- A company's employees work 5 days and rest 2 days each week. 
    The company requires at least 32 employees to be on duty every day from Monday to Sunday.
    Prove that the company needs at least 45 employees. -/
theorem min_employees_needed (employee_work_days_per_week : ℕ) (employee_rest_days_per_week : ℕ) (min_employees_on_duty_per_day : ℕ)
  (days_per_week : ℕ) (total_employee_days_needed_per_week : ℕ) (weekly_contribution_per_employee : ℕ)
  (n_employees : ℕ) : 
  employee_work_days_per_week = 5 → 
  employee_rest_days_per_week = 2 →
  min_employees_on_duty_per_day = 32 →
  days_per_week = 7 →
  total_employee_days_needed_per_week = min_employees_on_duty_per_day * days_per_week →
  weekly_contribution_per_employee = employee_work_days_per_week →
  n_employees = ((total_employee_days_needed_per_week : ℚ) / weekly_contribution_per_employee).ceil.to_nat →
  n_employees = 45 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  -- start proof here
  sorry

end min_employees_needed_l12_12106


namespace trailing_zeros_in_100_factorial_l12_12049

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the function to count the factors of p in n!
def count_factors (p n : ℕ) : ℕ :=
  if p ≤ 1 then 0
  else match n with
       | 0 => 0
       | (n + 1) => (n + 1) / p + count_factors p (n / p)

-- Define the function to count trailing zeros in n!
def trailing_zeros (n : ℕ) : ℕ :=
  min (count_factors 2 n) (count_factors 5 n)

-- The theorem we want to prove, stating that the number of trailing zeros in 100! is 24
theorem trailing_zeros_in_100_factorial : trailing_zeros 100 = 24 :=
  sorry

end trailing_zeros_in_100_factorial_l12_12049


namespace dot_product_BA_BC_eq_4_l12_12811

-- Define the problem conditions
variables (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
variable [right_triangle ABC] -- Assuming there is a way to define a right_triangle

-- Defining the sides of the triangle
def AB := (4 : ℝ)
def AC := (2 * real.sqrt 3 : ℝ)
def BC := (2 : ℝ)

-- Theorem stating the dot product result given the conditions
theorem dot_product_BA_BC_eq_4 :
    ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C] [right_triangle ABC],
      AB = 4 → AC = 2 * real.sqrt 3 → BC = 2 → 
      (BA • BC) = 4 := 
by
    sorry

end dot_product_BA_BC_eq_4_l12_12811


namespace cos_theta_eq_neg_one_l12_12420

noncomputable theory

variables (θ : ℝ) (hθ1 : 6 * tan θ = 2 * sin θ) (hθ2 : 0 < θ) (hθ3 : θ < π)

theorem cos_theta_eq_neg_one : cos θ = -1 :=
sorry

end cos_theta_eq_neg_one_l12_12420


namespace quadrilateral_area_inequality_l12_12856

theorem quadrilateral_area_inequality 
  (a b c d : ℝ) (S : ℝ)
  (h : quadrilateral_area a b c d = S) :
  S ≤ (a * c + b * d) / 2 :=
by
  sorry

end quadrilateral_area_inequality_l12_12856


namespace sqrt_of_9_is_3_l12_12896

theorem sqrt_of_9_is_3 {x : ℝ} (h₁ : x * x = 9) (h₂ : x ≥ 0) : x = 3 := sorry

end sqrt_of_9_is_3_l12_12896


namespace emily_lemon_juice_fraction_l12_12163

/-- 
Emily places 6 ounces of tea into a twelve-ounce cup and 6 ounces of honey into a second cup
of the same size. Then she adds 3 ounces of lemon juice to the second cup. Next, she pours half
the tea from the first cup into the second, mixes thoroughly, and then pours one-third of the
mixture in the second cup back into the first. 
Prove that the fraction of the mixture in the first cup that is lemon juice is 1/7.
--/
theorem emily_lemon_juice_fraction :
  let cup1_tea := 6
  let cup2_honey := 6
  let cup2_lemon_juice := 3
  let cup1_tea_transferred := cup1_tea / 2
  let cup1 := cup1_tea - cup1_tea_transferred
  let cup2 := cup2_honey + cup2_lemon_juice + cup1_tea_transferred
  let mix_ratio (x y : ℕ) := (x : ℚ) / (x + y)
  let cup1_after_transfer := cup1 + (cup2 / 3)
  let cup2_tea := cup1_tea_transferred
  let cup2_honey := cup2_honey
  let cup2_lemon_juice := cup2_lemon_juice
  let cup1_lemon_transferred := 1
  cup1_tea + (cup2 / 3) = 3 + (cup2_tea * (1 / 3)) + 1 + (cup2_honey * (1 / 3)) + cup2_lemon_juice / 3 →
  cup1 / (cup1 + cup2_honey) = 1/7 :=
sorry

end emily_lemon_juice_fraction_l12_12163


namespace triangle_cosine_max_value_l12_12468

theorem triangle_cosine_max_value (A B C : ℝ) (h : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 2) 
  (h_triangle : A + B + C = π) : 
  cos A + cos B + 2 * cos C ≤ sqrt 5 :=
sorry

end triangle_cosine_max_value_l12_12468


namespace wickets_in_last_match_l12_12116

theorem wickets_in_last_match (w : ℕ) : 
  let avg_before_last_match : ℚ := 12.4
  let runs_last_match : ℕ := 26
  let avg_decrease : ℚ := 0.4
  let wickets_before_last_match : ℕ := 115
  let total_runs_before : ℚ := wickets_before_last_match * avg_before_last_match
  let newly_avg : ℚ := avg_before_last_match - avg_decrease
  let total_wickets_after_last_match : ℕ := wickets_before_last_match + w
  let total_runs_after_last_match : ℚ := total_runs_before + runs_last_match
  in newly_avg * total_wickets_after_last_match = total_runs_after_last_match → w = 6 :=
by
  sorry

end wickets_in_last_match_l12_12116


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12394

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12394


namespace solve_log_eq_correct_l12_12724

noncomputable def solve_log_eq (a : ℝ) : set ℝ :=
  if a = 0 then ∅
  else if a > 0 then {x | ∃ n : ℤ, x = Real.arccot a + 2 * Real.pi * n}
  else {x | ∃ m : ℤ, x = Real.pi + Real.arccot a + 2 * Real.pi * m}

theorem solve_log_eq_correct (a : ℝ) :
  solve_log_eq a = 
    if a = 0 then ∅
    else if a > 0 then {x | ∃ n : ℤ, x = Real.arccot a + 2 * Real.pi * n}
    else {x | ∃ m : ℤ, x = Real.pi + Real.arccot a + 2 * Real.pi * m} := by
  sorry

end solve_log_eq_correct_l12_12724


namespace odd_function_zero_unique_l12_12704

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = - f (- x)

def functional_eq (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x + y) * f (x - y) = f x ^ 2 * f y ^ 2

theorem odd_function_zero_unique
  (h_odd : odd_function f)
  (h_func_eq : functional_eq f) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end odd_function_zero_unique_l12_12704


namespace bisection_method_root_l12_12945

def f (x : ℝ) : ℝ := x^3 - 2*x - 1

theorem bisection_method_root (h1 : 1 < 2) (h_f1 : f 1 < 0) (h_f2 : f 2 > 0) (h_f3_2: f (3/2) < 0) :
  ∃ x, (x = 1 ∨ x = 2 ∨ x = 3/2) → f x = 0 → x ∈ (Ioo (3/2) 2) :=
by
  sorry

end bisection_method_root_l12_12945


namespace relatively_prime_dates_in_September_l12_12147

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem relatively_prime_dates_in_September :
  let num_days := 30
  let month := 9
  let relatively_prime_days := {d : ℕ | 1 ≤ d ∧ d ≤ num_days ∧ relatively_prime month d}
  set.size relatively_prime_days = 20 := 
sorry

end relatively_prime_dates_in_September_l12_12147


namespace exists_monochromatic_triangle_in_K6_l12_12544

/-- In a complete graph with 6 vertices where each edge is colored either red or blue,
    there exists a set of 3 vertices such that the edges joining them are all the same color. -/
theorem exists_monochromatic_triangle_in_K6 (color : Fin 6 → Fin 6 → Prop)
  (h : ∀ {i j : Fin 6}, i ≠ j → (color i j ∨ ¬ color i j)) :
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  ((color i j ∧ color j k ∧ color k i) ∨ (¬ color i j ∧ ¬ color j k ∧ ¬ color k i)) :=
by
  sorry

end exists_monochromatic_triangle_in_K6_l12_12544


namespace hypotenuse_is_2_sqrt_25_point_2_l12_12121

open Real

noncomputable def hypotenuse_length_of_right_triangle (ma mb : ℝ) (a b c : ℝ) : ℝ :=
  if h1 : ma = 6 ∧ mb = sqrt 27 then
    c
  else
    0

theorem hypotenuse_is_2_sqrt_25_point_2 :
  hypotenuse_length_of_right_triangle 6 (sqrt 27) a b (2 * sqrt 25.2) = 2 * sqrt 25.2 :=
by
  sorry -- proof to be filled

end hypotenuse_is_2_sqrt_25_point_2_l12_12121


namespace rectangular_coord_eqn_line_rectangular_coord_eqn_curve_max_distance_point_to_line_l12_12772

-- Define the polar coordinate equation for line l
def polar_coord_eqn (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ - Real.pi / 6) - 3 * Real.sqrt 3 = 0

-- Define the parametric equations for curve C
def parametric_curve_eqn (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, Real.sqrt 3 * Real.sin α)

-- Prove the rectangular coordinate equation for line l
theorem rectangular_coord_eqn_line {x y : ℝ} (ρ θ : ℝ) (h : polar_coord_eqn ρ θ) : 
  x - Real.sqrt 3 * y + 3 * Real.sqrt 3 = 0 :=
sorry

-- Prove the rectangular coordinate equation for curve C
theorem rectangular_coord_eqn_curve {x y α : ℝ} (hx : x = Real.cos α) (hy : y = Real.sqrt 3 * Real.sin α) : 
  x^2 + (y^2) / 3 = 1 :=
sorry

-- Prove the maximum distance d_max from any point P on curve C to line l
theorem max_distance_point_to_line {α : ℝ} (x y : ℝ) (hx : x = Real.cos α) (hy : y = Real.sqrt 3 * Real.sin α) : 
  ∃ d_max, d_max = (Real.sqrt 10 + 3 * Real.sqrt 3) / 2 :=
sorry

end rectangular_coord_eqn_line_rectangular_coord_eqn_curve_max_distance_point_to_line_l12_12772


namespace business_value_l12_12612

-- Define the conditions
variable (V : ℝ) -- Total value of the business
variable (man_shares : ℝ := (2/3) * V) -- Man's share in the business
variable (sold_shares_value : ℝ := (3/4) * man_shares) -- Value of sold shares
variable (sale_price : ℝ := 45000) -- Price the shares were sold for

-- State the theorem to be proven
theorem business_value (h : (3/4) * (2/3) * V = 45000) : V = 90000 := by
  sorry

end business_value_l12_12612


namespace parabola_focus_distance_l12_12209

noncomputable theory

open Real

theorem parabola_focus_distance (p : ℝ) (m : ℝ) (h1 : p > 0)
  (h2 : (m^2 = 8)) (h3 : sqrt ((4 - p / 2)^2 + m^2) = 17 / 4) : 
    p = 1 / 2 :=
  sorry

end parabola_focus_distance_l12_12209


namespace population_estimation_l12_12699

theorem population_estimation (initial_population : ℕ) (initial_year : ℕ) (tripling_period : ℕ) 
(target_population : ℕ) : initial_population = 500 →
initial_year = 2000 →
tripling_period = 25 →
target_population = 9000 →
∃ y, y = 2075 :=
by
  intros h_initial_population h_initial_year h_tripling_period h_target_population
  use 2075
  sorry

end population_estimation_l12_12699


namespace abs_onePointFive_sub_sqrtTwo_is_itself_l12_12010

noncomputable def onePointFive := 1.5
noncomputable def sqrtTwo := Real.sqrt 2

theorem abs_onePointFive_sub_sqrtTwo_is_itself :
  abs (onePointFive - sqrtTwo) = onePointFive - sqrtTwo :=
by
  have h : sqrtTwo < onePointFive := by norm_num; exact Real.sqrt_lt.mpr (by norm_num)
  rw abs_of_nonneg (sub_nonneg_of_le h)
  sorry

end abs_onePointFive_sub_sqrtTwo_is_itself_l12_12010


namespace positive_difference_of_complementary_angles_in_ratio_five_to_four_l12_12914

theorem positive_difference_of_complementary_angles_in_ratio_five_to_four
  (a b : ℝ)
  (h1 : a / b = 5 / 4)
  (h2 : a + b = 90) :
  |a - b| = 10 :=
sorry

end positive_difference_of_complementary_angles_in_ratio_five_to_four_l12_12914


namespace abs_onePointFive_sub_sqrtTwo_is_itself_l12_12009

noncomputable def onePointFive := 1.5
noncomputable def sqrtTwo := Real.sqrt 2

theorem abs_onePointFive_sub_sqrtTwo_is_itself :
  abs (onePointFive - sqrtTwo) = onePointFive - sqrtTwo :=
by
  have h : sqrtTwo < onePointFive := by norm_num; exact Real.sqrt_lt.mpr (by norm_num)
  rw abs_of_nonneg (sub_nonneg_of_le h)
  sorry

end abs_onePointFive_sub_sqrtTwo_is_itself_l12_12009


namespace k_h_of_3_eq_79_l12_12787

def h (x : ℝ) : ℝ := x^3
def k (x : ℝ) : ℝ := 3 * x - 2

theorem k_h_of_3_eq_79 : k (h 3) = 79 := by
  sorry

end k_h_of_3_eq_79_l12_12787


namespace mashed_potatoes_suggestion_count_l12_12531

def number_of_students_suggesting_bacon := 394
def extra_students_suggesting_mashed_potatoes := 63
def number_of_students_suggesting_mashed_potatoes := number_of_students_suggesting_bacon + extra_students_suggesting_mashed_potatoes

theorem mashed_potatoes_suggestion_count :
  number_of_students_suggesting_mashed_potatoes = 457 := by
  sorry

end mashed_potatoes_suggestion_count_l12_12531


namespace clea_ride_down_time_l12_12675

theorem clea_ride_down_time (c s d : ℝ) (h1 : d = 70 * c) (h2 : d = 28 * (c + s)) :
  (d / s) = 47 := by
  sorry

end clea_ride_down_time_l12_12675


namespace point_in_second_quadrant_l12_12731

def point := (ℝ × ℝ)

def second_quadrant (p : point) : Prop := p.1 < 0 ∧ p.2 > 0

theorem point_in_second_quadrant : second_quadrant (-1, 2) :=
sorry

end point_in_second_quadrant_l12_12731


namespace exists_positive_integer_not_div_by_primes_l12_12143

-- Define the gcd function as necessary
def gcd (a b : ℤ) : ℤ := sorry
def gcd_multiple (a : List ℤ) : ℤ := a.foldr gcd 0

-- Define what it means for a polynomial to be primitive.
def isPrimitive (P : Polynomial ℤ) : Prop :=
  gcd_multiple (P.coeff 0 :: P.coeff 1 :: P.coeffs.toList) = 1

-- Define the theorem statement.
theorem exists_positive_integer_not_div_by_primes (P : Polynomial ℤ) (S : Set ℕ)
  (hdeg : P.degree < 1398) (hprim : isPrimitive P) (hS : ∀ p ∈ S, p > 1398) :
  ∃ n : ℕ, ∀ p ∈ S, ¬(p ∣ P.eval n) :=
begin
  -- Proof goes here
  sorry
end

end exists_positive_integer_not_div_by_primes_l12_12143


namespace monotonicity_of_f_tangent_intersection_points_l12_12333

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12333


namespace theta_range_l12_12767

theorem theta_range (θ : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (Real.cos θ) * x^2 - (4 * Real.sin θ) * x + 6)
  (h2 : ∀ x, f x > 0)
  (h3 : 0 < θ ∧ θ < Real.pi) : θ ∈ set.Ioo 0 (Real.pi / 3) :=
by
  sorry

end theta_range_l12_12767


namespace simplify_power_l12_12891

theorem simplify_power (x : ℝ) : (3 * x^4)^4 = 81 * x^16 :=
by sorry

end simplify_power_l12_12891


namespace four_is_square_root_of_sixteen_l12_12610

theorem four_is_square_root_of_sixteen : (4 : ℝ) * (4 : ℝ) = 16 :=
by
  sorry

end four_is_square_root_of_sixteen_l12_12610


namespace ruby_height_l12_12429

/-- Height calculations based on given conditions -/
theorem ruby_height (Janet_height : ℕ) (Charlene_height : ℕ) (Pablo_height : ℕ) (Ruby_height : ℕ) 
  (h₁ : Janet_height = 62) 
  (h₂ : Charlene_height = 2 * Janet_height)
  (h₃ : Pablo_height = Charlene_height + 70)
  (h₄ : Ruby_height = Pablo_height - 2) : Ruby_height = 192 := 
by
  sorry

end ruby_height_l12_12429


namespace total_liquid_poured_out_l12_12657

noncomputable def capacity1 := 2
noncomputable def capacity2 := 6
noncomputable def percentAlcohol1 := 0.3
noncomputable def percentAlcohol2 := 0.4
noncomputable def totalCapacity := 10
noncomputable def finalConcentration := 0.3

theorem total_liquid_poured_out :
  capacity1 + capacity2 = 8 :=
by
  sorry

end total_liquid_poured_out_l12_12657


namespace solve_sqrt_equation_l12_12173

theorem solve_sqrt_equation (x : ℝ) :
  (sqrt (5 * x - 4) + 12 / sqrt (5 * x - 4) = 8) -> (x = 8 ∨ x = 8 / 5) :=
by
  intros h
  sorry

end solve_sqrt_equation_l12_12173


namespace median_and_mode_of_scores_l12_12457

def scores : List ℕ := [80, 80, 80, 80, 85, 85, 85, 85, 85, 85, 90, 90, 90, 90, 90, 90, 90, 90, 95, 95]

def median (l : List ℕ) : ℕ :=
let sorted := l.qsort (· ≤ ·)
let n := sorted.length
if n % 2 = 0 then
  (sorted.get (n / 2 - 1) + sorted.get (n / 2)) / 2
else
  sorted.get (n / 2)

def mode (l : List ℕ) : ℕ :=
l.frequency.keys.maxBy (fun k => l.frequency.count k)

theorem median_and_mode_of_scores :
  median scores = 87.5 ∧ mode scores = 90 :=
by
  -- TODO: Provide the proof here
  sorry

end median_and_mode_of_scores_l12_12457


namespace monotonicity_and_tangent_intersection_l12_12247

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12247


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12342

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12342


namespace find_f_one_l12_12851

theorem find_f_one (f : ℝ → ℝ) (h : ∀ x y : ℝ, f(x) * f(y) - f(x * y) = x^2 + y^2 - x * y) : f(1) = 2 :=
sorry

end find_f_one_l12_12851


namespace convert_563_base8_to_base7_l12_12151

theorem convert_563_base8_to_base7 :
  let base8 := 5 * 8^2 + 6 * 8^1 + 3 * 8^0
  in let base10 := base8
  in let base7 := 1 * 7^3 + 1 * 7^2 + 6 * 7^1 + 2 * 7^0
  in base10 = base7 := by
  sorry

end convert_563_base8_to_base7_l12_12151


namespace composite_shape_surface_area_l12_12640

-- Define the conditions as constants
def radius_hemisphere : ℝ := 8
def radius_cone : ℝ := 8
def height_cone : ℝ := 8

-- The main theorem statement to prove
theorem composite_shape_surface_area :
  let base_area_hemisphere := π * radius_hemisphere^2 in
  let curved_surface_hemisphere := 2 * π * radius_hemisphere^2 in
  let slant_height_cone := Real.sqrt (radius_cone^2 + height_cone^2) in
  let lateral_surface_cone := π * radius_cone * slant_height_cone in
  base_area_hemisphere + curved_surface_hemisphere + lateral_surface_cone = (192 + 64 * Real.sqrt 2) * π := 
by
  sorry

end composite_shape_surface_area_l12_12640


namespace monotonicity_and_tangent_intersection_l12_12254

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12254


namespace expression_meaningful_l12_12432

theorem expression_meaningful (x : ℝ) : (∃ y, y = 4 / (x - 5)) ↔ x ≠ 5 :=
by
  sorry

end expression_meaningful_l12_12432


namespace compute_fraction_product_l12_12677

theorem compute_fraction_product :
  (1 / 3)^4 * (1 / 5) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l12_12677


namespace total_trees_in_gray_areas_l12_12052

theorem total_trees_in_gray_areas (x y : ℕ) (h1 : 82 + x = 100) (h2 : 82 + y = 90) :
  x + y = 26 :=
by
  sorry

end total_trees_in_gray_areas_l12_12052


namespace sum_f_from_1_to_1024_l12_12494

def int_part_log2 (m : ℕ) : ℕ := 
  if m = 0 then 0 
  else Nat.log2 m

theorem sum_f_from_1_to_1024 : 
  (∑ m in Finset.range 1025, int_part_log2 m) = 8204 :=
begin
  sorry
end

end sum_f_from_1_to_1024_l12_12494


namespace added_expression_correct_l12_12944

noncomputable def added_expression (k : ℕ) : ℝ :=
  (1 / (2 * k + 3 : ℝ)) - (1 / (2 * k + 2 : ℝ))

theorem added_expression_correct (k : ℕ) (hk : 1 < k) :
  added_expression k = (1 / (2 * k + 3 : ℝ)) - (1 / (2 * k + 2 : ℝ)) :=
by
  sorry

end added_expression_correct_l12_12944


namespace find_r_over_s_at_0_l12_12552

noncomputable def r (k a : ℝ) := (3 : ℝ) * k * (Polynomial.X - Polynomial.C (2 : ℝ)) * (Polynomial.X - Polynomial.C a)
noncomputable def s := (Polynomial.X - Polynomial.C (2 : ℝ)) * (Polynomial.X + Polynomial.C (3 : ℝ)) * (Polynomial.X - Polynomial.C (1 : ℝ))

theorem find_r_over_s_at_0 (k a : ℝ) : 
  (r k a / s).eval 0 = -1 := 
by 
  -- Proof steps would go here
  sorry

end find_r_over_s_at_0_l12_12552


namespace geometry_problem_l12_12515

structure Point where
  x : Int
  y : Int

def midpoint (P R : Point) : Point :=
  { x := (P.x + R.x) / 2, y := (P.y + R.y) / 2 }

def reflect_x (P : Point) : Point :=
  { x := P.x, y := -P.y }

def rotate_180 (P : Point) : Point :=
  { x := -P.x, y := -P.y }

def sum_coords (P : Point) : Int :=
  P.x + P.y

theorem geometry_problem :
  let P := { x := 1, y := 4 }
  let R := { x := 13, y := 16 }
  let M := midpoint P R
  let M' := midpoint (reflect_x P) (reflect_x R)
  let M'' := rotate_180 M'
  sum_coords M'' = 3 := sorry

end geometry_problem_l12_12515


namespace parallelogram_count_l12_12802

theorem parallelogram_count (m n : ℕ) : (choose m 2) * (choose n 2) = number_of_parallelograms m n :=
sorry

end parallelogram_count_l12_12802


namespace largest_prime_divisor_of_given_expression_l12_12712

-- Define the given expression
def given_expression : ℕ := 36^2 + 45^2

-- Define the definition of a prime number and the largest prime divisor
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m ∣ n → m = n

def largest_prime_divisor (n : ℕ) : ℕ :=
  list.max' (list.filter is_prime (list.divisors n))

-- State the theorem to be proven
theorem largest_prime_divisor_of_given_expression : 
  largest_prime_divisor given_expression = 71 :=
by 
  -- Note: The proof is omitted with 'sorry' as instructed
  sorry

end largest_prime_divisor_of_given_expression_l12_12712


namespace range_of_a_l12_12733

theorem range_of_a (a : ℝ) : 
  (log a (1 / 4) < 2 ∧ (1 / 4)^(a - 1) > 2) ↔ (0 < a ∧ a < 1 / 2) := by
  sorry

end range_of_a_l12_12733


namespace kanul_total_amount_l12_12830

theorem kanul_total_amount (T : ℝ) (R : ℝ) (M : ℝ) (C : ℝ)
  (hR : R = 80000)
  (hM : M = 30000)
  (hC : C = 0.2 * T)
  (hT : T = R + M + C) : T = 137500 :=
by {
  sorry
}

end kanul_total_amount_l12_12830


namespace monotonicity_f_tangent_points_l12_12369

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12369


namespace base_of_triangle_is_24_l12_12046

def triangle_sides_sum := 50
def left_side : ℕ := 12
def right_side := left_side + 2
def base := triangle_sides_sum - left_side - right_side

theorem base_of_triangle_is_24 :
  base = 24 :=
by 
  have h : left_side = 12 := rfl
  have h2 : right_side = 14 := by simp [right_side, h]
  have h3 : base = 24 := by simp [base, triangle_sides_sum, h, h2]
  exact h3

end base_of_triangle_is_24_l12_12046


namespace classifyGreatPolynomials_l12_12118

-- Definition of a "great" polynomial
def isGreat (P : Polynomial ℝ) (a : ℤ) : Prop :=
  a > 1 ∧ (∀ x : ℤ, ∃ z : ℤ, a * (P.eval x) = P.eval z)

-- Main theorem to prove
theorem classifyGreatPolynomials (P : Polynomial ℝ) (a : ℤ) :
  isGreat P a → ∃ (c : ℝ) (x0 : ℚ) (d : ℕ), P = (Polynomial.C c) * (Polynomial.X - Polynomial.C x0) ^ d :=
by
  sorry

end classifyGreatPolynomials_l12_12118


namespace function_D_is_odd_function_D_is_decreasing_l12_12664

def f_D (x : ℝ) : ℝ := -x * |x|

theorem function_D_is_odd (x : ℝ) : f_D (-x) = -f_D x := by
  sorry

theorem function_D_is_decreasing (x y : ℝ) (h : x < y) : f_D x > f_D y := by
  sorry

end function_D_is_odd_function_D_is_decreasing_l12_12664


namespace range_of_m_l12_12403

noncomputable def f (m : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 1/2 then 10 * x - m else x * Real.exp x - 2 * m * x + m

theorem range_of_m (m e : ℝ) : 
  (∀ (x : ℝ), f m x = 0 → ∃ ! (x : ℝ), f m x = 0) → e < m ∧ m ≤ 5 :=
sorry

end range_of_m_l12_12403


namespace sector_area_90_deg_radius_2_l12_12652

def area_of_sector (angle : ℝ) (radius : ℝ) : ℝ :=
  (angle / 360) * π * radius^2

theorem sector_area_90_deg_radius_2 : area_of_sector 90 2 = π :=
  sorry

end sector_area_90_deg_radius_2_l12_12652


namespace modulus_of_complex_l12_12227

theorem modulus_of_complex :
  let z : ℂ := (⟨(Real.sqrt 3 / 2 : ℝ), (- (3 / 2 : ℝ))⟩ : ℂ) in
  Complex.abs z = Real.sqrt 3 :=
by
  -- Sorry skips the proof
  sorry

end modulus_of_complex_l12_12227


namespace cylinder_height_l12_12649

noncomputable def height_of_cylinder_inscribed_in_sphere : ℝ := 4 * Real.sqrt 10

theorem cylinder_height :
  ∀ (R_cylinder R_sphere : ℝ), R_cylinder = 3 → R_sphere = 7 →
  (height_of_cylinder_inscribed_in_sphere = 4 * Real.sqrt 10) := by
  intros R_cylinder R_sphere h1 h2
  sorry

end cylinder_height_l12_12649


namespace compute_alpha_l12_12847

-- Declaring that we are using complex numbers
variable (α β : ℂ)

-- Conditions given in the problem
-- Condition 1 and 2: α, β are complex numbers (implicitly handled by type declaration)
-- Condition 3: α + β is a positive real number
def alpha_plus_beta_is_positive_real (α β : ℂ) : Prop := ∃ r : ℝ, r > 0 ∧ α + β = ↑r

-- Condition 4: i(α - 3β) is a positive real number
def i_times_alpha_minus_3beta_is_positive_real (α β : ℂ) : Prop := 
  ∃ r : ℝ, r > 0 ∧ i * (α - 3 * β) = ↑r

-- Condition 5: β = 4 + 3i
def beta_value : ℂ := 4 + 3i

-- The theorem to compute α
theorem compute_alpha (α : ℂ) (β : ℂ) (h1 : alpha_plus_beta_is_positive_real α β) 
  (h2 : i_times_alpha_minus_3beta_is_positive_real α β) 
  (h3 : β = beta_value) : α = 12 - 3i := 
sorry

end compute_alpha_l12_12847


namespace students_math_eng_not_hist_is_5_l12_12625

variable (students : Nat)
variable (n1 n2 : Nat)

-- Conditions
axiom total_students : students = 228
axiom all_subjects : ∀ s, s < students → (takes_subject s english ∨ takes_subject s mathematics ∨ takes_subject s history)
axiom math_eng_not_hist_eq_only_math : n1 = number_of_students_only_math
axiom no_only_eng_hist : (number_of_students_only_english = 0) ∧ (number_of_students_only_history = 0)
axiom math_hist_not_eng : number_of_students_math_hist_not_eng = 6
axiom eng_hist_five_times_all_three : number_of_students_eng_hist = 5 * n2
axiom n2_nonzero_even : (n2 > 0) ∧ (even n2)
axiom total_equation : n1 + 3 * n2 = 11

-- Goal
theorem students_math_eng_not_hist_is_5 : n1 = 5 :=
sorry

end students_math_eng_not_hist_is_5_l12_12625


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12297

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12297


namespace packing_big_boxes_l12_12417

def total_items := 8640
def items_per_small_box := 12
def small_boxes_per_big_box := 6

def num_big_boxes (total_items items_per_small_box small_boxes_per_big_box : ℕ) : ℕ :=
  (total_items / items_per_small_box) / small_boxes_per_big_box

theorem packing_big_boxes : num_big_boxes total_items items_per_small_box small_boxes_per_big_box = 120 :=
by
  sorry

end packing_big_boxes_l12_12417


namespace sqrt_diff_ineq_sum_sq_gt_sum_prod_l12_12985

-- First proof problem: Prove that sqrt(11) - 2 * sqrt(3) > 3 - sqrt(10)
theorem sqrt_diff_ineq : (Real.sqrt 11 - 2 * Real.sqrt 3) > (3 - Real.sqrt 10) := sorry

-- Second proof problem: Prove that a^2 + b^2 + c^2 > ab + bc + ca given a, b, and c are real numbers that are not all equal
theorem sum_sq_gt_sum_prod (a b c : ℝ) (h : ¬ (a = b ∧ b = c ∧ a = c)) : a^2 + b^2 + c^2 > a * b + b * c + c * a := sorry

end sqrt_diff_ineq_sum_sq_gt_sum_prod_l12_12985


namespace circumference_variables_l12_12525

noncomputable def is_constant (x : ℝ) : Prop := ∀ y : ℝ, x = y -- Definition for constants.

theorem circumference_variables :
  ∃ C R : ℝ, ∃ two pi : ℝ, (two = 2) ∧ (π = Real.pi) ∧ (C = 2 * π * R) ∧ 
  (¬ R = C) ∧ is_constant two ∧ is_constant π :=
by
  use 2 * π * arbitrary ℝ, arbitrary ℝ, 2, π
  split
  . rfl
  split
  . rfl
  split
  . exact Real.pi
  split
  . sorry -- Relating \(C = 2πR\)
  split
  . sorry -- Showing \(R ≠ C\)
  split
  . exact λ x, rfl
  . exact λ x, rfl

end circumference_variables_l12_12525


namespace monotonicity_tangent_points_l12_12274

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12274


namespace solve_proportion_l12_12614

noncomputable def x : ℝ := 0.6

theorem solve_proportion (x : ℝ) (h : 0.75 / x = 10 / 8) : x = 0.6 :=
by
  sorry

end solve_proportion_l12_12614


namespace monotonicity_of_f_tangent_intersection_points_l12_12332

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12332


namespace age_of_eldest_boy_l12_12977

theorem age_of_eldest_boy (x : ℕ) (h1 : (3*x + 5*x + 7*x) / 3 = 15) :
  7 * x = 21 :=
sorry

end age_of_eldest_boy_l12_12977


namespace tangent_line_find_a_l12_12435

theorem tangent_line_find_a (a : ℝ) (f : ℝ → ℝ) (tangent : ℝ → ℝ) (x₀ : ℝ)
  (hf : ∀ x, f x = x + 1/x - a * Real.log x)
  (h_tangent : ∀ x, tangent x = x + 1)
  (h_deriv : deriv f x₀ = deriv tangent x₀)
  (h_eq : f x₀ = tangent x₀) :
  a = -1 :=
sorry

end tangent_line_find_a_l12_12435


namespace part1_monotonicity_part2_tangent_intersection_l12_12268

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12268


namespace cricket_bat_weight_l12_12161

-- Define the conditions as Lean definitions
def weight_of_basketball : ℕ := 36
def weight_of_basketballs (n : ℕ) := n * weight_of_basketball
def weight_of_cricket_bats (m : ℕ) := m * (weight_of_basketballs 4 / 8)

-- State the theorem and skip the proof
theorem cricket_bat_weight :
  weight_of_cricket_bats 1 = 18 :=
by
  sorry

end cricket_bat_weight_l12_12161


namespace elsa_final_marbles_l12_12162

def start_marbles : ℕ := 40
def lost_breakfast : ℕ := 3
def given_susie : ℕ := 5
def new_marbles : ℕ := 12
def returned_marbles : ℕ := 2 * given_susie

def final_marbles : ℕ :=
  start_marbles - lost_breakfast - given_susie + new_marbles + returned_marbles

theorem elsa_final_marbles : final_marbles = 54 := by
  sorry

end elsa_final_marbles_l12_12162


namespace smallest_z_minus_x_l12_12936

theorem smallest_z_minus_x (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x * y * z = 9!)
  (h5 : x < y)
  (h6 : y < z) :
  z - x = 228 :=
sorry

end smallest_z_minus_x_l12_12936


namespace ball_hits_count_in_sector_l12_12682

theorem ball_hits_count_in_sector (θ α : ℝ) 
  (hθ : θ = 30 * (Real.pi / 180))
  (hα : α = 110 * (Real.pi / 180)) :
  let hits := Nat.floor ((α - 20 * (Real.pi / 180)) / θ) + 1 in
  hits = 3 :=
by
  sorry

end ball_hits_count_in_sector_l12_12682


namespace simplify_expression_l12_12782
theorem simplify_expression (c : ℝ) : 
    (3 * c + 6 - 6 * c) / 3 = -c + 2 := 
by 
    sorry

end simplify_expression_l12_12782


namespace cost_of_four_enchiladas_and_five_tacos_l12_12883

-- Define the cost of an enchilada and a taco
variables (e t : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := e + 4 * t = 2.30
def condition2 : Prop := 4 * e + t = 3.10

-- Define the final cost of four enchiladas and five tacos
def cost : ℝ := 4 * e + 5 * t

-- State the theorem we need to prove
theorem cost_of_four_enchiladas_and_five_tacos 
  (h1 : condition1 e t) 
  (h2 : condition2 e t) : 
  cost e t = 4.73 := 
sorry

end cost_of_four_enchiladas_and_five_tacos_l12_12883


namespace tangent_parallel_to_line_l12_12923

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_to_line (a : ℝ) (b : ℝ) :
  (f' a = 4) → (f a = b) →
  (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = -4) :=
by
  intros h1 h2
  have solution1 : a = 1 ∨ a = -1, from sorry,
  cases solution1 with ha1 ha2,
  { left,
    have hb : b = 0, from sorry,
    exact ⟨ha1, hb⟩ },
  { right,
    have hb : b = -4, from sorry,
    exact ⟨ha2, hb⟩ }

end tangent_parallel_to_line_l12_12923


namespace can_transform_rectangles_l12_12590

-- Define the size of the rectangles
def rect_width : ℕ := 9
def rect_length : ℕ := 12

-- Define the specific color pattern of the rectangles (abstractly as a function)
def color_pattern (x y : ℕ) : ℕ := (abstract pattern function)

-- The theorem stating that the left rectangle can be transformed into the right rectangle
theorem can_transform_rectangles :
  ∃ (a1 a2 a3 a4 : ℕ × ℕ) (b1 b2 b3 b4 : ℕ × ℕ),
    (a1.1 * a1.2 + a2.1 * a2.2 + a3.1 * a3.2 + a4.1 * a4.2 = rect_width * rect_length) ∧
    (b1.1 * b1.2 + b2.1 * b2.2 + b3.1 * b3.2 + b4.1 * b4.2 = rect_width * rect_length) ∧
    (∀ i j, i < rect_width ∧ j < rect_length → color_pattern i j = color_pattern ((reassemble_function (i, j)).1) ((reassemble_function (i, j)).2)) :=
begin
  sorry
end

end can_transform_rectangles_l12_12590


namespace tangent_product_l12_12783

theorem tangent_product (m : ℕ) :
  (∀ x ∈ {1, 2, 3, ..., 15}, (1 + Real.tan (x * Real.pi / 180)) * (1 + Real.tan ((45 - x) * Real.pi / 180)) = 2) →
  (∏ x in Finset.range 15 + 1, (1 + Real.tan (x * Real.pi / 180))) = 2 ^ m →
  m = 7 :=
by
  sorry

end tangent_product_l12_12783


namespace bill_and_harry_nuts_l12_12139

theorem bill_and_harry_nuts :
  ∀ (sue_nuts : ℕ) (harry_nuts : ℕ) (bill_nuts : ℕ),
    sue_nuts = 48 →
    harry_nuts = (2.25 * sue_nuts).to_nat →
    bill_nuts = (6.5 * harry_nuts).to_nat →
    bill_nuts + harry_nuts = 810 :=
by
  intros sue_nuts harry_nuts bill_nuts h1 h2 h3
  rw [h1, h2, h3]
  sorry

end bill_and_harry_nuts_l12_12139


namespace boat_travel_distance_downstream_l12_12099

-- Define the conditions given in the problem
def speed_boat_still_water := 22 -- in km/hr
def speed_stream := 5 -- in km/hr
def time_downstream := 2 -- in hours

-- Define a function to compute the effective speed downstream
def effective_speed_downstream (speed_boat: ℝ) (speed_stream: ℝ) : ℝ :=
  speed_boat + speed_stream

-- Define a function to compute the distance travelled downstream
def distance_downstream (speed: ℝ) (time: ℝ) : ℝ :=
  speed * time

-- The main theorem to prove
theorem boat_travel_distance_downstream :
  distance_downstream (effective_speed_downstream speed_boat_still_water speed_stream) time_downstream = 54 :=
by
  -- Proof is to be filled in later
  sorry

end boat_travel_distance_downstream_l12_12099


namespace monotonicity_f_tangent_points_l12_12363

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12363


namespace distinct_sums_count_l12_12486

def distinct_elements_sum (S : Set ℝ) (k : ℕ) : Set ℝ :=
  { s | ∃ (x : Fin k → ℝ), ((∀ i j, i ≠ j → x i ≠ x j) ∧ (∀ i, x i ∈ S) ∧ s = (Finset.univ : Finset (Fin k)).sum (λ i, x i)) }

theorem distinct_sums_count (n k : ℕ) (S : Set ℝ) (hS : S.card = n) (hk : k ≤ n) :
  (distinct_elements_sum S k).card ≥ k * (n - k) + 1 :=
sorry

end distinct_sums_count_l12_12486


namespace monotonicity_of_f_tangent_intersection_l12_12285

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12285


namespace find_z_l12_12181

theorem find_z (a b p q : ℝ) (z : ℝ) 
  (cond : (z + a + b = q * (p * z - a - b))) : 
  z = (a + b) * (q + 1) / (p * q - 1) :=
sorry

end find_z_l12_12181


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12316

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12316


namespace euler_totient_mul_l12_12529

namespace EulerTotient

open Nat

def coprime (a b : ℕ) : Prop := gcd a b = 1

def euler_totient (n : ℕ) : ℕ := n * ((n.prime_divisor_list.map (λ p, 1 - 1 / p)).prod)

theorem euler_totient_mul (a b : ℕ) (h_coprime : coprime a b) :
  euler_totient a * euler_totient b = euler_totient (a * b) :=
sorry

end EulerTotient


end euler_totient_mul_l12_12529


namespace monotonicity_of_f_tangent_intersection_points_l12_12328

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12328


namespace seating_arrangements_l12_12057

-- Define the twelve chairs
def Chairs : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

-- Define a couple as a pair representing man and woman
structure Couple where
  man  : ℕ
  woman : ℕ

-- Specify the condition that couples do not sit next to, two chairs apart or across from each other
def validPosition (m w : ℕ) : Prop :=
  Chairs.contains m ∧
  Chairs.contains w ∧
  m ≠ w ∧
  ¬ (m = w + 1 ∨ m = w - 1 ∨ m = w + 2 ∨ m = w - 2 ∨ abs (m - w) = 6)

-- Define the problem of seating six married couples
def validSeatingArrangement (arrangement : Finset (ℕ × ℕ)) : Prop :=
  arrangement.card = 6 ∧
  ∀ (c : Couple), arrangement.contains (c.man, c.woman) → validPosition c.man c.woman

-- State the main theorem we want to prove
theorem seating_arrangements : 
  ∃ (arrangement : Finset (ℕ × ℕ)),
  validSeatingArrangement arrangement ∧ arrangement.card = 16 := 
sorry

end seating_arrangements_l12_12057


namespace sum_integer_solutions_abs_lt_abs_sub_lt_ten_l12_12067

theorem sum_integer_solutions_abs_lt_abs_sub_lt_ten :
  (∑ n in Finset.filter (λ n : ℤ, |n| < |n - 4| ∧ |n - 4| < 10) (Finset.Ico (-6) 2), n) = -20 :=
  sorry

end sum_integer_solutions_abs_lt_abs_sub_lt_ten_l12_12067


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12318

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12318


namespace fruit_basket_count_l12_12781

theorem fruit_basket_count :
  (∑ apples in (2..7), ∑ oranges in (0..12), 1) +
  (∑ oranges in (2..12), ∑ apples in (0..7), 1) -
  ∑ apples in (2..7), 1 -
  ∑ oranges in (2..12), 1 = 101 :=
by
  sorry

end fruit_basket_count_l12_12781


namespace seniors_playing_all_three_l12_12595

variables {α : Type} [Fintype α]
variables (F B L : Finset α)

def total_seniors := F ∪ B ∪ L
def football := F
def baseball := B
def lacrosse := L
def football_lacrosse := F ∩ L
def baseball_football := B ∩ F
def baseball_lacrosse := B ∩ L
def all_three := F ∩ B ∩ L

variable (n : ℕ)

theorem seniors_playing_all_three (h1 : total_seniors = 85)
    (h2 : football.card = 74)
    (h3 : baseball.card = 26)
    (h4 : football_lacrosse.card = 17)
    (h5 : baseball_football.card = 18)
    (h6 : baseball_lacrosse.card = 13)
    (h7 : lacrosse.card = 2 * n) :
    all_three.card = 11 :=
by sorry

end seniors_playing_all_three_l12_12595


namespace largest_possible_n_l12_12133

theorem largest_possible_n 
  (a b : ℕ → ℤ)
  (h1 : a 1 = 1)
  (h2 : b 1 = 1)
  (h3 : ∀ n, 1 < a 2 ∧ a 2 ≤ b 2)
  (h4 : ∃ n, a n * b n = 2210)
  : ∃ n, (∃ (x : ℤ) (y : ℤ), a n = 1 + (n - 1) * x ∧ b n = 1 + (n - 1) * y) ∧ n ≤ 170 :=
  sorry

end largest_possible_n_l12_12133


namespace part1_monotonicity_part2_tangent_intersection_l12_12261

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12261


namespace sum_of_squares_ways_l12_12809

theorem sum_of_squares_ways : 
  ∃ ways : ℕ, ways = 2 ∧
    (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = 100) ∧ 
    (∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x^2 + y^2 + z^2 + w^2 = 100) := 
sorry

end sum_of_squares_ways_l12_12809


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12309

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12309


namespace unique_prime_sum_diff_l12_12705

-- Define that p is a prime number that satisfies both conditions
def sum_two_primes (p a b : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime a ∧ Nat.Prime b ∧ p = a + b

def diff_two_primes (p c d : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime c ∧ Nat.Prime d ∧ p = c - d

-- Main theorem to prove: The only prime p that satisfies both conditions is 5
theorem unique_prime_sum_diff (p : ℕ) :
  (∃ a b, sum_two_primes p a b) ∧ (∃ c d, diff_two_primes p c d) → p = 5 :=
by
  sorry

end unique_prime_sum_diff_l12_12705


namespace jelly_cost_l12_12696

theorem jelly_cost (N C J : ℕ) (hC : C > 0) (hJ : J > 0) (hN : N > 1) (h : N * (3 * C + 6 * J) = 312) :
  (6 * N * J) / 100 = 0.72 := by
  sorry

end jelly_cost_l12_12696


namespace max_reflections_max_reflections_example_l12_12644

-- Definition of the conditions
def angle_cda := 10  -- angle in degrees
def max_angle := 90  -- practical limit for angle of reflections

-- Given that the angle of incidence after n reflections is 10n degrees,
-- prove that the largest possible n is 9 before exceeding practical limits.
theorem max_reflections (n : ℕ) (h₁ : angle_cda = 10) (h₂ : max_angle = 90) :
  10 * n ≤ 90 :=
by sorry

-- Specific case instantiating n = 9
theorem max_reflections_example : (10 : ℕ) * 9 ≤ 90 := max_reflections 9 rfl rfl

end max_reflections_max_reflections_example_l12_12644


namespace holidays_per_month_l12_12129

theorem holidays_per_month (total_holidays : ℕ) (months_in_year : ℕ) (holidays_in_month : ℕ) 
    (h1 : total_holidays = 48) (h2 : months_in_year = 12) : holidays_in_month = 4 := 
by
  sorry

end holidays_per_month_l12_12129


namespace geometric_progression_periodic_l12_12519

theorem geometric_progression_periodic
  (p k n : ℕ)
  (b q : ℕ → ℕ)
  (h_geometric : ∀ n, b n = b 0 * q n)
  (h_p_adic : ∀ i, b i ∈ ℤ) -- All b_i are p-adic integers
  (h_not_divisible : ¬ divides p q)
  : b n % 10^k = b (n + p^(k-1)*(p-1)) % 10^k := by
  sorry

end geometric_progression_periodic_l12_12519


namespace javiers_wife_took_30_percent_l12_12479

noncomputable def percentage_cookies_taken (baked_cookies : ℕ) (daughter_took : ℕ) (remaining_cookies : ℕ) : ℕ :=
  let P : ℚ := 30 -- The percentage we need to prove
  let wife_took := (P / 100) * baked_cookies
  let after_wife_took := baked_cookies - wife_took
  let after_daughter_took := after_wife_took - daughter_took
  let javier_ate := after_daughter_took / 2
  let not_eaten := after_daughter_took - javier_ate
  if not_eaten = remaining_cookies then P.toNat else 0

theorem javiers_wife_took_30_percent (baked_cookies daughter_took remaining_cookies : ℕ) (h_baked : baked_cookies = 200) (h_daughter : daughter_took = 40) (h_remaining : remaining_cookies = 50) : 
  percentage_cookies_taken baked_cookies daughter_took remaining_cookies = 30 :=
by
  have eq1 : (200 - (30 / 100) * 200 - 40 : ℚ) = 110 := by norm_num
  have eq2 : (200 - (30 / 100) * 200 - 40) / 2 = 55 := by norm_num
  have : 110 - 55 = 50 := by norm_num
  rw [percentage_cookies_taken, if_pos this]
  simp
  norm_cast
  assumption_mod_cast
  sorry

end javiers_wife_took_30_percent_l12_12479


namespace base_b_digit_count_l12_12894

noncomputable def number_of_possible_b : ℕ := 2

theorem base_b_digit_count (b : ℕ) (h1 : b ≥ 2) (h2 : b^3 ≤ 250) (h3 : 250 < b^4) : b = 5 ∨ b = 6 :=
begin
  sorry
end

end base_b_digit_count_l12_12894


namespace number_of_sequences_fixed_l12_12681

def L1 : (ℚ × ℚ) → (ℚ × ℚ) := λ p, (-p.2, p.1)
def R1 : (ℚ × ℚ) → (ℚ × ℚ) := λ p, (p.2, -p.1)
def H1 : (ℚ × ℚ) → (ℚ × ℚ) := λ p, (p.2, p.1)
def V1 : (ℚ × ℚ) → (ℚ × ℚ) := λ p, (-p.2, -p.1)

def transform (seq : List ((ℚ × ℚ) → (ℚ × ℚ))) : (ℚ × ℚ) → (ℚ × ℚ) :=
  seq.foldr (· ∘ ·) id

def positions :=
  [(2, 2), (-2, 2), (-2, -2), (2, -2)]

def fixed_by_transforms (seq : List ((ℚ × ℚ) → (ℚ × ℚ))) : Prop :=
  positions.all (λ p, transform seq p ∈ positions)

def num_sequences_fixed : ℕ :=
  ((List.replicate 4 [L1, R1, H1, V1]).product).filter fixed_by_transforms).length

theorem number_of_sequences_fixed :
  num_sequences_fixed = 165 :=
sorry

end number_of_sequences_fixed_l12_12681


namespace distinct_bead_arrangements_on_bracelet_l12_12810

open Nat

-- Definition of factorial
def fact : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * fact n

-- Theorem stating the number of distinct arrangements of 7 beads on a bracelet
theorem distinct_bead_arrangements_on_bracelet : 
  fact 7 / 14 = 360 := 
by 
  sorry

end distinct_bead_arrangements_on_bracelet_l12_12810


namespace base_of_triangle_is_24_l12_12045

def triangle_sides_sum := 50
def left_side : ℕ := 12
def right_side := left_side + 2
def base := triangle_sides_sum - left_side - right_side

theorem base_of_triangle_is_24 :
  base = 24 :=
by 
  have h : left_side = 12 := rfl
  have h2 : right_side = 14 := by simp [right_side, h]
  have h3 : base = 24 := by simp [base, triangle_sides_sum, h, h2]
  exact h3

end base_of_triangle_is_24_l12_12045


namespace total_money_raised_l12_12111

def tickets_sold : ℕ := 25
def ticket_price : ℕ := 2
def num_15_donations : ℕ := 2
def donation_15_amount : ℕ := 15
def donation_20_amount : ℕ := 20

theorem total_money_raised : 
  tickets_sold * ticket_price + num_15_donations * donation_15_amount + donation_20_amount = 100 := 
by sorry

end total_money_raised_l12_12111


namespace find_triangle_base_l12_12040

theorem find_triangle_base (left_side : ℝ) (right_side : ℝ) (base : ℝ) 
  (h_left : left_side = 12) 
  (h_right : right_side = left_side + 2)
  (h_sum : left_side + right_side + base = 50) :
  base = 24 := 
sorry

end find_triangle_base_l12_12040


namespace probability_of_XXOXOXO_l12_12728

theorem probability_of_XXOXOXO :
  let ways_to_arrange : ℕ := Nat.choose 7 4 in
  let favorable_outcome : ℕ := 1 in
  let probability : ℚ := favorable_outcome / ways_to_arrange in
  probability = 1 / 35 :=
by {
  let ways_to_arrange := Nat.choose 7 4,
  let favorable_outcome := 1,
  let probability := favorable_outcome / ways_to_arrange,
  have h : probability = 1 / 35,
  sorry
}

end probability_of_XXOXOXO_l12_12728


namespace intersecting_sums_l12_12020

theorem intersecting_sums :
  let f : ℝ → ℝ := λ x, x^3 - 3 * x + 2
  let g : ℝ → ℝ := λ x, x^2 - 2 * x + 2
  ∃ x1 x2 x3 y1 y2 y3: ℝ,
  (f x1 = g x1) ∧ (f x2 = g x2) ∧ (f x3 = g x3) ∧
  (x1 + x2 + x3 = 0) ∧ (y1 + y2 + y3 = 5) ∧
  (y1 = x1^2 - 2 * x1 + 2) ∧ (y2 = x2^2 - 2 * x2 + 2) ∧ (y3 = x3^2 - 2 * x3 + 2) :=
by sorry

end intersecting_sums_l12_12020


namespace monotonicity_tangent_intersection_points_l12_12382

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12382


namespace log_two_bound_l12_12946

theorem log_two_bound (h₁ : 2^11 = 2048) (h₂ : 2^12 = 4096) (h₃ : 10^4 = 10000) : 
  log 10 2 < 4/11 :=
by
  sorry

end log_two_bound_l12_12946


namespace inverse_function_l12_12062

def f (x : ℝ) : ℝ := 10 - 7 * x

def g (x : ℝ) : ℝ := (10 - x) / 7

theorem inverse_function :
  (∀ x : ℝ, f (g x) = x) ∧ (∀ y : ℝ, g (f y) = y) :=
by
  -- Here is the proof, which is omitted as per the instructions.
  sorry

end inverse_function_l12_12062


namespace domain_of_g_l12_12145

noncomputable def g (x : ℝ) : ℝ := real.sqrt (4 - real.sqrt (7 - real.sqrt (2 * x)))

theorem domain_of_g :
  {x : ℝ | 0 ≤ x ∧ x ≤ 24.5} = {x : ℝ | ∃ y : ℝ, g y = real.sqrt (4 - real.sqrt (7 - real.sqrt (2 * y)))} :=
by
  sorry

end domain_of_g_l12_12145


namespace exists_point_O_l12_12621

open Set

theorem exists_point_O (A : Set ℝ → Set ℝ → Prop) (hA : IsCompact A ∧ Convex ℝ A) :
    ∃ O ∈ A, ∀ (X Y : ℝ) (XX' : ℝ → Prop), (XX' X ∧ XX' Y ∧ O ∈ XX') → 
    (X ∈ frontier A ∧ Y ∈ frontier A) →
    (1 / 2 : ℝ) ≤ (dist O X / dist O Y) ∧ 
    (dist O Y / dist O X) ≤ 2 :=
by sorry

end exists_point_O_l12_12621


namespace polynomial_min_degree_l12_12004

theorem polynomial_min_degree :
  ∀ (P : Polynomial ℚ), P ≠ 0 →
  (2 - Real.sqrt 5) ∈ P.root_set ℚ → (-2 - Real.sqrt 5) ∈ P.root_set ℚ →
  (3 + Real.sqrt 11) ∈ P.root_set ℚ → (3 - Real.sqrt 11) ∈ P.root_set ℚ →
  P.degree ≥ 6 :=
by
  sorry

end polynomial_min_degree_l12_12004


namespace find_x_l12_12198

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 3) (h2 : y = 27) : x = 108 := by
  sorry

end find_x_l12_12198


namespace distribute_oranges_l12_12998

theorem distribute_oranges (distinct_oranges : ℕ) (sons : ℕ) (chosen_sons : ℕ)
  (h_oranges : distinct_oranges = 5) 
  (h_sons : sons = 8) 
  (h_chosen_sons : chosen_sons = 5) : 
  (nat.choose sons chosen_sons * nat.factorial distinct_oranges) = 6720 := 
by 
  rw [h_oranges, h_sons, h_chosen_sons]
  have h_binom : nat.choose 8 5 = 56 := 
  by sorry
  have h_fact : nat.factorial 5 = 120 := 
  by sorry
  rw [h_binom, h_fact]
  exact rfl

end distribute_oranges_l12_12998


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12397

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12397


namespace liquid_x_percentage_l12_12084

theorem liquid_x_percentage (a_weight b_weight : ℝ) (a_percentage b_percentage : ℝ)
  (result_weight : ℝ) (x_weight_result : ℝ) (x_percentage_result : ℝ) :
  a_weight = 500 → b_weight = 700 → a_percentage = 0.8 / 100 →
  b_percentage = 1.8 / 100 → result_weight = a_weight + b_weight →
  x_weight_result = a_weight * a_percentage + b_weight * b_percentage →
  x_percentage_result = (x_weight_result / result_weight) * 100 →
  x_percentage_result = 1.3833 :=
by sorry

end liquid_x_percentage_l12_12084


namespace crushing_load_value_l12_12697

-- Given definitions
def W : ℕ := 3
def T : ℕ := 2
def H : ℕ := 6
def L : ℕ := (30 * W^3 * T^5) / H^3

-- Theorem statement
theorem crushing_load_value :
  L = 120 :=
by {
  -- We provided definitions using the given conditions.
  -- Placeholder for proof is provided
  sorry
}

end crushing_load_value_l12_12697


namespace candidate_defeated_by_l12_12670

-- Definitions as per conditions
def totalPolledVotes : ℕ := 12600
def invalidVotes : ℕ := 100
def validVotes : ℕ := totalPolledVotes - invalidVotes
def defeatedCandidatePercentage : ℝ := 0.30
def winningCandidatePercentage : ℝ := 1 - defeatedCandidatePercentage

-- Calculations based on the definitions
def D : ℕ := (defeatedCandidatePercentage * validVotes.toReal).toNat
def W : ℕ := (winningCandidatePercentage * validVotes.toReal).toNat

-- The statement to prove
theorem candidate_defeated_by : W - D = 5000 :=
by
  sorry

end candidate_defeated_by_l12_12670


namespace find_x_value_l12_12581

theorem find_x_value {C S x : ℝ}
  (h1 : C = 100 * (1 + x / 100))
  (h2 : S - C = 10 / 9)
  (h3 : S = 100 * (1 + x / 100)):
  x = 10 :=
by
  sorry

end find_x_value_l12_12581


namespace monotonicity_tangent_intersection_points_l12_12377

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12377


namespace graph_shift_correct_l12_12938

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x) - Real.sqrt 3 * Real.cos (3 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (3 * x)

theorem graph_shift_correct :
  ∀ (x : ℝ), f x = g (x - (5 * Real.pi / 18)) :=
sorry

end graph_shift_correct_l12_12938


namespace monotonicity_and_tangent_intersections_l12_12351

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12351


namespace find_number_approx_l12_12427

theorem find_number_approx :
  ∃ N : ℝ, (0.1 * 0.3 * (16 / 100 * ((0.5 * N)^3 / 4))^2 = 90) ∧ N ≈ 17.78 :=  -- '≈' denotes approximate equality
by
  sorry

end find_number_approx_l12_12427


namespace find_value_of_c_l12_12438

-- Given conditions
variables (A B C a b c : ℝ)
variables (triangle_ABC : triangle A B C)
variable (tan_A_eq_7_tan_B : tan A = 7 * tan B)
variable (a2_sub_b2_div_c_eq_4 : (a^2 - b^2) / c = 4)

-- Prove the value of c
theorem find_value_of_c (h₁ : tan A = 7 * tan B) (h₂ : (a^2 - b^2) / c = 4) : c = 16 / 3 :=
by
  sorry

end find_value_of_c_l12_12438


namespace tangent_line_to_curve_has_b_l12_12559

theorem tangent_line_to_curve_has_b :
  ∃ b : ℝ, ∀ (x y : ℝ), y = (1 / 2) * x - b ↔ (y = - (1 / 2) * x + real.log x) 
  ∧ (derivative (λ x, - (1 / 2) * x + real.log x)) x = (1 / 2) → b = 1 :=
begin
  sorry
end

end tangent_line_to_curve_has_b_l12_12559


namespace arithmetic_sequence_general_term_l12_12576

theorem arithmetic_sequence_general_term
    (a : ℕ → ℤ)
    (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
    (h_mean_26 : (a 2 + a 6) / 2 = 5)
    (h_mean_37 : (a 3 + a 7) / 2 = 7) :
    ∀ n, a n = 2 * n - 3 := 
by
  sorry

end arithmetic_sequence_general_term_l12_12576


namespace age_double_in_years_l12_12641

theorem age_double_in_years (S M X: ℕ) (h1: M = S + 22) (h2: S = 20) (h3: M + X = 2 * (S + X)) : X = 2 :=
by 
  sorry

end age_double_in_years_l12_12641


namespace award_distribution_l12_12893

theorem award_distribution (students : Finset ℕ) (awards : Finset ℕ) :
  |students| = 4 → 
  |awards| = 6 →
  (∀ s ∈ students, 1 ≤ s ∧ s ≤ 3) →
  (∃ d1 d2 : Finset ℕ, d1 ∪ d2 = awards ∧
                     d1.card = 2 ∧ d2.card = 2 ∧ 
                     (∃ d3 d4 : Finset ℕ, d3 ∪ d4 = awards ∧
                                         d3.card = 1 ∧ d4.card = 1) ∨
                     (∃ d1 d2 d3 : Finset ℕ, d1 ∪ d2 ∪ d3 = awards ∧ 
                                           d1.card = 3 ∧ d2.card = 2 ∧ d3.card = 1)) →
  ∃! ways, ways = 780 := by
  sorry

end award_distribution_l12_12893


namespace triangle_area_l12_12598

theorem triangle_area : 
  let p1 := (3, 2)
  let p2 := (3, -4)
  let p3 := (12, 2)
  let height := |2 - (-4)|
  let base := |12 - 3|
  let area := (1 / 2) * base * height
  area = 27 := sorry

end triangle_area_l12_12598


namespace monotonicity_and_tangent_intersections_l12_12347

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12347


namespace jack_jill_meet_distance_l12_12826

noncomputable def position_uphill (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def position_downhill (speed : ℝ) (time : ℝ) (start_time : ℝ) : ℝ :=
  6 - speed * (time - start_time)

theorem jack_jill_meet_distance
  (jack_start_diff : ℝ) (jack_speed_up : ℝ) (jack_speed_down : ℝ)
  (jill_speed_up : ℝ) (jill_speed_down : ℝ) : ℝ :=
  let jack_time_up : ℝ := 6 / jack_speed_up
  let jill_time_up : ℝ := 6 / jill_speed_up
  let jack_position : ℝ → ℝ :=
    λ t, if t ≤ jack_time_up then position_uphill jack_speed_up t
          else position_downhill jack_speed_down t jack_time_up
  let jill_position : ℝ → ℝ :=
    λ t, if t ≤ jill_time_up then position_uphill jill_speed_up (t - jack_start_diff)
          else position_downhill jill_speed_down t jill_time_up
  let meet_position := ∀ t, jack_position t = jill_position t
  (6 - (2137 / 378)) = 68 / 21 := by
  sorry

#eval jack_jill_meet_distance (1 / 6) 14 20 16 24

end jack_jill_meet_distance_l12_12826


namespace fixed_errors_correct_l12_12193

-- Conditions
def total_lines_of_code : ℕ := 4300
def lines_per_debug : ℕ := 100
def errors_per_debug : ℕ := 3

-- Question: How many errors has she fixed so far?
theorem fixed_errors_correct :
  (total_lines_of_code / lines_per_debug) * errors_per_debug = 129 := 
by 
  sorry

end fixed_errors_correct_l12_12193


namespace balance_possible_l12_12510

-- Defining the weights as a list of logarithms
def weights : List ℝ := List.map Real.log (List.range' 3 77)

-- Our goal is to prove that we can balance these weights on a scale such that the difference is minimal
theorem balance_possible :
  ∃ (L R : List ℝ), List.sum L = List.sum R := 
sorry

end balance_possible_l12_12510


namespace globe_surface_area_and_volume_l12_12109

theorem globe_surface_area_and_volume (d : ℝ) (hd : d = 10) :
  let r := d / 2 in
  let surface_area := 4 * Real.pi * r^2 in
  let volume := (4 / 3) * Real.pi * r^3 in
  surface_area = 100 * Real.pi ∧ volume = (500 / 3) * Real.pi :=
by
  have hr : r = 5 := by rw [hd, r]; norm_num
  have hs : surface_area = 100 * Real.pi := by calc
    surface_area = 4 * Real.pi * (5:ℝ)^2 : by rw [hr, surface_area]; norm_num
               _ = 100 * Real.pi       : by norm_num
  have hv : volume = (500 / 3) * Real.pi := by calc
    volume = (4 / 3) * Real.pi * (5:ℝ)^3 : by rw [hr, volume]; norm_num
          _ = (500 / 3) * Real.pi         : by norm_num
  exact ⟨hs, hv⟩

end globe_surface_area_and_volume_l12_12109


namespace monotonicity_and_tangent_intersections_l12_12356

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12356


namespace part1_monotonicity_part2_tangent_intersection_l12_12256

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12256


namespace unique_intersection_value_l12_12671

noncomputable def intersect_at_one_point (a : ℝ) : Prop :=
  let discriminant := 49 - 16 * a in
  discriminant = 0

theorem unique_intersection_value :
  intersect_at_one_point (49 / 16) :=
by
  let a := 49 / 16
  sorry

end unique_intersection_value_l12_12671


namespace problem_alpha_90_to_180_problem_alpha_270_to_360_l12_12972

theorem problem_alpha_90_to_180 (α : ℝ) (h1 : real.pi / 2 < α) (h2 : α < real.pi) :
  (real.sqrt ((1 + real.sin α) / (1 - real.sin α)) - real.sqrt ((1 - real.sin α) / (1 + real.sin α))
  = -2 * real.tan α) :=
sorry

theorem problem_alpha_270_to_360 (α : ℝ) (h1 : 3 * real.pi / 2 < α) (h2 : α < 2 * real.pi) :
  (real.sqrt ((1 + real.sin α) / (1 - real.sin α)) - real.sqrt ((1 - real.sin α) / (1 + real.sin α))
  = 2 * real.tan α) :=
sorry

end problem_alpha_90_to_180_problem_alpha_270_to_360_l12_12972


namespace monotonicity_tangent_points_l12_12270

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12270


namespace monotonicity_f_tangent_points_l12_12370

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12370


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12391

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12391


namespace ratio_of_focus_and_vertex_l12_12489

def parabola : Type := { p : ℝ × ℝ // p.snd = 4 * p.fst ^ 2 }

def vertex (V : parabola) : ℝ × ℝ := (0, 0)

def focus (F : parabola) : ℝ × ℝ := (0, 1/16)

def midpoint (a b : parabola) : ℝ × ℝ :=
  ((a.val.1 + b.val.1) / 2, 2 * (a.val.1 ^ 2 + b.val.1 ^ 2))

def locus (M : parabola → parabola → ℝ × ℝ) : Type :=
  { q : ℝ × ℝ // ∃ a b : parabola, M a b = q }

def vertex_locus (V : locus) : ℝ × ℝ := (0, 1/4)

def focus_locus (F : locus) : ℝ × ℝ := (0, 17/64)

def V1V2 := 1/4
def F1F2 := 1/64

theorem ratio_of_focus_and_vertex : F1F2 / V1V2 = 1 / 16 :=
by
  simp [F1F2, V1V2, div_eq_inv_mul, inv_eq_one_div]
  norm_num

end ratio_of_focus_and_vertex_l12_12489


namespace find_inverse_of_f_at_0_l12_12498

def f (x : ℝ) : ℝ := 2 * log (2 * x - 1)

theorem find_inverse_of_f_at_0 : (∃ x : ℝ, f x = 0) → ∃ x : ℝ, x = 1 :=
by
  sorry

end find_inverse_of_f_at_0_l12_12498


namespace isosceles_triangle_l12_12450

variable {D E F G : Type}
variable [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace G]

noncomputable def length (a b : D) : Real := dist a b

theorem isosceles_triangle {DE DF EG GF DG : Real} 
  (h1 : DE = 5)
  (h2 : DF = 5)
  (h3 : EG = 2 * GF)
  (h4 : DG ^ 2 + (2 * GF) ^ 2 = 25)
  (h5 : DG ^ 2 + GF ^ 2 = 25) : 
  length E F = 5 * Real.sqrt 3 :=
sorry

end isosceles_triangle_l12_12450


namespace integer_solutions_count_l12_12415

open Int

def number_of_solutions : ℤ :=
  (Finset.range 31).filter (λ n, -15 + n < 3 ∧ -15 + n + 4 > 0 ∧ -15 + n + 8 < 0 ∨
                             -15 + n < 3 ∧ -15 + n + 4 > 0 ∧ -15 + n + 8 > 0 ∨
                             -15 + n < 3 ∧ -15 + n + 4 < 0 ∧ -15 + n + 8 > 0).card

theorem integer_solutions_count :
  number_of_solutions = 14 :=
  sorry

end integer_solutions_count_l12_12415


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12314

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12314


namespace average_weight_group_when_D_joins_l12_12898

variables (A B C D E : ℤ) (X : ℤ)

-- Condition 1
def condition1 : Prop := (A + B + C) / 3 = 60

-- Condition 2
def condition2 : Prop := A = 87

-- Condition 3
def condition3 : Prop := 87 + B + C + D = 4 * X

-- Condition 4
def condition4 : Prop := E = D + 3

-- Condition 5
def condition5 : Prop := (B + C + D + E) / 4 = 64

-- The statement we want to prove
theorem average_weight_group_when_D_joins 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4)
  (h5 : condition5) : X = 65 := 
  by sorry

end average_weight_group_when_D_joins_l12_12898


namespace monotonicity_and_tangent_intersections_l12_12348

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12348


namespace total_tickets_sold_l12_12053

-- Define the conditions
def cost_adult : ℕ := 6
def cost_kid : ℕ := 2
def total_profit : ℕ := 750
def kid_tickets : ℕ := 75

-- Define the statement we need to prove
theorem total_tickets_sold : ∃ A K, K = 75 ∧ 6 * A + 2 * K = 750 ∧ A + K = 175 :=
by {
  -- Definitions
  let A := 100,
  let K := 75,
  -- Proof
  use [A, K],
  -- Conditions from problem statement
  split,
  { refl },
  split,
  { exact rfl, },  
  calc A + K = 100 + 75 : by refl
            ... = 175  : by refl,
  sorry
}

end total_tickets_sold_l12_12053


namespace find_a_l12_12752

theorem find_a (a : ℝ) (h : {0, a} = {1, a^2 - a}) : a = 1 :=
by
  sorry

end find_a_l12_12752


namespace _l12_12990

noncomputable def chord_lengths : Prop :=
  let radius : ℝ := 6
  let dist_CD : ℝ := 3
  let dist_EF : ℝ := 4
  let length_CD := 2 * (real.sqrt (radius^2 - dist_CD^2))  -- process involving the Pythagorean theorem for chord CD
  let length_EF := 2 * (real.sqrt (radius^2 - dist_EF^2))  -- process involving the Pythagorean theorem for chord EF
  (length_CD = 6 * real.sqrt 3) ∧ (length_EF = 4 * real.sqrt 5)  -- lengths of CD and EF

#check chord_lengths -- Ensure that the definition is correct and type checks successfully

-- A formal proof would be required here, but it is outside the scope of this task.

end _l12_12990


namespace correlations_l12_12075

-- Definitions based on conditions
def student_attitude → academic_performance : Prop := sorry
def teacher_level → academic_performance : Prop := sorry
def student_height → academic_performance : Prop := sorry
def economic_condition → academic_performance : Prop := sorry

-- The "proof" of correlation
theorem correlations :
  student_attitude → academic_performance ∧ teacher_level → academic_performance :=
by
  sorry

end correlations_l12_12075


namespace length_of_first_roll_l12_12122

theorem length_of_first_roll 
  (w1 w2 : ℝ) (l2 : ℝ) (w1_eq : w1 = 5) (w2_eq : w2 = 15) (l2_eq : l2 = 75) :
  ∃ l1 : ℝ, l1 = 25 :=
by
  have prop : (l2 / w2 = 75 / 15) ∧ (w1_eq = 5) := by sorry
  sorry

end length_of_first_roll_l12_12122


namespace coefficient_of_xy3_l12_12462

theorem coefficient_of_xy3 :
  let p := (1 + 2 * (Polynomial.X : Polynomial ℚ))^6 * (1 + Polynomial.Y)^5,
      xy3_term := Polynomial.X * Polynomial.Y^3 in
  Polynomial.coeff xy3_term 120 :=
by
  sorry

end coefficient_of_xy3_l12_12462


namespace prop1_prop2_prop3_prop4_true_propositions_l12_12228

-- Definitions of the conditions
variable {l m n : Line}
variable {α β γ : Plane}

-- Propositions
theorem prop1 (h1 : m ∥ l) (h2 : l ⟂ α) : m ⟂ α :=
sorry

theorem prop2 (h1 : m ∥ l) (h2 : l ∥ α) : m ∥ α :=
sorry

theorem prop3 (h1 : α ∩ β = l) (h2 : β ∩ γ = m) (h3 : γ ∩ α = n) : l ∥ m ∥ n :=
sorry

theorem prop4 (h1 : α ∩ β = l) (h2 : β ∩ γ = m) (h3 : γ ∩ α = n) (h4 : n ∥ β) : m ∥ l :=
sorry

-- Main theorem proving that propositions ① and ④ are true given the conditions
theorem true_propositions : (prop1) ∧ ¬(prop2) ∧ ¬(prop3) ∧ (prop4) :=
by {
  split,
  { apply prop1; sorry },
  split,
  { intro h, apply prop2; sorry },
  split,
  { intro h, apply prop3; sorry },
  { apply prop4; sorry }
}

end prop1_prop2_prop3_prop4_true_propositions_l12_12228


namespace probability_of_selected_cubes_l12_12108

-- Total number of unit cubes
def total_unit_cubes : ℕ := 125

-- Number of cubes with exactly two blue faces (from edges not corners)
def two_blue_faces : ℕ := 9

-- Number of unpainted unit cubes
def unpainted_cubes : ℕ := 51

-- Calculate total combinations of choosing 2 cubes out of 125
def total_combinations : ℕ := Nat.choose total_unit_cubes 2

-- Calculate favorable outcomes: one cube with 2 blue faces and one unpainted cube
def favorable_outcomes : ℕ := two_blue_faces * unpainted_cubes

-- Calculate probability
def probability : ℚ := favorable_outcomes / total_combinations

-- The theorem we want to prove
theorem probability_of_selected_cubes :
  probability = 3 / 50 :=
by
  -- Show that the probability indeed equals 3/50
  sorry

end probability_of_selected_cubes_l12_12108


namespace find_x_for_parallel_vectors_l12_12412

-- Define the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (4, x)
def b : ℝ × ℝ := (-4, 4)

-- Define parallelism condition for two 2D vectors
def are_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Define the main theorem statement
theorem find_x_for_parallel_vectors (x : ℝ) (h : are_parallel (a x) b) : x = -4 :=
by sorry

end find_x_for_parallel_vectors_l12_12412


namespace largest_prime_divisor_of_sum_of_squares_l12_12715

theorem largest_prime_divisor_of_sum_of_squares :
  ∃ p : ℕ, prime p ∧ p ∣ (36^2 + 45^2) ∧ ∀ q : ℕ, prime q ∧ q ∣ (36^2 + 45^2) → q ≤ p := by
  sorry

end largest_prime_divisor_of_sum_of_squares_l12_12715


namespace prob_exactly_2_successes_correct_prob_at_least_2_successes_correct_prob_at_most_2_successes_correct_l12_12098

noncomputable def prob_exactly_2_successes (success_rate : ℝ) (num_shots : ℕ) : ℝ :=
  let p := success_rate
  let q := 1 - success_rate
  let n := num_shots
  let k := 2
  (nat.choose n k) * (p ^ k) * (q ^ (n - k))

theorem prob_exactly_2_successes_correct : 
  prob_exactly_2_successes 0.7 4 = 0.2646 :=
by
  unfold prob_exactly_2_successes 
  norm_num
  sorry

noncomputable def prob_at_least_2_successes (success_rate : ℝ) (num_shots : ℕ) : ℝ :=
  let p := success_rate
  let q := 1 - success_rate
  let n := num_shots
  1 - ((nat.choose n 0) * (p ^ 0) * (q ^ n) + (nat.choose n 1) * (p ^ 1) * (q ^ (n - 1)))

theorem prob_at_least_2_successes_correct : 
  prob_at_least_2_successes 0.7 4 = 0.9163 :=
by
  unfold prob_at_least_2_successes
  norm_num
  sorry

noncomputable def prob_at_most_2_successes (success_rate : ℝ) (num_shots : ℕ) : ℝ :=
  let p := success_rate
  let q := 1 - success_rate
  let n := num_shots
  ((nat.choose n 0) * (p ^ 0) * (q ^ n)) + 
  ((nat.choose n 1) * (p ^ 1) * (q ^ (n - 1))) + 
  ((nat.choose n 2) * (p ^ 2) * (q ^ (n - 2)))

theorem prob_at_most_2_successes_correct : 
  prob_at_most_2_successes 0.7 4 = 0.3483 :=
by
  unfold prob_at_most_2_successes
  norm_num
  sorry

end prob_exactly_2_successes_correct_prob_at_least_2_successes_correct_prob_at_most_2_successes_correct_l12_12098


namespace find_m_l12_12231

-- Define the quadratic equation
def quadratic_equation (x : ℂ) (m : ℝ) := x^2 - (1 - complex.i) * x + (m + 2 * complex.i) = 0

-- Define the condition for the equation having a real root
def has_real_root (m : ℝ) := ∃ x : ℝ, quadratic_equation x m

-- State the theorem to prove
theorem find_m (m : ℝ) : has_real_root m → m = -6 :=
by
  sorry

end find_m_l12_12231


namespace probability_student_less_than_25_l12_12616

def total_students : ℕ := 100

-- Percentage conditions translated to proportions
def proportion_male : ℚ := 0.48
def proportion_female : ℚ := 0.52

def proportion_male_25_or_older : ℚ := 0.50
def proportion_female_25_or_older : ℚ := 0.20

-- Definition of probability that a randomly selected student is less than 25 years old.
def probability_less_than_25 : ℚ :=
  (proportion_male * (1 - proportion_male_25_or_older)) +
  (proportion_female * (1 - proportion_female_25_or_older))

theorem probability_student_less_than_25 :
  probability_less_than_25 = 0.656 := by
  sorry

end probability_student_less_than_25_l12_12616


namespace part1_monotonicity_part2_tangent_intersection_l12_12262

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12262


namespace correct_substitution_l12_12188

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem correct_substitution (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ∈ ℝ) (h₃ : c ∈ ℝ) :
  ∀ (h : ℝ → ℝ), (∀ t : ℝ, f a b c (h t) = f a b c t) → 
  (h = (λ t, real.log t) ↔ range h = set.univ) :=
by {
  sorry
}

end correct_substitution_l12_12188


namespace sum_of_roots_l12_12965

theorem sum_of_roots (x : ℝ) (h : (x - 6)^2 = 16) : (∃ a b : ℝ, a + b = 12 ∧ (x = a ∨ x = b)) :=
by
  sorry

end sum_of_roots_l12_12965


namespace average_percentage_for_all_students_l12_12428

variables (n1 n2 : ℕ) (avg1 avg2 : ℝ)

noncomputable def total_percentage : ℝ := n1 * avg1 + n2 * avg2

noncomputable def average_percentage : ℝ := total_percentage n1 n2 avg1 avg2 / (n1 + n2)

theorem average_percentage_for_all_students :
  (n1 = 15) → (avg1 = 73) → (n2 = 10) → (avg2 = 88) → average_percentage n1 n2 avg1 avg2 = 79 :=
by
  intros,
  sorry

end average_percentage_for_all_students_l12_12428


namespace debbie_number_l12_12890

theorem debbie_number : ∃ n ∈ finset.range (1, 1201), 
  (n ≠ 3 + 4 * k  ∀ k ∈ nat ∧ n ≠ 3 + 4 * k + 4 * m * 7 ∀ k, m ∈ nat ∧ n ≠ 11 + 16 * l ∀ l ∈ nat) -> 
  n = 1187 := 
sorry

end debbie_number_l12_12890


namespace find_x_l12_12777

noncomputable def vector_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_x (x : ℝ) :
  let a := (1, 2*x + 1)
  let b := (2, 3)
  (vector_parallel a b) → x = 1 / 4 :=
by
  intro h
  have h_eq := h
  sorry  -- proof is not needed as per instruction

end find_x_l12_12777


namespace inverse_geometric_sequence_l12_12031

-- Define that a, b, c form a geometric sequence
def geometric_sequence (a b c : ℝ) := b^2 = a * c

-- Define the theorem: if b^2 = a * c, then a, b, c form a geometric sequence
theorem inverse_geometric_sequence (a b c : ℝ) (h : b^2 = a * c) : geometric_sequence a b c :=
by
  sorry

end inverse_geometric_sequence_l12_12031


namespace sequence_inequality_l12_12213

noncomputable def a_seq : ℕ → ℝ
| 0     => 1
| (n+1) => (n+2) / (a_seq n)

theorem sequence_inequality (n : ℕ) :
  (∑ i in Finset.range (n + 1), 1 / a_seq i) ≥ 2 * (Real.sqrt (n + 2) - 1) :=
sorry

end sequence_inequality_l12_12213


namespace find_a_f_monotonically_increasing_find_range_of_lg_x_l12_12401

theorem find_a 
  (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_f_def : ∀ x, f x = a * 3^x + 1 / 3^x) :
  a = 1 :=
  sorry

theorem f_monotonically_increasing 
  (f : ℝ → ℝ)
  (h_f_def : ∀ x, f x = 3^x + 1 / 3^x) :
  (∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2) :=
  sorry

theorem find_range_of_lg_x 
  (f : ℝ → ℝ)
  (h_f_def : ∀ x, f x = 3^x + 1 / 3^x)
  (h_monotone : ∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2) :
  (∀ x, f (Real.log10 x) < f (Real.log10 10) ↔ 1/10 < x ∧ x < 10) :=
  sorry

end find_a_f_monotonically_increasing_find_range_of_lg_x_l12_12401


namespace selling_price_equal_loss_l12_12571

theorem selling_price_equal_loss (CP SP SP' : ℝ) (hCP : CP = 49) (hSP : SP = 56)  
  (profit : SP - CP = CP - SP') : 
  SP' = 42 :=
by 
  have h1 : 56 - 49 = CP - SP', by rw [hCP, hSP] at profit; exact profit
  have h2 : 7 = 49 - SP', by norm_num at h1; exact h1
  rw sub_eq_iff_eq_add at h2
  exact h2.symm

end selling_price_equal_loss_l12_12571


namespace min_value_a2b3c_l12_12623

variable {a b c : ℝ}

theorem min_value_a2b3c (ha : a > 0) (hb : b > 0) (hc : c > 0) (H : 1/a + 2/b + 3/c = 2) :
  a + 2b + 3c = 18 ↔ a = 3 ∧ b = 3 ∧ c = 3 :=
  by 
    sorry

end min_value_a2b3c_l12_12623


namespace opposite_sides_of_line_l12_12234

theorem opposite_sides_of_line (a : ℝ) :
  let p1 := (2, -1)
  let p2 := (-3, 2)
  let line_eq (x y : ℝ) := x - 2 * y + a
  (line_eq p1.1 p1.2) * (line_eq p2.1 p2.2) < 0 ↔ -4 < a ∧ a < 7 :=
by
  let p1 := (2, -1)
  let p2 := (-3, 2)
  let line_eq (x y : ℝ) := x - 2 * y + a
  have h1 : line_eq p1.1 p1.2 = 4 + a := by simp [line_eq]
  have h2 : line_eq p2.1 p2.2 = -7 + a := by simp [line_eq]
  calc
    (4 + a) * (-7 + a) < 0 ↔ a^2 - 3*a - 28 < 0   := by ring_nf
                      ... ↔ -4 < a ∧ a < 7         := sorry


end opposite_sides_of_line_l12_12234


namespace complement_B_in_U_l12_12499

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x = 1}
def U : Set ℕ := A ∪ B

theorem complement_B_in_U : (U \ B) = {2, 3} := by
  sorry

end complement_B_in_U_l12_12499


namespace polynomial_degree_and_coefficient_l12_12224

theorem polynomial_degree_and_coefficient 
  (f : ℕ → ℕ → ℤ)
  (h_degree : ∀ x y, degree (f x y • x^2 * y ^ (4 + 1) + x * y^2 + -3 * x^5 + -6) = 7)
  (h_coeff : coeff (f 2 5) = 8) :
  let m := 4,
      n := -8 in 
  (m = 4 ∧ n = -8) ∧ 
  let p := -3 * x^5 + 8 * x^2 * y^5 + x * y^2 + -6 in p = -3 * x^5 + 8 * x^2 * y^5 + x * y^2 + -6 :=
by
  sorry

end polynomial_degree_and_coefficient_l12_12224


namespace monotonicity_tangent_points_l12_12271

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12271


namespace max_value_function_l12_12024

theorem max_value_function (x : ℝ) (h : x < 0) : 
  ∃ y_max, (∀ x', x' < 0 → (x' + 4 / x') ≤ y_max) ∧ y_max = -4 := 
sorry

end max_value_function_l12_12024


namespace initial_apples_l12_12524

theorem initial_apples (X : ℕ) (h : X - 2 + 3 = 5) : X = 4 :=
sorry

end initial_apples_l12_12524


namespace baguettes_sold_third_batch_l12_12542

-- Definitions of the conditions
def daily_batches : ℕ := 3
def baguettes_per_batch : ℕ := 48
def baguettes_sold_first_batch : ℕ := 37
def baguettes_sold_second_batch : ℕ := 52
def baguettes_left : ℕ := 6

theorem baguettes_sold_third_batch : 
  daily_batches * baguettes_per_batch - (baguettes_sold_first_batch + baguettes_sold_second_batch + baguettes_left) = 49 :=
by sorry

end baguettes_sold_third_batch_l12_12542


namespace team_selection_ways_l12_12467

theorem team_selection_ways : 
  (nat.choose 8 4) * (nat.choose 10 3) = 8400 := 
by
  sorry

end team_selection_ways_l12_12467


namespace monotonicity_of_f_tangent_intersection_points_l12_12322

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12322


namespace seq_sum_correct_l12_12918

def seq (x : ℕ → ℝ) : Prop :=
  x 0 = 1 ∧ x 1 = 2 ∧ ∀ n ≥ 2, n * (n + 1) * x (n + 1) = n * (n - 1) * x n - (n - 2) * x (n - 1)

theorem seq_sum_correct (x : ℕ → ℝ) (h : seq x) :
  (∑ n in Finset.range 51, x n / x (n + 1)) + 1327.5 = 1327.5 :=
by
  sorry

end seq_sum_correct_l12_12918


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12305

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12305


namespace find_y_l12_12615

theorem find_y (x y : Int) (h1 : x + y = 280) (h2 : x - y = 200) : y = 40 := 
by 
  sorry

end find_y_l12_12615


namespace monotonicity_of_f_tangent_intersection_l12_12283

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12283


namespace distinct_triangles_count_l12_12414

-- Define the 3x3 grid as a set of points
def points : finset (ℕ × ℕ) :=
  {(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)}

-- Total number of ways to choose 3 points out of 9
def total_combinations : ℕ :=
  nat.choose 9 3

-- Number of degenerate (collinear) cases - 3 rows + 3 columns + 2 diagonals
def degenerate_cases : ℕ :=
  3 + 3 + 2

-- Number of valid triangles
def valid_triangles : ℕ :=
  total_combinations - degenerate_cases

theorem distinct_triangles_count : valid_triangles = 76 :=
  by
    trivial -- Placeholder for the actual proof

end distinct_triangles_count_l12_12414


namespace linear_function_through_points_area_of_triangle_OAB_l12_12748

noncomputable def linearFunction (k b : ℝ) (x : ℝ) := k * x + b

theorem linear_function_through_points (k b : ℝ) :
  (linearFunction k b 1 = 2) ∧ (linearFunction k b (-1) = 6) → 
  (k = -2) ∧ (b = 4) :=
by
  sorry

theorem area_of_triangle_OAB (k b : ℝ) (h : k = -2 ∧ b = 4) :
  let A := (2 : ℝ, 0),
      B := (0, 4 : ℝ),
      O := (0, 0 : ℝ) in
  let OA := 2, OB := 4 in
  (1 / 2) * OA * OB = 4 :=
by
  sorry

end linear_function_through_points_area_of_triangle_OAB_l12_12748


namespace min_possible_frac_l12_12431

theorem min_possible_frac (x A C : ℝ) (hx : x ≠ 0) (hC_pos : 0 < C) (hA_pos : 0 < A)
  (h1 : x^2 + (1/x)^2 = A)
  (h2 : x - 1/x = C)
  (hC : C = Real.sqrt 3):
  A / C = (5 * Real.sqrt 3) / 3 := by
  sorry

end min_possible_frac_l12_12431


namespace geometric_sequence_proof_l12_12219

theorem geometric_sequence_proof (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (n : ℕ) 
    (h1 : a 2 = 8) 
    (h2 : S 3 = 28) 
    (h3 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
    (h4 : ∀ n, a n = a 1 * q^(n-1)) 
    (h5 : q > 1) :
    (∀ n, a n = 2^(n + 1)) ∧ (∀ n, (a n)^2 > S n + 7) := sorry

end geometric_sequence_proof_l12_12219


namespace monotonicity_tangent_points_l12_12275

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12275


namespace cube_dot_path_length_eq_l12_12638

def cube_edge := 2
def dot_initial_position : ℝ := 1
def full_rotation_path_length (r : ℝ) := 2 * Real.pi * r

theorem cube_dot_path_length_eq :
  full_rotation_path_length dot_initial_position = 2 * Real.pi := by
  sorry

end cube_dot_path_length_eq_l12_12638


namespace time_to_run_square_field_l12_12413

theorem time_to_run_square_field (side_length : ℝ) (running_speed_kmph : ℝ) (h₁ : side_length = 50) (h₂ : running_speed_kmph = 10) : 
  let running_speed_mps := running_speed_kmph * (1000 / 3600) in
  let perimeter := 4 * side_length in
  let time := perimeter / running_speed_mps in
  time ≈ 72 := sorry

end time_to_run_square_field_l12_12413


namespace max_abc_value_l12_12833

variables (a b c : ℕ)

theorem max_abc_value : 
  (a > 0) → (b > 0) → (c > 0) → a + 2 * b + 3 * c = 100 → abc ≤ 6171 := 
by sorry

end max_abc_value_l12_12833


namespace min_cells_marked_l12_12065

theorem min_cells_marked (n : ℕ) (hn : 0 < n) : 
  ∃ (M : Finset (Fin n × Fin n)), 
    (M.card = n) ∧ 
    (∀ m > n / 2, 
      ∀ i j, i + m ≤ n → j + m ≤ n → 
      (∃ k ∈ M, k.1.1 = i + (m / 2) ∧ 
       ∀ d ∈ Finset.range m, (k = (⟨i + d, _⟩, ⟨j + d, _⟩)) ∨ (k = (⟨i + d, _⟩, ⟨j + m - d - 1, _⟩)))) := 
sorry

end min_cells_marked_l12_12065


namespace dot_product_CG_AD_l12_12747

noncomputable def point := ℝ × ℝ

def midpoint (A B : point) : point :=
  ((fst A + fst B) / 2, (snd A + snd B) / 2)

def centroid (A B C : point) : point :=
  ((fst A + fst B + fst C) / 3, (snd A + snd B + snd C) / 3)

def vector (P Q : point) : point :=
  (fst Q - fst P, snd Q - snd P)

def dot_product (v w : point) : ℝ :=
  fst v * fst w + snd v * snd w

theorem dot_product_CG_AD (A B C : point) (h_eq_tri : dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2 ) :
  let G := centroid A B C,
      D := midpoint B C,
      CG := vector C G,
      AD := vector A D
  in dot_product CG AD = -1 :=
sorry

end dot_product_CG_AD_l12_12747


namespace no_nonzero_sequence_perfect_square_l12_12160

theorem no_nonzero_sequence_perfect_square :
  ¬ ∃ (a : ℕ → ℕ) (N : ℕ) (h : N > 0),
      (∀ (i : ℕ), a i ≠ 0) ∧
      (∀ (k : ℕ), k > N → ∃ (x : ℕ), x ^ 2 = (finset.range k).sum (λ i, a (i + 1) * 10 ^ i)) :=
by
  sorry

end no_nonzero_sequence_perfect_square_l12_12160


namespace monotonicity_f_tangent_points_l12_12366

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12366


namespace math_problem_l12_12536

theorem math_problem (y : ℝ) (h : sqrt (64 - y^2) - sqrt (36 - y^2) = 4) : sqrt (64 - y^2) + sqrt (36 - y^2) = 7 := 
sorry

end math_problem_l12_12536


namespace function_satisfies_conditions_l12_12172

def f (m n : ℕ) : ℕ := m * n

theorem function_satisfies_conditions :
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → 2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  (∀ m : ℕ, f m 0 = 0) ∧
  (∀ n : ℕ, f 0 n = 0) := 
by {
  sorry
}

end function_satisfies_conditions_l12_12172


namespace monotonicity_of_f_tangent_intersection_l12_12288

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12288


namespace exist_coloring_method_l12_12444

theorem exist_coloring_method 
  (m n : ℕ) 
  (paths : Finset (ℕ × ℕ)) 
  (flower_beds : Finset ℕ) 
  (color : Finset ℕ) 
  (connects : paths → ℕ × ℕ)
  (h1 : paths.card = m * n)
  (h2 : ∀ (p ∈ paths), connects p ∈ flower_beds × flower_beds)
  (h3 : color.card = m)
  (h4 : ∀ (b ∈ flower_beds), ∀ (c1 c2 ∈ color), ∃! (p ∈ paths), connects p = (b, c1) ∨ connects p = (b, c2) → c1 ≠ c2) :
  ∃ (c : paths → ℕ), (∀ (p ∈ paths), c p ∈ color) ∧ (∀ (c ∈ color), (Finset.filter (λ p, c p = c) paths).card = n) :=
sorry

end exist_coloring_method_l12_12444


namespace part_a_part_b_l12_12871

-- Definitions for the function f and conditions.
def nonnegative (f : ℝ → ℝ) (x : ℝ) : Prop := 0 ≤ f x
def bounded (f : ℝ → ℝ) : Prop := f 1 = 1
def subadditive (f : ℝ → ℝ) : Prop := ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → 0 ≤ x₂ → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≤ f x₁ + f x₂

-- Main statement encoding the problem:
theorem part_a (f : ℝ → ℝ) (x : ℝ) (h1 : ∀ x, x ∈ Icc (0:ℝ) 1 → nonnegative f x) (h2 : bounded f) (h3 : subadditive f) : x ∈ Icc (0:ℝ) 1 → f x ≤ 2 * x := sorry

theorem part_b (f : ℝ → ℝ) (x : ℝ) (h1 : ∀ x, x ∈ Icc (0:ℝ) 1 → nonnegative f x) (h2 : bounded f) (h3 : subadditive f) : ¬(∀ x ∈ Icc (0:ℝ) 1, f x ≤ 1.9 * x) :=
begin
  by_contra h,
  -- Counterexample function (as part of conclusion of part b)
  let f : ℝ → ℝ := λ x, if (1 / 2) < x then 1 else 0,
  have hx : x = 51 / 100, from sorry,
  specialize h (51 / 100) (by norm_num),
  rw [if_pos] at h,
  { linarith, },
  { norm_num }
end

end part_a_part_b_l12_12871


namespace f_five_l12_12491

def f : ℕ → ℕ
| x => if x ≥ 10 then x - 2 else f (f (x + 6))

theorem f_five : f 5 = 11 := by
  sorry

end f_five_l12_12491


namespace invariant_ratio_parallel_lines_l12_12019

theorem invariant_ratio_parallel_lines
  (A B C D M : ℝ)
  (h_parallel_AB_CD : ∀ t : ℝ, (A + t) * B = (C + t) * D)  -- A parallel condition
  (h_intersection : M ∈ (AB ∩ CD))                          -- M is intersection of AB and CD
  (h_similar_triangles : ∀ t : ℝ, A / C = B / D)            -- Similar triangles because of parallel lines
  : (A * B) / (C * D) = k :=
sorry

end invariant_ratio_parallel_lines_l12_12019


namespace acute_triangle_altitudes_eq_l12_12517

theorem acute_triangle_altitudes_eq {A B C H H1 H2 H3 : Point} (h_acute : acute_angle_triangle A B C)
  (h_orthocenter : orthocenter A B C H) (h1 : foot_of_altitude A B C H1) 
  (h2 : foot_of_altitude B A C H2) (h3 : foot_of_altitude C A B H3) :
  (distance A H * distance A H1 + distance B H * distance B H2 + distance C H * distance C H3) = 
  (1 / 2) * (distance A B ^ 2 + distance B C ^ 2 + distance C A ^ 2) := 
sorry

end acute_triangle_altitudes_eq_l12_12517


namespace arithmetic_sequence_difference_l12_12535

variable (b : ℕ → ℝ)
variable (d : ℝ)
variable (h₁ : ∑ i in (finset.range 50), b i = 150)
variable (h₂ : ∑ i in (finset.Ico 50 100), b i = 250)
variable (h₃ : ∀ n, b (n + 1) = b n + d)

theorem arithmetic_sequence_difference :
  (∃ d : ℝ, (∀ n, b (n + 1) = b n + d) ∧
    (∑ i in (finset.range 50), b i = 150) ∧
    (∑ i in (finset.Ico 50 100), b i = 250) ∧
    (d = 1 / 25)) :=
begin
  sorry
end

end arithmetic_sequence_difference_l12_12535


namespace average_salary_increase_l12_12013

theorem average_salary_increase 
  (A1 : ℕ) (num_employees : ℕ) (manager_salary : ℕ) :
  A1 = 1500 → num_employees = 20 → manager_salary = 22500 →
  let total1 := num_employees * A1 in
  let total2 := total1 + manager_salary in
  let A2 := total2 / (num_employees + 1) in
  A2 - A1 = 1000 :=
by 
  intros hA1 hNumEmp hManagerSal 
  sorry

end average_salary_increase_l12_12013


namespace ant_meet_at_QS_is_five_l12_12588

-- Condition definitions
def triangle_sides (PQ QR RP : ℕ) : Prop := 
  PQ = 7 ∧ QR = 8 ∧ RP = 9

def ant_meeting_point (P Q R S : ℕ) : Prop :=
  ∃ (PQ QR RP : ℕ), triangle_sides PQ QR RP ∧ 
  (ant_meeting_distance PQ QR RP = 12)

-- Ant meeting distance function definition
def ant_meeting_distance (PQ QR RP : ℕ) : ℕ :=
  (PQ + QR + RP) / 2

-- Theorem statement expressing the problem
theorem ant_meet_at_QS_is_five (P Q R S : ℕ) 
  (h: triangle_sides 7 8 9) (a: ant_meeting_point P Q R S) : 
  ( ∃ QS : ℕ, QS = 5 ) :=
sorry

end ant_meet_at_QS_is_five_l12_12588


namespace math_problem_l12_12858

variables {f : ℝ → ℝ} (a b c : ℝ)

-- Condition: f is defined on (0, 2π)
-- Condition: f(x) = f(2π - x) for 0 < x < π
-- Condition: f(x) sin(x) - f'(x) cos(x) < 0 for 0 < x < π

def sym_about_pi (f : ℝ → ℝ) : Prop :=
∀ x, 0 < x ∧ x < π → f(x) = f(2 * π - x)

def condition_on_f (f : ℝ → ℝ) : Prop :=
∀ x, 0 < x ∧ x < π → f(x) * (sin x) - (deriv f x) * (cos x) < 0

-- Definitions of a, b, and c based on f
noncomputable def a := 1 / 2 * f (π / 3)
noncomputable def b := 0
noncomputable def c := - (√3 / 2) * f (7 * π / 6)

-- The theorem we want to prove
theorem math_problem (h_sym : sym_about_pi f) (h_cond : condition_on_f f) : a < b ∧ b < c :=
by
  -- The proof goes here
  sorry

end math_problem_l12_12858


namespace least_three_digit_8_heavy_l12_12658

def is_8_heavy (n : ℕ) : Prop :=
  n % 8 > 5

theorem least_three_digit_8_heavy : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ is_8_heavy(n) ∧ ∀ (m : ℕ), 100 ≤ m ∧ m < 1000 ∧ is_8_heavy(m) → n ≤ m :=
by
  sorry

end least_three_digit_8_heavy_l12_12658


namespace side_lengths_of_triangle_l12_12718

noncomputable def calculate_side_lengths (S : ℝ) (x : ℝ) : ℝ × ℝ × ℝ :=
  let k := real.sqrt (S * real.tan (x / 2)) in
  let BC := 2 * k in
  let AB_AC := real.sqrt (S / (real.sin (x / 2) * real.cos (x / 2))) in
  (BC, AB_AC, AB_AC)

theorem side_lengths_of_triangle (S : ℝ) (x : ℝ) :
  ∃ (BC AB AC : ℝ), 
    (BC = 2 * real.sqrt (S * real.tan (x / 2))) ∧
    (AB = real.sqrt (S / (real.sin (x / 2) * real.cos (x / 2)))) ∧ 
    (AC = real.sqrt (S / (real.sin (x / 2) * real.cos (x / 2)))) := by
  use (2 * real.sqrt (S * real.tan (x / 2))),
      (real.sqrt (S / (real.sin (x / 2) * real.cos (x / 2)))),
      (real.sqrt (S / (real.sin (x / 2) * real.cos (x / 2))))
  split
  . 
    rfl
  . 
    split
    . 
      rfl
    . 
      rfl

end side_lengths_of_triangle_l12_12718


namespace ant_meet_at_QS_is_five_l12_12589

-- Condition definitions
def triangle_sides (PQ QR RP : ℕ) : Prop := 
  PQ = 7 ∧ QR = 8 ∧ RP = 9

def ant_meeting_point (P Q R S : ℕ) : Prop :=
  ∃ (PQ QR RP : ℕ), triangle_sides PQ QR RP ∧ 
  (ant_meeting_distance PQ QR RP = 12)

-- Ant meeting distance function definition
def ant_meeting_distance (PQ QR RP : ℕ) : ℕ :=
  (PQ + QR + RP) / 2

-- Theorem statement expressing the problem
theorem ant_meet_at_QS_is_five (P Q R S : ℕ) 
  (h: triangle_sides 7 8 9) (a: ant_meeting_point P Q R S) : 
  ( ∃ QS : ℕ, QS = 5 ) :=
sorry

end ant_meet_at_QS_is_five_l12_12589


namespace length_of_bridge_l12_12556

theorem length_of_bridge (length_train : ℝ) (speed_kmh : ℝ) (time_sec : ℝ) (speed_ms : ℝ) (total_distance : ℝ) (bridge_length : ℝ) :
  length_train = 160 →
  speed_kmh = 45 →
  time_sec = 30 →
  speed_ms = 45 * (1000 / 3600) →
  total_distance = speed_ms * time_sec →
  bridge_length = total_distance - length_train →
  bridge_length = 215 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end length_of_bridge_l12_12556


namespace profit_calculation_l12_12887

theorem profit_calculation (boxes_bought : ℕ) (cost_per_box : ℕ) (pens_per_box : ℕ) (packages_per_box : ℕ) (packages_price : ℕ) (sets_per_box : ℕ) (sets_price : ℕ) :
  boxes_bought = 12 ∧
  cost_per_box = 10 ∧
  pens_per_box = 30  ∧ 
  packages_per_box = 5  ∧ 
  packages_price = 3 ∧ 
  sets_per_box = 3 ∧
  sets_price = 2 →
  let total_cost := boxes_bought * cost_per_box in
  let total_pens := boxes_bought * pens_per_box in
  let boxes_used_for_packages := 5 in -- based on given conditions
  let packages := boxes_used_for_packages * packages_per_box in
  let revenue_from_packages := packages * packages_price in
  let boxes_left := boxes_bought - boxes_used_for_packages in
  let pens_left := boxes_left * pens_per_box in
  let sets := pens_left / sets_per_box in
  let revenue_from_pens := sets * sets_price in
  let total_revenue := revenue_from_packages + revenue_from_pens in
  let profit := total_revenue - total_cost in
  profit = 95 :=
by 
  intros;
  -- Ensure that all declared variables and their calculations are consistent with the problem conditions shown above
  sorry

end profit_calculation_l12_12887


namespace algebraic_expression_value_l12_12786

noncomputable def a : ℝ := Real.sqrt 6 + 1
noncomputable def b : ℝ := Real.sqrt 6 - 1

theorem algebraic_expression_value :
  a^2 + a * b = 12 + 2 * Real.sqrt 6 :=
sorry

end algebraic_expression_value_l12_12786


namespace prime_square_minus_one_divisible_by_24_l12_12516

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (h_prime : Prime p) (h_gt_3 : p > 3) : 
  ∃ k : ℕ, p^2 - 1 = 24 * k := by
sorry

end prime_square_minus_one_divisible_by_24_l12_12516


namespace rational_inequality_condition_l12_12857

theorem rational_inequality_condition (a b : ℚ) (h : | |a| + a - b | < | a + | a - b | |) : a > 0 ∧ b > 0 :=
sorry

end rational_inequality_condition_l12_12857


namespace coefficient_of_x2_in_expansion_l12_12018

-- Define the polynomials p(x) and q(x)
def p (x : ℝ) := x^2 + x + 1
def q (x : ℝ) := (1 - x)^6

-- Define the expansion result polynomial r(x)
noncomputable def r (x : ℝ) := p(x) * q(x)

-- Statement to prove
theorem coefficient_of_x2_in_expansion : coeff (r x) x^2 = 10 := sorry

end coefficient_of_x2_in_expansion_l12_12018


namespace percentage_above_wholesale_cost_l12_12120

def wholesale_cost : ℝ := 200
def paid_price : ℝ := 228
def discount_rate : ℝ := 0.05

theorem percentage_above_wholesale_cost :
  ∃ P : ℝ, P = 20 ∧ 
    paid_price = (1 - discount_rate) * (wholesale_cost + P/100 * wholesale_cost) :=
by
  sorry

end percentage_above_wholesale_cost_l12_12120


namespace n19_minus_n7_div_30_l12_12878

theorem n19_minus_n7_div_30 (n : ℕ) (h : 0 < n) : 30 ∣ (n^19 - n^7) :=
sorry

end n19_minus_n7_div_30_l12_12878


namespace collinear_probability_in_rectangular_array_l12_12821

noncomputable def prob_collinear (total_dots chosen_dots favorable_sets : ℕ) : ℚ :=
  favorable_sets / (Nat.choose total_dots chosen_dots)

theorem collinear_probability_in_rectangular_array :
  prob_collinear 20 4 2 = 2 / 4845 :=
by
  sorry

end collinear_probability_in_rectangular_array_l12_12821


namespace sequence_sum_eq_10_over_11_implies_10_l12_12910

theorem sequence_sum_eq_10_over_11_implies_10
  (a : ℕ → ℚ)
  (ha : ∀ n : ℕ, 0 < n → a n = 1 / (n * (n + 1)))
  (h_sum : ∑ k in Finset.range n, a (k + 1) = 10 / 11) :
  n = 10 := by
  sorry

end sequence_sum_eq_10_over_11_implies_10_l12_12910


namespace monotonicity_of_f_tangent_intersection_l12_12284

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12284


namespace speed_of_train_l12_12655

open Real

-- Define the conditions as given in the problem
def length_of_bridge : ℝ := 650
def length_of_train : ℝ := 200
def time_to_pass_bridge : ℝ := 17

-- Define the problem statement which needs to be proved
theorem speed_of_train : (length_of_bridge + length_of_train) / time_to_pass_bridge = 50 :=
by
  sorry

end speed_of_train_l12_12655


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12398

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12398


namespace lucy_l12_12500

-- Define rounding function to nearest ten
def round_to_nearest_ten (x : Int) : Int :=
  if x % 10 < 5 then x - x % 10 else x + (10 - x % 10)

-- Define the problem with given conditions
def lucy_problem : Prop :=
  let sum := 68 + 57
  round_to_nearest_ten sum = 130

-- Statement of proof problem
theorem lucy's_correct_rounded_sum : lucy_problem := by
  sorry

end lucy_l12_12500


namespace slope_of_line_l12_12179

-- Definition of the line equation
def lineEquation (x y : ℝ) : Prop := 4 * x - 7 * y = 14

-- The statement that we need to prove
theorem slope_of_line : ∀ x y, lineEquation x y → ∃ m, m = 4 / 7 :=
by {
  sorry
}

end slope_of_line_l12_12179


namespace smallest_degree_polynomial_l12_12005

noncomputable def polynomial_with_roots (a b c d : ℚ) : ℚ[X] :=
  (X - (a - real.sqrt b)) * (X - (a + real.sqrt b)) *
  (X - (-a - real.sqrt b)) * (X - (-a + real.sqrt b)) *
  (X - (c + real.sqrt d)) * (X - (c - real.sqrt d))

theorem smallest_degree_polynomial :
  ∃ p : ℚ[X], 
    (polynomial_with_roots 2 5 3 11) = p ∧
    p.degree = 6 :=
by
  sorry

end smallest_degree_polynomial_l12_12005


namespace angle_AD_BE_l12_12999

theorem angle_AD_BE {
  A B C E D : Type*,
  -- Triangle ABC is a right triangle
  (h₁ : ∠ACB = 90°)
  -- Points E and D are where the perpendicular from hypotenuse AB intersects AC and BC respectively
  (h₂ : perpendicular A B AC E)
  (h₃ : perpendicular A B BC D)
  -- Lines AD and BE we want to find the angle between these lines
  (h₄ : line A D)
  (h₅ : line B E) 
}: 
  angle_between_lines h₄ h₅ = 90° :=
sorry

end angle_AD_BE_l12_12999


namespace largest_nonrepresentable_integer_l12_12979

theorem largest_nonrepresentable_integer :
  (∀ a b : ℕ, 8 * a + 15 * b ≠ 97) ∧ (∀ n : ℕ, n > 97 → ∃ a b : ℕ, n = 8 * a + 15 * b) :=
sorry

end largest_nonrepresentable_integer_l12_12979


namespace cost_per_square_foot_white_washing_l12_12547

def dimensions (L W H : ℝ) := (L, W, H)
def door (dL dW : ℝ) := (dL, dW)
def window (wL wW : ℝ) := (wL, wW)
def num_windows := 3
def cost (C : ℝ) := C

theorem cost_per_square_foot_white_washing
    (L W H dL dW wL wW total_cost : ℝ)
    (room : dimensions L W H)
    (door : door dL dW)
    (window : window wL wW)
    (num_windows : ℕ := 3)
    (total_cost := 8154) :
    (total_cost / ((2 * (L * H) + 2 * (W * H)) - (dL * dW + num_windows * (wL * wW)))) = 9 := 
    sorry

end cost_per_square_foot_white_washing_l12_12547


namespace cowboy_must_travel_l12_12637

noncomputable def cowboy_distance : ℝ :=
  let P := (0 : ℝ, 0 : ℝ)  -- Cowboy's starting point (0, 0) because it’s relative
  let river_y := 3         -- River's y-coordinate
  let P_river := (0, river_y)  -- Reflection point across the river
  let Q := (10, 6)         -- Cabin’s location
  let distance := 3 + real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P_river.2) ^ 2) -- Total distance
  distance

theorem cowboy_must_travel :
  cowboy_distance = 3 + real.sqrt 109 :=
by
  sorry

end cowboy_must_travel_l12_12637


namespace sum_of_digits_of_max_4_digit_product_24_is_13_l12_12488

theorem sum_of_digits_of_max_4_digit_product_24_is_13 :
  ∃ (M : ℕ), (1000 ≤ M ∧ M < 10000) ∧ (digits M).prod = 24 ∧ (digits M).sum = 13 := sorry

end sum_of_digits_of_max_4_digit_product_24_is_13_l12_12488


namespace probability_rational_terms_non_adjacent_l12_12673

noncomputable def probability_non_adjacent_rational_terms : ℚ :=
  let n := 8
  let rational_positions := [0, 4, 8]
  let total_positions := n + 1
  let total_permutations := fact total_positions
  let irrational_permutations := fact (total_positions - rational_positions.length)
  let combinatorial_gaps := nat.choose (total_positions - rational_positions.length + 1) rational_positions.length
  irrational_permutations * combinatorial_gaps / total_permutations

theorem probability_rational_terms_non_adjacent :
  probability_non_adjacent_rational_terms = 5 / 24 :=
sorry

end probability_rational_terms_non_adjacent_l12_12673


namespace range_of_m_l12_12776

def vector.ab (m : ℝ) := (m, 1 : ℝ)
def vector.bc (m : ℝ) := (2 - m, -4 : ℝ)
def vector.ac := (2, -3 : ℝ)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem range_of_m (m : ℝ) : dot_product (vector.ab m) vector.ac > 11 ↔ m > 7 :=
by
  -- This is a placeholder for the proof.
  sorry

end range_of_m_l12_12776


namespace toothpicks_in_large_triangle_l12_12446

theorem toothpicks_in_large_triangle (n : ℕ) (h : n = 1001) : 
  let T := (n * (n + 1)) / 2 in 
  let toothpicks := 3 * T / 2 in 
  toothpicks = 752252 :=
by {
  have n_eq : n = 1001 := h,
  let T := (n * (n + 1)) / 2,
  have T_calc : T = 501501 := by {
    rw n_eq,
    calc
      T = (1001 * 1002) / 2 : by rw n_eq
      ... = 501501 : by norm_num,
  },
  
  let toothpicks := 3 * T / 2,
  have toothpicks_calc : toothpicks = 752252 := by {
    rw T_calc,
    calc
      toothpicks = 3 * 501501 / 2 : by rw T_calc
      ... = 752252 : by norm_num,
  },
  
  exact toothpicks_calc,
}

end toothpicks_in_large_triangle_l12_12446


namespace a_n_fifth_term_l12_12212

theorem a_n_fifth_term :
  (∀ n : ℕ, S (n + 1) = 2 * (n + 1) * (n + 2)) →
  (∀ n : ℕ, a (n + 1) = S (n + 1) - S n) →
  a 5 = 20 :=
by
  sorry

end a_n_fifth_term_l12_12212


namespace smallest_n_for_mod7_multiple_l12_12533

theorem smallest_n_for_mod7_multiple (x y : ℤ) (h1 : x ≡ 3 [MOD 7]) (h2 : y ≡ -3 [MOD 7]) :
  ∃ n, n = 5 ∧ (x^2 + x * y + y^2 + n) % 7 = 0 :=
by {
  -- Skipping the proof body
  sorry,
}

end smallest_n_for_mod7_multiple_l12_12533


namespace monotonicity_and_tangent_intersections_l12_12352

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12352


namespace theta_in_range_l12_12766

theorem theta_in_range (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π)
  (h3 : ∀ x : ℝ, (cos θ) * x^2 - (4 * sin θ) * x + 6 > 0) :
  θ < π / 3 :=
sorry

end theta_in_range_l12_12766


namespace find_C_l12_12081

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 700) 
  (h2 : A + C = 300) 
  (h3 : B + C = 600) 
  : C = 200 := sorry

end find_C_l12_12081


namespace library_table_count_l12_12912

def base6_to_base10 (n : Nat) : Nat :=
  let d0 := (n % 10)
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 * 36 + d1 * 6 + d0 

theorem library_table_count (chairs people_per_table : Nat) (h1 : chairs = 231) (h2 : people_per_table = 3) :
    Nat.ceil ((base6_to_base10 chairs) / people_per_table) = 31 :=
by
  sorry

end library_table_count_l12_12912


namespace general_term_of_geometric_sequence_l12_12206

noncomputable def geometric_sequence (n : ℕ) : ℝ := sorry

theorem general_term_of_geometric_sequence:
  let a_1 := geometric_sequence 1,
      a_2 := geometric_sequence 2,
      a_3 := geometric_sequence 3,
      a_4 := geometric_sequence 4,
      S_n (n : ℕ) := ∑ i in finset.range n, geometric_sequence i.succ in
  (a_1 + a_3 = 5 / 2) →
  (a_2 + a_4 = 5 / 4) →
  ∀ n : ℕ, geometric_sequence n = 2 ^ (2 - n) :=
begin
  intros,
  sorry
end

end general_term_of_geometric_sequence_l12_12206


namespace arithmetic_sequence_general_formula_sum_bn_first_n_terms_no_n_for_1440_sum_l12_12740

theorem arithmetic_sequence_general_formula (d : ℝ) (h1 : d ≠ 0) (a1 : ℝ)
  (h2 : 4 * a1 + 6 * d = 20)
  (h3 : (a1 + d)^2 = a1 * (a1 + 3 * d)) :
  ∃ (a : ℕ → ℝ), a 1 = 2 ∧ ∀ (n : ℕ), a n = 2 * n := sorry

theorem sum_bn_first_n_terms (n : ℕ) 
  (a : ℕ → ℝ) (h : ∀ (n : ℕ), a n = 2 * n) : 
  ∑ i in Finset.range n, i * (4 : ℝ) ^ i = 
  (n / 3) * 4^(n+1) - 4 / 9 * (4^n - 1) := sorry

theorem no_n_for_1440_sum (n : ℕ) 
  (S : ℕ → ℝ) 
  (h : ∀ n, S n = (n / 3) * 4^(n+1) - 4 / 9 * (4^n - 1)) : 
  ¬ ∃ n, n > 0 ∧ S n = 1440 := sorry

end arithmetic_sequence_general_formula_sum_bn_first_n_terms_no_n_for_1440_sum_l12_12740


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12392

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12392


namespace floor_e_eq_2_l12_12165

noncomputable def e_approx : ℝ := 2.71828

theorem floor_e_eq_2 : ⌊e_approx⌋ = 2 :=
sorry

end floor_e_eq_2_l12_12165


namespace monotonicity_and_tangent_intersections_l12_12355

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12355


namespace total_cost_in_currency_B_l12_12986

noncomputable def cost_per_cap_A : ℝ := 2
noncomputable def discount_rate : ℝ := 0.10
noncomputable def conversion_rate : ℝ := 3

-- Proving the total cost for 6 bottle caps in currency B is 33.60
theorem total_cost_in_currency_B (n : ℝ) (d : ℝ) (c : ℝ) : 
  (n = 6) → 
  (d = discount_rate) → 
  (c = conversion_rate) → 
  ((4 * cost_per_cap_A) - (d * (4 * cost_per_cap_A)) + (2 * cost_per_cap_A)) * c = 33.60 := 
by
  intro n hn
  intro d hd
  intro c hc
  rw [hn, hd, hc]
  calc
    ((4 * cost_per_cap_A) - (discount_rate * (4 * cost_per_cap_A)) + (2 * cost_per_cap_A)) * conversion_rate
        = ((4 * 2) - (0.10 * (4 * 2)) + (2 * 2)) * 3 : by congr
    ... = (8 - 0.8 + 4) * 3 : by norm_num
    ... = 11.2 * 3 : by norm_num
    ... = 33.6 : by norm_num

end total_cost_in_currency_B_l12_12986


namespace total_money_raised_l12_12114

-- Given conditions:
def tickets_sold : Nat := 25
def ticket_price : ℚ := 2
def num_donations_15 : Nat := 2
def donation_15 : ℚ := 15
def donation_20 : ℚ := 20

-- Theorem statement proving the total amount raised is $100
theorem total_money_raised
  (h1 : tickets_sold = 25)
  (h2 : ticket_price = 2)
  (h3 : num_donations_15 = 2)
  (h4 : donation_15 = 15)
  (h5 : donation_20 = 20) :
  (tickets_sold * ticket_price + num_donations_15 * donation_15 + donation_20) = 100 := 
by
  sorry

end total_money_raised_l12_12114


namespace gigi_has_15_jellybeans_l12_12527

variable (G : ℕ) -- G is the number of jellybeans Gigi has
variable (R : ℕ) -- R is the number of jellybeans Rory has
variable (L : ℕ) -- L is the number of jellybeans Lorelai has eaten

-- Conditions
def condition1 := R = G + 30
def condition2 := L = 3 * (G + R)
def condition3 := L = 180

-- Proof statement
theorem gigi_has_15_jellybeans (G R L : ℕ) (h1 : condition1 G R) (h2 : condition2 G R L) (h3 : condition3 L) : G = 15 := by
  sorry

end gigi_has_15_jellybeans_l12_12527


namespace monotonicity_of_f_tangent_intersection_points_l12_12327

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12327


namespace measurable_iff_preimage_singleton_in_σ_l12_12487

variables (Ω : Type*) (E : set ℝ) (ξ : Ω → ℝ) (𝓕 : set (set Ω))

-- Assume E is countable
axiom countable_E : set.countable E

-- Assume ξ maps Ω to E
axiom mapping_to_E : ∀ ω : Ω, ξ ω ∈ E

-- Assume 𝓕 is a sigma-algebra
axiom sigma_algebra_𝓕 : is_sigma_algebra Ω 𝓕

theorem measurable_iff_preimage_singleton_in_σ :
  (measurable_set 𝓕 ξ) ↔ (∀ x ∈ E, { ω : Ω | ξ ω = x } ∈ 𝓕) :=
begin
  sorry
end

end measurable_iff_preimage_singleton_in_σ_l12_12487


namespace longest_side_of_rectangle_l12_12864

theorem longest_side_of_rectangle
  (m : ℕ) (l w : ℕ)
  (h1 : 2 * l + 2 * w = m)
  (h2 : l * w = 12 * (m / 2)) :
  max l w = 72 := by 
  have : m = 240 := by sorry
  have : 12 * (m / 2) = 2880 := by sorry
  sorry

end longest_side_of_rectangle_l12_12864


namespace convert_563_base8_to_base7_l12_12152

theorem convert_563_base8_to_base7 :
  let base8 := 5 * 8^2 + 6 * 8^1 + 3 * 8^0
  in let base10 := base8
  in let base7 := 1 * 7^3 + 1 * 7^2 + 6 * 7^1 + 2 * 7^0
  in base10 = base7 := by
  sorry

end convert_563_base8_to_base7_l12_12152


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12345

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12345


namespace closest_perfect_square_to_312_l12_12967

theorem closest_perfect_square_to_312 :
  ∃ n : ℤ, (n^2 = 324) ∧ 
           (∀ m : ℤ, (m^2 = 289) → abs(312 - 324) < abs(312 - 289)) :=
by
  sorry

end closest_perfect_square_to_312_l12_12967


namespace factorize_cubic_l12_12166

theorem factorize_cubic (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  sorry

end factorize_cubic_l12_12166


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12310

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12310


namespace elephants_ratio_l12_12564

theorem elephants_ratio (x : ℝ) (w : ℝ) (g : ℝ) (total : ℝ) :
  w = 70 →
  total = 280 →
  g = x * w →
  w + g = total →
  x = 3 :=
by 
  intros h1 h2 h3 h4
  sorry

end elephants_ratio_l12_12564


namespace monotonicity_f_tangent_points_l12_12361

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12361


namespace eq_970299_l12_12607

theorem eq_970299 : 98^3 + 3 * 98^2 + 3 * 98 + 1 = 970299 :=
by
  sorry

end eq_970299_l12_12607


namespace monotonicity_f_tangent_points_l12_12364

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12364


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12300

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12300


namespace part1_monotonicity_part2_tangent_intersection_l12_12259

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12259


namespace find_radius_l12_12635

noncomputable def large_circle_radius : ℝ := 2

def small_circle_radius (r : ℝ) : Prop :=
  let hexagon_side := 2 * r
  let dist_center_to_small := large_circle_radius + r
  (dist_center_to_small)^2 = 
    hexagon_side^2 + hexagon_side^2 - 2 * hexagon_side * hexagon_side * Real.cos (2*Real.pi/3)

theorem find_radius (r : ℝ) (h : small_circle_radius r) : 
  r = (2 * (1 + Real.sqrt 11)) / 11 :=
begin
  sorry
end

end find_radius_l12_12635


namespace geom_prog_identity_l12_12880

-- Define that A, B, C are the n-th, p-th, and k-th terms respectively of the same geometric progression.
variables (a r : ℝ) (n p k : ℕ) (A B C : ℝ)

-- Assume A = ar^(n-1), B = ar^(p-1), C = ar^(k-1)
def isGP (a r : ℝ) (n p k : ℕ) (A B C : ℝ) : Prop :=
  A = a * r^(n-1) ∧ B = a * r^(p-1) ∧ C = a * r^(k-1)

-- Define the statement to be proved
theorem geom_prog_identity (h : isGP a r n p k A B C) : A^(p-k) * B^(k-n) * C^(n-p) = 1 :=
sorry

end geom_prog_identity_l12_12880


namespace proof_problem_l12_12135

variables {A B C D E : Type*} [normed_group A] [normed_group B] [normed_group C] [normed_group D] [normed_group E]
variables (α : ℝ) (a b : ℝ)
variables (A D E : A) (B C D E : B) (AD AE BE CE DE : ℝ)

noncomputable def internal_angle_bisector (A D : A) : Prop := sorry -- precise definition is skipped for brevity
noncomputable def external_angle_bisector (A E : A) : Prop := sorry -- precise definition is skipped for brevity
noncomputable def on_extension (B C E : B) : Prop := sorry -- precise definition is skipped for brevity
noncomputable def on_side (B C D : B) : Prop := sorry -- precise definition is skipped for brevity

variables (triangle_ABC : Type*) (AD_internal_bisector : internal_angle_bisector A D) (AE_external_bisector : external_angle_bisector A E)
variables (D_on_side_BC : on_side B C D) (E_on_extension_BC : on_extension B C E)

theorem proof_problem :
  (1 / BE + 1 / CE = 2 / DE) :=
by sorry

end proof_problem_l12_12135


namespace jill_salary_l12_12974

-- Defining the conditions
variables (S : ℝ) -- Jill's net monthly salary
variables (discretionary_income : ℝ) -- One fifth of her net monthly salary
variables (vacation_fund : ℝ) -- 30% of discretionary income into a vacation fund
variables (savings : ℝ) -- 20% of discretionary income into savings
variables (eating_out_socializing : ℝ) -- 35% of discretionary income on eating out and socializing
variables (leftover : ℝ) -- The remaining amount, which is $99

-- Given Conditions
-- One fifth of her net monthly salary left as discretionary income
def one_fifth_of_salary : Prop := discretionary_income = (1/5) * S

-- 30% into a vacation fund
def vacation_allocation : Prop := vacation_fund = 0.30 * discretionary_income

-- 20% into savings
def savings_allocation : Prop := savings = 0.20 * discretionary_income

-- 35% on eating out and socializing
def socializing_allocation : Prop := eating_out_socializing = 0.35 * discretionary_income

-- This leaves her with $99
def leftover_amount : Prop := leftover = 99

-- Eqution considering all conditions results her leftover being $99
def income_allocation : Prop := 
  vacation_fund + savings + eating_out_socializing + leftover = discretionary_income

-- The main proof goal: given all the conditions, Jill's net monthly salary is $3300
theorem jill_salary : 
  one_fifth_of_salary S discretionary_income → 
  vacation_allocation discretionary_income vacation_fund → 
  savings_allocation discretionary_income savings → 
  socializing_allocation discretionary_income eating_out_socializing → 
  leftover_amount leftover → 
  income_allocation discretionary_income vacation_fund savings eating_out_socializing leftover → 
  S = 3300 := by sorry

end jill_salary_l12_12974


namespace no_infinite_sequence_l12_12178

noncomputable def f (α β x : ℝ) : ℝ := (x + α) / (β * x + 1)

theorem no_infinite_sequence (α β : ℝ) (a : ℝ) 
    (hαβ : α * β > 0) 
    (h : ∀ n : ℕ, ∃ x : ℝ, x = (λ n, if n = 0 then a else f α β (x (n - 1))) n) :
  a ∈ { x | x = √(α / β) ∨ x = -√(α / β) } :=
sorry

end no_infinite_sequence_l12_12178


namespace parabola_distance_l12_12092

noncomputable def parabola : Type := 
{ P Q : ℝ × ℝ // P.2 ^ 2 = 4 * P.1 ∧ Q.2 ^ 2 = 4 * Q.1 }

def focus : (ℝ × ℝ) := (1, 0)

def distance (A B : ℝ × ℝ) : ℝ :=
real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem parabola_distance (P Q : parabola) (hP : distance P.1 focus = 2) (hQ : distance Q.1 focus = 5) :
  distance P.1 Q.1 = 3 * real.sqrt 5 ∨ distance P.1 Q.1 = real.sqrt 13 :=
sorry

end parabola_distance_l12_12092


namespace compute_fraction_product_l12_12676

theorem compute_fraction_product :
  (1 / 3)^4 * (1 / 5) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l12_12676


namespace melody_chinese_pages_l12_12507

theorem melody_chinese_pages :
  ∀ (x : ℕ),  ∀ (total_pages : ℕ), (total_pages = 20 + 16 + 8 + x) →
    let english := 20
    let science := 16
    let civics := 8
    let fraction := 1 / 4
    let read_english := fraction * english
    let read_science := fraction * science
    let read_civics := fraction * civics
    let total_read_tomorrow := 14 in
    (total_read_tomorrow = read_english + read_science + read_civics + (fraction * x)) →
    x = 12 :=
by
  intros x total_pages h
  sorry

end melody_chinese_pages_l12_12507


namespace ball_hits_ground_l12_12907

theorem ball_hits_ground (t : ℚ) : 
  (∃ t ≥ 0, (-4.9 * (t^2 : ℝ) + 5 * t + 10 = 0)) → t = 100 / 49 :=
by
  sorry

end ball_hits_ground_l12_12907


namespace find_p_plus_q_l12_12439

open Real

noncomputable theory

-- Definitions needed to reformulate the problem with conditions
def triangle_XYZ (XY XJ YJ YL : ℝ) (angleX angleZ : ℝ) : Prop :=
  XY = 15 ∧ angleX = 45 ∧ angleZ = 30 ∧ 
  XJ = 15 * (sqrt 2 / 2) ∧
  YJ = 15 * (sqrt 2 / 2) ∧
  YL = 15 * sqrt 2 / 2 

def perpendicular (XJ YZ : ℝ) : Prop := XJ = YZ * (sqrt 2 / 2)

-- The problem statement verifying the correct final value
theorem find_p_plus_q :
  ∀ (XY XJ YJ YL JL XO XQ QO : ℝ),
  triangle_XYZ XY XJ YJ YL 45 30 →
  perpendicular XJ (15 * sqrt 2) →
  JL = 15 * sqrt 2 →
  XO = 15 * sqrt 2 →
  QO = XO / 2 →
  XQ = XO - QO →
  XQ^2 = 225 / 2 →
  let p := 225 
  let q := 2
  in p + q = 227 :=
begin
  -- Import requirements
  intros,
  sorry, -- Proof omitted, we provided only the statement
end

end find_p_plus_q_l12_12439


namespace canned_boxes_equation_l12_12969

theorem canned_boxes_equation (x : ℕ) (h₁: x ≤ 300) :
  2 * 14 * x = 32 * (300 - x) :=
by
sorry

end canned_boxes_equation_l12_12969


namespace twin_primes_divisible_by_12_l12_12518

def isTwinPrime (p q : ℕ) : Prop :=
  p < q ∧ Nat.Prime p ∧ Nat.Prime q ∧ q - p = 2

theorem twin_primes_divisible_by_12 {p q r s : ℕ} 
  (h1 : isTwinPrime p q) 
  (h2 : p > 3) 
  (h3 : isTwinPrime r s) 
  (h4 : r > 3) :
  12 ∣ (p * r - q * s) := by
  sorry

end twin_primes_divisible_by_12_l12_12518


namespace line_through_circle_center_perpendicular_to_given_line_l12_12157

theorem line_through_circle_center_perpendicular_to_given_line :
  let center := (-1, 2)
  ∧ (∀ (x y : ℝ), (x - center.1)² + (y - center.2)² = 5 ↔ x^2 + y^2 + 2x - 4y = 0)
  ∧ (center.1 = -1 ∧ center.2 = 2)
  ∧ (∀ (x y : ℝ), 2*x + 3*y = 0 → y = -2/3 * x)
  ∧ (∀ (m : ℝ), m = 3/2 → 
     (∃ (c : ℝ), ∀ (x y : ℝ), y = m * x + c ↔ 3*x - 2*y + 7 = 0)):
  ∃ (x y : ℝ), 3*x - 2*y + 7 = 0 :=
by sorry

end line_through_circle_center_perpendicular_to_given_line_l12_12157


namespace number_of_tilings_5x1_using_all_colors_l12_12627

def tile_ways_board_5x1 : ℕ :=
  let three_pieces := 6 * (3^3 - 3 * 2^3 + 3)
  let four_pieces := 4 * (3^4 - 4 * 2^4 + 3 * 2^2)
  three_pieces + four_pieces

theorem number_of_tilings_5x1_using_all_colors :
  tile_ways_board_5x1 = 152 :=
by
  -- Proof goes here
  sorry

end number_of_tilings_5x1_using_all_colors_l12_12627


namespace sam_bought_17_mystery_books_l12_12138

def adventure_books := 13
def used_books := 15
def new_books := 15
def total_books := used_books + new_books
def mystery_books := total_books - adventure_books

theorem sam_bought_17_mystery_books : mystery_books = 17 := by
  sorry

end sam_bought_17_mystery_books_l12_12138


namespace dustin_reads_more_pages_l12_12695

theorem dustin_reads_more_pages (dustin_rate_per_hour : ℕ) (sam_rate_per_hour : ℕ) : 
  (dustin_rate_per_hour = 75) → (sam_rate_per_hour = 24) → 
  (dustin_rate_per_hour * 40 / 60 - sam_rate_per_hour * 40 / 60 = 34) :=
by
  sorry

end dustin_reads_more_pages_l12_12695


namespace total_price_of_sweaters_l12_12805

theorem total_price_of_sweaters (price_shirts : ℝ) (num_shirts : ℕ) (num_sweaters : ℕ) (price_diff : ℝ)
  (h1 : price_shirts = 400) (h2 : num_shirts = 25) (num_sweaters = 75) (h3 : price_diff = 4) :
  let avg_price_shirt := price_shirts / num_shirts,
      avg_price_sweater := avg_price_shirt + price_diff,
      total_price_sweaters := avg_price_sweater * num_sweaters in
  total_price_sweaters = 1500 :=
  sorry

end total_price_of_sweaters_l12_12805


namespace cost_of_fencing_l12_12061

-- Define the conditions
def width_garden : ℕ := 12
def length_playground : ℕ := 16
def width_playground : ℕ := 12
def price_per_meter : ℕ := 15
def area_playground : ℕ := length_playground * width_playground
def area_garden : ℕ := area_playground
def length_garden : ℕ := area_garden / width_garden
def perimeter_garden : ℕ := 2 * (length_garden + width_garden)
def cost_fencing : ℕ := perimeter_garden * price_per_meter

-- State the theorem
theorem cost_of_fencing : cost_fencing = 840 := by
  sorry

end cost_of_fencing_l12_12061


namespace count_arrangements_l12_12455

theorem count_arrangements (students : Fin 5) (A B : Fin 5) (hA_ne_B : A ≠ B) : 
  ∃ n : Nat, n = 72 ∧ 
    n = (5.factorial - (4.factorial * 2.factorial)) := 
sorry

end count_arrangements_l12_12455


namespace Jack_sent_correct_number_of_BestBuy_cards_l12_12474

def price_BestBuy_gift_card : ℕ := 500
def price_Walmart_gift_card : ℕ := 200
def initial_BestBuy_gift_cards : ℕ := 6
def initial_Walmart_gift_cards : ℕ := 9

def total_price_of_initial_gift_cards : ℕ :=
  (initial_BestBuy_gift_cards * price_BestBuy_gift_card) +
  (initial_Walmart_gift_cards * price_Walmart_gift_card)

def price_of_Walmart_sent : ℕ := 2 * price_Walmart_gift_card
def value_of_gift_cards_remaining : ℕ := 3900

def prove_sent_BestBuy_worth : Prop :=
  total_price_of_initial_gift_cards - value_of_gift_cards_remaining - price_of_Walmart_sent = 1 * price_BestBuy_gift_card

theorem Jack_sent_correct_number_of_BestBuy_cards :
  prove_sent_BestBuy_worth :=
by
  sorry

end Jack_sent_correct_number_of_BestBuy_cards_l12_12474


namespace multiple_of_7_l12_12073

theorem multiple_of_7 :
  ∃ k : ℤ, 77 = 7 * k :=
sorry

end multiple_of_7_l12_12073


namespace smallest_number_l12_12620

theorem smallest_number (a b c d e : ℕ) (h₁ : a = 12) (h₂ : b = 16) (h₃ : c = 18) (h₄ : d = 21) (h₅ : e = 28) : 
    ∃ n : ℕ, (n - 4) % Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = 0 ∧ n = 1012 :=
by
    sorry

end smallest_number_l12_12620


namespace fourth_power_nested_sqrt_l12_12140

noncomputable def nested_sqrt := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))

theorem fourth_power_nested_sqrt :
  (nested_sqrt ^ 4) = 6 + 4 * Real.sqrt (2 + Real.sqrt 2) :=
sorry

end fourth_power_nested_sqrt_l12_12140


namespace line_through_two_points_l12_12913

theorem line_through_two_points (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (3, 4)) :
  ∃ k b : ℝ, (∀ x y : ℝ, (y = k * x + b) ↔ ((x, y) = A ∨ (x, y) = B)) ∧ (k = 1) ∧ (b = 1) := 
by
  sorry

end line_through_two_points_l12_12913


namespace sum_of_areas_l12_12123

theorem sum_of_areas (radii : ℕ → ℝ) (areas : ℕ → ℝ) (h₁ : radii 0 = 2) 
  (h₂ : ∀ n, radii (n + 1) = radii n / 3) 
  (h₃ : ∀ n, areas n = π * (radii n) ^ 2) : 
  ∑' n, areas n = (9 * π) / 2 := 
by 
  sorry

end sum_of_areas_l12_12123


namespace area_ratio_of_triangles_l12_12144

universe u
noncomputable def circle (α : Type u) := {c : α // sorry}

variables {α : Type u} [metric_space α]

def tangent_points (ω : circle α) (a b c : α) := sorry
def exsimilicenter (ω₁ ω₂ : circle α) := sorry
def homothety (P : α) (ω₁ ω₂ : circle α) := sorry

theorem area_ratio_of_triangles {A1 B1 C1 A2 B2 C2 : α} 
  (ω1 ω2 : circle α) (r1 r2 : ℝ) 
  (hω1 : tangent_points ω1 A1 B1 C1)
  (hω2 : tangent_points ω2 A2 B2 C2)
  (no_common_point : α → ¬ (ω1 = ω2))
  (hP : exsimilicenter ω1 ω2)
  (hHom : homothety hP ω1 ω2) :
  ∃ (r1 r2 : ℝ), (r1 / r2)^2 = ((area (triangle A1 B1 C1)) / (area (triangle A2 B2 C2))) :=
sorry

end area_ratio_of_triangles_l12_12144


namespace pure_imaginary_solution_of_poly_l12_12942

theorem pure_imaginary_solution_of_poly :
  (∀ k : ℝ, (k^2 = 17) → (x = k * I)) →
  ∃ x : ℂ, (x^4 + 2*x^3 + 6*x^2 + 34*x + 49 = 0) ∧ (x = I * Real.sqrt 17 ∨ x = -I * Real.sqrt 17) :=
by
sorrey

end pure_imaginary_solution_of_poly_l12_12942


namespace fourth_power_sqrt_expression_l12_12909

-- Definition of the expression y
def y : ℝ := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2))))

-- Theorem statement to prove that y^4 equals the specified expression
theorem fourth_power_sqrt_expression :
    y^4 = 10 + 4 * Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2))) + 2 * Real.sqrt (2 + Real.sqrt 2) :=
by
  sorry

end fourth_power_sqrt_expression_l12_12909


namespace algebraic_expression_decrease_l12_12816

theorem algebraic_expression_decrease (x y : ℝ) :
  let original_expr := 2 * x^2 * y
  let new_expr := 2 * ((1 / 2) * x) ^ 2 * ((1 / 2) * y)
  let decrease := ((original_expr - new_expr) / original_expr) * 100
  decrease = 87.5 := by
  sorry

end algebraic_expression_decrease_l12_12816


namespace probability_of_passing_test_l12_12465

theorem probability_of_passing_test (p : ℝ) (h : p + p * (1 - p) + p * (1 - p)^2 = 0.784) : p = 0.4 :=
sorry

end probability_of_passing_test_l12_12465


namespace final_bill_amount_l12_12628

variable (initial_bill : ℝ) (late_charge : ℝ)
variable (days : ℕ)

def bill_after_30_days (initial_bill : ℝ) (late_charge : ℝ) : ℝ :=
  initial_bill * (1 + late_charge / 100)

def bill_after_60_days (initial_bill : ℝ) (late_charge : ℝ) : ℝ :=
  bill_after_30_days initial_bill late_charge * (1 + late_charge / 100)

def bill_after_90_days (initial_bill : ℝ) (late_charge : ℝ) : ℝ :=
  bill_after_60_days initial_bill late_charge * (1 + late_charge / 100)

theorem final_bill_amount :
  bill_after_90_days 500 2 = 530.604 :=
sorry

end final_bill_amount_l12_12628


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12340

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12340


namespace tangent_product_l12_12497

noncomputable section

open Real

theorem tangent_product 
  (x y k1 k2 : ℝ) :
  (x / 2) ^ 2 + y ^ 2 = 1 ∧ 
  (x, y) = (-3, -3) ∧ 
  k1 + k2 = 18 / 5 ∧
  k1 * k2 = 8 / 5 → 
  (3 * k1 - 3) * (3 * k2 - 3) = 9 := 
by
  intros 
  sorry

end tangent_product_l12_12497


namespace solve_system_of_equations_l12_12000

theorem solve_system_of_equations
  (a b c d x y z u : ℝ)
  (h1 : a^3 * x + a^2 * y + a * z + u = 0)
  (h2 : b^3 * x + b^2 * y + b * z + u = 0)
  (h3 : c^3 * x + c^2 * y + c * z + u = 0)
  (h4 : d^3 * x + d^2 * y + d * z + u = 1) :
  x = 1 / ((d - a) * (d - b) * (d - c)) ∧
  y = -(a + b + c) / ((d - a) * (d - b) * (d - c)) ∧
  z = (a * b + b * c + c * a) / ((d - a) * (d - b) * (d - c)) ∧
  u = - (a * b * c) / ((d - a) * (d - b) * (d - c)) :=
sorry

end solve_system_of_equations_l12_12000


namespace veryNiceSequences_count_8_l12_12214

def isVeryNiceSequence (n : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ k, 2 ≤ k → k ≤ n →
    ∃ i j, i < j ∧ seq i = k - 1 ∧ seq j = k

theorem veryNiceSequences_count_8 :
  let n := 8
  let veryNiceSequences : Finset (ℕ → ℕ) := 
    {seq | ∃ (h : seq : Fin n → ℕ), isVeryNiceSequence n seq }
  veryNiceSequences.card = 40320 := sorry

end veryNiceSequences_count_8_l12_12214


namespace old_man_gold_coins_l12_12131

theorem old_man_gold_coins (x y : ℕ) (h1 : x - y = 1) (h2 : x^2 - y^2 = 25 * (x - y)) : x + y = 25 := 
sorry

end old_man_gold_coins_l12_12131


namespace find_c_and_centroid_l12_12237

-- Define vectors and orthogonality conditions
def vertex_a : ℝ × ℝ := (-1, 0)
def vertex_b : ℝ × ℝ := (4, 0)
def vertex_c (c : ℝ) : ℝ × ℝ := (0, c)

def vector_ac (c : ℝ) : ℝ × ℝ := (1, c)
def vector_bc (c : ℝ) : ℝ × ℝ := (-4, c)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

def is_perpendicular (c : ℝ) : Prop := dot_product (vector_ac c) (vector_bc c) = 0

-- Define centroid calculation
def centroid_x : ℝ := (vertex_a.1 + vertex_b.1 + 0) / 3
def centroid_y (c : ℝ) : ℝ := (vertex_a.2 + vertex_b.2 + c) / 3

theorem find_c_and_centroid :
  (∃ c : ℝ, (c = 2 ∨ c = -2) ∧ is_perpendicular c) ∧ 
  (∃ c : ℝ, (c = 2 ∨ c = -2) ∧ ((centroid_x = 1 ∧ centroid_y c = 2 / 3) ∨ (centroid_x = 1 ∧ centroid_y c = -2 / 3))) :=
by
  sorry

end find_c_and_centroid_l12_12237


namespace largest_prime_divisor_of_given_expression_l12_12713

-- Define the given expression
def given_expression : ℕ := 36^2 + 45^2

-- Define the definition of a prime number and the largest prime divisor
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m ∣ n → m = n

def largest_prime_divisor (n : ℕ) : ℕ :=
  list.max' (list.filter is_prime (list.divisors n))

-- State the theorem to be proven
theorem largest_prime_divisor_of_given_expression : 
  largest_prime_divisor given_expression = 71 :=
by 
  -- Note: The proof is omitted with 'sorry' as instructed
  sorry

end largest_prime_divisor_of_given_expression_l12_12713


namespace sum_of_square_areas_l12_12471

open Real

theorem sum_of_square_areas (XZ XY : ℝ) (hXZ : XZ = 13) (hXY : XY = 12) 
  (right_angle : ∃ Y Z X, angle YZX = π / 2 ∧ dist Y Z = XY ∧ dist Z X = XZ) : 
  let YZ := sqrt (XZ^2 - XY^2) in
  let area_WXZY := XZ^2 in
  let area_YZUV := YZ^2 in
  area_WXZY + area_YZUV = 194 :=
by
  sorry

end sum_of_square_areas_l12_12471


namespace fibonacci_tiling_equivalence_l12_12837

def F : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n + 2) := F n + F (n + 1)

def f : ℕ → ℤ
| 0       := 1
| 1       := 1
| (n + 2) := f n + f (n + 1)

theorem fibonacci_tiling_equivalence (n : ℕ) : F n = f (n - 1) :=
sorry

end fibonacci_tiling_equivalence_l12_12837


namespace jane_current_age_l12_12476

noncomputable def JaneAge : ℕ := 34

theorem jane_current_age : 
  ∃ J : ℕ, 
    (∀ t : ℕ, t ≥ 18 ∧ t - 18 ≤ JaneAge - 18 → t ≤ JaneAge / 2) ∧
    (JaneAge - 12 = 23 - 12 * 2) ∧
    (23 = 23) →
    J = 34 := by
  sorry

end jane_current_age_l12_12476


namespace total_number_of_houses_l12_12976

theorem total_number_of_houses (hd : ℕ) (hc : ℕ) (hbc : ℕ) (h1 : hd = 40) (h2 : hc = 30) (h3 : hbc = 10) : hd + hc - hbc = 60 := by
  rw [h1, h2, h3]
  norm_num

end total_number_of_houses_l12_12976


namespace monotonicity_tangent_intersection_points_l12_12383

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12383


namespace max_areas_divided_by_disk_l12_12992

-- Define the problem and the required conditions for the disk division
variable (n : ℕ)
axiom n_pos : n > 0

-- Assume the disk is divided by 2n equally spaced radii, one secant line, and one chord that does not intersect the secant
def max_non_overlapping_areas (n : ℕ) : ℕ := 4 * n - 1

-- The theorem stating that the maximum number of non-overlapping areas is 4n - 1
theorem max_areas_divided_by_disk :
  max_non_overlapping_areas n = 4 * n - 1 :=
sorry

end max_areas_divided_by_disk_l12_12992


namespace point_T_on_AD_l12_12523

theorem point_T_on_AD
  (A B C D O T : Type)
  (circle1 : circle)
  (circle2 : circle)
  (inscribed1 : inscribed_quadrilateral A B C D circle1)
  (diam1 : diameter AD circle1)
  (int1 : intersection_point (diagonal AC) (diagonal BD) O)
  (center_circle2 : center O circle2)
  (tangent : is_tangent circle2 BC)
  (int_tangents : intersection_tangents_from_points B C T circle2) :
  lies_on_segment T AD :=
begin
  sorry
end

end point_T_on_AD_l12_12523


namespace find_f_log_l12_12492

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^(x+1) else if x < 0 then -2^(-x+1) else 0

theorem find_f_log :
  let f (x : ℝ) := if x > 0 then 2^(x+1) else if x < 0 then -2^(-x+1) else 0
  in f (Real.log 3 / Real.log (1/4)) = -2 * Real.sqrt 3 :=
begin
  sorry
end

end find_f_log_l12_12492


namespace geometric_sequence_sum_l12_12441

theorem geometric_sequence_sum :
  ∃ (a : ℕ), 
   (∀ n, a > 0) ∧ 
   (∀ n, a * 2^(n-1)) ∧
   (a + a * 2 + a * 2^2 = 21) → 
   (a * 2^2 + a * 2^3 + a * 2^4 = 84) := sorry

end geometric_sequence_sum_l12_12441


namespace find_w_l12_12154

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1, 2], ![4, 1]]

def w : Matrix (Fin 2) (Fin 1) ℚ :=
  ![![1 / 243], ![-1 / 324]]

def I2 : Matrix (Fin 2) (Fin 2) ℚ :=
  1

theorem find_w : 
  ((B ^ 6 + B ^ 4 + B ^ 2 + I2) ⬝ w = ![![1], ![16]]) :=
begin
  sorry
end

end find_w_l12_12154


namespace monotonicity_and_tangent_intersection_l12_12250

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12250


namespace even_binomial_coefficients_l12_12148

theorem even_binomial_coefficients (n : ℕ) (h_pos: 0 < n) : 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → 2 ∣ Nat.choose n k) ↔ ∃ k : ℕ, n = 2^k :=
by
  sorry

end even_binomial_coefficients_l12_12148


namespace circle_tangent_to_y_axis_l12_12815

theorem circle_tangent_to_y_axis (m : ℝ) :
  (0 < m) → (∀ p : ℝ × ℝ, (p.1 - m)^2 + p.2^2 = 4 ↔ p.1 ^ 2 = p.2^2) → (m = 2 ∨ m = -2) :=
by
  sorry

end circle_tangent_to_y_axis_l12_12815


namespace part1_monotonicity_part2_tangent_intersection_l12_12266

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12266


namespace desired_average_l12_12512

variable (avg_4_tests : ℕ)
variable (score_5th_test : ℕ)

theorem desired_average (h1 : avg_4_tests = 78) (h2 : score_5th_test = 88) : (4 * avg_4_tests + score_5th_test) / 5 = 80 :=
by
  sorry

end desired_average_l12_12512


namespace apple_box_weights_l12_12934

theorem apple_box_weights (a b c d : ℤ) 
  (h1 : a + b + c = 70)
  (h2 : a + b + d = 80)
  (h3 : a + c + d = 73)
  (h4 : b + c + d = 77) : 
  a = 23 ∧ b = 27 ∧ c = 20 ∧ d = 30 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end apple_box_weights_l12_12934


namespace total_money_raised_l12_12113

-- Given conditions:
def tickets_sold : Nat := 25
def ticket_price : ℚ := 2
def num_donations_15 : Nat := 2
def donation_15 : ℚ := 15
def donation_20 : ℚ := 20

-- Theorem statement proving the total amount raised is $100
theorem total_money_raised
  (h1 : tickets_sold = 25)
  (h2 : ticket_price = 2)
  (h3 : num_donations_15 = 2)
  (h4 : donation_15 = 15)
  (h5 : donation_20 = 20) :
  (tickets_sold * ticket_price + num_donations_15 * donation_15 + donation_20) = 100 := 
by
  sorry

end total_money_raised_l12_12113


namespace susie_initial_amount_l12_12895

-- Definitions for conditions:
def initial_amount (X : ℝ) : Prop :=
  X + 0.20 * X = 240

-- Main theorem to prove:
theorem susie_initial_amount (X : ℝ) (h : initial_amount X) : X = 200 :=
by 
  -- structured proof will go here
  sorry

end susie_initial_amount_l12_12895


namespace frustum_volume_fraction_l12_12124

theorem frustum_volume_fraction (base_edge : ℝ) (original_altitude : ℝ) (fraction : ℝ) 
  (h_base_edge : base_edge = 40) (h_altitude : original_altitude = 15) 
  (h_fraction : fraction = 1/3) :
  let volume_ratio := (1 - (fraction ^ 3)) in 
  volume_ratio = 26/27 :=
by
  sorry

end frustum_volume_fraction_l12_12124


namespace sin_A_plus_C_over_sin_A_plus_sin_C_equals_l12_12813

section proof_problem

variables (a b c x y : ℝ)
variables (A : ℝ := 3)
variables (C : ℝ := -3)

-- Condition: B lies on the ellipse
def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 25 = 1

-- Main theorem to prove
theorem sin_A_plus_C_over_sin_A_plus_sin_C_equals : 
  (∃ x y, on_ellipse x y) →
  (∃ a b c, true) →
  (∃ A C, (A = 3 ∧ C = -3)) →
  (∃ B, B = (x, y) ∧ on_ellipse x y) →
  (sin (A + C) / (sin A + sin C) = 3 / 5) :=
sorry

end proof_problem

end sin_A_plus_C_over_sin_A_plus_sin_C_equals_l12_12813


namespace radar_detects_object_in_square_l12_12090

noncomputable def mysterious_trajectory (x : ℝ) : ℝ := 
  (((((x^5 - 2013)^5 - 2013)^5 - 2013)^5 - 2013)^5 - 2013)

def radar_line (x : ℝ) : ℝ := x + 2013

def grid_square (x : ℝ) (y : ℝ) : (ℤ × ℤ) :=
  (Int.floor x, Int.floor y)

theorem radar_detects_object_in_square : 
  (∃ x : ℝ, mysterious_trajectory x = radar_line x ∧ x ∈ [4, 5] ∧ y ∈ [2017, 2018]) →
  grid_square x (radar_line x) = (4, 2017) :=
by
  sorry

end radar_detects_object_in_square_l12_12090


namespace find_largest_n_l12_12711

/-- Find the largest integer n such that:
 1. n^2 can be expressed as the difference of two consecutive cubes,
 2. 5n + 103 is a perfect square -/

theorem find_largest_n :
  ∃ n : ℤ, n = 313 ∧ (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧ ∃ k : ℤ, 5 * n + 103 = k^2 :=
by
  have n := 313
  use n
  split
  . exact rfl
  split
  . use 113
    rw [←rfl, ←rfl]
    norm_num
  . use 41
    rw [←rfl, ←rfl]
    norm_num
  sorry

end find_largest_n_l12_12711


namespace cost_price_radio_l12_12545

theorem cost_price_radio (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1305) 
  (h2 : loss_percentage = 0.13) 
  (h3 : SP = C * (1 - loss_percentage)) :
  C = 1500 := 
by 
  sorry

end cost_price_radio_l12_12545


namespace ellipse_equation_range_of_op_dot_oq_l12_12238

-- Ellipse E : x^2/a^2 + y^2/b^2 = 1 where a > b > 0
variables (a b c : ℝ) (h₁ : a > b) (h₂ : b > 0)

-- The ellipse passes through the point (0, √2)
axiom pass_point : ∃ x y, x = 0 ∧ y = real.sqrt 2 ∧ (x^2 / a^2) + (y^2 / b^2) = 1

-- The eccentricity of the ellipse is √6 / 3
axiom eccentricity : c / a = real.sqrt 6 / 3

-- Point O as the coordinate origin
def origin : ℝ × ℝ := (0, 0)

-- Proofs of the required statements
theorem ellipse_equation : ∃ a b, a = real.sqrt 6 ∧ b = real.sqrt 2 ∧ ( ∀ x y : ℝ, (x^2 / 6) + (y^2 / 2) = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1 ) :=
sorry

theorem range_of_op_dot_oq : ∃ status : ℝ → ℝ → (ℝ × ℝ) → Prop, status = λ x y O, x = 0 ∧ y ≥ -6 ∧ y < 10/3 :=
sorry

end ellipse_equation_range_of_op_dot_oq_l12_12238


namespace tens_digit_6_pow_19_l12_12966

def tens_digit (n : ℕ) : ℕ := (6^n / 10) % 10

theorem tens_digit_6_pow_19 : tens_digit 19 = 1 := by
  have pattern : ∀ n, tens_digit (n + 4) = tens_digit n :=
    sorry
  have key_calculation : tens_digit 3 = 1 :=
    sorry
  rw [← @nat.mod_add_div 19 4, pattern],
  exact key_calculation

end tens_digit_6_pow_19_l12_12966


namespace fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25_l12_12597

def positive_integers_up_to (n : ℕ) : List ℕ :=
  List.range' 1 n

def divisible_by_lcm (lcm : ℕ) (lst : List ℕ) : List ℕ :=
  lst.filter (λ x => x % lcm = 0)

noncomputable def fraction_divisible_by_both (n a b : ℕ) : ℚ :=
  let lcm_ab := Nat.lcm a b
  let elems := positive_integers_up_to n
  let divisible_elems := divisible_by_lcm lcm_ab elems
  divisible_elems.length / n

theorem fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25 :
  fraction_divisible_by_both 100 3 4 = (2 : ℚ) / 25 :=
by
  sorry

end fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25_l12_12597


namespace monotonicity_and_tangent_intersections_l12_12359

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12359


namespace shaded_area_l12_12103

-- Define the properties of the circles.
def circle1_radius : ℝ := 4 -- radius of the circle centered at A
def circle2_radius : ℝ := 7 -- radius of the circle centered at B
def distance_between_centers : ℝ := 3 -- distance between centers A and B

-- Define the areas of the two circles.
def circle1_area : ℝ := π * (circle1_radius ^ 2)
def circle2_area : ℝ := π * (circle2_radius ^ 2)

-- The problem translates to proving the shaded area is equal to 33π.
theorem shaded_area : circle2_area - circle1_area = 33 * π := by
  sorry

end shaded_area_l12_12103


namespace fruit_fly_cell_division_l12_12808

/-- Genetic properties of fruit flies:
  1. Fruit flies have 2N = 8 chromosomes.
  2. Alleles A/a and B/b are inherited independently.
  3. Genotype AaBb is given.
  4. This genotype undergoes cell division without chromosomal variation.

Prove that:
Cells with a genetic composition of AAaaBBbb contain 8 or 16 chromosomes.
-/
theorem fruit_fly_cell_division (genotype : ℕ → ℕ) (A a B b : ℕ) :
  genotype 2 = 8 ∧
  (A + a + B + b = 8) ∧
  (genotype 0 = 2 * 4) →
  (genotype 1 = 8 ∨ genotype 1 = 16) :=
by
  sorry

end fruit_fly_cell_division_l12_12808


namespace integer_valued_polynomial_l12_12521

def polynomial : ℚ[X] :=
  (1/24) * X^4 + (1/12) * X^3 - (25/24) * X^2 + (11/12) * X + 1

theorem integer_valued_polynomial : 
  ∃ (a b c d : ℤ), polynomial = a * (X * (X - 1) * (X - 2) * (X - 3)) / 24 +
                                      b * (X * (X - 1) * (X - 2)) / 6 +
                                      c * (X * (X - 1)) / 2 + d :=
sorry

end integer_valued_polynomial_l12_12521


namespace isosceles_triangle_congruent_side_length_l12_12899

theorem isosceles_triangle_congruent_side_length :
  ∀ (D E F : Type) [HasLength D E F],
  let base := 30
  let area := 120
  let height := area / (base / 2)
  let half_base := base / 2
  let side_length := Math.sqrt (half_base^2 + height^2)
  side_length = 17 :=
by
  intros D E F
  sorry

end isosceles_triangle_congruent_side_length_l12_12899


namespace center_of_pan_lies_under_pancake_l12_12867

noncomputable def pan_area : ℝ := 1

structure Pan :=
(area : ℝ := pan_area)
(center : ℝ × ℝ := (0, 0))

structure Pancake (pan : Pan) :=
(area : ℝ)
(is_convex : Prop)
(area_greater_than_half : Prop)

axiom convex_pancake_on_pan (pan : Pan) (pancake : Pancake pan) :
  pancake.area > 1 / 2 →
  (∀ (x y : ℝ × ℝ), pancake.is_convex → convex_hull pancake = ⇑(pan.center)) → sorry

theorem center_of_pan_lies_under_pancake (pan : Pan) (pancake : Pancake pan)
  (h_area: pancake.area > 1 / 2)
  (h_convex: pancake.is_convex) :
  let center := pan.center in
  center ∈ convex_hull pancake :=
sorry

end center_of_pan_lies_under_pancake_l12_12867


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12344

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12344


namespace find_value_of_x_l12_12784

variable (k x : ℝ)

-- Given condition
def condition := log x / log k * log k / log 7 = 4

-- Prove statement
theorem find_value_of_x (h : condition k x) : x = 2401 :=
sorry

end find_value_of_x_l12_12784


namespace findHyperbolaEquation_l12_12207

noncomputable def hyperbolaEquation (a b : ℝ) (h : a > 0) (k : b > 0) :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem findHyperbolaEquation (c : ℝ) (hC : c = 2)
  (a b : ℝ) (hA : a > 0) (hB : b > 0)
  (asymptote_condition : b = sqrt(3) * a) 
  (focal_points_condition : c^2 = a^2 + b^2) :
  hyperbolaEquation 1 (sqrt 3) := 
begin
  -- Placeholder for proof
  sorry
end

end findHyperbolaEquation_l12_12207


namespace circle_arc_length_l12_12872

open Real

theorem circle_arc_length {R α : ℝ} (M A O B C : Point) (hR : R = 21) 
  (h_sin_alpha : sin α = sqrt 40 / 7)
  (h_point_M : M ∈ segment A O)
  (h_points_BC : B ∈ circle O R ∧ C ∈ circle O R)
  (h_angles : ∠ AMB = α ∧ ∠ OMC = α) : 
  distance B C = 18 := 
sorry

end circle_arc_length_l12_12872


namespace dog_ate_3_apples_l12_12192

theorem dog_ate_3_apples :
  (∀ (initial_on_tree initial_on_ground remaining_apples : ℕ)
    (initial_total_apples : ℕ ≠ 0),
    initial_on_tree = 5 →
    initial_on_ground = 8 →
    remaining_apples = 10 →
    initial_total_apples = initial_on_tree + initial_on_ground →
    remaining_apples + (initial_on_tree + initial_on_ground - remaining_apples) =
    10 + 3 :=
by
  intros initial_on_tree initial_on_ground remaining_apples initial_total_apples
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact 3

end dog_ate_3_apples_l12_12192


namespace initial_mixture_two_l12_12861

theorem initial_mixture_two (x : ℝ) (h : 0.25 * (x + 0.4) = 0.10 * x + 0.4) : x = 2 :=
by
  sorry

end initial_mixture_two_l12_12861


namespace monotonicity_and_tangent_intersection_l12_12245

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12245


namespace ship_distance_graph_shape_l12_12653

-- Define the points and paths
def point (name : String) := name

-- Definitions for points A, B, C, and X
def A := point "A"
def B := point "B"
def C := point "C"
def X := point "X"

-- Definitions for the paths
def elliptical_path := {A, B}
def straight_path := {B, C}

-- Condition: B is on an elliptical path with X as one of the foci
def elliptical_path_with_focus (foci : point) (p1 p2 : point) : Prop :=
  p1 ∈ elliptical_path ∧ p2 ∈ elliptical_path ∧ foci = X

-- Condition: The angle between line XB and the path BC is 30 degrees
def angle_in_degrees (line1 line2 : Set (point × point)) (degrees : ℝ) : Prop :=
  line1 = {(X, B)} ∧ line2 = {(B, C)} ∧ degrees = 30

-- The theorem statement
theorem ship_distance_graph_shape :
  elliptical_path_with_focus X A B ∧ angle_in_degrees {(X, B)} {(B, C)} 30 →
  (∃ shape, shape = "valley followed by a rising slope") :=
by
  intros h
  sorry

end ship_distance_graph_shape_l12_12653


namespace monotonicity_and_tangent_intersection_l12_12246

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12246


namespace smallest_n_valid_subset_l12_12149

-- Condition: Define the Set S
def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 98 }

-- Condition: Define subset properties
def has_coprime_property (A : Finset ℕ) : Prop :=
  ∃ B C : Finset ℕ, B.card = 5 ∧ C.card = 5 ∧ B ∪ C = A ∧ (∃ b ∈ B, ∀ x ∈ B \ {b}, gcd b x = 1)
  ∧ (∃ c ∈ C, ∀ y ∈ C \ {c}, gcd c y > 1)

-- Condition: Any subset of S with n elements has the coprime property
def valid_subset (T : Finset ℕ) (n : ℕ) : Prop :=
  T.card = n → ∃ A : Finset ℕ, A ⊆ T ∧ A.card = 10 ∧ has_coprime_property A 

-- Question: Determine the smallest n such that any subset of size n has the coprime property
theorem smallest_n_valid_subset :
  ∃ n : ℕ, n ≥ 1 ∧ (∀ T : Finset ℕ, T ⊆ S → valid_subset T n) ∧
  ∀ m : ℕ, m < n → ¬ (∀ T : Finset ℕ, T ⊆ S → valid_subset T m) :=
begin
  sorry
end

end smallest_n_valid_subset_l12_12149


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12319

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12319


namespace hyperbola_foci_property_l12_12406

noncomputable def hyperbola (x y b : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / b^2) = 1

theorem hyperbola_foci_property (x y b : ℝ) (h : hyperbola x y b) (b_pos : b > 0) (PF1 : ℝ) (PF2 : ℝ) (hPF1 : PF1 = 5) :
  PF2 = 11 :=
by
  sorry

end hyperbola_foci_property_l12_12406


namespace sum_of_coefficients_with_distinct_negative_roots_l12_12901

theorem sum_of_coefficients_with_distinct_negative_roots :
  (∑ k in {k : ℤ | ∃ r s : ℤ, r ≠ s ∧ r < 0 ∧ s < 0 ∧ r * s = 48 ∧ k = r + s}, k) = -124 :=
by
  sorry

end sum_of_coefficients_with_distinct_negative_roots_l12_12901


namespace difference_in_distances_l12_12572

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def distance_covered (r : ℝ) (revolutions : ℕ) : ℝ :=
  circumference r * revolutions

theorem difference_in_distances :
  let r1 := 22.4
  let r2 := 34.2
  let revolutions := 400
  let D1 := distance_covered r1 revolutions
  let D2 := distance_covered r2 revolutions
  D2 - D1 = 29628 :=
by
  sorry

end difference_in_distances_l12_12572


namespace domain_of_g_l12_12175

-- Define the function g(x) = (x^2 + 4x - 7) / sqrt(6 - x)
def g (x : ℝ) := (x^2 + 4*x - 7) / Real.sqrt (6 - x)

-- The problem statement: Prove that the domain of g(x) is (-∞, 6)
theorem domain_of_g : ∀ x : ℝ, (6 - x > 0) → Real.sqrt (6 - x) ≠ 0 → x < 6 :=
by sorry

end domain_of_g_l12_12175


namespace f_s_not_multiplicative_l12_12094

noncomputable def r_s (s n : ℕ) : ℕ := sorry

noncomputable def f_s (s n : ℕ) : ℝ := 1 / (2 * s) * (r_s s n)

-- Prove that f_s is not multiplicative for s ≠ 1, 2, 4, 8
theorem f_s_not_multiplicative (s : ℕ) (h: s ≠ 1 ∧ s ≠ 2 ∧ s ≠ 4 ∧ s ≠ 8) :
  ∃ (m n : ℕ), coprime m n ∧ f_s s (m * n) ≠ f_s s m * f_s s n :=
sorry

end f_s_not_multiplicative_l12_12094


namespace monotonicity_tangent_points_l12_12276

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12276


namespace parallelogram_count_l12_12804

theorem parallelogram_count (m n : ℕ) : 
  ∃ p : ℕ, p = (m.choose 2) * (n.choose 2) :=
by
  sorry

end parallelogram_count_l12_12804


namespace sum_first_20_terms_l12_12807

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the conditions stated in the problem
variables {a : ℕ → ℤ}
variables (h_arith : is_arithmetic_sequence a)
variables (h_sum_first_three : a 1 + a 2 + a 3 = -24)
variables (h_sum_18_to_20 : a 18 + a 19 + a 20 = 78)

-- State the theorem to prove
theorem sum_first_20_terms : (Finset.range 20).sum a = 180 :=
by
  sorry

end sum_first_20_terms_l12_12807


namespace area_difference_between_S1_S2_l12_12668

theorem area_difference_between_S1_S2 :
  let θ := 90
  let radius := 1
  let sector_area := (θ : ℝ) / 360 * π * radius^2
  let triangle_area := 1/2 * radius * radius * real.sin (θ / 180 * π)
  let total_sector_area := 2 * sector_area
  let total_triangle_area := 2 * triangle_area
  total_sector_area - total_triangle_area = (3 * real.sqrt 3 / 8 - π / 6) :=
by
  let θ := 90 : ℝ
  let radius := 1 : ℝ
  let sector_area := θ / 360 * π * radius^2
  let triangle_area := 1/2 * radius * radius * real.sin (θ / 180 * π)
  let total_sector_area := 2 * sector_area
  let total_triangle_area := 2 * triangle_area
  have area_difference := total_sector_area - total_triangle_area
  show area_difference = (3 * real.sqrt 3 / 8 - π / 6),
  sorry

end area_difference_between_S1_S2_l12_12668


namespace equilateral_triangle_AM_eq_BM_plus_CM_l12_12908

theorem equilateral_triangle_AM_eq_BM_plus_CM
  (A B C M O : Point)
  (hABC: EquilateralTriangle A B C)
  (hO: Circle O)
  (hInscribed : Inscribed A B C O)
  (hM_on_BC : OnSegment M B C) :
  AM = BM + CM := 
sorry

end equilateral_triangle_AM_eq_BM_plus_CM_l12_12908


namespace perfect_square_mod_3_l12_12642

theorem perfect_square_mod_3 (n : ℤ) : n^2 % 3 = 0 ∨ n^2 % 3 = 1 :=
sorry

end perfect_square_mod_3_l12_12642


namespace surface_area_of_parallelepiped_l12_12904

open Real

theorem surface_area_of_parallelepiped 
  (a b c : ℝ)
  (x y z : ℝ)
  (h1: a^2 = x^2 + y^2)
  (h2: b^2 = x^2 + z^2)
  (h3: c^2 = y^2 + z^2) :
  2 * (sqrt ((x * y)) + sqrt ((x * z)) + sqrt ((y * z)))  =
  sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2)) +
  sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2)) +
  sqrt ((a^2 + c^2 - b^2) * (b^2 + c^2 - a^2)) :=
by
  sorry

end surface_area_of_parallelepiped_l12_12904


namespace sum_of_solutions_l12_12068

theorem sum_of_solutions : 
  ∑ n in Finset.filter (λ n : ℤ, (|n| < |n - 4| ∧ |n - 4| < 10)) (Finset.range 20).map (λ x, ↑x - 6), n = -14 := 
by
  sorry

end sum_of_solutions_l12_12068


namespace paul_mowing_money_l12_12511

theorem paul_mowing_money (M : ℝ) 
  (h1 : 2 * M = 6) : 
  M = 3 :=
by 
  sorry

end paul_mowing_money_l12_12511


namespace partition_count_l12_12694

theorem partition_count (A B : Finset ℕ) :
  (∀ n, n ∈ A ∨ n ∈ B) ∧ 
  (∀ n, n ∈ A → 1 ≤ n ∧ n ≤ 9) ∧ 
  (∀ n, n ∈ B → 1 ≤ n ∧ n ≤ 9) ∧ 
  (A ∩ B = ∅) ∧ 
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
  (8 * A.sum id = B.sum id) ∧ 
  (A.sum id + B.sum id = 45) → 
  ∃! (num_ways : ℕ), num_ways = 3 :=
sorry

end partition_count_l12_12694


namespace part_a_part_b_lim_part_b_ineq_l12_12483

noncomputable def a_seq (a1 : ℝ) (a1_gt_2 : a1 > 2) : ℕ → ℝ 
| 0 => a1
| n+1 => a1 + 2 / (a_seq a1 a1_gt_2 n)

theorem part_a
  (a1 : ℝ) (a1_gt_2 : a1 > 2) 
  (a_seq := a_seq a1 a1_gt_2) :
  ∀ n : ℕ, a_seq (2 * n - 1) + a_seq (2 * n) > 4 :=
sorry

theorem part_b_lim
  (a1 : ℝ) (a1_gt_2 : a1 > 2)
  (a_seq := a_seq a1 a1_gt_2) :
  filter.tendsto (λ n, a_seq n) filter.at_top (𝓝 2) :=
sorry

theorem part_b_ineq
  (a1 : ℝ) (a1_gt_2 : a1 > 2)
  (a_seq := a_seq a1 a1_gt_2) :
  ∀ x : ℝ, ∀ n : ℕ, ∑ i in finset.range n, real.sqrt (x^2 + (a_seq i)^2) > n * real.sqrt (x^2 + 2^2) :=
sorry

end part_a_part_b_lim_part_b_ineq_l12_12483


namespace most_suitable_candidate_l12_12663

-- Definitions for variances
def variance_A := 3.4
def variance_B := 2.1
def variance_C := 2.5
def variance_D := 2.7

-- We start the theorem to state the most suitable candidate based on given variances and average scores.
theorem most_suitable_candidate :
  (variance_A = 3.4) ∧ (variance_B = 2.1) ∧ (variance_C = 2.5) ∧ (variance_D = 2.7) →
  true := 
by
  sorry

end most_suitable_candidate_l12_12663


namespace train_length_in_meters_l12_12656

theorem train_length_in_meters (speed_kmph : ℝ) (time_sec : ℝ) (length_m : ℝ) : 
  speed_kmph = 60 → time_sec = 5 → length_m = speed_kmph * 1000 / 3600 * time_sec → 
  length_m = 83.35 :=
by
  intros hs ht hl
  rw [hs, ht, hl]
  sorry

end train_length_in_meters_l12_12656


namespace kerosene_cost_l12_12442

-- Define the cost of a dozen eggs in terms of the cost of a pound of rice
def cost_of_dozen_eggs (cost_rice : ℝ) : ℝ := cost_rice

-- Define the cost of an egg
def cost_of_one_egg (cost_dozen_eggs : ℝ) : ℝ := cost_dozen_eggs / 12

-- Define the cost of 8 eggs
def cost_of_eight_eggs (cost_one_egg : ℝ) : ℝ := 8 * cost_one_egg

-- Define the cost of a half-liter of kerosene in terms of the cost of 8 eggs
def cost_of_half_liter_kerosene (cost_eight_eggs : ℝ) : ℝ := cost_eight_eggs

-- Define the cost of a liter of kerosene
def cost_of_liter_kerosene (cost_half_liter_kerosene : ℝ) : ℝ := 2 * cost_half_liter_kerosene

-- Define the conversion from dollars to cents
def dollars_to_cents (dollars : ℝ) : ℝ := dollars * 100

-- Prove that the cost of a liter of kerosene is 44 cents given the conditions
theorem kerosene_cost : cost_of_liter_kerosene (cost_of_half_liter_kerosene (cost_of_eight_eggs (cost_of_one_egg (cost_of_dozen_eggs 0.33)))) * 100 = 44 :=
by
  sorry

end kerosene_cost_l12_12442


namespace largest_prime_factor_5040_is_7_l12_12954

-- Definition of the condition: the prime factorization of 5040
def prime_factorization_5040 : list ℕ := [2, 2, 2, 2, 3, 3, 5, 7]

-- Predicate to check if a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m:ℕ, m ∣ n → m = 1 ∨ m = n

-- Predicate to check if a list contains only primes
def all_primes (l: list ℕ) : Prop :=
  ∀ x, x ∈ l → is_prime x

-- Statement of the problem
theorem largest_prime_factor_5040_is_7 :
  all_primes prime_factorization_5040 ∧ 
  list.prod prime_factorization_5040 = 5040 ∧
  list.maximum prime_factorization_5040 = 7 :=
sorry

end largest_prime_factor_5040_is_7_l12_12954


namespace smallest_positive_period_l12_12689

def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem smallest_positive_period : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ 
  (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi :=
sorry

end smallest_positive_period_l12_12689


namespace min_sum_of_arith_seq_l12_12215

theorem min_sum_of_arith_seq 
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith_seq : ∀ n, a n = a 0 + (n - 1) * d) -- Arithmetic sequence definition
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) -- Sum of first n terms
  (h_a1 : a 1 = -11)
  (h_a4_a6 : a 4 + a 6 = -6) :
  ∃ n, S n ≤ S m ∀ m, n = 6 :=
by
  sorry

end min_sum_of_arith_seq_l12_12215


namespace gcd_1734_816_l12_12059

theorem gcd_1734_816 : Nat.gcd 1734 816 = 102 := by
  sorry

end gcd_1734_816_l12_12059


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12390

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12390


namespace monotonicity_tangent_points_l12_12279

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12279


namespace find_y_perpendicular_l12_12222

theorem find_y_perpendicular (y : ℝ) (A B : ℝ × ℝ) (a : ℝ × ℝ)
  (hA : A = (-1, 2))
  (hB : B = (2, y))
  (ha : a = (2, 1))
  (h_perp : (B.1 - A.1) * a.1 + (B.2 - A.2) * a.2 = 0) :
  y = -4 :=
sorry

end find_y_perpendicular_l12_12222


namespace smallest_n_l12_12688

def smallest_integer_n (n : ℤ) : Prop :=
  n^2 - 13 * n + 36 ≤ 0

theorem smallest_n : ∃ n, smallest_integer_n n ∧ ∀ m, smallest_integer_n m → n ≤ m :=
begin
  use 4,
  split,
  { -- Prove that 4 satisfies the condition
    simp [smallest_integer_n],
    norm_num
  },
  { -- Prove that 4 is the smallest integer satisfying the condition
    intros m h,
    simp [smallest_integer_n] at *,
    sorry
  }
end

end smallest_n_l12_12688


namespace probability_interval_l12_12033

-- Definitions for normal distribution and probability intervals according to conditions
def normal_distribution (μ σ : ℝ) (X : ℝ → ℝ) := 
  -- X follows normal distribution N(μ, σ^2)
  ∀ (x : ℝ), X x = 1 / (σ * real.sqrt (2 * real.pi)) * exp (-(x - μ)^2 / (2 * σ^2))

-- Given the conditions of the problem:
def empirical_rule_1σ (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  (∫ x in (μ - σ), (μ + σ), X x) = 0.6826

def empirical_rule_2σ (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  (∫ x in (μ - 2 * σ), (μ + 2 * σ), X x) = 0.9544

def empirical_rule_3σ (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  (∫ x in (μ - 3 * σ), (μ + 3 * σ), X x) = 0.9974

-- Proof statement ensuring probability within specific interval
theorem probability_interval (μ σ : ℝ) (X : ℝ → ℝ)
  (h1 : normal_distribution μ σ X)
  (h2 : empirical_rule_1σ μ σ X)
  (h3 : empirical_rule_2σ μ σ X)
  (h4 : empirical_rule_3σ μ σ X) :
  (∫ x in 110, 120, X x) = 0.1359 :=
sorry

end probability_interval_l12_12033


namespace length_of_plot_l12_12645

-- Define the conditions
def width : ℝ := 60
def num_poles : ℕ := 60
def dist_between_poles : ℝ := 5
def num_intervals : ℕ := num_poles - 1
def perimeter : ℝ := num_intervals * dist_between_poles

-- Define the theorem and the correctness condition
theorem length_of_plot : 
  perimeter = 2 * (length + width) → 
  length = 87.5 :=
by
  sorry

end length_of_plot_l12_12645


namespace QK_calculation_l12_12447

theorem QK_calculation (PQRS : Type) [square PQRS] (M : point PQRS) (J K : point PQRS)
  (PQ : line PQRS) (side_length : ℝ) (center : midpoint PQRS M) (J_on_PQ : point_on_line PQRS PQ J)
  (K_on_PQ : point_on_line PQRS PQ K) (PJ_lt_QK : PJ < QK) (angle_JMK : angle J M K = 30)
  (JK_length : dist J K = 350) :
  ∃ v w u : ℕ, u ∉ (λ p, p * p : ℕ → ℕ) ∧ QK = v + w * real.sqrt u ∧
  v + w + u = 603 := by
  sorry

end QK_calculation_l12_12447


namespace coefficient_x7_in_polynomial_expansion_l12_12466

noncomputable theory

def polynomial := (1 + 2 * x - x^2)^4

theorem coefficient_x7_in_polynomial_expansion :
  polynomial.coeff 7 = -8 := 
sorry

end coefficient_x7_in_polynomial_expansion_l12_12466


namespace math_problem_l12_12789

theorem math_problem (x : ℝ) (h : x * (real.sqrt (x^2 - 1)) + 1 / (x + (real.sqrt (x^2 - 1))) = 21) :
  x^2 * (real.sqrt (x^4 - 1)) + 1 / (x^2 + (real.sqrt (x^4 - 1))) = 10201 / 400 := sorry

end math_problem_l12_12789


namespace sum_of_fourth_powers_l12_12180

def equation_to_solve (x : ℝ) : Prop := x^243 = 243^81

theorem sum_of_fourth_powers : 
  (∀ x : ℝ, equation_to_solve x → x^4) = 3^(20/3) := 
sorry

end sum_of_fourth_powers_l12_12180


namespace monotonicity_and_tangent_intersection_l12_12249

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12249


namespace count_arrangements_l12_12456

theorem count_arrangements (students : Fin 5) (A B : Fin 5) (hA_ne_B : A ≠ B) : 
  ∃ n : Nat, n = 72 ∧ 
    n = (5.factorial - (4.factorial * 2.factorial)) := 
sorry

end count_arrangements_l12_12456


namespace fraction_power_multiply_l12_12679

theorem fraction_power_multiply :
  ((1 : ℚ) / 3)^4 * ((1 : ℚ) / 5) = (1 / 405 : ℚ) :=
by sorry

end fraction_power_multiply_l12_12679


namespace find_pairs_l12_12933

def Point := (ℤ × ℤ)

def P : Point := (1, 1)
def Q : Point := (4, 5)
def valid_pairs : List Point := [(4, 1), (7, 5), (10, 9), (1, 5), (4, 9)]

def area (P Q R : Point) : ℚ :=
  (1 / 2 : ℚ) * ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)).natAbs : ℚ)

theorem find_pairs :
  {pairs : List Point // ∀ (a b : ℤ), (0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 10 ∧ area P Q (a, b) = 6) ↔ (a, b) ∈ pairs} :=
  ⟨valid_pairs, by sorry⟩

end find_pairs_l12_12933


namespace right_triangle_area_l12_12558

theorem right_triangle_area (a b c : ℕ) (habc : a = 3 ∧ b = 4 ∧ c = 5) : 
  (a * a + b * b = c * c) → 
  1 / 2 * (a * b) = 6 :=
by
  sorry

end right_triangle_area_l12_12558


namespace infinite_series_sum_l12_12702

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) / 4^(n + 1)) + (∑' n : ℕ, 1 / 2^(n + 1)) = 13 / 9 := 
sorry

end infinite_series_sum_l12_12702


namespace sin_cos_proof_l12_12418

variable {θ a b : ℝ}

-- Condition
def cond (θ a b : ℝ) : Prop :=
  (sin θ)^6 / a + (cos θ)^6 / b = 1 / (a + b)

-- Statement to prove
theorem sin_cos_proof (h : cond θ a b) : 
  (sin θ)^12 / a^5 + (cos θ)^12 / b^5 = 1 / (a + b)^5 :=
sorry

end sin_cos_proof_l12_12418


namespace perimeter_ratio_l12_12422

-- Given triangles ABC and DEF
variable {A B C D E F : Type*}

-- ABC and DEF are triangles (represented by 6 points)
@[ext] structure Triangle (P Q R : Type*) :=
(side : (P × P × P))

variable [has_add A] [has_add B] [has_add C] [has_add D] [has_add E] [has_add F]

-- Perimeter function for a Triangle
def perimeter {P Q R : Type*} [has_add P] (T : Triangle P Q R) : P :=
  match T with
  | ⟨(a, b, c)⟩ => a + b + c

-- Similarity of triangles definition
def are_similar {P Q R S T U} (Δ₁ : Triangle P Q R) (Δ₂ : Triangle S T U) (k : ℕ) : Prop :=
  k * perimeter Δ₁ = perimeter Δ₂

-- Conditions
axiom similarity (ΔABC : Triangle A B C) (ΔDEF : Triangle D E F) : are_similar ΔABC ΔDEF 2

-- Prove that the ratio of the perimeters is 1/2
theorem perimeter_ratio (ΔABC : Triangle A B C) (ΔDEF : Triangle D E F) 
  (h : are_similar ΔABC ΔDEF 2) : 
  perimeter ΔABC = (1 / 2) * perimeter ΔDEF := 
sorry

end perimeter_ratio_l12_12422


namespace domain_of_sqrt_plus_ln_l12_12905

theorem domain_of_sqrt_plus_ln (x : ℝ) : 0 ≤ x ∧ x < 1 ↔ ∃ y, y = sqrt x + Real.log (1 - x) :=
by
    sorry

end domain_of_sqrt_plus_ln_l12_12905


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12338

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12338


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12308

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12308


namespace bus_ride_time_l12_12862

/--
Given:
- Total trip time is 480 minutes.
- Walking time is 15 minutes.
- Waiting time is twice the walking time.
- Train ride time is 360 minutes.
Prove:
- The bus ride time is 75 minutes.
-/
theorem bus_ride_time (total_time : ℕ) (walk_time : ℕ) (train_ride_time : ℕ) (bus_ride_time : ℕ) :
  total_time = 480 →
  walk_time = 15 →
  train_ride_time = 360 →
  bus_ride_time = total_time - (train_ride_time + walk_time + 2 * walk_time) →
  bus_ride_time = 75 :=
by {
  intros,
  unfold bus_ride_time,
  sorry
}

end bus_ride_time_l12_12862


namespace find_values_l12_12922

theorem find_values (a b c : ℕ) 
    (h1 : a + b + c = 1024) 
    (h2 : c = b - 88) 
    (h3 : a = b + c) : 
    a = 712 ∧ b = 400 ∧ c = 312 :=
by {
    sorry
}

end find_values_l12_12922


namespace area_PQRS_is_90_l12_12458

-- Definitions of the conditions
variables (PQRS : Type) [rectangle PQRS]
variables (M N : PQRS) (MQ PS SQ SR : ℝ)
variables (M_on_PQ : M ∈ PQ) (N_on_PS : N ∈ PS)
variables (MQ_eq : MQ = 3) (PN_eq : PN = 5)
variables (angle_S_trisected : is_trisected (∠PQR) (∠SM) (∠SN))

-- Translate the proof problem
theorem area_PQRS_is_90 :
  (∃ PQRS, angle_S_trisected ∧ M_on_PQ ∧ N_on_PS ∧ MQ_eq ∧ PN_eq) →
  area PQRS = 90 :=
begin
  sorry
end

end area_PQRS_is_90_l12_12458


namespace logarithm_problem_l12_12984

noncomputable def log_base (a b : ℝ) := Real.log b / Real.log a

theorem logarithm_problem :
  log_base 0.5 0.125 + log_base 2 (log_base 3 (log_base 4 64)) = 3 :=
by
  have h1 : log_base 0.5 0.125 = 3 := 
    sorry, -- this follows from the computation log_base 0.5 0.125 = 3
  have h2 : log_base 4 64 = 3 := 
    sorry, -- this follows from the computation log_base 4 64 = 3
  have h3 : log_base 3 3 = 1 := 
    sorry, -- this follows from the computation log_base 3 3 = 1
  have h4 : log_base 2 1 = 0 := 
    sorry, -- this follows from the computation log_base 2 1 = 0
  
  rw [h1, h2, h3, h4],
  -- Applying all the computed values to get the result
  linarith

end logarithm_problem_l12_12984


namespace probability_ratio_l12_12167

theorem probability_ratio (cards : finset (fin 50)) 
  (h1 : ∀ n : ℕ, n ∈ cards → n ≤ 50)
  (h2 : ∀ n ∈ cards, (1 ≤ n) ∧ (n ≤ 10))
  (each_number_five : ∀ n : fin 10, (cards.filter (λ c, (c % 10) = n)).card = 5) :
  let p := 10 / (nat.choose 50 5) in
  let q := (90 * 5 * 5) / (nat.choose 50 5) in
  q / p = 225 :=
by sorry

end probability_ratio_l12_12167


namespace intersection_M_N_l12_12410

def M : Set ℝ := {x | (0 ≤ x ∧ x ≤ 1)}
def N : Set ℝ := Icc (-1) 1

theorem intersection_M_N :
  M ∩ N = Icc 0 1 :=
by
  sorry

end intersection_M_N_l12_12410


namespace problem_a_problem_b_problem_c_problem_d_l12_12759
-- Lean 4 code for the proof problems

noncomputable def z1 : ℂ := -2 + 5 * Complex.i
noncomputable def z2 : ℂ := 3 - 4 * Complex.i

theorem problem_a : z1 + z2 = 1 + Complex.i := sorry
theorem problem_b : z2 - z1 = 5 - 9 * Complex.i := sorry
theorem problem_c : z1 * z2 = 14 + 23 * Complex.i := sorry
theorem problem_d : z1 / z2 = -26 / 25 + (7 / 25) * Complex.i := sorry

end problem_a_problem_b_problem_c_problem_d_l12_12759


namespace monotonicity_and_tangent_intersection_l12_12243

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12243


namespace slices_served_yesterday_l12_12646

theorem slices_served_yesterday
  (lunch_slices : ℕ)
  (dinner_slices : ℕ)
  (total_slices_today : ℕ)
  (h1 : lunch_slices = 7)
  (h2 : dinner_slices = 5)
  (h3 : total_slices_today = 12) :
  (total_slices_today - (lunch_slices + dinner_slices) = 0) :=
by {
  sorry
}

end slices_served_yesterday_l12_12646


namespace perfect_square_trinomial_solution_l12_12426

theorem perfect_square_trinomial_solution (m : ℝ) :
  (∃ a : ℝ, (∀ x : ℝ, x^2 - 2*(m+3)*x + 9 = (x - a)^2))
  → m = 0 ∨ m = -6 :=
by
  sorry

end perfect_square_trinomial_solution_l12_12426


namespace number_of_bricks_l12_12107

theorem number_of_bricks (b1_hours b2_hours combined_hours: ℝ) (reduction_rate: ℝ) (x: ℝ):
  b1_hours = 12 ∧ 
  b2_hours = 15 ∧ 
  combined_hours = 6 ∧ 
  reduction_rate = 15 ∧ 
  (combined_hours * ((x / b1_hours) + (x / b2_hours) - reduction_rate) = x) → 
  x = 1800 :=
by
  sorry

end number_of_bricks_l12_12107


namespace bowtie_area_fraction_l12_12937

-- Definition of a triangle and its medians
structure Triangle :=
  (A B C : Point)

structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Definition of a median function
def median (T : Triangle) (midpoint1 midpoint2 : Point) : Line :=
  sorry -- Placeholder for the actual median definition

-- Definition of a centroid function
def centroid (T : Triangle) : Point :=
  sorry -- Placeholder for the centroid definition

-- Definition to calculate the area of sub-triangles formed
def triangleArea (tri : Triangle) : ℝ :=
  sorry -- Placeholder for area calculation

-- Given a triangle with area 1
axiom triangleArea_is_one (T : Triangle) : triangleArea T = 1

-- Proving the bowtie shape area fraction within the triangle
theorem bowtie_area_fraction (T : Triangle) (midpoint1 midpoint2 trisect_points : list Point) 
  (h_midpoints: midpoint1 = midpoint (T.A, T.B) ∧ midpoint2 = midpoint (T.B, T.C)) 
  (h_trisect : trisect_points = [trisect(T.C)]) : 
  let bowtie_area := calculateBowtieArea T midpoint1 midpoint2 trisect_points in
  bowtie_area = (1 / 30 : ℝ) :=
by
  sorry -- Proof here 

end bowtie_area_fraction_l12_12937


namespace complex_power_equality_l12_12758

theorem complex_power_equality :
  (\(\frac{\sqrt{2}}{2} + \frac{\sqrt{2}}{2}i\)^8 = 1) := sorry

end complex_power_equality_l12_12758


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12386

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12386


namespace largest_non_sum_of_even_multiple_60_and_composite_l12_12063

theorem largest_non_sum_of_even_multiple_60_and_composite :
  ∃ n : ℕ, n = 127 ∧ (∀ (a b : ℕ), a > 0 ∧ a % 2 = 0 ∧ b > 0 ∧ b < 60 ∧ ¬ is_prime b → n ≠ 60 * a + b) :=
sorry

end largest_non_sum_of_even_multiple_60_and_composite_l12_12063


namespace wave_number_prob_l12_12639

def is_wave_number (a1 a2 a3 a4 a5 : ℕ) : Prop :=
  a1 < a2 ∧ a2 > a3 ∧ a3 < a4 ∧ a4 > a5

def all_digits (a1 a2 a3 a4 a5 : ℕ) : Prop :=
  {a1, a2, a3, a4, a5} = {1, 2, 3, 4, 5} ∧ 
  list.nodup [a1, a2, a3, a4, a5]

def probability_of_wave_number : Prop :=
  ∃ (a1 a2 a3 a4 a5 : ℕ), all_digits a1 a2 a3 a4 a5 ∧ is_wave_number a1 a2 a3 a4 a5 → 
  (number of valid "wave numbers" = 2 / total number of permutations = 5!) = 2 / 15

theorem wave_number_prob : probability_of_wave_number := sorry

end wave_number_prob_l12_12639


namespace find_midpoint_locus_l12_12716

/- Conditions -/
/-- Two given skew perpendicular lines -/
structure skew_perpendicular_lines (l1 l2 : line) :=
  (skew : ¬ ∃ (P : point), P ∈ l1 ∧ P ∈ l2)
  (perpendicular : ∃ (P1 P2 : point) (d1 d2 : vector), P1 ∈ l1 ∧ P2 ∈ l2 ∧ (d1, d2) perp)

/-- The length of the segment -/
noncomputable def segment_length (d : ℝ) := d

/- Correct Answer: The geometric locus of the midpoints of the segments -/
noncomputable def midpoint_locus (l1 l2 : line) (d : ℝ) (a : ℝ) : set point :=
  {M : point | ∃ (P1 P2 : point), P1 ∈ l1 ∧ P2 ∈ l2 ∧ dist P1 P2 = d ∧ 
    M = midpoint P1 P2 ∧ dist M (midpoint (projection l1 P1) (projection l2 P2)) = (sqrt (d^2 - a^2)) / 2}

theorem find_midpoint_locus (l1 l2 : line) (d : ℝ) (a : ℝ) (h1 : skew_perpendicular_lines l1 l2)
  (h2 : segment_length d) :
  midpoint_locus l1 l2 d a = circle_centered_midpoint_with_radius_midpoint_projection (midpoint (projection l1 P1) (projection l2 P2)) ((sqrt (d^2 - a^2)) / 2) :=
sorry

end find_midpoint_locus_l12_12716


namespace pyramid_cross_section_area_l12_12978

-- Define the pyramid properties and conditions
structure Pyramid :=
  (a b c : ℝ)    -- sides of the base triangle
  (ta : ℝ)      -- height of the pyramid
  (a_eq_b_eq_c : a = b ∧ b = c ∧ a = sqrt 3)
  (ta_eq_a: ta = a)

-- Define the conditions of the plane intersection
structure IntersectionConditions :=
  (sphere_center : ℝ)
  (angle_with_base : ℝ)
  (intersection_point_M : ℝ -> ℝ -> Prop) -- M such that MB = 2AM
  (distance_from_A : ℝ)

-- Define the area of cross-section that needs to be proven
noncomputable def cross_section_area (p : Pyramid) (ic : IntersectionConditions) : ℝ :=
  if ic.angle_with_base = (60 : ℝ) / 180 * π ∧ p.a_eq_b_eq_c ∧ p.ta_eq_a ∧ ic.distance_from_A = 0.25 then
    (11 * sqrt 3) / 30
  else
    0

-- Prove the area under the given conditions
theorem pyramid_cross_section_area (p : Pyramid) (ic : IntersectionConditions)
  (h1 : ic.angle_with_base = (60 : ℝ) / 180 * π)
  (h2 : p.a_eq_b_eq_c ∧ p.ta_eq_a)
  (h3 : ic.distance_from_A = 0.25)
  : cross_section_area p ic = (11 * sqrt 3) / 30 :=
sorry -- Proof skipped

end pyramid_cross_section_area_l12_12978


namespace part1_monotonicity_part2_tangent_intersection_l12_12257

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12257


namespace problem1_problem2_problem3_l12_12459

/-- Given the total cost of type A and type B notebooks is 288 yuan, prove that 
9 type A notebooks and 21 type B notebooks are purchased. -/
theorem problem1 (x : ℕ) (H : 11 * x + 9 * (30 - x) = 288) : x = 9 ∧ (30 - x) = 21 :=
begin
  have : 2 * x = 18,
  { linarith },
  exact ⟨by linarith, by linarith⟩,
end

/-- Prove that the number of type C notebooks purchased and the total cost 
when purchasing type A, B, and C notebooks are correctly expressed. -/
theorem problem2 (m n : ℕ) : (30 - m - n) = 30 - m - n ∧ (11 * m + 9 * n + 6 * (30 - m - n)) = 5 * m + 3 * n + 180 :=
by linarith

/-- Given the total cost of purchasing type A, B, and C notebooks is 188 yuan, prove that m = 1 
(assuming that the solutions must be positive integers). -/
theorem problem3 (m n : ℕ) (Hm : 5 * m + 3 * n = 8) : m = 1 :=
begin
  have : 5 * m = 5,
  { linarith },
  exact (mul_left_inj' (by norm_num : 5 ≠ 0)).1 this
end

end problem1_problem2_problem3_l12_12459


namespace elliptical_eccentricity_correct_equation_of_line_correct_l12_12217

noncomputable def eccentricity_of_ellipse
(a b : ℝ) (h : a > b ∧ a > 0 ∧ b > 0)
(F1 F2 : (ℝ × ℝ)) (hfoci : F1 = (-1, 0) ∧ F2 = (1, 0))
(A : ℝ × ℝ) (hA : A = (4/3, 1/3)) :
    ℝ := 
a / (1 + 1)

theorem elliptical_eccentricity_correct 
(a b : ℝ) (h : a = sqrt 2 ∧ b * b + 1 = 1)
(F1 F2 : (ℝ × ℝ)) 
(hfoci : F1 = (-1, 0) ∧ F2 = (1, 0))
(A : ℝ × ℝ) 
(hA : A = (4/3, 1/3))
(e : ℝ)
(he : e = 1 / sqrt 2) :
eccentricity_of_ellipse a b h F1 F2 hfoci A hA = e := 
sorry

noncomputable def line_through_point_with_slope 
(k : ℝ) (x1 : ℝ) :
    (ℝ × ℝ) → Prop :=
λ P, ∃ y, P = (x1, y) ∧ y = k * (P.1 - x1)

theorem equation_of_line_correct
(F2 : ℝ × ℝ) (P Q : ℝ × ℝ)
(hPQ : F2 = (1, 0) ∧ 
  ∃ k : ℝ, ∢ P = F2 ∧ ∢ Q = F2)
(Hp1 : (P.1 + 1) * (Q.1 + 1) + P.2 * Q.2 = 0)
(k : ℝ) (hk : k = sqrt 7 / 7) :
(line_through_point_with_slope k 1 F2) ∨
(line_through_point_with_slope (-k) 1 F2) :=
sorry

end elliptical_eccentricity_correct_equation_of_line_correct_l12_12217


namespace angle_CBD_is_30_l12_12574

-- Definitions based on given conditions
variables {A B C D M : Point}
variables {circle : Circle}
variables (diameterAB : Segment) 
variables (outsideC : Point)
variables (D M : Point)
variables (intersect_AC_D : LineSegment AC ∩ Circle = {D})
variables (intersect_BC_M : LineSegment BC ∩ Circle = {M})
variables (area_ratio : Area(Δ DCM) / Area(Δ ACB) = 1 / 4)

-- Define angle θ
noncomputable def θ : ℝ := 30

-- Prove θ = 30
theorem angle_CBD_is_30 
  (diameter_AB : AB = circle.diameter)
  (C_outside_circle : C ∉ circle)
  (AC_intersect_D : intersects AC circle D)
  (BC_intersect_M : intersects BC circle M)
  (area_condition : Area(Δ DCM) / Area(Δ ACB) = 1 / 4) :
  ∠CBD = θ :=
begin
  sorry
end

end angle_CBD_is_30_l12_12574


namespace distance_marcos_needs_to_cross_l12_12504

theorem distance_marcos_needs_to_cross (time_in_hours : ℕ → ℝ) (speed : ℝ) (h_time : time_in_hours 10 = 1 / 6) (h_speed : speed = 30) : 
  speed * (time_in_hours 10) = 5 :=
by
  rw [h_time, h_speed]
  norm_num
  sorry

end distance_marcos_needs_to_cross_l12_12504


namespace largest_x_quadratic_inequality_l12_12684

theorem largest_x_quadratic_inequality : 
  ∃ (x : ℝ), (x^2 - 10 * x + 24 ≤ 0) ∧ (∀ y, (y^2 - 10 * y + 24 ≤ 0) → y ≤ x) :=
sorry

end largest_x_quadratic_inequality_l12_12684


namespace exists_large_abs_value_l12_12221

-- Definitions for the problem
variables {n : ℕ} {a : ℕ → ℤ}  -- Polynomial coefficients
variables {x : ℕ → ℤ}          -- Given sequence of integers
hypothesis (h : ∀ i j, i < j → x i < x j)  -- Condition
noncomputable def P (x : ℤ) := ∑ i in finset.range (n + 1), a i * x^(n - i)

-- Statement to prove
theorem exists_large_abs_value : ∃ j (h : j ≤ n), 
  |P (x j)| ≥ nat.factorial n / 2^n :=
sorry

end exists_large_abs_value_l12_12221


namespace max_int_value_of_x_l12_12036

noncomputable def f (x : ℝ) := log x - 2 * x + 11

theorem max_int_value_of_x (x : ℝ) (h_eq : f x = 0) (h_ineq : x ≤ x) : ∃ n : ℤ, (x ∈ (5, 6)) ∧ ↑n ≤ x ∧ n = 5 :=
by
  sorry

end max_int_value_of_x_l12_12036


namespace concurrency_of_YZ_EF_BC_l12_12484

-- Definitions of altitudes and antipodal points based on conditions
def is_altitude (B E C A : Point) (tri : Triangle A B C) : Prop :=
  is_perpendicular_line_from_to B E A C

def is_antipodal (D A : Point) (circ : Circumcircle A B C) : Prop :=
  is_diameter A D circ

-- Main theorem statement
theorem concurrency_of_YZ_EF_BC {A B C D E F Y Z : Point} (ABC_circ : Circumcircle A B C)
  (h_BE : is_altitude B E C A) (h_CF : is_altitude C F A B) (h_D : is_antipodal D A ABC_circ)
  (h_DE_Y : on_circumcircle (line D E) (Y) ABC_circ)
  (h_DF_Z : on_circumcircle (line D F) (Z) ABC_circ) :
  concurrent (line Y Z) (line E F) (line B C) :=
sorry

end concurrency_of_YZ_EF_BC_l12_12484


namespace abs_1_5_minus_sqrt2_eq_l12_12008

-- Conditions
def approx_sqrt2 : ℝ := 1.414

-- Statement of the problem
theorem abs_1_5_minus_sqrt2_eq (h : ∥real.sqrt 2 - approx_sqrt2∥ < ε) : 
  |(1.5 - real.sqrt 2)| = 1.5 - real.sqrt 2 :=
by
  sorry

end abs_1_5_minus_sqrt2_eq_l12_12008


namespace area_of_QCA_l12_12156

noncomputable def area_of_triangle (x p : ℝ) (hx_pos : 0 < x) (hp_bounds : 0 < p ∧ p < 15) : ℝ :=
  1 / 2 * x * (15 - p)

theorem area_of_QCA (x : ℝ) (p : ℝ) (hx_pos : 0 < x) (hp_bounds : 0 < p ∧ p < 15) :
  area_of_triangle x p hx_pos hp_bounds = 1 / 2 * x * (15 - p) :=
sorry

end area_of_QCA_l12_12156


namespace period_of_cos2x_l12_12686

def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem period_of_cos2x : ∃ T > 0, ∀ x, f (x + T) = f x :=
by
  use π
  sorry

end period_of_cos2x_l12_12686


namespace theta_in_range_l12_12765

theorem theta_in_range (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π)
  (h3 : ∀ x : ℝ, (cos θ) * x^2 - (4 * sin θ) * x + 6 > 0) :
  θ < π / 3 :=
sorry

end theta_in_range_l12_12765


namespace ones_digit_sum_l12_12963

theorem ones_digit_sum (n : ℕ) (h : 2023 = 4 * (2023 / 4) + 3) : 
  (∑ k in Finset.range (2023 + 1), k^2023) % 10 = 6 :=
by
  sorry

end ones_digit_sum_l12_12963


namespace monotonicity_tangent_intersection_points_l12_12378

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12378


namespace locus_is_plane_l12_12177

noncomputable def locus_of_points (A B C: EuclideanSpace ℝ 3) (p q r d : ℝ) (X : EuclideanSpace ℝ 3) : Prop :=
  p * (dist_sq X A) + q * (dist_sq X B) + r * (dist_sq X C) = d

theorem locus_is_plane (A B C : EuclideanSpace ℝ 3) (p q r d : ℝ) (X : EuclideanSpace ℝ 3)
  (h_sum : p + q + r = 0) : 
  locus_of_points A B C p q r d X ↔ 
  ∃ α β δ : ℝ, α * X.1 + β * (X.2) + δ = 0 :=
sorry

end locus_is_plane_l12_12177


namespace Person3IsTriussian_l12_12508

def IsTriussian (person : ℕ) : Prop := if person = 3 then True else False

def Person1Statement : Prop := ∀ i j k : ℕ, i = 1 → j = 2 → k = 3 → (IsTriussian i = (IsTriussian j ∧ IsTriussian k) ∨ (¬IsTriussian j ∧ ¬IsTriussian k))

def Person2Statement : Prop := ∀ i j : ℕ, i = 2 → j = 3 → (IsTriussian j = False)

def Person3Statement : Prop := ∀ i j : ℕ, i = 3 → j = 1 → (IsTriussian j = False)

theorem Person3IsTriussian : (Person1Statement ∧ Person2Statement ∧ Person3Statement) → IsTriussian 3 :=
by 
  sorry

end Person3IsTriussian_l12_12508


namespace largest_prime_factor_5040_l12_12957

theorem largest_prime_factor_5040 : ∃ p, Nat.Prime p ∧ p ∣ 5040 ∧ (∀ q, Nat.Prime q ∧ q ∣ 5040 → q ≤ p) := by
  use 7
  constructor
  · exact Nat.prime_7
  constructor
  · exact dvd.intro (2^4 * 3^2 * 5) rfl
  · intros q hq
    cases hq with hq1 hq2
    exact Nat.le_of_dvd (Nat.pos_of_ne_zero (λ hq3, by linarith)) hq2
  sorry

end largest_prime_factor_5040_l12_12957


namespace Juan_run_time_l12_12829

theorem Juan_run_time
  (d : ℕ) (s : ℕ) (t : ℕ)
  (H1: d = 80)
  (H2: s = 10)
  (H3: t = d / s) :
  t = 8 := 
sorry

end Juan_run_time_l12_12829


namespace smallest_positive_period_l12_12720

-- Define the function
def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

-- Definition of the period formula
def period (ω : ℝ) : ℝ := 2 * Real.pi / ω

-- State the theorem
theorem smallest_positive_period : period 2 = Real.pi := by
  sorry

end smallest_positive_period_l12_12720


namespace total_teeth_removed_l12_12189

theorem total_teeth_removed :
  let first_removed := 32 * 1 / 4 + 2 + 1 in
  let second_removed := 32 * 3 / 8 + 3 + 2 in
  let third_removed := 32 * 1 / 2 + 1 + 3 in
  let fourth_removed := 4 + 5 + 2 in
  first_removed + second_removed + third_removed + fourth_removed = 59 :=
by
  sorry

end total_teeth_removed_l12_12189


namespace theta_value_l12_12155

open Set Real

def distance_set (S T : Set (ℝ × ℝ)) : Set ℝ :=
  {d | ∃ P ∈ S, ∃ Q ∈ T, d = dist P Q}

def S (k θ : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x : ℝ, p = (x, k * x + (sqrt 5) * tan θ)}

def T : Set (ℝ × ℝ) :=
  {p | ∃ x : ℝ, p = (x, sqrt (4 * x^2 + 1))}

theorem theta_value (k : ℝ) (θ : ℝ) (h : distance_set (S k θ) T = Ioi 1) :
    θ = -π / 4 :=
  sorry

end theta_value_l12_12155


namespace sum_of_real_solutions_eq_l12_12721

/-- The sum of all real solutions to the given equation is equal to 4.5. -/
theorem sum_of_real_solutions_eq :
  let f := λ x : ℝ, (x - 3) / (x^2 + 5 * x + 4),
      g := λ x : ℝ, (x - 6) / (x^2 - 12 * x + 1) in
  (∀ x : ℝ, f x = g x → x ∈ {x | 2 * x^2 - 9 * x - 3 = 0}) →
  ∑ s in {x | 2 * x^2 - 9 * x - 3 = 0}, id s = 4.5 :=
by
  let f (x : ℝ) := (x - 3) / (x^2 + 5 * x + 4)
  let g (x : ℝ) := (x - 6) / (x^2 - 12 * x + 1)
  intro h
  sorry

end sum_of_real_solutions_eq_l12_12721


namespace time_to_cook_one_potato_l12_12634

-- Definitions for the conditions
def total_potatoes : ℕ := 16
def cooked_potatoes : ℕ := 7
def remaining_minutes : ℕ := 45

-- Lean theorem that asserts the equivalence of the problem statement to the correct answer
theorem time_to_cook_one_potato (total_potatoes cooked_potatoes remaining_minutes : ℕ) 
  (h_total : total_potatoes = 16) 
  (h_cooked : cooked_potatoes = 7) 
  (h_remaining : remaining_minutes = 45) :
  (remaining_minutes / (total_potatoes - cooked_potatoes) = 5) :=
by
  -- Using sorry to skip proof
  sorry

end time_to_cook_one_potato_l12_12634


namespace family_ages_sum_today_l12_12580

theorem family_ages_sum_today (A B C D E : ℕ) (h1 : A + B + C + D = 114) (h2 : E = D - 14) :
    (A + 5) + (B + 5) + (C + 5) + (E + 5) = 120 :=
by
  sorry

end family_ages_sum_today_l12_12580


namespace ratio_of_x_to_y_l12_12794

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x + 2 * y) / (2 * x - y) = 5 / 4) : x / y = -13 / 2 := 
by 
  sorry

end ratio_of_x_to_y_l12_12794


namespace zero_vector_incorrect_statement_l12_12885

theorem zero_vector_incorrect_statement
  (zero_vec : ℝ → ℝ)
  (h1 : ∀ v: ℝ → ℝ, v ≠ zero_vec ∨ zero_vec.direction = arbitrary)
  (h2 : zero_vec.length = 0)
  (h3 : ∀ v: ℝ → ℝ, zero_vec.parallel_to v) :
  "The zero vector has no direction" = incorrect :=
sorry

end zero_vector_incorrect_statement_l12_12885


namespace intersection_points_lie_on_conic_l12_12832

noncomputable def cyclic_octagon := (A : ℕ → ℝ × ℝ) (hA : ∀ i, i ∈ finset.range 8 → 
  circle (A 1)) -- Assumes a cyclic octagon, circles require fourteen points.

noncomputable def points_of_intersection (A : ℕ → ℝ × ℝ) (i : ℕ) (mod8 : i % 8) :=
  intersection (line (A i, A (i+1))) (line (A (i+3), A (i+4)))

theorem intersection_points_lie_on_conic (A : ℕ → ℝ × ℝ) (hA : ∀ i, i ∈ finset.range 8 → circle (A 1)) :
  ∃ C : conic, ∀ i : ℕ, i ∈ finset.range 8 → on_conic (points_of_intersection (A i (i % 8))) C :=
sorry

end intersection_points_lie_on_conic_l12_12832


namespace problem_statement_l12_12761

-- Definitions based on the conditions provided
def ω : ℝ := 1
def f (x : ℝ) : ℝ := 4 * Real.cos (ω * x) * Real.sin (ω * x - Real.pi / 6)

-- Statement of the mathematical problem
theorem problem_statement (x : ℝ) :
  (∀ x ∈ Ioo 0 Real.pi, ((x ∈ Ioo 0 (Real.pi / 3)) ∨ (x ∈ Ioo (5 * Real.pi / 6) Real.pi)) →
    ∀ x₁ x₂ ∈ Ioo 0 Real.pi, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x ∈ Icc (Real.pi / 8) (3 * Real.pi / 8), 
   (f (Real.pi / 8) = 1) ∧
   (f (3 * Real.pi / 8) = (Real.sqrt 6 - Real.sqrt 2) / 2 - 1)) := 
sorry

end problem_statement_l12_12761


namespace least_possible_value_of_n_l12_12834

theorem least_possible_value_of_n (n : ℕ) (a b : ℕ → ℚ) 
    (h : ∀ x : ℝ, x^2 + x + 4 = ∑ i in finset.range n, (a i * x + b i)^2) : n = 5 :=
begin
  sorry
end

end least_possible_value_of_n_l12_12834


namespace math_problem_l12_12537

-- Define the function f and its inverse f_inv
variable {α : Type} [Field α] 

noncomputable def f (x : α) : α := sorry
noncomputable def f_inv (y : α) : α := sorry

-- Conditions and hypothesis
theorem math_problem (h : (2 : α, 3 : α) ∈ { (x, y) | y = 2 * f x }) : 
  ( (3/2 : α, (f_inv (3/2) / 4) : α) ∈ { (x, y) | y = (f_inv x) / 4 } ) ∧ 
  (3/2 + (f_inv (3/2) / 4) = 2) :=
sorry

end math_problem_l12_12537


namespace train_speed_l12_12943

noncomputable def speed_of_each_train (v : ℕ) : ℕ := 27

theorem train_speed
  (length_of_each_train : ℕ)
  (crossing_time : ℕ)
  (crossing_condition : 2 * (length_of_each_train * crossing_time) / (2 * crossing_time) = 15 / 2)
  (conversion_factor : ∀ n, 1 = 3.6 * n → ℕ) :
  speed_of_each_train 27 = 27 :=
by
  exact rfl

end train_speed_l12_12943


namespace hyperbola_with_common_foci_l12_12707

noncomputable def ellipse_eqn := ∀ (x y : ℝ), 
  x^2 / 9 + y^2 / 4 = 1

noncomputable def hyperbola_eqn := ∀ (x y : ℝ), 
  x^2 / 4 - y^2 = 1

noncomputable def ellipse_foci := ∃ (c : ℝ),
  ∃ (a b : ℝ), a = 3 ∧ b = 2 ∧ c = real.sqrt (a^2 - b^2) ∧ c = real.sqrt 5

noncomputable def hyperbola_foci_eccentricity := ∃ (a b : ℝ), 
  ∃ (c : ℝ), c = real.sqrt 5 ∧ a = 2 ∧ c = real.sqrt (a^2 + b^2) ∧ 
  real.sqrt(a^2 + b^2) / a = real.sqrt 5 / 2

theorem hyperbola_with_common_foci (x y : ℝ) 
  (ellipse : ∀ (x y : ℝ), x^2 / 9 + y^2 / 4 = 1)
  (ellipse_foci_conditions : ∃ (c : ℝ), ∃ (a b : ℝ), a = 3 ∧ b = 2 ∧ c = real.sqrt(a^2 - b^2) 
                            ∧ c = real.sqrt 5)
  (hyperbola_eccentricity_conditions : ∃ (a b : ℝ), ∃ (c : ℝ), c = real.sqrt 5 
                                        ∧ a = 2 ∧ c = real.sqrt(a^2 + b^2)
                                        ∧ real.sqrt(a^2 + b^2) / a = real.sqrt 5 / 2) :
  (x^2 / 4 - y^2 = 1) := 
sorry

end hyperbola_with_common_foci_l12_12707


namespace side_lengths_H1_l12_12509

noncomputable def inradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let A := sqrt (s * (s - a) * (s - b) * (s - c))
  A / s

theorem side_lengths_H1 :
  ∃ (a b c : ℝ), b = (a + c) / 2 ∧ 
  (∃ a2 b2 c2, a2 = a - 10 ∧ b2 = b - 10 ∧ c2 = c - 10 ∧ inradius a2 b2 c2 = inradius a b c - 5) ∧
  (∃ a3 b3 c3, a3 = a + 14 ∧ b3 = b + 14 ∧ c3 = c + 14 ∧ inradius a3 b3 c3 = inradius a b c + 5) ∧
  a = 25 ∧ b = 38 ∧ c = 51 :=
by
  sorry

end side_lengths_H1_l12_12509


namespace ratio_of_height_to_radius_max_volume_l12_12573

theorem ratio_of_height_to_radius_max_volume (r h : ℝ) (h_surface_area : 2 * Real.pi * r^2 + 2 * Real.pi * r * h = 6 * Real.pi) :
  (exists (max_r : ℝ) (max_h : ℝ), 2 * r * max_r + 2 * r * max_h = 6 * Real.pi ∧ 
                                  max_r = 1 ∧ 
                                  max_h = 2 ∧ 
                                  (max_h / max_r) = 2) :=
by
  sorry

end ratio_of_height_to_radius_max_volume_l12_12573


namespace ratio_son_grandson_l12_12506

-- Define the conditions
variables (Markus_age Son_age Grandson_age : ℕ)
axiom Markus_twice_son : Markus_age = 2 * Son_age
axiom sum_ages : Markus_age + Son_age + Grandson_age = 140
axiom Grandson_age_20 : Grandson_age = 20

-- Define the goal to prove
theorem ratio_son_grandson : (Son_age : ℚ) / Grandson_age = 2 :=
by
  sorry

end ratio_son_grandson_l12_12506


namespace find_coords_C_l12_12874

-- Define the coordinates of given points
def A : ℝ × ℝ := (13, 7)
def B : ℝ × ℝ := (5, -1)
def D : ℝ × ℝ := (2, 2)

-- The proof problem wrapped in a lean theorem
theorem find_coords_C (C : ℝ × ℝ) 
  (h1 : AB = AC) (h2 : (D.1, D.2) = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) :
  C = (-1, 5) :=
sorry

end find_coords_C_l12_12874


namespace combine_quadratic_radicals_l12_12795

theorem combine_quadratic_radicals (x : ℝ) (h : 3 * x + 5 = 2 * x + 7) : x = 2 :=
by
  sorry

end combine_quadratic_radicals_l12_12795


namespace find_conjugate_of_Z_l12_12734

   -- Defining the problem
   variable (a : ℝ) (i : ℂ) (Z : ℂ)

   -- Given conditions
   def complex_number : Prop := Z = a + i
   def sum_with_conjugate : Prop := Z + conj Z = 4

   -- Define the goal to be proven
   def conjugate_of_Z_correct : Prop := conj Z = 2 - i

   -- Final statement of the theorem
   theorem find_conjugate_of_Z 
      (a_real : a ∈ ℝ) 
      (Z_def : complex_number a Z) 
      (sum_cond : sum_with_conjugate Z) : 
      conjugate_of_Z_correct Z := 
   by
      sorry
   
end find_conjugate_of_Z_l12_12734


namespace solution_set_of_inequality_l12_12921

theorem solution_set_of_inequality (x : ℝ) : 3 - 2x > 7 → x < -2 := 
by 
  sorry

end solution_set_of_inequality_l12_12921


namespace part1_part2_l12_12735

-- Given conditions for part (1)
def condition1 (z : ℂ) : Prop := complex.abs z = real.sqrt 2
def condition2 (z : ℂ) : Prop := complex.im z = 1
def condition3 (z : ℂ) : Prop := complex.re z < 0

-- Result for part (1)
def result1 : ℂ := -1 + complex.I

-- Proof statement for part (1)
theorem part1 (z : ℂ) :
  condition1 z ∧ condition2 z ∧ condition3 z → z = result1 :=
by
  sorry

-- Given condition for part (2)
def mz2_is_pure_imaginary (m : ℝ) (z : ℂ) : Prop := ∃ k : ℝ, m ^ 2 + m + m * z ^ 2 = k * complex.I

-- Result for part (2)
def result2 : ℝ := -1

-- Proof statement for part (2)
theorem part2 (m : ℝ) (z : ℂ) :
  z = result1 ∧ mz2_is_pure_imaginary m z → m = result2 :=
by
  sorry

end part1_part2_l12_12735


namespace min_pos_period_f_l12_12025

def f (x : ℝ) : ℝ := abs (sin (2 * x) - cos (2 * x))

theorem min_pos_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ 
                                (∀ t > 0, (∀ x, f (x + t) = f x) → t ≥ π / 2) :=
sorry

end min_pos_period_f_l12_12025


namespace average_price_of_initial_fruit_l12_12669

theorem average_price_of_initial_fruit (A O : ℕ) (h1 : A + O = 10) (h2 : (40 * A + 60 * (O - 6)) / (A + O - 6) = 45) : 
  (40 * A + 60 * O) / 10 = 54 :=
by 
  sorry

end average_price_of_initial_fruit_l12_12669


namespace non_red_female_fish_count_l12_12132

theorem non_red_female_fish_count :
  ∀ (total_fish male_fraction red_female_fraction : ℚ),
  total_fish = 45 → male_fraction = 2/3 → red_female_fraction = 1/3 →
  let male_fish := male_fraction * total_fish in
  let total_female_fish := total_fish - male_fish in
  let red_female_fish := red_female_fraction * total_female_fish in
  let non_red_female_fish := total_female_fish - red_female_fish in
  non_red_female_fish = 10 :=
by
  intros total_fish male_fraction red_female_fraction
  intros h1 h2 h3
  simp only
  sorry

end non_red_female_fish_count_l12_12132


namespace height_of_parallelogram_l12_12709

theorem height_of_parallelogram (A B H : ℕ) (hA : A = 308) (hB : B = 22) (h_eq : H = A / B) : H = 14 := 
by sorry

end height_of_parallelogram_l12_12709


namespace alternating_binomial_sum_l12_12700

theorem alternating_binomial_sum :
  (∑ k in Finset.range 51, (-1)^k * Nat.choose 50 k) = 0 :=
by sorry

end alternating_binomial_sum_l12_12700


namespace monotonicity_tangent_intersection_points_l12_12375

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12375


namespace valid_arrangements_l12_12453

def five_students := fin 5

def arrangements (students : fin 5) :=
  if ∃ (i j : fin 5), i ≠ j ∧ (i = 2 ∧ j = 4 ∨ i = 4 ∧ j = 2)
  then 72 else sorry

theorem valid_arrangements (students : fin 5) (h: ¬(∃ (i j : fin 5), i ≠ j ∧ (i = 2 ∧ j = 4 ∨ i = 4 ∧ j = 2))): 
  arrangements students = 72 :=
sorry

end valid_arrangements_l12_12453


namespace number_of_zeros_of_f_l12_12685

def f (x : ℝ) : ℝ := 
  if x = 0 then 0 
  else x - (1 / x)

theorem number_of_zeros_of_f : 
  {x : ℝ | f x = 0}.finite.count = 3 := 
by
  sorry

end number_of_zeros_of_f_l12_12685


namespace sequence_non_existence_l12_12223

variable (α β : ℝ)
variable (r : ℝ)

theorem sequence_non_existence 
  (hαβ : α * β > 0) :  
  (∃ (x : ℕ → ℝ), x 0 = r ∧ ∀ n, x (n + 1) = (x n + α) / (β * (x n) + 1) → false) ↔ 
  r = - (1 / β) :=
sorry

end sequence_non_existence_l12_12223


namespace part1_monotonicity_part2_tangent_intersection_l12_12260

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12260


namespace exists_covering_triangle_l12_12534

open Real

noncomputable def area_of_triangle (A B C : Point) : ℝ := sorry

variables {S : set Point}
variables [finite S]

theorem exists_covering_triangle (h : ∀ (A B C : Point), A ∈ S → B ∈ S → C ∈ S → area_of_triangle A B C ≤ 1) :
  ∃ (A' B' C' : Point), area_of_triangle A' B' C' = 4 ∧ ∀ (P : Point), P ∈ S → P ∈ interior_of_triangle A' B' C' :=
sorry

end exists_covering_triangle_l12_12534


namespace simplify_expression_l12_12530

variable (a b : ℤ)

theorem simplify_expression : 
  (32 * a + 45 * b) + (15 * a + 36 * b) - (27 * a + 41 * b) = 20 * a + 40 * b := 
by sorry

end simplify_expression_l12_12530


namespace length_AB_equals_6_l12_12218

def ellipse_centered_at_origin_eccentricity_half (a b : ℕ) : Prop :=
  ∃ (E : ℝ × ℝ → Prop), 
  (∀ (x y : ℝ), E (x, y) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧ 
  (a = 4) ∧ 
  (b^2 = 12) ∧ 
  (1 / 2 = 2 / a)

def parabola_focus_and_directrix (parabola_focus : ℝ × ℝ) (directrix_eq : ℝ → Prop) : Prop :=
  parabola_focus = (2, 0) ∧ directrix_eq = (λ x, x = -2)

theorem length_AB_equals_6 :
  ellipse_centered_at_origin_eccentricity_half 4 (√12) ∧
  parabola_focus_and_directrix (2, 0) (λ x, x = -2) →
  let A : ℝ × ℝ := (-2, 3),
      B : ℝ × ℝ := (-2, -3) in
  |(A.2 - B.2)| = 6
:= by
  sorry

end length_AB_equals_6_l12_12218


namespace f_is_decreasing_max_k_value_l12_12242

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_is_decreasing : ∀ x > 0, (∃ y > x, f y < f x) :=
by
  sorry

theorem max_k_value : ∃ k : ℕ, (∀ x > 0, f x > k / (x + 1)) ∧ k = 3 :=
by
  sorry

end f_is_decreasing_max_k_value_l12_12242


namespace compute_alpha_l12_12843

noncomputable def alpha (β : ℂ) (x y : ℝ) : ℂ := (3 / 4) * x - (3 / 4) * y * complex.I

theorem compute_alpha (α β : ℂ) (x y : ℝ) 
  (h1 : α + β = x)
  (h2 : i * (α - 3 * β) = y)
  (h3 : β = 4 + 3 * complex.I)
  (h4 : x > 0)
  (h5 : y > 0) : 
  α = 12 - 3 * complex.I :=
by
  sorry

end compute_alpha_l12_12843


namespace total_num_wheels_l12_12800

-- Conditions
def num_cars : ℕ := 14
def num_bikes : ℕ := 5
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

-- Total number of wheels from cars
def total_wheels_car := num_cars * wheels_per_car

-- Total number of wheels from bikes
def total_wheels_bike := num_bikes * wheels_per_bike

-- The proof goal statement
theorem total_num_wheels : total_wheels_car + total_wheels_bike = 66 :=
by {
  sorry,
}

end total_num_wheels_l12_12800


namespace range_of_a_l12_12436

theorem range_of_a (a : ℝ) :
  (∀ x, (x - 2)/5 + 2 ≤ x - 4/5 ∨ x ≤ a) → a ≥ 3 :=
by
  sorry

end range_of_a_l12_12436


namespace largest_natural_gas_reserves_l12_12903
noncomputable def top_country_in_natural_gas_reserves : String :=
  "Russia"

theorem largest_natural_gas_reserves (countries : Fin 4 → String) :
  countries 0 = "Russia" → 
  countries 1 = "Finland" → 
  countries 2 = "United Kingdom" → 
  countries 3 = "Norway" → 
  top_country_in_natural_gas_reserves = countries 0 :=
by
  intros h_russia h_finland h_uk h_norway
  rw [h_russia]
  sorry

end largest_natural_gas_reserves_l12_12903


namespace area_PQRS_equals_275_plus_130_sqrt_11_l12_12981

variables {PQRS T U P' S' R' : Type}
variable [has_scalar ℝ PQRS]
variables (a b c : ℤ) (phi sin_phi : ℝ)

-- Definitions of the various given conditions
variable (rectangle_PQRS : is_rectangle PQRS)
variable (T_on_PQ : is_on T PQ)
variable (U_on_SR : is_on U SR)
variable (QT_lt_RU : QT < RU)
variable (S_maps_to_S' : maps_to S S' PQ)
variable (R_maps_to_R' : maps_to R R' PQ)
variable (angle_equivalence : angle_eq (angle P' S' R') (angle R' U T))
variable (PR'_eq_8 : distance P' R' = 8)
variable (QT_eq_26 : distance Q T = 26)

-- The statement to be proven
theorem area_PQRS_equals_275_plus_130_sqrt_11 :
  (area PQRS (a + b * sqrt c)) ∧ (a + b + c = 416) :=
by {
  sorry
}

end area_PQRS_equals_275_plus_130_sqrt_11_l12_12981


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12396

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12396


namespace cars_through_toll_booth_l12_12449

noncomputable def total_cars_in_week (n_mon n_tue n_wed n_thu n_fri n_sat n_sun : ℕ) : ℕ :=
  n_mon + n_tue + n_wed + n_thu + n_fri + n_sat + n_sun 

theorem cars_through_toll_booth : 
  let n_mon : ℕ := 50
  let n_tue : ℕ := 50
  let n_wed : ℕ := 2 * n_mon
  let n_thu : ℕ := 2 * n_mon
  let n_fri : ℕ := 50
  let n_sat : ℕ := 50
  let n_sun : ℕ := 50
  total_cars_in_week n_mon n_tue n_wed n_thu n_fri n_sat n_sun = 450 := 
by 
  sorry

end cars_through_toll_booth_l12_12449


namespace max_k_value_l12_12788

theorem max_k_value (x1 x2 x3 x4 : ℝ) (h1 : x1 > x2) (h2 : x2 > x3) (h3 : x3 > x4) (h4 : x4 > 0)
  (h : log (2014) / log (x1 / x2) + log (2014) / log (x2 / x3) + log (2014) / log (x3 / x4) ≥ log (2014) / log (x1 / x4) * k) :
  k ≤ 9 :=
sorry

end max_k_value_l12_12788


namespace monotonicity_and_tangent_intersections_l12_12354

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12354


namespace max_complete_dresses_l12_12661

namespace DressMaking

-- Define the initial quantities of fabric
def initial_silk : ℕ := 600
def initial_satin : ℕ := 400
def initial_chiffon : ℕ := 350

-- Define the quantities given to each of 8 friends
def silk_per_friend : ℕ := 15
def satin_per_friend : ℕ := 10
def chiffon_per_friend : ℕ := 5

-- Define the quantities required to make one dress
def silk_per_dress : ℕ := 5
def satin_per_dress : ℕ := 3
def chiffon_per_dress : ℕ := 2

-- Calculate the remaining quantities
def remaining_silk : ℕ := initial_silk - 8 * silk_per_friend
def remaining_satin : ℕ := initial_satin - 8 * satin_per_friend
def remaining_chiffon : ℕ := initial_chiffon - 8 * chiffon_per_friend

-- Calculate the maximum number of dresses that can be made
def max_dresses_silk : ℕ := remaining_silk / silk_per_dress
def max_dresses_satin : ℕ := remaining_satin / satin_per_dress
def max_dresses_chiffon : ℕ := remaining_chiffon / chiffon_per_dress

-- The main theorem indicating the number of complete dresses
theorem max_complete_dresses : max_dresses_silk = 96 ∧ max_dresses_silk ≤ max_dresses_satin ∧ max_dresses_silk ≤ max_dresses_chiffon := by
  sorry

end DressMaking

end max_complete_dresses_l12_12661


namespace lcm_of_6_8_10_l12_12064

theorem lcm_of_6_8_10 : Nat.lcm (Nat.lcm 6 8) 10 = 120 := 
  by sorry

end lcm_of_6_8_10_l12_12064


namespace max_distance_from_C_to_l_product_distances_M_to_A_and_B_eq_1_l12_12460

-- Definitions for the curve C and line l
def curve_C (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos α, sin α)

def line_l (x y : ℝ) : Prop :=
  x - y - 6 = 0

def distance_pt_to_line (c : ℝ × ℝ) : ℝ :=
  let (x, y) := c
  (abs (sqrt 3 * cos y - sin y - 6)) / sqrt 2

-- Problem 1: Maximum Distance from point on C to line l is 4√2.
theorem max_distance_from_C_to_l : 
  ∀ α : ℝ, distance_pt_to_line (curve_C α) ≤ 4 * sqrt 2 ∧ 
  (∃ α_max, distance_pt_to_line (curve_C α_max) = 4 * sqrt 2) :=
sorry

-- Definitions for line l₁ and points of intersection A and B
def line_l1 (t : ℝ) : ℝ × ℝ :=
  (-1 + sqrt 2 / 2 * t, sqrt 2 / 2 * t)

def curve_C_standard (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

-- Problem 2: Product of distances from M to points A and B is 1
theorem product_distances_M_to_A_and_B_eq_1 :
  ∃ t1 t2 : ℝ, curve_C_standard (line_l1 t1).1 (line_l1 t1).2 ∧ curve_C_standard (line_l1 t2).1 (line_l1 t2).2 ∧
  t1 * t2 = -1 ∧ 
  (abs (sqrt 2 / 2 * (t1 - t2))) = 1 :=
sorry

end max_distance_from_C_to_l_product_distances_M_to_A_and_B_eq_1_l12_12460


namespace compute_alpha_l12_12848

-- Declaring that we are using complex numbers
variable (α β : ℂ)

-- Conditions given in the problem
-- Condition 1 and 2: α, β are complex numbers (implicitly handled by type declaration)
-- Condition 3: α + β is a positive real number
def alpha_plus_beta_is_positive_real (α β : ℂ) : Prop := ∃ r : ℝ, r > 0 ∧ α + β = ↑r

-- Condition 4: i(α - 3β) is a positive real number
def i_times_alpha_minus_3beta_is_positive_real (α β : ℂ) : Prop := 
  ∃ r : ℝ, r > 0 ∧ i * (α - 3 * β) = ↑r

-- Condition 5: β = 4 + 3i
def beta_value : ℂ := 4 + 3i

-- The theorem to compute α
theorem compute_alpha (α : ℂ) (β : ℂ) (h1 : alpha_plus_beta_is_positive_real α β) 
  (h2 : i_times_alpha_minus_3beta_is_positive_real α β) 
  (h3 : β = beta_value) : α = 12 - 3i := 
sorry

end compute_alpha_l12_12848


namespace tetrahedral_marble_count_hypertetrahedral_marble_count_l12_12780

-- Problem statement for 3-dimensional case
theorem tetrahedral_marble_count (N : ℕ) : 
  ∑ k in finset.range (N + 1), (k * (k + 1) / 2) = ((N + 2) * (N + 1) * N) / 6 
:= sorry

-- Generalization for d-dimensional case
theorem hypertetrahedral_marble_count (N d : ℕ) : 
  ∑ k in finset.range (N + 1), (multichoose (k + d - 1) d) = multichoose (N + d - 1) d 
:= sorry

end tetrahedral_marble_count_hypertetrahedral_marble_count_l12_12780


namespace red_users_count_l12_12626

noncomputable def total_students : ℕ := 70
noncomputable def green_users : ℕ := 52
noncomputable def both_colors_users : ℕ := 38

theorem red_users_count : 
  ∀ (R : ℕ), total_students = green_users + R - both_colors_users → R = 56 :=
by
  sorry

end red_users_count_l12_12626


namespace monotonicity_of_f_tangent_intersection_l12_12294

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12294


namespace percent_within_one_std_dev_l12_12989

theorem percent_within_one_std_dev (m d : ℝ) (dist : ℝ → ℝ)
  (symm : ∀ x, dist (m + x) = dist (m - x))
  (less_than_upper_bound : ∀ x, (x < (m + d)) → dist x < 0.92) :
  ∃ p : ℝ, p = 0.84 :=
by
  sorry

end percent_within_one_std_dev_l12_12989


namespace log_decreasing_interval_l12_12562

theorem log_decreasing_interval :
  ∀ (y : ℝ → ℝ), (y = λ x, log 3 (x^2 - 2 * x)) → (∀ x, x ∈ Ioo (-∞ : ℝ) 0 → (x^2 - 2 * x) > 0 → ∀ (u₁ u₂ : ℝ), u₁ < u₂ → log 3 u₁ > log 3 u₂) -> 
  (∀ x₁ x₂, x₁ < x₂ → y x₁ > y x₂) :=
begin
  sorry
end

end log_decreasing_interval_l12_12562


namespace solve_equation_l12_12578

theorem solve_equation :
  {x : ℝ | (x + 1) * (x + 3) = x + 1} = {-1, -2} :=
sorry

end solve_equation_l12_12578


namespace find_f_neg1_l12_12554

namespace MathProblem

def f : ℝ → ℝ := sorry

lemma symmetry_x2 {x : ℝ} : f(x) = f(4 - x) := sorry
lemma symmetry_yaxis {x : ℝ} : f(-x) = f(x) := sorry
lemma f_at_3 : f(3) = 3 := sorry

theorem find_f_neg1 : f(-1) = 3 :=
by
  sorry

end MathProblem

end find_f_neg1_l12_12554


namespace product_of_real_roots_eq_one_l12_12717

theorem product_of_real_roots_eq_one :
  (∃ (x : ℝ), x ≠ 0 ∧ x ^ (Real.log x / Real.log 2) = 32) →
  (∏ x in set_of (λ x : ℝ, x ^ (Real.log x / Real.log 2) = 32), x) = 1 :=
by
  sorry

end product_of_real_roots_eq_one_l12_12717


namespace find_p_l12_12086

theorem find_p (m n p : ℚ) 
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + p = (n + 18) / 6 - 2 / 5) : 
  p = 3 := 
by 
  sorry

end find_p_l12_12086


namespace rounding_to_nearest_tenth_l12_12886

theorem rounding_to_nearest_tenth (x : ℝ) (h1 : x = 78.46582) (h2 : ∃ a b c d e : ℕ, x = a + b * 0.1 + c * 0.01 + d * 0.001 + e * 0.0001 ∧ b = 4 ∧ c = 6) : 
  (round (x * 10) / 10 = 78.5) :=
by
  sorry

end rounding_to_nearest_tenth_l12_12886


namespace analysis_of_statements_l12_12076

theorem analysis_of_statements :
  (∀ x : ℝ, x^2 - x - 2 ≠ 0) ∧
  (¬(∃ x : ℝ, x^2 + 4*x + m = 0) → m > 4) ∧
  ((∀ a b c : ℝ, ab^2 ≠ cb^2 ∨ b ≠ 0 → a ≠ c) ∧
  (∀ a : ℝ, a > 1 → 1/a < 1 ∧ (¬(a > 1) → (a < 0 → 1/a < 1)))) := sorry

end analysis_of_statements_l12_12076


namespace present_value_calc_l12_12047

def depreciation_rate : Real := 0.21
def selling_price_two_years : Real := 116615
def profit_two_years : Real := 24000
def depreciation_factor (years : Nat) : Real := Real.pow (1 - depreciation_rate) years

theorem present_value_calc (P : Real) (h : P * depreciation_factor 2 + profit_two_years = selling_price_two_years) :
  P ≈ 148349.77 := sorry

end present_value_calc_l12_12047


namespace lowest_price_in_2015_lowest_sales_revenue_in_2015_l12_12798

noncomputable def f (x : ℕ) : ℝ :=
  1 / 2 * (x^2 - 12 * x + 69)

noncomputable def g (x : ℕ) : ℝ :=
  x + 12

theorem lowest_price_in_2015 :
  ∃ x ∈ set.Icc (1 : ℕ) 12, f x = 16.5 := sorry

theorem lowest_sales_revenue_in_2015 :
  ∃ x ∈ set.Icc (1 : ℕ) 12, x = 5 ∧ f x * g x = 28.9 := sorry

end lowest_price_in_2015_lowest_sales_revenue_in_2015_l12_12798


namespace part1_part2_part3_l12_12101

noncomputable def functional_relationship (x : ℝ) : ℝ := -x + 26

theorem part1 (x y : ℝ) (hx6 : x = 6 ∧ y = 20) (hx8 : x = 8 ∧ y = 18) (hx10 : x = 10 ∧ y = 16) :
  ∀ (x : ℝ), functional_relationship x = -x + 26 := 
by
  sorry

theorem part2 (x : ℝ) (h_price_range : 6 ≤ x ∧ x ≤ 12) : 
  14 ≤ functional_relationship x ∧ functional_relationship x ≤ 20 :=
by
  sorry

noncomputable def gross_profit (x : ℝ) : ℝ := x * (functional_relationship x - 4)

theorem part3 (hx : 1 ≤ x) (hy : functional_relationship x ≤ 10):
  gross_profit (16 : ℝ) = 120 :=
by
  sorry

end part1_part2_part3_l12_12101


namespace area_of_equilateral_triangle_l12_12211

theorem area_of_equilateral_triangle (a : ℝ) (h : a > 0) : 
  let l := a / 3 in
  let S := (√3 / 4) * l^2 in
  S = (√3 * a^2) / 36 :=
by
  let l := a / 3
  let S := (√3 / 4) * l^2
  sorry

end area_of_equilateral_triangle_l12_12211


namespace limit_M_n_div_n_l12_12836

noncomputable def M_n (n : ℕ) : ℚ :=
  -- Placeholder function definition
  sorry

theorem limit_M_n_div_n :
  tendsto (λ n : ℕ, (M_n n : ℝ) / (n : ℝ)) at_top (𝓝 (11/18)) :=
sorry

end limit_M_n_div_n_l12_12836


namespace midpoints_and_P_coplanar_l12_12029

noncomputable theory

-- Definitions of points on the tetrahedron and the plane
variables (A B C D X Y Z T P : Point)
variables (α : Plane)

-- Facts about how plane α intersects the tetrahedron
axiom plane_intersects_edges :
  ∀ {A B C D X Y Z T : Point} (α : Plane),
    intersects_at_edge α A B X ∧
    intersects_at_edge α B C Y ∧
    intersects_at_edge α C D Z ∧
    intersects_at_edge α D A T

-- Definition of circle "ω" with diameter XZ
variable ω : Circle

-- Conditions that Y and T lie on circle ω
axiom points_on_circle : on_circle Y ω ∧ on_circle T ω

-- Point P on plane α with tangents PY and PT to circle ω
axiom tangents_from_P : tangent_to_circle P Y ω ∧ tangent_to_circle P T ω

-- Midpoints of the edges AB, BC, CD, DA
variables (M_AB M_BC M_CD M_DA : Point)

axiom midpoints_definition :
  midpoint A B M_AB ∧
  midpoint B C M_BC ∧
  midpoint C D M_CD ∧
  midpoint D A M_DA

-- Goal: All midpoints and point P lie on the same plane
theorem midpoints_and_P_coplanar :
  coplanar {M_AB, M_BC, M_CD, M_DA, P} :=
sorry

end midpoints_and_P_coplanar_l12_12029


namespace monotonicity_f_tangent_points_l12_12360

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12360


namespace monotonicity_of_f_tangent_intersection_l12_12292

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12292


namespace savannah_wrapped_gifts_with_second_roll_l12_12528

theorem savannah_wrapped_gifts_with_second_roll (total_gifts rolls_used roll_1_gifts roll_3_gifts roll_2_gifts : ℕ) 
  (h1 : total_gifts = 12) 
  (h2 : rolls_used = 3) 
  (h3 : roll_1_gifts = 3) 
  (h4 : roll_3_gifts = 4)
  (h5 : total_gifts - roll_1_gifts - roll_3_gifts = roll_2_gifts) :
  roll_2_gifts = 5 := 
by
  sorry

end savannah_wrapped_gifts_with_second_roll_l12_12528


namespace A_union_B_C_of_A_inter_B_l12_12773

section
variable {α : Type*} [LinearOrder α] {x : α}

def A : Set ℝ := {x : ℝ | abs (x - 1) ≤ 2}
def B : Set ℝ := {x : ℝ | x > 0 ∧ x < 1/2}

theorem A_union_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := 
by sorry

theorem C_of_A_inter_B : (set.univ \ (A ∩ B)) = {x : ℝ | x ≤ 0 ∨ x ≥ 1/2} := 
by sorry

end

end A_union_B_C_of_A_inter_B_l12_12773


namespace function_passing_through_origin_l12_12461

theorem function_passing_through_origin : (∃ (y : ℝ → ℝ), y = λ x, x^2 ∧ y 0 = 0) :=
by
  use (λ x, x^2)
  split
  · refl
  · simp

end function_passing_through_origin_l12_12461


namespace derivative_of_x_logx_l12_12174

noncomputable def function_derivative (x : ℝ) : ℝ := (λ x, x * Real.log x)' x

theorem derivative_of_x_logx (x : ℝ) (h : x > 0) : function_derivative x = Real.log x + 1 :=
by
  sorry

end derivative_of_x_logx_l12_12174


namespace monotonicity_of_f_tangent_intersection_l12_12286

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12286


namespace negation_collinearity_with_zero_vector_l12_12563

variable (V : Type) [AddCommGroup V] [Module ℝ V]

def is_collinear_with_zero (v : V) : Prop := 
  ∃ (k : ℝ), v = k • (0 : V)

theorem negation_collinearity_with_zero_vector :
  (∀ v : V, is_collinear_with_zero V v) ↔ (∃ v : V, ¬ is_collinear_with_zero V v) :=
sorry

end negation_collinearity_with_zero_vector_l12_12563


namespace abs_1_5_minus_sqrt2_eq_l12_12007

-- Conditions
def approx_sqrt2 : ℝ := 1.414

-- Statement of the problem
theorem abs_1_5_minus_sqrt2_eq (h : ∥real.sqrt 2 - approx_sqrt2∥ < ε) : 
  |(1.5 - real.sqrt 2)| = 1.5 - real.sqrt 2 :=
by
  sorry

end abs_1_5_minus_sqrt2_eq_l12_12007


namespace q_div_p_eq_225_l12_12169

-- Definitions
def num_cards : ℕ := 50
def num_values : ℕ := 10
def cards_per_value : ℕ := 5
def draw_cards : ℕ := 5

-- Conditions
def all_five_same (p : ℚ) : Prop :=
  p = 10 / ((num_cards.choose draw_cards) : ℚ)

def four_one_diff (q : ℚ) : Prop :=
  q = 2250 / ((num_cards.choose draw_cards) : ℚ)

-- Problem Statement
theorem q_div_p_eq_225 (p q : ℚ) 
  (hp : all_five_same p) 
  (hq : four_one_diff q) : 
  q / p = 225 :=
sorry

end q_div_p_eq_225_l12_12169


namespace range_of_m_l12_12021

theorem range_of_m (m : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ x - (m^2 - 2 * m + 4) * y + 6 > 0) →
  -1 < m ∧ m < 3 :=
by
  intros h
  rcases h with ⟨x, y, hx, hy, hineq⟩
  rw [hx, hy] at hineq
  sorry

end range_of_m_l12_12021


namespace valid_arrangements_l12_12452

def five_students := fin 5

def arrangements (students : fin 5) :=
  if ∃ (i j : fin 5), i ≠ j ∧ (i = 2 ∧ j = 4 ∨ i = 4 ∧ j = 2)
  then 72 else sorry

theorem valid_arrangements (students : fin 5) (h: ¬(∃ (i j : fin 5), i ≠ j ∧ (i = 2 ∧ j = 4 ∨ i = 4 ∧ j = 2))): 
  arrangements students = 72 :=
sorry

end valid_arrangements_l12_12452


namespace prob_teacherA_studentB_same_group_l12_12693

-- Definitions for our conditions
def num_teachers := 2
def num_students := 4
def groups := 2
def teachers_per_group := 1
def students_per_group := 2

-- Probability calculation (we just state it)
noncomputable def probability_same_group : ℚ :=
  (nat.choose 3 1 : ℚ) / (nat.choose 2 1 * nat.choose 4 2 / nat.factorial 2 : ℚ)

-- The main theorem to prove the probability is 1/2
theorem prob_teacherA_studentB_same_group : probability_same_group = 1 / 2 :=
by
  sorry

end prob_teacherA_studentB_same_group_l12_12693


namespace monotonicity_tangent_points_l12_12273

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12273


namespace harvest_unripe_oranges_l12_12050

theorem harvest_unripe_oranges (R T D U: ℕ) (h1: R = 28) (h2: T = 2080) (h3: D = 26)
  (h4: T = D * (R + U)) :
  U = 52 :=
by
  sorry

end harvest_unripe_oranges_l12_12050


namespace finite_solutions_l12_12982

theorem finite_solutions (a : ℕ) : set.finite { (x, y) : ℕ × ℕ | x! = y^2 + a^2 } :=
sorry

end finite_solutions_l12_12982


namespace exists_infinitely_many_a_not_in_union_intervals_l12_12736

theorem exists_infinitely_many_a_not_in_union_intervals (λ : ℝ) (hλ : 0 < λ) :
  ∃ infinitely_many (a : ℝ), 0 < a ∧ ∀ (m n : ℕ), 
    a < m^2 / n^2 ∨ a > (m^2 + λ) / n^2 :=
sorry

end exists_infinitely_many_a_not_in_union_intervals_l12_12736


namespace smallest_number_div_by_6_l12_12501

-- Define the digits set {1, 2, 3, 6, 9}
def digits := {1, 2, 3, 6, 9}

-- Function to verify a number contains exactly these digits
def contains_digits (n : ℕ) : Prop :=
  ∀ digit ∈ digits, ∃ i < 5, (n / 10^i) % 10 = digit

-- Function to check divisibility by 6
def divisible_by_6 (n : ℕ) : Prop :=
  (n % 2 = 0) ∧ ((1 + 2 + 3 + 6 + 9) % 3 = 0)

-- Function to compare digits sum
def smallest_permutation (n1 n2 : ℕ) : ℕ :=
  if n1 < n2 then n1 else n2

-- Lean statement
theorem smallest_number_div_by_6 : ∃ n, 
    contains_digits n ∧ divisible_by_6 n ∧ (∀ m, contains_digits m ∧ divisible_by_6 m → n ≤ m) :=
begin
  use 12396,
  split,
  { -- verify 12396 contains digits {1, 2, 3, 6, 9}
    sorry
  },
  split,
  { -- verify 12396 is divisible by 6
    sorry
  },
  { -- verify 12396 is the smallest such number
    intros m H,
    sorry
  }
end

end smallest_number_div_by_6_l12_12501


namespace game_is_unfair_l12_12624

-- Definitions
def color := {r : ℕ // r = 1 ∨ r = 2} -- 1 for red, 2 for yellow
def bag_content : list color := [⟨1, or.inl rfl⟩, ⟨1, or.inl rfl⟩, ⟨2, or.inr rfl⟩]
def draw (bag : list color) : color := bag.head -- simplified draw action for the purpose of this problem

-- Winning condition
def XiaoYing_wins (draw1 draw2 : color) : Prop := draw1 = draw2

-- Game unfairness condition
def game_unfair : Prop :=
  ¬ (∀ (draws1 draws2 : color), probability (XiaoYing_wins draws1 draws2) == probability (¬ XiaoYing_wins draws1 draws2))

-- Main theorem statement
theorem game_is_unfair : game_unfair := sorry

end game_is_unfair_l12_12624


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12335

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12335


namespace exists_m_value_l12_12749

-- Definitions for line equation, circle equation, and area of triangle
def line_eq (m : ℝ) (x y : ℝ) : Prop := x - m * y + 2 = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4
def area_of_triangle_eq (A B O : (ℝ × ℝ)) (area : ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  let (xO, yO) := O
  0.5 * abs (xA * (yB - yO) + xB * (yO - yA) + xO * (yA - yB)) = area

-- The points A, B, and the origin O for the area calculation
def A : (ℝ × ℝ) := (-2, 0)
def O : (ℝ × ℝ) := (0, 0)

theorem exists_m_value (m : ℝ) (B : ℝ × ℝ) (hB_line : line_eq m B.1 B.2) (hB_circle : circle_eq B.1 B.2) :
    area_of_triangle_eq A B O 2 :=
begin
    -- Proof to be provided
    sorry
end

end exists_m_value_l12_12749


namespace sum_of_solutions_l12_12069

theorem sum_of_solutions : 
  ∑ n in Finset.filter (λ n : ℤ, (|n| < |n - 4| ∧ |n - 4| < 10)) (Finset.range 20).map (λ x, ↑x - 6), n = -14 := 
by
  sorry

end sum_of_solutions_l12_12069


namespace number_of_people_l12_12136

theorem number_of_people (clinks : ℕ) (h : clinks = 45) : ∃ x : ℕ, x * (x - 1) / 2 = clinks ∧ x = 10 :=
by
  sorry

end number_of_people_l12_12136


namespace monotonicity_f_tangent_points_l12_12372

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12372


namespace complement_of_angle_l12_12796

theorem complement_of_angle (supplement : ℝ) (h_supp : supplement = 130) (original_angle : ℝ) (h_orig : original_angle = 180 - supplement) : 
  (90 - original_angle) = 40 := 
by 
  -- proof goes here
  sorry

end complement_of_angle_l12_12796


namespace cistern_emptying_time_l12_12636

theorem cistern_emptying_time (R L : ℝ) (hR : R = 1 / 6) (hL : L = 1 / 6 - 1 / 8) :
    1 / L = 24 := by
  -- The proof is omitted
  sorry

end cistern_emptying_time_l12_12636


namespace no_integer_solutions_l12_12158

theorem no_integer_solutions :
  ∀ x y : ℤ, x^3 + 4 * x^2 - 11 * x + 30 ≠ 8 * y^3 + 24 * y^2 + 18 * y + 7 :=
by sorry

end no_integer_solutions_l12_12158


namespace car_distance_travelled_l12_12988

theorem car_distance_travelled (time_hours : ℝ) (time_minutes : ℝ) (time_seconds : ℝ)
    (actual_speed : ℝ) (reduced_speed : ℝ) (distance : ℝ) :
    time_hours = 1 → 
    time_minutes = 40 →
    time_seconds = 48 →
    actual_speed = 34.99999999999999 → 
    reduced_speed = (5 / 7) * actual_speed → 
    distance = reduced_speed * ((time_hours + time_minutes / 60 + time_seconds / 3600) : ℝ) →
    distance = 42 := sorry

end car_distance_travelled_l12_12988


namespace teacher_arrangement_l12_12980

theorem teacher_arrangement (teachers : Finset ℕ) (h_card : teachers.card = 6):
  ∃(A B C : Finset ℕ), A.card = 1 ∧ B.card = 2 ∧ C.card = 3 ∧ A ∪ B ∪ C = teachers ∧ disjoint A B ∧ disjoint B C ∧ disjoint C A ∧ (6.choose 1) * (5.choose 2) * (3.choose 3) = 60 :=
by
  sorry

end teacher_arrangement_l12_12980


namespace identify_correct_statement_l12_12609

-- Definitions based on conditions
def population (athletes : ℕ) : Prop := athletes = 1000
def is_individual (athlete : ℕ) : Prop := athlete ≤ 1000
def is_sample (sampled_athletes : ℕ) (sample_size : ℕ) : Prop := sampled_athletes = 100 ∧ sample_size = 100

-- Theorem statement based on the conclusion
theorem identify_correct_statement (athletes : ℕ) (sampled_athletes : ℕ) (sample_size : ℕ)
    (h1 : population athletes) (h2 : ∀ a, is_individual a) (h3 : is_sample sampled_athletes sample_size) : 
    (sampled_athletes = 100) ∧ (sample_size = 100) :=
by
  sorry

end identify_correct_statement_l12_12609


namespace value_of_polynomial_l12_12605

theorem value_of_polynomial :
  98^3 + 3 * (98^2) + 3 * 98 + 1 = 970299 :=
by sorry

end value_of_polynomial_l12_12605


namespace interest_rate_per_annum_l12_12546

-- Definitions of conditions
def principal : ℝ := 625
def time_period : ℕ := 2
def interest_difference : ℝ := 1
def rate (r : ℝ) : Prop := principal * ((1 + r)^time_period - 1) - principal * r * time_period = interest_difference

-- Theorem statement
theorem interest_rate_per_annum : ∃ r : ℝ, rate r ∧ r = 0.04 := 
begin
  sorry
end

end interest_rate_per_annum_l12_12546


namespace solve_math_problem_l12_12225

-- Math problem definition
def math_problem (A : ℝ) : Prop :=
  (0 < A ∧ A < (Real.pi / 2)) ∧ (Real.cos A = 3 / 5) →
  Real.sin (2 * A) = 24 / 25

-- Example theorem statement in Lean
theorem solve_math_problem (A : ℝ) : math_problem A :=
sorry

end solve_math_problem_l12_12225


namespace monotonicity_of_f_tangent_intersection_l12_12291

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12291


namespace proof_problem_l12_12987

-- Define the probability of green light and red light
def p_green : ℚ := 3 / 4
def p_red : ℚ := 1 / 4

-- Define the probability distribution
def prob_distribution (k : ℕ) : ℚ :=
  if k = 0 then p_red
  else if k = 1 then p_red * p_green
  else if k = 2 then p_green * p_green * p_red
  else if k = 3 then p_green * p_green * p_green * p_red
  else if k = 4 then p_green ^ 4
  else 0

-- Define the expected value
noncomputable def E_xi : ℚ := 
  0 * prob_distribution 0 +
  1 * prob_distribution 1 +
  2 * prob_distribution 2 +
  3 * prob_distribution 3 +
  4 * prob_distribution 4

-- Define the probability that the car stops at most after 3 intersections
noncomputable def P_xi_le_3 : ℚ :=
  prob_distribution 0 +
  prob_distribution 1 +
  prob_distribution 2 +
  prob_distribution 3

-- Putting it all together into Lean theorem statement
theorem proof_problem : 
  E_xi = 525 / 256 ∧
  P_xi_le_3 = 175 / 256 :=
by 
  sorry

end proof_problem_l12_12987


namespace impossible_path_2018_grid_l12_12866

theorem impossible_path_2018_grid :
  ¬((∃ (path : Finset (Fin 2018 × Fin 2018)), 
    (0, 0) ∈ path ∧ (2017, 2017) ∈ path ∧ 
    (∀ {x y}, (x, y) ∈ path → (x + 1, y) ∈ path ∨ (x, y + 1) ∈ path ∨ (x - 1, y) ∈ path ∨ (x, y - 1) ∈ path) ∧ 
    (∀ {x y}, (x, y) ∈ path → (Finset.card path = 2018 * 2018)))) :=
by 
  sorry

end impossible_path_2018_grid_l12_12866


namespace equal_chords_subtended_equal_arcs_l12_12077

theorem equal_chords_subtended_equal_arcs {r : ℝ} {c : Circle r} {ch1 ch2 : Chord c} (h : ch1.length = ch2.length) : ch1.arc = ch2.arc :=
  sorry

end equal_chords_subtended_equal_arcs_l12_12077


namespace largest_prime_factor_5040_is_7_l12_12956

-- Definition of the condition: the prime factorization of 5040
def prime_factorization_5040 : list ℕ := [2, 2, 2, 2, 3, 3, 5, 7]

-- Predicate to check if a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m:ℕ, m ∣ n → m = 1 ∨ m = n

-- Predicate to check if a list contains only primes
def all_primes (l: list ℕ) : Prop :=
  ∀ x, x ∈ l → is_prime x

-- Statement of the problem
theorem largest_prime_factor_5040_is_7 :
  all_primes prime_factorization_5040 ∧ 
  list.prod prime_factorization_5040 = 5040 ∧
  list.maximum prime_factorization_5040 = 7 :=
sorry

end largest_prime_factor_5040_is_7_l12_12956


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12334

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12334


namespace monotonicity_of_f_tangent_intersection_l12_12282

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12282


namespace kim_monthly_revenue_l12_12831

-- Define the cost to open the store
def initial_cost : ℤ := 25000

-- Define the monthly expenses
def monthly_expenses : ℤ := 1500

-- Define the number of months
def months : ℕ := 10

-- Define the revenue per month
def revenue_per_month (total_revenue : ℤ) (months : ℕ) : ℤ := total_revenue / months

theorem kim_monthly_revenue :
  ∃ r, revenue_per_month r months = 4000 :=
by 
  let total_expenses := monthly_expenses * months
  let total_revenue := initial_cost + total_expenses
  use total_revenue
  unfold revenue_per_month
  sorry

end kim_monthly_revenue_l12_12831


namespace smallest_K_inequality_l12_12159

theorem smallest_K_inequality :
  ∀ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 →
  (∃ K : ℝ, K = (Real.sqrt 6) / 3 ∧
    (K + (a + b + c) / 3 ≥ (K + 1) * Real.sqrt ((a^2 + b^2 + c^2) / 3))) :=
by
  intros a b c
  intros h_a_bounds h_b_bounds h_c_bounds
  use (Real.sqrt 6) / 3
  split
  · rfl
  · sorry

end smallest_K_inequality_l12_12159


namespace find_n_is_largest_solution_l12_12493

theorem find_n_is_largest_solution :
  ∃ n p q r : ℝ, n = p + sqrt (q + sqrt r) ∧ 
                 (4 / (n - 4) + 6 / (n - 6) + 18 / (n - 18) + 20 / (n - 20) = n^2 - 13 * n - 6) ∧ 
                 p = 13 ∧ q = 53 ∧ r = 249 ∧ p + q + r = 315 :=
by
  use 13 + sqrt (53 + sqrt 249)
  use 13, 53, 249
  split
  rfl
  split
  sorry
  split
  rfl
  split
  rfl
  split
  rfl
  norm_num


end find_n_is_largest_solution_l12_12493


namespace original_number_l12_12146

theorem original_number (x : ℕ) : x % 5 = 4 :=
by {
  -- Conditions: Adding 3 to x and taking modulo 5 should result in 2
  have h1 : (4 + 3) % 5 = 2 := by norm_num,
  -- Given 4 is the smallest x that must be true
  have h2 : (x + 5 = 9 ∧ (x + 3) % 5 = 2) → x = 4 := by {
    intro h,
    cases h with h_left h_right,
    rw h_left,
    exact nat.mod_eq_of_lt (add_lt_add_left (by norm_num) _)
  },
  -- Conclusion: if x worked with +3%5=2 before, then x+5=9 confirms 4 is the number
  exact h2.2 h1
}

end original_number_l12_12146


namespace monotonicity_tangent_points_l12_12281

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12281


namespace problem1_problem2_l12_12093

open Nat

-- Definitions used based on conditions
def A (n m : ℕ) : ℕ := factorial n / factorial (n - m)
def C (n r : ℕ) : ℕ := n.choose r

-- Problem 1: Prove that 2 * A 8 5 + 7 * A 8 4 / A 8 8 - A 9 5 = 1 / 15
theorem problem1 : (2 * A 8 5 + 7 * A 8 4) / (A 8 8 - A 9 5) = 1 / 15 := sorry

-- Problem 2: Prove that C 200 198 + C 200 196 + 2 * C 200 197 = 67331650
theorem problem2 : C 200 198 + C 200 196 + 2 * C 200 197 = 67331650 := sorry

end problem1_problem2_l12_12093


namespace area_of_enclosed_region_is_zero_l12_12950

theorem area_of_enclosed_region_is_zero :
  ∀ (x y : ℝ), (x^2 - 10 * x + 5 * y + 50 = 25 + 15 * y - y^2) → (y > x - 4) → 0 := 
sorry

end area_of_enclosed_region_is_zero_l12_12950


namespace decimal_to_binary_l12_12951

theorem decimal_to_binary (n : ℕ) (h : n = 101) : nat.binary_repr n = "1100101" :=
by
  -- Here, we would actually prove the statement, but it's skipped for the purpose of this example.
  sorry

end decimal_to_binary_l12_12951


namespace continuity_at_origin_discontinuity_at_points_l12_12824

def f (x y z : ℝ) : ℝ :=
if hyz : y^2 + z^2 ≠ 0 then x^4 + (2 * x * y * z) / (y^2 + z^2) else x^4

theorem continuity_at_origin : ContinuousAt (λ P : ℝ × ℝ × ℝ, f P.1 P.2.1 P.2.2) (0, 0, 0) :=
sorry

theorem discontinuity_at_points (x : ℝ) (hx : x ≠ 0) : 
  ¬ ContinuousAt (λ P : ℝ × ℝ × ℝ, f P.1 P.2.1 P.2.2) (x, 0, 0) :=
sorry

end continuity_at_origin_discontinuity_at_points_l12_12824


namespace proof_f_f_minus_two_l12_12760

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.sqrt x
  else if 0 < x ∧ x < 1 then 1 / x
  else 2^x

theorem proof_f_f_minus_two : f (f (-2)) = 4 := by
  sorry

end proof_f_f_minus_two_l12_12760


namespace twentieth_number_in_twentieth_row_l12_12865

-- Assume the sequence ends with n^2 at the nth row
def last_number_in_nth_row (n : ℕ) : ℕ := n * n

-- Define a function to find the m-th number in the nth row
-- Given the problem specifics, the first number in nth row is (n-1)^2 + 1
def number_in_nth_row (n m : ℕ) : ℕ :=
  last_number_in_nth_row (n - 1) + m

-- Theorem stating the 20th number in the 20th row
theorem twentieth_number_in_twentieth_row : number_in_nth_row 20 20 = 381 := 
by
  -- Translate the specific calculations from the problem steps
  have last_in_19th := last_number_in_nth_row 19
  rw [last_number_in_nth_row, last_in_19th]
  have nth_20th := number_in_nth_row 20 20
  rw [number_in_nth_row, last_in_19th]
  sorry -- Proof is omitted for now

end twentieth_number_in_twentieth_row_l12_12865


namespace monotonicity_intervals_and_k_range_l12_12399

noncomputable def f (x k : ℝ) : ℝ := log (x - 1) - k * (x - 1) + 1

noncomputable def g (x k : ℝ) : ℝ :=
  1 / 3 * x ^ 3 - log (x + 1) + f (x + 2) k

theorem monotonicity_intervals_and_k_range (k : ℝ) : 
  (∀ x1 x2 ∈ Icc 0 1, abs (g x1 k - g x2 k) ≤ 1) →
  ((k ≤ 0 → ∀ x > 1, f' x k > 0) ∧ 
   (k > 0 → ∀ x > 1, (1 < x ∧ x < (k+1)/k) → f' x k < 0 ∧ (x > (k+1)/k) → f' x k > 0)) ∧
  (-2/3 ≤ k ∧ k ≤ 4/3) := sorry


end monotonicity_intervals_and_k_range_l12_12399


namespace susie_charges_per_slice_l12_12538

theorem susie_charges_per_slice
  (price_whole_pizza : ℕ)
  (slices_sold : ℕ)
  (whole_pizzas_sold : ℕ)
  (total_revenue : ℕ)
  (slices_per_pizza : ℕ)
  (price_per_slice : ℕ)
  (h_price_whole_pizza : price_whole_pizza = 15)
  (h_slices_sold : slices_sold = 24)
  (h_whole_pizzas_sold : whole_pizzas_sold = 3)
  (h_total_revenue : total_revenue = 117)
  (h_slices_per_pizza : slices_per_pizza = 8)
  (h_revenue_eqn : slices_sold * price_per_slice + whole_pizzas_sold * price_whole_pizza = total_revenue) :
  price_per_slice = 3 :=
by {
  rw [h_price_whole_pizza, h_slices_sold, h_whole_pizzas_sold, h_total_revenue] at h_revenue_eqn,
  have h : 24 * price_per_slice + 3 * 15 = 117 := h_revenue_eqn,
  have h1 : 24 * price_per_slice + 45 = 117 := h,
  have h2 : 24 * price_per_slice = 72,
  calc 
    price_per_slice = 72 / 24 := by sorry
}

end susie_charges_per_slice_l12_12538


namespace exists_two_points_C_l12_12210

noncomputable def circle_through_two_points_tangent_to_plane (plane S : Type) (point A B : Type) (tangent_ratio : Rat) : Prop :=
  ∃ C : Type, (is_tangent_to_plane S C) ∧ (circle_passing_through_points A B C) ∧ (AC_ratio BC = tangent_ratio)

theorem exists_two_points_C (S : Type) (A B : Type) (AC_ratio BC : Rat) : 
  circle_through_two_points_tangent_to_plane S A B (2 : 1) →
  ∃ C1 C2 : Type, C1 ≠ C2 ∧ 
  circle_through_two_points_tangent_to_plane S A B (2 : 1) :=
sorry

end exists_two_points_C_l12_12210


namespace max_blocks_in_box_l12_12600

def volume (l w h : ℕ) : ℕ := l * w * h

-- Define the dimensions of the box and the block
def box_length := 4
def box_width := 3
def box_height := 2
def block_length := 3
def block_width := 1
def block_height := 1

-- Define the volumes of the box and the block using the dimensions
def V_box : ℕ := volume box_length box_width box_height
def V_block : ℕ := volume block_length block_width block_height

theorem max_blocks_in_box : V_box / V_block = 8 :=
  sorry

end max_blocks_in_box_l12_12600


namespace learning_machine_price_reduction_l12_12575

theorem learning_machine_price_reduction (x : ℝ) (h1 : 2000 * (1 - x) * (1 - x) = 1280) : 2000 * (1 - x)^2 = 1280 :=
by
  sorry

end learning_machine_price_reduction_l12_12575


namespace problem_1_l12_12622

variables (a b c : ℝ^3)
variables (norm_a : ∥a∥ = 2) (norm_b : ∥b∥ = 2) (norm_c : ∥c∥ = 1)
variables (non_collinear : ¬ collinear ℝ [a, b, c])
variables (equal_angles : ∀ (u v : ℝ^3), u ≠ v → u ≠ c → (∠ u v = 2 * π / 3))

theorem problem_1 : ∥a + b + c∥ = 1 := sorry

end problem_1_l12_12622


namespace monotonicity_and_tangent_intersection_l12_12248

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12248


namespace monotonicity_of_f_tangent_intersection_points_l12_12331

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12331


namespace parallelepiped_volume_l12_12421

variables (a b : ℝ^3)

def is_unit_vector (v : ℝ^3) : Prop :=
  ∥v∥ = 1

def angle_between (u v : ℝ^3) (θ : ℝ) : Prop :=
  real.angle_vector u v = θ

theorem parallelepiped_volume
  (ha : is_unit_vector a)
  (hb : is_unit_vector b)
  (angle_ab : angle_between a b (π / 4)) :
  volume_of_parallelepiped_generated_by a (a + a × b) b = 1 / 2 :=
begin
  sorry  -- proof to be completed
end

end parallelepiped_volume_l12_12421


namespace angle_ACB_l12_12818

-- Definitions based on conditions
def angle_A := 110
def angle_ABE := 145
def supplementary (x y : ℝ) := x + y = 180
def triangle_sum (x y z : ℝ) := x + y + z = 180

-- The theorem to be proven
theorem angle_ACB :
  let angle_ABC := 180 - angle_ABE in
  triangle_sum angle_A angle_ABC 35 →
  supplementary angle_ABC angle_ABE →
  angle_ACB = 35 :=
by
  intro h1 h2
  sorry

end angle_ACB_l12_12818


namespace train_speeds_proof_l12_12586

-- Defining the initial conditions
variables (v_g v_p v_e : ℝ)
variables (t_g t_p t_e : ℝ) -- t_g, t_p, t_e are the times for goods, passenger, and express trains respectively

-- Conditions given in the problem
def goods_train_speed := v_g 
def passenger_train_speed := 90 
def express_train_speed := 1.5 * 90

-- Passenger train catches up with the goods train after 4 hours
def passenger_goods_catchup := 90 * 4 = v_g * (t_g + 4) - v_g * t_g

-- Express train catches up with the passenger train after 3 hours
def express_passenger_catchup := 1.5 * 90 * 3 = 90 * (3 + 4)

-- Theorem to prove the speeds of each train
theorem train_speeds_proof (h1 : 90 * 4 = v_g * (t_g + 4) - v_g * t_g)
                           (h2 : 1.5 * 90 * 3 = 90 * (3 + 4)) :
    v_g = 90 ∧ v_p = 90 ∧ v_e = 135 :=
by {
  sorry
}

end train_speeds_proof_l12_12586


namespace total_investment_is_correct_l12_12948

def Raghu_investment : ℕ := 2300
def Trishul_investment (Raghu_investment : ℕ) : ℕ := Raghu_investment - (Raghu_investment / 10)
def Vishal_investment (Trishul_investment : ℕ) : ℕ := Trishul_investment + (Trishul_investment / 10)

theorem total_investment_is_correct :
    let Raghu_inv := Raghu_investment;
    let Trishul_inv := Trishul_investment Raghu_inv;
    let Vishal_inv := Vishal_investment Trishul_inv;
    Raghu_inv + Trishul_inv + Vishal_inv = 6647 :=
by
    sorry

end total_investment_is_correct_l12_12948


namespace value_of_y_l12_12617

theorem value_of_y (x y : ℕ) (h1 : y = 0.125 * x) (h2 : y * x = 10000) : y = 35 :=
by
  sorry

end value_of_y_l12_12617


namespace eccentricity_of_hyperbola_l12_12742

noncomputable def hyperbola_eccentricity (a b c : ℝ) : ℝ :=
  c / a

def foci_coordinates (a b : ℝ) : ℝ × ℝ :=
  (real.sqrt (a^2 + b^2), real.sqrt (a^2 + b^2))

def is_equilateral_triangle (a b c : ℝ) : Prop :=
  let F1 := (-c, 0)
  let F2 := (c, 0)
  let A := (-c, b^2 / a)
  let B := (-c, -b^2 / a)
  let AB := real.sqrt (4 * (b^2 / a)^2)
  let F1F2 := 2 * c
  let F2A := real.sqrt ((-c - c)^2 + (0 - b^2 / a)^2)
  let F2B := real.sqrt ((-c - c)^2 + (0 + b^2 / a)^2)
  (AB = F1F2 ∧ AB = F2A ∧ AB = F2B)

theorem eccentricity_of_hyperbola (a b c : ℝ) (h_cond : a^2 + b^2 = c^2)
  (h_equilateral : is_equilateral_triangle a b c) :
  hyperbola_eccentricity a b c = real.sqrt 3 := by
  sorry

end eccentricity_of_hyperbola_l12_12742


namespace trajectory_of_P_and_max_triangle_area_l12_12814

theorem trajectory_of_P_and_max_triangle_area
    (P : ℝ × ℝ) (M : ℝ × ℝ) (d : ℝ) (A B D : ℝ × ℝ)
    (C_eq_trajectory : ∀ P, (P.1 - 1)^2 + P.2^2 = d^2 → d = |P.1 - 2| → (PM : ℝ) → PM = Real.sqrt ((P.1 - 1)^2 + P.2^2) → 
      PM / d = Real.sqrt (2) / 2 → P.1 ^ 2 / 2 + P.2 ^ 2 = 1)
    (max_area_eq : ∀ l : ℝ → ℝ, (l ≠ λ x, 0) → ∀ l_int_C_at_A_B : l(A.1) = A.2 ∧ l(B.1) = B.2, 
      A.1 ^ 2 / 2 + A.2 ^ 2 = 1 ∧ B.1 ^ 2 / 2 + B.2 ^ 2 = 1 →
      ∀ midpoint_D : D = ((A.1 + B.1) / 2 , (A.2 + B.2) / 2), 
        let intersect_OD_x_eq_2 := (D.2 = 2 / 2 * D.1) in 
        let delta_area := S(Δ OAB) in 
        S(Δ OAB) = (sqrt (2) / 3) * sqrt (m^2 * (3 - m^2)) and
        m ≠ 0 and (m = sqrt(3)/2 ∨ m = -sqrt(3)/2)) :
    (∃ P, ∃ l : ℝ → ℝ, (P.1 ^ 2 / 2 + P.2 ^ 2 = 1) ∧ 
        (max_area_eq (y = -x + m)) ∧ 
        m = sqrt(3)/2 ∨ m = -sqrt(3)/2) := Sorry

end trajectory_of_P_and_max_triangle_area_l12_12814


namespace solve_for_x_minus_y_l12_12430

theorem solve_for_x_minus_y (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 24) : x - y = 4 := 
by
  sorry

end solve_for_x_minus_y_l12_12430


namespace polar_line_eq_l12_12820

theorem polar_line_eq (ρ θ : ℝ) : (ρ * Real.cos θ = 1) ↔ (ρ = Real.cos θ ∨ ρ = Real.sin θ ∨ 1 / Real.cos θ = ρ) := by
  sorry

end polar_line_eq_l12_12820


namespace order_of_magnitude_l12_12566

noncomputable def log_0_7_2 := Real.logBase 0.7 2
noncomputable def log_0_7_0_8 := Real.logBase 0.7 0.8
noncomputable def exp_neg_2 := 0.9 ^ (-2)

theorem order_of_magnitude :
  log_0_7_2 < log_0_7_0_8 ∧ log_0_7_0_8 < exp_neg_2 :=
by
  sorry

end order_of_magnitude_l12_12566


namespace total_length_of_river_is_80_l12_12023

-- Definitions based on problem conditions
def straight_part_length := 20
def crooked_part_length := 3 * straight_part_length
def total_length_of_river := straight_part_length + crooked_part_length

-- Theorem stating that the total length of the river is 80 miles
theorem total_length_of_river_is_80 :
  total_length_of_river = 80 := by
    -- The proof is omitted
    sorry

end total_length_of_river_is_80_l12_12023


namespace train_passing_time_correct_l12_12473

noncomputable def train_passing_time (L1 L2 : ℕ) (S1 S2 : ℕ) : ℝ :=
  let S1_mps := S1 * (1000 / 3600)
  let S2_mps := S2 * (1000 / 3600)
  let relative_speed := S1_mps + S2_mps
  let total_length := L1 + L2
  total_length / relative_speed

theorem train_passing_time_correct :
  train_passing_time 105 140 45 36 = 10.89 := by
  sorry

end train_passing_time_correct_l12_12473


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12341

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12341


namespace free_throws_count_l12_12567

-- Definitions based on the conditions
variables (a b x : ℕ) -- Number of 2-point shots, 3-point shots, and free throws respectively.

-- Condition: Points from two-point shots equal the points from three-point shots
def points_eq : Prop := 2 * a = 3 * b

-- Condition: Number of free throws is twice the number of two-point shots
def free_throws_eq : Prop := x = 2 * a

-- Condition: Total score is adjusted to 78 points
def total_score : Prop := 2 * a + 3 * b + x = 78

-- Proof problem statement
theorem free_throws_count (h1 : points_eq a b) (h2 : free_throws_eq a x) (h3 : total_score a b x) : x = 26 :=
sorry

end free_throws_count_l12_12567


namespace square_area_divided_into_triangles_eq_area_l12_12667

theorem square_area_divided_into_triangles_eq_area (s DG : ℝ) (h1 : 8 * (1/2) * (s * s / 8) = s * s) (h2 : DG = s / √2) (h3 : DG = 5) :
  s^2 = 50 :=
by {
  sorry
}

end square_area_divided_into_triangles_eq_area_l12_12667


namespace fixed_errors_correct_l12_12194

-- Conditions
def total_lines_of_code : ℕ := 4300
def lines_per_debug : ℕ := 100
def errors_per_debug : ℕ := 3

-- Question: How many errors has she fixed so far?
theorem fixed_errors_correct :
  (total_lines_of_code / lines_per_debug) * errors_per_debug = 129 := 
by 
  sorry

end fixed_errors_correct_l12_12194


namespace monotonicity_and_tangent_intersection_l12_12253

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12253


namespace express_in_scientific_notation_l12_12464

theorem express_in_scientific_notation (n : ℤ) : 
  let billion := 1000000000 in
  1.36 * billion = 1.36 * 10^n → n = 9 :=
by sorry

end express_in_scientific_notation_l12_12464


namespace simplify_and_evaluate_l12_12892

theorem simplify_and_evaluate 
  (a b : ℤ)
  (h1 : a = 2)
  (h2 : b = -1) : 
  (2 * a^2 * b - 4 * a * b^2) - 2 * (a * b^2 + a^2 * b) = -12 := 
by
  rw [h1, h2]
  sorry

end simplify_and_evaluate_l12_12892


namespace probability_bulls_win_series_l12_12539

theorem probability_bulls_win_series :
  let P_Bulls := 1 / 4 in
  let P_Heat := 3 / 4 in
  let scenarios := 6 in  -- number of ways to have 2 wins for each team out of first 4 games
  let prob_2wins_bulls := scenarios * (P_Bulls ^ 2) * (P_Heat ^ 2) in
  let prob_bulls_win_game_5 := P_Bulls in
  prob_2wins_bulls * prob_bulls_win_game_5 = 27 / 512 := by
  -- proof goes here
  sorry

end probability_bulls_win_series_l12_12539


namespace monotonicity_tangent_intersection_points_l12_12374

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12374


namespace weight_of_new_person_l12_12618

-- Define the conditions
variable (n : ℕ) (initial_avg : ℕ) (increase : ℕ) (old_weight : ℕ) (new_weight : ℕ)
variable (h_n : n = 12) (h_initial_avg : initial_avg = 4) (h_increase : increase = 48) 
variable (h_old_weight : old_weight = 58) (h_new_weight_relation : old_weight + increase = new_weight)

-- Proof statement
theorem weight_of_new_person : new_weight = 106 := by
  -- Apply the given conditions
  rw [h_old_weight, h_increase] at h_new_weight_relation
  exact h_new_weight_relation

-- Include a sorry just in case
#check weight_of_new_person

end weight_of_new_person_l12_12618


namespace trisect_angle_l12_12522

theorem trisect_angle (xO P T S : ℝ × ℝ) (k : ℝ) (h1 : k > 0)
  (hxO : xO = (0, 0))
  (h_eq : ∀ (P : ℝ × ℝ), P.1 * P.2 = k)
  (hP : 2 * real.sqrt (P.1^2 + P.2^2) = real.sqrt (S.1^2 + S.2^2))
  (hT : T.2 = P.2)
  (hT_parallel : T.1 > P.1) :
  ∠ TPS = (∠ POx) / 3 := sorry

end trisect_angle_l12_12522


namespace problem_proof_l12_12134

noncomputable def problem_statement (O₁ O₂ : Point) (A B C D K : Point) : Prop :=
  tangent O₁ O₂ ∧
  tangent_ext O₁ A O₂ C ∧
  tangent_int O₁ B O₂ D ∧
  intersect AB CD K →
  lies_on_line K O₁ O₂

axiom tangent (O₁ O₂ : Point) : Prop

axiom tangent_ext (O₁ A O₂ C : Point) : Prop

axiom tangent_int (O₁ B O₂ D : Point) : Prop

axiom intersect (AB CD : Line) (K : Point) : Prop

axiom lies_on_line (K O₁ O₂ : Point) : Prop

theorem problem_proof (O₁ O₂ A B C D K : Point) : problem_statement O₁ O₂ A B C D K := by
  sorry

end problem_proof_l12_12134


namespace sum_of_extrema_and_inflection_points_coords_is_zero_l12_12916

-- Definitions for the problem
def P (x : ℝ) : ℝ := 3 * x^5 - 250 * x^3 + 735 * x

-- Lean theorem statement
theorem sum_of_extrema_and_inflection_points_coords_is_zero :
  let critical_points := {x | P.deriv x = 0}.finset
  let inflection_points := {x | P.deriv.deriv x = 0}.finset
  let all_points := critical_points ∪ inflection_points
  ∑ x in all_points, x = 0 :=
by
  sorry

end sum_of_extrema_and_inflection_points_coords_is_zero_l12_12916


namespace part1_monotonicity_part2_tangent_intersection_l12_12267

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12267


namespace area_of_FDBG_l12_12940

theorem area_of_FDBG {AB AC AD AE : ℝ} (hAB : AB = 60) (hAC : AC = 30) (hTriangleArea : 0.5 * AB * AC = 180)
                    (hAD : AD = 20) (hAE : AE = 10) :
  let D : (p : ℝ) × ℝ := (20, 0)
  let E : (p : ℝ) × ℝ := (10, 0)
  let Area_ABC := 180
  let Area_ADE := (20 * 10) / (60 * 30) * 180
  let Area_AFE := (1/7) * Area_ADE
  let Area_AGC := (30 * (AC / 3)) / (60 * AC) * Area_ABC
  in (Area_ABC - (Area_ADE + Area_AGC - Area_AFE)) = 130 + 6 / 7 :=
by 
  sorry

end area_of_FDBG_l12_12940


namespace intersection_parabola_line_l12_12028

-- Defining the conditions
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def line (a x y : ℝ) : Prop := 2 * x + y + a = 0
def point_A := (1 : ℝ, 2 : ℝ)
def parabola_focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)  -- assuming the focus of the parabola y^2 = 2px

-- Theorem statement to be proven
theorem intersection_parabola_line (a p : ℝ) (F A B : ℝ × ℝ)
  (h₁ : point_A = (1, 2))
  (h₂ : parabola p A.1 A.2)
  (h₃ : line a A.1 A.2)
  (h₄ : ∀ x y, (parabola p x y ∧ line a x y) → (x, y) = A ∨ (x, y) = B)
  (h₅ : F = parabola_focus p) :
  abs (F.1 - A.1) + abs (F.2 - A.2) + abs (F.1 - B.1) + abs (F.2 - B.2) = 7 := 
sorry

end intersection_parabola_line_l12_12028


namespace algebraic_expression_value_l12_12425

theorem algebraic_expression_value : 
  ∀ (a b : ℝ), (∃ x, x = -2 ∧ a * x - b = 1) → 4 * a + 2 * b + 7 = 5 :=
by
  intros a b h
  cases' h with x hx
  cases' hx with hx1 hx2
  rw [hx1] at hx2
  sorry

end algebraic_expression_value_l12_12425


namespace find_x_l12_12762

def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x ≤ 0 then sin (π * x / 2)
  else if 0 < x ∧ x < 1 then log 2 (x + 1)
  else 0  -- default value for undefined regions

theorem find_x (x : ℝ) (h : f x = -1/2) : x = -1/3 :=
by sorry

end find_x_l12_12762


namespace kerosene_cost_l12_12443

-- Define the cost of a dozen eggs in terms of the cost of a pound of rice
def cost_of_dozen_eggs (cost_rice : ℝ) : ℝ := cost_rice

-- Define the cost of an egg
def cost_of_one_egg (cost_dozen_eggs : ℝ) : ℝ := cost_dozen_eggs / 12

-- Define the cost of 8 eggs
def cost_of_eight_eggs (cost_one_egg : ℝ) : ℝ := 8 * cost_one_egg

-- Define the cost of a half-liter of kerosene in terms of the cost of 8 eggs
def cost_of_half_liter_kerosene (cost_eight_eggs : ℝ) : ℝ := cost_eight_eggs

-- Define the cost of a liter of kerosene
def cost_of_liter_kerosene (cost_half_liter_kerosene : ℝ) : ℝ := 2 * cost_half_liter_kerosene

-- Define the conversion from dollars to cents
def dollars_to_cents (dollars : ℝ) : ℝ := dollars * 100

-- Prove that the cost of a liter of kerosene is 44 cents given the conditions
theorem kerosene_cost : cost_of_liter_kerosene (cost_of_half_liter_kerosene (cost_of_eight_eggs (cost_of_one_egg (cost_of_dozen_eggs 0.33)))) * 100 = 44 :=
by
  sorry

end kerosene_cost_l12_12443


namespace jaylen_has_2_cucumbers_l12_12827

-- Definitions based on given conditions
def carrots_jaylen := 5
def bell_peppers_kristin := 2
def green_beans_kristin := 20
def total_vegetables_jaylen := 18

def bell_peppers_jaylen := 2 * bell_peppers_kristin
def green_beans_jaylen := (green_beans_kristin / 2) - 3

def known_vegetables_jaylen := carrots_jaylen + bell_peppers_jaylen + green_beans_jaylen
def cucumbers_jaylen := total_vegetables_jaylen - known_vegetables_jaylen

-- The theorem to prove
theorem jaylen_has_2_cucumbers : cucumbers_jaylen = 2 :=
by
  -- We'll place the proof here
  sorry

end jaylen_has_2_cucumbers_l12_12827


namespace largest_prime_divisor_of_sum_of_squares_l12_12714

theorem largest_prime_divisor_of_sum_of_squares :
  ∃ p : ℕ, prime p ∧ p ∣ (36^2 + 45^2) ∧ ∀ q : ℕ, prime q ∧ q ∣ (36^2 + 45^2) → q ≤ p := by
  sorry

end largest_prime_divisor_of_sum_of_squares_l12_12714


namespace ajax_initial_weight_l12_12660

def initial_weight (pounds: ℝ) (kilo_to_pounds: ℝ) : ℝ := pounds / kilo_to_pounds

theorem ajax_initial_weight (weight_loss_per_hour : ℝ)
                           (exercise_hours_per_day : ℝ)
                           (days_per_week : ℝ)
                           (number_of_weeks : ℝ)
                           (final_weight_pounds : ℝ)
                           (kilo_to_pounds : ℝ) :
                           initial_weight ((final_weight_pounds + 
                           (weight_loss_per_hour * exercise_hours_per_day * 
                           days_per_week * number_of_weeks)) ) kilo_to_pounds = 80 :=

by 
  sorry

end ajax_initial_weight_l12_12660


namespace monotonicity_tangent_points_l12_12269

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12269


namespace quadratic_complete_square_l12_12032

theorem quadratic_complete_square (a b c : ℝ) :
  (8*x^2 - 48*x - 288) = a*(x + b)^2 + c → a + b + c = -355 := 
  by
  sorry

end quadratic_complete_square_l12_12032


namespace cube_inequality_contradiction_l12_12882

theorem cube_inequality_contradiction (a b : Real) (h : a > b) : ¬(a^3 <= b^3) := by
  sorry

end cube_inequality_contradiction_l12_12882


namespace find_m_value_l12_12785

noncomputable def possible_value_of_m (m : ℝ) : Prop :=
  {x : ℝ | 0 ≤ x ∧ x ≤ 1} ∩ {x : ℝ | x^2 - 2*x + m > 0} = ∅

theorem find_m_value :
  possible_value_of_m 0 :=
by
  sorry

end find_m_value_l12_12785


namespace a_n_formula_sum_inequality_l12_12838

noncomputable def seq (a : ℕ → ℝ) : ℕ → ℝ
  | 0     => 0
  | (n+1) => a (n + 1) + seq a n

theorem a_n_formula (a : ℕ → ℝ) (a1_pos : 0 < a 1) (h : ∀ n, sqrt (seq a (n + 1)) = sqrt (a 1) * (n + 1)) :
  ∀ n, a (n + 1) = (2 * (n + 1) - 1) * a 1 :=
sorry

theorem sum_inequality (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = (2 * (n + 1) - 1)) :
  ∑ i in range n, 1 / (a i.succ * a (i + 1).succ) < 1 / 2 :=
sorry

end a_n_formula_sum_inequality_l12_12838


namespace largest_possible_difference_l12_12633

theorem largest_possible_difference 
  (weight_A weight_B weight_C : ℝ)
  (hA : 24.9 ≤ weight_A ∧ weight_A ≤ 25.1)
  (hB : 24.8 ≤ weight_B ∧ weight_B ≤ 25.2)
  (hC : 24.7 ≤ weight_C ∧ weight_C ≤ 25.3) :
  ∃ w1 w2 : ℝ, (w1 = weight_C ∧ w2 = weight_C ∧ abs (w1 - w2) = 0.6) :=
by
  sorry

end largest_possible_difference_l12_12633


namespace triangles_not_necessarily_congruent_l12_12757

theorem triangles_not_necessarily_congruent 
  {α β γ : Type} [metric_space α] [metric_space β]
  (T1 T2 : triangle α) (a1 a2 b1 b2 : ℝ)
  (A1 A2 : angle β) 
  (h1 : T1.side1 = a1) (h2 : T1.side2 = b1)
  (h3 : T1.angle = A1)
  (h4 : T2.side1 = a2) (h5 : T2.side2 = b2)
  (h6 : T2.angle = A2)
  (h7 : a1 = a2) (h8 : b1 = b2) (h9 : A1 = A2) 
: ¬(T1 ≅ T2) := sorry

end triangles_not_necessarily_congruent_l12_12757


namespace translate_point_l12_12182

theorem translate_point (x y : ℝ) (dx dy : ℝ) :
  (x, y) = (3, -2) → (dx, dy) = (-2, 3) → (x + dx, y + dy) = (1, 1) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end translate_point_l12_12182


namespace find_alpha_l12_12842

noncomputable def α_and_β_are_complex_numbers (α β : ℂ) : Prop := true
noncomputable def α_plus_β_is_positive_real (α β : ℂ) : Prop := ∃ r : ℝ, r > 0 ∧ (α + β) = r
noncomputable def i_α_minus_3β_is_positive_real (α β : ℂ) : Prop := ∃ r : ℝ, r > 0 ∧ (i * (α - 3 * β)) = r
def beta_value : ℂ := 4 + 3 * complex.i

theorem find_alpha (α : ℂ) (β : ℂ := beta_value) 
  (h1 : α_and_β_are_complex_numbers α β) 
  (h2 : α_plus_β_is_positive_real α β)
  (h3 : i_α_minus_3β_is_positive_real α β) 
  (h4 : β = beta_value) : α = 3 - 3 * complex.i := 
by
  sorry

end find_alpha_l12_12842


namespace express_bus_max_time_l12_12089

theorem express_bus_max_time (t1 t2 t3 t4 : ℕ) (h : t1 + t2 + t3 + t4 = 20) :
  max t1 t2 t3 t4 ≤ 17 :=
sorry

end express_bus_max_time_l12_12089


namespace theta_range_l12_12768

theorem theta_range (θ : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (Real.cos θ) * x^2 - (4 * Real.sin θ) * x + 6)
  (h2 : ∀ x, f x > 0)
  (h3 : 0 < θ ∧ θ < Real.pi) : θ ∈ set.Ioo 0 (Real.pi / 3) :=
by
  sorry

end theta_range_l12_12768


namespace good_set_exists_l12_12596

def is_good_set (A : List ℕ) : Prop :=
  ∀ i ∈ A, i > 0 ∧ ∀ j ∈ A, i ≠ j → i ^ 2015 % (List.prod (A.erase i)) = 0

theorem good_set_exists (n : ℕ) (h : 3 ≤ n ∧ n ≤ 2015) : 
  ∃ A : List ℕ, A.length = n ∧ ∀ (a : ℕ), a ∈ A → a > 0 ∧ is_good_set A :=
sorry

end good_set_exists_l12_12596


namespace factorize_polynomial_l12_12703

theorem factorize_polynomial (c : ℝ) :
  (x : ℝ) → (x - 1) * (x - 3) = x^2 - 4 * x + c → c = 3 :=
by 
  sorry

end factorize_polynomial_l12_12703


namespace solve_for_x_l12_12532

theorem solve_for_x (x : ℚ) : 
  (5 * x + 8 * x = 350 - 9 * (x + 8)) → 
  (x = 139 / 11) :=
by
  intro h
  sorry

end solve_for_x_l12_12532


namespace total_money_raised_l12_12112

def tickets_sold : ℕ := 25
def ticket_price : ℕ := 2
def num_15_donations : ℕ := 2
def donation_15_amount : ℕ := 15
def donation_20_amount : ℕ := 20

theorem total_money_raised : 
  tickets_sold * ticket_price + num_15_donations * donation_15_amount + donation_20_amount = 100 := 
by sorry

end total_money_raised_l12_12112


namespace crayon_ratio_l12_12481

theorem crayon_ratio :
  ∀ (Karen Beatrice Gilbert Judah : ℕ),
    Karen = 128 →
    Beatrice = Karen / 2 →
    Beatrice = Gilbert →
    Gilbert = 4 * Judah →
    Judah = 8 →
    Beatrice / Gilbert = 1 :=
by
  intros Karen Beatrice Gilbert Judah hKaren hBeatrice hEqual hGilbert hJudah
  sorry

end crayon_ratio_l12_12481


namespace monotonicity_tangent_intersection_points_l12_12376

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12376


namespace mixed_number_eval_l12_12142

theorem mixed_number_eval :
  -|-(18/5 : ℚ)| - (- (12 /5 : ℚ)) + (4/5 : ℚ) = - (2 / 5 : ℚ) :=
by
  sorry

end mixed_number_eval_l12_12142


namespace strategy_game_cost_l12_12056

def total_amount : ℝ := 35.52
def cost_football_game : ℝ := 14.02
def cost_batman_game : ℝ := 12.04
def cost_strategy_game : ℝ := total_amount - cost_football_game - cost_batman_game

theorem strategy_game_cost :
  cost_strategy_game = 9.46 :=
by
  rfl

end strategy_game_cost_l12_12056


namespace area_difference_of_squares_l12_12087

theorem area_difference_of_squares (d1 d2 : ℝ) (h1 : d1 = 19) (h2 : d2 = 17) : 
  let s1 := d1 / Real.sqrt 2
  let s2 := d2 / Real.sqrt 2
  let area1 := s1 * s1
  let area2 := s2 * s2
  (area1 - area2) = 36 :=
by
  sorry

end area_difference_of_squares_l12_12087


namespace difference_of_areas_l12_12991

theorem difference_of_areas (r : ℝ) (h_r : r = 3) (s : ℕ) (h_s : s = 6) :
  let circle_area := (real.pi * r^2)
  let triangle_area := ((real.sqrt 3) / 4 * s^2 : ℝ)
  (circle_area - triangle_area) = 9 * (real.pi - real.sqrt 3) := by
  sorry

end difference_of_areas_l12_12991


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12395

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12395


namespace range_of_m_l12_12764

noncomputable def f (m x : ℝ) : ℝ := m * (x - 1 / x) - 2 * Real.log x
noncomputable def g (m x : ℝ) : ℝ := -m / x

theorem range_of_m (m : ℝ) :
  (∃ x₀ ∈ Set.Icc 1 Real.exp 1, f m x₀ < g m x₀) → m < 2 / Real.exp 1 :=
by
  sorry

end range_of_m_l12_12764


namespace tangent_line_eqn_extreme_value_range_of_m_l12_12404

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) - 2 * x

-- Problem 1: Tangent line equation
theorem tangent_line_eqn : (∀ x : ℝ, x = 0 → let y := f x in y = 1) :=
begin
  sorry,
end

-- Problem 2: Extreme value of f(x)
theorem extreme_value : ∀ x : ℝ, f(x) ≥ f(0) :=
begin
  sorry,
end

-- Problem 3: Inequality and range of m
theorem range_of_m (m : ℝ) : (∀ x : ℝ, f(x) > 2 * (Real.exp 1 - 1) * x + m) → m < 0 :=
begin
  sorry,
end

end tangent_line_eqn_extreme_value_range_of_m_l12_12404


namespace probability_cheryl_same_color_is_1_over_28_l12_12629

def totalWays : ℕ := (nat.choose 9 3) * (nat.choose 6 3) * (nat.choose 3 3)
def favorableOutcomes : ℕ := 3 * (nat.choose 6 3) * (nat.choose 3 3)
def probability : ℚ := favorableOutcomes / totalWays

theorem probability_cheryl_same_color_is_1_over_28 : probability = 1/28 := 
by
  sorry

end probability_cheryl_same_color_is_1_over_28_l12_12629


namespace parallelogram_area_l12_12643

theorem parallelogram_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let y_top := a
  let y_bottom := -b
  let x_left := -c + 2*y
  let x_right := d - 2*y 
  (d + c) * (a + b) = ad + ac + bd + bc :=
by
  sorry

end parallelogram_area_l12_12643


namespace probability_ratio_l12_12168

theorem probability_ratio (cards : finset (fin 50)) 
  (h1 : ∀ n : ℕ, n ∈ cards → n ≤ 50)
  (h2 : ∀ n ∈ cards, (1 ≤ n) ∧ (n ≤ 10))
  (each_number_five : ∀ n : fin 10, (cards.filter (λ c, (c % 10) = n)).card = 5) :
  let p := 10 / (nat.choose 50 5) in
  let q := (90 * 5 * 5) / (nat.choose 50 5) in
  q / p = 225 :=
by sorry

end probability_ratio_l12_12168


namespace smallest_middle_side_length_l12_12557

theorem smallest_middle_side_length 
  (a : ℕ) (h : ℕ) 
  (ha1 : a > 0)
  (ha2 : triangle_acute_angle (a : ℝ) (a+1 : ℝ) (a+2 : ℝ))
  (ha3 : (h:ℝ)^2 = (a^2 + (a - 3)/2 * (a - 3)/2)
  (ha4 : (h:ℝ)^2 = (a^2 + (a + 5)/2 * (a + 5)/2))
  (hdiff : abs ((a+1 - (a-3)/2) - (a-3)/2) = 4 ∨ abs ((a+1 - (a+5)/2) - (a+5)/2) = 4) 
  : (a+1) = 26 :=
sorry

end smallest_middle_side_length_l12_12557


namespace largest_prime_factor_5040_l12_12958

theorem largest_prime_factor_5040 : ∃ p, Nat.Prime p ∧ p ∣ 5040 ∧ (∀ q, Nat.Prime q ∧ q ∣ 5040 → q ≤ p) := by
  use 7
  constructor
  · exact Nat.prime_7
  constructor
  · exact dvd.intro (2^4 * 3^2 * 5) rfl
  · intros q hq
    cases hq with hq1 hq2
    exact Nat.le_of_dvd (Nat.pos_of_ne_zero (λ hq3, by linarith)) hq2
  sorry

end largest_prime_factor_5040_l12_12958


namespace parabola_focus_directrix_distance_l12_12770

theorem parabola_focus_directrix_distance 
  (p : ℝ) 
  (hp : 3 = p * (1:ℝ)^2) 
  (hparabola : ∀ x : ℝ, y = p * x^2 → x^2 = (1/3:ℝ) * y)
  : (distance_focus_directrix : ℝ) = (1 / 6:ℝ) :=
  sorry

end parabola_focus_directrix_distance_l12_12770


namespace original_triangle_area_l12_12548

theorem original_triangle_area (A_new : ℝ) (h1 : A_new = 32) : ∃ A_original : ℝ, A_original = 8 :=
by {
  use 8,
  sorry
}

end original_triangle_area_l12_12548


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12306

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12306


namespace possible_values_for_p_l12_12935

-- Definitions for the conditions
variables {a b c p : ℝ}

-- Assumptions
def distinct (a b c : ℝ) := ¬(a = b) ∧ ¬(b = c) ∧ ¬(c = a)
def main_eq (a b c p : ℝ) := a + (1 / b) = p ∧ b + (1 / c) = p ∧ c + (1 / a) = p

-- Theorem statement
theorem possible_values_for_p (h1 : distinct a b c) (h2 : main_eq a b c p) : p = 1 ∨ p = -1 := 
sorry

end possible_values_for_p_l12_12935


namespace part1_monotonicity_part2_tangent_intersection_l12_12265

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12265


namespace express_a_b_find_a_b_m_n_find_a_l12_12884

-- 1. Prove that a = m^2 + 5n^2 and b = 2mn given a + b√5 = (m + n√5)^2
theorem express_a_b (a b m n : ℤ) (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) :
  a = m ^ 2 + 5 * n ^ 2 ∧ b = 2 * m * n := sorry

-- 2. Prove there exists positive integers a = 6, b = 2, m = 1, and n = 1 such that 
-- a + b√5 = (m + n√5)^2.
theorem find_a_b_m_n : ∃ (a b m n : ℕ), a = 6 ∧ b = 2 ∧ m = 1 ∧ n = 1 ∧ 
  (a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) := sorry

-- 3. Prove a = 46 or a = 14 given a + 6√5 = (m + n√5)^2 and a, m, n are positive integers.
theorem find_a (a m n : ℕ) (h : a + 6 * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) :
  a = 46 ∨ a = 14 := sorry

end express_a_b_find_a_b_m_n_find_a_l12_12884


namespace polar_coordinates_of_P_l12_12235

noncomputable def point_p_rect : ℝ × ℝ := (-1, real.sqrt 3)

noncomputable def rho (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

noncomputable def theta (x y : ℝ) : ℝ := 
if y / x < 0 then real.atan2 y x + real.pi else real.atan2 y x

theorem polar_coordinates_of_P :
  let ⟨x, y⟩ := point_p_rect in
  let r := rho x y in
  let t := theta x y in
  (r, t) = (2, 2 * real.pi / 3) :=
by
  sorry

end polar_coordinates_of_P_l12_12235


namespace heptagon_sum_l12_12513

def permutation : list ℕ := [1, 14, 8, 13, 9, 12, 6, 5, 2, 7, 3, 4]

def sides (l : list ℕ) : list (ℕ × ℕ × ℕ) :=
  [(l[0], l[1], l[2]), (l[2], l[3], l[4]), (l[4], l[5], l[6]), 
   (l[6], l[7], l[8]), (l[8], l[9], l[10]), (l[10], l[11], l[0])]

theorem heptagon_sum : ∃ l : list ℕ, sides l = [(1, 14, 4), (14, 2, 3), (3, 7, 9), (9, 4, 6), (6, 5, 8), (8, 13, 5), (13, 1, 2)] ∧ 
  ∀ (a b c : ℕ), (a, b, c) ∈ sides l → a + b + c = 19 :=
sorry

end heptagon_sum_l12_12513


namespace prob_draw_2_groupA_is_one_third_game_rule_is_unfair_l12_12932

-- Definitions
def groupA : List ℕ := [2, 4, 6]
def groupB : List ℕ := [3, 5]
def card_count_A : ℕ := groupA.length
def card_count_B : ℕ := groupB.length

-- Condition 1: Probability of drawing the card with number 2 from group A
def prob_draw_2_groupA : ℚ := 1 / card_count_A

-- Condition 2: Game Rule Outcomes
def is_multiple_of_3 (n : ℕ) : Bool := n % 3 == 0

def outcomes : List (ℕ × ℕ) := [(2, 3), (2, 5), (4, 3), (4, 5), (6, 3), (6, 5)]

def winning_outcomes_A : List (ℕ × ℕ) :=List.filter (λ p => is_multiple_of_3 (p.1 * p.2)) outcomes
def winning_outcomes_B : List (ℕ × ℕ) := List.filter (λ p => ¬ is_multiple_of_3 (p.1 * p.2)) outcomes

def prob_win_A : ℚ := winning_outcomes_A.length / outcomes.length
def prob_win_B : ℚ := winning_outcomes_B.length / outcomes.length

-- Proof problems
theorem prob_draw_2_groupA_is_one_third : prob_draw_2_groupA = 1 / 3 := sorry

theorem game_rule_is_unfair : prob_win_A ≠ prob_win_B := sorry

end prob_draw_2_groupA_is_one_third_game_rule_is_unfair_l12_12932


namespace monotonicity_and_tangent_intersections_l12_12350

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12350


namespace triangle_base_l12_12042

theorem triangle_base (sum_of_sides : ℕ) (left_side : ℕ) (right_side : ℕ) (base : ℕ)
  (h1 : sum_of_sides = 50)
  (h2 : right_side = left_side + 2)
  (h3 : left_side = 12) :
  base = 24 :=
by
  sorry

end triangle_base_l12_12042


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12336

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12336


namespace monotonicity_of_f_tangent_intersection_l12_12293

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12293


namespace distribution_two_books_each_distribution_one_two_three_books_distribution_into_three_piles_l12_12930

-- (1)
theorem distribution_two_books_each (books: ℕ) (people: ℕ) (b1 b2 : ℕ) (b3 : ℕ) (p1 p2 p3 : ℕ)
    (h_books: books = 6) (h_people: people = 3) (h_two_books_each: b1 = 2 ∧ b2 = 2 ∧ b3 = 2 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) :
    (6.choose 2) * (4.choose 2) * (2.choose 2) = 90 := by
  sorry

-- (2)
theorem distribution_one_two_three_books (books: ℕ) (people: ℕ) (b1 b2 b3 : ℕ) (p1 p2 p3 : ℕ)
    (h_books: books = 6) (h_people: people = 3) (h_one_two_three_books: b1 = 1 ∧ b2 = 2 ∧ b3 = 3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) :
    (6.choose 1) * (5.choose 2) * (3.choose 3) * nat.factorial 3 = 360 := by
  sorry

-- (3)
theorem distribution_into_three_piles (books: ℕ) (h_books: books = 6) :
    (6.choose 2) * (4.choose 2) * (2.choose 2) / nat.factorial 3 = 15 := by
  sorry

end distribution_two_books_each_distribution_one_two_three_books_distribution_into_three_piles_l12_12930


namespace part1_monotonicity_part2_tangent_intersection_l12_12264

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12264


namespace math_problem_proof_l12_12164

theorem math_problem_proof :
  let a := real.sqrt (16 / 9) in
  let b := 16 / 9 in
  let c := (16 / 9) ^ 2 in
  int.ceil a - int.ceil b + int.ceil c = 4 :=
by 
  sorry 

end math_problem_proof_l12_12164


namespace expected_attempts_for_10_keys_10_suitcases_l12_12110

noncomputable def expected_attempts (n : ℕ) : ℝ :=
  (n * (n + 3)) / 4 - (Real.log n + 0.577)

theorem expected_attempts_for_10_keys_10_suitcases :
  abs (expected_attempts 10 - 29.62) < 0.01 := 
by
  sorry

end expected_attempts_for_10_keys_10_suitcases_l12_12110


namespace strictly_increasing_interval_l12_12710

def f (x : ℝ) : ℝ := Real.logb (1 / 2) (-x^2 - 2x + 3)

theorem strictly_increasing_interval : ∀ x ∈ Ioo (-1 : ℝ) 1, ∀ y ∈ Ioo (-1 : ℝ) 1, (x < y → f(x) < f(y)) :=
by
  sorry

end strictly_increasing_interval_l12_12710


namespace problem_inequality_l12_12791

theorem problem_inequality (n : ℕ) (x : Fin n.succ → ℝ)
  (h : ∀ i : Fin n.succ, 1 / 4 < x i ∧ x i < 1) :
  (∑ i : Fin n, Real.log (x i.succ - 1 / 4) / Real.log (x i) ) ≥ 2 * n := by
  sorry

end problem_inequality_l12_12791


namespace probability_between_lines_l12_12150

open Real

-- Define the equations of lines l and n
def line_l (x : ℝ) : ℝ := -2 * x + 8
def line_n (x : ℝ) : ℝ := -3 * x + 9

-- Area under line l in the 1st quadrant
def area_under_l : ℝ := (1 / 2) * 4 * 8

-- Area under line n in the 1st quadrant
def area_under_n : ℝ := (1 / 2) * 3 * 9

-- Area between lines l and n
def area_between_ln : ℝ := area_under_l - area_under_n

-- The required probability
def required_probability : ℝ := area_between_ln / area_under_l

theorem probability_between_lines :
  abs (required_probability - 0.16) < 0.01 :=
by
  sorry

end probability_between_lines_l12_12150


namespace volume_of_solid_l12_12674

noncomputable def volume_of_revolution : ℝ :=
  π * ∫ x in 2..3, (-x^2 + 5*x - 6)^2

theorem volume_of_solid :
  volume_of_revolution = π * ∫ x in 2..3, (-x^2 + 5*x - 6)^2 :=
sorry

end volume_of_solid_l12_12674


namespace compute_alpha_l12_12844

noncomputable def alpha (β : ℂ) (x y : ℝ) : ℂ := (3 / 4) * x - (3 / 4) * y * complex.I

theorem compute_alpha (α β : ℂ) (x y : ℝ) 
  (h1 : α + β = x)
  (h2 : i * (α - 3 * β) = y)
  (h3 : β = 4 + 3 * complex.I)
  (h4 : x > 0)
  (h5 : y > 0) : 
  α = 12 - 3 * complex.I :=
by
  sorry

end compute_alpha_l12_12844


namespace jake_bought_more_notebooks_l12_12505

variable (notebook_cost : ℝ)
variable (marie_paid : ℝ := 3.75)
variable (jake_paid : ℝ := 5.00)
variable (n : ℕ)

noncomputable def notebooks_bought_difference : ℕ :=
  (jake_paid - marie_paid) / notebook_cost

theorem jake_bought_more_notebooks :
  notebook_cost > 0.25 →
  ∀ k, (k * notebook_cost = marie_paid → ∃ m, m * notebook_cost = jake_paid ∧ (m - k = 5)) :=
by
  intros hc hmarie
  sorry

end jake_bought_more_notebooks_l12_12505


namespace sqrt_k_kn_eq_k_sqrt_kn_l12_12593

theorem sqrt_k_kn_eq_k_sqrt_kn (k n : ℕ) (h : k = Nat.sqrt (n + 1)) : 
  Real.sqrt (k * (k / n)) = k * Real.sqrt (k / n) := 
sorry

end sqrt_k_kn_eq_k_sqrt_kn_l12_12593


namespace find_triangle_height_l12_12817

-- Define the problem conditions
def Rectangle.perimeter (l : ℕ) (w : ℕ) : ℕ := 2 * l + 2 * w
def Rectangle.area (l : ℕ) (w : ℕ) : ℕ := l * w
def Triangle.area (b : ℕ) (h : ℕ) : ℕ := (b * h) / 2

-- Conditions
namespace Conditions
  -- Perimeter of the rectangle is 60 cm
  def rect_perimeter (l w : ℕ) : Prop := Rectangle.perimeter l w = 60
  -- Base of the right triangle is 15 cm
  def tri_base : ℕ := 15
  -- Areas of the rectangle and the triangle are equal
  def equal_areas (l w h : ℕ) : Prop := Rectangle.area l w = Triangle.area tri_base h
end Conditions

-- Proof problem: Given these conditions, prove h = 30
theorem find_triangle_height (l w h : ℕ) 
  (h1 : Conditions.rect_perimeter l w)
  (h2 : Conditions.equal_areas l w h) : h = 30 :=
  sorry

end find_triangle_height_l12_12817


namespace find_possible_triples_l12_12706

theorem find_possible_triples :
  ∀ (m n p : ℕ), prime p → 2 ^ m * p ^ 2 + 1 = n ^ 5 → (m, n, p) = (1, 3, 11) :=
by
  sorry

end find_possible_triples_l12_12706


namespace aliceneeds12fills_l12_12662

theorem aliceneeds12fills (total_sugar : ℚ) (cup_size : ℚ) (h1 : total_sugar = 3 + 3/4) (h2 : cup_size = 1/3) : 
  let fills := total_sugar / cup_size in
  fills.ceil = 12 :=
by
  sorry

end aliceneeds12fills_l12_12662


namespace part_a_part_b_l12_12034

-- Definition of the sequence
def x : ℕ → ℝ
| 0       := 2
| (n + 1) := (2 + x n) / (1 - 2 * x n)

theorem part_a (n : ℕ) : x n ≠ 0 := 
sorry

theorem part_b : ¬ ∃ T : ℕ, ∀ n : ℕ, x (n + T) = x n :=
sorry

end part_a_part_b_l12_12034


namespace balls_picked_at_random_eq_two_l12_12097

-- Define the initial conditions: number of balls of each color
def num_red_balls : ℕ := 5
def num_blue_balls : ℕ := 4
def num_green_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_blue_balls + num_green_balls

-- Define the given probability
def given_probability : ℚ := 0.15151515151515152

-- Define the probability calculation for picking two red balls
def probability_two_reds : ℚ :=
  (num_red_balls / total_balls) * ((num_red_balls - 1) / (total_balls - 1))

-- The theorem to prove
theorem balls_picked_at_random_eq_two :
  probability_two_reds = given_probability → n = 2 :=
by
  sorry

end balls_picked_at_random_eq_two_l12_12097


namespace parallel_lines_slope_l12_12691

theorem parallel_lines_slope {a : ℝ} 
    (h1 : ∀ x y : ℝ, 4 * y + 3 * x - 5 = 0 → y = -3 / 4 * x + 5 / 4)
    (h2 : ∀ x y : ℝ, 6 * y + a * x + 4 = 0 → y = -a / 6 * x - 2 / 3)
    (h_parallel : ∀ x₁ y₁ x₂ y₂ : ℝ, (4 * y₁ + 3 * x₁ - 5 = 0 ∧ 6 * y₂ + a * x₂ + 4 = 0) → -3 / 4 = -a / 6) : 
  a = 4.5 := sorry

end parallel_lines_slope_l12_12691


namespace part1_monotonicity_part2_tangent_intersection_l12_12258

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l12_12258


namespace parabola_standard_form_t_constant_l12_12230

-- Define the given condition of the directrix of parabola C
def directrix : ℝ := -1/4

-- Define the standard equation of parabola C
def parabola (y x : ℝ) : Prop := y^2 = x

-- Question (I): Prove the standard equation of the parabola
theorem parabola_standard_form : 
    ∃ p : ℝ, parabola y x ↔ y^2 = x := 
    sorry

-- Define the line l through point P(t, 0) intersects parabola C at points A and B
def line_l (t m y : ℝ) : Prop := t = x - my

-- Define the condition that the circle with diameter AB passes through the origin
def circle_diameter_passing_origin (A B origin : Point) : Prop :=
    let ⟨x1, y1⟩ := A,
        ⟨x2, y2⟩ := B in 
    (x1 * x2 + y1 * y2 = 0)

-- Question (II): Prove t is a constant and determine this constant
theorem t_constant :
    ∀ t m y : ℝ,
    ∃ (a b : ℝ), 
    (line_l t m y) ∧
    (circle_diameter_passing_origin (a, b) (a, b) (0, 0)) → 
        t = 1 ∨ t = 0 :=
    sorry

end parabola_standard_form_t_constant_l12_12230


namespace max_remainder_div_l12_12071

theorem max_remainder_div (quotient divisor remainder : ℕ) (hquot : quotient = 18) (hdiv : divisor = 6) (hrem : remainder = 5) :
  quotient * divisor + remainder = 113 :=
by {
  rw [hquot, hdiv, hrem],
  exact rfl,
}

end max_remainder_div_l12_12071


namespace final_result_l12_12184

noncomputable def n_shuffling (n : ℕ) : Type := {σ : Fin n → Fin n // ∃! (i : Fin n), σ i ≠ i}

noncomputable def f_condition (q n : ℕ) (f : (Fin n → ZMod q) → ZMod q) : Prop :=
∀ (i : Fin n) (x : Fin n → ZMod q) (y z : ZMod q),
  f (x ∘ Function.update i y) + f (x ∘ Function.update i z) = 
  f (x ∘ Function.update i (y + z))

noncomputable def f_symmetry (q n : ℕ) (f : (Fin n → ZMod q) → ZMod q)
  (σ : n_shuffling n) : Prop :=
∀ (x : Fin n → ZMod q),
  f x = - f (σ.val ∘ x)

noncomputable def epsilon (q n : ℕ)
  (σ1 σ2 σ3 : n_shuffling n) : ℝ := sorry

noncomputable def kappa (q n : ℕ)
  (σ1 σ2 σ3 : n_shuffling n) : ℝ :=
- Real.log q (- Real.log (λ (ε : nnreal), (ε - 1) / (q - 1))
    (ε q n σ1 σ2 σ3))

noncomputable def pi (n : ℕ) : ℝ :=
(p(n) : ℝ) / (q(n) : ℝ)

theorem final_result (p q : ℝ → ℝ) (n : ℕ)
  (h_inf : ∀ (n : ℕ), ∃! (k : ℕ), pi n = p(n) / q(n))
  (hq_nonzero : q (-3) ≠ 0) :
  (p(-3) / q(-3)) = 1 :=
sorry

end final_result_l12_12184


namespace ones_digit_sum_of_powers_l12_12962

-- Given conditions that for any positive integer n and exponent k that is three more than a multiple of 4,
-- the ones digit of n^k is the ones digit of n.
def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_sum_of_powers :
  (List.range 2023).sumBy (λ n, ones_digit (n+1)) % 10 = 6 :=
sorry

end ones_digit_sum_of_powers_l12_12962


namespace upper_half_plane_mapping_to_w_plane_l12_12502

noncomputable def mapping_function (h : ℝ) (z : ℂ) : ℂ :=
  (2 * h / real.pi) * (complex.sqrt (z^2 - 1) + complex.arcsin (1 / z))

theorem upper_half_plane_mapping_to_w_plane (h : ℝ) (z : ℂ) (Im_z_pos : 0 < z.im) : ℂ :
  let w := mapping_function h z in
  ∃ w1 w2 w3 w4 : ℂ,
    w1 = -h ∧ w2 = -∞ ∧ w3 = h ∧ w4 = ∞ ∧
    ∃ z1 z2 z3 z4 : ℂ,
      z1 = -1 ∧ z2 = 0 ∧ z3 = 1 ∧ z4 = ∞ ∧
      (w = mapping_function h z) := 
sorry

end upper_half_plane_mapping_to_w_plane_l12_12502


namespace probability_of_draw_l12_12968

theorem probability_of_draw 
  (P_A_wins : ℝ)
  (P_A_not_loses : ℝ)
  (h1 : P_A_wins = 0.3)
  (h2 : P_A_not_loses = 0.8) :
  ∃ P_draw : ℝ, P_draw = 0.5 :=
by
  let P_draw := P_A_not_loses - P_A_wins
  have h3: P_draw = 0.8 - 0.3, from calc
    P_draw = P_A_not_loses - P_A_wins : by rfl
    ... = 0.8 - 0.3 : by rw [h1, h2]
  existsi P_draw
  rw h3
  norm_num
  done

end probability_of_draw_l12_12968


namespace monotonicity_tangent_points_l12_12272

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12272


namespace exists_someone_with_all_names_l12_12659

-- Definitions based on the conditions
def played_with_each (participants : Finset α) :=
  ∀ p ∈ participants, ∀ q ∈ participants, p ≠ q → (played_game p q)

def defeated (p q : α) : Prop
  := (* some condition representing p defeated q *) sorry

def defeated_and_adopted_names (p : α) (names : Finset α) : Finset α :=
  names ∪ { n | ∃ q, q ∈ names ∧ defeated q n }

def no_self_name (p : α) (names : Finset α) : Prop :=
  p ∉ names

-- The main statement
theorem exists_someone_with_all_names
  (participants : Finset α)
  (H_played_with_each : played_with_each participants)
  (H_defeated_names : ∀ p : α, p ∈ participants -> ∃ names : Finset α, (∀ q ∈ names, defeated p q) ∧ (names = defeated_and_adopted_names p names) ∧ no_self_name p names) :
  ∃ p ∈ participants, ∀ q ∈ participants, q ≠ p → q ∈ (defeated_and_adopted_names p (names_of_participant p)) :=
begin
  sorry
end

end exists_someone_with_all_names_l12_12659


namespace ones_digit_sum_of_powers_l12_12961

-- Given conditions that for any positive integer n and exponent k that is three more than a multiple of 4,
-- the ones digit of n^k is the ones digit of n.
def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_sum_of_powers :
  (List.range 2023).sumBy (λ n, ones_digit (n+1)) % 10 = 6 :=
sorry

end ones_digit_sum_of_powers_l12_12961


namespace product_of_roots_l12_12141

noncomputable def cuberoot (x : ℝ) : ℝ := x^(1/3)
noncomputable def sixthroot (x : ℝ) : ℝ := x^(1/6)

theorem product_of_roots : cuberoot 8 * sixthroot 64 = 4 :=
by
  -- Definitions of the roots
  have h1 : cuberoot 8 = 2,
  { sorry },
  have h2 : sixthroot 64 = 2,
  { sorry },
  -- Use h1 and h2 to finish the proof
  rw [h1, h2],
  exact mul_self 2  

end product_of_roots_l12_12141


namespace coeff_a2b3_l12_12599

theorem coeff_a2b3 (a b c : ℚ) : ((a + b) ^ 5 * (c + 1 / c) ^ 8).coeff (a^2 * b^3) = 700 := by
  sorry

end coeff_a2b3_l12_12599


namespace triangle_perimeter_correct_l12_12126

def side_a : ℕ := 15
def side_b : ℕ := 8
def side_c : ℕ := 10
def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter_correct :
  perimeter side_a side_b side_c = 33 := by
sorry

end triangle_perimeter_correct_l12_12126


namespace part1_part2_max_part2_min_l12_12860

def vector_a (x : ℝ) : ℝ × ℝ := (real.sqrt 3 * real.sin x, real.sin x)
def vector_b (x : ℝ) : ℝ × ℝ := (real.cos x, real.sin x)
def f (x : ℝ) : ℝ := let a := vector_a x in let b := vector_b x in a.1 * b.1 + a.2 * b.2

theorem part1 (hx : 0 ≤ x ∧ x ≤ real.pi / 2) (ha_eq_hb : (vector_a x).fst ^ 2 + (vector_a x).snd ^ 2 = (vector_b x).fst ^ 2 + (vector_b x).snd ^ 2) : 
  x = real.pi / 6 :=
sorry

theorem part2_max (hx : 0 ≤ x ∧ x ≤ real.pi / 2) : 
  ∃x, f(x) = 3 / 2 :=
sorry

theorem part2_min (hx : 0 ≤ x ∧ x ≤ real.pi / 2) : 
  ∃x, f(x) = 0 :=
sorry

end part1_part2_max_part2_min_l12_12860


namespace min_lateral_surface_area_l12_12790

noncomputable def minimum_lateral_surface_area (V : ℝ) : ℝ :=
  -- Define the volume of the cone
  let h := 2 in
  let r := sqrt (4 / h) in
  let S := π * r * sqrt(r^2 + h^2) in
  S

theorem min_lateral_surface_area (V : ℝ) (h : ℝ) (r : ℝ) (S : ℝ) :
  V = 4 * π / 3 → (V = π / 3 * r^2 * h) →
  (S = π * r * sqrt (r^2 + h^2)) →
  (minimum_lateral_surface_area V) = 2 * sqrt 3 * π :=
by
  intros
  sorry

end min_lateral_surface_area_l12_12790


namespace cylinder_height_in_sphere_l12_12648

-- Definitions based on conditions
def radius_sphere : ℝ := 7
def radius_cylinder : ℝ := 3

-- Mathematical equivalent proof problem
theorem cylinder_height_in_sphere : 
  ∃ h : ℝ, (radius_sphere)^2 = (radius_cylinder)^2 + (h / 2)^2 ∧ h = 4 * real.sqrt 10 :=
begin
  sorry
end

end cylinder_height_in_sphere_l12_12648


namespace parallelogram_side_length_l12_12117

theorem parallelogram_side_length {s : ℝ} (h1 : 0 < s) 
  (h2 : let base := 3 * s; let height := s * real.sqrt 3 / 2 in base * height = 9 * real.sqrt 3) 
  : s = real.sqrt 6 :=
sorry

end parallelogram_side_length_l12_12117


namespace find_triangle_base_l12_12038

theorem find_triangle_base (left_side : ℝ) (right_side : ℝ) (base : ℝ) 
  (h_left : left_side = 12) 
  (h_right : right_side = left_side + 2)
  (h_sum : left_side + right_side + base = 50) :
  base = 24 := 
sorry

end find_triangle_base_l12_12038


namespace returning_players_l12_12919

def baseball_team : Type := 
  {new_players : ℕ // new_players = 48} × 
  {groups : ℕ // groups = 9} × 
  {players_per_group : ℕ // players_per_group = 6}

theorem returning_players (team : baseball_team) : 
  ∃ returning : ℕ, returning = (team.2.1 * team.2.2) - team.1 :=
by {
  use (team.2.1 * team.2.2) - team.1,
  sorry
}

end returning_players_l12_12919


namespace draws_alternate_no_consecutive_same_color_l12_12630

-- Defining the total number of balls and the count of each color.
def total_balls : ℕ := 15
def white_balls : ℕ := 5
def black_balls : ℕ := 5
def red_balls : ℕ := 5

-- Defining the probability that the draws alternate in colors with no two consecutive balls of the same color.
def probability_no_consecutive_same_color : ℚ := 162 / 1001

theorem draws_alternate_no_consecutive_same_color :
  (white_balls + black_balls + red_balls = total_balls) →
  -- The resulting probability based on the given conditions.
  probability_no_consecutive_same_color = 162 / 1001 := by
  sorry

end draws_alternate_no_consecutive_same_color_l12_12630


namespace base_of_triangle_is_24_l12_12044

def triangle_sides_sum := 50
def left_side : ℕ := 12
def right_side := left_side + 2
def base := triangle_sides_sum - left_side - right_side

theorem base_of_triangle_is_24 :
  base = 24 :=
by 
  have h : left_side = 12 := rfl
  have h2 : right_side = 14 := by simp [right_side, h]
  have h3 : base = 24 := by simp [base, triangle_sides_sum, h, h2]
  exact h3

end base_of_triangle_is_24_l12_12044


namespace problem1_problem2_problem3_l12_12187

-- Define the splitting transformation.
def splitting_transformation (a b : ℕ) (h : 0 < a ∧ 0 < b) : Prop :=
  1 / (a * b) = (1 / a) * (1 / a - 1 / b)

-- (1) Proof for 1 / 20 using splitting transformation with 2 and 10.
theorem problem1 : splitting_transformation 2 10 (by norm_num) := sorry

-- (2) Proof for 1 / 15 using splitting transformation with 3 and 5.
theorem problem2 : splitting_transformation 3 5 (by norm_num) := sorry

-- (3) Simplification of the given series.
theorem problem3 (x : ℕ) : sum (λ k, 1 / ((x + 2 * k) * (x + 2 * (k + 1)))) = 1002 / (x * (x + 2024)) := sorry

end problem1_problem2_problem3_l12_12187


namespace range_of_x_if_angle_is_obtuse_l12_12778

-- Define the vectors a and b given the conditions
def vector_a (x : ℝ) : EuclideanSpace ℝ (Fin 3) := ![x, 2, 0]
def vector_b (x : ℝ) : EuclideanSpace ℝ (Fin 3) := ![3, 2 - x, x^2]

-- Lean 4 statement for proving the range of x
theorem range_of_x_if_angle_is_obtuse :
  ∀ x : ℝ, (inner (vector_a x) (vector_b x) < 0) → (x < -4) := 
by
  intros
  sorry

end range_of_x_if_angle_is_obtuse_l12_12778


namespace linear_inequality_solution_l12_12203

theorem linear_inequality_solution {x y m n : ℤ} 
  (h_table: (∀ x, if x = -2 then y = 3 
                else if x = -1 then y = 2 
                else if x = 0 then y = 1 
                else if x = 1 then y = 0 
                else if x = 2 then y = -1 
                else if x = 3 then y = -2 
                else true)) 
  (h_eq: m * x - n = y) : 
  x ≥ -1 :=
sorry

end linear_inequality_solution_l12_12203


namespace slope_of_line_l12_12602

theorem slope_of_line (x y : ℝ) : (4 * y = 5 * x - 20) → (y = (5/4) * x - 5) :=
by
  intro h
  sorry

end slope_of_line_l12_12602


namespace continuity_of_f_at_5_l12_12520

theorem continuity_of_f_at_5 (f : ℝ → ℝ) (x₀ : ℝ)
  (hf : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x₀| < δ → |f x - f x₀| < ε) :
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 5| < δ → |f x - 98| < ε) :=
by
  sorry

noncomputable def f : ℝ → ℝ := λ x, 4 * x^2 - 2
noncomputable def x₀ : ℝ := 5

example : (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x₀| < δ → |f x - f x₀| < ε) :=
by
  have hf : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 5| < δ → |f x - 98| < ε := sorry
  exact hf

end continuity_of_f_at_5_l12_12520


namespace erika_savings_l12_12698

-- Definitions based on conditions
def gift_cost : ℕ := 250
def cake_cost : ℕ := 25
def leftover : ℕ := 5
def rick_savings := gift_cost / 2

-- Condition: total amount Erika and Rick have together
def total_savings := gift_cost + cake_cost + leftover

-- Prove Erika's savings
theorem erika_savings : ∃ e : ℕ, e + rick_savings = total_savings ∧ e = 155 :=
by
  have total_savings_eq : total_savings = gift_cost + cake_cost + leftover := rfl
  rw total_savings_eq
  have rick_savings_eq : rick_savings = gift_cost / 2 := rfl
  rw rick_savings_eq
  existsi (total_savings - rick_savings)
  split
  { sorry }
  { rw [total_savings_eq, rick_savings_eq]
    norm_num
    exact eq.refl 155 }

end erika_savings_l12_12698


namespace statement_B_incorrect_l12_12490

variables {a b c : Type} -- represent lines
variables {α β : Type} -- represent planes
variables {Projection : Type → Type → Type} -- a function representing the projection of line c onto α

-- Definitions of the relevant conditions
def line_subset_plane (b : Type) (α : Type) : Prop := ∀ x ∈ b, x ∈ α
def line_not_subset_plane (c : Type) (α : Type) : Prop := ∃ x ∈ c, x ∉ α
def planes_parallel (α β : Type) : Prop := ∀ x ∈ α, x ∈ β
def line_perpendicular_plane (c : Type) (α : Type) : Prop := ∀ x ∈ c, ∀ y ∈ α, x ⊥ y
def line_perpendicular (x y : Type) : Prop := ∀ p ∈ x, ∀ q ∈ y, p ⊥ q
def line_parallel (x y : Type) : Prop := ∃ u ∈ x, ∃ v ∈ y, u // v

-- Statement that we need to disprove under the given conditions
theorem statement_B_incorrect
  (h1 : line_subset_plane b α)
  (h2 : line_not_subset_plane c α)
  (h3 : planes_parallel α β)
  (h4 : line_perpendicular_plane c α)
  (h5 : ∀ (b β : Type), line_perpendicular b β → planes_parallel α β)
  (h6 : ∀ (a : Type), Projection c α = a → line_perpendicular a b → line_perpendicular b c)
  (h7 : ∀ (b c : Type), line_parallel b c → line_parallel c α) : 
  ¬(∀ (b β : Type), planes_parallel α β → line_perpendicular b β → ¬line_perpendicular α β) := sorry

end statement_B_incorrect_l12_12490


namespace gallons_bought_l12_12185

variable (total_needed : ℕ) (existing_paint : ℕ) (needed_more : ℕ)

theorem gallons_bought (H : total_needed = 70) (H1 : existing_paint = 36) (H2 : needed_more = 11) : 
  total_needed - existing_paint - needed_more = 23 := 
sorry

end gallons_bought_l12_12185


namespace derivative_f_at_1_l12_12732

noncomputable def f (x : Real) : Real := x^3 * Real.sin x

theorem derivative_f_at_1 : deriv f 1 = 3 * Real.sin 1 + Real.cos 1 := by
  sorry

end derivative_f_at_1_l12_12732


namespace cost_per_pancake_correct_l12_12478

-- Define the daily rent expense
def daily_rent := 30

-- Define the daily supplies expense
def daily_supplies := 12

-- Define the number of pancakes needed to cover expenses
def number_of_pancakes := 21

-- Define the total daily expenses
def total_daily_expenses := daily_rent + daily_supplies

-- Define the cost per pancake calculation
def cost_per_pancake := total_daily_expenses / number_of_pancakes

-- The theorem to prove the cost per pancake
theorem cost_per_pancake_correct :
  cost_per_pancake = 2 := 
by
  sorry

end cost_per_pancake_correct_l12_12478


namespace n19_minus_n7_div_30_l12_12879

theorem n19_minus_n7_div_30 (n : ℕ) (h : 0 < n) : 30 ∣ (n^19 - n^7) :=
sorry

end n19_minus_n7_div_30_l12_12879


namespace number_of_valid_pairings_l12_12926

-- Define the context of the problem
def people : Type := Fin 12

-- Definition of knowing relationship
def knows (x y : people) : Prop :=
  (x + 1 = y ∨ x - 1 = y ∨ x + 6 = y ∨ x - 6 = y)

def valid_pairs : List (people × people) :=
   [ (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), 
     (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 0),
     (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11) ]

-- Definition of a valid pairing
def valid_pairing (pairs : List (people × people)) : Prop :=
  ∀ (a b c : people), 
    (knows a b ∧ knows b c ∧ knows a c) → (a ≠ b ∧ b ≠ c ∧ a ≠ c)

-- The main statement of the problem
theorem number_of_valid_pairings :
  ∃ (pairs : List (List (people × people))),
  (∀ p ∈ pairs, valid_pairing p) ∧ length pairs = 14 := 
sorry

end number_of_valid_pairings_l12_12926


namespace seniors_playing_all_three_l12_12594

variables {α : Type} [Fintype α]
variables (F B L : Finset α)

def total_seniors := F ∪ B ∪ L
def football := F
def baseball := B
def lacrosse := L
def football_lacrosse := F ∩ L
def baseball_football := B ∩ F
def baseball_lacrosse := B ∩ L
def all_three := F ∩ B ∩ L

variable (n : ℕ)

theorem seniors_playing_all_three (h1 : total_seniors = 85)
    (h2 : football.card = 74)
    (h3 : baseball.card = 26)
    (h4 : football_lacrosse.card = 17)
    (h5 : baseball_football.card = 18)
    (h6 : baseball_lacrosse.card = 13)
    (h7 : lacrosse.card = 2 * n) :
    all_three.card = 11 :=
by sorry

end seniors_playing_all_three_l12_12594


namespace analytical_expression_of_f_range_of_real_number_k_range_of_x_that_satisfies_condition_l12_12409

-- Given the power function f(x) = (m^2 - 3m + 3) * x^{2 - m^2} is monotonically increasing on ℝ.

def power_function (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - 3 * m + 3) * x ^ (2 - m^2)

-- Given g(x) = k * f^2(x) + (k - 3) * f(x) + 1 has at least one zero point on the right side of the origin.

def g_function (k : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  k * (f x) ^ 2 + (k - 3) * (f x) + 1

-- Definitions for h1, h2, and h3 based on given conditions.
def h1_function (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  abs ((f x) ^ 2 - 3)

def h2_function (h1 : ℝ → ℝ) (x : ℝ) : ℝ :=
  abs (h1 x - 3)

def h3_function (h2 : ℝ → ℝ) (x : ℝ) : ℝ :=
  abs (h2 x - 3)

-- Prove the solution.
theorem analytical_expression_of_f :
  ∃ m, ∀ x, power_function m x = x :=
sorry

theorem range_of_real_number_k :
  ∃ (k : ℝ), ∀ (x : ℝ), g_function k (λ x, x) x = 0 → x > 0 :=
sorry

theorem range_of_x_that_satisfies_condition :
  ∀ (f : ℝ → ℝ), (∀ x, h3_function (h2_function (h1_function f)) x = h1_function f x) → 
  ∀ x, x ∈ Set.Icc (-Real.sqrt 6) (Real.sqrt 6) :=
sorry

end analytical_expression_of_f_range_of_real_number_k_range_of_x_that_satisfies_condition_l12_12409


namespace cyclic_quadrilateral_AIMN_l12_12852

/--
Let \(ABC\) be a triangle with incentre \(I\). 
The angle bisectors \(AI, BI\), and \(CI\) meet \([BC],[CA]\), and \([AB]\) at \(D, E\), 
and \(F\) respectively. The perpendicular bisector of \([AD]\) intersects the lines \(BI\) 
and \(CI\) at \(M\) and \(N\) respectively.
Then, \(A, I, M, \) and \(N\) lie on a circle.
-/
theorem cyclic_quadrilateral_AIMN
  (A B C I D E F M N : Type)
  [triangle ABC with_incentre I]
  [angle_bisectors A I B I C I meet_at D E F]
  [perpendicular_bisector A D intersects B I C I at M N] :
  cyclic_quadrilateral A I M N := sorry

end cyclic_quadrilateral_AIMN_l12_12852


namespace greatest_possible_median_l12_12012

theorem greatest_possible_median
  (k m r s t : ℕ) 
  (h_pos : ∀ x, x ∈ {k, m, r, s, t} → x > 0)
  (h_order : k < m ∧ m < r ∧ r < s ∧ s < t)
  (h_mean : (k + m + r + s + t) / 5 = 16)
  (h_t : t = 42) :
  r ≤ 32 :=
sorry

end greatest_possible_median_l12_12012


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12388

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12388


namespace julia_tuesday_kids_l12_12480

theorem julia_tuesday_kids :
  ∃ x : ℕ, (∃ y : ℕ, y = 6 ∧ y = x + 1) → x = 5 := 
by
  sorry

end julia_tuesday_kids_l12_12480


namespace distance_between_A_and_B_l12_12096

theorem distance_between_A_and_B 
  (d : ℕ) -- The distance we want to prove
  (ha : ∀ (t : ℕ), d = 700 * t)
  (hb : ∀ (t : ℕ), d + 400 = 2100 * t) :
  d = 1700 := 
by
  sorry

end distance_between_A_and_B_l12_12096


namespace admission_fee_for_children_l12_12011

theorem admission_fee_for_children (x : ℝ) :
  (∀ (admission_fee_adult : ℝ) (total_people : ℝ) (total_fees_collected : ℝ) (children_admitted : ℝ) (adults_admitted : ℝ),
    admission_fee_adult = 4 ∧
    total_people = 315 ∧
    total_fees_collected = 810 ∧
    children_admitted = 180 ∧
    adults_admitted = total_people - children_admitted ∧
    total_fees_collected = children_admitted * x + adults_admitted * admission_fee_adult
  ) → x = 1.5 := sorry

end admission_fee_for_children_l12_12011


namespace power_function_properties_l12_12555

theorem power_function_properties (α : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ α) 
    (h_point : f (1/2) = 2 ) :
    (∀ x : ℝ, f x = 1 / x) ∧ (∀ x : ℝ, 0 < x → (f x) < (f (x / 2))) ∧ (∀ x : ℝ, f (-x) = - (f x)) :=
by
  sorry

end power_function_properties_l12_12555


namespace a8_value_l12_12738

variable {an : ℕ → ℕ}

def S (n : ℕ) : ℕ := n ^ 2

theorem a8_value : an 8 = S 8 - S 7 := by
  sorry

end a8_value_l12_12738


namespace ellipsoid_volume_div_pi_l12_12853

noncomputable def ellipsoid_projection_min_area : ℝ := 9 * Real.pi
noncomputable def ellipsoid_projection_max_area : ℝ := 25 * Real.pi
noncomputable def ellipsoid_circle_projection_area : ℝ := 16 * Real.pi
noncomputable def ellipsoid_volume (a b c : ℝ) : ℝ := (4/3) * Real.pi * a * b * c

theorem ellipsoid_volume_div_pi (a b c : ℝ)
  (h_min : (a * b = 9))
  (h_max : (b * c = 25))
  (h_circle : (b = 4)) :
  ellipsoid_volume a b c / Real.pi = 75 := 
  by
    sorry

end ellipsoid_volume_div_pi_l12_12853


namespace monotonicity_tangent_intersection_points_l12_12384

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12384


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12307

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12307


namespace opposite_face_l12_12869

/--
Given a cube with faces labeled as А, Б, В, Г, Д, Е, and the following views:
1. Faces visible from the first viewpoint: А, Б, Г
2. Faces visible from the second viewpoint: Г, Д, Е
3. Faces visible from the third viewpoint: В, Д, Б
Prove that the face opposite to the one labeled Д is labeled Б.
-/
theorem opposite_face (A B V G D E : Type) 
  (first_view : set (Type)) (second_view : set (Type)) (third_view : set (Type)) : 
  first_view = {A, B, G} ∧ second_view = {G, D, E} ∧ third_view = {V, D, B} → 
  (opposite_face D = B) :=
by
  sorry

end opposite_face_l12_12869


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l12_12337

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l12_12337


namespace functional_ineq_l12_12835

noncomputable def f : ℝ → ℝ := sorry

theorem functional_ineq (h1 : ∀ x > 1400^2021, x * f x ≤ 2021) (h2 : ∀ x : ℝ, 0 < x → f x = f (x + 2) + 2 * f (x * (x + 2))) : 
  ∀ x : ℝ, 0 < x → x * f x ≤ 2021 :=
sorry

end functional_ineq_l12_12835


namespace domain_g_l12_12953

def g (x : ℝ) : ℝ := 1 / ((x - 2) ^ 2 + (x + 2) ^ 2 + 4)

theorem domain_g : ∀ x : ℝ, (x - 2) ^ 2 + (x + 2) ^ 2 + 4 ≠ 0 := 
by
  intro x
  have h : 2 * x ^ 2 + 12 > 0 := by
    calc
      2 * x ^ 2 + 12 = 2 * (x ^ 2 + 6) : by ring
      _ > 0 : by norm_num; apply add_pos_of_nonneg_of_pos; norm_num
  exact ne_of_gt h

end domain_g_l12_12953


namespace ellipse_equation_max_area_triangle_OPQ_l12_12750

-- Defining the conditions for the ellipse and the given information
variables {a b c : ℝ}
variables (h1 : a > b) (h2 : b > 0)
variables (h3 : a + c = 2 + real.sqrt 3) (h4 : a - c = 2 - real.sqrt 3)
variables (h5 : ∀ (P : ℝ × ℝ) (hP : P ∈ set_of (λ ⟨x, y⟩, (x^2 / a^2) + (y^2 / b^2) = 1)), 
                  |(P.1 - c)| = 2 + real.sqrt 3 ∨ |(P.1 - c)| = 2 - real.sqrt 3)

-- Goal 1: Proving the equation of the ellipse C
theorem ellipse_equation : (a = 2 ∧ b = 1) → (∀ (x y : ℝ), (x^2 / 4) + y^2 = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1) := by
  sorry

-- Goal 2: Finding the maximum area of △OPQ given the conditions
variables {l : ℝ → ℝ}
variables (hl : ¬ ∃ x, l x = 0)
variables (h6 : ∀ (P Q : ℝ × ℝ), P ∈ set_of (λ ⟨x, y⟩, (x^2 / 4) + y^2 = 1) →
                                  Q ∈ set_of (λ ⟨x, y⟩, (x^2 / 4) + y^2 = 1) →
                                  let m := ((Q.2 - P.2) / (Q.1 - P.1)) in
                                  m ≠ 0 → let k1 := P.2 / P.1, k2 := Q.2 / Q.1 in 
                                           k1 * m = m * k2 )

theorem max_area_triangle_OPQ : (∃ (m : ℝ), 0 < m^2 ∧ m^2 < 2) → ( ∃ m, (m^2 = 1) ∧ let d := (2 * |m| / real.sqrt 5) in |10 - 5 * m^2| = 1) := by
  sorry

end ellipse_equation_max_area_triangle_OPQ_l12_12750


namespace monotonicity_tangent_points_l12_12277

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l12_12277


namespace probability_sum_remaining_odd_l12_12730

theorem probability_sum_remaining_odd :
  let S := {1, 2, 3, 4}
  let total_choices := Nat.choose 4 2
  let favorable_choices := 4
  (favorable_choices / total_choices : ℚ) = 2 / 3 :=
by
  sorry

end probability_sum_remaining_odd_l12_12730


namespace triangle_incircle_inequality_l12_12482

variable {a b c : ℕ}

theorem triangle_incircle_inequality (h1 : ∃ (a b c : ℕ), a ≤ b ∧ b + c > a ∧ a + c > b ∧ a + b > c)
                                     (h2 : ∀ (d e : ℕ), d = a + b - c → e = a + b - c →
                                        abs((d^2 + b^2 - 2 * b * d * cos C) - (e^2 + a^2 - 2 * a * e * cos C)) ≤ 2) :
  a = b :=
by
  sorry

end triangle_incircle_inequality_l12_12482


namespace find_m_value_l12_12551

noncomputable def quadratic_has_complex_root_with_real_coeffs (m n : ℝ) : Prop :=
  (Complex := (-3 + 2 * Complex.I)) ∧ (Complex.conj (-3 + 2 * Complex.I) = (-3 - 2 * Complex.I)) ∧ 
  (root1 : Complex) = -3 + 2 * Complex.I ∧ 
  (root2 : Complex) = -3 - 2 * Complex.I ∧
  isRoot := (root1) || isRoot(root2)

theorem find_m_value (m n : ℝ) (h : quadratic_has_complex_root_with_real_coeffs m n) :
  m = 6 := sorry

end find_m_value_l12_12551


namespace magnitude_of_sum_l12_12411

noncomputable def vector_a : ℝ × ℝ := (1, -real.sqrt 3)
def magnitude_b : ℝ := 1
def angle_between_a_b : ℝ := real.pi * 2 / 3 -- 120 degrees in radians

theorem magnitude_of_sum (b : ℝ × ℝ) 
  (hb : |b| = magnitude_b)
  (angle : ∃ θ : ℝ, θ = angle_between_a_b ∧ cos θ = -1/2):
  let sum := (vector_a.1 + b.1, vector_a.2 + b.2) in
  |sum| = real.sqrt 3:= 
sorry

end magnitude_of_sum_l12_12411


namespace area_ratio_l12_12469

variables {A B C D E F P Q R : Type}
variables [linear_ordered_field Type] [ordered_add_comm_group Type]

noncomputable theory

-- Defining points and corresponding ratios
def BD_DC (BD DC : ℝ) : Prop := BD / DC = 1 / 3
def CE_EA (CE EA : ℝ) : Prop := CE / EA = 1 / 3
def AF_FB (AF FB : ℝ) : Prop := AF / FB = 1 / 3

-- The main theorem stating the desired result
theorem area_ratio (BD DC CE EA AF FB : ℝ) (hBD : BD_DC BD DC) (hCE : CE_EA CE EA)
                  (hAF : AF_FB AF FB) : 
  (area_triangle P Q R / area_triangle A B C) = 1 / 64 :=
sorry

end area_ratio_l12_12469


namespace monotonicity_of_f_tangent_intersection_points_l12_12323

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12323


namespace count_arrangements_l12_12454

theorem count_arrangements (students : Fin 5) (A B : Fin 5) (hA_ne_B : A ≠ B) : 
  ∃ n : Nat, n = 72 ∧ 
    n = (5.factorial - (4.factorial * 2.factorial)) := 
sorry

end count_arrangements_l12_12454


namespace find_spicy_curries_now_l12_12995

-- Definitions based on conditions
def peppers_very_spicy := 3
def peppers_spicy := 2
def peppers_mild := 1

def very_spicy_curries_previous := 30
def spicy_curries_previous := 30
def mild_curries_previous := 10

def mild_curries_now := 90
def pepper_reduction := 40

-- Prove the number of spicy curries they buy peppers for now
theorem find_spicy_curries_now :
  let total_peppers_previous := peppers_very_spicy * very_spicy_curries_previous +
                                peppers_spicy * spicy_curries_previous +
                                peppers_mild * mild_curries_previous in
  let total_peppers_now := total_peppers_previous - pepper_reduction in
  let peppers_mild_now := peppers_mild * mild_curries_now in
  ∃ x : ℕ, peppers_mild_now + peppers_spicy * x = total_peppers_now ∧ x = 15 :=
by {
  sorry
}

end find_spicy_curries_now_l12_12995


namespace reduced_form_fraction_l12_12687

noncomputable def fraction_reduced (b y : ℝ) : ℝ :=
  (sqrt (4 * b ^ 2 + y ^ 4) - (y ^ 4 - 4 * b ^ 2) / sqrt (4 * b ^ 2 + y ^ 4)) / (4 * b ^ 2 + y ^ 4) ^ (2 / 3)

theorem reduced_form_fraction (b y : ℝ) :
  fraction_reduced b y = 8 * b ^ 2 / (4 * b ^ 2 + y ^ 4) :=
by
  sorry

end reduced_form_fraction_l12_12687


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12312

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12312


namespace necessary_condition_not_sufficient_condition_l12_12091

variable (x : ℝ)

def quadratic_condition : Prop := x^2 - 3 * x + 2 > 0
def interval_condition : Prop := x < 1 ∨ x > 4

theorem necessary_condition : interval_condition x → quadratic_condition x := by sorry

theorem not_sufficient_condition : ¬ (quadratic_condition x → interval_condition x) := by sorry

end necessary_condition_not_sufficient_condition_l12_12091


namespace geo_seq_a6_eight_l12_12463

-- Definitions based on given conditions
variable (a : ℕ → ℝ) -- the sequence
variable (q : ℝ) -- common ratio
-- Conditions for a_1 * a_3 = 4 and a_4 = 4
def geometric_sequence := ∃ (q : ℝ), ∀ n : ℕ, a (n + 1) = a n * q
def condition1 := a 1 * a 3 = 4
def condition2 := a 4 = 4

-- Proof problem: Prove a_6 = 8 given the conditions above
theorem geo_seq_a6_eight (h1 : condition1 a) (h2 : condition2 a) (hs : geometric_sequence a) : 
  a 6 = 8 :=
sorry

end geo_seq_a6_eight_l12_12463


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12315

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12315


namespace not_necessarily_circle_l12_12994

open Set

-- Definitions of conditions
def convex_figure (F : Set ℝ) : Prop :=
  Convex ℝ F

def equilateral_triangle (T : Set ℝ) : Prop :=
  -- A placeholder definition for an equilateral triangle with side length 1
  sorry 

def can_translate_triangle_to_boundary (F T : Set ℝ) : Prop :=
  ∃ t : ℝ × ℝ, ∀ v ∈ T, v + t ∈ boundary F

-- Main proof problem
theorem not_necessarily_circle (F : Set ℝ) :
  convex_figure F →
  (∀ T, equilateral_triangle T → can_translate_triangle_to_boundary F T) →
  ¬(∀ x, F = metric.ball x 1) :=
by
  intros h_convex h_property
  -- To be proved: there exists non-circle convex figures with the given property.
  sorry

end not_necessarily_circle_l12_12994


namespace profit_calculation_l12_12888

theorem profit_calculation (boxes_bought : ℕ) (cost_per_box : ℕ) (pens_per_box : ℕ) (packages_per_box : ℕ) (packages_price : ℕ) (sets_per_box : ℕ) (sets_price : ℕ) :
  boxes_bought = 12 ∧
  cost_per_box = 10 ∧
  pens_per_box = 30  ∧ 
  packages_per_box = 5  ∧ 
  packages_price = 3 ∧ 
  sets_per_box = 3 ∧
  sets_price = 2 →
  let total_cost := boxes_bought * cost_per_box in
  let total_pens := boxes_bought * pens_per_box in
  let boxes_used_for_packages := 5 in -- based on given conditions
  let packages := boxes_used_for_packages * packages_per_box in
  let revenue_from_packages := packages * packages_price in
  let boxes_left := boxes_bought - boxes_used_for_packages in
  let pens_left := boxes_left * pens_per_box in
  let sets := pens_left / sets_per_box in
  let revenue_from_pens := sets * sets_price in
  let total_revenue := revenue_from_packages + revenue_from_pens in
  let profit := total_revenue - total_cost in
  profit = 95 :=
by 
  intros;
  -- Ensure that all declared variables and their calculations are consistent with the problem conditions shown above
  sorry

end profit_calculation_l12_12888


namespace floor_sum_min_value_l12_12424

theorem floor_sum_min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (⌊(x + y + z) / x⌋ + ⌊(x + y + z) / y⌋ + ⌊(x + y + z) / z⌋) = 7 :=
sorry

end floor_sum_min_value_l12_12424


namespace polar_to_cartesian_equation_l12_12812

def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

theorem polar_to_cartesian_equation (rho theta : ℝ) (h : rho = 2 * Real.cos theta) :
  let (x, y) := polar_to_cartesian rho theta in ((x - 1)^2 + y^2 = 1) :=
by
  let (x, y) := polar_to_cartesian rho theta
  have h₁ : rho^2 = (2 * Real.cos theta)^2 := by { rw h, sorry }
  have h₂ : rho^2 = x^2 + y^2 := by { sorry }
  have h₃ : 2 * rho * Real.cos theta = 2 * x := by { sorry }
  rw [h₁, h₂, h₃]
  sorry

end polar_to_cartesian_equation_l12_12812


namespace eq_970299_l12_12608

theorem eq_970299 : 98^3 + 3 * 98^2 + 3 * 98 + 1 = 970299 :=
by
  sorry

end eq_970299_l12_12608


namespace cylinder_height_l12_12650

noncomputable def height_of_cylinder_inscribed_in_sphere : ℝ := 4 * Real.sqrt 10

theorem cylinder_height :
  ∀ (R_cylinder R_sphere : ℝ), R_cylinder = 3 → R_sphere = 7 →
  (height_of_cylinder_inscribed_in_sphere = 4 * Real.sqrt 10) := by
  intros R_cylinder R_sphere h1 h2
  sorry

end cylinder_height_l12_12650


namespace probability_of_one_event_occuring_l12_12030

-- Define the probabilities of the events
variables (p1 p2 : ℝ)

-- Assume probabilities are bound within [0, 1]
axiom h_p1 : 0 ≤ p1 ∧ p1 ≤ 1
axiom h_p2 : 0 ≤ p2 ∧ p2 ≤ 1

-- Theorem statement: probability of occurrence of exactly one of the events
theorem probability_of_one_event_occuring (A1 A2 : Prop) (p1 p2 : ℝ) 
  [ProbA1 : Decidable A1] [ProbA2 : Decidable A2] 
  (indep : ∀ (A1_true : A1) (A2_true : A2), A1_true = A2_true → false) :
  (p1 * (1 - p2) + (1 - p1) * p2) = p1 * (1 - p2) + (1 - p1) * p2 :=
sorry

end probability_of_one_event_occuring_l12_12030


namespace notebooks_multiple_of_3_l12_12997

theorem notebooks_multiple_of_3 (N : ℕ) (h1 : ∃ k : ℕ, N = 3 * k) :
  ∃ k : ℕ, N = 3 * k :=
by
  sorry

end notebooks_multiple_of_3_l12_12997


namespace least_prime_factor_of_5_pow_4_sub_5_pow_2_l12_12960

theorem least_prime_factor_of_5_pow_4_sub_5_pow_2 : 
  let n := 5 ^ 4 - 5 ^ 2 in
  ∃ p : ℕ, prime p ∧ p ∣ n ∧ ∀ q : ℕ, prime q ∧ q ∣ n → p ≤ q :=
begin
  let n := 5 ^ 4 - 5 ^ 2,
  use 2,
  sorry
end

end least_prime_factor_of_5_pow_4_sub_5_pow_2_l12_12960


namespace find_f_x_minus_1_l12_12763

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem find_f_x_minus_1 (x : ℝ) : f (x - 1) = 2 * x - 3 := by
  sorry

end find_f_x_minus_1_l12_12763


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12295

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12295


namespace g_value_l12_12400

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def g (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ) - 1

theorem g_value (ω φ : ℝ) (h : ∀ x : ℝ, f ω φ (π / 4 - x) = f ω φ (π / 4 + x)) :
  g ω φ (π / 4) = -1 :=
sorry

end g_value_l12_12400


namespace solve_inequality_l12_12001

theorem solve_inequality (x : ℝ) :
  (x + 3) / (x - 1) > (4 * x + 5) / (3 * x + 4) →
  (7 - real.sqrt 66 < x) ∧ (x < 7 + real.sqrt 66) :=
by
  intro h
  sorry

end solve_inequality_l12_12001


namespace solution_set_leq_2_l12_12205

theorem solution_set_leq_2 (x y m n : ℤ)
  (h1 : m * 0 - n = 1)
  (h2 : m * 1 - n = 0)
  (h3 : y = m * x - n) :
  x ≥ -1 ↔ m * x - n ≤ 2 :=
by {
  sorry
}

end solution_set_leq_2_l12_12205


namespace largest_prime_factor_5040_is_7_l12_12955

-- Definition of the condition: the prime factorization of 5040
def prime_factorization_5040 : list ℕ := [2, 2, 2, 2, 3, 3, 5, 7]

-- Predicate to check if a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m:ℕ, m ∣ n → m = 1 ∨ m = n

-- Predicate to check if a list contains only primes
def all_primes (l: list ℕ) : Prop :=
  ∀ x, x ∈ l → is_prime x

-- Statement of the problem
theorem largest_prime_factor_5040_is_7 :
  all_primes prime_factorization_5040 ∧ 
  list.prod prime_factorization_5040 = 5040 ∧
  list.maximum prime_factorization_5040 = 7 :=
sorry

end largest_prime_factor_5040_is_7_l12_12955


namespace integral_part_of_expression_l12_12741

theorem integral_part_of_expression (n : ℕ) (h : n ≥ 62) :
  (⌊∑ k in Finset.range (n - 1), 
    (1 / ∏ m in Finset.range (k + 1), (1 + (m.succ : ℕ) / n)) - 
    ∑ k in Finset.range (n - 1),
    (∏ m in Finset.range (k + 1), (1 - (m.succ : ℕ) / n))⌋ : ℤ) = 0 :=
  sorry

end integral_part_of_expression_l12_12741


namespace monotonicity_tangent_intersection_points_l12_12379

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12379


namespace max_range_of_temperatures_l12_12897

theorem max_range_of_temperatures (avg_temp : ℝ) (low_temp : ℝ) (days : ℕ) (total_temp: ℝ) (high_temp : ℝ) 
  (h1 : avg_temp = 60) (h2 : low_temp = 50) (h3 : days = 5) (h4 : total_temp = avg_temp * days) 
  (h5 : total_temp = 300) (h6 : 4 * low_temp + high_temp = total_temp) : 
  high_temp - low_temp = 50 := 
by
  sorry

end max_range_of_temperatures_l12_12897


namespace page_numbers_sum_l12_12570

theorem page_numbers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 136080) : n + (n + 1) + (n + 2) = 144 :=
by
  sorry

end page_numbers_sum_l12_12570


namespace only_zero_sol_l12_12176

theorem only_zero_sol (x y z t : ℤ) : x^2 + y^2 + z^2 + t^2 = 2 * x * y * z * t → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 :=
by
  sorry

end only_zero_sol_l12_12176


namespace arithmetic_to_geometric_l12_12983

theorem arithmetic_to_geometric (a1 a2 a3 a4 d : ℝ)
  (h_arithmetic : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d)
  (h_d_nonzero : d ≠ 0):
  ((a2^2 = a1 * a3 ∨ a2^2 = a1 * a4 ∨ a3^2 = a1 * a4 ∨ a3^2 = a2 * a4) → (a1 / d = 1 ∨ a1 / d = -4)) :=
by {
  sorry
}

end arithmetic_to_geometric_l12_12983


namespace sam_needs_change_l12_12448

noncomputable def toy_prices : List ℝ := [0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50]

def sam_initial_quarters : ℝ := 12.0 * 0.25

def favorite_toy_price : ℝ := 4.00

def total_permutations (n : ℕ) := (Finset.range n).card.fact

def favorable_permutations : ℕ := (Finset.range 7).card.fact

def probability_no_change_needed : ℝ := favorable_permutations / total_permutations 10

def probability_change_needed : ℝ := 1 - probability_no_change_needed

theorem sam_needs_change : probability_change_needed = 719 / 720 :=
by
  sorry

end sam_needs_change_l12_12448


namespace average_is_equal_l12_12014

theorem average_is_equal (x : ℝ) :
  (1 / 3) * (2 * x + 4 + 5 * x + 3 + 3 * x + 8) = 3 * x - 5 → 
  x = -30 :=
by
  sorry

end average_is_equal_l12_12014


namespace sum_of_x_cos_equation_l12_12723

theorem sum_of_x_cos_equation (x : ℝ) (hx1 : cos (3 * x) ^ 3 + cos (5 * x) ^ 3 = 8 * cos (4 * x) ^ 3 * cos (x) ^ 3)
(hx2 : 100 < x ∧ x < 200) :
      x = 150 ∨ x = 126 ∨ x = 162 ∨ x = 198 ∨ x = 112.5 ∨ x = 157.5 →
      (150 + 126 + 162 + 198 + 112.5 + 157.5 = 906) := by
sorry

end sum_of_x_cos_equation_l12_12723


namespace find_n_l12_12568

def has_property (n : ℕ) : Prop :=
  ∃ (x y : ℕ), n = 1000 * x + y ∧ 0 ≤ y ∧ y < 1000 ∧ x = n^(1/3 : ℚ) ∧ x^3 = n

theorem find_n : {n : ℕ // has_property n} :=
  ⟨ 32768, have h1 : 32768 = 1000 * 32 + 768, by norm_num,
      have h2 : 0 ≤ 768, by norm_num,
      have h3 : 768 < 1000, by norm_num,
      have h4 : 32 = 32768^(1/3 : ℚ), by norm_num,
      have h5 : 32^3 = 32768, by norm_num,
      ⟨ 32, 768, h1, h2, h3, h4, h5 ⟩ ⟩

end find_n_l12_12568


namespace compute_alpha_l12_12845

noncomputable def alpha (β : ℂ) (x y : ℝ) : ℂ := (3 / 4) * x - (3 / 4) * y * complex.I

theorem compute_alpha (α β : ℂ) (x y : ℝ) 
  (h1 : α + β = x)
  (h2 : i * (α - 3 * β) = y)
  (h3 : β = 4 + 3 * complex.I)
  (h4 : x > 0)
  (h5 : y > 0) : 
  α = 12 - 3 * complex.I :=
by
  sorry

end compute_alpha_l12_12845


namespace coordinates_of_point_P_l12_12437

noncomputable def curve := λ x : ℝ, x^4 - x

theorem coordinates_of_point_P :
  ∃ x y : ℝ, y = curve x ∧ (curve x).derivative = 3 ∧ (x = 1 ∧ y = 0) :=
by
  sorry

end coordinates_of_point_P_l12_12437


namespace triangle_base_l12_12041

theorem triangle_base (sum_of_sides : ℕ) (left_side : ℕ) (right_side : ℕ) (base : ℕ)
  (h1 : sum_of_sides = 50)
  (h2 : right_side = left_side + 2)
  (h3 : left_side = 12) :
  base = 24 :=
by
  sorry

end triangle_base_l12_12041


namespace valid_arrangements_l12_12451

def five_students := fin 5

def arrangements (students : fin 5) :=
  if ∃ (i j : fin 5), i ≠ j ∧ (i = 2 ∧ j = 4 ∨ i = 4 ∧ j = 2)
  then 72 else sorry

theorem valid_arrangements (students : fin 5) (h: ¬(∃ (i j : fin 5), i ≠ j ∧ (i = 2 ∧ j = 4 ∨ i = 4 ∧ j = 2))): 
  arrangements students = 72 :=
sorry

end valid_arrangements_l12_12451


namespace max_combinations_for_n_20_l12_12993

def num_combinations (s n k : ℕ) : ℕ :=
if n = 0 then if s = 0 then 1 else 0
else if s < n then 0
else if k = 0 then 0
else num_combinations (s - k) (n - 1) (k - 1) + num_combinations s n (k - 1)

theorem max_combinations_for_n_20 : ∀ s k, s = 20 ∧ k = 9 → num_combinations s 4 k = 12 :=
by
  intros s k h
  cases h
  sorry

end max_combinations_for_n_20_l12_12993


namespace problem_l12_12241

noncomputable def f (x : ℝ) (α β : ℝ) : ℝ :=
  if x < 0 then sin (x + α) else cos (x + β)

theorem problem (α β : ℝ) (hα : α = π / 3) (hβ : β = π / 6) :
  ∀ x : ℝ, f x α β = f (-x) α β :=
by
  sorry

end problem_l12_12241


namespace oranges_sold_l12_12949

/-
Wendy sells apples at $1.50 each, and oranges at $1 each.
In the morning, she sold 40 apples and 30 oranges.
In the afternoon, she sold 50 apples. Her total sales for the day were $205.
We want to show that she sold 40 oranges in the afternoon.
-/

def apple_price : ℝ := 1.50
def orange_price : ℝ := 1.0
def morning_apples_sold : ℕ := 40
def morning_oranges_sold : ℕ := 30
def afternoon_apples_sold : ℕ := 50
def total_sales : ℝ := 205

def morning_sales :=
  morning_apples_sold * apple_price + morning_oranges_sold * orange_price

def afternoon_apple_sales :=
  afternoon_apples_sold * apple_price

def total_known_sales :=
  morning_sales + afternoon_apple_sales

def afternoon_orange_sales :=
  total_sales - total_known_sales

def afternoon_oranges_sold :=
  afternoon_orange_sales / orange_price

theorem oranges_sold : afternoon_oranges_sold = 40 := by
  -- the proof goes here
  sorry

end oranges_sold_l12_12949


namespace total_notebooks_distributed_l12_12613

theorem total_notebooks_distributed :
  ∀ (N C : ℕ), 
    (N / C = C / 8) →
    (N = 16 * (C / 2)) →
    N = 512 := 
by
  sorry

end total_notebooks_distributed_l12_12613


namespace find_values_of_a_and_b_l12_12745

noncomputable def a : ℝ := 4
noncomputable def b : ℝ := 2

theorem find_values_of_a_and_b
  (h1a : a > 1)
  (h1b : b > 1)
  (h2 : Real.log a / Real.log b + Real.log b / Real.log a = 5 / 2)
  (h3 : a^b = b^a) :
  a = 4 ∧ b = 2 :=
begin
  sorry
end

end find_values_of_a_and_b_l12_12745


namespace general_formula_sum_b_n_l12_12751

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}

-- Given conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

variables (d : ℝ)
axiom h1 : is_arithmetic_sequence a d
axiom h2 : a 2 + a 4 = 14
axiom h3 : (a 2)^2 = a 1 * a 6

-- Define b_n and S_n
def b (n : ℕ) : ℝ := 1 / (a n * a (n + 1))
def S (n : ℕ) : ℝ := ∑ i in finset.range n, b i

-- Prove the general formula for {a_n}
theorem general_formula (n : ℕ) : a n = 3 * n - 2 :=
by sorry

-- Prove the sum of the first n terms of {b_n}
theorem sum_b_n (n : ℕ) : S n = n / (3 * n + 1) :=
by sorry

end general_formula_sum_b_n_l12_12751


namespace incorrect_statement_l12_12971

theorem incorrect_statement : 
  ¬(∀ (p q : Prop), (¬p ∧ ¬q) → (¬p ∧ ¬q)) := 
    sorry

end incorrect_statement_l12_12971


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12303

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12303


namespace least_value_of_k_squared_l12_12495

theorem least_value_of_k_squared (p x y : ℝ) (h₁ : (x - √p) ^ 2 + (y - √p) ^ 2 = 1)
    (k : ℝ) (h₂ : k = y / (x - 3)) : ∃ q, q = k^2 ∧ q = 8 :=
begin
  sorry -- Proof not required
end

end least_value_of_k_squared_l12_12495


namespace triangle_base_l12_12043

theorem triangle_base (sum_of_sides : ℕ) (left_side : ℕ) (right_side : ℕ) (base : ℕ)
  (h1 : sum_of_sides = 50)
  (h2 : right_side = left_side + 2)
  (h3 : left_side = 12) :
  base = 24 :=
by
  sorry

end triangle_base_l12_12043


namespace at_least_one_real_root_l12_12726

theorem at_least_one_real_root (c : ℝ) : (c^2 - 8 ≥ 0) ∨ (12 - 4c ≥ 0) := 
begin
  have h : (c - 2)^2 ≥ 0 := by apply pow_two_nonneg,
  have discriminant_sum_nonneg : (c^2 - 8 + 12 - 4c) = (c - 2)^2,
  { ring },
  rw discriminant_sum_nonneg at h,
  exact or.imp id id (sq_nonneg (c - 2)),
end

end at_least_one_real_root_l12_12726


namespace n_pow_19_minus_n_pow_7_div_30_l12_12877

theorem n_pow_19_minus_n_pow_7_div_30 (n : ℕ) (hn : 0 < n) : 30 ∣ (n^19 - n^7) :=
sorry

end n_pow_19_minus_n_pow_7_div_30_l12_12877


namespace sandy_spent_money_l12_12889

theorem sandy_spent_money :
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 :=
by
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  have total_spent : shorts + shirt + jacket = 33.56 := sorry
  exact total_spent

end sandy_spent_money_l12_12889


namespace area_estimation_correct_l12_12611

-- definition for the region Ω under the curve y = x^2 and bounded by y = 4
def region_under_curve (x : ℝ) : ℝ :=
  x ^ 2

-- define the condition for the random point generation.
def point_in_region (a b : ℝ) : Prop :=
  let a1 := 4 * a - 2 in
  let b1 := 4 * b in
  b1 < a1 ^ 2

noncomputable def estimate_area_of_region (m n : ℕ) (output_n : ℕ) : ℝ :=
  let total_area := 16 in -- area of the bounding rectangle
  let probability_in_region := (output_n.to_real) / (m.to_real) in
  total_area * probability_in_region

-- given condition M = 100
def M : ℕ := 100

-- given output n = 34
def output_n : ℕ := 34

theorem area_estimation_correct :
  estimate_area_of_region M output_n = 10.56 :=
by {
  sorry
}

end area_estimation_correct_l12_12611


namespace sum_arithmetic_sequence_min_n_l12_12216

theorem sum_arithmetic_sequence_min_n (aₙ : ℕ → ℝ) (Sₙ : ℕ → ℝ)
  (h_arithmetic : ∃a₁ d, aₙ = λ n, a₁ + (n - 1) * d)
  (h_sum_formula : ∀ n, Sₙ n = n / 2 * (2 * (aₙ 1) + (n - 1) * (aₙ 1 - aₙ 0))) 
  (h_S10 : Sₙ 10 = -2) 
  (h_S20 : Sₙ 20 = 16) :
  ∃ n, Sₙ n = min (Sₙ n) ∧ n = 6 :=
sorry

end sum_arithmetic_sequence_min_n_l12_12216


namespace monotonicity_of_f_tangent_intersection_l12_12287

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l12_12287


namespace expected_sides_of_red_polygon_l12_12924

-- Define the conditions
def isChosenWithinSquare (F : ℝ × ℝ) (side_length: ℝ) : Prop :=
  0 ≤ F.1 ∧ F.1 ≤ side_length ∧ 0 ≤ F.2 ∧ F.2 ≤ side_length

def pointF (side_length: ℝ) : ℝ × ℝ := sorry
def foldToF (vertex: ℝ × ℝ) (F: ℝ × ℝ) : ℝ := sorry

-- Define the expected number of sides of the resulting red polygon
noncomputable def expected_sides (side_length : ℝ) : ℝ :=
  let P_g := 2 - (Real.pi / 2)
  let P_o := (Real.pi / 2) - 1 
  (3 * P_o) + (4 * P_g)

-- Prove the expected number of sides equals 5 - π / 2
theorem expected_sides_of_red_polygon (side_length : ℝ) :
  expected_sides side_length = 5 - (Real.pi / 2) := 
  by sorry

end expected_sides_of_red_polygon_l12_12924


namespace complex_solution_l12_12433

-- Define the complex number z and its conditions
theorem complex_solution (z : ℂ) (h : (1 + complex.i) * z = complex.i) : 
  z = (1 / 2 : ℂ) + (1 / 2) * complex.i := 
  sorry

end complex_solution_l12_12433


namespace trains_meet_80_km_from_A_l12_12939

-- Define the speeds of the trains
def speed_train_A : ℝ := 60 
def speed_train_B : ℝ := 90 

-- Define the distance between locations A and B
def distance_AB : ℝ := 200 

-- Define the time when the trains meet
noncomputable def meeting_time : ℝ := distance_AB / (speed_train_A + speed_train_B)

-- Define the distance from location A to where the trains meet
noncomputable def distance_from_A (speed_A : ℝ) (meeting_time : ℝ) : ℝ :=
  speed_A * meeting_time

-- Prove the statement
theorem trains_meet_80_km_from_A :
  distance_from_A speed_train_A meeting_time = 80 :=
by
  -- leaving the proof out, it's just an assumption due to 'sorry'
  sorry

end trains_meet_80_km_from_A_l12_12939


namespace fraction_of_cookies_l12_12191

-- Given conditions
variables 
  (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (H1 : Mike_cookies = 3 * Millie_cookies)
  (H2 : Millie_cookies = 4)
  (H3 : Frank_cookies = 3)

-- Proof statement
theorem fraction_of_cookies (Millie_cookies Mike_cookies Frank_cookies : ℕ)
  (H1 : Mike_cookies = 3 * Millie_cookies)
  (H2 : Millie_cookies = 4)
  (H3 : Frank_cookies = 3) : 
  (Frank_cookies / Mike_cookies : ℚ) = 1 / 4 :=
by
  sorry

end fraction_of_cookies_l12_12191


namespace identify_wallet_with_fewest_coins_l12_12048

-- Definitions of wallets and movement of coins
noncomputable def wallets : Fin 8 → ℕ := λ _, 13

noncomputable def move_coin (w : Fin 8) : Fin 8 → ℕ
| ⟨0, _⟩ := if w = 1 then 14 else if w = 0 then 12 else 13
| ⟨7, _⟩ := if w = 6 then 14 else if w = 7 then 12 else 13
| ⟨i+1, _⟩ := if i+1 = w then 14 else if i = w then 12 else 13

-- Balance scale comparison definition
def balance_compare (a b : ℕ) : Ordering :=
if a < b then Ordering.lt else if a > b then Ordering.gt else Ordering.eq

theorem identify_wallet_with_fewest_coins 
    (w : Fin 8) :
    ∃ w₁ w₂ w₃ : Fin 8,
    (balance_compare (move_coin w 3) (move_coin w 6) = Ordering.eq → balance_compare (move_coin w 1) (move_coin w 4) = Ordering.lt ∧ w₃ = 1 ∨ balance_compare (move_coin w 1) (move_coin w 4) = Ordering.gt ∧ w₃ = 4 ∨ balance_compare (move_coin w 1) (move_coin w 4) = Ordering.eq ∧ w₃ = 7) ∧
    (balance_compare (move_coin w 3) (move_coin w 6) = Ordering.gt → balance_compare (move_coin w 2) (move_coin w 6) = Ordering.lt ∧ w₃ = 2 ∨ balance_compare (move_coin w 2) (move_coin w 6) = Ordering.gt ∧ w₃ = 6) ∧
    (balance_compare (move_coin w 3) (move_coin w 6) = Ordering.lt → balance_compare (move_coin w 3) (move_coin w 5) = Ordering.lt ∧ w₃ = 3 ∨ balance_compare (move_coin w 3) (move_coin w 5) = Ordering.gt ∧ w₃ = 5) :=
sorry

end identify_wallet_with_fewest_coins_l12_12048


namespace ending_number_divisible_by_3_l12_12928

theorem ending_number_divisible_by_3 (n : ℕ) :
  (∀ k, 0 ≤ k ∧ k < 13 → ∃ m, 10 ≤ m ∧ m ≤ n ∧ m % 3 = 0) →
  n = 48 :=
by
  intro h
  sorry

end ending_number_divisible_by_3_l12_12928


namespace functions_equiv_l12_12130

theorem functions_equiv (x : ℝ) : (∀ x : ℝ, (Real.cbrt x) ^ 3 = x) → 
  ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = x) ∧ (∀ x : ℝ, f x = (Real.cbrt x) ^ 3) :=
by
  sorry

end functions_equiv_l12_12130


namespace find_n_l12_12485

-- Definition of points and lengths
def P := (0, 0) : ℝ × ℝ
def Q := (0, 90) : ℝ × ℝ
def R := (120, 0) : ℝ × ℝ

def PQ := 90
def PR := 120
def QR := 150

-- Inradius derived from conditions
def inradius_PQR := 30

-- Points on sides
def S := (60, 0) : ℝ × ℝ
def T := (75, 15) : ℝ × ℝ
def U := (0, 60) : ℝ × ℝ
def V := (50, 42.5) : ℝ × ℝ

-- Centers of the inscribed circles
def center_C2 := (75, 15) : ℝ × ℝ
def center_C3 := (50, 42.5) : ℝ × ℝ

def distance_center_C2_C3 := Real.sqrt ((75 - 50)^2 + (15 - 42.5)^2)

theorem find_n : distance_center_C2_C3 = Real.sqrt (10 * 138.125) := sorry

end find_n_l12_12485


namespace sin_210_eq_neg_half_l12_12690

theorem sin_210_eq_neg_half:
  (sin (210 * real.pi / 180)) = -1 / 2 := 
sorry

end sin_210_eq_neg_half_l12_12690


namespace vector_addition_in_triangle_l12_12472

theorem vector_addition_in_triangle
  (A B C D : Type)
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] 
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D]
  (AB AC AD BD DC : A)
  (h1 : BD = 2 • DC) :
  AD = (1/3 : ℝ) • AB + (2/3 : ℝ) • AC :=
sorry

end vector_addition_in_triangle_l12_12472


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12389

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12389


namespace remembers_umbrella_prob_l12_12900

theorem remembers_umbrella_prob 
    (P_forgets : ℚ) 
    (h_forgets : P_forgets = 5 / 8) : 
    ∃ P_remembers : ℚ, P_remembers = 3 / 8 := 
by
    sorry

end remembers_umbrella_prob_l12_12900


namespace oliver_candy_after_giving_l12_12183

theorem oliver_candy_after_giving (initial_candy : ℕ) (candy_given : ℕ) (remaining_candy : ℕ) 
  (h1 : initial_candy = 78) (h2 : candy_given = 10) : remaining_candy = initial_candy - candy_given := 
by {
  have h3 : remaining_candy = 78 - 10 := by rw [← h1, ← h2];
  exact h3;
}

end oliver_candy_after_giving_l12_12183


namespace system_of_equations_solution_l12_12579

theorem system_of_equations_solution :
  ∃ (x y : ℝ), (2 * x + y = 3) ∧ (x - y = 3) ∧ (x = 2 ∧ y = -1) :=
begin
  use [2, -1],
  simp,
  split,
  { exact calc 2 * 2 + (-1) = 4 - 1 : by norm_num
                          ... = 3      : by norm_num },
  split,
  { exact calc 2 - (-1) = 2 + 1 : by norm_num 
                      ... = 3     : by norm_num },
  exact ⟨rfl, rfl⟩,
end

end system_of_equations_solution_l12_12579


namespace minimum_button_presses_to_exit_l12_12078

def arms_after (r y : ℕ) : ℕ := 3 + r - 2 * y
def doors_after (y g : ℕ) : ℕ := 3 + y - 2 * g

theorem minimum_button_presses_to_exit :
  ∃ r y g : ℕ, arms_after r y = 0 ∧ doors_after y g = 0 ∧ r + y + g = 9 :=
sorry

end minimum_button_presses_to_exit_l12_12078


namespace monotonicity_of_f_intersection_points_of_tangent_l12_12302

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l12_12302


namespace fraction_books_sold_l12_12104

theorem fraction_books_sold (B : ℕ) (F : ℚ)
  (hc1 : F * B * 4 = 288)
  (hc2 : F * B + 36 = B) :
  F = 2 / 3 :=
by
  sorry

end fraction_books_sold_l12_12104


namespace monotonicity_and_tangent_intersections_l12_12353

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l12_12353


namespace second_to_last_in_pascals_triangle_l12_12601

theorem second_to_last_in_pascals_triangle (n : ℕ) (hn : n = 57) :
  (finset.range 57).sum binomial 56 = 56 :=
by
  sorry

end second_to_last_in_pascals_triangle_l12_12601


namespace problem_equivalent_l12_12850

noncomputable def f (a b : ℝ) := λ x : ℝ, 2 * x^2 + a * x + b
noncomputable def g (c d : ℝ) := λ x : ℝ, 3 * x^2 + c * x + d
def vertex_x_f (a : ℝ) := -a / 4
def vertex_x_g (c : ℝ) := -c / 6

def intersects_at (f g : ℝ → ℝ) (p : ℝ × ℝ) :=
f p.1 = p.2 ∧ g p.1 = p.2

def same_minimum_value (f g : ℝ → ℝ) (a b c d : ℝ) : Prop :=
2 * (vertex_x_f a)^2 + a * (vertex_x_f a) + b = 
3 * (vertex_x_g c)^2 + c * (vertex_x_g c) + d

def vertex_root_condition (a b c d : ℝ) : Prop :=
3 * (vertex_x_f a)^2 + c * (vertex_x_f a) + d =
2 * (vertex_x_g c)^2 + a * (vertex_x_g c) + b

theorem problem_equivalent (a b c d : ℝ)
  (h1 : vertex_x_f a ∈ { x | g c d x = 0 })
  (h2 : vertex_x_g c ∈ { x | f a b x = 0 })
  (h3 : intersects_at (f a b) (g c d) (50, -200))
  (h4 : same_minimum_value (f a b) (g c d) a b c d)
  (h5 : vertex_root_condition a b c d) :
  a + c = -720 :=
sorry

end problem_equivalent_l12_12850


namespace range_of_f_x_lt_1_l12_12423

theorem range_of_f_x_lt_1 (x : ℝ) (f : ℝ → ℝ) (h : f x = x^3) : f x < 1 ↔ x < 1 := by
  sorry

end range_of_f_x_lt_1_l12_12423


namespace complex_polygon_area_l12_12190

noncomputable def area_of_resulting_polygon (side_length : ℝ) (angles : List ℝ) : ℝ :=
  let r := (side_length * Real.sqrt 2) / 2
  let triangle_area (a b : ℝ) := (1 / 2) * r^2 * Real.sin (b - a)
  let total_triang_area := List.sum $ List.map (λ pair, triangle_area pair.1 pair.2) (angles.zip angles.tail)
  4 * total_triang_area + 4 * side_length^2

theorem complex_polygon_area (side_length : ℝ) (angles : List ℝ) (h1 : side_length = 8)
    (h2 : angles = [0, 20 * Real.pi / 180, 50 * Real.pi / 180, 80 * Real.pi / 180]) :
    area_of_resulting_polygon side_length angles = 685.856 :=
  sorry

end complex_polygon_area_l12_12190


namespace length_of_bridge_l12_12654

-- Given conditions as definitions
def length_of_train : ℝ := 140
def time_to_cross_bridge : ℝ := 28.997680185585153
def speed_of_train_kmph : ℝ := 36
def speed_of_train_mps : ℝ := speed_of_train_kmph * (1000 / 3600)

-- The statement we need to prove
theorem length_of_bridge:
  let distance := speed_of_train_mps * time_to_cross_bridge in
  let bridge_length := distance - length_of_train in
  bridge_length = 149.97680185585153 :=
by
  sorry

end length_of_bridge_l12_12654


namespace range_of_varpi_l12_12434

theorem range_of_varpi (varpi : ℝ) (h : varpi > 0) :
  (∀ x y : ℝ, (x ∈ set.Icc (π / 3) (π / 2)) → (y ∈ set.Icc (π / 3) (π / 2)) → x < y → sin (varpi * y) < sin (varpi * x)) ↔ (3 / 2 ≤ varpi ∧ varpi ≤ 3) := 
sorry

end range_of_varpi_l12_12434


namespace area_of_regular_octagon_with_alternating_sides_l12_12119

-- Definitions based on conditions
def is_regular_octagon (sides : Fin 8 → ℝ) : Prop :=
  (∀ i, sides (Fin8 (2 * i)) = 1) ∧ (∀ i, sides (Fin8 (2 * i + 1)) = Real.sqrt 2)

def area_of_octagon (sides : Fin 8 → ℝ) : ℝ := sorry

-- The mathematical equivalent proof statement
theorem area_of_regular_octagon_with_alternating_sides (sides : Fin 8 → ℝ) (h : is_regular_octagon sides) :
  area_of_octagon sides = 7 := sorry

end area_of_regular_octagon_with_alternating_sides_l12_12119


namespace find_polynomial_l12_12171

theorem find_polynomial (p : ℝ → ℝ) :
  (p 3 = 0) →
  (p (-1) = 0) →
  (p (-3) = 18) →
  (∀ n, degree p < 3) →
  p = (λ x, (3/2)*x^2 - 3*x - 9/2) :=
by
  intro h1 h2 h3 h4
  sorry

end find_polynomial_l12_12171


namespace sum_of_digits_of_N_is_19_l12_12970

-- Given facts about N
variables (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) 
           (h2 : N % 10 = 7) 
           (h3 : N % 11 = 7) 
           (h4 : N % 12 = 7)

-- Main theorem statement
theorem sum_of_digits_of_N_is_19 : 
  ((N / 100) + ((N % 100) / 10) + (N % 10) = 19) := sorry

end sum_of_digits_of_N_is_19_l12_12970


namespace parallelogram_count_l12_12801

theorem parallelogram_count (m n : ℕ) : (choose m 2) * (choose n 2) = number_of_parallelograms m n :=
sorry

end parallelogram_count_l12_12801


namespace base_seven_sum_of_product_l12_12569

theorem base_seven_sum_of_product : 
  let a := 13   -- 16_7 in base 10
  let b := 15   -- 21_7 in base 10
  let prod := a * b
  let prod_base_seven := 276  -- 195_10 in base 7
  let sum_digits := 2 + 7 + 6  -- sum of digits 276 in base 7
  in sum_digits = 3 := sorry

end base_seven_sum_of_product_l12_12569


namespace monotonicity_tangent_intersection_points_l12_12385

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l12_12385


namespace ratio_of_distances_l12_12239

-- Conditions
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

def midpoint_on_y_axis (p f1 : ℝ × ℝ) : Prop := (p.1 + f1.1) / 2 = 0

-- The proof statement
theorem ratio_of_distances (P : ℝ × ℝ) (F1 := (-2, 0) : ℝ × ℝ) (F2 := (2, 0) : ℝ × ℝ)
  (hP_ellipse : ellipse P.1 P.2)
  (h_midpoint : midpoint_on_y_axis P F1) :
  let d1 := Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)
      d2 := Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)
  in d1 / d2 = 5 / 3 := by
  sorry

end ratio_of_distances_l12_12239


namespace prob_either_A_or_B_hired_l12_12105

-- Definitions
def graduates := ["A", "B", "C", "D", "E"]
def num_graduates := 5
def num_to_hire := 2
def total_combinations : ℕ := Nat.choose num_graduates num_to_hire
def combinations_without_A_B : ℕ := Nat.choose (num_graduates - 2) num_to_hire

-- Probability Calculation
def prob_none_A_B : ℚ := combinations_without_A_B / total_combinations
def prob_either_A_B : ℚ := 1 - prob_none_A_B

-- Proof statement
theorem prob_either_A_or_B_hired : prob_either_A_B = 7 / 10 := by
  sorry

end prob_either_A_or_B_hired_l12_12105


namespace monotonicity_of_f_tangent_intersection_points_l12_12330

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l12_12330


namespace difference_max_min_l12_12727

noncomputable def f (x : ℝ) : ℝ := 3 - sin x - 2 * cos x ^ 2

theorem difference_max_min (x : ℝ) 
  (h1 : x ∈ Set.Icc (Real.pi / 6) (7 * Real.pi / 6)) :
  (f (1) - f (-1 / 2)) = 9 / 8 :=
by
  sorry

end difference_max_min_l12_12727


namespace age_problem_l12_12440

variables (K M A B : ℕ)

theorem age_problem
  (h1 : K + 7 = 3 * M)
  (h2 : M = 5)
  (h3 : A + B = 2 * M + 4)
  (h4 : A = B - 3)
  (h5 : K + B = M + 9) :
  K = 8 ∧ M = 5 ∧ B = 6 ∧ A = 3 :=
sorry

end age_problem_l12_12440


namespace monotonicity_and_tangent_intersection_l12_12244

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l12_12244


namespace problem_l12_12839

variable {U : Type} [universal_set U]

def ast (X Y : set U) : set U := compl (X ∩ Y)

theorem problem (X Y Z : set U) : ast (ast X Y) Z = (compl X ∩ compl Y) ∪ Z :=
by 
  sorry

end problem_l12_12839


namespace arithmetic_sequence_terms_before_neg20_l12_12416

theorem arithmetic_sequence_terms_before_neg20 :
  ∃ n : ℕ, (100 + (n - 1) * (-4) = -20) ∧ (n - 1 = 30) :=
by
  sorry

end arithmetic_sequence_terms_before_neg20_l12_12416


namespace four_digit_numbers_l12_12779

theorem four_digit_numbers (d_t d_h d_tens d_units : ℕ) :
  (1 <= d_t ∧ d_t <= 9) ∧ (0 <= d_h ∧ d_h < d_t) ∧ (0 <= d_tens ∧ d_tens <= 9) ∧ (0 <= d_units ∧ d_units <= 9) 
  → ∑ d_t in {1, 2, 3, 4, 5, 6, 7, 8, 9}, ∑ d_h in {0, 1, 2, 3, 4, 5, 6, 7, 8}.filter (λ d_h, d_h < d_t) 1 * 10 * 10 = 4500 := by
sorry

end four_digit_numbers_l12_12779


namespace fake_handbag_ratio_l12_12587

theorem fake_handbag_ratio (total_purses : ℕ) (total_handbags : ℕ) 
  (half_purses_fake : total_purses / 2) (authentic_items : ℕ) 
  (total_purses = 26) (total_handbags = 24) (authentic_items = 31) :
  (total_handbags - (authentic_items - (total_purses - half_purses_fake)) = 6) → 
  (6 / total_handbags = 1 / 4) :=
by
  sorry

end fake_handbag_ratio_l12_12587


namespace inequality_for_positive_numbers_l12_12881

theorem inequality_for_positive_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (ab / (a^2 + 3b^2) + bc / (b^2 + 3c^2) + ca / (c^2 + 3a^2)) ≤ 3 / 4 :=
by
  sorry

end inequality_for_positive_numbers_l12_12881


namespace ones_digit_sum_l12_12964

theorem ones_digit_sum (n : ℕ) (h : 2023 = 4 * (2023 / 4) + 3) : 
  (∑ k in Finset.range (2023 + 1), k^2023) % 10 = 6 :=
by
  sorry

end ones_digit_sum_l12_12964


namespace intersection_A_B_l12_12775

def A : Set ℕ := {70, 1946, 1997, 2003}
def B : Set ℕ := {1, 10, 70, 2016}

theorem intersection_A_B : A ∩ B = {70} := by
  sorry

end intersection_A_B_l12_12775


namespace monotonicity_f_tangent_points_l12_12362

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12362


namespace ratio_of_angles_l12_12822

theorem ratio_of_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 2 * B = A + C) (h3 : A < B ∧ B < C) :
  (A / ?m_1 = 1) ∧ (B / ?m_1 = 2) ∧ (C / ?m_1 = 3) :=
sorry

end ratio_of_angles_l12_12822


namespace valid_tiling_prob_l12_12680

noncomputable def a_n (n : ℕ) : ℤ :=
if n = 1 then 2
else if n = 2 then 6
else 2 * a_n(n-1) + 2 * a_n(n-2)

noncomputable def p_n (n : ℕ) : ℚ :=
(a_n n) * (1 / 4)^n

noncomputable def A (x : ℚ) : ℚ :=
(2 * x + 2 * x^2) / (1 - 2 * x - 2 * x^2)

theorem valid_tiling_prob : A (1 / 8) = 9 / 23 :=
sorry

end valid_tiling_prob_l12_12680


namespace sum_of_possible_values_l12_12208

variable {S : ℝ} (h : S ≠ 0)

theorem sum_of_possible_values (h : S ≠ 0) : ∃ N : ℝ, N ≠ 0 ∧ 6 * N + 2 / N = S → ∀ N1 N2 : ℝ, (6 * N1 + 2 / N1 = S ∧ 6 * N2 + 2 / N2 = S) → (N1 + N2) = S / 6 :=
by
  sorry

end sum_of_possible_values_l12_12208


namespace time_reduced_fraction_l12_12996

theorem time_reduced_fraction
  (T : ℝ)
  (V : ℝ)
  (hV : V = 42)
  (D : ℝ)
  (hD_1 : D = V * T)
  (V' : ℝ)
  (hV' : V' = V + 21)
  (T' : ℝ)
  (hD_2 : D = V' * T') :
  (T - T') / T = 1 / 3 :=
by
  -- Proof omitted
  sorry

end time_reduced_fraction_l12_12996


namespace value_range_of_f_l12_12582

noncomputable def f (x : ℝ) : ℝ := Real.log 2 (-x^2 + 2 * Real.sqrt 2)

theorem value_range_of_f 
  (h : ∀ x : ℝ, -x^2 + 2 * Real.sqrt 2 ∈ Set.Iic (2 * Real.sqrt 2)) :
  (Set.image f Set.univ) = Set.Iic (3 / 2) :=
by
  sorry

end value_range_of_f_l12_12582


namespace triangle_ABC_area_l12_12060

/-- Define points A, B, and C in the 2D plane. -/
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (8, 3)
def C : ℝ × ℝ := (6, -2)

/-- Define a function to calculate the area of a triangle given its vertices. -/
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- The proof statement. -/
theorem triangle_ABC_area : triangle_area A B C = 5 := by
  sorry

end triangle_ABC_area_l12_12060


namespace vendor_thrown_away_percentage_l12_12083

-- Define the initial number of apples, sales, and throw away percentages as conditions
def initial_apples : ℕ := 100
def first_day_sale_percentage : ℝ := 0.50
def first_day_throw_percentage : ℝ := 0.20
def second_day_sale_percentage : ℝ := 0.50

-- Calculate the thrown away amounts for the proof
noncomputable def first_day_thrown_away :=
  (initial_apples * (1 - first_day_sale_percentage) * first_day_throw_percentage).natAbs

noncomputable def second_day_thrown_away :=
  (initial_apples * (1 - first_day_sale_percentage) * (1 - first_day_throw_percentage) * (1 - second_day_sale_percentage)).natAbs

noncomputable def total_thrown_away :=
  first_day_thrown_away + second_day_thrown_away

noncomputable def thrown_away_percentage : ℝ :=
  (total_thrown_away : ℝ) / (initial_apples : ℝ) * 100

-- Prove that the total thrown away percentage is 30%
theorem vendor_thrown_away_percentage : 
  thrown_away_percentage = 30 :=
  by
  sorry

end vendor_thrown_away_percentage_l12_12083


namespace monotonicity_of_f_tangent_intersection_coordinates_l12_12387

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l12_12387


namespace cost_of_painting_cube_l12_12902

noncomputable def paint_cost : ℝ :=
let cost_per_kg := 36.50 in
let coverage_per_kg := 16 in
let num_faces := 6 in
let side_length := 8 in
let area_per_face := side_length * side_length in
let total_area := num_faces * area_per_face in
let paint_required := total_area / coverage_per_kg in
paint_required * cost_per_kg

theorem cost_of_painting_cube : paint_cost = 876 :=
by
  sorry

end cost_of_painting_cube_l12_12902


namespace range_of_2alpha_minus_beta_over_3_l12_12229

theorem range_of_2alpha_minus_beta_over_3 (α β : ℝ) (hα : 0 < α) (hα' : α < π / 2) (hβ : 0 < β) (hβ' : β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := 
sorry

end range_of_2alpha_minus_beta_over_3_l12_12229


namespace subset_a_values_l12_12774

theorem subset_a_values 
    (a : ℝ)
    (A : set ℝ)
    (B : set ℝ)
    (hA : A = {-1, 1})
    (hB : B = {x : ℝ | a * x + 2 = 0})
    (h_subset : B ⊆ A) :
    a ∈ {-2, 0, 2} :=
sorry

end subset_a_values_l12_12774


namespace monotonicity_of_f_tangent_line_intersection_points_l12_12313

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l12_12313


namespace f_96_l12_12917

noncomputable def f : ℕ → ℝ := sorry -- assume f is defined somewhere

axiom f_property (a b k : ℕ) (h : a + b = 3 * 2^k) : f a + f b = 2 * k^2

theorem f_96 : f 96 = 20 :=
by
  -- Here we should provide the proof, but for now we use sorry
  sorry

end f_96_l12_12917


namespace Jane_mom_jars_needed_l12_12477

theorem Jane_mom_jars_needed : 
  ∀ (total_tomatoes jar_capacity : ℕ), 
  total_tomatoes = 550 → 
  jar_capacity = 14 → 
  ⌈(total_tomatoes: ℚ) / jar_capacity⌉ = 40 := 
by 
  intros total_tomatoes jar_capacity htotal hcapacity
  sorry

end Jane_mom_jars_needed_l12_12477


namespace quaternary_to_decimal_l12_12952

theorem quaternary_to_decimal : 
  ∀ (d1 d2 d3 : ℕ), 
  d1 = 2 → d2 = 0 → d3 = 1 →
  (d1 * 4^2 + d2 * 4^1 + d3 * 4^0) = 33 :=
by
  intros d1 d2 d3 h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end quaternary_to_decimal_l12_12952


namespace triangle_quadratic_solutions_l12_12739

theorem triangle_quadratic_solutions (A B C : ℝ) (hA: 0 < A) (hB: 0 < B) (hC: 0 < C)
    (h_sum: A + B + C = π)
    (h_acute: A < π / 2 ∧ B < π / 2 ∧ C < π / 2) :
  ∀ x : ℝ,
    (x^2 * sin B^2 + x * (cos C^2 - cos A^2 + sin B^2) + sin A^2 * cos C^2 = 0) →
    (x = 0) ∨ (x = -(sin A * cos C / sin B)) :=
by
  intros x h
  sorry

end triangle_quadratic_solutions_l12_12739


namespace zhang_shan_sales_prediction_l12_12080

theorem zhang_shan_sales_prediction (x : ℝ) (y : ℝ) (h : x = 34) (reg_eq : y = 2 * x + 60) : y = 128 :=
by
  sorry

end zhang_shan_sales_prediction_l12_12080


namespace initial_tape_amount_l12_12828

structure Field :=
  (width : ℝ)
  (length : ℝ)

def tape_leftover : ℝ := 90
def joy_field : Field := { width := 20, length := 60 }

def perimeter (F : Field) : ℝ := 2 * (F.width + F.length)

def initial_tape_length (F : Field) (leftover : ℝ) : ℝ := (perimeter F) + leftover

theorem initial_tape_amount : initial_tape_length joy_field tape_leftover = 250 := by
  sorry

end initial_tape_amount_l12_12828


namespace monotonicity_f_tangent_points_l12_12368

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12368


namespace average_entry_exit_time_is_correct_l12_12631

-- Define the conditions for car and storm positions at time t
def car_position (t : ℝ) : ℝ × ℝ := (0.75 * t, 0)
def storm_center_position (t : ℝ) : ℝ × ℝ := (0, 120 - 0.5 * t)
def storm_radius : ℝ := 60

-- Defining the distance between car and storm center at time t
def distance_square (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

-- Condition that the car is within the storm circle
def car_in_storm (t : ℝ) : Prop :=
  distance_square (car_position t) (storm_center_position t) ≤ storm_radius ^ 2

-- Proving the average entry/exit time
noncomputable def average_time : ℝ :=
  -(-(120 : ℝ)) / (2 * 0.8125)

theorem average_entry_exit_time_is_correct :
  average_time ≈ 73.846 := by
  sorry

end average_entry_exit_time_is_correct_l12_12631


namespace rotated_graph_eqn_l12_12911

theorem rotated_graph_eqn (x : ℝ) : (∃ y : ℝ, y = exp x) → (∃ y : ℝ, y = log (-x)) :=
by
  intro h
  cases h with y hy
  use log (-x)
  sorry

end rotated_graph_eqn_l12_12911


namespace product_negative_probability_l12_12051

def set := {-7, -3, -1, 2, 5, 8}

def count_negatives (s : Set ℤ) : ℕ := 
    s.count (λ x, x < 0)

def count_positives (s : Set ℤ) : ℕ := 
    s.count (λ x, x > 0)

def choose (n k : ℕ) : ℕ := 
    nat.choose n k

def total_ways : ℕ := 
    choose (set.card) 3

def favorable_ways : ℕ := 
    (choose (count_negatives set) 1) * (choose (count_positives set) 2) + 
    (choose (count_negatives set) 3)

noncomputable def probability : ℚ := 
    favorable_ways / total_ways

theorem product_negative_probability : 
    probability = 1 / 2 :=
by
    sorry

end product_negative_probability_l12_12051


namespace maximum_value_m_l12_12402

-- Define the function
def f (x : ℝ) (m : ℝ) : ℝ := x^4 - (1 / 3) * m * x^3 + (1 / 2) * x^2 + 1

-- Define the derivative of the function
def f' (x : ℝ) (m : ℝ) : ℝ := 4 * x^3 - m * x^2 + x

-- Define the condition for monotonicity on (0, 1)
def monotonic_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x, 0 < x ∧ x < 1 → f' x m ≥ 0

-- The main theorem stating the maximum value of m
theorem maximum_value_m : ∀ m : ℝ, monotonic_increasing_on_interval m → m ≤ 4 := 
sorry

end maximum_value_m_l12_12402


namespace factory_output_restoration_l12_12619

theorem factory_output_restoration (O : ℝ) :
  let new_output := O * 1.10 * 1.60 in
  let decrease_needed := new_output - O in
  let percentage_decrease := (decrease_needed / new_output) * 100 in
  percentage_decrease ≈ 43.18 :=
by
  sorry

end factory_output_restoration_l12_12619


namespace count_b_ensuring_multiple_of_7_l12_12849

noncomputable def e1 (b : ℕ) := b^3 + 3^b + b * 3^((b + 1) / 2)
noncomputable def e2 (b : ℕ) := b^3 + 3^b - b * 3^((b + 1) / 2)

theorem count_b_ensuring_multiple_of_7 :
  (finset.range 501).filter (λ b, 1 ≤ b ∧ ((e1 b) * (e2 b)) % 7 = 0).card = 71 :=
by sorry

end count_b_ensuring_multiple_of_7_l12_12849


namespace minimum_disks_needed_l12_12475

-- Given conditions:
def total_files : Nat := 35
def disk_capacity : Float := 2.0
def file_size_large : Float := 0.95
def file_size_medium : Float := 0.85
def file_size_small : Float := 0.5
def large_files : Nat := 5
def medium_files : Nat := 15
def small_files := total_files - (large_files + medium_files)
def remaining_files (paired_files : Nat) := total_files - paired_files
def capacity_used (used_disks : Nat) := (2.0 * Float.ofNat used_disks)

-- The minimum number of disks needed to store all files:
theorem minimum_disks_needed : ∃ (n : Nat), n = 14 :=
begin
  sorry
end

end minimum_disks_needed_l12_12475


namespace change_is_five_l12_12115

noncomputable def haircut_cost := 15
noncomputable def payment := 20
noncomputable def counterfeit := 20
noncomputable def exchanged_amount := (10 : ℤ) + 10
noncomputable def flower_shop_amount := 20

def change_given (payment haircut_cost: ℕ) : ℤ :=
payment - haircut_cost

theorem change_is_five : 
  change_given payment haircut_cost = 5 :=
by 
  sorry

end change_is_five_l12_12115


namespace monotonicity_f_tangent_points_l12_12371

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l12_12371
