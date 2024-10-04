import Data.Nat.Basic
import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.AddEquiv.Basic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.CharZero
import Mathlib.Algebra.Equiv
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Trig.Functions
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry.Eq
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Circulant
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.SinCos
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.GraphTheory.Basic
import Mathlib.Init.Algebra.Order
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ConditionalProbability
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Polynomial
import Real

namespace yellow_marbles_count_l225_225473

theorem yellow_marbles_count 
  (total_marbles red_marbles blue_marbles : ℕ) 
  (h_total : total_marbles = 85) 
  (h_red : red_marbles = 14) 
  (h_blue : blue_marbles = 3 * red_marbles) :
  (total_marbles - (red_marbles + blue_marbles)) = 29 :=
by
  sorry

end yellow_marbles_count_l225_225473


namespace division_of_repeating_decimals_l225_225219

noncomputable def repeating_to_fraction (r : ℚ) : ℚ := 
  if r == 0.36 then 4 / 11 
  else if r == 0.12 then 4 / 33 
  else 0

theorem division_of_repeating_decimals :
  (repeating_to_fraction 0.36) / (repeating_to_fraction 0.12) = 3 :=
by
  sorry

end division_of_repeating_decimals_l225_225219


namespace num_subsets_at_least_4_adj_l225_225701

def num_adjacent_subsets (n : ℕ) (k : ℕ) : ℕ :=
  if k = 8 then 1 else if 4 ≤ k ∧ k < 8 then 8 else 0

theorem num_subsets_at_least_4_adj (n : ℕ) (h : n = 8) : 
  (∑ k in finset.Icc 4 8, num_adjacent_subsets n k) = 33 := by sorry

end num_subsets_at_least_4_adj_l225_225701


namespace decagon_triangle_probability_l225_225354

theorem decagon_triangle_probability :
  let vertices := (fin 10)  -- 10 vertices of a decagon
  let total_triangles := (vertices.choose 3)  -- Choose 3 vertices to form a triangle
  let favorable_outcomes := 60  -- From the conditions, we get 50 for one side and 10 for two sides (60 in total)
  let total_outcomes := 120  -- Total ways to choose 3 vertices out of 10
  probability := (favorable_outcomes / total_outcomes) = 1 / 2 :=
  by sorry

end decagon_triangle_probability_l225_225354


namespace domain_of_f_l225_225404

def f (x : ℝ) : ℝ := Real.log (1 - x) + x ^ (-1 / 2)

theorem domain_of_f :
  {x : ℝ | 0 < x ∧ x < 1} = {x : ℝ | x ∈ set.Ioo 0 1} := by
  sorry

end domain_of_f_l225_225404


namespace circumscribed_circle_double_inscribed_l225_225807

theorem circumscribed_circle_double_inscribed 
  (a : ℝ) : 
  let r_inscribed := a,
      r_circumscribed := a * Real.sqrt 2 in 
  (π * (r_circumscribed^2) = 2 * π * (r_inscribed^2)) :=
by
  let r_inscribed := a
  let r_circumscribed := a * Real.sqrt 2
  sorry

end circumscribed_circle_double_inscribed_l225_225807


namespace abc_is_772_l225_225802

noncomputable def find_abc (a b c : ℝ) : ℝ :=
if h₁ : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * (b + c) = 160 ∧ b * (c + a) = 168 ∧ c * (a + b) = 180
then 772 else 0

theorem abc_is_772 (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
(h₄ : a * (b + c) = 160) (h₅ : b * (c + a) = 168) (h₆ : c * (a + b) = 180) :
  find_abc a b c = 772 := by
  sorry

end abc_is_772_l225_225802


namespace monic_quadratic_with_root_l225_225743

theorem monic_quadratic_with_root (p : ℝ[X]) (h1 : p.monic) (h2 : p.eval (2 - 3 * Complex.I) = 0) : 
  p = X^2 - 4 * X + 13 := 
by 
  sorry

end monic_quadratic_with_root_l225_225743


namespace layers_tall_l225_225517

def total_cards (n_d c_d : ℕ) : ℕ := n_d * c_d
def layers (total c_l : ℕ) : ℕ := total / c_l

theorem layers_tall (n_d c_d c_l : ℕ) (hn_d : n_d = 16) (hc_d : c_d = 52) (hc_l : c_l = 26) : 
  layers (total_cards n_d c_d) c_l = 32 := by
  sorry

end layers_tall_l225_225517


namespace maximize_probability_l225_225640

noncomputable def probability_function (x : ℝ) : ℝ :=
  120 * x / ((x + 30) * (x + 120))

theorem maximize_probability :
  let x := 60 in
  probability_function x = 4 / 9 :=
by
  let x := 60
  have h1: x > 0 := by norm_num
  show probability_function x = 4 / 9
  sorry

end maximize_probability_l225_225640


namespace problem_solve_l225_225506

variable {y : Fin 50 → ℝ} (h1 : ∑ i, y i = 2) (h2 : ∑ i, y i / (2 - y i) = 2)

theorem problem_solve :
  (∑ i, y i ^ 2 / (2 - y i)) = 0 :=
by
  sorry

end problem_solve_l225_225506


namespace cubic_sum_solution_l225_225546

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem cubic_sum_solution :
  let α := cube_root 17
  let β := cube_root 67
  let γ := cube_root 137
  ∃ u v w : ℝ, 
  (u - α) * (u - β) * (u - γ) = 2 / 5 ∧ 
  (v - α) * (v - β) * (v - γ) = 2 / 5 ∧ 
  (w - α) * (w - β) * (w - γ) = 2 / 5 ∧ 
  (u ≠ v ∧ u ≠ w ∧ v ≠ w) → 
  u^3 + v^3 + w^3 = 221 + 6/5 - 3 * 17 * 67 * 137 :=
begin
  -- Proof omitted
  sorry
end

end cubic_sum_solution_l225_225546


namespace number_of_zeros_of_f_l225_225787

noncomputable def f (x : ℝ) : ℝ := 2^x - 3*x

theorem number_of_zeros_of_f : ∃ a b : ℝ, (f a = 0 ∧ f b = 0 ∧ a ≠ b) ∧ ∀ x : ℝ, f x = 0 → x = a ∨ x = b :=
sorry

end number_of_zeros_of_f_l225_225787


namespace recurring_division_l225_225226

def recurring_36_as_fraction : ℚ := 36 / 99
def recurring_12_as_fraction : ℚ := 12 / 99

theorem recurring_division :
  recurring_36_as_fraction / recurring_12_as_fraction = 3 := 
sorry

end recurring_division_l225_225226


namespace orthogonal_vectors_l225_225332

theorem orthogonal_vectors (x : ℝ) :
  (3 * x - 4 * 6 = 0) → x = 8 :=
by
  intro h
  sorry

end orthogonal_vectors_l225_225332


namespace function_form_l225_225580

theorem function_form (f : ℝ → ℝ) (b : ℝ) 
  (h1 : ∀ x, f(3 * x) = 3 * x^2 + b) 
  (h2 : f 1 = 0) : 
  f = fun x => (1 / 3) * x^2 - (1 / 3) := 
by sorry

end function_form_l225_225580


namespace solve_sqrt_equation_l225_225748

theorem solve_sqrt_equation (p : ℝ) (hp : 0 < p) :
  (|p| < 1/2 ∧ ∃ x : ℝ, x = √((p^2 + 1/4) / (1 - 4p^2))
    ∨ x = -√((p^2 + 1/4) / (1 - 4p^2))
    ∧ (√(x^2 + 2 * p * x - p^2) - √(x^2 - 2 * p * x - p^2) = 1))
  ∨ (|p| ≥ 1/2 ∧ ¬∃ x : ℝ, (√(x^2 + 2 * p * x - p^2) - √(x^2 - 2 * p * x - p^2) = 1)) :=
sorry

end solve_sqrt_equation_l225_225748


namespace parabola_focus_segment_length_l225_225808

theorem parabola_focus_segment_length (a : ℝ) (h₀ : a > 0) 
  (h₁ : ∀ x, abs x * abs (1 / a) = 4) : a = 1/4 := 
sorry

end parabola_focus_segment_length_l225_225808


namespace find_angle_C_find_area_of_triangle_l225_225087

-- Part (1)
theorem find_angle_C (A B C : ℝ) (a b c : ℝ) (h1 : b * sin B - a * sin A = sqrt 3 * a * cos A - sqrt 3 * b * cos B) (h2 : a ≠ b) :
  C = π / 3 := 
sorry

-- Part (2)
theorem find_area_of_triangle (A B C : ℝ) (a b c : ℝ) (h1 : b * sin B - a * sin A = sqrt 3 * a * cos A - sqrt 3 * b * cos B) (h2 : a ≠ b) (h3 : c = 2 * sqrt 2) (h4 : median_C_to_D := 2) :
  let area := 0.5 * a * b * sin C
  area = sqrt 3 :=
sorry

end find_angle_C_find_area_of_triangle_l225_225087


namespace no_common_terms_except_one_l225_225686

-- Definitions of the sequences x_n and y_n
def x_seq : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n + 2) := x_seq (n + 1) + 2 * x_seq n

def y_seq : ℕ → ℕ
| 0 := 1
| 1 := 7
| (n + 2) := 2 * y_seq (n + 1) + 3 * y_seq n

-- Proof statement:
theorem no_common_terms_except_one :
  ∀ n m, (x_seq n = y_seq m) → (x_seq n = 1) :=
by
  sorry

end no_common_terms_except_one_l225_225686


namespace sequence_properties_l225_225457

def seq_a : ℕ → ℚ
| 1     := 1
| (n+1) := 2 * seq_a n / (2 + seq_a n)

theorem sequence_properties :
  seq_a 2 = 2 / 3 ∧
  seq_a 3 = 1 / 2 ∧
  seq_a 4 = 2 / 5 ∧
  (∀ n : ℕ, n ≥ 1 → seq_a n = 2 / (n + 1)) ∧
  (∀ n : ℕ, n ≥ 1 → (let b_n := seq_a n / n in b_n = 2 / (n * (n + 1)))) ∧
  (∀ n : ℕ, n ≥ 1 → (let S_n := ∑ i in finset.range n.succ, seq_a (i + 1) / (i + 1) in S_n = 2 * n / (n + 1))) :=
begin
  sorry
end

end sequence_properties_l225_225457


namespace concyclic_points_l225_225673

theorem concyclic_points 
  (ABCD : Quadrilateral)
  (H_inscribed : Inscribed ABCD)
  (H_circles : ∀ (A B C D : Point), ∃ (Circle_A B Circle_B C Circle_C D Circle_D A : Circle), 
                InCircle A B C D Circle_A B ∧ InCircle B C D A Circle_B C ∧ InCircle C D A B Circle_C D ∧ InCircle D A B C Circle_D A) :
  ∃ (C : Circle), OnCircle A_1 B_1 C_1 D_1 C :=
sorry

end concyclic_points_l225_225673


namespace sequence_a_n_formula_smallest_n_for_T_n_l225_225366

theorem sequence_a_n_formula (n : ℕ) (S : ℕ → ℕ) 
  (hS : ∀ k, S k = 3 * k^2 - 2 * k) : 
  (a : ℕ → ℕ) (∀ k, a k = S k - S (k - 1))  → a n = 6 * n - 5 := 
by
  sorry

theorem smallest_n_for_T_n (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (a : ℕ → ℕ)
  (hS : ∀ k,  S k = 3 * k^2 - 2 * k) 
  (ha : ∀ k, a k = S k - S (k - 1))
  (hb : ∀ k, b k = 3 / (a k * a (k + 1)))
  (hT : ∀ k, T k = ∑ i in finset.range k, b i):
  ∃ n, |T n - 1 / 2| < 1 / 100 ∧ n = 9 :=
by 
  sorry

end sequence_a_n_formula_smallest_n_for_T_n_l225_225366


namespace recurring_fraction_division_l225_225224

/--
Given x = 0.\overline{36} and y = 0.\overline{12}, prove that x / y = 3.
-/
theorem recurring_fraction_division 
  (x y : ℝ)
  (h1 : x = 0.36 + 0.0036 + 0.000036 + 0.00000036 + ......) -- representation of 0.\overline{36}
  (h2 : y = 0.12 + 0.0012 + 0.000012 + 0.00000012 + ......) -- representation of 0.\overline{12}
  : x / y = 3 :=
  sorry

end recurring_fraction_division_l225_225224


namespace subset_exists_l225_225752

theorem subset_exists (n : ℕ) (A B : finset ℕ) (hA : A ⊆ finset.range n) (hB : B ⊆ finset.range n) (hAn : A.nonempty) (hBn : B.nonempty) :
  ∃ D ⊆ A + B, (D.card ≥ (A.card * B.card) / (2 * n)) ∧ ((D + D) ⊆ (2 : set ℕ) • (A + B)) :=
sorry

end subset_exists_l225_225752


namespace weight_of_other_grains_weight_of_other_grains_proved_l225_225922

-- Define all necessary conditions
variables (x w : ℝ) (s_rice s_other : ℝ)
axiom total_weight : w = 1534
axiom sample_rice_grains : s_rice = 254
axiom sample_other_grains : s_other = 56

-- Theorem statement
theorem weight_of_other_grains : 
  x = 338 ↔ (x / w = s_other / s_rice) :=
by
  -- Applying given conditions
  have hw : w = 1534 := total_weight
  have hsr : s_rice = 254 := sample_rice_grains
  have hso : s_other = 56 := sample_other_grains

  sorry

-- Prove that given the conditions, the weight of other grains x is 338
theorem weight_of_other_grains_proved : 
  x = 338 :=
by
  apply weight_of_other_grains.mpr
  simp only [hw, hsr, hso]
  field_simp
  norm_cast
  linarith

end weight_of_other_grains_weight_of_other_grains_proved_l225_225922


namespace isosceles_right_triangle_solution_l225_225608

theorem isosceles_right_triangle_solution (a b : ℝ) (area : ℝ) 
  (h1 : a = b) (h2 : XY = a * Real.sqrt 2) (h3 : area = (1/2) * a * b) (h4 : area = 36) : 
  XY = 12 :=
by
  sorry

end isosceles_right_triangle_solution_l225_225608


namespace slope_range_of_line_intersecting_hyperbola_l225_225053

open Set Real

-- Definition of the hyperbola and its properties.
def is_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 16) - (y^2 / 9) = 1

-- Definitions for line passing through a point (focus in this case).
def line_through_point (k : ℝ) (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - x₀) + y₀

-- Left focus F1 of the hyperbola.
def F1 : ℝ × ℝ := (-4, 0)

-- The main theorem to be proven.
theorem slope_range_of_line_intersecting_hyperbola {A B : ℝ × ℝ} (k : ℝ) :
  ((∃ x y, is_hyperbola x y ∧ line_through_point k F1.1 F1.2 x y) →
   ((k ∈ Icc (-∞) (-3/4)) ∨ (k ∈ Icc (3/4) ∞)) → False) :=
by
  sorry

end slope_range_of_line_intersecting_hyperbola_l225_225053


namespace p_divisible_by_2003_l225_225864

theorem p_divisible_by_2003
  (p q : ℤ) (h_coprime : Int.gcd p q = 1)
  (h_fraction : (p : ℚ) / q = 1 - (1/2 : ℚ) + (1/3 : ℚ) - (1/4 : ℚ) + ... - (1/1334 : ℚ) + (1/1335 : ℚ))
  : 2003 ∣ p :=
sorry

end p_divisible_by_2003_l225_225864


namespace satisfies_diff_eq_l225_225537

def y (x : ℝ) : ℝ := -x * cos x + 3 * x

def y_prime (x : ℝ) : ℝ := -cos x + x * sin x + 3

theorem satisfies_diff_eq {x : ℝ} :
  x * y_prime x = y x + x^2 * sin x := 
by
  sorry

end satisfies_diff_eq_l225_225537


namespace point_C_divides_AE_l225_225030

theorem point_C_divides_AE
  (A B C D E : Type)
  [triangle_right : is_right_triangle A B C]
  [point_D_on_extension_BC : is_point_on_extension D B C]
  [tangent_AD_to_circumcircle_omega_ABC : tangent AD (circumcircle.triangle A B C)]
  [intersection_AC_circumcircle_ABD : intersects AC (circumcircle.triangle A B D) E]
  [tangent_bisector_AD_to_omega : tangent (angle_bisector ADE) (circumcircle.triangle A B C)] :
  divides_in_ratio C A E 1 2 :=
sorry

end point_C_divides_AE_l225_225030


namespace monic_quadratic_with_roots_l225_225732

-- Define the given conditions and roots
def root1 : ℂ := 2 - 3 * complex.i
def root2 : ℂ := 2 + 3 * complex.i

-- Define the polynomial with these roots
def poly (x : ℂ) : ℂ := x^2 - 4 * x + 13

-- Define the proof problem
theorem monic_quadratic_with_roots 
  (h1 : poly root1 = 0) 
  (h2 : poly root2 = 0) 
  (h_monic : ∀ x : ℂ, ∃! p : polynomial ℂ, p.monic ∧ p.coeff 2 = 1) : 
  ∃ p : polynomial ℝ, p.monic ∧ ∀ x : ℂ, p.eval x = poly x := 
sorry

end monic_quadratic_with_roots_l225_225732


namespace a10_b10_l225_225139

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem a10_b10 : a^10 + b^10 = 123 :=
by
  sorry

end a10_b10_l225_225139


namespace circumradius_eq_exradius_opposite_side_BC_l225_225387

noncomputable def R : ℝ := sorry -- circumradius
noncomputable def r_a : ℝ := sorry -- exradius opposite side BC
def circumcircle_incircle (O I : Type*) (C_incenter C_circumcenter : ℝ) (r_a r : ℝ) :=
  C_incenter = r_a -> C_circumcenter = r

variables {A B C : Type*} (O I : Type*) (AD : Line) (OD : Segment O I)
  (C_circumcenter C_incenter: ℝ) [circumcenter_def : O = circumcircle_incircle]
  [incenter_def : I = circumcircle_incircle]

theorem circumradius_eq_exradius_opposite_side_BC 
  (h1: O = circumcenter O I ∆ABC) 
  (h2: I = incenter O I ∆ABC)
  (h3: altitude_passes_through_incenter AD O) 
  (h4: I ∈ segment O AD) :
  R = r_a := by sorry

end circumradius_eq_exradius_opposite_side_BC_l225_225387


namespace monotonicity_and_range_l225_225050

noncomputable def f (a x : ℝ) := exp (2 * x) - a * exp x - a^2 * x

theorem monotonicity_and_range (a : ℝ) :
  (a > 0 → (∀ x, x > real.log a → (deriv (f a) x ≥ 0)) ∧ (∀ x, x < real.log a → (deriv (f a) x ≤ 0))) ∧
  (a = 0 → ∀ x, deriv (f a) x > 0) ∧
  (a < 0 → (∀ x, x > real.log (- a / 2) → (deriv (f a) x ≥ 0)) ∧ (∀ x, x < real.log (- a / 2) → (deriv (f a) x ≤ 0))) ∧
  ((∀ x, f a x ≥ 0) → (-2 * real.exp(3 / 4) ≤ a ∧ a ≤ 1)) :=
  by sorry

end monotonicity_and_range_l225_225050


namespace line_of_symmetry_is_x_eq_0_l225_225438

variable (f : ℝ → ℝ)

theorem line_of_symmetry_is_x_eq_0 :
  (∀ y, f (10 + y) = f (10 - y)) → ( ∃ l, l = 0 ∧ ∀ x,  f (10 + l + x) = f (10 + l - x)) := 
by
  sorry

end line_of_symmetry_is_x_eq_0_l225_225438


namespace Sara_spent_on_hotdog_l225_225913

-- Define the given constants
def totalCost : ℝ := 10.46
def costSalad : ℝ := 5.10

-- Define the value we need to prove
def costHotdog : ℝ := 5.36

-- Statement to prove
theorem Sara_spent_on_hotdog : totalCost - costSalad = costHotdog := by
  sorry

end Sara_spent_on_hotdog_l225_225913


namespace seating_arrangement_l225_225713

variable {M I P A : Prop}

def first_fact : ¬ M := sorry
def second_fact : ¬ A := sorry
def third_fact : ¬ M → I := sorry
def fourth_fact : I → P := sorry

theorem seating_arrangement : ¬ M → (I ∧ P) :=
by
  intros hM
  have hI : I := third_fact hM
  have hP : P := fourth_fact hI
  exact ⟨hI, hP⟩

end seating_arrangement_l225_225713


namespace problem1_l225_225681

theorem problem1 : sqrt 9 + cbrt (-1:ℝ) - sqrt 0 + sqrt (1 / 4) = 5 / 2 :=
by
  sorry

end problem1_l225_225681


namespace min_inequality_l225_225861

variable (A B C P Q R K : Point)
variable (Γ : Circle)
variable (circumcircle_ABC : Circumcircle)
variable (hexagon_on_circumcircle : ConvexHexagon Γ [A, R, B, P, C, Q])
variable (A_1 B_1 C_1 : Point)
variable (line_AP : Line)
variable (line_BQ : Line)
variable (line_CR : Line)

variable (circle_tangent_to_Γ_at_A : Circle)
variable (circle_tangent_to_Γ_at_B : Circle)
variable (circle_tangent_to_Γ_at_C : Circle)

variable (A1_property : SecondIntersectionPoint circle_tangent_to_Γ_at_A line_AP A_1)
variable (B1_property : SecondIntersectionPoint circle_tangent_to_Γ_at_B line_BQ B_1)
variable (C1_property : SecondIntersectionPoint circle_tangent_to_Γ_at_C line_CR C_1)

theorem min_inequality :
  min (PA_1 / AA_1, min (QB_1 / BB_1, RC_1 / CC_1)) ≤ 1 :=
by
  sorry

end min_inequality_l225_225861


namespace average_speed_is_approximately_l225_225251

/-- Define the distance between points P and Q as D -/
variables (D : ℝ)

/-- Define the speed from P to Q as 30 km/hr -/
def speed_PQ : ℝ := 30

/-- Define the increased return speed from Q to P as 30% more than 30 km/hr -/
def speed_QP : ℝ := 1.3 * speed_PQ

/-- Define the time taken to travel from P to Q -/
def time_PQ : ℝ := D / speed_PQ

/-- Define the time taken to travel from Q to P -/
def time_QP : ℝ := D / speed_QP

/-- Calculate the total distance for the round trip -/
def total_distance : ℝ := 2 * D

/-- Calculate the total time for the round trip -/
def total_time : ℝ := time_PQ D + time_QP D

/-- Define the average speed for the round trip -/
def average_speed : ℝ := total_distance D / total_time D

theorem average_speed_is_approximately :
  average_speed D ≈ 33.91 :=
by
  sorry

end average_speed_is_approximately_l225_225251


namespace find_f_prime_at_1_l225_225390

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x * (deriv f 1) - 1
def f_prime (x : ℝ) : ℝ := deriv f x

theorem find_f_prime_at_1 : f_prime 1 = -3 := by
  sorry

end find_f_prime_at_1_l225_225390


namespace sequence_general_term_l225_225932

theorem sequence_general_term (a : ℕ → ℤ)
  (h₀ : a 1 = -1)
  (h₁ : a 2 = 3)
  (h₂ : a 3 = -7)
  (h₃ : a 4 = 15) :
  a = λ n, (-1)^n * (2^n - 1) :=
by
  sorry

end sequence_general_term_l225_225932


namespace irrational_implies_irrational_l225_225596

-- Define irrational number proposition
def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

-- Define the main proposition to prove
theorem irrational_implies_irrational (a : ℝ) : is_irrational (a - 2) → is_irrational a :=
by
  sorry

end irrational_implies_irrational_l225_225596


namespace correct_transformation_l225_225239

structure Point :=
  (x : ℝ)
  (y : ℝ)

def rotate180 (p : Point) : Point :=
  Point.mk (-p.x) (-p.y)

def is_rotation_180 (p p' : Point) : Prop :=
  rotate180 p = p'

theorem correct_transformation (C D : Point) (C' D' : Point) 
  (hC : C = Point.mk 3 (-2)) 
  (hC' : C' = Point.mk (-3) 2)
  (hD : D = Point.mk 2 (-5)) 
  (hD' : D' = Point.mk (-2) 5) :
  is_rotation_180 C C' ∧ is_rotation_180 D D' :=
by
  sorry

end correct_transformation_l225_225239


namespace product_negative_probability_l225_225215

noncomputable def prob_product_negative : ℚ :=
  let mySet := {-5, -8, 7, 4, -2} : finset ℤ
  let pairs := mySet.powerset.filter (λ s, s.card = 2)
  let negative_product_pairs := pairs.filter (λ s, s.prod id < 0)
  negative_product_pairs.card / pairs.card

theorem product_negative_probability : prob_product_negative = 3 / 5 := by
  sorry

end product_negative_probability_l225_225215


namespace garden_width_is_14_l225_225551

theorem garden_width_is_14 (w : ℝ) (h1 : ∃ (l : ℝ), l = 3 * w ∧ l * w = 588) : w = 14 :=
sorry

end garden_width_is_14_l225_225551


namespace integer_pairs_count_l225_225014

theorem integer_pairs_count :
  { (x : ℤ) × (y : ℤ) | 2 * x ≥ 3 * y ∧ 3 * x ≥ 4 * y ∧ 5 * x - 7 * y ≤ 20 }.card = 231 :=
by
  sorry

end integer_pairs_count_l225_225014


namespace problem_l225_225280

open Real

noncomputable def f (x : ℝ) : ℝ := log x / log 2

theorem problem (f : ℝ → ℝ) (h : ∀ (x y : ℝ), f (x * y) = f x + f y) : 
  (∀ x : ℝ, f x = log x / log 2) :=
sorry

end problem_l225_225280


namespace find_m_l225_225794

noncomputable theory

variables {m : ℝ}

def vector_a : ℝ × ℝ := (-2, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)

def parallel_vectors (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem find_m (m : ℝ) (h_parallel : parallel_vectors (vector_a - (2 : ℝ) • vector_b m) (vector_b m)) :
  m = - (2 / 3) :=
  sorry

end find_m_l225_225794


namespace cone_volume_l225_225997

theorem cone_volume (S L : ℝ) (hS : S > 0) (hL : L > 0) : 
  ∃ V : ℝ, V = S * L :=
by {
  use S * L,
  exact rfl,
}

end cone_volume_l225_225997


namespace complex_number_imaginary_l225_225075

theorem complex_number_imaginary (m : ℝ) (h : (m + 1 : ℂ).re + (m - 1 : ℂ).im * complex.I = m + 1 + (m - 1) * complex.I) : m ≠ 1 :=
by {
  -- The complex number is imaginary implies its real part is 0
  have h_real : (m + 1 : ℂ).re = 0,
  sorry,
  -- This implies m + 1 = 0
  have h_m_eq_zero : m + 1 = 0,
  sorry,
  -- Therefore, m is not equal to 1.
  exact ne_of_eq_of_ne h_m_eq_zero.symm (sorry)
}

end complex_number_imaginary_l225_225075


namespace not_possible_expression_l225_225127

open Nat

noncomputable def expression (p_i p_i1 p_i2 : ℕ) : ℕ :=
  (p_i * p_i1 - p_i2^2) / (p_i + p_i1)

theorem not_possible_expression (p : ℕ → ℕ) (hp : ∀ n, Prime (p n)) :
  ¬(∀ i, Natural (expression (p i) (p (i + 1)) (p (i + 2)))) :=
sorry

end not_possible_expression_l225_225127


namespace find_function_expression_and_extrema_l225_225401

theorem find_function_expression_and_extrema 
  (A ω φ : ℝ)
  (hA_pos : A > 0) 
  (hω_pos : ω > 0) 
  (hφ_range : 0 < φ ∧ φ < π / 2)
  (h_period : ∀ x, ℝ, f(x) = f(x + π))
  (h_lowest_point : f(2 * π / 3) = -2) :
  (f(x) = 2 * sin (2 * x + π / 6)) ∧ 
  (∀ x ∈ Icc 0 (π / 12), 
    1 ≤ f(x) ∧ f(x) ≤ sqrt 3 ) :=
sorry

end find_function_expression_and_extrema_l225_225401


namespace two_digit_num_square_ends_in_self_l225_225322

theorem two_digit_num_square_ends_in_self {x : ℕ} (hx : 10 ≤ x ∧ x < 100) (hx0 : x % 10 ≠ 0) : 
  (x * x % 100 = x) ↔ (x = 25 ∨ x = 76) :=
sorry

end two_digit_num_square_ends_in_self_l225_225322


namespace find_f1_plus_gneg1_l225_225379

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom relation : ∀ x : ℝ, f x - g x = (1 / 2) ^ x

-- Proof statement
theorem find_f1_plus_gneg1 : f 1 + g (-1) = -2 :=
by
  -- Proof goes here
  sorry

end find_f1_plus_gneg1_l225_225379


namespace movie_tickets_l225_225187

theorem movie_tickets (r h : ℕ) (h1 : r = 25) (h2 : h = 3 * r + 18) : h = 93 :=
by
  sorry

end movie_tickets_l225_225187


namespace terminating_fraction_count_l225_225348

theorem terminating_fraction_count :
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 1500 ∧ ∀ n, decimal_repr (n / 2800) terminates) → count_multiples 7 1500 = 214 :=
by
  sorry

end terminating_fraction_count_l225_225348


namespace proof_xt_simplified_l225_225264

noncomputable def xt_simplified_sum : ℚ :=
  let h := 20 / (1 + (real.sqrt 15)^(1/3)) in
  let xt := 3 * h / 2 in
  let m := 30 in 
  let n := 1 in
  m + n

theorem proof_xt_simplified : Xt_simplified_sum = 30 :=
by {
  sorry
}

end proof_xt_simplified_l225_225264


namespace sum_range_l225_225871

noncomputable def f (x : ℝ) : ℝ :=
sorry

axiom f_prop : ∀ x y : ℝ, f x * f y = f (x + y)
axiom f_nonzero : ∀ x : ℝ, f x ≠ 0

def a (n : ℕ) : ℝ :=
if n = 0 then 0 else f n

def S_n (n : ℕ) : ℝ := if n = 0 then 0 else (1 - (1 / 2) ^ n)

theorem sum_range :
  ∀ n : ℕ, n ≥ 1 → (S_n n ∈ set.Ico (1 / 2) 1) :=
begin
  sorry
end

end sum_range_l225_225871


namespace tan_sum_l225_225426

theorem tan_sum (x y : ℝ)
  (h1 : Real.sin x + Real.sin y = 72 / 65)
  (h2 : Real.cos x + Real.cos y = 96 / 65) : 
  Real.tan x + Real.tan y = 868 / 112 := 
by sorry

end tan_sum_l225_225426


namespace find_lambda_in_interval_l225_225064

noncomputable def vecA (x : ℝ) := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))

noncomputable def vecB (x : ℝ) := (Real.cos (x / 2), -Real.sin (x / 2))

noncomputable def dotProduct (a : ℝ × ℝ) (b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (a : ℝ × ℝ) := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)

theorem find_lambda_in_interval (x : ℝ) (λ : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ Real.pi / 2)
  (hmin : ∃ (t : ℝ), t ∈ set.Icc 0 1 ∧ (dotProduct (vecA x) (vecB x) - 2 * λ * magnitude (vecA x + vecB x)) = -2) : 
  λ = Real.sqrt 2 / 2 := by
  sorry

end find_lambda_in_interval_l225_225064


namespace monic_quadratic_with_root_l225_225727

theorem monic_quadratic_with_root (a b : ℝ) (h_root1 : (a + b * I) ∈ ({2-3*I, 2+3*I} : set ℂ)) :
  ∃ p : polynomial ℂ, polynomial.monic p ∧ p.coeff 0 = 13 ∧ p.coeff 1 = -4 ∧ p.coeff 2 = 1 ∧ p.eval (a + b * I) = 0 := 
by
  sorry

end monic_quadratic_with_root_l225_225727


namespace gcd_of_polynomials_l225_225036

/-- Given that a is an odd multiple of 7877, the greatest common divisor of
       7a^2 + 54a + 117 and 3a + 10 is 1. -/
theorem gcd_of_polynomials (a : ℤ) (h1 : a % 2 = 1) (h2 : 7877 ∣ a) :
  Int.gcd (7 * a ^ 2 + 54 * a + 117) (3 * a + 10) = 1 :=
sorry

end gcd_of_polynomials_l225_225036


namespace problem_I_problem_II_l225_225444

noncomputable def side_lengths (a b c : ℝ) (A B C : ℝ) := 
  c = 2 ∧ C = π / 3

theorem problem_I (a b : ℝ) (h : side_lengths a b 2 (π / 3)) 
  (area_eq : 1/2 * a * b * (Real.sin $ π / 3) = sqrt 3) : a = 2 ∧ b = 2 :=
by sorry

theorem problem_II (a b : ℝ) (h : side_lengths a b 2 (π / 3))
  (sin_relation : Real.sin (a / 2 / b) = 2 * Real.sin b)
  : 1/2 * a * b * (Real.sin $ π / 3) = 2 * sqrt 3 / 3 :=
by sorry

end problem_I_problem_II_l225_225444


namespace maximum_distance_between_lines_l225_225502

def line1 (m : ℝ) : ℝ × ℝ → Prop :=
λ p, (m + 1) * p.1 + m * p.2 + 2 - m = 0

def line2 (m : ℝ) : ℝ × ℝ → Prop :=
λ p, (m + 1) * p.1 + m * p.2 + b = 0 -- This is general form; b needs determination based on (1, 2)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
Real.sqrt((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem maximum_distance_between_lines :
  ∀ m : ℝ, ∀ p2 : ℝ × ℝ, line2 p2 -> distance (-2, 3) p2 ≤ Real.sqrt(10) :=
by
  sorry

end maximum_distance_between_lines_l225_225502


namespace johns_final_push_time_l225_225260

-- Definitions and initial conditions.
def john_initial_distance_behind_steve : ℝ := 12
def john_speed : ℝ := 4.2
def steve_speed : ℝ := 3.7
def john_final_distance_ahead_of_steve : ℝ := 2

-- The statement we want to prove:
theorem johns_final_push_time : ∃ t : ℝ, john_speed * t = steve_speed * t + john_initial_distance_behind_steve + john_final_distance_ahead_of_steve ∧ t = 28 := 
by 
  -- Adding blank proof body
  sorry

end johns_final_push_time_l225_225260


namespace sara_spent_on_hotdog_l225_225910

-- Define variables for the costs
def costSalad : ℝ := 5.1
def totalLunchBill : ℝ := 10.46

-- Define the cost of the hotdog
def costHotdog : ℝ := totalLunchBill - costSalad

-- The theorem we need to prove
theorem sara_spent_on_hotdog : costHotdog = 5.36 := by
  -- Proof would go here (if required)
  sorry

end sara_spent_on_hotdog_l225_225910


namespace Carly_fourth_week_running_distance_l225_225303

theorem Carly_fourth_week_running_distance :
  let week1_distance_per_day := 2
  let week2_distance_per_day := (week1_distance_per_day * 2) + 3
  let week3_distance_per_day := week2_distance_per_day * (9 / 7)
  let week4_intended_distance_per_day := week3_distance_per_day * 0.9
  let week4_actual_distance_per_day := week4_intended_distance_per_day * 0.5
  let week4_days_run := 5 -- due to 2 rest days
  (week4_actual_distance_per_day * week4_days_run) = 20.25 := 
by 
    -- We use sorry here to skip the proof
    sorry

end Carly_fourth_week_running_distance_l225_225303


namespace question1_total_over_or_underweight_question2_total_sale_price_l225_225631

theorem question1_total_over_or_underweight :
  (2 * (-2.5) + 4 * (-2) + 2 * (-0.5) + 1 * 0 + 3 * 1 + 8 * 1.5) = 1 :=
by
  sorry

theorem question2_total_sale_price :
  let total_weight := 20 * 18 + 1 in
  total_weight * 7 = 2527 :=
by
  sorry

end question1_total_over_or_underweight_question2_total_sale_price_l225_225631


namespace lcm_150_414_l225_225586

theorem lcm_150_414 : Nat.lcm 150 414 = 10350 :=
by
  sorry

end lcm_150_414_l225_225586


namespace min_side_length_of_square_l225_225434

theorem min_side_length_of_square (a : ℝ) (h1 : ∃ p1 p2 p3 p4 p5 p6 : polygon, 
  are_resulting_parts_from_dividing_square a 
  [p1, p2, p3, p4, p5, p6]) 
  (h2 : ∃ (p : polygon), p ∈ [p1, p2, p3, p4, p5, p6] ∧ 
  diameter p ≥ real.sqrt 13) :
  a ≥ 6 := 
by
  sorry

end min_side_length_of_square_l225_225434


namespace polynomials_equal_at_one_l225_225191

-- Define P as a real polynomial
variable (P : Polynomial ℝ)

-- Define Q as another polynomial with its coefficients formed
-- by replacing each nonzero coefficient of P with their mean
noncomputable def Q : Polynomial ℝ := 
  let mean_coeff := (P.coeffs.filter (≠ 0)).sum / (P.coeffs.filter (≠ 0)).length
  ∑ i in P.support, if P.coeff i ≠ 0 then monomial i mean_coeff else 0

-- Prove that P evaluated at 1 equals Q evaluated at 1
theorem polynomials_equal_at_one 
  (P : Polynomial ℝ) :
  P.eval 1 = Q P.eval 1 :=
by
  sorry

end polynomials_equal_at_one_l225_225191


namespace quadratic_function_value_at_13_l225_225301

theorem quadratic_function_value_at_13 
  (d e f : ℝ) 
  (p : ℝ → ℝ) 
  (h_def : ∀ x, p(x) = d*x^2 + e*x + f) 
  (h_symmetry : ∀ x, p(x) = p(7 - x)) 
  (h_point : p (-6) = 0) 
  : p 13 = 0 :=
sorry

end quadratic_function_value_at_13_l225_225301


namespace find_a_l225_225439

noncomputable def unique_quad_solution (a : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1^2 - a * x1 + a = 1 → x2^2 - a * x2 + a = 1 → x1 = x2

theorem find_a (a : ℝ) (h : unique_quad_solution a) : a = 2 :=
sorry

end find_a_l225_225439


namespace find_a_and_b_l225_225406

theorem find_a_and_b :
  (∃ a b : ℝ, ∀ x : ℝ, (x = 1 → f x = 10) ∧ (f' x = 0 → x = 1) → 
    (a = -4 ∧ b = 11)) :=
begin
  sorry
end

noncomputable def f (x : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * a * x - b

end find_a_and_b_l225_225406


namespace arthur_walked_distance_l225_225297

theorem arthur_walked_distance :
  let east_blocks := 8
  let north_blocks := 10
  let south_blocks := 3
  let west_blocks := 5
  let blocks_per_mile := 4
  let total_blocks := east_blocks + north_blocks + south_blocks + west_blocks
in total_blocks / blocks_per_mile = 6.5 :=
by 
  let east_blocks := 8
  let north_blocks := 10
  let south_blocks := 3
  let west_blocks := 5
  let blocks_per_mile := 4
  let total_blocks := east_blocks + north_blocks + south_blocks + west_blocks
  show total_blocks / blocks_per_mile = 6.5, from sorry

end arthur_walked_distance_l225_225297


namespace equivalence_of_circumradii_l225_225957

noncomputable def symmetry_point {P : Type*} [MetricSpace P]
  (A M : P) : P := sorry -- Definition of symmetry point.

noncomputable def circumcircle {P : Type*} [MetricSpace P]
  (A B C : P) : set P := sorry -- Definition of a circumcircle.

noncomputable def intersect {P : Type*} [MetricSpace P]
  (C₁ C₂ : set P) (exclude : P) : P := sorry -- Intersection avoiding a point.

noncomputable def circumradius {P : Type*} [MetricSpace P]
  (A B C : P) : ℝ := sorry -- Circumradius of a triangle.

variables (P : Type*) [MetricSpace P]
variables (O₁ O₂ M N A B C D E F : P)

def problem_conditions :=
  is_intersect (O₁ ∩ O₂) M N ∧
  tangent_touch (O₁ ∩ tangent_near M) A ∧
  tangent_touch (O₂ ∩ tangent_near M) B ∧
  symmetry_point A M = C ∧
  symmetry_point B M = D ∧
  E = intersect (circumcircle D C M) O₁ M ∧
  F = intersect (circumcircle D C M) O₂ M

theorem equivalence_of_circumradii
  (h : problem_conditions P O₁ O₂ M N A B C D E F) :
  circumradius M E F = circumradius N E F := 
sorry

end equivalence_of_circumradii_l225_225957


namespace dot_product_eq_l225_225307

def vector1 : ℝ × ℝ := (-3, 0)
def vector2 : ℝ × ℝ := (7, 9)

theorem dot_product_eq :
  (vector1.1 * vector2.1 + vector1.2 * vector2.2) = -21 :=
by
  sorry

end dot_product_eq_l225_225307


namespace sum_of_coordinates_l225_225624

theorem sum_of_coordinates (x_coords : Fin 80 → ℝ) (y : ℝ) :
  (∑ i, x_coords i = 1600) →
  (∑ j : Fin 80, let mid_x : Fin 80 → ℝ := λ k, (x_coords k + x_coords ((k + 1) % 80)) / 2 
                 in let mid_y : Fin 80 → ℝ := λ k, (y + y) / 2  -- place holder for y-coordinates
                 in mid_x j = x_coords 0 + x_coords 1 + ... + x_coords 79) →  -- This needs proper  definition
  ∑ k, (mid_x k) = 1600 ∧ ∑ l, (mid_y l) = y :=
by
  intros x_sum mid_x
  sorry

end sum_of_coordinates_l225_225624


namespace exterior_angle_bisector_ratio_l225_225174

variable {A B C D : Type}
variable [MetricSpace A]
variable [MetricSpace B]
variable [MetricSpace C]
variable [MetricSpace D]

theorem exterior_angle_bisector_ratio
  (h : ∃ (A B C D : Point), is_triangle A B C ∧ is_on_external_angle_bisector C D ∧ is_on_line_segment A B D) :
  ratio AD BD = ratio AC BC :=
sorry

end exterior_angle_bisector_ratio_l225_225174


namespace smallestThreeDigitNumberWithPerfectSquare_l225_225342

def isThreeDigitNumber (a : ℕ) : Prop := 100 ≤ a ∧ a ≤ 999

def formsPerfectSquare (a : ℕ) : Prop := ∃ n : ℕ, 1001 * a + 1 = n * n

theorem smallestThreeDigitNumberWithPerfectSquare :
  ∀ a : ℕ, isThreeDigitNumber a → formsPerfectSquare a → a = 183 :=
by
sorry

end smallestThreeDigitNumberWithPerfectSquare_l225_225342


namespace count_odd_numbers_301_699_l225_225797

theorem count_odd_numbers_301_699 :
  let f := λ n, 301 + (n - 1) * 2 in
  ∀ (n : ℕ), 1 ≤ n → f n ≤ 699 → n = 200 :=
by
  let f := λ n, 301 + (n - 1) * 2
  intro n hn₁ hn₂
  have h₁ : 301 ≤ 301 + (n - 1) * 2 := by linarith
  have h₂ : 301 + (n - 1) * 2 ≤ 699 := hn₂
  sorry

end count_odd_numbers_301_699_l225_225797


namespace parallel_lines_l225_225378

open Set

variables {a b c : Line} {α β γ δ : Plane}

-- Define the lines and planes and specify their properties
axiom distinct_lines : ∀ (x y z : Line), x ≠ y ∧ y ≠ z ∧ x ≠ z
axiom distinct_planes : ∀ (u v w x : Plane), u ≠ v ∧ v ≠ w ∧ w ≠ x ∧ u ≠ w ∧ u ≠ x ∧ v ≠ x

-- Define plane intersections resulting in lines
axiom intersection_a : α ∩ β = { a }
axiom intersection_b : γ ∩ δ = { b }

-- Define the parallelism conditions among planes
axiom parallel_planes_1 : α ∥ γ
axiom parallel_planes_2 : β ∥ δ

-- The theorem to prove
theorem parallel_lines (distinct_lines a b c) (distinct_planes α β γ δ) : a ∥ b :=
by { sorry }

end parallel_lines_l225_225378


namespace min_value_expression_l225_225129

noncomputable section

variables {x y : ℝ}

theorem min_value_expression (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 1 ∧ 
    (∃ min_val : ℝ, min_val = (x^2 / (x + 2) + y^2 / (y + 1)) ∧ min_val = 1 / 4)) :=
  sorry

end min_value_expression_l225_225129


namespace H_G_A_Q_ordering_l225_225859

variable {x y : ℝ}

-- Conditions
theorem H_G_A_Q_ordering
  (hx : 0 < x)
  (hy : 0 < y)
  (hxy : x < y)
  : 
  let H := (2 * x * y) / (x + y)
  let G := Real.sqrt (x * y)
  let A := (x + y) / 2
  let Q := Real.sqrt ((x^2 + y^2) / 2)
  in (G - H ≤ A - G) ∧ (A - G ≤ Q - A) ∧ (Q - A ≤ G - H) :=
by sorry

end H_G_A_Q_ordering_l225_225859


namespace vector_expression_identity_l225_225418

variables (E : Type) [AddCommGroup E] [Module ℝ E]
variables (e1 e2 : E)
variables (a b : E)
variables (cond1 : a = (3 : ℝ) • e1 - (2 : ℝ) • e2) (cond2 : b = (e2 - (2 : ℝ) • e1))

theorem vector_expression_identity :
  (1 / 3 : ℝ) • a + b + a - (3 / 2 : ℝ) • b + 2 • b - a = -2 • e1 + (5 / 6 : ℝ) • e2 :=
sorry

end vector_expression_identity_l225_225418


namespace shortest_chord_length_intercepted_l225_225785

def circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 5}

def point_A := (2, 1)

def is_shortest_chord (line : ℝ × ℝ → Prop) (chord_length : ℝ) := 
  line point_A ∧ chord_length = 2

theorem shortest_chord_length_intercepted 
: ∀ (line : ℝ × ℝ → Prop), circle.point_A → (∃ chord_length : ℝ, is_shortest_chord line chord_length) :=
by
  sorry

end shortest_chord_length_intercepted_l225_225785


namespace average_charge_per_person_l225_225983

-- Define the given conditions
def charge_first_day : ℝ := 15
def charge_second_day : ℝ := 7.5
def charge_third_day : ℝ := 2.5

def attendance_ratio_first_day : ℕ := 2
def attendance_ratio_second_day : ℕ := 5
def attendance_ratio_third_day : ℕ := 13

-- Average charge per person statement
theorem average_charge_per_person (x : ℝ) :
  let visitors_first_day := attendance_ratio_first_day * x,
      visitors_second_day := attendance_ratio_second_day * x,
      visitors_third_day := attendance_ratio_third_day * x,
      total_revenue := (visitors_first_day * charge_first_day) +
                       (visitors_second_day * charge_second_day) +
                       (visitors_third_day * charge_third_day),
      total_visitors := visitors_first_day + visitors_second_day + visitors_third_day in
  (total_revenue / total_visitors) = 5 := 
by
  -- proof goes here
  sorry

end average_charge_per_person_l225_225983


namespace parity_of_expression_l225_225872

theorem parity_of_expression (o1 o2 n : ℕ) (h1 : o1 % 2 = 1) (h2 : o2 % 2 = 1) : 
  ((o1 * o1 + n * (o1 * o2)) % 2 = 1 ↔ n % 2 = 0) :=
by sorry

end parity_of_expression_l225_225872


namespace james_bought_six_shirts_l225_225113

-- Definitions for the given conditions
def discount : ℝ := 0.5
def original_price : ℝ := 20
def total_amount_paid : ℝ := 60

-- Definition for the number of t-shirts James bought
def num_tshirts : ℝ := total_amount_paid / (original_price * discount)

-- The theorem to prove
theorem james_bought_six_shirts : num_tshirts = 6 :=
by
  -- skipping the proof with sorry
  sorry

end james_bought_six_shirts_l225_225113


namespace cubics_identity_l225_225786

variable (a b c x y z : ℝ)

theorem cubics_identity (X Y Z : ℝ)
  (h1 : X = a * x + b * y + c * z)
  (h2 : Y = a * y + b * z + c * x)
  (h3 : Z = a * z + b * x + c * y) :
  X^3 + Y^3 + Z^3 - 3 * X * Y * Z = 
  (x^3 + y^3 + z^3 - 3 * x * y * z) * (a^3 + b^3 + c^3 - 3 * a * b * c) :=
sorry

end cubics_identity_l225_225786


namespace nonneg_implies_constraints_l225_225863

noncomputable def f (a b A B : ℝ) (x : ℝ) : ℝ := 
  1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x)

theorem nonneg_implies_constraints (a b A B : ℝ)
  (h : ∀ x : ℝ, f a b A B x ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
begin
  sorry
end

end nonneg_implies_constraints_l225_225863


namespace problem_ABC_projections_similar_and_cyclic_l225_225254

universe u

noncomputable theory

variables {α : Type u} [EuclideanGeometry α]

-- Assume ABC is a triangle with specific vertices A, B, C
variables (A B C A₁ B₁ C₁ A₂ B₂ C₂ O₁ O₂ : α)

-- Conditions
def projections (A B C A₁ B₁ C₁ A₂ B₂ C₂ O₁ O₂ : α) : Prop :=
  (is_projection A₁ A B C O₁ ∧ is_projection B₁ B A C O₁ ∧ is_projection C₁ C A B O₁) ∧
  (is_projection A₂ A B C O₂ ∧ is_projection B₂ B A C O₂ ∧ is_projection C₂ C A B O₂)

-- Triangles are similar
def similar (A B C A₁ B₁ C₁ A₂ B₂ C₂ : α) : Prop :=
  triangle_similar A B C A₁ B₁ C₁ ∧ triangle_similar A B C A₂ B₂ C₂ ∧
  triangle_similar A₁ B₁ C₁ A₂ B₂ C₂

-- Triangles are equal
def equal_triangles (A₁ B₁ C₁ A₂ B₂ C₂ : α) : Prop :=
  triangle_equal A₁ B₁ C₁ A₂ B₂ C₂

-- Points lie on the same circle
def cyclic (A₁ B₁ C₁ A₂ B₂ C₂ : α) (O : α) : Prop :=
  circle_circumscribes A₁ B₁ C₁ A₂ B₂ C₂ O

-- Conditions of perpendicularity
def perpendicular_sides (A B C A₁ B₁ C₁ A₂ B₂ C₂ : α) : Prop :=
  perpendicular_to_sides A B C A₁ B₁ C₁ ∧ perpendicular_to_sides A B C A₂ B₂ C₂

-- Conditions of equal and collinear midpoints
def equal_segments (A₁ B₁ C₁ A₂ B₂ C₂ : α) (M : α) : Prop :=
  equal_and_collinear_segments A₁ A₂ B₁ B₂ C₁ C₂ M

-- The statement of the problem
theorem problem_ABC_projections_similar_and_cyclic (A B C A₁ B₁ C₁ A₂ B₂ C₂ O₁ O₂ : α) (O M : α) :
  projections A B C A₁ B₁ C₁ A₂ B₂ C₂ O₁ O₂ →
  similar A B C A₁ B₁ C₁ A₂ B₂ C₂ →
  equal_triangles A₁ B₁ C₁ A₂ B₂ C₂ →
  cyclic A₁ B₁ C₁ A₂ B₂ C₂ O →
  perpendicular_sides A B C A₁ B₁ C₁ A₂ B₂ C₂ →
  equal_segments A₁ B₁ C₁ A₂ B₂ C₂ M :=
sorry

end problem_ABC_projections_similar_and_cyclic_l225_225254


namespace debate_teams_l225_225928

theorem debate_teams (girls boys : ℕ) (h_girls : girls = 4) (h_boys : boys = 6) :
  (nat.choose girls 3) * (nat.choose boys 3) = 80 :=
by
  rw [h_girls, h_boys]
  simp [nat.choose]
  sorry

end debate_teams_l225_225928


namespace find_a_if_y_has_three_zeros_l225_225320

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≠ 4 then 2 / |x - 4| else a

def y (x : ℝ) (a : ℝ) : ℝ :=
  f x a - 2

theorem find_a_if_y_has_three_zeros :
  ∃ a : ℝ, (∀ x : ℝ, y x a = 0 → (x = 3 ∨ x = 4 ∨ x = 5)) ↔ a = 2 :=
by
  sorry

end find_a_if_y_has_three_zeros_l225_225320


namespace trigonometric_identity_l225_225374

noncomputable def tan (α : ℝ) := Mathlib.sin α / Mathlib.cos α
noncomputable def expression (α : ℝ) := (Mathlib.sin α - 3 * Mathlib.cos α) / (Mathlib.sin α + Mathlib.cos α)

theorem trigonometric_identity (α : ℝ) (h : tan α = 2) : expression α = -1 / 3 := 
by 
  sorry

end trigonometric_identity_l225_225374


namespace roland_thread_length_l225_225533

noncomputable def length_initial : ℝ := 12
noncomputable def length_two_thirds : ℝ := (2 / 3) * length_initial
noncomputable def length_increased : ℝ := length_initial + length_two_thirds
noncomputable def length_half_increased : ℝ := (1 / 2) * length_increased
noncomputable def length_total : ℝ := length_increased + length_half_increased
noncomputable def length_inches : ℝ := length_total / 2.54

theorem roland_thread_length : length_inches = 11.811 :=
by sorry

end roland_thread_length_l225_225533


namespace cheryl_initial_strawberries_l225_225305

theorem cheryl_initial_strawberries (leftover_per_bucket : ℕ) (taken_per_bucket : ℕ) (total_buckets : ℕ)
(h_leftover : leftover_per_bucket = 40)
(h_taken : taken_per_bucket = 20)
(h_buckets : total_buckets = 5) :
  total_buckets * (leftover_per_bucket + taken_per_bucket) = 300 :=
by
  rw [h_leftover, h_taken, h_buckets]
  norm_num
  sorry

end cheryl_initial_strawberries_l225_225305


namespace surface_area_prism_cut_l225_225285

noncomputable def surfaceAreaQVWX (h : ℕ) (s : ℕ) (V W X : EuclideanSpace ℝ (Fin 3)) (QVX QWX VWX: Set (EuclideanSpace ℝ (Fin 3))) : ℝ :=
  (sqrt 3 / 4) * s^2 + sorry -- Placeholders for the other calculated areas. Adjust accordingly.

theorem surface_area_prism_cut:
  ∀ (PQRSTU : EuclideanSpace ℝ (Fin 3)) (h : ℕ) (s : ℕ)
    (V W X : EuclideanSpace ℝ (Fin 3))
    (QVX QWX VWX QVW : Set (EuclideanSpace ℝ (Fin 3))),
  is_midpoint PQRSTU V W X →
  QVW = setOfPoints [V, W, Q] →
  QVX = setOfPoints [Q, V, X] →
  QWX = setOfPoints [Q, W, X] →
  VWX = setOfPoints [V, W, X] →
  h = 20 →
  s = 10 →
  surfaceAreaQVWX h s V W X = (25 * (sqrt 3) / 4) + sorry :=
begin
    -- The proof steps will go here
    sorry,
end

end surface_area_prism_cut_l225_225285


namespace abc_sum_l225_225177

theorem abc_sum :
  ∃ a b c : ℤ, 
    (x^2 + 9 * x + 20 = (x + a) * (x + b)) ∧
    (x^2 + 7 * x - 30 = (x + b) * (x - c)) ∧
    a + b + c = 12 := 
by
  use 4, 5, 3
  split
  sorry

# You would complete the proof part by removing sorry and providing the necessary validation steps, but as instructed, we only provide the problem statement here.

end abc_sum_l225_225177


namespace inverse_value_of_1_over_2_l225_225052

def f (x : ℝ) : ℝ := 2 ^ x
def g (y : ℝ) : ℝ := Real.log y / Real.log 2

theorem inverse_value_of_1_over_2 : g (1 / 2) = -1 := sorry

end inverse_value_of_1_over_2_l225_225052


namespace polynomial_roots_l225_225007

noncomputable def polynomial : Polynomial ℝ := 
  Polynomial.C 3 * Polynomial.X^4 +
  Polynomial.C 2 * Polynomial.X^3 -
  Polynomial.C 7 * Polynomial.X^2 +
  Polynomial.C 2 * Polynomial.X +
  Polynomial.C 3

-- Define the roots for which we need to prove the polynomial equation is zero
def r1 : ℝ := _ -- Calculation needed
def r2 : ℝ := _ -- Calculation needed
def r3 : ℝ := _ -- Calculation needed
def r4 : ℝ := _ -- Calculation needed

theorem polynomial_roots : 
  polynomial.eval r1 = 0 ∧ 
  polynomial.eval r2 = 0 ∧ 
  polynomial.eval r3 = 0 ∧ 
  polynomial.eval r4 = 0 := 
sorry

end polynomial_roots_l225_225007


namespace new_average_after_17th_l225_225271

def old_average (A : ℕ) (n : ℕ) : ℕ :=
  A -- A is the average before the 17th inning

def runs_in_17th : ℕ := 84 -- The score in the 17th inning

def average_increase : ℕ := 3 -- The increase in average after the 17th inning

theorem new_average_after_17th (A : ℕ) (n : ℕ) (h1 : n = 16) (h2 : old_average A n + average_increase = A + 3) :
  (old_average A n) + average_increase = 36 :=
by
  sorry

end new_average_after_17th_l225_225271


namespace complex_power_eq_neg_i_l225_225037

theorem complex_power_eq_neg_i (i : ℂ) (h : i * i = -1) : (Complex.div ((1 : ℂ) + i) ((1 : ℂ) - i)) ^ 2007 = -i :=
by
  sorry

end complex_power_eq_neg_i_l225_225037


namespace x_squared_plus_y_squared_l225_225382

theorem x_squared_plus_y_squared (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
(h3 : x * y + x + y = 71)
(h4 : x^2 * y + x * y^2 = 880) :
x^2 + y^2 = 146 :=
sorry

end x_squared_plus_y_squared_l225_225382


namespace power_complex_expression_l225_225989

theorem power_complex_expression :
  ( ( ( (√3) / 2 ) + ( (Complex.I) / 2 ) ) ^ 2023 ) = ( -( (√3) / 2 ) - ( (Complex.I) / 2 ) ) := 
by
  sorry

end power_complex_expression_l225_225989


namespace binom_sum_squares_l225_225680

-- Declare binomial coefficients
def binom : ℕ → ℕ → ℕ 
| n k := Nat.choose n k

-- Define the sum of squares of binomial coefficients
def sum_of_squares (m n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), (binom k m)^2

-- Define the specific sum we're interested in
def specific_sum := sum_of_squares 2 11

-- State the theorem
theorem binom_sum_squares : specific_sum = 220 := 
  by 
    -- Proof will be inserted here
    sorry

end binom_sum_squares_l225_225680


namespace prove_sine_function_l225_225792

theorem prove_sine_function :
  ∃ A ω φ, (∀ x, f(x) = A * sin(ω * x + φ)) → 
  (∀ y, f(y) ≤ 2) → 
  (∃ T, (T > 0) ∧ (∀ z, f(z + T) = f(z))) → 
  (f(0) = 1) → 
  (f(x) = 2 * sin(x - π / 6)) :=
sorry

end prove_sine_function_l225_225792


namespace ratio_of_areas_is_correct_l225_225498

-- Define the conditions
def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 ≠ B.1 ∧ B.2 = C.2 ∧ C.1 ≠ D.1 ∧ D.2 = A.2 ∧
  dist A B = 8 ∧ dist B C = 4

def equilateral_triangle_center (A B : ℝ × ℝ) (ext : ℝ × ℝ) : ℝ × ℝ :=
  let mid := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let height := (dist A B) * (real.sqrt 3 / 2) / 3
  ((if ext.2 < mid.2 then mid.1 else mid.1), (if ext.2 < mid.2 then mid.2 - height else mid.2 + height))

-- Define the points of the rectangle
def A := (0, 0) : ℝ × ℝ
def B := (8, 0) : ℝ × ℝ
def C := (8, 4) : ℝ × ℝ
def D := (0, 4) : ℝ × ℝ

-- Define the centers of the equilateral triangles
def E := equilateral_triangle_center A B (0, -1)
def F := equilateral_triangle_center B C (1, 0)
def G := equilateral_triangle_center C D (0, 1)
def H := equilateral_triangle_center D A (-1, 0)

-- A noncomputable definition for the ratio of the areas
noncomputable def ratio_area_quadrilateral_rectangle : ℝ :=
  let area_ABCD := 8 * 4
  let area_quadrilateral_EFGH := -- complex calculation of area based on vertices
    sorry
  area_quadrilateral_EFGH / area_ABCD

-- The final proof statement of the problem
theorem ratio_of_areas_is_correct :
  is_rectangle A B C D →
  ratio_area_quadrilateral_rectangle = (5 + 2 * real.sqrt 3) / 8 :=
by
  sorry

end ratio_of_areas_is_correct_l225_225498


namespace seating_arrangement_l225_225712

variable {M I P A : Prop}

def first_fact : ¬ M := sorry
def second_fact : ¬ A := sorry
def third_fact : ¬ M → I := sorry
def fourth_fact : I → P := sorry

theorem seating_arrangement : ¬ M → (I ∧ P) :=
by
  intros hM
  have hI : I := third_fact hM
  have hP : P := fourth_fact hI
  exact ⟨hI, hP⟩

end seating_arrangement_l225_225712


namespace angle_theta_change_l225_225310

-- Define the conditions

-- A right triangle ABC with hypotenuse AB
structure RightTriangle (A B C : Type _) : Type _ :=
  (hypotenuse : ∀ AB, Angle A B C = 90)

-- Point P inside the triangle dividing it into two equal areas
structure PointP (A B C P : Type _) :=
  (inside : ∀ P, P ∈ Triangle A B C)
  (equal_area : ∀ α1 α2 : ℝ, α1 = α2)

-- Define the proof problem
theorem angle_theta_change (A B C P : Type _) [RightTriangle A B C] [PointP A B C P] : 
  ∀ (θ : ℝ), θ ∈ Set.Icc 0 90 :=
sorry

end angle_theta_change_l225_225310


namespace four_students_same_acquaintances_l225_225697

theorem four_students_same_acquaintances
  (students : Finset ℕ)
  (acquainted : ∀ s ∈ students, (students \ {s}).card ≥ 68)
  (count : students.card = 102) :
  ∃ n, ∃ cnt, cnt ≥ 4 ∧ (∃ S, S ⊆ students ∧ S.card = cnt ∧ ∀ x ∈ S, (students \ {x}).card = n) :=
sorry

end four_students_same_acquaintances_l225_225697


namespace proof_average_l225_225921

def average_two (x y : ℚ) : ℚ := (x + y) / 2
def average_three (x y z : ℚ) : ℚ := (x + y + z) / 3

theorem proof_average :
  average_three (2 * average_three 3 2 0) (average_two 0 3) (1 * 3) = 47 / 18 :=
by
  sorry

end proof_average_l225_225921


namespace white_circle_area_l225_225520

theorem white_circle_area (edge_length : ℕ) (total_paint_area : ℕ) (faces : ℕ) (total_face_area : ℕ)
  (paint_per_face : ℕ) (white_area_per_face : ℕ) : edge_length = 12 → total_paint_area = 432 →
  faces = 6 → total_face_area = 6 * (edge_length * edge_length) →
  paint_per_face = total_paint_area / faces →
  white_area_per_face = (edge_length * edge_length) - paint_per_face →
  white_area_per_face = 72 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1] at h4 h5 h6
  rw [h2, h3, mul_comm 12 12, nat.mul_div_left 432 6 (by norm_num)] at h4 h5 h6
  rw [nat.mul_sub_right_distrib, nat.sub_self, add_zero] at h6
  exact h6

end white_circle_area_l225_225520


namespace magnitude_of_conjugate_complex_l225_225429

-- Define the problem
theorem magnitude_of_conjugate_complex:
  ∃ (z : ℂ), z = conj (3 + 4 * I) ∧ ∥z∥ = 5 :=
by
  sorry

end magnitude_of_conjugate_complex_l225_225429


namespace pizza_percentage_increase_l225_225170

-- Define the radii of the pizzas
def radius1 : ℝ := 5
def radius2 : ℝ := 3

-- Define the areas using the formula for the area of a circle
def area1 : ℝ := Real.pi * radius1^2
def area2 : ℝ := Real.pi * radius2^2

-- Define the percentage increase
def percentage_increase : ℝ := ((area1 - area2) / area2) * 100

-- Statement to prove
theorem pizza_percentage_increase : Int.round percentage_increase = 178 :=
by
  sorry

end pizza_percentage_increase_l225_225170


namespace at_least_one_not_less_than_two_l225_225504

theorem at_least_one_not_less_than_two
  (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (2 ≤ (y / x + y / z)) ∨ (2 ≤ (z / x + z / y)) ∨ (2 ≤ (x / z + x / y)) :=
sorry

end at_least_one_not_less_than_two_l225_225504


namespace find_k_plus_j_l225_225907

noncomputable def length_EG (EV EP VF : ℕ) (area_ratio : ℚ) : ℝ :=
  let w := (3 / 2) * Real.sqrt 10 in
  w * 50

theorem find_k_plus_j (EV EP VF : ℕ) (ratio : ℚ) (EG_calculated : ℝ) :
  (EG_calculated = (3 / 2) * 50 * Real.sqrt 10) → 
  (EV = 100) → (EP = 150) → (VF = 200) → (ratio = 2 / 3) → 
  let k := 75 in
  let j := 10 in
  (k + j = 85) :=
by
  intros h_eq h_ev h_ep h_vf h_ratio
  sorry

end find_k_plus_j_l225_225907


namespace Jim_paycheck_correct_l225_225469

noncomputable def Jim_paycheck_after_deductions (gross_pay : ℝ) (retirement_percentage : ℝ) (tax_deduction : ℝ) : ℝ :=
  gross_pay - (gross_pay * retirement_percentage) - tax_deduction

theorem Jim_paycheck_correct :
  Jim_paycheck_after_deductions 1120 0.25 100 = 740 :=
by sorry

end Jim_paycheck_correct_l225_225469


namespace correct_average_marks_l225_225172

theorem correct_average_marks 
  (n : ℕ) (wrong_avg : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ)
  (h1 : n = 10)
  (h2 : wrong_avg = 100)
  (h3 : wrong_mark = 90)
  (h4 : correct_mark = 10) :
  (n * wrong_avg - wrong_mark + correct_mark) / n = 92 :=
by
  sorry

end correct_average_marks_l225_225172


namespace karl_drove_420_miles_l225_225476

theorem karl_drove_420_miles :
  ∀ (car_mileage_per_gallon : ℕ)
    (tank_capacity : ℕ)
    (initial_drive_miles : ℕ)
    (gas_purchased : ℕ)
    (destination_tank_fraction : ℚ),
    car_mileage_per_gallon = 30 →
    tank_capacity = 16 →
    initial_drive_miles = 420 →
    gas_purchased = 10 →
    destination_tank_fraction = 3 / 4 →
    initial_drive_miles + (destination_tank_fraction * tank_capacity - (tank_capacity - (initial_drive_miles / car_mileage_per_gallon)) + gas_purchased) * car_mileage_per_gallon = 420 :=
by
  intros car_mileage_per_gallon tank_capacity initial_drive_miles gas_purchased destination_tank_fraction
  intro h1 -- car_mileage_per_gallon = 30
  intro h2 -- tank_capacity = 16
  intro h3 -- initial_drive_miles = 420
  intro h4 -- gas_purchased = 10
  intro h5 -- destination_tank_fraction = 3 / 4
  sorry

end karl_drove_420_miles_l225_225476


namespace Δ5_is_zero_l225_225684

def v (n : ℕ) : ℕ := n^4 + 2 * n^2 + 2

def Δ1 (v : ℕ → ℕ) (n : ℕ) : ℕ := v (n + 1) - v n

def Δk (k : ℕ) (v : ℕ → ℕ) : (ℕ → ℕ)
| 0 := v
| (k + 1) := Δ1 (Δk k v)

theorem Δ5_is_zero (v : ℕ → ℕ) (n : ℕ) :
  Δk 5 v n = 0 :=
sorry

end Δ5_is_zero_l225_225684


namespace coeff_sum_even_indices_l225_225069

theorem coeff_sum_even_indices (a : ℕ → ℤ) (f : ℤ → ℤ → Prop)
  (h1 : (1 + (X : ℤ))(2 - X)^2015 = ∑ i in 0..2016, a i * X^i)
  (h2 : f (X : ℤ) := (∑ i in range 2017, (-1)^i * a i * X^i))
  : (∑ i in range 0..2017 steps 2, a i) = -2^2015 := sorry

end coeff_sum_even_indices_l225_225069


namespace total_pupils_across_schools_l225_225947

theorem total_pupils_across_schools :
  let schoolA_pupils := 542 + 387 in
  let schoolB_pupils := 713 + 489 in
  let schoolC_pupils := 628 + 361 in
  schoolA_pupils + schoolB_pupils + schoolC_pupils = 3120 :=
by
  let schoolA_pupils := 542 + 387
  let schoolB_pupils := 713 + 489
  let schoolC_pupils := 628 + 361
  have hA : schoolA_pupils = 929 := rfl
  have hB : schoolB_pupils = 1202 := rfl
  have hC : schoolC_pupils = 989 := rfl
  show schoolA_pupils + schoolB_pupils + schoolC_pupils = 3120
  calc
    schoolA_pupils + schoolB_pupils + schoolC_pupils
    = 929 + 1202 + 989 : by rw [hA, hB, hC]
    ... = 3120 : by norm_num


end total_pupils_across_schools_l225_225947


namespace max_marks_is_400_l225_225290

-- Given conditions
def passing_mark (M : ℝ) : ℝ := 0.30 * M
def student_marks : ℝ := 80
def marks_failed_by : ℝ := 40
def pass_marks : ℝ := student_marks + marks_failed_by

-- Statement to prove
theorem max_marks_is_400 (M : ℝ) (h : passing_mark M = pass_marks) : M = 400 :=
by sorry

end max_marks_is_400_l225_225290


namespace tony_halfway_time_l225_225953

variable (d_groceries d_haircut d_doctor d_school d_postoffice : ℝ)
variable (speed : ℝ)

/- Conditions -/
axiom h1 : d_groceries = 10
axiom h2 : d_haircut = 15
axiom h3 : d_doctor = 5
axiom h4 : d_school = 20
axiom h5 : d_postoffice = 25
axiom h6 : speed = 50

/- Definition of total distance and halfway distance -/
def total_distance := d_groceries + d_haircut + d_doctor + d_school + d_postoffice
def halfway_distance := total_distance / 2

/- Time calculation -/
def time_to_halfway (d_groceries d_haircut d_doctor d_school d_postoffice speed : ℝ) :=
  halfway_distance / speed

theorem tony_halfway_time : time_to_halfway d_groceries d_haircut d_doctor d_school d_postoffice speed = 0.75 := by
  rw [←h1, ←h2, ←h3, ←h4, ←h5, ←h6]
  unfold time_to_halfway halfway_distance total_distance
  norm_num
  sorry

end tony_halfway_time_l225_225953


namespace prime_factors_of_product_l225_225423

def is_prime (n : ℕ) : Prop := Nat.Prime n
def product (a b c : ℕ) := a * b * c

theorem prime_factors_of_product (a b c d : ℕ) :
  a = 3 * 29 →
  is_prime b →
  c = 7 * 13 →
  d = 3 * 31 →
  ∃ p,
    Nat.Prime p ∧
  (∏₁ := p 
  a * b * c + 1) *
  d) =
    7 :=
by
  sorry

end prime_factors_of_product_l225_225423


namespace find_point_P_l225_225493

/-- The point P such that AP = BP = CP = DP for given points A, B, C, D in 3D space -/
theorem find_point_P :
  ∃ P : ℝ × ℝ × ℝ, 
    let A := (10, 0, 0 : ℝ × ℝ × ℝ),
        B := (0, -6, 0 : ℝ × ℝ × ℝ),
        C := (0, 0, 8 : ℝ × ℝ × ℝ),
        D := (0, 0, 0 : ℝ × ℝ × ℝ) in
      dist P A = dist P B ∧
      dist P B = dist P C ∧
      dist P C = dist P D ∧
      P = (5, -3, 4) :=
by
  sorry

end find_point_P_l225_225493


namespace alpha_perpendicular_beta_l225_225122

noncomputable def skew_lines (a b : Line) : Prop := a.nonIntersect b && ¬(a // b)

noncomputable def perpendicular_line_plane (l : Line) (p : Plane) : Prop := ⟪l , p⟫ = 0

noncomputable def perpendicular_planes (p1 p2 : Plane) : Prop := ⟪p1, p2⟫ = 0

constant Line : Type
constant Plane : Type
constant a b : Line
constant α β : Plane

axiom skew_a_b : skew_lines a b
axiom a_perpendicular_α : perpendicular_line_plane a α
axiom b_perpendicular_β : perpendicular_line_plane b β
axiom a_notin_β : ¬(is_subline a β)
axiom b_notin_α : ¬(is_subline b α)

theorem alpha_perpendicular_beta :
  perpendicular_planes α β ↔ perpendicular_lines a b :=
sorry

end alpha_perpendicular_beta_l225_225122


namespace cost_to_fill_pool_l225_225115

-- Definitions based on the conditions
def cubic_foot_to_liters: ℕ := 25
def pool_depth: ℕ := 10
def pool_width: ℕ := 6
def pool_length: ℕ := 20
def cost_per_liter: ℕ := 3

-- Statement to be proved
theorem cost_to_fill_pool : 
  (pool_depth * pool_width * pool_length * cubic_foot_to_liters * cost_per_liter) = 90000 := 
by 
  sorry

end cost_to_fill_pool_l225_225115


namespace find_point_P_l225_225491

/-- The point P such that AP = BP = CP = DP for given points A, B, C, D in 3D space -/
theorem find_point_P :
  ∃ P : ℝ × ℝ × ℝ, 
    let A := (10, 0, 0 : ℝ × ℝ × ℝ),
        B := (0, -6, 0 : ℝ × ℝ × ℝ),
        C := (0, 0, 8 : ℝ × ℝ × ℝ),
        D := (0, 0, 0 : ℝ × ℝ × ℝ) in
      dist P A = dist P B ∧
      dist P B = dist P C ∧
      dist P C = dist P D ∧
      P = (5, -3, 4) :=
by
  sorry

end find_point_P_l225_225491


namespace sum_ratio_arithmetic_sequence_l225_225024

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = (n * (a 1 + a n)) / 2

-- Given conditions
theorem sum_ratio_arithmetic_sequence :
  ∀ (a S : ℕ → ℝ) (d : ℝ),
  arithmetic_sequence a d →
  sum_of_first_n_terms S a →
  a 8 / a 7 = 13 / 5 →
  S 15 / S 13 = 3 :=
begin
  intros,
  sorry
end

end sum_ratio_arithmetic_sequence_l225_225024


namespace crayons_distribution_l225_225884

theorem crayons_distribution (x C : ℝ) : 
  (∀ C, 0.70 * C = 30 * x) → 
  (x / 210 = x / C) → 
  C = 210 → 
  x = 4.9 :=
by
  intro h1 h2 h3
  have : 0.70 * 210 = 30 * x := by
    rw [h3] at h1
    exact h1 210
  have : 147 = 30 * x := by
    rw [mul_comm] at this
    exact this
  have : x = 4.9 := by
    rw [←div_eq_iff_mul_eq'] at this
    exact this
  exact this

end crayons_distribution_l225_225884


namespace find_F_neg_a_l225_225266

-- Definitions of odd functions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Definition of F
def F (f g : ℝ → ℝ) (x : ℝ) := 3 * f x + 5 * g x + 2

theorem find_F_neg_a (f g : ℝ → ℝ) (a : ℝ)
  (hf : is_odd f) (hg : is_odd g) (hFa : F f g a = 3) : F f g (-a) = 1 :=
by
  sorry

end find_F_neg_a_l225_225266


namespace third_number_exists_function_f_exists_l225_225253

-- Define E and M as given in the conditions
def E (a b : ℝ) : ℝ := (a + b) / (1 - a * b)
def M (b c : ℝ) : ℝ := (b + c) / (b * c - 1)

-- Define the function f as given in the condition of Part b
def f (x y : ℝ) : ℝ := (x + y) / (1 - x * y)

-- Theorem statement for part a: existence of a third number H
theorem third_number_exists (a b c : ℝ) (h1 : E a b ≠ 0) (h2 : M b c ≠ 0) (h3 : 1 + a * c ≠ 0) :
  ∃ H : ℝ, E a b + M b c + H = E a b * M b c * H ∧ H = (c - a) / (1 + a * c) :=
sorry

-- Theorem statement for part b: existence of function f
theorem function_f_exists (a b c : ℝ) (h1 : a * b ≠ 1) (h2 : b * c ≠ 1) :
  ∃ (H : ℝ), H = f (-(E a b)) (-(M b c)) :=
sorry

end third_number_exists_function_f_exists_l225_225253


namespace ellipse_equation_fixed_point_l225_225367

theorem ellipse_equation (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a)
  (h₃ : (2^2)/(a^2) + 0/(b^2) = 1)
  (h₄ : (1/2)^2 = (1 - (b^2 / a^2))) :
  ∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1 ↔
  x^2 / a^2 + y^2 / b^2 = 1) := by
  sorry

theorem fixed_point (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a)
  (h₃ : (2^2)/(a^2) + 0/(b^2) = 1)
  (h₄ : (1/2)^2 = (1 - (b^2 / a^2)))
  (P : ℝ × ℝ) (h₅ : P.1 = -1)
  (MN : set (ℝ × ℝ)) (h₆ : ∀ M N ∈ MN, P = (M + N) / 2)
  (l : set (ℝ × ℝ)) (h₇ : ∀ Q ∈ MN, l ∈ perpThrough P Q) :
  fixed_point l (-1/4, 0) := by
  sorry

end ellipse_equation_fixed_point_l225_225367


namespace solve_system_l225_225165

noncomputable def is_solution (x y : ℝ) : Prop :=
  2 * (log x / log (1 / y) - 2 * log y / log (x^2)) + 5 = 0 ∧
  x * y^2 = 32 ∧
  x > 0 ∧ x ≠ 1 ∧
  y > 0 ∧ y ≠ 1

theorem solve_system : 
  is_solution 2 4 ∧ is_solution (4 * real.sqrt 2) (2 * real.root 4 2) :=
sorry

end solve_system_l225_225165


namespace prob_sum_9_is_1_div_5_l225_225961

def r : set ℤ := { 2 , 3 , 4 , 5 }
def b : set ℤ := { 4 , 5 , 6 , 7 , 8 }

def count_pairs_with_sum_9 : ℕ :=
  {p | p.1 ∈ r ∧ p.2 ∈ b ∧ p.1 + p.2 = 9}.to_finset.card

def total_pairs : ℕ :=
  (r.to_finset.card) * (b.to_finset.card)

noncomputable def probability_sum_9 : ℚ :=
  count_pairs_with_sum_9 / total_pairs

theorem prob_sum_9_is_1_div_5 : probability_sum_9 = 1 / 5 :=
by
  sorry

end prob_sum_9_is_1_div_5_l225_225961


namespace trapezoid_is_planar_l225_225245

-- Define a trapezoid type with basic properties in Lean
structure Trapezoid (α : Type) :=
(a1 : α) (a2 : α) (b1 : α) (b2 : α)
-- Assume the two parallel lines define a plane
(ax : ∃ u v w x : α, u ≠ v ∧ {u, v} = {a1, a2} ∧ {w, x} = {b1, b2})

theorem trapezoid_is_planar (α : Type) [Plane α] (T : Trapezoid α) : 
  ∃ (P : Plane α), ∀ p ∈ T, p ∈ P :=
by
  sorry

end trapezoid_is_planar_l225_225245


namespace ions_electron_shell_structure_l225_225352

theorem ions_electron_shell_structure
  (a b n m : ℤ) 
  (same_electron_shell_structure : a + n = b - m) :
  a + m = b - n :=
by
  sorry

end ions_electron_shell_structure_l225_225352


namespace first_n_digits_same_l225_225152

theorem first_n_digits_same (n : ℕ) : 
  |(5 + Real.sqrt 26)^n - Real.floor ((5 + Real.sqrt 26)^n)| < 10^(-n) :=
begin
  sorry
end

end first_n_digits_same_l225_225152


namespace intersection_points_on_circle_l225_225020

theorem intersection_points_on_circle (u : ℝ) :
  ∃ (r : ℝ), ∀ (x y : ℝ), (u * x - 3 * y - 2 * u = 0) ∧ (2 * x - 3 * u * y + u = 0) → (x^2 + y^2 = r^2) :=
sorry

end intersection_points_on_circle_l225_225020


namespace circle_tangent_to_axes_distance_l225_225432

-- Definitions for the given conditions
def is_tangent_to_axes (c : ℝ × ℝ) (r : ℝ) : Prop :=
  c.1 = r ∧ c.2 = r

def distance_from_center_to_line (c : ℝ × ℝ) : ℝ :=
  (abs (2 * c.1 - c.2 - 3)) / (Real.sqrt (2^2 + (-1)^2))

-- The main theorem
theorem circle_tangent_to_axes_distance
  (c : ℝ × ℝ)
  (r : ℝ)
  (h_tangent : is_tangent_to_axes c r)
  (h_pass : (1 - c.1)^2 + (2 - c.2)^2 = r^2) :
  distance_from_center_to_line c = (2 * Real.sqrt(5)) / 5 :=
sorry

end circle_tangent_to_axes_distance_l225_225432


namespace greatest_three_digit_integer_l225_225009

def is_prime (n : ℕ) : Prop := sorry -- Assume a definition of prime (or use library function if available)

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def n := 995

theorem greatest_three_digit_integer :
  n = 995 ∧ (100 ≤ n ∧ n ≤ 999) ∧ is_prime (n + 2) ∧ ¬ (factorial n % sum_first_n n = 0) :=
by
  -- actual proof will go here
  sorry

end greatest_three_digit_integer_l225_225009


namespace meeting_time_is_correct_l225_225142

-- Definitions
def time_of_meeting (Fedya_start : ℕ) (Fedya_meets_Vera_distance_ratio : ℚ) (Vera_speed_to_Fedya_speed_ratio : ℚ) (Fedya_earlier_meets_Vera_distance_ratio : ℚ) : ℕ :=
  let t := 2 in -- from calculations in the solution steps
  Fedya_start + t

-- Constants/Conditions given in the problem
def Fedya_start_time : ℕ := 9 -- 9 AM
def Fedya_meets_Vera_distance_ratio : ℚ := 1 / 3 -- Fedya covers 1/3 distance by the time they meet
def Vera_speed_to_Fedya_speed_ratio : ℚ := 2 / 1 -- Vera's speed is twice that of Fedya's speed
def Fedya_earlier_meets_Vera_distance_ratio : ℚ := 1 / 2 -- If Fedya leaves 1 hour earlier, he covers 1/2 of the distance

theorem meeting_time_is_correct :
  time_of_meeting Fedya_start_time Fedya_meets_Vera_distance_ratio Vera_speed_to_Fedya_speed_ratio Fedya_earlier_meets_Vera_distance_ratio = 11 := -- represents 11:00 AM
by
  sorry -- proof is omitted

end meeting_time_is_correct_l225_225142


namespace task_completion_time_l225_225666

theorem task_completion_time :
  let rate_A := 1 / 10
  let rate_B := 1 / 15
  let rate_C := 1 / 15
  let combined_rate := rate_A + rate_B + rate_C
  let working_days_A := 2
  let working_days_B := 1
  let rest_day_A := 1
  let rest_days_B := 2
  let work_done_A := rate_A * working_days_A
  let work_done_B := rate_B * working_days_B
  let work_done_C := rate_C * (working_days_A + rest_day_A)
  let work_done := work_done_A + work_done_B + work_done_C
  let remaining_work := 1 - work_done
  let total_days := (work_done / combined_rate) + rest_day_A + rest_days_B
  total_days = 4 + 1 / 7 := by sorry

end task_completion_time_l225_225666


namespace monic_quadratic_with_roots_l225_225731

-- Define the given conditions and roots
def root1 : ℂ := 2 - 3 * complex.i
def root2 : ℂ := 2 + 3 * complex.i

-- Define the polynomial with these roots
def poly (x : ℂ) : ℂ := x^2 - 4 * x + 13

-- Define the proof problem
theorem monic_quadratic_with_roots 
  (h1 : poly root1 = 0) 
  (h2 : poly root2 = 0) 
  (h_monic : ∀ x : ℂ, ∃! p : polynomial ℂ, p.monic ∧ p.coeff 2 = 1) : 
  ∃ p : polynomial ℝ, p.monic ∧ ∀ x : ℂ, p.eval x = poly x := 
sorry

end monic_quadratic_with_roots_l225_225731


namespace cistern_problem_l225_225276

theorem cistern_problem (T : ℝ) (h1 : (1 / 2 - 1 / T) = 1 / 2.571428571428571) : T = 9 :=
by
  sorry

end cistern_problem_l225_225276


namespace ribbons_count_l225_225830

theorem ribbons_count (ribbons : ℕ) 
  (yellow_frac purple_frac orange_frac : ℚ)
  (black_ribbons : ℕ)
  (h1 : yellow_frac = 1/4)
  (h2 : purple_frac = 1/3)
  (h3 : orange_frac = 1/6)
  (h4 : ribbons - (yellow_frac * ribbons + purple_frac * ribbons + orange_frac * ribbons) = black_ribbons) :
  ribbons * orange_frac = 160 / 6 :=
by {
  sorry
}

end ribbons_count_l225_225830


namespace minimize_difference_l225_225321

open BigOperators

/-- Given 16 numbers 1/2002, 1/2003, ..., 1/2017, partition them into two groups A and B such that 
the absolute difference of their sums is minimized. -/
theorem minimize_difference :
  let numbers := (list.range 16).map (λ i, 1 / (2002 + i : ℝ)),
      group1 := [0, 2, 4, 6, 8, 10, 12, 14].map (numbers.get ∘ id),
      group2 := [1, 3, 5, 7, 9, 11, 13, 15].map (numbers.get ∘ id),
      A := group1.sum,
      B := group2.sum
  in |A - B| = minimized :=
begin
  sorry
end

end minimize_difference_l225_225321


namespace find_daily_wage_c_l225_225256

noncomputable def daily_wage_c (A : ℝ) : ℝ := (5 / 3) * A

theorem find_daily_wage_c :
  let A := 81,
      total_earnings := (6 * A) + (9 * (4 / 3) * A) + (4 * (5 / 3) * A) + (8 * (2) * A) + (5 * (7 / 3) * A),
      C := daily_wage_c A in
  total_earnings = 4276 → C = 135 :=
by
  intros
  sorry

end find_daily_wage_c_l225_225256


namespace quadrilateral_perimeter_ge_diagonal_l225_225164

variables {A B C D K L M N : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited K] [Inhabited L] [Inhabited M] [Inhabited N]

-- Definition of a rectangle and its diagonal
structure Rect (A B C D : Type) :=
(length : ℝ)
(width : ℝ)
(diagonal : ℝ := real.sqrt (length ^ 2 + width ^ 2))

-- Definition that quadrilateral vertices lie on the rectangle sides
def inscribed_quadrilateral (R : Rect A B C D) (KLMN : Type) (K : KLMN) (L : KLMN) (M : KLMN) (N : KLMN) : Prop :=
(K ∈ segment R.A R.B ∧
 L ∈ segment R.B R.C ∧
 M ∈ segment R.C R.D ∧
 N ∈ segment R.D R.A)

-- Statement of the problem
theorem quadrilateral_perimeter_ge_diagonal (R : Rect A B C D) (KLMN : Type) (K : KLMN) (L : KLMN) (M : KLMN) (N : KLMN) :
  inscribed_quadrilateral R KLMN K L M N →
  perimeter KLMN ≥ 2 * R.diagonal :=
sorry

end quadrilateral_perimeter_ge_diagonal_l225_225164


namespace train_cross_bridge_time_l225_225602

noncomputable def time_to_cross_bridge (L_train : ℕ) (v_kmph : ℕ) (L_bridge : ℕ) : ℝ :=
  let v_mps := (v_kmph * 1000) / 3600
  let total_distance := L_train + L_bridge
  total_distance / v_mps

theorem train_cross_bridge_time :
  time_to_cross_bridge 145 54 660 = 53.67 := by
    sorry

end train_cross_bridge_time_l225_225602


namespace count_false_statements_l225_225317

-- Definition of the problem conditions as propositions
def prop1 := ∀ (l1 l2 l3 : ℕ), equal_angles_with_third_line l1 l2 l3 → parallel l1 l2
def prop2 := ∀ (l1 l2 l3 : ℕ), perpendicular_to_third_line l1 l3 ∧ perpendicular_to_third_line l2 l3 → parallel l1 l2
def prop3 := ∀ (l1 l2 l3 : ℕ), parallel_to_third_line l1 l3 ∧ parallel_to_third_line l2 l3 → parallel l1 l2

-- The goal to prove the correct number of false propositions
theorem count_false_statements : (¬ prop1) ∧ (¬ prop2) ∧ prop3 → count_false (prop1, prop2, prop3) = 2 := by 
  sorry

end count_false_statements_l225_225317


namespace minimum_value_PA_PF_l225_225410

-- Definitions based on conditions
def parabola (p : ℝ) (p_gt_zero : p > 0) : set (ℝ × ℝ) := {y | ∃ x, y.2^2 = 2*p*x}
def focus (p : ℝ) : (ℝ × ℝ) := (p/2, 0)
def point_A : ℝ × ℝ := (4, 1)

-- The mathematically equivalent proof problem in Lean 4
theorem minimum_value_PA_PF (p : ℝ) (p_gt_zero : p > 0) (h1 : ∀ M ∈ parabola p p_gt_zero, dist M (focus p) = 2)
  (P : ℝ × ℝ) (h2 : P ∈ parabola p p_gt_zero) : 
  ∃ D : ℝ × ℝ, D.1 < P.1 ∧ D.2 = 0 ∧ dist P (focus p) = dist P D ∧ dist point_A P + dist point_A D = 6 :=
sorry

end minimum_value_PA_PF_l225_225410


namespace correct_option_l225_225240

-- Definition of the exponentiation rules for the given options.
def optionA (a b : ℝ) : Prop := (a * b)^5 = a * b^5
def optionB (a : ℝ) : Prop := a^8 / a^2 = a^6
def optionC (a : ℝ) : Prop := (a^2)^3 = a^5
def optionD (a : ℝ) : Prop := a^2 * a^3 = a^6

-- Main theorem stating the correctness of optionB.
theorem correct_option (a b : ℝ) : optionB a :=
begin
  sorry  -- Proof goes here
end

end correct_option_l225_225240


namespace solution_set_absolute_value_l225_225200

theorem solution_set_absolute_value (x : ℝ) : 
  (|2 - x| ≥ 1) ↔ (x ≤ 1 ∨ x ≥ 3) :=
by
  -- Proof goes here
  sorry

end solution_set_absolute_value_l225_225200


namespace exists_plane_not_through_subset_l225_225104

noncomputable def points_in_general_position (M : set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ {p1 p2 p3 p4 : ℝ × ℝ × ℝ} (h1 : p1 ∈ M) (h2 : p2 ∈ M) (h3 : p3 ∈ M) (h4 : p4 ∈ M),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1 → ¬ collinear (p1, p2, p3) ∧ ¬ coplanar (p1, p2, p3, p4)

theorem exists_plane_not_through_subset
  {n : ℕ} (h : 5 ≤ n) (M : set (ℝ × ℝ × ℝ))
  (hM : points_in_general_position M) (hcard : M.card = n) :
  ∀ (A : set (ℝ × ℝ × ℝ)), A.card = n - 3 → 
    ∃ p1 p2 p3 : ℝ × ℝ × ℝ, {p1, p2, p3} ⊆ M ∧ ¬ ∃ x ∈ A, x ∈ (p1, p2, p3) :=
by
  sorry

end exists_plane_not_through_subset_l225_225104


namespace midpoint_CF_l225_225453

theorem midpoint_CF
  (O A B P C D F E : Type)
  (h1 : tangent PA (circle O) A)
  (h2 : tangent PB (circle O) B)
  (h3 : secant PCD (circle O))
  (h4 : parallel CF PB)
  (h5 : intersects CF AB E)
  (h6 : intersects CF BD F) :
  midpoint E CF :=
sorry

end midpoint_CF_l225_225453


namespace find_n_l225_225283

theorem find_n (n a b : ℕ) 
  (h1 : a > 1)
  (h2 : a ∣ n)
  (h3 : b > a)
  (h4 : b ∣ n)
  (h5 : ∀ m, 1 < m ∧ m < a → ¬ m ∣ n)
  (h6 : ∀ m, a < m ∧ m < b → ¬ m ∣ n)
  (h7 : n = a^a + b^b)
  : n = 260 :=
by sorry

end find_n_l225_225283


namespace simplify_cot10_tan5_l225_225158

theorem simplify_cot10_tan5 : Real.cot 10 + Real.tan 5 = Real.csc 10 := by
  sorry

end simplify_cot10_tan5_l225_225158


namespace book_club_choices_l225_225893

theorem book_club_choices :
  ∃ (n : ℕ), n = 5 ∧ (∃ (ways : ℕ), ways = Nat.choose 5 3 ∧ ways = 10) :=
by
  use 5
  constructor
  · rfl
  use Nat.choose 5 3
  constructor
  · rfl
  · have h : Nat.choose 5 3 = 10 := by
      calc
        Nat.choose 5 3 = 5! / (3! * 2!) : by sorry
        _ = 10 : by sorry
    exact h

end book_club_choices_l225_225893


namespace sitting_people_l225_225707

variables {M I P A : Prop}

-- Conditions
axiom M_not_sitting : ¬ M
axiom A_not_sitting : ¬ A
axiom if_M_not_sitting_then_I_sitting : ¬ M → I
axiom if_I_sitting_then_P_sitting : I → P

theorem sitting_people : I ∧ P :=
by
  have I_sitting : I := if_M_not_sitting_then_I_sitting M_not_sitting
  have P_sitting : P := if_I_sitting_then_P_sitting I_sitting
  exact ⟨I_sitting, P_sitting⟩

end sitting_people_l225_225707


namespace valid_conditions_x_y_z_l225_225970

theorem valid_conditions_x_y_z (x y z : ℤ) :
  x = y - 1 ∧ z = y + 1 ∨ x = y ∧ z = y + 1 ↔ x * (x - y) + y * (y - x) + z * (z - y) = 1 :=
sorry

end valid_conditions_x_y_z_l225_225970


namespace real_roots_of_quadratic_l225_225077

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem real_roots_of_quadratic (m : ℝ) : (∃ x : ℝ, x^2 - x - m = 0) ↔ m ≥ -1/4 := by
  sorry

end real_roots_of_quadratic_l225_225077


namespace equation_solution_count_l225_225927

theorem equation_solution_count :
  let y := (x + 1)^2,
      xy_y := 1 in
  (exists (x_real : ℝ), (y = (x_real + 1)^2) /\ (x_real * y + y = 1)) /\
  (count_solutions (λ x : ℂ, (y = (x + 1)^2) /\ (x * y + y = 1)) = 3) /\
  (count_solutions (λ x : ℝ, (y = (x + 1)^2) /\ (x * y + y = 1)) = 1) :=
  sorry

-- Helper function to count the solutions in a given domain
def count_solutions {α : Type*} (f : α → Prop) : ℕ :=
  sorry

end equation_solution_count_l225_225927


namespace sum_of_angles_FC_correct_l225_225528

noncomputable def circleGeometry (A B C D E F : Point)
  (onCircle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E)
  (arcAB : ℝ) (arcDE : ℝ) : Prop :=
  let arcFull := 360;
  let angleF := 6;  -- Derived from the intersecting chords theorem
  let angleC := 36; -- Derived from the inscribed angle theorem
  arcAB = 60 ∧ arcDE = 72 ∧
  0 ≤ angleF ∧ 0 ≤ angleC ∧
  angleF + angleC = 42

theorem sum_of_angles_FC_correct (A B C D E F : Point) 
  (onCircle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E) :
  circleGeometry A B C D E F onCircle 60 72 :=
by
  sorry  -- Proof to be filled

end sum_of_angles_FC_correct_l225_225528


namespace LCM_of_two_numbers_l225_225982

theorem LCM_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 14) (h2 : a * b = 2562) : Nat.lcm a b = 183 :=
by
  sorry

end LCM_of_two_numbers_l225_225982


namespace monotonicity_of_f_range_of_t_l225_225403

noncomputable def f (x : ℝ) : ℝ := x^2 / Real.log x

theorem monotonicity_of_f :
  (∀ x ∈ Ioo (0 : ℝ) 1, deriv f x < 0) ∧ 
  (∀ x ∈ Ioo (1 : ℝ) (Real.sqrt Real.exp 1), deriv f x < 0) ∧ 
  (∀ x ∈ Ioi (Real.sqrt Real.exp 1), deriv f x > 0) :=
sorry

noncomputable def g (x : ℝ) : ℝ := Real.log x / x

theorem range_of_t (t : ℝ) :
  (∃ x1 x2 ∈ Icc (1 / Real.exp 1 : ℝ) 1 ∪ Ioc 1 (Real.exp 2), 
    tf x1 - x1 = 0 ∧ tf x2 - x2 = 0 ∧ x1 ≠ x2) →
  t ∈ Icc (2 / Real.exp 2 ^ 2) (1 / Real.exp 1) :=
sorry

end monotonicity_of_f_range_of_t_l225_225403


namespace rectangle_problem_l225_225532

theorem rectangle_problem 
  (AB BC EB CG DF : ℝ)
  (r1 : AB = 8)
  (r2 : BC = 6)
  (r3 : EB = 3)
  (r4 : CG = 3)
  (r5 : DF = 4) :
  let A := (0, 6), B := (8, 6), C := (8, 0), D := (0, 0),
      E := (5, 6), F := (4, 0), G := (8, 3),
      P := (40 / 27, 20 / 27), Q := (240 / 51, 80 / 17),
      EF := Real.sqrt ((5 - 4)^2 + (6 - 0)^2),
      PQ := abs ((240 / 51) - (40 / 27))
  in PQ / EF = 1040 / (1377 * Real.sqrt 37) :=
by
  -- omitted proof steps
  sorry

end rectangle_problem_l225_225532


namespace goldie_worked_hours_last_week_l225_225421

variable (H : ℕ)
variable (money_per_hour : ℕ := 5)
variable (hours_this_week : ℕ := 30)
variable (total_earnings : ℕ := 250)

theorem goldie_worked_hours_last_week :
  H = (total_earnings - hours_this_week * money_per_hour) / money_per_hour :=
sorry

end goldie_worked_hours_last_week_l225_225421


namespace tiling_23x23_with_2x2_3x3_tiling_23x23_with_2x2_3x3_1x1_l225_225850

noncomputable def tiling_possible_without_1x1 (n : ℕ) : Prop :=
n = 23 ∧ ¬ is_tilable_with_2x2_3x3 n

noncomputable def tiling_possible_with_1x1 (n : ℕ) : Prop :=
n = 23 ∧ is_tilable_with_2x2_3x3_1x1 n

axiom is_tilable_with_2x2_3x3 : ℕ → Prop
axiom is_tilable_with_2x2_3x3_1x1 : ℕ → Prop

theorem tiling_23x23_with_2x2_3x3 : tiling_possible_without_1x1 23 :=
by {
    have h1 : 23 = 23, from rfl,
    have h2 : ¬ is_tilable_with_2x2_3x3 23, from sorry,
    exact ⟨h1, h2⟩
}

theorem tiling_23x23_with_2x2_3x3_1x1 : tiling_possible_with_1x1 23 :=
by {
    have h1 : 23 = 23, from rfl,
    have h2 : is_tilable_with_2x2_3x3_1x1 23, from sorry,
    exact ⟨h1, h2⟩
}

end tiling_23x23_with_2x2_3x3_tiling_23x23_with_2x2_3x3_1x1_l225_225850


namespace problem_solution_l225_225508

theorem problem_solution (y : Fin 50 → ℝ)
  (h1 : (∑ i, y i) = 2)
  (h2 : (∑ i, y i / (2 - y i)) = 2) :
  (∑ i, y i * y i / (2 - y i)) = 0 := 
sorry

end problem_solution_l225_225508


namespace intersect_midpoints_l225_225510

variables {A B C D P : Type}
variables [real_vector_space A] [real_vector_space B] [real_vector_space C] [real_vector_space D]
variables (a b c d : A)

-- Definitions of points being the midpoints
def is_midpoint {X Y Z : A} (m : A) : Prop := 2 • m = X + Y 

-- Segment joining midpoints
def segment_midpoints (X Y Z W : A) (k l m n : A) : Prop :=
  is_midpoint X Y k ∧ is_midpoint Y Z l ∧ is_midpoint Z W m ∧ is_midpoint W X n

-- The statement
theorem intersect_midpoints (a b c d k l m n : A)
  (h_non_coplanar : ¬ (coplanar {a, b, c, d}))
  (h_mid_segs : segment_midpoints a b c d k l m n) :
  ∃ p : A, is_midpoint k m p ∧ is_midpoint l n p := by
  sorry

end intersect_midpoints_l225_225510


namespace fred_newspaper_earnings_l225_225858

def fred_weekend_earnings : ℕ := 90
def fred_washing_cars_earnings : ℕ := 74

theorem fred_newspaper_earnings :
  ∃ x : ℕ, x = fred_weekend_earnings - fred_washing_cars_earnings ∧ x = 16 :=
by
  use 16
  simp [fred_weekend_earnings, fred_washing_cars_earnings]
  sorry

end fred_newspaper_earnings_l225_225858


namespace seating_arrangement_count_l225_225956

/-- Twelve chairs are evenly spaced around a round table and numbered clockwise from 1 through 12.
Six married couples are to sit in the chairs with men and women alternating, and no one sits either
next to or across from their spouse. Prove that the number of such seating arrangements is 288.
-/
theorem seating_arrangement_count : 
  let chairs := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let men := [m1, m2, m3, m4, m5, m6] in
  let women := [w1, w2, w3, w4, w5, w6] in
  let arrangements := {arr | 
    alternating_seats arr ∧ 
    (∀ i j, spouses arr i j → ¬adjacent_or_across arr i j)} in
  arrangements.card = 288 :=
by
  sorry

end seating_arrangement_count_l225_225956


namespace germs_per_dish_l225_225603

theorem germs_per_dish (total_germs : ℝ) (num_petri_dishes : ℝ) (h1 : total_germs = 0.036 * 10^5) (h2 : num_petri_dishes = 18000 * 10^(-3)) : 
  total_germs / num_petri_dishes = 200 := 
by 
  -- placeholder for the proof
  sorry

end germs_per_dish_l225_225603


namespace minimum_n_exists_l225_225542

theorem minimum_n_exists : ∃ n : ℕ, (∀ x y z : ℝ, x^2 + y^2 + z^2 ≤ n * (x^4 + y^4 + z^4)) ∧ ∀ m : ℕ, (∀ x y z : ℝ, x^2 + y^2 + z^2 ≤ m * (x^4 + y^4 + z^4)) → n ≤ m :=
begin
  use 3,
  split,
  {
    intros x y z,
    -- Proof of (x^2 + y^2 + z^2) ≤ 3 * (x^4 + y^4 + z^4)
    sorry
  },
  {
    intros m hm,
    have h := hm 1 1 1,
    -- Using example x = y = z = 1 to compare n and m
    calc 3 : ℕ ≤ n : ℕ := sorry,
  },
end

end minimum_n_exists_l225_225542


namespace range_of_m_l225_225058

def P := {x : ℝ | x^2 ≤ 4}
def M (m : ℝ) := {m}
def condition (m : ℝ) := M m ⊆ P

theorem range_of_m (m : ℝ) (h : condition m) : -2 ≤ m ∧ m ≤ 2 := by
  unfold condition at h
  unfold M at h
  dsimp at h
  have h1 : m ∈ P := h rfl
  unfold P at h1
  rw set.mem_setOf_eq at h1
  exact ⟨real.sqrt_nonneg _ ▸ le_of_sq_nonneg h1, h1⟩

set_option linter.unusedVariables false

end range_of_m_l225_225058


namespace egg_problem_l225_225250

theorem egg_problem (c : ℕ) (h1 : 15 * c - 3 > 200) : 15 * c - 3 = 207 :=
by
  have h2 : c ≥ 14 := sorry
  have h3 : 15 * 14 - 3 = 207 := rfl
  exact h3
sorry

end egg_problem_l225_225250


namespace sequence_characterization_l225_225687

theorem sequence_characterization (n : ℕ) (x : fin (n+2) → ℕ) :
  (∀ k, x k ≤ n) ∧ (∀ k, 0 ≤ x k) ∧ (∑ k, x k = n + 1) →
  (x = (λ k, if k = 0 then 2 else if k = 2 then 2 else 0) ∨
   x = (λ k, if k = 0 then 1 else if k = 1 then 2 else if k = 2 then 1 else 0) ∨
   (∃ m, m ≥ 3 ∧ x = (λ k, if k = 0 then m else if k = 1 then 2 else if k = 2 then 1 
                                 else if k = m then 1 else 0))) :=
sorry

end sequence_characterization_l225_225687


namespace distance_traveled_by_bus_l225_225643

noncomputable def total_distance : ℕ := 900
noncomputable def distance_by_plane : ℕ := total_distance / 3
noncomputable def distance_by_bus : ℕ := 360
noncomputable def distance_by_train : ℕ := (2 * distance_by_bus) / 3

theorem distance_traveled_by_bus :
  distance_by_plane + distance_by_train + distance_by_bus = total_distance :=
by
  sorry

end distance_traveled_by_bus_l225_225643


namespace length_of_platform_is_correct_l225_225600

noncomputable def length_of_platform : ℝ :=
  let train_length := 200 -- in meters
  let train_speed := 80 * 1000 / 3600 -- kmph to m/s
  let crossing_time := 22 -- in seconds
  (train_speed * crossing_time) - train_length

theorem length_of_platform_is_correct :
  length_of_platform = 2600 / 9 :=
by 
  -- proof would go here
  sorry

end length_of_platform_is_correct_l225_225600


namespace total_potatoes_l225_225467

theorem total_potatoes (jane_potatoes mom_potatoes dad_potatoes : Nat) 
  (h1 : jane_potatoes = 8)
  (h2 : mom_potatoes = 8)
  (h3 : dad_potatoes = 8) :
  jane_potatoes + mom_potatoes + dad_potatoes = 24 :=
by
  sorry

end total_potatoes_l225_225467


namespace find_set_A_l225_225413

def M : Set ℤ := {1, 3, 5, 7, 9}

def satisfiesCondition (A : Set ℤ) : Prop :=
  A ≠ ∅ ∧
  (∀ a ∈ A, a + 4 ∈ M) ∧
  (∀ a ∈ A, a - 4 ∈ M)

theorem find_set_A : ∃ A : Set ℤ, satisfiesCondition A ∧ A = {5} :=
  by
    sorry

end find_set_A_l225_225413


namespace yellow_marbles_count_l225_225472

theorem yellow_marbles_count 
  (total_marbles red_marbles blue_marbles : ℕ) 
  (h_total : total_marbles = 85) 
  (h_red : red_marbles = 14) 
  (h_blue : blue_marbles = 3 * red_marbles) :
  (total_marbles - (red_marbles + blue_marbles)) = 29 :=
by
  sorry

end yellow_marbles_count_l225_225472


namespace even_diagonal_pluses_l225_225838

-- Definitions based on the conditions of the problem
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Given conditions as Lean definitions
variable (grid : ℕ → ℕ → bool)
variable h_total_even : is_even (Finset.card (Finset.filter (λ (c : ℕ × ℕ), grid c.1 c.2 = tt) (Finset.range 11 × Finset.range 11)))
variable h_sub_square_even : ∀ i j : ℕ, i < 10 → j < 10 → is_even (Finset.card (Finset.filter (λ (c : ℕ × ℕ), grid c.1 c.2 = tt) (Finset.range i 2 × Finset.range j 2)))

-- Statement to prove
theorem even_diagonal_pluses (grid : ℕ → ℕ → bool)
  (h_total_even : is_even (Finset.card (Finset.filter (λ (c : ℕ × ℕ), grid c.1 c.2 = tt) (Finset.range 11 × Finset.range 11))))
  (h_sub_square_even : ∀ i j : ℕ, i < 10 → j < 10 → is_even (Finset.card (Finset.filter (λ (c : ℕ × ℕ), grid c.1 c.2 = tt) (Finset.range i 2 × Finset.range j 2)))) :
  is_even (Finset.card (Finset.filter (λ (c : ℕ × ℕ), c.1 = c.2 ∧ grid c.1 c.2 = tt) (Finset.range 11 × Finset.range 11))) :=
sorry

end even_diagonal_pluses_l225_225838


namespace cut_scene_length_l225_225630

theorem cut_scene_length (original_length final_length : ℕ) (h1 : original_length = 60) (h2 : final_length = 52) : original_length - final_length = 8 := 
by 
  sorry

end cut_scene_length_l225_225630


namespace carolyn_silverware_knives_percentage_correct_l225_225304

-- Define initial quantities
def initial_knives : ℕ := 6
def initial_forks : ℕ := 12
def initial_spoons : ℕ := 18
def initial_teaspoons : ℕ := 2 * initial_forks

-- Define changes after trades
def knives_after_trade1 : ℤ := initial_knives - 10
def forks_after_trade1 : ℕ := initial_forks + 6
def teaspoons_after_trade1 : ℕ := initial_teaspoons + 4

def forks_after_trade2 : ℕ := forks_after_trade1 - 8
def spoons_after_trade2 : ℕ := initial_spoons + 3
def teaspoons_after_trade2 : ℕ := teaspoons_after_trade1 + 5

-- Define changes after purchase
def final_knives : ℤ := knives_after_trade1 + 7
def final_forks : ℕ := forks_after_trade2 + 5
def final_teaspoons : ℕ := teaspoons_after_trade2 + 12
def final_spoons : ℕ := spoons_after_trade2

-- Calculate total silverware
def total_silverware : ℕ := (final_knives.toNat : ℕ) + final_forks + final_spoons + final_teaspoons

-- Calculate percentage of knives
def percentage_of_knives : ℚ := (final_knives.toNat:ℚ) * 100 / total_silverware.toRat

theorem carolyn_silverware_knives_percentage_correct : percentage_of_knives = 3.57 := 
by
  sorry

end carolyn_silverware_knives_percentage_correct_l225_225304


namespace n_fraction_sum_l225_225976

theorem n_fraction_sum {n : ℝ} {lst : List ℝ} (h_len : lst.length = 21) 
(h_mem : n ∈ lst) 
(h_avg : n = 4 * (lst.erase n).sum / 20) :
  n = (lst.sum) / 6 :=
by
  sorry

end n_fraction_sum_l225_225976


namespace problem_area_triangle_RSO_l225_225836

noncomputable def area_triangle_RSO (E F G H R N S M O : Point) (p : ℝ) : Prop :=
  Parallelogram E F G H ∧
  divides_in_ratio E R F G 2 1 at N ∧
  meets_extended E H R ∧
  bisects G S E H at M ∧
  meets_extended G S F at S ∧
  intersects E R G S at O ∧
  parallelogram_area E F G H = p → 
  triangle_area R S O = 49 * p / 36

-- Statement of the problem as a proof objective
theorem problem_area_triangle_RSO
  (E F G H R N S M O : Point) (p : ℝ)
  (h_parallelogram : Parallelogram E F G H)
  (h_divides : divides_in_ratio E R F G 2 1 at N)
  (h_meets1 : meets_extended E H R)
  (h_bisect : bisects G S E H at M)
  (h_meets2 : meets_extended G S F at S)
  (h_intersect : intersects E R G S at O)
  (h_area : parallelogram_area E F G H = p) :
  triangle_area R S O = 49 * p / 36 :=
sorry

end problem_area_triangle_RSO_l225_225836


namespace max_min_f_l225_225949

def f (x1 x2 x3 : ℝ) : ℝ :=
  Real.sqrt (x1^2 + x2 * x3) + Real.sqrt (x2^2 + x1 * x3) + Real.sqrt (x3^2 + x1 * x2)

theorem max_min_f (x1 x2 x3 : ℝ) (h1 : x1 ≥ 0) (h2 : x2 ≥ 0) (h3 : x3 ≥ 0) (h_sum : x1 + x2 + x3 ≤ 2) :
    (∃ x1 x2 x3, f x1 x2 x3 = 0) ∧ (∃ x1 x2 x3, f x1 x2 x3 = 3) :=
by {
    sorry
}

end max_min_f_l225_225949


namespace cos_2theta_eq_neg1_l225_225372

theorem cos_2theta_eq_neg1 (θ : ℝ) (h : 2 ^ (-5/3 + sin (2*θ)) + 2 = 2 ^ (1/3 + sin θ)) : cos (2 * θ) = -1 :=
sorry

end cos_2theta_eq_neg1_l225_225372


namespace cookies_per_bag_l225_225888

theorem cookies_per_bag
  (chocolate_chip_cookies : ℕ) (oatmeal_cookies : ℕ) (baggies : ℕ)
  (h1 : chocolate_chip_cookies = 33) (h2 : oatmeal_cookies = 2) (h3 : baggies = 7) :
  (chocolate_chip_cookies + oatmeal_cookies) / baggies = 5 :=
  by
  let total_cookies := chocolate_chip_cookies + oatmeal_cookies
  have h_total : total_cookies = 35 := by
    rw [h1, h2]
    norm_num
  have h_div : total_cookies / baggies = 5 := by
    rw [h3, h_total]
    norm_num
  exact h_div

end cookies_per_bag_l225_225888


namespace polynomial_remainder_l225_225877

noncomputable def h (x : ℕ) := x^5 + x^4 + x^3 + x^2 + x + 1

theorem polynomial_remainder (x : ℕ) : (h (x^10)) % (h x) = 5 :=
sorry

end polynomial_remainder_l225_225877


namespace students_all_three_classes_l225_225826

variables (H M E HM HE ME HME : ℕ)

-- Conditions from the problem
def student_distribution : Prop :=
  H = 12 ∧
  M = 17 ∧
  E = 36 ∧
  HM + HE + ME = 3 ∧
  86 = H + M + E - (HM + HE + ME) + HME

-- Prove the number of students registered for all three classes
theorem students_all_three_classes (h : student_distribution H M E HM HE ME HME) : HME = 24 :=
  by sorry

end students_all_three_classes_l225_225826


namespace find_monic_quadratic_real_coefficients_l225_225739

-- Define the necessary conditions
def is_monic_quadratic (f : ℝ[X]) : Prop :=
  f.degree = Polynomial.degree (Polynomial.X ^ 2) ∧ f.leadingCoeff = 1

def has_real_coefficients (f : ℝ[X]) : Prop :=
  ∀ n, (Polynomial.coeff f n).im = 0

def root_exists (f : ℝ[X]) (z : ℂ) : Prop :=
  f.eval z = 0

-- The main theorem statement
theorem find_monic_quadratic_real_coefficients (z : ℂ) (f : ℝ[X]) (h : z = 2 - 3 * I) :
  is_monic_quadratic f ∧ has_real_coefficients f ∧ root_exists f z ∧ root_exists f z.conj ↔ f = Polynomial.C 13 + Polynomial.X * (Polynomial.X - 4) :=
sorry

end find_monic_quadratic_real_coefficients_l225_225739


namespace compare_neg_frac1_l225_225267

theorem compare_neg_frac1 : (-3 / 7 : ℝ) < (-8 / 21 : ℝ) :=
sorry

end compare_neg_frac1_l225_225267


namespace value_of_f_log2_3_2_min_value_of_f_l225_225870

-- Define the function f(x)
noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then 2^(-x) else (Real.logBase 3 (x / 3)) * (Real.logBase 3 (x / 9))

-- Prove f(log2(3/2)) = 2/3
theorem value_of_f_log2_3_2 : f (Real.logBase 2 (3 / 2)) = 2 / 3 :=
by
  sorry

-- Prove the minimum value of f(x) is -1/4
theorem min_value_of_f : ∃ x : ℝ, f x = -1 / 4 :=
by
  sorry

end value_of_f_log2_3_2_min_value_of_f_l225_225870


namespace trumpet_fraction_l225_225273

theorem trumpet_fraction (T_Trombone : ℝ) (T_Trumpet_or_Trombone : ℝ) (h1 : T_Trombone = 0.12) (h2 : T_Trumpet_or_Trombone = 0.63) : 
  ∃ T_Trumpet : ℝ, T_Trumpet + T_Trombone = T_Trumpet_or_Trombone ∧ T_Trumpet = 0.51 :=
by
  use 0.51
  split
  · rw [h1, h2]
    norm_num
  · norm_num
  sorry

end trumpet_fraction_l225_225273


namespace sum_of_altitudes_correct_l225_225552

noncomputable def sum_of_altitudes {x y : ℝ} (hx : 10 * x + 3 * y = 30) (hx_pos : x > 0) (hy_pos : y > 0) : ℝ :=
  let vertex1 : (ℝ × ℝ) := (0, 0)
  let vertex2 : (ℝ × ℝ) := (x, 0)
  let vertex3 : (ℝ × ℝ) := (0, y)
  let height1 := x
  let height2 := y
  let height3 := 30 / (Real.sqrt (10 ^ 2 + 3 ^ 2))
  height1 + height2 + height3

theorem sum_of_altitudes_correct : sum_of_altitudes (by linarith) (by linarith) (by linarith) = 13 + 30 / Real.sqrt 109 :=
  sorry

end sum_of_altitudes_correct_l225_225552


namespace cylinder_height_comparison_l225_225960

theorem cylinder_height_comparison (r1 h1 r2 h2 : ℝ)
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
by {
  -- Proof steps here, not required per instruction
  sorry
}

end cylinder_height_comparison_l225_225960


namespace find_f_pi_over_3_l225_225042

def f (x : ℝ) : ℝ := Real.sin (2 * (x + π / 6) + π / 6)

theorem find_f_pi_over_3 : f (π / 3) = -1 / 2 :=
by
  sorry

end find_f_pi_over_3_l225_225042


namespace incorrect_statement_C_l225_225894

theorem incorrect_statement_C :
  (∀ (b h : ℝ), b > 0 → h > 0 → 2 * (b * h) = (2 * b) * h) ∧
  (∀ (r h : ℝ), r > 0 → h > 0 → 2 * (π * r^2 * h) = π * r^2 * (2 * h)) ∧
  (∀ (a : ℝ), a > 0 → 4 * (a^3) ≠ (2 * a)^3) ∧
  (∀ (a b : ℚ), b ≠ 0 → a / (2 * b) ≠ (a / 2) / b) ∧
  (∀ (x : ℝ), x < 0 → 2 * x < x) :=
by
  sorry

end incorrect_statement_C_l225_225894


namespace hexagon_area_l225_225229

/-- Define a circle with a given radius and an inscribed regular hexagon. -/
def area_hexagon_in_circle (r : ℝ) (area_circle : ℝ) : ℝ :=
  if area_circle = 400 * Real.pi ∧ r = 20 then
    6 * ((r^2 * Real.sqrt 3) / 4)
  else
    0

theorem hexagon_area (r : ℝ) (area_circle : ℝ) : 
  area_circle = 400 * Real.pi ∧ r = 20 → area_hexagon_in_circle r area_circle = 600 * Real.sqrt 3 :=
by
  sorry

end hexagon_area_l225_225229


namespace sufficient_not_necessary_condition_l225_225612

theorem sufficient_not_necessary_condition : 
  ∀ x : ℝ, (x = 2 → (x + 1) * (x - 2) = 0) ∧ ((x + 1) * (x - 2) = 0 → (x = 2 ∨ x = -1)) :=
by
  intro x
  constructor
  · intro h
    rw [h]
    norm_num
  · intro h
    cases h
    · left
      assumption
    · right
      assumption
  sorry

end sufficient_not_necessary_condition_l225_225612


namespace exists_positive_integers_for_system_l225_225694

theorem exists_positive_integers_for_system :
  ∃ (x : Fin 1985 → ℕ) (y z : ℕ),
    (∀ i j : Fin 1985, i ≠ j → x i ≠ x j) ∧
    (∑ i, (x i)^2 = y^3) ∧
    (∑ i, (x i)^3 = z^2) :=
by
  sorry

end exists_positive_integers_for_system_l225_225694


namespace Sara_spent_on_hotdog_l225_225912

-- Define the given constants
def totalCost : ℝ := 10.46
def costSalad : ℝ := 5.10

-- Define the value we need to prove
def costHotdog : ℝ := 5.36

-- Statement to prove
theorem Sara_spent_on_hotdog : totalCost - costSalad = costHotdog := by
  sorry

end Sara_spent_on_hotdog_l225_225912


namespace same_candy_probability_l225_225628

noncomputable def candy_probability : ℚ :=
let total_permutations := (20.choose 3 : ℚ),
    alice_picks_three_red := (12.choose 3 : ℚ) / total_permutations,
    alice_picks_two_red_one_green := ((12.choose 2 : ℕ) * (8.choose 1 : ℕ) : ℚ) / total_permutations,
    bob_picks_three_red := (9.choose 3 : ℚ) / (17.choose 3 : ℚ),
    bob_picks_two_red_one_green := ((10.choose 2 : ℕ) * (7.choose 1 : ℕ) : ℚ) / (17.choose 3 : ℚ) in
alice_picks_three_red * bob_picks_three_red + alice_picks_two_red_one_green * bob_picks_two_red_one_green

theorem same_candy_probability :
    candy_probability = 231 / 1060 :=
sorry

end same_candy_probability_l225_225628


namespace smallest_n_for_g_gt_15_l225_225071

noncomputable def g (n : ℕ) : ℕ := 
  let x := (10^n : ℚ) / (3^n : ℚ)
  let digits_after_dp := (x - x.floor) * (10^n)
  list.sum (digits_after_dp.digits 10)

theorem smallest_n_for_g_gt_15 :
  ∃ n : ℕ, n > 0 ∧ g(n) > 15 ∧ ∀ m : ℕ, m < n → g(m) ≤ 15 :=
by
  sorry

end smallest_n_for_g_gt_15_l225_225071


namespace intersection_of_sets_l225_225062

variable {x : ℝ}

def SetA : Set ℝ := {x | x + 1 > 0}
def SetB : Set ℝ := {x | x - 3 < 0}

theorem intersection_of_sets : SetA ∩ SetB = {x | -1 < x ∧ x < 3} :=
by sorry

end intersection_of_sets_l225_225062


namespace find_x1_l225_225398

noncomputable def parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

theorem find_x1 
  (a h k m x1 : ℝ)
  (h1 : parabola a h k (-1) = 2)
  (h2 : parabola a h k 1 = -2)
  (h3 : parabola a h k 3 = 2)
  (h4 : parabola a h k (-2) = m)
  (h5 : parabola a h k x1 = m) :
  x1 = 4 := 
sorry

end find_x1_l225_225398


namespace monic_quadratic_with_roots_l225_225734

-- Define the given conditions and roots
def root1 : ℂ := 2 - 3 * complex.i
def root2 : ℂ := 2 + 3 * complex.i

-- Define the polynomial with these roots
def poly (x : ℂ) : ℂ := x^2 - 4 * x + 13

-- Define the proof problem
theorem monic_quadratic_with_roots 
  (h1 : poly root1 = 0) 
  (h2 : poly root2 = 0) 
  (h_monic : ∀ x : ℂ, ∃! p : polynomial ℂ, p.monic ∧ p.coeff 2 = 1) : 
  ∃ p : polynomial ℝ, p.monic ∧ ∀ x : ℂ, p.eval x = poly x := 
sorry

end monic_quadratic_with_roots_l225_225734


namespace sum_even_integers_gt_5_lt_20_l225_225235

theorem sum_even_integers_gt_5_lt_20 : 
  (∑ i in finset.filter (λ x, x % 2 = 0) (finset.Ico 6 20), i) = 84 :=
by
  sorry

end sum_even_integers_gt_5_lt_20_l225_225235


namespace find_missing_number_l225_225327

theorem find_missing_number (x : ℕ) (h : 10111 - 10 * 2 * x = 10011) : x = 5 :=
sorry

end find_missing_number_l225_225327


namespace square_distance_B_center_l225_225641

open Real

section
variable (A B C : Point) (O : Point)
variable (r : ℝ) (radius_eq : r = 8)
variable (AB_eq : dist A B = 8)
variable (BC_eq : dist B C = 4)
variable (angle_ABC_is_right : ∠ ABC = π / 2)
variable (circle_center : O = (0, 0))
variable (B_on_circle : dist B O = r)
variable (circle_radius : r^2 = 64)
variable (circle_eq : ∀ P, dist P O = r ↔ P.x^2 + P.y^2 = 64)

theorem square_distance_B_center :
  let B := Point.mk (-2) 0 in
  let square_distance := (-2)^2 + 0^2 in
  square_distance = 4 :=
by sorry

end

end square_distance_B_center_l225_225641


namespace maximum_t_l225_225809

theorem maximum_t {a b t : ℝ} (ha : 0 < a) (hb : a < b) (ht : b < t)
  (h_condition : b * Real.log a < a * Real.log b) : t ≤ Real.exp 1 :=
sorry

end maximum_t_l225_225809


namespace find_five_value_l225_225179

def f (x : ℝ) : ℝ := x^2 - x

theorem find_five_value : f 5 = 20 := by
  sorry

end find_five_value_l225_225179


namespace find_a_in_binomial_expansion_l225_225841

theorem find_a_in_binomial_expansion (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, (x ≠ 0) → 
  (let expr := (sqrt x + (a / x)) ^ 6 in
  (∃ (r : ℕ), r = 2 ∧ a^2 * (Nat.choose 6 2) = 60))) → a = 2 :=
by
  intros x hx expr h_const_term
  sorry

end find_a_in_binomial_expansion_l225_225841


namespace intersecting_lines_l225_225083

theorem intersecting_lines (m : ℝ) :
  (∃ (x y : ℝ), y = 2 * x ∧ x + y = 3 ∧ m * x + 2 * y + 5 = 0) ↔ (m = -9) :=
by
  sorry

end intersecting_lines_l225_225083


namespace find_f_prime_at_1_l225_225389

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x * (deriv f 1) - 1
def f_prime (x : ℝ) : ℝ := deriv f x

theorem find_f_prime_at_1 : f_prime 1 = -3 := by
  sorry

end find_f_prime_at_1_l225_225389


namespace find_point_P_l225_225490

/-- The point P such that AP = BP = CP = DP for given points A, B, C, D in 3D space -/
theorem find_point_P :
  ∃ P : ℝ × ℝ × ℝ, 
    let A := (10, 0, 0 : ℝ × ℝ × ℝ),
        B := (0, -6, 0 : ℝ × ℝ × ℝ),
        C := (0, 0, 8 : ℝ × ℝ × ℝ),
        D := (0, 0, 0 : ℝ × ℝ × ℝ) in
      dist P A = dist P B ∧
      dist P B = dist P C ∧
      dist P C = dist P D ∧
      P = (5, -3, 4) :=
by
  sorry

end find_point_P_l225_225490


namespace ways_to_select_computers_l225_225915

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the number of Type A and Type B computers
def num_type_a := 4
def num_type_b := 5

-- Define the total number of computers to select
def total_selected := 3

-- Define the calculation for number of ways to select the computers ensuring both types are included
def ways_to_select := binomial num_type_a 2 * binomial num_type_b 1 + binomial num_type_a 1 * binomial num_type_b 2

-- State the theorem
theorem ways_to_select_computers : ways_to_select = 70 :=
by
  -- Proof will be provided here
  sorry

end ways_to_select_computers_l225_225915


namespace quarters_left_l225_225147

theorem quarters_left (initial_amount spent_on_pizza spent_on_soda spent_on_jeans : ℝ)
  (h_initial : initial_amount = 40)
  (h_pizza : spent_on_pizza = 2.75)
  (h_soda : spent_on_soda = 1.50)
  (h_jeans : spent_on_jeans = 11.50) :
  let remaining_amount := initial_amount - (spent_on_pizza + spent_on_soda + spent_on_jeans)
  in (remaining_amount / 0.25) = 97 :=
by
  sorry

end quarters_left_l225_225147


namespace sarah_walk_probability_l225_225675

def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

def coprime (p q : ℕ) : Prop := gcd p q = 1

theorem sarah_walk_probability (p q : ℕ) (hp : coprime p q) (p_val : p = 4) (q_val : q = 7) :
  let num_gates := 15
  let step_distance := 80
  let max_walk_distance := 320
  let num_valid_walks := num_gates * (2 * (max_walk_distance / step_distance))
  let total_pairs := num_gates * (num_gates - 1)
  num_valid_walks = 120 ∧ total_pairs = 210 ∧ ((120/210) = (4/7)) → p + q = 11 :=
by
  sorry

end sarah_walk_probability_l225_225675


namespace peter_large_glasses_l225_225143

theorem peter_large_glasses (cost_small cost_large total_money small_glasses change num_large_glasses : ℕ)
    (h1 : cost_small = 3)
    (h2 : cost_large = 5)
    (h3 : total_money = 50)
    (h4 : small_glasses = 8)
    (h5 : change = 1)
    (h6 : total_money - change = 49)
    (h7 : small_glasses * cost_small = 24)
    (h8 : 49 - 24 = 25)
    (h9 : 25 / cost_large = 5) :
  num_large_glasses = 5 :=
by
  sorry

end peter_large_glasses_l225_225143


namespace one_fourth_of_eight_times_x_plus_two_l225_225074

theorem one_fourth_of_eight_times_x_plus_two (x : ℝ) : 
  (1 / 4) * (8 * x + 2) = 2 * x + 1 / 2 :=
by
  sorry

end one_fourth_of_eight_times_x_plus_two_l225_225074


namespace recurring_division_l225_225227

def recurring_36_as_fraction : ℚ := 36 / 99
def recurring_12_as_fraction : ℚ := 12 / 99

theorem recurring_division :
  recurring_36_as_fraction / recurring_12_as_fraction = 3 := 
sorry

end recurring_division_l225_225227


namespace intersection_complement_l225_225059

-- Definitions
def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 2}

-- Statement to prove
theorem intersection_complement :
  (((I \ B) ∩ A : Set ℕ) = {3, 5}) :=
by
  sorry

end intersection_complement_l225_225059


namespace girls_daisies_impossible_l225_225206

theorem girls_daisies_impossible :
  ∀ (n : ℕ), n = 33 → ¬ ∀ g : ℕ, (1 ≤ g ∧ g ≤ n) →
  ∃ d : ℕ, d = g + 2 → ∀ x : ℕ, (1 ≤ x ∧ x ≤ d) →
  ((x mod 2 = 1) → false ) :=
by
  sorry

end girls_daisies_impossible_l225_225206


namespace calc_triangle_PQR_area_l225_225565

-- Given:
-- 1. Triangle ABC is a right triangle with right angle at A
-- 2. R is the midpoint of the hypotenuse BC
-- 3. Point P on AB such that CP = BP
-- 4. Point Q on BP such that triangle PQR is equilateral
-- 5. The area of triangle ABC is 27

theorem calc_triangle_PQR_area (A B C P Q R : ℝ^2) (hABC : A ≠ B) (hA90 : angle_fixed_90 A B C)
  (hR_midpoint : midpoint R B C) (hP_on_AB : P_is_on_AB P A B) 
  (hCP_eq_BP : distance C P = distance B P) (hQ_on_BP : Q_is_on_BP Q B P)
  (hPQR_eq_eq_tri : Equilateral_PQR P Q R) (h_area_ABC : area_triangle A B C = 27) :
  area_triangle P Q R = 9 / 2 :=
sorry

end calc_triangle_PQR_area_l225_225565


namespace pages_left_to_read_l225_225248

-- Defining the given conditions
def total_pages : ℕ := 500
def read_first_night : ℕ := (20 * total_pages) / 100
def read_second_night : ℕ := (20 * total_pages) / 100
def read_third_night : ℕ := (30 * total_pages) / 100

-- The total pages read over the three nights
def total_read : ℕ := read_first_night + read_second_night + read_third_night

-- The remaining pages to be read
def remaining_pages : ℕ := total_pages - total_read

theorem pages_left_to_read : remaining_pages = 150 :=
by
  -- Leaving the proof as a placeholder
  sorry

end pages_left_to_read_l225_225248


namespace rectangle_area_l225_225287

theorem rectangle_area (side_length width length : ℝ) (h_square_area : side_length^2 = 36)
  (h_width : width = side_length) (h_length : length = 2.5 * width) :
  width * length = 90 :=
by 
  sorry

end rectangle_area_l225_225287


namespace f_neg3_f_2_l225_225512

def f (x : Int) : Int :=
  if x < 0 then 3 * x + 4 else 4 * x - 1

theorem f_neg3 : f (-3) = -5 :=
by
  unfold f
  simp
  sorry

theorem f_2 : f 2 = 7 :=
by
  unfold f
  simp
  sorry

end f_neg3_f_2_l225_225512


namespace min_value_of_one_over_a_and_one_over_b_l225_225121

noncomputable def minValue (a b : ℝ) : ℝ :=
  if 2 * a + 3 * b = 1 then 1 / a + 1 / b else 0

theorem min_value_of_one_over_a_and_one_over_b :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 1 ∧ minValue a b = 65 / 6 :=
by
  sorry

end min_value_of_one_over_a_and_one_over_b_l225_225121


namespace businessmen_drink_none_l225_225299

open Finset

-- Definitions based on conditions
def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 13
def soda_drinkers : ℕ := 8
def coffee_and_tea_drinkers : ℕ := 6
def coffee_and_soda_drinkers : ℕ := 2
def tea_and_soda_drinkers : ℕ := 3
def all_three_drinkers : ℕ := 1

-- Theorem stating the number of businessmen who drank none of the beverages
theorem businessmen_drink_none :
  total_businessmen - (coffee_drinkers + tea_drinkers + soda_drinkers 
  - coffee_and_tea_drinkers - coffee_and_soda_drinkers - tea_and_soda_drinkers 
  + all_three_drinkers) = 4 :=
  sorry

end businessmen_drink_none_l225_225299


namespace trigonometric_simplification_l225_225919

theorem trigonometric_simplification :
  (sin (30 * (Real.pi / 180)) + sin (60 * (Real.pi / 180))) /
  (cos (30 * (Real.pi / 180)) + cos (60 * (Real.pi / 180))) = 
  tan (45 * (Real.pi / 180)) :=
by {
  sorry
}

end trigonometric_simplification_l225_225919


namespace cmvt_satisfaction_l225_225150

-- We define the functions f and g
def f (x : ℝ) : ℝ := Real.sin x
def g (x : ℝ) : ℝ := Real.cos x

-- Define the interval [0, π/2]
def interval := Set.Icc 0 (Real.pi / 2)

theorem cmvt_satisfaction :
  ∃ c ∈ Set.Ioo 0 (Real.pi / 2), (Real.cos c / -Real.sin c) = -1 :=
sorry

end cmvt_satisfaction_l225_225150


namespace gcd_4830_3289_l225_225549

theorem gcd_4830_3289 : Nat.gcd 4830 3289 = 23 :=
by sorry

end gcd_4830_3289_l225_225549


namespace AM_GM_problem_l225_225815

theorem AM_GM_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := 
sorry

end AM_GM_problem_l225_225815


namespace fill_blank1_fill_blank2_product_pattern_l225_225905

theorem fill_blank1 : sqrt (1 - 9 / 25) = 4 / 5 := sorry

theorem fill_blank2 : sqrt (1 - 13 / 49) = 6 / 7 := sorry

theorem product_pattern :
  (List.range 80).product.map (λ n : ℕ, if n % 2 ≠ 0 then sqrt (1 - (n ^ 2) / ((2 * n + 1) ^ 2)) else 1) = 1 / 80 := sorry

end fill_blank1_fill_blank2_product_pattern_l225_225905


namespace no_intersecting_plane_exists_l225_225770

theorem no_intersecting_plane_exists
  (O : Point)
  (arcs : list Arc)
  (h_sum_less_than_2pi : ∑ arc_ang_measure arcs < 2 * Real.pi) :
  ∃ plane : Plane, ∀ arc ∈ arcs, ¬ intersects plane arc :=
sorry

end no_intersecting_plane_exists_l225_225770


namespace sum_of_specific_primes_l225_225204

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_odd_prime : ℕ := 3
def largest_prime_less_than_50 : ℕ := 47
def smallest_prime_greater_than_60 : ℕ := 61

theorem sum_of_specific_primes :
  smallest_odd_prime + largest_prime_less_than_50 + smallest_prime_greater_than_60 = 111 :=
by
  have h1 : smallest_odd_prime = 3 := rfl
  have h2 : largest_prime_less_than_50 = 47 := rfl
  have h3 : smallest_prime_greater_than_60 = 61 := rfl
  calc
    smallest_odd_prime + largest_prime_less_than_50 + smallest_prime_greater_than_60
      = 3 + 47 + 61 : by rw [h1, h2, h3]
    ... = 111 : sorry

end sum_of_specific_primes_l225_225204


namespace thirty_percent_first_digit_one_l225_225154

noncomputable def first_digit (x : ℕ) : ℕ :=
let s := (2^x).toString in s.head!.toNat!

theorem thirty_percent_first_digit_one :
  ∃ S : Finset ℕ, (S.card ≥ 300000) ∧ (∀ n ∈ S, first_digit n = 1) :=
sorry

end thirty_percent_first_digit_one_l225_225154


namespace projections_of_intersection_cyclic_quad_circumscribed_l225_225153

-- Definitions based on conditions:
variables {α : Type} [LinearOrderedField α] 

structure Point (α : Type) := (x y : α)
structure Quadrilateral (α : Type) := (A B C D : Point α)

def is_cyclic (quad : Quadrilateral α) : Prop := sorry
def intersect_at (p1 p2 p3 p4 : Point α) : Point α := sorry
def orthogonal_projection (p : Point α) (line_pt1 line_pt2 : Point α) : Point α := sorry
def is_circumscribed (quad : Quadrilateral α) : Prop := sorry

-- Given problem statement:
theorem projections_of_intersection_cyclic_quad_circumscribed (quad : Quadrilateral α)
    (h_cyclic : is_cyclic quad)
    (h_projection_not_on_extension : ∀ (O M N K L : Point α), ¬(M.x = N.x ∧ K.x = L.x)) :
    let O := intersect_at quad.A quad.C quad.B quad.D in
    let M := orthogonal_projection O quad.A quad.B in
    let N := orthogonal_projection O quad.B quad.C in
    let K := orthogonal_projection O quad.C quad.D in
    let L := orthogonal_projection O quad.D quad.A in
    is_circumscribed (Quadrilateral.mk M N K L) := 
sorry

end projections_of_intersection_cyclic_quad_circumscribed_l225_225153


namespace son_age_is_10_l225_225642

-- Define the conditions
variables (S F : ℕ)
axiom condition1 : F = S + 30
axiom condition2 : F + 5 = 3 * (S + 5)

-- State the theorem to prove the son's age
theorem son_age_is_10 : S = 10 :=
by
  sorry

end son_age_is_10_l225_225642


namespace probability_one_ball_of_each_color_selected_l225_225574

theorem probability_one_ball_of_each_color_selected :
  let totalWays := Nat.choose 9 3
  let favorableWays := 3 * 3 * 3
  (favorableWays : ℚ) / totalWays = 9 / 28 := 
by
  -- total number of ways to choose 3 balls out of 9
  let totalWays := Nat.choose 9 3
  -- number of ways to choose one ball of each color
  let favorableWays := 3 * 3 * 3
  have h_frac : (favorableWays : ℚ) / totalWays = (27 : ℚ) / 84 := by
    -- Placeholder for detailed proof
    sorry
  have h_simplified : (27 : ℚ) / 84 = 9 / 28 := by
    -- Placeholder for fraction simplification proof
    sorry
  exact Eq.trans h_frac h_simplified

end probability_one_ball_of_each_color_selected_l225_225574


namespace correct_operation_l225_225241

theorem correct_operation (a b : ℝ) : 
  (-a^3 * b)^2 = a^6 * b^2 :=
by
  sorry

end correct_operation_l225_225241


namespace radius_of_circle_l225_225479

-- Definitions for the problem setup
variables {A B C D E : Point}
variables [triangle : IsTriangle A B C]
variables [is_isosceles : IsIsosceles A B C (by exact 26) (by exact 26) (by exact 20)]
variables [altitude_A : Altitude A B C D]
variables [altitude_B : Altitude B C E]

-- Statement to prove
theorem radius_of_circle :
  radius_circle_passing_through_D_and_tangent_to_AC_at_E (by exact A) (by exact B) (by exact C) (by exact D) (by exact E) = 65 / 12 :=
sorry

end radius_of_circle_l225_225479


namespace geo_seq_log_sum_l225_225825

theorem geo_seq_log_sum
  (a : ℕ → ℝ)
  (h1 : ∀ n m, a (n + m) = a n * a m) 
  (h_pos : ∀ n, 0 < a n) 
  (h_prod : a 5 * a 6 = 27) :
  (∑ i in (Finset.range 10).map (λ i : ℕ, i + 1), Real.logb 3 (a i)) = 15 :=
  sorry

end geo_seq_log_sum_l225_225825


namespace complex_number_solution_l225_225047

theorem complex_number_solution :
  ∃ z : ℂ, (3 + 4 * complex.I) * z = 1 - 2 * complex.I ∧ z = -1/5 + (-2/5) * complex.I :=
by
  sorry

end complex_number_solution_l225_225047


namespace impossible_measure_1_liter_with_buckets_l225_225417

theorem impossible_measure_1_liter_with_buckets :
  ¬ (∃ k l : ℤ, k * Real.sqrt 2 + l * (2 - Real.sqrt 2) = 1) :=
by
  sorry

end impossible_measure_1_liter_with_buckets_l225_225417


namespace find_maximum_value_l225_225431

open Real

noncomputable def maximum_value (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : ℝ :=
  2 + sqrt 5

theorem find_maximum_value (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  sqrt (4 * a + 1) + sqrt (4 * b + 1) + sqrt (4 * c + 1) > maximum_value a b c h₁ h₂ h₃ h₄ :=
by
  sorry

end find_maximum_value_l225_225431


namespace sum_of_cubes_l225_225166

theorem sum_of_cubes (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 5) (h3 : abc = -6) : a^3 + b^3 + c^3 = -36 :=
sorry

end sum_of_cubes_l225_225166


namespace partition_problem_l225_225034

theorem partition_problem (r n : ℕ) (hr : 0 < r) (hn : 0 < n) :
  ∃ m : ℕ, (∀ (S : Finset (Fin (m + 1))) (hS : S.card = m + 1) (A : Fin (m + 1) → Fin r),
    (∃ a b : Fin (m + 1), a ≠ b ∧ A a = A b ∧ (a.val < (n + 1) * b.val / n))) ∧ 
    (∀ m', (∀ (S : Finset (Fin (m' + 1))) (hS : S.card = m' + 1) (A : Fin (m' + 1) → Fin r),
    (∃ a b : Fin (m' + 1), a ≠ b ∧ A a = A b ∧ (a.val < (n + 1) * b.val / n)) → m <= m'))) :=
begin
  use (n + 1) * r,
  split,
  { intros S hS A, 
    have h : ∀ (A : Fin (n + 1) * r + 1 → Fin r) (a b : Fin (n + 1) * r + 1), 
      a ≠ b → (A a = A b → a.val < ((n + 1) * b.val / n)),
    sorry },
  { intros m' h,
    have h' : ∀ (m' : ℕ) (hm' : m' < (n + 1) * r), 
      (∃ (A : Fin (m' + 1) → Fin r), 
        ∃ (a b : Fin (m' + 1)), a ≠ b ∧ A a = A b ∧ (a.val < ((n + 1) * b.val / n))) → false,
    sorry },
end

end partition_problem_l225_225034


namespace find_x_l225_225795

-- Define vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, 5)
def c (x : ℝ) : ℝ × ℝ := (3, x)

-- Dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Compute 8a - b
def sum_vec : ℝ × ℝ :=
  (8 * a.1 - b.1, 8 * a.2 - b.2)

-- Prove that x = 4 given condition
theorem find_x (x : ℝ) (h : dot_product sum_vec (c x) = 30) : x = 4 :=
by
  sorry

end find_x_l225_225795


namespace sum_of_first_nine_terms_l225_225377

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem sum_of_first_nine_terms (h_arith : is_arithmetic_sequence a)
    (h_sum : a 3 + a 4 + a 8 = 9) : (finset.range 9).sum a = 27 := 
sorry

end sum_of_first_nine_terms_l225_225377


namespace least_value_smallest_integer_l225_225171

noncomputable def least_possible_value (A B C D : ℤ) :=
  A + B + C + D = 4 * 70 ∧ D = 90 → A

theorem least_value_smallest_integer :
  ∃ A B C D : ℤ, A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  least_possible_value A B C D = 187 :=
by sorry

end least_value_smallest_integer_l225_225171


namespace smallest_three_digit_number_with_property_l225_225341

theorem smallest_three_digit_number_with_property :
  ∃ a : ℕ, 100 ≤ a ∧ a ≤ 999 ∧ ∃ n : ℕ, 1001 * a + 1 = n^2 ∧ ∀ b : ℕ, 100 ≤ b ∧ b ≤ 999 ∧ (∃ m : ℕ, 1001 * b + 1 = m^2) → a ≤ b :=
begin
  sorry,
end

end smallest_three_digit_number_with_property_l225_225341


namespace letters_symmetry_l225_225097

theorem letters_symmetry (people : Fin 20) (sends : Fin 20 → Finset (Fin 20)) (h : ∀ p, (sends p).card = 10) :
  ∃ i j : Fin 20, i ≠ j ∧ j ∈ sends i ∧ i ∈ sends j :=
by
  sorry

end letters_symmetry_l225_225097


namespace factorize_expression_l225_225716

theorem factorize_expression (x y : ℝ) : x^2 * y - 2 * x * y^2 + y^3 = y * (x - y)^2 := 
sorry

end factorize_expression_l225_225716


namespace peter_bought_large_glasses_l225_225145

-- Define the conditions as Lean definitions
def total_money : ℕ := 50
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def small_glasses_bought : ℕ := 8
def change_left : ℕ := 1

-- Define the number of large glasses bought
def large_glasses_bought (total_money : ℕ) (cost_small_glass : ℕ) (cost_large_glass : ℕ) (small_glasses_bought : ℕ) (change_left : ℕ) : ℕ :=
  let total_spent := total_money - change_left
  let spent_on_small := cost_small_glass * small_glasses_bought
  let spent_on_large := total_spent - spent_on_small
  spent_on_large / cost_large_glass

-- The theorem to be proven
theorem peter_bought_large_glasses : large_glasses_bought total_money cost_small_glass cost_large_glass small_glasses_bought change_left = 5 :=
by
  sorry

end peter_bought_large_glasses_l225_225145


namespace find_n_l225_225766

def divisors (n : ℕ) : ℕ → Prop :=
  λ d, d > 0 ∧ n % d = 0

def has_unique_pairwise_sums (divs : List ℕ) (sums : List ℕ) : Prop :=
  (list.nodup sums) ∧ ∀ d1 d2, d1 ∈ divs → d2 ∈ divs → d1 < d2 → (d1 + d2) ∈ sums

theorem find_n (n : ℕ) :
  (∃ sums, sums = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 46, 48, 50, 54, 60] ∧
           has_unique_pairwise_sums (list.filter (divisors n) (list.range (n+1))) sums) →
  n = 45 :=
by
  -- proof to be filled in
  sorry

end find_n_l225_225766


namespace largest_angle_exceeds_120_l225_225090

theorem largest_angle_exceeds_120 
  (D E F : Type) [metric_space D] [metric_space E] [metric_space F]
  (d e f : ℝ)
  (h1 : e = 2)
  (h2 : f = 2)
  (h3 : d > 2 * real.sqrt 2)
  (angle_D : ℝ)
  (law_of_cosines : d^2 = e^2 + f^2 - 2 * e * f * real.cos angle_D)
  : angle_D > 120 :=
sorry

end largest_angle_exceeds_120_l225_225090


namespace convex_covering_l225_225481

noncomputable theory

open Set

theorem convex_covering (k n : ℕ) (hk : k > 0) (hnk : n > k) :
  ∃ (F : Set ℝ) (F1 F2 ... Fn : Set ℝ),
    (∀ S : Finset (Fin n), S.card = k → ¬ (F ⊆ ⋃ i ∈ S, F i)) ∧
    (∀ S : Finset (Fin n), S.card = k + 1 → (F ⊆ ⋃ i ∈ S, F i)) :=
sorry

end convex_covering_l225_225481


namespace largest_int_same_cost_l225_225950

-- Definitions of cost calculations for both options
def decimal_cost (n : ℕ) : ℕ := (n.digits 10).sum
def binary_cost (n : ℕ) : ℕ := ((n.digits 2).map (λ d, if d = 1 then 2 else 1)).sum

-- The main theorem 
theorem largest_int_same_cost : ∃ n < 500, decimal_cost n = binary_cost n ∧ 
  (∀ m, m < 500 → decimal_cost m = binary_cost m → m ≤ n) :=
sorry

end largest_int_same_cost_l225_225950


namespace at_least_one_not_less_than_two_l225_225777

theorem at_least_one_not_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a + (1 / b) ≥ 2 ∨ b + (1 / a) ≥ 2 :=
by
  sorry

end at_least_one_not_less_than_two_l225_225777


namespace larger_number_l225_225942

theorem larger_number (x y: ℝ) 
  (h1: x + y = 40)
  (h2: x - y = 6) :
  x = 23 := 
by
  sorry

end larger_number_l225_225942


namespace cyclist_average_speed_l225_225634

theorem cyclist_average_speed :
  ∀ (a : ℝ), 
  let initial_speed := 20,
      speed_increase := 1.2,
      new_speed := speed_increase * initial_speed,
      distance_first_third := a / 3,
      distance_remaining := 2 * a / 3,
      time_first_third := distance_first_third / initial_speed,
      time_remaining := distance_remaining / new_speed,
      total_time := time_first_third + time_remaining,
      avg_speed := a / total_time
  in avg_speed = 22.5 :=
by
  sorry

end cyclist_average_speed_l225_225634


namespace find_monic_quadratic_real_coefficients_l225_225735

-- Define the necessary conditions
def is_monic_quadratic (f : ℝ[X]) : Prop :=
  f.degree = Polynomial.degree (Polynomial.X ^ 2) ∧ f.leadingCoeff = 1

def has_real_coefficients (f : ℝ[X]) : Prop :=
  ∀ n, (Polynomial.coeff f n).im = 0

def root_exists (f : ℝ[X]) (z : ℂ) : Prop :=
  f.eval z = 0

-- The main theorem statement
theorem find_monic_quadratic_real_coefficients (z : ℂ) (f : ℝ[X]) (h : z = 2 - 3 * I) :
  is_monic_quadratic f ∧ has_real_coefficients f ∧ root_exists f z ∧ root_exists f z.conj ↔ f = Polynomial.C 13 + Polynomial.X * (Polynomial.X - 4) :=
sorry

end find_monic_quadratic_real_coefficients_l225_225735


namespace triangles_with_positive_area_count_l225_225425

-- Given the points on a 6x6 grid
def points := {(i, j) | i j : ℕ, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6}

-- Define a function to determine if three points are collinear
def collinear (p1 p2 p3 : ℕ × ℕ) : Prop :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  let (x3, y3) := p3 in
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

-- Define a function to count the number of triangles with positive area
def count_triangles_with_positive_area : ℕ :=
{
  let point_list := points.to_list
  let combinations := point_list.to_finset.powerset_len 3
  let valid_triangles := combinations.filter (λ s, ¬collinear s.elem 0 s.elem 1 s.elem 2)
  valid_triangles.card
}

theorem triangles_with_positive_area_count : count_triangles_with_positive_area = 6786 :=
by
  sorry

end triangles_with_positive_area_count_l225_225425


namespace line_equation_l225_225176

theorem line_equation
(point_pass : (λ P : ℝ × ℝ, P = (-4, -1)))
(x_intercept_twice_y_intercept : ∀ l : ℝ × ℝ → Prop, (∃ (P : ℝ × ℝ), l P ∧ P.1 / P.2 = 2)) :
(l = (λ P : ℝ × ℝ, P.2 = (1 / 4) * P.1) ∨ l = (λ P : ℝ × ℝ, P.1 + 2 * P.2 + 6 = 0)) :=
sorry

end line_equation_l225_225176


namespace problem_statement_l225_225044

-- Define an arithmetic sequence with first term a1 and common difference d
noncomputable def arithmetic_seq (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define a geometric sequence condition
def is_geometric_seq (a₁ a₃ a₁₁ : ℕ) : Prop := a₃ * a₃ = a₁ * a₁₁

-- Define the given sequences
def a_n (n : ℕ) := arithmetic_seq 2 3 n
def b_n (n : ℕ) := a_n n - 2^n - 1/2

-- Define the sum of the first n terms
noncomputable def sum_b_n (n : ℕ) : ℕ := (finset.range n).sum (λ i, b_n (i + 1))

-- The main proof statement
theorem problem_statement :
  (∀ n, a_n n = 3 * n - 1) ∧
  ∀ n, sum_b_n n = (3 * n * n / 2) - 2^(n + 1) + 2 :=
by
  sorry

end problem_statement_l225_225044


namespace last_integer_in_sequence_l225_225658

noncomputable def sequence : ℕ → ℚ
| 0       := 1024000
| (n + 1) := sequence n / 4

theorem last_integer_in_sequence : ∃ n : ℕ, sequence n = 250 ∧ sequence (n + 1) ≠ 250 ∧ ((sequence (n + 1) * 4) ∉ ℤ) :=
begin
  use 6,
  split,
  { 
    unfold sequence,
    simp only [sequence],
    norm_num,
  },
  {
    split,
    {
      unfold sequence,
      simp only [sequence],
      norm_num,
    },
    {
      unfold sequence,
      simp only [sequence],
      norm_num,
      norm_cast,
      intro h,
      interval_cases h, -- deals with cases integers close to the real number
    }
  }
end

end last_integer_in_sequence_l225_225658


namespace tangent_circumcircle_AMN_l225_225263

theorem tangent_circumcircle_AMN 
  {A B C D E M N : Point}
  {O1 O2 : Circle}
  (h1 : is_median A B C D)
  (h2 : is_on AD E)
  (h3 : passes_through O1 B E)
  (h4 : touches_at O1 BC B)
  (h5 : passes_through O2 C E)
  (h6 : touches_at O2 BC C)
  (h7 : intersects_at O1 AB M)
  (h8 : intersects_at O2 AC N) :
  tangent_to_circumcircle_AMN A M N O1 ∧ tangent_to_circumcircle_AMN A M N O2 := 
sorry

end tangent_circumcircle_AMN_l225_225263


namespace smallestThreeDigitNumberWithPerfectSquare_l225_225343

def isThreeDigitNumber (a : ℕ) : Prop := 100 ≤ a ∧ a ≤ 999

def formsPerfectSquare (a : ℕ) : Prop := ∃ n : ℕ, 1001 * a + 1 = n * n

theorem smallestThreeDigitNumberWithPerfectSquare :
  ∀ a : ℕ, isThreeDigitNumber a → formsPerfectSquare a → a = 183 :=
by
sorry

end smallestThreeDigitNumberWithPerfectSquare_l225_225343


namespace valid_four_digit_numbers_l225_225935

def four_digit_numbers_no_adjacent_zero_three : ℕ :=
  66

theorem valid_four_digit_numbers :
  let digits := {0, 1, 3, 5, 7}
  (∃ (n : ℕ), n = 4) ∧  -- Four digits long
  (∀ d ∈ digits, d ≠ 0 ∧ d ≠ 3) →  -- Digits 0 and 3 are not adjacent
  four_digit_numbers_no_adjacent_zero_three = 66 :=
by {
  sorry
}

end valid_four_digit_numbers_l225_225935


namespace polar_eq_and_slope_of_l_l225_225844

-- Given conditions as definitions
def circle_eq (x y : ℝ) : Prop := (x + 6)^2 + y^2 = 25

def line_param (t α : ℝ) : ℝ × ℝ := (t * cos α, t * sin α)

def intersect_AB (A B : ℝ × ℝ) : ℝ := real.dist A B = sqrt 10

-- The proof problem
theorem polar_eq_and_slope_of_l (x y t α : ℝ) :
  circle_eq x y →
  (∀ t, (x, y) = line_param t α) →
  let A := (line_param 1 α) in
  let B := (line_param (-1) α) in
  intersect_AB A B →
  ∃ ρ : ℝ, ρ^2 + 12 * ρ * cos α + 11 = 0 ∧ (tan α) = ±sqrt(15) / 3 :=
begin
  intros,
  sorry  -- proof to be filled
end

end polar_eq_and_slope_of_l_l225_225844


namespace ellipse_focus_dot_product_l225_225868

theorem ellipse_focus_dot_product
  (P : ℝ × ℝ)
  (a b c : ℝ)
  (h_ellipse : ∀ (x y : ℝ), (x, y) ∈ set_of (λ P, P = P) → x^2 / a^2 + y^2 / b^2 = 1)
  (h_a : a = 2)
  (h_b : b = sqrt 3)
  (h_c : c = 1)
  (F1 F2 : ℝ × ℝ)
  (h_F1 : F1 = (c, 0))
  (h_F2 : F2 = (-c, 0))
  (angle_F1P_F2 : ∃ θ : ℝ, θ = real.pi / 3)
  (cosine_law : ∀ (m n : ℝ), cos (real.pi / 3) = (m^2 + n^2 - 4) / (2 * m * n))
  :
  let m := dist P F1,
      n := dist P F2
  in (m + n = 2 * a) → (m * n = 4) → ((directed.from_vectors P F1 P F2).dot < (2))
  (directed.from_vectors P F1 P F2).dot < 2 :=
sorry

end ellipse_focus_dot_product_l225_225868


namespace laci_is_correct_l225_225718

-- Definitions of lengths a, b, and c representing the sides of the triangle
variable {a b c : ℝ} (h₀: a > 0) (h₁: b > 0) (h₂: c > 0)

-- Definition of the welded triangle by assigning masses proportional to the lengths
def welded_triangle := {a b c : ℝ // a > 0 ∧ b > 0 ∧ c > 0}

-- Centroid determination method by Feri
def feri_centroid (T : welded_triangle) : ℝ × ℝ := 
sorry -- Placeholder for actual function calculating the centroid using Feri's method

-- Centroid determination method by Laci
def laci_centroid (T : welded_triangle) : ℝ × ℝ := 
sorry -- Placeholder for actual function calculating the centroid using Laci's method

-- The proof problem statement
theorem laci_is_correct (T : welded_triangle) : 
  ∀ (a b c : ℝ), (feri_centroid T ≠ welded_triangle_centroid T) ∧ (laci_centroid T = welded_triangle_centroid T) := 
begin 
  sorry -- Proof goes here
end

end laci_is_correct_l225_225718


namespace value_of_a_l225_225043

theorem value_of_a (a : ℝ) :
  (∀ x y : ℝ, x + a^2 * y + 6 = 0 → (a-2) * x + 3 * a * y + 2 * a = 0) →
  (a = 0 ∨ a = -1) :=
by
  sorry

end value_of_a_l225_225043


namespace june_ride_time_l225_225856

-- June's time to ride a certain distance at a constant speed
def time_to_ride (distance : ℝ) (time_for_one_mile : ℝ) : ℝ :=
  distance * time_for_one_mile

theorem june_ride_time :
  let time_for_one_mile := 4 in
  let distance_to_bernards := 3.5 in
  time_to_ride distance_to_bernards time_for_one_mile = 14 := by
sorry

end june_ride_time_l225_225856


namespace complex_product_is_10i_l225_225679

noncomputable def complex_product (a b c : ℂ) : ℂ := a * b * c

theorem complex_product_is_10i :
  complex_product (1 + complex.I) (2 + complex.I) (3 + complex.I) = 10 * complex.I :=
by
  -- definition of complex multiplication and imaginary unit is given in the problem condition
  sorry

end complex_product_is_10i_l225_225679


namespace cross_section_area_of_sphere_l225_225824

theorem cross_section_area_of_sphere (A B C D A1 B1 C1 D1 X Y Z : V3 ℝ)
  (h_cube : is_cube ABCD_A1B1C1D1)
  (h_edge_1 : ∀ p q, edge_length p q = 1)
  (h_X_center : is_center X AA1B1B)
  (h_Y_center : is_center Y BB1C1C)
  (h_Z_on_diag : Z ∈ line BD ∧ dist D Z = 3 * dist Z B)
  : area_of_cross_section XYZ (circumscribed_sphere ABCD_A1B1C1D1) = 5 * π / 8 :=
sorry

end cross_section_area_of_sphere_l225_225824


namespace sin_alpha_minus_2beta_eq_zero_l225_225801

theorem sin_alpha_minus_2beta_eq_zero (α β : ℝ) (h : 1 / (sin (2 * β)) = 1 / (tan β) - 1 / (tan α)) : sin (α - 2 * β) = 0 :=
by
  -- Proof goes here
  sorry

end sin_alpha_minus_2beta_eq_zero_l225_225801


namespace arrange_sequence_l225_225117

open Nat

def S := { n : ℕ | n ≤ 1994 } -- The set S

variables (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_a_in_S : a ∈ S) (h_b_in_S : b ∈ S) 
  (h_rel_prime : gcd a b = 1)

theorem arrange_sequence (hS : ∀ n ∈ S, n < 1995): 
  ∃ (s : Fin 1995 → ℕ), 
    (∀ i : Fin 1994, 
     s (i + 1) - s i ≡ a [MOD 1995] ∨ 
     s (i + 1) - s i ≡ -a [MOD 1995] ∨ 
     s (i + 1) - s i ≡ b [MOD 1995] ∨ 
     s (i + 1) - s i ≡ -b [MOD 1995]) :=
sorry

end arrange_sequence_l225_225117


namespace average_age_of_team_l225_225606

def total_age (A : ℕ) (N : ℕ) := A * N
def wicket_keeper_age (A : ℕ) := A + 3
def remaining_players_age (A : ℕ) (N : ℕ) (W : ℕ) := (total_age A N) - (A + W)

theorem average_age_of_team
  (A : ℕ)
  (N : ℕ)
  (H1 : N = 11)
  (H2 : A = 28)
  (W : ℕ)
  (H3 : W = wicket_keeper_age A)
  (H4 : (wicket_keeper_age A) = A + 3)
  : (remaining_players_age A N W) / (N - 2) = A - 1 :=
by
  rw [H1, H2, H3, H4]; sorry

end average_age_of_team_l225_225606


namespace equidistant_points_l225_225489

variables (P : ℝ × ℝ × ℝ)
variables (A B C D : ℝ × ℝ × ℝ)

def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem equidistant_points :
  let A := (10, 0, 0)
  let B := (0, -6, 0)
  let C := (0, 0, 8)
  let D := (0, 0, 0)
  P = (5, -3, 4)
  P = (5, -3, 4) → (dist P A = dist P B) ∧ (dist P A = dist P C) ∧ (dist P A = dist P D) :=
begin
  intros,
  sorry
end

end equidistant_points_l225_225489


namespace cos_double_angle_l225_225774

theorem cos_double_angle (α : ℝ) (h : Real.cos (α + Real.pi / 2) = 3 / 5) : Real.cos (2 * α) = 7 / 25 :=
by 
  sorry

end cos_double_angle_l225_225774


namespace second_option_feasible_l225_225709

def Individual : Type := String
def M : Individual := "M"
def I : Individual := "I"
def P : Individual := "P"
def A : Individual := "A"

variable (is_sitting : Individual → Prop)

-- Given conditions
axiom fact1 : ¬ is_sitting M
axiom fact2 : ¬ is_sitting A
axiom fact3 : ¬ is_sitting M → is_sitting I
axiom fact4 : is_sitting I → is_sitting P

theorem second_option_feasible :
  is_sitting I ∧ is_sitting P ∧ ¬ is_sitting M ∧ ¬ is_sitting A :=
by
  sorry

end second_option_feasible_l225_225709


namespace problem_statement_l225_225217

noncomputable def find_p_q_r : ℕ × ℕ × ℕ :=
  let n := 120 - 60 * Real.sqrt 2
  let p := 120
  let q := 60
  let r := 2
  (p, q, r)

theorem problem_statement :
  let (p, q, r) := find_p_q_r
  p + q + r = 182 := 
by
  let (p, q, r) := find_p_q_r
  have hp : p = 120 := rfl
  have hq : q = 60 := rfl
  have hr : r = 2 := rfl
  rw [hp, hq, hr]
  exact rfl

end problem_statement_l225_225217


namespace smallest_munificence_monic_cubic_l225_225314

def is_monic_cubic (p : ℝ[X]) : Prop :=
  p.degree = 3 ∧ p.leadingCoeff = 1

def munificence (p : ℝ[X]) (I : set ℝ) : ℝ :=
  sup (set.image (λ x, abs (eval x p)) I)

theorem smallest_munificence_monic_cubic :
  ∃ p : ℝ[X], is_monic_cubic p ∧ munificence p { x | -2 ≤ x ∧ x ≤ 2 } = 2 :=
by
  sorry

end smallest_munificence_monic_cubic_l225_225314


namespace shaded_area_of_two_overlapping_circles_l225_225958

theorem shaded_area_of_two_overlapping_circles
  (radius : ℝ)
  (h_radius : radius = 10)
  (h_overlap : ∀ C1 C2 : circle, contains_25_percent_C1_of_C2 C1 C2 ∧ contains_25_percent_C2_of_C1 C1 C2)
  (shaded_area : ℝ) :
  shaded_area = 57.08 :=
by
  sorry

end shaded_area_of_two_overlapping_circles_l225_225958


namespace cost_price_per_meter_l225_225599

-- Define the given conditions
def selling_price : ℕ := 8925
def meters : ℕ := 85
def profit_per_meter : ℕ := 35

-- Define the statement to be proved
theorem cost_price_per_meter :
  (selling_price - profit_per_meter * meters) / meters = 70 := 
by
  sorry

end cost_price_per_meter_l225_225599


namespace min_dot_product_of_quadrilateral_l225_225103

noncomputable def minimum_dot_product (A B C D P Q : ℝ × ℝ)
  (AB BC AD : ℝ) (angle_DAB : ℝ) (BPQ_area : ℝ) : ℝ :=
if (AB = 4 ∧ BC = 2 ∧ angle_DAB = π / 3 ∧ BPQ_area = (sqrt 3 / 32)) then
  97 / 16 else 0

theorem min_dot_product_of_quadrilateral :
  ∀ (A B C D P Q : ℝ × ℝ) (AB BC AD : ℝ) (angle_DAB : ℝ) (BPQ_area : ℝ),
  AB = 4 → BC = 2 → angle_DAB = π / 3 → BPQ_area = (sqrt 3 / 32) →
  minimum_dot_product A B C D P Q AB BC AD angle_DAB BPQ_area = 97 / 16 :=
by {
  sorry
}

end min_dot_product_of_quadrilateral_l225_225103


namespace num_of_solutions_l225_225745

def functional_equation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f(a + b) + f(ab - 1) = f(a) * f(b) + 1

theorem num_of_solutions :
  ∃ (f1 f2 : ℤ → ℤ),
    functional_equation f1 ∧ functional_equation f2 ∧
    (∀ f : ℤ → ℤ, functional_equation f → f = f1 ∨ f = f2) :=
sorry

end num_of_solutions_l225_225745


namespace area_trapezoid_proof_l225_225995

variable {m n b : ℝ}

def area_trapezoid (m n b : ℝ) : ℝ :=
  (b + m + (m * n) / (b - n)) * Real.sqrt (m * n)

theorem area_trapezoid_proof
  (m n b : ℝ)
  (h_m_pos : 0 < m)
  (h_n_pos : 0 < n)
  (h_b_pos : b > n) : 
  area_trapezoid m n b = (b + m + (m * n) / (b - n)) * Real.sqrt (m * n) := 
by
  sorry

end area_trapezoid_proof_l225_225995


namespace radius_of_circumcircle_of_ABC_is_R_M_is_orthocenter_of_ABC_l225_225211

-- Define the setup and make necessary assumptions
variables {R : ℝ} {O₁ O₂ O₃ A B C M : euclidean_space ℝ (fin 2)}

-- Conditions for part (a)
def circles_equal_radius (h : ∀ i, dist O_i M = R) := 
  dist O₁ M = R ∧ dist O₂ M = R ∧ dist O₃ M = R 

def pairwise_intersections (h : ∀ i j, i ≠ j → (∃! P, dist O_i P = R ∧ dist O_j P = R)) :=
  ∃! A, dist O₁ A = R ∧ dist O₂ A = R ∧ ∃! B, dist O₂ B = R ∧ dist O₃ B = R ∧ ∃! C, dist O₃ C = R ∧ dist O₁ C = R

-- The main statements to be proven
theorem radius_of_circumcircle_of_ABC_is_R 
  (h₁ : circles_equal_radius) (h₂ : pairwise_intersections) : 
  dist (circumcenter A B C) A = R := sorry

theorem M_is_orthocenter_of_ABC
  (h₁ : circles_equal_radius) (h₂ : pairwise_intersections) :
  is_orthocenter M A B C := sorry

end radius_of_circumcircle_of_ABC_is_R_M_is_orthocenter_of_ABC_l225_225211


namespace isosceles_triangle_angle_l225_225560

open Real

theorem isosceles_triangle_angle (x : ℝ) (hx : 0 < x ∧ x < 90) :
  (∃ (α β : ℝ), (α = 11.25 ∧ β = 45) ∧ 
                (sin 3 x = sin x ∧ sin x = sin (5 * x)) ∧ 
                (x = α ∨ x = β)) :=
by 
  assume h,
  sorry

end isosceles_triangle_angle_l225_225560


namespace y_plus_z_value_l225_225091

theorem y_plus_z_value (v w x y z S : ℕ) 
  (h1 : 196 + x + y = S)
  (h2 : 269 + z + 123 = S)
  (h3 : 50 + x + z = S) : 
  y + z = 196 := 
sorry

end y_plus_z_value_l225_225091


namespace part1_part2_l225_225881

def A := {n | 1 ≤ n ∧ n ≤ 366}
def P (B : Set ℕ) : Prop := ∃ a b, a ≠ b ∧ a ∈ A ∧ b ∈ A ∧ B = {a, b} ∧ 17 ∣ (a + b)
def two_element_subsets_with_P : ℕ := 3928

theorem part1 : 
  {B : Set ℕ | ∃ a b, a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ 17 ∣ (a + b) ∧ B = {a, b}}.finite.to_finset.card = two_element_subsets_with_P := sorry

def pairwise_disjoint (S : Set (Set ℕ)) : Prop :=
  ∀ B1 B2 ∈ S, B1 ≠ B2 → B1 ∩ B2 = ∅

def max_disjoint_subsets_with_P : ℕ := 179

theorem part2 :
  ∃ (S : Set (Set ℕ)), 
  pairwise_disjoint S ∧
  {B | B ∈ S ∧ (∃ a b, B = {a, b} ∧ a ∈ A ∧ b ∈ A ∧ 17 ∣ (a + b))}.finite.to_finset.card = max_disjoint_subsets_with_P := sorry

end part1_part2_l225_225881


namespace distribute_pencils_l225_225021

def number_of_ways_to_distribute_pencils (pencils friends : ℕ) : ℕ :=
  Nat.choose (pencils - friends + friends - 1) (friends - 1)

theorem distribute_pencils :
  number_of_ways_to_distribute_pencils 4 4 = 35 :=
by
  sorry

end distribute_pencils_l225_225021


namespace time_for_second_train_to_cross_l225_225601

def length_first_train : ℕ := 100
def speed_first_train : ℕ := 10
def length_second_train : ℕ := 150
def speed_second_train : ℕ := 15
def distance_between_trains : ℕ := 50

def total_distance : ℕ := length_first_train + length_second_train + distance_between_trains
def relative_speed : ℕ := speed_second_train - speed_first_train

theorem time_for_second_train_to_cross :
  total_distance / relative_speed = 60 :=
by
  -- Definitions and intermediate steps would be handled in the proof here
  sorry

end time_for_second_train_to_cross_l225_225601


namespace op_identity_l225_225084

def op (x y : ℝ) := 3^x - y

theorem op_identity (a : ℝ) : op a (op a a) = a := by
  sorry

end op_identity_l225_225084


namespace equalize_values_a_equalize_values_b_equalize_values_c_l225_225698

-- Configuration (a)
theorem equalize_values_a (a1 a2 a3 a4 a5 a6 a7 a8 : ℤ) 
  (h1 : (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) % 2 = 1):
  ¬ ∃ k, ∀ i ∈ {a1, a2, a3, a4, a5, a6, a7, a8}, i = k :=
by sorry

-- Configuration (b)
theorem equalize_values_b (b1 b2 b3 b4 b5 b6 b7 b8 : ℤ) 
  (h1 : (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8) % 2 = 0):
  ∃ k, ∀ i ∈ {b1, b2, b3, b4, b5, b6, b7, b8}, i = k :=
by sorry

-- Configuration (c)
theorem equalize_values_c (c1 c2 c3 c4 c5 c6 c7 c8 : ℤ) 
  (h1 : (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8) % 2 = 1):
  ¬ ∃ k, ∀ i ∈ {c1, c2, c3, c4, c5, c6, c7, c8}, i = k :=
by sorry

end equalize_values_a_equalize_values_b_equalize_values_c_l225_225698


namespace compare_fractions_l225_225834

variable {a b : ℝ}

theorem compare_fractions (h1 : 3 * a > b) (h2 : b > 0) :
  (a / b) > ((a + 1) / (b + 3)) :=
by
  sorry

end compare_fractions_l225_225834


namespace cos_sin_n_eq_one_l225_225163

theorem cos_sin_n_eq_one (n : ℕ) (h : n > 0) : 
  (if even n then {x | cos x ^ n - sin x ^ n = 1} = {x | ∃ m : ℤ, x = m * π}
  else {x | cos x ^ n - sin x ^ n = 1} = {x | ∃ m : ℤ, x = 2 * m * π ∨ x = 2 * m * π - π / 2}) :=
sorry

end cos_sin_n_eq_one_l225_225163


namespace transformed_mean_stddev_l225_225436

-- Definitions of mean and standard deviation, and transformation.
def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) (m : ℝ) : ℝ :=
  (data.map (λ x => (x - m)^2)).sum / data.length

def stddev (data : List ℝ) (m : ℝ) : ℝ :=
  Real.sqrt (variance data m)

-- Given conditions
variable (x : List ℝ) (hx_length : x.length = 2017)
variable (hx_mean : mean x = 4)
variable (hx_stddev : stddev x 4 = 3)

-- Transformation
def y (x_i : ℝ) : ℝ := -3 * (x_i - 2)
def y_data : List ℝ := x.map y

-- Proof problem statement
theorem transformed_mean_stddev :
  mean y_data = -6 ∧ stddev y_data (-6) = 9 := by
  sorry

end transformed_mean_stddev_l225_225436


namespace H_is_abelian_H_not_finitely_generated_l225_225480

section
-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 0], ![0, 1]]
def B : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 1], ![0, 1]]

-- Define the subgroup G generated by A and B
def G : Subgroup (Matrix (Fin 2) (Fin 2) ℝ) := 
  Subgroup.closure {A, B}

-- Define the subset H of matrices in G where a11 = a22 = 1
def H : Subgroup (Matrix (Fin 2) (Fin 2) ℝ) := 
{
  carrier := { M | M ∈ G ∧ M 0 0 = 1 ∧ M 1 1 = 1 },
  one_mem' := sorry,  -- To be proved that the identity matrix with a11 = a22 = 1 is in H
  mul_mem' := sorry,  -- To be proved that H is closed under multiplication
  inv_mem' := sorry   -- To be proved that H is closed under taking inverses
}

-- The statements to be proved.
-- Part (a): H is an abelian subgroup of G
theorem H_is_abelian : ∀ (M N: Matrix (Fin 2) (Fin 2) ℝ), M ∈ H → N ∈ H → M * N = N * M := sorry

-- Part (b): H is not finitely generated
theorem H_not_finitely_generated : ¬ ∃ (S : Finset (Matrix (Fin 2) (Fin 2) ℝ)), S.finite ∧ Subgroup.closure S.to_set = H := sorry
end

end H_is_abelian_H_not_finitely_generated_l225_225480


namespace problem_solution_l225_225507

theorem problem_solution (y : Fin 50 → ℝ)
  (h1 : (∑ i, y i) = 2)
  (h2 : (∑ i, y i / (2 - y i)) = 2) :
  (∑ i, y i * y i / (2 - y i)) = 0 := 
sorry

end problem_solution_l225_225507


namespace range_of_exponential_abs_l225_225559

theorem range_of_exponential_abs (x : ℝ) : 
  ∃ y, y = 2^|x| ∧ y ∈ set.Ici 1 :=
sorry

end range_of_exponential_abs_l225_225559


namespace n_base_conversion_l225_225622

theorem n_base_conversion :
  ∃ (a b n : ℕ), (1 ≤ a ∧ a ≤ 5) ∧ (1 ≤ b ∧ b ≤ 5) ∧ 
  (n = 37 * a + 6 * b) ∧ (n = 15 * b + a) ∧ n = 61 :=
by {
  existsi 1, existsi 4, existsi 61,
  split; norm_num,
  split; norm_num,
  split; norm_num,
  split; norm_num,
  refl,
  refl,
  }

end n_base_conversion_l225_225622


namespace average_score_l225_225822

theorem average_score (T : ℝ) (M F : ℝ) (avgM avgF : ℝ) 
  (h1 : M = 0.4 * T) 
  (h2 : M + F = T) 
  (h3 : avgM = 75) 
  (h4 : avgF = 80) : 
  (75 * M + 80 * F) / T = 78 := 
  by 
  sorry

end average_score_l225_225822


namespace coplanar_lines_parallel_if_in_plane_and_parallel_to_plane_l225_225350

noncomputable theory

variables {m n : Type} {α : Type}

-- Define what it means for a line to be a subset of a plane
def line_subset_of_plane (m : Type) (α : Type) : Prop := sorry
-- Define what it means for a line to be parallel to a plane
def line_parallel_to_plane (n : Type) (α : Type) : Prop := sorry
-- Define what it means for lines to be coplanar
def lines_coplanar (m n : Type) : Prop := sorry
-- Define what it means for lines to be parallel
def lines_parallel (m n : Type) : Prop := sorry

-- Lean 4 statement for the problem
theorem coplanar_lines_parallel_if_in_plane_and_parallel_to_plane 
  (h1 : line_subset_of_plane m α) 
  (h2 : line_parallel_to_plane n α) 
  (h3 : lines_coplanar m n) : 
  lines_parallel m n := 
sorry

end coplanar_lines_parallel_if_in_plane_and_parallel_to_plane_l225_225350


namespace a_5_is_19_l225_225847

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 5
  else sequence (n - 1) + sequence (n - 2)

theorem a_5_is_19 : sequence 5 = 19 := by
  sorry

end a_5_is_19_l225_225847


namespace sarah_students_surveyed_l225_225534

theorem sarah_students_surveyed (s t: ℝ) (hin1 : 75.2 * 0.01 * t = s) (hin2 : 62.1 * 0.01 * s = 41) :
  t = 90 := 
by
  have hs : 0.621 * s = 41, from sorry
  have hs_calculated : s = 67, from sorry
  have ht : 0.752 * t = s, from hin1
  have ht_calculated : t = 90, from sorry
  exact ht_calculated

-- Sorry, proof skipped

end sarah_students_surveyed_l225_225534


namespace arithmetic_sequence_properties_l225_225449

variable (a : ℕ → ℤ) -- definition of the arithmetic sequence
variable (S : ℕ → ℤ) -- definition of the sum of the first n terms

def arithmetic_sequence (a1 d : ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

def sum_of_arithmetic_sequence (a1 d : ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (2 * a1 + (n - 1) * d)) / 2

theorem arithmetic_sequence_properties :
  let a1 := -4
  let a8 := -18
  let d := -2
  let S8 := -88
  (∃ d, arithmetic_sequence a1 d a) → (a 0 = a1 ∧ a 7 = a8) →
  (∃ S8, sum_of_arithmetic_sequence a1 d S) → (S 8 = S8) :=
by
  intro a1 a8 d S8 h1 h2 h3
  cases h2
  cases h3
  sorry

end arithmetic_sequence_properties_l225_225449


namespace problem_solution_l225_225496

def point_equidistant (P A B C D : EclideanSpace ℝ (Fin 3)) : Prop :=
  (dist P A = dist P B) ∧ (dist P B = dist P C) ∧ (dist P C = dist P D)

def P := (5 : ℝ, -3 : ℝ, 4 : ℝ)
def A := (10 : ℝ, 0 : ℝ, 0 : ℝ)
def B := (0 : ℝ, -6 : ℝ, 0 : ℝ)
def C := (0 : ℝ, 0 : ℝ, 8 : ℝ)
def D := (0 : ℝ, 0 : ℝ, 0 : ℝ)

theorem problem_solution :
  ∃ P : EuclideanSpace ℝ (Fin 3), point_equidistant P A B C D ∧ P = (5, -3, 4) :=
by
  use (5, -3, 4) -- specifying the point P we obtained from the solution
  unfold point_equidistant -- ensuring the definition is expanded for equality checks
  sorry -- skipping the proof

end problem_solution_l225_225496


namespace greatest_integer_c_l225_225337

theorem greatest_integer_c (c : ℤ) :
  (∀ x : ℝ, x^2 + (c : ℝ) * x + 10 ≠ 0) → c = 6 :=
by
  sorry

end greatest_integer_c_l225_225337


namespace find_number_of_piles_l225_225518

theorem find_number_of_piles 
  (Q : ℕ) 
  (h1 : Q = Q) 
  (h2 : ∀ (piles : ℕ), piles = 3) 
  (total_coins : ℕ) 
  (h3 : total_coins = 30) 
  (e : 6 * Q = total_coins) :
  Q = 5 := 
by sorry

end find_number_of_piles_l225_225518


namespace concyclic_points_of_acute_triangle_l225_225866

theorem concyclic_points_of_acute_triangle 
  (A B C M N P D E F : Type*) 
  [acute_triangle : is_acute_triangle A B C]
  [midpoint_M : midpoint M B C] 
  [midpoint_N : midpoint N C A] 
  [midpoint_P : midpoint P A B]
  (perp_bisect_D : is_perpendicular_bisector D A B AM)
  (perp_bisect_E : is_perpendicular_bisector E A C AM)
  [intersection_F : intersection F (line B D) (line C E)] :
  are_concyclic A N F P :=
by { sorry }

end concyclic_points_of_acute_triangle_l225_225866


namespace angle_C_is_right_angle_l225_225456

theorem angle_C_is_right_angle (A B C D E F : Point) (H : ConvexQuadrilateral A B C D) 
  (angle_A : Angle A B D = 90) (angle_C_le : Angle C D B <= 90)
  (BE_perp_AC : Perpendicular B E (Line AC))
  (DF_perp_AC : Perpendicular D F (Line AC))
  (AE_eq_CF : Distance A E = Distance C F) : Angle C D B = 90 :=
by {
  sorry
}

end angle_C_is_right_angle_l225_225456


namespace f_at_zero_f_positive_f_increasing_l225_225279

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) : true
axiom f_nonzero : f 0 ≠ 0
axiom f_pos_gt1 (x : ℝ) : x > 0 → f x > 1
axiom f_add (a b : ℝ) : f (a + b) = f a * f b

theorem f_at_zero : f 0 = 1 :=
sorry

theorem f_positive (x : ℝ) : f x > 0 :=
sorry

theorem f_increasing (x₁ x₂ : ℝ) : x₁ < x₂ → f x₁ < f x₂ :=
sorry

end f_at_zero_f_positive_f_increasing_l225_225279


namespace centroid_vector_relation_l225_225778

open_locale big_operators

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

def centroid (a b c : V) : V :=
(1/3 : ℝ) • (a + b + c)

theorem centroid_vector_relation
  (C A B G : V)
  (ha : A - C = a)
  (hb : B - C = b)
  (hG : G = centroid C A B) :
  G - C = (1/3 : ℝ) • (a + b) :=
by 
  rw [centroid, add_sub_cancel', sub_eq_neg_add, neg_add, sub_eq_neg_add, ← add_assoc, add_assoc A B, add_right_comm A B C, ← add_smul, ← add_smul, one_smul, one_smul]; sorry

end centroid_vector_relation_l225_225778


namespace reservoir_percentage_before_storm_l225_225293

variable (total_capacity : ℝ)
variable (water_after_storm : ℝ := 220 + 110)
variable (percentage_after_storm : ℝ := 0.60)
variable (original_contents : ℝ := 220)

theorem reservoir_percentage_before_storm :
  total_capacity = water_after_storm / percentage_after_storm →
  (original_contents / total_capacity) * 100 = 40 :=
by
  sorry

end reservoir_percentage_before_storm_l225_225293


namespace area_of_quadrilateral_OBEC_l225_225638

theorem area_of_quadrilateral_OBEC :
  let A := (x : ℝ, y : ℝ) => y = -3 * x
  ∧ let B := (0, 12 : ℝ)
  ∧ let C := (6, 0 : ℝ)
  ∧ let E := (3, 3 : ℝ)
  ∧ let O := (0, 0 : ℝ)
  in area_quadrilateral O B E C = 22.5 :=
begin
  sorry
end

end area_of_quadrilateral_OBEC_l225_225638


namespace recurring_division_l225_225225

def recurring_36_as_fraction : ℚ := 36 / 99
def recurring_12_as_fraction : ℚ := 12 / 99

theorem recurring_division :
  recurring_36_as_fraction / recurring_12_as_fraction = 3 := 
sorry

end recurring_division_l225_225225


namespace cos_double_angle_l225_225772

-- Let's represent the problem and then state the theorem.
variables {α : ℝ} (h_first_quadrant : 0 < α ∧ α < π/2) (h_condition : sin α - cos α = (√10) / 5)

theorem cos_double_angle : cos (2 * α) = -4 / 5 :=
by sorry

end cos_double_angle_l225_225772


namespace custom_op_evaluation_l225_225428

def custom_op (x y : ℕ) : ℕ :=
  x * y - 3 * x

theorem custom_op_evaluation : (custom_op 6 2) - (custom_op 2 6) = -12 := by
  sorry

end custom_op_evaluation_l225_225428


namespace sample_size_is_27_l225_225281

theorem sample_size_is_27 :
  let grains := 40
  let vegetable_oils := 10
  let animal_products := 20
  let fruits_and_vegetables := 20
  let total_varieties := grains + vegetable_oils + animal_products + fruits_and_vegetables
  let sampled_animal_products := 6
  let proportion := (sampled_animal_products : ℕ) / animal_products.toFloat
  let n := (sampled_animal_products * total_varieties) / (animal_products / total_varieties)
  n = 27 := by
    sorry

end sample_size_is_27_l225_225281


namespace incorrect_inequality_l225_225359

theorem incorrect_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) : ¬ (log 2 a > log 3 b) :=
sorry

end incorrect_inequality_l225_225359


namespace total_amount_from_shares_l225_225665

theorem total_amount_from_shares (W X Y Z P : ℝ) 
(H1 : X = (3 / 2) * W) 
(H2 : Y = (1 / 3) * W) 
(H3 : Z = (3 / 4) * W) 
(H4 : P = (5 / 8) * W) 
(H5 : Y = 36) : 
W + X + Y + Z + P = 454.5 :=
by
  sorry

end total_amount_from_shares_l225_225665


namespace ribbons_count_l225_225829

theorem ribbons_count (ribbons : ℕ) 
  (yellow_frac purple_frac orange_frac : ℚ)
  (black_ribbons : ℕ)
  (h1 : yellow_frac = 1/4)
  (h2 : purple_frac = 1/3)
  (h3 : orange_frac = 1/6)
  (h4 : ribbons - (yellow_frac * ribbons + purple_frac * ribbons + orange_frac * ribbons) = black_ribbons) :
  ribbons * orange_frac = 160 / 6 :=
by {
  sorry
}

end ribbons_count_l225_225829


namespace monic_quadratic_with_root_l225_225725

theorem monic_quadratic_with_root (a b : ℝ) (h_root1 : (a + b * I) ∈ ({2-3*I, 2+3*I} : set ℂ)) :
  ∃ p : polynomial ℂ, polynomial.monic p ∧ p.coeff 0 = 13 ∧ p.coeff 1 = -4 ∧ p.coeff 2 = 1 ∧ p.eval (a + b * I) = 0 := 
by
  sorry

end monic_quadratic_with_root_l225_225725


namespace line_tangent_to_circle_l225_225318

theorem line_tangent_to_circle : 
  ∀ (x y : ℝ), (2 * x - y - 5 = 0) → (x^2 + y^2 = 5) → (abs(2 * 0 - 1 * 0 - 5) / real.sqrt(2^2 + (-1)^2) = real.sqrt 5) :=
by
  intros x y h1 h2
  assume h_center : (x = 0 ∧ y = 0),
  have h_distance : abs(2 * 0 - 1 * 0 - 5) / real.sqrt(2^2 + (-1)^2) = 5 / real.sqrt(5) := sorry,
  rw h_center at h_distance,
  show 5 / real.sqrt(5) = real.sqrt(5) := sorry

end line_tangent_to_circle_l225_225318


namespace AI_perpendicular_AO_l225_225987

variables {A B C D E F G H I O : Type}

-- Let's assume the conditions of the problem within Lean
variables (midpoint_D : (D = midpoint A B))
          (midpoint_E : (E = midpoint A C))
          (orthocenter_H : orthocenter H A B C)
          (circumcircle_O : circle O A B C)
          (line_DH : extends_to_circle D H circumcircle_O F)
          (line_EH : extends_to_circle E H circumcircle_O G)
          (intersection_I : intersects D E G F I)

-- The goal to prove
theorem AI_perpendicular_AO (circumcircle_O : circle O A B C)
                            (midpoint_D : D = midpoint A B)
                            (midpoint_E : E = midpoint A C)
                            (orthocenter_H : orthocenter H A B C)
                            (line_DH : extends_to_circle D H circumcircle_O F)
                            (line_EH : extends_to_circle E H circumcircle_O G)
                            (intersection_I : intersects D E G F I) :
                            perpendicular (line_through A I) (line_through A O) :=
begin
  sorry
end

end AI_perpendicular_AO_l225_225987


namespace move_last_digit_to_front_divisible_by_7_l225_225028

open Int

noncomputable def original_number (n : ℕ) (a : ℤ) (b : ℤ) : ℤ :=
  10 * a + b

noncomputable def transformed_number (n : ℕ) (a : ℤ) (b : ℤ) : ℤ :=
  10^(6 * n - 1) * b + a

theorem move_last_digit_to_front_divisible_by_7 (n : ℕ) (a : ℤ) (b : ℤ) 
  (A : ℤ) (B : ℤ) (hA_digits : A = original_number n a b) (hA_div : 7 ∣ A) : 7 ∣ B :=
  let B := transformed_number n a b in
  sorry

end move_last_digit_to_front_divisible_by_7_l225_225028


namespace speed_in_still_water_l225_225977

theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h_up : upstream_speed = 25) (h_down : downstream_speed = 41) :
  (upstream_speed + downstream_speed) / 2 = 33 :=
by
  rw [h_up, h_down]
  norm_num
  sorry

end speed_in_still_water_l225_225977


namespace maximum_area_ABD_l225_225823

variables {A B C D E F : Type} [ConvexQuadrilateral A B C D] 
variables [Midpoint E B C] [Midpoint F C D]
variables {areas : List ℕ}

-- Given that the areas of triangles are consecutive natural numbers
def consecutive_natural_areas (areas : List ℕ) : Prop :=
  ∃ (n : ℕ), areas = [n, n+1, n+2, n+3]

-- Given the specific case we want to prove
theorem maximum_area_ABD (h : consecutive_natural_areas areas) : 
  (∃ ABD_area : ℕ, ABD_area = 6) :=
by
  sorry

end maximum_area_ABD_l225_225823


namespace find_monic_quadratic_real_coefficients_l225_225737

-- Define the necessary conditions
def is_monic_quadratic (f : ℝ[X]) : Prop :=
  f.degree = Polynomial.degree (Polynomial.X ^ 2) ∧ f.leadingCoeff = 1

def has_real_coefficients (f : ℝ[X]) : Prop :=
  ∀ n, (Polynomial.coeff f n).im = 0

def root_exists (f : ℝ[X]) (z : ℂ) : Prop :=
  f.eval z = 0

-- The main theorem statement
theorem find_monic_quadratic_real_coefficients (z : ℂ) (f : ℝ[X]) (h : z = 2 - 3 * I) :
  is_monic_quadratic f ∧ has_real_coefficients f ∧ root_exists f z ∧ root_exists f z.conj ↔ f = Polynomial.C 13 + Polynomial.X * (Polynomial.X - 4) :=
sorry

end find_monic_quadratic_real_coefficients_l225_225737


namespace greatest_length_of_cords_l225_225889

theorem greatest_length_of_cords (a b c : ℝ) (h₁ : a = Real.sqrt 20) (h₂ : b = Real.sqrt 50) (h₃ : c = Real.sqrt 98) :
  ∃ (d : ℝ), d = 1 ∧ ∀ (k : ℝ), (k = a ∨ k = b ∨ k = c) → ∃ (n m : ℕ), k = d * (n : ℝ) ∧ d * (m : ℝ) = (m : ℝ) := by
sorry

end greatest_length_of_cords_l225_225889


namespace dividend_rate_correct_l225_225662

def stock_price : ℝ := 150
def yield_percentage : ℝ := 0.08
def dividend_rate : ℝ := stock_price * yield_percentage

theorem dividend_rate_correct : dividend_rate = 12 := by
  sorry

end dividend_rate_correct_l225_225662


namespace proof_triangle_l225_225461

noncomputable def triangle_problem (A B C : Type) [euclidean_geometry A B C] 
  (BAC : angle A B C = 94 * pi / 180)
  (ACB : angle A C B = 39 * pi / 180) : Prop :=
  BC * BC = AC * AC + AC * AB
  
theorem proof_triangle (A B C : Type) [euclidean_geometry A B C]
  (h1 : angle A B C = 94 * pi / 180)
  (h2 : angle A C B = 39 * pi / 180) :
  triangle_problem A B C :=
begin
  sorry
end

end proof_triangle_l225_225461


namespace geometric_sequence_strictly_increasing_iff_l225_225869

noncomputable def geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q^(n-1)

theorem geometric_sequence_strictly_increasing_iff (a_1 q : ℝ) :
  (∀ n : ℕ, geometric_sequence a_1 q (n+2) > geometric_sequence a_1 q n) ↔ 
  (∀ n : ℕ, geometric_sequence a_1 q (n+1) > geometric_sequence a_1 q n) := 
by
  sorry

end geometric_sequence_strictly_increasing_iff_l225_225869


namespace number_of_pigs_is_11_l225_225981

-- Definitions based on given conditions
def is_total_heads (P H : ℕ) : ℕ := P + H
def is_total_legs (P H : ℕ) : ℕ := 4 * P + 2 * H
def legs_condition (P H : ℕ) : Prop := is_total_legs P H = 2 * is_total_heads P H + 22

-- Proof problem statement
theorem number_of_pigs_is_11 (H : ℕ) (h : legs_condition 11 H) : ∃ P : ℕ, P = 11 :=
by {
  use 11,
  sorry
}

end number_of_pigs_is_11_l225_225981


namespace find_monic_quadratic_real_coefficients_l225_225738

-- Define the necessary conditions
def is_monic_quadratic (f : ℝ[X]) : Prop :=
  f.degree = Polynomial.degree (Polynomial.X ^ 2) ∧ f.leadingCoeff = 1

def has_real_coefficients (f : ℝ[X]) : Prop :=
  ∀ n, (Polynomial.coeff f n).im = 0

def root_exists (f : ℝ[X]) (z : ℂ) : Prop :=
  f.eval z = 0

-- The main theorem statement
theorem find_monic_quadratic_real_coefficients (z : ℂ) (f : ℝ[X]) (h : z = 2 - 3 * I) :
  is_monic_quadratic f ∧ has_real_coefficients f ∧ root_exists f z ∧ root_exists f z.conj ↔ f = Polynomial.C 13 + Polynomial.X * (Polynomial.X - 4) :=
sorry

end find_monic_quadratic_real_coefficients_l225_225738


namespace round_robin_10_players_l225_225990

theorem round_robin_10_players : @Nat.choose 10 2 = 45 := by
  sorry

end round_robin_10_players_l225_225990


namespace sum_b_series_l225_225876

def b : ℕ → ℕ
| 1 := 2
| 2 := 2
| 3 := 2
| (n + 1) := if h : n > 2 then
               let r := b n,
                   s := b (n - 1) * b (n - 2),
                   delta := -108 * s + 81 * r^2
               in if delta > 0 then 3
                  else if delta = 0 then 1
                  else 1 
             else 2

theorem sum_b_series : (∑ n in finset.range 1000, b (n + 1)) = 1003 :=
by sorry

end sum_b_series_l225_225876


namespace perimeter_of_DEF_le_half_perimeter_of_ABC_l225_225448

-- Definitions for geometry entities and types
variables {A B C D E F : Type}
variables [HasDist A] [HasDist B] [HasDist C] [HasDist D] [HasDist E] [HasDist F]

-- Assumption for the properties of the triangle
variable (h1 : IsAcuteTriangle A B C)
variable (h2 : IsAltitude A B D)
variable (h3 : IsAltitude B C E)
variable (h4 : IsAltitude C A F)

-- Theorem statement
theorem perimeter_of_DEF_le_half_perimeter_of_ABC 
  (h1 : IsAcuteTriangle A B C)
  (h2 : IsAltitude A B D)
  (h3 : IsAltitude B C E)
  (h4 : IsAltitude C A F) :
  dist D E + dist D F + dist E F ≤ 1 / 2 * (dist A B + dist B C + dist C A) :=
sorry

end perimeter_of_DEF_le_half_perimeter_of_ABC_l225_225448


namespace sum_not_complete_residue_system_l225_225878

theorem sum_not_complete_residue_system
  (n : ℕ) (hn : Even n)
  (a b : Fin n → Fin n)
  (ha : ∀ i : Fin n, ∃ j : Fin n, a j = i)
  (hb : ∀ i : Fin n, ∃ j : Fin n, b j = i) :
  ¬ (∀ k : Fin n, ∃ i : Fin n, a i + b i = k) :=
sorry

end sum_not_complete_residue_system_l225_225878


namespace exists_x0_l225_225902

theorem exists_x0 : ∃ x0 : ℝ, x0^2 + 2*x0 + 1 ≤ 0 :=
sorry

end exists_x0_l225_225902


namespace f_bijective_l225_225125

noncomputable def f (p q : ℕ) (hq : q > 0) (hpq : Nat.gcd p q = 1) : ℕ :=
  let primeDivisorsQ := (Nat.factors q).toFinset
  let productPrimeDivisorsQ := primeDivisorsQ.prod id
  (p * q) ^ 2 / productPrimeDivisorsQ

theorem f_bijective :
  ∃ (f : ℕ → ℕ → (ℕ → 0⊕ ℕ)), ∀ (p q : ℕ), q > 0 → Nat.gcd p q = 1 →
  ∀ (a b : ℕ), f a b = f p q → (a,b) = (p,q) ∧ set.inj_on f { (p, q) | q > 0 ∧ Nat.gcd p q = 1 } ∧ 
  ∀ n : ℕ, ∃ (p q : ℕ), q > 0 ∧ Nat.gcd p q = 1 ∧ f p q = n := 
begin
  sorry
end

end f_bijective_l225_225125


namespace problem_solution_l225_225495

def point_equidistant (P A B C D : EclideanSpace ℝ (Fin 3)) : Prop :=
  (dist P A = dist P B) ∧ (dist P B = dist P C) ∧ (dist P C = dist P D)

def P := (5 : ℝ, -3 : ℝ, 4 : ℝ)
def A := (10 : ℝ, 0 : ℝ, 0 : ℝ)
def B := (0 : ℝ, -6 : ℝ, 0 : ℝ)
def C := (0 : ℝ, 0 : ℝ, 8 : ℝ)
def D := (0 : ℝ, 0 : ℝ, 0 : ℝ)

theorem problem_solution :
  ∃ P : EuclideanSpace ℝ (Fin 3), point_equidistant P A B C D ∧ P = (5, -3, 4) :=
by
  use (5, -3, 4) -- specifying the point P we obtained from the solution
  unfold point_equidistant -- ensuring the definition is expanded for equality checks
  sorry -- skipping the proof

end problem_solution_l225_225495


namespace range_of_a_l225_225810

def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem range_of_a (a : ℝ) :
  (∀ x y, 3 ≤ x ∧ x ≤ y → (x^2 - 2*a*x + 2) ≤ (y^2 - 2*a*y + 2)) → a ≤ 3 := 
sorry

end range_of_a_l225_225810


namespace Jim_paycheck_after_deductions_l225_225470

def calculateRemainingPay (grossPay : ℕ) (retirementPercentage : ℕ) 
                          (taxDeduction : ℕ) : ℕ :=
  let retirementAmount := (grossPay * retirementPercentage) / 100
  let afterRetirement := grossPay - retirementAmount
  let afterTax := afterRetirement - taxDeduction
  afterTax

theorem Jim_paycheck_after_deductions :
  calculateRemainingPay 1120 25 100 = 740 := 
by
  sorry

end Jim_paycheck_after_deductions_l225_225470


namespace ribbon_count_proof_l225_225827

theorem ribbon_count_proof :
  (∃ N : ℕ, (1 / 4) * N + (1 / 3) * N + (1 / 6) * N + 40 = N ∧ (∃ orange_ribbons : ℕ, orange_ribbons = (1 / 6) * N ∧ orange_ribbons ≈ 27)) :=
sorry

end ribbon_count_proof_l225_225827


namespace quadratic_roots_transformed_l225_225135

theorem quadratic_roots_transformed (p q r u v : ℝ) (huv : p * u ^ 2 + q * u + r = 0) (hvuv : p * v ^ 2 + q * v + r = 0) :
  ∃ (P : ℝ[X]), P.roots = {p^2 * u + q, p^2 * v + q} ∧ P.coeff 2 = 1 ∧ P.coeff 1 = -q ∧ P.coeff 0 = pr := 
sorry

end quadratic_roots_transformed_l225_225135


namespace mean_of_set_median_12_l225_225555

/-- Given a set of five elements {n, n + 4, n + 7, n + 11, n + 16}, 
    the median of this set is defined as the third largest element. -/
def median (n : ℕ) : ℕ := n + 7

/-- Given a set of five elements {n, n + 4, n + 7, n + 11, n + 16}, 
    the mean of this set is defined as the average of these elements. -/
def mean (n : ℕ) : ℝ := (n + (n + 4) + (n + 7) + (n + 11) + (n + 16)) / 5

/-- Prove that if the median of the set {n, n + 4, n + 7, n + 11, n + 16} is 12, 
    then the mean of this set is 12.6. -/
theorem mean_of_set_median_12 : ∀ (n : ℕ), median n = 12 → mean n = 12.6 := by
  intros n h
  sorry

end mean_of_set_median_12_l225_225555


namespace find_ordered_pairs_l225_225196

theorem find_ordered_pairs (b s : ℕ) (h_b_pos : 0 < b) (h_s_pos : 0 < s) 
  (h_log_sum : ∑ i in Finset.range 20, log 4 (b * s^i) = 4012) :
  ∃ n, n = 210 :=
by {
  sorry,
}

end find_ordered_pairs_l225_225196


namespace FO_gt_DI_l225_225455

variables {F I D O E : Type} [MetricSpace F] [MetricSpace I] [MetricSpace D] [MetricSpace O] [MetricSpace E]
variables (FIDO : ConvexQuadrilateral F I D O)
variables (h1 : FI = DO) (h2 : FI > DI)
variables (h3 : ∠FIO = ∠DIO)

theorem FO_gt_DI (h1: FI = DO) (h2: FI > DI) (h3: ∠FIO = ∠DIO) : FO > DI :=
sorry

end FO_gt_DI_l225_225455


namespace total_savings_correct_l225_225663

theorem total_savings_correct :
  let price_chlorine := 10
  let discount1_chlorine := 0.20
  let discount2_chlorine := 0.10
  let price_soap := 16
  let discount1_soap := 0.25
  let discount2_soap := 0.05
  let price_wipes := 8
  let bogo_discount_wipes := 0.50
  let quantity_chlorine := 4
  let quantity_soap := 6
  let quantity_wipes := 8
  let final_chlorine_price := (price_chlorine * (1 - discount1_chlorine)) * (1 - discount2_chlorine)
  let final_soap_price := (price_soap * (1 - discount1_soap)) * (1 - discount2_soap)
  let final_wipes_price_per_two := price_wipes + price_wipes * bogo_discount_wipes
  let final_wipes_price := final_wipes_price_per_two / 2
  let total_original_price := quantity_chlorine * price_chlorine + quantity_soap * price_soap + quantity_wipes * price_wipes
  let total_final_price := quantity_chlorine * final_chlorine_price + quantity_soap * final_soap_price + quantity_wipes * final_wipes_price
  let total_savings := total_original_price - total_final_price
  total_savings = 55.80 :=
by sorry

end total_savings_correct_l225_225663


namespace line_circle_intersection_l225_225441

theorem line_circle_intersection :
  ∀ (θ t : ℝ),
  let x_circle := -1 + 2 * Real.cos θ,
      y_circle := 3 + 2 * Real.sin θ,
      x_line := 2 * t - 1,
      y_line := 6 * t - 1,
      center_circle := (-1, 3 : ℝ),
      radius_circle := 2,
      standard_eq_circle := (x_circle + 1)^2 + (y_circle - 3)^2 = 4,
      standard_eq_line := 3 * x_line - y_line + 2 = 0,
      distance_center_to_line := |3 * (-1) - 3 + 2| / Real.sqrt (3^2 + (-1)^2)
  in (0 ≤ distance_center_to_line) ∧ (distance_center_to_line < radius_circle) ∧ ¬ (∃ t, 3 * (2 * t - 1) - (6 * t - 1) + 2 = 0)
: (∃ θ t, True) := sorry

end line_circle_intersection_l225_225441


namespace angle_AEF_is_90_degrees_l225_225110

-- Define points A, E, F and the angle AEF
def A := (0, 0)
def E := (3, 1)
def F := (3, 0)

open Real EuclideanGeometry

-- Define the math proof problem statement
theorem angle_AEF_is_90_degrees : angle A E F = 90 :=
by
  -- Insert proof steps here
  sorry

end angle_AEF_is_90_degrees_l225_225110


namespace division_of_repeating_decimals_l225_225221

noncomputable def repeating_to_fraction (r : ℚ) : ℚ := 
  if r == 0.36 then 4 / 11 
  else if r == 0.12 then 4 / 33 
  else 0

theorem division_of_repeating_decimals :
  (repeating_to_fraction 0.36) / (repeating_to_fraction 0.12) = 3 :=
by
  sorry

end division_of_repeating_decimals_l225_225221


namespace num_subsets_at_least_4_adj_l225_225702

def num_adjacent_subsets (n : ℕ) (k : ℕ) : ℕ :=
  if k = 8 then 1 else if 4 ≤ k ∧ k < 8 then 8 else 0

theorem num_subsets_at_least_4_adj (n : ℕ) (h : n = 8) : 
  (∑ k in finset.Icc 4 8, num_adjacent_subsets n k) = 33 := by sorry

end num_subsets_at_least_4_adj_l225_225702


namespace at_least_two_even_l225_225975

theorem at_least_two_even (n : Fin 1998 → ℕ) (h : n 1997 ^ 2 = ∑ i in Finset.range 1997, (n i) ^ 2) :
  ∃ i j, i ≠ j ∧ (n i % 2 = 0) ∧ (n j % 2 = 0) :=
sorry

end at_least_two_even_l225_225975


namespace prove_m_equals_9_given_split_l225_225346

theorem prove_m_equals_9_given_split (m : ℕ) (h : 1 < m) (h1 : m^3 = 73) : m = 9 :=
sorry

end prove_m_equals_9_given_split_l225_225346


namespace probability_four_even_four_odd_l225_225325

theorem probability_four_even_four_odd (n : ℕ) (k : ℕ) (p : ℝ) :
  n = 8 → k = 4 → p = 1 / 2 → 
  (nat.choose n k * p^k * (1 - p)^(n - k)) = 35 / 128 :=
by
  intros n_eq k_eq p_eq
  rw [n_eq, k_eq, p_eq]
  -- Simplify and prove the arithmetic (omitted here)
  sorry

end probability_four_even_four_odd_l225_225325


namespace max_min_sum_eq_two_l225_225407

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 2 + Real.sqrt 2 * Real.sin (x + Real.pi / 4)) / (2 * x ^ 2 + Real.cos x)

theorem max_min_sum_eq_two (a b : ℝ) (h_max : ∀ x, f x ≤ a) (h_min : ∀ x, b ≤ f x) (h_max_val : ∃ x, f x = a) (h_min_val : ∃ x, f x = b) :
  a + b = 2 := 
sorry

end max_min_sum_eq_two_l225_225407


namespace regular_quadrilateral_pyramid_dihedral_angles_l225_225511

theorem regular_quadrilateral_pyramid_dihedral_angles 
  (α β : ℝ) 
  (h1 : regular_quadrilateral_pyramid P ABCD)
  (h2 : dihedral_angle (lateral_face_base P ABCD) α)
  (h3 : dihedral_angle (adjacent_lateral_faces P ABCD) β) :
  2 * (real.cos β) + (real.cos (2 * α)) = -1 := by
  sorry

end regular_quadrilateral_pyramid_dihedral_angles_l225_225511


namespace isosceles_triangle_inscribed_in_circle_l225_225465

open Set

-- Given an isosceles triangle ABC with AB = AC inscribed in circle ω.
-- Let D be an arbitrary point inside BC such that BD ≠ DC.
-- Ray AD intersects ω again at E.
-- Point F (other than E) is chosen on ω such that ∠DFE = 90°.
-- Line FE intersects rays AB and AC at points X and Y, respectively.
-- Prove that ∠XDE = ∠EDY.

theorem isosceles_triangle_inscribed_in_circle
(ABC E D F X Y : Type) [circle ABC ω]
(isosceles_triangle: ∀ a b c: ABC, ab = ac)
(point_D_cond: ∀ b c : BC, BD ≠ DC)
(ray_AD_intersects: ∀ a d : AD, intersects_again_at ω E)
(perpendicular_cond: ∀ d f e : DEF, ∠DFE = π / 2)
(intersect_cond: ∀ e fe : FE, intersects_at_rays AB AC X Y)
: ∠XDE = ∠EDY :=
by
  sorry

end isosceles_triangle_inscribed_in_circle_l225_225465


namespace find_f_prime_at_1_l225_225392

def f (x : ℝ) (f_prime_at_1 : ℝ) : ℝ := x^3 + 2 * x * f_prime_at_1 - 1

theorem find_f_prime_at_1 (f_prime_at_1 : ℝ) 
  (h_deriv : ∀ x : ℝ, deriv (λ x, f x f_prime_at_1) x = 3 * x^2 + 2 * f_prime_at_1) :
  f_prime_at_1 = -3 :=
begin
  have h1 : deriv (λ x, f x f_prime_at_1) 1 = f_prime_at_1,
  { rw h_deriv,
    simp,},
  linarith,
end

end find_f_prime_at_1_l225_225392


namespace polar_to_rect_form_l225_225312

open Complex

theorem polar_to_rect_form : sqrt 3 * exp (13 * π * I / 4) = (sqrt 6 / 2) + (sqrt 6 / 2) * I := 
by
  sorry

end polar_to_rect_form_l225_225312


namespace find_x_l225_225054

def vector_a : ℝ × ℝ := (3, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -3)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def perpendicular (u v : ℝ × ℝ) : Prop := dot_product u v = 0

theorem find_x : ∀ x : ℝ, perpendicular vector_a (vector_b x) → x = 1 :=
by
  intro x hx,
  sorry

end find_x_l225_225054


namespace construct_triangle_DEF_with_same_circumcircle_and_incircle_l225_225118

-- Define the geometric entities and properties based on the given conditions
variables {α : Type*} [Planar α]
variable (O : Circumcircle α)
variable (I : Incircle α)
variables (A B C : Points α)
variables (D : Point α)

-- State the main theorem to be proven
theorem construct_triangle_DEF_with_same_circumcircle_and_incircle :
  (D ∈ O) → ∃ E F : Point α, IsTriangle DEF ∧ Circumcircle DEF = O ∧ Incircle DEF = I := 
by
  sorry

end construct_triangle_DEF_with_same_circumcircle_and_incircle_l225_225118


namespace gross_income_increase_8_percent_l225_225998

-- Define variables representing the initial conditions.
variables (X P : ℝ)

-- Define the condition that giving a 10% discount increases items sold by 20%.
def discounted_price : ℝ := 0.9 * P
def increased_sales : ℝ := 1.2 * X

-- Define the gross income with and without discount.
def gross_income_without_discount : ℝ := X * P
def gross_income_with_discount : ℝ := increased_sales * discounted_price

-- Define the percent increase in gross income.
def percent_increase_in_gross_income : ℝ :=
  ((gross_income_with_discount - gross_income_without_discount) / gross_income_without_discount) * 100

-- The theorem to be proved: the percent increase in gross income is 8%.
theorem gross_income_increase_8_percent :
  percent_increase_in_gross_income X P = 8 :=
sorry

end gross_income_increase_8_percent_l225_225998


namespace r_squared_plus_s_squared_l225_225879

theorem r_squared_plus_s_squared (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 8) : r^2 + s^2 = 32 :=
by
  sorry

end r_squared_plus_s_squared_l225_225879


namespace find_ratio_l225_225890
   
   -- Given Conditions
   variable (S T F : ℝ)
   variable (H1 : 30 + S + T + F = 450)
   variable (H2 : S > 30)
   variable (H3 : T > S)
   variable (H4 : F > T)
   
   -- The goal is to find the ratio S / 30
   theorem find_ratio :
     ∃ r : ℝ, r = S / 30 ↔ false :=
   by
     sorry
   
end find_ratio_l225_225890


namespace molecular_weight_C4H10_is_correct_l225_225013

def num_carbons : ℕ := 4
def num_hydrogens : ℕ := 10
def weight_carbon : ℝ := 12.01
def weight_hydrogen : ℝ := 1.008

def molecular_weight_C4H10 : ℝ := (num_carbons * weight_carbon) + (num_hydrogens * weight_hydrogen)

theorem molecular_weight_C4H10_is_correct : molecular_weight_C4H10 = 58.12 := by
  calculate sorry

end molecular_weight_C4H10_is_correct_l225_225013


namespace square_area_l225_225659

theorem square_area :
  ∃ (s : ℝ), (8 * s - 2 = 30) ∧ (s ^ 2 = 16) :=
by
  sorry

end square_area_l225_225659


namespace largest_lcm_l225_225585

theorem largest_lcm :
  ∀ (a b c d e f : ℕ),
  a = Nat.lcm 18 2 →
  b = Nat.lcm 18 4 →
  c = Nat.lcm 18 6 →
  d = Nat.lcm 18 9 →
  e = Nat.lcm 18 12 →
  f = Nat.lcm 18 16 →
  max (max (max (max (max a b) c) d) e) f = 144 :=
by
  intros a b c d e f ha hb hc hd he hf
  sorry

end largest_lcm_l225_225585


namespace men_at_picnic_l225_225648

theorem men_at_picnic : 
  ∀ (M W A C : ℕ), 
    M + W + A + C = 240 ∧ 
    M = W + 40 ∧ 
    A = C + 40 ∧ 
    M + W = A 
    → M = 90 := 
by
  intros M W A C h
  cases h with h_sum h1
  cases h1 with h_m_w h2
  cases h2 with h_a_c h_m_w_a
  sorry

end men_at_picnic_l225_225648


namespace inequality_holds_iff_b_lt_a_l225_225315

theorem inequality_holds_iff_b_lt_a (a b : ℝ) :
  (∀ x : ℝ, (a + 1) * x^2 + a * x + a > b * (x^2 + x + 1)) ↔ b < a :=
by
  sorry

end inequality_holds_iff_b_lt_a_l225_225315


namespace percentage_traveled_at_35_mph_l225_225993

-- Define the variables and conditions
variables {D : ℝ} {P : ℝ}

-- Define the condition of the problem
def condition1 (P : ℝ) : Prop :=
  let distance_35 := P * D in
  let distance_65 := (1 - P) * D in
  let time_35 := distance_35 / 35 in
  let time_65 := distance_65 / 65 in
  let total_time := time_35 + time_65 in
  let avg_speed := D / total_time in
  avg_speed = 50

-- The theorem to prove
theorem percentage_traveled_at_35_mph (P_value : P = 0.35) : condition1 P :=
by {
    -- Specific details of the proof would go here.
    sorry
}

end percentage_traveled_at_35_mph_l225_225993


namespace tangent_line_eqn_l225_225930

theorem tangent_line_eqn (x : ℝ) (y : ℝ) (k : ℝ) (x0 : ℝ) :
  (∀ x, y = sqrt (x - 1)) ∧ x0 = 2 ∧ y' x0 = (1 / (2 * sqrt (x0 - 1))) ∧
  k = y' x0 ∧ k = sqrt (x0 - 1) / x0 ∧ y = y 0 + k * (x - x0) ∧
  x0 = (2 * sqrt (x0 - 1)) →
    x - 2 * y = 0 :=
by sorry

end tangent_line_eqn_l225_225930


namespace min_distinct_sums_l225_225986

theorem min_distinct_sums (n : ℕ) (hn : n ≥ 5) (s : Finset ℕ) 
  (hs : s.card = n) : 
  ∃ (t : Finset ℕ), (∀ (x y : ℕ), x ∈ s → y ∈ s → x < y → (x + y) ∈ t) ∧ t.card = 2 * n - 3 :=
by
  sorry

end min_distinct_sums_l225_225986


namespace quadratic_value_at_3_l225_225593

theorem quadratic_value_at_3 (a b c : ℝ) :
  (a * (-2)^2 + b * (-2) + c = -13 / 2) →
  (a * (-1)^2 + b * (-1) + c = -4) →
  (a * 0^2 + b * 0 + c = -2.5) →
  (a * 1^2 + b * 1 + c = -2) →
  (a * 2^2 + b * 2 + c = -2.5) →
  (a * 3^2 + b * 3 + c = -4) :=
by
  sorry

end quadratic_value_at_3_l225_225593


namespace stock_price_return_to_initial_l225_225832

variable (P₀ : ℝ) -- Initial price
variable (y : ℝ) -- Percentage increase during the fourth week

/-- The main theorem stating the required percentage increase in the fourth week -/
theorem stock_price_return_to_initial
  (h1 : P₀ * 1.30 * 0.75 * 1.20 = 117) -- Condition after three weeks
  (h2 : P₃ = P₀) : -- Price returns to initial
  y = -15 := 
by
  sorry

end stock_price_return_to_initial_l225_225832


namespace evaluate_expression_l225_225714

theorem evaluate_expression :
  abs ((4^2 - 8 * (3^2 - 12))^2) - abs (Real.sin (5 * Real.pi / 6) - Real.cos (11 * Real.pi / 3)) = 1600 :=
by
  sorry

end evaluate_expression_l225_225714


namespace james_touchdowns_per_game_l225_225851

-- Define the conditions
def touchdowns_per_game := Nat
def points_per_touchdown : ℕ := 6
def number_of_games : ℕ := 15
def two_point_conversions : ℕ := 6
def old_record_points : ℕ := 300
def points_beat_record : ℕ := 72

-- Define the total points calculation
def total_touchdown_points (T : touchdowns_per_game) : ℕ := points_per_touchdown * T * number_of_games
def total_conversion_points : ℕ := 2 * two_point_conversions
def total_points_scored : ℕ := old_record_points + points_beat_record

-- Statement that James scored a certain number of touchdowns per game
theorem james_touchdowns_per_game (T : touchdowns_per_game) :
  total_touchdown_points T + total_conversion_points = total_points_scored → T = 4 :=
by
  sorry

end james_touchdowns_per_game_l225_225851


namespace maximum_f_value_l225_225313

noncomputable def otimes (a b : ℝ) : ℝ :=
if a ≤ b then a else b

noncomputable def f (x : ℝ) : ℝ :=
otimes (3 * x^2 + 6) (23 - x^2)

theorem maximum_f_value : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 4 :=
sorry

end maximum_f_value_l225_225313


namespace point_satisfies_condition_l225_225482

-- Define Point in 3D space as tuple
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define distance between two points in 3D space
def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Given points A, B, C, D
def A : Point3D := ⟨10, 0, 0⟩
def B : Point3D := ⟨0, -6, 0⟩
def C : Point3D := ⟨0, 0, 8⟩
def D : Point3D := ⟨0, 0, 0⟩

-- Point P that satisfies AP = BP = CP = DP
def P : Point3D := ⟨5, -3, 4⟩

-- Proof that P satisfies the condition
theorem point_satisfies_condition : 
  distance P A = distance P B ∧ 
  distance P B = distance P C ∧ 
  distance P C = distance P D :=
by
  sorry

end point_satisfies_condition_l225_225482


namespace find_real_solutions_l225_225335

-- Defining the integer part of a real number
def int_part (x : ℝ) : ℤ := Int.floor x

-- Defining the fractional part of a real number
def frac_part (x : ℝ) : ℝ := x - ↑(int_part x)

-- Main theorem stating the solutions
theorem find_real_solutions (x : ℝ) : 
  (1 / (int_part x) + 1 / (int_part (2 * x)) = frac_part x + 1/3) →
  (x = 29 / 12 ∨ x = 19 / 6 ∨ x = 97 / 24) := by
  -- Placeholder for the proof
  sorry

end find_real_solutions_l225_225335


namespace bullseye_points_l225_225696

theorem bullseye_points (B : ℝ) (h : B + B / 2 = 75) : B = 50 :=
by
  sorry

end bullseye_points_l225_225696


namespace problem_statement_l225_225371

theorem problem_statement (x y z w : ℝ)
  (h1 : x + y + z + w = 0)
  (h7 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := 
sorry

end problem_statement_l225_225371


namespace max_triangles_without_tetrahedron_l225_225762

noncomputable def maxTrianglesNoTetrahedron (pts : Fin 9) := 27

theorem max_triangles_without_tetrahedron (points : Fin 9 → Fin 3) (h : ∀ (p1 p2 p3 p4 : Fin 9), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 → ¬AffineSpace.Collinear ℝ {points p1, points p2, points p3, points p4}) :
  ({ t : Finset (Fin 9) // t.card = 3 }).card ≤ maxTrianglesNoTetrahedron := sorry

end max_triangles_without_tetrahedron_l225_225762


namespace point_satisfies_condition_l225_225483

-- Define Point in 3D space as tuple
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define distance between two points in 3D space
def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Given points A, B, C, D
def A : Point3D := ⟨10, 0, 0⟩
def B : Point3D := ⟨0, -6, 0⟩
def C : Point3D := ⟨0, 0, 8⟩
def D : Point3D := ⟨0, 0, 0⟩

-- Point P that satisfies AP = BP = CP = DP
def P : Point3D := ⟨5, -3, 4⟩

-- Proof that P satisfies the condition
theorem point_satisfies_condition : 
  distance P A = distance P B ∧ 
  distance P B = distance P C ∧ 
  distance P C = distance P D :=
by
  sorry

end point_satisfies_condition_l225_225483


namespace problem_OC_OA_l225_225101

noncomputable def OC_OA_ratio : Prop :=
  ∀ (A B C D M O : ℝ × ℝ),
    -- Conditions of the problem
    A = (0, 6) ∧
    B = (6, 6) ∧
    C = (6, 0) ∧
    D = (0, 0) ∧
    M = (4.5, 0) ∧
    -- Definition of coordinates
    -- Equation of line AC is y = -x + 6
    -- Equation of line BM is y = 4x - 18
    O = (24/5, 6/5) ∧
    -- Distances
    let OC := Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2) in
    let OA := Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2) in
    -- Prove the ratio OC/OA is 1/4
    OC / OA = 1 / 4

theorem problem_OC_OA : OC_OA_ratio :=
by
  sorry

end problem_OC_OA_l225_225101


namespace monic_quadratic_with_root_l225_225740

theorem monic_quadratic_with_root (p : ℝ[X]) (h1 : p.monic) (h2 : p.eval (2 - 3 * Complex.I) = 0) : 
  p = X^2 - 4 * X + 13 := 
by 
  sorry

end monic_quadratic_with_root_l225_225740


namespace area_square_PQRS_l225_225541

open EuclideanGeometry

-- Definitions based directly on the problem conditions.
def right_triangle (L M N : Point) : Prop :=
  ∃ (right_angle : Triangle L M N = ∧ angle_lmn == 90),
  -- Further condition for right triangle

def Square (P Q R S : Point) : Prop :=
  ∀ (sides : line P Q = line Q R = line R S = line S P),
     (angles : angle PQR = angle QRS = angle RSP)

def inscribed (square : Square P Q R S) (triangle : right_triangle L M N) : Prop :=
  not $ is_inside(square_of(P, Q, R, S))

-- Assumptions from conditions
variable (L M N P Q R S O: Point) -- Introduce points
variable (h1 : LP = 15)
variable (h2 : MQ = 65)
variable (h3 : right_triangle L M N)
variable (h4 : Square P Q R S) -- square PQRS
variable (h5 : inscribed (Square square) (right_triangle L M N)) 

-- defining proof problem
theorem area_square_PQRS : 
  square PQRS →  area_of (square P Q R S) = 975 := 
sorry -- Proof is not required.


end area_square_PQRS_l225_225541


namespace num_lines_in_grid_l225_225796

theorem num_lines_in_grid (columns rows : ℕ) (H1 : columns = 4) (H2 : rows = 3) 
    (total_points : ℕ) (H3 : total_points = columns * rows) :
    ∃ lines, lines = 40 :=
by
  sorry

end num_lines_in_grid_l225_225796


namespace main_theorem_l225_225296

def no_positive_integers_satisfy (n k : ℕ) : Prop :=
  ¬ ∃ (n k : ℕ) (hn : 0 < n) (hk : 0 < k), (5 + 3 * Real.sqrt 2)^n = (3 + 5 * Real.sqrt 2)^k

theorem main_theorem : no_positive_integers_satisfy :=
begin
  sorry
end

end main_theorem_l225_225296


namespace find_x_l225_225018

theorem find_x (x : ℝ) (h_pos : x > 0) (h_area : (1 / 2) * x * (3 * x) = 54) : x = 6 :=
by
  sorry

end find_x_l225_225018


namespace octagon_shaded_area_l225_225309

theorem octagon_shaded_area :
  ∀ (s : ℝ), s = 8 →
  let θ := real.pi / 4 in
  let R := s / (2 * real.sin (θ/2)) in
  let shaded_area := (s * 1 / real.sqrt (2 - real.sqrt 2) / 2)^2 in
  shaded_area = 32 + 16 * real.sqrt 2 :=
sorry

end octagon_shaded_area_l225_225309


namespace seed_mixture_ryegrass_l225_225157

theorem seed_mixture_ryegrass (α : ℝ) :
  (0.4667 * 0.4 + 0.5333 * α = 0.32) -> α = 0.25 :=
by
  sorry

end seed_mixture_ryegrass_l225_225157


namespace count_isosceles_triple_digit_numbers_l225_225516

theorem count_isosceles_triple_digit_numbers : 
  ∃ n : ℕ, n = 165 ∧ ∀ (a b c : ℕ), 
  (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) →
  (a = b ∨ b = c ∨ a = c) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →
  (100 * a + 10 * b + c) = n :=
begin
  use 165,
  split,
  { refl },
  { intros a b c h1 h2,
    sorry },
end

end count_isosceles_triple_digit_numbers_l225_225516


namespace minimum_value_of_reciprocals_l225_225764

theorem minimum_value_of_reciprocals {m n : ℝ} 
  (hmn : m > 0 ∧ n > 0 ∧ (m * n > 0)) 
  (hline : 2 * m + 2 * n = 1) : 
  (1 / m + 1 / n) = 8 :=
sorry

end minimum_value_of_reciprocals_l225_225764


namespace area_triangle_ABC_l225_225298

variables {A B C D E F : Type}
variables [metric_space A] [metric_space B] [metric_space C]

noncomputable def isosceles_right_triangle (A B C : point) : Prop :=
  (distance A B = distance A C) ∧ ⟪B - A, C - A⟫ = 0

noncomputable def collinear_points (D A F : point) : Prop :=
  ∃ α β γ : ℝ, (α ≠ 0 ∨ β ≠ 0) ∧ α • D + β • A + γ • F = 0

noncomputable def height (triangle_side base_point : point) : ℝ :=
  distance triangle_side.base_point

noncomputable def area_of_triangle (A B C : point) : ℝ :=
  (1 / 2) * distance A B * distance A C

theorem area_triangle_ABC (A B C D E F : point) 
  (h1 : ℝ) (h2 : ℝ) (BC : ℝ) 
  (h_AB_eq_AC : isosceles_right_triangle A B C) 
  (h_BE_3CE : distance B E = 3 * distance E C)
  (h_DAF_collinear : collinear_points D A F) 
  (h_3h1_h2_eq_3 : 3 * h1 + h2 = 3)
  (h_area_DE_FE_eq_6 : area_of_triangle D B E + area_of_triangle F E C = 6) : 
  area_of_triangle A B C = 64 := 
sorry

end area_triangle_ABC_l225_225298


namespace probability_even_sums_l225_225557

open Nat

theorem probability_even_sums (numbers : Finset ℕ) (grid : Fin 4 → Fin 4 → ℕ) :
  (∀ i j, grid i j ∈ numbers) →
  numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} →
  (∀ i, numbers.sum (λ j, grid i j) % 2 = 0) →
  (∀ j, numbers.sum (λ i, grid i j) % 2 = 0) →
  (finset.prod (finset.range 4) (λ _ => finset.prod (finset.range 4) (λ _ => 16))) = 1 / 3584 :=
sorry

end probability_even_sums_l225_225557


namespace triangle_angle_cosine_l225_225459

theorem triangle_angle_cosine (A B C : ℝ) (sin_ratio : ℝ) (h1 : sin_ratio = 2/3) (h2 : sin A / sin B = 2 / 3) (h3 : sin B / sin C = 3 / 4) : 
  ∠ABC = real.arccos (11 / 16) :=
sorry

end triangle_angle_cosine_l225_225459


namespace maximum_time_for_3_digit_combination_lock_l225_225629

def max_time_to_open_briefcase : ℕ :=
  let num_combinations := 9 * 9 * 9
  let time_per_trial := 3
  num_combinations * time_per_trial

theorem maximum_time_for_3_digit_combination_lock :
  max_time_to_open_briefcase = 2187 :=
by
  sorry

end maximum_time_for_3_digit_combination_lock_l225_225629


namespace find_a_value_l225_225082

-- Define the complex number with parameter a
def complex_number (a : ℝ) : ℂ := (1 - a * real.i) * complex.i

-- Define the condition that the real part and imaginary part are opposites
def real_imag_opposite_condition (a : ℝ) : Prop := 
  complex.re (complex_number a) = -complex.im (complex_number a)

-- State the main theorem
theorem find_a_value (a : ℝ) (h : real_imag_opposite_condition a) : a = -1 :=
sorry

end find_a_value_l225_225082


namespace bacteria_after_7_hours_l225_225632

noncomputable def bacteria_growth (initial : ℝ) (t : ℝ) (k : ℝ) : ℝ := initial * (10 * (Real.exp (k * t)))

noncomputable def solve_bacteria_problem : ℝ :=
let doubling_time := 1 / 60 -- In hours, since 60 minutes is 1 hour
-- Given that it doubles in 1 hour, we expect the growth to be such that y = initial * (2) in 1 hour.
let k := Real.log 2 -- Since when t = 1, we have 10 * e^(k * 1) = 2 * 10
bacteria_growth 10 7 k

theorem bacteria_after_7_hours :
  solve_bacteria_problem = 1280 :=
by
  sorry

end bacteria_after_7_hours_l225_225632


namespace point_satisfies_condition_l225_225484

-- Define Point in 3D space as tuple
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define distance between two points in 3D space
def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Given points A, B, C, D
def A : Point3D := ⟨10, 0, 0⟩
def B : Point3D := ⟨0, -6, 0⟩
def C : Point3D := ⟨0, 0, 8⟩
def D : Point3D := ⟨0, 0, 0⟩

-- Point P that satisfies AP = BP = CP = DP
def P : Point3D := ⟨5, -3, 4⟩

-- Proof that P satisfies the condition
theorem point_satisfies_condition : 
  distance P A = distance P B ∧ 
  distance P B = distance P C ∧ 
  distance P C = distance P D :=
by
  sorry

end point_satisfies_condition_l225_225484


namespace division_of_repeating_decimals_l225_225220

noncomputable def repeating_to_fraction (r : ℚ) : ℚ := 
  if r == 0.36 then 4 / 11 
  else if r == 0.12 then 4 / 33 
  else 0

theorem division_of_repeating_decimals :
  (repeating_to_fraction 0.36) / (repeating_to_fraction 0.12) = 3 :=
by
  sorry

end division_of_repeating_decimals_l225_225220


namespace cos_inequality_for_y_l225_225720

theorem cos_inequality_for_y (x y : ℝ) (hx : 0 ≤ x) (hxπ : x ≤ π / 2) (hy : 0 ≤ y) (hyπ : y ≤ π / 2) :
  cos (x + y) ≥ cos x * cos y :=
by
  sorry

end cos_inequality_for_y_l225_225720


namespace find_pos_ints_l225_225722

theorem find_pos_ints (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
    (((m = 1) ∨ (a = 1) ∨ (a = 2 ∧ m = 3 ∧ 2 ≤ n)) →
    (a^m + 1 ∣ (a + 1)^n)) :=
by
  sorry

end find_pos_ints_l225_225722


namespace num_off_third_stop_l225_225210

variables (initial_people num_off_first num_off_second num_on_second num_on_third final_num after_first_stop after_second_stop x : ℕ)

def initial_condition := initial_people = 50
def first_stop_condition := num_off_first = 15
def second_stop_condition_off := num_off_second = 8
def second_stop_condition_on := num_on_second = 2
def third_stop_condition_on := num_on_third = 3
def final_condition := final_num = 28
def after_first_stop_condition := after_first_stop = initial_people - num_off_first
def after_second_stop_condition := after_second_stop =
    after_first_stop - num_off_second + num_on_second
def after_third_stop_condition := final_num =
    after_second_stop - x + num_on_third

theorem num_off_third_stop :
  initial_condition ∧ first_stop_condition ∧ second_stop_condition_off ∧
  second_stop_condition_on ∧ third_stop_condition_on ∧ final_condition ∧
  after_first_stop_condition ∧ after_second_stop_condition ∧ after_third_stop_condition → 
  x = 4 :=
begin
  intros,
  sorry,
end

end num_off_third_stop_l225_225210


namespace linear_function_difference_l225_225180

theorem linear_function_difference (g : ℝ → ℝ) (h1 : ∀ x : ℝ, g(x + 1) - g(x) = 4) : g 5 - g 8 = -12 :=
by
  sorry

end linear_function_difference_l225_225180


namespace number_of_divisibles_by_37_l225_225636

-- Define the sequence m_i as described in the problem
def sequence_m (i : ℕ) : ℕ :=
  if i = 1 then 1
  else 10 * sequence_m (i - 1) + 1

-- Define the proposition to be proved
theorem number_of_divisibles_by_37 : 
  (finset.range 2006).filter (λ i, sequence_m (i+1) % 37 = 0).card = 668 := 
by
  sorry

end number_of_divisibles_by_37_l225_225636


namespace count_subsets_with_four_adjacent_chairs_l225_225700

def count_at_least_four_adjacent_subsets (n : ℕ) (k : ℕ) (total : ℕ) : Prop :=
  ∃ (S : finset (fin n)), S.card = k ∧
  ∀ (i : finset.range k), S.filter (λ j, (j + i) % n ∈ S) = k

theorem count_subsets_with_four_adjacent_chairs :
  count_at_least_four_adjacent_subsets 8 4 288 := 
sorry

end count_subsets_with_four_adjacent_chairs_l225_225700


namespace smallest_base_10_l225_225691

def convert_85_base_9 := 5 + 8 * 9^1
def convert_210_base_6 := 0 + 1 * 6 + 2 * 6^2
def convert_1000_base_4 := 1 * 4^3
def convert_111111_base_2 := 1 + 1 * 2 + 1 * 2^2 + 1 * 2^3 + 1 * 2^4 + 1 * 2^5

theorem smallest_base_10 :
  min (min convert_85_base_9 convert_210_base_6) (min convert_1000_base_4 convert_111111_base_2)
  = convert_111111_base_2 :=
by
sorr

end smallest_base_10_l225_225691


namespace fouad_age_multiple_of_ahmed_age_in_7_years_l225_225668

theorem fouad_age_multiple_of_ahmed_age_in_7_years :
  let ahmed_age := 11
  let fouad_age := 26
  let years := 7
  let fouad_in_years := fouad_age + years
  ∃ k, fouad_in_years = k * ahmed_age ∧ k = 3 :=
by
  let ahmed_age := 11
  let fouad_age := 26
  let years := 7
  let fouad_in_years := fouad_age + years
  existsi 3
  split
  norm_num [fouad_in_years, ahmed_age]
  sorry

end fouad_age_multiple_of_ahmed_age_in_7_years_l225_225668


namespace time_for_one_paragraph_l225_225857

-- Definitions for the given conditions
def short_answer_time := 3 -- minutes
def essay_time := 60 -- minutes
def total_homework_time := 240 -- minutes
def essays_assigned := 2
def paragraphs_assigned := 5
def short_answers_assigned := 15

-- Function to calculate total time from given conditions
def total_time_for_essays (essays : ℕ) : ℕ :=
  essays * essay_time

def total_time_for_short_answers (short_answers : ℕ) : ℕ :=
  short_answers * short_answer_time

def total_time_for_paragraphs (paragraphs : ℕ) : ℕ :=
  total_homework_time - (total_time_for_essays essays_assigned + total_time_for_short_answers short_answers_assigned)

def time_per_paragraph (paragraphs : ℕ) : ℕ :=
  total_time_for_paragraphs paragraphs / paragraphs_assigned

-- Proving the question part
theorem time_for_one_paragraph : 
  time_per_paragraph paragraphs_assigned = 15 := by
  sorry

end time_for_one_paragraph_l225_225857


namespace analytical_expression_find_n_derivative_positive_l225_225405

-- Part (1): Analytical expression
theorem analytical_expression (f : ℝ → ℝ) (b a : ℝ) (h : ∀ x, f x = x^2 + b * x - a * log x)
  (tangent_condition : deriv f 1 = -5) :
  (f 1 = 1^2 + b * 1 - a * log 1) → 
  (∃ (a b : ℝ), a = 6 ∧ b = -1 ∧ ∀ x > 0, f x = x^2 - x - 6 * log x) :=
sorry

-- Part (2): Smallest n such that zero x₀ of f(x) lies in (n,n+1)
theorem find_n (f : ℝ → ℝ) (b a : ℝ) (h : ∀ x, f x = x^2 + b * x - a * log x)
  (domain_condition : ∀ x, x > 0 → f x ∈ Set.Icc (0 : ℝ) (n : ℝ) (n + 1 : ℝ))
  (zero_condition : ∃ x₀ : ℝ, f x₀ = 0) :
  ∃ (n : ℕ), n = 3 :=
sorry

-- Part (3): Prove f'(x) > 0 given specific conditions
theorem derivative_positive (f : ℝ → ℝ) (b : ℝ) (a : ℝ) :
  (∀ x, f x = x^2 + b * x - log x) →
  (a = 1) →
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (let x₀ := (x₁ + x₂) / 2 in deriv f x₀ > 0) :=
sorry

end analytical_expression_find_n_derivative_positive_l225_225405


namespace correct_proposition_l225_225685

theorem correct_proposition (p q : Prop) (a b : ℤ) (x y a b c : ℝ) : 
  (¬(p ∧ q) ∧ ¬(p ∨ q) → ¬p ∧ ¬q) ∧
  (x^2 ≠ y^2 ↔ (x ≠ y ∨ x ≠ -y)) ∧
  (¬(even a ∨ even b) → ¬even (a + b)) ∧
  (∀ x ∈ ℝ, ax^2 + bx + c ≤ 0 ↔ (a < 0 ∧ b^2 - 4 * a * c ≤ 0)) :=
by
  sorry

end correct_proposition_l225_225685


namespace simplify_expression_l225_225920

variable (a : ℝ)

theorem simplify_expression : 3 * a^2 - a * (2 * a - 1) = a^2 + a :=
by
  sorry

end simplify_expression_l225_225920


namespace gain_percent_l225_225076

variable (C S : ℝ)

theorem gain_percent 
  (h : 81 * C = 45 * S) : ((4 / 5) * 100) = 80 := 
by 
  sorry

end gain_percent_l225_225076


namespace tom_combinations_l225_225672

theorem tom_combinations : 
  let monday_choices := 1,
      tuesday_choices := 2,
      wednesday_choices := 3,
      thursday_choices := 3,
      friday_choices := 2 in
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 36 :=
by
  sorry

end tom_combinations_l225_225672


namespace initial_erasers_calculation_l225_225208

variable (initial_erasers added_erasers total_erasers : ℕ)

theorem initial_erasers_calculation
  (total_erasers_eq : total_erasers = 270)
  (added_erasers_eq : added_erasers = 131) :
  initial_erasers = total_erasers - added_erasers → initial_erasers = 139 := by
  intro h
  rw [total_erasers_eq, added_erasers_eq] at h
  simp at h
  exact h

end initial_erasers_calculation_l225_225208


namespace no_divisible_seven_digit_num_l225_225571

def seven_digit_number (d : List ℕ) : Prop :=
  d.perm (List.range' 1 7)

theorem no_divisible_seven_digit_num (a b : ℕ) (ha: seven_digit_number (Nat.digits 10 a)) (hb: seven_digit_number (Nat.digits 10 b)) (h_neq: a ≠ b) : ¬ (b % a = 0) :=
sorry

end no_divisible_seven_digit_num_l225_225571


namespace ab_greater_than_a_plus_b_l225_225806

theorem ab_greater_than_a_plus_b (a b : ℝ) (h₁ : a ≥ 2) (h₂ : b > 2) : a * b > a + b :=
  sorry

end ab_greater_than_a_plus_b_l225_225806


namespace monic_quadratic_with_root_l225_225741

theorem monic_quadratic_with_root (p : ℝ[X]) (h1 : p.monic) (h2 : p.eval (2 - 3 * Complex.I) = 0) : 
  p = X^2 - 4 * X + 13 := 
by 
  sorry

end monic_quadratic_with_root_l225_225741


namespace sum_of_2001_terms_l225_225056

theorem sum_of_2001_terms (a : ℕ → ℤ) 
  (h1 : ∀ n ≥ 3, a n = a (n-1) - a (n-2))
  (sum_1492 : (∑ i in finset.range 1492, a i) = 1985)
  (sum_1985 : (∑ i in finset.range 1985, a i) = 1492) : 
  (∑ i in finset.range 2001, a i) = 986 :=
sorry

end sum_of_2001_terms_l225_225056


namespace equation_of_line_AB_l225_225323

theorem equation_of_line_AB :
  let circle1 := ∀ x y : ℝ, x^2 + (y - 2)^2 = 4
  let circle_with_diameter (A B : ℝ × ℝ) := 
    let (x1, y1) := A
    let (x2, y2) := B in
    ∀ x y : ℝ, (x - (x1 + x2) / 2)^2 + (y - (y1 + y2) / 2)^2 = ((x2 - x1)^2 + (y2 - y1)^2) / 4 
  let is_tangent (p : ℝ × ℝ) (c_eq : ∀ x y : ℝ, Prop) := -- Definition of tangency
    ∃ k : ℝ, (f : ∀ x y : ℝ, k*x + y = some y_value)
  let point_tangent_A := is_tangent (-2,6) circle1 ∧ is_tangent A circle1
  let point_tangent_B := is_tangent (-2,6) circle1 ∧ is_tangent B circle1
  let circle2 := circle_with_diameter (-2, 6) (0, 2)
  let AB := ∀ x y : ℝ, x - 2*y = -6 in 
  ∀ x y : ℝ, (circle1 x y) ∧ (circle2 x y) → AB x y :=
sorry

end equation_of_line_AB_l225_225323


namespace sum_of_first_9_terms_of_arithmetic_sequence_l225_225776

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sum_of_first_9_terms_of_arithmetic_sequence 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 2 + a 8 = 18) 
  (h3 : sum_of_first_n_terms a S) :
  S 9 = 81 :=
sorry

end sum_of_first_9_terms_of_arithmetic_sequence_l225_225776


namespace smallest_x_palindrome_l225_225965

def is_palindrome (n : ℕ) : Prop :=
  n.toString.reverse = n.toString

theorem smallest_x_palindrome (x : ℕ) :
  (x > 0 ∧ is_palindrome (x + 4321) ∧ x + 4321 ≥ 4521) → x = 233 :=
begin
  sorry
end

end smallest_x_palindrome_l225_225965


namespace footballs_in_second_set_l225_225540

theorem footballs_in_second_set (F S : ℕ) (cost_1 : ℕ) (cost_2 : ℕ) (H1 : F + 3 * S = 220) (H2 : 3 * F + S = 155) (H3 : S = 50) : F = 35 ∧ S = 50 ∧ 3 = 3 :=
by
  -- Definitions used directly as given
  rw H3 at H1 H2,
  have h_f := calc 
    3 * F + 50 = 155 : H2
    3 * F = 105 : by linarith
    F = 35 : by linarith,
  rw h_f at H1,

  have num_footballs := calc
    F + 3 * 50 = 220 : H1
    35 + 150 = 220 : by linarith,
 
  -- Conclusion of the theorem
  split, 
  exact h_f,
  exact H3,
  exact eq.refl 3,
  
  complete []

end footballs_in_second_set_l225_225540


namespace range_of_A_max_value_of_f_l225_225416

-- Definitions of given conditions and required results.
noncomputable def triangle_conditions (a b c : ℝ) (A B C : ℝ) (S : ℝ) :=
  c * cos A = 4 / b ∧ S ≥ 2

-- Definition of function f
noncomputable def f (A : ℝ) : ℝ :=
  cos A ^ 2 + sqrt 3 * sin (π/2 + A/2) ^ 2 - sqrt 3 / 2

-- Main theorem statements
theorem range_of_A (a b c A B C S : ℝ) (h : triangle_conditions a b c A B C S) :
  (π/4) ≤ A ∧ A < π/2 :=
  sorry

theorem max_value_of_f (a b c A B C S : ℝ) (h : triangle_conditions a b c A B C S) :
  (π/4) ≤ A ∧ A < π/2 → ∃ max_f_value, max_f_value = 1/2 + sqrt 6 / 4 :=
  sorry

#eval range_of_A
#eval max_value_of_f

end range_of_A_max_value_of_f_l225_225416


namespace true_statements_count_is_zero_l225_225400

def proposition_1 : Prop :=
  ∀ s, (s.has_two_parallel_faces ∧ ∀ f, f ≠ s.parallel_faces → f.is_parallelogram) → s.is_prism

def proposition_2 : Prop :=
  ∀ s, (s.has_one_polygonal_face ∧ ∀ f, f ≠ s.polygonal_face → f.is_triangle) → s.is_pyramid

def proposition_3 : Prop :=
  ∀ p, (p.is_pyramid ∧ ∃ plane, plane.parallel_to p.base) → (p.cut_with plane).is_frustum

theorem true_statements_count_is_zero (h1 : ¬ proposition_1) (h2 : ¬ proposition_2) (h3 : ¬ proposition_3) : 
  (if proposition_1 then 1 else 0) + (if proposition_2 then 1 else 0) + (if proposition_3 then 1 else 0) = 0 :=
by
  sorry

end true_statements_count_is_zero_l225_225400


namespace problem1_problem2_l225_225365

-- Define the sequence {a_n}
def a : ℕ → ℕ 
| 0       := 1  -- Since a_1 = 1
| (n + 1) := 2 * (finset.range (n + 1)).sum (λ i, a i) + 1  -- a_{n+1} = 2S_n + 1

/-- Problem 1: Prove the general formula for a_n -/
theorem problem1 : ∀ n, a n = 3^(n-1) :=
by
  sorry

-- Define the sequence {b_n} which is { (2n+1)/a_n }
def b (n : ℕ) : ℚ := (2 * n + 1) / a n

-- Define the sum of the first n terms of the sequence {b_n}
def T (n : ℕ) := (finset.range n).sum (λ i, b (i+1))

/-- Problem 2: Prove the sum formula for T_n -/
theorem problem2 : ∀ n, T n = 9 - 9 * (n + 2) / (2 * 3^n) :=
by
  sorry

end problem1_problem2_l225_225365


namespace coeff_x3_in_x1_plus_x_to_6_l225_225616

theorem coeff_x3_in_x1_plus_x_to_6 (x : ℝ) : 
  coefficient (expansion (x * (1 + x)^6)) 3 = 15 :=
sorry

end coeff_x3_in_x1_plus_x_to_6_l225_225616


namespace vehicle_flow_mod_15_l225_225896

theorem vehicle_flow_mod_15
  (vehicle_length : ℝ := 5)
  (max_speed : ℕ := 100)
  (speed_interval : ℕ := 10)
  (distance_multiplier : ℕ := 10)
  (N : ℕ := 2000) :
  (N % 15) = 5 := 
sorry

end vehicle_flow_mod_15_l225_225896


namespace series_converges_to_16_l225_225503

noncomputable def s : ℝ := 
  Classical.some (Real.exists_pos_solution (λ x : ℝ, x^4 + (1/4)*x - (1/2)))

theorem series_converges_to_16 :
  s^4 + (1/4)*s - (1/2) = 0 →
  (∑' n : ℕ, (n + 1) * s ^ (1 + 3 * n)) = 16 :=
by {
  intro h,
  let r : ℝ := ∑' n : ℕ, (n + 1) * s ^ (1 + 3 * n),
  sorry
}

end series_converges_to_16_l225_225503


namespace ratio_of_areas_l225_225452

theorem ratio_of_areas (ABCD : Type) [square ABCD] 
  (A B C D E F : ABCD)
  (s : ℝ)
  (h_side_length : ∀ a b : ABCD, side_length s)
  (h_ratio_AE_EB : AE = 3 * EB)
  (h_ratio_DF_FC : DF = 3 * FC) :
  ratio_of_areas (triangle_area B F C) (square_area ABCD) = 1 / 8 := 
sorry

end ratio_of_areas_l225_225452


namespace jane_runs_3_km_l225_225852

-- We assume Jane's running time and her speed are given
def jane_time_hours : ℕ := 2
def jane_speed_m_per_min : ℕ := 25

-- We convert hours to minutes
def hours_to_minutes (hours : ℕ) : ℕ := hours * 60

-- We calculate total distance in meters
def total_distance_meters (time_hours : ℕ) (speed_m_per_min : ℕ) : ℕ :=
  speed_m_per_min * (hours_to_minutes time_hours)

-- We convert meters to kilometers
def meters_to_kilometers (meters : ℕ) : ℝ := meters / 1000

-- Define Jane's total running distance in kilometers
def jane_distance_km : ℝ := meters_to_kilometers (total_distance_meters jane_time_hours jane_speed_m_per_min)

theorem jane_runs_3_km : jane_distance_km = 3 := 
by
  sorry

end jane_runs_3_km_l225_225852


namespace gcd_324_243_l225_225182

-- Define the numbers involved in the problem.
def a : ℕ := 324
def b : ℕ := 243

-- State the theorem that the GCD of a and b is 81.
theorem gcd_324_243 : Nat.gcd a b = 81 := by
  sorry

end gcd_324_243_l225_225182


namespace inverse_function_log_eq_log_base_2_l225_225394

variable {a : ℝ} (ha : 0 < a ∧ a ≠ 1) (h : (a^2, a) ∈ (λ x, Real.log x / Real.log a) '' Set.univ)

theorem inverse_function_log_eq_log_base_2 : a = 2 → (λ x, Real.log x / Real.log a) = (λ x, Real.log x / Real.log 2) :=
by
  intros ha h
  sorry

end inverse_function_log_eq_log_base_2_l225_225394


namespace probability_of_one_marble_each_color_l225_225753

theorem probability_of_one_marble_each_color :
  let red := 3
  let blue := 3
  let green := 3
  let yellow := 3
  let total_marbles := red + blue + green + yellow
  let selection_size := 4
  (4 * 3^4 : ℚ) / finset.card (finset.Ico 1 (finset.card (finset.range total_marbles).choose selection_size)) = 9 / 55 := 
by {
  sorry
}

end probability_of_one_marble_each_color_l225_225753


namespace min_value_of_16b_over_ac_l225_225088

variables {A B C : Type} [real_linear_space A] [real_linear_space B] [real_linear_space C]
variables (a b c : ℝ) (angle_A angle_B angle_C : ℝ) (BC AB : A)

-- Conditions from the problem
def conditions :=
  0 < angle_B ∧ angle_B < real.pi / 2 ∧
  real.cos angle_B * real.cos angle_B + 0.5 * real.sin (2 * angle_B) = 1 ∧
  real.norm (BC + AB) = 3

-- The theorem we want to prove
theorem min_value_of_16b_over_ac (h: conditions) :
  16 * b / (a * c) = (16 / 3) * (2 - real.sqrt 2) :=
sorry

end min_value_of_16b_over_ac_l225_225088


namespace smallest_perfect_cube_divisor_l225_225124

theorem smallest_perfect_cube_divisor (p q r : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] [hr : Fact (Nat.Prime r)] (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ k : ℕ, (k = (p * q * r^2)^3) ∧ (∃ n, n = p * q^3 * r^4 ∧ n ∣ k) := 
sorry

end smallest_perfect_cube_divisor_l225_225124


namespace exists_coprime_l225_225265

noncomputable def sequence (K : ℕ) (hK : 1 < K) : ℕ → ℕ
| 1       := 1
| 2       := K
| (n + 3) := K * sequence hK (n + 2) - sequence hK (n + 1)

theorem exists_coprime {K : ℕ} (hK : 1 < K) (n : ℕ) :
  ∃ m, n < m ∧ Nat.gcd ((sequence hK m) - 1) (sequence hK n) = 1 :=
sorry

end exists_coprime_l225_225265


namespace projection_correct_angle_range_equilateral_angle_perpendicular_dot_product_zero_l225_225597

variables {a b : EuclideanSpace ℝ (Fin 2)}
noncomputable def projection (a b : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  (a • b) / (∥b∥^2) • b

theorem projection_correct (a b : EuclideanSpace ℝ (Fin 2)) :
  projection a b = ((a • b) / ∥b∥) * (b / ∥b∥) :=
sorry

theorem angle_range (a b : EuclideanSpace ℝ (Fin 2)) (h : a • b < 0) :
  (π / 2 < real.angle a b) ∧ (real.angle a b ≤ π) :=
sorry

theorem equilateral_angle {A B C : EuclideanSpace ℝ (Fin 2)} (h : is_equilateral_triangle A B C) :
  real.angle (B - A) (C - B) ≠ π / 3 :=
sorry 

theorem perpendicular_dot_product_zero (a b : EuclideanSpace ℝ (Fin 2)) (h : a • b = 0) :
  real.angle a b = π / 2 :=
sorry

end projection_correct_angle_range_equilateral_angle_perpendicular_dot_product_zero_l225_225597


namespace composite_shape_rotation_l225_225595

open Function

theorem composite_shape_rotation
  (S T P : Type) -- Types representing Square, Triangle, and Pentagon
  (composite_shape : S ⊕ T ⊕ P)
  (initial_positions : Σ x : S ⊕ T ⊕ P, (x = Sum.inl Sum.inl S) ∨ (x = Sum.inr (S ⊕ T)) ∨ (x = Sum.inr (P ⊕ T)))
  (rotate_150 : Π x : S ⊕ T ⊕ P, S ⊕ T ⊕ P) -- Rotation function
  (rotation_150_property : 
    ∀ x,
      let rotated := rotate_150 x in
      (x = Sum.inl Sum.inl S → rotated = Sum.inr (P ⊕ T))
      ∧ (x = Sum.inr (T ⊕ S) → rotated = Sum.inl Sum.inl S)
      ∧ (x = Sum.inr (P ⊕ T) → rotated = Sum.inr (T ⊕ S)))
  :
  (let rotated_square := rotate_150 (Sum.inl Sum.inl S),
       rotated_triangle := rotate_150 (Sum.inr (T ⊕ S)),
       rotated_pentagon := rotate_150 (Sum.inr (P ⊕ T))
   in
   rotated_square = Sum.inr (T ⊕ S) 
   ∧ rotated_triangle = Sum.inr (P ⊕ T) 
   ∧ rotated_pentagon = Sum.inl Sum.inl S) :=
sorry

end composite_shape_rotation_l225_225595


namespace recurring_fraction_division_l225_225222

/--
Given x = 0.\overline{36} and y = 0.\overline{12}, prove that x / y = 3.
-/
theorem recurring_fraction_division 
  (x y : ℝ)
  (h1 : x = 0.36 + 0.0036 + 0.000036 + 0.00000036 + ......) -- representation of 0.\overline{36}
  (h2 : y = 0.12 + 0.0012 + 0.000012 + 0.00000012 + ......) -- representation of 0.\overline{12}
  : x / y = 3 :=
  sorry

end recurring_fraction_division_l225_225222


namespace volume_ratio_l225_225948

theorem volume_ratio (a b : ℝ) (h : a^2 / b^2 = 9 / 25) : b^3 / a^3 = 125 / 27 :=
by
  -- Skipping the proof by adding 'sorry'
  sorry

end volume_ratio_l225_225948


namespace quadratic_inequality_solution_sets_l225_225364

theorem quadratic_inequality_solution_sets (a b c x : ℝ) :
  (a = 1 ∧ b = -2 →
    (ax^2 + x + b > 0 → ((ax^2 + x + b).roots = (-2, 1))
      ∧ 
      (ax^2 - (c + b)x + bc < 0 → 
        (c = -2 → (ax^2 - (c + b)x + bc).roots = ∅) ∧
        (c > -2 → ∃ y z : ℝ, (ax^2 - (c + b)x + bc).roots = Ioo y z ∧ y = -2 ∧ z = c) ∧
        (c < -2 → ∃ y z : ℝ, (ax^2 - (c + b)x + bc).roots = Ioo y z ∧ y = c ∧ z = -2)))) :=
sorry

end quadratic_inequality_solution_sets_l225_225364


namespace union_complement_PQ_l225_225414

noncomputable def P : set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
noncomputable def Q : set ℝ := {x | x^2 ≥ 4}
noncomputable def complementQ : set ℝ := {x | -2 < x ∧ x < 2}

theorem union_complement_PQ :
  P ∪ complementQ = {x | -2 < x ∧ x ≤ 3} :=
by
  sorry

end union_complement_PQ_l225_225414


namespace length_of_side_of_triangle_poq_l225_225149

theorem length_of_side_of_triangle_poq : 
  ∀ (p : ℝ) (P Q : ℝ×ℝ), P = (p, -p^2) ∧ Q = (-p, -p^2) → 
  dist (0, 0) P = dist (0, 0) Q → 
  dist (0, 0) P = √2 := by sorry

end length_of_side_of_triangle_poq_l225_225149


namespace vika_made_84_dollars_l225_225908

-- Define the amount of money Saheed, Kayla, and Vika made
variable (S K V : ℕ)

-- Given conditions
def condition1 : Prop := S = 4 * K
def condition2 : Prop := K = V - 30
def condition3 : Prop := S = 216

-- Statement to prove
theorem vika_made_84_dollars (S K V : ℕ) (h1 : condition1 S K) (h2 : condition2 K V) (h3 : condition3 S) : 
  V = 84 :=
by sorry

end vika_made_84_dollars_l225_225908


namespace circumcircle_radius_l225_225031

theorem circumcircle_radius (a b c : ℕ) (h1 : a = 3) (h2 : b = 5) (h3 : c = 7) : 
  let R := c / (2 * (Real.sin (Real.acos ((a^2 + b^2 - c^2) / (2 * a * b)))) ) in 
  R = 7 * Real.sqrt 3 / 3 :=
by
  sorry

end circumcircle_radius_l225_225031


namespace equidistant_points_l225_225488

variables (P : ℝ × ℝ × ℝ)
variables (A B C D : ℝ × ℝ × ℝ)

def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem equidistant_points :
  let A := (10, 0, 0)
  let B := (0, -6, 0)
  let C := (0, 0, 8)
  let D := (0, 0, 0)
  P = (5, -3, 4)
  P = (5, -3, 4) → (dist P A = dist P B) ∧ (dist P A = dist P C) ∧ (dist P A = dist P D) :=
begin
  intros,
  sorry
end

end equidistant_points_l225_225488


namespace polynomial_solution_l225_225000

theorem polynomial_solution (P : ℝ → ℝ) (hP : ∀ x : ℝ, (x + 1) * P (x - 1) + (x - 1) * P (x + 1) = 2 * x * P x) :
  ∃ (a d : ℝ), ∀ x : ℝ, P x = a * x^3 - a * x + d := 
sorry

end polynomial_solution_l225_225000


namespace X_investment_l225_225247

theorem X_investment (Y_investment : ℕ) (profit_ratio_XY : ℕ × ℕ) (simpl_ratio_XY : ℕ × ℕ)
  (profit_ratio_simpl_eq : profit_ratio_XY = (2, 6))
  (simplified_ratio_eq : simpl_ratio_XY = (1, 3))
  (Y_investment_amount : Y_investment = 15000) :
  let part_value := Y_investment / simpl_ratio_XY.2 in
  let X_investment := part_value * simpl_ratio_XY.1 in
  X_investment = 5000 :=
by
  sorry

end X_investment_l225_225247


namespace distance_from_M0_to_plane_is_4sqrt14_l225_225003

-- Define points in 3D space
structure Point3D := (x : ℝ) (y : ℝ) (z : ℝ)

-- Given points
def M0 : Point3D := ⟨14, -3, 7⟩
def M1 : Point3D := ⟨2, -1, -2⟩
def M2 : Point3D := ⟨1, 2, 1⟩
def M3 : Point3D := ⟨5, 0, -6⟩

-- Function to compute distance from point to plane
def distance_to_plane (p : Point3D) (A B C D : ℝ) : ℝ :=
  abs (A * p.x + B * p.y + C * p.z + D) / real.sqrt (A^2 + B^2 + C^2)

-- Plane equation coefficients derived from the points M1, M2, M3
noncomputable def plane_coefficients : (ℝ × ℝ × ℝ × ℝ) :=
  let A := -3 in
  let B := 1 in
  let C := -2 in
  let D := 3 in
  (A, B, C, D)

-- The final theorem to prove the distance is 4√14
theorem distance_from_M0_to_plane_is_4sqrt14 :
  distance_to_plane M0 (plane_coefficients.1) (plane_coefficients.2) (plane_coefficients.3) (plane_coefficients.4) =
  4 * real.sqrt 14 :=
by
  sorry

end distance_from_M0_to_plane_is_4sqrt14_l225_225003


namespace projection_onto_plane_correct_l225_225016

   noncomputable def vectorProjectionToPlane : ℝ × ℝ × ℝ :=
     let v := (2, 3, -1)
     let n := (5, 4, -7)
     let dot_product_v_n := 2 * 5 + 3 * 4 + -1 * -7
     let magnitude_squared_n := 5^2 + 4^2 + (-7)^2
     let scalar_projection := dot_product_v_n / magnitude_squared_n
     let projection_onto_n := (scalar_projection * 5, scalar_projection * 4, scalar_projection * -7)
     let projection_onto_plane := (2 - projection_onto_n.1, 3 - projection_onto_n.2, -1 - projection_onto_n.3)
     projection_onto_plane

   theorem projection_onto_plane_correct : vectorProjectionToPlane = (35/90, 154/90, -7/90) :=
   by
     sorry
   
end projection_onto_plane_correct_l225_225016


namespace point_satisfies_condition_l225_225485

-- Define Point in 3D space as tuple
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define distance between two points in 3D space
def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Given points A, B, C, D
def A : Point3D := ⟨10, 0, 0⟩
def B : Point3D := ⟨0, -6, 0⟩
def C : Point3D := ⟨0, 0, 8⟩
def D : Point3D := ⟨0, 0, 0⟩

-- Point P that satisfies AP = BP = CP = DP
def P : Point3D := ⟨5, -3, 4⟩

-- Proof that P satisfies the condition
theorem point_satisfies_condition : 
  distance P A = distance P B ∧ 
  distance P B = distance P C ∧ 
  distance P C = distance P D :=
by
  sorry

end point_satisfies_condition_l225_225485


namespace remainder_1632_mul_2024_mod_400_l225_225588

theorem remainder_1632_mul_2024_mod_400 :
  (1632 * 2024) % 400 = 368 :=
by {
  have h1: 1632 % 400 = 32 := sorry,
  have h2: 2024 % 400 = 24 := sorry,
  have h3: (32 * 24) % 400 = 768 % 400 := sorry,
  have h4: 768 % 400 = 368 := sorry,
  rw [←h1, ←h2, h3, h4],
  exact rfl,
}

end remainder_1632_mul_2024_mod_400_l225_225588


namespace crayons_selection_l225_225945

theorem crayons_selection :
  let total_crayons := 15
  let metallic_crayons := 2
  let non_metallic_crayons := total_crayons - metallic_crayons
  let choose (n k : ℕ) := n! / (k! * (n - k)!)
  2 * choose 13 4 = 1430 :=
by
  let total_crayons := 15
  let metallic_crayons := 2
  let non_metallic_crayons := total_crayons - metallic_crayons
  let choose (n k : ℕ) := n! / (k! * (n - k)!)
  exact have h_choose : choose 13 4 = 715 from sorry,
        by rw [h_choose]; norm_num

end crayons_selection_l225_225945


namespace mixture_ratio_proof_l225_225831

noncomputable def initial_mixture_volume : ℕ := 60
noncomputable def added_water_volume : ℕ := 24
noncomputable def milk_ratio : ℕ := 3
noncomputable def water_ratio : ℕ := 2
noncomputable def juice_ratio : ℕ := 1
noncomputable def total_ratio_parts : ℕ := milk_ratio + water_ratio + juice_ratio

noncomputable def milk_volume : ℕ := (milk_ratio * initial_mixture_volume) / total_ratio_parts
noncomputable def water_volume : ℕ := (water_ratio * initial_mixture_volume) / total_ratio_parts
noncomputable def juice_volume : ℕ := (juice_ratio * initial_mixture_volume) / total_ratio_parts

noncomputable def new_water_volume : ℕ := water_volume + added_water_volume
noncomputable def new_total_volume : ℕ := initial_mixture_volume + added_water_volume

noncomputable def gcd (a b : ℕ) : ℕ := nat.gcd a b
noncomputable def gcd_three (a b c : ℕ) : ℕ := gcd (gcd a b) c

noncomputable def simplified_milk_volume : ℕ := milk_volume / gcd_three milk_volume new_water_volume juice_volume
noncomputable def simplified_water_volume : ℕ := new_water_volume / gcd_three milk_volume new_water_volume juice_volume
noncomputable def simplified_juice_volume : ℕ := juice_volume / gcd_three milk_volume new_water_volume juice_volume

theorem mixture_ratio_proof :
  (simplified_milk_volume = 15) ∧ (simplified_water_volume = 22) ∧ (simplified_juice_volume = 5) := 
by
  sorry

end mixture_ratio_proof_l225_225831


namespace radius_of_circle_l225_225924

-- Definitions based on conditions
def area_of_circle (r : ℝ) : ℝ := π * r^2

-- The problem statement
theorem radius_of_circle (r : ℝ) (h : area_of_circle r = 196 * π) : r = 14 := by
  sorry

end radius_of_circle_l225_225924


namespace triangle_area_is_48_l225_225577

-- Define the equations of the lines
def line1 (x : ℝ) : ℝ := 3 * x - 2
def line2 (x : ℝ) : ℝ := (1 / 3) * x + 2 / 3

-- Prove the area of the triangle described is 48 square units
theorem triangle_area_is_48 :
  let A := (19 : ℝ, 7 : ℝ) in
  let B := (3 : ℝ, 7 : ℝ) in
  let C := (1 : ℝ, 1 : ℝ) in
  let area := (1 / 2) * (B.1 - A.1) * (A.2 - C.2) in
  area = 48 :=
by
  sorry

end triangle_area_is_48_l225_225577


namespace movie_tickets_l225_225188

theorem movie_tickets (r h : ℕ) (h1 : r = 25) (h2 : h = 3 * r + 18) : h = 93 :=
by
  sorry

end movie_tickets_l225_225188


namespace students_passed_l225_225568

noncomputable def total_students : ℕ := 360
noncomputable def bombed : ℕ := (5 * total_students) / 12
noncomputable def not_bombed : ℕ := total_students - bombed
noncomputable def no_show : ℕ := (7 * not_bombed) / 15
noncomputable def remaining_after_no_show : ℕ := not_bombed - no_show
noncomputable def less_than_D : ℕ := 45
noncomputable def remaining_after_less_than_D : ℕ := remaining_after_no_show - less_than_D
noncomputable def technical_issues : ℕ := remaining_after_less_than_D / 8
noncomputable def passed_students : ℕ := remaining_after_less_than_D - technical_issues

theorem students_passed : passed_students = 59 := by
  sorry

end students_passed_l225_225568


namespace base8_addition_l225_225719

theorem base8_addition (X Y : ℕ) 
  (h1 : 5 * 8 + X + Y + 3 * 8 + 2 = 6 * 64 + 4 * 8 + X) :
  X + Y = 16 := by
  sorry

end base8_addition_l225_225719


namespace parallelogram_area_l225_225647

theorem parallelogram_area (base height : ℝ) (h_base : base = 10) (h_height : height = 7) :
  base * height = 70 := by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l225_225647


namespace polynomial_condition_satisfaction_l225_225334

open Polynomial

theorem polynomial_condition_satisfaction (P : Polynomial ℝ) :
  (∀ x : ℝ, x * (P.comp (X - C 1)).eval x = (x - 2) * P.eval x) →
  ∃ a : ℝ, P = C a * X^2 - C a * X :=
by
  intro H
  sorry

end polynomial_condition_satisfaction_l225_225334


namespace smoking_lung_cancer_independence_test_l225_225840

theorem smoking_lung_cancer_independence_test (k : ℝ) (alpha : ℝ) :
  k = 6.635 → alpha = 0.01 → 
  (∃ correct_statement : Prop, correct_statement = 
  "If it is concluded from the statistic that there is a relationship between smoking and lung cancer with a probability of error not exceeding \(0.01\), it means there is a \(1\%\) possibility of making an error in judgment.") :=
by
  intros h1 h2
  sorry

end smoking_lung_cancer_independence_test_l225_225840


namespace max_value_of_function_l225_225027

theorem max_value_of_function (x : ℝ) (h : x > 0) : 
    (∀ y, y = x * sqrt (1 - 4 * x^2) → y ≤ 1 / 4) ∧ (∃ y, y = x * sqrt (1 - 4 * x^2) ∧ y = 1 / 4) := 
sorry

end max_value_of_function_l225_225027


namespace ribbon_count_proof_l225_225828

theorem ribbon_count_proof :
  (∃ N : ℕ, (1 / 4) * N + (1 / 3) * N + (1 / 6) * N + 40 = N ∧ (∃ orange_ribbons : ℕ, orange_ribbons = (1 / 6) * N ∧ orange_ribbons ≈ 27)) :=
sorry

end ribbon_count_proof_l225_225828


namespace equal_angles_l225_225867

theorem equal_angles 
  {A B C D E F P Q : Type} [Geometry A B C D E F P Q] -- Assume a geometry structure
  (hD: D ∈ line_segment A C)
  (hE: E ∈ line_segment B D)
  (hF: F ∈ line_segment B C)
  (h_eq_angles: ∠ BAE = ∠ CAF)
  (hP: P ∈ line_segment B C)
  (hQ: Q ∈ line_segment B D)
  (h_parallel: E ≠ P ∧ EP ≈ QF ∧ EP ≈ DC): 
  ∠ BAP = ∠ QAC :=
sorry

end equal_angles_l225_225867


namespace second_number_is_16_l225_225100

-- Define the sequence with the given numbers.
def sequence : List Nat := [2, 4, 14, 6, 12, 8, 16]

-- Define the specific problem of determining whether the second number in the sequence is 16.
theorem second_number_is_16 : sequence[1] = 16 :=
by
  sorry

end second_number_is_16_l225_225100


namespace number_of_buckets_l225_225570

-- Defining the conditions
def total_mackerels : ℕ := 27
def mackerels_per_bucket : ℕ := 3

-- The theorem to prove
theorem number_of_buckets :
  total_mackerels / mackerels_per_bucket = 9 :=
sorry

end number_of_buckets_l225_225570


namespace emptying_time_first_vessel_emptying_time_second_vessel_equal_water_levels_time_l225_225959
noncomputable theory

def original_height : ℝ := sorry
def radius_base : ℝ := sorry
def radius_opening : ℝ := sorry
def g : ℝ := 9.8  -- acceleration due to gravity

def time_first_vessel (H R r : ℝ) : ℝ :=
  (2 * R^2) / (3 * r^2) * sqrt(H / (2 * g))

def time_second_vessel (H R r : ℝ) : ℝ :=
  (16 * R^2) / (9 * r^2) * sqrt(H / (2 * g))

def moment_water_levels_equal (H : ℝ) : ℝ :=
  0.15 * H

theorem emptying_time_first_vessel :
  time_first_vessel original_height radius_base radius_opening = (2 * radius_base ^ 2) / (3 * radius_opening ^ 2) * sqrt(original_height / (2 * g)) := sorry

theorem emptying_time_second_vessel :
  time_second_vessel original_height radius_base radius_opening = (16 * radius_base ^ 2) / (9 * radius_opening ^ 2) * sqrt(original_height / (2 * g)) := sorry

theorem equal_water_levels_time :
  moment_water_levels_equal original_height = 0.15 * original_height := sorry

end emptying_time_first_vessel_emptying_time_second_vessel_equal_water_levels_time_l225_225959


namespace boat_stream_ratio_l225_225282

theorem boat_stream_ratio (B S : ℝ) (h : 1 / (B - S) = 2 * (1 / (B + S))) : B / S = 3 :=
by
  sorry

end boat_stream_ratio_l225_225282


namespace probability_adjacent_seats_correct_l225_225944

noncomputable def total_number_of_ways : ℕ := Nat.choose 100 2

noncomputable def number_of_adjacent_pairs : ℕ := 10 * 9

noncomputable def probability_adjacent_seats : ℚ := number_of_adjacent_pairs / total_number_of_ways

theorem probability_adjacent_seats_correct :
  probability_adjacent_seats = 1 / 55 := 
  sorry

end probability_adjacent_seats_correct_l225_225944


namespace real_roots_of_quadratic_l225_225078

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem real_roots_of_quadratic (m : ℝ) : (∃ x : ℝ, x^2 - x - m = 0) ↔ m ≥ -1/4 := by
  sorry

end real_roots_of_quadratic_l225_225078


namespace odd_n_sum_vector_length_ge_one_l225_225614

-- Given definitions based on conditions
variables {O : Point} {g : Line} 
variables {P : ℕ → Point}
variables (OP : ℕ → Vector)
variables (n : ℕ)
variables (unit_vectors : ∀ i, ∥ OP i ∥ = 1)
variables (same_plane : ∀ i, PointOnPlane (P i) g)
variables (same_side : ∀ i, PointSameSide (P i) g)

-- Theorem to be proved
theorem odd_n_sum_vector_length_ge_one (h_odd : n % 2 = 1) :
  ∥ ∑ i in (range n), (OP i) ∥ ≥ 1 := 
sorry

end odd_n_sum_vector_length_ge_one_l225_225614


namespace analytical_expression_tangent_line_l225_225393
-- Import the necessary library

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := (1 / 4) * (f' 1) * x^2 + 2 * (f 1) * x - 4

-- Given conditions
axiom f_derivative_at_one : f' 1 = 8
axiom f_at_one : f 1 = 2

-- Part (Ⅰ) - Proving the analytical expression of the function
theorem analytical_expression :
  f x = 2 * x^2 + 4 * x - 4 := by
  sorry

-- Part (Ⅱ) - Proving the equation of the tangent line
theorem tangent_line :
  let x := 0 in
  let y := f 0 in
  let m := f' 0 in
  y = -4 ∧ m = 4 ∧ (λ x' y', y' = m * x' + y) := by
  sorry

end analytical_expression_tangent_line_l225_225393


namespace number_of_yellow_marbles_l225_225475

theorem number_of_yellow_marbles 
  (total_marbles : ℕ) 
  (red_marbles : ℕ) 
  (blue_marbles : ℕ) 
  (yellow_marbles : ℕ)
  (h1 : total_marbles = 85) 
  (h2 : red_marbles = 14) 
  (h3 : blue_marbles = 3 * red_marbles) 
  (h4 : yellow_marbles = total_marbles - (red_marbles + blue_marbles)) :
  yellow_marbles = 29 :=
  sorry

end number_of_yellow_marbles_l225_225475


namespace sequences_satisfy_general_form_l225_225061

variables {θ : ℝ}

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 1 then 1 else a_n (n-1) * real.cos θ - b_n (n-1) * real.sin θ

noncomputable def b_n (n : ℕ) : ℝ :=
  if n = 1 then real.tan θ else a_n (n-1) * real.sin θ + b_n (n-1) * real.cos θ

theorem sequences_satisfy_general_form :
  (∀ n, a_n n = real.sec θ * real.cos (n * θ)) ∧ (∀ n, b_n n = real.sec θ * real.sin (n * θ)) :=
sorry

end sequences_satisfy_general_form_l225_225061


namespace evaluate_f_log3_2_l225_225788

noncomputable def f : ℝ → ℝ := λ x, 3^x + 9^x

theorem evaluate_f_log3_2 : f (Real.log 2 / Real.log 3) = 6 :=
by
  sorry

end evaluate_f_log3_2_l225_225788


namespace sum_first_24_odd_numbers_l225_225605

theorem sum_first_24_odd_numbers : 
  (∑ i in range 24, (2 * i + 1)) = 576 :=
sorry

end sum_first_24_odd_numbers_l225_225605


namespace ethanol_in_full_tank_l225_225295

def total_ethanol (volume_A volume_B ethanol_percent_A ethanol_percent_B : ℝ) : ℝ :=
  (volume_A * ethanol_percent_A) + (volume_B * ethanol_percent_B)

def final_volume (tank_capacity volume_A : ℝ) : ℝ := tank_capacity - volume_A

theorem ethanol_in_full_tank:
  ∀ (tank_capacity volume_A ethanol_percent_A ethanol_percent_B : ℝ),
  tank_capacity = 204 ∧ volume_A = 66 ∧ ethanol_percent_A = 0.12 ∧ ethanol_percent_B = 0.16 →
  total_ethanol volume_A (final_volume tank_capacity volume_A) ethanol_percent_A ethanol_percent_B = 30 :=
by { intros, sorry }

end ethanol_in_full_tank_l225_225295


namespace sitting_people_l225_225705

variables {M I P A : Prop}

-- Conditions
axiom M_not_sitting : ¬ M
axiom A_not_sitting : ¬ A
axiom if_M_not_sitting_then_I_sitting : ¬ M → I
axiom if_I_sitting_then_P_sitting : I → P

theorem sitting_people : I ∧ P :=
by
  have I_sitting : I := if_M_not_sitting_then_I_sitting M_not_sitting
  have P_sitting : P := if_I_sitting_then_P_sitting I_sitting
  exact ⟨I_sitting, P_sitting⟩

end sitting_people_l225_225705


namespace quadrilateral_diagonal_length_l225_225336

theorem quadrilateral_diagonal_length (D A₁ A₂ : ℝ) (hA₁ : A₁ = 9) (hA₂ : A₂ = 6) (Area : ℝ) (hArea : Area = 165) :
  (1/2) * D * (A₁ + A₂) = Area → D = 22 :=
by
  -- Use the given conditions and solve to obtain D = 22
  intros
  sorry

end quadrilateral_diagonal_length_l225_225336


namespace longest_path_CP_equidistant_CD_l225_225275

theorem longest_path_CP_equidistant_CD (O A B C D P : Point)
  (h1 : diameter O A B = 16)
  (h2 : distance A C = 3)
  (h3 : distance B D = 5)
  (h4 : on_circle O P) :
  (distance C P + distance P D) = 8 * Real.sqrt 5 ↔
    equidistant P C D := 
sorry

end longest_path_CP_equidistant_CD_l225_225275


namespace lambda_range_l225_225370

def Point : Type := ℝ × ℝ

def A : Point := (2, 3)
def B : Point := (5, 4)
def C : Point := (7, 10)

def vector_sub (p1 p2 : Point) : Point := (p1.1 - p2.1, p1.2 - p2.2)

def vector_add (p1 p2 : Point) : Point := (p1.1 + p2.1, p1.2 + p2.2)

def scalar_mul (k : ℝ) (p : Point) : Point := (k * p.1, k * p.2)

theorem lambda_range (λ : ℝ) (P : Point) :
  vector_sub P A = vector_add (vector_sub B A) (scalar_mul λ (vector_sub C A)) →
  P.1 < 0 → P.2 < 0 →
  λ < -1 :=
by
  sorry

end lambda_range_l225_225370


namespace ellipse_equation_slope_product_lambda_range_l225_225373

-- Define the given conditions as mathematical structures and properties.
def Ellipse (a b c : ℝ) (center : ℝ × ℝ) (focus : ℝ × ℝ) : Prop :=
  center = (0, 0) ∧ focus = (0, 1) ∧ a > b ∧ b > 0 ∧ c = sqrt (a^2 - b^2)

-- Define the concept of eccentricity.
def eccentricity (a c : ℝ) (e : ℝ) : Prop :=
  e = c / a

-- Define the equation of an ellipse.
def ellipse_eq (x y a b : ℝ) : Prop :=
  y^2 / a^2 + x^2 / b^2 = 1

-- Problem statement (I): Prove the equation of the ellipse.
theorem ellipse_equation : ∃ (a b : ℝ), Ellipse a b 1 (0, 0) (0, 1) ∧ eccentricity a 1 (1 / 2) ∧ ellipse_eq y x 2 3 := 
sorry

-- Problem statement (II.1): Prove the value of k1 * k2.
theorem slope_product (M N : ℝ × ℝ) (k1 k2 : ℝ) (λ : ℝ) :
  (M ≠ N) ∧ (ellipse_eq (M.1) (M.2) 2 3) ∧ (ellipse_eq (N.1) (N.2) 2 3) ∧ (M.1 ≠ N.1) →
  k1 = (N.2 - M.2) / (N.1 - M.1) →
  k2 = (N.2 + M.2) / (N.1 + M.1) →
  -4 / 3 = k1 * k2 := 
sorry

-- Problem statement (II.2): Prove the range of λ given coordinates of A.
theorem lambda_range (A : ℝ × ℝ) (λ : ℝ) :
   A = (3 / 2, 1) ∧ eccentricity 2 1 (1 / 2) →
   -2 < λ ∧ λ < 2 ∧ λ ≠ 0 := 
sorry

end ellipse_equation_slope_product_lambda_range_l225_225373


namespace minimum_value_abs_a_plus_b_l225_225029

theorem minimum_value_abs_a_plus_b {ω φ a b : ℝ}
  (hω_pos : ω > 0)
  (hφ_bound : |φ| < π/2)
  (h_period : 2 * π / ω = 2)
  (hf_point : 2 * sin (φ) + 2 = 3)
  (h_symmetry_center : (a, b) = (-13/6, 2)) :
  abs (a + b) = 1/6 := 
sorry

end minimum_value_abs_a_plus_b_l225_225029


namespace meaningful_expression_l225_225591

theorem meaningful_expression (x : ℝ) : (∃ y, y = 5 / (Real.sqrt (x + 1))) ↔ x > -1 :=
by
  sorry

end meaningful_expression_l225_225591


namespace least_square_tiles_l225_225654

-- Define the dimensions of the room in centimeters
def room_length_cm : ℕ := 720
def room_width_cm : ℕ := 432

-- The following statement verifies the correct number of tiles
theorem least_square_tiles (length width : ℕ) (h_length : length = room_length_cm) (h_width : width = room_width_cm) : 
  let tile_size := Nat.gcd length width in
  let num_tiles_length := length / tile_size in
  let num_tiles_width := width / tile_size in
  num_tiles_length * num_tiles_width = 15 := 
by
  sorry

end least_square_tiles_l225_225654


namespace exactly_one_topic_not_chosen_l225_225202

-- Define the context and properties of the problem
def teachers : ℕ := 4
def topics : ℕ := 4

-- Define the target property
theorem exactly_one_topic_not_chosen : 
  (number_of_ways_that_exactly_one_topic_is_not_chosen teachers topics = 112) :=
sorry -- proof to be provided

end exactly_one_topic_not_chosen_l225_225202


namespace no_solution_exists_l225_225695

theorem no_solution_exists :
  ¬ ∃ a b : ℝ, a^2 + 3 * b^2 + 2 = 3 * a * b :=
by
  sorry

end no_solution_exists_l225_225695


namespace find_sin_and_sides_of_triangle_l225_225445

-- Definitions for the sides and angles in a triangle
variable {A B C : ℝ}
variable {a b c : ℝ}

-- Given conditions
variable h1 : cos (2 * C) = -1 / 4
variable h2 : a = 2
variable h3 : 2 * sin A = sin C

-- Required to prove in the Lean statement format
theorem find_sin_and_sides_of_triangle:
  sin C = sqrt 10 / 4 ∧
  (a = 2 → 2 * sin A = sin C → c = 4 ∧ (b = sqrt 6 ∨ b = 2 * sqrt 6)) :=
by
  sorry

end find_sin_and_sides_of_triangle_l225_225445


namespace num_proper_subsets_eq_seven_l225_225556

theorem num_proper_subsets_eq_seven {α : Type} [DecidableEq α] (s : Set α) (h : s = {1, 2, 3}) :
  s.powerset.card - 1 = 7 := by
  sorry

end num_proper_subsets_eq_seven_l225_225556


namespace two_pow_neg_y_l225_225070

theorem two_pow_neg_y (y : ℤ) (h : 2^(4 * y) = 16) : 2^(-y) = 1/2 :=
by {
  sorry
}

end two_pow_neg_y_l225_225070


namespace area_of_isosceles_triangle_l225_225102

-- Define the isosceles triangle and altitude
variables {A B C D : Type} [EuclideanGeometry] (is_isosceles : AB = BC)
variable {α : ℝ} -- angle BAC
variable {R : ℝ} -- radius of the circle passing through A, C, D

theorem area_of_isosceles_triangle 
  (h1 : isosceles) 
  (h2 : altitude CD) 
  (h3 : ∠BAC = α) 
  (h4 : circle_passing_through AC D radius R) : 
  area ABC = R^2 * tan(α) :=
sorry

end area_of_isosceles_triangle_l225_225102


namespace emma_avg_speed_last_segment_l225_225704

theorem emma_avg_speed_last_segment
  (total_distance : ℕ) (total_time_hours : ℕ)
  (speed_first_segment : ℕ) (speed_second_segment : ℕ) 
  (last_segment_mins : ℕ) (avg_speed_last_segment : ℕ) :
  total_distance = 150 →
  total_time_hours = 2 →
  speed_first_segment = 50 →
  speed_second_segment = 75 →
  last_segment_mins = 40 →
  avg_speed_last_segment = 100 := 
by
  intro h1 h2 h3 h4 h5
  -- We have:
  -- h1 : total_distance = 150
  -- h2 : total_time_hours = 2
  -- h3 : speed_first_segment = 50
  -- h4 : speed_second_segment = 75
  -- h5 : last_segment_mins = 40
  -- Therefore, we conclude:
  exact h5

end emma_avg_speed_last_segment_l225_225704


namespace age_problem_l225_225259

-- Define the conditions
variables (a b c : ℕ)

-- Assumptions based on conditions
theorem age_problem (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 37) : b = 14 :=
by {
  sorry   -- Placeholder for the detailed proof
}

end age_problem_l225_225259


namespace monic_quadratic_with_root_l225_225729

theorem monic_quadratic_with_root (a b : ℝ) (h_root1 : (a + b * I) ∈ ({2-3*I, 2+3*I} : set ℂ)) :
  ∃ p : polynomial ℂ, polynomial.monic p ∧ p.coeff 0 = 13 ∧ p.coeff 1 = -4 ∧ p.coeff 2 = 1 ∧ p.eval (a + b * I) = 0 := 
by
  sorry

end monic_quadratic_with_root_l225_225729


namespace absolute_value_of_h_l225_225562

theorem absolute_value_of_h {h : ℝ} :
  (∀ x : ℝ, (x^2 + 2 * h * x = 3) → (∃ r s : ℝ, r + s = -2 * h ∧ r * s = -3 ∧ r^2 + s^2 = 10)) →
  |h| = 1 :=
by
  sorry

end absolute_value_of_h_l225_225562


namespace modulus_of_complex_number_l225_225933

noncomputable def imaginary_unit : ℂ := complex.I

noncomputable def complex_number : ℂ := (2 * imaginary_unit) / (1 + imaginary_unit)

theorem modulus_of_complex_number : complex.abs complex_number = real.sqrt 2 := by
  sorry

end modulus_of_complex_number_l225_225933


namespace largest_number_of_right_angles_in_convex_octagon_l225_225584

theorem largest_number_of_right_angles_in_convex_octagon : 
  ∀ (angles : Fin 8 → ℝ), 
  (∀ i, 0 < angles i ∧ angles i < 180) → 
  (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5 + angles 6 + angles 7 = 1080) → 
  ∃ k, k ≤ 6 ∧ (∀ i < 8, if angles i = 90 then k = 6 else true) := 
by 
  sorry

end largest_number_of_right_angles_in_convex_octagon_l225_225584


namespace multiply_expression_l225_225522

theorem multiply_expression (x : ℝ) : (x^4 + 12 * x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end multiply_expression_l225_225522


namespace necessary_condition_l225_225284

theorem necessary_condition :
  ∃ x : ℝ, (x < 0 ∨ x > 2) → (2 * x^2 - 5 * x - 3 ≥ 0) :=
sorry

end necessary_condition_l225_225284


namespace a_is_sqrt_5_l225_225749

noncomputable def satisfying_a : Set ℝ :=
  {a : ℝ | ∀ x > 0, ((2 * x - 2 * a + log (x / a)) * (-2 * x^2 + a * x + 5)) ≤ 0}

theorem a_is_sqrt_5 : satisfying_a = {Real.sqrt 5} :=
sorry

end a_is_sqrt_5_l225_225749


namespace remove_blue_balls_for_80_percent_l225_225671

variable (totalBalls : ℕ) (redPercentage : ℚ) (desiredRedPercentage : ℚ) (redBalls : ℕ) (conditionBalls : ℕ)

-- Conditions from part a)
def urn_conditions : Prop :=
  totalBalls = 150 ∧ redPercentage = 0.40 ∧ desiredRedPercentage = 0.80 ∧
  redBalls = (redPercentage * totalBalls).toNat ∧
  conditionBalls = (totalBalls - redBalls)

-- Proof we need to establish
theorem remove_blue_balls_for_80_percent (totalBalls : ℕ) (redPercentage : ℚ) (desiredRedPercentage : ℚ) (redBalls : ℕ) (conditionBalls : ℕ) :
  urn_conditions totalBalls redPercentage desiredRedPercentage redBalls conditionBalls →
  ∃ blueBallsToRemove : ℕ, blueBallsToRemove = 75 ∧ 
  (redBalls.toRat / (totalBalls.toRat - blueBallsToRemove.toRat)) = desiredRedPercentage :=
by
  sorry

end remove_blue_balls_for_80_percent_l225_225671


namespace theater_ticket_sales_l225_225189

theorem theater_ticket_sales (R H : ℕ) (h1 : R = 25) (h2 : H = 3 * R + 18) : H = 93 :=
by
  sorry

end theater_ticket_sales_l225_225189


namespace number_of_equilateral_triangles_l225_225548

theorem number_of_equilateral_triangles :
  (∀ k : ℤ, k ∈ (Icc (-12 : ℤ) (12 : ℤ))) →
  (∀ x : ℝ, ∃ y1 y2 y3 : ℝ, (y1 = k) ∧ (y2 = sqrt (3) * x + 3 * k) ∧ (y3 = -sqrt (3) * x + 3 * k)) →
  count_equilateral_triangles 3 (Icc (-12 : ℤ) (12 : ℤ)) = 144 :=
by
  sorry

end number_of_equilateral_triangles_l225_225548


namespace sum_f_infinite_l225_225345

noncomputable def f (n : ℕ) : ℝ := ∑' k, (1 : ℝ) / k^n

theorem sum_f_infinite : ∑' n, f n = ∞ := sorry

end sum_f_infinite_l225_225345


namespace digit_150_of_5_div_37_l225_225582

theorem digit_150_of_5_div_37 : 
  let decimal_expansion : List ℕ := [1, 3, 5]
  stream := List.cycle decimal_expansion
in stream.nth 149 = 3 :=
sorry

end digit_150_of_5_div_37_l225_225582


namespace line_L_equation_l225_225812

noncomputable def line1_slope : ℝ := 2 / 3
noncomputable def line1_intercept : ℝ := 4

noncomputable def line2_slope : ℝ := (1 / 2) * line1_slope
noncomputable def line2_intercept : ℝ := 2 * line1_intercept

theorem line_L_equation :
  ∀ x : ℝ, (x : ℝ) ∈ (Set.Univ : Set ℝ) → 
  (∃ y : ℝ, y = line2_slope * x + line2_intercept) → 
  y = (1 / 3) * x + 8 :=
by
  sorry

end line_L_equation_l225_225812


namespace sum_of_single_digit_y_divisible_by_36_l225_225692

theorem sum_of_single_digit_y_divisible_by_36 :
  let last_two_digits := 72
  let sum_of_digits (y : ℕ) := 3 + 6 + y + 7 + 2
  let valid_y_values := {y | y < 10 ∧ last_two_digits % 4 = 0 ∧ sum_of_digits y % 9 = 0}
  ∑ y in valid_y_values, y = 9 :=
by
  let last_two_digits := 72
  let sum_of_digits y := 3 + 6 + y + 7 + 2
  let valid_y_values := {y | y < 10 ∧ last_two_digits % 4 = 0 ∧ sum_of_digits y % 9 = 0}
  sorry

end sum_of_single_digit_y_divisible_by_36_l225_225692


namespace square_area_l225_225228

theorem square_area (side_length : ℝ) (h : side_length = 10) : side_length * side_length = 100 := by
  sorry

end square_area_l225_225228


namespace range_of_f_l225_225402

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^3 + x else Real.sin x

theorem range_of_f : set.range f = set.Icc (-1 : ℝ) (Real.top) := by
  sorry

end range_of_f_l225_225402


namespace triangle_area_difference_l225_225842

theorem triangle_area_difference 
(angle_EAB_right : ∠EAB = 90)
(angle_ABC_right : ∠ABC = 90)
(length_AB : AB = 5)
(length_BC : BC = 8)
(length_AE : AE = 10)
(intersection_D : ∃ D, ∃ AC BE, D ∈ AC ∧ D ∈ BE) :
  area △ADE - area △BDC = 5 := by
  sorry

end triangle_area_difference_l225_225842


namespace length_segment_AB_l225_225338

-- Define the line and the circle
def line_eq (x y : ℝ) : Prop := y = -x + 2
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the distance formula
def distance_from_center_to_line (O : ℝ × ℝ) : ℝ :=
  let (x₀, y₀) := O in
  abs ((-x₀ + y₀ + 2) / real.sqrt 2)

lemma length_of_chord {O : ℝ × ℝ} (hO : O = (0, 0)) (r d : ℝ) (h_r : r = real.sqrt 3) (h_d : d = distance_from_center_to_line O)
  : 2 * real.sqrt (r^2 - d^2) = 2 :=
begin
  -- The actual proof would go here
  sorry
end

theorem length_segment_AB : 2 * real.sqrt ((real.sqrt 3)^2 - (distance_from_center_to_line (0,0))^2) = 2 :=
begin
  have center : (0,0) = (0,0), by refl,
  have radius : real.sqrt 3 = real.sqrt 3, by refl,
  have distance : distance_from_center_to_line (0,0) = real.sqrt 2, by
    rw [distance_from_center_to_line],
    rw [abs_of_nonneg],
    rw [div_eq_mul_inv],
    rw [mul_one_div],
    -- by finishing the calculations
  
    sorry,
  apply length_of_chord,
  { sorry },
  { exact radius },
  { exact distance },
end

end length_segment_AB_l225_225338


namespace roots_difference_squared_l225_225800

theorem roots_difference_squared
  {Φ ϕ : ℝ}
  (hΦ : Φ^2 - Φ - 2 = 0)
  (hϕ : ϕ^2 - ϕ - 2 = 0)
  (h_diff : Φ ≠ ϕ) :
  (Φ - ϕ)^2 = 9 :=
by sorry

end roots_difference_squared_l225_225800


namespace range_of_f_range_of_x_for_g_less_f_l225_225515

noncomputable def f (x : ℝ) : ℝ := |x - 2| + x
noncomputable def g (x : ℝ) : ℝ := |x + 1|

theorem range_of_f : set.range f = set.Ici 2 := sorry

theorem range_of_x_for_g_less_f : {x : ℝ | g x < f x} = (set.Ioo (-3) 1) ∪ (set.Ioi 3) := sorry

end range_of_f_range_of_x_for_g_less_f_l225_225515


namespace exists_zero_point_interval_01_l225_225463

noncomputable def f (x : ℝ) : ℝ := x^3 + 3*x - 3

theorem exists_zero_point_interval_01 :
  ∃ x ∈ (Set.Icc 0 1), f x = 0 :=
begin
  -- according to given values
  have f0 : f 0 = -3 := by norm_num,
  have f1 : f 1 = 1 := by norm_num,
  -- by the Intermediate Value Theorem
  have h : ∃ x ∈ (Set.Icc 0 1), f x = 0 := by {
    apply intermediate_value_theorem,
    sorry },
  exact h,
end

end exists_zero_point_interval_01_l225_225463


namespace employed_females_percentage_l225_225258

theorem employed_females_percentage (E M : ℝ) (hE : E = 60) (hM : M = 42) : ((E - M) / E) * 100 = 30 := by
  sorry

end employed_females_percentage_l225_225258


namespace angle_C_of_triangle_l225_225849

theorem angle_C_of_triangle (A B C : ℝ) (h1 : A + B = 110) (h2 : A + B + C = 180) : C = 70 := 
by
  sorry

end angle_C_of_triangle_l225_225849


namespace at_least_one_odd_round_trip_l225_225185

theorem at_least_one_odd_round_trip :
  ∀ (localities : Fin 1983 → Type) (airlines : Fin 10 → Type), 
  (∀ i j : Fin 1983, ∃ a : Fin 10, DirectService localities i j a) →
  (∀ a : Fin 10, RoundTripService localities a) →
  ∃ a : Fin 10, OddRoundTrip localities a :=
by
  sorry

end at_least_one_odd_round_trip_l225_225185


namespace cd_squared_eq_ae_times_bf_l225_225362

-- Define the Circle, points, and lines with the given conditions
variable (O : Type) [MetricSpace O]
variable (A B C D E F : O)
variable (circle_O : Circle O)
variable (chord_AB : Segment A B)
variable (tangent_ECF : Tangent E C F circle_O)

-- Define perpendicularity conditions
variable (perpendicular_CD : Perpendicular C D chord_AB)
variable (perpendicular_AE : Perpendicular A E tangent_ECF)
variable (perpendicular_BF : Perpendicular B F tangent_ECF)

-- Prove the required equation
theorem cd_squared_eq_ae_times_bf :
  ∀(CD AE BF : ℝ), (CD^2 = AE * BF) := by
  sorry

end cd_squared_eq_ae_times_bf_l225_225362


namespace part1_part2_l225_225035

-- Definitions of the sequences
def a (n : ℕ) : ℕ := 3 * n - 1
def b (n : ℕ) : ℕ := 3^n + n

-- Main theorem statements
theorem part1 {b : ℕ → ℕ} : (b 1 = 4) → (∀ n, b (n + 1) = 3 * (b n) - 2 * n + 1) → 
  ∀ m: ℕ, b (m + 1) - (m + 1) = 3 * (b m - m) ∧
  ∃ m, b m = a m := 
begin 
  sorry
end

theorem part2 (n : ℕ) : ∑ i in finset.range n, (\(i+1) in finset.range (3n-1)), 
  (a i = b i →  (b (3 * i) + n - 1) = c (∑ i) = ((9 * ((27^n) - 1)) / 26) +  ((n * ((3 * n) + 1))) / 2) := 
begin
  sorry
end

end part1_part2_l225_225035


namespace prime_exponent_of_3_l225_225151

theorem prime_exponent_of_3 (n : ℕ) (h₁ : nat.prime (1 + 2^n + 4^n)) : ∃ k : ℕ, n = 3^k :=
sorry

end prime_exponent_of_3_l225_225151


namespace sitting_people_l225_225706

variables {M I P A : Prop}

-- Conditions
axiom M_not_sitting : ¬ M
axiom A_not_sitting : ¬ A
axiom if_M_not_sitting_then_I_sitting : ¬ M → I
axiom if_I_sitting_then_P_sitting : I → P

theorem sitting_people : I ∧ P :=
by
  have I_sitting : I := if_M_not_sitting_then_I_sitting M_not_sitting
  have P_sitting : P := if_I_sitting_then_P_sitting I_sitting
  exact ⟨I_sitting, P_sitting⟩

end sitting_people_l225_225706


namespace sum_even_integers_between_5_and_20_l225_225234

theorem sum_even_integers_between_5_and_20 : 
  (finset.filter (λ x, x % 2 = 0) (finset.Ico 6 20)).sum id = 84 :=
begin
  sorry
end

end sum_even_integers_between_5_and_20_l225_225234


namespace solve_for_y_l225_225162

noncomputable def roots := [(-126 + Real.sqrt 13540) / 8, (-126 - Real.sqrt 13540) / 8]

theorem solve_for_y (y : ℝ) :
  (8*y^2 + 176*y + 2) / (3*y + 74) = 4*y + 2 →
  y = roots[0] ∨ y = roots[1] :=
by
  intros
  sorry

end solve_for_y_l225_225162


namespace find_a_b_max_min_values_of_f_l225_225051

section problem

variable {f : ℝ → ℝ} {a b : ℝ}

-- Given function and conditions
def f (x : ℝ) : ℝ := x^3 - 3 * a * x^2 + 3 * b * x
def tangent_line (x : ℝ) : ℝ := 1 - 12 * x

-- Given condition of tangency at the point (1, -11)
def tangent_condition : Prop := 
  f(1) = -11 ∧ (∃ m : ℝ, (∀ x, tangent_line x = -11 + m * (x - 1)).imp (λ _, m = 12) (λ t, t.snd = f'(1)))

-- Given interval
def interval : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

-- Correct answers to be proven
theorem find_a_b (h : tangent_condition) : a = 1 ∧ b = -3 := sorry

theorem max_min_values_of_f (h : tangent_condition) :
  (∃ max min : ℝ, (∀ x ∈ interval, f(x) ≤ max) ∧ (∀ x ∈ interval, f(x) ≥ min) ∧ max = 5 ∧ min = -27) := sorry

end problem

end find_a_b_max_min_values_of_f_l225_225051


namespace sector_area_correct_l225_225272

-- Define the given conditions as hypotheses
def central_angle : ℝ := 2 -- angle in radians
def arc_length : ℝ := 4 -- arc length in cm
def radius : ℝ := arc_length / central_angle
def sector_area : ℝ := (1 / 2) * (radius ^ 2) * central_angle

-- Specify the target theorem
theorem sector_area_correct : sector_area = 4 := by 
  sorry

end sector_area_correct_l225_225272


namespace imaginary_part_conjugate_l225_225132

open Complex

theorem imaginary_part_conjugate (z : ℂ) (h : z / (1 - I) = 1 + 2 * I) :
  Complex.im (conj z) = -1 := by
sorry

end imaginary_part_conjugate_l225_225132


namespace counterexamples_to_proposition_l225_225195

-- Definition of the conditions
def is_positive_odd_number (n : ℕ) : Prop := n > 0 ∧ n % 2 = 1

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.foldl (· + ·) 0

def no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits, d ≠ 0

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The proof problem
theorem counterexamples_to_proposition :
  {n : ℕ | is_positive_odd_number n ∧ sum_of_digits n = 4 ∧ no_zero_digits n ∧ ¬ is_prime n}.to_finset.card = 2 :=
by
  sorry

end counterexamples_to_proposition_l225_225195


namespace arithmetic_sequence_of_reciprocal_expression_for_a_l225_225769

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

axiom a_n_condition {n : ℕ} (hn : n ≥ 2) : a n + 2 * S n * S (n - 1) = 0
axiom a_1_condition : a 1 = 1 / 2

theorem arithmetic_sequence_of_reciprocal (n : ℕ) (hn : n ≥ 1) :
  1 / S n = 2 * n :=
sorry

theorem expression_for_a (n : ℕ) :
  a n = if n = 1 then 1 / 2 else -1 / (2 * n * (n - 1)) :=
sorry

end arithmetic_sequence_of_reciprocal_expression_for_a_l225_225769


namespace prime_pairs_solution_l225_225721

def is_prime (n : ℕ) : Prop := Nat.Prime n

def conditions (p q : ℕ) : Prop := 
  p^2 ∣ q^3 + 1 ∧ q^2 ∣ p^6 - 1

theorem prime_pairs_solution :
  ({(p, q) | is_prime p ∧ is_prime q ∧ conditions p q} = {(3, 2), (2, 3)}) :=
by
  sorry

end prime_pairs_solution_l225_225721


namespace translation_equivalence_l225_225955

variable {α : Type} [LinearOrderedField α]

def translate_left_2 (f : α → α) (x : α) : α := f(x + 2)
def translate_down_3 (g : α → α) (x : α) : α := g(x) - 3

theorem translation_equivalence (f : α → α) :
  ∀ x : α, translate_down_3 (translate_left_2 f) x = f(x + 2) - 3 :=
by
  intros
  unfold translate_left_2
  unfold translate_down_3
  sorry

end translation_equivalence_l225_225955


namespace box_dimensions_l225_225887

theorem box_dimensions (x y z : ℝ) (h1 : x * y * z = 160) 
  (h2 : y * z = 80) (h3 : x * z = 40) (h4 : x * y = 32) : 
  x = 4 ∧ y = 8 ∧ z = 10 :=
by
  -- Placeholder for the actual proof steps
  sorry

end box_dimensions_l225_225887


namespace smallest_total_cells_marked_l225_225914

-- Definitions based on problem conditions
def grid_height : ℕ := 8
def grid_width : ℕ := 13

def squares_per_height : ℕ := grid_height / 2
def squares_per_width : ℕ := grid_width / 2

def initial_marked_cells_per_square : ℕ := 1
def additional_marked_cells_per_square : ℕ := 1

def number_of_squares : ℕ := squares_per_height * squares_per_width
def initial_marked_cells : ℕ := number_of_squares * initial_marked_cells_per_square
def additional_marked_cells : ℕ := number_of_squares * additional_marked_cells_per_square

def total_marked_cells : ℕ := initial_marked_cells + additional_marked_cells

-- Statement of the proof problem
theorem smallest_total_cells_marked : total_marked_cells = 48 := by 
    -- Proof is not required as per the instruction
    sorry

end smallest_total_cells_marked_l225_225914


namespace count_special_four_digit_integers_l225_225424

def four_digit_integer (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
def less_than_3000 (n : ℕ) : Prop := n < 3000
def at_least_two_digits_same (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10 % 10, n / 10 % 10 % 10, n % 10] in
  ∃ i j, i ≠ j ∧ digits.nth i = digits.nth j
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem count_special_four_digit_integers : 
  ∃ (count : ℕ), count = 80 ∧ ∀ (n : ℕ), 
    four_digit_integer(n) ∧ less_than_3000(n) ∧ at_least_two_digits_same(n) ∧ divisible_by_5(n) → ∃ count = 80 :=
by
  sorry

end count_special_four_digit_integers_l225_225424


namespace ends_with_two_zeros_l225_225751

theorem ends_with_two_zeros (x y : ℕ) (h : (x^2 + x * y + y^2) % 10 = 0) : (x^2 + x * y + y^2) % 100 = 0 :=
sorry

end ends_with_two_zeros_l225_225751


namespace max_skew_lines_l225_225356

/-- Define the properties of a cube's edges and face diagonals. -/
structure Cube :=
  (edges : set (line))
  (face_diagonals : set (line))
  (body_diagonals : set (line))
  (condition : ∀ (L1 L2 : line), L1 ∈ edges ∪ face_diagonals ∪ body_diagonals → 
                                  L2 ∈ edges ∪ face_diagonals ∪ body_diagonals → 
                                  L1 ≠ L2 → skew L1 L2)

/-- The main theorem: the maximum value of n line segments so that any two of them are skew. -/
theorem max_skew_lines (cube : Cube) : ∃ n : ℕ, n = 4 ∧
  ∀ (lines : set line), (lines ⊆ cube.edges ∪ cube.face_diagonals ∪ cube.body_diagonals) → 
                         (∀ L1 L2 ∈ lines, L1 ≠ L2 → skew L1 L2) → 
                         lines.card = n :=
sorry

end max_skew_lines_l225_225356


namespace nells_ace_cards_now_l225_225138

-- Definitions based on conditions
def baseball_cards_initial := 438
def ace_cards_initial := 18
def baseball_cards_now := 178
def difference_bc_ac := 123

-- Proof statement to show the number of Ace cards Nell has now
theorem nells_ace_cards_now (baseball_cards_initial = 438) (ace_cards_initial = 18)
    (baseball_cards_now = 178) (difference_bc_ac = 123) :
    ∃ A : ℕ, baseball_cards_now = A + difference_bc_ac ∧ A = 55 :=
by
    sorry

end nells_ace_cards_now_l225_225138


namespace remainder_degree_l225_225968

theorem remainder_degree (f : Polynomial ℝ) (g : Polynomial ℝ) (deg_g : g.degree = 7) :
  ∃ d, d ∈ {0, 1, 2, 3, 4, 5, 6} ∧ (f % g).degree = d :=
sorry

end remainder_degree_l225_225968


namespace min_positive_period_max_value_l225_225186

noncomputable def function_y (x : ℝ) : ℝ := 4 * (Real.sin x) * (Real.cos x)

theorem min_positive_period (y : ℝ) (x : ℝ) (hx : function_y x = y) : 
  ∃ (p : ℝ), p > 0 ∧ (∀ z : ℝ, function_y (z + p) = function_y z) :=
by
  existsi π
  split
  linarith
  intros
  sorry

theorem max_value (x : ℝ) : 
  ∃ (m : ℝ), (∀ z : ℝ, function_y z ≤ m) ∧ (∃ w : ℝ, function_y w = m) :=
by
  existsi 2
  split
  intros
  sorry
  existsi π / 4
  sorry

end min_positive_period_max_value_l225_225186


namespace ellipse_and_fixed_point_l225_225032

-- Define the properties of ellipse
def ellipse_center_origin (e : ℝ) (focus : (ℝ × ℝ)) (C_eq : ℝ → ℝ → Prop) : Prop :=
  e = real.sqrt 2 / 2 ∧ focus = (0, 1) ∧ ∀ x y, C_eq x y ↔ x^2 + y^2 / 2 = 1

-- Define the properties about point T existence related to circle AB passing through it
def fixed_point_T (C_eq : ℝ → ℝ → Prop) (T : ℝ × ℝ) : Prop :=
  T = (1, 0) ∧ ∀ l S A B,
    (l S ∧ l A ∧ l B ∧ S = (-1 / 3, 0) ∧ C_eq A.1 A.2 ∧ C_eq B.1 B.2) →
    let AB_diam_circle (x y : ℝ) := (x - (A.1 + B.1) / 2)^2 + (y - (A.2 + B.2) / 2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4
    in AB_diam_circle T.1 T.2

-- Complete theorem statement
theorem ellipse_and_fixed_point :
  ∃ C_eq, ellipse_center_origin (real.sqrt 2 / 2) (0, 1) C_eq ∧ fixed_point_T C_eq (1, 0) :=
sorry

end ellipse_and_fixed_point_l225_225032


namespace find_triplets_geometric_and_arithmetic_prog_l225_225212

theorem find_triplets_geometric_and_arithmetic_prog :
  ∃ a1 a2 b1 b2,
    (a2 = a1 * ((12:ℚ) / a1) ∧ 12 = a1 * ((12:ℚ) / a1)^2) ∧
    (b2 = b1 + ((9:ℚ) - b1) / 2 ∧ 9 = b1 + 2 * (((9:ℚ) - b1) / 2)) ∧
    ((a1 = b1) ∧ (a2 = b2)) ∧ 
    (∀ (a1 a2 : ℚ), ((a1 = -9) ∧ (a2 = -6)) ∨ ((a1 = 15) ∧ (a2 = 12))) :=
by sorry

end find_triplets_geometric_and_arithmetic_prog_l225_225212


namespace peter_large_glasses_l225_225144

theorem peter_large_glasses (cost_small cost_large total_money small_glasses change num_large_glasses : ℕ)
    (h1 : cost_small = 3)
    (h2 : cost_large = 5)
    (h3 : total_money = 50)
    (h4 : small_glasses = 8)
    (h5 : change = 1)
    (h6 : total_money - change = 49)
    (h7 : small_glasses * cost_small = 24)
    (h8 : 49 - 24 = 25)
    (h9 : 25 / cost_large = 5) :
  num_large_glasses = 5 :=
by
  sorry

end peter_large_glasses_l225_225144


namespace ferry_P_travel_time_l225_225355

-- Define the constants and conditions
def ferry_speed_p : ℝ := 8
def ferry_ratio_distance_q_p : ℝ := 3
def ferry_speed_difference : ℝ := 4
def ferry_time_difference : ℝ := 2

-- Define the time variables 
variables (t_P t_Q : ℝ)

-- Ferry Q speed in terms of ferry P speed
def ferry_speed_q := ferry_speed_p + ferry_speed_difference

-- Ferry Q time in terms of ferry P time
def ferry_time_q := t_P + ferry_time_difference

-- Define the distances traveled by ferry P and ferry Q
def distance_p := ferry_speed_p * t_P
def distance_q := ferry_ratio_distance_q_p * distance_p

-- Lean statement to prove
theorem ferry_P_travel_time : t_P = 2 :=
by {
  -- Setting up the equation
  have h1 : distance_q = ferry_speed_q * ferry_time_q,
  exact (show ferry_ratio_distance_q_p * (ferry_speed_p * t_P) = (ferry_speed_p + ferry_speed_difference) * (t_P + ferry_time_difference), 
         from sorry),
  
  -- Solving the equation for t_P
  have h2 : 3 * 8 * t_P = 12 * (t_P + 2),
  exact (show 24 * t_P = 12 * t_P + 24, from h1),
  
  have h3 : 12 * t_P = 24,
  exact (show 24 * t_P - 12 * t_P = 24, from h2),

  have h4 : t_P = 2,
  exact (show 12 * t_P = 24, from h3),

  assumption
}

end ferry_P_travel_time_l225_225355


namespace number_of_country_albums_l225_225246

-- Declare the variables and the conditions
variables (c : ℕ) -- number of country albums
constants (pop_albums : ℕ) (songs_per_album : ℕ) (total_songs : ℕ)

-- Set the values for constants
noncomputable def pop_albums := 2
noncomputable def songs_per_album := 9
noncomputable def total_songs := 72

-- State the theorem to be proved
theorem number_of_country_albums : c = 6 ↔ 
  ∃ k : ℕ, k = (total_songs - pop_albums * songs_per_album) / songs_per_album ∧ k = 6 := 
sorry

end number_of_country_albums_l225_225246


namespace find_circle_radius_l225_225105

theorem find_circle_radius (AB CD EF r : ℝ) (h_parallel: ∀ (x : ℝ), tangent x AB = tangent x CD)
  (h_AB: AB = 7) (h_CD: CD = 12) (h_EF: EF = 25)
  (h_tangents: ∃ E F D, tangent D EF ∧ tangent E AB ∧ tangent F CD ∧ r = radius EF) :
  r = 5 :=
sorry

end find_circle_radius_l225_225105


namespace abs_difference_of_opposite_signs_l225_225072

theorem abs_difference_of_opposite_signs (a b : ℝ) (ha : |a| = 4) (hb : |b| = 2) (hdiff : a * b < 0) : |a - b| = 6 := 
sorry

end abs_difference_of_opposite_signs_l225_225072


namespace initial_storks_count_l225_225621

theorem initial_storks_count :
  ∃ (S : ℕ), 
    let birds := 4 in
    let storks := S in
    let joined_storks := 6 in
    let total_birds := birds in
    let total_storks := storks + joined_storks in
    total_storks = total_birds + 5 → 
    S = 3 :=
begin
  let birds := 4,
  let joined_storks := 6,
  let total_birds := birds,
  assume (S : ℕ) (h : S + joined_storks = total_birds + 5),
  use S,
  rw add_comm at h,
  rw [← h, add_comm, add_sub_cancel] at h,
  exact rfl,
end

end initial_storks_count_l225_225621


namespace minimum_area_of_quadrilateral_is_2sqrt2_l225_225771

noncomputable def C := (1 : ℝ, 1 : ℝ)

noncomputable def radius : ℝ := 1

def on_line (P : ℝ × ℝ) : Prop :=
  3 * P.1 + 4 * P.2 + 8 = 0

def is_tangent (P A : ℝ × ℝ) : Prop :=
  let circle := (x : ℝ, y : ℝ) → x^2 + y^2 - 2 * x - 2 * y + 1 = 0
  ∃ (k : ℝ), circle P = 0 ∧ circle A = 0 ∧ (P.1 - A.1) * (P.2 - A.2) = -1

theorem minimum_area_of_quadrilateral_is_2sqrt2
  (P : ℝ × ℝ) (A B : ℝ × ℝ)
  (hP : on_line P)
  (hPA : is_tangent P A)
  (hPB : is_tangent P B) :
  (2 * sqrt 2 : ℝ) = 2 * sqrt 2 := by 
  -- Proof statement without the actual proof
  sorry

end minimum_area_of_quadrilateral_is_2sqrt2_l225_225771


namespace annual_rent_per_square_foot_l225_225934

theorem annual_rent_per_square_foot 
  (monthly_rent : ℕ) 
  (length : ℕ) 
  (width : ℕ) 
  (area : ℕ)
  (annual_rent : ℕ) : 
  monthly_rent = 3600 → 
  length = 18 → 
  width = 20 → 
  area = length * width → 
  annual_rent = monthly_rent * 12 → 
  annual_rent / area = 120 :=
by
  sorry

end annual_rent_per_square_foot_l225_225934


namespace slow_rook_moves_exactly_15_moves_none_l225_225140

theorem slow_rook_moves_exactly_15_moves_none :
  ∀ (board : chessboard) (slow_rook : piece) (blind_king : piece) (sleeping_pawns : finset (ℕ × ℕ)),
    chessboard.rows board = 6 ∧ chessboard.columns board = 9 →
    piece.position slow_rook = (6, 1) →
    piece.position blind_king = (1, 1) →
    sleeping_pawns.card = 8 →
    (∀ p1 p2 ∈ sleeping_pawns, p1.2 ≠ p2.2) →                      -- No two Sleeping Pawns are in the same column.
    (∀ p ∈ sleeping_pawns, p.1 ≠ 6 ∧ p.1 ≠ 1) →                    -- No Sleeping Pawn shares a row with the Slow Rook or the Blind King.
    (∀ p ∈ sleeping_pawns, p.2 ≠ 9) →                               -- No Sleeping Pawn is in the rightmost column.
    rook_moves_exactly_tiles slow_rook slow_rook_moves = 15 → 
    False :=   
by 
  sorry

end slow_rook_moves_exactly_15_moves_none_l225_225140


namespace max_value_reciprocal_l225_225773

theorem max_value_reciprocal (A B C D E F : Type*) [MetricSpace A] [HasZero A] [HasOne A]
  (AB BC CA : A)
  (angle_BAC : A = 90)
  (is_square_ADEF : quadrilateral A D E F)
  (side_length_ADEF : A = 1)
  : max (1 / AB + 1 / BC + 1 / CA) = 1 + Real.sqrt(2) / 2 := sorry

end max_value_reciprocal_l225_225773


namespace mr_smith_payment_l225_225891

theorem mr_smith_payment {balance : ℝ} {percentage : ℝ} 
  (h_bal : balance = 150) (h_percent : percentage = 0.02) :
  (balance + balance * percentage) = 153 :=
by
  sorry

end mr_smith_payment_l225_225891


namespace chord_eq_line_l225_225411

-- Define the parabola and the midpoint of the chord
def parabola (x y : ℝ) : Prop := x^2 = -2 * y
def midpoint (xa ya xb yb : ℝ) (mx my : ℝ) : Prop := (xa + xb) / 2 = mx ∧ (ya + yb) / 2 = my

-- Statement: Given the parabola and the midpoint of the chord, prove the line equation
theorem chord_eq_line (x1 y1 x2 y2 : ℝ) (h1 : parabola x1 y1) (h2 : parabola x2 y2) 
  (h3 : midpoint x1 y1 x2 y2 (-1) (-5)) : 
  (∃ (m b : ℝ), (y1 = m * x1 + b) ∧ (y2 = m * x2 + b) ∧ (y1 + y2) = (m * (x1 + x2) + 2 * b) / 2) :=
sorry

end chord_eq_line_l225_225411


namespace max_value_of_expression_l225_225130

theorem max_value_of_expression 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : 4 * a + 5 * b < 100) :
  ab (100 - 4 * a - 5 * b) ≤ 50000 / 27 := 
sorry

end max_value_of_expression_l225_225130


namespace part1_distance_min_part2_positive_a_l225_225883

-- Define the functions f and g
def f (a x : ℝ) := a * x + Real.log x
def g (a x : ℝ) := a^2 * x^2

-- Define the statement for part (1)
theorem part1_distance_min (a : ℝ) (h : a = -1) : 
  let x := 1 / 2,
      y := f a x in
  Abs ((x - y + 3) / Real.sqrt 2) = (1 / 2) * (4 + Real.log 2) * Real.sqrt 2 :=
sorry

-- Define the statement for part (2)
theorem part2_positive_a : 
  ∃ (a : ℝ), (0 < a) ∧ (∀ x > 0, f a x ≤ g a x) ↔ (1 ≤ a) :=
sorry

end part1_distance_min_part2_positive_a_l225_225883


namespace find_some_number_l225_225615

-- Definitions based on the given condition
def some_number : ℝ := sorry
def equation := some_number * 3.6 / (0.04 * 0.1 * 0.007) = 990.0000000000001

-- An assertion/proof that given the equation, some_number equals 7.7
theorem find_some_number (h : equation) : some_number = 7.7 :=
sorry

end find_some_number_l225_225615


namespace peter_bought_large_glasses_l225_225146

-- Define the conditions as Lean definitions
def total_money : ℕ := 50
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def small_glasses_bought : ℕ := 8
def change_left : ℕ := 1

-- Define the number of large glasses bought
def large_glasses_bought (total_money : ℕ) (cost_small_glass : ℕ) (cost_large_glass : ℕ) (small_glasses_bought : ℕ) (change_left : ℕ) : ℕ :=
  let total_spent := total_money - change_left
  let spent_on_small := cost_small_glass * small_glasses_bought
  let spent_on_large := total_spent - spent_on_small
  spent_on_large / cost_large_glass

-- The theorem to be proven
theorem peter_bought_large_glasses : large_glasses_bought total_money cost_small_glass cost_large_glass small_glasses_bought change_left = 5 :=
by
  sorry

end peter_bought_large_glasses_l225_225146


namespace general_term_l225_225793

variable {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b)

-- Definition of the sequence x_n
def x : ℕ → ℝ
| 0       := 0
| (n + 1) := x n + a + real.sqrt (b^2 + 4 * a * (x n))

theorem general_term (n : ℕ) : x n = a * n^2 + b * n := 
sorry

end general_term_l225_225793


namespace color_triangle_l225_225524

noncomputable theory

open set real

-- Define a point in 2D plane
structure Point :=
(x : ℝ)
(y : ℝ)

-- Points colored either blue or red
inductive Color
| blue
| red

-- Given several points in the plane with a color
structure ColoredPlane :=
(points : finset (Point × Color))
(color_condition : ∀ c ∈ points, ∀ d ∈ points, ∀ e ∈ points, 
  (c.2 = d.2) → (d.2 = e.2) → (c ≠ d) → (d ≠ e) → (e ≠ c) → ¬ collinear {c.1, d.1, e.1})
(minColorPoints : ∀ b : Color, 3 ≤ (points.filter (λ p, p.2 = b)).card)

-- Prove that some three points of the same color form a non-degenerate triangle
-- with each side having at most two points of the other color
theorem color_triangle (plane : ColoredPlane) :
  ∃ (triangle : finset Point) (c : Color),
    triangle.card = 3 ∧
    (∀ p ∈ triangle, ∃ q ∈ plane.points, (q.1 = p) ∧ (q.2 = c)) ∧
    ∀ x y ∈ triangle, (x ≠ y) → (line_segment x y).finset_points.filter (λ p, p ∈ plane.points ∧ p.2 ≠ c).card ≤ 2 :=
sorry

end color_triangle_l225_225524


namespace horner_multiplications_additions_l225_225579

def f (x : ℝ) : ℝ := 6 * x^6 + 5

def x : ℝ := 2

theorem horner_multiplications_additions :
  (6 : ℕ) = 6 ∧ (6 : ℕ) = 6 := 
by 
  sorry

end horner_multiplications_additions_l225_225579


namespace combined_parent_age_difference_l225_225160

def father_age_at_sobha_birth : ℕ := 38
def mother_age_at_brother_birth : ℕ := 36
def brother_younger_than_sobha : ℕ := 4
def sister_younger_than_brother : ℕ := 3
def father_age_at_sister_birth : ℕ := 45
def mother_age_at_youngest_birth : ℕ := 34
def youngest_younger_than_sister : ℕ := 6

def mother_age_at_sobha_birth := mother_age_at_brother_birth - brother_younger_than_sobha
def father_age_at_youngest_birth := father_age_at_sister_birth + youngest_younger_than_sister

def combined_age_difference_at_sobha_birth := father_age_at_sobha_birth - mother_age_at_sobha_birth
def compounded_difference_at_sobha_brother_birth := 
  (father_age_at_sobha_birth + brother_younger_than_sobha) - mother_age_at_brother_birth
def mother_age_at_sister_birth := mother_age_at_brother_birth + sister_younger_than_brother
def compounded_difference_at_sobha_sister_birth := father_age_at_sister_birth - mother_age_at_sister_birth
def compounded_difference_at_youngest_birth := father_age_at_youngest_birth - mother_age_at_youngest_birth

def combined_age_difference := 
  combined_age_difference_at_sobha_birth + 
  compounded_difference_at_sobha_brother_birth + 
  compounded_difference_at_sobha_sister_birth + 
  compounded_difference_at_youngest_birth 

theorem combined_parent_age_difference : combined_age_difference = 35 := by
  sorry

end combined_parent_age_difference_l225_225160


namespace back_seat_can_hold_8_people_l225_225820

def totalPeopleOnSides : ℕ :=
  let left_seats := 15
  let right_seats := left_seats - 3
  let people_per_seat := 3
  (left_seats + right_seats) * people_per_seat

def bus_total_capacity : ℕ := 89

def back_seat_capacity : ℕ :=
  bus_total_capacity - totalPeopleOnSides

theorem back_seat_can_hold_8_people : back_seat_capacity = 8 := by
  sorry

end back_seat_can_hold_8_people_l225_225820


namespace locus_of_feet_of_perpendiculars_from_focus_l225_225754

def parabola_locus (p : ℝ) : Prop :=
  ∀ x y : ℝ, (y^2 = (p / 2) * x)

theorem locus_of_feet_of_perpendiculars_from_focus (p : ℝ) :
    parabola_locus p :=
by
  sorry

end locus_of_feet_of_perpendiculars_from_focus_l225_225754


namespace jars_contain_k_balls_eventually_l225_225201

theorem jars_contain_k_balls_eventually
  (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hkp : k < 2 * p + 1) :
  ∃ n : ℕ, ∃ x y : ℕ, x + y = 2 * p + 1 ∧ (x = k ∨ y = k) :=
by
  sorry

end jars_contain_k_balls_eventually_l225_225201


namespace number_of_valid_sequences_l225_225509

/- 
Let \( (a_1, a_2, \ldots, a_{15}) \) be a list of the first 15 positive integers such that 
for each \( 2 \leq i \leq 15 \), either \( a_i + 1 \) or \( a_i - 1 \) (or both) appear 
somewhere before \( a_i \) in the list. Additionally, \( a_1 \) must be either 1 or 2, 
and \( a_{15} \) must be either 14 or 15. How many such lists are there?
-/

/-- 
  There are exactly 16384 valid sequences that can be formed with the given conditions 
  and constraints.
-/
theorem number_of_valid_sequences : 
  (∃ (s : list ℕ), s.length = 15 ∧ 
    (∀ i, 2 ≤ i ∧ i ≤ 15 → (s.nth (i - 1)).map (λ ai, (s.take (i - 1)).any (λ aj, aj = ai + 1 ∨ aj = ai - 1)) = some true) ∧
    ((s.head' = some 1 ∨ s.head' = some 2) ∧ 
    ((s.last = some 14 ∧ s.get_or_else 13 0 = 13) ∨ (s.last = some 15 ∧ s.get_or_else 13 0 = 14)))) → 
    (count_valid_sequences s = 16384) :=
sorry

end number_of_valid_sequences_l225_225509


namespace beetle_closest_to_ant_l225_225141

noncomputable def find_closest_positions (M0 N0 : ℝ × ℝ) 
  (initial_angle_ant initial_angle_beetle : ℝ) 
  (radius_ant radius_beetle speed_ratio : ℝ)
  : set (ℝ × ℝ) :=
{ (2, 2 * Real.sqrt 3), (-4, 0), (2, -2 * Real.sqrt 3) }

theorem beetle_closest_to_ant :
  let M0 := (-1, Real.sqrt 3)
  let N0 := (2 * Real.sqrt 3, 2)
  let α0 := Real.arctan (Real.sqrt 3 / -1)
  let β0 := Real.arctan (2 / (2 * Real.sqrt 3))
  let ra := 2
  let rb := 4
  let v_ratio := 2
  find_closest_positions M0 N0 α0 β0 ra rb v_ratio
  = { (2, 2 * Real.sqrt 3), (-4, 0), (2, -2 * Real.sqrt 3) } :=
by 
  sorry

end beetle_closest_to_ant_l225_225141


namespace second_player_strategy_divisible_by_77_l225_225216

theorem second_player_strategy_divisible_by_77 :
  ∀ (player_move : ℕ → ℕ), 
    (∀ i, 1 ≤ player_move i ∧ player_move i ≤ 8) →
    (∀ i, 7 ≤ 12 → i + 6 ≤ 12 → player_move (i + 6) = 9 - player_move i) →
    ∃ (n : ℕ), n = list.foldl (λ acc i, acc * 10 + player_move i) 0 [0,1,2,3,4,5,6,7,8,9,10,11] ∧ n % 77 = 0 :=
by sorry

end second_player_strategy_divisible_by_77_l225_225216


namespace number_of_students_with_uncool_parents_l225_225207

theorem number_of_students_with_uncool_parents (total_students cool_dads cool_moms cool_both : ℕ) :
    total_students = 40 →
    cool_dads = 20 →
    cool_moms = 25 →
    cool_both = 10  →
    (total_students - (cool_dads - cool_both + cool_moms - cool_both + cool_both) = 5) :=
by
  intros h_total h_dads h_moms h_both
  rw [h_total, h_dads, h_moms, h_both]
  norm_num
  sorry

end number_of_students_with_uncool_parents_l225_225207


namespace number_of_real_roots_l225_225500

-- Definitions of the non-zero real numbers and the positive real number
variables {a b c k : ℝ}
variable (x : ℝ)

-- Hypotheses ensuring the conditions from the given problem
axiom h₁ : a ≠ 0
axiom h₂ : b ≠ 0
axiom h₃ : c ≠ 0
axiom h₄ : k > 0

-- The main statement to be proved in Lean
theorem number_of_real_roots :
  \(\begin{vmatrix} k*x & c & -b \\ -c & k*x & a \\ b & -a & k*x \end{vmatrix} = 0 \rightarrow 
    x = 0 \wedge ∀ y, \(\begin{vmatrix} k*y & c & -b \\ -c & k*y & a \\ b & -a & k*y \end{vmatrix} = 0 \rightarrow y = 0) :=
sorry

end number_of_real_roots_l225_225500


namespace julio_twice_james_in_years_l225_225855

noncomputable def years_until_julio_twice_james := 
  let x := 14
  (36 + x = 2 * (11 + x))

theorem julio_twice_james_in_years : 
  years_until_julio_twice_james := 
  by 
  sorry

end julio_twice_james_in_years_l225_225855


namespace sum_first_49_nat_nums_l225_225747

theorem sum_first_49_nat_nums : (Finset.range 50).sum (fun x => x) = 1225 := 
by
  sorry

end sum_first_49_nat_nums_l225_225747


namespace circle_trajectory_l225_225363

theorem circle_trajectory (x y : ℝ) (h1 : (x-5)^2 + (y+7)^2 = 16) (h2 : ∃ c : ℝ, c = ((x + 1 - 5)^2 + (y + 1 + 7)^2)): 
    ((x-5)^2+(y+7)^2 = 25 ∨ (x-5)^2+(y+7)^2 = 9) :=
by
  -- Proof is omitted
  sorry

end circle_trajectory_l225_225363


namespace maximum_sum_of_numbers_in_grid_l225_225818

theorem maximum_sum_of_numbers_in_grid :
  ∀ (grid : List (List ℕ)) (rect_cover : (ℕ × ℕ) → (ℕ × ℕ) → Prop),
  (∀ x y, rect_cover x y → x ≠ y → x.1 < 6 → x.2 < 6 → y.1 < 6 → y.2 < 6) →
  (∀ x y z w, rect_cover x y ∧ rect_cover z w → 
    (x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∨ (x.1 = z.1 ∨ x.2 = z.2) → 
    (x.1 = z.1 ∧ x.2 = y.2 ∨ x.2 = z.2 ∧ x.1 = y.1)) → False) →
  (36 = 6 * 6) →
  18 = 36 / 2 →
  342 = (18 * 19) :=
by
  intro grid rect_cover h_grid h_no_common_edge h_grid_size h_num_rectangles
  sorry

end maximum_sum_of_numbers_in_grid_l225_225818


namespace order_numbers_l225_225592

theorem order_numbers : (5 / 2 : ℝ) < (3 : ℝ) ∧ (3 : ℝ) < Real.sqrt (10) := 
by
  sorry

end order_numbers_l225_225592


namespace round_to_nearest_hundredth_l225_225218

theorem round_to_nearest_hundredth (x : ℝ) (h : x = 7.985) : 
  Float.round (x * 100) / 100 = 7.99 :=
by
  sorry

end round_to_nearest_hundredth_l225_225218


namespace candy_difference_l225_225353

theorem candy_difference (Frankie_candies Max_candies : ℕ) (hF : Frankie_candies = 74) (hM : Max_candies = 92) :
  Max_candies - Frankie_candies = 18 :=
by
  sorry

end candy_difference_l225_225353


namespace divisible_iff_l225_225903

theorem divisible_iff (m n k : ℕ) (h : m > n) : 
  (3^(k+1)) ∣ (4^m - 4^n) ↔ (3^k) ∣ (m - n) := 
sorry

end divisible_iff_l225_225903


namespace recreation_transportation_percentage_l225_225477

theorem recreation_transportation_percentage (W : ℝ) :
  let last_week_recreation := 0.20 * W,
      last_week_transportation := 0.10 * W,
      this_week_wages := 0.70 * W,
      this_week_recreation := 0.25 * this_week_wages,
      this_week_transportation := 0.15 * this_week_wages in
  let last_week_total := last_week_recreation + last_week_transportation,
      this_week_total := this_week_recreation + this_week_transportation,
      percentage := (this_week_total / last_week_total) * 100 in
  percentage ≈ 93.33 := -- Use approximation if necessary
sorry

end recreation_transportation_percentage_l225_225477


namespace quadratic_properties_l225_225351

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem quadratic_properties :
  let y := quadratic_function in
  (∀ x : ℝ, y x = x^2 + 2 * x - 3) ∧
  (∃ k : ℝ, ∀ x : ℝ, y x = (x + 1)^2 - 4 ∧
    k = -1 ∧ ∀ y_value, y k = y_value ∧
    y (k + x) > y k ∧ y (k - x) > y k) ∧
  (∀ x : ℝ, if x < -1 then ∀ x₁ x₂, x₁ < x₂ -> y x₁ > y x₂ else
    if x > -1 then ∀ x₁ x₂, x₁ < x₂ -> y x₁ < y x₂ else True) ∧
  (∃ min max : ℝ, min = -4 ∧ max = 12 ∧
    ∀ x ∈ set.Icc (-2 : ℝ) 3, min ≤ y x ∧ y x ≤ max) :=
by {
  let y := quadratic_function,
  sorry
}

end quadratic_properties_l225_225351


namespace cubic_roots_of_unity_l225_225882

namespace ComplexRoots

theorem cubic_roots_of_unity (α β : ℂ) (h1 : |α| = 1) (h2 : |β| = 1) (h3 : α + β + 1 = 0) :
  α^3 = 1 ∧ β^3 = 1 :=
by
  sorry

end ComplexRoots

end cubic_roots_of_unity_l225_225882


namespace inequality_division_l225_225427

variable (m n : ℝ)

theorem inequality_division (h : m > n) : (m / 4) > (n / 4) :=
sorry

end inequality_division_l225_225427


namespace range_h_is_0_3_l225_225319

noncomputable def h (x : ℝ) : ℝ := 3 / (1 + 5 * x^2)

theorem range_h_is_0_3 : 
  (∃ c d : ℝ, (∀ y : ℝ, ∃ x : ℝ, h x = y ↔ y ∈ set.Ioc c d) ∧ c + d = 3) :=
by 
  sorry

end range_h_is_0_3_l225_225319


namespace tangent_line_equation_at_1_l225_225005

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 + 2

def df (x : ℝ) : ℝ := 3 * x^2 - 4 * x

theorem tangent_line_equation_at_1 :
    let slope := df 1 in
    let point := (1 : ℝ, f 1) in
    let tangent_line := (λ x y : ℝ, x + y - 2 = 0) in
    slope = -1 ∧ point = (1, 1) ∧ tangent_line 1 1 :=
by
  sorry

end tangent_line_equation_at_1_l225_225005


namespace rectangle_square_area_ratio_eq_one_l225_225964

theorem rectangle_square_area_ratio_eq_one (r l w s: ℝ) (h1: l = 2 * w) (h2: r ^ 2 = (l / 2) ^ 2 + w ^ 2) (h3: s ^ 2 = 2 * r ^ 2) : 
  (l * w) / (s ^ 2) = 1 :=
by
sorry

end rectangle_square_area_ratio_eq_one_l225_225964


namespace inverse_sum_mod_7_l225_225581

theorem inverse_sum_mod_7 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ : ℤ) % 7 = 3 := by
  sorry

end inverse_sum_mod_7_l225_225581


namespace root_in_interval_l225_225550

def f (x : ℝ) : ℝ := 4 * x ^ 3 - 5 * x + 6

theorem root_in_interval : 
  (f (-2) < 0) ∧ (f (-1) > 0) → ∃ x : ℝ, x ∈ Ioo (-2 : ℝ) (-1 : ℝ) ∧ f x = 0 :=
begin
  sorry
end

end root_in_interval_l225_225550


namespace sqrt_inequality_and_equality_condition_l225_225529

theorem sqrt_inequality_and_equality_condition (a b c : ℝ) :
  (sqrt (a^2 + a * b + b^2) + sqrt (a^2 + a * c + c^2) >= sqrt (3 * a^2 + (a + b + c)^2))
    ∧ ((a = 0 ∧ b * c ≥ 0) ∨ b = c) :=
begin
  sorry,
end

end sqrt_inequality_and_equality_condition_l225_225529


namespace certain_number_x_l225_225232

theorem certain_number_x :
  ∃ x : ℤ, (287 * 287 + 269 * 269 - x * (287 * 269) = 324) ∧ (x = 2) := 
by {
  use 2,
  sorry
}

end certain_number_x_l225_225232


namespace geometric_locus_centers_l225_225419

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_parallel (l1 l2 : set Point) : Prop := 
  ∃ k : ℝ, ∀ p ∈ l1, ∀ q ∈ l2, (q.y - p.y) = k * (q.x - p.x)

noncomputable def geometric_locus (A B : Point) (l : set Point) : set (set Point) :=
  if is_parallel {A, B} l then
    {r : set Point | ∀ p ∈ r, ∃ q ∈ l, (p.y - q.y) = 0}
  else
    {r : set Point | ∀ p ∈ r, ∃ q₁ q₂ ∈ l, ¬(p = q₁ ∨ p = q₂) ∧ 
                             ((p.x - q₁.x) * (q₂.y - q₁.y) = (p.y - q₁.y) * (q₂.x - q₁.x))}

theorem geometric_locus_centers
  (A B : Point) (l : set Point) : 
  ∃ c : set Point, c ∈ geometric_locus A B l :=
sorry

end geometric_locus_centers_l225_225419


namespace translate_parabola_l225_225954

/--
Translate the function y = -2x^2 + 4x + 1 by moving its graph 
2 units to the left and then 3 units up.
The equation of the resulting graph is y = -2(x + 1)^2 + 6.
-/
theorem translate_parabola :
  ∀ x : ℝ, y = -2*x^2 + 4*x + 1,
  y' = -2*(x + 1)^2 + 6 → true :=
begin
  let f : ℝ → ℝ := λ x, -2*x^2 + 4*x + 1,
  let g : ℝ → ℝ := λ x, f(x + 2) + 3,
  have : ∀ x, g(x - 1) = -2*(x + 1)^2 + 6 := sorry,
  trivial
end

end translate_parabola_l225_225954


namespace race_time_difference_l225_225819

-- Define Malcolm's speed, Joshua's speed, and the distance
def malcolm_speed := 6 -- minutes per mile
def joshua_speed := 7 -- minutes per mile
def race_distance := 15 -- miles

-- Statement of the theorem
theorem race_time_difference :
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 15 :=
by sorry

end race_time_difference_l225_225819


namespace distance_between_trees_l225_225620

def yard_length : ℕ := 350
def num_trees : ℕ := 26
def num_intervals : ℕ := num_trees - 1

theorem distance_between_trees :
  yard_length / num_intervals = 14 := 
sorry

end distance_between_trees_l225_225620


namespace five_op_two_l225_225501

-- Definition of the operation
def op (a b : ℝ) := 3 * a + 4 * b

-- The theorem statement
theorem five_op_two : op 5 2 = 23 := by
  sorry

end five_op_two_l225_225501


namespace sharks_in_pelican_bay_l225_225750

theorem sharks_in_pelican_bay 
  (P : ℕ) 
  (h1 : ∀(s p : ℕ), s = 2 * p) 
  (h2 : 1 / 3 * P + (20 : ℕ) = P) 
  : 2 * 30 = 60 := 
begin
  sorry
end

end sharks_in_pelican_bay_l225_225750


namespace log_comparison_l225_225360

theorem log_comparison (a b c : ℝ)
  (h1 : a = Real.log 3 / Real.log 2)
  (h2 : b = 4 / 3)
  (h3 : c = Real.log 4 / Real.log 3) :
  c < b ∧ b < a := 
begin
  sorry
end

end log_comparison_l225_225360


namespace perimeter_quadrilateral_l225_225108

theorem perimeter_quadrilateral
  (right_angle_AEB : ∀ (A B E : Type) [decidable_eq A] [decidable_eq B] [decidable_eq E] (ABE : triangle A B E), angle A E B = 60)
  (right_angle_BCE : ∀ (B C E : Type) [decidable_eq B] [decidable_eq C] [decidable_eq E] (BCE : triangle B C E), angle B E C = 60)
  (equilateral_CDE : ∀ (C D E : Type) [decidable_eq C] [decidable_eq D] [decidable_eq E] (CDE : triangle C D E), 
      (side C D = side D E) ∧ (side D E = side E C))
  (AE : ℝ = 20 * real.sqrt 3):
  let AB := 30 in
  let BE := 10 * real.sqrt 3 in
  let BC := 15 in
  let CD := 10 * real.sqrt 3 in
  let DA := 15 * real.sqrt 2 + 20 * real.sqrt 3 in
  AB + BC + CD + DA = 45 + 30 * real.sqrt 3 + 15 * real.sqrt 2 := by
  sorry

end perimeter_quadrilateral_l225_225108


namespace probability_of_at_least_6_consecutive_heads_in_9_tosses_l225_225635

theorem probability_of_at_least_6_consecutive_heads_in_9_tosses :
  let outcomes := 2^9 in
  let favorable_outcomes := 4 + 3 + 2 + 1 in
  favorable_outcomes / outcomes = (5 : ℚ) / 256 := sorry

end probability_of_at_least_6_consecutive_heads_in_9_tosses_l225_225635


namespace avg_reading_time_l225_225326

theorem avg_reading_time (emery_book_time serena_book_time emery_article_time serena_article_time : ℕ)
    (h1 : emery_book_time = 20)
    (h2 : emery_article_time = 2)
    (h3 : emery_book_time * 5 = serena_book_time)
    (h4 : emery_article_time * 3 = serena_article_time) :
    (emery_book_time + emery_article_time + serena_book_time + serena_article_time) / 2 = 64 := by
  sorry

end avg_reading_time_l225_225326


namespace cupcakes_per_package_l225_225464

-- Define the conditions
variables (initial_cupcakes : Nat) (eaten_cupcakes : Nat) (packages : Nat)
variable (remaining_cupcakes : Nat)

-- Assume conditions according to the problem
def cupcakes_baked := initial_cupcakes = 39
def cupcakes_eaten := eaten_cupcakes = 21
def number_of_packages := packages = 6
def remaining_cupcakes_def := remaining_cupcakes = initial_cupcakes - eaten_cupcakes

-- Define the statement that needs to be proven: each package contains 3 cupcakes
theorem cupcakes_per_package : cupcakes_baked ∧ cupcakes_eaten ∧ number_of_packages ∧ remaining_cupcakes_def → remaining_cupcakes / packages = 3 :=
by
  intros
  sorry

end cupcakes_per_package_l225_225464


namespace find_a_plus_b_l225_225409

-- Definitions according to the conditions provided
variables {a b x : ℝ}

-- Condition that the inequality has a specific solution set
def inequality_solution (a : ℝ) (b : ℝ) : Prop :=
  ∀ x : ℝ, (1 < x ∧ x < b) ↔ (ax^2 - 3*x + 2 < 0)

-- The proof problem: proving that a + b = 3
theorem find_a_plus_b (a b : ℝ) (h : inequality_solution a b) : a + b = 3 :=
begin
  sorry
end

end find_a_plus_b_l225_225409


namespace correct_operation_l225_225242

-- Define the terms and operations as per the conditions
variable (a : ℝ)

-- Condition A
def op_A := a^2 * a^3 = a^6

-- Condition B
def op_B := a^4 / a^2 = a^2

-- Condition C
def op_C := (a^3)^2 = a^5

-- Condition D
def op_D := 2 * a^2 + 3 * a^2 = 5 * a^4

-- Theorem stating the correct operation
theorem correct_operation : op_B :=
by
  sorry

end correct_operation_l225_225242


namespace man_speed_l225_225625

noncomputable def speed_of_man (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600 in
  (train_speed_ms * crossing_time - train_length) / crossing_time

theorem man_speed :
  speed_of_man 900 53.99568034557235 63 ≈ 0.832 := sorry

end man_speed_l225_225625


namespace angle_equivalence_l225_225443

-- Definitions of points, lines, and other geometric constructs
variables {A B C M T D : Type}
variables [AffineSpace ℝ ℝ] {tr : Triangle ℝ A B C}

-- Definition of the properties and conditions
noncomputable def midpoint (A C : Type) : Type := sorry -- Definition of midpoint to be provided
noncomputable def extension (TM BC: Type) : D := sorry -- Definition of extension intersection to be provided

-- Given conditions in the problem
axiom angle_ABC_right : ∠ABC = 90
axiom midpoint_AC : M = midpoint A C
axiom perpendicular_AT_AC : ⊥ AT AC
axiom intersection_TM_BC_D : TM → BC = D

-- The theorem to prove
theorem angle_equivalence : ∠ABT = ∠CAD :=
sorry

end angle_equivalence_l225_225443


namespace conjugate_in_fourth_quadrant_l225_225361

-- Define the complex number z
def z : ℂ := 1 + 2*I

-- Define the conjugate of z, \overline{z}
def conj_z : ℂ := conj(z)

-- Define a function to identify the quadrant of a point in the complex plane
def quadrant (p : ℂ) : string :=
  if p.re > 0 then
    if p.im > 0 then "First quadrant"
    else "Fourth quadrant"
  else
    if p.im > 0 then "Second quadrant"
    else "Third quadrant"

-- The theorem stating the problem
theorem conjugate_in_fourth_quadrant :
  quadrant conj_z = "Fourth quadrant" :=
sorry

end conjugate_in_fourth_quadrant_l225_225361


namespace correct_statement_l225_225972

theorem correct_statement :
  (∃ (A : Prop), A = (2 * x^3 - 4 * x - 3 ≠ 3)) ∧
  (∃ (B : Prop), B = ((2 + 3) ≠ 6)) ∧
  (∃ (C : Prop), C = (-4 * x^2 * y = -4)) ∧
  (∃ (D : Prop), D = (1 = 1 ∧ 1 = 1 / 8)) →
  (C) :=
by sorry

end correct_statement_l225_225972


namespace evaluate_f_l225_225311

def f (x : ℝ) : ℝ := x^2 + 3 * Real.sqrt x + x

theorem evaluate_f :
  3 * f 3 + f 9 = 135 + 9 * Real.sqrt 3 := by
  sorry

end evaluate_f_l225_225311


namespace seq_divisibility_l225_225128

/-- Definition of the sequence u_n. -/
noncomputable def u : ℕ → ℕ 
| 1     := 1
| 2     := 0
| 3     := 1
| (n+3) := u (n+1) + u n

theorem seq_divisibility :
  ∀ n : ℕ, n ≥ 1 → 
    u (2 * n) - (u (n - 1))^2 % u n = 0 ∧ 
    u (2 * n + 1) - (u (n + 1))^2 % u n = 0 := 
by
  sorry

end seq_divisibility_l225_225128


namespace sum_of_digits_in_repeating_decimal_l225_225573

theorem sum_of_digits_in_repeating_decimal (d : List ℕ) (n₁ n₂ n₃ n₄ : ℕ) :
  d = [2, 5, 8, 7] →
  n₁ = 1 →
  n₂ = 99 →
  n₃ = 199 →
  n₄ = 299 →
  (d.get (n₁ % d.length - 1) + d.get (n₂ % d.length - 1) +
   d.get (n₃ % d.length - 1) + d.get (n₄ % d.length - 1)) = 26 := by
  intros
  sorry

end sum_of_digits_in_repeating_decimal_l225_225573


namespace side_length_120_l225_225653

-- Definitions of the sides in terms of variables a and d.
def sides_are_ap (a d : ℝ) : Prop :=
  (let c := a + 2 * d in a^2 + (a + d)^2 = c^2)

-- Hypothesis: One of the sides is 120.
def one_of_sides_is_120 (a d : ℝ) : Prop :=
  a = 120 ∨ a + d = 120 ∨ a + 2 * d = 120

-- The goal to prove that one of the sides of the right triangle could be 120 given the conditions.
theorem side_length_120 {a d : ℝ} (h : sides_are_ap a d) : one_of_sides_is_120 a d :=
by
  sorry

end side_length_120_l225_225653


namespace hyperbola_imaginary_axis_length_l225_225183

theorem hyperbola_imaginary_axis_length : 
  (λ (x y : ℝ), x^2 / 16 - y^2 / 8 = 1) →
  ∃ b : ℝ, b = 2 * Real.sqrt 2 ∧ 2 * b = 4 * Real.sqrt 2 :=
sorry

end hyperbola_imaginary_axis_length_l225_225183


namespace reforestation_year_total_subsidy_l225_225161

open Real

-- Define the initial conditions
def total_cultivation_sloping_land : ℝ := 91e6
def western_region_fraction : ℝ := 0.7
def reforested_area_2002 : ℝ := 5.15e6
def annual_increase_rate : ℝ := 0.12
def subsidy_grain_per_mu : ℝ := 300
def price_per_jin : ℝ := 0.7
def annual_subsidy_per_mu : ℝ := 20

-- Define the required reforestation area and total subsidies
def required_reforestation_area : ℝ := western_region_fraction * total_cultivation_sloping_land
def total_subsidy_per_mu : ℝ := (subsidy_grain_per_mu * price_per_jin) + annual_subsidy_per_mu

theorem reforestation_year : ∃ n : ℕ, 2002 + n = 2009 ∧ 
  reforested_area_2002 * (1 - (1 + annual_increase_rate)^n) / (-annual_increase_rate) ≥ required_reforestation_area :=
by
  sorry

theorem total_subsidy : 
  total_subsidy_per_mu * reforested_area_2002 * (1 - (1 + annual_increase_rate)^7) / (-annual_increase_rate) ≈ 1.44e9 :=
by
  sorry

end reforestation_year_total_subsidy_l225_225161


namespace burmese_python_eats_alligators_l225_225626

theorem burmese_python_eats_alligators (snake_length : ℝ) (alligator_length : ℝ) (alligator_per_week : ℝ) (total_alligators : ℝ) :
  snake_length = 1.4 → alligator_length = 0.5 → alligator_per_week = 1 → total_alligators = 88 →
  (total_alligators / alligator_per_week) * 7 = 616 := by
  intros
  sorry

end burmese_python_eats_alligators_l225_225626


namespace cos_angle_F1P2_eq_one_third_l225_225784

noncomputable def ellipse (x y : ℝ) (m : ℝ) : Prop :=
x^2 / 2 + y^2 / m = 1

noncomputable def hyperbola (x y : ℝ) : Prop :=
y^2 / 3 - x^2 = 1

theorem cos_angle_F1P2_eq_one_third {m : ℝ} {x y : ℝ} {P : ℝ × ℝ}
(F₁ F₂ : ℝ × ℝ) (h₁ : F₁ = (0, -2)) (h₂ : F₂ = (0, 2))
(intersect_ellipse : ellipse P.1 P.2 m)
(intersect_hyperbola : hyperbola P.1 P.2)
(common_foci : ∀ {F₁ F₂}, F₁ = (0, -2) ∧ F₂ = (0, 2) → ellipse P.1 P.2 m ∧ hyperbola P.1 P.2 → cos_angle_F1P2_eq_one_third) :
  cos_angle_F1P2_eq_one_third (cos_angle_F1P2_eq_one_third F₁ F₂ P)  :=
by
  sorry

end cos_angle_F1P2_eq_one_third_l225_225784


namespace cyclic_quad_solution_l225_225095

variable {α : Type} [LinearOrderedField α]

noncomputable def cyclic_quadrilateral (A B C D O P M N E F : α) : Prop :=
  ∃ (A B C D : α) (O : α) (P : α) (M N : α), 
    cyclic_quad A B C D O ∧ 
    intersect_ac_bd_at_p A C B D P ∧ 
    midpoint_chord P M N ∧ 
    intersect_mn_ab_at_e_cd_at_f M N A B E C D F ∧ 
    PE_eq_PF P E F = true

def cyclic_quad_proof : Prop := 
  ∃ (A B C D O P M N E F : α), cyclic_quadrilateral A B C D O P M N E F
  
theorem cyclic_quad_solution : cyclic_quad_proof := sorry

end cyclic_quad_solution_l225_225095


namespace min_amount_spent_l225_225984

def box_volume (length width height : ℕ) : ℕ := length * width * height

def num_boxes_needed (total_volume box_volume : ℕ) : ℕ :=
  (total_volume + box_volume - 1) / box_volume  -- Division rounding up

def total_cost (num_boxes : ℕ) (cost_per_box : ℚ) : ℚ :=
  num_boxes * cost_per_box

theorem min_amount_spent
  (length width height : ℕ)
  (box_cost : ℚ)
  (total_volume : ℕ) :
  length = 20 →
  width = 20 →
  height = 12 →
  box_cost = 0.40 →
  total_volume = 2160000 →
  total_cost (num_boxes_needed total_volume (box_volume length width height)) box_cost = 180 :=
by
  intros h_length h_width h_height h_cost h_volume
  rw [h_length, h_width, h_height]  -- Utilizing given dimensions
  rw h_cost  -- Utilizing given cost per box
  rw h_volume  -- Utilizing given total volume
  sorry  -- Proof of correctness for the remaining steps

end min_amount_spent_l225_225984


namespace find_m_n_l225_225057

-- Define the set A
def set_A : Set ℝ := {x | |x + 2| < 3}

-- Define the set B in terms of a variable m
def set_B (m : ℝ) : Set ℝ := {x | (x - m) * (x - 2) < 0}

-- State the theorem
theorem find_m_n (m n : ℝ) (hA : set_A = {x | -5 < x ∧ x < 1}) (h_inter : set_A ∩ set_B m = {x | -1 < x ∧ x < n}) : 
  m = -1 ∧ n = 1 :=
by
  -- Proof is omitted
  sorry

end find_m_n_l225_225057


namespace number_of_valid_pairs_l225_225875

-- Define the condition we need
def condition (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 / (a: ℚ) - 1 / (b: ℚ) = 1 / 2023)

-- Define the statement to check the number of integer pairs (a, b) that satisfy the condition
theorem number_of_valid_pairs : 
  (finset.filter (λ ab, condition ab.1 ab.2) (finset.prod (finset.range 20000) (finset.range 20000))).card = 7 :=
sorry

end number_of_valid_pairs_l225_225875


namespace problem_solve_l225_225505

variable {y : Fin 50 → ℝ} (h1 : ∑ i, y i = 2) (h2 : ∑ i, y i / (2 - y i) = 2)

theorem problem_solve :
  (∑ i, y i ^ 2 / (2 - y i)) = 0 :=
by
  sorry

end problem_solve_l225_225505


namespace distinct_flavors_count_l225_225168

theorem distinct_flavors_count (red_candies : ℕ) (green_candies : ℕ)
  (h_red : red_candies = 0 ∨ red_candies = 1 ∨ red_candies = 2 ∨ red_candies = 3 ∨ red_candies = 4 ∨ red_candies = 5 ∨ red_candies = 6)
  (h_green : green_candies = 0 ∨ green_candies = 1 ∨ green_candies = 2 ∨ green_candies = 3 ∨ green_candies = 4 ∨ green_candies = 5) :
  ∃ unique_flavors : Finset (ℚ), unique_flavors.card = 25 :=
by
  sorry

end distinct_flavors_count_l225_225168


namespace sum_even_integers_gt_5_lt_20_l225_225236

theorem sum_even_integers_gt_5_lt_20 : 
  (∑ i in finset.filter (λ x, x % 2 = 0) (finset.Ico 6 20), i) = 84 :=
by
  sorry

end sum_even_integers_gt_5_lt_20_l225_225236


namespace num_perfect_square_factors_of_4410_l225_225558

-- Define the conditions for the exponents in the prime factorization
def valid_exponent (a b c d : ℕ) : Prop :=
  a ∈ {0, 1} ∧ b ∈ {0, 1, 2} ∧ c ∈ {0, 1} ∧ d ∈ {0, 1, 2}

-- Define the condition for a factor to be a perfect square
def is_perfect_square (a b c d : ℕ) : Prop :=
  a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ d % 2 = 0

-- Main theorem statement using the conditions
theorem num_perfect_square_factors_of_4410 : 
  (finset.filter 
    (λ (abcd : ℕ × ℕ × ℕ × ℕ), is_perfect_square abcd.1 abcd.2 abcd.3 abcd.4)
    ({(a, b, c, d) | valid_exponent a b c d}).to_finset
  ).card = 4 := sorry

end num_perfect_square_factors_of_4410_l225_225558


namespace hyperbola_triangle_perimeter_ellipse_product_pf1_pf2_l225_225048

-- Define the first problem related to the hyperbola
theorem hyperbola_triangle_perimeter 
  (F1 F2 : ℝ × ℝ) (A B: ℝ × ℝ) (line_slope : ℝ) 
  (conic_eq : ∀ x y : ℝ, x^2 - y^2 = 1)
  (line_eq : ∀ x y : ℝ, y = line_slope * (x + √2))
  (F1_eqn : F1 = (-√2, 0))
  (F2_eqn : F2 = (√2, 0))
  (A_eqn : conic_eq A.1 A.2 ∧ line_eq A.1 A.2)
  (B_eqn : conic_eq B.1 B.2 ∧ line_eq B.1 B.2) :
  dist A B + dist A F2 + dist B F2 = 12 := 
sorry

-- Define the second problem related to the ellipse
theorem ellipse_product_pf1_pf2 
  (F1 F2 P : ℝ × ℝ) 
  (conic_eq : ∀ x y : ℝ, x^2 + y^2 / 4 = 1)
  (F1_eqn : F1 = (-√3, 0))
  (F2_eqn : F2 = (√3, 0))
  (P_on_conic : conic_eq P.1 P.2)
  (dist_to_foci_sum_eq : ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = 4) :
  ∃ x ∈ set.Icc (2 - √3) (2 + √3), 
    max (x * (4 - x)) = 4 ∧ min (x * (4 - x)) = 1 :=
sorry

end hyperbola_triangle_perimeter_ellipse_product_pf1_pf2_l225_225048


namespace lola_wins_probability_l225_225821

theorem lola_wins_probability (p : ℚ) (h : p = 3 / 7) : 1 - p = 4 / 7 :=
by
  rw [h]
  norm_num
  sorry

end lola_wins_probability_l225_225821


namespace team_t_speed_l225_225576

theorem team_t_speed (v t : ℝ) (h1 : 300 = v * t) (h2 : 300 = (v + 5) * (t - 3)) : v = 20 :=
by 
  sorry

end team_t_speed_l225_225576


namespace area_six_layers_l225_225017

theorem area_six_layers
  (A : ℕ → ℕ)
  (h1 : A 1 + A 2 + A 3 = 280)
  (h2 : A 2 = 54)
  (h3 : A 3 = 28)
  (h4 : A 4 = 14)
  (h5 : A 1 + 2 * A 2 + 3 * A 3 + 4 * A 4 + 6 * A 6 = 500)
  : A 6 = 9 := 
sorry

end area_six_layers_l225_225017


namespace caps_relationship_l225_225991

-- Define the number of caps made in the first three weeks
def caps_week1 : ℕ := 320
def caps_week2 : ℕ := 400
def caps_week3 : ℕ := 300

-- Define the total target number of caps
def total_caps : ℕ := 1360

-- Define the sum of caps made in the first three weeks
def caps_first_three_weeks : ℕ := caps_week1 + caps_week2 + caps_week3

-- Define the number of caps made in the fourth week
def caps_week4 : ℕ := total_caps - caps_first_three_weeks

-- The theorem statement we need to prove
theorem caps_relationship : caps_week4 + caps_first_three_weeks = total_caps :=
by
  unfold caps_week4 caps_first_three_weeks total_caps caps_week1 caps_week2 caps_week3
  simp
  exact Nat.add_sub_cancel' (caps_week1 + caps_week2 + caps_week3) total_caps

end caps_relationship_l225_225991


namespace unique_solution_xp_eq_1_l225_225530

theorem unique_solution_xp_eq_1 (x p q : ℕ) (h1 : x ≥ 2) (h2 : p ≥ 2) (h3 : q ≥ 2):
  ((x + 1)^p - x^q = 1) ↔ (x = 2 ∧ p = 2 ∧ q = 3) :=
by 
  sorry

end unique_solution_xp_eq_1_l225_225530


namespace therapist_charge_l225_225994

theorem therapist_charge (F A : ℝ) 
    (h1 : F + 4 * A = 400)
    (h2 : F + 2 * A = 252) : F - A = 30 := 
by 
    sorry

end therapist_charge_l225_225994


namespace cone_volume_correct_lateral_surface_area_correct_l225_225199

def slant_height (h : ℝ) : Prop := h = 17
def height (h : ℝ) : Prop := h = 15 

noncomputable def cone_radius (l : ℝ) (h : ℝ) : ℝ :=
  Real.sqrt (l^2 - h^2)

noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := 
  (1 / 3) * Real.pi * r^2 * h

noncomputable def lateral_surface_area (r : ℝ) (l : ℝ) : ℝ :=
  Real.pi * r * l

theorem cone_volume_correct :
  ∀ (l h r volume : ℝ), slant_height l → height h → 
  r = cone_radius l h →
  volume = cone_volume r h →
  volume = 320 * Real.pi := by
  intros l h r volume hl hh hr hv
  sorry

theorem lateral_surface_area_correct :
  ∀ (l r lat_area : ℝ), slant_height l → height h → 
  r = cone_radius l h →
  lat_area = lateral_surface_area r l → 
  lat_area = 136 * Real.pi := by
  intros l r lat_area hl hh hr ha
  sorry

end cone_volume_correct_lateral_surface_area_correct_l225_225199


namespace angle_ADC_is_70_l225_225880

-- Define the trapezoid and its properties
variables {A B C D : Type} [Points A B C D]
variables (AB CD: Line)
variables (trapezoid_ABCD: Trapezoid ABCD AB CD)
variables (angle_BAC angle_ABC: Angle)
variables (condition_ABC_parallel_CD : AB.parallel CD)

-- Angles and length conditions
variables (angle_BAC_eq_25 : angle_BAC = 25)
variables (angle_ABC_eq_125 : angle_ABC = 125)
variables (length_AB_AD_eq_CD : length AB + length AD = length CD)

-- Target Angle
variable (angle_ADC: Angle)

-- The theorem statement
theorem angle_ADC_is_70 : angle_ADC = 70 :=
sorry

end angle_ADC_is_70_l225_225880


namespace new_person_weight_l225_225173

theorem new_person_weight
    (avg_weight_20 : ℕ → ℕ)
    (total_weight_20 : ℕ)
    (avg_weight_21 : ℕ)
    (count_20 : ℕ)
    (count_21 : ℕ) :
    avg_weight_20 count_20 = 58 →
    total_weight_20 = count_20 * avg_weight_20 count_20 →
    avg_weight_21 = 53 →
    count_21 = count_20 + 1 →
    ∃ (W : ℕ), total_weight_20 + W = count_21 * avg_weight_21 ∧ W = 47 := 
by 
  sorry

end new_person_weight_l225_225173


namespace parabolic_proof_l225_225396

noncomputable def parabolic_intersection (k : ℝ) (p : ℝ) : Prop :=
  k > 0 ∧ 
  ∃ (F E B : ℝ × ℝ),
  let l := λ (x: ℝ), k * x + 1 in 
  let C := λ (x y: ℝ), x^2 = 2 * p * y in 
  let F := (0, p / 2) in 
  let E := (-2 / k, -1) in 
  let B := (2 / k, 3) in 
  l (0) = p / 2 ∧
  E = (-2 / k, -1) ∧
  C (2 / k) (3) ∧
  (2 * 3) = (4 / k^2) ∧ 
  ∀ (x y: ℝ), (x + (2 / k)) / 2 = 0 ∧ (y + 3) / 2 = 1

theorem parabolic_proof : 
  ∀ (k p : ℝ), parabolic_intersection k p → p = 2 ∧ k = (real.sqrt 3) / 3 :=
by sorry

end parabolic_proof_l225_225396


namespace ratio_of_15th_term_l225_225499

theorem ratio_of_15th_term (a d b e : ℤ) :
  (∀ n : ℕ, (n * (2 * a + (n - 1) * d)) / (n * (2 * b + (n - 1) * e)) = (7 * n^2 + 1) / (4 * n^2 + 27)) →
  (a + 14 * d) / (b + 14 * e) = 7 / 4 :=
by sorry

end ratio_of_15th_term_l225_225499


namespace uralan_matches_l225_225454

theorem uralan_matches (teams : ℕ) (matches_per_team : ℕ) (uralan_play : teams > 1) : matches_per_team = 2 → states : ℕ :=
  have num_matches := (teams - 1) * matches_per_team
  have uralan_matches : this.num_matches = 30 := sorry

end uralan_matches_l225_225454


namespace triangle_inequality_product_l225_225155

theorem triangle_inequality_product (x y z : ℝ) (h1 : x + y > z) (h2 : x + z > y) (h3 : y + z > x) :
  (x + y + z) * (x + y - z) * (x + z - y) * (z + y - x) ≤ 4 * x^2 * y^2 := 
by
  sorry

end triangle_inequality_product_l225_225155


namespace like_terms_group_C_l225_225971

def likeTerms (a b : List (String × Float)) : Bool :=
  a.map Prod.fst = b.map Prod.fst &&
  a.foldr (fun (_, x) y => x + y) 0 = b.foldr (fun (_, x) y => x + y) 0

noncomputable def terms_A := [("x", 2.0), ("y", 1.0)]
noncomputable def terms_B := [("x", 1.0), ("y", 2.0)]
noncomputable def terms_C1 := [("m", 1.0), ("n", 1.0)]
noncomputable def terms_C2 := [("n", 1.0), ("m", 1.0)]
noncomputable def terms_D1 := [("a", 1.0), ("b", 1.0)]
noncomputable def terms_D2 := [("a", 1.0), ("b", 1.0), ("c", 1.0)]

theorem like_terms_group_C :
  (likeTerms terms_C1 terms_C2) ∧
  ¬ (likeTerms terms_A terms_B) ∧
  ¬ (likeTerms terms_B terms_D1) ∧
  ¬ (likeTerms terms_D1 terms_D2) :=
by
  sorry

end like_terms_group_C_l225_225971


namespace speed_down_correct_l225_225644

noncomputable def speed_down (average_speed speed_up : ℝ) : ℝ :=
  (2 * speed_up * average_speed) / (2 * average_speed - speed_up)

theorem speed_down_correct :
  speed_down 20.869565217391305 16 ≈ 30.08 :=
by
  let v := speed_down 20.869565217391305 16
  have h : v = 30.08 :=
    calc
      v = (2 * 16 * 20.869565217391305) / (16 + v) : by sorry -- calculate v by solving the equation
      ... = 30.08 : by sorry -- Final calculation step
  exact h

end speed_down_correct_l225_225644


namespace max_neg_coeff_no_real_roots_l225_225261

open Polynomial

noncomputable def max_neg_coeff (n : ℕ) : ℕ :=
  let p : Polynomial ℝ := Polynomial.sum (Finset.range (n + 1)) (λ k, if k % 2 = 0 then Polynomial.X ^ k else - Polynomial.X ^ k)
  in sorry

theorem max_neg_coeff_no_real_roots (h : ∀ (p : Polynomial ℝ), p.eval (0 : ℝ) * p.eval (1 : ℝ) > 0) : max_neg_coeff 2010 = 1005 :=
  sorry

end max_neg_coeff_no_real_roots_l225_225261


namespace find_a_value_l225_225086

noncomputable def solve_for_a (y : ℝ) (a : ℝ) : Prop :=
  0 < y ∧ (a * y) / 20 + (3 * y) / 10 = 0.6499999999999999 * y 

theorem find_a_value (y : ℝ) (a : ℝ) (h : solve_for_a y a) : a = 7 := 
by 
  sorry

end find_a_value_l225_225086


namespace equidistant_points_l225_225487

variables (P : ℝ × ℝ × ℝ)
variables (A B C D : ℝ × ℝ × ℝ)

def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem equidistant_points :
  let A := (10, 0, 0)
  let B := (0, -6, 0)
  let C := (0, 0, 8)
  let D := (0, 0, 0)
  P = (5, -3, 4)
  P = (5, -3, 4) → (dist P A = dist P B) ∧ (dist P A = dist P C) ∧ (dist P A = dist P D) :=
begin
  intros,
  sorry
end

end equidistant_points_l225_225487


namespace find_y_z_l225_225383

theorem find_y_z 
  (y z : ℝ) 
  (h_mean : (8 + 15 + 22 + 5 + y + z) / 6 = 12) 
  (h_diff : y - z = 6) : 
  y = 14 ∧ z = 8 := 
by
  sorry

end find_y_z_l225_225383


namespace probability_William_wins_l225_225973

theorem probability_William_wins : 
  let p := 1/3 in
  let r := (2/3) ^ 2 in
  p / (1 - r) = 3/5 :=
by
  let p := 1/3
  let r := (2/3) ^ 2
  have h1 : p = 1 / 3 := rfl
  have h2 : r = (2 / 3) * (2 / 3) := rfl
  rw [h1, h2]
  calc
    p / (1 - r) = (1 / 3) / (1 - (2 / 3) * (2 / 3)) : by rfl
             ... = (1 / 3) / (1 - 4 / 9) : by rfl
             ... = (1 / 3) / (5 / 9) : by norm_num
             ... = (1 / 3) * (9 / 5) : by rw [div_eq_mul_inv]
             ... = 3 / 5 : by norm_num

end probability_William_wins_l225_225973


namespace eccentricity_is_sqrt2_l225_225873

noncomputable def hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (P F₁ F₂ : ℝ × ℝ) (on_hyperbola : P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1)
  (P_sum : (real.dist P F₁ + real.dist P F₂) = 4 * a) 
  (smallest_angle_sine : real.sin (real.angle P F₁ F₂) = 1/3) : 
  real := 
  ((real.dist P F₁)^2 - 2 * (real.dist P F₁) * (real.dist P F₂) * real.cos (real.angle P F₁ F₂) + (real.dist P F₂)^2 = 1) 

theorem eccentricity_is_sqrt2 
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (P F₁ F₂ : ℝ × ℝ)
  (on_hyperbola : P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1)
  (P_sum : real.dist P F₁ + real.dist P F₂ = 4 * a) 
  (smallest_angle_sine : real.sin (real.angle P F₁ F₂) = 1/3) :
  let e := sqrt2 in
  e = sqrt 2 := 
  sorry

end eccentricity_is_sqrt2_l225_225873


namespace primes_not_exist_d_10_primes_not_exist_d_11_l225_225609

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def exists_three_distinct_primes_d_10 : Prop :=
  ¬∃ (p q r : ℕ), 
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    is_prime p ∧ is_prime q ∧ is_prime r ∧
    (q * r ∣ (p ^ 2 + 10)) ∧ 
    (r * p ∣ (q ^ 2 + 10)) ∧ 
    (p * q ∣ (r ^ 2 + 10))

def exists_three_distinct_primes_d_11 : Prop :=
  ¬∃ (p q r : ℕ), 
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    is_prime p ∧ is_prime q ∧ is_prime r ∧
    (q * r ∣ (p ^ 2 + 11)) ∧ 
    (r * p ∣ (q ^ 2 + 11)) ∧ 
    (p * q ∣ (r ^ 2 + 11))

theorem primes_not_exist_d_10 : exists_three_distinct_primes_d_10 := by 
  sorry

theorem primes_not_exist_d_11 : exists_three_distinct_primes_d_11 := by 
  sorry

end primes_not_exist_d_10_primes_not_exist_d_11_l225_225609


namespace min_distance_curve_and_line_l225_225274

noncomputable def curve_C (α : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos α, Real.sin α)

def line_l (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 = 3

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_distance_curve_and_line :
  ∃ α β : ℝ, α ∈ Icc 0 (2 * π) ∧ line_l (curve_C β) ∧ distance (curve_C α) (curve_C β) = Real.sqrt 2 / 2 :=
sorry

end min_distance_curve_and_line_l225_225274


namespace sum_middle_three_cards_equals_8_l225_225952

-- Defining the sets of cards
def orangeCards := {2, 3, 4, 5, 6, 7}  -- Orange cards numbered 2 through 7
def greenCards := {2, 3, 4, 5, 6}      -- Green cards numbered 2 through 6

-- Condition: Stack alternates in color and each number on an orange card divides evenly into the number on each neighboring green card.
def isValidStack (stack : List ℕ) : Prop :=
  stack.length = 9 ∧
  ∀ i, i < 8 → 
    (i % 2 = 0 → stack.get! i ∈ orangeCards ∧ stack.get! (i + 1) ∈ greenCards ∧ stack.get! (i + 1) % stack.get! i = 0) ∧
    (i % 2 = 1 → stack.get! i ∈ greenCards ∧ stack.get! (i + 1) ∈ orangeCards ∧ stack.get! (i + 1) % stack.get! i = 0)

-- Define sum of the numbers on the middle three cards in the stack
def middleThreeSum (stack : List ℕ) : ℕ := stack.get! 3 + stack.get! 4 + stack.get! 5

-- The theorem statement
theorem sum_middle_three_cards_equals_8 (stack : List ℕ) (h : isValidStack stack) : 
  middleThreeSum stack = 8 := by
  -- We skip the proof here
  sorry

end sum_middle_three_cards_equals_8_l225_225952


namespace fence_length_l225_225664

theorem fence_length (n : ℕ) (d_3_6 : ℝ) (total_posts : ℕ) (spaces_between_posts : ℕ)
  (condition1 : total_posts = 12)
  (condition2 : d_3_6 = 3.3)
  (condition3 : total_posts = n + 1)
  (condition4 : n = spaces_between_posts)
  (condition5 : d_3_6 = 3 * 1.1) :
  spaces_between_posts * 1.1 = 12.1 :=
by
  sorry

end fence_length_l225_225664


namespace depth_of_pit_l225_225096

noncomputable def field_length : ℝ := 20
noncomputable def field_width : ℝ := 10
noncomputable def pit_length : ℝ := 8
noncomputable def pit_width : ℝ := 5
noncomputable def height_rise : ℝ := 0.5

theorem depth_of_pit :
  let field_area := field_length * field_width,
      pit_area := pit_length * pit_width,
      remaining_area := field_area - pit_area,
      volume_of_earth := remaining_area * height_rise in
      volume_of_earth / pit_area = 2 :=
  sorry

end depth_of_pit_l225_225096


namespace certain_number_is_4_l225_225430

theorem certain_number_is_4 (x : ℝ) : (1/5)^x * (1/4)^2 = 1/(10^4) → x = 4 := by
  sorry -- Proof not required

end certain_number_is_4_l225_225430


namespace perfect_square_trinomial_m_l225_225805

theorem perfect_square_trinomial_m (m : ℤ) :
  (∃ a : ℤ, x^2 + (m - 2) * x + 9 = (x + a)^2) ∨
  (∃ a : ℤ, x^2 + (m - 2) * x + 9 = (x - a)^2) ↔ (m = 8 ∨ m = -4) :=
sorry

end perfect_square_trinomial_m_l225_225805


namespace perimeter_is_six_cm_l225_225040

noncomputable def radius_of_sector (area : ℝ) (θ : ℝ) : ℝ :=
  real.sqrt (2 * area / θ)

def arc_length (r : ℝ) (θ : ℝ) : ℝ :=
  r * θ

def perimeter_of_sector (r : ℝ) (θ : ℝ) : ℝ :=
  arc_length r θ + 2 * r

theorem perimeter_is_six_cm (h_area : ℝ) (h_theta : ℝ) (h_area_eq : h_area = 2) (h_theta_eq : h_theta = 4) :
  perimeter_of_sector (radius_of_sector h_area h_theta) h_theta = 6 :=
by {
  sorry
}

end perimeter_is_six_cm_l225_225040


namespace extreme_maximum_f_range_of_a_l225_225619

noncomputable def f (x : ℝ) : ℝ := log x / x

-- Assertion: Extreme maximum point of f(x) is at x = e
theorem extreme_maximum_f :
  ∃ (x : ℝ), x = Real.exp 1 ∧ (∀ y, f y ≤ f x) :=
sorry

noncomputable def g (x a : ℝ) : ℝ := log x - a * x^2 + (log 2) / 2

-- Assertion: If the maximum value of g(x) is greater than (a / 2) - 1, then 0 < a < 1
theorem range_of_a (a : ℝ) (h : ∀ x, g x a ≤ g (1 / (Real.sqrt (2 * a))) a)
  (H_max : g (1 / (Real.sqrt (2 * a))) a > (a / 2) - 1) : 0 < a ∧ a < 1 :=
sorry

end extreme_maximum_f_range_of_a_l225_225619


namespace two_values_of_a_for_line_passing_through_parabola_vertex_l225_225349

theorem two_values_of_a_for_line_passing_through_parabola_vertex :
  let vertex_y := a^2 in
  let line_y := x + a in
  ∃ exactly_two_values_of_a : ℝ, (vertex_y = line_y) :=
sorry

end two_values_of_a_for_line_passing_through_parabola_vertex_l225_225349


namespace cube_root_fraction_simplification_l225_225330

theorem cube_root_fraction_simplification :
  (∛(6 / 12.75) = (2 / ∛17)) := by
  sorry

end cube_root_fraction_simplification_l225_225330


namespace max_x_plus_inv_x_l225_225567

theorem max_x_plus_inv_x 
  (x : ℝ) 
  (y : Fin 100 → ℝ) 
  (hx_pos : 0 < x) 
  (hy_pos : ∀ i, 0 < y i) 
  (h_sum : x + ∑ i, y i = 102) 
  (h_inv_sum : x⁻¹ + ∑ i, (y i)⁻¹ = 102) : 
  x + x⁻¹ ≤ 404 / 102 := 
by
  sorry

end max_x_plus_inv_x_l225_225567


namespace area_of_triangle_PQR_is_correct_l225_225660

noncomputable def point := (ℝ × ℝ × ℝ)

def V : point := (0, 0, 0)
def A : point := (15, 15, 0)
def B : point := (-15, 15, 0)
def C : point := (-15, -15, 0)
def D : point := (15, -15, 0)

def VP_dist := 10
def VQ_dist := 20
def VR_dist := 15

-- Calculate coordinates of P, Q, R based on the given VP_dist, VQ_dist, and VR_dist
def P : point :=
  let a := 15 * VP_dist / 40
  (15 - a, 15 - a, VP_dist)

def Q : point :=
  let b := 15 * VQ_dist / 40
  (-15 + b, 15 - b, VQ_dist)

def R : point :=
  let c := 15 * VR_dist / 40
  (-15 + c, -15 + c, VR_dist)

noncomputable def area_of_triangle (P Q R : point) : ℝ :=
  let (x1, y1, z1) := P
  let (x2, y2, z2) := Q
  let (x3, y3, z3) := R
  let u := (x2 - x1, y2 - y1, z2 - z1)
  let v := (x3 - x1, y3 - y1, z3 - z1)
  let cross_prod := (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)
  (0.5 * (cross_prod.1^2 + cross_prod.2^2 + cross_prod.3^2).sqrt)

theorem area_of_triangle_PQR_is_correct :
  area_of_triangle P Q R = -- the computed area value
  sorry

end area_of_triangle_PQR_is_correct_l225_225660


namespace probability_point_in_triangle_l225_225649

theorem probability_point_in_triangle (x y : ℝ) 
  (hx : 0 ≤ x) (hx4 : x ≤ 4) 
  (hy : 0 ≤ y) (hy5 : y ≤ 5) : 
  (set.Icc (0 : ℝ) 4) ×ˢ (set.Icc (0 : ℝ) 5) → (2 / 5 : ℝ) := 
sorry

end probability_point_in_triangle_l225_225649


namespace pipe_B_to_A_ratio_l225_225291

noncomputable def pipe_rate_A : ℝ := 1 / 70
noncomputable def pipe_rate_B (B : ℝ) : Prop := 
  let C := 2 * B in
  pipe_rate_A + B + C = 1 / 10

theorem pipe_B_to_A_ratio (B : ℝ) (hB : pipe_rate_B B) : B / pipe_rate_A = 2 :=
by
  let A := pipe_rate_A
  have hA : A = 1 / 70 := rfl
  calc
    B / A = (1 / 35) / (1 / 70) := by
      rw [hB]
      have : A + B + 2 * B = 1 / 10 := sorry
      sorry  -- provide the proof steps if necessary
    ... = 2 := sorry

end pipe_B_to_A_ratio_l225_225291


namespace number_of_incorrect_statements_l225_225669

theorem number_of_incorrect_statements
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = Real.sin x + Real.cos x)
  (h2 : f' (Real.pi / 4) = 0)
  (h3 : ∀ a b m, ¬((a < b) ∧ (am^2 < bm^2)))
  (h4 : ∀ p q, (p ∨ q) ⇔ (p ∧ q))
  (h5 : ¬ (∀ x : ℝ, x^2 - 3 * x - 2 ≥ 0))
  (h6 : ∃ x0 : ℝ, x0^2 - 3 * x0 - 2 < 0) :
  ∃ incorrect_statements : ℕ, incorrect_statements = 2 :=
by
  sorry

end number_of_incorrect_statements_l225_225669


namespace number_of_yellow_marbles_l225_225474

theorem number_of_yellow_marbles 
  (total_marbles : ℕ) 
  (red_marbles : ℕ) 
  (blue_marbles : ℕ) 
  (yellow_marbles : ℕ)
  (h1 : total_marbles = 85) 
  (h2 : red_marbles = 14) 
  (h3 : blue_marbles = 3 * red_marbles) 
  (h4 : yellow_marbles = total_marbles - (red_marbles + blue_marbles)) :
  yellow_marbles = 29 :=
  sorry

end number_of_yellow_marbles_l225_225474


namespace compound_interest_amount_l225_225198

theorem compound_interest_amount:
  let SI := (5250 * 4 * 2) / 100
  let CI := 2 * SI
  let P := 420 / 0.21 
  CI = P * ((1 + 0.1) ^ 2 - 1) →
  SI = 210 →
  CI = 420 →
  P = 2000 :=
by
  sorry

end compound_interest_amount_l225_225198


namespace half_angle_third_quadrant_l225_225775

theorem half_angle_third_quadrant (θ : ℝ) (k : ℤ) (h1 : θ ∈ set.Ioo (2 * k * Real.pi + Real.pi / 2) (2 * k * Real.pi + Real.pi)) (h2 : |Real.sin (θ / 2)| = -Real.sin (θ / 2)) : 
  θ / 2 ∈ set.Ioo (k * Real.pi + Real.pi / 2) (k * Real.pi + Real.pi) :=
sorry

end half_angle_third_quadrant_l225_225775


namespace football_prob_l225_225652

open Classical

noncomputable def team : Type := Fin 10
noncomputable def football_team_members : Finset team := {0, 1}
noncomputable def basketball_team_members : Finset team := ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} \ football_team_members)

def num_football_members (selection : Finset team) : ℕ := selection.filter (λ x => x ∈ football_team_members).card

theorem football_prob (selection : Finset team) (h_condition : num_football_members selection = 2 ∨ num_football_members selection = 1)
                      (h_size : selection.card = 2 / ∃ x ∈ football_team_members, x ∈ selection) :
  (num_football_members selection = 2) → (probability of the other is from football team in given condition == 1 / 9) :=
by
  sorry

end football_prob_l225_225652


namespace percentage_busy_l225_225895

variable (total_capacity available_space : ℕ)

def used_space (total_capacity available_space : ℕ) : ℕ :=
  total_capacity - available_space

def percentage_used (total_capacity used_space : ℕ) : ℚ :=
  (used_space.toRat / total_capacity.toRat) * 100

theorem percentage_busy (h1 : total_capacity = 16) (h2 : available_space = 8) :
  percentage_used total_capacity (used_space total_capacity available_space) = 50 := by
  sorry

end percentage_busy_l225_225895


namespace part1_part2_l225_225789

variables {x a : ℝ}

def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x + a^2)

theorem part1 (h : a = real.sqrt 2) : 
  ∀ x, f x a ≥ 6 ↔ x ≤ -7/2 ∨ x ≥ 5/2 :=
sorry

theorem part2 (h : ∃ x0 : ℝ, f x0 a < 4 * a) : 
  2 - real.sqrt 3 < a ∧ a < 2 + real.sqrt 3 :=
sorry

end part1_part2_l225_225789


namespace larger_expression_l225_225969

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

def first_expr : ℝ :=
  ∏ i in (finset.range 1010).map (λ n, 2*n + 4), log3 i
  
def second_expr : ℝ :=
  2 * (∏ i in (finset.range 1010).map (λ n, 2*n + 3), log3 i)

theorem larger_expression :
  first_expr > second_expr :=
by
  sorry

end larger_expression_l225_225969


namespace distance_C_distance_BC_l225_225133

variable (A B C D : ℕ)

theorem distance_C
  (hA : A = 350)
  (hAB : A + B = 600)
  (hABCD : A + B + C + D = 1500)
  (hD : D = 275)
  : C = 625 :=
by
  sorry

theorem distance_BC
  (A B C D : ℕ)
  (hA : A = 350)
  (hAB : A + B = 600)
  (hABCD : A + B + C + D = 1500)
  (hD : D = 275)
  : B + C = 875 :=
by
  sorry

end distance_C_distance_BC_l225_225133


namespace Jim_paycheck_correct_l225_225468

noncomputable def Jim_paycheck_after_deductions (gross_pay : ℝ) (retirement_percentage : ℝ) (tax_deduction : ℝ) : ℝ :=
  gross_pay - (gross_pay * retirement_percentage) - tax_deduction

theorem Jim_paycheck_correct :
  Jim_paycheck_after_deductions 1120 0.25 100 = 740 :=
by sorry

end Jim_paycheck_correct_l225_225468


namespace triangle_ratio_l225_225848

theorem triangle_ratio
  (A B C D E : Type)
  [tri : IsTriangle A B C]
  (h1 : Bisects B D ∠(A,B,C))
  (h2 : ⟂ (B,E) (A,C)) :
  (AD CD AB BC : ℝ)
  (h3 : SegmentRatio AD CD = SegmentRatio AB BC) 
  (h4 : EC = AC - AD) :
  SegmentRatio AD EC = SegmentRatio AB BC := 
sorry

end triangle_ratio_l225_225848


namespace count_subsets_with_four_adjacent_chairs_l225_225699

def count_at_least_four_adjacent_subsets (n : ℕ) (k : ℕ) (total : ℕ) : Prop :=
  ∃ (S : finset (fin n)), S.card = k ∧
  ∀ (i : finset.range k), S.filter (λ j, (j + i) % n ∈ S) = k

theorem count_subsets_with_four_adjacent_chairs :
  count_at_least_four_adjacent_subsets 8 4 288 := 
sorry

end count_subsets_with_four_adjacent_chairs_l225_225699


namespace midpoint_PQR_RS_l225_225446

variable (PQRS : Type) [ConvexQuadrilateral PQRS]
variable (P Q R S A B : Point)
variable (Area : Triangle -> ℝ)
variable (PR : Line)
variable (RS : Line)
variable (ratio_k : ℝ)
variable (line_through_Q_cut_PR_A_RS_B : Line)

-- Conditions
axiom areas_ratio : 
  (Area (triangle P Q S) = 3 * ratio_k ∧ 
   Area (triangle Q R S) = 4 * ratio_k ∧ 
   Area (triangle P Q R) = ratio_k)

axiom proportionality :
  (PA : PR.cut A) ∧ 
  (RB : RS.cut B) ∧ 
  (PA_length : length PA / length PR = RB_length : length RB / length RS)

-- Questions to prove
theorem midpoint_PQR_RS :
  is_midpoint A PR ∧ 
  is_midpoint B RS :=
sorry

end midpoint_PQR_RS_l225_225446


namespace geometric_sequence_fifth_term_is_32_l225_225564

-- Defining the geometric sequence conditions
variables (a r : ℝ)

def third_term := a * r^2 = 18
def fourth_term := a * r^3 = 24
def fifth_term := a * r^4

theorem geometric_sequence_fifth_term_is_32 (h1 : third_term a r) (h2 : fourth_term a r) : 
  fifth_term a r = 32 := 
by
  sorry

end geometric_sequence_fifth_term_is_32_l225_225564


namespace cube_negative_iff_l225_225513

theorem cube_negative_iff (x : ℝ) : x < 0 ↔ x^3 < 0 :=
sorry

end cube_negative_iff_l225_225513


namespace prove_correct_option_l225_225243

-- Definitions corresponding to the conditions of the problem.
def certain_event (E : Prop) : Prop := E = True
def probability (E : Prop) (p : ℝ) : Prop := 
  if E then p = 1 else p = 0
def mutually_exclusive (E1 E2 : Prop) : Prop :=
  ¬(E1 ∧ E2)
def complementary (E1 E2 : Prop) : Prop :=
  mutually_exclusive E1 E2 ∧ (E1 ∨ E2) = True
def classical_probability_model (E : Prop) : Prop := True -- Generic true due to observational nature

-- The goal is to prove that certain events, complementary events, and classical probability models 
-- correspond to true propositions ①④⑤.
theorem prove_correct_option : 
  (∀ E, certain_event E → probability E 1) ∧
  (∀ E, ¬ probability E 1.1) ∧
  (∀ E1 E2, complementary E1 E2 → mutually_exclusive E1 E2) ∧
  (classical_probability_model (λ E, E)) :=
by
  sorry

end prove_correct_option_l225_225243


namespace minimum_area_sum_l225_225874

variables {A B C M X Y : Type} [linear_ordered_field Type]

-- Definitions based on conditions
def triangle (A B C : Type) := true
def is_midpoint (M : Type) (B C : Type) := true
def projection (M : Type) (A B : Type) := true
constant BC AB AC : ℚ

-- Given conditions
axiom condition1 : triangle A B C
axiom condition2 : AB = AC = (25/14 : ℚ) * BC
axiom condition3 : is_midpoint M B C
axiom condition4 : projection M A B
axiom condition5 : projection M A C
axiom condition6 : ∀ (area_ABC area_AXMY : ℕ), area_ABC > 0 ∧ area_AXMY > 0

-- Lean statement for the proof
theorem minimum_area_sum : 
  ∃ (area_ABC area_AXMY : ℕ), 
  area_ABC + area_AXMY = 1201 :=
sorry

end minimum_area_sum_l225_225874


namespace minimum_platforms_needed_l225_225618

theorem minimum_platforms_needed (num_slabs1 : ℕ) (weight1 : ℕ) (num_slabs2 : ℕ) (weight2 : ℕ) (platform_capacity : ℕ) :
  num_slabs1 = 120 ∧ weight1 = 7 ∧ num_slabs2 = 80 ∧ weight2 = 9 ∧ platform_capacity = 40 →
  ceil ((num_slabs1 * weight1 + num_slabs2 * weight2) / platform_capacity) = 40 :=
by {
  intros h,
  sorry
}

end minimum_platforms_needed_l225_225618


namespace Clea_escalator_time_l225_225682

variable {s : ℝ} -- speed of the escalator at its normal operating speed
variable {c : ℝ} -- speed of Clea walking down the escalator
variable {d : ℝ} -- distance of the escalator

theorem Clea_escalator_time :
  (30 * (c + s) = 72 * c) →
  (s = (7 * c) / 5) →
  (t = (72 * c) / ((3 / 2) * s)) →
  t = 240 / 7 :=
by
  sorry

end Clea_escalator_time_l225_225682


namespace mod_arithmetic_l225_225331

theorem mod_arithmetic :
  let inv_7 := 8
  let inv_13 := 39
  (3 * inv_7 + 9 * inv_13) % 56 = 39 :=
by
  let inv_7 := 8
  let inv_13 := 39
  calc (3 * inv_7 + 9 * inv_13) % 56
    = (3 * 8 + 9 * 39) % 56 : by simp
    = 375 % 56 : by norm_num
    = 39 : by norm_num

end mod_arithmetic_l225_225331


namespace value_of_expression_l225_225967

theorem value_of_expression (a : ℝ) (h : a = 3) : (3 * a^(-2) - a^(-2) / 3) / a^2 = 8 / 243 :=
by
  sorry

end value_of_expression_l225_225967


namespace probability_A_beats_B_l225_225523

-- Definitions of conditions for the problem
def nineTeamsTournament (numTeams numMatches : ℕ) : Prop :=
  numTeams = 9 ∧ numMatches = (9 * 8) / 2

def noTiesAllowed : Prop := ∀ (A B : Type), A ≠ B → (A = 1 ∨ B = 1)

def equalWinProbability (A B : Type) : Prop := 
  ∀ (m : Type), (m = 1 ∧ m ≠ A ∧ m ≠ B) → (0.5 = 0.5)

def winnerScoresPoint (A B : Type) : Prop :=
  ∀ (m : Type), m = A ∨ m = B → (A + 1 = 1 ∨ B + 1 = 1)

def totalPointsDetermineRanking : Prop :=
  ∀ (teams : List (Type × ℕ)), 
    teams = List.sort (λ (t1 t2 : Type × ℕ), t1.snd ≥ t2.snd)

def firstMatchResult (A B : Type) : Prop :=  A = 1 ∧ B = 0

-- Problem statement in Lean 4
theorem probability_A_beats_B : 
  (nineTeamsTournament 9 36) → 
  noTiesAllowed → 
  equalWinProbability nat nat → 
  winnerScoresPoint nat nat → 
  totalPointsDetermineRanking → 
  firstMatchResult nat nat → 
  (⟦ 9714/8192 ⟧) := 
sorry

end probability_A_beats_B_l225_225523


namespace find_number_l225_225238

theorem find_number (x : ℤ) : 3 * ((x - 50) / 4) + 28 = 73 → x = 110 :=
by
  intro h,
  sorry

end find_number_l225_225238


namespace find_ab_l225_225339

theorem find_ab :
  ∃ (a b : ℕ), a < b ∧ 
    (∀ (ha : 0 < a) (hb : 0 < b), 
      (real.sqrt (1 + real.sqrt (25 + 20 * real.sqrt 2)) = real.sqrt a + real.sqrt b)) :=
begin
  use [2, 6], -- Specific values of a and b
  split, 
  exact lt_of_le_of_lt (nat.le_refl 2) (nat.lt_succ_self 5), -- a < b
  intros ha hb, 
  have a_pos : 0 < 2 := by norm_num,
  have b_pos : 0 < 6 := by norm_num,
  have eq1 : real.sqrt(25 + 20 * real.sqrt 2) = 7 := sorry, -- Simplification step
  have eq2 : real.sqrt(1 + 7) = real.sqrt 8 := by rw eq1,
  have eq3 : real.sqrt(8) = real.sqrt 2 + real.sqrt 6 := sorry, -- Validated equivalence
  exact eq3
end

end find_ab_l225_225339


namespace part1_part2_part3_l225_225134

def f (x : ℝ) : ℝ := Real.log10 (Real.log (1/2) (1/2 * x - 1))

def A : Set ℝ := {x | 2 < x ∧ x < 4}
def B : Set ℝ := {x | x < 1 ∨ x ≥ 3}

theorem part1 : A ∪ B = Set.union {x | x < 1} {x | x > 2} :=
sorry

theorem part2 : (Set.compl B) ∩ A = {x | 2 < x ∧ x < 3} :=
sorry

theorem part3 (a : ℝ) (h₁ : 2^a ∈ A) (h₂ : Real.log2 (2 * a - 1) ∈ B) : 1 < a ∧ a < 3/2 :=
sorry

end part1_part2_part3_l225_225134


namespace percentage_decrease_correct_l225_225937

noncomputable def percentage_decrease_during_second_year
  (P : ℝ)
  (price_increase_first_year : P * 1.30)
  (price_end_two_years : P * 1.04) : ℝ :=
  let price_end_first_year := P * 1.30 in
  let X := (price_end_first_year - price_end_two_years) / price_end_first_year in
  X

theorem percentage_decrease_correct 
  (P : ℝ)
  (price_increase_first_year : P * 1.30 = 1.30 * P)
  (price_end_two_years : 1.30 * P - (percentage_decrease_during_second_year P (P * 1.30) (P * 1.04) * 1.30 * P) = 1.04 * P) :
  percentage_decrease_during_second_year P (P * 1.30) (P * 1.04) = 0.2 :=
by
  sorry

end percentage_decrease_correct_l225_225937


namespace rebecca_has_22_eggs_l225_225906

-- Define the conditions
def number_of_groups : ℕ := 11
def eggs_per_group : ℕ := 2

-- Define the total number of eggs calculated from the conditions.
def total_eggs : ℕ := number_of_groups * eggs_per_group

-- State the theorem and provide the proof outline.
theorem rebecca_has_22_eggs : total_eggs = 22 := by {
  -- Proof will go here, but for now we put sorry to indicate it is not yet provided.
  sorry
}

end rebecca_has_22_eggs_l225_225906


namespace example1_example2_l225_225756

open Real

theorem example1 : 
  ∀ (f : ℝ → ℝ), f x = (x^2 - 10*x + 9) / (x^2 - 9*x + 8) → 
  tendsto f (nhds 1) (nhds (8/7)) :=
sorry

theorem example2 : 
  ∀ (g : ℝ → ℝ), g x = tan x / cos x → 
  tendsto g (nhds 0) (nhds 0) :=
sorry

end example1_example2_l225_225756


namespace maximize_profit_l225_225269

theorem maximize_profit :
  ∀ (buy_price sell_price return_price : ℝ) (sales_pattern : (ℕ × ℕ) × ℕ) (days_in_month : ℕ),
  buy_price = 0.20 ∧ sell_price = 0.30 ∧ return_price = 0.05 ∧ 
  sales_pattern = ((20, 400), 10) ∧ days_in_month = 30 →
  let x := 400 in
  let profit := 20 * sell_price * x + 10 * sell_price * 250 + 10 * return_price * (x - 250) - days_in_month * buy_price * x in
  profit = 825 :=
begin
  sorry
end

end maximize_profit_l225_225269


namespace dart_in_center_square_probability_l225_225999

noncomputable def decagon_side_length : ℝ := 1
noncomputable def mid_square_side_length : ℝ := 0.8090
noncomputable def decagon_area : ℝ := 10 / 4 * Real.cot (Real.pi / 10) * (decagon_side_length / (2 * Real.sin (Real.pi / 10))) ^ 2
noncomputable def square_area : ℝ := mid_square_side_length ^ 2

theorem dart_in_center_square_probability :
  (square_area / decagon_area) = 0.085 :=
by
   sorry

end dart_in_center_square_probability_l225_225999


namespace hyperbola_eccentricity_eq_sqrt5_l225_225004

variables {a b : ℝ}

noncomputable def eccentricity_of_hyperbola (a b: ℝ) (h1: a > 0) (h2: b > 0) :=
  let c := sqrt (a^2 + b^2) in
  c / a

theorem hyperbola_eccentricity_eq_sqrt5 (a : ℝ) (h_a : a > 0) :
  ∃ b : ℝ, b = 2 * a ∧ eccentricity_of_hyperbola a b h_a (by linarith [h_a]) = sqrt 5 :=
begin
  use 2 * a,
  split,
  { refl },
  { unfold eccentricity_of_hyperbola,
    rw [sqrt_add (square_nonneg a) (square_nonneg (2 * a)), pow_two, pow_two, mul_assoc],
    field_simp [h_a.ne.symm, ne_of_gt (by linarith [h_a])],
    ring }
end

end hyperbola_eccentricity_eq_sqrt5_l225_225004


namespace arithmetic_seq_sum_mul_3_l225_225678

-- Definition of the arithmetic sequence
def arithmetic_sequence := [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121]

-- Prove that 3 times the sum of the arithmetic sequence is 3663
theorem arithmetic_seq_sum_mul_3 : 
  3 * (arithmetic_sequence.sum) = 3663 :=
by
  sorry

end arithmetic_seq_sum_mul_3_l225_225678


namespace monic_quadratic_with_root_l225_225728

theorem monic_quadratic_with_root (a b : ℝ) (h_root1 : (a + b * I) ∈ ({2-3*I, 2+3*I} : set ℂ)) :
  ∃ p : polynomial ℂ, polynomial.monic p ∧ p.coeff 0 = 13 ∧ p.coeff 1 = -4 ∧ p.coeff 2 = 1 ∧ p.eval (a + b * I) = 0 := 
by
  sorry

end monic_quadratic_with_root_l225_225728


namespace find_angles_l225_225689

theorem find_angles 
  (angle1_small : ℝ)
  (angle2_small : ℝ)
  (angle1_large : ℝ)
  (angle2_large : ℝ)
  (sum_of_angles_triangle : ∀ (a b c : ℝ), a + b + c = 180) 
  (exterior_angle_theorem : ∀ (a b c : ℝ), c = 180 - (a + b))
  (h_angle1_small : angle1_small = 70)
  (h_angle2_small : angle2_small = 50)
  (h_angle1_large : angle1_large = 45)
  (h_angle2_large : angle2_large = 50) :
  ∃ (alpha beta : ℝ), alpha = 120 ∧ beta = 85 :=
by 
let third_angle_small := 180 - (angle1_small + angle2_small)
have h_third_angle_small : third_angle_small = 60 := by
  rw [h_angle1_small, h_angle2_small]
  norm_num
let alpha := 180 - third_angle_small
have h_alpha : alpha = 120 := by
  rw [h_third_angle_small]
  norm_num
let beta := 180 - (angle1_large + angle2_large)
have h_beta : beta = 85 := by
  rw [h_angle1_large, h_angle2_large]
  norm_num
use [alpha, beta]
rw [h_alpha, h_beta]
suffices: alpha = 120 ∧ beta = 85
sorry

end find_angles_l225_225689


namespace sum_of_x_l225_225966

-- Definitions for the conditions
def mean (a b c d e : ℝ) := (a + b + c + d + e) / 5
def product (a b c d e : ℝ) := a * b * c * d * e
def median (a b c d e : ℝ) := (List.sorted [a, b, c, d, e])!!.nth 2 

-- The problem statement
theorem sum_of_x :
  let x := 6 in
  (median 3 5 7 18 x = mean 3 5 7 18 x - 1) →
  (product 3 5 7 18 x = 11340) →
  x = 6 :=
sorry

end sum_of_x_l225_225966


namespace monic_quadratic_with_root_l225_225744

theorem monic_quadratic_with_root (p : ℝ[X]) (h1 : p.monic) (h2 : p.eval (2 - 3 * Complex.I) = 0) : 
  p = X^2 - 4 * X + 13 := 
by 
  sorry

end monic_quadratic_with_root_l225_225744


namespace area_of_triangle_OAB_l225_225375

-- Define OA and OB as vectors in space
def OA : ℝ × ℝ × ℝ := (2, 0, 1)
def OB : ℝ × ℝ × ℝ := (0, 2, 0)

-- Define the function to compute the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Define the function to compute the area of the triangle OAB
def area_triangle_OAB (OA OB : ℝ × ℝ × ℝ) : ℝ :=
  1 / 2 * magnitude(OA) * magnitude(OB)

-- The statement we need to prove
theorem area_of_triangle_OAB : area_triangle_OAB OA OB = real.sqrt 5 :=
by sorry

end area_of_triangle_OAB_l225_225375


namespace general_term_formula_sum_b_seq_first_n_terms_l225_225376

-- Define the sequence {a_n} as an increasing arithmetic sequence given the conditions
def is_arithmetic_seq (a : Nat → Int) : Prop :=
  ∃ d, d > 0 ∧ (∀ n, a (n + 1) = a n + d)

axiom a_seq : Nat → Int
axiom a_increasing_arith : is_arithmetic_seq a_seq
axiom a_2_a_4_product : a_seq 2 * a_seq 4 = 21
axiom a_1_plus_a_5 : a_seq 1 + a_seq 5 = 10

-- First problem: Proving the general term formula
theorem general_term_formula : ∀ n, a_seq n = 2 * n - 1 :=
by
  sorry

-- Define sequence {c_n} with the given condition
def c_seq : Nat → Int := λ n, 2

-- Define sequence {b_n} with the given condition
def b_seq : Nat → Int := λ n, 2^(n+1)

-- Second problem: Proving the sum of the first n terms of {b_n}
def sum_first_n_terms (b : Nat → Int) (n : Nat) : Int :=
  (List.range n).sum (λ i, b i)

theorem sum_b_seq_first_n_terms (n : Nat) : sum_first_n_terms b_seq n = 2^(n+2) - 4 :=
by
  sorry

end general_term_formula_sum_b_seq_first_n_terms_l225_225376


namespace quadratic_real_roots_range_l225_225080

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (x^2 - x - m = 0)) ↔ m ≥ -1 / 4 :=
by sorry

end quadratic_real_roots_range_l225_225080


namespace jancy_currency_notes_l225_225466

theorem jancy_currency_notes (x y : ℕ) (h1 : 70 * x + 50 * y = 5000) (h2 : y = 2) : x + y = 72 :=
by
  -- proof goes here
  sorry

end jancy_currency_notes_l225_225466


namespace sum_g_l225_225347

noncomputable def g (n : ℕ) : ℚ :=
if (real.log 4 n).is_rational then real.log 4 n else 0

theorem sum_g (S : ℚ) (h : S = (∑ n in finset.range 2024, g n)) : S = 55 / 2 :=
sorry

end sum_g_l225_225347


namespace ryan_total_pages_correct_l225_225249

/-- Ryan's brother reads one book each day with 200 pages. -/
def brother_pages_per_book : ℕ := 200

/-- Both Ryan and his brother finish reading in a week (7 days). -/
def days_in_week : ℕ := 7

/-- Ryan reads 100 more pages per day compared to his brother. -/
def extra_pages_per_day : ℕ := 100

/-- Ryan's brother reads brother_pages_per_book pages per day. -/
def brother_pages_per_day : ℕ := brother_pages_per_book

/-- Ryan reads brother_pages_per_day + extra_pages_per_day pages per day. -/
def ryan_pages_per_day : ℕ := brother_pages_per_day + extra_pages_per_day

/-- The total number of pages Ryan reads in a week (7 days). -/
def ryan_total_pages : ℕ := ryan_pages_per_day * days_in_week

theorem ryan_total_pages_correct : ryan_total_pages = 2100 := by
  unfold ryan_total_pages
  unfold ryan_pages_per_day
  unfold brother_pages_per_day
  unfold days_in_week
  unfold brother_pages_per_book
  unfold extra_pages_per_day
  calc
    (brother_pages_per_book + extra_pages_per_day) * days_in_week
        = (200 + 100) * 7            : by rfl
    ... = 300 * 7                    : by rfl
    ... = 2100                       : by rfl

end ryan_total_pages_correct_l225_225249


namespace area_of_equilateral_triangle_l225_225886

theorem area_of_equilateral_triangle
  (X Y Z Q : Type)
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (QX QY QZ XY XZ YZ : ℝ)
  (QXZ_eq_5 : dist Q X = 5)
  (QYZ_eq_7 : dist Q Y = 7)
  (QZZ_eq_13 : dist Q Z = 13)
  (triangle_XYZ : equilateral_triangle X Y Z) :
  triangle_area X Y Z = 56 := 
sorry

end area_of_equilateral_triangle_l225_225886


namespace monic_quadratic_with_roots_l225_225730

-- Define the given conditions and roots
def root1 : ℂ := 2 - 3 * complex.i
def root2 : ℂ := 2 + 3 * complex.i

-- Define the polynomial with these roots
def poly (x : ℂ) : ℂ := x^2 - 4 * x + 13

-- Define the proof problem
theorem monic_quadratic_with_roots 
  (h1 : poly root1 = 0) 
  (h2 : poly root2 = 0) 
  (h_monic : ∀ x : ℂ, ∃! p : polynomial ℂ, p.monic ∧ p.coeff 2 = 1) : 
  ∃ p : polynomial ℝ, p.monic ∧ ∀ x : ℂ, p.eval x = poly x := 
sorry

end monic_quadratic_with_roots_l225_225730


namespace sum_even_integers_between_5_and_20_l225_225233

theorem sum_even_integers_between_5_and_20 : 
  (finset.filter (λ x, x % 2 = 0) (finset.Ico 6 20)).sum id = 84 :=
begin
  sorry
end

end sum_even_integers_between_5_and_20_l225_225233


namespace number_of_two_digit_factors_2_pow_18_minus_1_is_zero_l225_225068

theorem number_of_two_digit_factors_2_pow_18_minus_1_is_zero :
  (∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ n ∣ (2^18 - 1)) = false :=
by sorry

end number_of_two_digit_factors_2_pow_18_minus_1_is_zero_l225_225068


namespace total_hours_worked_l225_225996

-- Definition of the given conditions.
def hours_software : ℕ := 24
def hours_help_user : ℕ := 17
def percentage_other_services : ℚ := 0.4

-- Statement to prove.
theorem total_hours_worked : ∃ (T : ℕ), hours_software + hours_help_user + percentage_other_services * T = T ∧ T = 68 :=
by {
  -- The proof will go here.
  sorry
}

end total_hours_worked_l225_225996


namespace solution_set_of_inequality_l225_225041

theorem solution_set_of_inequality (a b : ℝ) (h1 : a > 0) (h2 : 2 * a + b = 0) : 
  {x : ℝ | b * x^2 - a * x > 0} = set.Ioo (-1/2) 0 :=
by 
  sorry

end solution_set_of_inequality_l225_225041


namespace max_value_cos_plus_three_sin_l225_225011

theorem max_value_cos_plus_three_sin :
  ∀ x : ℝ, ∃ M : ℝ, M = √10 ∧ ∀ x : ℝ, cos x + 3 * sin x ≤ M := 
sorry

end max_value_cos_plus_three_sin_l225_225011


namespace exists_double_area_square_l225_225897

-- Definition for a point in a grid
structure Point :=
  (x : ℕ) (y : ℕ)

-- Definition of a legal square in the grid
structure Square :=
  (a b c d : Point)

-- Area function for a square where the side length is determined by the distance between points
def area (S : Square) : ℕ :=
  let side_length := Math.sqrt (abs ((S.a.x - S.b.x)^2 + (S.a.y - S.b.y)^2)) in
  side_length * side_length

-- Definition of a legal square as vertices being grid points
def is_legal_square (S : Square) : Prop :=
    S.a.x ≠ S.b.x ∧ S.a.y ≠ S.b.y ∧
    S.a.x ≠ S.c.x ∧ S.a.y ≠ S.c.y ∧
    S.a.x ≠ S.d.x ∧ S.a.y ≠ S.d.y ∧
    S.b.x ≠ S.c.x ∧ S.b.y ≠ S.c.y ∧
    S.b.x ≠ S.d.x ∧ S.b.y ≠ S.d.y ∧
    S.c.x ≠ S.d.x ∧ S.c.y ≠ S.d.y

theorem exists_double_area_square (S : Square) 
  (h : is_legal_square S) : ∃ T : Square, is_legal_square T ∧ area T = 2 * area S :=
by
  sorry

end exists_double_area_square_l225_225897


namespace triangle_area_and_side_l225_225817

theorem triangle_area_and_side (A B C a b c : ℝ) (h1 : b * sin A = a * sin C) (h2 : c = 1) :
  b = 1 ∧ (1 / 2 * b * c * sin A <= 1 / 2) :=
by sorry

end triangle_area_and_side_l225_225817


namespace problem1_problem2_l225_225613

-- Problem 1
theorem problem1 :
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 48 + Real.sqrt 54 = 4 + Real.sqrt 6) :=
by
  sorry

-- Problem 2
theorem problem2 :
  (Real.sqrt 8 + Real.sqrt 32 - Real.sqrt 2 = 5 * Real.sqrt 2) :=
by
  sorry

end problem1_problem2_l225_225613


namespace range_of_a_l225_225357

def f (x : ℝ) : ℝ := 2^(x-2) - 1
def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a * Real.exp x

theorem range_of_a {a : ℝ} :
  (∃ α ∈ (set_of (λ α, f α = 0)),
   ∃ β ∈ (set_of (λ β, g β a = 0)) (|α - β| < 1)) →
  (a ∈ (set.Ioc (1 / Real.exp 1) (4 / (Real.exp 2)^2)))
:= sorry

end range_of_a_l225_225357


namespace product_of_remaining_numbers_is_12_l225_225203

noncomputable def final_numbers_product : ℕ := 
  12

theorem product_of_remaining_numbers_is_12 :
  ∀ (initial_ones initial_twos initial_threes initial_fours : ℕ)
  (erase_add_op : Π (a b c : ℕ), Prop),
  initial_ones = 11 ∧ initial_twos = 22 ∧ initial_threes = 33 ∧ initial_fours = 44 ∧
  (∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c → erase_add_op a b c) →
  (∃ (final1 final2 final3 : ℕ), erase_add_op 11 22 33 → final1 * final2 * final3 = final_numbers_product) :=
sorry

end product_of_remaining_numbers_is_12_l225_225203


namespace avg_age_women_correct_l225_225925

-- Let t1, t2, and t3 be the ages of the three men
def t1 := 18
def t2 := 22
def t3 := 28

-- Total age of the three replaced men
def age_men3 := t1 + t2 + t3

-- Number of people (7) and the increase in average age (3 years)
def n := 7
def increase_avg_age := 3

-- Calculate the total increase in age for 7 people
def total_age_increase := n * increase_avg_age

-- Total age of the three women
def age_women3 := age_men3 + total_age_increase

-- The average age of the three women
def avg_age_women := age_women3 / 3

-- Theorem to prove their average age is 29.67 years
theorem avg_age_women_correct : avg_age_women = 29.67 :=
by
  simp [age_men3, total_age_increase, age_women3, avg_age_women]
  sorry

end avg_age_women_correct_l225_225925


namespace vinegar_weight_per_ml_l225_225521

theorem vinegar_weight_per_ml (bowl_capacity oil_fraction vinegar_fraction oil_weight_per_ml total_weight : ℝ)
    (h_bowl : bowl_capacity = 150)
    (h_oil_frac : oil_fraction = 2/3)
    (h_vinegar_frac : vinegar_fraction = 1/3)
    (h_oil_weight : oil_weight_per_ml = 5)
    (h_total_weight : total_weight = 700) :
    let oil_volume := oil_fraction * bowl_capacity,
        vinegar_volume := vinegar_fraction * bowl_capacity,
        oil_weight := oil_volume * oil_weight_per_ml,
        vinegar_weight := total_weight - oil_weight,
        vinegar_weight_per_ml := vinegar_weight / vinegar_volume
    in vinegar_weight_per_ml = 4 := 
by sorry

end vinegar_weight_per_ml_l225_225521


namespace cone_sphere_problem_l225_225286

-- Define the necessary conditions as Lean definitions
def cone_base_radius : ℝ := 15
def cone_height : ℝ := 30
def sphere_radius_expr (b d : ℝ) : ℝ := b * Real.sqrt d - b

-- Translate the main question statement into Lean
theorem cone_sphere_problem :
  ∃ b d : ℝ, sphere_radius_expr b d = 15 * Real.sqrt 5 - 15 ∧ b + d = 20 :=
by
  sorry

end cone_sphere_problem_l225_225286


namespace problem_one_problem_two_l225_225408

-- Define the function f
def f (a c : ℝ) (x : ℝ) : ℝ := - (1 / 2) * x ^ 2 + (1 / 2) * a * (5 - a) * x + c

-- Problem (1): If c = 16, prove that f(2) > 0 ⟺ -2 < a < 7
theorem problem_one (a : ℝ) : f a 16 2 > 0 ↔ -2 < a ∧ a < 7 :=
by
  sorry

-- Problem (2): If a = 4, prove that ∀ x ∈ Iic 1, f(x) < 0 ⟺ c < -3/2
theorem problem_two (c : ℝ) : (∀ x : ℝ, x ≤ 1 → f 4 c x < 0) ↔ c < -3 / 2 :=
by
  sorry

end problem_one_problem_two_l225_225408


namespace power_function_inverse_l225_225780

theorem power_function_inverse:
  ∃ (α : ℝ), (∀ x : ℝ, f x = x ^ α) ∧ 
  (f⁻¹ 6 = 36) →
  (f \left(\frac{1}{9}) = \frac{1}{3}) :=
by
  -- Placeholder for the function.
  let f (x : ℝ) : ℝ := x ^ α
  have hα := -- Proof that α = 1/2
  sorry

end power_function_inverse_l225_225780


namespace eden_average_speed_approx_l225_225324

def distance_segment1 := 30
def speed_segment1 := 20
def distance_segment2 := 50
def speed_segment2 := 40
def distance_segment3 := 75
def speed_segment3 := 25
def distance_segment4 := 45
def speed_segment4 := 35

def total_distance := distance_segment1 + distance_segment2 + distance_segment3 + distance_segment4
def time_segment1 := distance_segment1 / speed_segment1
def time_segment2 := distance_segment2 / speed_segment2
def time_segment3 := distance_segment3 / speed_segment3
def time_segment4 := distance_segment4 / speed_segment4
def total_time := time_segment1 + time_segment2 + time_segment3 + time_segment4
def average_speed := total_distance / total_time

theorem eden_average_speed_approx : abs (average_speed - 28.44) < 0.01 := by
  sorry

end eden_average_speed_approx_l225_225324


namespace circumscribed_radius_l225_225657

theorem circumscribed_radius (φ : ℝ) :  
  (8 : ℝ) * real.sec (φ / 2) = 8 / real.cos (φ / 2) := 
by 
  sorry

end circumscribed_radius_l225_225657


namespace total_number_of_games_l225_225566

theorem total_number_of_games (n : ℕ) (h : n = 10) : ∑ i in finset.range n, i = (10 * 9) / 2 :=
by {
  rw h,
  sorry
}

end total_number_of_games_l225_225566


namespace no_integer_solution_l225_225798

theorem no_integer_solution :
  ∀ (x y : ℤ), ¬(x^4 + x + y^2 = 3 * y - 1) :=
by
  intros x y
  sorry

end no_integer_solution_l225_225798


namespace part1_part2_l225_225767

section InequalityProofs

variable (a : ℝ) (x : ℝ)

def quadratic_function (x : ℝ) (a : ℝ) : ℝ :=
  -x^2 + 2 * a * x - 3

theorem part1 (a : ℝ) : 
  a = -2 → (∃ x : ℝ, -3 < x ∧ x < -1 ∧ quadratic_function x a < 0) :=
by
  sorry

theorem part2 (a : ℝ) : 
  (∀ x ∈ set.Icc (1 : ℝ) (5 : ℝ), quadratic_function x a < 3 * a * x) → a > -2 * Real.sqrt 3 :=
by
  sorry

end InequalityProofs

end part1_part2_l225_225767


namespace distance_vertex_to_asymptote_l225_225791

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the vertex of the hyperbola
def vertex : ℝ × ℝ := (1, 0)

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = sqrt 3 * x

-- Define the distance formula from a point to a line
def distance (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- Prove that the distance from the vertex to the asymptote is sqrt(3)/2
theorem distance_vertex_to_asymptote : 
  distance 1 0 (sqrt 3) (-1) 0 = sqrt 3 / 2 :=
by 
  sorry

end distance_vertex_to_asymptote_l225_225791


namespace interval_monotonic_increase_sin_of_double_theta_l225_225063

-- Definitions of vectors and the function f.
def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, -Real.sqrt 3 * Real.cos x)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def f (x : ℝ) : ℝ := Real.dot_product (vector_a x) (vector_b x) + 1

-- Statement for the interval of monotonic increase.
theorem interval_monotonic_increase :
  ∀ (x : ℝ) (k : ℤ), k ∈ ℤ → (f (x) = Real.cos(2*x + Real.arccos(-sqrt 3)) + 1) →
  (∃ (I : set ℝ), I = { x | (π/3 + k*π ≤ x) ∧ (x ≤ 5*π/6 + k*π)}) :=
sorry

-- Statement for the value of sin(2θ).
theorem sin_of_double_theta (θ : ℝ) :
  θ ∈ Icc (π/3) (2*π/3) ∧ f(θ) = 5/6 → 
  Real.sin (2*θ) = (2* Real.sqrt 3 - Real.sqrt 5) / 6 :=
sorry

end interval_monotonic_increase_sin_of_double_theta_l225_225063


namespace perimeter_of_figure_l225_225308

def side_length : ℕ := 1
def num_vertical_stacks : ℕ := 2
def num_squares_per_stack : ℕ := 3
def gap_between_stacks : ℕ := 1
def squares_on_top : ℕ := 3
def squares_on_bottom : ℕ := 2

theorem perimeter_of_figure : 
  (2 * side_length * squares_on_top) + (2 * side_length * squares_on_bottom) + 
  (2 * num_squares_per_stack * num_vertical_stacks) + (2 * num_squares_per_stack * squares_on_top)
  = 22 :=
by
  sorry

end perimeter_of_figure_l225_225308


namespace square_area_l225_225257

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem square_area (vertex : ℝ × ℝ) (center : ℝ × ℝ) :
  vertex = (-6, -4) ∧ center = (3, 2) → 
  let d := 2 * distance vertex center in
  let s := d / sqrt 2 in
  let area := s^2 in
  area = 234 := 
by
  intros h
  cases h
  let v := vertex
  let c := center
  have hvc : distance v c = sqrt 117 := sorry
  have d_def : d = 2 * sqrt 117 := sorry
  have s_def : s = 6 * sqrt (13/2) := sorry
  have area_def : area = (6 * sqrt (13/2))^2 := sorry
  have area_calc : area = 234 := sorry
  exact area_calc

end square_area_l225_225257


namespace minimal_rooms_l225_225676

theorem minimal_rooms (k : ℕ) : 
  ∃ n : ℕ, 
    (∀ m : ℕ,
      (k = 2 * m ∧ n = 100 * (m + 1)) ∨ 
      (k = 2 * m + 1 ∧ n = 100 * (m + 1) + 1)) :=
begin
  sorry
end

end minimal_rooms_l225_225676


namespace range_of_a_not_empty_solution_set_l225_225813

theorem range_of_a_not_empty_solution_set :
  {a : ℝ | ∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0} =
  {a : ℝ | a ∈ {a : ℝ | a < -2} ∪ {a : ℝ | a ≥ 6 / 5}} :=
sorry

end range_of_a_not_empty_solution_set_l225_225813


namespace problem_l225_225358

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def log7 (x : ℝ) : ℝ := Real.log x / Real.log 7

theorem problem (x : ℝ) (h : log7 (log3 (log2 x)) = 0) : x^(-1/2) = (Real.sqrt 2) / 4 :=
begin
  sorry
end

end problem_l225_225358


namespace numCreaturesWith6Legs_11_l225_225526

def numCreaturesWith6Legs (x y : ℕ) : Prop :=
  x + y = 20 ∧ 6 * x + 10 * y = 156

theorem numCreaturesWith6Legs_11 :
  ∃ x y : ℕ, numCreaturesWith6Legs x y ∧ x = 11 :=
by
  exists 11
  exists 9
  split
  . split
    . rfl
    . norm_num
  . rfl

end numCreaturesWith6Legs_11_l225_225526


namespace three_letter_words_with_A_and_B_l225_225422

theorem three_letter_words_with_A_and_B : 
  let letters := {'A', 'B', 'C', 'D', 'E'}
  let num_3_letter_words := 5^3
  let num_without_A := 4^3
  let num_without_B := 4^3
  let num_without_A_and_B := 3^3
  let num_without_A_or_B := num_without_A + num_without_B - num_without_A_and_B
  let num_with_A_and_B := num_3_letter_words - num_without_A_or_B
  num_with_A_and_B = 24 := 
by
  sorry

end three_letter_words_with_A_and_B_l225_225422


namespace basketball_and_volleyball_students_l225_225575

variable (x : ℕ) (y : ℕ)

def soccer_students (x : ℕ) : ℕ := 2 * x - 2
def volleyball_students (x : ℕ) (y : ℕ) : ℕ := (2 * x - y) / 2 + 2

theorem basketball_and_volleyball_students :
  y = 34 → soccer_students x = y → 2 * x - 2 = 34 → x = 20 ∧ volleyball_students x 6 = 19 :=
by
  intro h_y h_soccer h_x
  rw [h_soccer, h_y] at h_x
  have h1 : 2 * x = 36 := by linarith
  have h2 : x = 20 := by linarith
  split
  { exact h2 }
  { rw [h2]
    norm_num }
  sorry

end basketball_and_volleyball_students_l225_225575


namespace monic_quadratic_with_root_l225_225742

theorem monic_quadratic_with_root (p : ℝ[X]) (h1 : p.monic) (h2 : p.eval (2 - 3 * Complex.I) = 0) : 
  p = X^2 - 4 * X + 13 := 
by 
  sorry

end monic_quadratic_with_root_l225_225742


namespace theater_ticket_sales_l225_225190

theorem theater_ticket_sales (R H : ℕ) (h1 : R = 25) (h2 : H = 3 * R + 18) : H = 93 :=
by
  sorry

end theater_ticket_sales_l225_225190


namespace tan_alpha_value_l225_225385

-- Definitions based on conditions
def isFourthQuadrant (angle : ℝ) : Prop := angle ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)

def pointOnTerminalSide (P : ℝ × ℝ) (angle : ℝ) : Prop := 
  P.1 = 4 ∧ ∃ y, ∀ α, isFourthQuadrant α ∧ sin α = y / 5 → P = (4, y)

-- Statement that needs proof
theorem tan_alpha_value (α y : ℝ) (h1 : isFourthQuadrant α) (h2 : pointOnTerminalSide (4, y) α) (h3 : sin α = y / 5) : tan α = -3 / 4 :=
sorry

end tan_alpha_value_l225_225385


namespace find_number_l225_225085

theorem find_number (x n : ℤ) 
  (h1 : 0 < x) (h2 : x < 7) 
  (h3 : x < 15) 
  (h4 : -1 < x) (h5 : x < 5) 
  (h6 : x < 3) (h7 : 0 < x) 
  (h8 : x + n < 4) 
  (hx : x = 1): 
  n < 3 := 
sorry

end find_number_l225_225085


namespace angle_of_inclination_l225_225001

theorem angle_of_inclination (α : ℝ) : 
  (∃ α, ∀ x y : ℝ, (√3 * x - y - 3 = 0) → α = 60) := sorry

end angle_of_inclination_l225_225001


namespace mara_crayons_l225_225519

theorem mara_crayons (M : ℕ):
  (0.10 * M + 10 = 14) → M = 40 :=
by 
  sorry

end mara_crayons_l225_225519


namespace find_angle_C_l225_225462

-- Given Conditions
constant A : ℝ 
constant a : ℝ 
constant c : ℝ 
constant C : ℝ 

axiom A_eq : A = π / 3
axiom a_eq : a = 3
axiom c_eq : c = sqrt 6

-- Law of Sines assumption
axiom law_of_sines (a b c : ℝ) (A B C : ℝ) : a / real.sin A = c / real.sin C

-- Prove that the value of angle C under the given conditions equals π/4 or 3π/4
theorem find_angle_C : A = π / 3 → a = 3 → c = sqrt 6 → (C = π / 4 ∨ C = 3 * π / 4) :=
by
  intro A_eq a_eq c_eq
  apply or.inl
  sorry

end find_angle_C_l225_225462


namespace Bob_wins_game_l225_225527

theorem Bob_wins_game : ∀ (a b : ℕ), 
  (a = 47) ∧ (b = 2016) → 
  (∀ pair : ℕ × ℕ, pair.fst > pair.snd → 
  ¬(pair.fst - pair.snd = 0)) → 
  ∃ n, n = 2016 ∧ nat.even n →
  Bob makes the last move
:= sorry

end Bob_wins_game_l225_225527


namespace collinear_A₂_B₂_C₂_l225_225898

noncomputable theory
open_locale classical

variables {A B C A₁ B₁ C₁ A₂ B₂ C₂ : Type}

-- Define the inputs
variables (triangle : triangle A B C)
variables (A₁_on_BC : lies_on A₁ (line B C)) 
variables (B₁_on_CA : lies_on B₁ (line C A))
variables (C₁_on_AB : lies_on C₁ (line A B))
variables (collinear_A₁_B₁_C₁ : collinear [A₁, B₁, C₁])

-- Define the angle bisectors
variables (l_A : angle_bisector A B C) 
variables (l_B : angle_bisector B C A)
variables (l_C : angle_bisector C A B)

-- Define the symmetric lines and intersection points
variables (A₂_at_symmetric : symmetric_with_respect l_A (line A A₁) (line B C) A₂)
variables (B₂_at_symmetric : symmetric_with_respect l_B (line B B₁) (line C A) B₂)
variables (C₂_at_symmetric : symmetric_with_respect l_C (line C C₁) (line A B) C₂)

-- Prove that A₂, B₂, and C₂ are collinear
theorem collinear_A₂_B₂_C₂ :
  collinear [A₂, B₂, C₂] :=
sorry

end collinear_A₂_B₂_C₂_l225_225898


namespace probability_of_A_given_B_l225_225677

def medical_teams : Type := {A, B, C, D}
def countries : Type := {Country1, Country2, Country3, Country4}

def EventA (assignment : medical_teams → countries) : Prop :=
  Function.Injective assignment

def EventB (assignment : medical_teams → countries) : Prop :=
  ∃ c : countries, ∀ t ≠ medical_teams.A, assignment t ≠ c

theorem probability_of_A_given_B : 
  P(EventA | EventB) = 2 / 9 :=
by
  sorry

end probability_of_A_given_B_l225_225677


namespace inequality_order_l225_225760

noncomputable def a : ℝ := Real.logBase 2 (7 / 5)
noncomputable def b : ℝ := 2 ^ (2 / 5)
noncomputable def c : ℝ := Real.log 2

theorem inequality_order : a < c ∧ c < b :=
by
  -- sorry to skip the proof
  sorry

end inequality_order_l225_225760


namespace trigonometric_simplification_l225_225918

theorem trigonometric_simplification :
  (sin (15 * Real.pi / 180) + sin (30 * Real.pi / 180) + sin (45 * Real.pi / 180) + sin (60 * Real.pi / 180) + sin (75 * Real.pi / 180)) / 
  (cos (10 * Real.pi / 180) * cos (20 * Real.pi / 180) * cos (30 * Real.pi / 180) * 2) = 
  2 * Real.sqrt 2 * (cos (22.5 * Real.pi / 180) * cos (7.5 * Real.pi / 180)) / (cos (10 * Real.pi / 180) * cos (20 * Real.pi / 180) * cos (30 * Real.pi / 180)) :=
by sorry

end trigonometric_simplification_l225_225918


namespace log_cubed_root_sixteen_log_exp_l225_225302

theorem log_cubed_root_sixteen :
  logBase 2 (cubeRoot 16) = (4 : ℝ) / 3 := by
  sorry

def logBase (b x : ℝ) := Real.log x / Real.log b

def cubeRoot (x : ℝ) := x^(1 / 3)

variables (b : ℝ) (a : ℝ) (c : ℝ)
theorem log_exp :
  logBase b (a^c) = c * logBase b a := by
  sorry

example :
  logBase 2 2 = 1 := by
  sorry

end log_cubed_root_sixteen_log_exp_l225_225302


namespace length_DE_approx_l225_225460

noncomputable def length_DE (A B C D E : Point) : Real :=
  let AD := 100   -- m
  let EB := 200   -- m
  let ∠ACD := 30  -- degrees
  let ∠DCE := 50  -- degrees
  let ∠ECB := 35  -- degrees
  sorry

theorem length_DE_approx (A B C D E : Point) (h1 : AD = 100) (h2 : EB = 200) 
  (h3 : ∠ACD = 30) (h4 : ∠DCE = 50) (h5 : ∠ECB = 35) : length_DE A B C D E = 116.3 :=
sorry

end length_DE_approx_l225_225460


namespace find_abc_l225_225758

open Set

variables {R : Type*} [linear_ordered_field R]

def quadratic_set (a b : R) : Set R := {x | x^2 + a * x + b = 0}

theorem find_abc : ∃ (a b c : R),
  quadratic_set a b ∪ quadratic_set c 15 = {3, 5} ∧
  quadratic_set a b ∩ quadratic_set c 15 = {3} ∧
  a = -6 ∧ b = 9 ∧ c = -8 :=
begin
  use [-6, 9, -8],
  split,
  { ext x,
    split,
    { intro hx,
      cases hx with hA hB,
      { rw [quadratic_set, mem_set_of_eq] at hA, 
        by_cases H : x=3,
        { rw H,
          exact or.inl rfl,},
        { rw H at hA,
          exfalso,
          sorry }, },
      { rw [quadratic_set, mem_set_of_eq] at hB, 
        by_cases H : x=3,
        { rw H,
          exact or.inl rfl,},
        { rw H at hB,
          exfalso,
          sorry }, }},
    { intro hx,
      cases hx,
      { left, 
        rw [quadratic_set, mem_set_of_eq],
        sorry },
      { right,
        rw [quadratic_set, mem_set_of_eq],
        sorry }}, },
  split,
  { ext x,
    split,
    { intro hx,
      rw [mem_inter_iff, quadratic_set, quadratic_set, mem_set_of_eq, mem_set_of_eq] at hx,
      sorry },
    { rw [quadratic_set, quadratic_set, mem_set_of_eq, mem_set_of_eq],
      intro hx,
      have h : x^2 - 8 * x + 15 = 0 := sorry,
      exact mem_inter.mpr ⟨sorry, sorry⟩ }, },
  split, refl,
  split, refl,
  refl
end

end find_abc_l225_225758


namespace other_root_is_minus_5_l225_225572

-- conditions
def polynomial (x : ℝ) := x^4 - x^3 - 18 * x^2 + 52 * x + (-40 : ℝ)
def r1 := 2
def f_of_r1_eq_zero : polynomial r1 = 0 := by sorry -- given condition

-- the proof problem
theorem other_root_is_minus_5 : ∃ r, polynomial r = 0 ∧ r ≠ r1 ∧ r = -5 :=
by
  sorry

end other_root_is_minus_5_l225_225572


namespace load_transportable_l225_225639

theorem load_transportable :
  ∃ (n : ℕ), n ≤ 11 ∧ (∀ (box_weight : ℕ) (total_weight : ℕ),
    total_weight = 13500 ∧ 
    box_weight ≤ 350 ∧ 
    (n * 1500) ≥ total_weight) :=
by
  sorry

end load_transportable_l225_225639


namespace enclosed_polygons_l225_225651

theorem enclosed_polygons (n : ℕ) :
  (∃ α β : ℝ, (15 * β) = 360 ∧ β = 180 - α ∧ (15 * α) = 180 * (n - 2) / n) ↔ n = 15 :=
by sorry

end enclosed_polygons_l225_225651


namespace salmon_at_rest_oxygen_l225_225300

noncomputable def oxy_consumption (x : ℝ) : ℝ :=
  (1 / 2) * log 3 (x / 100 * real.pi)

theorem salmon_at_rest_oxygen (x : ℝ) : oxy_consumption x = 0 → x = 100 / real.pi := 
by
sorry

end salmon_at_rest_oxygen_l225_225300


namespace area_of_cyclic_quadrilateral_l225_225939

theorem area_of_cyclic_quadrilateral (A B C D : Point) (R : ℝ) (φ : ℝ)
  (h_inscribed : IsInscribed A B C D R) (h_phi : AngleBetweenDiagonals A B C D = φ) :
  let S := 2 * R^2 * sin (angle A B C) * sin (angle B C D) * sin φ in
  area A B C D = S := by
  sorry

end area_of_cyclic_quadrilateral_l225_225939


namespace initial_cloves_l225_225137

theorem initial_cloves (used_cloves left_cloves initial_cloves : ℕ) (h1 : used_cloves = 86) (h2 : left_cloves = 7) : initial_cloves = 93 :=
by
  sorry

end initial_cloves_l225_225137


namespace maximum_volume_maximum_surface_area_l225_225184

variables (A B C D P Q : Type)
variables (a b k V A : ℝ)

-- Conditions
variable tetrahedron : [tetrahedron ABCD]
variable edges : [edge A B has length a, edge C D has length b]
variable midpoints : [segment P Q connects midpoints A B and C D, segment P Q has length k]

-- Proofs
theorem maximum_volume : V = (a * b * k) / 6 := sorry

theorem maximum_surface_area : A = a * sqrt(k^2 + (b^2 / 4)) + b * sqrt(k^2 + (a^2 / 4)) := sorry

end maximum_volume_maximum_surface_area_l225_225184


namespace emily_cards_l225_225703

theorem emily_cards (initial_cards : ℕ) (total_cards : ℕ) (given_cards : ℕ) 
  (h1 : initial_cards = 63) (h2 : total_cards = 70) 
  (h3 : total_cards = initial_cards + given_cards) : 
  given_cards = 7 := 
by 
  sorry

end emily_cards_l225_225703


namespace simplify_sqrt4_l225_225590

theorem simplify_sqrt4 (a b : ℕ) (h : 0 < a ∧ 0 < b ∧ (sqrt 4 (2^5 * 3^4) = a * sqrt 4 b)) : a + b = 164 := 
  sorry

end simplify_sqrt4_l225_225590


namespace find_m_value_l225_225980

theorem find_m_value :
  62519 * 9999 = 625127481 :=
  by sorry

end find_m_value_l225_225980


namespace rectangular_park_area_l225_225650

/-- Define the conditions for the rectangular park -/
def rectangular_park (w l : ℕ) : Prop :=
  l = 3 * w ∧ 2 * (w + l) = 72

/-- Prove that the area of the rectangular park is 243 square meters -/
theorem rectangular_park_area (w l : ℕ) (h : rectangular_park w l) : w * l = 243 := by
  sorry

end rectangular_park_area_l225_225650


namespace recurring_fraction_division_l225_225223

/--
Given x = 0.\overline{36} and y = 0.\overline{12}, prove that x / y = 3.
-/
theorem recurring_fraction_division 
  (x y : ℝ)
  (h1 : x = 0.36 + 0.0036 + 0.000036 + 0.00000036 + ......) -- representation of 0.\overline{36}
  (h2 : y = 0.12 + 0.0012 + 0.000012 + 0.00000012 + ......) -- representation of 0.\overline{12}
  : x / y = 3 :=
  sorry

end recurring_fraction_division_l225_225223


namespace paul_time_4_laps_l225_225900

-- Define Paul's speeds and distances
def first_segment_distance : ℕ := 200
def second_segment_distance : ℕ := 400
def first_segment_speed : ℕ := 3
def second_segment_speed : ℕ := 6
def laps : ℕ := 4

-- Calculate time for segments
def time_for_first_segment : ℚ := first_segment_distance / first_segment_speed
def time_for_second_segment : ℚ := second_segment_distance / second_segment_speed
def total_time_for_one_lap : ℚ := time_for_first_segment + time_for_second_segment

-- Total time for 4 laps
def total_time : ℚ := laps * total_time_for_one_lap

-- Convert total time to seconds, minutes, and seconds
def total_time_seconds : ℚ := total_time
def total_time_minutes : ℕ := total_time_seconds.toNat / 60
def remaining_seconds : ℕ := total_time_seconds.toNat % 60

-- Prove that Paul took 8 minutes and 53 seconds to run 4 laps
theorem paul_time_4_laps :
  total_time_minutes = 8 ∧ remaining_seconds = 53 :=
by
  -- The main calculation must be done here, but since we don't need to prove it now, we'll use sorry.
  sorry

end paul_time_4_laps_l225_225900


namespace watch_cost_l225_225066

theorem watch_cost (number_of_dimes : ℕ) (value_of_dime : ℝ) (h : number_of_dimes = 50) (hv : value_of_dime = 0.10) :
  number_of_dimes * value_of_dime = 5.00 :=
by
  sorry

end watch_cost_l225_225066


namespace expand_expression_l225_225328

theorem expand_expression (x y : ℝ) : 
  5 * (4 * x^2 + 3 * x * y - 4) = 20 * x^2 + 15 * x * y - 20 := 
by 
  sorry

end expand_expression_l225_225328


namespace problem_l225_225089

noncomputable def triangle_angle_C (a b c A : ℝ) (h1 : c ≠ 0)
  (h2 : b / c = Real.sqrt 3 * Real.sin A + Real.cos A) : Prop :=
  ∃ C : ℝ, C = Real.pi / 6

noncomputable def max_area_of_triangle (a b A : ℝ) (h1 : ∃ C, C = Real.pi / 6)
  (h2 : b / 2 = Real.sqrt 3 * Real.sin A + Real.cos A) : Prop :=
  ∃ area : ℝ, area = 2 + Real.sqrt 3

theorem problem (a b c A : ℝ) (h1 : c ≠ 0)
  (h2 : b / c = Real.sqrt 3 * Real.sin A + Real.cos A) :
  triangle_angle_C a b c A h1 h2 ∧
  max_area_of_triangle a b A
    (triangle_angle_C a b c A h1 h2)
    (by { have h : c = 2, sorry }) :=
sorry

end problem_l225_225089


namespace standard_poodle_taller_than_miniature_l225_225940

theorem standard_poodle_taller_than_miniature :
  ∀ (standard_poodle toy_poodle : ℕ) (miniature_poodle : ℕ → ℕ),
    standard_poodle = 28 →
    toy_poodle = 14 →
    miniature_poodle toy_poodle = toy_poodle + 6 →
    standard_poodle - miniature_poodle toy_poodle = 8 :=
by
  intros standard_poodle toy_poodle miniature_poodle Hsp Htp Hmp
  rw [Hsp, Htp, Hmp]
  simp
  sorry

end standard_poodle_taller_than_miniature_l225_225940


namespace second_option_feasible_l225_225708

def Individual : Type := String
def M : Individual := "M"
def I : Individual := "I"
def P : Individual := "P"
def A : Individual := "A"

variable (is_sitting : Individual → Prop)

-- Given conditions
axiom fact1 : ¬ is_sitting M
axiom fact2 : ¬ is_sitting A
axiom fact3 : ¬ is_sitting M → is_sitting I
axiom fact4 : is_sitting I → is_sitting P

theorem second_option_feasible :
  is_sitting I ∧ is_sitting P ∧ ¬ is_sitting M ∧ ¬ is_sitting A :=
by
  sorry

end second_option_feasible_l225_225708


namespace periodic_even_function_sum_l225_225384

noncomputable def f : ℝ → ℝ :=
λ x, if -3 ≤ x ∧ x ≤ -1 then -(x + 2)^2
else if -1 < x ∧ x ≤ 0 then 2^x + 1
else if x < -3 then f (3 - x)
else if x > 0 then f (-x)
else 0

theorem periodic_even_function_sum :
  (f 1) + (f 2) + (f 3) + ... + (f 2018) = -673 :=
sorry

end periodic_even_function_sum_l225_225384


namespace second_day_price_l225_225604

noncomputable theory
open_locale classical

variables (O P : ℝ)

-- Conditions
def equal_amounts (O W : ℝ) : Prop := O = W
def first_day_volume (O : ℝ) : ℝ := 2 * O
def second_day_volume (O : ℝ) : ℝ := 3 * O
def equal_revenue (O : ℝ) (P : ℝ) : Prop := 2 * O * 0.50 = 3 * O * P

-- Theorem
theorem second_day_price 
  (O : ℝ) 
  (W : ℝ) 
  (h1 : O = W) 
  (h2 : first_day_volume O = 2 * O) 
  (h3 : second_day_volume O = 3 * O) 
  (h4 : equal_revenue O P) : 
  P = 2 * 0.50 / 3 := 
sorry

end second_day_price_l225_225604


namespace johns_order_cost_after_discount_l225_225289

theorem johns_order_cost_after_discount (n_items : ℕ) (cost_per_item : ℕ) (discount_percentage : ℚ) (threshold : ℕ) (n_items = 7) (cost_per_item = 200) (discount_percentage = 10 / 100) (threshold = 1000) :
    let total_cost := n_items * cost_per_item in
    let eligible_amount := total_cost - threshold in
    let discount := discount_percentage * eligible_amount in
    let final_cost := total_cost - discount in
    final_cost = 1360 := by
  sorry

end johns_order_cost_after_discount_l225_225289


namespace range_of_slope_on_CurveC_is_correct_l225_225268

-- Define the parametric equations of the curve C
def CurveC (θ : ℝ) : ℝ × ℝ := 
  let x := -2 + Real.cos θ
  let y := Real.sin θ
  (x, y)

def slope_on_CurveC_range : Set ℝ := {m | ∃ θ : ℝ, let (x, y) := CurveC θ in m = y / x}

theorem range_of_slope_on_CurveC_is_correct : 
  slope_on_CurveC_range = Set.Icc (-Real.sqrt 3 / 3) (Real.sqrt 3 / 3) :=
by 
  sorry

end range_of_slope_on_CurveC_is_correct_l225_225268


namespace internal_tangent_length_l225_225209

theorem internal_tangent_length (A B C D E : Point) (r₁ r₂ : ℝ) :
    circle_center_radius A 6 → circle_center_radius B 3 → 
    touches_internally A C 6 → touches_internally B D 3 → 
    coll AB E → coll CD E → 
    distance A E = 10 → distance C D = 12 :=
by
  sorry

end internal_tangent_length_l225_225209


namespace chessboard_partition_perimeter_l225_225019

-- Suppose m is a natural number such that m >= 1.
variable (m : ℕ) (hm : m ≥ 1)

-- Define the problem statement: The smallest possible sum of the perimeters of the rectangles in the partition.
theorem chessboard_partition_perimeter : sorry

end chessboard_partition_perimeter_l225_225019


namespace smallest_multiple_of_18_and_40_l225_225589

-- Define the conditions
def multiple_of_18 (n : ℕ) : Prop := n % 18 = 0
def multiple_of_40 (n : ℕ) : Prop := n % 40 = 0

-- Prove that the smallest number that meets the conditions is 360
theorem smallest_multiple_of_18_and_40 : ∃ n : ℕ, multiple_of_18 n ∧ multiple_of_40 n ∧ ∀ m : ℕ, (multiple_of_18 m ∧ multiple_of_40 m) → n ≤ m :=
  by
    let n := 360
    -- We have to prove that 360 is the smallest number that is a multiple of both 18 and 40
    sorry

end smallest_multiple_of_18_and_40_l225_225589


namespace pe_vector_match_l225_225033

-- Definitions for vectors and equilateral triangle setup
noncomputable def vec {R : Type} [CommRing R] := (R × R × R)

variables {A B C D P E : vec ℝ}
variables {AB AC AD PE PB PC : vec ℝ}
variables {side_length : ℝ}

-- Given conditions
def equilateral_triangle (A B C : vec ℝ) (side_length : ℝ) : Prop :=
  (A - B).norm = side_length ∧ (B - C).norm = side_length ∧ (C - A).norm = side_length

def midpoint (D B C : vec ℝ) : Prop :=
  D = (B + C) / 2

def on_segment (P A D : vec ℝ) : Prop :=
  ∃ λ : ℝ, 0 < λ ∧ λ < 1 ∧ P = λ • (D - A) + A

def perpendicular (PE AC : vec ℝ) : Prop :=
  PE.dot AC = 0

def dot_product_condition (PB PC : vec ℝ) : Prop :=
  PB.dot PC = -2/3

-- The main statement to be proved
theorem pe_vector_match : 
  ∀ (A B C D P E : vec ℝ)
  (side_length : ℝ),
  equilateral_triangle A B C side_length →
  midpoint D B C →
  on_segment P A D →
  perpendicular (E - P) (C - A) →
  dot_product_condition (B - P) (C - P) →
  (E - P) = -1/3 • (B - A) + 1/6 • (C - A) :=
by
  sorry

end pe_vector_match_l225_225033


namespace trajectory_equation_find_k_tangent_line_to_trajectory_l225_225006

-- Given point P(x, y) with y ≥ 0, and the distance condition |PM| = |PN| + 1/2, 
-- prove that the equation of the trajectory of point P is y^2 - 2y + (x^2 + 1/4) = 0.
theorem trajectory_equation (x y : ℝ) (hy : y ≥ 0) 
    (h : (sqrt (x^2 + (y - 1/2)^2)) = y + 1/2) :
    y^2 - 2 * y + (x^2 + 1 / 4) = 0 := 
sorry

-- Given the equation of the trajectory y^2 - 2y + (x^2 + 1/4) = 0,
-- if the line l: y = kx + 1 intersects the trajectory at points A and B and |AB| = 2√6,
-- show that k = ±1.
theorem find_k (k : ℝ) (h_intersect : ∀ x, (x^2 - 2*k*x - 2 = 0) → (
   let y1 := k * x + 1 in (y1 * y1) - 2 * y1 + (x^2 + 1 / 4) = 0))
    (h_AB : ∃ x1 x2, x1 + x2 = 2 * k ∧ x1 * x2 = -2 ∧ 
            (sqrt ((x1 - x2)^2 + ((k * x1 + 1) - (k * x2 + 1))^2)) = 2 * sqrt 6) :
    k = 1 ∨ k = -1 :=
sorry

-- Given point Q(1, 1/2) on the trajectory y^2 - 2y + (x^2 + 1/4) = 0,
-- prove that the equation of the tangent line to the trajectory at Q is 2x - 2y - 1 = 0.
theorem tangent_line_to_trajectory (x y : ℝ) (h_trajectory : y^2 - 2*y + (x^2 + 1/4) = 0) :
    ∀ (x1 := 1) (y0 := 1 / 2), 
    (y - y0 = (x - x1)) → (2*x - 2*y - 1 = 0) :=
sorry

end trajectory_equation_find_k_tangent_line_to_trajectory_l225_225006


namespace circle_area_l225_225814

-- Define the conditions and the final statement
theorem circle_area :
  (∀ (r : ℝ), 3 * (1 / (2 * Real.pi * r)) = 1 / 2 * (2 * r)) →
  (∀ (r : ℝ), r > 0 → r^2 = 3 / (2 * Real.pi)) →
  (∀ (r : ℝ), r > 0 → π * r^2 = 3 / 2) := by
  intro H1 H2 r r_pos,
  have r_sq : r^2 = 3 / (2 * Real.pi) := H2 r r_pos,
  calc
    π * r^2 = π * (3 / (2 * Real.pi)) : by rw [r_sq]
    ... = 3 / 2 : by ring

end circle_area_l225_225814


namespace tax_on_other_items_l225_225525

noncomputable def percentage_tax_on_other_items (total_amount spent_on_clothing spent_on_food spent_on_other_items tax_on_clothing total_tax : ℝ) : ℝ :=
  total_tax - tax_on_clothing - total_amount * 0

theorem tax_on_other_items
  (total_amount spent_on_clothing spent_on_food spent_on_other_items tax_on_clothing total_tax percentage_tax : ℝ)
  (h1 : spent_on_clothing = 0.4 * total_amount)
  (h2 : spent_on_food = 0.3 * total_amount)
  (h3 : spent_on_other_items = 0.3 * total_amount)
  (h4 : tax_on_clothing = 0.04 * spent_on_clothing)
  (h5 : total_tax = 0.04 * total_amount)
  (h6 : total_amount = 100) : 
  percentage_tax = 8 := 
by
  let tax_on_other_items := percentage_tax_on_other_items total_amount spent_on_clothing spent_on_food spent_on_other_items tax_on_clothing total_tax
  have h7 : tax_on_other_items = 2.4 := sorry
  have h8 : percentage_tax = (tax_on_other_items / spent_on_other_items) * 100 := sorry
  have h9 : percentage_tax = 8 := 
    by simp [h7, h3]; sorry
  exact h9

end tax_on_other_items_l225_225525


namespace find_value_l225_225046

theorem find_value (m n : ℤ) (h : 2 * m + n - 2 = 0) : 2 * m + n + 1 = 3 :=
by { sorry }

end find_value_l225_225046


namespace tutors_smallest_common_students_l225_225974

theorem tutors_smallest_common_students:
  let zack_group := 14
  let karen_group := 10
  let julie_group := 15
  ∃ n : ℕ, n = Nat.lcm (Nat.lcm zack_group karen_group) julie_group ∧ n = 210 :=
begin
  sorry
end

end tutors_smallest_common_students_l225_225974


namespace right_triangle_area_l225_225109

theorem right_triangle_area (PQ QR : ℝ) (hPQ : PQ = 10) (hQR : QR = 26) (hRPQ : ∠RPQ = 90) :
  ∃ PR, PR = sqrt (QR^2 - PQ^2) ∧ (1/2) * PQ * PR = 120 :=
by {
  sorry
}

end right_triangle_area_l225_225109


namespace evaluate_statements_l225_225244

theorem evaluate_statements : ℕ :=
by
  let statement1 := ∀ x : ℝ, (x = 2 → ¬((x^2 - 4) / (x - 2) = 0)) ∧ (x = -2 → ((x^2 - 4) / (x - 2) = 0))
  let statement2 := ∀ a : ℝ, ((a + 1)^0 = 1) → (a ≠ 0)
  let statement3 := ∀ x : ℝ, 
    (x + (1/2) ≥ 0) ∧ (x ≠ 0) → x ≥ - (1/2)
  let statement4 := ∀ m : ℝ, (m > 0) → 
    (true -- Mathlib does not inherently support graph visualization. Assuming the statement defines the given conditions in some coherent visual representation)
  let correct_statement1 := ¬ statement1
  let correct_statement2 := ¬ statement2
  let correct_statement3 := ¬ statement3
  let correct_statement4 := statement4
  let correct_count := [correct_statement1, correct_statement2, correct_statement3, correct_statement4].count(True)
  exact correct_count

end evaluate_statements_l225_225244


namespace holes_in_compartment_l225_225988

theorem holes_in_compartment :
  ∀ (rect : Type) (holes : ℕ) (compartments : ℕ),
  compartments = 9 →
  holes = 20 →
  (∃ (compartment : rect ) (n : ℕ), n ≥ 3) :=
by
  intros rect holes compartments h_compartments h_holes
  sorry

end holes_in_compartment_l225_225988


namespace find_integer_mod_l225_225010

theorem find_integer_mod (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 14) (h3 : n ≡ 15478 [MOD 15]) : n = 13 :=
sorry

end find_integer_mod_l225_225010


namespace second_option_feasible_l225_225710

def Individual : Type := String
def M : Individual := "M"
def I : Individual := "I"
def P : Individual := "P"
def A : Individual := "A"

variable (is_sitting : Individual → Prop)

-- Given conditions
axiom fact1 : ¬ is_sitting M
axiom fact2 : ¬ is_sitting A
axiom fact3 : ¬ is_sitting M → is_sitting I
axiom fact4 : is_sitting I → is_sitting P

theorem second_option_feasible :
  is_sitting I ∧ is_sitting P ∧ ¬ is_sitting M ∧ ¬ is_sitting A :=
by
  sorry

end second_option_feasible_l225_225710


namespace fraction_simplification_l225_225329

theorem fraction_simplification :
  (2/5 + 3/4) / (4/9 + 1/6) = (207/110) := by
  sorry

end fraction_simplification_l225_225329


namespace target_average_income_l225_225656

variable (past_incomes : List ℕ) (next_average : ℕ)

def total_past_income := past_incomes.sum
def total_next_income := next_average * 5
def total_ten_week_income := total_past_income past_incomes + total_next_income next_average

theorem target_average_income (h1 : past_incomes = [406, 413, 420, 436, 395])
                              (h2 : next_average = 586) :
  total_ten_week_income past_incomes next_average / 10 = 500 := by
  sorry

end target_average_income_l225_225656


namespace bread_consumption_l225_225946

-- Definitions using conditions
def members := 4
def slices_snacks := 2
def slices_per_loaf := 12
def total_loaves := 5
def total_days := 3

-- The main theorem to prove
theorem bread_consumption :
  (3 * members * (B + slices_snacks) = total_loaves * slices_per_loaf) → B = 3 :=
by
  intro h
  sorry

end bread_consumption_l225_225946


namespace problem_solution_l225_225494

def point_equidistant (P A B C D : EclideanSpace ℝ (Fin 3)) : Prop :=
  (dist P A = dist P B) ∧ (dist P B = dist P C) ∧ (dist P C = dist P D)

def P := (5 : ℝ, -3 : ℝ, 4 : ℝ)
def A := (10 : ℝ, 0 : ℝ, 0 : ℝ)
def B := (0 : ℝ, -6 : ℝ, 0 : ℝ)
def C := (0 : ℝ, 0 : ℝ, 8 : ℝ)
def D := (0 : ℝ, 0 : ℝ, 0 : ℝ)

theorem problem_solution :
  ∃ P : EuclideanSpace ℝ (Fin 3), point_equidistant P A B C D ∧ P = (5, -3, 4) :=
by
  use (5, -3, 4) -- specifying the point P we obtained from the solution
  unfold point_equidistant -- ensuring the definition is expanded for equality checks
  sorry -- skipping the proof

end problem_solution_l225_225494


namespace solve_arithmetic_seq_l225_225107

-- Define the arithmetic sequence and sum of terms
def arithmetic_sequence (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
def sum_arithmetic_sequence (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Main theorem to prove
theorem solve_arithmetic_seq :
  let a1 := -2015
  let S : ℕ → ℤ := sum_arithmetic_sequence a1
  (2 * (S 6) - 3 * (S 4) = 24) →
  S 2015 = -2015 :=
by
  sorry

end solve_arithmetic_seq_l225_225107


namespace quadratic_difference_l225_225015

noncomputable def p : ℕ := 81
noncomputable def q : ℕ := 2
noncomputable def equation := 2 * x^2 - 5 * x - 7 = 0

theorem quadratic_difference (h : ∀ x : ℝ, equation):
  p + q = 83 :=
by
  sorry

end quadratic_difference_l225_225015


namespace sqrt_sum_abs_eq_l225_225344

theorem sqrt_sum_abs_eq (x : ℝ) :
    (Real.sqrt (x^2 + 6 * x + 9) + Real.sqrt (x^2 - 6 * x + 9)) = (|x - 3| + |x + 3|) := 
by 
  sorry

end sqrt_sum_abs_eq_l225_225344


namespace vector_subtraction_l225_225333

theorem vector_subtraction :
  let v1 := (3 : ℤ, -8 : ℤ)
  let v2 := (2 : ℤ, -6 : ℤ)
  let s := 5 : ℤ
  let v_result := (v1.fst - s * v2.fst, v1.snd - s * v2.snd)
  v_result = (-7 : ℤ, 22 : ℤ) :=
by
  let v1 := (3 : ℤ, -8 : ℤ)
  let v2 := (2 : ℤ, -6 : ℤ)
  let s := 5 : ℤ
  let v_result := (v1.fst - s * v2.fst, v1.snd - s * v2.snd)
  show v_result = (-7 : ℤ, 22 : ℤ)
  sorry

end vector_subtraction_l225_225333


namespace parabola_eq_proof_max_area_proof_l225_225781

-- Given conditions
def focus : (ℝ × ℝ) := (0, 1)
def parabola_eq (x y p : ℝ) : Prop := x^2 = 2 * p * y

-- Problems
noncomputable def question1 (p : ℝ) : Prop := p = 2 → parabola_eq x y p ↔ x^2 = 4y

noncomputable def A (x₁ : ℝ) : ℝ × ℝ := (x₁, x₁^2 / 4)
noncomputable def B (x₂ : ℝ) : ℝ × ℝ := (x₂, x₂^2 / 4)
noncomputable def C (x₃ : ℝ) : ℝ × ℝ := (x₃, x₃^2 / 4)

noncomputable def FA (x₁ : ℝ) : (ℝ × ℝ) := (x₁, x₁^2 / 4 - 1)
noncomputable def FB (x₂ : ℝ) : (ℝ × ℝ) := (x₂, x₂^2 / 4 - 1)
noncomputable def FC (x₃ : ℝ) : (ℝ × ℝ) := (x₃, x₃^2 / 4 - 1)

noncomputable def vector_sum_zero (x₁ x₂ x₃ : ℝ) : Prop := (λ (fa fb fc : (ℝ × ℝ)) → fa + fb + fc = (0, 0)) (FA x₁) (FB x₂) (FC x₃)

noncomputable def max_area (S : ℝ) : Prop :=
  ∀ x₁ x₂ x₃, vector_sum_zero x₁ x₂ x₃ → S = (3 / 2) * (Real.sqrt 6) / 2

-- Prove the parabola equation and max area of triangle ABC
theorem parabola_eq_proof : ∀ p, question1 p :=
by sorry

theorem max_area_proof : ∃ S, max_area S :=
by sorry

end parabola_eq_proof_max_area_proof_l225_225781


namespace x_squared_minus_y_squared_l225_225803

theorem x_squared_minus_y_squared (x y : ℚ) (h1 : x + y = 2 / 5) (h2 : x - y = 1 / 10) : x ^ 2 - y ^ 2 = 1 / 25 :=
by
  sorry

end x_squared_minus_y_squared_l225_225803


namespace point_below_parabola_l225_225194

theorem point_below_parabola (a b c : ℝ) (h : 2 < a + b + c) : 
  2 < c + b + a :=
by
  sorry

end point_below_parabola_l225_225194


namespace find_multiple_l225_225646

theorem find_multiple (x m : ℕ) (h₁ : x = 69) (h₂ : x - 18 = m * (86 - x)) : m = 3 :=
by
  sorry

end find_multiple_l225_225646


namespace find_x_for_geometric_series_l225_225683

theorem find_x_for_geometric_series :
  let S1 := (∑' n : ℕ, (1 / 3) ^ n)
  let S2 := (∑' n : ℕ, (-1 / 3) ^ n)
  let S3 (x : ℕ) := (∑' n : ℕ, (1 / (x : ℝ)) ^ n)
  S1 = 3 / 2 → S2 = 3 / 4 → (∃ x : ℝ, S1 * S2 = S3 x ∧ x = 9) := by
  intro S1_eq S2_eq
  let S1 := (∑' n : ℕ, (1 / 3) ^ n)
  let S2 := (∑' n : ℕ, (-1 / 3) ^ n)
  let S3 (x : ℕ) := (∑' n : ℕ, (1 / (x : ℝ)) ^ n)
  have h := (3 / 2) * (3 / 4) = 9 / 8
  sorry

end find_x_for_geometric_series_l225_225683


namespace added_water_proof_l225_225098

variable (total_volume : ℕ) (milk_ratio water_ratio : ℕ) (added_water : ℕ)

theorem added_water_proof 
  (h1 : total_volume = 45) 
  (h2 : milk_ratio = 4) 
  (h3 : water_ratio = 1) 
  (h4 : added_water = 3) 
  (milk_volume : ℕ)
  (water_volume : ℕ)
  (h5 : milk_volume = (milk_ratio * total_volume) / (milk_ratio + water_ratio))
  (h6 : water_volume = (water_ratio * total_volume) / (milk_ratio + water_ratio))
  (new_ratio : ℕ)
  (h7 : new_ratio = milk_volume / (water_volume + added_water)) : added_water = 3 :=
by
  sorry

end added_water_proof_l225_225098


namespace area_of_triangle_ABC_l225_225623

-- Definitions for the problem's conditions
def square_side : ℝ := 12
def point_A : ℝ × ℝ := (square_side / 2, square_side)
def point_C : ℝ × ℝ := (square_side, square_side / 2)
def point_B : ℝ × ℝ := (0, 3 * square_side / 4)

-- The statement to prove
theorem area_of_triangle_ABC : 
  let base : ℝ := Real.sqrt ((point_C.1 - point_A.1)^2 + (point_C.2 - point_A.2)^2)
  let height : ℝ := abs ((-1 * (point_B.1 - point_A.1) + point_A.2 - point_B.2) / Real.sqrt (1^2 + 1))
  let area : ℝ := (1 / 2) * base * height
  base = 3 * Real.sqrt 5 ∧ height = (9 * Real.sqrt 2) / 2 ∧ area = 27 * Real.sqrt 10 / 4 :=
begin
  sorry
end

end area_of_triangle_ABC_l225_225623


namespace employee_pays_correct_amount_l225_225978

def wholesale_cost : ℝ := 200
def markup_percentage : ℝ := 0.20
def discount_percentage : ℝ := 0.10

def retail_price (wholesale: ℝ) (markup_percentage: ℝ) : ℝ :=
  wholesale * (1 + markup_percentage)

def discount_amount (price: ℝ) (discount_percentage: ℝ) : ℝ :=
  price * discount_percentage

def final_price (retail: ℝ) (discount: ℝ) : ℝ :=
  retail - discount

theorem employee_pays_correct_amount : final_price (retail_price wholesale_cost markup_percentage) 
                                                     (discount_amount (retail_price wholesale_cost markup_percentage) discount_percentage) = 216 := 
by
  sorry

end employee_pays_correct_amount_l225_225978


namespace part_a_l225_225255

theorem part_a (n : ℕ) :
  1 + ∑ k in Finset.range (n / 3 + 1), Nat.choose n (3 * k) = (1 / 3) * (Real.exp (n * Real.log 2) + 2 * Real.cos (n * Real.pi / 3)) :=
sorry

end part_a_l225_225255


namespace find_theta_count_l225_225316

-- Used conditions from the original problem
def interval : Set ℝ := {θ | 0 < θ ∧ θ ≤ 2 * Real.pi}
def equation (θ : ℝ) : Prop := 2 + 4 * Real.sin (2 * θ) - 3 * Real.cos (4 * θ) = 0

theorem find_theta_count : (Set.count {θ | θ ∈ interval ∧ equation θ} = 8) := 
sorry

end find_theta_count_l225_225316


namespace least_positive_integer_fac_6370_factorial_l225_225230

theorem least_positive_integer_fac_6370_factorial :
  ∃ (n : ℕ), (∀ m : ℕ, (6370 ∣ m.factorial) → m ≥ n) ∧ n = 14 :=
by
  sorry

end least_positive_integer_fac_6370_factorial_l225_225230


namespace find_monic_quadratic_real_coefficients_l225_225736

-- Define the necessary conditions
def is_monic_quadratic (f : ℝ[X]) : Prop :=
  f.degree = Polynomial.degree (Polynomial.X ^ 2) ∧ f.leadingCoeff = 1

def has_real_coefficients (f : ℝ[X]) : Prop :=
  ∀ n, (Polynomial.coeff f n).im = 0

def root_exists (f : ℝ[X]) (z : ℂ) : Prop :=
  f.eval z = 0

-- The main theorem statement
theorem find_monic_quadratic_real_coefficients (z : ℂ) (f : ℝ[X]) (h : z = 2 - 3 * I) :
  is_monic_quadratic f ∧ has_real_coefficients f ∧ root_exists f z ∧ root_exists f z.conj ↔ f = Polynomial.C 13 + Polynomial.X * (Polynomial.X - 4) :=
sorry

end find_monic_quadratic_real_coefficients_l225_225736


namespace combination_lock_opens_l225_225237

theorem combination_lock_opens (f : ℕ → ℕ) :
  ∃ (i j : ℕ), (1 ≤ i ∧ i ≤ j ∧ j ≤ 26) ∧ (∑ k in finset.Ico i j, f k) % 26 = 0 :=
by
  have h : ∀ n, S n = (∑ k in finset.range n, f k) % 26
  sorry

end combination_lock_opens_l225_225237


namespace average_value_is_2020_l225_225633

namespace CardsAverage

theorem average_value_is_2020 (n : ℕ) (h : (2020 * 3 * ((n * (n + 1)) + 2) = n * (n + 1) * (2 * n + 1) + 6 * (n + 1))) : n = 3015 := 
by
  sorry

end CardsAverage

end average_value_is_2020_l225_225633


namespace max_valid_subset_l225_225611

def is_not_square (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ≠ n

def valid_subset (M : Finset ℕ) : Prop :=
  M ⊆ {1, 2, 3, ..., 15} ∧ 
  ∀ (a b c : ℕ), a ≠ b → b ≠ c → c ≠ a → a ∈ M → b ∈ M → c ∈ M → is_not_square (a * b * c)

theorem max_valid_subset :
  ∃ (M : Finset ℕ), valid_subset M ∧ M.card = 10 :=
  sorry

end max_valid_subset_l225_225611


namespace cos_A_in_quadrilateral_l225_225450

theorem cos_A_in_quadrilateral
  (AB CD : ℝ) (x y : ℝ)
  (α : ℝ) -- angles A and C
  (h1 : AB = 200) (h2 : CD = 200) -- Given sides
  (h3 : x + y = 320)              -- Perimeter condition leading to this
  (h4 : AD ≠ BC)                  -- Given AD ≠ BC
  (h5 : ∠A = α) (h6 : ∠C = α)    -- Angles A and C are equal
  : cos(α) = 4 / 5 :=
by
  sorry

end cos_A_in_quadrilateral_l225_225450


namespace rosie_laps_l225_225715

theorem rosie_laps (lou_distance : ℝ) (track_length : ℝ) (lou_speed_factor : ℝ) (rosie_speed_multiplier : ℝ) 
    (number_of_laps_by_lou : ℝ) (number_of_laps_by_rosie : ℕ) :
  lou_distance = 3 ∧ 
  track_length = 1 / 4 ∧ 
  lou_speed_factor = 0.75 ∧ 
  rosie_speed_multiplier = 2 ∧ 
  number_of_laps_by_lou = lou_distance / track_length ∧ 
  number_of_laps_by_rosie = rosie_speed_multiplier * number_of_laps_by_lou → 
  number_of_laps_by_rosie = 18 := 
sorry

end rosie_laps_l225_225715


namespace first_pipe_fill_time_l225_225578

theorem first_pipe_fill_time (T : ℝ) : 
  let first_pipe_rate := 1 / T
  let second_pipe_rate := 1 / 20
  let drain_pipe_rate := 45 / 675
  let combined_rate := 1 / 15
  (first_pipe_rate + second_pipe_rate - drain_pipe_rate = combined_rate) →
  T = 4 :=
by 
sory

end first_pipe_fill_time_l225_225578


namespace find_y_intercept_l225_225724

-- Conditions
def line_equation (x y : ℝ) : Prop := 4 * x + 7 * y - 3 * x * y = 28

-- Statement (Proof Problem)
theorem find_y_intercept : ∃ y : ℝ, line_equation 0 y ∧ (0, y) = (0, 4) := by
  sorry

end find_y_intercept_l225_225724


namespace lina_next_birthday_on_monday_l225_225885

-- Define conditions
def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def day_of_week_on_17_jan (year : ℕ) : ℕ :=
  if is_leap_year year then 
    (5 + (year - 2020) + ((year - 2020) / 4 - (year - 2020) / 100 + (year - 2020) / 400) * 1) % 7 
  else 
    (5 + (1 * (year - 2020)) + ((year - 2020) / 4 - (year - 2020) / 100 + (year - 2020) / 400) * 2) % 7

-- Prove the year when January 17th falls on a Monday
theorem lina_next_birthday_on_monday: ∃ year : ℕ, year > 2020 ∧ day_of_week_on_17_jan year = 1 := by
  existsi 2023
  sorry

end lina_next_birthday_on_monday_l225_225885


namespace problem_solution_l225_225497

def point_equidistant (P A B C D : EclideanSpace ℝ (Fin 3)) : Prop :=
  (dist P A = dist P B) ∧ (dist P B = dist P C) ∧ (dist P C = dist P D)

def P := (5 : ℝ, -3 : ℝ, 4 : ℝ)
def A := (10 : ℝ, 0 : ℝ, 0 : ℝ)
def B := (0 : ℝ, -6 : ℝ, 0 : ℝ)
def C := (0 : ℝ, 0 : ℝ, 8 : ℝ)
def D := (0 : ℝ, 0 : ℝ, 0 : ℝ)

theorem problem_solution :
  ∃ P : EuclideanSpace ℝ (Fin 3), point_equidistant P A B C D ∧ P = (5, -3, 4) :=
by
  use (5, -3, 4) -- specifying the point P we obtained from the solution
  unfold point_equidistant -- ensuring the definition is expanded for equality checks
  sorry -- skipping the proof

end problem_solution_l225_225497


namespace distance_from_sphere_center_to_triangle_plane_l225_225667

theorem distance_from_sphere_center_to_triangle_plane (a b c R r d : ℝ) 
  (h_a : a = 13) (h_b : b = 14) (h_c : c = 15) (h_R : R = 10) (h_r : r = 4) :
  d = 2 * real.sqrt 21 → real.sqrt(R^2 - r^2) = d := by 
sorry

end distance_from_sphere_center_to_triangle_plane_l225_225667


namespace average_salary_of_feb_mar_apr_may_l225_225545

theorem average_salary_of_feb_mar_apr_may
  (avg_salary_jan_feb_mar_apr : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_salary_feb_mar_apr : ℝ)
  (total_salary_feb_mar_apr_may: ℝ)
  (n_months: ℝ): 
  avg_salary_jan_feb_mar_apr = 8000 ∧ 
  salary_jan = 6100 ∧ 
  salary_may = 6500 ∧ 
  total_salary_feb_mar_apr = (avg_salary_jan_feb_mar_apr * 4 - salary_jan) ∧
  total_salary_feb_mar_apr_may = (total_salary_feb_mar_apr + salary_may) ∧
  n_months = (total_salary_feb_mar_apr_may / 8100) →
  n_months = 4 :=
by
  intros 
  sorry

end average_salary_of_feb_mar_apr_may_l225_225545


namespace imaginary_part_of_conjugate_l225_225381

def imaginary_part (z : ℂ) : ℝ := z.im

def conj (z : ℂ) : ℂ := complex.conj z

theorem imaginary_part_of_conjugate :
  ∀ (z : ℂ), z = -1/3 + (2 * real.sqrt 2) / 3 * complex.I → 
  imaginary_part (conj z) = - (2 * real.sqrt 2) / 3 :=
by
  intros z h_z
  rw h_z
  sorry

end imaginary_part_of_conjugate_l225_225381


namespace range_of_sum_x_l225_225386

-- Define the points and their properties
variables {x1 x2 x3 y : ℝ}

-- Conditions based on the problem
def point_A_on_line := y = 3 * x1 + 19
def points_B_C_on_parabola := (y = x2^2 + 4 * x2 - 1) ∧ (y = x3^2 + 4 * x3 - 1)
def equal_y_coordinates := y = y ∧ y = y ∧ y = y
def ordered_x := x1 < x2 ∧ x2 < x3

-- The statement which needs to be proved
theorem range_of_sum_x: 
  point_A_on_line ∧ points_B_C_on_parabola ∧ equal_y_coordinates ∧ ordered_x 
  → -12 < x1 + x2 + x3 ∧ x1 + x2 + x3 < -9 :=
by sorry

end range_of_sum_x_l225_225386


namespace sum_of_perimeters_of_triangle_ACD_l225_225148

theorem sum_of_perimeters_of_triangle_ACD :
  ∃ (s : ℝ), s = 36 + 12 * real.sqrt 3 ∧
  ∃ (A B C D : ℝ×ℝ) (AB BC AD BD CD : ℝ),
  (AB = 12) ∧ (BC = 16) ∧ (AD = 2 * CD) ∧ 
  (AD % 1 = 0) ∧ (BD % 1 = 0) ∧ 
  s = AD + CD + BD := by
  sorry

end sum_of_perimeters_of_triangle_ACD_l225_225148


namespace b_is_geometric_sequence_product_b_1_to_b_20_max_product_b_l225_225045

variable {a : ℕ → ℤ} (b : ℕ → ℝ := λ n, 3^(a n))

-- (1) Prove that the sequence {b_n} is a geometric sequence
theorem b_is_geometric_sequence (d : ℤ) (h_arithmetic : ∀ n, a (n + 1) = a n + d) : 
  ∀ n ≥ 1, b (n + 1) / b n = 3^d :=
by sorry

-- (2) Find the product b_1 * b_2 * ... * b_20 given a_8 + a_{13} = m
theorem product_b_1_to_b_20 (m : ℤ) (h_sum : a 8 + a 13 = m) : 
  ∏ n in Finset.range 20, b (n + 1) = 3^(10 * m) :=
by sorry

-- (3) Prove maximum value of product b_1 * b_2 * ... * b_n is 3^(75/2) given certain conditions
theorem max_product_b (h_prod : b 3 * b 5 = 3^9) (h_sum_4_6 : a 4 + a 6 = 3) :
  ∃ n, ∏ k in Finset.range n, b (k + 1) = 3^(75 / 2) :=
by sorry

end b_is_geometric_sequence_product_b_1_to_b_20_max_product_b_l225_225045


namespace cone_volume_l225_225440

-- Definitions used in conditions
def lateral_area (r : ℝ) (l : ℝ) : ℝ := π * r * l
def volume (r : ℝ) (h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- Given constants
def SA : ℝ := 9 * Real.sqrt 2 * π  -- Lateral surface area is given as 9√2π
def θ : ℝ := π / 4                -- Angle between slant height and base is given as π/4

theorem cone_volume : 
  ∃ (r : ℝ), (∀ (r l h : ℝ), (l = Real.sqrt 2 * r ∧ h = r ∧ lateral_area r l = SA) → volume r h = 9 * π) :=
by
  sorry

end cone_volume_l225_225440


namespace a_n_formula_l225_225126

noncomputable def sqrt3 := Real.sqrt 3

def a (n : ℕ) : ℕ :=
  if n = 1 then 3
  else if n = 2 then 8
  else 2 * a (n-1) + 2 * a (n-2)

theorem a_n_formula (n : ℕ) :
  a n = (Real.floor $ (1/2 + 1/sqrt3) * (sqrt3 + 1)^n + 0.5) := sorry

end a_n_formula_l225_225126


namespace solve_square_problem_l225_225023

-- Define the initial structure made up of 12 sticks arranged to form squares.
def initial_structure : Set Stick := 
  {s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉, s₁₀, s₁₁, s₁₂}

-- Define the condition that removing 2 sticks results in 2 squares.
def remove_2_sticks_results_in_2_squares (s1 s2 : Stick) : Prop :=
  (s1 ∈ initial_structure) ∧ (s2 ∈ initial_structure) ∧
  let remaining_sticks := initial_structure \ {s1, s2} in
  exactly_two_squares remaining_sticks

-- State the theorem to be proven.
theorem solve_square_problem : ∃ (s1 s2 : Stick), remove_2_sticks_results_in_2_squares s1 s2 :=
  by sorry

end solve_square_problem_l225_225023


namespace diagonal_ratio_five_sided_polygon_l225_225192

theorem diagonal_ratio_five_sided_polygon :
  ∀ (n : ℕ), n = 5 → (n * (n - 3) / 2) / n = 1 := by
  intro n
  simp
  sorry

end diagonal_ratio_five_sided_polygon_l225_225192


namespace range_of_m_l225_225790

noncomputable def f : ℝ → ℝ :=
λ x, if x < 0 then Real.log (-x) else -Real.log x

theorem range_of_m (m : ℝ) : (f m > f (-m)) ↔ (m < -1 ∨ (0 < m ∧ m < 1)) :=
by
  -- Proof would go here
  sorry

end range_of_m_l225_225790


namespace centers_of_squares_form_square_l225_225899

-- Definitions for the conditions
variables (A B C D P Q R S : Type) [EuclideanGeometry]
variables (AB BC CD DA : Segment)
variables (SquaresConstructedExternally : ∀ (X Y : Point), Segment X Y → Square)
variables (CentersFormSquare : ∀ (P Q R S : Point), IsSquare P Q R S)

-- Conditions
axiom parallelogram_ABCD : Parallelogram A B C D
axiom center_P : IsCenter P (SquaresConstructedExternally A B AB)
axiom center_Q : IsCenter Q (SquaresConstructedExternally B C BC)
axiom center_R : IsCenter R (SquaresConstructedExternally C D CD)
axiom center_S : IsCenter S (SquaresConstructedExternally D A DA)

-- The statement to prove
theorem centers_of_squares_form_square :
  CentersFormSquare P Q R S := 
  sorry

end centers_of_squares_form_square_l225_225899


namespace squirrel_acorns_l225_225661

theorem squirrel_acorns :
  ∀ (total_acorns : ℕ)
    (first_month_percent second_month_percent third_month_percent : ℝ)
    (first_month_consumed second_month_consumed third_month_consumed : ℝ),
    total_acorns = 500 →
    first_month_percent = 0.40 →
    second_month_percent = 0.30 →
    third_month_percent = 0.30 →
    first_month_consumed = 0.20 →
    second_month_consumed = 0.25 →
    third_month_consumed = 0.15 →
    let first_month_acorns := total_acorns * first_month_percent
    let second_month_acorns := total_acorns * second_month_percent
    let third_month_acorns := total_acorns * third_month_percent
    let remaining_first_month := first_month_acorns - (first_month_consumed * first_month_acorns)
    let remaining_second_month := second_month_acorns - (second_month_consumed * second_month_acorns)
    let remaining_third_month := third_month_acorns - (third_month_consumed * third_month_acorns)
    remaining_first_month + remaining_second_month + remaining_third_month = 400 := 
by
  intros 
    total_acorns
    first_month_percent second_month_percent third_month_percent
    first_month_consumed second_month_consumed third_month_consumed
    h_total
    h_first_percent
    h_second_percent
    h_third_percent
    h_first_consumed
    h_second_consumed
    h_third_consumed
  let first_month_acorns := total_acorns * first_month_percent
  let second_month_acorns := total_acorns * second_month_percent
  let third_month_acorns := total_acorns * third_month_percent
  let remaining_first_month := first_month_acorns - (first_month_consumed * first_month_acorns)
  let remaining_second_month := second_month_acorns - (second_month_consumed * second_month_acorns)
  let remaining_third_month := third_month_acorns - (third_month_consumed * third_month_acorns)
  sorry

end squirrel_acorns_l225_225661


namespace impossible_coloring_l225_225610

-- Conditions: 200 points, with pairwise connections
def num_points : ℕ := 200
def hues : ℕ := 7
def hues' : ℕ := 10

-- Define segments and their properties
axiom segments : set (ℕ × ℕ)
axiom segments_no_intersect : ∀ s1 s2 ∈ segments, s1 ≠ s2 → ¬∃ p, p ∈ s1 ∧ p ∈ s2

-- Colors definition
def color (x : ℕ) : ℕ := sorry -- This is a placeholder as actual coloring function

-- Main theorem statements
theorem impossible_coloring (K : ℕ) (hK : K = hues ∨ K = hues') :
  ¬ ∃ (coloring : ℕ → ℕ), (∀ p < num_points, coloring p < K) ∧ 
  (∀ p1 p2, p1 < num_points → p2 < num_points → (p1, p2) ∈ segments → coloring p1 ≠ coloring p2) :=
begin
  sorry,
end

end impossible_coloring_l225_225610


namespace find_point_P_l225_225492

/-- The point P such that AP = BP = CP = DP for given points A, B, C, D in 3D space -/
theorem find_point_P :
  ∃ P : ℝ × ℝ × ℝ, 
    let A := (10, 0, 0 : ℝ × ℝ × ℝ),
        B := (0, -6, 0 : ℝ × ℝ × ℝ),
        C := (0, 0, 8 : ℝ × ℝ × ℝ),
        D := (0, 0, 0 : ℝ × ℝ × ℝ) in
      dist P A = dist P B ∧
      dist P B = dist P C ∧
      dist P C = dist P D ∧
      P = (5, -3, 4) :=
by
  sorry

end find_point_P_l225_225492


namespace least_difference_l225_225816

noncomputable def t1 := 6
noncomputable def t2 := 12
noncomputable def t3 := 18
noncomputable def t4 := 24
noncomputable def y := 23
noncomputable def z := 25

theorem least_difference :
  (∀ x : ℕ, x ∈ {t1, t2, t3, t4} → (∃ y z : ℕ,
    y = 23 ∧
    z = 25 ∧
    x < y ∧ y < z ∧ y - x > 5 ∧ x % 2 = 0 ∧ x % 3 = 0 ∧ y % 2 = 1 ∧ z % 2 = 1 ∧ prime y ∧ 20 < y ∧ z % 5 = 0 ∧ 1 < x ∧ x < 30) → z - x = 19) :=
sorry

end least_difference_l225_225816


namespace correct_propositions_l225_225049
noncomputable def proposition1 : Prop :=
  ∀ (A : Set ℕ) (freq_A : ℕ → ℕ) (prob_A : ℕ → ℝ), 
  (∀ n, freq_A n / n ≈ prob_A n) → True

noncomputable def proposition2 : Prop :=
  ∀ (x : ℕ → ℝ) (n : ℕ), 
  n ≥ 2 → 
  let S := Real.sqrt ((∑ i in Finset.range n, (x i - (∑ j in Finset.range n, x j) / n)^2) / (n - 1))
  in True

noncomputable def proposition3 : Prop :=
  ∀ (population : Set ℕ) (sample : Set ℕ), 
  (∀ x, x ∈ population → x ∈ sample) → True

noncomputable def proposition4 : Prop :=
  ∀ (population : Set ℕ) (parts : List (Set ℕ)) (sample : Set ℕ), 
  (∀ p in parts, ∃ x, x ∈ p ∧ x ∈ sample) → False

theorem correct_propositions : proposition1 ∧ proposition2 ∧ proposition3 ∧ ¬proposition4 :=
by
  split
  all_goals { sorry }

end correct_propositions_l225_225049


namespace eval_expr_l225_225804

theorem eval_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  (x^(2 * y) * y^(3 * x) / (y^(2 * y) * x^(3 * x))) = x^(2 * y - 3 * x) * y^(3 * x - 2 * y) :=
by
  sorry

end eval_expr_l225_225804


namespace total_students_in_class_is_15_l225_225067

noncomputable def choose (n k : ℕ) : ℕ := sorry -- Define a function for combinations
noncomputable def permute (n k : ℕ) : ℕ := sorry -- Define a function for permutations

variables (x m n : ℕ) (hx : choose x 4 = m) (hn : permute x 2 = n) (hratio : m * 2 = n * 13)

theorem total_students_in_class_is_15 : x = 15 :=
sorry

end total_students_in_class_is_15_l225_225067


namespace locus_centers_of_circles_l225_225553

theorem locus_centers_of_circles (P : ℝ × ℝ) (a : ℝ) (a_pos : 0 < a):
  {O : ℝ × ℝ | dist O P = a} = {O : ℝ × ℝ | dist O P = a} :=
by
  sorry

end locus_centers_of_circles_l225_225553


namespace range_of_a_l225_225055

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

noncomputable def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) :
  (a > -2 ∧ a < -1) ∨ (a ≥ 1) :=
by
  sorry

end range_of_a_l225_225055


namespace intersection_empty_implies_range_l225_225757

-- Define the sets A and B
def setA := {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 3}
def setB (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 1}

-- Prove that if A ∩ B = ∅, then 1 < a < 2
theorem intersection_empty_implies_range (a : ℝ) (h : setA ∩ setB a = ∅) : 1 < a ∧ a < 2 :=
by
  sorry

end intersection_empty_implies_range_l225_225757


namespace minimum_omega_l225_225811

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x) = f (-x)

noncomputable def shifted_function (ω : ℝ) : (ℝ → ℝ) :=
  λ x : ℝ, abs (sin (ω * x + (ω * π / 12 - π / 8)))

theorem minimum_omega
  (ω : ℝ)
  (h_shift : ω > 0)
  (h_even : is_even (shifted_function ω)) :
  ω = 3 / 2 :=
by
  sorry

end minimum_omega_l225_225811


namespace trajectory_of_P_below_x_axis_l225_225645

theorem trajectory_of_P_below_x_axis (x y : ℝ) (P_below_x_axis : y < 0)
    (tangent_to_parabola : ∃ A B: ℝ × ℝ, A.1^2 = 2 * A.2 ∧ B.1^2 = 2 * B.2 ∧ (x^2 + y^2 = 1))
    (AB_tangent_to_circle : ∀ (x0 y0 : ℝ), x0^2 + y0^2 = 1 → x0 * x + y0 * y = 1) :
    y^2 - x^2 = 1 :=
sorry

end trajectory_of_P_below_x_axis_l225_225645


namespace geometric_sequence_a1_a13_l225_225099

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a1_a13 (a : ℕ → ℝ)
    (h_geo : geometric_sequence a)
    (h_log_sum : log (a 4) + log (a 7) + log (a 10) = 3) :
    a 1 * a 13 = 100 :=
sorry

end geometric_sequence_a1_a13_l225_225099


namespace total_number_of_numbers_l225_225544

theorem total_number_of_numbers (avg : ℝ) (sum1 sum2 sum3 : ℝ) (N : ℝ) :
  avg = 3.95 →
  sum1 = 2 * 3.8 →
  sum2 = 2 * 3.85 →
  sum3 = 2 * 4.200000000000001 →
  avg = (sum1 + sum2 + sum3) / N →
  N = 6 :=
by
  intros h_avg h_sum1 h_sum2 h_sum3 h_total
  sorry

end total_number_of_numbers_l225_225544


namespace pentagon_arithmetic_progression_angle_l225_225923

theorem pentagon_arithmetic_progression_angle (a n : ℝ) 
  (h1 : a + (a + n) + (a + 2 * n) + (a + 3 * n) + (a + 4 * n) = 540) :
  a + 2 * n = 108 :=
by
  sorry

end pentagon_arithmetic_progression_angle_l225_225923


namespace find_k_value_l225_225038

noncomputable def k_value (a b : ℝ) (k : ℝ) : Prop :=
  let magnitude_a := 4
  let magnitude_b := 3
  (a + k * b) • (a - k * b) = 0

theorem find_k_value (a b : ℝ) (k : ℝ) (ha : ∥a∥ = 4) (hb : ∥b∥ = 3)
  (h_perp : (a + k * b) • (a - k * b) = 0) : k = 4 / 3 ∨ k = -4 / 3 := sorry

end find_k_value_l225_225038


namespace polynomial_identity_l225_225167

theorem polynomial_identity (x y : ℝ) (h : x - y = 1) : 
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 := 
  sorry

end polynomial_identity_l225_225167


namespace light_distance_50_years_l225_225929

theorem light_distance_50_years :
  let light_distance_one_year := 6 * 10^12 in
  let light_distance_50_years := light_distance_one_year * 50 in
  light_distance_50_years = 3 * 10^14 :=
by
  let light_distance_one_year := 6 * 10^12
  let light_distance_50_years := light_distance_one_year * 50
  calc
    light_distance_50_years
    _ = 6 * 10^12 * 50 : by rw [←light_distance_one_year]
    _ = 3 * 10^14 : by norm_num

end light_distance_50_years_l225_225929


namespace matrix_multiplication_is_correct_l225_225306

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -3], ![0, 2]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![15, -7], ![20, -16]]

theorem matrix_multiplication_is_correct : A ⬝ B = C :=
by
  sorry

end matrix_multiplication_is_correct_l225_225306


namespace julia_tag_kids_monday_l225_225854

-- Definitions based on conditions
def total_tag_kids (M T : ℕ) : Prop := M + T = 20
def tag_kids_Tuesday := 13

-- Problem statement
theorem julia_tag_kids_monday (M : ℕ) : total_tag_kids M tag_kids_Tuesday → M = 7 := 
by
  intro h
  sorry

end julia_tag_kids_monday_l225_225854


namespace fixed_point_of_d_prime_l225_225478

theorem fixed_point_of_d_prime 
  {A B C D M E F P Q : Type*}
  [IsTriangle A B C]
  [IsCentroid AD A B C]
  (d : Line)
  (hd : IsPerpendicular d AD)
  (hM : OnLine M d)
  (hE : E = Midpoint M B)
  (hF : F = Midpoint M C)
  (P : Point)
  (Q : Point)
  (hd_perp_E : IsPerpendicular (LineThrough E P) d)
  (hd_perp_F : IsPerpendicular (LineThrough F Q) d)
  (d' : Line)
  (hd_prime : IsPerpendicular d' (LineThrough P Q))
  (hM_prime : OnLine M d')
  : ∃ fixed_point : Point, fixed_point = ⟨0, m + (b * c) / a⟩ := 
sorry

end fixed_point_of_d_prime_l225_225478


namespace next_geometric_term_l225_225587

-- Define the terms of the geometric sequence
def term1 : ℕ → ℝ := λ n, 2 * (4 * x)^n

-- Define the specific terms we consider in the sequence
def t0 : ℝ := 2
def t1 : ℝ := 8 * x
def t2 : ℝ := 32 * x^2
def t3 : ℝ := 128 * x^3

-- Define the common ratio
def r : ℝ := 4 * x

-- Define the next term we need to prove
def t4 : ℝ := 512 * x^4

-- Theorem statement
theorem next_geometric_term :
  t3 * r = t4 := by
  sorry

end next_geometric_term_l225_225587


namespace minimum_candies_l225_225092

variables (c z : ℕ) (total_candies : ℕ)

def remaining_red_candies := (3 * c) / 5
def remaining_green_candies := (2 * z) / 5
def remaining_total_candies := remaining_red_candies + remaining_green_candies
def red_candies_fraction := remaining_red_candies * 8 = 3 * remaining_total_candies

theorem minimum_candies (h1 : 5 * c = 2 * z) (h2 : red_candies_fraction) :
  total_candies ≥ 35 := sorry

end minimum_candies_l225_225092


namespace answer_key_combinations_l225_225451

theorem answer_key_combinations : 
  (2^3 - 2) * 4^2 = 96 := 
by 
  -- Explanation about why it equals to this multi-step skipped, directly written as sorry.
  sorry

end answer_key_combinations_l225_225451


namespace portion_of_larger_jar_full_l225_225979

noncomputable def smaller_jar_capacity (S L : ℝ) : Prop :=
  (1 / 5) * S = (1 / 4) * L

noncomputable def larger_jar_capacity (L : ℝ) : ℝ :=
  (1 / 5) * (5 / 4) * L

theorem portion_of_larger_jar_full (S L : ℝ) 
  (h1 : smaller_jar_capacity S L) : 
  (1 / 4) * L + (1 / 4) * L = (1 / 2) * L := 
sorry

end portion_of_larger_jar_full_l225_225979


namespace modulus_of_conjugate_is_ten_l225_225783

-- Define the complex number z
def z : ℂ := (3 + complex.I) ^ 2

-- Define the conjugate of z bar_z
def bar_z : ℂ := conj z

-- State the theorem to prove
theorem modulus_of_conjugate_is_ten : complex.abs bar_z = 10 :=
by sorry

end modulus_of_conjugate_is_ten_l225_225783


namespace line_segments_cannot_form_triangle_l225_225569

theorem line_segments_cannot_form_triangle (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 7 = 21)
    (h3 : ∀ n, a n < a (n+1)) (h4 : ∀ i j k, a i + a j ≤ a k) :
    a 6 = 13 :=
    sorry

end line_segments_cannot_form_triangle_l225_225569


namespace volume_of_snow_l225_225294

theorem volume_of_snow (length width end_depth : ℝ) (start_depth : ℝ := 0)
  (h_length : length = 30)
  (h_width : width = 3)
  (h_end_depth : end_depth = 1)
  (h_start_depth : start_depth = 0) : 
  (length * width * ((end_depth + start_depth) / 2) = 45) :=
by 
  -- Definitions as conditions
  have avg_depth : ℝ := (end_depth + start_depth) / 2,
  have volume : ℝ := length * width * avg_depth,
  -- Given conditions from hypotheses
  rw [h_length, h_width, h_end_depth, h_start_depth],
  -- Simplify the expression
  simp only [length, width, end_depth, start_depth, avg_depth, volume],
  -- Show the correct volume
  exact eq.symm (calc 30 * 3 * (1 / 2) = 45 : by norm_num)

end volume_of_snow_l225_225294


namespace min_xy_value_l225_225039

theorem min_xy_value (x y : ℝ) 
  (h : 1 + cos^2 (x + y - 1) = (x^2 + y^2 + 2 * (x + 1) * (1 - y)) / (x - y + 1))
  : ∃ k : ℤ, xy = (1 + k * Real.pi)^2 / 4 ∧ ∀ t : ℤ, (1 + t * Real.pi)^2 / 4 ≥ 1/4 :=
sorry

end min_xy_value_l225_225039


namespace trig_identity_l225_225065

theorem trig_identity (x : ℝ) (h0 : -3 * Real.pi / 2 < x) (h1 : x < -Real.pi) (h2 : Real.tan x = -3) :
  Real.sin x * Real.cos x = -3 / 10 :=
sorry

end trig_identity_l225_225065


namespace fraction_of_selected_films_in_color_l225_225598

variables (x y : ℕ)

def B := 20 * x
def C := 8 * y

def selected_bw := (y / x : ℚ) / 100 * B
def selected_color := C

theorem fraction_of_selected_films_in_color :
  (selected_color / (selected_bw + selected_color) : ℚ) = 40 / 41 :=
by sorry

end fraction_of_selected_films_in_color_l225_225598


namespace bread_remaining_after_four_days_l225_225136

theorem bread_remaining_after_four_days
    (initial_pieces : ℕ)
    (fraction_day1 : ℚ)
    (fraction_day2 : ℚ)
    (fraction_day3 : ℚ)
    (fraction_day4 : ℚ) :
    initial_pieces = 500 →
    fraction_day1 = 1/4 →
    fraction_day2 = 2/5 →
    fraction_day3 = 3/8 →
    fraction_day4 = 1/3 →
    (let pieces_remaining_day1 := initial_pieces * (1 - fraction_day1) in
     let pieces_remaining_day2 := pieces_remaining_day1 * (1 - fraction_day2) in
     let pieces_remaining_day3 := pieces_remaining_day2 * (1 - fraction_day3) in
     pieces_remaining_day3 - round(piece_fraction_day3 * fraction_day3) = 141 →
     let pieces_remaining_day4 := pieces_remaining_day3 - round(pieces_remaining_day3 * fraction_day4) in
     pieces_remaining_day4 = 94) :=
by
  intros h_initial h_frac1 h_frac2 h_frac3 h_frac4
  have h_day1 : pieces_remaining_day1 = 500 * (1 - 1/4) := by sorry
  have h_day2 : pieces_remaining_day2 = 375 * (1 - 2/5) := by sorry
  have h_day3 : pieces_remaining_day3 = 225 * (1 - 3/8) := by sorry
  have h_day4 : pieces_remaining_day4 = 141 -  round(141 * 1/3) := by sorry
  sorry

end bread_remaining_after_four_days_l225_225136


namespace sin_cos_product_l225_225759

theorem sin_cos_product (α : ℝ) (h : sin α - cos α = (Real.sqrt 2) / 2) : 
  sin α * cos α = 1 / 4 :=
sorry

end sin_cos_product_l225_225759


namespace proof_equal_segments_l225_225901

-- Define the geometry problem: existence of the points T, P, Q, X, Y with the given properties
open EuclideanGeometry

variables {A B C T P Q X Y : Point}
variables (h_acute : acute (angle A B C))
variables (h_bisector : bisector_point T (angle A B C))
variables (h_circle_S : ∃ (S : Circle), diameter S = B T ∧ S P ∧ S Q)
variables (h_tangent_A : ∃ (circleA : Circle), circleA A ∧ tangent_at circleA S P ∧ intersects_with_line circleA AC X)
variables (h_tangent_C : ∃ (circleC : Circle), circleC C ∧ tangent_at circleC S Q ∧ intersects_with_line circleC AC Y)

-- Main goal: prove the equality TX = TY
theorem proof_equal_segments (h_acute : acute (angle A B C)) (h_bisector : bisector_point T (angle A B C)) 
    (h_circle_S : ∃ (S : Circle), diameter S = B T ∧ S P ∧ S Q)
    (h_tangent_A : ∃ (circleA : Circle), circleA A ∧ tangent_at circleA S P ∧ intersects_with_line circleA AC X)
    (h_tangent_C : ∃ (circleC : Circle), circleC C ∧ tangent_at circleC S Q ∧ intersects_with_line circleC AC Y) :
    distance T X = distance T Y :=
by
  sorry

end proof_equal_segments_l225_225901


namespace complex_number_in_second_quadrant_l225_225397

theorem complex_number_in_second_quadrant
  (a : ℝ) (z : ℂ) (h : z = a + complex.I * real.sqrt 3)
  (h1 : z.im > 0) (h2 : z.re < 0) (h3 : complex.abs z = 2) :
  z = -1 + complex.I * real.sqrt 3 :=
by
  sorry

end complex_number_in_second_quadrant_l225_225397


namespace min_speed_to_arrive_before_cara_l225_225607

theorem min_speed_to_arrive_before_cara (d : ℕ) (sc : ℕ) (tc : ℕ) (sd : ℕ) (td : ℕ) (hd : ℕ) :
  d = 180 ∧ sc = 30 ∧ tc = d / sc ∧ hd = 1 ∧ td = tc - hd ∧ sd = d / td ∧ (36 < sd) :=
sorry

end min_speed_to_arrive_before_cara_l225_225607


namespace edge_coloring_balance_l225_225278

-- Conditions: G is a connected graph with at least one vertex of odd degree.
variables {G : Type*} [fintype G] [graph G] [connected G]
           (h : ∃ v : G, odd (degree v))

-- Question: Prove that there exists an edge coloring such that the absolute difference between the number of red edges and blue edges at each vertex does not exceed 1.
theorem edge_coloring_balance :
  ∃ (coloring : edge_set G → ℕ),
    (∀ e, 0 ≤ coloring e ∧ coloring e ≤ 1) ∧  -- coloring is either 0 (red) or 1 (blue)
    (∀ v : G, abs((Σ (e ∈ edge_set G), if (coloring e = 0) then 1 else 0) -
                (Σ (e ∈ edge_set G), if (coloring e = 1) then 1 else 0)) ≤ 1) :=
sorry

end edge_coloring_balance_l225_225278


namespace sphere_radius_in_cube_l225_225538

theorem sphere_radius_in_cube (r : ℝ) (n : ℕ) (side_length : ℝ) 
  (h1 : side_length = 2) 
  (h2 : n = 16)
  (h3 : ∀ (i : ℕ), i < n → (center_distance : ℝ) = 2 * r)
  (h4: ∀ (i : ℕ), i < n → (face_distance : ℝ) = r) : 
  r = 1 :=
by
  sorry

end sphere_radius_in_cube_l225_225538


namespace sara_spent_on_hotdog_l225_225911

-- Define variables for the costs
def costSalad : ℝ := 5.1
def totalLunchBill : ℝ := 10.46

-- Define the cost of the hotdog
def costHotdog : ℝ := totalLunchBill - costSalad

-- The theorem we need to prove
theorem sara_spent_on_hotdog : costHotdog = 5.36 := by
  -- Proof would go here (if required)
  sorry

end sara_spent_on_hotdog_l225_225911


namespace range_of_x_l225_225799

theorem range_of_x (x : ℝ) : 
  ¬ (x ∈ set.Icc 2 5 ∨ x ∈ (set.Iio 1 ∪ set.Ioi 4)) → x ∈ set.Ico 1 2 :=
sorry

end range_of_x_l225_225799


namespace monic_quadratic_with_roots_l225_225733

-- Define the given conditions and roots
def root1 : ℂ := 2 - 3 * complex.i
def root2 : ℂ := 2 + 3 * complex.i

-- Define the polynomial with these roots
def poly (x : ℂ) : ℂ := x^2 - 4 * x + 13

-- Define the proof problem
theorem monic_quadratic_with_roots 
  (h1 : poly root1 = 0) 
  (h2 : poly root2 = 0) 
  (h_monic : ∀ x : ℂ, ∃! p : polynomial ℂ, p.monic ∧ p.coeff 2 = 1) : 
  ∃ p : polynomial ℝ, p.monic ∧ ∀ x : ℂ, p.eval x = poly x := 
sorry

end monic_quadratic_with_roots_l225_225733


namespace find_2013th_term_l225_225547

def sum_of_digits (n : Nat) : Nat :=
  Nat.digits 10 n |>.sum

def recurrence (a_n : Nat) : Nat :=
  (sum_of_digits a_n) * 13

def sequence : Nat → Nat
| 0     => 934
| (n+1) => recurrence (sequence n)

theorem find_2013th_term : sequence 2012 = 130 := 
  sorry

end find_2013th_term_l225_225547


namespace arithmetic_sequence_thm_l225_225106

theorem arithmetic_sequence_thm
  (a : ℕ → ℝ)
  (h1 : a 1 + a 4 + a 7 = 48)
  (h2 : a 2 + a 5 + a 8 = 40)
  (d : ℝ)
  (h3 : ∀ n, a (n + 1) = a n + d) :
  a 3 + a 6 + a 9 = 32 :=
by {
  sorry
}

end arithmetic_sequence_thm_l225_225106


namespace Jim_paycheck_after_deductions_l225_225471

def calculateRemainingPay (grossPay : ℕ) (retirementPercentage : ℕ) 
                          (taxDeduction : ℕ) : ℕ :=
  let retirementAmount := (grossPay * retirementPercentage) / 100
  let afterRetirement := grossPay - retirementAmount
  let afterTax := afterRetirement - taxDeduction
  afterTax

theorem Jim_paycheck_after_deductions :
  calculateRemainingPay 1120 25 100 = 740 := 
by
  sorry

end Jim_paycheck_after_deductions_l225_225471


namespace part1_part2_part3_l225_225412

-- Given sequence definition
def seq_a (n : ℕ) : ℚ :=
  if h : n = 1 then 1 else
    seq_a (n - 1) / (2 * seq_a (n - 1) + 1)

-- Part 1: Proving that {1/a_n} is an arithmetic sequence
theorem part1 (n : ℕ) : ∃ d : ℚ, ∀ m : ℕ, 1 / seq_a (m + 1) - 1 / seq_a m = d := sorry

-- Part 2: Finding the general term formula for {a_n}
theorem part2 (n : ℕ) : seq_a n = 1 / (2 * n - 1) := sorry

-- Part 3: Proving T_n < 3/4
noncomputable def seq_b (n : ℕ) : ℚ :=
  if h : (2 / (1 / seq_a n + 1)) ≠ 0 -- This condition is to avoid dividing by zero
  then 2 / (1 / seq_a n + 1)
  else 0

noncomputable def sum_T (n : ℕ) : ℚ :=
  ∑ i in (Finset.range n), seq_b i * seq_b (i + 2)

theorem part3 (n : ℕ) : sum_T n < 3 / 4 := sorry

end part1_part2_part3_l225_225412


namespace solve_fraction_equation_l225_225723

theorem solve_fraction_equation :
  {x : ℝ // (1 / (x^2 + 12 * x - 9) + 1 / (x^2 + 3 * x - 9) + 1 / (x^2 - 14 * x - 9) = 0)} =
  {(3 : ℝ), (1 : ℝ), (-3 : ℝ), (-9 : ℝ)} :=
by
  sorry

end solve_fraction_equation_l225_225723


namespace sum_increased_consecutive_integers_product_990_l225_225938

theorem sum_increased_consecutive_integers_product_990 
  (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 990) :
  (a + 2) + (b + 2) + (c + 2) = 36 :=
sorry

end sum_increased_consecutive_integers_product_990_l225_225938


namespace centroid_vector_sum_zero_centroid_pass_through_equation_l225_225779

variables {G A B O M P Q : Type}
variables {a b : vector} -- Assuming vector is a predefined type
variables (m n : ℝ) -- Scalars

-- Given conditions
def is_centroid (G : Type) (A B O : Type) : Prop := sorry
def is_midpoint (M : Type) (A B : Type) : Prop := sorry
def passes_through_centroid (P Q : Type) (G : Type) : Prop := sorry
def vector_OA_eq (A O : Type) (a : vector) : Prop := sorry
def vector_OB_eq (B O : Type) (b : vector) : Prop := sorry
def vector_OP_eq (P O : Type) (a : vector) (m : ℝ) : Prop := sorry
def vector_OQ_eq (Q O : Type) (b : vector) (n : ℝ) : Prop := sorry

-- Problems

-- (1) Prove that vector sum from centroid to vertices is 0
theorem centroid_vector_sum_zero (h1 : is_centroid G A B O) : 
  (\overrightarrow{GA} + \overrightarrow{GB} + \overrightarrow{GO} = 0) := sorry

-- (2) Prove the given relationship between m and n
theorem centroid_pass_through_equation (h2 : is_centroid G A B O) 
  (h3 : passes_through_centroid P Q G) 
  (h4 : vector_OA_eq A O a) 
  (h5 : vector_OB_eq B O b)
  (h6 : vector_OP_eq P O a m) 
  (h7 : vector_OQ_eq Q O b n) :
  (\frac{1}{m} + \frac{1}{n} = 3) := sorry

end centroid_vector_sum_zero_centroid_pass_through_equation_l225_225779


namespace bug_final_position_l225_225536

/-- Define the circle points and their jump rules based on the problem statement. --/
def circle_points := {1, 2, 3, 4, 5, 6, 7}

def jump (current_point : ℕ) : ℕ :=
  if current_point % 3 = 0 then (current_point + 2) % 7
  else (current_point + 3) % 7

theorem bug_final_position : (foldl (fun pos _ => jump pos) 7 (range 2023)) = 1 :=
by
  sorry

end bug_final_position_l225_225536


namespace remainder_of_9_pow_1995_mod_7_l225_225231

theorem remainder_of_9_pow_1995_mod_7 : (9^1995) % 7 = 1 := 
by 
sorry

end remainder_of_9_pow_1995_mod_7_l225_225231


namespace find_f_prime_at_1_l225_225391

def f (x : ℝ) (f_prime_at_1 : ℝ) : ℝ := x^3 + 2 * x * f_prime_at_1 - 1

theorem find_f_prime_at_1 (f_prime_at_1 : ℝ) 
  (h_deriv : ∀ x : ℝ, deriv (λ x, f x f_prime_at_1) x = 3 * x^2 + 2 * f_prime_at_1) :
  f_prime_at_1 = -3 :=
begin
  have h1 : deriv (λ x, f x f_prime_at_1) 1 = f_prime_at_1,
  { rw h_deriv,
    simp,},
  linarith,
end

end find_f_prime_at_1_l225_225391


namespace length_linear_function_alpha_increase_l225_225843

variable (l : ℝ) (l₀ : ℝ) (t : ℝ) (α : ℝ)

theorem length_linear_function 
  (h_formula : l = l₀ * (1 + α * t)) : 
  ∃ (f : ℝ → ℝ), (∀ t, f t = l₀ + l₀ * α * t ∧ (l = f t)) :=
by {
  -- Proof would go here
  sorry
}

theorem alpha_increase 
  (h_formula : l = l₀ * (1 + α * t))
  (h_initial : t = 1) :
  α = (l - l₀) / l₀ :=
by {
  -- Proof would go here
  sorry
}

end length_linear_function_alpha_increase_l225_225843


namespace who_stole_the_pan_l225_225112

def Frog_statement := "Lackey-Lech stole the pan"
def LackeyLech_statement := "I did not steal any pan"
def KnaveOfHearts_statement := "I stole the pan"

axiom no_more_than_one_liar : ∀ (frog_is_lying : Prop) (lackey_lech_is_lying : Prop) (knave_of_hearts_is_lying : Prop), (frog_is_lying → ¬ lackey_lech_is_lying) ∧ (frog_is_lying → ¬ knave_of_hearts_is_lying) ∧ (lackey_lech_is_lying → ¬ knave_of_hearts_is_lying)

theorem who_stole_the_pan : KnaveOfHearts_statement = "I stole the pan" :=
sorry

end who_stole_the_pan_l225_225112


namespace area_of_isosceles_right_triangle_l225_225837

theorem area_of_isosceles_right_triangle
  (XYZ : Type)
  [RightTriangle XYZ]
  [AngleEqual XYZ X Z]
  (XY : Length XYZ 10) :
  Area XYZ = 50 :=
sorry

end area_of_isosceles_right_triangle_l225_225837


namespace eccentricity_of_hyperbola_l225_225839

noncomputable def hyperbola_eccentricity {a b : ℝ} (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : ℝ :=
  let h := ∃ (x y : ℝ), (x / a)^2 - (y / b)^2 = 1 
  let ab_eqn := OF • AB = FA • FB 
  sqrt 2 + 1

theorem eccentricity_of_hyperbola {a b : ℝ} (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ) 
  (h : ∃ (x y : ℝ), (x / a)^2 - (y / b)^2 = 1)
  (ab_eqn : OF • AB = FA • FB) : hyperbola_eccentricity C F l A B = sqrt 2 + 1 := 
by
  sorry

end eccentricity_of_hyperbola_l225_225839


namespace sandy_total_spent_on_clothes_l225_225909

theorem sandy_total_spent_on_clothes :
  let shorts := 13.99
  let shirt := 12.14 
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 := 
by
  sorry

end sandy_total_spent_on_clothes_l225_225909


namespace horizontal_length_42_inch_tv_l225_225951

theorem horizontal_length_42_inch_tv (d : ℝ) (a b : ℝ) (h_d : d = 42) (h_aspect : a / b = 16 / 9) : 
  ∃ x : ℝ, a = 16 * x ∧ b = 9 * x ∧ ∀ horiz_len, horiz_len = 16 * (42 / Real.sqrt 337) := 
begin
  use 42 / Real.sqrt 337,
  split,
  { 
    show a = 16 * (42 / Real.sqrt 337),
    -- add proof here
    sorry 
  },
  split,
  { 
    show b = 9 * (42 / Real.sqrt 337),
    -- add proof here
    sorry 
  },
  {
    intro horiz_len,
    show horiz_len = 16 * (42 / Real.sqrt 337),
    -- add proof here
    sorry
  }
end

end horizontal_length_42_inch_tv_l225_225951


namespace intersection_correct_union_correct_intersection_complement_correct_l225_225060

def U := ℝ
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -3 ∨ x > 1}
def C_U_A : Set ℝ := {x | x ≤ 0 ∨ x > 2}
def C_U_B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

theorem intersection_correct : (A ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

theorem union_correct : (A ∪ B) = {x : ℝ | x < -3 ∨ x > 0} :=
sorry

theorem intersection_complement_correct : (C_U_A ∩ C_U_B) = {x : ℝ | -3 ≤ x ∧ x ≤ 0} :=
sorry

end intersection_correct_union_correct_intersection_complement_correct_l225_225060


namespace non_empty_subsets_count_l225_225936

theorem non_empty_subsets_count :
  ∃ s : set ℕ, s = {1, 2} ∧ (s.powerset.filter (λ t, t ≠ ∅)).card = 3 :=
by
  let s := {1, 2}
  existsi s
  split
  . exact rfl
  . sorry

end non_empty_subsets_count_l225_225936


namespace repair_cost_total_l225_225116

-- Define the inputs
def labor_cost_rate : ℤ := 75
def labor_hours : ℤ := 16
def part_cost : ℤ := 1200

-- Define the required computation and proof statement
def total_repair_cost : ℤ :=
  let labor_cost := labor_cost_rate * labor_hours
  labor_cost + part_cost

theorem repair_cost_total : total_repair_cost = 2400 := by
  -- Proof would go here
  sorry

end repair_cost_total_l225_225116


namespace monic_quadratic_with_root_l225_225726

theorem monic_quadratic_with_root (a b : ℝ) (h_root1 : (a + b * I) ∈ ({2-3*I, 2+3*I} : set ℂ)) :
  ∃ p : polynomial ℂ, polynomial.monic p ∧ p.coeff 0 = 13 ∧ p.coeff 1 = -4 ∧ p.coeff 2 = 1 ∧ p.eval (a + b * I) = 0 := 
by
  sorry

end monic_quadratic_with_root_l225_225726


namespace parabola_passes_origin_two_distinct_points_on_x_axis_a_not_equal_negative_half_condition_on_M_and_N_problem_solution_l225_225765

variable {a b c : ℝ}

def parabola (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_passes_origin (h₁ : b = -2 * a)
  (hA : parabola 2 = 0) : parabola 0 = 0 :=
by
  sorry

theorem two_distinct_points_on_x_axis (h₂ : c ≠ 4 * a)
  (hA : parabola 2 = 0) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ parabola x₁ = 0 ∧ parabola x₂ = 0 :=
by
  sorry

theorem a_not_equal_negative_half (hBOC: ∀ x : ℝ, parabola x = 0 → x = 2 ∨ x = 0)
  (hC : parabola 0 = c)
  (hOC : |c| = ∥(0, c)∥) : a ≠ -1 / 2 :=
by
  sorry

theorem condition_on_M_and_N (h3 : ∀ x1 x2 : ℝ, −1 < x2 ∧ x2 < x1 → parabola x1 > parabola x2):
  8 * a + c ≤ 0 :=
by
  sorry

theorem problem_solution
  (h₁ : b = -2 * a)
  (h₂ : c ≠ 4 * a)
  (hA : parabola 2 = 0)
  (hBOC: ∀ x : ℝ, parabola x = 0 → x = 2 ∨ x = 0)
  (hC : parabola 0 = c)
  (hOC : |c| = ∥(0, c)∥)
  (h3 : ∀ x1 x2 : ℝ, −1 < x2 ∧ x2 < x1 → parabola x1 > parabola x2)
  : 8 * a + c ≤ 0 ∧ parabola 0 = 0 ∧ (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ parabola x₁ = 0 ∧ parabola x₂ = 0) :=
by
  exact ⟨condition_on_M_and_N h3, parabola_passes_origin h₁ hA, two_distinct_points_on_x_axis h₂ hA⟩

end parabola_passes_origin_two_distinct_points_on_x_axis_a_not_equal_negative_half_condition_on_M_and_N_problem_solution_l225_225765


namespace solve_trig_equation_l225_225539

noncomputable def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, x = k * 360 ∨ x = 90 + k * 360

theorem solve_trig_equation (x : ℝ) :
  (sin x + cos x + sin x * cos x = 1) ↔ solution_set x :=
sorry

end solve_trig_equation_l225_225539


namespace negative_product_of_consecutive_terms_l225_225458

noncomputable def a : ℕ → ℚ
| 0 := 3
| (n+1) := (a n - (2/3))

theorem negative_product_of_consecutive_terms :
  (a 5) * (a 6) < 0 :=
by {
  sorry
}

end negative_product_of_consecutive_terms_l225_225458


namespace find_n_l225_225442

def f (n : ℝ) (x : ℝ) : ℝ := (n^2 - 3*n + 3) * x^(n^2 - 2*n)

theorem find_n {n : ℝ} :
  (∀ x > 0, (f n x) ≤ 0) → (n = 1) :=
by
  sorry

end find_n_l225_225442


namespace distance_calculations_l225_225002

def dist_between_planes : ℝ :=
  let n := (4, -2, 2) in
  let magnitude_n := (4^2 + (-2)^2 + 2^2)^(1/2) in
  let plane1_point := (5/2 : ℝ, 0, 0) in
  (| 4 * (5 / 2) - 2 * 0 + 2 * 0 - 2 |) / magnitude_n

def dist_point_to_plane (p : ℝ × ℝ × ℝ) : ℝ :=
  let n := (4, -2, 2) in
  let magnitude_n := (4^2 + (-2)^2 + 2^2)^(1/2) in
  (| 4 * p.1 - 2 * p.2 + 2 * p.3 - 2 |) / magnitude_n

theorem distance_calculations :
  dist_between_planes = 2 * real.sqrt 6 / 3 ∧
  dist_point_to_plane (1, 0, 0) = real.sqrt 6 / 3 :=
by
  sorry

end distance_calculations_l225_225002


namespace similar_triangles_height_l225_225963

theorem similar_triangles_height (h_smaller : ℝ) (ratio_areas : ℝ) (height_larger : ℝ) 
  (h_smaller_val : h_smaller = 4) (ratio_areas_val : ratio_areas = 1 / 25) : 
  height_larger = 20 :=
begin
  -- Given two similar triangles
  -- The ratio of their areas is 1:25 (which is 1/25)
  -- The height of the smaller triangle is 4 cm
  -- We need to prove that the height of the larger triangle is 20 cm
  sorry
end

end similar_triangles_height_l225_225963


namespace expected_survivors_approx_l225_225093

-- Define the initial conditions
def probability_dying_first_month : ℝ := 1/8
def probability_dying_second_month : ℝ := 1/6
def probability_dying_third_month : ℝ := 1/4
def disease_rate : ℝ := 0.05
def reduction_factor_due_to_disease : ℝ := 0.5
def total_population : ℕ := 800

-- Define the survival probabilities
def survival_rate_first_month : ℝ := 1 - probability_dying_first_month
def survival_rate_diseased_second_month : ℝ := 1 - (probability_dying_second_month * (1 + reduction_factor_due_to_disease))
def survival_rate_healthy_second_month : ℝ := 1 - probability_dying_second_month
def survival_rate_second_month : ℝ :=
  (1 - disease_rate) * survival_rate_healthy_second_month + disease_rate * survival_rate_diseased_second_month
def survival_rate_third_month : ℝ := 1 - probability_dying_third_month

-- Calculate the overall survival rate for three months
def overall_survival_rate : ℝ :=
  survival_rate_first_month * survival_rate_second_month * survival_rate_third_month

-- Expected number of survivors
def expected_survivors : ℝ := total_population * overall_survival_rate

-- The theorem stating the expected number of survivors is approximately 435
theorem expected_survivors_approx : abs (expected_survivors - 435) < 1 :=
by
  sorry

end expected_survivors_approx_l225_225093


namespace trapezoid_scalar_product_l225_225926

open Real -- We work with real numbers.
open EuclideanGeometry -- To work with vectors and Euclidean geometry.

/--
The bases \(AB\) and \(CD\) of trapezoid \(ABCD\) are 155 and 13 respectively,
and its lateral sides are mutually perpendicular. 
Find the scalar product of the vectors \(\overrightarrow{AC}\) and \(\overrightarrow{BD}\).
-/
theorem trapezoid_scalar_product :
  let AB := 155
  let CD := 13
  ∀ (a b : ℝ^2) (OA OB : ℝ^2), 
    (OA · OB = 0) → -- Perpendicular sides condition.
    (\|AO\| ^ 2 + \|BO\| ^ 2 = AB ^ 2) → -- Magnitude condition obtained in the solution.
    (a + CD *†OB) = b + CD *†OA
  (inner (a - CD/AB *† b) (b - CD/AB *† a)) = -2015 :=
sorry -- Proof omitted.

end trapezoid_scalar_product_l225_225926


namespace complex_transform_to_polar_l225_225213

theorem complex_transform_to_polar (θ : ℝ) :
  let z := (1 + real.sin θ + real.cos θ * complex.I) / (1 + real.sin θ - real.cos θ * complex.I) in
  z = complex.cos (real.pi / 2 - θ) + complex.sin (real.pi / 2 - θ) * complex.I :=
by
  sorry

end complex_transform_to_polar_l225_225213


namespace animal_arrangement_count_l225_225169

theorem animal_arrangement_count :
  let chickens := 3
  let dogs := 3
  let cats := 4
  let total_cages := 10
  (∑ (x : Finset (Fin (total_cages - chickens + 1))) in
    (∑ (y : Finset (Fin ((total_cages - chickens - dogs + 1)))) in
      (∑ (z : Finset (Fin ((total_cages - chickens - dogs - cats + 1)))) in
          if disjoint x (y ∪ z) ∧ disjoint y z ∧ disjoint z x then 1 else 0)) = 5184 := sorry

end animal_arrangement_count_l225_225169


namespace pies_baked_l225_225892

/-- Mrs. Hilt baked 16.0 pecan pies and 14.0 apple pies. She needs 5.0 times this amount.
    Prove that the total number of pies she has to bake is 150.0. -/
theorem pies_baked (pecan_pies : ℝ) (apple_pies : ℝ) (times : ℝ)
  (h1 : pecan_pies = 16.0) (h2 : apple_pies = 14.0) (h3 : times = 5.0) :
  times * (pecan_pies + apple_pies) = 150.0 := by
  sorry

end pies_baked_l225_225892


namespace components_le_20_components_le_n_squared_div_4_l225_225835

-- Question part b: 8x8 grid, can the number of components be more than 20
theorem components_le_20 {c : ℕ} (h1 : c = 64 / 4) : c ≤ 20 := by
  sorry

-- Question part c: n x n grid, can the number of components be more than n^2 / 4
theorem components_le_n_squared_div_4 (n : ℕ) (h2 : n > 8) {c : ℕ} (h3 : c = n^2 / 4) : 
  c ≤ n^2 / 4 := by
  sorry

end components_le_20_components_le_n_squared_div_4_l225_225835


namespace total_gems_l225_225292

theorem total_gems (diamonds rubies : ℕ) (h1 : diamonds = 45) (h2 : rubies = 5110) : diamonds + rubies = 5155 :=
by
  rw [h1, h2]
  show 45 + 5110 = 5155
  sorry

end total_gems_l225_225292


namespace board_rook_placement_l225_225670

-- Define the color function for the board
def color (n i j : ℕ) : ℕ :=
  min (i + j - 1) (2 * n - i - j + 1)

-- Conditions: It is possible to place n rooks such that no two attack each other and 
-- no two rooks stand on cells of the same color
def non_attacking_rooks (n : ℕ) (rooks : Fin n → Fin n) : Prop :=
  ∀ i j : Fin n, i ≠ j → rooks i ≠ rooks j ∧ color n i.val (rooks i).val ≠ color n j.val (rooks j).val

-- Main theorem to be proven
theorem board_rook_placement (n : ℕ) :
  (∃ rooks : Fin n → Fin n, non_attacking_rooks n rooks) →
  n % 4 = 0 ∨ n % 4 = 1 :=
by
  intros h
  sorry

end board_rook_placement_l225_225670


namespace min_abs_sum_l225_225073

theorem min_abs_sum (x : ℝ) : (∃ x : ℝ, ∀ y : ℝ, (|y - 2| + |y - 47| ≥ |x - 2| + |x - 47|)) → (|x - 2| + |x - 47| = 45) :=
by
  sorry

end min_abs_sum_l225_225073


namespace jim_gave_away_675_cards_l225_225853

def total_cards_gave_away
  (cards_per_set : ℕ)
  (sets_to_brother sets_to_sister sets_to_friend : ℕ)
  : ℕ :=
  (sets_to_brother + sets_to_sister + sets_to_friend) * cards_per_set

theorem jim_gave_away_675_cards
  (cards_per_set : ℕ)
  (sets_to_brother sets_to_sister sets_to_friend : ℕ)
  (h_brother : sets_to_brother = 15)
  (h_sister : sets_to_sister = 8)
  (h_friend : sets_to_friend = 4)
  (h_cards_per_set : cards_per_set = 25)
  : total_cards_gave_away cards_per_set sets_to_brother sets_to_sister sets_to_friend = 675 :=
by
  sorry

end jim_gave_away_675_cards_l225_225853


namespace graph_symmetry_y_axis_l225_225845

theorem graph_symmetry_y_axis :
  ∀ (x : ℝ), 2^(-x) = (1/2)^x :=
by
  sorry

end graph_symmetry_y_axis_l225_225845


namespace smallest_k_bound_l225_225535

noncomputable def v_seq : ℕ → ℝ
| 0 := 1/3
| (k+1) := 3/2 * v_seq k - 3/2 * (v_seq k)^2

def limit_M : ℝ := 1/2

theorem smallest_k_bound:
  ∃ k, k = 5 ∧ |v_seq k - limit_M| ≤ 1/2^20 :=
by
  sorry

end smallest_k_bound_l225_225535


namespace PQ_relationship_l225_225435

-- Define the sets P and Q
def P := {x : ℝ | x >= 5}
def Q := {x : ℝ | 5 <= x ∧ x <= 7}

-- Statement to be proved
theorem PQ_relationship : Q ⊆ P ∧ Q ≠ P :=
by
  sorry

end PQ_relationship_l225_225435


namespace angle_bisector_median_tangent_identity_l225_225447

-- Definition of right triangle with angle bisector and median
variables {A B C D M : Type}
variables [Triangle ABC C = 90°]
variables [AngleBisector AD A]
variables [Median AM A B]
variables {α : ℝ} {φ : ℝ}

-- The statement to be proven
theorem angle_bisector_median_tangent_identity
  (h1 : ∠C = 90°)
  (h2 : AngleBisector AD A)
  (h3 : Median AM A B)
  (h4 : ∠DAM = φ) :
  tan φ = (tan (α / 2)) ^ 3 := 
sorry

end angle_bisector_median_tangent_identity_l225_225447


namespace proving_n_equals_five_l225_225437

theorem proving_n_equals_five (n : ℕ) (h : ∃ r : ℕ, (-1)^r * 2^(n-r) * Nat.choose n r * x^((n - 3 * r) / 2) = -80 ∧ (n - 3 * r) / 2 = 1) : n = 5 :=
by
  sorry -- Proof omitted

end proving_n_equals_five_l225_225437


namespace area_swept_opposite_side_area_swept_full_rotation_l225_225368

theorem area_swept_opposite_side (c : ℝ) :
  let t1 := (c^2 / 4) * (2 * Real.pi + 3 * Real.sqrt 3) in
  t1 = (c^2 / 4) * (2 * Real.pi + 3 * Real.sqrt 3) := by
  sorry

theorem area_swept_full_rotation (c : ℝ) :
  let t2 := (c^2 / 3) * (Real.pi + 6 * Real.sqrt 3) in
  t2 = (c^2 / 3) * (Real.pi + 6 * Real.sqrt 3) := by
  sorry

end area_swept_opposite_side_area_swept_full_rotation_l225_225368


namespace expression_value_l225_225081

theorem expression_value (a b : ℚ) (h : a + 2 * b = 0) : 
  abs (a / |b| - 1) + abs (|a| / b - 2) + abs (|a / b| - 3) = 4 :=
sorry

end expression_value_l225_225081


namespace modulus_problem_l225_225262

theorem modulus_problem : (13 ^ 13 + 13) % 14 = 12 :=
by
  sorry

end modulus_problem_l225_225262


namespace real_part_is_neg4_l225_225388

def real_part_of_z (z : ℂ) : ℝ :=
  z.re

theorem real_part_is_neg4 (i : ℂ) (h : i^2 = -1) :
  real_part_of_z ((3 + 4 * i) * i) = -4 := by
  sorry

end real_part_is_neg4_l225_225388


namespace find_f_expression_l225_225380

theorem find_f_expression (f : ℝ → ℝ) (h_diff : ∀ x ∈ ((set.Ioo (-∞:ℝ) 1) ∪ (set.Ioo 1 ∞)), differentiable_at ℝ f x) (h_eq : ∀ x ∈ ((set.Ioo (-∞:ℝ) 1) ∪ (set.Ioo 1 ∞)), f x = (deriv f 2) * x^2 + x * f x + x) : 
  ∀ x ∈ ((set.Ioo (-∞:ℝ) 1) ∪ (set.Ioo 1 ∞)), f x = (x^2 + x) / (1 - x) := 
by sorry

end find_f_expression_l225_225380


namespace symmetric_function_l225_225415

variable {α : Type*} [NonEmpty α] [LinearOrder α]

-- Given first function and its inverse
variable (ϕ : α → α)
variable (ϕ_inv : α → α)

-- Assume inverse relationship
axiom inv_property : ∀ x, ϕ (ϕ_inv x) = x ∧ ϕ_inv (ϕ x) = x

-- Define the third function
noncomputable def third_function (x : α) : α :=
  -ϕ_inv (-x)

-- The proof goal
theorem symmetric_function : ∀ x, third_function ϕ ϕ_inv x = -ϕ_inv (-x) :=
by
  intro x
  simp only [third_function, neg_inj]
  exact sorry

end symmetric_function_l225_225415


namespace quadratic_real_roots_range_l225_225079

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (x^2 - x - m = 0)) ↔ m ≥ -1 / 4 :=
by sorry

end quadratic_real_roots_range_l225_225079


namespace find_function_l225_225025

def f (x : ℝ) (k : ℝ) (b : ℝ) : ℝ :=
  k * x + b

theorem find_function (k b : ℝ) (h1 : k < 0) (h2 : ∀ x, f (f x k b) k b = 4 * x + 1) : 
  f = fun x => -2 * x - 1 := 
sorry

end find_function_l225_225025


namespace avg_integer_N_between_fractions_l225_225583

theorem avg_integer_N_between_fractions (N : ℕ) (h1 : (2 : ℚ) / 5 < N / 42) (h2 : N / 42 < 1 / 3) : 
  N = 15 := 
by
  sorry

end avg_integer_N_between_fractions_l225_225583


namespace find_ellipse_and_point_M_l225_225131

noncomputable def ellipse_equation {a b : ℝ} : Prop :=
  a > b ∧ b > 0 ∧ (P : ℝ × ℝ) (P = (1, 3 / 2)) ∧ 
  (∀ x y, (x, y) ∈ C ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  ∃ F1 F2 A B : ℝ × ℝ, F1 = (-1, 0) ∧ line (F1) ∩ C = {A, B} ∧ (perimeter F2 A B = 8)

theorem find_ellipse_and_point_M {a b : ℝ} (h : ellipse_equation) :
  (∀ x y, (x, y) ∈ C ↔ x^2 / 4 + y^2 / 3 = 1) ∧ 
  ∃ M : ℝ × ℝ, M = (-4,0) ∧ (∀ A B : ℝ × ℝ, 
  line (M) ∩ {A, B} →
  ∠ (F1 M A) = ∠ (F1 M B)) :=
sorry

end find_ellipse_and_point_M_l225_225131


namespace smallest_three_digit_number_with_property_l225_225340

theorem smallest_three_digit_number_with_property :
  ∃ a : ℕ, 100 ≤ a ∧ a ≤ 999 ∧ ∃ n : ℕ, 1001 * a + 1 = n^2 ∧ ∀ b : ℕ, 100 ≤ b ∧ b ≤ 999 ∧ (∃ m : ℕ, 1001 * b + 1 = m^2) → a ≤ b :=
begin
  sorry,
end

end smallest_three_digit_number_with_property_l225_225340


namespace boat_speed_correct_l225_225561

def speed_of_boat_in_still_water (c d t : ℝ) (h1 : c = 5) (h2 : d = 11.25) (h3 : t = 27/60) : ℝ :=
  let V_d := d / t in
  let V_b := V_d - c in
  20

theorem boat_speed_correct {c d t : ℝ} (h1 : c = 5) (h2 : d = 11.25) (h3 : t = 27/60) :
  speed_of_boat_in_still_water c d t h1 h2 h3 = 20 := by
  sorry

end boat_speed_correct_l225_225561


namespace reflection_slope_l225_225193

noncomputable def calc_slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem reflection_slope (m b : ℝ) :
  let p1 := (-4, 2) in
  let p2 := (6, -2) in
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2) in
  let original_slope := calc_slope p1 p2 in
  let reflected_slope := -5/2 in
  let line_condition := (midpoint.2 = reflected_slope * midpoint.1 + b) in
  calc_slope p1 p2 = -2/5 →
  m = -(1 / original_slope) →
  line_condition →
  m + b = 0 :=
sorry

end reflection_slope_l225_225193


namespace find_bases_of_isosceles_trapezoid_l225_225637

noncomputable def isosceles_trapezoid_bases (c d : ℝ) (h : c < d) : Prop :=
  ∃ b1 b2 : ℝ, b1 = (sqrt (d + c) + sqrt (d - c))^2 / 2 ∧ b2 = (sqrt (d + c) - sqrt (d - c))^2 / 2

theorem find_bases_of_isosceles_trapezoid (c d : ℝ) (h : c < d) :
  ∃ (b1 b2 : ℝ), b1 = (sqrt (d + c) + sqrt (d - c))^2 / 2 ∧ b2 = (sqrt (d + c) - sqrt (d - c))^2 / 2 := 
sorry

end find_bases_of_isosceles_trapezoid_l225_225637


namespace fg_inequality_l225_225123

variables (f g : ℝ → ℝ)
variables (a b x : ℝ)

theorem fg_inequality (hf : differentiable ℝ f) (hg : differentiable ℝ g) 
  (h_cond : (f' x * g x + f x * g' x) < 0) (h_interval : a < x ∧ x < b) :
  f x * g x > f b * g b :=
sorry

end fg_inequality_l225_225123


namespace find_attendeesTuesday_l225_225617

def attendeesMonday : ℕ := 10
def attendeesTuesday : ℕ
def attendeesWednesday : ℕ := 10
def attendeesThursday : ℕ := 10
def attendeesFriday : ℕ := 10

def total_attendees (T : ℕ) : ℕ := attendeesMonday + T + attendeesWednesday + attendeesThursday + attendeesFriday

def average_attendance (T : ℕ) : ℕ := total_attendees T / 5

theorem find_attendeesTuesday (T : ℕ) (h : average_attendance T = 11) : T = 15 := by 
  sorry

end find_attendeesTuesday_l225_225617


namespace part_I_part_II_l225_225514

def f (x a : ℝ) : ℝ := |2 * x + 1| + |2 * x - a| + a

theorem part_I (x : ℝ) (h₁ : f x 3 > 7) : sorry := sorry

theorem part_II (a : ℝ) (h₂ : ∀ (x : ℝ), f x a ≥ 3) : sorry := sorry

end part_I_part_II_l225_225514


namespace area_union_of_triangles_l225_225111

noncomputable def area_of_union {D E F D' E' F' H : Type*}
  [linear_ordered_field H]
  (DE EF DF : H) (hDE : DE = 8) (hEF : EF = 15) (hDF : DF = 17)
  (H_is_median_intersection : Prop)
  (rotation_180 : Prop) :
  real :=
  60
  
theorem area_union_of_triangles:
  ∀ (DE EF DF : ℝ)
  (hDE : DE = 8)
  (hEF : EF = 15)
  (hDF : DF = 17),
  let s := (DE + EF + DF) / 2 in
  let area_DEF := real.sqrt (s * (s - DE) * (s - EF) * (s - DF)) in
  area_DEF = 60 :=
by
  intro DE EF DF hDE hEF hDF
  let s := (DE + EF + DF) / 2
  let area_DEF := real.sqrt (s * (s - DE) * (s - EF) * (s - DF))
  have h_area_DEF : area_DEF = real.sqrt (20 * 12 * 5 * 3) := sorry
  have h_sqrt_3600 : real.sqrt 3600 = 60 := sorry
  rw [h_area_DEF, h_sqrt_3600]
  refl

end area_union_of_triangles_l225_225111


namespace card_game_is_unfair_l225_225627

noncomputable def card_game_fairness : Prop :=
  let outcomes := [(5, 5), (5, 6), (5, 7), (6, 5), (6, 6), (6, 7), (7, 5), (7, 6), (7, 7)]
  let even_wins := outcomes.filter (fun p => (p.1 * p.2) % 2 = 0)
  let odd_wins := outcomes.filter (fun p => (p.1 * p.2) % 2 = 1)
  even_wins.length ≠ odd_wins.length

theorem card_game_is_unfair : card_game_fairness :=
by 
  -- We state that the game is unfair because the number of outcomes where B wins (even product)
  -- is not equal to the number of outcomes where A wins (odd product).
  let outcomes := [(5, 5), (5, 6), (5, 7), (6, 5), (6, 6), (6, 7), (7, 5), (7, 6), (7, 7)]
  let even_wins := outcomes.filter (fun p => (p.1 * p.2) % 2 = 0)
  let odd_wins := outcomes.filter (fun p => (p.1 * p.2) % 2 = 1)
  have h1 : even_wins.length = 5 := sorry -- This would involve enumerating the outcomes with even product
  have h2 : odd_wins.length = 4 := sorry -- This would involve enumerating the outcomes with odd product
  show even_wins.length ≠ odd_wins.length, from sorry

end card_game_is_unfair_l225_225627


namespace square_side_length_nearest_cm_l225_225197

theorem square_side_length_nearest_cm (A : ℝ) (hA : A = 42.25) :
  ((Real.sqrt A).ceil : ℝ) = 7 := by
  -- proof skipped
  sorry

end square_side_length_nearest_cm_l225_225197


namespace factorize_expression_l225_225717

theorem factorize_expression (x y : ℝ) : x^2 * y - 2 * x * y^2 + y^3 = y * (x - y)^2 := 
sorry

end factorize_expression_l225_225717


namespace probability_given_A_l225_225755

-- Define the sample space
def sample_space : set (ℕ × ℕ) := {(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)}

-- Define events A and B
def event_A : set (ℕ × ℕ) := {x ∈ sample_space | (x.1 + x.2) % 2 = 0}
def event_B : set (ℕ × ℕ) := {x ∈ sample_space | x.1 % 2 = 0 ∧ x.2 % 2 = 0}

-- Define the intersection of A and B
def event_A_and_B : set (ℕ × ℕ) := event_A ∩ event_B

theorem probability_given_A (h : ∀ x ∈ event_A, ¬ x ∈ event_B) : 
  ∑' (x : event_B), (1 : ℝ) / (event_A ∩ event_B).to_finset.card =
  1/4 := 
sorry

end probability_given_A_l225_225755


namespace number_of_persons_l225_225543

theorem number_of_persons (T : ℝ) (N : ℝ) : 
  (T / N) - 3 = (T - 42 + 12) / N → N = 10 := 
by
  intro h
  have h1 : T / N - 3 = (T - 30) / N := by
    simp [h]
  have h2 : T - 3 * N = T - 30 := by
    simp [h1]
  sorry

end number_of_persons_l225_225543


namespace inequality_XC_XA_XB_l225_225369

-- Definition of points in 3D space
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

-- Helper function to compute squared distance between two points
def squared_distance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2

-- Given points A, B, C and respective squared distances
axiom A B C : Point
axiom non_collinear : A ≠ B ∧ B ≠ C ∧ C ≠ A -- non-collinearity
axiom inequality_AB_CA_CB :
  squared_distance A B ≥ squared_distance C A + squared_distance C B

-- The statement we need to prove
theorem inequality_XC_XA_XB (X : Point) : 
  squared_distance X C ≤ squared_distance X A + squared_distance X B :=
sorry

end inequality_XC_XA_XB_l225_225369


namespace simplify_expression_l225_225159

theorem simplify_expression (b : ℝ) :
  (1 * (2 * b) * (3 * b^2) * (4 * b^3) * (5 * b^4) * (6 * b^5)) = 720 * b^15 :=
by
  sorry

end simplify_expression_l225_225159


namespace members_ordered_orange_juice_l225_225277

theorem members_ordered_orange_juice :
  ∀ (total_members : ℕ) (p : ℚ) (q : ℚ) (r : ℚ),
  total_members = 75 →
  p = 3 / 7 →
  q = 1 / 4 →
  r = total_members * p →
  let remaining_members := total_members - r in
  let mango_juice_members := remaining_members * q in
  remaining_members - mango_juice_members = 32 :=
by
  intros total_members p q r
  intros ht hp hq hr
  let remaining_members := total_members - r
  let mango_juice_members := remaining_members * q
  exact sorry

end members_ordered_orange_juice_l225_225277


namespace trains_pass_each_other_in_l225_225962

def length_first_train : ℝ := 350 -- in meters
def length_second_train : ℝ := 450 -- in meters
def speed_first_train : ℝ := 63 * (1000 / 3600) -- converted to m/s
def speed_second_train : ℝ := 81 * (1000 / 3600) -- converted to m/s

theorem trains_pass_each_other_in :
  let relative_speed := speed_second_train - speed_first_train
  let total_length := length_first_train + length_second_train
  total_length / relative_speed = 160 := by
  sorry

end trains_pass_each_other_in_l225_225962


namespace relationship_of_variables_l225_225761

variable {a b c d : ℝ}

theorem relationship_of_variables 
  (h1 : d - a < c - b) 
  (h2 : c - b < 0) 
  (h3 : d - b = c - a) : 
  d < c ∧ c < b ∧ b < a := 
sorry

end relationship_of_variables_l225_225761


namespace percentage_increase_l225_225270

theorem percentage_increase (original final : ℝ) (h1 : original = 90) (h2 : final = 135) : ((final - original) / original) * 100 = 50 := 
by
  sorry

end percentage_increase_l225_225270


namespace probability_in_equilateral_triangle_l225_225860

def equilateral_triangle (A B C : Point) : Prop :=
  -- Definition of an equilateral triangle with vertices A, B, and C
  dist A B = dist B C ∧ dist B C = dist C A

def sum_of_distances_to_sides (P A B C : Point) : ℝ :=
  let d1 := dist_point_to_line P (A, B) in
  let d2 := dist_point_to_line P (B, C) in
  let d3 := dist_point_to_line P (C, A) in
  d1 + d2 + d3

noncomputable def altitude_of_equilateral_triangle (a : ℝ) : ℝ :=
  a * sqrt(3) / 2

theorem probability_in_equilateral_triangle (A B C : Point) (P : Point) (hABC : equilateral_triangle A B C) :
  ∃ P, P ∈ triangle_interior A B C → 
  (∑ d in {dist_to_side P A B, dist_to_side P B C, dist_to_side P C A}, d) ≤ altitude_of_equilateral_triangle (dist A B) * 1 / 2 ↔ 
  probability P in A B C = 1 / 4 :=
  sorry

end probability_in_equilateral_triangle_l225_225860


namespace deriv_at_one_equals_cos_one_plus_one_l225_225026

def f (x : ℝ) : ℝ := Real.sin x + Real.log x

theorem deriv_at_one_equals_cos_one_plus_one : (deriv f 1) = Real.cos 1 + 1 := by
  sorry

end deriv_at_one_equals_cos_one_plus_one_l225_225026


namespace min_area_excenter_triangle_l225_225119

def triangleABC (A B C : Point) : Prop := side_length A B = 26 ∧ side_length B C = 28 ∧ side_length A C = 30

def isOnInterior (X A C : Point) : Prop := X ∈ segment A C

def isExcenter (E X : Point) (A B C : Point) : Prop := 
  ∃ Δ ξ, Δ = triangle A B X ∧ ξ = incenter Δ ∧ E = oposite_excenter Δ X

def minimum_area_triangle_e1e2 (A B C X E1 E2 : Point) : Real := 
  ∃ Δ, Δ = triangle A E1 E2 ∧ minimum possible area Δ = 260

theorem min_area_excenter_triangle (A B C X E1 E2 : Point) 
  (hABC : triangleABC A B C) 
  (hX : isOnInterior X A C) 
  (hE1 : isExcenter E1 X A B C)
  (hE2 : isExcenter E2 X B C A) :
  minimum_area_triangle_e1e2 A B C X E1 E2 :=
sorry

end min_area_excenter_triangle_l225_225119


namespace min_value_expression_l225_225012

theorem min_value_expression (x y z : ℝ) : ∃ v, v = 0 ∧ ∀ x y z : ℝ, x^2 + 2 * x * y + 3 * y^2 + 2 * x * z + 3 * z^2 ≥ v := 
by 
  use 0
  sorry

end min_value_expression_l225_225012


namespace committee_count_l225_225674

theorem committee_count :
  ∃ num_committees : ℕ,
  (let physics_male := 3,
       physics_female := 3,
       chemistry_male := 3,
       chemistry_female := 3,
       biology_male := 3,
       biology_female := 3 in
   let total_committees := 
     (choose biology_male 1 * choose biology_female 1) *
     (choose (physics_female + chemistry_female) 3 * choose (physics_male + chemistry_male) 1) + 
     (choose biology_female 2) * 
     (choose (physics_female + chemistry_female) 2 * choose (physics_male + chemistry_male) 2) in
   total_committees = 1755) :=
begin
  use 1755,
  sorry
end

end committee_count_l225_225674


namespace height_of_wall_is_2_l225_225992

-- Definitions and conditions
def brick_length_cm : ℝ := 20
def brick_width_cm : ℝ := 10
def brick_height_cm : ℝ := 7.5

def wall_length_m : ℝ := 29
def wall_width_m : ℝ := 0.75
def number_of_bricks : ℝ := 29000

-- Converting volume of one brick from cm³ to m³
def volume_of_one_brick_m³ : ℝ := (brick_length_cm * brick_width_cm * brick_height_cm) / 1000000

-- Total volume of required bricks
def total_volume_of_bricks_m³ : ℝ := volume_of_one_brick_m³ * number_of_bricks

-- Proof statement
theorem height_of_wall_is_2 :
  ∃ h : ℝ, wall_length_m * h * wall_width_m = total_volume_of_bricks_m³ ∧ h = 2 :=
by
  sorry

end height_of_wall_is_2_l225_225992


namespace total_area_of_removed_triangles_l225_225288

theorem total_area_of_removed_triangles (a b : ℝ)
  (square_side : ℝ := 16)
  (triangle_hypotenuse : ℝ := 8)
  (isosceles_right_triangle : a = b ∧ a^2 + b^2 = triangle_hypotenuse^2) :
  4 * (1 / 2 * a * b) = 64 :=
by
  -- Sketch of the proof:
  -- From the isosceles right triangle property and Pythagorean theorem,
  -- a^2 + b^2 = 8^2 ⇒ 2 * a^2 = 64 ⇒ a^2 = 32 ⇒ a = b = 4√2
  -- The area of one triangle is (1/2) * a * b = 16
  -- Total area of four such triangles is 4 * 16 = 64
  sorry

end total_area_of_removed_triangles_l225_225288


namespace distance_to_weekend_class_l225_225114

theorem distance_to_weekend_class:
  ∃ d v : ℝ, (d = v * (1 / 2)) ∧ (d = (v + 10) * (3 / 10)) → d = 7.5 :=
by
  sorry

end distance_to_weekend_class_l225_225114


namespace triangle_not_necessarily_equilateral_l225_225156

noncomputable theory

open_locale real

-- Definitions of points and triangles
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Reflect point P over the perpendicular bisector of line segment between points Q and R
def reflect (P Q R : Point) : Point := sorry

-- Define the initial triangle ABC
def ABC : Triangle := sorry

-- Define the transformation to obtain A', B', C'
def A' : Point := reflect (reflect ABC.A ABC.B ABC.C) ABC.C ABC.A
def B' : Point := reflect (reflect ABC.B ABC.C ABC.A) ABC.A ABC.B
def C' : Point := reflect (reflect ABC.C ABC.A ABC.B) ABC.B ABC.C

-- Define the resulting triangle A'B'C'
def A'B'C' : Triangle := ⟨A', B', C'⟩

-- Predicate to check if two triangles are congruent
def congruent (T1 T2 : Triangle) : Prop := sorry

-- Predicate to check if a triangle is equilateral
def equilateral (T : Triangle) : Prop := sorry

-- The theorem to prove that if triangle A'B'C' is congruent to triangle ABC,
-- it does not imply that triangle ABC is necessarily equilateral
theorem triangle_not_necessarily_equilateral :
  congruent ABC A'B'C' → ¬ equilateral ABC :=
sorry

end triangle_not_necessarily_equilateral_l225_225156


namespace vector_proof_l225_225420

def vector_a : ℝ × ℝ := (-4, 3)
def vector_b : ℝ × ℝ := (5, 6)

def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem vector_proof
  (a := vector_a)
  (b := vector_b) :
  3 * (vector_magnitude a)^2 - 4 * (dot_product a b) = 83 := by
  sorry

end vector_proof_l225_225420


namespace percentage_seniors_with_cars_is_40_l225_225205

noncomputable def percentage_of_seniors_with_cars 
  (total_students: ℕ) (seniors: ℕ) (lower_grades: ℕ) (percent_cars_all: ℚ) (percent_cars_lower_grades: ℚ) : ℚ :=
  let total_with_cars := percent_cars_all * total_students
  let lower_grades_with_cars := percent_cars_lower_grades * lower_grades
  let seniors_with_cars := total_with_cars - lower_grades_with_cars
  (seniors_with_cars / seniors) * 100

theorem percentage_seniors_with_cars_is_40
  : percentage_of_seniors_with_cars 1800 300 1500 0.15 0.10 = 40 := 
by
  -- Proof is omitted
  sorry

end percentage_seniors_with_cars_is_40_l225_225205


namespace exp_pi_gt_pi_exp_l225_225594

theorem exp_pi_gt_pi_exp (h : Real.pi > Real.exp 1) : Real.exp Real.pi > Real.pi ^ Real.exp 1 := by
  sorry

end exp_pi_gt_pi_exp_l225_225594


namespace area_of_quadrilateral_A_l225_225917

/-- Given a convex quadrilateral ABCD where AB = 6, BB' = 6, BC = 7, CC' = 7, 
    CD = 8, DD' = 8, DA = 9, AA' = 9, and the area of ABCD is 10,
    prove that the area of quadrilateral A'B'C'D' is 50. -/
theorem area_of_quadrilateral_A'B'C'D' (AB BB' BC CC' CD DD' DA AA' : ℕ)
  (area_ABCD : ℕ) (convex : true)
  (h1 : AB = 6) (h2 : BB' = 6) (h3 : BC = 7)
  (h4 : CC' = 7) (h5 : CD = 8) (h6 : DD' = 8)
  (h7 : DA = 9) (h8 : AA' = 9) (h_area : area_ABCD = 10) :
  let A'B'C'D' := 50 in
  A'B'C'D' = 50 :=
by
  sorry

end area_of_quadrilateral_A_l225_225917


namespace part_I_part_II_l225_225916

-- Part (I)
theorem part_I (x a : ℝ) (h_a : a = 3) (h : abs (x - a) + abs (x + 5) ≥ 2 * abs (x + 5)) : x ≤ -1 := 
sorry

-- Part (II)
theorem part_II (a : ℝ) (h : ∀ x : ℝ, abs (x - a) + abs (x + 5) ≥ 6) : a ≥ 1 ∨ a ≤ -11 := 
sorry

end part_I_part_II_l225_225916


namespace determine_integers_a_b_c_l225_225688

theorem determine_integers_a_b_c :
  (∃ (a b c : ℤ), ∀ (x : ℤ), (x - a) * (x - 15) + 4 = (x + b) * (x + c)) →
  (∃ a : ℤ, a = 10 ∨ a = 25) :=
begin
  sorry
end

end determine_integers_a_b_c_l225_225688


namespace find_a_l225_225395

theorem find_a (a : ℝ) :
  (∀ x, f : ℝ → ℝ = λ x, sin x + a * cos x) ∧
  (∀ x, x - 2 * (x - 5 * π / 3) = f (5 * π / 3 - (x - 5 * π / 3))) →
  a = - (sqrt 3) / 3 :=
by
  sorry

end find_a_l225_225395


namespace solution_xyz_uniqueness_l225_225985

theorem solution_xyz_uniqueness (x y z : ℝ) :
  x + y + z = 3 ∧ x^2 + y^2 + z^2 = 3 ∧ x^3 + y^3 + z^3 = 3 → x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end solution_xyz_uniqueness_l225_225985


namespace infinitly_many_natural_numbers_not_prime_power_numerator_l225_225531

def harmonic_sum (n : ℕ) : ℚ := (finset.range (n+1)).sum (λ i, (1 / (i + 1 : ℚ)))

def irreducible_fraction (S : ℚ) : ℚ × ℚ :=
  let ⟨n, d, _, h₁, _⟩ := rat.num_denom S in (n, d)

theorem infinitly_many_natural_numbers_not_prime_power_numerator :
  ∃ᶠ n in filter.at_top, let ⟨A, B⟩ := irreducible_fraction (harmonic_sum n) in
  ¬ (∃ p k : ℕ, nat.prime p ∧ A = p^k) :=
begin
  sorry
end

end infinitly_many_natural_numbers_not_prime_power_numerator_l225_225531


namespace ABCD_cyclic_l225_225904

noncomputable theory

variables {A B C D K L M N : Type*} [quad : quadrilateral A B C D] 
  [circumscribed : circumscribed_quadrilateral A B C D]
  (h1 : ¬ ∠AC = ⊥) 
  (h2 : ∃ K L M N, intersect_angle_bisectors A B C D AC BD K L M N)
  (h3 : cyclic_quadrilateral K L M N)

theorem ABCD_cyclic (A B C D : Type*) [quad : quadrilateral A B C D]
  [circumscribed : circumscribed_quadrilateral A B C D]
  (h1 : ¬ ∠AC = ⊥) 
  (h2 : ∃ K L M N, intersect_angle_bisectors A B C D AC BD K L M N)
  (h3 : cyclic_quadrilateral K L M N) :
  cyclic_quadrilateral A B C D :=
sorry

end ABCD_cyclic_l225_225904


namespace binom_coeff_and_term_l225_225690

variable {x : ℝ}

def binom_expansion_term (n r : ℕ) (a b : ℝ) : ℝ :=
  (choose n r) * a^(n-r) * b^r

theorem binom_coeff_and_term 
  (h : 9.choose 3 = 84)
  (a b : ℝ)
  (term := binom_expansion_term 9 3 (x^2) (-1 / (2*x))) :
  a = 84 ∧ b = -21 / 2 :=
by
  sorry

end binom_coeff_and_term_l225_225690


namespace xiao_wang_fourth_place_l225_225022

section Competition
  -- Define the participants and positions
  inductive Participant
  | XiaoWang : Participant
  | XiaoZhang : Participant
  | XiaoZhao : Participant
  | XiaoLi : Participant

  inductive Position
  | First : Position
  | Second : Position
  | Third : Position
  | Fourth : Position

  open Participant Position

  -- Conditions given in the problem
  variables
    (place : Participant → Position)
    (hA1 : place XiaoWang = First → place XiaoZhang = Third)
    (hA2 : place XiaoWang = First → place XiaoZhang ≠ Third)
    (hB1 : place XiaoLi = First → place XiaoZhao = Fourth)
    (hB2 : place XiaoLi = First → place XiaoZhao ≠ Fourth)
    (hC1 : place XiaoZhao = Second → place XiaoWang = Third)
    (hC2 : place XiaoZhao = Second → place XiaoWang ≠ Third)
    (no_ties : ∀ x y, place x = place y → x = y)
    (half_correct : ∀ p, (p = A → ((place XiaoWang = First ∨ place XiaoZhang = Third) ∧ (place XiaoWang ≠ First ∨ place XiaoZhang ≠ Third)))
                          ∧ (p = B → ((place XiaoLi = First ∨ place XiaoZhao = Fourth) ∧ (place XiaoLi ≠ First ∨ place XiaoZhao ≠ Fourth)))
                          ∧ (p = C → ((place XiaoZhao = Second ∨ place XiaoWang = Third) ∧ (place XiaoZhao ≠ Second ∨ place XiaoWang ≠ Third)))) 

  -- The goal to prove
  theorem xiao_wang_fourth_place : place XiaoWang = Fourth :=
  sorry
end Competition

end xiao_wang_fourth_place_l225_225022


namespace quotient_remainder_division_l225_225746

theorem quotient_remainder_division (x : ℝ) :
  let dividend := x^5 - 21 * x^3 + 9 * x^2 - 16 * x + 8 in
  let divisor := x - 2 in
  let quotient := x^4 + 2 * x^3 - 17 * x^2 - 25 * x - 66 in
  let remainder := -124 in
  dividend = divisor * quotient + remainder :=
by
  sorry

end quotient_remainder_division_l225_225746


namespace length_PQ_proof_l225_225782

noncomputable def parametric_equation_C (α : ℝ) : ℝ × ℝ :=
  let x := -1 + (Real.sqrt 2) * Real.cos α
  let y := 1 + (Real.sqrt 2) * Real.sin α
  (x, y)

noncomputable def line_l (x : ℝ) : ℝ :=
  1 / 2 * (x + 1)

def ray_OM_theta := Real.pi * 3 / 4

def polar_equation_C (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.sin (θ - Real.pi / 4)

def polar_equation_l (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ - 2 * ρ * Real.sin θ + 1 = 0

def length_PQ (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem length_PQ_proof :
  length_PQ (2 * Real.sqrt 2 * Real.sin (ray_OM_theta - Real.pi / 4), 0) ((Real.sqrt 2 / 3, 0)) = 5 * Real.sqrt 2 / 3 :=
  sorry

end length_PQ_proof_l225_225782


namespace businessman_expenditure_l225_225931

theorem businessman_expenditure (P : ℝ) (h1 : P * 1.21 = 24200) : P = 20000 := 
by sorry

end businessman_expenditure_l225_225931


namespace larger_number_l225_225943

theorem larger_number (x y: ℝ) 
  (h1: x + y = 40)
  (h2: x - y = 6) :
  x = 23 := 
by
  sorry

end larger_number_l225_225943


namespace gcd_231_154_l225_225008

def find_gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_231_154 : find_gcd 231 154 = 77 := by
  sorry

end gcd_231_154_l225_225008


namespace race_dead_heat_l225_225252

variable (v_B v_A L x : ℝ)

theorem race_dead_heat (h : v_A = 17 / 14 * v_B) : x = 3 / 17 * L :=
by
  sorry

end race_dead_heat_l225_225252


namespace incorrect_statements_equal_four_l225_225178

-- Definitions for each of the conditions
def prehistoric_horse_height_pred_invalid : Prop :=
  ∀ (x y : ℕ), (x ≠ y → ¬(regression_equation_applies x y))

def orchid_germination_pred_invalid : Prop :=
  ∀ (x y : ℕ), (x ≠ y → ¬(regression_equation_applies x y))

def feed_sales_pred_invalid : Prop :=
  ∀ (x y : ℕ), (x ≠ y → ¬(deterministic_relation x y))

def weight_pred_invalid : Prop :=
  ∀ (x y : ℕ), (x ≠ y → ¬(regression_equation_applies x y))

-- Combine all the conditions
def all_statements_invalid : Prop :=
  prehistoric_horse_height_pred_invalid ∧ 
  orchid_germination_pred_invalid ∧ 
  feed_sales_pred_invalid ∧ 
  weight_pred_invalid

-- The proof statement
theorem incorrect_statements_equal_four (h : all_statements_invalid) : 
  (number_of_incorrect_statements h = 4) :=
by
  sorry

end incorrect_statements_equal_four_l225_225178


namespace sum_of_roots_ln_eq_4_l225_225941
open Real

theorem sum_of_roots_ln_eq_4 (m : ℝ) :
  let f := λ x : ℝ, log (abs (x - 2)) - m
  let roots := { x | f x = 0 }
  ∃ a b : ℝ, a ∈ roots ∧ b ∈ roots ∧ a ≠ b ∧ a + b = 4 :=
by
  sorry

end sum_of_roots_ln_eq_4_l225_225941


namespace transformed_sin_graph_l225_225181

theorem transformed_sin_graph (x : ℝ) : 
  (λ x, sin (2 * (x - π / 4)) + 1) x = (λ x, sin 2 x) x + 1 :=
by
  sorry

end transformed_sin_graph_l225_225181


namespace area_of_trapezoid_PQRS_is_147_l225_225833

variables (P Q R S T : Type) [metric_space T]
variables (area : T → T → T → ℝ)

def trapezoid (PQRS : Prop) := 
  ∃ (P Q R S: T), PQRS = (P, Q, R, S) ∧ 
  (∃ (PR_inter_QS_at_T : Prop), PR_inter_QS_at_T = (P, R, Q, S, T))

-- Conditions
axiom PQ_parallel_RS : ∀ (P Q R S : T), trapezoid (P, Q, R, S)
axiom area_pqt : area P Q T = 75
axiom area_pst : area P S T = 30

-- Question
theorem area_of_trapezoid_PQRS_is_147 :
  ∀ (P Q R S T : T), PQ_parallel_RS P Q R S → area PQRS = 147 := 
by 
  sorry  -- Proof to be filled

end area_of_trapezoid_PQRS_is_147_l225_225833


namespace cyclic_quadrilateral_iff_equal_angles_l225_225865

-- Definitions based on provided conditions
variables {A B C P Q : Type}
variables [IsTriangle A B C] -- Assume a type class that encapsulates the property of a triangle ABC
variables [IsOnBC P] [IsOnBC Q] (P_ne_Q : P ≠ Q)
variables {O1 O2 O3 O4 : Type}
variables [IsCircumcenter O1 A B P] [IsCircumcenter O2 A B Q]
variables [IsCircumcenter O3 A C P] [IsCircumcenter O4 A C Q]
variables (angle_BAP : α) (angle_CAQ : α)

-- The statement to prove
theorem cyclic_quadrilateral_iff_equal_angles :
  IsCyclicQuadrilateral O1 O2 O3 O4 ↔ angle_BAP = angle_CAQ :=
sorry

end cyclic_quadrilateral_iff_equal_angles_l225_225865


namespace resulting_figure_is_rectangle_l225_225399

noncomputable def hyperbola : Set (ℝ × ℝ) :=
  { p | ∃ x, ∃ y, p = (x, y) ∧ x * y = 20 }

noncomputable def circle : Set (ℝ × ℝ) :=
  { p | ∃ x, ∃ y, p = (x, y) ∧ x^2 + y^2 = 50 }

noncomputable def intersections : Set (ℝ × ℝ) :=
  hyperbola ∩ circle

theorem resulting_figure_is_rectangle :
  let points := intersections in
  ∃ P Q R S : ℝ × ℝ, P ∈ points ∧ Q ∈ points ∧ R ∈ points ∧ S ∈ points ∧
  (P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S) ∧
  (dist P Q = dist R S) ∧ (dist P R = dist Q S) ∧ 
  (dist P S = dist Q R) ∧ (angle P Q R = π/2 ∨ angle Q R S = π/2 ∨ angle R S P = π/2 ∨ angle S P Q = π/2)
  :=
sorry

end resulting_figure_is_rectangle_l225_225399


namespace line_skew_to_another_in_plane_l225_225433

variables {Point Line Plane : Type}
variables (l : Line) (α : Plane)
variables (intersects : ∃ p : Point, p ∈ l ∧ p ∈ α)

-- Assuming standard definitions and axioms about lines and planes.
axiom line_intersects_plane_implies_not_parallel_to_plane : 
  (intersects → ∃ k : Line, k ∈ α ∧ k ≠ l ∧ ∀ p : Point, p ∈ k → p ∉ l)

theorem line_skew_to_another_in_plane :
  intersects → ∃ k : Line, k ∈ α ∧ ∀ p : Point, p ∈ k → ¬ (p ∈ l) :=
begin
  sorry
end

end line_skew_to_another_in_plane_l225_225433


namespace seating_arrangement_l225_225711

variable {M I P A : Prop}

def first_fact : ¬ M := sorry
def second_fact : ¬ A := sorry
def third_fact : ¬ M → I := sorry
def fourth_fact : I → P := sorry

theorem seating_arrangement : ¬ M → (I ∧ P) :=
by
  intros hM
  have hI : I := third_fact hM
  have hP : P := fourth_fact hI
  exact ⟨hI, hP⟩

end seating_arrangement_l225_225711


namespace range_of_a_l225_225763

-- Define the function f(x) piecewise
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x >= 2 then (a - 2) * x else -1/8 * x - 1/2

-- Define the condition for the function to be decreasing
theorem range_of_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) → 
  a ≤ 13 / 8 :=
begin
  sorry -- proof to be filled in
end

end range_of_a_l225_225763


namespace rope_rounds_l225_225655

theorem rope_rounds
  (r1 r2 : ℝ)
  (C1 C2 : ℝ)
  (rounds1 : ℕ)
  (length_rope : ℝ)
  (H1 : r1 = 14)
  (H2 : r2 = 20)
  (H3 : C1 = 2 * Real.pi * r1)
  (H4 : C2 = 2 * Real.pi * r2)
  (H5 : rounds1 = 70)
  (H6 : length_rope = rounds1 * C1) :
  length_rope / C2 = 49 := by
  sorry

end rope_rounds_l225_225655


namespace construct_midlines_l225_225768

structure Rectangle :=
(A B C D O : Point)
(is_Rectangle : IsRectangle A B C D)
(center_O : Center A B C D = O)

theorem construct_midlines (r : Rectangle) : 
  ∃ M N : Point, Midline r.A r.B r.O M ∧ Midline r.B r.C r.O N ∧ M = r.O ∧ N = r.O :=
by
  sorry

end construct_midlines_l225_225768


namespace cube_volume_surface_area_value_l225_225693

theorem cube_volume_surface_area_value (x : ℝ) : 
  (∃ s : ℝ, s = (6 * x)^(1 / 3) ∧ 6 * s^2 = 2 * x) → 
  x = 1 / 972 :=
by {
  sorry
}

end cube_volume_surface_area_value_l225_225693


namespace inradius_ABC_l225_225214

-- Define the triangle ABC
variables (A B C : Type) [metric_space A] [metric_space B] [metric_space C] [triangle_ABC : triangle A B C]

-- Conditions given in the problem
def right_angle_at_C (h1 : angle B C A = 90) : Prop := true
def angle_A_60 (h2 : angle C A B = 60) : Prop := true
def AC_10 (h3 : dist A C = 10) : Prop := true

-- The property we need to prove
theorem inradius_ABC:
  ∀ A B C, right_angle_at_C ∧ angle_A_60 ∧ AC_10 → incircle_radius A B C = 5 * (sqrt 3 - 1) :=
by 
  intros,
  sorry

end inradius_ABC_l225_225214


namespace find_larger_number_l225_225563

theorem find_larger_number (a b : ℕ) (h1 : a + b = 96) (h2 : a = b + 12) : a = 54 :=
sorry

end find_larger_number_l225_225563


namespace image_of_image_fixed_l225_225862

universe u

-- Define the finite set S
variable (S : Type u) [fintype S]

-- Define the set of all functions from S to S
def A := S → S

-- Assume f is an element of A
variable (f : A S)

-- Let T be the image of S under f
def T : finset S := finset.image f (finset.univ : finset S)

-- Assume the given condition
axiom h : ∀ (g : A S), g ≠ f → (f ∘ g ∘ f ≠ g ∘ f ∘ g)

-- The theorem to show
theorem image_of_image_fixed : finset.image f T = T :=
sorry

end image_of_image_fixed_l225_225862


namespace assignment_statement_correct_l225_225554

-- Definitions for the conditions:
def cond_A : Prop := ∀ M : ℕ, (M = M + 3)
def cond_B : Prop := ∀ M : ℕ, (M = M + (3 - M))
def cond_C : Prop := ∀ M : ℕ, (M = M + 3)
def cond_D : Prop := true ∧ cond_A ∧ cond_B ∧ cond_C

-- Theorem statement proving the correct interpretation of the assignment is condition B
theorem assignment_statement_correct : cond_B :=
by
  sorry

end assignment_statement_correct_l225_225554


namespace robot_step_into_trap_l225_225120

theorem robot_step_into_trap
  (ε : ℝ) (hε : 0 < ε) :
  ∃ n : ℕ, ∃ k : ℤ, abs ((n : ℝ) * real.sqrt 2 - (k : ℝ)) < ε :=
by
  sorry

end robot_step_into_trap_l225_225120


namespace an_formula_exists_m_n_l225_225846

def Sn (n : ℕ) : ℕ := n * (n + 1) / 2
def an (n : ℕ) : ℕ := if n = 1 then 1 else n
def bn (n : ℕ) : ℝ := an n / 2^n
noncomputable def Tn (n : ℕ) : ℝ := ∑ k in finset.range n, bn (k + 1)

theorem an_formula (n : ℕ) : an n = n := by
  sorry

theorem exists_m_n : ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ Tn n = an m := by
  use 1
  use 2
  sorry

end an_formula_exists_m_n_l225_225846


namespace KH_perpendicular_CD_l225_225094

-- Definition of the convex quadrilateral, with conditions
variables {A B C D M N K H : Type}
variables [PlaneGeometry] -- Assuming we have a PlaneGeometry structure
variable h1 : ConvexQuadrilateral A B C D
variable h2 : ∠A = ∠C
variable h3 : OnLineSegment M A B
variable h4 : OnLineSegment N B C
variable h5 : Parallel MN AD
variable h6 : Length MN = 2 * Length AD
variable h7 : Midpoint K M N
variable h8 : Orthocenter H A B C

-- Theorem statement: KH is perpendicular to CD
theorem KH_perpendicular_CD : Perpendicular KH CD := by
  sorry

end KH_perpendicular_CD_l225_225094


namespace equidistant_points_l225_225486

variables (P : ℝ × ℝ × ℝ)
variables (A B C D : ℝ × ℝ × ℝ)

def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem equidistant_points :
  let A := (10, 0, 0)
  let B := (0, -6, 0)
  let C := (0, 0, 8)
  let D := (0, 0, 0)
  P = (5, -3, 4)
  P = (5, -3, 4) → (dist P A = dist P B) ∧ (dist P A = dist P C) ∧ (dist P A = dist P D) :=
begin
  intros,
  sorry
end

end equidistant_points_l225_225486


namespace distance_point_to_line_l225_225175

/-- The distance from the point (1, -2) to the line x - y = 1 is √2. -/
theorem distance_point_to_line : ∀ (x y : ℝ), (x = 1) ∧ (y = -2) →
  let d := abs (x - y - 1) / real.sqrt (1^2 + (-1)^2) 
  in d = real.sqrt 2 :=
by
  intros x y h
  obtain ⟨hx, hy⟩ := h
  sorry

end distance_point_to_line_l225_225175
