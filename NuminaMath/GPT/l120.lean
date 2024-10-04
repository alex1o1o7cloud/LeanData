import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.ConicSections
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field
import Mathlib.Algebra.Geometry
import Mathlib.Algebra.Group
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.IntermediateValueTheorem
import Mathlib.Analysis.Calculus.Limits
import Mathlib.Analysis.Limits
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Default
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Analytic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.GroupTheory.GroupAction.Basic
import Mathlib.GroupTheory.Hom
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.Init.Data.List.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.MeasureTheory.Integral.SetIntegral
import Mathlib.NumberTheory.Lcm
import Mathlib.Probability
import Mathlib.Probability.Chebyshev
import Mathlib.Tactic
import Mathlib.Tactic.NormNum
import Mathlib.Textbook
import Mathlib.Topology.Algebra.Order
import ProbTheory.ProbabilityMassFunction
import Real

namespace perimeter_of_square_C_l120_120453

theorem perimeter_of_square_C (s_A s_B s_C : ℕ) (hpA : 4 * s_A = 16) (hpB : 4 * s_B = 32) (hC : s_C = s_A + s_B - 2) :
  4 * s_C = 40 := 
by
  sorry

end perimeter_of_square_C_l120_120453


namespace ratio_a_over_c_l120_120648

variables {a b c x1 x2 : Real}
variables (h1 : x1 + x2 = -a) (h2 : x1 * x2 = b) (h3 : b = 2 * a) (h4 : c = 4 * b)
           (ha_nonzero : a ≠ 0) (hb_nonzero : b ≠ 0) (hc_nonzero : c ≠ 0)

theorem ratio_a_over_c : a / c = 1 / 8 :=
by
  have hc_eq : c = 8 * a := by
    rw [h4, h3]
    simp
  rw [hc_eq]
  field_simp [ha_nonzero]
  norm_num
  sorry -- additional steps if required

end ratio_a_over_c_l120_120648


namespace decreasing_exponential_range_l120_120530

theorem decreasing_exponential_range {a : ℝ} (h : ∀ x y : ℝ, x < y → (a + 1)^x > (a + 1)^y) : -1 < a ∧ a < 0 :=
sorry

end decreasing_exponential_range_l120_120530


namespace DE_value_l120_120903

theorem DE_value {AG GF FC HJ DE : ℝ} (h1 : AG = 2) (h2 : GF = 13) 
  (h3 : FC = 1) (h4 : HJ = 7) : DE = 2 * Real.sqrt 22 :=
sorry

end DE_value_l120_120903


namespace scientific_notation_population_l120_120240

theorem scientific_notation_population :
    ∃ (a b : ℝ), (b = 5 ∧ 1412.60 * 10 ^ 6 = a * 10 ^ b ∧ a = 1.4126) :=
sorry

end scientific_notation_population_l120_120240


namespace polynomial_value_bound_l120_120510

-- Define conditions and the polynomial
variables {a : ℕ → ℤ} {x : ℕ → ℤ} {n : ℕ}
noncomputable def P (x : ℤ) : ℤ := x^n + a 1 * x^(n-1) + a 2 * x^(n-2) + ... + a n

-- Statement of the theorem
theorem polynomial_value_bound :
  (∀ i j : ℕ, i < n → i < j → x i < x j) → 
  ∃ j ∈ Finset.range (n + 1), |P (x j)| ≥ Nat.factorial n / 2^n :=
sorry -- proof omitted for brevity

end polynomial_value_bound_l120_120510


namespace monotonicity_intervals_range_of_a_l120_120815

section
variable {a : ℝ}
def f (x : ℝ) : ℝ := a * x + x * log x

theorem monotonicity_intervals (a : ℝ): -- Part (I)
  f' : ℝ := log x
  (f' (1) = 0 → a = -1) →
  ∀ x, 0 < x ∧ x < 1 → f' x < 0 ∧ 1 < x → f' x > 0 :=
sorry

theorem range_of_a (a : ℝ): -- Part (II)
  (∀x: ℝ, x > 1 → f(x) < x^2) →
  a ≤ 1 :=
sorry
end

end monotonicity_intervals_range_of_a_l120_120815


namespace find_correct_function_l120_120756

-- Definitions of functions
def f1 (x : ℝ) : ℝ := 1 / x
def f2 (x : ℝ) : ℝ := Real.exp (-x)
def f3 (x : ℝ) : ℝ := -x^3
def f4 (x : ℝ) : ℝ := Real.log x

-- Properties to check

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_monotonically_decreasing (f : ℝ → ℝ) : Prop := ∀ ⦃x y⦄, x < y → f x > f y

-- Lean 4 statement
theorem find_correct_function :
  is_odd f1 ∧ is_monotonically_decreasing f1 = false ∧
  is_odd f2 ∧ is_monotonically_decreasing f2 = false ∧
  is_odd f3 ∧ is_monotonically_decreasing f3 = true ∧
  is_odd f4 ∧ is_monotonically_decreasing f4 = false :=
sorry

end find_correct_function_l120_120756


namespace bottom_row_bricks_l120_120541

theorem bottom_row_bricks {x : ℕ} 
  (c1 : ∀ i, i < 5 → (x - i) > 0)
  (c2 : x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 100) : 
  x = 22 := 
by
  sorry

end bottom_row_bricks_l120_120541


namespace mn_value_l120_120521

theorem mn_value (m n : ℤ) (h1 : m = n + 2) (h2 : 2 * m + n = 4) : m * n = 0 := by
  sorry

end mn_value_l120_120521


namespace powers_of_two_not_powers_of_eight_l120_120867

theorem powers_of_two_not_powers_of_eight :
  {n : ℤ | ∃ k : ℤ, 0 ≤ k ∧ 2^k = n ∧ n < 1000000000} ∩ {n : ℤ | ∃ m : ℤ, 0 ≤ m ∧ 8^m = n} = ∅ →
  #{n : ℤ | ∃ k : ℤ, 0 ≤ k ∧ 2^k = n ∧ n < 1000000000} = 30 →
  #{n : ℤ | ∃ m : ℤ, 0 ≤ m ∧ 8^m = n ∧ n < 1000000000} = 10 →
  #{n : ℤ | ∃ k : ℤ, 0 ≤ k ∧ 2^k = n ∧ n < 1000000000 ∧ ∀ m, 8^m ≠ n} = 20 :=
by sorry

end powers_of_two_not_powers_of_eight_l120_120867


namespace books_sold_on_Tuesday_l120_120409

theorem books_sold_on_Tuesday 
  (initial_stock : ℕ)
  (books_sold_Monday : ℕ)
  (books_sold_Wednesday : ℕ)
  (books_sold_Thursday : ℕ)
  (books_sold_Friday : ℕ)
  (books_not_sold : ℕ) :
  initial_stock = 800 →
  books_sold_Monday = 60 →
  books_sold_Wednesday = 20 →
  books_sold_Thursday = 44 →
  books_sold_Friday = 66 →
  books_not_sold = 600 →
  ∃ (books_sold_Tuesday : ℕ), books_sold_Tuesday = 10
:= by
  intros h_initial h_monday h_wednesday h_thursday h_friday h_not_sold
  sorry

end books_sold_on_Tuesday_l120_120409


namespace solve_problem_l120_120616

-- Define the points and the rectangle
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 6 }
def B : Point := { x := 8, y := 6 }
def C : Point := { x := 8, y := 0 }
def D : Point := { x := 0, y := 0 }
def E : Point := { x := 6, y := 6 }
def F : Point := { x := 3, y := 0 }
def G : Point := { x := 8, y := 4 }

-- Function to calculate slope of a line
def slope (P1 P2 : Point) : ℝ :=
  (P2.y - P1.y) / (P2.x - P1.x)

-- Define the equations of the lines based on the previously calculated slopes in the solution
def line_AG : ℝ → ℝ := λ x => -0.25 * x + 6
def line_AC : ℝ → ℝ := λ x => -0.75 * x + 6
def line_EF : ℝ → ℝ := λ x => 2 * x - 12

-- Use the slope values and equations to find intersections P and Q
def P : Point := { x := 72 / 11, y := (-0.75) * (72 / 11) + 6 }
def Q : Point := { x := 72 / 9, y := (-0.25) * (72 / 9) + 6 }

-- Calculate the distance between two points
def distance (P1 P2 : Point) : ℝ :=
  real.sqrt((P2.x - P1.x) ^ 2 + (P2.y - P1.y) ^ 2)

-- The length of EF and PQ
def length_EF : ℝ := distance E F
def length_PQ : ℝ := distance P Q

-- The ratio PQ/EF
def ratio_PQ_EF : ℝ := length_PQ / length_EF

theorem solve_problem : ratio_PQ_EF = 16 / (33 * real.sqrt 5) := by
  sorry

end solve_problem_l120_120616


namespace number_of_roots_l120_120101

noncomputable theory
open Real

def domain (x : ℝ) := abs x ≤ sqrt 14
def equation (x : ℝ) := sin x - cos (2 * x) = 0

theorem number_of_roots : 
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, domain x) ∧ (∀ x ∈ xs, equation x) ∧ xs.card = 6 := 
by sorry

end number_of_roots_l120_120101


namespace determine_n_l120_120200

noncomputable def S : ℕ → ℝ := sorry -- define arithmetic series sum
noncomputable def a_1 : ℝ := sorry -- define first term
noncomputable def d : ℝ := sorry -- define common difference

axiom S_6 : S 6 = 36
axiom S_n {n : ℕ} (h : n > 0) : S n = 324
axiom S_n_minus_6 {n : ℕ} (h : n > 6) : S (n - 6) = 144

theorem determine_n (n : ℕ) (h : n > 0) : n = 18 := by {
  sorry
}

end determine_n_l120_120200


namespace roque_total_time_l120_120564

def travel_time (day: String) (method: String) (rain: Bool) (extra_time: Nat) : Nat :=
  match (day, method, rain, extra_time) with
  | ("Monday", "walk", true, 0) => 2 * (2 + 2 * (2 / 10))
  | ("Tuesday", "bike", _, 0) => 2 * 1
  | ("Wednesday", "walk", _, 30) => 2 * (2 + 30 / 60)
  | ("Thursday", "walk", true, 0) => 2 * (2 + 2 * (2 / 10))
  | ("Friday", "bike", _, 15) => 2 * (1 + 15 / 60)
  | _ => 0

def total_week_time : Nat :=
  4.8 * 1 + 2 * 1 + 5 * 1 + 4.8 * 1 + 2.5 * 1

theorem roque_total_time : total_week_time = 19.1 := 
by rfl


end roque_total_time_l120_120564


namespace find_common_students_l120_120151

theorem find_common_students
  (total_english : ℕ)
  (total_math : ℕ)
  (difference_only_english_math : ℕ)
  (both_english_math : ℕ) :
  total_english = both_english_math + (both_english_math + 10) →
  total_math = both_english_math + both_english_math →
  difference_only_english_math = 10 →
  total_english = 30 →
  total_math = 20 →
  both_english_math = 10 :=
by
  intros
  sorry

end find_common_students_l120_120151


namespace find_area_of_triangle_l120_120536

variables {A B C : ℝ} {a b c S : ℝ}

-- Condition: Given the relationship between cosines and sines in \triangle ABC
axiom cosine_sine_relationship (h : (cos A - 2 * cos C) / cos B = (2 * c - a) / b) :
  sin C / sin A = 2

-- Prove: Given cos B = 1/4 and b = 2, find the area S of \triangle ABC is \sqrt{15} / 4
theorem find_area_of_triangle (h1 : cos B = 1 / 4) (h2 : b = 2) :
  S = sqrt 15 / 4 :=
  sorry

end find_area_of_triangle_l120_120536


namespace GoldenRabbitCards_count_l120_120885

theorem GoldenRabbitCards_count :
  let total_cards := 10000
  let non_golden_combinations := 8 * 8 * 8 * 8
  let golden_cards := total_cards - non_golden_combinations
  golden_cards = 5904 :=
by
  let total_cards := 10000
  let non_golden_combinations := 8 * 8 * 8 * 8
  let golden_cards := total_cards - non_golden_combinations
  sorry

end GoldenRabbitCards_count_l120_120885


namespace polygon_interior_angles_540_implies_5_sides_l120_120304

theorem polygon_interior_angles_540_implies_5_sides (n : ℕ) :
  (n - 2) * 180 = 540 → n = 5 :=
by
  sorry

end polygon_interior_angles_540_implies_5_sides_l120_120304


namespace a5_value_l120_120948

def sequence_sum (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (Finset.range n).sum a

theorem a5_value (a : ℕ → ℤ) (h : ∀ n : ℕ, 0 < n → sequence_sum n a = (1 / 2 : ℚ) * (a n : ℚ) + 1) :
  a 5 = 2 := by
  sorry

end a5_value_l120_120948


namespace time_for_first_half_is_15_l120_120603

-- Definitions of the conditions in Lean
def floors := 20
def time_per_floor_next_5 := 5
def time_per_floor_final_5 := 16
def total_time := 120

-- Theorem statement
theorem time_for_first_half_is_15 :
  ∃ T, (T + (5 * time_per_floor_next_5) + (5 * time_per_floor_final_5) = total_time) ∧ (T = 15) :=
by
  sorry

end time_for_first_half_is_15_l120_120603


namespace ratio_a_c_l120_120654

theorem ratio_a_c {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : ∀ x : ℝ, x^2 + a * x + b = 0 → 2 * x^2 + 2 * a * x + 2 * b = 0)
  (h2 : ∀ x : ℝ, x^2 + b * x + c = 0 → x^2 + b * x + c = 0) :
  a / c = 1 / 8 :=
by
  sorry

end ratio_a_c_l120_120654


namespace probability_X_lt_mu_minus_2sigma_l120_120347

noncomputable def X (μ σ : ℝ) : ℝ → ℝ := sorry

def normal_dist (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  ∀ x, X x ∼ N(μ, σ^2)

def probability_X_gt_mu_minus_sigma (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  P (λ x, X x > μ - σ) = 0.8413

theorem probability_X_lt_mu_minus_2sigma (μ σ : ℝ) (X : ℝ → ℝ)
  (h1 : normal_dist μ σ X) (h2 : probability_X_gt_mu_minus_sigma μ σ X) :
  P (λ x, X x < μ - 2σ) = 0.0228 :=
sorry

end probability_X_lt_mu_minus_2sigma_l120_120347


namespace power_equality_l120_120525

theorem power_equality (n : ℝ) : (9:ℝ)^4 = (27:ℝ)^n → n = (8:ℝ) / 3 :=
by
  sorry

end power_equality_l120_120525


namespace determine_a_b_no_real_roots_l120_120851

-- Part 1
theorem determine_a_b 
  (a b : ℝ) 
  (x : ℂ) 
  (hx : x = 1 - (complex.I * real.sqrt 3)) 
  (h : x / (a : ℂ) + (b : ℂ) / x = 1) : 
  a = 2 ∧ b = 2 := 
sorry

-- Part 2
theorem no_real_roots 
  (a b : ℝ) 
  (h_pos : a > 0) 
  (h_ratio : b / a > 1 / 4) : 
  ¬ ∃ x : ℝ, x / a + b / x = 1 := 
sorry

end determine_a_b_no_real_roots_l120_120851


namespace dolls_count_l120_120753

theorem dolls_count (lisa_dolls : ℕ) (vera_dolls : ℕ) (sophie_dolls : ℕ) (aida_dolls : ℕ)
  (h1 : vera_dolls = 2 * lisa_dolls)
  (h2 : sophie_dolls = 2 * vera_dolls)
  (h3 : aida_dolls = 2 * sophie_dolls)
  (hl : lisa_dolls = 20) :
  aida_dolls + sophie_dolls + vera_dolls + lisa_dolls = 300 :=
by
  sorry

end dolls_count_l120_120753


namespace max_volume_small_cube_l120_120036

theorem max_volume_small_cube (a : ℝ) (h : a = 2) : (a^3 = 8) := by
  sorry

end max_volume_small_cube_l120_120036


namespace number_of_roots_l120_120114

noncomputable def roots_equation_count : ℝ :=
  let interval := Icc (-real.sqrt 14) (real.sqrt 14)
  ∑ x in interval, (if sqrt (14 - x^2) * (sin x - cos (2 * x)) = 0 then 1 else 0)

theorem number_of_roots : roots_equation_count = 6 := by {
  sorry
}

end number_of_roots_l120_120114


namespace count_visible_factor_numbers_200_to_250_l120_120384

def is_visible_factor_number (n : ℕ) : Prop :=
  let digits := (toString n).toList
  let non_zero_digits := digits.filter (λ d => d ≠ '0')
  (non_zero_digits.all (λ d => n % (d.toNat - '0'.toNat) = 0))

def visible_factor_numbers_in_range (m n : ℕ) : List ℕ :=
  (List.range' m (n - m + 1)).filter is_visible_factor_number

theorem count_visible_factor_numbers_200_to_250 : 
  visible_factor_numbers_in_range 200 250 = List.range' 200 22 
:=
  sorry

end count_visible_factor_numbers_200_to_250_l120_120384


namespace polynomial_root_count_l120_120582

-- Define polynomials P, Q, and R
variable (P Q R : ℂ[X])
variable (P_deg : P.degree = 2) (Q_deg : Q.degree = 4) (R_deg : R.degree = 7)
variable (P_const : P.coeff 0 = 3) (Q_const : Q.coeff 0 = 1) (R_const : R.coeff 0 = 4)

-- The theorem to prove
theorem polynomial_root_count :
  let f := (P^2 + Q - R) in
  f.degree = 7 → ∃ s : finset ℂ, (∀ z ∈ s, f.eval z = 0) ∧ s.card = 7 := sorry

end polynomial_root_count_l120_120582


namespace quadratic_real_roots_condition_sufficient_l120_120634

theorem quadratic_real_roots_condition_sufficient (m : ℝ) : (m < 1 / 4) → ∃ x : ℝ, x^2 + x + m = 0 :=
by
  sorry

end quadratic_real_roots_condition_sufficient_l120_120634


namespace power_evaluation_l120_120450

theorem power_evaluation : (27 : ℝ)^(5/3) = 243 := 
by {
  have h1 : (27 : ℝ) = (3^3 : ℝ), by norm_num,
  rw [h1],
  have h2 : (3^3 : ℝ)^(5/3) = (3 : ℝ)^(3 * (5/3)), by norm_num,
  rw [h2],
  have h3 : (3 : ℝ)^(3 * (5/3)) = (3 : ℝ)^5, by norm_num,
  rw [h3],
  norm_num
}

end power_evaluation_l120_120450


namespace trig_values_on_line_l120_120845

theorem trig_values_on_line (a : ℝ) (h : a ≠ 0) :
  let x := 12 * a
      y := 5 * a
      r := real.sqrt ((12 * a)^2 + (5 * a)^2)
  in (5 * x = 12 * y) →
    (if a > 0 then 
     real.sin (real.atan2 y x) = 5 / 13 ∧ 
     real.cos (real.atan2 y x) = 12 / 13 ∧ 
     real.tan (real.atan2 y x) = 5 / 12
    else 
     real.sin (real.atan2 y x) = - (5 / 13) ∧ 
     real.cos (real.atan2 y x) = - (12 / 13) ∧ 
     real.tan (real.atan2 y x) = 5 / 12) := by
  sorry

end trig_values_on_line_l120_120845


namespace correct_propositions_l120_120963

variables {Line Plane : Type} (m n : Line) (α β γ : Plane)

-- Distinct lines and planes
axiom distinct_lines : m ≠ n
axiom distinct_planes_αβγ : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Propositions
axiom prop_1 : (α ∥ β → α ∥ γ) → β ∥ γ
axiom prop_3 : (m ⊥ α → m ∥ β) → α ⊥ β

theorem correct_propositions : prop_1 ∧ prop_3 := by
  sorry

end correct_propositions_l120_120963


namespace number_of_roots_l120_120103

noncomputable theory
open Real

def domain (x : ℝ) := abs x ≤ sqrt 14
def equation (x : ℝ) := sin x - cos (2 * x) = 0

theorem number_of_roots : 
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, domain x) ∧ (∀ x ∈ xs, equation x) ∧ xs.card = 6 := 
by sorry

end number_of_roots_l120_120103


namespace eccentricity_of_ellipse_equation_of_ellipse_l120_120167

noncomputable def midpoint (A B : ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem eccentricity_of_ellipse (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a) 
  (h₃ : ∃ x1 x2 : ℝ, (x1 + x2)/2 = 2 ∧ (-x1 + 3 + -x2 + 3)/2 = 1) :
  sqrt (1 - b^2 / a^2) = sqrt 2 / 2 :=
sorry

theorem equation_of_ellipse (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a)
  (h₃ : ∃ x1 x2 : ℝ, (x1 + x2)/2 = 2 ∧ (-x1 + 3 + -x2 + 3)/2 = 1) 
  (h₄ : (2 * b^2 - 4 * b * sqrt (1 - b^2 / a^2) + 3 - sqrt (1 - b^2 / a^2)^2 * 6 = 6)) :
  (0 < b) → (b < a) → ellipse (18) (9) := 
sorry

end eccentricity_of_ellipse_equation_of_ellipse_l120_120167


namespace sequence_formula_l120_120394

open Nat

theorem sequence_formula (a : ℕ → ℝ) :
  (a 1 = 1) ∧ 
  (a 2 = 1/2) ∧ 
  (∀ n > 2, a n = 1/n * (∑ i in finset.range(n-1), a i)) →
  ∀ n ≥ 2, a n = ∑ i in finset.range(n-1), (↑((-1)^(i+2)) / (↑(i+2)!)) :=
by
  sorry

end sequence_formula_l120_120394


namespace tetrahedron_paths_l120_120243

/--
  Consider a tetrahedron with vertices labeled V1, V2, V3, and V4.
  We want to prove that the number of paths from V1 to V3, traversing exactly 2 edges, is 2.
-/
def number_of_paths (V : Type) [Fintype V] [DecidableEq V] (edges : V → V → Prop) (V1 V3 : V) : ℕ :=
  Fintype.card {p : V × V // edges V1 p.1 ∧ edges p.1 V3 ∧ p.1 ≠ V1 ∧ p.1 ≠ V3}

theorem tetrahedron_paths (V1 V2 V3 V4 : Type) :
  let V := {V1, V2, V3, V4}
  let edges : V → V → Prop := λ u v, (u ≠ v) ∧ (u = V1 ∨ u = V2 ∨ u = V3 ∨ u = V4) ∧ (v = V1 ∨ v = V2 ∨ v = V3 ∨ v = V4)
  number_of_paths V edges V1 V3 = 2 :=
by
  sorry

end tetrahedron_paths_l120_120243


namespace first_player_wins_with_optimal_play_l120_120737

-- Define the conditions of the chessboard game
structure ChessboardGame :=
  (initial_position : ℕ × ℕ)
  (adjacent : ℕ × ℕ → Set (ℕ × ℕ))
  (not_occupied : Set (ℕ × ℕ))

axiom adjacent_def (x y : ℕ × ℕ) : adjacent x y ↔ (abs (x.fst - y.fst) = 1 ∧ x.snd = y.snd) ∨ (x.fst = y.fst ∧ abs (x.snd - y.snd) = 1)

-- Define the winning condition for the first player under the given strategy
def first_player_wins (game : ChessboardGame) : Prop :=
  ∃ (strategy : ℕ × ℕ → ℕ × ℕ), ∀ position ∈ game.not_occupied, 
  (∀ move ∈ game.adjacent position, move ∉ game.not_occupied) → 
  ∃ next_move ∈ game.adjacent position, next_move ∉ game.not_occupied 
  ∧ for all subsequent moves, the first player can adhere to the strategy.

theorem first_player_wins_with_optimal_play (game : ChessboardGame) :
  first_player_wins game :=
sorry

end first_player_wins_with_optimal_play_l120_120737


namespace chef_cooked_additional_wings_l120_120730

def total_chicken_wings_needed (friends : ℕ) (wings_per_friend : ℕ) : ℕ :=
  friends * wings_per_friend

def additional_chicken_wings (total_needed : ℕ) (already_cooked : ℕ) : ℕ :=
  total_needed - already_cooked

theorem chef_cooked_additional_wings :
  let friends := 4
  let wings_per_friend := 4
  let already_cooked := 9
  additional_chicken_wings (total_chicken_wings_needed friends wings_per_friend) already_cooked = 7 := by
  sorry

end chef_cooked_additional_wings_l120_120730


namespace find_divisor_l120_120637

/-- Given a division problem where the dividend is 131, the quotient is 9, and the remainder is 5,
    this theorem proves that the divisor is 14. -/
theorem find_divisor : 
  ∃ d : ℕ, let dividend := 131 in let quotient := 9 in let remainder := 5 in
  dividend = (d * quotient + remainder) :=
begin
  use 14,
  let dividend := 131,
  let quotient := 9,
  let remainder := 5,
  simp [dividend, quotient, remainder],
  norm_num
end

end find_divisor_l120_120637


namespace range_of_sum_l120_120144

theorem range_of_sum (x y : ℝ) (h : x^2 + x + y^2 + y = 0) : 
  -2 ≤ x + y ∧ x + y ≤ 0 :=
sorry

end range_of_sum_l120_120144


namespace number_of_balls_sold_l120_120975

-- Definitions from conditions
def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 120
def loss : ℕ := 5 * cost_price_per_ball

-- Mathematically equivalent proof statement
theorem number_of_balls_sold (n : ℕ) (h : n * cost_price_per_ball - selling_price = loss) : n = 11 :=
  sorry

end number_of_balls_sold_l120_120975


namespace base8_to_base6_conversion_l120_120436

theorem base8_to_base6_conversion : 
  ∀ (n : ℕ), (n = 753) → n' = 2135 :=
begin
  intros,
  sorry
end

end base8_to_base6_conversion_l120_120436


namespace ratio_of_bath_to_blowdry_l120_120602

theorem ratio_of_bath_to_blowdry (t_total t_b v d : ℕ) (h_t_total : t_total = 60) (h_t_b : t_b = 20) (h_v : v = 6) (h_d : d = 3) :
  let t_bd := t_total - (t_b + d / v * 60) in
  t_bd / t_b = 1 / 2 :=
by
  sorry

end ratio_of_bath_to_blowdry_l120_120602


namespace gianna_savings_l120_120009

-- Definition for daily savings and number of days in a year
def daily_savings := 39
def days_in_year := 365

-- Theorem to prove that the total savings at the end of the year is $14,235
theorem gianna_savings:
  daily_savings * days_in_year = 14235 := 
by
  simp [daily_savings, days_in_year]
  -- multiplication would be used here.
  sorry

end gianna_savings_l120_120009


namespace max_disconnected_cells_correct_l120_120051

noncomputable def max_disconnected_cells {n : ℕ} (h1 : Odd n) (h2 : n ≥ 3) : ℕ :=
  (n + 1) ^ 2 / 4 + 1

theorem max_disconnected_cells_correct (n : ℕ) (h1 : Odd n) (h2 : n ≥ 3) :
  ∃ m, m = max_disconnected_cells h1 h2 :=
begin
  use (n + 1) ^ 2 / 4 + 1,
  refl,
end

end max_disconnected_cells_correct_l120_120051


namespace sum_of_coefficients_l120_120765

theorem sum_of_coefficients (x y z : ℤ) (h : x = 1 ∧ y = 1 ∧ z = 1) :
    (x - 2 * y + 3 * z) ^ 12 = 4096 :=
by
  sorry

end sum_of_coefficients_l120_120765


namespace max_prob_xi_expected_value_xi_l120_120896

noncomputable def cards : List ℕ := [1, 1, 1, 2, 2, 2, 3, 3]

def xi_values := {2, 3, 4, 5, 6}

def prob_xi (xi : ℕ) : ℚ := 
  (match xi with
  | 2 => 3 * 3 / (8 * 8)
  | 3 => 2 * 3 * 3 / (8 * 8)
  | 4 => (3 * 3 + 2 * 3 * 2) / (8 * 8)
  | 5 => 2 * 3 * 2 / (8 * 8)
  | 6 => 2 * 2 / (8 * 8)
  | _ => 0)

def expected_xi : ℚ := 
  2 * (3 * 3 / 64) +
  3 * (2 * 3 * 3 / 64) +
  4 * ((3 * 3 + 2 * 3 * 2) / 64) +
  5 * (2 * 3 * 2 / 64) +
  6 * (2 * 2 / 64)

theorem max_prob_xi : ∀ n ∈ xi_values, prob_xi n ≤ prob_xi 4 := by
  intros n hn
  sorry

theorem expected_value_xi : expected_xi = 15 / 4 := by
  sorry

end max_prob_xi_expected_value_xi_l120_120896


namespace min_cells_marked_l120_120330

/-- The minimum number of cells that need to be marked in a 50x50 grid so
each 1x6 vertical or horizontal strip has at least one marked cell is 416. -/
theorem min_cells_marked {n : ℕ} : n = 416 → 
  (∀ grid : Fin 50 × Fin 50, ∃ cells : Finset (Fin 50 × Fin 50), 
    (∀ (r c : Fin 50), (r = 6 * i + k ∨ c = 6 * i + k) →
      (∃ (cell : Fin 50 × Fin 50), cell ∈ cells)) →
    cells.card = n) := 
sorry

end min_cells_marked_l120_120330


namespace min_marked_cells_l120_120332

-- Define the dimensions of the grid
def grid_width : ℕ := 50
def grid_height : ℕ := 50
def strip_width : ℕ := 6

-- Define the total number of strips
def total_strips : ℕ := (grid_width * (grid_height / strip_width)) + (grid_height * (grid_width / strip_width))

-- Statement of the theorem
theorem min_marked_cells : total_strips = 416 :=
by
  Sorry -- Proof goes here 

end min_marked_cells_l120_120332


namespace A_share_of_annual_gain_l120_120406

variables (x : ℝ) (annual_gain : ℝ)

-- Conditions
def A_investment_ratio := x * 12
def B_investment_ratio := x * 2 * 6
def C_investment_ratio := x * 3 * 4
def total_investment_ratio := A_investment_ratio + B_investment_ratio + C_investment_ratio

def annual_gain_value : ℝ := 21000

-- Proof statement
theorem A_share_of_annual_gain : (A_investment_ratio / total_investment_ratio) * annual_gain_value = 7000 :=
by
  sorry

end A_share_of_annual_gain_l120_120406


namespace total_spent_l120_120354

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_friends : ℝ := 4
def discounted_price (orig_price : ℝ) (discount : ℝ) : ℝ := orig_price * discount
def total_cost (num_friends : ℝ) (unit_cost : ℝ) : ℝ := num_friends * unit_cost

theorem total_spent :
  total_cost number_of_friends (discounted_price original_price discount_rate) = 40 :=
by
  simp [total_cost, discounted_price, original_price, discount_rate, number_of_friends]
  norm_num
  sorry

end total_spent_l120_120354


namespace parametric_eqn_and_max_sum_l120_120166

noncomputable def polar_eq (ρ θ : ℝ) := ρ^2 = 4 * ρ * (Real.cos θ + Real.sin θ) - 6

theorem parametric_eqn_and_max_sum (θ : ℝ):
  (∃ (x y : ℝ), (2 + Real.sqrt 2 * Real.cos θ, 2 + Real.sqrt 2 * Real.sin θ) = (x, y)) ∧
  (∃ (θ : ℝ), θ = Real.pi / 4 → (3, 3) = (3, 3) ∧ 6 = 6) :=
by {
  sorry
}

end parametric_eqn_and_max_sum_l120_120166


namespace norm_w_eq_sqrt_two_l120_120945

-- Define the conditions as functions in Lean
variable (s : ℝ) (w : ℂ)
hypothesis h1 : |s| < 3
hypothesis h2 : w + (2 / w) = s

theorem norm_w_eq_sqrt_two : |w| = √2 :=
by
  assume s : ℝ
  assume w : ℂ
  assume h1 : |s| < 3
  assume h2 : w + (2 / w) = s
  sorry

end norm_w_eq_sqrt_two_l120_120945


namespace sequence_periodic_from_step_l120_120546

def sequence (a : ℕ) (h : a % 2 = 1) : ℕ → ℕ 
| 0     := arbitrary ℕ 
| (n+1) := if sequence n % 2 = 0 then sequence n / 2 else a + sequence n

theorem sequence_periodic_from_step (a : ℕ) (h : a % 2 = 1) (u0 : ℕ): ∃ N, ∀ n ≥ N, ∃ k, sequence a h u0 (n + k) = sequence a h u0 n := 
sorry

end sequence_periodic_from_step_l120_120546


namespace range_of_m_l120_120477

theorem range_of_m (m : ℝ)
  (h₁ : (m^2 - 4) ≥ 0)
  (h₂ : (4 * (m - 2)^2 - 16) < 0) :
  1 < m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l120_120477


namespace number_of_knights_l120_120707

-- Define the properties of an individual in the context of knights and liars
inductive Person
| knight : Person
| liar : Person

-- Define the configuration of 80 people around the circular table
def CircularTable : Type := Fin 80 → Person

-- Define the statement that each person declares about the 11 immediate people after them
def declaration (table : CircularTable) (i : Fin 80) : Prop :=
  let next_11 := List.map (λ j => table (⟨(i + j) % 80, sorry⟩)) (List.range 1 12)
  9 ≤ List.count (λ p => p = Person.liar) next_11

-- Definition that the whole table satisfies the declaration constraint
def validTable (table : CircularTable) : Prop :=
  ∀ i : Fin 80, declaration table i

-- Definition to count the number of knights
def countKnights (table : CircularTable) : ℕ :=
  Finset.univ.count (λ i => table i = Person.knight)

theorem number_of_knights {table : CircularTable} (h : validTable table) :
  countKnights table = 20 :=
sorry

end number_of_knights_l120_120707


namespace min_elements_in_T_l120_120910

-- Definitions of the sets S and T
def S : set (ℤ × ℤ) := {p | 1 ≤ p.fst ∧ p.fst ≤ 5 ∧ 1 ≤ p.snd ∧ p.snd ≤ 5}

-- A predicate for T satisfying the condition
def good_T (T : set (ℤ × ℤ)) : Prop :=
  ∀ P ∈ S, ∃ Q ∈ T, P ≠ Q ∧ ¬∃ R, R ∈ S ∧ (R ≠ P ∧ R ≠ Q) ∧ collinear ({P, Q, R} : set (ℤ × ℤ))

#check collinear -- checking if collinear is in the library, otherwise define it

-- The main statement
theorem min_elements_in_T : ∃ T : set (ℤ × ℤ), good_T T ∧ T.finite ∧ T.card = 2 :=
sorry

end min_elements_in_T_l120_120910


namespace chromium_percentage_l120_120901

theorem chromium_percentage (x : ℝ) :
  (15 * x / 100 + 30 * 8 / 100 = 45 * 9.333333333333334 / 100) → x = 12 :=
by
  intro h,
  sorry

end chromium_percentage_l120_120901


namespace arithmetic_sequence_sum_geometric_sequence_ratio_l120_120045

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a (n + 1) = a n + d

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℕ) (q : ℚ) :=
  ∀ n, a (n + 1) = a n * q
  
-- Prove the sum of the first n terms for an arithmetic sequence
theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 3 ∧ (∀ n, S n = (n * (3 + a (n + 1) - 1)) / 2) ∧ is_arithmetic_sequence a 4 → 
  S n = 2 * n^2 + n :=
sorry

-- Prove the range of the common ratio for a geometric sequence
theorem geometric_sequence_ratio (a : ℕ → ℕ) (S : ℕ → ℚ) (q : ℚ) :
  a 1 = 3 ∧ is_geometric_sequence a q ∧ ∃ lim : ℚ, (∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) ∧ lim < 12 → 
  -1 < q ∧ q < 1 ∧ q ≠ 0 ∧ q < 3/4 :=
sorry

end arithmetic_sequence_sum_geometric_sequence_ratio_l120_120045


namespace jail_sentence_goods_value_l120_120604

-- Definitions for all given conditions
def BaseSentence (goods : ℝ) (X : ℝ) : ℝ := goods / X
def ThirdOffenseIncrease (sentence : ℝ) : ℝ := sentence * 1.25
def TotalSentence (base_sentence : ℝ) : ℝ := ThirdOffenseIncrease(base_sentence) + 2

-- Prove that given the conditions, the value of X is 5000 dollars.
theorem jail_sentence_goods_value (goods : ℝ) (X : ℝ) :
  goods = 40000 →
  TotalSentence(BaseSentence(goods, X)) = 12 →
  X = 5000 := 
by
  intros h1 h2
  sorry

end jail_sentence_goods_value_l120_120604


namespace focus_of_parabola_l120_120484

/-- Given a quadratic function f(x) = ax^2 + bx + 2 where a ≠ 0, and for any real number x, it holds that |f(x)| ≥ 2,
    prove that the coordinates of the focus of the parabolic curve are (0, 1 / (4 * a) + 2). -/
theorem focus_of_parabola (a b : ℝ) (h_a : a ≠ 0)
  (h_f : ∀ x : ℝ, |a * x^2 + b * x + 2| ≥ 2) :
  (0, (1 / (4 * a) + 2)) = (0, (1 / (4 * a) + 2)) :=
by
  sorry

end focus_of_parabola_l120_120484


namespace sin_alpha_minus_pi_over_6_l120_120597

variable (α : ℝ)

theorem sin_alpha_minus_pi_over_6 (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 6) = 3 / 5) : 
  Real.sin (α - π / 6) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end sin_alpha_minus_pi_over_6_l120_120597


namespace ordered_triples_count_l120_120099

open Nat

theorem ordered_triples_count :
  ∃ (count : ℕ), (count = 10) ∧
  (∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
   lcm x y = 180 → lcm x z = 800 → lcm y z = 1200 →
   count = 10) :=
sorry

end ordered_triples_count_l120_120099


namespace johns_average_speed_l120_120922

/-- John and Beth each drove from Smallville to Crown City by different routes.
John completed the trip in 30 minutes.
Beth's route was 5 miles longer and it took her 20 minutes more than John to complete the trip.
Beth's average speed on this trip was 30 miles per hour.
Given these conditions, John's average speed was 40 miles per hour. -/
theorem johns_average_speed :
  ∀ (john_time_beth_longer john_time_minutes beth_speed from_hours to_hours),
    john_time_beth_longer = 20 / 60 → 
    john_time_minutes = 30 / 60 →
    from_hours = beth_speed / to_hours →
    beth_speed = 30 →
    to_hours = 5 / 6 →
    from_hours - 5 = 20 / (1 / 2) →
    from_hours = 25 →
    (20 / (1 / 2)) = 40 ∧ 25 = from_hours := 
by {
    intros,
    sorry
}

end johns_average_speed_l120_120922


namespace solve_equation_l120_120624

theorem solve_equation (x y z : ℕ) :
  (∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2 * n + 2) ↔ (x^2 + 3 * y^2 = 2^z) :=
by
  sorry

end solve_equation_l120_120624


namespace total_spent_l120_120356

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_friends : ℕ := 4

theorem total_spent : (original_price * (1 - discount_rate) * number_of_friends) = 40 := by
  sorry

end total_spent_l120_120356


namespace softballs_with_new_budget_l120_120661

-- Definitions for conditions
def original_budget : ℕ := 15 * 5
def budget_increase : ℕ := original_budget * 20 / 100
def new_budget : ℕ := original_budget + budget_increase
def cost_per_softball : ℕ := 9

-- The statement to prove
theorem softballs_with_new_budget : (new_budget / cost_per_softball) = 10 :=
by
  have h1 : original_budget = 75 := by norm_num
  have h2 : budget_increase = 15 := by norm_num
  have h3 : new_budget = 90 := by norm_num
  show (new_budget / cost_per_softball) = 10, from by norm_num

end softballs_with_new_budget_l120_120661


namespace exist_three_lines_l120_120606

def Point : Type := ℝ × ℝ

def A : Point := (1, 1)
def B : Point := (2, 2)
def C : Point := (3, 3)
def D : Point := (1, 3)
def E : Point := (2, 4)
def F : Point := (3, 5)

def Line (P Q : Point) : Point → Prop := λ R, ∃ k : ℝ, R = (P.1 + k * (Q.1 - P.1), P.2 + k * (Q.2 - P.2))

def intersects_at (l1 l2 l3 : Point → Prop) (P : Point) :=
  l1 P ∧ l2 P ∧ l3 P

theorem exist_three_lines :
  ∃ l1 l2 l3 : Point → Prop,
    (∀ p ∈ {A, B, C, D, E, F}, l1 p ∨ l2 p ∨ l3 p) ∧
    (∃ P1 P2, P1 ∈ {A, B, C, D, E, F} ∧ P2 ∈ {A, B, C, D, E, F} ∧ P1 ≠ P2 ∧ l1 P1 ∧ l1 P2) ∧
    (∃ P1 P2, P1 ∈ {A, B, C, D, E, F} ∧ P2 ∈ {A, B, C, D, E, F} ∧ P1 ≠ P2 ∧ l2 P1 ∧ l2 P2) ∧
    (∃ P1 P2, P1 ∈ {A, B, C, D, E, F} ∧ P2 ∈ {A, B, C, D, E, F} ∧ P1 ≠ P2 ∧ l3 P1 ∧ l3 P2) ∧
    (∃ P, intersects_at l1 l2 l3 P) :=
by
  sorry

end exist_three_lines_l120_120606


namespace largest_perfect_square_factor_1760_l120_120325

theorem largest_perfect_square_factor_1760 :
  ∃ n, (∃ k, n = k^2) ∧ n ∣ 1760 ∧ ∀ m, (∃ j, m = j^2) ∧ m ∣ 1760 → m ≤ n := by
  sorry

end largest_perfect_square_factor_1760_l120_120325


namespace coloring_problem_l120_120007

-- Define a type for the colors
inductive Color
| red
| white
| blue

-- We will consider 12 vertices (dots)
def V : Type := Fin 12

-- Define the adjacency relation
def adjacent : V → V → Prop :=
  λ v1 v2, 
  match (v1, v2) with
  -- Relations from first triangle
  | (⟨0,_⟩, ⟨1,_⟩) => true
  | (⟨1,_⟩, ⟨2,_⟩) => true
  | (⟨2,_⟩, ⟨0,_⟩) => true
  -- Relations from second triangle
  | (⟨3,_⟩, ⟨4,_⟩) => true
  | (⟨4,_⟩, ⟨5,_⟩) => true
  | (⟨5,_⟩, ⟨3,_⟩) => true
  | (⟨2,_⟩, ⟨3,_⟩) => true
  -- Relations from third triangle
  | (⟨6,_⟩, ⟨7,_⟩) => true
  | (⟨7,_⟩, ⟨8,_⟩) => true
  | (⟨8,_⟩, ⟨6,_⟩) => true
  | (⟨5,_⟩, ⟨6,_⟩) => true
  -- Relations from fourth triangle
  | (⟨9,_⟩, ⟨10,_⟩) => true
  | (⟨10,_⟩, ⟨11,_⟩) => true
  | (⟨11,_⟩, ⟨9,_⟩) => true
  | (⟨8,_⟩, ⟨9,_⟩) => true
  -- All other pairs are not adjacent
  | (_, _) => false
  end

-- Define a coloring as a function from vertices to colors
def Coloring := V → Color

-- Define the valid coloring condition
def valid_coloring (c : Coloring) : Prop :=
  ∀ v1 v2, adjacent v1 v2 → c v1 ≠ c v2

-- Define the main theorem statement
theorem coloring_problem : ∃ n : ℕ, n = 162 ∧ (finset.univ : finset Coloring).filter valid_coloring = finset.range n :=
by sorry

end coloring_problem_l120_120007


namespace distance_between_points_on_line_l120_120040

theorem distance_between_points_on_line (p q r s : ℝ) (h₁ : 2 * p - 3 * q + 6 = 0) (h₂ : 2 * r - 3 * s + 6 = 0) : 
  ((p - r)^2 + ((2 * p + 6) / 3 - (2 * r + 6) / 3)^2).sqrt = (Real.sqrt 13 / 3) * (p - r).abs := 
by
  sorry

end distance_between_points_on_line_l120_120040


namespace part1_part2_l120_120822

-- Define the complex number z in terms of m
def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

-- State the condition where z is a purely imaginary number
def purelyImaginary (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 = 0 ∧ m^2 - 3 * m + 2 ≠ 0

-- State the condition where z is in the second quadrant.
def inSecondQuadrant (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 < 0 ∧ m^2 - 3 * m + 2 > 0

-- Part 1: Prove that m = -1/2 given that z is purely imaginary.
theorem part1 : purelyImaginary m → m = -1/2 :=
sorry

-- Part 2: Prove the range of m for z in the second quadrant.
theorem part2 : inSecondQuadrant m → -1/2 < m ∧ m < 1 :=
sorry

end part1_part2_l120_120822


namespace complex_square_l120_120876

-- Define the complex number z
def z : ℂ := 5 - 3 * Complex.i

-- The statement we want to prove
theorem complex_square : z^2 = 16 - 30 * Complex.i := 
by
  sorry

end complex_square_l120_120876


namespace general_term_a_general_term_b_sum_of_first_n_terms_l120_120834

-- Define the sequences given the conditions
variables {a_n b_n : ℕ → ℕ}
variables (h_arithmetic : ∀ n, a_n n = n)
variables (h_geometric : ∀ n, b_n n = 2^(n-1))
variables (h_conditions : 
  b_n 1 = a_n 1 ∧ 
  b_n 3 = a_n 4 ∧ 
  b_n 1 + b_n 2 + b_n 3 = a_n 3 + a_n 4)

-- Define the sequences
def a (n : ℕ) := a_n n
def b (n : ℕ) := b_n n
def c (n : ℕ) := a n * b n
def T (n : ℕ) := ∑ i in range n, c (i + 1)

-- State the required proofs
theorem general_term_a : ∀ n, a n = n := 
by 
  intro n
  rw h_arithmetic
  sorry

theorem general_term_b : ∀ n, b n = 2^(n-1) := 
by 
  intro n
  rw h_geometric
  sorry

theorem sum_of_first_n_terms : ∀ n, T n = (n - 1) * 2^n + 1 :=
by 
  intro n
  sorry

end general_term_a_general_term_b_sum_of_first_n_terms_l120_120834


namespace daria_needs_to_earn_l120_120440

variable (ticket_cost : ℕ) (current_money : ℕ) (total_tickets : ℕ)

def total_cost (ticket_cost : ℕ) (total_tickets : ℕ) : ℕ :=
  ticket_cost * total_tickets

def money_needed (total_cost : ℕ) (current_money : ℕ) : ℕ :=
  total_cost - current_money

theorem daria_needs_to_earn :
  total_cost 90 4 - 189 = 171 :=
by
  sorry

end daria_needs_to_earn_l120_120440


namespace min_value_inequality_l120_120946

open Real

theorem min_value_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 9) :
  ( (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ) ≥ 9 :=
sorry

end min_value_inequality_l120_120946


namespace greening_problem_l120_120363

theorem greening_problem
    (A_B_ratio : ∀ (x : ℕ), ∀ (A_x : ℕ), A_x = 2 * x)
    (time_difference : ∀ (x : ℕ), x ≠ 0 → (400 / (2 * x)) + 4 = (400 / x))
    (cost_limit : (0.4 * $ days_A + 0.25 * ((1800 - 100 * $ days_A) / 50)) <= 8)
    : A_x = 100 ∧ x = 50 ∧ $ days_A >= 10 :=
by
  sorry


end greening_problem_l120_120363


namespace solution_interval_l120_120964

theorem solution_interval:
  ∃ x : ℝ, (x^3 = 2^(2-x)) ∧ 1 < x ∧ x < 2 :=
by
  sorry

end solution_interval_l120_120964


namespace scientific_notation_of_population_l120_120228

theorem scientific_notation_of_population :
  (141260 : ℝ) = 1.4126 * 10^5 :=
sorry

end scientific_notation_of_population_l120_120228


namespace percent_nonunion_part_time_women_l120_120543

noncomputable def percent (part: ℚ) (whole: ℚ) : ℚ := part / whole * 100

def employees : ℚ := 100
def men_ratio : ℚ := 54 / 100
def women_ratio : ℚ := 46 / 100
def full_time_men_ratio : ℚ := 70 / 100
def part_time_men_ratio : ℚ := 30 / 100
def full_time_women_ratio : ℚ := 60 / 100
def part_time_women_ratio : ℚ := 40 / 100
def union_full_time_ratio : ℚ := 60 / 100
def union_part_time_ratio : ℚ := 50 / 100

def men := employees * men_ratio
def women := employees * women_ratio
def full_time_men := men * full_time_men_ratio
def part_time_men := men * part_time_men_ratio
def full_time_women := women * full_time_women_ratio
def part_time_women := women * part_time_women_ratio
def total_full_time := full_time_men + full_time_women
def total_part_time := part_time_men + part_time_women

def union_full_time := total_full_time * union_full_time_ratio
def union_part_time := total_part_time * union_part_time_ratio
def nonunion_full_time := total_full_time - union_full_time
def nonunion_part_time := total_part_time - union_part_time

def nonunion_part_time_women_ratio : ℚ := 50 / 100
def nonunion_part_time_women := part_time_women * nonunion_part_time_women_ratio

theorem percent_nonunion_part_time_women : 
  percent nonunion_part_time_women nonunion_part_time = 52.94 :=
by
  sorry

end percent_nonunion_part_time_women_l120_120543


namespace area_of_rhombus_l120_120645

theorem area_of_rhombus (d1 : ℝ) (s : ℝ) (h1 : d1 = 6) (h2 : s^2 - 2 * s - 15 = 0) (h3 : s > 0) :
  (1 / 2 * d1 * 2 * real.sqrt (s^2 - (d1 / 2)^2) = 24) :=
by
  sorry

end area_of_rhombus_l120_120645


namespace quadratic_rewrite_l120_120256

theorem quadratic_rewrite (p : ℝ) (n : ℝ)
  (h1 : x^2 + p*x + 1/4 = (x + n)^2 - 1/8)
  (h2 : p < 0) :
  p = -sqrt(6) / 2 := by
    sorry

end quadratic_rewrite_l120_120256


namespace sam_age_two_years_ago_l120_120570

theorem sam_age_two_years_ago (J S : ℕ) (h1 : J = 3 * S) (h2 : J + 9 = 2 * (S + 9)) : S - 2 = 7 :=
sorry

end sam_age_two_years_ago_l120_120570


namespace total_pages_in_book_is_250_l120_120920

-- Definitions
def avg_pages_first_part := 36
def days_first_part := 3
def avg_pages_second_part := 44
def days_second_part := 3
def pages_last_day := 10

-- Calculate total pages
def total_pages := (days_first_part * avg_pages_first_part) + (days_second_part * avg_pages_second_part) + pages_last_day

-- Theorem statement
theorem total_pages_in_book_is_250 : total_pages = 250 := by
  sorry

end total_pages_in_book_is_250_l120_120920


namespace number_of_roots_l120_120105

noncomputable theory
open Real

def domain (x : ℝ) := abs x ≤ sqrt 14
def equation (x : ℝ) := sin x - cos (2 * x) = 0

theorem number_of_roots : 
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, domain x) ∧ (∀ x ∈ xs, equation x) ∧ xs.card = 6 := 
by sorry

end number_of_roots_l120_120105


namespace period_tan_sec_cot_csc_l120_120692

theorem period_tan_sec_cot_csc (x : ℝ) :
  ∃ T, T = π ∧ (∀ x, y = tan x + sec x + cot x + csc x → y = tan (x + T) + sec (x + T) + cot (x + T) + csc (x + T)) :=
begin
  use π,
  split,
  { refl },
  {
    intro x,
    intro y,
    intro h,
    rw [←h],
    apply congr_arg,
    rw [add_comm x π],
    apply congr_arg,
    rw add_comm,
    rw tan_periodic,
    rw sec_periodic,
    rw cot_periodic,
    rw csc_periodic,
  },
    sorry
end

end period_tan_sec_cot_csc_l120_120692


namespace sequence_general_term_l120_120663

-- Define the sequence conditions
def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = 2 * a n + 1 / 2

-- Define the general term of the sequence
def general_term (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a n = 3 * 2^(n-2) - 1 / 2

-- Theorem stating the given conditions imply the general term
theorem sequence_general_term (a : ℕ → ℚ) :
  sequence a → general_term a :=
by
  sorry

end sequence_general_term_l120_120663


namespace base6_to_decimal_45_eq_29_l120_120437

theorem base6_to_decimal_45_eq_29 : base_to_decimal 6 [4, 5] = 29 :=
by
  sorry

end base6_to_decimal_45_eq_29_l120_120437


namespace equilateral_triangle_cut_l120_120438

theorem equilateral_triangle_cut :
  ∀ (T : Type) [metric_space T] [normed_group T] [normed_space ℝ T],
  ∃ (parts : list (set T)), 
    (∀ part ∈ parts, is_equilateral_triangle part) ∧ 
    (length parts = 6) ∧ 
    (rearrange_to form 7 identical equilateral triangles) :=
by
  sorry

end equilateral_triangle_cut_l120_120438


namespace number_of_roots_l120_120113

noncomputable def roots_equation_count : ℝ :=
  let interval := Icc (-real.sqrt 14) (real.sqrt 14)
  ∑ x in interval, (if sqrt (14 - x^2) * (sin x - cos (2 * x)) = 0 then 1 else 0)

theorem number_of_roots : roots_equation_count = 6 := by {
  sorry
}

end number_of_roots_l120_120113


namespace no_solution_for_a_l120_120791

theorem no_solution_for_a {a : ℝ} :
  (a ∈ Set.Iic (-32) ∪ Set.Ici 0) →
  ¬ ∃ x : ℝ,  9 * |x - 4 * a| + |x - a^2| + 8 * x - 4 * a = 0 :=
by
  intro h
  sorry

end no_solution_for_a_l120_120791


namespace solve_equation1_solve_equation2_l120_120992

noncomputable def equation1_solutions : set ℝ :=
  {x | x^2 + 4 * x - 2 = 0}

noncomputable def equation2_solutions : set ℝ :=
  {x | 2 * x^2 - 3 * x + 1 = 0}

theorem solve_equation1 : equation1_solutions = {-2 + Real.sqrt 6, -2 - Real.sqrt 6} :=
  sorry

theorem solve_equation2 : equation2_solutions = {1 / 2, 1} :=
  sorry

end solve_equation1_solve_equation2_l120_120992


namespace reciprocal_of_mixed_num_l120_120298

-- Define the fraction representation of the mixed number -1 1/2
def mixed_num_to_improper (a : ℚ) : ℚ := -3/2

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Prove the statement
theorem reciprocal_of_mixed_num : reciprocal (mixed_num_to_improper (-1.5)) = -2/3 :=
by
  -- skip proof
  sorry

end reciprocal_of_mixed_num_l120_120298


namespace equalize_chessboard_l120_120449

-- Define the chessboard as a type
def Chessboard := Array (Array Int)

-- Define the dimensions of the chessboard
def rows := 2018
def cols := 2019

-- Define the operation on the chessboard
def operation (board : Chessboard) (selected_cells : Array (Nat × Nat)) : Chessboard :=
  board.mapWithIndex (λ i row =>
    row.mapWithIndex (λ j cell =>
      if (i, j) ∈ selected_cells then
        let neighbors := [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        let valid_neighbors := neighbors.filter (λ (x, y) => x >= 0 ∧ x < rows ∧ y >= 0 ∧ y < cols)
        let mean := valid_neighbors.map (λ (x, y) => board[x]![y]!).sum / valid_neighbors.length
        mean
      else
        cell
    )
  )

-- The main theorem: it's not always possible to make all cell values equal
theorem equalize_chessboard : ¬ ∃ (fin_ops : (Chessboard → Chessboard) → Prop),
  ∀ board : Chessboard, ∃ steps : Nat, (fin_ops operation)^steps board = board.map (λ row => row.map (λ _ => board[0]![0]!)) := 
sorry

end equalize_chessboard_l120_120449


namespace alicia_sundae_cost_l120_120339

theorem alicia_sundae_cost (C_Yvette C_Brant C_Josh final_bill tip_percent : ℝ) 
  (h_Yvette : C_Yvette = 9) 
  (h_Brant : C_Brant = 10) 
  (h_Josh : C_Josh = 8.5) 
  (h_tip_percent : tip_percent = 0.20) 
  (h_final_bill : final_bill = 42) : 
  let total_known_cost := C_Yvette + C_Brant + C_Josh in
  let cost_before_tip := final_bill / (1 + tip_percent) in
  cost_before_tip - total_known_cost = 7.5 := 
by
  sorry

end alicia_sundae_cost_l120_120339


namespace number_of_distinct_colorings_l120_120689

open Finset GroupTheory

noncomputable def distinct_vertex_colorings (m : ℕ) : ℕ :=
  (1 / 24 : ℚ) * (m ^ 8 + 17 * m ^ 4 + 6 * m ^ 2)

theorem number_of_distinct_colorings (m : ℕ) :
  distinct_vertex_colorings m = (1 / 24 : ℚ) * (m ^ 8 + 17 * m ^ 4 + 6 * m ^ 2) :=
sorry

end number_of_distinct_colorings_l120_120689


namespace roots_polynomial_d_l120_120255

theorem roots_polynomial_d (c d u v : ℝ) (ru rpush rv rpush2 : ℝ) :
    (u + v + ru = 0) ∧ (u+3 + v-2 + rpush2 = 0) ∧
    (d + 153 = -(u + 3) * (v - 2) * (ru)) ∧ (d + 153 = s) ∧ (s = -(u + 3) * (v - 2) * (rpush2 - 1)) →
    d = 0 :=
by
  sorry

end roots_polynomial_d_l120_120255


namespace division_of_complex_l120_120712

-- Define the complex numbers involved
def complex1 := (1 : ℂ) + (1 : ℂ) * I
def complex2 := (3 : ℂ) - (4 : ℂ) * I
def result := (- 1 / 25 : ℂ) + (7 / 25 : ℂ) * I

-- The theorem to be stated
theorem division_of_complex : complex1 / complex2 = result := 
by 
  -- Theorems and conditions used
  sorry

end division_of_complex_l120_120712


namespace find_a6_l120_120487

def a_seq (n : ℕ) : ℕ → ℝ

variable (a_seq : ℕ → ℝ)

axiom a2_eq_3 : a_seq 2 = 3
axiom a3_plus_a5 : a_seq 3 + a_seq 5 = 12
axiom arithmetic_sequence_property : ∃ d : ℝ, ∀ n : ℕ, a_seq (n+1) = a_seq n + d

theorem find_a6 : a_seq 6 = 9 := by
  sorry

end find_a6_l120_120487


namespace problem1_problem2_l120_120503

-- Define the function f and its derivative
def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := - (1 / 3) * x^3 + b * x^2 + c * x + b * c
def f' (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := - x^2 + 2 * b * x + c

-- Problem 1: Prove values of b and c given the extremum condition
theorem problem1 (b c : ℝ) : 
  (f 1 b c = - (4 / 3)) ∧ (f' 1 b c = 0) → b = -1 ∧ c = 3 :=
by sorry

-- Define the function g as modification of f for the second problem
def g (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := f x b c - c * (x + b)
def g' (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := f' x b c - c

-- Problem 2: Prove the range of b given the slope condition
theorem problem2 (b : ℝ) :
  (∀ x, (1/2 < x ∧ x < 3) → g' x b 3 ≤ 2) → b ≤ Real.sqrt 2 :=
by sorry

end problem1_problem2_l120_120503


namespace part1_part2_part3_l120_120813

noncomputable def f (x : ℝ) : ℝ := exp x - x^2

-- 1. Proof that a = 1 and b = e - 2
theorem part1 (a b : ℝ) (h1 : f(1) = (e - a)) (h2 : deriv f 1 = e - 2 * a) :
  a = 1 ∧ b = e - 2 := by
  sorry

-- 2. Proof that the maximum value of y = f(x) on [0, 1] is e - 1
theorem part2 : ∃ x ∈ set.Icc (0:ℝ) 1, ∀ y ∈ set.Icc (0:ℝ) 1, f y ≤ f x ∧ f x = e - 1 := by
  sorry

-- 3. Proof that e^x + (1-e)x - x * ln x - 1 ≥ 0 for x > 0
theorem part3 (x : ℝ) (hx : 0 < x) : exp x + (1 - e) * x - x * log x - 1 ≥ 0 := by
  sorry

end part1_part2_part3_l120_120813


namespace problem_solution_l120_120826

-- Define the vectors
def vectorAB : ℝ × ℝ × ℝ := (2, -1, -4)
def vectorAD : ℝ × ℝ × ℝ := (4, 2, 0)
def vectorAP : ℝ × ℝ × ℝ := (-1, 2, -1)

-- Define dot product for 3D vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the result goals
def AP_perp_AB : Prop :=
  dot_product vectorAP vectorAB = 0

def AP_perp_AD : Prop :=
  dot_product vectorAP vectorAD = 0

def AP_normal_plane : Prop :=
  (dot_product vectorAP vectorAB = 0) ∧ (dot_product vectorAP vectorAD = 0)

-- Prove correct conclusions (1), (2), and (3)
theorem problem_solution : AP_perp_AB ∧ AP_perp_AD ∧ AP_normal_plane :=
by
  sorry

end problem_solution_l120_120826


namespace athlete_A_most_stable_l120_120282

noncomputable def athlete_A_variance : ℝ := 0.019
noncomputable def athlete_B_variance : ℝ := 0.021
noncomputable def athlete_C_variance : ℝ := 0.020
noncomputable def athlete_D_variance : ℝ := 0.022

theorem athlete_A_most_stable :
  athlete_A_variance < athlete_B_variance ∧
  athlete_A_variance < athlete_C_variance ∧
  athlete_A_variance < athlete_D_variance :=
by {
  sorry
}

end athlete_A_most_stable_l120_120282


namespace intersection_P_complement_Q_l120_120491

-- Defining the sets P and Q
def R := Set ℝ
def P : Set ℝ := {x | x^2 + 2 * x - 3 = 0}
def Q : Set ℝ := {x | Real.log x < 1}
def complement_R_Q : Set ℝ := {x | x ≤ 0 ∨ x ≥ Real.exp 1}
def intersection := {x | x ∈ P ∧ x ∈ complement_R_Q}

-- Statement of the theorem
theorem intersection_P_complement_Q : 
  intersection = {-3} :=
by
  sorry

end intersection_P_complement_Q_l120_120491


namespace smallest_n_for_divisibility_l120_120218

def geometric_sequence (a r n : ℕ) : ℕ := a * r^(n-1)

theorem smallest_n_for_divisibility :
  ∃ n : ℕ, ∀ a1 a2 : ℕ,
  a1 = 2 →
  a2 = 70 →
  ∀ n ≥ 1, 
  let r := a2 / a1 in
  (geometric_sequence a1 r n) % 5000000 = 0 →
  n = 8 :=
by
  intro n a1 a2 h1 h2
  let r := a2 / a1
  intro hn
  sorry

end smallest_n_for_divisibility_l120_120218


namespace smallest_positive_period_intervals_of_increase_graph_transformation_l120_120507

def f (x : ℝ) : ℝ := 
  Real.sin (2 * x + Real.pi / 6) + 3 / 2

theorem smallest_positive_period : 
  (∃ T > 0, ∀ x : ℝ, f (x + T) = f x) ∧ ∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ Real.pi :=
sorry

theorem intervals_of_increase :
  ∀ k : ℤ, 
  ∀ x : ℝ, 
    (k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6) → 
    (f' x > 0) ∧ 
    (f' (x + Real.pi) = f' x) :=
sorry

theorem graph_transformation :
  ∀ x : ℝ, 
  f x = Real.sin (2 * (x + Real.pi / 12)) + 3 / 2 :=
sorry

end smallest_positive_period_intervals_of_increase_graph_transformation_l120_120507


namespace probability_product_multiple_of_8_l120_120882

-- Define the set from which numbers are chosen.
def S : set ℕ := {2, 4, 6, 8}

-- Define what it means for the product of two numbers to be a multiple of 8.
def is_multiple_of_8 (a b : ℕ) : Prop :=
  8 ∣ (a * b)

-- Define the probability calculation.
def probability_of_event {α : Type*} (s : finset α) (p : α → Prop) [decidable_pred p] : ℚ :=
  s.filter p).card / s.card

-- Define all pairs of numbers chosen from the set S.
def pairs := { (a, b) | a ∈ S ∧ b ∈ S ∧ a ≠ b}

-- The theorem statement.
theorem probability_product_multiple_of_8 : 
  probability_of_event pairs (λ (p : ℕ × ℕ), is_multiple_of_8 p.fst p.snd) = 2 / 3 := 
sorry

end probability_product_multiple_of_8_l120_120882


namespace _l120_120783

noncomputable def t_value_theorem (a b x d t y : ℕ) (h1 : a + b = x) (h2 : x + d = t) (h3 : t + a = y) (h4 : b + d + y = 16) : t = 8 :=
by sorry

end _l120_120783


namespace monotonicity_of_f_range_of_a_l120_120017

-- Definitions given in the conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x - (Real.sin x) / (Real.cos x)^3

-- Problem 1: Monotonicity of f(x) when a = 8
theorem monotonicity_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) (a : ℝ) (ha : a = 8) :
  (0 < x ∧ x < Real.pi / 4 → (∀ (b : ℝ), (x < b ∧ b < Real.pi / 2) → f a b > f a x)) ∧
  (Real.pi / 4 < x ∧ x < Real.pi / 2 → (∀ (b : ℝ), (Real.pi / 4 < b ∧ b < x) → f a b < f a x)) :=
sorry

-- Problem 2: Range of a such that f(x) < sin 2x for all x in (0, π/2)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x < Real.pi / 2) → f a x < Real.sin (2 * x)) ↔ (a ≤ 3) :=
sorry

end monotonicity_of_f_range_of_a_l120_120017


namespace students_in_college_l120_120550

def students_proofs (P S : ℕ) : Prop :=
  (S = 15 * P) ∧ (S + P = 40000) ∧ (S = 37500)

theorem students_in_college (P S : ℕ) (h : students_proofs P S) : S = 37500 :=
by
  cases h with h1 h2
  exact h2.right

end students_in_college_l120_120550


namespace find_p_max_area_PAB_l120_120086

-- Given Problem Conditions
def parabola (p : ℝ) := { pt : ℝ × ℝ // pt.1^2 = 2 * p * pt.2 }
def circle := { pt : ℝ × ℝ // pt.1^2 + (pt.2 + 4)^2 = 1 }
def focus (p : ℝ) := (0, p / 2)

-- Part (1): Proof to find p
theorem find_p (dist : ℝ) (h1 : dist = 4) (p : ℝ) (h2 : p > 0) 
  (h3 : ∀ (F : ℝ × ℝ) (M : ℝ × ℝ), F = focus p → M ∈ circle → abs (dist_of_points F M) - 1 = dist) : p = 2 := 
sorry

-- Part (2): Proof for maximum area of ∆PAB
theorem max_area_PAB (p : ℝ) (hp : p = 2)
  (P : ℝ × ℝ) (hP : P ∈ circle) 
  (A B : ℝ × ℝ) (hA : A ∈ parabola p) (hB : B ∈ parabola p)
  (PA_tangent : is_tangent P A (parabola p))
  (PB_tangent : is_tangent P B (parabola p)) : 
  ∃ (max_area : ℝ), max_area = 20 * real.sqrt 5 :=
sorry

-- Helper Definitions (may need to be defined precisely for real implementation)
noncomputable def dist_of_points (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def is_tangent (P A : ℝ × ℝ) (para : parabola) : Prop := sorry
noncomputable def is_tangent (P B : ℝ × ℝ) (para : parabola) : Prop := sorry

end find_p_max_area_PAB_l120_120086


namespace sequence_properties_l120_120043

theorem sequence_properties (a : ℕ → ℝ)
  (h1 : a 1 = 1 / 5)
  (h2 : ∀ n : ℕ, n > 1 → a (n - 1) / a n = (2 * a (n - 1) + 1) / (1 - 2 * a n)) :
  (∀ n : ℕ, n > 0 → (1 / a n) - (1 / a (n - 1)) = 4) ∧
  (∀ m k : ℕ, m > 0 ∧ k > 0 → a m * a k = a (4 * m * k + m + k)) :=
by
  sorry

end sequence_properties_l120_120043


namespace seating_arrangements_l120_120671

theorem seating_arrangements :
  let seats := 12
  let empty_between (x y : ℕ) := (x < y) ∧ (y < 12 - x) ∧ (12 - x - y > 2)
  let a_between_b_and_c (a b c : ℕ) := 
    (empty_between a b) ∧ (empty_between b c) ∧ (a < b) ∧ (b < c)
  (a b c : ℕ)
  (valid_seating := a_between_b_and_c a b c) 
  (total_arrangements := 2 * (Nat.choose 8 5)) :
  (valid_seating → total_arrangements = 112) := sorry

end seating_arrangements_l120_120671


namespace sin_BCQ_eq_sin_BAP_l120_120577

theorem sin_BCQ_eq_sin_BAP
  (A B C P Q D E : Point)
  (hABC : Triangle A B C)
  (hP : Inside P (Triangle A B C))
  (hPAC_PCB : ∠ P A C = ∠ P C B)
  (hD : Midpoint D P C)
  (hE : Line A P ∩ Line B C = E)
  (hQ : Line B P ∩ Line D E = Q) :
  sin (Angle B C Q) = sin (Angle B A P) :=
sorry

end sin_BCQ_eq_sin_BAP_l120_120577


namespace locus_of_centers_of_tangent_circles_l120_120818

variables (P K R : Point) (k : Circle) (E1 E2 : Point)

-- Assume E1 and E2 are tangent points from P to k, R is the intersection of PK and E1E2
variable (tangent_points : Tangents P k [E1, E2])
variable (intersect_R : ∃ R, Collinear [P, K, R] ∧ R ∈ Line.mk E1 E2)

-- The locus statement
theorem locus_of_centers_of_tangent_circles :
  ∀ g : Sphere, contains g k →
  ∃ K_g : Point, center_of (tangent_circle g P) K_g ∧ ∀ Q : Point,
  -- The locus of K_g is circle in the plane perpendicular to k with diameter RP excluding P
  (Q ∈ Circle_diameter_perpendicular R P K ∧ Q ≠ P) ↔ center_of (tangent_circle g P) Q := 
  sorry

end locus_of_centers_of_tangent_circles_l120_120818


namespace max_midpoint_x_l120_120161

def f (x : ℝ) : ℝ := Real.log x

def coordinates_P (m : ℝ) := (m, Real.log m)

def tangent_line (P : ℝ × ℝ) (x : ℝ) :=
  P.2 + (1 / P.1) * (x - P.1)

def perpendicular_line (P : ℝ × ℝ) (x : ℝ) :=
  P.2 - P.1 * (x - P.1)

def point_E (m : ℝ) : ℝ × ℝ :=
  (m - m * Real.log m, 0)

def point_F (m : ℝ) : ℝ × ℝ :=
  (m + Real.log m / m, 0)

def midpoint_x_coordinate (E F : ℝ × ℝ) : ℝ :=
  (E.1 + F.1) / 2

theorem max_midpoint_x (P : ℝ × ℝ) (hP : 0 < P.1) :
  let m := P.1 in
  let E := point_E m in
  let F := point_F m in
  midpoint_x_coordinate E F ≤ (1/2) * (Real.exp 1 + 1 / Real.exp 1) :=
sorry

end max_midpoint_x_l120_120161


namespace line_D_does_not_employ_stillness_and_motion_l120_120655

def line_A := "The bridge echoes with the distant barking of dogs, and the courtyard is empty with people asleep."
def line_B := "The stove fire illuminates the heaven and earth, and the red stars are mixed with the purple smoke."
def line_C := "The cold trees begin to have bird activities, and the frosty bridge has no human passage yet."
def line_D := "The crane cries over the quiet Chu mountain, and the frost is white on the autumn river in the morning."

def employs_stillness_and_motion (line : String) : Bool :=
  match line with
  | line_A => true
  | line_B => true
  | line_C => true
  | _ => false

theorem line_D_does_not_employ_stillness_and_motion : ¬ employs_stillness_and_motion line_D :=
by
  -- The proof would go here, but for the sake of this task we can use sorry.
  sorry

end line_D_does_not_employ_stillness_and_motion_l120_120655


namespace contemporaries_probability_l120_120686

theorem contemporaries_probability :
  (∃ x y : ℕ, 0 ≤ x ∧ x ≤ 600 ∧ 0 ≤ y ∧ y ≤ 600 ∧ x < y + 120 ∧ y < x + 100) →
  (193 : ℚ) / 200 = (∑ x in (finset.range 600), ∑ y in (finset.range 600), 
              if (x < y + 120 ∧ y < x + 100) then 1 else 0) / (600 * 600) :=
begin
  sorry
end

end contemporaries_probability_l120_120686


namespace conditional_probability_l120_120694

variable (A B : Event) (P : Event → ℚ)

-- Conditions
axiom prob_A : P A = 1/6
axiom prob_B : P B = 1/2
axiom prob_AB : P (A ∧ B) = 1/12

theorem conditional_probability : P (A ∧ B) / P B = 1 / 6 :=
by
  -- Proof goes here
  sorry

end conditional_probability_l120_120694


namespace solve_system_of_equations_l120_120271

theorem solve_system_of_equations :
  ∃ (x y : ℤ), (x - y = 2) ∧ (2 * x + y = 7) ∧ (x = 3) ∧ (y = 1) :=
by
  sorry

end solve_system_of_equations_l120_120271


namespace problem_correct_l120_120718


def problem : ℕ := 101 * 101 - 99 * 99

theorem problem_correct : problem = 400 := by
  sorry

end problem_correct_l120_120718


namespace tangent_line_equation_l120_120795

theorem tangent_line_equation (a : ℝ) :
  let curve := λ x : ℝ, x^3 - 3 * x^2 + a in
  let P := (1, -1) in
  let tangent_line := λ x : ℝ, -3 * x + 2 in
  tangent_line P.1 = P.2 ∧ ∀ x, (curve x - curve 1) / (x - 1) = (-3 : ℝ) := by
{
  sorry
}

end tangent_line_equation_l120_120795


namespace find_range_of_a_l120_120854

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x ^ 3 + x ^ 2 + a * x

noncomputable def g (x : ℝ) : ℝ := 1 / Real.exp x

theorem find_range_of_a (a : ℝ) :
  (∃ x1 ∈ Set.Icc (1 / 2 : ℝ) 2, ∃ x2 ∈ Set.Icc (1 / 2 : ℝ) 2, 
    fderiv ℝ (f a) x1 ≤ g x2) ↔
  a ≤ (Real.sqrt Real.exp 1) / Real.exp 1 - (5 / 4) :=
by
  sorry

end find_range_of_a_l120_120854


namespace count_visible_factor_numbers_200_250_l120_120382

def is_visible_factor_number (n : ℕ) : Prop :=
  (∀ d ∈ (repr n).toList.map (λ c => c.toNat - '0'.toNat), d ≠ 0 → n % d = 0)

def count_visible_factor_numbers (a b : ℕ) : ℕ :=
  (a.b.to_list).count is_visible_factor_number

theorem count_visible_factor_numbers_200_250 : count_visible_factor_numbers 200 250 = 21 :=
by sorry

end count_visible_factor_numbers_200_250_l120_120382


namespace total_gray_trees_l120_120676

theorem total_gray_trees :
  (∃ trees_first trees_second trees_third gray1 gray2,
    trees_first = 100 ∧
    trees_second = 90 ∧
    trees_third = 82 ∧
    gray1 = trees_first - trees_third ∧
    gray2 = trees_second - trees_third ∧
    trees_first + trees_second - 2 * trees_third = gray1 + gray2) →
  (gray1 + gray2 = 26) :=
by
  intros
  sorry

end total_gray_trees_l120_120676


namespace efficacy_rate_is_80_percent_l120_120907

-- Define the total number of people surveyed
def total_people : ℕ := 20

-- Define the number of people who find the new drug effective
def effective_people : ℕ := 16

-- Calculate the efficacy rate
def efficacy_rate (effective : ℕ) (total : ℕ) : ℚ := effective / total

-- The theorem to be proved
theorem efficacy_rate_is_80_percent : efficacy_rate effective_people total_people = 0.8 :=
by
  sorry

end efficacy_rate_is_80_percent_l120_120907


namespace number_of_roots_l120_120115

noncomputable def roots_equation_count : ℝ :=
  let interval := Icc (-real.sqrt 14) (real.sqrt 14)
  ∑ x in interval, (if sqrt (14 - x^2) * (sin x - cos (2 * x)) = 0 then 1 else 0)

theorem number_of_roots : roots_equation_count = 6 := by {
  sorry
}

end number_of_roots_l120_120115


namespace original_price_of_cycle_l120_120369

theorem original_price_of_cycle :
  ∃ (OP : ℝ), OP ≈ 930 ∧ ∃ (selling_price gain : ℝ), selling_price = 1210 ∧ gain = 30.107526881720432 ∧ 
  OP = selling_price / (1 + (gain / 100)) := sorry

end original_price_of_cycle_l120_120369


namespace incorrect_statement_D_l120_120278

noncomputable def Riemann_function : ℝ → ℝ
| x => if (∃ p q : ℕ, nat.coprime p q ∧ p > q ∧ x = (q : ℝ) / (p : ℝ)) then 
          (let p := nat.find_greatest (λ p, ∃ q, nat.coprime p q ∧ p > q ∧ x = (q : ℝ) / (p : ℝ)) x in
            (1 : ℝ) / (p : ℝ))
        else if x = 0 ∨ x = 1 ∨ ¬ (∃ p q : ℕ, nat.coprime p q ∧ p > q ∧ x = (q : ℝ) / (p : ℝ)) then (0 : ℝ)
        else (0 : ℝ)

theorem incorrect_statement_D : ∀ m : ℝ, m > 1 → ¬ ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ Riemann_function x = m / (x + 1) :=
by
  intro m hm
  sorry

end incorrect_statement_D_l120_120278


namespace repeating_decimal_sum_l120_120522

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), Nat.coprime a b ∧ (0.353535... = (a : ℝ) / (b : ℝ)) ∧ (a + b = 134) :=
sorry

end repeating_decimal_sum_l120_120522


namespace rabbit_distribution_problem_l120_120253

noncomputable theory

def parents := ["Peter", "Pauline", "Penelope"]
def children := ["Flopsie", "Mopsie", "Cotton_tail"]

def stores := finset.range 5

def isValidDistribution (distribution: list (option ℕ)) : Prop :=
  ∀ parent child, parent ∈ [0, 1, 2] → child ∈ [3, 4, 5] →
    (distribution[parent] ≠ distribution[child] ∨ distribution[parent] = none ∨ distribution[child] = none)

def countValidDistributions : ℕ :=
  (finset.range ((option ℕ) ^ 6)).filter
    (λ distribution, isValidDistribution (finset.univ.map distribution).val).card

theorem rabbit_distribution_problem :
  countValidDistributions = 398 :=
sorry

end rabbit_distribution_problem_l120_120253


namespace maximum_value_of_chords_l120_120482

noncomputable def max_sum_of_chords (r : ℝ) (h1 : r = 5) 
(h2 : ∃ PA PB PC : ℝ, PA = 2 * PB ∧ PA^2 + PB^2 + PC^2 = (2 * r)^2) : ℝ := 
  6 * Real.sqrt 10

theorem maximum_value_of_chords (P : Point) (r : ℝ) (h1 : r = 5) 
(h2 : ∃ PA PB PC : ℝ, PA = 2 * PB ∧ PA^2 + PB^2 + PC^2 = (2 * r)^2) : 
  PA + PB + PC ≤ 6 * Real.sqrt 10 :=
by
  sorry

end maximum_value_of_chords_l120_120482


namespace range_transform_l120_120533

variable (f : ℝ → ℝ)

theorem range_transform 
    (h : ∀ x, 1 ≤ f(x) ∧ f(x) ≤ 3) : (∀ y, ∃ x, y = 1 - 2 * f(x + 3) ↔ -5 ≤ y ∧ y ≤ -1) :=
by
  sorry

end range_transform_l120_120533


namespace constructed_expression_equals_original_l120_120559

variable (a : ℝ)

theorem constructed_expression_equals_original : 
  a ≠ 0 → 
  ((1/a) / ((1/a) * (1/a)) - (1/a)) / (1/a) = (a + 1) * (a - 1) :=
by
  intro h
  sorry

end constructed_expression_equals_original_l120_120559


namespace least_xy_value_l120_120830

theorem least_xy_value (x y : ℕ) (hposx : x > 0) (hposy : y > 0) (h : 1/x + 1/(3*y) = 1/8) :
  xy = 96 :=
by
  sorry

end least_xy_value_l120_120830


namespace volume_of_solid_correct_l120_120665

noncomputable def volume_of_solid : Real :=
  let v : ℝ × ℝ × ℝ := (x, y, z)
  let c : ℝ × ℝ × ℝ := (6, -24, 12)
  if (v.1 * v.1 + v.2 * v.2 + v.3 * v.3 = v.1 * c.1 + v.2 * c.2 + v.3 * c.3) then 756 * Real.sqrt 21 * Real.pi else 0

theorem volume_of_solid_correct : volume_of_solid = 756 * Real.sqrt 21 * Real.pi := 
  sorry

end volume_of_solid_correct_l120_120665


namespace sixteen_powers_five_equals_four_power_ten_l120_120719

theorem sixteen_powers_five_equals_four_power_ten : 
  (16 * 16 * 16 * 16 * 16 = 4 ^ 10) :=
by
  sorry

end sixteen_powers_five_equals_four_power_ten_l120_120719


namespace concyclic_iff_orthocenter_l120_120160

noncomputable def Triangle (α : Type _) [normedAddCommGroup α] [innerProductSpace ℝ α] := {a b c : α // ∃ x y z : ℝ, 0 < x ∧ x < y ∧ y < z ∧ y = dist a b ∧ x = dist a c ∧ z = dist b c}

variable {α : Type _} [normedAddCommGroup α] [innerProductSpace ℝ α]

def altitude (A B C : α) (D : α) : Prop := ∃ h : ℝ, dist D h = dist A B ∧ dist D h = dist A C

def orthocenter (t : Triangle α) (P : α) : Prop := ∃ B C D, altitude B C D

def circumcenter (B C D : α) (Q : α) : Prop := ∃ O, dist B C = dist Q O ∧ dist B D = dist Q O

def is_concyclic {α : Type _} [topologicalSpace α] [normedAddCommGroup α] [innerProductSpace ℝ α] (p1 p2 p3 p4 : α) : Prop :=
∃ (circ : Circle α), p1 ∈ circ ∧ p2 ∈ circ ∧ p3 ∈ circ ∧ p4 ∈ circ

theorem concyclic_iff_orthocenter
  (A B C D P E F O1 O2 : α)
  (h_tri_acute : ∀ A B C : α, ∃ a b c : ℝ, 0 < a ∧ a < b ∧ b < c ∧ b = dist A B ∧ a = dist A C ∧ c = dist B C)
  (h_ad_altitude : altitude A B C D)
  (h_pe_perp_ac : ∃ E : α, dist P E = 0 ∧ E ∈ line A C)
  (h_pf_perp_ab : ∃ F : α, dist P F = 0 ∧ F ∈ line A B)
  (h_bdf_circumcenter : circumcenter B D F O1)
  (h_cde_circumcenter : circumcenter C D E O2) :
  is_concyclic O1 O2 E F ↔ orthocenter (⟨A, B, C, h_tri_acute _ _ _⟩ : Triangle α) P :=
sorry

end concyclic_iff_orthocenter_l120_120160


namespace even_quadruples_sum_eq_100_l120_120962

theorem even_quadruples_sum_eq_100 : 
  (∃ (m : ℕ), (∀ (x : ℕ), 0 < x → x % 2 = 0 → ∑ i in finset.range 4, x = 100 → m = 18424)) → (18424 / 100 = 184.24) :=
by
  sorry

end even_quadruples_sum_eq_100_l120_120962


namespace solve_differential_eq_l120_120623

noncomputable def general_solution (C₁ C₂ C₃ : ℝ) : ℝ → ℝ :=
  λ x, C₁ + C₂ * exp x * cos x + C₃ * exp x * sin x +
       (1 / 65) * (cos (4 * x) - (7 / 4) * sin (4 * x)) +
       (1 / 10) * (1 / 2 * sin (2 * x) - cos (2 * x)) +
       (3 / 2) * x

theorem solve_differential_eq (C₁ C₂ C₃ : ℝ) :
  ∃ y : ℝ → ℝ, (∀ x : ℝ,
  deriv (deriv (deriv y x)) x - 2 * deriv (deriv y x) x + 2 * deriv y x = 
  4 * cos x * cos (3 * x) + 6 * sin x ^ 2) ∧
  y = general_solution C₁ C₂ C₃ :=
begin
  sorry
end

end solve_differential_eq_l120_120623


namespace count_triangles_in_fig_l120_120868

-- Define the main hypothesis for the problem
theorem count_triangles_in_fig : 
  let n := 102 -- the number of triangles as found (48 + 36 + 12 + 4 + 2)
  in ∃ figure ℕ, figure = 102 := by
  -- Detailed mathematical steps are skipped here due to 'sorry'
  sorry

end count_triangles_in_fig_l120_120868


namespace domain_of_function_l120_120289

variable {x : ℝ}

def valid_input (x : ℝ) : Prop := x + 1 ≥ 0 ∧ x ≠ 0

theorem domain_of_function : {x : ℝ | valid_input x} = {x : ℝ | x ∈ Ico (-1 : ℝ) 0 ∪ Ioi 0} :=
by
  sorry

end domain_of_function_l120_120289


namespace factory_production_exceeds_120k_l120_120147

theorem factory_production_exceeds_120k 
  (P₀ : ℕ := 40000)
  (growth_rate : ℝ := 1.20)
  (target : ℕ := 120000)
  (lg2 : ℝ := 0.3010)
  (lg3 : ℝ := 0.4771)
  (log_approx : ℝ := 0.0790) :
  ∃ (n : ℕ), (2014 + n = 2021) ∧ (P₀ * (growth_rate^n) > target) :=
begin
  sorry
end

end factory_production_exceeds_120k_l120_120147


namespace largest_prime_factor_of_sum_of_divisors_of_300_l120_120581

theorem largest_prime_factor_of_sum_of_divisors_of_300 :
  let N := Nat.divisorsSum 300 in Nat.greatestPrimeFactor N = 31 :=
by sorry

end largest_prime_factor_of_sum_of_divisors_of_300_l120_120581


namespace compare_powers_l120_120478

def m := (1 / 3)^(1 / 5)
def n := (1 / 4)^(1 / 3)
def p := (1 / 5)^(1 / 4)

theorem compare_powers : n < p ∧ p < m := by
  -- Define the values explicitly
  let m := (1 : ℝ) / 3 ^ (1 / 5 : ℝ)
  let n := (1 : ℝ) / 4 ^ (1 / 3 : ℝ)
  let p := (1 : ℝ) / 5 ^ (1 / 4 : ℝ)

  -- Comparison logic definition
  have h1 : n < p := by
    sorry -- Insert the detailed comparisons here

  have h2 : p < m := by
    sorry -- Insert the detailed comparisons here

  exact ⟨h1, h2⟩

end compare_powers_l120_120478


namespace enclosed_area_of_curve_l120_120633

-- Statement of the problem in Lean
theorem enclosed_area_of_curve (n : ℕ) (s : ℝ) (arc_length angle_radius : ℝ)
  (h_n : n = 12) 
  (h_s : s = 3)
  (h_arc_length : arc_length = 2 * real.pi / 3) 
  (h_angle_radius : angle_radius = 2 * real.pi / 3) :
  let radius := arc_length / angle_radius in
  let sector_area := angle_radius / (2 * real.pi) * real.pi * radius^2 in
  let total_sector_area := n * sector_area in
  let octagon_area := 2 * (1 + real.sqrt 2) * s^2 in
  total_sector_area + octagon_area = 54 + 54 * real.sqrt 2 + 4 * real.pi :=
by
  -- sorry is used here to indicate the proof is omitted 
  sorry

end enclosed_area_of_curve_l120_120633


namespace determine_T_n_given_S2019_l120_120803

variable {a d : ℚ} -- variables for the first term and common difference of the arithmetic sequence

def S (n : ℕ) : ℚ := n * (2 * a + (n - 1) * d) / 2

def T (n : ℕ) : ℚ := ∑ k in (finset.range n).map nat.succ, S k

theorem determine_T_n_given_S2019 (S2019 : ℚ) (h : S 2019 = S2019) : 
  ∃ n : ℕ, T n = (S2019 * 3 / 2019 * (3027 + a)) :=
sorry

end determine_T_n_given_S2019_l120_120803


namespace greatest_y_value_l120_120626

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -2) : y ≤ 1 :=
sorry

end greatest_y_value_l120_120626


namespace log2_x_leq_0_implies_0_lt_x_leq_1_l120_120527

theorem log2_x_leq_0_implies_0_lt_x_leq_1 (x : ℝ) (h1 : ∃ (a : ℝ), log 2 x = a ∧ a ≤ 0) : 0 < x ∧ x ≤ 1 := by
  sorry

end log2_x_leq_0_implies_0_lt_x_leq_1_l120_120527


namespace solve_log_equation_l120_120269

noncomputable def log_equation_solution (x : ℝ) : Prop :=
  log x - 4 * log 5 = -3

theorem solve_log_equation :
  ∃ x : ℝ, log_equation_solution x ∧ x = 0.625 :=
by
  use 0.625
  unfold log_equation_solution
  sorry

end solve_log_equation_l120_120269


namespace roots_of_equation_l120_120124

theorem roots_of_equation : 
  (∃ s ∈ ([- real.sqrt 14, real.sqrt 14]).to_set, 
  (∀ x ∈ s, (real.sqrt (14 - x^2)) * (real.sin x - real.cos (2 * x)) = 0) ∧ 
  ( set.analyse (eq (set.card s) 6))) :=
sorry

end roots_of_equation_l120_120124


namespace range_of_a_monotonically_decreasing_l120_120531

variable {a : ℝ}

def function_is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≥ f y

def f (x : ℝ) : ℝ := cos x - a * x

theorem range_of_a_monotonically_decreasing :
  function_is_monotonically_decreasing f ↔ 1 ≤ a :=
sorry

end range_of_a_monotonically_decreasing_l120_120531


namespace alice_numbers_l120_120755

theorem alice_numbers (x y : ℝ) (h1 : x * y = 12) (h2 : x + y = 7) : (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) :=
by
  sorry

end alice_numbers_l120_120755


namespace find_m_from_cos_l120_120064

-- Given conditions and required proof
theorem find_m_from_cos (m : ℝ) 
  (h1 : ∃ m, ∃ α, ∃ P : ℝ × ℝ, P = (8 * m, 3) ∧ cos α = -4 / 5 ∧ α ∈ Set.Icc 0 (2 * Real.pi)) :
  m = -1 / 2 :=
by
  -- Adding the mathematical equality derived from the given conditions
  cases h1 with m' h1
  rcases h1 with ⟨α, P, hP, hcos, hα⟩
  have h : (8 * m') / sqrt (64 * m' ^ 2 + 9) = -4 / 5,
  { rw [hcos],
    sorry
  },
  -- Simplifying the equality to find m'
  have hm' := calc
    8 * m' / sqrt (64 * m' ^ 2 + 9) = -4 / 5 : h
    ...                    (more steps here)
    ... : sorry,
  -- Showing that m' must be -1/2
  exact sorry

end find_m_from_cos_l120_120064


namespace roots_count_eq_six_l120_120109

noncomputable def number_of_roots : ℝ := 
  let I := Icc (-real.sqrt 14) (real.sqrt 14)
  in
  let f := λ x, √(14 - x^2) * (real.sin x - real.cos (2 * x))
  in
  set.finite (set_of (λ x, f x = 0) ∩ I).to_finset.card

theorem roots_count_eq_six : number_of_roots = 6 := by
  sorry

end roots_count_eq_six_l120_120109


namespace floor_abs_sum_l120_120789

theorem floor_abs_sum : (Int.floor (Real.abs (-5.7)) + Real.abs (Int.floor (-5.7))) = 11 :=
by
  sorry

end floor_abs_sum_l120_120789


namespace increasing_on_neg_reals_l120_120836

variable (f : ℝ → ℝ)

def even_function : Prop := ∀ x : ℝ, f (-x) = f x

def decreasing_on_pos_reals : Prop := ∀ x1 x2 : ℝ, (0 < x1 ∧ 0 < x2 ∧ x1 < x2) → f x1 > f x2

theorem increasing_on_neg_reals
  (hf_even : even_function f)
  (hf_decreasing : decreasing_on_pos_reals f) :
  ∀ x1 x2 : ℝ, (x1 < 0 ∧ x2 < 0 ∧ x1 < x2) → f x1 < f x2 :=
by sorry

end increasing_on_neg_reals_l120_120836


namespace fulcrum_towards_wood_l120_120904

-- Definitions corresponding to conditions
variables (ρq ρБ ρД : ℝ) (VЧ VД : ℝ) (mЧ mД : ℝ)
variables (h1 : (ρq - ρБ) * VЧ = (ρД - ρБ) * VД)
variables (h2 : ρq > ρД)
variables (h3 : VЧ < VД)
variables (h4 : mЧ < mД)

-- Statement of the theorem
theorem fulcrum_towards_wood :
  ∃ (fulcrum_adjustment : String), fulcrum_adjustment = "The fulcrum must be moved towards the wooden ball." :=
by
  sorry

end fulcrum_towards_wood_l120_120904


namespace simplify_expression_l120_120267

theorem simplify_expression : 8 * (15 / 9) * (-45 / 40) = -1 :=
  by
  sorry

end simplify_expression_l120_120267


namespace vector_dot_product_l120_120472

variables (a b : ℝ × ℝ)
axiom h_a : a = (2, 3)
axiom h_b : b = (-1, 2)

theorem vector_dot_product : (let c := (a.1 + 2 * b.1, a.2 + 2 * b.2) in c.1 * b.1 + c.2 * b.2) = 14 :=
by {
  -- Begin by using the axioms and properties of vectors
  have c_def : let c := (a.1 + 2 * b.1, a.2 + 2 * b.2) in c = (0, 7),
  {
    rw [h_a, h_b],
    simp, -- simplify the vector addition and scalar multiplication
  },
  rw c_def,
  -- Calculate the dot product (0, 7) · (-1, 2)
  simp,
  norm_num, -- simplify the resulting arithmetic
}

end vector_dot_product_l120_120472


namespace maximum_area_l120_120353

-- Define the triangle and the conditions
variable {A B C : ℝ}
variable {a b c : ℝ}

-- Conditions from the problem
def conditions : Prop :=
  (a = 2 ∧ (2 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C)

-- Define the area function
def area (b c : ℝ) : ℝ :=
  (1 / 2) * b * c * (Real.sin A)

-- The theorem stating the maximum area
theorem maximum_area (h : conditions) : 
  ∃ b c, area b c = Real.sqrt 3 := 
sorry

end maximum_area_l120_120353


namespace count_valid_three_digit_numbers_l120_120097

-- Define the set of digits we can use
def digits := {4, 0, 6}

-- Noncomputable to allow sets and subset definitions
noncomputable def valid_digit := {d : ℕ | d = 4 ∨ d = 0 ∨ d = 6}

-- Define what makes a number three-digit without zero at hundreds place
def is_valid_three_digit_number (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n / 10) % 10 in
  let d3 := n % 10 in
  d1 ∈ valid_digit ∧ d2 ∈ valid_digit ∧ d3 ∈ valid_digit ∧ d1 ≠ 0

-- Define the set of all possible valid three-digit numbers, within range 100 to 999
noncomputable def valid_three_digit_numbers : set ℕ :=
  {n | 100 ≤ n ∧ n ≤ 999 ∧ is_valid_three_digit_number n}

-- Statement to prove
theorem count_valid_three_digit_numbers : 
  Finset.card (Finset.filter is_valid_three_digit_number (Finset.range 1000)) = 4 :=
sorry

end count_valid_three_digit_numbers_l120_120097


namespace roots_count_eq_six_l120_120107

noncomputable def number_of_roots : ℝ := 
  let I := Icc (-real.sqrt 14) (real.sqrt 14)
  in
  let f := λ x, √(14 - x^2) * (real.sin x - real.cos (2 * x))
  in
  set.finite (set_of (λ x, f x = 0) ∩ I).to_finset.card

theorem roots_count_eq_six : number_of_roots = 6 := by
  sorry

end roots_count_eq_six_l120_120107


namespace BK_parallel_AE_l120_120823

variables {A B C D E K : Type}
variables [linear_ordered_field A B C D E K]
variables [affine_space ℝ A]
variables {P : Type} [affine_space ℝ P]
variables (a b c d e k : P)
variables [convex (set.insert b {c, d, e})]

-- Conditions
def AE_parallel_CD : Prop := affine_parallel (ℕ) (![a, e]) (![c, d])
def AB_eq_BC : Prop := (dist a b = dist b c)

-- Definitions for the points
def angle_bisector_A : P := sorry
def angle_bisector_C : P := sorry

def K : P := sorry

-- Main theorem to be proven
theorem BK_parallel_AE
  (h1 : AE_parallel_CD)
  (h2 : AB_eq_BC)
  (h3 : angle_bisector_A = K)
  (h4 : angle_bisector_C = K)
  : affine_parallel (ℕ) (![b, K]) (![a, e]) :=
sorry

end BK_parallel_AE_l120_120823


namespace henry_collection_cost_l120_120352

def initial_figures : ℕ := 3
def total_needed : ℕ := 8
def cost_per_figure : ℕ := 6

theorem henry_collection_cost : 
  let needed_figures := total_needed - initial_figures
  let total_cost := needed_figures * cost_per_figure
  total_cost = 30 := 
by
  let needed_figures := total_needed - initial_figures
  let total_cost := needed_figures * cost_per_figure
  sorry

end henry_collection_cost_l120_120352


namespace exists_infinitely_many_pairs_l120_120252

theorem exists_infinitely_many_pairs
  (k : ℕ) (hk : k > 1) :
  ∃ (m n : ℕ), m = 2^k - 2 ∧ n = 2^k * (2^k - 2) ∧
  (∀ p : ℕ, p.prime → p ∣ m ↔ p ∣ n) ∧
  (∀ p : ℕ, p.prime → p ∣ (m + 1) ↔ p ∣ ((2^k - 1) ^ 2)) :=
by
  sorry

end exists_infinitely_many_pairs_l120_120252


namespace sample_size_stratified_sampling_l120_120678

theorem sample_size_stratified_sampling : 
    (∃ n_A n_B n_C : ℕ, 
        n_A + n_B + 2 = 9 ∧ 
        n_C = 2 ∧
        n_A > 0 ∧ n_B > 0) :=
by
  use 3, 4, 2
  simp
  sorry

end sample_size_stratified_sampling_l120_120678


namespace perfect_cubes_between_2_pow_9_and_2_pow_17_l120_120519

def countPerfectCubesBetween (a b : ℕ) : ℕ :=
  let lower := Nat.ceil ((Real.sqrt (Real.sqrt (a : ℝ))))
  let upper := Nat.floor ((Real.sqrt (Real.sqrt (b : ℝ))))
  upper - lower + 1

theorem perfect_cubes_between_2_pow_9_and_2_pow_17 :
  countPerfectCubesBetween (2^9 + 1) (2^17 + 1) = 42 :=
by
  sorry

end perfect_cubes_between_2_pow_9_and_2_pow_17_l120_120519


namespace sum_coordinates_B_l120_120832

-- Define the coordinates of point A and the midpoint M
def A := (5, -1)
def M := (4, 3)

-- Define the coordinates of point B
variables (x y : ℤ)

-- Define the conditions based on the midpoint formula
def condition_x := (A.1 + x)/2 = M.1
def condition_y := (A.2 + y)/2 = M.2

-- Define the sum of the coordinates of point B
def sum_of_coordinates := x + y

-- State the theorem
theorem sum_coordinates_B (h1 : condition_x) (h2 : condition_y) : sum_of_coordinates = 10 :=
sorry

end sum_coordinates_B_l120_120832


namespace scientific_notation_of_population_l120_120226

theorem scientific_notation_of_population :
  (141260 : ℝ) = 1.4126 * 10^5 :=
sorry

end scientific_notation_of_population_l120_120226


namespace tan_theta_eq_sqrt3_l120_120454

theorem tan_theta_eq_sqrt3 (θ : ℝ) (h1 : (π / 4) < θ) (h2 : θ < (π / 2)) 
  (h3 : tan θ + tan (2 * θ) + tan (3 * θ) + tan (4 * θ) = 0) :
  tan θ = Real.sqrt 3 :=
by
  sorry

end tan_theta_eq_sqrt3_l120_120454


namespace area_of_figure_l120_120762

noncomputable def curve_x (t : ℝ) : ℝ := 8 * real.sqrt 2 * cos t ^ 3
noncomputable def curve_y (t : ℝ) : ℝ := real.sqrt 2 * sin t ^ 3

-- The required proof statement
theorem area_of_figure :
  ∫ t in -π / 4..π / 4, (curve_x t) * (deriv curve_y t) =
  (3 * π / 2) + 2 := by
  sorry

end area_of_figure_l120_120762


namespace min_students_scoring_at_least_60_l120_120265

theorem min_students_scoring_at_least_60
  (total_score : ℕ)
  (top_scores : list ℕ)
  (min_score : ℕ)
  (max_same_score : ℕ) :
  total_score = 8250 → top_scores = [88, 85, 80] → min_score = 30 → max_same_score = 3 →
  ∃ (n : ℕ), n ≥ 61 ∧ (∀ scores : list ℕ, list.length scores = n ∧ 
    list.all scores (λ x, x ≥ 60) → list.sum scores = 8250) :=
begin
  sorry
end

end min_students_scoring_at_least_60_l120_120265


namespace divide_triangle_l120_120897

theorem divide_triangle (H : ℝ) (k p : ℕ) (h : ℝ)
  (h_eq : h = H / real.sqrt ↑k) :
  ∃ (lines : fin p → ℝ), 
    (∀ i, 0 ≤ lines i ∧ lines i ≤ H) ∧
    (∀ i j, i ≠ j → lines i ≠ lines j) ∧
    (∀ i, area_of_segment lines i = (1 - (1 / ↑k)) * triangle_area / p) :=
begin
  sorry
end

end divide_triangle_l120_120897


namespace cubic_roots_quadratic_l120_120090

theorem cubic_roots_quadratic (A B C p : ℚ)
  (hA : A ≠ 0)
  (h1 : (∀ x : ℚ, A * x^2 + B * x + C = 0 ↔ x = (root1) ∨ x = (root2)))
  (h2 : root1 + root2 = - B / A)
  (h3 : root1 * root2 = C / A)
  (new_eq : ∀ x : ℚ, x^2 + p*x + q = 0 ↔ x = root1^3 ∨ x = root2^3) :
  p = (B^3 - 3 * A * B * C) / A^3 :=
by
  sorry

end cubic_roots_quadratic_l120_120090


namespace scientific_notation_of_population_l120_120227

theorem scientific_notation_of_population :
  (141260 : ℝ) = 1.4126 * 10^5 :=
sorry

end scientific_notation_of_population_l120_120227


namespace problem_I_problem_II_l120_120500

-- Define the function f as given
def f (x m : ℝ) : ℝ := x^2 + (m-1)*x - m

-- Problem (I)
theorem problem_I (x : ℝ) : -2 < x ∧ x < 1 ↔ f x 2 < 0 := sorry

-- Problem (II)
theorem problem_II (m : ℝ) : ∀ x, f x m + 1 ≥ 0 ↔ -3 ≤ m ∧ m ≤ 1 := sorry

end problem_I_problem_II_l120_120500


namespace one_fourth_one_third_two_fifths_l120_120977

theorem one_fourth_one_third_two_fifths (N : ℝ)
  (h₁ : 0.40 * N = 300) :
  (1/4) * (1/3) * (2/5) * N = 25 := 
sorry

end one_fourth_one_third_two_fifths_l120_120977


namespace trululu_not_exist_l120_120752

-- Definitions for conditions
def Weekday : Type := { day : String // day ≠ "Saturday" ∧ day ≠ "Sunday" }

-- Assume facts from the problem:
variables (day : Weekday)
variables (tweedle1_statement : Prop) (tweedle2_statement : Prop)
variable (barmaglot : String → Prop)

-- Condition: The second individual's statement is always true 
axiom tweedle2_true : tweedle2_statement = True

-- Condition: Barmaglot lies on Monday, Tuesday, Wednesday
axiom barmaglot_behavior :
  ∀ (d : String), d = "Monday" ∨ d = "Tuesday" ∨ d = "Wednesday" → barmaglot d = False

-- Problem statement in Lean 4
theorem trululu_not_exist (day = "Thursday") : ¬ tweedle1_statement :=
by {
  sorry
}

end trululu_not_exist_l120_120752


namespace problem_part1_problem_part2_l120_120014

noncomputable def f (x : ℝ) : ℝ :=
  Math.cos x + Math.cos (x + Real.pi / 2)

theorem problem_part1 : f(Real.pi / 12) = Real.sqrt 2 / 2 := 
  sorry

theorem problem_part2 (α β : ℝ) (hαβ: α ∈ Ioo (-Real.pi / 2) 0 ∧ β ∈ Ioo (-Real.pi / 2) 0)
  (h1 : f(α + 3 * Real.pi / 4) = -3 * Real.sqrt 2 / 5)
  (h2 : f(Real.pi / 4 - β) = -5 * Real.sqrt 2 / 13) :
  Real.cos (α + β) = 16 / 65 :=
  sorry

end problem_part1_problem_part2_l120_120014


namespace exists_disjoint_sets_l120_120985

theorem exists_disjoint_sets :
  ∃ (A : Finₓ 2014 → Set ℕ) 
    (h_nonempty : ∀ k, A k ≠ ∅)
    (h_disjoint : ∀ i j, i ≠ j → A i ∩ A j = ∅)
    (h_union : ⋃ i, A i = Set.univ \ {0}),
    ∀ a b : ℕ, a > 0 → b > 0 → 
    ∃ k, a ∈ A k ∧ b ∈ A k ∨ a ∈ A k ∧ gcd a b ∈ A k ∨ b ∈ A k ∧ gcd a b ∈ A k :=
by
  -- Proof omitted
  sorry

end exists_disjoint_sets_l120_120985


namespace cistern_length_l120_120367

-- Definitions of the given conditions
def width : ℝ := 4
def depth : ℝ := 1.25
def total_wet_surface_area : ℝ := 49

-- Mathematical problem: prove the length of the cistern
theorem cistern_length : ∃ (L : ℝ), (L * width + 2 * L * depth + 2 * width * depth = total_wet_surface_area) ∧ L = 6 :=
by
sorry

end cistern_length_l120_120367


namespace shell_thickness_l120_120714

theorem shell_thickness (R : ℝ) (x : ℝ) (h : (4 / 3) * π * ((R + x)^3 - R^3) = (4 / 3) * π * R^3) : 
  x = (real.cbrt 2 - 1) * R :=
by
  sorry

end shell_thickness_l120_120714


namespace calculate_f_ff_f_27_l120_120217

def f (x : ℝ) : ℝ := 
  if x < 15 then x^3 - 4 
  else x - 20

theorem calculate_f_ff_f_27 : f (f (f 27)) = 319 := 
by
  -- formal proof goes here
  sorry

end calculate_f_ff_f_27_l120_120217


namespace number_of_vehicles_l120_120368

-- Definitions
def time : ℕ := 115
def bridge_length : ℕ := 298
def speed : ℕ := 4
def vehicle_length : ℕ := 6
def distance_between_vehicles : ℕ := 20

-- Theorem statement
theorem number_of_vehicles :
  let total_distance := speed * time in
  let convoy_length := total_distance - bridge_length in
  let n := (convoy_length + distance_between_vehicles) / (vehicle_length + distance_between_vehicles) in
  n = 7 := 
by
  sorry

end number_of_vehicles_l120_120368


namespace isosceles_triangle_properties_l120_120495

noncomputable theory

-- Define the conditions
variables (m h : ℝ)

-- Define the resultant sides, area, inradius, and circumradius
def side_1 : ℝ := (m^2 + h^2) / (2 * h)
def side_2 : ℝ := (m^2 + h^2)^2 / (4 * h * (m^2 - h^2))
def area : ℝ := (m^2 + h^2)^2 / (8 * h * (m^2 - h^2))
def inradius : ℝ := (m^2 + h^2) / (4 * m)
def circumradius : ℝ := (m^2 + h^2)^3 / (16 * m * h^2 * (m^2 - h^2))

-- The theorem stating the relations
theorem isosceles_triangle_properties :
  ∃ (a b T ρ R : ℝ),
  a = side_1 m h ∧
  b = side_2 m h ∧
  T = area m h ∧
  ρ = inradius m h ∧
  R = circumradius m h := 
by
  use (side_1 m h)
  use (side_2 m h)
  use (area m h)
  use (inradius m h)
  use (circumradius m h)
  split
  { refl }
  split
  { refl }
  split
  { refl }
  split
  { refl }
  { refl }

end isosceles_triangle_properties_l120_120495


namespace problem_1_part_1_problem_1_part_2_problem_2_l120_120141

-- Definitions for weakly increasing functions
def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x < f y

def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f y < f x

def weakly_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  is_increasing_on f I ∧ is_decreasing_on (λ x, f x / x) I

-- Problem statements translated to Lean 4
theorem problem_1_part_1 : weakly_increasing (λ x : ℝ, x + 4) { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

theorem problem_1_part_2 : ¬weakly_increasing (λ x : ℝ, x^2 + 4x + 2) { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

theorem problem_2 (b m : ℝ) :
  weakly_increasing (λ x : ℝ, x^2 + (m - 1 / 2) * x + b) { x : ℝ | 0 < x ∧ x ≤ 1 } ↔
  (m ≥ 0.5) ∧ (b ≥ 1) :=
sorry

end problem_1_part_1_problem_1_part_2_problem_2_l120_120141


namespace how_many_three_digit_numbers_without_5s_and_8s_l120_120130

def is_valid_hundreds_digit (d : ℕ) : Prop := d ≠ 0 ∧ d ≠ 5 ∧ d ≠ 8
def is_valid_digit (d : ℕ) : Prop := d ≠ 5 ∧ d ≠ 8

theorem how_many_three_digit_numbers_without_5s_and_8s : 
  (∃ count : ℕ, count = 
    (∑ d1 in (finset.range 10).filter is_valid_hundreds_digit, 
      ∑ d2 in (finset.range 10).filter is_valid_digit, 
        ∑ d3 in (finset.range 10).filter is_valid_digit, 1)) = 448 :=
by
  sorry

end how_many_three_digit_numbers_without_5s_and_8s_l120_120130


namespace trapezium_area_l120_120343

theorem trapezium_area (a b h : ℝ) (h₁ : a = 20) (h₂ : b = 16) (h₃ : h = 15) : 
  1 / 2 * (a + b) * h = 270 :=
by
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end trapezium_area_l120_120343


namespace teachers_lineup_l120_120895

structure SchoolTeachers where
  teachers_A : Fin 2
  teachers_B : Fin 2
  teachers_C : Fin 1

def no_adjacent_same_school (teachers : List (Fin 5)) : Prop :=
  ∀ i, (i < 4) → (teachers[i] != teachers[i + 1])

theorem teachers_lineup : ∃ (N : Nat), 
  let schools := SchoolTeachers.mk (Fin.mk 0 (by linarith)) (Fin.mk 1 (by linarith)) (Fin.mk 2 (by linarith))
  ∧ no_adjacent_same_school [schools.teachers_A.val, schools.teachers_A.succ.val, schools.teachers_B.val, schools.teachers_B.succ.val, schools.teachers_C.val] 
  ∧ (N = 48) :=
by
  sorry

end teachers_lineup_l120_120895


namespace distinct_arithmetic_progression_roots_l120_120466

theorem distinct_arithmetic_progression_roots (a b : ℝ) : 
  (∃ (d : ℝ), d ≠ 0 ∧ ∀ x, x^3 + a * x + b = 0 ↔ x = -d ∨ x = 0 ∨ x = d) → a < 0 ∧ b = 0 :=
by
  sorry

end distinct_arithmetic_progression_roots_l120_120466


namespace medium_sized_fir_trees_count_l120_120308

theorem medium_sized_fir_trees_count 
  (total_trees : ℕ) (ancient_oaks : ℕ) (saplings : ℕ)
  (h1 : total_trees = 96)
  (h2 : ancient_oaks = 15)
  (h3 : saplings = 58) :
  total_trees - ancient_oaks - saplings = 23 :=
by 
  sorry

end medium_sized_fir_trees_count_l120_120308


namespace average_rate_trip_l120_120318

noncomputable def run_distance : ℝ := 5.2
noncomputable def run_rate : ℝ := 9.5
noncomputable def swim_distance : ℝ := 5.2
noncomputable def swim_rate : ℝ := 4.2

noncomputable def time_run : ℝ := (run_distance / run_rate) * 60
noncomputable def time_swim : ℝ := (swim_distance / swim_rate) * 60
noncomputable def total_time : ℝ := time_run + time_swim
noncomputable def total_distance : ℝ := run_distance + swim_distance

noncomputable def average_rate (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem average_rate_trip : average_rate total_distance total_time ≈ 0.0971 := sorry

end average_rate_trip_l120_120318


namespace compute_n_sum_of_odds_l120_120806

theorem compute_n_sum_of_odds (n : ℕ) (hn : 0 < n) :
  9000 ≤ (Finset.range (2 * n)).filter (λ k, (n^2 - n) < 2*k+1 ∧ 2*k+1 < (n^2 + n)).sum id ∧ 
  (Finset.range (2 * n)).filter (λ k, (n^2 - n) < 2*k+1 ∧ 2*k+1 < (n^2 + n)).sum id ≤ 10000 
  ↔ n = 21 := by
  sorry

end compute_n_sum_of_odds_l120_120806


namespace largest_prime_factor_always_divides_sum_of_sequence_l120_120373

theorem largest_prime_factor_always_divides_sum_of_sequence
    (sequence : List ℕ)
    (h₁ : ∀ i, i < sequence.length - 1 → 
              (sequence.get i % 100, sequence.get i / 10 % 10) = 
              (sequence.get (i + 1) / 100 % 10, sequence.get (i + 1) / 10 % 10))
    (h₂ : (sequence.get (sequence.length - 1) % 100, sequence.get (sequence.length - 1) / 10 % 10) = 
              (sequence.get 0 / 100 % 10, sequence.get 0 / 10 % 10))
    (h₃ : ∀ i, 100 ≤ sequence.get i ∧ sequence.get i < 1000) :
  37 ∣ sequence.sum :=
by
  sorry

end largest_prime_factor_always_divides_sum_of_sequence_l120_120373


namespace isosceles_triangle_angle_CBD_l120_120716

theorem isosceles_triangle_angle_CBD (A B C D : Type) 
    (isosceles : AC = BC) 
    (angle_C_50 : m∠C = 50) : 
    m∠CBD = 115 :=
by 
   sorry

end isosceles_triangle_angle_CBD_l120_120716


namespace min_games_needed_l120_120801

axiom boys_girls_count : ∃ (boys girls : ℕ), boys = 5 ∧ girls = 5
axiom each_game_spec : ∀ (boys girls : ℕ), boys = 5 → girls = 5 → ∀ g1 g2 b1 b2 : ℕ, g1 ≠ g2 ∧ b1 ≠ b2

noncomputable def least_number_of_games (boys girls : ℕ) : ℕ :=
  if h : boys = 5 ∧ girls = 5 then 25 else 0

theorem min_games_needed : ∀ (boys girls : ℕ), boys = 5 → girls = 5 → least_number_of_games boys girls = 25 :=
by {
  intros boys girls h1 h2,
  simp [least_number_of_games],
  exact if_pos ⟨h1, h2⟩
}

#check min_games_needed

end min_games_needed_l120_120801


namespace morgan_first_sat_score_l120_120970

theorem morgan_first_sat_score (x : ℝ) (h : 1.10 * x = 1100) : x = 1000 :=
sorry

end morgan_first_sat_score_l120_120970


namespace slope_of_line_through_points_l120_120376

theorem slope_of_line_through_points (x1 y1 x2 y2 : ℝ) (hx : x1 = 3) (hy : y1 = 4) (hx_intercept : x2 = 1) (hy_intercept : y2 = 0) : 
  let m := (y2 - y1) / (x2 - x1) 
  in m = 2 := 
by 
  sorry

end slope_of_line_through_points_l120_120376


namespace cartesian_eq_curve_C2_min_distance_curves_C1_C2_l120_120859

noncomputable def curve_C1_param (t : ℝ) : ℝ × ℝ := (2 * t - 1, -4 * t - 2)

noncomputable def curve_C2_polar (θ : ℝ) : ℝ := 2 / (1 - Real.cos θ)

theorem cartesian_eq_curve_C2 :
  ∀ x y : ℝ, (∃ θ : ℝ, x = 2 / (1 - Real.cos θ) ∧ y = 2 * Real.sin θ) ↔ y^2 = 4 * (x - 1) :=
by sorry

theorem min_distance_curves_C1_C2 :
  (∀ t θ : ℝ, t ∈ ℝ ∧ θ ∈ ℝ → (2 * (2 * t - 1) + (-4 * t - 2) + 4 = 0)) →
  ∃ d : ℝ, d = (2 * abs (θ^2 + θ + 1) / (Real.sqrt 5)) ∧ d ≥ 3 * Real.sqrt 5 / 10 :=
by sorry

end cartesian_eq_curve_C2_min_distance_curves_C1_C2_l120_120859


namespace maximize_area_intersection_correct_l120_120390

noncomputable def maximize_area_intersection (A B C D K : Point) (h_trap: is_trapezoid A B C D) (h_on_base: K ∈ Segment A D) : Point :=
  let M : Point := Point_on_BC_div (Ratio AK KD)
  M

theorem maximize_area_intersection_correct (A B C D K M : Point) (h_trap: is_trapezoid A B C D) (h_on_base: K ∈ Segment A D) (h_div: divides_in_ratio M B C (Ratio AK KD)) :
  (∀ M', ¬divides_in_ratio M' B C (Ratio AK KD) → area_intersection_triangles A M D B K C ≤ area_intersection_triangles A M' D B K C) :=
sorry

end maximize_area_intersection_correct_l120_120390


namespace exponent_multiplication_l120_120768

theorem exponent_multiplication :
  (-1 / 2 : ℝ) ^ 2022 * (2 : ℝ) ^ 2023 = 2 :=
by sorry

end exponent_multiplication_l120_120768


namespace sum_fraction_ge_n_l120_120961

theorem sum_fraction_ge_n {n : ℕ} (hn : 0 < n) 
  (a : Fin n → ℕ) (h_distinct : Function.Injective a)
  (h_positive : ∀ i, 0 < a i) :
  ∑ i in Finset.range n, (a i.succ) / i.succ ≥ n := 
sorry

end sum_fraction_ge_n_l120_120961


namespace solve_quadratic_sum_l120_120639

theorem solve_quadratic_sum (a b : ℕ) (x : ℝ) (h₁ : x^2 + 10 * x = 93)
  (h₂ : x = Real.sqrt a - b) (ha_pos : 0 < a) (hb_pos : 0 < b) : a + b = 123 := by
  sorry

end solve_quadratic_sum_l120_120639


namespace isabel_candy_left_l120_120562

variable (x y z : ℕ) -- Assuming x, y, z are non-negative integers

theorem isabel_candy_left (x y z : ℕ) : 
  let initial_candy := 325
  let friend_candy := 145
  let total_received := initial_candy + friend_candy + x + y
  let candy_left := total_received - z
  in candy_left = 470 + x + y - z :=
by
  let initial_candy := 325
  let friend_candy := 145
  let total_received := initial_candy + friend_candy + x + y
  let candy_left := total_received - z
  sorry

end isabel_candy_left_l120_120562


namespace molly_age_l120_120344

-- Definitions based on conditions
def ratio (S M : ℕ) : Prop := S * 3 = M * 4
def future_age (S : ℕ) : Prop := S + 6 = 42

-- The main theorem stating the problem and the answer
theorem molly_age
  (S M : ℕ)
  (h1 : ratio S M)
  (h2 : future_age S) :
  M = 27 :=
begin
  sorry
end

end molly_age_l120_120344


namespace average_mowing_per_month_l120_120224

theorem average_mowing_per_month (mow_apr_sep mow_oct_mar total_months : ℕ) :
  (mow_apr_sep = 6 * 15) →
  (mow_oct_mar = 6 * 3) →
  (total_months = 12) →
  (mow_apr_sep + mow_oct_mar) / total_months = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.add_comm (6 * 15) (6 * 3)]
  norm_num
  sorry

end average_mowing_per_month_l120_120224


namespace henrietta_has_three_bedrooms_l120_120864

theorem henrietta_has_three_bedrooms
  (living_room_walls_sqft : ℕ)
  (bedroom_walls_sqft : ℕ)
  (num_bedrooms : ℕ)
  (gallon_coverage_sqft : ℕ)
  (h1 : living_room_walls_sqft = 600)
  (h2 : bedroom_walls_sqft = 400)
  (h3 : gallon_coverage_sqft = 600)
  (h4 : num_bedrooms = 3) : 
  num_bedrooms = 3 :=
by
  exact h4

end henrietta_has_three_bedrooms_l120_120864


namespace trajectory_equation_and_minimum_area_l120_120857

-- Define the parabola
def parabola (x : ℝ) : ℝ := (1 / 2) * x^2

-- Define point Q
def Q : ℝ × ℝ := (1, 1)

-- Define the trajectory of point P
def trajectory_P (k : ℝ) : ℝ × ℝ := (k, k - 1)

-- Define the minimum area and the corresponding line equation
def minimum_area_and_line_AB : ℝ × (ℝ → ℝ) := (1, λ x, x)

theorem trajectory_equation_and_minimum_area :
  (∀ x, ∃ y, (x, y) = trajectory_P x) ∧
  (∃ (k : ℝ), minimum_value_triangle_area k ∧ ∀ x, line_eq_AB x x = minimum_area_and_line_AB.2 x) :=
by
  sorry

end trajectory_equation_and_minimum_area_l120_120857


namespace keychain_arrangement_count_l120_120900

theorem keychain_arrangement_count :
  ∃ (n : ℕ), n = 8 ∧ ∀ (house_key car_key work_key garage_key some_other_key : Type),
  ∃ (arrangements : list (list Type)),
  length arrangements = 8 ∧
  ∀ arrangement : list Type, arrangement ∈ arrangements → 
    (arrangement.contains house_key) ∧
    (arrangement.contains car_key) ∧ 
    (arrangement.contains work_key) ∧ 
    (arrangement.contains garage_key) ∧ 
    (arrangement.contains some_other_key) ∧
    (adjacent arrangement house_key car_key) ∧
    (adjacent arrangement work_key garage_key) :=
sorry

/-- Helper function to check if two elements are adjacent in a circular list --/
def adjacent {α : Type*} (l : list α) (x y : α) : Prop :=
x ≠ y ∧ (∃ (i j : ℕ), l.nth i = some x ∧ l.nth ((j + 1) % l.length) = some y)

end keychain_arrangement_count_l120_120900


namespace modulus_conjugate_of_z_l120_120034

noncomputable def z : ℂ := 1 + (1 / complex.i)

theorem modulus_conjugate_of_z :
  complex.abs (complex.conj z) = real.sqrt 2 :=
sorry

end modulus_conjugate_of_z_l120_120034


namespace math_problem_l120_120532

open Real

-- Conditions extracted from the problem
def cond1 (a b : ℝ) : Prop := -|2 - a| + b = 5
def cond2 (a b : ℝ) : Prop := -|8 - a| + b = 3
def cond3 (c d : ℝ) : Prop := |2 - c| + d = 5
def cond4 (c d : ℝ) : Prop := |8 - c| + d = 3
def cond5 (a c : ℝ) : Prop := 2 < a ∧ a < 8
def cond6 (a c : ℝ) : Prop := 2 < c ∧ c < 8

-- Proof problem: Given the conditions, prove that a + c = 10
theorem math_problem (a b c d : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) (h3 : cond3 c d) (h4 : cond4 c d)
  (h5 : cond5 a c) (h6 : cond6 a c) : a + c = 10 := 
by
  sorry

end math_problem_l120_120532


namespace heartsuit_example_l120_120464

def heartsuit (x y : ℝ) : ℝ := x + 1 / y

theorem heartsuit_example : heartsuit 3 (heartsuit 3 3) = 33 / 10 :=
by sorry

end heartsuit_example_l120_120464


namespace part_one_part_two_l120_120880

def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃a b⦄, a ∈ I → b ∈ I → a < b → f a < f b

def is_single_anti_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  is_increasing_on f I ∧ is_increasing_on (λ x, f x / x) I

def f (x : ℝ) := Real.log x
def g (x : ℝ) := 2 * x + 2 / x + a * Real.log x

theorem part_one : ¬ is_single_anti_decreasing f (set.Ioc 0 1) :=
by sorry

theorem part_two (a : ℝ) : is_single_anti_decreasing g (set.Ici 1) → 0 ≤ a ∧ a ≤ 4 :=
by sorry

end part_one_part_two_l120_120880


namespace speed_downstream_is_correct_l120_120666

-- Definitions corresponding to the conditions
def speed_boat_still_water : ℕ := 60
def speed_current : ℕ := 17

-- Definition of speed downstream from the conditions and proving the result
theorem speed_downstream_is_correct :
  speed_boat_still_water + speed_current = 77 :=
by
  -- Proof is omitted
  sorry

end speed_downstream_is_correct_l120_120666


namespace isosceles_triangle_length_pq_l120_120549

open Real

/-- In an isosceles triangle PQR with PQ = PR, the medians QS and RS meet at the centroid S and are
perpendicular. If QS = 10, RS = 10, and ∠QPS = 30°, then PQ = 10√3. -/
theorem isosceles_triangle_length_pq 
  (P Q R S : ℝ)
  (hPQR_isosceles : PQ = PR)
  (hQRS_perpendicular : ∠QPS = 30)
  (hQS : QS = 10)
  (hRS : RS = 10)
  (hMedians_perpendicular : (medians_intersect_at_centroid Q R S ∧ perpendicular QS RS)) :
  PQ = 10 * sqrt 3 :=
sorry

end isosceles_triangle_length_pq_l120_120549


namespace value_range_of_function_l120_120670

theorem value_range_of_function : 
  ∀ (x : ℝ), 0 < (1/2)^(x^2 - 2 * x) ∧ (1/2)^(x^2 - 2 * x) ≤ 2 :=
by
  sorry

end value_range_of_function_l120_120670


namespace wont_appear_at_corner_l120_120274

theorem wont_appear_at_corner (n : ℕ) : 
  let evens := list.range' 1 1005 |>.map (λ x, 2 * x),
      corners := list.range' 0 126 |>.map (λ k, 6 + k * 8)
  in ¬(2010 ∈ corners) :=
by
  let evens := list.range' 1 1005 |>.map (λ x, 2 * x),
      corners := list.range' 0 126 |>.map (λ k, 6 + k * 8)
  show ¬(2010 ∈ corners)
  sorry

end wont_appear_at_corner_l120_120274


namespace incorrect_statement_C_condition_A_condition_B_condition_D_evaluate_statements_l120_120807

open Function

/-- Given the function y = x + 2, show that the claim "when x > 2, y < 4" is incorrect. -/
theorem incorrect_statement_C : ∀ x : ℝ, x > 2 → ¬(x + 2 < 4) :=
by
  intro x hx
  unfold not
  intro h
  linarith

/-- Condition A: The graph passes through the point (1, 3) -/
theorem condition_A : (1 : ℝ) + 2 = 3 := rfl

/-- Condition B: The graph intersects the x-axis at the point (-2, 0) -/
theorem condition_B : (-2 : ℝ) + 2 = 0 := rfl

/-- Condition D: The graph does not pass through the fourth quadrant -/
theorem condition_D : ∀ x : ℝ, x > 0 → ¬((x + 2) < 0) :=
by
  intro x hx
  unfold not
  intro h
  linarith

-- Bundle all statements together to indicate the proven statements and checking
theorem evaluate_statements :
    condition_A ∧ condition_B ∧ incorrect_statement_C ∧ condition_D :=
by
  constructor
  · exact condition_A
  constructor
  · exact condition_B
  constructor
  · exact incorrect_statement_C
  · exact condition_D

end incorrect_statement_C_condition_A_condition_B_condition_D_evaluate_statements_l120_120807


namespace determine_d_minus_b_l120_120415

theorem determine_d_minus_b 
  (a b c d : ℕ) 
  (h1 : a^5 = b^4)
  (h2 : c^3 = d^2)
  (h3 : c - a = 19) 
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  : d - b = 757 := 
  sorry

end determine_d_minus_b_l120_120415


namespace telephone_pole_break_height_l120_120401

/-- A telephone pole that is 10 meters tall was struck by lightning and broken into two pieces.
The top piece, AB, has fallen down such that it forms a 30-degree angle with the ground and remains connected to the main pole at B,
which is still perpendicular to the ground at C. Prove that the height above the ground where the pole broke, BC, is 10/3 meters. --/
theorem telephone_pole_break_height 
  (pole_height : ℝ := 10)
  (angle_AB_ground : ℝ := 30)
  (triangle_ratios : 1 / sqrt 3 * 2 = 2 * 1) :
  ∃ (BC : ℝ), BC = 10 / 3 :=
by
  sorry

end telephone_pole_break_height_l120_120401


namespace curling_teams_l120_120913

-- Define the problem conditions and state the theorem
theorem curling_teams (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
sorry

end curling_teams_l120_120913


namespace monotonicity_f_8_range_a_condition_l120_120023

def f (a : ℝ) (x : ℝ) : ℝ := a * x - sin x / cos x ^ 3

-- Prove that f(x) is increasing on (0, π/4) and decreasing on (π/4, π/2) when a = 8.
theorem monotonicity_f_8 (x : ℝ) (h : 0 < x ∧ x < π / 2) (hf : f 8 x = 8 * x - sin x / cos x ^ 3) :
  (0 < x ∧ x < π / 4 → deriv (f 8) x > 0) ∧ (π / 4 < x ∧ x < π / 2 → deriv (f 8) x < 0) :=
sorry

-- Prove that ∀ x ∈ (0, π/2), f(x) < sin 2x ↔ a ≤ 3.
theorem range_a_condition (a : ℝ) (h : ∀ x, 0 < x ∧ x < π / 2 → f a x < sin (2 * x)) :
  a ≤ 3 :=
sorry

end monotonicity_f_8_range_a_condition_l120_120023


namespace least_number_of_stamps_l120_120410

theorem least_number_of_stamps (s t : ℕ) (h : 5 * s + 7 * t = 50) : s + t = 8 :=
sorry

end least_number_of_stamps_l120_120410


namespace range_of_m_l120_120494

theorem range_of_m (m : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → x1 < 1 → x2 < 1 → y x1 ≥ y x2) → m ≥ 1 :=
by
  let y (x : ℝ) := x ^ 2 - 2 * m * x + 1
  sorry

end range_of_m_l120_120494


namespace nat_square_iff_divisibility_l120_120266

theorem nat_square_iff_divisibility (A : ℕ) :
  (∃ k : ℕ, A = k^2) ↔ (∀ n : ℕ, ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ n ∣ ((A + i) * (A + i) - A)) :=
sorry

end nat_square_iff_divisibility_l120_120266


namespace smallest_altitude_l120_120319

variables (P Q R : Type) [triangle : triangle P Q R] (h1 h2 : ℕ) 

noncomputable def third_altitude (alt1 alt2 : ℕ) : ℕ := 
  -- Function definition will implement the logic above
  sorry

theorem smallest_altitude (P Q R : Type) [triangle P Q R] :
  ∀ (h1 h2 : ℕ), h1 = 5 → h2 = 15 → ∃ h3 : ℕ, third_altitude h1 h2 = h3 ∧ h3 = 7 :=
begin
  intros h1 h2 h1_eq h2_eq,
  use 7,
  split,
  { unfold third_altitude, 
    sorry },  -- Proof for the calculation of third altitude
  { refl }
end

end smallest_altitude_l120_120319


namespace positive_integer_solution_l120_120445

theorem positive_integer_solution (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^y - y^x = 1 ↔ (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) :=
begin
  sorry
end

end positive_integer_solution_l120_120445


namespace minimum_value_2a_plus_b_l120_120831

theorem minimum_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (1 / (a + 1)) + (2 / (b - 2)) = 1 / 2) : 2 * a + b ≥ 16 := 
sorry

end minimum_value_2a_plus_b_l120_120831


namespace part_a_part_b_l120_120197

namespace proof_problem

-- Define the conditions.
def is_in_set_S (x : ℕ) : Prop :=
  ∃ (y m : ℕ), y > 0 ∧ m > 0 ∧ y^2 - 2^m = x^2

-- The first part of the problem: Find all elements of S.
theorem part_a (x : ℕ) : is_in_set_S x ↔ x ∈ {1, 2, 3, 6} ∨ ∃ k, x = 2^(k+2) - 2^(k-1) :=
sorry

-- The second part of the problem: Find all x such that both x and (x + 1) are in S.
theorem part_b (x : ℕ) : is_in_set_S x ∧ is_in_set_S (x + 1) ↔ x = 1 ∨ x = 2 :=
sorry

end proof_problem

end part_a_part_b_l120_120197


namespace locus_of_circle_centers_l120_120558

theorem locus_of_circle_centers (a : ℝ) (x0 y0 : ℝ) :
  { (α, β) | (x0 - α)^2 + (y0 - β)^2 = a^2 } = 
  { (x, y) | (x - x0)^2 + (y - y0)^2 = a^2 } :=
by
  sorry

end locus_of_circle_centers_l120_120558


namespace f_is_odd_l120_120835

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then x - 2013 
  else if x < 0 then x + 2013
  else 0
-- Prove that f(x) = x + 2013 for x < 0, given f is an odd function and f(x) = x - 2013 for x > 0.
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  assume x,
  by_cases h : x > 0,
  { rw [f, f, if_pos h, if_neg (lt_irrefl (-x))],
    simpa using h },
  by_cases h : x = 0,
  { simp [h] },
  { rw [not_lt] at h,
    have h₁: -x > 0 := by linarith,
    rw [f, f, if_neg (h), if_pos h₁],
    simpa using h },
  sorry

end f_is_odd_l120_120835


namespace angle_sum_of_isosceles_triangle_l120_120214

theorem angle_sum_of_isosceles_triangle (A B C P M : Point) 
  (is_isosceles: ∃C : Point, A ≠ B ∧ C ≠ A ∧ C ≠ B ∧ dist A C = dist B C)
  (angle_condition: ∠P A B = ∠P B C)
  (midpoint: midpoint A B M) :
  ∠A P M + ∠B P C = 180 :=
sorry

end angle_sum_of_isosceles_triangle_l120_120214


namespace first_digit_is_6_iff_divisible_by_3_l120_120593

noncomputable def first_digit_after_decimal (n : ℕ) : ℕ :=
  let s := ∑ k in Finset.range (n + 1) \ {0}, k * (k + 1) / n
  in Int.to_nat ((s * 10) % 10)  -- extract the digit after the decimal place

theorem first_digit_is_6_iff_divisible_by_3 (n : ℕ) (h_pos : n > 0) :
  first_digit_after_decimal n = 6 ↔ 3 ∣ n :=
sorry

end first_digit_is_6_iff_divisible_by_3_l120_120593


namespace domino_two_layer_cover_l120_120251

theorem domino_two_layer_cover (n m : ℕ) : 
  ∃ (f1 f2 : (ℕ × ℕ) → bool), 
  (∀ x y, x < 2 * n → y < 2 * m → f1 (x, y) ≠ f2 (x, y)) ∧ 
  (∀ x y, x < 2 * n → y < 2 * m → 
    (f1 (x, y) → f1 (x + 1, y) ∨ f1 (x, y + 1)) ∧ 
    (f2 (x, y) → f2 (x + 1, y) ∨ f2 (x, y + 1))) :=
begin
  sorry
end

end domino_two_layer_cover_l120_120251


namespace exists_three_road_networks_l120_120341

noncomputable def road_network_problem (n : ℕ) (points : Fin n → ℝ × ℝ) : Prop :=
  n ≥ 6 ∧
  (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    ¬ collinear 
      (points i) 
      (points j) 
      (points k)) ∧
  (∃ G1 G2 G3 : Fin n → Fin n → Prop,
    (∀ (Gi : Fin n → Fin n → Prop), Gi = G1 ∨ Gi = G2 ∨ Gi = G3 →
      (∀ i j : Fin n, Gi i j → connected_segment (points i) (points j) ∧ 
        (∀ k l : Fin n, k ≠ i ∨ l ≠ j → ¬ intersect (segment (points i) (points j)) (segment (points k) (points l))))) ∧
    (∀ i j : Fin n, G1 i j → ¬ G2 i j ∧ ¬ G3 i j) ∧
    (∀ i j : Fin n, G2 i j → ¬ G1 i j ∧ ¬ G3 i j) ∧
    (∀ i j : Fin n, G3 i j → ¬ G1 i j ∧ ¬ G2 i j))

theorem exists_three_road_networks 
  (n : ℕ) (points : Fin n → ℝ × ℝ) 
  (h1 : n ≥ 6)
  (h2 : ∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → 
    ¬ collinear 
      (points i) 
      (points j) 
      (points k)) 
  : road_network_problem n points :=
sorry

end exists_three_road_networks_l120_120341


namespace calculate_power_expression_l120_120769

theorem calculate_power_expression : 4 ^ 2009 * (-0.25) ^ 2008 - 1 = 3 := 
by
  -- steps and intermediate calculations go here
  sorry

end calculate_power_expression_l120_120769


namespace weekday_miles_proof_weekend_miles_proof_l120_120968

-- Definition for the weekday scenario
def weekday_miles (total_cost : ℝ) (rental_fee : ℝ) (tax_rate : ℝ) (per_mile_charge : ℝ) : ℝ :=
  let total_rental_fee := rental_fee + (rental_fee * tax_rate)
  let per_mile_charge_cost := total_cost - total_rental_fee
  per_mile_charge_cost / per_mile_charge

-- Definition for the weekend scenario
def weekend_miles (total_cost : ℝ) (rental_fee : ℝ) (tax_rate : ℝ) (per_mile_charge : ℝ) : ℝ :=
  let total_rental_fee := rental_fee + (rental_fee * tax_rate)
  let per_mile_charge_cost := total_cost - total_rental_fee
  per_mile_charge_cost / per_mile_charge

-- Main statements
theorem weekday_miles_proof : weekday_miles 95.74 20.99 0.1 0.25 ≈ 290 :=
by sorry

theorem weekend_miles_proof : weekend_miles 95.74 24.99 0.1 0.25 ≈ 273 :=
by sorry

end weekday_miles_proof_weekend_miles_proof_l120_120968


namespace visibleFactorNumbersCount_l120_120388

-- Define what it means for a number to be a visible factor number
def isVisibleFactorNumber (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ d ∈ digits, d ≠ 0 → n % d = 0

-- Define the range of numbers from 200 to 250
def range200to250 := {n : ℕ | n >= 200 ∧ n <= 250}

-- The main theorem statement
theorem visibleFactorNumbersCount : 
  {n | n ∈ range200to250 ∧ isVisibleFactorNumber n}.card = 22 := 
by sorry

end visibleFactorNumbersCount_l120_120388


namespace find_p_max_area_triangle_l120_120085

-- Define the parabola equation and the circle equation
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus of the parabola
def focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

-- Define the condition of the minimum distance
def min_dist_condition (p : ℝ) : Prop :=
  let F := focus p in
  let distance := λ (F M : (ℝ × ℝ)), Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) in
  ∀ (M : ℝ × ℝ), circle M.1 M.2 → distance F M - 1 = 4

-- Define the first part to prove p = 2 given the minimum distance condition
theorem find_p :
  ∃ (p : ℝ), p > 0 ∧ min_dist_condition p ↔ p = 2 := 
sorry

-- Define the condition for the maximum area of the triangle given tangents to the parabola and a point on the circle
def max_area_condition (p : ℝ) : Prop :=
  ∀ (x y : ℝ), circle x y → 
    let M := (x, y) in
    let tangents := λ (A B M : (ℝ × ℝ)), True in -- Placeholder for tangents
    True -- Placeholder for the maximum area condition

-- Define the second part to prove the maximum area given p = 2
theorem max_area_triangle :
  ∃ (area : ℝ), max_area_condition 2 ↔ area = 20 * Real.sqrt 5 :=
sorry

end find_p_max_area_triangle_l120_120085


namespace equidistant_point_l120_120458

theorem equidistant_point :
  ∃ (x y : ℝ),
    (x = -25 / 2 ∧ y = -6) ∧
    ((x - 2)^2 + y^2 + 1^2 = (x - 1)^2 + (y - 2)^2 + (-1)^2) ∧
    ((x - 2)^2 + y^2 + 1^2 = x^2 + (y - 3)^2 + 3^2) :=
by
  use [-25 / 2, -6]
  split
  { simp }
  split
  { calc
      (-25 / 2 - 2)^2 + (-6)^2 + 1 = (-25 / 2 + -2)^2 + 36 + 1 : by simp
      ... = (1 / 2)^2 + 36 + 1 : by ring
      ... = 1 / 4 + 36 + 1 : by simp
      ... = 145 / 4 : by ring
      ... = ((-25 / 2 + 1)^2 + (-8)^2 + (-1)^2) : by ring
      ... = 145 / 4 : by simp }
  calc
    (-25 / 2 - 2)^2 + (-6)^2 + 1 = (-25 / 2 + -2)^2 + 36 + 1: by simp
    ... =  (29 / 2)^2 + (9)^2 + 9 : by ring
    ... = 1264 / 4, 36, 81, and 9
    ... = ((-25 / 2)^2 + (x - 3)^2 + (3)^2) : by ring
    ... = 1264 / 4 := by simp

end equidistant_point_l120_120458


namespace sum_of_n_with_perfect_square_difference_l120_120429

theorem sum_of_n_with_perfect_square_difference :
  let S := {n : ℕ | ∃ x : ℕ, n^2 - 3000 = x^2}
  ∑ n in S, n = 1872 :=
sorry

end sum_of_n_with_perfect_square_difference_l120_120429


namespace scientific_notation_population_l120_120238

theorem scientific_notation_population :
    ∃ (a b : ℝ), (b = 5 ∧ 1412.60 * 10 ^ 6 = a * 10 ^ b ∧ a = 1.4126) :=
sorry

end scientific_notation_population_l120_120238


namespace probability_of_success_l120_120320

theorem probability_of_success 
  (pA : ℚ) (pB : ℚ) 
  (hA : pA = 2 / 3) 
  (hB : pB = 3 / 5) :
  1 - ((1 - pA) * (1 - pB)) = 13 / 15 :=
by
  sorry

end probability_of_success_l120_120320


namespace jesse_budget_exceeded_l120_120569

noncomputable def totalCarpetArea : ℝ :=
  let rect_area := 5 * (19 * 18 : ℝ)
  let square_area := 5 * (15 * 15 : ℝ)
  let triangle_area := 4 * (1 / 2 * 12 * 10 : ℝ)
  let trapezoid_area := 3 * ((10 + 14) / 2 * 8 : ℝ)
  let circle_area := 2 * (Real.pi * (6 ^ 2 : ℝ))
  let ellipse_area := Real.pi * (14 / 2 : ℝ) * (8 / 2 : ℝ)
  rect_area + square_area + triangle_area + trapezoid_area + circle_area + ellipse_area

def carpetCost (totalArea : ℝ) (costPerSqFt : ℝ) : ℝ :=
  totalArea * costPerSqFt

def budgetSufficient (budget cost : ℝ) : Prop :=
  budget >= cost

theorem jesse_budget_exceeded :
  let total_area := totalCarpetArea
  total_area ≈ 4279.1586 →
  budgetSufficient 10000 (carpetCost total_area 5) = False :=
by
  intros h_area_approx
  unfold totalCarpetArea at h_area_approx
  unfold carpetCost
  unfold budgetSufficient
  sorry

end jesse_budget_exceeded_l120_120569


namespace monotonicity_f_8_range_a_condition_l120_120025

def f (a : ℝ) (x : ℝ) : ℝ := a * x - sin x / cos x ^ 3

-- Prove that f(x) is increasing on (0, π/4) and decreasing on (π/4, π/2) when a = 8.
theorem monotonicity_f_8 (x : ℝ) (h : 0 < x ∧ x < π / 2) (hf : f 8 x = 8 * x - sin x / cos x ^ 3) :
  (0 < x ∧ x < π / 4 → deriv (f 8) x > 0) ∧ (π / 4 < x ∧ x < π / 2 → deriv (f 8) x < 0) :=
sorry

-- Prove that ∀ x ∈ (0, π/2), f(x) < sin 2x ↔ a ≤ 3.
theorem range_a_condition (a : ℝ) (h : ∀ x, 0 < x ∧ x < π / 2 → f a x < sin (2 * x)) :
  a ≤ 3 :=
sorry

end monotonicity_f_8_range_a_condition_l120_120025


namespace min_a4_in_geom_progression_l120_120069

-- Define the sequence terms in a geometric progression with a common ratio
def geo_seq (a1 r : ℚ) (n : ℕ) : ℚ := a1 * r^n

-- Define the conditions for p, q, and r
def coprime_pos_ints (p q : ℕ) : Prop := (Nat.gcd p q = 1) ∧ (p > q) ∧ (p ≥ 2) ∧ (q > p)

-- State the problem as a theorem in Lean 4
theorem min_a4_in_geom_progression : 
  ∃ (a1 r : ℚ) (p q : ℕ), 
    geo_seq a1 (q / p) 3 = 27 ∧ 
    ((q / p > 1) ∧ ¬(q / p).denom = 1 ∧ coprime_pos_ints p q) := 
sorry

end min_a4_in_geom_progression_l120_120069


namespace tangent_line_perpendicular_to_given_line_l120_120140

noncomputable def tangent_line_eq (x y : ℝ) (m : ℝ) : Prop :=
  4 * x - y = m

theorem tangent_line_perpendicular_to_given_line :
  ∃ m : ℝ, 
  ∀ x y : ℝ, 
  (tangent_line_eq ( 1 : ℝ) (1 : ℝ) (3 : ℝ)) ∧ 
  (tangent_line_eq x y m → (4 : ℝ) * x - y = 3)
  sorry

end tangent_line_perpendicular_to_given_line_l120_120140


namespace degrees_for_cherry_pie_l120_120889

theorem degrees_for_cherry_pie
  (n c a b : ℕ)
  (hc : c = 15)
  (ha : a = 10)
  (hb : b = 9)
  (hn : n = 48)
  (half_remaining_cherry : (n - (c + a + b)) / 2 = 7) :
  (7 / 48 : ℚ) * 360 = 52.5 := 
by sorry

end degrees_for_cherry_pie_l120_120889


namespace max_odd_integers_l120_120759

theorem max_odd_integers {a b c d e f g : ℕ} (h1 : a * b * c * d * e * f * g % 2 = 0) : (max (
    if a % 2 = 0 then 1 else 0) +
    (if b % 2 = 0 then 1 else 0) +
    (if c % 2 = 0 then 1 else 0) +
    (if d % 2 = 0 then 1 else 0) +
    (if e % 2 = 0 then 1 else 0) +
    (if f % 2 = 0 then 1 else 0) +
    (if g % 2 = 0 then 1 else 0)) = 6 :=
begin
  sorry
end

end max_odd_integers_l120_120759


namespace decreasing_parabola_range_l120_120005

theorem decreasing_parabola_range : 
  ∀ x : ℝ, (λ y, y = x^2 - 2*x) → (x < 1) :=
sorry

end decreasing_parabola_range_l120_120005


namespace g_of_10_l120_120079

noncomputable def f (x : ℝ) : ℝ := log 3 (x - 1) + 9

def is_inverse (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

theorem g_of_10 :
  (∃ g : ℝ → ℝ, is_inverse f g) → ∃ g : ℝ → ℝ, g 10 = 4 :=
by
  intros h
  sorry

end g_of_10_l120_120079


namespace problem_statement_l120_120995

-- Define the problem parameters with the constraints
def numberOfWaysToDistributeBalls (totalBalls : Nat) (initialDistribution : List Nat) : Nat :=
  -- Compute the number of remaining balls after the initial distribution
  let remainingBalls := totalBalls - initialDistribution.foldl (· + ·) 0
  -- Use the stars and bars formula to compute the number of ways to distribute remaining balls
  Nat.choose (remainingBalls + initialDistribution.length - 1) (initialDistribution.length - 1)

-- The boxes are to be numbered 1, 2, and 3, and each must contain at least its number of balls
def answer : Nat := numberOfWaysToDistributeBalls 9 [1, 2, 3]

-- Statement of the theorem
theorem problem_statement : answer = 10 := by
  sorry

end problem_statement_l120_120995


namespace base7_difference_l120_120793

theorem base7_difference :
  let a := 1 * 7^3 + 2 * 7^2 + 3 * 7^1 + 4 * 7^0
  let b := 2 * 7^2 + 3 * 7^1 + 4 * 7^0
  let diff := a - b
  let result := 6 * 7^2 + 6 * 7^1 + 1 * 7^0
  diff = result := 
begin
  sorry
end

end base7_difference_l120_120793


namespace count_visible_factor_numbers_200_250_l120_120381

def is_visible_factor_number (n : ℕ) : Prop :=
  (∀ d ∈ (repr n).toList.map (λ c => c.toNat - '0'.toNat), d ≠ 0 → n % d = 0)

def count_visible_factor_numbers (a b : ℕ) : ℕ :=
  (a.b.to_list).count is_visible_factor_number

theorem count_visible_factor_numbers_200_250 : count_visible_factor_numbers 200 250 = 21 :=
by sorry

end count_visible_factor_numbers_200_250_l120_120381


namespace merchant_profit_percentage_is_35_l120_120378

noncomputable def cost_price : ℝ := 100
noncomputable def markup_percentage : ℝ := 0.80
noncomputable def discount_percentage : ℝ := 0.25

-- Marked price after 80% markup
noncomputable def marked_price (cp : ℝ) (markup_pct : ℝ) : ℝ :=
  cp + (markup_pct * cp)

-- Selling price after 25% discount on marked price
noncomputable def selling_price (mp : ℝ) (discount_pct : ℝ) : ℝ :=
  mp - (discount_pct * mp)

-- Profit as the difference between selling price and cost price
noncomputable def profit (sp cp : ℝ) : ℝ :=
  sp - cp

-- Profit percentage
noncomputable def profit_percentage (profit cp : ℝ) : ℝ :=
  (profit / cp) * 100

theorem merchant_profit_percentage_is_35 :
  let cp := cost_price
  let mp := marked_price cp markup_percentage
  let sp := selling_price mp discount_percentage
  let prof := profit sp cp
  profit_percentage prof cp = 35 :=
by
  let cp := cost_price
  let mp := marked_price cp markup_percentage
  let sp := selling_price mp discount_percentage
  let prof := profit sp cp
  show profit_percentage prof cp = 35
  sorry

end merchant_profit_percentage_is_35_l120_120378


namespace isosceles_triangle_median_length_l120_120899

noncomputable def median_length (b h : ℝ) : ℝ :=
  let a := Real.sqrt ((b / 2) ^ 2 + h ^ 2)
  let m_a := Real.sqrt ((2 * a ^ 2 + 2 * b ^ 2 - a ^ 2) / 4)
  m_a

theorem isosceles_triangle_median_length :
  median_length 16 10 = Real.sqrt 146 :=
by
  sorry

end isosceles_triangle_median_length_l120_120899


namespace angle_ABC_is_60_degrees_l120_120579

noncomputable def angle_between_points (A B C : ℝ × ℝ × ℝ) : ℝ := 
  let AB := (A.1 - B.1, A.2 - B.2, A.3 - B.3)
  let AC := (C.1 - B.1, C.2 - B.2, C.3 - B.3)
  let dot_product := (AB.1 * AC.1 + AB.2 * AC.2 + AB.3 * AC.3)
  let norm_AB := Real.sqrt (AB.1^2 + AB.2^2 + AB.3^2)
  let norm_AC := Real.sqrt (AC.1^2 + AC.2^2 + AC.3^2)
  let cos_angle := dot_product / (norm_AB * norm_AC)
  Real.acos cos_angle * 180 / Real.pi

theorem angle_ABC_is_60_degrees :
  let A := (-4, 0, 6)
  let B := (-5, -1, 2)
  let C := (-6, -1, 3) 
  angle_between_points A B C = 60 :=
by 
  -- The proof is omitted. This line is only to satisfy the structure of the Lean problem.
  sorry

end angle_ABC_is_60_degrees_l120_120579


namespace owen_profit_l120_120979

noncomputable def total_cost (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ :=
  num_boxes * cost_per_box

noncomputable def total_number_of_masks (num_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  num_boxes * pieces_per_box

noncomputable def total_revenue (packets_sold_100_pieces : ℕ) (price_per_100_packet : ℕ)
                                (baggies_sold_10_pieces : ℕ) (price_per_10_baggie : ℕ) : ℕ :=
  packets_sold_100_pieces * price_per_100_packet + baggies_sold_10_pieces * price_per_10_baggie

noncomputable def profit (revenue : ℕ) (cost : ℕ) : ℕ :=
  revenue - cost

theorem owen_profit :
  let cost := total_cost 12 9,
      num_masks := total_number_of_masks 12 50,
      packets_sold_100 := 3,
      baggies_sold_10 := 30,
      revenue := total_revenue packets_sold_100 12 baggies_sold_10 3 in
  profit revenue cost = 18 :=
by
  let cost := total_cost 12 9
  let num_masks := total_number_of_masks 12 50
  let packets_sold_100 := 3
  let baggies_sold_10 := 30
  let revenue := total_revenue packets_sold_100 12 baggies_sold_10 3
  exact Eq.refl 18

end owen_profit_l120_120979


namespace fraction_of_red_marbles_l120_120887

theorem fraction_of_red_marbles (x : ℕ) (h1 : (2 / 3 : ℚ) * x > 0) :
  (let blue := (2 / 3 : ℚ) * x;
       red := x - blue in
   let new_red := 3 * red;
       new_total := blue + new_red in
   new_red / new_total = (3 / 5 : ℚ)) :=
by {
  -- Since we are only required to provide the mathematical statement,
  -- the proof is omitted.
  sorry
}

end fraction_of_red_marbles_l120_120887


namespace center_of_mass_independence_center_of_mass_constancy_l120_120696

-- Part (a)

theorem center_of_mass_independence
  (O A : ℝ^3)
  (h : A ∈ sphere O R) -- A is inside the sphere with center O
  (l1 l2 l3 : ℝ^3 → ℝ^3) -- three mutually perpendicular lines through A intersecting the sphere at six points
  (int_pts : fin 6 → ℝ^3)
  (h1 : ∀ i, int_pts i ∈ sphere O R)
  (h2 : ∀ i j, i ≠ j → int_pts i ≠ int_pts j) :
  ∀ (P1 P2 P3 : ℝ^3 → ℝ^3)
  (h_perpendicular : P1 ⊥ P2 ∧ P2 ⊥ P3 ∧ P1 ⊥ P3),
  ∃ Q, Q = center_of_mass (fin.elems int_pts) := sorry

-- Part (b)

theorem center_of_mass_constancy
  (O A : ℝ^3)
  (h : A ∈ sphere O R) -- A is inside the sphere with center O
  (vertices : fin 12 → ℝ^3)
  (h_vertices : ∀ i, vertices i ∈ sphere O R)
  (rotated_vertices : fin 12 → ℝ^3)
  (h_rotation : rotated_vertices = rotate vertices) :
  ∃ Q, Q = center_of_mass (fin.elems vertices) ∧ Q = center_of_mass (fin.elems rotated_vertices) := sorry

end center_of_mass_independence_center_of_mass_constancy_l120_120696


namespace range_u_inequality_le_range_k_squared_l120_120512

def D (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = k}

theorem range_u (k : ℝ) (hk : k > 0) :
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k → 0 < x1 * x2 ∧ x1 * x2 ≤ k^2 / 4 :=
sorry

theorem inequality_le (k : ℝ) (hk : k ≥ 1) :
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k →
  (1 / x1 - x1) * (1 / x2 - x2) ≤ (k / 2 - 2 / k)^2 :=
sorry

theorem range_k_squared (k : ℝ) :
  (0 < k^2 ∧ k^2 ≤ 4 * Real.sqrt 5 - 8) ↔
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k →
  (1 / x1 - x1) * (1 / x2 - x2) ≥ (k / 2 - 2 / k)^2 :=
sorry

end range_u_inequality_le_range_k_squared_l120_120512


namespace simplify_fraction_l120_120621

open Complex

theorem simplify_fraction :
  (3 + 3 * I) / (-1 + 3 * I) = -1.2 - 1.2 * I :=
by
  sorry

end simplify_fraction_l120_120621


namespace monotonic_decreasing_4_to_infinity_l120_120797

noncomputable def f (x : ℝ) : ℝ := Real.logb (1/2) (|x| - 4)

theorem monotonic_decreasing_4_to_infinity :
  ∀ x, x ∈ (Set.Ioi 4) → f(x) = Real.logb (1/2) (|x| - 4) ∧ MonotoneDecreasingOn (4, Top) f :=
by
  sorry

end monotonic_decreasing_4_to_infinity_l120_120797


namespace geometric_sequence_solution_l120_120039

variables (a : ℕ → ℝ) (q : ℝ)
-- Given conditions
def condition1 : Prop := abs (a 1) = 1
def condition2 : Prop := a 5 = -8 * a 2
def condition3 : Prop := a 5 > a 2
-- Proof statement
theorem geometric_sequence_solution :
  condition1 a → condition2 a → condition3 a → ∀ n, a n = (-2)^(n - 1) :=
sorry

end geometric_sequence_solution_l120_120039


namespace integral_solution_l120_120348

noncomputable def integral_ex : ℝ → ℝ := by
  sorry

theorem integral_solution :
  ∀ (x : ℝ) (C : ℝ), integral_ex x = 2 * Real.log (Real.abs x) - 1 / (2 * (x + 1)^2) + C :=
by
  sorry

end integral_solution_l120_120348


namespace exists_point_at_distance_l120_120824

def Line : Type := sorry
def Point : Type := sorry
def distance (P Q : Point) : ℝ := sorry

variables (L : Line) (d : ℝ) (P : Point)

def is_at_distance (Q : Point) (L : Line) (d : ℝ) := ∃ Q, distance Q L = d

theorem exists_point_at_distance :
  ∃ Q : Point, is_at_distance Q L d :=
sorry

end exists_point_at_distance_l120_120824


namespace max_bk_at_k_l120_120777
open Nat Real

theorem max_bk_at_k :
  let B_k (k : ℕ) := (choose 2000 k) * (0.1 : ℝ) ^ k
  ∃ k : ℕ, (k = 181) ∧ (∀ m : ℕ, B_k m ≤ B_k k) :=
sorry

end max_bk_at_k_l120_120777


namespace alpha_beta_diff_l120_120062

theorem alpha_beta_diff 
  (α β : ℝ)
  (h1 : α + β = 17)
  (h2 : α * β = 70) : |α - β| = 3 :=
by
  sorry

end alpha_beta_diff_l120_120062


namespace number_of_integers_in_series_l120_120810

theorem number_of_integers_in_series : 
  let series := [sqrt 8192, 8192^(1/3), 8192^(1/4), 8192^(1/5), 8192^(1/6), ..., 8192^(1/n)]
  in count_integers(series) = 1 :=
by
  sorry

end number_of_integers_in_series_l120_120810


namespace calculate_tip_l120_120630
noncomputable theory

def Tayzia_haircut_cost : ℝ := 48
def daughter_haircut_cost : ℝ := 36
def son_haircut_cost : ℝ := 40
def hair_treatment_cost : ℝ := 20
def tip_percentage : ℝ := 0.20

def total_cost : ℝ :=
  Tayzia_haircut_cost + 2 * daughter_haircut_cost + son_haircut_cost + hair_treatment_cost

def tip_amount : ℝ :=
  tip_percentage * total_cost

theorem calculate_tip : tip_amount = 36 := by
  sorry

end calculate_tip_l120_120630


namespace natasha_speed_proof_l120_120699

noncomputable def natasha_average_speed_climbing 
  (hours_climbing : ℝ)
  (hours_descending : ℝ)
  (total_avg_speed : ℝ)
  (total_distance : ℝ := total_avg_speed * (hours_climbing + hours_descending))
  (distance_to_top : ℝ := total_distance / 2)
  (climbing_avg_speed : ℝ := distance_to_top / hours_climbing) : ℝ :=
  climbing_avg_speed

theorem natasha_speed_proof :
  natasha_average_speed_climbing 4 2 1.5 = 1.125 :=
by
  simp [natasha_average_speed_climbing]
  have h1 : total_distance = 1.5 * 6 := rfl
  have h2 : hours_climbing + hours_descending = 6 := rfl
  have h3 : total_distance = 9 := by simp [h1]
  have h4 : distance_to_top = 4.5 := by simp [h3]
  have h5 : climbing_avg_speed = 4.5 / 4 := rfl
  have h6 : climbing_avg_speed = 1.125 := by norm_num
  exact h6

end natasha_speed_proof_l120_120699


namespace distance_walked_on_fifth_day_l120_120555

-- Define the conditions and then the problem
theorem distance_walked_on_fifth_day
    (a : ℕ → ℝ)
    (a1 : ℝ)
    (r : ℝ)
    (sum_a6 : ℝ)
    (h1 : ∀ n, a n = a1 * r ^ n)
    (h2 : r = 1 / 2)
    (h3 : sum (λ (n : ℕ), a n) (range 6) = 378) :
  a 5 = 12 := 
  sorry

end distance_walked_on_fifth_day_l120_120555


namespace area_of_unpainted_region_l120_120685

def width1 : ℝ := 3
def width2 : ℝ := 5
def angle : ℝ := π / 4 -- 45 degrees in radians

noncomputable def area_unpainted (w1 w2 ang : ℝ) : ℝ :=
  w1 * w2 * Real.sin ang

theorem area_of_unpainted_region :
  area_unpainted width1 width2 angle = 15 * Real.sqrt 2 / 2 :=
by
  sorry

end area_of_unpainted_region_l120_120685


namespace num_workers_l120_120346

-- Define the number of workers (n) and the initial contribution per worker (x)
variable (n x : ℕ)

-- Condition 1: The total contribution is Rs. 3 lacs
axiom h1 : n * x = 300000

-- Condition 2: If each worker contributed Rs. 50 more, the total would be Rs. 3.75 lacs
axiom h2 : n * (x + 50) = 375000

-- Proof Problem: Prove that the number of workers (n) is 1500
theorem num_workers : n = 1500 :=
by
  -- The proof will go here
  sorry

end num_workers_l120_120346


namespace dorothy_tax_percentage_l120_120782

/-- Dorothy earns $60000 a year from her work. After paying the taxes, she has $49200 left. 
    Prove that the percentage of her income she needs to pay in taxes is 18%. --/
theorem dorothy_tax_percentage (income : ℕ) (after_taxes : ℕ) (tax_percentage : ℕ) :
  income = 60000 → after_taxes = 49200 → tax_percentage = 18 :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end dorothy_tax_percentage_l120_120782


namespace parallelogram_area_l120_120002

theorem parallelogram_area : 
  let z1 : ℂ := complex.sqrt (5 + 5 * real.sqrt 7 * complex.i), 
      z2 : ℂ := complex.sqrt (3 + 3 * real.sqrt 2 * complex.i) in
  abs (z1 - conjugate z1) * abs (z2 - conjugate z2) = 5 * real.sqrt 7 - 3 * real.sqrt 2 := 
by
  sorry

end parallelogram_area_l120_120002


namespace daria_needs_to_earn_l120_120439

variable (ticket_cost : ℕ) (current_money : ℕ) (total_tickets : ℕ)

def total_cost (ticket_cost : ℕ) (total_tickets : ℕ) : ℕ :=
  ticket_cost * total_tickets

def money_needed (total_cost : ℕ) (current_money : ℕ) : ℕ :=
  total_cost - current_money

theorem daria_needs_to_earn :
  total_cost 90 4 - 189 = 171 :=
by
  sorry

end daria_needs_to_earn_l120_120439


namespace composition_of_two_central_symmetries_is_translation_composition_translation_and_central_symmetry_is_central_symmetry_l120_120697

-- Problem a: Composition of two central symmetries
theorem composition_of_two_central_symmetries_is_translation (O1 O2 A : Point) :
  let S_O1 := central_symmetry O1
  let S_O2 := central_symmetry O2
  is_translation (S_O2 (S_O1 A)) := 
sorry

-- Problem b: Composition of a translation and a central symmetry
theorem composition_translation_and_central_symmetry_is_central_symmetry (O1 O2 : Point) (a : Vector) :
  let T_a := translation a
  let S_O1 := central_symmetry O1
  let S_O2 := central_symmetry (O1 + a / 2)
  is_central_symmetry (S_O2 (T_a (S_O1 _))) :=
sorry

end composition_of_two_central_symmetries_is_translation_composition_translation_and_central_symmetry_is_central_symmetry_l120_120697


namespace focus_of_ellipse_l120_120417

-- Definitions from conditions
def major_axis_endpoints := (1, -2) ∧ (7, -2)
def minor_axis_endpoints := (3, 1) ∧ (3, -5)
def center_of_ellipse := (3, -2)

-- Proof problem
theorem focus_of_ellipse :
  (compute_focus_x_coord major_axis_endpoints minor_axis_endpoints = 3) ∧ 
  (compute_focus_y_coord major_axis_endpoints minor_axis_endpoints = -2) :=
sorry

end focus_of_ellipse_l120_120417


namespace purely_imaginary_m_eq_neg_half_second_quadrant_m_range_l120_120819

noncomputable def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

-- Part (1)
theorem purely_imaginary_m_eq_neg_half (m : ℝ) (h : z m.imaginary = 0) : m = -1 / 2 :=
by
  sorry

-- Part (2)
theorem second_quadrant_m_range (m : ℝ) (h1 : z m.real < 0) (h2 : z m.imaginary > 0) : -1 / 2 < m ∧ m < 1 :=
by
  sorry

end purely_imaginary_m_eq_neg_half_second_quadrant_m_range_l120_120819


namespace geun_bae_fourth_day_jumps_l120_120181

-- Define a function for number of jump ropes Geun-bae does on each day
def jump_ropes (n : ℕ) : ℕ :=
  match n with
  | 0     => 15
  | n + 1 => 2 * jump_ropes n

-- Theorem stating the number of jump ropes Geun-bae does on the fourth day
theorem geun_bae_fourth_day_jumps : jump_ropes 3 = 120 := 
by {
  sorry
}

end geun_bae_fourth_day_jumps_l120_120181


namespace subset_implication_l120_120094

def S : Set ℝ := {x | (x + 2) / (x - 5) < 0}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2 * a + 15 }

theorem subset_implication (a : ℝ) :
  (S ⊆ P a) → a ∈ Icc (-5 : ℝ) (-3 : ℝ) := by
  sorry

end subset_implication_l120_120094


namespace arithmetic_sequence_with_prime_factors_l120_120933

open Nat

theorem arithmetic_sequence_with_prime_factors (n d : ℕ) :
  ∃ (a : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ p : ℕ, Prime p ∧ p ∣ a i ∧ p ≥ i) ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a (i + 1) = a i + d) :=
sorry

end arithmetic_sequence_with_prime_factors_l120_120933


namespace range_of_k_l120_120506

noncomputable def f (x k : ℝ) : ℝ := (Real.exp x / x) + (k / 2) * x^2 - k * x

theorem range_of_k (k : ℝ) :
  (∀ x ∈ Set.Ioi 0, Deriv f x k = 0 → x = 1) →
  k ∈ Set.Ici (- Real.exp 2 / 4) :=
by
  sorry 

end range_of_k_l120_120506


namespace solve_equation_roots_count_l120_120118

theorem solve_equation_roots_count :
  ∀ (x : ℝ), abs x ≤ real.sqrt 14 → 
  (sqrt (14 - x ^ 2) * (sin x - cos (2 * x)) = 0) → 
  (set.count {x | abs x ≤ real.sqrt 14 ∧ (sqrt (14 - x ^ 2) * (sin x - cos (2 * x)) = 0)} = 6) :=
by sorry

end solve_equation_roots_count_l120_120118


namespace amelia_dinner_l120_120412

theorem amelia_dinner :
  let m := 60
  let c1 := 15
  let c2 := c1 + 5
  let d := 0.25 * c2
  in m - (c1 + c2 + d) = 20 := 
by
  sorry

end amelia_dinner_l120_120412


namespace smallest_n_with_739_condition_l120_120288

theorem smallest_n_with_739_condition :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ Nat.gcd m n = 1 ∧ m < n ∧ 
    (∃ k (hk : 0 < k), k < 1000 ∧ 1000*m = 739*n + k) ∧ n = 739 :=
sorry

end smallest_n_with_739_condition_l120_120288


namespace percentage_increase_l120_120292

theorem percentage_increase (lowest_price highest_price : ℝ) (h_low : lowest_price = 15) (h_high : highest_price = 25) :
  ((highest_price - lowest_price) / lowest_price) * 100 = 66.67 :=
by
  sorry

end percentage_increase_l120_120292


namespace minimum_average_score_l120_120317

theorem minimum_average_score (s1 s2 s3 : ℝ) (required_avg : ℝ) (score1 score2 score3 : ℝ) :
  (s1 = 80) ∧ (s2 = 90) ∧ (s3 = 78) ∧ (required_avg = 85) →
  (2 * (required_avg * 5 - score1 - score2 - score3)) / 2 = 88.5 :=
by 
  intros h,
  rcases h with ⟨hs1, hs2, hs3, havg⟩,
  simp [hs1, hs2, hs3, havg],
  linarith

end minimum_average_score_l120_120317


namespace quadratic_real_roots_and_m_values_l120_120091

theorem quadratic_real_roots_and_m_values (m : ℝ) : 
  (∀ (a b c : ℝ), a = 1 → b = m - 2 → c = m - 3 → (b^2 - 4 * a * c) ≥ 0) ∧
  (∀ (x1 x2 : ℝ), (x1 + x2 = -(m - 2)) → (2 * x1 + x2 = m + 1) →
    m = 0 ∨ m = 4 / 3) :=
begin
  split,
  { intros a b c ha hb hc,
    rw [ha, hb, hc],
    calc (m - 2)^2 - 4 * 1 * (m - 3)
        = (m - 2)^2 - 4 * (m - 3) : by ring
    ... = m^2 - 4 * m + 4 - 4 * m + 12 : by ring
    ... = m^2 - 8 * m + 16 : by ring
    ... = (m - 4)^2 : by ring
    ... ≥ 0 : by apply pow_two_nonneg,
  },
  {
    intros x1 x2 hsum heqn,
    have h1 : x1 = 2 * m - 1,
    { 
      -- The corresponding Lean proof would go here, just showing steps
      sorry
    },
    have h2 : (x1 = -2/3) ∨ (x1 = 1), 
    { 
      -- The corresponding Lean proof would go here
      sorry,
    },
    have values_of_m : m = 0 ∨ m = 4 / 3,
    { 
      -- The corresponding Lean proof would go here
      sorry,
    },
    exact values_of_m,
  },
end

end quadratic_real_roots_and_m_values_l120_120091


namespace probability_of_at_least_one_red_is_five_sixths_l120_120539

noncomputable def probability_at_least_one_red : ℚ :=
  let total_events := (finset.card (finset.image (λ (s : finset ℕ), s.card) (finset.range 4).powerset)) - 1 in
  let favorable_events := (finset.card (finset.image (λ (s : finset ℕ), if s.card = 2 then if 0 ∈ s ∨ 1 ∈ s then 1 else 0 else 0) (finset.range 4).powerset)) - 1 in
  favorable_events / total_events

theorem probability_of_at_least_one_red_is_five_sixths 
  (r w : ℕ) (total_balls := r + w) (n := (total_balls.choose 2)) 
  (at_least_one_red := (r.choose 2) + (r.choose 1) * (w.choose 1)) 
  (p := at_least_one_red / n) 
  (total_events := 6) (favorable_events := 5) : 
  r = 2 → w = 2 → p = 5 / 6 := 
by
  intros h1 h2
  have n_calc : n = total_events := 
    by simp [h1, h2, nat.choose, total_events]
  have m_calc : at_least_one_red = favorable_events :=
    by simp [h1, h2, nat.choose, favorable_events]
  simp [n_calc, m_calc, p]
  sorry

end probability_of_at_least_one_red_is_five_sixths_l120_120539


namespace problem_statement_l120_120196

def exists_triangle_with_area_gt_four (M : Finset (EuclideanSpace ℝ (Fin 2))) : Prop :=
  (∀ (a b c : EuclideanSpace ℝ (Fin 2)), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c →
    abs ((a - b) ⬝ (c - b) / 2) > 3) → 
  ∃ (a b c : EuclideanSpace ℝ (Fin 2)), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
    abs ((a - b) ⬝ (c - b) / 2) > 4

theorem problem_statement : ∀ (M : Finset (EuclideanSpace ℝ (Fin 2))), 
  M.card = 5 →
  exists_triangle_with_area_gt_four M :=
sorry

end problem_statement_l120_120196


namespace range_of_a_extrema_of_y_l120_120960

variable {a b c : ℝ}

def setA (a b c : ℝ) : Prop := a^2 - b * c - 8 * a + 7 = 0
def setB (a b c : ℝ) : Prop := b^2 + c^2 + b * c - b * a + b = 0

theorem range_of_a (h: ∃ a b c : ℝ, setA a b c ∧ setB a b c) : 1 ≤ a ∧ a ≤ 9 :=
sorry

theorem extrema_of_y (h: ∃ a b c : ℝ, setA a b c ∧ setB a b c) 
  (y : ℝ) 
  (hy1 : y = a * b + b * c + a * c)
  (hy2 : ∀ x y z : ℝ, setA x y z → setB x y z → y = x * y + y * z + x * z) : 
  y = 88 ∨ y = -56 :=
sorry

end range_of_a_extrema_of_y_l120_120960


namespace sandy_savings_l120_120193

theorem sandy_savings (S : ℝ) :
  let last_year_savings := 0.10 * S
  let this_year_salary := 1.10 * S
  let this_year_savings := 1.65 * last_year_savings
  let P := this_year_savings / this_year_salary
  P * 100 = 15 :=
by
  let last_year_savings := 0.10 * S
  let this_year_salary := 1.10 * S
  let this_year_savings := 1.65 * last_year_savings
  let P := this_year_savings / this_year_salary
  have hP : P = 0.165 / 1.10 := by sorry
  have hP_percent : P * 100 = 15 := by sorry
  exact hP_percent

end sandy_savings_l120_120193


namespace abc_sum_eq_three_bc_l120_120011

-- Definitions and conditions
variables {A B C O H B' C' : Type} 
variables [HasAngle A] [IsCircumcenter O A B C] [IsOrthocenter H A B C]
variables [Intersects (line_through O H) (line_through A B) B']
variables [Intersects (line_through O H) (line_through A C) C']
variables [AngleEq A 60]

-- The proof statement (proposition to prove)
theorem abc_sum_eq_three_bc' :
  AB + AC = 3 * B'C' :=
sorry

end abc_sum_eq_three_bc_l120_120011


namespace Sn_correct_Tn_correct_l120_120068

def a_n (n : ℕ) : ℕ := 3^(n-1)
def S_n (n : ℕ) := (3^n - 1) / 2
def f (m : ℕ) : ℕ := (m.digits 3).sum

noncomputable def T_n (n : ℕ) :=
  ∑ i in range n, (S_n i) * (f (S_n i))

theorem Sn_correct (n : ℕ) :
  S_n n = (3^n - 1) / 2 :=
  sorry

theorem Tn_correct (n : ℕ) :
  T_n n = (3 / 8) + ((2*n - 1) * 3^(n+1) / 8) - (n * (n + 1) / 4) :=
  sorry

end Sn_correct_Tn_correct_l120_120068


namespace volume_limit_l120_120932

open Real

noncomputable def volume_of_solid (n : ℕ) : ℝ :=
  let VC := π * ((n + 1)^2 / 2 - n^2 / 2)
  let vl := π * (1 / (2 * sqrt n) * ∫ x in n..n+1, (1 / (2 * sqrt n) * x + sqrt n / 2)^2)
  VC - vl

theorem volume_limit (V : ℕ → ℝ) (a b : ℝ) :
  (∀ n, V n = volume_of_solid n) →
  (a = 0 ∧ b = π / 6) →
  filter.tendsto (fun n => n^a * V n) filter.at_top (nhds b) :=
begin
  intros hV h_ab,
  rcases h_ab with ⟨ha, hb⟩, 
  simp only [ha, pow_zero, one_mul], 
  sorry,
end

end volume_limit_l120_120932


namespace difference_sum_5_to_7_and_sum_8_to_10_l120_120547

variable (scores : ℕ → ℝ)

-- Top 10 average scores
def avg_top_10 := (scores 0 + scores 1 + scores 2 + scores 3 + scores 4 + scores 5 + scores 6 + scores 7 + scores 8 + scores 9) / 10

-- Top 7 average scores
def avg_top_7 := (scores 0 + scores 1 + scores 2 + scores 3 + scores 4 + scores 5 + scores 6) / 7

-- Top 4 average scores
def avg_top_4 := (scores 0 + scores 1 + scores 2 + scores 3) / 4

-- Condition 1: The average score of the top 7 participants is 3 points lower than the average score of the top 4 participants.
axiom condition1 : avg_top_7 = avg_top_4 - 3

-- Condition 2: The average score of the top 10 participants is 4 points lower than the average score of the top 7 participants.
axiom condition2 : avg_top_10 = avg_top_7 - 4

-- Sum of the scores from 5th to 7th participants
def sum_5_to_7 := scores 4 + scores 5 + scores 6

-- Sum of the scores from 8th to 10th participants
def sum_8_to_10 := scores 7 + scores 8 + scores 9

-- Statement to prove
theorem difference_sum_5_to_7_and_sum_8_to_10 :
  sum_5_to_7 - sum_8_to_10 = 28 :=
sorry

end difference_sum_5_to_7_and_sum_8_to_10_l120_120547


namespace alpha_beta_value_l120_120067

theorem alpha_beta_value :
  ∃ α β : ℝ, (α^2 - 2 * α - 4 = 0) ∧ (β^2 - 2 * β - 4 = 0) ∧ (α + β = 2) ∧ (α^3 + 8 * β + 6 = 30) :=
by
  sorry

end alpha_beta_value_l120_120067


namespace find_number_l120_120244

theorem find_number (X : ℝ) (h : X / 3 = 1.694915254237288 * 236) : X = 1200 :=
by
  calc 
    X = 3 * (1.694915254237288 * 236) : sorry
    ... = 1200 : sorry

end find_number_l120_120244


namespace f_4_1981_eq_l120_120291

noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| (x+1), 0 => f x 1
| (x+1), (y+1) => f x (f (x+1) y)

theorem f_4_1981_eq : f 4 1981 = 2^1984 - 3 := 
by
  sorry

end f_4_1981_eq_l120_120291


namespace moscow_inequality_l120_120688

theorem moscow_inequality (a : ℕ → ℕ) (n : ℕ) (h_distinct : ∀ i j, i < j → a i < a j) :
  ∑ k in finset.range n, (a k)^7 + (a k)^5 ≥ 2 * (∑ k in finset.range n, (a k)^3)^2 := 
sorry

end moscow_inequality_l120_120688


namespace exponential_grows_faster_than_polynomial_l120_120082

theorem exponential_grows_faster_than_polynomial (x : ℝ) (h : 0 < x) : 2^x > x^2 := sorry

end exponential_grows_faster_than_polynomial_l120_120082


namespace proof_S_squared_l120_120614

variables {a b c p S r r_a r_b r_c : ℝ}

-- Conditions
axiom cond1 : r * p = r_a * (p - a)
axiom cond2 : r * r_a = (p - b) * (p - c)
axiom cond3 : r_b * r_c = p * (p - a)
axiom heron : S^2 = p * (p - a) * (p - b) * (p - c)

-- Proof statement
theorem proof_S_squared : S^2 = r * r_a * r_b * r_c :=
by sorry

end proof_S_squared_l120_120614


namespace roots_of_equation_l120_120122

theorem roots_of_equation : 
  (∃ s ∈ ([- real.sqrt 14, real.sqrt 14]).to_set, 
  (∀ x ∈ s, (real.sqrt (14 - x^2)) * (real.sin x - real.cos (2 * x)) = 0) ∧ 
  ( set.analyse (eq (set.card s) 6))) :=
sorry

end roots_of_equation_l120_120122


namespace construct_perpendicular_from_point_outside_circle_l120_120480

theorem construct_perpendicular_from_point_outside_circle
  (circle : Type)
  (center : Point)
  (A B P E : Point)
  (AB line : Line)
  (h1 : IsCircle circle)
  (h2 : Center circle = center)
  (h3 : LineThrough center)
  (h4 : Intersects line circle at A B)
  (h5 : PointOutsideCircle P circle)
  (h6 : IntersectsAt E (Extension (PointLine P A) (PointLine P C)) (circle intersect (PointLine P D)))
  (h7 : Thales' theorem circle A B P E) :
  Perpendicular (PointLine P E) AB :=
sorry

end construct_perpendicular_from_point_outside_circle_l120_120480


namespace largest_C_exists_l120_120658

noncomputable def a_seq (n : ℕ) : ℝ := sorry

noncomputable def b_seq (n : ℕ) : ℝ :=
  ∑ k in finset.range n, (1 - (a_seq (k - 1) / a_seq k)) * (1 / real.sqrt (a_seq k))

theorem largest_C_exists (x : ℝ) (h : 0 ≤ x ∧ x < 2) :
  ∃ (a_seq : ℕ → ℝ), 
    (∀ n, 1 = a_seq 0 ∧ (∀ m, m ≤ n → a_seq m ≤ a_seq (m + 1))) ∧ 
    (∃ (N : ℕ), ∀ n ≥ N, b_seq n > x) :=
sorry

end largest_C_exists_l120_120658


namespace problem_l120_120667

-- Definitions for the given problem
def f (x : ℝ) : ℝ := x^2

-- The sequence a_k defined recursively
def a : ℕ → ℝ
| 0     := 16  -- a_1 (converted to 0-indexed for Lean)
| (n+1) := (3 / 2) * a n

-- The proof problem statement: to show a_1 + a_3 + a_5 = 133
theorem problem (a : ℕ → ℝ) (h₀ : a 0 = 16) (h : ∀ k, a (k + 1) = (3 / 2) * a k) :
  a 0 + a 2 + a 4 = 133 := by
  sorry

end problem_l120_120667


namespace points_on_same_side_after_25_seconds_l120_120249

def movement_time (side_length : ℕ) (perimeter : ℕ)
  (speed_A speed_B : ℕ) (start_mid_B : ℕ) : ℕ :=
  25

theorem points_on_same_side_after_25_seconds (side_length : ℕ) (perimeter : ℕ)
  (speed_A speed_B : ℕ) (start_mid_B : ℕ) :
  side_length = 100 ∧ perimeter = 400 ∧ speed_A = 5 ∧ speed_B = 10 ∧ start_mid_B = 50 →
  movement_time side_length perimeter speed_A speed_B start_mid_B = 25 :=
by
  intros h
  sorry

end points_on_same_side_after_25_seconds_l120_120249


namespace triangle_is_right_l120_120148

theorem triangle_is_right (A B C a b c : ℝ) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
    (h₄ : A + B + C = π) (h_eq : a * (Real.cos C) + c * (Real.cos A) = b * (Real.sin B)) : B = π / 2 :=
by
  sorry

end triangle_is_right_l120_120148


namespace farmer_planted_radishes_l120_120370

/-- Define variables and given conditions -/
variables (beans seedlings_per_row_beans pumpkin seeds_per_row_pumpkin radishes_per_row rows_per_bed beds : ℕ)

-- Given conditions
variables (H1 : beans = 64)
variables (H2 : seedlings_per_row_beans = 8)
variables (H3 : pumpkin = 84)
variables (H4 : seeds_per_row_pumpkin = 7)
variables (H5 : radishes_per_row = 6)
variables (H6 : rows_per_bed = 2)
variables (H7 : beds = 14)

/-- The theorem proving that the number of radishes planted is 48 -/
theorem farmer_planted_radishes : 
  (beans / seedlings_per_row_beans) + (pumpkin / seeds_per_row_pumpkin) <= (beds * rows_per_bed) →
  radishes_per_row * ((beds * rows_per_bed) - (beans / seedlings_per_row_beans) - (pumpkin / seeds_per_row_pumpkin)) = 48 :=
by
  intros H
  sorry

end farmer_planted_radishes_l120_120370


namespace sum_of_non_repeating_elements_is_constant_l120_120310

theorem sum_of_non_repeating_elements_is_constant :
  let grid (i j : ℕ) := 10 * (i - 1) + j in
  (∀ (s : Finset (Fin 10 × Fin 10)), s.card = 10 →
    (∀ (i1 i2 j1 j2 : Fin 10), (i1, j1) ∈ s → (i2, j2) ∈ s → i1 ≠ i2 → j1 ≠ j2) →
      s.sum (λ (ij : Fin 10 × Fin 10), grid ij.1.1.succ ij.2.1.succ) = 505) :=
by
  let grid := λ (i j : ℕ), 10 * (i - 1) + j
  intro s
  sorry

end sum_of_non_repeating_elements_is_constant_l120_120310


namespace janet_wait_time_l120_120921

-- define the initial speeds and their changes
def initial_speed_janet : ℕ := 30
def halfway_decrease_janet : ℝ := 0.15
def initial_speed_sister : ℕ := 12
def hourly_increase_sister : ℝ := 0.20
def lake_width : ℕ := 60

-- define the computed speeds
def halfway_speed_janet := initial_speed_janet - (initial_speed_janet * halfway_decrease_janet).to_nat
def speed_after_1hr_sister := initial_speed_sister + (initial_speed_sister * hourly_increase_sister).to_nat

-- define the time calculations
def janet_first_half_time : ℝ := 30 / initial_speed_janet
def janet_second_half_time : ℝ := 30 / halfway_speed_janet
def janet_total_time : ℝ := janet_first_half_time + janet_second_half_time

def sister_first_hour_distance : ℕ := initial_speed_sister
def sister_remaining_distance : ℕ := lake_width - sister_first_hour_distance
def sister_remaining_time : ℝ := sister_remaining_distance / speed_after_1hr_sister
def sister_total_time : ℝ := 1 + sister_remaining_time

def waiting_time_janet : ℝ := sister_total_time - janet_total_time

theorem janet_wait_time :
  waiting_time_janet = 2.156862745 :=
by
  sorry

end janet_wait_time_l120_120921


namespace least_pos_int_x_l120_120329

theorem least_pos_int_x (x : ℕ) (h1 : ∃ k : ℤ, (3 * x + 43) = 53 * k) 
  : x = 21 :=
sorry

end least_pos_int_x_l120_120329


namespace max_ab_value_l120_120058

namespace MathProof

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + b^2 - 6 * a = 0) : 
  ab ≤ 27 * sqrt 3 / 4 :=
sorry

end MathProof

end max_ab_value_l120_120058


namespace min_max_product_of_squares_l120_120959

theorem min_max_product_of_squares (a b c d : ℝ)
    (h1 : a^2 + b^2 + 2a - 4b + 4 = 0)
    (h2 : c^2 + d^2 - 4c + 4d + 4 = 0) :
    let min_val := (real.sqrt ((-1 - 2)^2 + (2 + 2)^2) - 1 - 2)^2,
        max_val := (real.sqrt ((-1 - 2)^2 + (2 + 2)^2) + 1 + 2)^2 in
    min_val * max_val = 16 :=
by
  sorry  -- Proof goes here

end min_max_product_of_squares_l120_120959


namespace proof_l120_120065

-- Definitions and conditions
def B : ℝ × ℝ := (-1, 0)

def circle (A : ℝ × ℝ) : Prop :=
  let (x, y) := A in (x - 7)^2 + y^2 = 16

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  let (x₁, y₁) := A in
  let (x₂, y₂) := B in
  ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

def trajectory_eq (M : ℝ × ℝ) : Prop :=
  let (x, y) := M in (x - 3)^2 + y^2 = 4

def line_through_C (C : ℝ × ℝ) (L : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, (∀ x : ℝ, L x = m * x + b) ∧ (let (x, y) := C in L x = y) ∧ (let (x, y) := C in abs m = 1)

def point_C (a : ℝ) : ℝ × ℝ := (2, a)

def tangent_to_curve (L : ℝ → ℝ) (curve_eq : ℝ × ℝ → Prop) : Prop :=
  ∃ x : ℝ, let y := L x in curve_eq (x, y)

-- Problem statement: Proving the required properties
theorem proof :
  (∀ A, circle A → ∃ M, M = midpoint A B ∧ trajectory_eq M) ∧
  (∀ a, a > 0 → (tangent_to_curve (line_through_C (point_C a)) trajectory_eq) →
    a = (4 * real.sqrt 5) / 5 ∨ a = 1 + 2 * real.sqrt 2) ∧
  ∀ a, (a = (4 * real.sqrt 5) / 5 ∨ a = 1 + 2 * real.sqrt 2) →
    (∃ L : ℝ → ℝ, line_through_C (point_C a) L ∧ tangent_to_curve L trajectory_eq) :=
by sorry

end proof_l120_120065


namespace half_angle_quadrant_l120_120063

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π + π < α ∧ α < 2 * k * π + (3 * π / 2)) : 
  (∃ j : ℤ, j * π + (π / 2) < (α / 2) ∧ (α / 2) < j * π + (3 * π / 4)) :=
  by sorry

end half_angle_quadrant_l120_120063


namespace sum_of_digits_decrease_by_10_percent_l120_120735

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum -- Assuming this method computes the sum of the digits

theorem sum_of_digits_decrease_by_10_percent :
  ∃ (n m : ℕ), m = 11 * n / 10 ∧ sum_of_digits m = 9 * sum_of_digits n / 10 :=
by
  sorry

end sum_of_digits_decrease_by_10_percent_l120_120735


namespace scientific_notation_141260_million_l120_120232

theorem scientific_notation_141260_million :
  ∃ (a : ℝ) (n : ℤ), 141260 * 10^6 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4126 ∧ n = 5 :=
by
  sorry

end scientific_notation_141260_million_l120_120232


namespace geometric_sequence_formulas_l120_120892

theorem geometric_sequence_formulas
    {a_2 a_3 : ℚ} {S : ℕ → ℚ}
    (h1 : a_3 - a_2 * (3 / 2)⁻¹ = 16 / 27)
    (h2 : a_2 = -2 / 9)
    (hS : ∀ n, S n = n ^ 2) :
    (∀ n, a_n = -2 / 3 * (1 / 3) ^ (n - 1)) ∧ (∀ n, b_n = 2n - 1) ∧ 
    (∀ n, T_n = -2 + (2n + 2) * (1 / 3) ^ n) :=
begin
  sorry
end

end geometric_sequence_formulas_l120_120892


namespace intersection_M_N_l120_120861

def M : Set ℝ := { y | ∃ x : ℝ, y = x^2 - 1 }

def N : Set ℝ := { y | ∃ x : ℝ, y = sqrt (4 - x^2) ∧ -2 ≤ x ∧ x ≤ 2 }

theorem intersection_M_N : (M ∩ N) = { y | -1 ≤ y ∧ y ≤ 2 } :=
by
  sorry

end intersection_M_N_l120_120861


namespace line_up_ways_l120_120552

theorem line_up_ways (people : Finset ℕ) (youngest : ℕ) (h1 : people.card = 5) (h2 : youngest ∈ people) :
  (∃ (cnt : ℕ), cnt = 72 ∧
    ∃ (arrangements : Finset (Fin 5 → ℕ)),
      arrangements.card = cnt ∧
      ∀ a ∈ arrangements, a ≠ (λ x, if x = 0 ∨ x = 4 then youngest else a x) ) := 
  sorry

end line_up_ways_l120_120552


namespace polynomial_fibonacci_l120_120938

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

-- Define the main theorem
theorem polynomial_fibonacci {P : ℕ → ℕ} 
  (h1 : ∀ k, 992 ≤ k ∧ k ≤ 1982 → P k = fibonacci k)
  (h2 : degree P = 990) 
  : P 1983 = fibonacci 1983 - 1 := 
sorry

end polynomial_fibonacci_l120_120938


namespace triangle_problem_l120_120047

-- Definitions for the given conditions
def point := ℝ × ℝ

def A : point := (4, 2)
def B : point := (0, 5)

-- Definition of the line equations and properties
def line_eq (m b : ℝ) (p : point) : Prop := (p.1 - p.2 * m + b = 0)

-- The specific line conditions:
def line_equal_intercept1 : point → Prop := line_eq 1 0
def line_equal_intercept2 : point → Prop := λ p, (p.1 + p.2 - 6 = 0)

-- The triangle conditions
def on_line (m b : ℝ) (p : point) : Prop := (p.2 = m * p.1 + b)
def C_on_line : point → Prop := on_line 3 0

-- Area condition of the triangle ABC
def area_triangle (A B C : point) : ℝ := 
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

-- Main theorem
theorem triangle_problem :
  (∀ C : point, C_on_line C → 
    (area_triangle A B C = 10 → (C = (8 / 3, 8)))) ∧
  (∀ C : point, line_equal_intercept1 A ∨ line_equal_intercept2 A) :=
  by
  sorry

end triangle_problem_l120_120047


namespace solve_equation_roots_count_l120_120119

theorem solve_equation_roots_count :
  ∀ (x : ℝ), abs x ≤ real.sqrt 14 → 
  (sqrt (14 - x ^ 2) * (sin x - cos (2 * x)) = 0) → 
  (set.count {x | abs x ≤ real.sqrt 14 ∧ (sqrt (14 - x ^ 2) * (sin x - cos (2 * x)) = 0)} = 6) :=
by sorry

end solve_equation_roots_count_l120_120119


namespace intersection_A_B_l120_120878

def set_A : Set ℝ := {x : ℝ | |x| = x}
def set_B : Set ℝ := {x : ℝ | x^2 + x ≥ 0}
def set_intersection : Set ℝ := {x : ℝ | 0 ≤ x}

theorem intersection_A_B :
  (set_A ∩ set_B) = set_intersection :=
by
  sorry

-- You can verify if the Lean code builds successfully using Lean 4 environment.

end intersection_A_B_l120_120878


namespace compute_expression_l120_120431

theorem compute_expression : 9 + 7 * (5 - Real.sqrt 16)^2 = 16 := by
  sorry

end compute_expression_l120_120431


namespace max_value_of_trig_function_l120_120275

theorem max_value_of_trig_function (x : ℝ) (hx : ∃ a b c : ℝ, a + b + c = π ∧ x = min a (min b c)) :
  (sin x + cos x + sin x * cos x) ≤ (sqrt 2 + 1 / 2) :=
sorry

end max_value_of_trig_function_l120_120275


namespace sequence_correct_l120_120093

/-- Define the sequence {a_n} satisfying the given recurrence relations -/
def sequence (a : ℕ → ℕ) : Prop := 
  a 1 = 1 ∧ (∀ n : ℕ, 4 * a n * a (n + 1) = (a n + a (n + 1) - 1) ^ 2) ∧ (∀ n : ℕ, n > 1 → a n > a (n-1))

/-- The goal is to prove the sequence is given by a_n = n^2 -/
theorem sequence_correct {a : ℕ → ℕ} (h : sequence a) : ∀ n, a n = n^2 :=
sorry

end sequence_correct_l120_120093


namespace part_I_part_II_l120_120499

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2

def m (a : ℝ) (x : ℝ) : ℝ := (f a x)' x

theorem part_I (a : ℝ) : (m a 1)' = 3 → a = 2 :=
by
-- (Mathematically equivalent statement: m(x) = f'(x) with additional constraint)
-- (Mathematically equivalent result: a = 2 when m'(1) = 3)
sorry

def g (a : ℝ) (x : ℝ) : ℝ := f a x - a * x^2 + a * x

theorem part_II (a : ℝ) : (∀ x : ℝ, 0 < x → (g a x)' x ≥ 0) → a ≥ 0 :=
by
-- (Mathematically equivalent statement: g(x) = (\ln x + ax))
-- (Mathematically equivalent result: a ≥ 0 when g(x) is monotonically increasing)
sorry

end part_I_part_II_l120_120499


namespace marbles_arrangement_l120_120754

def marbles := ["Glassy", "Ruby", "Diamond", "Comet", "Pearl"]

def not_adjacent (l : List String) : Prop :=
  ¬ (let indices := (l.indexOf "Diamond") :: (l.indexOf "Pearl") :: [] in
     match indices with
     | [d, p] => (d = p + 1) ∨ (p = d + 1)
     | _ => false ) 

theorem marbles_arrangement :
  (List.permutations marbles).countp not_adjacent = 72 :=
sorry

end marbles_arrangement_l120_120754


namespace probability_union_of_events_l120_120679

theorem probability_union_of_events :
  let event_A := {n | n ∈ [1, 3, 5]}  -- A is odd numbers
  let event_B := {n | n ∈ [1, 2, 3]}  -- B is numbers <= 3
  let fair_die := {1, 2, 3, 4, 5, 6}  -- fair die outcomes
  let P := λ e, (e.card:ℚ) / 6           -- probability of the event e
  P (event_A ∪ event_B) = 2 / 3 :=
by
  sorry

end probability_union_of_events_l120_120679


namespace slower_truck_speed_is_20_kmh_l120_120687

noncomputable def speed_of_slower_truck : ℝ :=
  let l : ℝ := 250  -- length of each truck in meters
  let v1 : ℝ := 30  -- speed of the faster truck in km/hr
  let v1_ms : ℝ := v1 / 3.6  -- convert speed of faster truck to m/s
  let t : ℝ := 35.997120230381576 -- time taken for the slower truck to pass the driver of the faster truck in seconds
  let d : ℝ := 2 * l -- total distance to be covered
  let v2_ms := (d / t) - v1_ms -- calculate the speed of the slower truck in m/s
  let v2 : ℝ := v2_ms * 3.6 -- convert the speed of the slower truck to km/hr
  v2

theorem slower_truck_speed_is_20_kmh (l v1 t : ℝ) (h_l : l = 250) (h_v1 : v1 = 30) (h_t : t = 35.997120230381576) :
  speed_of_slower_truck = 20 :=
by
  rw [speed_of_slower_truck]
  have h_v1_ms : v1 / 3.6 = 30 / 3.6 := by rw [h_v1]
  have v1_ms := 30 / 3.6
  have h_d : 2 * l = 2 * 250 := by rw [h_l]
  have d := 2 * 250
  have v2_ms : (d / t) - v1_ms = (2 * 250 / 35.997120230381576) - (30 / 3.6) := by rw [h_t, h_d, h_v1_ms]
  have v2 : v2_ms * 3.6 = ((2 * 250 / 35.997120230381576) - (30 / 3.6)) * 3.6
  rw [v2]
  norm_num
  sorry

end slower_truck_speed_is_20_kmh_l120_120687


namespace min_sum_xi_xj_l120_120213

noncomputable def sum_xi_xj_min (n : ℕ) : ℝ :=
  if even n then - (n / 2)
  else - ((n - 1) / 2)

theorem min_sum_xi_xj (n : ℕ) (x : Fin n → ℝ)
  (h : ∀ i, x i = 0 ∨ x i = 1 ∨ x i = -1 ) :
  (∑ i, ∑ j in finset.Ico i.succ n, x i * x j) = sum_xi_xj_min n :=
by
  sorry

end min_sum_xi_xj_l120_120213


namespace least_positive_integer_x_l120_120326

theorem least_positive_integer_x
  (n : ℕ)
  (h : (3 * n)^2 + 2 * 43 * (3 * n) + 43^2) % 53 = 0
  (h_pos : 0 < n)
  : n = 21 :=
sorry

end least_positive_integer_x_l120_120326


namespace find_f_1_over_5_l120_120839

open Real

variable (f : ℝ → ℝ)
variable (domain : Set ℝ := Set.Ioi 0)

def monotonic_on_domain : Prop :=
  ∀ x y ∈ domain, x < y → f x ≤ f y

def functional_condition : Prop :=
  ∀ x ∈ domain, f (f x - 1 / x) = 2

theorem find_f_1_over_5 (h_monotonic : monotonic_on_domain f)
                        (h_functional : functional_condition f) :
  f (1 / 5) = 6 :=
  sorry

end find_f_1_over_5_l120_120839


namespace pairs_count_l120_120100

theorem pairs_count : 
  (finset.card {p : ℕ × ℕ | let m := p.1, n := p.2 in 
    m ≠ n ∧ ∃ k : ℕ, odd k ∧ 2^k * (m+n) = 50688}) = 33760 := 
  by sorry

end pairs_count_l120_120100


namespace find_ratio_l120_120884

-- Define the conditions in the problem
variables {A B C AC AB : ℝ}
variables [triangle_ABC : Triangle ABC]

-- Define the conditions outlined in the problem
axiom condition1 : 2 * sin^2 (A / 2) = sqrt 3 * sin A
axiom condition2 : sin (B - C) = 2 * cos B * sin C

-- Define the theorem, asserting the value of AC / AB under the given conditions
theorem find_ratio :
  (AC / AB) = (1 + sqrt 13) / 2 :=
by 
  -- sorry to skip the proof steps
  sorry

end find_ratio_l120_120884


namespace amount_daria_needs_l120_120442

theorem amount_daria_needs (ticket_cost : ℕ) (total_tickets : ℕ) (current_money : ℕ) (needed_money : ℕ) 
  (h1 : ticket_cost = 90) 
  (h2 : total_tickets = 4) 
  (h3 : current_money = 189) 
  (h4 : needed_money = 360 - 189) 
  : needed_money = 171 := by
  -- proof omitted
  sorry

end amount_daria_needs_l120_120442


namespace parabola_intersects_once_compare_y_values_l120_120858

noncomputable def parabola (x : ℝ) (m : ℝ) : ℝ := -2 * x^2 + 4 * x + m

theorem parabola_intersects_once (m : ℝ) : 
  ∃ x, parabola x m = 0 ↔ m = -2 := 
by 
  sorry

theorem compare_y_values (x1 x2 m : ℝ) (h1 : x1 > x2) (h2 : x2 > 2) : 
  parabola x1 m < parabola x2 m :=
by 
  sorry

end parabola_intersects_once_compare_y_values_l120_120858


namespace range_of_m_l120_120514

variables (m : ℝ)

def a : ℝ × ℝ × ℝ := (2, -1, 2)
def b : ℝ × ℝ × ℝ := (-4, 2, m)

-- Condition: angle between a and b is obtuse, i.e., dot product is negative
def dot_product_obtuse (a b : ℝ × ℝ × ℝ) : Prop :=
  let dot_product := (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) in
  dot_product < 0

-- Prove this statement about m
theorem range_of_m (m : ℝ) :
  dot_product_obtuse a b → (m < 5 ∧ m ≠ -4) :=
sorry

end range_of_m_l120_120514


namespace solve_for_z_l120_120622

theorem solve_for_z : ∃ z, 2 * 2^z + sqrt (16 * 16^z) = 32 ↔ z = real.log (2) ( (-1 + real.sqrt 257) / 8 ) :=
begin
  sorry
end

end solve_for_z_l120_120622


namespace roots_count_eq_six_l120_120106

noncomputable def number_of_roots : ℝ := 
  let I := Icc (-real.sqrt 14) (real.sqrt 14)
  in
  let f := λ x, √(14 - x^2) * (real.sin x - real.cos (2 * x))
  in
  set.finite (set_of (λ x, f x = 0) ∩ I).to_finset.card

theorem roots_count_eq_six : number_of_roots = 6 := by
  sorry

end roots_count_eq_six_l120_120106


namespace reflection_line_l120_120683

theorem reflection_line {P Q R P' Q' R' : ℝ × ℝ} (hP : P = (3, 4)) (hQ : Q = (8, 9)) (hR : R = (-5, 7))
                       (hP' : P' = (3, -6)) (hQ' : Q' = (8, -11)) (hR' : R' = (-5, -9)) :
  ∃ M : ℝ → ℝ, ∀ x, M(x) = -1 :=
by
  sorry

end reflection_line_l120_120683


namespace locate_z_in_second_quadrant_l120_120656

def complex_num_z := complex.I * (complex.I + 2)
def is_in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

theorem locate_z_in_second_quadrant : is_in_second_quadrant complex_num_z :=
by
  sorry

end locate_z_in_second_quadrant_l120_120656


namespace min_distance_curves_l120_120041

theorem min_distance_curves (P Q : ℝ × ℝ) (h1 : P.2 = (1/3) * Real.exp P.1) (h2 : Q.2 = Real.log (3 * Q.1)) :
  ∃ d : ℝ, d = Real.sqrt 2 * (Real.log 3 - 1) ∧ d = |P.1 - Q.1| := sorry

end min_distance_curves_l120_120041


namespace least_xy_value_l120_120828

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
sorry

end least_xy_value_l120_120828


namespace min_x2_y2_eq_16_then_product_zero_l120_120883

theorem min_x2_y2_eq_16_then_product_zero
  (x y : ℝ)
  (h1 : ∃ x y : ℝ, (x^2 + y^2 = 16 ∧ ∀ a b : ℝ, a^2 + b^2 ≥ 16) ) :
  (x + 4) * (y - 4) = 0 := 
sorry

end min_x2_y2_eq_16_then_product_zero_l120_120883


namespace line_perpendicular_to_plane_l120_120481

noncomputable def direction_vector_l : ℝ × ℝ × ℝ := (1, -1, 1)
noncomputable def normal_vector_alpha : ℝ × ℝ × ℝ := (-1, 1, -1)

-- Define a function to check if two vectors are collinear
def are_collinear (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (v1.1, v1.2, v1.3) = (k * v2.1, k * v2.2, k * v2.3)

-- Main theorem which we state without proof
theorem line_perpendicular_to_plane :
  are_collinear direction_vector_l normal_vector_alpha →
  line_perpendicular_to_plane :=
begin
  sorry
end

end line_perpendicular_to_plane_l120_120481


namespace number_of_knights_l120_120706

-- We define the notion of knights and liars
inductive PersonType
| knight : PersonType
| liar : PersonType

-- Define the input parameters
def people_count : Nat := 80
def statement_range : Nat := 11
def min_liars_in_statement : Nat := 9

-- Assume we have a round table with people's types
def round_table : Fin people_count → PersonType := sorry

-- Define a predicate for the statement each person makes
def statement (i : Fin people_count) : Prop :=
  (Finset.card (Finset.filter (λ j, round_table (i + j + 1) = PersonType.liar) (Finset.range statement_range))) ≥ min_liars_in_statement

-- Define a predicate for the knights' correctness of statement
def knights_statement_correct (i : Fin people_count) :=
  round_table i = PersonType.knight → statement i

-- The main theorem to state the number of knights
theorem number_of_knights :
  (Finset.card (Finset.filter (λ i, round_table i = PersonType.knight) (Finset.univ : Finset (Fin people_count)))) = 20 :=
sorry

end number_of_knights_l120_120706


namespace solve_for_x_l120_120508

theorem solve_for_x (x : ℝ) (k : ℤ) (h: x ≠ (2 * k + 1) * Real.pi) :
  (sqrt 3) * tan (x / 2) = 1 → ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3 :=
by
  sorry

end solve_for_x_l120_120508


namespace ratio_a_c_l120_120653

theorem ratio_a_c {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : ∀ x : ℝ, x^2 + a * x + b = 0 → 2 * x^2 + 2 * a * x + 2 * b = 0)
  (h2 : ∀ x : ℝ, x^2 + b * x + c = 0 → x^2 + b * x + c = 0) :
  a / c = 1 / 8 :=
by
  sorry

end ratio_a_c_l120_120653


namespace find_r_plus_s_l120_120646

theorem find_r_plus_s :
  ∃ T : ℝ × ℝ, let r := T.1, s := T.2,
  ∀ P Q : ℝ × ℝ,
    P = (16, 0) → 
    Q = (0, 8) →
    s = 4 →
    r = 8 →
    r + s = 12 :=
begin
  sorry
end

end find_r_plus_s_l120_120646


namespace pump_out_time_l120_120362

-- Define the given conditions as constants
def length : ℝ := 30 -- feet
def width : ℝ := 40 -- feet
def depth : ℝ := 2 -- feet (from 24 inches)

def pump_rate : ℝ := 10 -- gallons per minute per pump
def number_of_pumps : ℕ := 4
def gallons_per_cubic_foot : ℝ := 7.5

-- Define the total volume of water in cubic feet
def volume : ℝ := length * width * depth

-- Convert the volume to gallons
def volume_in_gallons : ℝ := volume * gallons_per_cubic_foot

-- Compute the total pumping rate
def total_pumping_rate : ℝ := number_of_pumps * pump_rate

-- The expected time in minutes to pump out all the water
def time : ℝ := volume_in_gallons / total_pumping_rate

-- The theorem statement to be proven
theorem pump_out_time : time = 450 :=
  by
  -- Proof is skipped
  sorry

end pump_out_time_l120_120362


namespace find_XW_l120_120169

namespace TriangleXYZ

theorem find_XW 
  (XY : ℝ) (XZ : ℝ) (YW_WWZ_ratio : ℝ) (XW : ℝ)
  (XY_eq : XY = 13)
  (XZ_eq : XZ = 20)
  (YW_WWZ_ratio_eq : YW_WWZ_ratio = 3 / 7) :
  XW = real.sqrt (4681 / 40) :=
by
  -- We have the conditions stated above
  -- XY = 13, XZ = 20, and YW:WZ = 3:7
  sorry

end TriangleXYZ

end find_XW_l120_120169


namespace irrational_count_correct_l120_120640

noncomputable def evaluate_list := 
  let l := [
    (- real.sqrt 25),
    (real.pi / 2),
    (0 : ℝ),
    (0.566666666...), -- Express periodic decimal properly if necessary
    (real.cbrt 4),
    (-0.1010010001...) -- Describing the pattern in lean may need a sequence definition
  ] in
    count_irrational l = 3

theorem irrational_count_correct : evaluate_list = true := sorry

end irrational_count_correct_l120_120640


namespace school_can_buy_softballs_l120_120659

theorem school_can_buy_softballs :
  (let original_budget := 15 * 5 in
   let increase := 0.2 * original_budget in
   let new_budget := original_budget + increase in
   new_budget / 9 = 10) :=
begin
  -- Using let-expressions to define intermediate steps
  let original_budget := 15 * 5,
  let increase := 0.2 * original_budget,
  let new_budget := original_budget + increase,
  -- Verifying the final number of softballs that can be bought is 10
  have h : new_budget / 9 = 10,
  { rw [← mul_div_right_comm, show 9 * 10 = 90, by norm_num] },
  exact h,
end

end school_can_buy_softballs_l120_120659


namespace min_sum_of_factors_of_144_is_neg_145_l120_120940

theorem min_sum_of_factors_of_144_is_neg_145 
  (a b : ℤ) 
  (h : a * b = 144) : 
  a + b ≥ -145 := 
sorry

end min_sum_of_factors_of_144_is_neg_145_l120_120940


namespace time_to_pass_post_l120_120175

variable (length_of_train : ℝ) (speed_kmph : ℝ) (speed_mps : ℝ)

def initial_conditions : Prop :=
  length_of_train = 150 ∧ speed_kmph = 120 ∧ speed_mps = 120 * (1000 / 3600)

-- Prove that time to pass the telegraph post is approximately 4.5 seconds
theorem time_to_pass_post (h : initial_conditions) : 150 / 33.33 ≈ 4.5 :=
by sorry

end time_to_pass_post_l120_120175


namespace sequence_behavior_l120_120031

noncomputable def sequence (x : ℕ → ℝ) (x1 : ℝ) (hx1_pos : 0 < x1) (hx1_ne1 : x1 ≠ 1) : Prop :=
∀ (n : ℕ), x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)

theorem sequence_behavior (x : ℕ → ℝ) (x1 : ℝ) (hx1_pos : 0 < x1) (hx1_ne1 : x1 ≠ 1)
(hseq : ∀ (n : ℕ), x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)) :
(∀ n, x n < x (n + 1)) ∨ (∀ n, x n > x (n + 1)) :=
sorry

end sequence_behavior_l120_120031


namespace polyhedron_faces_same_edges_l120_120738

theorem polyhedron_faces_same_edges (n : ℕ) :
  (∃ S : finset ℕ, S.card = n + 1 ∧ ∀ f g ∈ S, f = g) :=
begin
  -- conditions
  let F := 7 * n,

  -- conclusion
  sorry
end

end polyhedron_faces_same_edges_l120_120738


namespace sum_integer_parts_solutions_l120_120303

noncomputable theory
open Real

theorem sum_integer_parts_solutions :
  let poly := (λ x : ℝ, x^4 - x^3 - 2 * sqrt 5 * x^2 - 7 * x^2 + sqrt 5 * x + 3 * x + 7 * sqrt 5 + 17)
  let solutions := {x : ℝ | poly x = 0 ∧ x > 0}
  let integer_parts := {⌊x⌋ | x ∈ solutions}
  ∑ x in integer_parts, x = 5 :=
sorry

end sum_integer_parts_solutions_l120_120303


namespace more_candidates_selected_l120_120544

theorem more_candidates_selected (n : ℕ) (pA pB : ℝ) 
  (hA : pA = 0.06) (hB : pB = 0.07) (hN : n = 8200) :
  (pB * n - pA * n) = 82 :=
by
  sorry

end more_candidates_selected_l120_120544


namespace eight_digit_number_divisibility_l120_120937

theorem eight_digit_number_divisibility (a b c d : ℕ) (Z : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) 
(h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : d ≤ 9) (hZ : Z = 1001 * (1000 * a + 100 * b + 10 * c + d)) : 
  10001 ∣ Z := 
  by sorry

end eight_digit_number_divisibility_l120_120937


namespace frustum_lateral_surface_area_correct_l120_120281

-- Definitions of conditions from the problem
def upper_base_area : ℝ := 25 -- cm^2
def lower_base_diameter : ℝ := 20 -- cm
def slant_height : ℝ := 10 -- cm

-- Required to calculate the lateral surface area of the frustum
noncomputable def lateral_surface_area : ℝ :=
  let r := Real.sqrt (upper_base_area / pi) in
  let R := lower_base_diameter / 2 in
  pi * slant_height * (R + r)

-- Theorem statement
theorem frustum_lateral_surface_area_correct :
  lateral_surface_area = 150 * pi :=
by
  sorry

end frustum_lateral_surface_area_correct_l120_120281


namespace total_population_approx_l120_120888

noncomputable def total_population (men_percentage : ℝ) (men_count : ℕ) : ℝ :=
  men_count / men_percentage

theorem total_population_approx (h : 0.45 * P = 58000) : P = 128889 :=
by
-- let more assertions if needed
-- sorry

end total_population_approx_l120_120888


namespace divides_expression_l120_120931

theorem divides_expression (n : ℕ) (h1 : n ≥ 3) 
  (h2 : Prime (4 * n + 1)) : (4 * n + 1) ∣ (n^(2 * n) - 1) :=
by
  sorry

end divides_expression_l120_120931


namespace center_element_l120_120157

-- Define the conditions
variables (n : ℕ) (a : ℕ → ℕ → ℝ)
hypothesis (h_n : n = 2015)
hypothesis (h_pos : ∀ i j, 0 < a i j)
hypothesis (h_row : ∀ i, ∏ j in finset.range n, a i j = 1)
hypothesis (h_col : ∀ j, ∏ i in finset.range n, a i j = 1)
hypothesis (h_square : ∀ i j, ∀ i2 j2, (i2 - i = 1007 ∧ j2 - j = 1007) → ∏ i' in finset.range 1008, ∏ j' in finset.range 1008, a (i + i') (j + j') = 2)

-- The goal to prove
theorem center_element (h : 0 < 1007 < 2015) : a 1007 1007 = 16 :=
sorry

end center_element_l120_120157


namespace constant_term_expansion_eq_1120_l120_120136

noncomputable def a : ℝ := ∫ x in 0..π, Real.sin x

theorem constant_term_expansion_eq_1120 :
  let expansion_const_term : ℝ := (x - (a / x))^8
  in expansion_const_term
  = 1120 :=
by
  have ha : a = 2 := by
    calc
      a = ∫ x in 0..π, Real.sin x : by rfl
      ... = -Real.cos π - (-Real.cos 0) : by sorry
      ... = 2 : by sorry
  sorry
  -- Further steps would involve using the binomial theorem to expand (x - 2/x)^8

end constant_term_expansion_eq_1120_l120_120136


namespace roots_count_eq_six_l120_120108

noncomputable def number_of_roots : ℝ := 
  let I := Icc (-real.sqrt 14) (real.sqrt 14)
  in
  let f := λ x, √(14 - x^2) * (real.sin x - real.cos (2 * x))
  in
  set.finite (set_of (λ x, f x = 0) ∩ I).to_finset.card

theorem roots_count_eq_six : number_of_roots = 6 := by
  sorry

end roots_count_eq_six_l120_120108


namespace count_visible_factor_numbers_200_250_l120_120380

def is_visible_factor_number (n : ℕ) : Prop :=
  (∀ d ∈ (repr n).toList.map (λ c => c.toNat - '0'.toNat), d ≠ 0 → n % d = 0)

def count_visible_factor_numbers (a b : ℕ) : ℕ :=
  (a.b.to_list).count is_visible_factor_number

theorem count_visible_factor_numbers_200_250 : count_visible_factor_numbers 200 250 = 21 :=
by sorry

end count_visible_factor_numbers_200_250_l120_120380


namespace rectangle_ratio_l120_120808

theorem rectangle_ratio (s x y : ℝ) 
  (h1 : 4 * (x * y) + s^2 = 9 * s^2)
  (h2 : x + s = 3 * s)
  (h3 : s + 2 * y = 3 * s) :
  x / y = 2 :=
by
  sorry

end rectangle_ratio_l120_120808


namespace necessary_and_sufficient_condition_for_unit_vectors_eq_l120_120055

variables {R : Type*} [LinearOrderedField R]
variables (a b : R^3) (ha : a ≠ 0) (hb : b ≠ 0)

open_locale classical

theorem necessary_and_sufficient_condition_for_unit_vectors_eq :
  ((a / ∥a∥) = (b / ∥b∥)) ↔ (∃ k : R, k > 0 ∧ a = k • b) :=
begin
  sorry
end

end necessary_and_sufficient_condition_for_unit_vectors_eq_l120_120055


namespace monotonicity_of_f_range_of_a_l120_120019

noncomputable def f (a x : ℝ) : ℝ := a * x - (Real.sin x / Real.cos x ^ 3)

noncomputable def g (a x : ℝ) : ℝ := f a x - Real.sin (2 * x)

theorem monotonicity_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) : 
  (8 = 8) -> 
  ((0 < x ∧ x < Real.pi / 4) -> f 8 x > 0) ∧ 
  ((Real.pi / 4 < x ∧ x < Real.pi / 2) -> f 8 x < 0) :=
by
  sorry

theorem range_of_a (a : ℝ) (x : ℝ) (h3 : 0 < x) (h4 : x < Real.pi / 2) :
  (f a x < Real.sin (2 * x)) -> 
  a ∈ (-∞, 3] :=
by
  sorry

end monotonicity_of_f_range_of_a_l120_120019


namespace solve_equation_roots_count_l120_120116

theorem solve_equation_roots_count :
  ∀ (x : ℝ), abs x ≤ real.sqrt 14 → 
  (sqrt (14 - x ^ 2) * (sin x - cos (2 * x)) = 0) → 
  (set.count {x | abs x ≤ real.sqrt 14 ∧ (sqrt (14 - x ^ 2) * (sin x - cos (2 * x)) = 0)} = 6) :=
by sorry

end solve_equation_roots_count_l120_120116


namespace alberto_biked_more_l120_120636

theorem alberto_biked_more (alberto_miles : ℕ) (bjorn_miles : ℕ) (h1 : alberto_miles = 60) (h2 : bjorn_miles = 45) : alberto_miles - bjorn_miles = 15 := 
by
  rw [h1, h2]
  exact rfl

end alberto_biked_more_l120_120636


namespace norm_prob_l120_120843

noncomputable def X : Type := sorry

axiom normal_distribution (μ : ℝ) (σ2 : ℝ) : Type :=
sorry

axiom P (X : Type) (a b : ℝ) : ℝ :=
sorry

theorem norm_prob : ∃ σ2 : ℝ, (normal_distribution 2 σ2) ∧ 
  (P X 0 2 = 0.3) → (P X 4 (⊤) = 0.2) :=
sorry

end norm_prob_l120_120843


namespace T_formula_l120_120596

noncomputable def T : ℕ → ℕ
| 0 := 2
| 1 := 3
| 2 := 6
| (n + 3) := (n + 4) * T (n + 2) - 4 * (n + 3) * T (n + 1) + (4 * (n + 3) - 8) * T n

theorem T_formula (n : ℕ) : T n = nat.factorial n + 2^n := by
  sorry

end T_formula_l120_120596


namespace sum_ratio_of_arithmetic_sequence_l120_120599

open Real

-- Conditions: An arithmetic sequence, sum formula for arithmetic sequence, given expression for a₅ and a₃
theorem sum_ratio_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (∀ n : ℝ, S n = n / 2 * (2 * a 1 + (n - 1) * d)) →
  a 5 = 5 * a 3 →
  ∫ x in 0..2, (2 * x + 1 / 2) = 5 →
  S 9 / S 5 = 9 :=
by
  sorry

end sum_ratio_of_arithmetic_sequence_l120_120599


namespace count_integers_between_3100_and_3600_with_increasing_and_distinct_digits_l120_120518

def is_increasing_digits (n : ℕ) : Prop :=
  let digits := n.digits in
  digits.foldr (λ d (acc : ℕ × bool), (d, acc.snd && d > acc.fst)) (10, true) = (digits.head, true)

def distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits in
  digits.nodup

theorem count_integers_between_3100_and_3600_with_increasing_and_distinct_digits :
  ∃ count : ℕ, count = 61 ∧ ∀ n, 3100 ≤ n ∧ n ≤ 3600 ∧ is_increasing_digits n ∧ distinct_digits n → n ∈ [3100, 3600] :=
by
  -- Skip the proof
  sorry

end count_integers_between_3100_and_3600_with_increasing_and_distinct_digits_l120_120518


namespace at_least_one_rectangle_sum_of_areas_rectangles_l120_120211

-- Conditions: Statement of the given problem
def n (n : ℕ) : Prop := n ≥ 1
def regular_4n_gon_decomposition (n : ℕ) :=
  exists (decomp : list (set (ℝ × ℝ))), (∀ p ∈ decomp, parallelogram p) 
    ∧ regular_polygon (4 * n) (⋃ p ∈ decomp, p)

-- Lean statement for part (a)
theorem at_least_one_rectangle {n : ℕ} (hg : n n) 
  (h_decomp : regular_4n_gon_decomposition n) : 
  ∃ (p : set (ℝ × ℝ)), p ∈ (classical.some h_decomp) ∧ is_rectangle p :=
sorry

-- Lean statement for part (b)
theorem sum_of_areas_rectangles {n : ℕ} (hg : n n)
  (h_decomp : regular_4n_gon_decomposition n) : 
  ∑ (p : set (ℝ × ℝ)) in (classical.some h_decomp), (area p) = n :=
sorry

end at_least_one_rectangle_sum_of_areas_rectangles_l120_120211


namespace find_radius_of_inscribed_circle_find_angle_BDC_l120_120986

-- Define the geometrical and numerical conditions
variables (A B C D O : Point)
variables (O_1 O_2 K T : Point)
variables (r : ℝ)
variable (α : ℝ)

-- Conditions description
axiom cond1 : inscribed (Q A B C D O)
axiom cond2 : RadiusSame (C1 O_1) (C2 O_2)
axiom cond3 : tangent_at_point (C1 O_1) (line B K C)
axiom cond4 : tangent_at_point (C2 O_2) (line D T A)
axiom cond5 : BK = 3 * sqrt 3
axiom cond6 : DT = sqrt 3
axiom cond7 : circumcenter (triangle B O C) = O_1

-- Problem statement part (a)
theorem find_radius_of_inscribed_circle :
  r = 3 := by
  sorry

-- Additional properties for second part
axiom cond8 : ∠ABC + ∠ADC = 180
axiom cond9 : ∠KBO_1 = α
axiom cond10 : ∠TDO_2 = 90 - α

-- Problem statement part (b)
theorem find_angle_BDC :
  ∠BDC = 30 := by
  sorry

end find_radius_of_inscribed_circle_find_angle_BDC_l120_120986


namespace coalsBurnedEveryTwentyMinutes_l120_120364

-- Definitions based on the conditions
def totalGrillingTime : Int := 240
def coalsPerBag : Int := 60
def numberOfBags : Int := 3
def grillingInterval : Int := 20

-- Derived definitions based on conditions
def totalCoals : Int := numberOfBags * coalsPerBag
def numberOfIntervals : Int := totalGrillingTime / grillingInterval

-- The Lean theorem we want to prove
theorem coalsBurnedEveryTwentyMinutes : (totalCoals / numberOfIntervals) = 15 := by
  sorry

end coalsBurnedEveryTwentyMinutes_l120_120364


namespace copper_zinc_ratio_l120_120340

theorem copper_zinc_ratio (total_weight : ℝ) (zinc_weight : ℝ) 
  (h_total_weight : total_weight = 70) (h_zinc_weight : zinc_weight = 31.5) : 
  (70 - 31.5) / 31.5 = 77 / 63 :=
by
  have h_copper_weight : total_weight - zinc_weight = 38.5 :=
    by rw [h_total_weight, h_zinc_weight]; norm_num
  sorry

end copper_zinc_ratio_l120_120340


namespace angle_C_in_triangle_l120_120170

theorem angle_C_in_triangle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A + B = 115) : C = 65 := 
by 
  sorry

end angle_C_in_triangle_l120_120170


namespace find_n_sequence_sum_l120_120083

theorem find_n_sequence_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₀ : ∀ n, a n = (2^n - 1) / 2^n)
  (h₁ : S 6 = 321 / 64) :
  ∃ n, S n = 321 / 64 ∧ n = 6 := 
by 
  sorry

end find_n_sequence_sum_l120_120083


namespace prob_A1_selected_prob_neither_B1_C1_selected_l120_120307

-- Conditions
def volunteers : List (String × String × String) :=
  [("A1", "B1", "C1"), ("A1", "B1", "C2"), ("A1", "B2", "C1"), ("A1", "B2", "C2"),
   ("A1", "B3", "C1"), ("A1", "B3", "C2"), ("A2", "B1", "C1"), ("A2", "B1", "C2"),
   ("A2", "B2", "C1"), ("A2", "B2", "C2"), ("A2", "B3", "C1"), ("A2", "B3", "C2"),
   ("A3", "B1", "C1"), ("A3", "B1", "C2"), ("A3", "B2", "C1"), ("A3", "B2", "C2"),
   ("A3", "B3", "C1"), ("A3", "B3", "C2")]

def selected_event_a1 : List (String × String × String) :=
  [("A1", "B1", "C1"), ("A1", "B1", "C2"), ("A1", "B2", "C1"), ("A1", "B2", "C2"),
   ("A1", "B3", "C1"), ("A1", "B3", "C2")]

def selected_event_neither_b1_c1 : List (String × String × String) :=
  [("A1", "B2", "C2"), ("A1", "B3", "C2"), ("A2", "B2", "C2"), ("A2", "B3", "C2"),
   ("A3", "B2", "C2"), ("A3", "B3", "C2"), ("A1", "B2", "C1"), ("A1", "B3", "C1"),
   ("A2", "B2", "C1"), ("A2", "B3", "C1"), ("A3", "B2", "C1"), ("A3", "B3", "C1"),
   ("A1", "B2", "C2"), ("A1", "B3", "C2"), ("A2", "B2", "C2")]

-- Lean 4 statements for proofs
theorem prob_A1_selected : 
  (selected_event_a1.length : ℚ) / (volunteers.length : ℚ) = 1 / 3 :=
  by
    sorry

theorem prob_neither_B1_C1_selected :
  (selected_event_neither_b1_c1.length : ℚ) / (volunteers.length : ℚ) = 5 / 6 :=
  by
    sorry

end prob_A1_selected_prob_neither_B1_C1_selected_l120_120307


namespace area_ratio_of_centroids_of_convex_quad_l120_120035

noncomputable def centroid (A B P : ℝ^2) : ℝ^2 :=
  (A + B + P) / 3

theorem area_ratio_of_centroids_of_convex_quad
  {A B C D P G1 G2 G3 G4 : ℝ^2}
  (h_convex : convex_quad A B C D)
  (hP_interior : is_interior_point P A B C D)
  (hG1 : G1 = centroid A B P)
  (hG2 : G2 = centroid B C P)
  (hG3 : G3 = centroid C D P)
  (hG4 : G4 = centroid D A P) :
  (area_of_quad G1 G2 G3 G4) / (area_of_quad A B C D) = 1 / 9 :=
sorry

end area_ratio_of_centroids_of_convex_quad_l120_120035


namespace third_square_perimeter_l120_120701

-- Define the conditions
def perimeter_square1 : ℝ := 40
def perimeter_square2 : ℝ := 32

-- Function to calculate the perimeter of the third square
noncomputable def perimeter_square3 : ℝ := 
  let side_square1 := perimeter_square1 / 4 in
  let side_square2 := perimeter_square2 / 4 in
  let area_square1 := side_square1 ^ 2 in
  let area_square2 := side_square2 ^ 2 in
  let area_difference := area_square1 - area_square2 in
  let side_square3 := real.sqrt area_difference in
  4 * side_square3

-- The theorem statement to prove
theorem third_square_perimeter : perimeter_square3 = 24 := by
  sorry

end third_square_perimeter_l120_120701


namespace roots_of_equation_l120_120125

theorem roots_of_equation : 
  (∃ s ∈ ([- real.sqrt 14, real.sqrt 14]).to_set, 
  (∀ x ∈ s, (real.sqrt (14 - x^2)) * (real.sin x - real.cos (2 * x)) = 0) ∧ 
  ( set.analyse (eq (set.card s) 6))) :=
sorry

end roots_of_equation_l120_120125


namespace inequality_proof_l120_120862

variable (a b : ℝ)

theorem inequality_proof (h1 : -1 < b) (h2 : b < 0) (h3 : a < 0) : 
  (a * b > a * b^2) ∧ (a * b^2 > a) := 
by
  sorry

end inequality_proof_l120_120862


namespace at_most_70_percent_acute_triangles_l120_120814

theorem at_most_70_percent_acute_triangles (points : Finset (ℝ × ℝ))
  (h_size : points.card = 100)
  (h_no_collinear : ∀ (p1 p2 p3 : (ℝ × ℝ)), p1 ∈ points → p2 ∈ points → p3 ∈ points → ¬ collinear p1 p2 p3) :
  ∃ (t : Finset (Finset (ℝ × ℝ))), 
    t.card = (points.card.choose 3) ∧ 
    (∀ (tri ∈ t), acute_triangle tri) → t.card ≤ (7 * (points.card.choose 3)) / 10 :=
sorry

end at_most_70_percent_acute_triangles_l120_120814


namespace room_tiling_problem_correct_l120_120393

noncomputable def room_tiling_problem : Prop :=
  let room_length := 6.72
  let room_width := 4.32
  let tile_size := 0.3
  let room_area := room_length * room_width
  let tile_area := tile_size * tile_size
  let num_tiles := (room_area / tile_area).ceil
  num_tiles = 323

theorem room_tiling_problem_correct : room_tiling_problem := 
  sorry

end room_tiling_problem_correct_l120_120393


namespace increasing_function_a_values_l120_120585

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 4 then a * x - 8 else x^2 - 2 * a * x

theorem increasing_function_a_values (a : ℝ) (h : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ ≤ f a x₂) : 
  0 < a ∧ a ≤ 2 :=
sorry

end increasing_function_a_values_l120_120585


namespace largest_prime_factor_always_divides_sum_of_sequence_l120_120372

theorem largest_prime_factor_always_divides_sum_of_sequence
    (sequence : List ℕ)
    (h₁ : ∀ i, i < sequence.length - 1 → 
              (sequence.get i % 100, sequence.get i / 10 % 10) = 
              (sequence.get (i + 1) / 100 % 10, sequence.get (i + 1) / 10 % 10))
    (h₂ : (sequence.get (sequence.length - 1) % 100, sequence.get (sequence.length - 1) / 10 % 10) = 
              (sequence.get 0 / 100 % 10, sequence.get 0 / 10 % 10))
    (h₃ : ∀ i, 100 ≤ sequence.get i ∧ sequence.get i < 1000) :
  37 ∣ sequence.sum :=
by
  sorry

end largest_prime_factor_always_divides_sum_of_sequence_l120_120372


namespace transform_matrix_l120_120816

-- Define the initial and target matrices
def initial_matrix : Matrix ℕ 3 3 :=
  ![[1, 2, 3], [4, 5, 6], [7, 8, 9]]

def target_matrix : Matrix ℕ 3 3 :=
  ![[1, 4, 7], [2, 5, 8], [3, 6, 9]]

-- Define the allowed operations
def add_to_row (m : Matrix ℕ 3 3) (r : Fin 3) (k : ℕ) : Matrix ℕ 3 3 :=
  m.updateRow r (fun x => x + k)

def sub_from_col (m : Matrix ℕ 3 3) (c : Fin 3) (k : ℕ) : Matrix ℕ 3 3 :=
  m.updateCol c (fun x => x - k)

-- Statement of the problem as a theorem
theorem transform_matrix : ∃ (ops : List (Matrix ℕ 3 3 → Matrix ℕ 3 3)), 
  (List.foldl (fun m f => f m) initial_matrix ops) = target_matrix :=
sorry

end transform_matrix_l120_120816


namespace divisibility_by_cube_greater_than_1_l120_120578

theorem divisibility_by_cube_greater_than_1 (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b)
  (hdiv : (a + b^3) % (a^2 + 3 * a * b + 3 * b^2 - 1) = 0) :
  ∃ k : ℕ, 1 < k ∧ k^3 ∣ a^2 + 3 * a * b + 3 * b^2 - 1 := 
by {
  sorry
}

end divisibility_by_cube_greater_than_1_l120_120578


namespace intersection_P_Q_l120_120936

noncomputable def P : Set ℝ := { x | x < 1 }
noncomputable def Q : Set ℝ := { x | x^2 < 4 }

theorem intersection_P_Q :
  P ∩ Q = { x | -2 < x ∧ x < 1 } :=
by 
  sorry

end intersection_P_Q_l120_120936


namespace intersection_A_B_l120_120513

open Set

def A := {-1, 0, 1, 2, 3}

def B := {x : ℝ | (x + 1) * (x - 2) < 0}

theorem intersection_A_B :
  A ∩ B = {0, 1} :=
sorry

end intersection_A_B_l120_120513


namespace minimum_people_who_like_both_l120_120158

theorem minimum_people_who_like_both
    (total_people : ℕ)
    (vivaldi_likers : ℕ)
    (chopin_likers : ℕ)
    (people_surveyed : total_people = 150)
    (like_vivaldi : vivaldi_likers = 120)
    (like_chopin : chopin_likers = 90) :
    ∃ (both_likers : ℕ), both_likers = 60 ∧
                            vivaldi_likers + chopin_likers - both_likers ≤ total_people :=
by 
  sorry

end minimum_people_who_like_both_l120_120158


namespace perfect_square_n_l120_120465

theorem perfect_square_n (m : ℤ) :
  ∃ (n : ℤ), (n = 7 * m^2 + 6 * m + 1 ∨ n = 7 * m^2 - 6 * m + 1) ∧ ∃ (k : ℤ), 7 * n + 2 = k^2 :=
by
  sorry

end perfect_square_n_l120_120465


namespace ellipse_point_range_l120_120848

theorem ellipse_point_range :
  let a := 2
  let b := sqrt 2
  let c := sqrt 2
  let F := (sqrt 2, 0)
  (forall x y, (x^2 / a + y^2 / b = 1) ∧ (k ∈ Icc (- sqrt 2 / 2) (sqrt 2 / 2)) ∧ (k ≠ 0) ∧
    (m ≠ 0) ∧
    (x = x1 + x2) ∧
    (x0^2 / a + y0^2 / b = 1)
    -> sqrt 2 ≤ Real.sqrt (x0^2 + y0^2) ∧
       Real.sqrt (x0^2 + y0^2) ≤ sqrt 3

end ellipse_point_range_l120_120848


namespace rationalize_denominator_l120_120254

theorem rationalize_denominator (a b : ℚ) :
  (a = -7/4) ∧ (b = -3/4) ∧ (∀ C : ℚ, C = 5) →
  (∀ (x y : ℚ), (x = 2 + real.sqrt C) → (y = 1 - real.sqrt C) →
    ∃ u v : ℚ, (u = 7) ∧ (v = 3) →
    (real.sqrt C ≠ 0) →
    (u + v * real.sqrt C) / -4 = a + b * real.sqrt C) :=
by
  sorry

end rationalize_denominator_l120_120254


namespace athlete_A_most_stable_l120_120283

noncomputable def athlete_A_variance : ℝ := 0.019
noncomputable def athlete_B_variance : ℝ := 0.021
noncomputable def athlete_C_variance : ℝ := 0.020
noncomputable def athlete_D_variance : ℝ := 0.022

theorem athlete_A_most_stable :
  athlete_A_variance < athlete_B_variance ∧
  athlete_A_variance < athlete_C_variance ∧
  athlete_A_variance < athlete_D_variance :=
by {
  sorry
}

end athlete_A_most_stable_l120_120283


namespace polynomial_evaluation_l120_120451

theorem polynomial_evaluation :
  ∀ (x : ℤ), x = 4 → x^3 - x^2 + x - 1 = 51 :=
by
  intros x hx
  rw hx
  norm_num
  sorry

end polynomial_evaluation_l120_120451


namespace range_of_a_inequality_l120_120207

noncomputable def f (n : ℕ) (a : ℝ) (x : ℝ) : ℝ := log ((∑ i in finset.range (n - 1), (i+1)^x + n^x * a) / n)

-- Problem 1: Statement
theorem range_of_a (a : ℝ) (n : ℕ) (h : n ≥ 2) :
  (∀ x ∈ Iic (1 : ℝ), (1 + 2^x + 3^x + ⋯ + (n-1)^x + n^x * a) / n > 0) ↔ a ∈ Ioo ((1-n) / 2) ∞ :=
sorry

-- Problem 2: Statement
theorem inequality (a : ℝ) (n : ℕ) (x : ℝ) (h : 2 ≤ n) (ha : a ∈ Ioo 0 1) (hx : x ≠ 0) :
  2 * f n a x < f n a (2 * x) :=
sorry

end range_of_a_inequality_l120_120207


namespace determinant_of_matrix_A_l120_120452

noncomputable def matrix_A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![x + 2, x + 1, x], 
    ![x, x + 2, x + 1], 
    ![x + 1, x, x + 2]]

theorem determinant_of_matrix_A (x : ℝ) :
  (matrix_A x).det = x^2 + 11 * x + 9 :=
by sorry

end determinant_of_matrix_A_l120_120452


namespace avg_daily_sales_with_3_dollar_reduction_reduction_for_target_daily_profit_l120_120397

theorem avg_daily_sales_with_3_dollar_reduction 
  (initial_sales : ℕ) (increase_per_dollar : ℕ) (price_reduction : ℕ) :
  initial_sales = 20 →
  increase_per_dollar = 2 →
  price_reduction = 3 →
  initial_sales + price_reduction * increase_per_dollar = 26 :=
by
  intros hsales hincrease hreduce
  rw [hsales, hincrease, hreduce]
  exact Nat.add_assoc 20 (2 * 3) 0 ▸ rfl

theorem reduction_for_target_daily_profit
  (initial_profit : ℕ) (initial_sales : ℕ) (target_profit : ℕ) :
  initial_profit = 40 →
  initial_sales = 20 →
  target_profit = 1200 →
  ∃ (x : ℕ), (initial_profit - x) * (initial_sales + 2 * x) = target_profit ∧ (initial_profit - x ≥ 25) :=
by
  rintros hprofit hsales htarget
  use 10
  rw [hprofit, hsales, htarget]
  split
  · exact Eq.trans ((40 - 10) * (20 + 2 * 10)) (Nat.add_mul 30 60 40) ▸ Nat.one_mul 1 1200 ▸ rfl
  exact le_of_eq rfl

end avg_daily_sales_with_3_dollar_reduction_reduction_for_target_daily_profit_l120_120397


namespace sum_of_a_b_c_l120_120135

theorem sum_of_a_b_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc1 : a * b + c = 47) (habc2 : b * c + a = 47) (habc3 : a * c + b = 47) : a + b + c = 48 := 
sorry

end sum_of_a_b_c_l120_120135


namespace frame_cell_count_l120_120891

theorem frame_cell_count (dim : ℕ) (width : ℕ) (total_cells : ℕ) 
  (h_dim : dim = 254) (h_width : width = 2) 
  (h_total_cells : total_cells = 2016) : 
  (dim * dim - (dim - 2 * width) * (dim - 2 * width)) = total_cells :=
by
  -- Given conditions inferred from the problem statement
  rw [h_dim, h_width]
  -- Rewrite the goal based on given conditions
  have h1 : 254 * 254 - (254 - 2 * 2) * (254 - 2 * 2) = 64516 - 62500 :=
    by 
      rw [nat.mul_sub_right_distrib, nat.mul_sub_right_distrib]
      -- Break it down into smaller arithmetic steps
      norm_num
  -- Verify the final calculation equals the expected total cells
  rw h_total_cells
  exact h1

end frame_cell_count_l120_120891


namespace inequality_lemma_l120_120952

variable {n : ℕ}
variable (a : Fin n → ℝ)
variable (h : ∀ i, 0 < a i)

theorem inequality_lemma :
  ( ∑ i, a i )^2 / (2 * ∑ i, (a i)^2) ≤ 
  ∑ i, (a i) / (a ((i + 1) % n) + a ((i + 2) % n)) :=
by
  sorry

end inequality_lemma_l120_120952


namespace junior_score_l120_120890

variable (n : ℕ)
variable (j_score : ℕ) -- score for each junior

-- Conditions
def total_students := n
def proportion_juniors := 0.2 * n
def proportion_seniors := 0.8 * n
def average_class_score := 80
def average_senior_score := 78
def total_class_score := average_class_score * total_students
def total_senior_score := average_senior_score * proportion_seniors

-- Define the total Junior's score according to the conditions
def total_junior_score := total_class_score - total_senior_score

-- Prove the score of each junior
theorem junior_score : j_score = total_junior_score / proportion_juniors :=
sorry

end junior_score_l120_120890


namespace limit_calculation_l120_120763

theorem limit_calculation :
  tendsto (λ n : ℕ, (2 * n^2 - 1 : ℝ) / (n^2 + n + 1)) at_top (𝓝 2) :=
by { sorry }

end limit_calculation_l120_120763


namespace solve_system_of_equations_l120_120302

theorem solve_system_of_equations : ∃ (x y : ℝ), 3 * x + y = 5 ∧ x + 3 * y = 7 ∧ x = 1 ∧ y = 2 :=
by
  use 1, 2
  split
  { -- Proof for 3 * x + y = 5
    calc
      3 * (1 : ℝ) + 2 = 3 + 2 := by ring
                  ... = 5 := by ring
  }
  split
  { -- Proof for x + 3 * y = 7
    calc
      (1 : ℝ) + 3 * 2 = 1 + 6 := by ring
                  ... = 7 := by ring
  }
  split
  { -- Proof that x = 1
    exact rfl
  }
  { -- Proof that y = 2
    exact rfl
  }

end solve_system_of_equations_l120_120302


namespace BS_parallel_AM_AMBS_is_rhombus_l120_120929

-- Definitions based on conditions
variables (A B C H M D S : Point)
variables (AB AC AH : Line)
variables [IsIsosceles AB AC] [IsAltitude AH] [PerpendicularBisector H AH] [Circumcircle BMD S]

-- Prove parallel condition
theorem BS_parallel_AM (h1 : IsIsosceles AB AC) (h2 : IsAltitude AH) (h3 : PerpendicularBisector H AH)
  (h4 : Circumcircle BMD S) : Parallel BS AM :=
sorry

-- Prove rhombus condition
theorem AMBS_is_rhombus (h1 : IsIsosceles AB AC) (h2 : IsAltitude AH) (h3 : PerpendicularBisector H AH)
  (h4 : Circumcircle BMD S) : IsRhombus AMBS :=
sorry

end BS_parallel_AM_AMBS_is_rhombus_l120_120929


namespace average_cuts_per_month_l120_120222

theorem average_cuts_per_month :
  (6 * 15 + 6 * 3) / 12 = 9 :=
by
  unfold has_div.div
  unfold has_add.add
  unfold has_mul.mul
  sorry

end average_cuts_per_month_l120_120222


namespace max_non_manager_employees_A_l120_120150

-- Define necessary conditions
variables (active_managers_A total_managers_A : ℕ) (active_managers_B total_managers_B : ℕ) 
variables (male_non_managers female_non_managers : ℕ) (non_managers : ℕ)

-- Define department specific conditions
-- Active managers in department A = total managers in A - managers on vacation
def active_managers_A : ℕ := 8 - 4

-- Department A ratio requirement
def ratio_requirement_A : active_managers_A * 33 < non_managers * 9 := sorry

-- Gender ratio requirement for non-managers in department A
def gender_ratio_requirement_A : male_non_managers * 2 ≤ female_non_managers := sorry

-- Total non-manager employees considering part-time as 0.5 full-time
def total_non_managers : ℕ := male_non_managers + female_non_managers

-- Proving the maximum number of non-manager employees is 12
theorem max_non_manager_employees_A : total_non_managers ≤ 12 :=
by {
  -- Prove parts
  sorry
}

end max_non_manager_employees_A_l120_120150


namespace angle_AOD_120_l120_120165

-- Definitions: angles and their relationships
variables {O A B C D : Type} [linear_order C]
variables (angle : O → O → O → C)
variables (perp : O → O → Prop)

-- Conditions from the problem
def conditions (angle : O → O → O → C) (perp : O → O → Prop) :=
  perp O A ∧ perp O C ∧ perp O B ∧ perp O D ∧
  angle O A D = 2 * angle O B C ∧
  angle O B C = 60

-- The theorem to prove
theorem angle_AOD_120 (angle : O → O → O → C) (perp : O → O → Prop) :
  conditions angle perp → angle O A D = 120 :=
by
  intros h
  sorry

end angle_AOD_120_l120_120165


namespace h_of_neg2_eq_11_l120_120573

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x ^ 2 + 1
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_neg2_eq_11 : h (-2) = 11 := by
  sorry

end h_of_neg2_eq_11_l120_120573


namespace total_listening_days_l120_120965

-- Definitions
variables {x y z t : ℕ}

-- Problem statement
theorem total_listening_days (x y z t : ℕ) : (x + y + z) * t = ((x + y + z) * t) :=
by sorry

end total_listening_days_l120_120965


namespace max_banner_area_l120_120747

theorem max_banner_area (x y : ℕ) (cost_constraint : 330 * x + 450 * y ≤ 10000) : x * y ≤ 165 :=
by
  sorry

end max_banner_area_l120_120747


namespace probability_factor_lt_10_l120_120334

theorem probability_factor_lt_10 (n : ℕ) (h : n = 90) :
  (∃ factors_lt_10 : ℕ, ∃ total_factors : ℕ,
    factors_lt_10 = 7 ∧ total_factors = 12 ∧ (factors_lt_10 / total_factors : ℚ) = 7 / 12) :=
by sorry

end probability_factor_lt_10_l120_120334


namespace three_digit_no_5_no_8_l120_120134

theorem three_digit_no_5_no_8 : 
  let valid_digits := {0, 1, 2, 3, 4, 6, 7, 9}
  let valid_hundreds := valid_digits \ {0}
  (set.card valid_hundreds) * (set.card valid_digits) * (set.card valid_digits) = 448 :=
by
  let valid_digits := {0, 1, 2, 3, 4, 6, 7, 9}
  let valid_hundreds := valid_digits \ {0}
  have h1 : set.card valid_digits = 8 := by norm_num
  have h2 : set.card valid_hundreds = 7 := by norm_num
  suffices h : (7 : ℕ) * 8 * 8 = 448 by exact h
  norm_num

end three_digit_no_5_no_8_l120_120134


namespace fundraiser_goal_l120_120731

noncomputable def individual_needs : List ℕ := [350, 450, 500, 550, 600, 650, 450, 550]
def collective_need : ℕ := 3500

def earnings_expenditures : List (ℕ × ℕ) := [
  (800, 100),
  (950, 150),
  (500, 50),
  (700, 75),
  (550, 100)
]

def total_individual_needs (needs : List ℕ) : ℕ :=
  needs.foldl (λ acc a => acc + a) 0

def total_earnings_expenditures (ee : List (ℕ × ℕ)) : ℕ :=
  ee.foldl (λ acc a => acc + a.1 - a.2) 0

theorem fundraiser_goal:
  let total_individual := total_individual_needs individual_needs in
  let total_needed := total_individual + collective_need in
  let total_raised := total_earnings_expenditures earnings_expenditures in
  let more_needed := total_needed - total_raised in
  individual_needs = [350, 450, 500, 550, 600, 650, 450, 550] ∧
  collective_need + 475 = 3975 :=
by
  sorry

end fundraiser_goal_l120_120731


namespace decreasing_intervals_range_of_g_l120_120504

section
-- Given function f(x)
def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - 2 * (Real.sin (x + 3 * Real.pi / 2)) * (Real.cos (x - Real.pi / 3)) - 3 / 2

-- Predicate for monotonically decreasing interval
def is_monotonically_decreasing (a b : ℝ) (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x

-- Question 1: Prove the interval where f(x) is monotonically decreasing
theorem decreasing_intervals : 
  ∀ k : ℤ, is_monotonically_decreasing (π / 12 + k * π) (7 * π / 12 + k * π) f :=
sorry

-- Given function g(x)
def g (x : ℝ) : ℝ := sqrt 3 * Real.sin (2 * x - Real.pi / 3) + sqrt 3 / 2

-- Question 2: Prove the range of g(x) when x ∈ [0, π/2]
theorem range_of_g : 
  set.range (λ x, g x) ⊆ set.Icc ((-3 + sqrt 3) / 2) ((3 * sqrt 3) / 2) :=
sorry
end

end decreasing_intervals_range_of_g_l120_120504


namespace distinct_digit_addition_l120_120902

theorem distinct_digit_addition (E F G H : ℕ) (h₁ : E ≠ F) (h₂ : E ≠ G) (h₃ : E ≠ H) (h₄ : F ≠ G) (h₅ : F ≠ H) (h₆ : G ≠ H)
  (h₇ : F < 10) (h₈ : E < 10) (h₉ : G < 10) (h₁₀ : H < 10)
  (h₁₁ : EFGGF + FGEEH = HFHHH) : 
  cardinality { H : ℕ | ∃ E F G, E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H ∧ F < 10 ∧ E < 10 ∧ G < 10 ∧ H < 10 ∧ 
               EFGGF = nat.to_digits 10 [E, F, G, G, F].reverse ∧
               FGEEH = nat.to_digits 10 [F, G, E, E, H].reverse ∧
               HFHHH = nat.to_digits 10 [H, F, H, H, H].reverse ∧
               EFGGF + FGEEH = HFHHH } = 9 :=
by
  sorry

end distinct_digit_addition_l120_120902


namespace inequality_proof_l120_120053

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) : x^6 - y^6 + 2 * y^3 < real.pi / 2 :=
by
  sorry

end inequality_proof_l120_120053


namespace min_pairs_continuous_function_l120_120592

noncomputable def f : ℝ → ℝ := sorry

theorem min_pairs_continuous_function :
  (∀ x ∈ set.Icc 0 2015, ∃ y ∈ set.Icc 0 2015, f(x) = f(y) ∧ (x - y) ∈ ℕ) →
  (f(0) = f(2015)) →
  (∀ x ∈ set.Icc 0 2015, continuous_on f (set.Icc 0 2015)) →
  ∃ k, k = 2015 ∧ ∀ x y, (x ∈ set.Icc 0 2015 ∧ y ∈ set.Icc 0 2015 ∧ f(x) = f(y) ∧ (x - y) ∈ ℕ) ↔ (k = 2015) :=
begin
  sorry
end

end min_pairs_continuous_function_l120_120592


namespace sum_unique_BC_div_by_12_l120_120870

theorem sum_unique_BC_div_by_12 : 
  (∑ bc in {bc : ℕ × ℕ | 
    let (B, C) := bc in 
    0 ≤ B ∧ B ≤ 9 ∧ 
    0 ≤ C ∧ C ≤ 9 ∧ 
    B + C + 4 % 3 = 0 ∧ 
    (C = 0 ∨ C = 2 ∨ C = 6 ∨ C = 8)},
  (bc.1 + bc.2)) = 57 := 
sorry

end sum_unique_BC_div_by_12_l120_120870


namespace number_of_roots_l120_120112

noncomputable def roots_equation_count : ℝ :=
  let interval := Icc (-real.sqrt 14) (real.sqrt 14)
  ∑ x in interval, (if sqrt (14 - x^2) * (sin x - cos (2 * x)) = 0 then 1 else 0)

theorem number_of_roots : roots_equation_count = 6 := by {
  sorry
}

end number_of_roots_l120_120112


namespace standard_segments_odd_l120_120850

   -- Define predicates for points being marked red or blue
   inductive Color
   | red
   | blue

   -- Define the condition for a segment being standard
   def is_standard_segment (c1 c2 : Color) : Prop := 
     match c1, c2 with
     | Color.red, Color.blue => True
     | Color.blue, Color.red => True
     | _, _ => False

   -- Define the overall problem statement
   theorem standard_segments_odd (n : ℕ) 
     (colors : List Color) 
     (hlen : colors.length = n + 2) 
     (hstart : colors.head = some Color.red) 
     (hend : colors.last = some Color.blue) : 
     ∃ k, (countp (λ (c1 c2 : Color), is_standard_segment c1 c2) 
     (colors.zip (colors.tail)) = 2 * k + 1) := 
   sorry
   
end standard_segments_odd_l120_120850


namespace height_of_bills_tv_l120_120421

theorem height_of_bills_tv
  (width_bill_tv : ℕ)
  (height_bob_tv length_bob_tv : ℕ)
  (weight_per_square_inch : ℕ)
  (weight_difference : ℕ)
  (area_bob_tv := height_bob_tv * length_bob_tv)
  (weight_bob_tv := area_bob_tv * weight_per_square_inch)
  (area_bill_tv (height_bill_tv : ℕ) := width_bill_tv * height_bill_tv)
  (weight_bill_tv (height_bill_tv : ℕ) := area_bill_tv height_bill_tv * weight_per_square_inch) :
  width_bill_tv = 48 →
  height_bob_tv = 70 →
  length_bob_tv = 60 →
  weight_per_square_inch = 4 →
  weight_difference = 150 * 16 →
  weight_bob_tv - weight_bill_tv 75 = weight_difference →
  height_bill_tv = 75 := 
by
  intro width_bill_tv_eq48
  intro height_bob_tv_eq70
  intro length_bob_tv_eq60
  intro weight_per_square_inch_eq4
  intro weight_difference_eq2400
  intro weight_eq
  sorry

end height_of_bills_tv_l120_120421


namespace hotel_charges_proof_l120_120998

noncomputable def hotel_charges : Prop :=
  let charge_R_single := 100
  let charge_R_double := 100
  let charge_R_suite := 100
  let charge_G_single := charge_R_single * 0.7
  let charge_G_double := charge_R_double * 0.9
  let charge_P_single := charge_R_single * 0.45
  let charge_P_double := charge_R_double * 0.7
  let charge_Q_single := charge_R_single * 0.65
  let charge_Q_double := charge_R_double * 0.8
  let charge_Q_suite := charge_R_suite * 0.7
  (charge_G_single = charge_R_single * 0.7) ∧
  (charge_G_double = charge_R_double * 0.9) ∧
  (charge_Q_suite = charge_R_suite * 0.7) →
  let percentages := [30, 10, 30]
  (percentages.minimum = 10) ∧ 
  (percentages.maximum = 30) ∧ 
  (percentages.maximum - percentages.minimum = 20)

theorem hotel_charges_proof : hotel_charges :=
  sorry

end hotel_charges_proof_l120_120998


namespace find_one_over_x_six_l120_120778

noncomputable def log_base_change {a b : ℝ} (base : ℝ) (x : ℝ) : ℝ := (Real.log x) / (Real.log base)

theorem find_one_over_x_six (x : ℝ) (hx : log_base_change (2*x^3) 2 + log_base_change (4*x^4) 2 = -1) :
  (1 / x^6) = 4 := 
sorry

end find_one_over_x_six_l120_120778


namespace perfect_square_condition_l120_120323

theorem perfect_square_condition (x y : ℕ) :
  ∃ k : ℕ, (x + y)^2 + 3*x + y + 1 = k^2 ↔ x = y := 
by 
  sorry

end perfect_square_condition_l120_120323


namespace arithmetic_sequence_zero_l120_120498

noncomputable def f (x : ℝ) : ℝ :=
  0.3 ^ x - Real.log x / Real.log 2

theorem arithmetic_sequence_zero (a b c x : ℝ) (h_seq : a < b ∧ b < c) (h_pos_diff : b - a = c - b)
    (h_f_product : f a * f b * f c > 0) (h_fx_zero : f x = 0) : ¬ (x < a) :=
by
  sorry

end arithmetic_sequence_zero_l120_120498


namespace find_c_values_for_n_l120_120198

-- Definition of d(n) (number of divisors of n)
def divisors_count (n : ℕ) : ℕ := (List.range (n + 1)).count (λ m, n % m == 0)

-- Definition of φ(n) (Euler's totient function)
noncomputable def euler_totient (n : ℕ) : ℕ :=
(List.range n).filter (λ m, Nat.gcd n m == 1).length

-- Statement of the problem in Lean
theorem find_c_values_for_n (n : ℕ) (c : ℕ) : 
  ∃ c : ℕ,
  divisors_count n + euler_totient n = n + c :=
begin
  sorry
end

end find_c_values_for_n_l120_120198


namespace handshake_problem_l120_120672

theorem handshake_problem :
  let companies := 5
  let representatives_per_company := 5
  let total_people := companies * representatives_per_company
  let possible_handshakes_per_person := total_people - representatives_per_company - 1
  let total_possible_handshakes := total_people * possible_handshakes_per_person
  let unique_handshakes := total_possible_handshakes / 2
  unique_handshakes = 250 :=
by 
  let companies := 5
  let representatives_per_company := 5
  let total_people := companies * representatives_per_company
  let possible_handshakes_per_person := total_people - representatives_per_company - 1
  let total_possible_handshakes := total_people * possible_handshakes_per_person
  let unique_handshakes := total_possible_handshakes / 2
  sorry

end handshake_problem_l120_120672


namespace perpendicular_parallel_l120_120052

variables {a b : Line} {α : Plane}

-- Definition of perpendicular and parallel relations should be available
-- since their exact details were not provided, placeholder functions will be used for demonstration

-- Placeholder definitions for perpendicular and parallel (they should be accurately defined elsewhere)
def perp (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry

theorem perpendicular_parallel {a b : Line} {α : Plane}
    (a_perp_alpha : perp a α)
    (b_perp_alpha : perp b α)
    : parallel a b :=
sorry

end perpendicular_parallel_l120_120052


namespace least_positive_integer_x_l120_120327

theorem least_positive_integer_x
  (n : ℕ)
  (h : (3 * n)^2 + 2 * 43 * (3 * n) + 43^2) % 53 = 0
  (h_pos : 0 < n)
  : n = 21 :=
sorry

end least_positive_integer_x_l120_120327


namespace quadratic_abs_diff_sum_l120_120434

theorem quadratic_abs_diff_sum (a b c: ℤ) (h_eq: a = 5) (h_eq_b: b = -13) (h_eq_c: c = -6) 
  (p q: ℤ) (root_formula: ∀ x, a * x^2 + b * x + c = 0)
  (h_diff: |root_formula (1 + (-b - p.sqrt()) / (2 * a)) - root_formula (1 - (-b - p.sqrt()) / (2 * a))| = √p / q) 
  (h_squarefree: ∀ n: ℤ, n^2 ∣ p → n = 1) :
  p + q = 294 := 
begin
  sorry
end

end quadratic_abs_diff_sum_l120_120434


namespace purely_imaginary_m_eq_neg_half_second_quadrant_m_range_l120_120820

noncomputable def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

-- Part (1)
theorem purely_imaginary_m_eq_neg_half (m : ℝ) (h : z m.imaginary = 0) : m = -1 / 2 :=
by
  sorry

-- Part (2)
theorem second_quadrant_m_range (m : ℝ) (h1 : z m.real < 0) (h2 : z m.imaginary > 0) : -1 / 2 < m ∧ m < 1 :=
by
  sorry

end purely_imaginary_m_eq_neg_half_second_quadrant_m_range_l120_120820


namespace marksman_probability_l120_120677

variables (A B C : Prop)
          (P : Prop → ℝ)
          (hA : P A = 0.6)
          (hB : P B = 0.7)
          (hC : P C = 0.75)
          (independent : (P A * P B * P C = P (A ∩ B ∩ C)))

theorem marksman_probability :
  P (A ∪ B ∪ C) = 0.97 :=
by
  have hAc : P (¬A) = 1 - P A, from sorry,
  have hBc : P (¬B) = 1 - P B, from sorry,
  have hCc : P (¬C) = 1 - P C, from sorry,
  have h_comp : P (¬A ∩ ¬B ∩ ¬C) = P (¬A) * P (¬B) * P (¬C), from sorry,
  calc
    P (A ∪ B ∪ C)
      = 1 - P (¬A ∩ ¬B ∩ ¬C) : by sorry
  ... = 1 - (P (¬A) * P (¬B) * P (¬C)) : by rw h_comp
  ... = 1 - ((1 - P A) * (1 - P B) * (1 - P C)) : by rw [hAc, hBc, hCc]
  ... = 1 - (0.4 * 0.3 * 0.25) : by rw [hA, hB, hC]
  ... = 1 - 0.03 : by sorry
  ... = 0.97 : by sorry

end marksman_probability_l120_120677


namespace find_a_intersections_l120_120078

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x < 1 then sin x else x^3 - 9*x^2 + 25*x + a

theorem find_a_intersections :
  {a : ℝ | (∃ x < 1, f a x = x) ∧ (∃ x ≥ 1, f a x = x) ∧ ∃ y ≥ 1, x ≠ y ∧ f a x = x ∧ f a y = y} = {-20, -16} :=
sorry

end find_a_intersections_l120_120078


namespace inequality_must_hold_l120_120874

theorem inequality_must_hold (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c :=
by {
  sorry
}

end inequality_must_hold_l120_120874


namespace proof_part_1_proof_part_2_l120_120071

-- Given conditions definitions
def quad_inequality_1 (c b : ℝ) : set ℝ := {x | c * x^2 + x + b < 0}
def quad_inequality_2 (b c : ℝ) : set ℝ := {x | b * x^2 + x + c > 0}
def quad_inequality_3 (a : ℝ) : set ℝ := {x | x^2 + x < a^2 - a}

-- Required conditions in problem statement
def condition_inequality_1 : quad_inequality_1 2 (-1) = {x | -1 < x ∧ x < 1 / 2} := sorry
def condition_inequality_2 : quad_inequality_2 (-1) 2 = {x | -1 < x ∧ x < 2} := sorry
def condition_union : let N := quad_inequality_3 a in
  (quad_inequality_2 (-1) 2) ∪ (set.univ \ N) = set.univ := sorry

-- Proving equivalence and range
theorem proof_part_1 : quad_inequality_2 (-1) 2 = {x | -1 < x ∧ x < 2} := sorry
theorem proof_part_2 : ∀ a : ℝ, 
  (quad_inequality_3 a = ∅ ∨ 
  quad_inequality_3 a = {x | a - 1 < x ∧ x < -a}) ∨
  (quad_inequality_3 a = {x | -a < x ∧ x < a - 1}) → 
  (0 ≤ a ∧ a ≤ 1) := sorry

end proof_part_1_proof_part_2_l120_120071


namespace brownie_pieces_count_l120_120358

theorem brownie_pieces_count :
  let tray_length := 24
  let tray_width := 20
  let brownie_length := 3
  let brownie_width := 2
  let tray_area := tray_length * tray_width
  let brownie_area := brownie_length * brownie_width
  let pieces_count := tray_area / brownie_area
  pieces_count = 80 :=
by
  let tray_length := 24
  let tray_width := 20
  let brownie_length := 3
  let brownie_width := 2
  let tray_area := 24 * 20
  let brownie_area := 3 * 2
  let pieces_count := tray_area / brownie_area
  have h1 : tray_length * tray_width = 480 := by norm_num
  have h2 : brownie_length * brownie_width = 6 := by norm_num
  have h3 : pieces_count = 80 := by norm_num
  exact h3

end brownie_pieces_count_l120_120358


namespace proof_concyclic_area_equal_D_triangleABC_area_N_is_four_times_H_l120_120553

open_locale real

variables {A B C H N₁ N₂ N₃ M₁ M₂ M₃ D₁ D₂ D₃ : Point}
variables (triangleABC : Triangle A B C)
variables (orthocenterH : orthocenter triangleABC = H)
variables (N₁_is_reflection : reflection H (line B C) = N₁)
variables (N₂_is_reflection : reflection H (line C A) = N₂)
variables (N₃_is_reflection : reflection H (line A B) = N₃)
variables (M₁_is_midpoint : midpoint B C = M₁)
variables (M₂_is_midpoint : midpoint C A = M₂)
variables (M₃_is_midpoint : midpoint A B = M₃)
variables (D₁_is_reflection : reflection M₁ (line B C) = D₁)
variables (D₂_is_reflection : reflection M₂ (line C A) = D₂)
variables (D₃_is_reflection : reflection M₃ (line A B) = D₃)

theorem proof_concyclic :
  are_cyclic [N₁, N₂, N₃, D₁, D₂, D₃] :=
sorry

theorem area_equal_D_triangleABC :
  area (Triangle.mk D₁ D₂ D₃) = area (Triangle.mk A B C) :=
sorry

theorem area_N_is_four_times_H :
  area (Triangle.mk N₁ N₂ N₃) = 4 * area (Triangle.mk H₁ H₂ H₃) :=
sorry

end proof_concyclic_area_equal_D_triangleABC_area_N_is_four_times_H_l120_120553


namespace circle_equation_intercepts_l120_120725

theorem circle_equation_intercepts (A B : ℝ × ℝ) (S : ℝ) 
    (hA : A = (4, 2)) (hB : B = (-1, 3)) (hS : S = 2) :
    ∃ D E F : ℝ, 
      (A.1 ^ 2 + D * A.1 + A.2 ^ 2 + E * A.2 + F = 0) ∧ 
      (B.1 ^ 2 - D * B.1 + B.2 ^ 2 + E * B.2 + F = 0) ∧ 
      (|D| + |E| = S) ∧ 
      (x y : ℝ, (x ^ 2 + D * x + y ^ 2 + E * y + F = 0) ↔ (x - 1) ^ 2 + y ^ 2 = 13) :=
by
  use -2, 0, -12
  split
  case inl => sorry
  case inr =>
    split
    case inl => sorry
    case inr =>
        split
        case inl => sorry
        case inr => sorry

end circle_equation_intercepts_l120_120725


namespace number_of_knights_l120_120705

-- We define the notion of knights and liars
inductive PersonType
| knight : PersonType
| liar : PersonType

-- Define the input parameters
def people_count : Nat := 80
def statement_range : Nat := 11
def min_liars_in_statement : Nat := 9

-- Assume we have a round table with people's types
def round_table : Fin people_count → PersonType := sorry

-- Define a predicate for the statement each person makes
def statement (i : Fin people_count) : Prop :=
  (Finset.card (Finset.filter (λ j, round_table (i + j + 1) = PersonType.liar) (Finset.range statement_range))) ≥ min_liars_in_statement

-- Define a predicate for the knights' correctness of statement
def knights_statement_correct (i : Fin people_count) :=
  round_table i = PersonType.knight → statement i

-- The main theorem to state the number of knights
theorem number_of_knights :
  (Finset.card (Finset.filter (λ i, round_table i = PersonType.knight) (Finset.univ : Finset (Fin people_count)))) = 20 :=
sorry

end number_of_knights_l120_120705


namespace chess_tournament_possible_l120_120675

section ChessTournament

structure Player :=
  (name : String)
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

def points (p : Player) : ℕ :=
  p.wins + p.draws / 2

def is_possible (A B C : Player) : Prop :=
  (points A > points B) ∧ (points A > points C) ∧
  (points C < points B) ∧
  (A.wins < B.wins) ∧ (A.wins < C.wins) ∧
  (C.wins > B.wins)

theorem chess_tournament_possible (A B C : Player) :
  is_possible A B C :=
  sorry

end ChessTournament

end chess_tournament_possible_l120_120675


namespace hexagon_AF_length_l120_120551

theorem hexagon_AF_length (BC CD DE EF : ℝ) (angleB angleC angleD angleE : ℝ) (angleF : ℝ) 
  (hBC : BC = 2) (hCD : CD = 2) (hDE : DE = 2) (hEF : EF = 2)
  (hangleB : angleB = 135) (hangleC : angleC = 135) (hangleD : angleD = 135) (hangleE : angleE = 135)
  (hangleF : angleF = 90) :
  ∃ (a b : ℝ), (AF = a + 2 * Real.sqrt b) ∧ (a + b = 6) :=
by
  sorry

end hexagon_AF_length_l120_120551


namespace greg_needs_more_money_l120_120863

-- Definitions based on conditions
def cost_scooter : ℝ := 90
def cost_helmet : ℝ := 30
def cost_lock : ℝ := 15
def sales_tax_rate : ℝ := 0.1
def amount_saved : ℝ := 57

-- Theorem to prove the amount Greg needs is 91.5
theorem greg_needs_more_money : 
  let total_cost_before_tax : ℝ := cost_scooter + cost_helmet + cost_lock in
  let sales_tax : ℝ := sales_tax_rate * total_cost_before_tax in
  let total_cost_with_tax : ℝ := total_cost_before_tax + sales_tax in
  let amount_needed : ℝ := total_cost_with_tax - amount_saved in
  amount_needed = 91.5 := 
by
  sorry

end greg_needs_more_money_l120_120863


namespace factorization_correct_l120_120787

theorem factorization_correct (x : ℝ) :
  (x - 3) * (x - 1) * (x - 2) * (x + 4) + 24 = (x - 2) * (x + 3) * (x^2 + x - 8) := 
sorry

end factorization_correct_l120_120787


namespace seats_selection_l120_120212

theorem seats_selection (n k d : ℕ) (hn : n ≥ 4) (hk : k ≥ 2) (hd : d ≥ 2) (hkd : k * d ≤ n) :
  ∃ ways : ℕ, ways = (n / k) * Nat.choose (n - k * d + k - 1) (k - 1) :=
sorry

end seats_selection_l120_120212


namespace probability_second_ball_black_l120_120721

-- Condition definitions
def total_balls := 10
def black_balls := 3
def white_balls := 7

-- Event definition
def total_ways_to_draw_two_balls := 10 * 9
def ways_to_draw_second_black := black_balls * 9

-- Theorem statement
theorem probability_second_ball_black :
  (ways_to_draw_second_black : ℚ) / total_ways_to_draw_two_balls = 3 / 10 :=
by 
  -- Placeholder for the actual proof
  sorry

end probability_second_ball_black_l120_120721


namespace factorial_sum_mod_15_l120_120145

theorem factorial_sum_mod_15 :
  (∑ i in Finset.range 51, Nat.factorial i) % 15 = 3 := 
sorry

end factorial_sum_mod_15_l120_120145


namespace school_can_buy_softballs_l120_120660

theorem school_can_buy_softballs :
  (let original_budget := 15 * 5 in
   let increase := 0.2 * original_budget in
   let new_budget := original_budget + increase in
   new_budget / 9 = 10) :=
begin
  -- Using let-expressions to define intermediate steps
  let original_budget := 15 * 5,
  let increase := 0.2 * original_budget,
  let new_budget := original_budget + increase,
  -- Verifying the final number of softballs that can be bought is 10
  have h : new_budget / 9 = 10,
  { rw [← mul_div_right_comm, show 9 * 10 = 90, by norm_num] },
  exact h,
end

end school_can_buy_softballs_l120_120660


namespace price_increase_decrease_l120_120395

theorem price_increase_decrease (P : ℝ) (y : ℝ) (hP : P > 0) (hy : y > 0) :
  P * (1 - (y / 100)^2) = 0.90 * P → y = 32 :=
by
  intro h
  have : 1 - (y / 100)^2 = 0.90 := by
    calc
      1 - (y / 100)^2 = (P * (1 - (y / 100)^2)) / P := by rw mul_div_cancel_left (1 - (y / 100)^2) hP
      _ = (0.90 * P) / P := by rw h
      _ = 0.90 := by ring
  sorry

end price_increase_decrease_l120_120395


namespace solve_system_of_equations_l120_120272

theorem solve_system_of_equations :
  ∃ x y : ℝ, (x - y = 2) ∧ (2 * x + y = 7) ∧ x = 3 ∧ y = 1 :=
by
  sorry

end solve_system_of_equations_l120_120272


namespace largest_class_students_l120_120156

theorem largest_class_students :
  ∃ x : ℕ, (x + (x - 4) + (x - 8) + (x - 12) + (x - 16) + (x - 20) + (x - 24) +
  (x - 28) + (x - 32) + (x - 36) = 100) ∧ x = 28 :=
by
  sorry

end largest_class_students_l120_120156


namespace inequality_proof_l120_120471

theorem inequality_proof
  (p q a b c d e : Real)
  (hpq : 0 < p ∧ p ≤ a ∧ p ≤ b ∧ p ≤ c ∧ p ≤ d ∧ p ≤ e)
  (hq : a ≤ q ∧ b ≤ q ∧ c ≤ q ∧ d ≤ q ∧ e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e)
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p)) ^ 2 :=
sorry

end inequality_proof_l120_120471


namespace samantha_correct_percentage_l120_120262

theorem samantha_correct_percentage :
  let test1_probs := 20;
  let score1 := 0.75;
  let test2_probs := 50;
  let score2 := 0.85;
  let test3_probs := 15;
  let score3 := 0.60;
  let correct_test1 := score1 * test1_probs;
  let correct_test2 := (Real.ceil (score2 * test2_probs));
  let correct_test3 := score3 * test3_probs;
  let total_correct := correct_test1 + correct_test2 + correct_test3;
  let total_probs := test1_probs + test2_probs + test3_probs;
  (total_correct / total_probs) = 0.79 :=
by
  simp only [score1, score2, score3, test1_probs, test2_probs, test3_probs,
             correct_test1, correct_test2, correct_test3, total_correct, total_probs];
  -- skipped proof
  sorry

end samantha_correct_percentage_l120_120262


namespace distance_knoxville_to_LA_l120_120413

theorem distance_knoxville_to_LA : 
  let knoxville := complex.mk 900 1200
  let los_angeles := 0 : ℂ
  complex.abs (knoxville - los_angeles) = 1500 := 
begin
  let knoxville := complex.mk 900 1200,
  let los_angeles := 0,
  show complex.abs (knoxville - los_angeles) = 1500,
  sorry
end

end distance_knoxville_to_LA_l120_120413


namespace sum_and_product_of_roots_l120_120459

theorem sum_and_product_of_roots (y : ℝ) :
  let a := 1
      b := -1500
      c := 750
  in ∀ (y1 y2 : ℝ), (y1 + y2 = 1500) ∧ (y1 * y2 = 750) ↔ (y1^2 + b*y1 + c = 0 ∧ y2^2 + b*y2 + c = 0) :=
sorry

end sum_and_product_of_roots_l120_120459


namespace find_z_l120_120837

open Complex

theorem find_z (z : ℂ) (h : ∥z∥ ^ 2 + (z + conj z) * I = (3 - I) / (2 + I)) : 
  z = -1 / 2 + real.sqrt 3 / 2 * I ∨ z = -1 / 2 - real.sqrt 3 / 2 * I := 
by
  sorry

end find_z_l120_120837


namespace number_of_tiles_proof_l120_120997

-- Definitions for conditions stated explicitly
def area_of_room : ℝ := 360
def length_to_width_ratio : ℝ := 2
def tile_size_in_inches : ℝ := 4
def feet_to_inches (feet : ℝ) : ℝ := feet * 12

-- Width calculation based on area and length-to-width ratio
def width_in_feet : ℝ := real.sqrt (area_of_room / length_to_width_ratio)
def width_in_inches : ℝ := feet_to_inches width_in_feet

-- Number of tiles calculation based on width in inches and tile size
def number_of_tiles_along_width : ℝ := width_in_inches / tile_size_in_inches

-- The theorem to be proven
theorem number_of_tiles_proof :
  number_of_tiles_along_width = 18 * real.sqrt 5 := by
  sorry

end number_of_tiles_proof_l120_120997


namespace distance_between_planes_l120_120794

-- Definitions of the first and second plane
def plane1 (x y z : ℝ) : Prop := 2 * x - 3 * y + 6 * z - 4 = 0
def plane2 (x y z : ℝ) : Prop := 4 * x - 6 * y + 12 * z + 9 = 0

-- Theorem stating that the distance between these two planes is 17/14
theorem distance_between_planes : 
  ∃ d : ℝ, d = 17 / 14 ∧ 
  ∀ x y z : ℝ, plane1 x y z → 
  ∃ x' y' z' : ℝ, plane2 x' y' z' ∧ 
  (d = abs ((4 * x - 6 * y + 12 * (z) + 9) / (real.sqrt ((4 * 4) + (-6 * -6) + (12 * 12))))) := 
sorry

end distance_between_planes_l120_120794


namespace rectangle_long_side_eq_12_l120_120392

theorem rectangle_long_side_eq_12 (s : ℕ) (a b : ℕ) (congruent_triangles : true) (h : a + b = s) (short_side_is_8 : s = 8) : a + b + 4 = 12 :=
by
  sorry

end rectangle_long_side_eq_12_l120_120392


namespace inequality_proof_l120_120950

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable (ha : ∀ i, 0 < a i)

theorem inequality_proof :
  2 * ∑ i in Finset.range n, (a i)^2 ≥ 
    (∑ i in Finset.range (n - 1), a i * a (i + 1 % n) + a (n - 1) * (a 0 + a 1)) := sorry

end inequality_proof_l120_120950


namespace part1_part2_l120_120942

noncomputable def f (x : ℝ) : ℝ := |2 * x - 2| + |x + 2|

theorem part1 (x : ℝ) : (-3 ≤ x ∧ x ≤ 3/2) ↔ (f x ≤ 6 - x) :=
by
  intros,
  sorry  -- Proof implied by the condition interpretations from solution steps.

theorem part2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 3) :
  (1 / a + 1 / b + 4 / c) ≥ 16 / 3 :=
by
  intros,
  sorry  -- Proof implied by the given conditions and using inequalities.

end part1_part2_l120_120942


namespace length_PK_eq_half_perimeter_l120_120980

-- Defining the problem conditions
variables {A B C P K : Point}
variables {AP AK : Line}
variable (p : ℝ)

-- Defining the points and conditions
def Perpendiculars (A B C P K : Point) (AP AK : Line) : Prop :=
  IsPerpendicular A AP ∧ IsPerpendicular A AK ∧ 
  IsAngleBisector (ExteriorAngle B) AP ∧ IsAngleBisector (ExteriorAngle C) AK

def Perimeter (A B C : Point) (p : ℝ) : Prop :=
  Distance A B + Distance B C + Distance C A = p

-- The theorem to prove the length of PK
theorem length_PK_eq_half_perimeter
  (hperpendiculars : Perpendiculars A B C P K AP AK)
  (hperimeter : Perimeter A B C p) :
  SegmentLength P K = p / 2 :=
by
  sorry

end length_PK_eq_half_perimeter_l120_120980


namespace surplus_deficit_problem_l120_120168

theorem surplus_deficit_problem :
  ∃ x y : ℕ, (8 * x - y = 3 ∧ 7 * x - y = -4) ∧ x = 7 ∧ y = 53 :=
by
  -- Define the conditions as hypotheses
  let x := 7
  let y := 53
  
  have h1 : 8 * x - y = 3 := by
    sorry
  
  have h2 : 7 * x - y = -4 := by
    sorry
  
  use [x, y]
  
  -- Combine the hypotheses to form the conclusion
  exact ⟨⟨h1, h2⟩, rfl, rfl⟩

end surplus_deficit_problem_l120_120168


namespace edges_of_remaining_solid_l120_120745

def side_length_original_cube : ℕ := 5
def side_length_smaller_cube : ℕ := 2
def number_of_corners : ℕ := 8

theorem edges_of_remaining_solid : ∀(side_length_original_cube = 5) (side_length_smaller_cube = 2) (number_of_corners = 8), 
  remaining_edges = 48 :=
by
  sorry

end edges_of_remaining_solid_l120_120745


namespace initial_amount_l120_120609

theorem initial_amount (cost_bread cost_butter cost_juice total_remain total_amount : ℕ) :
  cost_bread = 2 →
  cost_butter = 3 →
  cost_juice = 2 * cost_bread →
  total_remain = 6 →
  total_amount = cost_bread + cost_butter + cost_juice + total_remain →
  total_amount = 15 := by
  intros h_bread h_butter h_juice h_remain h_total
  sorry

end initial_amount_l120_120609


namespace max_three_cards_l120_120225

theorem max_three_cards (n m p : ℕ) (h : n + m + p = 8) (sum : 3 * n + 4 * m + 5 * p = 33) 
  (n_le_10 : n ≤ 10) (m_le_10 : m ≤ 10) (p_le_10 : p ≤ 10) : n ≤ 3 := 
sorry

end max_three_cards_l120_120225


namespace ants_in_field_l120_120743

-- Defining constants
def width_feet : ℕ := 500
def length_feet : ℕ := 600
def ants_per_square_inch : ℕ := 4
def inches_per_foot : ℕ := 12

-- Converting dimensions from feet to inches
def width_inches : ℕ := width_feet * inches_per_foot
def length_inches : ℕ := length_feet * inches_per_foot

-- Calculating the area of the field in square inches
def field_area_square_inches : ℕ := width_inches * length_inches

-- Calculating the total number of ants
def total_ants : ℕ := ants_per_square_inch * field_area_square_inches

-- Theorem statement
theorem ants_in_field : total_ants = 172800000 :=
by
  -- Proof is skipped
  sorry

end ants_in_field_l120_120743


namespace distance_F_to_y_eq_x_l120_120912

def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  (|A * x0 + B * y0 + C|) / (real.sqrt (A^2 + B^2))

theorem distance_F_to_y_eq_x :
  let F := (1 : ℝ, 0 : ℝ) in
  let line_y_eq_x := (1 : ℝ, -1 : ℝ, 0 : ℝ) in
  distance_from_point_to_line 1 0 1 (-1) 0 = (real.sqrt 2) / 2 := by
  sorry

end distance_F_to_y_eq_x_l120_120912


namespace solid_is_cone_l120_120879

-- Define what it means for a solid to have a given view as an isosceles triangle or a circle.
structure Solid :=
(front_view : ℝ → ℝ → Prop)
(left_view : ℝ → ℝ → Prop)
(top_view : ℝ → ℝ → Prop)

-- Definition of isosceles triangle view
def isosceles_triangle (x y : ℝ) : Prop := 
  -- not specifying details of this relationship as a placeholder
  sorry

-- Definition of circle view with a center
def circle_with_center (x y : ℝ) : Prop := 
  -- not specifying details of this relationship as a placeholder
  sorry

-- Define the solid that satisfies the conditions in the problem
def specified_solid (s : Solid) : Prop :=
  (∀ x y, s.front_view x y → isosceles_triangle x y) ∧
  (∀ x y, s.left_view x y → isosceles_triangle x y) ∧
  (∀ x y, s.top_view x y → circle_with_center x y)

-- Given proof problem statement
theorem solid_is_cone (s : Solid) (h : specified_solid s) : 
  ∃ cone, cone = s :=
sorry

end solid_is_cone_l120_120879


namespace crow_brings_worms_exactly_15_times_l120_120726

def distance_between_nest_and_ditch : ℕ := 400
def time_in_hours : ℝ := 1.5
def speed_in_kmph : ℝ := 8

theorem crow_brings_worms_exactly_15_times :
  let speed_in_m_per_min := (speed_in_kmph * 1000) / 60 in
  let time_in_minutes := time_in_hours * 60 in
  let total_distance := speed_in_m_per_min * time_in_minutes in
  let distance_per_round_trip := distance_between_nest_and_ditch * 2 in
  total_distance / distance_per_round_trip = 15 :=
by {
  sorry
}

end crow_brings_worms_exactly_15_times_l120_120726


namespace douglas_won_percentage_l120_120898

theorem douglas_won_percentage (p_X p_Y : ℝ) (r : ℝ) (V : ℝ) (h1 : p_X = 0.76) (h2 : p_Y = 0.4000000000000002) (h3 : r = 2) :
  (1.52 * V + 0.4000000000000002 * V) / (2 * V + V) * 100 = 64 := by
  sorry

end douglas_won_percentage_l120_120898


namespace sum_log_floor_ceil_l120_120430

theorem sum_log_floor_ceil:
  (∑ k in Finset.range 501, k * ((Nat.ceil (Real.log k / Real.log 3)) - (Nat.floor (Real.log k / Real.log 3)))) = 124886 :=
by
  sorry

end sum_log_floor_ceil_l120_120430


namespace least_xy_value_l120_120827

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
sorry

end least_xy_value_l120_120827


namespace scientific_notation_141260_million_l120_120230

theorem scientific_notation_141260_million :
  ∃ (a : ℝ) (n : ℤ), 141260 * 10^6 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4126 ∧ n = 5 :=
by
  sorry

end scientific_notation_141260_million_l120_120230


namespace total_oranges_picked_l120_120966

-- Definitions corresponding to the conditions
def MaryOranges : ℕ := 14
def JasonOranges : ℕ := 41

-- The statement we want to prove
theorem total_oranges_picked : (MaryOranges + JasonOranges) = 55 :=
by 
  -- We add the numbers as given in the problem
  calc MaryOranges + JasonOranges = 14 + 41 : by rw [MaryOranges, JasonOranges]
                              ... = 55 : by norm_num

end total_oranges_picked_l120_120966


namespace percentage_of_sikh_boys_is_10_l120_120155

theorem percentage_of_sikh_boys_is_10 (total_boys : ℕ)
  (perc_muslim : ℝ) (perc_hindu : ℝ) (other_comm_boys : ℕ)
  (H_total_boys : total_boys = 850)
  (H_perc_muslim : perc_muslim = 0.40)
  (H_perc_hindu : perc_hindu = 0.28)
  (H_other_comm_boys : other_comm_boys = 187) :
  ((total_boys - ( (perc_muslim * total_boys) + (perc_hindu * total_boys) + other_comm_boys)) / total_boys) * 100 = 10 :=
by
  sorry

end percentage_of_sikh_boys_is_10_l120_120155


namespace days_to_fill_tank_l120_120565

-- Definitions based on the problem conditions
def tank_capacity_liters : ℕ := 50
def liters_to_milliliters : ℕ := 1000
def rain_collection_per_day : ℕ := 800
def river_collection_per_day : ℕ := 1700
def total_collection_per_day : ℕ := rain_collection_per_day + river_collection_per_day
def tank_capacity_milliliters : ℕ := tank_capacity_liters * liters_to_milliliters

-- Statement of the proof that Jacob needs 20 days to fill the tank
theorem days_to_fill_tank : tank_capacity_milliliters / total_collection_per_day = 20 := by
  sorry

end days_to_fill_tank_l120_120565


namespace solve_system_of_equations_l120_120270

theorem solve_system_of_equations :
  ∃ (x y : ℤ), (x - y = 2) ∧ (2 * x + y = 7) ∧ (x = 3) ∧ (y = 1) :=
by
  sorry

end solve_system_of_equations_l120_120270


namespace gain_percentage_l120_120761

theorem gain_percentage (selling_price gain : ℕ) (h_sp : selling_price = 110) (h_gain : gain = 10) :
  (gain * 100) / (selling_price - gain) = 10 :=
by
  sorry

end gain_percentage_l120_120761


namespace trigonometric_identity_proof_l120_120812

open Real

theorem trigonometric_identity_proof (α : ℝ) (h₁ : tan(α + (π / 4)) = 1 / 2) (h₂ : α ∈ Ioo (-π / 2) 0) :
  (2 * (sin α)^2 + sin (2 * α)) / (cos (α - (π / 4))) = - (2 * sqrt 5) / 5 :=
sorry

end trigonometric_identity_proof_l120_120812


namespace monotonic_increase_interval_l120_120081

noncomputable def piDiv6 : ℝ := Real.pi / 6

theorem monotonic_increase_interval (ω varphi : ℝ) (k : ℤ) (has_pi2 : ω > 0) (varphi_bound : |varphi| < Real.pi / 2)
  (symmetry_distance : ∀ x, f x = 2 * sin (ω * x + varphi) →
    (∃ c : ℝ, f (x + c) = f x ∧ f (x - c) = f x ∧ c = Real.pi / (2 * ω)))
  (even_function_g : ∀ x, g x = 2 * sin (2 * (x - piDiv6) + varphi) → g x = g (-x)) :
  ∃ (k : ℤ), ∃ (l m : ℝ), l = k * Real.pi - piDiv6 ∧ m = k * Real.pi + Real.pi / 3 :=
begin
  sorry -- Proof will be filled in
end

end monotonic_increase_interval_l120_120081


namespace triangle_is_right_l120_120535

noncomputable theory

variables {α : Type*} [linear_ordered_field α]
variables (A B C : α)

def is_right_triangle (A B C : α) : Prop :=
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2

theorem triangle_is_right (h : sin (A + B) * sin (A - B) = sin^2 C) :
  is_right_triangle A B C := 
sorry

end triangle_is_right_l120_120535


namespace two_identical_circles_in_triangle_l120_120560

variable {A B C : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Conditions
def is_tangent (circle : A → ℝ → Prop) (side : B → Prop) : Prop :=
  ∃ t ∈ sides, ∀ p ∈ circle, dist p t = circle.radius

def is_identical (circle1 circle2 : A → A → Prop) : Prop :=
  ∀ x y, circle1 x y ↔ circle2 x y

def touch_each_other (circle1 circle2 : A → Prop) : Prop :=
  ∃ p, circle1 p ∧ circle2 p

-- Main geometric proposition
theorem two_identical_circles_in_triangle
  (triangle : A → A → A → Prop)
  (circle1 circle2 : A → ℝ → Prop) :
  (∃ (A B C : A),
  ∃ (sideAB sideAC : B → Prop),
  triangle A B C ∧
  is_tangent circle1 sideAB ∧
  is_tangent circle1 sideAC ∧
  is_identical circle1 circle2 ∧ 
  touch_each_other circle1 circle2) :=
sorry

end two_identical_circles_in_triangle_l120_120560


namespace sin_cos_sum_identity_l120_120717

theorem sin_cos_sum_identity : 
  sin (15 * Real.pi / 180) * cos (75 * Real.pi / 180) + cos (15 * Real.pi / 180) * sin (105 * Real.pi / 180) = 1 :=
by
  sorry

end sin_cos_sum_identity_l120_120717


namespace find_y_l120_120032

theorem find_y (a b c x : ℝ) (p q r y : ℝ) (h1 : log a / p = log b / q) 
  (h2 : log b / q = log c / r) (h3 : log c / r = log x) (h4 : x ≠ 1) 
  (h5 : a^2 / (b * c) = x ^ y) : y = 2 * p - q - r := 
by 
  sorry

end find_y_l120_120032


namespace gwen_received_money_l120_120461

theorem gwen_received_money (spent remaining : ℕ) (h_spent : spent = 2) (h_remaining : remaining = 5) :
  (received : ℕ) = spent + remaining := by
  have received := spent + remaining
  have h_received : received = 7 := by
    rw [h_spent, h_remaining]
    exact add_comm 2 5
  sorry  -- Proof steps are not required.

end gwen_received_money_l120_120461


namespace Proposition_Validation_l120_120077

-- Definitions for Proposition 1
def prop1 (a b c : ℂ) : Prop := (a^2 + b^2 > c^2) → (a^2 + b^2 - c^2 > 0)

-- Definitions for Proposition 2
def prop2 (a b c : ℂ) : Prop := (a^2 + b^2 - c^2 > 0) → (a^2 + b^2 > c^2)

-- The main statement to check both propositions.
theorem Proposition_Validation :
  (∀ (a b c : ℂ), prop1 a b c) ∧ (¬ ∀ (a b c : ℂ), prop2 a b c) :=
begin
  split,
  { intros a b c h,
    have h_real : (a^2 + b^2) > (c^2) → (a^2 + b^2 - c^2) > 0,
    { intro h1,
      linarith, },
    exact h_real h, },
  { intro h,
    have ex_counter : exists (a b c : ℂ), (a^2 + b^2 - c^2 > 0) ∧ ¬(a^2 + b^2 > c^2),
    { use [2 + I, I, 2 * complex.I.sqrt],
      simp [complex.I_eq_neg_one_symm],
      norm_num,
      split,
      { norm_num, },
      { norm_num, }, },
    rcases ex_counter with ⟨a, b, c, h1, h2⟩,
    exact ⟨a, b, c, h1, h2⟩, },
end

end Proposition_Validation_l120_120077


namespace arithmetic_sequence_sum_l120_120162

-- Define the arithmetic sequence {a_n}
noncomputable def a_n (n : ℕ) : ℝ := sorry

-- Given condition
axiom h1 : a_n 3 + a_n 7 = 37

-- Proof statement
theorem arithmetic_sequence_sum : a_n 2 + a_n 4 + a_n 6 + a_n 8 = 74 :=
by
  sorry

end arithmetic_sequence_sum_l120_120162


namespace Georgia_Gave_Mary_4_Buttons_l120_120470

theorem Georgia_Gave_Mary_4_Buttons :
  ∀ (yellow black green left given : ℕ),
    yellow = 4 → black = 2 → green = 3 → left = 5 →
    given = (yellow + black + green) - left →
    given = 4 :=
by
  intros yellow black green left given h_y h_b h_g h_l h_give
  rw [h_y, h_b, h_g] at h_give
  have h_sum : (4 + 2 + 3) = 9 := by norm_num
  rw h_sum at h_give
  rw h_l at h_give
  have h_goal : 9 - 5 = 4 := by norm_num
  rw h_goal at h_give
  exact h_give

end Georgia_Gave_Mary_4_Buttons_l120_120470


namespace sum_inverses_of_roots_l120_120460

open Polynomial

theorem sum_inverses_of_roots (a b c : ℝ) (h1 : a^3 - 2020 * a + 1010 = 0)
    (h2 : b^3 - 2020 * b + 1010 = 0) (h3 : c^3 - 2020 * c + 1010 = 0) :
    (1/a) + (1/b) + (1/c) = 2 := 
  sorry

end sum_inverses_of_roots_l120_120460


namespace sum_of_squares_eq_product_l120_120324

def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem sum_of_squares_eq_product (n : ℕ) : 
  (∑ k in range (n + 1), (fibonacci (k + 1)) ^ 2) = fibonacci n * fibonacci (n + 1) :=
sorry

end sum_of_squares_eq_product_l120_120324


namespace final_weight_loss_percentage_l120_120338

-- Definitions
variable (W : ℝ) -- Initial weight

-- Conditions
def weight_after_loss (W : ℝ) : ℝ := 0.86 * W
def weight_with_clothes (W : ℝ) : ℝ := (weight_after_loss W) * 1.02
def percentage_loss_measured (W : ℝ) : ℝ := 100 * (W - weight_with_clothes W) / W

-- Theorem statement
theorem final_weight_loss_percentage : percentage_loss_measured W = 12.28 :=
by
  sorry

end final_weight_loss_percentage_l120_120338


namespace twoCatsCanAlwaysCatchMouse_mouseCanAlwaysEvadeThreeCats_l120_120321

-- Define the chessboard and the game setup
namespace CatAndMouseGame

-- Define coordinates on an 8x8 chessboard
structure Position where
  x : Nat
  y : Nat
  h1: x > 0
  h2: x ≤ 8
  h3: y > 0
  h4: y ≤ 8

-- Definitions of conditions
variable (Mouse Cat : Type)

-- Two cats placed on the edge can always catch the mouse
theorem twoCatsCanAlwaysCatchMouse (m : Mouse) (c1 c2 : Cat) (pos_m : Position) (pos_c1 pos_c2 : Position) 
  (h_mouse_not_on_edge : pos_m.x ≠ 1 ∧ pos_m.x ≠ 8 ∧ pos_m.y ≠ 1 ∧ pos_m.y ≠ 8)
  (h_cat1_on_edge : pos_c1.x = 1 ∨ pos_c1.x = 8 ∨ pos_c1.y = 1 ∨ pos_c1.y = 8)
  (h_cat2_on_edge : pos_c2.x = 1 ∨ pos_c2.x = 8 ∨ pos_c2.y = 1 ∨ pos_c2.y = 8) :
  ∃ path : list Position, ∀ (p : Position), p ∈ path → p = pos_m := 
sorry

-- One mouse and three cats case, with mouse's initial two-step move
theorem mouseCanAlwaysEvadeThreeCats (pmouse pcat1 pcat2 pcat3 : Position) :
  ∃ evasion_path : list Position, ∀ (p : Position), p ∈ evasion_path → p = pmouse :=
sorry

end CatAndMouseGame

end twoCatsCanAlwaysCatchMouse_mouseCanAlwaysEvadeThreeCats_l120_120321


namespace exists_infinite_n_dividing_F_F_n_not_dividing_F_n_l120_120195

open Nat

-- Definition of the Fibonacci sequence
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib n + fib (n+1)

-- Statement of the proof problem
theorem exists_infinite_n_dividing_F_F_n_not_dividing_F_n :
  ∃ (infinitely_many_n: ℕ → Prop), (∀ n, infinitely_many_n n → (n ∣ fib (fib n) ∧ ¬ n ∣ fib n)) :=
by
  sorry

end exists_infinite_n_dividing_F_F_n_not_dividing_F_n_l120_120195


namespace winning_candidate_percentage_correct_l120_120673

-- Definition of the problem
def votes (c1 c2 c3 : ℕ) : ℕ := c1 + c2 + c3

def winning_candidate_votes (c1 c2 c3 : ℕ) : ℕ := max c1 (max c2 c3)

def winning_percentage (candidate_votes total_votes : ℕ) : ℚ :=
(candidate_votes.toRat / total_votes.toRat) * 100

-- Given conditions
def c1 : ℕ := 4136
def c2 : ℕ := 7636
def c3 : ℕ := 11628

-- Calculate total votes
def total_votes : ℕ := votes c1 c2 c3

-- Calculate winning candidate votes
def winner_votes : ℕ := winning_candidate_votes c1 c2 c3

-- Calculate the winning candidate's percentage
def winning_percentage_approx : ℚ := winning_percentage winner_votes total_votes

-- The theorem to prove
theorem winning_candidate_percentage_correct : 
(abs (winning_percentage_approx - 51.93) < 0.01) :=
sorry

end winning_candidate_percentage_correct_l120_120673


namespace softballs_with_new_budget_l120_120662

-- Definitions for conditions
def original_budget : ℕ := 15 * 5
def budget_increase : ℕ := original_budget * 20 / 100
def new_budget : ℕ := original_budget + budget_increase
def cost_per_softball : ℕ := 9

-- The statement to prove
theorem softballs_with_new_budget : (new_budget / cost_per_softball) = 10 :=
by
  have h1 : original_budget = 75 := by norm_num
  have h2 : budget_increase = 15 := by norm_num
  have h3 : new_budget = 90 := by norm_num
  show (new_budget / cost_per_softball) = 10, from by norm_num

end softballs_with_new_budget_l120_120662


namespace line_through_M_bisected_eq_l120_120245

theorem line_through_M_bisected_eq (M : Point (ℝ 0 1)) (l1 l2 : Line) :
  (l1.equation = λ x y, x - 3 * y + 10 = 0) →
  (l2.equation = λ x y, 2 * x + y - 8 = 0) →
  ∃ l : Line, (M ∈ l) ∧ (l.equation = λ x y, y = (-1/3) * x + 1) :=
by
  let M := ⟨0, 1⟩
  let l1 := λ (x y : ℝ), x - 3 * y + 10 = 0
  let l2 := λ (x y : ℝ), 2 * x + y - 8 = 0
  use λ (x y : ℝ), y = (-1 / 3) * x + 1
  use sorry

end line_through_M_bisected_eq_l120_120245


namespace min_value_expression_l120_120216

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
    (\frac{5 * c}{a + b} + \frac{5 * a}{b + c} + \frac{3 * b}{a + c}) + 1 ≥ 7.25 := 
sorry

end min_value_expression_l120_120216


namespace number_of_sunglasses_sold_l120_120403

noncomputable def pairs_of_sunglasses_sold : ℕ :=
  let x := 10 in
  x

theorem number_of_sunglasses_sold
  (selling_price : ℕ) (cost_price : ℕ) (sign_cost : ℕ) (x : ℕ)
  (h_selling_price : selling_price = 30)
  (h_cost_price : cost_price = 26)
  (h_sign_cost : sign_cost = 20)
  (h_profit_eq : 2 * (selling_price - cost_price) * x = sign_cost * 2) :
  x = pairs_of_sunglasses_sold :=
by {
  rw [h_selling_price, h_cost_price] at h_profit_eq,
  have profit := 4,  -- 30 - 26
  rw [← h_sign_cost] at h_profit_eq,
  rw [nat.mul_left_inj (by norm_num : 2 ≠ 0) 2 profit at h_profit_eq],
  exact h_profit_eq
}

end number_of_sunglasses_sold_l120_120403


namespace calculation_eq_l120_120713

theorem calculation_eq : 
  -|(-1 : ℝ)| + real.sqrt 9 - real.cbrt (-8 : ℝ) + (real.pi + 1) ^ 0 - (1 / 2) ^ -2 - |real.sqrt 3 - 2| = real.sqrt 3 - 1 := 
by 
  sorry

end calculation_eq_l120_120713


namespace pa_qa_pb_qb_pc_qc_inequality_l120_120206

theorem pa_qa_pb_qb_pc_qc_inequality
    (A B C P Q : Type)
    [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q]
    {a b c : ℝ} (hABC : Triangle A B C)
    (hpa : MetricSpace.dist P A = PA)
    (hqa : MetricSpace.dist Q A = QA)
    (hpb : MetricSpace.dist P B = PB)
    (hqb : MetricSpace.dist Q B = QB)
    (hpc : MetricSpace.dist P C = PC)
    (hqc : MetricSpace.dist Q C = QC) :
    a * PA * QA + b * PB * QB + c * PC * QC ≥ a * b * c :=
by
    sorry

end pa_qa_pb_qb_pc_qc_inequality_l120_120206


namespace arrange_in_ascending_order_l120_120584

/-- Let a = sin (5 * pi / 7), b = cos (2 * pi / 7), and c = tan (2 * pi / 7). -/
def a : Real := Real.sin (5 * Real.pi / 7)
def b : Real := Real.cos (2 * Real.pi / 7)
def c : Real := Real.tan (2 * Real.pi / 7)

theorem arrange_in_ascending_order : b < a ∧ a < c := 
by
  sorry

end arrange_in_ascending_order_l120_120584


namespace part1_part2_l120_120038

-- Definition of function f
def f (x : ℝ) (b : ℝ) (a : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

-- Theorem: f(x) is an odd function implies a = 2 and b = 1
theorem part1 (h : ∀ x : ℝ, f(-x) 1 2 = -f(x) 1 2) :
  ∃ a b : ℝ, a = 2 ∧ b = 1 :=
sorry

-- Theorem: For all x ∈ ℝ, f(x^2 - x) + f(2x^2 - t) < 0 implies t < -1/12
theorem part2 (t : ℝ) (ht : ∀ x : ℝ, f(x^2 - x) 1 2 + f(2 * x^2 - t) 1 2 < 0) :
  t < -1/12 :=
sorry

end part1_part2_l120_120038


namespace carolyn_stitches_rate_l120_120428

/-!
# Proof Problem
Carolyn can sew some stitches per minute. A flower takes 60 stitches to embroider, a unicorn takes 180 stitches, and Godzilla takes 800 stitches. Carolyn wants to embroider Godzilla crushing 3 unicorns and 50 flowers. She needs to spend 1085 minutes embroidering. Prove that Carolyn can sew 4 stitches per minute.
-/

noncomputable def stitches_per_minute (s: ℕ) : Prop :=
  let flowers_stitches := 60 * 50
  let unicorns_stitches := 180 * 3
  let godzilla_stitches := 800
  let total_stitches := flowers_stitches + unicorns_stitches + godzilla_stitches
  total_stitches = 4340 ∧ 1085 * s = total_stitches

theorem carolyn_stitches_rate : ∃ s : ℕ, stitches_per_minute s ∧ s = 4 :=
by {
  have h1: 60 * 50 = 3000 := by norm_num,
  have h2: 180 * 3 = 540 := by norm_num,
  have h3: 800 = 800 := by norm_num,
  have h4: 3000 + 540 + 800 = 4340 := by norm_num,
  existsi 4,
  split,
  unfold stitches_per_minute,
  norm_num,
  exact h4,
  norm_num,
}

end carolyn_stitches_rate_l120_120428


namespace sin_405_eq_sin_585_eq_l120_120772

theorem sin_405_eq : sin (405 * π / 180) = real.sqrt 2 / 2 := by
  sorry

theorem sin_585_eq : sin (585 * π / 180) = - real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sin_585_eq_l120_120772


namespace profit_is_55_l120_120259

-- Define the given conditions:
def cost_of_chocolates (bars: ℕ) (price_per_bar: ℕ) : ℕ :=
  bars * price_per_bar

def cost_of_packaging (bars: ℕ) (cost_per_bar: ℕ) : ℕ :=
  bars * cost_per_bar

def total_sales : ℕ :=
  90

def total_cost (cost_of_chocolates cost_of_packaging: ℕ) : ℕ :=
  cost_of_chocolates + cost_of_packaging

def profit (total_sales total_cost: ℕ) : ℕ :=
  total_sales - total_cost

-- Given values:
def bars: ℕ := 5
def price_per_bar: ℕ := 5
def cost_per_packaging_bar: ℕ := 2

-- Define the profit calculation theorem:
theorem profit_is_55 : 
  profit total_sales (total_cost (cost_of_chocolates bars price_per_bar) (cost_of_packaging bars cost_per_packaging_bar)) = 55 :=
by {
  -- The proof will be inserted here
  sorry
}

end profit_is_55_l120_120259


namespace monotonicity_of_f_range_of_a_l120_120021

noncomputable def f (a x : ℝ) : ℝ := a * x - (Real.sin x / Real.cos x ^ 3)

noncomputable def g (a x : ℝ) : ℝ := f a x - Real.sin (2 * x)

theorem monotonicity_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) : 
  (8 = 8) -> 
  ((0 < x ∧ x < Real.pi / 4) -> f 8 x > 0) ∧ 
  ((Real.pi / 4 < x ∧ x < Real.pi / 2) -> f 8 x < 0) :=
by
  sorry

theorem range_of_a (a : ℝ) (x : ℝ) (h3 : 0 < x) (h4 : x < Real.pi / 2) :
  (f a x < Real.sin (2 * x)) -> 
  a ∈ (-∞, 3] :=
by
  sorry

end monotonicity_of_f_range_of_a_l120_120021


namespace solve_y_l120_120958

theorem solve_y : ∀ y : ℚ, (9 * y^2 + 8 * y - 2 = 0) ∧ (27 * y^2 + 62 * y - 8 = 0) → y = 1 / 9 :=
by
  intro y h
  cases h
  sorry

end solve_y_l120_120958


namespace pages_read_on_saturday_l120_120568

namespace BookReading

def total_pages : ℕ := 93
def pages_read_sunday : ℕ := 20
def pages_remaining : ℕ := 43

theorem pages_read_on_saturday :
  total_pages - (pages_read_sunday + pages_remaining) = 30 :=
by
  sorry

end BookReading

end pages_read_on_saturday_l120_120568


namespace boys_from_school_A_not_studying_science_l120_120149

theorem boys_from_school_A_not_studying_science (total_boys number_of_boys_school_A number_of_boys_studying_science number_of_boys_not_studying_science : ℕ) 
  (h1 : total_boys = 450)
  (h2 : number_of_boys_school_A = 0.20 * total_boys)
  (h3 : number_of_boys_studying_science = 0.30 * number_of_boys_school_A)
  (h4 : number_of_boys_not_studying_science = number_of_boys_school_A - number_of_boys_studying_science) :
  number_of_boys_not_studying_science = 63 := sorry

end boys_from_school_A_not_studying_science_l120_120149


namespace find_h_plus_k_l120_120799

theorem find_h_plus_k (h k : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 6*x - 4*y = 4 ↔ (x - h)^2 + (y - k)^2 = 17) →
  h + k = 5 :=
begin
  sorry
end

end find_h_plus_k_l120_120799


namespace correct_number_of_three_digit_numbers_l120_120128

def count_valid_three_digit_numbers : Nat :=
  let hundreds := [1, 2, 3, 4, 6, 7, 9].length
  let tens_units := [0, 1, 2, 3, 4, 6, 7, 9].length
  hundreds * tens_units * tens_units

theorem correct_number_of_three_digit_numbers :
  count_valid_three_digit_numbers = 448 :=
by
  unfold count_valid_three_digit_numbers
  sorry

end correct_number_of_three_digit_numbers_l120_120128


namespace quadratic_roots_l120_120991

theorem quadratic_roots (x : ℝ) : 
  (2 * x^2 - 4 * x - 5 = 0) ↔ 
  (x = (2 + Real.sqrt 14) / 2 ∨ x = (2 - Real.sqrt 14) / 2) :=
by
  sorry

end quadratic_roots_l120_120991


namespace flight_duration_sum_l120_120925

theorem flight_duration_sum (h m : ℕ) (h_hours : h = 11) (m_minutes : m = 45) (time_limit : 0 < m ∧ m < 60) :
  h + m = 56 :=
by
  sorry

end flight_duration_sum_l120_120925


namespace range_of_AD_dot_BC_l120_120918

noncomputable def point_D_is_on_BC (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1

noncomputable def vector_AD 
  (x : ℝ) (AB AC BC : EuclideanSpace ℝ (Fin 2) → ℝ) 
  (H1 : AB (EuclideanSpace.mk [2]) = 2)
  (H2 : AC (EuclideanSpace.mk [1]) = 1)
  : EuclideanSpace ℝ (Fin 2) → ℝ := 
    λ p, (1 - x) * AB p + x * AC p

noncomputable def dot_product_AD_BC 
  (x : ℝ) (AB AC BC : EuclideanSpace ℝ (Fin 2) → ℝ)
  (H1 : AB (EuclideanSpace.mk [2]) = 2)
  (H2 : AC (EuclideanSpace.mk [1]) = 1)
  (H_angle : innerProductSpace.ofReal (Math.sin (120)) = -1/2) :=
  (vector_AD x AB AC BC H1 H2) (EuclideanSpace.mk [5 - 5 * x])

theorem range_of_AD_dot_BC :
  ∀ (D : EuclideanSpace ℝ (Fin 2))
  (AB AC BC : EuclideanSpace ℝ (Fin 2) → ℝ)
  (H1 : AB (EuclideanSpace.mk [2]) = 2)
  (H2 : AC (EuclideanSpace.mk [1]) = 1)
  (H_angle : innerProductSpace.ofReal (Math.sin (120)) = -1/2),
  ∃ y ∈ segment ℝ (-5 : ℝ) (0 : ℝ),
  ∃ x : ℝ, 
  (0 ≤ x ∧ x ≤ 1) ∧ 
  vector_AD x AB AC BC H1 H2 (EuclideanSpace.mk [5 - 5 * x]) = y := 
λ D AB AC BC H1 H2 H_angle,
begin
  sorry
end

end range_of_AD_dot_BC_l120_120918


namespace geologists_probability_l120_120905

/-- Six roads radiate from the center of a circular field, dividing it into six equal sectors.
Two geologists start at the center, each traveling at 5 km/h along random roads. 
The probability that the distance between them will be more than 8 km after one hour is 0.5. -/
theorem geologists_probability : 
  let speed := 5
  let distance_traveled := speed 
  let total_roads := 6 
  let total_outcomes := total_roads^2
  let favorable_outcomes := total_roads * 3
  let probability := favorable_outcomes / total_outcomes
  in probability = 0.5 := by
  sorry

end geologists_probability_l120_120905


namespace part1_part2_l120_120855

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  Real.log x + a * x - 1 / x + b

noncomputable def g (x : ℝ) (a b : ℝ) : ℝ :=
  f x a b + 2 / x

theorem part1 (a b : ℝ) :
  (∀ x > 0, (g x a b)' ≤ 0) → a ≤ -1/4 :=
sorry

theorem part2 (a b : ℝ) :
  (∀ x > 0, f x a b ≤ 0) → a ≤ 1 - b :=
sorry

end part1_part2_l120_120855


namespace unique_solution_system_eqns_l120_120993

theorem unique_solution_system_eqns :
  ∃ (x y : ℝ), (2 * x - 3 * |y| = 1 ∧ |x| + 2 * y = 4 ∧ x = 2 ∧ y = 1) :=
sorry

end unique_solution_system_eqns_l120_120993


namespace count_valid_digits_l120_120004

theorem count_valid_digits : 
  {n : ℕ | n < 10 ∧ ∃ k : ℕ, 25 * n = k * n ∧ 25 * n % 5 = 0}.card = 1 :=
sorry

end count_valid_digits_l120_120004


namespace original_rulers_eq_l120_120311

-- Define the conditions in Lean
variables (x : ℕ) (tim_addition total_rulers : ℕ)
axioms
  (h_tim_addition : tim_addition = 14)
  (h_total_rulers : total_rulers = 25)
  (h_condition : total_rulers = x + tim_addition)

-- Statement of the problem
theorem original_rulers_eq : x = 11 :=
by
  sorry

end original_rulers_eq_l120_120311


namespace sum_of_squares_of_geometric_sequence_l120_120072

open_locale big_operators

variables {α : Type*} [comm_ring α]

-- Definitions based on given conditions
def Sn (n : ℕ) : α := 2^n - 1
def a (n : ℕ) : α := if n = 0 then 0 else Sn n - Sn (n - 1)

-- Proposition to prove
theorem sum_of_squares_of_geometric_sequence (n : ℕ) : 
  ∑ i in finset.range (n + 1), (a i)^2 = (1 / 3) * (4^n - 1) :=
by
  sorry

end sum_of_squares_of_geometric_sequence_l120_120072


namespace area_of_EFCD_l120_120914

-- Definitions and conditions of the problem
def is_midpoint (P Q R : Point) : Prop :=
  midpoint P Q = R

def is_trapezoid (A B C D : Point) : Prop :=
  parallel A B C D ∧ parallel C D A B

constants (A B C D E F G : Point)
          (length_AB : ℝ)
          (length_CD : ℝ)
          (altitude_ABCD : ℝ)
          (midpoints_AD_BC : is_midpoint A D E ∧ is_midpoint B C F)
          (length_AB_eq : length_AB = 10)
          (length_CD_eq : length_CD = 25)
          (altitude_ABCD_eq : altitude_ABCD = 15)
          (trapezoid_ABCD : is_trapezoid A B C D)
          (intersect_diagonals : intersects A C B D G)

theorem area_of_EFCD :
  let area_EFCD := 7.5 * ((17.5 + 25) / 2) in
    area_EFCD = 159.375 := 
by {
  sorry
}

end area_of_EFCD_l120_120914


namespace ratio_a_c_l120_120652

theorem ratio_a_c {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : ∀ x : ℝ, x^2 + a * x + b = 0 → 2 * x^2 + 2 * a * x + 2 * b = 0)
  (h2 : ∀ x : ℝ, x^2 + b * x + c = 0 → x^2 + b * x + c = 0) :
  a / c = 1 / 8 :=
by
  sorry

end ratio_a_c_l120_120652


namespace romeo_total_profit_is_55_l120_120260

-- Defining the conditions
def number_of_bars : ℕ := 5
def cost_per_bar : ℕ := 5
def packaging_cost_per_bar : ℕ := 2
def total_selling_price : ℕ := 90

-- Defining the profit calculation
def total_cost_per_bar := cost_per_bar + packaging_cost_per_bar
def selling_price_per_bar := total_selling_price / number_of_bars
def profit_per_bar := selling_price_per_bar - total_cost_per_bar
def total_profit := profit_per_bar * number_of_bars

-- Proving the total profit
theorem romeo_total_profit_is_55 : total_profit = 55 :=
by
  sorry

end romeo_total_profit_is_55_l120_120260


namespace value_of_a_minus_b_l120_120070

theorem value_of_a_minus_b
  (a b : ℝ)
  (h1 : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1 / 2 < x ∧ x < 1 / 3) :
  a - b = -10 :=
by sorry

end value_of_a_minus_b_l120_120070


namespace inequality_proof_l120_120526

noncomputable def a : ℝ := log 2 / log (0.2)
noncomputable def b : ℝ := log 3 / log (0.2)
noncomputable def c : ℝ := 2 ^ 0.2

theorem inequality_proof : b < a ∧ a < c :=
  by
  sorry

end inequality_proof_l120_120526


namespace three_digit_no_5_no_8_l120_120133

theorem three_digit_no_5_no_8 : 
  let valid_digits := {0, 1, 2, 3, 4, 6, 7, 9}
  let valid_hundreds := valid_digits \ {0}
  (set.card valid_hundreds) * (set.card valid_digits) * (set.card valid_digits) = 448 :=
by
  let valid_digits := {0, 1, 2, 3, 4, 6, 7, 9}
  let valid_hundreds := valid_digits \ {0}
  have h1 : set.card valid_digits = 8 := by norm_num
  have h2 : set.card valid_hundreds = 7 := by norm_num
  suffices h : (7 : ℕ) * 8 * 8 = 448 by exact h
  norm_num

end three_digit_no_5_no_8_l120_120133


namespace athlete_stable_performance_l120_120284

theorem athlete_stable_performance 
  (A_var : ℝ) (B_var : ℝ) (C_var : ℝ) (D_var : ℝ)
  (avg_score : ℝ)
  (hA_var : A_var = 0.019)
  (hB_var : B_var = 0.021)
  (hC_var : C_var = 0.020)
  (hD_var : D_var = 0.022)
  (havg : avg_score = 13.2) :
  A_var < B_var ∧ A_var < C_var ∧ A_var < D_var :=
by {
  sorry
}

end athlete_stable_performance_l120_120284


namespace measure_angle_CAD_l120_120617

-- Definitions of the problem based on the conditions
variables (A B C D E : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (dist : (A -> B -> C -> D -> E -> ℝ))

-- Conditions
def right_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧
  ∠ABC = 90

def is_square (B C D E : Point) : Prop :=
  ∠BCD = 90 ∧
  dist B C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E B

variables {A B C D E : Point}
variables [right_triangle A B C] [is_square B C D E]

-- Proving the measure of angle CAD
theorem measure_angle_CAD : 
∠CAD = 22.5 :=
by
  sorry

end measure_angle_CAD_l120_120617


namespace sum_abs_diff_is_18_l120_120574

noncomputable def sum_of_possible_abs_diff (a b c d : ℝ) : ℝ :=
  let possible_values := [
      abs ((a + 2) - (d - 7)),
      abs ((a + 2) - (d + 1)),
      abs ((a + 2) - (d - 1)),
      abs ((a + 2) - (d + 7)),
      abs ((a - 2) - (d - 7)),
      abs ((a - 2) - (d + 1)),
      abs ((a - 2) - (d - 1)),
      abs ((a - 2) - (d + 7))
  ]
  possible_values.foldl (· + ·) 0

theorem sum_abs_diff_is_18 (a b c d : ℝ) (h1 : abs (a - b) = 2) (h2 : abs (b - c) = 3) (h3 : abs (c - d) = 4) :
  sum_of_possible_abs_diff a b c d = 18 := by
  sorry

end sum_abs_diff_is_18_l120_120574


namespace monotonicity_of_f_range_of_a_l120_120020

noncomputable def f (a x : ℝ) : ℝ := a * x - (Real.sin x / Real.cos x ^ 3)

noncomputable def g (a x : ℝ) : ℝ := f a x - Real.sin (2 * x)

theorem monotonicity_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) : 
  (8 = 8) -> 
  ((0 < x ∧ x < Real.pi / 4) -> f 8 x > 0) ∧ 
  ((Real.pi / 4 < x ∧ x < Real.pi / 2) -> f 8 x < 0) :=
by
  sorry

theorem range_of_a (a : ℝ) (x : ℝ) (h3 : 0 < x) (h4 : x < Real.pi / 2) :
  (f a x < Real.sin (2 * x)) -> 
  a ∈ (-∞, 3] :=
by
  sorry

end monotonicity_of_f_range_of_a_l120_120020


namespace avg_daily_sales_with_3_dollar_reduction_reduction_for_target_daily_profit_l120_120396

theorem avg_daily_sales_with_3_dollar_reduction 
  (initial_sales : ℕ) (increase_per_dollar : ℕ) (price_reduction : ℕ) :
  initial_sales = 20 →
  increase_per_dollar = 2 →
  price_reduction = 3 →
  initial_sales + price_reduction * increase_per_dollar = 26 :=
by
  intros hsales hincrease hreduce
  rw [hsales, hincrease, hreduce]
  exact Nat.add_assoc 20 (2 * 3) 0 ▸ rfl

theorem reduction_for_target_daily_profit
  (initial_profit : ℕ) (initial_sales : ℕ) (target_profit : ℕ) :
  initial_profit = 40 →
  initial_sales = 20 →
  target_profit = 1200 →
  ∃ (x : ℕ), (initial_profit - x) * (initial_sales + 2 * x) = target_profit ∧ (initial_profit - x ≥ 25) :=
by
  rintros hprofit hsales htarget
  use 10
  rw [hprofit, hsales, htarget]
  split
  · exact Eq.trans ((40 - 10) * (20 + 2 * 10)) (Nat.add_mul 30 60 40) ▸ Nat.one_mul 1 1200 ▸ rfl
  exact le_of_eq rfl

end avg_daily_sales_with_3_dollar_reduction_reduction_for_target_daily_profit_l120_120396


namespace no_integer_exists_such_that_sqrt_n_minus_1_plus_sqrt_n_plus_1_is_rational_l120_120179

theorem no_integer_exists_such_that_sqrt_n_minus_1_plus_sqrt_n_plus_1_is_rational :
  ¬ ∃ n : ℤ, ∃ r : ℚ, (↑(n - 1)).sqrt + (↑(n + 1)).sqrt = r :=
sorry

end no_integer_exists_such_that_sqrt_n_minus_1_plus_sqrt_n_plus_1_is_rational_l120_120179


namespace student_tickets_count_l120_120313

-- Defining the parameters and conditions
variables (A S : ℕ)
variables (h1 : A + S = 59) (h2 : 4 * A + 5 * S / 2 = 222.50)

-- The statement to prove
theorem student_tickets_count : S = 9 :=
by
  sorry

end student_tickets_count_l120_120313


namespace correct_props_l120_120205

-- Define conditions
variables (m n : Line) (α β γ : Plane)
variable [is_different m n] -- m and n are different lines
variable [is_different α β γ] -- α, β, γ are different planes

-- Define propositions
def prop1 : Prop := m ⟂ α ∧ n ⟂ α → m ∥ n
def prop2 : Prop := α ∩ γ = m ∧ β ∩ γ = n ∧ m ∥ n → α ∥ β
def prop3 : Prop := α ∥ β ∧ β ∥ γ ∧ m ⟂ α → m ⟂ γ
def prop4 : Prop := γ ⟂ α ∧ γ ⟂ β → α ∥ β

-- Prove the correct choices
theorem correct_props : prop1 ∧ prop3 ∧ ¬prop2 ∧ ¬prop4 :=
by sorry

end correct_props_l120_120205


namespace stickers_distribution_l120_120517

-- Define the mathematical problem: distributing 10 stickers among 5 sheets with each sheet getting at least one sticker.

def partitions_count (n k : ℕ) : ℕ := sorry

theorem stickers_distribution (n : ℕ) (k : ℕ) (h₁ : n = 10) (h₂ : k = 5) :
  partitions_count (n - k) k = 7 := by
  sorry

end stickers_distribution_l120_120517


namespace midpoint_of_diameter_l120_120638

theorem midpoint_of_diameter : 
  let p1 := (3, -7)
  let p2 := (-5, 5)
  let M := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  in M = (-1, -1) :=
by
  let p1 := (3, -7)
  let p2 := (-5, 5)
  let M := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  show M = (-1, -1)
  sorry

end midpoint_of_diameter_l120_120638


namespace fish_population_decline_l120_120786

theorem fish_population_decline :
  let P := 1.0 in
  (0.7 ^ 9) < 0.05 :=
by
  sorry

end fish_population_decline_l120_120786


namespace range_of_k_l120_120505

noncomputable def f (x k : ℝ) : ℝ := (Real.exp x / x) + (k / 2) * x^2 - k * x

theorem range_of_k (k : ℝ) :
  (∀ x ∈ Set.Ioi 0, Deriv f x k = 0 → x = 1) →
  k ∈ Set.Ici (- Real.exp 2 / 4) :=
by
  sorry 

end range_of_k_l120_120505


namespace angle_measure_l120_120999

-- Define the problem conditions
def angle (x : ℝ) : Prop :=
  let complement := 3 * x + 6
  x + complement = 90

-- The theorem to prove
theorem angle_measure : ∃ x : ℝ, angle x ∧ x = 21 := 
sorry

end angle_measure_l120_120999


namespace gasoline_tank_capacity_l120_120728

theorem gasoline_tank_capacity
  (initial_fill : ℝ) (final_fill : ℝ) (gallons_used : ℝ) (x : ℝ)
  (h1 : initial_fill = 3 / 4)
  (h2 : final_fill = 1 / 3)
  (h3 : gallons_used = 18)
  (h4 : initial_fill * x - final_fill * x = gallons_used) :
  x = 43 :=
by
  -- Skipping the proof
  sorry

end gasoline_tank_capacity_l120_120728


namespace highest_common_factor_evaluation_l120_120954

theorem highest_common_factor_evaluation:
  ∀ (a : ℝ), 
  hcf (x^3 - 20*x^2 + x - a) (x^4 + 3*x^2 + 2) = x^2 + 1 → 
  (x^2 + 1) = hcf (x^3 - 20*x^2 + x - a) (x^4 + 3*x^2 + 2) → 
  (hcf (x^3 - 20*x^2 + x - a) (x^4 + 3*x^2 + 2)).eval 1 = 2 :=
by
  intros a hcf_eq hcf_h
  sorry

end highest_common_factor_evaluation_l120_120954


namespace regions_divided_by_graph_l120_120432

noncomputable def number_of_regions : ℕ :=
48

theorem regions_divided_by_graph :
  ∀ (x y z : ℝ),
  (x = 0 ∨ y = 0 ∨ z = 0 ∨ x + y = 0 ∨ y + z = 0 ∨ z + x = 0 ∨ x - y = 0 ∨ y - z = 0 ∨ z - x = 0) →
  number_of_regions = 48 :=
begin
  sorry
end

end regions_divided_by_graph_l120_120432


namespace tank_capacity_l120_120139

theorem tank_capacity (T : ℝ) (h : (3 / 4) * T + 7 = (7 / 8) * T) : T = 56 := 
sorry

end tank_capacity_l120_120139


namespace possible_ages_count_l120_120727

theorem possible_ages_count : 
  let age_digits := [3, 3, 3, 5, 1, 8] in
  prime_digits := [3, 5] in
  ∃ n, n = 40 ∧ (∀ d ∈ prime_digits, 
  let remaining_digits := age_digits.erase d in
  (nat.factorial 5) / (nat.factorial 3)) = 20 :=
by
  sorry

end possible_ages_count_l120_120727


namespace inequality_lemma_l120_120953

variable {n : ℕ}
variable (a : Fin n → ℝ)
variable (h : ∀ i, 0 < a i)

theorem inequality_lemma :
  ( ∑ i, a i )^2 / (2 * ∑ i, (a i)^2) ≤ 
  ∑ i, (a i) / (a ((i + 1) % n) + a ((i + 2) % n)) :=
by
  sorry

end inequality_lemma_l120_120953


namespace sin_squared_plus_sin_double_eq_one_l120_120012

variable (α : ℝ)
variable (h : Real.tan α = 1 / 2)

theorem sin_squared_plus_sin_double_eq_one : Real.sin α ^ 2 + Real.sin (2 * α) = 1 :=
by
  -- sorry to indicate the proof is skipped
  sorry

end sin_squared_plus_sin_double_eq_one_l120_120012


namespace sum_S_2023_l120_120073

noncomputable def a : ℕ → ℤ 
| 0     := 0
| 1     := 3
| 2     := 1
| (n+3) := a (n+2) - a (n+1)

def S (n : ℕ) : ℤ := ∑ i in finset.range (n + 1), a i

theorem sum_S_2023 : S 2023 = 3 := 
sorry

end sum_S_2023_l120_120073


namespace ring_arrangement_l120_120869

/-- A proof that given 4 distinct rings, where one is a Canadian ring that must be worn by itself, and 5 distinct fingers, the total number of ways to wear the rings on the fingers is 600. --/
theorem ring_arrangement (r1 r2 r3 r4 : Type) (f1 f2 f3 f4 f5 : Type) :
  r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 →
  f1 ≠ f2 ∧ f1 ≠ f3 ∧ f1 ≠ f4 ∧ f1 ≠ f5 ∧ f2 ≠ f3 ∧ f2 ≠ f4 ∧
  f2 ≠ f5 ∧ f3 ≠ f4 ∧ f3 ≠ f5 ∧ f4 ≠ f5 →
  (∃ i ∈ [f1, f2, f3, f4, f5], r1 = i.singleton) →
  ∃ (arrangments : ℕ), arrangements = 600 := by
  sorry

end ring_arrangement_l120_120869


namespace cuboid_height_l120_120796

/-- Given a cuboid with surface area 2400 cm², length 15 cm, and breadth 10 cm,
    prove that the height is 42 cm. -/
theorem cuboid_height (SA l w : ℝ) (h : ℝ) : 
  SA = 2400 → l = 15 → w = 10 → 2 * (l * w + l * h + w * h) = SA → h = 42 :=
by
  intros hSA hl hw hformula
  sorry

end cuboid_height_l120_120796


namespace tangent_line_at_x_axis_intersection_l120_120337

noncomputable def f (x : ℝ) : ℝ := (x^5 - 1) / 5

theorem tangent_line_at_x_axis_intersection : ∃ m b, 
  (∀ x, f(x) = 0 → ((m * x + b = 0) ∧ (m = 1 ∧ b = -1))) := sorry

end tangent_line_at_x_axis_intersection_l120_120337


namespace number_of_propositions_in_statements_l120_120414

def is_proposition (s : String) : Prop :=
  s = "The empty set is a proper subset of any set" ∨
  s = "Natural numbers are even"

def problem_statements : List String :=
  ["The empty set is a proper subset of any set",
   "Find the roots of x^2 - 3x - 4 = 0",
   "What are the integers that satisfy 3x - 2 > 0?",
   "Close the door",
   "Are two lines perpendicular to the same line necessarily parallel?",
   "Natural numbers are even"]

def number_of_propositions (lst : List String) : Nat :=
  lst.filter is_proposition |>.length

theorem number_of_propositions_in_statements :
  number_of_propositions problem_statements = 2 := by
  sorry

end number_of_propositions_in_statements_l120_120414


namespace even_digit_palindromic_square_l120_120625

def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

theorem even_digit_palindromic_square :
  ∃ n, is_palindrome (n * n) ∧ (toString (n * n)).length % 2 = 0 :=
by
  use 836
  sorry

end even_digit_palindromic_square_l120_120625


namespace fraction_squares_sum_l120_120520

theorem fraction_squares_sum (x a y b z c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h1 : x / a + y / b + z / c = 3) (h2 : a / x + b / y + c / z = -3) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 15 := 
by 
  sorry

end fraction_squares_sum_l120_120520


namespace length_of_segment_CD_l120_120612

theorem length_of_segment_CD (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y)
  (h_ratio1 : x = (3 / 5) * (3 + y))
  (h_ratio2 : (x + 3) / y = 4 / 7)
  (h_RS : 3 = 3) :
  x + 3 + y = 273.6 :=
by
  sorry

end length_of_segment_CD_l120_120612


namespace speed_of_current_l120_120734

theorem speed_of_current (m c : ℝ) (h1 : m + c = 20) (h2 : m - c = 18) : c = 1 :=
by
  sorry

end speed_of_current_l120_120734


namespace increasing_function_a_values_l120_120586

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 4 then a * x - 8 else x^2 - 2 * a * x

theorem increasing_function_a_values (a : ℝ) (h : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ ≤ f a x₂) : 
  0 < a ∧ a ≤ 2 :=
sorry

end increasing_function_a_values_l120_120586


namespace lambda_range_l120_120842

/-
  Given the general term of the sequence {an} is an = n + λ / n,
  n ∈ ℕ*, and {an} is a monotonically increasing sequence,
  then the range of real number λ is (-∞, 2).
-/
theorem lambda_range (λ : ℝ) : (∀ n : ℕ, 0 < n → n + (λ / n) < (n + 1) + (λ / (n + 1))) → λ < 2 :=
by
  sorry

end lambda_range_l120_120842


namespace p_sufficient_not_necessary_for_q_l120_120860

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := 1 < x ∧ x < 3
def q (x : ℝ) : Prop := 3^x > 1

-- The theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q : 
  (∀ x : ℝ, p x → q x) ∧ (¬ ∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l120_120860


namespace romeo_total_profit_is_55_l120_120261

-- Defining the conditions
def number_of_bars : ℕ := 5
def cost_per_bar : ℕ := 5
def packaging_cost_per_bar : ℕ := 2
def total_selling_price : ℕ := 90

-- Defining the profit calculation
def total_cost_per_bar := cost_per_bar + packaging_cost_per_bar
def selling_price_per_bar := total_selling_price / number_of_bars
def profit_per_bar := selling_price_per_bar - total_cost_per_bar
def total_profit := profit_per_bar * number_of_bars

-- Proving the total profit
theorem romeo_total_profit_is_55 : total_profit = 55 :=
by
  sorry

end romeo_total_profit_is_55_l120_120261


namespace three_digit_no_5_no_8_l120_120132

theorem three_digit_no_5_no_8 : 
  let valid_digits := {0, 1, 2, 3, 4, 6, 7, 9}
  let valid_hundreds := valid_digits \ {0}
  (set.card valid_hundreds) * (set.card valid_digits) * (set.card valid_digits) = 448 :=
by
  let valid_digits := {0, 1, 2, 3, 4, 6, 7, 9}
  let valid_hundreds := valid_digits \ {0}
  have h1 : set.card valid_digits = 8 := by norm_num
  have h2 : set.card valid_hundreds = 7 := by norm_num
  suffices h : (7 : ℕ) * 8 * 8 = 448 by exact h
  norm_num

end three_digit_no_5_no_8_l120_120132


namespace ratio_a_over_c_l120_120649

variables {a b c x1 x2 : Real}
variables (h1 : x1 + x2 = -a) (h2 : x1 * x2 = b) (h3 : b = 2 * a) (h4 : c = 4 * b)
           (ha_nonzero : a ≠ 0) (hb_nonzero : b ≠ 0) (hc_nonzero : c ≠ 0)

theorem ratio_a_over_c : a / c = 1 / 8 :=
by
  have hc_eq : c = 8 * a := by
    rw [h4, h3]
    simp
  rw [hc_eq]
  field_simp [ha_nonzero]
  norm_num
  sorry -- additional steps if required

end ratio_a_over_c_l120_120649


namespace mean_variance_subtract_constant_l120_120877

-- Define the mean of a set
def mean (S : Set ℝ) : ℝ := (S.to_finset.sum id) / S.to_finset.card

-- Define the variance of a set
noncomputable def variance (S : Set ℝ) : ℝ :=
  let m := mean S in
  (S.to_finset.sum (λ x, (x - m)^2)) / S.to_finset.card

theorem mean_variance_subtract_constant (S : Set ℝ) (c : ℝ) (hS : S.nonempty) (hNZ : c ≠ 0) :
  mean {x - c | x ∈ S} ≠ mean S ∧ variance {x - c | x ∈ S} = variance S :=
sorry

end mean_variance_subtract_constant_l120_120877


namespace angle_BAM_gt_BCM_l120_120048

-- Define the conditions of the problem
variables {A B C M : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variables {AB BC CM BM : Real}
variables (|AB| < |BC| : Prop)
-- Point M lies on the median from vertex B implies some conditions on distances between points,
-- we need to formalize these conditions accordingly.

-- The theorem to state in Lean 4
theorem angle_BAM_gt_BCM (h : |AB| < |BC|) (M_on_median : M lies_on median_of B in triangle ABC) : ∠ BAM > ∠ BCM :=
sorry

end angle_BAM_gt_BCM_l120_120048


namespace total_ceilings_to_paint_l120_120969

theorem total_ceilings_to_paint (ceilings_painted_this_week : ℕ) 
                                (ceilings_painted_next_week : ℕ)
                                (ceilings_left_to_paint : ℕ) 
                                (h1 : ceilings_painted_this_week = 12) 
                                (h2 : ceilings_painted_next_week = ceilings_painted_this_week / 4) 
                                (h3 : ceilings_left_to_paint = 13) : 
    ceilings_painted_this_week + ceilings_painted_next_week + ceilings_left_to_paint = 28 :=
by
  sorry

end total_ceilings_to_paint_l120_120969


namespace clea_ride_time_with_delay_l120_120182

noncomputable def time_to_ride_escalator_with_delay (t_walk_static t_walk_moving t_delay : ℕ) : ℕ :=
  let walking_speed := (t_walk_static:ℝ)⁻¹
  let escalator_speed := (t_walk_moving:ℝ)⁻¹ - walking_speed
  (walking_speed + escalator_speed)⁻¹.toNat + t_delay

theorem clea_ride_time_with_delay : 
  time_to_ride_escalator_with_delay 75 30 5 = 55 :=
by
  sorry

end clea_ride_time_with_delay_l120_120182


namespace Lee_charge_per_lawn_l120_120194

theorem Lee_charge_per_lawn
  (x : ℝ)
  (mowed_lawns : ℕ)
  (total_earned : ℝ)
  (tips : ℝ)
  (tip_amount : ℝ)
  (num_customers_tipped : ℕ)
  (earnings_from_mowing : ℝ)
  (total_earning_with_tips : ℝ) :
  mowed_lawns = 16 →
  total_earned = 558 →
  num_customers_tipped = 3 →
  tip_amount = 10 →
  tips = num_customers_tipped * tip_amount →
  earnings_from_mowing = mowed_lawns * x →
  total_earning_with_tips = earnings_from_mowing + tips →
  total_earning_with_tips = total_earned →
  x = 33 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end Lee_charge_per_lawn_l120_120194


namespace trajectory_eq_l120_120096

variables {V : Type*} [inner_product_space ℝ V] (OA OB : V)
variables λ μ : ℝ

-- Given conditions
hypothes_is
(HA : ∥OA∥ = 1)
(HB : ∥OB∥ = 1)
(HAB : ⟪OA, OB⟫ = 0)

def OC := λ • OA + μ • OB

noncomputable def M := (OA + OB) / 2

-- The desired property to prove in lean
theorem trajectory_eq : 
  ∥OC - M∥ = 1 ↔ (λ - 1 / 2)^2 + (μ - 1 / 2)^2 = 1 :=
sorry

end trajectory_eq_l120_120096


namespace fifth_equation_l120_120056

noncomputable def equation_1 : Prop := 2 * 1 = 2
noncomputable def equation_2 : Prop := 2 ^ 2 * 1 * 3 = 3 * 4
noncomputable def equation_3 : Prop := 2 ^ 3 * 1 * 3 * 5 = 4 * 5 * 6

theorem fifth_equation
  (h1 : equation_1)
  (h2 : equation_2)
  (h3 : equation_3) :
  2 ^ 5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 :=
by {
  sorry
}

end fifth_equation_l120_120056


namespace unique_poly_value_l120_120739

-- Definitions based on conditions and question

def disrespectful_quad_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ r s : ℝ, p = λ x, x^2 - (r + s) * x + r * s ∧ 
  (∀ y, p y = r ∨ p y = s) ∧ 
  -- p(p(x)) = 0 has exactly three real solutions.
  (∃ a b c : ℝ, (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ (p (p x) = 0 → ∃ a b c : ℝ, x = a ∨ x = b ∨ x = c))

def max_sum_roots_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ r s : ℝ, p = λ x, x^2 - (r + s) * x + r * s ∧ 
  ∀ q : ℝ → ℝ, disrespectful_quad_polynomial q → 
  ∃ rq sq : ℝ, q = λ x, x^2 - (rq + sq) * x + rq * sq ∧ 
  r + s ≥ rq + sq

def unique_max_sum_roots_polynomial (p : ℝ → ℝ) : Prop :=
  max_sum_roots_polynomial p ∧ ∀ q : ℝ → ℝ, 
  max_sum_roots_polynomial q → p = q

-- Lean statement for our problem
theorem unique_poly_value : 
  ∃ p : ℝ → ℝ, unique_max_sum_roots_polynomial p ∧ p 1 = 5 / 16 :=
by
  sorry

end unique_poly_value_l120_120739


namespace tetrahedron_volume_proof_l120_120159

noncomputable def tetrahedron_volume (a b : ℝ) : ℝ :=
  (a^2 * real.sqrt (4 * b^2 - 2 * a^2)) / 12

theorem tetrahedron_volume_proof (a b : ℝ) (h : 0 < a ∧ a < b * real.sqrt 2) :
  tetrahedron_volume a b = (a^2 * real.sqrt (4 * b^2 - 2 * a^2)) / 12 :=
by sorry

end tetrahedron_volume_proof_l120_120159


namespace ann_counted_more_cars_l120_120567

def cars_counted_by_ann (A : ℕ) : Prop := 0.85 * A = 300
def total_cars_counted (A F : ℕ) : Prop := 300 + A + F = 983
def ann_more_than_alfred (A F : ℕ) : Prop := A - F = 23

theorem ann_counted_more_cars (A F : ℕ) :
  cars_counted_by_ann A →
  total_cars_counted A F →
  ann_more_than_alfred A F :=
by
  sorry

end ann_counted_more_cars_l120_120567


namespace common_point_of_three_circles_l120_120486

variable (A B C E : Point)
variable (triangle_ABC : Triangle A B C)
variable (E_on_BC : OnLine E (Line B C))

noncomputable def circumcircle_ABC : Circle := circumscribedCircle triangle_ABC
noncomputable def circle_through_E_tangent_at_B : Circle := circleTangentAtPoint E B (Line A B)
noncomputable def circle_through_E_tangent_at_C : Circle := circleTangentAtPoint E C (Line A C)

theorem common_point_of_three_circles :
  ∃ (D : Point), OnCircle D (circumcircle_ABC A B C) ∧ OnCircle D (circle_through_E_tangent_at_B A B C E) ∧ OnCircle D (circle_through_E_tangent_at_C A B C E) :=
by
  sorry

end common_point_of_three_circles_l120_120486


namespace bead_arrangement_probability_l120_120809

theorem bead_arrangement_probability :
  let beads := multiset.of_list ["red", "red", "red", "red", "white", "white", "white", "blue", "blue", "green"] in
  let total_arrangements := (10.factorial / (4.factorial * 3.factorial * 2.factorial * 1.factorial)) in
  let valid_arrangements := calc_valid_arrangements beads in
  let probability := valid_arrangements / total_arrangements in
  probability = 1 / 25 :=
sorry

noncomputable def calc_valid_arrangements (beads : multiset String) : ℕ := sorry

end bead_arrangement_probability_l120_120809


namespace speed_of_train_l120_120750

theorem speed_of_train (length : ℝ) (time : ℝ) (conversion_factor : ℝ) (speed_kmh : ℝ) 
  (h1 : length = 240) (h2 : time = 16) (h3 : conversion_factor = 3.6) :
  speed_kmh = (length / time) * conversion_factor := 
sorry

end speed_of_train_l120_120750


namespace math_problem_l120_120767

theorem math_problem :
  (∑ k in Finset.range 10 + 1, Real.log (4 ^ (k^2)) / Real.log (10^k)) *
  (∑ k in Finset.range 50 + 1, Real.log (36 ^ k) / Real.log (16^k)) = 2686.25 :=
by
  sorry

end math_problem_l120_120767


namespace tangent_lines_l120_120497

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x

theorem tangent_lines (x y : ℝ) :
  (-- Question I
   ((tangent_point : x = 2 ∧ y = 2 ∧ (tangent_line : ∃ m b, m * 2 + b = 2 ∧ tangent_line = 5 * x - y - 8)) ∧
   -- Question II
   ((point_of_tangency : ∃ m n, m * f(m) + n = 0) ∧ ((tangent_line : y = x) ∨ (tangent_line : y = 0))
  )) :=
sorry

end tangent_lines_l120_120497


namespace pump_out_time_l120_120359

-- Define the conditions
def floor_length : ℝ := 30 
def floor_width : ℝ := 40 
def water_depth_inches : ℝ := 24 
def water_depth_feet : ℝ := water_depth_inches / 12
def number_of_pumps : ℕ := 4 
def pump_rate : ℝ := 10 
def gallons_per_cubic_foot : ℝ := 7.5 

-- Calculate the volume of water in cubic feet
def volume_cubic_feet : ℝ := floor_length * floor_width * water_depth_feet

-- Convert the volume to gallons
def volume_gallons : ℝ := volume_cubic_feet * gallons_per_cubic_foot

-- Total pumping rate
def total_pumping_rate : ℝ := number_of_pumps * pump_rate

-- Total time to pump out the water
def total_time : ℝ := volume_gallons / total_pumping_rate

-- Prove that the total time to pump out the water is 450 minutes
theorem pump_out_time : total_time = 450 := by
  sorry

end pump_out_time_l120_120359


namespace probability_both_blue_l120_120184

-- Conditions defined as assumptions
def jarC_red := 6
def jarC_blue := 10
def total_buttons_in_C := jarC_red + jarC_blue

def after_transfer_buttons_in_C := (3 / 4) * total_buttons_in_C

-- Carla removes the same number of red and blue buttons
-- and after transfer, 12 buttons remain in Jar C
def removed_buttons := total_buttons_in_C - after_transfer_buttons_in_C
def removed_red_buttons := removed_buttons / 2
def removed_blue_buttons := removed_buttons / 2

def remaining_red_in_C := jarC_red - removed_red_buttons
def remaining_blue_in_C := jarC_blue - removed_blue_buttons
def remaining_buttons_in_C := remaining_red_in_C + remaining_blue_in_C

def total_buttons_in_D := removed_buttons
def transferred_blue_buttons := removed_blue_buttons

-- Probability calculations
def probability_blue_in_C := remaining_blue_in_C / remaining_buttons_in_C
def probability_blue_in_D := transferred_blue_buttons / total_buttons_in_D

-- Proof
theorem probability_both_blue :
  (probability_blue_in_C * probability_blue_in_D) = (1 / 3) := 
by
  -- sorry is used here to skip the actual proof
  sorry

end probability_both_blue_l120_120184


namespace max_volume_pyramid_l120_120974

-- Conditions:
variables {S A B C O : ℝ}

-- Sphere area condition:
def sphere_area (r : ℝ) : Prop := 4 * π * r^2 = 60 * π

-- Equilateral triangle ABC condition:
def equilateral_triangle (A B C : ℝ) : Prop := A = B ∧ B = C

-- Distance from center O to plane ABC is sqrt(3):
def distance_center_to_plane (O : ℝ) : Prop := O = sqrt 3

-- Plane SAB is perpendicular to plane ABC condition:
def perpendicular_planes (S A B C : ℝ) : Prop := true -- Encoding this spatially is complex but assume it as a true condition for simplicity

-- Main theorem statement:
theorem max_volume_pyramid (r : ℝ) :
  sphere_area r → equilateral_triangle A B C → distance_center_to_plane O → perpendicular_planes S A B C →
  1/3 * (sqrt 3 / 4 * 6^2) * (3 * sqrt 3) = 27 :=
by
  intros h1 h2 h3 h4
  sorry

end max_volume_pyramid_l120_120974


namespace rhombus_properties_l120_120996

section Rhombus
  noncomputable def diag1 : ℝ := 19
  noncomputable def diag2 : ℝ := 28

  variable (area sum_of_diags : ℝ)
  variable (a ∠alpha : ℝ)

  def rhombus_area : Prop := 
    (diag1 + diag2 = sum_of_diags) ∧ (0.5 * diag1 * diag2 = area)

  def side_length (diag1 diag2 : ℝ) : ℝ :=
    Real.sqrt ((diag1 / 2) ^ 2 + (diag2 / 2) ^ 2)

  def approximate (a b margin: ℝ) : Prop :=
    abs (a - b) < margin

  theorem rhombus_properties : 
    rhombus_area 266 47 → 
    approximate a (side_length diag1 diag2) 0.01 ∧ 
    ∠alpha = 2 * Real.arctan (diag1 / diag2)
  := by 
    sorry
end Rhombus

end rhombus_properties_l120_120996


namespace smallest_λ_l120_120483

noncomputable def λ_min (n : ℕ) : ℝ :=
  (n^2 - n) / (4 * n + 2)

theorem smallest_λ (n : ℕ) (h : n ≥ 2) (x : ℕ → ℝ) 
  (hx : ∑ i in finset.range n, (i + 1) * x i = 0) :
  (∑ i in finset.range n, x i)^2 ≤ λ_min n * (∑ i in finset.range n, (x i)^2) := 
sorry

end smallest_λ_l120_120483


namespace original_savings_l120_120405

variable (A B : ℕ)

-- A's savings are 5 times that of B's savings
def cond1 : Prop := A = 5 * B

-- If A withdraws 60 yuan and B deposits 60 yuan, then B's savings will be twice that of A's savings
def cond2 : Prop := (B + 60) = 2 * (A - 60)

-- Prove the original savings of A and B
theorem original_savings (h1 : cond1 A B) (h2 : cond2 A B) : A = 100 ∧ B = 20 := by
  sorry

end original_savings_l120_120405


namespace sum_maximum_n_l120_120488

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ (n : ℕ), a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (s : ℕ → ℤ) : Prop := 
  ∀ n : ℕ, s n = ∑ i in range n, a i

-- Given conditions
def cond1 (a : ℕ → ℤ) : Prop := a 0 + a 2 + a 4 = 105
def cond2 (a : ℕ → ℤ) : Prop := a 1 + a 3 + a 5 = 99

-- Goal: To prove s_n reaches its maximum when n = 20
theorem sum_maximum_n (a : ℕ → ℤ) (s : ℕ → ℤ) : 
  arithmetic_sequence a → sum_of_first_n_terms a s → cond1 a → cond2 a → ∀ m : ℕ, s m ≤ s 20 :=
by 
  sorry

end sum_maximum_n_l120_120488


namespace area_of_square_land_l120_120695

theorem area_of_square_land (total_distance_in_meters : ℕ) (total_laps : ℕ)
  (h_total_distance : total_distance_in_meters = 4800)
  (h_total_laps : total_laps = 4)
  : ∃ area_in_hectares : ℕ, area_in_hectares = 9 :=
by
  -- Definitions based on conditions
  let perimeter_one_lap := total_distance_in_meters / total_laps
  let side_length := perimeter_one_lap / 4
  let area_in_square_meters := side_length * side_length
  let area_in_hectares := area_in_square_meters / 10000
  
  -- Proof
  exists 9
  sorry

end area_of_square_land_l120_120695


namespace cos_pi_over_6_minus_a_eq_5_over_12_l120_120057

theorem cos_pi_over_6_minus_a_eq_5_over_12 (a : ℝ) (h : Real.sin (Real.pi / 3 + a) = 5 / 12) :
  Real.cos (Real.pi / 6 - a) = 5 / 12 :=
by
  sorry

end cos_pi_over_6_minus_a_eq_5_over_12_l120_120057


namespace angle_A_triangle_area_l120_120534

-- Definitions of the problem
def LawOfCosines (a b c A : ℝ) := a^2 = b^2 + c^2 - 2 * b * c * Real.cos A

noncomputable def f (x : ℝ) := (Real.sqrt 3) * Real.sin (x / 2) * Real.cos (x / 2) + (Real.cos (x / 2))^2

-- Proof that the measure of angle A is π/3 given the conditions
theorem angle_A {a b c : ℝ} (h : b^2 + c^2 - a^2 = b * c) : ∃ A : ℝ, 0 < A ∧ A < π ∧ Real.cos A = 1/2 :=
sorry

-- Proof that the area of triangle ABC is √3 given the conditions
theorem triangle_area {a b c B : ℝ} (ha : a = 2) (hb : f B = 1.5) (A : ℝ) (hA : A = π/3) : 
  let S := (1/2) * a^2 * Real.sin (π/3) in S = √3 :=
sorry

end angle_A_triangle_area_l120_120534


namespace part1_monotonicity_part2_range_of_a_l120_120027

-- Part (1)
def f1 (x : ℝ) : ℝ := 8 * x - (Real.sin x / (Real.cos x)^3)

theorem part1_monotonicity (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) :
  (0 < x ∧ x < Real.pi / 4 → Deriv f1 x > 0) ∧ (Real.pi / 4 < x ∧ x < Real.pi / 2 → Deriv f1 x < 0) :=
sorry

-- Part (2)
def f2 (a : ℝ) (x : ℝ) : ℝ := a * x - (Real.sin x / (Real.cos x)^3)

theorem part2_range_of_a (a : ℝ) (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) :
  (∀ x, f2 a x < Real.sin (2 * x)) ↔ a ≤ 3 :=
sorry

end part1_monotonicity_part2_range_of_a_l120_120027


namespace diamond_operation_l120_120805

theorem diamond_operation :
  (∀ (a b : ℝ), a ⊛ b = Real.sqrt (a^2 + b^2)) →
  ((3 ⊛ 4) ⊛ (6 ⊛ 8) = 5 * Real.sqrt 5) :=
by
  intro h
  have step1 : 3 ⊛ 4 = 5 := h 3 4
  have step2 : 6 ⊛ 8 = 10 := h 6 8
  calc
    (3 ⊛ 4) ⊛ (6 ⊛ 8) = 5 ⊛ 10 : by rw [step1, step2]
                      ... = Real.sqrt (5^2 + 10^2) : by rw h
                      ... = Real.sqrt (25 + 100) : by rw [pow_two 5, pow_two 10]
                      ... = Real.sqrt (125) : by rw add_comm
                      ... = 5 * Real.sqrt 5 : by norm_num [Real.sqrt, mul_assoc]

end diamond_operation_l120_120805


namespace exists_circle_perpendicular_to_two_circles_l120_120447

noncomputable section

structure Circle (ℝ : Type) :=
(center : ℝ × ℝ)
(radius : ℝ)

def is_perpendicular (c1 c2 : Circle ℝ) : Prop :=
sorry  -- Assume the implementation details of perpendicular circles

theorem exists_circle_perpendicular_to_two_circles 
   (A : ℝ × ℝ)
   (S1 S2 : Circle ℝ) :
   ∃ (C : Circle ℝ), 
     C.center = A ∧ 
     is_perpendicular C S1 ∧ 
     is_perpendicular C S2 :=
begin
   sorry -- proof goes here
end

end exists_circle_perpendicular_to_two_circles_l120_120447


namespace molecular_weight_H2O_correct_l120_120423

-- Define atomic weights as constants
def atomic_weight_hydrogen : ℝ := 1.008
def atomic_weight_oxygen : ℝ := 15.999

-- Define the number of atoms in H2O
def num_hydrogens : ℕ := 2
def num_oxygens : ℕ := 1

-- Define molecular weight calculation for H2O
def molecular_weight_H2O : ℝ :=
  num_hydrogens * atomic_weight_hydrogen + num_oxygens * atomic_weight_oxygen

-- State the theorem that this molecular weight is 18.015 amu
theorem molecular_weight_H2O_correct :
  molecular_weight_H2O = 18.015 :=
by
  sorry

end molecular_weight_H2O_correct_l120_120423


namespace vacation_cost_l120_120703

theorem vacation_cost (C : ℝ) (h1 : C / 3 - C / 4 = 30) : C = 360 :=
by
  sorry

end vacation_cost_l120_120703


namespace coordinates_of_focus_with_greater_x_coordinate_l120_120418

noncomputable def focus_of_ellipse_with_greater_x_coordinate : (ℝ × ℝ) :=
  let center : ℝ × ℝ := (3, -2)
  let a : ℝ := 3 -- semi-major axis length
  let b : ℝ := 2 -- semi-minor axis length
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let focus_x : ℝ := 3 + c
  (focus_x, -2)

theorem coordinates_of_focus_with_greater_x_coordinate :
  focus_of_ellipse_with_greater_x_coordinate = (3 + Real.sqrt 5, -2) := 
sorry

end coordinates_of_focus_with_greater_x_coordinate_l120_120418


namespace max_value_f_increasing_intervals_f_l120_120219

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, -3 * Real.cos x)
noncomputable def c (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.sin x)
noncomputable def f (x : ℝ) : ℝ := a x.1 * (b x.1 + c x.1) + a x.2 * (b x.2 + c x.2)

theorem max_value_f :
  (∃ x : ℝ, f x = 2 + Real.sqrt 2) ∧ (∀ T > 0, T = π) := 
sorry

theorem increasing_intervals_f :
  ∀ k : ℤ, f (k * π + 3 * π / 8) = f (k * π + 7 * π / 8) := 
sorry

end max_value_f_increasing_intervals_f_l120_120219


namespace ratio_a_over_c_l120_120650

variables {a b c x1 x2 : Real}
variables (h1 : x1 + x2 = -a) (h2 : x1 * x2 = b) (h3 : b = 2 * a) (h4 : c = 4 * b)
           (ha_nonzero : a ≠ 0) (hb_nonzero : b ≠ 0) (hc_nonzero : c ≠ 0)

theorem ratio_a_over_c : a / c = 1 / 8 :=
by
  have hc_eq : c = 8 * a := by
    rw [h4, h3]
    simp
  rw [hc_eq]
  field_simp [ha_nonzero]
  norm_num
  sorry -- additional steps if required

end ratio_a_over_c_l120_120650


namespace symmetric_circle_equation_l120_120290

-- Define the original circle and the line of symmetry
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2
def line_of_symmetry (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Proving the equation of the symmetric circle
theorem symmetric_circle_equation :
  (∀ x y : ℝ, original_circle x y ↔ (x + 3)^2 + (y - 2)^2 = 2) :=
by
  sorry

end symmetric_circle_equation_l120_120290


namespace monotonicity_of_f_range_of_a_l120_120016

-- Definitions given in the conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x - (Real.sin x) / (Real.cos x)^3

-- Problem 1: Monotonicity of f(x) when a = 8
theorem monotonicity_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) (a : ℝ) (ha : a = 8) :
  (0 < x ∧ x < Real.pi / 4 → (∀ (b : ℝ), (x < b ∧ b < Real.pi / 2) → f a b > f a x)) ∧
  (Real.pi / 4 < x ∧ x < Real.pi / 2 → (∀ (b : ℝ), (Real.pi / 4 < b ∧ b < x) → f a b < f a x)) :=
sorry

-- Problem 2: Range of a such that f(x) < sin 2x for all x in (0, π/2)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x < Real.pi / 2) → f a x < Real.sin (2 * x)) ↔ (a ≤ 3) :=
sorry

end monotonicity_of_f_range_of_a_l120_120016


namespace avg_growth_rate_20_max_profit_78_l120_120295
noncomputable theory

-- Define the monthly average growth rate problem
def monthly_avg_growth_rate (july_sales sept_sales july_growth sept_growth: ℕ) : Prop :=
  ∃ x : ℝ, (july_growth = july_sales ∧ sept_growth = sept_sales) ∧
           5 * (1 + x)^2 = 7.2

-- Prove that the monthly average growth rate is 20%
theorem avg_growth_rate_20 : monthly_avg_growth_rate 50000 72000 5000 7200 :=
begin
  refine ⟨0.2, ⟨rfl, rfl⟩, _⟩,
  sorry
end

-- Define the daily profit function and the maximization problem
def maximize_profit (cost price base_sales : ℝ) (daily_sales_increase : ℝ) : ℝ :=
  let sales : ℝ := base_sales + daily_sales_increase * (price - cost) in
  let profit_per_unit : ℝ := price - cost in
  let profit : ℝ := sales * profit_per_unit in
  profit

theorem max_profit_78 (cost price base_sales profit_margin daily_sales_increase : ℝ) 
  (h : profit_margin = 0.3 * cost) :
  ∃ p w, p = 78 ∧ w = maximize_profit 60 p 20 2 ∧ w = 1152 :=
begin
  refine ⟨78, maximize_profit 60 78 20 2, _, _, _⟩,
  { sorry },
  { sorry },
  { sorry }
end


end avg_growth_rate_20_max_profit_78_l120_120295


namespace exists_arithmetic_seq_perfect_powers_l120_120989

def is_perfect_power (x : ℕ) : Prop := ∃ (a k : ℕ), k > 1 ∧ x = a^k

theorem exists_arithmetic_seq_perfect_powers (n : ℕ) (hn : n > 1) :
  ∃ (a d : ℕ) (seq : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → seq i = a + (i - 1) * d)
  ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → is_perfect_power (seq i)
  ∧ d ≠ 0 :=
sorry

end exists_arithmetic_seq_perfect_powers_l120_120989


namespace sequence_general_term_l120_120044

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 0 then 0 else if n = 1 then 2 else n + 1

theorem sequence_general_term (n : ℕ) (h : ∀ {k : ℕ}, k > 0 → sequence (k + 1) - sequence k = 1) :
  sequence n = n + 1 := by
  cases n
  · simp [sequence]
  · induction n with n ih
    · trivial
    · simp [sequence] at *
      sorry

end sequence_general_term_l120_120044


namespace cosine_difference_triangle_l120_120173

theorem cosine_difference_triangle 
  (A B C M O Q R : Type*)
  [Point A] [Point B] [Point C] [Point M] [Point O] [Point Q] [Point R]
  (angle_BAC : Real := π / 3)
  (ratio_condition : Real) 
  (h1 : ratio_condition = Real.sqrt 7 * (QR / AR))
  (h2 : ∠BAC = π / 3) :
  ∃ cos_val : Real, cos_val = -1 / 8 := 
begin
  existsi (-1/8),
  exact sorry,
end

end cosine_difference_triangle_l120_120173


namespace correct_addition_l120_120554

theorem correct_addition
  (incorrect_sum: ℕ)
  (incorrect_units_digit: ℕ)
  (correct_units_digit: ℕ)
  (incorrect_tens_digit: ℕ)
  (correct_tens_digit: ℕ)
  (incorrect_addend: ℕ) :
  (incorrect_sum = 111) →
  (incorrect_units_digit = 8) →
  (correct_units_digit = 5) →
  (incorrect_tens_digit = 4) →
  (correct_tens_digit = 7) →
  (incorrect_addend = 48) →
  let correct_units_correction := incorrect_sum - (incorrect_units_digit - correct_units_digit) in
  let correct_tens_correction := correct_units_correction + ((correct_tens_digit - incorrect_tens_digit) * 10) in
  correct_tens_correction = 138 :=
by
  intros
  simp
  sorry

end correct_addition_l120_120554


namespace expected_determinant_l120_120927

noncomputable theory

-- Defining the matrix A and required conditions in Lean
def A (n : ℕ) : matrix (fin (2 * n)) (fin (2 * n)) ℝ :=
  λ i j, if (i, j) ∈ (λ p : fin (2 * n) × fin (2 * n), set.Icc 0 1) then 1 / 2 else 0 -- Prob 1/2

-- Define the transpose
def transpose {m n} (A : matrix (fin m) (fin n) ℝ) : matrix (fin n) (fin m) ℝ :=
  λ i j, A j i

-- Define defining X as A - A^T
def X (A : matrix (fin (2 * n)) (fin (2 * n)) ℝ) : matrix (fin (2 * n)) (fin (2 * n)) ℝ :=
  A - transpose A

-- The expected value
def expected_value (X : matrix (fin (2 * n)) (fin (2 * n)) ℝ) : ℝ := 
  Sorry -- The detailed calculation for the expectation goes here

theorem expected_determinant (n : ℕ) :
  expected_value (λ (i j : fin (2 * n)), A i j - A j i) = (factorial (2 * n - 1)) / (2^n) :=
sorry

end expected_determinant_l120_120927


namespace length_of_QZ_l120_120908

-- Given is the basic setup for the geometry problem
variables {A B Q Y Z : Point}
variables (AZ BQ QY : ℝ)
variables (AB_YZ_parallel: ¬(A = B) ∧ ¬(Y = Z) ∧ parallel (line_through A B) (line_through Y Z))
variables (d_AZ : dist A Z = 56) 
variables (d_BQ : dist B Q = 18)
variables (d_QY : dist Q Y = 36)

-- Defining lengths for AZ, BQ, and QY
noncomputable def length_QZ (AQ QZ : ℝ) : ℝ := QZ

theorem length_of_QZ (AQ QZ : ℝ) :
  AQ + QZ = 56 ∧ AQ * 2 = QZ ∧ dist B Q = 18 ∧ dist Q Y = 36 ∧ parallel (line_through A B) (line_through Y Z)
  → QZ = 112 / 3 :=
by
  sorry

end length_of_QZ_l120_120908


namespace part_I_part_II_l120_120856

noncomputable def f (x a : ℝ) := sin x + a * cos x

theorem part_I (a : ℝ) : 
  (∃ a, f (x := -π/4) a ∧ ∀ x, monotone_increasing_on f [2 * k * π - π / 4, 2 * k * π + 3 * π / 4]) → a = -1 :=
sorry

theorem part_II (α β : ℝ) 
  (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
  (h1 : f (α + π/4) (-1) = sqrt 10 / 5) 
  (h2 : f (β + 3 * π / 4) (-1) = 3 * sqrt 5 / 5) : 
  sin (α + β) = sqrt 2 / 2 :=
sorry

end part_I_part_II_l120_120856


namespace least_pos_int_x_l120_120328

theorem least_pos_int_x (x : ℕ) (h1 : ∃ k : ℤ, (3 * x + 43) = 53 * k) 
  : x = 21 :=
sorry

end least_pos_int_x_l120_120328


namespace scientific_notation_141260_million_l120_120231

theorem scientific_notation_141260_million :
  ∃ (a : ℝ) (n : ℤ), 141260 * 10^6 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4126 ∧ n = 5 :=
by
  sorry

end scientific_notation_141260_million_l120_120231


namespace scientific_notation_conversion_l120_120234

theorem scientific_notation_conversion (x : ℝ) (h_population : x = 141260000) :
  x = 1.4126 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l120_120234


namespace sum_of_squares_iff_2n_sum_of_squares_l120_120618

theorem sum_of_squares_iff_2n_sum_of_squares (n : ℕ) (hn : n > 0) :
  (∃ a b : ℤ, n = a^2 + b^2) ↔ (∃ c d : ℤ, 2 * n = c^2 + d^2) :=
begin
  sorry
end

end sum_of_squares_iff_2n_sum_of_squares_l120_120618


namespace monotonicity_of_f_range_of_a_l120_120018

-- Definitions given in the conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x - (Real.sin x) / (Real.cos x)^3

-- Problem 1: Monotonicity of f(x) when a = 8
theorem monotonicity_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) (a : ℝ) (ha : a = 8) :
  (0 < x ∧ x < Real.pi / 4 → (∀ (b : ℝ), (x < b ∧ b < Real.pi / 2) → f a b > f a x)) ∧
  (Real.pi / 4 < x ∧ x < Real.pi / 2 → (∀ (b : ℝ), (Real.pi / 4 < b ∧ b < x) → f a b < f a x)) :=
sorry

-- Problem 2: Range of a such that f(x) < sin 2x for all x in (0, π/2)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x < Real.pi / 2) → f a x < Real.sin (2 * x)) ↔ (a ≤ 3) :=
sorry

end monotonicity_of_f_range_of_a_l120_120018


namespace real_part_is_one_l120_120033

noncomputable def real_part_of_z (a : ℝ) (z : ℂ) : ℝ :=
  if h : z / (2 + a * complex.I) = 2 / (1 + complex.I) ∧ complex.im z = -3 then
    complex.re z
  else
    0 -- unreachable case

theorem real_part_is_one (a : ℝ) (z : ℂ) (h1 : z / (2 + a * complex.I) = 2 / (1 + complex.I))
  (h2 : complex.im z = -3) : real_part_of_z a z = 1 :=
by
  sorry

end real_part_is_one_l120_120033


namespace determinant_problem_l120_120939

open Matrix

variables {α : Type*} [CommRing α]

def E (u v w x : Vector α) : α :=
  det ![![u[0], v[0], w[0], x[0]],
         ![u[1], v[1], w[1], x[1]],
         ![u[2], v[2], w[2], x[2]],
         ![u[3], v[3], w[3], x[3]]]

theorem determinant_problem
  (u v w x : Vector α) :
  det ![![u[0] + v[0], v[0] + w[0], w[0] + x[0], x[0] + u[0]],
         ![u[1] + v[1], v[1] + w[1], w[1] + x[1], x[1] + u[1]],
         ![u[2] + v[2], v[2] + w[2], w[2] + x[2], x[2] + u[2]],
         ![u[3] + v[3], v[3] + w[3], w[3] + x[3], x[3] + u[3]]] 
  = E u v w x :=
sorry

end determinant_problem_l120_120939


namespace sandy_books_l120_120263

theorem sandy_books (benny_books : ℕ) (tim_books : ℕ) (total_books : ℕ) (h1 : benny_books = 24) (h2 : tim_books = 33) (h3 : total_books = 67) :
  total_books - (benny_books + tim_books) = 10 :=
by
  rw [h1, h2, h3]
  sorry

end sandy_books_l120_120263


namespace part1_part2_l120_120821

-- Define the complex number z in terms of m
def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

-- State the condition where z is a purely imaginary number
def purelyImaginary (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 = 0 ∧ m^2 - 3 * m + 2 ≠ 0

-- State the condition where z is in the second quadrant.
def inSecondQuadrant (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 < 0 ∧ m^2 - 3 * m + 2 > 0

-- Part 1: Prove that m = -1/2 given that z is purely imaginary.
theorem part1 : purelyImaginary m → m = -1/2 :=
sorry

-- Part 2: Prove the range of m for z in the second quadrant.
theorem part2 : inSecondQuadrant m → -1/2 < m ∧ m < 1 :=
sorry

end part1_part2_l120_120821


namespace triangle_BC_length_l120_120919

theorem triangle_BC_length (A : ℝ) (AC : ℝ) (S : ℝ) (BC : ℝ)
  (h1 : A = 60) (h2 : AC = 16) (h3 : S = 220 * Real.sqrt 3) :
  BC = 49 :=
by
  sorry

end triangle_BC_length_l120_120919


namespace area_of_triangle_ABC_l120_120628

theorem area_of_triangle_ABC :
  ∃ (ABC : Triangle), ABC.right ∧
  ABC.angles B = 90 ∧
  ABC.angles A = 45 ∧
  ABC.sides BC = 30 ∧
  area ABC = 225 :=
by
  sorry

end area_of_triangle_ABC_l120_120628


namespace positive_integer_expression_l120_120006

theorem positive_integer_expression (x : ℝ) (h : x ≠ 0) : (|x + |x|| / x = 2) ↔ (0 < x) := by
  sorry

end positive_integer_expression_l120_120006


namespace trios_total_songs_l120_120003

theorem trios_total_songs (a t m : ℕ) (h1 : 6 ≤ a ∧ a ≤ 8) (h2 : 6 ≤ t ∧ t ≤ 8) (h3 : 6 ≤ m ∧ m ≤ 8) (h4 : 14 + a + t + m = 33):
  let P := (9 + 5 + a + t + m) / 3 in P = 11 :=
by
  sorry

end trios_total_songs_l120_120003


namespace gcd_m_n_l120_120208

   -- Define m and n according to the problem statement
   def m : ℕ := 33333333
   def n : ℕ := 666666666

   -- State the theorem we want to prove
   theorem gcd_m_n : Int.gcd m n = 3 := by
     -- put proof here
     sorry
   
end gcd_m_n_l120_120208


namespace solution_set_l120_120501

noncomputable def f : ℝ → ℝ := sorry
def f' (x : ℝ) : ℝ := sorry

theorem solution_set
  (f_defined : ∀ x : ℝ, f x ∈ ℝ)
  (f_at_1 : f 1 = 1)
  (f_deriv_lt : ∀ x : ℝ, f' x < 1/3)
  (h_f_deriv : ∀ x : ℝ, has_deriv_at f (f' x) x) :
  { x : ℝ | f x < x / 3 + 2 / 3 } = { x | x > 1 } :=
sorry

end solution_set_l120_120501


namespace distance_traveled_l120_120608

theorem distance_traveled (d : ℕ) :
  let southward_distance := 20
  let first_right_distance := 10
  let second_right_distance := 20
  let third_turn_distance := d
  let total_distance := 30
  distance_from_home southward_distance first_right_distance second_right_distance third_turn_distance = 10 + d ∧ 10 + d = total_distance → d = 20 := by
  intros
  sorry

end distance_traveled_l120_120608


namespace melinda_textbooks_problem_l120_120967

theorem melinda_textbooks_problem :
  let total_ways := (choose 13 4) * (choose 9 4) * (choose 5 5),
      favorable_ways := choose 9 1 * choose 8 4 * choose 4 4,
      probability := favorable_ways / total_ways,
      gcd := Nat.gcd 1 4120
  in gcd = 1 ∧ (1 + 4120 = 4121) :=
by
  sorry

end melinda_textbooks_problem_l120_120967


namespace problem_subtraction_of_negatives_l120_120690

theorem problem_subtraction_of_negatives :
  12.345 - (-3.256) = 15.601 :=
sorry

end problem_subtraction_of_negatives_l120_120690


namespace extremum_increasing_decreasing_no_zeros_l120_120476

noncomputable def f (a x : ℝ) : ℝ := Real.log x - Real.exp (x + a)

theorem extremum_increasing_decreasing (a : ℝ) (h : x = 1) :
  (∀ x ∈ Ioo 0 1, deriv (f a) x > 0) ∧ (∀ x ∈ Ioi 1, deriv (f a) x < 0) :=
by
  sorry

theorem no_zeros (a x : ℝ) (h : a ≥ -2) : f a x ≠ 0 :=
by
  sorry

end extremum_increasing_decreasing_no_zeros_l120_120476


namespace find_a13_l120_120496

variable {a1 d : ℤ}

-- Definition of the arithmetic sequence
def arithmetic_seq (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Conditions
axiom h1 : a1 + (arithmetic_seq 9) = 16
axiom h2 : arithmetic_seq 4 = 1

-- Proof statement
theorem find_a13 : arithmetic_seq 13 = 64 :=
by
  sorry

end find_a13_l120_120496


namespace find_n_values_l120_120455

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def A_n_k (n k : ℕ) : ℕ := (10^n + 54 * 10^k - 1) / 9

def every_A_n_k_prime (n : ℕ) : Prop :=
  ∀ k, k < n → is_prime (A_n_k n k)

theorem find_n_values :
  ∀ n : ℕ, every_A_n_k_prime n → n = 1 ∨ n = 2 := sorry

end find_n_values_l120_120455


namespace modulus_of_z_l120_120142

-- Define the given complex number
def z : ℂ := (1 + 3 * complex.I) / (1 - 2 * complex.I)

-- Define the desired modulus value
def desired_modulus : ℝ := Real.sqrt 2

-- The proof goal
theorem modulus_of_z : complex.abs z = desired_modulus := 
by sorry

end modulus_of_z_l120_120142


namespace series1_converges_conditionally_series2_converges_conditionally_series3_converges_absolutely_series4_diverges_l120_120561

noncomputable theory

open Classical
open Filter BigOperators Topology

-- Series 1: Converges conditionally
theorem series1_converges_conditionally : ∃ (s : Summable) (n : ℕ), Summable (λ n : ℕ, (-1)^(n-1) * (1 / n)) ∧ ¬ Summable (λ n : ℕ, 1 / n) := sorry

-- Series 2: Converges conditionally
theorem series2_converges_conditionally : ∃ (s : Summable) (n : ℕ), Summable (λ n : ℕ, (-1)^(n-1) * (1 / (2 * n - 1))) ∧ ¬ Summable (λ n : ℕ, 1 / (2 * n - 1)) := sorry

-- Series 3: Converges absolutely
theorem series3_converges_absolutely (α : ℤ) : ∃ (s : Summable) (n : ℕ), Summable (λ n : ℕ, (sin (↑n * ↑α)) / n^2) ∧ Summable (λ n : ℕ, abs ((sin (↑n * ↑α)) / n^2)) := sorry

-- Series 4: Diverges
theorem series4_diverges : ¬∃ (s : Summable) (n : ℕ), Summable (λ n : ℕ, (-1)^(n-1) * ((n + 1) / n)) := sorry

end series1_converges_conditionally_series2_converges_conditionally_series3_converges_absolutely_series4_diverges_l120_120561


namespace percentage_change_coffee_in_july_is_100_l120_120635

section PercentageChangeCoffee

-- Defining the initial conditions
variables (G C : ℝ) (C' : ℝ) (P : ℝ)

-- Conditions and equalities from the problem statement
def green_tea_coffee_equal (G C : ℝ) : Prop := G = C
def green_tea_price_in_july (G : ℝ) : Prop := 0.3 * G = 0.3
def mixture_cost (G C' : ℝ) : Prop := 0.45 + 1.5 * C' = 3.45

-- Definition of percentage change formula
def percentage_change (C C' : ℝ) : ℝ := ((C' - C) / C) * 100

-- The main theorem statement
theorem percentage_change_coffee_in_july_is_100
  (G C C' : ℝ)
  (h1 : green_tea_coffee_equal G C)
  (h2 : green_tea_price_in_july G)
  (h3 : mixture_cost G C')
  : percentage_change C C' = 100 :=
sorry

end PercentageChangeCoffee

end percentage_change_coffee_in_july_is_100_l120_120635


namespace polyhedron_volume_l120_120164

def polyhedron_volume_condition (P Q R S T U V : Type) [Triangle P] [Triangle Q] [Triangle R] [Square S] [Square T] [Square U] [EquilateralTriangle V] 
  (side_length_square : ℝ) (isosceles_right_triangle_leg : ℝ) : Prop := 
  side_length_square = 2 ∧ isosceles_right_triangle_leg = 2

theorem polyhedron_volume (P Q R S T U V : Type) [Triangle P] [Triangle Q] [Triangle R] [Square S] [Square T] [Square U] [EquilateralTriangle V] 
  (cond : polyhedron_volume_condition P Q R S T U V 2 2) : 
  volume_of_polyhedron P Q R S T U V = (8 - 2 * Real.sqrt 2) / 3 :=
sorry

end polyhedron_volume_l120_120164


namespace area_of_triangle_DEF_l120_120724

noncomputable def triangle_area (r1 r2 : ℝ) (a b : ℝ) : ℝ :=
  let DG := real.sqrt (a^2 - r1^2) in
  let QF := (a + b + 3) / DG in
  1/2 * 2 * QF * 10

theorem area_of_triangle_DEF:
  ∀ (r1 r2 a b : ℝ), 
   r1 = 2 ∧ r2 = 3 ∧ a = 5 ∧ b = 5 →
   triangle_area r1 r2 a b = 130 * real.sqrt 21 / 21 :=
by
  intros r1 r2 a b h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  cases h6,
  simp [triangle_area, h1, h3, h4, h6],
  sorry

end area_of_triangle_DEF_l120_120724


namespace find_m_if_sin_cos_roots_l120_120872

open Real

theorem find_m_if_sin_cos_roots (θ m : ℝ) (h1 : IsRoot (λ x, 4*x^2 + 2*m*x + m) (sin θ)) 
  (h2 : IsRoot (λ x, 4*x^2 + 2*m*x + m) (cos θ)) : m = 1 - sqrt 5 := 
by
  sorry

end find_m_if_sin_cos_roots_l120_120872


namespace count_visible_factor_numbers_200_to_250_l120_120385

def is_visible_factor_number (n : ℕ) : Prop :=
  let digits := (toString n).toList
  let non_zero_digits := digits.filter (λ d => d ≠ '0')
  (non_zero_digits.all (λ d => n % (d.toNat - '0'.toNat) = 0))

def visible_factor_numbers_in_range (m n : ℕ) : List ℕ :=
  (List.range' m (n - m + 1)).filter is_visible_factor_number

theorem count_visible_factor_numbers_200_to_250 : 
  visible_factor_numbers_in_range 200 250 = List.range' 200 22 
:=
  sorry

end count_visible_factor_numbers_200_to_250_l120_120385


namespace circle_center_radius_and_tangent_line_l120_120492

theorem circle_center_radius_and_tangent_line (l : ℝ → ℝ → Prop)
  (A : ℝ × ℝ)
  (hA : A = (-6, 7))
  (C : ℝ → ℝ → Prop)
  (hC : ∀ x y, C x y ↔ x^2 + y^2 - 8*x + 6*y + 21 = 0)
  (tangent : ∀ x y, l x y → C x y ↔ (x-4)^2 + (y+3)^2 = 4 ∧ l x y ↔ dist (x, y) (4, -3) = 2) :
  (∃ c ∈ (ℝ × ℝ), c = (4, -3) ∧ (∀ r ∈ ℝ, r = 2) ∧ 
  ((∀ k ∈ ℝ, k = -3/4 ∨ k = -4/3 → 
  ∀ x y, l x y ↔ y - 7 = k * (x + 6)) → l = (λ x y, 3*x + 4*y - 10 = 0 ∨ 4*x + 3*y + 3 = 0))) :=
sorry

end circle_center_radius_and_tangent_line_l120_120492


namespace compute_b_div_a_l120_120529

theorem compute_b_div_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : (a + b * complex.I)^5 = (a - b * complex.I)^5) : b / a = real.sqrt 5 :=
sorry

end compute_b_div_a_l120_120529


namespace visibleFactorNumbersCount_l120_120386

-- Define what it means for a number to be a visible factor number
def isVisibleFactorNumber (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ d ∈ digits, d ≠ 0 → n % d = 0

-- Define the range of numbers from 200 to 250
def range200to250 := {n : ℕ | n >= 200 ∧ n <= 250}

-- The main theorem statement
theorem visibleFactorNumbersCount : 
  {n | n ∈ range200to250 ∧ isVisibleFactorNumber n}.card = 22 := 
by sorry

end visibleFactorNumbersCount_l120_120386


namespace transformed_area_l120_120511

variable (f : ℝ → ℝ)

def area_under_curve (f : ℝ → ℝ) : ℝ := ∫ x in 0..1, f x

theorem transformed_area (A : ℝ) (hf : ∫ x in 0..1, f x = A) :
  ∫ x in 0..4, 4 * f (2 * (x - 1)) = 4 * A :=
by
  sorry

end transformed_area_l120_120511


namespace kanul_total_amount_l120_120571

theorem kanul_total_amount (T : ℝ) (h1 : 500 + 400 + 0.10 * T = T) : T = 1000 :=
  sorry

end kanul_total_amount_l120_120571


namespace max_f_max_ab_plus_bc_l120_120853

def f (x : ℝ) := |x - 3| - 2 * |x + 1|

theorem max_f : ∃ (m : ℝ), m = 4 ∧ (∀ x : ℝ, f x ≤ m) := 
  sorry

theorem max_ab_plus_bc (a b c : ℝ) : a > 0 ∧ b > 0 → a^2 + 2 * b^2 + c^2 = 4 → (ab + bc) ≤ 2 :=
  sorry

end max_f_max_ab_plus_bc_l120_120853


namespace scientific_notation_population_l120_120241

theorem scientific_notation_population :
    ∃ (a b : ℝ), (b = 5 ∧ 1412.60 * 10 ^ 6 = a * 10 ^ b ∧ a = 1.4126) :=
sorry

end scientific_notation_population_l120_120241


namespace f_sin_x_eq_sin_17x_l120_120469

variable (f : ℝ → ℝ)

theorem f_sin_x_eq_sin_17x (h : ∀ x, f (cos x) = cos (17 * x)) : ∀ x, f (sin x) = sin (17 * x) :=
sorry

end f_sin_x_eq_sin_17x_l120_120469


namespace compare_negative_sqrt_values_l120_120771

theorem compare_negative_sqrt_values : -3 * Real.sqrt 3 > -2 * Real.sqrt 7 := 
sorry

end compare_negative_sqrt_values_l120_120771


namespace find_m_l120_120732

def point (α : Type) := (α × α)

noncomputable def slope {α : Type} [LinearOrderedField α] 
(p1 p2 : point α) : α := 
(p2.2 - p1.2) / (p2.1 - p1.1)

theorem find_m (m : ℚ) : 
  let p1 := (2, 5 : ℚ) in
  let p2 := (-3, m) in
  let p3 := (15, -1 : ℚ) in
  slope p1 p2 = slope p2 p3 → 
  m = 95 / 13 :=
by
  intros h
  sorry

end find_m_l120_120732


namespace number_of_permutations_l120_120210

theorem number_of_permutations (n : ℕ) (h : n > 1) :
  (∃ p : (list ℕ) → Prop, 
     (∀ (a : list ℕ), list.perm (list.range n) a → p a → 
        ∃ i, i ∈ list.range (n - 1) ∧ a.nth_le i sorry > a.nth_le (i + 1) sorry) ∧
      (∀ i, i ∈ list.range (n - 1) → 
        ∃ a : list ℕ, list.perm (list.range n) a ∧ p a ∧ a.nth_le i sorry > a.nth_le (i + 1) sorry) ∧
      ∀ j, j ∈ list.range (n - 1) → 
        ∃! (a : list ℕ), list.perm (list.range n) a ∧ p a ∧ a.nth_le j sorry > a.nth_le (j + 1) sorry
  ) :=
  2^n - n - 1 := sorry

end number_of_permutations_l120_120210


namespace mass_percentage_H_equals_4_84_l120_120001

-- Define the given condition: mass percentage of H is 4.84%
def mass_percentage_H (mass_H total_mass_compound : ℝ) : ℝ :=
  (mass_H / total_mass_compound) * 100

-- Predicate to represent the given statement that the mass percentage is 4.84%
def mass_percentage_4_84 (mass_H total_mass_compound : ℝ) : Prop :=
  mass_percentage_H mass_H total_mass_compound = 4.84

-- Proof statement that mass percentage of H in the compound equals 4.84%
theorem mass_percentage_H_equals_4_84
  (mass_H total_mass_compound : ℝ) (h_given : mass_percentage_H mass_H total_mass_compound = 4.84) :
  mass_percentage_4_84 mass_H total_mass_compound :=
by
  exact h_given
  sorry

end mass_percentage_H_equals_4_84_l120_120001


namespace rounding_example_l120_120720

theorem rounding_example :
  (Real.round (8.095 * 100) / 100 = 8.10) ∧ (Int.round 8.095 = 8) :=
by
  sorry

end rounding_example_l120_120720


namespace collinear_vectors_triangle_C_sum_of_sides_l120_120840

theorem collinear_vectors_triangle_C
  (A B C a b c : ℝ)
  (habc : a, b, c)
  (h1 : (2*b - a) * cos C = c * cos A)
  (h2 : a / sin A = b / sin B = c / sin C)
  (h3 : sin (A + C) = sin B) :
  C = π / 3 :=
sorry

theorem sum_of_sides
  (A B C a b c : ℝ)
  (hC : C = π / 3)
  (hc : c = sqrt 3)
  (harea : (1/2) * a * b * sin (π / 3) = sqrt 3 / 2) :
  a + b = 3 :=
sorry

end collinear_vectors_triangle_C_sum_of_sides_l120_120840


namespace required_tiles_0_4m_l120_120247

-- Defining given conditions
def num_tiles_0_3m : ℕ := 720
def side_length_0_3m : ℝ := 0.3
def side_length_0_4m : ℝ := 0.4

-- The problem statement translated to Lean 4
theorem required_tiles_0_4m : (side_length_0_4m ^ 2) * (405 : ℝ) = (side_length_0_3m ^ 2) * (num_tiles_0_3m : ℝ) := 
by
  -- Skipping the proof
  sorry

end required_tiles_0_4m_l120_120247


namespace Kelly_games_given_away_l120_120572

theorem Kelly_games_given_away : ∀ (total_games games_left : ℕ), 
  total_games = 50 → games_left = 35 → total_games - games_left = 15 :=
by
  intros total_games games_left h1 h2
  rw [h1, h2]
  sorry

end Kelly_games_given_away_l120_120572


namespace brennan_second_round_files_l120_120422

theorem brennan_second_round_files :
  ∀ (initial_files second_round_total valuable_files_first valuable_files_second : ℕ)
    (deleted_ratio_first deleted_ratio_second : ℚ),
  initial_files = 800 →
  deleted_ratio_first = 0.70 →
  deleted_ratio_second = 3 / 5 →
  valuable_files_first = initial_files * (1 - deleted_ratio_first) →
  valuable_files_second = 400 - valuable_files_first →
  valuable_files_second = second_round_total * (1 - deleted_ratio_second) →
  second_round_total = 400 :=
begin
  intros initial_files second_round_total valuable_files_first valuable_files_second deleted_ratio_first deleted_ratio_second,
  rintros rfl rfl rfl rfl rfl,
  sorry
end

end brennan_second_round_files_l120_120422


namespace distance_foci_to_line_l120_120180

noncomputable def ellipse_equation (rho theta : ℝ) : Prop :=
  rho^2 = 12 / (3 * (Real.cos theta)^2 + 4 * (Real.sin theta)^2)

def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 + Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

def line_standard (x y : ℝ) : Prop :=
  x - y - 2 = 0

def ellipse_cartesian (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

def point_to_line_distance (x y a b c : ℝ) : ℝ :=
  Real.abs (a*x + b*y + c) / Real.sqrt (a^2 + b^2)

def sum_of_distances_to_line (F1 F2 : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  point_to_line_distance F1.1 F1.2 a b c + point_to_line_distance F2.1 F2.2 a b c

theorem distance_foci_to_line :
  let f1 := (-1 : ℝ, 0 : ℝ)
  let f2 := (1 : ℝ, 0 : ℝ)
  let a := 1
  let b := -1
  let c := -2
  sum_of_distances_to_line f1 f2 a b c = 2 * Real.sqrt 2 :=
by
  sorry

end distance_foci_to_line_l120_120180


namespace Ryan_dig_time_alone_l120_120336

theorem Ryan_dig_time_alone :
  ∃ R : ℝ, ∀ Castel_time together_time,
    Castel_time = 6 ∧ together_time = 30 / 11 →
    (1 / R + 1 / Castel_time = 11 / 30) →
    R = 5 :=
by 
  sorry

end Ryan_dig_time_alone_l120_120336


namespace rational_numbers_are_integers_l120_120276

theorem rational_numbers_are_integers (a b : ℝ) (h₀ : a ≠ b)
  (h₁ : ∀ n : ℕ, a ^ n - b ^ n ∈ ℤ) : a ∈ ℤ ∧ b ∈ ℤ :=
sorry

end rational_numbers_are_integers_l120_120276


namespace horner_poly_evaluation_l120_120425

def poly (x : ℝ) : ℝ := 3 * x^6 + 4 * x^5 - 5 * x^4 - 6 * x^3 + 7 * x^2 - 8 * x + 1

theorem horner_poly_evaluation (x : ℝ) : x = 0.4 →
  ((((((3 * x + 4) * x - 5) * x - 6) * x + 7) * x - 8) * x + 1 = poly x
  ∧ ((6, 6) = (6, 6)) :=
by
  intro h
  simp [h, poly]
  sorry

end horner_poly_evaluation_l120_120425


namespace gray_area_l120_120174

theorem gray_area (r : ℝ) (h1 : 4 = 3 * r - r) : 
  let outer_area := real.pi * (3 * r) ^ 2,
      inner_area := real.pi * r ^ 2,
      gray_region_area := outer_area - inner_area
  in
  gray_region_area = 32 * real.pi :=
by sorry

end gray_area_l120_120174


namespace combined_surface_area_hemisphere_l120_120485

theorem combined_surface_area_hemisphere (t : ℝ) : 
  let r := Real.sqrt (3 / Real.pi) in
  let exterior_surface_area := 6 in
  let interior_surface_area := 2 * Real.pi * (r - t) ^ 2 in
  let base_area := 3 in
  let W := exterior_surface_area + interior_surface_area + base_area in
  W = 12 - 4 * t * Real.sqrt (3 / Real.pi) + 2 * Real.pi * t ^ 2 :=
by
  let r := Real.sqrt (3 / Real.pi)
  let exterior_surface_area := 6
  let interior_surface_area := 2 * Real.pi * (r - t) ^ 2
  let base_area := 3
  let W := exterior_surface_area + interior_surface_area + base_area
  calc
    W = exterior_surface_area + interior_surface_area + base_area := rfl
    ... = 6 + (2 * Real.pi * (r - t) ^ 2) + 3 := by simp [exterior_surface_area, base_area]
    ... = 6 + (6 - 4 * t * Real.sqrt (3 / Real.pi) + 2 * Real.pi * t ^ 2) + 3 := by sorry
    ... = 12 - 4 * t * Real.sqrt (3 / Real.pi) + 2 * Real.pi * t ^ 2 := by sorry

end combined_surface_area_hemisphere_l120_120485


namespace identity_proof_l120_120615

theorem identity_proof
  (M N x a b : ℝ)
  (h₀ : x ≠ a)
  (h₁ : x ≠ b)
  (h₂ : a ≠ b) :
  (Mx + N) / ((x - a) * (x - b)) =
  (((M *a + N) / (a - b)) * (1 / (x - a))) - 
  (((M * b + N) / (a - b)) * (1 / (x - b))) :=
sorry

end identity_proof_l120_120615


namespace athlete_stable_performance_l120_120285

theorem athlete_stable_performance 
  (A_var : ℝ) (B_var : ℝ) (C_var : ℝ) (D_var : ℝ)
  (avg_score : ℝ)
  (hA_var : A_var = 0.019)
  (hB_var : B_var = 0.021)
  (hC_var : C_var = 0.020)
  (hD_var : D_var = 0.022)
  (havg : avg_score = 13.2) :
  A_var < B_var ∧ A_var < C_var ∧ A_var < D_var :=
by {
  sorry
}

end athlete_stable_performance_l120_120285


namespace solve_equation_roots_count_l120_120117

theorem solve_equation_roots_count :
  ∀ (x : ℝ), abs x ≤ real.sqrt 14 → 
  (sqrt (14 - x ^ 2) * (sin x - cos (2 * x)) = 0) → 
  (set.count {x | abs x ≤ real.sqrt 14 ∧ (sqrt (14 - x ^ 2) * (sin x - cos (2 * x)) = 0)} = 6) :=
by sorry

end solve_equation_roots_count_l120_120117


namespace sum_zero_and_positive_union_l120_120928

open Finset
open Nat

theorem sum_zero_and_positive_union 
  (A : Finset ℝ)
  (A_subsets : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → Finset ℝ)
  (sum_A_zero : (∑ x in A, x) = 0)
  (positive_sum : ∀ (xvec : ℕ → ℝ) (h : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → xvec i ∈ A_subsets i), (∑ i in (range n).filter (λ i, 1 ≤ i ∧ i ≤ n), xvec i) > 0) :
  
  ∃ (k : ℕ) (i_seq : Finset ℕ), 
    1 ≤ k ∧ k ≤ n ∧
    (∀ (i : ℕ), i ∈ i_seq → 1 ≤ i ∧ i ≤ n) ∧
    ((big_union A_subsets i_seq).card < (k * A.card) / n) :=
begin
  sorry
end

end sum_zero_and_positive_union_l120_120928


namespace max_disrespectful_polynomial_value_l120_120741

noncomputable def disrespectful (p : ℝ → ℝ) :=
  (∀ x, p(x) = x^2 + a * x + b) ∧
  (p ∘ p).rootSet.count = 3

noncomputable def unique_max_polynomial (p: ℝ → ℝ) :=
  disrespectful p ∧
  ∀ q, disrespectful q → p.rootSum ≥ q.rootSum

theorem max_disrespectful_polynomial_value :
  (∃ p: ℝ → ℝ, unique_max_polynomial p ∧ (p 1 = 5 / 16)) :=
begin
  sorry
end

end max_disrespectful_polynomial_value_l120_120741


namespace roots_of_equation_l120_120121

theorem roots_of_equation : 
  (∃ s ∈ ([- real.sqrt 14, real.sqrt 14]).to_set, 
  (∀ x ∈ s, (real.sqrt (14 - x^2)) * (real.sin x - real.cos (2 * x)) = 0) ∧ 
  ( set.analyse (eq (set.card s) 6))) :=
sorry

end roots_of_equation_l120_120121


namespace xy_yz_zx_over_x2_y2_z2_l120_120590

theorem xy_yz_zx_over_x2_y2_z2 (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h_sum : x + y + z = 1) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by
  sorry

end xy_yz_zx_over_x2_y2_z2_l120_120590


namespace cos_theta_planes_l120_120204

def normal_vector_plane_1 : ℝ × ℝ × ℝ := (3, -1, 2)
def normal_vector_plane_2 : ℝ × ℝ × ℝ := (9, -3, -6)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2).sqrt

def cos_theta (n1 n2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product n1 n2 / (magnitude n1 * magnitude n2)

theorem cos_theta_planes :
  cos_theta normal_vector_plane_1 normal_vector_plane_2 = 3 / 7 :=
by
  sorry

end cos_theta_planes_l120_120204


namespace part1_monotonicity_part2_range_of_a_l120_120030

-- Part (1)
def f1 (x : ℝ) : ℝ := 8 * x - (Real.sin x / (Real.cos x)^3)

theorem part1_monotonicity (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) :
  (0 < x ∧ x < Real.pi / 4 → Deriv f1 x > 0) ∧ (Real.pi / 4 < x ∧ x < Real.pi / 2 → Deriv f1 x < 0) :=
sorry

-- Part (2)
def f2 (a : ℝ) (x : ℝ) : ℝ := a * x - (Real.sin x / (Real.cos x)^3)

theorem part2_range_of_a (a : ℝ) (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) :
  (∀ x, f2 a x < Real.sin (2 * x)) ↔ a ≤ 3 :=
sorry

end part1_monotonicity_part2_range_of_a_l120_120030


namespace adrien_winning_strategy_l120_120607

/--
On the table, there are 2023 tokens. Adrien and Iris take turns removing at least one token and at most half of the remaining tokens at the time they play. The player who leaves a single token on the table loses the game. Adrien starts first. Prove that Adrien has a winning strategy.
-/
theorem adrien_winning_strategy : ∃ strategy : ℕ → ℕ, 
  ∀ n:ℕ, (n = 2023 ∧ 1 ≤ strategy n ∧ strategy n ≤ n / 2) → 
    (∀ u : ℕ, (u = n - strategy n) → (∃ strategy' : ℕ → ℕ , 
      ∀ m:ℕ, (m = u ∧ 1 ≤ strategy' m ∧ strategy' m ≤ m / 2) → 
        (∃ next_u : ℕ, (next_u = m - strategy' m → next_u ≠ 1 ∨ (m = 1 ∧ u ≠ 1 ∧ next_u = 1)))))
:= sorry

end adrien_winning_strategy_l120_120607


namespace WidgetsPerHour_l120_120186

theorem WidgetsPerHour 
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (widgets_per_week : ℕ) 
  (H1 : hours_per_day = 8)
  (H2 : days_per_week = 5)
  (H3 : widgets_per_week = 800) : 
  widgets_per_week / (hours_per_day * days_per_week) = 20 := 
sorry

end WidgetsPerHour_l120_120186


namespace median_of_sample_l120_120744

variables {x1 x2 x3 x4 x5 : ℝ}

-- Given conditions
axiom h_order : x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < x5
axiom h_less_than_minus_2 : x1 < -2 ∧ x2 < -2 ∧ x3 < -2 ∧ x4 < -2 ∧ x5 < -2

-- Question to prove
theorem median_of_sample :
  let sample := [2, -x1, x2, x3, -x4, x5].sort in
  sample.nth_le 2 (by dec_trivial) + sample.nth_le 3 (by dec_trivial) = 2 * ((2 + x5) / 2) :=
sorry

end median_of_sample_l120_120744


namespace smallest_a_l120_120080

noncomputable def f : ℝ → ℝ := sorry

theorem smallest_a :
  (∀ x : ℝ, 0 < x → f (2 * x) = 2 * f x) →
  (∀ x : ℝ, 1 < x ∧ x ≤ 2 → f x = 2 - x) →
  (f(2012) = f(92)) →
  (∀ y : ℝ, f y = f(2012) → 0 < y → 92 ≤ y) :=
sorry

end smallest_a_l120_120080


namespace triangle_CPM_equilateral_l120_120976

-- Define points and properties of the problem
variables {Point : Type*}
variables [metric_space Point]

structure EquilateralTriangle (A B C : Point) : Prop :=
(eq1 : dist A B = dist B C)
(eq2 : dist B C = dist C A)

structure Midpoint (M A B : Point) : Prop :=
(mp : dist A M = dist M B)

-- Given conditions
variables {A B C D E M P : Point}
variables (equiABC : EquilateralTriangle A B C)
variables (equiCDE : EquilateralTriangle C D E)
variables (midpointM : Midpoint M A D)
variables (midpointP : Midpoint P B E)

-- Goal: Prove that triangle CPM is equilateral
theorem triangle_CPM_equilateral : EquilateralTriangle C P M :=
sorry

end triangle_CPM_equilateral_l120_120976


namespace field_perimeter_l120_120674

noncomputable def outer_perimeter (posts : ℕ) (post_width_inches : ℝ) (spacing_feet : ℝ) : ℝ :=
  let posts_per_side := posts / 4
  let gaps_per_side := posts_per_side - 1
  let post_width_feet := post_width_inches / 12
  let side_length := gaps_per_side * spacing_feet + posts_per_side * post_width_feet
  4 * side_length

theorem field_perimeter : 
  outer_perimeter 32 5 4 = 125 + 1/3 := 
by
  sorry

end field_perimeter_l120_120674


namespace ben_chairs_in_10_days_l120_120758

def number_of_chairs (days hours_per_shift hours_rocking_chair hours_dining_chair hours_armchair : ℕ) : ℕ × ℕ × ℕ :=
  let rocking_chairs_per_day := hours_per_shift / hours_rocking_chair
  let remaining_hours_after_rocking_chairs := hours_per_shift % hours_rocking_chair
  let dining_chairs_per_day := remaining_hours_after_rocking_chairs / hours_dining_chair
  let remaining_hours_after_dining_chairs := remaining_hours_after_rocking_chairs % hours_dining_chair
  if remaining_hours_after_dining_chairs >= hours_armchair then
    (days * rocking_chairs_per_day, days * dining_chairs_per_day, days * (remaining_hours_after_dining_chairs / hours_armchair))
  else
    (days * rocking_chairs_per_day, days * dining_chairs_per_day, 0)

theorem ben_chairs_in_10_days :
  number_of_chairs 10 8 5 3 6 = (10, 10, 0) :=
by 
  sorry

end ben_chairs_in_10_days_l120_120758


namespace selection_positional_distinct_l120_120398

-- Define the conditions based on the problem.
def candidates : ℕ := 12
def ways_to_select : ℕ := 132

-- Define positional selection bijection based on permutation.
def positional_selection (n r : ℕ) : ℕ := n * (n - r + 1)

-- The main statement that captures the proof.
theorem selection_positional_distinct :
  ways_to_select = positional_selection candidates 2 :=
by
  simp [candidates, ways_to_select, positional_selection]
  sorry

end selection_positional_distinct_l120_120398


namespace valid_pairs_count_l120_120930

def S : Set ℕ := {1, 22, 333, 4444, 55555, 666666, 7777777, 88888888, 999999999}

noncomputable def num_valid_pairs : ℕ :=
  Set.toFinset {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2 ∧ p.1 ∣ p.2}.card

theorem valid_pairs_count : num_valid_pairs = 14 := 
  sorry

end valid_pairs_count_l120_120930


namespace slope_of_line_with_45_deg_inclination_l120_120293

theorem slope_of_line_with_45_deg_inclination (l : ℝ) (h : l = 45) : real.tan l = 1 :=
by sorry

end slope_of_line_with_45_deg_inclination_l120_120293


namespace max_sum_at_n_eq_12_l120_120548

-- Definition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Definition of the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

-- Given constants and conditions
constants {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}
axiom a_1_positive : a 1 > 0
axiom seq_condition : 3 * (a 1 + 4 * d) = 5 * (a 1 + 7 * d)

-- Theorem stating that the sum S_n is maximized at n=12
theorem max_sum_at_n_eq_12 : (∃ n, S n = S 12) :=
  sorry

end max_sum_at_n_eq_12_l120_120548


namespace hyperbola_asymptote_equation_l120_120509

theorem hyperbola_asymptote_equation (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b)
  (F₁ F₂ : ℝ × ℝ) (P : ℝ × ℝ)
  (h₃ : F₁ = (-c, 0)) (h₄ : F₂ = (c, 0))
  (h₅ : P = (a^2 + c^2) / (2 * c), (b / a) * ((a^2 + c^2) / (2 * c) - c))
  (h₆ : ((a^2 + c^2) / (2*c) - c)^2 / a^2 >  (-(b/a) * ((a^2 + c^2) / (2 * c) - c))^2 / b^2) :
  b^2 = 4 * a^2 → asymptote_eqn (a b : ℝ) (asym_eqn : String) :=
begin
  sorry
end

end hyperbola_asymptote_equation_l120_120509


namespace solution_set_x_fx_neg_l120_120598

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
∀ (x : ℝ), f (-x) = -f (x)

noncomputable def is_increasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ (x y : ℝ), x ∈ s → y ∈ s → x < y → f (x) < f (y)

theorem solution_set_x_fx_neg (f : ℝ → ℝ) (s := set.Ioi 0) (a := -3) :
  is_odd f →
  is_increasing f (set.Ioi 0) →
  f (-3) = 0 →
  {x : ℝ | x * f x < 0} = set.Ioo (-3 : ℝ) 0 ∪ set.Ioo (0 : ℝ) 3 := 
by
  intros h1 h2 h3
  sorry

end solution_set_x_fx_neg_l120_120598


namespace paint_needed_for_new_statues_l120_120528

-- Conditions
def pint_for_original : ℕ := 1
def original_height : ℕ := 8
def num_statues : ℕ := 320
def new_height : ℕ := 2
def scale_ratio : ℚ := (new_height : ℚ) / (original_height : ℚ)
def area_ratio : ℚ := scale_ratio ^ 2

-- Correct Answer
def total_paint_needed : ℕ := 20

-- Theorem to be proved
theorem paint_needed_for_new_statues :
  pint_for_original * num_statues * area_ratio = total_paint_needed := 
by
  sorry

end paint_needed_for_new_statues_l120_120528


namespace general_form_of_curve_range_of_x_plus_y_l120_120075

noncomputable def parametric_curve (φ : ℝ) : ℝ × ℝ :=
  (4 * Real.cos φ, 3 * Real.sin φ)

theorem general_form_of_curve :
  ∀ φ : ℝ, let (x, y) := parametric_curve φ in
  (x^2 / 16) + (y^2 / 9) = 1 := by
  sorry

theorem range_of_x_plus_y :
  ∀ φ : ℝ, let (x, y) := parametric_curve φ in
  -5 ≤ x + y ∧ x + y ≤ 5 := by
  sorry

end general_form_of_curve_range_of_x_plus_y_l120_120075


namespace AP_perpendicular_BC_l120_120684

-- Definitions corresponding to the conditions in a).
def triangle (A B C : Point) : Prop := sorry
def hyperbola (h : ℝ → ℝ → Prop) : Prop := 
  ∃ a b : ℝ, ∀ x y, h x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1
def point_on_line_extended (P A B : Point) : Prop := sorry
def tangent_to_hyperbola (h : ℝ → ℝ → Prop) (P : Point) (l : Point → Prop) : Prop := sorry
def intersect_at (l1 l2 : Point → Prop) (P : Point) : Prop := sorry

variables (A B C E F P : Point)
variables (h : ℝ → ℝ → Prop)
variables (tangent_E tangent_F : Point → Prop)

-- Conditions
axiom triangle_ABC : triangle A B C
axiom BC_real_axis_hyperbola : ∃ h, hyperbola h ∧ ∀ P, h P.x 0 = false
axiom E_on_extension_AB : point_on_line_extended E A B
axiom F_on_extension_AC : point_on_line_extended F A C
axiom tangents_E_F_intersect_P : tangent_to_hyperbola h E tangent_E ∧
                                tangent_to_hyperbola h F tangent_F ∧
                                intersect_at tangent_E tangent_F P

-- Question Rewrite
theorem AP_perpendicular_BC :
  ∃ l1 l2 : Point → Prop, intersect_at l1 l2 A ∧ (l1 P = true ∧ l2 B = true) → 
  (intersection_at P B C) :
  (triangle A B C) ∧ (intersection_at h B C = intersect_A_at h P):
  sorry

end AP_perpendicular_BC_l120_120684


namespace number_of_distinct_products_l120_120098

noncomputable def givenSet : Set ℕ := {1, 2, 4, 7, 13}

theorem number_of_distinct_products : 
  {n : ℕ | ∃ (xs : Finset ℕ), (∀ x ∈ xs, x ∈ givenSet) ∧ 2 ≤ xs.card ∧ n = xs.prod id}.to_finset.card = 11 := 
by 
  sorry

end number_of_distinct_products_l120_120098


namespace handshake_remainder_l120_120894

theorem handshake_remainder (M : ℕ) (h1 : ∀ p, p < 8 → (number_of_shakes p = 2)) : 
  M = 5355 → M % 1000 = 355 :=
by
  sorry

end handshake_remainder_l120_120894


namespace trapezoid_area_l120_120792

def trapezoid_diagonals_and_height (AC BD h : ℕ) :=
  (AC = 17) ∧ (BD = 113) ∧ (h = 15)

theorem trapezoid_area (AC BD h : ℕ) (area1 area2 : ℕ) 
  (H : trapezoid_diagonals_and_height AC BD h) :
  (area1 = 900 ∨ area2 = 780) :=
by
  sorry

end trapezoid_area_l120_120792


namespace ball_total_distance_after_five_bounces_l120_120400

theorem ball_total_distance_after_five_bounces
  (initial_height : ℕ)
  (rebound_ratio : ℚ)
  (h_initial : initial_height = 80)
  (h_rebound : rebound_ratio = 2 / 3) :
  let total_distance := initial_height +
    2 * (initial_height * (rebound_ratio ^ 1 + rebound_ratio ^ 2 + rebound_ratio ^ 3 + rebound_ratio ^ 4 + rebound_ratio ^ 5)) in
  total_distance = 11280 / 81 := sorry

end ball_total_distance_after_five_bounces_l120_120400


namespace number_of_distinct_15_ominoes_l120_120776

/--
Definition of an n-omino: A subset of an infinite grid consisting of n connected unit squares.
-/
def is_n_omino (n : ℕ) (squares : set (ℤ × ℤ)) : Prop :=
  ∃(conn : ℤ × ℤ → ℤ × ℤ → Prop)[decidable_rel conn], 
  finite squares ∧ card squares = n ∧ connected squares conn

/--
Two n-ominoes are equivalent if one can be obtained from the other by translations and rotations.
-/
def equivalent_n_ominoes (n : ℕ) (s1 s2 : set (ℤ × ℤ)) : Prop :=
  ∃ t : ℤ × ℤ, ∃ r : ℤ × ℤ → ℤ × ℤ,
  is_translation t ∧ is_rotation r ∧ (image r (image (λ p, (p.1 + t.1, p.2 + t.2)) s1)) = s2

/--
Number of distinct 15-ominoes in an infinite grid.
-/
theorem number_of_distinct_15_ominoes : 
  {ominoes | is_n_omino 15 ominoes}.quotient (equivalent_n_ominoes 15) = 3426576 :=
sorry

end number_of_distinct_15_ominoes_l120_120776


namespace focus_of_ellipse_l120_120416

-- Definitions from conditions
def major_axis_endpoints := (1, -2) ∧ (7, -2)
def minor_axis_endpoints := (3, 1) ∧ (3, -5)
def center_of_ellipse := (3, -2)

-- Proof problem
theorem focus_of_ellipse :
  (compute_focus_x_coord major_axis_endpoints minor_axis_endpoints = 3) ∧ 
  (compute_focus_y_coord major_axis_endpoints minor_axis_endpoints = -2) :=
sorry

end focus_of_ellipse_l120_120416


namespace scientific_notation_population_l120_120239

theorem scientific_notation_population :
    ∃ (a b : ℝ), (b = 5 ∧ 1412.60 * 10 ^ 6 = a * 10 ^ b ∧ a = 1.4126) :=
sorry

end scientific_notation_population_l120_120239


namespace how_many_three_digit_numbers_without_5s_and_8s_l120_120129

def is_valid_hundreds_digit (d : ℕ) : Prop := d ≠ 0 ∧ d ≠ 5 ∧ d ≠ 8
def is_valid_digit (d : ℕ) : Prop := d ≠ 5 ∧ d ≠ 8

theorem how_many_three_digit_numbers_without_5s_and_8s : 
  (∃ count : ℕ, count = 
    (∑ d1 in (finset.range 10).filter is_valid_hundreds_digit, 
      ∑ d2 in (finset.range 10).filter is_valid_digit, 
        ∑ d3 in (finset.range 10).filter is_valid_digit, 1)) = 448 :=
by
  sorry

end how_many_three_digit_numbers_without_5s_and_8s_l120_120129


namespace Kyler_wins_1_game_l120_120611

theorem Kyler_wins_1_game
  (peter_wins : ℕ)
  (peter_losses : ℕ)
  (emma_wins : ℕ)
  (emma_losses : ℕ)
  (kyler_losses : ℕ)
  (total_games : ℕ)
  (kyler_wins : ℕ)
  (htotal : total_games = (peter_wins + peter_losses + emma_wins + emma_losses + kyler_wins + kyler_losses) / 2)
  (hpeter : peter_wins = 4 ∧ peter_losses = 2)
  (hemma : emma_wins = 3 ∧ emma_losses = 3)
  (hkyler_losses : kyler_losses = 3)
  (htotal_wins_losses : total_games = peter_wins + emma_wins + kyler_wins) : kyler_wins = 1 :=
by
  sorry

end Kyler_wins_1_game_l120_120611


namespace max_area_rectangle_l120_120751

-- Define the conditions using Lean
def is_rectangle (length width : ℕ) : Prop :=
  2 * (length + width) = 34

-- Define the problem as a theorem in Lean
theorem max_area_rectangle : ∃ (length width : ℕ), is_rectangle length width ∧ length * width = 72 :=
by
  sorry

end max_area_rectangle_l120_120751


namespace digit_57_of_1_over_13_l120_120691

/-- 
  Prove that the 57th digit after the decimal point of the decimal representation of 1 / 13 
  is 6, given the repeating block of the decimal representation being "076923" with a period of 6 digits.
-/
theorem digit_57_of_1_over_13 :
  let repeating_seq := [0, 7, 6, 9, 2, 3]
  (repeating_seq.cyclic_nth 56) = 6 :=
by
  sorry

namespace Array

def cyclic_nth {α} [Inhabited α] (a : Array α) (n : Nat) : α :=
  a.get! (n % a.size)

end Array

end digit_57_of_1_over_13_l120_120691


namespace number_of_roots_l120_120102

noncomputable theory
open Real

def domain (x : ℝ) := abs x ≤ sqrt 14
def equation (x : ℝ) := sin x - cos (2 * x) = 0

theorem number_of_roots : 
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, domain x) ∧ (∀ x ∈ xs, equation x) ∧ xs.card = 6 := 
by sorry

end number_of_roots_l120_120102


namespace find_p_max_area_PAB_l120_120087

-- Given Problem Conditions
def parabola (p : ℝ) := { pt : ℝ × ℝ // pt.1^2 = 2 * p * pt.2 }
def circle := { pt : ℝ × ℝ // pt.1^2 + (pt.2 + 4)^2 = 1 }
def focus (p : ℝ) := (0, p / 2)

-- Part (1): Proof to find p
theorem find_p (dist : ℝ) (h1 : dist = 4) (p : ℝ) (h2 : p > 0) 
  (h3 : ∀ (F : ℝ × ℝ) (M : ℝ × ℝ), F = focus p → M ∈ circle → abs (dist_of_points F M) - 1 = dist) : p = 2 := 
sorry

-- Part (2): Proof for maximum area of ∆PAB
theorem max_area_PAB (p : ℝ) (hp : p = 2)
  (P : ℝ × ℝ) (hP : P ∈ circle) 
  (A B : ℝ × ℝ) (hA : A ∈ parabola p) (hB : B ∈ parabola p)
  (PA_tangent : is_tangent P A (parabola p))
  (PB_tangent : is_tangent P B (parabola p)) : 
  ∃ (max_area : ℝ), max_area = 20 * real.sqrt 5 :=
sorry

-- Helper Definitions (may need to be defined precisely for real implementation)
noncomputable def dist_of_points (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def is_tangent (P A : ℝ × ℝ) (para : parabola) : Prop := sorry
noncomputable def is_tangent (P B : ℝ × ℝ) (para : parabola) : Prop := sorry

end find_p_max_area_PAB_l120_120087


namespace crayons_count_l120_120246

theorem crayons_count 
  (initial_crayons erasers : ℕ) 
  (erasers_count end_crayons : ℕ) 
  (initial_erasers : erasers = 38) 
  (end_crayons_more_erasers : end_crayons = erasers + 353) : 
  initial_crayons = end_crayons := 
by 
  sorry

end crayons_count_l120_120246


namespace regression_slope_neg1_l120_120299

variables (x y : Real)

def regression_eq : y = 2 - x := sorry

theorem regression_slope_neg1 (h : y = 2 - x) : 
  (y = 2 - (x + 1)) → (y = (2 - x) - 1) :=
begin
  intros h1,
  sorry
end

end regression_slope_neg1_l120_120299


namespace angle_of_inclination_of_x_eq_3_l120_120632

theorem angle_of_inclination_of_x_eq_3 : 
  angle_of_inclination (line_eq_of_x_eq_const 3) = 90 :=
by sorry

def line_eq_of_x_eq_const (c : ℝ) : line := Line.mk c
def angle_of_inclination (l : line) : ℝ := 
  match l with
  | Line.mk _ => 90 -- Since any line of the form x = const is vertical.

end angle_of_inclination_of_x_eq_3_l120_120632


namespace min_marked_cells_l120_120333

-- Define the dimensions of the grid
def grid_width : ℕ := 50
def grid_height : ℕ := 50
def strip_width : ℕ := 6

-- Define the total number of strips
def total_strips : ℕ := (grid_width * (grid_height / strip_width)) + (grid_height * (grid_width / strip_width))

-- Statement of the theorem
theorem min_marked_cells : total_strips = 416 :=
by
  Sorry -- Proof goes here 

end min_marked_cells_l120_120333


namespace monotonicity_f_8_range_a_condition_l120_120024

def f (a : ℝ) (x : ℝ) : ℝ := a * x - sin x / cos x ^ 3

-- Prove that f(x) is increasing on (0, π/4) and decreasing on (π/4, π/2) when a = 8.
theorem monotonicity_f_8 (x : ℝ) (h : 0 < x ∧ x < π / 2) (hf : f 8 x = 8 * x - sin x / cos x ^ 3) :
  (0 < x ∧ x < π / 4 → deriv (f 8) x > 0) ∧ (π / 4 < x ∧ x < π / 2 → deriv (f 8) x < 0) :=
sorry

-- Prove that ∀ x ∈ (0, π/2), f(x) < sin 2x ↔ a ≤ 3.
theorem range_a_condition (a : ℝ) (h : ∀ x, 0 < x ∧ x < π / 2 → f a x < sin (2 * x)) :
  a ≤ 3 :=
sorry

end monotonicity_f_8_range_a_condition_l120_120024


namespace amount_daria_needs_l120_120441

theorem amount_daria_needs (ticket_cost : ℕ) (total_tickets : ℕ) (current_money : ℕ) (needed_money : ℕ) 
  (h1 : ticket_cost = 90) 
  (h2 : total_tickets = 4) 
  (h3 : current_money = 189) 
  (h4 : needed_money = 360 - 189) 
  : needed_money = 171 := by
  -- proof omitted
  sorry

end amount_daria_needs_l120_120441


namespace number_of_student_tickets_sold_l120_120316

variable (A S : ℝ)

theorem number_of_student_tickets_sold
  (h1 : A + S = 59)
  (h2 : 4 * A + 2.5 * S = 222.50) :
  S = 9 :=
by sorry

end number_of_student_tickets_sold_l120_120316


namespace problem_l120_120833

noncomputable def cos_angle_POQ (O P Q : ℝ × ℝ) : ℝ :=
  let cos_xOP := 4 / 5 in
  let sin_xOP := 3 / 5 in
  let cos_xOQ := 5 / 13 in
  let sin_xOQ := -12 / 13 in
  cos_xOP * cos_xOQ - sin_xOP * sin_xOQ

theorem problem 
  (O P Q : ℝ × ℝ)
  (P_on_circle : P.1^2 + P.2^2 = 1)
  (Q_on_circle : Q.1^2 + Q.2^2 = 1)
  (P_first_quadrant : 0 ≤ P.1 ∧ 0 ≤ P.2)
  (Q_fourth_quadrant : 0 ≤ Q.1 ∧ Q.2 ≤ 0)
  (P_x : P.1 = 4 / 5)
  (Q_x : Q.1 = 5 / 13) :
  cos_angle_POQ O P Q = 56 / 65 := sorry

end problem_l120_120833


namespace total_birds_count_l120_120389

def cage1_parrots := 9
def cage1_finches := 4
def cage1_canaries := 7

def cage2_parrots := 5
def cage2_parakeets := 8
def cage2_finches := 10

def cage3_parakeets := 15
def cage3_finches := 7
def cage3_canaries := 3

def cage4_parrots := 10
def cage4_parakeets := 5
def cage4_finches := 12

def total_birds := cage1_parrots + cage1_finches + cage1_canaries +
                   cage2_parrots + cage2_parakeets + cage2_finches +
                   cage3_parakeets + cage3_finches + cage3_canaries +
                   cage4_parrots + cage4_parakeets + cage4_finches

theorem total_birds_count : total_birds = 95 :=
by
  -- Proof is omitted here.
  sorry

end total_birds_count_l120_120389


namespace VIP_ticket_price_l120_120972

variable (total_savings : ℕ) 
variable (num_VIP_tickets : ℕ)
variable (num_regular_tickets : ℕ)
variable (price_per_regular_ticket : ℕ)
variable (remaining_savings : ℕ)

theorem VIP_ticket_price 
  (h1 : total_savings = 500)
  (h2 : num_VIP_tickets = 2)
  (h3 : num_regular_tickets = 3)
  (h4 : price_per_regular_ticket = 50)
  (h5 : remaining_savings = 150) :
  (total_savings - remaining_savings) - (num_regular_tickets * price_per_regular_ticket) = num_VIP_tickets * 100 := 
by
  sorry

end VIP_ticket_price_l120_120972


namespace max_disrespectful_polynomial_value_l120_120742

noncomputable def disrespectful (p : ℝ → ℝ) :=
  (∀ x, p(x) = x^2 + a * x + b) ∧
  (p ∘ p).rootSet.count = 3

noncomputable def unique_max_polynomial (p: ℝ → ℝ) :=
  disrespectful p ∧
  ∀ q, disrespectful q → p.rootSum ≥ q.rootSum

theorem max_disrespectful_polynomial_value :
  (∃ p: ℝ → ℝ, unique_max_polynomial p ∧ (p 1 = 5 / 16)) :=
begin
  sorry
end

end max_disrespectful_polynomial_value_l120_120742


namespace coordinates_of_focus_with_greater_x_coordinate_l120_120419

noncomputable def focus_of_ellipse_with_greater_x_coordinate : (ℝ × ℝ) :=
  let center : ℝ × ℝ := (3, -2)
  let a : ℝ := 3 -- semi-major axis length
  let b : ℝ := 2 -- semi-minor axis length
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let focus_x : ℝ := 3 + c
  (focus_x, -2)

theorem coordinates_of_focus_with_greater_x_coordinate :
  focus_of_ellipse_with_greater_x_coordinate = (3 + Real.sqrt 5, -2) := 
sorry

end coordinates_of_focus_with_greater_x_coordinate_l120_120419


namespace reflect_over_x_axis_l120_120287

def coords (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem reflect_over_x_axis :
  coords (-6, -9) = (-6, 9) :=
by
  sorry

end reflect_over_x_axis_l120_120287


namespace space_per_bush_l120_120411

theorem space_per_bush (side_length : ℝ) (num_sides : ℝ) (num_bushes : ℝ) (h1 : side_length = 16) (h2 : num_sides = 3) (h3 : num_bushes = 12) :
  (num_sides * side_length) / num_bushes = 4 :=
by
  sorry

end space_per_bush_l120_120411


namespace daily_sales_profit_10th_day_max_profit_34th_day_profit_max_34th_day_min_sales_profit_k_l120_120722

-- Definitions according to the problem's conditions
def selling_price (x : ℕ) : ℝ := - (1 / 2) * x + 55
def quantity_sold (x : ℕ) : ℕ := 5 * x + 50
def cost_price : ℝ := 18
def profit (x : ℕ) : ℝ := (selling_price x - cost_price) * (quantity_sold x)

-- Part (1) statement: Prove the profit on the 10th day is 3200 yuan
theorem daily_sales_profit_10th_day : profit 10 = 3200 := sorry

-- Part (2) statement: Prove the max profit between the 34th and 50th day
-- occurs on the 34th day and is 4400 yuan
def profit_function (x : ℕ) : ℝ := - (5 / 2) * (x ^ 2) + 160 * x + 1850

theorem max_profit_34th_day : 
  ∀ x : ℕ, 34 ≤ x → x ≤ 50 → profit_function x ≤ profit_function 34 := sorry

theorem profit_max_34th_day :
  profit_function 34 = 4400 := sorry

-- Part (3) statement: Prove increasing the selling price by k = 5.3 yuan 
-- results in minimum profit being 5460 yuan between the 30th and 40th day
def new_profit (x : ℕ) (k : ℝ) : ℝ :=
  (selling_price x + k - cost_price) * (quantity_sold x)

theorem min_sales_profit_k : 
  (∀ x : ℕ, 30 ≤ x → x ≤ 40 → (new_profit x 5.3) ≥ 5460) := sorry

end daily_sales_profit_10th_day_max_profit_34th_day_profit_max_34th_day_min_sales_profit_k_l120_120722


namespace number_40_necessarily_mentioned_l120_120306

/--
Consider an island with 33 residents, each belonging to one of the following categories: knights, liars, and dreamers. 
Each resident was asked how many knights are there among them, and they provided 10 different answers. 
It is known that:
1. Knights always tell the truth.
2. Liars always give an incorrect number that hasn't been mentioned yet.
3. Dreamers always give a number that is one greater than the previous answer.
Each different answer was given by more than one resident.
Prove that the number 40 was necessarily mentioned.
-/
theorem number_40_necessarily_mentioned : 
    ∃ answers : Finset ℕ,
      (33 ∈ answers ∧ 
       40 ∈ answers ∧ 
       10 ≤ answers.card) ∧ 
      ∀ n ∈ answers, 
        (n = 33 ∨ n ≠ 33 ∧ (n > 33 → ((n < 40 ∧ n.succ ∈ answers) ∨ (n > 40 ∧ ∀ k < n, k ∉ answers)))) := 
sorry

end number_40_necessarily_mentioned_l120_120306


namespace amusement_park_admission_l120_120279

def number_of_children (children_fee : ℤ) (adults_fee : ℤ) (total_people : ℤ) (total_fees : ℤ) : ℤ :=
  let y := (total_fees - total_people * children_fee) / (adults_fee - children_fee)
  total_people - y

theorem amusement_park_admission :
  number_of_children 15 40 315 8100 = 180 :=
by
  -- Fees in cents to avoid decimals
  sorry  -- Placeholder for the proof

end amusement_park_admission_l120_120279


namespace t_is_perfect_square_l120_120042

variable (n : ℕ) (hpos : 0 < n)
variable (t : ℕ) (ht : t = 2 + 2 * Nat.sqrt (1 + 12 * n^2))

theorem t_is_perfect_square (n : ℕ) (hpos : 0 < n) (t : ℕ) (ht : t = 2 + 2 * Nat.sqrt (1 + 12 * n^2)) : 
  ∃ k : ℕ, t = k * k := 
sorry

end t_is_perfect_square_l120_120042


namespace sine_probability_l120_120909

theorem sine_probability (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  (∃ y, 0 ≤ y ∧ y ≤ 2 ∧ 
    (sin ((π / 2) * y) ≥ sqrt 3 / 2) ↔
    (x = real.uniform 0 2 y)) → 
  real.uniform 0 2 x = 1 / 3 :=
sorry

end sine_probability_l120_120909


namespace magic_card_deck_earnings_l120_120710

theorem magic_card_deck_earnings 
  (initial_decks : ℕ) (remaining_decks : ℕ) (price_per_deck : ℕ) 
  (sold_decks : ℕ) (total_earnings : ℕ) :
  initial_decks = 16 → 
  remaining_decks = 8 → 
  price_per_deck = 7 → 
  sold_decks = initial_decks - remaining_decks → 
  total_earnings = sold_decks * price_per_deck → 
  total_earnings = 56 := 
by {
  intros h1 h2 h3 h4 h5,
  sorry
}

end magic_card_deck_earnings_l120_120710


namespace range_of_a_l120_120066

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≥ 0) ↔ -1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l120_120066


namespace sin_A_plus_sin_C_maximum_value_l120_120537

theorem sin_A_plus_sin_C_maximum_value (A B C a b c : ℝ) (h1 : a * cos A = b * sin B) (h2 : B > π / 2) :
  ∃ A, (0 < A ∧ A < π / 4) ∧ ((A + π/2 + (π/2 - 2 * A) = π)) ∧ (sin A + sin (π/2 - 2 * A) = 9 / 8) :=
  sorry

end sin_A_plus_sin_C_maximum_value_l120_120537


namespace seq_contains_divisible_term_l120_120984

open Nat

theorem seq_contains_divisible_term (n : ℕ) (h1 : odd n) (h2 : 1 < n) :
  ∃ k : ℕ, k < n ∧ n ∣ (2 ^ k - 1) := by
  sorry

end seq_contains_divisible_term_l120_120984


namespace inheritance_amount_l120_120322

theorem inheritance_amount (x : ℝ) 
  (federal_tax : ℝ := 0.25 * x) 
  (state_tax : ℝ := 0.15 * (x - federal_tax)) 
  (city_tax : ℝ := 0.05 * (x - federal_tax - state_tax)) 
  (total_tax : ℝ := 20000) :
  (federal_tax + state_tax + city_tax = total_tax) → 
  x = 50704 :=
by
  intros h
  sorry

end inheritance_amount_l120_120322


namespace scientific_notation_conversion_l120_120237

theorem scientific_notation_conversion (x : ℝ) (h_population : x = 141260000) :
  x = 1.4126 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l120_120237


namespace bisector_segments_angle_bisector_length_l120_120620

variables (a b c : ℝ) (p : ℝ)

/- Conditions -/
def triangle_side_a := a = b + c
def semi_perimeter := p = (a + b + c) / 2

/- Statements to prove -/
theorem bisector_segments :
  let m := (a * c) / (b + c)
  let n := (a * b) / (b + c)
  m + n = a ∧ m / n = c / b ∧ m = (a * c) / (b + c) ∧ n = (a * b) / (b + c) :=
by
  sorry

theorem angle_bisector_length :
  let m := (a * c) / (b + c)
  let n := (a * b) / (b + c)
  let la := (2 / (b + c)) * Real.sqrt(b * c * p * (p - a))
  la^2 = b * c - (m * n) :=
by
  sorry

end bisector_segments_angle_bisector_length_l120_120620


namespace sum_f_l120_120956

def A_n (n : ℕ) : Set ℕ := {k | k ∈ Finset.range (n + 1)}

def valid_sub (T : Set ℕ) : Prop := 
  ∀ a b ∈ T, a ≠ b → ¬(abs (a - b) = 4 ∨ abs (a - b) = 7)

def max_elem (n : ℕ) : ℕ :=
  (A_n n).to_finset.powerset.filter (valid_sub).max' sorry -- Ensuring there is at least one valid set.

def f (n : ℕ) := max_elem n

theorem sum_f : (Finset.range 2023).sum f = 929740 := 
  by sorry

end sum_f_l120_120956


namespace sum_of_sequence_120_to_130_l120_120766

def seq_sum_120_130 : ℕ :=
  let seq := (List.range' 120 11) in
  seq.sum

theorem sum_of_sequence_120_to_130 : seq_sum_120_130 = 1375 := by
  sorry

end sum_of_sequence_120_to_130_l120_120766


namespace students_knowing_both_languages_l120_120350

-- Define the universes of the problem
variables (U : Type) [Fintype U]
variables (G L : Finset U)

-- Define the conditions
theorem students_knowing_both_languages (hG : (G.card : ℝ) / (Fintype.card U) = 0.85)
                                        (hL : (L.card : ℝ) / (Fintype.card U) = 0.75)
                                        (hGU : (G ∪ L).card = Fintype.card U) :
  ((G ∩ L).card : ℝ) / (Fintype.card U) = 0.6 :=
begin
  sorry
end

end students_knowing_both_languages_l120_120350


namespace sneaker_price_l120_120733

noncomputable def final_price (original_price : ℝ) (coupon : ℝ) (promo_disc : ℝ) (event_disc : ℝ) (member_disc : ℝ) (sales_tax : ℝ) : ℝ :=
let price_after_coupon := original_price - coupon
    promo_discount := price_after_coupon * promo_disc
    price_after_promo := price_after_coupon - promo_discount
    event_discount := price_after_promo * event_disc
    price_after_event := price_after_promo - event_discount
    member_discount := price_after_event * member_disc
    price_after_member := price_after_event - member_discount
    tax := price_after_member * sales_tax
in (price_after_member + tax).round

theorem sneaker_price : 
  final_price 120 10 0.05 0.03 0.10 0.07 = 97.61 :=
by
  -- Insert proof here
  sorry

end sneaker_price_l120_120733


namespace pure_imaginary_frac_l120_120941

theorem pure_imaginary_frac (a b : ℝ) (h_nonzero_a : a ≠ 0) (h_nonzero_b : b ≠ 0) 
(h : ∃ c : ℝ, (3 - 9 * complex.I) * (a + b * complex.I) = c * complex.I) : a / b = -3 := by
  sorry

end pure_imaginary_frac_l120_120941


namespace number_of_student_tickets_sold_l120_120315

variable (A S : ℝ)

theorem number_of_student_tickets_sold
  (h1 : A + S = 59)
  (h2 : 4 * A + 2.5 * S = 222.50) :
  S = 9 :=
by sorry

end number_of_student_tickets_sold_l120_120315


namespace angle_equality_l120_120351

variables {A B C D P Q K M : Type*}
    [midpoint M B C]
    [trapezoid A B C D]
    [chosen_point P A D]
    [meets P M C D Q]
    [between C Q D]
    [perpendicular_through P A D B Q K]

theorem angle_equality :
  ∀ {A B C D P Q K M : Type*},
    midpoint M B C →
    trapezoid A B C D →
    chosen_point P A D →
    meets P M C D Q →
    between C Q D →
    perpendicular_through P A D B Q K →
  angle Q B C = angle K D A :=
begin
  sorry
end

end angle_equality_l120_120351


namespace minimum_value_f_5_l120_120013

noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

def f_n (n : ℕ) : (ℝ → ℝ) :=
  Nat.recOn n f (λ n' g, f ∘ g)

theorem minimum_value_f_5 :
  (∀ x : ℝ, x ∈ Set.Icc ((1:ℝ)/2) 1 → f_5 x ≥ (1 : ℝ)/12) ∧ (∃ x : ℝ, x ∈ Set.Icc ((1:ℝ)/2) 1 ∧ f_5 x = (1:ℝ)/12) := sorry

end minimum_value_f_5_l120_120013


namespace range_of_PA_PB_l120_120061

-- Define the circles C1 and C2
def point_on_circle1 (A B : ℝ × ℝ) : Prop := 
  A.1 ^ 2 + A.2 ^ 2 = 1 ∧ B.1 ^ 2 + B.2 ^ 2 = 1

def distance_AB (A B : ℝ × ℝ) : Prop := 
  (real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) = real.sqrt 3

def point_on_circle2 (P : ℝ × ℝ) : Prop :=
  (P.1 - 3) ^ 2 + (P.2 - 4) ^ 2 = 1

-- Define the midpoint of points A and B
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the distance formula
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

-- Prove the range of values for |PA + PB|
theorem range_of_PA_PB (A B P : ℝ × ℝ) (h_circle1 : point_on_circle1 A B) (h_AB : distance_AB A B) 
  (h_circle2 : point_on_circle2 P) : 
  ∃ l u : ℝ, l = 7 ∧ u = 13 ∧ l ≤ 2 * distance P (midpoint A B) ∧ 2 * distance P (midpoint A B) ≤ u := 
sorry

end range_of_PA_PB_l120_120061


namespace set_of_possible_values_of_a_inequality_mn_in_T_l120_120475

variable (a x : ℝ)

-- Define the function f(x) = |x-1| + |x+2|
def f (x : ℝ) := |x - 1| + |x + 2|

-- Prove that T = {a | -sqrt 3 < a < sqrt 3 }
theorem set_of_possible_values_of_a (a : ℝ) :
  (∀ x : ℝ, f x > a^2) ↔ (-sqrt 3 < a ∧ a < sqrt 3) :=
by
  sorry

-- Define the set T
def T := {a : ℝ | -sqrt 3 < a ∧ a < sqrt 3}

-- Prove sqrt 3 |m + n| < |mn + 3| for all m, n in T
theorem inequality_mn_in_T {m n : ℝ} (hm : m ∈ T) (hn : n ∈ T) :
  sqrt 3 * |m + n| < |m * n + 3| :=
by
  sorry

end set_of_possible_values_of_a_inequality_mn_in_T_l120_120475


namespace find_p_max_area_triangle_l120_120084

-- Define the parabola equation and the circle equation
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus of the parabola
def focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

-- Define the condition of the minimum distance
def min_dist_condition (p : ℝ) : Prop :=
  let F := focus p in
  let distance := λ (F M : (ℝ × ℝ)), Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) in
  ∀ (M : ℝ × ℝ), circle M.1 M.2 → distance F M - 1 = 4

-- Define the first part to prove p = 2 given the minimum distance condition
theorem find_p :
  ∃ (p : ℝ), p > 0 ∧ min_dist_condition p ↔ p = 2 := 
sorry

-- Define the condition for the maximum area of the triangle given tangents to the parabola and a point on the circle
def max_area_condition (p : ℝ) : Prop :=
  ∀ (x y : ℝ), circle x y → 
    let M := (x, y) in
    let tangents := λ (A B M : (ℝ × ℝ)), True in -- Placeholder for tangents
    True -- Placeholder for the maximum area condition

-- Define the second part to prove the maximum area given p = 2
theorem max_area_triangle :
  ∃ (area : ℝ), max_area_condition 2 ↔ area = 20 * Real.sqrt 5 :=
sorry

end find_p_max_area_triangle_l120_120084


namespace log_sum_geom_seq_l120_120893

variable {a : ℕ → ℝ} -- Define a as a sequence indexed by natural numbers.

-- Problem condition
def geometric_sequence (a : ℕ → ℝ) := 
  ∀ m n p q : ℕ, a m > 0 → a n > 0 → a p > 0 → a q > 0 → (m + n = p + q) → (a m * a n = a p * a q)

-- Given condition in the problem
def condition (a : ℕ → ℝ) := a 3 * a 8 = 9

theorem log_sum_geom_seq (a : ℕ → ℝ) [geometric_sequence a] (positive_terms : ∀ n : ℕ, a n > 0) 
  (h : condition a) : 
  Real.log 3 (a 1) + Real.log 3 (a 10) = 2 := 
sorry

end log_sum_geom_seq_l120_120893


namespace professor_gamble_lottery_l120_120981

-- Define the problem where we need to pick 6 different integers such that their product is a square or cube
theorem professor_gamble_lottery :
  -- Conditions: Picking 6 different integers from 1 to 49
  ∃ (S : Set ℕ) (hS1 : S ⊆ Finset.range 50) (hS2 : S.card = 6),
  -- Condition: Product of these integers is either a square or a cube
  (∃ (x : ℕ), (∃ k, x ^ (2*k) = (∏ i in S, i)) ∨ (∃ k, x ^ (3*k) = (∏ i in S, i))) →
  -- Question: Probability of holding a winning ticket is 1/12
  (1 : ℚ) / ↑12 :=
sorry

end professor_gamble_lottery_l120_120981


namespace pump_out_time_l120_120360

-- Define the conditions
def floor_length : ℝ := 30 
def floor_width : ℝ := 40 
def water_depth_inches : ℝ := 24 
def water_depth_feet : ℝ := water_depth_inches / 12
def number_of_pumps : ℕ := 4 
def pump_rate : ℝ := 10 
def gallons_per_cubic_foot : ℝ := 7.5 

-- Calculate the volume of water in cubic feet
def volume_cubic_feet : ℝ := floor_length * floor_width * water_depth_feet

-- Convert the volume to gallons
def volume_gallons : ℝ := volume_cubic_feet * gallons_per_cubic_foot

-- Total pumping rate
def total_pumping_rate : ℝ := number_of_pumps * pump_rate

-- Total time to pump out the water
def total_time : ℝ := volume_gallons / total_pumping_rate

-- Prove that the total time to pump out the water is 450 minutes
theorem pump_out_time : total_time = 450 := by
  sorry

end pump_out_time_l120_120360


namespace trilinear_distance_squared_l120_120595

variables {α β γ : ℝ} {x1 y1 z1 x2 y2 z2 : ℝ}

theorem trilinear_distance_squared 
  (hα : α + β + γ = real.pi) :
  let mn_squared := (cos α / (sin β * sin γ)) * (x1 - x2)^2 + 
                    (cos β / (sin γ * sin α)) * (y1 - y2)^2 + 
                    (cos γ / (sin α * sin β)) * (z1 - z2)^2
  in mn_squared = (x1 - x2)^2 / (sin β * sin γ) * cos α + 
                  (y1 - y2)^2 / (sin γ * sin α) * cos β + 
                  (z1 - z2)^2 / (sin α * sin β) * cos γ :=
sorry

end trilinear_distance_squared_l120_120595


namespace operator_add_satisfies_equation_l120_120906

theorem operator_add_satisfies_equation :
  -2 * real.sqrt 3 + real.sqrt 27 = real.sqrt 3 :=
by 
  -- Simplify sqrt(27)
  have h1 : real.sqrt 27 = 3 * real.sqrt 3 := by sorry
  -- Apply the identified operator and simplify
  rw [h1]
  norm_num

end operator_add_satisfies_equation_l120_120906


namespace surface_area_of_solid_l120_120631

-- Define a unit cube and the number of cubes
def unitCube : Type := { faces : ℕ // faces = 6 }
def numCubes : ℕ := 10

-- Define the surface area contribution from different orientations
def surfaceAreaFacingUs (cubes : ℕ) : ℕ := 2 * cubes -- faces towards and away
def verticalSidesArea (heightCubes : ℕ) : ℕ := 2 * heightCubes -- left and right vertical sides
def horizontalSidesArea (widthCubes : ℕ) : ℕ := 2 * widthCubes -- top and bottom horizontal sides

-- Define the surface area for the given configuration of 10 cubes
def totalSurfaceArea (cubes : ℕ) (height : ℕ) (width : ℕ) : ℕ :=
  (surfaceAreaFacingUs cubes) + (verticalSidesArea height) + (horizontalSidesArea width)

-- Assumptions based on problem description
def heightCubes : ℕ := 3
def widthCubes : ℕ := 4

-- The theorem we want to prove
theorem surface_area_of_solid : totalSurfaceArea numCubes heightCubes widthCubes = 34 := by
  sorry

end surface_area_of_solid_l120_120631


namespace roots_of_equation_l120_120123

theorem roots_of_equation : 
  (∃ s ∈ ([- real.sqrt 14, real.sqrt 14]).to_set, 
  (∀ x ∈ s, (real.sqrt (14 - x^2)) * (real.sin x - real.cos (2 * x)) = 0) ∧ 
  ( set.analyse (eq (set.card s) 6))) :=
sorry

end roots_of_equation_l120_120123


namespace cos_2α_value_beta_value_l120_120516

open Real

variables (α β : ℝ)
variables (cos_α sin_α : ℝ) (m n : ℝ × ℝ)

-- Conditions
def conditions : Prop :=
  α ∈ Ioo 0 (π / 2) ∧
  β ∈ Ioo 0 (π / 2) ∧
  m = (cos α, -1) ∧
  n = (2, sin α) ∧
  scalarProduct (cos α) (-1) (2) (sin α) = 0 ∧ -- vector perpendicular condition m⊥n
  cos_α = cos α ∧
  sin_α = sin α ∧
  cos_α^2 = 1 / 5 ∧
  sin (α - β) = sqrt (10) / 10

-- Questions
theorem cos_2α_value (h : conditions α β cos_α sin_α m n) : cos (2 * α) = -3 / 5 :=
sorry

theorem beta_value (h : conditions α β cos_α sin_α m n) : β = π / 4 :=
sorry

end cos_2α_value_beta_value_l120_120516


namespace claire_gerbils_l120_120698

theorem claire_gerbils (G H : ℕ) (h1 : G + H = 92) (h2 : (1/4 : ℚ) * G + (1/3 : ℚ) * H = 25) : G = 68 :=
sorry

end claire_gerbils_l120_120698


namespace solve_for_x_l120_120669

theorem solve_for_x : (3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5) = 800.0000000000001 → x = 2.5 :=
by
  sorry

end solve_for_x_l120_120669


namespace least_common_multiple_of_5_to_10_is_2520_l120_120342

-- Definitions of the numbers
def numbers : List ℤ := [5, 6, 7, 8, 9, 10]

-- Definition of prime factorization for verification (optional, keeping it simple)
def prime_factors (n : ℤ) : List ℤ :=
  if n = 5 then [5]
  else if n = 6 then [2, 3]
  else if n = 7 then [7]
  else if n = 8 then [2, 2, 2]
  else if n = 9 then [3, 3]
  else if n = 10 then [2, 5]
  else []

-- The property to be proved: The least common multiple of numbers is 2520
theorem least_common_multiple_of_5_to_10_is_2520 : ∃ n : ℕ, (∀ m ∈ numbers, m ∣ n) ∧ n = 2520 := by
  use 2520
  sorry

end least_common_multiple_of_5_to_10_is_2520_l120_120342


namespace subset_sum_difference_l120_120479

theorem subset_sum_difference {n : ℕ} (a b : Fin n → ℝ) (h : ∀ i, |a i - b i| < 1) :
  ∀ (s : Finset (Fin n)), |(∑ i in s, a i) - (∑ i in s, b i)| ≤ (n + 1) / 4 := 
sorry

end subset_sum_difference_l120_120479


namespace min_value_expression_l120_120583

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (x : ℝ), x = a^2 + 2 * b^2 + (1 / a^2) + (2 * b / a) ∧ x = sqrt 6 :=
by sorry

end min_value_expression_l120_120583


namespace log_condition_l120_120711

theorem log_condition (x : ℝ) : x > 1 ↔ log10 x > 0 :=
by
sorry

end log_condition_l120_120711


namespace odd_function_properties_l120_120143

noncomputable def f : ℝ → ℝ := sorry -- f is some odd function, to be defined
def interval1 : set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def interval2 : set ℝ := {x | -7 ≤ x ∧ x ≤ -3}

theorem odd_function_properties (f_odd : ∀ x, f(-x) = -f(x))
  (f_increasing : ∀ x y ∈ interval1, x < y → f(x) < f(y))
  (f_min_value : ∀ x ∈ interval1, f(x) ≥ 5 ∧ (∃ y ∈ interval1, f(y) = 5)) :
  (∀ x y ∈ interval2, x < y → f(x) < f(y)) ∧ (∃ z ∈ interval2, f(z) = -5) :=
sorry -- proof to be provided

end odd_function_properties_l120_120143


namespace max_cars_passed_in_hour_l120_120605

noncomputable def max_cars_in_hour (speed_per_car_length : ℕ → ℕ) (safety_distance_per_speed : ℕ → ℕ) : ℕ :=
  let n := (1000000 / 5) - 1
  (speed_per_car_length n) * (n / (n + 1))

def lane1_speed_per_car_length (n : ℕ) : ℕ := 15 * n
def lane1_safety_distance_per_speed (speed : ℕ) : ℕ := 5 * (speed / 15) 

def lane2_speed_per_car_length (n : ℕ) : ℕ := 10 * n
def lane2_safety_distance_per_speed (speed : ℕ) : ℕ := 5 * (speed / 10) 

def N := max_cars_in_hour lane1_speed_per_car_length lane1_safety_distance_per_speed
def M := max_cars_in_hour lane2_speed_per_car_length lane2_safety_distance_per_speed

def sum_of_quotients (N M : ℕ) := (N / 10) + (M / 10)

theorem max_cars_passed_in_hour : sum_of_quotients N M = 500 := by
  sorry

end max_cars_passed_in_hour_l120_120605


namespace find_value_of_a_l120_120757

-- Define variables and constants
variable (a : ℚ)
variable (b : ℚ := 3 * a)
variable (c : ℚ := 4 * b)
variable (d : ℚ := 6 * c)
variable (total : ℚ := 186)

-- State the theorem
theorem find_value_of_a (h : a + b + c + d = total) : a = 93 / 44 := by
  sorry

end find_value_of_a_l120_120757


namespace pump_out_time_l120_120361

-- Define the given conditions as constants
def length : ℝ := 30 -- feet
def width : ℝ := 40 -- feet
def depth : ℝ := 2 -- feet (from 24 inches)

def pump_rate : ℝ := 10 -- gallons per minute per pump
def number_of_pumps : ℕ := 4
def gallons_per_cubic_foot : ℝ := 7.5

-- Define the total volume of water in cubic feet
def volume : ℝ := length * width * depth

-- Convert the volume to gallons
def volume_in_gallons : ℝ := volume * gallons_per_cubic_foot

-- Compute the total pumping rate
def total_pumping_rate : ℝ := number_of_pumps * pump_rate

-- The expected time in minutes to pump out all the water
def time : ℝ := volume_in_gallons / total_pumping_rate

-- The theorem statement to be proven
theorem pump_out_time : time = 450 :=
  by
  -- Proof is skipped
  sorry

end pump_out_time_l120_120361


namespace solve_equation_roots_count_l120_120120

theorem solve_equation_roots_count :
  ∀ (x : ℝ), abs x ≤ real.sqrt 14 → 
  (sqrt (14 - x ^ 2) * (sin x - cos (2 * x)) = 0) → 
  (set.count {x | abs x ≤ real.sqrt 14 ∧ (sqrt (14 - x ^ 2) * (sin x - cos (2 * x)) = 0)} = 6) :=
by sorry

end solve_equation_roots_count_l120_120120


namespace scientific_notation_141260_million_l120_120233

theorem scientific_notation_141260_million :
  ∃ (a : ℝ) (n : ℤ), 141260 * 10^6 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4126 ∧ n = 5 :=
by
  sorry

end scientific_notation_141260_million_l120_120233


namespace comic_book_stackings_l120_120971

-- Definitions for the problem conditions
def number_spiderman_comics := 8
def number_archie_comics := 6
def number_garfield_comics := 7

-- Factorial function for convenience
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Calculate permutations within each group
def permutations_spiderman := fact number_spiderman_comics
def permutations_archie := fact (number_archie_comics - 1) -- one less because one specific comic is fixed on top
def permutations_garfield := fact number_garfield_comics

-- Calculate permutations for arranging groups
def permutations_groups := fact 2 -- only two groups' positions to decide (Spiderman and Garfield)

-- Total arrangements
def total_arrangements := permutations_spiderman * permutations_archie * permutations_garfield * permutations_groups

-- The proof statement
theorem comic_book_stackings : total_arrangements = 4864460800 := by
  -- the detailed proof will be written here
  sorry

end comic_book_stackings_l120_120971


namespace largest_prime_divisor_expr_largest_prime_divisor_is_13_l120_120798

-- Define the explicit expression
def expr : ℕ := 36^2 + 49^2

-- Lean proof statement to show the largest prime divisor of expr is 13
theorem largest_prime_divisor_expr : ∀ p : ℕ, prime p → p ∣ expr → p ≤ 13 :=
by
  sorry

-- Additional statement to show that 13 is indeed the largest prime divisor
theorem largest_prime_divisor_is_13 : prime 13 ∧ 13 ∣ expr :=
by
  sorry

end largest_prime_divisor_expr_largest_prime_divisor_is_13_l120_120798


namespace geometric_progression_ratio_l120_120983

theorem geometric_progression_ratio (b_1 : ℝ) (q : ℝ) (n : ℕ) (hq : q ≠ 1) :
  let S_n := b_1 * (q^n - 1) / (q - 1),
      b_n := b_1 * q^(n - 1)
  in (S_n - b_n) / (S_n - b_1) = 1 / q :=
by sorry

end geometric_progression_ratio_l120_120983


namespace maria_workers_problem_l120_120435

-- Define the initial conditions
def initial_days : ℕ := 40
def days_passed : ℕ := 10
def fraction_completed : ℚ := 2/5
def initial_workers : ℕ := 10

-- Define the required minimum number of workers to complete the job on time
def minimum_workers_required : ℕ := 5

-- The theorem statement
theorem maria_workers_problem 
  (initial_days : ℕ)
  (days_passed : ℕ)
  (fraction_completed : ℚ)
  (initial_workers : ℕ) :
  ( ∀ (total_days remaining_days : ℕ), 
    initial_days = 40 ∧ days_passed = 10 ∧ fraction_completed = 2/5 ∧ initial_workers = 10 → 
    remaining_days = initial_days - days_passed ∧ 
    total_days = initial_days ∧ 
    fraction_completed + (remaining_days / total_days) = 1) →
  minimum_workers_required = 5 := 
sorry

end maria_workers_problem_l120_120435


namespace panteleimon_twos_l120_120886

-- Define the variables
variables (P_5 P_4 P_3 P_2 G_5 G_4 G_3 G_2 : ℕ)

-- Define the conditions
def conditions :=
  P_5 + P_4 + P_3 + P_2 = 20 ∧
  G_5 + G_4 + G_3 + G_2 = 20 ∧
  P_5 = G_4 ∧
  P_4 = G_3 ∧
  P_3 = G_2 ∧
  P_2 = G_5 ∧
  (5 * P_5 + 4 * P_4 + 3 * P_3 + 2 * P_2 = 5 * G_5 + 4 * G_4 + 3 * G_3 + 2 * G_2)

-- The proof goal
theorem panteleimon_twos (h : conditions P_5 P_4 P_3 P_2 G_5 G_4 G_3 G_2) : P_2 = 5 :=
sorry

end panteleimon_twos_l120_120886


namespace eggs_per_chicken_per_day_l120_120190

-- Define the conditions
def chickens : ℕ := 8
def price_per_dozen : ℕ := 5
def total_revenue : ℕ := 280
def weeks : ℕ := 4
def eggs_per_dozen : ℕ := 12
def days_per_week : ℕ := 7

-- Theorem statement on how many eggs each chicken lays per day
theorem eggs_per_chicken_per_day :
  (chickens * ((total_revenue / price_per_dozen * eggs_per_dozen) / (weeks * days_per_week))) / chickens = 3 :=
by
  sorry

end eggs_per_chicken_per_day_l120_120190


namespace abs_sum_lt_abs_sum_of_neg_product_l120_120474

theorem abs_sum_lt_abs_sum_of_neg_product 
  (a b : ℝ) : ab < 0 ↔ |a + b| < |a| + |b| := 
by 
  sorry

end abs_sum_lt_abs_sum_of_neg_product_l120_120474


namespace domain_of_h_l120_120456

noncomputable def h (x : ℝ) : ℝ := (x^3 - 3*x^2 + 5*x - 2) / (x^2 - 5*x + 6)

theorem domain_of_h :
  {x : ℝ | ∃ h x, true ∧ (x ≠ 2 ∧ x ≠ 3)} = {x : ℝ | x < 2} ∪ {x : ℝ | x ≠ 2 ∧ x ≠ 3} ∪ {x : ℝ | 3 < x} :=
sorry

end domain_of_h_l120_120456


namespace volume_of_parallelepiped_l120_120215

variables (a b : ℝ^3)
variables (hab : ∥a∥ = 1) (hbb : ∥b∥ = 1) (theta : ℝ) (hθ : θ = Real.pi / 4)
variables (h_angle : Real.angle.cos theta = a.angleCos b)

-- Function calculating the volume of the parallelepiped
def volume_parallelepiped (u v w : ℝ^3) : ℝ :=
  abs (u.dot (v.cross w))

theorem volume_of_parallelepiped : 
  volume_parallelepiped (a + 2 • b) (b + (b ⊗ a)) b = 1 / 2 :=
by 
  sorry

end volume_of_parallelepiped_l120_120215


namespace simplify_expression_l120_120268

theorem simplify_expression : 8 * (15 / 9) * (-45 / 40) = -1 :=
  by
  sorry

end simplify_expression_l120_120268


namespace impossible_to_connect_15_telephones_l120_120163

theorem impossible_to_connect_15_telephones :
  ¬ ∃ (G : SimpleGraph (Fin 15)), ∀ v : (Fin 15), G.degree v = 5 :=
by
  sorry

end impossible_to_connect_15_telephones_l120_120163


namespace median_sin_cos_sixth_power_l120_120917

theorem median_sin_cos_sixth_power 
  (A B C : Type) 
  [MetricSpace A] 
  [MetricSpace B]
  [MetricSpace C]
  (AB BC : ℕ)
  (m : ℝ)
  (hAB : AB = 6) 
  (hBC : BC = 4) 
  (hMedian : m = Real.sqrt 10): 
  ∃ (A B : ℝ), (sin (A/2))^6 + (cos (A/2))^6 = (211/256) :=
by
  sorry

end median_sin_cos_sixth_power_l120_120917


namespace range_of_a_l120_120220

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * a * x + a + 2 ≤ 0 → 1 ≤ x ∧ x ≤ 4) ↔ a ∈ Set.Ioo (-1 : ℝ) (18 / 7) ∨ a = 18 / 7 := 
by
  sorry

end range_of_a_l120_120220


namespace average_sales_after_discount_l120_120277

theorem average_sales_after_discount :
  let sales := [120, 80, 60, 150, 90] in
  let total_sales := sales.sum in
  let discount := 0.1 * total_sales in
  let total_after_discount := total_sales - discount in
  let number_of_months := 5 in
  total_after_discount / number_of_months = 90 := by
  sorry

end average_sales_after_discount_l120_120277


namespace monotonicity_f_8_range_a_condition_l120_120026

def f (a : ℝ) (x : ℝ) : ℝ := a * x - sin x / cos x ^ 3

-- Prove that f(x) is increasing on (0, π/4) and decreasing on (π/4, π/2) when a = 8.
theorem monotonicity_f_8 (x : ℝ) (h : 0 < x ∧ x < π / 2) (hf : f 8 x = 8 * x - sin x / cos x ^ 3) :
  (0 < x ∧ x < π / 4 → deriv (f 8) x > 0) ∧ (π / 4 < x ∧ x < π / 2 → deriv (f 8) x < 0) :=
sorry

-- Prove that ∀ x ∈ (0, π/2), f(x) < sin 2x ↔ a ≤ 3.
theorem range_a_condition (a : ℝ) (h : ∀ x, 0 < x ∧ x < π / 2 → f a x < sin (2 * x)) :
  a ≤ 3 :=
sorry

end monotonicity_f_8_range_a_condition_l120_120026


namespace exists_rectangle_ABCD_l120_120049

open EuclideanGeometry

variables {K L M A B C D : Point}
variables {ℓ₁ ℓ₂ ℓ₃ : Line}

-- Conditions based on the problem statement.
axiom h1 : triangle K L M
axiom h2 : A ∈ extension LK
axiom h3 : B ∈ KM
axiom h4 : C ∈ KL
axiom h5 : D ∈ LM
axiom h6 : isRectangle A B C D

-- The theorem to be proven.
theorem exists_rectangle_ABCD (hcond : h1 ∧ h2 ∧ h3 ∧ h4 ∧ h5 ∧ h6) : 
  ∃ (A B C D : Point) (ABCD : Rectangle), 
    B ∈ KM ∧ C ∈ KL ∧ D ∈ LM ∧ isRectangle A B C D :=
by 
  sorry

end exists_rectangle_ABCD_l120_120049


namespace correct_propositions_l120_120852

-- Let p1 be the statement for Proposition ①
def p1 := ∀ x : ℝ, ¬ (∃ x : ℝ, x^2 - x > 0) ↔ (x^2 - x ≤ 0)

-- Let p2 be the statement for Proposition ②
def p2 := (∀ a b m : ℝ, (am^2 < bm^2) → (a < b)) → (¬ (∀ a b m : ℝ, (a < b) → (am^2 < bm^2)))

-- Let p3 be the statement for Proposition ③
def p3 := ∀ x : ℝ, ((f : ℝ → ℝ) → (Odd f) → (∀ x > 0, f x = 2^x) → (∀ x < 0, f x = -2^(-x)))

-- Let p4 be the statement for Proposition ④
def p4 := ∀ (ξ : ℝ) (σ : ℝ), (Normal ξ 1 σ^2) → ( Probability (ClosedSegment 0 1) ξ 0.3) → 
         (Probability (ClosedSegment ξ 2) ≥ 0.2)

-- Define the final proposition indicating the correctness of propositions ①, ③, and ④
theorem correct_propositions : p1 ∧ p3 ∧ p4 :=
sorry

end correct_propositions_l120_120852


namespace total_animal_legs_l120_120408

theorem total_animal_legs : 
  let dogs := 2 in
  let chickens := 1 in
  let cats := 3 in
  let spiders := 4 in
  let octopuses := 5 in
  let dog_legs := 4 in
  let chicken_legs := 2 in
  let cat_legs := 4 in
  let spider_legs := 8 in
  let octopus_legs := 0 in
  (dogs * dog_legs + chickens * chicken_legs + cats * cat_legs + spiders * spider_legs + octopuses * octopus_legs) = 54 :=
by 
  repeat { sorry }

end total_animal_legs_l120_120408


namespace abs_eq_linear_eq_l120_120444

theorem abs_eq_linear_eq (x : ℝ) : (|x - 5| = 3 * x + 1) ↔ x = 1 := by
  sorry

end abs_eq_linear_eq_l120_120444


namespace monotonicity_of_f_range_of_a_l120_120022

noncomputable def f (a x : ℝ) : ℝ := a * x - (Real.sin x / Real.cos x ^ 3)

noncomputable def g (a x : ℝ) : ℝ := f a x - Real.sin (2 * x)

theorem monotonicity_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) : 
  (8 = 8) -> 
  ((0 < x ∧ x < Real.pi / 4) -> f 8 x > 0) ∧ 
  ((Real.pi / 4 < x ∧ x < Real.pi / 2) -> f 8 x < 0) :=
by
  sorry

theorem range_of_a (a : ℝ) (x : ℝ) (h3 : 0 < x) (h4 : x < Real.pi / 2) :
  (f a x < Real.sin (2 * x)) -> 
  a ∈ (-∞, 3] :=
by
  sorry

end monotonicity_of_f_range_of_a_l120_120022


namespace perpendicular_lines_l120_120095

theorem perpendicular_lines (a : ℝ) 
  (l1_perp: ∀ x y : ℝ, ax + (a + 2) * y + 1 = 0) 
  (l2_perp: ∀ x y : ℝ, x + a * y + 2 = 0) 
  (perpendicular : ∀ l1 l2 : Prop, l1 ↔ l2 → l1 ∧ l2 → l1_perp ≠ l2_perp)
  : a = 0 ∨ a = -3 := sorry

end perpendicular_lines_l120_120095


namespace females_with_advanced_degrees_count_l120_120152

def company : Type := sorry

variables (total_employees : company) (total_females : company) (total_advanced_degrees : company)
variables (total_college_degrees_only : company) (total_males_college_degrees_only : company)

-- Given conditions
axiom total_employees_val : total_employees = 160
axiom total_females_val : total_females = 90
axiom total_advanced_degrees_val : total_advanced_degrees = 80
axiom total_college_degrees_only_val : total_college_degrees_only = 160 - 80
axiom total_males_college_degrees_only_val : total_males_college_degrees_only = 40

-- Derived calculations based on given conditions
def total_males : company := total_employees - total_females
def males_with_advanced_degrees : company := total_males - total_males_college_degrees_only
def females_with_advanced_degrees : company := total_advanced_degrees - males_with_advanced_degrees

-- The theorem to prove
theorem females_with_advanced_degrees_count : females_with_advanced_degrees = 50 :=
sorry

end females_with_advanced_degrees_count_l120_120152


namespace relationship_P_Q_l120_120871

variable (a : ℝ)
variable (P : ℝ := Real.sqrt a + Real.sqrt (a + 5))
variable (Q : ℝ := Real.sqrt (a + 2) + Real.sqrt (a + 3))

theorem relationship_P_Q (h : 0 ≤ a) : P < Q :=
by
  sorry

end relationship_P_Q_l120_120871


namespace find_A_solution_l120_120790

theorem find_A_solution (A : ℝ) (h : 32 * A^3 = 42592) : A = 11 :=
sorry

end find_A_solution_l120_120790


namespace simple_random_sampling_properties_l120_120335

-- Definitions of properties
def finite_population (pop : Type) [Fintype pop] : Prop := true
def extracted_one_by_one : Prop := true
def without_replacement : Prop := true
def equal_probability_sampling (pop : Type) [Fintype pop] : Prop := true

-- The main theorem to prove
theorem simple_random_sampling_properties (pop : Type) [Fintype pop] :
  finite_population pop ∧ extracted_one_by_one ∧ without_replacement ∧ equal_probability_sampling pop :=
by
  split; try {trivial}; -- For finite_population and extracted_one_by_one
  split; try {trivial}; -- For without_replacement and equal_probability_sampling
  sorry

end simple_random_sampling_properties_l120_120335


namespace transformation_result_l120_120682

theorem transformation_result :
  ∀ x : ℝ, (sin (2 * x + π / 6) - 1 + vec1 + sin (4 * x - π / 6)) = sin (4 * x - π / 6) :=
by
  sorry

end transformation_result_l120_120682


namespace impossible_to_tile_10x10_board_with_T_L_I_tetrominoes_l120_120427

theorem impossible_to_tile_10x10_board_with_T_L_I_tetrominoes :
  let board : nat := 100,
      squares_per_tetromino := 4,
      black_squares := 50,
      white_squares := 50 in
  (¬ ∃ (tiles : nat), tiles * squares_per_tetromino = board 
       ∧ tiles = black_squares ∧ (∀ tile, tile % 4 = 1)) :=
begin
  sorry
end

end impossible_to_tile_10x10_board_with_T_L_I_tetrominoes_l120_120427


namespace range_of_y_l120_120490

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b (y : ℝ) : ℝ × ℝ := (4, y)

-- Define the vector sum
def a_plus_b (y : ℝ) : ℝ × ℝ := (a.1 + (b y).1, a.2 + (b y).2)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove the angle between a and a + b is acute and y ≠ -8
theorem range_of_y (y : ℝ) :
  (dot_product a (a_plus_b y) > 0) ↔ (y < 4.5 ∧ y ≠ -8) :=
by
  sorry

end range_of_y_l120_120490


namespace total_number_of_squares_up_to_50th_ring_l120_120775

def number_of_squares_up_to_50th_ring : Nat :=
  let central_square := 1
  let sum_rings := (50 * (50 + 1)) * 4  -- Using the formula for arithmetic series sum where a = 8 and d = 8 and n = 50
  central_square + sum_rings

theorem total_number_of_squares_up_to_50th_ring : number_of_squares_up_to_50th_ring = 10201 :=
  by  -- This statement means we believe the theorem is true and will be proven.
    sorry                                                      -- Proof omitted, will need to fill this in later

end total_number_of_squares_up_to_50th_ring_l120_120775


namespace tangent_parallel_to_line_l120_120668

theorem tangent_parallel_to_line (x1 x2 y1 y2: ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (H1 : f = λ x, x^3 + x - 2)
  (H2 : f' = λ x, 3*x^2 + 1)
  (H3 : f'(x1) = 4 ∧ f'(x2) = 4) :
  (f x1 = y1) ∧ (x1, y1) = (1, 0) ∨ (x1, y1) = (-1, -4) :=
by
  -- Proof will be completed here
  sorry

end tangent_parallel_to_line_l120_120668


namespace probability_inequality_abs_ge_one_l120_120379

theorem probability_inequality_abs_ge_one (x : ℝ) (hx : x ∈ Set.Icc (-3 : ℝ) 3) :
  ∃ (p : ℝ), p = 1 / 3 ∧ ProbabilityTheory.prob {x | |x + 1| - |x - 2| ≥ 1} p := by
sorry

end probability_inequality_abs_ge_one_l120_120379


namespace right_triangle_area_l120_120402

theorem right_triangle_area
  (a b c : ℕ) 
  (h_a : a = 7) 
  (h_b : b = 24) 
  (h_c : c = 25) 
  (h_right_triangle : a * a + b * b = c * c) : 
  (is_right_triangle : a * a + b * b = c * c) ∧ (area : ℕ := a * b / 2) :=
by
  -- Proof of right angle triangle
  have h : a * a + b * b = c * c, from h_right_triangle,
  split,
  exact h,
  -- Calculation of area
  exact (a * b / 2)

#eval right_triangle_area 7 24 25 rfl rfl rfl sorry

end right_triangle_area_l120_120402


namespace maximum_value_lemma_l120_120463

noncomputable def maxExpression (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  ∑ i, (x i) ^ 4 - (x i) ^ 5

theorem maximum_value_lemma (n : ℕ) (x : Fin n → ℝ) 
  (h1 : ∑ i, x i = 1) 
  (h2 : ∀ i, 0 ≤ x i) : 
  maxExpression n x ≤ 1 / 12 :=
sorry

end maximum_value_lemma_l120_120463


namespace area_of_circle_is_approximately_l120_120366

-- Define constants and conditions
def radius : ℝ := 11
def pi_approx : ℝ := 3.14159

-- Define the function to calculate the area of a circle
noncomputable def area_of_circle (r : ℝ) : ℝ := pi_approx * r^2

-- Define the theorem statement of the problem
theorem area_of_circle_is_approximately :
  area_of_circle radius ≈ 379.94 := sorry

end area_of_circle_is_approximately_l120_120366


namespace profit_is_55_l120_120258

-- Define the given conditions:
def cost_of_chocolates (bars: ℕ) (price_per_bar: ℕ) : ℕ :=
  bars * price_per_bar

def cost_of_packaging (bars: ℕ) (cost_per_bar: ℕ) : ℕ :=
  bars * cost_per_bar

def total_sales : ℕ :=
  90

def total_cost (cost_of_chocolates cost_of_packaging: ℕ) : ℕ :=
  cost_of_chocolates + cost_of_packaging

def profit (total_sales total_cost: ℕ) : ℕ :=
  total_sales - total_cost

-- Given values:
def bars: ℕ := 5
def price_per_bar: ℕ := 5
def cost_per_packaging_bar: ℕ := 2

-- Define the profit calculation theorem:
theorem profit_is_55 : 
  profit total_sales (total_cost (cost_of_chocolates bars price_per_bar) (cost_of_packaging bars cost_per_packaging_bar)) = 55 :=
by {
  -- The proof will be inserted here
  sorry
}

end profit_is_55_l120_120258


namespace range_of_M_l120_120935

theorem range_of_M (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h_sum : a + b + c = 1) :
  let M := (1 / a - 1) * (1 / b - 1) * (1 / c - 1) in M ≥ 8 :=
sorry

end range_of_M_l120_120935


namespace initial_pairs_of_shoes_l120_120601

theorem initial_pairs_of_shoes (lost_shoes pairs_left : ℕ) (h1 : lost_shoes = 9) (h2 : pairs_left = 18) : 
  let initial_individual_shoes := 2 * pairs_left + lost_shoes in
  let initial_pairs := initial_individual_shoes / 2 in
  (initial_pairs = 22 ∧ initial_individual_shoes % 2 = 1) :=
by
  sorry

end initial_pairs_of_shoes_l120_120601


namespace product_of_terms_l120_120764

theorem product_of_terms :
  (∏ n in Finset.range 11 \ Finset.singleton 0, (1 - 1 / (n + 2)^2 : ℚ)) = 13 / 24 :=
by sorry

end product_of_terms_l120_120764


namespace pipes_ratio_l120_120748

theorem pipes_ratio (Ra Rb Rc : ℝ)
  (h1 : Rc = Rb)
  (h2 : Rb = Ra)
  (h3 : 3 * Ra = 1 / 5)
  (h4 : Ra = 1 / 35) :
  Rc / Rb = 1 ∧ Rb / Ra = 1 :=
by {
  rw [h1, h2],
  split;
  rw div_self;
  exact (ne_of_gt (div_pos zero_lt_one (show (35:ℝ) > 0, by norm_num))),
  exact (ne_of_gt (div_pos zero_lt_one (show (5:ℝ) > 0, by norm_num))),
  sorry
}

end pipes_ratio_l120_120748


namespace det_transformation_matrix_l120_120774

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![k, 0], ![0, k]]

def rotation_matrix_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  dilation_matrix 3 ⬝ rotation_matrix_90_ccw

theorem det_transformation_matrix : (transformation_matrix.det) = 9 := by
  sorry

end det_transformation_matrix_l120_120774


namespace pencil_case_choices_l120_120399

theorem pencil_case_choices : 
  let n := 6 in 
  2^n = 64 :=
by
  sorry

end pencil_case_choices_l120_120399


namespace range_of_a_for_F1_range_of_a_for_F2_range_of_a_for_F1_or_F2_l120_120010

theorem range_of_a_for_F1 (a : ℝ) : 
  (2 * a + 1) / a ≤ 1 ↔ a ∈ set.Icc (-1 : ℝ) 0 :=
by sorry

theorem range_of_a_for_F2 (a : ℝ) : 
  (∃ x ∈ set.Icc (0 : ℝ) 2, -x ^ 3 + 3 * x + 2 * a - 1 = 0) ↔ a ∈ set.Icc (-1 / 2 : ℝ) (3 / 2) :=
by sorry

theorem range_of_a_for_F1_or_F2 (a : ℝ) : 
  ( (2 * a + 1) / a ≤ 1 ∨ ∃ x ∈ set.Icc (0 : ℝ) 2, -x ^ 3 + 3 * x + 2 * a - 1 = 0 ) ↔ a ∈ set.Icc (-1 : ℝ) (3 / 2) :=
by sorry

end range_of_a_for_F1_range_of_a_for_F2_range_of_a_for_F1_or_F2_l120_120010


namespace scientific_notation_conversion_l120_120235

theorem scientific_notation_conversion (x : ℝ) (h_population : x = 141260000) :
  x = 1.4126 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l120_120235


namespace solve_system_of_equations_l120_120273

theorem solve_system_of_equations :
  ∃ x y : ℝ, (x - y = 2) ∧ (2 * x + y = 7) ∧ x = 3 ∧ y = 1 :=
by
  sorry

end solve_system_of_equations_l120_120273


namespace robert_ate_more_chocolates_l120_120257

-- Define the number of chocolates eaten by Robert and Nickel
def robert_chocolates : ℕ := 12
def nickel_chocolates : ℕ := 3

-- State the problem as a theorem to prove
theorem robert_ate_more_chocolates :
  robert_chocolates - nickel_chocolates = 9 :=
by
  sorry

end robert_ate_more_chocolates_l120_120257


namespace rachel_colored_l120_120987

def total_pictures (book1: ℕ) (book2: ℕ): ℕ := book1 + book2

def remaining_pictures (total: ℕ) (colored: ℕ): ℕ := total - colored

theorem rachel_colored (book1 book2 remaining: ℕ) (h₁: book1 = 23) (h₂: book2 = 32) (h₃: remaining = 11):
  remaining_pictures (total_pictures book1 book2) remaining = 44 :=
by
  rw [h₁, h₂, h₃]
  unfold total_pictures remaining_pictures
  norm_num
  sorry

end rachel_colored_l120_120987


namespace arithmetic_sequence_sum_of_transformed_sequence_l120_120301

/-- Problem 1 -/
theorem arithmetic_sequence (a : ℕ → ℤ) 
  (h0 : a 1 = -13) 
  (h1 : ∀ n : ℕ, (1 : ℚ) / a n - (2 : ℚ) / (a n * a (n + 1)) - (1 : ℚ) / a (n + 1) = 0) :
  ∃ d : ℤ, ∃ a_1 : ℤ, (a 1 = a_1) ∧ (a (n + 1) = a n + d) :=
sorry

/-- Problem 2 -/
theorem sum_of_transformed_sequence (S : ℕ → ℤ) (T : ℕ → ℚ) 
  (h0 : ∀ n : ℕ, S (n+1) = S n + (a (n+1)))
  (h1 : ∀ n : ℕ, T n = ∑ i in finset.range n, (S i / (i + 1))) :
  T n = (1/2 : ℚ) * n^2 - (27/2 : ℚ) * n :=
sorry

end arithmetic_sequence_sum_of_transformed_sequence_l120_120301


namespace necessary_condition_for_quadratic_inequality_l120_120089

theorem necessary_condition_for_quadratic_inequality (a : ℝ) :
  (∃ x ∈ set.Icc 1 3, x^2 - a * x + 4 < 0) → a > 3 :=
by
  sorry

end necessary_condition_for_quadratic_inequality_l120_120089


namespace greatest_possible_average_l120_120664

/-- The set S consists of 9 distinct positive integers. The average of the two smallest integers in
    S is 5. The average of the two largest integers in S is 22. The greatest possible average of all
    integers in S is 16. -/
theorem greatest_possible_average (S : Finset ℕ) (h_distinct : S.card = 9) 
    (h_pos : ∀ x ∈ S, 0 < x) 
    (h_avg_min : (S.min' (λ x hx, Nat.lt_of_le_of_ne x.zero_le (h_pos x hx))) + 
                 S.min' (λ x hx, Nat.lt_of_le_of_ne x.zero_le (h_pos x hx) ∘ S.erase
                           (S.min' (λ x hx, Nat.lt_of_le_of_ne x.zero_le (h_pos x hx))))) / 2 = 5) 
    (h_avg_max : (S.max' 0) + (S.max' (S.erase (S.max' 0)) 0) / 2 = 22) 
    : (S.sum id) / 9 = 16 :=
sorry

end greatest_possible_average_l120_120664


namespace chebyshev_number_of_variables_l120_120563

open MeasureTheory Probability MeasureTheory.Measure

variables {ι : Type*} {Ω : Type*} [MeasureSpace Ω] {X : ι → Ω → ℝ}
variables (hx : ∀ i, has_variance (X i) ∧ (variance (X i) ≤ 4)) (n : ℕ)
variables (ε : ℝ) (hε : ε = 0.25) (p : ℝ) (hp : p = 0.99)

theorem chebyshev_number_of_variables :
  (P (ω, ℝ) in measure_space.measure_space (Ω) => 
    abs ((∑ i in finset.range n, X i ω) / n - (∑ i in finset.range n, Ε (X i)) / n) ≤ ε) > p →
  n ≥ 6400 :=
begin
  sorry
end

end chebyshev_number_of_variables_l120_120563


namespace infinite_composite_n_dn_dn1_perfect_square_l120_120802

def is_composite (n : ℕ) : Prop :=
  ∃ d, d > 1 ∧ d < n ∧ n % d = 0

def largest_proper_divisor (n : ℕ) : ℕ :=
  (List.filter (λ d, d > 1 ∧ d < n ∧ n % d = 0) (List.range n)).maximum' (by {
    use 1,
    simp,
    exact ⟨1, nat.prime_two⟩,
  })

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem infinite_composite_n_dn_dn1_perfect_square :
  ∃ᵢ (n: ℕ), is_composite n ∧ is_composite (n + 1) ∧ is_perfect_square (largest_proper_divisor n + largest_proper_divisor (n + 1)) :=
  sorry

end infinite_composite_n_dn_dn1_perfect_square_l120_120802


namespace graduation_ceremony_l120_120154

theorem graduation_ceremony (teachers administrators graduates chairs : ℕ) 
  (h1 : teachers = 20) 
  (h2 : administrators = teachers / 2) 
  (h3 : graduates = 50) 
  (h4 : chairs = 180) :
  (chairs - (teachers + administrators + graduates)) / graduates = 2 :=
by 
  sorry

end graduation_ceremony_l120_120154


namespace find_number_l120_120693

theorem find_number (x : ℝ) (h : (3/4 : ℝ) * x = 93.33333333333333) : x = 124.44444444444444 := 
by
  -- Proof to be filled in
  sorry

end find_number_l120_120693


namespace tan_theta_l120_120202

theorem tan_theta (k θ : ℝ) (hk : k > 0)
      (D : Matrix (Fin 2) (Fin 2) ℝ := ![![k, 0], ![0, k]])
      (R : Matrix (Fin 2) (Fin 2) ℝ := ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]])
      (P : Matrix (Fin 2) (Fin 2) ℝ := ![![0, 1], ![1, 0]])
      (h : P.mul R.mul D = ![![6, -3], ![3, 6]]) :
    Real.tan θ = 2 :=
by
  sorry

end tan_theta_l120_120202


namespace equilateral_heptagon_perpendicular_sum_eq_l120_120177

-- The Lean statement for the given problem:
theorem equilateral_heptagon_perpendicular_sum_eq 
  (A : Fin 7 → ℝ × ℝ) (O : ℝ × ℝ) (H : Fin 7 → ℝ × ℝ)
  (h1 : ∀ i : Fin 7, ∃ j : Fin 7, H j = some (θ := true) ((λ x : ℝ × ℝ, O - x) '' ({A i}) ∩ line_segment ℝ (A i) (A (i+1) % 7)))
  (h2 : ∀ i : Fin 7, eq_dist (A i) (A ((i + 1) % 7)) (A ((i + 2) % 7))) :
  ∑ i in Finset.range 7, dist (A i) (H i) = ∑ i in Finset.range 7, dist (H i) (A ((i + 1) % 7)) :=
begin
  sorry
end

end equilateral_heptagon_perpendicular_sum_eq_l120_120177


namespace unique_poly_value_l120_120740

-- Definitions based on conditions and question

def disrespectful_quad_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ r s : ℝ, p = λ x, x^2 - (r + s) * x + r * s ∧ 
  (∀ y, p y = r ∨ p y = s) ∧ 
  -- p(p(x)) = 0 has exactly three real solutions.
  (∃ a b c : ℝ, (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ (p (p x) = 0 → ∃ a b c : ℝ, x = a ∨ x = b ∨ x = c))

def max_sum_roots_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ r s : ℝ, p = λ x, x^2 - (r + s) * x + r * s ∧ 
  ∀ q : ℝ → ℝ, disrespectful_quad_polynomial q → 
  ∃ rq sq : ℝ, q = λ x, x^2 - (rq + sq) * x + rq * sq ∧ 
  r + s ≥ rq + sq

def unique_max_sum_roots_polynomial (p : ℝ → ℝ) : Prop :=
  max_sum_roots_polynomial p ∧ ∀ q : ℝ → ℝ, 
  max_sum_roots_polynomial q → p = q

-- Lean statement for our problem
theorem unique_poly_value : 
  ∃ p : ℝ → ℝ, unique_max_sum_roots_polynomial p ∧ p 1 = 5 / 16 :=
by
  sorry

end unique_poly_value_l120_120740


namespace johns_profit_l120_120187

-- Definitions based on Conditions
def original_price_per_bag : ℝ := 4
def discount_percentage : ℝ := 0.10
def discounted_price_per_bag := original_price_per_bag * (1 - discount_percentage)
def bags_bought : ℕ := 30
def cost_per_bag : ℝ := if bags_bought >= 20 then discounted_price_per_bag else original_price_per_bag
def total_cost := bags_bought * cost_per_bag
def bags_sold_to_adults : ℕ := 20
def bags_sold_to_children : ℕ := 10
def price_per_bag_for_adults : ℝ := 8
def price_per_bag_for_children : ℝ := 6
def revenue_from_adults := bags_sold_to_adults * price_per_bag_for_adults
def revenue_from_children := bags_sold_to_children * price_per_bag_for_children
def total_revenue := revenue_from_adults + revenue_from_children
def profit := total_revenue - total_cost

-- Lean Statement to be Proven
theorem johns_profit : profit = 112 :=
by
  sorry

end johns_profit_l120_120187


namespace least_xy_value_l120_120829

theorem least_xy_value (x y : ℕ) (hposx : x > 0) (hposy : y > 0) (h : 1/x + 1/(3*y) = 1/8) :
  xy = 96 :=
by
  sorry

end least_xy_value_l120_120829


namespace find_digits_l120_120300

def repeating_decimal_abab (a b : ℕ) : ℝ :=
  (101 * a + 11 * b) / 9999

def repeating_decimal_abcabc (a b c : ℕ) : ℝ :=
  (100100 * a + 10010 * b + 1001 * c) / 999999

theorem find_digits (a b c : ℕ) (h_a : a < 10) (h_b : b < 10) (h_c : c < 10) :
  repeating_decimal_abab a b + repeating_decimal_abcabc a b c = 33 / 37 ↔
  a = 4 ∧ b = 4 ∧ c = 7 :=
sorry

end find_digits_l120_120300


namespace polynomial_remainder_l120_120781

noncomputable def polynomial_remainder_eval : ℤ :=
  let f := λ x : ℤ, x ^ 15 - 2
  in f (-2)

theorem polynomial_remainder :
  polynomial_remainder_eval = -32770 :=
by
  sorry

end polynomial_remainder_l120_120781


namespace inequality_proof_l120_120209

theorem inequality_proof
(n : ℕ) (a : ℕ → ℝ)
(hn : n > 0)
(h0n : ∀ i : ℕ, i < n → 0 ≤ a i)
(h1n : ∀ i : ℕ, i < n → a i ≤ 1) :
  (∏ i in finset.range n, (1 - (a i) ^ n)) ≤ (1 - finset.prod (finset.range n) (λ i, a i)) ^ n :=
sorry

end inequality_proof_l120_120209


namespace multiplication_problem_l120_120873

-- Define the problem in Lean 4.
theorem multiplication_problem (a b : ℕ) (ha : a < 10) (hb : b < 10) 
  (h : (30 + a) * (10 * b + 4) = 126) : a + b = 7 :=
sorry

end multiplication_problem_l120_120873


namespace factorization_correct_l120_120788

-- Define the given expression
def expression (a b : ℝ) : ℝ := 9 * a^2 * b - b

-- Define the factorized form
def factorized_form (a b : ℝ) : ℝ := b * (3 * a + 1) * (3 * a - 1)

-- Theorem stating that the factorization is correct
theorem factorization_correct (a b : ℝ) : expression a b = factorized_form a b := by
  sorry

end factorization_correct_l120_120788


namespace percentage_of_cash_is_20_l120_120189

theorem percentage_of_cash_is_20
  (raw_materials : ℕ)
  (machinery : ℕ)
  (total_amount : ℕ)
  (h_raw_materials : raw_materials = 35000)
  (h_machinery : machinery = 40000)
  (h_total_amount : total_amount = 93750) :
  (total_amount - (raw_materials + machinery)) * 100 / total_amount = 20 :=
by
  sorry

end percentage_of_cash_is_20_l120_120189


namespace fewer_seats_right_side_l120_120540

theorem fewer_seats_right_side
  (left_seats : ℕ)
  (people_per_seat : ℕ)
  (back_seat_capacity : ℕ)
  (total_capacity : ℕ)
  (h1 : left_seats = 15)
  (h2 : people_per_seat = 3)
  (h3 : back_seat_capacity = 12)
  (h4 : total_capacity = 93)
  : left_seats - (total_capacity - (left_seats * people_per_seat + back_seat_capacity)) / people_per_seat = 3 :=
  by sorry

end fewer_seats_right_side_l120_120540


namespace sum_of_operations_on_digits_of_2345_l120_120433

noncomputable def digit_operations_sum : ℝ :=
  let d₁ := (2 : ℝ) ^ 2
  let d₂ := (3 : ℝ) * real.sin (real.pi / 6)
  let d₃ := (4 : ℝ) ^ 3
  let d₄ := (4 : ℝ) ^ 3 - 5
  d₁ + d₂ + d₃ + d₄

theorem sum_of_operations_on_digits_of_2345 :
  digit_operations_sum = 128.5 :=
by 
  sorry

end sum_of_operations_on_digits_of_2345_l120_120433


namespace cosine_difference_triangle_l120_120172

theorem cosine_difference_triangle 
  (A B C M O Q R : Type*)
  [Point A] [Point B] [Point C] [Point M] [Point O] [Point Q] [Point R]
  (angle_BAC : Real := π / 3)
  (ratio_condition : Real) 
  (h1 : ratio_condition = Real.sqrt 7 * (QR / AR))
  (h2 : ∠BAC = π / 3) :
  ∃ cos_val : Real, cos_val = -1 / 8 := 
begin
  existsi (-1/8),
  exact sorry,
end

end cosine_difference_triangle_l120_120172


namespace decompose_arccos_plus_one_l120_120443

noncomputable def chebyshev_fourier_expansion : Real → Real :=
  λ x, Real.arccos x + 1

noncomputable def chebyshev_polynomial (n : Nat) : Real → Real :=
  λ x, Real.cos (n * Real.arccos x)

noncomputable def inner_product (u v : Real → Real) : Real :=
  ∫ (x : Real) in -1..1, u x * v x * (1 / Real.sqrt (1 - x ^ 2))

theorem decompose_arccos_plus_one :
    ∀ x ∈ Set.Icc (-1 : Real) 1,
    chebyshev_fourier_expansion x =
    (π + 2) / 2 + (2 / π) * ∑' n in Nat.succ, (λ n, (Real.ofNat n ^ -2 * (ite (Even n) -1 1 - 1)) * chebyshev_polynomial n x)
by
  intro x hx
  sorry

end decompose_arccos_plus_one_l120_120443


namespace inclination_angle_l120_120784

-- Define the parametric representations for curve C and line l
def curveC_parametric (θ : ℝ) : ℝ × ℝ :=
  ( -6 + 5 * Real.cos θ,
    5 * Real.sin θ )

def lineL_parametric (t α : ℝ) : ℝ × ℝ :=
  ( t * Real.cos α,
    t * Real.sin α )

-- Define functions for polar equation of curve C and condition for line l
def polar_eq_curveC (ρ θ : ℝ) : Prop :=
  ρ^2 + 12 * ρ * Real.cos θ + 11 = 0

def polar_eq_lineL (ρ₁ ρ₂ α : ℝ) : Prop :=
  ρ₁ + ρ₂ = -12 * Real.cos α ∧ ρ₁ * ρ₂ = 11

-- Define the proof problem
theorem inclination_angle (α : ℝ) :
  (∃ (θ ρ : ℝ), polar_eq_curveC ρ θ) ∧
  (∃ (ρ₁ ρ₂ : ℝ), polar_eq_lineL ρ₁ ρ₂ α ∧ |ρ₁ - ρ₂| = 2 * Real.sqrt 7) →
  α = Real.pi / 4 ∨ α = 3 * Real.pi / 4 :=
sorry

end inclination_angle_l120_120784


namespace lambda_value_l120_120915

theorem lambda_value (A B C D P : Point) (λ : ℝ) 
  (h1 : collinear A C D) (h2 : D ∈ segment A C)
  (h3 : P ∈ segment B D) 
  (h4 : vector_eq (vector C D) (vector D A)) 
  (h5 : vector_eq (vector A P) (λ • vector A B + (1/6) • vector A C)) 
  : λ = 2/3 := 
begin
  sorry
end

end lambda_value_l120_120915


namespace area_half_l120_120800

theorem area_half (width height : ℝ) (h₁ : width = 25) (h₂ : height = 16) :
  (width * height) / 2 = 200 :=
by
  -- The formal proof is skipped here
  sorry

end area_half_l120_120800


namespace ninth_term_of_geometric_sequence_l120_120729

theorem ninth_term_of_geometric_sequence (a r : ℕ) (h1 : a = 3) (h2 : a * r^6 = 2187) : a * r^8 = 19683 := by
  sorry

end ninth_term_of_geometric_sequence_l120_120729


namespace convex_polyhedra_with_n_diagonals_l120_120178

theorem convex_polyhedra_with_n_diagonals (n : ℕ) : 
  ∃ (P : Type) [convex_polyhedron P], count_diagonals P = n :=
sorry

end convex_polyhedra_with_n_diagonals_l120_120178


namespace visibleFactorNumbersCount_l120_120387

-- Define what it means for a number to be a visible factor number
def isVisibleFactorNumber (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ d ∈ digits, d ≠ 0 → n % d = 0

-- Define the range of numbers from 200 to 250
def range200to250 := {n : ℕ | n >= 200 ∧ n <= 250}

-- The main theorem statement
theorem visibleFactorNumbersCount : 
  {n | n ∈ range200to250 ∧ isVisibleFactorNumber n}.card = 22 := 
by sorry

end visibleFactorNumbersCount_l120_120387


namespace lara_gives_betty_l120_120192

variables (X Y : ℝ)

-- Conditions
-- Lara has spent X dollars
-- Betty has spent Y dollars
-- Y is greater than X
theorem lara_gives_betty (h : Y > X) : (Y - X) / 2 = (X + Y) / 2 - X :=
by
  sorry

end lara_gives_betty_l120_120192


namespace correct_average_weight_l120_120700

theorem correct_average_weight (avg_weight : ℝ) (num_boys : ℕ) (incorrect_weight correct_weight : ℝ)
  (h1 : avg_weight = 58.4) (h2 : num_boys = 20) (h3 : incorrect_weight = 56) (h4 : correct_weight = 62) :
  (avg_weight * ↑num_boys + (correct_weight - incorrect_weight)) / ↑num_boys = 58.7 := by
  sorry

end correct_average_weight_l120_120700


namespace planes_parallel_if_line_perpendicular_to_both_l120_120943

variables {Line Plane : Type}
variables (l : Line) (α β : Plane)

-- Assume we have a function parallel that checks if a line is parallel to a plane
-- and a function perpendicular that checks if a line is perpendicular to a plane. 
-- Also, we assume a function parallel_planes that checks if two planes are parallel.
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

theorem planes_parallel_if_line_perpendicular_to_both
  (h1 : perpendicular l α) (h2 : perpendicular l β) : parallel_planes α β :=
sorry

end planes_parallel_if_line_perpendicular_to_both_l120_120943


namespace integer_value_of_a_l120_120146

theorem integer_value_of_a (a : ℤ) :
  (∑ x in (finset.filter (λ x, x > a - 1 ∧ x ≤ 5) (finset.Icc (-100) 100)), x) = 14 → a = 2 :=
by sorry

end integer_value_of_a_l120_120146


namespace area_of_triangle_ABC_l120_120704

noncomputable def triangle_area (a b c : ℝ) (gamma : ℝ) : ℝ := 
  1/2 * a * b * Math.sin γ

noncomputable def triangle_A1CB1_area (a b c : ℝ) (gamma : ℝ) : ℝ :=
  a^2 * b^2 / ((a + c) * (b + c)) * Math.sin γ / 2

noncomputable def triangle_DCE_area (a b c : ℝ) (gamma : ℝ) : ℝ := 
  1/2 * (a + c) * (b + c) * Math.sin γ

theorem area_of_triangle_ABC (a b c gamma : ℝ) 
  (h1 : AB = AD = BE)
  (h2 : triangle_DCE_area a b c gamma = 9)
  (h3 : triangle_A1CB1_area a b c gamma = 4) :
  triangle_area a b c gamma = 6 :=
sorry

end area_of_triangle_ABC_l120_120704


namespace min_square_distance_l120_120250

theorem min_square_distance (x y z w : ℝ) (h1 : x * y = 4) (h2 : z^2 + 4 * w^2 = 4) : (x - z)^2 + (y - w)^2 ≥ 1.6 :=
sorry

end min_square_distance_l120_120250


namespace fixed_point_exists_find_ellipse_equation_l120_120050

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

def tangent_line (x y : ℝ) : Prop :=
  y = x + sqrt 6

def line_passing_through_fixed_point (x y : ℝ) (k m : ℝ) (h : k ≠ 0) : Prop :=
  ∃ x1 x2 y1 y2 : ℝ,
    ellipse x1 y1 ∧ ellipse x2 y2 ∧
    y1 = k * x1 + m ∧
    y2 = k * x2 + m ∧
    (x1 - 2) * (x2 - 2) + y1 * y2 = 0 ∧
    sqrt (7 * (m / k)^2 + 16 * (m / k) + 4) = 0

theorem fixed_point_exists (k : ℝ) (h : k ≠ 0) :
  ∀ m : ℝ, ∃ x y : ℝ, line_passing_through_fixed_point x y k m h → (x, y) = (2 / 7, 0) :=
sorry

theorem find_ellipse_equation :
  ∃ a b : ℝ, 0 < b ∧ b < a ∧ a = 2 ∧ b = sqrt 3 ∧ ellipse = (λ x y, (x^2 / (a^2)) + (y^2 / (b^2)) = 1) :=
sorry

end fixed_point_exists_find_ellipse_equation_l120_120050


namespace product_of_primes_l120_120613

theorem product_of_primes (n : ℕ) : 
  ∃ primes : list ℕ, (∀ p ∈ primes, nat.prime p) ∧ 7 ^ 7 ^ n + 1 = primes.prod ∧ primes.length = 2 * n + 3 :=
sorry

end product_of_primes_l120_120613


namespace find_b_l120_120957

-- Given conditions
def p (x : ℝ) : ℝ := 2 * x - 7
def q (x : ℝ) (b : ℝ) : ℝ := 3 * x - b

-- Assertion we need to prove
theorem find_b (b : ℝ) (h : p (q 3 b) = 3) : b = 4 := 
by
  sorry

end find_b_l120_120957


namespace number_of_ways_to_select_people_l120_120681

theorem number_of_ways_to_select_people : 
  ∀ (total_people : ℕ) (specific_people : ℕ), 
  total_people = 12 → specific_people = 3 → 
  (finset.card { s : finset ℕ // s.card = 5 ∧ s.filter (λ x, x ∈ {1, 2, 3}).card ≤ 2 }) 
  = 756 
:=
by 
  intros total_people specific_people h_total_people h_specific_people
  sorry

end number_of_ways_to_select_people_l120_120681


namespace circle_equation_and_triangle_area_l120_120817

theorem circle_equation_and_triangle_area :
  (∃ (a : ℝ) (M : ℝ → ℝ → Prop), (∀ (x y : ℝ), M x y ↔ (x - a)^2 + y^2 = 1) ∧
   (center := a)
   (l : ℝ → ℝ := λ x, 3 * x - 1)
   (chord_length := λ M, 2 - chord_length^2 = (2 * sqrt 15) / 5)
   (center_below_line := a < 1/3) 
   (a = 1 ∨ a = -1/3)) ∧ 
  (∀ (t : ℝ), -3 ≤ t ∧ t ≤ -1 → 
   (∃(A B : ℝ × ℝ)
     (tangent_A : ℝ → ℝ := λ x, (1 - t^2) / (2 * t) * x + t),
     (tangent_B : ℝ → ℝ := λ x, (1 - (t + 4)^2) / (2 * (t+4)) * x + t + 4), 
    (C := intersection tangent_A tangent_B),
    S : ℝ, S = 4 * (1 - (1 / ((t^2) + 4 * t + 1)) ) ,
    (min_S : S = 16/3 ∧ max_S : S = 6)

end circle_equation_and_triangle_area_l120_120817


namespace complex_number_quadrant_l120_120847

def complex_quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else if z.re > 0 ∧ z.im < 0 then 4
  else 0 -- this case won't occur in this specific problem

theorem complex_number_quadrant :
  let z := (1 : ℂ) + (1:ℂ)*I + (1:ℂ)*I^2 + (1:ℂ)*I^3 + ... + (1:ℂ)*I^2019 + |(3:ℂ) - (4:ℂ)*I| / ((3:ℂ) + (4:ℂ)*I)
  in complex_quadrant z = 4 :=
by
  sorry

end complex_number_quadrant_l120_120847


namespace count_winning_hands_l120_120926

theorem count_winning_hands :
  (nat.choose 15 5) - (nat.choose 7 5) = 2982 := 
by sorry

end count_winning_hands_l120_120926


namespace curved_surface_area_cone_l120_120702

/-- The curved surface area (CSA) of a cone with slant height 10 cm and radius 5 cm
is approximately 157.08 cm².
-/
theorem curved_surface_area_cone {π : ℝ} (hπ : π ≈ 3.14159) (r l : ℝ) (hr : r ≈ 5) (hl : l ≈ 10) :
  (π * r * l) ≈ 157.08 :=
by
  sorry

end curved_surface_area_cone_l120_120702


namespace area_of_right_triangle_with_given_squares_l120_120846

theorem area_of_right_triangle_with_given_squares (A B C : ℝ) (hA : A = 64) (hB : B = 36)
  (hC : C = 121) : ∃ T : ℝ, T = 24 :=
by 
  have h1: sqrt A = 8, { rw [hA, Real.sqrt_square_eq_abs], norm_num, },
  have h2: sqrt B = 6, { rw [hB, Real.sqrt_square_eq_abs], norm_num, },
  use (1/2) * (sqrt A) * (sqrt B),
  rw [h1, h2],
  norm_num,
  sorry

end area_of_right_triangle_with_given_squares_l120_120846


namespace intersection_point_lines_l120_120780

theorem intersection_point_lines :
  ∃ x y : ℚ, (5 * x - 2 * y = 4) ∧ (3 * x + 4 * y = 16) ∧ x = 24 / 13 ∧ y = 34 / 13 :=
by
  use (24 / 13), (34 / 13)
  split
  calc
    5 * (24 / 13) - 2 * (34 / 13) = (5 * 24 - 2 * 34) / 13 : by ring
                           ...    = (120 - 68) / 13        : by norm_num
                           ...    = 52 / 13                : by norm_num
                           ...    = 4                      : by norm_num
  calc
    3 * (24 / 13) + 4 * (34 / 13) = (3 * 24 + 4 * 34) / 13 : by ring
                           ...    = (72 + 136) / 13        : by norm_num
                           ...    = 208 / 13               : by norm_num
                           ...    = 16                     : by norm_num
  split
  rfl
  rfl

end intersection_point_lines_l120_120780


namespace total_spent_l120_120355

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_friends : ℝ := 4
def discounted_price (orig_price : ℝ) (discount : ℝ) : ℝ := orig_price * discount
def total_cost (num_friends : ℝ) (unit_cost : ℝ) : ℝ := num_friends * unit_cost

theorem total_spent :
  total_cost number_of_friends (discounted_price original_price discount_rate) = 40 :=
by
  simp [total_cost, discounted_price, original_price, discount_rate, number_of_friends]
  norm_num
  sorry

end total_spent_l120_120355


namespace determine_b_sign_l120_120643

noncomputable def quadratic (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x - 4

theorem determine_b_sign (b : ℝ) :
  (∀ x, x ≤ -1 → derivative (quadratic x b) x ≤ 0) ∧ (∀ x, x ≥ -1 → derivative (quadratic x b) x ≥ 0) → b > 0 :=
by
  sorry

end determine_b_sign_l120_120643


namespace fraction_identity_l120_120426

theorem fraction_identity (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) : 
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := 
by
  sorry

end fraction_identity_l120_120426


namespace chord_line_eq_l120_120723

-- Define the given ellipse
def ellipse (x y : ℝ) : Prop :=
  (x^2) / 16 + (y^2) / 4 = 1

-- Define the chord's conditions
def midpoint (x1 y1 x2 y2 mx my : ℝ) : Prop :=
  mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2

-- Define the point M that bisects the chord
def point_M (x y : ℝ) : Prop :=
  x = 2 ∧ y = 1

-- The main theorem to prove
theorem chord_line_eq :
  ∃ line : ℝ → ℝ → Prop, (∀ (x y : ℝ), line x y ↔ x + 2y = 4) ∧
  (∃ x1 y1 x2 y2 : ℝ,
    ellipse x1 y1 ∧ ellipse x2 y2 ∧
    midpoint x1 y1 x2 y2 2 1 ∧ 
    (∃ m : ℝ, m = -1 / 2 ∧ line x1 y1 ∧ line x2 y2)) :=
by sorry

end chord_line_eq_l120_120723


namespace Jake_has_9_peaches_l120_120566

variable (Jake Steven Jill : ℕ)
variable (h1 : Steven = 16)
variable (h2 : Jake = Steven - 7)

theorem Jake_has_9_peaches (Jake Steven : ℕ) (h1 : Steven = 16) (h2 : Jake = Steven - 7) : Jake = 9 :=
by
  rw [h1] at h2
  rw [h2]
  rfl
sorry

end Jake_has_9_peaches_l120_120566


namespace imag_conj_z_l120_120074

def complex_eq (z : ℂ) : Prop := (1 + 2 * complex.I) * z = 4 + 3 * complex.I

theorem imag_conj_z :
  ∀ z : ℂ, complex_eq z → complex.im (conj z) = 1 :=
by
  sorry

end imag_conj_z_l120_120074


namespace solve_expression_l120_120473

theorem solve_expression (a b : ℝ) 
  (h₁ : sqrt (a - 2 * b + 4) + (a + b - 5)^2 = 0) :
  4 * sqrt a - sqrt 24 / sqrt b = 2 * sqrt 2 := 
sorry

end solve_expression_l120_120473


namespace average_and_variance_l120_120841

variable (avg8 var8 : ℝ) (new_point : ℝ)

-- The initial conditions
def initial_conditions : Prop :=
  avg8 = 5 ∧ var8 = 2 ∧ new_point = 5

-- The new average and variance to be proven
def new_avg := (avg8 * 8 + new_point) / 9
def new_var := (var8 * 8) / 9

theorem average_and_variance (h : initial_conditions avg8 var8 new_point) : 
  new_avg avg8 new_point = 5 ∧ new_var var8 < 2 :=
by
  sorry

end average_and_variance_l120_120841


namespace distance_between_closest_points_l120_120770

theorem distance_between_closest_points 
  (cx1 cy1 cx2 cy2 : ℝ) (r1 r2 : ℝ)
  (h1 : cx1 = 5) (h2 : cy1 = 5) (h3 : r1 = 5)
  (h4 : cx2 = 22) (h5 : cy2 = 13) (h6 : r2 = 13)
  (hx1 : cy1 = r1) (hx2 : cy2 = r2) :
  (sqrt ((cx2 - cx1)^2 + (cy2 - cy1)^2) - (r1 + r2)) = sqrt 353 - 18 :=
by
  rw [h1, h2, h3, h4, h5, h6, hx1, hx2]
  sorry

end distance_between_closest_points_l120_120770


namespace average_mowing_per_month_l120_120223

theorem average_mowing_per_month (mow_apr_sep mow_oct_mar total_months : ℕ) :
  (mow_apr_sep = 6 * 15) →
  (mow_oct_mar = 6 * 3) →
  (total_months = 12) →
  (mow_apr_sep + mow_oct_mar) / total_months = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.add_comm (6 * 15) (6 * 3)]
  norm_num
  sorry

end average_mowing_per_month_l120_120223


namespace max_value_at_a_l120_120502

noncomputable def f (x : ℝ) := - (2 * fdd1 / 3) * x.sqrt - x^2

theorem max_value_at_a (fdd1 : ℝ) :
  let a := (4 ^ (1 / 3)) / 4 in
  fdd1 = -3 / 2 → ∀ x : ℝ, f x ≤ f a := sorry

end max_value_at_a_l120_120502


namespace plane_MNK_divides_AB_l120_120046

-- Definitions representing the vertices and segment points
variable (A B C D M N K : Point)

-- Conditions as per the problem statement
variables (AM_MD_eq_1_3 BN_NC_eq_1_1 CK_KD_eq_1_2 : Rat)

-- The main theorem to prove
theorem plane_MNK_divides_AB (AM_MD_eq_1_3 : AM / MD = 1 / 3)
                             (BN_NC_eq_1_1 : BN / NC = 1 / 1)
                             (CK_KD_eq_1_2 : CK / KD = 1 / 2) :
                             exists (r : Rat), r = 2 / 3 :=
begin
  sorry -- Proof to be constructed
end

end plane_MNK_divides_AB_l120_120046


namespace proof_problem_l120_120994

variable {a b c : ℝ}

theorem proof_problem (h_cond : 0 < a ∧ a < b ∧ b < c) : 
  a * c < b * c ∧ a + b < b + c ∧ c / a > c / b := by
  sorry

end proof_problem_l120_120994


namespace geometric_sequence_seventh_term_l120_120374

theorem geometric_sequence_seventh_term (r : ℕ) (r_pos : 0 < r) 
  (h1 : 3 * r^4 = 243) : 
  3 * r^6 = 2187 :=
by
  sorry

end geometric_sequence_seventh_term_l120_120374


namespace convex_function_inequality_l120_120591

theorem convex_function_inequality
  {f : ℝ → ℝ} (hf_convex : ∀ (a b c : ℝ), a ≤ b → b ≤ c → (f b - f a) / (b - a) ≤ (f c - f a) / (c - a) ∧ (f c - f a) / (c - a) ≤ (f c - f b) / (c - b))
  (x y : ℕ → ℝ) (h_x_sorted : ∀ i j, i ≤ j → x i ≤ x j) (h_y_sorted : ∀ i j, i ≤ j → y i ≤ y j)
  (h_sum_le : ∀ p : ℕ, p ≤ n → (∑ i in range p, x i) ≤ (∑ i in range p, y i))
  (h_sum_eq : (∑ i in range n, x i) = (∑ i in range n, y i)) :
  (∑ i in range n, f (x i)) ≥ (∑ i in range n, f (y i)) :=
sorry

end convex_function_inequality_l120_120591


namespace find_x_range_l120_120838

-- Given definition for a decreasing function
def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x > f y

-- The main theorem to prove
theorem find_x_range (f : ℝ → ℝ) (h_decreasing : is_decreasing f) :
  {x : ℝ | f (|1 / x|) < f 1} = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
sorry

end find_x_range_l120_120838


namespace problem_l120_120203

open Real
open Matrix

def vec3 := Fin 3 → ℝ

noncomputable def a : vec3 := sorry
noncomputable def b : vec3 := sorry
noncomputable def m : vec3 := ![4, 8, 10]

-- Midpoint condition
def midpoint (u v m : vec3) : Prop := (u + v) / 2 = m

-- Dot product condition
def dot_product (u v : vec3) : ℝ := 
  u 0 * v 0 + u 1 * v 1 + u 2 * v 2

-- Cross product condition
def cross_product (u v : vec3) : vec3 :=
  ![
    u 1 * v 2 - u 2 * v 1,
    u 2 * v 0 - u 0 * v 2,
    u 0 * v 1 - u 1 * v 0
  ]

-- Norm squared of a vector
def norm_squared (u : vec3) : ℝ := 
  dot_product u u

theorem problem :
  midpoint a b m ∧ dot_product a b = 12 ∧ cross_product a b = ![0, 0, 6]
  → norm_squared a + norm_squared b = 696 := by
  sorry

end problem_l120_120203


namespace patient_treatment_volume_l120_120736

noncomputable def total_treatment_volume : ℝ :=
  let drop_rate1 := 15     -- drops per minute for the first drip
  let ml_rate1 := 6 / 120  -- milliliters per drop for the first drip
  let drop_rate2 := 25     -- drops per minute for the second drip
  let ml_rate2 := 7.5 / 90 -- milliliters per drop for the second drip
  let total_time := 4 * 60 -- total minutes including breaks
  let break_time := 4 * 10 -- total break time in minutes
  let actual_time := total_time - break_time -- actual running time in minutes
  let total_drops1 := actual_time * drop_rate1
  let total_drops2 := actual_time * drop_rate2
  let volume1 := total_drops1 * ml_rate1
  let volume2 := total_drops2 * ml_rate2
  volume1 + volume2 -- total volume from both drips

theorem patient_treatment_volume : total_treatment_volume = 566.67 :=
  by
    -- Place the necessary calculation steps as assumptions or directly as one-liner
    sorry

end patient_treatment_volume_l120_120736


namespace hyperbola_eccentricity_l120_120493

theorem hyperbola_eccentricity (a b c : ℝ) (h_asymp : ∀ x, (∃ y1 y2 : ℝ, y1 = (1/3) * x ∧ y2 = -(1/3) * x) → y = y1 ∨ y = y2)
  (h_hyperbola : ∀ x y, x^2/a^2 - y^2/b^2 = 1) :
  (∃ e : ℝ, e = real.sqrt 10 / 3 ∨ e = real.sqrt 10) :=
begin
  sorry
end

end hyperbola_eccentricity_l120_120493


namespace new_girl_weight_l120_120286

theorem new_girl_weight :
  ∃ (W_new : ℝ), 
    (∀ (W : ℝ), (W_new != 50 ∧ (W - 50 + W_new) / 10 = W / 10 + 5)) :=
by
  let W := sorry, -- This step translates the use of \forall in mathlib
  let W_new := 100, -- This step translates the final step of the original problem
  sorry -- This is to complete the proof
end

end new_girl_weight_l120_120286


namespace most_stable_performance_l120_120008

-- Given variances for the students' scores
def variance_A : ℝ := 2.1
def variance_B : ℝ := 3.5
def variance_C : ℝ := 9
def variance_D : ℝ := 0.7

-- Prove that student D has the most stable performance
theorem most_stable_performance : 
  variance_D < variance_A ∧ variance_D < variance_B ∧ variance_D < variance_C := 
  by 
    sorry

end most_stable_performance_l120_120008


namespace relationship_x_y_with_M_l120_120947

def x : ℝ := 1 / (3 - 5 * Real.sqrt 2)
def y : ℝ := 3 + Real.sqrt 2 * Real.pi

def M : set ℝ := {m | ∃ a b : ℚ, m = (a : ℝ) + Real.sqrt 2 * (b : ℝ)}

theorem relationship_x_y_with_M :
  x ∈ M ∧ y ∉ M :=
by
  sorry

end relationship_x_y_with_M_l120_120947


namespace count_visible_factor_numbers_200_to_250_l120_120383

def is_visible_factor_number (n : ℕ) : Prop :=
  let digits := (toString n).toList
  let non_zero_digits := digits.filter (λ d => d ≠ '0')
  (non_zero_digits.all (λ d => n % (d.toNat - '0'.toNat) = 0))

def visible_factor_numbers_in_range (m n : ℕ) : List ℕ :=
  (List.range' m (n - m + 1)).filter is_visible_factor_number

theorem count_visible_factor_numbers_200_to_250 : 
  visible_factor_numbers_in_range 200 250 = List.range' 200 22 
:=
  sorry

end count_visible_factor_numbers_200_to_250_l120_120383


namespace how_many_three_digit_numbers_without_5s_and_8s_l120_120131

def is_valid_hundreds_digit (d : ℕ) : Prop := d ≠ 0 ∧ d ≠ 5 ∧ d ≠ 8
def is_valid_digit (d : ℕ) : Prop := d ≠ 5 ∧ d ≠ 8

theorem how_many_three_digit_numbers_without_5s_and_8s : 
  (∃ count : ℕ, count = 
    (∑ d1 in (finset.range 10).filter is_valid_hundreds_digit, 
      ∑ d2 in (finset.range 10).filter is_valid_digit, 
        ∑ d3 in (finset.range 10).filter is_valid_digit, 1)) = 448 :=
by
  sorry

end how_many_three_digit_numbers_without_5s_and_8s_l120_120131


namespace difference_between_local_and_face_value_l120_120345

def numeral := 657903

def local_value (n : ℕ) : ℕ :=
  if n = 7 then 70000 else 0

def face_value (n : ℕ) : ℕ :=
  n

theorem difference_between_local_and_face_value :
  local_value 7 - face_value 7 = 69993 :=
by
  sorry

end difference_between_local_and_face_value_l120_120345


namespace average_cuts_per_month_l120_120221

theorem average_cuts_per_month :
  (6 * 15 + 6 * 3) / 12 = 9 :=
by
  unfold has_div.div
  unfold has_add.add
  unfold has_mul.mul
  sorry

end average_cuts_per_month_l120_120221


namespace find_inverse_l120_120875

def f (x : ℝ) : ℝ := (x^5 - 1) / 3

theorem find_inverse : ∃ a, f a = -31/96 ∧ a = 1/2 :=
by
  use (1/2)
  split
  apply rfl
  sorry

end find_inverse_l120_120875


namespace expression_for_rth_term_l120_120804

noncomputable def sum_of_n_terms (n : ℕ) : ℕ := 5 * n + 4 * n^2 + 1

theorem expression_for_rth_term (r : ℕ) : 
  let S_r := sum_of_n_terms r in
  let S_r_minus_1 := sum_of_n_terms (r - 1) in
  S_r - S_r_minus_1 = 8 * r :=
by
  sorry

end expression_for_rth_term_l120_120804


namespace blocks_remaining_l120_120988

def initial_blocks : ℕ := 55
def blocks_eaten : ℕ := 29

theorem blocks_remaining : initial_blocks - blocks_eaten = 26 := by
  sorry

end blocks_remaining_l120_120988


namespace correct_number_of_three_digit_numbers_l120_120126

def count_valid_three_digit_numbers : Nat :=
  let hundreds := [1, 2, 3, 4, 6, 7, 9].length
  let tens_units := [0, 1, 2, 3, 4, 6, 7, 9].length
  hundreds * tens_units * tens_units

theorem correct_number_of_three_digit_numbers :
  count_valid_three_digit_numbers = 448 :=
by
  unfold count_valid_three_digit_numbers
  sorry

end correct_number_of_three_digit_numbers_l120_120126


namespace roots_count_eq_six_l120_120110

noncomputable def number_of_roots : ℝ := 
  let I := Icc (-real.sqrt 14) (real.sqrt 14)
  in
  let f := λ x, √(14 - x^2) * (real.sin x - real.cos (2 * x))
  in
  set.finite (set_of (λ x, f x = 0) ∩ I).to_finset.card

theorem roots_count_eq_six : number_of_roots = 6 := by
  sorry

end roots_count_eq_six_l120_120110


namespace roots_polynomial_base_n_representation_l120_120944

theorem roots_polynomial_base_n_representation (n : ℕ) (hn : 8 < n) (hroot : n^2 - (2*n + 1)*n + ((2*n + 1) * n + n) = 0) :
  (n^2 + n = fin.natCoe 110 n) := 
by 
  sorry

end roots_polynomial_base_n_representation_l120_120944


namespace partitions_of_N_at_most_two_parts_eq_floor_div_2_plus_1_l120_120619

theorem partitions_of_N_at_most_two_parts_eq_floor_div_2_plus_1 (N : ℕ) : 
  number_of_partitions_at_most_two_parts N = (N / 2) + 1 :=
sorry

end partitions_of_N_at_most_two_parts_eq_floor_div_2_plus_1_l120_120619


namespace triangle_angle_uncertain_l120_120171

theorem triangle_angle_uncertain
  {A B C D : Type*}
  [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D]
  (AD BD CD : ℝ)
  (hAD : AD > 0) (hBD : BD > 0) (hCD : CD > 0)
  (altitude : ∀ T, T = AD)
  (condition : AD^2 = BD * CD) :
  ¬ (angle_less_than BAC 90 ∨ angle_equal_to BAC 90 ∨ angle_greater_than BAC 90) :=
by
  sorry

end triangle_angle_uncertain_l120_120171


namespace adah_practiced_total_hours_l120_120407

theorem adah_practiced_total_hours :
  let minutes_per_day := 86
  let days_practiced := 2
  let minutes_other_days := 278
  let total_minutes := (minutes_per_day * days_practiced) + minutes_other_days
  let total_hours := total_minutes / 60
  total_hours = 7.5 :=
by
  sorry

end adah_practiced_total_hours_l120_120407


namespace arithmetic_sequence_proof_l120_120556

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}

def arithmetic_seq (a_1 d : ℕ) : (ℕ → ℕ) := λ n, a_1 + (n - 1) * d
def sum_first_n (a_1 d : ℕ) (n : ℕ) : ℕ := n * a_1 + n * (n - 1) / 2 * d

theorem arithmetic_sequence_proof :
  ∀ (a_1 d : ℕ), 
    (arithmetic_seq a_1 d 3 + sum_first_n a_1 d 5 = 12) →
    (arithmetic_seq a_1 d 4 + sum_first_n a_1 d 7 = 24) →
    ((arithmetic_seq a_1 d 5 + sum_first_n a_1 d 9) = 40) :=
by
  sorry

end arithmetic_sequence_proof_l120_120556


namespace find_prime_pair_l120_120349

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_prime_pair (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p > q) (h_prime : is_prime (p^5 - q^5)) : (p, q) = (3, 2) := 
  sorry

end find_prime_pair_l120_120349


namespace exists_line_through_ellipse_diameter_circle_origin_l120_120076

theorem exists_line_through_ellipse_diameter_circle_origin :
  ∃ m : ℝ, (m = (4 * Real.sqrt 3) / 3 ∨ m = -(4 * Real.sqrt 3) / 3) ∧
  ∀ (x y : ℝ), (x^2 + 2 * y^2 = 8) → (y = x + m) → (x^2 + (x + m)^2 = 8) :=
by
  sorry

end exists_line_through_ellipse_diameter_circle_origin_l120_120076


namespace number_of_correct_statements_l120_120060

-- Definitions based on the given conditions
variables (m n : Line) (α β γ : Plane)

-- Given conditions as premises
theorem number_of_correct_statements :
  (α // β) → (β // γ) → (γ // α) →
  (α ⊥ γ) → (β // γ) → (α ⊥ β) →
  (m ⊥ β) → (m ⊥ n) → (n ⊂ β) → (n // β) →
  (3 = 3) :=
by
  sorry

end number_of_correct_statements_l120_120060


namespace inequality_system_solution_l120_120881

theorem inequality_system_solution (a b : ℝ) (h : ∀ x : ℝ, x > -a → x > -b) : a ≥ b :=
by
  sorry

end inequality_system_solution_l120_120881


namespace polynomial_roots_and_coefficients_l120_120312

theorem polynomial_roots_and_coefficients 
  (a b c d e : ℝ)
  (h1 : a = 2)
  (h2 : 256 * a + 64 * b + 16 * c + 4 * d + e = 0)
  (h3 : -81 * a + 27 * b - 9 * c + 3 * d + e = 0)
  (h4 : 625 * a + 125 * b + 25 * c + 5 * d + e = 0) :
  (b + c + d) / a = 151 := 
by
  sorry

end polynomial_roots_and_coefficients_l120_120312


namespace license_plates_count_l120_120866

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_non_zero_multiple_of_3 (n : ℕ) : Prop :=
  n = 3 ∨ n = 6 ∨ n = 9

theorem license_plates_count : 
  (26^3 * 10 * 4 * 3) = 1757600 :=
by
  have letters := 26 ^ 3
  have first_digit := 10
  have second_digit := 4  -- Number of primes less than 10 (2, 3, 5, 7)
  have third_digit := 3   -- Number of non-zero multiples of 3 (3, 6, 9)
  calc
    26^3 * 10 * 4 * 3 = 1757600 : sorry

end license_plates_count_l120_120866


namespace vector_dot_product_l120_120811

variables (a b : ℝ × ℝ)
variables (ha : a = (1, -1)) (hb : b = (-1, 2))

theorem vector_dot_product : 
  ((2 • a + b) • a) = -1 :=
by
  -- This is where the proof would go
  sorry

end vector_dot_product_l120_120811


namespace sum_of_fractions_l120_120424

theorem sum_of_fractions : 
  (7 / 8 + 3 / 4) = (13 / 8) :=
by
  sorry

end sum_of_fractions_l120_120424


namespace pathway_bricks_total_is_280_l120_120191

def total_bricks (n : ℕ) : ℕ :=
  let odd_bricks := 2 * (1 + 1 + ((n / 2) - 1) * 2)
  let even_bricks := 4 * (1 + 2 + (n / 2 - 1) * 2)
  odd_bricks + even_bricks
   
theorem pathway_bricks_total_is_280 (n : ℕ) (h : total_bricks n = 280) : n = 10 :=
sorry

end pathway_bricks_total_is_280_l120_120191


namespace find_sin_GAC_l120_120715

-- Define the points and lengths as given in the problem
variables (A G C : Point)
variables (AB CD EF GH AD BC EH FG AE BF CG DH : ℝ)

-- Define the lengths as provided in the conditions
axiom length_conditions :
  AB = 4 ∧ CD = 4 ∧ EF = 4 ∧ GH = 4 ∧ AD = 2 ∧ BC = 2 ∧ EH = 2 ∧ FG = 2 ∧ AE = 6 ∧ BF = 6 ∧ CG = 6 ∧ DH = 6

-- Define the theorem to be proved
theorem find_sin_GAC (h : length_conditions) : 
  sin_angle G A C = 3 * sqrt 10 / 10 :=
by 
sorry

end find_sin_GAC_l120_120715


namespace interval_contains_root_l120_120000

-- Define the function in question
def func (x : ℝ) : ℝ := 2^x + x

theorem interval_contains_root :
  ∃ x ∈ Ioo (-1 : ℝ) (-1/2 : ℝ), func x = 0 :=
by
  -- Here, we would use the Intermediate Value Theorem and the evaluations at the end points
  -- Since it's just a statement, we write sorry for the proof
  sorry

end interval_contains_root_l120_120000


namespace order_of_constants_l120_120037

noncomputable def f : ℝ → ℝ := sorry

def f' (x : ℝ) := deriv f x

theorem order_of_constants 
  (h0 : ∀ x, 0 < x ∧ x < 2 * Real.pi → f x = f (2 * Real.pi - x))
  (h1 : ∀ x, 0 < x ∧ x < 2 * Real.pi → f x * Real.sin x - f' x * Real.cos x < 0) :
  let a := (1 / 2) * f (Real.pi / 3),
      b := 0,
      c := -(Real.sqrt 3 / 2) * f (7 * Real.pi / 6)
  in a < b ∧ b < c :=
by sorry

end order_of_constants_l120_120037


namespace intersection_at_single_point_l120_120280

variables {P : Type} [euclidean_geometry P]
variables {A B C A0 B0 A1 B1 D T : P}

-- Define points and lines
variables (triangle_ABC : triangle A B C)
variables (angle_bisector_CD : is_angle_bisector C D ∠ ABC)
variables (intersect_AB : line A B ∋ D)
variables (l1 : line A0 C0)
variables (l2 : line B0 C0)
variables (intersect_l1_l2_B1 : l1 ∩ l2 = {B1})
variables (intersect_l1_l2_A1 : l1 ∩ l2 = {A1})

-- Objective statement of the proof
theorem intersection_at_single_point :
  ∃ T : P, collinear T A (intersection_point l1 l2) ∧ 
           collinear T B (intersection_point l1 l2) ∧ 
           collinear T A0 B0 :=
sorry

end intersection_at_single_point_l120_120280


namespace total_spent_l120_120357

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_friends : ℕ := 4

theorem total_spent : (original_price * (1 - discount_rate) * number_of_friends) = 40 := by
  sorry

end total_spent_l120_120357


namespace part1_monotonicity_part2_range_of_a_l120_120029

-- Part (1)
def f1 (x : ℝ) : ℝ := 8 * x - (Real.sin x / (Real.cos x)^3)

theorem part1_monotonicity (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) :
  (0 < x ∧ x < Real.pi / 4 → Deriv f1 x > 0) ∧ (Real.pi / 4 < x ∧ x < Real.pi / 2 → Deriv f1 x < 0) :=
sorry

-- Part (2)
def f2 (a : ℝ) (x : ℝ) : ℝ := a * x - (Real.sin x / (Real.cos x)^3)

theorem part2_range_of_a (a : ℝ) (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) :
  (∀ x, f2 a x < Real.sin (2 * x)) ↔ a ≤ 3 :=
sorry

end part1_monotonicity_part2_range_of_a_l120_120029


namespace total_cost_of_program_l120_120978

theorem total_cost_of_program 
(OS_overhead : ℝ) 
(per_millisecond_cost : ℝ) 
(data_tape_cost : ℝ) 
(time_in_seconds : ℝ)
(total_cost : ℝ) 
(h1 : OS_overhead = 1.07) 
(h2 : per_millisecond_cost = 0.023) 
(h3 : data_tape_cost = 5.35) 
(h4 : time_in_seconds = 1.5) 
(h5 : total_cost = 1.07 + (1500 * 0.023) + 5.35) : 
total_cost = 40.92 := 
by 
  have milliseconds_time : ℝ := time_in_seconds * 1000
  have computer_time_cost : ℝ := milliseconds_time * per_millisecond_cost
  have total_cost_eq : ℝ := OS_overhead + computer_time_cost + data_tape_cost
  rw [h1, h2, h3, h4] at total_cost_eq
  calc total_cost_eq = 40.92 : sorry

end total_cost_of_program_l120_120978


namespace min_frac_sum_pos_real_l120_120589

variable {x y z w : ℝ}

theorem min_frac_sum_pos_real (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) (h_sum : x + y + z + w = 1) : 
  (x + y + z) / (x * y * z * w) ≥ 144 := 
sorry

end min_frac_sum_pos_real_l120_120589


namespace g_at_3_l120_120137

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem g_at_3 : g 3 = 147 :=
by
  -- Proof omitted for brevity
  sorry

end g_at_3_l120_120137


namespace ratio_of_professionals_l120_120545

-- Define the variables and conditions as stated in the problem.
variables (e d l : ℕ)

-- The condition about the average ages leading to the given equation.
def avg_age_condition : Prop := (40 * e + 50 * d + 60 * l) / (e + d + l) = 45

-- The statement to prove that given the average age condition, the ratio is 1:1:3.
theorem ratio_of_professionals (h : avg_age_condition e d l) : e = d + 3 * l :=
sorry

end ratio_of_professionals_l120_120545


namespace inequality_ln_l120_120982

theorem inequality_ln (x : ℝ) (h₁ : x > -1) (h₂ : x ≠ 0) :
    (2 * abs x) / (2 + x) < abs (Real.log (1 + x)) ∧ abs (Real.log (1 + x)) < (abs x) / Real.sqrt (1 + x) :=
by
  sorry

end inequality_ln_l120_120982


namespace periodicity_of_sequence_l120_120092

theorem periodicity_of_sequence :
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, 1 ≤ n → 
  let a : ℕ → ℚ :=
    λ n, match n with
    | 0     => 2
    | n + 1 => (5 * a n - 13) / (3 * a n - 7)
  in a (n + p) = a n :=
by
  sorry

end periodicity_of_sequence_l120_120092


namespace median_inequality_l120_120176

noncomputable def median (a b c : ℝ) : ℝ :=
  (1 / 2) * real.sqrt (2 * b^2 + 2 * c^2 - a^2)

theorem median_inequality {a b c : ℝ} (h : a ≠ b ∨ b ≠ c ∨ c ≠ a)
  (hlta : a + b > c) (hltb : a + c > b) (hltc : b + c > a) :
  2 * (median a b c) > real.sqrt(a * (4 * (b + c - a))) :=
sorry

end median_inequality_l120_120176


namespace sum_of_digits_of_d_l120_120448

noncomputable section

def exchange_rate : ℚ := 8/5
def euros_after_spending (d : ℚ) : ℚ := exchange_rate * d - 80

theorem sum_of_digits_of_d {d : ℚ} (h : euros_after_spending d = d) : 
  d = 135 ∧ 1 + 3 + 5 = 9 := 
by 
  sorry

end sum_of_digits_of_d_l120_120448


namespace quadratic_sum_l120_120296

theorem quadratic_sum (x : ℝ) :
  (∃ a b c : ℝ, 6 * x^2 + 48 * x + 162 = a * (x + b) ^ 2 + c ∧ a + b + c = 76) :=
by
  sorry

end quadratic_sum_l120_120296


namespace hector_jump_more_than_penelope_l120_120610

noncomputable def penelope_waddles_count : ℕ := 50
noncomputable def hector_jumps_count : ℕ := 15
noncomputable def number_of_poles : ℕ := 51
noncomputable def distance_between_first_and_51st_pole : ℝ := 6336 -- feet

def number_of_gaps : ℕ := number_of_poles - 1
def total_waddles : ℕ := penelope_waddles_count * number_of_gaps
def total_jumps : ℕ := hector_jumps_count * number_of_gaps

def penelope_waddle_length : ℝ := distance_between_first_and_51st_pole / total_waddles
def hector_jump_length : ℝ := distance_between_first_and_51st_pole / total_jumps

def length_difference : ℝ := hector_jump_length - penelope_waddle_length

theorem hector_jump_more_than_penelope :
  length_difference = 5.9136 :=
by sorry

end hector_jump_more_than_penelope_l120_120610


namespace no_arrangement_possible_l120_120468

-- State the problem with definitions and conditions
def Domino :=
  { n : ℕ // n <= 6 }

def dominoes : set (Domino × Domino) :=
  { (⟨i, h_i⟩, ⟨j, h_j⟩) | i <= 6 ∧ j <= 6 }

def dominoes_without_six : set (Domino × Domino) :=
  { d ∈ dominoes | d.1.val ≠ 6 ∧ d.2.val ≠ 6 }

def can_arrange_in_line (s : set (Domino × Domino)) : Prop :=
  ∃ (f : ℕ → Dominos) (n : ℕ), ∀ k < n, (f(k), f(k+1)) ∈ s

theorem no_arrangement_possible : ¬ can_arrange_in_line dominoes_without_six :=
sorry

end no_arrangement_possible_l120_120468


namespace sum_in_base_four_l120_120294

theorem sum_in_base_four (m n : ℕ) (h₁ : m = 289) (h₂ : n = 37) :
  nat.to_digits 4 (m + n) = [1, 1, 0, 1, 2] := by
  sorry

end sum_in_base_four_l120_120294


namespace original_ratio_of_boarders_to_day_students_l120_120657

theorem original_ratio_of_boarders_to_day_students (B D : ℕ) (hB : B = 120) (new_boarders : ℕ) (h_new_boarders : new_boarders = 30)
  (new_ratio : ℕ × ℕ) (h_new_ratio : new_ratio = (1, 2)) :
  B.toRat / D.toRat = (1 : ℚ) / (2.5 : ℚ) :=
by
  sorry

end original_ratio_of_boarders_to_day_students_l120_120657


namespace part1_monotonicity_part2_range_of_a_l120_120028

-- Part (1)
def f1 (x : ℝ) : ℝ := 8 * x - (Real.sin x / (Real.cos x)^3)

theorem part1_monotonicity (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) :
  (0 < x ∧ x < Real.pi / 4 → Deriv f1 x > 0) ∧ (Real.pi / 4 < x ∧ x < Real.pi / 2 → Deriv f1 x < 0) :=
sorry

-- Part (2)
def f2 (a : ℝ) (x : ℝ) : ℝ := a * x - (Real.sin x / (Real.cos x)^3)

theorem part2_range_of_a (a : ℝ) (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) :
  (∀ x, f2 a x < Real.sin (2 * x)) ↔ a ≤ 3 :=
sorry

end part1_monotonicity_part2_range_of_a_l120_120028


namespace total_floor_tiles_l120_120375

-- Define the given conditions
def smaller_square_tiles : ℕ := 441
def tile_size : ℕ := 1

-- Prove that the total number of tiles on the floor is 729
theorem total_floor_tiles : 
  let s := Int.sqrt smaller_square_tiles in
  let side_length := s + 2 * 3 in
  side_length * side_length = 729 := by
  sorry

end total_floor_tiles_l120_120375


namespace johns_net_profit_l120_120923

def grossIncome : ℤ := 30000
def carCost : ℤ := 20000
def monthlyMaintenance : ℤ := 300
def annualInsurance : ℤ := 1200
def tireReplacement : ℤ := 400
def tradeInValue : ℤ := 6000
def taxRate : ℝ := 0.15

def totalExpenses : ℤ := (monthlyMaintenance * 12) + annualInsurance + tireReplacement + (carCost - tradeInValue)
def taxes : ℤ := (taxRate * grossIncome).toInt
def netProfit : ℤ := grossIncome - totalExpenses - taxes

theorem johns_net_profit : netProfit = 6300 :=
by 
  sorry  -- Proof is omitted

end johns_net_profit_l120_120923


namespace sequence_bound_l120_120949

theorem sequence_bound (n : ℕ) (h : 0 ≤ n ∧ n ≤ 998) (a : ℕ → ℝ) 
  (h₀ : a 0 = 1994) 
  (h_rec : ∀ k, a (k + 1) = (1994^2) / (a k + 1)) : 
  1994 - n ≤ a n ∧ a n < 1995 - n :=
begin
  sorry
end

end sequence_bound_l120_120949


namespace color_regions_l120_120911

theorem color_regions (n : ℕ) (h : n > 0) :
  ∃ (coloring : ℕ → ℕ), 
  (∀ r1 r2 : ℕ, separated_by_line r1 r2 → coloring r1 ≠ coloring r2) :=
sorry

end color_regions_l120_120911


namespace find_M_l120_120305

theorem find_M (a b c M : ℝ) (h1 : a + b + c = 120) (h2 : a - 9 = M) (h3 : b + 9 = M) (h4 : 9 * c = M) : 
  M = 1080 / 19 :=
by sorry

end find_M_l120_120305


namespace possible_values_of_a_l120_120587

def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 4 then (a * x - 8) else (x^2 - 2 * a * x)

theorem possible_values_of_a (a : ℝ) : (0 < a ∧ a ≤ 2) ↔ (∀ x : ℝ, x < 4 → (f x a)' > 0) ∧ (∀ x : ℝ, x ≥ 4 → (f x a)' ≥ 0) ∧ (4 * a - 8 ≤ 16 - 8 * a) :=
by
  sorry

end possible_values_of_a_l120_120587


namespace integral_of_piecewise_function_l120_120642

noncomputable def f : ℝ → ℝ :=
fun x => if x ≤ 0 then 4 - x else real.sqrt(4 - x^2)

theorem integral_of_piecewise_function :
  ∫ x in (-2 : ℝ)..2, f x = 10 + Real.pi :=
by
  sorry

end integral_of_piecewise_function_l120_120642


namespace exists_coprime_among_five_consecutive_l120_120709

theorem exists_coprime_among_five_consecutive (n : ℤ) : 
  ∃ k ∈ {n, n+1, n+2, n+3, n+4}, ∀ m ∈ {n, n+1, n+2, n+3, n+4}, m ≠ k → Int.gcd k m = 1 := 
by
  sorry

end exists_coprime_among_five_consecutive_l120_120709


namespace katie_speed_l120_120785

theorem katie_speed (eugene_speed : ℝ)
  (brianna_ratio : ℝ)
  (katie_ratio : ℝ)
  (h1 : eugene_speed = 4)
  (h2 : brianna_ratio = 2 / 3)
  (h3 : katie_ratio = 7 / 5) :
  katie_ratio * (brianna_ratio * eugene_speed) = 56 / 15 := 
by
  sorry

end katie_speed_l120_120785


namespace problem_statement_l120_120462

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def f1 (n : ℕ) : ℕ :=
  let r := n % 3
  let sum_digits := sum_of_digits n
  (sum_digits * sum_digits) + (r + 1)

def fk : ℕ → ℕ → ℕ
| 1, n => f1 n
| (k+1), n => fk k (f1 n)

theorem problem_statement : fk 1990 2345 = 3 :=
by
  sorry

end problem_statement_l120_120462


namespace length_of_AC_l120_120916

variable (A B C : Type) -- Assume A, B, C are types representing vertices of the triangle
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] -- Assume they have metric space properties
variables (AB AC BC : ℝ) -- Assume AB, AC, BC are real numbers representing lengths of the sides

-- Conditions
variable (right_triangle : ∀ {A B C : Type}, B ∠ A C = 90) -- Right-angled at B
variable (sin_A : (AB / AC) = 4 / 5) -- Sin A = 4/5
variable (AB_val : AB = 4) -- AB = 4

-- Proof that AC = 5
theorem length_of_AC : AC = 5 :=
by sorry

end length_of_AC_l120_120916


namespace max_students_l120_120309

theorem max_students 
  (x : ℕ) 
  (h_lt : x < 100)
  (h_mod8 : x % 8 = 5) 
  (h_mod5 : x % 5 = 3) 
  : x = 93 := 
sorry

end max_students_l120_120309


namespace matrix_problem_l120_120201

def A : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![20 / 3, 4 / 3],
  ![-8 / 3, 8 / 3]
]
def B : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![0, 0], -- Correct values for B can be computed from conditions if needed
  ![0, 0]
]

theorem matrix_problem (A B : Matrix (Fin 2) (Fin 2) ℚ)
  (h1 : A + B = A * B)
  (h2 : A * B = ![
  ![20 / 3, 4 / 3],
  ![-8 / 3, 8 / 3]
]) :
  B * A = ![
    ![20 / 3, 4 / 3],
    ![-8 / 3, 8 / 3]
  ] :=
sorry

end matrix_problem_l120_120201


namespace monotonicity_of_f_range_of_a_l120_120015

-- Definitions given in the conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x - (Real.sin x) / (Real.cos x)^3

-- Problem 1: Monotonicity of f(x) when a = 8
theorem monotonicity_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) (a : ℝ) (ha : a = 8) :
  (0 < x ∧ x < Real.pi / 4 → (∀ (b : ℝ), (x < b ∧ b < Real.pi / 2) → f a b > f a x)) ∧
  (Real.pi / 4 < x ∧ x < Real.pi / 2 → (∀ (b : ℝ), (Real.pi / 4 < b ∧ b < x) → f a b < f a x)) :=
sorry

-- Problem 2: Range of a such that f(x) < sin 2x for all x in (0, π/2)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x < Real.pi / 2) → f a x < Real.sin (2 * x)) ↔ (a ≤ 3) :=
sorry

end monotonicity_of_f_range_of_a_l120_120015


namespace class_scores_mean_l120_120973

theorem class_scores_mean 
  (F S : ℕ) (Rf Rs : ℚ)
  (hF : F = 90)
  (hS : S = 75)
  (hRatio : Rf / Rs = 2 / 3) :
  (F * (2/3 * Rs) + S * Rs) / (2/3 * Rs + Rs) = 81 := by
    sorry

end class_scores_mean_l120_120973


namespace Tanya_max_candies_l120_120629

theorem Tanya_max_candies (numbers : Finset ℕ) (h_range : ∀ x ∈ numbers, 1 ≤ x ∧ x ≤ 30) 
  (h_size : numbers.card = 30) : 
  ∃ (arrangement : List ℕ), (∀ (i : ℕ), i < 30 → 
    abs ((arrangement.nth_le i (by linarith)) - (arrangement.nth_le ((i + 1) % 30) (by linarith))) ≥ 14) ∧ 
    (∀ (i : ℕ), i < 30 → abs ((arrangement.nth_le i (by linarith)) - (arrangement.nth_le ((i + 1) % 30) (by linarith))) ≤ 15) :=
by
  sorry

end Tanya_max_candies_l120_120629


namespace max_value_of_inverse_l120_120199

noncomputable def f (x y z : ℝ) : ℝ := (1/4) * x^2 + 2 * y^2 + 16 * z^2

theorem max_value_of_inverse (x y z a b c : ℝ) (h : a + b + c = 1) (pos_intercepts : a > 0 ∧ b > 0 ∧ c > 0)
  (point_on_plane : (x/a + y/b + z/c = 1)) (pos_points : x > 0 ∧ y > 0 ∧ z > 0) :
  ∀ (k : ℕ), 21 ≤ k → k < (f x y z)⁻¹ :=
sorry

end max_value_of_inverse_l120_120199


namespace equal_sides_of_hexagon_l120_120557

axiom Point : Type

variables (A B C D E F : Point) (circle : Set Point)
variable [InscribedHexagon : A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∧ E ∈ circle ∧ F ∈ circle]
variable [HexagonCond1 : A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∧ E ∈ circle ∧ F ∈ circle]
variable [HexagonCond2 : A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∧ E ∈ circle ∧ F ∈ circle]
variable [HexagonCond3 : A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∧ E ∈ circle ∧ F ∈ circle]

axiom parallel (p1 p2 : Point) : Prop
axiom length (p1 p2 : Point) : ℝ
axiom area (p1 p2 p3 : Point) : ℝ

variable [ParallelCond1 : parallel A B D E]
variable [ParallelCond2 : parallel B C E F]
variable [ParallelCond3 : parallel C D F A]
variable [AreaCondition : area A B C D E F = 2 * area A C E]
  
theorem equal_sides_of_hexagon :
  length A B = length D E ∧
  length B C = length E F ∧
  length C D = length F A :=
sorry

end equal_sides_of_hexagon_l120_120557


namespace lineup_with_conditional_pair_l120_120377

theorem lineup_with_conditional_pair :
  ∃ n, 
  n = 8 → 
  (∃ pair_in_lineup, 
    pair_in_lineup = 2 → 
    let pairs_as_unit := n - (pair_in_lineup - 1) in 
    ∃ r, 
    r = factorial pairs_as_unit * factorial pair_in_lineup → 
    r = 10080) :=
begin
  sorry
end

end lineup_with_conditional_pair_l120_120377


namespace no_valid_k_values_l120_120760

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def roots_are_primes (k : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 57 ∧ p * q = k

theorem no_valid_k_values : ∀ k : ℕ, ¬ roots_are_primes k := by
  sorry

end no_valid_k_values_l120_120760


namespace hyperbola_eccentricity_range_l120_120849

variables (a b : ℝ) (e : ℝ)

def right_focus_of_ellipse : Prop := 
  let F := (2, 0) in
  (F = (2, 0)) ∧ (16 ≠ 0) ∧ (12 ≠ 0)

def hyperbola_asymptote_constraint : Prop :=
  let F := (2, 0) in
  let d := (2 * b) / (Real.sqrt (b ^ 2 + a ^ 2)) in
  (d < Real.sqrt 3)

theorem hyperbola_eccentricity_range 
  (ha_pos : a > 0) (hb_pos : b > 0) 
  (hf : right_focus_of_ellipse) 
  (hc : hyperbola_asymptote_constraint a b) :
  1 < e ∧ e < 2 :=
sorry

end hyperbola_eccentricity_range_l120_120849


namespace log_expression_defined_l120_120446

theorem log_expression_defined {x : ℝ} (h : x > 1002^1003) :
  ∃ y1 y2 y3 y4, y1 = log 1002 x ∧ 
                 y2 = log 1003 y1 ∧ 
                 y3 = log 1004 y2 ∧ 
                 y4 = log 1005 y3 :=
begin
  sorry
end

end log_expression_defined_l120_120446


namespace siding_cost_l120_120264

-- Define given dimensions and costs
def wall_area (width : ℝ) (height : ℝ) : ℝ :=
  width * height

def roof_area (width : ℝ) (height : ℝ) : ℝ :=
  2 * (width * height)

def total_cost (area_per_section : ℝ) (cost_per_section : ℝ) (total_area : ℝ) : ℝ :=
  let required_sections := (total_area / area_per_section).ceil in
  required_sections * cost_per_section

theorem siding_cost : total_cost 96 27.3 128 = 54.6 := 
  by
  -- all definitions and parameters are given in the statement
  sorry

end siding_cost_l120_120264


namespace initial_paintings_l120_120248

theorem initial_paintings (paintings_per_day : ℕ) (days : ℕ) (total_paintings : ℕ) (initial_paintings : ℕ) 
  (h1 : paintings_per_day = 2) 
  (h2 : days = 30) 
  (h3 : total_paintings = 80) 
  (h4 : total_paintings = initial_paintings + paintings_per_day * days) : 
  initial_paintings = 20 := by
  sorry

end initial_paintings_l120_120248


namespace field_area_valid_values_l120_120600

theorem field_area_valid_values :
  ∀ (S : ℝ) (a : ℝ),
  (10 * 300 * S ≤ 10000) →
  (S > 0) →
  (|6 * |y| + 3| ≥ 3) →
  (2 * |2 * |x| - a| ≤ 9) →
  (a ≥ -4.5) →
  (a ≤ 4.5) →
  (∃ i ∈ {-4, -3, -2, -1, 0, 1, 2, 3, 4} : set ℝ, a = i) :=
by sorry

end field_area_valid_values_l120_120600


namespace nature_of_roots_Q_l120_120088

noncomputable def Q (x : ℝ) : ℝ := x^6 - 4 * x^5 + 3 * x^4 - 7 * x^3 - x^2 + x + 10

theorem nature_of_roots_Q : 
  ∃ (negative_roots positive_roots : Finset ℝ),
    (∀ r ∈ negative_roots, r < 0) ∧
    (∀ r ∈ positive_roots, r > 0) ∧
    negative_roots.card = 1 ∧
    positive_roots.card > 1 ∧
    ∀ r, r ∈ negative_roots ∨ r ∈ positive_roots → Q r = 0 :=
sorry

end nature_of_roots_Q_l120_120088


namespace number_of_knights_l120_120708

-- Define the properties of an individual in the context of knights and liars
inductive Person
| knight : Person
| liar : Person

-- Define the configuration of 80 people around the circular table
def CircularTable : Type := Fin 80 → Person

-- Define the statement that each person declares about the 11 immediate people after them
def declaration (table : CircularTable) (i : Fin 80) : Prop :=
  let next_11 := List.map (λ j => table (⟨(i + j) % 80, sorry⟩)) (List.range 1 12)
  9 ≤ List.count (λ p => p = Person.liar) next_11

-- Definition that the whole table satisfies the declaration constraint
def validTable (table : CircularTable) : Prop :=
  ∀ i : Fin 80, declaration table i

-- Definition to count the number of knights
def countKnights (table : CircularTable) : ℕ :=
  Finset.univ.count (λ i => table i = Person.knight)

theorem number_of_knights {table : CircularTable} (h : validTable table) :
  countKnights table = 20 :=
sorry

end number_of_knights_l120_120708


namespace combined_work_time_l120_120467

theorem combined_work_time (A B C D : ℕ) (hA : A = 10) (hB : B = 15) (hC : C = 20) (hD : D = 30) :
  1 / (1 / A + 1 / B + 1 / C + 1 / D) = 4 := by
  -- Replace the following "sorry" with your proof.
  sorry

end combined_work_time_l120_120467


namespace second_cat_weight_l120_120404

theorem second_cat_weight :
  ∀ (w1 w2 w3 w_total : ℕ), 
    w1 = 2 ∧ w3 = 4 ∧ w_total = 13 → 
    w_total = w1 + w2 + w3 → 
    w2 = 7 :=
by
  sorry

end second_cat_weight_l120_120404


namespace distinct_three_digit_numbers_count_l120_120865

theorem distinct_three_digit_numbers_count : 
  (finset.univ.image (λ (n : finset (fin 4)), n.val)).card = 24 := 
sorry

end distinct_three_digit_numbers_count_l120_120865


namespace positive_number_condition_l120_120391

theorem positive_number_condition (y : ℝ) (h: 0.04 * y = 16): y = 400 := 
by sorry

end positive_number_condition_l120_120391


namespace angle_JIK_right_angle_l120_120580

open EuclideanGeometry

variables {A B C I J K : Point}
variables {Γ Γₐ : Circle}

noncomputable def incircle (ABC : Triangle) : Circle := sorry
noncomputable def excircle (ABC : Triangle) (A : Point) : Circle := sorry
noncomputable def tangent_point (Γ : Circle) (BC : Line) : Point := sorry
noncomputable def intersection_closest_to_A (Γ : Circle) (AJ : Line) (A : Point) : Point := sorry

theorem angle_JIK_right_angle
  (ABC : Triangle)
  (Γ := incircle ABC)
  (Γₐ := excircle ABC A)
  (points : tangent_point Γ (Line.mk B C) = I ∧ tangent_point Γₐ (Line.mk B C) = J)
  (K := intersection_closest_to_A Γ (Line.mk A J) A) :
  ∠JIK = 90 :=
sorry

end angle_JIK_right_angle_l120_120580


namespace find_abc_sum_l120_120627

theorem find_abc_sum :
  ∃ a b c : ℝ, 
    (∀ x : ℝ, f (x + 4) = 4 * x ^ 2 + 9 * x + 5) ∧ 
    (∀ x : ℝ, f x = a * x ^ 2 + b * x + c) ∧ 
    (a + b + c = 14) :=
by
  sorry

end find_abc_sum_l120_120627


namespace bernardo_silvia_probability_l120_120420

/-- This theorem states that if Bernardo randomly picks 3 distinct numbers from {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} and Silvia randomly picks 3 distinct numbers from {1, 2, 3, 4, 5, 6, 7, 8, 9}, both arranging them in descending order to form a three-digit number, then the probability that Bernardo's number is greater than Silvia's number is 217/336. -/
theorem bernardo_silvia_probability :
  let bernardo_set := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let silvia_set := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let bernardo_picks := finset.choose 3 bernardo_set
  let silvia_picks := finset.choose 3 silvia_set
  let bernardo_num := bernardo_picks.map (λ x, -x).sort (≤)
  let silvia_num := silvia_picks.map (λ x, -x).sort (≤)
  let bernardo_larger := bernardo_num > silvia_num
  probability bernardo_larger =
    (217/336 : ℚ) :=
sorry

end bernardo_silvia_probability_l120_120420


namespace rake_yard_alone_time_l120_120183

-- Definitions for the conditions
def brother_time := 45 -- Brother takes 45 minutes
def together_time := 18 -- Together it takes 18 minutes

-- Define and prove the time it takes you to rake the yard alone based on given conditions
theorem rake_yard_alone_time : 
  ∃ (x : ℕ), (1 / (x : ℚ) + 1 / (brother_time : ℚ) = 1 / (together_time : ℚ)) ∧ x = 30 :=
by
  sorry

end rake_yard_alone_time_l120_120183


namespace correct_number_of_three_digit_numbers_l120_120127

def count_valid_three_digit_numbers : Nat :=
  let hundreds := [1, 2, 3, 4, 6, 7, 9].length
  let tens_units := [0, 1, 2, 3, 4, 6, 7, 9].length
  hundreds * tens_units * tens_units

theorem correct_number_of_three_digit_numbers :
  count_valid_three_digit_numbers = 448 :=
by
  unfold count_valid_three_digit_numbers
  sorry

end correct_number_of_three_digit_numbers_l120_120127


namespace number_of_girls_in_colins_class_l120_120538

variables (g b : ℕ)

theorem number_of_girls_in_colins_class
  (h1 : g / b = 3 / 4)
  (h2 : g + b = 35)
  (h3 : b > 15) :
  g = 15 :=
sorry

end number_of_girls_in_colins_class_l120_120538


namespace max_M_squared_l120_120934

theorem max_M_squared (x y : ℝ) (hxy : x ≠ y) (hx : 0 < x) (hy : 0 < y) (h_eq : x^3 + 2013 * y = y^3 + 2013 * x) :
    let M := (Real.sqrt 3 + 1) * x + 2 * y 
    in M^2 ≤ 16104 :=
by {
    sorry
}

end max_M_squared_l120_120934


namespace ratio_a_over_c_l120_120647

variables {a b c x1 x2 : Real}
variables (h1 : x1 + x2 = -a) (h2 : x1 * x2 = b) (h3 : b = 2 * a) (h4 : c = 4 * b)
           (ha_nonzero : a ≠ 0) (hb_nonzero : b ≠ 0) (hc_nonzero : c ≠ 0)

theorem ratio_a_over_c : a / c = 1 / 8 :=
by
  have hc_eq : c = 8 * a := by
    rw [h4, h3]
    simp
  rw [hc_eq]
  field_simp [ha_nonzero]
  norm_num
  sorry -- additional steps if required

end ratio_a_over_c_l120_120647


namespace percent_carbonated_water_first_solution_l120_120746

theorem percent_carbonated_water_first_solution 
  (sol1_lemonade_percent sol2_lemonade_percent sol2_carbonated_percent mix_carbonated_percent portion_sol1 : ℝ) 
  (h_sol1_lemonade : sol1_lemonade_percent = 20)
  (h_sol2_lemonade : sol2_lemonade_percent = 45)
  (h_sol2_carbonated : sol2_carbonated_percent = 55)
  (h_mix_carbonated : mix_carbonated_percent = 60)
  (h_portion_sol1 : portion_sol1 = 0.1999999999999997) :
  (portion_sol1 * (100 - sol1_lemonade_percent) + (1 - portion_sol1) * sol2_carbonated_percent = mix_carbonated_percent) :=
by
  simp only [*, sub_eq_add_neg]
  have eq1 : (100 - sol1_lemonade_percent) = (100 - 20) := by rw [h_sol1_lemonade]
  have eq2 : (1 - portion_sol1) = (0.8) := by norm_num[portion_sol1]
  have eq3 : (portion_sol1) = (0.2)      := by norm_num[portion_sol1]
  rw [eq3, eq2, h_sol2_carbonated, h_mix_carbonated, eq1]
  have : (0.2 * (100 - 20) + 0.8 * 55) = 60 := by norm_num
  exact eq.mp this eq.refl

end percent_carbonated_water_first_solution_l120_120746


namespace polar_to_rectangular_translation_l120_120779

namespace PolarToRectangular

open Real

def polarToRectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * cos θ, r * sin θ)

def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

theorem polar_to_rectangular_translation :
  let p := polarToRectangular 4 (π / 6)
  let t := (2, -1)
  translate p t = (2 * sqrt 3 + 2, 1) := 
by
  sorry

end PolarToRectangular

end polar_to_rectangular_translation_l120_120779


namespace find_y_l120_120138

theorem find_y (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -5) : y = 18 := by
  -- Proof can go here
  sorry

end find_y_l120_120138


namespace ellipse_eccentricity_l120_120489

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h : a > b) (h1: b > 0) (tangent: bx - ay + 2 * a * b = 0) 
  : ℝ := by 
  sorry
  
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : let c := sqrt(a^2 - b^2) 
         in is_tangent (circle (0, 0) a) (line bx - ay + 2 * a * b = 0)) 
  : eccentricity_of_ellipse a b h1 h2 h3 = sqrt(6) / 3 := 
  by sorry

end ellipse_eccentricity_l120_120489


namespace incenter_of_triangle_C_K1_K2_l120_120576

noncomputable theory

open EuclideanGeometry

variables {A B C I K1 K2 : Point}

def incenter (A B C I : Point) : Prop :=
  is_incenter I A B C

def tangency_points (A B C I K1 K2 : Point) : Prop :=
  is_tangency_point K1 (segment B C) (circle I (dist I K1)) ∧
  is_tangency_point K2 (segment A C) (circle I (dist I K1))

theorem incenter_of_triangle_C_K1_K2
  (I_incenter : incenter A B C I)
  (K1_tangent : tangency_points A B C I K1 K2)
  : ∃ J : Point, is_incenter J C K1 K2 ∧ lies_on J (line C I) ∧ J ≠ I :=
sorry

end incenter_of_triangle_C_K1_K2_l120_120576


namespace find_n_l120_120457

theorem find_n (n : ℕ) (h : n * n.factorial + 2 * n.factorial = 5040) : n = 5 :=
sorry

end find_n_l120_120457


namespace tank_capacity_is_correct_l120_120749

-- Definition of the problem conditions
def initial_fraction := 1 / 3
def added_water := 180
def final_fraction := 2 / 3

-- Capacity of the tank
noncomputable def tank_capacity : ℕ := 540

-- Proof statement
theorem tank_capacity_is_correct (x : ℕ) :
  (initial_fraction * x + added_water = final_fraction * x) → x = tank_capacity := 
by
  -- This is where the proof would go
  sorry

end tank_capacity_is_correct_l120_120749


namespace scientific_notation_correct_l120_120242

def num_people : ℝ := 2580000
def scientific_notation_form : ℝ := 2.58 * 10^6

theorem scientific_notation_correct : num_people = scientific_notation_form :=
by
  sorry

end scientific_notation_correct_l120_120242


namespace find_d_l120_120523

theorem find_d (d : ℚ) (h : ∀ x : ℚ, 4*x^3 + 17*x^2 + d*x + 28 = 0 → x = -4/3) : d = 155 / 9 :=
sorry

end find_d_l120_120523


namespace find_sum_of_areas_l120_120365

theorem find_sum_of_areas (diameter : ℝ) (h1 : diameter = 10) 
  (C_on_circle : ∀ x y : ℝ, x^2 + y^2 = (diameter / 2)^2 ∧ x = 0 ∧ y = diameter / 2) 
  (right_angle_at_C : ∃ A B : ℝ × ℝ, A.1 = -diameter / 2 ∧ A.2 = 0 ∧ B.1 = diameter / 2 ∧ B.2 = 0 ∧ ∠ABC = 90) :
  let r := diameter / 2 in
  let area_circle := real.pi * r^2 in
  let area_triangle := (1/2) * diameter/2 * diameter/2 in
  let area_segments := area_circle - area_triangle in
  ∃ a b c : ℕ, a = 25 ∧ b = 0 ∧ c = 0 ∧ a + b + c = 25 :=
by
  sorry

end find_sum_of_areas_l120_120365


namespace scientific_notation_conversion_l120_120236

theorem scientific_notation_conversion (x : ℝ) (h_population : x = 141260000) :
  x = 1.4126 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l120_120236


namespace ratio_a_c_l120_120651

theorem ratio_a_c {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : ∀ x : ℝ, x^2 + a * x + b = 0 → 2 * x^2 + 2 * a * x + 2 * b = 0)
  (h2 : ∀ x : ℝ, x^2 + b * x + c = 0 → x^2 + b * x + c = 0) :
  a / c = 1 / 8 :=
by
  sorry

end ratio_a_c_l120_120651


namespace ring_arrangements_leftmost_digits_l120_120054

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def arrangements_of_six_rings_is (n m : ℕ) : Prop :=
  let total_rings := choose 10 6 * Nat.factorial 6 * choose 10 4
  let leftmost_digits := (total_rings / 10 ^ (Nat.log10 total_rings - 2))
  leftmost_digits = 317

theorem ring_arrangements_leftmost_digits :
  arrangements_of_six_rings_is 10 6 :=
by
  sorry

end ring_arrangements_leftmost_digits_l120_120054


namespace no_perpendicular_vector_exists_l120_120515

variables {V : Type*} [inner_product_space ℝ V]

def non_coplanar (a b c : V) : Prop :=
  ¬(∀ (x y z : ℝ), x • a + y • b + z • c = 0 → (x = 0 ∧ y = 0 ∧ z = 0))

theorem no_perpendicular_vector_exists (a b c : V) 
  (h_non_coplanar : non_coplanar a b c) : 
  ¬ ∃ (d : V), d ≠ 0 ∧ (inner_product_space.has_inner.inner d a = 0) ∧
                            (inner_product_space.has_inner.inner d b = 0) ∧
                            (inner_product_space.has_inner.inner d c = 0) :=
by sorry

end no_perpendicular_vector_exists_l120_120515


namespace possible_values_of_a_l120_120588

def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 4 then (a * x - 8) else (x^2 - 2 * a * x)

theorem possible_values_of_a (a : ℝ) : (0 < a ∧ a ≤ 2) ↔ (∀ x : ℝ, x < 4 → (f x a)' > 0) ∧ (∀ x : ℝ, x ≥ 4 → (f x a)' ≥ 0) ∧ (4 * a - 8 ≤ 16 - 8 * a) :=
by
  sorry

end possible_values_of_a_l120_120588


namespace student_tickets_count_l120_120314

-- Defining the parameters and conditions
variables (A S : ℕ)
variables (h1 : A + S = 59) (h2 : 4 * A + 5 * S / 2 = 222.50)

-- The statement to prove
theorem student_tickets_count : S = 9 :=
by
  sorry

end student_tickets_count_l120_120314


namespace equilateral_triangle_of_condition_l120_120059

theorem equilateral_triangle_of_condition (a b c : ℝ) (h : a^2 + 2 * b^2 + c^2 - 2 * b * (a + c) = 0) : a = b ∧ b = c :=
by
  /- Proof goes here -/
  sorry

end equilateral_triangle_of_condition_l120_120059


namespace number_of_roots_l120_120104

noncomputable theory
open Real

def domain (x : ℝ) := abs x ≤ sqrt 14
def equation (x : ℝ) := sin x - cos (2 * x) = 0

theorem number_of_roots : 
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, domain x) ∧ (∀ x ∈ xs, equation x) ∧ xs.card = 6 := 
by sorry

end number_of_roots_l120_120104


namespace max_value_of_m_l120_120844

noncomputable def maximum_m (a b : ℝ) (h₁ : a < 0) (h₂ : b > 0)
  (h₃ : ∀ x, -(1/2) < x ∧ x < 1 → ax^2 + bx + 1 > 0)
  (h₄ : ∀ x, x ∈ set.Ici 4 → b * x^2 - m * x - 2 * a ≥ 0) : ℝ :=
5

theorem max_value_of_m {a b : ℝ} (h₁ : a < 0) (h₂ : b > 0)
  (h₃ : ∀ x, -(1/2) < x ∧ x < 1 → ax^2 + bx + 1 > 0)
  (h₄ : ∀ x, x ∈ set.Ici 4 → b * x^2 - m * x - 2 * a ≥ 0) :
  maximum_m a b h₁ h₂ h₃ h₄ = 5 :=
begin
  -- The proof is omitted as requested
  sorry
end

end max_value_of_m_l120_120844


namespace possible_new_perimeters_l120_120371

theorem possible_new_perimeters
  (initial_tiles := 8)
  (initial_shape := "L")
  (initial_perimeter := 12)
  (additional_tiles := 2)
  (new_perimeters := [12, 14, 16]) :
  True := sorry

end possible_new_perimeters_l120_120371


namespace mn_value_l120_120524

axiom like_terms (a b : ℝ) (m n : ℕ) : like_terms (5 * (a ^ 3) * (b ^ n)) (-3 * (a ^ m) * (b ^ 2))

theorem mn_value (h1 : 3 = m) (h2 : n = 2) : m * n = 6 :=
by
  rw [h1, h2]
  exact (3 * 2)
  sorry

end mn_value_l120_120524


namespace sum_evaluation_l120_120955

def g (x : ℚ) : ℚ := x^2 * (1 - x)^2

theorem sum_evaluation : 
  (∑ k in finset.range 2022, (-1)^(k+1) * g((k+1 : ℚ) / 2023)) = (1012 * 1011 / 2023^2)^2 := 
sorry

end sum_evaluation_l120_120955


namespace EFGH_is_parallelogram_l120_120594

variables {A B C D E F G H : Type} [geom_structure : geometry A B C D E F G H]

axiom convex_quadrilateral (ABCD : Set Point) : Convex ABCD

axiom equilateral_triangles (ABE : Triangle) (BCF : Triangle) (CDG : Triangle) (DAH : Triangle) :
  Equilateral ABE ∧ Equilateral BCF ∧ Equilateral CDG ∧ Equilateral DAH

axiom directed_outward (ABE CDG : Triangle) : OutwardDirected ABE ∧ OutwardDirected CDG
axiom directed_inward (BCF DAH : Triangle) : InwardDirected BCF ∧ InwardDirected DAH

theorem EFGH_is_parallelogram :
  Parallelogram (Quadrilateral.mk E F G H) :=
sorry

end EFGH_is_parallelogram_l120_120594


namespace area_of_AFE_l120_120644

noncomputable def point := ℝ × ℝ

def AB_parallel_CD (A B C D: point) :=
  (B.2 = A.2) ∧ (C.2 = D.2)

def is_isosceles_trapezoid (A B C D: point) : Prop :=
  AB_parallel_CD A B C D ∧
  (dist A D = dist B C) ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 6^2) ∧
  ((A.1 - D.1)^2 + (A.2 - D.2)^2 = 5^2) ∧
  (A.1 - B.1 = 0 ∨ B.1 - A.1 = 6) ∧
  (D.1 - A.1 = 5 * real.cos (60 * real.pi / 180))

def reflect_off_CB (A B E: point) : Prop :=
  -- hypothetical condition for reflection, to be specified as needed
  sorry

noncomputable def area_of_triangle (A B C: point) : ℝ :=
  real.abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

theorem area_of_AFE (A B C D E F: point)
  (h1: is_isosceles_trapezoid A B C D)
  (h2: reflect_off_CB A B E)
  (h3: dist A F = 3) 
  : area_of_triangle A F E = 3 * real.sqrt 3 :=
sorry

end area_of_AFE_l120_120644


namespace cost_per_acre_proof_l120_120680

def cost_of_land (tac tl : ℕ) (hc hcc hcp heq : ℕ) (ttl : ℕ) : ℕ := ttl - (hc + hcc + hcp + heq)

def cost_per_acre (total_land : ℕ) (cost_land : ℕ) : ℕ := cost_land / total_land

theorem cost_per_acre_proof (tac tl hc hcc hcp heq ttl epl : ℕ) 
  (h1 : tac = 30)
  (h2 : hc = 120000)
  (h3 : hcc = 20 * 1000)
  (h4 : hcp = 100 * 5)
  (h5 : heq = 6 * 100 + 6000)
  (h6 : ttl = 147700) :
  cost_per_acre tac (cost_of_land tac tl hc hcc hcp heq ttl) = epl := by
  sorry

end cost_per_acre_proof_l120_120680


namespace inequality_proof_l120_120951

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable (ha : ∀ i, 0 < a i)

theorem inequality_proof :
  2 * ∑ i in Finset.range n, (a i)^2 ≥ 
    (∑ i in Finset.range (n - 1), a i * a (i + 1 % n) + a (n - 1) * (a 0 + a 1)) := sorry

end inequality_proof_l120_120951


namespace min_cells_marked_l120_120331

/-- The minimum number of cells that need to be marked in a 50x50 grid so
each 1x6 vertical or horizontal strip has at least one marked cell is 416. -/
theorem min_cells_marked {n : ℕ} : n = 416 → 
  (∀ grid : Fin 50 × Fin 50, ∃ cells : Finset (Fin 50 × Fin 50), 
    (∀ (r c : Fin 50), (r = 6 * i + k ∨ c = 6 * i + k) →
      (∃ (cell : Fin 50 × Fin 50), cell ∈ cells)) →
    cells.card = n) := 
sorry

end min_cells_marked_l120_120331


namespace problem_l120_120153

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ n

theorem problem (a r : ℝ) (h1 : 0 < a) (h2 : 0 < r)
  (h3 : log (geometric_sequence a r 2) + log (geometric_sequence a r 7) + log (geometric_sequence a r 12) = 6) :
  a * geometric_sequence a r 14 = 10000 :=
by
  sorry

end problem_l120_120153


namespace car_owners_without_motorcycles_l120_120542

theorem car_owners_without_motorcycles:
  ∀ (N C M: ℕ) (hN: N = 500) (hC: C = 450) (hM: M = 80),
  ∃ x: ℕ, x = C - (C + M - N) ∧ x = 420 :=
by
  intros N C M hN hC hM
  use C - (C + M - N)
  split
  { rw [hN, hC, hM] }
  { sorry }

end car_owners_without_motorcycles_l120_120542


namespace juan_more_marbles_l120_120773

theorem juan_more_marbles (connie_marbles : ℕ) (juan_marbles : ℕ) (h1 : connie_marbles = 323) (h2 : juan_marbles = 498) :
  juan_marbles - connie_marbles = 175 :=
by
  -- Proof goes here
  sorry

end juan_more_marbles_l120_120773


namespace number_of_valid_rods_l120_120188

theorem number_of_valid_rods :
  let rods := List.range 51 \ {0}
  let placed_rods := {5, 12, 25}
  let remaining_rods := rods.diff placed_rods.to_list
  let valid_rods := remaining_rods.filter (λ l, 9 ≤ l ∧ l ≤ 41)
  valid_rods.length = 33 :=
by
  sorry

end number_of_valid_rods_l120_120188


namespace group_morph_or_involution_l120_120575

variables {G : Type*} [Group G]

theorem group_morph_or_involution (f : G → G) 
  (h_morph : ∀ x y : G, f (x * y) = f x * f y)
  (h_choice : ∀ x : G, f x = x ∨ f x = x⁻¹)
  (h_no_order_4 : ∀ x : G, ¬ (x^4 = 1 ∧ ¬ x^2 = 1)) :
  (∀ x : G, f x = x) ∨ (∀ x : G, f x = x⁻¹) :=
sorry

end group_morph_or_involution_l120_120575


namespace solve_for_x_l120_120990

theorem solve_for_x :
  let x := (√(3^2 + 4^2)) / (√(4 + 1))
  in x = √5 :=
by
  sorry

end solve_for_x_l120_120990


namespace S_21_eq_262_max_term_c_n_l120_120825

-- Definition of the sequence and sum conditions
def a : ℕ → ℕ
| 1 := 2
| n + 1 := sorry -- because we don't need internal sequence for proving Sn

def S : ℕ → ℕ
| 0 := 0
| 1 := 2
| n := S n = (n+1)*(n+2) - S (n-1)

-- Definition of the sequence c
def c (n : ℕ) : ℚ := (-1)^n * n / S n

-- Proof that given the conditions, S 21 is 262
theorem S_21_eq_262 : S 21 = 262 :=
sorry

-- Proof for the maximum term of the sequence c_n is 1/5
theorem max_term_c_n : ∃ n, c n = 1 / 5 ∧ ∀ m, c m ≤ c n :=
sorry

end S_21_eq_262_max_term_c_n_l120_120825


namespace number_of_roots_l120_120111

noncomputable def roots_equation_count : ℝ :=
  let interval := Icc (-real.sqrt 14) (real.sqrt 14)
  ∑ x in interval, (if sqrt (14 - x^2) * (sin x - cos (2 * x)) = 0 then 1 else 0)

theorem number_of_roots : roots_equation_count = 6 := by {
  sorry
}

end number_of_roots_l120_120111


namespace first_day_exceeds_250_l120_120185

def paperclips_on_day (n : ℕ) : ℕ := 5 * 2^(n - 1)

theorem first_day_exceeds_250 : ∃ n : ℕ, n ≥ 1 ∧ paperclips_on_day n > 250 ∧ 
  ( ∀ k : ℕ, k < n → paperclips_on_day k ≤ 250 ) :=
by {
  use 8,
  split,
  { exact nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.le_refl 0))))))), },
  split,
  { 
    have h : 5 * 2^(7) > 250,
    calc 5 * 2 ^ 7 = 5 * 128 : by rw nat.pow_succ
                      ... = 640 : by norm_num
                      ... > 250 : by norm_num,
    exact h,
  },
  { 
    intro k,
    intro h_lt,
    have k_le_eq_7_expr : k ≤ 8 - 1,
    calc k ≤ 7 : h_lt
        ... = 8 - 1 : rfl,
    cases k,
    { exact le_succ 320 },
    cases k,
    exact le_succ 320,
    cases k,
    exact le_succ 320,
    cases k,
    exact le_succ 320,
    cases k,
    exact le_succ 320,
    cases k,
    exact le_succ 320,
    cases k,
    exact le_succ 320,
    cases k,
    exact le_succ 320,
    { exact le_succ 320 }
  )
}

end first_day_exceeds_250_l120_120185


namespace John_to_floor_pushups_l120_120924

theorem John_to_floor_pushups:
  let days_per_week := 5
  let reps_per_day := 1
  let total_reps_per_stage := 15
  let stages := 3 -- number of stages: wall, high elevation, low elevation
  let total_days_needed := stages * total_reps_per_stage
  let total_weeks_needed := total_days_needed / days_per_week
  total_weeks_needed = 9 := by
  -- Here we will define the specifics of the proof later.
  sorry

end John_to_floor_pushups_l120_120924


namespace range_of_function_l120_120297

theorem range_of_function : 
  (⋃ x : ℝ, { y | y = √3 * sin (2 * x) + 2 * cos (x) ^ 2 - 1 }) = Icc (-2 : ℝ) 2 :=
sorry

end range_of_function_l120_120297


namespace exponential_function_decrease_l120_120641

theorem exponential_function_decrease {f : ℝ → ℝ} (h1 : ∀ x, f'' x + f x < 0) :
  e^2 * f 2018 < f 2016 :=
sorry

end exponential_function_decrease_l120_120641


namespace scientific_notation_of_population_l120_120229

theorem scientific_notation_of_population :
  (141260 : ℝ) = 1.4126 * 10^5 :=
sorry

end scientific_notation_of_population_l120_120229
