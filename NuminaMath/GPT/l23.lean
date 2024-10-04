import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.EuclideanDomain.Basic
import Mathlib.Algebra.GeometricSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Probability.Basic
import Mathlib.Data.Fin
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Interval.Basic
import Mathlib.Geometry.Euclidean.Circle
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.Logic.Basic
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import tactic

namespace sequence_limit_l23_23934

open Real

theorem sequence_limit :
  (∀ n : ℕ, 1 + 2 + 3 + ... + n = n * (n + 1) / 2) →
  filter.tendsto (λ n : ℕ, ((n * (n + 1) / 2) / (sqrt (9 * n^4 + 1)))) filter.at_top (𝓝 (1 / 6)) :=
by
  intros h
  -- rest of the proof
  sorry

end sequence_limit_l23_23934


namespace determine_f2_values_l23_23422

-- Define the function and conditions
def f (x : ℝ) : ℝ := sorry -- To be defined

theorem determine_f2_values :
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 6 * f x * y) →
  (f 2 = 0 ∨ f 2 = 4) ∧
  (f 2 = 0 ∨ f 2 = 4) →
  let n := 2 in
  let s := 0 + 4 in
  n * s = 8 :=
by
  sorry

end determine_f2_values_l23_23422


namespace A_false_B_true_C_true_D_true_l23_23152

theorem A_false :
  ¬ ∃ x, ∀ y = (x^2 + 1) / x, y = 2 :=
by
  sorry

theorem B_true (x : ℝ) (h : x > 1) :
  (∀ y, y = 2*x + 4 / (x - 1) - 1 → y ≥ 4 * real.sqrt 2 + 1) :=
by
  sorry

theorem C_true (x y : ℝ) (h : x + 2 * y = 3 * x * y) (hx : 0 < x) (hy : 0 < y) :
  (2 * x + y ≥ 3) :=
by
  sorry

theorem D_true (x y : ℝ) (h : 9 * x^2 + y^2 + x * y = 1) :
  ∃ c, c = (3 * x + y) ∧ c ≤ (2 * real.sqrt 21 / 7) :=
by
  sorry

end A_false_B_true_C_true_D_true_l23_23152


namespace largest_prime_factor_1547_l23_23909

theorem largest_prime_factor_1547 : ∃ p, Nat.Prime p ∧ p ∣ 1547 ∧ ∀ q, Nat.Prime q ∧ q ∣ 1547 → q ≤ p :=
by
  let p := 17
  have h1 : Nat.Prime 7 := sorry
  have h2 : 1547 % 7 = 0 := sorry
  have h3 : 1547 / 7 = 221 := sorry
  have h4 : 221 = 13 * 17 := sorry
  have h5 : Nat.Prime 13 := sorry
  have h6 : Nat.Prime 17 := sorry
  use p
  split
  · exact h6
  split
  · exact sorry
  intro q
  intro hq
  have hprime_q := hq.1
  have hdiv_q := hq.2
  contrapose!
  intro h
  exact sorry

end largest_prime_factor_1547_l23_23909


namespace exists_v_min_norm_l23_23418

def smallest_value_norm (v : ℝ × ℝ) : Prop :=
  ⟪∥v + ⟨4, 2⟩∥ = 10 ∧ ∥v∥ = 10 - 2 * Real.sqrt 5⟫

theorem exists_v_min_norm : ∃ v : ℝ × ℝ, smallest_value_norm v :=
  sorry

end exists_v_min_norm_l23_23418


namespace solve_equation1_solve_equation2_l23_23839

theorem solve_equation1 : ∀ x : ℝ, (x + 5) ^ 2 = 16 ↔ (x = -9 ∨ x = -1) := by
  intro x
  split
  { intro h
    rw pow_two at h
    rw eq_comm at h
    have h' : (x + 5) * (x + 5) = (4 * 4) := h
    rcases (mul_eq_mul_left_iff.mp h') with (hx | hx)
    { rw (add_eq_add_iff_eq_eq.mp hx).right
      right }
    { rw (add_eq_add_iff_eq_eq.mp hx).left
      left }
    exact zero_ne_one.symm }

theorem solve_equation2 : ∀ x : ℝ, x ^ 2 - 4 * x - 12 = 0 ↔ (x = 6 ∨ x = -2) := by
  intro x
  split
  { intro h
    rw eq_comm, rw eq_zero_iff_factor_eq_zero
    rcases (factors_eq_zero_iff.mp h) with (hx | hx)
    { rw (eq_zero_iff_eq.mp hx) }
    { rw (eq_zero_iff_eq.mp hx) }
    left }
    right }
  { intro h
    rcases h with (hx | hx)
    { rw hx
      simp }
    { rw hx
      simp }
  }

end solve_equation1_solve_equation2_l23_23839


namespace Jill_llamas_count_l23_23789

theorem Jill_llamas_count :
  let initial_pregnant_with_one_calf := 9
  let initial_pregnant_with_twins := 5
  let total_calves_born := (initial_pregnant_with_one_calf * 1) + (initial_pregnant_with_twins * 2)
  let calves_after_trade := total_calves_born - 8
  let initial_pregnant_lamas := initial_pregnant_with_one_calf + initial_pregnant_with_twins
  let total_lamas_after_birth := initial_pregnant_lamas + total_calves_born
  let lamas_after_trade := total_lamas_after_birth - 8 + 2
  let lamas_sold := lamas_after_trade / 3
  let final_lamas := lamas_after_trade - lamas_sold
  final_lamas = 18 :=
by
  sorry

end Jill_llamas_count_l23_23789


namespace quadrilateral_existence_a_quadrilateral_existence_b_quadrilateral_existence_c_l23_23233

-- Part (a)
theorem quadrilateral_existence_a 
  (A B C D : Type) [EuclideanGeometry A] 
  (AB AD AC : ℝ)
  (angleDiffBD : ℝ) 
  (hAB : 0 < AB)
  (hAD : 0 < AD)
  (hAC : 0 < AC) 
  (hanglesBD : -π < angleDiffBD ∧ angleDiffBD < π) :
  ∃ ABCD : Quadrilateral, 
    length ABCD.AB = AB ∧
    length ABCD.AD = AD ∧
    length ABCD.AC = AC ∧
    ∠ ABCD.B - ∠ ABCD.D = angleDiffBD ∧
    bisects ABCD.AC ABCD.ABC ∧
    bisects ABCD.AC ABCD.ADC := 
sorry

-- Part (b)
theorem quadrilateral_existence_b 
  (A B C D : Type) [EuclideanGeometry A] 
  (BC CD : ℝ)
  (ratioAB_AD angleDiffBD : ℝ) 
  (hBC : 0 < BC)
  (hCD : 0 < CD)
  (hratio : 0 < ratioAB_AD)
  (hanglesBD : -π < angleDiffBD ∧ angleDiffBD < π) :
  ∃ ABCD : Quadrilateral, 
    length ABCD.BC = BC ∧
    length ABCD.CD = CD ∧
    length ABCD.AB / length ABCD.AD = ratioAB_AD ∧
    ∠ ABCD.B - ∠ ABCD.D = angleDiffBD ∧
    bisects ABCD.AC ABCD.ABC ∧
    bisects ABCD.AC ABCD.ADC := 
sorry

-- Part (c)
theorem quadrilateral_existence_c 
  (A B C D : Type) [EuclideanGeometry A] 
  (AB AD AC ratioBC_CD : ℝ)
  (hAB : 0 < AB)
  (hAD : 0 < AD)
  (hAC : 0 < AC)
  (hratio : 0 < ratioBC_CD) :
  ∃ ABCD : Quadrilateral, 
    length ABCD.AB = AB ∧
    length ABCD.AD = AD ∧
    length ABCD.AC = AC ∧
    length ABCD.BC / length ABCD.CD = ratioBC_CD ∧
    bisects ABCD.AC ABCD.ABC ∧
    bisects ABCD.AC ABCD.ADC := 
sorry

end quadrilateral_existence_a_quadrilateral_existence_b_quadrilateral_existence_c_l23_23233


namespace legs_inequality_l23_23096

-- Definitions of the two right triangles with equal hypotenuses
variables {A B C A_1 B_1 C_1 : Type}
variable [metric_space A]
variable [metric_space B]
variable [metric_space C]
variable [metric_space A_1]
variable [metric_space B_1]
variable [metric_space C_1]
variable (AB A_1B_1 AC BC A_1C_1 B_1C_1 : ℝ)

-- Given conditions
axiom hypotenuse_eq : AB = A_1B_1

-- Proof problem statement
theorem legs_inequality :
  AB = A_1B_1 → (AC ≤ A_1C_1 ∨ BC ≤ B_1C_1) :=
begin
  intros,
  sorry,
end

end legs_inequality_l23_23096


namespace dilation_matrix_l23_23260

open Matrix

theorem dilation_matrix (k : ℝ) (hk : k = 5) : 
  ∃ (A : Matrix (Fin 2) (Fin 2) ℝ), A = ![![5, 0], ![0, 5]] :=
by
  use ![![5, 0], ![0, 5]]
  sorry

end dilation_matrix_l23_23260


namespace find_QD_l23_23515

noncomputable theory

open Real

section

variables {E D F Q : Point}
variables (QE QF QD : ℝ)

-- Define the conditions as given in the problem.
def right_triangle (E D F : Point) : Prop :=
  (angle E D F) = π / 2

def point_inside_triangle (Q E D F : Point) : Prop :=
  (angle E Q F) = 2 * π / 3 ∧
  (angle F Q D) = 2 * π / 3 ∧
  (angle D Q E) = 2 * π / 3

def specific_lengths (Q E F : Point) (QE QF : ℝ) : Prop :=
  dist Q E = QE ∧ dist Q F = QF

-- Construct the main theorem
theorem find_QD
  (h_triangle : right_triangle E D F)
  (h_point : point_inside_triangle Q E D F)
  (h_lengths : specific_lengths Q E F 8 12) :
  dist Q D = 16 :=
sorry

end

end find_QD_l23_23515


namespace people_who_like_both_l23_23750

-- Conditions
variables (total : ℕ) (a : ℕ) (b : ℕ) (none : ℕ)
-- Express the problem
theorem people_who_like_both : total = 50 → a = 23 → b = 20 → none = 14 → (a + b - (total - none) = 7) :=
by
  intros
  sorry

end people_who_like_both_l23_23750


namespace initial_legos_500_l23_23786

-- Definitions and conditions from the problem
def initial_legos (x : ℕ) : Prop :=
  let used_pieces := x / 2
  let remaining_pieces := x - used_pieces
  let boxed_pieces := remaining_pieces - 5
  boxed_pieces = 245

-- Statement to be proven
theorem initial_legos_500 : initial_legos 500 :=
by
  -- Proof goes here
  sorry

end initial_legos_500_l23_23786


namespace y2_is_quadratic_l23_23540

def quadratic_form (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ p q r : ℝ, p ≠ 0 ∧ ∀ x : ℝ, f x = quadratic_form p q r x

def y1 (x : ℝ) : ℝ := 2 * x + 1
def y2 (x : ℝ) : ℝ := -5 * x^2 - 3
def y3 (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def y4 (x : ℝ) : ℝ := x^3 + x + 1

theorem y2_is_quadratic : is_quadratic y2 :=
begin
  -- proof goes here
  sorry
end

end y2_is_quadratic_l23_23540


namespace BO_OE_ratio_correct_l23_23295

-- Definitions from the conditions
variable {A B C D O E : Point}
variable (ABCD : parallelogram A B C D) (angle_B : ∠B = 60)
variable (O_circumcenter : circumcenter O A B C)
variable (E_on_ext_angle_bisector : E = point_of_intersection_of_BO_with_exterior_angle_bisector_D_line O B D)
variable (BO_OE_ratio : ratio (length (segment B O)) (length (segment O E)) = 1 / 2)

-- The theorem to prove
theorem BO_OE_ratio_correct :
  ∀ (A B C D O E : Point) (ABCD : parallelogram A B C D) (angle_B : ∠B = 60)
  (O_circumcenter : circumcenter O A B C)
  (E_on_ext_angle_bisector : E = point_of_intersection_of_BO_with_exterior_angle_bisector_D_line O B D),
  ratio (length (segment B O)) (length (segment O E)) = 1 / 2 := by
  sorry

end BO_OE_ratio_correct_l23_23295


namespace days_from_thursday_l23_23521

theorem days_from_thursday (n : ℕ) (h : n = 53) : 
  (n % 7 = 4) ∧ (n % 7 = 4 → "Thursday" + 4 days = "Monday") :=
by 
  have h1 : n % 7 = 4 := by sorry
  have h2 : "Thursday" + 4 days = "Monday" := by sorry
  exact ⟨h1, h2 h1⟩

end days_from_thursday_l23_23521


namespace birch_trees_probability_l23_23192

theorem birch_trees_probability :
  ∃ (m n : ℕ), m / n = 2 / 95 ∧ (∀ div : ℕ, div ∣ m → div ∣ n → div = 1) :=
begin
  let total_trees := 15,
  let non_birch_trees := 9,
  let birch_trees := 6,
  -- Total ways to arrange birch trees without restrictions
  let total_arrangements := nat.choose total_trees birch_trees,
  -- Number of ways to place birch trees such that no two are adjacent
  let non_adjacent_ways := nat.choose (non_birch_trees + 1) birch_trees,
  -- The required probability
  let prob := non_adjacent_ways / total_arrangements,
  use (2, 95),
  split,
  have h1: 2 / 95 = prob, sorry,
  exact h1,
  -- Prove that 2 and 95 are relatively prime
  intros d h2 h3,
  sorry
end

end birch_trees_probability_l23_23192


namespace roots_product_sum_l23_23425

theorem roots_product_sum (p q r : ℂ) (h : (6 : ℂ) * (X : Polynomial ℂ)^3 - 9 * X^2 + 17 * X - 12 = 0) 
  (hp : Polynomial.C 6 * X^3 + Polynomial.C (-9) * X^2 + Polynomial.C 17 * X + Polynomial.C (-12) = 0) :
  p * q + q * r + r * p = 17 / 6 :=
by
  sorry

end roots_product_sum_l23_23425


namespace greatest_prime_factor_22_20_l23_23746

def double_factorial (x : ℕ) : ℕ :=
  if x = 0 then 1 else if x = 1 then 1 else x * double_factorial (x - 2)

def product_even (n : ℕ) : ℕ := 
  if n % 2 = 0 then double_factorial n else 1

theorem greatest_prime_factor_22_20 : 
  ∃ p : ℕ, nat.prime p ∧ p = 23 ∧ p = nat.greatest_prime_factor (product_even 22 + product_even 20) :=
sorry

end greatest_prime_factor_22_20_l23_23746


namespace intersection_of_M_and_N_l23_23359

-- Define sets M and N
def M := {x : ℝ | (x + 2) * (x - 1) < 0}
def N := {x : ℝ | x + 1 < 0}

-- State the theorem for the intersection M ∩ N
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < -1} :=
sorry

end intersection_of_M_and_N_l23_23359


namespace base_five_to_base_ten_l23_23906

theorem base_five_to_base_ten : 
  let b := 5 in 
  let x := 123 % b in
  let y := (123 / b) % b in
  let z := (123 / b) / b in
  z * b^2 + y * b + x = 38 :=
by {
  let b := 5,
  let x := 123 % b,  -- least significant digit
  let y := (123 / b) % b,  -- middle digit
  let z := (123 / b) / b,  -- most significant digit
  have hx : x = 3 := by norm_num,  -- 123 % 5 = 3
  have hy : y = 2 := by norm_num,  -- (123 / 5) % 5 = 2
  have hz : z = 1 := by norm_num,  -- (123 / 5) / 5 = 1
  rw [hx, hy, hz],
  norm_num
}

end base_five_to_base_ten_l23_23906


namespace brendan_taxes_correct_l23_23974

-- Definitions based on conditions
def hourly_wage : ℝ := 6
def shifts : (ℕ × ℕ) := (2, 8)
def additional_shift : ℕ := 12
def tip_rate : ℝ := 12
def tax_rate : ℝ := 0.20
def tip_reporting_fraction : ℝ := 1 / 3

-- Calculation based on conditions
noncomputable def total_hours : ℕ := (shifts.1 * shifts.2) + additional_shift
noncomputable def wage_income : ℝ := hourly_wage * total_hours
noncomputable def total_tips : ℝ := tip_rate * total_hours
noncomputable def reported_tips : ℝ := total_tips * tip_reporting_fraction
noncomputable def total_reported_income : ℝ := wage_income + reported_tips
noncomputable def taxes_paid : ℝ := total_reported_income * tax_rate

-- The proof problem statement
theorem brendan_taxes_correct : taxes_paid = 56 := by {
  sorry
}

end brendan_taxes_correct_l23_23974


namespace sequence_exceeds_one_l23_23878

theorem sequence_exceeds_one (k : ℝ) (h₀ : 0 < k) (h1 : k < 1) :
  ∀ n : ℕ, 1 ≤ n → (seq : ℕ → ℝ) (seq 1 = 1 + k) (seq (n + 1) = 1 / (seq n) + k) → seq n > 1 :=
by
  sorry

end sequence_exceeds_one_l23_23878


namespace cannot_determine_position_l23_23541

-- Define the conditions
def east_longitude_122_north_latitude_43_6 : Prop := true
def row_6_seat_3_in_cinema : Prop := true
def group_1_in_classroom : Prop := false
def island_50_nautical_miles_north_northeast_another : Prop := true

-- Define the theorem
theorem cannot_determine_position :
  ¬ ((east_longitude_122_north_latitude_43_6 = false) ∧
     (row_6_seat_3_in_cinema = false) ∧
     (island_50_nautical_miles_north_northeast_another = false) ∧
     (group_1_in_classroom = true)) :=
by
  sorry

end cannot_determine_position_l23_23541


namespace net_effect_on_sale_l23_23917

theorem net_effect_on_sale (P Q : ℝ)  (h₁ : (0.8 * P * 1.8 * Q) - (P * Q) = 0.44 * P * Q) : 
  (P - 0.2 * P) * (Q + 0.8 * Q) - P * Q = 0.44 * P * Q := 
by
  have hp : 0.8 * P = P - 0.2 * P := by linarith
  have hq : 1.8 * Q = Q + 0.8 * Q := by linarith
  rw [hp, hq] at h₁
  exact h₁

end net_effect_on_sale_l23_23917


namespace find_a_l23_23697

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 + a) / x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x - f a x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x

theorem find_a (a : ℝ) : (∃ x0 ∈ Icc 1 Real.exp 1, g a x0 ≥ f a x0) →
  (a ≥ (Real.exp 1)^2 + 1 / (Real.exp 1 - 1)) ∨ (a ≤ -2) :=
by
  sorry

end find_a_l23_23697


namespace sin_x_eq_l23_23368

theorem sin_x_eq : 
  ∀ (a b x : ℝ), 
  a > b ∧ b > 0 ∧ 0 < x ∧ x < π / 2 ∧ cot x = (a^2 - b^2) / (2 * a * b) 
  → sin x = (2 * a * b) / (a^2 + b^2) := 
by sorry

end sin_x_eq_l23_23368


namespace girls_more_than_boys_l23_23503

-- Defining the conditions
def ratio_boys_girls : Nat := 3 / 4
def total_students : Nat := 42

-- Defining the hypothesis based on conditions
theorem girls_more_than_boys : (total_students * ratio_boys_girls) / (3 + 4) * (4 - 3) = 6 := by
  sorry

end girls_more_than_boys_l23_23503


namespace sum_of_intercepts_eq_16_l23_23955

noncomputable def line_eq (x y : ℝ) : Prop :=
  y + 3 = -3 * (x - 5)

def x_intercept : ℝ := 4
def y_intercept : ℝ := 12

theorem sum_of_intercepts_eq_16 : 
  (line_eq x_intercept 0) ∧ (line_eq 0 y_intercept) → x_intercept + y_intercept = 16 :=
by
  intros h
  sorry

end sum_of_intercepts_eq_16_l23_23955


namespace area_enclosed_by_graph_l23_23903

theorem area_enclosed_by_graph : 
  ∃ A : ℝ, (∀ x y : ℝ, |x| + |3 * y| = 9 ↔ (x = 9 ∨ x = -9 ∨ y = 3 ∨ y = -3)) → A = 54 :=
by
  sorry

end area_enclosed_by_graph_l23_23903


namespace area_of_centroid_curve_proof_l23_23021

noncomputable def area_of_centroid_curve (AB : ℝ) (C : ℤ) : ℝ :=
  if AB = 30 ∧ C ≠ 0 then 25 * Real.pi else 0

theorem area_of_centroid_curve_proof :
  ∀ (AB : ℝ) (C : ℤ), AB = 30 → C ≠ 0 → area_of_centroid_curve AB C = 25 * Real.pi :=
by
  intros
  rw area_of_centroid_curve
  split_ifs
  case _ h1 h2 =>
    sorry
  case _ h1' h2' =>
    contradiction

end area_of_centroid_curve_proof_l23_23021


namespace original_fraction_l23_23742

theorem original_fraction (x y : ℝ) (hxy : x / y = 5 / 7)
  (hx : 1.20 * x / (0.90 * y) = 20 / 21) : x / y = 5 / 7 :=
by {
  sorry
}

end original_fraction_l23_23742


namespace right_triangle_hypotenuse_l23_23052

theorem right_triangle_hypotenuse (n : ℤ) : 
  let a := 2 * n + 1
      b := 2 * n * (n + 1)
      c := (a ^ 2 + b ^ 2).sqrt
  in c = 2 * n ^ 2 + 2 * n + 1 :=
sorry

end right_triangle_hypotenuse_l23_23052


namespace julia_money_remaining_l23_23790

theorem julia_money_remaining 
  (initial_amount : ℝ)
  (tablet_percentage : ℝ)
  (phone_percentage : ℝ)
  (game_percentage : ℝ)
  (case_percentage : ℝ) 
  (final_money : ℝ) :
  initial_amount = 120 → 
  tablet_percentage = 0.45 → 
  phone_percentage = 1/3 → 
  game_percentage = 0.25 → 
  case_percentage = 0.10 → 
  final_money = initial_amount * (1 - tablet_percentage) * (1 - phone_percentage) * (1 - game_percentage) * (1 - case_percentage) →
  final_money = 29.70 :=
by
  intros
  sorry

end julia_money_remaining_l23_23790


namespace find_a1_l23_23105

theorem find_a1 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 1 / (1 - a n))
  (h2 : a 2 = 2)
  : a 1 = 1 / 2 :=
sorry

end find_a1_l23_23105


namespace exists_set_of_2014_with_2012_special_l23_23594

def is_special (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a^b + b

theorem exists_set_of_2014_with_2012_special : 
  ∃ S : Finset ℕ, S.card = 2014 ∧ (S.filter is_special).card = 2012 :=
by
  sorry

end exists_set_of_2014_with_2012_special_l23_23594


namespace unknown_subtraction_problem_l23_23745

theorem unknown_subtraction_problem (x y : ℝ) (h1 : x = 40) (h2 : x / 4 * 5 + 10 - y = 48) : y = 12 :=
by
  sorry

end unknown_subtraction_problem_l23_23745


namespace bike_license_combinations_l23_23207

theorem bike_license_combinations : 
  let letters := 3
  let digits := 10
  let total_combinations := letters * digits^4
  total_combinations = 30000 := by
  let letters := 3
  let digits := 10
  let total_combinations := letters * digits^4
  sorry

end bike_license_combinations_l23_23207


namespace projections_concyclic_l23_23406

variables {P O R W X Y Z : Point}
variables {A B C D : ConvexQuadrilateral}
variables {PQ : Line}

-- Definitions from the given problem conditions
def is_convex_quadrilateral (ABCD : ConvexQuadrilateral) : Prop := convex ABCD
def point_on_line (p : Point) (L : Line) : Prop := p ∈ L
def orthogonal_projection (p : Point) (L : Line) : Point := sorry -- Definition placeholder
def is_concyclic {A B C D : Point} : Prop := ∃ (c : Circle), A ∈ c ∧ B ∈ c ∧ C ∈ c ∧ D ∈ c 

-- The statement to prove
theorem projections_concyclic (ABCD : ConvexQuadrilateral) (O : Point) (P Q : Point) (PQ : Line)
  (hconvex : is_convex_quadrilateral ABCD)
  (hO : point_on_line O (diagonal AC ∩ diagonal BD))
  (hP : point_on_line P (line AB ∩ line CD))
  (hQ : point_on_line Q (line BC ∩ line DA))
  (hR : orthogonal_projection O PQ = R) :
  let W := orthogonal_projection R (line AB),
      X := orthogonal_projection R (line BC),
      Y := orthogonal_projection R (line CD),
      Z := orthogonal_projection R (line DA) in
  is_concyclic W X Y Z :=
sorry

end projections_concyclic_l23_23406


namespace cooking_time_l23_23561

theorem cooking_time (total_potatoes cooked_potatoes : ℕ) (cook_time_per_potato : ℕ) 
  (h1 : total_potatoes = 16) (h2 : cooked_potatoes = 7) (h3 : cook_time_per_potato = 5) :
  (total_potatoes - cooked_potatoes) * cook_time_per_potato = 45 :=
by 
  rw [h1, h2, h3]
  norm_num

end cooking_time_l23_23561


namespace none_of_these_conditions_is_40_l23_23753

variable (total people_with_migraine people_with_insomnia people_with_anxiety 
          people_with_migraine_and_insomnia people_with_migraine_and_anxiety 
          people_with_insomnia_and_anxiety people_with_all_three : ℕ)

-- Define the conditions from the problem
def total := 150
def people_with_migraine := 90
def people_with_insomnia := 60
def people_with_anxiety := 30
def people_with_migraine_and_insomnia := 20
def people_with_migraine_and_anxiety := 10
def people_with_insomnia_and_anxiety := 15
def people_with_all_three := 5

-- Define the number of people with none of these conditions
def people_with_none_of_these_conditions : ℕ :=
  total - (people_with_migraine_only + 
          people_with_insomnia_only +
          people_with_anxiety_only +
          people_with_migraine_and_insomnia - people_with_all_three +
          people_with_migraine_and_anxiety - people_with_all_three +
          people_with_insomnia_and_anxiety - people_with_all_three +
          people_with_all_three)

theorem none_of_these_conditions_is_40 :
  people_with_none_of_these_conditions = 40 := by
  sorry  -- proof not needed

end none_of_these_conditions_is_40_l23_23753


namespace prob_k_gnomes_fall_exp_gnomes_falling_l23_23768

variables (n k : ℕ) (p : ℝ)
hypotheses 
  (hn : 0 < n)
  (hp : 0 < p) (hp1 : p < 1)
  (hk : 0 ≤ k) (hk1 : k ≤ n)

open ProbabilityTheory
  
def probability_k_gnomes_fall := 
  p * (1 - p) ^ (n - k)

def expected_gnomes_fall :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p

theorem prob_k_gnomes_fall (hprob : 0 < p ∧ p < 1) : 
  ∀ n k : ℕ, 0 ≤ k ∧ k ≤ n → probability_k_gnomes_fall n k p = p * (1 - p) ^ (n - k) :=
by sorry

theorem exp_gnomes_falling (hprob : 0 < p ∧ p < 1) : 
  ∀ n : ℕ, 0 < n → expected_gnomes_fall n p = n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p :=
by sorry

end prob_k_gnomes_fall_exp_gnomes_falling_l23_23768


namespace counting_integers_between_multiples_l23_23992

theorem counting_integers_between_multiples :
  let smallest_perfect_square_multiple := 900 in
  let smallest_perfect_cube_multiple := 27000 in
  let num_integers := (smallest_perfect_cube_multiple / 30) - (smallest_perfect_square_multiple / 30) + 1 in
  smallest_perfect_square_multiple = 30 * 30 ∧ 
  smallest_perfect_cube_multiple = 900 * 30 ∧ 
  num_integers = 871 :=
by
  sorry

end counting_integers_between_multiples_l23_23992


namespace condition_equivalence_l23_23685

variable (p : ℕ) [Fact (Nat.prime p)]
variable (S : Finset ℕ) (hS : S = Finset.range (p - 1) + 1)
variable (a : ℕ → ℕ) (ha : ∀ i ∈ S, a i ≠ 0 ∧ a i < p)

theorem condition_equivalence :
  (∀ i j ∈ S, a i % p = a j % p) ↔ (∀ M ⊆ S, M.nonempty → ∑ i in M, a i % p ≠ 0) := by
  sorry

end condition_equivalence_l23_23685


namespace maximum_profit_3_le_a_le_5_maximum_profit_f_g_3_le_a_le_5_g_5_lt_a_le_7_l23_23180

noncomputable def f (x a : ℝ) : ℝ := (x - 30)^2 * (x - 10 - a)

theorem maximum_profit_3_le_a_le_5 (a : ℝ) (ha : 3 ≤ a ∧ a ≤ 5) :
    ∀ (x : ℝ), 20 ≤ x ∧ x ≤ 25 → f x a ≤ f 20 a := 
    sorry

theorem maximum_profit_f (a : ℝ) (ha : 5 < a ∧ a ≤ 7) :
    ∀ (x : ℝ), 20 ≤ x ∧ x ≤ 25 → f x a ≤ f ((2 * a + 50) / 3) a :=
    sorry

theorem g_3_le_a_le_5 (a : ℝ) (ha : 3 ≤ a ∧ a ≤ 5) :
    g a = 1000 - 10 * a :=
    sorry

theorem g_5_lt_a_le_7 (a : ℝ) (ha : 5 < a ∧ a ≤ 7) :
    g a = 4 * (a - 20)^2 / 27 :=
    sorry

end maximum_profit_3_le_a_le_5_maximum_profit_f_g_3_le_a_le_5_g_5_lt_a_le_7_l23_23180


namespace quadratic_ineq_solution_range_l23_23711

theorem quadratic_ineq_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2*x^2 - 8*x - 4 - a > 0) ↔ a < -4 :=
by
  sorry

end quadratic_ineq_solution_range_l23_23711


namespace abs_diff_eq_sqrt37_l23_23001

-- Given conditions
variables (p q : ℝ)
variables (h1 : p * q = 6) (h2 : p + q = 7)

-- The statement to be proven
theorem abs_diff_eq_sqrt37 (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = Real.sqrt 37 := 
by 
  sorry

end abs_diff_eq_sqrt37_l23_23001


namespace evaluate_expression_l23_23248

theorem evaluate_expression :
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  (x^2 * y^4 * z * w = - (243 / 256)) := 
by {
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  sorry
}

end evaluate_expression_l23_23248


namespace find_b_l23_23649

-- Define the problem parameters
variables {a b c d : ℝ}
variables {z w : ℂ}
noncomputable theory

-- State the main theorem
theorem find_b (h1 : z * w = 7 + 2 * complex.I)
               (h2 : z.conj + w.conj = -1 + 3 * complex.I)
               (h3 : ∀ x : ℂ, x^4 + a * x^3 + b * x^2 + c * x + d = 0) :
  b = 6 :=
sorry

end find_b_l23_23649


namespace bee_flight_problem_l23_23948

noncomputable def total_bee_flight_distance (r : ℝ) : ℝ :=
  let d := 2 * r
  let c := Real.sqrt (d^2 - 90^2)
  d + c + 90

theorem bee_flight_problem :
  let r := 50
  total_bee_flight_distance r ≈ 233.59 := sorry

end bee_flight_problem_l23_23948


namespace cameron_list_count_l23_23985

theorem cameron_list_count : 
  (∃ (n m : ℕ), n = 900 ∧ m = 27000 ∧ (∀ k : ℕ, (30 * k) ≥ n ∧ (30 * k) ≤ m → ∃ count : ℕ, count = 871)) :=
by
  sorry

end cameron_list_count_l23_23985


namespace arithmetic_progression_a6_l23_23308

theorem arithmetic_progression_a6 (a1 d : ℤ) (h1 : a1 + (a1 + d) + (a1 + 2 * d) = 168) (h2 : (a1 + 4 * d) - (a1 + d) = 42) : 
  a1 + 5 * d = 3 := 
sorry

end arithmetic_progression_a6_l23_23308


namespace solve_for_y_l23_23838

theorem solve_for_y (y : ℝ) : (3^y + 9 = 4 * 3^y - 34) ↔ (y = Real.log 3 (43 / 3)) :=
by
  sorry

end solve_for_y_l23_23838


namespace cameron_list_length_l23_23999

-- Definitions of multiples
def smallest_multiple_perfect_square := 900
def smallest_multiple_perfect_cube := 27000
def multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

-- Problem statement
theorem cameron_list_length :
  ∀ n, 900 ≤ n ∧ n ≤ 27000 ∧ multiple_of_30 n ->
  (871 = (900 - 30 + 1)) :=
sorry

end cameron_list_length_l23_23999


namespace max_unique_distances_in_21_gon_l23_23006

theorem max_unique_distances_in_21_gon : 
  ∃ (n : ℕ), n ≤ 5 ∧ (∀ (A B : ℕ), A ≠ B → A ∈ (set.range (fin 21)) → B ∈ (set.range (fin 21))
    → distance (A_i A_j) ≠ distance (A_k A_l)) :=
by
  sorry

end max_unique_distances_in_21_gon_l23_23006


namespace find_intersection_l23_23678

def setA : Set ℝ := { x | abs (x - 1) < 2 }
def setB : Set ℝ := { x | 2^x ≥ 1 }
def intersection (A B : Set ℝ) : Set ℝ := A ∩ B

theorem find_intersection :
  intersection setA setB = { x | 0 ≤ x ∧ x < 3 } := 
by 
  sorry

end find_intersection_l23_23678


namespace simplify_expression_l23_23545

theorem simplify_expression :
    (-2) - (-10) + (-6) - (+5) = -2 + 10 - 6 - 5 :=
    by 
        sorry -- Proof steps aren't required

end simplify_expression_l23_23545


namespace triangles_sharing_edges_l23_23290

-- Define the problem with the following conditions:
def six_points : Type := fin 6
def is_coplanar (a b c d : six_points) : Prop := sorry
def points_are_not_coplanar (P : six_points → Prop) : Prop := 
  ∀ a b c d, P a → P b → P c → P d → ¬is_coplanar a b c d

def connects (a b : six_points) : Prop := sorry
def distinct_pairs_of_triangles_share_edge (segments : set (six_points × six_points)) : Prop := 
  ∃ t1 t2 t3 t4 t5 t6 : six_points,
    ((connects t1 t2) ∧ (connects t2 t3) ∧ (connects t3 t1)) ∧
    ((connects t4 t5) ∧ (connects t5 t6) ∧ (connects t6 t4)) ∧
    (t1 ≠ t4 ∨ t2 ≠ t5 ∨ t3 ≠ t6)

theorem triangles_sharing_edges (P : six_points → Prop) (H : points_are_not_coplanar P) 
  (S : set (six_points × six_points)) (Hs : S.card = 10) : 
  ∃ (n ≥ 2), ∀ segments ∈ S, distinct_pairs_of_triangles_share_edge S :=
sorry

end triangles_sharing_edges_l23_23290


namespace cube_root_simplify_l23_23462

theorem cube_root_simplify :
  (∛(8 + 27) * ∛(8 + ∛27)) = ∛385 :=
by
  sorry

end cube_root_simplify_l23_23462


namespace january_salary_l23_23558

variable (J F M A My : ℕ)

axiom average_salary_1 : (J + F + M + A) / 4 = 8000
axiom average_salary_2 : (F + M + A + My) / 4 = 8400
axiom may_salary : My = 6500

theorem january_salary : J = 4900 :=
by
  /- To be filled with the proof steps applying the given conditions -/
  sorry

end january_salary_l23_23558


namespace day_53_days_from_thursday_is_monday_l23_23523

def day_of_week : Type := {n : ℤ // n % 7 = n}

def Thursday : day_of_week := ⟨4, by norm_num⟩
def Monday : day_of_week := ⟨1, by norm_num⟩

theorem day_53_days_from_thursday_is_monday : 
  (⟨(4 + 53) % 7, by norm_num⟩ : day_of_week) = Monday := 
by 
  sorry

end day_53_days_from_thursday_is_monday_l23_23523


namespace area_rhombus_abs_eq_9_l23_23900

open Real

theorem area_rhombus_abs_eq_9 :
  (∃ d1 d2 : ℝ, (d1 = 18) ∧ (d2 = 6) ∧ (1/2 * d1 * d2 = 54)) :=
begin
  use 18,
  use 6,
  split,
  { refl },
  split,
  { refl },
  { norm_num }
end

end area_rhombus_abs_eq_9_l23_23900


namespace max_area_rectangle_l23_23045

theorem max_area_rectangle (p : ℝ) (a b : ℝ) (h : p = 2 * (a + b)) : 
  ∃ S : ℝ, S = a * b ∧ (∀ (a' b' : ℝ), p = 2 * (a' + b') → S ≥ a' * b') → a = b :=
by
  sorry

end max_area_rectangle_l23_23045


namespace line_through_points_l23_23854

theorem line_through_points (x1 y1 x2 y2 : ℝ) (m b : ℝ) 
  (h1 : x1 = -3) (h2 : y1 = 1) (h3 : x2 = 1) (h4 : y2 = 3)
  (h5 : y1 = m * x1 + b) (h6 : y2 = m * x2 + b) :
  m + b = 3 := 
sorry

end line_through_points_l23_23854


namespace problem1_problem2_problem3_l23_23221

-- Problem 1
theorem problem1 :
  1 - 1^2022 + ((-1/2)^2) * (-2)^3 * (-2)^2 - |Real.pi - 3.14|^0 = -10 :=
by sorry

-- Problem 2
variables (a b : ℝ)

theorem problem2 :
  a^3 * (-b^3)^2 + (-2 * a * b)^3 = a^3 * b^6 - 8 * a^3 * b^3 :=
by sorry

-- Problem 3
theorem problem3 (a b : ℝ) :
  (2 * a^3 * b^2 - 3 * a^2 * b - 4 * a) * 2 * b = 4 * a^3 * b^3 - 6 * a^2 * b^2 - 8 * a * b :=
by sorry

end problem1_problem2_problem3_l23_23221


namespace bikes_in_garage_l23_23124

theorem bikes_in_garage : ∀ (B C : ℕ), C = 16 ∧ 2 * B + 4 * C = 82 → B = 9 :=
by
  intros B C
  intro h
  cases h with hC hWheels
  sorry

end bikes_in_garage_l23_23124


namespace principal_amount_is_600_l23_23138

-- Define the conditions
def SI : ℝ := 160
def R : ℝ := 0.0666666666666667
def T : ℕ := 4

-- Define the principal computation
def principal (SI : ℝ) (R : ℝ) (T : ℕ) : ℝ := SI / (R * T)

-- Prove that the computed principal is 600
theorem principal_amount_is_600 : principal SI R T = 600 := 
by sorry

end principal_amount_is_600_l23_23138


namespace problem1_problem2_l23_23329

-- The first problem
theorem problem1 (x : ℝ) (h : Real.tan x = 3) :
  (2 * Real.sin (Real.pi - x) + 3 * Real.cos (-x)) /
  (Real.sin (x + Real.pi / 2) - Real.sin (x + Real.pi)) = 9 / 4 :=
by
  sorry

-- The second problem
theorem problem2 (x : ℝ) (h : Real.tan x = 3) :
  2 * Real.sin x ^ 2 - Real.sin (2 * x) + Real.cos x ^ 2 = 13 / 10 :=
by
  sorry

end problem1_problem2_l23_23329


namespace hoseok_position_backwards_l23_23822

-- Define the number of people and the positioning condition
def num_people := 9

-- Define that Hoseok is the tallest
def hoseok_is_tallest (pos : ℕ) : Prop := pos = num_people

-- Theorem stating that Hoseok will be the 9th person from the back when lined up from tallest to shortest
theorem hoseok_position_backwards (pos : ℕ) : hoseok_is_tallest pos → pos = 1 → (num_people - pos + 1) = 9 := 
by simp [hoseok_is_tallest]; sorry

-- Example instantiation
example : hoseok_position_backwards 9 := by 
  unfold hoseok_position_backwards hoseok_is_tallest;
  simp; sorry

end hoseok_position_backwards_l23_23822


namespace find_sixth_term_l23_23305

open Nat

-- Given conditions
def arithmetic_progression (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ (n : ℕ), a (n + 1) = a n + d

def sum_of_first_three_terms (a : ℕ → ℤ) : Prop :=
  a 1 + a 2 + a 3 = 168

def second_minus_fifth (a : ℕ → ℤ) : Prop :=
  a 2 - a 5 = 42

-- Prove question == answer given conditions
theorem find_sixth_term :
  ∀ (a : ℕ → ℤ), arithmetic_progression a → sum_of_first_three_terms a → second_minus_fifth a → a 6 = 0 :=
by
  sorry

end find_sixth_term_l23_23305


namespace consecutive_integers_sum_l23_23502

theorem consecutive_integers_sum (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end consecutive_integers_sum_l23_23502


namespace triangle_construction_l23_23234

-- Definitions of elements
variable (k : Circle) (O : Point) (r : ℝ) (M : Point) (A B : Point) (d : ℝ)

-- Conditions and statement
theorem triangle_construction
  (hO : k.center = O)
  (hr : k.radius = r)
  (hM_in : M ∈ k)
  (hA_in : A ∈ k)
  (hB_in : B ∈ k)
  (hAB : (distance A B) = d)
  (0 < d) (d ≤ 2 * r)
  (hM_O_dist : |r - sqrt (4 * r^2 - d^2)| < distance O M ∧ distance O M < r + sqrt (4 * r^2 - d^2)) :
  ∃ (C: Point), Triangle M A B ∧ C ∈ k ∧ Orthocenter M A B = C ∧ side_length A B = d := 
sorry

end triangle_construction_l23_23234


namespace bus_driver_regular_rate_l23_23179

theorem bus_driver_regular_rate (hours := 60) (total_pay := 1200) (regular_hours := 40) (overtime_rate_factor := 1.75) :
  ∃ R : ℝ, 40 * R + 20 * (1.75 * R) = 1200 ∧ R = 16 := 
by
  sorry

end bus_driver_regular_rate_l23_23179


namespace find_value_l23_23370

variables {p q s u : ℚ}

theorem find_value
  (h1 : p / q = 5 / 6)
  (h2 : s / u = 7 / 15) :
  (5 * p * s - 3 * q * u) / (6 * q * u - 5 * p * s) = -19 / 73 :=
sorry

end find_value_l23_23370


namespace three_digit_numbers_with_sum_27_l23_23636

theorem three_digit_numbers_with_sum_27 :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 27}.card = 1 :=
by
  sorry

end three_digit_numbers_with_sum_27_l23_23636


namespace days_y_needs_l23_23163

theorem days_y_needs
  (d : ℝ)
  (h1 : (1:ℝ) / 21 * 14 = 1 - 5 * (1 / d)) :
  d = 10 :=
sorry

end days_y_needs_l23_23163


namespace AlisonMoneyGBP_l23_23966

noncomputable def AlisonMoneyInGBP : ℝ := 2370.37

theorem AlisonMoneyGBP (Alison Brittany Brooke Kent Charlie Daniella Elle : ℝ) 
  (h1 : Alison = Brittany / 2)
  (h2 : Brittany = 4 * (Brooke / 1.15))
  (h3 : Brooke = 2 * 800)
  (h4 : Kent = 1000)
  (h5 : Charlie = Kent * 2 - (Kent * 2 * 0.20))
  (ConvertedCharlieJPYtoGBP : Charlie / 300 * 150)
  (CharlieLendDaniella : Daniella = 0.5 * (Charlie / 300))
  (DaniellaAfterConversion : ConvertedCharlieJPYtoGBP = 3 * Elle)
  : Alison = AlisonMoneyInGBP :=
sorry

end AlisonMoneyGBP_l23_23966


namespace ratio_of_areas_l23_23668

noncomputable def area_ratio (a : ℝ) : ℝ :=
  let side_triangle : ℝ := a
  let area_triangle : ℝ := (1 / 2) * side_triangle * side_triangle
  let height_rhombus : ℝ := side_triangle * Real.sin (Real.pi / 3)
  let area_rhombus : ℝ := height_rhombus * side_triangle
  area_rhombus / area_triangle

theorem ratio_of_areas (a : ℝ) (h : a > 0) : area_ratio a = 3 := by
  -- The proof would be here
  sorry

end ratio_of_areas_l23_23668


namespace no_such_x_exists_l23_23692

-- Define the primary condition of the problem
def condition_1 (x : ℝ) : Prop :=
  arccos (4 / 5) - arccos (-4 / 5) = arcsin x

-- Define the goal based on the condition provided
theorem no_such_x_exists : ¬∃ x : ℝ, condition_1 x :=
by
  sorry

end no_such_x_exists_l23_23692


namespace susie_earnings_l23_23472

-- Define the constants and conditions
def price_per_slice : ℕ := 3
def price_per_whole_pizza : ℕ := 15
def slices_sold : ℕ := 24
def whole_pizzas_sold : ℕ := 3

-- Calculate earnings from slices and whole pizzas
def earnings_from_slices : ℕ := slices_sold * price_per_slice
def earnings_from_whole_pizzas : ℕ := whole_pizzas_sold * price_per_whole_pizza
def total_earnings : ℕ := earnings_from_slices + earnings_from_whole_pizzas

-- Prove that the total earnings are $117
theorem susie_earnings : total_earnings = 117 := by
  sorry

end susie_earnings_l23_23472


namespace A_false_B_true_C_true_D_true_l23_23153

theorem A_false :
  ¬ ∃ x, ∀ y = (x^2 + 1) / x, y = 2 :=
by
  sorry

theorem B_true (x : ℝ) (h : x > 1) :
  (∀ y, y = 2*x + 4 / (x - 1) - 1 → y ≥ 4 * real.sqrt 2 + 1) :=
by
  sorry

theorem C_true (x y : ℝ) (h : x + 2 * y = 3 * x * y) (hx : 0 < x) (hy : 0 < y) :
  (2 * x + y ≥ 3) :=
by
  sorry

theorem D_true (x y : ℝ) (h : 9 * x^2 + y^2 + x * y = 1) :
  ∃ c, c = (3 * x + y) ∧ c ≤ (2 * real.sqrt 21 / 7) :=
by
  sorry

end A_false_B_true_C_true_D_true_l23_23153


namespace max_intersections_three_circles_one_line_l23_23529

theorem max_intersections_three_circles_one_line (c1 c2 c3 : Circle) (L : Line) :
  greatest_number_points_of_intersection c1 c2 c3 L = 12 :=
sorry

end max_intersections_three_circles_one_line_l23_23529


namespace find_circle_and_line_l23_23667

-- Definitions based on given problem conditions
def A : (ℝ × ℝ) := (1, 1)
def B : (ℝ × ℝ) := (2, -2)
def l (x y : ℝ) := x + y + 5 = 0
def D : (ℝ × ℝ) := (-1, -1)
def chord_length : ℝ := 2 * Real.sqrt 21

-- Lean statement to prove the equivalent proof problem
theorem find_circle_and_line 
  (C : ℝ × ℝ) 
  (hC_passes_through_A : (C.1 + 3)^2 + (C.2 + 2)^2 = 25)
  (hC_on_line_l : l C.1 C.2)
  (m_eqn : ℝ → ℝ → Prop)
  (hD_on_m : m_eqn (-1) (-1)) 
  (hChord_length : ∀ P Q, P ≠ Q → ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = chord_length^2 → m_eqn P.1 P.2 ∧ m_eqn Q.1 Q.2 → False) :
  (∀ x y, (x + 3)^2 + (y + 2)^2 = 25) → (∀ x y, (x = -1 ∨ 3 * x + 4 * y + 7 = 0)) :=
by 
  sorry

end find_circle_and_line_l23_23667


namespace prove_problem_l23_23475

def problem (a b c d : ℝ) : Prop :=
  (a + b + c + d + 105) / 5 = 90 → (a + b + c + d) / 4 = 86.25

theorem prove_problem (a b c d : ℝ) : problem a b c d :=
by
  intro h
  have sum_eq_450 : a + b + c + d + 105 = 450 := sorry
  have sum_minus_105 : a + b + c + d = 345 := sorry
  show (a + b + c + d) / 4 = 86.25, from sorry

end prove_problem_l23_23475


namespace height_flagstaff_l23_23188

variables (s_1 s_2 h_2 : ℝ)
variable (h : ℝ)

-- Define the conditions as given
def shadow_flagstaff := s_1 = 40.25
def shadow_building := s_2 = 28.75
def height_building := h_2 = 12.5
def similar_triangles := (h / s_1) = (h_2 / s_2)

-- Prove the height of the flagstaff
theorem height_flagstaff : shadow_flagstaff s_1 ∧ shadow_building s_2 ∧ height_building h_2 ∧ similar_triangles h s_1 h_2 s_2 → h = 17.5 :=
by sorry

end height_flagstaff_l23_23188


namespace frank_initial_candy_l23_23279

theorem frank_initial_candy (n : ℕ) (h1 : n = 21) (h2 : 2 > 0) :
  2 * n = 42 :=
by
  --* Use the hypotheses to establish the required proof
  sorry

end frank_initial_candy_l23_23279


namespace tangent_line_parameter_l23_23564

theorem tangent_line_parameter (t : ℝ) (h : t > 0) : 
  (∃ (ρ θ : ℝ), ρ * cos θ = t ∧ ρ = 2 * sin θ ∧ ∀ (x y : ℝ), x^2 + (y - 1)^2 = 1 → x = t) 
  → t = 1 := 
by sorry

end tangent_line_parameter_l23_23564


namespace birds_find_more_than_60_percent_millet_on_wednesday_l23_23436

theorem birds_find_more_than_60_percent_millet_on_wednesday :
  (∀ n : ℕ, n ≥ 0 → (millet_amount n) = 0.4 * ((1 - 0.7^n) / 0.3))
  → (∀ n : ℕ, n ≥ 0 → (total_seeds n) = 1)
  → (millet_ratio 3 > 0.6) :=
sorry

def millet_amount (n : ℕ) : ℝ :=
0.4 * ((1 - 0.7^n) / 0.3)

def total_seeds (n : ℕ) : ℝ :=
1

def millet_ratio (n : ℕ) : ℝ :=
millet_amount n / total_seeds n

end birds_find_more_than_60_percent_millet_on_wednesday_l23_23436


namespace first_cat_blue_eyed_kittens_l23_23024

variable (B : ℕ)
variable (C1 : 35 * (B + 17) = 100 * (B + 4))

theorem first_cat_blue_eyed_kittens : B = 3 :=
by
  -- proof
  sorry

end first_cat_blue_eyed_kittens_l23_23024


namespace hyperbola_asymptotes_eqn_l23_23354

noncomputable def asymptotes_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a/b) = 1/(2*sqrt 2)) 
    (c : ℝ) (ec : c = 3 * a) : Prop := 
  ∀ x y : ℝ, (x = 2 * sqrt 2 * y) ∨ (x = -2 * sqrt 2 * y)

theorem hyperbola_asymptotes_eqn (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (e : ℝ) (h_ecc : e = 3) :
  (asymptotes_of_hyperbola a b h1 h2 ((a/b) = 1/(2*sqrt 2)) (e : ℝ) (ec : b = a * sqrt (e^2 - 1))) :=
by sorry

end hyperbola_asymptotes_eqn_l23_23354


namespace cost_price_A_min_cost_bshelves_l23_23597

-- Define the cost price of type B bookshelf
def costB_bshelf : ℝ := 300

-- Define the cost price of type A bookshelf
def costA_bshelf : ℝ := 1.2 * costB_bshelf

-- Define the total number of bookshelves
def total_bshelves : ℕ := 60

-- Define the condition for type A and type B bookshelves count
def typeBshelves := λ (typeAshelves : ℕ) => total_bshelves - typeAshelves
def typeBshelves_constraints := λ (typeAshelves : ℕ) => total_bshelves - typeAshelves ≤ 2 * typeAshelves

-- Define the equation for the costs
noncomputable def total_cost (typeAshelves : ℕ) : ℝ :=
  360 * typeAshelves + 300 * (total_bshelves - typeAshelves)

-- Define the goal: cost price of type A bookshelf is 360 yuan
theorem cost_price_A : costA_bshelf = 360 :=
by 
  sorry

-- Define the goal: the school should buy 20 type A bookshelves and 40 type B bookshelves to minimize cost
theorem min_cost_bshelves : ∃ typeAshelves : ℕ, typeAshelves = 20 ∧ typeBshelves typeAshelves = 40 :=
by
  sorry

end cost_price_A_min_cost_bshelves_l23_23597


namespace quadratic_roots_transformation_l23_23815

noncomputable def transformed_polynomial (p q r : ℝ) : Polynomial ℝ :=
  Polynomial.X^2 + (p*q + 2*q)*Polynomial.X + (p^3*r + p*q^2 + q^2)

noncomputable def original_polynomial (p q r : ℝ) : Polynomial ℝ :=
  p * Polynomial.X^2 + q * Polynomial.X + r

theorem quadratic_roots_transformation (p q r : ℝ) (u v : ℝ)
  (huv1 : u + v = -q / p)
  (huv2 : u * v = r / p) :
  transformed_polynomial p q r = Polynomial.monomial 2 1 +
    Polynomial.monomial 1 (p*q + 2*q) +
    Polynomial.monomial 0 (p^3*r + p*q^2 + q^2) :=
by {
  sorry
}

end quadratic_roots_transformation_l23_23815


namespace absolute_value_of_x_l23_23437

variable (x : ℝ)

theorem absolute_value_of_x (h: (| (3 + x) - (3 - x) |) = 8) : |x| = 4 :=
by sorry

end absolute_value_of_x_l23_23437


namespace average_age_of_9_students_l23_23848

theorem average_age_of_9_students (avg_age_17_students : ℕ)
                                   (num_students : ℕ)
                                   (avg_age_5_students : ℕ)
                                   (num_5_students : ℕ)
                                   (age_17th_student : ℕ) :
    avg_age_17_students = 17 →
    num_students = 17 →
    avg_age_5_students = 14 →
    num_5_students = 5 →
    age_17th_student = 75 →
    (144 / 9) = 16 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_age_of_9_students_l23_23848


namespace mat_inverse_sum_l23_23863

theorem mat_inverse_sum (a b c d : ℝ)
  (h1 : -2 * a + 3 * d = 1)
  (h2 : a * c - 12 = 0)
  (h3 : -8 + b * d = 0)
  (h4 : 4 * c - 4 * b = 0)
  (abc : a = 3 * Real.sqrt 2)
  (bb : b = 2 * Real.sqrt 2)
  (cc : c = 2 * Real.sqrt 2)
  (dd : d = (1 + 6 * Real.sqrt 2) / 3) :
  a + b + c + d = 9 * Real.sqrt 2 + 1 / 3 := by
  sorry

end mat_inverse_sum_l23_23863


namespace fewer_cans_today_l23_23922

variable (nc_sarah_yesterday : ℕ)
variable (nc_lara_yesterday : ℕ)
variable (nc_alex_yesterday : ℕ)
variable (nc_sarah_today : ℕ)
variable (nc_lara_today : ℕ)
variable (nc_alex_today : ℕ)

-- Given conditions
def yesterday_collected_cans : Prop :=
  nc_sarah_yesterday = 50 ∧
  nc_lara_yesterday = nc_sarah_yesterday + 30 ∧
  nc_alex_yesterday = 90

def today_collected_cans : Prop :=
  nc_sarah_today = 40 ∧
  nc_lara_today = 70 ∧
  nc_alex_today = 55

theorem fewer_cans_today :
  yesterday_collected_cans nc_sarah_yesterday nc_lara_yesterday nc_alex_yesterday →
  today_collected_cans nc_sarah_today nc_lara_today nc_alex_today →
  (nc_sarah_yesterday + nc_lara_yesterday + nc_alex_yesterday) -
  (nc_sarah_today + nc_lara_today + nc_alex_today) = 55 :=
by
  intros h1 h2
  sorry

end fewer_cans_today_l23_23922


namespace absolute_value_of_x_l23_23439

variable (x : ℝ)

theorem absolute_value_of_x (h: (| (3 + x) - (3 - x) |) = 8) : |x| = 4 :=
by sorry

end absolute_value_of_x_l23_23439


namespace impossible_to_identify_compound_l23_23638

theorem impossible_to_identify_compound (mass_percentage_O : ℝ) (h : mass_percentage_O = 28.57) :
  ∀ compound : Type, ¬ (compound = unique_by_mass_percentage_O := 28.57) :=
by
  sorry

end impossible_to_identify_compound_l23_23638


namespace value_of_a_l23_23144

theorem value_of_a (a : ℝ) : (1 / (Real.log 3 / Real.log a) + 1 / (Real.log 4 / Real.log a) + 1 / (Real.log 5 / Real.log a) = 1) → a = 60 :=
by
  sorry

end value_of_a_l23_23144


namespace quadratic_inequality_solution_set_l23_23882

theorem quadratic_inequality_solution_set (x : ℝ) :
  (x^2 - 3 * x - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 4) :=
sorry

end quadratic_inequality_solution_set_l23_23882


namespace nadia_wins_strategy_l23_23433

theorem nadia_wins_strategy :
  ∀ thousands_digit hundreds_digit tens_digit units_digit : Fin 8,
  ∃ (Martha_digits Nadia_digits : List (Fin 8))
    (hm : Martha_digits.length = 4) (hn : Nadia_digits.length = 4),
  let sum := (Martha_digits ++ Nadia_digits).sum in
  (sum % 6 = 0) :=
by
  -- Proof not required, so we use sorry
  sorry

end nadia_wins_strategy_l23_23433


namespace find_closest_point_on_line_l23_23262

def closest_point_on_line (x1 y1 : ℝ) (x2 y2 : ℝ) : Prop :=
  ∃ (px py : ℝ), 
    (py = (px - 3) / 3) ∧
    (px, py) = (33 / 10, 1 / 10) ∧
    ∀ (x y : ℝ), (y = (x - 3) / 3) → (Real.dist (px, py) (x1, y1) ≤ Real.dist (x, y) (x1, y1))

theorem find_closest_point_on_line : closest_point_on_line 4 (-2) 

end find_closest_point_on_line_l23_23262


namespace solve_functional_equation_l23_23013

noncomputable def functional_equation (α β : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → f(x) * f(y) = y^α * f(x / 2) + x^β * f(y / 2)

theorem solve_functional_equation (α β : ℝ) (f : ℝ → ℝ) :
  (∀ x y : ℝ, 0 < x → 0 < y → f(x) * f(y) = y^α * f(x / 2) + x^β * f(y / 2)) →
  (∀ x : ℝ, 0 < x → f(x) = 2^(1-α) * x^α ∨ f(x) = 0) :=
  sorry

end solve_functional_equation_l23_23013


namespace prob_k_gnomes_fall_exp_gnomes_falling_l23_23767

variables (n k : ℕ) (p : ℝ)
hypotheses 
  (hn : 0 < n)
  (hp : 0 < p) (hp1 : p < 1)
  (hk : 0 ≤ k) (hk1 : k ≤ n)

open ProbabilityTheory
  
def probability_k_gnomes_fall := 
  p * (1 - p) ^ (n - k)

def expected_gnomes_fall :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p

theorem prob_k_gnomes_fall (hprob : 0 < p ∧ p < 1) : 
  ∀ n k : ℕ, 0 ≤ k ∧ k ≤ n → probability_k_gnomes_fall n k p = p * (1 - p) ^ (n - k) :=
by sorry

theorem exp_gnomes_falling (hprob : 0 < p ∧ p < 1) : 
  ∀ n : ℕ, 0 < n → expected_gnomes_fall n p = n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p :=
by sorry

end prob_k_gnomes_fall_exp_gnomes_falling_l23_23767


namespace exists_triangle_divisible_into_101_congruent_triangles_l23_23241

theorem exists_triangle_divisible_into_101_congruent_triangles : 
  ∃ T : Triangle, (∃ n : ℕ, n = 101 ∧ T.can_be_divided_into_n_congruent_triangles n) :=
by
  sorry

end exists_triangle_divisible_into_101_congruent_triangles_l23_23241


namespace a_is_perfect_square_l23_23035

theorem a_is_perfect_square (a b : ℕ) (h : ab ∣ (a^2 + b^2 + a)) : (∃ k : ℕ, a = k^2) :=
sorry

end a_is_perfect_square_l23_23035


namespace part_I_solution_part_II_solution_l23_23705

-- Definitions for the problem
def f (x a : ℝ) : ℝ := |x - a| + |x - 1|

-- Part I: When a = 2, solve the inequality f(x) < 4
theorem part_I_solution (x : ℝ) : f x 2 < 4 ↔ x > -1/2 ∧ x < 7/2 :=
by sorry

-- Part II: Range of values for a such that f(x) ≥ 2 for all x
theorem part_II_solution (a : ℝ) : (∀ x, f x a ≥ 2) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by sorry

end part_I_solution_part_II_solution_l23_23705


namespace solve_for_x_l23_23468

theorem solve_for_x : (3 / 4 - 2 / 5 = 1 / x) → x = 20 / 7 := 
by
  intro h
  have h_eq : 3 / 4 - 2 / 5 = 7 / 20 := sorry
  rw [h_eq] at h
  exact (eq_div_iff (by norm_num)).mp h

end solve_for_x_l23_23468


namespace ac_value_l23_23095

-- Define the quadratic function that intersects axes at the vertices of an equilateral triangle
def quadratic_function (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + c

-- Define the condition that the graph intersects the axes forming an equilateral triangle
def intersects_equilateral_triangle (a c : ℝ) : Prop :=
  let delta := (a * 0^2 + c)
  let x_intersect := real.sqrt (-c / a)
  let h := (2 * x_intersect) * real.sqrt(3) / 2
  c = h

-- The main theorem we want to prove
theorem ac_value (a c : ℝ) (hac : intersects_equilateral_triangle a c) : a * c = -3 := sorry

end ac_value_l23_23095


namespace smallest_angle_solution_l23_23641

noncomputable def find_smallest_angle : ℝ :=
  classical.some (Exists.some (Icc 0 360) (λ x, sin (3 * x) * sin (4 * x) = cos (3 * x) * cos (4 * x)))

theorem smallest_angle_solution : find_smallest_angle = 90 / 7 := sorry

end smallest_angle_solution_l23_23641


namespace FM_eq_FN_l23_23395

-- Definitions related to the problem
variable (A B C D E F M N O : Point)
variable (circumCircle : Circle)
variable (triangle_ABC : Triangle A B C)

-- Given conditions
axiom angle_bisector_AD : angleBisector A B C A D
axiom intersects_at_D : intersects (lineProj AD circumCircle) D
axiom midpoint_E : isMidpoint E B C
axiom perpendicular_EF_AD : perp EF AD
axiom perp_through_F : forall P Q : Point, P ∈ lineProj EF -> perp P F Q -> Q ∈ {M, N} -> intersects (lineProj DF) Q

-- To Prove
theorem FM_eq_FN : distance F M = distance F N :=
sorry

end FM_eq_FN_l23_23395


namespace sequence_limit_l23_23933

open Real

theorem sequence_limit :
  (∀ n : ℕ, 1 + 2 + 3 + ... + n = n * (n + 1) / 2) →
  filter.tendsto (λ n : ℕ, ((n * (n + 1) / 2) / (sqrt (9 * n^4 + 1)))) filter.at_top (𝓝 (1 / 6)) :=
by
  intros h
  -- rest of the proof
  sorry

end sequence_limit_l23_23933


namespace first_group_average_score_l23_23751

noncomputable def average_score (scores : List ℤ) : ℚ :=
  scores.sum / scores.length

theorem first_group_average_score :
  let class_average := 80
  let score_diffs := [2, 3, -3, -5, 12, 14, 10, 4, -6, 4, -11, -7, 8, -2]
  let average_of_diffs := average_score score_diffs
  (class_average + average_of_diffs = 81.64) :=
by
  let class_average := 80
  let score_diffs := [2, 3, -3, -5, 12, 14, 10, 4, -6, 4, -11, -7, 8, -2]
  have avg_diffs := average_score score_diffs
  have avg_diffs_val : avg_diffs = (1.64 : ℚ) :=
    sorry -- Proof of the calculation of average score differences
  show (class_average : ℚ) + avg_diffs = 81.64 /-
  (class_average + 1.64 = 81.64) using avg_diffs_val -/
    sorry -- Conclude the proof that 80 + 1.64 = 81.64

end first_group_average_score_l23_23751


namespace q_minus_p_l23_23800

theorem q_minus_p : 
  ∃ (p q : ℕ), 
    p % 13 = 7 ∧ 
    p ≥ 100 ∧ 
    q % 13 = 7 ∧ 
    q ≥ 1000 ∧ 
    q - p = 897 :=
begin
  sorry
end

end q_minus_p_l23_23800


namespace probability_of_drawing_white_ball_l23_23772

def balls : List String := ["white", "black", "black"]
def total_balls : Nat := balls.length
def white_balls : Nat := balls.count (· == "white")
def probability_white : ℚ := white_balls / total_balls

theorem probability_of_drawing_white_ball :
  probability_white = 1 / 3 :=
  sorry

end probability_of_drawing_white_ball_l23_23772


namespace angle_QPS_l23_23516

noncomputable def PQ : ℝ := 1 -- Normalizing to 1 for simplicity
noncomputable def QR : ℝ := PQ -- Since PQ = QR
noncomputable def PR : ℝ := 1 -- Normalizing to 1 for simplicity
noncomputable def RS : ℝ := PR -- Since PR = RS

def angle_PQR : ℝ := 50
def angle_PRS : ℝ := 160

def angle_PQS : ℝ := (180 - angle_PQR) / 2
def angle_PRQ : ℝ := (180 - angle_PRS) / 2

theorem angle_QPS :
  angle_PQS - angle_PRQ = 55 := by
  sorry

end angle_QPS_l23_23516


namespace steve_speed_back_correct_l23_23928

-- Define the conditions
def distance_to_work := 10 -- The distance from Steve's house to work in km
def time_on_road := 6 -- Total time spent on the road in hours

-- Variables representing Steve's speeds
variables (v : ℕ) (speed_to_work : ℕ) (speed_back : ℕ)

-- Define speed on the way to and back from work
def speed_to_work := v
def speed_back := 2 * v

-- Define the time spent on road in each direction
def time_to_work := distance_to_work / speed_to_work
def time_back := distance_to_work / speed_back

-- Prove that the total speed back is correct
theorem steve_speed_back_correct : 
  (time_to_work + time_back = time_on_road) → speed_back = 5 := 
by
  sorry

end steve_speed_back_correct_l23_23928


namespace sufficient_not_necessary_condition_l23_23661

theorem sufficient_not_necessary_condition (x y : ℝ) (h1 : x ≥ 1) (h2 : y ≥ 2) : 
  x + y ≥ 3 ∧ (¬ (∀ x y : ℝ, x + y ≥ 3 → x ≥ 1 ∧ y ≥ 2)) := 
by {
  sorry -- The actual proof goes here.
}

end sufficient_not_necessary_condition_l23_23661


namespace ellipse_x_intercept_l23_23604

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2))

theorem ellipse_x_intercept (x : ℝ) :
  let f1 := (-2, 1)
  let f2 := (4, 1)
  let p1 := (0, 0)

  distance f1 p1 + distance f2 p1 = distance f1 (x, 0) + distance f2 (x, 0) →
  x = 6.27 :=
begin
  sorry
end

end ellipse_x_intercept_l23_23604


namespace problem_7_38_solution_l23_23158

theorem problem_7_38_solution (x : ℝ) (h : x > 1) : x^(log (x^2 - 1) / log (x^2)) = 5 ↔ x = sqrt 26 := 
sorry

end problem_7_38_solution_l23_23158


namespace intersection_of_A_and_B_l23_23326

-- Define sets A and B
def setA : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}
def setB : Set ℝ := {x : ℝ | x < 2}

-- Prove that A ∩ B = (-3, 2)
theorem intersection_of_A_and_B : {x : ℝ | x ∈ setA ∧ x ∈ setB} = {x : ℝ | -3 < x ∧ x < 2} := 
by 
  sorry

end intersection_of_A_and_B_l23_23326


namespace part_a_part_b_l23_23793

theorem part_a (n : ℕ) (hn : n ≥ 2) :
  ∃ pairs : list (ℕ × ℕ), (∀ (x, y) ∈ pairs, 1 ≤ x ∧ x ≤ n ∧ 1 ≤ y ∧ y ≤ n) ∧
  no_intersecting_pairs pairs ∧
  greatest_integers_use pairs (ceil (n / 2)) :=
sorry

theorem part_b (n : ℕ) (hn : n ≥ 2) :
  ∃ tagging : ℕ → ℕ, (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 2n → 1 ≤ tagging i ∧ tagging i ≤ n ∧ 
    (∃! j, tagging i = tagging j) ) ∧ 
  (∀ pairs : list (ℕ × ℕ), (∀ (x, y) ∈ pairs, 1 ≤ x ∧ x ≤ 2n ∧ 1 ≤ y ∧ y ≤ 2n) ∧ 
    no_intersecting_pairs pairs → 
    greatest_integers_use pairs (ceil (n / 2))) :=
sorry

def no_intersecting_pairs : list (ℕ × ℕ) → Prop := sorry
def greatest_integers_use : list (ℕ × ℕ) → ℕ → Prop := sorry

end part_a_part_b_l23_23793


namespace count_triangles_non_collinear_l23_23099

theorem count_triangles_non_collinear :
  let points := { p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4 }
  let is_collinear : Π (p1 p2 p3 : ℤ × ℤ), Prop :=
    λ ⟨x1, y1⟩ ⟨x2, y2⟩ ⟨x3, y3⟩, (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)
  let triangles := { (p1, p2, p3) ∈ points × points × points // ¬is_collinear p1 p2 p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 }
  triangles.cards = 516 :=
by
  sorry

end count_triangles_non_collinear_l23_23099


namespace solve_m_l23_23735

def is_homogeneous_polynomial (p : List (List (ℕ × ℕ))) : Prop :=
  ∃ d, ∀ term in p, term.foldl (λ acc (var_exp : ℕ × ℕ), acc + var_exp.snd) 0 = d

theorem solve_m (m : ℕ) : 
  is_homogeneous_polynomial [[(m+2, 1), (2, 2)], [(1, 1), (3, 2), (2, 3)]] → m = 2 :=
by
  sorry

end solve_m_l23_23735


namespace village_population_equal_in_7_years_l23_23132

theorem village_population_equal_in_7_years :
  let population_x (t : ℕ) := 68000 - 2676 * t
  let population_y (t : ℕ) := 42000 + 1178 * t
  ∃ t : ℕ, population_x t = population_y t ∧ t = 7 :=
begin
  let population_x := λ t : ℕ, 68000 - 2676 * t,
  let population_y := λ t : ℕ, 42000 + 1178 * t,
  use 7,
  simp [population_x, population_y],
  norm_num,
end

end village_population_equal_in_7_years_l23_23132


namespace correct_statements_l23_23885

-- Definitions based on the conditions and question
def S (n : ℕ) : ℤ := -n^2 + 7 * n + 1

-- Definition of the sequence an
def a (n : ℕ) : ℤ := 
  if n = 1 then 7 
  else S n - S (n - 1)

-- Theorem statements based on the correct answers derived from solution
theorem correct_statements :
  (∀ n : ℕ, n > 4 → a n < 0) ∧ (S 3 = S 4 ∧ (∀ m : ℕ, S m ≤ S 3)) :=
by {
  sorry
}

end correct_statements_l23_23885


namespace cameron_list_count_l23_23986

theorem cameron_list_count : 
  (∃ (n m : ℕ), n = 900 ∧ m = 27000 ∧ (∀ k : ℕ, (30 * k) ≥ n ∧ (30 * k) ≤ m → ∃ count : ℕ, count = 871)) :=
by
  sorry

end cameron_list_count_l23_23986


namespace prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l23_23756

variables (n k : ℕ) (p : ℝ)
variables (h_pos : 0 < p) (h_lt_one : p < 1)

-- Probability that exactly k gnomes fall
theorem prob_exactly_k_gnomes_fall (h_k_le_n : k ≤ n) :
  prob_speed (exactly_k_gnomes_fall n k p) = p * (1 - p)^(n - k) := sorry

-- Expected number of fallen gnomes
theorem expected_fallen_gnomes : 
  expected_falls n p = n + 1 - 1/p + (1 - p)^(n + 1)/p := sorry

end prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l23_23756


namespace absolute_value_of_x_l23_23438

variable (x : ℝ)

theorem absolute_value_of_x (h: (| (3 + x) - (3 - x) |) = 8) : |x| = 4 :=
by sorry

end absolute_value_of_x_l23_23438


namespace probability_two_points_square_l23_23412

def gcd (a b c : Nat) : Nat := Nat.gcd (Nat.gcd a b) c  

theorem probability_two_points_square {a b c : ℕ} (hx : gcd a b c = 1)
  (h : (26 - Real.pi) / 32 = (a - b * Real.pi) / c) : a + b + c = 59 :=
by
  sorry

end probability_two_points_square_l23_23412


namespace negation_proposition_l23_23493

noncomputable theory

open Real

theorem negation_proposition :
  ¬ (∀ x : ℝ, exp x - x - 1 ≥ 0) ↔ ∃ x : ℝ, exp x - x - 1 < 0 :=
by sorry

end negation_proposition_l23_23493


namespace units_digit_sum_of_factorials_is_3_l23_23267

theorem units_digit_sum_of_factorials_is_3 :
  (∑ k in Finset.range 2024, Nat.factorial k) % 10 = 3 :=
by
  sorry

end units_digit_sum_of_factorials_is_3_l23_23267


namespace square_area_divided_diagonal_l23_23186

-- Define the problem statement
def square_diagonal_division_area (A B C D E F O : Point) : Prop :=
  let AC := dist A C
  let AE := dist A E
  let EF := dist E F
  let FC := dist F C in
  is_square A B C D ∧ 
  perpendicular B L A C ∧ 
  perpendicular D L' A C ∧ 
  AE = EF ∧ EF = FC ∧ 
  AE = 2 ∧
  area_square A B C D = 18

theorem square_area_divided_diagonal :
  ∀ (A B C D E F O : Point),
  square_diagonal_division_area A B C D E F O :=
sorry

end square_area_divided_diagonal_l23_23186


namespace find_g_inv_sum_l23_23405

def g (x : ℝ) : ℝ :=
if x < 10 then x + 2 else 3 * x + 1

def g_inv (y : ℝ) : ℝ :=
if y = 5 then 3 else if y = 28 then 9 else 0

theorem find_g_inv_sum : g_inv 5 + g_inv 28 = 12 :=
by
  sorry

end find_g_inv_sum_l23_23405


namespace DE_equals_BD_l23_23473

variables {A B C D E : Type*}
variables [linear_ordered_field A]

-- Define the isosceles triangle with angle B = 108 degrees
def isosceles_triangle_ABC (A B C : Type*) :=
∃ (AB BC : ℝ), AB = BC ∧ ∠B = 108

-- Define bisector AD
def bisector_AD (A B C D : Type*) :=
∃ (AD : ℝ), isosceles_triangle_ABC A B C ∧ mid D A C

-- Define perpendicular
def perpendicular_to_AD (D E : Type*) :=
∃ PE DE, perpendicular PE AD DE ∧ intersect PE E AC

-- Main theorem to prove DE = BD
theorem DE_equals_BD
  (isosceles_triangle_ABC A B C)
  (bisector_AD A B C D)
  (perpendicular_to_AD D E) :
  DE = BD :=
sorry

end DE_equals_BD_l23_23473


namespace bike_cost_l23_23514

theorem bike_cost (days_in_two_weeks : ℕ) 
  (bracelets_per_day : ℕ)
  (price_per_bracelet : ℕ)
  (total_bracelets : ℕ)
  (total_money : ℕ) 
  (h1 : days_in_two_weeks = 2 * 7)
  (h2 : bracelets_per_day = 8)
  (h3 : price_per_bracelet = 1)
  (h4 : total_bracelets = days_in_two_weeks * bracelets_per_day)
  (h5 : total_money = total_bracelets * price_per_bracelet) :
  total_money = 112 :=
sorry

end bike_cost_l23_23514


namespace eq_of_divisibility_l23_23172

theorem eq_of_divisibility (a b : ℕ) (h : (a^2 + b^2) ∣ (a * b)) : a = b :=
  sorry

end eq_of_divisibility_l23_23172


namespace rent_percentage_increase_l23_23193

theorem rent_percentage_increase
  (original_avg_rent : ℕ)
  (friends : ℕ)
  (new_avg_rent : ℕ)
  (original_rent : ℕ)
  (percentage_increase : ℕ) :
  original_avg_rent = 800 →
  friends = 4 →
  new_avg_rent = 880 →
  original_rent = 1600 →
  percentage_increase = 20 :=
by
  intros h1 h2 h3 h4,
  sorry

end rent_percentage_increase_l23_23193


namespace watched_commercials_eq_100_l23_23403

variable (x : ℕ) -- number of people who watched commercials
variable (s : ℕ := 27) -- number of subscribers
variable (rev_comm : ℝ := 0.50) -- revenue per commercial
variable (rev_sub : ℝ := 1.00) -- revenue per subscriber
variable (total_rev : ℝ := 77.00) -- total revenue

theorem watched_commercials_eq_100 (h : rev_comm * (x : ℝ) + rev_sub * (s : ℝ) = total_rev) : x = 100 := by
  sorry

end watched_commercials_eq_100_l23_23403


namespace max_points_of_intersection_l23_23534

theorem max_points_of_intersection (circles : ℕ) (line : ℕ) (h_circles : circles = 3) (h_line : line = 1) : 
  ∃ points_of_intersection, points_of_intersection = 12 :=
by
  -- Proof here (omitted)
  sorry

end max_points_of_intersection_l23_23534


namespace area_of_rectangular_toilet_l23_23136

def length : ℝ := 5
def width : ℝ := 17 / 20
def area := length * width

theorem area_of_rectangular_toilet : area = 4.25 := by
  -- Placeholder for actual proof
  sorry

end area_of_rectangular_toilet_l23_23136


namespace abs_x_equals_4_l23_23443

-- Define the points A and B as per the conditions
def point_A (x : ℝ) : ℝ := 3 + x
def point_B (x : ℝ) : ℝ := 3 - x

-- Define the distance between points A and B
def distance (x : ℝ) : ℝ := abs ((point_A x) - (point_B x))

theorem abs_x_equals_4 (x : ℝ) (h : distance x = 8) : abs x = 4 :=
by
  sorry

end abs_x_equals_4_l23_23443


namespace convert_rectangular_form_l23_23235

def r : ℝ := Real.sqrt 2
def θ : ℝ := 13 * Real.pi / 6

theorem convert_rectangular_form : r * (Complex.cos θ + Complex.sin θ * Complex.I) = 
  (Real.sqrt 6 / 2 + (Real.sqrt 2 / 2) * Complex.I) :=
by
  sorry

end convert_rectangular_form_l23_23235


namespace maximize_profit_under_constraints_l23_23951

noncomputable def max_profit : ℕ :=
  let profit (x : ℕ) : ℕ := (-100 : ℤ) * x + 10000
  nat.max (profit 12) (nat.max (profit 13) (nat.max (profit 14) (profit 15)))

theorem maximize_profit_under_constraints : max_profit = 8800 :=
by
  let x := 12 -- This is where the max profit is achieved.
  have : x ∈ {12, 13, 14, 15}, by sorry
  have h₁ : 1600 * x + 2500 * (20 - x) ≤ 39200 := by sorry
  have h₂ : prof x≥ 8500 := by sorry
  have : max_profit = prof x :=
    begin
      apply nat.max_eq,
      sorry
    end
  sorry

end maximize_profit_under_constraints_l23_23951


namespace range_of_m_l23_23360

def A (m : ℝ) : set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ (m / 2 ≤ (x - 2) ^ 2 + y ^ 2 ∧ (x - 2) ^ 2 + y ^ 2 ≤ m ^ 2)}

def B (m : ℝ) : set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ (2 * m ≤ x + y ∧ x + y ≤ 2 * m + 1)}

theorem range_of_m (m : ℝ) :
  (∃ (p : ℝ × ℝ), p ∈ A m ∧ p ∈ B m) ↔ (0.5 ≤ m ∧ m ≤ 2 + real.sqrt 2) :=
sorry

end range_of_m_l23_23360


namespace divide_to_equal_parts_l23_23446

structure Cube (n : ℕ) :=
  (structure : ℕ → ℕ → ℕ → bool) -- a function defining the presence of a unit cube at each position

def largeCube : Cube 3 := ⟨λ x y z, x < 3 ∧ y < 3 ∧ z < 3⟩

def gluedCube : Cube 1 := ⟨λ x y z, x = 1 ∧ y = 1 ∧ z = 3⟩

def combinedFigure : Cube 4 :=
  ⟨λ x y z,
    (x < 3 ∧ y < 3 ∧ z < 3) ∨
    (x = 1 ∧ y = 1 ∧ z = 3)⟩

theorem divide_to_equal_parts :
  ∃ parts : fin 7 → Cube 4,
    (∀ i : fin 7, (comb := combinedFigure.structure) ∧ 
                    (¬ comb.parts i.structure) ∧ 
                    ∀ x y z: ℕ, 
                    ((parts i).structure x y z → 
                       (x < 4 ∧ y < 4 ∧ z < 4))) ∧
    (∀ (i j : fin 7), i ≠ j → disjoint (parts i).structure (parts j).structure) ∧
    (∀ x y z : ℕ, combinedFigure.structure x y z ↔ ∃ i : fin 7, (parts i).structure x y z) := 
  sorry

end divide_to_equal_parts_l23_23446


namespace sam_used_10_pounds_of_spicy_meat_mix_l23_23458

theorem sam_used_10_pounds_of_spicy_meat_mix 
  (total_links : ℕ) 
  (eaten_links : ℕ) 
  (remaining_links : ℕ) 
  (remaining_weight : ℕ) 
  (ounces_per_pound : ℕ) 
  (H1 : total_links = 40)
  (H2 : eaten_links = 12)
  (H3 : remaining_links = total_links - eaten_links)
  (H4 : remaining_weight = 112)
  (H5 : ∀ l, l < total_links → ounces_per_saussage link = 4)
  (H6 : ounces_per_pound = 16) : 
  (total_links * 4 / ounces_per_pound) = 10 := 
   by sorry

end sam_used_10_pounds_of_spicy_meat_mix_l23_23458


namespace pattys_hamburger_varieties_l23_23447

theorem pattys_hamburger_varieties :
  let condiments := 8
  let condiment_combinations := 2 ^ condiments
  let patty_options := 4
  condiment_combinations * patty_options = 1024 :=
by 
  let condiments := 8
  let condiment_combinations := 2 ^ condiments
  let patty_options := 4
  calc
  condiment_combinations * patty_options
  = (2 ^ 8) * 4      : rfl
  = 256 * 4          : by norm_num
  = 1024             : by norm_num

end pattys_hamburger_varieties_l23_23447


namespace inverse_of_f_is_neg_g_neg_l23_23487

variable {X : Type}
variable (f g : X → X)

def symmetric_about_line (h k : X → X) (L : X × X → Prop) : Prop :=
  ∀ (P : X × X), L P → ∃ P' : X × X, L P' ∧ h P.1 = k P'.2 ∧ h P.2 = k P'.1

theorem inverse_of_f_is_neg_g_neg
    (hf_sym : symmetric_about_line f g (λ P, P.1 + P.2 = 0)) :
  ∀ (y : X), (∃ x, f x = y) ↔ (∃ x, y = - g (- x)) :=
by
  sorry

end inverse_of_f_is_neg_g_neg_l23_23487


namespace sum_of_solutions_is_267_l23_23058

open Set

noncomputable def inequality (x : ℝ) : Prop :=
  sqrt (x^2 + x - 56) - sqrt (x^2 + 25*x + 136) < 8 * sqrt ((x - 7) / (x + 8))

noncomputable def valid_integers : Set ℝ :=
  {x | x ∈ Icc (-25 : ℝ) 25 ∧ (x ∈ (-20 : ℝ, -18) ∨ x ∈ Ici (7 : ℝ))}

theorem sum_of_solutions_is_267 :
  ∑ i in (Icc (-25 : ℝ) 25).to_finset.filter (λ x, inequality x), x = 267 :=
sorry

end sum_of_solutions_is_267_l23_23058


namespace minimum_balls_drawn_l23_23573

theorem minimum_balls_drawn (
  red green yellow blue white purple : ℕ
) (h_red : red = 30)
  (h_green : green = 24)
  (h_yellow : yellow = 16)
  (h_blue : blue = 14)
  (h_white : white = 12)
  (h_purple : purple = 4) :
  ∃ (n : ℕ), n = 60 ∧ (∀ drawn : ℕ, drawn ≥ n → ∃ color : ℕ, (if color = 1 then drawn ≥ h_red else if color = 2 then drawn ≥ h_green else if color = 3 then drawn ≥ h_yellow else if color = 4 then drawn ≥ h_blue else if color = 5 then drawn ≥ h_white else drawn ≥ h_purple) >= 12) :=
by {
  sorry
}

end minimum_balls_drawn_l23_23573


namespace three_digit_number_property_l23_23071

theorem three_digit_number_property (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) (h_neq : a ≠ c) : 
  let n := 100 * a + 10 * b + c in
  let rev_n := 100 * c + 10 * b + a in
  let diff := if n > rev_n then n - rev_n else rev_n - n in
  let rev_diff := let d := diff in 100 * (d % 10) + 10 * ((d / 10) % 10) + (d / 100) in
  diff + rev_diff = 1089 :=
by
  sorry

end three_digit_number_property_l23_23071


namespace min_value_x_plus_y_l23_23293

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 / x + 1 / y = 1 / 2) : x + y ≥ 18 := sorry

end min_value_x_plus_y_l23_23293


namespace total_goals_proof_l23_23627

-- Definitions based on the conditions
def first_half_team_a := 8
def first_half_team_b := first_half_team_a / 2
def first_half_team_c := first_half_team_b * 2

def second_half_team_a := first_half_team_c
def second_half_team_b := first_half_team_a
def second_half_team_c := second_half_team_b + 3

-- Total scores for each team
def total_team_a := first_half_team_a + second_half_team_a
def total_team_b := first_half_team_b + second_half_team_b
def total_team_c := first_half_team_c + second_half_team_c

-- Total goals for all teams
def total_goals := total_team_a + total_team_b + total_team_c

-- The theorem to be proved
theorem total_goals_proof : total_goals = 47 := by
  sorry

end total_goals_proof_l23_23627


namespace find_f_neg1_l23_23860

-- Define the function f and its inverse relationship with logarithm
def f (x : ℝ) : ℝ := sorry  -- f should be defined based on log_2 but we skip the definition here

-- Conditions mentioned in the problem
axiom inv_function : ∀ y : ℝ, f y = 2 ^ y
axiom log_condition : logBase 2 (x : ℝ) = -1

-- Required proof statement
theorem find_f_neg1 : f (-1) = 1 / 2 :=
by
  -- Skip the actual proof here
  sorry

end find_f_neg1_l23_23860


namespace number_of_tests_initially_l23_23400

theorem number_of_tests_initially (n : ℕ) (h1 : (90 * n) / n = 90)
  (h2 : ((90 * n) - 75) / (n - 1) = 95) : n = 4 :=
sorry

end number_of_tests_initially_l23_23400


namespace mode_I_swaps_mode_II_swaps_l23_23784

-- Define the original and target strings
def original_sign := "MEGYEI TAKARÉKPÉNZTÁR R. T."
def target_sign := "TATÁR GYERMEK A PÉNZT KÉRI."

-- Define a function for adjacent swaps needed to convert original_sign to target_sign
def adjacent_swaps (orig : String) (target : String) : ℕ := sorry

-- Define a function for any distant swaps needed to convert original_sign to target_sign
def distant_swaps (orig : String) (target : String) : ℕ := sorry

-- The theorems we want to prove
theorem mode_I_swaps : adjacent_swaps original_sign target_sign = 85 := sorry

theorem mode_II_swaps : distant_swaps original_sign target_sign = 11 := sorry

end mode_I_swaps_mode_II_swaps_l23_23784


namespace correct_statement_is_B_l23_23653

-- Defining the conditions
variable (fatContent age : Type)
variable (correlFatAge : ∃ a, correlation fatContent a)
variable (carWeight distancePerLiter : Type)
variable (negCorrelWeightDistance : correlation carWeight distancePerLiter < 0)
variable (smoking health : Type)
variable (negCorrelSmokingHealth : correlation smoking health < 0)
variable (temperature hotDrinkSales : Type)
variable (negCorrelTempHotDrinkSales : correlation temperature hotDrinkSales < 0)

-- Theorem statement
theorem correct_statement_is_B : correlation carWeight distancePerLiter < 0 := by
  -- Proof omitted
  sorry

end correct_statement_is_B_l23_23653


namespace prob_difference_l23_23572

-- Define the box with specific counts of marbles
def red_marbles := 1500
def black_marbles := 1500
def blue_marbles := 1500
def total_marbles := red_marbles + black_marbles + blue_marbles

-- Define the probabilities for the problem
noncomputable def Ps : ℚ :=
  ( (red_marbles * (red_marbles - 1)) / 2  + (black_marbles * (black_marbles - 1)) / 2 + (blue_marbles * (blue_marbles - 1)) / 2 ) /
  ((total_marbles * (total_marbles - 1)) / 2)

noncomputable def Pd : ℚ :=
  (red_marbles * black_marbles + red_marbles * blue_marbles + black_marbles * blue_marbles) /
  ((total_marbles * (total_marbles - 1)) / 2)

-- State the goal to prove
theorem prob_difference : | Ps - Pd | = 1 / 3 :=
  by
  sorry

end prob_difference_l23_23572


namespace least_k_2011_l23_23637

theorem least_k_2011 : 
  ∀ (M : Matrix (Fin 2011) (Fin 2011) ℤ), ∃ (M' : Matrix (Fin 2011) (Fin 2011) ℤ), 
  (∀ (r : Fin 2011), (M'.rowSum r).sum ≠ (M'.colSum r).sum ∨ r = (4022 - r)) ∧
  (∃ (k : ℕ), k ≤ 2681 ∧ (∃ r c, M r c ≠ M' r c)) := 
sorry

end least_k_2011_l23_23637


namespace sequence_not_all_distinct_l23_23298

theorem sequence_not_all_distinct {P : ℝ[X]} (degree_2003 : P.degree = 2003)
    (leading_coeff_1 : P.leading_coeff = 1)
    (a : ℕ → ℤ)
    (h₁ : P.eval (a 1) = 0)
    (h₂ : ∀ n > 0, P.eval (a (n + 1)) = a n) :
    ∃ i j : ℕ, i ≠ j ∧ a i = a j :=
begin
  sorry
end

end sequence_not_all_distinct_l23_23298


namespace susie_earnings_l23_23471

-- Define the constants and conditions
def price_per_slice : ℕ := 3
def price_per_whole_pizza : ℕ := 15
def slices_sold : ℕ := 24
def whole_pizzas_sold : ℕ := 3

-- Calculate earnings from slices and whole pizzas
def earnings_from_slices : ℕ := slices_sold * price_per_slice
def earnings_from_whole_pizzas : ℕ := whole_pizzas_sold * price_per_whole_pizza
def total_earnings : ℕ := earnings_from_slices + earnings_from_whole_pizzas

-- Prove that the total earnings are $117
theorem susie_earnings : total_earnings = 117 := by
  sorry

end susie_earnings_l23_23471


namespace cuboid_unshaded_face_area_l23_23853

theorem cuboid_unshaded_face_area 
  (x : ℝ)
  (h1 : ∀ a  : ℝ, a = 4*x) -- Condition: each unshaded face area = 4 * shaded face area
  (h2 : 18*x = 72)         -- Condition: total surface area = 72 cm²
  : 4*x = 16 :=            -- Conclusion: area of one visible unshaded face is 16 cm²
by
  sorry

end cuboid_unshaded_face_area_l23_23853


namespace total_tickets_l23_23513

theorem total_tickets (A C : ℕ) (cost_adult cost_child total_cost : ℝ) 
  (h1 : cost_adult = 5.50) 
  (h2 : cost_child = 3.50) 
  (h3 : C = 16) 
  (h4 : total_cost = 83.50) 
  (h5 : cost_adult * A + cost_child * C = total_cost) : 
  A + C = 21 := 
by 
  sorry

end total_tickets_l23_23513


namespace area_rhombus_abs_eq_9_l23_23901

open Real

theorem area_rhombus_abs_eq_9 :
  (∃ d1 d2 : ℝ, (d1 = 18) ∧ (d2 = 6) ∧ (1/2 * d1 * d2 = 54)) :=
begin
  use 18,
  use 6,
  split,
  { refl },
  split,
  { refl },
  { norm_num }
end

end area_rhombus_abs_eq_9_l23_23901


namespace probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l23_23765

theorem probability_exactly_k_gnomes_fall (n k : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in p * q^(n - k) = p * (1 - p)^(n - k) := 
sorry

theorem expected_number_of_gnomes_fall (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in 
  (∑ j in finset.range n, (1 - q^(j+1))) = n + 1 - (1 / p) + ((1 - p)^(n+1) / p) :=
sorry

end probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l23_23765


namespace tangent_line_at_1_monotonicity_conditions_range_of_m_l23_23348

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (2 - a) * Real.log x + 1 / x + 2 * a * x

theorem tangent_line_at_1 (a : ℝ) (h : a = 0) :
  ∃ m b, ∀ x, f x a = m * x + b → m = 1 ∧ b = -1 :=
sorry

theorem monotonicity_conditions (a : ℝ) (h : a < 0) :
  (a > -2 ∧ increasing_on (Icc 1 (1 / a)) (f x a) ∧ decreasing_on (Ioc (1 / a) ∞) (f x a)) ∨
  (a = -2 ∧ decreasing_on (Ioi 0) (f x a)) ∨
  (a < -2 ∧ increasing_on (Ioo 0 (1 / (a))) (f x a) ∧ decreasing_on (Ici (1 / a)) (f x a)) :=
sorry

theorem range_of_m (m a : ℝ) (h1 : a ∈ Ioo (-3 : ℝ) (-2 : ℝ)) (x1 x2 : ℝ) (h2 : x1 ∈ Icc (1 : ℝ) (3 : ℝ)) (h3 : x2 ∈ Icc (1 : ℝ) (3 : ℝ))
  (h : (m + Real.log 3) * a - 2 * Real.log 3 > abs (f x1 a - f x2 a)) :     
  m <= -13 / 3 :=
sorry

end tangent_line_at_1_monotonicity_conditions_range_of_m_l23_23348


namespace parity_negative_triangles_l23_23670

-- Define a set M of n points in the plane
variable {M : Set (ℝ × ℝ)} (n m : ℕ)

-- Assume no three points in M are collinear
axiom no_three_collinear : ∀ (a b c : ℝ × ℝ), a ∈ M → b ∈ M → c ∈ M → 
                            (¬ collinear a b c ∨ a = b ∨ b = c ∨ a = c)

-- Every line segment connecting two points in M is labeled with either +1 or -1
variable (label : (ℝ × ℝ) → (ℝ × ℝ) → ℤ)
axiom label_one_neg_one : ∀ (a b : ℝ × ℝ), a ∈ M → b ∈ M → 
                          (label a b = 1 ∨ label a b = -1)

-- The number of segments labeled -1 is m
axiom num_neg_segments : ∃ (neg_segments : Set ((ℝ × ℝ) × (ℝ × ℝ))),
                          (∀ seg ∈ neg_segments, label seg.1 seg.2 = -1) ∧ 
                          (neg_segments.pairwise (λ x y, x.1 ≠ y.1 ∧ x.2 ≠ y.2)) ∧ 
                          (neg_segments.card = m)

-- A triangle is negative if the product of its edge labels is -1
def negative_triangle (a b c : ℝ × ℝ) : Prop :=
  (label a b) * (label b c) * (label c a) = -1

-- Prove the number of negative triangles is congruent to the product mn mod 2
theorem parity_negative_triangles {k : ℕ} :
  (∃ (triangles : Set (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)), 
    (triangles.card = k ∧ ∀ (t ∈ triangles), negative_triangle t.1 t.2 t.3)) → 
  k ≡ n * m [MOD 2] :=
sorry

end parity_negative_triangles_l23_23670


namespace find_S_9_l23_23672

-- Defining the arithmetic sequence with sum of the first n terms Sn and specific term an
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n+1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
axiom S_15 : S 15 = 30
axiom a_7 : a 6 = 1
axiom sum_arithmetic : ∀ n, S n = n * (a 0 + a (n-1)) / 2

-- Problem statement to prove that S9 = -9
theorem find_S_9 : is_arithmetic_sequence a → S 9 = -9 :=
by 
  intros h_arith_seq,
  have hp1 := S_15,
  have hp2 := a_7,
  have hp3 := sum_arithmetic,
  sorry

end find_S_9_l23_23672


namespace similar_triangles_A2B2C2_ABC_l23_23427

variables (A B C A1 B1 C1 A2 B2 C2 : Type*)
variables [scalene_triangle : scalene_triangle ABC]
variables [similar_triangles : similar_triangle (triangle.mk A B C) (triangle.mk A1 B1 C1)]

-- Define points on lines
variables [is_on_segment : is_on_segment A1 B C]
variables [is_on_segment : is_on_segment B1 C A]
variables [is_on_segment : is_on_segment C1 A B]

-- Define similar triangles
variables [triangle_similarity : triangle_similarity (triangle.mk A B C) (triangle.mk A1 B1 C1)]

-- Define point A2 on B1C1 such that AA2 = A1A2
variables [is_on_line_A2 : is_on_line A2 B1 C1]
variables [equal_dist_A2 : equal_distance (distance A A2) (distance A1 A2)]

variables [is_on_line_B2 : is_on_line B2 A1 C1]
variables [equal_dist_B2 : equal_distance (distance B B2) (distance B1 B2)]

variables [is_on_line_C2 : is_on_line C2 A1 B1]
variables [equal_dist_C2 : equal_distance (distance C C2) (distance C1 C2)]

theorem similar_triangles_A2B2C2_ABC :
  similar_triangle (triangle.mk A2 B2 C2) (triangle.mk A B C) :=
sorry

end similar_triangles_A2B2C2_ABC_l23_23427


namespace triangle_area_30_26_10_l23_23381

theorem triangle_area_30_26_10 :
    let a := 30
    let b := 26
    let c := 10
    let s := (a + b + c) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
    area ≈ 126.72 :=
by
    let a := 30
    let b := 26
    let c := 10
    let s := (a + b + c) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
    have h : area ≈ 126.72 := by
        sorry
    exact h

end triangle_area_30_26_10_l23_23381


namespace probability_of_getting_exactly_5_heads_l23_23374

noncomputable def num_ways_to_get_heads (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_of_getting_exactly_5_heads :
  let total_outcomes := 2 ^ 10
  let num_heads_5 := num_ways_to_get_heads 10 5
  let probability := num_heads_5 / total_outcomes
  probability = (63 : ℚ) / 256 :=
by
  sorry

end probability_of_getting_exactly_5_heads_l23_23374


namespace max_n_value_l23_23695

noncomputable def f (x : ℝ) : ℝ := x + 4 / x - 1

theorem max_n_value (x : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → x i ∈ set.Icc (1/4 : ℝ) 4)
  (h2 : ∑ i in finset.range (n - 1), f (x i) = f (x n)) :
  n ≤ 6 :=
begin
  sorry
end

end max_n_value_l23_23695


namespace coins_after_10_hours_l23_23891

def numberOfCoinsRemaining : Nat :=
  let hour1_coins := 20
  let hour2_coins := hour1_coins + 30
  let hour3_coins := hour2_coins + 30
  let hour4_coins := hour3_coins + 40
  let hour5_coins := hour4_coins - (hour4_coins * 20 / 100)
  let hour6_coins := hour5_coins + 50
  let hour7_coins := hour6_coins + 60
  let hour8_coins := hour7_coins - (hour7_coins / 5)
  let hour9_coins := hour8_coins + 70
  let hour10_coins := hour9_coins - (hour9_coins * 15 / 100)
  hour10_coins

theorem coins_after_10_hours : numberOfCoinsRemaining = 200 := by
  sorry

end coins_after_10_hours_l23_23891


namespace vector_expression_eval_l23_23249

open Real

noncomputable def v1 : ℝ × ℝ := (3, -8)
noncomputable def v2 : ℝ × ℝ := (2, -4)
noncomputable def k : ℝ := 5

theorem vector_expression_eval : (v1.1 - k * v2.1, v1.2 - k * v2.2) = (-7, 12) :=
  by sorry

end vector_expression_eval_l23_23249


namespace sound_level_properties_l23_23114

theorem sound_level_properties
  (lg : ℝ → ℝ)
  (I : ℝ)
  (a b L1 : ℝ)
  (Lnormal : ℝ)
  (IT : ℝ)
  (H1 : L1 = a + b * lg I)
  (H2 : L1 = 120)            -- When I = 1 W/m^2
  (H3 : L1 = 0)              -- When I = 10^-12 W/m^2
  (H4 : I = 10^-6) 
  (H5 : L1 = 80) 
  : (L1 = 10 * lg (10^(12) * I)) ∧ 
    (I = (10^(1/10))^(L1 - 120)) ∧ 
    (Lnormal = 60) ∧ 
    (IT = 10^-4) :=
by
  sorry

end sound_level_properties_l23_23114


namespace algebraic_expr_pos_int_vals_l23_23277

noncomputable def algebraic_expr_ineq (x : ℕ) : Prop :=
  x > 0 ∧ ((x + 1)/3 - (2*x - 1)/4 ≥ (x - 3)/6)

theorem algebraic_expr_pos_int_vals : {x : ℕ | algebraic_expr_ineq x} = {1, 2, 3} :=
sorry

end algebraic_expr_pos_int_vals_l23_23277


namespace f_10_eq_83_l23_23191

def f : ℕ → ℕ
| 1       := 2
| 2       := 3
| (n + 3) := 2 * f (n + 2) - f (n + 1) + 2 * (n + 3)

theorem f_10_eq_83 : f 10 = 83 :=
by
  sorry

end f_10_eq_83_l23_23191


namespace women_lawyers_percentage_l23_23946

theorem women_lawyers_percentage (T : ℕ) (h1 : 0.40 * T = w) (h2 : 0.08 = (L * 0.40)) : L = 0.20 :=
by
  sorry

end women_lawyers_percentage_l23_23946


namespace lame_king_max_visits_l23_23196

-- Define the problem specifics
def is_valid_king_move (current target : ℕ × ℕ) : Prop :=
  let (cx, cy) := current in
  let (tx, ty) := target in
  (abs (int.cx - int.tx) ≤ 1 ∧ abs (int.cy - int.ty) ≤ 1) ∧ (cx ≠ tx)

def lame_king_max_cells (n : ℕ) (start : ℕ × ℕ) : ℕ :=
  if n = 7 ∧ start = (0, 0) then 43
  else 0

-- The main theorem to prove
theorem lame_king_max_visits : lame_king_max_cells 7 (0, 0) = 43 :=
  by
    sorry  -- proof omitted

end lame_king_max_visits_l23_23196


namespace age_of_15th_student_l23_23075

theorem age_of_15th_student 
  (total_age_15_students : ℕ)
  (total_age_3_students : ℕ)
  (total_age_11_students : ℕ)
  (h1 : total_age_15_students = 225)
  (h2 : total_age_3_students = 42)
  (h3 : total_age_11_students = 176) :
  total_age_15_students - (total_age_3_students + total_age_11_students) = 7 :=
by
  sorry

end age_of_15th_student_l23_23075


namespace at_most_one_cube_l23_23504

theorem at_most_one_cube (a : ℕ → ℕ) (h₁ : ∀ n, a (n + 1) = a n ^ 2 + 2018) :
  ∃! n, ∃ m : ℕ, a n = m ^ 3 := sorry

end at_most_one_cube_l23_23504


namespace total_prime_factors_4_to_11_mul_7_to_3_mul_11_to_2_l23_23266

theorem total_prime_factors_4_to_11_mul_7_to_3_mul_11_to_2 :
  (let expr := ((4: ℕ)^(11) * (7: ℕ)^(3) * (11: ℕ)^(2)) in
   ∑ p in (multiset.to_finset (multiset.replicate 22 2 + multiset.replicate 3 7 + multiset.replicate 2 11)), multiset.count p (multiset.replicate 22 2 + multiset.replicate 3 7 + multiset.replicate 2 11)) = 27 :=
begin
  sorry
end

end total_prime_factors_4_to_11_mul_7_to_3_mul_11_to_2_l23_23266


namespace BM_squared_eq_Delta_cot_B_div_2_l23_23958

-- Definitions for points and triangles
variable {A B C M : ℝ}
variable {areaABC areaABM areaBMC : ℝ}
variable {rABC rABM rBMC : ℝ}

-- Assumptions
variables (H1 : M ∈ segment A C)
variables (H2 : inscribed_circle_radius ABM = inscribed_circle_radius BMC)

-- Define the final goal
theorem BM_squared_eq_Delta_cot_B_div_2 
  (H1 : M ∈ segment A C) 
  (H2 : inscribed_circle_radius ABM = inscribed_circle_radius BMC) 
  (areaABC : ℝ) 
  (cot_B_div_2 : ℝ) 
  (Delta : ℝ) : 
  BM^2 = Delta * cot_B_div_2 := 
sorry

end BM_squared_eq_Delta_cot_B_div_2_l23_23958


namespace total_spent_christy_tanya_l23_23225

def total_cost_face_moisturizer (quantity : ℕ) (price : ℕ) (discount : ℕ) : ℕ :=
  let total := quantity * price
  total - (total * discount / 100)

def total_cost_body_lotion (quantity : ℕ) (price : ℕ) (discount : ℕ) : ℕ :=
  let total := quantity * price
  total - (total * discount / 100)

theorem total_spent_christy_tanya :
  let tanya_face := total_cost_face_moisturizer 2 50 10 in
  let tanya_lotion := total_cost_body_lotion 4 60 15 in
  let tanya_spent := tanya_face + tanya_lotion in

  let christy_face := total_cost_face_moisturizer 3 50 10 in
  let christy_lotion := total_cost_body_lotion 5 60 15 in
  let christy_spent := christy_face + christy_lotion in

  let total_spent := tanya_spent + christy_spent in

  christy_spent = 2 * tanya_spent →
  total_spent = 684 :=
by 
  intros
  sorry

end total_spent_christy_tanya_l23_23225


namespace part1_part2_part3_part3_expectation_l23_23451

/-- Conditions setup -/
noncomputable def gameCondition (Aacc Bacc : ℝ) :=
  (Aacc = 0.5) ∧ (Bacc = 0.6)

def scoreDist (X:ℤ) : ℝ :=
  if X = -1 then 0.3
  else if X = 0 then 0.5
  else if X = 1 then 0.2
  else 0

def tieProbability : ℝ := 0.2569

def roundDist (Y:ℤ) : ℝ :=
  if Y = 2 then 0.13
  else if Y = 3 then 0.13
  else if Y = 4 then 0.74
  else 0

def roundExpectation : ℝ := 3.61

/-- Proof Statements -/
theorem part1 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (X : ℤ), scoreDist X = if X = -1 then 0.3 else if X = 0 then 0.5 else if X = 1 then 0.2 else 0 :=
by sorry

theorem part2 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  tieProbability = 0.2569 :=
by sorry

theorem part3 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (Y : ℤ), roundDist Y = if Y = 2 then 0.13 else if Y = 3 then 0.13 else if Y = 4 then 0.74 else 0 :=
by sorry

theorem part3_expectation (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) :
  roundExpectation = 3.61 :=
by sorry

end part1_part2_part3_part3_expectation_l23_23451


namespace sqrt_meaningful_range_l23_23728

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 2) : x ≥ 2 :=
sorry

end sqrt_meaningful_range_l23_23728


namespace pretty_number_theorem_verify_ratio_l23_23222

def is_12_pretty (n : ℕ) : Prop :=
  n > 0 ∧ (∃ d, nat.totient n = d ∧ d = 12) ∧ (n % 12 = 0)

noncomputable def sum_12_pretty_under_1000 : ℕ :=
  ∑ n in finset.range 1000, if is_12_pretty n then n else 0

theorem pretty_number_theorem : sum_12_pretty_under_1000 = 486 :=
by sorry

theorem verify_ratio : (sum_12_pretty_under_1000 : ℚ) / 12 = 40.5 :=
by sorry

end pretty_number_theorem_verify_ratio_l23_23222


namespace simplify_expression_l23_23467

theorem simplify_expression : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
    ((Real.sqrt 3 - 2 * Real.sqrt 5 - 1) * (-16 - 2 * Real.sqrt 3)) / 244 := 
  sorry

end simplify_expression_l23_23467


namespace boris_clock_time_l23_23164

-- Define a function to compute the sum of digits of a number.
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the problem
theorem boris_clock_time (h m : ℕ) :
  sum_digits h + sum_digits m = 6 ∧ h + m = 15 ↔
  (h, m) = (0, 15) ∨ (h, m) = (1, 14) ∨ (h, m) = (2, 13) ∨ (h, m) = (3, 12) ∨
  (h, m) = (4, 11) ∨ (h, m) = (5, 10) ∨ (h, m) = (10, 5) ∨ (h, m) = (11, 4) ∨
  (h, m) = (12, 3) ∨ (h, m) = (13, 2) ∨ (h, m) = (14, 1) ∨ (h, m) = (15, 0) :=
by sorry

end boris_clock_time_l23_23164


namespace base_s_computation_l23_23392

theorem base_s_computation (s : ℕ) (h : 550 * s + 420 * s = 1100 * s) : s = 7 := by
  sorry

end base_s_computation_l23_23392


namespace part1_boundary_function_part2_boundary_function_l23_23778

def is_boundary_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ set.Icc (a - 1) (a + 1), ∃ y ∈ set.Icc (b - 1) (b + 1), f x = y

theorem part1_boundary_function (a b : ℝ) :
  is_boundary_function (λ x, x) a b → a = b :=
by
  intros h
  have h₁ : set.Icc (a - 1) (a + 1) ⊆ set.Icc (b - 1) (b + 1),
  { exact h }
  sorry

theorem part2_boundary_function (m : ℝ) :
  ∀ n, n = - (1/2) * m^2 →
  is_boundary_function (λ x, - (1/2) * x^2) m n → 
  - (1/2 : ℝ) ≤ m ∧ m ≤ (1/2 : ℝ) :=
by
  intros n hn h
  sorry

end part1_boundary_function_part2_boundary_function_l23_23778


namespace tickets_spent_correct_l23_23216

/-- Tom won 32 tickets playing 'whack a mole'. -/
def tickets_whack_mole : ℕ := 32

/-- Tom won 25 tickets playing 'skee ball'. -/
def tickets_skee_ball : ℕ := 25

/-- Tom is left with 50 tickets after spending some on a hat. -/
def tickets_left : ℕ := 50

/-- The total number of tickets Tom won from both games. -/
def tickets_total : ℕ := tickets_whack_mole + tickets_skee_ball

/-- The number of tickets Tom spent on the hat. -/
def tickets_spent : ℕ := tickets_total - tickets_left

-- Prove that the number of tickets Tom spent on the hat is 7.
theorem tickets_spent_correct : tickets_spent = 7 := by
  -- Proof goes here
  sorry

end tickets_spent_correct_l23_23216


namespace inscribed_circle_radius_l23_23779

theorem inscribed_circle_radius (R : ℝ) (hR : 0 < R) : 
  ∃ (x : ℝ), x = R * (sqrt 2 - 1) := 
sorry

end inscribed_circle_radius_l23_23779


namespace cameron_list_count_l23_23996

theorem cameron_list_count :
  let numbers := {n : ℕ | 30 ≤ n ∧ n ≤ 900}
  in set.card numbers = 871 :=
sorry -- proof is omitted

end cameron_list_count_l23_23996


namespace eccentricity_of_hyperbola_is_five_l23_23409

theorem eccentricity_of_hyperbola_is_five (a b : ℝ) (ha : a > 0) (hb : b > 0) 
(F1 F2 P : ℝ × ℝ)
(P_on_hyperbola : P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1)
(hyperbola_foci : F1 = (-c, 0) ∧ F2 = (c, 0) ∧ c = sqrt(a ^ 2 + b ^ 2))
(PF1_perp_PF2 : (P.1 - F1.1, P.2 - F1.2) • (P.1 - F2.1, P.2 - F2.2) = 0)
(triangle_arith_seq : PF1 = PF2 + 2a) : 
sqrt (a ^ 2 + b ^ 2) / a = 5 := 
sorry

end eccentricity_of_hyperbola_is_five_l23_23409


namespace sum_of_first_ten_nice_numbers_l23_23590

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def is_nice (n : ℕ) : Prop :=
  (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p * q) ∨
  (∃ (p : ℕ), is_prime p ∧ n = p^3)

theorem sum_of_first_ten_nice_numbers : 
  (6 + 8 + 10 + 14 + 15 + 21 + 22 + 26 + 27 + 33) = 182 :=
by {
  -- Summing up the values
  have h : (6 + 8 + 10 + 14 + 15 + 21 + 22 + 26 + 27 + 33) = 182, from sorry,
  exact h,
}

end sum_of_first_ten_nice_numbers_l23_23590


namespace smallest_k_for_polygon_l23_23363

-- Definitions and conditions
def equiangular_decagon_interior_angle : ℝ := 144

-- Question transformation into a proof problem
theorem smallest_k_for_polygon (k : ℕ) (hk : k > 1) :
  (∀ (n2 : ℕ), n2 = 10 * k → ∃ (interior_angle : ℝ), interior_angle = k * equiangular_decagon_interior_angle ∧
  n2 ≥ 3) → k = 2 :=
by
  sorry

end smallest_k_for_polygon_l23_23363


namespace camille_total_birds_l23_23223

theorem camille_total_birds :
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  let finches := robins / 2
  cardinals + robins + blue_jays + sparrows + pigeons + finches = 55 :=
by
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  let finches := robins / 2
  show cardinals + robins + blue_jays + sparrows + pigeons + finches = 55
  sorry

end camille_total_birds_l23_23223


namespace perimeter_six_triangles_l23_23618

def side_length (i : ℕ) : ℝ := 1 / 2^(i - 1)

def perimeter (n : ℕ) : ℝ := ∑ i in finset.range n, 2 * side_length (i + 1)

theorem perimeter_six_triangles :
  perimeter 6 = (127 / 32 : ℝ) :=
by
  sorry

end perimeter_six_triangles_l23_23618


namespace num_factors_of_48_multiple_of_6_l23_23718

theorem num_factors_of_48_multiple_of_6 :
  let factors_of_48 := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48] in
  let multiples_of_6 := filter (fun x => x % 6 = 0) factors_of_48 in
  multiples_of_6.length = 4 := by
  sorry

end num_factors_of_48_multiple_of_6_l23_23718


namespace angle_BVU_in_square_is_16_degrees_l23_23548

theorem angle_BVU_in_square_is_16_degrees
  (ABCD : Type)
  [square : is_square ABCD]
  (U D A B : Points)
  (angle_UDA : angle U D A = 29) : 
  angle B V U = 16 :=
sorry

end angle_BVU_in_square_is_16_degrees_l23_23548


namespace find_sixth_term_l23_23302

open Nat

-- Given conditions
def arithmetic_progression (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ (n : ℕ), a (n + 1) = a n + d

def sum_of_first_three_terms (a : ℕ → ℤ) : Prop :=
  a 1 + a 2 + a 3 = 168

def second_minus_fifth (a : ℕ → ℤ) : Prop :=
  a 2 - a 5 = 42

-- Prove question == answer given conditions
theorem find_sixth_term :
  ∀ (a : ℕ → ℤ), arithmetic_progression a → sum_of_first_three_terms a → second_minus_fifth a → a 6 = 0 :=
by
  sorry

end find_sixth_term_l23_23302


namespace smallest_norm_v_l23_23415

-- Given definitions and conditions
variable (v : ℝ × ℝ)
def v_add_vector_norm_eq_10 := ∥⟨v.1 + 4, v.2 + 2⟩∥ = 10

-- The proof statement we need to prove
theorem smallest_norm_v (h : v_add_vector_norm_eq_10 v) : 
  ∥v∥ = 10 - 2 * Real.sqrt 5 :=
sorry

end smallest_norm_v_l23_23415


namespace symmetric_about_x_eq_2_l23_23335

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Icc (-2) (0) then 2^x - 2^(-x) + x else sorry  -- Placeholder for general definition outside [-2,5]

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem symmetric_about_x_eq_2 
  (h_even : ∀ x : ℝ, (x * f x) = (-x) * f (-x))
  (h_eq : ∀ x : ℝ, f(x-1) + f(x+3) = 0) 
  (h_f : ∀ x : ℝ, x ∈ Icc (-2) 0 → f x = 2^x - 2^(-x) + x) : 
  ∀ x : ℝ, f(x) = -f(x+4) := 

by
  sorry

end symmetric_about_x_eq_2_l23_23335


namespace f_neg2017_add_f_2018_eq_one_l23_23334

noncomputable def f : ℝ → ℝ := sorry -- Need to define the function f

-- Even function condition
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Periodic function condition
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Condition for [0, 2)
def f_def (x : ℝ) : ℝ := if (0 ≤ x ∧ x < 2) then Real.log2 (x + 1) else sorry -- To be made complete

-- The main theorem to prove
theorem f_neg2017_add_f_2018_eq_one :
  is_even f ∧ is_periodic f 2 ∧ (∀ x, 0 ≤ x ∧ x < 2 → f x = Real.log2 (x + 1)) →
  f (-2017) + f 2018 = 1 :=
by
  intro h
  sorry

end f_neg2017_add_f_2018_eq_one_l23_23334


namespace abs_x_equals_4_l23_23445

-- Define the points A and B as per the conditions
def point_A (x : ℝ) : ℝ := 3 + x
def point_B (x : ℝ) : ℝ := 3 - x

-- Define the distance between points A and B
def distance (x : ℝ) : ℝ := abs ((point_A x) - (point_B x))

theorem abs_x_equals_4 (x : ℝ) (h : distance x = 8) : abs x = 4 :=
by
  sorry

end abs_x_equals_4_l23_23445


namespace find_distance_l23_23888

variable (D : ℝ) (v1 v2 : ℝ)

-- The speeds of the first and second trains
def speed1 : ℝ := 75
def speed2 : ℝ := 44

-- The time difference condition
def time_diff_condition : Prop := 
  (D / speed2) - (D / speed1) = 4

-- The distance between Calcutta and Kanyakumari
theorem find_distance (h : time_diff_condition) : D = 13200 / 31 := by
  sorry

end find_distance_l23_23888


namespace greatest_factor_of_power_l23_23908

theorem greatest_factor_of_power (x : ℕ) : (∃ (n : ℕ), n > 0 ∧ 3 ^ n ∣ 9 ^ 7) → x = 14 :=
by
  -- conditions
  have h1 : 9 = 3 ^ 2 := by norm_num
  
  -- begin proof (partially)
  have h2 : 9 ^ 7 = (3 ^ 2) ^ 7 := by rw h1
  have h3 : (3 ^ 2) ^ 7 = 3 ^ 14 := by rw pow_mul
  have h4 : 3 ^ x ∣ 3 ^ 14 ↔ x ≤ 14 := by apply pow_dvd_pow_iff le_refl
  
  -- prove final statement
  existsi 14
  split
  { exact zero_lt_succ _ } -- n > 0
  { rw ←h3, exact pow_dvd_pow 3 (le_refl _) } -- 3^14 is a factor of 9^7
  
  -- given the greatest positive integer x, it matches our expectation
  sorry -- final proof step missed for demonstration purposes

end greatest_factor_of_power_l23_23908


namespace find_a6_l23_23315

variable (a_n : ℕ → ℤ) (d : ℤ)

-- Conditions
axiom sum_first_three_terms (S3 : a_n 1 + a_n 2 + a_n 3 = 168)
axiom diff_terms (diff_a2_a5 : a_n 2 - a_n 5 = 42)

-- Definition of arithmetic progression 
def arith_prog (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d

-- Proving that a6 = 3
theorem find_a6 (a1 : ℤ) (proof_S3 : a1 + (a1 + d) + (a1 + 2*d) = 168)
  (proof_diff : (a1 + d) - (a1 + 4*d) = 42) : a1 + 5*d = 3 :=
by
  sorry

end find_a6_l23_23315


namespace math_proof_problem_l23_23396

-- Conditions
def curve_C (α : ℝ) (x y : ℝ) : Prop :=
  (x = 2 * sqrt 3 * cos α) ∧ (y = 2 * sin α) ∧ (0 < α) ∧ (α < π)

def point_P_polar : ℝ × ℝ := (4 * sqrt 2, π / 4)

def line_l_polar_eq (ρ θ : ℝ) : Prop :=
  ρ * sin (θ - π / 4) + 5 * sqrt 2 = 0

-- Equivalent proof problem
theorem math_proof_problem :
  (∀ ρ θ : ℝ, (line_l_polar_eq ρ θ) → (ρ * sin θ - ρ * cos θ + 10 = 0)) ∧ 
  (∀ (α : ℝ) (x y : ℝ), (curve_C α x y) → (y > 0) → (x^2 / 12 + y^2 / 4 = 1)) ∧ 
  (∃ (α x y Mx My : ℝ), (curve_C α x y) ∧
    let P := point_P_polar in
    let M := (sqrt 3 * cos α + 2, sin α + 2) in
    let d := (|sqrt 3 * cos α - sin α - 10| / sqrt 2) in
    d = 6 * sqrt 2) :=
sorry

end math_proof_problem_l23_23396


namespace range_of_m_l23_23505

noncomputable def has_two_solutions (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 = x₁ + m ∧ x₂^2 = x₂ + m 

theorem range_of_m (m : ℝ) : has_two_solutions m ↔ m > -(1/4) :=
sorry

end range_of_m_l23_23505


namespace triangle_sides_are_3_4_5_l23_23874

-- Definitions of the problem's conditions
variables {a b c : ℕ} -- Side lengths of the triangle are natural numbers (integers)
variable (r : ℕ) -- Radius of the inscribed circle
noncomputable def semiperimeter := (a + b + c) / 2 -- Definition of the semiperimeter
noncomputable def area (p : ℕ) : ℝ := real.sqrt (p * (p - a) * (p - b) * (p - c)) -- Area using Heron's formula

-- Main Theorem Statement
theorem triangle_sides_are_3_4_5 (h_r : r = 1) (habc : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  let p := semiperimeter a b c in
  (a = 3 ∧ b = 4 ∧ c = 5) ∨ (a = 3 ∧ b = 5 ∧ c = 4) ∨
  (a = 4 ∧ b = 3 ∧ c = 5) ∨ (a = 4 ∧ b = 5 ∧ c = 3) ∨
  (a = 5 ∧ b = 3 ∧ c = 4) ∨ (a = 5 ∧ b = 4 ∧ c = 3) :=
begin
  sorry
end

end triangle_sides_are_3_4_5_l23_23874


namespace find_number_l23_23539

theorem find_number (n : ℕ) (h : n + 19 = 47) : n = 28 :=
by {
    sorry
}

end find_number_l23_23539


namespace temp_below_zero_negative_l23_23744

theorem temp_below_zero_negative (temp_below_zero : ℤ) : temp_below_zero = -3 ↔ temp_below_zero < 0 := by
  sorry

end temp_below_zero_negative_l23_23744


namespace range_of_m_l23_23285

def p (x : ℝ) : Prop := abs (1 - (x - 1) / 3) ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : 
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) → 9 ≤ m :=
by
  sorry

end range_of_m_l23_23285


namespace second_butcher_packages_l23_23495

theorem second_butcher_packages (a b c: ℕ) (weight_per_package total_weight: ℕ)
    (first_butcher_packages: ℕ) (third_butcher_packages: ℕ)
    (cond1: a = 10) (cond2: b = 8) (cond3: weight_per_package = 4)
    (cond4: total_weight = 100):
    c = (total_weight - (first_butcher_packages * weight_per_package + third_butcher_packages * weight_per_package)) / weight_per_package →
    c = 7 := 
by 
  have first_butcher_packages := 10
  have third_butcher_packages := 8
  have weight_per_package := 4
  have total_weight := 100
  sorry

end second_butcher_packages_l23_23495


namespace digits_problem_solution_l23_23862

def digits_proof_problem (E F G H : ℕ) : Prop :=
  (E, F, G) = (5, 0, 5) → H = 0

theorem digits_problem_solution 
  (E F G H : ℕ)
  (h1 : F + E = E ∨ F + E = E + 10)
  (h2 : E ≠ 0)
  (h3 : E = 5)
  (h4 : 5 + G = H)
  (h5 : 5 - G = 0) :
  H = 0 := 
by {
  sorry -- proof goes here
}

end digits_problem_solution_l23_23862


namespace sum_of_numbers_l23_23537

-- Definitions for the numbers involved
def n1 : Nat := 1235
def n2 : Nat := 2351
def n3 : Nat := 3512
def n4 : Nat := 5123

-- Proof statement
theorem sum_of_numbers :
  n1 + n2 + n3 + n4 = 12221 := by
  sorry

end sum_of_numbers_l23_23537


namespace smallest_positive_angle_l23_23644

theorem smallest_positive_angle (x : ℝ) (hx_pos : 0 < x) (hx_deg : x = 90 / 7): 
  ∃ (x : ℝ), (sin (3 * x) * sin (4 * x) = cos (3 * x) * cos (4 * x)) ∧ 0 < x ∧ x = 90 / 7 :=
sorry

end smallest_positive_angle_l23_23644


namespace quadratic_solution_symmetry_l23_23299

variable (a b c n : ℝ)
variable (h₀ : a ≠ 0)
variable (h₁ : a * (-5)^2 + b * (-5) + c = -2.79)
variable (h₂ : a * 1^2 + b * 1 + c = -2.79)
variable (h₃ : a * 2^2 + b * 2 + c = 0)
variable (h₄ : a * 3^2 + b * 3 + c = n)

theorem quadratic_solution_symmetry :
  (x = 3 ∨ x = -7) ↔ (a * x^2 + b * x + c = n) :=
sorry

end quadratic_solution_symmetry_l23_23299


namespace right_triangle_third_side_right_triangle_third_side_3_2_l23_23490

theorem right_triangle_third_side (a b c : ℝ) (h : a^2 + b^2 = c^2 ∨ c^2 + b^2 = a^2) :
  c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (a^2 - b^2) :=
by {
  sorry
}

-- Specific instance with given sides 3 and 2
theorem right_triangle_third_side_3_2 :
  ∃ c : ℝ, c = Real.sqrt (3^2 + 2^2) ∨ c = Real.sqrt (3^2 - 2^2) :=
begin
  use Real.sqrt (3^2 + 2^2),
  left,
  sorry
} ∨ ∃ c : ℝ, c = Real.sqrt (3^2 - 2^2) :=
begin
  use Real.sqrt (3^2 - 2^2),
  right,
  sorry
}


end right_triangle_third_side_right_triangle_third_side_3_2_l23_23490


namespace tables_made_this_month_l23_23178

theorem tables_made_this_month (T : ℕ) 
  (h1: ∀ t, t = T → t - 3 < t) 
  (h2 : T + (T - 3) = 17) :
  T = 10 := by
  sorry

end tables_made_this_month_l23_23178


namespace _l23_23489

noncomputable def no_possible_q : Prop :=
  ∀ (q : ℚ), ∀ (f : ℚ → ℚ), 
    (f = λ x => x^2 + a * x + b ∧
    (f.coeff 0 = 1 ∨ f.coeff 1 = 1 ∨ f.coeff 2 = 1) ∧ 
    (a = q ∧ b = q^2 ∨ a = q^2 ∧ b = q ∨ a = 1 ∧ b = q)) ∧
    (q > 0) →
    (q ≠ f.root_diff) 

noncomputable theorem proof_no_possible_q : no_possible_q :=
by sorry

end _l23_23489


namespace prob_X_distribution_prob_tie_prob_Y_distribution_expected_Y_l23_23449

def X := {-1, 0, 1}
def A_accuracy := 0.5
def B_accuracy := 0.6

theorem prob_X_distribution :
  ∀ (x : X),
  (x = -1) → (P(X = -1) = 0.3) ∧
  (x = 0) → (P(X = 0) = 0.5) ∧
  (x = 1) → (P(X = 1) = 0.2) := by sorry

theorem prob_tie :
  P(tie) = 0.2569 := by sorry

def Y := {2, 3, 4}

theorem prob_Y_distribution :
  ∀ (y : Y),
  (y = 2) → (P(Y = 2) = 0.13) ∧
  (y = 3) → (P(Y = 3) = 0.13) ∧
  (y = 4) → (P(Y = 4) = 0.74) := by sorry

theorem expected_Y :
  E(Y) = 3.61 := by sorry

end prob_X_distribution_prob_tie_prob_Y_distribution_expected_Y_l23_23449


namespace time_to_empty_tank_l23_23963

theorem time_to_empty_tank (x y m : ℝ) (h1 : 1 + 9 / x = 9 / y)
                             (h2 : 1 + 3 / x = 3 * 2 / y) :
  m = 9 / 5 :=
by
  have hx : x = 9, from by sorry
  have hy : y = 9 / 2, from by sorry
  have : 1 + m / 9 = 3 * 2 / (9 / 2) * m, from by sorry
  have : 1 + m / 9 = 6 / (9 / 2) * m, from by sorry
  have : 1 + m / 9 = 4/3 * m, from by sorry
  have : 1 + m / 9 - 1 = 4/3 * m - 1, from by sorry
  have : m / 9 = 4/3 * m - 1, from by sorry
  have : 3 * m / 9 = 4 * m - 3, from by sorry
  have : m = (4 * m - 3) / (3 / 3), from by sorry
  have : m = 9 / 5, from by sorry
  assumption

end time_to_empty_tank_l23_23963


namespace slope_line_angle_l23_23881

noncomputable def slope_of_line (a b c : ℝ) : ℝ := -a / b

noncomputable def angle_of_slope (m : ℝ) : ℝ := arctan m

theorem slope_line_angle :
  let a := 1
  let b := √3
  let c := 2
  let slope := slope_of_line a b c
  let θ := angle_of_slope slope
  θ = 5 * π / 6 :=
by
  let a := 1
  let b := √3
  let c := 2
  let slope := slope_of_line a b c
  let θ := angle_of_slope slope
  have h : slope = - (1 / √3), by sorry
  have hθ : θ = real.arctan (- (1 / √3)), by sorry
  have hθ_correct : θ = 5 * π / 6, by sorry
  exact hθ_correct

end slope_line_angle_l23_23881


namespace parabola_intersection_difference_l23_23498

theorem parabola_intersection_difference :
  let a := 0
  let c := 2 / 3
  c - a = 2 / 3 :=
by
-- Define the parabolas
let y1 (x : ℝ) := 2 * x^2 - 4 * x + 4
let y2 (x : ℝ) := -x^2 - 2 * x + 4
-- Find solutions for intersection points
have h₁ : 2 * a^2 - 4 * a + 4 = -a^2 - 2 * a + 4 := by sorry
have h₂ : 2 * c^2 - 4 * c + 4 = -c^2 - 2 * c + 4 := by sorry
-- Assume c ≥ a
have h₃ : c ≥ a := by sorry
-- Check the difference
show c - a = 2 / 3 by sorry

end parabola_intersection_difference_l23_23498


namespace factorial_difference_l23_23142

open Nat 

theorem factorial_difference : (11! - 10!) / 9! = 100 := by
  have h1 : 11! = 11 * 10! := by 
    sorry

  have h2 : (11 * 10! - 10!) / 9! = (10! * (11 - 1)) / 9! := by 
    sorry

  have h3 : (10! * 10) / 9! = (10 * 9! * 10) / 9! := by
    sorry

  have h4 : (10 * 9! * 10) / 9! = 10 * 10 := by
    sorry

  show 100 = 100 from by
    sorry

end factorial_difference_l23_23142


namespace slip_2_5_in_A_or_C_l23_23365

-- Define the slips and their values
def slips : List ℚ := [1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 4.5, 5, 5.5, 6]

-- Define the cups
inductive Cup
| A | B | C | D | E | F

open Cup

-- Define the given cups constraints
def sum_constraints : Cup → ℚ
| A => 6
| B => 7
| C => 8
| D => 9
| E => 10
| F => 10

-- Initial conditions for slips placement
def slips_in_cups (c : Cup) : List ℚ :=
match c with
| F => [1.5]
| B => [4]
| _ => []

-- We'd like to prove that:
def slip_2_5_can_go_into : Prop :=
  (slips_in_cups A = [2.5] ∧ slips_in_cups C = [2.5])

theorem slip_2_5_in_A_or_C : slip_2_5_can_go_into :=
sorry

end slip_2_5_in_A_or_C_l23_23365


namespace part_I_part_II_l23_23350

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) - 2 * (cos x)^2

-- (I)
theorem part_I : f (π / 3) = sqrt 3 / 2 - 1 / 2 :=
by
  sorry

-- (II)
theorem part_II : ∃ x, x ∈ Icc 0 (π / 2) ∧ f x = sqrt 2 - 1 ∧ ∀ y ∈ Icc 0 (π / 2), f y ≤ f x :=
by
  sorry

end part_I_part_II_l23_23350


namespace quadratic_coefficients_l23_23081

theorem quadratic_coefficients :
  ∀ (x : ℝ), x^2 - x + 3 = 0 → (1, -1, 3) :=
by
  intro x
  intro h
  have quadratic_coeff : x^2 - x + 3 = 1 * x^2 + (-1) * x + 3 := by simp
  exact (1, -1, 3) 
  sorry

end quadratic_coefficients_l23_23081


namespace find_a6_l23_23314

variable (a_n : ℕ → ℤ) (d : ℤ)

-- Conditions
axiom sum_first_three_terms (S3 : a_n 1 + a_n 2 + a_n 3 = 168)
axiom diff_terms (diff_a2_a5 : a_n 2 - a_n 5 = 42)

-- Definition of arithmetic progression 
def arith_prog (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d

-- Proving that a6 = 3
theorem find_a6 (a1 : ℤ) (proof_S3 : a1 + (a1 + d) + (a1 + 2*d) = 168)
  (proof_diff : (a1 + d) - (a1 + 4*d) = 42) : a1 + 5*d = 3 :=
by
  sorry

end find_a6_l23_23314


namespace range_of_m_l23_23324

noncomputable def f (x : ℝ) : ℝ := x - 2
noncomputable def g (x m : ℝ) : ℝ := x^2 - 2 * m * x + 4

def condition (m : ℝ) : Prop :=
  ∀ x1 ∈ set.Icc 1 2, ∃ x2 ∈ set.Icc 4 5, g x1 m = f x2

theorem range_of_m : {m : ℝ | condition m} = set.Icc (5/4) (Real.sqrt 2) := sorry

end range_of_m_l23_23324


namespace overall_percentage_good_fruits_l23_23206

theorem overall_percentage_good_fruits
  (oranges_bought : ℕ)
  (bananas_bought : ℕ)
  (apples_bought : ℕ)
  (pears_bought : ℕ)
  (oranges_rotten_percent : ℝ)
  (bananas_rotten_percent : ℝ)
  (apples_rotten_percent : ℝ)
  (pears_rotten_percent : ℝ)
  (h_oranges : oranges_bought = 600)
  (h_bananas : bananas_bought = 400)
  (h_apples : apples_bought = 800)
  (h_pears : pears_bought = 200)
  (h_oranges_rotten : oranges_rotten_percent = 0.15)
  (h_bananas_rotten : bananas_rotten_percent = 0.03)
  (h_apples_rotten : apples_rotten_percent = 0.12)
  (h_pears_rotten : pears_rotten_percent = 0.25) :
  let total_fruits := oranges_bought + bananas_bought + apples_bought + pears_bought
  let rotten_oranges := oranges_rotten_percent * oranges_bought
  let rotten_bananas := bananas_rotten_percent * bananas_bought
  let rotten_apples := apples_rotten_percent * apples_bought
  let rotten_pears := pears_rotten_percent * pears_bought
  let good_oranges := oranges_bought - rotten_oranges
  let good_bananas := bananas_bought - rotten_bananas
  let good_apples := apples_bought - rotten_apples
  let good_pears := pears_bought - rotten_pears
  let total_good_fruits := good_oranges + good_bananas + good_apples + good_pears
  (total_good_fruits / total_fruits) * 100 = 87.6 :=
by
  sorry

end overall_percentage_good_fruits_l23_23206


namespace size_of_slope_at_point_l23_23698

def f (x : ℝ) : ℝ := (Real.exp (2 * x - 2)) / x

theorem size_of_slope_at_point (θ : ℝ) (hθ : θ = Real.pi / 4) :
  ∀ (f' : ℝ → ℝ), 
  f' = λ x, (2 * x * (Real.exp (2 * x - 2)) - (Real.exp (2 * x - 2))) / (x ^ 2) →
  f' 1 = 1 →
  θ = Real.pi / 4 := 
by
  sorry

end size_of_slope_at_point_l23_23698


namespace Susie_earnings_l23_23470

theorem Susie_earnings :
  let price_per_slice := 3 in
  let slices_sold := 24 in
  let price_per_pizza := 15 in
  let pizzas_sold := 3 in
  let earnings_from_slices := price_per_slice * slices_sold in
  let earnings_from_pizzas := price_per_pizza * pizzas_sold in
  let total_earnings := earnings_from_slices + earnings_from_pizzas in
  total_earnings = 117 :=
by
  sorry

end Susie_earnings_l23_23470


namespace problem1_problem2_l23_23683

noncomputable def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in range (n + 1), a i

theorem problem1 (a : ℕ → ℕ) (h : ∀ n, (n + 2) * (n * (a 1 - 2)) = Sn a n) :
  a = (λ n, 2 * n + 1) :=
sorry

theorem problem2 (a : ℕ → ℕ) (T : ℕ → ℝ)
  (h1 : a = (λ n, 2 * n + 1))
  (hT : ∀ n, T n = ∑ i in range (n + 1), 1 / (a i * a (i + 1))) :
  ∀ n, T n = n / (6 * n + 9) :=
sorry

end problem1_problem2_l23_23683


namespace find_second_number_l23_23507

-- The Lean statement for the given math problem:

theorem find_second_number
  (x y z : ℝ)  -- Represent the three numbers
  (h1 : x = 2 * y)  -- The first number is twice the second
  (h2 : z = (1/3) * x)  -- The third number is one-third of the first
  (h3 : x + y + z = 110)  -- The sum of the three numbers is 110
  : y = 30 :=  -- The second number is 30
sorry

end find_second_number_l23_23507


namespace translate_even_function_l23_23894

theorem translate_even_function (ϕ : ℝ) (h1 : -real.pi / 2 < ϕ) (h2 : ϕ < real.pi / 2) :
  let g := λ x, real.sin (2 * x + ϕ)
  let f := λ x, real.sin (2 * (x + real.pi / 8) + ϕ)
  (∀ x, f x = f (-x)) → ϕ = real.pi / 4 :=
by
  intros
  sorry

end translate_even_function_l23_23894


namespace starting_lineups_l23_23073

theorem starting_lineups (n : ℕ) (table : Finset ℕ) (h_n_eq : n = 15)
  (h_table_size : table.card = 12) :
  (Finset.card (Finset.filter (λ x, ¬(0 ∈ x ∧ 1 ∈ x ∧ 2 ∈ x)) 
  (Finset.powerset_len 6 (Finset.range n)))) = 3300 := 
sorry

end starting_lineups_l23_23073


namespace orchestra_musicians_l23_23479

theorem orchestra_musicians : ∃ (m n : ℕ), (m = n^2 + 11) ∧ (m = n * (n + 5)) ∧ m = 36 :=
by {
  sorry
}

end orchestra_musicians_l23_23479


namespace units_digit_sum_of_factorials_is_3_l23_23268

theorem units_digit_sum_of_factorials_is_3 :
  (∑ k in Finset.range 2024, Nat.factorial k) % 10 = 3 :=
by
  sorry

end units_digit_sum_of_factorials_is_3_l23_23268


namespace smoke_diagram_total_height_l23_23025

theorem smoke_diagram_total_height : 
  ∀ (h1 h2 h3 h4 h5 : ℕ),
    h1 < h2 ∧ h2 < h3 ∧ h3 < h4 ∧ h4 < h5 ∧ 
    (h2 - h1 = 2) ∧ (h3 - h2 = 2) ∧ (h4 - h3 = 2) ∧ (h5 - h4 = 2) ∧ 
    (h5 = h1 + h2) → 
    h1 + h2 + h3 + h4 + h5 = 50 := 
by 
  sorry

end smoke_diagram_total_height_l23_23025


namespace log_a_b_eq_pi_l23_23952

theorem log_a_b_eq_pi (a b : ℝ) (h₁ : r = log 10 (a^3)) (h₂ : C = log 10 (b^6)) :
  log a b = π :=
by 
sorry

end log_a_b_eq_pi_l23_23952


namespace height_divides_perimeter_isosceles_l23_23455

-- Definitions of the triangle and its properties
variables {a b c : ℝ} (h_a : ℝ) (A B C : Type) [tri : triangle A B C a b c]

noncomputable def is_isosceles (A B C : Type) [triangle A B C a b c] :=
  b = c

-- The main theorem stating the problem
theorem height_divides_perimeter_isosceles (h_divides_perimeter : h_a = (a + b + c) / 2) :
  is_isosceles A B C := 
sorry

end height_divides_perimeter_isosceles_l23_23455


namespace exists_v_min_norm_l23_23417

def smallest_value_norm (v : ℝ × ℝ) : Prop :=
  ⟪∥v + ⟨4, 2⟩∥ = 10 ∧ ∥v∥ = 10 - 2 * Real.sqrt 5⟫

theorem exists_v_min_norm : ∃ v : ℝ × ℝ, smallest_value_norm v :=
  sorry

end exists_v_min_norm_l23_23417


namespace tangent_length_l23_23199

-- Define the conditions as assumptions
variable (r : ℝ) (d : ℝ)

-- Define the statement of the problem
theorem tangent_length (h1 : r = 10) (h2 : d = 26) : sqrt (d^2 - r^2) = 24 :=
by
  -- These stubs are placeholders for where the mathematical checks would occur
  sorry

end tangent_length_l23_23199


namespace bonus_trigger_sales_amount_l23_23962

theorem bonus_trigger_sales_amount (total_sales S : ℝ) (h1 : 0.09 * total_sales = 1260)
  (h2 : 0.03 * (total_sales - S) = 120) : S = 10000 :=
sorry

end bonus_trigger_sales_amount_l23_23962


namespace bryce_raisins_l23_23364

theorem bryce_raisins (x : ℕ) (h1 : x = 2 * (x - 8)) : x = 16 :=
by
  sorry

end bryce_raisins_l23_23364


namespace equilateral_triangle_ratio_l23_23969

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 9) = 1
def B : (ℝ × ℝ) := (0, 3)
def A := (-2 * Real.sqrt 3 / 2, 0)
def C := (2 * Real.sqrt 3 / 2, 0)
def is_equilateral (AB BC CA : ℝ) : Prop := AB = BC ∧ BC = CA
def F_1_F_2 : ℝ := 2 * Real.sqrt 7
def distance (p1 p2 : (ℝ × ℝ)) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem equilateral_triangle_ratio :
  is_equilateral (distance B A) (distance B C) (distance A C) →
  ellipse_eq A.1 A.2 →
  ellipse_eq C.1 C.2 →
  distance B A / F_1_F_2 = Real.sqrt 21 / 7 :=
by
  sorry

end equilateral_triangle_ratio_l23_23969


namespace waiter_customer_count_l23_23208

def initial_customers := 33
def customers_left := 31
def new_customers := 26

theorem waiter_customer_count :
  (initial_customers - customers_left) + new_customers = 28 :=
by
  -- This is a placeholder for the proof that can be filled later.
  sorry

end waiter_customer_count_l23_23208


namespace arithmetic_geometric_sequence_S30_l23_23673

variable (S : ℕ → ℝ)

theorem arithmetic_geometric_sequence_S30 :
  S 10 = 10 →
  S 20 = 30 →
  S 30 = 70 := by
  intros h1 h2
  -- proof steps go here
  sorry

end arithmetic_geometric_sequence_S30_l23_23673


namespace quadratic_eq_coefficients_l23_23078

theorem quadratic_eq_coefficients :
  ∃ (a b c : ℤ), (a = 1 ∧ b = -1 ∧ c = 3) ∧ (∀ x : ℤ, a * x^2 + b * x + c = x^2 - x + 3) :=
by
  use 1, -1, 3
  split
  { split; refl }
  { intro x
    simp }
  sorry

end quadratic_eq_coefficients_l23_23078


namespace possible_values_of_derivative_l23_23808

noncomputable def differentiable_function_condition (f : ℝ → ℝ) := 
  (0 < ∀ (x : ℝ), x < 1 → differentiable_at ℝ f x) ∧ 
  (∀ (n : ℕ), ∀ (a : ℕ), odd a ∧ 0 < a ∧ a < 2^n →
    ∃ (b : ℕ), odd b ∧ b < 2^n ∧ f (a / 2^n : ℝ) = b / 2^n)

theorem possible_values_of_derivative (f : ℝ → ℝ) (hf : differentiable_function_condition f) : 
  f' (1 / 2 : ℝ) ∈ {-1, 1} :=
sorry

end possible_values_of_derivative_l23_23808


namespace Joshua_share_correct_l23_23401

noncomputable def Joshua_share (J : ℝ) : ℝ :=
  3 * J

noncomputable def Jasmine_share (J : ℝ) : ℝ :=
  J / 2

theorem Joshua_share_correct (J : ℝ) (h : J + 3 * J + J / 2 = 120) :
  Joshua_share J = 80.01 := by
  sorry

end Joshua_share_correct_l23_23401


namespace sum_of_consecutive_integers_l23_23499

theorem sum_of_consecutive_integers (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
by
  sorry

end sum_of_consecutive_integers_l23_23499


namespace polynomial_double_root_at_center_l23_23046

noncomputable def complex_root_of_unity : ℂ := complex.exp(2 * real.pi * complex.I / 3)

theorem polynomial_double_root_at_center 
  (a b c : ℂ)
  (w : ℂ)
  (h1 : ∃ z : ℂ, z ≠ 0 ∧ ∀ (r : list ℂ), r = [w +  z, w + complex_root_of_unity * z, w + complex_root_of_unity^2 * z] ∧ 
        (∀ x ∈ r, is_root (λ x : ℂ, x^3 + a * x^2 + b * x + c) x))
  (h2 : complex_root_of_unity^3 = 1)
  (h3 : 1 + complex_root_of_unity + complex_root_of_unity^2 = 0) 
  : is_root (λ x : ℂ, 3 * x^2 + 2 * a * x + b) w ∧ derivative (λ x : ℂ, 3 * x^2 + 2 * a * x + b) w = 0 := 
begin
  sorry
end

end polynomial_double_root_at_center_l23_23046


namespace unit_digit_calc_l23_23645

theorem unit_digit_calc : (8 * 19 * 1981 - 8^3) % 10 = 0 := by
  sorry

end unit_digit_calc_l23_23645


namespace gcd_count_l23_23538

-- Problem conditions
def product_eq_180 (a b : ℕ) : Prop :=
  a * b = 180

-- Definition to compute GCD
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Main statement
theorem gcd_count (a b : ℕ) (h : product_eq_180 a b) : 
  ({d | ∃ a b, product_eq_180 a b ∧ gcd a b = d}.to_finset.card = 4) :=
sorry

end gcd_count_l23_23538


namespace sufficient_but_not_necessary_not_necessary_l23_23170

theorem sufficient_but_not_necessary (a b : ℝ) (h : a > b ∧ b > 0) : (a^2 > b^2) := by
  sorry

theorem not_necessary (a b : ℝ) : ¬ (a^2 > b^2 → a > b ∧ b > 0) := by
  have counterexample : (-(3:ℝ))^2 > (-(4:ℝ))^2 := by
    simp
  simp [counterexample]
  sorry

end sufficient_but_not_necessary_not_necessary_l23_23170


namespace cubic_polynomial_p_value_l23_23185

noncomputable def p (x : ℝ) : ℝ := sorry

theorem cubic_polynomial_p_value :
  (∀ n ∈ ({1, 2, 3, 5} : Finset ℝ), p n = 1 / n ^ 2) →
  p 4 = 1 / 150 := 
by
  intros h
  sorry

end cubic_polynomial_p_value_l23_23185


namespace degree_of_polynomial_power_l23_23526

-- Define the polynomial f(x)
noncomputable def f (x : ℝ) : ℝ := 5 * x^3 + 7 * x + 2

-- Define the exponent
def exponent : ℕ := 10

-- The theorem statement
theorem degree_of_polynomial_power :
  ∀ (x : ℝ), (degree ((f x) ^ exponent) = 30) :=
by
  sorry

end degree_of_polynomial_power_l23_23526


namespace sum_of_inverses_gt_one_l23_23115

variable (a1 a2 a3 S : ℝ)

theorem sum_of_inverses_gt_one
  (h1 : a1 > 1)
  (h2 : a2 > 1)
  (h3 : a3 > 1)
  (h_sum : a1 + a2 + a3 = S)
  (ineq1 : a1^2 / (a1 - 1) > S)
  (ineq2 : a2^2 / (a2 - 1) > S)
  (ineq3 : a3^2 / (a3 - 1) > S) :
  1 / (a1 + a2) + 1 / (a2 + a3) + 1 / (a3 + a1) > 1 := by
  sorry

end sum_of_inverses_gt_one_l23_23115


namespace probability_divisible_by_3_l23_23555

def is_prime_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_valid_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  let d1 := n / 10
  let d2 := n % 10
  in is_prime_digit d1 ∧ is_prime_digit d2

def sum_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  in d1 + d2

def is_divisible_by_3 (n : ℕ) : Prop :=
  sum_of_digits n % 3 = 0

theorem probability_divisible_by_3 :
  (finset.filter is_divisible_by_3 (finset.filter is_valid_two_digit_prime (finset.range 100))).card.to_rat /
  (finset.filter is_valid_two_digit_prime (finset.range 100)).card.to_rat = (1 / 3 : ℚ) :=
by 
  sorry

end probability_divisible_by_3_l23_23555


namespace bacteria_after_10_hours_l23_23182

def bacteria_count (hours : ℕ) : ℕ :=
  2^hours

theorem bacteria_after_10_hours : bacteria_count 10 = 1024 := by
  sorry

end bacteria_after_10_hours_l23_23182


namespace counting_integers_between_multiples_l23_23993

theorem counting_integers_between_multiples :
  let smallest_perfect_square_multiple := 900 in
  let smallest_perfect_cube_multiple := 27000 in
  let num_integers := (smallest_perfect_cube_multiple / 30) - (smallest_perfect_square_multiple / 30) + 1 in
  smallest_perfect_square_multiple = 30 * 30 ∧ 
  smallest_perfect_cube_multiple = 900 * 30 ∧ 
  num_integers = 871 :=
by
  sorry

end counting_integers_between_multiples_l23_23993


namespace value_of_e_l23_23372

variable (e : ℝ)
noncomputable def eq1 : Prop :=
  ((10 * 0.3 + 2) / 4 - (3 * 0.3 - e) / 18 = (2 * 0.3 + 4) / 3)

theorem value_of_e : eq1 e → e = 6 := by
  intro h
  sorry

end value_of_e_l23_23372


namespace number_of_moles_of_OC_NH2_2_formed_l23_23261

-- Definition: Chemical reaction condition
def reaction_eqn (x y : ℕ) : Prop := 
  x ≥ 1 ∧ y ≥ 2 ∧ x * 2 = y

-- Theorem: Prove that combining 3 moles of CO2 and 6 moles of NH3 results in 3 moles of OC(NH2)2
theorem number_of_moles_of_OC_NH2_2_formed (x y : ℕ) 
(h₁ : reaction_eqn x y)
(h₂ : x = 3)
(h₃ : y = 6) : 
x =  y / 2 :=
by {
    -- Proof is not provided
    sorry 
}

end number_of_moles_of_OC_NH2_2_formed_l23_23261


namespace range_of_quadratic_l23_23102

theorem range_of_quadratic : 
  (set.range (λ x : ℝ, x^2 - 4*x + 6) ∩ set.Ico 1 5) = set.Ico 2 11 := 
sorry

end range_of_quadratic_l23_23102


namespace general_formula_geometric_sequence_exists_l23_23318

variable {n k : ℕ}
variable {a : ℝ} (ha : a ≠ 0)

def a_n (n : ℕ) : ℝ := (2 * n - 1) * a
def S_n (n : ℕ) : ℝ := n^2 * a

theorem general_formula (n : ℕ) : 
  (∃ a_n : ℕ → ℝ, ∀ n, a_n n = (2 * n - 1) * a) ∧ 
  (∃ S_n : ℕ → ℝ, ∀ n, S_n n = n^2 * a) :=
by
  sorry

theorem geometric_sequence_exists : 
  ∃ (n k : ℕ), (S_n n = n^2 * a) ∧ (S_n (n + 1) = (n + 1)^2 * a) ∧ (S_n (n + k) = (n + k)^2 * a) ∧ ((n = 1) ∧ (k = 3)) :=
by
  sorry

end general_formula_geometric_sequence_exists_l23_23318


namespace order_of_numbers_l23_23660

open Real

theorem order_of_numbers (a b c : ℝ) 
  (h₁ : a = (3/5)^(-1/3))
  (h₂ : b = (4/3)^(-1/2))
  (h₃ : c = log (3/5)) : 
  a > b ∧ b > c := by
  sorry

end order_of_numbers_l23_23660


namespace line_through_A_parallel_l23_23089

noncomputable def lineEquation (x y : ℝ) : ℝ := x + 2*y - 8

theorem line_through_A_parallel (x y : ℝ) (hA : (2, 3) = (x, y))
    (hP : ∀ t : ℝ, (2, 3) satisfies 2 * x + 4 * y + t = 0) : 
  lineEquation (2 : ℝ) (3 : ℝ) = (0 : ℝ) :=
by
  sorry

end line_through_A_parallel_l23_23089


namespace fixed_ray_exists_l23_23896

variables {α : Type} [inner_product_space ℝ α]
variables (circle1 circle2 : set α) (r1 r2 : ℝ) (vertex : α)
  (gray_side black_side : α → Prop)

-- Conditions
def non_overlapping_circles := disjoint circle1 circle2
def touch_circle1 := ∀ x ∈ circle1, gray_side x
def touch_circle2 := ∀ y ∈ circle2, black_side y
def contact_not_at_vertex := ¬(vertex ∈ circle1 ∨ vertex ∈ circle2)

-- Proposition
def exists_fixed_ray :=
  ∃ ray : α → Prop, 
    (∀ (P : α), (gray_side P → ray P) ∧ (black_side P → ray P))
    ∧ (∀ angle_position : α, ray angle_position)

-- Theorem
theorem fixed_ray_exists 
  (h1 : non_overlapping_circles)
  (h2 : touch_circle1)
  (h3 : touch_circle2)
  (h4 : contact_not_at_vertex) :
  exists_fixed_ray :=
sorry

end fixed_ray_exists_l23_23896


namespace simon_age_gap_from_half_alvin_age_l23_23210

theorem simon_age_gap_from_half_alvin_age :
  ∀ (alvin_age simon_age : ℕ), 
  alvin_age = 30 → 
  simon_age = 10 → 
  (15 - simon_age) = 5 :=
by
  intros alvin_age simon_age h_alvin h_simon
  rw [h_alvin, h_simon]
  -- Alvin's age is 30, half of Alvin's age is 15
  have half_alvin := 30 / 2
  -- Simon's age is 10
  have simon_goal := half_alvin - 10
  -- Subtract Simon's age from half of Alvin's age should be 5
  show simon_goal = 5
  sorry

end simon_age_gap_from_half_alvin_age_l23_23210


namespace cameron_list_count_l23_23988

theorem cameron_list_count : 
  (∃ (n m : ℕ), n = 900 ∧ m = 27000 ∧ (∀ k : ℕ, (30 * k) ≥ n ∧ (30 * k) ≤ m → ∃ count : ℕ, count = 871)) :=
by
  sorry

end cameron_list_count_l23_23988


namespace probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l23_23763

theorem probability_exactly_k_gnomes_fall (n k : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in p * q^(n - k) = p * (1 - p)^(n - k) := 
sorry

theorem expected_number_of_gnomes_fall (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in 
  (∑ j in finset.range n, (1 - q^(j+1))) = n + 1 - (1 / p) + ((1 - p)^(n+1) / p) :=
sorry

end probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l23_23763


namespace percentage_increase_l23_23871

theorem percentage_increase (original_price new_price : ℝ) (h₁ : original_price = 300) (h₂ : new_price = 480) :
  ((new_price - original_price) / original_price) * 100 = 60 :=
by
  -- Proof goes here
  sorry

end percentage_increase_l23_23871


namespace factorial_div_l23_23140

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem factorial_div : (factorial 11 - factorial 10) / factorial 9 = 100 := 
by
  sorry

end factorial_div_l23_23140


namespace total_tank_capacity_l23_23584

-- Definitions based on conditions
def initial_condition (w c : ℝ) : Prop := w / c = 1 / 3
def after_adding_five (w c : ℝ) : Prop := (w + 5) / c = 1 / 2

-- The problem statement
theorem total_tank_capacity (w c : ℝ) (h1 : initial_condition w c) (h2 : after_adding_five w c) : c = 30 :=
sorry

end total_tank_capacity_l23_23584


namespace greatest_possible_five_digit_multiple_of_six_l23_23518

theorem greatest_possible_five_digit_multiple_of_six :
  ∃ (n : ℕ), (digits : List ℕ) (h1 : digits ~ [2, 5, 6, 7, 9])
              (h2 : ∃ a ∈ digits, even a)
              (h3 : list.sum digits % 3 = 0)
              (h4 : list.to_number digits = n),
              n = 97632 := 
by
  sorry

end greatest_possible_five_digit_multiple_of_six_l23_23518


namespace a_is_perfect_square_l23_23036

theorem a_is_perfect_square (a b : ℕ) (h : ab ∣ (a^2 + b^2 + a)) : (∃ k : ℕ, a = k^2) :=
sorry

end a_is_perfect_square_l23_23036


namespace equilateral_triangle_area_l23_23797

theorem equilateral_triangle_area (A B C P : ℝ × ℝ)
  (hABC : ∃ a b c : ℝ, a = b ∧ b = c ∧ a = dist A B ∧ b = dist B C ∧ c = dist C A)
  (hPA : dist P A = 10)
  (hPB : dist P B = 8)
  (hPC : dist P C = 12) :
  ∃ (area : ℝ), area = 104 :=
by
  sorry

end equilateral_triangle_area_l23_23797


namespace solve_inequality_l23_23061

theorem solve_inequality (a : ℝ) : 
    (∀ x : ℝ, x^2 + (a + 2)*x + 2*a < 0 ↔ 
        (if a < 2 then -2 < x ∧ x < -a
         else if a = 2 then false
         else -a < x ∧ x < -2)) :=
by
  sorry

end solve_inequality_l23_23061


namespace range_of_a_l23_23713

theorem range_of_a (a : ℝ) : (-∞, 0] ∩ {1, 3, a} ≠ ∅ → a ∈ (-∞, 0] :=
by
  intro h
  -- Proof steps go here
  sorry

end range_of_a_l23_23713


namespace find_amount_l23_23174

theorem find_amount (x : ℝ) (A : ℝ) (h1 : 0.65 * x = 0.20 * A) (h2 : x = 230) : A = 747.5 := by
  sorry

end find_amount_l23_23174


namespace value_of_expression_l23_23426

theorem value_of_expression (r s : ℝ) (h₁ : 3 * r^2 - 5 * r - 7 = 0) (h₂ : 3 * s^2 - 5 * s - 7 = 0) : 
  (9 * r^2 - 9 * s^2) / (r - s) = 15 :=
sorry

end value_of_expression_l23_23426


namespace geometric_sum_s6_l23_23336

noncomputable def geometric_sequence (a n : ℕ) (q : ℚ) := a * q^n

theorem geometric_sum_s6 (q a1 : ℚ) (h_q_pos : 0 < q) (h_q_lt_one : q < 1) 
  (h_a1_eq : a1 = 4 * (a1 * q) * (a1 * q^3)) 
  (h_arithmetic_mean : (a1 * q^5 + (3/4) * (a1 * q^3)) / 2 = a1 * q^4) : 
  let a_1 := a1 in
  let a_2 := a1 * q in
  let a_3 := a1 * q^2 in
  let a_4 := a1 * q^3 in
  let a_5 := a1 * q^4 in
  let a_6 := a1 * q^5 in
  let S_6 := a_1 * (1 - q^6) / (1 - q) in
  S_6 = 63/4 :=
begin
  sorry
end

end geometric_sum_s6_l23_23336


namespace probability_white_given_popped_is_7_over_12_l23_23571

noncomputable def probability_white_given_popped : ℚ :=
  let P_W := 0.4
  let P_Y := 0.4
  let P_R := 0.2
  let P_popped_given_W := 0.7
  let P_popped_given_Y := 0.5
  let P_popped_given_R := 0
  let P_popped := P_popped_given_W * P_W + P_popped_given_Y * P_Y + P_popped_given_R * P_R
  (P_popped_given_W * P_W) / P_popped

theorem probability_white_given_popped_is_7_over_12 : probability_white_given_popped = 7 / 12 := 
  by
    sorry

end probability_white_given_popped_is_7_over_12_l23_23571


namespace exponents_sum_21_36_l23_23162

noncomputable def consecutiveMultiplesOfThree (x y : ℕ) : List ℕ :=
  List.filter (λ n, n % 3 = 0) (List.range' x (y + 1))

noncomputable def productOfList (lst : List ℕ) : ℕ :=
  lst.foldl (· * ·) 1

noncomputable def primeFactorsExponentsSum (n : ℕ) : ℕ :=
  (n.factorization.sum (λ _ exponent, exponent))

theorem exponents_sum_21_36 :
  primeFactorsExponentsSum (productOfList (consecutiveMultiplesOfThree 21 36)) = 18 :=
by
  sorry

end exponents_sum_21_36_l23_23162


namespace sum_of_distinct_prime_divisors_2520_l23_23912

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def distinct_prime_divisors_sum (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime p ∧ p ∣ n) (finset.range (n + 1))).sum id

theorem sum_of_distinct_prime_divisors_2520 :
  distinct_prime_divisors_sum 2520 = 17 :=
by
sorry

end sum_of_distinct_prime_divisors_2520_l23_23912


namespace abs_of_negative_l23_23134

theorem abs_of_negative (a : ℝ) (h : a < 0) : |a| = -a :=
sorry

end abs_of_negative_l23_23134


namespace ellie_total_distance_after_six_steps_l23_23246

-- Define the initial conditions and parameters
def initial_position : ℚ := 0
def target_distance : ℚ := 5
def step_fraction : ℚ := 1 / 4
def steps : ℕ := 6

-- Define the function that calculates the sum of the distances walked
def distance_walked (n : ℕ) : ℚ :=
  let first_term := target_distance * step_fraction
  let common_ratio := 3 / 4
  first_term * (1 - common_ratio^n) / (1 - common_ratio)

-- Define the theorem we want to prove
theorem ellie_total_distance_after_six_steps :
  distance_walked steps = 16835 / 4096 :=
by 
  sorry

end ellie_total_distance_after_six_steps_l23_23246


namespace Cameron_list_count_l23_23979

theorem Cameron_list_count : 
  let lower_bound := 900
  let upper_bound := 27000
  let step := 30
  let n_min := lower_bound / step
  let n_max := upper_bound / step
  n_max - n_min + 1 = 871 :=
by
  sorry

end Cameron_list_count_l23_23979


namespace Cameron_list_count_l23_23983

theorem Cameron_list_count : 
  let lower_bound := 900
  let upper_bound := 27000
  let step := 30
  let n_min := lower_bound / step
  let n_max := upper_bound / step
  n_max - n_min + 1 = 871 :=
by
  sorry

end Cameron_list_count_l23_23983


namespace smallest_n_repeating_251_l23_23292

theorem smallest_n_repeating_251 (m n : ℕ) (hmn : m < n) (coprime : Nat.gcd m n = 1) :
  (∃ m : ℕ, ∃ n : ℕ, Nat.gcd m n = 1 ∧ m < n ∧ let r := real.to_rat_repr (↥m / ↥n) in (r.2 ≥ 1000 * m mod n = 251)) → n = 127 :=
sorry

end smallest_n_repeating_251_l23_23292


namespace days_to_complete_job_l23_23373

theorem days_to_complete_job (m₁ m₂ d₁ d₂ total_man_days : ℝ)
    (h₁ : m₁ = 30)
    (h₂ : d₁ = 8)
    (h₃ : total_man_days = 240)
    (h₄ : total_man_days = m₁ * d₁)
    (h₅ : m₂ = 40) :
    d₂ = total_man_days / m₂ := by
  sorry

end days_to_complete_job_l23_23373


namespace depth_of_water_in_cistern_l23_23579

-- Define the given constants
def length_cistern : ℝ := 6
def width_cistern : ℝ := 5
def total_wet_area : ℝ := 57.5

-- Define the area of the bottom of the cistern
def area_bottom (length : ℝ) (width : ℝ) : ℝ := length * width

-- Define the area of the longer sides of the cistern in contact with water
def area_long_sides (length : ℝ) (depth : ℝ) : ℝ := 2 * length * depth

-- Define the area of the shorter sides of the cistern in contact with water
def area_short_sides (width : ℝ) (depth : ℝ) : ℝ := 2 * width * depth

-- Define the total wet surface area based on depth of the water
def total_wet_surface_area (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ := 
    area_bottom length width + area_long_sides length depth + area_short_sides width depth

-- Define the proof statement
theorem depth_of_water_in_cistern : ∃ h : ℝ, h = 1.25 ∧ total_wet_surface_area length_cistern width_cistern h = total_wet_area := 
by
  use 1.25
  sorry

end depth_of_water_in_cistern_l23_23579


namespace arithmetic_mean_factor_l23_23846

theorem arithmetic_mean_factor 
  (n d : ℕ)
  (h1 : d = 8 * n + 1)
  (h2 : (n + (d - 1) / 2) = 5 * n) : 
  (n + (d - 1)) / (n + (d - 1) / 2) = 1.8 := 
sorry

end arithmetic_mean_factor_l23_23846


namespace weight_difference_l23_23477

noncomputable def W_A : ℝ := 78

variable (W_B W_C W_D W_E : ℝ)

axiom cond1 : (W_A + W_B + W_C) / 3 = 84
axiom cond2 : (W_A + W_B + W_C + W_D) / 4 = 80
axiom cond3 : (W_B + W_C + W_D + W_E) / 4 = 79

theorem weight_difference : W_E - W_D = 6 :=
by
  have h1 : W_A = 78 := rfl
  sorry

end weight_difference_l23_23477


namespace other_place_is_rectangle_or_square_l23_23611

theorem other_place_is_rectangle_or_square (angles_counted : Nat)
  (sum_of_angles : Nat)
  (rectangular_park_angles : Nat)
  (park_is_rectangular : rectangular_park_angles = 4)
  (total_angles_sum : angles_counted + rectangular_park_angles = sum_of_angles)
  (sum_is_8 : sum_of_angles = 8) :
  angles_counted = 4 ∧ rectangular_park_angles = 4 ∧ (∃ (other_place : Type), 
    ∀ (A B C D : other_place), 
    (angles_counted = 4 ∧ 
     ∃ p q r s : other_place, 
     ∃ (right_angle : p = q) (right_angle2 : q = r) (right_angle3 : r = s) (right_angle4 : s = p), 
      right_angle ∧ right_angle2 ∧ right_angle3 ∧ right_angle4)) := 
by
  sorry

end other_place_is_rectangle_or_square_l23_23611


namespace number_of_zeros_of_f_l23_23494

def f (x : ℝ) : ℝ := log x / log 2 - x + 2

theorem number_of_zeros_of_f : ∃! x : ℝ, f x = 0 := 
sorry

end number_of_zeros_of_f_l23_23494


namespace max_band_members_l23_23510

theorem max_band_members (n : ℤ) (h1 : 30 * n % 21 = 9) (h2 : 30 * n < 1500) : 30 * n ≤ 1470 :=
by
  -- Proof to be filled in later
  sorry

end max_band_members_l23_23510


namespace unique_x_floor_eq_20_7_l23_23239

theorem unique_x_floor_eq_20_7 : ∀ x : ℝ, (⌊x⌋ + x + 1/2 = 20.7) → x = 10.2 :=
by
  sorry

end unique_x_floor_eq_20_7_l23_23239


namespace coefficient_of_x2_in_expr_l23_23613

-- Definitions for the polynomial terms
def poly1 := 5 * (x + x^2 - x^4)
def poly2 := -4 * (x^2 - 2x^3 + 3x^6)
def poly3 := 3 * (3x^2 - x^7)
def expr := poly1 + poly2 + poly3

-- Proof statement
theorem coefficient_of_x2_in_expr :
  -- The expression we are working with
  let expr := 5 * (x + x^2 - x^4) - 4 * (x^2 - 2x^3 + 3x^6) + 3 * (3x^2 - x^7)
  -- The coefficient of x^2 in this expression is 10
  coefficient(x^2, expr) = 10 :=
sorry

end coefficient_of_x2_in_expr_l23_23613


namespace solve_exponents_l23_23055

-- Definitions formulated from the conditions
def base_exponentiation_identity (a b : ℝ) : a^b = a^b := by trivial

-- The main theorem to prove the mathematical equivalence
theorem solve_exponents (x y : ℝ) : 
  5^(x + y + 4) = 625^x ↔ y = 3x - 4 := by 
  sorry

end solve_exponents_l23_23055


namespace cameron_list_count_l23_23995

theorem cameron_list_count :
  let numbers := {n : ℕ | 30 ≤ n ∧ n ≤ 900}
  in set.card numbers = 871 :=
sorry -- proof is omitted

end cameron_list_count_l23_23995


namespace foci_distance_of_ellipse_l23_23094

/-- The problem states that we have an ellipse defined by a specific equation
    and asks us to find the distance between its foci. -/
theorem foci_distance_of_ellipse :
  let F1 := (1, 3)
  let F2 := (-7, -2)
  sqrt ((F1.1 + F2.1) ^ 2 + (F1.2 + F2.2) ^ 2) = sqrt 89 := by
  sorry

end foci_distance_of_ellipse_l23_23094


namespace find_N_l23_23605

theorem find_N 
    (N : ℕ)
    (P_same_color : (5 / 13) * (18 / (18 + N)) + (8 / 13) * (N / (18 + N)) = 0.62) :
    N = 59 :=
by
  sorry

end find_N_l23_23605


namespace geometry_theorem_l23_23608

noncomputable def problem (O1 O2 A B C D K M N E F : Type) [AddGroup O1] [AddGroup O2] [AddGroup A] [AddGroup B] 
  [AddGroup C] [AddGroup D] [AddGroup K] [AddGroup M] [AddGroup N] [AddGroup E] [AddGroup F] : Prop :=
  ∃ (intersect_circles : (O1 ∩ O2) = {A, B}),
  ∃ (CD_line : line_segment C D) (contains_A : A ∈ CD_line) (C_on_O1 : C ∈ O1) (D_on_O2 : D ∈ O2),
  ∃ (K_on_CD : K ∈ CD_line) (K_not_endpoints : K ≠ C ∧ K ≠ D),
  ∃ (KM_parallel_BD : parallel (line_segment K M) (line_segment B D)) (M_on_BC : M ∈ line B C),
  ∃ (KN_parallel_BC : parallel (line_segment K N) (line_segment B C)) (N_on_BD : N ∈ line B D),
  ∃ (ME_perp_BC : perpendicular (line_segment M E) (line B C)) (E_on_O1 : E ∈ O1),
  ∃ (NF_perp_BD : perpendicular (line_segment N F) (line B D)) (F_on_O2 : F ∈ O2),
  perpendicular (line_segment K E) (line_segment K F)

theorem geometry_theorem : problem O1 O2 A B C D K M N E F := 
  by sorry

end geometry_theorem_l23_23608


namespace largest_n_for_sin_cos_l23_23257

theorem largest_n_for_sin_cos (n : ℕ) (x : ℝ) (h_n : ∀ x : ℝ, sin x ^ n + cos x ^ n ≥ 2 / n) : n = 4 := by
  sorry   -- proof omitted

end largest_n_for_sin_cos_l23_23257


namespace larger_cross_section_distance_l23_23517

def area1 : ℝ := 256 * Real.sqrt 2
def area2 : ℝ := 576 * Real.sqrt 2
def distance_between_planes : ℝ := 10
def ratio_of_areas : ℝ := area1 / area2
def ratio_of_sides := Real.sqrt ratio_of_areas

theorem larger_cross_section_distance (h : ℝ) (a1 a2 d : ℝ) (r : ℝ) (s : ℝ) :
  a1 = 256 * Real.sqrt 2 →
  a2 = 576 * Real.sqrt 2 → 
  d = 10 →
  r = a1 / a2 →
  s = Real.sqrt r →
  h - s * h = d →
  h = 30 :=
by
  intros h a1 a2 d r s 
  intro h_def
  intro a1_def
  intro a2_def
  intro d_def
  intro r_def
  intro s_def
  intro h_equation
  sorry

end larger_cross_section_distance_l23_23517


namespace certain_number_is_approx_l23_23117

theorem certain_number_is_approx (x : ℝ) :
  (0.625 * x * 28.9) / (0.0017 * 0.025 * 8.1) = 382.5 → x = 0.2915 :=
begin
  assume h,
  sorry
end

end certain_number_is_approx_l23_23117


namespace determinant_triangle_l23_23812

theorem determinant_triangle (A B C : ℝ) (h : A + B + C = Real.pi) :
  Matrix.det ![![Real.cos A ^ 2, Real.tan A, 1],
               ![Real.cos B ^ 2, Real.tan B, 1],
               ![Real.cos C ^ 2, Real.tan C, 1]] = 0 := by
  sorry

end determinant_triangle_l23_23812


namespace solve_system_l23_23062

theorem solve_system (x y z a : ℝ) 
  (h1 : x + y + z = a) 
  (h2 : x^2 + y^2 + z^2 = a^2) 
  (h3 : x^3 + y^3 + z^3 = a^3) : 
  (x = 0 ∧ y = 0 ∧ z = a) ∨ 
  (x = 0 ∧ y = a ∧ z = 0) ∨ 
  (x = a ∧ y = 0 ∧ z = 0) := 
sorry

end solve_system_l23_23062


namespace largest_subset_no_square_sum_l23_23721

theorem largest_subset_no_square_sum :
  ∃ (S : Set ℕ), S ⊆ {2, 3, 4, 5, 6, 7, 8, 9, 10, 11} ∧
                 ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → ¬ is_square (a + b) ∧
                 S.card = 7 :=
sorry

end largest_subset_no_square_sum_l23_23721


namespace circumcenter_lies_on_circumcircle_ABO_l23_23795

open EuclideanGeometry

variables {A B C O P X G : Point} {Ω : Circle}

-- Given conditions
axiom center_of_circumcircle : is_circumcenter O (triangle A B C)
axiom P_on_arc_AC_not_containing_B : on_arc_not_containing P A C B Ω
axiom X_on_line_BC : online X (line B C)
axiom PX_perpendicular_AC : perpendicular (line P X) (line A C)

-- Definition for circumcenter of triangle BXP
def circumcenter_BXP := is_circumcenter G (triangle B X P)

-- Circumcircle of triangle ABO
def circumcircle_ABO := circumcircle O (triangle A B O)

-- Proof statement: circumcenter of BXP lies on the circumcircle of ABO
theorem circumcenter_lies_on_circumcircle_ABO :
  circumcenter_BXP → on_circumcircle G circumcircle_ABO :=
sorry

end circumcenter_lies_on_circumcircle_ABO_l23_23795


namespace rectangle_area_l23_23869

theorem rectangle_area (L W : ℕ) (h1 : 2 * L + 2 * W = 280) (h2 : L = 5 * (W / 2)) : L * W = 4000 :=
sorry

end rectangle_area_l23_23869


namespace certain_amount_eq_3_l23_23916

theorem certain_amount_eq_3 (x A : ℕ) (hA : A = 5) (h : A + (11 + x) = 19) : x = 3 :=
by
  sorry

end certain_amount_eq_3_l23_23916


namespace speed_of_mans_train_is_correct_l23_23589

def speed_of_goods_train_kmph : ℝ := 20
def length_of_goods_train_m : ℝ := 420
def passing_time_sec : ℝ := 18

def speed_of_goods_train_mps : ℝ := speed_of_goods_train_kmph / 3.6

def relative_speed_mps : ℝ := length_of_goods_train_m / passing_time_sec

def speed_of_mans_train_mps : ℝ := relative_speed_mps - speed_of_goods_train_mps

def speed_of_mans_train_kmph : ℝ := speed_of_mans_train_mps * 3.6

theorem speed_of_mans_train_is_correct :
  speed_of_mans_train_kmph = 63.972 := by
  sorry

end speed_of_mans_train_is_correct_l23_23589


namespace number_of_tables_is_correct_l23_23173

variable (students_per_table total_students tables : Nat)

-- Conditions
def students_per_table := 6
def total_students := 204

-- Theorem
theorem number_of_tables_is_correct (h : total_students = students_per_table * tables) :
  tables = 34 :=
by
  sorry

end number_of_tables_is_correct_l23_23173


namespace option_A_not_correct_option_B_correct_option_C_correct_option_D_correct_l23_23156

theorem option_A_not_correct 
  (x : ℝ) : ¬ (∀ y, y = (x^2 + 1)/x → y ≥ 2) := 
sorry

theorem option_B_correct 
  (x y : ℝ) (h : x > 1) (hy : y = 2x + (4 / (x - 1)) - 1) : 
  y ≥ 4 * Real.sqrt 2 + 1 :=
sorry

theorem option_C_correct 
  {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 3 * x * y) : 
  2 * x + y ≥ 3 := 
sorry

theorem option_D_correct 
  {x y : ℝ} (h : 9 * x^2 + y^2 + x * y = 1) : 
  3 * x + y ≤ (2 * Real.sqrt 21) / 7 := 
sorry

end option_A_not_correct_option_B_correct_option_C_correct_option_D_correct_l23_23156


namespace find_m_of_quadratic_eq_roots_l23_23383

theorem find_m_of_quadratic_eq_roots :
  ∃ m : ℝ, (∀ x : ℝ, (9 * x^2 + 5 * x + m = 0)
  → (x = (-5 + real.sqrt (-371)) / 18 ∨ x = (-5 - real.sqrt (-371)) / 18)) 
  → m = 11 :=
begin
  sorry
end

end find_m_of_quadratic_eq_roots_l23_23383


namespace coffee_beans_weight_l23_23617

variable (C P : ℝ)

def coffee_cost_conditions : Prop :=
  C + P = 40 ∧ 
  5.50 * C + 4.25 * P = 40 * 4.60

theorem coffee_beans_weight (h : coffee_cost_conditions C P) : C = 11.2 :=
by
  obtain ⟨h_weight, h_cost⟩ := h
  have h1 : P = 40 - C := by linarith
  rw [h1] at h_cost
  simp only [add_subassoc] at h_cost
  have h2 : 5.50*C + 4.25*(40 - C) = 184 := by linarith [h_cost]
  simp only [mul_sub] at h2
  have h3 : 5.50*C + 170 - 4.25*C = 184 := by linarith [h2]
  have h4 : 1.25*C = 14 := by linarith [h3]
  simp only [div_eq_mul_inv] at h4
  have h5 : C = 14 / 1.25 := by linarith [h4]
  norm_num at h5
  exact h5

#check coffee_beans_weight

end coffee_beans_weight_l23_23617


namespace find_m_equals_powers_of_2_l23_23003

-- Define the function r_k(n) as the remainder of n divided by k
def r_k (n k : ℕ) : ℕ := n - k * (n / k)

-- Define the function r(n) as the sum of r_k(n) for k going from 1 to n
def r (n : ℕ) : ℕ := (Finset.range n).sum (λ k, r_k (n) (k + 1))

-- Main theorem to prove
theorem find_m_equals_powers_of_2 (m : ℕ) (h1 : 1 < m) (h2 : m ≤ 2014) :
  r m = r (m - 1) ↔ ∃ s : ℕ, m = 2^s ∧ 1 < 2^s ∧ 2^s ≤ 2014 :=
by
  sorry

end find_m_equals_powers_of_2_l23_23003


namespace find_a_and_tangent_line_find_theta_range_l23_23346

noncomputable theory

def f (x a : ℝ) : ℝ := x^3 - 2 * x^2 + a * x

def tangent_perpendicular (x a : ℝ) : Prop :=
  3 * x^2 - 4 * x + a = -1

def discriminant (a : ℝ) : Prop :=
  (4 * 1) * (a + 1) = 16

theorem find_a_and_tangent_line :
  ∃ (a : ℝ), discriminant a ∧
  (a = 3 ∧ ∀ x : ℝ,
    tangent_perpendicular x a →
    3 * x + f x a - 8 = 0)
:= sorry

theorem find_theta_range :
  ∀ (a : ℝ), (a = 3) →
  ∃ (theta_min theta_max : ℝ),
    theta_min = - real.pi / 4 ∧ 
    theta_max = real.pi / 2 ∧ 
    ∀ (x : ℝ), 3 * x^2 - 4 * x + 3 ≥ -1 →
    (theta_min < atan (3 * x^2 - 4 * x + 3) ∧ atan (3 * x^2 - 4 * x + 3) < theta_max)
:= sorry

end find_a_and_tangent_line_find_theta_range_l23_23346


namespace graph_paper_fold_l23_23593

theorem graph_paper_fold (m n : ℚ) :
  let midpoint_a_b := ((1 + 5) / 2, (3 + 1) / 2),
      slope_a_b := (1 - 3) / (5 - 1),
      slope_perp := -1 / slope_a_b,
      line_fold := λ x, 2 * x - 4,
      midpoint_c_d := ((8 + m) / 2, (4 + n) / 2)
  in midpoint_a_b = (3, 2) ∧ 
     slope_a_b = -1/2 ∧
     slope_perp = 2 ∧
     line_fold 3 = 2 * 3 - 4 ∧ -- Line passes through the midpoint
     (2 * n - 8 = -m + 8) ∧
     (midpoint_c_d.2 = line_fold midpoint_c_d.1) →
  m = 16 / 3 ∧ n = 16 / 3 ∧ m + n = 32 / 3 :=
by
  sorry

end graph_paper_fold_l23_23593


namespace mu_squared_minus_lambda_squared_l23_23491

theorem mu_squared_minus_lambda_squared :
  let √3 := Real.sqrt 3
  let line_eq (x y : ℝ) := √3 * x - y - √3 = 0
  let parabola_eq (x y : ℝ) := y^2 = 4 * x
  let A_x := 3
  let A_y := 2 * √3
  let B_x := 1 / 3
  let B_y := -2 * √3 / 3
  let F := (1, 0)
  let OF := F
  let OA := (A_x, A_y)
  let OB := (B_x, B_y)
  let λ : ℝ := 1 / 4
  let μ : ℝ := 3 / 4
  in  OF = λ * OA + μ * OB → μ^2 - λ^2 = 1 / 2 := 
by {
  intros,
  sorry
}

end mu_squared_minus_lambda_squared_l23_23491


namespace student_enrollment_difference_l23_23519

theorem student_enrollment_difference :
  let varsity := 1150
  let northwest := 1530
  let central := 1850
  let greenbriar := 1680
  let riverside := 1320
  vars max_enrollment := max varsity (max northwest (max central (max greenbriar riverside)))
  vars min_enrollment := min varsity (min northwest (min central (min greenbriar riverside)))
  (max_enrollment - min_enrollment) = 700 := 
by
  sorry

end student_enrollment_difference_l23_23519


namespace product_of_fractions_l23_23220

theorem product_of_fractions :
  (1 / 3) * (3 / 5) * (5 / 7) = 1 / 7 :=
  sorry

end product_of_fractions_l23_23220


namespace solve_m_quadratic_l23_23651

theorem solve_m_quadratic (m : ℤ) (h : (m - 2) * x^(m^2 - 2) - 3 * x + 1 = 0) : m = -2 :=
by {
  sorry
}

end solve_m_quadratic_l23_23651


namespace average_weight_increase_l23_23119

noncomputable def average_increase (A : ℝ) : ℝ :=
  let initial_total := 10 * A
  let new_total := initial_total + 25
  let new_average := new_total / 10
  new_average - A

theorem average_weight_increase (A : ℝ) : average_increase A = 2.5 := by
  sorry

end average_weight_increase_l23_23119


namespace Tony_walking_speed_l23_23893

theorem Tony_walking_speed :
  ∃ W : ℝ, W = 3 ∧
  (let walking_distance_per_day := 3;
       running_distance_per_day := 10;
       running_speed := 5;
       exercise_hours_per_week := 21;
       days_per_week := 7 in
   let walking_distance_per_week := walking_distance_per_day * days_per_week;
       running_distance_per_week := running_distance_per_day * days_per_week;
       walking_time_per_week := walking_distance_per_week / W;
       running_time_per_week := running_distance_per_week / running_speed in
   walking_time_per_week + running_time_per_week = exercise_hours_per_week) :=
begin
  existsi 3,
  split,
  { -- W = 3
    refl },
  { -- The exercise time calculation equality holds.
    let walking_distance_per_day := 3,
    let running_distance_per_day := 10,
    let running_speed := 5,
    let exercise_hours_per_week := 21,
    let days_per_week := 7,
    
    let walking_distance_per_week := walking_distance_per_day * days_per_week,
    let running_distance_per_week := running_distance_per_day * days_per_week,
    let walking_time_per_week := walking_distance_per_week / 3,
    let running_time_per_week := running_distance_per_week / running_speed,

    have h1 : walking_distance_per_week = 21,
    { refl },
    have h2 : running_distance_per_week = 70,
    { refl },
    have h_walking_time : walking_time_per_week = 7,
    { simp [walking_distance_per_week, h1, div_eq_mul_inv, div_eq_mul_inv,
            mul_comm, mul_assoc, mul_inv_cancel, ne_of_gt, nat.cast_pos.mpr(lt_of_le_of_lt zero_le_one
            zero_lt_one)] },
    have h_running_time : running_time_per_week = 14,
    { norm_num1 },
    simp only [h_walking_time, h_running_time],
    norm_num1, }
end

end Tony_walking_speed_l23_23893


namespace not_even_nor_odd_l23_23397

def g (x : ℝ) : ℝ := 2 ^ (x^2 - 4*x + 3) - 2 * |x|

theorem not_even_nor_odd : 
  (∃ x : ℝ, g (-x) ≠ g x) ∧ (∃ x : ℝ, g (-x) ≠ -g x) := by
  sorry

end not_even_nor_odd_l23_23397


namespace Danny_wrappers_collection_l23_23620

variable {n_wrappers n_bottle_caps_in_park n_current_bottle_caps : ℕ}

-- Conditions
variables (found_wrappers : n_wrappers = 46)
variables (found_bottle_caps : n_bottle_caps_in_park = 50)
variables (current_bottle_caps : n_current_bottle_caps = 21)
variables (bottle_caps_more_than_wrappers : n_bottle_caps_in_park = n_wrappers + 4)

-- Proof problem statement
theorem Danny_wrappers_collection (h1 : found_wrappers) (h2 : found_bottle_caps)
  (h3 : current_bottle_caps) (h4 : bottle_caps_more_than_wrappers) :
  n_wrappers = 46 := 
by sorry

end Danny_wrappers_collection_l23_23620


namespace trail_mix_dried_fruit_percentage_l23_23927

def sue_trail_mix := (nuts: ℝ, dried_fruit: ℝ)
def jane_trail_mix := (nuts: ℝ, chocolate_chips: ℝ)
def combined_trail_mix := (nuts: ℝ, dried_fruit: ℝ)

theorem trail_mix_dried_fruit_percentage :
  (sue_trail_mix 0.30 0.70) → 
  (jane_trail_mix 0.60 0.40) → 
  (combined_trail_mix 0.45 _) → 
  ∃ x, combined_trail_mix 0.45 x ∧ x = 0.35 :=
by
  sorry

end trail_mix_dried_fruit_percentage_l23_23927


namespace perpendiculars_intersect_at_circumcenter_l23_23806

variables {A B C M N K B1 C1 A1 : Type}
-- Conditions
variables [triangle A B C]
variables [regular_triangle A M B] [regular_triangle B N C] [regular_triangle C K A]
variables [midpoint B1 M N] [midpoint C1 N K] [midpoint A1 K M]

-- Question: Prove that the perpendiculars to AC, AB, and BC through the midpoints of MN, NK, and KM respectively intersect at the same point
theorem perpendiculars_intersect_at_circumcenter : ∃ O : Type, 
  (perpendicular_through_midpoint B1 A C) ∧ (perpendicular_through_midpoint C1 A B) ∧ (perpendicular_through_midpoint A1 B C) ∧ 
  concurrent_at_circumcenter O :=
sorry

end perpendiculars_intersect_at_circumcenter_l23_23806


namespace not_a_proposition_l23_23543

-- Definitions based on conditions
def A : Prop := "Whales are mammals"
def B : String := "Have you finished your homework?"
def C : Prop := "All plants need water"
def D : Prop := "Real numbers include zero"

-- Proposition definition
def is_proposition (s : String) : Prop :=
  s = "Whales are mammals" ∨ s = "All plants need water" ∨ s = "Real numbers include zero"

-- Proof statement
theorem not_a_proposition : ¬ is_proposition B :=
sorry

end not_a_proposition_l23_23543


namespace area_of_cos_integral_l23_23074

theorem area_of_cos_integral : 
  (∫ x in (0:ℝ)..(3 * Real.pi / 2), |Real.cos x|) = 3 :=
by
  sorry

end area_of_cos_integral_l23_23074


namespace problem_composite_for_n_geq_9_l23_23647

theorem problem_composite_for_n_geq_9 (n : ℤ) (h : n ≥ 9) : ∃ k m : ℤ, (2 ≤ k ∧ 2 ≤ m ∧ n + 7 = k * m) :=
by
  sorry

end problem_composite_for_n_geq_9_l23_23647


namespace a_is_perfect_square_l23_23032

theorem a_is_perfect_square (a b : ℕ) (h : ∃ (k : ℕ), a^2 + b^2 + a = k * a * b) : ∃ n : ℕ, a = n^2 := by
  sorry

end a_is_perfect_square_l23_23032


namespace conversion_1_conversion_2_trigonometric_values_l23_23562

-- Definitions
def degrees_to_radians (deg: ℝ) : ℝ := deg * (Real.pi / 180)
def radians_to_degrees (rad: ℝ) : ℝ := rad * (180 / Real.pi)

def P := (2 * Real.sin (Real.pi / 6), -2 * Real.cos (Real.pi / 6))

-- Proves
theorem conversion_1 : degrees_to_radians (-15) = -Real.pi / 12 :=
by sorry

theorem conversion_2 : radians_to_degrees (7 * Real.pi / 12) = 105 :=
by sorry

theorem trigonometric_values (α : ℝ) :
  -- Given terminal side of α passes through P(2sin30, -2cos30)
  P = (1, -Real.sqrt 3) →
  -- Prove the trigonometric values
  Real.sin α = -Real.sqrt 3 / 2 ∧ Real.cos α = 1 / 2 ∧ Real.tan α = -Real.sqrt 3 :=
by 
sorry

end conversion_1_conversion_2_trigonometric_values_l23_23562


namespace spherical_coordinates_of_neg_y_l23_23959

def spherical_coordinates_neq (x y z: ℝ) : Prop :=
∃ x y z : ℝ,
  x = 3 * Real.sin (π / 4) * Real.cos (5 * π / 6) ∧
  y = 3 * Real.sin (π / 4) * Real.sin (5 * π / 6) ∧
  z = 3 * Real.cos (π / 4)

theorem spherical_coordinates_of_neg_y (x y z : ℝ) (h : spherical_coordinates_neq x y z) :
  (3, 7 * π / 6, π / 4) = 
  let θ_new := 2 * π - 5 * π / 6 in
  (3, θ_new, π / 4) :=
by
  sorry

end spherical_coordinates_of_neg_y_l23_23959


namespace points_in_triangle_with_area_at_most_4_l23_23941

noncomputable def triangle_enclosure (k : ℕ) (points : Fin k → ℝ × ℝ) : Prop :=
  ∀ (A B C : Fin k),
  let area (p1 p2 p3 : ℝ × ℝ) := (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2
  in abs (area (points A) (points B) (points C)) ≤ 1 →
     ∃ (p1 p2 p3 : ℝ × ℝ), 
     let enclosing_area := abs (area p1 p2 p3)
     in (∀ (i : Fin k), 
          let xi := points i
          in enclosing_area ≤ 4 ∧ 
              (xi.1 - p1.1) * (p2.2 - p1.2) - (xi.2 - p1.2) * (p2.1 - p1.1) ≤ 0) -- xi is inside the triangle

theorem points_in_triangle_with_area_at_most_4 (k : ℕ) (points : Fin k → ℝ × ℝ) :
  triangle_enclosure k points :=
sorry

end points_in_triangle_with_area_at_most_4_l23_23941


namespace monotonic_intervals_a_eq_1_range_of_a_for_inequality_l23_23707

-- Define the function f(x) given a parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x * exp x - a * (x^2 + 2 * x)

-- Statement that asserts the monotonic intervals when a = 1
theorem monotonic_intervals_a_eq_1 : 
  (monotonic_on ((λ x, f 1 x) : ℝ → ℝ) set.Iio (-1) ∧
   antitone_on ((λ x, f 1 x) : ℝ → ℝ) (set.Ioo (-1) (real.log 2)) ∧
   monotonic_on ((λ x, f 1 x) : ℝ → ℝ) set.Ioi (real.log 2)) :=
sorry

-- Statement that asserts the range of a for f(x) ≤ 0 when x < 0
theorem range_of_a_for_inequality : 
  (∀ x < 0, f a x ≤ 0) ↔ (0 < a ∧ a ≤ 1/exp 1) :=
sorry

end monotonic_intervals_a_eq_1_range_of_a_for_inequality_l23_23707


namespace f_is_n_l23_23010

noncomputable def f : ℕ+ → ℤ :=
  sorry

def f_defined_for_all_positive_integers (n : ℕ+) : Prop :=
  ∃ k, f n = k

def f_is_integer (n : ℕ+) : Prop :=
  ∃ k : ℤ, f n = k

def f_two_is_two : Prop :=
  f 2 = 2

def f_multiply_rule (m n : ℕ+) : Prop :=
  f (m * n) = f m * f n

def f_ordered (m n : ℕ+) (h : m > n) : Prop :=
  f m > f n

theorem f_is_n (n : ℕ+) :
  (f_defined_for_all_positive_integers n) →
  (f_is_integer n) →
  (f_two_is_two) →
  (∀ m n, f_multiply_rule m n) →
  (∀ m n (h : m > n), f_ordered m n h) →
  f n = n :=
sorry

end f_is_n_l23_23010


namespace find_abs_x_l23_23440

-- Given conditions
def A (x : ℝ) : ℝ := 3 + x
def B (x : ℝ) : ℝ := 3 - x
def distance (a b : ℝ) : ℝ := abs (a - b)

-- Problem statement: Prove |x| = 4 given the conditions
theorem find_abs_x (x : ℝ) (h : distance (A x) (B x) = 8) : abs x = 4 := 
  sorry

end find_abs_x_l23_23440


namespace lemming_average_distance_l23_23197

def average_distance_to_square_sides (side_length : ℝ) (d1 d2 move_distance : ℝ) : ℝ :=
  let diagonal_length := Real.sqrt (2 * (side_length ^ 2))
  let fraction_of_diagonal := d1 / diagonal_length
  let x1 := (fraction_of_diagonal * side_length)
  let y1 := (fraction_of_diagonal * side_length)
  let x2 := x1
  let y2 := y1 + d2
  let distance_to_left := x2
  let distance_to_bottom := y2
  let distance_to_right := side_length - x2
  let distance_to_top := side_length - y2
  (distance_to_left + distance_to_bottom + distance_to_right + distance_to_top) / 4

theorem lemming_average_distance (side_length d1 d2 : ℝ) :
  side_length = 8 → d1 = 4.8 → d2 = 2.5 →
  average_distance_to_square_sides side_length d1 d2 = 4 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- Proof continues here
  sorry

end lemming_average_distance_l23_23197


namespace monthly_pension_supplement_l23_23067

theorem monthly_pension_supplement 
  (initial_age : ℕ) 
  (start_age : ℕ)
  (contribution_period_years : ℕ) 
  (monthly_contribution : ℕ) 
  (annual_interest_rate : ℝ) 
  (retirement_age : ℕ) 
  (years_after_retirement : ℕ) :
  initial_age = 39 → 
  start_age = 40 →
  contribution_period_years = 20 →
  monthly_contribution = 7000 →
  annual_interest_rate = 0.09 →
  retirement_age = 60 →
  years_after_retirement = 15 →
  let annual_contribution := (monthly_contribution * 12 : ℕ)
  let future_value := annual_contribution * ((1 + annual_interest_rate) ^ contribution_period_years - 1) / annual_interest_rate * (1 + annual_interest_rate)
  let total_accumulation := future_value
  let monthly_supplement := total_accumulation / (years_after_retirement * 12) in
  monthly_supplement ≈ 26023.45 :=
begin
  intros h_initial_age h_start_age h_contribution_period h_monthly_contribution h_interest_rate h_retirement_age h_years_after_retirement,
  let annual_contribution := (monthly_contribution * 12 : ℕ),
  have h_annual_contribution : annual_contribution = 84000, by sorry,
  -- (continue with the definition using the factual computations if needed, ending with the approximate value)
  let future_value := annual_contribution * ((1 + annual_interest_rate) ^ contribution_period_years - 1) / annual_interest_rate * (1 + annual_interest_rate),
  have h_future_value : future_value ≈ 4684220.554, by sorry,
  let total_accumulation := future_value,
  let monthly_supplement := total_accumulation / (years_after_retirement * 12),
  have h_monthly_supplement : monthly_supplement ≈ 26023.45, by sorry,
  exact h_monthly_supplement
end

end monthly_pension_supplement_l23_23067


namespace find_minimum_area_l23_23420

noncomputable def right_triangle_incenter_problem : Prop :=
  ∃ (A B C Y J1 J2 : Point) (α β γ : ℝ),
  let a := 40
  let b := 30
  let c := 50
  let triangle_ABC := Triangle.mk A B C,
  let is_right_triangle_ABC := (angle A B C = π / 2)
  ∧ (dist A B = 40)
  ∧ (dist A C = 30)
  ∧ (dist B C = 50)
  ∧ (Y ∈ LineSegment.mk B C)
  ∧ (J1 = incenter_triangle_A_B_Y)
  ∧ (J2 = incenter_triangle_A_C_Y)
  ∧ (area_triangle_A_J1_J2 = 75)

theorem find_minimum_area (A B C Y J1 J2 : Point) (α β γ : ℝ) :
  right_triangle_incenter_problem :=
begin
  sorry
end

end find_minimum_area_l23_23420


namespace brendan_taxes_l23_23976

def total_hours (num_8hr_shifts : ℕ) (num_12hr_shifts : ℕ) : ℕ :=
  (num_8hr_shifts * 8) + (num_12hr_shifts * 12)

def total_wage (hourly_wage : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_wage * hours_worked

def total_tips (hourly_tips : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_tips * hours_worked

def reported_tips (total_tips : ℕ) (report_fraction : ℕ) : ℕ :=
  total_tips / report_fraction

def reported_income (wage : ℕ) (tips : ℕ) : ℕ :=
  wage + tips

def taxes (income : ℕ) (tax_rate : ℚ) : ℚ :=
  income * tax_rate

theorem brendan_taxes (num_8hr_shifts num_12hr_shifts : ℕ)
    (hourly_wage hourly_tips report_fraction : ℕ) (tax_rate : ℚ) :
    (hourly_wage = 6) →
    (hourly_tips = 12) →
    (report_fraction = 3) →
    (tax_rate = 0.2) →
    (num_8hr_shifts = 2) →
    (num_12hr_shifts = 1) →
    taxes (reported_income (total_wage hourly_wage (total_hours num_8hr_shifts num_12hr_shifts))
            (reported_tips (total_tips hourly_tips (total_hours num_8hr_shifts num_12hr_shifts))
            report_fraction))
          tax_rate = 56 :=
by
  intros
  sorry

end brendan_taxes_l23_23976


namespace sqrt_meaningful_iff_ge_two_l23_23727

-- State the theorem according to the identified problem and conditions
theorem sqrt_meaningful_iff_ge_two (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 2)) → x ≥ 2 :=
by
  sorry  -- Proof placeholder

end sqrt_meaningful_iff_ge_two_l23_23727


namespace inequality_sum_div_l23_23457

theorem inequality_sum_div (n : ℕ) (h : 0 < n) :
  (∑ k in Finset.range n, 1 / ((k + 1) * (3 * (k + 1) - 1))) ≥ 1 / (n + 1) := by
  sorry

end inequality_sum_div_l23_23457


namespace recycling_rate_l23_23131

-- Define the conditions
def total_pounds : Nat := 36
def total_points : Nat := 4

-- Define the question (we're looking to prove the rate of pounds per point)
def pounds_per_point : Nat := total_pounds / total_points

-- Define the statement that needs to be proved
theorem recycling_rate : pounds_per_point = 9 := by
  const total_pounds = 36
  const total_points = 4
  exact 36 / 4 = 9
sorry

end recycling_rate_l23_23131


namespace solve_for_y_l23_23377

theorem solve_for_y (x y : ℝ) (h1 : x = 8) (h2 : x^(3 * y) = 8) : y = 1 / 3 := 
by
  sorry

end solve_for_y_l23_23377


namespace option_b_option_c_option_d_l23_23149

theorem option_b (x : ℝ) (h : x > 1) : (∀ y, y = 2*x + 4 / (x - 1) - 1 → y ≥ 4*Real.sqrt 2 + 1) :=
by
  sorry

theorem option_c (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3 * x * y) : 2*x + y ≥ 3 :=
by
  sorry

theorem option_d (x y : ℝ) (h : 9*x^2 + y^2 + x*y = 1) : 3*x + y ≤ 2*Real.sqrt 21 / 7 :=
by
  sorry

end option_b_option_c_option_d_l23_23149


namespace B_finishes_remaining_work_in_8_days_l23_23550

-- Definitions of conditions
constant work_total : ℕ := 1  -- Total work is considered as a whole (1 unit)

constant A_day : ℕ := 5       -- A can finish the work in 5 days
constant B_day : ℕ := 16      -- B can finish the work in 16 days
constant together_days : ℕ := 2 -- A and B work together for 2 days

-- Defining the proof problem
theorem B_finishes_remaining_work_in_8_days :
  let A_work_rate : ℚ := 1 / A_day
      B_work_rate : ℚ := 1 / B_day
      combined_work_rate : ℚ := A_work_rate + B_work_rate
      work_done_together : ℚ := combined_work_rate * together_days
      remaining_work : ℚ := work_total - work_done_together
      B_remaining_days : ℚ := remaining_work * B_day
  in B_remaining_days ≤ 8 :=
by {
  have hA : A_work_rate = 1 / 5 := rfl,
  have hB : B_work_rate = 1 / 16 := rfl,
  have hCombined : combined_work_rate = 1 / 5 + 1 / 16 :=
    by rw [hA, hB],
  have hWorkDoneTogether : work_done_together = (1 / 5 + 1 / 16) * 2 :=
    by rw [hCombined],
  have hRemaining : remaining_work = 1 - (21 / 40) :=
    by rw [hWorkDoneTogether],
  have hBRemainingDays : B_remaining_days = (19 / 40) * 16 :=
    by rw [hRemaining],
  have : B_remaining_days = 7.6 :=
    by norm_num [hBRemainingDays],
  linarith only [this],
  sorry,
}

end B_finishes_remaining_work_in_8_days_l23_23550


namespace symmetric_graph_g_l23_23663

def f (x : ℝ) : ℝ := (2 * x + 3) / (x - 1)
def f_inv (y : ℝ) : ℝ := (y + 3) / (y - 2)
def f_inv_shifted (x : ℝ) : ℝ := f_inv (x + 1)

theorem symmetric_graph_g :
  (∀ x, ∃ y, g x = y ↔ f_inv_shifted y = x) →
  g 3 = 7 / 2 :=
by
  sorry

end symmetric_graph_g_l23_23663


namespace ratio_of_areas_of_cone_parts_l23_23582

-- Define the problem conditions
def cone_sections (height : ℝ) (r1 r2 r3 : ℝ) (S1 S2 S3 : ℝ) : Prop :=
  r1 / r2 = 1 / 2 ∧ r1 / r3 = 1 / 3 ∧
  S1 / S2 = 1 / 4 ∧ S1 / S3 = 1 / 9 ∧
  S2 > S1 ∧ S3 > S2

-- Introduce the main theorem
theorem ratio_of_areas_of_cone_parts 
  (height : ℝ) 
  (r1 r2 r3 : ℝ) 
  (S1 S2 S3 : ℝ) 
  (h_cone_sections : cone_sections height r1 r2 r3 S1 S2 S3) :
  S1 : (S2 - S1) : (S3 - S2) = 1 : 3 : 5 :=
sorry

end ratio_of_areas_of_cone_parts_l23_23582


namespace find_a_l23_23875

noncomputable def a_probability (a : ℝ) : Prop := 
  ∑ k in {1, 2, 3, 4}, a * k = 1

theorem find_a (a : ℝ) : a_probability a → a = 1 / 10 :=
by
  sorry

end find_a_l23_23875


namespace find_b_for_continuity_at_2_l23_23814

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if h : x ≤ 2 then 4 * x^2 + 5 else b * x + 3

theorem find_b_for_continuity_at_2 (b : ℝ) : (∀ x, f x b = if x ≤ 2 then 4 * x^2 + 5 else b * x + 3) ∧ 
  (f 2 b = 21) ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x b - f 2 b| < ε) → 
  b = 9 :=
by
  sorry

end find_b_for_continuity_at_2_l23_23814


namespace function_monotonic_decreasing_interval_l23_23700

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6)

theorem function_monotonic_decreasing_interval :
  ∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
  ∀ y ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
  (x ≤ y → f y ≤ f x) :=
by
  sorry

end function_monotonic_decreasing_interval_l23_23700


namespace num_ways_to_buy_souvenirs_is_266_l23_23628

-- Define the problem and the given conditions.
def num_types : ℕ := 11
def price_10_yuan_types : ℕ := 8
def price_5_yuan_types : ℕ := 3
def total_spent : ℕ := 50
def max_quantity_each_type : ℕ := 1

-- The problem statement: the number of different ways to buy the souvenirs is 266.
theorem num_ways_to_buy_souvenirs_is_266 :
  number_of_ways_to_spend_50_yuan num_types price_10_yuan_types price_5_yuan_types total_spent max_quantity_each_type = 266 :=
sorry

end num_ways_to_buy_souvenirs_is_266_l23_23628


namespace find_a_b_l23_23915

theorem find_a_b :
  ∃ (a b : ℚ), 
    (∀ x : ℚ, x = 2 → (a * x^3 - 6 * x^2 + b * x - 5 - 3 = 0)) ∧
    (∀ x : ℚ, x = -1 → (a * x^3 - 6 * x^2 + b * x - 5 - 7 = 0)) ∧
    (a = -2/3 ∧ b = -52/3) :=
by {
  sorry
}

end find_a_b_l23_23915


namespace consecutive_integers_sum_l23_23501

theorem consecutive_integers_sum (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end consecutive_integers_sum_l23_23501


namespace not_always_product_greater_l23_23101

-- Define the premise and the conclusion
theorem not_always_product_greater (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b < 1) : a * b < a :=
sorry

end not_always_product_greater_l23_23101


namespace four_identical_pairwise_differences_l23_23289

theorem four_identical_pairwise_differences (S : Finset ℕ) (h_distinct : S.card = 20) (h_range : ∀ x ∈ S, 0 ≤ x ∧ x < 70) :
  ∃ d ∈ (S.product S).image (λ p, (p.2 - p.1).natAbs), (S.product S).count (λ p, (p.2 - p.1).natAbs = d) ≥ 4 :=
by
  -- Sorry is used to skip the proof.
  sorry

end four_identical_pairwise_differences_l23_23289


namespace integer_solutions_count_eq_two_l23_23623

theorem integer_solutions_count_eq_two : 
  {x : ℤ | (x - 3) ^ (28 - x^2) = 1}.to_finset.card = 2 := 
  sorry

end integer_solutions_count_eq_two_l23_23623


namespace triangle_median_theorem_l23_23300

theorem triangle_median_theorem (A B C O : Point) (median : Medians A B C intersect_at O) :
  let AB := dist A B
  let BC := dist B C
  let CA := dist C A
  let OA := dist O A
  let OB := dist O B
  let OC := dist O C
  in AB^2 + BC^2 + CA^2 = 3 * (OA^2 + OB^2 + OC^2) := 
sorry

end triangle_median_theorem_l23_23300


namespace range_of_a_l23_23794

theorem range_of_a {a : ℝ} : (∀ x : ℝ, (x^2 + 2 * (a + 1) * x + a^2 - 1 = 0) → (x = 0 ∨ x = -4)) → (a = 1 ∨ a ≤ -1) := 
by {
  sorry
}

end range_of_a_l23_23794


namespace amount_of_bill_l23_23887

noncomputable def TD : ℝ := 360
noncomputable def BD : ℝ := 418.9090909090909
noncomputable def FV (TD BD : ℝ) : ℝ := TD * BD / (BD - TD)

theorem amount_of_bill :
  FV TD BD = 2568 :=
by
  sorry

end amount_of_bill_l23_23887


namespace sector_ratio_is_2_over_9_l23_23043

noncomputable def circle_sector_ratio (angle_AOC angle_DOB : ℝ) : ℝ :=
  let angle_AOB := 180
  let angle_COD := angle_AOB - angle_AOC - angle_DOB
  angle_COD / 360

theorem sector_ratio_is_2_over_9 {O A B C D : Type} (angle_AOC angle_DOB : ℝ)
    (hAOC : angle_AOC = 40)
    (hDOB : angle_DOB = 60)
    (hAOB : angle_AOB = 180) :
    circle_sector_ratio angle_AOC angle_DOB = 2 / 9 :=
by
  rw [circle_sector_ratio, hAOC, hDOB, hAOB]
  sorry

end sector_ratio_is_2_over_9_l23_23043


namespace downstream_rate_l23_23588

/--  
A man's rowing conditions and rates:
- The man's upstream rate is U = 12 kmph.
- The man's rate in still water is S = 7 kmph.
- We need to prove that the man's downstream rate D is 14 kmph.
-/
theorem downstream_rate (U S D : ℝ) (hU : U = 12) (hS : S = 7) : D = 14 :=
by
  -- Proof to be filled here
  sorry

end downstream_rate_l23_23588


namespace original_average_age_l23_23557

variable (N : ℕ) (A : ℕ) (orig_avg new_students : ℕ) (new_avg : ℕ)

-- Conditions
def orig_avg_age : Prop := orig_avg = 40
def new_students_avg_age : Prop := (N * orig_avg + 12 * 32) / (N + 12) = new_avg
def age_decrease : Prop := A = orig_avg - 6

-- Prove that given the conditions, the original average age is 40.
theorem original_average_age :
  orig_avg = 40 → orig_avg_age → new_students_avg_age → age_decrease → orig_avg = 40 :=
by
  intros
  sorry

end original_average_age_l23_23557


namespace infinite_set_of_points_in_plane_l23_23165

noncomputable def infinite_set_of_points_exists : Prop :=
  ∃ (P : ℕ → ℝ × ℝ),
  (∀ i j k : ℕ, (i ≠ j ∧ j ≠ k ∧ i ≠ k) → ¬ collinear (P i) (P j) (P k)) ∧
  (∀ i j : ℕ, i ≠ j → is_rational (dist (P i) (P j)))

theorem infinite_set_of_points_in_plane :
  infinite_set_of_points_exists :=
sorry

end infinite_set_of_points_in_plane_l23_23165


namespace seventh_row_is_correct_no_all_black_row_row_2014_is_correct_l23_23837

-- Define the Circle type as either black or white
inductive Circle
| Black : Circle
| White : Circle

-- Define the rule for determining the next circle color
def next_circle (left : Circle) (right : Circle) : Circle :=
  match left, right with
  | Circle.Black, Circle.Black => Circle.Black
  | Circle.White, Circle.White => Circle.Black
  | _, _ => Circle.White

-- Function to compute the next row given the previous row
def next_row (prev_row : List Circle) : List Circle :=
  if prev_row.length < 2 then []
  else (prev_row.zipWith next_circle (prev_row.tail.getD prev_row)).take prev_row.length

-- Define the initial row 
def initial_row : List Circle := [Circle.White, Circle.Black, Circle.White, Circle.Black, Circle.Black, Circle.White]

-- Recursive function to compute the nth row, given the initial row
def nth_row (n : Nat) (row : List Circle) : List Circle :=
  match n with
  | 0 => row
  | Nat.succ k => nth_row k (next_row row)

-- Theorem for seventh row
theorem seventh_row_is_correct : nth_row 7 initial_row = [Circle.White, Circle.White, Circle.White, Circle.White, Circle.Black, Circle.Black] := sorry

-- Theorem for no all-black row
theorem no_all_black_row : ∀ n : Nat, list.all (nth_row n initial_row) (λ c, c = Circle.Black) = false := sorry

-- Theorem for 2014th row
theorem row_2014_is_correct : nth_row 2014 initial_row = [Circle.Black, Circle.White, Circle.Black, Circle.Black, Circle.Black, Circle.White] := sorry

end seventh_row_is_correct_no_all_black_row_row_2014_is_correct_l23_23837


namespace locus_of_midpoint_is_hyperbola_and_line_is_tangent_l23_23004

noncomputable def hyperbola := {p : ℝ × ℝ | p.1^2 - p.2^2 = 2}

theorem locus_of_midpoint_is_hyperbola_and_line_is_tangent
  (A : ℝ)
  (h_area : ∀ l : ℝ → ℝ, 
    ∃ s t : ℝ, 0 < s → s < t →
    ∫ x in s..t, (l x - (1 / x)) = A) :
  (∃ k : ℝ, 
    ∀ (s t : ℝ), 0 < s → s < t → 
    (∃ (X Y : ℝ), 
      (Y = (1 + k)^2 / (4 * k * X)) ∧ 
      (X = (s + k * s) / 2) ∧ 
      (Y = (1 + k) / (2 * k * s)))) ∧
  (∀ (X : ℝ), 
    ∃ (m : ℝ), 
    (∂/∂X (1 + k)^2 / (4 * k * X)) = - (1 / (k * s^2))) ∧
  (∀ (X : ℝ),
    l X = (s + k * s - X) / (k * s^2)) := 
sorry

end locus_of_midpoint_is_hyperbola_and_line_is_tangent_l23_23004


namespace smallest_expression_l23_23731

theorem smallest_expression (a b : ℝ) (h : b < 0) : a + b < a ∧ a < a - b :=
by
  sorry

end smallest_expression_l23_23731


namespace part_one_part_two_l23_23693

noncomputable theory

def f (x : ℝ) : ℝ := cos (2 * x) + 2 * sin x * sin x

theorem part_one:
  (∀ x : ℝ, f x + f (x + π) = 2) 
  ∧ (∀ x : ℝ, x = k * π → f x = 2) 
  ∧ (∀ x : ℝ, f x = 0 -> x ≠ k * π * 0.5) :=
sorry

variables {A : ℝ} {b : ℝ} {a : ℝ}

theorem part_two (hA : 0 < A ∧ A < π/2) (ha : a = 7) (hb : b = 5) (hfA : f A = 0):
  (triangle_area a b A = 10) :=
sorry

end part_one_part_two_l23_23693


namespace sequence_prime_n_form_l23_23018

theorem sequence_prime_n_form (a : ℕ → ℕ) (h1 : a 1 = 3) (h2 : a 2 = 7)
    (h3 : ∀ n, 2 ≤ n → (a n)^2 + 5 = a (n - 1) * a (n + 1)) :
    (∀ n, Prime (a n + (-1)^n) → ∃ m : ℕ, n = 3^m) :=
begin
  sorry
end

end sequence_prime_n_form_l23_23018


namespace additional_handshakes_required_l23_23215

theorem additional_handshakes_required (people : ℕ) (initial_handshakes : ℕ) : 
  ∀ (people = 10) (initial_handshakes = 3), 
  let total_handshakes := people * (people - 1) / 2 in
  total_handshakes - initial_handshakes = 42 :=
by
  intros
  unfold total_handshakes
  sorry

end additional_handshakes_required_l23_23215


namespace kopeck_suffice_l23_23431

-- Defining the initial conditions
def initial_conditions (x y : ℝ) : Prop :=
  x + y = 1 ∧ 1 ≥ 0.6 * x + 1.2 * y

-- The statement we want to prove
theorem kopeck_suffice (x y : ℝ) (h : initial_conditions x y) : 3 * x ≥ 2.88 * x :=
by
  obtain ⟨hx, hy⟩ := h
  -- Simplify the given conditions to find the necessary relationships
  have h1 : y = 2 * x,
  sorry
  -- With y = 2x, we need to show 3x ≥ 2.88x
  have h2 : 3 * x = 3 * x,
  sorry
  have h3 : 2.88 * x + 0 < 3 * x,
  sorry
  exact le_of_lt h3

end kopeck_suffice_l23_23431


namespace period_and_interval_and_a_value_l23_23699

noncomputable def fx (x : ℝ) : ℝ := sin (7 * π / 6 - 2 * x) - 2 * (sin x) ^ 2 + 1

theorem period_and_interval_and_a_value (k : ℤ) :
  (∀ x : ℝ, fx (x + π) = fx x) ∧
  (∀ x : ℝ, k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 → ∀ y : ℝ, x ≤ y → fx x ≤ fx y) ∧
  (∃ A b c : ℝ, fx A = 1/2 ∧ b = A ∧ c = 3 * sqrt 2 ∧ 2 * c = b + A ∧ b * c * cos A = 9 ∧ c = 3 * sqrt 2) := 
sorry

end period_and_interval_and_a_value_l23_23699


namespace Cameron_list_count_l23_23982

theorem Cameron_list_count : 
  let lower_bound := 900
  let upper_bound := 27000
  let step := 30
  let n_min := lower_bound / step
  let n_max := upper_bound / step
  n_max - n_min + 1 = 871 :=
by
  sorry

end Cameron_list_count_l23_23982


namespace ratio_DE_AE_l23_23607

theorem ratio_DE_AE (A B C D E : Point) (m : ℝ) (circumcircle : Circle) :
  IsCircumscribedCircle circumcircle A B C →
  Diameter circumcircle AD →
  IntersectsAt AD BC E →
  AE = AC →
  Ratio BE CE = m →
  Ratio DE AE = 2 / (m + 1) :=
by
  sorry

end ratio_DE_AE_l23_23607


namespace sum_of_divisibles_l23_23733

theorem sum_of_divisibles (x a : ℕ) (h1 : x = 5) (h2 : 2 * a ≡ 0 [MOD 3])
  (valid_digits : a ∈ {0, 3, 6, 9}) :
  let nums := if a = 0 then [50505, 53535, 56565, 59595] else [] 
  let sum_nums := list.sum nums
  sum_nums = 220200 := by
  sorry

end sum_of_divisibles_l23_23733


namespace problem_l23_23000

def g (x : ℤ) : ℤ := 3 * x^2 - x + 4

theorem problem : g (g 3) = 2328 := by
  sorry

end problem_l23_23000


namespace mod25_pow_inverse_l23_23899

theorem mod25_pow_inverse :
  (let x := 5 in 
   let a := x 
   in (a + 0)^2) % 25 = 0 := 
by 
let a := 5
exact (a + 0)^2 % 25 = 0


end mod25_pow_inverse_l23_23899


namespace arithmetic_sequence_sum_l23_23391

-- Define that a sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : a 2 + a 10 = 16) : a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l23_23391


namespace smallest_positive_angle_l23_23643

theorem smallest_positive_angle (x : ℝ) (hx_pos : 0 < x) (hx_deg : x = 90 / 7): 
  ∃ (x : ℝ), (sin (3 * x) * sin (4 * x) = cos (3 * x) * cos (4 * x)) ∧ 0 < x ∧ x = 90 / 7 :=
sorry

end smallest_positive_angle_l23_23643


namespace range_of_a_l23_23109

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x + 5 > 3 ∧ x > a ∧ x ≤ -2) ↔ a ≤ -2 :=
by
  sorry

end range_of_a_l23_23109


namespace days_from_thursday_l23_23520

theorem days_from_thursday (n : ℕ) (h : n = 53) : 
  (n % 7 = 4) ∧ (n % 7 = 4 → "Thursday" + 4 days = "Monday") :=
by 
  have h1 : n % 7 = 4 := by sorry
  have h2 : "Thursday" + 4 days = "Monday" := by sorry
  exact ⟨h1, h2 h1⟩

end days_from_thursday_l23_23520


namespace factorial_div_l23_23141

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem factorial_div : (factorial 11 - factorial 10) / factorial 9 = 100 := 
by
  sorry

end factorial_div_l23_23141


namespace total_cost_bicycle_helmet_l23_23949

-- Let h represent the cost of the helmet
def helmet_cost := 40

-- Let b represent the cost of the bicycle
def bicycle_cost := 5 * helmet_cost

-- We need to prove that the total cost (bicycle + helmet) is equal to 240
theorem total_cost_bicycle_helmet : bicycle_cost + helmet_cost = 240 := 
by
  -- This will skip the proof, we only need the statement
  sorry

end total_cost_bicycle_helmet_l23_23949


namespace price_of_16_apples_equiv_12_cucumbers_l23_23736

-- Define the cost equivalences as given in the problem
variables (apple banana cucumber : Type)
variable (cost : Type)
variables (cost_eq : apple → banana → Prop) (cost_ratio : banana → cucumber → Prop)

-- Conditions given in the problem
axiom eight_apples_four_bananas : ∀ (a1 a2 a3 a4 a5 a6 a7 a8 : apple) (b1 b2 b3 b4 : banana), 
  cost_eq (a1, a2, a3, a4, a5, a6, a7, a8) (b1, b2, b3, b4)

axiom two_bananas_three_cucumbers : ∀ (b1 b2 : banana) (c1 c2 c3 : cucumber), 
  cost_ratio (b1, b2) (c1, c2, c3)

theorem price_of_16_apples_equiv_12_cucumbers : 
  ∀ (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 : apple) 
    (c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 : cucumber), 
    cost_eq (a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16) 
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12) :=
sorry

end price_of_16_apples_equiv_12_cucumbers_l23_23736


namespace find_bases_l23_23389

theorem find_bases {F1 F2 : ℝ} (R1 R2 : ℕ) 
                   (hR1 : R1 = 9)
                   (hR2 : R2 = 6)
                   (hF1_R1 : F1 = 0.484848 * 9^2 / (9^2 - 1))
                   (hF2_R1 : F2 = 0.848484 * 9^2 / (9^2 - 1))
                   (hF1_R2 : F1 = 0.353535 * 6^2 / (6^2 - 1))
                   (hF2_R2 : F2 = 0.535353 * 6^2 / (6^2 - 1))
                   : R1 + R2 = 15 :=
by
  sorry

end find_bases_l23_23389


namespace largest_n_for_sin_cos_inequality_l23_23258

theorem largest_n_for_sin_cos_inequality :
  ∀ (x : ℝ), sin x ^ 4 + cos x ^ 4 ≥ 1 / 2 :=
by
  -- The proof follows but is omitted here
  sorry

end largest_n_for_sin_cos_inequality_l23_23258


namespace finite_sequence_operations_pentagon_finite_sequence_operations_poly_l23_23831

def is_pentagon_sequence_finite (vertices : Fin 5 → ℤ)
  (h_sum_positive: 0 < (Fin.sum_univ vertices)) : Prop := sorry

def is_poly_sequence_finite (n : ℕ) (h_odd: n % 2 = 1)
  (vertices : Fin n → ℤ) (h_sum_positive: 0 < (Fin.sum_univ vertices)) : Prop := sorry

theorem finite_sequence_operations_pentagon
  (vertices : Fin 5 → ℤ)
  (h_sum_positive : 0 < (Fin.sum_univ vertices)) :
  is_pentagon_sequence_finite vertices h_sum_positive := sorry

theorem finite_sequence_operations_poly
  (n : ℕ)
  (h_odd : n % 2 = 1)
  (vertices : Fin n → ℤ)
  (h_sum_positive : 0 < (Fin.sum_univ vertices)) :
  is_poly_sequence_finite n h_odd vertices h_sum_positive := sorry

end finite_sequence_operations_pentagon_finite_sequence_operations_poly_l23_23831


namespace cameron_list_count_l23_23994

theorem cameron_list_count :
  let numbers := {n : ℕ | 30 ≤ n ∧ n ≤ 900}
  in set.card numbers = 871 :=
sorry -- proof is omitted

end cameron_list_count_l23_23994


namespace partition_orthogonal_parallelepiped_l23_23133

-- Condition definitions
structure OrthogonalParallelepiped where
  -- Define the necessary properties of an orthogonal parallelepiped

def is_right_faced (tetrahedron: Tetrahedron) : Prop :=
  -- Define what it means for a tetrahedron to be "right-faced"
  sorry

theorem partition_orthogonal_parallelepiped (op : OrthogonalParallelepiped) :
  ∃ tetrahedra : fin 6 → Tetrahedron, ∀ i, is_right_faced (tetrahedra i) :=
  sorry

end partition_orthogonal_parallelepiped_l23_23133


namespace simplify_radicals_l23_23464

theorem simplify_radicals :
  (Real.sqrt 18 * Real.cbrt 24 = 6 * Real.sqrt 2 * Real.cbrt 3) :=
by sorry

end simplify_radicals_l23_23464


namespace A_N_M_C_are_concyclic_l23_23936

open EuclideanGeometry

variables {A B C D Q R P S M N : Point}
variables (convex : ConvexQuadrilateral A B C D)
variables (angle_DAB_eq_90 : ∠ D A B = 90) (angle_BCD_eq_90 : ∠ B C D = 90)
variables (angle_ABC_gt_CDA : ∠ A B C > ∠ C D A)
variables (on_BC_Q : SegContains [B, C] Q) (on_CD_R : SegContains [C, D] R)
variables (intersects_AB_at_P : LineContains (LineThrough P Q) (LineThrough A B))
variables (intersects_AD_at_S : LineContains (LineThrough P Q) (LineThrough A D))
variables (dist_PQ_eq_RS : Dist P Q = Dist R S)
variables (midpoint_BD_M : Midpoint B D M) (midpoint_QR_N : Midpoint Q R N)

theorem A_N_M_C_are_concyclic : Concyclic A N M C := sorry

end A_N_M_C_are_concyclic_l23_23936


namespace total_farm_tax_collected_from_village_l23_23484

-- Define the conditions
variables (C T: ℝ)  -- total cultivated land and total taxable land
variables (tax_rate: ℝ)  -- farm tax rate
constant H1 : T = 0.60 * C  -- farm tax is levied on 60% of the cultivated land
constant H2 : 480 = 0.16 * T * tax_rate  -- Mr. William paid $480 as farm tax and his land is 16% of total taxable land

-- Theorem statement: the total farm tax collected from the village is $3000
theorem total_farm_tax_collected_from_village : (T * tax_rate) = 3000 :=
by
  sorry

end total_farm_tax_collected_from_village_l23_23484


namespace andrew_age_l23_23606

variable (a g s : ℝ)

theorem andrew_age :
  g = 10 * a ∧ g - s = a + 45 ∧ s = 5 → a = 50 / 9 := by
  sorry

end andrew_age_l23_23606


namespace function_passes_through_fixed_point_l23_23694

noncomputable def f (a x : ℝ) := a^(x+1) - 1

theorem function_passes_through_fixed_point (a : ℝ) (h_pos : 0 < a) (h_not_one : a ≠ 1) :
  f a (-1) = 0 := by
  sorry

end function_passes_through_fixed_point_l23_23694


namespace decimal_count_l23_23943

/-- Definition for counting the number of decimal digits in a number -/
def decimal_digits (n : ℝ) : ℕ :=
  if n = 0 then 0
  else
    let s := n.to_string in
    let fraction_part := s.split_on '.' |>.get_or_else 1 "" in
    fraction_part.length

/-- Problem statement: Prove that the decimal number 0.049 has 3 decimal digits -/
theorem decimal_count : decimal_digits 0.049 = 3 :=
by
  sorry

end decimal_count_l23_23943


namespace quadratic_eq_two_roots_of_pos_discrim_l23_23276

example (k : ℝ) : ∃ x y : ℝ, x ≠ y ∧ (x^2 - (k-3)*x - (k-1) = 0) ∧ (y^2 - (k-3)*y - (k-1) = 0) := 
by
  let Δ := (k-3)^2 + 4*(k-1)
  have hΔ : Δ > 0 := by sorry
  have h := quadratic_eq_two_roots_of_pos_discrim (by linarith)
  exact h

-- A supporting lemma that states that a quadratic equation with a positive discriminant has two distinct real roots
theorem quadratic_eq_two_roots_of_pos_discrim {a b c : ℝ} (h_pos : b^2 - 4*a*c > 0) : 
  ∃ x y : ℝ, x ≠ y ∧ (a*x^2 + b*x + c = 0) ∧ (a*y^2 + b*y + c = 0) := 
by
  sorry

end quadratic_eq_two_roots_of_pos_discrim_l23_23276


namespace time_to_cross_l23_23130

-- Define the parameters for the problem
def speed1 : ℝ := 90 -- in km/hr
def speed2 : ℝ := 90 -- in km/hr
def len1 : ℝ := 1.10 -- in km
def len2 : ℝ := 0.9 -- in km
def relative_speed : ℝ := speed1 + speed2 -- in km/hr
def total_distance : ℝ := len1 + len2 -- in km
def speed_conversion_factor : ℝ := 1 / 3600 -- converts km/hr to km/s
def relative_speed_in_km_per_s : ℝ := relative_speed * speed_conversion_factor -- in km/s

-- The theorem to prove
theorem time_to_cross : total_distance / relative_speed_in_km_per_s = 40 := by
  -- The compute time using distance and speed
  calc
    total_distance / relative_speed_in_km_per_s = (len1 + len2) / (relative_speed * speed_conversion_factor) : by rfl
    ... = 2.00 / (180 * (1/3600)) : by rfl
    ... = 2.00 / 0.05 : by rfl
    ... = 40 : by rfl

-- Placeholder for proof
sorry

end time_to_cross_l23_23130


namespace a_le_neg2_l23_23111

theorem a_le_neg2 (a : ℝ) : (∀ x : ℝ, (x + 5 > 3) → (x > a)) → a ≤ -2 :=
by
  intro h
  have h_neg : ∀ x : ℝ, (x > -2) → (x > a) := 
    by 
      intro x hx
      exact h x (by linarith)

  specialize h_neg (-1) (by linarith)
  linarith

end a_le_neg2_l23_23111


namespace tom_catches_twice_as_much_l23_23818

/-- Melanie catches 8 trout. Tom catches 2 times as many trout as Melanie. 
    How many trout did Tom catch? The correct answer is 16.  -/
theorem tom_catches_twice_as_much (trout_melanie : ℕ) (h_melanie : trout_melanie = 8) (h_tom : ∀ t, t = 2 * trout_melanie) :
  ∃ t, t = 16 :=
by
  use 2 * trout_melanie
  rw [h_melanie]
  norm_num

end tom_catches_twice_as_much_l23_23818


namespace find_duplicate_page_number_l23_23100

theorem find_duplicate_page_number (n : ℕ) (h1 : ∑ i in Finset.range (n + 1), i = n * (n + 1) / 2)
  (h2 : (∑ i in Finset.range (n + 1), i) + m = 2550) : 
  m = 80 :=
by
  sorry

end find_duplicate_page_number_l23_23100


namespace least_number_subtracted_378461_l23_23554

def least_number_subtracted (n : ℕ) : ℕ :=
  n % 13

theorem least_number_subtracted_378461 : least_number_subtracted 378461 = 5 :=
by
  -- actual proof would go here
  sorry

end least_number_subtracted_378461_l23_23554


namespace radius_of_inscribed_circle_l23_23051

-- Define the sector as a semi-circle with radius 4 cm
constant Sector : Type
constant Circle : Type
constant tangent : Circle → Sector → Prop

-- Define our specific semi-circle sector and inscribed circle
noncomputable def semi_circle_sector : Sector := sorry
noncomputable def inscribed_circle (r : ℝ) : Circle := sorry

def radius_of_semi_circle_sector : ℝ := 4

axiom tangency_conditions : tangent (inscribed_circle r) semi_circle_sector

-- The theorem stating the radius of the inscribed circle
theorem radius_of_inscribed_circle :
  (∃ r : ℝ, r = 4 * (Real.sqrt 2 - 1) ∧ tangent (inscribed_circle r) semi_circle_sector) :=
begin
  use (4 * (Real.sqrt 2 - 1)),
  split,
  { refl },
  { exact tangency_conditions },
end

end radius_of_inscribed_circle_l23_23051


namespace inequality_problem_l23_23284

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.logb 2 (1 / 3)
noncomputable def c : ℝ := Real.logb (1 / 2) (1 / 3)

theorem inequality_problem :
  c > a ∧ a > b := by
  sorry

end inequality_problem_l23_23284


namespace parabola_equation_l23_23740

theorem parabola_equation (d : ℝ) (p : ℝ) (x y : ℝ) (h1 : d = 2) (h2 : y = 2) (h3 : x = 1) :
  y^2 = 4 * x :=
sorry

end parabola_equation_l23_23740


namespace faster_train_speed_l23_23897

theorem faster_train_speed (v : ℝ) (h_total_length : 100 + 100 = 200) 
  (h_cross_time : 8 = 8) (h_speeds : 3 * v = 200 / 8) : 2 * v = 50 / 3 :=
sorry

end faster_train_speed_l23_23897


namespace perpendiculars_midpoints_concurrent_l23_23965

variables {A B C D X A' B' C' D' M N P Q : Type}

def cyclic_quadrilateral (A B C D : Type) : Prop :=
  ∃ O : Type, O ∈ circle A B C ∧ O ∈ circle B C D ∧ O ∈ circle C D A ∧ O ∈ circle D A B

def is_perpendicular (l1 l2 : Type) : Prop :=
  ∃ P : Type, P ∈ l1 ∧ P ∈ l2 ∧ angle l1 l2 = 90

def midpoint (X Y : Type) : Type :=
  ...

def is_concurrent (l1 l2 l3 l4 : Type) : Prop :=
  ∃ P : Type, P ∈ l1 ∧ P ∈ l2 ∧ P ∈ l3 ∧ P ∈ l4

-- Main statement
theorem perpendiculars_midpoints_concurrent
  (hcyclic : cyclic_quadrilateral A B C D)
  (h_intersect : X = AC ∩ BD)
  (hAA'_BD : is_perpendicular AA' BD ∧ A' ∈ BD)
  (hCC'_BD : is_perpendicular CC' BD ∧ C' ∈ BD)
  (hBB'_AC : is_perpendicular BB' AC ∧ B' ∈ AC)
  (hDD'_AC : is_perpendicular DD' AC ∧ D' ∈ AC)
  (hM : M = midpoint A B)
  (hN : N = midpoint B C)
  (hP : P = midpoint C D)
  (hQ : Q = midpoint D A)
  : ∃ O : Type, is_concurrent (perpendicular M CD) (perpendicular N DA) (perpendicular P AB) (perpendicular Q BC) := 
sorry

end perpendiculars_midpoints_concurrent_l23_23965


namespace probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l23_23760

noncomputable def probability_k_gnomes_fall (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p * (1 - p) ^ (n - k)

noncomputable def expected_number_of_fallen_gnomes (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p)

theorem probability_k_gnomes_fall_correct (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  probability_k_gnomes_fall n k p h = p * (1 - p) ^ (n - k) :=
by sorry

theorem expected_number_of_fallen_gnomes_correct (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  expected_number_of_fallen_gnomes n p h = n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p) :=
by sorry

end probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l23_23760


namespace arithmetic_sequence_sum_seven_l23_23319

noncomputable def sequence_sum : ℕ → ℚ
| 1 := 1
| n := 1 + (n - 1) * (1 / 3)

theorem arithmetic_sequence_sum_seven :
  let a1 := 1 in
  let d := (1 / 3) in
  let a := λ n, a1 + (n - 1) * d in
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 14 :=
by
  sorry

end arithmetic_sequence_sum_seven_l23_23319


namespace number_of_terms_in_cube_root_l23_23719

theorem number_of_terms_in_cube_root (n : ℕ) (h : ∛(n * 27) = 54) : n = 5832 :=
by sorry

end number_of_terms_in_cube_root_l23_23719


namespace part1_part2_part3_l23_23351

section
variables {a : ℝ} (f : ℝ → ℝ) (f' : ℝ → ℝ)

def tangent_line (a : ℝ) (p : ℝ × ℝ) : Prop := 
  ∃ m b, p.2 = m * p.1 + b ∧ 2*p.1 + 2*p.2 - 3 = 0

theorem part1 (h : a = 2) (p : ℝ × ℝ) : 
  tangent_line (λ x, (1/2) * x^2 - a * log x) p :=
  sorry

theorem part2 : 
  (∀ x > 1, x - (a / x) ≥ 0) → a ≤ 1 :=
  sorry

theorem part3 : a ≠ 0 → 
  (if a < 0 ∨ a = real.exp 1 then ∀ x, (f x = 0) → (f' x = 0) 
  else if 0 < a ∧ a < real.exp 1 then forall x, f x ≠ 0 
  else true) := 
  sorry

end

end part1_part2_part3_l23_23351


namespace graph_symmetry_l23_23352

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

theorem graph_symmetry (ω φ : ℝ) (hω : 0 < ω) (hφ : |φ| < Real.pi / 2)
  (h_sym_distance : ∀ x y, f ω φ x = f ω φ y → |x - y| = Real.pi / 4)
  (h_shifted_symmetry : ∀ x, f ω φ (x + 3 * Real.pi / 16) = f ω φ (-x - 3 * Real.pi / 16)) :
  (∀ x, f ω φ x = f ω φ (-x) → x = π / 16 ∨ x = -π / 4) :=
sorry

end graph_symmetry_l23_23352


namespace surface_integral_sphere_l23_23614

theorem surface_integral_sphere (R : ℝ) :
  ∫∫ σ, (λ (x y z : ℝ), x * (differential y * differential z) + y * (differential x * differential z) + z * (differential x * differential y)) 
    <: {p : ℝ × ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 + p.3 ^ 2 = R ^ 2}>
  = 4 * π * R ^ 3 := 
sorry

end surface_integral_sphere_l23_23614


namespace calculate_Delta_l23_23730

-- Define the Delta operation
def Delta (a b : ℚ) : ℚ := (a^2 + b^2) / (1 + a^2 * b^2)

-- Constants for the specific problem
def two := (2 : ℚ)
def three := (3 : ℚ)
def four := (4 : ℚ)

theorem calculate_Delta : Delta (Delta two three) four = 5945 / 4073 := by
  sorry

end calculate_Delta_l23_23730


namespace range_of_function_l23_23805

open Real

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∃ y, y = sin x - 2 * cos x + 32 / (125 * sin x * (1 - cos x)) ∧ y ≥ 2 / 5 :=
sorry

end range_of_function_l23_23805


namespace election_winner_votes_difference_l23_23126

theorem election_winner_votes_difference (V : ℝ) (h1 : 0.62 * V = 1054) : 0.24 * V = 408 :=
by
  sorry

end election_winner_votes_difference_l23_23126


namespace b_divides_a_l23_23937

-- Define the natural numbers N, a, and b
variable (N a b : ℕ)

-- Define the sets of red and blue numbers
variable (reds blues : Finset ℕ)

-- Set conditions based on the problem statement
variable (N_bound : ∀ x ∈ reds ∪ blues, N^3 ≤ x ∧ x ≤ N^3 + N)
variable (reds_size : reds.card = a)
variable (blues_size : blues.card = b)
variable (reds_blue_sum_div : blues.sum ∣ reds.sum)

-- The theorem that needs proof
theorem b_divides_a : b ∣ a :=
  sorry

end b_divides_a_l23_23937


namespace problem_1_problem_2_l23_23703

def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x + 1)

theorem problem_1 : 
  ∀ x : ℝ, f x < 4 → (-4/3 : ℝ) < x ∧ x < 4/3 :=
by
  sorry

theorem problem_2 (x₀ : ℝ) (t : ℝ) :
  f x₀ < real.log 2 (real.sqrt (t^2 - 1)) → t < -3 ∨ t > 3 :=
by
  sorry

end problem_1_problem_2_l23_23703


namespace choose_shooter_to_compete_l23_23892

-- Define the average scores and variances for the shooters
def avg_scores : List ℝ := [9.6, 8.9, 9.6, 9.6]
def variances : List ℝ := [1.4, 0.8, 2.3, 0.8]

-- Condition: Identify shooters with the highest average scores (9.6)
def highest_avg_shooters : List Nat := [0, 2, 3]  -- indices of shooters A, C, D

-- Condition: Variance values for shooters A, C, D are 1.4, 2.3, 0.8 respectively
def variances_highest_avg : List ℝ := [1.4, 2.3, 0.8]

-- Define selector function to pick shooter based on lowest variance
def select_shooter (highest_avg_shooters : List ℕ) (variances_highest_avg : List ℝ) : Nat :=
  highest_avg_shooters[variances_highest_avg.indexOf (variances_highest_avg.min!)]

-- Proof problem statement
theorem choose_shooter_to_compete :
  select_shooter highest_avg_shooters variances_highest_avg = 3 := by
  sorry

end choose_shooter_to_compete_l23_23892


namespace buses_more_than_vans_l23_23194

-- Definitions based on conditions
def vans : Float := 6.0
def buses : Float := 8.0
def people_per_van : Float := 6.0
def people_per_bus : Float := 18.0

-- Calculate total people in vans and buses
def total_people_vans : Float := vans * people_per_van
def total_people_buses : Float := buses * people_per_bus

-- Prove the difference
theorem buses_more_than_vans : total_people_buses - total_people_vans = 108.0 :=
by
  sorry

end buses_more_than_vans_l23_23194


namespace minimum_distance_l23_23811

theorem minimum_distance (z : ℂ) (h : |z - 2| + |z - 7 * I| = 10) : |z| = 1.4 :=
sorry

end minimum_distance_l23_23811


namespace original_number_of_boys_l23_23161

theorem original_number_of_boys (n : ℕ) (W : ℕ) 
  (h1 : W = n * 35) 
  (h2 : W + 135 = (n + 3) * 36) : 
  n = 27 := 
by 
  sorry

end original_number_of_boys_l23_23161


namespace Susie_earnings_l23_23469

theorem Susie_earnings :
  let price_per_slice := 3 in
  let slices_sold := 24 in
  let price_per_pizza := 15 in
  let pizzas_sold := 3 in
  let earnings_from_slices := price_per_slice * slices_sold in
  let earnings_from_pizzas := price_per_pizza * pizzas_sold in
  let total_earnings := earnings_from_slices + earnings_from_pizzas in
  total_earnings = 117 :=
by
  sorry

end Susie_earnings_l23_23469


namespace exists_subset_with_4th_power_product_l23_23669

def is_4th_power (n : ℕ) : Prop :=
  ∃ m : ℕ, m ^ 4 = n

theorem exists_subset_with_4th_power_product
  (M : Finset ℕ)
  (hM_size : M.card = 1985)
  (hM_condition : ∀ n ∈ M, ∀ p ∣ n, p ∈ {2, 3, 5, 7, 11, 13, 17, 19, 23}) :
  ∃ S ⊆ M, S.card = 4 ∧ is_4th_power (S.prod id) :=
by
  sorry

end exists_subset_with_4th_power_product_l23_23669


namespace counting_integers_between_multiples_l23_23990

theorem counting_integers_between_multiples :
  let smallest_perfect_square_multiple := 900 in
  let smallest_perfect_cube_multiple := 27000 in
  let num_integers := (smallest_perfect_cube_multiple / 30) - (smallest_perfect_square_multiple / 30) + 1 in
  smallest_perfect_square_multiple = 30 * 30 ∧ 
  smallest_perfect_cube_multiple = 900 * 30 ∧ 
  num_integers = 871 :=
by
  sorry

end counting_integers_between_multiples_l23_23990


namespace smallest_n_f_greater_21_l23_23424

-- Definition of the function f
def f (n : ℕ) : ℕ :=
  Nat.find (λ k, n ∣ Nat.factorial k)

-- Definition that n is a multiple of 21
def is_multiple_of_21 (n : ℕ) : Prop :=
  ∃ r : ℕ, n = 21 * r

-- The theorem we are proving
theorem smallest_n_f_greater_21 (n : ℕ) (h : is_multiple_of_21 n) : f(n) > 21 ↔ n = 483 :=
by {
  sorry
}

end smallest_n_f_greater_21_l23_23424


namespace circle_center_second_quadrant_tangent_lines_l23_23666

noncomputable def circle_equation (x y : ℝ) (D E : ℝ) : ℝ := 
  x^2 + y^2 + D * x + E * y + 3

theorem circle_center_second_quadrant (D E : ℝ) (x y : ℝ) :
  (circle_equation x y D E = 0) ∧ 
  (x + y - 1 = 0) ∧ 
  (⁤sqrt ((D^2 + E^2 - 12) / 4) = sqrt 2) ∧ 
  (D > 0) ∧ (E < 0) → 
  (circle_equation x y 2 (-4) = 0) := by
  sorry

noncomputable def line_equation (x y a: ℝ) : Prop :=
  x + y = a

theorem tangent_lines (a : ℝ) (x y : ℝ) :
  (line_equation x y a) ∧ 
  (a ≠ 0) ∧ 
  ((|(-1) + 2 - a| / sqrt 2) = sqrt 2) → 
  (line_equation x y (-1) ∨ line_equation x y 3) := by
  sorry

end circle_center_second_quadrant_tangent_lines_l23_23666


namespace find_abs_x_l23_23441

-- Given conditions
def A (x : ℝ) : ℝ := 3 + x
def B (x : ℝ) : ℝ := 3 - x
def distance (a b : ℝ) : ℝ := abs (a - b)

-- Problem statement: Prove |x| = 4 given the conditions
theorem find_abs_x (x : ℝ) (h : distance (A x) (B x) = 8) : abs x = 4 := 
  sorry

end find_abs_x_l23_23441


namespace si_perpendicular_zt_l23_23428

open EuclideanGeometry

theorem si_perpendicular_zt
  {A B C O I E F T Z S : Point}
  (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ A)
  (hCircumcenter : Circumcenter A B C O)
  (hIncenter : Incenter A B C I)
  (hE : OrthogonalProjection I (Segment A B) E)
  (hF : OrthogonalProjection I (Segment A C) F)
  (hT : LineThrough E I ∩ LineThrough O C T)
  (hZ : LineThrough F I ∩ LineThrough O B Z)
  (hS : IntersectionOfTangentsAtBc A B C S) :
  Perpendicular (LineThrough S I) (LineThrough Z T) := by
  sorry

end si_perpendicular_zt_l23_23428


namespace max_intersections_three_circles_one_line_l23_23528

theorem max_intersections_three_circles_one_line (c1 c2 c3 : Circle) (L : Line) :
  greatest_number_points_of_intersection c1 c2 c3 L = 12 :=
sorry

end max_intersections_three_circles_one_line_l23_23528


namespace sum_of_consecutive_integers_l23_23500

theorem sum_of_consecutive_integers (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
by
  sorry

end sum_of_consecutive_integers_l23_23500


namespace contrapositive_proposition_l23_23480

theorem contrapositive_proposition :
  (∀ x : ℝ, (x^2 < 4 → -2 < x ∧ x < 2)) ↔ (∀ x : ℝ, (x ≤ -2 ∨ x ≥ 2 → x^2 ≥ 4)) :=
by
  sorry

end contrapositive_proposition_l23_23480


namespace imaginary_part_l23_23430

noncomputable def z (z : ℂ) : Prop :=
  (1 + 2 * complex.I) / z = complex.I

theorem imaginary_part (z : ℂ) (h : (1 + 2 * complex.I) / z = complex.I) : complex.imaginaryPart z = -1 :=
by
  sorry

end imaginary_part_l23_23430


namespace probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l23_23764

theorem probability_exactly_k_gnomes_fall (n k : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in p * q^(n - k) = p * (1 - p)^(n - k) := 
sorry

theorem expected_number_of_gnomes_fall (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in 
  (∑ j in finset.range n, (1 - q^(j+1))) = n + 1 - (1 / p) + ((1 - p)^(n+1) / p) :=
sorry

end probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l23_23764


namespace part1_part2_part3_part3_expectation_l23_23450

/-- Conditions setup -/
noncomputable def gameCondition (Aacc Bacc : ℝ) :=
  (Aacc = 0.5) ∧ (Bacc = 0.6)

def scoreDist (X:ℤ) : ℝ :=
  if X = -1 then 0.3
  else if X = 0 then 0.5
  else if X = 1 then 0.2
  else 0

def tieProbability : ℝ := 0.2569

def roundDist (Y:ℤ) : ℝ :=
  if Y = 2 then 0.13
  else if Y = 3 then 0.13
  else if Y = 4 then 0.74
  else 0

def roundExpectation : ℝ := 3.61

/-- Proof Statements -/
theorem part1 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (X : ℤ), scoreDist X = if X = -1 then 0.3 else if X = 0 then 0.5 else if X = 1 then 0.2 else 0 :=
by sorry

theorem part2 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  tieProbability = 0.2569 :=
by sorry

theorem part3 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (Y : ℤ), roundDist Y = if Y = 2 then 0.13 else if Y = 3 then 0.13 else if Y = 4 then 0.74 else 0 :=
by sorry

theorem part3_expectation (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) :
  roundExpectation = 3.61 :=
by sorry

end part1_part2_part3_part3_expectation_l23_23450


namespace evaluate_expression_at_neg_one_l23_23629

theorem evaluate_expression_at_neg_one : 
  (\frac{4 + (-1) * (4 + (-1)) - 4^2}{-1 - 4 + (-1)^3} = \frac{5}{2}) :=
by
  sorry

end evaluate_expression_at_neg_one_l23_23629


namespace find_a6_l23_23311

-- Define an arithmetic progression.
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the necessary conditions given in the problem.
def conditions (a d : ℕ) : Prop :=
  (arithmetic_progression a d 1 + arithmetic_progression a d 2 + arithmetic_progression a d 3 = 168) ∧
  (arithmetic_progression a d 2 - arithmetic_progression a d 5 = 42)

-- State the theorem with the final value assertion.
theorem find_a6 (a d : ℕ) (h : conditions a (-14)) : 
  arithmetic_progression a (-14) 6 = 3 := 
sorry

end find_a6_l23_23311


namespace distance_lines_eq_two_l23_23483

def line1 := (3, -4, -2)  -- coefficients A, B, C1 for the first line
def line2 := (3, -4, 8)   -- coefficients A, B, C2 for the second line

def distance_between_lines (A B C1 C2 : ℤ) : ℝ :=
  (abs (C2 - C1 : ℤ)) / (real.sqrt (A^2 + B^2 : ℤ))

theorem distance_lines_eq_two :
  distance_between_lines 3 (-4) (-2) 8 = 2 := by
  sorry

end distance_lines_eq_two_l23_23483


namespace one_more_square_possible_l23_23281

def grid_size : ℕ := 29
def total_cells : ℕ := grid_size * grid_size
def number_of_squares_removed : ℕ := 99
def cells_per_square : ℕ := 4
def total_removed_cells : ℕ := number_of_squares_removed * cells_per_square
def remaining_cells : ℕ := total_cells - total_removed_cells

theorem one_more_square_possible :
  remaining_cells ≥ cells_per_square :=
sorry

end one_more_square_possible_l23_23281


namespace incorrect_proposition_l23_23624

theorem incorrect_proposition (p q : Prop) :
  ¬(¬(p ∧ q) → ¬p ∧ ¬q) := sorry

end incorrect_proposition_l23_23624


namespace rightmost_vertex_x_coordinate_l23_23118

-- Given the vertices lie on the graph y = e^x at four consecutive negative x-coordinates
-- and the area of the quadrilateral formed by these vertices is e^2 - e^(-2),
-- prove that the x-coordinate of the rightmost vertex is 2.

noncomputable def x_coordinates_of_vertices (n : ℤ) : List ℤ :=
  [n, n + 1, n + 2, n + 3]

def coordinates (n : ℤ) : List (ℤ × ℝ) :=
  x_coordinates_of_vertices n |>.map (λ x, (x, Real.exp x))

def area_of_quadrilateral (coords : List (ℤ × ℝ)) : ℝ := 
  (1 / 2) * 
  (|coords.ilast 3.1 + coords.head 4.0|)

theorem rightmost_vertex_x_coordinate :
  ∃ n : ℤ, area_of_quadrilateral (coordinates (-n)) = Real.exp 2 - Real.exp (-2) 
  ∧ (List.maximum (x_coordinates_of_vertices n)).get_or_else 0 = 2 := 
begin
  sorry 
end

end rightmost_vertex_x_coordinate_l23_23118


namespace abs_x_equals_4_l23_23444

-- Define the points A and B as per the conditions
def point_A (x : ℝ) : ℝ := 3 + x
def point_B (x : ℝ) : ℝ := 3 - x

-- Define the distance between points A and B
def distance (x : ℝ) : ℝ := abs ((point_A x) - (point_B x))

theorem abs_x_equals_4 (x : ℝ) (h : distance x = 8) : abs x = 4 :=
by
  sorry

end abs_x_equals_4_l23_23444


namespace mink_skins_per_coat_l23_23398

def initial_minks : ℕ := 30
def babies_per_mink : ℕ := 6
def activist_release_fraction : ℝ := 0.5
def coats_produced : ℕ := 7

theorem mink_skins_per_coat : 
  let total_minks := initial_minks + initial_minks * babies_per_mink in
  let remaining_minks := (total_minks : ℝ) * (1 - activist_release_fraction) in
  remaining_minks / (coats_produced : ℝ) = 15 := 
by
  sorry

end mink_skins_per_coat_l23_23398


namespace prob_k_gnomes_fall_exp_gnomes_falling_l23_23766

variables (n k : ℕ) (p : ℝ)
hypotheses 
  (hn : 0 < n)
  (hp : 0 < p) (hp1 : p < 1)
  (hk : 0 ≤ k) (hk1 : k ≤ n)

open ProbabilityTheory
  
def probability_k_gnomes_fall := 
  p * (1 - p) ^ (n - k)

def expected_gnomes_fall :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p

theorem prob_k_gnomes_fall (hprob : 0 < p ∧ p < 1) : 
  ∀ n k : ℕ, 0 ≤ k ∧ k ≤ n → probability_k_gnomes_fall n k p = p * (1 - p) ^ (n - k) :=
by sorry

theorem exp_gnomes_falling (hprob : 0 < p ∧ p < 1) : 
  ∀ n : ℕ, 0 < n → expected_gnomes_fall n p = n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p :=
by sorry

end prob_k_gnomes_fall_exp_gnomes_falling_l23_23766


namespace carol_tom_combined_weight_mildred_heavier_than_carol_tom_combined_l23_23820

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9
def tom_weight : ℕ := 20

theorem carol_tom_combined_weight :
  carol_weight + tom_weight = 29 := by
  sorry

theorem mildred_heavier_than_carol_tom_combined :
  mildred_weight - (carol_weight + tom_weight) = 30 := by
  sorry

end carol_tom_combined_weight_mildred_heavier_than_carol_tom_combined_l23_23820


namespace acute_triangle_angle_A_range_of_bc_l23_23771

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}
variable (Δ : ∃ (A B C : ℝ), a = sqrt 2 ∧ ∀ (a b c A B C : ℝ), 
  (a = sqrt 2) ∧ (b = b) ∧ (c = c) ∧ 
  (sin A * cos A / cos (A + C) = a * c / (b^2 - a^2 - c^2)))

-- Problem statement
theorem acute_triangle_angle_A (h : Δ) : A = π / 4 :=
sorry

theorem range_of_bc (h : Δ) : 0 < b * c ∧ b * c ≤ 2 + sqrt 2 :=
sorry

end acute_triangle_angle_A_range_of_bc_l23_23771


namespace equilateral_triangle_indefinite_construction_l23_23886

structure Triangle :=
  (a b c : ℝ)
  (side_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a)

noncomputable def semi_perimeter (t : Triangle) : ℝ :=
  (t.a + t.b + t.c) / 2

theorem equilateral_triangle_indefinite_construction
  (t : Triangle)
  (new_sides : Triangle) :
  new_sides.a = semi_perimeter t - t.a →
  new_sides.b = semi_perimeter t - t.b →
  new_sides.c = semi_perimeter t - t.c →
  (∀ k: ℕ, ∃ t_k : Triangle,
    t_k.a = (semi_perimeter t) / 2 ^ k - t.a / 2 ^ k ∧
    t_k.b = (semi_perimeter t) / 2 ^ k - t.b / 2 ^ k ∧
    t_k.c = (semi_perimeter t) / 2 ^ k - t.c / 2 ^ k) →
  t.a = t.b ∧ t.b = t.c :=
begin
  sorry
end

end equilateral_triangle_indefinite_construction_l23_23886


namespace find_sixth_term_l23_23303

open Nat

-- Given conditions
def arithmetic_progression (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ (n : ℕ), a (n + 1) = a n + d

def sum_of_first_three_terms (a : ℕ → ℤ) : Prop :=
  a 1 + a 2 + a 3 = 168

def second_minus_fifth (a : ℕ → ℤ) : Prop :=
  a 2 - a 5 = 42

-- Prove question == answer given conditions
theorem find_sixth_term :
  ∀ (a : ℕ → ℤ), arithmetic_progression a → sum_of_first_three_terms a → second_minus_fifth a → a 6 = 0 :=
by
  sorry

end find_sixth_term_l23_23303


namespace exponential_comparison_l23_23659

theorem exponential_comparison (a b c : ℝ) (h₁ : a = 0.5^((1:ℝ)/2))
                                          (h₂ : b = 0.5^((1:ℝ)/3))
                                          (h₃ : c = 0.5^((1:ℝ)/4)) : 
  a < b ∧ b < c := by
  sorry

end exponential_comparison_l23_23659


namespace marvin_solved_yesterday_l23_23098

variables (M : ℕ)

def Marvin_yesterday := M
def Marvin_today := 3 * M
def Arvin_yesterday := 2 * M
def Arvin_today := 6 * M
def total_problems := Marvin_yesterday + Marvin_today + Arvin_yesterday + Arvin_today

theorem marvin_solved_yesterday :
  total_problems M = 480 → M = 40 :=
sorry

end marvin_solved_yesterday_l23_23098


namespace men_left_bus_l23_23574

theorem men_left_bus (M W : ℕ) (initial_passengers : M + W = 72) 
  (women_half_men : W = M / 2) 
  (equal_men_women_after_changes : ∃ men_left : ℕ, ∀ W_new, W_new = W + 8 → M - men_left = W_new → M - men_left = 32) :
  ∃ men_left : ℕ, men_left = 16 :=
  sorry

end men_left_bus_l23_23574


namespace greatest_possible_value_of_y_l23_23844

-- Definitions according to problem conditions
variables {x y : ℤ}

-- The theorem statement to prove
theorem greatest_possible_value_of_y (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
sorry

end greatest_possible_value_of_y_l23_23844


namespace range_b_values_l23_23353

theorem range_b_values (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : ∀ x, f x = Real.exp x - 1) 
  (hg : ∀ x, g x = -x^2 + 4*x - 3) 
  (h : f a = g b) : 
  b ∈ Set.univ :=
by sorry

end range_b_values_l23_23353


namespace derek_lowest_score_l23_23622

theorem derek_lowest_score:
  ∀ (score1 score2 max_points target_avg min_score tests_needed last_test1 last_test2 : ℕ),
  score1 = 85 →
  score2 = 78 →
  max_points = 100 →
  target_avg = 84 →
  min_score = 60 →
  tests_needed = 4 →
  last_test1 >= min_score →
  last_test2 >= min_score →
  last_test1 <= max_points →
  last_test2 <= max_points →
  (score1 + score2 + last_test1 + last_test2) = target_avg * tests_needed →
  min last_test1 last_test2 = 73 :=
by
  sorry

end derek_lowest_score_l23_23622


namespace total_flowers_l23_23511

-- Define the constants and hypotheses
def num_pots : ℕ := 350

-- The ratio of flowers to total items in each pot is 3:5
axiom flower_stick_ratio (x y T : ℝ) (hT : T = x + y) (hx : x = (3/5) * T) : T = x + y

-- Prove the total number of flowers in all pots is (3/5) of the total number of items
theorem total_flowers (x y T : ℝ) (hT : T = x + y) (hx : x = (3/5) * T) :
  let total_items := (num_pots * T) in
  let total_flowers := (num_pots * x) in
  total_flowers = (3/5) * total_items :=
by
  sorry

end total_flowers_l23_23511


namespace arithmetic_mean_of_fractions_l23_23612

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  (a + b) / 2 = 11 / 16 :=
by 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  show (a + b) / 2 = 11 / 16
  sorry

end arithmetic_mean_of_fractions_l23_23612


namespace max_area_triangle_PAB_l23_23236

open Real

noncomputable def ellipse_eq (x y : ℝ) : Prop := 
  (x^2 / 16) + (y^2 / 9) = 1

def point_A : (ℝ × ℝ) := (4, 0)
def point_B : (ℝ × ℝ) := (0, 3)

theorem max_area_triangle_PAB (P : ℝ × ℝ) (hP : ellipse_eq P.1 P.2) : 
  ∃ S, S = 6 * (sqrt 2 + 1) := 
sorry

end max_area_triangle_PAB_l23_23236


namespace distinct_equilateral_triangles_in_dodecagon_l23_23301

theorem distinct_equilateral_triangles_in_dodecagon (A : ℕ → ℝ × ℝ) (regular_dodecagon : ∀ i j, A ((i - 1) % 12 + 1) ≠ A ((j - 1) % 12 + 1)) :
  ∃ n : ℕ, n = 128 ∧ 
  ∀ T : set (ℝ × ℝ), (∀ p q r : ℕ, (A p ∈ T ∧ A q ∈ T ∧ A r ∈ T) → p ≠ q ∧ q ≠ r ∧ r ≠ p → equilateral T) ↔  (card T = n) :=
sorry

end distinct_equilateral_triangles_in_dodecagon_l23_23301


namespace convert_base_five_to_ten_l23_23904

theorem convert_base_five_to_ten : ∃ n : ℕ, n = 38 ∧ (1 * 5^2 + 2 * 5^1 + 3 * 5^0 = n) :=
by
  sorry

end convert_base_five_to_ten_l23_23904


namespace prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l23_23757

variables (n k : ℕ) (p : ℝ)
variables (h_pos : 0 < p) (h_lt_one : p < 1)

-- Probability that exactly k gnomes fall
theorem prob_exactly_k_gnomes_fall (h_k_le_n : k ≤ n) :
  prob_speed (exactly_k_gnomes_fall n k p) = p * (1 - p)^(n - k) := sorry

-- Expected number of fallen gnomes
theorem expected_fallen_gnomes : 
  expected_falls n p = n + 1 - 1/p + (1 - p)^(n + 1)/p := sorry

end prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l23_23757


namespace two_dice_probability_l23_23592

def diameter_is_sum_of_dice (d : ℕ) : Prop :=
  d ∈ (2 : Finset ℕ).erase 12

def circle_inequality (d : ℕ) : Prop :=
  d * (2 - d) > 0

theorem two_dice_probability : 
  (∃ d, diameter_is_sum_of_dice d ∧ circle_inequality d) → 
  ∑ d in (2 : Finset ℕ).erase 12, if d * (2 - d) > 0 then (1 / 36 : ℚ) else 0 = (1 / 36 : ℚ) :=
by
  sorry

end two_dice_probability_l23_23592


namespace max_b_sqrt_a_max_sqrt_a_plus_b_l23_23658

theorem max_b_sqrt_a (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b^2 = 1) :
  b * sqrt a ≤ 1 / 2 := sorry

theorem max_sqrt_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b^2 = 1) :
  sqrt a + b ≤ sqrt 2 := sorry

end max_b_sqrt_a_max_sqrt_a_plus_b_l23_23658


namespace tulips_after_addition_l23_23876

theorem tulips_after_addition
  (T S : ℕ)
  (ratio : ℕ → ℕ → Prop := λ t s, t * 4 = s * 3)
  (initial_sunflowers : S = 36)
  (added_sunflowers : 12) :
  (∃ T', ratio T' (S + added_sunflowers) ∧ T' = 36) :=
by
  sorry

end tulips_after_addition_l23_23876


namespace five_congruent_subtriangles_possible_l23_23577

-- Given definitions from conditions
def triangle := sorry  -- placeholder for the general definition of a triangle
def smaller_triangles (t : triangle) : list triangle := sorry  -- list of smaller triangles obtained from partitioning t

-- Lean 4 statement to prove
theorem five_congruent_subtriangles_possible (t : triangle) :
  ∃ (ts : list triangle), length ts = 5 ∧ (∀ x ∈ ts, congruent x x) :=
sorry

end five_congruent_subtriangles_possible_l23_23577


namespace find_Q_x_l23_23008

noncomputable def Q : ℝ → ℝ := sorry

variables (Q0 Q1 Q2 : ℝ)

axiom Q_def : ∀ x, Q x = Q0 + Q1 * x + Q2 * x^2
axiom Q_minus_2 : Q (-2) = -3

theorem find_Q_x : ∀ x, Q x = (3 / 5) * (1 + x - x^2) :=
by 
  -- Proof to be completed
  sorry

end find_Q_x_l23_23008


namespace subcommittee_ways_l23_23953

theorem subcommittee_ways:
  let R := 10 in
  let D := 8 in
  let choose_r := Nat.choose R 4 in
  let choose_d := Nat.choose D 3 in
  choose_r * choose_d = 11760 :=
by
  sorry

end subcommittee_ways_l23_23953


namespace larger_number_of_hcf_and_lcm_factors_l23_23929

theorem larger_number_of_hcf_and_lcm_factors :
  ∃ (a b : ℕ), (∀ d, d ∣ a ∧ d ∣ b → d ≤ 20) ∧ (∃ x y, x * y * 20 = a * b ∧ x * 20 = a ∧ y * 20 = b ∧ x > y ∧ x = 15 ∧ y = 11) → max a b = 300 :=
by sorry

end larger_number_of_hcf_and_lcm_factors_l23_23929


namespace A_false_B_true_C_true_D_true_l23_23151

theorem A_false :
  ¬ ∃ x, ∀ y = (x^2 + 1) / x, y = 2 :=
by
  sorry

theorem B_true (x : ℝ) (h : x > 1) :
  (∀ y, y = 2*x + 4 / (x - 1) - 1 → y ≥ 4 * real.sqrt 2 + 1) :=
by
  sorry

theorem C_true (x y : ℝ) (h : x + 2 * y = 3 * x * y) (hx : 0 < x) (hy : 0 < y) :
  (2 * x + y ≥ 3) :=
by
  sorry

theorem D_true (x y : ℝ) (h : 9 * x^2 + y^2 + x * y = 1) :
  ∃ c, c = (3 * x + y) ∧ c ≤ (2 * real.sqrt 21 / 7) :=
by
  sorry

end A_false_B_true_C_true_D_true_l23_23151


namespace zero_in_interval_tangent_line_translate_graph_domain_function_dot_product_condition_l23_23944

noncomputable def f (x : ℝ) := - (1 / x) + log x

theorem zero_in_interval (h1 : f 2 * f 3 < 0) : ∃ x, 2 < x ∧ x < 3 ∧ f x = 0 := by
  apply IntermediateValueTheorem
  sorry

def curve (x : ℝ) := 4 * x - x ^ 3 

theorem tangent_line (tangent_eq : ∀ (x y : ℝ), y = curve x → y = x - 2) : ∀ x, curve (-1) = -3 ∧ deriv curve -1 = -1 := by
  sorry

def translation_vector := (1, -1 : ℝ × ℝ)

theorem translate_graph (h_trans : ∀ x, 2^(x+translation_vector.1) + translation_vector.2 = 2^(x+1)) : false := by
  sorry

noncomputable def log_half (x : ℝ) := logb (1/2 : ℝ) (x^2 - 1)

noncomputable def sqrt_log_half (x : ℝ) := sqrt (log_half x)

theorem domain_function (h_dom : ∀ x, sqrt_log_half x = sqrt (logb (1/2) (x^2 - 1))) : 
  ∀ x, (x ∈ set.Icc (-sqrt 2) (-1) ∨ x ∈ set.Icc 1 (sqrt 2)) := by
  sorry

theorem dot_product_condition (a b : ℝ × ℝ) (h_dot : a.1 * b.1 + a.2 * b.2 > 0) : ∃ θ, θ < π / 2 := by
  sorry

end zero_in_interval_tangent_line_translate_graph_domain_function_dot_product_condition_l23_23944


namespace players_per_group_l23_23104

-- Definitions for given conditions
def num_new_players : Nat := 48
def num_returning_players : Nat := 6
def num_groups : Nat := 9

-- Proof that the number of players in each group is 6
theorem players_per_group :
  let total_players := num_new_players + num_returning_players
  total_players / num_groups = 6 := by
  sorry

end players_per_group_l23_23104


namespace purely_imaginary_complex_number_l23_23378

theorem purely_imaginary_complex_number (x : ℝ) :
  (x^2 - 1 = 0) → (∃ x : ℝ, x = 1 ∨ x = -1) :=
by
  intros h
  use x
  split
  { have h1 : x = 1 := sorry
    exact h1 }
  { have h2 : x = -1 := sorry
    exact h2 }

end purely_imaginary_complex_number_l23_23378


namespace days_from_thursday_l23_23522

theorem days_from_thursday (n : ℕ) (h : n = 53) : 
  (n % 7 = 4) ∧ (n % 7 = 4 → "Thursday" + 4 days = "Monday") :=
by 
  have h1 : n % 7 = 4 := by sorry
  have h2 : "Thursday" + 4 days = "Monday" := by sorry
  exact ⟨h1, h2 h1⟩

end days_from_thursday_l23_23522


namespace conjugate_of_z_l23_23739

noncomputable def z : ℂ := (i ^ 2018) / (1 - i) ^ 2

theorem conjugate_of_z :
  conj z = (1/2 : ℂ) * i :=
by
  sorry

end conjugate_of_z_l23_23739


namespace domain_f_l23_23087

def f (x : ℝ) : ℝ := 1 / (Real.sqrt (1 - 2 * x))

theorem domain_f : ∀ x : ℝ, f x ∈ ℝ → x < 1 / 2 := by
  sorry

end domain_f_l23_23087


namespace race_course_length_l23_23575

theorem race_course_length (v : ℝ) (d : ℝ) (h1 : d = 7 * (d - 120)) : d = 140 :=
sorry

end race_course_length_l23_23575


namespace range_of_function_l23_23802

theorem range_of_function :
  ∀ x : ℝ,
  (0 < x ∧ x < (π / 2)) →
  ∃ y : ℝ, 
  y = (sin x - 2 * cos x + (32 / (125 * sin x * (1 - cos x)))) ∧ y ≥ 2 / 5 :=
sorry

end range_of_function_l23_23802


namespace condition_for_fg_eq_gf_l23_23376

def f (a b c : ℝ) : ℝ → ℝ := λ x, a * x^2 + b * x + c
def g (d e f : ℝ) : ℝ → ℝ := λ x, d * x^2 + e * x + f

theorem condition_for_fg_eq_gf (a b c d e f : ℝ) (hb : b = 0) (he : e = 0)
  (hf : ℝ → ℝ := f a b c) (hg : ℝ → ℝ := g d e f) :
  (hf (hg x) = hg (hf x)) ↔ (d = a^2 + a - 1 ∨ c = 0) :=
sorry

end condition_for_fg_eq_gf_l23_23376


namespace ellipse_C2_equation_and_dot_product_range_l23_23341

def is_ellipse (a b larger: ℕ) (x y: ℝ) : Prop :=
  (a > b ∧ larger >= a) ∧ (x^2) / (a^2) + (y^2) / (b^2) = 1

def eccentricity (a b : ℕ) : ℝ :=
  real.sqrt (1 - (b^2 / a^2))

theorem ellipse_C2_equation_and_dot_product_range 
  (x y : ℝ) 
  (x_A : ℕ) 
  (a b c : ℕ) 
  (e : ℝ := eccentricity 2 4) 
  (A : ℕ × ℝ := (2, 0))
  (E F : ℕ → ℝ := lambda k, (1, k * (sqrt 3 / 2)))
  (M : ℕ → ℝ := lambda k, (3, -(sqrt 3) / 2))
  (N : ℕ → ℝ := lambda k, (3, (sqrt 3) / 2))
  :
  is_ellipse 2 1 2 x y → ∃ (x_A : ℕ), (x_A = 2) ∧ (∃ (coord_range : ℝ → Prop),
  (coord_range (λ k, (3 - E k, ((A.2 (3 - E k)) / (E k - x_A))))) ∧ (∀ (k : ℝ), 1 < k ∧ k < (5/4)))
:= by 
  intros h
  sorry

end ellipse_C2_equation_and_dot_product_range_l23_23341


namespace day_53_days_from_thursday_is_monday_l23_23524

def day_of_week : Type := {n : ℤ // n % 7 = n}

def Thursday : day_of_week := ⟨4, by norm_num⟩
def Monday : day_of_week := ⟨1, by norm_num⟩

theorem day_53_days_from_thursday_is_monday : 
  (⟨(4 + 53) % 7, by norm_num⟩ : day_of_week) = Monday := 
by 
  sorry

end day_53_days_from_thursday_is_monday_l23_23524


namespace nonnegative_fraction_iff_interval_l23_23652

theorem nonnegative_fraction_iff_interval (x : ℝ) : 
  0 ≤ x ∧ x < 3 ↔ 0 ≤ (x^2 - 12 * x^3 + 36 * x^4) / (9 - x^3) := by
  sorry

end nonnegative_fraction_iff_interval_l23_23652


namespace find_a6_l23_23312

-- Define an arithmetic progression.
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the necessary conditions given in the problem.
def conditions (a d : ℕ) : Prop :=
  (arithmetic_progression a d 1 + arithmetic_progression a d 2 + arithmetic_progression a d 3 = 168) ∧
  (arithmetic_progression a d 2 - arithmetic_progression a d 5 = 42)

-- State the theorem with the final value assertion.
theorem find_a6 (a d : ℕ) (h : conditions a (-14)) : 
  arithmetic_progression a (-14) 6 = 3 := 
sorry

end find_a6_l23_23312


namespace new_student_weight_l23_23850

theorem new_student_weight (W : ℕ) : 
  let W_old := 96
  let avg_decrease := 8
  let num_students := 4
  W_new = W_old + (num_students * avg_decrease) :=
  W_new = 160 :=
by
  sorry

end new_student_weight_l23_23850


namespace number_of_boxes_in_case_l23_23022

theorem number_of_boxes_in_case (boxes : ℕ) (eggs_per_box : ℕ) (total_eggs : ℕ) (cases : ℕ) :
  boxes = 3 ∧ eggs_per_box = 7 ∧ total_eggs = 21 → cases = 3 :=
by
  assume h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  sorry

end number_of_boxes_in_case_l23_23022


namespace cameron_list_count_l23_23998

theorem cameron_list_count :
  let numbers := {n : ℕ | 30 ≤ n ∧ n ≤ 900}
  in set.card numbers = 871 :=
sorry -- proof is omitted

end cameron_list_count_l23_23998


namespace area_enclosed_by_graph_l23_23902

theorem area_enclosed_by_graph : 
  ∃ A : ℝ, (∀ x y : ℝ, |x| + |3 * y| = 9 ↔ (x = 9 ∨ x = -9 ∨ y = 3 ∨ y = -3)) → A = 54 :=
by
  sorry

end area_enclosed_by_graph_l23_23902


namespace sqrt_of_36_l23_23883

theorem sqrt_of_36 :
  {x : ℝ // x^2 = 36} = {6, -6} :=
sorry

end sqrt_of_36_l23_23883


namespace domain_of_function_l23_23086

noncomputable def domain_f (x : ℝ) : Prop :=
  -x^2 + 2 * x + 3 > 0 ∧ 1 - x > 0 ∧ x ≠ 0

theorem domain_of_function :
  {x : ℝ | domain_f x} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end domain_of_function_l23_23086


namespace nat_perfect_square_l23_23040

theorem nat_perfect_square (a b : ℕ) (h : ∃ k : ℕ, a^2 + b^2 + a = k * a * b) : ∃ m : ℕ, a = m * m := by
  sorry

end nat_perfect_square_l23_23040


namespace percentage_reduction_in_price_of_oil_l23_23195

theorem percentage_reduction_in_price_of_oil :
  ∀ (P : ℝ),
  (∀ (P : ℝ), 0 < P → (800 / P + 5 = 800 / 16)) →
  ∃ ε > 0, abs (((P - 16) / P) * 100 - 10.01) < ε :=
begin
  intros P h,
  have h₁ : P = 800 / 45 := sorry,
  have h₂ : 0 < P := sorry,
  rw h₁ at *,
  use 0.1,
  split,
  { norm_num },
  { apply abs_sub_lt,
    { rw h₁,
      norm_num },
    { rw h₁,
      norm_num } }
end

end percentage_reduction_in_price_of_oil_l23_23195


namespace expected_final_set_size_l23_23816

noncomputable def final_expected_set_size : ℚ :=
  let n := 8
  let initial_size := 255
  let steps := initial_size - 1
  n * (2^7 / initial_size)

theorem expected_final_set_size :
  final_expected_set_size = 1024 / 255 :=
by
  sorry

end expected_final_set_size_l23_23816


namespace first_player_wins_l23_23823

-- Define the game setup and the conditions
structure game_state :=
  (xs: ℤ)
  (ys: ℤ)
  (condition: xs ^ 2 + ys ^ 2 ≤ 1010)

def can_first_player_win (s : set game_state) : Prop := sorry

-- Main theorem stating that the first player can guarantee a win
theorem first_player_wins (s : set game_state):
  (∀ P1 P2 ∈ s, ((P1.xs - P2.xs)^2 + (P1.ys - P2.ys)^2) > 0 →
     ¬ (P1 = (-P2))) →
  can_first_player_win s :=
sorry

end first_player_wins_l23_23823


namespace desired_value_l23_23796

noncomputable def find_sum (a b c : ℝ) (p q r : ℝ) : ℝ :=
  a / p + b / q + c / r

theorem desired_value (a b c : ℝ) (h1 : p = a / 2) (h2 : q = b / 2) (h3 : r = c / 2) :
  find_sum a b c p q r = 6 :=
by
  sorry

end desired_value_l23_23796


namespace counting_integers_between_multiples_l23_23989

theorem counting_integers_between_multiples :
  let smallest_perfect_square_multiple := 900 in
  let smallest_perfect_cube_multiple := 27000 in
  let num_integers := (smallest_perfect_cube_multiple / 30) - (smallest_perfect_square_multiple / 30) + 1 in
  smallest_perfect_square_multiple = 30 * 30 ∧ 
  smallest_perfect_cube_multiple = 900 * 30 ∧ 
  num_integers = 871 :=
by
  sorry

end counting_integers_between_multiples_l23_23989


namespace angle_at_2_15_l23_23217

def angle_hour_hand (hours minutes : ℕ) : ℝ := (hours % 12) * 30 + minutes * 0.5

def angle_minute_hand (minutes : ℕ) : ℝ := minutes * 6

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  let ha := angle_hour_hand hours minutes
  let ma := angle_minute_hand minutes
  abs (ma - ha)

theorem angle_at_2_15 : angle_between_hands 2 15 = 22.5 :=
by
  -- Using given definitions and conditions
  sorry

end angle_at_2_15_l23_23217


namespace max_nP_for_20x20_grid_l23_23665

theorem max_nP_for_20x20_grid : 
  ∀ (P : ℕ × ℕ → Prop), 
    (∀ i j, P i j → (1 ≤ i ∧ i ≤ 20 ∧ 1 ≤ j ∧ j ≤ 20)) → -- valid grid coordinates
    ∀ (R : list (ℕ × ℕ × ℕ × ℕ)), -- list of rectangles R represented as (x1, y1, x2, y2)
    (∀ (r : ℕ × ℕ × ℕ × ℕ), r ∈ R → (1 ≤ r.1 ∧ r.1 ≤ 20 ∧ 1 ≤ r.2 ∧ r.2 ≤ 20 ∧
                                      1 ≤ r.3 ∧ r.3 ≤ 20 ∧ 1 ≤ r.4 ∧ r.4 ≤ 20)) → -- rectangle boundaries in grid
    (∀ (r : ℕ × ℕ × ℕ × ℕ), r ∈ R → 
      (∃ a b, (P (r.1, r.2) = a ∧ P (r.3, r.4) = b ∧ a + b ≤ 2))) → -- no rectangle contains more than two black cells
    ∀ (n : ℕ), -- n is the number of rectangles containing at most one black cell
      (n = min_list (R.map (λ r, if (∃ a, P (r.1, r.2) = a ∧ a ≤ 1 ∧ ∃ b, P (r.3, r.4) = b ∧ b ≤ 1) then 1 else 0))) →
    n ≤ 20 := 
begin
  sorry
end

end max_nP_for_20x20_grid_l23_23665


namespace inequality_proof_l23_23684

theorem inequality_proof 
  (a b n : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : n > 0) 
  (h4 : ∃ k : ℕ, n.factorial = k * (a.factorial * b.factorial)) : 
  a + b ≤ n + 1 + 2 * (log 2 n) :=
by
  sorry

end inequality_proof_l23_23684


namespace simplify_sqrt_mul_cubert_l23_23466

theorem simplify_sqrt_mul_cubert:
  sqrt 18 * cbrt 24 = 6 * 2^(1/2 : ℝ) * 3^(1/3 : ℝ) :=
sorry

end simplify_sqrt_mul_cubert_l23_23466


namespace Danielle_has_6_rooms_l23_23717

/-- Heidi's apartment has 3 times as many rooms as Danielle's apartment. --/
def H (d : ℕ) : ℕ := 3 * d

/-- Grant's apartment has 1/9 as many rooms as Heidi's apartment. --/
def G (h : ℕ) : ℕ := h / 9

/-- Jane's apartment has 3/4 as many rooms as Grant's apartment. --/
def J (g : ℕ) : ℕ := 3 * g / 4

/-- Grant's apartment has 2 rooms. --/
axiom G_rooms : ∀ h : ℕ, G(h) = 2

theorem Danielle_has_6_rooms : ∀ d : ℕ, 3 * d = 18 → d = 6 :=
by
  intros d h_eq
  calc
    d = 18 / 3 := by rw [←h_eq]; ring
    _ = 6 := rfl

end Danielle_has_6_rooms_l23_23717


namespace product_of_real_parts_of_solutions_l23_23088

-- Define the given equation as a condition
def given_equation (x : ℂ) : Prop := x^2 + 2 * x = -3 + 4 * complex.I

-- Define the statement to prove that the product of the real parts of solutions is approximately 0.3077
noncomputable def product_real_parts (x : ℂ) (y : ℂ) : ℂ 
  := (complex.re x) * (complex.re y)

theorem product_of_real_parts_of_solutions :
  ∃ x y : ℂ, given_equation x ∧ given_equation y ∧ x ≠ y 
       ∧ abs (product_real_parts x y - 0.3077) < 0.0001 :=
by
  sorry

end product_of_real_parts_of_solutions_l23_23088


namespace smallest_norm_v_l23_23414

-- Given definitions and conditions
variable (v : ℝ × ℝ)
def v_add_vector_norm_eq_10 := ∥⟨v.1 + 4, v.2 + 2⟩∥ = 10

-- The proof statement we need to prove
theorem smallest_norm_v (h : v_add_vector_norm_eq_10 v) : 
  ∥v∥ = 10 - 2 * Real.sqrt 5 :=
sorry

end smallest_norm_v_l23_23414


namespace geometric_sum_thm_l23_23884

variable (S : ℕ → ℝ)

theorem geometric_sum_thm (h1 : S n = 48) (h2 : S (2 * n) = 60) : S (3 * n) = 63 :=
sorry

end geometric_sum_thm_l23_23884


namespace product_of_ab_l23_23371

theorem product_of_ab (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 13) : a * b = -6 :=
by
  sorry

end product_of_ab_l23_23371


namespace combined_tax_rate_correct_l23_23821

variable (M : ℝ)

def income_mindy  : ℝ := 4 * M
def income_bickley : ℝ := 2 * M
def income_exidor  : ℝ := M / 2

def tax_mork : ℝ := 0.45 * M
def tax_mindy  : ℝ := 0.20 * income_mindy M
def tax_bickley : ℝ := 0.25 * income_bickley M
def tax_exidor  : ℝ := 0.30 * income_exidor M

def total_tax : ℝ :=
  tax_mork M + tax_mindy M + tax_bickley M + tax_exidor M

def total_income : ℝ :=
  M + income_mindy M + income_bickley M + income_exidor M

def combined_tax_rate : ℝ :=
  total_tax M / total_income M

theorem combined_tax_rate_correct :
  combined_tax_rate M = 0.2533 :=
sorry

end combined_tax_rate_correct_l23_23821


namespace sum_of_first_n_natural_numbers_l23_23294

-- Define the sum of first n natural numbers
def S : ℕ → ℕ 
| 0       := 0
| (n + 1) := S n + (n + 1)

-- The theorem to be proved: for any natural number n, S(n) = n * (n + 1) / 2
theorem sum_of_first_n_natural_numbers (n : ℕ) : S n = n * (n + 1) / 2 := by
  sorry

end sum_of_first_n_natural_numbers_l23_23294


namespace BO_OE_ratio_correct_l23_23296

-- Definitions from the conditions
variable {A B C D O E : Point}
variable (ABCD : parallelogram A B C D) (angle_B : ∠B = 60)
variable (O_circumcenter : circumcenter O A B C)
variable (E_on_ext_angle_bisector : E = point_of_intersection_of_BO_with_exterior_angle_bisector_D_line O B D)
variable (BO_OE_ratio : ratio (length (segment B O)) (length (segment O E)) = 1 / 2)

-- The theorem to prove
theorem BO_OE_ratio_correct :
  ∀ (A B C D O E : Point) (ABCD : parallelogram A B C D) (angle_B : ∠B = 60)
  (O_circumcenter : circumcenter O A B C)
  (E_on_ext_angle_bisector : E = point_of_intersection_of_BO_with_exterior_angle_bisector_D_line O B D),
  ratio (length (segment B O)) (length (segment O E)) = 1 / 2 := by
  sorry

end BO_OE_ratio_correct_l23_23296


namespace length_AB_l23_23355

noncomputable theory

-- Define the parabola and conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : (ℝ × ℝ) := (1, 0)
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k * (x - focus.1)
def area_of_triangle (A B : ℝ × ℝ) : ℝ := 0.5 * (A.1 * B.2 - A.2 * B.1) -- Using Shoelace formula for area

-- Theorem to be proven
theorem length_AB (A B : ℝ × ℝ) (k : ℝ) :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  line_through_focus k A.1 A.2 ∧ line_through_focus k B.1 B.2 ∧
  area_of_triangle A B = √6 →
  abs (A.1 - B.1) = 6 :=
by
  sorry

end length_AB_l23_23355


namespace center_of_mass_eq_center_of_sphere_l23_23830

-- Definitions for the regular polyhedron and its vertices
variables {n : ℕ} (P : Fin n → ℝ^3) [regular_polyhedron : regular_polyhedron P]

-- Center of the inscribed (circumscribed) sphere at point O
variables (O : ℝ^3)

-- Vectors from the center O to each vertex
def v (i : Fin n) : ℝ^3 := P i - O

-- Proof statement
theorem center_of_mass_eq_center_of_sphere : 
  ∑ i : Fin n, v O P i = 0 := 
by 
  sorry

end center_of_mass_eq_center_of_sphere_l23_23830


namespace task_assignments_count_l23_23898

theorem task_assignments_count (S : Finset (Fin 5)) :
  ∃ (assignments : Fin 5 → Fin 3),  
    (∀ t, assignments t ≠ t) ∧ 
    (∀ v, ∃ t, assignments t = v) ∧ 
    (∀ t, (t = 4 → assignments t = 1)) ∧ 
    S.card = 60 :=
by sorry

end task_assignments_count_l23_23898


namespace count_squared_numbers_between_10_and_100_l23_23201

def is_squared_number (n : ℕ) : Prop :=
  let a := n / 10 in
  let b := n % 10 in
  let reversed := b * 10 + a in
  (n + reversed) % 11 == 0 ∧ ∃ k, (n + reversed) = k * k

theorem count_squared_numbers_between_10_and_100 : 
  (Finset.filter (λ n, is_squared_number n) (Finset.range 100)).filter (λ n, n >= 10) = 8 :=
by sorry

end count_squared_numbers_between_10_and_100_l23_23201


namespace right_triangle_of_complex_numbers_l23_23940

theorem right_triangle_of_complex_numbers (z1 z2 : ℂ) (A B O : ℂ)
  (hA : A = z1) (hB : B = z2) (hO : O = 0) 
  (h : |z1 + z2| = |z1 - z2|) : 
  ∠ A O B = π / 2 :=
by
  sorry

end right_triangle_of_complex_numbers_l23_23940


namespace cone_volume_l23_23585

theorem cone_volume {r_sector : ℝ} (h_sector : r_sector = 6) (angle_sector : ℝ) (h_angle : angle_sector = 5/6) :
  let circumference := 2 * π * r_sector,
      arc_length := angle_sector * circumference,
      r_cone_base := arc_length / (2 * π),
      slant_height := r_sector,
      h_cone := sqrt (r_sector^2 - r_cone_base^2) in
  (1/3) * π * (r_cone_base)^2 * h_cone = (25 / 3) * π * sqrt (11) := 
by 
  sorry

end cone_volume_l23_23585


namespace smallest_pos_period_f_monotonic_increase_intervals_calculate_g_at_pi_over_6_l23_23349

-- Define the original function f(x)
def f (x : ℝ) : ℝ := 2 * sqrt 3 * (sin x) ^ 2 + sin (2 * x)

-- Define g(x) after transformations
def g (x : ℝ) : ℝ := 2 * sin x + sqrt 3

-- Prove each part of the solution
theorem smallest_pos_period_f : (smallest_period f) = π := by
  sorry

theorem monotonic_increase_intervals : 
  ∀ k : ℤ, (monotone_increase_interval f (k * π - π / 12, k * π + 5 * π / 12)) := by
  sorry

theorem calculate_g_at_pi_over_6 : 
  g (π / 6) = sqrt 3 + 1 := by
  sorry

end smallest_pos_period_f_monotonic_increase_intervals_calculate_g_at_pi_over_6_l23_23349


namespace mango_tree_start_count_l23_23945

variable (M : ℕ)

-- Conditions
def ripe_mangoes := (3/5 : ℚ) * M
def lindsay_eats := (60/100 : ℚ) * ripe_mangoes
def remaining_ripe_mangoes := (40/100 : ℚ) * ripe_mangoes

-- Theorem statement
theorem mango_tree_start_count (h1 : ripe_mangoes M = (3/5 : ℚ) * M)
                          (h2 : lindsay_eats M = (60/100 : ℚ) * (3/5 : ℚ) * M)
                          (h3 : remaining_ripe_mangoes M = 96) :
  M = 400 :=
sorry  -- Proof omitted

end mango_tree_start_count_l23_23945


namespace range_of_dot_product_l23_23325

open Real

-- Define the unit circle centered at O
def unit_circle (O : Point) (P : Point) : Prop := 
  dist O P = 1

-- Define the context with the given conditions
variables (O A B C : Point)

-- Point A is inside the unit circle and the distance from O to A is 1/2
axiom AO_half : dist O A = 1/2

-- Points B and C are on the unit circle
axiom B_on_unit_circle : unit_circle O B
axiom C_on_unit_circle : unit_circle O C

-- Define the dot product in 2D vector space
def dot_product (u v : Point) := u.1 * v.1 + u.2 * v.2

-- Define the vectors AC and BC
def vector_AC := (C.1 - A.1, C.2 - A.2)
def vector_BC := (C.1 - B.1, C.2 - B.2)

-- Define the problem statement to be proven
theorem range_of_dot_product : 
  -1/8 ≤ dot_product vector_AC vector_BC ∧ dot_product vector_AC vector_BC ≤ 3 := 
sorry

end range_of_dot_product_l23_23325


namespace probability_not_snowing_l23_23872

theorem probability_not_snowing (p_snow : ℚ) (h : p_snow = 5 / 8) : 1 - p_snow = 3 / 8 :=
by
  rw [h]
  sorry

end probability_not_snowing_l23_23872


namespace OK_perpendicular_PQ_l23_23212

-- Define the basic geometrical objects
variable (O A B C D K P Q : Point)

-- Define the circle centered at O
def Circle (O : Point) := {P : Point | dist O P = radius}

-- Define the conditions
variable (circle : Circle O)
variable (chord1 : Chord circle)
variable (chord2 : Chord circle)
variable (points_A_B : A ∈ chord1 ∧ B ∈ chord1)
variable (points_C_D : C ∈ chord2 ∧ D ∈ chord2)
variable (intersect_K : ChordsIntersectAt K chord1 chord2)

-- Define the tangents
variable (tangent_A : Tangent circle A P)
variable (tangent_B : Tangent circle B P)
variable (tangent_C : Tangent circle C Q)
variable (tangent_D : Tangent circle D Q)

-- Goal: Prove OK ⊥ PQ
theorem OK_perpendicular_PQ 
  (O_center : IsCenter O circle)
  (AB_intersect_CD_at_K : ChordsIntersectAt K chord1 chord2)
  (tangents_intersect_PQ : TangentsIntersectAt P Q tangent_A tangent_B tangent_C tangent_D)
  : Perpendicular (LineSegment O K) (LineSegment P Q) :=
sorry

end OK_perpendicular_PQ_l23_23212


namespace flagstaff_height_l23_23189

theorem flagstaff_height 
  (s1 : ℝ) (s2 : ℝ) (hb : ℝ) (h : ℝ)
  (H1 : s1 = 40.25) (H2 : s2 = 28.75) (H3 : hb = 12.5) 
  (H4 : h / s1 = hb / s2) : 
  h = 17.5 :=
by
  sorry

end flagstaff_height_l23_23189


namespace marathon_speed_ratio_l23_23819

theorem marathon_speed_ratio (M D : ℝ) (J : ℝ) (H1 : D = 9) (H2 : J = 4/3 * M) (H3 : M + J + D = 23) :
  D / M = 3 / 2 :=
by
  sorry

end marathon_speed_ratio_l23_23819


namespace solution_to_axb_eq_0_l23_23198

theorem solution_to_axb_eq_0 (a b x : ℝ) (h₀ : a ≠ 0) (h₁ : (0, 4) ∈ {p : ℝ × ℝ | p.snd = a * p.fst + b}) (h₂ : (-3, 0) ∈ {p : ℝ × ℝ | p.snd = a * p.fst + b}) :
  x = -3 :=
by
  sorry

end solution_to_axb_eq_0_l23_23198


namespace cost_sum_in_WD_l23_23602

def watch_cost_loss (W : ℝ) : ℝ := 0.9 * W
def watch_cost_gain (W : ℝ) : ℝ := 1.04 * W
def bracelet_cost_gain (B : ℝ) : ℝ := 1.08 * B
def bracelet_cost_reduced_gain (B : ℝ) : ℝ := 1.02 * B

theorem cost_sum_in_WD :
  ∃ W B : ℝ, 
    watch_cost_loss W + 196 = watch_cost_gain W ∧ 
    bracelet_cost_gain B - 100 = bracelet_cost_reduced_gain B ∧ 
    (W + B / 1.5 = 2511.11) :=
sorry

end cost_sum_in_WD_l23_23602


namespace pet_store_cages_l23_23169

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage remaining_puppies num_cages : ℕ)
  (h1 : initial_puppies = 102) 
  (h2 : sold_puppies = 21) 
  (h3 : puppies_per_cage = 9) 
  (h4 : remaining_puppies = initial_puppies - sold_puppies)
  (h5 : num_cages = remaining_puppies / puppies_per_cage) : 
  num_cages = 9 := 
by
  sorry

end pet_store_cages_l23_23169


namespace part1_solution_part2_solution_l23_23704

noncomputable def f (x : ℝ) : ℝ := abs (x + 5) - abs (x - 1)

theorem part1_solution (x : ℝ) :
  (-6 ≤ x ∧ x ≤ -4) ∨ (x ≥ 6) ↔ f x ≤ x :=
sorry

theorem part2_solution (a b : ℝ) (h : log a + log (2 * b) = log (a + 4 * b + 6)) :
  9 ≤ a * b :=
sorry

end part1_solution_part2_solution_l23_23704


namespace find_area_of_square_l23_23827

-- Definitions:
def Point := ℝ × ℝ
def Line := Point → Point → (ℝ × ℝ)

-- Conditions:
variable A B C D E F : Point
variable s : ℝ

-- Midpoint of BC
def isMidpoint (E: Point) (B C: Point) : Prop :=
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Definition of the square condition and areas
def square (A B C D: Point) : Prop :=
  A = (0, 0) ∧ B = (s, 0) ∧ C = (s, s) ∧ D = (0, s)

def line (P Q: Point) : Line := λ x y, (Q.1 - P.1, Q.2 - P.2)

-- Area condition:
def areaDFEC : ℝ := 36

-- Theorem statement:
theorem find_area_of_square (h1 : square A B C D)
                            (h2 : isMidpoint E B C)
                            (h3 : line intersection (line D E) (line A C) = F)
                            (h4 : areaDFEC = 36) :
  s * s = 144 :=
sorry


end find_area_of_square_l23_23827


namespace pumac_grader_remainder_l23_23569

/-- A PUMaC grader is grading the submissions of forty students s₁, s₂, ..., s₄₀ for the
    individual finals round, which has three problems.
    After grading a problem of student sᵢ, the grader either:
    * grades another problem of the same student, or
    * grades the same problem of the student sᵢ₋₁ or sᵢ₊₁ (if i > 1 and i < 40, respectively).
    He grades each problem exactly once, starting with the first problem of s₁
    and ending with the third problem of s₄₀.
    Let N be the number of different orders the grader may grade the students’ problems in this way.
    Prove: N ≡ 78 [MOD 100] -/

noncomputable def grading_orders_mod : ℕ := 2 * (3 ^ 38) % 100

theorem pumac_grader_remainder :
  grading_orders_mod = 78 :=
by
  sorry

end pumac_grader_remainder_l23_23569


namespace divide_into_equal_product_groups_l23_23452

theorem divide_into_equal_product_groups (s : Finset ℕ) (h : s = {12, 15, 33, 44, 51, 85}) :
  ∃ (g1 g2 : Finset ℕ), g1 ∪ g2 = s ∧ g1 ∩ g2 = ∅ ∧ (g1.product id).1 = (g2.product id).1 :=
by
  sorry

end divide_into_equal_product_groups_l23_23452


namespace intersection_distance_l23_23691

/-- We have two curves: one given by a parametric equation and another by a standard equation: 
  Curve 1: 
    x = 2 - t * sin(π / 6) 
    y = -1 + t * sin(π / 6) 
  Curve 2: 
    x^2 + y^2 = 8 
  We want to prove that the distance between the intersection points of these two curves is √30.
-/
theorem intersection_distance :
  (∃ t : ℝ, (λ (t : ℝ), (2 - t * Real.sin (Real.pi / 6), -1 + t * Real.sin (Real.pi / 6)))
    (t) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 8 }).dist (∃ t' : ℝ, 
    (λ (t' : ℝ), (2 - t' * Real.sin (Real.pi / 6), -1 + t' * Real.sin (Real.pi / 6)))
    (t') ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 8 }) = Real.sqrt 30 := sorry

end intersection_distance_l23_23691


namespace evelyn_average_sheets_per_day_l23_23202

theorem evelyn_average_sheets_per_day :
  let sheets_per_week := 2 + 4 + 6;
  let total_weeks := 48;
  let total_sheets := sheets_per_week * total_weeks;
  let total_days_off := 8;
  let working_days_per_week := 3;
  let total_working_days := working_days_per_week * total_weeks - total_days_off;
  (total_sheets / total_working_days : ℚ) ≈ 4 :=
by {
  let sheets_per_week := 12;
  let total_weeks := 48;
  let total_sheets := sheets_per_week * total_weeks;
  let total_days_off := 8;
  let working_days_per_week := 3;
  let total_working_days := working_days_per_week * total_weeks - total_days_off;
  -- Calculate the average sheets per day
  let average_sheets_per_day := (total_sheets : ℚ) / total_working_days;
  -- Ensure it is approximately 4
  have : average_sheets_per_day ≈ 4 := by {
    -- Numerically approximate
    calc average_sheets_per_day = (576 : ℚ) / 136 : by norm_num
                         ... ≈ 4 : by norm_num
  };
  exact sorry
}

end evelyn_average_sheets_per_day_l23_23202


namespace brendan_taxes_l23_23975

def total_hours (num_8hr_shifts : ℕ) (num_12hr_shifts : ℕ) : ℕ :=
  (num_8hr_shifts * 8) + (num_12hr_shifts * 12)

def total_wage (hourly_wage : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_wage * hours_worked

def total_tips (hourly_tips : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_tips * hours_worked

def reported_tips (total_tips : ℕ) (report_fraction : ℕ) : ℕ :=
  total_tips / report_fraction

def reported_income (wage : ℕ) (tips : ℕ) : ℕ :=
  wage + tips

def taxes (income : ℕ) (tax_rate : ℚ) : ℚ :=
  income * tax_rate

theorem brendan_taxes (num_8hr_shifts num_12hr_shifts : ℕ)
    (hourly_wage hourly_tips report_fraction : ℕ) (tax_rate : ℚ) :
    (hourly_wage = 6) →
    (hourly_tips = 12) →
    (report_fraction = 3) →
    (tax_rate = 0.2) →
    (num_8hr_shifts = 2) →
    (num_12hr_shifts = 1) →
    taxes (reported_income (total_wage hourly_wage (total_hours num_8hr_shifts num_12hr_shifts))
            (reported_tips (total_tips hourly_tips (total_hours num_8hr_shifts num_12hr_shifts))
            report_fraction))
          tax_rate = 56 :=
by
  intros
  sorry

end brendan_taxes_l23_23975


namespace six_points_concyclic_l23_23791

variables {A B C H : Point}
variables {γA γB γC : Circle}
variables {A1 A2 B1 B2 C1 C2 : Point}

-- Definitions for points and circles
noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def midpoint (P Q : Point) : Point := sorry
noncomputable def circle_passing_through (center : Point) (point : Point) : Circle := sorry
noncomputable def intersects (circ : Circle) (line : Line) : set Point := sorry
noncomputable def are_concyclic (points : set Point) : Prop := sorry

-- Conditions
axiom orthocenter_condition : H = orthocenter A B C
axiom circle_gA_condition : γA = circle_passing_through (midpoint B C) H
axiom intersects_BC_condition : set_eq (intersects γA (line_through B C)) {A1, A2}
axiom circle_gB_condition : γB = circle_passing_through (midpoint A C) H
axiom intersects_AC_condition : set_eq (intersects γB (line_through A C)) {B1, B2}
axiom circle_gC_condition : γC = circle_passing_through (midpoint A B) H
axiom intersects_AB_condition : set_eq (intersects γC (line_through A B)) {C1, C2}

-- Theorem to prove concyclicity
theorem six_points_concyclic :
  are_concyclic {A1, A2, B1, B2, C1, C2} :=
sorry

end six_points_concyclic_l23_23791


namespace log_ratio_l23_23218

theorem log_ratio : (Real.logb 2 16) / (Real.logb 2 4) = 2 := sorry

end log_ratio_l23_23218


namespace calculate_expression_l23_23615

theorem calculate_expression :
  (2 ^ (1/3) * 8 ^ (1/3) + 18 / (3 * 3) - 8 ^ (5/3)) = 2 ^ (4/3) - 30 :=
by
  sorry

end calculate_expression_l23_23615


namespace behavior_on_interval_6_8_l23_23486

noncomputable def f : ℝ → ℝ := sorry

-- Definition of an even function
def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

-- Definition of a periodic function with period 2
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x : ℝ, f (x + p) = f x

-- Definition of a decreasing function on an interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

-- Definition of an increasing function on an interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- The main theorem to be proven
theorem behavior_on_interval_6_8 :
  is_even f →
  is_periodic f 2 →
  is_decreasing_on f (-1) 0 →
  (is_increasing_on f 6 7 ∧ is_decreasing_on f 7 8) :=
by
  assume h_even h_periodic h_decreasing,
  sorry

end behavior_on_interval_6_8_l23_23486


namespace selection_methods_l23_23566

theorem selection_methods :
  ∃ (ways_with_girls : ℕ), ways_with_girls = Nat.choose 6 4 - Nat.choose 4 4 ∧ ways_with_girls = 14 := by
  sorry

end selection_methods_l23_23566


namespace dihedral_angle_N_CM_B_distance_B_to_plane_CMN_l23_23394

-- Define the given conditions
variables (A B C S M N : Point)
variables (h : is_equilateral_triangle ABC 4)
variables (p : plane_perpendicular (plane SAC) (plane ABC))
variables (sa : dist S A = 2 * sqrt 3)
variables (sc : dist S C = 2 * sqrt 3)
variables (m : midpoint M A B)
variables (n : midpoint N S B)

-- Problem 1: Prove the dihedral angle N-CM-B
theorem dihedral_angle_N_CM_B : 
  dihedral_angle N (line_through C M) (line_through M B) = arctan (2 * sqrt 2) := 
sorry

-- Problem 2: Prove the distance from B to the plane CMN
theorem distance_B_to_plane_CMN : 
  distance_to_plane B (plane_through C M N) = (4 * sqrt 2) / 3 := 
sorry

end dihedral_angle_N_CM_B_distance_B_to_plane_CMN_l23_23394


namespace impossibility_sum_of_shaded_cells_l23_23393

-- Define the grid as a type
def Grid := Array (Array ℕ)

-- Define the problem conditions as a Lean statement
theorem impossibility_sum_of_shaded_cells :
  ∀ (G : Grid), 
    (∀ i : Fin 5, ∀ j : Fin 5, G[i][j] ∈ {1, 2, 3, 4, 5}) →
    (∀ i : Fin 5, (G[i].toList.nodup)) → -- unique numbers in each row
    (∀ j : Fin 5, ((G.map (fun row => row[j])).toList.nodup)) → -- unique numbers in each column
    (G.toList.diag.nodup) → (G.toList.antidiag.nodup) → -- unique numbers in both diagonals
    (let shaded_indices := [(0, 0), (1, 1), (2, 2), (3, 2), (4, 1)]; -- hypothetical indices for shaded cells
        shaded_sum := shaded_indices.foldl (fun acc (i, j) => acc + G[i][j]) 0) ≠ 19 :=
by
  sorry

end impossibility_sum_of_shaded_cells_l23_23393


namespace arithmetic_progression_a6_l23_23306

theorem arithmetic_progression_a6 (a1 d : ℤ) (h1 : a1 + (a1 + d) + (a1 + 2 * d) = 168) (h2 : (a1 + 4 * d) - (a1 + d) = 42) : 
  a1 + 5 * d = 3 := 
sorry

end arithmetic_progression_a6_l23_23306


namespace keiko_speed_l23_23402

theorem keiko_speed
  (s : ℝ)  -- Keiko's speed in meters per second
  (b : ℝ)  -- Radius of the inner semicircle
  (a : ℝ)  -- Length of the straight sides of the track
  (h1 : ∀ t : ℝ, t > 0 → ∃ c, 2 * c = t)  -- Every positive real number t can be expressed as twice some real number c.
  (h2 : s > 0)  -- Keiko's speed is positive
  (h3 : 0 < b) -- Radius of inner semicircle is positive
  (h_time_diff : (2 * a + 2 * real.pi * (b + 8)) / s = (2 * a + 2 * real.pi * b) / s + 48) :
  s = real.pi / 3 := 
sorry

end keiko_speed_l23_23402


namespace color_of_203rd_marble_l23_23205

-- Defining colors as an enumeration
inductive Color
| blue
| red
| green

-- Defining the marble sequence pattern
def marblePattern : List Color := 
  List.repeat Color.blue 6 ++ List.repeat Color.red 5 ++ List.repeat Color.green 4

-- Function to find the color of the nth marble
def marbleColor (n : Nat) : Color := 
  marblePattern.get! ((n - 1) % marblePattern.length)

-- The theorem to prove
theorem color_of_203rd_marble : marbleColor 203 = Color.red := 
by sorry

end color_of_203rd_marble_l23_23205


namespace factorial_difference_l23_23143

open Nat 

theorem factorial_difference : (11! - 10!) / 9! = 100 := by
  have h1 : 11! = 11 * 10! := by 
    sorry

  have h2 : (11 * 10! - 10!) / 9! = (10! * (11 - 1)) / 9! := by 
    sorry

  have h3 : (10! * 10) / 9! = (10 * 9! * 10) / 9! := by
    sorry

  have h4 : (10 * 9! * 10) / 9! = 10 * 10 := by
    sorry

  show 100 = 100 from by
    sorry

end factorial_difference_l23_23143


namespace number_of_unpainted_cubes_l23_23565

theorem number_of_unpainted_cubes 
  (C1 : "A cube painted blue on all of its surfaces")
  (C2 : "The cube is cut into 27 smaller cubes of equal size") : 
  ∃ n : ℕ, n = 1 := 
by
  sorry

end number_of_unpainted_cubes_l23_23565


namespace find_B_l23_23108

def is_prime_203B21 (B : ℕ) : Prop :=
  2 ≤ B ∧ B < 10 ∧ Prime (200000 + 3000 + 100 * B + 20 + 1)

theorem find_B : ∃ B, is_prime_203B21 B ∧ ∀ B', is_prime_203B21 B' → B' = 5 := by
  sorry

end find_B_l23_23108


namespace find_abs_x_l23_23442

-- Given conditions
def A (x : ℝ) : ℝ := 3 + x
def B (x : ℝ) : ℝ := 3 - x
def distance (a b : ℝ) : ℝ := abs (a - b)

-- Problem statement: Prove |x| = 4 given the conditions
theorem find_abs_x (x : ℝ) (h : distance (A x) (B x) = 8) : abs x = 4 := 
  sorry

end find_abs_x_l23_23442


namespace propositions_imply_implication_l23_23229

theorem propositions_imply_implication (p q r : Prop) :
  ( ((p ∧ q ∧ ¬r) → ((p ∧ q) → r) = False) ∧ 
    ((¬p ∧ q ∧ r) → ((p ∧ q) → r) = True) ∧ 
    ((p ∧ ¬q ∧ r) → ((p ∧ q) → r) = True) ∧ 
    ((¬p ∧ ¬q ∧ ¬r) → ((p ∧ q) → r) = True) ) → 
  ( (∀ (x : ℕ), x = 3) ) :=
by
  sorry

end propositions_imply_implication_l23_23229


namespace option_A_not_correct_option_B_correct_option_C_correct_option_D_correct_l23_23154

theorem option_A_not_correct 
  (x : ℝ) : ¬ (∀ y, y = (x^2 + 1)/x → y ≥ 2) := 
sorry

theorem option_B_correct 
  (x y : ℝ) (h : x > 1) (hy : y = 2x + (4 / (x - 1)) - 1) : 
  y ≥ 4 * Real.sqrt 2 + 1 :=
sorry

theorem option_C_correct 
  {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 3 * x * y) : 
  2 * x + y ≥ 3 := 
sorry

theorem option_D_correct 
  {x y : ℝ} (h : 9 * x^2 + y^2 + x * y = 1) : 
  3 * x + y ≤ (2 * Real.sqrt 21) / 7 := 
sorry

end option_A_not_correct_option_B_correct_option_C_correct_option_D_correct_l23_23154


namespace q_correct_l23_23488

def q (x : ℝ) : ℝ := (12 * x^2 - 48) / 5

theorem q_correct (x : ℝ) (hx₁ : x ≠ -2) (hx₂ : x ≠ 2) : 
  (∀ x, x = 3 → q x = 12) ∧ 
  (∀ y, y = -2 → is_limit (λ n, q y) (∞) x) ∧ 
  (∀ z, z = 2 → is_limit (λ n, q z) (∞) x) :=
by 
  split
  repeat { sorry }

end q_correct_l23_23488


namespace quadratic_eq_coefficients_l23_23076

theorem quadratic_eq_coefficients :
  ∃ (a b c : ℤ), (a = 1 ∧ b = -1 ∧ c = 3) ∧ (∀ x : ℤ, a * x^2 + b * x + c = x^2 - x + 3) :=
by
  use 1, -1, 3
  split
  { split; refl }
  { intro x
    simp }
  sorry

end quadratic_eq_coefficients_l23_23076


namespace minimize_expr_l23_23664

noncomputable def min_expr (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) : ℝ :=
  let x := b + 3 * c
  let y := 8 * c + 4 * a
  let z := 3 * a + 2 * b
  (a / x) + (b / y) + (9 * c / z)

theorem minimize_expr (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
    min_expr a b c h₁ h₂ h₃ = 47 / 48 :=
begin
  sorry
end

end minimize_expr_l23_23664


namespace length_of_AE_l23_23777

theorem length_of_AE (AD AE EB EF: ℝ) (h_AD: AD = 80) (h_EB: EB = 40) (h_EF: EF = 30) 
  (h_eq_area: 2 * ((EB * EF) + (1 / 2) * (ED * (AD - EF))) = AD * (AD - AE)) : AE = 15 :=
  sorry

end length_of_AE_l23_23777


namespace scramble_words_count_l23_23859

-- Definitions based on the conditions
def alphabet_size : Nat := 25
def alphabet_size_no_B : Nat := 24

noncomputable def num_words_with_B : Nat :=
  let total_without_restriction := 25^1 + 25^2 + 25^3 + 25^4 + 25^5
  let total_without_B := 24^1 + 24^2 + 24^3 + 24^4 + 24^5
  total_without_restriction - total_without_B

-- Lean statement to prove the result
theorem scramble_words_count : num_words_with_B = 1692701 :=
by
  sorry

end scramble_words_count_l23_23859


namespace decreasing_function_range_l23_23347

noncomputable def f (a x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a / x

theorem decreasing_function_range (a : ℝ) :
  (∀ x y, x < y → f a x ≥ f a y) ↔ (1 / 6 ≤ a ∧ a < 1 / 3) :=
begin
  have h1 : 3 * a - 1 < 0 ↔ a < 1 / 3,
  sorry,
  have h2 : 0 < a,
  sorry,
  have h3 : a ≥ 1 / 6,
  sorry,
  split,
  { intro h,
    split,
    { apply h3.mpr,
      sorry },
    { apply h1.mpr,
      sorry } },
  { intro ha,
    sorry }
end

end decreasing_function_range_l23_23347


namespace range_of_m_l23_23091

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem range_of_m 
  (m : ℝ) 
  (H1: ∀ x ∈ set.Icc (-1) m, f x ≤ 10) 
  (H2: ∃ x ∈ set.Icc (-1) m, f x = 10) 
  (H3: ∃ x ∈ set.Icc (-1) m, f x = 1) : 
  m ∈ set.Icc 2 5 :=
sorry

end range_of_m_l23_23091


namespace correct_operation_l23_23146

-- Defining the options as hypotheses
variable {a b : ℕ}

theorem correct_operation (hA : 4*a + 3*b ≠ 7*a*b)
    (hB : a^4 * a^3 = a^7)
    (hC : (3*a)^3 ≠ 9*a^3)
    (hD : a^6 / a^2 ≠ a^3) :
    a^4 * a^3 = a^7 := by
  sorry

end correct_operation_l23_23146


namespace num_representable_integers_l23_23366

theorem num_representable_integers :
  let is_valid_coeff (b : Fin 3) := b ∈ {0, 1, 2}
  in (Finset.card (Finset.univ.filter (λ (f : Fin 8 → Fin 3), 
        ∀ i, is_valid_coeff (f i))) = 6561) := 
by 
  sorry

end num_representable_integers_l23_23366


namespace divisors_count_48n5_l23_23650

theorem divisors_count_48n5 (n : ℕ) (h1 : 0 < n) (h2 : (132 * n^3).numDivisors = 132) :
  (48 * n^5).numDivisors = 105 :=
by
  sorry

end divisors_count_48n5_l23_23650


namespace Mary_is_10_years_younger_l23_23603

theorem Mary_is_10_years_younger
  (betty_age : ℕ)
  (albert_age : ℕ)
  (mary_age : ℕ)
  (h1 : albert_age = 2 * mary_age)
  (h2 : albert_age = 4 * betty_age)
  (h_betty : betty_age = 5) :
  (albert_age - mary_age) = 10 :=
  by
  sorry

end Mary_is_10_years_younger_l23_23603


namespace complex_number_of_vector_AB_l23_23084

-- Define vector AB
def vector_AB := (2, -3)

-- Define the corresponding complex number
def complex_number_corresponding_to_vector (v : ℤ × ℤ) : ℂ :=
  v.1 + v.2 * complex.I

-- Statement that the complex number corresponding to vector AB is 2 - 3i
theorem complex_number_of_vector_AB : 
  complex_number_corresponding_to_vector vector_AB = 2 - 3 * complex.I := 
sorry

end complex_number_of_vector_AB_l23_23084


namespace oil_drop_probability_l23_23384

/-- Define the side length of the square hole  --/
def side_length_square_hole : ℝ := 1

/-- Define the diameter of the circular copper coin --/
def diameter_circular_coin : ℝ := 3

/-- Define the area of the square hole --/
def area_square_hole : ℝ := side_length_square_hole ^ 2

/-- Define the radius of the circular copper coin --/
def radius_circular_coin : ℝ := diameter_circular_coin / 2

/-- Define the area of the circular copper coin --/
def area_circular_coin : ℝ := π * (radius_circular_coin ^ 2)

/-- Define the probability that a drop of oil falls into the square hole --/
def probability_oil_drop_in_hole : ℝ := area_square_hole / area_circular_coin

/-- The theorem to prove the probability -/
theorem oil_drop_probability :
  probability_oil_drop_in_hole = 4 / (9 * π) := sorry

end oil_drop_probability_l23_23384


namespace marble_cut_percentage_l23_23964

theorem marble_cut_percentage
  (initial_weight : ℝ)
  (final_weight : ℝ)
  (x : ℝ)
  (first_week_cut : ℝ)
  (second_week_cut : ℝ)
  (third_week_cut : ℝ) :
  initial_weight = 190 →
  final_weight = 109.0125 →
  first_week_cut = (1 - x / 100) →
  second_week_cut = 0.85 →
  third_week_cut = 0.9 →
  (initial_weight * first_week_cut * second_week_cut * third_week_cut = final_weight) →
  x = 24.95 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end marble_cut_percentage_l23_23964


namespace stratified_sampling_girls_count_l23_23580

theorem stratified_sampling_girls_count :
  (boys girls sampleSize totalSample : ℕ) →
  boys = 36 →
  girls = 18 →
  sampleSize = 6 →
  totalSample = boys + girls →
  (sampleSize * girls) / totalSample = 2 :=
by
  intros boys girls sampleSize totalSample h_boys h_girls h_sampleSize h_totalSample
  sorry

end stratified_sampling_girls_count_l23_23580


namespace sqrt_product_simplified_l23_23219

variable (p : ℝ)

theorem sqrt_product_simplified :
  (sqrt (15 * p^3) * sqrt (20 * p^2) * sqrt (30 * p^5)) = (30 * p^5 * sqrt 10) := 
sorry

end sqrt_product_simplified_l23_23219


namespace pension_supplement_correct_l23_23070

noncomputable def future_value_annuity_due (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * ((1 + r)^n - 1) / r * (1 + r)

noncomputable def monthly_pension_supplement : ℝ :=
  let monthly_contribution := 7000
  let annual_contribution := 12 * monthly_contribution
  let annual_interest_rate := 0.09
  let contributions_period_years := 20
  let accumulated_amount := future_value_annuity_due annual_contribution annual_interest_rate contributions_period_years
  let distribution_period_months := 15 * 12
  accumulated_amount / distribution_period_months

theorem pension_supplement_correct :
  monthly_pension_supplement ≈ 26023.45 :=
by
  sorry

end pension_supplement_correct_l23_23070


namespace count_sums_to_5_l23_23390

theorem count_sums_to_5 : 
  let count_ways (n : ℕ) := nat.card { l : list ℕ // l.sum = n ∧ ∀ x ∈ l, x > 0 }
  count_ways 5 = 16 :=
by
  sorry

end count_sums_to_5_l23_23390


namespace acid_solution_replaced_l23_23063

theorem acid_solution_replaced (P : ℝ) :
  (0.5 * 0.50 + 0.5 * P = 0.35) → P = 0.20 :=
by
  intro h
  sorry

end acid_solution_replaced_l23_23063


namespace articles_correct_l23_23939

-- Define the problem conditions
def refersToSpecific (word : String) : Prop :=
  word = "keyboard"

def refersToGeneral (word : String) : Prop :=
  word = "computer"

-- Define the articles
def the_article : String := "the"
def a_article : String := "a"

-- State the theorem for the corresponding solution
theorem articles_correct :
  refersToSpecific "keyboard" → refersToGeneral "computer" →  
  (the_article, a_article) = ("the", "a") :=
by
  intro h1 h2
  sorry

end articles_correct_l23_23939


namespace SwitchedSeq_is_supermartingale_l23_23813

-- Define an abstract type for a filtration (sequence of sigma algebras).
axiom Filtration (α : Type) (n : ℕ) : Type 

-- Define supermartingales processes.
axiom Supermartingale (α : Type) (n : ℕ) [ProbabilityMeasure α] (F : Filtration α n) : Type

-- Define stopping times relative to a given filtration.
axiom StoppingTime (α : Type) (n : ℕ) [ProbabilityMeasure α] (F : Filtration α n) : Type

-- Define the "switched" sequences \(\zeta_k\).
def SwitchedSeq (α : Type) (n : ℕ) [ProbabilityMeasure α] (F : Filtration α n) 
  (ξ η : Supermartingale α n F) (τ : StoppingTime α n F) (k : ℕ) :=
  if τ > k then ξ else η

-- Main proof sketch to show \(\zeta\) is a supermartingale given \(\mathrm{P}(\xi_{\tau} \geq \eta_{\tau}) = 1\).
theorem SwitchedSeq_is_supermartingale 
  (α : Type) (n : ℕ) [ProbabilityMeasure α] 
  (F : Filtration α n) 
  (ξ η : Supermartingale α n F) 
  (τ : StoppingTime α n F) 
  (h : ∀ t : ℕ, t ≤ n → t ∈ τ → \xi t ≥ η t) :
  Supermartingale α n F :=
sorry  -- Detailed proof is omitted for this theorem.

end SwitchedSeq_is_supermartingale_l23_23813


namespace sequence_sum_is_9_l23_23228

-- Define the sequence recursively as per the conditions
def sequence (n : ℕ) : ℝ :=
  if n = 1 then 2
  else if n = 2 then 3
  else (1/4) * sequence (n - 1) + (1/5) * sequence (n - 2)

-- Define the infinite sum of the sequence
noncomputable def sequence_sum : ℝ := ∑' n, sequence n

-- Statement of the problem in Lean
theorem sequence_sum_is_9 : sequence_sum = 9 :=
  sorry

end sequence_sum_is_9_l23_23228


namespace three_digit_numbers_l23_23253

theorem three_digit_numbers (a b c n : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
    (h5 : 0 ≤ c) (h6 : c ≤ 9) (h7 : n = 100 * a + 10 * b + c) (h8 : 10 * b + c = (100 * a + 10 * b + c) / 5) :
    n = 125 ∨ n = 250 ∨ n = 375 := 
by 
  sorry

end three_digit_numbers_l23_23253


namespace a_is_perfect_square_l23_23031

theorem a_is_perfect_square (a b : ℕ) (h : ∃ (k : ℕ), a^2 + b^2 + a = k * a * b) : ∃ n : ℕ, a = n^2 := by
  sorry

end a_is_perfect_square_l23_23031


namespace base_five_to_base_ten_l23_23907

theorem base_five_to_base_ten : 
  let b := 5 in 
  let x := 123 % b in
  let y := (123 / b) % b in
  let z := (123 / b) / b in
  z * b^2 + y * b + x = 38 :=
by {
  let b := 5,
  let x := 123 % b,  -- least significant digit
  let y := (123 / b) % b,  -- middle digit
  let z := (123 / b) / b,  -- most significant digit
  have hx : x = 3 := by norm_num,  -- 123 % 5 = 3
  have hy : y = 2 := by norm_num,  -- (123 / 5) % 5 = 2
  have hz : z = 1 := by norm_num,  -- (123 / 5) / 5 = 1
  rw [hx, hy, hz],
  norm_num
}

end base_five_to_base_ten_l23_23907


namespace A_finishes_remaining_work_in_2_days_l23_23177

/-- 
Given that A's daily work rate is 1/6 of the work and B's daily work rate is 1/15 of the work,
and B has already completed 2/3 of the work, 
prove that A can finish the remaining work in 2 days.
-/
theorem A_finishes_remaining_work_in_2_days :
  let A_work_rate := (1 : ℝ) / 6
  let B_work_rate := (1 : ℝ) / 15
  let B_work_in_10_days := (10 : ℝ) * B_work_rate
  let remaining_work := (1 : ℝ) - B_work_in_10_days
  let days_for_A := remaining_work / A_work_rate
  B_work_in_10_days = 2 / 3 → 
  remaining_work = 1 / 3 → 
  days_for_A = 2 :=
by
  sorry

end A_finishes_remaining_work_in_2_days_l23_23177


namespace prove_coins_authenticity_l23_23047

-- Define the coins and their authenticity
def Coin : Type := ℕ
def fake : Coin → Prop
def real : Coin → Prop

-- Define the balance scale function
def balance_scale : (list Coin) → (list Coin) → Prop := sorry

-- Define the conditions for the problem
def first_weighing : Prop :=
  balance_scale [1] [8] →
  fake 1 ∧ real 8

def second_weighing : Prop :=
  balance_scale [2, 3, 8] [1, 9, 10] →
  fake 2 ∧ fake 3 ∧ real 9 ∧ real 10

def third_weighing : Prop :=
  balance_scale [4, 5, 6, 7, 8, 9, 10] [1, 2, 3, 11, 12, 13, 14] →
  fake 4 ∧ fake 5 ∧ fake 6 ∧ fake 7 ∧ real 11 ∧ real 12 ∧ real 13 ∧ real 14

-- Combine all conditions
def all_weighings : Prop :=
  first_weighing ∧ second_weighing ∧ third_weighing

-- The theorem to prove
theorem prove_coins_authenticity : all_weighings → 
  (∀ i, 1 ≤ i ∧ i ≤ 7 → fake i) ∧ (∀ i, 8 ≤ i ∧ i ≤ 14 → real i) :=
by 
  intro h,
  sorry

end prove_coins_authenticity_l23_23047


namespace number_of_rhombuses_of_8_small_triangles_l23_23211

-- Define the conditions: large triangle and small triangles
def large_triangle_side_length : ℕ := 10
def small_triangle_side_length : ℕ := 1
def total_small_triangles : ℕ := 100

-- Define the main theorem to be proved
theorem number_of_rhombuses_of_8_small_triangles : 
  large_triangle_side_length = 10 → 
  small_triangle_side_length = 1 → 
  total_small_triangles = 100 → 
  Exists (λ n, n = 84) :=
by 
  intros; 
  use 84;
  sorry

end number_of_rhombuses_of_8_small_triangles_l23_23211


namespace solve_quadratic_l23_23113

theorem solve_quadratic (x : ℝ) : (x^2 + 2*x = 0) ↔ (x = 0 ∨ x = -2) :=
by
  sorry

end solve_quadratic_l23_23113


namespace rhombus_area_is_correct_l23_23559

def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

theorem rhombus_area_is_correct :
  area_of_rhombus 13 20 = 130 :=
by
  -- proof skipped
  sorry

end rhombus_area_is_correct_l23_23559


namespace ellipse_of_sum_of_distances_l23_23232

noncomputable def point (α : Type) := α
variable {α : Type} [pseudo_metric_space α]

theorem ellipse_of_sum_of_distances (A B : point α) (d : ℝ) (P : point α) :
    dist A B = d → dist P A + dist P B = 2 * d → ∃ E : set (point α), is_ellipse E A B d ∧ P ∈ E := by
  sorry

end ellipse_of_sum_of_distances_l23_23232


namespace max_product_of_triangle_sides_l23_23671

theorem max_product_of_triangle_sides (a c : ℝ) (ha : a ≥ 0) (hc : c ≥ 0) :
  ∃ b : ℝ, b = 4 ∧ ∃ B : ℝ, B = 60 * (π / 180) ∧ a^2 + c^2 - a * c = b^2 ∧ a * c ≤ 16 :=
by
  sorry

end max_product_of_triangle_sides_l23_23671


namespace scientific_notation_correct_l23_23106

/-- Given the weight of the "人" shaped gate of the Three Gorges ship lock -/
def weight_kg : ℝ := 867000

/-- The scientific notation representation of the given weight -/
def scientific_notation_weight_kg : ℝ := 8.67 * 10^5

theorem scientific_notation_correct :
  weight_kg = scientific_notation_weight_kg :=
sorry

end scientific_notation_correct_l23_23106


namespace find_radius_of_ω_l23_23783

noncomputable def point : Type := ℝ × ℝ
noncomputable def circle : Type := point × ℝ

variables (K L M : point)
variables (ω ω₁ ω₂ : circle)

-- The condition that L and M are points of intersection between ω₂ and ω
variables (intersects : ∀ p, (p = L ∨ p = M) → (∃ c₁ c₂ : circle, c₁ ∈ {ω₂} ∧ c₂ ∈ {ω} ∧ ∃ p : point, p ∈ (circle_intersection c₁ c₂)))
-- The collinearity condition
variables (collinear : collinear ℝ {K, L, M})

-- Radii conditions
variables (r₁ r₂ : ℝ) (hr₁ : r₁ = 4) (hr₂ : r₂ = 7) (h₁ : ω₁ = (K, r₁)) (h₂ : ω₂ = (L, r₂)) (r : ℝ) (hω : ω = (origin, r))

-- The main statement
theorem find_radius_of_ω : r = 11 := sorry

end find_radius_of_ω_l23_23783


namespace age_difference_l23_23570

-- Define the hypothesis and statement
theorem age_difference (A B C : ℕ) 
  (h1 : A + B = B + C + 15)
  (h2 : C = A - 15) : 
  (A + B) - (B + C) = 15 :=
by
  sorry

end age_difference_l23_23570


namespace no_scalar_exists_l23_23240

theorem no_scalar_exists (v : ℝ^3) : ¬ ∃ d : ℝ, 
  (unit_vector i) × (v ⨯ (unit_vector j)) + (unit_vector j) × (v ⨯ (unit_vector k)) + (unit_vector k) × (v ⨯ (unit_vector i)) = d • v :=
by
  sorry

end no_scalar_exists_l23_23240


namespace max_points_of_intersection_l23_23535

theorem max_points_of_intersection (circles : ℕ) (line : ℕ) (h_circles : circles = 3) (h_line : line = 1) : 
  ∃ points_of_intersection, points_of_intersection = 12 :=
by
  -- Proof here (omitted)
  sorry

end max_points_of_intersection_l23_23535


namespace probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l23_23758

noncomputable def probability_k_gnomes_fall (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p * (1 - p) ^ (n - k)

noncomputable def expected_number_of_fallen_gnomes (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p)

theorem probability_k_gnomes_fall_correct (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  probability_k_gnomes_fall n k p h = p * (1 - p) ^ (n - k) :=
by sorry

theorem expected_number_of_fallen_gnomes_correct (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  expected_number_of_fallen_gnomes n p h = n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p) :=
by sorry

end probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l23_23758


namespace min_value_of_sum_of_powers_l23_23287

theorem min_value_of_sum_of_powers (x y : ℝ) (h : x + 3 * y = 1) : 
  2^x + 8^y ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_of_sum_of_powers_l23_23287


namespace change_correct_l23_23722

def cost_gum : ℕ := 350
def cost_protractor : ℕ := 500
def amount_paid : ℕ := 1000

theorem change_correct : amount_paid - (cost_gum + cost_protractor) = 150 := by
  sorry

end change_correct_l23_23722


namespace geometric_seq_a8_l23_23116

noncomputable def geometric_seq_term (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n-1)

noncomputable def geometric_seq_sum (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - r^n) / (1 - r)

theorem geometric_seq_a8
  (a₁ r : ℝ)
  (h1 : geometric_seq_sum a₁ r 3 = 7/4)
  (h2 : geometric_seq_sum a₁ r 6 = 63/4)
  (h3 : r ≠ 1) :
  geometric_seq_term a₁ r 8 = 32 :=
by
  sorry

end geometric_seq_a8_l23_23116


namespace sum_of_solutions_is_267_l23_23057

open Set

noncomputable def inequality (x : ℝ) : Prop :=
  sqrt (x^2 + x - 56) - sqrt (x^2 + 25*x + 136) < 8 * sqrt ((x - 7) / (x + 8))

noncomputable def valid_integers : Set ℝ :=
  {x | x ∈ Icc (-25 : ℝ) 25 ∧ (x ∈ (-20 : ℝ, -18) ∨ x ∈ Ici (7 : ℝ))}

theorem sum_of_solutions_is_267 :
  ∑ i in (Icc (-25 : ℝ) 25).to_finset.filter (λ x, inequality x), x = 267 :=
sorry

end sum_of_solutions_is_267_l23_23057


namespace marcia_savings_l23_23214

def hat_price := 60
def regular_price (n : ℕ) := n * hat_price
def discount_price (discount_percentage: ℕ) (price: ℕ) := price - (price * discount_percentage) / 100
def promotional_price := hat_price + discount_price 25 hat_price + discount_price 35 hat_price

theorem marcia_savings : (regular_price 3 - promotional_price) * 100 / regular_price 3 = 20 :=
by
  -- The proof steps would follow here.
  sorry

end marcia_savings_l23_23214


namespace smallest_b_for_N_fourth_power_l23_23938

theorem smallest_b_for_N_fourth_power : 
  ∃ (b : ℤ), (∀ n : ℤ, 7 * b^2 + 7 * b + 7 = n^4) ∧ b = 18 :=
by
  sorry

end smallest_b_for_N_fourth_power_l23_23938


namespace max_intersections_three_circles_one_line_l23_23532

theorem max_intersections_three_circles_one_line : 
  ∀ (C1 C2 C3 : Circle) (L : Line), 
  same_paper C1 C2 C3 L → 
  max_intersections C1 C2 C3 L = 12 := 
sorry

end max_intersections_three_circles_one_line_l23_23532


namespace probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l23_23762

theorem probability_exactly_k_gnomes_fall (n k : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in p * q^(n - k) = p * (1 - p)^(n - k) := 
sorry

theorem expected_number_of_gnomes_fall (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in 
  (∑ j in finset.range n, (1 - q^(j+1))) = n + 1 - (1 / p) + ((1 - p)^(n+1) / p) :=
sorry

end probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l23_23762


namespace probability_correct_dial_l23_23023

theorem probability_correct_dial : 
  let num_options_first_three : ℕ := 2,
      num_options_last_four : ℕ := 24 -- 4! = 24
  in (1 / (num_options_first_three * num_options_last_four) : ℚ) = 1 / 48 := 
by 
  sorry

end probability_correct_dial_l23_23023


namespace investment_of_c_l23_23159

theorem investment_of_c (P_b P_a P_c C_a C_b C_c : ℝ)
  (h1 : P_b = 2000) 
  (h2 : P_a - P_c = 799.9999999999998)
  (h3 : C_a = 8000)
  (h4 : C_b = 10000)
  (h5 : P_b / C_b = P_a / C_a)
  (h6 : P_c / C_c = P_a / C_a)
  : C_c = 4000 :=
by 
  sorry

end investment_of_c_l23_23159


namespace find_a6_l23_23317

variable (a_n : ℕ → ℤ) (d : ℤ)

-- Conditions
axiom sum_first_three_terms (S3 : a_n 1 + a_n 2 + a_n 3 = 168)
axiom diff_terms (diff_a2_a5 : a_n 2 - a_n 5 = 42)

-- Definition of arithmetic progression 
def arith_prog (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d

-- Proving that a6 = 3
theorem find_a6 (a1 : ℤ) (proof_S3 : a1 + (a1 + d) + (a1 + 2*d) = 168)
  (proof_diff : (a1 + d) - (a1 + 4*d) = 42) : a1 + 5*d = 3 :=
by
  sorry

end find_a6_l23_23317


namespace Prudence_sleep_weeks_l23_23280

def Prudence_sleep_per_week : Nat := 
  let nights_sleep_weekday := 6
  let nights_sleep_weekend := 9
  let weekday_nights := 5
  let weekend_nights := 2
  let naps := 1
  let naps_days := 2
  weekday_nights * nights_sleep_weekday + weekend_nights * nights_sleep_weekend + naps_days * naps

theorem Prudence_sleep_weeks (w : Nat) (h : w * Prudence_sleep_per_week = 200) : w = 4 :=
by
  sorry

end Prudence_sleep_weeks_l23_23280


namespace unit_fraction_decomposition_l23_23250

theorem unit_fraction_decomposition (n : ℕ) (hn : 0 < n): 
  (1 : ℚ) / n = (1 : ℚ) / (2 * n) + (1 : ℚ) / (3 * n) + (1 : ℚ) / (6 * n) :=
by
  sorry

end unit_fraction_decomposition_l23_23250


namespace negation_of_existential_l23_23867

theorem negation_of_existential :
  (¬ (∃ x : ℝ, x^2 - x - 1 > 0)) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
sorry

end negation_of_existential_l23_23867


namespace sum_of_arithmetic_series_105_to_120_l23_23977

theorem sum_of_arithmetic_series_105_to_120 : 
  (∑ k in finset.Icc 105 120, k) = 1800 :=
by
  sorry

end sum_of_arithmetic_series_105_to_120_l23_23977


namespace XY_and_Z_collinear_l23_23560

variables {C O A B M S T E F X Y P Q R Z : Type*}

-- Chord AB of circle O
constants (h1 : is_chord O A B)

-- M is midpoint of arc AB
constants (h2 : is_midpoint_arc O A B M)

-- C is a point outside circle O
constants (h3 : is_external_point O C)

-- Tangents CS and CT from C to circle O
constants (h4 : is_tangent O C S) (h5 : is_tangent O C T)

-- MS and MT intersect AB at points E and F respectively
constants (h6 : intersect_at_line M S A B E) (h7 : intersect_at_line M T A B F)

-- Perpendiculars from E and F to AB intersect OS and OT at X and Y respectively
constants (h8 : is_perpendicular E F A B) (h9 : intersect_at_perpendicular E F O S X) (h10 : intersect_at_perpendicular E F O T Y)

-- Secant through C intersects circle O at points P and Q
constants (h11 : is_secant C O P Q)

-- MP intersects AB at R
constants (h12 : intersect_at_line M P A B R)

-- Z is the circumcenter of triangle PQR
constants (h13 : circumcenter P Q R Z)

-- Prove that X, Y, and Z are collinear
theorem XY_and_Z_collinear : collinear X Y Z :=
sorry

end XY_and_Z_collinear_l23_23560


namespace find_a6_l23_23316

variable (a_n : ℕ → ℤ) (d : ℤ)

-- Conditions
axiom sum_first_three_terms (S3 : a_n 1 + a_n 2 + a_n 3 = 168)
axiom diff_terms (diff_a2_a5 : a_n 2 - a_n 5 = 42)

-- Definition of arithmetic progression 
def arith_prog (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d

-- Proving that a6 = 3
theorem find_a6 (a1 : ℤ) (proof_S3 : a1 + (a1 + d) + (a1 + 2*d) = 168)
  (proof_diff : (a1 + d) - (a1 + 4*d) = 42) : a1 + 5*d = 3 :=
by
  sorry

end find_a6_l23_23316


namespace limit_series_is_8_l23_23331

open Nat Real

/-- Prove that the limit of the given series is 8. -/
theorem limit_series_is_8 (a_n : ℕ → ℝ)
  (h1 : ∀ n : ℕ, n ≥ 2 → a_n = (binom n 2) * (2 : ℝ)^(n-2)) :
  (∃ L : ℝ, L = 8 ∧
    is_limit (λ n : ℕ, ∑ k in range (n - 1), 2^(k+2) / a_n (k+2)) L) :=
by
  sorry

end limit_series_is_8_l23_23331


namespace sum_integer_solutions_in_interval_l23_23060

theorem sum_integer_solutions_in_interval :
  (∑ x in (set.Icc (-25 : ℤ) (25 : ℤ)) \ {x : ℤ | (x^2 + x - 56).sqrt - (x^2 + 25*x + 136).sqrt < 8 * ((x - 7) / (x + 8)).sqrt}, (x : ℤ)).sum = 267 :=
by
  sorry

end sum_integer_solutions_in_interval_l23_23060


namespace compute_volume_tetrahedron_formed_by_red_vertices_l23_23184

-- Define a cube side length
def side_length : ℝ := 10

-- Define that each vertex is colored either blue or red alternately
-- For simplicity, we assume an indexing scheme that correctly sets the alternation.
-- color_vertex is a placeholder for this alternation rule.
def color_vertex : ℤ × ℤ × ℤ → Prop
| (x, y, z) => (x + y + z) % 2 = 0

-- Volume of the cube
def volume_cube : ℝ := side_length ^ 3

-- Volume of the red tetrahedron formed by red vertices
def volume_red_tetrahedron : ℝ := 333.33

theorem compute_volume_tetrahedron_formed_by_red_vertices :
  volume_red_tetrahedron ≈ 333.33 :=
by 
  -- The proof will go here
  sorry

end compute_volume_tetrahedron_formed_by_red_vertices_l23_23184


namespace complex_parts_l23_23640

open Complex

-- Define the complex number z = 2 / (1 + i)
def z : ℂ := 2 / (1 + Complex.i)

-- State the theorem that the real part of z is 1 and the imaginary part is -1
theorem complex_parts :
  Complex.re z = 1 ∧ Complex.im z = -1 :=
sorry

end complex_parts_l23_23640


namespace installation_cost_is_310_l23_23049

-- Define the given conditions
def labelled_price : ℝ := 12500 / 0.80
def required_selling_price : ℝ := labelled_price + 0.16 * labelled_price
def actual_selling_price : ℝ := 18560
def transport_cost : ℝ := 125

-- Define the installation cost and prove it equals 310
theorem installation_cost_is_310 : 
  let extra_amount := actual_selling_price - required_selling_price in
  let installation_cost := extra_amount - transport_cost in
  installation_cost = 310 :=
by 
  let lp := labelled_price
  let sp := required_selling_price
  let ea := actual_selling_price - sp
  let ic := ea - transport_cost
  have h1 : lp = 15625 := by sorry
  have h2 : sp = 18125 := by sorry
  have h3 : ea = 435 := by sorry
  show ic = 310 from sorry

end installation_cost_is_310_l23_23049


namespace guard_team_duty_coverage_l23_23386

structure Guard :=
(rank : ℕ)
(duty_cycle : ℕ → Prop)

def Guard.is_on_duty (g : Guard) (day : ℕ) : Prop :=
  day % (2 * g.rank) < g.rank

def team_guards_ensure_daily_duty_coverage (guards : List Guard) : Prop :=
  ∀ day, ∃ g ∈ guards, g.is_on_duty day

theorem guard_team_duty_coverage (guards : List Guard) 
  (h1 : ∀ g1 g2 ∈ guards, g1 ≠ g2 → g1.rank ≥ 3 * g2.rank)
  (h2 : ∀ g ∈ guards, ∃ N, g.duty_cycle N ∧ N = g.rank ∧ (forall k, g.is_on_duty k ↔ k % (2 * N) < N)) :
  team_guards_ensure_daily_duty_coverage guards :=
begin
  sorry
end

end guard_team_duty_coverage_l23_23386


namespace waynes_son_time_to_shovel_l23_23157

-- Definitions based on the conditions
variables (S W : ℝ) (son_rate : S = 1 / 21) (wayne_rate : W = 6 * S) (together_rate : 3 * (S + W) = 1)

theorem waynes_son_time_to_shovel : 
  1 / S = 21 :=
by
  -- Proof will be provided later
  sorry

end waynes_son_time_to_shovel_l23_23157


namespace sqrt_meaningful_iff_ge_two_l23_23726

-- State the theorem according to the identified problem and conditions
theorem sqrt_meaningful_iff_ge_two (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 2)) → x ≥ 2 :=
by
  sorry  -- Proof placeholder

end sqrt_meaningful_iff_ge_two_l23_23726


namespace cot_minus_double_sin_l23_23226

theorem cot_minus_double_sin :
  let x := Real.sin (Real.pi / 9)
  in Real.cot (Real.pi / 18) - 2 * x = (4 - 2 * Real.sqrt 3) / (Real.sqrt 2 * (Real.sqrt 3 - 1)) := by
  let x := Real.sin (Real.pi / 9)
  sorry

end cot_minus_double_sin_l23_23226


namespace white_sox_wins_l23_23244

theorem white_sox_wins 
  (total_games : ℕ) 
  (games_won : ℕ) 
  (games_lost : ℕ)
  (win_loss_difference : ℤ) 
  (total_games_condition : total_games = 162) 
  (lost_games_condition : games_lost = 63) 
  (win_loss_diff_condition : (games_won : ℤ) - games_lost = win_loss_difference) 
  (win_loss_difference_value : win_loss_difference = 36) 
  : games_won = 99 :=
by
  sorry

end white_sox_wins_l23_23244


namespace line_through_points_slope_intercept_sum_l23_23856

theorem line_through_points_slope_intercept_sum :
  ∃ m b : ℝ, (∀ x y : ℝ, (y = m * x + b) → ((((x, y) = (-3, 1)) ∨ ((x, y) = (1, 3))) ⇒ y = m * x + b)) ∧ (m + b = 3) :=
begin
  sorry
end

end line_through_points_slope_intercept_sum_l23_23856


namespace count_decreasing_digits_3_digit_numbers_l23_23720

theorem count_decreasing_digits_3_digit_numbers : 
  ∃ n : ℕ, (∀ abc : ℕ, 100 ≤ abc ∧ abc ≤ 999 → 
    let a := abc / 100 in
    let b := (abc / 10) % 10 in
    let c := abc % 10 in
    a > b ∧ b > c → True) ∧ n = 84 := 
sorry

end count_decreasing_digits_3_digit_numbers_l23_23720


namespace calligraphy_only_students_l23_23845

-- Define the sets and the given cardinalities
variables (C A M : Set ℕ)
variables (card_C : Set.card C = 29)
variables (card_CA : Set.card (C ∩ A) = 13)
variables (card_CM : Set.card (C ∩ M) = 12)
variables (card_CAM : Set.card (C ∩ A ∩ M) = 5)

-- Define the theorem to prove the number of students only in the calligraphy class
theorem calligraphy_only_students :
  Set.card (C \ (A ∪ M)) = 9 :=
by
  -- Skip the proof here with sorry
  sorry

end calligraphy_only_students_l23_23845


namespace find_a_value_l23_23288

-- Problem statement
theorem find_a_value (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) :
  (∀ x : ℝ, x^2 + 2 * x^10 = a + a1 * (x+1) + a2 * (x+1)^2 + a3 * (x+1)^3 + a4 * (x+1)^4 + a5 * (x+1)^5 + a6 * (x+1)^6 + a7 * (x+1)^7 + a8 * (x+1)^8 + a9 * (x+1)^9 + a10 * (x+1)^(10)) → a = 3 :=
by sorry

end find_a_value_l23_23288


namespace find_x_value_l23_23924

noncomputable def solve_x (x : ℝ) :=
  let initial_salt := 0.20 * x in
  let remaining_volume := (3 / 4) * x in
  let total_salt := initial_salt + 16 in
  let total_volume := remaining_volume + 8 + 16 in
  total_salt / total_volume = 1 / 3

theorem find_x_value : ∃ x : ℝ, solve_x x ∧ x = 160 := 
by {
  use 160,
  unfold solve_x,
  sorry
}

end find_x_value_l23_23924


namespace nearest_integer_bn_division_36_25_l23_23648

noncomputable def least_common_multiple (n : ℕ) : ℕ :=
  Nat.lcm (Finset.range n).filter(λ m, m > 0).val

noncomputable def bn (n : ℕ) : ℕ :=
  1 + least_common_multiple n

lemma bn_definition (n : ℕ) : bn n = 1 + least_common_multiple n := rfl

theorem nearest_integer_bn_division_36_25 :
  let b36 := bn 36
  let b25 := bn 25
  let ratio := (b36 : ℚ) / b25
  in Int.nearest ratio = 1798 :=
by
  sorry

end nearest_integer_bn_division_36_25_l23_23648


namespace a_le_neg2_l23_23112

theorem a_le_neg2 (a : ℝ) : (∀ x : ℝ, (x + 5 > 3) → (x > a)) → a ≤ -2 :=
by
  intro h
  have h_neg : ∀ x : ℝ, (x > -2) → (x > a) := 
    by 
      intro x hx
      exact h x (by linarith)

  specialize h_neg (-1) (by linarith)
  linarith

end a_le_neg2_l23_23112


namespace sum_geom_seq_terms_l23_23330

variable {α : Type*}
variable [Field α]

def geom_seq (a r : α) (n : ℕ) : α := a * r^n

def partial_sum (a r : α) (n : ℕ) : α := 
  if r = 1 then a * n else a * (1 - r^(n+1)) / (1 - r)

variable {a r : α}

theorem sum_geom_seq_terms :
  geom_seq a r 3 - geom_seq a r 0 + geom_seq a r 15 - geom_seq a r 12 = 32 :=
by
  -- This is where the proof would go, but it's omitted as per instructions
  sorry

end sum_geom_seq_terms_l23_23330


namespace solve_exponential_equation_l23_23056

theorem solve_exponential_equation (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3^x + 4^y = 5^z ↔ x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end solve_exponential_equation_l23_23056


namespace proof_part_a_l23_23925

variable {α : Type} [LinearOrder α]

structure ConvexQuadrilateral (α : Type) :=
(a b c d : α)
(a'b'c'd' : α)
(ab_eq_a'b' : α)
(bc_eq_b'c' : α)
(cd_eq_c'd' : α)
(da_eq_d'a' : α)
(angle_A_gt_angle_A' : Prop)
(angle_B_lt_angle_B' : Prop)
(angle_C_gt_angle_C' : Prop)
(angle_D_lt_angle_D' : Prop)

theorem proof_part_a (Quad : ConvexQuadrilateral ℝ) : 
  Quad.angle_A_gt_angle_A' → 
  Quad.angle_B_lt_angle_B' ∧ Quad.angle_C_gt_angle_C' ∧ Quad.angle_D_lt_angle_D' := sorry

end proof_part_a_l23_23925


namespace right_triangles_needed_to_cover_equilateral_l23_23911

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s^2 * Real.sqrt 3) / 4

noncomputable def area_right_triangle (leg : ℝ) : ℝ :=
  (leg^2) / 2

noncomputable def minimum_number_of_right_triangles (side_len : ℝ) (leg_len : ℝ) : ℝ :=
  let area_large_triangle := area_equilateral_triangle side_len
  let area_small_triangle := area_right_triangle leg_len
  Real.ceil (area_large_triangle / area_small_triangle / 2)

theorem right_triangles_needed_to_cover_equilateral (s : ℝ) (leg : ℝ) 
  (hs : s = 7) (hleg : leg = 1) : minimum_number_of_right_triangles s leg = 85 := 
by
  rw [hs, hleg]
  unfold minimum_number_of_right_triangles 
  unfold area_equilateral_triangle area_right_triangle
  sorry

end right_triangles_needed_to_cover_equilateral_l23_23911


namespace evaluation_l23_23630

noncomputable def omega : ℂ := 7 + 3 * complex.I
def expression := omega^2 + 4 * omega + 65

theorem evaluation :
  complex.abs expression = 20605 :=
sorry

end evaluation_l23_23630


namespace calculate_S2018_l23_23780

def seq_a : ℕ → ℝ
| 1     := real.sqrt 2
| (n+2) := real.sqrt ((seq_a (n+1))^2 + 2)

def seq_b (n : ℕ) : ℝ :=
  4 / ((seq_a n)^2 * (seq_a (n+1))^2)

def seq_S (n : ℕ) : ℝ :=
  ∑ i in finset.range n, seq_b (i + 1)

theorem calculate_S2018 :
  seq_S 2018 = 2018 / 2019 := 
sorry

end calculate_S2018_l23_23780


namespace Cameron_list_count_l23_23980

theorem Cameron_list_count : 
  let lower_bound := 900
  let upper_bound := 27000
  let step := 30
  let n_min := lower_bound / step
  let n_max := upper_bound / step
  n_max - n_min + 1 = 871 :=
by
  sorry

end Cameron_list_count_l23_23980


namespace smallest_angle_ratio_l23_23864

theorem smallest_angle_ratio (k : ℕ) (h1 : 2 * k + 3 * k + 4 * k = 180) : 2 * 20 = 40 :=
begin
  have h2 : k = 20,
  {
    linarith,
  },
  rw h2,
  norm_num,
end

end smallest_angle_ratio_l23_23864


namespace sarah_interviewed_students_l23_23050

theorem sarah_interviewed_students :
  let oranges := 70
  let pears := 120
  let apples := 147
  let strawberries := 113
  oranges + pears + apples + strawberries = 450 := by
sorry

end sarah_interviewed_students_l23_23050


namespace titmice_all_on_one_tree_l23_23120

-- Define the problem conditions
def titmice (n : ℕ) (m : ℕ) := List (Fin n) 

-- Define the bird movement mechanics
def move_titmouse (config : titmice 2021 120) (i j : Fin 120) : titmice 2021 120 :=
  if config.get? i > config.get? j then config else sorry

-- Define the finite number of moves
def finite_moves : nat := sorry

-- Define the proof problem
theorem titmice_all_on_one_tree :
  ∀ config : titmice 2021 120,
  ∃ moves : nat, 
  ∃ final_config : titmice 2021 120, 
  (∀ i j : Fin 120, final_config.get? i = final_config.get? j) :=
begin
   sorry  
end

end titmice_all_on_one_tree_l23_23120


namespace units_digit_factorial_sum_l23_23272

theorem units_digit_factorial_sum : 
  (∑ n in (Finset.range 2024), (nat.factorial n) % 10) % 10 = 3 := 
by 
  sorry

end units_digit_factorial_sum_l23_23272


namespace line_through_points_slope_intercept_sum_l23_23857

theorem line_through_points_slope_intercept_sum :
  ∃ m b : ℝ, (∀ x y : ℝ, (y = m * x + b) → ((((x, y) = (-3, 1)) ∨ ((x, y) = (1, 3))) ⇒ y = m * x + b)) ∧ (m + b = 3) :=
begin
  sorry
end

end line_through_points_slope_intercept_sum_l23_23857


namespace minimum_area_triangle_OAB_l23_23775

-- Given
variable (b : ℝ) (k : ℝ)
variable (h1 : b > 0) (h2 : k ≠ 0)
variable (S : ℝ)
variable (A B : ℝ × ℝ)
variable (h3 : A = (-b / k, 0)) (h4 : B = (0, b))
variable (h5 : S = 1 / 2 * b * (-b / k))

-- Proving
theorem minimum_area_triangle_OAB :
  (∀ (b : ℝ), b > 2 → (∃ k : ℝ, k = (2 * b - b ^ 2) / (2 * (b + 3)))) → 
  (min {S : ℝ | S = (1 / 2 * b * (-b / k))} = 7 + 2 * real.sqrt 10) :=
by 
  sorry

end minimum_area_triangle_OAB_l23_23775


namespace inequality_2x1_plus_x2_gt_e_l23_23799

open Real

-- Define the function f
def f (x m : ℝ) : ℝ := ln x + m * x

-- Parts (Ⅰ) and (Ⅱ) can be derived specifically, but we focus on part (Ⅲ) for this example
theorem inequality_2x1_plus_x2_gt_e (m n : ℝ) (x1 x2 : ℝ) (h1 : x1 < x2) 
  (h2 : f x1 m = (m + 1) * x1 + n - 2)
  (h3 : f x2 m = (m + 1) * x2 + n - 2) : 
  2 * x1 + x2 > exp 1 :=
by
  -- Here would be the proof, but we're adding a placeholder
  sorry

end inequality_2x1_plus_x2_gt_e_l23_23799


namespace problem_l23_23868

theorem problem (
  h1 : 4 * 7 + 5 = 33,
  h2 : 300 / 6 = 50
) : 
  4 * 7 + 5 = 33 ∧ 50 * 6 = 300 := by
  split
  exact h1
  rw ← h2
  norm_num
  sorry

end problem_l23_23868


namespace tangent_line_at_one_l23_23701

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - 3 * x^2 + x + b

theorem tangent_line_at_one (a b : ℝ) (h₁ : a ≠ 0) (h₂ : Deriv (f a b) 1 = -2) (h₃ : f a b 1 = -3) : 
  f a b = λ x, x^3 - 3 * x^2 + x - 2 := by
sorry

end tangent_line_at_one_l23_23701


namespace length_AB_l23_23328

theorem length_AB 
  (P : ℝ × ℝ) 
  (hP : 3 * P.1 + 4 * P.2 + 8 = 0)
  (C : ℝ × ℝ := (1, 1))
  (A B : ℝ × ℝ)
  (hA : (A.1 - 1)^2 + (A.2 - 1)^2 = 1 ∧ (3 * A.1 + 4 * A.2 + 8 ≠ 0))
  (hB : (B.1 - 1)^2 + (B.2 - 1)^2 = 1 ∧ (3 * B.1 + 4 * B.2 + 8 ≠ 0)) :
  dist A B = 4 * Real.sqrt 2 / 3 := sorry

end length_AB_l23_23328


namespace uncle_ben_parking_probability_l23_23203

theorem uncle_ben_parking_probability :
  let total_spaces := 20
  let cars := 15
  let rv_spaces := 3
  let total_combinations := Nat.choose total_spaces cars
  let non_adjacent_empty_combinations := Nat.choose (total_spaces - rv_spaces) cars
  (1 - (non_adjacent_empty_combinations / total_combinations)) = (232 / 323) := by
  sorry

end uncle_ben_parking_probability_l23_23203


namespace inequality_proof_l23_23291

theorem inequality_proof
  (a b c d : ℝ)
  (h1 : 0 < a)
  (h2 : a ≤ b)
  (h3 : b ≤ c)
  (h4 : c ≤ d)
  (h5 : a + b + c + d = 1) :
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 := by
  sorry

end inequality_proof_l23_23291


namespace p_value_for_roots_l23_23710

theorem p_value_for_roots (α β : ℝ) (h1 : 3 * α^2 + 5 * α + 2 = 0) (h2 : 3 * β^2 + 5 * β + 2 = 0)
  (hαβ : α + β = -5/3) (hαβ_prod : α * β = 2/3) : p = -49/9 :=
by
  sorry

end p_value_for_roots_l23_23710


namespace polynomial_remainder_correct_l23_23264

noncomputable def polynomial_division_remainder : Prop :=
  let dividend : Polynomial ℚ := Polynomial.C 1 + Polynomial.X ^ 4
  let divisor : Polynomial ℚ := Polynomial.C 4 + Polynomial.X - Polynomial.X ^ 2
  let expected_remainder : Polynomial ℚ := Polynomial.C 1 - Polynomial.C 8 * Polynomial.X
  Polynomial.mod_by_monic dividend divisor = expected_remainder

theorem polynomial_remainder_correct : polynomial_division_remainder :=
  sorry

end polynomial_remainder_correct_l23_23264


namespace max_expression_value_l23_23342

noncomputable def A : ℝ := 15682 + (1 / 3579)
noncomputable def B : ℝ := 15682 - (1 / 3579)
noncomputable def C : ℝ := 15682 * (1 / 3579)
noncomputable def D : ℝ := 15682 / (1 / 3579)
noncomputable def E : ℝ := 15682.3579

theorem max_expression_value :
  D = 56109138 ∧ D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end max_expression_value_l23_23342


namespace log2_T_eq_1009_l23_23413

-- Define the function representing the given expansion
def expansion : ℂ → ℤ → ℂ :=
  λ z n, (z + I * 1) ^ n

-- Define T as the sum of all the real coefficients of the expansion of (1 + I*x)^2018
noncomputable def T : ℂ :=
  (expansion 1 2018).re

theorem log2_T_eq_1009 : Real.log2 T = 1009 :=
sorry

end log2_T_eq_1009_l23_23413


namespace charley_initial_pencils_l23_23224

theorem charley_initial_pencils (P : ℕ) (lost_initially : P - 6 = (P - 1/3 * (P - 6) - 6)) (current_pencils : P - 1/3 * (P - 6) - 6 = 16) : P = 30 := 
sorry

end charley_initial_pencils_l23_23224


namespace conditional_prob_correct_l23_23128

/-- Define the events A and B as per the problem -/
def event_A (x y : ℕ) : Prop := (x + y) % 2 = 0

def event_B (x y : ℕ) : Prop := (x % 2 = 0 ∨ y % 2 = 0) ∧ x ≠ y

/-- Define the probability of event A -/
def prob_A : ℚ := 1 / 2

/-- Define the combined probability of both events A and B occurring -/
def prob_A_and_B : ℚ := 1 / 6

/-- Calculate the conditional probability P(B | A) -/
def conditional_prob : ℚ := prob_A_and_B / prob_A

theorem conditional_prob_correct : conditional_prob = 1 / 3 := by
  -- This is where you would provide the proof if required
  sorry

end conditional_prob_correct_l23_23128


namespace parabola_focus_l23_23481

theorem parabola_focus (a : ℝ) (ha : a ≠ 0) : 
  ∃ (x y : ℝ), (x = 0) ∧ (y = 1 / (16 * a)) ∧ (y = 4 * a * x ^ 2) :=
by
  existsi 0
  existsi 1 / (16 * a)
  split
  { refl }
  split
  { sorry }
  { sorry }

end parabola_focus_l23_23481


namespace find_n_l23_23375

theorem find_n (n : ℕ) (h : 2^n = 2 * 16^2 * 4^3) : n = 15 :=
by
  sorry

end find_n_l23_23375


namespace arithmetic_progression_a6_l23_23309

theorem arithmetic_progression_a6 (a1 d : ℤ) (h1 : a1 + (a1 + d) + (a1 + 2 * d) = 168) (h2 : (a1 + 4 * d) - (a1 + d) = 42) : 
  a1 + 5 * d = 3 := 
sorry

end arithmetic_progression_a6_l23_23309


namespace cubic_roots_identity_l23_23829

theorem cubic_roots_identity (x1 x2 p q : ℝ) 
  (h1 : x1^2 + p * x1 + q = 0) 
  (h2 : x2^2 + p * x2 + q = 0) :
  (x1^3 + x2^3 = 3 * p * q - p^3) ∧ 
  (x1^3 - x2^3 = (p^2 - q) * Real.sqrt (p^2 - 4 * q) ∨ 
   x1^3 - x2^3 = -(p^2 - q) * Real.sqrt (p^2 - 4 * q)) :=
by
  sorry

end cubic_roots_identity_l23_23829


namespace limit_sum_evaluation_l23_23247

theorem limit_sum_evaluation :
  (∀ n : ℕ, 1 ≤ n →
    (∑ r in finset.range (n + 1), ∑ s in finset.range (n + 1), (5 * r^4 - 18 * r^2 * s^2 + 5 * s^4)) / (n^5 : ℝ)) →
  tendsto (λ n, ∑ r in finset.range (n + 1), ∑ s in finset.range (n + 1), (5 * r^4 - 18 * r^2 * s^2 + 5 * s^4) / (n^5 : ℝ)) at_top (𝓝 (-1)) :=
by sorry

end limit_sum_evaluation_l23_23247


namespace sum_binomial_coefficients_of_expansion_eq_64_l23_23478

theorem sum_binomial_coefficients_of_expansion_eq_64 
  (T5_is_constant : ∀ Cn_4 n, Cn_4 * (2: ℝ)^ (n-4) * (x: ℝ)^ (n-6) = Cn_4 * 2^ (n-4) * x^ (n-6) ∧ x ≠ 0):
  ∀ (n: ℕ), n = 6 → (2 ^ n) = 64 := 
by
  intros n hn
  rw hn
  -- where we expect the calculation for 2^6
  have : 2 ^ 6 = 64 := by norm_num
  exact this

-- If added, this condition should support continuity in proof
#check sum_binomial_coefficients_of_expansion_eq_64

end sum_binomial_coefficients_of_expansion_eq_64_l23_23478


namespace total_pears_sold_l23_23204

theorem total_pears_sold (sold_morning : ℕ) (sold_afternoon : ℕ) (h_morning : sold_morning = 120) (h_afternoon : sold_afternoon = 240) :
  sold_morning + sold_afternoon = 360 :=
by
  sorry

end total_pears_sold_l23_23204


namespace units_digit_sum_of_factorials_is_3_l23_23269

theorem units_digit_sum_of_factorials_is_3 :
  (∑ k in Finset.range 2024, Nat.factorial k) % 10 = 3 :=
by
  sorry

end units_digit_sum_of_factorials_is_3_l23_23269


namespace range_of_a_l23_23016

noncomputable def f (x : ℝ) : ℝ := sorry

theorem range_of_a :
  (∀ x : ℝ, f (x + 5) = f x) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (f 2 > 1) ∧ 
  (f 3 = (λ a : ℝ, (a^2 + a + 3) / (a - 3)) a) ↔ 
  (a ∈ Iio (-2) ∪ Ioo 0 3) :=
sorry

end range_of_a_l23_23016


namespace possible_values_of_derivative_l23_23807

noncomputable def differentiable_function_condition (f : ℝ → ℝ) := 
  (0 < ∀ (x : ℝ), x < 1 → differentiable_at ℝ f x) ∧ 
  (∀ (n : ℕ), ∀ (a : ℕ), odd a ∧ 0 < a ∧ a < 2^n →
    ∃ (b : ℕ), odd b ∧ b < 2^n ∧ f (a / 2^n : ℝ) = b / 2^n)

theorem possible_values_of_derivative (f : ℝ → ℝ) (hf : differentiable_function_condition f) : 
  f' (1 / 2 : ℝ) ∈ {-1, 1} :=
sorry

end possible_values_of_derivative_l23_23807


namespace option_b_option_c_option_d_l23_23150

theorem option_b (x : ℝ) (h : x > 1) : (∀ y, y = 2*x + 4 / (x - 1) - 1 → y ≥ 4*Real.sqrt 2 + 1) :=
by
  sorry

theorem option_c (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3 * x * y) : 2*x + y ≥ 3 :=
by
  sorry

theorem option_d (x y : ℝ) (h : 9*x^2 + y^2 + x*y = 1) : 3*x + y ≤ 2*Real.sqrt 21 / 7 :=
by
  sorry

end option_b_option_c_option_d_l23_23150


namespace limiting_reactant_and_product_moles_l23_23752

noncomputable def moles_HC2H3O2 : ℝ := 3
noncomputable def moles_NaHCO3 : ℝ := 2.5
noncomputable def percent_yield : ℝ := 0.85

def balanced_reaction (moles_HC2H3O2 moles_NaHCO3 : ℝ) : Prop :=
  moles_HC2H3O2 = moles_NaHCO3

theorem limiting_reactant_and_product_moles :
  (∀ moles_HC2H3O2 moles_NaHCO3, balanced_reaction moles_HC2H3O2 moles_NaHCO3) →
  moles_NaHCO3 < moles_HC2H3O2 →
  let theoretical_yield := moles_NaHCO3 in
  let actual_yield := theoretical_yield * percent_yield in
  actual_yield = 2.125 ∧ moles_HC2H3O2 > moles_NaHCO3 :=
by
  intros hreaction hlimiting
  let theoretical_yield := moles_NaHCO3
  let actual_yield := theoretical_yield * percent_yield
  have hactual_yield : actual_yield = 2.125 := sorry
  exact ⟨hactual_yield, hlimiting⟩

end limiting_reactant_and_product_moles_l23_23752


namespace number_of_intersection_points_l23_23230

theorem number_of_intersection_points (A : ℝ) (hA : A > 0) :
  ∃ (P : set (ℝ × ℝ)), P.countable ∧ P.card = 4 ∧
  ∀ p ∈ P, (∃ x y : ℝ, y = A * x^2 ∧ y^2 + 3 = x^2 + 4 * y) :=
by
  sorry

end number_of_intersection_points_l23_23230


namespace f_is_odd_f_is_increasing_f_range_l23_23421

-- Condition that a > 1
variables (a : ℝ) (ha : a > 1)

-- Definition of the function f(x) = (a^x - 1) / (a^x + 1)
def f (x : ℝ) : ℝ := (a^x - 1) / (a^x + 1)

-- Problem 1: f(x) is an odd function
theorem f_is_odd : ∀ (x : ℝ), f a (-x) = -f a x :=
by
  sorry

-- Problem 2: f(x) is an increasing function on ℝ
theorem f_is_increasing : ∀ (x y : ℝ), x < y → f a x < f a y :=
by
  sorry

-- Problem 3: The range of f(x) is (-1, 1)
theorem f_range : ∀ (y : ℝ), ∃ (x : ℝ), f a x = y ↔ y > -1 ∧ y < 1 :=
by
  sorry

end f_is_odd_f_is_increasing_f_range_l23_23421


namespace votes_lost_by_l23_23576

theorem votes_lost_by (total_votes : ℕ) (candidate_percentage : ℕ) : total_votes = 20000 → candidate_percentage = 10 → 
  (total_votes * candidate_percentage / 100 - total_votes * (100 - candidate_percentage) / 100 = 16000) :=
by
  intros h_total_votes h_candidate_percentage
  have vote_candidate := total_votes * candidate_percentage / 100
  have vote_rival := total_votes * (100 - candidate_percentage) / 100
  have votes_diff := vote_rival - vote_candidate
  rw [h_total_votes, h_candidate_percentage] at *
  sorry

end votes_lost_by_l23_23576


namespace value_of_x_add_y_l23_23682

theorem value_of_x_add_y (x y : ℝ) 
  (h1 : x + Real.sin y = 2023)
  (h2 : x + 2023 * Real.cos y = 2021)
  (h3 : (Real.pi / 4) ≤ y ∧ y ≤ (3 * Real.pi / 4)) : 
  x + y = 2023 - (Real.sqrt 2) / 2 + (3 * Real.pi) / 4 := 
sorry

end value_of_x_add_y_l23_23682


namespace quotient_is_10_l23_23824

theorem quotient_is_10 (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 161)
  (h2 : divisor = 16)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) : 
  quotient = 10 := 
by
  sorry

end quotient_is_10_l23_23824


namespace suitable_chart_for_rope_skipping_scores_l23_23919

-- Definitions
inductive ChartType
| Bar
| Line
| Pie

-- Conditions
def suitableChartToRepresentRopeSkippingScores (ct : ChartType) : Prop :=
  ct = ChartType.Bar

-- Statement
theorem suitable_chart_for_rope_skipping_scores : ∃ ct : ChartType, suitableChartToRepresentRopeSkippingScores ct :=
by
  use ChartType.Bar
  triv
  sorry

end suitable_chart_for_rope_skipping_scores_l23_23919


namespace part1_part2_l23_23688

-- Define the quadrilateral and conditions
variables {α : Type} [linear_ordered_field α]

structure Square (α : Type) :=
(a b c d p q : α)

noncomputable def geometric_condition (sq : Square α) (AB CD : α) (r1 r2 r3 : α) : Prop :=
  AB = 1 ∧
  (sq.a ≠ sq.b ∧ sq.b ≠ sq.c ∧ sq.c ≠ sq.d ∧ sq.d ≠ sq.a) ∧
  sq.p ∈ set.Ioo sq.c sq.d ∧
  sq.q ∈ set.Ioo sq.b sq.c

-- Theorem statement for part 1
theorem part1 (sq : Square α) (r1 r2 r3 : α) :
  geometric_condition sq 1 ∧ r1^2 = (r2 + r3)^2 →
  r1^2 ≥ 4 * r2 * r3 ∧ (r1^2 = 4 * r2 * r3 → sq.p = (sq.c + sq.d) / 2) :=
sorry

-- Theorem statement for part 2
theorem part2 (sq : Square α) (r1 r2 r3 : α) :
  geometric_condition sq 1 r1 r2 r3 →
  3 - 2 * real.sqrt 2 < r1^2 + r2^2 + r3^2 ∧ r1^2 + r2^2 + r3^2 < 1 / 2 :=
sorry

end part1_part2_l23_23688


namespace smallest_whole_number_above_perimeter_triangle_l23_23139

theorem smallest_whole_number_above_perimeter_triangle (s : ℕ) (h1 : 12 < s) (h2 : s < 26) :
  53 = Nat.ceil ((7 + 19 + s : ℕ) / 1) := by
  sorry

end smallest_whole_number_above_perimeter_triangle_l23_23139


namespace smallest_integer_with_remainders_l23_23967

theorem smallest_integer_with_remainders :
  ∃ n > 1, (n % 4 = 1) ∧ (n % 5 = 1) ∧ (n % 6 = 1) ∧ n = 61 :=
begin
  sorry
end

end smallest_integer_with_remainders_l23_23967


namespace simplify_sqrt_mul_cubert_l23_23465

theorem simplify_sqrt_mul_cubert:
  sqrt 18 * cbrt 24 = 6 * 2^(1/2 : ℝ) * 3^(1/3 : ℝ) :=
sorry

end simplify_sqrt_mul_cubert_l23_23465


namespace fraction_not_integer_l23_23792

theorem fraction_not_integer (a b : ℕ) (h : a ≠ b) (parity: (a % 2 = b % 2)) 
(h_pos_a : 0 < a) (h_pos_b : 0 < b) : ¬ ∃ k : ℕ, (a! + b!) = k * 2^a := 
by sorry

end fraction_not_integer_l23_23792


namespace quadrilateral_circumscribed_l23_23407

structure ConvexQuad (A B C D : Type) := 
  (is_convex : True)
  (P : Type)
  (interior : True)
  (angle_APB_angle_CPD_eq_angle_BPC_angle_DPA : True)
  (angle_PAD_angle_PCD_eq_angle_PAB_angle_PCB : True)
  (angle_PDC_angle_PBC_eq_angle_PDA_angle_PBA : True)

theorem quadrilateral_circumscribed (A B C D : Type) (quad : ConvexQuad A B C D) : True := 
sorry

end quadrilateral_circumscribed_l23_23407


namespace angle_BMD_right_angle_l23_23209

section Parallelogram

variables {A B C D K M : Point}
variables [parallelogram A B C D]

-- Defining the conditions
def K_point (AK BD : ℝ) : Prop := AK = BD
def midpoint (M CK : Segment) : Prop := ⟪M, CK⟫

-- Goal: Prove that the angle ∠BMD = 90°
theorem angle_BMD_right_angle (AK BD : ℝ) (H1 : K_point A K BD) (H2 : midpoint M C K) : angle B M D = 90 :=
sorry

end Parallelogram

end angle_BMD_right_angle_l23_23209


namespace max_弄_is_9_l23_23776

constant char_num : Type
constant 表 : char_num
constant 一 : char_num
constant 故 : char_num
constant 如 : char_num
constant 虚 : char_num
constant 弄 : char_num
constant idiom : char_num → char_num → char_num → char_num → Prop

axiom idiom1 : idiom 虚 (-) (-) 表
axiom idiom2 : idiom 表 (-) (-) 一
axiom idiom3 : idiom 一 (-) (-) 故
axiom idiom4 : idiom 故 弄 (-) 虚

axiom sum_idiom : ∀ x y z w : char_num, idiom x y z w → x + y + z + w = 21

axiom order : 表 > 一 ∧ 一 > 故 ∧ 故 > 如 ∧ 如 > 虚

axiom unique_numbers : ∀ x y : char_num, x ≠ y → x ≠ y

constant max_num : char_num

noncomputable def maximum_弄 : char_num :=
maximize (char_num := 弄)

theorem max_弄_is_9 : max_num = 9 := sorry

end max_弄_is_9_l23_23776


namespace digit_b_divisible_by_7_l23_23090

theorem digit_b_divisible_by_7 (B : ℕ) (h : 0 ≤ B ∧ B ≤ 9) 
  (hdiv : (4000 + 110 * B + 3) % 7 = 0) : B = 0 :=
by
  sorry

end digit_b_divisible_by_7_l23_23090


namespace f_1001_value_l23_23877

noncomputable def f : ℕ → ℝ := sorry

theorem f_1001_value :
  (∀ a b n : ℕ, a + b = 2^n → f a + f b = n^2) →
  f 1 = 1 →
  f 1001 = 83 :=
by
  intro h₁ h₂
  sorry

end f_1001_value_l23_23877


namespace volleyball_tournament_inequality_l23_23072

theorem volleyball_tournament_inequality
  (x : ℕ → ℕ)  -- Points scored by each team, indexed from 1 to 10
  (h : ∑ i in finRange 10, x i = 45) :  -- Sum of points is 45, from the 45 matches
  (∑ k in finRange 10, k * (x k) ≥ 165) :=  -- Our goal, the weighted sum
sorry

end volleyball_tournament_inequality_l23_23072


namespace line_through_points_l23_23855

theorem line_through_points (x1 y1 x2 y2 : ℝ) (m b : ℝ) 
  (h1 : x1 = -3) (h2 : y1 = 1) (h3 : x2 = 1) (h4 : y2 = 3)
  (h5 : y1 = m * x1 + b) (h6 : y2 = m * x2 + b) :
  m + b = 3 := 
sorry

end line_through_points_l23_23855


namespace probability_abs_diff_two_l23_23654

-- Define the set of numbers we are considering
def number_set : set ℕ := {5, 6, 7, 8}

-- Define the property of pairs having an absolute difference of 2
def abs_diff_two (a b : ℕ) : Prop := abs (a - b) = 2

-- Calculate total number of ways to pick 2 different numbers from the set
def total_pairs := (number_set.card * (number_set.card - 1)) / 2

-- Define the set of pairs that have an absolute difference of 2
def valid_pairs := {pair | pair ∈ (number_set.prod number_set) ∧ abs_diff_two pair.1 pair.2 ∧ pair.1 ≠ pair.2}

-- The probability calculation as a fraction of valid pairs to total pairs
def probability := (valid_pairs.card : ℚ) / total_pairs

-- The theorem we need to prove
theorem probability_abs_diff_two : probability = 1 / 3 :=
by
  sorry

end probability_abs_diff_two_l23_23654


namespace find_x_squared_inverse_squared_l23_23930

variable (x : ℝ)

theorem find_x_squared_inverse_squared (h : x + 1 / x = 2.5) :
  x^2 + 1 / x^2 = 4.25 :=
begin
  -- Proof goes here
  sorry
end

end find_x_squared_inverse_squared_l23_23930


namespace dice_sums_l23_23889

open Finset

-- Define the finite types for the dice values
def Dice := Fin 6

-- Function to calculate the number of valid combinations for a given sum
def numberOfCombinations (sum : ℕ) : ℕ :=
  (univ : Finset (Dice × Dice × Dice)).filter (λ (abc : Dice × Dice × Dice), abc.1.val + 1 + abc.2.val + 1 + abc.3.val + 1 = sum).card

-- The theorem statement, proving the counts for both sums
theorem dice_sums : numberOfCombinations 5 = 6 ∧ numberOfCombinations 6 = 10 :=
by
  sorry

end dice_sums_l23_23889


namespace cameron_list_count_l23_23984

theorem cameron_list_count : 
  (∃ (n m : ℕ), n = 900 ∧ m = 27000 ∧ (∀ k : ℕ, (30 * k) ≥ n ∧ (30 * k) ≤ m → ∃ count : ℕ, count = 871)) :=
by
  sorry

end cameron_list_count_l23_23984


namespace emily_stickers_l23_23914

theorem emily_stickers:
  ∃ S : ℕ, (S % 4 = 2) ∧
           (S % 6 = 2) ∧
           (S % 9 = 2) ∧
           (S % 10 = 2) ∧
           (S > 2) ∧
           (S = 182) :=
  sorry

end emily_stickers_l23_23914


namespace exists_permutation_adjacent_diff_2_or_3_exists_permutation_100_adjacent_diff_2_or_3_l23_23552

open Finset
open Function

-- Part (a)
theorem exists_permutation_adjacent_diff_2_or_3 :
  ∃ (σ : Perm (Fin 8)), ∀ i : Fin 8, |(σ (i + 1) - σ i)| = 2 ∨ |(σ (i + 1) - σ i)| = 3 :=
sorry

-- Part (b)
theorem exists_permutation_100_adjacent_diff_2_or_3 :
  ∃ (σ : Perm (Fin 100)), ∀ i : Fin 99, |(σ (i + 1) - σ i)| = 2 ∨ |(σ (i + 1) - σ i)| = 3 :=
sorry

end exists_permutation_adjacent_diff_2_or_3_exists_permutation_100_adjacent_diff_2_or_3_l23_23552


namespace count_valid_two_digit_integers_l23_23323

def digits : Set ℕ := {3, 5, 7, 9}

def valid_two_digit_integers (n : ℕ) : Prop :=
  n ∈ {x * 10 + y | x ∈ digits ∧ y ∈ digits ∧ x ≠ y ∧ x + y > 10}

theorem count_valid_two_digit_integers :
  (Finset.card (Finset.filter valid_two_digit_integers (Finset.range 100))) = 4 :=
by
  sorry

end count_valid_two_digit_integers_l23_23323


namespace infinite_equal_pairs_l23_23321

theorem infinite_equal_pairs
  (a : ℤ → ℝ)
  (h : ∀ k : ℤ, a k = 1/4 * (a (k - 1) + a (k + 1)))
  (k p : ℤ) (hne : k ≠ p) (heq : a k = a p) :
  ∃ infinite_pairs : ℕ → (ℤ × ℤ), 
  (∀ n : ℕ, (infinite_pairs n).1 ≠ (infinite_pairs n).2) ∧
  (∀ n : ℕ, a (infinite_pairs n).1 = a (infinite_pairs n).2) :=
sorry

end infinite_equal_pairs_l23_23321


namespace prop2_prop3_l23_23361

variables {m n : Type*} [line m] [line n]
variables {α β : Type*} [plane α] [plane β]

-- Definitions of geometric relations
def is_subset (l : Type*) (π : Type*) [line l] [plane π] := sorry
def is_perpendicular (l : Type*) (π : Type*) [line l] [plane π] := sorry
def is_parallel (x y : Type*) := sorry

-- Proposition 2: If α is parallel to β and m is a subset of α, then m is parallel to β.
theorem prop2 (α β : Type*) [plane α] [plane β] (m : Type*) [line m] 
  (h1 : is_parallel α β) (h2 : is_subset m α) : is_parallel m β := 
sorry

-- Proposition 3: If n is perpendicular to α, n is perpendicular to β, and m is perpendicular to α, then m is perpendicular to β.
theorem prop3 (n m : Type*) [line n] [line m] (α β : Type*) [plane α] [plane β]
  (h1 : is_perpendicular n α) (h2 : is_perpendicular n β) (h3 : is_perpendicular m α) 
  : is_perpendicular m β := 
sorry

end prop2_prop3_l23_23361


namespace min_value_problem_inequality_solution_l23_23696

-- Definition of the function
noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Part (i): Minimum value problem
theorem min_value_problem (a : ℝ) (minF : ∀ x : ℝ, f x a ≥ 2) : a = 0 ∨ a = -4 :=
by
  sorry

-- Part (ii): Inequality solving problem
theorem inequality_solution (x : ℝ) (a : ℝ := 2) : f x a ≤ 6 ↔ -3 ≤ x ∧ x ≤ 3 :=
by
  sorry

end min_value_problem_inequality_solution_l23_23696


namespace village_population_growth_l23_23387

theorem village_population_growth (P A : ℕ) 
  (h1 : 0.60 * P = A)
  (h2 : 0.70 * A = 18000)
  (h3 : 5% growth in adults : A_next = A + 0.05 * A) :
  P_next = 45000 :=
by
  sorry

end village_population_growth_l23_23387


namespace annual_decrease_due_to_migration_l23_23870

theorem annual_decrease_due_to_migration :
  ∃ x : ℝ, (1 + 22.5043 / 100) = (1 + (8 - x) / 100)^3 ∧ abs (x - 0.75) < 0.01 :=
begin
  sorry
end

end annual_decrease_due_to_migration_l23_23870


namespace range_of_a_l23_23357

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a < x ∧ x < a + 1) → (-2 ≤ x ∧ x ≤ 2)) ↔ -2 ≤ a ∧ a ≤ 1 :=
by 
  sorry

end range_of_a_l23_23357


namespace sum_of_possible_values_of_x_in_isosceles_triangle_l23_23895

theorem sum_of_possible_values_of_x_in_isosceles_triangle (x : ℝ) (h_isosceles : ∃ (a b : ℝ), a = 60 ∧ isosceles_triangle a b x) : (∑ y in { y | ∃(a b : ℝ), a = 60 ∧ isosceles_triangle a b y }, y) = 180 :=
by sorry

end sum_of_possible_values_of_x_in_isosceles_triangle_l23_23895


namespace wolf_hunger_if_eats_11_kids_l23_23935

variable (p k : ℝ)  -- Define the satiety values of a piglet and a kid.
variable (H : ℝ)    -- Define the satiety threshold for "enough to remove hunger".

-- Conditions from the problem:
def condition1 : Prop := 3 * p + 7 * k < H  -- The wolf feels hungry after eating 3 piglets and 7 kids.
def condition2 : Prop := 7 * p + k > H      -- The wolf suffers from overeating after eating 7 piglets and 1 kid.

-- Statement to prove:
theorem wolf_hunger_if_eats_11_kids (p k H : ℝ) 
  (h1 : condition1 p k H) (h2 : condition2 p k H) : 11 * k < H :=
by
  sorry

end wolf_hunger_if_eats_11_kids_l23_23935


namespace range_of_function_l23_23286

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x ≤ π / 3) :
    ∀ y, y = sin (x + π / 3) + sin (x - π / 3) + sqrt 3 * cos x + 1 ↔ (√3 + 1 ≤ y ∧ y ≤ 3) :=
begin
  sorry
end

end range_of_function_l23_23286


namespace graph_independent_set_l23_23410

noncomputable def f (G : Type) (V : set G) (d : G → ℕ) : ℝ :=
  ∑ v in V, (1 : ℝ) / (1 + d v)

theorem graph_independent_set (G : Type) [fintype G] [decidable_eq G] (V : set G) (d : G → ℕ) :
  ∃ I : set G, ∀ v1 v2 ∈ I, v1 ≠ v2 → ¬(∃ e, e ∈ (edges G) ∧ v1 ∈ e ∧ v2 ∈ e) ∧ I.card ≥ f G V d := sorry

end graph_independent_set_l23_23410


namespace max_value_of_4x_plus_3y_l23_23686

theorem max_value_of_4x_plus_3y (x y : ℝ) :
  x^2 + y^2 = 18 * x + 8 * y + 10 → 
  4 * x + 3 * y ≤ 63 :=
begin
  sorry
end

end max_value_of_4x_plus_3y_l23_23686


namespace day_53_days_from_thursday_is_monday_l23_23525

def day_of_week : Type := {n : ℤ // n % 7 = n}

def Thursday : day_of_week := ⟨4, by norm_num⟩
def Monday : day_of_week := ⟨1, by norm_num⟩

theorem day_53_days_from_thursday_is_monday : 
  (⟨(4 + 53) % 7, by norm_num⟩ : day_of_week) = Monday := 
by 
  sorry

end day_53_days_from_thursday_is_monday_l23_23525


namespace a_is_perfect_square_l23_23037

theorem a_is_perfect_square (a b : ℕ) (h : ab ∣ (a^2 + b^2 + a)) : (∃ k : ℕ, a = k^2) :=
sorry

end a_is_perfect_square_l23_23037


namespace problem_solution_l23_23633

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def distinct_digits (n : ℕ) : Prop :=
  let digits := nat.digits 10 n
  list.nodup digits

def satisfies_conditions (N : ℕ) : Prop :=
  let two_digit_combinations := (list.erase (nat.digits 10 N) (λ _ , λ l, list.length l > 7))
  N > 10^8 ∧ N < 10^9 ∧ distinct_digits N ∧
  list.length (list.filter is_prime (list.map (λ l, nat.of_digits 10 l) two_digit_combinations)) ≤ 1

theorem problem_solution : satisfies_conditions 391524680 :=
sorry

end problem_solution_l23_23633


namespace Layla_Kristin_hockey_games_l23_23404

theorem Layla_Kristin_hockey_games
  (layla_goals : ℕ)
  (kristin_fewer_goals : ℕ)
  (average_goals : ℕ)
  (total_games : ℕ)
  (H1: layla_goals = 104)
  (H2: kristin_fewer_goals = 24)
  (H3: average_goals = 92)
  (Number_of_Games_eq : total_games = 2) :
  (∀ K : ℕ, K = layla_goals - kristin_fewer_goals → 
    let total_goals := layla_goals + K in
    average_goals = total_goals / total_games →
    total_games = 2) :=
begin
  sorry
end

end Layla_Kristin_hockey_games_l23_23404


namespace ordered_pair_solution_l23_23639

/--
Let \( \cos(30^\circ) = \frac{\sqrt{3}}{2} \) and \( \sec(30^\circ) = \frac{2\sqrt{3}}{3} \).
Prove that the ordered pair \( (x, y) \) of integers such that
\[ \sqrt{25 - 24 \cos(30^\circ)} = x + y \sec(30^\circ) \]
is \( x = 5 \) and \( y = -3 \).
-/
theorem ordered_pair_solution :
  let cos30 := Real.cos (π / 6)
  let sec30 := 1 / cos30
  ∃ (x y : ℤ), (Real.sqrt (25 - 24 * cos30) = x + y * sec30) ∧ x = 5 ∧ y = -3 := by
    -- Define known values
    have h_cos30 : cos30 = Real.sqrt 3 / 2 := by sorry
    have h_sec30 : sec30 = 2 * Real.sqrt 3 / 3 := by sorry
    -- Find ordered pair x, y
    use 5, -3
    -- Prove the main equation
    have h_step := calc
      Real.sqrt (25 - 24 * (Real.sqrt 3 / 2))
        = Real.sqrt (25 - 12 * Real.sqrt 3) : by sorry
        ... = 5 - 2 * Real.sqrt 3 : by sorry -- by completing the square and knowing (5 - 2 * sqrt(3))^2 calculation
    show 5 - 2 * Real.sqrt 3 = 5 + (-3) * (2 * Real.sqrt 3 / 3) from by sorry
    -- Conclude that x = 5 and y = -3 satisfy the equation
    exact ⟨rfl, rfl⟩

end ordered_pair_solution_l23_23639


namespace number_of_freshmen_in_sample_l23_23596

theorem number_of_freshmen_in_sample :
  let total_students := 2000
  let sophomores := 630
  let juniors := 720
  let sample_size := 200
  let freshmen := total_students - sophomores - juniors
  let sampling_ratio := total_students / sample_size
  let freshmen_sampled := freshmen / sampling_ratio
  freshmen_sampled = 65 :=
begin
  sorry
end

end number_of_freshmen_in_sample_l23_23596


namespace vector_ad_l23_23171

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]
variables (a b : V)

def point_on_side (D B C : V) : Prop :=
  ∃ (k : ℝ), k = 1 / 3 ∧ D - B = k • (C - B)

theorem vector_ad (D A B C : V) (h1 : point_on_side D B C) (h2 : A = B) (h3 : C = b) :
  D - A = 1 / 3 • (2 • a + b) :=
by
  sorry

end vector_ad_l23_23171


namespace proposition_true_l23_23147

theorem proposition_true (a b : ℝ) (h1 : 0 > a) (h2 : a > b) : (1/a) < (1/b) := 
sorry

end proposition_true_l23_23147


namespace sum_of_real_roots_f_x_eq_1_in_neg1_7_l23_23586

-- Definitions from conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def symmetric_about_2_minus_x (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = f x

def monotonic_decreasing_on_0_1 (f : ℝ → ℝ) : Prop :=
  ∀ {x y : ℝ}, 0 ≤ x → x < y → y < 1 → f x > f y

def has_real_root_in_0_1_for_neg_1 (f : ℝ → ℝ) : Prop :=
  ∃ x, 0 ≤ x ∧ x < 1 ∧ f x = -1

-- The main theorem to be proved
theorem sum_of_real_roots_f_x_eq_1_in_neg1_7 (f : ℝ → ℝ)
  (Hodd : odd_function f)
  (Hsymmetric : symmetric_about_2_minus_x f)
  (Hdec : monotonic_decreasing_on_0_1 f)
  (Hroot : has_real_root_in_0_1_for_neg_1 f) :
  (∑ x in ((λ x, x) '' (λ x, f x = 1 ∧ -1 ≤ x ∧ x ≤ 7)), x) = 12 :=
sorry

end sum_of_real_roots_f_x_eq_1_in_neg1_7_l23_23586


namespace not_enough_info_sweets_l23_23826

theorem not_enough_info_sweets
    (S : ℕ)         -- Initial number of sweet cookies.
    (initial_salty : ℕ := 6)  -- Initial number of salty cookies given as 6.
    (eaten_sweets : ℕ := 20)   -- Number of sweet cookies Paco ate.
    (eaten_salty : ℕ := 34)    -- Number of salty cookies Paco ate.
    (diff_eaten : eaten_salty - eaten_sweets = 14) -- Paco ate 14 more salty cookies than sweet cookies.
    : (∃ S', S' = S) → False :=  -- Conclusion: Not enough information to determine initial number of sweet cookies S.
by
  sorry

end not_enough_info_sweets_l23_23826


namespace range_of_k_l23_23380

variable (k : ℝ)

-- Definition of function h(x).
def h (x : ℝ) := 2 * x - k

-- Range of k such that h(x) is increasing on (1, +∞).
theorem range_of_k : (∀ x y ∈ Ioi 1, h x ≤ h y) →
  k ∈ set.Ici (-2) :=
begin
  assume h_incr,
  -- Proof is omitted.
  sorry,
end

end range_of_k_l23_23380


namespace solve_inequality_l23_23382

theorem solve_inequality (a b : ℝ) (h₁ : ∀ x, x ∈ Ioo (-3 : ℝ) (-1) ↔ x^2 + a * x + b < 0)
  : (∀ x, x ∈ Icc (-1 : ℝ) (-1 / 3) ↔ b * x^2 + a * x + 1 ≤ 0) :=
sorry

end solve_inequality_l23_23382


namespace flagstaff_height_l23_23190

theorem flagstaff_height 
  (s1 : ℝ) (s2 : ℝ) (hb : ℝ) (h : ℝ)
  (H1 : s1 = 40.25) (H2 : s2 = 28.75) (H3 : hb = 12.5) 
  (H4 : h / s1 = hb / s2) : 
  h = 17.5 :=
by
  sorry

end flagstaff_height_l23_23190


namespace min_CD_squared_diff_l23_23011

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3)) + (Real.sqrt (y + 6)) + (Real.sqrt (z + 12))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2)) + (Real.sqrt (y + 2)) + (Real.sqrt (z + 2))
noncomputable def f (x y z : ℝ) : ℝ := (C x y z) ^ 2 - (D x y z) ^ 2

theorem min_CD_squared_diff (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  f x y z ≥ 41.4736 :=
sorry

end min_CD_squared_diff_l23_23011


namespace robots_meeting_same_station_l23_23509

/-- The track AB is 16.8 meters long, with a station set every 2.4 meters. 
Two robots, A and B, start from station A at the same time, reach station B, then return, and repeatedly move between stations A and B. 
Both robots move at a speed of 0.8 meters per second. Robot A rests for 1 second at each station it reaches, while robot B does not rest. 
They stop moving after 2 minutes from their departure. 
We want to prove that they arrive at the same station (including the starting and terminal stations) 6 times. -/
theorem robots_meeting_same_station :
  let track_length := 16.8 
  let station_distance := 2.4 
  let robot_speed := 0.8 
  let rest_time := 1 
  let total_time := 120 
  let num_stations := track_length / station_distance + 1
  let segment_time_A := station_distance / robot_speed + rest_time 
  let segment_time_B := station_distance / robot_speed 
  ∀ t, 0 ≤ t ≤ total_time → 
    let num_segments_A := t / segment_time_A 
    let num_segments_B := t / segment_time_B 
    (num_segments_A = num_segments_B) → t = 24 * k → t ≤ total_time → k = 5 
    → (5 + 1 = 6) 
  := sorry

end robots_meeting_same_station_l23_23509


namespace find_x_l23_23251

theorem find_x (x : ℝ) (h : 4 ^ (Real.log x / Real.log 7) = 64) : x = 343 :=
sorry

end find_x_l23_23251


namespace cut_and_reassemble_parallelogram_l23_23129

theorem cut_and_reassemble_parallelogram 
  (P₁ P₂ : Parallelogram) 
  (common_side : Segment) 
  (h1 : P₁.area = P₂.area) 
  (h2 : common_side ∈ P₁.edges) 
  (h3 : common_side ∈ P₂.edges) : 
  ∃ (parts : List Parallelogram), rearrange parts P₁ P₂ :=
sorry

end cut_and_reassemble_parallelogram_l23_23129


namespace perimeter_equilateral_triangle_l23_23970

-- Definitions based on conditions
variables (s : ℕ)

def is_equilateral (s : ℕ) : Prop := s > 0

def is_isosceles_triangle (s b : ℕ) (perimeter : ℕ) : Prop :=
  perimeter = s + s + b

-- Lean Statement
theorem perimeter_equilateral_triangle (s : ℕ) (h₁ : is_equilateral s) 
  (h₂ : is_isosceles_triangle s 15 55) :
  3 * s = 60 :=
by
  -- Conditions and premises
  unfold is_isosceles_triangle at h₂
  rw [Nat.add_assoc, Nat.add_comm 15 s, ←Nat.add_assoc] at h₂
  have h₃ : 55 = 2 * s + 15 := h₂
  have h₄ : 2 * s = 40 := by linarith
  have h₅ : s = 20 := by linarith
  rw [h₅, Nat.mul_comm] -- Concluding step
  sorry

end perimeter_equilateral_triangle_l23_23970


namespace p_sufficient_but_not_necessary_for_q_l23_23676

def conditions (x : Real) : Prop :=
  ln x > 0 ∧ exp x > 1

def p (x : Real) : Prop := ln x > 0
def q (x : Real) : Prop := exp x > 1

theorem p_sufficient_but_not_necessary_for_q (x : Real) :
  (p x → q x) ∧ ¬ (q x → p x) :=
by 
  split
  sorry
  sorry

end p_sufficient_but_not_necessary_for_q_l23_23676


namespace find_functions_l23_23634

def satisfiesFunctionalEquation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f(f(x) + y + 1) = x + f(y) + 1

theorem find_functions (f : ℤ → ℤ) (h : satisfiesFunctionalEquation f) :
  (∀ n : ℤ, f(n) = n) ∨ (∀ n : ℤ, f(n) = -n - 2) :=
sorry

end find_functions_l23_23634


namespace pension_supplement_correct_l23_23069

noncomputable def future_value_annuity_due (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * ((1 + r)^n - 1) / r * (1 + r)

noncomputable def monthly_pension_supplement : ℝ :=
  let monthly_contribution := 7000
  let annual_contribution := 12 * monthly_contribution
  let annual_interest_rate := 0.09
  let contributions_period_years := 20
  let accumulated_amount := future_value_annuity_due annual_contribution annual_interest_rate contributions_period_years
  let distribution_period_months := 15 * 12
  accumulated_amount / distribution_period_months

theorem pension_supplement_correct :
  monthly_pension_supplement ≈ 26023.45 :=
by
  sorry

end pension_supplement_correct_l23_23069


namespace sarah_flour_total_l23_23459

noncomputable def pounds_of_flour : ℝ :=
  let rye_flour := 5
  let whole_wheat_bread_flour := 10
  let chickpea_flour_g := 1800
  let whole_wheat_pastry_flour := 2
  let all_purpose_flour_g := 500
  let grams_per_pound := 454
  let chickpea_flour := (chickpea_flour_g : ℝ) / grams_per_pound
  let all_purpose_flour := (all_purpose_flour_g : ℝ) / grams_per_pound
  rye_flour + whole_wheat_bread_flour + chickpea_flour + whole_wheat_pastry_flour + all_purpose_flour

theorem sarah_flour_total : pounds_of_flour ≈ 22.06 :=
  by
  sorry

end sarah_flour_total_l23_23459


namespace percentage_subtracted_l23_23843

-- Define the given condition
axiom subtracting_percentage (a p : ℝ) : (a - p * a) = 0.94 * a

-- The main statement to prove
theorem percentage_subtracted (a : ℝ) : p = 0.06 :=
by
  -- Use the given condition
  have h : (a - p * a) = 0.94 * a := subtracting_percentage a p
  -- Simplify to find p
  sorry

end percentage_subtracted_l23_23843


namespace problem_a_max_value_problem_b_infinitely_many_solutions_l23_23942

-- Definitions of the problem conditions
noncomputable def condition_16xyz_eq_product_square (x y z : ℝ) : Prop :=
  16 * x * y * z = (x + y)^2 * (x + z)^2

noncomputable def sum_xyz_leq_M (x y z M : ℝ) : Prop :=
  x + y + z ≤ M

-- Problem (a): Finding the maximum value M
theorem problem_a_max_value (x y z : ℝ) (h : condition_16xyz_eq_product_square x y z) : x + y + z ≤ 4 := 
  sorry

-- Existence of infinitely many positive rational solutions for problem (b)
theorem problem_b_infinitely_many_solutions : 
  ∃∞ (x y z : ℚ), condition_16xyz_eq_product_square x y z ∧ (x + y + z = 4) :=
  sorry

end problem_a_max_value_problem_b_infinitely_many_solutions_l23_23942


namespace power_function_value_l23_23741

theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h₁ : f = λ x : ℝ, x^α) (h₂ : f 3 = 27) : f 2 = 8 :=
sorry

end power_function_value_l23_23741


namespace inequality_proof_l23_23245

theorem inequality_proof (x y : ℝ) (h : |x - 2 * y| = 5) : x^2 + y^2 ≥ 5 := 
  sorry

end inequality_proof_l23_23245


namespace tangent_parallel_at_x1_area_under_curve_l23_23339

noncomputable def f (x : ℝ) : ℝ := x^2 + 2

theorem tangent_parallel_at_x1 (a m : ℝ) (h0 : f = λ x, a * x^2 + 2) (h1 : deriv f 1 = 2) : 
    f = λ x, 1 * x^2 + 2 := by
  sorry

theorem area_under_curve :
  let f := λ x : ℝ, x^2 + 2,
      g := λ x : ℝ, 3 * x,
      S := (∫ x in (1 : ℝ)..(2 : ℝ), g x - f x)
  in S = 1 / 6 :=
by
  sorry

end tangent_parallel_at_x1_area_under_curve_l23_23339


namespace concyclic_B_C_B1_C1_l23_23012

open EuclideanGeometry

variable {A B C H E F X B1 C1 : Point}

def acute_scalene_triangle (A B C : Point) : Prop := 
  is_triangle A B C ∧ acute A B C ∧ acute B C A ∧ acute C A B

def orthocenter (H A B C : Point) : Prop :=
  is_orthocenter H A B C

def foot_perpendicular (H A X : Point) (line : Line) : Prop := 
  perpendicular_line_through_point H line ∧ line_contains_point A line ∧ line_contains_point X line 

def parallel_lines (A X F E : Point) : Prop :=
  parallel (line_through A X) (line_through E F)

def line_contains_point_member (line : Line) (point : Point) : Prop :=
  line_contains_point point line

def line_intersection (B1 H XF : Line) (AC : Line) E F : Prop :=
  line_intersection B1 XF = line_intersection AC ∧ parallel (line_through B B1) (line_through A C) ∧
  line_intersection C1 XE = line_intersection AB ∧ parallel (line_through C C1) (line_through A B)

theorem concyclic_B_C_B1_C1
  (acute_scalene_triangle_ABC : acute_scalene_triangle A B C)
  (orthocenter_H : orthocenter H A B C)
  (line_intersection_E : line_intersection (line_through B H) (line_through A C) E)
  (line_intersection_E : line_intersection (line_through C H) (line_through A B) F)
  (foot_perpendicular_X : foot_perpendicular H A X (line_through E F))
  (B1_on_XF : line_contains_point_member (line_through X F) B1)
  (C1_on_XE : line_contains_point_member (line_through X E) C1)
  (B1_parallel_AC : parallel_lines B1 X A C)
  (C1_parallel_AB : parallel_lines C1 X A B):
  cyclic {B, C, B1, C1} :=
by
  sorry

end concyclic_B_C_B1_C1_l23_23012


namespace roots_of_quadratic_l23_23879

theorem roots_of_quadratic (x : ℝ) : (x - 3) ^ 2 = 25 ↔ (x = 8 ∨ x = -2) :=
by sorry

end roots_of_quadratic_l23_23879


namespace sum_is_24000_l23_23556

theorem sum_is_24000 (P : ℝ) (R : ℝ) (T : ℝ) : 
  (R = 5) → (T = 2) →
  ((P * (1 + R / 100)^T - P) - (P * R * T / 100) = 60) →
  P = 24000 :=
by
  sorry

end sum_is_24000_l23_23556


namespace jasonPears_l23_23787

-- Define the conditions
def keithPears : Nat := 47
def mikePears : Nat := 12
def totalPears : Nat := 105

-- Define the theorem stating the number of pears Jason picked
theorem jasonPears : (totalPears - keithPears - mikePears) = 46 :=
by 
  sorry

end jasonPears_l23_23787


namespace average_weight_of_all_girls_l23_23160

theorem average_weight_of_all_girls 
    (avg_weight_group1 : ℝ) (avg_weight_group2 : ℝ) 
    (num_girls_group1 : ℕ) (num_girls_group2 : ℕ) 
    (h1 : avg_weight_group1 = 50.25) 
    (h2 : avg_weight_group2 = 45.15) 
    (h3 : num_girls_group1 = 16) 
    (h4 : num_girls_group2 = 8) : 
    (avg_weight_group1 * num_girls_group1 + avg_weight_group2 * num_girls_group2) / (num_girls_group1 + num_girls_group2) = 48.55 := 
by 
    sorry

end average_weight_of_all_girls_l23_23160


namespace a_is_perfect_square_l23_23030

theorem a_is_perfect_square (a b : ℕ) (h : ∃ (k : ℕ), a^2 + b^2 + a = k * a * b) : ∃ n : ℕ, a = n^2 := by
  sorry

end a_is_perfect_square_l23_23030


namespace find_starting_number_l23_23123

theorem find_starting_number (k m : ℕ) (hk : 67 = (m - k) / 3 + 1) (hm : m = 300) : k = 102 := by
  sorry

end find_starting_number_l23_23123


namespace satisfy_inequality_l23_23254

theorem satisfy_inequality (x : ℝ) : 
  (4 * x + 2 > (x - 1) ^ 2) ∧ ((x - 1) ^ 2 > 3 * x + 6) ↔ 
  (3 + 2 * real.sqrt 10 < x) ∧ (x < (5 + 3 * real.sqrt 5) / 2) :=
sorry

end satisfy_inequality_l23_23254


namespace find_a6_l23_23310

-- Define an arithmetic progression.
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the necessary conditions given in the problem.
def conditions (a d : ℕ) : Prop :=
  (arithmetic_progression a d 1 + arithmetic_progression a d 2 + arithmetic_progression a d 3 = 168) ∧
  (arithmetic_progression a d 2 - arithmetic_progression a d 5 = 42)

-- State the theorem with the final value assertion.
theorem find_a6 (a d : ℕ) (h : conditions a (-14)) : 
  arithmetic_progression a (-14) 6 = 3 := 
sorry

end find_a6_l23_23310


namespace midpoint_trajectory_fixed_point_line_l23_23709

-- Definitions for conditions
def parabola (x y : ℝ) (p : ℝ) : Prop := y^2 = 2 * p * x
def circumscribes_triangle (parabola_eq : ℝ → ℝ → ℝ → Prop) (A B C : ℝ × ℝ) : Prop := 
  parabola_eq A.1 A.2 p ∧ parabola_eq B.1 B.2 p ∧ parabola_eq C.1 C.2 p ∧ 
  (A.2 = 0 ∧ A.1 = 0) ∧ (B.2 ≠ 0 ∧ C.2 ≠ 0) ∧ 
  (A.2 - B.2 ≠ 0) ∧ (A.2 - C.2 ≠ 0) ∧ 
  (B.2 ≠ A.2) ∧ (C.2 ≠ A.2) ∧ 
  (B.1 ≠ A.1) ∧ (C.1 ≠ A.1)

-- Problems translated to Lean statements

-- (I) Trajectory of midpoint M of hypotenuse BC
theorem midpoint_trajectory (p : ℝ) (hp : 0 < p) (A : ℝ × ℝ := (0, 0)) (B C : ℝ × ℝ) :
  circumscribes_triangle (@parabola p) A B C →
  ∃ M : ℝ × ℝ, ∀ x y, M = (x, y) → y^2 = (p / 4) * (x - 8 * p) :=
sorry

-- (II) Fixed point for line containing BC
theorem fixed_point_line (p t₀ : ℝ) (hp : 0 < p) (A : ℝ × ℝ := (t₀^2 / (2 * p), t₀)) (B C : ℝ × ℝ) :
  circumscribes_triangle (@parabola p) A B C →
  ∃ P : ℝ × ℝ, P = (2 * p + t₀^2 / (2 * p), -t₀) :=
sorry

end midpoint_trajectory_fixed_point_line_l23_23709


namespace orthocenter_on_A1C1_iff_perpendicular_l23_23005

-- Declare the structure of the problem

variables {A B C A1 C1 A' C' H : Type}

-- Hypotheses
axiom incircle_tangent_BC : A1 ∈ line_segment B C
axiom incircle_tangent_AB : C1 ∈ line_segment A B
axiom excircle_tangent_BC : A' ∈ line_ext B C
axiom excircle_tangent_AB : C' ∈ line_ext A B

-- The statement to prove
theorem orthocenter_on_A1C1_iff_perpendicular (A B C A1 C1 A' C' H : Type)
  (tan_A1_BC : A1 ∈ line_segment B C)
  (tan_C1_AB : C1 ∈ line_segment A B)
  (tan_A'_BC_ext : A' ∈ line_ext B C)
  (tan_C'_AB_ext : C' ∈ line_ext A B)
  (orthocenter_of_ABC : H = orthocenter_of A B C) :
  (H ∈ line_segment A1 C1) ↔ (is_perpendicular (line_segment A' C1) (line_segment B A)) := 
sorry

end orthocenter_on_A1C1_iff_perpendicular_l23_23005


namespace find_circles_tangent_to_axes_l23_23546

def tangent_to_axes_and_passes_through (R : ℝ) (P : ℝ × ℝ) :=
  let center := (R, R)
  (P.1 - R) ^ 2 + (P.2 - R) ^ 2 = R ^ 2

theorem find_circles_tangent_to_axes (x y : ℝ) :
  (tangent_to_axes_and_passes_through 1 (2, 1) ∧ tangent_to_axes_and_passes_through 1 (x, y)) ∨
  (tangent_to_axes_and_passes_through 5 (2, 1) ∧ tangent_to_axes_and_passes_through 5 (x, y)) :=
by {
  sorry
}

end find_circles_tangent_to_axes_l23_23546


namespace consecutive_prime_sums_l23_23978

def first15Primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def consecutiveSums (lst : List Nat) : List Nat :=
  (List.inits lst).filter (λ l => l ≠ []).map List.sum

def isPrimeAndNotSquare (n : Nat) : Prop :=
  Nat.Prime n ∧ ¬ (∃ k : Nat, k * k = n)

def primeAndNotSquareSumsCount : List Nat → Nat :=
  List.filter isPrimeAndNotSquare ⋙ List.length

theorem consecutive_prime_sums : primeAndNotSquareSumsCount (consecutiveSums first15Primes) = 6 :=
by
  sorry

end consecutive_prime_sums_l23_23978


namespace sum_of_100_consecutive_integers_l23_23145

theorem sum_of_100_consecutive_integers (n : ℕ) (S : ℕ) (hS : S = 1627384950) :
  ∃ k : ℕ, S = 100 * (k + (k + 1) + (k + 2) + ... + (k + 99)) :=
by
  sorry

end sum_of_100_consecutive_integers_l23_23145


namespace circumcircles_tangent_of_acute_triangle_l23_23508

-- Lean statement for the given proof problem
theorem circumcircles_tangent_of_acute_triangle
  (A B C : Point) -- Vertices of the triangle
  (l : Line) -- Tangent line to circumcircle of triangle ABC
  (C' A' B' : Point) -- Intersection points of tangent with sides AB, BC, CA respectively
  (H : Point) -- Orthocenter of the triangle
  (A1 B1 C1 : Point) -- Points on A'H, B'H, C'H respectively with specific distances
  (circumcircle_ABC : Circle) -- Circumcircle of triangle ABC
  (circumcircle_A1B1C1 : Circle) -- Circumcircle of triangle A1B1C1
  (tangent : Tangent l circumcircle_ABC) -- l is tangent to the circumcircle of ABC
  (intersect_A' : LineIntersect l (line_through A B) A')
  (intersect_B' : LineIntersect l (line_through B C) B')
  (intersect_C' : LineIntersect l (line_through C A) C')
  (orthocenter : Orthocenter H A B C)
  (distance_AH_A1 : distance A H = distance A A1)
  (distance_BH_B1 : distance B H = distance B B1)
  (distance_CH_C1 : distance C H = distance C C1) :
  Tangent circumcircle_ABC circumcircle_A1B1C1 :=
by
  sorry

end circumcircles_tangent_of_acute_triangle_l23_23508


namespace ny_cases_l23_23399

variable (N C T : ℕ)

-- Assume the following conditions are true
axiom h1 : C = T + 400
axiom h2 : N = 0.5 * C
axiom h3 : N + C + T = 3600

-- Prove the number of confirmed cases in New York is 800
theorem ny_cases : N = 800 :=
sorry

end ny_cases_l23_23399


namespace convert_base_five_to_ten_l23_23905

theorem convert_base_five_to_ten : ∃ n : ℕ, n = 38 ∧ (1 * 5^2 + 2 * 5^1 + 3 * 5^0 = n) :=
by
  sorry

end convert_base_five_to_ten_l23_23905


namespace largest_n_for_sin_cos_l23_23256

theorem largest_n_for_sin_cos (n : ℕ) (x : ℝ) (h_n : ∀ x : ℝ, sin x ^ n + cos x ^ n ≥ 2 / n) : n = 4 := by
  sorry   -- proof omitted

end largest_n_for_sin_cos_l23_23256


namespace BO_OE_ratio_correct_l23_23297

-- Definitions from the conditions
variable {A B C D O E : Point}
variable (ABCD : parallelogram A B C D) (angle_B : ∠B = 60)
variable (O_circumcenter : circumcenter O A B C)
variable (E_on_ext_angle_bisector : E = point_of_intersection_of_BO_with_exterior_angle_bisector_D_line O B D)
variable (BO_OE_ratio : ratio (length (segment B O)) (length (segment O E)) = 1 / 2)

-- The theorem to prove
theorem BO_OE_ratio_correct :
  ∀ (A B C D O E : Point) (ABCD : parallelogram A B C D) (angle_B : ∠B = 60)
  (O_circumcenter : circumcenter O A B C)
  (E_on_ext_angle_bisector : E = point_of_intersection_of_BO_with_exterior_angle_bisector_D_line O B D),
  ratio (length (segment B O)) (length (segment O E)) = 1 / 2 := by
  sorry

end BO_OE_ratio_correct_l23_23297


namespace quadratic_eq_coefficients_l23_23079

theorem quadratic_eq_coefficients :
  ∃ (a b c : ℤ), (a = 1 ∧ b = -1 ∧ c = 3) ∧ (∀ x : ℤ, a * x^2 + b * x + c = x^2 - x + 3) :=
by
  use 1, -1, 3
  split
  { split; refl }
  { intro x
    simp }
  sorry

end quadratic_eq_coefficients_l23_23079


namespace duty_roster_arrangements_l23_23243

theorem duty_roster_arrangements :
  ∃ (X : ℕ), (X = fintype.card {l : list (fin 6) | l.nodup ∧ (∀ (i : fin 5), ¬((l.nth_le i sorry = some 0 ∧ l.nth_le (i+1) sorry = some 1) ∨
                                                                         (l.nth_le i sorry = some 1 ∧ l.nth_le (i+1) sorry = some 0)) ∧
                                                       (¬((l.nth_le i sorry = some 2 ∧ l.nth_le (i+1) sorry = some 3) ∨
                                                          (l.nth_le i sorry = some 3 ∧ l.nth_le (i+1) sorry = some 2))))) ∧
                                                       X = 336 :=
begin
  sorry
end

end duty_roster_arrangements_l23_23243


namespace units_digit_even_product_10_to_100_l23_23913

def is_even (n : ℕ) : Prop := n % 2 = 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_even_product_10_to_100 : 
  units_digit (∏ i in Finset.filter is_even (Finset.Icc 10 100), i) = 0 :=
by
  sorry

end units_digit_even_product_10_to_100_l23_23913


namespace sasha_questions_per_hour_l23_23834

-- Define the total questions and the time she worked, and the remaining questions
def total_questions : ℕ := 60
def time_worked : ℕ := 2
def remaining_questions : ℕ := 30

-- Define the number of questions she completed
def questions_completed := total_questions - remaining_questions

-- Define the rate at which she completes questions per hour
def questions_per_hour := questions_completed / time_worked

-- The theorem to prove
theorem sasha_questions_per_hour : questions_per_hour = 15 := 
by
  -- Here we would prove the theorem, but we're using sorry to skip the proof for now
  sorry

end sasha_questions_per_hour_l23_23834


namespace sum_lent_out_l23_23601

theorem sum_lent_out (P R : ℝ) (h1 : 780 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 7) / 100) : P = 684 := 
  sorry

end sum_lent_out_l23_23601


namespace nat_perfect_square_l23_23039

theorem nat_perfect_square (a b : ℕ) (h : ∃ k : ℕ, a^2 + b^2 + a = k * a * b) : ∃ m : ℕ, a = m * m := by
  sorry

end nat_perfect_square_l23_23039


namespace tangent_line_at_A_extreme_values_l23_23706

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a * Real.log x

theorem tangent_line_at_A (a : ℝ) (h : a = 2) :
  ∀ (x : ℝ), (x + (f x a) - 2 = 0) :=
by
  sorry

theorem extreme_values (a : ℝ) :
  (if a ≤ 0 then ∀ x : ℝ, f x a is increasing on (0, +∞) and has no extreme values
   else ∃ x : ℝ, (differentiable := True) and (f x a is local minimum ∧ f x a = a - a * Real.log a)) :=
by
  sorry

end tangent_line_at_A_extreme_values_l23_23706


namespace total_canoes_by_end_of_april_l23_23972

def N_F : ℕ := 4
def N_M : ℕ := 3 * N_F
def N_A : ℕ := 3 * N_M
def total_canoes : ℕ := N_F + N_M + N_A

theorem total_canoes_by_end_of_april : total_canoes = 52 := by
  sorry

end total_canoes_by_end_of_april_l23_23972


namespace socks_diff_color_than_shorts_prob_l23_23064

theorem socks_diff_color_than_shorts_prob :
  let socks_colors := {red, blue, white}
  let shorts_colors := {red, blue, green, white}
  let total_configurations := 3 * 4
  let matching_configurations := 3
  let non_matching_configurations := total_configurations - matching_configurations
  let probability_non_matching := non_matching_configurations / total_configurations
  probability_non_matching = 3 / 4 :=
by
  sorry

end socks_diff_color_than_shorts_prob_l23_23064


namespace smallest_m_for_fibonacci_representations_l23_23044

-- Definition of Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fibonacci (n+1) + fibonacci n

-- Main problem statement
theorem smallest_m_for_fibonacci_representations :
  ∃ (m : ℕ), (∀ (x : ℕ), x ∈ finset.range 2018 → ∃ (s : finset ℕ), ∑ i in s, fibonacci i = fibonacci x) ∧ m = 1009 := 
sorry

end smallest_m_for_fibonacci_representations_l23_23044


namespace cameron_list_count_l23_23997

theorem cameron_list_count :
  let numbers := {n : ℕ | 30 ≤ n ∧ n ≤ 900}
  in set.card numbers = 871 :=
sorry -- proof is omitted

end cameron_list_count_l23_23997


namespace quadratic_coeff_nonzero_l23_23379

theorem quadratic_coeff_nonzero {m x : ℝ} (h : m * x^2 + 3 * x - 4 = 3 * x^2) : m ≠ 3 :=
by
  have h_eqn : (m - 3) * x^2 + 3 * x - 4 = 0 := by sorry
  have h0 : m - 3 ≠ 0 := by sorry
  exact h0

end quadratic_coeff_nonzero_l23_23379


namespace range_of_a_l23_23358

noncomputable def set_A (a : ℝ) : Set ℝ := {x | x < a}
noncomputable def set_B : Set ℝ := {x | 1 < x ∧ x < 2}
noncomputable def complement_B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2 }

theorem range_of_a (a : ℝ) : (set_A a ∪ complement_B) = Set.univ ↔ 2 ≤ a := 
by 
  sorry

end range_of_a_l23_23358


namespace ratio_triangle_areas_l23_23065

theorem ratio_triangle_areas (x : ℝ) (xpos : 0 < x) (xless : x < 1)
  (h1 : x^2 + x - 1 = 0) :
  let AXY_area := (1 - x) / 2
  let XCY_area := x^2 / 2
  in (AXY_area / XCY_area = Real.sqrt 5) :=
sorry

end ratio_triangle_areas_l23_23065


namespace solve_system_of_equations_l23_23840

theorem solve_system_of_equations :
  ∃ x y : ℝ, 4 * x - 6 * y = -3 ∧ 9 * x + 3 * y = 6.3 ∧ x = 0.436 ∧ y = 0.792 :=
by
  sorry

end solve_system_of_equations_l23_23840


namespace perfect_square_iff_odd_divisors_l23_23388

theorem perfect_square_iff_odd_divisors (N : ℕ) (p : ℕ → Prop) (α β γ λ : ℕ) :
  (∃ (p1 p2 p3 ... pk : ℕ), (N = p1^α * p2^β * p3^γ * ... * pk^λ) ∧
  (∀ x, x ∈ {p1, p2, p3, ..., pk} → prime x)) →
  ((α + 1) * (β + 1) * (γ + 1) * ... * (λ + 1) % 2 = 1 ↔ (∃ b, N = b^2)) :=
sorry

end perfect_square_iff_odd_divisors_l23_23388


namespace sparse_set_exists_P_eq_P_bad_angel_NP_ne_P_bad_angel_sparse_set_angel_language_l23_23020

-- Problem 1: 
theorem sparse_set_exists
  (k : ℕ)
  (S : ℕ → list (list ℕ))
  (h_sparse : ∀ i, ∃ p : ℕ → ℕ, ∀ n, (S i).countp (λ x, x.length = n) ≤ p n):
  ∃ T M, sparse T ∧ polytime (λ x i, M x i T) := sorry

-- Problem 2:
theorem P_eq_P_bad_angel :
  \exists P:Type, (P = P_bad_angel):
  sorry
theorem NP_ne_P_bad_angel :
  \exists NP:Type, NP ≠P_bad_angel:
  sorry

-- Problem 3:
theorem sparse_set_angel_language
  (L : ℕ → list ℕ)
  (h_L_angel : ∀ x n, (x ∈ L) ↔ (∃ p : ℕ → ℕ, ∃ α : ℕ → list ℕ, polytime α ∧ polytime (λ (x : list ℕ) (α, n : ℕ), (x, α n) ) ):
    sparse S_L ∧ ∃ M, polytime (λ x n, M x n S_L) := sorry

end sparse_set_exists_P_eq_P_bad_angel_NP_ne_P_bad_angel_sparse_set_angel_language_l23_23020


namespace find_range_of_m_l23_23454

open Real

-- Definition for proposition p (the discriminant condition)
def real_roots (m : ℝ) : Prop := (3 * 3) - 4 * m ≥ 0

-- Definition for proposition q (ellipse with foci on x-axis conditions)
def is_ellipse (m : ℝ) : Prop := 
  9 - m > 0 ∧ 
  m - 2 > 0 ∧ 
  9 - m > m - 2

-- Lean statement for the mathematically equivalent proof problem
theorem find_range_of_m (m : ℝ) : (real_roots m ∧ is_ellipse m) → (2 < m ∧ m ≤ 9 / 4) := 
by
  sorry

end find_range_of_m_l23_23454


namespace find_k_value_l23_23263

theorem find_k_value
  (a : ℂ) (b : ℂ) (k : ℕ) : 
  a = 5 → b = 14 →
  (∃ z₁ z₂ z₃ : ℂ, 
    (z₁ = a + b * complex.i) ∧ (z₂ = a) ∧ (z₃ = a - b * complex.i) ∧
    (z₁ + z₂ + z₃ = 15) ∧
    (z₁ * z₂ * z₃ = 1105) ∧
    (z₁ * z₂ + z₂ * z₃ + z₃ * z₁ = k)) →
  k = 271 :=
by sorry

end find_k_value_l23_23263


namespace possible_values_of_deriv_l23_23810

noncomputable def differentiable_function (f : ℝ → ℝ) [∀ x ∈ set.Ioo 0 1, differentiable_at ℝ f x] 
  (h_deriv_cont : continuous_on (deriv f) (set.Ioo 0 1)) : Prop :=
∀ n : ℕ, ∀ a : ℕ, a < 2^n ∧ odd a → ∃ b : ℕ, b < 2^n ∧ odd b ∧ f (a / 2^n) = b / 2^n

theorem possible_values_of_deriv (f : ℝ → ℝ) 
  (h_diff_cont : differentiable_function f) :
  deriv f (1 / 2) = 1 ∨ deriv f (1 / 2) = -1 :=
sorry

end possible_values_of_deriv_l23_23810


namespace nat_perfect_square_l23_23038

theorem nat_perfect_square (a b : ℕ) (h : ∃ k : ℕ, a^2 + b^2 + a = k * a * b) : ∃ m : ℕ, a = m * m := by
  sorry

end nat_perfect_square_l23_23038


namespace factorization_correct_l23_23918

theorem factorization_correct (x y : ℝ) : 
  x * (x - y) - y * (x - y) = (x - y) ^ 2 :=
by 
  sorry

end factorization_correct_l23_23918


namespace minimum_time_for_tomato_egg_soup_l23_23485

noncomputable def cracking_egg_time : ℕ := 1
noncomputable def washing_chopping_tomatoes_time : ℕ := 2
noncomputable def boiling_tomatoes_time : ℕ := 3
noncomputable def adding_eggs_heating_time : ℕ := 1
noncomputable def stirring_egg_time : ℕ := 1

theorem minimum_time_for_tomato_egg_soup :
  washing_chopping_tomatoes_time + boiling_tomatoes_time + adding_eggs_heating_time = 6 :=
by
  -- proof to be filled
  sorry

end minimum_time_for_tomato_egg_soup_l23_23485


namespace units_digit_factorial_sum_l23_23270

theorem units_digit_factorial_sum : 
  (∑ n in (Finset.range 2024), (nat.factorial n) % 10) % 10 = 3 := 
by 
  sorry

end units_digit_factorial_sum_l23_23270


namespace arithmetic_geometric_value_l23_23677

-- Definitions and annotations
variables {a1 a2 b1 b2 : ℝ}
variable {d : ℝ} -- common difference for the arithmetic sequence
variable {q : ℝ} -- common ratio for the geometric sequence

-- Assuming input values for the initial elements of the sequences
axiom h1 : -9 = -9
axiom h2 : -9 + 3 * d = -1
axiom h3 : b1 = -9 * q
axiom h4 : b2 = -9 * q^2

-- The desired equality to prove
theorem arithmetic_geometric_value :
  b2 * (a2 - a1) = -8 :=
sorry

end arithmetic_geometric_value_l23_23677


namespace average_unit_price_l23_23600

theorem average_unit_price (unit_price_A unit_price_B unit_price_C unit_price_D : ℝ)
  (quantity_A quantity_B quantity_C quantity_D total_pens : ℕ) :
  unit_price_A = 5 → 
  unit_price_B = 3 →
  unit_price_C = 2 →
  unit_price_D = 1 →
  quantity_A = 5 →
  quantity_B = 8 →
  quantity_C = 27 →
  quantity_D = 10 →
  total_pens = 50 →
  (quantity_A + quantity_B + quantity_C + quantity_D = total_pens) →
  (unit_price_A * quantity_A + unit_price_B * quantity_B + unit_price_C * quantity_C + unit_price_D * quantity_D) / total_pens = 2.26 :=
begin
  intros hA hB hC hD qA qB qC qD tP sum_pens,
  sorry
end

end average_unit_price_l23_23600


namespace constant_term_in_binomial_expansion_l23_23852

theorem constant_term_in_binomial_expansion :
  let binomial_expansion := (x - (1 / (2 * x^3)))^8
  ∃ t, is_constant_term binomial_expansion t ∧ t = 7 :=
by
  sorry

end constant_term_in_binomial_expansion_l23_23852


namespace servings_in_box_l23_23950

theorem servings_in_box :
  (total_cups : ℕ) (cups_per_serving : ℕ) (h1 : total_cups = 18) (h2 : cups_per_serving = 2) :
  total_cups / cups_per_serving = 9 :=
by
  sorry

end servings_in_box_l23_23950


namespace quadratic_root_shift_l23_23801

theorem quadratic_root_shift (r s : ℝ)
    (hr : 2 * r^2 - 8 * r + 6 = 0)
    (hs : 2 * s^2 - 8 * s + 6 = 0)
    (h_sum_roots : r + s = 4)
    (h_prod_roots : r * s = 3)
    (b : ℝ) (c : ℝ)
    (h_b : b = - (r - 3 + s - 3))
    (h_c : c = (r - 3) * (s - 3)) : c = 0 :=
  by sorry

end quadratic_root_shift_l23_23801


namespace largest_n_for_sin_cos_inequality_l23_23259

theorem largest_n_for_sin_cos_inequality :
  ∀ (x : ℝ), sin x ^ 4 + cos x ^ 4 ≥ 1 / 2 :=
by
  -- The proof follows but is omitted here
  sorry

end largest_n_for_sin_cos_inequality_l23_23259


namespace proposition_D_l23_23542

/-- Lean statement for proving the correct proposition D -/
theorem proposition_D {a b : ℝ} (h : |a| < b) : a^2 < b^2 :=
sorry

end proposition_D_l23_23542


namespace mike_trip_representation_l23_23435

-- Definitions based on conditions
def drives_slowly_through_city_traffic : Prop := true
def stops_for_gas (minutes : ℕ) : Prop := minutes = 15
def continues_slowly_until_highway : Prop := true
def drives_rapidly_on_highway : Prop := true
def stops_for_shopping (hours : ℕ) : Prop := hours = 1.5
def encounters_traffic_on_return : Prop := true
def drives_slowly_back_through_city_traffic : Prop := true

-- Defining the overall conditions
def mike_trip_conditions : Prop :=
  drives_slowly_through_city_traffic ∧
  stops_for_gas 15 ∧
  continues_slowly_until_highway ∧
  drives_rapidly_on_highway ∧
  stops_for_shopping 1.5 ∧
  encounters_traffic_on_return ∧
  drives_slowly_back_through_city_traffic

-- The theorem statement
theorem mike_trip_representation : mike_trip_conditions → (Graph = "A") := by
  sorry

end mike_trip_representation_l23_23435


namespace option_A_not_correct_option_B_correct_option_C_correct_option_D_correct_l23_23155

theorem option_A_not_correct 
  (x : ℝ) : ¬ (∀ y, y = (x^2 + 1)/x → y ≥ 2) := 
sorry

theorem option_B_correct 
  (x y : ℝ) (h : x > 1) (hy : y = 2x + (4 / (x - 1)) - 1) : 
  y ≥ 4 * Real.sqrt 2 + 1 :=
sorry

theorem option_C_correct 
  {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 3 * x * y) : 
  2 * x + y ≥ 3 := 
sorry

theorem option_D_correct 
  {x y : ℝ} (h : 9 * x^2 + y^2 + x * y = 1) : 
  3 * x + y ≤ (2 * Real.sqrt 21) / 7 := 
sorry

end option_A_not_correct_option_B_correct_option_C_correct_option_D_correct_l23_23155


namespace selection_count_l23_23773

-- Definition of the problem parameters and constraints
def valid_selection (s : Set ℕ) : Prop :=
  s.card = 5 ∧ ∀ x ∈ s, ∀ y ∈ s, x ≠ y → abs (x - y) ≠ 1

-- Set of numbers from 1 to 18
def numbers := {x : ℕ | 1 ≤ x ∧ x ≤ 18}

-- Main theorem statement
theorem selection_count : (Finset.filter valid_selection (Finset.powerset (Finset.range' 1 18))).card = 2002 :=
sorry

end selection_count_l23_23773


namespace dessert_menu_count_is_192_l23_23578

-- Defining the set of desserts
inductive Dessert
| cake | pie | ice_cream

-- Function to count valid dessert menus (not repeating on consecutive days) with cake on Friday
def countDessertMenus : Nat :=
  -- Let's denote Sunday as day 1 and Saturday as day 7
  let sunday_choices := 3
  let weekday_choices := 2 -- for Monday to Thursday (no repeats consecutive)
  let weekend_choices := 2 -- for Saturday and Sunday after
  sunday_choices * weekday_choices^4 * 1 * weekend_choices^2

-- Theorem stating the number of valid dessert menus for the week
theorem dessert_menu_count_is_192 : countDessertMenus = 192 :=
  by
    -- Actual proof is omitted
    sorry

end dessert_menu_count_is_192_l23_23578


namespace f_f_x_eq_8_has_5_solutions_l23_23724

def f (x : ℝ) : ℝ :=
if x ≥ -2 then x^2 - 1 else x + 4

def number_of_solutions : ℕ :=
5

theorem f_f_x_eq_8_has_5_solutions :
  {x : ℝ | f (f x) = 8}.to_finset.card = number_of_solutions :=
by
  sorry

end f_f_x_eq_8_has_5_solutions_l23_23724


namespace area_of_A_l23_23231

open Complex

def is_in_region (z : ℂ) : Prop :=
  let x := re z
  let y := im z
  0 ≤ x ∧ x ≤ 50 ∧
  0 ≤ y ∧ y ≤ 50 ∧
  (x - 25)^2 + y^2 ≥ 625 ∧
  x^2 + (y - 25)^2 ≥ 625

noncomputable def area_of_region : ℝ :=
  2500 - 312.5 * Real.pi

theorem area_of_A : ∃ A, (∀ z : ℂ, is_in_region z ↔ z ∈ A) ∧
  (measure_theory.measure_space.measure (set.univ : set ℂ) A = area_of_region) :=
sorry

end area_of_A_l23_23231


namespace tangent_line_l23_23851

-- Let O, O1, O2, C, C1, C2, M, N, A, B, E, F be as stated in the problem
variables {O O1 O2 : Type} 
variables {C C1 C2 M N A B E F : Type}
variables [MetricSpace O] [MetricSpace O1] [MetricSpace O2] [MetricSpace C] [MetricSpace C1] [MetricSpace C2]

-- Conditions of the problem
variable (r r1 r2 : ℝ)  -- Radii of C, C1 and C2
variable (h_tangent1 : MetricSpace.tangent_of C C1 M)
variable (h_tangent2 : MetricSpace.tangent_of C C2 N)
variable (h_c1_passing_o2 : MetricSpace.passing_through C1 O2)
variable (h_ab_common_chord : MetricSpace.common_chord C1 C2 A B)
variable (h_ma_meeting_e : MetricSpace.meeting_at C1 M A E)
variable (h_mb_meeting_f : MetricSpace.meeting_at C1 M B F)

-- The theorem we need to prove
theorem tangent_line (h_conditions : h_tangent1 ∧ h_tangent2 ∧ h_c1_passing_o2 ∧ h_ab_common_chord ∧ h_ma_meeting_e ∧ h_mb_meeting_f) : 
  MetricSpace.tangent_of_line EF C2 :=
begin
  sorry,  -- Proof goes here
end

end tangent_line_l23_23851


namespace distinct_polynomials_in_X_l23_23921

noncomputable def X : set (polynomial ℝ) :=
{p | p = polynomial.X ∨
    (∃ q, q ∈ X ∧ p = polynomial.X * q) ∨
    (∃ q, q ∈ X ∧ p = polynomial.X + (1 - polynomial.X) * q)}

theorem distinct_polynomials_in_X (r s : polynomial ℝ) (hr : r ∈ X) (hs : s ∈ X) (hdistinct : r ≠ s) :
  ∀ x, 0 < x ∧ x < 1 → r.eval x ≠ s.eval x :=
by
  intros x hx
  -- Detailed proof steps should be filled here.
  sorry

end distinct_polynomials_in_X_l23_23921


namespace KH_perpendicular_CD_l23_23385

-- Definitions for the conditions
variables {A B C D M N K H : Type*}
variables [convex_quadrilateral A B C D]
variables (bad_eq_bcd : angle BAD = angle BCD)
variables (M_on_AB : on_segment AB M) (N_on_BC : on_segment BC N)
variables (AD_parallel_MN : parallel AD MN) (MN_eq_2AD : MN = 2 * AD)
variables (H_orthocenter_ABC : orthocenter H A B C)
variables (K_midpoint_MN : midpoint_n K M N)

-- The final statement
theorem KH_perpendicular_CD :
  perpendicular KH CD :=
sorry

end KH_perpendicular_CD_l23_23385


namespace number_of_dogs_not_liking_either_l23_23770

variable (total_dogs dogs_like_chicken dogs_like_beef dogs_like_both : ℕ)

def dogs_not_liking_either : ℕ := total_dogs - (dogs_like_chicken + dogs_like_beef - dogs_like_both)

theorem number_of_dogs_not_liking_either 
  (h1 : total_dogs = 75)
  (h2 : dogs_like_chicken = 13)
  (h3 : dogs_like_beef = 55)
  (h4 : dogs_like_both = 8) : dogs_not_liking_either total_dogs dogs_like_chicken dogs_like_beef dogs_like_both = 15 :=
by
  simp [dogs_not_liking_either, h1, h2, h3, h4]
  sorry

end number_of_dogs_not_liking_either_l23_23770


namespace recurrence_solution_equiv_l23_23712

noncomputable def recurrence_relation (a : ℕ → ℝ) : Prop :=
  a 0 = real.sqrt 2 / 2 ∧
  ∀ n, a (n + 1) = real.sqrt ((a n + 2 - real.sqrt (2 - a n)) / 2)

noncomputable def solution (n : ℕ) : ℝ :=
  real.sqrt 2 * real.cos (real.pi / 4 + real.pi / (12 * 2^n))

theorem recurrence_solution_equiv (a : ℕ → ℝ) :
  recurrence_relation a → ∀ n, a n = solution n :=
sorry

end recurrence_solution_equiv_l23_23712


namespace max_value_ln_x_plus_x_l23_23662

theorem max_value_ln_x_plus_x (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ Real.exp 1) : 
  ∃ y, y = Real.log x + x ∧ y ≤ Real.log (Real.exp 1) + Real.exp 1 :=
sorry

end max_value_ln_x_plus_x_l23_23662


namespace probability_heads_on_11th_toss_l23_23547

variable (Xiaofang : Type) [ProbabilitySpace Xiaofang]

def fair_coin (P : ProbabilitySpace Xiaofang) := ∀ outcome : Event P, 
  (outcome = heads ∨ outcome = tails) →  
  measure (eq heads) = 1/2

def independent_tosses (P : ProbabilitySpace Xiaofang) := ∀ n : ℕ, 
  ∀ outcomes : Fin n → Event P, pairwise Independent outcomes

theorem probability_heads_on_11th_toss
  {P : ProbabilitySpace Xiaofang}
  (fair : fair_coin P)
  (indep : independent_tosses P) :
  true :=
sorry

end probability_heads_on_11th_toss_l23_23547


namespace gcd_of_sum_of_cubes_and_increment_l23_23646

theorem gcd_of_sum_of_cubes_and_increment {n : ℕ} (h : n > 3) : Nat.gcd (n^3 + 27) (n + 4) = 1 :=
by sorry

end gcd_of_sum_of_cubes_and_increment_l23_23646


namespace sqrt_meaningful_range_l23_23729

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 2) : x ≥ 2 :=
sorry

end sqrt_meaningful_range_l23_23729


namespace exists_infinite_set_l23_23167

-- Define the type of points in the plane as pairs of real numbers.
structure Point :=
(x : ℝ)
(y : ℝ)

-- Definition of the distance between two points.
def distance (A B : Point) : ℝ := real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2)

-- Definition to assert that no three points are collinear.
def not_collinear (A B C : Point) : Prop :=
¬(A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y) = 0)

-- Definition to assert that the distance between two points is rational.
def distance_rational (A B : Point) : Prop :=
∃ (q : ℚ), abs (distance A B - q) = 0

-- Definition to assert a set of points meets the problem's conditions.
def infinite_set_satisfies (S : set Point) : Prop :=
  set.infinite S ∧
  (∀ A B C ∈ S, A ≠ B ∧ B ≠ C ∧ C ≠ A → not_collinear A B C) ∧
  (∀ A B ∈ S, A ≠ B → distance_rational A B)

-- The main statement establishing the existence of such a set.
theorem exists_infinite_set : ∃ S : set Point, infinite_set_satisfies S :=
by
  sorry

end exists_infinite_set_l23_23167


namespace quadratic_coefficients_l23_23083

theorem quadratic_coefficients :
  ∀ (x : ℝ), x^2 - x + 3 = 0 → (1, -1, 3) :=
by
  intro x
  intro h
  have quadratic_coeff : x^2 - x + 3 = 1 * x^2 + (-1) * x + 3 := by simp
  exact (1, -1, 3) 
  sorry

end quadratic_coefficients_l23_23083


namespace tokens_per_pitch_l23_23432

theorem tokens_per_pitch 
  (tokens_macy : ℕ) (tokens_piper : ℕ)
  (hits_macy : ℕ) (hits_piper : ℕ)
  (misses_total : ℕ) (p : ℕ)
  (h1 : tokens_macy = 11)
  (h2 : tokens_piper = 17)
  (h3 : hits_macy = 50)
  (h4 : hits_piper = 55)
  (h5 : misses_total = 315)
  (h6 : 28 * p = hits_macy + hits_piper + misses_total) :
  p = 15 := 
by 
  sorry

end tokens_per_pitch_l23_23432


namespace probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l23_23761

noncomputable def probability_k_gnomes_fall (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p * (1 - p) ^ (n - k)

noncomputable def expected_number_of_fallen_gnomes (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p)

theorem probability_k_gnomes_fall_correct (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  probability_k_gnomes_fall n k p h = p * (1 - p) ^ (n - k) :=
by sorry

theorem expected_number_of_fallen_gnomes_correct (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  expected_number_of_fallen_gnomes n p h = n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p) :=
by sorry

end probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l23_23761


namespace limit_evaluation_l23_23681

variable {α : Type*} [NormedField α] [NormedSpace ℝ α] {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem limit_evaluation (f : ℝ → ℝ) (x₀ a : ℝ) 
  (h : deriv f x₀ = a) : 
  tendsto (λ Δx : ℝ, (f (x₀ + Δx) - f (x₀ - 3 * Δx)) / (2 * Δx)) (𝓝 0) (𝓝 (2 * a)) :=
sorry

end limit_evaluation_l23_23681


namespace probability_first_draw_second_given_second_draw_first_l23_23176

open ProbabilityTheory

-- Definitions of events
def first_draw_second (ω : SampleSpace) : Prop := -- Definition of first draw being second-class item
def second_draw_first (ω : SampleSpace) : Prop := -- Definition of second draw being first-class item

-- Sample space setup based on the problem conditions
def sample_space := {ω : SampleSpace | -- Conditions for sample space }

-- Main theorem statement
theorem probability_first_draw_second_given_second_draw_first :
  (probability (Set.Inter (Set {ω | second_draw_first ω}) (Set {ω | first_draw_second ω})))
  / (probability (Set {ω | second_draw_first ω})) = 1 / 2 := sorry

end probability_first_draw_second_given_second_draw_first_l23_23176


namespace smallest_angle_solution_l23_23642

noncomputable def find_smallest_angle : ℝ :=
  classical.some (Exists.some (Icc 0 360) (λ x, sin (3 * x) * sin (4 * x) = cos (3 * x) * cos (4 * x)))

theorem smallest_angle_solution : find_smallest_angle = 90 / 7 := sorry

end smallest_angle_solution_l23_23642


namespace train_passes_platform_in_200_seconds_l23_23923

-- Define the length of the train
def length_of_train : ℝ := 1200

-- Define the time to cross the tree
def time_to_cross_tree : ℝ := 120

-- Define the length of the platform
def length_of_platform : ℝ := 800

-- Define the speed of the train based on distance / time
def speed_of_train : ℝ := length_of_train / time_to_cross_tree

-- Define the total distance needed to pass the platform
def total_distance_to_pass_platform : ℝ := length_of_train + length_of_platform

-- Define the time it will take to pass the platform based on total distance / speed
def time_to_pass_platform : ℝ := total_distance_to_pass_platform / speed_of_train

-- Prove the statement
theorem train_passes_platform_in_200_seconds : time_to_pass_platform = 200 := by
  -- Calculation steps would go here
  sorry

end train_passes_platform_in_200_seconds_l23_23923


namespace shorter_side_length_l23_23961

theorem shorter_side_length (a b : ℕ) (h1 : 2*a + 2*b = 48) (h2 : a * b = 140) : b = 10 ∨ a = 10 :=
by 
suffices : a + b = 24 
by 
  have : (a - 14) * (a - 10) = 0 := by
    -- Here one would factor the quadratic equation derived from the equations
    sorry
  cases this
  case or.inl
    have : a = 14
    have : b = 10
  case or.inr
    have : a = 10
    have : b = 14
  
sorry

end shorter_side_length_l23_23961


namespace parallel_segments_slope_l23_23858

theorem parallel_segments_slope (k : ℝ) :
  let A := (-3 : ℝ, 2 : ℝ)
  let B := (1 : ℝ, 8 : ℝ)
  let X := (3 : ℝ, -6 : ℝ)
  let Y := (11 : ℝ, k)
  let slope := λ p1 p2 : ℝ × ℝ, (p2.2 - p1.2) / (p2.1 - p1.1)
  slope A B = slope X Y → k = 6 := 
begin
  sorry
end

end parallel_segments_slope_l23_23858


namespace systematic_sampling_selected_students_l23_23460

def is_systematic_sampling {α : Type} (population : List α) (step : ℕ) : List α → Prop
| []        := true
| (x :: xs) := xs = List.drop step population ∧ is_systematic_sampling population step xs

theorem systematic_sampling_selected_students :
  ∃ selected_students : List ℕ,
    selected_students = [5 + 10 * i | i in Finset.range 5] ∧
    is_systematic_sampling (List.range 50) 10 selected_students :=
by
  let selected_students := [5, 15, 25, 35, 45]
  use selected_students
  split
  {
    -- Prove that selected_students equals to the desired list
    rw [List.range 5]
    rfl
  }
  {
    -- Prove that selected_students meets systematic sampling condition
    sorry
  }


end systematic_sampling_selected_students_l23_23460


namespace box_base_length_max_l23_23103

noncomputable def V (x : ℝ) := x^2 * ((60 - x) / 2)

theorem box_base_length_max 
  (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 60)
  (h3 : ∀ y : ℝ, 0 < y ∧ y < 60 → V x ≥ V y)
  : x = 40 :=
sorry

end box_base_length_max_l23_23103


namespace max_short_sighted_rooks_l23_23568

-- Define short-sighted rook properties and the maximal placement problem
def isShortSightedRook (board_size steps row col : ℕ) : Prop :=
  row <= board_size ∧ col <= board_size ∧ steps = 60

def non_attacking_rooks (board_size max_rooks : ℕ) (positions : List (ℕ × ℕ)) : Prop :=
  ∀ p1 p2 ∈ positions, 
    p1 ≠ p2 → 
    abs (p1.1 - p2.1) > 60 ∧ abs (p1.2 - p2.2) > 60

-- State the theorem for the maximum number of non-attacking short-sighted rooks
theorem max_short_sighted_rooks (board_size : ℕ) (steps : ℕ) : 
  board_size = 100 → 
  steps = 60 → 
  ∃ (positions : List (ℕ × ℕ)), 
    isShortSightedRook board_size steps ∧ 
    non_attacking_rooks board_size 178 positions := sorry

end max_short_sighted_rooks_l23_23568


namespace units_digit_factorial_sum_l23_23271

theorem units_digit_factorial_sum : 
  (∑ n in (Finset.range 2024), (nat.factorial n) % 10) % 10 = 3 := 
by 
  sorry

end units_digit_factorial_sum_l23_23271


namespace soccer_league_teams_l23_23125

theorem soccer_league_teams (n : ℕ) (h : n * (n - 1) / 2 = 105) : n = 15 :=
by
  -- Proof will go here
  sorry

end soccer_league_teams_l23_23125


namespace option_b_option_c_option_d_l23_23148

theorem option_b (x : ℝ) (h : x > 1) : (∀ y, y = 2*x + 4 / (x - 1) - 1 → y ≥ 4*Real.sqrt 2 + 1) :=
by
  sorry

theorem option_c (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3 * x * y) : 2*x + y ≥ 3 :=
by
  sorry

theorem option_d (x y : ℝ) (h : 9*x^2 + y^2 + x*y = 1) : 3*x + y ≤ 2*Real.sqrt 21 / 7 :=
by
  sorry

end option_b_option_c_option_d_l23_23148


namespace cubic_polynomial_solution_l23_23278

theorem cubic_polynomial_solution (x : ℝ) :
  x^3 + 6*x^2 + 11*x + 6 = 12 ↔ x = -1 ∨ x = -2 ∨ x = -3 := by
  sorry

end cubic_polynomial_solution_l23_23278


namespace percentage_decrease_in_area_l23_23743

noncomputable def original_radius (r : ℝ) : ℝ := r
noncomputable def new_radius (r : ℝ) : ℝ := 0.5 * r
noncomputable def original_area (r : ℝ) : ℝ := Real.pi * r ^ 2
noncomputable def new_area (r : ℝ) : ℝ := Real.pi * (0.5 * r) ^ 2

theorem percentage_decrease_in_area (r : ℝ) (hr : 0 ≤ r) :
  ((original_area r - new_area r) / original_area r) * 100 = 75 :=
by
  sorry

end percentage_decrease_in_area_l23_23743


namespace angle_bisector_H_on_median_BK_l23_23322

noncomputable def isosceles_triangle (A B C : Type) (h : triangle A B C) := (h.AB = h.AC)

noncomputable def select_point_M (A B C M : Type) (h : triangle A B C) := (h.AC = h.CM)

noncomputable def select_point_N (B C M N : Type) (hBN : segment B C) := (hBN.BN = hBN.MN)

theorem angle_bisector_H_on_median_BK
  (A B C M N H : Type)
  (hABC : triangle A B C)
  (hAB : hABC.AB = hABC.AC)
  (hM : segment A B)
  (hCM : hABC.CM = hABC.AC)
  (hN : segment B C)
  (hBN : hN.BN = hN.MN)
  (hNH : angle_bisector N H) :
  lies_on_median H B K :=
sorry

end angle_bisector_H_on_median_BK_l23_23322


namespace evaluate_square_l23_23632

variable {x : ℝ}

theorem evaluate_square (x : ℝ) : 
  (8 - real.sqrt (x^2 + 64))^2 = x^2 + 128 - 16 * real.sqrt (x^2 + 64) :=
sorry

end evaluate_square_l23_23632


namespace average_speed_approx_15_l23_23747

noncomputable def distance_meters : ℝ := 5000
noncomputable def time_minutes : ℝ := 19
noncomputable def time_seconds : ℝ := 6

noncomputable def distance_kilometers : ℝ := distance_meters / 1000
noncomputable def time_hours : ℝ := (time_minutes + time_seconds / 60) / 60

noncomputable def average_speed : ℝ := distance_kilometers / time_hours

theorem average_speed_approx_15 : average_speed ≈ 15 :=
by
  sorry

end average_speed_approx_15_l23_23747


namespace intersection_on_circle_l23_23456

def parabola1 (X : ℝ) : ℝ := X^2 + X - 41
def parabola2 (Y : ℝ) : ℝ := Y^2 + Y - 40

theorem intersection_on_circle (X Y : ℝ) :
  parabola1 X = Y ∧ parabola2 Y = X → X^2 + Y^2 = 81 :=
by {
  sorry
}

end intersection_on_circle_l23_23456


namespace exists_infinite_set_l23_23168

-- Define the type of points in the plane as pairs of real numbers.
structure Point :=
(x : ℝ)
(y : ℝ)

-- Definition of the distance between two points.
def distance (A B : Point) : ℝ := real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2)

-- Definition to assert that no three points are collinear.
def not_collinear (A B C : Point) : Prop :=
¬(A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y) = 0)

-- Definition to assert that the distance between two points is rational.
def distance_rational (A B : Point) : Prop :=
∃ (q : ℚ), abs (distance A B - q) = 0

-- Definition to assert a set of points meets the problem's conditions.
def infinite_set_satisfies (S : set Point) : Prop :=
  set.infinite S ∧
  (∀ A B C ∈ S, A ≠ B ∧ B ≠ C ∧ C ≠ A → not_collinear A B C) ∧
  (∀ A B ∈ S, A ≠ B → distance_rational A B)

-- The main statement establishing the existence of such a set.
theorem exists_infinite_set : ∃ S : set Point, infinite_set_satisfies S :=
by
  sorry

end exists_infinite_set_l23_23168


namespace magnitude_of_difference_l23_23657

def vector_a := (1 : ℝ, 2 : ℝ)
def vector_b (x : ℝ) := (x, 6)
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem magnitude_of_difference : 
  (∃ x : ℝ, parallel vector_a (vector_b x)) → 
  ∃ x : ℝ, ∥(vector_a.1 - (vector_b x).1, vector_a.2 - (vector_b x).2)∥ = 2 * Real.sqrt 5 :=
by
  sorry

end magnitude_of_difference_l23_23657


namespace min_value_of_ratio_l23_23715

noncomputable def minimum_ratio (m : ℝ) (hm : m > 0) : ℝ :=
  |(2^m - 2^(8 / (2*m + 1))) / (2^(-m) - 2^(-8 / (2*m + 1)))|

theorem min_value_of_ratio : ∃ m, m > 0 ∧ minimum_ratio m m = 8*Real.sqrt 2 :=
by {
  sorry
}

end min_value_of_ratio_l23_23715


namespace range_of_a_l23_23017

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≥ 0 then a * Real.sin x + 2 else x^2 + 2 * a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ∃ y ≥ 1, y = f a x) ↔ (a ∈ Set.Iio (1/2) ∪ Set.Icc 1 2) :=
by
  sorry -- Proof is omitted as per instruction.

end range_of_a_l23_23017


namespace max_points_of_intersection_l23_23533

theorem max_points_of_intersection (circles : ℕ) (line : ℕ) (h_circles : circles = 3) (h_line : line = 1) : 
  ∃ points_of_intersection, points_of_intersection = 12 :=
by
  -- Proof here (omitted)
  sorry

end max_points_of_intersection_l23_23533


namespace units_digit_of_factorial_sum_l23_23275

theorem units_digit_of_factorial_sum :
  (1! + 2! + 3! + 4! + (↓∑ k in Icc 5 2023, k!)) % 10 = 3 := by
  sorry

end units_digit_of_factorial_sum_l23_23275


namespace range_of_m_intersecting_ellipse_longest_chord_line_equation_l23_23340

-- Definitions for conditions
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1
def line (x y : ℝ) (m : ℝ) : Prop := y = x + m

-- Main theorem part 1
theorem range_of_m_intersecting_ellipse (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ (-real.sqrt 5 / 2 ≤ m ∧ m ≤ real.sqrt 5 / 2) :=
sorry

-- Main theorem part 2
theorem longest_chord_line_equation :
  (∃ m : ℝ, (∀ x y : ℝ, ellipse x y → line x y m) ∧ 
   ∀ m1 m2 : ℝ, m1 = 0 → ‖(line m1) - (line m2)‖ ≤ ‖(line m1) - (line 0)‖) ↔
  (line y x 0) :=
sorry

end range_of_m_intersecting_ellipse_longest_chord_line_equation_l23_23340


namespace find_a6_l23_23313

-- Define an arithmetic progression.
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the necessary conditions given in the problem.
def conditions (a d : ℕ) : Prop :=
  (arithmetic_progression a d 1 + arithmetic_progression a d 2 + arithmetic_progression a d 3 = 168) ∧
  (arithmetic_progression a d 2 - arithmetic_progression a d 5 = 42)

-- State the theorem with the final value assertion.
theorem find_a6 (a d : ℕ) (h : conditions a (-14)) : 
  arithmetic_progression a (-14) 6 = 3 := 
sorry

end find_a6_l23_23313


namespace glass_price_l23_23842

theorem glass_price
  (num_dolls : ℕ) (num_clocks : ℕ) (num_glasses : ℕ)
  (price_per_doll : ℕ) (price_per_clock : ℕ) (total_spent : ℕ) (profit : ℕ)
  (num_dolls = 3) (num_clocks = 2) (num_glasses = 5)
  (price_per_doll = 5) (price_per_clock = 15) (total_spent = 40) (profit = 25) :
  ∃ price_per_glass : ℕ, price_per_glass = 4 :=
by
  sorry

end glass_price_l23_23842


namespace units_digit_of_factorial_sum_l23_23274

theorem units_digit_of_factorial_sum :
  (1! + 2! + 3! + 4! + (↓∑ k in Icc 5 2023, k!)) % 10 = 3 := by
  sorry

end units_digit_of_factorial_sum_l23_23274


namespace eval_at_d_eq_4_l23_23631

theorem eval_at_d_eq_4 : ((4: ℕ) ^ 4 - (4: ℕ) * ((4: ℕ) - 2) ^ 4) ^ 4 = 136048896 :=
by
  sorry

end eval_at_d_eq_4_l23_23631


namespace inequalities_hold_l23_23680

variable {a b : ℝ}

theorem inequalities_hold (h₀ : a > 0) (h₁ : b > 0) :
  (2 * a * b / (a + b) ≤ (a + b) / 2) ∧
  (sqrt (a * b) ≤ (a + b) / 2) ∧
  ((a + b) / 2 ≤ sqrt ((a^2 + b^2) / 2)) ∧
  (b^2 / a + a^2 / b ≥ a + b) :=
by
  sorry

end inequalities_hold_l23_23680


namespace area_of_triangle_medians_proof_l23_23865

noncomputable def area_of_triangle_medians (m1 m2 m3 : ℝ) : ℝ :=
  (4 / 3) * real.sqrt ((m1 + m2 + m3) / 2 * ((m1 + m2 + m3) / 2 - m1) * ((m1 + m2 + m3) / 2 - m2) * ((m1 + m2 + m3) / 2 - m3))

theorem area_of_triangle_medians_proof :
  area_of_triangle_medians 3 4 5 = 8 :=
by
  rw [area_of_triangle_medians, real.sqrt (6 * (6 - 3) * (6 - 4) * (6 - 5))]
  norm_num
  sorry

end area_of_triangle_medians_proof_l23_23865


namespace floor_powers_divisible_by_17_l23_23429

noncomputable def greatest_positive_root (p : Polynomial ℝ) : ℝ :=
  if h : ∃ x, p.eval x = 0 ∧ 0 < x then classical.some h else 0

theorem floor_powers_divisible_by_17 :
  let a := greatest_positive_root (Polynomial.Coeffs [1, -3, 0, 1]) in
  (∃ x ∈ (real.roots (Polynomial.Coeffs [1, -3, 0, 1])), x = a ∧ ∀ n ≥ 2, 
  (⌊a^1788⌋ % 17 = 0 ∧ ⌊a^1988⌋ % 17 = 0)) := sorry

end floor_powers_divisible_by_17_l23_23429


namespace base_conversion_l23_23598

-- Define the number in octal format
def octal_num : ℕ := 7 * 8^2 + 3 * 8^1 + 2 * 8^0

-- Define the number in decimal format
def decimal_num : ℕ := 474

-- Define the number in hexadecimal format
def hex_num : string := "1DA"

-- Prove the equivalence of base conversions
theorem base_conversion : 
  (octal_num = decimal_num) ∧ 
  (string_of_nat 16 decimal_num = hex_num) :=
by 
  sorry

end base_conversion_l23_23598


namespace total_cost_price_correct_l23_23583

-- Define the given selling prices
def SP_computer_table : ℝ := 3000
def SP_bookshelf : ℝ := 2400
def SP_dining_table_set : ℝ := 12000
def SP_sofa_set : ℝ := 18000

-- Define the cost price computation formula given the condition that SP = CP * 1.20
def CP (SP : ℝ) : ℝ := SP / 1.20

-- Using the formula to compute each cost price
def CP_computer_table : ℝ := CP SP_computer_table
def CP_bookshelf : ℝ := CP SP_bookshelf
def CP_dining_table_set : ℝ := CP SP_dining_table_set
def CP_sofa_set : ℝ := CP SP_sofa_set

-- Define the total cost price
def total_CP : ℝ := CP_computer_table + CP_bookshelf + CP_dining_table_set + CP_sofa_set

-- The proof problem statement:
theorem total_cost_price_correct : total_CP = 29500 := by
  sorry

end total_cost_price_correct_l23_23583


namespace units_digit_sum_l23_23009

theorem units_digit_sum (S : ℕ) (hS : S = 1! + 2! + 3! + 4! + (∑ k in finset.Icc 5 99, k!)) : 
  (S % 10 = 3) := 
by 
  sorry

end units_digit_sum_l23_23009


namespace pow_neg_one_diff_l23_23616

theorem pow_neg_one_diff (n : ℤ) (h1 : n = 2010) (h2 : n + 1 = 2011) :
  (-1)^2010 - (-1)^2011 = 2 := 
by
  sorry

end pow_neg_one_diff_l23_23616


namespace S9_is_27_l23_23338

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (n : ℕ)
variable (d a1 : ℝ)

-- Definitions
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2 : ℝ) * (2 * a 1 + (n - 1) * d)

-- Hypotheses
axiom h1 : arithmetic_seq a d
axiom h2 : a 1 + d = 3 * (a 1 + 3 * d) - 6

-- Theorem to be proved
theorem S9_is_27 (h1 : arithmetic_seq a d) (h2 : a 1 + d = 3 * (a 1 + 3 * d) - 6) (h3 : sum_terms a S) :
  S 9 = 27 :=
  sorry

end S9_is_27_l23_23338


namespace quadratic_coefficients_l23_23082

theorem quadratic_coefficients :
  ∀ (x : ℝ), x^2 - x + 3 = 0 → (1, -1, 3) :=
by
  intro x
  intro h
  have quadratic_coeff : x^2 - x + 3 = 1 * x^2 + (-1) * x + 3 := by simp
  exact (1, -1, 3) 
  sorry

end quadratic_coefficients_l23_23082


namespace a_is_perfect_square_l23_23033

theorem a_is_perfect_square (a b : ℕ) (h : ∃ (k : ℕ), a^2 + b^2 + a = k * a * b) : ∃ n : ℕ, a = n^2 := by
  sorry

end a_is_perfect_square_l23_23033


namespace quadratic_eq_coefficients_l23_23077

theorem quadratic_eq_coefficients :
  ∃ (a b c : ℤ), (a = 1 ∧ b = -1 ∧ c = 3) ∧ (∀ x : ℤ, a * x^2 + b * x + c = x^2 - x + 3) :=
by
  use 1, -1, 3
  split
  { split; refl }
  { intro x
    simp }
  sorry

end quadratic_eq_coefficients_l23_23077


namespace max_area_of_triangle_OAB_l23_23320

noncomputable def ellipseE (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

def pointA (x1 y1 : ℝ) (m t : ℝ) : Prop :=
  x1 = ty1 + m ∧ (x1^2 / 2) + y1^2 = 1

def pointB (x2 y2 : ℝ) (m t : ℝ) : Prop :=
  x2 = ty2 + m ∧ (x2^2 / 2) + y2^2 = 1

def vectorPA (x1 y1 : ℝ) : ℝ × ℝ :=
  (x1 - 5/4, y1)

def vectorPB (x2 y2 : ℝ) : ℝ × ℝ :=
  (x2 - 5/4, y2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def area_triangle (O A B : ℝ × ℝ) : ℝ :=
  1/2 * abs (O.1 * A.2 + A.1 * B.2 + B.1 * O.2 - O.2 * A.1 - A.2 * B.1 - B.2 * O.1)

theorem max_area_of_triangle_OAB : ∀ (m t : ℝ)
    (O : ℝ × ℝ) (A B : ℝ × ℝ),
  (m > 3/4) →
  (O = (0, 0)) →
  (A.1 = t * A.2 + m ∧ (A.1^2 / 2) + A.2^2 = 1) →
  (B.1 = t * B.2 + m ∧ (B.1^2 / 2) + B.2^2 = 1) →
  (dot_product (vectorPA A.1 A.2) (vectorPB B.1 B.2)) = const →
  ∃ (a : ℝ), a ≤ (sqrt 2 / 2) ∧
    (∀ (a' : ℝ), a' = area_triangle O A B → a' ≤ a) :=
sorry

end max_area_of_triangle_OAB_l23_23320


namespace no_permutable_power_of_two_l23_23625

theorem no_permutable_power_of_two (N : ℕ) (h1 : ∃ k : ℕ, N = 2^k) (h2 : ∃ l : ℕ, N ≠ 2^l ∧ permute_eq_digits (N, 2^l)) : false :=
sorry

end no_permutable_power_of_two_l23_23625


namespace range_of_a_l23_23110

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x + 5 > 3 ∧ x > a ∧ x ≤ -2) ↔ a ≤ -2 :=
by
  sorry

end range_of_a_l23_23110


namespace probability_of_C_l23_23954

theorem probability_of_C (p_A p_B p_C : ℚ)
  (h_A : p_A = 2 / 7) 
  (h_B : p_B = 1 / 7)
  (h_C : p_C = (1 - p_A - p_B) / 3) :
  p_C = 4 / 21 :=
by 
  have h : 1 = p_A + p_B + 3 * p_C, by linarith,
  simp [h_A, h_B] at h,
  linarith [h].


end probability_of_C_l23_23954


namespace range_of_given_parabolic_function_l23_23496

noncomputable def range_of_parabola (a : ℝ) (b : ℝ) (c : ℝ) (x_low : ℝ) (x_high : ℝ) : 
  set ℝ :=
{ y : ℝ | ∃ x : ℝ, x_low ≤ x ∧ x ≤ x_high ∧ y = a*x^2 + b*x + c }

theorem range_of_given_parabolic_function : 
  range_of_parabola (-1/3) 0 2 (-1) 5 = set.Icc (-19/3) 2 :=
sorry

end range_of_given_parabolic_function_l23_23496


namespace perfect_square_condition_l23_23026

theorem perfect_square_condition (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : ∃ k : ℕ, a = k^2 :=
by
  sorry

end perfect_square_condition_l23_23026


namespace digit_2023_in_fractional_expansion_l23_23255

theorem digit_2023_in_fractional_expansion :
  ∃ d : ℕ, (d = 4) ∧ (∃ n_block : ℕ, n_block = 6 ∧ (∃ p : Nat, p = 2023 ∧ ∃ r : ℕ, r = p % n_block ∧ r = 1)) :=
sorry

end digit_2023_in_fractional_expansion_l23_23255


namespace nth_equation_l23_23356

open Nat

theorem nth_equation (n : ℕ) (hn : n > 0) : 
  (\sum i in Finset.range n, (-1)^(i+1) * (i+1)^2) = (-1)^(n+1) * (n * (n + 1) / 2) := by
  sorry

end nth_equation_l23_23356


namespace smallest_n_f_greater_21_l23_23423

-- Definition of the function f
def f (n : ℕ) : ℕ :=
  Nat.find (λ k, n ∣ Nat.factorial k)

-- Definition that n is a multiple of 21
def is_multiple_of_21 (n : ℕ) : Prop :=
  ∃ r : ℕ, n = 21 * r

-- The theorem we are proving
theorem smallest_n_f_greater_21 (n : ℕ) (h : is_multiple_of_21 n) : f(n) > 21 ↔ n = 483 :=
by {
  sorry
}

end smallest_n_f_greater_21_l23_23423


namespace probability_even_from_list_is_half_l23_23971

theorem probability_even_from_list_is_half :
  let l := [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  let total_count := l.length
  let even_count := (l.filter (λ n, n % 2 == 0)).length
  (even_count / total_count.toReal) = (1 / 2) :=
by
  let l := [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  let total_count := l.length
  let even_count := (l.filter (λ n, n % 2 == 0)).length
  show (even_count / total_count.toReal) = 1 / 2
  sorry

end probability_even_from_list_is_half_l23_23971


namespace box_volume_l23_23474

theorem box_volume (L W H : ℝ) (h1 : L * W = 120) (h2 : W * H = 72) (h3 : L * H = 60) : L * W * H = 720 := 
by sorry

end box_volume_l23_23474


namespace max_intersections_three_circles_one_line_l23_23530

theorem max_intersections_three_circles_one_line : 
  ∀ (C1 C2 C3 : Circle) (L : Line), 
  same_paper C1 C2 C3 L → 
  max_intersections C1 C2 C3 L = 12 := 
sorry

end max_intersections_three_circles_one_line_l23_23530


namespace lines_intersect_at_3_6_l23_23492

theorem lines_intersect_at_3_6 (c d : ℝ) 
  (h1 : 3 = 2 * 6 + c) 
  (h2 : 6 = 2 * 3 + d) : 
  c + d = -9 := by 
  sorry

end lines_intersect_at_3_6_l23_23492


namespace infinite_set_of_points_in_plane_l23_23166

noncomputable def infinite_set_of_points_exists : Prop :=
  ∃ (P : ℕ → ℝ × ℝ),
  (∀ i j k : ℕ, (i ≠ j ∧ j ≠ k ∧ i ≠ k) → ¬ collinear (P i) (P j) (P k)) ∧
  (∀ i j : ℕ, i ≠ j → is_rational (dist (P i) (P j)))

theorem infinite_set_of_points_in_plane :
  infinite_set_of_points_exists :=
sorry

end infinite_set_of_points_in_plane_l23_23166


namespace Liliane_Alice_relationship_l23_23785

variables (J_x J_y : ℝ)

def Liliane_apples := 1.35 * J_x
def Liliane_bananas := 0.80 * J_y

def Alice_apples := 1.15 * J_x
def Alice_bananas := J_y

theorem Liliane_Alice_relationship :
  (Liliane_apples J_x J_y - Alice_apples J_x J_y) / (Alice_apples J_x J_y) = 0.1739 ∧
  (Liliane_bananas J_x J_y - Alice_bananas J_x J_y) / (Alice_bananas J_x J_y) = -0.20 :=
by 
  sorry

end Liliane_Alice_relationship_l23_23785


namespace horse_revolutions_l23_23587

variable (distance_from_center_A distance_from_center_B : ℝ)
variable (revolutions_A revolutions_B : ℕ)
variable (circumference : ℝ → ℝ)

-- Definitions based on the conditions
def circumference (r : ℝ) := 2 * Real.pi * r
def distance_traveled (r : ℝ) (n : ℕ) := n * circumference r

-- Specific values provided in the problem
def distance_from_center_A := 36
def distance_from_center_B := 12
def revolutions_A := 40

-- The proof problem statement
theorem horse_revolutions :
  revolutions_B = 120 := sorry

end horse_revolutions_l23_23587


namespace initial_number_is_ten_l23_23591

theorem initial_number_is_ten (x : ℝ) : (x + 14) * 14 - 24 = 13 * 24 → x = 10 :=
by
  intro h
  have h_eq : (x + 14) * 14 - 24 = 312 := h
  have h_mult : (x + 14) * 14 = 336 := by linarith
  have h_div : x + 14 = 24 := by linarith
  exact eq_sub_of_add_eq h_div

end initial_number_is_ten_l23_23591


namespace corn_harvest_l23_23482

theorem corn_harvest (x y : ℕ) 
  (h1 : 4340 = x * y)
  (h2 : y < 40)
  (h3 : (y + 5) * (x + 14) = 5520) : 
  x = 124 ∧ y = 35 :=
by 
  have h4 := calc
    (y + 5) * (x + 14) = 5520       : by assumption
    ... = ((y + 5) * (x + 14))
  sorry

end corn_harvest_l23_23482


namespace perpendicular_diagonals_iff_point_exists_l23_23512

theorem perpendicular_diagonals_iff_point_exists
  (ABCD : ℝ) (P : ℝ → ℝ → ℝ)
  (angle_PAB : ℝ) (angle_PDC : ℝ) (angle_PBC : ℝ) (angle_PAD : ℝ)
  (angle_PCD : ℝ) (angle_PBA : ℝ) (angle_PDA : ℝ) (angle_PCB : ℝ) :
  (∃ P : ℝ × ℝ, angle_PAB P + angle_PDC P = 90 ∧ angle_PBC P + angle_PAD P = 90 ∧
   angle_PCD P + angle_PBA P = 90 ∧ angle_PDA P + angle_PCB P = 90)
  ↔ (∃ AC BD : ℝ × ℝ, AC ⟂ BD) :=
begin
  sorry
end

end perpendicular_diagonals_iff_point_exists_l23_23512


namespace num_valid_arrangements_l23_23835

-- Definitions:
open Set

def is_strictly_ascending (l : List ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → i < l.length → j < l.length → l.nth_le i sorry < l.nth_le j sorry

def is_strictly_descending (l : List ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → i < l.length → j < l.length → l.nth_le i sorry > l.nth_le j sorry

def valid_arrangement (l : List ℕ) (k : ℕ) : Prop :=
  l.length = 7 ∧ 1 ≤ k ∧ k ≤ 7 ∧
  (is_strictly_ascending (l.remove_nth k) ∨ is_strictly_descending (l.remove_nth k))

-- The theorem to prove:
theorem num_valid_arrangements : 
  {l : List ℕ // l.perm (List.range (1:ℕ) (7+1))} → (∃ n : ℕ, n = 14) :=
by
  intro l
  use 14
  sorry

end num_valid_arrangements_l23_23835


namespace conic_sections_with_foci_at_F2_zero_l23_23674

theorem conic_sections_with_foci_at_F2_zero (a b m n: ℝ) (h1 : a > b) (h2: b > 0) (h3: m > 0) (h4: n > 0) (h5: a^2 - b^2 = 4) (h6: m^2 + n^2 = 4):
  (∀ x y: ℝ, x^2 / (a^2) + y^2 / (b^2) = 1) ∧ (∀ x y: ℝ, x^2 / (11/60) + y^2 / (11/16) = 1) ∧ 
  ∀ x y: ℝ, x^2 / (m^2) - y^2 / (n^2) = 1 ∧ ∀ x y: ℝ, 5*x^2 / 4 - 5*y^2 / 16 = 1 := 
sorry

end conic_sections_with_foci_at_F2_zero_l23_23674


namespace counting_measure_properties_l23_23048

open Set MeasureTheory

-- Define the counting measure on the real numbers
def counting_measure (A : Set ℝ) : ℝ≥0∞ :=
  if finite A then ↑(A.to_finset.card) else ∞

-- Constructed measure space
def counting_measure_space : MeasureSpace ℝ :=
  ⟨counting_measure⟩

-- Theorem statement in Lean
theorem counting_measure_properties :
  let μ := counting_measure_space.to_measure in
  MeasureTheory.SigmaFinite μ ∧
  ¬∃ G : ℝ → ℝ, (∀ a b : ℝ, a < b → μ (Ioc a b) = ENNReal.ofReal (G b - G a)) ∧
  ∀ x : ℝ, ∀ U : Set ℝ, IsOpen U ∧ x ∈ U → μ U < ∞ :=
by
  sorry

end counting_measure_properties_l23_23048


namespace find_fraction_l23_23007

def point := (ℝ × ℝ × ℝ)

variables (O : point) (a b c : ℝ) (A B C : point)
variables (d e f : ℝ) (p q r : ℝ)

def on_plane (a b c p q r : ℝ) : Prop :=
  ∀ (x y z : ℝ), x / (2 * p) + y / (2 * q) + z / (2 * r) = 1 → 
  (x, y, z) = (a, b, c)

def sphere_center (O A B C D : point) (P : point) : Prop :=
  let dist_sq (P Q : point) : ℝ := (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2 in
  dist_sq O P = dist_sq A P ∧ dist_sq O P = dist_sq B P ∧ dist_sq O P = dist_sq C P ∧ 
  dist_sq O P = dist_sq D P

theorem find_fraction 
  (O : point) (a b c : ℝ) (A B C : point) (d e f : ℝ) (p q r : ℝ)
  (hA : A = (2 * p, 0, 0)) (hB : B = (0, 2 * q, 0)) (hC : C = (0, 0, 2 * r))
  (hPlane : on_plane a b c p q r a b c)
  (hSphere : sphere_center O A B C (d, e, f) (p, q, r)) :
  a / p + b / q + c / r = 2 := sorry

end find_fraction_l23_23007


namespace proof_set_size_bound_l23_23599

noncomputable theory

open_locale classical

theorem proof_set_size_bound
  (S : set (ℤ × ℤ × ℤ))
  (n : ℕ)
  (h1 : ∀ (x y z : ℤ), (x, y, z) ∈ S → 1 ≤ x ∧ x ≤ n ∧ 1 ≤ y ∧ y ≤ n ∧ 1 ≤ z ∧ z ≤ n)
  (h2 : ∀ ⦃p1 p2 p3 p4 p5 p6 : ℤ⦄, 
        p1 ≠ p2 ∨ p3 ≠ p4 ∨ p5 ≠ p6 → 
        ((p1, p3, p5) ∈ S ∧ (p2, p4, p6) ∈ S) → 
        (p1 - p2)^2 + (p3 - p4)^2 + (p5 - p6)^2 ≠ (p2 - p1)^2 + (p4 - p3)^2 + (p6 - p5)^2):
  |S| < min ((n + 2) * real.sqrt (n / 3 : ℝ)) (n * real.sqrt 6) :=
sorry

end proof_set_size_bound_l23_23599


namespace g_g_3_equals_72596100_over_3034921_l23_23238

noncomputable def g (x : ℚ) : ℚ := x^(-2) + x^(-2) / (1 + x^(-2))

theorem g_g_3_equals_72596100_over_3034921 : g(g(3)) = 72596100 / 3034921 := by
  sorry

end g_g_3_equals_72596100_over_3034921_l23_23238


namespace possible_values_of_deriv_l23_23809

noncomputable def differentiable_function (f : ℝ → ℝ) [∀ x ∈ set.Ioo 0 1, differentiable_at ℝ f x] 
  (h_deriv_cont : continuous_on (deriv f) (set.Ioo 0 1)) : Prop :=
∀ n : ℕ, ∀ a : ℕ, a < 2^n ∧ odd a → ∃ b : ℕ, b < 2^n ∧ odd b ∧ f (a / 2^n) = b / 2^n

theorem possible_values_of_deriv (f : ℝ → ℝ) 
  (h_diff_cont : differentiable_function f) :
  deriv f (1 / 2) = 1 ∨ deriv f (1 / 2) = -1 :=
sorry

end possible_values_of_deriv_l23_23809


namespace find_A_l23_23621

def clubsuit (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

theorem find_A (A : ℝ) : clubsuit A 6 = 31 → A = 10.5 :=
by
  intro h
  sorry

end find_A_l23_23621


namespace sum_of_areas_of_six_rectangles_eq_572_l23_23053

theorem sum_of_areas_of_six_rectangles_eq_572 :
  let lengths := [1, 3, 5, 7, 9, 11]
  let areas := lengths.map (λ x => 2 * x^2)
  areas.sum = 572 :=
by 
  sorry

end sum_of_areas_of_six_rectangles_eq_572_l23_23053


namespace range_of_a_l23_23714

open Set

variable (a : ℝ)

noncomputable def I := univ ℝ
noncomputable def A := {x : ℝ | x ≤ a + 1}
noncomputable def B := {x : ℝ | x ≥ 1}
noncomputable def complement_B := {x : ℝ | x < 1}

theorem range_of_a (h : A a ⊆ complement_B) : a < 0 := sorry

end range_of_a_l23_23714


namespace double_seven_eighth_l23_23135

theorem double_seven_eighth (n : ℕ) (h : n = 48) : 2 * (7 / 8 * n) = 84 := by
  sorry

end double_seven_eighth_l23_23135


namespace woman_wait_time_for_catchup_l23_23551

-- Definitions used in the problem conditions
def man_speed_per_hour : ℝ := 6
def woman_speed_per_hour : ℝ := 12
def waiting_time_minutes : ℝ := 10

-- Conversion factors
def minutes_per_hour : ℝ := 60

-- Speed in miles per minute
def man_speed_per_minute : ℝ := man_speed_per_hour / minutes_per_hour
def woman_speed_per_minute : ℝ := woman_speed_per_hour / minutes_per_hour

-- Distance covered by woman in 10 minutes
def distance_covered_by_woman : ℝ := woman_speed_per_minute * waiting_time_minutes

-- Relative speed in miles per minute
def relative_speed_per_minute : ℝ := (woman_speed_per_hour - man_speed_per_hour) / minutes_per_hour

-- Time required for man to catch up
def catch_up_time : ℝ := distance_covered_by_woman / relative_speed_per_minute

-- Statement to prove
theorem woman_wait_time_for_catchup : catch_up_time = 20 := by
  sorry

end woman_wait_time_for_catchup_l23_23551


namespace range_of_function_l23_23803

theorem range_of_function :
  ∀ x : ℝ,
  (0 < x ∧ x < (π / 2)) →
  ∃ y : ℝ, 
  y = (sin x - 2 * cos x + (32 / (125 * sin x * (1 - cos x)))) ∧ y ≥ 2 / 5 :=
sorry

end range_of_function_l23_23803


namespace tan_alpha_minus_beta_l23_23687

-- Defining the acute angles and the conditions given
variables (α β : ℝ)
variable (h1 : 0 < α ∧ α < π / 2)
variable (h2 : 0 < β ∧ β < π / 2)
variable (h3 : sin α - sin β = -1/2)
variable (h4 : cos α - cos β = 1/2)

-- The theorem to prove
theorem tan_alpha_minus_beta :
  tan (α - β) = -sqrt 7 / 3 :=
sorry

end tan_alpha_minus_beta_l23_23687


namespace person_completion_time_l23_23841

theorem person_completion_time (x : ℝ) (h₁ : Ashutosh can complete the job in 10 hours)
    (h₂ : The person works for 9 hours and completes part of the job)
    (h₃ : Ashutosh completes the remaining job in 4 hours) :
    x = 15 := 
by
    -- Translate to appropriate mathematical statements:
    let completion_rate_person := 1 / x
    let completion_rate_ashutosh := 1 / 10
    have work_done_person := 9 * completion_rate_person
    have work_done_ashutosh := 4 * completion_rate_ashutosh
    have total_work_done := work_done_person + work_done_ashutosh
    have full_job := 1
    have equation := total_work_done = full_job
    -- Solve equation for x
    sorry

end person_completion_time_l23_23841


namespace perfect_square_condition_l23_23027

theorem perfect_square_condition (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : ∃ k : ℕ, a = k^2 :=
by
  sorry

end perfect_square_condition_l23_23027


namespace simplify_radicals_l23_23463

theorem simplify_radicals :
  (Real.sqrt 18 * Real.cbrt 24 = 6 * Real.sqrt 2 * Real.cbrt 3) :=
by sorry

end simplify_radicals_l23_23463


namespace std_dev_5_8_11_l23_23506

-- Define the three numbers as constants
def x1 := 5
def x2 := 8
def x3 := 11

-- Mean of the numbers
def mean := (x1 + x2 + x3) / 3

-- Variance of the numbers
def variance := (1 / 3 : ℚ) * ((x1 - mean)^2 + (x2 - mean)^2 + (x3 - mean)^2)

-- Standard Deviation
def standard_deviation := Real.sqrt variance

-- Statement to prove
theorem std_dev_5_8_11 : standard_deviation = Real.sqrt 6 :=
by
  sorry -- Proof goes here

end std_dev_5_8_11_l23_23506


namespace lines_tangent_to_two_circles_l23_23453

-- Definitions for our problem setting
def point := ℝ × ℝ

def dist (p q : point) : ℝ :=
  real.sqrt ((p.fst - q.fst)^2 + (p.snd - q.snd)^2)

noncomputable def circle (center : point) (radius : ℝ) : set point :=
  { x | dist x center = radius }

def tangents_count (P Q : point) (radiusP radiusQ : ℝ) : ℕ :=
  sorry  -- This is a placeholder for the tangent line counting function

-- Problem statement
theorem lines_tangent_to_two_circles (P Q : point) :
  dist P Q = 8 →
  tangents_count P Q 3 4 = 4 :=
by
  intros hPQ
  sorry  -- Proof would be added here


end lines_tangent_to_two_circles_l23_23453


namespace workshop_workers_l23_23476

theorem workshop_workers (W N: ℕ) 
  (h1: 8000 * W = 70000 + 6000 * N) 
  (h2: W = 7 + N) : 
  W = 14 := 
  by 
    sorry

end workshop_workers_l23_23476


namespace sufficient_not_necessary_l23_23725

theorem sufficient_not_necessary (x : ℝ) : (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 1) ∧ ¬((x ≠ 1) → (x^2 - 3 * x + 2 ≠ 0)) :=
by
  sorry

end sufficient_not_necessary_l23_23725


namespace even_factors_count_l23_23732

theorem even_factors_count (n : ℕ) (h : n = 2^4 * 3^2 * 5 * 7) : 
  ∃ k : ℕ, k = 48 ∧ ∃ a b c d : ℕ, 
  1 ≤ a ∧ a ≤ 4 ∧
  0 ≤ b ∧ b ≤ 2 ∧
  0 ≤ c ∧ c ≤ 1 ∧
  0 ≤ d ∧ d ≤ 1 ∧
  k = (4 - 1 + 1) * (2 + 1) * (1 + 1) * (1 + 1) := by
  sorry

end even_factors_count_l23_23732


namespace parabola_line_through_focus_intersection_has_length_8_l23_23957

noncomputable def parabola_line_intersection_length (focus_x focus_y : ℝ) (x1 x2 : ℝ) (y1 y2 : ℝ) : ℝ :=
  if y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 ∧ x1 + x2 = 6 then
    real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  else
    0

theorem parabola_line_through_focus_intersection_has_length_8 :
  ∀ A B : ℝ × ℝ,
  A.1 + B.1 = 6 ∧ 
  A.2^2 = 4 * A.1 ∧ 
  B.2^2 = 4 * B.1 ∧ 
  parabola_line_intersection_length 1 0 A.1 B.1 A.2 B.2 = 8 :=
begin
  intros A B,
  sorry,
end

end parabola_line_through_focus_intersection_has_length_8_l23_23957


namespace simplify_fraction_l23_23461

theorem simplify_fraction :
  10 * (15 / 8) * (-40 / 45) = -(50 / 3) :=
sorry

end simplify_fraction_l23_23461


namespace a_is_perfect_square_l23_23034

theorem a_is_perfect_square (a b : ℕ) (h : ab ∣ (a^2 + b^2 + a)) : (∃ k : ℕ, a = k^2) :=
sorry

end a_is_perfect_square_l23_23034


namespace g_function_property_l23_23093

variable {g : ℝ → ℝ}
variable {a b : ℝ}

theorem g_function_property 
  (h1 : ∀ a c : ℝ, c^3 * g a = a^3 * g c)
  (h2 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 :=
  sorry

end g_function_property_l23_23093


namespace perpendicular_line_eq_equal_intercepts_lines_eq_l23_23337

-- Definitions of the given conditions
def point_A := (2, 3 : ℝ)

-- Problem 1: Equation of l1 when it is perpendicular to l2
def l2 := λ x y : ℝ, x + 2 * y + 4 = 0

theorem perpendicular_line_eq :
  (∀ x y : ℝ, l2 x y → l1 x y = 2 * x - y - 1 = 0 ) :=
  sorry

-- Problem 2: Equation of l1 with equal intercepts on coordinate axes
def equal_intercepts_line_eq1 := λ x y : ℝ, 3 * x - 2 * y = 0
def equal_intercepts_line_eq2 := λ x y : ℝ, x + y - 5 = 0

theorem equal_intercepts_lines_eq : 
  (l1 point_A.1 point_A.2 →
  (∀ x y : ℝ, (l1 x y = equal_intercepts_line_eq1 x y ∨ l1 x y = equal_intercepts_line_eq2 x y))) :=
  sorry

end perpendicular_line_eq_equal_intercepts_lines_eq_l23_23337


namespace CO_perpendicular_PQ_l23_23411

-- Definitions and assumptions extracted directly from the problem conditions
variables {A B C O P Q : Type} [MetricSpace Type]
variables (triangle : Triangle A B C)
variables (circumcenter : Circumcenter A B C O)
variables (circle : Circle O A B)
variables (intersect1 : Intersect (circle ∩ Line B C) P)
variables (intersect2 : Intersect (circle ∩ Line C A) Q)

-- Conclusively, we need to prove CO is perpendicular to PQ.
theorem CO_perpendicular_PQ : Perpendicular (Line C O) (Line P Q) :=
sorry

end CO_perpendicular_PQ_l23_23411


namespace prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l23_23755

variables (n k : ℕ) (p : ℝ)
variables (h_pos : 0 < p) (h_lt_one : p < 1)

-- Probability that exactly k gnomes fall
theorem prob_exactly_k_gnomes_fall (h_k_le_n : k ≤ n) :
  prob_speed (exactly_k_gnomes_fall n k p) = p * (1 - p)^(n - k) := sorry

-- Expected number of fallen gnomes
theorem expected_fallen_gnomes : 
  expected_falls n p = n + 1 - 1/p + (1 - p)^(n + 1)/p := sorry

end prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l23_23755


namespace sum_integer_solutions_in_interval_l23_23059

theorem sum_integer_solutions_in_interval :
  (∑ x in (set.Icc (-25 : ℤ) (25 : ℤ)) \ {x : ℤ | (x^2 + x - 56).sqrt - (x^2 + 25*x + 136).sqrt < 8 * ((x - 7) / (x + 8)).sqrt}, (x : ℤ)).sum = 267 :=
by
  sorry

end sum_integer_solutions_in_interval_l23_23059


namespace probability_at_least_eight_stayed_correct_l23_23042

noncomputable def probability_at_least_eight_stayed (n : ℕ) (c : ℕ) (p : ℚ) : ℚ :=
  let certain_count := c
  let unsure_count := n - c
  let k := 3
  let prob_eight := 
    (Nat.choose unsure_count k : ℚ) * (p^k) * ((1 - p)^(unsure_count - k))
  let prob_nine := p^unsure_count
  prob_eight + prob_nine

theorem probability_at_least_eight_stayed_correct :
  probability_at_least_eight_stayed 9 5 (3/7) = 513 / 2401 :=
by
  sorry

end probability_at_least_eight_stayed_correct_l23_23042


namespace cos_2x_identity_l23_23723

theorem cos_2x_identity (x : ℝ) (hx : sin x + cos x + tan x + cot x + sec x + csc x = 9) : 
  cos (2 * x) = 1 - 2 * (9 * sin (2 * x) - 2) ^ 2 := 
by 
  sorry

end cos_2x_identity_l23_23723


namespace intersection_A_B_l23_23656

noncomputable def A : Set ℝ := { y | ∃ x : ℝ, y = Real.sin x }
noncomputable def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_A_B : A ∩ B = { y | 0 ≤ y ∧ y ≤ 1 } :=
by 
  sorry

end intersection_A_B_l23_23656


namespace tan_roots_of_polynomial_l23_23836

theorem tan_roots_of_polynomial :
  ∀ r : ℕ, r < 15 ∧ Nat.coprime r 15 → 
  (let x := Real.tan (r * Real.pi / 15) in x^8 - 92 * x^6 + 134 * x^4 - 28 * x^2 + 1 = 0) :=
begin
  intros r hr,
  let x := Real.tan (r * Real.pi / 15),
  sorry,  -- Proof omitted
end

end tan_roots_of_polynomial_l23_23836


namespace total_problems_l23_23947

theorem total_problems (rounds problems_per_round : ℕ) (h1 : rounds = 7) (h2 : problems_per_round = 3) : 
  rounds * problems_per_round = 21 := by
  simp [h1, h2]
  sorry

end total_problems_l23_23947


namespace carpet_area_required_l23_23788

-- Define the dimensions of Section A and Section B
def sectionA_length : ℝ := 12
def sectionA_width : ℝ := 8
def sectionB_length : ℝ := 10
def sectionB_width : ℝ := 5

-- Calculate the areas assuming no significant reduction by the diagonal wall in Section A
def area_sectionA : ℝ := sectionA_length * sectionA_width
def area_sectionB : ℝ := sectionB_length * sectionB_width
def total_area : ℝ := area_sectionA + area_sectionB

-- Theorem stating the approximate carpet area needed for the whole floor
theorem carpet_area_required : total_area ≈ 146 := by
  have areaA : area_sectionA = 96 := by rfl
  have areaB : area_sectionB = 50 := by rfl
  have total : total_area = 146 := by rw [areaA, areaB]; rfl
  exact total

end carpet_area_required_l23_23788


namespace perimeter_change_l23_23734

theorem perimeter_change (s h : ℝ) 
  (h1 : 2 * (1.3 * s + 0.8 * h) = 2 * (s + h)) :
  (2 * (0.8 * s + 1.3 * h) = 1.1 * (2 * (s + h))) :=
by
  sorry

end perimeter_change_l23_23734


namespace range_of_a_l23_23344

-- Define the decreasing nature of the function and derive the required range for a
theorem range_of_a {a : ℝ} :
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y → log (2 - a*x) / log 2 > log (2 - a*y) / log 2) → 
  (0 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l23_23344


namespace color_count_3x3_grid_l23_23832

def count_colorings : ℕ := 
  let colorings_0_greens := 1
  let colorings_1_green := 3 * 3
  let colorings_2_greens := 3 * 3 * 2
  let colorings_3_greens := 6
  colorings_0_greens + colorings_1_green + colorings_2_greens + colorings_3_greens

theorem color_count_3x3_grid :
  count_colorings = 34 :=
by
  unfold count_colorings
  norm_num
  sorry

end color_count_3x3_grid_l23_23832


namespace find_N_product_l23_23610

variables (M L : ℤ) (N : ℤ)

theorem find_N_product
  (h1 : M = L + N)
  (h2 : M + 3 = (L + N + 3))
  (h3 : L - 5 = L - 5)
  (h4 : |(L + N + 3) - (L - 5)| = 4) :
  N = -4 ∨ N = -12 → (-4 * -12) = 48 :=
by sorry

end find_N_product_l23_23610


namespace colorful_prod_bounds_l23_23183

def is_colorful (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits = digits.nodup

def mirror_image (n : ℕ) : ℕ :=
  n.digits 10.reverse'.ofDigits 10

theorem colorful_prod_bounds :
  ∀ (a b : ℕ), is_colorful a → is_colorful b → mirror_image a = b → 
  (1000 ≤ a * b ∧ a * b ≤ 9999) ↔ (a * b = 1008 ∨ a * b = 8722) :=
by sorry

end colorful_prod_bounds_l23_23183


namespace rearrange_digits_divisible_by_7_l23_23675

theorem rearrange_digits_divisible_by_7 (N : ℕ) : 
  ∃ (a₃ a₂ a₁ a₀ : ℕ), 
  (({a₃, a₂, a₁, a₀} = {1, 9, 8, 4}) ∧ 7 ∣ (N + a₃ * 1000 + a₂ * 100 + a₁ * 10 + a₀)) :=
sorry

end rearrange_digits_divisible_by_7_l23_23675


namespace real_condition_l23_23265

noncomputable def z (x : ℝ) : ℂ :=
  (sin x + sin (2 * x) + complex.I * (2 * (cos x)^2 * (sin x) - tan x)) / (cos x - complex.I)

theorem real_condition (x : ℝ) :
  (∃ k : ℤ, x = k * real.pi) ↔ (∃ r : ℝ, z(x).im = 0) :=
sorry

end real_condition_l23_23265


namespace greatest_possible_area_ABCD_l23_23619

noncomputable def greatest_area (a b c d : ℝ) (ac bd : ℝ) 
  (cyclic : ∀ abc d, abc ∈ {a, b, c} → {
    Proposition.mk (Angle.parallel orthogonalProp abc d) true }) 
  (H1 : a + c = 12) 
  (H2 : b + d = 13) : Prop :=
  ∃ (Q : Quadrilateral ℝ), 
    Q.is_cyclic ∧ 
    Q.diagonals_perpendicular ∧ 
    Q.side_lengths = {a, b, c, d} ∧ 
    Q.area = 36

theorem greatest_possible_area_ABCD : 
  ∀ (a b c d ac bd : ℝ), 
  cyclic a b c d ∧ 
  ac⁡⊥ bd ∧ 
  a + c = 12 ∧ 
  b + d = 13 →
  (∃ (Q : Quadrilateral ℝ), 
    Q.is_cyclic ∧ 
    Q.diagonals_perpendicular ∧ 
    Q.side_lengths = {a, b, c, d} ∧ 
    Q.area = 36) :=
sorry

end greatest_possible_area_ABCD_l23_23619


namespace min_positive_period_f_increasing_interval_f_l23_23343

noncomputable def f (x : ℝ) : ℝ := (Math.sin x + Math.cos x) ^ 2 + Math.cos (2 * x)

theorem min_positive_period_f : ∃ p > 0, ∀ x, f (x + p) = f x ∧ p = Real.pi :=
by sorry

theorem increasing_interval_f : ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 8 → ∀ a ∈ Icc 0 x, ∀ b ∈ Icc 0 x, a < b → f a ≤ f b :=
by sorry

end min_positive_period_f_increasing_interval_f_l23_23343


namespace permutation_and_combination_results_l23_23227

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

def C (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem permutation_and_combination_results :
  A 5 2 = 20 ∧ C 6 3 + C 6 4 = 35 := by
  sorry

end permutation_and_combination_results_l23_23227


namespace trig_identity_l23_23549

theorem trig_identity (α : ℝ) :
  (4.52 * (sin (6 * α) + sin (7 * α) + sin (8 * α) + sin (9 * α)) / (cos (6 * α) + cos (7 * α) + cos (8 * α) + cos (9 * α))) = 4.52 * tan ((15 * α) / 2) :=
by
  sorry

end trig_identity_l23_23549


namespace arith_sqrt_9_is_3_l23_23847

-- Define the arithmetic square root of a number
def arith_sqrt (x : ℝ) : ℝ := Real.sqrt x  -- Assuming Real.sqrt is the non-negative root

-- Define the given number x and its arithmetic square root
def x : ℝ := 9
axiom sqr_x_non_neg : arith_sqrt 9 = 3

-- State the proof problem
theorem arith_sqrt_9_is_3 : arith_sqrt x = 3 := by
  sorry

end arith_sqrt_9_is_3_l23_23847


namespace ilya_incorrect_l23_23828

theorem ilya_incorrect (s t : ℝ) : ¬ (s + t = s * t ∧ s * t = s / t) :=
by
  sorry

end ilya_incorrect_l23_23828


namespace sqrt_eq_conditions_l23_23635

theorem sqrt_eq_conditions (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) :
    (sqrt (3 - x) + sqrt x = 2) ↔ (x = 1 + sqrt 2 ∨ x = 1 - sqrt 2) :=
by
  sorry

end sqrt_eq_conditions_l23_23635


namespace arithmetic_progression_a6_l23_23307

theorem arithmetic_progression_a6 (a1 d : ℤ) (h1 : a1 + (a1 + d) + (a1 + 2 * d) = 168) (h2 : (a1 + 4 * d) - (a1 + d) = 42) : 
  a1 + 5 * d = 3 := 
sorry

end arithmetic_progression_a6_l23_23307


namespace largest_value_is_E_l23_23544

def A := 3 + 1 + 2 + 5
def B := 3 * 1 + 2 + 5
def C := 3 + 1 * 2 + 5
def D := 3 + 1 + 2^2 + 5
def E := 3 * 1 * 2 * 5

theorem largest_value_is_E : E > max (max (max A B) (max C D)) sorry

end largest_value_is_E_l23_23544


namespace number_of_students_who_liked_both_l23_23748

theorem number_of_students_who_liked_both (n a b c : ℕ) (hn : n = 50) (ha : a = 28) (hb : b = 20) (hc : c = 14) : 
  let total_who_liked_one := n - c,
      total_liked_either := a + b 
  in total_liked_either - total_who_liked_one = 12 :=
by 
  sorry

end number_of_students_who_liked_both_l23_23748


namespace brendan_taxes_correct_l23_23973

-- Definitions based on conditions
def hourly_wage : ℝ := 6
def shifts : (ℕ × ℕ) := (2, 8)
def additional_shift : ℕ := 12
def tip_rate : ℝ := 12
def tax_rate : ℝ := 0.20
def tip_reporting_fraction : ℝ := 1 / 3

-- Calculation based on conditions
noncomputable def total_hours : ℕ := (shifts.1 * shifts.2) + additional_shift
noncomputable def wage_income : ℝ := hourly_wage * total_hours
noncomputable def total_tips : ℝ := tip_rate * total_hours
noncomputable def reported_tips : ℝ := total_tips * tip_reporting_fraction
noncomputable def total_reported_income : ℝ := wage_income + reported_tips
noncomputable def taxes_paid : ℝ := total_reported_income * tax_rate

-- The proof problem statement
theorem brendan_taxes_correct : taxes_paid = 56 := by {
  sorry
}

end brendan_taxes_correct_l23_23973


namespace problem_part1_problem_part2_l23_23563

noncomputable def rectangle (A B C D : Point) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ (∃ a b : ℝ, ∥A - B∥ = a ∧ ∥A - D∥ = b ∧ a ≠ b ∧
    ∥B - C∥ = a ∧ ∥C - D∥ = b)

noncomputable def midpoint (P Q R : Point) : Prop := 
  ∥P - R∥ = ∥Q - R∥

noncomputable def perpendicular (P Q R : Plane) : Prop := 
  ∀ x : Point, x ∈ P → ∀ y : Point, y ∈ Q → x - y ≠ 0

structure geometry := 
  (Point : Type)
  (Plane : Point → Point → Point → Prop)

open geometry

variables (geom : geometry)
variables (A B C D E F P G : geom.Point)
variables (plane_ABCD plane_PAF plane_PDF : geom.Plane)

def conditions :=
  rectangle A B C D ∧
  (∥A - D∥ = 4) ∧ (∥A - B∥ = 2) ∧
  midpoint A B E ∧ midpoint B C F ∧
  perpendicular P A plane_ABCD

theorem problem_part1 (h : conditions A B C D E F P plane_ABCD) :
  geom.perpendicular plane_PDF plane_PAF :=
sorry

theorem problem_part2 (h : conditions A B C D E F P plane_ABCD) (quarter_point_condition : geom.perpendicular A P plane_ABCD) :
  G = midpoint A P G ∧ P = AG :=
sorry

end problem_part1_problem_part2_l23_23563


namespace problem_l23_23689

noncomputable def f : ℝ → ℝ :=
λ x, if x ∈ Ico 0 2 then Real.log (x + 1) / Real.log 2 else 0 -- Piecewise representation

theorem problem (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_periodic : ∀ x, 0 ≤ x → f (x + 2) = f x)
  (h_piecewise : ∀ x, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2) :
  f (-2011) + f 2012 = -1 :=
sorry

end problem_l23_23689


namespace relationship_a_b_c_l23_23092

variables (f : ℝ → ℝ)

-- Conditions
axiom diff_f : differentiable ℝ f
axiom symm_f : ∀ x : ℝ, f (1 + x) = f (3 - x)
axiom mono_f : ∀ x : ℝ, x < 2 → (x - 2) * deriv f x < 0

-- Definitions for the specific values
def a := f 0
def b := f (1 / 2)
def c := f 3

-- Theorem: Prove the relationship between a, b, and c
theorem relationship_a_b_c : a < b ∧ b < c :=
by sorry

end relationship_a_b_c_l23_23092


namespace projection_of_u_onto_v_l23_23873

open Real

-- Definitions of vectors
def u : Fin 2 → ℝ := ![3, 4]
def w : Fin 2 → ℝ := ![6, -2]
def v : Fin 2 → ℝ := ![18 / 13, -6 / 13]

-- Proving that the projection of u onto v is as given
theorem projection_of_u_onto_v :
  let proj_u_v := (u ⬝ v / (v ⬝ v)) • v
  proj_u_v = ![1.5, -0.5] := by
    sorry

end projection_of_u_onto_v_l23_23873


namespace quadratic_coefficients_l23_23080

theorem quadratic_coefficients :
  ∀ (x : ℝ), x^2 - x + 3 = 0 → (1, -1, 3) :=
by
  intro x
  intro h
  have quadratic_coeff : x^2 - x + 3 = 1 * x^2 + (-1) * x + 3 := by simp
  exact (1, -1, 3) 
  sorry

end quadratic_coefficients_l23_23080


namespace smallest_angle_at_17_30_l23_23213

theorem smallest_angle_at_17_30 : ∀ (h m : ℕ), h = 5 → m = 30 →
  let angle := min (abs ((60 * h - 11 * m) / 2)) (360 - abs ((60 * h - 11 * m) / 2)) in
  angle = 15 :=
by intros h m h_eq m_eq; sorry

end smallest_angle_at_17_30_l23_23213


namespace roots_expression_value_l23_23333

theorem roots_expression_value (x1 x2 : ℝ) (h1 : x1 + x2 = 5) (h2 : x1 * x2 = 2) :
  2 * x1 - x1 * x2 + 2 * x2 = 8 :=
by
  sorry

end roots_expression_value_l23_23333


namespace x_value_unique_l23_23242

theorem x_value_unique (x : ℝ) (h : ∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7 = 0) :
  x = 3 / 2 :=
sorry

end x_value_unique_l23_23242


namespace probability_of_two_points_is_three_sevenths_l23_23066

/-- Define the problem's conditions and statement. -/
def num_choices (n : ℕ) : ℕ :=
  match n with
  | 1 => 4  -- choose 1 option from 4
  | 2 => 6  -- choose 2 options from 4 (binomial coefficient)
  | 3 => 4  -- choose 3 options from 4 (binomial coefficient)
  | _ => 0

def total_ways : ℕ := 14  -- Total combinations of choosing 1 to 3 options from 4

def two_points_ways : ℕ := 6  -- 3 ways for 1 correct, 3 ways for 2 correct (B, C, D combinations)

def probability_two_points : ℚ :=
  (two_points_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_two_points_is_three_sevenths :
  probability_two_points = (3 / 7 : ℚ) :=
sorry

end probability_of_two_points_is_three_sevenths_l23_23066


namespace monthly_pension_supplement_l23_23068

theorem monthly_pension_supplement 
  (initial_age : ℕ) 
  (start_age : ℕ)
  (contribution_period_years : ℕ) 
  (monthly_contribution : ℕ) 
  (annual_interest_rate : ℝ) 
  (retirement_age : ℕ) 
  (years_after_retirement : ℕ) :
  initial_age = 39 → 
  start_age = 40 →
  contribution_period_years = 20 →
  monthly_contribution = 7000 →
  annual_interest_rate = 0.09 →
  retirement_age = 60 →
  years_after_retirement = 15 →
  let annual_contribution := (monthly_contribution * 12 : ℕ)
  let future_value := annual_contribution * ((1 + annual_interest_rate) ^ contribution_period_years - 1) / annual_interest_rate * (1 + annual_interest_rate)
  let total_accumulation := future_value
  let monthly_supplement := total_accumulation / (years_after_retirement * 12) in
  monthly_supplement ≈ 26023.45 :=
begin
  intros h_initial_age h_start_age h_contribution_period h_monthly_contribution h_interest_rate h_retirement_age h_years_after_retirement,
  let annual_contribution := (monthly_contribution * 12 : ℕ),
  have h_annual_contribution : annual_contribution = 84000, by sorry,
  -- (continue with the definition using the factual computations if needed, ending with the approximate value)
  let future_value := annual_contribution * ((1 + annual_interest_rate) ^ contribution_period_years - 1) / annual_interest_rate * (1 + annual_interest_rate),
  have h_future_value : future_value ≈ 4684220.554, by sorry,
  let total_accumulation := future_value,
  let monthly_supplement := total_accumulation / (years_after_retirement * 12),
  have h_monthly_supplement : monthly_supplement ≈ 26023.45, by sorry,
  exact h_monthly_supplement
end

end monthly_pension_supplement_l23_23068


namespace max_intersections_three_circles_one_line_l23_23531

theorem max_intersections_three_circles_one_line : 
  ∀ (C1 C2 C3 : Circle) (L : Line), 
  same_paper C1 C2 C3 L → 
  max_intersections C1 C2 C3 L = 12 := 
sorry

end max_intersections_three_circles_one_line_l23_23531


namespace prob_X_distribution_prob_tie_prob_Y_distribution_expected_Y_l23_23448

def X := {-1, 0, 1}
def A_accuracy := 0.5
def B_accuracy := 0.6

theorem prob_X_distribution :
  ∀ (x : X),
  (x = -1) → (P(X = -1) = 0.3) ∧
  (x = 0) → (P(X = 0) = 0.5) ∧
  (x = 1) → (P(X = 1) = 0.2) := by sorry

theorem prob_tie :
  P(tie) = 0.2569 := by sorry

def Y := {2, 3, 4}

theorem prob_Y_distribution :
  ∀ (y : Y),
  (y = 2) → (P(Y = 2) = 0.13) ∧
  (y = 3) → (P(Y = 3) = 0.13) ∧
  (y = 4) → (P(Y = 4) = 0.74) := by sorry

theorem expected_Y :
  E(Y) = 3.61 := by sorry

end prob_X_distribution_prob_tie_prob_Y_distribution_expected_Y_l23_23448


namespace set_intersection_l23_23327

noncomputable def A : Set ℝ := {x | 2^x > 1}
noncomputable def B : Set ℝ := {x | Real.log x > 1}
noncomputable def C : Set ℝ := {x | 0 < x ∧ x ≤ Real.exp 1}

theorem set_intersection :
  A ∩ (Set.univ \ B) = C := sorry

end set_intersection_l23_23327


namespace find_coords_P_l23_23362

variables {M N P : ℝ × ℝ}
def M : ℝ × ℝ := (3, 2)
def N : ℝ × ℝ := (-5, -5)
def vec (A B : ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) := (k * v.1, k * v.2)

theorem find_coords_P (x y : ℝ) (hP : P = (x, y)) 
  (h : vec M P = scalar_mul (1/2) (vec M N)) : 
  P = (-1, -3/2) :=
sorry

end find_coords_P_l23_23362


namespace meals_neither_vegan_kosher_nor_gluten_free_l23_23434

def total_clients : ℕ := 50
def n_vegan : ℕ := 10
def n_kosher : ℕ := 12
def n_gluten_free : ℕ := 6
def n_both_vegan_kosher : ℕ := 3
def n_both_vegan_gluten_free : ℕ := 4
def n_both_kosher_gluten_free : ℕ := 2
def n_all_three : ℕ := 1

/-- The number of clients who need a meal that is neither vegan, kosher, nor gluten-free. --/
theorem meals_neither_vegan_kosher_nor_gluten_free :
  total_clients - (n_vegan + n_kosher + n_gluten_free - n_both_vegan_kosher - n_both_vegan_gluten_free - n_both_kosher_gluten_free + n_all_three) = 30 :=
by
  sorry

end meals_neither_vegan_kosher_nor_gluten_free_l23_23434


namespace nat_perfect_square_l23_23041

theorem nat_perfect_square (a b : ℕ) (h : ∃ k : ℕ, a^2 + b^2 + a = k * a * b) : ∃ m : ℕ, a = m * m := by
  sorry

end nat_perfect_square_l23_23041


namespace white_square_area_l23_23817

theorem white_square_area
    (edge_length : ℝ)
    (total_paint : ℝ)
    (total_surface_area : ℝ)
    (green_paint_per_face : ℝ)
    (white_square_area_per_face: ℝ) :
    edge_length = 12 →
    total_paint = 432 →
    total_surface_area = 6 * (edge_length ^ 2) →
    green_paint_per_face = total_paint / 6 →
    white_square_area_per_face = (edge_length ^ 2) - green_paint_per_face →
    white_square_area_per_face = 72
:= sorry

end white_square_area_l23_23817


namespace limit_of_sequence_l23_23931

open Real

theorem limit_of_sequence :
  (∀ (n : ℕ), ((∑ i in finset.range (n + 1), ↑i) : ℝ) = (n * (n + 1)) / 2) →
  (∃ l : ℝ, tendsto (λ n : ℕ, (∑ i in finset.range (n + 1), ↑i) / sqrt (9 * (n : ℝ)^4 + 1)) at_top (𝓝 l) ∧ l = 1 / 6) :=
by
  intro H_sum
  have H : ∑ i in finset.range (n + 1), i = n * (n + 1) / 2 := H_sum n
  sorry

end limit_of_sequence_l23_23931


namespace probability_B_given_A_l23_23890

-- Define the events A and B
def event_A (x y : ℕ) : Prop :=
  (x % 2 = 1) ∧ (y % 2 = 1)

def event_B (x y : ℕ) : Prop :=
  (x + y = 4)

-- Define the probability measure on the space of dice rolls
noncomputable def P (s : set (ℕ × ℕ)) : ℚ :=
  (s.to_finset.card : ℚ) / 36

-- Define the conditional probability P(B|A)
noncomputable def P_B_given_A : ℚ :=
  let A_outcomes := {pair | event_A pair.fst pair.snd} in
  let B_given_A_outcomes := {pair | event_A pair.fst pair.snd ∧ event_B pair.fst pair.snd} in
  (B_given_A_outcomes.to_finset.card : ℚ) / (A_outcomes.to_finset.card : ℚ)

-- Prove that P(B|A) is 2/9
theorem probability_B_given_A : P_B_given_A = 2 / 9 :=
by
  sorry

end probability_B_given_A_l23_23890


namespace limit_of_sequence_l23_23932

open Real

theorem limit_of_sequence :
  (∀ (n : ℕ), ((∑ i in finset.range (n + 1), ↑i) : ℝ) = (n * (n + 1)) / 2) →
  (∃ l : ℝ, tendsto (λ n : ℕ, (∑ i in finset.range (n + 1), ↑i) / sqrt (9 * (n : ℝ)^4 + 1)) at_top (𝓝 l) ∧ l = 1 / 6) :=
by
  intro H_sum
  have H : ∑ i in finset.range (n + 1), i = n * (n + 1) / 2 := H_sum n
  sorry

end limit_of_sequence_l23_23932


namespace vector_at_t_4_l23_23956

-- Define the problem statements and parameters
def vector_at_t_1 : ℝ × ℝ × ℝ := (4, 5, 9)
def vector_at_t_3 : ℝ × ℝ × ℝ := (1, 0, -2)

def vector_on_line (t : ℝ) (a d : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.1 + t * d.1, a.2 + t * d.2, a.3 + t * d.3)

theorem vector_at_t_4 :
  ∃ (a d : ℝ × ℝ × ℝ),
    vector_at_t_1 = vector_on_line 1 a d ∧
    vector_at_t_3 = vector_on_line 3 a d ∧
    vector_on_line 4 a d = (-1, 0, -15) :=
by
  sorry

end vector_at_t_4_l23_23956


namespace map_scale_l23_23553

theorem map_scale 
  (distance_on_map : ℝ) 
  (time_travelled : ℝ) 
  (speed : ℝ) 
  (actual_distance : ℝ := time_travelled * speed) 
  (scale : ℝ := distance_on_map / actual_distance) 
  (distance_on_map = 5) 
  (time_travelled = 3.5) 
  (speed = 60) : 
  scale = 1 / 42 := 
by
  sorry

end map_scale_l23_23553


namespace max_intersections_three_circles_one_line_l23_23527

theorem max_intersections_three_circles_one_line (c1 c2 c3 : Circle) (L : Line) :
  greatest_number_points_of_intersection c1 c2 c3 L = 12 :=
sorry

end max_intersections_three_circles_one_line_l23_23527


namespace largest_perfect_square_factor_of_7560_l23_23137

theorem largest_perfect_square_factor_of_7560 :
  ∃ n : ℕ, is_largest_perfect_square_factor 7560 n ∧ n = 36 := sorry

-- Helper definition to specify the largest perfect square factor
def is_largest_perfect_square_factor (n m : ℕ) : Prop :=
  m^2 ∣ n ∧ ∀ k : ℕ, (k^2 ∣ n → k^2 ≤ m^2)

end largest_perfect_square_factor_of_7560_l23_23137


namespace sequence_finite_values_l23_23798

noncomputable def g (x : ℝ) : ℝ := x^2 - 6 * x + 8

def x_sequence (x0 : ℝ) : ℕ → ℝ
| 0 := x0
| (n + 1) := g (x_sequence n)

theorem sequence_finite_values (x0 : ℝ) : 
  ∀ x0, ¬(∃ N, ∀ n m : ℕ, n < N → m < N → x_sequence x0 n = x_sequence x0 m) :=
sorry

end sequence_finite_values_l23_23798


namespace chromium_percentage_l23_23926

noncomputable def chromium_percentage_in_new_alloy 
    (chromium_percentage_first: ℝ) 
    (weight_first: ℝ) 
    (chromium_percentage_second: ℝ) 
    (weight_second: ℝ) : ℝ :=
    (((chromium_percentage_first * weight_first / 100) + (chromium_percentage_second * weight_second / 100)) 
    / (weight_first + weight_second)) * 100

theorem chromium_percentage 
    (chromium_percentage_first: ℝ) 
    (weight_first: ℝ) 
    (chromium_percentage_second: ℝ) 
    (weight_second: ℝ) 
    (h1 : chromium_percentage_first = 10) 
    (h2 : weight_first = 15) 
    (h3 : chromium_percentage_second = 8) 
    (h4 : weight_second = 35) :
    chromium_percentage_in_new_alloy chromium_percentage_first weight_first chromium_percentage_second weight_second = 8.6 :=
by 
  rw [h1, h2, h3, h4]
  simp [chromium_percentage_in_new_alloy]
  norm_num


end chromium_percentage_l23_23926


namespace probability_f_leq_zero_l23_23702

noncomputable def f (k x : ℝ) : ℝ := k * x - 1

theorem probability_f_leq_zero : 
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) →
  (∀ k ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f k x ≤ 0) →
  (∃ k ∈ Set.Icc (-2 : ℝ) (1 : ℝ), f k x ≤ 0) →
  ((1 - (-2)) / (2 - (-2)) = 3 / 4) :=
by sorry

end probability_f_leq_zero_l23_23702


namespace triangle_count_l23_23367

-- Define the conditions of the problem

-- There is a large rectangle divided into 4 smaller rectangles of equal size
def large_rectangle_divided_into_4_equal_smaller_rectangles : Prop := 
  ∃ (r : Rectangle), 
    ∃ (r₁ r₂ r₃ r₄ : Rectangle), 
      r₁.area = r₂.area ∧ r₂.area = r₃.area ∧ r₃.area = r₄.area ∧ 
      r1.height = r.height / 2 ∧ r2.height = r.height / 2 ∧
      r3.height = r.height / 2 ∧ r4.height = r.height / 2 ∧
      r1.width = r.width / 2 ∧ r2.width = r.width / 2 ∧
      r3.width = r.width / 2 ∧ r4.width = r.width / 2 ∧
      r1 ∈ r ∧ r2 ∈ r ∧ r3 ∈ r ∧ r4 ∈ r

-- Each smaller rectangle is divided by a diagonal line from the bottom left to the top right corner
def each_smaller_rectangle_divided_diagonal : Prop :=
  ∀ (r : Rectangle), 
    (r ∈ r₁ ∨ r ∈ r₂ ∨ r ∈ r₃ ∨ r ∈ r₄) → 
      ∃ (d : Diagonal), d ∈ r ∧ (d.start = r.bottom_left ∧ d.end = r.top_right)

-- Additional vertical lines divide each smaller rectangle into two right triangles
def additional_vertical_lines_in_smaller_rectangles : Prop :=
  ∀ (r : Rectangle), 
    (r ∈ r₁ ∨ r ∈ r₂ ∨ r ∈ r₃ ∨ r ∈ r₄) → 
      ∃ (v : Line), 
        v.start.y = r.bottom.y ∧ v.end.y = r.top.y ∧ v.start.x = v.end.x ∧ 
        v.start.x ≠ r.left ∧ v.start.x ≠ r.right ∧ divides_into_two_right_triangles r v  -- Assuming divides_into_two_right_triangles is defined

-- Now we state the problem using the above conditions

theorem triangle_count (r : Rectangle)
  (h₁ : large_rectangle_divided_into_4_equal_smaller_rectangles)
  (h₂ : each_smaller_rectangle_divided_diagonal)
  (h₃ : additional_vertical_lines_in_smaller_rectangles) : 
  count_triangles r = 20 := 
sorry

end triangle_count_l23_23367


namespace rectangle_area_is_1638_l23_23960

-- Define the width of the rectangle
def width : ℝ := 42

-- Define the total length of ten rectangles
def total_length_of_ten_rectangles : ℝ := 390

-- Calculate the length of one rectangle
def length_of_one_rectangle : ℝ := total_length_of_ten_rectangles / 10

-- The area of one rectangle
def area_of_rectangle : ℝ := width * length_of_one_rectangle

-- The theorem to prove the area is 1638 square inches
theorem rectangle_area_is_1638 : area_of_rectangle = 1638 := 
sorry

end rectangle_area_is_1638_l23_23960


namespace Cameron_list_count_l23_23981

theorem Cameron_list_count : 
  let lower_bound := 900
  let upper_bound := 27000
  let step := 30
  let n_min := lower_bound / step
  let n_max := upper_bound / step
  n_max - n_min + 1 = 871 :=
by
  sorry

end Cameron_list_count_l23_23981


namespace length_of_rectangular_floor_is_correct_l23_23861

def breadth (x : ℝ) := x
def length (x : ℝ) := 3 * x
def area (x : ℝ) := length x * breadth x
def painting_cost := 100
def rate_per_sq_meter := 2
def calculated_area := painting_cost / rate_per_sq_meter
def correct_length := 12.24

theorem length_of_rectangular_floor_is_correct (x : ℝ) (h : area x = calculated_area) :
  length x ≈ correct_length :=
by 
  sorry

end length_of_rectangular_floor_is_correct_l23_23861


namespace common_ratio_l23_23283

theorem common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) (h1 : ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q))
(h2 : S 3 = 3 * a 0) (h3 : a 1 = a 0 * q) (h4 : a 2 = a 0 * q ^ 2) : q = 1 ∨ q = -2 := 
sorrry

end common_ratio_l23_23283


namespace triangle_ABC_l23_23782

-- Given conditions and definitions

variables {A B C : Type} [real A B C]

def CA : ℝ := 1
def CB : ℝ := 2
def angleC : ℝ := 60 / 180 * real.pi -- Converting 60 degrees to radians

-- Proof statements to be proven

theorem triangle_ABC (CA_eq : CA = 1) 
                     (CB_eq : CB = 2) 
                     (angle_C_eq : angleC = real.pi / 3) :
  ∃ (AB : ℝ) (angleA : ℝ) (S_triangle_ABC : ℝ),
      AB = real.sqrt(3) ∧
      angleA = real.pi / 2 ∧ 
      S_triangle_ABC = real.sqrt(3) / 2 :=
sorry

end triangle_ABC_l23_23782


namespace negative_sum_l23_23679

theorem negative_sum (a b c x y z : ℝ) (h1 : 0 < b - c) (h2 : b - c < a) (h3 : a < b + c)
  (h4 : ax + by + cz = 0) : ayz + bzx + cxy < 0 :=
sorry

end negative_sum_l23_23679


namespace polar_equation_of_curve_C_minimum_length_of_tangent_l23_23690

-- Definitions based on the conditions provided
def parametric_equation_curve_C (α : ℝ) : ℝ × ℝ :=
  (3 + 3 * Real.cos α, 2 * Real.sin α)

def polar_equation_line_l (ρ θ : ℝ) : Prop :=
  √2 * ρ * Real.sin (θ - π / 4) = 1

-- Proving the statements
theorem polar_equation_of_curve_C :
  ∃ ρ θ : ℝ, (parametric_equation_curve_C α).fst = ρ * Real.cos θ ∧ (parametric_equation_curve_C α).snd = ρ * Real.sin θ → 
  ρ^2 - 6 * ρ * Real.cos θ + 5 = 0 := 
sorry

theorem minimum_length_of_tangent :
  ∃ d : ℝ, (polar_equation_line_l ρ θ) → 
  let center_distance := (3 - 0 + 1) / √2 in
  center_distance = 2 * √2 → 
  min_tangent_length = (Real.sqrt ((2 * √2)^2 - 2^2)) = 2 :=
sorry

end polar_equation_of_curve_C_minimum_length_of_tangent_l23_23690


namespace share_of_A_in_profit_l23_23175

noncomputable def calculate_investment_share
  (init_invest_A : ℝ) (init_invest_B : ℝ)
  (invest_change_A_7 : ℝ) (invest_change_B_7 : ℝ)
  (invest_change_A_11 : ℝ) (invest_change_B_11 : ℝ)
  (total_profit : ℝ)
  : ℝ :=
let invest_A_1_to_6 := init_invest_A * 6 in
let invest_B_1_to_6 := init_invest_B * 6 in
let invest_A_7_to_10 := (init_invest_A + invest_change_A_7) * 4 in
let invest_B_7_to_10 := (init_invest_B + invest_change_B_7) * 4 in
let invest_A_11_to_12 := (init_invest_A + invest_change_A_7 + invest_change_A_11) * 2 in
let invest_B_11_to_12 := (init_invest_B + invest_change_B_7 + invest_change_B_11) * 2 in
let total_invest_A := invest_A_1_to_6 + invest_A_7_to_10 + invest_A_11_to_12 in
let total_invest_B := invest_B_1_to_6 + invest_B_7_to_10 + invest_B_11_to_12 in
let ratio_A := total_invest_A / (total_invest_A + total_invest_B) in
total_profit * ratio_A

theorem share_of_A_in_profit :
  calculate_investment_share 5000 8000 2000 (-3000) (-1500) 2000 9500 ≈ 4341.06 :=
sorry

end share_of_A_in_profit_l23_23175


namespace parabola_vertex_locus_l23_23497

def parabola_locus (a b : ℝ) (vertex : ℝ × ℝ) : Prop :=
  let s := -b / (2 * a) in
  let t := (4 * a - b^2) / (4 * a) in
  8 * a^2 + 4 * a * b = b^3 → vertex = (s, t) ∧ s * t = 1

theorem parabola_vertex_locus (a b s t : ℝ) (h_condition : 8 * a^2 + 4 * a * b = b^3) : 
  let s := -b / (2 * a) in
  let t := (4 * a - b^2) / (4 * a) in
  (s, t) = (s, t) → s * t = 1 :=
sorry

end parabola_vertex_locus_l23_23497


namespace prob_k_gnomes_fall_exp_gnomes_falling_l23_23769

variables (n k : ℕ) (p : ℝ)
hypotheses 
  (hn : 0 < n)
  (hp : 0 < p) (hp1 : p < 1)
  (hk : 0 ≤ k) (hk1 : k ≤ n)

open ProbabilityTheory
  
def probability_k_gnomes_fall := 
  p * (1 - p) ^ (n - k)

def expected_gnomes_fall :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p

theorem prob_k_gnomes_fall (hprob : 0 < p ∧ p < 1) : 
  ∀ n k : ℕ, 0 ≤ k ∧ k ≤ n → probability_k_gnomes_fall n k p = p * (1 - p) ^ (n - k) :=
by sorry

theorem exp_gnomes_falling (hprob : 0 < p ∧ p < 1) : 
  ∀ n : ℕ, 0 < n → expected_gnomes_fall n p = n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p :=
by sorry

end prob_k_gnomes_fall_exp_gnomes_falling_l23_23769


namespace min_area_quadrilateral_PACB_l23_23332

noncomputable theory

-- Define the center of the circle C(1, 1)
def C := (1, 1)

-- Define the radius of the circle r = 1
def r := 1

-- Define the line equation on which point P lies: 3x + 4y + 8 = 0
def line (x y : ℝ) := 3 * x + 4 * y + 8 = 0

-- Define the circle equation x^2 + y^2 - 2x - 2y + 1 = 0
def circle (x y : ℝ) := x^2 + y^2 - 2 * x - 2 * y + 1 = 0

-- Define the minimum distance d from the center C to the line (using the distance formula)
def min_distance :=
  let numerator := abs (3 * 1 + 4 * 1 + 8) in
  let denominator := real.sqrt (3 ^ 2 + 4 ^ 2) in
  numerator / denominator

-- The lengths PA and PB of the tangent lines to the circle
def tangent_length := real.sqrt (min_distance ^ 2 - r ^ 2)

-- Minimum area of the quadrilateral PACB
def min_area := 2 * (1 / 2 * tangent_length * r)

theorem min_area_quadrilateral_PACB : min_area = 2 * real.sqrt 2 := sorry

end min_area_quadrilateral_PACB_l23_23332


namespace find_sixth_term_l23_23304

open Nat

-- Given conditions
def arithmetic_progression (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ (n : ℕ), a (n + 1) = a n + d

def sum_of_first_three_terms (a : ℕ → ℤ) : Prop :=
  a 1 + a 2 + a 3 = 168

def second_minus_fifth (a : ℕ → ℤ) : Prop :=
  a 2 - a 5 = 42

-- Prove question == answer given conditions
theorem find_sixth_term :
  ∀ (a : ℕ → ℤ), arithmetic_progression a → sum_of_first_three_terms a → second_minus_fifth a → a 6 = 0 :=
by
  sorry

end find_sixth_term_l23_23304


namespace cannot_form_square_with_sticks_l23_23122

theorem cannot_form_square_with_sticks
    (num_1cm_sticks : ℕ)
    (num_2cm_sticks : ℕ)
    (num_3cm_sticks : ℕ)
    (num_4cm_sticks : ℕ)
    (len_1cm_stick : ℕ)
    (len_2cm_stick : ℕ)
    (len_3cm_stick : ℕ)
    (len_4cm_stick : ℕ)
    (sum_lengths : ℕ) :
    num_1cm_sticks = 6 →
    num_2cm_sticks = 3 →
    num_3cm_sticks = 6 →
    num_4cm_sticks = 5 →
    len_1cm_stick = 1 →
    len_2cm_stick = 2 →
    len_3cm_stick = 3 →
    len_4cm_stick = 4 →
    sum_lengths = num_1cm_sticks * len_1cm_stick + 
                  num_2cm_sticks * len_2cm_stick + 
                  num_3cm_sticks * len_3cm_stick + 
                  num_4cm_sticks * len_4cm_stick →
    ∃ (s : ℕ), sum_lengths = 4 * s → False := 
by
  intros num_1cm_sticks_eq num_2cm_sticks_eq num_3cm_sticks_eq num_4cm_sticks_eq
         len_1cm_stick_eq len_2cm_stick_eq len_3cm_stick_eq len_4cm_stick_eq
         sum_lengths_def

  sorry

end cannot_form_square_with_sticks_l23_23122


namespace sin_double_angle_plus_pi_over_three_l23_23015

theorem sin_double_angle_plus_pi_over_three (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos (α + π / 6) = 4 / 5) :
  sin (2 * α + π / 3) = 24 / 25 :=
begin
  sorry
end

end sin_double_angle_plus_pi_over_three_l23_23015


namespace remainder_division_twice_l23_23002

theorem remainder_division_twice (q1 r1 q2 r2 : Polynomial ℚ)
  (h1 : Polynomial.divMod (X^10) (X - (⅓ : ℚ)) = (q1, r1))
  (h2 : Polynomial.divMod q1 (X - (⅓ : ℚ)) = (q2, r2)) :
  r2 = (1 : ℚ) / 19683 := 
sorry

end remainder_division_twice_l23_23002


namespace height_flagstaff_l23_23187

variables (s_1 s_2 h_2 : ℝ)
variable (h : ℝ)

-- Define the conditions as given
def shadow_flagstaff := s_1 = 40.25
def shadow_building := s_2 = 28.75
def height_building := h_2 = 12.5
def similar_triangles := (h / s_1) = (h_2 / s_2)

-- Prove the height of the flagstaff
theorem height_flagstaff : shadow_flagstaff s_1 ∧ shadow_building s_2 ∧ height_building h_2 ∧ similar_triangles h s_1 h_2 s_2 → h = 17.5 :=
by sorry

end height_flagstaff_l23_23187


namespace deborah_oranges_zero_l23_23121

-- Definitions for given conditions.
def initial_oranges : Float := 55.0
def oranges_added_by_susan : Float := 35.0
def total_oranges_after : Float := 90.0

-- Defining Deborah's oranges in her bag.
def oranges_in_bag : Float := total_oranges_after - (initial_oranges + oranges_added_by_susan)

-- The theorem to be proved.
theorem deborah_oranges_zero : oranges_in_bag = 0 := by
  -- Placeholder for the proof.
  sorry

end deborah_oranges_zero_l23_23121


namespace probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l23_23759

noncomputable def probability_k_gnomes_fall (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p * (1 - p) ^ (n - k)

noncomputable def expected_number_of_fallen_gnomes (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p)

theorem probability_k_gnomes_fall_correct (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  probability_k_gnomes_fall n k p h = p * (1 - p) ^ (n - k) :=
by sorry

theorem expected_number_of_fallen_gnomes_correct (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  expected_number_of_fallen_gnomes n p h = n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p) :=
by sorry

end probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l23_23759


namespace half_sum_same_color_l23_23866

noncomputable theory
open_locale classical

variables {N : ℕ} -- Number of colors
variables {colors : ℕ → ℕ} -- The coloring function

-- Conditions: 
-- 1. Natural numbers are painted in N colors
-- 2. There are infinitely many numbers of each color (this is implicitly handled by assuming the function is well-defined over ℕ)

-- We assume a function half_sum_color that maps the color of the half-sum of two numbers of the same parity based on their colors
variables (colors_half_sum : ∀ a b, a % 2 = b % 2 → colors (a + b) / 2 = f (colors a) (colors b))

theorem half_sum_same_color
  {a b : ℕ} (ha : a % 2 = b % 2) (hc : colors a = colors b) :
  colors (a + b) / 2 = colors a := 
sorry

end half_sum_same_color_l23_23866


namespace exists_right_triangle_area_eq_perimeter_l23_23626

theorem exists_right_triangle_area_eq_perimeter :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a + b + c = (a * b) / 2 ∧ a ≠ b ∧ 
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 12 ∧ b = 5 ∧ c = 13) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10)) :=
by
  sorry

end exists_right_triangle_area_eq_perimeter_l23_23626


namespace circumcircles_common_point_and_line_passes_midpoint_l23_23781

open EuclideanGeometry

variables {A B C B' C' H I M : Point}

-- Definitions derived from the conditions
def midpoint (P Q : Point) : Point :=
  (P + Q) / 2

-- Hypotheses
axiom triangle_abc : Triangle A B C
axiom midpoint_B' : B' = midpoint A C
axiom midpoint_C' : C' = midpoint A B
axiom foot_H : foot (lineThrough A B) (lineThrough A C) = H

-- Theorem statement
theorem circumcircles_common_point_and_line_passes_midpoint :
  ∃ (I : Point), (circumcircle (triangle A B' C')).circle I ∧ 
                 (circumcircle (triangle B C' H)).circle I ∧ 
                 (circumcircle (triangle B' C H)).circle I ∧ 
                 lineThrough H I = lineThrough H M ∧
                 M = midpoint B' C' :=
sorry

end circumcircles_common_point_and_line_passes_midpoint_l23_23781


namespace average_marks_of_second_class_is_60_l23_23849

noncomputable def average_marks_second_class (avg_marks_1st_class : ℕ) (num_students_1st_class : ℕ) (num_students_2nd_class : ℕ) (combined_avg_marks : ℕ) : ℕ :=
  let total_students := num_students_1st_class + num_students_2nd_class
  let total_marks := combined_avg_marks * total_students
  let total_marks_1st_class := avg_marks_1st_class * num_students_1st_class
  let total_marks_2nd_class := total_marks - total_marks_1st_class
  total_marks_2nd_class / num_students_2nd_class

theorem average_marks_of_second_class_is_60 :
  average_marks_second_class 40 12 28 54 = 60 :=
by
  unfold average_marks_second_class
  simp
  sorry

end average_marks_of_second_class_is_60_l23_23849


namespace garage_sale_items_l23_23609

/-- 
There are 39 items sold at the garage sale given that all prices are different, and the price 
of a radio is the 15th highest and 25th lowest price among them.
-/
theorem garage_sale_items (h1 : ∀ (x y : ℕ), x ≠ y → (p x ≠ p y))
                         (h2 : ∃ (i : ℕ), (rank i = 15 ∧ rank i = 25)) : 
                         ∃ n : ℕ, n = 39 :=
begin
  sorry
end

end garage_sale_items_l23_23609


namespace cameron_list_count_l23_23987

theorem cameron_list_count : 
  (∃ (n m : ℕ), n = 900 ∧ m = 27000 ∧ (∀ k : ℕ, (30 * k) ≥ n ∧ (30 * k) ≤ m → ∃ count : ℕ, count = 871)) :=
by
  sorry

end cameron_list_count_l23_23987


namespace max_value_expr_equals_four_sqrt_six_minus_six_l23_23014

noncomputable def max_value_expr (y : ℝ) : ℝ :=
  (y^2 + 3 - real.sqrt (y^4 + 9)) / y

theorem max_value_expr_equals_four_sqrt_six_minus_six {y : ℝ} (hy : 0 < y) :
  ∀ x, max_value_expr x ≤ 4 * real.sqrt 6 - 6 :=
sorry

end max_value_expr_equals_four_sqrt_six_minus_six_l23_23014


namespace counting_integers_between_multiples_l23_23991

theorem counting_integers_between_multiples :
  let smallest_perfect_square_multiple := 900 in
  let smallest_perfect_cube_multiple := 27000 in
  let num_integers := (smallest_perfect_cube_multiple / 30) - (smallest_perfect_square_multiple / 30) + 1 in
  smallest_perfect_square_multiple = 30 * 30 ∧ 
  smallest_perfect_cube_multiple = 900 * 30 ∧ 
  num_integers = 871 :=
by
  sorry

end counting_integers_between_multiples_l23_23991


namespace find_m_l23_23708

noncomputable def hyperbola (m : ℝ) : Prop :=
  mx^2 + y^2 = 1

noncomputable def conjugate_axis_length (m : ℝ) : ℝ :=
  2 * real.sqrt (1 / -m)

noncomputable def transverse_axis_length : ℝ :=
  2

theorem find_m (m : ℝ) (h : hyperbola m) 
  (h_ax_len : conjugate_axis_length m = 2 * transverse_axis_length) : 
  m = -1 / 4 := 
by
  sorry

end find_m_l23_23708


namespace stationery_calculation_l23_23655

theorem stationery_calculation :
  let g := 25 in
  let l := 3 * g in
  let b := g + 10 in
  let d := b / 2 in
  g - (l + b + d) = -102.5 :=
by
  sorry

end stationery_calculation_l23_23655


namespace saree_final_price_l23_23880

noncomputable def saree_original_price : ℝ := 5000
noncomputable def first_discount_rate : ℝ := 0.20
noncomputable def second_discount_rate : ℝ := 0.15
noncomputable def third_discount_rate : ℝ := 0.10
noncomputable def fourth_discount_rate : ℝ := 0.05
noncomputable def tax_rate : ℝ := 0.12
noncomputable def luxury_tax_rate : ℝ := 0.05
noncomputable def custom_fee : ℝ := 200
noncomputable def exchange_rate_to_usd : ℝ := 0.013

theorem saree_final_price :
  let price_after_first_discount := saree_original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  let price_after_third_discount := price_after_second_discount * (1 - third_discount_rate)
  let price_after_fourth_discount := price_after_third_discount * (1 - fourth_discount_rate)
  let tax := price_after_fourth_discount * tax_rate
  let luxury_tax := price_after_fourth_discount * luxury_tax_rate
  let total_charges := tax + luxury_tax + custom_fee
  let total_price_rs := price_after_fourth_discount + total_charges
  let final_price_usd := total_price_rs * exchange_rate_to_usd
  abs (final_price_usd - 46.82) < 0.01 :=
by sorry

end saree_final_price_l23_23880


namespace problem_solution_l23_23369

theorem problem_solution :
  2 ^ 2000 - 3 * 2 ^ 1999 + 2 ^ 1998 - 2 ^ 1997 + 2 ^ 1996 = -5 * 2 ^ 1996 :=
by  -- initiate the proof script
  sorry  -- means "proof is omitted"

end problem_solution_l23_23369


namespace solve_alcohol_mixture_l23_23054

-- Definitions based on given conditions
def volume_x := 200 -- milliliters
def percent_alcohol_x := 0.10 -- 10 percent
def percent_alcohol_y := 0.30 -- 30 percent
def desired_percent_alcohol := 0.20 -- 20 percent

-- Main statement: Prove that adding 200 milliliters of solution y to 200 milliliters of solution x results in desired concentration
theorem solve_alcohol_mixture : 
  ∀ (V_y : ℝ), 
  (V_y = 200) →
  (volume_x * percent_alcohol_x + V_y * percent_alcohol_y) / (volume_x + V_y) = desired_percent_alcohol :=
by
  sorry

end solve_alcohol_mixture_l23_23054


namespace maximize_profit_l23_23181

noncomputable section

def price (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then 60
  else if 100 < x ∧ x ≤ 600 then 62 - 0.02 * x
  else 0

def profit (x : ℕ) : ℝ :=
  (price x - 40) * x

theorem maximize_profit :
  ∃ x : ℕ, (1 ≤ x ∧ x ≤ 600) ∧ (∀ y : ℕ, (1 ≤ y ∧ y ≤ 600 → profit y ≤ profit x)) ∧ profit x = 6050 :=
by sorry

end maximize_profit_l23_23181


namespace perfect_square_condition_l23_23028

theorem perfect_square_condition (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : ∃ k : ℕ, a = k^2 :=
by
  sorry

end perfect_square_condition_l23_23028


namespace smallest_norm_v_l23_23416

-- Given definitions and conditions
variable (v : ℝ × ℝ)
def v_add_vector_norm_eq_10 := ∥⟨v.1 + 4, v.2 + 2⟩∥ = 10

-- The proof statement we need to prove
theorem smallest_norm_v (h : v_add_vector_norm_eq_10 v) : 
  ∥v∥ = 10 - 2 * Real.sqrt 5 :=
sorry

end smallest_norm_v_l23_23416


namespace purely_imaginary_a_eq_neg2_l23_23738

noncomputable def complex.imaginary_part_zero (z : ℂ) : Prop :=
  z.re = 0

theorem purely_imaginary_a_eq_neg2 {a : ℝ} (h : complex.imaginary_part_zero (⟨(a : ℂ) + (1 : ℂ) * complex.I, 1 + 2 * complex.I⟩)) :
  a = -2 := sorry

end purely_imaginary_a_eq_neg2_l23_23738


namespace ellipse_tangent_circle_radius_correct_l23_23968

noncomputable def ellipse_tangent_circle_radius : ℝ := 
  let a := 6
  let b := 3
  let c := sqrt (a ^ 2 - b ^ 2)
  let f := (c, 0)
  let r := 6 - 3 * sqrt 3
  r

theorem ellipse_tangent_circle_radius_correct :
  ellipse_tangent_circle_radius = 6 - 3 * sqrt 3 :=
sorry

end ellipse_tangent_circle_radius_correct_l23_23968


namespace rectangle_within_rectangle_l23_23019

theorem rectangle_within_rectangle (a b c d : ℝ) (ha : a < c) (hb : c ≤ d) (hc : d < b) (hd : a * b < c * d) :
  (\( \(b^2 - a^2\right)^2 ≤ \( \(bc - ad\right)^2 + \( \(bd - ac\right)^2\) :=
sorry

end rectangle_within_rectangle_l23_23019


namespace dividend_is_correct_l23_23567

theorem dividend_is_correct :
  ∀ (d q r D : ℕ), d = 17 ∧ q = 9 ∧ r = 8 ∧ D = 161 → (d * q) + r = D :=
by
  intros d q r D h
  obtain ⟨hd, hq, hr, hD⟩ := h
  rw [hd, hq, hr, hD]
  sorry

end dividend_is_correct_l23_23567


namespace greg_sisters_count_l23_23716

theorem greg_sisters_count :
  ∀ (total_bars: ℕ) (days: ℕ) (bars_per_sister: ℕ), 
  total_bars = 20 → 
  days = 7 → 
  bars_per_sister = 5 → 
  let remaining_bars := total_bars - days in
  let bars_after_trade := remaining_bars - 3 in 
  bars_after_trade / bars_per_sister = 2 :=
by
  intros total_bars days bars_per_sister h1 h2 h3
  let remaining_bars := total_bars - days
  let bars_after_trade := remaining_bars - 3
  have h4 : bars_after_trade = 10 := by 
    calc 
      bars_after_trade =
        total_bars - days - 3 : by sorry
  have h5 : bars_per_sister = 5 := by
    sorry
  calc 
    bars_after_trade / bars_per_sister = 
      10 / 5 := by 
        rw [h4, h5]
    _ = 2 := by
        sorry   

end greg_sisters_count_l23_23716


namespace possible_t_sum_l23_23536

noncomputable def is_isosceles_triangle (A B C : (ℝ × ℝ)) : Prop :=
  let dist (P Q : (ℝ × ℝ)) := (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2
  dist A B = dist A C ∨ dist A B = dist B C ∨ dist A C = dist B C

theorem possible_t_sum 
  (t : ℝ)
  (h1 : 0 ≤ t) 
  (h2 : t ≤ 360)
  (isosceles : is_isosceles_triangle (cos 40, sin 40) (cos 60, sin 60) (cos t, sin t)) :
  ∑ t in {20, 80, 50, 230}, t = 380 :=
sorry

end possible_t_sum_l23_23536


namespace rhombus_side_length_l23_23107

/-
  Define the length of the rhombus diagonal and the area of the rhombus.
-/
def diagonal1 : ℝ := 20
def area : ℝ := 480

/-
  The theorem states that given these conditions, the length of each side of the rhombus is 26 m.
-/
theorem rhombus_side_length (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = diagonal1) (h2 : A = area):
  2 * 26 * 26 * 2 = A * 2 * 2 + (d1 / 2) * (d1 / 2) :=
sorry

end rhombus_side_length_l23_23107


namespace total_alligators_spotted_l23_23833

-- Define variables and conditions
variables (x : ℕ)
variables (g2_1 g2_2 g3_1 g3_2 g3_3 : ℕ)

-- Conditions
def condition1 := 30 -- Samara saw 30 alligators
def condition2 := g2_1 + g2_2 = 54 -- Group 2 saw 54 alligators, with one seeing x more than the other
def condition3 := g3_1 + g3_2 + g3_3 = 36 -- Group 3 saw total 36 alligators in the afternoon with average 12 each
def condition4 := (∀ i ∈ [g3_1, g3_2, g3_3], i = 15) -- Each friend in Group 3 ended up seeing 15 alligators after the increase

-- Statement
theorem total_alligators_spotted : 
  (condition2) ∧ 
  (condition3) ∧ 
  (condition4) →
  30 + (23 + 31) + (15 * 3) = 129 :=
begin
  sorry
end

end total_alligators_spotted_l23_23833


namespace max_digit_d_l23_23252

theorem max_digit_d (d f : ℕ) (h₁ : d ≤ 9) (h₂ : f ≤ 9) (h₃ : (18 + d + f) % 3 = 0) (h₄ : (12 - (d + f)) % 11 = 0) : d = 1 :=
sorry

end max_digit_d_l23_23252


namespace cupcakes_sold_l23_23237

theorem cupcakes_sold (reduced_price_cupcake reduced_price_cookie : ℝ) (num_cookies total_revenue : ℝ) :
  reduced_price_cupcake = 1.50 → reduced_price_cookie = 1.00 → num_cookies = 8 → total_revenue = 32 →
  ∃ c : ℝ, c = 16 :=
by
  intros h1 h2 h3 h4
  -- define the variables and outcomes based on the conditions
  let revenue_cookies := reduced_price_cookie * num_cookies
  let revenue_cupcakes := total_revenue - revenue_cookies
  let num_cupcakes := revenue_cupcakes / reduced_price_cupcake
  use num_cupcakes
  have h5: revenue_cookies = 8, from calc
    revenue_cookies = reduced_price_cookie * num_cookies : by rw [h2, h3]; norm_num
  have h6: revenue_cupcakes = 24, from calc
    revenue_cupcakes = total_revenue - revenue_cookies : by rw [h4, h5]; norm_num
  show num_cupcakes = 16, from calc
    num_cupcakes = revenue_cupcakes / reduced_price_cupcake : by rw [h1]; norm_num
  injection
  end

end cupcakes_sold_l23_23237


namespace union_sets_l23_23408

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4, 6}

theorem union_sets : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_sets_l23_23408


namespace circle_equation_tangent_lines_l23_23282

-- Define the points A, B, and line l
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (1, -2)
def l (x y : ℝ) : Prop := 2*x + y - 4 = 0

-- Define point N
def N : ℝ × ℝ := (5, 3)

-- Equation of circle function
def circle (M : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  (x - M.1)^2 + (y - M.2)^2 = r^2

-- Equation of line function
def line (p : ℝ × ℝ) (k : ℝ) (x y: ℝ) : Prop :=
  y = k * (x - p.1) + p.2

-- Part 1: Prove the equation of the circle
theorem circle_equation : ∃ M : ℝ × ℝ, (M = (3, -2)) ∧ (circle M 2 x y) :=
sorry

-- Part 2: Prove the equations of the tangent lines
theorem tangent_lines : (x = 5) ∨ (21 * x - 20 * y - 45 = 0) :=
sorry

end circle_equation_tangent_lines_l23_23282


namespace prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l23_23754

variables (n k : ℕ) (p : ℝ)
variables (h_pos : 0 < p) (h_lt_one : p < 1)

-- Probability that exactly k gnomes fall
theorem prob_exactly_k_gnomes_fall (h_k_le_n : k ≤ n) :
  prob_speed (exactly_k_gnomes_fall n k p) = p * (1 - p)^(n - k) := sorry

-- Expected number of fallen gnomes
theorem expected_fallen_gnomes : 
  expected_falls n p = n + 1 - 1/p + (1 - p)^(n + 1)/p := sorry

end prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l23_23754


namespace demand_decrease_for_revenue_preservation_l23_23200

variable (P Q : ℝ)

theorem demand_decrease_for_revenue_preservation:
  (new_price := 1.30 * P)
  (new_demand := Q / 1.30)
  (revenue_preserved := new_price * new_demand = P * Q):
  (demand_decrease := 1 - (1 / 1.30)) = 0.2308 :=
by
  sorry

end demand_decrease_for_revenue_preservation_l23_23200


namespace range_of_function_l23_23804

open Real

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∃ y, y = sin x - 2 * cos x + 32 / (125 * sin x * (1 - cos x)) ∧ y ≥ 2 / 5 :=
sorry

end range_of_function_l23_23804


namespace quadrilateral_area_l23_23595

-- Define the problem conditions
def KL : ℝ := 40
def KM : ℝ := 24
def midpoint (a b : ℝ) := (a + b) / 2

def KN : ℝ := midpoint 0 KL  -- N is midpoint of KL
def KO : ℝ := midpoint 0 KM  -- O is midpoint of KM

-- The proof statement that the area of quadrilateral KNMO is 480 square units
theorem quadrilateral_area (KL : ℝ) (KM : ℝ) (KN : ℝ) (KO : ℝ) (A_KNMO : ℝ) :
  KL = 40 → KM = 24 → KN = 20 → KO = 12 → A_KNMO = 480 := 
by
  intros
  -- The calculations of area would be done here
  sorry

end quadrilateral_area_l23_23595


namespace units_digit_of_factorial_sum_l23_23273

theorem units_digit_of_factorial_sum :
  (1! + 2! + 3! + 4! + (↓∑ k in Icc 5 2023, k!)) % 10 = 3 := by
  sorry

end units_digit_of_factorial_sum_l23_23273


namespace percentage_entree_cost_l23_23127

-- Conditions
def total_spent : ℝ := 50.0
def num_appetizers : ℝ := 2
def cost_per_appetizer : ℝ := 5.0
def total_appetizer_cost : ℝ := num_appetizers * cost_per_appetizer
def total_entree_cost : ℝ := total_spent - total_appetizer_cost

-- Proof Problem
theorem percentage_entree_cost :
  (total_entree_cost / total_spent) * 100 = 80 :=
sorry

end percentage_entree_cost_l23_23127


namespace least_integer_to_multiple_of_3_l23_23910

theorem least_integer_to_multiple_of_3 : ∃ n : ℕ, n > 0 ∧ (527 + n) % 3 = 0 ∧ ∀ m : ℕ, m > 0 → (527 + m) % 3 = 0 → m ≥ n :=
sorry

end least_integer_to_multiple_of_3_l23_23910


namespace fraction_of_clerical_staff_reduced_l23_23581

theorem fraction_of_clerical_staff_reduced 
  (total_employees : ℕ)
  (clerical_fraction : ℚ)
  (remaining_fraction : ℚ)
  (clerical_employees_before_reduction : ℚ := clerical_fraction * total_employees)
  (clerical_employees_after_reduction : ℚ := clerical_employees_before_reduction - clerical_employees_before_reduction * 1/3)
  (total_employees_after_reduction : ℚ := total_employees - clerical_employees_before_reduction * 1/3) :
  clerical_employees_after_reduction / total_employees_after_reduction = remaining_fraction :=
by
  -- Given details
  have h1 : total_employees = 3600 := rfl
  have h2 : clerical_fraction = 1/6 := rfl
  have h3 : remaining_fraction = 0.1176470588235294 := rfl
  -- Correct answer
  have answer : 1/3 = 1/3 := rfl
  sorry

end fraction_of_clerical_staff_reduced_l23_23581


namespace greatest_power_of_three_in_factorial_twenty_five_l23_23737

theorem greatest_power_of_three_in_factorial_twenty_five :
  ∃ n : ℕ, (∀ m : ℕ, (m = 3^n → m ∣ factorial 25) ∧ n = 10) :=
by {
  sorry
}

end greatest_power_of_three_in_factorial_twenty_five_l23_23737


namespace solve_problem_l23_23749

open Real
open EuclideanGeometry
open Finset
open TopologicalSpace

noncomputable def problem (A B C D M N : Point) : Prop :=
  Quadrilateral.inscribed A B C D ∧
  dist A M = dist M B ∧
  dist D N = dist N C ∧
  Angle A M D = 58 ∧
  Angle D N C = 58

theorem solve_problem (A B C D M N : Point) (h : problem A B C D M N) : Angle M N C = 58 := sorry

end solve_problem_l23_23749


namespace range_of_a_for_perpendicular_tangents_l23_23345

def f (a x : ℝ) : ℝ := a * x + Real.sin x + Real.cos x

theorem range_of_a_for_perpendicular_tangents :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ let m := Real.cos x₁ - Real.sin x₁ in
                              let n := Real.cos x₂ - Real.sin x₂ in
                              (a + m) * (a + n) = -1) ↔ -1 ≤ a ∧ a ≤ 1 := sorry

end range_of_a_for_perpendicular_tangents_l23_23345


namespace integers_between_neg_sqrt2_and_sqrt2_l23_23920

theorem integers_between_neg_sqrt2_and_sqrt2 : 
  (-sqrt 2 : ℝ) < -1 ∧ -1 < (sqrt 2 : ℝ) ∧
  (-sqrt 2 : ℝ) < 0 ∧ 0 < (sqrt 2 : ℝ) ∧
  (-sqrt 2 : ℝ) < 1 ∧ 1 < (sqrt 2 : ℝ) :=
by
  sorry

end integers_between_neg_sqrt2_and_sqrt2_l23_23920


namespace find_right_triangle_area_l23_23774

-- Given conditions and statements
variables {A B C D E G : Type} [has_inner A B C] [triangle ABC]

-- Define right triangle with given conditions
def right_triangle (T : triangle ABC) : Prop :=
  angle T BAC = 90°

def AD_median_and_altitude (T : triangle ABC) (AD : A ↔ D) : Prop :=
  is_median T AD ∧ is_altitude T AD

def medians_given (AD BE : A ↔ D) : Prop :=
  length AD = 18 ∧ length BE = 24

def find_area_of_triangle {A B C : Type} (T : triangle ABC) 
  (ht : right_triangle T)
  (h_ad : AD_median_and_altitude T AD)
  (h_med : medians_given AD BE) : Real :=
  area T = 432

-- The main statement of the problem
theorem find_right_triangle_area
  {A B C D E G : Type}
  [has_inner A B C]
  [triangle ABC]
  (ht : right_triangle T)
  (h_ad : AD_median_and_altitude T AD)
  (h_med : medians_given AD BE) :
  area T = 432 :=
sorry

end find_right_triangle_area_l23_23774


namespace perfect_square_condition_l23_23029

theorem perfect_square_condition (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : ∃ k : ℕ, a = k^2 :=
by
  sorry

end perfect_square_condition_l23_23029


namespace parallel_lines_distance_l23_23085

noncomputable def l1 : (ℝ × ℝ) → ℝ := λ (x y : ℝ), 3*x + 4*y + 6
noncomputable def l2 (a : ℝ) : (ℝ × ℝ) → ℝ := λ (x y : ℝ), (a + 1)*x + 2*a*y + 1
noncomputable def distance_between_parallel_lines (c1 c2 a b : ℝ) (l1 l2 : ℝ → ℝ → ℝ) :=
  |c1 - c2| / (Real.sqrt ((a ^ 2) + (b ^ 2)))

theorem parallel_lines_distance (a : ℝ) 
  (h_parallel: 3 * 2*a - 4 * (a + 1) = 0) :
  let a_val := 2 in
  distance_between_parallel_lines 6 1 3 4 l1 (l2 a_val) = 1 := 
by
  sorry

end parallel_lines_distance_l23_23085


namespace value_to_add_l23_23097

theorem value_to_add (a b c d n : ℕ) (h1 : a = 24) (h2 : b = 32) (h3 : c = 36) (h4 : d = 54) (h5 : n = 861) : 
  ∃ k : ℕ, (nat.lcm a (nat.lcm b (nat.lcm c d)) - n = k) ∧ k = 3 :=
by
  intros
  have A : a = 24 := h1
  have B : b = 32 := h2
  have C : c = 36 := h3
  have D : d = 54 := h4
  have N : n = 861 := h5
  existsi 3
  split
  . rw [A, B, C, D, N]
    simp
  . simp
  sorry

end value_to_add_l23_23097


namespace triangle_cosine_l23_23825

theorem triangle_cosine (a b c r: ℝ) (h1: a = 15) (h2: b + c = 27) (h3: r = 4) :
  (cos_angle_opposite := (b^2 + c^2 - a^2) / (2 * b * c)) →
  cos_angle_opposite = 5 / 13 := sorry

end triangle_cosine_l23_23825


namespace exists_v_min_norm_l23_23419

def smallest_value_norm (v : ℝ × ℝ) : Prop :=
  ⟪∥v + ⟨4, 2⟩∥ = 10 ∧ ∥v∥ = 10 - 2 * Real.sqrt 5⟫

theorem exists_v_min_norm : ∃ v : ℝ × ℝ, smallest_value_norm v :=
  sorry

end exists_v_min_norm_l23_23419
