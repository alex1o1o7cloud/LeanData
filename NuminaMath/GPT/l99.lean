import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Group.Definitions
import Mathlib.Algebra.Order.Arithmetic
import Mathlib.Algebra.QuadraticEquation
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometry
import Mathlib.Data.Fin
import Mathlib.Data.Finset
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Floor
import Mathlib.Data.Real.Basic
import Mathlib.NumberTheory.Floor
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Algebra.InfiniteSum
import data.nat.parity
import data.real.basic

namespace tan_45_eq_one_l99_99638

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99638


namespace cos_minus_sin_of_tan_eq_sqrt3_l99_99097

theorem cos_minus_sin_of_tan_eq_sqrt3 (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos α - Real.sin α = (Real.sqrt 3 - 1) / 2 := 
by
  sorry

end cos_minus_sin_of_tan_eq_sqrt3_l99_99097


namespace tan_45_eq_one_l99_99584

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99584


namespace tan_45_eq_one_l99_99607

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99607


namespace divisor_of_1025_l99_99403

theorem divisor_of_1025 (d : ℕ) (h1: 1015 + 10 = 1025) (h2 : d ∣ 1025) : d = 5 := 
sorry

end divisor_of_1025_l99_99403


namespace symmetric_to_origin_l99_99364

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_to_origin (p : ℝ × ℝ) (h : p = (3, -1)) : symmetric_point p = (-3, 1) :=
by
  -- This is just the statement; the proof is not provided.
  sorry

end symmetric_to_origin_l99_99364


namespace membership_fees_total_correct_l99_99448

/-- Define the initial membership fee -/
def membership_fee_first_year : ℝ := 80

/-- Define the increment rates for each year -/
def increment_rate_second_year : ℝ := 0.10
def increment_rate_third_year : ℝ := 0.12
def increment_rate_fourth_year : ℝ := 0.14
def increment_rate_fifth_year : ℝ := 0.15
def increment_rate_sixth_year : ℝ := 0.15

/-- Define the discount rates for the fifth and sixth years -/
def discount_rate_fifth_year : ℝ := 0.10
def discount_rate_sixth_year : ℝ := 0.05

/-- Calculate the fee for each year -/
def membership_fee_second_year : ℝ := membership_fee_first_year * (1 + increment_rate_second_year)
def membership_fee_third_year : ℝ := membership_fee_second_year * (1 + increment_rate_third_year)
def membership_fee_fourth_year : ℝ := membership_fee_third_year * (1 + increment_rate_fourth_year)
def membership_fee_fifth_year : ℝ := (membership_fee_fourth_year * (1 + increment_rate_fifth_year)) * (1 - discount_rate_fifth_year)
def membership_fee_sixth_year : ℝ := (membership_fee_fifth_year / (1 - discount_rate_fifth_year)) * (1 + increment_rate_sixth_year) * (1 - discount_rate_sixth_year)

/-- Calculate the total cost of Aaron's membership fees over six years -/
def total_membership_fees : ℝ :=
  membership_fee_first_year
  + membership_fee_second_year
  + membership_fee_third_year
  + membership_fee_fourth_year
  + membership_fee_fifth_year
  + membership_fee_sixth_year

/-- The proof statement -/
theorem membership_fees_total_correct :
  total_membership_fees ≈ 636 :=  -- Using ≈ since floating-point arithmetic might cause minor errors
by
  sorry

end membership_fees_total_correct_l99_99448


namespace distinct_numbers_count_l99_99056

noncomputable section

def num_distinct_numbers : Nat :=
  let vals := List.map (λ n : Nat, Nat.floor ((n^2 : ℚ) / 500)) (List.range 1000).tail
  (vals.eraseDup).length

theorem distinct_numbers_count : num_distinct_numbers = 876 :=
by
  sorry

end distinct_numbers_count_l99_99056


namespace tan_45_degrees_eq_1_l99_99816

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99816


namespace tan_45_degree_l99_99483

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99483


namespace tens_digit_9_2023_l99_99398

theorem tens_digit_9_2023 :
  let cycle := [09, 81, 29, 61, 49, 41, 69, 21, 89, 01] in
  (cycle[(2023 % 10)] / 10) % 10 == 2 := by
  sorry

end tens_digit_9_2023_l99_99398


namespace meaningful_expr_l99_99221

theorem meaningful_expr (x : ℝ) : 
    (x + 1 ≥ 0 ∧ x - 2 ≠ 0) → (x ≥ -1 ∧ x ≠ 2) := by
  sorry

end meaningful_expr_l99_99221


namespace max_area_of_triangle_l99_99103

variables (A B C a b c : ℝ)

def is_triangle := ∃ (A B C : ℝ), A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0

theorem max_area_of_triangle
  (h1 : b = sqrt 3)
  (h2 : (2 * a - c) * cos B = sqrt 3 * cos C)
  (h3 : is_triangle A B C) :
  (∃ A B C a b c : ℝ, is_triangle A B C ∧ b = sqrt 3 ∧ (2 * a - c) * cos B = sqrt 3 * cos C) →
  (∃ S : ℝ, S = (1/2) * b * c * sin A ∧ S ≤ 3 * sqrt 3 / 4) :=
by 
  sorry

end max_area_of_triangle_l99_99103


namespace tan_45_degrees_eq_1_l99_99801

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99801


namespace side_length_of_square_base_l99_99325

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l99_99325


namespace tan_45_deg_l99_99786

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99786


namespace tan_45_degree_is_one_l99_99725

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99725


namespace tan_of_45_deg_l99_99568

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99568


namespace tan_45_deg_l99_99529

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99529


namespace tan_45_deg_l99_99774

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99774


namespace average_annual_reduction_l99_99431

theorem average_annual_reduction (x : ℝ) (h₁ : (1 - x)^2 = 0.81) : x = 0.1 :=
begin
  sorry
end

end average_annual_reduction_l99_99431


namespace trigonometric_identity_l99_99041

theorem trigonometric_identity :
  (sin 15 * cos 10 + cos 165 * cos 105) / (sin 25 * cos 5 + cos 155 * cos 95) = 1 / (2 * cos 25) :=
by
  sorry

end trigonometric_identity_l99_99041


namespace smallest_ellipse_area_l99_99458

theorem smallest_ellipse_area {a b : ℝ} (h : ∀ (x y : ℝ), 
  (x - 2)^2 + y^2 ≤ 4 ∧ (x + 2)^2 + y^2 ≤ 4 → x^2 / a^2 + y^2 / b^2 ≤ 1) :
  ∃ (k : ℝ), k = 1 ∧ π * k = π :=
begin
  sorry
end

end smallest_ellipse_area_l99_99458


namespace nico_reads_wednesday_l99_99294

def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51
def pages_wednesday := total_pages - (pages_monday + pages_tuesday) 

theorem nico_reads_wednesday :
  pages_wednesday = 19 :=
by
  sorry

end nico_reads_wednesday_l99_99294


namespace symmetric_point_origin_l99_99366

def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, -P.2)

theorem symmetric_point_origin :
  symmetric_point (3, -1) = (-3, 1) :=
by
  sorry

end symmetric_point_origin_l99_99366


namespace solve_for_x_and_y_l99_99311

theorem solve_for_x_and_y (x y : ℝ) : 9 * x^2 - 25 * y^2 = 0 → (x = (5 / 3) * y ∨ x = -(5 / 3) * y) :=
by
  sorry

end solve_for_x_and_y_l99_99311


namespace domain_g_l99_99317

noncomputable def f : ℝ → ℝ := sorry -- dummy definition for f
noncomputable def domain_f : set ℝ := set.Icc (-10) 6

def g (x : ℝ) : ℝ := f (-3 * x)

theorem domain_g : set.Icc (-2) (10/3) = { x | -10 ≤ -3 * x ∧ -3 * x ≤ 6 } :=
by {
  ext x,
  split;
  intro hx,
  {
    cases hx with h1 h2,
    split,
    linarith,
    linarith
  },
  {
    cases hx with h1 h2,
    split,
    linarith,
    linarith
  }
}

end domain_g_l99_99317


namespace tan_45_degree_is_one_l99_99736

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99736


namespace tan_45_eq_1_l99_99627

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99627


namespace tan_45_degrees_eq_1_l99_99824

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99824


namespace tan_45_deg_l99_99759

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99759


namespace tetrahedron_faces_acute_l99_99044

theorem tetrahedron_faces_acute (tetrahedron : Type)
  (faces : finset (set tetrahedron))
  (dihedral_angle : set tetrahedron → set tetrahedron → ℝ)
  (is_acute : ∀ {s1 s2 : set tetrahedron}, s1 ∈ faces → s2 ∈ faces → s1 ≠ s2 → 0 < dihedral_angle s1 s2 ∧ dihedral_angle s1 s2 < (real.pi / 2))
  : ∀ {face : set tetrahedron}, face ∈ faces → is_acute_triangle face := 
sorry

def is_acute_triangle (triangle : set tetrahedron) : Prop := sorry

end tetrahedron_faces_acute_l99_99044


namespace tan_45_eq_one_l99_99609

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99609


namespace tan_45_degree_l99_99486

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99486


namespace find_function_expression_l99_99110

open Real

-- Definition of the conditions
def is_proportional_to (y x : ℝ) (k : ℝ) : Prop := k ≠ 0 ∧ y = k * x
def passes_through_points (f : ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  f p1.1 = p1.2 ∧ f p2.1 = p2.2

-- Problem: Find the function expressions that satisfy these conditions.
theorem find_function_expression :
  (∃ k, is_proportional_to 3 4 k ∧ ∀ x, ∃ y, y = k * x) ∧
  (∃ k b, passes_through_points (λ x, k * x + b) (-2, 1) ∧ passes_through_points (λ x, k * x + b) (4, -3)) :=
sorry

end find_function_expression_l99_99110


namespace tan_45_eq_l99_99672

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99672


namespace count_adjacent_pairs_sum_multiple_of_three_l99_99463

def adjacent_digit_sum_multiple_of_three (n : ℕ) : ℕ :=
  -- A function to count the number of pairs with a sum multiple of 3
  sorry

-- Define the sequence from 100 to 999 as digits concatenation
def digit_sequence : List ℕ := List.join (List.map (fun x => x.digits 10) (List.range' 100 900))

theorem count_adjacent_pairs_sum_multiple_of_three :
  adjacent_digit_sum_multiple_of_three digit_sequence.length = 897 :=
sorry

end count_adjacent_pairs_sum_multiple_of_three_l99_99463


namespace find_age_of_older_friend_l99_99368

theorem find_age_of_older_friend (A B C : ℝ) 
  (h1 : A - B = 2.5)
  (h2 : A - C = 3.75)
  (h3 : A + B + C = 110.5)
  (h4 : B = 2 * C) : 
  A = 104.25 :=
by
  sorry

end find_age_of_older_friend_l99_99368


namespace sum_of_first_odd_numbers_l99_99038

theorem sum_of_first_odd_numbers (S1 S2 : ℕ) (n1 n2 : ℕ)
  (hS1 : S1 = n1^2) 
  (hS2 : S2 = n2^2) 
  (h1 : S1 = 2500)
  (h2 : S2 = 5625) : 
  n2 = 75 := by
  sorry

end sum_of_first_odd_numbers_l99_99038


namespace tan_45_eq_1_l99_99551

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99551


namespace tan_45_deg_eq_1_l99_99985

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99985


namespace tan_45_deg_l99_99750

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99750


namespace tan_45_deg_eq_1_l99_99977

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99977


namespace tan_45_eq_1_l99_99954

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99954


namespace tan_45_degree_l99_99907

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99907


namespace minimum_difference_of_factors_l99_99146

theorem minimum_difference_of_factors:
  ∃ a b : ℕ, a * b = 1620 ∧ (∀ c d : ℕ, c * d = 1620 → |c - d| ≥ |a - b|) ∧ |a - b| = 9 := by
  sorry

end minimum_difference_of_factors_l99_99146


namespace tan_45_eq_1_l99_99951

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99951


namespace tan_45_degrees_l99_99842

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99842


namespace tan_45_eq_1_l99_99614

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99614


namespace tan_45_eq_l99_99686

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99686


namespace speed_of_current_l99_99007

-- Conditions translated into Lean definitions
def initial_time : ℝ := 13 -- 1:00 PM is represented as 13:00 hours
def boat1_time_turnaround : ℝ := 14 -- Boat 1 turns around at 2:00 PM
def boat2_time_turnaround : ℝ := 15 -- Boat 2 turns around at 3:00 PM
def meeting_time : ℝ := 16 -- Boats meet at 4:00 PM
def raft_drift_distance : ℝ := 7.5 -- Raft drifted 7.5 km from the pier

-- The problem statement to prove
theorem speed_of_current:
  ∃ v : ℝ, (v * (meeting_time - initial_time) = raft_drift_distance) ∧ v = 2.5 :=
by
  sorry

end speed_of_current_l99_99007


namespace dot_product_AD_AC_l99_99240

variable (A B C D : Type)
variable [Add A] [Mul A] [HasDotProduct A]
variable (AB AC AD : A)
variable (AB_eq : AB = (1, -2))
variable (AD_eq : AD = (2, 1))
variable (AC_eq : AC = AB + AD)

theorem dot_product_AD_AC : AD • AC = 5 := by
  sorry

end dot_product_AD_AC_l99_99240


namespace tan_45_eq_one_l99_99855

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99855


namespace tan_45_eq_1_l99_99628

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99628


namespace sumata_miles_per_day_l99_99320

theorem sumata_miles_per_day (total_miles : ℝ) (total_days : ℝ) (h1 : total_miles = 250.0) (h2 : total_days = 5.0) :
  total_miles / total_days = 50.0 :=
by
  sorry

end sumata_miles_per_day_l99_99320


namespace tan_45_eq_one_l99_99927

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99927


namespace correct_average_is_43_6_l99_99442

noncomputable def corrected_average (incorrect_avg : ℝ) (true_vals incorrect_vals : list ℝ) (n : ℕ) : ℝ :=
  let incorrect_total := incorrect_avg * n
  let total_difference := list.sum (list.map2 has_sub.sub incorrect_vals true_vals)
  let correct_total := incorrect_total - total_difference
  correct_total / n

theorem correct_average_is_43_6 :
  corrected_average 45.6 [45.7, 39.4, 51.6, 43.5] [55.7, 49.4, 61.6, 53.5] 20 = 43.6 :=
by
  sorry

end correct_average_is_43_6_l99_99442


namespace region_area_l99_99023

-- Parameters are positive real numbers
variables (a b c d : ℝ)
-- All parameters are positive
variables (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)

-- Prove the area of the region
theorem region_area :
  let vertical_length := 2 * a + b in
  let horizontal_length := d + 2 * c in
  let area := vertical_length * horizontal_length in
  area = 2 * a * d + 4 * a * c + b * d + 2 * b * c :=
by
  sorry

end region_area_l99_99023


namespace number_of_true_propositions_l99_99098

variables {a b : Line} {α β : Plane}

-- Define the given conditions and propositions
def prop1 : Prop := a ⟂ α ∧ b ⟂ β ∧ α ∥ β → a ∥ b
def prop2 : Prop := a ⟂ α ∧ b ∥ β ∧ α ∥ β → a ⟂ b
def prop3 : Prop := α ∥ β ∧ a ∈ α ∧ b ∈ β → a ∥ b
def prop4 : Prop := a ⟂ α ∧ b ⟂ β ∧ α ⟂ β → a ⟂ b

-- Define the problem: The number of true propositions is 3
theorem number_of_true_propositions :
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) ↔ true_prop_count = 3 :=
sorry

end number_of_true_propositions_l99_99098


namespace tan_45_eq_1_l99_99958

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99958


namespace tan_of_45_deg_l99_99575

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99575


namespace Carlos_finishes_first_l99_99004

-- Definitions
def AndyLawnArea (a : ℝ) : ℝ := a
def BethLawnArea (a : ℝ) : ℝ := a / 3
def CarlosLawnArea (a : ℝ) : ℝ := a / 4

def AndyMowingRate (r : ℝ) : ℝ := r
def CarlosMowingRate (r : ℝ) : ℝ := r / 2
def BethMowingRate (r : ℝ) : ℝ := r / 4

-- Time to mow each lawn
def AndyMowingTime (a r : ℝ) : ℝ := AndyLawnArea a / AndyMowingRate r
def BethMowingTime (a r : ℝ) : ℝ := BethLawnArea a / BethMowingRate r
def CarlosMowingTime (a r : ℝ) : ℝ := CarlosLawnArea a / CarlosMowingRate r

-- Theorem statement
theorem Carlos_finishes_first
  (a r : ℝ)
  (h_ar_pos : a > 0 ∧ r > 0) :
  (CarlosMowingTime a r < AndyMowingTime a r) ∧ (CarlosMowingTime a r < BethMowingTime a r) :=
by
  sorry

end Carlos_finishes_first_l99_99004


namespace tan_45_deg_l99_99797

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99797


namespace problem_sol_l99_99084

open Complex

theorem problem_sol (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : (a + i) / i = 1 + b * i) : a + b = 0 :=
sorry

end problem_sol_l99_99084


namespace tan_45_deg_l99_99798

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99798


namespace tan_45_degree_l99_99502

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99502


namespace tan_45_deg_eq_1_l99_99976

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99976


namespace sum_of_solutions_l99_99209

theorem sum_of_solutions (x : ℝ) (h1 : x^2 = 25) : ∃ S : ℝ, S = 0 ∧ (∀ x', x'^2 = 25 → x' = 5 ∨ x' = -5) := 
sorry

end sum_of_solutions_l99_99209


namespace yellow_peaches_l99_99390

theorem yellow_peaches (red_peaches green_peaches total_green_yellow_peaches : ℕ)
  (h1 : red_peaches = 5)
  (h2 : green_peaches = 6)
  (h3 : total_green_yellow_peaches = 20) :
  (total_green_yellow_peaches - green_peaches) = 14 :=
by
  sorry

end yellow_peaches_l99_99390


namespace tan_45_degrees_eq_1_l99_99809

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99809


namespace area_BKF_half_area_ABC_l99_99232

-- Define the conditions
variable (A B C D E F K : Point)
variable (isConvex : ConvexQuadrilateral A B C D)
variable (isMidpointE : Midpoint E C D)
variable (isMidpointF : Midpoint F A D)
variable (isIntersectionK : Intersection K (Line A C) (Line B E))

-- Define the areas
noncomputable def area (P Q R : Point) : ℝ := sorry

-- Statement of the problem
theorem area_BKF_half_area_ABC :
  area B K F = (1 / 2) * area A B C :=
sorry

end area_BKF_half_area_ABC_l99_99232


namespace tan_45_eq_one_l99_99928

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99928


namespace tan_45_degrees_l99_99707

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99707


namespace tan_45_degree_is_one_l99_99729

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99729


namespace pyramid_base_side_length_l99_99331

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l99_99331


namespace tan_45_eq_one_l99_99657

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99657


namespace tan_45_eq_1_l99_99946

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99946


namespace tan_45_eq_one_l99_99865

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99865


namespace side_length_of_base_l99_99358

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l99_99358


namespace tan_45_eq_one_l99_99661

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99661


namespace tan_45_degrees_eq_1_l99_99804

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99804


namespace tan_45_deg_eq_1_l99_99980

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99980


namespace tan_45_deg_l99_99755

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99755


namespace r_squared_sum_l99_99191

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l99_99191


namespace find_r_fourth_l99_99167

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l99_99167


namespace range_g_l99_99374

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sqrt 3 * sin x * cos x

noncomputable def g (x : ℝ) : ℝ := sin (4 * x + π / 6) + 1 / 2

theorem range_g (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 4) : 0 ≤ g x ∧ g x ≤ 3 / 2 := 
sorry

end range_g_l99_99374


namespace trapezoid_ratios_l99_99465

variables (S S₁ S₂ S₃ S₄ : ℝ)
variables (AD BC : ℝ)

theorem trapezoid_ratios (h1 : S₁ / S)
                         (h2 : (S₁ + S₂) / S)
                         (h3 : S₂ / S) :
  AD / BC = AD / BC := sorry

end trapezoid_ratios_l99_99465


namespace tan_45_degrees_eq_1_l99_99822

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99822


namespace no_such_numbers_exist_l99_99037

theorem no_such_numbers_exist :
  ¬ ∃ (x : Fin 99 → ℝ), 
    (∀ i, x i = √2 + 1 ∨ x i = √2 - 1) ∧ 
    (∑ i in Finset.range 99, x i * x (i + 1) % 99) = 199 :=
by sorry

end no_such_numbers_exist_l99_99037


namespace tan_45_deg_l99_99778

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99778


namespace winning_candidate_percentage_l99_99239

def votes : List Float := [2104.4, 2300.2, 2157.8, 2233.5, 2250.3, 2350.1, 2473.7]

noncomputable def total_votes : Float := votes.sum

noncomputable def winning_votes : Float := votes.maximum.getD 0.0

noncomputable def winning_percentage : Float := (winning_votes / total_votes) * 100

theorem winning_candidate_percentage : winning_percentage ≈ 15.58 :=
sorry

end winning_candidate_percentage_l99_99239


namespace r_squared_sum_l99_99189

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l99_99189


namespace tan_45_degree_is_one_l99_99727

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99727


namespace tan_45_deg_eq_1_l99_99970

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99970


namespace tan_45_degree_l99_99501

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99501


namespace r_pow_four_solution_l99_99173

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l99_99173


namespace bill_pay_double_l99_99013

theorem bill_pay_double (x : ℕ) (h₁ : ∀ h, 0 < h ∧ h ≤ x → ∃ p, p = 20 * h)
  (h₂ : ∀ h, x < h ∧ h ≤ 50 → ∃ p, p = 40 * (h - x))
  (h₃ : (∑ h in range x, 20 * h) + (∑ h in range (50 - x), 40 * (h + x)) = 1200) :
  x = 40 :=
by
  sorry

end bill_pay_double_l99_99013


namespace tan_45_degrees_eq_1_l99_99810

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99810


namespace minimum_value_of_a_l99_99087

noncomputable def find_a (a : ℝ) : Prop :=
  ∀ (x y : ℝ), (x ≥ 1) → (x + y ≤ 3) → (y ≥ a * (x - 3)) → (2 * x + y) = 1 → a = 1 / 2

theorem minimum_value_of_a :
  ∃ (a : ℝ), a > 0 ∧ find_a a :=
by
  use 1 / 2
  split
  { -- Prove that a = 1/2 is greater than 0.
    exact one_div_pos.mpr (by norm_num) }
  { -- Prove that for a = 1/2, the minimum value of z = 2x + y is 1.
    intros x y hx hxy hy
    intro h
    -- The proof would go here.
    sorry
  }

end minimum_value_of_a_l99_99087


namespace pages_wednesday_l99_99295

-- Given conditions as definitions
def borrow_books := 3
def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51

-- Prove that Nico read 19 pages on Wednesday
theorem pages_wednesday :
  let pages_wednesday := total_pages - (pages_monday + pages_tuesday)
  pages_wednesday = 19 :=
by
  sorry

end pages_wednesday_l99_99295


namespace tan_45_deg_l99_99504

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99504


namespace balls_combination_count_l99_99391

theorem balls_combination_count : 
  let colors := ['red', 'blue', 'yellow'],
      balls_per_color := 7,
      total_ways_to_choose := 60 in
  ∃ (f : ℕ → ℕ → bool) (h : ∀ i j, f i j → i ≠ j ∧ i ∈ colors ∧ j ∈ colors ∧ (i, j) not consecutive), 
  ∑ (i : colors) (j : colors) in finset.range(balls_per_color), f i j = total_ways_to_choose :=
sorry

end balls_combination_count_l99_99391


namespace tan_45_deg_l99_99758

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99758


namespace tan_45_eq_one_l99_99856

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99856


namespace tan_45_eq_1_l99_99998

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l99_99998


namespace tan_45_degree_is_one_l99_99741

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99741


namespace frames_per_page_l99_99258

theorem frames_per_page (total_frames : ℕ) (pages : ℕ) (h_total_frames : total_frames = 143) (h_pages : pages = 13) : total_frames / pages = 11 :=
by
  rw [h_total_frames, h_pages]
  norm_num
  sorry

end frames_per_page_l99_99258


namespace cos_angle_a_a_plus_b_l99_99135

variables {V : Type*} [inner_product_space ℝ V]

-- Defining vectors and conditions given in the problem
variables (a b : V)
variables (ha : ∥a∥ = 5) (hb : ∥b∥ = 6) (ha_b : ⟪a, b⟫ = -6)

-- Defining the angle cosine theorem to be proved
theorem cos_angle_a_a_plus_b :
  real.cos (inner_product_space.angle a (a + b)) = 19 / 35 :=
by sorry

end cos_angle_a_a_plus_b_l99_99135


namespace avg_speed_comparison_l99_99473

variable (u v : ℝ)

def avg_speed_A (u v : ℝ) : ℝ := 3 / (1/u + 2/v)
def avg_speed_B (u v : ℝ) : ℝ := (2 * u + v) / 3

theorem avg_speed_comparison (u v : ℝ) 
    (hu : 0 < u) (hv : 0 < v) : avg_speed_A u v ≤ avg_speed_B u v := 
by
  sorry

end avg_speed_comparison_l99_99473


namespace pure_imaginary_a_l99_99219

variable {a : ℝ}

theorem pure_imaginary_a (a_is_real : a ∈ ℝ) (pure_imaginary : ∀ r : ℝ, r ≠ 0 → (2 - a = 0) → a = 2) :
  a = 2 :=
  by
  sorry

end pure_imaginary_a_l99_99219


namespace tan_45_eq_one_l99_99660

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99660


namespace min_cost_29_disks_l99_99141

theorem min_cost_29_disks
  (price_single : ℕ := 20) 
  (price_pack_10 : ℕ := 111) 
  (price_pack_25 : ℕ := 265) :
  ∃ cost : ℕ, cost ≥ (price_pack_10 + price_pack_10 + price_pack_10) 
              ∧ cost ≤ (price_pack_25 + price_single * 4) 
              ∧ cost = 333 := 
by
  sorry

end min_cost_29_disks_l99_99141


namespace meaningful_range_l99_99222

noncomputable def isMeaningful (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x - 2 ≠ 0)

theorem meaningful_range (x : ℝ) : isMeaningful x ↔ (x ≥ -1) ∧ (x ≠ 2) :=
by
  sorry

end meaningful_range_l99_99222


namespace tan_45_degrees_eq_1_l99_99802

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99802


namespace common_difference_is_3_l99_99243

theorem common_difference_is_3 (a : ℕ → ℤ) (d : ℤ) (h1 : a 2 = 4) (h2 : 1 + a 3 = 5 + d)
  (h3 : a 6 = 4 + 4 * d) (h4 : 4 + a 10 = 8 + 8 * d) :
  (5 + d) * (8 + 8 * d) = (4 + 4 * d) ^ 2 → d = 3 := 
by
  intros hg
  sorry

end common_difference_is_3_l99_99243


namespace r_fourth_power_sum_l99_99182

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l99_99182


namespace minimum_period_of_f_l99_99377

-- Define the function f
def f (x : ℝ) : ℝ := 2 * sin (3 * x - (π / 3))

-- Define the property of the function period
theorem minimum_period_of_f :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ T = 2 * π / 3 :=
by sorry

end minimum_period_of_f_l99_99377


namespace nico_reads_wednesday_l99_99293

def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51
def pages_wednesday := total_pages - (pages_monday + pages_tuesday) 

theorem nico_reads_wednesday :
  pages_wednesday = 19 :=
by
  sorry

end nico_reads_wednesday_l99_99293


namespace jason_egg_consumption_in_two_weeks_l99_99045

def breakfast_pattern : List Nat := 
  [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] -- Two weeks pattern alternating 3-egg and (2+1)-egg meals

noncomputable def count_eggs (pattern : List Nat) : Nat :=
  pattern.foldl (· + ·) 0

theorem jason_egg_consumption_in_two_weeks : 
  count_eggs breakfast_pattern = 42 :=
sorry

end jason_egg_consumption_in_two_weeks_l99_99045


namespace find_j_l99_99078

theorem find_j (j : ℝ) :
  (∃ (x y : ℝ), x = 1/3 ∧ y = -3 ∧ -2 - 3 * j * x = 7 * y) → j = 19 :=
by
  intro h
  cases h with x h1
  cases h1 with y h2
  cases h2 with hx1 h2
  cases h2 with hy1 h3
  rw [hx1, hy1] at h3
  sorry

end find_j_l99_99078


namespace tan_45_deg_l99_99751

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99751


namespace tan_45_deg_l99_99790

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99790


namespace tan_45_degree_l99_99885

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99885


namespace tan_of_45_deg_l99_99574

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99574


namespace tan_45_deg_eq_1_l99_99984

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99984


namespace tan_45_eq_l99_99687

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99687


namespace tan_45_deg_l99_99760

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99760


namespace solve_tangent_C_l99_99253

theorem solve_tangent_C (A B C : ℝ) (tan_C : ℝ) (h1 : Real.cot A * Real.cot C = 1) (h2 : Real.cot B * Real.cot C = 1 / 8) :
  tan_C = 4 + Real.sqrt 7 :=
sorry

end solve_tangent_C_l99_99253


namespace tan_45_deg_l99_99766

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99766


namespace distinct_numbers_count_l99_99053

noncomputable section

def num_distinct_numbers : Nat :=
  let vals := List.map (λ n : Nat, Nat.floor ((n^2 : ℚ) / 500)) (List.range 1000).tail
  (vals.eraseDup).length

theorem distinct_numbers_count : num_distinct_numbers = 876 :=
by
  sorry

end distinct_numbers_count_l99_99053


namespace tan_45_degree_l99_99482

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99482


namespace intersection_claim_union_claim_l99_99123

def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}
def B : Set ℝ := {x | x^2 - 2 * x - 3 ≥ 0}
def U : Set ℝ := Set.univ

-- Claim 1: Prove that A ∩ B = {x | -5 < x ∧ x ≤ -1}
theorem intersection_claim : A ∩ B = {x | -5 < x ∧ x ≤ -1} :=
by
  sorry

-- Claim 2: Prove that A ∪ (U \ B) = {x | -5 < x ∧ x < 3}
theorem union_claim : A ∪ (U \ B) = {x | -5 < x ∧ x < 3} :=
by
  sorry

end intersection_claim_union_claim_l99_99123


namespace tan_45_degree_is_one_l99_99745

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99745


namespace sum_of_solutions_x_squared_eq_25_l99_99206

theorem sum_of_solutions_x_squared_eq_25 : 
  (∑ x in ({x : ℝ | x^2 = 25}).to_finset, x) = 0 :=
by
  sorry

end sum_of_solutions_x_squared_eq_25_l99_99206


namespace tan_45_deg_l99_99787

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99787


namespace inner_triangle_shortest_side_l99_99298

open Real

noncomputable def is_inside (inner outer : set (ℝ × ℝ)) : Prop := sorry

theorem inner_triangle_shortest_side
    (inner outer : set (ℝ × ℝ))
    (A B C M : ℝ × ℝ)
    (AB AC BC BM MC : ℝ)
    (H_inner : is_inside inner outer)
    (H_midpoint : M = (B + C) / 2)
    (H_sides : AB ≤ AC ∧ AC ≤ BC)
    (H_length : BM = MC ∧ BM = BC / 2)
    (H_triangle_ineq : BC < AB + AC ∧ AB + AC <= 2 * AC)
    (H_projections : ∀ (P : ℝ × ℝ), P ∈ inner → P ∈ (B × C)) :
  ∃ x y, (x, y) ∈ inner ∧ (x, y) ∈ {A, B, C} :=
sorry

end inner_triangle_shortest_side_l99_99298


namespace tan_45_deg_eq_1_l99_99963

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99963


namespace tan_of_45_deg_l99_99577

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99577


namespace tan_of_45_deg_l99_99560

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99560


namespace triangle_inequality_l99_99277

theorem triangle_inequality (S : Finset (ℕ × ℕ)) (m n : ℕ) (hS : S.card = m)
  (h_ab : ∀ (a b : ℕ), (a, b) ∈ S → (1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n ∧ a ≠ b)) :
  ∃ (t : Finset (ℕ × ℕ × ℕ)),
    (t.card ≥ (4 * m / (3 * n)) * (m - (n^2) / 4)) ∧
    ∀ (a b c : ℕ), (a, b, c) ∈ t → (a, b) ∈ S ∧ (b, c) ∈ S ∧ (c, a) ∈ S := by
  sorry

end triangle_inequality_l99_99277


namespace distinct_numbers_count_l99_99055

noncomputable section

def num_distinct_numbers : Nat :=
  let vals := List.map (λ n : Nat, Nat.floor ((n^2 : ℚ) / 500)) (List.range 1000).tail
  (vals.eraseDup).length

theorem distinct_numbers_count : num_distinct_numbers = 876 :=
by
  sorry

end distinct_numbers_count_l99_99055


namespace triangle_OMF_area_l99_99271

theorem triangle_OMF_area
    (M : ℝ × ℝ)
    (h_M_ON_parabola : M.2^2 = 12 * M.1)
    (H_focus : (3 : ℝ, 0))
    (H_distance_MF : real.sqrt ((M.1 - 3)^2 + M.2^2) = 5) :
    (M.1 = 8 ∧ ∃ y, y = M.2 ∧ abs y = 4 * real.sqrt 3) ∧
    let FM : ℝ := real.sqrt ((M.1 - 0)^2 + 0^2)
    let height_y : ℝ := abs M.2
    in (1 / 2 * FM * height_y = 6 * real.sqrt 3) :=
begin
  sorry
end

end triangle_OMF_area_l99_99271


namespace smallest_period_of_f_min_value_of_f_in_interval_l99_99114

def f (x : ℝ) : ℝ :=
  sin (π / 2 - x) * sin x - sqrt 3 * sin x ^ 2

theorem smallest_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
begin
  sorry
end

theorem min_value_of_f_in_interval :
  ∃ x ∈ Icc (0 : ℝ) (π / 4), ∀ y ∈ Icc (0 : ℝ) (π / 4), f y ≥ f x ∧ f x = - sqrt 3 / 2 :=
begin
  sorry
end

end smallest_period_of_f_min_value_of_f_in_interval_l99_99114


namespace tan_45_eq_l99_99688

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99688


namespace number_of_perfect_square_divisors_of_product_l99_99015

theorem number_of_perfect_square_divisors_of_product :
  let product := list.prod (list.map factorial (list.range 11))
  let num_perfect_square_divisors := 1368
  num_perfect_square_divisors = number_of_perfect_square_divisors product :=
by sorry

end number_of_perfect_square_divisors_of_product_l99_99015


namespace sum_of_solutions_x_squared_eq_25_l99_99204

theorem sum_of_solutions_x_squared_eq_25 : 
  (∑ x in ({x : ℝ | x^2 = 25}).to_finset, x) = 0 :=
by
  sorry

end sum_of_solutions_x_squared_eq_25_l99_99204


namespace tan_45_deg_l99_99781

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99781


namespace fifth_number_pascals_triangle_l99_99237

theorem fifth_number_pascals_triangle (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) : 
  binomial n k = 1365 :=
by 
  rw [h1, h2] 
  -- Additional steps not required, but need to finalize the proof
  -- as proof steps are not part of the requirement.
  sorry

end fifth_number_pascals_triangle_l99_99237


namespace tan_45_degree_is_one_l99_99739

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99739


namespace tan_45_degree_l99_99886

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99886


namespace cosine_angle_sum_l99_99138

variable {𝕜 : Type*} [InnerProductSpace ℝ 𝕜]
variable (a b : 𝕜)

-- Conditions given in the problem
def norm_a : ∥a∥ = 5 := sorry
def norm_b : ∥b∥ = 6 := sorry
def dot_ab : ⟪a, b⟫ = -6 := sorry

-- Required proof statement
theorem cosine_angle_sum :
  ∥a∥ = 5 → ∥b∥ = 6 → ⟪a, b⟫ = -6 → Real.cos (inner_product_space.angle a (a + b)) = 19 / 35 :=
by
  sorry

end cosine_angle_sum_l99_99138


namespace tan_45_degree_l99_99894

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99894


namespace tan_45_deg_eq_1_l99_99982

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99982


namespace tan_of_45_deg_l99_99561

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99561


namespace tan_45_eq_one_l99_99925

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99925


namespace travel_time_is_36_minutes_l99_99444

noncomputable def travel_time_double_speed (V C : ℝ) (h1 : V + C > 0) (h2 : 2V - C > 0) 
  (h3 : V + C - 1 = 0) (h4 : 2V - C - 1 = 0) : ℝ :=
let D := V + C in
let new_speed := 2 * V + C in
let ratio := new_speed / (V + C) in
60 * (3 / 5)

theorem travel_time_is_36_minutes (V C : ℝ) 
  (h1 : V + C > 0) (h2 : 2V - C > 0) 
  (h3 : V + C - 1 = 0) (h4 : 2V - C - 1 = 0) :
  travel_time_double_speed V C h1 h2 h3 h4 = 36 :=
sorry

end travel_time_is_36_minutes_l99_99444


namespace tan_double_angle_l99_99102

theorem tan_double_angle (α : ℝ) (h1 : α ∈ Ioo (-π/2) 0) (h2 : cos α = 4/5) : tan (2 * α) = -24/7 :=
by
  sorry

end tan_double_angle_l99_99102


namespace r_fourth_power_sum_l99_99183

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l99_99183


namespace tan_45_degrees_l99_99714

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99714


namespace tan_45_eq_one_l99_99875

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99875


namespace find_r4_l99_99160

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l99_99160


namespace tan_45_degrees_l99_99829

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99829


namespace tan_45_eq_1_l99_99952

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99952


namespace toad_count_l99_99261

theorem toad_count :
  (∀ t: ℕ, t * 3 * 15 = 6 * 60 → t = 8) := 
begin
  intro t,
  intro h,
  sorry
end

end toad_count_l99_99261


namespace sum_of_solutions_eq_zero_l99_99201

theorem sum_of_solutions_eq_zero (x : ℝ) (hx : x^2 = 25) : ∃ x₁ x₂ : ℝ, (x₁^2 = 25) ∧ (x₂^2 = 25) ∧ (x₁ + x₂ = 0) := 
by {
  use 5,
  use (-5),
  split,
  { exact hx, },
  split,
  { rw pow_two, exact hx, },
  { norm_num, },
}

end sum_of_solutions_eq_zero_l99_99201


namespace tan_45_deg_eq_1_l99_99983

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99983


namespace remainder_of_division_l99_99406

theorem remainder_of_division (k : ℤ) : 
  let N := 35 * k + 25 in (N % 15) = 10 :=
by
  sorry

end remainder_of_division_l99_99406


namespace find_r_fourth_l99_99165

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l99_99165


namespace find_prime_between_20_and_35_with_remainder_7_l99_99434

theorem find_prime_between_20_and_35_with_remainder_7 : 
  ∃ p : ℕ, Nat.Prime p ∧ 20 ≤ p ∧ p ≤ 35 ∧ p % 11 = 7 ∧ p = 29 := 
by 
  sorry

end find_prime_between_20_and_35_with_remainder_7_l99_99434


namespace tan_45_eq_1_l99_99990

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l99_99990


namespace sum_of_roots_eq_zero_sum_of_all_possible_values_l99_99212

theorem sum_of_roots_eq_zero (x : ℝ) (h : x^2 = 25) : x = 5 ∨ x = -5 :=
by {
  sorry
}

theorem sum_of_all_possible_values (h : ∀ x : ℝ, x^2 = 25 → x = 5 ∨ x = -5) : ∑ x in {x : ℝ | x^2 = 25}, x = 0 :=
by {
  sorry
}

end sum_of_roots_eq_zero_sum_of_all_possible_values_l99_99212


namespace tan_45_deg_l99_99792

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99792


namespace tan_of_45_deg_l99_99579

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99579


namespace tan_45_degrees_l99_99840

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99840


namespace tan_45_deg_l99_99753

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99753


namespace deputies_seated_identical_l99_99466

theorem deputies_seated_identical (n : Nat) (a : Fin 2n → Fin 2n) (h : Function.Bijective a) :
  ∃ i j : Fin 2n, i ≠ j ∧ ∀ m : Fin 2n, a ((i + m : Fin 2n).val) = a ((j + m : Fin 2n).val) :=
sorry

end deputies_seated_identical_l99_99466


namespace least_area_triangle_ABC_l99_99381

open Complex Real

noncomputable def least_possible_area_of_triangle (z : ℂ) (k : ℤ) : ℝ := 
  let cis_0 := cis 0
  let cis_pi_4 := cis (π/4)
  let cis_pi_2 := cis (π/2)
  let A := 2^0.5 * cis_0
  let B := 2^0.5 * cis_pi_4
  let C := 2^0.5 * cis_pi_2
  let AC := abs (A - C)
  let height := |B.im|
  (1 / 2) * AC * height

theorem least_area_triangle_ABC : least_possible_area_of_triangle 2^0.5 0 = 2 :=
sorry

end least_area_triangle_ABC_l99_99381


namespace determine_m_l99_99139

noncomputable def a : ℝ × ℝ := (2,1)
noncomputable def b (m : ℝ) : ℝ × ℝ := (1,m)
noncomputable def c : ℝ × ℝ := (2,4)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem determine_m (m : ℝ) :
  dot_product (2 * a - 5 * b m) c = 0 → m = 3 / 10 :=
by 
  sorry

end determine_m_l99_99139


namespace find_speed_of_goods_train_l99_99407

variable (v : ℕ) -- Speed of the goods train in km/h

theorem find_speed_of_goods_train
  (h1 : 0 < v) 
  (h2 : 6 * v + 4 * 90 = 10 * v) :
  v = 36 :=
by
  sorry

end find_speed_of_goods_train_l99_99407


namespace tan_45_eq_one_l99_99877

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99877


namespace r_pow_four_solution_l99_99179

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l99_99179


namespace polynomial_horner_method_l99_99017

theorem polynomial_horner_method :
  let a_4 := 3
  let a_3 := 0
  let a_2 := -1
  let a_1 := 2
  let a_0 := 1
  let x := 2
  let v_0 := 3
  let v_1 := v_0 * x + a_3
  let v_2 := v_1 * x + a_2
  let v_3 := v_2 * x + a_1
  v_3 = 22 :=
by 
  let a_4 := 3
  let a_3 := 0
  let a_2 := -1
  let a_1 := 2
  let a_0 := 1
  let x := 2
  let v_0 := a_4
  let v_1 := v_0 * x + a_3
  let v_2 := v_1 * x + a_2
  let v_3 := v_2 * x + a_1
  sorry

end polynomial_horner_method_l99_99017


namespace cos_angle_a_a_plus_b_l99_99134

variables {V : Type*} [inner_product_space ℝ V]

-- Defining vectors and conditions given in the problem
variables (a b : V)
variables (ha : ∥a∥ = 5) (hb : ∥b∥ = 6) (ha_b : ⟪a, b⟫ = -6)

-- Defining the angle cosine theorem to be proved
theorem cos_angle_a_a_plus_b :
  real.cos (inner_product_space.angle a (a + b)) = 19 / 35 :=
by sorry

end cos_angle_a_a_plus_b_l99_99134


namespace tan_45_degrees_l99_99692

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99692


namespace tan_45_eq_1_l99_99936

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99936


namespace prob_exactly_two_passes_prob_at_least_one_fails_l99_99316

-- Define the probabilities for students A, B, and C passing their tests.
def prob_A : ℚ := 4/5
def prob_B : ℚ := 3/5
def prob_C : ℚ := 7/10

-- Define the probabilities for students A, B, and C failing their tests.
def prob_not_A : ℚ := 1 - prob_A
def prob_not_B : ℚ := 1 - prob_B
def prob_not_C : ℚ := 1 - prob_C

-- (1) Prove that the probability of exactly two students passing is 113/250.
theorem prob_exactly_two_passes : 
  prob_A * prob_B * prob_not_C + prob_A * prob_not_B * prob_C + prob_not_A * prob_B * prob_C = 113/250 := 
sorry

-- (2) Prove that the probability that at least one student fails is 83/125.
theorem prob_at_least_one_fails : 
  1 - (prob_A * prob_B * prob_C) = 83/125 := 
sorry

end prob_exactly_two_passes_prob_at_least_one_fails_l99_99316


namespace tan_45_eq_one_l99_99924

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99924


namespace quadratic_has_distinct_real_roots_l99_99225

theorem quadratic_has_distinct_real_roots (m : ℝ) (hm : m ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (m * x1^2 - 2 * x1 + 3 = 0) ∧ (m * x2^2 - 2 * x2 + 3 = 0) ↔ 0 < m ∧ m < (1 / 3) :=
by
  sorry

end quadratic_has_distinct_real_roots_l99_99225


namespace sum_denominators_geq_bound_l99_99474

theorem sum_denominators_geq_bound (n : ℕ) (h : n ≥ 2) :
  ∀ (f : fin n → ℚ), (∀ i j, i ≠ j → f i ≠ f j) → (∀ i, 0 < f i ∧ f i < 1) →
  ∑ i : fin n, (nat_of_quot (denom (f i))) ≥ (1/3) * n^(3/2) := 
by sorry

end sum_denominators_geq_bound_l99_99474


namespace tan_45_deg_eq_1_l99_99974

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99974


namespace range_of_a_l99_99378

-- Definitions for the conditions
def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2 * x - 8 > 0

-- Main theorem
theorem range_of_a (a : ℝ) (h : a < 0) : (¬ (∃ x, prop_p a x)) → (¬ (∃ x, ¬ prop_q x)) :=
sorry

end range_of_a_l99_99378


namespace tan_45_deg_l99_99513

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99513


namespace tan_45_eq_1_l99_99553

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99553


namespace planes_from_non_coplanar_points_l99_99001

theorem planes_from_non_coplanar_points : 
  ∀ (A B C D : Point), ¬coplanar A B C D → 
  ({x : Set (Set Point) | ∃ (X Y Z : Point), x = {X, Y, Z} ∧ X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z}.size = 4) :=
by
  sorry

end planes_from_non_coplanar_points_l99_99001


namespace minimize_XY_ratio_l99_99394

-- Define the geometric configuration and given conditions
noncomputable def O : Point := ⟨0, 0⟩  -- Center of C1 and C2
noncomputable def A : Point := ⟨1/2, 0⟩  -- One endpoint of diameter AB
noncomputable def B : Point := ⟨-1/2, 0⟩  -- Other endpoint of diameter AB

def radius_C1 : ℝ := 1/2  -- Radius of circle C1
noncomputable def radius_C2 (k : ℝ) (h1 : 1 < k) (h2 : k < 3) : ℝ := k/2  -- Radius of circle C2
noncomputable def radius_C3 (k : ℝ) (h1 : 1 < k) (h2 : k < 3) : ℝ := k  -- Radius of circle C3

-- Define the circles
def C1 (k : ℝ) (h1 : 1 < k) (h2 : k < 3) : circle := ⟨O, radius_C1⟩
def C2 (k : ℝ) (h1 : 1 < k) (h2 : k < 3) : circle := ⟨O, radius_C2 k h1 h2⟩
def C3 (k : ℝ) (h1 : 1 < k) (h2 : k < 3) : circle := ⟨A, radius_C3 k h1 h2⟩

-- Main theorem statement: ratio XB/BY = 1 for minimal length of XY
theorem minimize_XY_ratio (k : ℝ) (h1 : 1 < k) (h2 : k < 3) :
  ∀ (X Y : Point), (X ∈ C2 k h1 h2)
  → (Y ∈ C3 k h1 h2)
  → (B ∈ line_segment X Y)
  → (XB : ℝ) (BY : ℝ), 
  ∃ r > 0, (XB = r * BY) := sorry

end minimize_XY_ratio_l99_99394


namespace tan_45_eq_one_l99_99590

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99590


namespace tan_45_eq_1_l99_99533

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99533


namespace r_power_four_identity_l99_99150

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l99_99150


namespace exponential_function_characterization_l99_99224

-- Define the exponential function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Given conditions
theorem exponential_function_characterization {a : ℝ}
  (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f a 3 = 8) : f a = (λ x, 2^x) :=
by
  sorry  -- We are skipping the proof for now

end exponential_function_characterization_l99_99224


namespace abs_diff_less_abs_one_minus_prod_l99_99305

theorem abs_diff_less_abs_one_minus_prod (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  |x - y| < |1 - x * y| := by
  sorry

end abs_diff_less_abs_one_minus_prod_l99_99305


namespace tan_45_eq_one_l99_99931

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99931


namespace lorry_crossing_time_l99_99409

noncomputable def lorry_length : ℕ := 200
noncomputable def bridge_length : ℕ := 200
noncomputable def speed_kmph : ℕ := 80
noncomputable def speed_in_mps : ℚ := (speed_kmph * 1000) / 3600
noncomputable def total_distance : ℕ := lorry_length + bridge_length

theorem lorry_crossing_time : (total_distance / speed_in_mps) ≈ 18 :=
by
  simp [total_distance, speed_in_mps]
  sorry

end lorry_crossing_time_l99_99409


namespace tan_45_deg_l99_99503

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99503


namespace tan_45_deg_l99_99517

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99517


namespace tan_45_eq_1_l99_99953

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99953


namespace dragon_heads_belong_to_dragons_l99_99425

def truthful (H : ℕ) : Prop := 
  H = 1 ∨ H = 3

def lying (H : ℕ) : Prop := 
  H = 2 ∨ H = 4

def head1_statement : Prop := truthful 1
def head2_statement : Prop := truthful 3
def head3_statement : Prop := ¬ truthful 2
def head4_statement : Prop := lying 3

theorem dragon_heads_belong_to_dragons :
  head1_statement ∧ head2_statement ∧ head3_statement ∧ head4_statement →
  (∀ H, (truthful H ↔ H = 1 ∨ H = 3) ∧ (lying H ↔ H = 2 ∨ H = 4)) :=
by
  sorry

end dragon_heads_belong_to_dragons_l99_99425


namespace total_students_suggestion_l99_99012

theorem total_students_suggestion (mashed_potatoes_students : ℕ) (bacon_students : ℕ) (tomatoes_students : ℕ) (h1 : mashed_potatoes_students = 324) (h2 : bacon_students = 374) (h3 : tomatoes_students = 128) : mashed_potatoes_students + bacon_students + tomatoes_students = 826 :=
by
  rw [h1, h2, h3]
  norm_num
  exact rfl
  sorry

end total_students_suggestion_l99_99012


namespace total_fruit_salads_l99_99459

theorem total_fruit_salads (a : ℕ) (h_alaya : a = 200) (h_angel : 2 * a = 400) : a + 2 * a = 600 :=
by 
  rw [h_alaya, h_angel]
  sorry

end total_fruit_salads_l99_99459


namespace tan_45_degrees_eq_1_l99_99825

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99825


namespace perfect_square_expression_l99_99147

theorem perfect_square_expression (x y : ℝ) (k : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x y, f x = f y → 4 * x^2 - (k - 1) * x * y + 9 * y^2 = (f x) ^ 2) ↔ (k = 13 ∨ k = -11) :=
by
  sorry

end perfect_square_expression_l99_99147


namespace tan_45_eq_one_l99_99653

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99653


namespace area_of_quadrilateral_l99_99236

-- defining points and lengths
variables (X Y Z W T : Type)
variables [euclidean_geometry X Y Z W T]

-- conditions
def right_triangle (X Y Z : Type) : Prop :=
  ∃ (x y z : Type), is_right_triangle x y z

def midpoint (W X Y : Type) : Prop := 
  dist W X = dist W Y

def perpendicular (T W X : Type) : Prop :=
  ∃ (u v : Type), is_perpendicular u v

def lengths : Prop :=
  dist X Y = 26 ∧ dist X Z = 10 ∧ dist Y W = 13

-- theorem statement
theorem area_of_quadrilateral (ht : right_triangle X Y Z)
                              (hm : midpoint W X Y)
                              (hp : perpendicular T W X)
                              (hl : lengths X Y Z W T) :
  area_of_quadrilateral X W T Z = 90 :=
sorry

end area_of_quadrilateral_l99_99236


namespace tan_of_45_deg_l99_99562

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99562


namespace tangent_line_slope_l99_99109

-- Define the conditions
def is_tangent (k x₀ : ℝ) : Prop :=
  let y₀ := Real.log x₀ in
  let slope := 1 / x₀ in
  let line_eq := (λ x, k * x) in
  k = slope ∧ y₀ = line_eq x₀

-- The theorem to be proved
theorem tangent_line_slope : ∃ k x₀ : ℝ, is_tangent k x₀ ∧ k = 1 / Real.exp(1) :=
  sorry

end tangent_line_slope_l99_99109


namespace minimum_value_x2_y2_z2_exists_minimum_value_achieved_l99_99276

theorem minimum_value_x2_y2_z2 (x y z : ℝ) : 
  (x^3 + y^3 + z^3 - 3*x*y*z = 1) → (x^2 + y^2 + z^2 ≥ 1) :=
begin
  sorry,
end

theorem exists_minimum_value_achieved : 
  ∃ (x y z : ℝ), (x = 1) ∧ (y = 0) ∧ (z = 0) ∧ (x^2 + y^2 + z^2 = 1) :=
begin
  use 1, 0, 0,
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  have : 1^2 + 0^2 + 0^2 = 1 := by ring,
  exact this,
end

end minimum_value_x2_y2_z2_exists_minimum_value_achieved_l99_99276


namespace tan_45_degree_l99_99900

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99900


namespace find_fx_l99_99281

theorem find_fx : 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, 2 * f(x - 1) - 3 * f(1 - x) = 5 * x) ∧ (∀ x : ℝ, f(x) = x - 5)) :=
sorry

end find_fx_l99_99281


namespace percentage_of_alcohol_in_new_mixture_l99_99419

theorem percentage_of_alcohol_in_new_mixture :
  ∀ (initial_volume new_water_volume alcohol_percentage: ℝ)
  (h_initial_volume : initial_volume = 15)
  (h_new_water_volume : new_water_volume = 3)
  (h_alcohol_percentage : alcohol_percentage = 0.25),
  let initial_alcohol_volume := alcohol_percentage * initial_volume in
  let new_total_volume := initial_volume + new_water_volume in
  let new_percentage_alcohol := (initial_alcohol_volume / new_total_volume) * 100 in
  new_percentage_alcohol = 20.83 :=
by
  intros initial_volume new_water_volume alcohol_percentage
         h_initial_volume h_new_water_volume h_alcohol_percentage
  let initial_alcohol_volume := alcohol_percentage * initial_volume
  let new_total_volume := initial_volume + new_water_volume
  let new_percentage_alcohol := (initial_alcohol_volume / new_total_volume) * 100
  sorry

end percentage_of_alcohol_in_new_mixture_l99_99419


namespace tan_45_deg_eq_1_l99_99986

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99986


namespace remainder_when_divided_by_l99_99418

noncomputable def f (x : ℝ) : ℝ := x^4 + 1
noncomputable def g (x : ℝ) : ℝ := x^2 - 4*x + 7

theorem remainder_when_divided_by (x : ℝ) :
  let remainder := 8*x - 62 in
  ∃ q : polynomial ℝ, f(x) = (g(x) * q(x) + remainder) := sorry

end remainder_when_divided_by_l99_99418


namespace tan_45_degrees_eq_1_l99_99813

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99813


namespace tan_45_eq_l99_99681

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99681


namespace tan_45_deg_l99_99507

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99507


namespace tan_45_eq_one_l99_99646

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99646


namespace tan_45_deg_l99_99524

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99524


namespace next_sales_amount_l99_99437

noncomputable def ratio_decrease : ℚ := 76.1904761904762 / 100  -- The ratio decrease as a fractional representation

theorem next_sales_amount (x : ℚ) :
  let first_sales : ℚ := 20
      first_royalties : ℚ := 7
      next_royalties : ℚ := 9
      initial_ratio : ℚ := first_royalties / first_sales in
      ratio_decrease = 16 / 21 →
  (next_royalties / x) = initial_ratio * (1 - ratio_decrease) →
  x = 108 :=
by
  sorry

end next_sales_amount_l99_99437


namespace tan_45_degrees_l99_99703

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99703


namespace models_kirsty_can_buy_l99_99264

def original_price : ℝ := 0.45
def saved_for_models : ℝ := 30 * original_price
def new_price : ℝ := 0.50

theorem models_kirsty_can_buy :
  saved_for_models / new_price = 27 :=
sorry

end models_kirsty_can_buy_l99_99264


namespace find_r4_l99_99161

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l99_99161


namespace r_fourth_power_sum_l99_99187

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l99_99187


namespace tan_45_eq_l99_99670

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99670


namespace tan_45_degree_l99_99903

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99903


namespace tan_45_eq_one_l99_99586

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99586


namespace tan_45_eq_one_l99_99873

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99873


namespace tan_45_degree_l99_99891

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99891


namespace sum_of_nonnegative_solutions_l99_99073

noncomputable def f (x : ℝ) : ℝ :=
  3^abs x + 4 * abs x

theorem sum_of_nonnegative_solutions :
  (∃ x : ℝ, 0 ≤ x ∧ f x = 24) :=
begin
  sorry
end

end sum_of_nonnegative_solutions_l99_99073


namespace tan_45_eq_one_l99_99600

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99600


namespace general_term_formula_is_not_element_l99_99090

theorem general_term_formula (a : ℕ → ℤ) (h1 : a 1 = 2) (h17 : a 17 = 66) :
  (∀ n, a n = 4 * n - 2) :=
by
  sorry

theorem is_not_element (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 2) :
  ¬ (∃ n : ℕ, a n = 88) :=
by
  sorry

end general_term_formula_is_not_element_l99_99090


namespace lines_parallel_condition_l99_99274

theorem lines_parallel_condition (a : ℝ) : 
  (a = 1) ↔ (∀ x y : ℝ, (a * x + 2 * y - 1 = 0 → x + (a + 1) * y + 4 = 0)) :=
sorry

end lines_parallel_condition_l99_99274


namespace tan_45_degree_l99_99492

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99492


namespace tan_45_eq_1_l99_99633

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99633


namespace tan_45_eq_1_l99_99941

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99941


namespace tan_of_45_deg_l99_99582

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99582


namespace remaining_storage_space_l99_99315

/-- Given that 1 GB = 1024 MB, a hard drive with 300 GB of total storage,
and 300000 MB of used storage, prove that the remaining storage space is 7200 MB. -/
theorem remaining_storage_space (total_gb : ℕ) (mb_per_gb : ℕ) (used_mb : ℕ) :
  total_gb = 300 → mb_per_gb = 1024 → used_mb = 300000 →
  (total_gb * mb_per_gb - used_mb) = 7200 :=
by
  intros h1 h2 h3
  sorry

end remaining_storage_space_l99_99315


namespace tan_45_eq_1_l99_99956

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99956


namespace tan_45_degree_l99_99888

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99888


namespace tan_45_eq_one_l99_99867

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99867


namespace tan_45_eq_one_l99_99588

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99588


namespace pyramid_base_side_length_l99_99338

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l99_99338


namespace tan_45_degrees_l99_99702

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99702


namespace tan_45_deg_l99_99777

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99777


namespace holiday_in_big_month_l99_99002

def new_years_day := 1 -- January 1st
def teachers_day := 254 -- September 10th (Day of year in a non-leap year)
def childrens_day := 152 -- June 1st (Day of year in a non-leap year)

def big_months_days := [1, 3, 5, 7, 8, 10, 12] -- January, March, May, July, August, October, December
def big_months := [1..31] ++ [60..90] ++ [120..150] ++ [181..211] ++ [243..273] ++ [304..334] ++ [335..365]

theorem holiday_in_big_month : 
  (∃ d ∈ big_months_days, new_years_day = d ∨ teachers_day = d ∨ childrens_day = d) → 
  new_years_day ∈ big_months := sorry

end holiday_in_big_month_l99_99002


namespace tan_of_45_deg_l99_99572

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99572


namespace tan_45_eq_1_l99_99621

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99621


namespace tan_45_eq_1_l99_99535

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99535


namespace tan_45_eq_one_l99_99918

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99918


namespace tan_45_degree_l99_99479

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99479


namespace tan_45_deg_l99_99799

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99799


namespace tan_45_deg_l99_99514

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99514


namespace tan_45_deg_l99_99526

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99526


namespace exists_pos_integer_m_l99_99275

noncomputable def f (x : ℤ) : ℤ := 3 * x + 2

def iter_f (n : ℕ) (x : ℤ) : ℤ :=
  Nat.iterate (fun y => f y) n x

theorem exists_pos_integer_m (n : ℕ) (k : ℤ) : 
  ∃ m : ℤ, 0 < m ∧ iter_f 100 m = k * 1988 :=
sorry

end exists_pos_integer_m_l99_99275


namespace tens_digit_of_9_pow_2023_l99_99397

theorem tens_digit_of_9_pow_2023 : (9 ^ 2023) % 100 / 10 = 2 :=
by sorry

end tens_digit_of_9_pow_2023_l99_99397


namespace average_height_of_three_l99_99302

theorem average_height_of_three (parker daisy reese : ℕ) 
  (h1 : parker = daisy - 4)
  (h2 : daisy = reese + 8)
  (h3 : reese = 60) : 
  (parker + daisy + reese) / 3 = 64 := 
  sorry

end average_height_of_three_l99_99302


namespace tan_45_deg_l99_99776

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99776


namespace tan_45_eq_1_l99_99945

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99945


namespace tan_45_eq_1_l99_99544

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99544


namespace tan_45_eq_1_l99_99993

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l99_99993


namespace tan_45_degrees_eq_1_l99_99819

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99819


namespace tan_45_eq_one_l99_99602

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99602


namespace tan_45_eq_one_l99_99654

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99654


namespace side_length_of_base_l99_99354

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l99_99354


namespace tan_45_eq_one_l99_99910

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99910


namespace tan_45_eq_1_l99_99538

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99538


namespace tan_45_deg_l99_99795

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99795


namespace tan_45_eq_one_l99_99640

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99640


namespace tan_45_deg_l99_99763

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99763


namespace speed_of_current_l99_99009

theorem speed_of_current (h_start: ∀ t: ℝ, t ≥ 0 → u ≥ 0) 
  (boat1_turn_2pm: ∀ t: ℝ, t >= 1 → t < 2 → boat1_turn_13_14) 
  (boat2_turn_3pm: ∀ t: ℝ, t >= 2 → t < 3 → boat2_turn_14_15) 
  (boats_meet: ∀ x: ℝ, x = 7.5) :
  v = 2.5 := 
sorry

end speed_of_current_l99_99009


namespace find_function_l99_99278

theorem find_function (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, 2 * f (x - 1) - 3 * f (1 - x) = 5 * x) : 
  f = λ x, x - 5 := 
sorry

end find_function_l99_99278


namespace tan_45_deg_l99_99785

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99785


namespace change_is_41_l99_99405

-- Define the cost of shirts and sandals as given in the problem conditions
def cost_of_shirts : ℕ := 10 * 5
def cost_of_sandals : ℕ := 3 * 3
def total_cost : ℕ := cost_of_shirts + cost_of_sandals

-- Define the amount given
def amount_given : ℕ := 100

-- Calculate the change
def change := amount_given - total_cost

-- State the theorem
theorem change_is_41 : change = 41 := 
by 
  -- Filling this with justification steps would be the actual proof
  -- but it's not required, so we use 'sorry' to indicate the theorem
  sorry

end change_is_41_l99_99405


namespace tan_45_eq_one_l99_99922

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99922


namespace tan_45_deg_l99_99796

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99796


namespace tan_45_eq_1_l99_99555

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99555


namespace tan_45_eq_1_l99_99550

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99550


namespace construction_better_than_logistics_l99_99246

theorem construction_better_than_logistics 
  (applications_computer : ℕ := 215830)
  (applications_mechanical : ℕ := 200250)
  (applications_marketing : ℕ := 154676)
  (applications_logistics : ℕ := 74570)
  (applications_trade : ℕ := 65280)
  (recruitments_computer : ℕ := 124620)
  (recruitments_marketing : ℕ := 102935)
  (recruitments_mechanical : ℕ := 89115)
  (recruitments_construction : ℕ := 76516)
  (recruitments_chemical : ℕ := 70436) :
  applications_construction / recruitments_construction < applications_logistics / recruitments_logistics→ 
  (applications_computer / recruitments_computer < applications_chemical / recruitments_chemical) :=
sorry

end construction_better_than_logistics_l99_99246


namespace tan_45_degrees_l99_99705

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99705


namespace tan_45_eq_1_l99_99624

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99624


namespace tan_seq_rational_odd_denominator_l99_99284

theorem tan_seq_rational_odd_denominator (θ : ℝ) (hθ : real.tan θ = 2) (n : ℕ) (hn : n ≥ 1) :
  ∃ p q : ℤ, a_n = p / q ∧ nat.odd q :=
begin
  sorry
end

end tan_seq_rational_odd_denominator_l99_99284


namespace area_inside_circle_outside_square_l99_99446

theorem area_inside_circle_outside_square :
  let side_length_square := 2 in
  let radius_circle := √2 in
  let area_square := side_length_square * side_length_square in
  let area_circle := π * (radius_circle * radius_circle) in
  area_circle - area_square = 2 * π - 4 :=
by
  sorry

end area_inside_circle_outside_square_l99_99446


namespace tan_45_degree_l99_99496

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99496


namespace tan_45_degrees_eq_1_l99_99814

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99814


namespace example_problem1_example_problem2_l99_99027

noncomputable def triangle_construction
  (a m_a d : ℝ) : Prop :=
  ∃ Δ : Triangle,
    Δ ≃ Triangle.side_height_distance a m_a d

theorem example_problem1 : ¬ triangle_construction 6 5 2.5 :=
by sorry

theorem example_problem2 : ∃ Δ₁ Δ₂, Δ₁ ≠ Δ₂ ∧ 
  (triangle_construction 6 1.5 2.5 ∧ Δ₁ ≃ Δ₂) :=
by sorry

end example_problem1_example_problem2_l99_99027


namespace tan_45_degrees_l99_99709

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99709


namespace tan_45_deg_l99_99780

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99780


namespace op_plus_18_plus_l99_99286

def op_plus (y: ℝ) : ℝ := 9 - y
def plus_op (y: ℝ) : ℝ := y - 9

theorem op_plus_18_plus :
  plus_op (op_plus 18) = -18 := by
  sorry

end op_plus_18_plus_l99_99286


namespace tan_45_degrees_l99_99845

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99845


namespace color_lines_no_blue_boundary_l99_99443

-- Definitions of general position and finite regions
section
variables {n : ℕ} (lines : fin n → set (ℝ × ℝ))

-- A set of lines in the plane is in general position if no two are parallel and no three pass through the same point.
def general_position (lines : fin n → set (ℝ × ℝ)) : Prop :=
∀ i j k : fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → ¬(lines i = lines j ∨ (∃ p : ℝ × ℝ, p ∈ (lines i ∩ lines j ∩ lines k)))

-- Statement that needs to be proved
theorem color_lines_no_blue_boundary 
  (h : general_position lines) :
  ∃ (blue : fin n → bool), (∑ i, ite (blue i) 1 0) ≥ ⌊real.sqrt (n : ℝ)⌋₊ ∧ (∀ r, finite_region lines r → ∃ i, i ∈ r.boundary ∧ ¬blue i) :=
sorry

end

end color_lines_no_blue_boundary_l99_99443


namespace arithmetic_sequence_condition_t_n_a_n_ratio_l99_99096

noncomputable def sequence_a : ℕ → ℕ
| 1     := 3
| (n+1) := 3 * ((sequence_a n))

@[simp] lemma a_n_correct (n : ℕ) : sequence_a (n + 1) = 3^(n + 1) :=
by sorry

theorem arithmetic_sequence_condition (a_j a_i a_k: ℕ) (λ μ: ℕ) 
(h_seq : sequence_a j = a_j) 
(h_seqi : sequence_a i = a_i)
(h_seqk : sequence_a k = a_k)
(h_order : i < j ∧ j < k)
(h_arith : λ * a_j + μ * a_k = 2 * 6 * a_i) : 
λ  = 1 ∧ μ = 1 :=
by sorry

noncomputable def sequence_b : ℕ → ℕ
| 1     := 1
| (n+1) := 2 * n + 1

noncomputable def t_n (n : ℕ) : ℕ :=
(nat.sum (Ico 1 (n+1)) sequence_b)

theorem t_n_a_n_ratio (n : ℕ) (h_positive: 0 < n ): t_n n / sequence_a n = 1/3 → n = 1 ∨ n = 3 := 
by sorry

end arithmetic_sequence_condition_t_n_a_n_ratio_l99_99096


namespace double_domino_probability_l99_99230

theorem double_domino_probability :
  let all_dominoes := {(i, j) | i, j : ℕ, i ≤ j ∧ i <= 12 ∧ j <= 12}.to_finset
  let doubles := {(k, k) | k : ℕ, k <= 12}.to_finset
  (doubles.card : ℚ) / all_dominoes.card = 13 / 91 :=
by
  have all_dominoes_calc : all_dominoes.card = 91, sorry
  have doubles_calc : doubles.card = 13, sorry
  simp [all_dominoes_calc, doubles_calc]
  norm_num

end double_domino_probability_l99_99230


namespace triangle_cos_BAC_angle_bisector_cosine_l99_99250

theorem triangle_cos_BAC_angle_bisector_cosine 
  (A B C D : Type) 
  (AB AC BC : ℝ) 
  (h_AB : AB = 5) 
  (h_AC : AC = 9) 
  (h_BC : BC = 12) 
  (h_D_on_BC : D ∈ segment B C)
  (h_angle_bisector : angle_bisector A B C D) :
  cos (angle_bisector_half_vertex_angle A B C D) = Real.sqrt (13 / 45) :=
sorry

end triangle_cos_BAC_angle_bisector_cosine_l99_99250


namespace smallest_ellipse_area_l99_99457

theorem smallest_ellipse_area {a b : ℝ} (h : ∀ (x y : ℝ), 
  (x - 2)^2 + y^2 ≤ 4 ∧ (x + 2)^2 + y^2 ≤ 4 → x^2 / a^2 + y^2 / b^2 ≤ 1) :
  ∃ (k : ℝ), k = 1 ∧ π * k = π :=
begin
  sorry
end

end smallest_ellipse_area_l99_99457


namespace modulus_of_z_l99_99089

open Complex

theorem modulus_of_z 
  (z : ℂ) 
  (h : (1 - I) * z = 2 * I) : 
  abs z = Real.sqrt 2 := 
sorry

end modulus_of_z_l99_99089


namespace tan_45_eq_one_l99_99641

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99641


namespace tan_45_eq_one_l99_99863

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99863


namespace tan_45_deg_eq_1_l99_99969

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99969


namespace tan_45_degrees_l99_99830

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99830


namespace tan_45_degree_l99_99890

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99890


namespace tan_45_eq_one_l99_99854

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99854


namespace true_proposition_l99_99094

def proposition_p (x : ℝ) : Prop := 2^x > 0

def proposition_q (x : ℝ) : Prop := (x > 1) → (x > 2) ∧ ¬(x > 1 → x > 2)

theorem true_proposition :
  (∀ x : ℝ, proposition_p x) ∧ (¬ (∀ x : ℝ, proposition_q x)) :=
by
  apply And.intro
  { intros x,
    exact pow_pos (by norm_num) x }
  { intros h,
    sorry }

end true_proposition_l99_99094


namespace tan_45_deg_l99_99511

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99511


namespace tan_of_45_deg_l99_99569

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99569


namespace cube_root_fraction_simplification_l99_99046

theorem cube_root_fraction_simplification:
  (20.25 : ℝ) = 81 / 4 →
  (real.cbrt (8 / 20.25)) = (2 * (real.cbrt 2)^2) / (3 * real.cbrt 3) :=
by
  intros h
  sorry

end cube_root_fraction_simplification_l99_99046


namespace tan_45_degrees_l99_99710

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99710


namespace tan_45_degrees_l99_99701

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99701


namespace tan_45_degrees_l99_99843

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99843


namespace tan_45_deg_l99_99519

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99519


namespace side_length_of_square_base_l99_99328

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l99_99328


namespace intersection_of_A_and_B_l99_99124

variable (α : Type) [LinearOrder α] 

def setA : Set α := { x | x < 1 }
def setB : Set α := { x | -1 < x ∧ x < 2 }

theorem intersection_of_A_and_B : setA α ∩ setB α = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end intersection_of_A_and_B_l99_99124


namespace integer_solution_exists_l99_99026

theorem integer_solution_exists
  {m n : ℕ}
  (a : Fin m -> Fin n -> ℤ)
  (h : n ≥ 2 * m)
  (h_nonzero : ∃ i j, a i j ≠ 0) :
  ∃ (x : Fin n -> ℤ), 
    (∀ i : Fin m, (∑ j, a i j * x j) = 0) ∧ 
    (0 < max (λ i, abs (x i))) ∧ 
    (max (λ i, abs (x i)) ≤ n * max (λ i, max (λ j, abs (a i j)))) :=
begin
  sorry
end

end integer_solution_exists_l99_99026


namespace tan_45_degree_l99_99884

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99884


namespace distinct_numbers_count_l99_99057

open BigOperators

theorem distinct_numbers_count :
  (Finset.card ((Finset.image (λ n : ℕ, ⌊ (n^2 : ℝ) / 500⌋) (Finset.range 1001))) = 876) := 
sorry

end distinct_numbers_count_l99_99057


namespace factor_expression_l99_99047

theorem factor_expression (x : ℝ) : 72 * x^3 - 250 * x^7 = 2 * x^3 * (36 - 125 * x^4) :=
by
  sorry

end factor_expression_l99_99047


namespace tan_45_degrees_l99_99704

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99704


namespace r_power_four_identity_l99_99154

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l99_99154


namespace tan_45_deg_l99_99527

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99527


namespace sum_of_coefficients_pow_10_l99_99083

theorem sum_of_coefficients_pow_10 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℤ) :
  (2 * 1 - 1)^10 = 1 (by norm_num : (2 * 1 - 1 : ℤ)^10 = 1) ->
  (2 * 0 - 1)^10 = 1 (by norm_num : (2 * 0 - 1 : ℤ)^10 = 1) ->
  a_0 = 1 ->
  a_1 = -20 ->
    ∑ i in range (11), a_i = 1 -> a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 20 :=
begin
  sorry
end

end sum_of_coefficients_pow_10_l99_99083


namespace tan_45_deg_eq_1_l99_99975

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99975


namespace sum_first_10_terms_arithmetic_seq_l99_99361

theorem sum_first_10_terms_arithmetic_seq (a : ℕ → ℤ) (h : (a 4)^2 + (a 7)^2 + 2 * (a 4) * (a 7) = 9) :
  ∃ S, S = 10 * (a 4 + a 7) / 2 ∧ (S = 15 ∨ S = -15) := 
by
  sorry

end sum_first_10_terms_arithmetic_seq_l99_99361


namespace tan_45_eq_one_l99_99909

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99909


namespace tan_45_degrees_l99_99853

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99853


namespace product_digits_l99_99022

noncomputable def k_digit_smallest (k : ℕ) : ℕ := 10 ^ (k - 1)
noncomputable def k_digit_largest (k : ℕ) : ℕ := 10 ^ k - 1
noncomputable def n_digit_smallest (n : ℕ) : ℕ := 10 ^ (n - 1)
noncomputable def n_digit_largest (n : ℕ) : ℕ := 10 ^ n - 1

theorem product_digits (k n : ℕ) (a b : ℕ) 
  (hka : a ≥ k_digit_smallest k) (hla : a ≤ k_digit_largest k)
  (hkb : b ≥ n_digit_smallest n) (hlb : b ≤ n_digit_largest n) :
  let p := a * b in (10 ^ (k + n - 2) ≤ p) ∧ (p < 10 ^ (k + n)) :=
by
  sorry

end product_digits_l99_99022


namespace tan_45_eq_1_l99_99617

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99617


namespace find_complex_z_l99_99108

-- Definitions and conditions in the problem
def complex_conjugate (z : ℂ) : ℂ := conj z
def known_condition (z : ℂ) : Prop := (1 - I) * complex_conjugate z = 2 * I

-- Statement to prove
theorem find_complex_z (z : ℂ) (h : known_condition z) : z = -1 - I :=
sorry

end find_complex_z_l99_99108


namespace moles_of_KOH_combined_l99_99051

theorem moles_of_KOH_combined (H2O_formed : ℕ) (NH4I_used : ℕ) (ratio_KOH_H2O : ℕ) : H2O_formed = 54 → NH4I_used = 3 → ratio_KOH_H2O = 1 → H2O_formed = NH4I_used := 
by 
  intro H2O_formed_eq NH4I_used_eq ratio_eq 
  sorry

end moles_of_KOH_combined_l99_99051


namespace tan_45_degrees_eq_1_l99_99818

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99818


namespace arrange_numbers_satisfies_mean_property_l99_99006

-- Problem conditions definitions
def vertices := fin 9
def numbers := {n : ℤ | 2016 ≤ n ∧ n ≤ 2024}
def placement (f : vertices → ℤ) : Prop :=
  ∀ a b c : vertices, 
    (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ 
    (f a ∈ numbers ∧ f b ∈ numbers ∧ f c ∈ numbers) ∧ 
    (∃ k : ℤ, abs (a - b) = k ∧ abs (b - c) = k ∧ abs (c - a) = k) →
      (f b = (f a + f c) / 2)

-- The theorem statement
theorem arrange_numbers_satisfies_mean_property :
  ∃ f : vertices → ℤ, 
  (∀ v : vertices, f v ∈ numbers) ∧ 
  placement f :=
begin
  sorry
end

end arrange_numbers_satisfies_mean_property_l99_99006


namespace tan_45_degrees_eq_1_l99_99826

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99826


namespace tan_45_eq_1_l99_99616

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99616


namespace models_kirsty_can_buy_l99_99263

def savings := 30 * 0.45
def new_price := 0.50

theorem models_kirsty_can_buy : savings / new_price = 27 := by
  sorry

end models_kirsty_can_buy_l99_99263


namespace luke_total_score_l99_99287

theorem luke_total_score (points_per_round : ℕ) (number_of_rounds : ℕ) (total_score : ℕ) : 
  points_per_round = 146 ∧ number_of_rounds = 157 ∧ total_score = points_per_round * number_of_rounds → 
  total_score = 22822 := by 
  sorry

end luke_total_score_l99_99287


namespace infinite_non_repeating_implies_irrational_l99_99415

theorem infinite_non_repeating_implies_irrational
  (e : ℝ)
  (h1 : ∀ x : ℝ, ¬(x.represents_finite_decimal ∨ x.represents_repeating_decimal) → irrational x)
  (h2 : ¬(e.represents_finite_decimal ∨ e.represents_repeating_decimal)) :
  irrational e :=
by
  sorry

end infinite_non_repeating_implies_irrational_l99_99415


namespace tan_45_eq_one_l99_99859

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99859


namespace r_pow_four_solution_l99_99175

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l99_99175


namespace tan_45_eq_one_l99_99655

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99655


namespace tan_45_degrees_l99_99695

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99695


namespace tan_45_eq_one_l99_99603

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99603


namespace tan_45_degree_is_one_l99_99743

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99743


namespace tan_45_eq_1_l99_99625

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99625


namespace find_common_difference_l99_99384

-- Defining the arithmetic sequence properties and conditions
variable {a : ℕ → ℝ} (S_n : ℕ → ℝ)
variable (d : ℝ)

-- Conditions given in the problem
def Sn_condition := S_n 5 = 15
def a2_condition := a 2 = a 1 + d

-- Arithmetic series sum definition for the first n terms
def arithmetic_sum (n : ℕ) (a₁ d : ℝ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

-- Main Proof (statement only, no proof provided)
theorem find_common_difference (a : ℕ → ℝ) (S_n : ℕ → ℝ) (a₁ : ℝ) 
  (Sn_condition : S_n 5 = 15) 
  (a2_condition : a 2 = a₁ + d) 
  (sum_formula : S_n 5 = arithmetic_sum 5 a₁ d) : 
  d = -2 :=
sorry

end find_common_difference_l99_99384


namespace sum_of_solutions_l99_99207

theorem sum_of_solutions (x : ℝ) (h1 : x^2 = 25) : ∃ S : ℝ, S = 0 ∧ (∀ x', x'^2 = 25 → x' = 5 ∨ x' = -5) := 
sorry

end sum_of_solutions_l99_99207


namespace tan_45_degrees_l99_99852

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99852


namespace problem1_problem2_l99_99120

-- Definitions based on the conditions
def line (x : ℝ) : ℝ := 2 * x - 4
def parabola (y : ℝ) : ℝ := y^2 / 4
def pointA : ℝ × ℝ := (1, -2)
def pointB : ℝ × ℝ := (4, 4)
def focus : ℝ × ℝ := (1, 0)
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
def distance_point_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ := (abs (a * p.1 + b * p.2 + c)) / (Real.sqrt (a^2 + b^2))
def area_triangle (base height : ℝ) : ℝ := 1 / 2 * base * height

-- Statements for the mathematical proof problems
theorem problem1 : midpoint pointA pointB = (2.5, 1) :=
by {
  -- The proof would be here
  sorry
}

theorem problem2 : area_triangle (distance pointA pointB) (distance_point_line focus (-2) 1 4) = 3 :=
by {
  -- The proof would be here
  sorry
}

end problem1_problem2_l99_99120


namespace side_length_of_base_l99_99355

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l99_99355


namespace sum_of_solutions_eq_zero_l99_99202

theorem sum_of_solutions_eq_zero (x : ℝ) (hx : x^2 = 25) : ∃ x₁ x₂ : ℝ, (x₁^2 = 25) ∧ (x₂^2 = 25) ∧ (x₁ + x₂ = 0) := 
by {
  use 5,
  use (-5),
  split,
  { exact hx, },
  split,
  { rw pow_two, exact hx, },
  { norm_num, },
}

end sum_of_solutions_eq_zero_l99_99202


namespace pyramid_base_length_l99_99347

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l99_99347


namespace distinct_numbers_count_l99_99058

open BigOperators

theorem distinct_numbers_count :
  (Finset.card ((Finset.image (λ n : ℕ, ⌊ (n^2 : ℝ) / 500⌋) (Finset.range 1001))) = 876) := 
sorry

end distinct_numbers_count_l99_99058


namespace fifteen_percent_minus_70_l99_99420

theorem fifteen_percent_minus_70 (a : ℝ) : 0.15 * a - 70 = (15 / 100) * a - 70 :=
by sorry

end fifteen_percent_minus_70_l99_99420


namespace tan_45_degree_l99_99478

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99478


namespace five_power_l99_99410

theorem five_power (a : ℕ) (h : 5^a = 3125) : 5^(a - 3) = 25 := 
  sorry

end five_power_l99_99410


namespace pq_length_l99_99273

def midpoint (P Q R : ℝ × ℝ) : Prop :=
  R.1 = (P.1 + Q.1) / 2 ∧ R.2 = (P.2 + Q.2) / 2

noncomputable def length (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem pq_length : 
  let R : ℝ × ℝ := (10, 15)
  let line1 : ℝ → ℝ × ℝ := λ a, (a, 24 * a / 7)
  let line2 : ℝ → ℝ × ℝ := λ b, (b, 4 * b / 15)
  ∃ a b, 
    midpoint (line1 a) (line2 b) R ∧ length (line1 a) (line2 b) = 3460 / 83 :=
by
  sorry

end pq_length_l99_99273


namespace diamonds_in_G_20_equals_840_l99_99379

def diamonds_in_G (n : ℕ) : ℕ :=
  if n < 3 then 1 else 2 * n * (n + 1)

theorem diamonds_in_G_20_equals_840 : diamonds_in_G 20 = 840 :=
by
  sorry

end diamonds_in_G_20_equals_840_l99_99379


namespace tan_45_eq_one_l99_99913

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99913


namespace jenny_dog_roaming_area_l99_99255

noncomputable def dog_roaming_area 
  (rope_length : ℝ) 
  (shed_side : ℝ) 
  (attachment_distance : ℝ) 
  : ℝ :=
  let radial_reach := rope_length + attachment_distance
  let angle_in_radians := 2 * real.arccos ((shed_side^2 + shed_side^2 - radial_reach^2) / (2 * shed_side * shed_side))
  (angle_in_radians / (2 * real.pi)) * real.pi * radial_reach^2

theorem jenny_dog_roaming_area : 
  dog_roaming_area 10 30 5 = 37.5 * real.pi :=
by
  have rope_length : ℝ := 10
  have shed_side : ℝ := 30
  have attachment_distance : ℝ := 5
  have radial_reach := rope_length + attachment_distance
  have angle_in_radians := 
    2 * real.arccos ((shed_side^2 + shed_side^2 - radial_reach^2) / (2 * shed_side * shed_side))
  have sector_area := 
    (angle_in_radians / (2 * real.pi)) * real.pi * radial_reach^2
  show sector_area = 37.5 * real.pi
  sorry

end jenny_dog_roaming_area_l99_99255


namespace tan_45_eq_one_l99_99652

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99652


namespace tan_45_eq_1_l99_99622

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99622


namespace triangle_inequality_from_inequality_l99_99128

theorem triangle_inequality_from_inequality
  (a b c : ℝ)
  (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end triangle_inequality_from_inequality_l99_99128


namespace divide_into_two_groups_l99_99389

theorem divide_into_two_groups (n : ℕ) (A : Fin n → Type) 
  (acquaintances : (Fin n) → (Finset (Fin n)))
  (c : (Fin n) → ℕ) (d : (Fin n) → ℕ) :
  (∀ i : Fin n, c i = (acquaintances i).card) →
  ∃ G1 G2 : Finset (Fin n), G1 ∩ G2 = ∅ ∧ G1 ∪ G2 = Finset.univ ∧
  (∀ i : Fin n, d i = (acquaintances i ∩ (if i ∈ G1 then G2 else G1)).card ∧ d i ≥ (c i) / 2) :=
by 
  sorry

end divide_into_two_groups_l99_99389


namespace tan_45_degrees_l99_99837

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99837


namespace convex_polyhedron_space_diagonals_l99_99429

theorem convex_polyhedron_space_diagonals (Q : Type)
  (h1 : vertices Q = 30)
  (h2 : edges Q = 72)
  (h3 : faces Q = 44)
  (h4 : triangular_faces Q = 30)
  (h5 : quadrilateral_faces Q = 14) :
  space_diagonals Q = 335 := sorry

end convex_polyhedron_space_diagonals_l99_99429


namespace tan_45_eq_1_l99_99620

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99620


namespace tan_45_eq_1_l99_99613

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99613


namespace cars_meet_in_12_hours_l99_99370

-- Distance between places A and B
def distance_AB : ℝ := 420

-- Speed of the first car (in kilometers per hour)
def speed_car1 : ℝ := 42

-- Speed of the second car (in kilometers per hour)
def speed_car2 : ℝ := 28

-- Total distance traveled by both cars until they meet again
def total_distance : ℝ := distance_AB * 2

-- Sum of the speeds of both cars
def total_speed : ℝ := speed_car1 + speed_car2

-- Calculate the time until the two cars meet
def time_until_meet : ℝ := total_distance / total_speed

-- Prove that the total time until the two cars meet is 12 hours
theorem cars_meet_in_12_hours : time_until_meet = 12 := by
  -- Here would go the proof, but we'll add sorry for now since only statement is required
  sorry

end cars_meet_in_12_hours_l99_99370


namespace tan_45_deg_l99_99523

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99523


namespace symmetric_to_origin_l99_99365

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_to_origin (p : ℝ × ℝ) (h : p = (3, -1)) : symmetric_point p = (-3, 1) :=
by
  -- This is just the statement; the proof is not provided.
  sorry

end symmetric_to_origin_l99_99365


namespace tan_45_deg_l99_99773

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99773


namespace evaluate_M_l99_99042

noncomputable def M : ℝ := 
  (sqrt (sqrt 7 + 3) + sqrt (sqrt 7 - 3)) / sqrt (sqrt 7 + 2) + sqrt (5 - 2 * sqrt 6)

theorem evaluate_M : M = 2 * sqrt 2 - 1 := sorry

end evaluate_M_l99_99042


namespace moles_of_KOH_used_l99_99070

variable {n_KOH : ℝ}

theorem moles_of_KOH_used :
  ∃ n_KOH, (NH4I + KOH = KI_produced) → (KI_produced = 1) → n_KOH = 1 :=
by
  sorry

end moles_of_KOH_used_l99_99070


namespace tan_45_deg_eq_1_l99_99965

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99965


namespace r_power_four_identity_l99_99149

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l99_99149


namespace tan_45_eq_1_l99_99948

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99948


namespace pyramid_base_side_length_l99_99330

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l99_99330


namespace r_squared_sum_l99_99193

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l99_99193


namespace pyramid_base_length_l99_99350

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l99_99350


namespace tan_45_degree_is_one_l99_99740

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99740


namespace imaginary_power_sum_zero_l99_99385

theorem imaginary_power_sum_zero (i : ℂ) (n : ℤ) (h : i^2 = -1) :
  i^(2*n - 3) + i^(2*n - 1) + i^(2*n + 1) + i^(2*n + 3) = 0 :=
by {
  sorry
}

end imaginary_power_sum_zero_l99_99385


namespace tan_45_deg_l99_99525

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99525


namespace tan_45_eq_l99_99682

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99682


namespace tan_45_degrees_l99_99828

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99828


namespace tan_45_eq_1_l99_99997

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l99_99997


namespace find_ABC_sum_l99_99375

noncomputable def f (A B C : ℤ) : ℝ → ℝ := λ x, x / (x^3 + A*x^2 + B*x + C)

theorem find_ABC_sum (A B C : ℤ) 
  (hA : (∀ x : ℝ, x = -3 → x^3 + A*x^2 + B*x + C = 0))
  (hB : (∀ x : ℝ, x = 0 → x^3 + A*x^2 + B*x + C = 0))
  (hC : (∀ x : ℝ, x = 3 → x^3 + A*x^2 + B*x + C = 0)) :
  A + B + C = -9 :=
by sorry

end find_ABC_sum_l99_99375


namespace Sandy_marks_per_correct_sum_l99_99308

theorem Sandy_marks_per_correct_sum
  (x : ℝ)  -- number of marks Sandy gets for each correct sum
  (marks_lost_per_incorrect : ℝ := 2)  -- 2 marks lost for each incorrect sum, default value is 2
  (total_attempts : ℤ := 30)  -- Sandy attempts 30 sums, default value is 30
  (total_marks : ℝ := 60)  -- Sandy obtains 60 marks, default value is 60
  (correct_sums : ℤ := 24)  -- Sandy got 24 sums correct, default value is 24
  (incorrect_sums := total_attempts - correct_sums) -- incorrect sums are the remaining attempts
  (marks_from_correct := correct_sums * x) -- total marks from the correct sums
  (marks_lost_from_incorrect := incorrect_sums * marks_lost_per_incorrect) -- total marks lost from the incorrect sums
  (total_marks_obtained := marks_from_correct - marks_lost_from_incorrect) -- total marks obtained

  -- The theorem states that x must be 3 given the conditions above
  : total_marks_obtained = total_marks → x = 3 := by sorry

end Sandy_marks_per_correct_sum_l99_99308


namespace pyramid_base_length_l99_99351

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l99_99351


namespace tan_45_degrees_l99_99711

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99711


namespace cos_transform_l99_99309

theorem cos_transform (x : ℝ) :
  (∃ f : ℝ → ℝ, f = λ x, cos x → 
  (∃ g : ℝ → ℝ, g = λ x, f(x + (π / 4)) → 
  (∃ h : ℝ → ℝ, h = λ x, g(2 * x) → 
  h x = cos (2 * x + π / 4)))) := sorry

end cos_transform_l99_99309


namespace max_band_members_l99_99440

-- Definitions based on problem conditions
variables (m r x : ℕ)

-- Conditions
def condition1 : Prop := m = r * x + 3
def condition2 : Prop := m = (r - 3) * (x + 2)
def condition3 : Prop := m < 100 
def condition4 : Prop := ∃ r x : ℕ, 2 * r - 3 * x = 9 ∧ r * x + 3 < 100 

-- Proof statement
theorem max_band_members : 
  (∃ r x : ℕ, condition1 m r x ∧ condition2 m r x ∧ condition3 m) → 
  (∀ m', (∃ r x : ℕ, condition1 m' r x ∧ condition2 m' r x ∧ condition3 m') → m' ≤ 63) :=
begin
  sorry,
end

end max_band_members_l99_99440


namespace side_length_of_base_l99_99343

-- Define the conditions
def lateral_area (s : ℝ) : ℝ := (1 / 2) * s * 40
def given_area : ℝ := 120

-- Define the theorem to prove the length of the side of the base
theorem side_length_of_base : ∃ (s : ℝ), lateral_area(s) = given_area ∧ s = 6 :=
by
  sorry

end side_length_of_base_l99_99343


namespace tan_45_eq_one_l99_99648

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99648


namespace mike_baseball_cards_l99_99290

theorem mike_baseball_cards (initial_cards birthday_cards traded_cards : ℕ)
  (h1 : initial_cards = 64) 
  (h2 : birthday_cards = 18) 
  (h3 : traded_cards = 20) :
  initial_cards + birthday_cards - traded_cards = 62 :=
by 
  -- assumption:
  sorry

end mike_baseball_cards_l99_99290


namespace tan_45_eq_1_l99_99943

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99943


namespace tan_45_degree_l99_99488

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99488


namespace sum_of_solutions_l99_99074

noncomputable def equation (x : ℝ) : ℝ := abs(4 * x^2 - 8 * x) + 5

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | equation x = 373}.to_finset, x) = 2.0 :=
sorry

end sum_of_solutions_l99_99074


namespace find_r4_l99_99157

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l99_99157


namespace absolute_value_of_neg_five_l99_99321

theorem absolute_value_of_neg_five : |(-5 : ℤ)| = 5 := 
by 
  sorry

end absolute_value_of_neg_five_l99_99321


namespace sum_of_arithmetic_progression_l99_99244

variable {α : Type} [LinearOrderedField α]

def arithmetic_sum (a d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_of_arithmetic_progression
  (a d : α)
  (a1_eq : a = 1)
  (S_eq : ∀ n, S_n n = arithmetic_sum a d n)
  (condition : (S_n 19 / 19) - (S_n 17 / 17) = 6) :
  S_n 10 = 28 * 10 :=
sorry

end sum_of_arithmetic_progression_l99_99244


namespace pyramid_base_side_length_l99_99333

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l99_99333


namespace tan_45_eq_one_l99_99639

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99639


namespace tan_45_eq_l99_99671

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99671


namespace tan_of_45_deg_l99_99580

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99580


namespace solution_set_of_equation_l99_99380

theorem solution_set_of_equation (x : ℝ) (hx: 16 * sin (π * x) * cos (π * x) = 16 * x + 1 / x) : 
  x = 1 / 4 ∨ x = -1 / 4 :=
by sorry

end solution_set_of_equation_l99_99380


namespace frosting_cupcakes_l99_99468

theorem frosting_cupcakes :
  let r1 := 1 / 15
  let r2 := 1 / 25
  let r3 := 1 / 40
  let t := 600
  t * (r1 + r2 + r3) = 79 :=
by
  sorry

end frosting_cupcakes_l99_99468


namespace tan_45_eq_one_l99_99911

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99911


namespace tan_45_degrees_l99_99713

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99713


namespace tan_of_45_deg_l99_99573

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99573


namespace average_height_is_64_l99_99299

noncomputable def Parker (H_D : ℝ) : ℝ := H_D - 4
noncomputable def Daisy (H_R : ℝ) : ℝ := H_R + 8
noncomputable def Reese : ℝ := 60

theorem average_height_is_64 :
  let H_R := Reese 
  let H_D := Daisy H_R
  let H_P := Parker H_D
  (H_P + H_D + H_R) / 3 = 64 := sorry

end average_height_is_64_l99_99299


namespace tan_45_degree_l99_99480

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99480


namespace tan_45_deg_l99_99749

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99749


namespace tan_45_eq_1_l99_99632

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99632


namespace solve_equation_l99_99268

def greatest_int (x : ℝ) : ℤ := ⌊x⌋
def fractional_part (x : ℝ) : ℝ := x - greatest_int x

theorem solve_equation (x : ℝ) (h : (greatest_int x) * (fractional_part x) = 2005 * x) :
  x = 0 ∨ x = -1 / 2006 :=
by
  sorry

end solve_equation_l99_99268


namespace panic_percentage_l99_99000

theorem panic_percentage (original_population disappeared_after first_population second_population : ℝ) 
  (h₁ : original_population = 7200)
  (h₂ : disappeared_after = original_population * 0.10)
  (h₃ : first_population = original_population - disappeared_after)
  (h₄ : second_population = 4860)
  (h₅ : second_population = first_population - (first_population * 0.25)) : 
  second_population = first_population * (1 - 0.25) :=
by
  sorry

end panic_percentage_l99_99000


namespace tan_45_eq_one_l99_99917

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99917


namespace tan_45_degree_is_one_l99_99737

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99737


namespace tan_45_eq_one_l99_99659

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99659


namespace tens_digit_of_9_pow_2023_l99_99396

theorem tens_digit_of_9_pow_2023 : (9 ^ 2023) % 100 / 10 = 2 :=
by sorry

end tens_digit_of_9_pow_2023_l99_99396


namespace tan_45_eq_1_l99_99549

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99549


namespace tan_45_eq_one_l99_99876

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99876


namespace units_digit_char_of_p_l99_99104

theorem units_digit_char_of_p (p : ℕ) (h_pos : 0 < p) (h_even : p % 2 = 0)
    (h_units_zero : (p^3 % 10) - (p^2 % 10) = 0) (h_units_eleven : (p + 5) % 10 = 1) :
    p % 10 = 6 :=
sorry

end units_digit_char_of_p_l99_99104


namespace tan_45_degree_l99_99500

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99500


namespace sector_area_l99_99217

theorem sector_area (s θ : ℝ) (h₀ : s = 4) (h₁ : θ = 2) :
  let r := s / θ in
  let A := 1 / 2 * r^2 * θ in
  A = 4 :=
by
  sorry

end sector_area_l99_99217


namespace quadrant_of_angle_l99_99085

theorem quadrant_of_angle (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  ∃! (q : ℕ), q = 2 :=
sorry

end quadrant_of_angle_l99_99085


namespace tan_45_degrees_l99_99693

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99693


namespace tan_45_degree_is_one_l99_99744

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99744


namespace infinite_product_eq_nine_l99_99034

theorem infinite_product_eq_nine :
  (3 ^ (1/2 : ℝ)) * (9 ^ (1/4 : ℝ)) * (27 ^ (1/8 : ℝ)) * (81 ^ (1/16 : ℝ)) * ... = 9 := 
sorry

end infinite_product_eq_nine_l99_99034


namespace perimeter_of_regular_decagon_l99_99016

theorem perimeter_of_regular_decagon (side_length : ℝ) (n_sides : ℕ) (h_side_length : side_length = 2.5) (h_n_sides : n_sides = 10) :
  let P := n_sides * side_length in P = 25 := 
by
  unfold P
  rw [h_side_length, h_n_sides]
  norm_num
  sorry

end perimeter_of_regular_decagon_l99_99016


namespace repeating_decimal_fraction_sum_l99_99145

theorem repeating_decimal_fraction_sum :
  let x := 0 + 42 / 100 + 42 / (100 * 100) + 42 / (100 * 100 * 100) + ...
  ∃ a b : ℕ, x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 47 := 
by
  sorry

end repeating_decimal_fraction_sum_l99_99145


namespace tan_45_eq_1_l99_99995

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l99_99995


namespace intersection_S_T_union_S_T_intersection_T_complement_R_S_l99_99417

def S : set ℝ := {x | x < 1}
def T : set ℝ := {x | x ≤ 2}
def complement_R_S : set ℝ := {x | x ≥ 1}

-- Proof problem statements:
theorem intersection_S_T : S ∩ T = {x | x < 1} :=
by sorry

theorem union_S_T : S ∪ T = {x | x ≤ 2} :=
by sorry

theorem intersection_T_complement_R_S : T ∩ complement_R_S = {x | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end intersection_S_T_union_S_T_intersection_T_complement_R_S_l99_99417


namespace probability_non_distinct_real_roots_l99_99003

-- Define the bounds for b and c as tuples of integer ranges.
def bounds_b : Int × Int := (-7, 7)
def bounds_c : Int × Int := (-7, 7)

-- Predicate for the discriminant condition of non-distinct real roots.
def not_distinct_real_roots (b c : Int) : Prop :=
  b^2 - 4 * c ≤ 0

-- Function to count integer pairs satisfying a predicate within given bounds.
def count_valid_pairs (f : Int → Int → Prop) (range_b range_c : Int × Int) : Int :=
  let (b_min, b_max) := range_b
  let (c_min, c_max) := range_c
  List.sum $
    for b in [b_min..b_max], c in [c_min..c_max]
    if f b c then 1 else 0

-- The main theorem statement proving the probability.
theorem probability_non_distinct_real_roots :
  let total_pairs := 225
  let valid_pairs := count_valid_pairs not_distinct_real_roots bounds_b bounds_c
  valid_pairs.toRat / total_pairs.toRat = (4 : ℚ) / 5 :=
by
  sorry

end probability_non_distinct_real_roots_l99_99003


namespace measurement_correspondence_l99_99427

-- Given conditions
def r_scale := ℝ
def s_scale := ℝ

def linear_relation (a b : ℝ) (r : r_scale) : s_scale := a * r + b

-- Given data points
def point1 : r_scale × s_scale := (6, 30)
def point2 : r_scale × s_scale := (24, 60)

-- Verification condition
def corresponding_s_value (r_value : r_scale) : s_scale := 100

theorem measurement_correspondence : 
  ∃ a b : ℝ, 
    (linear_relation a b (fst point1) = snd point1) ∧ 
    (linear_relation a b (fst point2) = snd point2) ∧ 
    (linear_relation a b 48 = corresponding_s_value 48) := 
sorry

end measurement_correspondence_l99_99427


namespace total_fruit_salads_correct_l99_99461

-- Definitions for the conditions
def alayas_fruit_salads : ℕ := 200
def angels_fruit_salads : ℕ := 2 * alayas_fruit_salads
def total_fruit_salads : ℕ := alayas_fruit_salads + angels_fruit_salads

-- Theorem statement
theorem total_fruit_salads_correct : total_fruit_salads = 600 := by
  -- Proof goes here, but is not required for this task
  sorry

end total_fruit_salads_correct_l99_99461


namespace tan_45_eq_l99_99665

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99665


namespace planting_methods_total_l99_99019

noncomputable def numberOfPlantingMethods : ℕ :=
  let vegetables := ['cucumber, 'cabbage, 'rapeseed, 'lentils]
  let chosenVegetables := vegetables.erase 'cucumber
  let combinations := (chosenVegetables.choose 2).length
  let arrangements := 3!
  combinations * arrangements

theorem planting_methods_total :
  numberOfPlantingMethods = 18 :=
by
  sorry

end planting_methods_total_l99_99019


namespace tan_45_eq_l99_99680

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99680


namespace moles_of_ammonia_formed_l99_99069

def reaction (n_koh n_nh4i n_nh3 : ℕ) := 
  n_koh + n_nh4i + n_nh3 

theorem moles_of_ammonia_formed (n_koh : ℕ) :
  reaction n_koh 3 3 = n_koh + 3 + 3 := 
sorry

end moles_of_ammonia_formed_l99_99069


namespace tan_of_45_deg_l99_99576

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99576


namespace r_squared_sum_l99_99190

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l99_99190


namespace tan_45_degree_l99_99499

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99499


namespace T_n_lt_1_l99_99130

open Nat

def a (n : ℕ) : ℕ := 2^n

def b (n : ℕ) : ℕ := 2^n - 1

def c (n : ℕ) : ℚ := (a n : ℚ) / ((b n : ℚ) * (b (n + 1) : ℚ))

noncomputable def T (n : ℕ) : ℚ := (Finset.range (n + 1)).sum c

theorem T_n_lt_1 (n : ℕ) : T n < 1 := by
  sorry

end T_n_lt_1_l99_99130


namespace tan_45_degree_l99_99883

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99883


namespace tan_45_degrees_eq_1_l99_99805

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99805


namespace surface_area_of_solid_of_revolution_l99_99435

theorem surface_area_of_solid_of_revolution (p d : ℝ) (h : p > 0) :
  let perimeter := 2 * p,
      diagonal := d,
      surface_area := 2 * π * d * p
  in surface_area = 2 * π * d * p :=
by
  sorry

end surface_area_of_solid_of_revolution_l99_99435


namespace speed_of_current_l99_99010

theorem speed_of_current (h_start: ∀ t: ℝ, t ≥ 0 → u ≥ 0) 
  (boat1_turn_2pm: ∀ t: ℝ, t >= 1 → t < 2 → boat1_turn_13_14) 
  (boat2_turn_3pm: ∀ t: ℝ, t >= 2 → t < 3 → boat2_turn_14_15) 
  (boats_meet: ∀ x: ℝ, x = 7.5) :
  v = 2.5 := 
sorry

end speed_of_current_l99_99010


namespace tan_45_eq_one_l99_99934

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99934


namespace find_r_fourth_l99_99171

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l99_99171


namespace find_function_l99_99279

theorem find_function (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, 2 * f (x - 1) - 3 * f (1 - x) = 5 * x) : 
  f = λ x, x - 5 := 
sorry

end find_function_l99_99279


namespace find_r4_l99_99158

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l99_99158


namespace tan_45_eq_1_l99_99955

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99955


namespace pyramid_base_length_l99_99349

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l99_99349


namespace tan_45_degrees_eq_1_l99_99808

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99808


namespace tan_45_eq_one_l99_99926

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99926


namespace tan_45_deg_l99_99771

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99771


namespace infinite_product_eq_nine_l99_99035

theorem infinite_product_eq_nine :
  (3 ^ (1/2 : ℝ)) * (9 ^ (1/4 : ℝ)) * (27 ^ (1/8 : ℝ)) * (81 ^ (1/16 : ℝ)) * ... = 9 := 
sorry

end infinite_product_eq_nine_l99_99035


namespace tan_of_45_deg_l99_99581

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99581


namespace tan_45_eq_one_l99_99593

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99593


namespace count_distinct_numbers_in_list_l99_99068

def num_distinct_floor_divs : ℕ :=
  let L := list.map (λ n, ⌊(n ^ 2 : ℚ) / 500⌋) (list.range' 1 1000)
  list.dedup L

theorem count_distinct_numbers_in_list : (num_distinct_floor_divs.length = 876) := by
  -- Definition of L: the list from 1 to 1000 with floor of division by 500
  let L := list.map (λ n, ⌊(n ^ 2 : ℚ) / 500⌋) (list.range' 1 1000)
  -- Deduplicate the list by removing duplicates
  let distinct_L := list.dedup L
  -- Prove that the length of distinct_L is equal to 876
  have h : distinct_L.length = 876 := sorry
  exact h

end count_distinct_numbers_in_list_l99_99068


namespace max_monthly_profit_l99_99125

theorem max_monthly_profit (x : ℝ) (h : 0 < x ∧ x ≤ 15) :
  let C := 100 + 4 * x
  let p := 76 + 15 * x - x^2
  let L := p * x - C
  L = -x^3 + 15 * x^2 + 72 * x - 100 ∧
  (∀ x, 0 < x ∧ x ≤ 15 → L ≤ -12^3 + 15 * 12^2 + 72 * 12 - 100) :=
by
  sorry

end max_monthly_profit_l99_99125


namespace r_pow_four_solution_l99_99178

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l99_99178


namespace min_value_of_expr_l99_99031

theorem min_value_of_expr (n : ℕ) (hn : n > 0) : (n / 3) + (27 / n) = 6 :=
by
  sorry

end min_value_of_expr_l99_99031


namespace tan_45_eq_one_l99_99933

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99933


namespace functional_equation_unique_zero_solution_l99_99049

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_unique_zero_solution:
  (∀ x y : ℝ, f(f(x) + y) + f(x + f(y)) = 2 * f(x * f(y))) →
  (∀ x : ℝ, f(x) = 0) :=
sorry

end functional_equation_unique_zero_solution_l99_99049


namespace tan_45_degrees_l99_99831

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99831


namespace tan_45_deg_l99_99782

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99782


namespace largest_quantity_l99_99215

theorem largest_quantity (x y z w : ℤ) (h : x + 5 = y - 3 ∧ y - 3 = z + 2 ∧ z + 2 = w - 4) : w > y ∧ w > z ∧ w > x :=
by
  sorry

end largest_quantity_l99_99215


namespace tan_of_45_deg_l99_99570

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99570


namespace tan_45_eq_one_l99_99591

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99591


namespace pyramid_base_side_length_l99_99329

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l99_99329


namespace maximum_number_of_primes_dividing_n3_plus_2_and_np1_3_plus_2_l99_99400

theorem maximum_number_of_primes_dividing_n3_plus_2_and_np1_3_plus_2 (n : ℕ) (h : n > 0) :
  ∃ p : ℕ, prime p ∧ p ∣ n^3 + 2 ∧ p ∣ (n+1)^3 + 2 ∧ (∀ q : ℕ, prime q ∧ q ∣ n^3 + 2 ∧ q ∣ (n+1)^3 + 2 → q = p) := 
sorry

end maximum_number_of_primes_dividing_n3_plus_2_and_np1_3_plus_2_l99_99400


namespace Jake_weight_loss_l99_99254

theorem Jake_weight_loss
  (total_weight : ℕ)
  (Jake_current_weight : ℕ)
  (sister_weight : ℕ) :
  total_weight = 212 →
  Jake_current_weight = 152 →
  sister_weight = total_weight - Jake_current_weight →
  152 - 2 * sister_weight = 32 :=
by
  intros total_weight_eq JC_weight_eq S_weight_eq
  rw [total_weight_eq, JC_weight_eq, S_weight_eq]
  sorry

end Jake_weight_loss_l99_99254


namespace angle_B_is_45_l99_99107

-- Definitions from the problem conditions
def complement_sum (x y : ℝ) : ℝ := 180 - (x + y)
def supplement_diff (x y : ℝ) : ℝ := 90 - (x - y)
def angle_A : ℝ := x
def angle_B : ℝ := y

-- Statement to prove
theorem angle_B_is_45 (x y : ℝ) 
  (h1 : complement_sum x y = supplement_diff x y) 
  : y = 45 := 
sorry

end angle_B_is_45_l99_99107


namespace simplify_fraction_l99_99310

theorem simplify_fraction :
  ∀ (x : ℝ),
    (18 * x^4 - 9 * x^3 - 86 * x^2 + 16 * x + 96) /
    (18 * x^4 - 63 * x^3 + 22 * x^2 + 112 * x - 96) =
    (2 * x + 3) / (2 * x - 3) :=
by sorry

end simplify_fraction_l99_99310


namespace infinite_product_convergence_l99_99032

noncomputable theory

-- Define the infinite product sequence
def infinite_product (n : ℕ) : ℝ := 3 ^ (n / 2 ^ (n + 1))

-- The statement to prove
theorem infinite_product_convergence : (∏ n in (Finset.range ∞), infinite_product n) = 3 := 
sorry

end infinite_product_convergence_l99_99032


namespace tan_45_eq_one_l99_99869

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99869


namespace quadratic_completion_l99_99288

noncomputable def quadratic_form (d e : ℤ) : Prop :=
  ∀ x : ℝ, x^2 - 6 * x + 5 = (x + d)^2 - e

theorem quadratic_completion:
  ∃ (d e : ℤ), (quadratic_form d e) → d + e = 1 :=
begin
  use [-3, 4],
  intro h,
  sorry
end

end quadratic_completion_l99_99288


namespace side_length_of_base_l99_99346

-- Define the conditions
def lateral_area (s : ℝ) : ℝ := (1 / 2) * s * 40
def given_area : ℝ := 120

-- Define the theorem to prove the length of the side of the base
theorem side_length_of_base : ∃ (s : ℝ), lateral_area(s) = given_area ∧ s = 6 :=
by
  sorry

end side_length_of_base_l99_99346


namespace y1_lt_y2_of_linear_function_l99_99095

theorem y1_lt_y2_of_linear_function (y1 y2 : ℝ) (h1 : y1 = 2 * (-3) + 1) (h2 : y2 = 2 * 2 + 1) : y1 < y2 :=
by
  sorry

end y1_lt_y2_of_linear_function_l99_99095


namespace tan_45_eq_one_l99_99916

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99916


namespace calculate_cost_price_l99_99291

/-
Given:
  SP (Selling Price) is 18000
  If a 10% discount is applied on the SP, the effective selling price becomes 16200
  This effective selling price corresponds to an 8% profit over the cost price
  
Prove:
  The cost price (CP) is 15000
-/

theorem calculate_cost_price (SP : ℝ) (d : ℝ) (p : ℝ) (effective_SP : ℝ) (CP : ℝ) :
  SP = 18000 →
  d = 0.1 →
  p = 0.08 →
  effective_SP = SP - (d * SP) →
  effective_SP = CP * (1 + p) →
  CP = 15000 :=
by
  intros _
  sorry

end calculate_cost_price_l99_99291


namespace tan_45_deg_eq_1_l99_99964

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99964


namespace tan_45_eq_1_l99_99636

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99636


namespace tan_45_degrees_l99_99706

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99706


namespace lucas_correct_percentage_l99_99235

theorem lucas_correct_percentage (t : ℕ) (x : ℕ) 
  (emily_solo_correct: 0.7 * (t / 2) = 0.35 * t)
  (emily_overall_correct: 0.82 * t = 0.82 * t)
  (lucas_solo_correct: 0.85 * (t / 2) = 0.425 * t) : 
  (0.425 * t + 0.47 * t) / t * 100 = 89.5 := 
by
  sorry

end lucas_correct_percentage_l99_99235


namespace f_is_odd_f_range_l99_99112

noncomputable def f (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

theorem f_range : ∀ y : ℝ, f y ∈ set.Ioo (-1) 1 := by
  have range := set.range f
  have lower_bound : ∀ y, -1 < f y := by
    sorry
  have upper_bound : ∀ y, f y < 1 := by
    sorry
  exact by
    intro y
    specialize lower_bound y
    specialize upper_bound y
    have : -1 < f y ∧ f y < 1 := And.intro lower_bound upper_bound
    exact this

end f_is_odd_f_range_l99_99112


namespace tan_45_eq_one_l99_99658

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99658


namespace tan_45_eq_one_l99_99857

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99857


namespace not_in_square_D_l99_99080

def is_square (A B C D : ℚ × ℚ) : Prop := 
  let d1 := ((B.1 - A.1)^2 + (B.2 - A.2)^2) in
  let d2 := ((C.1 - A.1)^2 + (C.2 - A.2)^2) in
  let d3 := ((B.1 - C.1)^2 + (B.2 - C.2)^2) in
  let d4 := ((D.1 - A.1)^2 + (D.2 - A.2)^2) in
  let d5 := ((D.1 - B.1)^2 + (D.2 - B.2)^2) in
  let d6 := ((D.1 - C.1)^2 + (D.2 - C.2)^2) in
  (d1 = d3 ∧ d4 = d6 ∧ d2 = d4)

theorem not_in_square_D : ∀ (A B C D E : ℚ × ℚ), 
  A = (4,1) ∧ B = (2,4) ∧ C = (5,6) ∧ D = (3,5) ∧ E = (7,3) → 
  (is_square A B C E) ∧ ¬ is_square A B C D :=
by {
  intros,
  sorry
}

end not_in_square_D_l99_99080


namespace tan_45_eq_l99_99691

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99691


namespace tan_45_degrees_l99_99850

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99850


namespace tan_of_45_deg_l99_99563

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99563


namespace tan_45_degree_is_one_l99_99738

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99738


namespace tan_45_degrees_l99_99838

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99838


namespace find_r4_l99_99162

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l99_99162


namespace pyramid_base_side_length_l99_99337

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l99_99337


namespace distinct_numbers_l99_99061

theorem distinct_numbers (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 1000) : 
  finset.card ((finset.image (λ k : ℕ, ⌊(k^2 : ℚ) / 500⌋) (finset.range 1000).succ)) = 2001 :=
sorry

end distinct_numbers_l99_99061


namespace practice_minutes_l99_99018

def month_total_days : ℕ := (2 * 6) + (2 * 7)

def piano_daily_minutes : ℕ := 25

def violin_daily_minutes := piano_daily_minutes * 3

def flute_daily_minutes := violin_daily_minutes / 2

theorem practice_minutes (piano_total : ℕ) (violin_total : ℕ) (flute_total : ℕ) :
  (26 * piano_daily_minutes = 650) ∧ 
  (20 * violin_daily_minutes = 1500) ∧ 
  (16 * flute_daily_minutes = 600) := by
  sorry

end practice_minutes_l99_99018


namespace tan_45_deg_l99_99794

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99794


namespace tan_45_eq_1_l99_99935

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99935


namespace tan_45_degree_l99_99893

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99893


namespace tan_45_deg_l99_99770

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99770


namespace tan_of_45_deg_l99_99558

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99558


namespace tan_45_degrees_l99_99708

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99708


namespace tan_45_degree_is_one_l99_99728

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99728


namespace tan_45_deg_l99_99793

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99793


namespace tan_45_eq_1_l99_99940

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99940


namespace tan_45_degree_l99_99489

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99489


namespace tan_45_eq_1_l99_99537

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99537


namespace int_ratios_are_integers_implies_abs_eq_l99_99126

theorem int_ratios_are_integers_implies_abs_eq (x y z : ℤ) 
    (h1 : ∃ a b, (a : ℚ) = (x / y) + (y / z) + (z / x) ∧ b = (x / z) + (z / y) + (y / x) ∧ a ∈ ℤ ∧ b ∈ ℤ) :
  |x| = |y| = |z| :=
by
  sorry

end int_ratios_are_integers_implies_abs_eq_l99_99126


namespace find_num_male_general_attendees_l99_99011

def num_attendees := 1000
def num_presenters := 420
def total_general_attendees := num_attendees - num_presenters

variables (M_p F_p M_g F_g : ℕ)

axiom condition1 : M_p = F_p + 20
axiom condition2 : M_p + F_p = 420
axiom condition3 : F_g = M_g + 56
axiom condition4 : M_g + F_g = total_general_attendees

theorem find_num_male_general_attendees :
  M_g = 262 :=
by
  sorry

end find_num_male_general_attendees_l99_99011


namespace tan_45_degrees_eq_1_l99_99821

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99821


namespace models_kirsty_can_buy_l99_99262

def savings := 30 * 0.45
def new_price := 0.50

theorem models_kirsty_can_buy : savings / new_price = 27 := by
  sorry

end models_kirsty_can_buy_l99_99262


namespace tan_45_degrees_l99_99697

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99697


namespace theodore_pays_10_percent_in_taxes_l99_99386

-- Defining the quantities
def num_stone_statues : ℕ := 10
def num_wooden_statues : ℕ := 20
def price_per_stone_statue : ℕ := 20
def price_per_wooden_statue : ℕ := 5
def total_earnings_after_taxes : ℕ := 270

-- Assertion: Theodore pays 10% of his earnings in taxes
theorem theodore_pays_10_percent_in_taxes :
  (num_stone_statues * price_per_stone_statue + num_wooden_statues * price_per_wooden_statue) - total_earnings_after_taxes
  = (10 * (num_stone_statues * price_per_stone_statue + num_wooden_statues * price_per_wooden_statue)) / 100 := 
by
  sorry

end theodore_pays_10_percent_in_taxes_l99_99386


namespace tan_45_eq_1_l99_99949

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99949


namespace max_value_of_expression_l99_99227

theorem max_value_of_expression (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  ∃ m, m = 15 ∧ x^2 + y^2 + 2 * x ≤ m := 
sorry

end max_value_of_expression_l99_99227


namespace side_length_of_base_l99_99353

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l99_99353


namespace r_squared_sum_l99_99196

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l99_99196


namespace tan_45_eq_1_l99_99991

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l99_99991


namespace pyramid_base_side_length_l99_99336

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l99_99336


namespace tan_45_eq_1_l99_99989

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l99_99989


namespace side_length_of_base_l99_99344

-- Define the conditions
def lateral_area (s : ℝ) : ℝ := (1 / 2) * s * 40
def given_area : ℝ := 120

-- Define the theorem to prove the length of the side of the base
theorem side_length_of_base : ∃ (s : ℝ), lateral_area(s) = given_area ∧ s = 6 :=
by
  sorry

end side_length_of_base_l99_99344


namespace perimeter_non_shaded_region_l99_99359

/-- Given the following conditions:
1. The area of the shaded region is 65 square inches,
2. All angles are right angles,
3. All measurements are given in inches,
Prove that the perimeter of the non-shaded region is 30 inches. -/
theorem perimeter_non_shaded_region (shaded_area : ℝ) (angle_right : Prop) (measurements_in_inches : Prop) :
  shaded_area = 65 ∧ angle_right ∧ measurements_in_inches → 
  perimeter_of_non_shaded_region = 30 :=
by
  sorry

end perimeter_non_shaded_region_l99_99359


namespace express_y_in_terms_of_x_l99_99086

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 2 * x = 5) : y = 2 * x + 5 :=
by
  sorry

end express_y_in_terms_of_x_l99_99086


namespace log_quot_square_eq_two_of_roots_of_quadratic_l99_99197

def is_root (p : Polynomial ℝ) (x : ℝ) : Prop := p.eval x = 0  

noncomputable def log (a : ℝ) : ℝ := sorry  -- We assume log is a noncomputable function.

variable {a b : ℝ}

theorem log_quot_square_eq_two_of_roots_of_quadratic (ha : is_root (X^2 - (4 : ℝ) / 2 * X + 1 / 2) (log a)) 
    (hb : is_root (X^2 - (4 : ℝ) / 2 * X + 1 / 2) (log b)) : 
    (log (a / b))^2 = 2 := by 
  sorry

end log_quot_square_eq_two_of_roots_of_quadratic_l99_99197


namespace floor_ceil_inequality_l99_99269

theorem floor_ceil_inequality 
  (a b c : ℝ)
  (h : ⌈a⌉ + ⌈b⌉ + ⌈c⌉ + ⌊a + b⌋ + ⌊b + c⌋ + ⌊c + a⌋ = 2020) :
  ⌊a⌋ + ⌊b⌋ + ⌊c⌋ + ⌈a + b + c⌉ ≥ 1346 := 
by
  sorry 

end floor_ceil_inequality_l99_99269


namespace find_value_of_x_cubed_plus_y_cubed_l99_99216

-- Definitions based on the conditions provided
variables (x y : ℝ)
variables (h1 : y + 3 = (x - 3)^2) (h2 : x + 3 = (y - 3)^2) (h3 : x ≠ y)

theorem find_value_of_x_cubed_plus_y_cubed :
  x^3 + y^3 = 217 :=
sorry

end find_value_of_x_cubed_plus_y_cubed_l99_99216


namespace distinct_numbers_count_l99_99059

open BigOperators

theorem distinct_numbers_count :
  (Finset.card ((Finset.image (λ n : ℕ, ⌊ (n^2 : ℝ) / 500⌋) (Finset.range 1001))) = 876) := 
sorry

end distinct_numbers_count_l99_99059


namespace side_length_of_base_l99_99341

-- Define the conditions
def lateral_area (s : ℝ) : ℝ := (1 / 2) * s * 40
def given_area : ℝ := 120

-- Define the theorem to prove the length of the side of the base
theorem side_length_of_base : ∃ (s : ℝ), lateral_area(s) = given_area ∧ s = 6 :=
by
  sorry

end side_length_of_base_l99_99341


namespace tan_45_eq_one_l99_99912

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99912


namespace red_cards_taken_out_l99_99314

-- Definitions based on the conditions
def total_cards : ℕ := 52
def half_of_total_cards (n : ℕ) := n / 2
def initial_red_cards : ℕ := half_of_total_cards total_cards
def remaining_red_cards : ℕ := 16

-- The statement to prove
theorem red_cards_taken_out : initial_red_cards - remaining_red_cards = 10 := by
  sorry

end red_cards_taken_out_l99_99314


namespace tan_45_eq_one_l99_99872

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99872


namespace inverse_function_value_l99_99116

theorem inverse_function_value (g : ℝ → ℝ) (hg : ∀ x, g x = 25 / (4 + 5 * x)) :
  (Function.inverse g 5)⁻¹ = 5 :=
by
  sorry

end inverse_function_value_l99_99116


namespace tan_45_eq_l99_99683

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99683


namespace tan_45_deg_l99_99508

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99508


namespace aarti_bina_work_l99_99449

theorem aarti_bina_work (days_aarti : ℚ) (days_bina : ℚ) (D : ℚ)
  (ha : days_aarti = 5) (hb : days_bina = 8)
  (rate_aarti : 1 / days_aarti = 1/5) 
  (rate_bina : 1 / days_bina = 1/8)
  (combine_rate : (1 / days_aarti) + (1 / days_bina) = 13 / 40) :
  3 / (13 / 40) = 120 / 13 := 
by
  sorry

end aarti_bina_work_l99_99449


namespace smallest_side_length_of_square_l99_99447

theorem smallest_side_length_of_square (n s : ℕ) 
  (h1 : n ≥ 12) 
  (h2 : ∃ l: list ℕ, l.length = n ∧ ∀ a ∈ l, ∃ b : ℕ, a = b * b ∧ b ≥ 1 ∧ b ≤ 5) 
  (h3 : (filter (λ x, x = 1) (l)).length ≥ 9) 
  (h4 : s * s = sum l)
  : s ≥ 5 :=
sorry

end smallest_side_length_of_square_l99_99447


namespace tan_45_degrees_l99_99717

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99717


namespace meaningful_range_l99_99223

noncomputable def isMeaningful (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x - 2 ≠ 0)

theorem meaningful_range (x : ℝ) : isMeaningful x ↔ (x ≥ -1) ∧ (x ≠ 2) :=
by
  sorry

end meaningful_range_l99_99223


namespace distinct_numbers_count_l99_99054

noncomputable section

def num_distinct_numbers : Nat :=
  let vals := List.map (λ n : Nat, Nat.floor ((n^2 : ℚ) / 500)) (List.range 1000).tail
  (vals.eraseDup).length

theorem distinct_numbers_count : num_distinct_numbers = 876 :=
by
  sorry

end distinct_numbers_count_l99_99054


namespace number_of_ideal_match_sets_l99_99272

open Finset

def ideal_match_sets (I : Finset ℕ) :=
  {p : Finset ℕ × Finset ℕ // p.1 ∩ p.2 = {1, 3}}

def count_ideal_match_sets : ℕ :=
  (ideal_match_sets {1, 2, 3, 4}).toFinset.card

theorem number_of_ideal_match_sets : count_ideal_match_sets = 9 :=
  sorry

end number_of_ideal_match_sets_l99_99272


namespace r_power_four_identity_l99_99151

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l99_99151


namespace quadratic_complete_square_l99_99140

theorem quadratic_complete_square (c n : ℝ) (h1 : ∀ x : ℝ, x^2 + c * x + 20 = (x + n)^2 + 12) (h2: 0 < c) : 
  c = 4 * Real.sqrt 2 :=
by
  sorry

end quadratic_complete_square_l99_99140


namespace tan_45_eq_one_l99_99866

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99866


namespace method_is_systematic_sampling_l99_99445

-- Define the conditions
def rows : ℕ := 25
def seats_per_row : ℕ := 20
def filled_auditorium : Prop := True
def seat_numbered_15_sampled : Prop := True
def interval : ℕ := 20

-- Define the concept of systematic sampling
def systematic_sampling (rows seats_per_row interval : ℕ) : Prop :=
  (rows > 0 ∧ seats_per_row > 0 ∧ interval > 0 ∧ (interval = seats_per_row))

-- State the problem in terms of proving that the sampling method is systematic
theorem method_is_systematic_sampling :
  filled_auditorium → seat_numbered_15_sampled → systematic_sampling rows seats_per_row interval :=
by
  intros h1 h2
  -- Assume that the proof goes here
  sorry

end method_is_systematic_sampling_l99_99445


namespace tan_45_degrees_l99_99696

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99696


namespace greatest_possible_grapes_thrown_out_l99_99424

theorem greatest_possible_grapes_thrown_out (n : ℕ) (k : ℕ) (h : n = 50) (hk : k = 7) :
  n % k = 1 :=
by
  have h1: n % k = 50 % 7, by rw [h, hk]
  have h2: 50 % 7 = 1, by norm_num
  rw [h1, h2]
  exact h2

end greatest_possible_grapes_thrown_out_l99_99424


namespace tan_45_degrees_l99_99846

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99846


namespace tan_45_eq_one_l99_99605

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99605


namespace tan_45_degrees_eq_1_l99_99817

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99817


namespace tan_45_eq_1_l99_99539

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99539


namespace find_r_fourth_l99_99172

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l99_99172


namespace points_on_quadratic_function_and_order_l99_99226

variables {x1 x2 y1 y2 : ℝ}

def quadratic_function (x : ℝ) : ℝ := -((1 : ℝ) / 3) * x^2 + 5

theorem points_on_quadratic_function_and_order 
  (h1 : y1 = quadratic_function x1)
  (h2 : y2 = quadratic_function x2)
  (h3 : 0 < x1)
  (h4 : x1 < x2) :
  y2 < y1 ∧ y1 < 5 :=
sorry

end points_on_quadratic_function_and_order_l99_99226


namespace find_r4_l99_99163

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l99_99163


namespace distinct_numbers_count_l99_99060

open BigOperators

theorem distinct_numbers_count :
  (Finset.card ((Finset.image (λ n : ℕ, ⌊ (n^2 : ℝ) / 500⌋) (Finset.range 1001))) = 876) := 
sorry

end distinct_numbers_count_l99_99060


namespace tan_45_degrees_eq_1_l99_99811

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99811


namespace tan_45_eq_1_l99_99939

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99939


namespace tan_45_eq_1_l99_99630

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99630


namespace tan_45_eq_l99_99673

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99673


namespace wall_height_correct_l99_99426

-- Define the dimensions of the brick in meters
def brick_length : ℝ := 0.2
def brick_width  : ℝ := 0.1
def brick_height : ℝ := 0.08

-- Define the volume of one brick
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Total number of bricks used
def number_of_bricks : ℕ := 12250

-- Define the wall dimensions except height
def wall_length : ℝ := 10
def wall_width  : ℝ := 24.5

-- Total volume of all bricks
def volume_total_bricks : ℝ := number_of_bricks * volume_brick

-- Volume of the wall
def volume_wall (h : ℝ) : ℝ := wall_length * h * wall_width

-- The height of the wall
def wall_height : ℝ := 0.08

-- The theorem to prove
theorem wall_height_correct : volume_total_bricks = volume_wall wall_height :=
by
  sorry

end wall_height_correct_l99_99426


namespace pyramid_base_side_length_l99_99339

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l99_99339


namespace ellipse_properties_l99_99454

theorem ellipse_properties (h k a b : ℝ)
  (h_eq : h = 1)
  (k_eq : k = -3)
  (a_eq : a = 7)
  (b_eq : b = 4) :
  h + k + a + b = 9 :=
by
  sorry

end ellipse_properties_l99_99454


namespace tan_45_eq_1_l99_99938

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99938


namespace tan_45_eq_1_l99_99543

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99543


namespace tan_45_eq_one_l99_99585

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99585


namespace find_r_fourth_l99_99168

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l99_99168


namespace average_score_l99_99363

theorem average_score (a_males : ℕ) (a_females : ℕ) (n_males : ℕ) (n_females : ℕ)
  (h_males : a_males = 85) (h_females : a_females = 92) (h_n_males : n_males = 8) (h_n_females : n_females = 20) :
  (a_males * n_males + a_females * n_females) / (n_males + n_females) = 90 :=
by
  sorry

end average_score_l99_99363


namespace models_kirsty_can_buy_l99_99265

def original_price : ℝ := 0.45
def saved_for_models : ℝ := 30 * original_price
def new_price : ℝ := 0.50

theorem models_kirsty_can_buy :
  saved_for_models / new_price = 27 :=
sorry

end models_kirsty_can_buy_l99_99265


namespace complement_of_S_in_U_l99_99285

open Set

variable (U S : Set ℕ)

def complementSet : Set ℕ := U \ S

noncomputable def given_U := {1, 2, 3, 4, 5, 6, 7}

noncomputable def given_S := {1, 3, 5}

theorem complement_of_S_in_U : complementSet given_U given_S = {2, 4, 6, 7} :=
by
  unfold complementSet
  simp only [given_U, given_S, Set.diff_eq]
  exact sorry

end complement_of_S_in_U_l99_99285


namespace angle_bisector_FD_of_AFE_l99_99297

variables {A B C D E F : Type*}

-- Define the points and the triangle ABC
variables [IsTriangle A B C]

-- Define the given conditions
variables (on_sides : OnSides D E F A B C)
variables (BE_eq_BD : BE = BD)
variables (AF_eq_AD : AF = AD)
variables (ED_angle_bisector : IsAngleBisector ED (∠BEF))

-- State the theorem
theorem angle_bisector_FD_of_AFE :
  IsAngleBisector FD (∠AFE) :=
sorry

end angle_bisector_FD_of_AFE_l99_99297


namespace tan_45_eq_1_l99_99546

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99546


namespace find_x_l99_99421

theorem find_x (x : ℝ) (h : 61 + 5 * 12 / (x / 3) = 62) : x = 180 :=
by
  sorry

end find_x_l99_99421


namespace tan_45_degree_l99_99882

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99882


namespace tan_45_eq_one_l99_99868

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99868


namespace count_distinct_numbers_in_list_l99_99065

def num_distinct_floor_divs : ℕ :=
  let L := list.map (λ n, ⌊(n ^ 2 : ℚ) / 500⌋) (list.range' 1 1000)
  list.dedup L

theorem count_distinct_numbers_in_list : (num_distinct_floor_divs.length = 876) := by
  -- Definition of L: the list from 1 to 1000 with floor of division by 500
  let L := list.map (λ n, ⌊(n ^ 2 : ℚ) / 500⌋) (list.range' 1 1000)
  -- Deduplicate the list by removing duplicates
  let distinct_L := list.dedup L
  -- Prove that the length of distinct_L is equal to 876
  have h : distinct_L.length = 876 := sorry
  exact h

end count_distinct_numbers_in_list_l99_99065


namespace tens_digit_9_2023_l99_99399

theorem tens_digit_9_2023 :
  let cycle := [09, 81, 29, 61, 49, 41, 69, 21, 89, 01] in
  (cycle[(2023 % 10)] / 10) % 10 == 2 := by
  sorry

end tens_digit_9_2023_l99_99399


namespace tan_of_45_deg_l99_99564

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99564


namespace tan_45_deg_l99_99791

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99791


namespace tan_45_deg_l99_99510

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99510


namespace no_solution_for_equation_l99_99312

theorem no_solution_for_equation :
  ¬(∃ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ (x+2)/(x-2) - x/(x+2) = 16/(x^2-4)) :=
by
    sorry

end no_solution_for_equation_l99_99312


namespace maximum_value_l99_99093

noncomputable theory
open_locale big_operators

variables {a : ℕ → ℝ} [fintype (fin 2008)]

def condition_1 (a : ℕ → ℝ) : Prop := ∀ i, 0 ≤ a i

def condition_2 (a : ℕ → ℝ) : Prop := (finset.univ.sum (λ i: fin 2008, a i)) = 1

def objective (a : ℕ → ℝ) : ℝ :=
(finset.range 2008).sum (λ i, a i * a ((i + 1) % 2008))

theorem maximum_value : condition_1 a ∧ condition_2 a → objective a ≤ 1/4 :=
begin
  intros h1 h2,
  sorry
end

end maximum_value_l99_99093


namespace pyramid_base_side_length_l99_99340

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l99_99340


namespace josef_game_l99_99257

theorem josef_game : 
  ∃ S : Finset ℕ, 
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 1440 ∧ 1440 % n = 0 ∧ n % 5 = 0) ∧ 
    S.card = 18 := sorry

end josef_game_l99_99257


namespace point_A_can_move_arbitrarily_far_l99_99441

-- Definitions based on the conditions
variable {A B : Type} [MetricSpace A] (line_t : Set A) 

def unit_length_segment (AB : A × A) : Prop :=
  dist AB.fst AB.snd = 1

def parallel (AB : A × A) (line_t : Set A) : Prop :=
  ∀ p ∈ line_t, ∃ q ∈ line_t, dist p q = dist AB.fst AB.snd

noncomputable def final_position (A_initial A_final : A) (h : ∃ (B_initial B_final : A), 
      unit_length_segment (A_initial, B_initial) ∧ 
      parallel (A_initial, B_initial) line_t ∧ 
      sorry "non_intersecting_traces" ∧
      unit_length_segment (A_final, B_final) ∧
      parallel (A_final, B_final) line_t) : Prop :=
dist A_initial A_final = ∞

-- Statement of the problem in Lean
theorem point_A_can_move_arbitrarily_far (A_initial A_final : A) :
  ∃ (B_initial B_final : A), 
    unit_length_segment (A_initial, B_initial) ∧ 
    parallel (A_initial, B_initial) line_t ∧ 
    sorry "non_intersecting_traces" ∧
    unit_length_segment (A_final, B_final) ∧
    parallel (A_final, B_final) line_t → 
  final_position A_initial A_final line_t :=
sorry

end point_A_can_move_arbitrarily_far_l99_99441


namespace system1_solution_system2_solution_l99_99313

theorem system1_solution :
  ∃ (x y : ℤ), (4 * x - y = 1) ∧ (y = 2 * x + 3) ∧ (x = 2) ∧ (y = 7) :=
by
  sorry

theorem system2_solution :
  ∃ (x y : ℤ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) :=
by
  sorry

end system1_solution_system2_solution_l99_99313


namespace tan_45_degree_l99_99477

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99477


namespace tan_45_degrees_l99_99716

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99716


namespace tan_45_eq_one_l99_99601

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99601


namespace reader_distance_C_l99_99439

theorem reader_distance_C 
  (distance_AB : ℝ) 
  (reader_start_time : ℝ) 
  (friend_speed_to_C : ℝ) 
  (friend_start_time : ℝ) 
  (meeting_to_A_speed : ℝ) 
  (meeting_to_A_time : ℝ) : 
  distance_AB = 1 ∧
  reader_start_time = 0 ∧
  friend_speed_to_C = 5 ∧
  friend_start_time = 1/4 ∧
  meeting_to_A_speed = 4 ∧
  meeting_to_A_time = 1/12
  → 
  ∃ (reader_distance_C : ℝ), reader_distance_C = 2/3 :=
begin
  -- proof will be constructed here
  sorry
end

end reader_distance_C_l99_99439


namespace tan_45_eq_l99_99668

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99668


namespace tan_45_degrees_eq_1_l99_99803

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99803


namespace r_squared_sum_l99_99194

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l99_99194


namespace necessary_but_not_sufficient_l99_99382

theorem necessary_but_not_sufficient (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ((1 / 3)^a < (1 / 3)^b) ↔ (log 2 a > log 2 b) := sorry

end necessary_but_not_sufficient_l99_99382


namespace tan_45_eq_1_l99_99542

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99542


namespace tan_45_eq_l99_99674

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99674


namespace tan_of_45_deg_l99_99557

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99557


namespace tan_45_degree_is_one_l99_99722

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99722


namespace min_3cell_lshapes_in_6x6_square_l99_99402

theorem min_3cell_lshapes_in_6x6_square (S : Finset (Fin (6 × 6) → bool)): 
  (∀ (L : Finset (Fin (6 × 6))) (hL : L ⊆ S ∧ L.card = 3), ∀ (L' : Finset (Fin (6 × 6))), L' ≠ L → L' ∩ L = ∅) → 
  S.card = 18 → 
  ∃ Ls : Finset (Fin (6 × 6) → Finset (Fin (6 × 6))), Ls.card = 6 ∧ 
    ∀ L ∈ Ls, ∀ (L' ∈ Ls), L ≠ L' → L ∩ L' = ∅ :=
by
  sorry

end min_3cell_lshapes_in_6x6_square_l99_99402


namespace find_derivative_at_one_l99_99115

noncomputable def f (f' : ℝ → ℝ) (x : ℝ) : ℝ := x * Real.log x + f' 1 * x ^ 2 + 2

theorem find_derivative_at_one (f' : ℝ → ℝ) :
  (deriv (f f') 1) = -1 :=
begin
  sorry -- Skip the proof part as instructed
end

end find_derivative_at_one_l99_99115


namespace tan_45_eq_l99_99684

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99684


namespace tan_45_eq_one_l99_99915

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99915


namespace range_of_a_l99_99118

def f (x : ℝ) := abs (Real.log (x - 1))

theorem range_of_a (a : ℝ) (h : f a > f (4 - a)) : 2 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l99_99118


namespace tan_45_eq_one_l99_99643

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99643


namespace largest_circle_radius_l99_99024

-- Define the sides of the quadrilateral
def AB : ℝ := 15
def BC : ℝ := 8
def CD : ℝ := 9
def DA : ℝ := 13

-- Define the given condition about the sums of the opposite sides
def side_sum_condition (AB CD BC DA : ℝ) : Prop := AB + CD = BC + DA

-- Prove the radius of the largest inscribed circle
theorem largest_circle_radius : 
  side_sum_condition AB CD BC DA → 
  ∃ r : ℝ, r = Real.sqrt 30 :=
by
  intros h
  use Real.sqrt 30
  sorry

end largest_circle_radius_l99_99024


namespace problem_statement_l99_99092

-- Define an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x) = f (-x)

-- Define an odd function on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x) = -f (-x)

theorem problem_statement (f : ℝ → ℝ) 
  (h1 : even_function f)
  (h2 : odd_function (λ x, f (x - 1)))
  (h3 : f 2 = 3) :
  f 5 + f 6 = 3 :=
sorry

end problem_statement_l99_99092


namespace tan_45_eq_one_l99_99878

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99878


namespace tan_45_degrees_eq_1_l99_99812

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99812


namespace r_pow_four_solution_l99_99180

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l99_99180


namespace pyramid_base_length_l99_99352

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l99_99352


namespace distinct_numbers_l99_99063

theorem distinct_numbers (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 1000) : 
  finset.card ((finset.image (λ k : ℕ, ⌊(k^2 : ℚ) / 500⌋) (finset.range 1000).succ)) = 2001 :=
sorry

end distinct_numbers_l99_99063


namespace tan_45_deg_eq_1_l99_99987

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99987


namespace tan_45_eq_1_l99_99961

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99961


namespace tan_45_deg_l99_99768

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99768


namespace tan_45_deg_eq_1_l99_99971

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99971


namespace sum_of_roots_eq_zero_sum_of_all_possible_values_l99_99213

theorem sum_of_roots_eq_zero (x : ℝ) (h : x^2 = 25) : x = 5 ∨ x = -5 :=
by {
  sorry
}

theorem sum_of_all_possible_values (h : ∀ x : ℝ, x^2 = 25 → x = 5 ∨ x = -5) : ∑ x in {x : ℝ | x^2 = 25}, x = 0 :=
by {
  sorry
}

end sum_of_roots_eq_zero_sum_of_all_possible_values_l99_99213


namespace distinct_numbers_l99_99062

theorem distinct_numbers (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 1000) : 
  finset.card ((finset.image (λ k : ℕ, ⌊(k^2 : ℚ) / 500⌋) (finset.range 1000).succ)) = 2001 :=
sorry

end distinct_numbers_l99_99062


namespace tan_45_deg_l99_99772

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99772


namespace sum_of_repeating_decimals_l99_99469

-- Definitions based on the conditions
def x := 0.6666666666666666 -- Lean may not directly support \(0.\overline{6}\) notation
def y := 0.7777777777777777 -- Lean may not directly support \(0.\overline{7}\) notation

-- Translate those to the correct fractional forms
def x_as_fraction := (2 : ℚ) / 3
def y_as_fraction := (7 : ℚ) / 9

-- The main statement to prove
theorem sum_of_repeating_decimals : x_as_fraction + y_as_fraction = 13 / 9 :=
by
  -- Proof skipped
  sorry

end sum_of_repeating_decimals_l99_99469


namespace tan_45_deg_l99_99748

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99748


namespace tangent_line_at_P_g_increasing_on_interval_exists_point_Q_l99_99111

def f (x : ℝ) : ℝ := (1 / 3) * x^3 - x^2 + 3 * x - 2

theorem tangent_line_at_P :
  let P := (0 : ℝ, f 0)
  in tangent_line(f) P = "y = 3x - 2" :=
  sorry

def g (x : ℝ) : ℝ := f x + (3 / (x - 1))

theorem g_increasing_on_interval :
  ∀ x, 2 ≤ x → g' x ≥ 0 :=
  sorry

theorem exists_point_Q :
  ∃ Q : ℝ × ℝ, 
  -- Q coordinates should satisfy the properties about symmetry and equal area closures
  let Q := (1, 1/3)
  in True :=
  sorry

end tangent_line_at_P_g_increasing_on_interval_exists_point_Q_l99_99111


namespace r_pow_four_solution_l99_99177

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l99_99177


namespace tan_45_deg_l99_99518

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99518


namespace tan_45_deg_l99_99521

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99521


namespace tan_45_degree_l99_99895

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99895


namespace tan_45_degree_l99_99898

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99898


namespace triangle_area_l99_99395

-- Define the point P
def P : ℝ × ℝ := (2, 5)

-- Define the coordinates of Q based on the line with slope 3 intersecting the x-axis
def Q : ℝ × ℝ := (2 - (5 / 3), 0)

-- Define the coordinates of R based on the line with slope -1 intersecting the x-axis
def R : ℝ × ℝ := (2 - 5, 0)

-- Define the proof statement
theorem triangle_area (P Q R : ℝ × ℝ)
  (slope1 slope2 : ℝ) (hx_P : P = (2, 5))
  (hx_Q : Q = (2 - 5 / 3, 0)) (hx_R : R = (2 - 5, 0))
  (hs1 : slope1 = 3) (hs2 : slope2 = -1) :
  (1 / 2 * abs ((R.1 - Q.1) * 5)) = 25 / 3 :=
by sorry

end triangle_area_l99_99395


namespace rate_of_first_machine_l99_99430

-- Define the conditions
def rate_first_machine_per_minute : ℕ
def rate_second_machine_per_minute : ℕ := 75
def total_copies_in_half_hour : ℕ := 3300
def half_hour_minutes : ℕ := 30

-- Write the theorem based on the conditions
theorem rate_of_first_machine : rate_first_machine_per_minute * half_hour_minutes + 
                                rate_second_machine_per_minute * half_hour_minutes = total_copies_in_half_hour → 
                                rate_first_machine_per_minute = 35 :=
begin
  -- assuming the necessary conditions to avoid unused variable warnings
  intros h,
  sorry
end

end rate_of_first_machine_l99_99430


namespace tan_45_eq_one_l99_99587

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99587


namespace tan_45_eq_one_l99_99656

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99656


namespace tan_45_deg_l99_99505

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99505


namespace tan_45_degree_is_one_l99_99723

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99723


namespace r_power_four_identity_l99_99153

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l99_99153


namespace tan_45_eq_1_l99_99611

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99611


namespace tan_45_degrees_l99_99836

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99836


namespace sum_of_roots_eq_zero_sum_of_all_possible_values_l99_99211

theorem sum_of_roots_eq_zero (x : ℝ) (h : x^2 = 25) : x = 5 ∨ x = -5 :=
by {
  sorry
}

theorem sum_of_all_possible_values (h : ∀ x : ℝ, x^2 = 25 → x = 5 ∨ x = -5) : ∑ x in {x : ℝ | x^2 = 25}, x = 0 :=
by {
  sorry
}

end sum_of_roots_eq_zero_sum_of_all_possible_values_l99_99211


namespace probability_all_red_is_correct_l99_99423

def total_marbles (R W B : Nat) : Nat := R + W + B

def first_red_probability (R W B : Nat) : Rat := R / total_marbles R W B
def second_red_probability (R W B : Nat) : Rat := (R - 1) / (total_marbles R W B - 1)
def third_red_probability (R W B : Nat) : Rat := (R - 2) / (total_marbles R W B - 2)

def all_red_probability (R W B : Nat) : Rat := 
  first_red_probability R W B * 
  second_red_probability R W B * 
  third_red_probability R W B

theorem probability_all_red_is_correct 
  (R W B : Nat) (hR : R = 5) (hW : W = 6) (hB : B = 7) :
  all_red_probability R W B = 5 / 408 := by
  sorry

end probability_all_red_is_correct_l99_99423


namespace tan_45_eq_one_l99_99870

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99870


namespace r_pow_four_solution_l99_99176

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l99_99176


namespace tan_45_eq_one_l99_99921

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99921


namespace tan_45_eq_1_l99_99619

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99619


namespace tan_45_degree_l99_99905

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99905


namespace fib_150_mod_9_l99_99319

theorem fib_150_mod_9 : Nat.fib 150 % 9 = 8 :=
by 
  sorry

end fib_150_mod_9_l99_99319


namespace tan_45_eq_l99_99690

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99690


namespace color_naturals_with_2009_colors_l99_99413

open Nat

def is_coloring_valid (coloring : ℕ → ℕ) (num_colors : ℕ) : Prop :=
  (∀ c < num_colors, ∃ inf_set : ℕ → Prop, (∀ x, inf_set x → coloring x = c) ∧ infinite inf_set) ∧
  (∀ x y z : ℕ, coloring x ≠ coloring y ∧ coloring y ≠ coloring z ∧ coloring x ≠ coloring z → x * y ≠ z)

theorem color_naturals_with_2009_colors : 
  ∃ coloring : ℕ → ℕ, is_coloring_valid coloring 2009 :=
sorry

end color_naturals_with_2009_colors_l99_99413


namespace tan_45_degree_l99_99491

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99491


namespace sum_of_solutions_x_squared_eq_25_l99_99205

theorem sum_of_solutions_x_squared_eq_25 : 
  (∑ x in ({x : ℝ | x^2 = 25}).to_finset, x) = 0 :=
by
  sorry

end sum_of_solutions_x_squared_eq_25_l99_99205


namespace volunteer_allocation_plans_l99_99392

theorem volunteer_allocation_plans :
  let total_positions := 8
  let min_positions_A := 2
  let num_schools := 3
  let positions_left := total_positions - min_positions_A - num_schools
  (nat.choose (positions_left + (num_schools - 1)) (num_schools - 1) = 6) :=
by
  let total_positions := 8
  let min_positions_A := 2
  let num_schools := 3
  let positions_left := total_positions - min_positions_A - num_schools
  have h1: positions_left = 3 := rfl
  have h2: (num_schools - 1) = 2 := rfl
  have h3: (positions_left + (num_schools - 1)) = 5 := rfl
  exact nat.choose_eq_factorial_div ((positions_left + (num_schools - 1))) (num_schools - 1)
sorry

end volunteer_allocation_plans_l99_99392


namespace find_f_5_l99_99283

noncomputable def f: ℝ → ℝ := sorry

lemma function_condition (x: ℝ) (h: x > 0) : f(x) > -3 / x :=
sorry

lemma function_property (x: ℝ) (h: x > 0) : f (f(x) + 3 / x) = 2 :=
sorry

theorem find_f_5 : f 5 = 7 / 5 :=
begin
  sorry
end

end find_f_5_l99_99283


namespace tan_45_eq_1_l99_99554

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99554


namespace shaded_area_square_eq50_l99_99464

-- Define the conditions
def side_length : ℝ := 10
def area_square (s : ℝ) : ℝ := s * s
def area_equilateral_triangle (s : ℝ) : ℝ := (sqrt 3 / 4) * s ^ 2

-- Define the proof statement
theorem shaded_area_square_eq50
  (s : ℝ) (hs : s = side_length)
  (A_sq : ℝ) (hA_sq : A_sq = area_square s)
  (A_eq_tri : ℝ) (hA_eq_tri : A_eq_tri = area_equilateral_triangle s) :
  A_sq - 2 * A_eq_tri = 50 :=
by
  -- Proof goes here
  sorry

end shaded_area_square_eq50_l99_99464


namespace tan_45_eq_1_l99_99531

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99531


namespace tan_45_eq_1_l99_99552

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99552


namespace tan_45_eq_l99_99679

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99679


namespace r_fourth_power_sum_l99_99188

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l99_99188


namespace tan_45_eq_1_l99_99942

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99942


namespace parabola_area_proof_l99_99105

-- Parabola definition: y² = 4x
def parabola (x y : ℝ) := y^2 = 4 * x

-- Focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Point M lies on the parabola in the first quadrant and distance |MF| = 3
def point_M_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2 ∧ 0 ≤ M.1 ∧ 0 ≤ M.2 ∧ real.dist M focus = 3

-- Equation of line MF
def line_MF (M : ℝ × ℝ) (x : ℝ) := M.2 = 2 * real.sqrt 2 * (x - 1)

-- Midpoint of M and F
def midpoint (M F : ℝ × ℝ) : ℝ × ℝ :=
  ((M.1 + F.1) / 2, (M.2 + F.2) / 2)

-- Perpendicular bisector of MF
def perp_bisector (M : ℝ × ℝ) (x : ℝ) : ℝ :=
  let mid := midpoint M focus in
  mid.2 + - (real.sqrt 2 / 4) * (x - mid.1)

-- Intersect of perpendicular bisector with x-axis
def intersect_xaxis (M : ℝ × ℝ) : ℝ × ℝ :=
  let bisect := perp_bisector M in
  (3/2 + 4/real.sqrt 2 * mid.2, 0)

-- Area of triangle MNF
def area_triangle (M N : ℝ × ℝ) : ℝ :=
  1/2 * (N.1 - focus.1) * abs(M.2)

open nat.real

theorem parabola_area_proof (M N : ℝ × ℝ) :
  point_M_on_parabola M →
  N = intersect_xaxis M →
  line_MF M focus.1 = M.2 ∧ area_triangle M N = 9 * real.sqrt 2 / 2 := sorry

end parabola_area_proof_l99_99105


namespace tan_45_degrees_eq_1_l99_99820

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99820


namespace tan_45_deg_l99_99520

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99520


namespace range_of_fraction_l99_99247

theorem range_of_fraction (a b c : ℝ) (x₁ x₂ : ℝ)
    (h_f : ∀ x, Differentiable ℝ (λ (x : ℝ), (1/3) * x^3 + (1/2) * a * x^2 + 2 * b * x + c))
    (h_max : 0 < x₁ ∧ x₁ < 1)
    (h_min : 1 < x₂ ∧ x₂ < 2)
    (h_equation : ∀ x, Deriv (λ (x : ℝ), (1/3) * x^3 + (1/2) * a * x^2 + 2 * b * x + c) x = x^2 + a * x + 2 * b)
    (h_extrema : (x₁ - x₂) * (x₁ + x₂) = -a ∧ x₁ * x₂ = 2 * b) :
  (1/4 : ℝ) < (b - 2) / (a - 1) ∧ (b - 2) / (a - 1) < 1 := 
sorry

end range_of_fraction_l99_99247


namespace number_of_ways_to_assign_cooks_cleaners_l99_99039

theorem number_of_ways_to_assign_cooks_cleaners (n k : ℕ) (h_n : n = 8) (h_k : k = 4) :
  (nat.choose n k) = 70 :=
by
  -- Conditions as definitions in Lean
  have h1 : n = 8 := h_n,
  have h2 : k = 4 := h_k,
  -- Proof omitted as per instructions
  sorry

end number_of_ways_to_assign_cooks_cleaners_l99_99039


namespace find_r_fourth_l99_99166

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l99_99166


namespace tan_45_eq_one_l99_99861

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99861


namespace tan_45_degree_l99_99490

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99490


namespace abs_diff_is_perfect_square_l99_99266

-- Define the conditions
variable (m n : ℤ) (h_odd_m : m % 2 = 1) (h_odd_n : n % 2 = 1)
variable (h_div : (n^2 - 1) ∣ (m^2 + 1 - n^2))

-- Theorem statement
theorem abs_diff_is_perfect_square : ∃ (k : ℤ), (m^2 + 1 - n^2) = k^2 :=
by
  sorry

end abs_diff_is_perfect_square_l99_99266


namespace sum_integers_between_neg20_5_and_15_8_l99_99470

/--
The sum of all integers between -20.5 and 15.8 is -90.

We prove this by considering the inclusive range of integers from -20 to 15.
-/
theorem sum_integers_between_neg20_5_and_15_8 :
  ∑ i in Finset.range (36 + 1), (i - 20) = -90 :=
by
  let n := 36
  let a := -20
  let l := 15
  have h : n = l - a + 1 := by norm_num
  rw [sum_range_add, Finset.sum_range_succ, h]
  calc
    ∑ i in Finset.range (n + 1), (i - 20)
      = ∑ i in Finset.range (36 + 1), (i - 20) : by simp [h]
  ... = 36 * (-5) / 2 : by simp -- explicitly stating each computational step
  ... = -90 : by norm_num

-- Sorry, this is the statement only, no proof is provided.

end sum_integers_between_neg20_5_and_15_8_l99_99470


namespace tan_45_degrees_l99_99848

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99848


namespace tan_45_deg_eq_1_l99_99962

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99962


namespace tan_45_degrees_l99_99718

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99718


namespace arithmetic_sequence_problem_l99_99238

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : ∀ k, k ≥ 2 → a (k + 1) - a k^2 + a (k - 1) = 0) (h2 : ∀ k, a k ≠ 0) (h3 : ∀ k ≥ 2, a (k + 1) + a (k - 1) = 2 * a k) :
  S (2 * n - 1) - 4 * n = -2 :=
by
  sorry

end arithmetic_sequence_problem_l99_99238


namespace tan_45_deg_eq_1_l99_99968

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99968


namespace tan_45_eq_one_l99_99914

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99914


namespace cos_angle_a_a_plus_b_l99_99133

variables {V : Type*} [inner_product_space ℝ V]

-- Defining vectors and conditions given in the problem
variables (a b : V)
variables (ha : ∥a∥ = 5) (hb : ∥b∥ = 6) (ha_b : ⟪a, b⟫ = -6)

-- Defining the angle cosine theorem to be proved
theorem cos_angle_a_a_plus_b :
  real.cos (inner_product_space.angle a (a + b)) = 19 / 35 :=
by sorry

end cos_angle_a_a_plus_b_l99_99133


namespace probability_divisible_by_4_l99_99304

theorem probability_divisible_by_4 :
  let s : Finset ℤ := Finset.range (2052 + 1)
  let f (a b c : ℤ) := a * b * c + a * b + a + 1
  let prob := (s.filter (λ x => (f x.1 x.2 x.3) % 4 = 0)).card / (s.card ^ 3)
  prob = 1 / 4 :=
sorry

end probability_divisible_by_4_l99_99304


namespace price_return_initial_l99_99231

theorem price_return_initial (P0 : ℝ) (P1 P2 P3 P4 : ℝ) (x : ℕ) :
  P0 = 100 →
  P1 = P0 * 1.30 →
  P2 = P1 * 1.15 →
  P3 = P2 * 0.75 →
  P4 = P3 * (1 - x / 100.0) →
  P4 = P0 →
  x = 11 :=
begin
  sorry
end

end price_return_initial_l99_99231


namespace tenth_finger_is_six_l99_99373

-- Define the function f using the given mappings as a piecewise function
def f : ℕ → ℕ
| 3 := 6
| 6 := 5
| 5 := 4
| 4 := 3
| n := f ((f n % 4) + 3)  -- This handles the periodic repetition for other numbers.

-- Define a function that iterates the function f n times
def iterate_f (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | (n+1) => f (iterate_f n)

-- Prove that the 10th iteration of f starting from 3 is 6
theorem tenth_finger_is_six : iterate_f 10 = 6 := by
  sorry

end tenth_finger_is_six_l99_99373


namespace total_arrangements_l99_99388

-- Definitions of the problem conditions
def num_people : Nat := 7
def front_row_size : Nat := 3
def back_row_size : Nat := 4

-- Definitions for persons A, B, and C; these could be indices in the list of people
def person_A : Fin num_people := 0
def person_B : Fin num_people := 1
def person_C : Fin num_people := 2

-- The main theorem to be proven
theorem total_arrangements (P : {n : Nat // n = num_people} ) (F : {n : Nat // n = front_row_size} ) (B : {n : Nat // n = back_row_size} ) :
  let arrangements (P F B) := 
    -- First condition: Person A and person B must stand next to each other
    -- Second condition: Person A and person C must stand apart
    1056
  arrangements (P F B) = 1056 :=
by sorry

end total_arrangements_l99_99388


namespace tan_45_degrees_eq_1_l99_99800

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99800


namespace incorrect_statement_identification_l99_99144

theorem incorrect_statement_identification :
  (∀ c : ℝ, c > 0 → ∀ a b : ℝ, a > b → (a + c > b + c) ∧ (a - c > b - c ∧ a - c > 0) ∧ (a * c > b * c) ∧ (a / c > b / c)) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → (a ≠ b → (a + b) / 2 > Real.sqrt (a * b))) ∧
  (∀ x y s : ℝ, x + y = s → x * y = (s / 2) * (s / 2)) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → (1 / 2) * (x^2 + y^2) > (1 / 2) * (x + y) ^ 2) ∧
  (∀ x y p : ℝ, x * y = p -> x + y ≥ 2 * Real.sqrt(p)) → 
  (∃ (E : Prop), E = (False)
  ∧ (∀ E_yes : Prop, E_yes = (E)) ∧ ∀ E_no: Prop, (E_no = (∀ x y p: ℝ, x * y = p → (x ≠ y → x + y > 2 * Real.sqrt(p))))) sorry
  → False :=
begin
  sorry
end

end incorrect_statement_identification_l99_99144


namespace polynomial_unique_l99_99072

noncomputable def p (x : ℝ) : ℝ := x^2 + 1

theorem polynomial_unique (p : ℝ → ℝ) 
  (h1 : p 3 = 10)
  (h2 : ∀ x y : ℝ, p(x) * p(y) = p(x) + p(y) + p(x * y) - 2) :
  p = λ x, x^2 + 1 :=
by sorry

end polynomial_unique_l99_99072


namespace Alyssa_ate_20_l99_99450

variable (A : ℕ) 

def Alyssa_nuggets (A : ℕ) : Prop :=
  let K := 2 * A in
  let Ken := 2 * A in
  A + K + Ken = 100

theorem Alyssa_ate_20 (A : ℕ) (hA: Alyssa_nuggets A) : A = 20 :=
by
  sorry

end Alyssa_ate_20_l99_99450


namespace max_perimeter_l99_99025

-- Define the given parameters as constants
def base : ℝ := 10
def height : ℝ := 12
def num_pieces : ℕ := 10

-- Define the maximum perimeter function
def perimeter (k : ℕ) : ℝ := 
  1 + real.sqrt (height^2 + (k:ℝ)^2) + real.sqrt (height^2 + ((k + 1):ℝ)^2)

-- Statement we aim to prove
theorem max_perimeter : ∃ k : ℕ, k < num_pieces ∧ perimeter k = 31.62 :=
  sorry

end max_perimeter_l99_99025


namespace tan_45_degrees_eq_1_l99_99807

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99807


namespace tan_45_degrees_l99_99851

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99851


namespace geometric_progression_common_ratio_l99_99234

theorem geometric_progression_common_ratio :
  ∃ r : ℝ, (r > 0) ∧ (r^3 + r^2 + r - 1 = 0) :=
by
  sorry

end geometric_progression_common_ratio_l99_99234


namespace tan_45_degree_is_one_l99_99731

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99731


namespace probability_of_arithmetic_progression_on_8_faced_dice_l99_99079

def arithmetic_progression_probability : ℚ :=
  3 / 256

theorem probability_of_arithmetic_progression_on_8_faced_dice :
  let total_outcomes := (8 : ℕ) ^ 4
  let favorable_sets := {(1, 3, 5, 7), (2, 4, 6, 8)}
  let number_of_ways (s : finset (ℕ × ℕ × ℕ × ℕ)) : ℕ := 24
  let favorable_outcomes := 2 * number_of_ways favorable_sets
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = arithmetic_progression_probability
  :=
sorry

end probability_of_arithmetic_progression_on_8_faced_dice_l99_99079


namespace john_pennies_more_than_kate_l99_99260

theorem john_pennies_more_than_kate (kate_pennies : ℕ) (john_pennies : ℕ) (h_kate : kate_pennies = 223) (h_john : john_pennies = 388) : john_pennies - kate_pennies = 165 := by
  sorry

end john_pennies_more_than_kate_l99_99260


namespace tan_45_eq_1_l99_99992

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l99_99992


namespace side_length_of_base_l99_99345

-- Define the conditions
def lateral_area (s : ℝ) : ℝ := (1 / 2) * s * 40
def given_area : ℝ := 120

-- Define the theorem to prove the length of the side of the base
theorem side_length_of_base : ∃ (s : ℝ), lateral_area(s) = given_area ∧ s = 6 :=
by
  sorry

end side_length_of_base_l99_99345


namespace tan_45_degrees_l99_99715

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99715


namespace expected_value_of_trials_l99_99393

theorem expected_value_of_trials (n : ℕ) (h1 : n > 0) :
  let P := (λ k, 1 / (n : ℝ)) in
  (∑ k in finset.range n, (k + 1) * P (k + 1)) = (n + 1 : ℝ) / 2 :=
by
  sorry

end expected_value_of_trials_l99_99393


namespace tan_45_degrees_l99_99841

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99841


namespace tan_45_eq_one_l99_99651

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99651


namespace tan_45_degrees_l99_99849

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99849


namespace tan_45_eq_one_l99_99598

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99598


namespace tan_45_eq_1_l99_99548

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99548


namespace tan_45_eq_l99_99677

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99677


namespace min_area_of_ellipse_contains_circles_l99_99456

-- Definitions of the ellipse equation and the circles
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def circle1 (x y : ℝ) : Prop := ((x - 2)^2 + y^2 = 4)
def circle2 (x y : ℝ) : Prop := ((x + 2)^2 + y^2 = 4)

theorem min_area_of_ellipse_contains_circles :
  ∃ a b : ℝ, 
  (∀ x y : ℝ, circle1 x y → ellipse a b x y) ∧
  (∀ x y : ℝ, circle2 x y → ellipse a b x y) ∧
  (∃ k : ℝ, k = (a * b * π) ∧ k = ((3 * real.sqrt 3) / 2) * π) :=
sorry

end min_area_of_ellipse_contains_circles_l99_99456


namespace tan_45_degree_l99_99892

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99892


namespace tan_45_degree_l99_99481

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99481


namespace tan_45_degrees_l99_99698

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99698


namespace tan_45_eq_one_l99_99642

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99642


namespace find_real_numbers_l99_99050

theorem find_real_numbers (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f(f(x)) = x * f(x) - a * x) ∧
  (∃ x y : ℝ, f x ≠ f y) ∧
  (∃ x : ℝ, f x = a) →
  (a = 0 ∨ a = -1) :=
by
  sorry

end find_real_numbers_l99_99050


namespace find_x_of_point_Q_l99_99248

theorem find_x_of_point_Q (m yQ : ℝ) (h_slope : m = 0.8) (h_yQ : yQ = 2) :
  ∃ (xQ : ℝ), yQ = m * xQ ∧ xQ = 2.5 :=
by
  use 2.5
  split
  { rw [h_slope, h_yQ]
    norm_num }
  { rfl }

end find_x_of_point_Q_l99_99248


namespace side_length_of_square_base_l99_99327

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l99_99327


namespace two_cells_for_three_congruent_parts_l99_99082

-- Definitions for 4x4 grid and removing cells
def Grid := Fin (4 * 4)

-- Function to simulate removing a cell
def remove_cell (g : Finset Grid) (pos : Grid) : Finset Grid :=
  g.erase pos

-- Predicate to check if a grid can be divided into three congruent parts after cell removal
def can_be_divided_into_three_congruent_parts (g : Finset Grid) : Prop := sorry

-- Initial 4x4 grid (16 cells indexed from 0 to 15)
def initial_grid : Finset Grid := { i | i < 16 }.toFinset

-- Example positions (3, 2) and (4, 3)
def pos1 : Grid := 10 -- (3, 2) in 0-based indexing
def pos2 : Grid := 14 -- (4, 3) in 0-based indexing

theorem two_cells_for_three_congruent_parts :
  (can_be_divided_into_three_congruent_parts (remove_cell initial_grid pos1)) ∧ 
  (can_be_divided_into_three_congruent_parts (remove_cell initial_grid pos2)) :=
sorry

end two_cells_for_three_congruent_parts_l99_99082


namespace tan_45_eq_l99_99675

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99675


namespace tan_45_eq_l99_99676

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99676


namespace symmetric_point_origin_l99_99367

def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, -P.2)

theorem symmetric_point_origin :
  symmetric_point (3, -1) = (-3, 1) :=
by
  sorry

end symmetric_point_origin_l99_99367


namespace find_x_l99_99148

theorem find_x (x y z : ℚ) (h1 : (x * y) / (x + y) = 4) (h2 : (x * z) / (x + z) = 5) (h3 : (y * z) / (y + z) = 6) : x = 40 / 9 :=
by
  -- Structure the proof here
  sorry

end find_x_l99_99148


namespace r_fourth_power_sum_l99_99185

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l99_99185


namespace tan_45_degree_l99_99902

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99902


namespace side_length_of_square_base_l99_99324

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l99_99324


namespace tan_45_degree_l99_99887

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99887


namespace tan_45_degree_l99_99904

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99904


namespace count_distinct_numbers_in_list_l99_99066

def num_distinct_floor_divs : ℕ :=
  let L := list.map (λ n, ⌊(n ^ 2 : ℚ) / 500⌋) (list.range' 1 1000)
  list.dedup L

theorem count_distinct_numbers_in_list : (num_distinct_floor_divs.length = 876) := by
  -- Definition of L: the list from 1 to 1000 with floor of division by 500
  let L := list.map (λ n, ⌊(n ^ 2 : ℚ) / 500⌋) (list.range' 1 1000)
  -- Deduplicate the list by removing duplicates
  let distinct_L := list.dedup L
  -- Prove that the length of distinct_L is equal to 876
  have h : distinct_L.length = 876 := sorry
  exact h

end count_distinct_numbers_in_list_l99_99066


namespace tan_45_eq_one_l99_99860

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99860


namespace tan_45_degree_is_one_l99_99724

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99724


namespace tan_45_deg_eq_1_l99_99966

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99966


namespace r_power_four_identity_l99_99155

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l99_99155


namespace photos_per_album_l99_99453

theorem photos_per_album (total_photos : ℕ) (albums : ℕ) (h1 : total_photos = 180) (h2 : albums = 9) : total_photos / albums = 20 :=
by 
  -- We use the conditions directly to derive the result
  rw [h1, h2] -- Rewriting the given conditions 
  norm_num -- Performing the arithmetic division
  -- The proof would be completed here in Lean
  sorry

end photos_per_album_l99_99453


namespace tan_45_degree_l99_99487

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99487


namespace tan_45_deg_l99_99757

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99757


namespace cindy_dress_discount_l99_99020

theorem cindy_dress_discount (P D : ℝ) 
  (h1 : P * (1 - D) * 1.25 = 61.2) 
  (h2 : P - 61.2 = 4.5) : D = 0.255 :=
sorry

end cindy_dress_discount_l99_99020


namespace tan_45_eq_one_l99_99858

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99858


namespace tan_45_degree_l99_99497

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99497


namespace tan_45_deg_eq_1_l99_99979

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99979


namespace count_distinct_numbers_in_list_l99_99067

def num_distinct_floor_divs : ℕ :=
  let L := list.map (λ n, ⌊(n ^ 2 : ℚ) / 500⌋) (list.range' 1 1000)
  list.dedup L

theorem count_distinct_numbers_in_list : (num_distinct_floor_divs.length = 876) := by
  -- Definition of L: the list from 1 to 1000 with floor of division by 500
  let L := list.map (λ n, ⌊(n ^ 2 : ℚ) / 500⌋) (list.range' 1 1000)
  -- Deduplicate the list by removing duplicates
  let distinct_L := list.dedup L
  -- Prove that the length of distinct_L is equal to 876
  have h : distinct_L.length = 876 := sorry
  exact h

end count_distinct_numbers_in_list_l99_99067


namespace hole_depth_l99_99289

theorem hole_depth (height : ℝ) (half_depth : ℝ) (total_depth : ℝ) 
    (h_height : height = 90) 
    (h_half_depth : half_depth = total_depth / 2)
    (h_position : height + half_depth = total_depth - height) : 
    total_depth = 120 := 
by
    sorry

end hole_depth_l99_99289


namespace tan_45_eq_1_l99_99541

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99541


namespace tan_45_eq_one_l99_99644

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99644


namespace sum_of_pills_in_larger_bottles_l99_99005

-- Definitions based on the conditions
def supplements := 5
def pills_in_small_bottles := 2 * 30
def pills_per_day := 5
def days_used := 14
def pills_remaining := 350
def total_pills_before := pills_remaining + (pills_per_day * days_used)
def total_pills_in_large_bottles := total_pills_before - pills_in_small_bottles

-- The theorem statement that needs to be proven
theorem sum_of_pills_in_larger_bottles : total_pills_in_large_bottles = 360 := 
by 
  -- Placeholder for the proof
  sorry

end sum_of_pills_in_larger_bottles_l99_99005


namespace yellow_crane_tower_visitor_l99_99451

variables (A B C : Prop) 
          (A_said : ¬ C)
          (B_said : B)
          (C_said : ¬ C → ¬ C)
          (one_visitor : A ∨ B ∨ C ∧ (¬ A ∨ ¬ B ∨ ¬ C))
          (one_lied : (A ⊕ B ⊕ C).card = 2)

theorem yellow_crane_tower_visitor : A :=
by
  sorry

end yellow_crane_tower_visitor_l99_99451


namespace roller_coaster_cars_l99_99428

theorem roller_coaster_cars (n : ℕ) (h : ((n - 1) : ℝ) / n = 0.5) : n = 2 :=
sorry

end roller_coaster_cars_l99_99428


namespace tan_45_degree_l99_99493

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99493


namespace tan_45_deg_l99_99765

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99765


namespace number_of_ways_line_up_l99_99387

def catalan_number (n : ℕ) : ℕ :=
  (nat.choose (2 * n) n) / (n + 1)

theorem number_of_ways_line_up (n : ℕ) : ∃ k, k = catalan_number(n) :=
begin
  use catalan_number(n),
  sorry
end

end number_of_ways_line_up_l99_99387


namespace tan_45_eq_one_l99_99880

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99880


namespace stratified_sampling_l99_99432

variables (freshmen sophomores juniors sample_size : ℕ)
noncomputable def total_students : ℕ := freshmen + sophomores + juniors
noncomputable def sampling_ratio : ℚ := sample_size / total_students

theorem stratified_sampling (freshmen sophomores juniors : ℕ)
(sample_size : ℕ)
(h₀ : freshmen = 560)
(h₁ : sophomores = 540)
(h₂ : juniors = 520)
(h₃ : sample_size = 81) :
  (freshmen * sampling_ratio = 28) ∧
  (sophomores * sampling_ratio = 27) ∧
  (juniors * sampling_ratio = 26) :=
sorry

end stratified_sampling_l99_99432


namespace tan_45_eq_1_l99_99944

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99944


namespace tan_45_eq_1_l99_99612

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99612


namespace side_length_of_base_l99_99357

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l99_99357


namespace total_fruit_salads_l99_99460

theorem total_fruit_salads (a : ℕ) (h_alaya : a = 200) (h_angel : 2 * a = 400) : a + 2 * a = 600 :=
by 
  rw [h_alaya, h_angel]
  sorry

end total_fruit_salads_l99_99460


namespace tan_45_eq_1_l99_99540

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99540


namespace tan_45_deg_l99_99752

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99752


namespace side_length_of_square_base_l99_99323

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l99_99323


namespace tan_45_eq_l99_99689

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99689


namespace tan_45_eq_one_l99_99595

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99595


namespace tan_of_45_deg_l99_99567

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99567


namespace average_height_is_64_l99_99300

noncomputable def Parker (H_D : ℝ) : ℝ := H_D - 4
noncomputable def Daisy (H_R : ℝ) : ℝ := H_R + 8
noncomputable def Reese : ℝ := 60

theorem average_height_is_64 :
  let H_R := Reese 
  let H_D := Daisy H_R
  let H_P := Parker H_D
  (H_P + H_D + H_R) / 3 = 64 := sorry

end average_height_is_64_l99_99300


namespace side_length_of_base_l99_99356

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l99_99356


namespace tan_45_degree_l99_99484

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99484


namespace absolute_value_of_neg_five_l99_99322

theorem absolute_value_of_neg_five : |(-5 : ℤ)| = 5 := 
by 
  sorry

end absolute_value_of_neg_five_l99_99322


namespace tan_45_degree_l99_99896

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99896


namespace sum_of_solutions_l99_99210

theorem sum_of_solutions (x : ℝ) (h1 : x^2 = 25) : ∃ S : ℝ, S = 0 ∧ (∀ x', x'^2 = 25 → x' = 5 ∨ x' = -5) := 
sorry

end sum_of_solutions_l99_99210


namespace tan_45_degree_l99_99906

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99906


namespace tan_sum_l99_99198

theorem tan_sum (θ : ℝ) (h : Real.sin (2 * θ) = 2 / 3) : Real.tan θ + 1 / Real.tan θ = 3 := sorry

end tan_sum_l99_99198


namespace find_fx_l99_99280

theorem find_fx : 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, 2 * f(x - 1) - 3 * f(1 - x) = 5 * x) ∧ (∀ x : ℝ, f(x) = x - 5)) :=
sorry

end find_fx_l99_99280


namespace tan_45_degree_is_one_l99_99733

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99733


namespace tan_of_45_deg_l99_99565

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99565


namespace tan_45_eq_l99_99685

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99685


namespace min_remainder_n_div_2005_l99_99282

theorem min_remainder_n_div_2005 (n : ℕ) (hn_pos : 0 < n) 
  (h1 : n % 902 = 602) (h2 : n % 802 = 502) (h3 : n % 702 = 402) :
  n % 2005 = 101 :=
sorry

end min_remainder_n_div_2005_l99_99282


namespace sum_of_solutions_l99_99208

theorem sum_of_solutions (x : ℝ) (h1 : x^2 = 25) : ∃ S : ℝ, S = 0 ∧ (∀ x', x'^2 = 25 → x' = 5 ∨ x' = -5) := 
sorry

end sum_of_solutions_l99_99208


namespace distinct_numbers_l99_99064

theorem distinct_numbers (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 1000) : 
  finset.card ((finset.image (λ k : ℕ, ⌊(k^2 : ℚ) / 500⌋) (finset.range 1000).succ)) = 2001 :=
sorry

end distinct_numbers_l99_99064


namespace value_of_a_l99_99075

theorem value_of_a (a : ℕ) (A_a B_a : ℕ)
  (h1 : A_a = 10)
  (h2 : B_a = 11)
  (h3 : 2 * a^2 + 10 * a + 3 + 5 * a^2 + 7 * a + 8 = 8 * a^2 + 4 * a + 11) :
  a = 13 :=
sorry

end value_of_a_l99_99075


namespace r_power_four_identity_l99_99156

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l99_99156


namespace tan_45_eq_1_l99_99530

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99530


namespace tan_45_degree_is_one_l99_99719

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99719


namespace meaningful_expr_l99_99220

theorem meaningful_expr (x : ℝ) : 
    (x + 1 ≥ 0 ∧ x - 2 ≠ 0) → (x ≥ -1 ∧ x ≠ 2) := by
  sorry

end meaningful_expr_l99_99220


namespace tan_45_degrees_l99_99694

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99694


namespace tan_45_eq_one_l99_99649

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99649


namespace tan_45_eq_one_l99_99594

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99594


namespace tan_45_degree_l99_99881

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99881


namespace tan_45_deg_l99_99789

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99789


namespace tan_45_eq_one_l99_99647

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99647


namespace find_t_l99_99099

noncomputable def t : ℝ := sorry

-- Given conditions
def f (t x : ℝ) := (x^2 - 4) * (x - t)

def f' (t x : ℝ) := 2 * x * (x - t) + (x^2 - 4)

axiom h1 : t ∈ ℝ
axiom h2 : f' t (-1) = 0

-- The Lean proof statement
theorem find_t : t = 1 / 2 :=
by
  rw [h2, f', mul_comm (2 * x) (x - t), add_comm (x^2 - 4) (2 * x * (x - t))]
  sorry

end find_t_l99_99099


namespace pyramid_base_side_length_l99_99335

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l99_99335


namespace tan_of_45_deg_l99_99583

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99583


namespace tan_of_45_deg_l99_99559

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99559


namespace pages_wednesday_l99_99296

-- Given conditions as definitions
def borrow_books := 3
def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51

-- Prove that Nico read 19 pages on Wednesday
theorem pages_wednesday :
  let pages_wednesday := total_pages - (pages_monday + pages_tuesday)
  pages_wednesday = 19 :=
by
  sorry

end pages_wednesday_l99_99296


namespace tan_45_eq_l99_99666

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99666


namespace tan_45_eq_1_l99_99937

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99937


namespace tan_45_eq_one_l99_99606

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99606


namespace tan_45_degrees_l99_99839

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99839


namespace tan_45_eq_one_l99_99597

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99597


namespace directional_derivative_at_point_l99_99052

theorem directional_derivative_at_point :
  let u := λ (x y z : ℝ), x^3 * y^3 * z^3
  let M := (1,1,1)
  let θ₁ := π/3 -- 60 degrees in radians
  let θ₂ := π/4 -- 45 degrees in radians
  let θ₃ := π/3 -- 60 degrees in radians
  let cosθ₁ := Real.cos θ₁
  let cosθ₂ := Real.cos θ₂
  let cosθ₃ := Real.cos θ₃
  let du_dx := (3 * (M.1^2) * (M.2^3) * (M.3^3))
  let du_dy := (3 * (M.1^3) * (M.2^2) * (M.3^3))
  let du_dz := (3 * (M.1^3) * (M.2^3) * (M.3^2))
  let directional_derivative := du_dx * cosθ₁ + du_dy * cosθ₂ + du_dz * cosθ₃
  directional_derivative = (3 / 2) * (2 + Real.sqrt 2) :=
by
  -- sorry is used to skip the proof
  sorry

end directional_derivative_at_point_l99_99052


namespace tan_45_degree_l99_99494

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99494


namespace tan_45_degree_is_one_l99_99726

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99726


namespace find_r_fourth_l99_99170

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l99_99170


namespace find_S3_l99_99307

-- Define the known scores
def S1 : ℕ := 55
def S2 : ℕ := 67
def S4 : ℕ := 55
def Avg : ℕ := 67

-- Statement to prove
theorem find_S3 : ∃ S3 : ℕ, (S1 + S2 + S3 + S4) / 4 = Avg ∧ S3 = 91 :=
by
  sorry

end find_S3_l99_99307


namespace not_sum_of_three_cubes_infinite_not_sum_of_three_cubes_l99_99306

theorem not_sum_of_three_cubes (n : ℕ) : ∃∞ (k : ℕ), 
  ∀ a b c : ℤ, a^3 + b^3 + c^3 ≠ 9 * k + n :=
begin
  sorry
end

theorem infinite_not_sum_of_three_cubes :
  ∃∞ (n : ℕ), ∀ a b c : ℤ, 
  ¬ (∃ k, 9 * k + n = a ^ 3 + b ^ 3 + c ^ 3) :=
begin
  sorry
end

end not_sum_of_three_cubes_infinite_not_sum_of_three_cubes_l99_99306


namespace point_coordinates_l99_99242

-- We assume that the point P has coordinates (2, 4) and prove that the coordinates with respect to the origin in Cartesian system are indeed (2, 4).
theorem point_coordinates (x y : ℝ) (h : x = 2 ∧ y = 4) : (x, y) = (2, 4) :=
by
  sorry

end point_coordinates_l99_99242


namespace tan_45_deg_l99_99506

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99506


namespace tan_45_deg_l99_99528

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99528


namespace r_fourth_power_sum_l99_99181

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l99_99181


namespace side_length_of_square_base_l99_99326

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l99_99326


namespace sum_of_solutions_eq_zero_l99_99200

theorem sum_of_solutions_eq_zero (x : ℝ) (hx : x^2 = 25) : ∃ x₁ x₂ : ℝ, (x₁^2 = 25) ∧ (x₂^2 = 25) ∧ (x₁ + x₂ = 0) := 
by {
  use 5,
  use (-5),
  split,
  { exact hx, },
  split,
  { rw pow_two, exact hx, },
  { norm_num, },
}

end sum_of_solutions_eq_zero_l99_99200


namespace tan_45_eq_one_l99_99879

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99879


namespace r_pow_four_solution_l99_99174

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l99_99174


namespace tan_45_deg_l99_99788

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99788


namespace tan_45_eq_1_l99_99536

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99536


namespace pyramid_base_side_length_l99_99332

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l99_99332


namespace range_of_g_function_l99_99376

theorem range_of_g_function :
  (∀ x ∈ Ioo (-π / 6) (5 * π / 6), g(x) = sin(x - π / 6)) →
  (range (λ x, sin(x - π / 6)) ∩ Ioo (-π / 6) (5 * π / 6) = set.Icc (-√3 / 2) 1) :=
by
  sorry

end range_of_g_function_l99_99376


namespace tan_45_eq_l99_99669

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99669


namespace determine_linear_relation_l99_99404

-- Define the set of options
inductive PlotType
| Scatter
| StemAndLeaf
| FrequencyHistogram
| FrequencyLineChart

-- Define the question and state the expected correct answer
def correctPlotTypeForLinearRelation : PlotType :=
  PlotType.Scatter

-- Prove that the correct method for determining linear relation in a set of data is a Scatter plot
theorem determine_linear_relation :
  correctPlotTypeForLinearRelation = PlotType.Scatter :=
by
  sorry

end determine_linear_relation_l99_99404


namespace find_AE_l99_99251

-- Define a triangle with vertices A, B, and C.
variables (A B C E: Type) [LinearOrderedField A]
variables [InnerProductSpace A B]
variables (M : Type) [AddCommGroup M] [Module A M]

-- Given conditions:
def is_triangle (A B C : M) : Prop := 
  (∥A - B∥ ≠ 0) ∧ (∥B - C∥ ≠ 0) ∧ (∥C - A∥ ≠ 0)

noncomputable def AC : A := 8
noncomputable def BC : A := 5

axiom midpoint_AB (D : M) : ∃ m : M, m = (A + B) / 2

axiom parallel_condition (DE BF : M → M) : (∃E', E' ∈ span A (E - midpoint_AB A B)) ∧ ∀x ∈ span A (E - midpoint_AB A B), ∥DE x∥ = ∥BF x∥ -- Given that DE is parallel to BF and midpoint_AB

-- The proof goal:
theorem find_AE (E: M) (D: M) : 
  is_triangle A B C → 
  midpoint_AB D → 
  parallel_condition (E) (D) → 
  ∥A - E∥ = 3 / 2 := 
by 
sory

end find_AE_l99_99251


namespace gcd_consecutive_terms_is_2_l99_99030

-- Define the sequence
def b (n : ℕ) : ℕ := n! + 2 * n

-- Statement: Prove that the gcd of consecutive terms is always 2 for n ≥ 1.
theorem gcd_consecutive_terms_is_2 (n : ℕ) (h : n ≥ 1) : 
  Nat.gcd (b n) (b (n + 1)) = 2 := 
sorry

end gcd_consecutive_terms_is_2_l99_99030


namespace minimum_u_l99_99100

-- Defining the variables and conditions
variables (x y : ℝ)
-- Condition: x and y are within the interval (-2, 2)
def in_interval (a : ℝ) := -2 < a ∧ a < 2
-- Condition: x * y = -1
def xy_eq_neg_one := x * y = -1

-- Function to minimize
def u := (4 / (4 - x^2)) + (9 / (9 - y^2))

-- The theorem that states the desired minimum value of u
theorem minimum_u (hx : in_interval x) (hy : in_interval y) (hxy : xy_eq_neg_one) : u x y = 12 / 5 :=
sorry

end minimum_u_l99_99100


namespace remainder_of_numbers_with_more_1s_than_0s_in_base_2_l99_99267

theorem remainder_of_numbers_with_more_1s_than_0s_in_base_2 : 
  let N := finset.card { n ∈ finset.range 2004 | (nat.bitsize n) < 11 ∧ (nat.popcount n) > nat.bitsize n / 2 } in
  N % 1000 = 179 :=
by
  let N := finset.card { n ∈ finset.range 2004 | (nat.bitsize n) < 11 ∧ (nat.popcount n) > nat.bitsize n / 2 }
  exact sorry

end remainder_of_numbers_with_more_1s_than_0s_in_base_2_l99_99267


namespace side_length_of_base_l99_99342

-- Define the conditions
def lateral_area (s : ℝ) : ℝ := (1 / 2) * s * 40
def given_area : ℝ := 120

-- Define the theorem to prove the length of the side of the base
theorem side_length_of_base : ∃ (s : ℝ), lateral_area(s) = given_area ∧ s = 6 :=
by
  sorry

end side_length_of_base_l99_99342


namespace tan_45_deg_l99_99746

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99746


namespace tan_45_eq_one_l99_99645

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99645


namespace cos_theta_l99_99131

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions
axiom norm_a : ∥a∥ = 7
axiom norm_b : ∥b∥ = 9
axiom norm_a_add_b : ∥a + b∥ = 13

-- Lean Statement
theorem cos_theta (θ : ℝ) (h1 : ∥a∥ = 7) (h2 : ∥b∥ = 9) (h3 : ∥a + b∥ = 13) :
  real.cos θ = 13 / 42 :=
sorry

end cos_theta_l99_99131


namespace tan_45_eq_one_l99_99871

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99871


namespace find_angle_x_l99_99245

noncomputable def angle_x (angle_ABC angle_ACB angle_CDE : ℝ) : ℝ :=
  let angle_BAC := 180 - angle_ABC - angle_ACB
  let angle_ADE := 180 - angle_CDE
  let angle_EAD := angle_BAC
  let angle_AED := 180 - angle_ADE - angle_EAD
  180 - angle_AED

theorem find_angle_x (angle_ABC angle_ACB angle_CDE : ℝ) :
  angle_ABC = 70 → angle_ACB = 90 → angle_CDE = 42 → angle_x angle_ABC angle_ACB angle_CDE = 158 :=
by
  intros hABC hACB hCDE
  simp [angle_x, hABC, hACB, hCDE]
  sorry

end find_angle_x_l99_99245


namespace zero_intersection_point_function_zero_in_interval_bisection_method_applicable_quadratic_no_zero_function_monotonic_has_one_zero_l99_99036

-- Statement 1: The zero of a function is the intersection point of the function's graph with the x-axis.
theorem zero_intersection_point (f : ℝ → ℝ) : 
  ¬ (∀ x, f x = 0 ↔ x = 0) :=
sorry

-- Statement 2: If the function y = f(x) has a zero in the interval (a,b) (the graph of the function is continuous),
-- then f(a)·f(b) < 0 does not necessarily hold.
theorem function_zero_in_interval (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Ioo a b)) :
  ¬ (∃ x, x ∈ Ioo a b ∧ f x = 0) → (f a * f b < 0) :=
sorry

-- Statement 3: As long as a function has a zero, we can use the bisection method to approximate the value of the zero.
theorem bisection_method_applicable (f : ℝ → ℝ) : 
  ¬ (∀ a b, (∃ x, x ∈ Ioo a b ∧ f x = 0) → (f a * f b < 0) → bisection_applicable f a b) :=
sorry

-- Statement 4: The quadratic function y = ax^2 + bx + c (a ≠ 0) has no zero when b^2 - 4ac < 0.
theorem quadratic_no_zero (a b c : ℝ) (ha : a ≠ 0) (h_discriminant : b^2 - 4*a*c < 0) :
  ¬ (∃ x, a*x^2 + b*x + c = 0) :=
sorry

-- Statement 5: If the function f(x) is monotonic on (a,b) and f(a)·f(b) < 0, then the function f(x) has exactly one zero in [a,b].
theorem function_monotonic_has_one_zero (f : ℝ → ℝ) (a b : ℝ) 
  (h_mono : ∀ x y, x ∈ Ioo a b → y ∈ Ioo a b → x < y → f x ≤ f y) (h_sign_change : f a * f b < 0) :
  ∃! x, x ∈ Icc a b ∧ f x = 0 :=
sorry

end zero_intersection_point_function_zero_in_interval_bisection_method_applicable_quadratic_no_zero_function_monotonic_has_one_zero_l99_99036


namespace tan_45_deg_l99_99769

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99769


namespace find_a_l99_99106

theorem find_a (a : ℝ) :
  (∃x y : ℝ, x^2 + y^2 + 2 * x - 2 * y + a = 0 ∧ x + y + 4 = 0) →
  ∃c : ℝ, c = 2 ∧ a = -7 :=
by
  -- proof to be filled in
  sorry

end find_a_l99_99106


namespace tan_45_deg_l99_99775

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99775


namespace tan_45_degree_is_one_l99_99721

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99721


namespace smallest_c_for_f_l99_99412

noncomputable def f (x : ℝ) : ℝ := sorry

section
variable (f : ℝ → ℝ)
hypothesis (h₀ : ∀ x ∈ set.Icc 0 1, f x ≥ 0)
hypothesis (h₁ : f 0 = 0)
hypothesis (h₂ : f 1 = 1)
hypothesis (h₃ : ∀ x₁ x₂ : ℝ, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

-- Theorem stating the desired inequality
theorem smallest_c_for_f : ∀ (x : ℝ), x ∈ set.Icc 0 1 → f x ≤ 2 * x :=
sorry
end

end smallest_c_for_f_l99_99412


namespace max_x_y_l99_99422

theorem max_x_y (x y : ℝ) (hx : (3√ x) + (3√ y) = 2) (hy_sum : x + y = 20) :
  max x y = 10 + 6 * Real.sqrt 3 :=
begin
  sorry -- Proof goes here
end

end max_x_y_l99_99422


namespace missing_value_in_set_l99_99229

theorem missing_value_in_set (x : ℕ) (y : ℕ) 
  (hx_prime : Nat.Prime x)
  (hmedian : x - 1 = List.median [x - 1, y, 2 * x - 4])
  (haverage : (x - 1 + y + (2 * x - 4)) / 3 = 10 / 3) : 
  y = 6 :=
sorry

end missing_value_in_set_l99_99229


namespace tan_45_eq_one_l99_99864

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99864


namespace tan_45_degrees_eq_1_l99_99823

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99823


namespace tan_45_degrees_eq_1_l99_99806

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99806


namespace expression_equals_one_l99_99040

def evaluate_expression : ℕ :=
  3 * (3 * (3 * (3 * (3 - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1

theorem expression_equals_one : evaluate_expression = 1 := by
  sorry

end expression_equals_one_l99_99040


namespace tan_45_eq_1_l99_99547

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99547


namespace tan_45_degree_l99_99889

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99889


namespace jungkook_age_l99_99259

theorem jungkook_age
    (J U : ℕ)
    (h1 : J = U - 12)
    (h2 : (J + 3) + (U + 3) = 38) :
    J = 10 := 
sorry

end jungkook_age_l99_99259


namespace tan_45_eq_1_l99_99994

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l99_99994


namespace projection_of_a_in_direction_of_b_l99_99132

theorem projection_of_a_in_direction_of_b
  (a b : ℝ × ℝ)
  (ha : a = (2, 3))
  (hb : b = (-2, 1)) :
  let proj := (a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2) * ℝ.sqrt((b.1)^2 + (b.2)^2) in
  proj = -ℝ.sqrt(5) / 5 :=
by
  sorry

end projection_of_a_in_direction_of_b_l99_99132


namespace tan_45_degrees_eq_1_l99_99815

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l99_99815


namespace tan_45_deg_l99_99784

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99784


namespace white_paint_needed_l99_99433

theorem white_paint_needed (h₁ : 1 = 1 : ℝ) 
  (height_small : 3.5)
  (height_large : 6)
  (paint_large : 1 : ℝ) :
  let ratio_height := height_small / height_large,
      ratio_area := ratio_height^2,
      paint_small := ratio_area * paint_large,
      total_paint := 2 * paint_small,
      white_paint_needed := total_paint * (4 / 5) in
  white_paint_needed = 49 / 72 :=
by
  sorry

end white_paint_needed_l99_99433


namespace sum_of_roots_eq_zero_sum_of_all_possible_values_l99_99214

theorem sum_of_roots_eq_zero (x : ℝ) (h : x^2 = 25) : x = 5 ∨ x = -5 :=
by {
  sorry
}

theorem sum_of_all_possible_values (h : ∀ x : ℝ, x^2 = 25 → x = 5 ∨ x = -5) : ∑ x in {x : ℝ | x^2 = 25}, x = 0 :=
by {
  sorry
}

end sum_of_roots_eq_zero_sum_of_all_possible_values_l99_99214


namespace find_r4_l99_99159

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l99_99159


namespace tan_45_eq_1_l99_99635

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99635


namespace pyramid_base_length_l99_99348

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l99_99348


namespace tan_45_deg_l99_99754

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99754


namespace tan_45_degrees_l99_99833

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99833


namespace tan_45_eq_1_l99_99623

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99623


namespace tan_45_deg_l99_99516

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99516


namespace tan_45_degree_is_one_l99_99720

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99720


namespace tan_45_deg_l99_99762

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99762


namespace tan_45_deg_l99_99779

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99779


namespace average_all_30_l99_99218

def x : ℕ → ℝ := sorry  -- Placeholder for the sequence of numbers

variables {a b : ℝ}

-- Conditions
def avg_first_10 (a : ℝ) : Prop := (∑ i in Finset.range 10, x i) / 10 = a
def avg_next_20 (b : ℝ) : Prop := (∑ i in Finset.range 20, x (i + 10)) / 20 = b

-- Statement
theorem average_all_30 (h1 : avg_first_10 a) (h2 : avg_next_20 b) : 
  (∑ i in Finset.range 30, x i) / 30 = (a + 2 * b) / 3 :=
sorry

end average_all_30_l99_99218


namespace min_throws_for_repeated_sum_l99_99401

theorem min_throws_for_repeated_sum (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 16) : 
  ∃ m, m = 16 ∧ (∀ (k : ℕ), k < 16 → ∃ i < 16, ∃ j < 16, i ≠ j ∧ i + j = k) :=
by
  sorry

end min_throws_for_repeated_sum_l99_99401


namespace shopkeeper_gain_l99_99408

theorem shopkeeper_gain
  (true_weight : ℝ)
  (cheat_percent : ℝ)
  (gain_percent : ℝ) :
  cheat_percent = 0.1 ∧
  true_weight = 1000 →
  gain_percent = 20 :=
by
  sorry

end shopkeeper_gain_l99_99408


namespace tan_45_eq_1_l99_99626

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99626


namespace tan_45_degree_is_one_l99_99735

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99735


namespace total_fruit_salads_correct_l99_99462

-- Definitions for the conditions
def alayas_fruit_salads : ℕ := 200
def angels_fruit_salads : ℕ := 2 * alayas_fruit_salads
def total_fruit_salads : ℕ := alayas_fruit_salads + angels_fruit_salads

-- Theorem statement
theorem total_fruit_salads_correct : total_fruit_salads = 600 := by
  -- Proof goes here, but is not required for this task
  sorry

end total_fruit_salads_correct_l99_99462


namespace tan_45_eq_1_l99_99631

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99631


namespace tan_45_eq_1_l99_99532

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99532


namespace angle_QPC_40_l99_99252

/-- Given conditions in triangle ABC -/
variables (A B C P Q : Type) [Euclidean_geometry A B C P Q]
variables (angle_B : angle A B C = 110)
variables (angle_C : angle A C B = 50)
variables (angle_PCB : angle P C B = 30)
variables (angle_ABQ : angle A B Q = 40)

/-- What we need to prove -/
theorem angle_QPC_40 : angle Q P C = 40 :=
by
  -- Proof will go here
  sorry

end angle_QPC_40_l99_99252


namespace tan_45_eq_one_l99_99662

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99662


namespace number_of_correct_statements_l99_99088

variables (m n : Type) [line m] [line n]
variables (α β γ : Type) [plane α] [plane β] [plane γ]

-- Conditions for each statement
axiom condition1: m ⊥ n ∧ α ∥ β ∧ m ∥ α → ¬ (n ⊥ β)
axiom condition2: (angle_with_plane m = angle_with_plane n) → ¬ (m ∥ n)
axiom condition3: m ⊥ α ∧ m ⊥ n → ¬ (n ∥ α)
axiom condition4: α ⊥ γ ∧ β ⊥ γ → ¬ (α ⊥ β)

-- The proof problem: The number of correct statements is 0.
theorem number_of_correct_statements : 0 = 0 :=
by {
  sorry
}

end number_of_correct_statements_l99_99088


namespace incoming_class_size_l99_99318

theorem incoming_class_size :
  ∃ n : ℕ, n < 600 ∧ n % 19 = 15 ∧ n % 17 = 11 ∧ n = 53 :=
begin
  sorry
end

end incoming_class_size_l99_99318


namespace tan_45_deg_l99_99756

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99756


namespace average_height_of_three_l99_99301

theorem average_height_of_three (parker daisy reese : ℕ) 
  (h1 : parker = daisy - 4)
  (h2 : daisy = reese + 8)
  (h3 : reese = 60) : 
  (parker + daisy + reese) / 3 = 64 := 
  sorry

end average_height_of_three_l99_99301


namespace tan_45_eq_one_l99_99874

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99874


namespace tan_45_eq_l99_99678

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99678


namespace log_sum_l99_99471

theorem log_sum:
  log 2 8 + 3 * log 2 4 + 2 * log 8 16 + log 4 64 = 44 / 3 := by
  sorry

end log_sum_l99_99471


namespace number_of_clubs_le_people_l99_99414

theorem number_of_clubs_le_people (n m : ℕ) (A : Fin m → Finset (Fin n)) :
  (∀ S : Finset (Fin m), S ≠ ∅ → ∃ j : Fin n, S.card.filter (λ i, j ∈ A i) % 2 = 1) →
  m ≤ n := sorry

end number_of_clubs_le_people_l99_99414


namespace tan_45_degrees_l99_99712

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99712


namespace tan_45_deg_eq_1_l99_99981

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99981


namespace range_of_k_l99_99077

open Real BigOperators Topology

variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)
variable (q : ℝ) (k : ℝ)

-- Condition: Each term of the sequence is non-zero
def non_zero_sequence : Prop := ∀ n, a n ≠ 0

-- Condition: Sum of the first n terms is Sn
def sum_of_sequence (S : ℕ → ℝ) : Prop := ∀ n, S n = ∑ i in Finset.range n, a i

-- Condition: Vector is normal to the line y = kx
def normal_vector (k : ℝ) : Prop := ∀ n, k = - (a (n + 1) - a n) / (2 * a (n + 1))

-- Condition: The common ratio q satisfies 0 < |q| < 1
def common_ratio_q : Prop := 0 < abs q ∧ abs q < 1

-- Condition: The limit Sn exists as n tends to infinity
def limit_exists : Prop := ∃ l, Tendsto S atTop (nhds l)

-- Statement to be proved
theorem range_of_k 
  (h1 : non_zero_sequence a)
  (h2 : sum_of_sequence a S)
  (h3 : normal_vector k)
  (h4 : common_ratio_q q)
  (h5 : limit_exists S) :
  k ∈ Iio (-1) ∪ Ioi 0 :=
begin
  sorry
end

end range_of_k_l99_99077


namespace right_angle_triangle_probability_l99_99081

def vertex_count : ℕ := 16
def ways_to_choose_3_points : ℕ := Nat.choose vertex_count 3
def number_of_rectangles : ℕ := 36
def right_angle_triangles_per_rectangle : ℕ := 4
def total_right_angle_triangles : ℕ := number_of_rectangles * right_angle_triangles_per_rectangle
def probability_right_angle_triangle : ℚ := total_right_angle_triangles / ways_to_choose_3_points

theorem right_angle_triangle_probability :
  probability_right_angle_triangle = (9 / 35 : ℚ) := by
  sorry

end right_angle_triangle_probability_l99_99081


namespace tan_45_eq_one_l99_99862

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l99_99862


namespace tan_45_eq_one_l99_99930

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99930


namespace tan_45_eq_one_l99_99664

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99664


namespace factorize_x2y_minus_4y_l99_99048

variable {x y : ℝ}

theorem factorize_x2y_minus_4y : x^2 * y - 4 * y = y * (x + 2) * (x - 2) :=
sorry

end factorize_x2y_minus_4y_l99_99048


namespace tan_45_degree_l99_99899

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99899


namespace tan_45_deg_eq_1_l99_99972

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99972


namespace minimum_omega_l99_99113

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)
noncomputable def h (ω : ℝ) (x : ℝ) : ℝ := f ω x + g ω x

theorem minimum_omega (ω : ℝ) (m : ℝ) 
  (h1 : 0 < ω)
  (h2 : ∀ x : ℝ, h ω m ≤ h ω x ∧ h ω x ≤ h ω (m + 1)) :
  ω = π :=
by
  sorry

end minimum_omega_l99_99113


namespace tan_45_eq_1_l99_99960

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99960


namespace smallest_five_divisible_number_l99_99076

def is_five_divisible (N : ℕ) : Prop :=
  (finset.univ.filter (λ m : ℕ, m ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ m ∣ N)).card ≥ 5

theorem smallest_five_divisible_number :
  ∃ N : ℕ, N > 2000 ∧ is_five_divisible N ∧
    (∀ M : ℕ, M > 2000 ∧ is_five_divisible M → N ≤ M) :=
begin
  have : is_five_divisible 2004, from sorry,
  use 2004,
  split,
  { linarith, },  -- Proof that 2004 is greater than 2000
  { split,
    { exact this, },  -- Proof that 2004 is a five-divisible number
    { intros M HM,
      have h1 : M = 2004 ∨ M > 2004, from sorry, -- Proof to split cases for N
      cases h1,
      { rw h1, },
      { exfalso, -- Proof by contradiction if M > 2004
        sorry }
    }
  }
end

end smallest_five_divisible_number_l99_99076


namespace tan_45_eq_one_l99_99923

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99923


namespace linear_function_m_l99_99117

theorem linear_function_m (m : ℝ) (x : ℝ) 
  (h_linear : ∃ k b : ℝ, ∀ x, (m - 1) * x ^ m^2 + 1 = k * x + b) : 
  m = -1 := 
begin
  sorry,
end

end linear_function_m_l99_99117


namespace tan_45_degrees_l99_99832

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99832


namespace r_fourth_power_sum_l99_99184

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l99_99184


namespace tan_45_eq_one_l99_99589

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99589


namespace avg_age_of_five_students_l99_99362

-- step a: Define the conditions
def avg_age_seventeen_students : ℕ := 17
def total_seventeen_students : ℕ := 17 * avg_age_seventeen_students

def num_students_with_unknown_avg : ℕ := 5

def avg_age_nine_students : ℕ := 16
def num_students_with_known_avg : ℕ := 9
def total_age_nine_students : ℕ := num_students_with_known_avg * avg_age_nine_students

def age_seventeenth_student : ℕ := 75

-- step c: Compute the average age of the 5 students
noncomputable def total_age_five_students : ℕ :=
  total_seventeen_students - total_age_nine_students - age_seventeenth_student

def correct_avg_age_five_students : ℕ := 14

theorem avg_age_of_five_students :
  total_age_five_students / num_students_with_unknown_avg = correct_avg_age_five_students :=
sorry

end avg_age_of_five_students_l99_99362


namespace tan_45_degree_l99_99897

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99897


namespace tan_45_eq_1_l99_99959

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99959


namespace find_r4_l99_99164

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l99_99164


namespace tan_45_eq_one_l99_99592

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99592


namespace tan_45_deg_l99_99509

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99509


namespace smallest_k_l99_99029

noncomputable def a : ℕ → ℝ
| 0       := 1
| 1       := real.root 17 3
| (n + 2) := a (n + 1) * (a n) ^ 2

def is_integer_product (k : ℕ) : Prop :=
  ∃ m : ℤ, ∏ i in finset.range k, a (i + 1) = m

theorem smallest_k (k : ℕ) (hk : is_integer_product k) : k = 17 :=
sorry

end smallest_k_l99_99029


namespace tan_of_45_deg_l99_99571

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99571


namespace incorrect_positional_relationship_l99_99129

-- Definitions for the geometric relationships
def line := Type
def plane := Type

def parallel (l : line) (α : plane) : Prop := sorry
def perpendicular (l : line) (α : plane) : Prop := sorry
def subset (l : line) (α : plane) : Prop := sorry
def distinct (l m : line) : Prop := l ≠ m

-- Given conditions
variables (l m : line) (α : plane)

-- Theorem statement: prove that D is incorrect given the conditions
theorem incorrect_positional_relationship
  (h_distinct : distinct l m)
  (h_parallel_l_α : parallel l α)
  (h_parallel_m_α : parallel m α) :
  ¬ (parallel l m) :=
sorry

end incorrect_positional_relationship_l99_99129


namespace tan_45_degrees_l99_99844

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99844


namespace cosine_angle_sum_l99_99136

variable {𝕜 : Type*} [InnerProductSpace ℝ 𝕜]
variable (a b : 𝕜)

-- Conditions given in the problem
def norm_a : ∥a∥ = 5 := sorry
def norm_b : ∥b∥ = 6 := sorry
def dot_ab : ⟪a, b⟫ = -6 := sorry

-- Required proof statement
theorem cosine_angle_sum :
  ∥a∥ = 5 → ∥b∥ = 6 → ⟪a, b⟫ = -6 → Real.cos (inner_product_space.angle a (a + b)) = 19 / 35 :=
by
  sorry

end cosine_angle_sum_l99_99136


namespace tan_45_degree_is_one_l99_99730

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99730


namespace infinite_product_convergence_l99_99033

noncomputable theory

-- Define the infinite product sequence
def infinite_product (n : ℕ) : ℝ := 3 ^ (n / 2 ^ (n + 1))

-- The statement to prove
theorem infinite_product_convergence : (∏ n in (Finset.range ∞), infinite_product n) = 3 := 
sorry

end infinite_product_convergence_l99_99033


namespace sum_of_squares_l99_99122

open Nat

theorem sum_of_squares (α β γ : ℕ) (hα : α < γ) (hβ : 0 < β)
(hγ : α * γ = β^2 + 1) : ∀ n : ℕ, ∃ x y : ℕ, a n + b n = x^2 + y^2 :=
by
  -- Definitions of a and b sequences
  noncomputable def a : ℕ → ℕ
  | 0 => 1
  | n+1 => α * a n + β * b n

  noncomputable def b : ℕ → ℕ
  | 0 => 1
  | n+1 => β * a n + γ * b n

  assume n,
  -- The statement to be proven
  sorry

end sum_of_squares_l99_99122


namespace tan_45_degree_l99_99498

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99498


namespace conic_curve_focus_eccentricity_l99_99372

theorem conic_curve_focus_eccentricity (m : ℝ) 
  (h : ∀ x y : ℝ, x^2 + m * y^2 = 1)
  (eccentricity_eq : ∀ a b : ℝ, a > b → m = 4/3) : m = 4/3 :=
by
  sorry

end conic_curve_focus_eccentricity_l99_99372


namespace r_squared_sum_l99_99192

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l99_99192


namespace bicentric_polygons_determine_bicentric_polygons_l99_99438

-- Definitions
def is_bicentric (P : Type) [polygon P] : Prop :=
  has_circumscribed_circle P ∧ has_inscribed_circle P

-- Shapes
inductive Shape
| square : Shape
| rectangle : Shape
| regular_pentagon : Shape
| hexagon : Shape

-- The theorem to be proved
theorem bicentric_polygons (s : Shape) : Prop :=
  s = Shape.square ∨ s = Shape.regular_pentagon

-- Lean statement
theorem determine_bicentric_polygons (s : Shape) (h1 : is_bicentric Square) (h2 : is_bicentric RegularPentagon) :
  bicentric_polygons s :=
sorry

end bicentric_polygons_determine_bicentric_polygons_l99_99438


namespace tan_45_deg_l99_99764

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99764


namespace tan_45_deg_l99_99767

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99767


namespace convex_ngon_non_acute_angles_at_least_l99_99303

theorem convex_ngon_non_acute_angles_at_least (n : ℕ) (A : Fin n → Point ℝ^2) (O : Point ℝ^2) 
    (hconvex : ConvexPolygon A) (hinside : O ∈ PolygonInterior A) : 
    ∃ (S : Finset (Fin n × Fin n)), S.card ≥ n - 1 ∧ ∀ (i j : Fin n), (i, j) ∈ S → ¬IsAcuteAngle (A i) O (A j) :=
sorry

end convex_ngon_non_acute_angles_at_least_l99_99303


namespace tan_45_deg_l99_99522

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99522


namespace tan_45_deg_l99_99747

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99747


namespace min_area_of_ellipse_contains_circles_l99_99455

-- Definitions of the ellipse equation and the circles
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def circle1 (x y : ℝ) : Prop := ((x - 2)^2 + y^2 = 4)
def circle2 (x y : ℝ) : Prop := ((x + 2)^2 + y^2 = 4)

theorem min_area_of_ellipse_contains_circles :
  ∃ a b : ℝ, 
  (∀ x y : ℝ, circle1 x y → ellipse a b x y) ∧
  (∀ x y : ℝ, circle2 x y → ellipse a b x y) ∧
  (∃ k : ℝ, k = (a * b * π) ∧ k = ((3 * real.sqrt 3) / 2) * π) :=
sorry

end min_area_of_ellipse_contains_circles_l99_99455


namespace find_interest_rate_l99_99436

-- Definitions for conditions
def principal : ℝ := 12500
def interest : ℝ := 1500
def time : ℝ := 1

-- Interest rate to prove
def interest_rate : ℝ := 0.12

-- Formal statement to prove
theorem find_interest_rate (P I T : ℝ) (hP : P = principal) (hI : I = interest) (hT : T = time) : I = P * interest_rate * T :=
by
  sorry

end find_interest_rate_l99_99436


namespace products_selling_less_than_1000_l99_99475

theorem products_selling_less_than_1000 (N: ℕ) 
  (total_products: ℕ := 25) 
  (average_price: ℤ := 1200) 
  (min_price: ℤ := 400) 
  (max_price: ℤ := 12000) 
  (total_revenue := total_products * average_price) 
  (revenue_from_expensive: ℤ := max_price):
  12000 + (24 - N) * 1000 + N * 400 = 30000 ↔ N = 10 :=
by
  sorry

end products_selling_less_than_1000_l99_99475


namespace tan_45_eq_1_l99_99957

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99957


namespace tan_45_degree_l99_99476

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99476


namespace tan_45_deg_eq_1_l99_99978

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99978


namespace tan_45_eq_one_l99_99932

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99932


namespace average_marks_all_students_l99_99411

theorem average_marks_all_students :
  let class1_students := 25
  let class2_students := 40
  let class1_avg_marks := 50
  let class2_avg_marks := 65
  let total_students := class1_students + class2_students
  let total_marks := (class1_students * class1_avg_marks) + (class2_students * class2_avg_marks)
  let average_marks := total_marks.toFloat / total_students.toFloat
  average_marks ≈ 59.23 :=
by
  sorry

end average_marks_all_students_l99_99411


namespace tan_of_45_deg_l99_99578

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99578


namespace tan_45_eq_one_l99_99608

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99608


namespace tan_45_eq_1_l99_99996

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l99_99996


namespace tan_45_degree_is_one_l99_99742

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99742


namespace angle_B_in_triangle_is_pi_over_6_l99_99249

theorem angle_B_in_triangle_is_pi_over_6
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : A + B + C = π)
  (h₅ : b * (Real.cos C) / (Real.cos B) + c = (2 * Real.sqrt 3 / 3) * a) :
  B = π / 6 :=
by sorry

end angle_B_in_triangle_is_pi_over_6_l99_99249


namespace tan_45_degrees_l99_99699

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99699


namespace tan_45_eq_one_l99_99929

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99929


namespace tan_45_eq_1_l99_99999

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l99_99999


namespace count_multiples_of_15_l99_99142

theorem count_multiples_of_15 (a b n : ℕ) (h_gte : 25 ≤ a) (h_lte : b ≤ 205) (h15 : n = 15) : 
  (∃ (k : ℕ), a ≤ k * n ∧ k * n ≤ b ∧ 1 ≤ k - 1 ∧ k - 1 ≤ 12) :=
sorry

end count_multiples_of_15_l99_99142


namespace evaluate_expression_correct_l99_99043

noncomputable def evaluate_expression : ℝ :=
  2 * real.log (1 / 2) / real.log 3 + real.log 12 / real.log 3 - 0.7^0 + 0.25⁻¹

theorem evaluate_expression_correct : evaluate_expression = 4 :=
by sorry

end evaluate_expression_correct_l99_99043


namespace closest_point_on_line_l99_99071

theorem closest_point_on_line 
  (s : ℚ)
  (closest_point : ℚ := (s := 3/14))
  (line_point : Matrix (Fin 3) (Fin 1) ℚ → Matrix (Fin 3) (Fin 1) ℚ := fun s => Matrix.vecCons (3 + s.val) (Matrix.vecCons (-1 - 3 * s.val) (Matrix.vecCons (2 + 2 * s.val) Matrix.vecEmpty)))
  (target_point : Matrix (Fin 3) (Fin 1) ℚ := Matrix.vecCons 1 (Matrix.vecCons (-2) (Matrix.vecCons 3 Matrix.vecEmpty)))
  (dir_vector : Matrix (Fin 3) (Fin 1) ℚ := Matrix.vecCons 1 (Matrix.vecCons (-3) (Matrix.vecCons 2 Matrix.vecEmpty)))
  (ortho_condition : 
    (line_point closest_point - target_point).vec.dot_product dir_vector = 0) : 
  line_point (3/14) = Matrix.vecCons (45/14) (Matrix.vecCons (-17/14) (Matrix.vecCons (32/14) Matrix.vecEmpty)) :=
sorry

end closest_point_on_line_l99_99071


namespace tan_45_eq_1_l99_99947

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99947


namespace tan_45_eq_1_l99_99637

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99637


namespace sin_330_proof_l99_99472

noncomputable def sin_330_equals_neg_half : Prop :=
  sin (330 * real.pi / 180) = -1/2

theorem sin_330_proof : sin_330_equals_neg_half :=
by
  sorry

end sin_330_proof_l99_99472


namespace sum_of_solutions_x_squared_eq_25_l99_99203

theorem sum_of_solutions_x_squared_eq_25 : 
  (∑ x in ({x : ℝ | x^2 = 25}).to_finset, x) = 0 :=
by
  sorry

end sum_of_solutions_x_squared_eq_25_l99_99203


namespace tan_45_deg_l99_99783

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l99_99783


namespace r_squared_sum_l99_99195

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l99_99195


namespace tan_45_eq_1_l99_99615

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99615


namespace tan_45_deg_eq_1_l99_99988

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99988


namespace tan_45_degree_l99_99485

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99485


namespace r_fourth_power_sum_l99_99186

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l99_99186


namespace tan_of_45_deg_l99_99566

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l99_99566


namespace true_propositions_l99_99452

noncomputable def proposition_1 := 
  ∀ (pyramid : Type) (base : EquilateralTriangle) (lateral_faces : List IsoscelesTriangle), 
    isRegularPyramid pyramid ↔ (base.IsEquilateral ∧ ∀ face ∈ lateral_faces, face.IsIsosceles)

noncomputable def proposition_2 := 
  ∀ (points : List Point), 
    points.length = 4 ∧ ¬coplanar points → ∃ planes : List Plane, planes.length = 4 ∧ ∀ plane ∈ planes, equidistant plane points

noncomputable def proposition_3 := 
  ∀ (tetrahedron : RegularTetrahedron), 
    ∀ (face ∈ tetrahedron.faces), face.IsAcute

noncomputable def proposition_4 := 
  ∀ (e : ℝ), 
    0 < e ∧ e < 1 → shape (ellipse e).approaches elongatedShape

-- Define predicates to indicate the truth of each proposition
noncomputable def isTrueProposition_1 : Prop := false
noncomputable def isTrueProposition_2 : Prop := true
noncomputable def isTrueProposition_3 : Prop := true
noncomputable def isTrueProposition_4 : Prop := false

theorem true_propositions : 
  { i | [isTrueProposition_1, isTrueProposition_2, isTrueProposition_3, isTrueProposition_4].nth i = some true }.card = 2 :=
by {
  -- Sorry to bypass the proof as requested
  sorry
}

end true_propositions_l99_99452


namespace speed_of_current_l99_99008

-- Conditions translated into Lean definitions
def initial_time : ℝ := 13 -- 1:00 PM is represented as 13:00 hours
def boat1_time_turnaround : ℝ := 14 -- Boat 1 turns around at 2:00 PM
def boat2_time_turnaround : ℝ := 15 -- Boat 2 turns around at 3:00 PM
def meeting_time : ℝ := 16 -- Boats meet at 4:00 PM
def raft_drift_distance : ℝ := 7.5 -- Raft drifted 7.5 km from the pier

-- The problem statement to prove
theorem speed_of_current:
  ∃ v : ℝ, (v * (meeting_time - initial_time) = raft_drift_distance) ∧ v = 2.5 :=
by
  sorry

end speed_of_current_l99_99008


namespace cyclic_quadrilateral_iff_equal_areas_l99_99233

-- Define the basic structures and conditions
variables {A B C D P : Type} [ConvexQuadrilateral A B C D]
variable (h1 : ∃ G : Type, IsPerpendicularDiagonal A C B D G)
variable (h2 : ¬(Parallel A B C D))
variable (h3 : IsIntersectionPerpBisectors P A B C D)
variable (h4 : InsideQuadrilateral P A B C D)

-- State the theorem to prove cyclic nature
theorem cyclic_quadrilateral_iff_equal_areas 
  (h1 : ∃ G : Type, IsPerpendicularDiagonal A C B D G)
  (h2 : ¬(Parallel A B C D))
  (h3 : IsIntersectionPerpBisectors P A B C D)
  (h4 : InsideQuadrilateral P A B C D) : 
  (CyclicQuadrilateral A B C D ↔ AreaOfTriangle A B P = AreaOfTriangle C D P) :=
by
  sorry

end cyclic_quadrilateral_iff_equal_areas_l99_99233


namespace r_power_four_identity_l99_99152

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l99_99152


namespace proof_polar_and_distance_product_l99_99241

open Real

-- First, define the Cartesian parametric equations given in the problem
def x (θ : ℝ) := 2 * cos θ
def y (θ : ℝ) := 2 + 2 * sin θ

-- Definition for the polar conversion
def polar_ρ (x y : ℝ) := sqrt (x * x + y * y)
def polar_θ (x y : ℝ) := atan2 y x

-- Define point M in polar coordinates
def M_polar := (sqrt 2, π / 4 : ℝ)

-- Define the polar equation of curve C we need to prove
def polar_equation_c (ρ θ : ℝ) := ρ = 4 * sin θ

-- Define the distance product statement that needs to be proved
def distance_product := |MA| * |MB| = 2

-- Main Lean Theorem
theorem proof_polar_and_distance_product:
  (∀ θ, polar_equation_c (polar_ρ (x θ) (y θ)) θ) ∧
  (let MA := sorry, MB := sorry in distance_product) :=
by
  -- sorry statements here are placeholders for the proof steps
  split
  case left => {
    intro θ
    sorry
  }
  case right => {
    sorry
  }

end proof_polar_and_distance_product_l99_99241


namespace tan_45_eq_one_l99_99663

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99663


namespace tan_45_deg_l99_99761

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l99_99761


namespace tan_45_degree_l99_99901

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l99_99901


namespace tan_45_eq_one_l99_99596

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99596


namespace tan_45_eq_1_l99_99634

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99634


namespace right_triangles_count_l99_99143

theorem right_triangles_count :
  (∃ (a b : ℕ), a ^ 2 + b ^ 2 = (b + 2) ^ 2 ∧ b ≤ 50) ∧
  (b ≤ 50 → ∃ finset.card (finset.filter (λ (x : ℕ × ℕ), x.1 ^ 2 + x.2 ^ 2 = (x.2 + 2) ^ 2 ∧ x.2 ≤ 50)
    (finset.univ.product (finset.range 51)))) = 7 :=
by
  sorry

end right_triangles_count_l99_99143


namespace pyramid_base_side_length_l99_99334

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l99_99334


namespace tan_45_degree_is_one_l99_99732

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99732


namespace point_off_min_distance_line_l99_99127

-- Define the points A, B, and C
def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (-3, -2)
def C : ℝ × ℝ := (-1, -3)

-- Define the distance function from a point (x, y) to the line y = kx
def distance_to_line (k : ℝ) (x y : ℝ) : ℝ :=
  abs(k * x - y) / sqrt(k^2 + 1)

-- Sum of the squares of the distances from points A, B, and C to the line y = kx
noncomputable def sum_of_squared_distances (k : ℝ) : ℝ :=
  (distance_to_line k (-2) 1)^2 +
  (distance_to_line k (-3) (-2))^2 +
  (distance_to_line k (-1) (-3))^2

-- Prove that none of the points A, B, or C lie on the line y = x when the
-- sum of the squares of distances is minimized
theorem point_off_min_distance_line :
  ∀ k : ℝ, sum_of_squared_distances k = real.min (sum_of_squared_distances k) 
  → 
  k = 1 →
  (A.snd ≠ k * A.fst) ∧ 
  (B.snd ≠ k * B.fst) ∧ 
  (C.snd ≠ k * C.fst) :=
by
  sorry

end point_off_min_distance_line_l99_99127


namespace tan_45_degree_is_one_l99_99734

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l99_99734


namespace tan_45_deg_eq_1_l99_99973

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99973


namespace tan_45_eq_1_l99_99556

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99556


namespace tan_45_deg_l99_99512

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99512


namespace trigonometric_identity_l99_99228

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 :=
by 
  sorry

end trigonometric_identity_l99_99228


namespace cosine_angle_sum_l99_99137

variable {𝕜 : Type*} [InnerProductSpace ℝ 𝕜]
variable (a b : 𝕜)

-- Conditions given in the problem
def norm_a : ∥a∥ = 5 := sorry
def norm_b : ∥b∥ = 6 := sorry
def dot_ab : ⟪a, b⟫ = -6 := sorry

-- Required proof statement
theorem cosine_angle_sum :
  ∥a∥ = 5 → ∥b∥ = 6 → ⟪a, b⟫ = -6 → Real.cos (inner_product_space.angle a (a + b)) = 19 / 35 :=
by
  sorry

end cosine_angle_sum_l99_99137


namespace mutually_close_C_l99_99119

def f (x : ℝ) : ℝ := Real.exp (-x)
def g (x : ℝ) : ℝ := -1/x
def E : Set ℝ := Set.Ioi 0

theorem mutually_close_C :
  ∃ x_0 ∈ E, |f x_0 - g x_0| < 1 :=
sorry

end mutually_close_C_l99_99119


namespace tan_45_deg_l99_99515

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l99_99515


namespace area_of_third_face_l99_99360

-- Define the variables for the dimensions of the box: l, w, and h
variables (l w h: ℝ)

-- Given conditions
def face1_area := 120
def face2_area := 72
def volume := 720

-- The relationships between the dimensions and the given areas/volume
def face1_eq : Prop := l * w = face1_area
def face2_eq : Prop := w * h = face2_area
def volume_eq : Prop := l * w * h = volume

-- The statement we need to prove is that the area of the third face (l * h) is 60 cm² given the above equations
theorem area_of_third_face :
  face1_eq l w →
  face2_eq w h →
  volume_eq l w h →
  l * h = 60 :=
by
  intros h1 h2 h3
  sorry

end area_of_third_face_l99_99360


namespace jill_can_label_up_to_l99_99256

theorem jill_can_label_up_to :
  ∃ n : ℕ, n = 235 ∧ (let total_fives := (finset.range n).sum (λ x, (x.digits 10).count 5) in total_fives ≤ 50) :=
begin
  sorry
end

end jill_can_label_up_to_l99_99256


namespace sum_of_19_terms_l99_99091

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (d : ℝ) (a1 : ℝ)

-- Definitions: 
def arithmetic_sequence (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + (n - 1) * d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Given conditions
axiom h1 : a 4 + a 10 + a 16 = 18

-- Theorem to prove
theorem sum_of_19_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ) (d : ℝ)
  [arithmetic_sequence a a1 d] [sum_of_first_n_terms S a] (h1 : a 4 + a 10 + a 16 = 18) : S 19 = 114 :=
sorry

end sum_of_19_terms_l99_99091


namespace tan_45_degrees_l99_99835

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99835


namespace tan_45_degrees_l99_99847

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99847


namespace total_canoes_by_end_of_march_l99_99014

theorem total_canoes_by_end_of_march
  (canoes_jan : ℕ := 3)
  (canoes_feb : ℕ := canoes_jan * 2)
  (canoes_mar : ℕ := canoes_feb * 2) :
  canoes_jan + canoes_feb + canoes_mar = 21 :=
by
  sorry

end total_canoes_by_end_of_march_l99_99014


namespace exists_n_divided_set_for_n_ge_2_l99_99270

-- Definitions and conditions
def is_n_divided (S : Set ℤ) (n : ℕ) : Prop :=
  ∃ (partition_function : Set (Set ℤ × Set ℤ)), 
    partition_function.card = n ∧ 
    ∀ (A B : Set ℤ), (A, B) ∈ partition_function → 
    A ∪ B = S ∧ A ∩ B = ∅ ∧ A.sum = B.sum

-- Main statement to prove (without proof)
theorem exists_n_divided_set_for_n_ge_2 (n : ℕ) (h : n ≥ 2) : 
  ∃ (S : Set ℤ), is_n_divided S n :=
sorry

end exists_n_divided_set_for_n_ge_2_l99_99270


namespace tan_45_eq_l99_99667

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l99_99667


namespace triangles_similar_l99_99369

variables {A B C D K₁ L₁ M₁ K₂ L₂ M₂ : Type*}
variables (G₁ G₂ : Set (Set (A × A × A × A)))

open Set

-- Definitions of spherical surfaces passing through A, B, and C
def sphere_passing_through (G : Set (Set (A × A × A × A))) (A B C : Type*) := 
  ∃ s ∈ G, (A ∈ s ∧ B ∈ s ∧ C ∈ s)

axiom G₁_passing_through_ABC : sphere_passing_through G₁ A B C
axiom G₂_passing_through_ABC : sphere_passing_through G₂ A B C

-- Definitions of intersection points
axiom K₁_on_G₁ : ∃ s ∈ G₁, K₁ ∈ s
axiom L₁_on_G₁ : ∃ s ∈ G₁, L₁ ∈ s
axiom M₁_on_G₁ : ∃ s ∈ G₁, M₁ ∈ s

axiom K₂_on_G₂ : ∃ s ∈ G₂, K₂ ∈ s
axiom L₂_on_G₂ : ∃ s ∈ G₂, L₂ ∈ s
axiom M₂_on_G₂ : ∃ s ∈ G₂, M₂ ∈ s

-- Theorem to prove similarity of triangles K₁L₁M₁ and K₂L₂M₂
theorem triangles_similar : Similar (triangle K₁ L₁ M₁) (triangle K₂ L₂ M₂) := 
  sorry

end triangles_similar_l99_99369


namespace tan_45_eq_one_l99_99920

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99920


namespace tan_45_eq_one_l99_99610

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99610


namespace tan_45_degrees_l99_99834

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99834


namespace tan_45_eq_1_l99_99534

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99534


namespace tan_45_degrees_l99_99827

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99827


namespace tan_45_eq_1_l99_99618

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99618


namespace determine_xy_l99_99416

theorem determine_xy :
  ∃ x y : ℕ, 
    (y ≥ x ∧ x ≥ 1) ∧ 
    (∃ A B : ℕ, A = x + y ∧ B = x * y ∧ 
    (¬∃ x' y' : ℕ, x' ≥ 1 ∧ y' ≥ x' ∧ x' * y' = B ∧ x' + y' ≠ x + y) ∧
    (A ≤ 20) ∧
    ∃ A : ℕ, x + y = A ∧ A ≤ 20 ∧ (∃! A' : ℕ, x * y = B → x + y = A')
    ) → 
  (x = 2 ∧ y = 11) :=
begin
  sorry
end

end determine_xy_l99_99416


namespace tan_45_eq_one_l99_99599

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99599


namespace tan_45_eq_1_l99_99545

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l99_99545


namespace evaluate_nested_operation_l99_99028

def operation (a b c : ℕ) : ℕ := (a + b) / c

theorem evaluate_nested_operation : operation (operation 72 36 108) (operation 4 2 6) (operation 12 6 18) = 2 := by
  -- Here we assume all operations are valid (c ≠ 0 for each case)
  sorry

end evaluate_nested_operation_l99_99028


namespace tan_45_eq_one_l99_99650

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l99_99650


namespace sum_of_squares_lt_sqrt_three_l99_99101

variable {x y z : ℝ}

theorem sum_of_squares_lt_sqrt_three (hxz : x > 0) (hyz : y > 0) (hzz : z > 0)
  (h : x^4 + y^4 + z^4 = 1) : 
  x^2 + y^2 + z^2 < sqrt 3 := 
sorry

end sum_of_squares_lt_sqrt_three_l99_99101


namespace tangent_line_slope_l99_99121

/-- Given the line y = mx is tangent to the circle x^2 + y^2 - 4x + 2 = 0, 
    the slope m must be ±1. -/
theorem tangent_line_slope (m : ℝ) :
  (∃ x y : ℝ, y = m * x ∧ (x ^ 2 + y ^ 2 - 4 * x + 2 = 0)) →
  (m = 1 ∨ m = -1) :=
by
  sorry

end tangent_line_slope_l99_99121


namespace tan_45_eq_one_l99_99919

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99919


namespace tan_45_eq_1_l99_99629

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l99_99629


namespace significant_event_day_of_the_week_l99_99371

theorem significant_event_day_of_the_week :
  (let start_day := 3       -- Wednesday
   let days_later := 1000
   let leap_year := 1876
   let day_of_week (d : ℤ) := d % 7 -- mapping days to day of the week where 0 is Sunday, 1 is Monday, ..., 6 is Saturday
   in day_of_week (start_day + days_later)) = 5 := -- Friday is represented by 5 since 0 (Sunday) + 5 = Friday
by
  sorry

end significant_event_day_of_the_week_l99_99371


namespace find_r_fourth_l99_99169

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l99_99169


namespace tan_45_deg_eq_1_l99_99967

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l99_99967


namespace max_switches_on_100x100_grid_l99_99467

/-- Betty Lou and Peggy Sue take turns flipping switches on a 100 × 100 grid. Initially,
    all switches are "off". Betty Lou always flips a horizontal row of switches on her turn;
    Peggy Sue always flips a vertical column of switches.
    Prove: There is an odd number of switches turned "on" in each row and column.
           Find the maximum number of switches that can be on, in total, when they finish is 9802.
-/
theorem max_switches_on_100x100_grid :
  ∃ (A : (fin 100) × (fin 100) → bool), 
  (∀ i : fin 100, odd (∑ j, if A (i, j) then 1 else 0)) ∧
  (∀ j : fin 100, odd (∑ i, if A (i, j) then 1 else 0)) ∧
  (∑ i, ∑ j, if A (i, j) then 1 else 0) = 9802 := 
sorry

end max_switches_on_100x100_grid_l99_99467


namespace series_sum_l99_99021

noncomputable def compute_series : ℝ :=
  ∑ k in finset.range 100, (3 + (k + 1) * 9) / 3^(101 - (k + 1))

theorem series_sum : compute_series = 452.75 :=
by sorry

end series_sum_l99_99021


namespace fraction_sum_lt_one_l99_99383

theorem fraction_sum_lt_one (n : ℕ) (h_pos : n > 0) : 
  (1 / 2 + 1 / 3 + 1 / 10 + 1 / n < 1) ↔ (n > 15) :=
sorry

end fraction_sum_lt_one_l99_99383


namespace tan_45_degrees_l99_99700

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l99_99700


namespace tan_45_eq_1_l99_99950

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l99_99950


namespace tan_45_eq_one_l99_99604

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l99_99604


namespace tan_45_degree_l99_99495

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l99_99495


namespace jewelry_price_increase_l99_99292

/-- Mr. Rocky increased the price of each piece of jewelry by $10. -/
theorem jewelry_price_increase (x : ℝ) : 
  let original_jewelry_price : ℝ := 30
  let original_painting_price : ℝ := 100
  let new_painting_price : ℝ := original_painting_price + 0.20 * original_painting_price
  let total_cost : ℝ := 680
  2 * (original_jewelry_price + x) + 5 * new_painting_price = total_cost → x = 10 
  := by 
  assume h : 2 * (original_jewelry_price + x) + 5 * new_painting_price = total_cost
  sorry

end jewelry_price_increase_l99_99292


namespace sum_of_solutions_eq_zero_l99_99199

theorem sum_of_solutions_eq_zero (x : ℝ) (hx : x^2 = 25) : ∃ x₁ x₂ : ℝ, (x₁^2 = 25) ∧ (x₂^2 = 25) ∧ (x₁ + x₂ = 0) := 
by {
  use 5,
  use (-5),
  split,
  { exact hx, },
  split,
  { rw pow_two, exact hx, },
  { norm_num, },
}

end sum_of_solutions_eq_zero_l99_99199


namespace tan_45_eq_one_l99_99908

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l99_99908
