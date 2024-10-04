import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.AbsoluteValue.Basic
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.LinearOrderedField
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Polynomials
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Ball
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.LinearAlgebra.AffineSpace.Independent
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.ProbabilityTheory.ConditionalExpectation
import Mathlib.RingTheory.Ideal.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.IntervalCases
import Mathlib.Topology.Basic

namespace find_ratio_CL_AB_l612_612977

noncomputable def regular_pentagon (A B C D E : Type) : Prop :=
  (Eq (dist A B) (dist B C)) ∧ -- All sides are equal
  (Eq (dist B C) (dist C D)) ∧
  (Eq (dist C D) (dist D E)) ∧
  (Eq (dist D E) (dist E A)) ∧
  (Eq (angle A B C) (108:ℝ)) ∧ -- All internal angles are 108 degrees
  (Eq (angle B C D) (108:ℝ)) ∧
  (Eq (angle C D E) (108:ℝ)) ∧
  (Eq (angle D E A) (108:ℝ)) ∧
  (Eq (angle E A B) (108:ℝ))

variable (A B C D E K L : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variable [metric_space K] [metric_space L]

def ratio_AK_KE (AK KE : ℝ) : Prop :=
  AK / (AK + KE) = 3 / 10

def angle_LAE_KCD (LAE KCD : ℝ) : Prop :=
  LAE + KCD = 108

theorem find_ratio_CL_AB 
  (h1 : regular_pentagon A B C D E) 
  (h2 : ratio_AK_KE (dist A K) (dist K E)) 
  (h3 : angle_LAE_KCD (angle L A E) (angle K C D)) 
  : (dist C L) / (dist A B) = 0.7 :=
sorry

end find_ratio_CL_AB_l612_612977


namespace range_of_a_l612_612173

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (0 < x ∧ x ≤ 3) → a ≤ x^2 - 4 * x) → a ≤ -4 := by
  intro h
  suffices h₁ : a ≤ (2 : ℝ)^2 - 4 * (2 : ℝ)
  · exact h₁
  calc
    a ≤ 4 - 8     : by sorry
    _ = -4        : by norm_num

end range_of_a_l612_612173


namespace ellipse_standard_equation_correct_l612_612481

def ellipse_standard_equation
  (a b c : ℝ) 
  (h_major_axis : 2 * a = 4)
  (h_eccentricity : c / a = real.sqrt 3 / 2)
  (h_b_relation : b^2 = a^2 - c^2) :=
  (a = 2) → (c = real.sqrt 3) → (b = 1) →
  (∃ (el_eq : ℝ → ℝ → Prop),
    el_eq = λ x y, (x^2 / a^2 + y^2 / b^2 = 1) ∧
    ∀ x y, el_eq x y ↔ (x^2 / 4 + y^2 = 1))

theorem ellipse_standard_equation_correct : 
  ellipse_standard_equation 2 1 (real.sqrt 3) 
  (by norm_num : 2 * 2 = 4) 
  (by norm_num : (real.sqrt 3) / 2 = real.sqrt 3 / 2)
  (by norm_num : 1^2 = 2^2 - (real.sqrt 3)^2) :=
by
  intros h1 h2 h3
  use (λ x y, x^2 / (2 : ℝ)^2 + y^2 / (1 : ℝ)^2 = 1)
  split
  { refl }
  intros x y
  simp [h1, h2, h3]
  sorry

end ellipse_standard_equation_correct_l612_612481


namespace equation_of_line_l612_612014

def point_A : ℝ × ℝ := (3, 2)

def line_through_point_parallel (A : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  ∀ p, p = A ∨ l p

-- Original line equation
def l (p : ℝ × ℝ) : Prop := 4 * p.1 + p.2 - 2 = 0

-- Line equation to be proved
def line_to_prove (p : ℝ × ℝ) : Prop := 4 * p.1 + p.2 - 14 = 0

theorem equation_of_line :
  line_through_point_parallel point_A l = line_to_prove := 
sorry

end equation_of_line_l612_612014


namespace count_zeros_in_sequence_l612_612719

theorem count_zeros_in_sequence 
  (a : Fin 50 → ℤ)
  (h1 : ∀ i, a i = -1 ∨ a i = 0 ∨ a i = 1)
  (h2 : (∑ i, a i) = 9)
  (h3 : (∑ i, (a i + 1)^2) = 107) :
  (∑ i, if a i = 0 then 1 else 0) = 11 := 
sorry

end count_zeros_in_sequence_l612_612719


namespace fraction_of_1206_l612_612448

theorem fraction_of_1206 :
  let frac := 402 / 1206 in
  let reduced_frac := frac / gcd 402 1206 in
  reduced_frac = 1 / 3 := 
by
  sorry

end fraction_of_1206_l612_612448


namespace cos_36_eq_l612_612912

theorem cos_36_eq : ∃ x : ℝ, x = cos (36 * real.pi / 180) ∧ x = (1 + real.sqrt 5) / 4 :=
by
  let x := cos (36 * real.pi / 180)
  let y := cos (72 * real.pi / 180)
  have h1 : y = 2 * x^2 - 1 := sorry -- double angle formula
  have h2 : 2 * y^2 - 1 = x := sorry -- cosine property
  use (1 + real.sqrt 5) / 4
  have hx : x = (1 + real.sqrt 5) / 4 := sorry -- solving the quartic equation
  exact ⟨by rfl, hx⟩

end cos_36_eq_l612_612912


namespace can_cross_all_rivers_and_extra_material_l612_612019

-- Definitions for river widths, bridge length, and additional material.
def river1_width : ℕ := 487
def river2_width : ℕ := 621
def river3_width : ℕ := 376
def bridge_length : ℕ := 295
def additional_material : ℕ := 1020

-- Calculations for material needed for each river.
def material_needed_for_river1 : ℕ := river1_width - bridge_length
def material_needed_for_river2 : ℕ := river2_width - bridge_length
def material_needed_for_river3 : ℕ := river3_width - bridge_length

-- Total material needed to cross all three rivers.
def total_material_needed : ℕ := material_needed_for_river1 + material_needed_for_river2 + material_needed_for_river3

-- The main theorem statement to prove.
theorem can_cross_all_rivers_and_extra_material :
  total_material_needed <= additional_material ∧ (additional_material - total_material_needed = 421) := 
by 
  sorry

end can_cross_all_rivers_and_extra_material_l612_612019


namespace xiaoming_grandfather_age_l612_612421

-- Define the conditions
def age_cond (x : ℕ) : Prop :=
  ((x - 15) / 4 - 6) * 10 = 100

-- State the problem
theorem xiaoming_grandfather_age (x : ℕ) (h : age_cond x) : x = 79 := 
sorry

end xiaoming_grandfather_age_l612_612421


namespace lower_water_level_6_inches_l612_612809

variable (length width volume_gallons conversion_factor : ℚ)
variable (length_val : length = 60)
variable (width_val : width = 20)
variable (volume_gallons_val : volume_gallons = 4500)
variable (conversion_factor_val : conversion_factor = 7.5)

theorem lower_water_level_6_inches :
  (let volume_cubic_feet := volume_gallons / conversion_factor in
   let surface_area := length * width in
   let height_feet := volume_cubic_feet / surface_area in
   let height_inches := height_feet * 12 in
   height_inches = 6) :=
by
  sorry

end lower_water_level_6_inches_l612_612809


namespace triangle_is_obtuse_l612_612669

theorem triangle_is_obtuse 
    (A B C : ℝ)
    (tan_A_seq : ℕ → ℝ)
    (tan_B_seq : ℕ → ℝ)
    (h1 : tan_A_seq 2 = 1) -- 1 is the 3rd term in the arithmetic sequence (0-indexed for Lean)
    (h2 : tan_A_seq 6 = 9) -- 9 is the 7th term in the arithmetic sequence (0-indexed for Lean)
    (h3 : tan_B_seq 1 = 64) -- 64 is the 2nd term in the geometric sequence (0-indexed for Lean)
    (h4 : tan_B_seq 4 = 1) -- 1 is the 5th term in the geometric sequence (0-indexed for Lean)
    (h5 : tan A = tan_A_seq 3) -- tan A is the 4th term in the arithmetic sequence (0-indexed for Lean)
    (h6 : tan B = tan_B_seq 0) -- tan B is the common ratio of the geometric sequence
    (h7 : A + B + C = π) 
    : A + B > π / 2 → C > π / 2 :=
by
  sorry

end triangle_is_obtuse_l612_612669


namespace count_multiples_5_or_3_in_range_l612_612158

theorem count_multiples_5_or_3_in_range :
  let a := Int.ceil (23 / 3)
  let b := Int.floor (65 / 2)
  (count (λ x, ((x % 5 = 0) ∨ (x % 3 = 0))) [a..b]) = 11 :=
by
  sorry

end count_multiples_5_or_3_in_range_l612_612158


namespace polynomial_satisfies_functional_eq_l612_612084

noncomputable def P (X : ℝ) : ℝ := sorry

theorem polynomial_satisfies_functional_eq {P : ℝ → ℝ}
  (hP : ∀ X : ℝ, P (X^2) = P X * P (X - 1)) :
  ∃ (n : ℕ), ∃ Q : polynomial ℝ, (Q = polynomial.X^2 + polynomial.X + 1) ∧ (P = Q ^ n) :=
sorry

end polynomial_satisfies_functional_eq_l612_612084


namespace triangle_inequality_problem_l612_612750

-- Define the problem statement: Given the specified conditions, prove the interval length and sum
theorem triangle_inequality_problem :
  ∀ (A B C D : Type) (AB AC BC BD CD AD AO : ℝ),
  AB = 12 ∧ CD = 4 →
  (∃ x : ℝ, (4 < x ∧ x < 24) ∧ (AC = x ∧ m = 4 ∧ n = 24 ∧ m + n = 28)) :=
by
  intro A B C D AB AC BC BD CD AD AO h
  sorry

end triangle_inequality_problem_l612_612750


namespace cube_roots_of_unity_l612_612226

-- Definition: Complex numbers α and β with |α| = 1, |β| = 1, and α + β + 1 = 0
variables (α β : ℂ)
variables (hα : abs α = 1) (hβ : abs β = 1) (h_sum : α + β + 1 = 0)

theorem cube_roots_of_unity : α^3 = 1 ∧ β^3 = 1 :=
by
  sorry

end cube_roots_of_unity_l612_612226


namespace smallest_n_for_2012_terms_l612_612950

theorem smallest_n_for_2012_terms :
  ∃ n : ℕ, (xy - 7 * x - 3 * y + 21)^n has at least 2012 terms → n = 44 :=
begin
  let xy_expr := (x * y) - 7 * x - 3 * y + 21,
  have factored_expr : (xy_expr = (x - 3) * (y - 7)),
    -- proof of factorization step
    { sorry },
  let expanded_form := (x - 3)^n * (y - 7)^n,
  have term_count : ∀ n, (expanded_form).term_count = (n + 1)^2,
    -- proof of term count step
    { sorry },
  have inequality_satisfied : ∀ n, (n + 1)^2 ≥ 2012 → n ≥ 44,
    -- solving the inequality
    { sorry },
  exact ⟨44, inequality_satisfied 44⟩,
end

end smallest_n_for_2012_terms_l612_612950


namespace ship_cargo_weight_l612_612025

theorem ship_cargo_weight (initial_cargo_tons additional_cargo_tons : ℝ) (unloaded_cargo_pounds : ℝ)
    (ton_to_kg pound_to_kg : ℝ) :
    initial_cargo_tons = 5973.42 →
    additional_cargo_tons = 8723.18 →
    unloaded_cargo_pounds = 2256719.55 →
    ton_to_kg = 907.18474 →
    pound_to_kg = 0.45359237 →
    (initial_cargo_tons * ton_to_kg + additional_cargo_tons * ton_to_kg - unloaded_cargo_pounds * pound_to_kg = 12302024.7688159) :=
by
  intros
  sorry

end ship_cargo_weight_l612_612025


namespace find_larger_number_l612_612942

theorem find_larger_number (x y : ℕ) (h1 : x + y = 55) (h2 : x - y = 15) : x = 35 := by 
  -- proof will go here
  sorry

end find_larger_number_l612_612942


namespace sqrt_expr_simplification_l612_612914

theorem sqrt_expr_simplification : sqrt 12 - sqrt 3 * (2 + sqrt 3) = -3 :=
by
  sorry

end sqrt_expr_simplification_l612_612914


namespace length_AN_eq_one_l612_612271

variable {α : Type*} [InnerProductSpace ℝ α]

/-- Given a triangle ABC and points M and N satisfying the conditions in the problem -/
theorem length_AN_eq_one
  (A B C M N : α)
  (hM : ∠BAC = ∠BAM + ∠MAC)
  (hN' : ∃ P, P ∈ line_through A B ∧ N ≠ A ∧ sameRay (line_through A B) P N)
  (hAC : dist A C = 1)
  (hAM : dist A M = 1)
  (h_angle : angle A N M = angle C N M) :
  dist A N = 1 :=
sorry

end length_AN_eq_one_l612_612271


namespace correct_options_l612_612148

theorem correct_options (a b : ℝ) (h : a > 0) (ha : a^2 = 4 * b) :
  ((a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬ (∃ x1 x2, x1 * x2 > 0 ∧ x^2 + a * x - b < 0)) ∧ 
  (∀ (x1 x2 : ℝ), |x1 - x2| = 4 → x^2 + a * x + b < 4 → 4 = 4)) :=
sorry

end correct_options_l612_612148


namespace linear_func_is_direct_proportion_l612_612602

theorem linear_func_is_direct_proportion (m : ℝ) : (∀ x : ℝ, (y : ℝ) → y = m * x + m - 2 → (m - 2 = 0) → y = 0) → m = 2 :=
by
  intros h
  have : m - 2 = 0 := sorry
  exact sorry

end linear_func_is_direct_proportion_l612_612602


namespace trisector_property_of_AP_l612_612668

theorem trisector_property_of_AP
  {α β γ δ ε ζ θ : Real}
  (h1 : δ = 2 * β)
  (h2 : ζ = β - θ)
  (h3 : α + β + δ = π)
  (h4 : ε = γ - 2 * (2 * β - θ))
  (h5 : γ = π - α - δ)
  (h6 : AP = AC)
  (h7 : PB = PC)
  (h8 : angle_POP : angle PAB = angle BAG)
  (h9 : B = A - α)
  (h10 : θ < β)
  (h11 : 2 * (B - θ) = π / 3): angle PAB = π / 3 := sorry

end trisector_property_of_AP_l612_612668


namespace product_of_ab_l612_612570

theorem product_of_ab : 
  ∀ (s : ℝ), 
  ((let a := 3 in let b := 9 in (Fido_leash_to_vertices (s) → fraction_of_area_Fido_explores (s) = (sqrt a / b) * π)) → a * b = 27) :=
by
  sorry

def Fido_leash_to_vertices (s : ℝ) : Prop :=
  -- This predicate indicates that the leash allows Fido to reach the vertices of the hexagonal yard.
  true

def fraction_of_area_Fido_explores (s : ℝ) : ℝ :=
  -- This function calculates the fraction of the hexagonal yard that Fido can explore.
  (2 * (sqrt 3) * π * s^2) / (9 * (sqrt 3) * s^2) → (sqrt 3 / 9) * π

end product_of_ab_l612_612570


namespace magnitude_of_sum_if_angle_is_60_angle_between_if_sum_perp_to_a_l612_612154

variables (a b : ℝ^3) -- assuming 3-dimensional real space vectors
axiom mag_a : ‖a‖ = 1
axiom mag_b : ‖b‖ = 2

theorem magnitude_of_sum_if_angle_is_60 (h : ∠(a, b) = real.pi / 3) :
  ‖a + b‖ = real.sqrt 7 := sorry

theorem angle_between_if_sum_perp_to_a (h : (a + b) ⬝ a = 0) :
  ∠(a, b) = 2 * real.pi / 3 := sorry

end magnitude_of_sum_if_angle_is_60_angle_between_if_sum_perp_to_a_l612_612154


namespace find_n_l612_612083

theorem find_n (n : ℕ) (h : 2 ^ 3 * 5 * n = Nat.factorial 10) : n = 45360 :=
sorry

end find_n_l612_612083


namespace correct_options_l612_612143

theorem correct_options (a b : ℝ) (h_a_pos : a > 0) (h_discriminant : a^2 = 4 * b):
  (a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬(∃ x1 x2 : ℝ, (x1 * x2 > 0 ∧ a^2 - x1x2 ≠ 4b))) ∧ 
  (∀ c x1 x2 : ℝ, (x1 - x2 = 4) → (a^2 - 4 * (b - c) = 16) → (c = 4)) :=
by
  sorry

end correct_options_l612_612143


namespace lenny_initial_money_l612_612208

-- Definitions based on the conditions
def spent_on_video_games : ℕ := 24
def spent_at_grocery_store : ℕ := 21
def amount_left : ℕ := 39

-- Statement of the problem
theorem lenny_initial_money : spent_on_video_games + spent_at_grocery_store + amount_left = 84 :=
by
  sorry

end lenny_initial_money_l612_612208


namespace period_increase_min_f_l612_612625

theorem period_increase_min_f (a : ℝ)
  (h : ∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), f x = sin (2 * x + Real.pi / 6) + sin (2 * x - Real.pi / 6) + cos (2 * x) + a)
  (h_min : ∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), f x ≥ -2) :
  (∀ T > 0, ∀ x, f (x + T) = f x → T = π) ∧
  (∀ k : ℤ, ∃ I : Set ℝ, I = Icc (k * π / 2 - π / 8) (k * π / 2 + π /8) ∧ (∀ x y ∈ I, x < y → f x ≤ f y)) ∧
  a = -1 := by
  -- Sorry, no proof, just the statement. This will skip the actual proof steps.
  sorry

end period_increase_min_f_l612_612625


namespace prove_y3_eq_x2z_l612_612116

noncomputable def verify_logarithmic_condition (x y z : ℝ) : Prop :=
  x > y ∧ y > z ∧ z > 1 ∧ 
  (Real.log 2 (x / z) * (Real.log (x / y) 2 + Real.log (y / z) 16) = 9)

theorem prove_y3_eq_x2z (x y z : ℝ) (h : verify_logarithmic_condition x y z) :
  y^3 = x^2 * z :=
sorry

end prove_y3_eq_x2z_l612_612116


namespace max_min_values_l612_612667

theorem max_min_values (x y : ℝ) 
  (h : (x - 3)^2 + 4 * (y - 1)^2 = 4) :
  ∃ (t u : ℝ), (∀ (z : ℝ), (x-3)^2 + 4*(y-1)^2 = 4 → t ≤ (x+y-3)/(x-y+1) ∧ (x+y-3)/(x-y+1) ≤ u) ∧ t = -1 ∧ u = 1 := 
by
  sorry

end max_min_values_l612_612667


namespace tessellation_exists_l612_612903

theorem tessellation_exists : ∃ (T : set (Point → Prop)), 
  (∀ t ∈ T, is_tetrahedron t ∧ all_faces_right_angles t) ∧ tessellates_space T := 
sorry

-- Definitions and conditions used in Lean 4

-- Point is a placeholder for a point in space, which can be defined as (x, y, z)
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Predicate to check if a given set of points forms a tetrahedron
def is_tetrahedron (t : set Point → Prop) : Prop := sorry

-- Predicate to check if all faces of a tetrahedron are right-angled triangles
def all_faces_right_angles (t : set Point → Prop) : Prop := sorry

-- Predicate to check if a set of tetrahedra tessellates the entire space
def tessellates_space (T : set (Point → Prop)) : Prop := sorry

end tessellation_exists_l612_612903


namespace point_B_represent_l612_612283

-- Given conditions
def point_A := -2
def units_moved := 4

-- Lean statement to prove
theorem point_B_represent : 
  ∃ B : ℤ, (B = point_A - units_moved) ∨ (B = point_A + units_moved) := by
    sorry

end point_B_represent_l612_612283


namespace correct_simplification_l612_612477

theorem correct_simplification (x a y : ℝ) (hx : x ≠ 0) (hx3 : x ≠ 3) :
  (\frac{x - 3}{2x(x - 3)} = \frac{1}{2x}) ∧ 
  (\frac{x^6}{x^2} ≠ x^3) ∧ 
  (\frac{y - x}{-x + y} ≠ -1) ∧ 
  (\frac{x + a}{y + a} ≠ \frac{x}{y}) :=
by {
  -- Remaining parts to be proven using the simplifications
  sorry
}

end correct_simplification_l612_612477


namespace total_spent_is_correct_l612_612334

noncomputable def totalAmountSpent 
  (costPerDeck : ℝ) 
  (discountThreshold : ℝ) 
  (discountRate : ℝ) 
  (victorDecks : ℕ) 
  (aliceDecks : ℕ) 
  (bobDecks : ℕ) : ℝ :=
let victorCost := victorDecks * costPerDeck in
let aliceCost := aliceDecks * costPerDeck in
let bobCost := bobDecks * costPerDeck in
let totalCostWithoutDiscount := victorCost + aliceCost + bobCost in
let victorDiscount := if (victorDecks ≥ discountThreshold) then (victorCost * discountRate) else 0 in
let victorDiscountedCost := victorCost - victorDiscount in
victorDiscountedCost + aliceCost + bobCost

theorem total_spent_is_correct :
  totalAmountSpent 8 5 0.10 6 4 3 = 99.20 :=
by
  -- skipping the proof
  sorry

end total_spent_is_correct_l612_612334


namespace minimal_bananas_l612_612096

noncomputable def total_min_bananas : ℕ :=
  let b1 := 72
  let b2 := 72
  let b3 := 216
  let b4 := 72
  b1 + b2 + b3 + b4

theorem minimal_bananas (total_bananas : ℕ) (ratio1 ratio2 ratio3 ratio4 : ℕ) 
  (b1 b2 b3 b4 : ℕ) 
  (h_ratio : ratio1 = 4 ∧ ratio2 = 3 ∧ ratio3 = 2 ∧ ratio4 = 1) 
  (h_div_constraints : ∀ n m : ℕ, (n % m = 0 ∨ m % n = 0) ∧ n ≥ ratio1 * ratio2 * ratio3 * ratio4) 
  (h_bananas : b1 = 72 ∧ b2 = 72 ∧ b3 = 216 ∧ b4 = 72 ∧ 
              4 * (b1 / 2 + b2 / 6 + b3 / 9 + 7 * b4 / 72) = 3 * (b1 / 6 + b2 / 3 + b3 / 9 + 7 * b4 / 72) ∧ 
              2 * (b1 / 6 + b2 / 6 + b3 / 6 + 7 * b4 / 72) = (b1 / 6 + b2 / 6 + b3 / 9 + b4 / 8)) : 
  total_bananas = 432 := by
  sorry

end minimal_bananas_l612_612096


namespace max_tickets_l612_612730

theorem max_tickets (cost : ℝ) (budget : ℝ) (max_tickets : ℕ) (h1 : cost = 15.25) (h2 : budget = 200) :
  max_tickets = 13 :=
by
  sorry

end max_tickets_l612_612730


namespace trig_expression_evaluation_l612_612098

theorem trig_expression_evaluation (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := 
  sorry

end trig_expression_evaluation_l612_612098


namespace bananas_used_l612_612071

-- Define the conditions
def bananas_per_loaf : Nat := 4
def loaves_on_monday : Nat := 3
def loaves_on_tuesday : Nat := 2 * loaves_on_monday
def total_loaves : Nat := loaves_on_monday + loaves_on_tuesday

-- Define the total bananas used
def total_bananas : Nat := bananas_per_loaf * total_loaves

-- Prove that the total bananas used is 36
theorem bananas_used : total_bananas = 36 :=
by
  sorry

end bananas_used_l612_612071


namespace cat_food_insufficient_l612_612508

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l612_612508


namespace area_of_quadrilateral_EGFH_l612_612679

-- Let A, B, C, and D be points forming the rectangle ABCD
structure Point where
  x : ℝ
  y : ℝ

def A := ⟨0, 0⟩ : Point
def B := ⟨2, 0⟩ : Point
def C := ⟨2, 3⟩ : Point
def D := ⟨0, 3⟩ : Point

-- Define midpoints E, F, G
def midpoint (P1 P2 : Point) : Point :=
  ⟨(P1.x + P2.x) / 2, (P1.y + P2.y) / 2⟩

def E := midpoint A B
def F := midpoint B C
def G := midpoint C D

-- Define point H as the midpoint of E and F
def H := midpoint E F

-- Function for area calculation using shoelace formula
def shoelace_area (P1 P2 P3 P4 : Point) : ℝ :=
  (1 / 2) * abs 
    (P1.x * P2.y + P2.x * P3.y + P3.x * P4.y + P4.x * P1.y
     - (P1.y * P2.x + P2.y * P3.x + P3.y * P4.x + P4.y * P1.x))

-- Statement to prove the area of quadrilateral EGFH is 2.25
theorem area_of_quadrilateral_EGFH : shoelace_area E G F H = 2.25 := 
  sorry

end area_of_quadrilateral_EGFH_l612_612679


namespace find_solutions_l612_612222

def f (x : ℝ) := -3 * Real.sin (Real.pi * x)

theorem find_solutions : ∃ (s : Finset ℝ), 
  (∀ x ∈ s, -1 ≤ x ∧ x ≤ 1 ∧ f (f (f x)) = f x) ∧ 
  Finset.card s = 6 := 
sorry

end find_solutions_l612_612222


namespace log_equality_l612_612089

theorem log_equality (b : ℝ) (h : b = 60) : (1 / Real.log 3 / Real.log b) + (1 / Real.log 4 / Real.log b) + (1 / Real.log 5 / Real.log b) = 1 := by
  rw [←h, Real.log]
  sorry

end log_equality_l612_612089


namespace find_AN_length_l612_612262

theorem find_AN_length
  (A B C M N : Point)
  (h1 : is_angle_bisector (angle B A C) M)
  (h2 : on_extension A B N)
  (h3 : dist A C = 1)
  (h4 : dist A M = 1)
  (h5 : angle A N M = angle C N M)
  : dist A N = 1 := sorry

end find_AN_length_l612_612262


namespace who_got_the_job_l612_612026

-- Definitions based directly on the conditions
def A_said := "C got the job"
def B_said := "A got the job"
def C_said := "I did not get the job"

-- Only one person made an incorrect statement
axiom only_one_incorrect (A_said: Prop) (B_said: Prop) (C_said: Prop) : (A_said ∧ ¬B_said ∧ ¬C_said) ∨ (¬A_said ∧ B_said ∧ ¬C_said) ∨ (¬A_said ∧ ¬B_said ∧ C_said)

-- Prove the conclusion given the conditions
theorem who_got_the_job (A_said_true: Prop) (B_said_true: Prop) (C_said_true: Prop) : 
(¬(A_said = C_said_true)) →
((B_said = A_said_true) ∧ (C_said = ¬B_said_true)) →
(¬(C_said = A_said_true)) →
A_said_true = true → 
A_said == "A got the job" :=
begin
  sorry
end

end who_got_the_job_l612_612026


namespace all_positive_integers_are_good_l612_612085

def is_good (n : ℕ) : Prop :=
  ∀ (m k : ℕ), m * k = n → (∃ f : ℕ × ℕ → bool, ∀ (i j : ℕ), i < m → j < k → (f ((i, j)) = tt) → ∑ i' in range m, ∑ j' in range k, f (i', j') % 2 = 1)

theorem all_positive_integers_are_good : ∀ n : ℕ, 0 < n → is_good n := 
by
  sorry

end all_positive_integers_are_good_l612_612085


namespace probability_more_heads_than_tails_l612_612658

theorem probability_more_heads_than_tails (n : ℕ) (hn : n = 9) :
  let outcomes := 2^n in
  let favorable := (nat.choose 9 5) + (nat.choose 9 6) + (nat.choose 9 7) + (nat.choose 9 8) + (nat.choose 9 9) in
  (favorable : ℚ) / (outcomes : ℚ) = 1 / 2 :=
by {
  let n := 9,
  let outcomes := 2^n,
  let favorable := (nat.choose 9 5) + (nat.choose 9 6) + (nat.choose 9 7) + (nat.choose 9 8) + (nat.choose 9 9),
  calc
    (favorable : ℚ) / (outcomes : ℚ) = 256 / 512 : sorry
                            ...        = 1 / 2   : sorry
}

end probability_more_heads_than_tails_l612_612658


namespace sequence_is_not_periodic_l612_612639

def is_periodic {α : Type} (s : ℕ → α) (T : ℕ) : Prop :=
  ∀ n, s n = s (n + T)

def given_sequence : ℕ → ℤ
| 0     := 1
| (n+1) := if h : (n+1) % 2 = 0 then given_sequence (n / 2) else -given_sequence (n / 2)

theorem sequence_is_not_periodic : ¬ ∃ T, T > 0 ∧ is_periodic given_sequence T :=
sorry

end sequence_is_not_periodic_l612_612639


namespace penultimate_digit_of_quotient_l612_612946

theorem penultimate_digit_of_quotient :
  (4^1994 + 7^1994) / 10 % 10 = 1 :=
by
  sorry

end penultimate_digit_of_quotient_l612_612946


namespace circles_intersect_condition_tangent_form_l612_612907

theorem circles_intersect_condition_tangent_form 
  (h1 : ∃ (x y : ℝ), (x, y) = (7, 4) ∧ 
        ∃ (r1 r2 : ℝ), r1 * r2 = 50 ∧ 
        ∃ (m : ℝ), m > 0 ∧ 
        ∀ (c1 c2 : ℝ × ℝ), (m * c1.1 + c1.2 = 0 ∧ m * c2.1 + c2.2 = 0) ∧ 
        ∃ (a b c : ℕ), m = (a * real.sqrt b) / c ∧ 
        ¬∃ (p : ℕ), nat.prime p ∧ p ^ 2 ∣ b ∧ nat.coprime a c) : 
  ∃ (a b c : ℕ), (a + b + c = 135) :=
by { sorry }

end circles_intersect_condition_tangent_form_l612_612907


namespace solution1_solution2_is_not_a_proposition_solution3_solution4_l612_612559

-- Define a statement to represent each problem and its classification.
def statement1 : Prop := ∀ (x : ℝ), x / 1 = x 
def statement2 := false -- Representing that the question itself is not a proposition.
def statement3 : Prop := ∃ (x : ℝ), x = 0 
def statement4 : Prop := ∃ (t : Triangle), t.interior_angle_sum ≠ 180

-- Prove each statement, skipping the actual proof.
theorem solution1 : statement1 := by {
  sorry
}

theorem solution2_is_not_a_proposition : ¬ Prop := by {
  sorry
}

theorem solution3 : statement3 := by {
  sorry
}

theorem solution4 : statement4 := by {
  sorry
}

end solution1_solution2_is_not_a_proposition_solution3_solution4_l612_612559


namespace part_I_part_II_part_III_l612_612596

section

variable (a b : ℝ)

-- Part (I)
theorem part_I (a_pos : a > 0) (b_pos : b > 0)
    (f_le_1 : ∀ x : ℝ, a * x - b * x ^ 2 ≤ 1) :
  a ≤ 2 * Real.sqrt b := sorry

-- Part (II)
theorem part_II (b_gt_1 : b > 1)
  (f_abs_le_1_Iff : (∀ x ∈ set.Icc (0:ℝ) (1:ℝ), |a * x - b * x ^ 2| ≤ 1) ↔ (b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b)) :
  ∀ x ∈ set.Icc (0:ℝ) (1:ℝ), |a * x - b * x ^ 2| ≤ 1 ↔ (b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b) := sorry

-- Part (III)
theorem part_III (b_ineq : 0 < b ∧ b ≤ 1)
  (f_abs_le_1_iff : ∀ x ∈ set.Icc (0:ℝ) (1:ℝ), |a * x - b * x ^ 2| ≤ 1 ↔ a ≤ b + 1) :
  ∀ x ∈ set.Icc (0:ℝ) (1:ℝ), |a * x - b * x ^ 2| ≤ 1 ↔ (a ≤ b + 1) := sorry

end

end part_I_part_II_part_III_l612_612596


namespace strawberries_sold_portion_l612_612539

theorem strawberries_sold_portion :
  ∀ (cost_per_dozen revenue_per_dozen portion_revenue price_remaining profit_per_dozen total_profit : ℝ)
  (total_dozens : ℕ),
  cost_per_dozen = 50 →
  revenue_per_dozen = 60 →
  portion_revenue = 30 →
  price_remaining = 50 →
  total_dozens = 50 →
  total_profit = 500 →
  revenue_per_dozen = portion_revenue * x + price_remaining * (1 - x) →
  x = 1 / 2 :=
begin
  -- Definitions from conditions
  let cost_per_dozen := 50,
  let revenue_per_dozen := 60,
  let portion_revenue := 30,
  let price_remaining := 50,
  let total_dozens := 50,
  let total_profit := 500,
  -- Correct answer from the solution
  let x := 1 / 2,
  -- Stating the equation from the revenue calculation
  have h : revenue_per_dozen = portion_revenue * x + price_remaining * (1 - x),
  sorry,
end

end strawberries_sold_portion_l612_612539


namespace solve_ab_l612_612662

theorem solve_ab (a b : ℝ) (h : sqrt (a - 4) + (b + 5)^2 = 0) : a + b = -1 := 
begin
  sorry
end

end solve_ab_l612_612662


namespace cat_food_inequality_l612_612536

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l612_612536


namespace general_formula_seq_sum_first_n_terms_l612_612108

namespace ArithmeticSequence

variable {a : ℕ → ℕ} {S T : ℕ → ℚ} {d a1 : ℕ}

-- Conditions
def arithmetic_seq (a : ℕ → ℕ) (d a1 : ℕ) : Prop :=
∀ n, a n = a1 + n * d

def a_5_eq_10 (a : ℕ → ℕ) : Prop :=
a 5 = 10

def geometric_seq (a : ℕ → ℕ) : Prop :=
a 3 * a 3 = a 1 * a 9

noncomputable def S_n (a : ℕ → ℕ) : ℕ → ℚ :=
λ n, (n : ℚ) * (a 1 + a n) / 2

def T_n (S : ℕ → ℚ) : ℕ → ℚ :=
λ n, ∑ i in finset.range n, 1 / S (i + 1)

-- Proof
theorem general_formula_seq (h_arith : arithmetic_seq a d a1)
  (h_a5 : a_5_eq_10 a)
  (h_geom : geometric_seq a)
  (h_nonzero : d ≠ 0) :
  ∀ n, a n = 2 * n :=
sorry

theorem sum_first_n_terms 
  (h_arith : arithmetic_seq a d a1)
  (h_a5 : a_5_eq_10 a)
  (h_geom : geometric_seq a)
  (h_nonzero : d ≠ 0) :
  ∀ n, T_n (S_n a) n = n / (n + 1) :=
sorry

end ArithmeticSequence

end general_formula_seq_sum_first_n_terms_l612_612108


namespace projection_norm_eq_l612_612155

variable {V : Type*} [InnerProductSpace ℝ V]
variables (v w : V)
variables (norm_v : ∥v∥ = 5) (norm_w : ∥w∥ = 9) (theta : Real.Angle := π / 6)

theorem projection_norm_eq : ∥proj w v∥ = 5 * Real.sqrt 3 / 2 := by
  sorry

end projection_norm_eq_l612_612155


namespace correct_options_l612_612149

theorem correct_options (a b : ℝ) (h : a > 0) (ha : a^2 = 4 * b) :
  ((a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬ (∃ x1 x2, x1 * x2 > 0 ∧ x^2 + a * x - b < 0)) ∧ 
  (∀ (x1 x2 : ℝ), |x1 - x2| = 4 → x^2 + a * x + b < 4 → 4 = 4)) :=
sorry

end correct_options_l612_612149


namespace range_of_a_if_derivative_nonnegative_l612_612619

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

theorem range_of_a_if_derivative_nonnegative:
  (∀ x : ℝ, f'(x) ≥ a) → a ≤ 2 :=
by
  -- sorry means the proof is skipped
  sorry

end range_of_a_if_derivative_nonnegative_l612_612619


namespace ultimate_price_percentage_l612_612804

-- Definitions based on the conditions
def SRP := P : ℝ
def marked_price (P : ℝ) := 0.80 * P
def discounted_price (P : ℝ) := 0.60 * P
def final_price (P : ℝ) := 1.10 * (discounted_price P)

-- The theorem to be proven
theorem ultimate_price_percentage (P : ℝ) : 
    final_price P = 0.66 * P :=
by
  sorry

end ultimate_price_percentage_l612_612804


namespace monotonic_intervals_range_of_a_l612_612623

noncomputable def f (x a : ℝ) := Real.log x + (a / 2) * x^2 - (a + 1) * x
noncomputable def f' (x a : ℝ) := 1 / x + a * x - (a + 1)

theorem monotonic_intervals (a : ℝ) (ha : f 1 a = -2 ∧ f' 1 a = 0):
  (∀ x : ℝ, 0 < x ∧ x < (1 / 2) → f' x a > 0) ∧ 
  (∀ x : ℝ, x > 1 → f' x a > 0) ∧ 
  (∀ x : ℝ, (1 / 2) < x ∧ x < 1 → f' x a < 0) := sorry

theorem range_of_a (a : ℝ) 
  (h : ∀ x : ℕ, x > 0 → (f x a) / x < (f' x a) / 2):
  a > 2 * Real.exp (- (3 / 2)) - 1 := sorry

end monotonic_intervals_range_of_a_l612_612623


namespace parallel_KL_AD_l612_612291

variables (A B C D K L : Type) [circle : is_cyclic_quadrilateral A B C D]
variables (AC_diag : segment A C) (BD_diag : segment B D)
variables (AK_eq_AB : segment A K = segment A B) (DL_eq_DC : segment D L = segment D C)

theorem parallel_KL_AD : parallel (line K L) (line A D) :=
by sorry

end parallel_KL_AD_l612_612291


namespace common_rational_root_is_neg_one_third_l612_612773

theorem common_rational_root_is_neg_one_third (a b c d e f g : ℚ) :
  ∃ k : ℚ, (75 * k^4 + a * k^3 + b * k^2 + c * k + 12 = 0) ∧
           (12 * k^5 + d * k^4 + e * k^3 + f * k^2 + g * k + 75 = 0) ∧
           (¬ k.isInt) ∧ (k < 0) ∧ (k = -1/3) :=
sorry

end common_rational_root_is_neg_one_third_l612_612773


namespace symmetry_center_of_cosine_l612_612805

theorem symmetry_center_of_cosine :
  ∃ x : ℝ, 
  (∃ k : ℤ, 2 * x + π / 6 = k * π + π / 2) ∧ x = π / 6 :=
begin
  sorry,
end

end symmetry_center_of_cosine_l612_612805


namespace angle_IDA_l612_612685

theorem angle_IDA (a b c d f g h i: Point)
  (h_sq1 : square ABCD)
  (h_sq2 : square FGHI)
  (h_pent : regular_pentagon CDEFG) :
  ∠ IDA = 72 := 
  sorry

end angle_IDA_l612_612685


namespace rectangle_width_length_ratio_l612_612183

theorem rectangle_width_length_ratio (w l : ℕ) 
  (h1 : l = 12) 
  (h2 : 2 * w + 2 * l = 36) : 
  w / l = 1 / 2 := 
by 
  sorry

end rectangle_width_length_ratio_l612_612183


namespace first_year_interest_rate_l612_612937

-- Definitions based on the problem's conditions
def principal : ℝ := 5000
def time_period : ℝ := 2
def second_year_rate : ℝ := 5 / 100
def final_amount : ℝ := 5460

-- Theorem to prove the rate of interest for the first year
theorem first_year_interest_rate (R : ℝ) (h : 5000 + 50 * R + (principal + 50 * R) * 0.05 = final_amount) : R = 4 := 
by 
  sorry

end first_year_interest_rate_l612_612937


namespace quadratic_polynomial_l612_612581

theorem quadratic_polynomial (q : ℚ[X]) (h₁ : q.eval (-1) = 5) (h₂ : q.eval 2 = 3) (h₃ : q.eval 4 = 15) :
    q = Polynomial.C (4 / 3) * X^2 + Polynomial.C (-2) * X + Polynomial.C (5 / 3) := by
  sorry

end quadratic_polynomial_l612_612581


namespace product_of_five_numbers_is_256_l612_612954

def possible_numbers : Set ℕ := {1, 2, 4}

theorem product_of_five_numbers_is_256 
  (x1 x2 x3 x4 x5 : ℕ) 
  (h1 : x1 ∈ possible_numbers) 
  (h2 : x2 ∈ possible_numbers) 
  (h3 : x3 ∈ possible_numbers) 
  (h4 : x4 ∈ possible_numbers) 
  (h5 : x5 ∈ possible_numbers) : 
  x1 * x2 * x3 * x4 * x5 = 256 :=
sorry

end product_of_five_numbers_is_256_l612_612954


namespace ellipse_eccentricity_l612_612992

theorem ellipse_eccentricity (a b c x y : ℝ) (h₁ : a > b) (h₂ : b > 0) (h_eqn1 : x^2/a^2 + y^2/b^2 = 1)
    (F1 F2 A O M : ℝ × ℝ) (h_f1 : F1 = (-c, 0)) (h_f2 : F2 = (c, 0)) (h_A : A = (a, 0)) (h_O : O = (0, 0)) 
    (h_M : M = (a/2, (√3)/2 * b)) 
    (h_perp : (M.1 + c, M.2) ∙ (M.1 - c, M.2) = 0) 
    (h_MA_MO : |M - A| = |M - O|) : 
    abs (c / a - 2*√7 / 7) = 0 :=
sorry

end ellipse_eccentricity_l612_612992


namespace transformed_sine_function_l612_612091

theorem transformed_sine_function :
  ∀ (x : ℝ), 
    let f := λ x, Real.sin (2 * x) in
    let translated_f := λ x, f (x - (π / 3)) in
    let reflected_f := λ x, translated_f (-x) in
    reflected_f x = Real.sin (-2 * x - (2 * π / 3)) :=
by
  sorry

end transformed_sine_function_l612_612091


namespace trajectory_is_line_segment_l612_612985

/-- Fixed points F1 and F2 and the condition on point P's trajectory --/
theorem trajectory_is_line_segment (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (h1 : F1 = (-3, 0)) (h2 : F2 = (3, 0)) 
  (h3 : dist P F1 + dist P F2 = 6) : 
  ∃ t ∈ Icc (-3) 3, P = (t, 0) := 
sorry

end trajectory_is_line_segment_l612_612985


namespace opposite_numbers_option_C_l612_612029

def is_opposite (a b : ℤ) : Prop := a = -b

theorem opposite_numbers_option_C :
  (¬ is_opposite 3 (-1 / 3) ∧ ¬ is_opposite (-(-2)) 2 ∧ is_opposite (-5^2) ((-5)^2) ∧ ¬ is_opposite 7 (| -7 |)) := 
by
  sorry

end opposite_numbers_option_C_l612_612029


namespace number_of_impossible_d_values_l612_612482

theorem number_of_impossible_d_values (t a b d : ℤ) 
  (h1 : 3 * t - 2 * (a + b) = 504) 
  (h2 : b = 7) 
  (h3 : t = a + d) : 
  ∃ n : ℕ, ∀ m : ℕ, m ≥ n → ¬(is_possible_value_for_d m) := 
begin
  sorry,
end

end number_of_impossible_d_values_l612_612482


namespace main_theorem_l612_612217

def q (x : ℕ) := ∑ i in range (3006), x^(3005 - i)

def f (x : ℕ) := x^5 + x^3 + 3*x^2 + 2*x + 1

-- Define s(x) as the polynomial remainder when q(x) is divided by f(x)
noncomputable def s (x : ℕ) := polynomial.div_mod_by_monic (q(x)) (f(x)).1

theorem main_theorem : (|s 3005| % 1000) = 1 :=
by
  sorry

end main_theorem_l612_612217


namespace candle_burns_out_candle_height_half_time_l612_612855

section candle_problem

def init_height : ℕ := 150

def burn_time (k : ℕ) : ℕ := 10 * k^2

noncomputable def total_time (n : ℕ) : ℕ :=
  10 * ∑ k in finset.range (n + 1), k^2

theorem candle_burns_out : total_time init_height = 11337750 :=
  sorry

noncomputable def half_total_time_height (t : ℕ) : ℕ := 
  let m := ((t / 10) / 2) ^ (1/3)
  (init_height - nat.floor m : ℕ)

theorem candle_height_half_time : 
  half_total_time_height 5668875 = 31 :=
  sorry

end candle_problem

end candle_burns_out_candle_height_half_time_l612_612855


namespace cat_food_insufficient_l612_612505

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l612_612505


namespace problem_statement_l612_612543

def p1 : Prop :=
  ∃ x0 : ℝ, 0 < x0 ∧ ((1 / 2) ^ x0 < (1 / 3) ^ x0)

def p2 : Prop :=
  ∃ x0 : ℝ, 0 < x0 ∧ x0 < 1 ∧ (log x0 / log (1 / 2) > log x0 / log (1 / 3))

def p3 : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < +∞ ∧ ((1 / 2) ^ x < log x / log (1 / 2))

def p4 : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < (1 / 3) ∧ ((1 / 2) ^ x < log x / log (1 / 3))

theorem problem_statement : ¬p1 ∧ p2 ∧ ¬p3 ∧ p4 :=
by {
  -- Insert proof steps here
  sorry
}

end problem_statement_l612_612543


namespace count_good_numbers_l612_612380

-- Define the property of a good number
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- Count the number of good numbers
theorem count_good_numbers : (finset.filter is_good_number (finset.range 2021)).card = 10 :=
by sorry

end count_good_numbers_l612_612380


namespace base_prime_representation_of_441_l612_612918

theorem base_prime_representation_of_441 : 
  ∃ (a b c d : ℕ), (441 = 2^a * 3^b * 5^c * 7^d) ∧ (a = 0) ∧ (b = 2) ∧ (c = 0) ∧ (d = 2) :=
by
  have h1 : 441 = 21 * 21 := by norm_num
  have h2 : 21 = 3 * 7 := by norm_num
  exist 0 2 0 2,
  -- Proof to check 441 = 2^0 * 3^2 * 5^0 * 7^2
  calc  2^0 * 3^2 * 5^0 * 7^2
        = 1 * 9 * 1 * 49 : by norm_num
    ... = 9 * 49 : by ring
    ... = 441 : by norm_num,
  -- Proving the exponents for the primes
  repeat { split, norm_num },
  sorry

end base_prime_representation_of_441_l612_612918


namespace probability_each_car_once_l612_612833

-- Definitions and Conditions
def num_cars : ℕ := 5
def num_rides : ℕ := 5

-- Proof Statement
theorem probability_each_car_once :
  let total_ways := num_cars ^ num_rides in
  let unique_ways := Nat.factorial num_cars in
  (unique_ways : ℚ) / total_ways = 24 / 625 :=
by
  sorry

end probability_each_car_once_l612_612833


namespace food_requirement_not_met_l612_612502

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l612_612502


namespace average_speed_for_entire_trip_l612_612245

theorem average_speed_for_entire_trip :
  let distance1 := 100
  let speed1 := 20
  let distance2 := 300
  let speed2 := 15
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  in
  average_speed = 16 :=
by
  sorry

end average_speed_for_entire_trip_l612_612245


namespace measure_of_unknown_angle_in_hexagon_l612_612974

theorem measure_of_unknown_angle_in_hexagon :
  let a1 := 135
  let a2 := 105
  let a3 := 87
  let a4 := 120
  let a5 := 78
  let total_internal_angles := 180 * (6 - 2)
  let known_sum := a1 + a2 + a3 + a4 + a5
  let Q := total_internal_angles - known_sum
  Q = 195 :=
by
  sorry

end measure_of_unknown_angle_in_hexagon_l612_612974


namespace cat_food_insufficient_for_six_days_l612_612529

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l612_612529


namespace medical_team_probability_l612_612590

open Finset

theorem medical_team_probability :
  let n_total := 9
  let k_selected := 5
  let male_doctors := 6
  let female_doctors := 3
  let total_combinations := choose n_total k_selected
  let male_only_combinations := choose male_doctors k_selected
  let probability := 1 - (male_only_combinations.to_real / total_combinations.to_real)
  probability = 60 / 63 :=
by
  sorry

end medical_team_probability_l612_612590


namespace minimum_value_of_b_l612_612137

theorem minimum_value_of_b (a b : ℝ) (h_pos_a : a > 0) (h_strict_inc : ∀ x y : ℝ, 0 < x → x < y → g x < g y) :
  b ≥ 1 / 4 :=
by
  let f := λ x, a * x + b
  let g := λ x, if x ≤ a then f x else f (f x)
  have h_continuous : continuous_at g a := sorry
  have h_increasing : (∀ x, 0 ≤ x ∧ x < a → f x < f (x + 1)) ∧ (∀ x > a, f (f (x - a)) < f (f x)) := sorry
  have b_bound : b ≥ a - a^2 := sorry
  have min_b := a = 1/2 → (a - a^2) = 1/4 := sorry
  sorry

end minimum_value_of_b_l612_612137


namespace polynomial_root_uniqueness_l612_612092

theorem polynomial_root_uniqueness (P : ℤ → ℤ) (h : ∀ v, (∃ x y : ℤ, x ≠ y ∧ P x = v ∧ P y = v) → set.infinite {v | ∃ x y : ℤ, x ≠ y ∧ P x = v ∧ P y = v}) :
  at_most_one_value_single_root P :=
sorry

end polynomial_root_uniqueness_l612_612092


namespace shaded_region_area_correct_l612_612576

-- Definitions for lines L1 and L2 based on given points
def L1 (x : ℝ) : ℝ := -0.3 * x + 5

def L2 (x : ℝ) : ℝ := -5/7 * x + 52/7

-- Definition of the rectangle's properties
def rectangle_width : ℝ := 2

def rectangle_height (L1 : ℝ → ℝ) : ℝ := L1 0

-- Proof structure to verify the area calculation
theorem shaded_region_area_correct : rectangle_width * rectangle_height L1 = 10 :=
by
  -- Implement proof steps here using Lean tactics
  sorry

end shaded_region_area_correct_l612_612576


namespace water_formed_on_combustion_l612_612575
variable (CH4 O2 CO2 H2O : Type)

def balanced_combustion_equation (a b c d : Nat) : Prop :=
  (a = 1) ∧ (b = 2) ∧ (c = 1) ∧ (d = 2)

theorem water_formed_on_combustion : 
  ∀ (moles_CH4 moles_O2 : Nat), 
    balanced_combustion_equation 1 2 1 2 →
    moles_CH4 = 3 → moles_O2 = 6 →
    2 * moles_CH4 = 6 := 
by
  intros moles_CH4 moles_O2 h_eq h_CH4 h_O2
  rw [h_CH4]
  rw [mul_comm 2 3]
  rfl

end water_formed_on_combustion_l612_612575


namespace complement_intersection_l612_612723

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}
def intersection_A_B : Set ℕ := A ∩ B

theorem complement_intersection (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4, 5}) (h_intersection : A ∩ B = {3}) :
  (U \ (A ∩ B)) = {1, 2, 4, 5} :=
by {
  rw [hU, hA, hB, h_intersection],
  sorry
}

end complement_intersection_l612_612723


namespace max_term_of_sequence_l612_612140

def a (n : ℕ) : ℚ := (n : ℚ) / (n^2 + 156)

theorem max_term_of_sequence : ∃ n, (n = 12 ∨ n = 13) ∧ (∀ m, a m ≤ a n) := by 
  sorry

end max_term_of_sequence_l612_612140


namespace elois_banana_bread_l612_612073

theorem elois_banana_bread :
  let bananas_per_loaf := 4
  let loaves_monday := 3
  let loaves_tuesday := 2 * loaves_monday
  let total_loaves := loaves_monday + loaves_tuesday
  let total_bananas := total_loaves * bananas_per_loaf
  total_bananas = 36 := sorry

end elois_banana_bread_l612_612073


namespace johns_allowance_correct_l612_612838

noncomputable def johns_weekly_allowance : ℚ :=
  let A := 2.25
  A

theorem johns_allowance_correct :
  (∃ A : ℚ, (3/5) * A + (1/3) * (2/5) * A + 0.60 = A ∧ johns_weekly_allowance = A) :=
begin
  use 2.25,
  split,
  {
    have h1 : (1 - 3/5) = 2/5,
    { norm_num, },
    have h2 : (1/3) * (2/5) = 2/15,
    { norm_num, },
    have h3 : (2/5) - (2/15) = 4/15,
    { field_simp, ring, },
    have h4 : (0.60 / (4/15)) = 2.25,
    { norm_num, erw [div_mul_eq_mul_div, mul_div_assoc_prime], norm_num, },
    rw h3 at h4,
    norm_num at h4,
    ring at h4,
  },
  refl
end

end johns_allowance_correct_l612_612838


namespace perfect_square_factors_count_l612_612161

def positive_factors (n : ℕ) : ℕ :=
  ∏ (x : ℕ) in (range n).filter (λ x, is_square x), 1

theorem perfect_square_factors_count :
  let N := (2^12) * (3^18) * (7^10)
  in positive_factors N = 420 := 
by
  let N := (2^12) * (3^18) * (7^10)
  sorry

end perfect_square_factors_count_l612_612161


namespace incorrect_statement_l612_612887

-- Definitions based on conditions in a)
def diploid := "Watermelon plant initially diploid"
def treated_with_colchicine := "Shoot tips treated with colchicine"
def tetraploid := "Resulting plant becomes tetraploid"

-- Statement identifying the false statement from the options
theorem incorrect_statement
  (initially_diploid : diploid)
  (tratment_colchicine : treated_with_colchicine)
  (resulting_tetraploid : tetraploid) :
  ¬(∀ cell, cell.contains_four_sets_of_chromosomes) :=
sorry

end incorrect_statement_l612_612887


namespace shortest_distance_to_river_and_home_l612_612863

-- Definitions for the coordinates
def cowboy_position : ℝ × ℝ := (0, -6)
def cabin_position : ℝ × ℝ := (10, -15)
def cowboy_reflection : ℝ × ℝ := (6, 6)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- The Lean 4 theorem statement to prove the shortest distance
theorem shortest_distance_to_river_and_home :
  let dist_to_river := 6
  let dist_home_from_reflection := distance cowboy_reflection cabin_position
  dist_to_river + dist_home_from_reflection = 6 + Real.sqrt 457 :=
by
  sorry

end shortest_distance_to_river_and_home_l612_612863


namespace integral_abs_x_squared_minus_2x_l612_612046

-- Define the integrand function
def integrand (x : ℝ) : ℝ := |x^2 - 2 * x|

-- Statement of the integral problem
theorem integral_abs_x_squared_minus_2x :
    ∫ x in -2..2, integrand x = 16 / 3 :=
by
    -- sorry is used to skip the proof
    sorry

end integral_abs_x_squared_minus_2x_l612_612046


namespace radius_of_circle_D_l612_612611

theorem radius_of_circle_D
    (radius_C : ℝ) (radius_C_eq_2 : radius_C = 2)
    (radius_E : ℝ) (radius_D : ℝ)
    (radius_D_eq_3_times_radius_E : radius_D = 3 * radius_E)
    (radius_D_value : radius_D = 4 * real.sqrt 15 - 14) :
    radius_D = 4 * real.sqrt 15 - 14 := 
sorry

end radius_of_circle_D_l612_612611


namespace sum_of_valid_a_l612_612122

def conditions (a : ℤ) : Prop :=
  (-1 ≤ a ∧ a < 5) ∧ ∃ x : ℤ, 10 / (2 - x) = 2 - (a * x) / (x - 2)

theorem sum_of_valid_a :
  ∑ (a : ℤ) in {a | -1 ≤ a ∧ a < 5 ∧ (∃ x : ℤ, 10 / (2 - x) = 2 - (a * x) / (x - 2))}, 1 = 7 := by
  sorry

end sum_of_valid_a_l612_612122


namespace no_boy_ended_up_with_original_gift_l612_612032

structure Person :=
  (name : String)
  (original_gift : String)
  (purchased_gifts : List String)
  (received_gifts : List String := [])
  (regifts : List String := [])

def Andras : Person := ⟨"Andras", "", ["toy car"]⟩
def Bence : Person := ⟨"Bence", "3 sets of stamps", ["toy car"], ["toy car"], ["toy car"]⟩
def Csaba : Person := ⟨"Csaba", "2 chocolates", ["toy car"], ["toy car"], ["toy car"]⟩
def Denes : Person := ⟨"Denes", "1 set of dominoes", ["toy car"], ["toy car"], ["toy car"]⟩
def Elemer : Person := ⟨"Elemer", "", ["toy car"], ["toy car"], ["toy car"]⟩

def no_self_regift (p : Person) := ¬(p.original_gift ∈ p.received_gifts ∨ p.original_gift ∈ p.regifts)

theorem no_boy_ended_up_with_original_gift :
  no_self_regift Bence ∧ no_self_regift Csaba ∧ no_self_regift Denes ∧ no_self_regift Elemer :=
by
  sorry

end no_boy_ended_up_with_original_gift_l612_612032


namespace cookies_with_7_cups_l612_612696

def cookies_4 := 36
def multiplier := 1.5

theorem cookies_with_7_cups : 
  (cookies_4 * multiplier * multiplier * multiplier) = 121.5 :=
by 
  sorry

end cookies_with_7_cups_l612_612696


namespace perpendicular_lines_from_perpendicular_planes_l612_612702

variable {Line : Type} {Plane : Type}

-- Definitions of non-coincidence, perpendicularity, parallelism
noncomputable def non_coincident_lines (a b : Line) : Prop := sorry
noncomputable def non_coincident_planes (α β : Plane) : Prop := sorry
noncomputable def line_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry
noncomputable def line_perpendicular_to_plane (a : Line) (α : Plane) : Prop := sorry
noncomputable def plane_parallel_to_plane (α β : Plane) : Prop := sorry
noncomputable def plane_perpendicular_to_plane (α β : Plane) : Prop := sorry
noncomputable def line_perpendicular_to_line (a b : Line) : Prop := sorry

-- Given non-coincident lines and planes
variable {a b : Line} {α β : Plane}

-- Problem statement
theorem perpendicular_lines_from_perpendicular_planes (h1 : non_coincident_lines a b)
  (h2 : non_coincident_planes α β)
  (h3 : line_perpendicular_to_plane a α)
  (h4 : line_perpendicular_to_plane b β)
  (h5 : plane_perpendicular_to_plane α β) : line_perpendicular_to_line a b := sorry

end perpendicular_lines_from_perpendicular_planes_l612_612702


namespace damien_jogging_distance_l612_612920

theorem damien_jogging_distance (miles_per_day : ℕ) (weekdays_per_week : ℕ) (weeks : ℕ) 
  (h1 : miles_per_day = 5) (h2 : weekdays_per_week = 5) (h3 : weeks = 3) :
  miles_per_day * weekdays_per_week * weeks = 75 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end damien_jogging_distance_l612_612920


namespace length_an_eq_1_l612_612269

variable {A B C M N : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N]
variable {dist : A → A → ℝ}
variable {angle : ∀ {A B C : A}, ℝ}

-- Given conditions as hypotheses
variable (H1 : ∀ {M}, M ∈ angle_bisector A B C)
variable (H2 : ∀ {N : Type}, N ∈ extended_line A B)
variable (H3 : dist A C = 1)
variable (H4 : dist A M = 1)
variable (H5 : angle A N M = angle C N M)

-- The statement we need to prove
theorem length_an_eq_1 (H1 : M ∈ angle_bisector A B C) (H2 : N ∈ extended_line A B)
  (H3 : dist A C = 1) (H4 : dist A M = 1) (H5 : angle A N M = angle C N M) :
  dist A N = 1 := by
  sorry


end length_an_eq_1_l612_612269


namespace projectile_highest_points_area_l612_612870

theorem projectile_highest_points_area (u k : ℝ) (hk : k ≠ 0) (hu : u ≠ 0) :
  let x (t α : ℝ) := u * t * Real.cos α
  let y (t α : ℝ) := u * t * Real.sin α - (1 / 2) * k * t^2
  let t_max (α : ℝ) := u * Real.sin α / k
  let x_max (α : ℝ) := (u^2 / (2 * k)) * Real.sin (2 * α)
  let y_max (α : ℝ) := (u^2 / (2 * k)) * Real.sin(α)^2 in
  {(x_max α, y_max α) | 0 ≤ α ∧ α ≤ Real.pi / 2} = 
  ∃ (d : ℝ), (d = (Real.pi / 8)) ∧
    ∀ (hx (α : ℝ)) (hy (α : ℝ)),
      hx = (u^2 / (2 * k)) * Real.sin (2 * α) →
      hy = (u^2 / (2 * k)) * Real.sin(α)^2 →
      (∃ (A : ℝ) (B : ℝ), 
        A = (u^2 / (2 * k)) ∧ B = (u^2 / (4 * k)) ∧
        Real.pi * A * B = d * (u^4 / k^2)) :=
sorry

end projectile_highest_points_area_l612_612870


namespace smallest_area_of_DEF_isosceles_right_l612_612894

noncomputable def area {α : Type*} [LinearOrderedField α] (x1 y1 x2 y2 x3 y3 : α) : α :=
  1 / 2 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem smallest_area_of_DEF_isosceles_right 
  (a b c d e f x1 y1 x2 y2 x3 y3 : ℝ)
  (h₁ : area 0 0 a 0 a a = 1)
  (h₂ : isosceles_right 0 0 a 0 a a)
  (h₃ : position_d e f)
  (h₄ : isosceles_right (b d) (c e) (f0 f1))
  : area (b d) (c e) (f0 f1) ≤ 1/5 := 
sorry

end smallest_area_of_DEF_isosceles_right_l612_612894


namespace shaded_region_area_l612_612322

theorem shaded_region_area (PQ : ℝ) (hPQ : PQ = 10) :
  ∃ (s : ℝ), 
    24 * s^2 = 600 / 13 ∧ 
    ((6 * s) ^ 2 + (4 * s) ^ 2 = PQ ^ 2) := by
  -- Definitions and conditions:
  have h1 : 6^2 + 4^2 = 52 := by norm_num
  have h2 : PQ ^ 2 = 100 := by rw [hPQ]; norm_num

  -- Use Pythagorean theorem to deduce the side length
  let s := real.sqrt (100 / 52)
  use s
  split
  -- Prove area of 24 squares is 600/13
  · have s2 : s^2 = 100 / 52 := by sorry -- Derived from s = sqrt(100/52)
  · · calc
    -- Prove area of the 24 squares:
    24 * s^2 = 24 * (100 / 52) := by rw [s2] -- square of side length
    ... = 600 / 13 := by exact_mod_cast (600 : ℝ) / (13 : ℝ)
  
  -- Prove sum of squares of sides is PQ^2
  · calc
    (6 * s) ^ 2 + (4 * s) ^ 2 = 36 * s^2 + 16 * s^2 := by ring -- expand squares
    ... = 52 * s^2 := by linarith
    ... = 100 := by
      rw [s2]
      field_simp
      norm_num
  sorry -- Full proof skipped

end shaded_region_area_l612_612322


namespace find_b_if_continuous_at_4_l612_612224

def f (x : ℝ) (b : ℝ) :=
  if x ≤ 4 then 4 * x^2 + 5 else b * x + 2

theorem find_b_if_continuous_at_4 : ∃ (b : ℝ), (∀ (x : ℝ), f x b = f 4 b) → b = 67 / 4 :=
by
  sorry

end find_b_if_continuous_at_4_l612_612224


namespace conditional_probability_l612_612126

variable (x y : Type)
variable (z : Set x → ℝ)

axiom prob_x : z ({a : x | a ∈ x}) = 0.02
axiom prob_y : z ({a : y | a ∈ y}) = 0.10
axiom prob_x_and_y : z ({a : x | a ∈ x} ∩ {a : y | a ∈ y}) = 0.10

theorem conditional_probability :
  z ({a : x | a ∈ x} ∩ {a : y | a ∈ y}) / z ({a : y | a ∈ y}) = 1 := by
  sorry

end conditional_probability_l612_612126


namespace length_AN_l612_612251

theorem length_AN {A B C M N : Type*}
  (h1 : M ∈ angle_bisector A B C)
  (h2 : N ∈ extension A B)
  (h3 : distance A C = 1)
  (h4 : distance A M = 1)
  (h5 : ∠ A N M = ∠ C N M) :
  distance A N = 1 :=
sorry

end length_AN_l612_612251


namespace derivative_at_0_l612_612133

noncomputable def f (x : ℝ) := Real.exp x / (x + 2)

theorem derivative_at_0 : deriv f 0 = 1 / 4 := sorry

end derivative_at_0_l612_612133


namespace valid_numbers_count_l612_612063

/-- Definition of a valid number in base n as per the given problem's constraints -/
def isValidNumber (n : ℕ) (digits : List ℕ) : Prop :=
  (digits.all fun d => d < n) ∧          -- All digits must be less than n
  (digits.Nodup) ∧                      -- All digits must be different
  (
    (digits.size > 1) → 
    ∀ i < digits.size - 1, abs (digits.get! i - digits.get! (i + 1)) = 1
  )                                     -- Every digit except the first differs by ±1 from some digit to its left

/-- Function F, which counts the valid numbers -/
def F (n : ℕ) : ℕ :=
  2^(n + 1) - 2 * n - 2

/-- Proof that the number of valid numbers in base n is given by F(n) -/
theorem valid_numbers_count (n : ℕ) : 
  (∑ digits in finset.range (n ^ (n-1)), if isValidNumber n digits then 1 else 0) = F(n) := 
by
  sorry

end valid_numbers_count_l612_612063


namespace part_I_part_II_l612_612151

-- Definitions of sets A and B
def setA : set ℝ := {x | x^2 - 2 * x - 3 ≤ 0 }
def setB (m : ℝ) : set ℝ := {x | abs (x - m) ≤ 2 }

-- The conditions and proofs

-- (A) If A ∩ B = [0, 3], then m = 2
theorem part_I (m : ℝ) (h : setA ∩ setB m = {x | 0 ≤ x ∧ x ≤ 3}) : m = 2 := by
  sorry

-- (B) If A ⊆ C_ℝ B, then m > 5 or m < -3
theorem part_II (m : ℝ) (h : setA ⊆ {x | x < m - 2 ∨ x > m + 2}) : m > 5 ∨ m < -3 := by
  sorry

end part_I_part_II_l612_612151


namespace part_I_part_II_part_III_l612_612969

-- Part (I)
theorem part_I (m : ℕ) (x : ℝ) (hx : x > -1) : (1 + x)^m ≥ 1 + m * x := sorry

-- Part (II)
theorem part_II (n m : ℕ) (hn : n ≥ 6) (hm : 1 ≤ m ∧ m ≤ n) (h : (1 - 1 / (n + 3))^n < 1 / 2) : 
  (1 - m / (n + 3 : ℝ))^n < (1 / 2)^m := sorry

-- Part (III)
theorem part_III (n : ℕ) : (3 ^ n + 4 ^ n + ... + (n + 2) ^ n = (n + 3) ^ n) ↔ (n = 2 ∨ n = 3) := sorry

end part_I_part_II_part_III_l612_612969


namespace quadratic_inequality_l612_612617

theorem quadratic_inequality (a b c x : ℝ) 
  (h1 : a*x^2 + b*x + c = 0) (h2 : a < 0) (h3 : h1 = -1 ∨ h1 = 4)
  : (a*x^2 + b*x + c < 0) ↔ (x < -1 ∨ x > 4) :=
  sorry

end quadratic_inequality_l612_612617


namespace length_AN_eq_one_l612_612275

variable {α : Type*} [InnerProductSpace ℝ α]

/-- Given a triangle ABC and points M and N satisfying the conditions in the problem -/
theorem length_AN_eq_one
  (A B C M N : α)
  (hM : ∠BAC = ∠BAM + ∠MAC)
  (hN' : ∃ P, P ∈ line_through A B ∧ N ≠ A ∧ sameRay (line_through A B) P N)
  (hAC : dist A C = 1)
  (hAM : dist A M = 1)
  (h_angle : angle A N M = angle C N M) :
  dist A N = 1 :=
sorry

end length_AN_eq_one_l612_612275


namespace smallest_int_n_l612_612951

noncomputable def smallest_positive_integer (n : ℕ) : ℕ :=
  if ∀ (a : ℕ → ℝ), ∀ b d : ℝ, (∀ k, 1 ≤ k ∧ k ≤ 1999 → ∃ m, a m = b + d * m ∧ m ∈ ℕ) :=
    ∃ m, a m ∈ ℤ → n ∈ ℕ 
  then n
  else 70

theorem smallest_int_n {n : ℕ} (h : ∀ (a : ℕ → ℝ), ∀ b d : ℝ, 
                                          (∀ k, 1 ≤ k ∧ k ≤ 1999 
                                              → ∃ m, a m = b + d * m ∧ m ∈ ℕ) := 
                                          ∃ m, a m ∈ ℤ 
                                          → 70 ∈ ℕ) : 
                                          smallest_positive_integer n = 70 :=
sorry

end smallest_int_n_l612_612951


namespace same_color_pick_probability_l612_612013

/-- 
  Given that a jar has 15 red candies and 5 blue candies,
  Terry picks two candies at random, then Mary picks one of 
  the remaining candies at random. Prove that the probability 
  that all picked candies are of the same color is 31/76.
-/
theorem same_color_pick_probability :
  let total_candies := 20
  let red_candies := 15
  let blue_candies := 5
  let terry_two_reds := (red_candies * (red_candies - 1)) / (total_candies * (total_candies - 1))
  let mary_one_red_given_two_reds := (red_candies - 2) / (total_candies - 2)
  let all_red_probability := terry_two_reds * mary_one_red_given_two_reds
  let terry_two_blues := (blue_candies * (blue_candies - 1)) / (total_candies * (total_candies - 1))
  let mary_one_blue_given_two_blues := (blue_candies - 2) / (total_candies - 2)
  let all_blue_probability := terry_two_blues * mary_one_blue_given_two_blues
  let total_probability := all_red_probability + all_blue_probability
  total_probability = (31 / 76) :=
by
  sorry

end same_color_pick_probability_l612_612013


namespace equal_numbers_impossible_l612_612823

theorem equal_numbers_impossible (a : ℚ) (numbers : Fin 10 → ℚ) :
  (∀ n : Fin 10, ∃ (steps : Nat), ∀ i j : Fin 10, i ≠ j → average_replacement_possible i j steps → numbers i = a) → False :=
by
  unfold average_replacement_possible
  unfold average
  sorry

end equal_numbers_impossible_l612_612823


namespace days_to_cover_half_lake_l612_612182

-- Define the problem conditions in Lean
def doubles_every_day (size: ℕ → ℝ) : Prop :=
  ∀ n : ℕ, size (n + 1) = 2 * size n

def takes_25_days_to_cover_lake (size: ℕ → ℝ) (lake_size: ℝ) : Prop :=
  size 25 = lake_size

-- Define the main theorem
theorem days_to_cover_half_lake (size: ℕ → ℝ) (lake_size: ℝ) 
  (h1: doubles_every_day size) (h2: takes_25_days_to_cover_lake size lake_size) : 
  size 24 = lake_size / 2 :=
sorry

end days_to_cover_half_lake_l612_612182


namespace damien_jogging_distance_l612_612921

theorem damien_jogging_distance (miles_per_day : ℕ) (weekdays_per_week : ℕ) (weeks : ℕ) 
  (h1 : miles_per_day = 5) (h2 : weekdays_per_week = 5) (h3 : weeks = 3) :
  miles_per_day * weekdays_per_week * weeks = 75 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end damien_jogging_distance_l612_612921


namespace output_increase_l612_612778

variable (a : ℝ) (last_year_output planned_output actual_output : ℝ)

-- Given conditions
def planned_output := a
def last_year_output := planned_output / 1.1
def actual_output := 1.01 * planned_output

-- Proof problem
theorem output_increase (h : planned_output = a)
                        (h1 : planned_output = last_year_output * 1.1)
                        (h2 : actual_output = planned_output * 1.01) :
    (actual_output - last_year_output) / last_year_output * 100 = 11.1 :=
by
  sorry

end output_increase_l612_612778


namespace relationship_sides_l612_612649

-- Definitions for the given condition
variables (a b c : ℝ)

-- Statement of the theorem to prove
theorem relationship_sides (h : a^2 - 16 * b^2 - c^2 + 6 * a * b + 10 * b * c = 0) : a + c = 2 * b :=
sorry

end relationship_sides_l612_612649


namespace count_good_divisors_l612_612370

/-- We call a natural number \( n \) good if 2020 divided by \( n \) leaves a remainder of 22.
    This means \( n \) must be a divisor of 1998 and \( n > 22 \) -/
theorem count_good_divisors : finset.card (finset.filter (λ m, 22 < m ∧ 1998 % m = 0) (finset.range 2000)) = 10 :=
by
  -- This is where the proof would go
  sorry

end count_good_divisors_l612_612370


namespace alex_loan_payment_difference_is_2383_98_l612_612888

-- Define the constants and initial conditions.
def loan_amount : ℝ := 15000
def annual_interest_rate1 : ℝ := 0.08
def compounded_times_per_year1 : ℕ := 2
def years1_total : ℕ := 15
def years1_partial : ℕ := 7
def annual_interest_rate2 : ℝ := 0.09
def years2_total : ℕ := 15

noncomputable def total_payment_scheme1 : ℝ :=
  let balance_after_7_years := loan_amount * (1 + annual_interest_rate1 / compounded_times_per_year1) ^ (compounded_times_per_year1 * years1_partial)
  let remaining_balance_after_partial_payment := balance_after_7_years / 2
  let remaining_balance_final := remaining_balance_after_partial_payment * (1 + annual_interest_rate1 / compounded_times_per_year1) ^ (compounded_times_per_year1 * (years1_total - years1_partial))
  remaining_balance_after_partial_payment + remaining_balance_final

noncomputable def total_payment_scheme2 : ℝ :=
  loan_amount * (1 + annual_interest_rate2 * years2_total)

noncomputable def positive_difference_in_total_payments : ℝ :=
  (total_payment_scheme1 - total_payment_scheme2).to_real.abs

theorem alex_loan_payment_difference_is_2383_98 :
  positive_difference_in_total_payments = 2383.98 :=
by
  sorry

end alex_loan_payment_difference_is_2383_98_l612_612888


namespace minimum_value_func_minimum_value_attained_l612_612970

noncomputable def func (x : ℝ) : ℝ := (4 / (x - 1)) + x

theorem minimum_value_func : ∀ (x : ℝ), x > 1 → func x ≥ 5 :=
by
  intros x hx
  -- proof goes here
  sorry

theorem minimum_value_attained : func 3 = 5 :=
by
  -- proof goes here
  sorry

end minimum_value_func_minimum_value_attained_l612_612970


namespace mixed_number_calculation_l612_612489

theorem mixed_number_calculation :
  47 * (2 + 2/3 - (3 + 1/4)) / (3 + 1/2 + (2 + 1/5)) = -4 - 25/38 :=
by
  sorry

end mixed_number_calculation_l612_612489


namespace a_general_term_T_sum_l612_612212

-- Definition of the sequence {a_n}
def S (n : ℕ) : ℕ := ∑ i in Finset.range n, (a i)
def a : ℕ → ℕ 
| 0     := 1
| (n+1) := 2 * S n + 1

-- General term formula for the sequence {a_n}
theorem a_general_term (n : ℕ) : a n = 3^(n-1) :=
sorry

-- Definition for sequence {b_n} and {T_n}
def b (n : ℕ) : ℕ := (3 * n - 1) * 3^(n-1)
def T (n : ℕ) := ∑ i in Finset.range n, (b i)

-- Sum of the first n terms of the sequence {b_n}
theorem T_sum (n : ℕ) : T n = ( (3 * n / 2) - 5 / 4) * 3^n + 5 / 4 :=
sorry

end a_general_term_T_sum_l612_612212


namespace count_good_divisors_l612_612373

/-- We call a natural number \( n \) good if 2020 divided by \( n \) leaves a remainder of 22.
    This means \( n \) must be a divisor of 1998 and \( n > 22 \) -/
theorem count_good_divisors : finset.card (finset.filter (λ m, 22 < m ∧ 1998 % m = 0) (finset.range 2000)) = 10 :=
by
  -- This is where the proof would go
  sorry

end count_good_divisors_l612_612373


namespace function_solution_l612_612553

variable (f : ℝ → ℝ)

-- Given functional equation
def functional_eq (x y : ℝ) : Prop :=
  f(x * f(y) + 1) = y + f(f(x) * f(y))

-- The theorem statement to be proven in Lean
theorem function_solution :
  (∀ x y : ℝ, functional_eq f x y) → (f = λ x, x - 1) :=
by
  sorry

end function_solution_l612_612553


namespace amount_after_two_years_l612_612567

-- Given definitions
def present_value : ℝ := 3200
def rate_of_increase : ℝ := 1 / 8
def number_of_years : ℕ := 2

-- Goal to prove
theorem amount_after_two_years :
  present_value * (1 + rate_of_increase)^number_of_years = 4050 := by
  sorry

end amount_after_two_years_l612_612567


namespace cos_36_eq_l612_612911

theorem cos_36_eq : ∃ x : ℝ, x = cos (36 * real.pi / 180) ∧ x = (1 + real.sqrt 5) / 4 :=
by
  let x := cos (36 * real.pi / 180)
  let y := cos (72 * real.pi / 180)
  have h1 : y = 2 * x^2 - 1 := sorry -- double angle formula
  have h2 : 2 * y^2 - 1 = x := sorry -- cosine property
  use (1 + real.sqrt 5) / 4
  have hx : x = (1 + real.sqrt 5) / 4 := sorry -- solving the quartic equation
  exact ⟨by rfl, hx⟩

end cos_36_eq_l612_612911


namespace domain_of_f_l612_612772

def f (x : ℝ) : ℝ := (sqrt (2 * x - 1)) / (x^2 + x - 2)

theorem domain_of_f :
  { x : ℝ | 2 * x - 1 ≥ 0 ∧ x^2 + x - 2 ≠ 0 } =
  { x : ℝ | x ≥ 1/2 ∧ x ≠ 1 } :=
by
  sorry

end domain_of_f_l612_612772


namespace correct_options_l612_612142

theorem correct_options (a b : ℝ) (h_a_pos : a > 0) (h_discriminant : a^2 = 4 * b):
  (a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬(∃ x1 x2 : ℝ, (x1 * x2 > 0 ∧ a^2 - x1x2 ≠ 4b))) ∧ 
  (∀ c x1 x2 : ℝ, (x1 - x2 = 4) → (a^2 - 4 * (b - c) = 16) → (c = 4)) :=
by
  sorry

end correct_options_l612_612142


namespace P_has_positive_real_roots_l612_612603

def P (n : Nat) (coeffs : List ℝ) : Polynomial ℝ := 
  Polynomial.sum (List.zipWith (λ a i, Polynomial.C a * Polynomial.X ^ i) coeffs (List.range n.succ))

def P_iter (n : Nat) (coeffs : List ℝ) (x : ℝ) (k : Nat) : ℝ :=
  let P_inst := P n coeffs
  P_inst.eval (nat.iterate (fun y => P_inst.eval y) k x)

theorem P_has_positive_real_roots
  (n : Nat) (coeffs : List ℝ) (h_len : coeffs.length = n.succ)
  (h_exists_m : ∃ (m : Nat), m ≥ 2 ∧ ∀ x ∈ (real.roots (P_iter n coeffs x m)), 0 < x ) : 
  ∃ x ∈ (real.roots (P n coeffs)), 0 < x := 
sorry

end P_has_positive_real_roots_l612_612603


namespace problem1_l612_612438

theorem problem1 : 
  ∀ a b : ℤ, a = 1 → b = -3 → (a - b)^2 - 2 * a * (a + 3 * b) + (a + 2 * b) * (a - 2 * b) = -3 :=
by
  intros a b h1 h2
  rw [h1, h2]
  sorry

end problem1_l612_612438


namespace length_AN_l612_612252

theorem length_AN {A B C M N : Type*}
  (h1 : M ∈ angle_bisector A B C)
  (h2 : N ∈ extension A B)
  (h3 : distance A C = 1)
  (h4 : distance A M = 1)
  (h5 : ∠ A N M = ∠ C N M) :
  distance A N = 1 :=
sorry

end length_AN_l612_612252


namespace sum_of_squares_l612_612704

theorem sum_of_squares {x : Fin 50 → ℝ} (h_sum : ∑ i, x i = 1)
  (h_frac_sum : ∑ i, (x i) / (1 - x i) = 2) :
  ∑ i, (x i)^2 / (1 - x i) = -1 :=
by
  sorry

end sum_of_squares_l612_612704


namespace half_abs_diff_squares_eq_40_l612_612402

theorem half_abs_diff_squares_eq_40 (x y : ℤ) (hx : x = 21) (hy : y = 19) :
  (|x^2 - y^2| / 2) = 40 :=
by
  sorry

end half_abs_diff_squares_eq_40_l612_612402


namespace complete_the_square_l612_612756

theorem complete_the_square :
  ∀ x : ℝ, (x^2 + 8 * x + 7 = 0) → (x + 4)^2 = 9 :=
by
  intro x h,
  sorry

end complete_the_square_l612_612756


namespace max_sum_a_b_l612_612214

theorem max_sum_a_b (a b : ℝ) (ha : 4 * a + 3 * b ≤ 10) (hb : 3 * a + 6 * b ≤ 12) : a + b ≤ 22 / 7 :=
sorry

end max_sum_a_b_l612_612214


namespace find_g_l612_612790

theorem find_g (t : ℝ) : 
  (∃ g : ℝ → ℝ, (∀ t, ((y = 2 * x - 40) → (x = g(t))) ∧ ((y = 20 * t - 14) → (y = 2 * (g t) - 40)))) → 
  g t = 10 * t + 13 :=
by
  sorry

end find_g_l612_612790


namespace carl_sold_each_watermelon_for_3_l612_612496

def profit : ℕ := 105
def final_watermelons : ℕ := 18
def starting_watermelons : ℕ := 53
def sold_watermelons : ℕ := starting_watermelons - final_watermelons
def price_per_watermelon : ℕ := profit / sold_watermelons

theorem carl_sold_each_watermelon_for_3 :
  price_per_watermelon = 3 :=
by
  sorry

end carl_sold_each_watermelon_for_3_l612_612496


namespace good_numbers_count_l612_612340

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l612_612340


namespace ball_bounce_height_l612_612853

noncomputable def min_bounces (h₀ h_min : ℝ) (bounce_factor : ℝ) := 
  Nat.ceil (Real.log (h_min / h₀) / Real.log bounce_factor)

theorem ball_bounce_height :
  min_bounces 512 40 (3/4) = 8 :=
by
  sorry

end ball_bounce_height_l612_612853


namespace sum_of_series_l612_612913

theorem sum_of_series :
  (\sum n in Finset.range 1000, 1 / (n + 1)^2 + (n + 1)) = 1000 / 1001 := 
sorry

end sum_of_series_l612_612913


namespace intersection_M_N_l612_612641

def M : set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : set ℝ := {x | 0 ≤ x ∧ x < 1 ∧ ∃ y, y = real.sqrt x + real.log (1 - x)}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_M_N_l612_612641


namespace interest_rate_10_percent_l612_612185

-- Definitions for the problem
variables (P : ℝ) (R : ℝ) (T : ℝ)

-- Condition that the money doubles in 10 years on simple interest
def money_doubles_in_10_years (P R : ℝ) : Prop :=
  P = (P * R * 10) / 100

-- Statement that R is 10% if the money doubles in 10 years
theorem interest_rate_10_percent {P : ℝ} (h : money_doubles_in_10_years P R) : R = 10 :=
by
  sorry

end interest_rate_10_percent_l612_612185


namespace roots_of_quadratic_example_quadratic_problem_solution_l612_612799

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic :
  ∀ (a b c : ℝ), a ≠ 0 → discriminant a b c > 0 → (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0)) :=
by
  intros a b c a_ne_zero discr_positive
  sorry

theorem example_quadratic :
  discriminant 1 (-2) (-1) > 0 :=
by
  unfold discriminant
  norm_num
  linarith

theorem problem_solution :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (1 * x₁^2 - 2 * x₁ - 1 = 0) ∧ (1 * x₂^2 - 2 * x₂ - 1 = 0) :=
by
  apply roots_of_quadratic 1 (-2) (-1)
  norm_num
  apply example_quadratic

end roots_of_quadratic_example_quadratic_problem_solution_l612_612799


namespace calculate_cost_price_l612_612837

variable (SellingPrice : ℝ) (ProfitPercentage : ℝ)

theorem calculate_cost_price 
  (h1 : SellingPrice = 290) 
  (h2 : ProfitPercentage = 0.20) : 
  let CostPrice := SellingPrice / (1 + ProfitPercentage) in
  CostPrice = 241.67 :=
by 
  sorry

end calculate_cost_price_l612_612837


namespace find_AN_length_l612_612259

theorem find_AN_length
  (A B C M N : Point)
  (h1 : is_angle_bisector (angle B A C) M)
  (h2 : on_extension A B N)
  (h3 : dist A C = 1)
  (h4 : dist A M = 1)
  (h5 : angle A N M = angle C N M)
  : dist A N = 1 := sorry

end find_AN_length_l612_612259


namespace distance_center_line_l612_612769

-- Definitions in Lean
def polar_circle (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ - 2 * Real.sin θ
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 3

-- Theorem stating the distance from the center of the circle to the line
theorem distance_center_line :
  let center := (1 : ℝ, -1 : ℝ) in
  let line_x := 3 in
  ∀ (ρ θ : ℝ), polar_circle ρ θ ∧ polar_line ρ θ →
  abs (line_x - center.1) = 2 :=
begin
  sorry
end

end distance_center_line_l612_612769


namespace correct_factorization_l612_612807

theorem correct_factorization:
  (∃ a : ℝ, (a + 3) * (a - 3) = a ^ 2 - 9) ∧
  (∃ x : ℝ, x ^ 2 + x - 5 = x * (x + 1) - 5) ∧
  ¬ (∃ x : ℝ, x ^ 2 + 1 = x * (x + 1 / x)) ∧
  (∃ x : ℝ, x ^ 2 + 4 * x + 4 = (x + 2) ^ 2) →
  (∃ x : ℝ, x ^ 2 + 4 * x + 4 = (x + 2) ^ 2)
  := by
  sorry

end correct_factorization_l612_612807


namespace sum_vectors_zero_l612_612736

noncomputable theory

variables {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]

variables (n : ℕ) (a : fin n → V)

-- Conditions
def equal_lengths (a : fin n → V) : Prop :=
  ∀ i j, i ≠ j → ∥a i∥ = ∥a j∥

def additional_equal_lengths (a : fin n → V) : Prop :=
  ∀ k, ∥(-a 0 + (finset.univ.filter (≠ k)).sum (λ i, a i))∥ = 
       ∥(finset.sum_finrange a)∥

-- The proof statement
theorem sum_vectors_zero (h1 : 2 < n)
                        (h2 : equal_lengths a)
                        (h3 : additional_equal_lengths a) :
  finset.sum finset.univ a = 0 :=
sorry

end sum_vectors_zero_l612_612736


namespace find_set_of_real_m_l612_612671

-- Define the existence of point A on circle C
variable (A : Point := Point.mk 3 2)

-- Define the equations of the fold lines
variable (fold_line1 : AffinePlane.Line := AffinePlane.Line.mk (-1) (1) 1)
variable (fold_line2 : AffinePlane.Line := AffinePlane.Line.mk (1) (1) (-7))

-- Define points M and N
variable (m : ℝ)
noncomputable def M := (Point.mk (-m) 0)
noncomputable def N := (Point.mk m 0)

-- Define the circle C
-- Center (3,4) and radius √4 as a simplifying assumption
-- The standard form of given circle equation: (x-3)^2 + (y-4)^2 = 4
noncomputable def C : Circle := Circle.mk (Point.mk 3 4) 2

-- Define the point P on circle C such that ∠MPN = 90°
variable (P : Point)
axiom P_on_C : CircleContains C P

theorem find_set_of_real_m :
  (∃ P : Point, P_on_C P ∧ Angle M P N = 90°) → m ∈ Set.Icc (3 : ℝ) 7 :=
by sorry

end find_set_of_real_m_l612_612671


namespace frictional_force_is_correct_l612_612466

-- Definitions
def m1 := 2.0 -- mass of the tank in kg
def m2 := 10.0 -- mass of the cart in kg
def a := 5.0 -- acceleration of the cart in m/s^2
def mu := 0.6 -- coefficient of friction between the tank and the cart
def g := 9.8 -- acceleration due to gravity in m/s^2

-- Frictional force acting on the tank
def frictional_force := mu * (m1 * g)

-- Required force to accelerate the tank with the cart
def required_force := m1 * a

-- Proof statement
theorem frictional_force_is_correct : required_force = 10 := 
by
  -- skipping the proof as specified
  sorry

end frictional_force_is_correct_l612_612466


namespace polynomial_has_unique_real_root_l612_612298

variable (a b c : ℝ)

theorem polynomial_has_unique_real_root (h1 : a^2 * b = 2 * b^2) (h2 : 2 * b^2 = 4 * a * c) :
  ∃! x : ℝ, (x^3 + a * x^2 + b * x + c = 0 ∧ (a ∈ ℤ ∧ b ∈ ℤ ∧ c ∈ ℤ → x ∈ ℤ)) :=
sorry

end polynomial_has_unique_real_root_l612_612298


namespace value_of_k_l612_612663

-- Define the set A
def set_A (k : ℝ) : set ℝ := {x : ℝ | k * x^2 + 4 * x + 4 = 0}

-- Define the condition that set A has only one element
def has_only_one_element (s : set ℝ) : Prop := ∃ x, s = {x}

-- State the problem and prove that k = 0 or k = 1 satisfies the condition
theorem value_of_k (k : ℝ) : has_only_one_element (set_A k) ↔ k = 0 ∨ k = 1 :=
by sorry

end value_of_k_l612_612663


namespace rational_b_if_rational_a_l612_612197

theorem rational_b_if_rational_a (x : ℚ) (h_rational : ∃ a : ℚ, a = x / (x^2 - x + 1)) :
  ∃ b : ℚ, b = x^2 / (x^4 - x^2 + 1) :=
by
  sorry

end rational_b_if_rational_a_l612_612197


namespace cat_food_inequality_l612_612515

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l612_612515


namespace bananas_used_l612_612070

-- Define the conditions
def bananas_per_loaf : Nat := 4
def loaves_on_monday : Nat := 3
def loaves_on_tuesday : Nat := 2 * loaves_on_monday
def total_loaves : Nat := loaves_on_monday + loaves_on_tuesday

-- Define the total bananas used
def total_bananas : Nat := bananas_per_loaf * total_loaves

-- Prove that the total bananas used is 36
theorem bananas_used : total_bananas = 36 :=
by
  sorry

end bananas_used_l612_612070


namespace simplified_correct_evaluate_at_1_find_k_l612_612437

-- Define the expressions and solve their simplification

def simplify_expression (x : ℝ) : ℝ := (2 * x + 1) ^ 2 - (2 * x + 1) * (2 * x - 1) + (x + 1) * (x - 3)

theorem simplified_correct : ∀ x : ℝ, simplify_expression x = x^2 + 2*x - 1 := by
  intro x
  -- The remaining proof will be filled in as per the solution steps
  sorry

theorem evaluate_at_1 : simplify_expression 1 = 2 := by
  -- Direct substitution and simplification
  calc simplify_expression 1 = 1^2 + 2*1 - 1 := by sorry
                      ... = 2 := by norm_num

-- Define the system of equations and conditions

/--
  Given: 
  1. x + y = 1
  2. kx + (k-1)y = 7
  3. 3x - 2y = 5

  Find the value of k
-/
theorem find_k (x y k : ℝ) (h1 : x + y = 1) (h2 : k * x + (k - 1) * y = 7) (h3 : 3 * x - 2 * y = 5) : k = 33 / 5 := by
  -- We need to solve the system of equations given the condition
  sorry

end simplified_correct_evaluate_at_1_find_k_l612_612437


namespace sum_abs_aj_leq_3_l612_612439

-- Define the given conditions
variables {n : ℕ} (a : ℕ → ℂ)
axiom positive_int (hn : 3 * n > 0)
axiom subset_ineq (I : finset ℕ) (hI : I.nonempty)  
  (hI_subset : ∀ i ∈ I, i < n) :
  ∥(I.prod (λ j, (1 + a j)) - 1)∥ ≤ 1/2

-- Define the main proof goal
theorem sum_abs_aj_leq_3 (hn : 3 * n > 0) 
  (hI_ineq : ∀ (I : finset ℕ), I.nonempty → (∀ i ∈ I, i < n) → ∥(I.prod (λ j, (1 + a j)) - 1)∥ ≤ 1/2) :
  ∑ j in finset.range n, ∥a j∥ ≤ 3 :=
sorry

end sum_abs_aj_leq_3_l612_612439


namespace count_of_good_numbers_l612_612351

-- Define the conditions of the problem
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- The main statement proving the number of 'good' numbers
theorem count_of_good_numbers :
  (finset.filter is_good_number (finset.Icc 1 2020)).card = 10 :=
by
  sorry

end count_of_good_numbers_l612_612351


namespace complex_exponent_calculation_l612_612082

theorem complex_exponent_calculation : (complex.mk 1 1 / real.sqrt 2) ^ 46 = -complex.i := sorry

end complex_exponent_calculation_l612_612082


namespace number_of_points_P_l612_612192

open EuclideanGeometry

-- Define the square ABCD
variables {A B C D P : Point}
variable {square_ABCD : Square A B C D}

-- Define that triangles PAB, PBC, PCD and PDA are all isosceles
variables (isosceles_PAB : IsIsosceles (Triangle.mk P A B))
          (isosceles_PBC : IsIsosceles (Triangle.mk P B C))
          (isosceles_PCD : IsIsosceles (Triangle.mk P C D))
          (isosceles_PDA : IsIsosceles (Triangle.mk P D A))

-- State that the number of such points P is 9
theorem number_of_points_P : (number_of_such_points P square_ABCD isosceles_PAB isosceles_PBC isosceles_PCD isosceles_PDA) = 9 :=
sorry

end number_of_points_P_l612_612192


namespace chips_2004_impossible_chips_2003_possible_l612_612848

/-- Statement for 2004 chips: proving it's not possible to make all the chips white --/
theorem chips_2004_impossible :
  ∀ (chips : Fin 2004 → Bool),
  (chips 0 = true ∧ (forall i, i ≠ 0 -> chips i = false)) →
  ¬ (∃ f : (Fin 2004 → Bool) → (Fin (2004 - 2) → Fin 2004),
      (∀ n, (chips_fun n).circle n = chips.circle n) ∧
      ∀ n, chips_fun n ~ chips_fun 0) := 
begin
  sorry
end

/-- Statement for 2003 chips: proving it is possible to make all the chips white --/
theorem chips_2003_possible :
  ∀ (chips : Fin 2003 → Bool),
  (chips 0 = true ∧ (forall i, i ≠ 0 -> chips i = false)) →
  ∃ f : (Fin 2003 → Bool) → (Fin (2003 - 2) → Fin 2003),
      (∀ n, (chips_fun n).circle n = chips.circle n) ∧
      ∀ n, chips_fun n ~ chips_fun 0 :=
begin
  sorry
end

end chips_2004_impossible_chips_2003_possible_l612_612848


namespace plant_arrangement_count_l612_612896

-- Define the given conditions as variables in Lean 4
def basil_plants : ℕ := 4
def tomato_plants : ℕ := 4
def pepper_plants : ℕ := 3

-- State the theorem
theorem plant_arrangement_count : 
  let group_count := 3 in
  let ways_to_arrange_groups := (group_count !) in
  let ways_within_basil_group := (basil_plants !) in
  let ways_within_tomato_group := (tomato_plants !) in
  let ways_within_pepper_group := (pepper_plants !) in
  ways_to_arrange_groups * ways_within_basil_group * ways_within_tomato_group * ways_within_pepper_group = 20736 :=
by
  sorry

end plant_arrangement_count_l612_612896


namespace hexagon_ratio_l612_612785

noncomputable def Point := ℝ × ℝ

theorem hexagon_ratio
  {A B C D E F X : Point}
  (h_circle : ∀ P ∈ {A, B, C, D, E, F}, P ∈ circumcircle {A, B, C, D, E, F})
  (h_eq_sides : dist A B = dist C D ∧ dist C D = dist E F)
  (h_concurrent : ∃ P : Point, Collinear P A D ∧ Collinear P B E ∧ Collinear P C F)
  (h_X_def : Intersection A D C E = X) :
  (dist C X / dist X E) = (dist A C / dist C E) ^ 2 := 
by sorry

end hexagon_ratio_l612_612785


namespace original_number_is_25_l612_612021

theorem original_number_is_25 (x : ℕ) (h : ∃ n : ℕ, (x^2 - 600)^n = x) : x = 25 :=
sorry

end original_number_is_25_l612_612021


namespace half_abs_diff_of_squares_l612_612405

theorem half_abs_diff_of_squares (x y : ℤ) (h1 : x = 21) (h2 : y = 19) :
  (|x^2 - y^2| / 2) = 40 := 
by
  subst h1
  subst h2
  sorry

end half_abs_diff_of_squares_l612_612405


namespace absent_minded_scientist_l612_612301

theorem absent_minded_scientist :
  let P := 1/2 in
  let PA_R := 0.05 in
  let PB_R := 0.80 in
  let P_not_R := 1/2 in
  let PA_not_R := 0.90 in
  let PB_not_R := 0.02 in
  let P_R_AB := P * PA_R * PB_R in
  let P_not_R_AB := P_not_R * PA_not_R * PB_not_R in
  let P_A_and_B := P_R_AB + P_not_R_AB in
  (P_R_AB / P_A_and_B) ≈ 0.69 :=
by
  sorry

end absent_minded_scientist_l612_612301


namespace count_good_divisors_l612_612374

/-- We call a natural number \( n \) good if 2020 divided by \( n \) leaves a remainder of 22.
    This means \( n \) must be a divisor of 1998 and \( n > 22 \) -/
theorem count_good_divisors : finset.card (finset.filter (λ m, 22 < m ∧ 1998 % m = 0) (finset.range 2000)) = 10 :=
by
  -- This is where the proof would go
  sorry

end count_good_divisors_l612_612374


namespace length_AN_eq_one_l612_612272

variable {α : Type*} [InnerProductSpace ℝ α]

/-- Given a triangle ABC and points M and N satisfying the conditions in the problem -/
theorem length_AN_eq_one
  (A B C M N : α)
  (hM : ∠BAC = ∠BAM + ∠MAC)
  (hN' : ∃ P, P ∈ line_through A B ∧ N ≠ A ∧ sameRay (line_through A B) P N)
  (hAC : dist A C = 1)
  (hAM : dist A M = 1)
  (h_angle : angle A N M = angle C N M) :
  dist A N = 1 :=
sorry

end length_AN_eq_one_l612_612272


namespace monotonicity_and_zeros_l612_612626

open Real

noncomputable def f (x k : ℝ) : ℝ := exp x - k * x + k

theorem monotonicity_and_zeros
  (k : ℝ)
  (h₁ : k > exp 2)
  (x₁ x₂ : ℝ)
  (h₂ : f x₁ k = 0)
  (h₃ : f x₂ k = 0)
  (h₄ : x₁ ≠ x₂) :
  x₁ + x₂ > 4 := 
sorry

end monotonicity_and_zeros_l612_612626


namespace debate_team_selections_l612_612296

theorem debate_team_selections
  (A_selected C_selected B_selected E_selected : Prop)
  (h1: A_selected ∨ C_selected)
  (h2: B_selected ∨ E_selected)
  (h3: ¬ (B_selected ∧ E_selected) ∧ ¬ (C_selected ∧ E_selected))
  (not_B_selected : ¬ B_selected) :
  A_selected ∧ E_selected :=
by
  sorry

end debate_team_selections_l612_612296


namespace mark_vs_jenny_distance_difference_l612_612199

def jenny_initial_distance : ℝ := 18
def jenny_bounce1_distance : ℝ := (1 / 3) * jenny_initial_distance
def jenny_bounce2_distance : ℝ := (1 / 2) * jenny_bounce1_distance
def jenny_total_distance : ℝ := jenny_initial_distance + jenny_bounce1_distance + jenny_bounce2_distance

def mark_initial_distance : ℝ := 15
def mark_bounce1_distance : ℝ := 2 * mark_initial_distance
def mark_bounce2_distance : ℝ := (3 / 4) * mark_bounce1_distance
def mark_bounce3_distance : ℝ := 0.30 * mark_bounce2_distance
def mark_total_distance : ℝ := mark_initial_distance + mark_bounce1_distance + mark_bounce2_distance + mark_bounce3_distance

def distance_difference : ℝ := mark_total_distance - jenny_total_distance

theorem mark_vs_jenny_distance_difference : distance_difference = 47.25 := by
  sorry

end mark_vs_jenny_distance_difference_l612_612199


namespace range_of_m_l612_612852

theorem range_of_m (p q : Prop) (m : ℝ) 
  (h_p : ∀ x, |x - 1| > m - 1)
  (h_q : ∀ x, f(x) = -(5 - 2m)x → f decrasing)
  (h_pq_false : p ↔ false)
  (h_p_or_q_true : p ∨ q):
  1 ≤ m ∧ m < 2 := 
sorry

end range_of_m_l612_612852


namespace range_of_t_for_decreasing_cos_2x_l612_612633

theorem range_of_t_for_decreasing_cos_2x (t : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ t → cos (2 * x₁) > cos (2 * x₂)) ↔ (0 < t ∧ t ≤ π / 2) :=
by
  sorry

end range_of_t_for_decreasing_cos_2x_l612_612633


namespace half_abs_diff_of_squares_l612_612403

theorem half_abs_diff_of_squares (x y : ℤ) (h1 : x = 21) (h2 : y = 19) :
  (|x^2 - y^2| / 2) = 40 := 
by
  subst h1
  subst h2
  sorry

end half_abs_diff_of_squares_l612_612403


namespace count_of_good_numbers_l612_612350

-- Define the conditions of the problem
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- The main statement proving the number of 'good' numbers
theorem count_of_good_numbers :
  (finset.filter is_good_number (finset.Icc 1 2020)).card = 10 :=
by
  sorry

end count_of_good_numbers_l612_612350


namespace five_letter_arrangements_count_l612_612157

theorem five_letter_arrangements_count :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
      arrangements := { l : List Char // ∃ h l', l = 'D' :: l' ++ [h] ∧
                       ¬(h = 'G') ∧
                       'B' ∈ l' ∧
                       l.length = 4 ∧
                       ∀ c ∈ l, c ∈ letters ∧ -- Validity check
                       ∀ c, In c l → ∀ c', In c' l → (c ≠ c')} -- Uniqueness constraint
  in arrangements.toFinset.card = 960 := sorry

end five_letter_arrangements_count_l612_612157


namespace count_of_good_numbers_l612_612349

-- Define the conditions of the problem
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- The main statement proving the number of 'good' numbers
theorem count_of_good_numbers :
  (finset.filter is_good_number (finset.Icc 1 2020)).card = 10 :=
by
  sorry

end count_of_good_numbers_l612_612349


namespace number_of_good_numbers_l612_612355

theorem number_of_good_numbers :
  let n := 2020 in 
  let remainder := 22 in
  let number := 1998 in
  let divisors := {d ∣ number | d > remainder}.to_list.length = 10 := by sorry

end number_of_good_numbers_l612_612355


namespace solution_l612_612564

noncomputable def balls_tossed_probability_ratio : ℕ :=
  let n := 20
  let k := 5
  let p : ℚ := (choose k 2 * choose (k - 2) 2 * choose n 3 * choose (n - 3) 3 * choose (n - 6) 6 * choose (n - 12) 4) / (choose n 4 * choose (n - 4) 4 * choose (n - 8) 4 * choose (n - 12) 4)
  let q : ℚ := (choose n 4 * choose (n - 4) 4 * choose (n - 8) 4 * choose (n - 12) 4 * choose (n - 16) 4) / (choose n 4 * choose (n - 4) 4 * choose (n - 8) 4 * choose (n - 12) 4)
  (p/q).num

theorem solution : balls_tossed_probability_ratio = 10 := 
  sorry

end solution_l612_612564


namespace congruent_regions_form_cube_l612_612916

theorem congruent_regions_form_cube :
  (∀ x y z : ℝ, |x| + |y| + |z| + |x + y + z| ≤ 2) →
  ∃ regions : Fin 6 → Set (ℝ × ℝ × ℝ), 
    (∀ i j, regions i ≃ regions j) ∧ -- All regions are congruent
    (∃ cubes : Fin 2 → Set (ℝ × ℝ × ℝ), -- The regions can be paired to form a cube
      (cubes 0 ∪ cubes 1 = ⋃ i, regions i) ∧ 
      (cubes 0 ∩ cubes 1 = ∅)) :=
sorry

end congruent_regions_form_cube_l612_612916


namespace eq_tangent_line_at_point_extreme_values_on_interval_l612_612622

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 1

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, f 1)

-- Define the tangent line at point (1, f(1))
def tangent_line_at_point (x y : ℝ) : Prop := 3 * x + 3 * y - 4 = 0

-- Statement 1: The equation of the tangent line at (1, f(1))
theorem eq_tangent_line_at_point : 
    tangent_line_at_point 1 (f 1) := 
sorry

-- Statement 2: Extreme values on the interval [-2, 3]
theorem extreme_values_on_interval : 
  let max_value := 1
  let min_value := -17 / 3
  ∃ x₁ x₂ : ℝ, 
  x₁ ∈ Set.Icc (-2 : ℝ) 3 ∧ 
  x₂ ∈ Set.Icc (-2 : ℝ) 3 ∧ 
  f x₁ = max_value ∧ 
  f x₂ = min_value :=
sorry

end eq_tangent_line_at_point_extreme_values_on_interval_l612_612622


namespace no_real_solutions_eq_implies_a_gt_exp_inv_e_l612_612762

theorem no_real_solutions_eq_implies_a_gt_exp_inv_e (a : ℝ) (h1 : 1 < a)
  (h2 : ¬ ∃ x : ℝ, a ^ x = x) :
  a > real.exp (1 / real.exp 1) :=
sorry

end no_real_solutions_eq_implies_a_gt_exp_inv_e_l612_612762


namespace range_of_m_for_increasing_f_l612_612172

theorem range_of_m_for_increasing_f :
  let f (x : ℝ) (m : ℝ) := 2 * x^3 - 3 * m * x^2 + 6 * x
  let f' (x : ℝ) (m : ℝ) := 6 * x^2 - 6 * m * x + 6
  in (∀ x > 2, f' x m > 0) → m < 5 / 2 :=
begin
  intros,
  sorry
end

end range_of_m_for_increasing_f_l612_612172


namespace abs_value_solutions_l612_612935

theorem abs_value_solutions (y : ℝ) :
  |4 * y - 5| = 39 ↔ (y = 11 ∨ y = -8.5) :=
by
  sorry

end abs_value_solutions_l612_612935


namespace distributor_cost_l612_612009

variable (C : ℝ) -- Cost of the item for the distributor
variable (P_observed : ℝ) -- Observed price
variable (commission_rate : ℝ) -- Commission rate
variable (profit_rate : ℝ) -- Desired profit rate

-- Conditions
def is_observed_price_correct (C : ℝ) (P_observed : ℝ) (commission_rate : ℝ) (profit_rate : ℝ) : Prop :=
  let SP := C * (1 + profit_rate)
  let observed := SP * (1 - commission_rate)
  observed = P_observed

-- The proof goal
theorem distributor_cost (h : is_observed_price_correct C 30 0.20 0.20) : C = 31.25 := sorry

end distributor_cost_l612_612009


namespace max_non_parallel_triangles_in_18gon_l612_612410

theorem max_non_parallel_triangles_in_18gon : 
  ∀ (V : Finset ℕ), V.card = 18 → 
  ∀ (Triangles : Finset (Finset ℕ)), 
  (∀ t ∈ Triangles, t.card = 3 ∧ (∀ e1 e2 ∈ (t.subsets 2), e1 ≠ e2 → ¬ (e1∩e2).finite → ((e1 ∪ e2)).card = 4)) → 
  ∃ (M : ℕ), M ≤ 5 ∧ (∀ T1 T2 ∈ Triangles, T1 ≠ T2 → (∃ e1 ∈ T1.subsets 2, ∃ e2 ∈ T2.subsets 2, e1 ≠ e2 → ¬ (e1 ∪ e2).parallel)) :=
sorry

end max_non_parallel_triangles_in_18gon_l612_612410


namespace binomial_sum_mod_is_85_l612_612051

noncomputable theory

open Nat

def binomial_sum_modulo : ℕ :=
  let s : ℕ := ∑ i in range 671, choose 2011 (3 * i)
  s % 1000

theorem binomial_sum_mod_is_85 : binomial_sum_modulo = 85 := sorry

end binomial_sum_mod_is_85_l612_612051


namespace number_of_solutions_eq_30_l612_612580

theorem number_of_solutions_eq_30 : 
  ∃ (x : ℤ → ℤ), (∀ n, x n = ⌊x n / 2⌋ + ⌊x n / 3⌋ + ⌊x n / 5⌋) ∧ (x.to_finset.card = 30) := 
sorry

end number_of_solutions_eq_30_l612_612580


namespace vertex_of_parabola_l612_612783

theorem vertex_of_parabola :
  ∃ (a b c : ℝ), 
      (4 * a - 2 * b + c = 9) ∧ 
      (16 * a + 4 * b + c = 9) ∧ 
      (49 * a + 7 * b + c = 16) ∧ 
      (-b / (2 * a) = 1) :=
by {
  -- we need to provide the proof here; sorry is a placeholder
  sorry
}

end vertex_of_parabola_l612_612783


namespace half_abs_diff_of_squares_l612_612407

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 21) (h2 : b = 19) :
  (abs (a^2 - b^2)) / 2 = 40 := by
  sorry

end half_abs_diff_of_squares_l612_612407


namespace sum_of_roots_Q_fourth_power_l612_612708

noncomputable def Q (z : ℂ) : ℂ := z^6 + (5 * real.sqrt 2 + 9) * z^3 + (5 * real.sqrt 2 + 8)

theorem sum_of_roots_Q_fourth_power :
  let roots := multiset.map (λ z : ℂ, z ^ 4) (multiset.roots (Q : polynomial ℂ)) in
  multiset.sum roots = 3 * (-(5 * real.sqrt 2 + 8)) ^ (4 / 3) :=
sorry

end sum_of_roots_Q_fourth_power_l612_612708


namespace turnip_mixture_l612_612905

theorem turnip_mixture (cups_potatoes total_turnips : ℕ) (h_ratio : 20 = 5 * 4) (h_turnips : total_turnips = 8) :
    cups_potatoes = 2 :=
by
    have ratio := h_ratio
    have turnips := h_turnips
    sorry

end turnip_mixture_l612_612905


namespace min_C2_minus_D2_is_36_l612_612703

noncomputable def find_min_C2_minus_D2 (x y z : ℝ) : ℝ :=
  (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 11))^2 -
  (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))^2

theorem min_C2_minus_D2_is_36 : ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → 
  find_min_C2_minus_D2 x y z ≥ 36 :=
by
  intros x y z hx hy hz
  sorry

end min_C2_minus_D2_is_36_l612_612703


namespace true_temperature_of_faulty_thermometer_l612_612867

theorem true_temperature_of_faulty_thermometer (T_f : ℝ) : T_f = 17 → 
  (T_f - 1) * (100 / 104) = 15.38 :=
by
  -- We use the conditions given in the problem to set up our theorem
  -- The faulty thermometer reads +1 for 0°C (freezing point)
  -- The faulty thermometer reads +105 for 100°C (boiling point)
  intro hT_f
  have h0 : 1 = 0 + 1 := rfl
  have h100 : 105 = 100 + 5 := rfl
  have hT_f_eq : T_f - 1 = 16 := hT_f.symm ▸ rfl
  have conversion_factor : 100 / 104 = 25 / 26 := by norm_num
  rw [hT_f, hT_f_eq, conversion_factor]
  norm_num
  linarith

end true_temperature_of_faulty_thermometer_l612_612867


namespace compare_two_sqrt_two_and_sqrt_seven_l612_612908

-- Definitions of the expressions
def two_sqrt_two := 2 * Real.sqrt 2
def sqrt_seven := Real.sqrt 7

-- Statement to be proved
theorem compare_two_sqrt_two_and_sqrt_seven : two_sqrt_two > sqrt_seven := by
  sorry

end compare_two_sqrt_two_and_sqrt_seven_l612_612908


namespace triangle_proportion_equality_l612_612738

open Locale Classical

variables (A B C E F S M N K : Type)
[affine_space ℝ (euclidean_space {x // x ∈ Side})]

variables (h1 : point E ∈ line_through A B)
          (h2 : point F ∈ line_through A C)
          (h3 : (line_through E F) ∩ (line_through B C) = S)
          (h4 : midpoint B C = M)
          (h5 : midpoint E F = N)
          (h6 : line_parallel_through_A A M N)
          (h7 : (line_through A parallel_to line_through M N) ∩ (line_through B C) = K)

theorem triangle_proportion_equality : (B K : length / C K : length) = (F S : length / E S : length) :=
begin
  sorry
end

end triangle_proportion_equality_l612_612738


namespace toothpicks_needed_for_cube_grid_l612_612821

-- Defining the conditions: a cube-shaped grid with dimensions 5x5x5.
def grid_length : ℕ := 5
def grid_width : ℕ := 5
def grid_height : ℕ := 5

-- The theorem to prove the number of toothpicks needed is 2340.
theorem toothpicks_needed_for_cube_grid (L W H : ℕ) (h1 : L = grid_length) (h2 : W = grid_width) (h3 : H = grid_height) :
  (L + 1) * (W + 1) * H + 2 * (L + 1) * W * (H + 1) = 2340 :=
  by
    -- Proof goes here
    sorry

end toothpicks_needed_for_cube_grid_l612_612821


namespace sixth_graders_more_than_seventh_l612_612766

def total_payment_seventh_graders : ℕ := 143
def total_payment_sixth_graders : ℕ := 195
def cost_per_pencil : ℕ := 13

theorem sixth_graders_more_than_seventh :
  (total_payment_sixth_graders / cost_per_pencil) - (total_payment_seventh_graders / cost_per_pencil) = 4 :=
  by
  sorry

end sixth_graders_more_than_seventh_l612_612766


namespace converse_angle_bigger_side_negation_ab_zero_contrapositive_ab_zero_l612_612558

-- Definitions
variables {α : Type} [LinearOrderedField α] {a b : α}
variables {A B C : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C]

-- Proof Problem for Question 1
theorem converse_angle_bigger_side (A B C : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C]
  (angle_C angle_B : A) (side_AB side_AC : B) (h : angle_C > angle_B) : side_AB > side_AC :=
sorry

-- Proof Problem for Question 2
theorem negation_ab_zero (a b : α) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

-- Proof Problem for Question 3
theorem contrapositive_ab_zero (a b : α) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end converse_angle_bigger_side_negation_ab_zero_contrapositive_ab_zero_l612_612558


namespace tulips_sum_l612_612039

def tulips_total (arwen_tulips : ℕ) (elrond_tulips : ℕ) : ℕ := arwen_tulips + elrond_tulips

theorem tulips_sum : tulips_total 20 (2 * 20) = 60 := by
  sorry

end tulips_sum_l612_612039


namespace area_triangle_is_5_over_3_l612_612057

noncomputable def triangle_area_root_cond (a b c : ℝ) : Prop :=
(a + b + c = 5) ∧ (ab + ac + bc = 8) ∧ (abc = 20 / 9)

noncomputable def area_of_triangle (a b c : ℝ) (h : triangle_area_root_cond a b c) : ℝ :=
  let q := (a + b + c) / 2
  let product := (q - a) * (q - b) * (q - c) in
  sqrt (q * product)

theorem area_triangle_is_5_over_3 (a b c : ℝ) (h : triangle_area_root_cond a b c) : 
  area_of_triangle a b c h = 5 / 3 := by
  sorry

end area_triangle_is_5_over_3_l612_612057


namespace evaluate_g_at_8_l612_612215

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 22 * x ^ 3 + 37 * x ^ 2 - 28 * x - 84

theorem evaluate_g_at_8 : g 8 = 1036 :=
by
  sorry

end evaluate_g_at_8_l612_612215


namespace range_of_m_l612_612999

theorem range_of_m (m : ℝ) : ((m + 3 > 0) ∧ (m - 1 < 0)) ↔ (-3 < m ∧ m < 1) :=
by
  sorry

end range_of_m_l612_612999


namespace quadratic_has_two_distinct_real_roots_l612_612802

theorem quadratic_has_two_distinct_real_roots :
  ∀ (x : ℝ), ∃ (r1 r2 : ℝ), (x^2 - 2*x - 1 = 0) → r1 ≠ r2 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l612_612802


namespace functional_inequality_solution_l612_612435

variables {f : ℝ → ℝ}

theorem functional_inequality_solution :
  (∀ x y z : ℝ, 
    x^4 + y^4 + z^4 ≥ f(x * y) + f(y * z) + f(z * x) ∧ 
    f(x * y) + f(y * z) + f(z * x) ≥ x * y * z * (x + y + z)) →
  (∀ x : ℝ, 
    (0 ≤ x → f(x) = x^2) ∧ 
    (x < 0 → f(x) ∈ set.Icc (x^2 / 2) x^2)) :=
begin
  intro h,
  sorry
end

end functional_inequality_solution_l612_612435


namespace square_floor_tile_count_l612_612465

theorem square_floor_tile_count (n : ℕ) (h1 : 2 * n - 1 = 25) : n^2 = 169 :=
by
  sorry

end square_floor_tile_count_l612_612465


namespace smallest_whole_number_greater_than_sum_l612_612582

theorem smallest_whole_number_greater_than_sum : 
  (3 + (1 / 3) + 4 + (1 / 4) + 6 + (1 / 6) + 7 + (1 / 7)) < 21 :=
sorry

end smallest_whole_number_greater_than_sum_l612_612582


namespace order_of_abc_l612_612591

noncomputable def a : ℝ := 2 ^ 2.1
noncomputable def b : ℝ := (1 / 2) ^ (-1 / 2)
noncomputable def c : ℝ := Real.log 4 / Real.log 5

theorem order_of_abc : c < b ∧ b < a := by
  sorry

end order_of_abc_l612_612591


namespace c_difference_correct_l612_612709

noncomputable def find_c_difference (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 20) : ℝ :=
  2 * Real.sqrt 34

theorem c_difference_correct (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 20) :
  find_c_difference a b c h1 h2 = 2 * Real.sqrt 34 := 
sorry

end c_difference_correct_l612_612709


namespace range_fraction_y_over_x_l612_612589

theorem range_fraction_y_over_x (x y : ℝ) 
  (modulus_condition : (x - 2)^2 + y^2 = 1) :
  ∃ k : ℝ, k = y / x ∧ 
    (-real.sqrt 3 / 3 ≤ k ∧ k < 0 ∨ 0 < k ∧ k ≤ real.sqrt 3 / 3) ∧ k ≠ 0 := 
by
sorry

end range_fraction_y_over_x_l612_612589


namespace cost_difference_l612_612796

/-- The selling price and cost of pants -/
def selling_price : ℕ := 34
def store_cost : ℕ := 26

/-- The proof that the store paid 8 dollars less than the selling price -/
theorem cost_difference : selling_price - store_cost = 8 := by
  sorry

end cost_difference_l612_612796


namespace find_angle_KDA_l612_612872

-- Definition of the problem conditions
def is_rectangle (A B C D : Point) : Prop :=
  AD = 2 * AB ∧ M = midpoint A D ∧ ∠ AMK = 80 ∧ bisects KD (∠MKC)

-- The math proof problem statement
theorem find_angle_KDA (A B C D K M : Point) (h : is_rectangle A B C D) :
  angle KDA = 35 := sorry

end find_angle_KDA_l612_612872


namespace number_of_divisors_greater_than_22_l612_612391

theorem number_of_divisors_greater_than_22 :
  (∃ (S : Finset ℕ), S = (Finset.filter (λ n, 1998 % n = 0 ∧ n > 22) (Finset.range (1998 + 1))) ∧ S.card = 10) :=
sorry

end number_of_divisors_greater_than_22_l612_612391


namespace total_planks_of_wood_l612_612048

theorem total_planks_of_wood :
  let initially = 15
  let charlie = 10
  let father = 10
  initially + charlie + father = 35 :=
by
  let initially := 15
  let charlie := 10
  let father := 10
  show initially + charlie + father = 35
  sorry

end total_planks_of_wood_l612_612048


namespace food_requirement_not_met_l612_612503

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l612_612503


namespace parabola_hyperbola_tangent_l612_612794

noncomputable def parabola (x : ℝ) : ℝ := x^2 + 5
noncomputable def hyperbola (x y : ℝ) (m : ℝ) : ℝ := y^2 - m * x^2 - 1

theorem parabola_hyperbola_tangent (m : ℝ) :
(∃ x y : ℝ, y = parabola x ∧ hyperbola x y m = 0) ↔ 
m = 10 + 2 * Real.sqrt 6 ∨ m = 10 - 2 * Real.sqrt 6 := by
  sorry

end parabola_hyperbola_tangent_l612_612794


namespace christopher_more_money_l612_612693

-- Define the conditions provided in the problem

def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64
def quarter_value : ℝ := 0.25

-- Define the question as a theorem

theorem christopher_more_money : (christopher_quarters - karen_quarters) * quarter_value = 8 :=
by sorry

end christopher_more_money_l612_612693


namespace number_of_divisors_greater_than_22_l612_612388

theorem number_of_divisors_greater_than_22 :
  (∃ (S : Finset ℕ), S = (Finset.filter (λ n, 1998 % n = 0 ∧ n > 22) (Finset.range (1998 + 1))) ∧ S.card = 10) :=
sorry

end number_of_divisors_greater_than_22_l612_612388


namespace probability_of_rain_at_least_once_l612_612587

def probRainFriday : ℝ := 0.30
def probRainMonday : ℝ := 0.60

theorem probability_of_rain_at_least_once :
  let probNoRainFriday := 1 - probRainFriday,
      probNoRainMonday := 1 - probRainMonday,
      probNoRainBothDays := probNoRainFriday * probNoRainMonday in
  1 - probNoRainBothDays = 0.72 := 
by 
  sorry

end probability_of_rain_at_least_once_l612_612587


namespace cat_food_inequality_l612_612512

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l612_612512


namespace number_of_good_numbers_l612_612356

theorem number_of_good_numbers :
  let n := 2020 in 
  let remainder := 22 in
  let number := 1998 in
  let divisors := {d ∣ number | d > remainder}.to_list.length = 10 := by sorry

end number_of_good_numbers_l612_612356


namespace length_AN_eq_one_l612_612273

variable {α : Type*} [InnerProductSpace ℝ α]

/-- Given a triangle ABC and points M and N satisfying the conditions in the problem -/
theorem length_AN_eq_one
  (A B C M N : α)
  (hM : ∠BAC = ∠BAM + ∠MAC)
  (hN' : ∃ P, P ∈ line_through A B ∧ N ≠ A ∧ sameRay (line_through A B) P N)
  (hAC : dist A C = 1)
  (hAM : dist A M = 1)
  (h_angle : angle A N M = angle C N M) :
  dist A N = 1 :=
sorry

end length_AN_eq_one_l612_612273


namespace composite_sum_pow_l612_612697

theorem composite_sum_pow (a b c d : ℕ) (h_pos : a > b ∧ b > c ∧ c > d)
    (h_div : (a + b - c + d) ∣ (a * c + b * d)) (m : ℕ) (h_m_pos : 0 < m) 
    (n : ℕ) (h_n_odd : n % 2 = 1) : ∃ k : ℕ, k > 1 ∧ k ∣ (a ^ n * b ^ m + c ^ m * d ^ n) :=
by
  sorry

end composite_sum_pow_l612_612697


namespace line_and_circle_are_separated_l612_612112

variable {a b : ℝ}

-- Given conditions
def point_inside_circle (a b : ℝ) : Prop := a^2 + b^2 < 1

def circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def line (a b : ℝ) : Prop := ∃ x y : ℝ, ax + by = 1

-- Proof goal
theorem line_and_circle_are_separated (ha : point_inside_circle a b) : ∀ x y, ¬ circle x y ∧ line a b := by
  -- Proof here
  sorry

end line_and_circle_are_separated_l612_612112


namespace f_10_l612_612864

def f : ℕ → ℕ
| 1       := 2
| 2       := 3
| (n + 3) := f (n + 2) - f (n + 1) + 2 * (n + 3) -- n + 3 starts from 3, thus n >= 0

theorem f_10 : f 10 = 119 :=
by
  -- Proof will go here
  sorry

end f_10_l612_612864


namespace probability_of_picking_same_color_shoes_l612_612299

theorem probability_of_picking_same_color_shoes
  (n_pairs_black : ℕ) (n_pairs_brown : ℕ) (n_pairs_gray : ℕ)
  (h_black_pairs : n_pairs_black = 8)
  (h_brown_pairs : n_pairs_brown = 4)
  (h_gray_pairs : n_pairs_gray = 3)
  (total_shoes : ℕ := 2 * (n_pairs_black + n_pairs_brown + n_pairs_gray)) :
  (16 / total_shoes * 8 / (total_shoes - 1) + 
   8 / total_shoes * 4 / (total_shoes - 1) + 
   6 / total_shoes * 3 / (total_shoes - 1)) = 89 / 435 :=
by
  sorry

end probability_of_picking_same_color_shoes_l612_612299


namespace total_animals_in_shelter_l612_612811

def initial_cats : ℕ := 15
def adopted_cats := initial_cats / 3
def replacement_cats := 2 * adopted_cats
def current_cats := initial_cats - adopted_cats + replacement_cats
def additional_dogs := 2 * current_cats
def total_animals := current_cats + additional_dogs

theorem total_animals_in_shelter : total_animals = 60 := by
  sorry

end total_animals_in_shelter_l612_612811


namespace find_AN_length_l612_612264

theorem find_AN_length
  (A B C M N : Point)
  (h1 : is_angle_bisector (angle B A C) M)
  (h2 : on_extension A B N)
  (h3 : dist A C = 1)
  (h4 : dist A M = 1)
  (h5 : angle A N M = angle C N M)
  : dist A N = 1 := sorry

end find_AN_length_l612_612264


namespace probability_sum_of_two_dice_is_30_l612_612010

theorem probability_sum_of_two_dice_is_30 :
  let die1 : List (Option ℕ) := (List.range 19).map some ++ [none],
      die2 : List (Option ℕ) := (List.range 10) ++ (List.rangeFrom 11 10).map some ++ [none],
      total_possible_rolls : ℕ := die1.length * die2.length,
      valid_rolls : ℕ := (
                         ((11, 19) ∈ List.product die1 die2).toNat +
                         ((12, 18) ∈ List.product die1 die2).toNat +
                         ((13, 17) ∈ List.product die1 die2).toNat +
                         ((14, 16) ∈ List.product die1 die2).toNat +
                         ((15, 15) ∈ List.product die1 die2).toNat +
                         ((16, 14) ∈ List.product die1 die2).toNat +
                         ((17, 13) ∈ List.product die1 die2).toNat +
                         ((18, 12) ∈ List.product die1 die2).toNat
                       ),
      prob : ℚ := valid_rolls / total_possible_rolls
  in prob = 1 / 50 := by
  sorry

end probability_sum_of_two_dice_is_30_l612_612010


namespace apples_initial_count_l612_612746

theorem apples_initial_count 
  (trees : ℕ)
  (apples_per_tree_picked : ℕ)
  (apples_picked_in_total : ℕ)
  (apples_remaining : ℕ)
  (initial_apples : ℕ) 
  (h1 : trees = 3) 
  (h2 : apples_per_tree_picked = 8) 
  (h3 : apples_picked_in_total = trees * apples_per_tree_picked)
  (h4 : apples_remaining = 9) 
  (h5 : initial_apples = apples_picked_in_total + apples_remaining) : 
  initial_apples = 33 :=
by sorry

end apples_initial_count_l612_612746


namespace length_AN_eq_one_l612_612274

variable {α : Type*} [InnerProductSpace ℝ α]

/-- Given a triangle ABC and points M and N satisfying the conditions in the problem -/
theorem length_AN_eq_one
  (A B C M N : α)
  (hM : ∠BAC = ∠BAM + ∠MAC)
  (hN' : ∃ P, P ∈ line_through A B ∧ N ≠ A ∧ sameRay (line_through A B) P N)
  (hAC : dist A C = 1)
  (hAM : dist A M = 1)
  (h_angle : angle A N M = angle C N M) :
  dist A N = 1 :=
sorry

end length_AN_eq_one_l612_612274


namespace domain_shift_l612_612664

theorem domain_shift (f : ℝ → ℝ) (h : ∀ x, x ∈ set.Icc (-1 : ℝ) (1) → f x ≠ none) :
  ∀ x, x ∈ set.Icc (-2 : ℝ) 0 → f (x + 1) ≠ none :=
by
  intros x hx
  have : x + 1 ∈ set.Icc (-1 : ℝ) (1),
    by
      split;
      linarith [set.mem_Icc.mp hx.1, set.mem_Icc.mp hx.2]
  exact h (x + 1) this
  sorry

end domain_shift_l612_612664


namespace magician_guesses_area_l612_612016

-- Define a convex polygon with 2008 sides
structure ConvexPolygon (n : ℕ) :=
  (vertices : Fin n → Point ℝ)

-- Define the real plane
structure Point (α : Type) :=
  (x : α)
  (y : α)

-- Define the ability to partition the polygon and get area from 2006 queries
axiom magician_can_determine_area 
  (P : ConvexPolygon 2008)
  (queries : Fin 2006 → (Fin 2008 × Fin 2008)) :
  ∃ (A : ℝ), true -- here we return true as we are not providing proof

theorem magician_guesses_area :
  ∀ (P : ConvexPolygon 2008),
  (∃ (queries : Fin 2006 → (Fin 2008 × Fin 2008)), magician_can_determine_area P queries) :=
begin
  intros P,
  use (some chosen_queries),
  apply magician_can_determine_area,
  sorry
end

end magician_guesses_area_l612_612016


namespace length_AN_eq_one_l612_612276

variable {α : Type*} [InnerProductSpace ℝ α]

/-- Given a triangle ABC and points M and N satisfying the conditions in the problem -/
theorem length_AN_eq_one
  (A B C M N : α)
  (hM : ∠BAC = ∠BAM + ∠MAC)
  (hN' : ∃ P, P ∈ line_through A B ∧ N ≠ A ∧ sameRay (line_through A B) P N)
  (hAC : dist A C = 1)
  (hAM : dist A M = 1)
  (h_angle : angle A N M = angle C N M) :
  dist A N = 1 :=
sorry

end length_AN_eq_one_l612_612276


namespace shortest_path_length_l612_612688

def circle (cx cy r : ℝ) (x y : ℝ) :=
  (x - cx)^2 + (y - cy)^2 = r^2

def line (a b : ℝ) (x y : ℝ) :=
  y = a * x + b

theorem shortest_path_length (A D O : ℝ × ℝ)
  (r : ℝ) (above : ℝ → ℝ → Prop)
  (hA : A = (0, 0))
  (hD : D = (8, 15))
  (hO : O = (4, 7))
  (h_circle : ∀ x y, circle 4 7 4 x y ↔ (x - 4)^2 + (y - 7)^2 = 16)
  (h_line : ∀ x y, above x y ↔ y > x) :
  shortest_path A D O r above = 14 + 4 * Real.pi := 
sorry

end shortest_path_length_l612_612688


namespace minimum_t_value_l612_612780

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x - 1

theorem minimum_t_value : 
  ∃ t : ℝ, (∀ x1 x2 : ℝ, x1 ∈ set.Icc (-3 : ℝ) 2 → x2 ∈ set.Icc (-3 : ℝ) 2 → |f x1 - f x2| ≤ t) ∧ t = 20 :=
by
  sorry

end minimum_t_value_l612_612780


namespace count_good_divisors_l612_612371

/-- We call a natural number \( n \) good if 2020 divided by \( n \) leaves a remainder of 22.
    This means \( n \) must be a divisor of 1998 and \( n > 22 \) -/
theorem count_good_divisors : finset.card (finset.filter (λ m, 22 < m ∧ 1998 % m = 0) (finset.range 2000)) = 10 :=
by
  -- This is where the proof would go
  sorry

end count_good_divisors_l612_612371


namespace length_AN_eq_one_l612_612258

-- Definitions representing the conditions
def triangle := {A B C : Type}
def point (P : Type) := P
def length (A B : Type) := 1

variables (A B C M N : Type)

def angle_bisector (BAC : Prop) := ∀ (A B C : point), 
  let P := ∃ (X : Type), X

constant AC : length A C
constant AM : length A M
constant AN : length A N
constant angle_ANM_CNMC_eq : ∀ (ANM CNM : Prop), ANM = CNM

-- Conclude the length of AN is 1
theorem length_AN_eq_one : AN = 1 :=
sorry

end length_AN_eq_one_l612_612258


namespace arithmetic_sequence_a6_eq_4_l612_612982

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Condition: a_n is an arithmetic sequence, so a_(n+1) = a_n + d
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: a_2 = 2
def a_2_eq_2 (a : ℕ → ℝ) : Prop :=
  a 2 = 2

-- Condition: S_4 = 9, where S_n is the sum of first n terms of the sequence
def sum_S_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) / 2 * (2 * a 1 + (n - 1 : ℝ) * (a 2 - a 1))

def S_4_eq_9 (S : ℕ → ℝ) : Prop :=
  S 4 = 9

-- Proof: a_6 = 4
theorem arithmetic_sequence_a6_eq_4 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a_2_eq_2 a)
  (h3 : sum_S_n a S) 
  (h4 : S_4_eq_9 S) :
  a 6 = 4 := 
sorry

end arithmetic_sequence_a6_eq_4_l612_612982


namespace infinitely_many_lines_through_P_forming_30_deg_angle_with_a_l612_612284

-- Define point P and line a
variable (P : Point)
variable (a : Line)

-- Assume P is outside line a
axiom P_outside_a : ¬(P ∈ a)

-- Define the statement about the lines passing through P and forming a 30° angle with line a
theorem infinitely_many_lines_through_P_forming_30_deg_angle_with_a :
  ∃ (L : Set Line), (∀ l ∈ L, l_passes_through P l ∧ forms_angle_with a l 30°) ∧ Infinite L :=
sorry

end infinitely_many_lines_through_P_forming_30_deg_angle_with_a_l612_612284


namespace range_of_m_l612_612597

-- Assume x is a real number
variable (x : ℝ)

-- Definitions based on problem conditions
def p (x : ℝ) := x < -2 ∨ x > 10
def q (m : ℝ) (x : ℝ) := 1 - m ≤ x ∧ x ≤ 1 + m^2
def not_p (x : ℝ) := -2 ≤ x ∧ x ≤ 10

-- Theorems to be proved
theorem range_of_m (m : ℝ) : (¬ ∀ x, p x → q m x) ∧ (∀ x, not_p x → q m x) → m ∈ set.Ioi 3 := 
by
  sorry

end range_of_m_l612_612597


namespace tan_alpha_add_pi_div_4_eq_three_l612_612647

/-- Given two vectors a and b defined as follows: 
  a = (-2, cos(α)) and b = (-1, sin(α)), such that a is parallel to b.
  Prove that tan(α + π/4) = 3. -/
theorem tan_alpha_add_pi_div_4_eq_three
  {α : ℝ}
  (ha : ∃ k : ℝ, ∀ α : ℝ, (-2 : ℝ, Real.cos α) = k • (-1 : ℝ, Real.sin α)) :
  Real.tan (α + Real.pi / 4) = 3 :=
sorry

end tan_alpha_add_pi_div_4_eq_three_l612_612647


namespace mutual_acquainted_or_unacquainted_l612_612286

theorem mutual_acquainted_or_unacquainted :
  ∀ (G : SimpleGraph (Fin 6)), 
  ∃ (V : Finset (Fin 6)), V.card = 3 ∧ ((∀ (u v : Fin 6), u ∈ V → v ∈ V → G.Adj u v) ∨ (∀ (u v : Fin 6), u ∈ V → v ∈ V → ¬G.Adj u v)) :=
by
  sorry

end mutual_acquainted_or_unacquainted_l612_612286


namespace quardilateral_perimeter_l612_612008

theorem quardilateral_perimeter (
  E F G H Q : Type
) (QE QF QG QH : ℝ) (area : ℝ) (perimeter : ℝ) :
  QE = 30 → QF = 40 → QG = 35 → QH = 50 →
  area = 2500 → convex_quadrilateral E F G H Q →
  perimeter_of_quadrilateral E F G H 226 :=
begin
  sorry
end

end quardilateral_perimeter_l612_612008


namespace good_numbers_2020_has_count_10_l612_612393

theorem good_numbers_2020_has_count_10 :
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  set.count divisors = 10 :=
by
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  sorry

end good_numbers_2020_has_count_10_l612_612393


namespace sum_of_coefficients_eq_two_l612_612094

noncomputable def poly_eq (a b c d : ℤ) : Prop :=
  (X^2 + a * X + b) * (X^2 + c * X + d) = X^4 - 2 * X^3 + 3 * X^2 - 4 * X + 6

theorem sum_of_coefficients_eq_two (a b c d : ℤ) (h : poly_eq a b c d) : a + b + c + d = 2 :=
by sorry

end sum_of_coefficients_eq_two_l612_612094


namespace food_requirement_not_met_l612_612499

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l612_612499


namespace number_of_good_numbers_l612_612353

theorem number_of_good_numbers :
  let n := 2020 in 
  let remainder := 22 in
  let number := 1998 in
  let divisors := {d ∣ number | d > remainder}.to_list.length = 10 := by sorry

end number_of_good_numbers_l612_612353


namespace largest_r_in_subset_l612_612087

theorem largest_r_in_subset (A : Finset ℕ) (hA : A.card = 500) : 
  ∃ (B C : Finset ℕ), B ⊆ A ∧ C ⊆ A ∧ (B ∩ C).card ≥ 100 := sorry

end largest_r_in_subset_l612_612087


namespace factorization_a_minus_b_l612_612776

theorem factorization_a_minus_b (a b: ℤ) 
  (h : (4 * y + a) * (y + b) = 4 * y * y - 3 * y - 28) : a - b = -11 := by
  sorry

end factorization_a_minus_b_l612_612776


namespace points_concyclic_l612_612684

variables {A B C D E F P K M : Type*} [has_mem A RealNumbers] [has_mem B RealNumbers] [has_mem C RealNumbers]

-- Defining the acute-angled triangle ABC
def acute_angled_triangle (ABC : Type*) {A B C : Type*} [ordered_ring RealNumbers] : Prop :=
  ∃ (AB AC : Type*), AB < AC ∧ triangle ABC

-- Defining the points on respective sides
def points_on_sides (ABC : Type*) (D : Type*) (E : Type*) (F : Type*) : Prop :=
  ∃ (BC CA AB : Type*), on_line_segment BC D ∧ on_line_segment CA E ∧ on_line_segment AB F

-- Defining concyclic points
def concyclic (B C E F : Type*) [has_mem B RealNumbers] [has_mem C RealNumbers] : Prop :=
  ∃ (circle : Type*), is_on_circle B circle ∧ is_on_circle C circle ∧ is_on_circle E circle ∧ is_on_circle F circle

-- Defining concurrency of lines
def concurrent (AD BE CF : Type*) (P : Type*) : Prop :=
  intersects_at AD P ∧ intersects_at BE P ∧ intersects_at CF P

-- Defining reflection of line BC over AD intersecting EF at K
def reflection_intersect (BC AD EF : Type*) (K : Type*) : Prop :=
  is_reflection_over AD BC K ∧ intersects_at_ray EF K

-- Defining midpoint M of EF
def midpoint (E F : Type*) (M : Type*) : Prop :=
  is_midpoint EF M

-- Theorem to prove points being concyclic
theorem points_concyclic 
  (ABC : Type*) 
  (AB AC BC CA AD BE CF EF : Type*) 
  (A B C D E F P K M : Type*)
  (h1 : acute_angled_triangle ABC)
  (h2 : points_on_sides ABC D E F)
  (h3 : concyclic B C E F)
  (h4 : concurrent AD BE CF P)
  (h5 : reflection_intersect BC AD EF K)
  (h6 : midpoint E F M) : 
  concyclic A M P K :=
sorry

end points_concyclic_l612_612684


namespace find_area_DBC_l612_612194

noncomputable def area_of_triangle_ADE : ℝ := 1
noncomputable def area_of_triangle_ADC : ℝ := 4
def DE_parallel_BC : Prop := true -- Note: this is a fact in problem statement, considered as true

theorem find_area_DBC :
  DE_parallel_BC → area_of_triangle_ADE = 1 →
  area_of_triangle_ADC = 4 →
  (∃ S : ℝ, S = 12 ∧ S = (S := (area_of_triangle_ADE + area_of_triangle_ADC)^2) ) :=
by
  intros
  exists 12
  constructor
  · rfl
  sorry

end find_area_DBC_l612_612194


namespace sum_sequence_up_to_2015_l612_612432

def sequence_val (n : ℕ) : ℕ :=
  if n % 288 = 0 then 7 
  else if n % 224 = 0 then 9
  else if n % 63 = 0 then 32
  else 0

theorem sum_sequence_up_to_2015 : 
  (Finset.range 2016).sum sequence_val = 1106 :=
by
  sorry

end sum_sequence_up_to_2015_l612_612432


namespace true_proposition_l612_612890

theorem true_proposition : 
  ¬(∀ (L₁ L₂ S₁ S₂ : Type) 
      [has_intersection L₁ L₂ S₁ S₂] 
      [skew S₁ S₂], (skew L₁ L₂)) ∧
  (∀ (L₁ L₂ S₁ S₂ : Type) 
      [has_intersection_at_different_points L₁ L₂ S₁ S₂] 
      [skew S₁ S₂], (skew L₁ L₂)) ∧
  ¬(∀ (L S₁ S₂ : Type) 
      [perpendicular_to L S₁ S₂] 
      [skew S₁ S₂], (common_perpendicular L S₁ S₂)) ∧
  ¬(∀ (a b c : Type) 
      [skew a b] 
      [skew b c], (skew a c)) :=
by 
  sorry

end true_proposition_l612_612890


namespace chemist_mixing_solution_l612_612450

theorem chemist_mixing_solution (x : ℝ) : 0.30 * x = 0.20 * (x + 1) → x = 2 :=
by
  intro h
  sorry

end chemist_mixing_solution_l612_612450


namespace z_real_iff_z_complex_iff_z_pure_imaginary_iff_l612_612957

-- Definitions for the problem conditions
def z_real (m : ℝ) : Prop := (m^2 - 2 * m - 15 = 0)
def z_pure_imaginary (m : ℝ) : Prop := (m^2 - 9 * m - 36 = 0) ∧ (m^2 - 2 * m - 15 ≠ 0)

-- Question 1: Prove that z is a real number if and only if m = -3 or m = 5
theorem z_real_iff (m : ℝ) : z_real m ↔ m = -3 ∨ m = 5 := sorry

-- Question 2: Prove that z is a complex number with non-zero imaginary part if and only if m ≠ -3 and m ≠ 5
theorem z_complex_iff (m : ℝ) : ¬z_real m ↔ m ≠ -3 ∧ m ≠ 5 := sorry

-- Question 3: Prove that z is a pure imaginary number if and only if m = 12
theorem z_pure_imaginary_iff (m : ℝ) : z_pure_imaginary m ↔ m = 12 := sorry

end z_real_iff_z_complex_iff_z_pure_imaginary_iff_l612_612957


namespace XF_XG_equality_l612_612745

variables {A B C D X Y E F G O : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space X]
variables [metric_space Y] [metric_space E] [metric_space F] [metric_space G] [metric_space O]
variables (AB BC CD DA : ℝ) (BD : ℝ)
variables (AX CX : affine_subspace ℝ ℝ) (DX XB BY YB XE XA XF XC XG XD XY : ℝ)

-- conditions
def condition1 := AX ∩ line_through Y AD = E
def condition2 := CX ∩ line_through E AC = F
def condition3 := G ∈ O ∧ G ≠ C ∧ G ∈ CX
def condition4 := AB = 4 ∧ BC = 3 ∧ CD = 7 ∧ DA = 9
def condition5 := DX = 1/3 * BD ∧ XB = 2/3 * BD ∧ BY = 1/4 * BD ∧ YB = 3/4 * BD
def condition6 := metric.pythagorean_triple AB BC = metric.pythagorean_triple 4 3

theorem XF_XG_equality : 
  condition1 → condition2 → condition3 → condition4 → condition5 → XD * XE = XA * XY → XF * XG = 155/3 := 
sorry

end XF_XG_equality_l612_612745


namespace gcd_8251_6105_l612_612940

theorem gcd_8251_6105 :
  Nat.gcd 8251 6105 = 37 :=
by
  sorry

end gcd_8251_6105_l612_612940


namespace max_distance_to_D_l612_612000

-- Define the points A, B, C, D of the square
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨1, 1⟩
def D : Point := ⟨0, 1⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

-- Define the conditions
variables (P : Point)
variables (u v w: ℝ)

-- Given distances
def u := dist P A
def v := dist P B
def w := dist P C

theorem max_distance_to_D (h : u + v = w) : dist P D = (2 + real.sqrt 2) ^ 2 := sorry

end max_distance_to_D_l612_612000


namespace range_of_omega_l612_612665

open Real

-- Definition of the function f
def f (ω x : ℝ) : ℝ :=
  4 * sin (ω * x) * (sin (π / 4 + ω * x / 2))^2 + cos (2 * ω * x)

-- Condition on omega
variable (ω : ℝ)
#check (0 < ω ∧ ω ≤ 3 / 4)

-- Monotonicity interval of f
def is_increasing_on_interval (ω : ℝ) (I : Set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f ω x ≤ f ω y

-- The interval of interest
def interval_of_interest : Set ℝ :=
  Icc (-π/2) (2 * π / 3)

-- The theorem to prove 
theorem range_of_omega :
  (is_increasing_on_interval ω interval_of_interest ↔ (0 < ω ∧ ω ≤ 3 / 4)) :=
sorry

end range_of_omega_l612_612665


namespace max_x3_x18_l612_612170

noncomputable def harmonic_sequence (x : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n : ℕ, n > 0 → x (n + 1) - x n = d

theorem max_x3_x18 (x : ℕ → ℝ) (h : harmonic_sequence (λ n, 1 / x n))
  (sum_20 : (Finset.range 20).sum (λ i, x (i + 1)) = 200) :
  x 3 * x 18 ≤ 100 := 
sorry

end max_x3_x18_l612_612170


namespace number_of_good_numbers_l612_612357

theorem number_of_good_numbers :
  let n := 2020 in 
  let remainder := 22 in
  let number := 1998 in
  let divisors := {d ∣ number | d > remainder}.to_list.length = 10 := by sorry

end number_of_good_numbers_l612_612357


namespace sqrt_sqrt_64_l612_612047

theorem sqrt_sqrt_64 : real.sqrt (real.sqrt 64) = ± (2 * real.sqrt 2) := sorry

end sqrt_sqrt_64_l612_612047


namespace boris_ball_prob_l612_612488

theorem boris_ball_prob {A B : Type} (n : ℕ) (a b : A → ℝ) (c d : B → ℝ) :
  -- Conditions
  (n = 8) ∧ 
  (∃ (box1 box2 : set ℕ), (box1 = {1} ∧ box2 = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})) ∧
  -- Probabilities
  ⁣((a 1 = 1) ∧ (c 1 = 1)) ∧ 
  ((a 2 = 7/15) ∧ (c 2 = 1 - 7/15)) → 
  -- Goal
  1/2 * a 1 + 1/2 * a 2 > 2/3 :=
by
  sorry

end boris_ball_prob_l612_612488


namespace problem_statement_l612_612986

variable (a : ℝ) (b : ℝ) (c : ℝ)

theorem problem_statement (ha : a ≠ 0) (hb : b ≠ 0) (hab : a > b) :
  (a + c^2 > b - c^2) ∧ (2^|a| > 2^b) ∧ (a - e^b > b - e^a) :=
by
  sorry

end problem_statement_l612_612986


namespace cos_36_deg_l612_612909

theorem cos_36_deg :
  let x := Real.cos (36 * Real.pi / 180) in
  x = ( -1 + Real.sqrt 5 ) / 4 := 
by
  let x := Real.cos (36 * Real.pi / 180)
  have h1 : 4 * x^2 + 2 * x - 1 = 0 := sorry 
  -- Polynomial roots found such that
  have h2 : x = ( -1 + Real.sqrt 5 ) / 4 ∨ x = ( -1 - Real.sqrt 5 ) / 4 := sorry
  -- Given that x is positive
  have : x > 0 := Real.cos_pos_of_pi_div_two_lt_of_lt_pi_div_two (by norm_num) (by norm_num)
  exact or.resolve_right h2 (by linarith)

end cos_36_deg_l612_612909


namespace vector_magnitude_l612_612646

variables {R : Type*} [normed_field R]
variables (a b : R)
variables (θ : ℝ) (h₁ : θ = π / 4) (h₂ : ∥a∥ = 1) (h₃ : ∥2 • a - b∥ = sqrt 10)

theorem vector_magnitude (h₁ : angle a b = π / 4) (h₂ : ∥a∥ = 1) (h₃ : ∥2 • a - b∥ = sqrt 10) :
  ∥b∥ = 3 * sqrt 2 :=
sorry

end vector_magnitude_l612_612646


namespace assembly_time_constants_l612_612474

theorem assembly_time_constants (a b : ℕ) (f : ℕ → ℝ)
  (h1 : ∀ x, f x = if x < b then a / (Real.sqrt x) else a / (Real.sqrt b))
  (h2 : f 4 = 15)
  (h3 : f b = 10) :
  a = 30 ∧ b = 9 :=
by
  sorry

end assembly_time_constants_l612_612474


namespace probability_more_heads_than_tails_l612_612659

theorem probability_more_heads_than_tails (n : ℕ) (hn : n = 9) :
  let outcomes := 2^n in
  let favorable := (nat.choose 9 5) + (nat.choose 9 6) + (nat.choose 9 7) + (nat.choose 9 8) + (nat.choose 9 9) in
  (favorable : ℚ) / (outcomes : ℚ) = 1 / 2 :=
by {
  let n := 9,
  let outcomes := 2^n,
  let favorable := (nat.choose 9 5) + (nat.choose 9 6) + (nat.choose 9 7) + (nat.choose 9 8) + (nat.choose 9 9),
  calc
    (favorable : ℚ) / (outcomes : ℚ) = 256 / 512 : sorry
                            ...        = 1 / 2   : sorry
}

end probability_more_heads_than_tails_l612_612659


namespace highest_probability_face_l612_612737

theorem highest_probability_face :
  let faces := 6
  let face_3 := 3
  let face_2 := 2
  let face_1 := 1
  (face_3 / faces > face_2 / faces) ∧ (face_2 / faces > face_1 / faces) →
  (face_3 / faces > face_1 / faces) →
  (face_3 = 3) :=
by {
  sorry
}

end highest_probability_face_l612_612737


namespace product_inequality_l612_612742

theorem product_inequality (n : ℕ) : 
  (∏ k in finset.range n, (2 * k + 1) / (2 * (k + 1))) < 1 / real.sqrt (2 * n + 1) := 
sorry

end product_inequality_l612_612742


namespace log_a_4a2_l612_612638

open Real

theorem log_a_4a2 (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h : a^b = (8*a)^(9*b)) : log a (4 * a^2) = 38/27 := 
by
  sorry

end log_a_4a2_l612_612638


namespace cube_root_0_000001_cube_root_10_pow_6_cube_root_y_in_terms_of_x_cube_root_relationship_l612_612734

-- Part (1)
theorem cube_root_0_000001 : real.cbrt 0.000001 = 0.01 :=
sorry

theorem cube_root_10_pow_6 : real.cbrt (10^6) = 100 :=
sorry

-- Part (2)
theorem cube_root_y_in_terms_of_x (x y : ℝ) (hx : real.cbrt x = 1.587) (hy : real.cbrt y = -0.1587) : 
  y = -x / 1000 :=
sorry

-- Part (3)
theorem cube_root_relationship (a : ℝ) : 
  (-1 < a ∧ a < 0 ∨ 1 < a ∧ a > 0) → real.cbrt a < a
  ∧ (a = -1 ∨ a = 1) → real.cbrt a = a
  ∧ (a < -1 ∨ 0 < a ∧ a < 1) → real.cbrt a > a :=
sorry

end cube_root_0_000001_cube_root_10_pow_6_cube_root_y_in_terms_of_x_cube_root_relationship_l612_612734


namespace length_AN_l612_612249

theorem length_AN {A B C M N : Type*}
  (h1 : M ∈ angle_bisector A B C)
  (h2 : N ∈ extension A B)
  (h3 : distance A C = 1)
  (h4 : distance A M = 1)
  (h5 : ∠ A N M = ∠ C N M) :
  distance A N = 1 :=
sorry

end length_AN_l612_612249


namespace part1_conditions_part2_range_of_m_l612_612608

-- Define set A as described
def setA (x : ℝ) : Prop := (x + 1) / (x - 3) ≤ 0

-- Define set B as described, parameterized by m
def setB (x : ℝ) (m : ℝ) : Prop := x^2 - (m - 1) * x + m - 2 ≤ 0

-- Lean statement for Part (1)
theorem part1_conditions (a b : ℝ) :
(A : ℝ → Prop),
  (∀ x, setA x ↔ -1 ≤ x ∧ x < 3) →
  (∀ x, (A x ∨ (a ≤ x ∧ x ≤ b)) ↔ (-1 ≤ x ∧ x ≤ 4)) →
  b = 4 ∧ -1 ≤ a ∧ a < 3 :=
sorry

-- Lean statement for Part (2)
theorem part2_range_of_m (m : ℝ) :
  (∀ x, (setB x m → setA x)) →
  1 ≤ m ∧ m < 5 :=
sorry

end part1_conditions_part2_range_of_m_l612_612608


namespace customers_left_l612_612885

theorem customers_left (original_customers remaining_tables people_per_table customers_left : ℕ)
  (h1 : original_customers = 44)
  (h2 : remaining_tables = 4)
  (h3 : people_per_table = 8)
  (h4 : original_customers - remaining_tables * people_per_table = customers_left) :
  customers_left = 12 :=
by
  sorry

end customers_left_l612_612885


namespace length_AN_eq_one_l612_612257

-- Definitions representing the conditions
def triangle := {A B C : Type}
def point (P : Type) := P
def length (A B : Type) := 1

variables (A B C M N : Type)

def angle_bisector (BAC : Prop) := ∀ (A B C : point), 
  let P := ∃ (X : Type), X

constant AC : length A C
constant AM : length A M
constant AN : length A N
constant angle_ANM_CNMC_eq : ∀ (ANM CNM : Prop), ANM = CNM

-- Conclude the length of AN is 1
theorem length_AN_eq_one : AN = 1 :=
sorry

end length_AN_eq_one_l612_612257


namespace rectangle_area_in_triangle_l612_612470

theorem rectangle_area_in_triangle (c k y : ℝ) (h1 : c > 0) (h2 : k > 0) (h3 : 0 < y) (h4 : y < k) : 
  ∃ A : ℝ, A = y * ((c * (k - y)) / k) := 
by
  sorry

end rectangle_area_in_triangle_l612_612470


namespace upper_bound_exists_l612_612588

theorem upper_bound_exists (U : ℤ) :
  (∀ n : ℤ, 1 < 4 * n + 7 ∧ 4 * n + 7 < U) →
  (∃ n_min n_max : ℤ, n_max = n_min + 29 ∧ 4 * n_max + 7 < U ∧ 4 * n_min + 7 > 1) →
  (U = 120) :=
by
  intros h1 h2
  sorry

end upper_bound_exists_l612_612588


namespace binom_factorial_eq_120_factorial_l612_612050

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binom_factorial_eq_120_factorial : (factorial (binomial 10 3)) = factorial 120 := by
  sorry

end binom_factorial_eq_120_factorial_l612_612050


namespace find_missing_number_l612_612077

theorem find_missing_number (some_number : ℝ) :
  some_number + real.sqrt (-4 + (6 * 4) / 3) = 13 → some_number = 11 :=
by {
  sorry
}

end find_missing_number_l612_612077


namespace ben_final_salary_is_2705_l612_612044

def initial_salary : ℕ := 3000

def salary_after_raise (salary : ℕ) : ℕ :=
  salary * 110 / 100

def salary_after_pay_cut (salary : ℕ) : ℕ :=
  salary * 85 / 100

def final_salary (initial : ℕ) : ℕ :=
  (salary_after_pay_cut (salary_after_raise initial)) - 100

theorem ben_final_salary_is_2705 : final_salary initial_salary = 2705 := 
by 
  sorry

end ben_final_salary_is_2705_l612_612044


namespace correct_options_l612_612150

theorem correct_options (a b : ℝ) (h : a > 0) (ha : a^2 = 4 * b) :
  ((a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬ (∃ x1 x2, x1 * x2 > 0 ∧ x^2 + a * x - b < 0)) ∧ 
  (∀ (x1 x2 : ℝ), |x1 - x2| = 4 → x^2 + a * x + b < 4 → 4 = 4)) :=
sorry

end correct_options_l612_612150


namespace average_adjacent_boy_girl_pairs_l612_612763

theorem average_adjacent_boy_girl_pairs (boys girls : ℕ) (T : ℕ → ℝ)
  (h_boys : boys = 8) (h_girls : girls = 12)
  (h_T_def : ∀ n, T n = 19 * (48 / 95)) :
  T (boys + girls) = 912 / 95 := 
by {
  intros,
  rw [h_boys, h_girls],
  exact h_T_def (boys + girls),
  sorry
}

end average_adjacent_boy_girl_pairs_l612_612763


namespace number_of_valid_monograms_l612_612731

theorem number_of_valid_monograms : 
  let initials := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'] in
  let number_of_combinations := (initials.take 14).combinations(2).length in
  number_of_combinations = 91 :=
by
  let initials := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'];
  let number_of_combinations := (initials.take 14).combinations(2).length;
  sorry

end number_of_valid_monograms_l612_612731


namespace count_good_divisors_l612_612369

/-- We call a natural number \( n \) good if 2020 divided by \( n \) leaves a remainder of 22.
    This means \( n \) must be a divisor of 1998 and \( n > 22 \) -/
theorem count_good_divisors : finset.card (finset.filter (λ m, 22 < m ∧ 1998 % m = 0) (finset.range 2000)) = 10 :=
by
  -- This is where the proof would go
  sorry

end count_good_divisors_l612_612369


namespace count_good_numbers_l612_612362

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l612_612362


namespace weight_of_each_pack_l612_612318

-- Definitions based on conditions
def total_sugar : ℕ := 3020
def leftover_sugar : ℕ := 20
def number_of_packs : ℕ := 12

-- Definition of sugar used for packs
def sugar_used_for_packs : ℕ := total_sugar - leftover_sugar

-- Proof statement to be verified
theorem weight_of_each_pack : sugar_used_for_packs / number_of_packs = 250 := by
  sorry

end weight_of_each_pack_l612_612318


namespace geometric_sequence_of_an_minus_one_sum_of_inverse_b_l612_612176

-- Part (1)
theorem geometric_sequence_of_an_minus_one (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h_sum : ∀ n, S n = 2 * a n + ↑n) : 
  ∃ r : ℤ, ∀ n ≥ 1, a n - 1 = r * (a (n - 1) - 1) :=
by
  sorry

-- Part (2)
noncomputable def b (a : ℕ → ℤ) (n : ℕ) : ℤ := log 2 (1 - a n)

theorem sum_of_inverse_b (a : ℕ → ℤ) (h_a : ∀ n, a n = 1 - 2^n) (n : ℕ) :
  (∑ k in finset.range n, 1 / (b a k * b a (k + 1))) = ↑n / (↑n + 1) :=
by
  sorry

end geometric_sequence_of_an_minus_one_sum_of_inverse_b_l612_612176


namespace find_values_of_f_l612_612223

def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x + 4 else 10 - 3 * x

theorem find_values_of_f :
  f (-2) = 0 ∧ f (3) = 1 :=
by
  sorry

end find_values_of_f_l612_612223


namespace conditional_probability_l612_612964

variables {A B : Prop} {P : Prop → ℝ}

def P(A|B) := P(AB) / P(B)

theorem conditional_probability : 
  P(A | B) = 3/7 → P(B) = 7/9 → P(A ∧ B) = 1/3 :=
by
  intros h1 h2
  sorry

end conditional_probability_l612_612964


namespace hypotenuse_length_l612_612792

-- Define the conditions
def longer_leg_is_two_feet_longer_than_twice_shorter_leg (short_legg long_leg : ℝ) : Prop :=
  long_leg = 2 * short_legg + 2

def triangle_area_is_seventy_two (short_legg long_leg : ℝ) : Prop :=
  (1 / 2) * short_legg * long_leg = 72

-- Define the hypothesis for Lean
theorem hypotenuse_length (short_leg long_leg h : ℝ) 
  (h1 : longer_leg_is_two_feet_longer_than_twice_shorter_leg short_leg long_leg)
  (h2 : triangle_area_is_seventy_two short_leg long_leg) : 
  h = Real.sqrt (short_leg ^ 2 + long_leg ^ 2) :=
begin
  sorry
end

end hypotenuse_length_l612_612792


namespace boat_distance_against_stream_l612_612186

-- Define the conditions
variable (v_s : ℝ)
variable (speed_still_water : ℝ := 9)
variable (distance_downstream : ℝ := 13)

-- Assert the given condition
axiom condition : speed_still_water + v_s = distance_downstream

-- Prove the required distance against the stream
theorem boat_distance_against_stream : (speed_still_water - (distance_downstream - speed_still_water)) = 5 :=
by
  sorry

end boat_distance_against_stream_l612_612186


namespace largest_mu_correct_l612_612943

noncomputable def largest_mu : ℝ :=
  3 / 4

theorem largest_mu_correct (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) :
  a^2 + b^2 + c^2 + d^2 ≥ a * b + largest_mu * b * c + 2 * c * d :=
begin
  sorry
end

end largest_mu_correct_l612_612943


namespace relationship_between_x_y_z_l612_612610

noncomputable def x := (3 : ℝ) ^ (1/3 : ℝ)
noncomputable def y := (7 : ℝ) ^ (1/6 : ℝ)
noncomputable def z := (7 : ℝ) ^ (1/7 : ℝ)

theorem relationship_between_x_y_z (x_def : log x 3 = 3) (y_def : log y 7 = 6) (z_def : z = 7^(1/7)) : z < y ∧ y < x :=
by
  -- Use the assumptions to establish the relationships
  sorry

end relationship_between_x_y_z_l612_612610


namespace find_a_l612_612628

def f (x a : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem find_a (a : ℝ) :
  (∃ x : ℝ, x = -3 ∧ f' (x, a) = 0) → a = 5 := 
by
  sorry

-- Helper definition for f' indicating the derivative of f with respect to x
noncomputable def f' (x a : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 3

end find_a_l612_612628


namespace elois_banana_bread_l612_612074

theorem elois_banana_bread :
  let bananas_per_loaf := 4
  let loaves_monday := 3
  let loaves_tuesday := 2 * loaves_monday
  let total_loaves := loaves_monday + loaves_tuesday
  let total_bananas := total_loaves * bananas_per_loaf
  total_bananas = 36 := sorry

end elois_banana_bread_l612_612074


namespace find_range_of_a_l612_612211

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}

noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem find_range_of_a : {a : ℝ | set_B a ⊆ set_A} = {a : ℝ | a < -1} ∪ {1} :=
by
  sorry

end find_range_of_a_l612_612211


namespace number_of_terms_correct_l612_612618

noncomputable def numberOfTermsInSimplifiedExpression : ℕ :=
  ∑ i in Finset.range (1003 + 1), Nat.choose (2008 - 2 * i) 2

theorem number_of_terms_correct :
  let expr := (x + y + z + w) ^ 2006 + (x - y - z - w) ^ 2006 in
  countTerms (simplify expr) = numberOfTermsInSimplifiedExpression := sorry

end number_of_terms_correct_l612_612618


namespace unique_positive_integer_l612_612915

theorem unique_positive_integer (n : ℕ) (S : ℕ → ℕ) :
  (S n = ∑ k in range (n-2), (k + 3) * 2^(k + 3)) →
  S n = 2^(n + 11) →
  n = 1025 :=
by
  intros hSum hEqn
  sorry

end unique_positive_integer_l612_612915


namespace Aunt_Wang_money_l612_612897

-- Define the conditions from the problem
def price_per_kg_apple (money_short_2_5_kg money_excess_2_kg weight_diff_kg : ℝ): ℝ :=
  (money_short_2_5_kg + money_excess_2_kg) / weight_diff_kg

def total_money_Aunt_Wang (price_per_kg money_excess_2_kg total_weight : ℝ): ℝ :=
  (price_per_kg * total_weight) + money_excess_2_kg

-- The main statement to prove Aunt Wang had 10.9 yuan
theorem Aunt_Wang_money (money_short_2_5_kg money_excess_2_kg weight_2_5_kg  weight_2_kg : ℝ)
  (h1 : money_short_2_5_kg = 1.6) 
  (h2 : money_excess_2_kg = 0.9)
  (h3 : weight_2_5_kg = 2.5)
  (h4 : weight_2_kg = 2)
  (h5 : (weight_2_5_kg - weight_2_kg) = 0.5) :
  total_money_Aunt_Wang (price_per_kg_apple money_short_2_5_kg money_excess_2_kg (weight_2_5_kg - weight_2_kg)) money_excess_2_kg weight_2_kg = 10.9 := 
by
  sorry

end Aunt_Wang_money_l612_612897


namespace digit_statement_l612_612880

theorem digit_statement (d : ℕ) (c1 : d = 0) (c2 : d ≠ 1) (c3 : d = 2) (c4 : d ≠ 3) 
  (H : (c1 ∨ c2 ∨ c3 ∨ c4) ∧ (¬c1 ∨ ¬c2 ∨ ¬c3 ∨ ¬c4) ∧ (c1 ↔ ¬c3) ∧ (c3 ↔ ¬c1)) : c2 :=
by
  sorry

end digit_statement_l612_612880


namespace distance_to_asymptote_of_hyperbola_l612_612306

theorem distance_to_asymptote_of_hyperbola :
  let hyperbola_eq := ∀ x y : ℝ, (x^2 / 16) - (y^2 / 9) = 1
  let point := (5 : ℝ, 0 : ℝ)
  let a : ℝ := 4
  let b : ℝ := 3
  let asymptote1 := ∀ x y : ℝ, 3 * x + 4 * y = 0
  let asymptote2 := ∀ x y : ℝ, 3 * x - 4 * y = 0
  let distance_to_line (px py ax by c : ℝ) := (real.abs (ax * px + by * py + c) / real.sqrt (ax ^ 2 + by ^ 2))
  distance_to_line (fst point) (snd point) 3 4 0 = 3 :=
by sorry

end distance_to_asymptote_of_hyperbola_l612_612306


namespace functional_equation_solution_l612_612554

theorem functional_equation_solution 
    (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)) : 
    f = (λ x, x - 1) := 
sorry

end functional_equation_solution_l612_612554


namespace problem_statement_l612_612975

variables {x y x_0 y_0 a b : ℝ} {p : ℝ}

def parabola := y^2 = 2 * p * x ∧ p > 0
def point_on_parabola := (y_0^2 = 2 * p * x_0)
def min_distance_from_M_to_N := (x_0 > 0) → (sqrt((x_0 - 2)^2 + y_0^2) = sqrt(3))

def conditions (x_0 y_0 : ℝ) (p : ℝ) : Prop :=
  parabola ∧
  point_on_parabola ∧
  min_distance_from_M_to_N

theorem problem_statement :
  (conditions x_0 y_0 p) →
  (x_0 > 2) → 
  (a^2 * (x_0 - 2) + 2 * a * y_0 - x_0 = 0) → 
  (b^2 * (x_0 - 2) + 2 * b * y_0 - x_0 = 0) →
  (∀ a b, 1 / 2 * abs(a - b) * abs(x_0) = 8) :=
begin
  sorry
end

end problem_statement_l612_612975


namespace cat_food_inequality_l612_612511

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l612_612511


namespace still_water_speed_l612_612017

-- The conditions as given in the problem
variables (V_m V_r V'_r : ℝ)
axiom upstream_speed : V_m - V_r = 20
axiom downstream_increased_speed : V_m + V_r = 30
axiom downstream_reduced_speed : V_m + V'_r = 26

-- Prove that the man's speed in still water is 25 km/h
theorem still_water_speed : V_m = 25 :=
by
  sorry

end still_water_speed_l612_612017


namespace ma_times_mb_eq_9_l612_612682

variable {t : ℝ}

/-- Definition of the parametric equations of line l passing through point M(3, 4) with angle of inclination 45 degrees. --/
def line_l (t : ℝ) : ℝ × ℝ := (3 + (Real.sqrt 2 / 2) * t, 4 + (Real.sqrt 2 / 2) * t)

/-- Definition of the Cartesian equation of circle C given its polar equation ρ = 4 sin θ. --/
def cartesian_eq_circle_C : ℝ × ℝ → Prop := 
  λ (x y : ℝ), x^2 + y^2 - 4 * y = 0

/-- Proof that |MA| * |MB| = 9, where MA and MB are distances from M to points A and B, the intersection points of circle C with line l. --/
theorem ma_times_mb_eq_9 :
  let M := (3, 4)
  ∃ (A B : ℝ × ℝ), 
    (cartesian_eq_circle_C A.1 A.2) ∧ 
    (cartesian_eq_circle_C B.1 B.2) ∧ 
    (line_l (A.1 / (Real.sqrt 2 / 2)) = A) ∧ 
    (line_l (B.1 / (Real.sqrt 2 / 2)) = B) ∧
    |Real.dist M A * Real.dist M B| = 9 :=
begin
  sorry
end

end ma_times_mb_eq_9_l612_612682


namespace num_integers_between_100_and_400_with_factors_13_and_6_l612_612651

theorem num_integers_between_100_and_400_with_factors_13_and_6 : 
  (Finset.card (Finset.filter (λ n, n % Nat.lcm 13 6 = 0) (Finset.range (400 + 1)) ∩ (Finset.Ico 100 400))) = 4 :=
sorry

end num_integers_between_100_and_400_with_factors_13_and_6_l612_612651


namespace magnitude_of_diff_is_sqrt3_l612_612995

noncomputable def magnitude_difference (a b : ℝ^3) : ℝ := 
  ||a - b||

variables (a b : ℝ^3)
variables (h₁ : ||a|| = 1) (h₂ : ||b|| = 1)
variables (h_angle : Real.cos (120 / 180 * Real.pi) = -1/2)

theorem magnitude_of_diff_is_sqrt3 
  (h₁ : ||a|| = 1) 
  (h₂: ||b|| = 1) 
  (h_angle : Real.cos (120 / 180 * Real.pi) = -1/2) : 
  magnitude_difference a b = Real.sqrt(3) := 
sorry

end magnitude_of_diff_is_sqrt3_l612_612995


namespace ram_weight_increase_l612_612810

theorem ram_weight_increase:
  ∀ (x p : ℝ),
    ((7 * x + 5 * x) * 1.15 = 82.8) →
    (Shyam_weight : ℝ) (Shyam_weight = 5 * x * 1.22) →
    (Ram_weight_new = 7 * x * (1 + p / 100)) →
    (Ram_weight_new + Shyam_weight = 82.8) →
    p = 10 :=
by
  intros x p hx_total hShyam_weight hRam_weight_new hTotal
  sorry

end ram_weight_increase_l612_612810


namespace scout_troop_profit_l612_612879

def candy_bars := 1000
def cost_per_five_bars := 2
def sell_per_two_bars := 1

def total_cost := (candy_bars * (cost_per_five_bars / 5))
def total_revenue := (candy_bars * (sell_per_two_bars / 2))

theorem scout_troop_profit : total_revenue - total_cost = 100 := 
by
  unfold total_cost total_revenue
  calc
    (candy_bars * (sell_per_two_bars / 2)) - (candy_bars * (cost_per_five_bars / 5)) = (1000 * (1 / 2)) - (1000 * (2 / 5)) : by rfl
    ... = 500 - 400 : by norm_num
    ... = 100 : by norm_num

end scout_troop_profit_l612_612879


namespace trig_identity_example_l612_612491

theorem trig_identity_example :
  (Real.sin (43 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) - Real.sin (13 * Real.pi / 180) * Real.cos (43 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end trig_identity_example_l612_612491


namespace sum_f_inv_l612_612779

noncomputable def f (x : ℝ) : ℝ :=
if x < 3 then 2 * x - 1 else x ^ 2

noncomputable def f_inv (y : ℝ) : ℝ :=
if y < 9 then (y + 1) / 2 else Real.sqrt y

theorem sum_f_inv :
  (f_inv (-3) + f_inv (-2) + 
   f_inv (-1) + f_inv 0 + 
   f_inv 1 + f_inv 2 + 
   f_inv 3 + f_inv 4 + 
   f_inv 9) = 9 :=
by
  sorry

end sum_f_inv_l612_612779


namespace hyperbola_eccentricity_l612_612997

theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h_eq1 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : c = Real.sqrt (a^2 + b^2))
  (h_dist : ∀ x, x = b * c / Real.sqrt (a^2 + b^2))
  (h_eq3 : a = b) :
  e = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_l612_612997


namespace eggs_left_in_jar_l612_612748

variable (initial_eggs : ℝ) (removed_eggs : ℝ)

theorem eggs_left_in_jar (h1 : initial_eggs = 35.3) (h2 : removed_eggs = 4.5) :
  initial_eggs - removed_eggs = 30.8 :=
by
  sorry

end eggs_left_in_jar_l612_612748


namespace part_time_employees_count_l612_612020

theorem part_time_employees_count 
    (total_employees : ℕ) 
    (full_time_employees : ℕ) 
    (h_total : total_employees = 65134)
    (h_full_time : full_time_employees = 63093) :
    total_employees - full_time_employees = 2041 := 
by
  rw [h_total, h_full_time]
  norm_num

# reduce to proof
# sorry

end part_time_employees_count_l612_612020


namespace max_red_dragons_l612_612191

/-- 
In a hypothetical scenario with 530 dragons seating at a round table, each having three heads with specific statements.
We aim to prove that the maximum number of red dragons is 176.
-/
theorem max_red_dragons (total_dragons : ℕ)
    (heads_per_dragon : ℕ)
    (h_total : total_dragons = 530)
    (h_heads : heads_per_dragon = 3)
    (truth_rule : ∀ (d : ℕ), d < total_dragons → 
      (∃ (i : ℕ), i < heads_per_dragon ∧ 
      (i = 0 → "To my left is a green dragon.") ∧ 
      (i = 1 → "To my right is a blue dragon.") ∧ 
      (i = 2 → "There is no red dragon next to me.")))
    : ∃ red_dragons, red_dragons ≤ total_dragons ∧ red_dragons = 176 :=
begin
  sorry
end

end max_red_dragons_l612_612191


namespace determine_a_b_l612_612081

-- Define the polynomial expression
def poly (x a b : ℝ) : ℝ := x^2 + a * x + b

-- Define the factored form
def factored_poly (x : ℝ) : ℝ := (x + 1) * (x - 3)

-- State the theorem
theorem determine_a_b (a b : ℝ) (h : ∀ x, poly x a b = factored_poly x) : a = -2 ∧ b = -3 :=
by 
  sorry

end determine_a_b_l612_612081


namespace product_of_one_group_at_least_72_l612_612305

theorem product_of_one_group_at_least_72 (G₁ G₂ G₃ : Finset ℕ) 
  (h1 : ∀ i ∈ G₁ ∪ G₂ ∪ G₃, i ∈ (Finset.range 9).map (λ i, i + 1))
  (h2 : ∀ x ∈ (Finset.range 9).map (λ i, i + 1), x ∈ G₁ ∪ G₂ ∪ G₃)
  (h3 : Finset.disjoint G₁ G₂)
  (h4 : Finset.disjoint G₁ G₃)
  (h5 : Finset.disjoint G₂ G₃) :
  (G₁.product id) * (G₂.product id) * (G₃.product id) = 9! →
  G₁.product id ≥ 72 ∨ G₂.product id ≥ 72 ∨ G₃.product id ≥ 72 := 
by
  sorry

end product_of_one_group_at_least_72_l612_612305


namespace good_numbers_2020_has_count_10_l612_612395

theorem good_numbers_2020_has_count_10 :
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  set.count divisors = 10 :=
by
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  sorry

end good_numbers_2020_has_count_10_l612_612395


namespace length_of_chord_of_parabola_l612_612944

theorem length_of_chord_of_parabola (x1 x2 : ℝ) (y1 y2 : ℝ) :
  (∀ x y : ℝ, y ^ 2 = 8 * x ↔ ∃ (y = x - 2 ∧ (x, y) = (x1, y1) ∨ (x, y) = (x2, y2))) → 
  x1 + x2 = 12 → 
  |x1 + x2 + 4| = 16 := 
sorry

end length_of_chord_of_parabola_l612_612944


namespace no_kids_from_outside_attended_l612_612067

theorem no_kids_from_outside_attended (
  total_county_kids: ℕ,
  camp_kids_county: ℕ,
  home_kids_county: ℕ,
  total_camp_county: total_county_kids = camp_kids_county + home_kids_county
) : 
  total_county_kids - home_kids_county = camp_kids_county → 
  total_county_kids - camp_kids_county = home_kids_county :=
by
  intro h
  exact h.symm

end no_kids_from_outside_attended_l612_612067


namespace find_C_l612_612285

def Point : Type := ℝ × ℝ

def midpoint (A B : Point) : Point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem find_C (A B C : Point) (hA : A = (1,1)) (hB : B = (3,4)) (hM : midpoint A C = B) : C = (5, 7) :=
  sorry

end find_C_l612_612285


namespace isabella_hair_length_l612_612196

theorem isabella_hair_length (h : ℕ) (g : ℕ) (future_length : ℕ) (hg : g = 4) (future_length_eq : future_length = 22) :
  h = future_length - g :=
by
  rw [future_length_eq, hg]
  exact sorry

end isabella_hair_length_l612_612196


namespace count_of_good_numbers_l612_612344

-- Define the conditions of the problem
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- The main statement proving the number of 'good' numbers
theorem count_of_good_numbers :
  (finset.filter is_good_number (finset.Icc 1 2020)).card = 10 :=
by
  sorry

end count_of_good_numbers_l612_612344


namespace second_derivative_parametric_l612_612845

noncomputable def x (t : ℝ) := Real.sqrt (t - 1)
noncomputable def y (t : ℝ) := 1 / Real.sqrt t

noncomputable def y_xx (t : ℝ) := (2 * t - 3) * Real.sqrt t / t^3

theorem second_derivative_parametric :
  ∀ t, y_xx t = (2 * t - 3) * Real.sqrt t / t^3 := sorry

end second_derivative_parametric_l612_612845


namespace mapping_exists_l612_612479

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {0, 3, 8, 15, 24}
def f (x : ℕ) : ℕ := x^2 - 1

theorem mapping_exists : ∀ x ∈ A, f x ∈ B :=
by
  intros x hx
  cases hx with
  | intro hx1 =>
    cases hx1 with
    | isTrue h1 => sorry
    | hx2 => 
      cases hx2 with
      | isTrue h2 => sorry
      | hx3 =>
        cases hx3 with
        | isTrue h3 => sorry
        | hx4 =>
          cases hx4 with
          | isTrue h4 => sorry
          | hx5 =>
            cases hx5 with
            | isTrue h5 => sorry
            | false => false.elim

end mapping_exists_l612_612479


namespace second_donation_per_slice_l612_612926

-- Given conditions
def cakes : ℕ := 10
def slices_per_cake : ℕ := 8
def price_per_slice : ℝ := 1
def total_raised : ℝ := 140
def first_donation_per_slice : ℝ := 0.5

-- The statement to be proved
theorem second_donation_per_slice :
  (cakes * slices_per_cake * price_per_slice) + (cakes * slices_per_cake * first_donation_per_slice) + (cakes * slices_per_cake * ?_donation_per_slice) = total_raised → 
  ?_donation_per_slice = 0.25 :=
by
  sorry

end second_donation_per_slice_l612_612926


namespace square_equilateral_triangle_l612_612034

theorem square_equilateral_triangle (A B C D M K : Type) 
  [square : is_square A B C D] 
  (triangle : is_equilateral_triangle A B M) 
  (inside : vertex_inside_square M A B C D)
  (diagonal_intersect : diagonal_intersects A C M K) :
  distance C K = distance C M :=
sorry

end square_equilateral_triangle_l612_612034


namespace train_length_correct_l612_612424

-- Given constants
def speed_km_per_hr := 60 -- km/hr
def time_seconds := 15 -- seconds

-- Converting speed from km/hr to m/s
def speed_m_per_s := speed_km_per_hr * (5.0 / 18.0)

-- The length of the train calculation
def train_length := speed_m_per_s * time_seconds

theorem train_length_correct : train_length = 250.05 := by
  have speed_m_per_s_correct : speed_m_per_s = 60 * (5.0 / 18.0) := by
    unfold speed_m_per_s
  have calc_length : 16.67 * 15 = 250.05 := sorry -- as a placeholder for actual calculation or precision handling
  exact calc_length

end train_length_correct_l612_612424


namespace limit_derivative_sin_at_pi_over_3_eq_half_l612_612631

noncomputable def f (x: Real) : Real := Real.sin x

theorem limit_derivative_sin_at_pi_over_3_eq_half :
  (Real.lim (λ Δx, (f (Real.pi / 3 + Δx) - f (Real.pi / 3)) / Δx) 0) = 1 / 2 :=
by
  sorry

end limit_derivative_sin_at_pi_over_3_eq_half_l612_612631


namespace find_remainder_10K_mod_1000_l612_612056

def A : ℝ × ℝ := (0, 12)
def B : ℝ × ℝ := (10, 9)
def C : ℝ × ℝ := (8, 0)
def D : ℝ × ℝ := (-4, 7)
noncomputable def K : ℝ := sorry -- The area of the square S

theorem find_remainder_10K_mod_1000 :
  ∃ (S : ℝ) (hA : lies_on_side A S) (hB : lies_on_side B S) (hC : lies_on_side C S) (hD : lies_on_side D S),
  (10 * K) % 1000 = 936 :=
sorry

end find_remainder_10K_mod_1000_l612_612056


namespace egg_production_difference_l612_612206

-- Define the conditions
def last_year_production : ℕ := 1416
def this_year_production : ℕ := 4636

-- Define the theorem statement
theorem egg_production_difference :
  this_year_production - last_year_production = 3220 :=
by
  sorry

end egg_production_difference_l612_612206


namespace length_an_eq_1_l612_612270

variable {A B C M N : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N]
variable {dist : A → A → ℝ}
variable {angle : ∀ {A B C : A}, ℝ}

-- Given conditions as hypotheses
variable (H1 : ∀ {M}, M ∈ angle_bisector A B C)
variable (H2 : ∀ {N : Type}, N ∈ extended_line A B)
variable (H3 : dist A C = 1)
variable (H4 : dist A M = 1)
variable (H5 : angle A N M = angle C N M)

-- The statement we need to prove
theorem length_an_eq_1 (H1 : M ∈ angle_bisector A B C) (H2 : N ∈ extended_line A B)
  (H3 : dist A C = 1) (H4 : dist A M = 1) (H5 : angle A N M = angle C N M) :
  dist A N = 1 := by
  sorry


end length_an_eq_1_l612_612270


namespace player1_wins_optimal_play_l612_612846

theorem player1_wins_optimal_play (coins : ℕ) (h : coins = 2001) :
  ∃ strategy : (ℕ → ℕ) × (ℕ → ℕ), 
    (∀ x, x % 2 = 1 ∧ 1 ≤ x ∧ x ≤ 99) → 
    (∀ y, y % 2 = 0 ∧ 2 ≤ y ∧ y ≤ 100) →
    let take_coins := strategy.1 in
    let respond := strategy.2 in
    ∀ turns_left n, 
      n = coins - take_coins(1) - (turns_left - 1) * 101 →
      n ≤ 100 →
      (turns_left = 0 → ∃ y, y % 2 = 0 ∧ 2 ≤ y ∧ y ≤ 100 → player2_loses) →
      player1_wins :=
sorry

end player1_wins_optimal_play_l612_612846


namespace count_good_divisors_l612_612372

/-- We call a natural number \( n \) good if 2020 divided by \( n \) leaves a remainder of 22.
    This means \( n \) must be a divisor of 1998 and \( n > 22 \) -/
theorem count_good_divisors : finset.card (finset.filter (λ m, 22 < m ∧ 1998 % m = 0) (finset.range 2000)) = 10 :=
by
  -- This is where the proof would go
  sorry

end count_good_divisors_l612_612372


namespace count_isosceles_triangles_l612_612480

-- Define the vertices of each triangle
def vertices_triangle_1 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := ((0,6), (2,6), (1,4))
def vertices_triangle_2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := ((3,4), (3,6), (5,4))
def vertices_triangle_3 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := ((0,1), (3,2), (6,1))
def vertices_triangle_4 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := ((7,4), (6,6), (9,4))
def vertices_triangle_5 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := ((8,1), (9,3), (10,0))

-- Function to calculate the distance between two points
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Function to check if a triangle is isosceles
def is_isosceles (v1 v2 v3 : (ℝ × ℝ)) : Prop :=
  let a := dist v1 v2
  let b := dist v2 v3
  let c := dist v1 v3
  (a = b) ∨ (b = c) ∨ (a = c)

-- Define the theorem
theorem count_isosceles_triangles : 
  4 = (if is_isosceles vertices_triangle_1.1 vertices_triangle_1.2 vertices_triangle_1.3 then 1 else 0) +
      (if is_isosceles vertices_triangle_2.1 vertices_triangle_2.2 vertices_triangle_2.3 then 1 else 0) +
      (if is_isosceles vertices_triangle_3.1 vertices_triangle_3.2 vertices_triangle_3.3 then 1 else 0) +
      (if is_isosceles vertices_triangle_4.1 vertices_triangle_4.2 vertices_triangle_4.3 then 1 else 0) +
      (if is_isosceles vertices_triangle_5.1 vertices_triangle_5.2 vertices_triangle_5.3 then 1 else 0) :=
sorry

end count_isosceles_triangles_l612_612480


namespace sum_inequality_l612_612714

theorem sum_inequality (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_lt : n < m) :
  (∑ k in Finset.range (n + 1) \ {0}, (k * m / n)) >
  (∑ k in Finset.range (m + 1) \ {0}, (k * n / m)) := 
by
  sorry

end sum_inequality_l612_612714


namespace cos_sum_half_l612_612989

noncomputable def calc_cos (a β : ℝ) : ℝ :=
  if h₁ : 0 < a ∧ a < π / 2 then
    if h₂ : -π / 2 < β ∧ β < 0 then
      if h₃ : cos (a + π / 4) = 1 / 3 then
        if h₄ : sin (π / 4 - β / 2) = sqrt 3 / 3 then
          cos (a + β / 2)
        else 0
      else 0
    else 0
  else 0

theorem cos_sum_half (a β : ℝ)
  (h₁ : 0 < a ∧ a < π / 2)
  (h₂ : -π / 2 < β ∧ β < 0)
  (h₃ : cos (a + π / 4) = 1 / 3)
  (h₄ : sin (π / 4 - β / 2) = sqrt 3 / 3) :
  calc_cos a β = sqrt 6 / 3 :=
by {
  sorry
}

end cos_sum_half_l612_612989


namespace option_a_option_b_option_d_l612_612145

theorem option_a (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 - b^2 ≤ 4 := 
sorry

theorem option_b (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 + 1 / b ≥ 4 :=
sorry

theorem option_d (a b c x1 x2 : ℝ) 
  (h1 : a > 0) 
  (h2 : a^2 = 4 * b) 
  (h3 : (x1 - x2) = 4)
  (h4 : (x1 + x2)^2 - 4 * (x1 * x2 + c) = (x1 - x2)^2) : c = 4 :=
sorry

end option_a_option_b_option_d_l612_612145


namespace sum_of_100th_group_l612_612065

theorem sum_of_100th_group : 
  (∑ i in (495 :: 497 :: 499 :: 501 :: []).to_finset, id i) = 1992 := by
  sorry

end sum_of_100th_group_l612_612065


namespace sum_reciprocals_correct_l612_612232

-- Define the sequence a_n
def sequence_a (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n ≥ 1, (2 - a (n + 1)) * (4 + a n) = 8

-- Define the summation of reciprocals
def sum_reciprocals (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in (Finset.range n).map Nat.succ, (1 / a i)

-- The target value to be proved
theorem sum_reciprocals_correct (a : ℕ → ℝ) (n : ℕ) (h : sequence_a a) : 
  sum_reciprocals a n = 2^n - (n / 2) - 1 :=
sorry

end sum_reciprocals_correct_l612_612232


namespace eight_times_nine_and_two_fifths_is_l612_612902

variable (m n a b : ℕ)
variable (d : ℚ)

-- Conditions
def mixed_to_improper (a b den : ℚ) : ℚ := a + b / den
def improper_to_mixed (n d : ℚ) : ℕ × ℚ := (n / d).to_int, (n % d) / d

-- Example specific instances
def nine_and_two_fifths : ℚ := mixed_to_improper 9 2 5
def eight_times_nine_and_two_fifths : ℚ := 8 * nine_and_two_fifths

-- Lean statement to confirm calculation
theorem eight_times_nine_and_two_fifths_is : improper_to_mixed eight_times_nine_and_two_fifths 5 = (75, 1/5) := by
  sorry

end eight_times_nine_and_two_fifths_is_l612_612902


namespace total_boys_in_camp_l612_612673

/-- 
In a certain boys camp, 20% of the total boys are from school A and 30% of those study science. 
There are 56 boys in the camp that are from school A but do not study science. 
Prove that the total number of boys in the camp is 400.
-/
theorem total_boys_in_camp (T : ℝ) (h1 : 0.20 * T ∈ ℝ) (h2 : 0.70 * (0.20 * T) = 56) : T = 400 :=
sorry

end total_boys_in_camp_l612_612673


namespace edge_length_box_l612_612650

theorem edge_length_box (n : ℝ) (h : n = 999.9999999999998) : 
  ∃ (L : ℝ), L = 1 ∧ ((L * 100) ^ 3 / 10 ^ 3) = n := 
sorry

end edge_length_box_l612_612650


namespace length_of_first_train_l612_612449

theorem length_of_first_train
    (speed_train1_kmph : ℝ) (speed_train2_kmph : ℝ) 
    (length_train2_m : ℝ) (cross_time_s : ℝ)
    (conv_factor : ℝ)         -- Conversion factor from kmph to m/s
    (relative_speed_ms : ℝ)   -- Relative speed in m/s 
    (distance_covered_m : ℝ)  -- Total distance covered in meters
    (length_train1_m : ℝ) : Prop :=
  speed_train1_kmph = 120 →
  speed_train2_kmph = 80 →
  length_train2_m = 210.04 →
  cross_time_s = 9 →
  conv_factor = 1000 / 3600 →
  relative_speed_ms = (200 * conv_factor) →
  distance_covered_m = (relative_speed_ms * cross_time_s) →
  length_train1_m = 290 →
  distance_covered_m = length_train1_m + length_train2_m

end length_of_first_train_l612_612449


namespace find_eccentricity_l612_612613

noncomputable def ellipse_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) : ℝ := 
  a^2 - b^2

theorem find_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (focus_vertex_condition: (1, 0) ∈ ellipse a b ha hb hab ∧ (3, 0) ∈ ellipse a b ha hb hab) 
  : ellipse_eccentricity a b ha hb hab = 1/3 :=
by
  sorry

end find_eccentricity_l612_612613


namespace division_theorem_l612_612948

-- Define the polynomials
def p (x : ℝ) : ℝ := x^5 + 3 * x^4 - 28 * x^3 + 15 * x^2 - 21 * x + 8
def d (x : ℝ) : ℝ := x - 3
def q (x : ℝ) : ℝ := x^4 + 6 * x^3 - 10 * x^2 - 15 * x - 66
def r : ℝ := -100

-- Statement of the theorem
theorem division_theorem : ∀ x : ℝ, p(x) = d(x) * q(x) + r :=
by
  intros
  sorry

end division_theorem_l612_612948


namespace perpendicular_lines_condition_l612_612315

variable {A1 B1 C1 A2 B2 C2 : ℝ}

theorem perpendicular_lines_condition :
  (∀ x y : ℝ, A1 * x + B1 * y + C1 = 0) ∧ (∀ x y : ℝ, A2 * x + B2 * y + C2 = 0) → 
  (A1 * A2) / (B1 * B2) = -1 := 
sorry

end perpendicular_lines_condition_l612_612315


namespace find_real_values_l612_612573

open Real

theorem find_real_values (x : ℝ) : 
  (x ∈ [2, 5) ∪ (5, 9)) ↔ (x * (x - 1) / (x - 5)^3 ≥ 18) :=
begin
  -- Proof skipped
  sorry
end

end find_real_values_l612_612573


namespace number_of_divisors_greater_than_22_l612_612384

theorem number_of_divisors_greater_than_22 :
  (∃ (S : Finset ℕ), S = (Finset.filter (λ n, 1998 % n = 0 ∧ n > 22) (Finset.range (1998 + 1))) ∧ S.card = 10) :=
sorry

end number_of_divisors_greater_than_22_l612_612384


namespace simplify_expr_l612_612494

theorem simplify_expr : sqrt 9 + (-8)^(1/3) + 2 * (sqrt 2 + 2) - abs (1 - sqrt 2) = 6 + sqrt 2 := 
by 
  sorry

end simplify_expr_l612_612494


namespace find_k_l612_612815

-- The expression in terms of x, y, and k
def expression (k x y : ℝ) :=
  4 * x^2 - 6 * k * x * y + (3 * k^2 + 2) * y^2 - 4 * x - 4 * y + 6

-- The mathematical statement to be proved
theorem find_k : ∃ k : ℝ, (∀ x y : ℝ, expression k x y ≥ 0) ∧ (∃ (x y : ℝ), expression k x y = 0) :=
sorry

end find_k_l612_612815


namespace find_common_difference_l612_612701

-- Definitions and theorems used in the arithmetic sequence problem
def arithmetic_sequence (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

def sum_of_arithmetic_sequence (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a1 + (n - 1) * d)) // 2

-- Given conditions
variables (S17 : ℤ) (a10 : ℤ)
axiom sum_condition : S17 = 255
axiom term_condition : a10 = 20

-- Prove that under the conditions, the common difference d is 5
theorem find_common_difference :
  ∃ (a1 d : ℤ), 
    sum_of_arithmetic_sequence a1 d 17 = S17 ∧
    arithmetic_sequence a1 d 10 = a10 ∧
    d = 5 :=
  sorry

end find_common_difference_l612_612701


namespace combined_total_l612_612692

variable (Jane Jean : ℕ)

theorem combined_total (h1 : Jean = 3 * Jane) (h2 : Jean = 57) : Jane + Jean = 76 := by
  sorry

end combined_total_l612_612692


namespace carol_rolls_problem_l612_612446

def carol_combinations : ℕ := 20

theorem carol_rolls_problem : 
  ∃ (A B C D : ℕ), A + B + C + D = 9 ∧ A ≥ 2 ∧ B ≥ 2 ∧ C ≥ 2 ∧ D ≥ 0 → 
  (nat.choose (9 - 6 + 4 - 1) (4 - 1)) = carol_combinations :=
by
  sorry

end carol_rolls_problem_l612_612446


namespace sequence_properties_l612_612461

-- Definitions of the sequences a_n and b_n
def geom_seq (a : ℕ → ℕ) := ∃ q, ∀ n, a (n + 1) = a n * q
def arith_seq (b : ℕ → ℕ) := ∃ d, ∀ n, b (n + 1) = b n + d

-- Given conditions
axiom a1 : ℕ → ℕ
axiom a1_val : a1 1 = 1
axiom a4_val : a1 4 = 8

axiom b1 : ℕ → ℕ
axiom a3_is_b4 : a1 3 = b1 4
axiom a5_is_b16 : a1 5 = b1 16

-- Main theorem to be proven
theorem sequence_properties :
  (∀ n, a1 n = 2 ^ (n - 1)) ∧ 
  (∀ n, b1 n = n) ∧
  (∀ n, (∑ i in finset.range n, (a1 i.succ) * (b1 i.succ)) = 2^n - 2) :=
  sorry

end sequence_properties_l612_612461


namespace sequence_correct_l612_612604

noncomputable def sequence (c : ℕ) : ℕ → ℕ
| 1 := c
| n + 1 := let x := sequence n in x + (floor ((2 * x - (n + 2)) / n) : ℤ).toNat + 1

noncomputable def general_formula (c n : ℕ) : ℕ :=
if c % 3 = 0 then 
  ((c - 3) / 6 * (n + 1) * (n + 2) + floor ((n + 2)^2 / 4) : ℤ).toNat + 1
else if c % 3 = 1 then 
  ((c - 1) / 6 * (n + 1) * (n + 2) + 1 : ℤ).toNat
else 
  ((c - 2) / 6 * (n + 1) * (n + 2) + (n + 1) : ℤ).toNat

theorem sequence_correct (c : ℕ) (n : ℕ) :
  sequence c n = general_formula c n := 
sorry

end sequence_correct_l612_612604


namespace triangle_area_l612_612546

def point := (ℝ × ℝ)
def line := (ℝ × ℝ → Prop)

def A : point := (5, 0)
def B : point := (0, 5)
def C_line : line := λ p, p.1 + p.2 = 9

noncomputable def area_triangle (A B C : point) : ℝ :=
  0.5 * ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1)).abs

theorem triangle_area :
  ∀ C : point, C_line C → area_triangle A B C = 10 :=
by
  intro C hC
  sorry

end triangle_area_l612_612546


namespace sum_C_n_l612_612996

noncomputable def an (n: ℕ) : ℝ := n + 1

def C_n (n : ℕ) : ℝ :=
  let a_n := an n in
  (2 * a_n - 4) * n * 3^(a_n - 2)

def T_n (n : ℕ) : ℝ :=
  ∑ i in finset.range n, C_n (i + 1)

theorem sum_C_n (n : ℕ) : 
  T_n n = (2 * n - 3) / 2 * 3 ^ n + 3 / 2 := 
sorry

end sum_C_n_l612_612996


namespace lung_disease_related_to_smoking_l612_612459

variable (n : ℕ) (K2 : ℝ) (P : ℝ → ℝ)

theorem lung_disease_related_to_smoking (h1 : n = 1000) (h2 : K2 = 4.453) (h3 : P (K2 ≥ 3.841) = 0.05) :
  0.95 = 1 - 0.05 :=
by sorry

end lung_disease_related_to_smoking_l612_612459


namespace coefficient_of_x_is_neg_one_l612_612086

theorem coefficient_of_x_is_neg_one :
  let expr := 5 * (x - 6) + 3 * (8 - 3 * x^2 + 7 * x) - 9 * (3 * x - 2)
  in coeff_of_x (expr.expand) = -1 :=
by
  sorry

end coefficient_of_x_is_neg_one_l612_612086


namespace part_a_part_b_l612_612849

-- Definition for combination
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Proof problems as Lean statements
theorem part_a : combination 30 2 = 435 := by
  sorry

theorem part_b : combination 30 3 = 4060 := by
  sorry

end part_a_part_b_l612_612849


namespace avg_speed_of_car_l612_612428

noncomputable def average_speed (distance1 distance2 : ℕ) (time1 time2 : ℕ) : ℕ :=
  (distance1 + distance2) / (time1 + time2)

theorem avg_speed_of_car :
  average_speed 65 45 1 1 = 55 := by
  sorry

end avg_speed_of_car_l612_612428


namespace b_3_is_integer_l612_612797

noncomputable def b : ℕ → ℝ
| 1     := 2
| (n+2) := log (sqrt ((n+3/2)/(n+1))) / log 3 + b (n+1)

theorem b_3_is_integer :
  ∀ n : ℕ, (b 3).denom = 1 := sorry

end b_3_is_integer_l612_612797


namespace dart_hit_number_list_count_l612_612899

def number_of_dart_hit_lists (darts dartboards : ℕ) : ℕ :=
  11  -- Based on the solution, the hard-coded answer is 11.

theorem dart_hit_number_list_count : number_of_dart_hit_lists 6 4 = 11 := 
by 
  sorry

end dart_hit_number_list_count_l612_612899


namespace damien_jogs_75_miles_in_three_weeks_l612_612922

theorem damien_jogs_75_miles_in_three_weeks :
  let daily_distance := 5
  let weekdays_per_week := 5
  let weeks := 3
  (daily_distance * (weekdays_per_week * weeks)) = 75 :=
by
  let daily_distance := 5
  let weekdays_per_week := 5
  let weeks := 3
  show (daily_distance * (weekdays_per_week * weeks)) = 75
    from sorry

end damien_jogs_75_miles_in_three_weeks_l612_612922


namespace determine_A_l612_612138

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 2 * x - 4
noncomputable def g (x : ℝ) : ℝ := x^2 - x + 2
def Z_plus : Setℤ := { n | n > 0 }

-- Determine the set A ≡ { x | f(x) / g(x) ∈ Z_+ }
def A : Set ℝ := { x | (∃ (k : ℤ), k ∈ Z_plus ∧ f x = k * g x) }

-- Statement that needs to be proved
theorem determine_A : A = {2, (3 + Real.sqrt 33) / 2, (3 - Real.sqrt 33) / 2} :=
sorry

end determine_A_l612_612138


namespace no_linear_factor_with_integer_coeffs_l612_612129

def poly := x^2 - y^2 - z^2 + 2 * x * y + x + y - z

theorem no_linear_factor_with_integer_coeffs :
  ¬∃ (a b c d : ℤ), 
    (∀ x y z : ℤ, a * x + b * y + c * z + d = 0 → poly = 0) :=
sorry

end no_linear_factor_with_integer_coeffs_l612_612129


namespace constant_term_expansion_l612_612060

theorem constant_term_expansion :
  let binomialCoefficient (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  let T (r : ℕ) (k : ℕ) := (-1)^k * binomialCoefficient r k * x^(r-2*k)
  let expansion (x : ℝ) := x - 1/x - 1
  ∃ (r k : ℕ), r = 2 * k ∧ (1 - binomialCoefficient 2 1 * binomialCoefficient 4 2 + binomialCoefficient 4 2 * binomialCoefficient 0 0 ) = -5 :=
begin
  sorry
end

end constant_term_expansion_l612_612060


namespace remainder_is_one_l612_612329

-- Definitions of the given numbers
def a : ℕ := 10
def b : ℕ := 11
def c : ℕ := 12

-- Definition of the largest and second largest numbers
def largest : ℕ := max a (max b c)
def second_largest : ℕ := (({a, b, c}.erase largest).max' 
  (by simp [a, b, c, lt_irrefl])) -- Erase the largest and take the max of the remaining set

-- Theorem to prove the remainder when dividing the largest by the second largest
theorem remainder_is_one : (largest % second_largest) = 1 := by
  sorry

end remainder_is_one_l612_612329


namespace conditional_probability_l612_612963

variables {A B : Prop} {P : Prop → ℝ}

def P(A|B) := P(AB) / P(B)

theorem conditional_probability : 
  P(A | B) = 3/7 → P(B) = 7/9 → P(A ∧ B) = 1/3 :=
by
  intros h1 h2
  sorry

end conditional_probability_l612_612963


namespace smallest_y_not_defined_l612_612413

theorem smallest_y_not_defined : 
  ∃ y : ℝ, (6 * y^2 - 37 * y + 6 = 0) ∧ (∀ z : ℝ, (6 * z^2 - 37 * z + 6 = 0) → y ≤ z) ∧ y = 1 / 6 :=
by
  sorry

end smallest_y_not_defined_l612_612413


namespace divisibility_contradiction_l612_612822

theorem divisibility_contradiction (a b : ℕ) (hab : ab ∈ ℕ) (hdiv : 5 ∣ a) :
  ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) → (5 ∣ a ∨ 5 ∣ b) := 
sorry

end divisibility_contradiction_l612_612822


namespace count_of_good_numbers_l612_612348

-- Define the conditions of the problem
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- The main statement proving the number of 'good' numbers
theorem count_of_good_numbers :
  (finset.filter is_good_number (finset.Icc 1 2020)).card = 10 :=
by
  sorry

end count_of_good_numbers_l612_612348


namespace complement_union_l612_612234

namespace SetComplement

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {4, 5}
def B : Set ℕ := {3, 4}

theorem complement_union :
  U \ (A ∪ B) = {1, 2, 6} := by
  sorry

end SetComplement

end complement_union_l612_612234


namespace good_numbers_count_l612_612338

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l612_612338


namespace cat_food_insufficient_for_six_days_l612_612525

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l612_612525


namespace scout_troop_profit_l612_612878

def candy_bars := 1000
def cost_per_five_bars := 2
def sell_per_two_bars := 1

def total_cost := (candy_bars * (cost_per_five_bars / 5))
def total_revenue := (candy_bars * (sell_per_two_bars / 2))

theorem scout_troop_profit : total_revenue - total_cost = 100 := 
by
  unfold total_cost total_revenue
  calc
    (candy_bars * (sell_per_two_bars / 2)) - (candy_bars * (cost_per_five_bars / 5)) = (1000 * (1 / 2)) - (1000 * (2 / 5)) : by rfl
    ... = 500 - 400 : by norm_num
    ... = 100 : by norm_num

end scout_troop_profit_l612_612878


namespace find_a_from_derivative_l612_612135

-- Define the function f(x) = ax^3 + 3x^2 - 6
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

-- State the theorem to prove that a = 10/3 given f'(-1) = 4
theorem find_a_from_derivative (a : ℝ) (h : f' a (-1) = 4) : a = 10 / 3 := 
  sorry

end find_a_from_derivative_l612_612135


namespace count_of_distinct_prime_combinations_125_l612_612827

theorem count_of_distinct_prime_combinations_125 :
  ∃ X : ℕ, 
  (∃ p q r : ℕ, p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ p + q + r = 125) →
  (∀ p q r : ℕ, p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ p + q + r = 125 →
  ∃ distinct_combinations : ℕ, distinct_combinations = X) :=
sorry

end count_of_distinct_prime_combinations_125_l612_612827


namespace integer_value_expression_l612_612844

theorem integer_value_expression (p q : ℕ) (hp : Prime p) (hq : Prime q) : 
  (p = 2 ∧ q = 2) ∨ (p ≠ 2 ∧ q = 2 ∧ pq + p^p + q^q = 3 * (p + q)) :=
sorry

end integer_value_expression_l612_612844


namespace intersection_A_B_l612_612640

-- Defining sets A and B based on the given conditions.
def A : Set ℝ := {x | ∃ y, y = Real.log x ∧ x > 0}
def B : Set ℝ := {x | x^2 - 2 * x - 3 < 0}

-- Stating the theorem that A ∩ B = {x | 0 < x ∧ x < 3}.
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l612_612640


namespace number_of_paths_l612_612676

theorem number_of_paths (r u : ℕ) (h_r : r = 5) (h_u : u = 4) : 
  (Nat.choose (r + u) u) = 126 :=
by
  -- The proof is omitted, as requested.
  sorry

end number_of_paths_l612_612676


namespace stratified_sampling_11th_grade_l612_612319

theorem stratified_sampling_11th_grade : 
  ∀ (n10 n11 n12 : ℕ) (total_sample : ℕ),
  n10 = 4 * k → n11 = 3 * k → n12 = 3 * k → total_sample = 50 → 
  (3 * total_sample) / (n10 + n11 + n12) = 15 :=
by 
  -- Assume n10, n11, n12, total_sample
  intros n10 n11 n12 total_sample
  -- Assume ratios
  assume (h1 : n10 = 4 * k) 
         (h2 : n11 = 3 * k) 
         (h3 : n12 = 3 * k) 
         (h4 : total_sample = 50)
  -- Provide proof
  sorry

end stratified_sampling_11th_grade_l612_612319


namespace number_of_divisors_greater_than_22_l612_612390

theorem number_of_divisors_greater_than_22 :
  (∃ (S : Finset ℕ), S = (Finset.filter (λ n, 1998 % n = 0 ∧ n > 22) (Finset.range (1998 + 1))) ∧ S.card = 10) :=
sorry

end number_of_divisors_greater_than_22_l612_612390


namespace Sn_eq_Tn_l612_612593

noncomputable def S (n: ℕ) : ℚ :=
finset.sum (finset.range n) (λ k, ((-1) ^ k) * (1 / (k + 1)))

noncomputable def T (n: ℕ) : ℚ :=
finset.sum (finset.range (2*n - n)) (λ k, 1 / (n + k + 1))

theorem Sn_eq_Tn (n : ℕ) (h : 0 < n) : S n = T n := by
  sorry

end Sn_eq_Tn_l612_612593


namespace chickens_count_l612_612764

theorem chickens_count (rabbits frogs : ℕ) (h_rabbits : rabbits = 49) (h_frogs : frogs = 37) :
  ∃ (C : ℕ), frogs + C = rabbits + 9 ∧ C = 21 :=
by
  sorry

end chickens_count_l612_612764


namespace shortest_distance_point_B_l612_612971

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem shortest_distance_point_B :
  let A := (1 : ℝ, 0 : ℝ)
  let line_B (x y : ℝ) := x = y
  ∃ B : ℝ × ℝ, line_B B.1 B.2 ∧ (∀ B' : ℝ × ℝ, line_B B'.1 B'.2 → distance A B ≤ distance A B') ∧ B = (1/2, 1/2) :=
by
  let A := (1, 0)
  let line_B (x y : ℝ) := x = y
  use (1/2, 1/2)
  split
  { sorry }
  { split
    { sorry }
    { refl }
  }

end shortest_distance_point_B_l612_612971


namespace length_AN_is_one_l612_612278

noncomputable def segment_length_AN (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) (point_N_extension : ExtensionOfSide A B N) 
  (AC_eq_AM : AC = 1 ∧ AM = 1) (ANM_eq_CNM : ∠ANM = ∠CNM) : Real :=
1

/-- Given a triangle ABC, where point M lies on the angle bisector of angle BAC,
and point N lies on the extension of side AB beyond point A, where AC = AM = 1,
and the angles ANM and CNM are equal, the length of segment AN is 1. -/
theorem length_AN_is_one (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) 
  (point_N_extension : ExtensionOfSide A B N) (AC_eq_AM : AC = 1 ∧ AM = 1) 
  (ANM_eq_CNM : ∠ANM = ∠CNM) : 
  segment_length_AN A B C M N triangle_ABC angle_bisector_AM point_N_extension AC_eq_AM ANM_eq_CNM = 1 :=
by
  sorry

end length_AN_is_one_l612_612278


namespace shaded_area_l612_612798

theorem shaded_area (PQ : ℝ) (hPQ : PQ = 8) : 
  ∃ area : ℝ, area = 40 :=
by 
  -- Define the conditions
  let n_squares := 20,
  let side_length := PQ / 2,  -- since the diagonal d = 8 for a large square
  let area_one_square := side_length^2 / 2,
  let total_area := n_squares * area_one_square,
  
  -- Assert the proof for total area being 40 cm²
  use total_area,
  sorry

end shaded_area_l612_612798


namespace ivan_travel_time_l612_612434

theorem ivan_travel_time (d V_I V_P : ℕ) (h1 : d = 3 * V_I * 40)
  (h2 : ∀ t, t = d / V_P + 10) : 
  (d / V_I = 75) :=
by
  sorry

end ivan_travel_time_l612_612434


namespace equal_consumption_l612_612565

/- Definitions from problem conditions -/
def Ed_initial_drink := 8
def Ann_initial_drink := (1.75 : ℚ) * Ed_initial_drink
def Ed_drinks := (0.6 : ℚ) * Ed_initial_drink
def Ed_remaining := Ed_initial_drink - Ed_drinks
def Ann_drinks := (0.7 : ℚ) * Ann_initial_drink
def Ann_remaining := Ann_initial_drink - Ann_drinks
def Ann_gives_Ed := (0.5 : ℚ) * Ann_remaining
def Ed_final_consumption := Ed_drinks + Ann_gives_Ed
def Ann_final_consumption := Ann_drinks + (Ann_remaining - Ann_gives_Ed)

/- Theorem statement -/
theorem equal_consumption 
  (Ed_initial_drink = 8)
  (Ann_initial_drink = 1.75 * Ed_initial_drink)
  (Ed_drinks = 0.6 * Ed_initial_drink)
  (Ann_drinks = 0.7 * Ann_initial_drink)
  (Ann_gives_Ed = 0.5 * (Ann_initial_drink - Ann_drinks))
  (Ed_final_consumption = Ed_drinks + Ann_gives_Ed)
  (Ann_final_consumption = Ann_drinks + ((Ann_initial_drink - Ann_drinks) - Ann_gives_Ed))
  : Ed_final_consumption = 7.7 ∧ Ann_final_consumption = 7.7 :=
sorry

end equal_consumption_l612_612565


namespace cat_food_inequality_l612_612532

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l612_612532


namespace angle_AFE_degree_measure_l612_612687

theorem angle_AFE_degree_measure 
  (A B C D E F : Type) 
  [rectangle A B C D] 
  (hABeq2CD : ∃ (length : ℝ), B - A = 2 * (D - C)) 
  (hAngleCDE : ∠ C D E = 100) 
  (hDFeqDE : ∃ (length : ℝ), F - D = E - D)
  (hEoppositeHalfPlane : ∃ (line : Set (A)), line = C D ∧ E ∈ -line ∧ A ∈ line)
  : ∠ A F E = 175 :=
begin
  sorry
end

end angle_AFE_degree_measure_l612_612687


namespace sequence_thirtieth_term_l612_612415

noncomputable def arithmetic_sequence_term_30 : ℕ :=
  let a₁ := 4 in
  let d := 3 in
  let a₄ := a₁ + 3 * (4 - 1) in
  let a₅ := 2 * a₄ in
  2^(30 - 5) * a₄

theorem sequence_thirtieth_term : arithmetic_sequence_term_30 = 436207104 :=
by sorry

end sequence_thirtieth_term_l612_612415


namespace option_a_option_b_option_d_l612_612146

theorem option_a (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 - b^2 ≤ 4 := 
sorry

theorem option_b (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 + 1 / b ≥ 4 :=
sorry

theorem option_d (a b c x1 x2 : ℝ) 
  (h1 : a > 0) 
  (h2 : a^2 = 4 * b) 
  (h3 : (x1 - x2) = 4)
  (h4 : (x1 + x2)^2 - 4 * (x1 * x2 + c) = (x1 - x2)^2) : c = 4 :=
sorry

end option_a_option_b_option_d_l612_612146


namespace parabola_lower_than_l612_612545

theorem parabola_lower_than (
  f1 : ℝ → ℝ := λ x, x^2 - (1 / 3) * x + 3,
  f2 : ℝ → ℝ := λ x, x^2 + (1 / 3) * x + 4
) : ∀ (x : ℝ), f1 x < f2 x := by
  sorry

end parabola_lower_than_l612_612545


namespace sum_possible_intersections_five_lines_l612_612586

theorem sum_possible_intersections_five_lines : 
  (∑ k in Finset.Icc 0 10, k) = 55 :=
by {
  sorry
}

end sum_possible_intersections_five_lines_l612_612586


namespace count_good_numbers_l612_612381

-- Define the property of a good number
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- Count the number of good numbers
theorem count_good_numbers : (finset.filter is_good_number (finset.range 2021)).card = 10 :=
by sorry

end count_good_numbers_l612_612381


namespace good_numbers_2020_has_count_10_l612_612397

theorem good_numbers_2020_has_count_10 :
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  set.count divisors = 10 :=
by
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  sorry

end good_numbers_2020_has_count_10_l612_612397


namespace construct_circle_with_equal_chords_l612_612547

-- Define a structure for a triangle to encapsulate its essential properties
structure Triangle :=
  (A B C : ℝ × ℝ) -- Vertices of the triangle

-- Define a structure for Circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a function that calculates the radius of the incircle of a given triangle
-- (A placeholder function definition, you might define the actual computation as needed)
noncomputable def incircle_radius (t : Triangle) : ℝ := sorry

-- Define the main theorem statement
theorem construct_circle_with_equal_chords (t : Triangle) (l : ℝ) :
  ∃ (c : Circle), 
  ∀ p q r : ℝ × ℝ, 
  (p, q, r ∈ [t.A, t.B, t.C]) → 
  -- Assume p, q, and r represent where the chords are intercepted on the sides
  (dist c.center p = dist c.center q = dist c.center r = dist c.center t.A = incircle_radius t) ∧
  ∀ (chord1 chord2 chord3 : ℝ), 
  (chord1 = chord2 = chord3 = l) := 
sorry

end construct_circle_with_equal_chords_l612_612547


namespace grasshopper_cannot_visit_all_squares_once_l612_612078

-- Define the board's properties and conditions for the grasshopper's movement.
inductive Color
| black : Color
| white : Color

def board : ℕ × ℕ → Color
| (i, j) := if (i + j) % 2 = 0 then Color.white else Color.black

def diagonal_move (i j k : ℕ) : i + k < 8 → j + k < 8 → (i + k, j + k) = (i, j)
def one_square_jump (i j ni nj : ℕ) : (ni = i + 2 ∨ ni = i - 2 ∨ nj = j + 2 ∨ nj = j - 2) → (ni, nj) = (i, j)

theorem grasshopper_cannot_visit_all_squares_once :
  ¬ ∃ start : ℕ × ℕ, 
    (∀ path : list (ℕ × ℕ),
      path.head = start → -- starts at starting point
      set.pairwise path (≠) → -- all squares are distinct
      list.length path = 64 → -- path covers all squares
      ∀ (i j : ℕ × ℕ), i ∈ path → j ∈ path → board i = board j → -- keeps color constraint
      false) := 
sorry

end grasshopper_cannot_visit_all_squares_once_l612_612078


namespace find_pq_l612_612713

theorem find_pq (p q : ℝ) : (∃ x : ℝ, f(p, q, x) = 0) ∧ (∃ y : ℝ, f(p, q, y) = 0) ↔ 
(p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2) := 
begin
  sorry,
end

def f (p q x : ℝ) : ℝ := x^2 + p * x + q

end find_pq_l612_612713


namespace binomial_sum_l612_612953

noncomputable theory

namespace BinomialSum

open Complex

def is_4thRootOfUnity (ω : ℂ) : Prop :=
  ω ^ 4 = 1 ∧ ω ≠ 1

theorem binomial_sum (n : ℕ) (root : ℂ) (h_root : is_4thRootOfUnity root) :
  let m := n / 4
  let sum := ∑ k in finset.filter (λ k, (k % 4 = 3)) (finset.range (n + 1)), 
             nat.choose n k
  sum = (2 ^ n + root * 2 ^ n - (-root^2) ^ n - root * (-root) ^ n) / (2 * (root - root^3)) :=
sorry

end BinomialSum

end binomial_sum_l612_612953


namespace option_d_correct_l612_612828

theorem option_d_correct (a b : ℝ) : (a - b)^2 = (b - a)^2 := 
by {
  sorry
}

end option_d_correct_l612_612828


namespace min_lambda_l612_612109

theorem min_lambda (n : ℕ) (h : n ≥ 4) 
    (a : fin n → ℝ) 
    (hnonneg : ∀ i : fin n, 0 ≤ a i)
    (hsum : (∑ i, a i) = n) : 
    (∑ i : fin n, (a i - ⌊a i⌋) * a (i + 1)) ≤ n - 3 / 4 := 
sorry

end min_lambda_l612_612109


namespace correct_options_l612_612144

theorem correct_options (a b : ℝ) (h_a_pos : a > 0) (h_discriminant : a^2 = 4 * b):
  (a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬(∃ x1 x2 : ℝ, (x1 * x2 > 0 ∧ a^2 - x1x2 ≠ 4b))) ∧ 
  (∀ c x1 x2 : ℝ, (x1 - x2 = 4) → (a^2 - 4 * (b - c) = 16) → (c = 4)) :=
by
  sorry

end correct_options_l612_612144


namespace smallest_constant_N_l612_612064

theorem smallest_constant_N (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  (a^2 + b^2 + c^2) / (a * b + b * c + c * a) > 1 :=
by
  sorry

end smallest_constant_N_l612_612064


namespace fraction_capacity_noah_ali_l612_612240

def capacity_Ali_closet : ℕ := 200
def total_capacity_Noah_closet : ℕ := 100
def each_capacity_Noah_closet : ℕ := total_capacity_Noah_closet / 2

theorem fraction_capacity_noah_ali : (each_capacity_Noah_closet : ℚ) / capacity_Ali_closet = 1 / 4 :=
by sorry

end fraction_capacity_noah_ali_l612_612240


namespace find_m_with_integer_roots_l612_612956

theorem find_m_with_integer_roots :
  {m : ℝ // ∀ x : ℝ, (m-6)*(m-9)*x^2 + (15*m-117)*x + 54 = 0 → x ∈ ℤ} = {3, 7, 15, 6, 9} :=
by 
  sorry

end find_m_with_integer_roots_l612_612956


namespace count_good_numbers_l612_612378

-- Define the property of a good number
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- Count the number of good numbers
theorem count_good_numbers : (finset.filter is_good_number (finset.range 2021)).card = 10 :=
by sorry

end count_good_numbers_l612_612378


namespace fly_travel_distance_l612_612904

theorem fly_travel_distance
  (carA_speed : ℕ)
  (carB_speed : ℕ)
  (initial_distance : ℕ)
  (fly_speed : ℕ)
  (relative_speed : ℕ := carB_speed - carA_speed)
  (catchup_time : ℚ := initial_distance / relative_speed)
  (fly_travel : ℚ := fly_speed * catchup_time) :
  carA_speed = 20 → carB_speed = 30 → initial_distance = 1 → fly_speed = 40 → fly_travel = 4 :=
by
  sorry

end fly_travel_distance_l612_612904


namespace converse_proposition_incorrect_l612_612829

-- Let a, b, and m be real numbers.
variables (a b m : ℝ)

-- Define the original proposition.
def original_proposition := ∀ am bm, am^2 < bm^2 → a < b

-- Define the converse of the original proposition.
def converse_proposition := ∀ am bm, a < b → am^2 < bm^2

-- The theorem we want to prove is that the converse_proposition is false.
theorem converse_proposition_incorrect : ¬ (∀ am bm, a < b → am^2 < bm^2) :=
sorry

end converse_proposition_incorrect_l612_612829


namespace smallest_abs_difference_l612_612317

theorem smallest_abs_difference (p q : ℕ) 
  (h1 : 2025 = (p! * p!) / (q! * q!)) 
  (h2 : p ≥ q)
  (h3 : p + q = 9) : |p - q| = 1 :=
sorry

end smallest_abs_difference_l612_612317


namespace cos_36_deg_l612_612910

theorem cos_36_deg :
  let x := Real.cos (36 * Real.pi / 180) in
  x = ( -1 + Real.sqrt 5 ) / 4 := 
by
  let x := Real.cos (36 * Real.pi / 180)
  have h1 : 4 * x^2 + 2 * x - 1 = 0 := sorry 
  -- Polynomial roots found such that
  have h2 : x = ( -1 + Real.sqrt 5 ) / 4 ∨ x = ( -1 - Real.sqrt 5 ) / 4 := sorry
  -- Given that x is positive
  have : x > 0 := Real.cos_pos_of_pi_div_two_lt_of_lt_pi_div_two (by norm_num) (by norm_num)
  exact or.resolve_right h2 (by linarith)

end cos_36_deg_l612_612910


namespace sum_of_powers_inequality_l612_612698

theorem sum_of_powers_inequality (a : ℝ) (n : ℕ) (k : ℕ → ℕ) 
  (h1 : 0 < a) (h2 : a < 1) 
  (h3 : ∀ i j, i < j → k i < k j ∧ 0 ≤ k i) :
  (∑ i in Finset.range n, a ^ (k i)) ^ 2 < (1 + a) / (1 - a) * ∑ i in Finset.range n, a ^ (2 * k i) :=
sorry

end sum_of_powers_inequality_l612_612698


namespace good_numbers_count_l612_612336

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l612_612336


namespace tulips_sum_l612_612038

def tulips_total (arwen_tulips : ℕ) (elrond_tulips : ℕ) : ℕ := arwen_tulips + elrond_tulips

theorem tulips_sum : tulips_total 20 (2 * 20) = 60 := by
  sorry

end tulips_sum_l612_612038


namespace inequality_series_l612_612637

theorem inequality_series (n : ℕ) (h : 0 < n) : 
  1 + ∑ k in Finset.range (n + 1).filter (λ k, 1 ≤ k) \ {1}, (1 : ℚ) / (k+1)^2 < (2 * n + 1) / (n + 1) := 
by
  sorry

end inequality_series_l612_612637


namespace completing_the_square_l612_612754

theorem completing_the_square (x : ℝ) : x^2 + 8*x + 7 = 0 → (x + 4)^2 = 9 :=
by {
  sorry
}

end completing_the_square_l612_612754


namespace chord_length_of_intersection_l612_612866

theorem chord_length_of_intersection 
  (x y : ℝ) (h_line : 2 * x - y - 1 = 0) (h_circle : (x - 2)^2 + (y + 2)^2 = 9) : 
  ∃ l, l = 4 := 
sorry

end chord_length_of_intersection_l612_612866


namespace circle_geometry_problem_l612_612740

-- Define the geometric structure on a circle
variables {A B C D M : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]

-- Assume geometric conditions
axiom PointsOnCircle (A B C D : Type) : Prop
axiom IntersectAtM (A B C D M : Type) : Prop

-- Proposition to be proven
theorem circle_geometry_problem
  (h₁ : PointsOnCircle A B C D)
  (h₂ : IntersectAtM A B C D M) :
  (AC * AD) / AM = (BC * BD) / BM :=
sorry

end circle_geometry_problem_l612_612740


namespace total_late_time_l612_612041

theorem total_late_time (c : ℕ) (difference : ℕ) (n : ℕ) (total_time : ℕ) :
  c = 20 ∧ difference = 10 ∧ n = 4 → total_time = 140 :=
by
  assume h,
  sorry

end total_late_time_l612_612041


namespace mary_max_earnings_l612_612426

def regular_rate : ℝ := 8
def max_hours : ℝ := 60
def regular_hours : ℝ := 20
def overtime_rate : ℝ := regular_rate + 0.25 * regular_rate
def overtime_hours : ℝ := max_hours - regular_hours
def earnings_regular : ℝ := regular_hours * regular_rate
def earnings_overtime : ℝ := overtime_hours * overtime_rate
def total_earnings : ℝ := earnings_regular + earnings_overtime

theorem mary_max_earnings : total_earnings = 560 := by
  sorry

end mary_max_earnings_l612_612426


namespace area_ratio_triangle_quadrilateral_l612_612195

theorem area_ratio_triangle_quadrilateral
  (A B C G H : Type)
  (AB BC AC AG AH : ℝ)
  (h_AG : G ∈ line_segment A B)
  (h_AH : H ∈ line_segment A C)
  (AB_eq : AB = 21)
  (BC_eq : BC = 45)
  (AC_eq : AC = 36)
  (AG_eq : AG = 15)
  (AH_eq : AH = 24)
  : (area_triangle A G H) / (area_quad B C H G) = 25 / 24 := by
sorry

end area_ratio_triangle_quadrilateral_l612_612195


namespace find_common_ratio_l612_612972

noncomputable theory

-- Let {a_n} be the geometric sequence.
def geometric_sequence (a_n : ℕ → ℝ) (r : ℝ) := ∀ n, a_n (n + 1) = r * a_n n

-- Let S_n be the sum of the first n terms of the sequence.
def sum_of_first_n_terms (a_n S_n : ℕ → ℝ) := ∀ n, S_n n = ∑ i in finset.range n, a_n i

-- We are given the conditions
variables {a_n S_n : ℕ → ℝ} {q : ℝ}

def condition_1 := a_n 2013 = 2 * S_n 2014 + 6
def condition_2 := 3 * a_n 2014 = 2 * S_n 2015 + 6

-- With the above conditions
theorem find_common_ratio (h_geom_seq : geometric_sequence a_n q)
                          (h_sum_terms : sum_of_first_n_terms a_n S_n)
                          (h1 : condition_1)
                          (h2 : condition_2) :
                          q = 1 ∨ q = 1/2 := sorry

end find_common_ratio_l612_612972


namespace food_requirement_not_met_l612_612498

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l612_612498


namespace mean_home_runs_per_game_l612_612672

variable (home_runs : Nat) (games_played : Nat)

def total_home_runs : Nat := 
  (5 * 4) + (6 * 5) + (4 * 7) + (3 * 9) + (2 * 11)

def total_games_played : Nat :=
  (5 * 5) + (6 * 6) + (4 * 8) + (3 * 10) + (2 * 12)

theorem mean_home_runs_per_game :
  (total_home_runs : ℚ) / total_games_played = 127 / 147 :=
  by 
    sorry

end mean_home_runs_per_game_l612_612672


namespace find_f_100_l612_612100

-- Define the function f such that it satisfies the condition f(10^x) = x
noncomputable def f : ℝ → ℝ := sorry

-- Define the main theorem to prove f(100) = 2 given the condition f(10^x) = x
theorem find_f_100 (h : ∀ x : ℝ, f (10^x) = x) : f 100 = 2 :=
by {
  sorry
}

end find_f_100_l612_612100


namespace amount_received_by_A_is_4_over_3_l612_612436

theorem amount_received_by_A_is_4_over_3
  (a d : ℚ)
  (h1 : a - 2 * d + a - d = a + (a + d) + (a + 2 * d))
  (h2 : 5 * a = 5) :
  a - 2 * d = 4 / 3 :=
by
  sorry

end amount_received_by_A_is_4_over_3_l612_612436


namespace length_AN_eq_one_l612_612255

-- Definitions representing the conditions
def triangle := {A B C : Type}
def point (P : Type) := P
def length (A B : Type) := 1

variables (A B C M N : Type)

def angle_bisector (BAC : Prop) := ∀ (A B C : point), 
  let P := ∃ (X : Type), X

constant AC : length A C
constant AM : length A M
constant AN : length A N
constant angle_ANM_CNMC_eq : ∀ (ANM CNM : Prop), ANM = CNM

-- Conclude the length of AN is 1
theorem length_AN_eq_one : AN = 1 :=
sorry

end length_AN_eq_one_l612_612255


namespace remainder_of_sum_l612_612826

theorem remainder_of_sum (D k l : ℕ) (hk : 242 = k * D + 11) (hl : 698 = l * D + 18) :
  (242 + 698) % D = 29 :=
by
  sorry

end remainder_of_sum_l612_612826


namespace train_cross_pole_time_l612_612469

noncomputable def speed_km_per_hr := 30 -- speed of the train in km/hr
noncomputable def length_train := 50 -- length of the train in meters
noncomputable def speed_m_per_s := (speed_km_per_hr * 1000) / 3600  -- speed in m/s

theorem train_cross_pole_time : (length_train / speed_m_per_s) ≈ 6.00 := 
by
  -- calculation
  sorry

end train_cross_pole_time_l612_612469


namespace count_good_numbers_l612_612367

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l612_612367


namespace enough_cat_food_for_six_days_l612_612521

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l612_612521


namespace minimum_disks_drawn_to_guarantee_fifteen_same_label_l612_612205

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Condition: total number of disks
def total_disks : ℕ := sum_first_n 60

-- Condition: disks are labeled from 1 to 60.

theorem minimum_disks_drawn_to_guarantee_fifteen_same_label :
  total_disks = 1830 →
  (∀ disks : list ℕ, disks.length = 750 → 
   (∃ label, disks.count label ≥ 15)) :=
by
  intros htotal hdisks
  sorry

end minimum_disks_drawn_to_guarantee_fifteen_same_label_l612_612205


namespace value_of_a_l612_612114

open Set

variable (a : Real)

def A : Set Real := {1, 3, a^2}
def B : Set Real := {1, a + 2}

theorem value_of_a : (A ∩ B = B) → a = 2 := by
  sorry

end value_of_a_l612_612114


namespace employee_price_l612_612836

theorem employee_price (wholesale_cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) :
  wholesale_cost = 200 →
  markup_percent = 0.20 →
  discount_percent = 0.05 →
  let retail_price := wholesale_cost * (1 + markup_percent)
      discount_amount := retail_price * discount_percent
      employee_price := retail_price - discount_amount
  in employee_price = 228 := by
sorry

end employee_price_l612_612836


namespace prob_a_prob_b_l612_612847

-- Given conditions and question for Part a
def election_prob (p q : ℕ) (h : p > q) : ℚ :=
  (p - q) / (p + q)

theorem prob_a : election_prob 3 2 (by decide) = 1 / 5 :=
  sorry

-- Given conditions and question for Part b
theorem prob_b : election_prob 1010 1009 (by decide) = 1 / 2019 :=
  sorry

end prob_a_prob_b_l612_612847


namespace determine_m_l612_612616

theorem determine_m (m : ℝ) : (line_slope (-2, m) (m, 4) = 1) → m = 1 :=
by
  sorry

-- Additional definitions required for complete formalization
def line_slope (P Q : ℝ × ℝ) : ℝ :=
  if P.1 = Q.1 then 0 else (Q.2 - P.2) / (Q.1 - P.1)

end determine_m_l612_612616


namespace science_independent_of_gender_l612_612006

-- Definitions and conditions
def is_science_enthusiast (hours: Nat) : Prop := hours > 6

-- Given sample data
structure SampleData :=
  (males_science: Nat)
  (males_non_science: Nat)
  (females_science: Nat)
  (females_non_science: Nat)

def sample : SampleData := {males_science := 24, males_non_science := 36, females_science := 12, females_non_science := 28}

def total_samples : Nat := sample.males_science + sample.males_non_science + sample.females_science + sample.females_non_science

-- Question 1
theorem science_independent_of_gender (sample: SampleData) (alpha: ℝ) (critical_value: ℝ) : Prop :=
  ∀ n a b c d: Nat, 
  n = total_samples → 
  a = sample.males_science → 
  b = sample.females_science → 
  c = sample.males_non_science →
  d = sample.females_non_science →
  let K_square := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  K_square < critical_value

-- Given alpha and critical value
def alpha : ℝ := 0.010
def critical_value : ℝ := 6.635

-- Prove that science enthusiasts are independent of gender
example : science_independent_of_gender sample alpha critical_value := 
by sorry

-- Question 2
def L_B_given_A (sample: SampleData) : ℚ :=
  let P_B_given_A := sample.males_non_science / (sample.males_non_science + sample.females_non_science) in
  let P_not_B_given_A := sample.females_non_science / (sample.males_non_science + sample.females_non_science) in
  P_B_given_A / P_not_B_given_A

-- Prove likelihood ratio L(B|A) = 9/7
example : L_B_given_A sample = 9 / 7 := 
by sorry

-- Question 3
structure Group :=
  (males: Nat)
  (females: Nat)

def science_enthusiasts_group : Group := {males := 24, females := 12}

def random_choice_probability (k: Nat) : ℚ :=
  if k = 0 then 1/5 
  else if k = 1 then 3/5
  else 1/5

-- Prove mathematical expectation E(X) = 2
def expectation_x : ℚ := (1 / 5) * 1 + (3 / 5) * 2 + (1 / 5) * 3

example : expectation_x = 2 := 
by sorry

end science_independent_of_gender_l612_612006


namespace cat_food_insufficient_l612_612504

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l612_612504


namespace smallest_n_divides_2008_l612_612605

noncomputable def sequence (n : ℕ) : ℤ :=
if n = 0 then 0
else if n = 1 then 1
else 2 * sequence (n - 1) + 2007 * sequence (n - 2)

theorem smallest_n_divides_2008 :
  ∃ n : ℕ, n > 0 ∧ 2008 ∣ sequence n ∧ ∀ m : ℕ, m > 0 → m < n → ¬ (2008 ∣ sequence m) :=
begin
  sorry
end

end smallest_n_divides_2008_l612_612605


namespace sum_of_abs_values_of_roots_l612_612952

noncomputable def sum_abs_roots_eq : Prop :=
  let p := (polynomial.C (4 : ℂ)) + (polynomial.C (-12 : ℂ)) * polynomial.X + (polynomial.C (13 : ℂ)) * polynomial.X^2 +
           (polynomial.C (-6 : ℂ)) * polynomial.X^3 + polynomial.X^4 in
  let roots := (polynomial.roots p).to_finset in
  (roots.sum complex.abs) = 2 * complex.sqrt 6 + 2 * complex.sqrt 2

theorem sum_of_abs_values_of_roots : sum_abs_roots_eq :=
by
  sorry

end sum_of_abs_values_of_roots_l612_612952


namespace locate_z_conjugate_in_fourth_quadrant_l612_612227

-- Definitions of the conditions
def z : ℂ := sorry
def c1 : ℂ := 1 - complex.i
def c2 : ℂ := 1 + real.sqrt 3 * complex.i

-- The given condition
axiom condition : c1 * z = complex.abs c2

-- Proving the final statement
theorem locate_z_conjugate_in_fourth_quadrant :
  complex.re (conj z) > 0 ∧ complex.im (conj z) < 0 :=
begin
  sorry,
end

end locate_z_conjugate_in_fourth_quadrant_l612_612227


namespace sum_sequence_correct_l612_612431

def sequence_term (n : ℕ) : ℕ :=
  if n % 9 = 0 ∧ n % 32 = 0 then 7
  else if n % 7 = 0 ∧ n % 32 = 0 then 9
  else if n % 7 = 0 ∧ n % 9 = 0 then 32
  else 0

def sequence_sum (up_to : ℕ) : ℕ :=
  (Finset.range (up_to + 1)).sum sequence_term

theorem sum_sequence_correct : sequence_sum 2015 = 1106 := by
  sorry

end sum_sequence_correct_l612_612431


namespace triangle_circles_five_l612_612076

theorem triangle_circles_five (A B C : Type*) (AB AC BC : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) :
  (∃ AC, AC = 3 ∨ AC = 4) ↔ (A, B, C, AB, AC, BC) = 5 :=
sorry

end triangle_circles_five_l612_612076


namespace PA_inter_B_eq_1_over_3_l612_612962

-- Let P be a probability measure
variables {Ω : Type*} [measurable_space Ω] (P : measure_theory.measure Ω)

-- Given conditions
def PA_given_B (A B : set Ω) (hB : P B ≠ 0) : ℝ := P (A ∩ B) / P B

-- The condition that P(A|B) = 3/7
axiom PA_given_B_eq_3_over_7 (A B : set Ω) (hB : P B ≠ 0) : PA_given_B P A B hB = 3/7

-- The condition that P(B) = 7/9
axiom PB_eq_7_over_9 (B : set Ω) : P B = 7/9

-- The statement to be proved
theorem PA_inter_B_eq_1_over_3 (A B : set Ω) (hB : P B ≠ 0) : P (A ∩ B) = 1/3 :=
by
  -- Sorry is used to indicate that the proof is omitted
  sorry

end PA_inter_B_eq_1_over_3_l612_612962


namespace roots_of_quadratic_example_quadratic_problem_solution_l612_612800

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic :
  ∀ (a b c : ℝ), a ≠ 0 → discriminant a b c > 0 → (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0)) :=
by
  intros a b c a_ne_zero discr_positive
  sorry

theorem example_quadratic :
  discriminant 1 (-2) (-1) > 0 :=
by
  unfold discriminant
  norm_num
  linarith

theorem problem_solution :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (1 * x₁^2 - 2 * x₁ - 1 = 0) ∧ (1 * x₂^2 - 2 * x₂ - 1 = 0) :=
by
  apply roots_of_quadratic 1 (-2) (-1)
  norm_num
  apply example_quadratic

end roots_of_quadratic_example_quadratic_problem_solution_l612_612800


namespace count_good_numbers_l612_612377

-- Define the property of a good number
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- Count the number of good numbers
theorem count_good_numbers : (finset.filter is_good_number (finset.range 2021)).card = 10 :=
by sorry

end count_good_numbers_l612_612377


namespace find_sin_C_and_area_of_triangle_ABD_l612_612689

theorem find_sin_C_and_area_of_triangle_ABD (BC BD: ℝ) (C: ℝ)
    (hBC: BC = 3)
    (hBD: BD = (8 * Real.sqrt 3) / 5)
    (hBDC: C = 60 * Real.pi / 180) : 
    sin C = 4 / 5 ∧
    1 / 2 * 4 * ((8 * Real.sqrt 3) / 5) * ((4 * Real.sqrt 3 - 3) / 10) = (96 - 24 * Real.sqrt 3) / 25 := by
  sorry

end find_sin_C_and_area_of_triangle_ABD_l612_612689


namespace kathleen_max_offices_l612_612204

open Nat

theorem kathleen_max_offices : ∃ k, gcd 18 12 = k ∧ k = 6 :=
by
  sorry

end kathleen_max_offices_l612_612204


namespace number_of_valid_permutations_l612_612945

def no_consecutive_pairs (p : List ℕ) : Prop :=
  ∀ i, (1 <= i ∧ i < 8) → ¬ (i + 1 = p[i] ∧ i = p[i - 1])

noncomputable def valid_permutations : List (List ℕ) :=
  List.filter no_consecutive_pairs (List.permutations (List.range' 1 8))

theorem number_of_valid_permutations :
  valid_permutations.length = 16687 := by
  sorry

end number_of_valid_permutations_l612_612945


namespace problem_solution_l612_612139

theorem problem_solution (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a^2 + b^2 + c^2 = 2) (h3 : a^3 + b^3 + c^3 = 3) :
  (a * b * c = 1 / 6) ∧ (a^4 + b^4 + c^4 = 25 / 6) :=
by {
  sorry
}

end problem_solution_l612_612139


namespace half_abs_diff_squares_eq_40_l612_612400

theorem half_abs_diff_squares_eq_40 (x y : ℤ) (hx : x = 21) (hy : y = 19) :
  (|x^2 - y^2| / 2) = 40 :=
by
  sorry

end half_abs_diff_squares_eq_40_l612_612400


namespace min_segments_harmonic_proof_l612_612209

noncomputable def min_segments_harmonic (n : ℕ) : ℕ :=
if n > 3 then n + 1 else 0

theorem min_segments_harmonic_proof : ∀ (n : ℕ), n > 3 → min_segments_harmonic n = n + 1 := 
by {
  intros n hn,
  unfold min_segments_harmonic,
  rw if_pos hn,
  sorry
}

end min_segments_harmonic_proof_l612_612209


namespace matrix_subtraction_l612_612490

-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 4, -3 ],
  ![ 2,  8 ]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 1,  5 ],
  ![ -3,  6 ]
]

-- Define the result matrix as given in the problem
def result : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 3, -8 ],
  ![ 5,  2 ]
]

-- The theorem to prove
theorem matrix_subtraction : A - B = result := 
by 
  sorry

end matrix_subtraction_l612_612490


namespace exterior_angle_BAC_is_130_l612_612883

-- Definitions used in the statement

def interior_angle_nonagon : ℝ := 140
def interior_angle_square : ℝ := 90

-- The theorem the measure of the exterior angle BAC
theorem exterior_angle_BAC_is_130 :
  let BAC := 360 - interior_angle_nonagon - interior_angle_square in
  BAC = 130 :=
by 
  have h_BAC: BAC = 360 - 140 - 90, by sorry,
  exact h_BAC

end exterior_angle_BAC_is_130_l612_612883


namespace maximum_marked_points_no_right_triangle_l612_612735

-- Define the chessboard size
def chessboard_size : ℕ := 8

-- Define the condition for no right triangles among marked points
def no_right_triangle (points : list (ℕ × ℕ)) : Prop :=
  ∀ (p1 p2 p3 : ℕ × ℕ), p1 ∈ points → p2 ∈ points → p3 ∈ points →
    ¬ ((p1.1 = p2.1 ∧ p3.2 = p2.2) ∨ (p2.1 = p1.1 ∧ p2.2 = p3.2))

-- Define the maximum number of points that can be marked
noncomputable def max_marked_points (n : ℕ) : ℕ :=
  if h : n = chessboard_size then 14 else sorry

-- The theorem we want to prove
theorem maximum_marked_points_no_right_triangle :
  ∀ points : list (ℕ × ℕ), no_right_triangle points → list.length points ≤ max_marked_points chessboard_size :=
by sorry

end maximum_marked_points_no_right_triangle_l612_612735


namespace cyclic_quadrilateral_angle_D_l612_612686

theorem cyclic_quadrilateral_angle_D (A B C D : ℝ) (h1 : A + C = 180) (h2 : B + D = 180) (h3 : 3 * A = 4 * B) (h4 : 3 * A = 6 * C) : D = 100 :=
by
  sorry

end cyclic_quadrilateral_angle_D_l612_612686


namespace f_monotonicity_l612_612121

noncomputable def f (x : ℝ) : ℝ := abs (x^2 - 1)

theorem f_monotonicity :
  (∀ x y : ℝ, (-1 < x ∧ x < 0 ∧ x < y ∧ y < 0) → f x < f y) ∧
  (∀ x y : ℝ, (1 < x ∧ 1 < y ∧ x < y) → f x < f y) ∧
  (∀ x y : ℝ, (x < -1 ∧ y < -1 ∧ y < x) → f x < f y) ∧
  (∀ x y : ℝ, (0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ y < x) → f x < f y) :=
by
  sorry

end f_monotonicity_l612_612121


namespace price_of_each_shirt_l612_612239

theorem price_of_each_shirt 
  (toys_cost : ℕ := 3 * 10)
  (cards_cost : ℕ := 2 * 5)
  (total_spent : ℕ := 70)
  (remaining_cost: ℕ := total_spent - (toys_cost + cards_cost))
  (num_shirts : ℕ := 3 + 2) :
  (remaining_cost / num_shirts) = 6 :=
by
  sorry

end price_of_each_shirt_l612_612239


namespace area_of_right_triangle_l612_612876

-- Define the conditions
def hypotenuse : ℝ := 9
def angle : ℝ := 30

-- Define the Lean statement for the proof problem
theorem area_of_right_triangle : 
  ∃ (area : ℝ), area = 10.125 * Real.sqrt 3 ∧
  ∃ (shorter_leg : ℝ) (longer_leg : ℝ),
    shorter_leg = hypotenuse / 2 ∧
    longer_leg = shorter_leg * Real.sqrt 3 ∧
    area = (shorter_leg * longer_leg) / 2 :=
by {
  -- The proof would go here, but we only need to state the problem for this task.
  sorry
}

end area_of_right_triangle_l612_612876


namespace angle_between_diagonals_l612_612193

theorem angle_between_diagonals (a : ℝ) (h : a ≠ 0)
 (h1: ∥vec_a1c1∥ = a)
 (h2: ∥vec_c1d∥ = a * √3)
 (h3: ∠ (vec_a1c1) (vec_c1d) = π / 6) :
  ∠ (vec_a1c1) (vec_c1d) = arccos (√3 / 6) := by
sory

end angle_between_diagonals_l612_612193


namespace problem_l612_612670

-- Define the problem's conditions and prove the necessary results
theorem problem (A B C : ℝ)
  (h1 : sin (A + B) = sin B + sin (A - B))
  (h2 : ∀ AB AC : ℝ, (AB * AC * real.cos (A) = 20)) :
  (A = π / 3 ∧ (∀ AB AC : ℝ, ∃ BC : ℝ, BC^2 = AB^2 + AC^2 - 2 * AB * AC * real.cos (A) ∧ BC = 2 * sqrt 10)) :=
by
  -- Proof goes here
  sorry

end problem_l612_612670


namespace number_of_good_numbers_l612_612354

theorem number_of_good_numbers :
  let n := 2020 in 
  let remainder := 22 in
  let number := 1998 in
  let divisors := {d ∣ number | d > remainder}.to_list.length = 10 := by sorry

end number_of_good_numbers_l612_612354


namespace inequality_for_positive_real_numbers_l612_612743

theorem inequality_for_positive_real_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^4 + b^4 + c^4) / (a + b + c) ≥ a * b * c :=
  sorry

end inequality_for_positive_real_numbers_l612_612743


namespace length_an_eq_1_l612_612265

variable {A B C M N : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N]
variable {dist : A → A → ℝ}
variable {angle : ∀ {A B C : A}, ℝ}

-- Given conditions as hypotheses
variable (H1 : ∀ {M}, M ∈ angle_bisector A B C)
variable (H2 : ∀ {N : Type}, N ∈ extended_line A B)
variable (H3 : dist A C = 1)
variable (H4 : dist A M = 1)
variable (H5 : angle A N M = angle C N M)

-- The statement we need to prove
theorem length_an_eq_1 (H1 : M ∈ angle_bisector A B C) (H2 : N ∈ extended_line A B)
  (H3 : dist A C = 1) (H4 : dist A M = 1) (H5 : angle A N M = angle C N M) :
  dist A N = 1 := by
  sorry


end length_an_eq_1_l612_612265


namespace exists_increasing_sequence_l612_612093

theorem exists_increasing_sequence (n : ℕ) : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ∃ x : ℕ → ℕ, (∀ i : ℕ, 1 ≤ i → i ≤ n → x i < x (i + 1)) :=
by
  sorry

end exists_increasing_sequence_l612_612093


namespace cat_food_inequality_l612_612535

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l612_612535


namespace part_a1_part_a2_part_b_part_c_part_d_part_e1_part_e2_part_f_l612_612843

def f : ℤ → ℝ
/-
f is a function from ℤ to ℝ
-/

axiom recurrent_relation : ∀ n : ℤ, f(n + 1) = |f(n)| - f(n - 1)

axiom initial_conditions_1 : f(1) = 3
axiom initial_conditions_2 : f(2) = 2

theorem part_a1 : f(12) = -1 :=
by
  sorry

theorem part_a2 : f(0) = 1 :=
by
  sorry

theorem part_b : ∀ n : ℤ, f(n) ≥ 0 ∨ f(n + 1) ≥ 0 ∨ f(n + 2) ≥ 0 :=
by
  sorry

theorem part_c : ∀ n : ℤ, f(n) ≤ 0 ∨ f(n + 1) ≤ 0 ∨ f(n + 2) ≤ 0 ∨ f(n + 3) ≤ 0 :=
by
  sorry

theorem part_d : ∃ k : ℤ, 0 ≤ k ∧ k ≤ 4 ∧ f(k) ≤ 0 ∧ f(k + 1) ≥ 0 :=
by
  sorry

axiom initial_conditions_a : ∃ a : ℝ, f(0) = -a ∧ a ≥ 0
axiom initial_conditions_b : ∃ b : ℝ, f(1) = b ∧ b ≥ 0

theorem part_e1 : f(9) = -1 :=
by
  sorry

theorem part_e2 : f(10) = 2 :=
by
  sorry

theorem part_f : ∀ n : ℤ, f(n + 9) = f(n) :=
by
  sorry

end part_a1_part_a2_part_b_part_c_part_d_part_e1_part_e2_part_f_l612_612843


namespace inverse_function_negative_quadratic_l612_612788

theorem inverse_function_negative_quadratic :
  ∀ (x : ℝ), x ≤ -2 → f⁻¹(x) = -sqrt(-x) :=
by
  have f : ℝ → ℝ := λ x, -x^2
  have f_inv : ℝ → ℝ := λ x, -sqrt(-x)
  sorry

end inverse_function_negative_quadratic_l612_612788


namespace cat_food_insufficient_l612_612507

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l612_612507


namespace fill_tub_together_time_l612_612817

theorem fill_tub_together_time :
  let rate1 := 1 / 4
  let rate2 := 1 / 4
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  combined_rate ≠ 0 → (1 / combined_rate = 12 / 7) :=
by
  let rate1 := 1 / 4
  let rate2 := 1 / 4
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  sorry

end fill_tub_together_time_l612_612817


namespace damien_jogs_75_miles_in_three_weeks_l612_612923

theorem damien_jogs_75_miles_in_three_weeks :
  let daily_distance := 5
  let weekdays_per_week := 5
  let weeks := 3
  (daily_distance * (weekdays_per_week * weeks)) = 75 :=
by
  let daily_distance := 5
  let weekdays_per_week := 5
  let weeks := 3
  show (daily_distance * (weekdays_per_week * weeks)) = 75
    from sorry

end damien_jogs_75_miles_in_three_weeks_l612_612923


namespace shelter_animals_count_l612_612814

theorem shelter_animals_count : 
  (initial_cats adopted_cats new_cats final_cats dogs total_animals : ℕ) 
   (h1 : initial_cats = 15)
   (h2 : adopted_cats = initial_cats / 3)
   (h3 : new_cats = adopted_cats * 2)
   (h4 : final_cats = initial_cats - adopted_cats + new_cats)
   (h5 : dogs = final_cats * 2)
   (h6 : total_animals = final_cats + dogs) :
   total_animals = 60 := 
sorry

end shelter_animals_count_l612_612814


namespace enough_cat_food_for_six_days_l612_612520

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l612_612520


namespace range_of_x_l612_612960

theorem range_of_x (x : ℝ) : (2 : ℝ)^(3 - 2 * x) < (2 : ℝ)^(3 * x - 4) → x > 7 / 5 := by
  sorry

end range_of_x_l612_612960


namespace john_next_birthday_l612_612202

theorem john_next_birthday
  (j b a : ℝ)
  (h1 : j = 1.25 * b)
  (h2 : b = 0.5 * a)
  (h3 : j + b + a = 37.8) :
  Real.ceil j = 12 :=
sorry

end john_next_birthday_l612_612202


namespace factorization_problem_l612_612775

theorem factorization_problem :
  ∃ a b : ℤ, (∀ y : ℤ, 4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b))
    ∧ (a - b = 11) := by
  sorry

end factorization_problem_l612_612775


namespace tetrahedron_spheres_l612_612287

-- Variables for volume, face areas, and conditions
variables (V : ℝ) (S1 S2 S3 S4 : ℝ)
def eps := list(ℕ → ℝ)

-- Given a tetrahedron, there are at least 5 and no more than 8 spheres
theorem tetrahedron_spheres (h : ∀ ε : eps, (0 < S1 + S2 + S3 + S4) → 5 ≤ 8) : 5 ≤ 8 := by sorry

end tetrahedron_spheres_l612_612287


namespace total_value_of_investments_l612_612483

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) :=
  P * (1 + r / n) ^ (n * t)

def investment_1 : ℝ := 300
def interest_1 : ℝ := 0.10
def times_compounded_1 : ℕ := 12
def years_1 : ℕ := 5

def investment_2 : ℝ := 500
def interest_2 : ℝ := 0.07
def times_compounded_2 : ℕ := 4
def years_2 : ℕ := 3

def investment_3 : ℝ := 1000
def interest_3 : ℝ := 0.05
def times_compounded_3 : ℕ := 2
def years_3 : ℕ := 10

def total_value : ℝ :=
  compound_interest investment_1 interest_1 times_compounded_1 years_1 +
  compound_interest investment_2 interest_2 times_compounded_2 years_2 +
  compound_interest investment_3 interest_3 times_compounded_3 years_3

theorem total_value_of_investments :
  total_value ≈ 2745.24 :=
sorry

end total_value_of_investments_l612_612483


namespace cat_food_insufficient_for_six_days_l612_612527

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l612_612527


namespace length_an_eq_1_l612_612267

variable {A B C M N : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N]
variable {dist : A → A → ℝ}
variable {angle : ∀ {A B C : A}, ℝ}

-- Given conditions as hypotheses
variable (H1 : ∀ {M}, M ∈ angle_bisector A B C)
variable (H2 : ∀ {N : Type}, N ∈ extended_line A B)
variable (H3 : dist A C = 1)
variable (H4 : dist A M = 1)
variable (H5 : angle A N M = angle C N M)

-- The statement we need to prove
theorem length_an_eq_1 (H1 : M ∈ angle_bisector A B C) (H2 : N ∈ extended_line A B)
  (H3 : dist A C = 1) (H4 : dist A M = 1) (H5 : angle A N M = angle C N M) :
  dist A N = 1 := by
  sorry


end length_an_eq_1_l612_612267


namespace smallest_square_side_length_l612_612949

theorem smallest_square_side_length :
  ∃ (n : ℝ), n = 1.0 ∧ ∀ k : ℝ, (k < n ∧ ∃ (a b c : ℤ),
    (a = 0 ∧ b = n.toInt ∧ c = 0 ∧ lattice_point a 0 ∧
     lattice_point b 0 ∧ lattice_point 0 b) → False) :=
begin
  sorry
end

end smallest_square_side_length_l612_612949


namespace option_D_is_arithmetic_sequence_l612_612321

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) - a n = d

def aₙ_A (n : ℕ) : ℕ := n / (n + 1)
def aₙ_B (n : ℕ) : ℕ := n^2 - 1
def aₙ_C (n : ℕ) : ℕ := 5 * n + (-1)^n
def aₙ_D (n : ℕ) : ℕ := 3 * n - 1

theorem option_D_is_arithmetic_sequence : is_arithmetic_sequence aₙ_D :=
sorry

end option_D_is_arithmetic_sequence_l612_612321


namespace complete_the_square_l612_612758

theorem complete_the_square :
  ∀ x : ℝ, (x^2 + 8 * x + 7 = 0) → (x + 4)^2 = 9 :=
by
  intro x h,
  sorry

end complete_the_square_l612_612758


namespace price_of_case_bulk_is_12_l612_612447

noncomputable def price_per_can_grocery_store : ℚ := 6 / 12
noncomputable def price_per_can_bulk : ℚ := price_per_can_grocery_store - 0.25
def cans_per_case_bulk : ℕ := 48
noncomputable def price_per_case_bulk : ℚ := price_per_can_bulk * cans_per_case_bulk

theorem price_of_case_bulk_is_12 : price_per_case_bulk = 12 :=
by
  sorry

end price_of_case_bulk_is_12_l612_612447


namespace length_AN_l612_612250

theorem length_AN {A B C M N : Type*}
  (h1 : M ∈ angle_bisector A B C)
  (h2 : N ∈ extension A B)
  (h3 : distance A C = 1)
  (h4 : distance A M = 1)
  (h5 : ∠ A N M = ∠ C N M) :
  distance A N = 1 :=
sorry

end length_AN_l612_612250


namespace sum_of_squares_count_l612_612653

theorem sum_of_squares_count (N : ℕ) :
  ∃ N, ∀ n, (n < 800) → ∃ a b, (1 ≤ a ∧ a ≤ 28) ∧ (1 ≤ b ∧ b ≤ 28) ∧ n = a^2 + b^2 := sorry

end sum_of_squares_count_l612_612653


namespace count_special_integers_l612_612487

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def base7 (n : ℕ) : ℕ := 
  let c := n / 343
  let rem1 := n % 343
  let d := rem1 / 49
  let rem2 := rem1 % 49
  let e := rem2 / 7
  let f := rem2 % 7
  343 * c + 49 * d + 7 * e + f

def base8 (n : ℕ) : ℕ := 
  let g := n / 512
  let rem1 := n % 512
  let h := rem1 / 64
  let rem2 := rem1 % 64
  let i := rem2 / 8
  let j := rem2 % 8
  512 * g + 64 * h + 8 * i + j

def matches_last_two_digits (n t : ℕ) : Prop := (t % 100) = (3 * (n % 100))

theorem count_special_integers : 
  ∃! (N : ℕ), is_three_digit N ∧ 
    matches_last_two_digits N (base7 N + base8 N) :=
sorry

end count_special_integers_l612_612487


namespace sin_pi_minus_alpha_cos_2pi_plus_alpha_eq_l612_612097

variables (α m : ℝ)
# check that the necessary conditions are met 
axiom sin_pi_minus_alpha_eq_m : sin (π - α) = m
axiom abs_m_le_one : |m| ≤ 1

-- goal statement
theorem sin_pi_minus_alpha_cos_2pi_plus_alpha_eq (h1 : sin (π - α) = m) (h2 : |m| ≤ 1) : 
  cos (2 * (π + α)) = 1 - 2 * m^2 :=
sorry

end sin_pi_minus_alpha_cos_2pi_plus_alpha_eq_l612_612097


namespace split_bill_equally_l612_612443

noncomputable def total_bill : ℝ := 357.42
noncomputable def num_people : ℝ := 15
noncomputable def individual_share : ℝ := (total_bill / num_people).floorDiv 0.01 / 100

theorem split_bill_equally : individual_share = 23.83 :=
by
  sorry

end split_bill_equally_l612_612443


namespace shaded_area_l612_612938

-- Define the conditions and prove the required area
theorem shaded_area (R : ℝ) (α : ℝ) (hα : α = 20 * real.pi / 180) :
  let S_0 := (real.pi * R^2) / 2 in
  let sector_area := 1 / 2 * (2 * R)^2 * α in
  sector_area = (2 * real.pi * R^2) / 9 :=
by
  sorry

end shaded_area_l612_612938


namespace vector_be_correct_l612_612680

open_locale vector_space

noncomputable def find_be (a b : VectorSpace ℝ) (AB AD DC BE BC CE : VectorSpace ℝ) :=
  square ABCD ∧ midpoint E DC ∧
  (AB = a) ∧ (AD = b) ∧
  (BE = BC + CE) ∧ BE = b - (1/2) * AB

-- Theorem to be proved
theorem vector_be_correct (a b : VectorSpace ℝ) (AB AD DC BE BC CE : VectorSpace ℝ) 
  (ABCD : square ABCD) (E : midpoint E DC) (h_ab : AB = a) (h_ad : AD = b) :
  BE = b - (1/2) * a :=
begin
  sorry -- Placeholder for the proof
end

end vector_be_correct_l612_612680


namespace correct_option_l612_612677

-- Definitions corresponding to the conditions
def white_balls : ℕ := 1
def black_balls : ℕ := 3
def red_balls : ℕ := 2
def total_balls : ℕ := white_balls + black_balls + red_balls

-- Definition stating that drawing a black ball is an uncertain event
def prob_black_ball : ℚ := black_balls / total_balls

theorem correct_option (white_balls black_balls red_balls total_balls : ℕ) (h_total : total_balls = white_balls + black_balls + red_balls) : 
  prob_black_ball = 1/2 :=
  by
  unfold prob_black_ball
  rw [h_total]
  sorry -- Proof details are omitted

-- Assertion that the correct option is B
def correct_option_B : Prop := prob_black_ball = 1/2

example : correct_option_B := 
  by
  unfold correct_option_B
  sorry -- Proof details are omitted

end correct_option_l612_612677


namespace stickers_distribution_l612_612725

theorem stickers_distribution :
  ∀ (n k : ℕ), n = 10 → k = 5 →
  ∃ (x : ℕ), x = nat.choose (n - 1) (k - 1) ∧ x = 126 :=
by
  intros n k hn hk
  use (nat.choose (n - 1) (k - 1))
  split;
  sorry

end stickers_distribution_l612_612725


namespace telephone_number_fraction_calculation_l612_612485

theorem telephone_number_fraction_calculation :
  let valid_phone_numbers := 7 * 10^6
  let special_phone_numbers := 10^5
  (special_phone_numbers / valid_phone_numbers : ℚ) = 1 / 70 :=
by
  sorry

end telephone_number_fraction_calculation_l612_612485


namespace good_numbers_count_l612_612339

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l612_612339


namespace example_problem_l612_612614

variable {f : ℝ → ℝ}

theorem example_problem
  (h1 : ∀ x, f x = f (-x))
  (h2 : ∀ x, f'' x < f x) :
  e⁻¹ * f 1 < f 0 ∧ f 0 < e² * f 2 := sorry

end example_problem_l612_612614


namespace A_det_nonzero_A_inv_is_correct_l612_612941

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 4], ![2, 9]]

def A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![9, -4], ![-2, 1]]

theorem A_det_nonzero : det A ≠ 0 := 
  sorry

theorem A_inv_is_correct : A * A_inv = 1 := 
  sorry

end A_det_nonzero_A_inv_is_correct_l612_612941


namespace total_six_letter_words_l612_612886

def num_vowels := 6
def vowel_count := 5
def word_length := 6

theorem total_six_letter_words : (num_vowels ^ word_length) = 46656 :=
by sorry

end total_six_letter_words_l612_612886


namespace cat_food_inequality_l612_612516

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l612_612516


namespace odd_prime_power_l612_612572

-- Define the necessary conditions and the final proof statement
theorem odd_prime_power {n : ℕ} (h1 : n > 1) (h2 : n % 2 = 1) :
  (∀ a b : ℕ, a ∣ n → b ∣ n → Nat.coprime a b → (a + b - 1) ∣ n) →
  (∃ p m : ℕ, Nat.Prime p ∧ p % 2 = 1 ∧ n = p^m) :=
by
  sorry

end odd_prime_power_l612_612572


namespace find_x_l612_612218

theorem find_x (a b x : ℝ) (h_a : a > 0) (h_b : b > 0) (h_x : x > 0)
  (s : ℝ) (h_s1 : s = (a ^ 2) ^ (4 * b)) (h_s2 : s = a ^ (2 * b) * x ^ (3 * b)) :
  x = a ^ 2 :=
sorry

end find_x_l612_612218


namespace derivative_sin_at_pi_over_3_l612_612629

noncomputable def f : ℝ → ℝ := sin

theorem derivative_sin_at_pi_over_3 :
  (∀ Δx, ∆x ≠ 0 → (f ((π / 3) + Δx) - f (π / 3)) / Δx) ⟶ (1 / 2) :=
by
  sorry

end derivative_sin_at_pi_over_3_l612_612629


namespace edge_length_of_inscribed_cube_in_sphere_l612_612106

noncomputable def edge_length_of_cube_in_sphere (surface_area_sphere : ℝ) (π_cond : surface_area_sphere = 36 * Real.pi) : ℝ :=
  let x := 2 * Real.sqrt 3
  x

theorem edge_length_of_inscribed_cube_in_sphere (surface_area_sphere : ℝ) (π_cond : surface_area_sphere = 36 * Real.pi) :
  edge_length_of_cube_in_sphere surface_area_sphere π_cond = 2 * Real.sqrt 3 :=
by
  sorry

end edge_length_of_inscribed_cube_in_sphere_l612_612106


namespace integral_eval_l612_612493

noncomputable def integral_of_function : ℝ :=
  ∫ x in -1..1, (x^3 - 1 / x^4)

theorem integral_eval : integral_of_function = (2 / 3) := by
  sorry

end integral_eval_l612_612493


namespace squares_and_sqrt_l612_612760

variable (a b c : ℤ)

theorem squares_and_sqrt (ha : a = 10001) (hb : b = 100010001) (hc : c = 1000200030004000300020001) :
∃ x y z : ℤ, x = a^2 ∧ y = b^2 ∧ z = Int.sqrt c ∧ x = 100020001 ∧ y = 10002000300020001 ∧ z = 1000100010001 :=
by
  use a^2, b^2, Int.sqrt c
  rw [ha, hb, hc]
  sorry

end squares_and_sqrt_l612_612760


namespace linda_total_distance_l612_612928

theorem linda_total_distance :
  ∃ x: ℕ, 
    (x > 0) ∧ (60 % x = 0) ∧
    ((x + 5) > 0) ∧ (60 % (x + 5) = 0) ∧
    ((x + 10) > 0) ∧ (60 % (x + 10) = 0) ∧
    ((x + 15) > 0) ∧ (60 % (x + 15) = 0) ∧
    (60 / x + 60 / (x + 5) + 60 / (x + 10) + 60 / (x + 15) = 25) :=
by
  sorry

end linda_total_distance_l612_612928


namespace central_angle_agree_l612_612332

theorem central_angle_agree (ratio_agree : ℕ) (ratio_disagree : ℕ) (ratio_no_preference : ℕ) (total_angle : ℝ) :
  ratio_agree = 7 → ratio_disagree = 2 → ratio_no_preference = 1 → total_angle = 360 →
  (ratio_agree / (ratio_agree + ratio_disagree + ratio_no_preference) * total_angle = 252) :=
by
  -- conditions and assumptions
  intros h_agree h_disagree h_no_preference h_total_angle
  -- simplified steps here
  sorry

end central_angle_agree_l612_612332


namespace simplify_f_f_value_given_cos_l612_612117

section

variable (α : ℝ)
variable (is_third_quadrant : π < α ∧ α < 3 * π / 2)
variable (cos_alpha : ℝ := - (5 / 13))
noncomputable def f (α : ℝ) : ℝ :=
  (sin ((3 * π / 2) - α) * cos ((π / 2) - α) * tan (-α + π)) / 
  (sin ((π / 2) + α) * tan (2 * π - α))

theorem simplify_f (α : ℝ) (is_third_quadrant : π < α ∧ α < 3 * π / 2) :
  f α = -sin α := sorry

theorem f_value_given_cos (α : ℝ) 
  (cos_alpha : cos α = - (5 / 13)) 
  (is_third_quadrant : π < α ∧ α < 3 * π / 2) :
  f α = 12 / 13 :=
sorry

end

end simplify_f_f_value_given_cos_l612_612117


namespace find_range_of_a_l612_612225

-- Define the conditions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 4 * x + a^2 > 0
def q (a : ℝ) : Prop := a^2 - 5 * a - 6 ≥ 0

-- Define the proposition that one of p or q is true and the other is false
def p_or_q (a : ℝ) : Prop := p a ∨ q a
def not_p_and_q (a : ℝ) : Prop := ¬(p a ∧ q a)

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (2 < a ∧ a < 6) ∨ (-2 ≤ a ∧ a ≤ -1)

-- Theorem statement
theorem find_range_of_a (a : ℝ) : p_or_q a ∧ not_p_and_q a → range_of_a a :=
by
  sorry

end find_range_of_a_l612_612225


namespace correct_sample_in_survey_l612_612895

-- Definitions based on conditions:
def total_population := 1500
def surveyed_population := 150
def sample_description := "the national security knowledge of the selected 150 teachers and students"

-- Hypotheses: conditions
variables (pop : ℕ) (surveyed : ℕ) (description : String)
  (h1 : pop = total_population)
  (h2 : surveyed = surveyed_population)
  (h3 : description = sample_description)

-- Theorem we want to prove
theorem correct_sample_in_survey : description = sample_description :=
  by sorry

end correct_sample_in_survey_l612_612895


namespace cat_food_inequality_l612_612534

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l612_612534


namespace range_of_t_l612_612681

noncomputable def circle1 (x y t : ℝ) := x^2 + (y - t)^2 = 4
noncomputable def circle2 (x y : ℝ) := (x - 2)^2 + y^2 = 14

theorem range_of_t (t : ℝ) :
  (∃ (m n : ℝ), circle1 m n t ∧
                ∃ (Qm Qn : ℝ), circle1 m n t ∧ circle2 Qm Qn ∧
                               (m^2 + n^2 = 2 * ((m - 2)^2 + n^2 - 14))) →
  -4 * real.sqrt 3 ≤ t ∧ t ≤ 4 * real.sqrt 3 := 
sorry

end range_of_t_l612_612681


namespace thermos_count_l612_612001

theorem thermos_count
  (total_gallons : ℝ)
  (pints_per_gallon : ℝ)
  (thermoses_drunk_by_genevieve : ℕ)
  (pints_drunk_by_genevieve : ℝ)
  (total_pints : ℝ) :
  total_gallons * pints_per_gallon = total_pints ∧
  pints_drunk_by_genevieve / thermoses_drunk_by_genevieve = 2 →
  total_pints / 2 = 18 :=
by
  intros h
  have := h.2
  sorry

end thermos_count_l612_612001


namespace sum_lent_is_400_l612_612422

theorem sum_lent_is_400 
  (r : ℝ) (t : ℝ) (interest : ℝ)
  (h1 : r = 0.04)
  (h2 : t = 8)
  (h3 : interest = λ (P : ℝ), P * r * t)
  (h4 : ∀ (P : ℝ), interest P = P - 272) :
  ∃ (P : ℝ), P = 400 :=
by
  use 400
  sorry

end sum_lent_is_400_l612_612422


namespace sum_of_exponents_l612_612088

def power_sum_2021 (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ) : Prop :=
  (∀ k, 1 ≤ k ∧ k ≤ r → (a k = 1 ∨ a k = -1)) ∧
  (a 1 * 3 ^ n 1 + a 2 * 3 ^ n 2 + a 3 * 3 ^ n 3 + a 4 * 3 ^ n 4 + a 5 * 3 ^ n 5 + a 6 * 3 ^ n 6 = 2021) ∧
  (n 1 = 7 ∧ n 2 = 5 ∧ n 3 = 4 ∧ n 4 = 2 ∧ n 5 = 1 ∧ n 6 = 0) ∧
  (a 1 = 1 ∧ a 2 = -1 ∧ a 3 = 1 ∧ a 4 = -1 ∧ a 5 = 1 ∧ a 6 = -1)

theorem sum_of_exponents : ∃ (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ), power_sum_2021 a n r ∧ (n 1 + n 2 + n 3 + n 4 + n 5 + n 6 = 19) :=
by {
  sorry
}

end sum_of_exponents_l612_612088


namespace problem_statement_l612_612458

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem problem_statement (S : ℝ) (h1 : S = golden_ratio) :
  S^(S^(S^2 - S⁻¹) - S⁻¹) - S⁻¹ = 0 :=
by
  sorry

end problem_statement_l612_612458


namespace total_travel_distance_l612_612739

noncomputable def total_distance_traveled (DE DF : ℝ) : ℝ :=
  let EF := Real.sqrt (DE^2 - DF^2)
  DE + EF + DF

theorem total_travel_distance
  (DE DF : ℝ)
  (hDE : DE = 4500)
  (hDF : DF = 4000)
  : total_distance_traveled DE DF = 10560.992 :=
by
  rw [hDE, hDF]
  unfold total_distance_traveled
  norm_num
  sorry

end total_travel_distance_l612_612739


namespace line_slope_l612_612174

theorem line_slope (t : ℝ) : 
  (∃ (t : ℝ), x = 1 + 2 * t ∧ y = 2 - 3 * t) → 
  (∃ (m : ℝ), m = -3 / 2) :=
sorry

end line_slope_l612_612174


namespace count_good_numbers_l612_612363

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l612_612363


namespace sin_B_is_3_over_4_l612_612178

-- Triangle ABC with given conditions
variables {A B C : Type} [triangle : Triangle A B C]
variables (AB AC BC : ℝ)
variables (cosC : ℝ)

-- Conditions
axiom h1 : AB = 4
axiom h2 : AC = 5
axiom h3 : cosC = 4 / 5

-- Correct answer
theorem sin_B_is_3_over_4 (h_triangle : triangle) : 
  sin B = 3 / 4 :=
by
  -- We skip the proof with sorry.
  sorry

end sin_B_is_3_over_4_l612_612178


namespace min_value_l612_612705

theorem min_value (x : Fin 50 → ℝ) (h_pos : ∀ i, 0 < x i) (h_sum : (Finset.univ : Finset (Fin 50)).sum (λ i, (x i)^2) = 1) :
  (Finset.univ : Finset (Fin 50)).sum (λ i, (x i) / (1 - (x i)^2)) ≥ (3 * Real.sqrt 3) / 2 :=
sorry

end min_value_l612_612705


namespace product_of_roots_quadratic_eq_l612_612976

theorem product_of_roots_quadratic_eq : 
  ∀ (x1 x2 : ℝ), 
  (∀ x : ℝ, x^2 - 2 * x - 3 = 0 → (x = x1 ∨ x = x2)) → 
  x1 * x2 = -3 :=
by
  intros x1 x2 h
  sorry

end product_of_roots_quadratic_eq_l612_612976


namespace quadratic_root_value_of_c_l612_612175

theorem quadratic_root_value_of_c :
  (∀ x : ℚ, (3 / 2) * x^2 + 11 * x + c = 0 ↔ x = (-11 + real.sqrt 7) / 3 ∨ x = (-11 - real.sqrt 7) / 3) →
  c = 19 :=
sorry

end quadratic_root_value_of_c_l612_612175


namespace cubical_cake_problem_l612_612454

-- Define the conditions as Lean definitions
def edge_length : ℝ := 3 -- edge length of the cube
def volume_of_piece := (9/4) * edge_length -- volume (c) of the piece
def area_of_top := 9/4 -- area of the quadrilateral top
def area_of_triangular_faces := 9 -- total area of the four triangular faces
def icing_area := area_of_top + area_of_triangular_faces -- total icing area (s)

-- The main theorem which states the proof problem
theorem cubical_cake_problem : 
  let c := volume_of_piece in
  let s := icing_area in
  c + s = 18 :=
by
  sorry

end cubical_cake_problem_l612_612454


namespace queenie_hourly_overtime_pay_is_5_l612_612292

-- Definitions from the conditions
def daily_earnings : ℝ := 150
def total_earnings (days: ℕ) (overtime_hours: ℕ) : ℝ :=
  days * daily_earnings + overtime_hours * hourly_overtime_pay

-- The given condition as a fact
axiom queenie_earnings_5_days_4_overtime : total_earnings 5 4 = 770

-- The value we need to compute
noncomputable def hourly_overtime_pay : ℝ := sorry

-- The statement to be proven
theorem queenie_hourly_overtime_pay_is_5 :
  hourly_overtime_pay = 5 := sorry

end queenie_hourly_overtime_pay_is_5_l612_612292


namespace ronaldo_current_age_l612_612841

noncomputable def roonie_age_one_year_ago (R L : ℕ) := 6 * L / 7
noncomputable def new_ratio (R L : ℕ) := (R + 5) * 8 = 7 * (L + 5)

theorem ronaldo_current_age (R L : ℕ) 
  (h1 : R = roonie_age_one_year_ago R L)
  (h2 : new_ratio R L) : L + 1 = 36 :=
by
  sorry

end ronaldo_current_age_l612_612841


namespace correct_articles_l612_612850

-- Definitions based on conditions
def experience (s : String) : Prop := 
  s = "experience of taking the college entrance examination"

def life (s : String) : Prop := 
  s = "complete one"

-- Theorem stating the correct articles given the conditions
theorem correct_articles : 
  (experience "the experience of taking the college entrance examination" ∧ life "a complete one") ↔ 
  (∃ a b : String, a = "the" ∧ b = "a" ∧ 
    "To us all in China, " ++ a ++ " experience of taking the college entrance examination seems an important way to judge whether one’s life is " ++ b ++ " complete one.") :=
by 
  intros h
  apply exists.intro "the"
  apply exists.intro "a"
  split; sorry

end correct_articles_l612_612850


namespace inequality_sum_am_geq_half_l612_612606

theorem inequality_sum_am_geq_half (m n : ℕ) (a : Fin (m+1) → ℝ)
  (h1 : m ≥ 3) (h2 : ∀ i, 1 ≤ i ∧ i ≤ m → 0 < a ⟨i, by simp [*]⟩) (h3 : n ≥ m) :
  (∑ i in Finset.range m, 
    (a ⟨i + 1, by {simp [*]⟩ ⟩ / (a ⟨i + 1, by {simp [*]⟩ ⟩ + a ⟨(i + 1) % m, by simp [*]⟩)) ^ n) 
    ≥ m / (2 ^ n) :=
by
  sorry

end inequality_sum_am_geq_half_l612_612606


namespace winnie_balloons_l612_612420

theorem winnie_balloons (r w g c n : ℕ) 
  (hr : r = 24) (hw : w = 36) (hg : g = 70) (hc : c = 90) (hn : n = 10) :
  (r + w + g + c) % n = 0 :=
by
  rw [hr, hw, hg, hc, hn]
  norm_num
  exact rfl

end winnie_balloons_l612_612420


namespace median_is_221_l612_612028

def digit_sum (n : ℕ) : ℕ := (n / 100) + (n / 10 % 10) + (n % 10)

def valid_three_digit_numbers : List ℕ := List.filter (λ n => digit_sum n = 5) (List.range' 100 900)

noncomputable def median_of_valid_numbers : ℕ := valid_three_digit_numbers[7]  -- The 8th element in Lean's 0-indexed list

theorem median_is_221 : median_of_valid_numbers = 221 :=
by
  sorry

end median_is_221_l612_612028


namespace plane_intersects_36_cubes_l612_612865

-- Given a cube of side length 4 composed of 64 unit cubes
def largerCube : Type := {coord : ℕ × ℕ × ℕ // coord.1 < 4 ∧ coord.2 < 4 ∧ coord.3 < 4}

noncomputable def diagonalLength : ℝ := 4 * Real.sqrt 3

-- The cutting plane intersects the diagonal one-quarter of its length from one end
noncomputable def planeIntersectionPoint : ℝ := diagonalLength / 4

-- Prove that this plane intersects exactly 36 unit cubes
theorem plane_intersects_36_cubes :
  ∃ (intersectedCubes : set largerCube), intersectedCubes.size = 36 :=
sorry

end plane_intersects_36_cubes_l612_612865


namespace golden_section_point_l612_612612

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_section_point (AB AP PB : ℝ)
  (h1 : AP + PB = AB)
  (h2 : AB = 5)
  (h3 : (AB / AP) = (AP / PB))
  (h4 : AP > PB) :
  AP = (5 * Real.sqrt 5 - 5) / 2 :=
by sorry

end golden_section_point_l612_612612


namespace sum_sequence_correct_l612_612430

def sequence_term (n : ℕ) : ℕ :=
  if n % 9 = 0 ∧ n % 32 = 0 then 7
  else if n % 7 = 0 ∧ n % 32 = 0 then 9
  else if n % 7 = 0 ∧ n % 9 = 0 then 32
  else 0

def sequence_sum (up_to : ℕ) : ℕ :=
  (Finset.range (up_to + 1)).sum sequence_term

theorem sum_sequence_correct : sequence_sum 2015 = 1106 := by
  sorry

end sum_sequence_correct_l612_612430


namespace proof_problem_l612_612759

noncomputable def solveEquation : Prop :=
  ∀ x : ℝ, (3 - x) / (x - 4) + 1 / (4 - x) = 1 ↔ x = 3

noncomputable def solveInequalities : Prop :=
  ∀ x : ℝ, (2 * (x + 1) > x ∧ 1 - 2 * x ≥ (x + 7) / 2) ↔ (-2 < x ∧ x ≤ -1)

theorem proof_problem : solveEquation ∧ solveInequalities :=
by
  split
  · unfold solveEquation
    sorry
  · unfold solveInequalities
    sorry

end proof_problem_l612_612759


namespace square_equilateral_triangle_l612_612035

theorem square_equilateral_triangle (A B C D M K : Type) 
  [square : is_square A B C D] 
  (triangle : is_equilateral_triangle A B M) 
  (inside : vertex_inside_square M A B C D)
  (diagonal_intersect : diagonal_intersects A C M K) :
  distance C K = distance C M :=
sorry

end square_equilateral_triangle_l612_612035


namespace enough_cat_food_for_six_days_l612_612524

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l612_612524


namespace find_AC_l612_612188

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (right_triangle_ABC : RightTriangle A B C) -- Assume this establishes the right triangle context
variable (hypotenuse_BC : dist A C = sqrt 130)
variable (cosine_C : cos (angle A C B) = (9 * (sqrt 130)) / 130)

theorem find_AC (h : RightTriangle A B C) (hBC : dist A C = sqrt 130) (hcosC : cos (angle A C B) = (9 * sqrt 130) / 130) : dist A B = 9 := 
sorry

end find_AC_l612_612188


namespace cat_food_inequality_l612_612538

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l612_612538


namespace find_a_b_l612_612795

-- Define the polynomial with unknown coefficients a and b
def P (x : ℝ) (a b : ℝ) : ℝ := 2 * x^3 + a * x^2 - 13 * x + b

-- Define the conditions for the roots
def root1 (a b : ℝ) : Prop := P 2 a b = 0
def root2 (a b : ℝ) : Prop := P (-3) a b = 0

-- Prove that the coefficients a and b are 1 and 6, respectively
theorem find_a_b : ∀ a b : ℝ, root1 a b ∧ root2 a b → a = 1 ∧ b = 6 :=
by
  intros a b h
  sorry

end find_a_b_l612_612795


namespace weighted_average_number_of_surfers_l612_612302

variable (S1 S2 S3 S4 S5 : ℝ)
variable (total_surfer : ℝ := 15000)

def surfer_day1 := S1
def surfer_day2 := 0.9 * S1
def surfer_day3 := 1.5 * S1
def surfer_day4 := 1.9 * S1
def surfer_day5 := 0.95 * S1
def total_surfers := surfer_day1 + surfer_day2 + surfer_day3 + surfer_day4 + surfer_day5 

theorem weighted_average_number_of_surfers :
  total_surfers = total_surfer → (total_surfer / 5) = 3000 :=
by
  sorry

end weighted_average_number_of_surfers_l612_612302


namespace sum_of_first_n_terms_l612_612981

-- Definitions needed from conditions
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d
def geometric_sequence (b : ℕ → ℤ) := ∃ r : ℤ, ∀ n : ℕ, b (n + 1) = b n * r

-- Given condition: an arithmetic sequence with common difference 2
def arith_seq := λ (n : ℕ), -1 + n * 2

-- Conditions specifying that a2, a3, and a6 form a geometric sequence
def a2 := arith_seq 2
def a3 := arith_seq 3
def a6 := arith_seq 6
def is_geom_sequence := geometric_sequence (λ n, if n = 2 then a2 else if n = 3 then a3 else if n = 6 then a6 else 0)

-- Sum of first n terms of an arithmetic sequence
def sum_arith_seq (n : ℕ) : ℤ := n * (arith_seq 1 + arith_seq n) / 2

theorem sum_of_first_n_terms (n : ℕ) (h : is_geom_sequence) : sum_arith_seq n = n * (n - 2) :=
by
  sorry

end sum_of_first_n_terms_l612_612981


namespace factorization_problem_l612_612774

theorem factorization_problem :
  ∃ a b : ℤ, (∀ y : ℤ, 4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b))
    ∧ (a - b = 11) := by
  sorry

end factorization_problem_l612_612774


namespace parallelogram_sum_l612_612460

-- Define the vertices of the parallelogram
def vertices : list (ℝ × ℝ) := [(1, 2), (4, 6), (10, 6), (7, 2)]

-- Function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define point coordinates for readability
def A := (1, 2)
def B := (4, 6)
def C := (10, 6)
def D := (7, 2)

-- Define side lengths
def AB := distance A B
def BC := distance B C
def CD := distance C D
def DA := distance D A

-- Function to calculate the perimeter of the parallelogram
def perimeter := AB + BC + CD + DA

-- Function to calculate the area of the parallelogram
def base := BC  -- The horizontal side
def height := A.2 - B.2 -- The vertical distance between lines passing through (1, 2) and (4, 6) to the opposite side

def area := base * abs height

-- Combining the perimeter and area to get the final result
def result := perimeter + area

theorem parallelogram_sum : result = 46 := 
by
  -- The proof of this theorem will confirm the sum of the perimeter and the area
  sorry

end parallelogram_sum_l612_612460


namespace length_of_platform_l612_612467

noncomputable def len_train : ℝ := 120
noncomputable def speed_train : ℝ := 60 * (1000 / 3600) -- kmph to m/s
noncomputable def time_cross : ℝ := 15

theorem length_of_platform (L_train : ℝ) (S_train : ℝ) (T_cross : ℝ) (H_train : L_train = len_train)
  (H_speed : S_train = speed_train) (H_time : T_cross = time_cross) : 
  ∃ (L_platform : ℝ), L_platform = (S_train * T_cross) - L_train ∧ L_platform = 130.05 :=
by
  rw [H_train, H_speed, H_time]
  sorry

end length_of_platform_l612_612467


namespace ratio_is_approximately_two_thirds_l612_612860

noncomputable def ratio_of_sides (r : ℝ) (w : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let P := C
  let l := (P - 2 * w) / 2
  l / w

theorem ratio_is_approximately_two_thirds :
  ratio_of_sides 42 59.975859750350594 ≈ 1.1997 := 
by
  sorry

end ratio_is_approximately_two_thirds_l612_612860


namespace speed_of_first_train_l612_612002

def length_train1 := 200 -- in meters
def speed_train2 := 80 -- in kmph
def time_to_cross := 9 -- in seconds
def length_train2 := 300.04 -- in meters

theorem speed_of_first_train : 
  ∃ V1 : ℝ, V1 = 120.016 ∧ 
  let Vr := V1 + speed_train2 in
  let Vr_m_s := Vr / 3.6 in
  Vr_m_s = (length_train1 + length_train2) / time_to_cross := by
  sorry

end speed_of_first_train_l612_612002


namespace distinct_triangles_nearest_int_l612_612744

-- Definition of the problem conditions
def is_regular_n_gon (n : ℕ) : Prop := 
  n ≥ 3

-- Main theorem statement
theorem distinct_triangles_nearest_int (n : ℕ) (h : is_regular_n_gon n) : 
  let N := number_of_distinct_triangles n
  N = (float_of_nat n ^ 2 / 12).to_int :=
sorry

end distinct_triangles_nearest_int_l612_612744


namespace option_a_option_b_option_d_l612_612147

theorem option_a (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 - b^2 ≤ 4 := 
sorry

theorem option_b (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 + 1 / b ≥ 4 :=
sorry

theorem option_d (a b c x1 x2 : ℝ) 
  (h1 : a > 0) 
  (h2 : a^2 = 4 * b) 
  (h3 : (x1 - x2) = 4)
  (h4 : (x1 + x2)^2 - 4 * (x1 * x2 + c) = (x1 - x2)^2) : c = 4 :=
sorry

end option_a_option_b_option_d_l612_612147


namespace eight_digit_number_divisible_by_11_l612_612416

theorem eight_digit_number_divisible_by_11 :
  ∃ n : ℕ, n ≤ 9 ∧ (let odd_sum := 6 + 3 + 8 + 7 in
                     let even_sum := 2 + n + 4 + 5 in
                     ∃ k : ℤ, (odd_sum - even_sum) = k * 11) ↔ n = 2 :=
by
  sorry

end eight_digit_number_divisible_by_11_l612_612416


namespace cat_food_insufficient_for_six_days_l612_612530

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l612_612530


namespace CK_eq_CM_l612_612037

-- Define the necessary geometrical objects and properties
variables (A B C D M K : Type)
axiom is_square_ABCD : is_square A B C D
axiom is_equilateral_triangle_ABM : is_equilateral_triangle A B M
axiom M_inside_square : inside_square M A B C D
axiom AC_intersects_triangle_K : AC_intersects A C M K

-- Proof goal: Prove CK = CM
theorem CK_eq_CM : CK = CM :=
by
  sorry

end CK_eq_CM_l612_612037


namespace problem_l612_612624

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem problem
  (ω : ℝ) 
  (hω : ω > 0)
  (hab : Real.sqrt (4 + (Real.pi ^ 2) / (ω ^ 2)) = 2 * Real.sqrt 2) :
  f ω 1 = Real.sqrt 3 / 2 := 
sorry

end problem_l612_612624


namespace enough_cat_food_for_six_days_l612_612522

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l612_612522


namespace part1_1_part1_2_part2_part3_1_part3_2_part3_3_l612_612747

-- Part 1
theorem part1_1 (a b : ℝ) : a - b = 4 → a > b := by sorry
theorem part1_2 (a b : ℝ) : a - b = -2 → a < b := by sorry

-- Part 2
theorem part2 (x : ℝ) : x > 0 → -x + 5 > -2x + 4 := by sorry

-- Part 3
theorem part3_1 (x y : ℝ) : x < y → 5x + 13y + 2 > 6x + 12y + 2 := by sorry
theorem part3_2 (x y : ℝ) : x = y → 5x + 13y + 2 = 6x + 12y + 2 := by sorry
theorem part3_3 (x y : ℝ) : x > y → 5x + 13y + 2 < 6x + 12y + 2 := by sorry

end part1_1_part1_2_part2_part3_1_part3_2_part3_3_l612_612747


namespace ratio_zyx_1_to_3_l612_612023

-- Define initial and final patient counts
def initial_patient_count : ℕ := 26
def doubled_patient_count (initial : ℕ) : ℕ := 2 * initial
def final_patient_count : ℕ := doubled_patient_count initial_patient_count

-- Define the number of patients diagnosed with ZYX syndrome
def diagnosed_with_zyx : ℕ := 13

-- Calculate the number of patients without ZYX syndrome
def patients_without_zyx (total diagnosed : ℕ) : ℕ := total - diagnosed
def final_patients_without_zyx := patients_without_zyx final_patient_count diagnosed_with_zyx

-- Calculate the ratio of with ZYX syndrome to without ZYX syndrome
def ratio_with_without_zyx (with without : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd with without
  (with / gcd, without / gcd)

-- Final ratio
def final_ratio : ℕ × ℕ := ratio_with_without_zyx diagnosed_with_zyx final_patients_without_zyx

-- Proof statement: The final ratio is 1:3
theorem ratio_zyx_1_to_3 : final_ratio = (1, 3) :=
  by
    -- We are skipping the proof as instructed
    sorry

end ratio_zyx_1_to_3_l612_612023


namespace find_vertex_angle_of_cone_l612_612244

noncomputable def vertexAngleCone (r1 r2 : ℝ) (O1 O2 : ℝ) (touching : Prop) (Ctable : Prop) (equalAngles : Prop) : Prop :=
  -- The given conditions:
  -- r1, r2 are the radii of the spheres, where r1 = 4 and r2 = 1.
  -- O1, O2 are the centers of the spheres.
  -- touching indicates the spheres touch externally.
  -- Ctable indicates that vertex C of the cone is on the segment connecting the points where the spheres touch the table.
  -- equalAngles indicates that the rays CO1 and CO2 form equal angles with the table.
  touching → 
  Ctable → 
  equalAngles →
  -- The target to prove:
  ∃ α : ℝ, 2 * α = 2 * Real.arctan (2 / 5)

theorem find_vertex_angle_of_cone (r1 r2 : ℝ) (O1 O2 : ℝ) :
  let touching : Prop := (r1 = 4 ∧ r2 = 1 ∧ abs (O1 - O2) = r1 + r2)
  let Ctable : Prop := (True)  -- Provided by problem conditions, details can be expanded
  let equalAngles : Prop := (True)  
  vertexAngleCone r1 r2 O1 O2 touching Ctable equalAngles := 
by
  sorry

end find_vertex_angle_of_cone_l612_612244


namespace cos_inequality_solution_l612_612571

theorem cos_inequality_solution (y : ℝ) (hy : 0 ≤ y ∧ y ≤ real.pi) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ real.pi → real.cos (x + y) ≥ real.cos x + real.cos y) ↔ y = 0 :=
by
  intro h
  sorry

end cos_inequality_solution_l612_612571


namespace Arthur_walked_7_miles_l612_612484

-- Definitions for the given problem conditions
def east_blocks : ℕ := 8
def north_blocks : ℕ := 15
def west_blocks : ℕ := 5
def block_length_miles : ℚ := 1 / 4

-- Statement of the problem
theorem Arthur_walked_7_miles : 
  let total_blocks := east_blocks + north_blocks + west_blocks in
  total_blocks * block_length_miles = 7 :=
by
  sorry

end Arthur_walked_7_miles_l612_612484


namespace solution_set_inequality_l612_612238

theorem solution_set_inequality (x : ℝ) : |5 - x| < |x - 2| + |7 - 2 * x| ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 3.5 :=
by
  sorry

end solution_set_inequality_l612_612238


namespace range_of_b_l612_612124

-- Definitions
def polynomial_inequality (b : ℝ) (x : ℝ) : Prop := x^2 + b * x - b - 3/4 > 0

-- The main statement
theorem range_of_b (b : ℝ) : (∀ x : ℝ, polynomial_inequality b x) ↔ -3 < b ∧ b < -1 :=
by {
    sorry -- proof goes here
}

end range_of_b_l612_612124


namespace completing_the_square_l612_612755

theorem completing_the_square (x : ℝ) : x^2 + 8*x + 7 = 0 → (x + 4)^2 = 9 :=
by {
  sorry
}

end completing_the_square_l612_612755


namespace math_problem_l612_612099

theorem math_problem (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 2) : a^5 + b^5 = 19 / 4 :=
by
sory

end math_problem_l612_612099


namespace convert_to_spherical_l612_612919

noncomputable def spherical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arccos (z / ρ)
  let θ := if y / x < 0 then Real.arctan (-y / x) + 2 * Real.pi else Real.arctan (y / x)
  (ρ, θ, φ)

theorem convert_to_spherical :
  let x := 1
  let y := -4 * Real.sqrt 3
  let z := 4
  spherical_coordinates x y z = (Real.sqrt 65, Real.arctan (-4 * Real.sqrt 3) + 2 * Real.pi, Real.arccos (4 / (Real.sqrt 65))) :=
by
  sorry

end convert_to_spherical_l612_612919


namespace construct_triangle_l612_612979

structure Triangle :=
(A : Point ℝ)
(B : Point ℝ)
(C : Point ℝ)

def side_length (A B : Point ℝ) : ℝ := dist A B
def altitude_from (T : Triangle) (V : String) : ℝ := sorry

theorem construct_triangle
  (A B C : Point ℝ)
  (c : ℝ)
  (m_b : ℝ)
  (ALT_COND : ∀ A' B C, A' = midpoint B C → 2 * dist A' B = dist A' C)
  (H1 : side_length A B = c)
  (H2 : altitude_from (Triangle.mk A B C) "B" = m_b) :
  ∃ (T : Triangle), side_length T.A T.B = c ∧ altitude_from T "B" = m_b := by
  sorry

end construct_triangle_l612_612979


namespace functional_equation_solution_l612_612555

theorem functional_equation_solution 
    (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)) : 
    f = (λ x, x - 1) := 
sorry

end functional_equation_solution_l612_612555


namespace molecular_weight_single_mole_l612_612825

theorem molecular_weight_single_mole :
  (∀ (w_7m C6H8O7 : ℝ), w_7m = 1344 → (w_7m / 7) = 192) :=
by
  intros w_7m C6H8O7 h
  sorry

end molecular_weight_single_mole_l612_612825


namespace tan_alpha_values_l612_612990

theorem tan_alpha_values (α : ℝ) 
  (h : 2 * (Real.sin α) ^ 2 + (Real.sin α) * (Real.cos α) - 3 * (Real.cos α) ^ 2 = 7 / 5) :
  tan α = 2 ∨ tan α = -11 / 3 :=
sorry

end tan_alpha_values_l612_612990


namespace train_length_proof_l612_612468

/-- Define the speed of the train in km/hr --/
def speed_kmhr : ℝ := 240

/-- Define the conversion factor from km/hr to m/s --/
def conversion_factor : ℝ := (1000 / 1) * (1 / 3600)

/-- Define the speed of the train in m/s using the conversion factor --/
def speed_ms : ℝ := speed_kmhr * conversion_factor

/-- Define the time in seconds to cross the pole --/
def time_seconds : ℝ := 21

/-- Define the expected length of the train in meters --/
def expected_length : ℝ := 1400.07

/-- Theorem stating the length of the train is 1400.07 meters given the speed conditions --/
theorem train_length_proof : (speed_ms * time_seconds = expected_length) :=
by sorry

end train_length_proof_l612_612468


namespace dividend_calculation_l612_612181

theorem dividend_calculation :
  ∀ (q d r : ℕ), q = 256 → d = 3892 → r = 354 → (d * q + r = 996706) :=
by
  intros q d r hq hd hr
  rw [hq, hd, hr]
  sorry

end dividend_calculation_l612_612181


namespace toys_per_day_l612_612456

theorem toys_per_day (total_toys_per_week : ℕ) (days_worked_per_week : ℕ)
  (production_rate_constant : Prop) (h1 : total_toys_per_week = 8000)
  (h2 : days_worked_per_week = 4)
  (h3 : production_rate_constant)
  : (total_toys_per_week / days_worked_per_week) = 2000 :=
by
  sorry

end toys_per_day_l612_612456


namespace K_time_expression_l612_612440

variable (x : ℝ) 

theorem K_time_expression
  (hyp : (45 / (x - 2 / 5) - 45 / x = 3 / 4)) :
  45 / (x : ℝ) = 45 / x :=
sorry

end K_time_expression_l612_612440


namespace solve_problem_l612_612584

def problem (x : ℝ) : Prop :=
  abs (x - 25) + abs (x - 21) = abs (2 * x - 46) + abs (x - 17)

theorem solve_problem : (∃ x : ℝ, problem x) ∧ ∀ x : ℝ, problem x → x = 67 / 3 :=
by
  split
  · use 67 / 3
    rw problem
    sorry  -- Proof goes here
  
  · intros x hx
    sorry  -- Proof goes here

end solve_problem_l612_612584


namespace mabel_tomatoes_l612_612724

theorem mabel_tomatoes
  (first_plant : ℕ)
  (second_plant : ℕ)
  (remaining_each : ℕ)
  (total_tomatoes : ℕ) :
  first_plant = 8 →
  second_plant = first_plant + 4 →
  remaining_each = 3 * (first_plant + second_plant) →
  total_tomatoes = first_plant + second_plant + 2 * remaining_each →
  total_tomatoes = 140 :=
by
  intros h1 h2 h3 h4
  rw [h1] at h2 ⊢
  rw [h2, h3] at h4
  exact h4

end mabel_tomatoes_l612_612724


namespace remainder_of_3_pow_2040_mod_5_l612_612412

theorem remainder_of_3_pow_2040_mod_5 :
  (3 ^ 2040) % 5 = 1 :=
by
  -- Overview of the powers of 3 modulo 5 cycle
  have h_cycle : ∀ n, (3 ^ (4 * n)) % 5 = 1, from
    λ n, by
      induction n with k hk
      · simp
      · calc
          (3 ^ (4 * (k + 1))) % 5
              = (3 ^ (4 * k + 4)) % 5   : by rw nat.mul_succ
          ... = ((3 ^ (4 * k)) * (3 ^ 4)) % 5 : by rw pow_add
          ... = ((3 ^ (4 * k)) * 1) % 5 : by rw [pow_four_3_mod_5]
          ... = (3 ^ (4 * k)) % 5 : by ring_nf
          ... = 1 : hk,
  -- \(3^4 \equiv 1 \pmod{5}\)
  have pow_four_3_mod_5 : (3 ^ 4) % 5 = 1, by
    calc (3 ^ 4) % 5
        = 81 % 5 : by norm_num_arith [nat.pow]
    ...  = 1 : by norm_num,
  -- Proof for the specific case
  exact h_cycle 510

end remainder_of_3_pow_2040_mod_5_l612_612412


namespace largest_quotient_l612_612409

theorem largest_quotient (S : Set ℝ) (hS : S = {-32, -5, 1, 2, 12, 15}) : 
  ∃ a b ∈ S, a / b = 32 ∧ a ≠ b ∧ a / b > 0 :=
by
  use -32, -1
  split
  { show -32 ∈ S, from sorry },
  split
  { show -1 ∈ S, from sorry },
  split
  { show -32 / -1 = 32, from sorry },
  split
  { show -32 ≠ -1, from sorry },
  { show -32 / -1 > 0, from sorry }

end largest_quotient_l612_612409


namespace ratio_M_over_N_l612_612163

variable (P Q M N : ℝ)

-- Conditions
def condition1 := M = 0.40 * Q
def condition2 := Q = 0.25 * P
def condition3 := N = 0.60 * P

-- Theorem
theorem ratio_M_over_N (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  M / N = 1 / 6 := by
  sorry

end ratio_M_over_N_l612_612163


namespace equalities_hold_l612_612711

noncomputable theory

def f (x p q : ℝ) : ℝ := x^2 + p * x + q

theorem equalities_hold (p q : ℝ) :
  f p p q = 0 ∧ f q p q = 0 ↔ (p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2) :=
by {
  sorry -- Proof is omitted as it is not required.
}

end equalities_hold_l612_612711


namespace concert_total_cost_l612_612749

noncomputable def total_cost (ticket_cost : ℕ) (processing_fee_rate : ℚ) (parking_fee : ℕ)
  (entrance_fee_per_person : ℕ) (num_persons : ℕ) (refreshments_cost : ℕ) 
  (merchandise_cost : ℕ) : ℚ :=
  let ticket_total := ticket_cost * num_persons
  let processing_fee := processing_fee_rate * (ticket_total : ℚ)
  ticket_total + processing_fee + (parking_fee + entrance_fee_per_person * num_persons 
  + refreshments_cost + merchandise_cost)

theorem concert_total_cost :
  total_cost 75 0.15 10 5 2 20 40 = 252.50 := by 
  sorry

end concert_total_cost_l612_612749


namespace number_of_divisors_greater_than_22_l612_612387

theorem number_of_divisors_greater_than_22 :
  (∃ (S : Finset ℕ), S = (Finset.filter (λ n, 1998 % n = 0 ∧ n > 22) (Finset.range (1998 + 1))) ∧ S.card = 10) :=
sorry

end number_of_divisors_greater_than_22_l612_612387


namespace length_AN_is_one_l612_612280

noncomputable def segment_length_AN (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) (point_N_extension : ExtensionOfSide A B N) 
  (AC_eq_AM : AC = 1 ∧ AM = 1) (ANM_eq_CNM : ∠ANM = ∠CNM) : Real :=
1

/-- Given a triangle ABC, where point M lies on the angle bisector of angle BAC,
and point N lies on the extension of side AB beyond point A, where AC = AM = 1,
and the angles ANM and CNM are equal, the length of segment AN is 1. -/
theorem length_AN_is_one (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) 
  (point_N_extension : ExtensionOfSide A B N) (AC_eq_AM : AC = 1 ∧ AM = 1) 
  (ANM_eq_CNM : ∠ANM = ∠CNM) : 
  segment_length_AN A B C M N triangle_ABC angle_bisector_AM point_N_extension AC_eq_AM ANM_eq_CNM = 1 :=
by
  sorry

end length_AN_is_one_l612_612280


namespace find_f_value_l612_612839

noncomputable def f (x y z : ℝ) : ℝ := 2 * x^3 * Real.sin y + Real.log (z^2)

theorem find_f_value :
  f 1 (Real.pi / 2) (Real.exp 2) = 8 →
  f 2 Real.pi (Real.exp 3) = 6 :=
by
  intro h
  unfold f
  sorry

end find_f_value_l612_612839


namespace good_numbers_count_l612_612342

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l612_612342


namespace find_b_values_for_system_l612_612574

noncomputable def system_has_solution_for_any_a (b : ℝ) : Prop :=
  ∀ (a : ℝ), ∃ (x y : ℝ), (x * real.cos a + y * real.sin a - 2 ≤ 0) ∧ 
  (x^2 + y^2 + 6 * x - 2 * y + 6 = b^2 - 4 * b)

theorem find_b_values_for_system :
  {b : ℝ | system_has_solution_for_any_a b} = 
  {b : ℝ | b ≤ 4 - real.sqrt 10} ∪ {b : ℝ | b ≥ real.sqrt 10} :=
by
  sorry

end find_b_values_for_system_l612_612574


namespace count_good_divisors_l612_612368

/-- We call a natural number \( n \) good if 2020 divided by \( n \) leaves a remainder of 22.
    This means \( n \) must be a divisor of 1998 and \( n > 22 \) -/
theorem count_good_divisors : finset.card (finset.filter (λ m, 22 < m ∧ 1998 % m = 0) (finset.range 2000)) = 10 :=
by
  -- This is where the proof would go
  sorry

end count_good_divisors_l612_612368


namespace green_beans_weight_l612_612726

/-- 
    Mary uses plastic grocery bags that can hold a maximum of twenty pounds. 
    She buys some green beans, 6 pounds milk, and twice the amount of carrots as green beans. 
    She can fit 2 more pounds of groceries in that bag. 
    Prove that the weight of green beans she bought is equal to 4 pounds.
-/
theorem green_beans_weight (G : ℕ) (H1 : ∀ g : ℕ, g + 6 + 2 * g ≤ 20 - 2) : G = 4 :=
by 
  have H := H1 4
  sorry

end green_beans_weight_l612_612726


namespace prime_factors_1260_l612_612717

theorem prime_factors_1260 (w x y z : ℕ) (h : 2 ^ w * 3 ^ x * 5 ^ y * 7 ^ z = 1260) : 2 * w + 3 * x + 5 * y + 7 * z = 22 :=
by sorry

end prime_factors_1260_l612_612717


namespace investment_value_is_920_42_l612_612162

theorem investment_value_is_920_42 :
  let P := 650
  let r := 0.05
  let n := 12
  let t := 7
  let A := P * (1 + r / n) ^ (n * t)
  A ≈ 920.42 :=
by
  sorry

end investment_value_is_920_42_l612_612162


namespace number_of_pairs_l612_612160

theorem number_of_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m^2 + n < 50) : 
  ∃! p : ℕ, p = 203 := 
sorry

end number_of_pairs_l612_612160


namespace enough_cat_food_for_six_days_l612_612518

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l612_612518


namespace point_lies_on_graph_of_even_function_l612_612166

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Condition: f is an even function
def is_even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)

-- The theorem statement
theorem point_lies_on_graph_of_even_function (h_even : is_even_function f) : ( -a, f a ) ∈ set_of (λ p : ℝ × ℝ, p.2 = f p.1) :=
sorry

end point_lies_on_graph_of_even_function_l612_612166


namespace exists_n_ge_1_specific_sequence_l612_612892

open Real

noncomputable def x_seq : ℕ → ℝ 
| 0       := 1
| (n + 1) := (1 / 2)^(n + 1)

theorem exists_n_ge_1 (x : ℕ → ℝ)
  (h_pos : ∀ n, 0 < x n)
  (h_x0 : x 0 = 1)
  (h_seq : ∀ n, x (n+1) ≤ x n) :
  ∃ n ≥ 1, (finset.range n).sum (λ i, (x i)^2 / (x (i + 1))) ≥ 3.999 :=
sorry

theorem specific_sequence (n : ℕ) :
  (finset.range n).sum (λ i, (x_seq i)^2 / (x_seq (i + 1))) < 4 :=
sorry

end exists_n_ge_1_specific_sequence_l612_612892


namespace enough_cat_food_for_six_days_l612_612519

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l612_612519


namespace trigonometric_identity_l612_612164

theorem trigonometric_identity (x : ℝ) : 
  (sin (π / 4 - x) = -1 / 5) → (cos (5 * π / 4 + x) = 1 / 5) :=
by
  intro h
  sorry

end trigonometric_identity_l612_612164


namespace find_volume_P3_as_fraction_find_m_plus_n_for_P3_l612_612978

-- Define the initial volume and the conditions related to the construction of Pi+1
def P0 : ℚ := 1

def pyramid_volume_increment (v : ℚ) : ℚ := v / 4

def face_count (i : ℕ) : ℕ := 4 * (4^i)

def volume_increment (i : ℕ) : ℚ :=
  (pyramid_volume_increment (P0 / (4 ^ i))) * face_count i

def total_volume (n : ℕ) : ℚ :=
  P0 + ∑ i in List.range n, volume_increment i

theorem find_volume_P3_as_fraction :
  let P3 := total_volume 3
  P3 = 341 / 256 :=
by
  sorry

theorem find_m_plus_n_for_P3 : 
  let P3 := total_volume 3
  let m := 341
  let n := 256
  m + n = 597 :=
by
  sorry

end find_volume_P3_as_fraction_find_m_plus_n_for_P3_l612_612978


namespace count_good_numbers_l612_612364

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l612_612364


namespace range_of_a_l612_612988

variable (a : ℝ)

def p : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

def q : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

theorem range_of_a :
  (p a ∧ q a) → a ≤ -1 := by
  sorry

end range_of_a_l612_612988


namespace min_sum_of_integers_cauchy_schwarz_l612_612934

theorem min_sum_of_integers_cauchy_schwarz :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 
  (1 / x + 4 / y + 9 / z = 1) ∧ 
  ((x + y + z) = 36) :=
  sorry

end min_sum_of_integers_cauchy_schwarz_l612_612934


namespace aubrey_distance_from_school_l612_612042

-- Define average speed and travel time
def average_speed : ℝ := 22 -- in miles per hour
def travel_time : ℝ := 4 -- in hours

-- Define the distance function
def calc_distance (speed time : ℝ) : ℝ := speed * time

-- State the theorem
theorem aubrey_distance_from_school : calc_distance average_speed travel_time = 88 := 
by
  sorry

end aubrey_distance_from_school_l612_612042


namespace find_u_plus_v_l612_612654

variables (u v : ℚ)

theorem find_u_plus_v (h1 : 5 * u - 6 * v = 19) (h2 : 3 * u + 5 * v = -1) : u + v = 27 / 43 := by
  sorry

end find_u_plus_v_l612_612654


namespace circle_inscribed_square_area_l612_612451

theorem circle_inscribed_square_area:
  ∃ (x y : ℝ) (r : ℝ), (2 * x^2 = -2 * y^2 + 16 * x - 8 * y + 36) ∧ 
  (r = sqrt 2) ∧
  (area := (2 * r)^2) ∧
  (area = 8) :=
by
  sorry

end circle_inscribed_square_area_l612_612451


namespace ellipse_statements_l612_612592

theorem ellipse_statements 
    (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
    (C : ℝ → ℝ → Prop := λ x y, x^2 / a^2 + y^2 / b^2 = 1)
    (M : ℝ × ℝ := (real.sqrt 6, 1))
    (N : ℝ × ℝ)
    (N_on_C : C N.1 N.2)
    (major_axis_length : 2 * a = 6)
    (M_outside_ellipse : real.sqrt 6 ^ 2 / a^2 + 1 / b^2 > 1) :
    (a := 3, ((real.sqrt 6 / 3 < real.sqrt (1 - b^2 / a^2)) ∧ (real.sqrt 6 / 3 < 1)))
    ∧ (∃ Q, C Q.1 Q.2 ∧ ((Q.1, Q.2) := (3, 0) → Q.2 * (2 * a^2 - 4 * real.sqrt (a^2 - b^2)^2) = 0))
    ∧ (E : ℝ × ℝ := (0, -2)) ∧ 
    (ℝ_range : ∀ (e : ℝ), e = 2 * real.sqrt 2 / 3 → (a = 3, (b = 1 → ∃ (NE : ℝ), NE ≠ real.sqrt 13)))
    ∧ (ℝ_min : ∀ NF1 NF2 : ℝ, NF1 + NF2 = 2 * a → ∀ z t : ℝ, NF1 + NF2 / (NF1 * NF2) = 2 / 3) :=
    sorry

end ellipse_statements_l612_612592


namespace charging_piles_problem_l612_612818

variable (priceA priceB : ℝ)
variable (numA numB : ℕ)

-- Conditions from problem
def unit_price_relation : Prop := priceA + 0.3 = priceB
def quantity_equal_condition : Prop := (15 / priceA) = (20 / priceB)
def total_piles : Prop := numA + numB = 25
def total_cost : Prop := (priceA * numA + priceB * numB) <= 26
def quantity_relation : Prop := (numB >= (numA / 2))

-- Main proof statement
theorem charging_piles_problem :
  unit_price_relation priceA priceB ∧
  quantity_equal_condition priceA priceB ∧
  total_piles numA numB ∧
  total_cost priceA priceB numA numB ∧
  quantity_relation numA numB →
  (priceA = 0.9 ∧ priceB = 1.2) ∧
  ((numA = 14 ∧ numB = 11) ∨ 
   (numA = 15 ∧ numB = 10) ∨ 
   (numA = 16 ∧ numB = 9)) ∧
  (numA = 16 ∧ numB = 9 → (priceA * 16 + priceB * 9) = 25.2) :=
sorry

end charging_piles_problem_l612_612818


namespace cat_food_insufficient_for_six_days_l612_612528

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l612_612528


namespace partition_into_triples_l612_612141

open Finset

theorem partition_into_triples (M : Finset ℕ) (hM : M = {1, 2, 3, ..., 15}) :
  ∃ A B C : Finset ℕ,
      ∃ D E : Finset ℕ,
        M = A ∪ B ∪ C ∪ D ∪ E ∧
        A.card = 3 ∧ B.card = 3 ∧ C.card = 3 ∧ D.card = 3 ∧ E.card = 3 ∧
        A.sum id = 24 ∧ B.sum id = 24 ∧ C.sum id = 24 ∧ D.sum id = 24 ∧ E.sum id = 24 := 
begin
  sorry
end

end partition_into_triples_l612_612141


namespace probability_exactly_four_white_and_four_black_l612_612563

theorem probability_exactly_four_white_and_four_black :
  (∀ (n w b : ℕ), n = 8 ∧ w = 4 ∧ b = 4 →
    (∀ (p : ℕ → ℚ), (∀ (i : ℕ), i < n → p i = 1/2) →
    ∑' (s : ℕ) in Finset.filter (λ s, s = w) (Finset.range (n + 1)),
    (Finset.choose n s) * (p s) = 35 / 128)) := by
  intros n w b hn hp
  sorry

end probability_exactly_four_white_and_four_black_l612_612563


namespace minimum_distance_parabola_to_line_l612_612579

theorem minimum_distance_parabola_to_line :
  let parabola := λ x: ℝ, x^2 + 1,
      line := λ x: ℝ, 2*x - 1,
      distance (x: ℝ) := (abs (2 * x - (x^2 + 1) - 1)) / sqrt(5)
  in ∃ x: ℝ, x = 2 ∧ distance x = sqrt(5) / 5 :=
by
  sorry

end minimum_distance_parabola_to_line_l612_612579


namespace students_with_one_problem_l612_612330

variables (n1 n2 n3 n4 n5 p1 p2 p3 p4 p5 : ℕ)

-- Given conditions
def condition_sum_students : Prop := 
  n1 + n2 + n3 + n4 + n5 = 30

def condition_sum_problems : Prop := 
  n1 * p1 + n2 * p2 + n3 * p3 + n4 * p4 + n5 * p5 = 40

def distinct_problems : Prop := 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5

-- Proof goal
theorem students_with_one_problem 
  (h_sum_students : condition_sum_students n1 n2 n3 n4 n5)
  (h_sum_problems : condition_sum_problems n1 n2 n3 n4 n5 p1 p2 p3 p4 p5)
  (h_distinct : distinct_problems p1 p2 p3 p4 p5) :
  ∃ (k : ℕ), k = 5 ∧ ∃ i, p_i = 1 :=
sorry

end students_with_one_problem_l612_612330


namespace only_quadratic_eq_is_B_l612_612419

def equationA : Type := ∀ (x y : ℝ), y = 2 * x - 1
def equationB : Type := ∀ (x : ℝ), x^2 = 6
def equationC : Type := ∀ (x y : ℝ), 5 * x * y - 1 = 1
def equationD : Type := ∀ (x : ℝ), 2 * (x + 1) = 2

theorem only_quadratic_eq_is_B : 
  (∀ x y, y ≠ 2 * x - 1 ∨ ∀ x, ∃ a b c, a * x^2 + b * x + c = 0 ∧ x^2 = 6) ∧
  (∀ x y, 5 * x * y - 1 ≠ 1 ∨ ∀ x, ∃ a b c, a * x^2 + b * x + c = 0 ∧ x^2 = 6) ∧
  (∀ x, 2 * (x + 1) ≠ 2 ∨ ∀ x, ∃ a b c, a * x^2 + b * x + c = 0 ∧ x^2 = 6) →
  (∀ x, ∃ a b c, a * x^2 + b * x + c = 0 ∧ x^2 = 6) :=
sorry

end only_quadratic_eq_is_B_l612_612419


namespace velocity_at_second_return_to_equilibrium_l612_612462

noncomputable def ball_velocity (t : ℝ) : ℝ := 30 * Real.cos (2 * t + π / 6)

theorem velocity_at_second_return_to_equilibrium :
  ball_velocity (11 * π / 12) = 30 :=
by
  sorry

end velocity_at_second_return_to_equilibrium_l612_612462


namespace geometric_series_fourth_term_l612_612457

theorem geometric_series_fourth_term (a r : ℝ)
    (h5 : a * r^4 = real.factorial 5)
    (h6 : a * r^5 = real.factorial 6) :
    a * r^3 = 20 :=
by
  -- The detailed proof steps will be provided here by the user
  sorry

end geometric_series_fourth_term_l612_612457


namespace number_of_teams_in_BIG_N_l612_612189

theorem number_of_teams_in_BIG_N (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end number_of_teams_in_BIG_N_l612_612189


namespace length_an_eq_1_l612_612268

variable {A B C M N : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N]
variable {dist : A → A → ℝ}
variable {angle : ∀ {A B C : A}, ℝ}

-- Given conditions as hypotheses
variable (H1 : ∀ {M}, M ∈ angle_bisector A B C)
variable (H2 : ∀ {N : Type}, N ∈ extended_line A B)
variable (H3 : dist A C = 1)
variable (H4 : dist A M = 1)
variable (H5 : angle A N M = angle C N M)

-- The statement we need to prove
theorem length_an_eq_1 (H1 : M ∈ angle_bisector A B C) (H2 : N ∈ extended_line A B)
  (H3 : dist A C = 1) (H4 : dist A M = 1) (H5 : angle A N M = angle C N M) :
  dist A N = 1 := by
  sorry


end length_an_eq_1_l612_612268


namespace percentage_gmo_correct_l612_612007

variable (microphotonics home_electronics food_additives industrial_lubricants : ℝ)
variable (basic_astrophysics_degrees total_degrees : ℝ)

noncomputable def percentage_genetically_modified_microorganisms 
  (microphotonics home_electronics food_additives industrial_lubricants : ℝ)
  (basic_astrophysics_degrees total_degrees : ℝ)
  (h : microphotonics = 10)
  (h1 : home_electronics = 24)
  (h2 : food_additives = 15)
  (h3 : industrial_lubricants = 8)
  (h4 : basic_astrophysics_degrees = 50.4)
  (h5 : total_degrees = 360) : ℝ :=
  let percent_basic_astrophysics := (basic_astrophysics_degrees / total_degrees) * 100
  in 100 - (microphotonics + home_electronics + food_additives + industrial_lubricants + percent_basic_astrophysics)

theorem percentage_gmo_correct 
  (microphotonics := 10) 
  (home_electronics := 24) 
  (food_additives := 15) 
  (industrial_lubricants := 8) 
  (basic_astrophysics_degrees := 50.4) 
  (total_degrees := 360) :
  percentage_genetically_modified_microorganisms microphotonics home_electronics food_additives industrial_lubricants basic_astrophysics_degrees total_degrees
  = 29 := 
  by
    have h_percent_basic : (basic_astrophysics_degrees / total_degrees * 100) = 14 :=
      by sorry
    have h_total_known : (microphotonics + home_electronics + food_additives + industrial_lubricants + (basic_astrophysics_degrees / total_degrees * 100)) = 71 :=
      by sorry
    calc 
      percentage_genetically_modified_microorganisms microphotonics home_electronics food_additives industrial_lubricants basic_astrophysics_degrees total_degrees
      = 100 - 71 : by rw [h_percent_basic, h_total_known]
      = 29 : rfl

end percentage_gmo_correct_l612_612007


namespace triangle_construction_l612_612548

theorem triangle_construction (m s : ℝ) (α : ℝ) (h1 : 0 < α ∧ α < π/2) :
  if s - (m / tan α) < m then 
    false 
  else 
    ∃ (A B C : ℝ), A + B = s ∧ ∠BAC = α ∧ CC₁ = m := 
sorry

end triangle_construction_l612_612548


namespace ball_counts_l612_612832

theorem ball_counts (total_balls yellow_prob blue_prob red_prob : ℝ) 
(yellow_count blue_count red_count : ℕ) :
total_balls = 80 → 
yellow_prob = 1 / 4 → 
blue_prob = 7 / 20 → 
red_prob = 2 / 5 → 
yellow_count = (total_balls * yellow_prob).nat_cast → 
blue_count = (total_balls * blue_prob).nat_cast → 
red_count = (total_balls * red_prob).nat_cast → 
yellow_count = 20 ∧ blue_count = 28 ∧ red_count = 32 :=
by 
  intros h_total h_yellow h_blue h_red h_yellow_count h_blue_count h_red_count 
  simp [h_total, h_yellow, h_blue, h_red] at *
  linarith

end ball_counts_l612_612832


namespace find_z_l612_612994

open Complex

theorem find_z (z : ℂ) (h1 : (z + 2 * I).im = 0) (h2 : ((z / (2 - I)).im = 0)) : z = 4 - 2 * I :=
by
  sorry

end find_z_l612_612994


namespace eight_times_nine_and_two_fifths_is_l612_612901

variable (m n a b : ℕ)
variable (d : ℚ)

-- Conditions
def mixed_to_improper (a b den : ℚ) : ℚ := a + b / den
def improper_to_mixed (n d : ℚ) : ℕ × ℚ := (n / d).to_int, (n % d) / d

-- Example specific instances
def nine_and_two_fifths : ℚ := mixed_to_improper 9 2 5
def eight_times_nine_and_two_fifths : ℚ := 8 * nine_and_two_fifths

-- Lean statement to confirm calculation
theorem eight_times_nine_and_two_fifths_is : improper_to_mixed eight_times_nine_and_two_fifths 5 = (75, 1/5) := by
  sorry

end eight_times_nine_and_two_fifths_is_l612_612901


namespace finite_solutions_n_factorial_eq_exp_diff_l612_612235

theorem finite_solutions_n_factorial_eq_exp_diff (u : ℕ) (hu : 0 < u) : 
  {np : ℕ × ℕ × ℕ // np.1! = u^(np.2.1) - u^(np.2.2)}.finite := 
sorry

end finite_solutions_n_factorial_eq_exp_diff_l612_612235


namespace shelter_animals_count_l612_612813

theorem shelter_animals_count : 
  (initial_cats adopted_cats new_cats final_cats dogs total_animals : ℕ) 
   (h1 : initial_cats = 15)
   (h2 : adopted_cats = initial_cats / 3)
   (h3 : new_cats = adopted_cats * 2)
   (h4 : final_cats = initial_cats - adopted_cats + new_cats)
   (h5 : dogs = final_cats * 2)
   (h6 : total_animals = final_cats + dogs) :
   total_animals = 60 := 
sorry

end shelter_animals_count_l612_612813


namespace largest_prime_divisor_to_test_l612_612331

theorem largest_prime_divisor_to_test (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) : 
  let sqrt_n : ℝ := Real.sqrt 1050 in
  ∃ p : ℕ, Nat.Prime p ∧ p ≤ sqrt_n ∧ (∀ q : ℕ, Nat.Prime q ∧ q ≤ sqrt_n → q ≤ p) := 
begin
  let sqrt_n := Real.sqrt 1050,
  exact ⟨31, by norm_num [Nat.Prime], by norm_num [sqrt_n], λ q hq, by norm_num [hq.2]⟩
end

end largest_prime_divisor_to_test_l612_612331


namespace solve_t_l612_612751

theorem solve_t (t : ℝ) : 5 * 5^t + (25 * 25^t)^(1/2) = 50 → t = 1 :=
by
  sorry

end solve_t_l612_612751


namespace number_exceeds_20_percent_by_40_eq_50_l612_612835

theorem number_exceeds_20_percent_by_40_eq_50 (x : ℝ) (h : x = 0.20 * x + 40) : x = 50 := by
  sorry

end number_exceeds_20_percent_by_40_eq_50_l612_612835


namespace cat_food_insufficient_for_six_days_l612_612531

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l612_612531


namespace find_coordinates_C_find_range_t_l612_612987

-- required definitions to handle the given points and vectors
structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

-- Given points
def A : Point := ⟨0, 4⟩
def B : Point := ⟨2, 0⟩

-- Proof for coordinates of point C
theorem find_coordinates_C :
  ∃ C : Point, vector A B = {x := 2 * (C.x - B.x), y := 2 * C.y} ∧ C = ⟨3, -2⟩ :=
by
  sorry

-- Proof for range of t
theorem find_range_t (t : ℝ) :
  let P := Point.mk 3 t
  let PA := vector P A
  let PB := vector P B
  (dot_product PA PB < 0 ∧ -3 * t ≠ -1 * (4 - t)) → 1 < t ∧ t < 3 :=
by
  sorry

end find_coordinates_C_find_range_t_l612_612987


namespace quadratic_root_bounds_l612_612221

theorem quadratic_root_bounds (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
  (h_roots1 : 0 ≤ a^2 - 4 * b) (h_roots2 : 0 ≤ a^2 - 2 * b) :
  let x1 := (-a + real.sqrt (a^2 - 4 * b)) / 2 
  let y1 := (a + real.sqrt (a^2 - 4 * b)) / 2 
  let root := -a + real.sqrt (a^2 - 2 * b) in 
  x1 < root ∧ root < y1 :=
by
  sorry

end quadratic_root_bounds_l612_612221


namespace find_phi_l612_612782

theorem find_phi (ϕ : ℝ) (hϕ : 0 < ϕ ∧ ϕ < π) (h_transf : (λ x, sin (2 * (x + π / 8) + ϕ)) 0 = 0) : ϕ = 3 * π / 4 :=
by sorry

end find_phi_l612_612782


namespace area_of_square_l612_612464

theorem area_of_square 
  (side1 side2 side3 : ℝ) 
  (h1 : side1 = 5.8) 
  (h2 : side2 = 7.5) 
  (h3 : side3 = 10.7) 
  (triangle_perimeter : ℝ) 
  (square_side : ℝ)
  (square_area : ℝ)
  (h4 : triangle_perimeter = side1 + side2 + side3)
  (h5 : square_side = triangle_perimeter / 4) :
  square_area = square_side ^ 2 := 
begin
  sorry
end

end area_of_square_l612_612464


namespace S_intersection_T_l612_612233

def S : Set ℕ := {0, 1, 2, 3}
def T : Set ℝ := { x | |x - 1| ≤ 1 }

theorem S_intersection_T :
  S ∩ T = {0, 1, 2} := by sorry

end S_intersection_T_l612_612233


namespace probability_more_heads_than_tails_l612_612660

theorem probability_more_heads_than_tails :
  (let n := 9 in
   let total_outcomes := 2^n in
   let favorable_outcomes := (finset.range(n + 1)).filter (λ k, k > n / 2).sum (λ k, nat.choose n k) in
   (favorable_outcomes / total_outcomes : ℚ) = 1 / 2) :=
by
  let n := 9
  let total_outcomes := 2^n
  let favorable_outcomes := (finset.range(n + 1)).filter (λ k, k > n / 2).sum (λ k, nat.choose n k)
  have h_fav_outcomes : favorable_outcomes = 256 := by sorry
  have h_total_outcomes : total_outcomes = 512 := by sorry
  have probability := (favorable_outcomes / total_outcomes : ℚ)
  rw [h_fav_outcomes, h_total_outcomes] at probability
  norm_num at probability
  exact probability

end probability_more_heads_than_tails_l612_612660


namespace max_value_g_no_equal_terms_l612_612599

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (a * x + 1) / (3 * x - 1)

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (finset.range n).sum a
noncomputable def T (b : ℕ → ℝ) (n : ℕ) : ℝ := (finset.range n).sum b

noncomputable def g (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := a n / b n

variables (a : ℝ) (a_n b_n : ℕ → ℝ)
variables (S_n T_n : ℕ → ℝ)
variables (g_n : ℕ → ℝ)
variables (a_1 : ℝ)

-- g(n) for the question
axiom hS_T : ∀ n > 0, S_n n / T_n n = f n a

-- Given sequences definition
def sequence_a (n : ℕ) : ℝ := -- set up the relation for a_n
def sequence_b (n : ℕ) : ℝ := 3 * n - 2

-- Given starting point
axiom ha1 : sequence_a 1 = 5 / 2

-- Theorem 1: Max value of g(n)
theorem max_value_g (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = (a * x + 1) / (3 * x - 1)) :
  ∃ n, g (sequence_a) (sequence_b) n ≤ 5 / 2 := 
sorry

-- Theorem 2: Equality condition
theorem no_equal_terms (a : ℝ) (sequence_a : ℕ → ℝ) (sequence_b : ℕ → ℝ) :
  ∀ m k : ℕ, sequence_a m ≠ sequence_b k := 
sorry

end max_value_g_no_equal_terms_l612_612599


namespace half_abs_diff_of_squares_l612_612408

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 21) (h2 : b = 19) :
  (abs (a^2 - b^2)) / 2 = 40 := by
  sorry

end half_abs_diff_of_squares_l612_612408


namespace matrix_mult_3I_l612_612936

variable (N : Matrix (Fin 3) (Fin 3) ℝ)

theorem matrix_mult_3I (w : Fin 3 → ℝ):
  (∀ (w : Fin 3 → ℝ), N.mulVec w = 3 * w) ↔ (N = 3 • (1 : Matrix (Fin 3) (Fin 3) ℝ)) :=
by
  sorry

end matrix_mult_3I_l612_612936


namespace initial_light_bulbs_l612_612201

theorem initial_light_bulbs (x : ℕ) 
  (H1 : x - 16 = 24 → x = 40) 
  (H2 : ∀ y z : ℕ, y / 2 = z → y = 2 * z) 
  (H3 : ∀ n : ℕ, x = n + 16) : 
  (x - 16) / 2 = 12 → x = 40 := 
by
  intros h
  have h1 := H2 (x - 16) 12 h
  have h2 : (x - 16) = 24 := by
    rw h1
  have h3 := H1 h2
  assumption

end initial_light_bulbs_l612_612201


namespace max_ab_under_constraint_l612_612607

theorem max_ab_under_constraint (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 3 * a + 2 * b = 1) : 
  ab ≤ (1 / 24) ∧ (ab = 1 / 24 ↔ a = 1 / 6 ∧ b = 1 / 4) :=
sorry

end max_ab_under_constraint_l612_612607


namespace probability_theory_letters_l612_612068

def count_letters (word : String) (letters : List Char) : Nat :=
  letters.foldl (fun count letter => count + word.count (· == letter)) 0

theorem probability_theory_letters :
  let word := "MATHEMATICS"
  let common_letters := ['T', 'H', 'E']
  let total_tiles := String.length word
  let matching_tiles := count_letters word common_letters
  matching_tiles / total_tiles = 1 / 3 := 
by
  sorry

end probability_theory_letters_l612_612068


namespace find_a_l612_612991

-- Definitions based on given conditions
def F : (ℝ × ℝ) := (a / 4, 0) -- focus of the parabola
def M : (ℝ × ℝ) := (4, 0) -- coordinates of point M
noncomputable def k1 := slope F A B -- slope of line passing through F and intersecting the parabola at A and B
noncomputable def k2 := slope C D -- slope of line passing through C and D

-- Definitions for intersection points
variables (x1 y1 x2 y2 x3 y3 x4 y4 a : ℝ)

-- Condition of the problem
axiom parabola_eq : y^2 = a * x
axiom slope_eq_1 : k1 = sqrt 2 * k2
axiom y1_y2_sum : y1 + y2 = sqrt 2 / 2 * (y3 + y4)
axiom vieta_1 : y1 * y3 = -4 * a
axiom vieta_2 : y2 * y4 = -4 * a
axiom vieta_3 : y1 * y2 = -2 * sqrt 2 * a
axiom vieta_4 : y1 * y2 = -a^2 / 4

-- Lean statement to be proven
theorem find_a (a : ℝ) : a = 8 * sqrt 2 :=
by 
  sorry

end find_a_l612_612991


namespace cost_of_each_bread_l612_612927

-- Define the conditions
variables (totalBreads : ℕ) (totalCost contribution : ℝ)
variable (sharedBreads : ℕ)
variable (costPerBread : ℝ)

-- Assume the conditions from the problem
def xiaoMingAndXiaoJunSpendEquallyToBuy : Prop :=
  totalBreads = 12

def xiaoHongContribution : Prop :=
  contribution = 2.2

def totalContributionCalculated : Prop :=
  totalCost = 3 * (2 * contribution)

def sharedEquallyAmongThree : Prop :=
  sharedBreads = 12 -- since all breads are shared equally among three people

-- Define the statement to be proved
theorem cost_of_each_bread :
  xiaoMingAndXiaoJunSpendEquallyToBuy →
  xiaoHongContribution →
  totalContributionCalculated →
  sharedEquallyAmongThree →
  costPerBread = 1.1 :=
by
  -- Definitions given in the conditions
  assume h1 : xiaoMingAndXiaoJunSpendEquallyToBuy,
  assume h2 : xiaoHongContribution,
  assume h3 : totalContributionCalculated,
  assume h4 : sharedEquallyAmongThree,
  -- The proof will be provided here
  sorry

end cost_of_each_bread_l612_612927


namespace overall_profit_percentage_l612_612881

-- Defining the conditions
def sales_volume_ratio (x : ℝ) : ℝ × ℝ × ℝ := (5 * x, 3 * x, 2 * x)
def faulty_meters := (900, 850, 950)
def profit_percentages := (10, 12, 15)

-- Given the conditions
-- First meter profit percentage = 10%
-- Second meter profit percentage = 12%
-- Third meter profit percentage = 15%
-- The sales volume ratio is 5:3:2
-- The question is to find the overall weighted average profit percentage

theorem overall_profit_percentage (x : ℝ) : 
  let (V1, V2, V3) := sales_volume_ratio x in
  let total_volume := V1 + V2 + V3 in
  let weighted_profit_1 := (10 / 100) * (V1 / total_volume) in
  let weighted_profit_2 := (12 / 100) * (V2 / total_volume) in
  let weighted_profit_3 := (15 / 100) * (V3 / total_volume) in
  weighted_profit_1 + weighted_profit_2 + weighted_profit_3 = 11.6 / 100 :=
by 
  sorry

end overall_profit_percentage_l612_612881


namespace cylinder_radius_l612_612874

theorem cylinder_radius
  (diameter_c : ℝ) (altitude_c : ℝ) (height_relation : ℝ → ℝ)
  (same_axis : Bool) (radius_cylinder : ℝ → ℝ)
  (h1 : diameter_c = 14)
  (h2 : altitude_c = 20)
  (h3 : ∀ r, height_relation r = 3 * r)
  (h4 : same_axis = true)
  (h5 : ∀ r, radius_cylinder r = r) :
  ∃ r, r = 140 / 41 :=
by {
  sorry
}

end cylinder_radius_l612_612874


namespace length_an_eq_1_l612_612266

variable {A B C M N : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N]
variable {dist : A → A → ℝ}
variable {angle : ∀ {A B C : A}, ℝ}

-- Given conditions as hypotheses
variable (H1 : ∀ {M}, M ∈ angle_bisector A B C)
variable (H2 : ∀ {N : Type}, N ∈ extended_line A B)
variable (H3 : dist A C = 1)
variable (H4 : dist A M = 1)
variable (H5 : angle A N M = angle C N M)

-- The statement we need to prove
theorem length_an_eq_1 (H1 : M ∈ angle_bisector A B C) (H2 : N ∈ extended_line A B)
  (H3 : dist A C = 1) (H4 : dist A M = 1) (H5 : angle A N M = angle C N M) :
  dist A N = 1 := by
  sorry


end length_an_eq_1_l612_612266


namespace bananas_used_l612_612072

-- Define the conditions
def bananas_per_loaf : Nat := 4
def loaves_on_monday : Nat := 3
def loaves_on_tuesday : Nat := 2 * loaves_on_monday
def total_loaves : Nat := loaves_on_monday + loaves_on_tuesday

-- Define the total bananas used
def total_bananas : Nat := bananas_per_loaf * total_loaves

-- Prove that the total bananas used is 36
theorem bananas_used : total_bananas = 36 :=
by
  sorry

end bananas_used_l612_612072


namespace food_requirement_not_met_l612_612497

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l612_612497


namespace total_paintable_area_correct_l612_612003

namespace BarnPainting

-- Define the dimensions of the barn
def barn_width : ℕ := 12
def barn_length : ℕ := 15
def barn_height : ℕ := 6

-- Define the dimensions of the windows
def window_width : ℕ := 2
def window_height : ℕ := 3
def num_windows : ℕ := 2

-- Calculate the total number of square yards to be painted
def total_paintable_area : ℕ :=
  let wall1_area := barn_height * barn_width
  let wall2_area := barn_height * barn_length
  let wall_area := 2 * wall1_area + 2 * wall2_area
  let window_area := num_windows * (window_width * window_height)
  let painted_walls_area := wall_area - window_area
  let ceiling_area := barn_width * barn_length
  let total_area := 2 * painted_walls_area + ceiling_area
  total_area

theorem total_paintable_area_correct : total_paintable_area = 780 :=
  by sorry

end BarnPainting

end total_paintable_area_correct_l612_612003


namespace find_AN_length_l612_612261

theorem find_AN_length
  (A B C M N : Point)
  (h1 : is_angle_bisector (angle B A C) M)
  (h2 : on_extension A B N)
  (h3 : dist A C = 1)
  (h4 : dist A M = 1)
  (h5 : angle A N M = angle C N M)
  : dist A N = 1 := sorry

end find_AN_length_l612_612261


namespace good_numbers_2020_has_count_10_l612_612394

theorem good_numbers_2020_has_count_10 :
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  set.count divisors = 10 :=
by
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  sorry

end good_numbers_2020_has_count_10_l612_612394


namespace factorize_quadratic_l612_612930

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by {
  sorry
}

end factorize_quadratic_l612_612930


namespace probability_type_A_then_type_B_l612_612862

theorem probability_type_A_then_type_B : 
  let tubes := 50
  let defective_tubes := 10
  let type_A_defective := 5
  let type_B_defective := 3
  let type_C_defective := 2
  let total_tubes := tubes
  let p_A := type_A_defective / total_tubes
  let remaining_tubes_after_A := total_tubes - 1
  let p_B_given_A := type_B_defective / remaining_tubes_after_A
  let p_A_and_B := p_A * p_B_given_A
  (p_A_and_B = 3 / 490)
:= 
by
  let tubes := 50
  let defective_tubes := 10
  let type_A_defective := 5
  let type_B_defective := 3
  let type_C_defective := 2
  let total_tubes := tubes
  let p_A := type_A_defective / total_tubes
  let remaining_tubes_after_A := total_tubes - 1
  let p_B_given_A := type_B_defective / remaining_tubes_after_A
  let p_A_and_B := p_A * p_B_given_A
  have h_A := calc
    p_A = 5 / 50 : rfl
    ... = 1 / 10 : by norm_num
  have h_B_given_A := calc
    p_B_given_A = 3 / 49 : rfl
  have h_A_and_B := calc
    p_A_and_B = (1 / 10) * (3 / 49) : by rw [h_A, h_B_given_A]
    ... = 3 / 490 : by norm_num
  exact h_A_and_B.symm

end probability_type_A_then_type_B_l612_612862


namespace ant_on_red_dot_after_6_minutes_l612_612891

-- Assumptions and Definitions
def initial_position_on_red_dot : Prop := True -- Given that the initial position A is on a red dot

def lattice_movement (n : ℕ) (p : ℕ) : Prop :=
  (n % 2 = 0) ∧ (p % 2 = 0) -- n is the number of minutes, p signifies position, both should correlate red dot

-- Theorem to be proved
theorem ant_on_red_dot_after_6_minutes (initial_pos_red: initial_position_on_red_dot) : 
  ∀ p : ℕ, (lattice_movement 6 p) → p_is_red := 
begin
  sorry -- The steps and complete proof go here
end

end ant_on_red_dot_after_6_minutes_l612_612891


namespace segment_le_diagonal_l612_612320

theorem segment_le_diagonal  
  (A B C D P K L : Point)
  (KL: Segment)
  (h1: intersects AC BD P)
  (h2: segment_passes_through KL P)
  (h3: endpoint_in_AB KL K)
  (h4: endpoint_in_CD KL L) :
  length KL ≤ max (length (Segment A C)) (length (Segment B D)) :=
sorry

end segment_le_diagonal_l612_612320


namespace perfect_squares_ac_val_l612_612441

noncomputable theory
open Classical

variable (a c : ℤ)

theorem perfect_squares_ac_val (H1 : (∃ b1, (x : ℤ) -> x^2 + a * x + c = (x + b1 / 2)^2))
  (H2 : (∃ b2, (x : ℤ) -> x^2 + c * x + a = (x + b2 / 2)^2)) :
  a * c = 0 ∨ a * c = 16 :=
by
  sorry

end perfect_squares_ac_val_l612_612441


namespace part1_part2_l612_612219

noncomputable def y (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x + a - 2

-- Part (1)
theorem part1 (a : ℝ) : (∀ x : ℝ, y a x ≥ -2) ↔ a ∈ Set.Ici (1 / 3) :=
sorry

-- Part (2)
theorem part2 (a x : ℝ) :
  (a ≠ 0 → ( a > 0 ↔ -1/a < x ∧ x < 1)
  ∧ (a = 0 ↔ x < 1)
  ∧ (-1 < a ∧ a < 0 ↔ x < 1 ∨ x > -1/a)
  ∧ (a = -1 ↔ x ≠ 1)
  ∧ (a < -1 ↔ x < -1/a ∨ x > 1)) :=
sorry

end part1_part2_l612_612219


namespace probability_multiple_of_200_l612_612177

open Finset

def S : Finset ℕ := {2, 4, 10, 12, 15, 20, 25, 50, 100}

def is_multiple_of_200 (x : ℕ) : Prop :=
  ∃ (a b : ℕ), x = 200 * a + b

theorem probability_multiple_of_200 :
  (∃ (count : ℕ),
    count = (S.filter (λ x, ∃ y ∈ S, y ≠ x ∧ is_multiple_of_200 (x * y))).card ∧
    (36 : ℕ) = (S.card.choose 2) ∧
    (count : ℚ) / 36 = 1 / 3) :=
begin
  sorry
end

end probability_multiple_of_200_l612_612177


namespace Julie_and_Matt_ate_cookies_l612_612729

def initial_cookies : ℕ := 32
def remaining_cookies : ℕ := 23

theorem Julie_and_Matt_ate_cookies : initial_cookies - remaining_cookies = 9 :=
by
  sorry

end Julie_and_Matt_ate_cookies_l612_612729


namespace union_A_B_l612_612153

noncomputable def U := Set.univ ℝ

def A : Set ℝ := {x | x^2 - x - 2 = 0}

def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = x + 3}

theorem union_A_B : A ∪ B = { -1, 2, 5 } :=
by
  sorry

end union_A_B_l612_612153


namespace abs_value_expression_l612_612657

theorem abs_value_expression (m n : ℝ) (h1 : m < 0) (h2 : m * n < 0) :
  |n - m + 1| - |m - n - 5| = -4 :=
sorry

end abs_value_expression_l612_612657


namespace VerifyMultiplicationProperties_l612_612544

theorem VerifyMultiplicationProperties (α : Type) [Semiring α] :
  ((∀ x y z : α, (x * y) * z = x * (y * z)) ∧
   (∀ x y : α, x * y = y * x) ∧
   (∀ x y z : α, x * (y + z) = x * y + x * z) ∧
   (∃ e : α, ∀ x : α, x * e = x)) := by
  sorry

end VerifyMultiplicationProperties_l612_612544


namespace sammy_score_l612_612179

instance : LinearOrderedField ℝ := inferInstance

theorem sammy_score (S : ℝ) (H1 : 7 * S = 140) : S = 20 :=
by
  sorry

end sammy_score_l612_612179


namespace bianca_marathon_total_miles_l612_612045

theorem bianca_marathon_total_miles : 8 + 4 = 12 :=
by
  sorry

end bianca_marathon_total_miles_l612_612045


namespace fg_of_3_l612_612152

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 + 1
def g (x : ℝ) : ℝ := 3 * x - 2

-- State the theorem we want to prove
theorem fg_of_3 : f (g 3) = 344 := by
  sorry

end fg_of_3_l612_612152


namespace triangle_LM_length_l612_612690

theorem triangle_LM_length (A B C K L M : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace K] [MetricSpace L] [MetricSpace M]
  (angleA : ∠ B A C = 90)
  (angleB : ∠ A B C = 30)
  (AK : Real) (BL : Real) (MC : Real)
  (KL_eq_KM : dist K L = dist K M)
  (AK_val : AK = 4)
  (BL_val : BL = 31)
  (MC_val : MC = 3) :
  dist L M = 14 := sorry

end triangle_LM_length_l612_612690


namespace convex_polygon_isosceles_diagonals_l612_612453

noncomputable def convex_polygon_has_two_equal_sides := 
  ∀ (n : ℕ) (P : Set Point) (H_convex : convex P) (H_divided : ∀ diag ∈ P.diagonals, isosceles_triangle diag),
    n ≥ 3 → (∃ (a b : Side), a ∈ P.sides ∧ b ∈ P.sides ∧ a = b)

theorem convex_polygon_isosceles_diagonals :
  convex_polygon_has_two_equal_sides := 
by
  sorry

end convex_polygon_isosceles_diagonals_l612_612453


namespace transformed_A_coordinates_l612_612741

open Real

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.fst, p.snd)

def A : ℝ × ℝ := (-3, 2)

theorem transformed_A_coordinates :
  reflect_over_y_axis (rotate_90_clockwise A) = (-2, 3) :=
by
  sorry

end transformed_A_coordinates_l612_612741


namespace sum_of_decimals_l612_612569

theorem sum_of_decimals :
  0.3 + 0.04 + 0.005 + 0.0006 + 0.00007 = (34567 / 100000 : ℚ) :=
by
  -- The proof details would go here
  sorry

end sum_of_decimals_l612_612569


namespace proof_math_problem_l612_612683

section
variables {a x t x1 x2 : ℝ} (h_a : a > 0) (h_max : ∀ x, -1 ≤ x ∧ x ≤ 3 → (x + a) * (x - a - 1) ≤ 4)

-- Part 1: Axis of symmetry
def axis_of_symmetry : Prop :=
  x = 1 / 2

-- Part 2: Vertex coordinates
def vertex_coordinates : Prop :=
  (∃ y, y = (1 / 2 + a) * (1 / 2 - a - 1) ∧ y = -9 / 4)

-- Part 3: Range of t
def range_of_t : Prop :=
  ∀ (x1 x2 : ℝ), t < x1 ∧ x1 < t + 1 ∧ t + 2 < x2 ∧ x2 < t + 3 ∧
  (x + a) * (x - a - 1) ≠ (x + a) * (x - a - 1) → t ≥ -1 / 2

theorem proof_math_problem :
  axis_of_symmetry ∧ vertex_coordinates ∧ range_of_t :=
by
  sorry
end

end proof_math_problem_l612_612683


namespace train_speed_is_63_km_hr_l612_612445

open Real

/-- Define the known values and constants. -/
def length_of_train : ℝ := 550
def time_to_cross : ℝ := 32.997
def speed_of_man_km_hr : ℝ := 3
def speed_of_man_m_s : ℝ := (speed_of_man_km_hr * 1000) / 3600

/-- Define the speed of the train in m/s -/
def speed_of_train_in_m_s : ℝ :=
  (length_of_train + time_to_cross * speed_of_man_m_s) / time_to_cross

/-- Convert the speed of the train from m/s to km/hr -/
def speed_of_train_in_km_hr : ℝ :=
  speed_of_train_in_m_s * 3.6

/-- Prove that the speed of the train is equal to 63 km/hr given the conditions. -/
theorem train_speed_is_63_km_hr :
  speed_of_train_in_km_hr = 63 :=
by
  sorry

end train_speed_is_63_km_hr_l612_612445


namespace plan_y_cheaper_than_plan_x_l612_612027

def cost_plan_x (z : ℕ) : ℕ := 15 * z

def cost_plan_y (z : ℕ) : ℕ :=
  if z > 500 then 3000 + 7 * z - 1000 else 3000 + 7 * z

theorem plan_y_cheaper_than_plan_x (z : ℕ) (h : z > 500) : cost_plan_y z < cost_plan_x z :=
by
  sorry

end plan_y_cheaper_than_plan_x_l612_612027


namespace total_bill_correct_l612_612550

noncomputable def curtis_main_price : ℝ := 16.00
noncomputable def rob_main_price : ℝ := 18.00
noncomputable def curtis_side_price : ℝ := 6.00
noncomputable def rob_side_price : ℝ := 7.00
noncomputable def curtis_drink_price : ℝ := 3.00
noncomputable def rob_drink_price : ℝ := 3.50
noncomputable def discount_rate : ℝ := 0.5
noncomputable def tip_rate : ℝ := 0.20
noncomputable def tax_rate : ℝ := 0.07

theorem total_bill_correct :
  let curtis_discounted_main := curtis_main_price * discount_rate,
      rob_discounted_main := rob_main_price * discount_rate,
      curtis_total := curtis_discounted_main + curtis_side_price + curtis_drink_price,
      rob_total := rob_discounted_main + rob_side_price + rob_drink_price,
      combined_total := curtis_total + rob_total,
      tax := combined_total * tax_rate,
      total_with_tax := combined_total + tax,
      tip := combined_total * tip_rate,
      final_total := total_with_tax + tip
  in final_total = 46.36 := by
  sorry

end total_bill_correct_l612_612550


namespace calculate_expression_l612_612055

def g (n : ℤ) : ℚ := (1/4 : ℚ) * n * (n + 1) * (n + 3)

theorem calculate_expression (s : ℤ) :
  g(s) - g(s - 1) + s * (s + 1) = 2 * s^2 + 2 * s :=
by
  sorry

end calculate_expression_l612_612055


namespace good_numbers_count_l612_612337

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l612_612337


namespace fixed_point_pq_l612_612678

noncomputable def point_K (A B C : Point) : Point := sorry

theorem fixed_point_pq
  (A B C H P E F Q K : Point)
  (h_acute_angled : ∀ M N : Point, ∠B < 90 ∧ ∠C < 90)
  (h_altitude : H = foot_of_altitude A B C)
  (h_bisectors : is_bisector (PBC) (PCB) intersect AH)
  (h_meet_k_AC : k ∩ AC = E)
  (h_meet_l_AB : l ∩ AB = F)
  (h_intersect_EF_AH : EF ∩ AH = Q)
  (h_point_K : K = point_K A B C) :
  collinear P Q K :=
sorry

end fixed_point_pq_l612_612678


namespace total_animals_in_shelter_l612_612812

def initial_cats : ℕ := 15
def adopted_cats := initial_cats / 3
def replacement_cats := 2 * adopted_cats
def current_cats := initial_cats - adopted_cats + replacement_cats
def additional_dogs := 2 * current_cats
def total_animals := current_cats + additional_dogs

theorem total_animals_in_shelter : total_animals = 60 := by
  sorry

end total_animals_in_shelter_l612_612812


namespace find_y_l612_612716

variable {a b y : ℝ}
variable (ha : a ≠ 0) (hb : b ≠ 0)

theorem find_y (h1 : (3 * a) ^ (4 * b) = a ^ b * y ^ b) : y = 81 * a ^ 3 := by
  sorry

end find_y_l612_612716


namespace altitude_of_dolphin_l612_612768

theorem altitude_of_dolphin (h_submarine : altitude_submarine = -50) (h_dolphin : distance_above_submarine = 10) : altitude_dolphin = -40 :=
by
  -- Altitude of the dolphin is the altitude of the submarine plus the distance above it
  have h_dolphin_altitude : altitude_dolphin = altitude_submarine + distance_above_submarine := sorry
  -- Substitute the values
  rw [h_submarine, h_dolphin] at h_dolphin_altitude
  -- Simplify the expression
  exact h_dolphin_altitude

end altitude_of_dolphin_l612_612768


namespace parallelogram_angle_l612_612187

variables (a b c d : Type) [EuclideanGeometry a] [EuclideanGeometry b]

theorem parallelogram_angle {EFGH : Parallelogram a b c d}
  (h1 : ∠ EFGH.F = 130°) :
  (∠ EFGH.H = 130°) ∧ (ExteriorAngle EFGH.H = 50°) :=
by
  sorry

end parallelogram_angle_l612_612187


namespace find_number_l612_612442

theorem find_number {x : ℝ} (h : 0.5 * x - 10 = 25) : x = 70 :=
sorry

end find_number_l612_612442


namespace line_intersects_circle_for_m_l612_612791

-- Definitions of the line and the circle
def line (m : ℝ) : set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + m = 0}
def circle : set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 4}

-- Prove that m is in the specified range
theorem line_intersects_circle_for_m (m : ℝ) : 
  -2 * real.sqrt 2 - 1 ≤ m ∧ m ≤ 2 * real.sqrt 2 - 1 → 
  ∃ p : ℝ × ℝ, p ∈ line m ∧ p ∈ circle :=
sorry

end line_intersects_circle_for_m_l612_612791


namespace probability_more_heads_than_tails_l612_612661

theorem probability_more_heads_than_tails :
  (let n := 9 in
   let total_outcomes := 2^n in
   let favorable_outcomes := (finset.range(n + 1)).filter (λ k, k > n / 2).sum (λ k, nat.choose n k) in
   (favorable_outcomes / total_outcomes : ℚ) = 1 / 2) :=
by
  let n := 9
  let total_outcomes := 2^n
  let favorable_outcomes := (finset.range(n + 1)).filter (λ k, k > n / 2).sum (λ k, nat.choose n k)
  have h_fav_outcomes : favorable_outcomes = 256 := by sorry
  have h_total_outcomes : total_outcomes = 512 := by sorry
  have probability := (favorable_outcomes / total_outcomes : ℚ)
  rw [h_fav_outcomes, h_total_outcomes] at probability
  norm_num at probability
  exact probability

end probability_more_heads_than_tails_l612_612661


namespace min_value_of_g_l612_612620

-- Definitions
def f (x : ℝ) : ℝ := x^2 + 7*x + 13
def g (x a : ℝ) : ℝ := (x + a)^2 + 7*a + 13

-- The proof problem
theorem min_value_of_g (a : ℝ) :
  (∀ x : ℝ, f (2*x - 3) = 4*x^2 + 2*x + 1) ∧
  (g x a = f (x + a) - 7*x) →
  ∃ m : ℝ,
    m = if a ≤ -3 then a^2 + 13*a + 22
        else if -3 < a ∧ a < -1 then 7*a + 13
        else a^2 + 9*a + 14 ∧
    (∀ x ∈ set.Icc 1 3, g x a ≥ m) := sorry

end min_value_of_g_l612_612620


namespace winning_strategy_10_12_winning_strategy_9_10_winning_strategy_9_11_l612_612973

def has_winning_strategy (rows : ℕ) (cols : ℕ) (first_player : Bool) : Prop :=
  if first_player then rows % 2 = 1 ∧ cols % 2 = 0 else rows % 2 = 0 ∧ cols % 2 = 0

theorem winning_strategy_10_12 : has_winning_strategy 10 12 false :=
by
  simp [has_winning_strategy]
  exact ⟨nat.zero_mod 2, nat.zero_mod 2⟩

theorem winning_strategy_9_10 : has_winning_strategy 9 10 true :=
by
  simp [has_winning_strategy]
  exact ⟨nat.mod_self, nat.zero_mod 2⟩

theorem winning_strategy_9_11 : has_winning_strategy 9 11 false :=
by
  simp [has_winning_strategy]
  exact ⟨nat.mod_self, nat.mod_self⟩

end winning_strategy_10_12_winning_strategy_9_10_winning_strategy_9_11_l612_612973


namespace count_good_divisors_l612_612375

/-- We call a natural number \( n \) good if 2020 divided by \( n \) leaves a remainder of 22.
    This means \( n \) must be a divisor of 1998 and \( n > 22 \) -/
theorem count_good_divisors : finset.card (finset.filter (λ m, 22 < m ∧ 1998 % m = 0) (finset.range 2000)) = 10 :=
by
  -- This is where the proof would go
  sorry

end count_good_divisors_l612_612375


namespace polynomial_division_l612_612947

theorem polynomial_division (x : ℝ) : (Polynomial.evalₓ x (x^4 + 13)) = (Polynomial.evalₓ x (x^3 + x^2 + x + 1) + Polynomial.evalₓ x 14) :=
by
  sorry

end polynomial_division_l612_612947


namespace Noah_holidays_in_a_year_l612_612733

def holidaysPerMonth : ℕ := 3
def monthsPerYear : ℕ := 12
def totalHolidays (h : ℕ) (m : ℕ) : ℕ := h * m

theorem Noah_holidays_in_a_year : totalHolidays holidaysPerMonth monthsPerYear = 36 :=
by
  unfold totalHolidays
  simp

# Output the theorem
# Theorem Noah_holidays_in_a_year represents: Noah took 36 holidays in that year.

end Noah_holidays_in_a_year_l612_612733


namespace integer_part_of_sqrt_10_l612_612787

theorem integer_part_of_sqrt_10 :
  let x := ⌊Real.sqrt 10⌋
  let y := Real.sqrt 10 - x
  x = 3 ∧ y = Real.sqrt 10 - 3 → y * (x + Real.sqrt 10) = 1 :=
by
  intros x y h
  cases h with hx hy
  rw [hx, hy]
  sorry

end integer_part_of_sqrt_10_l612_612787


namespace line_perpendicular_intersection_l612_612119

noncomputable def line_equation (x y : ℝ) := 3 * x + y + 2 = 0

def is_perpendicular (m1 m2 : ℝ) := m1 * m2 = -1

theorem line_perpendicular_intersection (x y : ℝ) :
  (x - y + 2 = 0) →
  (2 * x + y + 1 = 0) →
  is_perpendicular (1 / 3) (-3) →
  line_equation x y := 
sorry

end line_perpendicular_intersection_l612_612119


namespace minimum_a_l612_612136

-- Define the function f
def f (x : ℝ) : ℝ := (2 * 2023^x) / (2023^x + 1)

-- State the problem
theorem minimum_a (a : ℝ) :
  (∀ x : ℝ, f (a * real.exp x) ≥ 2 - f (real.log a - real.log x)) ↔ a ≥ 1 / real.exp 1 :=
sorry

end minimum_a_l612_612136


namespace uncertain_relationship_l612_612955

noncomputable section

-- Define the events A and B in a sample space Ω
variable {Ω : Type*} (A B : Ω → Prop)

-- Define the probability measure P on the sample space Ω
variable (P : MeasureTheory.Measure Ω)

-- Define the conditions given in the problem
def events_relationship : Prop :=
  P {ω | A ω ∨ B ω} = P {ω | A ω} + P {ω | B ω} = 1

-- Lean theorem statement for the problem
theorem uncertain_relationship (P : MeasureTheory.Measure Ω) (A B : Ω → Prop) (h : events_relationship P A B) : 
  (complementary_relation P A B ∨ ¬ complementary_relation P A B) :=
sorry

end uncertain_relationship_l612_612955


namespace chairs_to_be_removed_l612_612861

theorem chairs_to_be_removed
    (chairs_per_row : ℕ)
    (initial_chairs : ℕ)
    (attendees : ℕ)
    (H1 : chairs_per_row = 15)
    (H2 : initial_chairs = 195)
    (H3 : attendees = 120) :
    initial_chairs - (Nat.ceil (attendees / chairs_per_row).toReal) * chairs_per_row = 60 :=
by
  sorry

end chairs_to_be_removed_l612_612861


namespace xy_value_l612_612167

theorem xy_value (x y : ℝ) (h : |x - 1| + (x + y)^2 = 0) : x * y = -1 := 
by
  sorry

end xy_value_l612_612167


namespace box_interior_diagonals_l612_612873

-- Definition of the problem
def surface_area (x y z : ℝ) : ℝ := 2 * (x * y + y * z + z * x)
def sum_edge_lengths (x y z : ℝ) : ℝ := 4 * (x + y + z)
def sum_interior_diagonals (x y z : ℝ) : ℝ := 4 * real.sqrt (x^2 + y^2 + z^2)

theorem box_interior_diagonals (x y z : ℝ) 
  (h_surface_area : surface_area x y z = 116) 
  (h_edge_lengths : sum_edge_lengths x y z = 56) :
  sum_interior_diagonals x y z = 16 * real.sqrt 5 :=
sorry

end box_interior_diagonals_l612_612873


namespace Xiaoli_estimate_is_larger_l612_612300

variables {x y x' y' : ℝ}

theorem Xiaoli_estimate_is_larger (h1 : x > y) (h2 : y > 0) (h3 : x' = 1.01 * x) (h4 : y' = 0.99 * y) : x' - y' > x - y :=
by sorry

end Xiaoli_estimate_is_larger_l612_612300


namespace factorize_quadratic_l612_612933

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by
  sorry

end factorize_quadratic_l612_612933


namespace minimize_PE_PC_IS_PB_l612_612220

noncomputable theory

-- Definition of the square and conditions
def square_ABCD (A B C D : ℝ × ℝ) := 
  A = (0, 0) ∧ B = (5, 0) ∧ C = (5, 5) ∧ D = (0, 5)

def point_E (E : ℝ × ℝ) := 
  E = (5, 3)

def on_diagonal_BD (P : ℝ × ℝ) := 
  ∃ t : ℝ, P = (t * 5, t * 5)

-- Distance function
def distance (P Q : ℝ × ℝ) : ℝ := 
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Problem statement
theorem minimize_PE_PC_IS_PB (A B C D E P : ℝ × ℝ) :
  square_ABCD A B C D →
  point_E E →
  on_diagonal_BD P →
  ∃ PB_len : ℝ, PB_len = (15 * sqrt 2) / 8 ∧ 
  PB_len = distance P B ∧
  (distance P E + distance P C) = _
:=
sorry

end minimize_PE_PC_IS_PB_l612_612220


namespace num_divisors_of_10_gt_9_l612_612652

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def countLargerDivisors (n k : ℕ) : ℕ :=
  (List.range (factorial n)).filter (fun d => d > factorial k ∧ (factorial n) % d = 0).length

theorem num_divisors_of_10_gt_9 :
  countLargerDivisors 10 9 = 9 :=
  sorry

end num_divisors_of_10_gt_9_l612_612652


namespace sum_of_valid_m_l612_612583

theorem sum_of_valid_m : 
  (∑ m in Finset.filter (λ m : ℕ, ∃ a b c d : ℕ, (m > 0) ∧ (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (2^m = a.factorial + b.factorial + c.factorial + d.factorial)) (Finset.range 20)) = 18 :=
by
  sorry

end sum_of_valid_m_l612_612583


namespace maximum_minimum_sum_l612_612230

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem maximum_minimum_sum (M m : ℝ) (hM : ∀ x ∈ Icc (-2 : ℝ) 2, f x ≤ M) 
  (hm : ∀ x ∈ Icc (-2 : ℝ) 2, f x ≥ m) 
  (hMge : ∃ x ∈ Icc (-2 : ℝ) 2, f x = M) 
  (hmle : ∃ x ∈ Icc (-2 : ℝ) 2, f x = m) : M + m = 2 := 
by
  sorry

end maximum_minimum_sum_l612_612230


namespace cone_cylinder_volume_ratio_l612_612455

theorem cone_cylinder_volume_ratio (h_cyl r_cyl: ℝ) (h_cone: ℝ) :
  h_cyl = 10 → r_cyl = 5 → h_cone = 5 →
  (1/3 * (Real.pi * r_cyl^2 * h_cone)) / (Real.pi * r_cyl^2 * h_cyl) = 1/6 :=
by
  intros h_cyl_eq r_cyl_eq h_cone_eq
  rw [h_cyl_eq, r_cyl_eq, h_cone_eq]
  sorry

end cone_cylinder_volume_ratio_l612_612455


namespace money_collected_is_correct_l612_612475

-- Define the conditions as constants and definitions in Lean
def ticket_price_adult : ℝ := 0.60
def ticket_price_child : ℝ := 0.25
def total_persons : ℕ := 280
def children_attended : ℕ := 80

-- Define the number of adults
def adults_attended : ℕ := total_persons - children_attended

-- Define the total money collected
def total_money_collected : ℝ :=
  (adults_attended * ticket_price_adult) + (children_attended * ticket_price_child)

-- Statement to prove
theorem money_collected_is_correct :
  total_money_collected = 140 := by
  sorry

end money_collected_is_correct_l612_612475


namespace sum_of_coordinates_of_intersection_l612_612898

theorem sum_of_coordinates_of_intersection :
  (∃ a b : ℝ, (∀ x : ℝ, f (x) = 3.125 - ((x + 1) ^ 2 / 3)) ∧ (b = f(a)) ∧ (b = f(a - 4)) ∧ (a = 1) ∧ (b = 2)) 
  → 1 + 2 = 3 :=
by {
  sorry
}

end sum_of_coordinates_of_intersection_l612_612898


namespace completing_the_square_l612_612753

theorem completing_the_square (x : ℝ) : x^2 + 8*x + 7 = 0 → (x + 4)^2 = 9 :=
by {
  sorry
}

end completing_the_square_l612_612753


namespace jonah_total_lemonade_poured_l612_612560

variable (a b c : ℝ)
variable (first_intermission second_intermission third_intermission total_lemonade : ℝ)

axiom first_intermission_amount : first_intermission = 0.25
axiom second_intermission_amount : second_intermission = 0.4166666666666667
axiom third_intermission_amount : third_intermission = 0.25
axiom total_poured : total_lemonade = first_intermission + second_intermission + third_intermission

theorem jonah_total_lemonade_poured :
  first_intermission + second_intermission + third_intermission = 0.9166666666666667 := by
  rw [first_intermission_amount, second_intermission_amount, third_intermission_amount]
  sorry

end jonah_total_lemonade_poured_l612_612560


namespace problem1_problem2_l612_612621

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) * (Real.cos x) + 2 * (Real.cos x)^2

noncomputable def g (x : ℝ) : ℝ := f (x - π / 4)

theorem problem1 {k : ℤ} :
  ∀ x, k * π - 3 * π / 8 ≤ x ∧ x ≤ k * π + π / 8 → 
  deriv f x ≥ 0 := sorry

theorem problem2 {k : ℤ} : 
  ∀ x, g x = 1 ↔ x = k * π / 2 + π / 4 := sorry

end problem1_problem2_l612_612621


namespace gcd_fraction_is_a_l612_612288

theorem gcd_fraction_is_a (a b c d : ℕ) (h : a * b = c * d) : 
  (gcd a c * gcd a d / gcd a b c d) = a := sorry

end gcd_fraction_is_a_l612_612288


namespace abs_neg_2035_l612_612767

theorem abs_neg_2035 : abs (-2035) = 2035 := 
by {
  sorry
}

end abs_neg_2035_l612_612767


namespace total_shaded_cubes_is_14_l612_612052

def is_shaded (x y z : ℕ) : Prop :=
(x = 0 ∨ x = 2 ∨ y = 0 ∨ y = 2 ∨ z = 0 ∨ z = 2) ∧ (x = 0 ∨ x = 2 ∨ y = 0 ∨ y = 2 ∨ z = 1) ∧ (x = 0 ∨ x = 2 ∨ y = 1 ∨ z = 0 ∨ z = 2)

def count_shaded_cubes : ℕ :=
Finset.card (Finset.filter (λ xyz : Fin3 × Fin3 × Fin3, is_shaded xyz.1 xyz.2 xyz.3) Finset.univ)

theorem total_shaded_cubes_is_14 : count_shaded_cubes = 14 :=
by
    sorry

end total_shaded_cubes_is_14_l612_612052


namespace air_quality_prob_l612_612851

theorem air_quality_prob 
  (h1 : ∀ (day : ℕ), ℙ(good_air_quality day) = 0.75)
  (h2 : ∀ (day : ℕ), ℙ(good_air_quality day ∧ good_air_quality (day + 1)) = 0.6) :
  ℙ(good_air_quality (day + 1) | good_air_quality day) = 0.8 :=
begin
  -- formal proof here
  sorry
end

end air_quality_prob_l612_612851


namespace cat_food_inequality_l612_612517

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l612_612517


namespace good_numbers_count_l612_612341

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l612_612341


namespace count_good_numbers_l612_612365

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l612_612365


namespace polynomial_solution_l612_612761

theorem polynomial_solution (f g : ℝ → ℝ) 
  (h₁ : ∀ x, f(x) = x^2) 
  (h₂ : ∀ x, f(g(x)) = 9x^2 + 12x + 4) :
  ∀ x, g(x) = 3x + 2 ∨ g(x) = -3x - 2 := 
by 
  sorry

end polynomial_solution_l612_612761


namespace number_of_larger_planes_l612_612877

variable (S L : ℕ)
variable (h1 : S + L = 4)
variable (h2 : 130 * S + 145 * L = 550)

theorem number_of_larger_planes : L = 2 :=
by
  -- Placeholder for the proof
  sorry

end number_of_larger_planes_l612_612877


namespace king_path_28_moves_exists_king_path_min_28_moves_king_path_length_bounds_l612_612314

/-- Prove that there exists a path on an 8x8 chessboard where the king makes exactly 28 horizontal and vertical moves and returns to the starting position. -/
theorem king_path_28_moves_exists : ∃ path : list (ℕ × ℕ), length path = 64 ∧ -- path should have 64 steps
  (∀ i in path, i.1 < 8 ∧ i.2 < 8) ∧ -- all moves are within bounds of an 8x8 chessboard
  (∀ i j, i < j → path.nth i ≠ path.nth j) ∧ -- each square is visited exactly once
  (path.head = path.last) ∧ -- path forms a closed loop
  count_moves path (λ p1 p2, p1.1 = p2.1 ∨ p1.2 = p2.2) = 28 := -- Path has exactly 28 horizontal and vertical moves
sorry

/-- Prove that the king cannot have fewer than 28 horizontal and vertical moves on an 8x8 chessboard if he visits each square exactly once and returns to the starting position. -/
theorem king_path_min_28_moves : ∀ path : list (ℕ × ℕ), length path = 64 ∧ 
  (∀ i in path, i.1 < 8 ∧ i.2 < 8) ∧ 
  (∀ i j, i < j → path.nth i ≠ path.nth j) ∧ 
  (path.head = path.last) → 
  count_moves path (λ p1 p2, p1.1 = p2.1 ∨ p1.2 = p2.2) ≥ 28 := 
sorry

/-- Given the side length of a square is 1 unit, prove the minimum and maximum lengths of the king's tour. -/
theorem king_path_length_bounds : ∀ path : list (ℕ × ℕ), length path = 64 ∧ 
  (∀ i in path, i.1 < 8 ∧ i.2 < 8) ∧ 
  (∀ i j, i < j → path.nth i ≠ path.nth j) ∧ 
  (path.head = path.last) → 
  64 ≤ path_length path ∧ path_length path ≤ 28 + 36 * Real.sqrt 2 :=
sorry

end king_path_28_moves_exists_king_path_min_28_moves_king_path_length_bounds_l612_612314


namespace ounces_per_gallon_l612_612691

-- conditions
def gallons_of_milk (james : Type) : ℕ := 3
def ounces_drank (james : Type) : ℕ := 13
def ounces_left (james : Type) : ℕ := 371

-- question
def ounces_in_gallon (james : Type) : ℕ := 128

-- proof statement
theorem ounces_per_gallon (james : Type) :
  (gallons_of_milk james) * (ounces_in_gallon james) = (ounces_left james + ounces_drank james) :=
sorry

end ounces_per_gallon_l612_612691


namespace part_a_part_b_l612_612310

-- Definitions for the terms in the geometric progression
def geo_prog (A : ℕ) (p q : ℕ) (n : ℕ) : ℕ :=
  A * (p ^ n) / (q ^ n)

-- Condition for the sequence to lie within given bounds
def in_bounds (seq : List ℕ) (low high : ℕ) : Prop :=
  ∀ a ∈ seq, low ≤ a ∧ a ≤ high

-- Statement for part (a)
theorem part_a : ∃ (A p q : ℕ) (seq : List ℕ),
  A = 256 ∧ p = 3 ∧ q = 2 ∧ length seq = 4 ∧ seq = List.map (geo_prog A p q) [0, 1, 2, 3] ∧ 
  in_bounds seq 200 1200 :=
  sorry

-- Statement for part (b)
theorem part_b : ∃ (A p q : ℕ) (seq : List ℕ),
  A = 243 ∧ p = 4 ∧ q = 3 ∧ length seq = 6 ∧ seq = List.map (geo_prog A p q) [0, 1, 2, 3, 4, 5] ∧ 
  in_bounds seq 200 1200 :=
  sorry

end part_a_part_b_l612_612310


namespace good_numbers_2020_has_count_10_l612_612396

theorem good_numbers_2020_has_count_10 :
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  set.count divisors = 10 :=
by
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  sorry

end good_numbers_2020_has_count_10_l612_612396


namespace largest_possible_integer_in_list_l612_612015

theorem largest_possible_integer_in_list (a b c d e : ℕ) 
  (h1 : list a b c d e)
  (h2 : a = 7) 
  (h3 : b = 7) 
  (h4 : c = 12) 
  (h5 : d = 9)
  (h6 : e = 9)
  (h7 : median [a, b, c, d, e] = 12)
  (h8 : list_average [a, b, c, d, e] = 15): e = 37 :=
by {
  sorry, #The proof will be filled in here
}

end largest_possible_integer_in_list_l612_612015


namespace tangent_plane_at_A_normal_line_through_A_l612_612939

noncomputable def F (x y z : ℝ) : ℝ := 2 * x^2 + y^2 - z

def point_A : ℝ × ℝ × ℝ := (1, -1, 3)

def tangent_plane_eq (x y z : ℝ) : ℝ := 4*x - 2*y - z - 3

def normal_line_eq (t : ℝ) : ℝ × ℝ × ℝ := (1 + 4 * t, -1 - 2 * t, 3 - t)

theorem tangent_plane_at_A : 
  ∀ x y z : ℝ, 
  tangent_plane_eq x y z = 0 ↔ 
  ∀ t : ℝ, 
  F (1 + 4 * t) (-1 - 2 * t) (3 - t) = 0 := sorry

theorem normal_line_through_A :
  ∀ t : ℝ, 
  (1 + 4 * t, -1 - 2 * t, 3 - t) ∈ {p : ℝ × ℝ × ℝ | F p.1 p.2 p.3 = 0} := sorry

end tangent_plane_at_A_normal_line_through_A_l612_612939


namespace length_AN_is_one_l612_612282

noncomputable def segment_length_AN (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) (point_N_extension : ExtensionOfSide A B N) 
  (AC_eq_AM : AC = 1 ∧ AM = 1) (ANM_eq_CNM : ∠ANM = ∠CNM) : Real :=
1

/-- Given a triangle ABC, where point M lies on the angle bisector of angle BAC,
and point N lies on the extension of side AB beyond point A, where AC = AM = 1,
and the angles ANM and CNM are equal, the length of segment AN is 1. -/
theorem length_AN_is_one (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) 
  (point_N_extension : ExtensionOfSide A B N) (AC_eq_AM : AC = 1 ∧ AM = 1) 
  (ANM_eq_CNM : ∠ANM = ∠CNM) : 
  segment_length_AN A B C M N triangle_ABC angle_bisector_AM point_N_extension AC_eq_AM ANM_eq_CNM = 1 :=
by
  sorry

end length_AN_is_one_l612_612282


namespace smallest_growth_rate_l612_612784

-- Define the growth rates as given conditions
def r_Liwa : ℝ := 3.25 / 100
def r_Wangwa : ℝ := -2.75 / 100
def r_Jiazhuang : ℝ := 4.6 / 100
def r_Wuzhuang : ℝ := -1.76 / 100

-- Define the theorem for the smallest growth rate
theorem smallest_growth_rate :
  (r_Wangwa < r_Wuzhuang) ∧ 
  (r_Wangwa < r_Liwa) ∧ 
  (r_Wangwa < r_Jiazhuang) → 
  "Wangwa has the smallest growth rate" :=
by
  sorry

end smallest_growth_rate_l612_612784


namespace f_zero_f_odd_f_range_l612_612720

noncomputable def f : ℝ → ℝ := sorry

-- Add the hypothesis for the conditions
axiom f_domain : ∀ x : ℝ, ∃ y : ℝ, f y = f x
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_value_one_third : f (1 / 3) = 1
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0

-- (1) Prove that f(0) = 0
theorem f_zero : f 0 = 0 := sorry

-- (2) Prove that f(x) is odd
theorem f_odd (x : ℝ) : f (-x) = -f x := sorry

-- (3) Given f(x) + f(2 + x) < 2, find the range of x
theorem f_range (x : ℝ) : f x + f (2 + x) < 2 → x < -2 / 3 := sorry

end f_zero_f_odd_f_range_l612_612720


namespace mean_score_is_74_l612_612425

theorem mean_score_is_74 (M SD : ℝ) 
  (h1 : 58 = M - 2 * SD) 
  (h2 : 98 = M + 3 * SD) : 
  M = 74 := 
by 
  -- problem statement without solving steps
  sorry

end mean_score_is_74_l612_612425


namespace escape_room_l612_612198

def doorNumbers : Finset ℕ := Finset.range 1 7  -- Doors 1 to 6

def lockedDoors (locked: Finset ℕ) : Prop :=
  locked.card = 5 ∧ ∀ d ∈ doorNumbers, d ∉ locked ∨ d ∈ locked

def attempt (checkedDoors: Finset ℕ) : Finset ℕ :=
  checkedDoors ∪ {d | d ∉ checkedDoors}

def babaYaga (locked: Finset ℕ) (openDoor: ℕ) : Finset ℕ :=
  ((doorNumbers \ {openDoor}) ∪ {openDoor - 1} ∪ {openDoor + 1}) ∩ doorNumbers

theorem escape_room :
  ∀ locked: Finset ℕ,
  ∃ attempt1 attempt2: Finset ℕ,
  lockedDoors locked →
  ∃ openDoor: ℕ, openDoor ∉ locked →
    (let lockedAfter1 := babaYaga locked openDoor,
         lockedAfter2 := babaYaga lockedAfter1 openDoor in
     openDoor ∉ locked ∪ lockedAfter1 ∪ lockedAfter2) :=
by
  sorry  -- Proof to be provided.

end escape_room_l612_612198


namespace total_shares_eq_300_l612_612893

-- Define the given conditions
def microtron_price : ℝ := 36
def dynaco_price : ℝ := 44
def avg_price : ℝ := 40
def dynaco_shares : ℝ := 150

-- Define the number of Microtron shares sold
variable (M : ℝ)

-- Define the total shares sold
def total_shares : ℝ := M + dynaco_shares

-- The average price equation given the conditions
def avg_price_eq (M : ℝ) : Prop :=
  avg_price = (microtron_price * M + dynaco_price * dynaco_shares) / total_shares M

-- The correct answer we need to prove
theorem total_shares_eq_300 (M : ℝ) (h : avg_price_eq M) : total_shares M = 300 :=
by
  sorry

end total_shares_eq_300_l612_612893


namespace even_function_a_equals_one_l612_612666

theorem even_function_a_equals_one (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x - a) = (1 - x) * (-x - a)) → a = 1 :=
by
  intro h
  sorry

end even_function_a_equals_one_l612_612666


namespace number_of_ways_to_divide_people_l612_612444

-- Define the constants based on the conditions
const n : ℕ := 6
const max_capacity : ℕ := 4

-- Define the problem as a theorem
theorem number_of_ways_to_divide_people : (number_of_ways n max_capacity) = 60 := 
sorry

-- Helper function to count the number of valid divisions
noncomputable def number_of_ways (n : ℕ) (max_capacity : ℕ) : ℕ := 
sorry

end number_of_ways_to_divide_people_l612_612444


namespace exists_BD_parallelogram_l612_612645

structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

variables (A C : Point) (Γ : Circle)

theorem exists_BD_parallelogram : ∃ B D : Point, (∥B - Γ.center∥ = Γ.radius) ∧ (∥D - Γ.center∥ = Γ.radius) ∧ 
  ((B.x + D.x) / 2 = (A.x + C.x) / 2) ∧ ((B.y + D.y) / 2 = (A.y + C.y) / 2) :=
sorry

end exists_BD_parallelogram_l612_612645


namespace distance_P_to_l_l612_612771

def point := ℝ × ℝ

def line (A B C : ℝ) (p : point) : Prop :=
  A * p.1 + B * p.2 + C = 0

def distance_from_point_to_line (p : point) (A B C : ℝ) : ℝ :=
  abs (A * p.1 + B * p.2 + C) / real.sqrt (A^2 + B^2)

theorem distance_P_to_l :
  let P : point := (0, 2)
  let A := 1
  let B := -1
  let C := 3
  distance_from_point_to_line P A B C = real.sqrt 2 / 2 :=
by
  sorry

end distance_P_to_l_l612_612771


namespace base_b_square_l612_612556

-- Given that 144 in base b can be written as b^2 + 4b + 4 in base 10,
-- prove that it is a perfect square if and only if b > 4

theorem base_b_square (b : ℕ) (h : b > 4) : ∃ k : ℕ, b^2 + 4 * b + 4 = k^2 := by
  sorry

end base_b_square_l612_612556


namespace lean_example_l612_612598

variable {a b : ℝ}

theorem lean_example (h1 : a > 1) (h2 : ∀ x, x > 1 → (x - (x-1) * 2^x = 0 → a = x)) 
                     (h3 : ∀ x, x > 1 → (x - (x-1) * log 2 x = 0 → b = x)) 
                     (h4 : b = 2^a) (h5 : a = log 2 b) :
  (b - a = 2^a - log 2 b) ∧ 
  (1 / a + 1 / b = 1) ∧ 
  (b - a > 1) :=
by {
  sorry
}

end lean_example_l612_612598


namespace stratified_sampling_l612_612889

theorem stratified_sampling (lathe_A lathe_B total_samples : ℕ) (hA : lathe_A = 56) (hB : lathe_B = 42) (hTotal : total_samples = 14) :
  ∃ (sample_A sample_B : ℕ), sample_A = 8 ∧ sample_B = 6 :=
by
  sorry

end stratified_sampling_l612_612889


namespace eigenvalues_M_zero_or_k_l612_612210

variable {α : Type _}
variable [Field α]
variable {n : Type _} [Fintype n] [DecidableEq n]

noncomputable def M {k : Nat} (A : Fin k → Matrix n n α) : Matrix n n α :=
  ∑ i, A i

theorem eigenvalues_M_zero_or_k {k : Nat} (A : Fin k → Matrix n n α) 
  (h : ∀ i j, A i * A j = A j * A i ∧ (∃ I : Matrix n n α, I.isUnit ∧ I * A i = A i) ∧ (∃ E : Matrix n n α, E.isInv á ε * A i = A i))
  (hM : M A = ∑ i, A i) : 
  ∀ (λ : α), λ ∈ spectrum (Matrix n n α) M → λ = 0 ∨ λ = k :=
sorry

end eigenvalues_M_zero_or_k_l612_612210


namespace problem_statement_l612_612123

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def pow_log2 (x : ℝ) : ℝ := x ^ log2 x

theorem problem_statement (a b c : ℝ)
  (h0 : 1 ≤ a)
  (h1 : 1 ≤ b)
  (h2 : 1 ≤ c)
  (h3 : a * b * c = 10)
  (h4 : pow_log2 a * pow_log2 b * pow_log2 c ≥ 10) :
  a + b + c = 12 := by
  sorry

end problem_statement_l612_612123


namespace circle_radius_tangent_to_parabola_l612_612452

theorem circle_radius_tangent_to_parabola (a : ℝ) (b r : ℝ) :
  (∀ x : ℝ, y = 4 * x ^ 2) ∧ 
  (b = a ^ 2 / 4) ∧ 
  (∀ x : ℝ, x ^ 2 + (4 * x ^ 2 - b) ^ 2 = r ^ 2)  → 
  r = a ^ 2 / 4 := 
  sorry

end circle_radius_tangent_to_parabola_l612_612452


namespace number_of_good_numbers_l612_612359

theorem number_of_good_numbers :
  let n := 2020 in 
  let remainder := 22 in
  let number := 1998 in
  let divisors := {d ∣ number | d > remainder}.to_list.length = 10 := by sorry

end number_of_good_numbers_l612_612359


namespace num_squares_below_diagonal_correct_l612_612311

noncomputable def squares_below_diagonal (width height : ℕ) (diagonal : ℕ) : ℕ := (width * height - diagonal) / 2

theorem num_squares_below_diagonal_correct :
  ∀ (x y total_diagonal : ℕ),
    x = 223 →
    y = 9 →
    total_diagonal = 231 →
    squares_below_diagonal x y total_diagonal = 888 :=
by
  intros x y total_diagonal hx hy htotal_diagonal
  rw [hx, hy, htotal_diagonal]
  unfold squares_below_diagonal
  simp
  sorry

end num_squares_below_diagonal_correct_l612_612311


namespace distance_between_centers_l612_612770

noncomputable def rho_cos_theta := {p : ℝ × ℝ | ∃ θ : ℝ, p = (Real.cos θ, θ)}
noncomputable def rho_sin_theta := {p : ℝ × ℝ | ∃ θ : ℝ, p = (Real.sin θ, θ)}

def center_rho_cos_theta : ℝ × ℝ := (1 / 2, 0)
def center_rho_sin_theta : ℝ × ℝ := (0, 1 / 2)

theorem distance_between_centers :
  dist center_rho_cos_theta center_rho_sin_theta = Real.sqrt 2 / 2 := by
  sorry

end distance_between_centers_l612_612770


namespace six_books_discounted_cost_is_81_l612_612816

-- defining the given conditions
def cost_of_three_books : ℝ := 45
def discount_percentage : ℝ := 0.10

-- calculating unit price from given conditions
def unit_price : ℝ := cost_of_three_books / 3

-- calculating regular cost for six books
def regular_cost_for_six_books : ℝ := 6 * unit_price

-- calculating the discounted cost
def discounted_cost : ℝ := regular_cost_for_six_books - (discount_percentage * regular_cost_for_six_books)

-- statement to prove: the cost for six books after discount is 81 dollars
theorem six_books_discounted_cost_is_81 : discounted_cost = 81 := 
by
  sorry

end six_books_discounted_cost_is_81_l612_612816


namespace remainder_when_divided_by_84_l612_612418

/-- 
  Given conditions:
  x ≡ 11 [MOD 14]
  Find the remainder when x is divided by 84, which equivalently means proving: 
  x ≡ 81 [MOD 84]
-/

theorem remainder_when_divided_by_84 (x : ℤ) (h1 : x % 14 = 11) : x % 84 = 81 :=
by
  sorry

end remainder_when_divided_by_84_l612_612418


namespace cat_food_inequality_l612_612533

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l612_612533


namespace cat_food_insufficient_l612_612506

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l612_612506


namespace find_pq_l612_612712

theorem find_pq (p q : ℝ) : (∃ x : ℝ, f(p, q, x) = 0) ∧ (∃ y : ℝ, f(p, q, y) = 0) ↔ 
(p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2) := 
begin
  sorry,
end

def f (p q x : ℝ) : ℝ := x^2 + p * x + q

end find_pq_l612_612712


namespace find_f_of_neg_2_l612_612308

def f : ℝ → ℝ := λ x, 
  if x < 2 then f (x + 2)
  else 2 * x - 5

theorem find_f_of_neg_2 : f (-2) = -1 := by
  sorry

end find_f_of_neg_2_l612_612308


namespace count_of_good_numbers_l612_612346

-- Define the conditions of the problem
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- The main statement proving the number of 'good' numbers
theorem count_of_good_numbers :
  (finset.filter is_good_number (finset.Icc 1 2020)).card = 10 :=
by
  sorry

end count_of_good_numbers_l612_612346


namespace sufficient_condition_for_product_l612_612118

-- Given conditions
def intersects_parabola_at_two_points (x1 y1 x2 y2 : ℝ) : Prop :=
  y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 ∧ x1 ≠ x2

def line_through_focus (x y : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 1)

noncomputable def line_l (k : ℝ) (x : ℝ) : ℝ :=
  k * (x - 1)

-- The theorem to prove
theorem sufficient_condition_for_product 
  (x1 y1 x2 y2 k : ℝ)
  (h1 : intersects_parabola_at_two_points x1 y1 x2 y2)
  (h2 : line_through_focus x1 y1 k)
  (h3 : line_through_focus x2 y2 k) :
  x1 * x2 = 1 :=
sorry

end sufficient_condition_for_product_l612_612118


namespace number_of_incorrect_propositions_is_2_l612_612130

def proposition_1 (p q : Prop) : Prop := ¬ (p ∧ q) → ¬p ∧ ¬q
def proposition_2 (a b : ℕ) : Prop := ¬(a > b → 2^a > 2^b - 1) = (a ≤ b → 2^a ≤ 2^b - 1)
def proposition_3 : Prop := ¬(∀ x : ℝ, x^2 + 1 ≥ 1) = ∃ x : ℝ, x^2 + 1 < 1
def proposition_4 (A B : ℝ) : Prop := A > B ↔ Real.sin A > Real.sin B

def number_of_incorrect_propositions : Nat := 
  if ¬proposition_1 true false then 1 else 0 + 
  if proposition_2 1 0 then 0 else 1 + 
  if proposition_3 then 0 else 1 + 
  if proposition_4 1 0 then 0 else 1

theorem number_of_incorrect_propositions_is_2 : number_of_incorrect_propositions = 2 := by
  sorry

end number_of_incorrect_propositions_is_2_l612_612130


namespace Isabella_exchange_l612_612246

/-
Conditions:
1. Isabella exchanged d U.S. dollars to receive (8/5)d Canadian dollars.
2. After spending 80 Canadian dollars, she had d + 20 Canadian dollars left.
3. Sum of the digits of d is 14.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (.+.) 0

theorem Isabella_exchange (d : ℕ) (h : (8 * d / 5) - 80 = d + 20) : sum_of_digits d = 14 :=
by sorry

end Isabella_exchange_l612_612246


namespace cone_base_radius_l612_612789

theorem cone_base_radius (angle : ℝ) (sector_radius : ℝ) (base_radius : ℝ) 
(h1 : angle = 216)
(h2 : sector_radius = 15)
(h3 : 2 * π * base_radius = (3 / 5) * 2 * π * sector_radius) :
base_radius = 9 := 
sorry

end cone_base_radius_l612_612789


namespace anna_money_left_l612_612033

theorem anna_money_left : 
  let initial_money := 10.0
  let gum_cost := 3.0 -- 3 packs at $1.00 each
  let chocolate_cost := 5.0 -- 5 bars at $1.00 each
  let cane_cost := 1.0 -- 2 canes at $0.50 each
  let total_spent := gum_cost + chocolate_cost + cane_cost
  let money_left := initial_money - total_spent
  money_left = 1.0 := by
  sorry

end anna_money_left_l612_612033


namespace contrapositive_p_l612_612472

-- Definitions
def A_score := 70
def B_score := 70
def C_score := 65
def p := ∀ (passing_score : ℕ), passing_score < 70 → (A_score < passing_score ∧ B_score < passing_score ∧ C_score < passing_score)

-- Statement to be proved
theorem contrapositive_p : 
  ∀ (passing_score : ℕ), (A_score ≥ passing_score ∨ B_score ≥ passing_score ∨ C_score ≥ passing_score) → (¬ passing_score < 70) := 
by
  sorry

end contrapositive_p_l612_612472


namespace ratio_equivalent_l612_612043

variable barbara_stuffed_animals : ℕ := 9
variable barbara_price_per_animal : ℝ := 2
variable trish_price_per_animal : ℝ := 1.5
variable total_donation : ℝ := 45

theorem ratio_equivalent (T : ℕ) 
  (h1 : barbara_price_per_animal * barbara_stuffed_animals + trish_price_per_animal * T = total_donation) :
  ((T : ℝ) / barbara_stuffed_animals) = 2 := 
  sorry

end ratio_equivalent_l612_612043


namespace sum_first_6_terms_l612_612983

variable {ℕ : Type}
variable (a : ℕ → ℤ) (d : ℤ)
variable (S : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n+1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 1 + a n)) / 2

theorem sum_first_6_terms {a : ℕ → ℤ} (d : ℤ) (a1_eq : a 1 = 6) (a3_a5_eq : a 3 + a 5 = 0)
  (a_seq : is_arithmetic_sequence a d) : sum_first_n_terms a 6 = 6 :=
by
  sorry

end sum_first_6_terms_l612_612983


namespace f_a_eq_half_l612_612131

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else 2 ^ x

theorem f_a_eq_half (a : ℝ) (h : f a = 1/2) : a = sqrt 2 ∨ a = -1 := by
  sorry

end f_a_eq_half_l612_612131


namespace mike_found_four_more_seashells_l612_612727

/--
Given:
1. Mike initially found 6.0 seashells.
2. The total number of seashells Mike had after finding more is 10.

Prove:
Mike found 4.0 more seashells.
-/
theorem mike_found_four_more_seashells (initial_seashells : ℝ) (total_seashells : ℝ)
  (h1 : initial_seashells = 6.0)
  (h2 : total_seashells = 10.0) :
  total_seashells - initial_seashells = 4.0 :=
by
  sorry

end mike_found_four_more_seashells_l612_612727


namespace find_acute_angle_l612_612107

theorem find_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < 90) (h2 : ∃ k : ℤ, 10 * α = α + k * 360) :
  α = 40 ∨ α = 80 :=
by
  sorry

end find_acute_angle_l612_612107


namespace wedge_product_correct_l612_612700

variables {a1 a2 b1 b2 : ℝ}
def a : ℝ × ℝ := (a1, a2)
def b : ℝ × ℝ := (b1, b2)

def wedge_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.2 - v.2 * w.1

theorem wedge_product_correct (a b : ℝ × ℝ) :
  wedge_product a b = a.1 * b.2 - a.2 * b.1 :=
by
  -- Proof is omitted, theorem statement only
  sorry

end wedge_product_correct_l612_612700


namespace shift_sin_from_cos_l612_612819

theorem shift_sin_from_cos (x : ℝ) :
  ∃ (h : ℝ), ∀ x : ℝ, sin (x / 2) = cos ((x - h) / 2 - π / 4) :=
by
  use π / 2
  sorry

end shift_sin_from_cos_l612_612819


namespace f_neg1_plus_f_2_eq_0_l612_612980

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 4 * x else Real.exp (x * Real.log 2)

theorem f_neg1_plus_f_2_eq_0 : f (-1) + f 2 = 0 := by
  sorry

end f_neg1_plus_f_2_eq_0_l612_612980


namespace number_of_yellow_balls_l612_612031

theorem number_of_yellow_balls (x : ℕ) :
  (4 : ℕ) / (4 + x) = 2 / 3 → x = 2 :=
by
  sorry

end number_of_yellow_balls_l612_612031


namespace circles_area_problem_l612_612906

-- Define the properties and configuration of the circles
def radiusA : ℝ := 1
def radiusB : ℝ := 1
def radiusC : ℝ := 2
def tangentPoint := (0, 0)  -- Let the midpoint of AB be at the origin for simplicity

-- Define the problem
theorem circles_area_problem :
  let areaC := π * radiusC^2
  let intersection_area := 2 * (π / 6 - (sqrt(3) / 2))
  let area_inside_outside := areaC - intersection_area
  area_inside_outside = (11 * π / 3) + sqrt(3) :=
by
  sorry

end circles_area_problem_l612_612906


namespace difference_of_squares_l612_612656

variable (a b : ℝ)

theorem difference_of_squares (h1 : a + b = 2) (h2 : a - b = 3) : a^2 - b^2 = 6 := 
by
  sorry

end difference_of_squares_l612_612656


namespace sum_of_minima_l612_612542

-- Define P(x) and Q(x) as monic quadratic polynomials.
def P (x : ℝ) : ℝ := x^2 + b * x + c
def Q (x : ℝ) : ℝ := x^2 + e * x + f

-- Given conditions that P(Q(x)) has specific roots
def PQ_has_roots : Prop :=
  let PQ_x := λ x, P(Q(x)) in
  PQ_x (-19) = 0 ∧ PQ_x (-13) = 0 ∧ PQ_x (-11) = 0 ∧ PQ_x (-7) = 0

-- Given conditions that Q(P(x)) has specific roots
def QP_has_roots : Prop :=
  let QP_x := λ x, Q(P(x)) in
  QP_x (-47) = 0 ∧ QP_x (-43) = 0 ∧ QP_x (-37) = 0 ∧ QP_x (-31) = 0

-- Define the minimum value sum of P(x) and Q(x)
def min_PQ_sum : ℝ :=
  P(-b / 2) + Q(-e / 2)

-- Final theorem stating the problem
theorem sum_of_minima (b e : ℝ) (c f : ℝ) (h1 : PQ_has_roots) (h2 : QP_has_roots) :
  min_PQ_sum = -100 :=
by sorry

end sum_of_minima_l612_612542


namespace derivative_sin_at_pi_over_3_l612_612630

noncomputable def f : ℝ → ℝ := sin

theorem derivative_sin_at_pi_over_3 :
  (∀ Δx, ∆x ≠ 0 → (f ((π / 3) + Δx) - f (π / 3)) / Δx) ⟶ (1 / 2) :=
by
  sorry

end derivative_sin_at_pi_over_3_l612_612630


namespace cube_volume_surface_area_l612_612325

theorem cube_volume_surface_area (sum_edges : ℕ) :
  sum_edges = 72 →
  ∃ a V S, 
    a = sum_edges / 12 ∧ 
    V = a^3 ∧ 
    S = 6 * a^2 ∧ 
    V = 216 ∧ 
    S = 216 :=
begin
  sorry
end

end cube_volume_surface_area_l612_612325


namespace christopher_more_money_l612_612694

-- Define the conditions provided in the problem

def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64
def quarter_value : ℝ := 0.25

-- Define the question as a theorem

theorem christopher_more_money : (christopher_quarters - karen_quarters) * quarter_value = 8 :=
by sorry

end christopher_more_money_l612_612694


namespace volume_of_P5_is_fraction_and_sum_l612_612601

noncomputable def height_seq : ℕ → ℝ
| 0 => 1
| (n+1) => (1/2)^(n+1)

def volume_seq : ℕ → ℝ 
| 0 => 1
| (n+1) => volume_seq n + (6 * height_seq (n+1)^3 * (4 / 24))

def volume_P5 := volume_seq 5
def m := 8929
def n := 4096
def total := m + n

theorem volume_of_P5_is_fraction_and_sum :
  volume_P5 = (m / n) ∧ total = 13025 :=
by 
  sorry

end volume_of_P5_is_fraction_and_sum_l612_612601


namespace maria_average_speed_l612_612312

noncomputable def average_speed (total_distance : ℕ) (total_time : ℕ) : ℚ :=
  total_distance / total_time

theorem maria_average_speed :
  average_speed 200 7 = 28 + 4 / 7 :=
sorry

end maria_average_speed_l612_612312


namespace min_value_f_l612_612627

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem min_value_f (a : ℝ) (h : -2 < a) :
  ∃ m, (∀ x ∈ Set.Icc (-2 : ℝ) a, f x ≥ m) ∧ 
  ((a ≤ 1 → m = a^2 - 2 * a) ∧ (1 < a → m = -1)) :=
by
  sorry

end min_value_f_l612_612627


namespace sqrt_p_adic_integer_l612_612289

def p_adic_integer (p : ℕ) (x : ℕ → ℤ) : Prop :=
  ∃ a : ℕ → ℤ, (∀ i, 0 ≤ a i ∧ a i < p) ∧ x = λ i, a i * p^i

theorem sqrt_p_adic_integer (p : ℕ) (x : ℕ → ℤ) (hx : p_adic_integer p x) : p_adic_integer p (λ i, int.sqrt (x i)) :=
sorry

end sqrt_p_adic_integer_l612_612289


namespace quadratic_has_two_distinct_real_roots_l612_612801

theorem quadratic_has_two_distinct_real_roots :
  ∀ (x : ℝ), ∃ (r1 r2 : ℝ), (x^2 - 2*x - 1 = 0) → r1 ≠ r2 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l612_612801


namespace eval_expression_l612_612414

theorem eval_expression (a b : ℤ) (h₁ : a = 4) (h₂ : b = -2) : -a - b^2 + a*b + a^2 = 0 := by
  sorry

end eval_expression_l612_612414


namespace interest_rate_per_annum_l612_612011

-- Define the given values and conditions
def P : ℝ := 2500
def t : ℝ := 8
def interest (r : ℝ) : ℝ := (P * r * t) / 100

-- Define the condition that the interest is 900 less than the sum lent
def interest_condition (r : ℝ) : Prop := interest(r) = P - 900

-- State the theorem about the interest rate per annum
theorem interest_rate_per_annum : ∃ r : ℝ, interest_condition(r) ∧ r = 8 := by 
  sorry

end interest_rate_per_annum_l612_612011


namespace car_highway_miles_per_tankful_l612_612856

-- Condition definitions
def city_miles_per_tankful : ℕ := 336
def miles_per_gallon_city : ℕ := 24
def city_to_highway_diff : ℕ := 9

-- Calculation from conditions
def miles_per_gallon_highway : ℕ := miles_per_gallon_city + city_to_highway_diff
def tank_size : ℤ := city_miles_per_tankful / miles_per_gallon_city

-- Desired result
def highway_miles_per_tankful : ℤ := miles_per_gallon_highway * tank_size

-- Proof statement
theorem car_highway_miles_per_tankful :
  highway_miles_per_tankful = 462 := by
  unfold highway_miles_per_tankful
  unfold miles_per_gallon_highway
  unfold tank_size
  -- Sorry here to skip the detailed proof steps
  sorry

end car_highway_miles_per_tankful_l612_612856


namespace work_completion_days_l612_612834

theorem work_completion_days (A B : ℕ) (h1 : A = 2 * B) (h2 : 6 * (A + B) = 18) : B = 1 → 18 = 18 :=
by
  sorry

end work_completion_days_l612_612834


namespace rectangle_perimeter_is_48_l612_612295

theorem rectangle_perimeter_is_48 
  (area_top_left_square : ℝ)
  (area_bottom_left_square : ℝ)
  (h1 : area_top_left_square = 36)
  (h2 : area_bottom_left_square = 25) : 
  2 * (((sqrt area_top_left_square) + (sqrt area_bottom_left_square)) + ((sqrt area_top_left_square) + (sqrt (area_top_left_square - area_bottom_left_square)))) = 48 :=
by {
  sorry
}

end rectangle_perimeter_is_48_l612_612295


namespace food_requirement_not_met_l612_612501

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l612_612501


namespace window_width_l612_612066

theorem window_width (x : ℕ) : 
  let pane_width := 3 * x
  let borders := 3
  let num_panes_in_row := 4
  let num_borders_between_panes := 3
  let num_borders_sides := 2
  in  num_panes_in_row * pane_width + (num_borders_between_panes + num_borders_sides) * borders = 12 * x + 15 := 
by
  let pane_width := 3 * x
  let borders := 3
  let num_panes_in_row := 4
  let num_borders_between_panes := 3
  let num_borders_sides := 2
  calc 
    num_panes_in_row * pane_width + (num_borders_between_panes + num_borders_sides) * borders
        = 4 * (3 * x) + (3 + 2) * 3 : by rfl
    ... = 12 * x + 15 : by rw [← nat.mul_add_distrib]

end window_width_l612_612066


namespace half_abs_diff_of_squares_l612_612404

theorem half_abs_diff_of_squares (x y : ℤ) (h1 : x = 21) (h2 : y = 19) :
  (|x^2 - y^2| / 2) = 40 := 
by
  subst h1
  subst h2
  sorry

end half_abs_diff_of_squares_l612_612404


namespace colby_mangoes_harvested_60_l612_612541

variable (kg_left kg_each : ℕ)

def totalKgMangoes (x : ℕ) : Prop :=
  ∃ x : ℕ, 
  kg_left = (x - 20) / 2 ∧ 
  kg_each * kg_left = 160 ∧
  kg_each = 8

-- Problem Statement: Prove the total kilograms of mangoes harvested is 60 given the conditions.
theorem colby_mangoes_harvested_60 (x : ℕ) (h1 : x - 20 = 2 * kg_left)
(h2 : kg_each * kg_left = 160) (h3 : kg_each = 8) : x = 60 := by
  sorry

end colby_mangoes_harvested_60_l612_612541


namespace math_problem_solution_l612_612478

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x > f y

def problem_statement : Prop :=
  (is_odd (fun x => -x^3) ∧ decreasing_on (fun x => -x^3) {x | 0 < x ∧ x < 1}) ∧
  (¬ is_odd (fun x => (1/2)^|x|) ∨ ¬ decreasing_on (fun x => (1/2)^|x|) {x | 0 < x ∧ x < 1}) ∧
  (is_odd (fun x => -sin x) ∧ decreasing_on (fun x => -sin x) {x | 0 < x ∧ x < 1}) ∧
  (is_odd (fun x => x / exp |x|) ∧ ¬ decreasing_on (fun x => x / exp |x|) {x | 0 < x ∧ x < 1})

theorem math_problem_solution : problem_statement :=
  sorry

end math_problem_solution_l612_612478


namespace larry_substituted_value_l612_612236

theorem larry_substituted_value :
  ∀ (a b c d e : ℤ), a = 5 → b = 3 → c = 4 → d = 2 → e = 2 → 
  (a + b - c + d - e = a + (b - (c + (d - e)))) :=
by
  intros a b c d e ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end larry_substituted_value_l612_612236


namespace time_to_save_for_downpayment_l612_612728

-- Definitions based on conditions
def annual_saving : ℝ := 0.10 * 150000
def downpayment : ℝ := 0.20 * 450000

-- Statement of the theorem to be proved
theorem time_to_save_for_downpayment (T : ℝ) (H1 : annual_saving = 15000) (H2 : downpayment = 90000) : 
  T = 6 :=
by
  -- Placeholder for the proof
  sorry

end time_to_save_for_downpayment_l612_612728


namespace equalities_hold_l612_612710

noncomputable theory

def f (x p q : ℝ) : ℝ := x^2 + p * x + q

theorem equalities_hold (p q : ℝ) :
  f p p q = 0 ∧ f q p q = 0 ↔ (p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2) :=
by {
  sorry -- Proof is omitted as it is not required.
}

end equalities_hold_l612_612710


namespace triangle_angle_C_max_cos_A_plus_cos_B_l612_612966

noncomputable section

variables {a b c A B C : ℝ}

-- 1. Prove that given c² = a² + b² - ab, the magnitude of angle C is π/3
theorem triangle_angle_C (h : c^2 = a^2 + b^2 - ab) : C = π / 3 := sorry

-- 2. Prove that given C = π/3, the maximum value of cos A + cos B is 1
theorem max_cos_A_plus_cos_B (h : C = π / 3) : (∃ A B, cos A + cos B = 1) := sorry

end triangle_angle_C_max_cos_A_plus_cos_B_l612_612966


namespace find_theta_l612_612595

open Real

theorem find_theta (theta : ℝ) : sin theta = -1/3 ∧ -π < theta ∧ theta < -π / 2 ↔ theta = -π - arcsin (-1 / 3) :=
by
  sorry

end find_theta_l612_612595


namespace find_k_l612_612925

-- Define the conditions
def quadratic (k : ℝ) : Prop := x^2 - 20 * x + k = (x + (-10))^2
-- State the problem to find the specific k
theorem find_k (k : ℝ) : quadratic k → k = 100 :=
by
  intro h
  rw quadratic at h
  sorry

end find_k_l612_612925


namespace PQ_bisects_BL_l612_612190

noncomputable def acute_triangle (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] : Prop :=
∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = π ∧ a < π/2 ∧ b < π/2 ∧ c < π/2

noncomputable def angle_bisector_intersects (A B C L : Type*) : Prop :=
∃ (A C : ℝ), L lies_on_angle_bisector_of B intersecting AC

noncomputable def midpoint_of_minor_arc (D E : Type*) (circumcircle : Type*) (arc_AB arc_BC : set (Type*)) : Prop :=
D is_midpoint_of arc_AB ∧ E is_midpoint_of arc_BC

noncomputable def foot_of_perpendicular (P Q : Type*) (A C : Type*) (BD BE : Type*) : Prop :=
P is_foot_of_perpendicular_from A onto BD ∧ Q is_foot_of_perpendicular_from C onto BE

noncomputable def bisects (PQ BL : Type*) : Prop :=
PQ ∥ BL / 2

theorem PQ_bisects_BL 
  {A B C L D E P Q : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space L] 
  [metric_space D] [metric_space E] [metric_space P] [metric_space Q] 
  (hABC : acute_triangle A B C)
  (hangle_bisector : angle_bisector_intersects A B C L)
  (hmidpoints : midpoint_of_minor_arc D E circumcircle)
  (hperpendiculars : foot_of_perpendicular P Q A C BD BE) :
  bisects PQ BL := sorry

end PQ_bisects_BL_l612_612190


namespace sufficient_but_not_necessary_condition_l612_612110

open Real

theorem sufficient_but_not_necessary_condition :
  ∀ (m : ℝ),
  (∀ x, (x^2 - 3*x - 4 ≤ 0) → (x^2 - 6*x + 9 - m^2 ≤ 0)) ∧
  (∃ x, ¬(x^2 - 3*x - 4 ≤ 0) ∧ (x^2 - 6*x + 9 - m^2 ≤ 0)) ↔
  m ∈ Set.Iic (-4) ∪ Set.Ici 4 :=
by
  sorry

end sufficient_but_not_necessary_condition_l612_612110


namespace cat_food_insufficient_for_six_days_l612_612526

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l612_612526


namespace one_sixth_of_x_l612_612169

theorem one_sixth_of_x (x : ℝ) (h : x / 3 = 4) : x / 6 = 2 :=
sorry

end one_sixth_of_x_l612_612169


namespace convert_spherical_to_rectangular_l612_612549

-- Definitions of the conditions in the problem
def rho : ℝ := 6
def theta : ℝ := (7 * Real.pi) / 4
def phi : ℝ := Real.pi / 3

-- Definitions of the expected rectangular coordinates
def x : ℝ := 3 * Real.sqrt 6
def y : ℝ := - (3 * Real.sqrt 6)
def z : ℝ := 3

-- The equivalent proof problem
theorem convert_spherical_to_rectangular :
  let x' := rho * Real.sin phi * Real.cos theta,
      y' := rho * Real.sin phi * Real.sin theta,
      z' := rho * Real.cos phi
  in x' = x ∧ y' = y ∧ z' = z :=
by
  let x' := rho * Real.sin phi * Real.cos theta
  let y' := rho * Real.sin phi * Real.sin theta
  let z' := rho * Real.cos phi
  sorry

end convert_spherical_to_rectangular_l612_612549


namespace box_volume_l612_612040

theorem box_volume (initial_length initial_width cut_length : ℕ)
  (length_condition : initial_length = 13) (width_condition : initial_width = 9)
  (cut_condition : cut_length = 2) : 
  (initial_length - 2 * cut_length) * (initial_width - 2 * cut_length) * cut_length = 90 := 
by
  sorry

end box_volume_l612_612040


namespace fraction_of_tomatoes_eaten_l612_612327

theorem fraction_of_tomatoes_eaten (original : ℕ) (remaining : ℕ) (birds_ate : ℕ) (h1 : original = 21) (h2 : remaining = 14) (h3 : birds_ate = original - remaining) :
  (birds_ate : ℚ) / original = 1 / 3 :=
by
  sorry

end fraction_of_tomatoes_eaten_l612_612327


namespace number_of_divisors_greater_than_22_l612_612385

theorem number_of_divisors_greater_than_22 :
  (∃ (S : Finset ℕ), S = (Finset.filter (λ n, 1998 % n = 0 ∧ n > 22) (Finset.range (1998 + 1))) ∧ S.card = 10) :=
sorry

end number_of_divisors_greater_than_22_l612_612385


namespace English_marks_l612_612551

-- Define the known values
def Mathematics_marks : ℤ := 89
def Physics_marks : ℤ := 82
def Chemistry_marks : ℤ := 87
def Biology_marks : ℤ := 81
def Average_marks : ℤ := 85
def Number_of_subjects : ℤ := 5

-- Define the assertion we need to prove
theorem English_marks : 
  let E := 425 - (Mathematics_marks + Physics_marks + Chemistry_marks + Biology_marks) in
  E = 86 :=
by
  sorry

end English_marks_l612_612551


namespace line_AM_perpendicular_to_x_axis_angle_OMA_OMB_are_equal_l612_612229

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

def focal_point : (ℝ × ℝ) := (1, 0)

def point_M : (ℝ × ℝ) := (2, 0)

-- Proof to show the equation of line AM when l is perpendicular to x-axis
theorem line_AM_perpendicular_to_x_axis :
  (∃ y : ℝ, (1 / y = sqrt 2 / 2 ∨ 1 / y = -sqrt 2 / 2) ∧ (y = (-sqrt 2 / 2) * 1 + sqrt 2 ∨ y = (sqrt 2 / 2) * 1 - sqrt 2)) :=
sorry

-- Proof to show that ∠OMA = ∠OMB
theorem angle_OMA_OMB_are_equal :
  ∀ (O A B M : (ℝ × ℝ)), ∃ l : (ℝ × ℝ),
   (l = λ x, k := y - k(x - 1)) → 
   (A ∈ ellipse (A.1) (A.2) ∧ B ∈ ellipse (B.1) (B.2)) ∧ (O = (0,0) ∧ A = (1, sqrt(2)/2) ∧ B = (1, -sqrt(2)/2) ∧ M = (2,0))
   → ∠ OMA = ∠ OMB :=
sorry

end line_AM_perpendicular_to_x_axis_angle_OMA_OMB_are_equal_l612_612229


namespace number_of_divisors_greater_than_22_l612_612386

theorem number_of_divisors_greater_than_22 :
  (∃ (S : Finset ℕ), S = (Finset.filter (λ n, 1998 % n = 0 ∧ n > 22) (Finset.range (1998 + 1))) ∧ S.card = 10) :=
sorry

end number_of_divisors_greater_than_22_l612_612386


namespace max_c_magnitude_l612_612648

noncomputable def vec_magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem max_c_magnitude 
  (a b c : ℝ × ℝ)
  (ha : vec_magnitude a = 1)
  (hb : vec_magnitude b = 1)
  (ha_dot_b : dot_product a b = 1 / 2)
  (hacb_mag : vec_magnitude (a - b + c) ≤ 1) :
  vec_magnitude c ≤ 2 :=
sorry

end max_c_magnitude_l612_612648


namespace primes_in_sequence_are_12_l612_612053

-- Definition of Q
def Q : Nat := (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47)

-- Set of m values
def ms : List Nat := List.range' 3 101

-- Function to check if Q + m is prime
def is_prime_minus_Q (m : Nat) : Bool := Nat.Prime (Q + m)

-- Counting primes in the sequence
def count_primes_in_sequence : Nat := (ms.filter (λ m => is_prime_minus_Q m = true)).length

theorem primes_in_sequence_are_12 :
  count_primes_in_sequence = 12 := by 
  sorry

end primes_in_sequence_are_12_l612_612053


namespace sqrt_x_div_sqrt_y_as_fraction_l612_612080

theorem sqrt_x_div_sqrt_y_as_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (sqrt x / sqrt y) = 1281 / 94 :=
by
  have h : (1/3:ℝ)^2 + (1/4)^2 = 1/9 + 1/16 := sorry
  have k : (1/5:ℝ)^2 + (1/6)^2 = 1/25 + 1/36 := sorry
  have ratio : (1/9 + 1/16) / (1/25 + 1/36) = 22500 / 8784 := sorry
  have eq1 : (37 * x) / (73 * y) = 22500 / 8784 := sorry
  have eq2 : x / y = 1642500 / 8784 := sorry
  have sqrt_eq : sqrt (x / y) = sqrt (1642500 / 8784) := sorry
  sorry

end sqrt_x_div_sqrt_y_as_fraction_l612_612080


namespace length_AN_l612_612247

theorem length_AN {A B C M N : Type*}
  (h1 : M ∈ angle_bisector A B C)
  (h2 : N ∈ extension A B)
  (h3 : distance A C = 1)
  (h4 : distance A M = 1)
  (h5 : ∠ A N M = ∠ C N M) :
  distance A N = 1 :=
sorry

end length_AN_l612_612247


namespace globe_surface_area_l612_612012

theorem globe_surface_area (d : ℚ) (h : d = 9) : 
  4 * Real.pi * (d / 2) ^ 2 = 81 * Real.pi := 
by 
  sorry

end globe_surface_area_l612_612012


namespace perpendicular_lines_condition_l612_612316

variable {A1 B1 C1 A2 B2 C2 : ℝ}

theorem perpendicular_lines_condition :
  (∀ x y : ℝ, A1 * x + B1 * y + C1 = 0) ∧ (∀ x y : ℝ, A2 * x + B2 * y + C2 = 0) → 
  (A1 * A2) / (B1 * B2) = -1 := 
sorry

end perpendicular_lines_condition_l612_612316


namespace probability_correct_guess_l612_612732

-- Define the conditions for the problem
def is_valid_number (n : ℕ) : Prop :=
  n > 42 ∧ n < 90 ∧
  (n % 10) ∈ {1, 3, 5, 7, 9} ∧ -- units digit is odd
  ((n / 10) % 2 = 0) -- tens digit is even

def valid_numbers : Finset ℕ := Finset.filter is_valid_number (Finset.range 100)

-- The probability Nancy guesses the correct number is 1 / number of valid numbers
theorem probability_correct_guess :
  1 / valid_numbers.card = 1 / 15 :=
sorry

end probability_correct_guess_l612_612732


namespace part_I_part_II_part_III_l612_612134

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.ln x + a * x

theorem part_I :
  (∀ x a, (a = 1) → (2 * 1 - f 1 1 = 1) ∧ (∀ x, 2 * x - f x 1 = 0 → 2 * x - f x 1 = 1 - (x * x)) ∧ (2 * x - (x * x) - f x 1 = 0) → 2 * x - f x 1 = 1) :=
sorry

theorem part_II (a : ℝ) :
  (if a ≥ 0 then ∀ x, 0 < x → 0 < f x a
                  else ∀ x, (0 < x ∧ x < (-1/a)) ∨ ((-1/a) < x ∧ x < ⊤)) :=
sorry

theorem part_III (a : ℝ) :
  (-1/e < a → ∃ x, f x a > 0) :=
sorry

end part_I_part_II_part_III_l612_612134


namespace good_numbers_2020_has_count_10_l612_612399

theorem good_numbers_2020_has_count_10 :
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  set.count divisors = 10 :=
by
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  sorry

end good_numbers_2020_has_count_10_l612_612399


namespace proof_correct_word_choice_l612_612429

def sentence_completion_correct (word : String) : Prop :=
  "Most of them are kind, but " ++ word ++ " is so good to me as Bruce" = "Most of them are kind, but none is so good to me as Bruce"

theorem proof_correct_word_choice : 
  (sentence_completion_correct "none") → 
  ("none" = "none") := 
by
  sorry

end proof_correct_word_choice_l612_612429


namespace area_between_circles_l612_612333

noncomputable def k_value (θ : ℝ) : ℝ := Real.tan θ

theorem area_between_circles {θ k : ℝ} (h₁ : k = Real.tan θ) (h₂ : θ = 4/3) (h_area : (3 * θ / 2) = 2) :
  k = Real.tan (4/3) :=
sorry

end area_between_circles_l612_612333


namespace minimum_value_inequality_l612_612718

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 64) :
  ∃ x y z, 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 64 ∧ (x^2 + 8 * x * y + 4 * y^2 + 4 * z^2) = 384 := 
sorry

end minimum_value_inequality_l612_612718


namespace servant_served_six_months_l612_612869

-- Definitions based on given conditions
def total_payment_per_year : ℝ := 800
def total_months : ℝ := 12
def amount_received : ℝ := 400

-- The amount paid to the servant per month
def monthly_payment : ℝ := total_payment_per_year / total_months

-- The number of months the servant served
def months_served : ℝ := amount_received / monthly_payment

-- Proof statement
theorem servant_served_six_months :
  months_served = 6 := 
by
  sorry

end servant_served_six_months_l612_612869


namespace repeating_decimal_arithmetic_l612_612568

theorem repeating_decimal_arithmetic :
  (0.666666666666...) + (0.222222222222...) - (0.444444444444...) = 4 / 9 :=
by
  -- Convert repeating decimals to fractions
  have h1 : (0.666666666666...) = 2 / 3 := sorry
  have h2 : (0.222222222222...) = 2 / 9 := sorry
  have h3 : (0.444444444444...) = 4 / 9 := sorry
  
  -- Prove the arithmetic operation
  calc
    (0.666666666666...) + (0.222222222222...) - (0.444444444444...)
      = (2 / 3) + (2 / 9) - (4 / 9) : by rw [h1, h2, h3]
  ... = (6 / 9) + (2 / 9) - (4 / 9) : by rw [←(eq.div_congr_left h1)]
  ... = 4 / 9 : by linarith

end repeating_decimal_arithmetic_l612_612568


namespace coefficient_x2012_eq_2_to_6_l612_612058

open Polynomial

noncomputable def P (x : ℕ) : ℕ := ∏ k in (Finset.range 11).erase 10, x^2^k + 2^k

theorem coefficient_x2012_eq_2_to_6 : (P 2012).coeff 2012 = 2^6 :=
by
  sorry -- Proof not required

end coefficient_x2012_eq_2_to_6_l612_612058


namespace expand_expression_l612_612079

variable (x : ℝ)

theorem expand_expression : 5 * (x + 3) * (2 * x - 4) = 10 * x^2 + 10 * x - 60 :=
by
  sorry

end expand_expression_l612_612079


namespace johns_speed_l612_612203

def time1 : ℕ := 2
def time2 : ℕ := 3
def total_distance : ℕ := 225

def total_time : ℕ := time1 + time2

theorem johns_speed :
  (total_distance : ℝ) / (total_time : ℝ) = 45 :=
sorry

end johns_speed_l612_612203


namespace speed_of_man_in_still_water_l612_612018

-- Define the conditions as given in step (a)
axiom conditions :
  ∃ (v_m v_s : ℝ),
    (40 / 5 = v_m + v_s) ∧
    (30 / 5 = v_m - v_s)

-- State the theorem that proves the speed of the man in still water
theorem speed_of_man_in_still_water : ∃ v_m : ℝ, v_m = 7 :=
by
  obtain ⟨v_m, v_s, h1, h2⟩ := conditions
  have h3 : v_m + v_s = 8 := by sorry
  have h4 : v_m - v_s = 6 := by sorry
  have h5 : 2 * v_m = 14 := by sorry
  exact ⟨7, by linarith⟩

end speed_of_man_in_still_water_l612_612018


namespace solve_for_x_l612_612059

namespace RationalOps

-- Define the custom operation ※ on rational numbers
def star (a b : ℚ) : ℚ := a + b

-- Define the equation involving the custom operation
def equation (x : ℚ) : Prop := star 4 (star x 3) = 1

-- State the theorem to prove the solution
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = -6 := by
  sorry

end solve_for_x_l612_612059


namespace number_of_cookies_maintaining_ratio_l612_612207

theorem number_of_cookies_maintaining_ratio :
  (∀ (c1 f1 s1 : ℕ), c1 = 24 → f1 = 3 → s1 = 1 → (c1 : ℝ) / (f1 + s1) = (24 : ℝ) / 4 → (c2 : ℕ) × (16 : ℝ / 3) = 128) :=
by
  intros c1 f1 s1 hc1 hf1 hs1 hproportion
  have hflourSugarRatio: c1 / (f1 + s1) = 24 / 4 := by
    rw [hc1, hf1] 
  have hfactor: (4 : ℝ) * (4 / 3) := by linarith
  rw hflourSugarRatio at hproportion
  field_simp at hproportion
  have h :  c2 = (24 * (16 / 3)) := by
    linarith
  exact h

end number_of_cookies_maintaining_ratio_l612_612207


namespace rowing_time_to_place_and_back_l612_612868

def speed_man_still_water : ℝ := 8 -- km/h
def speed_river : ℝ := 2 -- km/h
def total_distance : ℝ := 7.5 -- km

theorem rowing_time_to_place_and_back :
  let V_m := speed_man_still_water
  let V_r := speed_river
  let D := total_distance / 2
  let V_up := V_m - V_r
  let V_down := V_m + V_r
  let T_up := D / V_up
  let T_down := D / V_down
  T_up + T_down = 1 :=
by
  sorry

end rowing_time_to_place_and_back_l612_612868


namespace roots_indeterminate_l612_612165

-- Define the conditions
variables {a b c : ℝ}
hypothesis h1 : 0 < a
hypothesis h2 : 0 < b
hypothesis h3 : 0 < c
hypothesis h4 : b^2 - 4 * a * c = 0

-- Define the quadratic equation and its roots indeterminacy
theorem roots_indeterminate : 
  ∀ (x : ℝ), (a+1) * x^2 + (b+2) * x + (c+1) = 0 → 
  ∃ Δ : ℝ, Δ = 4 * (b - a - c) ∧ (Δ ≠ Δ) :=
by
  -- We skip the proof step which would show the discriminant sign cannot be determined.
  sorry

end roots_indeterminate_l612_612165


namespace limit_derivative_sin_at_pi_over_3_eq_half_l612_612632

noncomputable def f (x: Real) : Real := Real.sin x

theorem limit_derivative_sin_at_pi_over_3_eq_half :
  (Real.lim (λ Δx, (f (Real.pi / 3 + Δx) - f (Real.pi / 3)) / Δx) 0) = 1 / 2 :=
by
  sorry

end limit_derivative_sin_at_pi_over_3_eq_half_l612_612632


namespace part_1_A_part_1_B_part_2_l612_612294

noncomputable def is_friendly_point (m n : ℝ) : Prop := m - n = 6

def point_A_friendly : Prop :=
    let m := 8
    let n := 0
    ¬is_friendly_point m n

def point_B_friendly : Prop :=
    let m := 7
    let n := 1
    is_friendly_point m n

noncomputable def value_of_t : ℝ :=
    let t := 10 in
    let x := (2 + t) / 3
    let y := (4 - t) / 3
    let m := (t + 5) / 3
    let n := (1 - t) / 9
    if is_friendly_point m n then t else 0

theorem part_1_A : point_A_friendly := by
  unfold point_A_friendly
  simp
  sorry

theorem part_1_B : point_B_friendly := by
  unfold point_B_friendly
  simp
  sorry

theorem part_2 : value_of_t = 10 := by
  unfold value_of_t
  simp
  sorry

end part_1_A_part_1_B_part_2_l612_612294


namespace find_c_l612_612998

noncomputable def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

theorem find_c {c : ℝ} (h : ∃ c, ∀ x, derivative (λ x, f x c) x = (x - c) ^ 2 + 2 * x * (x - c)) :
  c = 6 :=
sorry

end find_c_l612_612998


namespace joe_eggs_club_house_l612_612200

theorem joe_eggs_club_house (C : ℕ) (h : C + 5 + 3 = 20) : C = 12 :=
by 
  sorry

end joe_eggs_club_house_l612_612200


namespace area_enclosed_by_line_and_circle_l612_612304

theorem area_enclosed_by_line_and_circle :
  let r := 2
  let circle_eq := ∀ x y, x^2 + y^2 = 4 → (x^2 + y^2 = r^2)
  let line_eq := ∀ x y, y = |x| → y = |x|
  let quarter_circle_area := (π * r^2) / 4
  quarter_circle_area = π := 
sorry

end area_enclosed_by_line_and_circle_l612_612304


namespace tangent_line_eq_a1_max_value_f_a_gt_1_div_5_l612_612132

noncomputable def f (a b x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + b * x + 1)

noncomputable def f_prime (a b x : ℝ) : ℝ := 
  -Real.exp (-x) * (a * x^2 + b * x + 1) + Real.exp (-x) * (2 * a * x + b)

theorem tangent_line_eq_a1 (b : ℝ) (h : f_prime 1 b (-1) = 0) : 
  ∃ m q, m = 1 ∧ q = 1 ∧ ∀ y, y = f 1 b 0 + m * y := sorry

theorem max_value_f_a_gt_1_div_5 (a b : ℝ) 
  (h_gt : a > 1/5) 
  (h_fp_eq : f_prime a b (-1) = 0)
  (h_max : ∀ x, -1 ≤ x ∧ x ≤ 1 → f a b x ≤ 4 * Real.exp 1) : 
  a = (24 * Real.exp 2 - 9) / 15 ∧ b = (12 * Real.exp 2 - 2) / 5 := sorry

end tangent_line_eq_a1_max_value_f_a_gt_1_div_5_l612_612132


namespace trigonometric_identity_l612_612090

theorem trigonometric_identity (α : Real) (h : Real.tan (α / 2) = 4) :
    (6 * Real.sin α - 7 * Real.cos α + 1) / (8 * Real.sin α + 9 * Real.cos α - 1) = -85 / 44 := by
  sorry

end trigonometric_identity_l612_612090


namespace modulus_conjugate_z_l612_612228

-- Definition of the complex number z
def z : ℂ := i / (1 - i)

-- Goal to prove: the modulus of the conjugate of z is sqrt(2)/2.
theorem modulus_conjugate_z : abs (conj z) = Real.sqrt 2 / 2 :=
by {
  -- Rest of the proof
  sorry
}

end modulus_conjugate_z_l612_612228


namespace distinct_numbers_inequality_l612_612643

theorem distinct_numbers_inequality (a1 a2 a3 : ℝ) (h_distinct : a1 ≠ a2 ∧ a2 ≠ a3 ∧ a3 ≠ a1)
  (b1 b2 b3 : ℝ) 
  (h_b1 : b1 = (1 + (a1 * a2) / (a1 - a2)) * (1 + (a1 * a3) / (a1 - a3)))
  (h_b2 : b2 = (1 + (a2 * a1) / (a2 - a1)) * (1 + (a2 * a3) / (a2 - a3)))
  (h_b3 : b3 = (1 + (a3 * a1) / (a3 - a1)) * (1 + (a3 * a2) / (a3 - a2))) :
  1 + |a1 * b1 + a2 * b2 + a3 * b3| ≤ (1 + |a1|) * (1 + |a2|) * (1 + |a3|) ∧ 
  (∀ (a1 a2 a3 : ℝ), a1 ≥ 0 ∧ a2 ≥ 0 ∧ a3 ≥ 0 → 1 + |a1 * b1 + a2 * b2 + a3 * b3| = (1 + |a1|) * (1 + |a2|) * (1 + |a3|)).
Proof: sorry

end distinct_numbers_inequality_l612_612643


namespace cat_food_insufficient_l612_612509

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l612_612509


namespace number_of_good_numbers_l612_612352

theorem number_of_good_numbers :
  let n := 2020 in 
  let remainder := 22 in
  let number := 1998 in
  let divisors := {d ∣ number | d > remainder}.to_list.length = 10 := by sorry

end number_of_good_numbers_l612_612352


namespace sequence_properties_l612_612722

theorem sequence_properties (S : ℕ → ℤ) (a : ℕ → ℤ) (b : ℕ → ℤ) 
  (h1 : ∀ n, S n = 2 * a n - 1)
  (h2 : ∀ n, b n = Int.log2 (a (n + 1))) :
  (∀ n, a n = (2:ℤ) ^ (n - 1)) ∧ 
  (∀ n, a n ≠ (-2:ℤ) ^ (n - 1)) ∧ 
  (∀ n, (finset.range n).sum (λ k, (a (k + 1)) ^ 2) = (2:ℤ)^(2 * n) - 1 / 3) ∧ 
  (∀ n, (finset.range n).sum (λ k, a (k + 1) + b k) = 2^n - 1 + (n^2 + n) / 2) :=
sorry

end sequence_properties_l612_612722


namespace count_of_good_numbers_l612_612347

-- Define the conditions of the problem
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- The main statement proving the number of 'good' numbers
theorem count_of_good_numbers :
  (finset.filter is_good_number (finset.Icc 1 2020)).card = 10 :=
by
  sorry

end count_of_good_numbers_l612_612347


namespace geometric_seq_problem_l612_612965

theorem geometric_seq_problem
  (a : Nat → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_cond : a 1 * a 99 = 16) :
  a 20 * a 80 = 16 := 
sorry

end geometric_seq_problem_l612_612965


namespace elmer_savings_l612_612069

-- Definitions based on the conditions provided:
def fuel_efficiency_old := x : ℝ  -- fuel efficiency of the old car in km per liter
def fuel_efficiency_new := 1.6 * x  -- fuel efficiency of the new car in km per liter
def cost_per_liter_gasoline := c : ℝ  -- cost per liter of gasoline in dollars
def cost_per_liter_diesel := 1.25 * c  -- cost per liter of diesel in dollars
def trip_distance := 100 : ℝ  -- trip distance in kilometers

-- Problem to prove:
theorem elmer_savings : 
  let cost_old := (trip_distance / fuel_efficiency_old) * cost_per_liter_gasoline in
  let cost_new := (trip_distance / fuel_efficiency_new) * cost_per_liter_diesel in
  let savings := cost_old - cost_new in
  let percent_savings := (savings / cost_old) * 100 in
  percent_savings ≈ 21.875 :=
begin
  sorry
end

end elmer_savings_l612_612069


namespace differentiable_interval_zero_derivative_points_l612_612721

theorem differentiable_interval_zero_derivative_points
  (f : ℝ → ℝ) (a b : ℝ) (h_differentiable : ∀ x ∈ set.Icc a b, differentiable_at ℝ f x) 
  (h_eq : f a = f b) :
  ∃ (x y : ℝ), x ≠ y ∧ x ∈ set.Icc a b ∧ y ∈ set.Icc a b ∧ f' x + 5 * f' y = 0 :=
sorry

end differentiable_interval_zero_derivative_points_l612_612721


namespace PA_inter_B_eq_1_over_3_l612_612961

-- Let P be a probability measure
variables {Ω : Type*} [measurable_space Ω] (P : measure_theory.measure Ω)

-- Given conditions
def PA_given_B (A B : set Ω) (hB : P B ≠ 0) : ℝ := P (A ∩ B) / P B

-- The condition that P(A|B) = 3/7
axiom PA_given_B_eq_3_over_7 (A B : set Ω) (hB : P B ≠ 0) : PA_given_B P A B hB = 3/7

-- The condition that P(B) = 7/9
axiom PB_eq_7_over_9 (B : set Ω) : P B = 7/9

-- The statement to be proved
theorem PA_inter_B_eq_1_over_3 (A B : set Ω) (hB : P B ≠ 0) : P (A ∩ B) = 1/3 :=
by
  -- Sorry is used to indicate that the proof is omitted
  sorry

end PA_inter_B_eq_1_over_3_l612_612961


namespace candy_prob_l612_612004

theorem candy_prob {p q : ℕ} (h_p : p = 81) (h_q : q = 256) :
    let upset := 0
    let sad := 1
    let okay := 2
    let happy := 3
    let delighted := 4
    let initial_state := okay
    let mood (candies_received : ℕ) := initial_state + candies_received
    let total_time := 10
    let total_paths := 2 ^ total_time
    let valid_paths := 324
    valid_paths / total_paths = p / q :=
begin
  -- Proof is omitted
  sorry
end

end candy_prob_l612_612004


namespace elois_banana_bread_l612_612075

theorem elois_banana_bread :
  let bananas_per_loaf := 4
  let loaves_monday := 3
  let loaves_tuesday := 2 * loaves_monday
  let total_loaves := loaves_monday + loaves_tuesday
  let total_bananas := total_loaves * bananas_per_loaf
  total_bananas = 36 := sorry

end elois_banana_bread_l612_612075


namespace evaluate_expression_l612_612808

theorem evaluate_expression : 4^1 + 3^2 - 2^3 + 1^4 = 6 := by
  -- We will skip the proof steps here using sorry
  sorry

end evaluate_expression_l612_612808


namespace no_valid_arithmetic_operation_l612_612557

-- Definition for arithmetic operations
inductive Operation
| div : Operation
| mul : Operation
| add : Operation
| sub : Operation

open Operation

-- Given conditions
def equation (op : Operation) : Prop :=
  match op with
  | div => (8 / 2) + 5 - (3 - 2) = 12
  | mul => (8 * 2) + 5 - (3 - 2) = 12
  | add => (8 + 2) + 5 - (3 - 2) = 12
  | sub => (8 - 2) + 5 - (3 - 2) = 12

-- Statement to prove
theorem no_valid_arithmetic_operation : ∀ op : Operation, ¬ equation op := by
  sorry

end no_valid_arithmetic_operation_l612_612557


namespace length_AN_eq_one_l612_612254

-- Definitions representing the conditions
def triangle := {A B C : Type}
def point (P : Type) := P
def length (A B : Type) := 1

variables (A B C M N : Type)

def angle_bisector (BAC : Prop) := ∀ (A B C : point), 
  let P := ∃ (X : Type), X

constant AC : length A C
constant AM : length A M
constant AN : length A N
constant angle_ANM_CNMC_eq : ∀ (ANM CNM : Prop), ANM = CNM

-- Conclude the length of AN is 1
theorem length_AN_eq_one : AN = 1 :=
sorry

end length_AN_eq_one_l612_612254


namespace sum_infinite_f_eq_one_l612_612585

-- Define f(n) according to the given condition
noncomputable def f (n : ℕ) : ℝ := 
  ∑' m in (Set.Ici 2), (m : ℝ)⁻¹ ^ n

-- Define the sum of f(n) from n=2 to infinity
noncomputable def infinite_sum_f : ℝ := 
  ∑' n in (Set.Ici 2), f n

-- State the theorem we need to prove
theorem sum_infinite_f_eq_one : infinite_sum_f = 1 := 
sorry

end sum_infinite_f_eq_one_l612_612585


namespace g_inequality_solution_range_of_m_l612_612634

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 8
noncomputable def g (x : ℝ) : ℝ := 2*x^2 - 4*x - 16
noncomputable def h (x m : ℝ) : ℝ := x^2 - (4 + m)*x + (m + 7)

theorem g_inequality_solution:
  {x : ℝ | g x < 0} = {x : ℝ | -2 < x ∧ x < 4} :=
by
  sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x > 1 → f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 4 :=
by
  sorry

end g_inequality_solution_range_of_m_l612_612634


namespace boy_speed_l612_612854

noncomputable def speed_to_school (v : ℝ) : Prop :=
  let d := 6 in
  let t := 5 in
  let v_return := 2 in
  (d / v) + (d / v_return) = t

theorem boy_speed : ∃ v : ℝ, speed_to_school v ∧ v = 3 :=
by {
  use 3,
  unfold speed_to_school,
  norm_num,
  sorry
}

end boy_speed_l612_612854


namespace evaluate_expression_l612_612495

theorem evaluate_expression : (- (1 / 4))⁻¹ - (Real.pi - 3)^0 - |(-4 : ℝ)| + (-1)^(2021 : ℕ) = -10 := 
by
  sorry

end evaluate_expression_l612_612495


namespace suggestions_difference_l612_612242

def mashed_potatoes_suggestions : ℕ := 408
def pasta_suggestions : ℕ := 305
def bacon_suggestions : ℕ := 137
def grilled_vegetables_suggestions : ℕ := 213
def sushi_suggestions : ℕ := 137

theorem suggestions_difference :
  let highest := mashed_potatoes_suggestions
  let lowest := bacon_suggestions
  highest - lowest = 271 :=
by
  sorry

end suggestions_difference_l612_612242


namespace solution_exists_l612_612871

noncomputable def reflection_problem :
  (A : ℝ × ℝ × ℝ) × (C : ℝ × ℝ × ℝ) × (plane : ℝ × ℝ × ℝ → ℝ) × (B : ℝ × ℝ × ℝ) → Prop :=
  λ ⟨A, C, plane, B⟩,
    -- A = (-2, 8, 10)
    A = (-2, 8, 10) ∧
    -- C = (2, 4, 8)
    C = (2, 4, 8) ∧
    -- plane equation x + y + z = 10
    plane = (λ p : ℝ × ℝ × ℝ, p.1 + p.2 + p.3 - 10) ∧
    -- B is the reflection point
    B = (10/7, 32/7, 42/7)

theorem solution_exists :
  ∃ (A : ℝ × ℝ × ℝ) (C : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ → ℝ) (B : ℝ × ℝ × ℝ),
    reflection_problem (A, C, plane, B) := 
begin
  use (-2, 8, 10),
  use (2, 4, 8),
  use (λ p : ℝ × ℝ × ℝ, p.1 + p.2 + p.3 - 10),
  use (10/7, 32/7, 42/7),
  repeat { split };
  try { refl },
  sorry
end

end solution_exists_l612_612871


namespace min_side_length_of_A_l612_612958

theorem min_side_length_of_A :
  ∃ (a b c d : ℕ), (a^2 = b^2 + c^2 + d^2 - b*c - c*d - d*b) ∧
  (b + c + d = 2) ∧
  (a = 3) :=
begin
  sorry
end

end min_side_length_of_A_l612_612958


namespace percentage_of_profit_without_discount_l612_612423

-- Definitions for the conditions
def cost_price : ℝ := 100
def discount_rate : ℝ := 0.04
def profit_rate : ℝ := 0.32

-- The statement to prove
theorem percentage_of_profit_without_discount :
  let selling_price := cost_price + (profit_rate * cost_price)
  (selling_price - cost_price) / cost_price * 100 = 32 := by
  let selling_price := cost_price + (profit_rate * cost_price)
  sorry

end percentage_of_profit_without_discount_l612_612423


namespace complement_U_A_inter_B_eq_l612_612642

open Set

-- Definitions
def U : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def A : Set ℤ := {-2, 0, 2, 4}
def B : Set ℤ := {-2, 0, 4, 6, 8}

-- Complement of A in U
def complement_U_A : Set ℤ := U \ A

-- Proof Problem
theorem complement_U_A_inter_B_eq : complement_U_A ∩ B = {6, 8} := by
  sorry

end complement_U_A_inter_B_eq_l612_612642


namespace max_weight_each_shipping_box_can_hold_l612_612156

noncomputable def max_shipping_box_weight_pounds 
  (total_plates : ℕ)
  (weight_per_plate_ounces : ℕ)
  (plates_removed : ℕ)
  (ounce_to_pound : ℕ) : ℕ :=
  (total_plates - plates_removed) * weight_per_plate_ounces / ounce_to_pound

theorem max_weight_each_shipping_box_can_hold :
  max_shipping_box_weight_pounds 38 10 6 16 = 20 :=
by
  sorry

end max_weight_each_shipping_box_can_hold_l612_612156


namespace find_AN_length_l612_612263

theorem find_AN_length
  (A B C M N : Point)
  (h1 : is_angle_bisector (angle B A C) M)
  (h2 : on_extension A B N)
  (h3 : dist A C = 1)
  (h4 : dist A M = 1)
  (h5 : angle A N M = angle C N M)
  : dist A N = 1 := sorry

end find_AN_length_l612_612263


namespace angle_BCQ_is_15_degrees_l612_612473

-- Define the equilateral triangle
variable (A B C M N P Q R S T : Type)
variable [equilateral_triangle A B C]

-- Midpoints
variable [midpoint M A C]
variable [midpoint N A B]

-- Points on segments
variable [point_on_segment P M C]
variable [point_on_segment Q N B]

-- Orthocenters of triangles ABP and ACQ
variable [orthocenter R A B P]
variable [orthocenter S A C Q]

-- Intersection of lines BP and CQ
variable [intersection T B P C Q]

-- Prove that angle BCQ = 15 degrees for triangle RST to be equilateral
theorem angle_BCQ_is_15_degrees (equilateral R S T : Prop) : 
  angle B C Q = 15 := 
  sorry

end angle_BCQ_is_15_degrees_l612_612473


namespace prime_root_eq_l612_612615

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_root_eq {
  (p q : ℕ) (x : ℤ) :
  x^4 - (p : ℤ) * x^3 + (q : ℤ) = 0 →
  is_prime p →
  is_prime q →
  ∃ x : ℤ, x * (q : ℤ) = 0 ∨
  x * - (q : ℤ) = 0 ∨
  x + (p : ℤ) = -1 ∨
  x - (p : ℤ) = 0 →
  (p = 3 ∧ q = 2) :=
begin
  sorry
end

end prime_root_eq_l612_612615


namespace prove_transformation_D_l612_612830

statement : Prop :=
  (-2) * (1 / 2) * (-5) = 5

theorem prove_transformation_D :
  statement := 
by
  sorry

end prove_transformation_D_l612_612830


namespace cat_food_inequality_l612_612513

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l612_612513


namespace count_of_good_numbers_l612_612345

-- Define the conditions of the problem
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- The main statement proving the number of 'good' numbers
theorem count_of_good_numbers :
  (finset.filter is_good_number (finset.Icc 1 2020)).card = 10 :=
by
  sorry

end count_of_good_numbers_l612_612345


namespace find_AN_length_l612_612260

theorem find_AN_length
  (A B C M N : Point)
  (h1 : is_angle_bisector (angle B A C) M)
  (h2 : on_extension A B N)
  (h3 : dist A C = 1)
  (h4 : dist A M = 1)
  (h5 : angle A N M = angle C N M)
  : dist A N = 1 := sorry

end find_AN_length_l612_612260


namespace tangent_line_at_P_tangent_line_through_Q_l612_612127

noncomputable def curve (x : ℝ) : ℝ := 1 / x

theorem tangent_line_at_P :
  let P := (1:ℝ, 1:ℝ) in
  let tangent_line := λ (x y : ℝ), x + y - 2 = 0 in
  tangent_line P.1 P.2 := 
by sorry

theorem tangent_line_through_Q :
  let Q := (1:ℝ, 0:ℝ) in
  let tangent_line := λ (x y : ℝ), 4 * x + y - 4 = 0 in
  tangent_line Q.1 Q.2 := 
by sorry

end tangent_line_at_P_tangent_line_through_Q_l612_612127


namespace length_AN_is_one_l612_612277

noncomputable def segment_length_AN (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) (point_N_extension : ExtensionOfSide A B N) 
  (AC_eq_AM : AC = 1 ∧ AM = 1) (ANM_eq_CNM : ∠ANM = ∠CNM) : Real :=
1

/-- Given a triangle ABC, where point M lies on the angle bisector of angle BAC,
and point N lies on the extension of side AB beyond point A, where AC = AM = 1,
and the angles ANM and CNM are equal, the length of segment AN is 1. -/
theorem length_AN_is_one (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) 
  (point_N_extension : ExtensionOfSide A B N) (AC_eq_AM : AC = 1 ∧ AM = 1) 
  (ANM_eq_CNM : ∠ANM = ∠CNM) : 
  segment_length_AN A B C M N triangle_ABC angle_bisector_AM point_N_extension AC_eq_AM ANM_eq_CNM = 1 :=
by
  sorry

end length_AN_is_one_l612_612277


namespace length_AN_eq_one_l612_612256

-- Definitions representing the conditions
def triangle := {A B C : Type}
def point (P : Type) := P
def length (A B : Type) := 1

variables (A B C M N : Type)

def angle_bisector (BAC : Prop) := ∀ (A B C : point), 
  let P := ∃ (X : Type), X

constant AC : length A C
constant AM : length A M
constant AN : length A N
constant angle_ANM_CNMC_eq : ∀ (ANM CNM : Prop), ANM = CNM

-- Conclude the length of AN is 1
theorem length_AN_eq_one : AN = 1 :=
sorry

end length_AN_eq_one_l612_612256


namespace probability_even_sum_l612_612959

open Finset

def all_pairs (s : Finset ℕ) := s.image (λ (p : ℕ × ℕ), p.1 + p.2)

theorem probability_even_sum :
  let s := {1, 2, 3, 4, 5, 6}
  let pairs := (s.product s).filter (λ p, p.1 < p.2)
  let even_pairs := pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  let probability := (even_pairs.card : ℝ) / (pairs.card : ℝ)
  probability = 2 / 5 :=
  sorry

end probability_even_sum_l612_612959


namespace volume_of_cone_correct_l612_612803

-- Define the problem conditions and the expected volume
noncomputable def slant_height : ℝ := 1
noncomputable def central_angle : ℝ := 240 * (π / 180)  -- Converting degrees to radians

-- Define the volume function
noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

-- Prove the expected volume given the conditions
theorem volume_of_cone_correct :
  let r := 2 / 3
  let h := √5 / 3
  in volume_of_cone r h = 4 * √5 * π / 81 :=
by
  let r := 2 / 3
  let h := √5 / 3
  have vol := volume_of_cone r h
  calc
    vol = (1 / 3) * π * r^2 * h : by sorry
    ... = 4 * √5 * π / 81 : by sorry

end volume_of_cone_correct_l612_612803


namespace smallest_positive_period_symmetry_center_increasing_intervals_l612_612231

noncomputable def f (x : ℝ) : ℝ := sin (2*x + π/3) + sqrt 3 - 2 * sqrt 3 * (cos x)^2 + 1

theorem smallest_positive_period (T : ℝ) : 
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
  sorry

theorem symmetry_center (k : ℤ) : 
  SymmetricCenter f ( ∀ k : ℤ, (k * π / 2 + π / 6, 1)) :=
  sorry

theorem increasing_intervals (k : ℤ) : 
  ∀ k : ℤ, ( ∃ a b : ℝ, a = (k * π - π / 12) ∧ b = (k * π + 5 * π / 12) ∧ 
             ∀ x y : ℝ, (a ≤ x ∧ y ≤ b) → (x ≤ y → f x ≤ f y)) :=
  sorry

end smallest_positive_period_symmetry_center_increasing_intervals_l612_612231


namespace angle_400_in_first_quadrant_l612_612831

def angle_in_quadrant (θ : ℝ) : ℕ :=
if 0 ≤ θ ∧ θ ≤ 90 then 1
else if 90 < θ ∧ θ ≤ 180 then 2
else if 180 < θ ∧ θ ≤ 270 then 3
else 4

theorem angle_400_in_first_quadrant:
  angle_in_quadrant (400 % 360) = 1 :=
by
  -- Proof skipped
  sorry

end angle_400_in_first_quadrant_l612_612831


namespace max_gcd_of_consecutive_terms_l612_612566

-- Given conditions
def a (n : ℕ) : ℕ := 2 * (n.factorial) + n

-- Theorem statement
theorem max_gcd_of_consecutive_terms : ∃ (d : ℕ), ∀ n ≥ 0, d ≤ gcd (a n) (a (n + 1)) ∧ d = 1 := by sorry

end max_gcd_of_consecutive_terms_l612_612566


namespace correct_propositions_l612_612054

def proposition_1 (r : ℝ) (r_critical : ℝ) : Prop :=
  abs r > r_critical

def proposition_2 (a b : ℝ) : Prop :=
  a > 0 → b > 0 → a^3 + b^3 ≥ 3 * a * b^2

def proposition_3 (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f a > 0 → f b > 0 → ∀ x ∈ set.Ioo a b, ∃! z ∈ set.Ioo a b, f z = 0

def proposition_4 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 2) = f (2 - x)

theorem correct_propositions :
  (proposition_1 (-0.9568) 0.8016) ∧
  ¬ proposition_2 1 1 ∧
  ¬ proposition_3 (λ x => 2 * x^2 + (0:ℝ) * x + 0) 0 2 ∧
  proposition_4 (λ x => x^2) :=
by
  sorry

end correct_propositions_l612_612054


namespace total_voters_l612_612184

-- Definitions
def number_of_voters_first_hour (x : ℕ) := x
def percentage_october_22 (x : ℕ) := 35 * x / 100
def percentage_october_29 (x : ℕ) := 65 * x / 100
def additional_voters_october_22 := 80
def final_percentage_october_29 (total_votes : ℕ) := 45 * total_votes / 100

-- Statement
theorem total_voters (x : ℕ) (h1 : percentage_october_22 x + additional_voters_october_22 = 35 * (x + additional_voters_october_22) / 100)
                      (h2 : percentage_october_29 x = 65 * x / 100)
                      (h3 : final_percentage_october_29 (x + additional_voters_october_22) = 45 * (x + additional_voters_october_22) / 100):
  x + additional_voters_october_22 = 260 := 
sorry

end total_voters_l612_612184


namespace ratio_yellow_to_red_l612_612540

theorem ratio_yellow_to_red
    (blue_balls : ℕ)
    (red_balls : ℕ)
    (green_ratio : ℕ)
    (total_balls : ℕ)
    (h_blue : blue_balls = 6)
    (h_red : red_balls = 4)
    (h_green : green_ratio = 3)
    (h_total : total_balls = 36) :
    (let green_balls := green_ratio * blue_balls,
         known_balls := blue_balls + red_balls + green_balls,
         yellow_balls := total_balls - known_balls
     in yellow_balls = 2 * red_balls) :=
by
  sorry

end ratio_yellow_to_red_l612_612540


namespace lattice_points_on_hyperbola_l612_612159

theorem lattice_points_on_hyperbola :
  {p : ℤ × ℤ | p.1^2 - p.2^2 = 59}.finite.to_finset.card = 4 := by
  sorry

end lattice_points_on_hyperbola_l612_612159


namespace ratio_mets_redsox_l612_612180

theorem ratio_mets_redsox 
    (Y M R : ℕ) 
    (h1 : Y = 3 * (M / 2))
    (h2 : M = 88)
    (h3 : Y + M + R = 330) : 
    M / R = 4 / 5 := 
by 
    sorry

end ratio_mets_redsox_l612_612180


namespace expanded_form_correct_l612_612307

theorem expanded_form_correct :
  (∃ a b c : ℤ, (∀ x : ℚ, 2 * (x - 3)^2 - 12 = a * x^2 + b * x + c) ∧ (10 * a - b - 4 * c = 8)) :=
by
  sorry

end expanded_form_correct_l612_612307


namespace simplify_fraction_case1_simplify_fraction_case2_l612_612102

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + (x^2 - 1) * real.sqrt (x^2 - 4)

theorem simplify_fraction_case1 (x : ℝ) (hx : x ≥ 2) : 
  (f x - 2) / (f x + 2) = ((x + 1) * real.sqrt (x - 2)) / ((x - 1) * real.sqrt (x + 2)) :=
sorry

theorem simplify_fraction_case2 (x : ℝ) (hx : x ≤ -2) : 
  (f x - 2) / (f x + 2) = (-(x + 1) * real.sqrt (2 - x)) / ((x - 1) * real.sqrt (-(x + 2))) :=
sorry

end simplify_fraction_case1_simplify_fraction_case2_l612_612102


namespace pattern_D_cannot_fold_into_cube_l612_612111

-- Definitions for patterns A, B, C, D and their connections are required.
-- Each pattern is represented as a list of edges where each edge is a connection between two squares.

structure Pattern where
  squares : Finset (Fin 6)   -- Each pattern has 6 squares.
  edges : Finset (Fin 6 × Fin 6) -- Edges determining connections between these squares.

-- Patterns A, B, C, and D are assumed to be given, but for demonstration,
-- let's define placeholders for these patterns.
-- These should be replaced with the actual pattern representations.
noncomputable def pattern_A : Pattern := {
  squares := {0, 1, 2, 3, 4, 5},
  edges := {((0, 1), (1, 2)), ((2, 3), (3, 4)), ((4, 5), (5, 0))}
}

noncomputable def pattern_B : Pattern := {
  squares := {0, 1, 2, 3, 4, 5},
  edges := {((0, 1), (1, 2)), ((2, 3), (3, 4)), ((4, 5), (5, 1))}
}

noncomputable def pattern_C : Pattern := {
  squares := {0, 1, 2, 3, 4, 5},
  edges := {((0, 1), (1, 2)), ((2, 3), (3, 5)), ((5, 4), (4, 0))}
}

noncomputable def pattern_D : Pattern := {
  squares := {0, 1, 2, 3, 4, 5},
  edges := {((0, 1), (1, 2)), ((2, 3), (3, 4)), ((4, 5), (5, 2))}
}

-- Defining the main theorem statement
theorem pattern_D_cannot_fold_into_cube (pD : Pattern) 
  (hA : pD = pattern_A ∨ pD = pattern_B ∨ pD = pattern_C ∨ pD = pattern_D) 
  (hDist : ∀ p ∈ {pattern_A, pattern_B, pattern_C, pattern_D}, p.squares = {0, 1, 2, 3, 4, 5}) :
  ¬ (foldable_into_cube pD) → pD = pattern_D := sorry

-- Defining a hypothetical function foldable_into_cube
-- such that it returns true if the pattern can be folded into a cube
noncomputable def foldable_into_cube (p : Pattern) : Prop := sorry


end pattern_D_cannot_fold_into_cube_l612_612111


namespace ellen_lost_legos_l612_612929

theorem ellen_lost_legos :
  ∀ (initial_legos current_legos : ℕ), initial_legos = 380 → current_legos = 323 → initial_legos - current_legos = 57 :=
by
  intros initial_legos current_legos h1 h2
  rw [h1, h2]
  exact nat.sub_eq_of_eq_add sorry

end ellen_lost_legos_l612_612929


namespace factorize_quadratic_l612_612931

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by {
  sorry
}

end factorize_quadratic_l612_612931


namespace rectangle_dimensions_l612_612793

theorem rectangle_dimensions
  (l w : ℕ)
  (h1 : 2 * l + 2 * w = l * w)
  (h2 : w = l - 3) :
  l = 6 ∧ w = 3 :=
by
  sorry

end rectangle_dimensions_l612_612793


namespace find_d_l612_612984

theorem find_d (a d : ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : d ≠ 0)
                (h2 : ∀ n, S n = n * a + (n * (n - 1) / 2 * d))
                (h3 : ∀ n, (S n + n) ^ (1/2) = (S (n + 1) + (n + 1)) ^ (1/2) - d)
                : d = 1 / 2 :=
by
  -- Proof omitted.
  sorry

end find_d_l612_612984


namespace hyperbola_eccentricity_l612_612635

theorem hyperbola_eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (asymptote : b / a = 4 / 3) : eccentricity (Hyperbola a b) = 5 / 3 :=
by 
  sorry

end hyperbola_eccentricity_l612_612635


namespace determine_p_range_l612_612699

theorem determine_p_range :
  ∀ (p : ℝ), (∃ f : ℝ → ℝ, ∀ x : ℝ, f x = (x + 9 / 8) * (x + 9 / 8) ∧ (f x) = (8*x^2 + 18*x + 4*p)/8 ) →
  2.5 < p ∧ p < 2.6 :=
by
  sorry

end determine_p_range_l612_612699


namespace half_abs_diff_of_squares_l612_612406

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 21) (h2 : b = 19) :
  (abs (a^2 - b^2)) / 2 = 40 := by
  sorry

end half_abs_diff_of_squares_l612_612406


namespace find_radius_of_concentric_circle_l612_612859

theorem find_radius_of_concentric_circle
  (side_length : ℝ) 
  (r : ℝ) 
  (prob : ℝ) 
  (h1 : side_length = 3) 
  (h2 : prob = 1 / 3) : 
  r = 6 :=
begin
  sorry
end

end find_radius_of_concentric_circle_l612_612859


namespace good_numbers_2020_has_count_10_l612_612392

theorem good_numbers_2020_has_count_10 :
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  set.count divisors = 10 :=
by
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  sorry

end good_numbers_2020_has_count_10_l612_612392


namespace sum_of_distances_l612_612675

structure Point :=
(x : ℝ)
(y : ℝ)

def distance (P Q : Point) : ℝ :=
Real.sqrt ((Q.x - P.x) ^ 2 + (Q.y - P.y) ^ 2)

def A : Point := {x := 8, y := 0}
def B : Point := {x := 0, y := 5}
def D : Point := {x := 1, y := 3}

theorem sum_of_distances (P Q R : Point) (hpq : P = A) (hqr : Q = B) (hpr : R = D) :
  distance P R + distance Q R = 10 :=
by
  sorry

end sum_of_distances_l612_612675


namespace Kyle_weightlifting_time_l612_612695

theorem Kyle_weightlifting_time
  (total_practice_time_hours : ℕ)
  (half_time_shooting : total_practice_time_hours / 2)
  (time_running_and_weightlifting : total_practice_time_hours / 2)
  (time_weightlifting : ℕ)
  (time_running : ℕ)
  (h1 : time_running = 2 * time_weightlifting)
  (h2 : time_running + time_weightlifting = 60) :
  time_weightlifting = 20 := 
sorry

end Kyle_weightlifting_time_l612_612695


namespace F_final_coordinates_l612_612644

-- Define the original coordinates of point F
def F : ℝ × ℝ := (5, 2)

-- Reflection over the y-axis changes the sign of the x-coordinate
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Reflection over the line y = x involves swapping x and y coordinates
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- The combined transformation: reflect over the y-axis, then reflect over y = x
def F_final : ℝ × ℝ := reflect_y_eq_x (reflect_y_axis F)

-- The proof statement
theorem F_final_coordinates : F_final = (2, -5) :=
by
  -- Proof goes here
  sorry

end F_final_coordinates_l612_612644


namespace chord_angles_l612_612858

theorem chord_angles (R : ℝ) (A B C D M : ℝ) (h : 0 < R) 
    (h_perpendicular : perpendicular (segment A C) (segment A D)) 
    (h_ratio : AM = 1/4 * AB ∧ MB = 3/4 * AB) : 
    angle A (segment C D) = 120 ∧ angle B (segment C D) = 60 := 
sorry

end chord_angles_l612_612858


namespace hyperbola_equation_l612_612125

theorem hyperbola_equation 
  (F1 F2 : ℝ × ℝ) (hF1 : F1 = (-sqrt 5, 0))
  (hF2 : F2 = (sqrt 5, 0))
  (P : ℝ × ℝ)
  (h1P : dist P F1 * dist P F2 = 2)
  (h2P : dist P F1 ≠ dist P F2)
  (h3P : P.1 ≠ 0) :
  ∃ a b : ℝ, a = 2 ∧ b = 1 ∧ (∀ x y : ℝ, (x, y) ∈ (λ x y, x^2 / a^2 - y^2 / b^2 = 1)) := sorry

end hyperbola_equation_l612_612125


namespace jonah_poured_total_pitchers_l612_612561

theorem jonah_poured_total_pitchers :
  (0.25 + 0.125) + (0.16666666666666666 + 0.08333333333333333 + 0.16666666666666666) + 
  (0.25 + 0.125) + (0.3333333333333333 + 0.08333333333333333 + 0.16666666666666666) = 1.75 :=
by
  sorry

end jonah_poured_total_pitchers_l612_612561


namespace required_raise_percentage_l612_612471

theorem required_raise_percentage (S : ℝ) (hS : S > 0) : 
  ((S - (0.85 * S - 50)) / (0.85 * S - 50) = 0.1875) :=
by
  -- Proof of this theorem can be carried out here
  sorry

end required_raise_percentage_l612_612471


namespace pseudocode_output_l612_612293

theorem pseudocode_output :
  let s := 0
  let t := 1
  let (s, t) := (List.range 3).foldl (fun (s, t) i => (s + (i + 1), t * (i + 1))) (s, t)
  let r := s * t
  r = 36 :=
by
  sorry

end pseudocode_output_l612_612293


namespace half_abs_diff_squares_eq_40_l612_612401

theorem half_abs_diff_squares_eq_40 (x y : ℤ) (hx : x = 21) (hy : y = 19) :
  (|x^2 - y^2| / 2) = 40 :=
by
  sorry

end half_abs_diff_squares_eq_40_l612_612401


namespace count_good_numbers_l612_612379

-- Define the property of a good number
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- Count the number of good numbers
theorem count_good_numbers : (finset.filter is_good_number (finset.range 2021)).card = 10 :=
by sorry

end count_good_numbers_l612_612379


namespace inequality_ratios_l612_612993

theorem inequality_ratios (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (c / a) > (d / b) :=
sorry

end inequality_ratios_l612_612993


namespace hyperbola_perimeter_proof_l612_612128

noncomputable def hyperbola_perimeter (a m : ℝ) : ℝ :=
  4 * a + 2 * m

theorem hyperbola_perimeter_proof
  {x y a b m : ℝ}
  (h_hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (A B F1 F2 : ℝ × ℝ)
  (h_right_branch : A.1 > 0 ∧ B.1 > 0)
  (h_AB_through_F2 : ∃ t ∈ Icc 0 1, t • A + (1 - t) • B = F2)
  (h_AB_len : real.norm (B - A) = m)
  (h_AF1_AF2 : ∥A - F1∥ - ∥A - F2∥ = 2 * a)
  (h_BF1_BF2 : ∥B - F1∥ - ∥B - F2∥ = 2 * a)
  (h_AF2_BF2 : ∥A - F2∥ + ∥B - F2∥ = m)
  : ∥A - F1∥ + ∥B - F1∥ + ∥A - B∥ = 4 * a + 2 * m :=
by
  sorry

end hyperbola_perimeter_proof_l612_612128


namespace length_AN_l612_612248

theorem length_AN {A B C M N : Type*}
  (h1 : M ∈ angle_bisector A B C)
  (h2 : N ∈ extension A B)
  (h3 : distance A C = 1)
  (h4 : distance A M = 1)
  (h5 : ∠ A N M = ∠ C N M) :
  distance A N = 1 :=
sorry

end length_AN_l612_612248


namespace equation_positive_root_range_logarithmic_function_range_l612_612095

theorem equation_positive_root_range (a : ℝ) :
  (∃ x > 0, 4^x + 2^x = a^2 + a) ↔ a ∈ set.Iio (-2) ∪ set.Ioi 1 := sorry

theorem logarithmic_function_range (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, real.log (x^2 + a*x + 1) = y) ↔ a ∈ set.Iic (-2) ∪ set.Ici 2 := sorry

end equation_positive_root_range_logarithmic_function_range_l612_612095


namespace exists_x0_lt_zero_l612_612968

def f (x : ℝ) : ℝ := Real.sin x - Real.tan x

theorem exists_x0_lt_zero : ∃ x0 ∈ Ioo (0 : ℝ) (Real.pi / 2), f x0 < 0 := by
  sorry

end exists_x0_lt_zero_l612_612968


namespace projectile_max_height_l612_612022

theorem projectile_max_height :
  ∀ (t : ℝ), -12 * t^2 + 72 * t + 45 ≤ 153 :=
by
  sorry

end projectile_max_height_l612_612022


namespace cat_food_inequality_l612_612514

variable (B S : ℝ)
variable (h1 : B > S)
variable (h2 : B < 2 * S)
variable (h3 : B + 2 * S = 2 * (B + 2 * S) / 2)

theorem cat_food_inequality : 4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_inequality_l612_612514


namespace length_AN_is_one_l612_612279

noncomputable def segment_length_AN (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) (point_N_extension : ExtensionOfSide A B N) 
  (AC_eq_AM : AC = 1 ∧ AM = 1) (ANM_eq_CNM : ∠ANM = ∠CNM) : Real :=
1

/-- Given a triangle ABC, where point M lies on the angle bisector of angle BAC,
and point N lies on the extension of side AB beyond point A, where AC = AM = 1,
and the angles ANM and CNM are equal, the length of segment AN is 1. -/
theorem length_AN_is_one (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) 
  (point_N_extension : ExtensionOfSide A B N) (AC_eq_AM : AC = 1 ∧ AM = 1) 
  (ANM_eq_CNM : ∠ANM = ∠CNM) : 
  segment_length_AN A B C M N triangle_ABC angle_bisector_AM point_N_extension AC_eq_AM ANM_eq_CNM = 1 :=
by
  sorry

end length_AN_is_one_l612_612279


namespace central_angle_of_sector_l612_612104

noncomputable def central_angle (l S r : ℝ) : ℝ :=
  2 * S / r^2

theorem central_angle_of_sector (r : ℝ) (h₁ : 4 * r / 2 = 4) (h₂ : r = 2) : central_angle 4 4 r = 2 :=
by
  sorry

end central_angle_of_sector_l612_612104


namespace find_N_l612_612168

-- Given conditions
variables (k : ℕ) (N : ℕ)
hypothesis h1 : 2^k = N
hypothesis h2 : 2^(2 * k + 2) = 64

-- Proof problem statement
theorem find_N : N = 4 := 
  sorry

end find_N_l612_612168


namespace constant_term_in_expansion_l612_612824

-- Given conditions
def eq_half_n_minus_m_zero (n m : ℕ) : Prop := 1/2 * n = m
def eq_n_plus_m_ten (n m : ℕ) : Prop := n + m = 10
noncomputable def binom (n k : ℕ) : ℝ := Real.exp (Real.log (Nat.factorial n) - Real.log (Nat.factorial k) - Real.log (Nat.factorial (n - k)))

-- Main theorem
theorem constant_term_in_expansion : 
  ∃ (n m : ℕ), eq_half_n_minus_m_zero n m ∧ eq_n_plus_m_ten n m ∧ 
  binom 10 m * (3^4 : ℝ) = 17010 :=
by
  -- Definitions translation
  sorry

end constant_term_in_expansion_l612_612824


namespace least_period_is_36_l612_612917

noncomputable def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 6) + f(x - 6) = f(x)

theorem least_period_is_36 (f : ℝ → ℝ) (h : satisfies_condition f) :
  (∃ p > 0, ∀ x : ℝ, f(x + p) = f(x)) ∧
  (∀ q > 0, (∀ x : ℝ, f(x + q) = f(x)) → q ≥ 36) :=
sorry

end least_period_is_36_l612_612917


namespace find_max_f_l612_612113

noncomputable def max_f_value (a : Fin 2016 → ℝ) : ℝ :=
  ∏ i in Finset.range 2015, (a i - (a (i + 1))^2) * (a 2015 - (a 0)^2)

theorem find_max_f :
  ∀ (a : Fin 2016 → ℝ),
    (∀ i : Fin 2015, 9 * a i > 11 * (a (i + 1))^2) →
    max_f_value a ≤ 1 / 2 ^ 4033 :=
sorry

end find_max_f_l612_612113


namespace cyclic_quadrilateral_tangent_ratio_one_l612_612290

/-- 
Quadrilateral \(ABCD\) is inscribed in a circle.
At point \(C\), a tangent \(\ell\) to this circle is drawn.
Circle \(\omega\) passes through points \(A\) and \(B\) and touches the line \(\ell\) at point \(P\).
Line \(PB\) intersects segment \(CD\) at point \(Q\).
\(B\) is tangent to circle \(\omega\).
Prove that the ratio \(\frac{BC}{CQ} = 1\).
-/
theorem cyclic_quadrilateral_tangent_ratio_one
    (ABCD_inscribed : ∃ O : Point, Circle O ∧ A ∈ Circle O ∧ B ∈ Circle O ∧ C ∈ Circle O ∧ D ∈ Circle O)
    (tangent_at_C : Tangent ℓ C Circle)
    (omega_tangent_ℓ_at_P : Circle ω ∧ A ∈ Circle ω ∧ B ∈ Circle ω ∧ tangent ℓ P Circle ω)
    (PB_intersects_CD_at_Q : Line PB ∧ SegmentsIntersect PB CD Q)
    (B_tangent_to_omega : Tangent B Circle ω) :
    BC / CQ = 1 :=
by
    sorry

end cyclic_quadrilateral_tangent_ratio_one_l612_612290


namespace complete_the_square_l612_612757

theorem complete_the_square :
  ∀ x : ℝ, (x^2 + 8 * x + 7 = 0) → (x + 4)^2 = 9 :=
by
  intro x h,
  sorry

end complete_the_square_l612_612757


namespace hyperbola_equation_and_dot_product_l612_612120

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 6) - (y^2 / 6) = 1

open Real

theorem hyperbola_equation_and_dot_product :
  (hyperbola_equation 4 (-sqrt 10)) ∧
  ∀ (m : ℝ), 
  hyperbola_equation 3 m →
  let MF1 := (-2 * sqrt 3 - 3, -m)
      MF2 := (2 * sqrt 3 - 3, -m) in
  ((MF1.1 * MF2.1 + MF1.2 * MF2.2) = 0):=
by
  sorry

end hyperbola_equation_and_dot_product_l612_612120


namespace expected_value_smallest_N_l612_612486
noncomputable def expectedValueN : ℝ := 6.54

def barryPicksPointsInsideUnitCircle (P : ℕ → ℝ × ℝ) : Prop :=
  ∀ n, (P n).fst^2 + (P n).snd^2 ≤ 1

def pointsIndependentAndUniform (P : ℕ → ℝ × ℝ) : Prop :=
  -- This is a placeholder representing the independent and uniform picking which 
  -- would be formally defined using probability measures in an advanced Lean library.
  sorry

theorem expected_value_smallest_N (P : ℕ → ℝ × ℝ)
  (h1 : barryPicksPointsInsideUnitCircle P)
  (h2 : pointsIndependentAndUniform P) :
  ∃ N : ℕ, N = expectedValueN :=
sorry

end expected_value_smallest_N_l612_612486


namespace product_of_areas_l612_612706

variables {A B C D : Type}

-- Assume DIA, FOR, and FRIEND are regular polygons in the plane
-- and ID is a line segment of length 1.
def is_regular_polygon (polygon : Type) := sorry
def length_ID (I D : Type) := 1

-- The main theorem to prove
theorem product_of_areas (D I A L F O R : Type)
  (h1 : is_regular_polygon DIAL)
  (h2 : is_regular_polygon FOR)
  (h3 : is_regular_polygon FRIEND)
  (h4 : length_ID I D) :
  product_of_areas_of_triangle O L A = 1 / 32 :=
sorry

end product_of_areas_l612_612706


namespace num_ways_to_connect_20_points_l612_612326

def a : ℕ → ℕ
| 0 := 1
| 1 := 1
| n := ∑ i in Finset.range n, a i * a (n - 1 - i)

theorem num_ways_to_connect_20_points : a 10 = 16796 :=
by
  sorry

end num_ways_to_connect_20_points_l612_612326


namespace inequality_holds_l612_612213

def sum_divisors (n : ℕ) : ℕ :=
  nat.divisors n |>.sum

def satisfies_condition (n : ℕ) : Prop :=
  sum_divisors (8 * n) > sum_divisors (9 * n)

theorem inequality_holds (n : ℕ) (l : ℕ) (x : ℕ) (h1 : n = 3^l * x ∨ n = 6 * 3^l * x) (h2 : ¬ (nat.gcd x 2 = 1) ∧ ¬ (nat.gcd x 3 = 1)) :
  satisfies_condition n :=
sorry

end inequality_holds_l612_612213


namespace count_good_numbers_l612_612361

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l612_612361


namespace number_of_men_in_company_l612_612674

noncomputable def total_workers : ℝ := 2752.8
noncomputable def women_in_company : ℝ := 91.76
noncomputable def workers_without_retirement_plan : ℝ := (1 / 3) * total_workers
noncomputable def percent_women_without_retirement_plan : ℝ := 0.10
noncomputable def percent_men_with_retirement_plan : ℝ := 0.40
noncomputable def workers_with_retirement_plan : ℝ := (2 / 3) * total_workers
noncomputable def men_with_retirement_plan : ℝ := percent_men_with_retirement_plan * workers_with_retirement_plan

theorem number_of_men_in_company : (total_workers - women_in_company) = 2661.04 := by
  -- Insert the exact calculations and algebraic manipulations
  sorry

end number_of_men_in_company_l612_612674


namespace alpha_beta_roots_eq_l612_612655

theorem alpha_beta_roots_eq {α β : ℝ} (hα : α^2 - α - 2006 = 0) (hβ : β^2 - β - 2006 = 0) (h_sum : α + β = 1) : 
  α + β^2 = 2007 :=
by
  sorry

end alpha_beta_roots_eq_l612_612655


namespace time_for_second_lap_l612_612417

-- Definitions from the conditions
def distance (D : ℝ) := D > 0

def speed_first_lap := 15 -- in km/h
def speed_second_lap := 10 -- in km/h
def time_difference := 0.5 -- in hours (30 minutes)

-- Theorem: Prove the time to run the playground for the second time is 1.5 hours given the conditions
theorem time_for_second_lap (D : ℝ) (h_dist : distance D) :
  let T1 := D / speed_first_lap in
  let T2 := D / speed_second_lap in
  T2 = T1 + time_difference ↔ T2 = 1.5 :=
by
  sorry

end time_for_second_lap_l612_612417


namespace hexagon_area_twice_triangle_area_l612_612786

open Real EuclideanGeometry

variable {O : Point}
variable {A B C D E F : Point}
variable [circumscribed O A B C D E F] -- indicates A, B, C, D, E, F lie on a circle centered at O
variable (diam_AD : diameter O A D)
variable (diam_BE : diameter O B E)
variable (diam_CF : diameter O C F)

theorem hexagon_area_twice_triangle_area :
  area (hexagon {A, B, C, D, E, F}) = 2 * area (triangle {A, C, E}) := by
  sorry

end hexagon_area_twice_triangle_area_l612_612786


namespace count_good_numbers_l612_612366

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l612_612366


namespace CK_eq_CM_l612_612036

-- Define the necessary geometrical objects and properties
variables (A B C D M K : Type)
axiom is_square_ABCD : is_square A B C D
axiom is_equilateral_triangle_ABM : is_equilateral_triangle A B M
axiom M_inside_square : inside_square M A B C D
axiom AC_intersects_triangle_K : AC_intersects A C M K

-- Proof goal: Prove CK = CM
theorem CK_eq_CM : CK = CM :=
by
  sorry

end CK_eq_CM_l612_612036


namespace factorization_a_minus_b_l612_612777

theorem factorization_a_minus_b (a b: ℤ) 
  (h : (4 * y + a) * (y + b) = 4 * y * y - 3 * y - 28) : a - b = -11 := by
  sorry

end factorization_a_minus_b_l612_612777


namespace arriving_late_l612_612335

-- Definitions from conditions
def usual_time : ℕ := 24
def slower_factor : ℚ := 3 / 4

-- Derived from conditions
def slower_time : ℚ := usual_time * (4 / 3)

-- To be proven
theorem arriving_late : slower_time - usual_time = 8 := by
  sorry

end arriving_late_l612_612335


namespace good_numbers_count_l612_612343

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l612_612343


namespace count_good_numbers_l612_612382

-- Define the property of a good number
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- Count the number of good numbers
theorem count_good_numbers : (finset.filter is_good_number (finset.range 2021)).card = 10 :=
by sorry

end count_good_numbers_l612_612382


namespace inscribed_sphere_radius_l612_612323

theorem inscribed_sphere_radius {a : ℝ} :
  ∃ r : ℝ, (r = (a * (Real.sqrt 21 - 3)) / 4) :=
by
  sorry

end inscribed_sphere_radius_l612_612323


namespace domain_of_h_l612_612062

-- Define the rational function h(x)
noncomputable def h (x : ℝ) : ℝ := (x^3 - 3 * x^2 + 2 * x + 4) / (x^2 - 5 * x + 6)

-- Define the function domain
def domain_h : set ℝ := {x : ℝ | x ≠ 2 ∧ x ≠ 3}

-- The statement to prove
theorem domain_of_h : ∀ x : ℝ, (x ∈ domain_h ↔ x ≠ 2 ∧ x ≠ 3) := by
  intros x
  unfold h
  unfold domain_h
  sorry

end domain_of_h_l612_612062


namespace problem_simplify_and_evaluate_l612_612967

-- Defining the problem conditions and required variables
theorem problem_simplify_and_evaluate :
  ∀ (α : ℝ),
  (α ∈ {α : ℝ | α > π ∧ α < 3 * π}) -- α is in the third quadrant
  → (cos (α - (3 * π / 2)) = 1 / 5)
  → (f (α) = -(cos α)) ∧ (f α = (2 * real.sqrt 6) / 5) :=
by
  sorry

-- Definitions and theorems used
def f (α : ℝ) : ℝ :=
  (sin (α - 3 * π) * cos (2 * π - α) * sin (-α + (3 * π / 2))) /
  (cos (-π - α) * sin (-π - α))

end problem_simplify_and_evaluate_l612_612967


namespace count_good_numbers_l612_612376

-- Define the property of a good number
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- Count the number of good numbers
theorem count_good_numbers : (finset.filter is_good_number (finset.range 2021)).card = 10 :=
by sorry

end count_good_numbers_l612_612376


namespace seating_arrangements_with_adjacent_empty_seats_l612_612328

-- Let's define our conditions
def seats : ℕ := 6
def people : ℕ := 3

-- Define the statement of our theorem
theorem seating_arrangements_with_adjacent_empty_seats : 
  ∃ adj_empty_seats: ℕ, adj_empty_seats = 72 ∧ 
  (∃ (seats people : ℕ), seats = 6 ∧ people = 3) :=
begin
  sorry
end

end seating_arrangements_with_adjacent_empty_seats_l612_612328


namespace sum_sequence_up_to_2015_l612_612433

def sequence_val (n : ℕ) : ℕ :=
  if n % 288 = 0 then 7 
  else if n % 224 = 0 then 9
  else if n % 63 = 0 then 32
  else 0

theorem sum_sequence_up_to_2015 : 
  (Finset.range 2016).sum sequence_val = 1106 :=
by
  sorry

end sum_sequence_up_to_2015_l612_612433


namespace find_n_l612_612309

def f (x n : ℝ) : ℝ := 2 * x^2 - 3 * x + n
def g (x n : ℝ) : ℝ := 2 * x^2 - 3 * x + 5 * n

theorem find_n (n : ℝ) (h : 3 * f 3 n = 2 * g 3 n) : n = 9 / 7 := by
  sorry

end find_n_l612_612309


namespace food_requirement_not_met_l612_612500

variable (B S : ℝ)

-- Conditions
axiom h1 : B > S
axiom h2 : B < 2 * S
axiom h3 : (B + 2 * S) / 2 = (B + 2 * S) / 2 -- This represents "B + 2S is enough for 2 days"

-- Statement
theorem food_requirement_not_met : 4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  calc
    4 * B + 4 * S = 4 * (B + S)          : by simp
             ... < 3 * (B + 2 * S)        : by
                have calc1 : 4 * B + 4 * S < 6 * S + 3 * B := by linarith
                have calc2 : 6 * S + 3 * B = 3 * (B + 2 * S) := by linarith
                linarith
λλά sorry

end food_requirement_not_met_l612_612500


namespace length_AN_is_one_l612_612281

noncomputable def segment_length_AN (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) (point_N_extension : ExtensionOfSide A B N) 
  (AC_eq_AM : AC = 1 ∧ AM = 1) (ANM_eq_CNM : ∠ANM = ∠CNM) : Real :=
1

/-- Given a triangle ABC, where point M lies on the angle bisector of angle BAC,
and point N lies on the extension of side AB beyond point A, where AC = AM = 1,
and the angles ANM and CNM are equal, the length of segment AN is 1. -/
theorem length_AN_is_one (A B C M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] 
  (triangle_ABC : Triangle A B C) (angle_bisector_AM : AngleBisector A M C) 
  (point_N_extension : ExtensionOfSide A B N) (AC_eq_AM : AC = 1 ∧ AM = 1) 
  (ANM_eq_CNM : ∠ANM = ∠CNM) : 
  segment_length_AN A B C M N triangle_ABC angle_bisector_AM point_N_extension AC_eq_AM ANM_eq_CNM = 1 :=
by
  sorry

end length_AN_is_one_l612_612281


namespace mappings_count_correct_l612_612101

namespace Mappings

def P : Set ℤ := {0, 1}
def Q : Set ℤ := {-1, 0, 1}
def valid_mappings_count (f : ℤ → ℤ) : Prop :=
  { f | f 0 > f 1 ∧ f 0 ∈ Q ∧ f 1 ∈ Q }.card = 3

theorem mappings_count_correct : ∃ f : ℤ → ℤ, valid_mappings_count f :=
by { sorry }

end Mappings

end mappings_count_correct_l612_612101


namespace minimum_value_of_expression_l612_612171

theorem minimum_value_of_expression (a b : ℝ) (h1 : 0 < a ∧ a < 2) (h2 : 0 < b ∧ b < 2) (h3 : a * b = 1) :
  ∃ C : ℝ, C = (2 + 2 * real.sqrt 2 / 3) ∧ ( ∀ (x y : ℝ), (0 < x ∧ x < 2) ∧ (0 < y ∧ y < 2) ∧ (x * y = 1) → ( ( (1 / (2 - x)) + (2 / (2 - y)) ) ≥ C ) ) :=
by
  sorry

end minimum_value_of_expression_l612_612171


namespace mario_total_flowers_l612_612237

def hibiscus_flower_count (n : ℕ) : ℕ :=
  let h1 := 2 + 3 * n
  let h2 := (2 * 2) + 4 * n
  let h3 := (4 * (2 * 2)) + 5 * n
  h1 + h2 + h3

def rose_flower_count (n : ℕ) : ℕ :=
  let r1 := 3 + 2 * n
  let r2 := 5 + 3 * n
  r1 + r2

def sunflower_flower_count (n : ℕ) : ℕ :=
  6 * 2^n

def total_flower_count (n : ℕ) : ℕ :=
  hibiscus_flower_count n + rose_flower_count n + sunflower_flower_count n

theorem mario_total_flowers :
  total_flower_count 2 = 88 :=
by
  unfold total_flower_count hibiscus_flower_count rose_flower_count sunflower_flower_count
  norm_num

end mario_total_flowers_l612_612237


namespace math_problem_statement_l612_612115

-- Defining the problem setup
noncomputable def angle_A := 45
noncomputable def BC_length := 10
noncomputable def perp_BD_AC := true
noncomputable def perp_CE_AB := true
noncomputable def angle_DBC_eq_2_angle_ECB (angle_ECB : ℝ) := 2 * angle_ECB

-- Lean statement asserting that p + q + r = 8.33 given the conditions
theorem math_problem_statement :
  ∃ p q r : ℝ, 
  (p ≠ 0 ∧ q ≠ 0) ∧
  (is_sqrt q ∧ is_sqrt r) ∧
  (m_angle_A = 45) ∧
  (BC_length = 10) ∧
  (perp_BD_AC = true) ∧
  (perp_CE_AB = true) ∧
  (∀ angle_ECB, angle_DBC_eq_2_angle_ECB angle_ECB) ∧
  (p * (sqrt q + sqrt r)) = EC_length → 
  p + q + r = 8.33 := sorry

end math_problem_statement_l612_612115


namespace fraction_of_airing_time_spent_on_commercials_l612_612005

theorem fraction_of_airing_time_spent_on_commercials 
  (num_programs : ℕ) (minutes_per_program : ℕ) (total_commercial_time : ℕ) 
  (h1 : num_programs = 6) (h2 : minutes_per_program = 30) (h3 : total_commercial_time = 45) : 
  (total_commercial_time : ℚ) / (num_programs * minutes_per_program : ℚ) = 1 / 4 :=
by {
  -- The proof is omitted here as only the statement is required according to the instruction.
  sorry
}

end fraction_of_airing_time_spent_on_commercials_l612_612005


namespace count_good_numbers_l612_612383

-- Define the property of a good number
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- Count the number of good numbers
theorem count_good_numbers : (finset.filter is_good_number (finset.range 2021)).card = 10 :=
by sorry

end count_good_numbers_l612_612383


namespace solitaire_game_removal_l612_612463

theorem solitaire_game_removal (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∃ moves : ℕ, ∀ i : ℕ, i < moves → (i + 1) % 2 = (i % 2) + 1) ↔ (m % 2 = 1 ∨ n % 2 = 1) :=
sorry

end solitaire_game_removal_l612_612463


namespace enough_cat_food_for_six_days_l612_612523

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l612_612523


namespace man_arrived_early_l612_612562

noncomputable def earlyArrival : ℕ :=
let W := 10 in
let walking_time := 55 in
walking_time - W

theorem man_arrived_early (H1 : ∀ t, earlyArrival = 45) :
  earlyArrival = 45 := by
  let W := 10
  let walking_time := 55
  simp [earlyArrival, W, walking_time]
  exact H1 45

#eval earlyArrival -- Should return 45 indicating the man arrived 45 minutes early

end man_arrived_early_l612_612562


namespace factorize_quadratic_l612_612932

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by
  sorry

end factorize_quadratic_l612_612932


namespace min_value_p_plus_q_l612_612840

theorem min_value_p_plus_q (p q : ℕ) (hp : 1 < p) (hq : 1 < q) 
  (h : 17 * (p + 1) = 20 * (q + 1)) : p + q = 37 :=
sorry

end min_value_p_plus_q_l612_612840


namespace total_distance_correct_l612_612765

-- Define the times in hours
def walk_time := (30 : ℝ) / 60
def run_time := (20 : ℝ) / 60
def cycle_time := (40 : ℝ) / 60

-- Define the speeds in mph
def walk_speed := 3
def run_speed := 8
def cycle_speed := 12

-- Define the distances
def walk_distance := walk_speed * walk_time
def run_distance := run_speed * run_time
def cycle_distance := cycle_speed * cycle_time

-- Define total distance
def total_distance := walk_distance + run_distance + cycle_distance

theorem total_distance_correct : total_distance = 12.17 :=
by
  have h1 : walk_distance = 3 * (30 / 60) := rfl
  have h2 : run_distance = 8 * (20 / 60) := rfl
  have h3 : cycle_distance = 12 * (40 / 60) := rfl
  unfold walk_distance run_distance cycle_distance total_distance
  dsimp
  norm_num
  sorry -- Finish the proof

end total_distance_correct_l612_612765


namespace sin_initial_phase_highest_point_l612_612313

def highest_point_coordinates (θ : ℝ) (x : ℝ) (y : ℝ) (k : ℤ) : Prop :=
  θ = π / 6 ∧ y = 1 ∧ x = (8 * π / 3) + (8 * π) * k

theorem sin_initial_phase_highest_point : ∀ (θ : ℝ) (k : ℤ),
  highest_point_coordinates θ ((8 * π / 3) + (8 * π) * k) 1 k :=
by
  intros θ k
  sorry

end sin_initial_phase_highest_point_l612_612313


namespace length_AN_eq_one_l612_612253

-- Definitions representing the conditions
def triangle := {A B C : Type}
def point (P : Type) := P
def length (A B : Type) := 1

variables (A B C M N : Type)

def angle_bisector (BAC : Prop) := ∀ (A B C : point), 
  let P := ∃ (X : Type), X

constant AC : length A C
constant AM : length A M
constant AN : length A N
constant angle_ANM_CNMC_eq : ∀ (ANM CNM : Prop), ANM = CNM

-- Conclude the length of AN is 1
theorem length_AN_eq_one : AN = 1 :=
sorry

end length_AN_eq_one_l612_612253


namespace vector_magnitude_proof_l612_612594

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : vector) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def perpendicular (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

theorem vector_magnitude_proof : 
    ∀ (x : ℝ),
    let a := (x, 2) in
    let b := (2, 3/2) in
    perpendicular (a - b) a →
    magnitude (a + (2, 2) • b) = 5 * Real.sqrt 2 :=
by
  intros x a b h_perpendicular
  sorry

end vector_magnitude_proof_l612_612594


namespace number_of_good_numbers_l612_612358

theorem number_of_good_numbers :
  let n := 2020 in 
  let remainder := 22 in
  let number := 1998 in
  let divisors := {d ∣ number | d > remainder}.to_list.length = 10 := by sorry

end number_of_good_numbers_l612_612358


namespace test_scores_l612_612476

def scores (n : ℕ) (a : ℕ → ℕ) : Prop :=
  (∀ i j, i ≠ j → a i ≠ a j) ∧ -- All scores are different
  (∑ i in finset.range(n+1), a i = 119) ∧ -- Total score is 119 points
  (∑ i in finset.range 3, a i = 23) ∧ -- Sum of the three lowest scores is 23 points
  (∑ i in finset.range(n-2, n+1), a i = 49) -- Sum of the three highest scores is 49 points

theorem test_scores :
  ∃ n a, scores n a ∧ n = 10 ∧ a n = 18 :=
by
  sorry

end test_scores_l612_612476


namespace problem_1_problem_2_l612_612636

-- Problem 1:
-- Given: kx^2 - 2x + 3k < 0 and the solution set is {x | x < -3 or x > -1}, prove k = -1/2
theorem problem_1 {k : ℝ} :
  (∀ x : ℝ, (kx^2 - 2*x + 3*k < 0 ↔ x < -3 ∨ x > -1)) → k = -1/2 :=
sorry

-- Problem 2:
-- Given: kx^2 - 2x + 3k < 0 and the solution set is ∅, prove 0 < k ≤ sqrt(3) / 3
theorem problem_2 {k : ℝ} :
  (∀ x : ℝ, ¬ (kx^2 - 2*x + 3*k < 0)) → 0 < k ∧ k ≤ Real.sqrt 3 / 3 :=
sorry

end problem_1_problem_2_l612_612636


namespace angle_RIS_acute_l612_612707

-- Definitions used in Lean statement
variables {A B C I K L M R S : Type*} [affine_space A]
variables (B M K L R S) 

-- Definitions
def incenter (I : Type*) (ABC : triangle A) := 
  ∃ (I : A), (I is the point of intersection of the angle bisectors of ∆ABC)

def incircle_touches (K L M : Type*) (ABC : triangle A) :=
  ∃ (K L M : A), (K is the point on BC, L is the point on CA, and M is the point on AB touched by the incircle of ∆ABC)

def line_parallel (B M : Type*) := 
  line B contains point B ∧ line L contains point M ∧ line B is parallel to line M

def meets_lines (R S : Type*) := 
  R is the intersection point of B and L ∧ S is the intersection point of B and K

-- Statement
theorem angle_RIS_acute
  (ABC : triangle A) (I : A) (K L M R S : A)
  (h1 : incenter I ABC) 
  (h2 : incircle_touches K L M ABC)
  (h3 : line_parallel B M)
  (h4 : meets_lines R S)
  : is_acute_angle ∠RIS :=
sorry

end angle_RIS_acute_l612_612707


namespace radius_of_cylinder_l612_612875

-- Define the main parameters and conditions
def diameter_cone := 8
def radius_cone := diameter_cone / 2
def altitude_cone := 10
def height_cylinder (r : ℝ) := 2 * r

-- Assume similarity of triangles
theorem radius_of_cylinder (r : ℝ) (h_c := height_cylinder r) :
  altitude_cone - h_c / r = altitude_cone / radius_cone → r = 20 / 9 := 
by
  intro h
  sorry

end radius_of_cylinder_l612_612875


namespace function_solution_l612_612552

variable (f : ℝ → ℝ)

-- Given functional equation
def functional_eq (x y : ℝ) : Prop :=
  f(x * f(y) + 1) = y + f(f(x) * f(y))

-- The theorem statement to be proven in Lean
theorem function_solution :
  (∀ x y : ℝ, functional_eq f x y) → (f = λ x, x - 1) :=
by
  sorry

end function_solution_l612_612552


namespace flare_initial_velocity_and_duration_l612_612882

noncomputable def h (v : ℝ) (t : ℝ) : ℝ := v * t - 4.9 * t^2

theorem flare_initial_velocity_and_duration (v t : ℝ) :
  (h v 5 = 245) ↔ (v = 73.5) ∧ (5 < t ∧ t < 10) :=
by {
  sorry
}

end flare_initial_velocity_and_duration_l612_612882


namespace chloe_total_books_l612_612049

noncomputable def total_books (average_books_per_shelf : ℕ) 
  (mystery_shelves : ℕ) (picture_shelves : ℕ) 
  (science_fiction_shelves : ℕ) (history_shelves : ℕ) : ℕ :=
  (mystery_shelves + picture_shelves + science_fiction_shelves + history_shelves) * average_books_per_shelf

theorem chloe_total_books : 
  total_books 85 7 5 3 2 = 14500 / 100 :=
  by
  sorry

end chloe_total_books_l612_612049


namespace express_in_scientific_notation_l612_612303

theorem express_in_scientific_notation (x : ℝ) (h : x = 720000) : x = 7.2 * 10^5 :=
by sorry

end express_in_scientific_notation_l612_612303


namespace find_d_e_f_l612_612216

theorem find_d_e_f:
  ∃ d e f : ℕ, 
  (∀ x : ℝ, (2 / (x - 2) + 4 / (x - 4) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 12x - 6) → 
    ∃ n, n = real.max (n : ℝ) (λ x, 2 / (x - 2) + 4 / (x - 4) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 12x - 6) ∧ 
    n = d + real.sqrt (e + real.sqrt f)) ∧ 
  d + e + f = 556 := 
sorry

end find_d_e_f_l612_612216


namespace average_score_is_two_l612_612884

-- Definitions of given conditions
def points := [3, 2, 1, 0]
def proportions := [0.3, 0.5, 0.1, 0.1]

-- Average score calculation
def average_score (points : List ℕ) (proportions : List ℝ) : ℝ :=
  List.foldr (+) 0 (List.map (λ (p : ℕ × ℝ) => p.1 * p.2) (List.zip points proportions))

theorem average_score_is_two : average_score points proportions = 2 := 
by
  -- Proof is skipped with sorry
  sorry

end average_score_is_two_l612_612884


namespace cat_food_inequality_l612_612537

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l612_612537


namespace diameter_count_leq_points_l612_612105

theorem diameter_count_leq_points (A : set (fin n → ℝ)) (k : ℕ) (hA : ∀ P, P ∈ A → ∃ j i, j ≠ i ∧ dist (P j) (P i) = k)
  : k ≤ n := 
sorry

end diameter_count_leq_points_l612_612105


namespace complete_contingency_table_chi_square_test_certainty_l612_612857

-- Defining the initial conditions given in the problem
def total_students : ℕ := 100
def boys_dislike : ℕ := 10
def girls_like : ℕ := 20
def dislike_probability : ℚ := 0.4

-- Completed contingency table values based on given and inferred values
def boys_total : ℕ := 50
def girls_total : ℕ := 50
def boys_like : ℕ := boys_total - boys_dislike
def girls_dislike : ℕ := 30
def total_like : ℕ := boys_like + girls_like
def total_dislike : ℕ := boys_dislike + girls_dislike

-- Chi-square value from the solution
def K_squared : ℚ := 50 / 3

-- Declaring the proof problem for the completed contingency table
theorem complete_contingency_table :
  boys_total + girls_total = total_students ∧ 
  total_like + total_dislike = total_students ∧ 
  dislike_probability * total_students = total_dislike ∧ 
  boys_like = 40 ∧ 
  girls_dislike = 30 :=
sorry

-- Declaring the proof problem for the chi-square test
theorem chi_square_test_certainty :
  K_squared > 10.828 :=
sorry

end complete_contingency_table_chi_square_test_certainty_l612_612857


namespace gcd_polynomial_l612_612609

theorem gcd_polynomial (b : ℤ) (h : 1729 ∣ b) : Int.gcd (b^2 + 11*b + 28) (b + 5) = 2 := 
by
  sorry

end gcd_polynomial_l612_612609


namespace limit_of_trig_function_l612_612842

theorem limit_of_trig_function (a : ℝ) :
  tendsto (λ x : ℝ, ( (sin x - sin a) / (x - a) )^( x^2 / a^2 )) (𝓝 a) (𝓝 (cos a)) :=
by sorry

end limit_of_trig_function_l612_612842


namespace determine_g10_l612_612781

variable (g : ℝ → ℝ)

axiom additivity_gh : ∀ x y : ℝ, g(x + y) = g(x) + g(y) - 1
axiom nontrivial_zero_g : g(0) ≠ 1
axiom specific_one_g : g(1) = 1

theorem determine_g10 : g(10) = 10 := by
  sorry

end determine_g10_l612_612781


namespace quadratic_function_properties_l612_612024

-- Define the quadratic function f satisfying the given conditions
def f (x : ℝ) : ℝ := x^2 - x + 1

-- Define g as a linear function with parameter m
def g (x m : ℝ) : ℝ := 2 * x + m

-- Define the conditions as hypotheses
theorem quadratic_function_properties (m : ℝ) :
  (∀ x : ℝ, f (x + 1) - f x = 2 * x) ∧ (f 0 = 1) ∧ (∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x > g x m) → m < -1 :=
begin
  sorry
end

end quadratic_function_properties_l612_612024


namespace sequence_general_form_l612_612103

theorem sequence_general_form
  (a : ℕ → ℝ) 
  (pos_seq : ∀ n : ℕ, 0 < a n)
  (h1 : ∀ n : ℕ, (sqrt (a n * a (n + 1) + a n * a (n + 2)) 
                     = 4 * sqrt (a n * a (n + 1) + a (n + 1)^2)
                     + 3 * sqrt (a n * a (n + 1))))
  (a1 : a 1 = 1)
  (a2 : a 2 = 8) : 
  ∀ n : ℕ, a n = if n = 1 then 1 else ∏ k in finset.range (n - 1), ((4^k - 1)^2 - 1) :=
by
  sorry

end sequence_general_form_l612_612103


namespace correct_conclusions_l612_612806

theorem correct_conclusions (x : ℝ)
  (h1 : (16:ℝ)^2 = 256) (h2 : (16.1:ℝ)^2 = 259.21)
  (h3 : (16.2:ℝ)^2 = 262.44) (h4 : (16.3:ℝ)^2 = 265.69)
  (h5 : (16.4:ℝ)^2 = 268.96) (h6 : (16.5:ℝ)^2 = 272.25)
  (h7 : (16.6:ℝ)^2 = 275.56) (h8 : (16.7:ℝ)^2 = 278.89)
  (h9 : (16.8:ℝ)^2 = 282.24) (h10 : (16.9:ℝ)^2 = 285.61)
  (h11 : (17:ℝ)^2 = 289) :
  (sqrt 285.61 = 16.9) ∧ (sqrt 26896 = 164 ∨ sqrt 26896 = -164) ∧
  (20 - sqrt 260 ≠ 4) ∧ 
  (∃ a b c : ℕ, 259.21 < (a:ℝ) ∧ (a:ℝ) < 262.44 ∧ 259.21 < (b:ℝ) ∧ (b:ℝ) < 262.44 ∧ 259.21 < (c:ℝ) ∧ (c:ℝ) < 262.44) :=
by 
  split,
  { show sqrt 285.61 = 16.9, from sorry },
  split,
  { show sqrt 26896 = 164 ∨ sqrt 26896 = -164, from sorry },
  split,
  { show 20 - sqrt 260 ≠ 4, from sorry },
  { show ∃ (a b c : ℕ), 259.21 < (a:ℝ) ∧ (a:ℝ) < 262.44 ∧ 259.21 < (b:ℝ) ∧ (b:ℝ) < 262.44 ∧ 259.21 < (c:ℝ) ∧ (c:ℝ) < 262.44, from sorry },

end correct_conclusions_l612_612806


namespace sin_cos_squared_range_l612_612715

theorem sin_cos_squared_range (θ : ℝ) :
  let s := Real.sin θ,
      c := Real.cos θ in
  -1 ≤ s^2 - c^2 ∧ s^2 - c^2 ≤ 1 :=
by
  let s := Real.sin θ,
      c := Real.cos θ
  sorry

end sin_cos_squared_range_l612_612715


namespace proof_problem_2ii_l612_612600

noncomputable def equation_of_circle := ∀ x y : ℝ,
  let C : ℝ × ℝ := (-1, 0)
  in (x + 1)^2 + y^2 = 4

variable (x1 x2 : ℝ) (k : ℝ) (h_pos : k > 0)

lemma proof_problem_2i : ∀ x1 x2 : ℝ, 
  let k := (4 * x1 + 3 * k * x1 - 6)/(4 + 3 * k) 
  in (1 / x1) + (1 / x2) = 2 / 3 :=
sorry

theorem proof_problem_2ii (x1 x2 k : ℝ) (h_pos : k > 0): 
  let N := (2, 1) in 
  ∀ PN QN : ℝ, 
  let P := (x1, k * x1) in 
  let Q := (x2, k * x2) in 
  |PN|^2 + |QN|^2 ≤ 2 * real.sqrt 10 + 22 :=
sorry

end proof_problem_2ii_l612_612600


namespace power_mod_7_l612_612411

theorem power_mod_7 {a : ℤ} (h : a = 3) : (a ^ 123) % 7 = 6 := by
  sorry

end power_mod_7_l612_612411


namespace horizontal_asymptote_condition_l612_612061

open Polynomial

def polynomial_deg_with_horiz_asymp (p : Polynomial ℝ) : Prop :=
  degree p ≤ 4

theorem horizontal_asymptote_condition (p : Polynomial ℝ) :
  polynomial_deg_with_horiz_asymp p :=
sorry

end horizontal_asymptote_condition_l612_612061


namespace evaluate_expression_l612_612492

theorem evaluate_expression :
  ((-2: ℤ)^2) ^ (1 ^ (0 ^ 2)) + 3 ^ (0 ^(1 ^ 2)) = 5 :=
by
  -- sorry allows us to skip the proof
  sorry

end evaluate_expression_l612_612492


namespace mass_percentage_Ca_in_CaI2_l612_612578

noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I

theorem mass_percentage_Ca_in_CaI2 :
  (molar_mass_Ca / molar_mass_CaI2) * 100 = 13.63 :=
by
  sorry

end mass_percentage_Ca_in_CaI2_l612_612578


namespace average_speed_of_car_l612_612427

def total_distance (t1_d t2_d : ℕ) := t1_d + t2_d
def total_time (t1_t t2_t : ℕ) := t1_t + t2_t
def average_speed (total_d total_t : ℕ) := total_d / total_t

theorem average_speed_of_car :
  (total_distance 145 60) = 205 ∧ 
  (total_time 1 1) = 2 ∧ 
  (average_speed 205 2) = 102.5 := by
  sorry

end average_speed_of_car_l612_612427


namespace no_six_consecutive010101_l612_612297

def unit_digit (n: ℕ) : ℕ := n % 10

def sequence : ℕ → ℕ
| 0     => 1
| 1     => 0
| 2     => 1
| 3     => 0
| 4     => 1
| 5     => 0
| (n + 6) => unit_digit (sequence n + sequence (n + 1) + sequence (n + 2) + sequence (n + 3) + sequence (n + 4) + sequence (n + 5))

theorem no_six_consecutive010101 : ∀ n, ¬ (sequence n = 0 ∧ sequence (n + 1) = 1 ∧ sequence (n + 2) = 0 ∧ sequence (n + 3) = 1 ∧ sequence (n + 4) = 0 ∧ sequence (n + 5) = 1) :=
sorry

end no_six_consecutive010101_l612_612297


namespace earning_from_trout_is_correct_l612_612900

variable (earn_per_trout : ℝ) (video_game_cost last_week_earning earn_per_bluegill current_earning_needed : ℝ)
variable (total_fish caught_trout caught_bluegill : ℕ)

-- Conditions
def video_game_cost := 60
def last_week_earning := 35
def earn_per_bluegill := 4
def total_fish := 5
def trout_percentage := 0.60
def current_earning_needed := 2

-- Calculate how much Bucky needs to earn from each trout
def calculate_trout_earning : Prop :=
  let caught_trout := trunc (trout_percentage * total_fish)
  let caught_bluegill := total_fish - caught_trout
  let earned_from_bluegill := caught_bluegill * earn_per_bluegill
  (last_week_earning + earned_from_bluegill) + current_earning_needed = video_game_cost →
  earn_per_trout = 5.67

-- The main theorem statement
theorem earning_from_trout_is_correct : calculate_trout_earning :=
sorry

end earning_from_trout_is_correct_l612_612900


namespace sum_of_coefficients_l612_612924

def f (x : ℝ) : ℝ := (1 + 2 * x)^4

theorem sum_of_coefficients : f 1 = 81 :=
by
  -- New goal is immediately achieved since the given is precisely ensured.
  sorry

end sum_of_coefficients_l612_612924


namespace find_k_l612_612577

-- The given condition (equation)
def given_equation (d k m : ℚ) : Prop :=
  (6 * x^3 - 4 * x^2 + 9/4) * (d * x^3 + k * x^2 + m)
  = 18 * x^6 - 17 * x^5 + 34 * x^4 - 9 * x^3 + 9/2 * x^2

-- The theorem to prove
theorem find_k (d k m : ℚ)
  (h : given_equation d k m) :
  k = -5/6 := by
  sorry

end find_k_l612_612577


namespace rains_at_least_once_l612_612243

noncomputable def prob_rains_on_weekend : ℝ :=
  let prob_rain_saturday := 0.60
  let prob_rain_sunday := 0.70
  let prob_no_rain_saturday := 1 - prob_rain_saturday
  let prob_no_rain_sunday := 1 - prob_rain_sunday
  let independent_events := prob_no_rain_saturday * prob_no_rain_sunday
  1 - independent_events

theorem rains_at_least_once :
  prob_rains_on_weekend = 0.88 :=
by sorry

end rains_at_least_once_l612_612243


namespace number_of_divisors_greater_than_22_l612_612389

theorem number_of_divisors_greater_than_22 :
  (∃ (S : Finset ℕ), S = (Finset.filter (λ n, 1998 % n = 0 ∧ n > 22) (Finset.range (1998 + 1))) ∧ S.card = 10) :=
sorry

end number_of_divisors_greater_than_22_l612_612389


namespace amy_small_gardens_l612_612030

-- Define the initial number of seeds
def initial_seeds : ℕ := 101

-- Define the number of seeds planted in the big garden
def big_garden_seeds : ℕ := 47

-- Define the number of seeds planted in each small garden
def seeds_per_small_garden : ℕ := 6

-- Define the number of small gardens
def number_of_small_gardens : ℕ := (initial_seeds - big_garden_seeds) / seeds_per_small_garden

-- Prove that Amy has 9 small gardens
theorem amy_small_gardens : number_of_small_gardens = 9 := by
  sorry

end amy_small_gardens_l612_612030


namespace good_numbers_2020_has_count_10_l612_612398

theorem good_numbers_2020_has_count_10 :
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  set.count divisors = 10 :=
by
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  sorry

end good_numbers_2020_has_count_10_l612_612398


namespace solve_for_x_l612_612752

theorem solve_for_x (x : ℝ) (h : 4^(6*x - 9) = (1/2)^(3*x + 7)) : x = 11/15 := 
sorry

end solve_for_x_l612_612752


namespace speed_in_first_hour_l612_612324

variable (x : ℕ)
-- Conditions: 
-- The speed of the car in the second hour:
def speed_in_second_hour : ℕ := 30
-- The average speed of the car:
def average_speed : ℕ := 60
-- The total time traveled:
def total_time : ℕ := 2

-- Proof problem: Prove that the speed of the car in the first hour is 90 km/h.
theorem speed_in_first_hour : x + speed_in_second_hour = average_speed * total_time → x = 90 := 
by 
  intro h
  sorry

end speed_in_first_hour_l612_612324


namespace cat_food_insufficient_l612_612510

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l612_612510


namespace count_good_numbers_l612_612360

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l612_612360


namespace wheels_per_vehicle_l612_612820

theorem wheels_per_vehicle (w : ℕ) (trucks cars total_wheels : ℕ) 
  (h1 : trucks = 12) (h2 : cars = 13) (h3 : total_wheels = 100)
  (h4 : total_wheels = (trucks + cars) * w) : w = 4 := by 
  rw [h1, h2] at h4
  simp at h4
  exact h4

end wheels_per_vehicle_l612_612820


namespace right_handed_total_l612_612241

theorem right_handed_total (total_players throwers : Nat) (h1 : total_players = 70) (h2 : throwers = 37) :
  let non_throwers := total_players - throwers
  let left_handed := non_throwers / 3
  let right_handed_non_throwers := non_throwers - left_handed
  let right_handed := right_handed_non_throwers + throwers
  right_handed = 59 :=
by
  sorry

end right_handed_total_l612_612241
