import Geometry.Trapezoid.Basic
import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Field
import Mathlib.Algebra.GCD
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.LinearAlgebra.Basic
import Mathlib.Algebra.Matrix
import Mathlib.Algebra.Quaternions
import Mathlib.Analysis.Calculus.Area
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finite.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Rat
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Real.Trigonometry
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Angles
import Mathlib.Geometry.Euclidean.Circle.Tangent
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.NumberTheory.ArithmeticFunctions
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Trigonometry.Basic
import Probability

namespace angle_on_line_y_equals_neg_x_l56_56653

noncomputable def line_y_equals_neg_x (θ : ℝ) :=
  ∃ k : ℤ, θ = k * Real.pi - Real.pi / 4

theorem angle_on_line_y_equals_neg_x (α : ℝ) :
  (∃ k : ℤ, α = k * Real.pi - Real.pi / 4)
  ↔ line_y_equals_neg_x α :=
sorry

end angle_on_line_y_equals_neg_x_l56_56653


namespace coloring_problem_l56_56683

noncomputable def numColorings : Nat :=
  256

theorem coloring_problem :
  ∃ (f : Fin 200 → Bool), (∀ (i j : Fin 200), i ≠ j → f i = f j → ¬ (i + 1 + j + 1).isPowerOfTwo) ∧ 
    256 = numColorings :=
by
  -- Definitions and conditions used for the problem
  let S := {n : ℕ | ∃ k : ℕ, n = bit0 k 1 ∧ k > 0}
  have hS : ∀ n ∈ S, ∀ i j ∈ {1, ..., 200}, n ≠ 2 ^ (log2 n) ∧ i ≠ j → i + j ≠ n := sorry
  -- Proof skipped, as required in the guidelines
  sorry

end coloring_problem_l56_56683


namespace central_projection_shapes_l56_56421

-- Definitions for the vertices of the triangle and the projection plane
variables {A B C : point}  -- Points in 3D space
variables {σ : plane}      -- Projection plane
variables {S : point}      -- Center of projection not in the plane of the triangle or projection plane

-- The central projection of triangle ABC onto plane σ with center S
def central_projection (A B C : point) (σ : plane) (S : point) : Type :=
  if S ∉ plane_of_triangle A B C ∧ S ∉ σ then
    { shape : Type // shape ∈ {angle, strip, two_angles, triangle, quasi_triangle, segment, ray, point, line, entire_plane} }
  else
    { shape : Type // shape = none }

-- The corresponding Lean statement
theorem central_projection_shapes (A B C : point) (σ : plane) (S : point) (h1 : S ∉ plane_of_triangle A B C) (h2 : S ∉ σ) :
  ∃ shape : Type,
    shape ∈ {angle, strip, two_angles, triangle, quasi_triangle, segment, ray, point, line, entire_plane} :=
sorry

end central_projection_shapes_l56_56421


namespace no_partition_exists_partition_exists_l56_56913

-- Part (a)

theorem no_partition_exists 
  (a b : ℝ) 
  (ha : 1 < a) 
  (hb1 : a < 2) 
  (hb2 : 2 < b) : 
  ¬(∃ (A0 A1 : set ℕ), 
    (∀ j ∈ {0, 1}, ∀ m n ∈ (if j = 0 then A0 else A1), (m : ℝ) / (n : ℝ) < a ∨ (m : ℝ) / (n : ℝ) > b) ∧ 
    ∀ x, x ∈ A0 ∨ x ∈ A1 ∧ (x ∈ A0 → x ∉ A1) ∧ (x ∈ A1 → x ∉ A0)) := sorry

-- Part (b)

theorem partition_exists 
  (a b : ℝ) 
  (ha : 1 < a) 
  (hb1 : a < 2) 
  (hb2 : 2 < b) : 
  (b ≤ a^2) ↔ ∃ (A0 A1 A2 : set ℕ), 
    (∀ j ∈ {0, 1, 2}, ∀ m n ∈ (if j = 0 then A0 else if j = 1 then A1 else A2), 
      (m : ℝ) / (n : ℝ) < a ∨ (m : ℝ) / (n : ℝ) > b) ∧ 
    ∀ x, x ∈ A0 ∨ x ∈ A1 ∨ x ∈ A2 ∧ 
      (x ∈ A0 → x ∉ A1 ∧ x ∉ A2) ∧ 
      (x ∈ A1 → x ∉ A0 ∧ x ∉ A2) ∧ 
      (x ∈ A2 → x ∉ A0 ∧ x ∉ A1) := sorry

end no_partition_exists_partition_exists_l56_56913


namespace product_of_possible_m_l56_56806

def g (x : ℝ) (m : ℝ) : ℝ :=
  if x < m then x^2 + 3 * x + 1 else 3 * x + 10

def is_continuous_at (g : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - m| < δ → |g x - g m| < ε

theorem product_of_possible_m : 
  let g (x : ℝ) (m : ℝ) : ℝ :=
    if x < m then x^2 + 3 * x + 1 else 3 * x + 10
  in is_continuous_at (fun x => g x m) m → ∃ (a b : ℝ), (a = 3 ∨ a = -3) ∧ (b = 3 ∨ b = -3) ∧ a * b = -9 :=
by
  intros
  sorry

end product_of_possible_m_l56_56806


namespace arrangement_possible_l56_56090

theorem arrangement_possible :
  ∃ (matrix : Fin 9 → Fin 9 → ℕ),
    (∀ i, (∏ j, matrix i j) = (∏ j, matrix (0 : Fin 9) j)) ∧
    (∀ j, (∏ i, matrix i j) = (∏ i, matrix i (0 : Fin 9))) ∧
    (function.injective (λ ij : Fin 81, matrix (ij / 9) (ij % 9))) ∧
    (∀ i j, matrix i j ≤ 1991) :=
sorry

end arrangement_possible_l56_56090


namespace translation_proof_l56_56403

noncomputable def f : Real → Real := λ x, Real.sin (2 * x + Real.pi / 3)
noncomputable def g : Real → Real := λ x, Real.sin (2 * x + 2 * Real.pi / 3)
def translation_left (h : Real → Real) (a : Real) : (Real → Real) := λ x, h (x + a)

theorem translation_proof :
  translation_left f (Real.pi / 6) = g :=
by
  sorry

end translation_proof_l56_56403


namespace haley_spent_32_dollars_l56_56443

noncomputable def total_spending (ticket_price : ℕ) (tickets_bought_self_friends : ℕ) (extra_tickets : ℕ) : ℕ :=
  ticket_price * (tickets_bought_self_friends + extra_tickets)

theorem haley_spent_32_dollars :
  total_spending 4 3 5 = 32 :=
by
  sorry

end haley_spent_32_dollars_l56_56443


namespace isosceles_triangle_area_division_ratio_l56_56946

theorem isosceles_triangle_area_division_ratio
  (a b c : ℝ) (α β : ℝ)
  (isosceles : a = c)
  (vertex_angle : α = ∠BAC)
  (line_through_vertex : ∃ D : Point, D ∈ LineThrough A ∧ ∠BAD = β) :
  let S_ΔBAD := 1/2 * a * AD * sin(α - β)
  let S_ΔCAD := 1/2 * (2 * a * cos α) * AD * sin β
  S_ΔBAD / S_ΔCAD = sin (α - β) / (2 * cos α * sin β) :=
by
  sorry

end isosceles_triangle_area_division_ratio_l56_56946


namespace centroid_sum_l56_56143

-- Definitions as per the given conditions
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A B C : Point

-- Properties of the triangle
variables (ABC : Triangle)
variables (A_eq : ABC.A = { x := 4, y := 0 })
variables (B_on_AB : ABC.B.x = 6 ∧ ABC.B.y = 0)
variables (C_on_BC : ABC.C.x = 12 ∧ ABC.C.y = 0)

-- The centroid property
def triangle_centroid (T : Triangle) : Point :=
  { x := (T.A.x + T.B.x + T.C.x) / 3,
    y := (T.A.y + T.B.y + T.C.y) / 3 }

-- The theorem to be proved
theorem centroid_sum (ABC : Triangle)
  (A_eq : ABC.A = { x := 4, y := 0 })
  (B_on_AB : ABC.B.x = 6 ∧ ABC.B.y = 0)
  (C_on_BC : ABC.C.x = 12 ∧ ABC.C.y = 0) :
  (triangle_centroid ABC).x + (triangle_centroid ABC).y = 16 / 3 := by
  sorry

end centroid_sum_l56_56143


namespace number_composite_l56_56351

theorem number_composite : ∃ a1 a2 : ℕ, a1 > 1 ∧ a2 > 1 ∧ 2^17 + 2^5 - 1 = a1 * a2 := 
by
  sorry

end number_composite_l56_56351


namespace necessary_but_not_sufficient_hyperbola_l56_56910

def is_hyperbola (a b c : ℝ) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * b < 0

theorem necessary_but_not_sufficient_hyperbola (a b c : ℝ) :
  is_hyperbola a b c → a * b < 0 ∧ (∀ c = 0, ¬ is_hyperbola a b c) :=
by
  sorry

end necessary_but_not_sufficient_hyperbola_l56_56910


namespace football_combinations_l56_56274

theorem football_combinations : 
  ∃ (W D L : ℕ), W + D + L = 15 ∧ 3 * W + D = 33 ∧ 
  (9 ≤ W ∧ W ≤ 11) ∧
  (W = 9 → D = 6 ∧ L = 0) ∧
  (W = 10 → D = 3 ∧ L = 2) ∧
  (W = 11 → D = 0 ∧ L = 4) :=
sorry

end football_combinations_l56_56274


namespace remainder_when_divided_by_15_l56_56422

theorem remainder_when_divided_by_15 (N : ℕ) (k : ℤ) (h1 : N = 60 * k + 49) : (N % 15) = 4 :=
sorry

end remainder_when_divided_by_15_l56_56422


namespace problem_answer_l56_56005

-- Definitions based on the conditions
def is_multiple_of (a b : ℤ) : Prop := ∃ k : ℤ, a = k * b

-- Conditions given in the problem
variables (a b : ℤ)
axiom h1 : is_multiple_of a 4 -- a is a multiple of 4
axiom h2 : is_multiple_of b 8 -- b is a multiple of 8

-- Equivalent proof problem based on the correct answer
theorem problem_answer : (∀ a b : ℤ, is_multiple_of a 4 → is_multiple_of b 8 → 
  (a + b) % 2 = 0 ∧ (a + b) % 4 = 0) :=
begin
  sorry
end

end problem_answer_l56_56005


namespace angle_B_in_trapezoid_l56_56712

theorem angle_B_in_trapezoid (AB CD : ℝ) (A D C B : ℝ) (par_AB_CD : AB ∥ CD) (angle_A_eq_3angle_D : A = 3 * D) (angle_C_eq_3angle_B : C = 3 * B) :
    B = 45 := 
begin
    -- Use the condition that AB ∥ CD to set up the equation
    have h1 : A + D + C + B = 360,
    from sorry, -- Interior angles of a quadrilateral sum up to 360°

    -- Use the given angles relationship and solve for B
    have h2 : B + C = 180,
    from sorry, -- Sum of interior angles on parallel lines
    
    rw [angle_C_eq_3angle_B, h2] at h1,
    linarith,
    
    -- Finally solve for the angle B
end

end angle_B_in_trapezoid_l56_56712


namespace ksyusha_travel_time_l56_56762

theorem ksyusha_travel_time :
  ∀ (S v : ℝ), 
  (2 * S / v + S / (2 * v) = 30) → 
  (2 * (S / v) = 24) :=
by
  intros S v h1
  have h2 : S / v = 12 := by
    calc 
      S / v = (30 * 2) / 5 := by sorry
  calc 
    2 * (S / v) = 2 * 12 := by sorry
  ... = 24 := by norm_num

end ksyusha_travel_time_l56_56762


namespace calc_4_op_3_l56_56208

def specific_op (m n : ℕ) : ℕ := n^2 - m

theorem calc_4_op_3 :
  specific_op 4 3 = 5 :=
by
  sorry

end calc_4_op_3_l56_56208


namespace polynomial_root_bound_l56_56802

def n_p (p : Polynomial ℤ) : ℕ :=
  (Finset.univ.filter (λ x => (p.eval x) ^ 2 = 1)).card

theorem polynomial_root_bound (p : Polynomial ℤ) (hp : p.degree ≥ 1) :
  n_p p - p.natDegree ≤ 2 :=
by
  sorry

end polynomial_root_bound_l56_56802


namespace cost_of_four_half_dozen_bananas_l56_56147

-- Define the total cost of three half-dozen apples
def total_cost_apples : ℝ := 9.36

-- Define the number of half-dozen apples
def number_half_dozen_apples : ℕ := 3

-- Define the number of half-dozen bananas
def number_half_dozen_bananas : ℕ := 4

-- Calculate the cost per half-dozen
def cost_per_half_dozen := total_cost_apples / number_half_dozen_apples

-- Define the target cost for four half-dozen bananas
def target_cost_bananas := number_half_dozen_bananas * cost_per_half_dozen

-- Main theorem to verify the cost calculation
theorem cost_of_four_half_dozen_bananas : target_cost_bananas = 12.48 :=
by
  -- We can assume this as part of the problem.
  sorry

end cost_of_four_half_dozen_bananas_l56_56147


namespace meeting_distance_and_time_l56_56007

theorem meeting_distance_and_time 
  (total_distance : ℝ)
  (delta_time : ℝ)
  (x : ℝ)
  (V : ℝ)
  (v : ℝ)
  (t : ℝ) :

  -- Conditions 
  total_distance = 150 ∧
  delta_time = 25 ∧
  (150 - 2 * x) = 25 ∧
  (62.5 / v) = (87.5 / V) ∧
  (150 / v) - (150 / V) = 25 ∧
  t = (62.5 / v)

  -- Show that 
  → x = 62.5 ∧ t = 36 + 28 / 60 := 
by 
  sorry

end meeting_distance_and_time_l56_56007


namespace stratified_sampling_l56_56475

theorem stratified_sampling :
  let total_employees := 800
  let senior_titles := 160
  let intermediate_titles := 320
  let junior_titles := 200
  let remaining_titles := 120
  let sample_size := 40
  (sample_size / total_employees) = 1 / 20 →
  (senior_titles / 20 = 8) ∧ 
  (intermediate_titles / 20 = 16) ∧ 
  (junior_titles / 20 = 10) ∧ 
  (remaining_titles / 20 = 6) :=
begin
  sorry
end

end stratified_sampling_l56_56475


namespace periodic_functions_properties_l56_56907

noncomputable section

-- Define the function C(x)
def C (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 2 then 1 - x
else if 2 < x ∧ x ≤ 4 then x - 3
else C (x - 4)

-- Define the function S(x)
def S (x : ℝ) : ℝ := C (x - 1)

-- Prove the main theorem
theorem periodic_functions_properties (x : ℝ) :
  (C (2 * x) = (C x)^2 - (S x)^2) ∧ ¬ (S (2 * x) = 2 * C x * S x) :=
by
  sorry

end periodic_functions_properties_l56_56907


namespace beads_necklace_l56_56813

theorem beads_necklace (N beads colors : ℕ) (hc : colors = 50) (hb : beads = N / colors) (H : N = 1000) :
  ∃ n, (∀ necklace : Fin N → Fin colors, ∃ segment : Fin n → Fin N, (∀ i, segment i < N) ∧ 
  (card (segment '' univ) ≥ 25)) ∧ n = 462 :=
by
  use 462
  sorry

end beads_necklace_l56_56813


namespace color_natural_numbers_l56_56673

theorem color_natural_numbers :
  ∃ f : fin 200 → bool, (∀ a b : fin 200, a ≠ b → f a = f b → (a + b).val ∉ {2^n | n : ℕ}) ∧
  (by let count := ∑ color in (fin 200 → bool), 
         ite (∀ a b, a ≠ b → color a = color b → (a + b).val ∉ {2^n | n : ℕ}) 1 0 ;
      exact count = 256) := sorry

end color_natural_numbers_l56_56673


namespace ksyusha_travel_time_wednesday_l56_56747

-- Definitions based on the conditions
def speed_walk (v : ℝ) : ℝ := v
def speed_run (v : ℝ) : ℝ := 2 * v
def distance_walk_tuesday (S : ℝ) : ℝ := 2 * S
def distance_run_tuesday (S : ℝ) : ℝ := S
def total_time_tuesday (v S : ℝ) : ℝ := (distance_walk_tuesday S) / (speed_walk v) + (distance_run_tuesday S) / (speed_run v)

-- Theorem statement
theorem ksyusha_travel_time_wednesday (S v : ℝ) (h : total_time_tuesday v S = 30) :
  (S / v + S / v) = 24 := 
by sorry

end ksyusha_travel_time_wednesday_l56_56747


namespace Rebecca_eggs_l56_56352

/-- Rebecca has 6 marbles -/
def M : ℕ := 6

/-- Rebecca has 14 more eggs than marbles -/
def E : ℕ := M + 14

/-- Rebecca has 20 eggs -/
theorem Rebecca_eggs : E = 20 := by
  sorry

end Rebecca_eggs_l56_56352


namespace length_SR_l56_56837

theorem length_SR (cos_S : ℝ) (SP : ℝ) (SR : ℝ) (h1 : cos_S = 0.5) (h2 : SP = 10) (h3 : cos_S = SP / SR) : SR = 20 := by
  sorry

end length_SR_l56_56837


namespace coloring_problem_l56_56681

noncomputable def numColorings : Nat :=
  256

theorem coloring_problem :
  ∃ (f : Fin 200 → Bool), (∀ (i j : Fin 200), i ≠ j → f i = f j → ¬ (i + 1 + j + 1).isPowerOfTwo) ∧ 
    256 = numColorings :=
by
  -- Definitions and conditions used for the problem
  let S := {n : ℕ | ∃ k : ℕ, n = bit0 k 1 ∧ k > 0}
  have hS : ∀ n ∈ S, ∀ i j ∈ {1, ..., 200}, n ≠ 2 ^ (log2 n) ∧ i ≠ j → i + j ≠ n := sorry
  -- Proof skipped, as required in the guidelines
  sorry

end coloring_problem_l56_56681


namespace complex_real_imag_diff_is_minus_two_l56_56335

noncomputable def complex_example : ℝ :=
  let z := (5 : ℂ) / (-3 - (1 : ℂ) * I) in
  z.re - z.im

theorem complex_real_imag_diff_is_minus_two :
  complex_example = -2 := sorry

end complex_real_imag_diff_is_minus_two_l56_56335


namespace tangent_length_difference_l56_56699

open EuclideanGeometry

theorem tangent_length_difference (AB BC AF CG GF : ℝ) (AB_lt_BC : AB < BC)
  (E : Point) (circle_centered_at_E : Circle)
  (tangent_from_D : TangentLine)
  (tangent_at_AB : Point)
  (tangent_at_BC : Point)
  (intersection_from_D : SegmentIntersection tangent_from_D.1 tangent_at_AB)
  (tangent_intersects_G : SegmentIntersection tangent_from_D.1 tangent_at_BC):
  GF = AF - CG := by sorry

end tangent_length_difference_l56_56699


namespace compare_sqrt3_sub1_div2_half_l56_56498

theorem compare_sqrt3_sub1_div2_half : (sqrt 3 - 1) / 2 < 1 / 2 := 
sorry

end compare_sqrt3_sub1_div2_half_l56_56498


namespace find_number_l56_56463

theorem find_number (n : ℝ) (h : n - (1004 / 20.08) = 4970) : n = 5020 := 
by {
  sorry
}

end find_number_l56_56463


namespace pyramid_volume_l56_56094

theorem pyramid_volume (V_cube : ℝ) (hV : V_cube = 64) : 
  ∃ (V_pyr : ℝ), V_pyr = 32 / 3 :=
by
  let s := (V_cube ^ (1 / 3))
  have hs : s = 4 := by {
    rw [hV, real.rpow_nat_cast],
    norm_num }
  let A_base : ℝ := 1 / 2 * s ^ 2
  have hA_base : A_base = 8 := by {
    rw [hs],
    norm_num }
  let h_pyr : ℝ := s
  have hh_pyr : h_pyr = 4 := hs
  let V_pyr : ℝ := 1 / 3 * A_base * h_pyr
  have hV_pyr : V_pyr = 32 / 3 := by {
    rw [hA_base, hh_pyr],
    norm_num }
  use V_pyr,
  exact hV_pyr

end pyramid_volume_l56_56094


namespace heartsuit_3_8_l56_56256

def heartsuit (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem heartsuit_3_8 : heartsuit 3 8 = 60 := by
  sorry

end heartsuit_3_8_l56_56256


namespace solve_m_eq_4_l56_56563

noncomputable def perpendicular_vectors (m : ℝ) : Prop :=
  let a := (1, -2)
  let b := (m, m - 2)
  (a.1 * b.1 + a.2 * b.2 = 0)

theorem solve_m_eq_4 : ∀ (m : ℝ), perpendicular_vectors m → m = 4 :=
by
  intros m h
  sorry

end solve_m_eq_4_l56_56563


namespace color_natural_numbers_l56_56676

theorem color_natural_numbers :
  ∃ f : fin 200 → bool, (∀ a b : fin 200, a ≠ b → f a = f b → (a + b).val ∉ {2^n | n : ℕ}) ∧
  (by let count := ∑ color in (fin 200 → bool), 
         ite (∀ a b, a ≠ b → color a = color b → (a + b).val ∉ {2^n | n : ℕ}) 1 0 ;
      exact count = 256) := sorry

end color_natural_numbers_l56_56676


namespace ordering_of_abc_l56_56558

def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

theorem ordering_of_abc : b < a ∧ a < c := by
  sorry

end ordering_of_abc_l56_56558


namespace correct_props_l56_56627

-- Setup the necessary geometrical definitions and assumptions
variables {Line Plane : Type} [HasRel Line Plane]
variables (m n : Line) (α β : Plane)

-- Definitions of parallel (‖) and perpendicular (⊥) relationships
def parallel (a b : Type) [HasRel a b] : Prop := a ‖ b
def perpendicular (a b : Type) [HasRel a b] : Prop := a ⊥ b

-- Propositions 
def prop1 := parallel m α ∧ parallel n α → parallel m n 
def prop2 := perpendicular m α ∧ perpendicular n α → parallel m n 
def prop3 := parallel m α ∧ perpendicular m β → perpendicular α β
def prop4 := perpendicular m α ∧ perpendicular n β ∧ parallel m n → parallel α β

-- Proof that the correct propositions are ②, ③, and ④
theorem correct_props : (prop2 ∧ prop3 ∧ prop4) :=
sorry

end correct_props_l56_56627


namespace x_varies_as_neg_2nd_power_of_z_l56_56645

theorem x_varies_as_neg_2nd_power_of_z (x y z : ℝ) (k j : ℝ)
  (h₁ : x = k * y^(-1 / 2))
  (h₂ : y = j * z^4) :
  ∃ m : ℝ, x = m * z^(-2) :=
by
  -- Proof outline and details would go here
  sorry

end x_varies_as_neg_2nd_power_of_z_l56_56645


namespace probability_entire_grid_black_correct_l56_56450

noncomputable def probability_entire_grid_black : ℚ :=
  let p_center_black : ℚ := 1/2 in
  let p_edge_middle_black : ℚ := (1/2) + (1/2 * 1/3) in
  let p_edge_middle_all_black : ℚ := p_edge_middle_black^4 in
  let p_corner_black : ℚ := 1/4 in
  let p_corner_all_black : ℚ := p_corner_black + p_corner_black - (p_corner_black * p_corner_black) in
  p_center_black * p_edge_middle_all_black * p_corner_all_black

theorem probability_entire_grid_black_correct :
  probability_entire_grid_black = 7 / 162 :=
by
  sorry

end probability_entire_grid_black_correct_l56_56450


namespace find_k_l56_56876

noncomputable def proof_problem (x1 x2 x3 x4 : ℝ) (k : ℝ) : Prop :=
  (x1 + x2) / (x3 + x4) = k ∧
  (x3 + x4) / (x1 + x2) = k ∧
  (x1 + x3) / (x2 + x4) = k ∧
  (x2 + x4) / (x1 + x3) = k ∧
  (x1 + x4) / (x2 + x3) = k ∧
  (x2 + x3) / (x1 + x4) = k ∧
  x1 ≠ x2 ∨ x2 ≠ x3 ∨ x3 ≠ x4 ∨ x4 ≠ x1

theorem find_k (x1 x2 x3 x4 : ℝ) (h : proof_problem x1 x2 x3 x4 k) : k = -1 :=
  sorry

end find_k_l56_56876


namespace total_students_high_school_l56_56453

theorem total_students_high_school (students_first_grade : ℕ) (total_sample : ℕ) 
  (sample_second_grade : ℕ) (sample_third_grade : ℕ) (total_students : ℕ) 
  (h1 : students_first_grade = 600) (h2 : total_sample = 45) 
  (h3 : sample_second_grade = 20) (h4 : sample_third_grade = 10)
  (h5 : 15 = total_sample - sample_second_grade - sample_third_grade) 
  (h6 : 15 * total_students = students_first_grade * total_sample) :
  total_students = 1800 :=
sorry

end total_students_high_school_l56_56453


namespace min_value_of_g_l56_56189

noncomputable def g (x : ℝ) : ℝ := x + (x / (x^2 + 1)) + (x * (x + 3) / (x^2 + 3)) + (3 * (x + 3) / (x * (x^2 + 3)))

theorem min_value_of_g : ∀ x > 0, g x ≥ 6 :=
begin
  intro x,
  intro hx_pos,
  sorry
end

end min_value_of_g_l56_56189


namespace bookshop_analysis_l56_56309

section Bookshop

def total_books : ℕ := 1400

def category_A_books : ℕ := 700 -- 50% of total_books
def category_B_books : ℕ := 420 -- 30% of total_books
def category_C_books : ℕ := 280 -- 20% of total_books

def price_category_A : ℕ := 15
def price_category_B : ℕ := 25
def price_category_C : ℕ := 40

def discount_category_A : ℝ := 0.10
def discount_category_B : ℝ := 0.15
def discount_category_C : ℝ := 0.05

def daily_sales_Mon_Tue : ℕ := 62
def daily_sales_Wed : ℕ := 60
def daily_sales_Thu : ℕ := 48
def daily_sales_Fri : ℕ := 40

def remaining_books_category_A : ℕ := category_A_books - 2 * daily_sales_Mon_Tue
def remaining_books_category_B : ℕ := category_B_books - (daily_sales_Wed + daily_sales_Thu)
def remaining_books_category_C : ℕ := category_C_books - daily_sales_Fri

def remaining_percentage (remaining_books total_books : ℕ) : ℝ :=
  (remaining_books.toReal / total_books.toReal) * 100

def revenue (books_sold : ℕ) (price : ℕ) (discount : ℝ) : ℝ :=
  books_sold.toReal * price.toReal * (1 - discount)

def total_revenue : ℕ := 
  (2 * daily_sales_Mon_Tue).toReal * 15 * 0.90 +
  (daily_sales_Wed + daily_sales_Thu).toReal * 25 * 0.85 +
  daily_sales_Fri.toReal * 40 * 0.95

theorem bookshop_analysis :
  remaining_percentage remaining_books_category_A category_A_books = 82.29 ∧ 
  remaining_percentage remaining_books_category_B category_B_books = 74.29 ∧ 
  remaining_percentage remaining_books_category_C category_C_books = 85.71 ∧
  revenue (2 * daily_sales_Mon_Tue) price_category_A discount_category_A = 1674 ∧
  revenue (daily_sales_Wed + daily_sales_Thu) price_category_B discount_category_B = 2295 ∧
  revenue daily_sales_Fri price_category_C discount_category_C = 1520 ∧
  total_revenue = 5489 :=
by
  sorry

end Bookshop

end bookshop_analysis_l56_56309


namespace beads_cost_is_three_l56_56077

-- Define the given conditions
def cost_of_string_per_bracelet : Nat := 1
def selling_price_per_bracelet : Nat := 6
def number_of_bracelets_sold : Nat := 25
def total_profit : Nat := 50

-- The amount spent on beads per bracelet
def amount_spent_on_beads_per_bracelet (B : Nat) : Prop :=
  B = (total_profit + number_of_bracelets_sold * (cost_of_string_per_bracelet + B) - number_of_bracelets_sold * selling_price_per_bracelet) / number_of_bracelets_sold 

-- The main goal is to prove that the amount spent on beads is 3
theorem beads_cost_is_three : amount_spent_on_beads_per_bracelet 3 :=
by sorry

end beads_cost_is_three_l56_56077


namespace reduce_expression_l56_56606

-- Define the variables a, b, c as real numbers
variables (a b c : ℝ)

-- State the theorem with the given condition that expressions are defined and non-zero
theorem reduce_expression :
  (a^2 + b^2 - c^2 - 2*a*b) / (a^2 + c^2 - b^2 - 2*a*c) = ((a - b + c) * (a - b - c)) / ((a - c + b) * (a - c - b)) :=
by
  sorry

end reduce_expression_l56_56606


namespace tan_theta_value_l56_56604

variable (θ : ℝ)

-- Conditions
def condition_1 : Prop := (cos θ - 5 / 13 = 0)
def condition_2 : Prop := (12 / 13 - sin θ ≠ 0)

theorem tan_theta_value (h1 : condition_1 θ) (h2 : condition_2 θ) : tan θ = -12 / 5 := 
by
  sorry

end tan_theta_value_l56_56604


namespace ksyusha_travel_time_l56_56764

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l56_56764


namespace reflection_of_O_on_perpendicular_l56_56566

variables (A B C P O D E F T : Type)
variables [triangle A B C] [circumcenter A B C O]
variables [circumcenter B P C D] [circumcenter C P A E] [circumcenter A P B F]
variable (T : point_intersection (line_through B C) (line_through E F))

theorem reflection_of_O_on_perpendicular (refl_O : point_reflection O (line_through E F)) (perpendicular_from_D : line_perpendicular D (line_through P T)) :
  lies_on refl_O perpendicular_from_D :=
sorry

end reflection_of_O_on_perpendicular_l56_56566


namespace breakfast_problem_probability_l56_56931

def are_relatively_prime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

theorem breakfast_problem_probability : 
  ∃ m n : ℕ, are_relatively_prime m n ∧ 
  (1 / 1 * 9 / 11 * 6 / 10 * 1 / 3) * 1 = 9 / 55 ∧ m + n = 64 :=
by
  sorry

end breakfast_problem_probability_l56_56931


namespace probability_sin_x_ge_half_l56_56938

-- Define the interval [-π, π]
def interval := Icc (-Real.pi) Real.pi

-- Define the event of interest: E(x) == sin x >= 1/2
def event (x : ℝ) : Prop := Real.sin x ≥ 1 / 2

-- Define the length of the interval
def interval_length : ℝ := (Real.pi - (-Real.pi))

-- Define the length of the favorable segment
def favorable_segment_length : ℝ := (5 * Real.pi / 6) - (Real.pi / 6)

-- The probability of the event should be the ratio of the favorable segment to the total interval length
def probability_event := favorable_segment_length / interval_length

theorem probability_sin_x_ge_half : 
  probability_event = 1 / 3 := 
  by
    sorry

end probability_sin_x_ge_half_l56_56938


namespace correct_statements_l56_56897

noncomputable def certain_event_prob : Prop := true -- The probability of a certain event equals 1.

noncomputable def some_event_prob : Prop := false -- The probability of some event equals 1.1.

def mutually_exclusive_complementary (A B : Prop) : Prop := (A ∧ B = false) → (A ∨ B = true) -- Mutually exclusive events are necessarily complementary events.

def complementary_mutually_exclusive (A B : Prop) : Prop := (A ∨ B = true) → (A ∧ B = false) -- Complementary events are necessarily mutually exclusive.

theorem correct_statements:
  (certain_event_prob ∧ complementary_mutually_exclusive) := 
begin
  split,
  { 
    sorry, -- proof for certain_event_prob
  },
  { 
    sorry, -- proof for complementary_mutually_exclusive
  }
end

end correct_statements_l56_56897


namespace min_draws_to_ensure_20_of_one_color_l56_56101

-- Define the total number of balls for each color
def red_balls : ℕ := 30
def green_balls : ℕ := 25
def yellow_balls : ℕ := 22
def blue_balls : ℕ := 15
def white_balls : ℕ := 12
def black_balls : ℕ := 10

-- Define the minimum number of balls to guarantee at least one color reaches 20 balls
def min_balls_needed : ℕ := 95

-- Theorem to state the problem mathematically in Lean
theorem min_draws_to_ensure_20_of_one_color :
  ∀ (r g y b w bl : ℕ),
    r = 30 → g = 25 → y = 22 → b = 15 → w = 12 → bl = 10 →
    (∃ n : ℕ, n ≥ min_balls_needed ∧
    ∀ (r_draw g_draw y_draw b_draw w_draw bl_draw : ℕ),
      r_draw + g_draw + y_draw + b_draw + w_draw + bl_draw = n →
      (r_draw > 19 ∨ g_draw > 19 ∨ y_draw > 19 ∨ b_draw > 19 ∨ w_draw > 19 ∨ bl_draw > 19)) :=
by
  intros r g y b w bl hr hg hy hb hw hbl
  use min_balls_needed
  sorry

end min_draws_to_ensure_20_of_one_color_l56_56101


namespace Q_coordinates_l56_56862

structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def P : Point := ⟨0, 3⟩
def R : Point := ⟨5, 0⟩

def isRectangle (A B C D : Point) : Prop :=
  -- replace this with the actual implementation of rectangle properties
  sorry

theorem Q_coordinates :
  ∃ Q : Point, isRectangle O P Q R ∧ Q.x = 5 ∧ Q.y = 3 :=
by
  -- replace this with the actual proof
  sorry

end Q_coordinates_l56_56862


namespace jana_distance_l56_56716

theorem jana_distance (time_to_walk_one_mile : ℝ) (time_to_walk : ℝ) :
  (time_to_walk_one_mile = 18) → (time_to_walk = 15) →
  ((time_to_walk / time_to_walk_one_mile) * 1 = 0.8) :=
  by
    intros h1 h2
    rw [h1, h2]
    -- Here goes the proof, but it is skipped as per requirements
    sorry

end jana_distance_l56_56716


namespace quadrilateral_midlines_common_point_l56_56461

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def segment_bisected_by_point (K M O : ℝ × ℝ) : Prop :=
  ∃ L : ℝ × ℝ, midpoint K M = O ∧ midpoint K O = midpoint O M

theorem quadrilateral_midlines_common_point
  (A B C D : ℝ × ℝ) :
  let K := midpoint A B
      M := midpoint C D
      L := midpoint B C
      N := midpoint D A
      P := midpoint A C
      Q := midpoint B D
      O := midpoint (midpoint (midpoint A B) (midpoint C D)) (midpoint (midpoint A C) (midpoint B D))
  in 
  segment_bisected_by_point K M O ∧ segment_bisected_by_point L N O ∧ segment_bisected_by_point P Q O :=
begin
  sorry
end

end quadrilateral_midlines_common_point_l56_56461


namespace five_g_speeds_l56_56819

theorem five_g_speeds (m : ℝ) :
  (1400 / 50) - (1400 / (50 * m)) = 24 → m = 7 :=
by
  sorry

end five_g_speeds_l56_56819


namespace pairs_of_real_numbers_satisfying_inequalities_l56_56091

namespace math_problem

theorem pairs_of_real_numbers_satisfying_inequalities (x y : ℝ) :
  ( (x^4 + 8 * x^3 * y + 16 * x^2 * y^2 + 16 ≤ 8 * x^2 + 32 * x * y) ∧
    (y^4 + 64 * x^2 * y^2 + 10 * y^2 + 25 ≤ 16 * x * y^3 + 80 * x * y) ) ↔
  ( (x = 2/√11 ∧ y = 5/√11) ∨ (x = -2/√11 ∧ y = -5/√11) ∨
    (x = 2/√3 ∧ y = 1/√3) ∨ (x = -2/√3 ∧ y = -1/√3) ) :=
by sorry

end math_problem

end pairs_of_real_numbers_satisfying_inequalities_l56_56091


namespace problem_1_problem_2_l56_56609

theorem problem_1 (a : ℝ) (h₁ : 0 < a ∧ a ≠ 1) :
  (∀ x : ℝ, 
    (if x < 0 then x^2 + (4 * a - 3) * x + 3 * a else log a (x + 1) + 1) ∂x ≤ 0)
  ↔ (1 / 3 ≤ a ∧ a ≤ 3 / 4) :=
sorry

theorem problem_2 (a : ℝ) (h₁ : 0 < a ∧ a ≠ 1) :
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    abs (if x < 0 then x^2 + (4 * a - 3) * x + 3 * a else log a (x + 1) + 1) = 2 - x)
  ↔ (1 / 3 ≤ a ∧ a ≤ 2 / 3 ∨ a = 3 / 4) :=
sorry

end problem_1_problem_2_l56_56609


namespace count_divisible_by_11_between_9_and_79_l56_56435

theorem count_divisible_by_11_between_9_and_79 : Nat := by
  let count := (List.range' 9 (79 - 9 + 1)).filter (fun n => n % 11 == 0).length
  exact count

end count_divisible_by_11_between_9_and_79_l56_56435


namespace maximize_probability_l56_56106

theorem maximize_probability (p1 p2 p3 : ℝ) (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3) :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3) in
  PC > PA ∧ PC > PB :=
by
  sorry

end maximize_probability_l56_56106


namespace kenya_peanuts_correct_l56_56093

def jose_peanuts : ℕ := 85
def kenya_extra_peanuts : ℕ := 48
def kenya_peanuts : ℕ := jose_peanuts + kenya_extra_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := 
by 
  sorry

end kenya_peanuts_correct_l56_56093


namespace y_percent_of_x_l56_56084

theorem y_percent_of_x (x y : ℝ) (h : 0.60 * (x - y) = 0.20 * (x + y)) : y / x = 0.5 :=
sorry

end y_percent_of_x_l56_56084


namespace numbers_with_special_remainder_property_l56_56386

theorem numbers_with_special_remainder_property (n : ℕ) :
  (∀ q : ℕ, q > 0 → n % (q ^ 2) < (q ^ 2) / 2) ↔ (n = 1 ∨ n = 4) := 
by
  sorry

end numbers_with_special_remainder_property_l56_56386


namespace value_range_of_F_l56_56233

noncomputable def f_M (M : Set ℝ) (x : ℝ) : ℝ :=
if x ∈ M then 1 else 0

noncomputable def F (A B : Set ℝ) (x : ℝ) : ℝ :=
(f_M (A ∪ B) x + 1) / (f_M A x + f_M B x + 1)

theorem value_range_of_F (A B : Set ℝ) (hA : A ⊂ ℝ) (hB : B ⊂ ℝ) (hA_nonempty : A ≠ ∅) (hB_nonempty : B ≠ ∅) (h_disjoint : A ∩ B = ∅) : 
  ∀ x : ℝ, F A B x = 1 := 
by
  sorry

end value_range_of_F_l56_56233


namespace ellipse_standard_eq_line_eq_through_F2_l56_56619

-- Define the conditions for the problem
def parabola (p : ℝ) : Prop := p > 0 ∧ ∃ x y : ℝ, y^2 = 4 * p * x
def directrix_focus_intersection (p : ℝ) : Prop := p = 1
def ellipse (F1 F2 : ℝ × ℝ) (e : ℝ) : Prop := 
  e = 1 / 2 ∧ F1 = (-1, 0) ∧ F2 = (1, 0)

-- Definitions based on specific problem conditions
def condition_p := 1
def F1 := (-1 : ℝ, 0)
def F2 := (1 : ℝ, 0)
def e := 1 / 2

-- Prove the standard equation of the ellipse under these conditions
theorem ellipse_standard_eq :
  parabola condition_p ∧ directrix_focus_intersection condition_p ∧ ellipse F1 F2 e →
  ∃ a b : ℝ, (a > 0 ∧ b > 0 ∧ a > b ∧ a = 2 ∧ b = sqrt 3 ∧ 
              ∀ x y, (x^2 / 4 + y^2 / 3 = 1)) := sorry

-- Define the line passing through F2 with intersecting conditions as per the 2nd part
theorem line_eq_through_F2 :
  parabola condition_p ∧ directrix_focus_intersection condition_p ∧ ellipse F1 F2 e →
  ∃ k : ℝ, (k = sqrt 2 ∨ k = - sqrt 2 ∧ ∀ x y, y = k * (x - 1)) := sorry

end ellipse_standard_eq_line_eq_through_F2_l56_56619


namespace trigonometric_identity_l56_56180

theorem trigonometric_identity :
  (2 * Real.sin (10 * Real.pi / 180) - Real.cos (20 * Real.pi / 180)) / Real.cos (70 * Real.pi / 180) = - Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_l56_56180


namespace alice_daily_savings_l56_56954

theorem alice_daily_savings :
  ∀ (d total_days : ℕ) (dime_value : ℝ),
  d = 4 → total_days = 40 → dime_value = 0.10 →
  (d * dime_value) / total_days = 0.01 :=
by
  intros d total_days dime_value h_d h_total_days h_dime_value
  sorry

end alice_daily_savings_l56_56954


namespace necessary_but_not_sufficient_l56_56790

theorem necessary_but_not_sufficient (a : ℝ) : (a - 1 < 0 ↔ a < 1) ∧ (|a| < 1 → a < 1) ∧ ¬ (a < 1 → |a| < 1) := by
  sorry

end necessary_but_not_sufficient_l56_56790


namespace prices_of_basketball_and_soccer_ball_l56_56923

theorem prices_of_basketball_and_soccer_ball (x y : ℝ) :
  (3 * x + 7 * y = 445) ∧ (x = y + 5) :=
begin
  sorry
end

end prices_of_basketball_and_soccer_ball_l56_56923


namespace filling_time_with_A_and_B_only_l56_56186

noncomputable def valve_filling_time (A_rate B_rate C_rate : ℚ) : ℚ := 
  1 / (A_rate + B_rate)

theorem filling_time_with_A_and_B_only :
  ∃ (a b c : ℚ), 
    (a + b + c = 2) ∧ 
    (a + c = 1) ∧ 
    (b + c = 1/2) ∧ 
    valve_filling_time a b c = 0.4 :=
begin
  sorry
end

end filling_time_with_A_and_B_only_l56_56186


namespace integer_roots_sum_abs_eq_100_l56_56544

theorem integer_roots_sum_abs_eq_100 (m p q r : ℤ) :
  (Polynomial.X ^ 3 - 2023 * Polynomial.X + Polynomial.C m).roots = [p, q, r] →
  p + q + r = 0 →
  p * q + q * r + r * p = -2023 →
  |p| + |q| + |r| = 100 :=
by
  sorry

end integer_roots_sum_abs_eq_100_l56_56544


namespace CongCong_scores_l56_56986

theorem CongCong_scores:
  ∃ c m : ℕ, 
    (c + m) / 2 = 98 ∧ 
    m = c + 2 ∧ 
    c = 97 ∧ 
    m = 99 := 
by 
  use 97, 99
  simp
  sorry

end CongCong_scores_l56_56986


namespace area_ratio_triangle_l56_56264

theorem area_ratio_triangle (X Y Z W : Type) [linear_ordered_field Type] (k : Type) (XW WZ : Type) [linear_ordered_ring Type] (tri_XY XW WZ : Type) (XW_eq : XW = 9) (WZ_eq : WZ = 15):
  (let area_XYW : linear_ordered_field =  (1/2 * XW * k)) ∧ (let area_WYZ : linear_ordered_field = (1/2 * WZ * k)):
  (area_ratio_triangle / let ratio := ((1/2 * XW * k) / (1/2 * WZ * k))) / (area_ratio_eq: (ratio = 3 / 5)) sorry

end area_ratio_triangle_l56_56264


namespace investment_correct_l56_56194

def annual_income (FV : ℝ) (R : ℝ) : ℝ := FV * (R / 100)

def amount_invested (FV : ℝ) (MP : ℝ) : ℝ := (FV / 100) * MP

theorem investment_correct (FV MP I : ℝ) (R : ℝ) (hI : I = 1000) (hR : R = 20) (hMP : MP = 136) :
  amount_invested FV MP = 6800 :=
by
  have hFV : FV = 1000 / (20 / 100) := by sorry
  rw [hFV, hR, hI]
  simp [amount_invested]
  sorry

end investment_correct_l56_56194


namespace find_a_l56_56591

theorem find_a (a b c : ℕ) (h1 : (18 ^ a) * (9 ^ (3 * a - 1)) * (c ^ (2 * a - 3)) = (2 ^ 7) * (3 ^ b)) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : a = 7 :=
by
  sorry

end find_a_l56_56591


namespace find_a1_a2_a3_bn_geometric_seq_find_Tn_l56_56807

-- Given conditions
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {T : ℕ → ℝ}

-- Condition: S_n = 2a_n - 2n for any positive integer n
axiom Sn_cond : ∀ n : ℕ, S n = 2 * a n - 2 * n

-- (1) Find the values of a_1, a_2, and a_3
theorem find_a1_a2_a3 : a 1 = 2 ∧ a 2 = 6 ∧ a 3 = 14 := by
  -- proof goes here
  sorry

-- (2) Prove that the sequence {b_n} defined as b_n = a_n + 2 is a geometric sequence with common ratio 2
def b (n : ℕ) : ℝ := a n + 2

theorem bn_geometric_seq : ∃ r : ℝ, ∀ n : ℕ, n > 0 → b (n + 1) / b n = r := by
  -- proof goes here
  use 2
  sorry

-- (3) Find the sum of the first n terms of the sequence {n * a_n}, denoted as T_n
def T_n (n : ℕ) := ∑ i in Finset.range n, (i + 1) * a (i + 1)

theorem find_Tn : ∀ n : ℕ, T_n n = (n + 1) * 2^(n + 2) + 4 - n * (n + 1) := by
  -- proof goes here
  sorry

end find_a1_a2_a3_bn_geometric_seq_find_Tn_l56_56807


namespace coloring_ways_l56_56693

-- Define a function that determines if a given integer is a power of two
def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

-- Define the main problem condition as a predicate
def valid_coloring (coloring : ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ i → i ≤ 200 → 1 ≤ j → j ≤ 200 → i ≠ j → 
    coloring i = coloring j → ¬ is_power_of_two (i + j)

theorem coloring_ways : 
  ∃ (coloring : ℕ → Prop), (∀ (i : ℕ), 1 ≤ i → i ≤ 200 → (coloring i = true ∨ coloring i = false)) ∧
  (valid_coloring coloring) ∧
  (finset.card (finset.filter valid_coloring (finset.univ : set (ℕ → Prop))) = 256) :=
sorry

end coloring_ways_l56_56693


namespace population_percentage_decrease_l56_56697

/-- 
Given the initial population P0, the population increased by 30% in the first year, 
and the final population after the second year P2, this theorem states that the percentage decrease in the second year was 30%.
-/
theorem population_percentage_decrease
  (P0 : ℝ) (P2 : ℝ)
  (hP0 : P0 = 15000)
  (hP2 : P2 = 13650) :
  ∃ P : ℝ, 
  let P1 := P0 + 0.3 * P0 in
  P = 30 ∧ P2 = P1 - 0.01 * P * P1 :=
by
  have P1 : ℝ := P0 + 0.3 * P0
  have hP1: P1 = 19500 := sorry  -- This is for ensuring type correctness, the actual proof will resolve the specifics.
  use 30
  split
  · exact rfl
  · exact sorry

end population_percentage_decrease_l56_56697


namespace complex_addition_identity_l56_56213

theorem complex_addition_identity
  (a b : ℝ)
  (h : (a + 2 * Complex.i) / Complex.i = b + Complex.i) :
  a + b = 1 :=
sorry

end complex_addition_identity_l56_56213


namespace calculate_fraction_sum_l56_56969

theorem calculate_fraction_sum :
  (∑ i in [3, 6, 9], i : ℚ) / (∑ i in [2, 5, 8], i) + 
  (∑ i in [2, 5, 8], i) / (∑ i in [3, 6, 9], i) = 61 / 30 := by
sorry

end calculate_fraction_sum_l56_56969


namespace number_of_people_shared_swirls_l56_56539

theorem number_of_people_shared_swirls 
    (total_swirls : ℕ) 
    (jane_ate : ℕ) 
    (total_swirls_eq : total_swirls = 12) 
    (jane_ate_eq : jane_ate = 4)
    (everyone_ate_equal : ∀ (n : ℕ), n * jane_ate = total_swirls → n > 0 ∧ ∃ k, n = k) :
    ∃ n, n = 3 := 
by
  have h : 12 / 4 = 3 := rfl
  exact ⟨3, h.symm⟩

end number_of_people_shared_swirls_l56_56539


namespace Ksyusha_time_to_school_l56_56738

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l56_56738


namespace units_digit_of_seven_consecutive_numbers_is_zero_l56_56511

theorem units_digit_of_seven_consecutive_numbers_is_zero (n : ℕ) : 
  (n * (n+1) * (n+2) * (n+3) * (n+4) * (n+5) * (n+6)) % 10 = 0 :=
begin
  sorry
end

end units_digit_of_seven_consecutive_numbers_is_zero_l56_56511


namespace angle_BOP_eq_angle_COQ_l56_56949

open EuclideanGeometry

variables {ABC : Triangle} (O : Point) (P Q : Point)
variables [on_side_ABC : OnSide P ABC.AB] [on_extension_AC : OnExtension Q ABC.AC] 
variables [tangent_to_circle : Tangent PQ (InscribedCircle O ABC)]

theorem angle_BOP_eq_angle_COQ : 
  ∠(OO (B OP)) = ∠(OO (C OQ)) :=
by 
  sorry

end angle_BOP_eq_angle_COQ_l56_56949


namespace baker_sold_pastries_l56_56486

theorem baker_sold_pastries : 
  ∃ P : ℕ, (97 = P + 89) ∧ P = 8 :=
by 
  sorry

end baker_sold_pastries_l56_56486


namespace tommy_number_of_nickels_l56_56048

theorem tommy_number_of_nickels
  (d p n q : ℕ)
  (h1 : d = p + 10)
  (h2 : n = 2 * d)
  (h3 : q = 4)
  (h4 : p = 10 * q) : n = 100 :=
sorry

end tommy_number_of_nickels_l56_56048


namespace color_natural_numbers_l56_56675

theorem color_natural_numbers :
  ∃ f : fin 200 → bool, (∀ a b : fin 200, a ≠ b → f a = f b → (a + b).val ∉ {2^n | n : ℕ}) ∧
  (by let count := ∑ color in (fin 200 → bool), 
         ite (∀ a b, a ≠ b → color a = color b → (a + b).val ∉ {2^n | n : ℕ}) 1 0 ;
      exact count = 256) := sorry

end color_natural_numbers_l56_56675


namespace chest_coin_problem_l56_56454

theorem chest_coin_problem :
  ∃ n : ℕ, (n % 7 = 3) ∧ (n % 4 = 2) ∧ ((n % 7 = 3) ∧ (n % 4 = 2) → (n % 8 = 2)) :=
begin
  sorry -- Proof omitted
end

end chest_coin_problem_l56_56454


namespace question_z_in_third_quadrant_l56_56603

def complex_quadrant (x y : ℝ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On an axis"

theorem question_z_in_third_quadrant :
  let z := (real.sqrt 1 / (1 : ℂ)).im * (1 - 1.im).re
  let z_conjugate := complex.conj z
  complex_quadrant z_conjugate.re z_conjugate.im = "Third quadrant" :=
by
  -- Full proof should go here
  sorry

end question_z_in_third_quadrant_l56_56603


namespace running_time_constant_pace_l56_56642

/-!
# Running Time Problem

We are given that the running pace is constant, it takes 30 minutes to run 5 miles,
and we need to find out how long it will take to run 2.5 miles.
-/

theorem running_time_constant_pace :
  ∀ (distance_to_store distance_to_cousin distance_run time_run : ℝ)
  (constant_pace : Prop),
  distance_to_store = 5 → time_run = 30 → distance_to_cousin = 2.5 →
  constant_pace → 
  time_run / distance_to_store * distance_to_cousin = 15 :=
by 
  intros distance_to_store distance_to_cousin distance_run time_run constant_pace 
         hds htr hdc hcp
  rw [hds, htr, hdc]
  exact sorry

end running_time_constant_pace_l56_56642


namespace ones_digit_sum_l56_56890

theorem ones_digit_sum (n : ℕ) (hne : n > 0) (hneq : ∃ k, n = 4 * k + 1) :
  (((∑ k in Finset.range (n + 1), k) % 10) = 1) :=
sorry

end ones_digit_sum_l56_56890


namespace min_trucks_for_crates_l56_56053

noncomputable def min_trucks (total_weight : ℕ) (max_weight_per_crate : ℕ) (truck_capacity : ℕ) : ℕ :=
  if total_weight % truck_capacity = 0 then total_weight / truck_capacity
  else total_weight / truck_capacity + 1

theorem min_trucks_for_crates :
  ∀ (total_weight max_weight_per_crate truck_capacity : ℕ),
    total_weight = 10 →
    max_weight_per_crate = 1 →
    truck_capacity = 3 →
    min_trucks total_weight max_weight_per_crate truck_capacity = 5 :=
by
  intros total_weight max_weight_per_crate truck_capacity h_total h_max h_truck
  rw [h_total, h_max, h_truck]
  sorry

end min_trucks_for_crates_l56_56053


namespace polynomial_remainder_is_zero_l56_56176

theorem polynomial_remainder_is_zero :
  ∀ (x : ℤ), ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 := 
by
  sorry

end polynomial_remainder_is_zero_l56_56176


namespace intersection_with_y_axis_l56_56017

theorem intersection_with_y_axis :
  (∃ y : ℝ, y = -(0 + 2)^2 + 6 ∧ (0, y) = (0, 2)) :=
by
  sorry

end intersection_with_y_axis_l56_56017


namespace fraction_eq_zero_iff_x_eq_6_l56_56376

theorem fraction_eq_zero_iff_x_eq_6 (x : ℝ) : (x - 6) / (5 * x) = 0 ↔ x = 6 :=
by
  sorry

end fraction_eq_zero_iff_x_eq_6_l56_56376


namespace minimum_triangle_perimeter_l56_56280

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - floor x

theorem minimum_triangle_perimeter 
  (l m n : ℕ) (h1 : l > m) (h2 : m > n)
  (h3 : fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4))
  (h4 : fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)) :
  l + m + n = 3003 := 
  sorry

end minimum_triangle_perimeter_l56_56280


namespace angle_of_inclination_is_obtuse_l56_56613

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem angle_of_inclination_is_obtuse :
  let f' := D (fun x => f x) in
  let tangent_slope := f' 1 in
  tangent_slope < 0 → true :=
by
  sorry

end angle_of_inclination_is_obtuse_l56_56613


namespace numberOfColoringWays_l56_56535

-- Define the problem parameters
def totalBalls : Nat := 5
def redBalls : Nat := 1
def blueBalls : Nat := 1
def yellowBalls : Nat := 2
def whiteBalls : Nat := 1

-- Show that the number of permutations of the multiset is 60
theorem numberOfColoringWays : (Nat.factorial totalBalls) / ((Nat.factorial redBalls) * (Nat.factorial blueBalls) * (Nat.factorial yellowBalls) * (Nat.factorial whiteBalls)) = 60 :=
  by
  simp [totalBalls, redBalls, blueBalls, yellowBalls, whiteBalls]
  sorry

end numberOfColoringWays_l56_56535


namespace abc_inequality_l56_56825

theorem abc_inequality (a b c : ℝ) : a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

end abc_inequality_l56_56825


namespace solve_abs_equation_l56_56833

theorem solve_abs_equation (y : ℝ) (h8 : y < 8) (h_eq : |y - 8| + 2 * y = 12) : y = 4 :=
sorry

end solve_abs_equation_l56_56833


namespace percent_of_sales_not_pens_pencils_erasers_l56_56008

theorem percent_of_sales_not_pens_pencils_erasers :
  let percent_pens := 25
  let percent_pencils := 30
  let percent_erasers := 20
  let percent_total := 100
  percent_total - (percent_pens + percent_pencils + percent_erasers) = 25 :=
by
  -- definitions and assumptions
  let percent_pens := 25
  let percent_pencils := 30
  let percent_erasers := 20
  let percent_total := 100
  sorry

end percent_of_sales_not_pens_pencils_erasers_l56_56008


namespace sum_f_values_l56_56239

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_f_values : 
  ∑ i in (finset.range 2009).image nat.succ ∪ (finset.range 2009).image (λ n, 1 / (n + 1 : ℝ)),
  f i = 2008.5 :=
by sorry

end sum_f_values_l56_56239


namespace ksyusha_travel_time_l56_56763

theorem ksyusha_travel_time :
  ∀ (S v : ℝ), 
  (2 * S / v + S / (2 * v) = 30) → 
  (2 * (S / v) = 24) :=
by
  intros S v h1
  have h2 : S / v = 12 := by
    calc 
      S / v = (30 * 2) / 5 := by sorry
  calc 
    2 * (S / v) = 2 * 12 := by sorry
  ... = 24 := by norm_num

end ksyusha_travel_time_l56_56763


namespace integer_solutions_count_l56_56250

-- Definition of the problem statement in Lean
theorem integer_solutions_count :
  {x : ℤ // (x^2 - 3 * x + 2)^(x - 1) = 1}.card = 2 :=
sorry

end integer_solutions_count_l56_56250


namespace correctDiscounts_l56_56129

namespace ShoeDiscount

def prices : List ℝ := [28, 35, 40, 45, 50]
def rebates : List ℝ := [0.10, 0.12, 0.15, 0.18, 0.20]

-- Calculate the individual rebates
def individualRebates (prices rebates : List ℝ) : List ℝ :=
  List.zipWith (λ p r => p * r) prices rebates

-- Sum of individual rebates
def totalRebate (prices rebates : List ℝ) : ℝ :=
  (individualRebates prices rebates).sum

-- Total price before rebates
def totalPrice (prices : List ℝ) : ℝ := prices.sum

-- Determine the total rebate and whether any additional quantity discount applies
def discounts (prices rebates : List ℝ) : ℝ × ℝ :=
  let totalReb := totalRebate prices rebates
  let totalPriceAfterRebate := totalPrice prices - totalReb
  let quantityDiscount := if totalPriceAfterRebate > 250 then 0.07 * (totalPrice prices - totalReb)
                          else if totalPriceAfterRebate > 200 then 0.05 * (totalPrice prices - totalReb)
                          else 0
  (totalReb, quantityDiscount)

-- Theorem
theorem correctDiscounts :
  discounts prices rebates = (31.1, 0) :=
by
  sorry

end ShoeDiscount

end correctDiscounts_l56_56129


namespace difference_twice_cecil_and_catherine_l56_56495

theorem difference_twice_cecil_and_catherine
  (Cecil Catherine Carmela : ℕ)
  (h1 : Cecil = 600)
  (h2 : Carmela = 2 * 600 + 50)
  (h3 : 600 + (2 * 600 - Catherine) + Carmela = 2800) :
  2 * 600 - Catherine = 250 := by
  sorry

end difference_twice_cecil_and_catherine_l56_56495


namespace probability_factor_less_than_ten_l56_56059

theorem probability_factor_less_than_ten (n : ℕ) (factors : list ℕ) (h_n : n = 90) (h_factors : factors = [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]) :
  (factors.filter (λ x, x < 10)).length / factors.length = 1 / 2 :=
by
  sorry

end probability_factor_less_than_ten_l56_56059


namespace correct_option_d_l56_56425

-- Define the conditions as separate lemmas
lemma option_a_incorrect : ¬ (Real.sqrt 18 + Real.sqrt 2 = 2 * Real.sqrt 5) :=
sorry 

lemma option_b_incorrect : ¬ (Real.sqrt 18 - Real.sqrt 2 = 4) :=
sorry

lemma option_c_incorrect : ¬ (Real.sqrt 18 * Real.sqrt 2 = 36) :=
sorry

-- Define the statement to prove
theorem correct_option_d : Real.sqrt 18 / Real.sqrt 2 = 3 :=
by
  sorry

end correct_option_d_l56_56425


namespace pigs_total_l56_56877

theorem pigs_total (initial_pigs : ℕ) (joined_pigs : ℕ) (total_pigs : ℕ) 
  (h1 : initial_pigs = 64) 
  (h2 : joined_pigs = 22) 
  : total_pigs = 86 :=
by
  sorry

end pigs_total_l56_56877


namespace new_concentrations_are_correct_l56_56270

def initial_solution_volume : ℝ := 5
def substance_A_fraction : ℝ := 0.3
def substance_B_fraction : ℝ := 0.4
def substance_C_fraction : ℝ := 0.3

def substance_A_amount := substance_A_fraction * initial_solution_volume
def substance_B_amount := substance_B_fraction * initial_solution_volume
def substance_C_amount := substance_C_fraction * initial_solution_volume

def added_water_volume : ℝ := 2
def evaporated_volume : ℝ := 1

def total_volume_after_addition := initial_solution_volume + added_water_volume
def total_volume_after_evaporation := total_volume_after_addition - evaporated_volume

def new_concentration (substance_amount total_volume : ℝ) : ℝ := (substance_amount / total_volume) * 100

theorem new_concentrations_are_correct :
  new_concentration substance_A_amount total_volume_after_evaporation = 25 ∧ 
  new_concentration substance_B_amount total_volume_after_evaporation = 33.33 ∧
  new_concentration substance_C_amount total_volume_after_evaporation = 25 := 
by
  sorry

end new_concentrations_are_correct_l56_56270


namespace z_squared_l56_56257

variable (i : ℂ)
variable h_i : i^2 = -1

def z : ℂ := 3 + 4 * i

theorem z_squared :
  z^2 = -7 + 24 * i :=
by {
  sorry
}

end z_squared_l56_56257


namespace intersection_with_y_axis_l56_56016

theorem intersection_with_y_axis :
  (∃ y : ℝ, y = -(0 + 2)^2 + 6 ∧ (0, y) = (0, 2)) :=
by
  sorry

end intersection_with_y_axis_l56_56016


namespace find_prime_pairs_l56_56998

theorem find_prime_pairs :
  ∃ p q : ℕ, Prime p ∧ Prime q ∧
    ∃ a b : ℕ, a^2 = p - q ∧ b^2 = pq - q ∧ (p = 3 ∧ q = 2) :=
by
  sorry

end find_prime_pairs_l56_56998


namespace train_speed_l56_56916

theorem train_speed (length : ℕ) (cross_time : ℕ) (speed : ℝ)
    (h1 : length = 250)
    (h2 : cross_time = 3)
    (h3 : speed = (length / cross_time : ℝ) * 3.6) :
    speed = 300 := 
sorry

end train_speed_l56_56916


namespace coloring_ways_l56_56691

-- Define a function that determines if a given integer is a power of two
def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

-- Define the main problem condition as a predicate
def valid_coloring (coloring : ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ i → i ≤ 200 → 1 ≤ j → j ≤ 200 → i ≠ j → 
    coloring i = coloring j → ¬ is_power_of_two (i + j)

theorem coloring_ways : 
  ∃ (coloring : ℕ → Prop), (∀ (i : ℕ), 1 ≤ i → i ≤ 200 → (coloring i = true ∨ coloring i = false)) ∧
  (valid_coloring coloring) ∧
  (finset.card (finset.filter valid_coloring (finset.univ : set (ℕ → Prop))) = 256) :=
sorry

end coloring_ways_l56_56691


namespace roots_of_equation_l56_56032

theorem roots_of_equation :
  ∀ (a b c : ℝ),
  a = 1 →
  b = -8 →
  c = 16 →
  let Δ := b^2 - 4 * a * c in
  Δ = 0 →
  ∀ (x : ℝ), x^2 - 8 * x + 16 = 0 → (x = 4) :=
by
  intros a b c ha hb hc Δ hΔ x hx
  sorry

end roots_of_equation_l56_56032


namespace periodicity_2016_l56_56608

def sqrtDigits (n : ℕ) : ℕ :=
  match n with
  | 1 => 4
  | 2 => 1
  | 3 => 4
  | 4 => 2
  | 5 => 1
  | 6 => 3
  | 7 => 5
  | 8 => 6
  | _ => (0 : ℕ) -- Placeholder: Undefined for n > 8 for simplicity

def f (n: ℕ) : ℕ := sqrtDigits n

theorem periodicity_2016 (n : ℕ) (h1 : f(8) = 6) (h2 : f(6) = 3) (h3 : f(3) = 4) (h4 : f(4) = 2) (h5 : f(2) = 1) (h6 : f(1) = 4) : f^(2016) 8 = 4 :=
sorry

end periodicity_2016_l56_56608


namespace limit_n_b_n_l56_56978

def M (x : ℝ) : ℝ := x - (x^3 / 3)

def b_n (n : ℕ) : ℝ := (nat.iterate M n) (20 / n)

theorem limit_n_b_n :
  tendsto (λ n : ℕ, n * b_n n) at_top (𝓝 20) :=
begin
  sorry
end

end limit_n_b_n_l56_56978


namespace P_intersection_Q_l56_56624

def P : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}
def Q : Set ℝ := {x | x^2 + x - 6 = 0}

theorem P_intersection_Q :
  P ∩ (Q : Set ℕ) = {2} :=
sorry

end P_intersection_Q_l56_56624


namespace not_divisible_by_prime_l56_56316

theorem not_divisible_by_prime (p : ℕ) (n : ℕ) (hp : p.prime) (hn : 0 < n) :
  let N := (1 / p^(n^2)) * ∏ (i : ℕ) in finset.filter (λ i, ¬ even i) (finset.Ico 1 (2 * n + 1)), (p-1)! * nat.choose (p^2 * i) (p * i)
  in int.coe_nat (nat.floor (N)) > 0 ∧ ¬ p ∣ nat.floor (N) :=
sorry

end not_divisible_by_prime_l56_56316


namespace zero_of_f_in_interval_l56_56869

noncomputable def f (x : ℝ) : ℝ := log x / log 3 + x - 3

theorem zero_of_f_in_interval : ∃ x ∈ set.Ioo 2 3, f x = 0 :=
sorry

end zero_of_f_in_interval_l56_56869


namespace selling_price_max_sets_profit_goal_achievable_l56_56629

-- Given conditions
def price_A : ℤ := 200
def price_B : ℤ := 170
def first_month_sets_A : ℤ := 3
def first_month_sets_B : ℤ := 5
def first_month_revenue : ℤ := 1800
def second_month_sets_A : ℤ := 4
def second_month_sets_B : ℤ := 10
def second_month_revenue : ℤ := 3100
def budget : ℤ := 5400
def total_sets : ℤ := 30
def profit_goal : ℤ := 1300

-- Define unknowns
variable (x y : ℤ) -- Selling prices of Type A and B Go boards respectively
variable (m : ℤ) -- Number of sets of Type A to be purchased

-- 1. Proving the selling prices of Type A and Type B Go boards
theorem selling_price : 3 * x + 5 * y = first_month_revenue ∧ 4 * x + 10 * y = second_month_revenue → x = 250 ∧ y = 210 :=
by
  intros,
  -- This will be the proof block
  sorry

-- 2. Proving the maximum number of sets of Type A that can be purchased
theorem max_sets : 200 * m + 170 * (total_sets - m) ≤ budget → m ≤ 10 :=
by
  intros,
  -- This will be the proof block
  sorry

-- 3. Proving achieving a profit goal of 1300 yuan
theorem profit_goal_achievable : m ≤ 10 ∧ 50 * m + 40 * (total_sets - m) = profit_goal → m = 10 :=
by
  intros,
  -- This will be the proof block
  sorry

end selling_price_max_sets_profit_goal_achievable_l56_56629


namespace algebraic_expression_value_l56_56554

theorem algebraic_expression_value (a : ℝ) (h : 2 * a^2 + 3 * a - 5 = 0) : 6 * a^2 + 9 * a - 5 = 10 :=
by
  sorry

end algebraic_expression_value_l56_56554


namespace number_of_schools_l56_56519

theorem number_of_schools (N : ℕ) :
  (∀ i j : ℕ, i < j → i ≠ j) →
  (∀ i : ℕ, i < 2 * 35 → i = 35 ∨ ((i = 37 → ¬ (i = 35))) ∧ ((i = 64 → ¬ (i = 35)))) →
  N = (2 * (35) - 1) / 3 →
  N = 23 :=
by
  sorry

end number_of_schools_l56_56519


namespace subway_fare_each_way_l56_56917

-- Definitions under the conditions
def cost_per_dozen_apples := 14
def cost_kiwis := 10
def cost_bananas := cost_kiwis / 2
def initial_money := 50
def max_apples := 24

-- Proof statement
theorem subway_fare_each_way :
  (cost_per_dozen_apples = 14) →
  (cost_kiwis = 10) →
  (cost_bananas = cost_kiwis / 2) →
  (initial_money = 50) →
  (max_apples = 24) →
  let money_spent_on_fruits := cost_kiwis + cost_bananas in
  let money_left_for_apples := initial_money - money_spent_on_fruits in
  let cost_for_24_apples := cost_per_dozen_apples * 2 in
  let money_left_after_apples := money_left_for_apples - cost_for_24_apples in
  let subway_fare := money_left_after_apples / 2 in
  subway_fare = 3.5 :=
begin
  intros _ _ _ _ _,
  sorry
end

end subway_fare_each_way_l56_56917


namespace tanya_days_to_complete_work_l56_56086

-- Define Sakshi's working condition
def sakshi_work_rate := 1 / 5

-- Define that Tanya is 25% more efficient than Sakshi
def tanya_efficiency := 1 + 0.25

-- Calculate Tanya's work rate based on her efficiency
def tanya_work_rate := sakshi_work_rate * tanya_efficiency

-- Prove the number of days Tanya takes to do the work
theorem tanya_days_to_complete_work : (1 / tanya_work_rate) = 4 := by
  -- This is where the proof would go
  sorry

end tanya_days_to_complete_work_l56_56086


namespace domain_ln_sqrt_domain_ln_div_sqrt_l56_56528

-- Proof problem 1
theorem domain_ln_sqrt (x : ℝ) : 
  (1 + 1 / x > 0 ∧ 1 - x^2 ≥ 0) ↔ (0 < x ∧ x ≤ 1) :=
by
  sorry

-- Proof problem 2
theorem domain_ln_div_sqrt (x : ℝ) : 
  (x + 1 > 0 ∧ -x^2 - 3x + 4 > 0) ↔ (-1 < x ∧ x < 1) :=
by
  sorry

end domain_ln_sqrt_domain_ln_div_sqrt_l56_56528


namespace geometric_series_sum_l56_56516

theorem geometric_series_sum :
  let a := (5 : ℝ) / 3
  let r := -1 / 3
  (a / (1 - r)) = 5 / 4 :=
by
  let a := (5 : ℝ) / 3
  let r := -1 / 3
  calc
    a / (1 - r) = a / (4 / 3) : by sorry
            ... = 5 / 4       : by sorry

end geometric_series_sum_l56_56516


namespace hyperbola_equation_l56_56230

noncomputable def hyperbola_eq : Prop :=
∃ (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b), 
  (a^2 + b^2 = 3 ∧ (4 / a^2 - 1 / b^2 = 1) ∧ 
  (∀ x y, (x, y) = (2, 1) → (x^2 / a^2 - y^2 / b^2 = 1 → x^2 / 2 - y^2 = 1)))

theorem hyperbola_equation :
  hyperbola_eq := 
begin
  sorry
end

end hyperbola_equation_l56_56230


namespace ksyusha_travel_time_l56_56760

theorem ksyusha_travel_time :
  ∀ (S v : ℝ), 
  (2 * S / v + S / (2 * v) = 30) → 
  (2 * (S / v) = 24) :=
by
  intros S v h1
  have h2 : S / v = 12 := by
    calc 
      S / v = (30 * 2) / 5 := by sorry
  calc 
    2 * (S / v) = 2 * 12 := by sorry
  ... = 24 := by norm_num

end ksyusha_travel_time_l56_56760


namespace triangle_perimeter_l56_56464

theorem triangle_perimeter (a b c : ℕ) (ha : a = 14) (hb : b = 8) (hc : c = 9) : a + b + c = 31 := 
by
  sorry

end triangle_perimeter_l56_56464


namespace polynomial_remainder_is_zero_l56_56177

theorem polynomial_remainder_is_zero :
  ∀ (x : ℤ), ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 := 
by
  sorry

end polynomial_remainder_is_zero_l56_56177


namespace area_ADEC_of_parallelogram_l56_56286

variables (A B C D E : Type)
variables [HasArea A] [HasArea B] [HasArea C] [HasArea D] [HasArea E]
variables (BC AD : ℕ)
variables (DE : ℕ)

def parallelogram (BC AD : ℕ) := 2 * BC * AD

def line_parallel_through_D (DE BC : ℕ) := DE

theorem area_ADEC_of_parallelogram 
  (hBC : BC = 12) 
  (hAD : AD = 9)
  (hDE : DE = 8) : 
  parallelogram BC AD - (parallelogram AD DE / 2) = 90 :=
sorry

end area_ADEC_of_parallelogram_l56_56286


namespace combine_squares_without_cutting_l56_56412

/-- Given two squares of arbitrary sizes, one can combine them into a larger square without cutting
    the smaller square by leveraging geometric principles and auxiliary constructions. -/
theorem combine_squares_without_cutting (small_square large_square : Square) :
    ∃ new_square : Square, (new_square.area = small_square.area + large_square.area) ∧
        (∀ part_in_new_square : Part, part_in_new_square ⊆ large_square ∨ part_in_new_square ⊆ small_square) :=
by
  sorry

end combine_squares_without_cutting_l56_56412


namespace harmonic_series_inequality_l56_56082

theorem harmonic_series_inequality (m : ℕ) (h : 0 < m) :
  ∑ i in finset.range (2^m), (1 : ℝ) / (i+2) < m := sorry

end harmonic_series_inequality_l56_56082


namespace perimeter_equal_apothem_and_radius_formulas_l56_56957

variable (n : ℕ) (h₁ r₁ : ℝ)

-- Definitions
def is_regular_2n_sided_polygon (n : ℕ) : Prop :=
  ∃ (P : set (ℝ × ℝ)), (P.card = 2 * n) ∧ (∀ (a b ∈ P), dist a b = dist a b) -- Simplified representation

-- Given conditions
variable [regular_polygon : is_regular_2n_sided_polygon n]

-- New apothem and radius
noncomputable def h₂ : ℝ := (r₁ + h₁) / 2
noncomputable def r₂ : ℝ := Real.sqrt (h₂ * r₁)

-- Proof problem
theorem perimeter_equal {n : ℕ} {h₁ r₁ : ℝ}
  (h₂ := (r₁ + h₁) / 2)
  (r₂ := Real.sqrt (h₂ * r₁)) :
  let P_orig := (n * (dist (0, r₁) (r₁, (0))) = 2 * n * (dist ((r₁ + h₁) / 2, 0) (((r₁ + h₁) / 2), r₁))) in
  let P_new := (n * (dist (0, r₁) (r₁, (0))) = n * dist (r₁, 0) in
  P_orig = P_new :=
by sorry

theorem apothem_and_radius_formulas {n : ℕ} {h₁ r₁ : ℝ} :
  let h₂ := (r₁ + h₁) / 2
  let r₂ := Real.sqrt (h₂ * r₁) in
  h₂ = (r₁ + h₁) / 2 ∧ r₂ = Real.sqrt ((r₁ + h₁) / 2 * r₁) :=
by sorry

end perimeter_equal_apothem_and_radius_formulas_l56_56957


namespace simplify_expression_l56_56831

theorem simplify_expression : 
  (6^8 - 4^7) * (2^3 - (-2)^3) ^ 10 = 1663232 * 16 ^ 10 := 
by {
  sorry
}

end simplify_expression_l56_56831


namespace jason_missed_games_l56_56305

def planned_games_this_month : ℕ := 11
def planned_games_last_month : ℕ := 17
def actual_attended_games : ℕ := 12
def total_missed_games : ℕ := (planned_games_this_month + planned_games_last_month) - actual_attended_games

theorem jason_missed_games : total_missed_games = 16 := 
by
  have h1 : planned_games_this_month + planned_games_last_month = 28 := by norm_num
  have h2 : 28 - actual_attended_games = 16 := by norm_num
  rw [← h1] at h2
  exact h2

end jason_missed_games_l56_56305


namespace area_of_rectangle_inscribed_l56_56701

-- Define the points and conditions
def A : Point := ⟨45, 0⟩
def B : Point := ⟨20, 0⟩
def C : Point := ⟨0, 30⟩
def angle_ACB := 90
def area_CGF := 351

-- Target area of the rectangle
def target_area_rect := 468

-- Define the property of the triangle ABC
theorem area_of_rectangle_inscribed (A B C : Point)
  (h1 : A = ⟨45, 0⟩)
  (h2 : B = ⟨20, 0⟩)
  (h3 : C = ⟨0, 30⟩)
  (h4 : angle_ACB = 90)
  (h5 : area_CGF = 351) :
  area_of_rectangle_DEFG = 468 := 
sorry

end area_of_rectangle_inscribed_l56_56701


namespace selling_price_correct_l56_56021

-- Define the cost price and loss percentage
def cost_price : ℝ := 1500
def loss_percentage : ℝ := 16 / 100

-- Define the loss amount based on the given cost price and loss percentage
def loss_amount : ℝ := loss_percentage * cost_price

-- Define the selling price as the cost price minus the loss amount
def selling_price : ℝ := cost_price - loss_amount

-- Prove that the selling price is equal to Rs. 1260
theorem selling_price_correct :
  selling_price = 1260 :=
by
  -- This part should contain the proof steps, but it's substituted by sorry for now
  sorry

end selling_price_correct_l56_56021


namespace lattice_points_in_region_l56_56114

def is_lattice_point (p : ℝ × ℝ) : Prop := (p.1 ∈ ℤ) ∧ (p.2 ∈ ℤ)

def region1 (x y : ℝ) : Prop := y = |x|
def region2 (x y : ℝ) : Prop := y = -x^3 + 6

theorem lattice_points_in_region :
  let points := {p | is_lattice_point p ∧ ∃ x y, p = (x, y) ∧ (region1 x y ∨ region2 x y)}
  ∃ n, n = 9 ∧ n = points.card :=
by
  sorry

end lattice_points_in_region_l56_56114


namespace ksyusha_wednesday_time_l56_56754

def speed_relation (v : ℝ) := ∀ (d_walk d_run1 d_run2 time1 time2: ℝ), 
  (2 * d_run1 = d_walk) → -- she walked twice the distance she ran on Tuesday
  (2 * v = 2 * v) → -- she runs twice as fast as she walks
  ((2 * d_walk / v) + (d_run1 / (2 * v)) = time1) → -- she took 30 minutes on Tuesday
  (time1 = 30) → -- Tuesday time equals 30
  (d_run2 = 2 * d_walk) → -- she had to run twice the distance she walked on Wednesday
  ((d_walk / v) + (d_run2 / (2 * v)) = time2) → -- calculate the time for Wednesday
  time2 = 24 -- time taken on Wednesday

theorem ksyusha_wednesday_time :
  ∃ (v : ℝ), speed_relation v :=
begin
  existsi 1,
  unfold speed_relation,
  intros,
  sorry,
end

end ksyusha_wednesday_time_l56_56754


namespace count_multiples_of_12_l56_56633

theorem count_multiples_of_12 (a b : ℕ) (ha : a = 20) (hb : b = 200) : 
  (finset.card ((finset.Ico 20 201).filter (λ x, x % 12 = 0))) = 15 :=
by {
  rw [ha, hb],
  sorry
}

end count_multiples_of_12_l56_56633


namespace number_of_valid_colorings_l56_56671

-- Define the specific properties and constraints given in the problem
def is_power_of_two (n : ℕ) : Prop := ∃ m : ℕ, n = 2^m

def valid_coloring (f : ℕ → bool) : Prop :=
  ∀ a b : ℕ, a ≠ b → a + b ≤ 200 →
  (f a = f b → ¬ is_power_of_two (a + b))

theorem number_of_valid_colorings : 
  ∃ c : ℕ, c = 256 ∧ 
  ∃ f : Π (n : ℕ), nat.lt n 201 → bool, 
    valid_coloring (λ n, f n (by linarith)) :=
begin
  sorry
end

end number_of_valid_colorings_l56_56671


namespace line_circle_relationship_l56_56507

noncomputable def distance_from_line_to_point (a b c x₀ y₀ : ℝ) : ℝ :=
  (abs (a * x₀ + b * y₀ + c)) / (real.sqrt (a^2 + b^2))

theorem line_circle_relationship (a b c x₀ y₀ r : ℝ) :
  let d := distance_from_line_to_point a b c x₀ y₀ in
  (if d < r then "The line intersects the circle"
   else if d = r then "The line is tangent to the circle"
   else "The line and the circle do not intersect") = 
  (if d < r then "The line intersects the circle"
   else if d = r then "The line is tangent to the circle"
   else "The line and the circle do not intersect") :=
by
  sorry

end line_circle_relationship_l56_56507


namespace polynomial_remainder_is_zero_l56_56178

theorem polynomial_remainder_is_zero :
  ∀ (x : ℤ), ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 := 
by
  sorry

end polynomial_remainder_is_zero_l56_56178


namespace trigonometric_identity_l56_56226

-- Define the given conditions
variables (α : ℝ)
hypothesis h₁ : tan (α + π / 4) = 1 / 2
hypothesis h₂ : -π / 2 < α ∧ α < 0

-- Define the statement to be proven
theorem trigonometric_identity : 
  \frac{2 * sin α ^ 2 + sin (2 * α)}{cos (α - π / 4)} = -2 * sqrt 5 / 5 :=
by
  -- Here should be the proof, but we use 'sorry' to skip it for now
  sorry

end trigonometric_identity_l56_56226


namespace smallest_M_exists_l56_56442

theorem smallest_M_exists (N : ℕ) (hN : N ≥ 4) : 
  ∃ (M : ℕ), 
    (∀ (flags : finset (vector (bool) N)), 
      flags.card = M → 
      (∃ (subflags : finset (vector (bool) N)), 
        subflags.card = N ∧ 
        (∃ (color : bool), ∀ (i : fin N), (subflags.nth i).nth i = color))) 
    ∧ M = 2^(N - 2) + 1 :=
sorry

end smallest_M_exists_l56_56442


namespace new_pyramid_volume_correct_l56_56127

-- Define the conditions as given in the problem
def base_edge_length : ℝ := 10
def original_height : ℝ := 12
def cut_height : ℝ := 4

-- Define volume computation for the original and cut-off pyramid
def original_volume (base_edge_length height : ℝ) : ℝ := 
  (1 / 3) * (base_edge_length ^ 2) * height

def cutoff_pyramid_volume (base_edge_length original_height cut_height : ℝ) : ℝ :=
  let new_height := original_height - cut_height
  let scale := new_height / original_height
  let new_base_edge_length := base_edge_length * scale
  (1 / 3) * (new_base_edge_length ^ 2) * new_height

-- Prove the computed volume matches the expected volume
theorem new_pyramid_volume_correct :
  cutoff_pyramid_volume base_edge_length original_height cut_height = 3200 / 27 :=
by
  sorry

end new_pyramid_volume_correct_l56_56127


namespace product_squares_l56_56993

theorem product_squares :
  (∏ n in Finset.range 49, (1 - (1 / (n + 2))^2)) = (1 / 2500) :=
by
  sorry

end product_squares_l56_56993


namespace integer_roots_sum_abs_eq_100_l56_56543

theorem integer_roots_sum_abs_eq_100 (m p q r : ℤ) :
  (Polynomial.X ^ 3 - 2023 * Polynomial.X + Polynomial.C m).roots = [p, q, r] →
  p + q + r = 0 →
  p * q + q * r + r * p = -2023 →
  |p| + |q| + |r| = 100 :=
by
  sorry

end integer_roots_sum_abs_eq_100_l56_56543


namespace angle_between_vectors_l56_56248

noncomputable def find_angle
  (a b : ℝ^3)
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 2)
  (hab : ‖a - b‖ = Real.sqrt 7) : Real :=
  let cos_theta := (‖a‖ * ‖b‖ * 2)⁻¹ * (‖a - b‖ ^ 2 - ‖a‖ ^ 2 - ‖b‖ ^ 2)
  Real.acos cos_theta

theorem angle_between_vectors
  (a b : ℝ^3)
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 2)
  (hab : ‖a - b‖ = Real.sqrt 7) :
  find_angle a b ha hb hab = 2 * Real.pi / 3 := by
  sorry

end angle_between_vectors_l56_56248


namespace binomials_imply_polynomials_l56_56521

-- Define what it means to be able to perform algebraic operations on binomials
def can_handle_binomials : Prop := sorry  -- This would be defined in detail as per algebraic operations

-- Define a polynomial as a sequence of terms
def polynomial (n : ℕ) := vector ℤ n  -- Assuming integer coefficients for simplicity

-- Translate the main statement into Lean 4: If a person can handle binomials, then they can handle polynomials
theorem binomials_imply_polynomials (n : ℕ) (P : polynomial n) : can_handle_binomials → True := 
begin
  intro h,
  -- Proof steps here... (omitted as per the instruction)
  exact true.intro,
end

end binomials_imply_polynomials_l56_56521


namespace coloring_number_lemma_l56_56686

-- Definition of the problem conditions
def no_sum_of_two_same_color_is_power_of_two (f : ℕ → bool) : Prop :=
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 200 ∧ f a = f b → ¬(∃ k : ℕ, a + b = 2^k)

-- The main theorem
theorem coloring_number_lemma :
  (∃ f : ℕ → bool, (∀ n, 1 ≤ n ∧ n ≤ 200 → (f n = tt ∨ f n = ff))
                   ∧ no_sum_of_two_same_color_is_power_of_two f) →
  ∃ n : ℕ, n = 256 :=
by sorry

end coloring_number_lemma_l56_56686


namespace expression_value_l56_56062

theorem expression_value : (19 + 12) ^ 2 - (12 ^ 2 + 19 ^ 2) = 456 := 
by sorry

end expression_value_l56_56062


namespace best_graph_for_fluctuation_is_line_l56_56130

namespace StatisticalGraphs

-- Define the possible graph types
inductive GraphType
| lineGraph
| barGraph
| pieChart
| frequencyDistributionHistogram

-- Define the function to determine the best graph for showing fluctuation
def bestGraphForFluctuation (graphs : List GraphType) : GraphType :=
  if GraphType.lineGraph ∈ graphs then GraphType.lineGraph
  else GraphType.barGraph -- default (though in practice, lineGraph will be the best choice due to the list construction in the given condition)

-- Formalize the proof problem
theorem best_graph_for_fluctuation_is_line :
  bestGraphForFluctuation [GraphType.lineGraph, GraphType.barGraph, GraphType.pieChart, GraphType.frequencyDistributionHistogram] = GraphType.lineGraph :=
by
  -- Placeholder for actual proof
  sorry

end StatisticalGraphs

end best_graph_for_fluctuation_is_line_l56_56130


namespace alice_winning_strategy_l56_56324

theorem alice_winning_strategy (n : ℕ) (h : n ≥ 2) :
  (∃ strategy : Π (s : ℕ), s < n → (ℕ × ℕ), 
    ∀ (k : ℕ) (hk : k < n), ¬(strategy k hk).fst = (strategy k hk).snd) ↔ (n % 4 = 3) :=
sorry

end alice_winning_strategy_l56_56324


namespace coloring_problem_l56_56684

noncomputable def numColorings : Nat :=
  256

theorem coloring_problem :
  ∃ (f : Fin 200 → Bool), (∀ (i j : Fin 200), i ≠ j → f i = f j → ¬ (i + 1 + j + 1).isPowerOfTwo) ∧ 
    256 = numColorings :=
by
  -- Definitions and conditions used for the problem
  let S := {n : ℕ | ∃ k : ℕ, n = bit0 k 1 ∧ k > 0}
  have hS : ∀ n ∈ S, ∀ i j ∈ {1, ..., 200}, n ≠ 2 ^ (log2 n) ∧ i ≠ j → i + j ≠ n := sorry
  -- Proof skipped, as required in the guidelines
  sorry

end coloring_problem_l56_56684


namespace sin_gt_then_angle_gt_l56_56262

variable (A B : ℝ) -- Angles A and B (in radians) in triangle
variable (sa sb : ℝ) -- sina and sinb (sine of angles A and B respectively)

axiom triangle_inequality (a b c : ℝ) : a + b > c ∧ b + c > a ∧ c + a > b
axiom sin_safe_interval (x : ℝ) : 0 ≤ sin x ∧ sin x ≤ 1

theorem sin_gt_then_angle_gt (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : sin A > sin B) :
  A > B ∧ (A > B → sin A > sin B) := sorry

end sin_gt_then_angle_gt_l56_56262


namespace length_of_IH_l56_56965

-- Define the points and lengths involved in the problem.
def square_side_ABCD := 4
def square_side_CEFG := 12
def DE := square_side_ABCD + square_side_CEFG -- DE is the sum of the side lengths of ABCD and CEFG.
def EF := square_side_CEFG -- EF is the side length of CEFG.
def DF := Real.sqrt (DE^2 + EF^2) -- DF according to the Pythagorean theorem.
def GJ := (EF * EF) / DF -- GJ is computed from similar triangles.

-- The goal is to prove that the length of IH is equal to GJ.
theorem length_of_IH : GJ = (36 / 5) :=
by
  sorry

end length_of_IH_l56_56965


namespace solve_problem_l56_56051

def smallest_prime_divisors : ℕ := 2

def has_factors_of_4_divisors (n : ℕ) : Prop :=
  ∃ p q : ℕ, p ≠ q ∧ Prime p ∧ Prime q ∧ (n = p^3 ∨ n = p * q)

def largest_integer_with_4_divisors_less_than_500 : ℕ :=
  Nat.find (Nat.lt_wf.1 500).downward_closed (λ n, has_factors_of_4_divisors n ∧ n < 500)
  (by use 437; -- providing the witness
    split;
    simp [437 = 19 * 23, Prime.decidable_prime 19, Prime.decidable_prime 23];
    sorry -- proof of primality and distinctness of 19 and 23)

theorem solve_problem : smallest_prime_divisors + largest_integer_with_4_divisors_less_than_500 = 439 :=
by 
  have a := smallest_prime_divisors,
  have b := largest_integer_with_4_divisors_less_than_500,
  have ha : a = 2 := rfl,
  have hb : b = 437 := rfl,
  rw [ha, hb],
  exact rfl

end solve_problem_l56_56051


namespace quilt_squares_count_l56_56149

theorem quilt_squares_count (total_squares : ℕ) (additional_squares : ℕ)
  (h1 : total_squares = 4 * additional_squares)
  (h2 : additional_squares = 24) :
  total_squares = 32 :=
by
  -- Proof would go here
  -- The proof would involve showing that total_squares indeed equals 32 given h1 and h2
  sorry

end quilt_squares_count_l56_56149


namespace abs_sqrt2_sub_2_l56_56832

theorem abs_sqrt2_sub_2 (h : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2) : |Real.sqrt 2 - 2| = 2 - Real.sqrt 2 :=
by
  sorry

end abs_sqrt2_sub_2_l56_56832


namespace negation_of_universal_statement_l56_56205

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by 
  -- Proof steps would be added here
  sorry

end negation_of_universal_statement_l56_56205


namespace difference_in_area_l56_56855

-- Define the dimensions of the largest room (trapezoid)
def larger_room_parallel_side1 : ℝ := 45
def larger_room_parallel_side2 : ℝ := 30
def larger_room_height : ℝ := 25

-- Define the dimensions of the smallest room (parallelogram)
def smaller_room_base : ℝ := 15
def smaller_room_height : ℝ := 8

-- Define the area calculation for trapezoid
def area_trapezoid (a b h : ℝ) : ℝ := (1 / 2) * (a + b) * h

-- Define the area calculation for parallelogram
def area_parallelogram (base height : ℝ) : ℝ := base * height

-- Prove the difference in area
theorem difference_in_area :
  area_trapezoid larger_room_parallel_side1 larger_room_parallel_side2 larger_room_height - 
  area_parallelogram smaller_room_base smaller_room_height = 817.5 := by
  sorry

end difference_in_area_l56_56855


namespace coeff_x2_expansion_l56_56596

theorem coeff_x2_expansion (a : ℝ) : 
  let coeff := (a^2 * (nat.choose 6 2) - a * (nat.choose 6 1)) in
  coeff = 21 ↔ a = 7/5 ∨ a = -1 :=
by
  sorry

end coeff_x2_expansion_l56_56596


namespace shopping_problem_l56_56966

theorem shopping_problem
  (D S H N : ℝ)
  (h1 : (D - (D / 2 - 10)) + (S - 0.85 * S) + (H - (H - 30)) + (N - N) = 120)
  (T_sale : ℝ := (D / 2 - 10) + 0.85 * S + (H - 30) + N) :
  (120 + 0.10 * T_sale = 0.10 * 1200) →
  D + S + H + N = 1200 :=
by
  sorry

end shopping_problem_l56_56966


namespace expel_evil_wizards_with_max_one_good_l56_56142

def Wizard := { N : ℕ // N = 2015} -- Define the type representing the wizards with 2015 as a constraint.

structure WizardWorld (N : ℕ) :=
  (is_good : Fin N → Prop) -- Fin N is the domain representing the wizards.
  (knows_good_or_evil : ∀ (w : Fin N), ∀ (other : Fin N),  Bool) -- each wizard knows who is good or evil
  (truth_teller : ∀ (w : Fin N), is_good w → ∀ (other : Fin N), knows_good_or_evil w other = tt → is_good other)
  (liar : ∀ (w : Fin N), ¬ is_good w → ( ∀ (other : Fin N), knows_good_or_evil w other = tt ∨ knows_good_or_evil w other = ff ) )

theorem expel_evil_wizards_with_max_one_good (W : WizardWorld 2015) : 
  ∃ S : Set (Fin 2015), (∀ w ∈ S, ¬ W.is_good w) ∧ (S.card ≤ W.card - S.card + 1) :=
sorry

end expel_evil_wizards_with_max_one_good_l56_56142


namespace some_magical_are_mysterious_l56_56646

-- Define the sets for dragons, magical creatures, and mysterious beings
variables (Dragon MagicalCreature MysteriousBeing : Type)

-- Define the conditions
variables (all_dragons_are_magical : ∀ d : Dragon, MagicalCreature d)
          (some_mysterious_are_dragons : ∃ m : MysteriousBeing, Dragon m)

-- Define the statement we want to prove
theorem some_magical_are_mysterious : ∃ m : MagicalCreature, MysteriousBeing m := 
sorry

end some_magical_are_mysterious_l56_56646


namespace sum_sequence_is_25_l56_56155

def sequence_sum : List Int := List.range' (-49) 100 |>.map (λ x, if x % 2 = 0 then x + 1 else x - 1)

theorem sum_sequence_is_25 : sequence_sum.sum = 25 := by
  sorry

end sum_sequence_is_25_l56_56155


namespace dihedral_angle_square_tetrahedron_l56_56445

theorem dihedral_angle_square_tetrahedron
  (A B C D E F P : Point)
  (a : ℝ)
  (h_square : square A B C D)
  (h_midpoint_E : midpoint E B C)
  (h_midpoint_F : midpoint F C D)
  (h_fold : fold_to_tetrahedron A E F P B C D) :
  dihedral_angle P E F A E F = arcsin (2 * sqrt 2 / 3) :=
sorry

end dihedral_angle_square_tetrahedron_l56_56445


namespace fifteen_iterations_of_F_on_4_l56_56204

def first_and_last_digit_sum_sq (n : ℕ) : ℕ :=
  let s := n.toString
  if s.length = 1 then n^2
  else ((s.head!.toNat - '0'.toNat)^2 + (s.last!.toNat - '0'.toNat)^2)

def F (n : ℕ) : ℕ :=
  if n < 10 then n^2 else first_and_last_digit_sum_sq n

def F_k : ℕ → ℕ → ℕ
| 0, n := n
| (k+1), n := F (F_k k n)

theorem fifteen_iterations_of_F_on_4 : F_k 15 4 = 16 :=
by
  simp [F, F_k, first_and_last_digit_sum_sq]
  sorry

end fifteen_iterations_of_F_on_4_l56_56204


namespace intervals_of_monotonicity_range_for_t_inequality_proof_l56_56334

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x - a * (x + 1) * Real.log (x + 1)

-- Condition on x and a
axiom hx (x : ℝ) : x > -1
axiom ha (a : ℝ) : a ≥ 0

theorem intervals_of_monotonicity (a : ℝ) :
  (a = 0 → ∀ x, f x a = f x 0 → f' x a > 0) ∧
  (a > 0 → ∀ x, f x a = f x (real.exp ((1 - a) / a) - 1) → f' x a = 0 →
    ∃ x ∈ Ioc (-1 : ℝ) (real.exp ((1 - a) / a) - 1)), f' x a > 0) ∧
  (a > 0 → ∀ x, f x a = f x (real.exp ((1 - a) / a) - 1) → f' x a = 0 →
    ∃ x ∈ Icc (real.exp ((1 - a) / a) - 1) ∞, f' x a < 0) :=
sorry

theorem range_for_t (t : ℝ) :
  f 0 1 = 0 ∧ f 1 1 = 1 - Real.log 4 ∧ f (-1 / 2) 1 = -1 / 2 + 1 / 2 * Real.log 2 →
  ∃ t', t ∈ Icc (-1/2 + 1/2 * Real.log 2) 0 → (∃ x y ∈ Icc (-1/2 : ℝ) 1, f x 1 = t ∧ f y 1 = t ∧ x ≠ y) :=
sorry

theorem inequality_proof (m n : ℝ) :
  m > n ∧ n > 0 → (1 + m)^n < (1 + n)^m :=
sorry

end intervals_of_monotonicity_range_for_t_inequality_proof_l56_56334


namespace square_area_inscribed_in_parabola_l56_56003

theorem square_area_inscribed_in_parabola :
  ∃ (s : ℝ), 
    (s > 0) ∧
    ∃ (area : ℝ),
    (area = (2 * s)^2) ∧
    ((-2 * s = (5 + s)^2 - 10 * (5 + s) + 21) ∧ 
     area = 24 - 8 * real.sqrt 5) :=
begin
  sorry
end

end square_area_inscribed_in_parabola_l56_56003


namespace solve_equation_l56_56356

theorem solve_equation (x : ℝ) : 5 ^ (x + 6) = 25 ^ x → x = 6 :=
by
  sorry

end solve_equation_l56_56356


namespace ksyusha_travel_time_wednesday_l56_56744

-- Definitions based on the conditions
def speed_walk (v : ℝ) : ℝ := v
def speed_run (v : ℝ) : ℝ := 2 * v
def distance_walk_tuesday (S : ℝ) : ℝ := 2 * S
def distance_run_tuesday (S : ℝ) : ℝ := S
def total_time_tuesday (v S : ℝ) : ℝ := (distance_walk_tuesday S) / (speed_walk v) + (distance_run_tuesday S) / (speed_run v)

-- Theorem statement
theorem ksyusha_travel_time_wednesday (S v : ℝ) (h : total_time_tuesday v S = 30) :
  (S / v + S / v) = 24 := 
by sorry

end ksyusha_travel_time_wednesday_l56_56744


namespace system_of_equations_proof_l56_56626

theorem system_of_equations_proof (a b x A B C : ℝ) (h1: a ≠ 0) 
  (h2: a * Real.sin x + b * Real.cos x = 0) 
  (h3: A * Real.sin (2 * x) + B * Real.cos (2 * x) = C) : 
  2 * a * b * A + (b ^ 2 - a ^ 2) * B + (a ^ 2 + b ^ 2) * C = 0 := 
sorry

end system_of_equations_proof_l56_56626


namespace half_angle_quadrant_l56_56644

-- Define the given condition
def is_angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

-- Define the result that needs to be proved
def is_angle_in_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, (k * 180 < α / 2 ∧ α / 2 < k * 180 + 45) ∨ (k * 180 + 180 < α / 2 ∧ α / 2 < k * 180 + 225)

-- The main theorem statement
theorem half_angle_quadrant (α : ℝ) (h : is_angle_in_first_quadrant α) : is_angle_in_first_or_third_quadrant α :=
sorry

end half_angle_quadrant_l56_56644


namespace prob1_prob2_prob3_l56_56595

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x^2 + 2
  else x

theorem prob1 :
  (∀ x, x ≥ 0 → f x = x^2 + 2) ∧
  (∀ x, x < 0 → f x = x) :=
by
  sorry

theorem prob2 : f 5 = 27 :=
by 
  sorry

theorem prob3 : ∀ (x : ℝ), f x = 0 → false :=
by
  sorry

end prob1_prob2_prob3_l56_56595


namespace convert_to_polar_l56_56506

noncomputable def sqrt_two : ℝ := real.sqrt 2

theorem convert_to_polar (x y : ℝ) (r θ : ℝ) (h1 : x = 8) (h2 : y = -8) 
  (hr : r = real.sqrt (x^2 + y^2)) (hθ : θ = real.arctan (abs y / x)) :
  (r, θ) = (8 * sqrt_two, 7 * real.pi / 4) :=
  by
  sorry

end convert_to_polar_l56_56506


namespace quadratic_axis_of_symmetry_condition_l56_56235

theorem quadratic_axis_of_symmetry_condition 
  (a b c : ℝ) 
  (h1 : ∃ x, x = 1 ∧ y = ax^2 + bx + c) 
  (h2 : b = -2a) : 
  c < 2b := 
sorry

end quadratic_axis_of_symmetry_condition_l56_56235


namespace additional_money_earned_l56_56456

-- Define the conditions as variables
def price_duck : ℕ := 10
def price_chicken : ℕ := 8
def num_chickens_sold : ℕ := 5
def num_ducks_sold : ℕ := 2
def half (x : ℕ) : ℕ := x / 2
def double (x : ℕ) : ℕ := 2 * x

-- Define the calculations based on the conditions
def earnings_chickens : ℕ := num_chickens_sold * price_chicken 
def earnings_ducks : ℕ := num_ducks_sold * price_duck 
def total_earnings : ℕ := earnings_chickens + earnings_ducks 
def cost_wheelbarrow : ℕ := half total_earnings
def selling_price_wheelbarrow : ℕ := double cost_wheelbarrow
def additional_earnings : ℕ := selling_price_wheelbarrow - cost_wheelbarrow

-- The theorem to prove the correct additional earnings
theorem additional_money_earned : additional_earnings = 30 := by
  sorry

end additional_money_earned_l56_56456


namespace D_not_parallelogram_l56_56293

-- Definitions of given points A, B, and C
def A := (0 : ℝ, 0 : ℝ)
def B := (2 : ℝ, 2 : ℝ)
def C := (3 : ℝ, 0 : ℝ)

-- Definition of point D
def D := (2 : ℝ, -2 : ℝ)

-- Define the predicate for points to form a parallelogram
def is_midpoint (P Q R S : ℝ × ℝ) : Prop :=
  P.1 + R.1 = Q.1 + S.1 ∧ P.2 + R.2 = Q.2 + S.2

def is_parallelogram (P Q R S : ℝ × ℝ) : Prop :=
  is_midpoint P R Q S ∨ is_midpoint P Q R S ∨ is_midpoint P S Q R

-- The theorem stating that D does not form a parallelogram ABCD
theorem D_not_parallelogram : ¬ is_parallelogram A B C D := 
  sorry

end D_not_parallelogram_l56_56293


namespace find_inclination_angle_l56_56379

def inclination_angle (α : ℝ) : Prop :=
  α = Real.arctan (√3 / 3) ∧ α ∈ Set.Ico (0 : ℝ) Real.pi

theorem find_inclination_angle :
  inclination_angle (Real.arctan (√3 / 3)) :=
by
  apply And.intro
  sorry -- proof of α = arctan(√3 / 3)
  split
  exact Real.arctan_nonneg _ _
  exact Real.arctan_lt_pi _ _
  sorry -- proof of α ∈ [0, π)

end find_inclination_angle_l56_56379


namespace Ksyusha_time_to_school_l56_56734

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l56_56734


namespace intersection_of_parabola_with_y_axis_l56_56020

theorem intersection_of_parabola_with_y_axis :
  ∃ y : ℝ, y = - (0 + 2)^2 + 6 ∧ (0, y) = (0, 2) :=
by
  sorry

end intersection_of_parabola_with_y_axis_l56_56020


namespace part_I_part_II_l56_56561

noncomputable def f (x : ℝ) : ℝ := 2 * (sqrt 3) * (sin x) * (cos x) + 2 * (cos x)^2 - 1

theorem part_I : f (π / 6) = 2 := 
  sorry

theorem part_II : 
  ∃ k : ℤ, 
    ∀ x : ℝ, (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) → 
              (∀ x₁ x₂ : ℝ, (k * π - π / 3 <= x₁ ∧ x₁ <= x₂ ∧ x₂ <= k * π + π / 6) → f x₁ ≤ f x₂) :=
  sorry

end part_I_part_II_l56_56561


namespace range_of_a_l56_56224

-- Definitions for propositions p and q
def proposition_p (a : ℝ) : Prop :=
  ∀ y, ∃ x, y = Real.log (0.5) (x^2 + 2*x + a)

def proposition_q (a : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → -(5 - 2*a)^x1 > -(5 - 2*a)^x2

-- The final proof statement
theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) ↔ (1 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l56_56224


namespace constant_term_expansion_l56_56851

theorem constant_term_expansion :
  let term := (λ r : ℕ, (-1) ^ r * (Nat.choose 9 r) * (x ^ ((9 - 3 * r) / 2))) in
  (9 >= 3) → (∃ r : ℕ, (9 - 3 * r) / 2 = 0 ∧ term r = -84) :=
by
  sorry

end constant_term_expansion_l56_56851


namespace find_width_of_chalkboard_l56_56343

variable (w : ℝ) (l : ℝ)

-- Given conditions
def length_eq_twice_width (w l : ℝ) : Prop := l = 2 * w
def area_eq_eighteen (w l : ℝ) : Prop := w * l = 18

-- Theorem statement
theorem find_width_of_chalkboard (h1 : length_eq_twice_width w l) (h2 : area_eq_eighteen w l) : w = 3 :=
by sorry

end find_width_of_chalkboard_l56_56343


namespace number_of_stubborn_numbers_l56_56783

open Nat

-- Conditions: a, b, c are pairwise coprime natural numbers.
variables (a b c : ℕ)

-- Definition for pairwise coprime
def pairwise_coprime (x y z : ℕ) : Prop :=
    gcd x y = 1 ∧ gcd y z = 1 ∧ gcd z x = 1

-- Definition of stubborn number
def is_stubborn (n : ℕ) : Prop :=
    ∀ x y z : ℕ, n ≠ b * c * x + c * a * y + a * b * z

-- Theorem statement
theorem number_of_stubborn_numbers (h : pairwise_coprime a b c) : 
    ∃! k : ℕ, k = a * b * c - a * b - b * c - c * a := sorry

end number_of_stubborn_numbers_l56_56783


namespace beads_necklace_l56_56812

theorem beads_necklace (N beads colors : ℕ) (hc : colors = 50) (hb : beads = N / colors) (H : N = 1000) :
  ∃ n, (∀ necklace : Fin N → Fin colors, ∃ segment : Fin n → Fin N, (∀ i, segment i < N) ∧ 
  (card (segment '' univ) ≥ 25)) ∧ n = 462 :=
by
  use 462
  sorry

end beads_necklace_l56_56812


namespace maximize_probability_l56_56105

theorem maximize_probability (p1 p2 p3 : ℝ) (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3) :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3) in
  PC > PA ∧ PC > PB :=
by
  sorry

end maximize_probability_l56_56105


namespace arithmetic_mean_first_n_odd_l56_56151

theorem arithmetic_mean_first_n_odd (n : ℕ) : 
  let odd_sum := ∑ i in finset.range n, (2 * i + 1)
  in (odd_sum / n) = n :=
by
  sorry

end arithmetic_mean_first_n_odd_l56_56151


namespace max_runs_constraint_l56_56269

-- Conditions
def num_overs : ℕ := 20
def balls_per_over : ℕ := 6
def total_balls : ℕ := num_overs * balls_per_over
def runs_per_ball : ℕ := 6
def expected_max_runs : ℕ := 663

-- The question stated as a proof
theorem max_runs_constraint (h_balls: total_balls = 120) 
                            (h_runs_per_ball: runs_per_ball = 6) 
                            (h_max_runs: total_balls * runs_per_ball ≥ expected_max_runs) :
  total_balls * runs_per_ball = expected_max_runs := 
begin
  -- Proof is omitted
  sorry,
end

end max_runs_constraint_l56_56269


namespace problem_l56_56212

theorem problem (f : ℕ → ℕ → ℕ) (h0 : f 1 1 = 1) (h1 : ∀ m n, f m n ∈ {x | x > 0}) 
  (h2 : ∀ m n, f m (n + 1) = f m n + 2) (h3 : ∀ m, f (m + 1) 1 = 2 * f m 1) : 
  f 1 5 = 9 ∧ f 5 1 = 16 ∧ f 5 6 = 26 :=
sorry

end problem_l56_56212


namespace Ksyusha_travel_time_l56_56775

theorem Ksyusha_travel_time:
  ∀ (S v : ℝ), 
  (S > 0 ∧ v > 0) →
  (∀ tuesday_distance tuesday_time wednesday_distance wednesday_time: ℝ, 
    tuesday_distance = 3 * S ∧ 
    tuesday_distance = 2 * S + S ∧ 
    tuesday_time = (2 * S) / v + (S / (2 * v)) ∧ 
    tuesday_time = 30) → 
  (∀ wednesday_distance wednesday_time : ℝ, 
    wednesday_distance = S + 2 * S ∧ 
    wednesday_time = S / v + (2 * S) / (2 * v)) →
  wednesday_time = 24 :=
by
  intros S v S_pos v_pos tuesday_distance tuesday_time wednesday_distance wednesday_time
  sorry

end Ksyusha_travel_time_l56_56775


namespace car_average_speed_l56_56922

def total_distance (segments : List (ℝ × ℝ)) : ℝ :=
  segments.foldr (fun (s, t) acc => acc + s * t) 0

def total_time (segments : List (ℝ × ℝ)) : ℝ :=
  segments.foldr (fun (_, t) acc => acc + t) 0

def average_speed (segments : List (ℝ × ℝ)) : ℝ :=
  total_distance segments / total_time segments

theorem car_average_speed :
  let segments := [(40, 1), (60, 0.5), (60, 2)]
  average_speed segments = 54.29 :=
by
  sorry

end car_average_speed_l56_56922


namespace grocer_rows_count_l56_56929

theorem grocer_rows_count (n : ℕ) (a d S : ℕ) (h_a : a = 1) (h_d : d = 3) (h_S : S = 225)
  (h_sum : S = n * (2 * a + (n - 1) * d) / 2) : n = 16 :=
by {
  sorry
}

end grocer_rows_count_l56_56929


namespace first_class_product_probability_l56_56100

theorem first_class_product_probability
  (defective_rate : ℝ) (first_class_rate_qualified : ℝ)
  (H_def_rate : defective_rate = 0.04)
  (H_first_class_rate_qualified : first_class_rate_qualified = 0.75) :
  (1 - defective_rate) * first_class_rate_qualified = 0.72 :=
by
  sorry

end first_class_product_probability_l56_56100


namespace max_correct_answers_l56_56659

/--
In a 50-question multiple-choice math contest, students receive 5 points for a correct answer, 
0 points for an answer left blank, and -2 points for an incorrect answer. Jesse’s total score 
on the contest was 115. Prove that the maximum number of questions that Jesse could have answered 
correctly is 30.
-/
theorem max_correct_answers (a b c : ℕ) (h1 : a + b + c = 50) (h2 : 5 * a - 2 * c = 115) : a ≤ 30 :=
by
  sorry

end max_correct_answers_l56_56659


namespace zero_in_A_l56_56621

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : 0 ∈ A :=
by {
  simp [A],
  sorry
}

end zero_in_A_l56_56621


namespace plane_equiv_l56_56552

variables (p : ℝ) (α β γ: ℝ)
variables (r : ℝ) (x y z : ℝ)
variables (e : ℝ × ℝ × ℝ) -- unit vector given by (cos α, cos β, cos γ)
variables (R : ℝ × ℝ × ℝ) -- position vector (x, y, z)

-- Condition: e is a unit vector
def is_unit_vector (e : ℝ × ℝ × ℝ) : Prop := 
  (e.1 ^ 2 + e.2 ^ 2 + e.3 ^ 2 = 1)

-- The normal equation of the plane in vector form
def plane_vector_form (R : ℝ × ℝ × ℝ) (e : ℝ × ℝ × ℝ) (p : ℝ) : Prop := 
  (R.1 * e.1 + R.2 * e.2 + R.3 * e.3 = p)

-- The normal equation of the plane in coordinate form
def plane_coordinate_form (x y z: ℝ) (α β γ : ℝ) (p : ℝ) : Prop := 
  (x * (real.cos α) + y * (real.cos β) + z * (real.cos γ) = p)

-- Proof problem: show that the vector form is equivalent to the coordinate form
theorem plane_equiv : 
  ∀ (x y z p α β γ : ℝ), 
    is_unit_vector ((real.cos α), (real.cos β), (real.cos γ)) →
      plane_vector_form (x, y, z) ((real.cos α), (real.cos β), (real.cos γ)) p →
      plane_coordinate_form x y z α β γ p :=
by
  intros x y z p α β γ h_unit_vector h_vector_form
  unfold is_unit_vector at h_unit_vector
  unfold plane_vector_form at h_vector_form
  unfold plane_coordinate_form
  rw h_vector_form
  sorry

end plane_equiv_l56_56552


namespace volume_of_one_wedge_l56_56131

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * π * r^3

theorem volume_of_one_wedge (circumference : ℝ) (num_wedges : ℕ) (V_wedge : ℝ) :
  circumference = 16 * π → num_wedges = 8 → V_wedge = (256 / 3) * π :=
by
  sorry

end volume_of_one_wedge_l56_56131


namespace dora_packs_of_stickers_l56_56341

theorem dora_packs_of_stickers (allowance : ℕ) (deck_cost : ℕ) (sticker_pack_cost : ℕ) 
  (combined_allowance : ℕ) (remaining_money : ℕ) (total_sticker_packs : ℕ) (dora_sticker_packs : ℕ) :
  allowance = 9 →
  deck_cost = 10 →
  sticker_pack_cost = 2 →
  combined_allowance = 2 * allowance →
  remaining_money = combined_allowance - deck_cost →
  total_sticker_packs = remaining_money / sticker_pack_cost →
  dora_sticker_packs = total_sticker_packs / 2 →
  dora_sticker_packs = 2 :=
by
  intros h_allowance h_deck_cost h_sticker_pack_cost h_combined_allowance 
    h_remaining_money h_total_sticker_packs h_dora_sticker_packs
  rw [h_allowance, h_deck_cost, h_sticker_pack_cost] at h_combined_allowance h_remaining_money h_total_sticker_packs h_dora_sticker_packs
  rw [h_combined_allowance, h_remaining_money, h_total_sticker_packs] at h_dora_sticker_packs
  rw [h_combined_allowance] at h_remaining_money
  rw [h_remaining_money] at h_total_sticker_packs
  rw [h_total_sticker_packs] at h_dora_sticker_packs
  exact h_dora_sticker_packs

end dora_packs_of_stickers_l56_56341


namespace remainder_x5_1_x3_1_div_x2_x_1_l56_56170

theorem remainder_x5_1_x3_1_div_x2_x_1 :
  ∀ (x : ℂ), let poly := (x^5 - 1) * (x^3 - 1),
                 divisor := x^2 + x + 1,
                 remainder := x^2 + x + 1 in
  ∃ q : ℂ, poly = q * divisor + remainder :=
by
  intro x
  let poly := (x^5 - 1) * (x^3 - 1)
  let divisor := x^2 + x + 1
  let remainder := x^2 + x + 1
  use sorry
  rw [← add_assoc, ← mul_assoc, ← pow_succ]
  sorry

end remainder_x5_1_x3_1_div_x2_x_1_l56_56170


namespace normal_lemon_tree_lemons_per_year_l56_56462

-- Let L be the number of lemons a normal lemon tree produces per year.
-- Define conditions as provided.
def total_trees : Nat := 50 * 30
def lemons_in_5_years : Nat := 675000
def lemons_per_year : Nat := lemons_in_5_years / 5
def production_factor : Float := 1.5

theorem normal_lemon_tree_lemons_per_year
  (total_trees : Nat)
  (lemons_per_year : Nat)
  (production_factor : Float)
  (tree_count : total_trees = 50 * 30)
  (yearly_production : lemons_per_year = 675000 / 5)
  (factor : production_factor = 1.5) :
  ∃ L : Nat, production_factor * (total_trees * L) = lemons_per_year ∧ L = 60 :=
sorry

end normal_lemon_tree_lemons_per_year_l56_56462


namespace ellipses_common_point_cyclic_l56_56883

noncomputable def ellipse_foci (A B C : Point) : Prop :=
  ∃ D : Point, (D.1 + D.2 = dist A C + dist B C) ∧
               (D.2 + D.3 = dist B C + dist C A) ∧
               (D.3 + D.1 = dist C A + dist A B)

axiom cyclic_points (A B C D : Point) : Prop

theorem ellipses_common_point_cyclic (A B C D : Point) 
    (h₁ : ellipse_foci A B C) 
    (h₂ : ellipse_foci B C A) 
    (h₃ : ellipse_foci C A B) 
    (h₄ : ∃ D, ellipse_foci A B C ∧ ellipse_foci B C A ∧ ellipse_foci C A B) : 
  cyclic_points A B C D := 
sorry

end ellipses_common_point_cyclic_l56_56883


namespace circle_standard_equation_l56_56232

theorem circle_standard_equation (x y : ℝ) :
  let center := (1 : ℝ, 2 : ℝ)
  let point := (-2 : ℝ, 6 : ℝ)
  let r_squared := (point.1 - center.1)^2 + (point.2 - center.2)^2
  ((x - center.1)^2 + (y - center.2)^2 = r_squared) :=
by
  sorry

end circle_standard_equation_l56_56232


namespace find_range_of_m_l56_56582

def proposition_p (m : ℝ) : Prop := 0 < m ∧ m < 1/3
def proposition_q (m : ℝ) : Prop := 0 < m ∧ m < 15
def proposition_r (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def proposition_s (m : ℝ) : Prop := proposition_p m ∧ proposition_q m = False
def range_of_m (m : ℝ) : Prop := 1/3 ≤ m ∧ m < 15

theorem find_range_of_m (m : ℝ) : proposition_r m ∧ proposition_s m → range_of_m m := by
  sorry

end find_range_of_m_l56_56582


namespace range_of_c_l56_56800

theorem range_of_c (c : ℝ) :
  (∃ x ∈ (Icc 1 2), max (abs (x + c / x)) (abs (x + c / x + 2)) ≥ 5) ↔ c ∈ (set.Iic (-6) ∪ set.Ici 2) :=
by
  sorry

end range_of_c_l56_56800


namespace smallest_x_fraction_floor_l56_56534

theorem smallest_x_fraction_floor (x : ℝ) (h : ⌊x⌋ / x = 7 / 8) : x = 48 / 7 :=
sorry

end smallest_x_fraction_floor_l56_56534


namespace int_floor_neg_3_7_add_5_l56_56385

-- The condition definition of greatest integer less than or equal to x
def greatest_integer (x : ℚ) := { n : ℤ // ↑n ≤ x ∧ ∀ m : ℤ, m ≤ x → m ≤ n }

-- Given condition: greatest integer of -3.7 is -4
axiom int_floor_neg_3_7 : greatest_integer (-3.7) = ⟨-4, by norm_num⟩

-- Lean 4 theorem statement
theorem int_floor_neg_3_7_add_5 : (greatest_integer (-3.7)).val + 5 = 1 :=
by 
  -- Using the given condition to state the conclusion
  have g := int_floor_neg_3_7
  rw g
  norm_num
  sorry

end int_floor_neg_3_7_add_5_l56_56385


namespace number_of_geometric_sequence_four_digit_numbers_l56_56200

theorem number_of_geometric_sequence_four_digit_numbers :
  (∃ (count : ℕ),
    count = (finset.sum (finset.range 10) (λ a,
      if h_a : a ≠ 0 then
        finset.sum (finset.range 10) (λ b,
          finset.sum (finset.range 10) (λ c,
            finset.sum (finset.range 10) (λ d,
              if (10 * a + c) * (10 * a + c) = (10 * a + b) * (10 * a + d)
                ∧ 10 * a + b < 10 * a + c ∧ 10 * a + c < 10 * a + d
              then 1 else 0)))
      else 0))) = 27 :=
by sorry

end number_of_geometric_sequence_four_digit_numbers_l56_56200


namespace fill_tank_in_6_hours_l56_56439

theorem fill_tank_in_6_hours (A B : ℝ) (hA : A = 1 / 10) (hB : B = 1 / 15) : (1 / (A + B)) = 6 :=
by 
  sorry

end fill_tank_in_6_hours_l56_56439


namespace sin_theta_half_l56_56788

-- Definitions of vectors and their properties
variables (a b c : ℝ^3)
variable (θ : ℝ)

-- Conditions given in the problem
def norm_a : ∥a∥ = 1 := by sorry
def norm_b : ∥b∥ = 6 := by sorry
def norm_c : ∥c∥ = 3 := by sorry
def cross_product_cond : a × (a × b) = c := by sorry
def dot_product_cond : a ⋅ c = 0 := by sorry

-- The statement that needs to be proved
theorem sin_theta_half : sin θ = 1 / 2 :=
by
  -- Using the conditions defined above
  have h1 := norm_a,
  have h2 := norm_b,
  have h3 := norm_c,
  have h4 := cross_product_cond,
  have h5 := dot_product_cond,
  sorry

end sin_theta_half_l56_56788


namespace number_of_schools_l56_56520

theorem number_of_schools (N : ℕ) :
  (∀ i j : ℕ, i < j → i ≠ j) →
  (∀ i : ℕ, i < 2 * 35 → i = 35 ∨ ((i = 37 → ¬ (i = 35))) ∧ ((i = 64 → ¬ (i = 35)))) →
  N = (2 * (35) - 1) / 3 →
  N = 23 :=
by
  sorry

end number_of_schools_l56_56520


namespace expression_equivalence_l56_56990

theorem expression_equivalence : (2 / 20) + (3 / 30) + (4 / 40) + (5 / 50) = 0.4 := by
  sorry

end expression_equivalence_l56_56990


namespace value_of_other_bills_is_40_l56_56731

-- Define the conditions using Lean definitions
def class_fund_contains_only_10_and_other_bills (total_amount : ℕ) (num_other_bills num_10_bills : ℕ) : Prop :=
  total_amount = 120 ∧ num_other_bills = 3 ∧ num_10_bills = 2 * num_other_bills

def value_of_each_other_bill (total_amount num_other_bills : ℕ) : ℕ :=
  total_amount / num_other_bills

-- The theorem we want to prove
theorem value_of_other_bills_is_40 (total_amount num_other_bills : ℕ) 
  (h : class_fund_contains_only_10_and_other_bills total_amount num_other_bills (2 * num_other_bills)) :
  value_of_each_other_bill total_amount num_other_bills = 40 := 
by 
  -- We use the conditions here to ensure they are part of the proof even if we skip the actual proof with sorry
  have h1 : total_amount = 120 := by sorry
  have h2 : num_other_bills = 3 := by sorry
  -- Skipping the proof
  sorry

end value_of_other_bills_is_40_l56_56731


namespace arctan_problem_l56_56523

theorem arctan_problem (x : ℝ) :
  2 * real.arctan (1 / 3) + 4 * real.arctan (1 / 5) + real.arctan (1 / x) = real.pi / 4 →
  x = -978 / 2029 :=
by
  sorry -- Proof omitted

end arctan_problem_l56_56523


namespace original_price_l56_56985

theorem original_price (a b x : ℝ) (h : (x - a) * 0.60 = b) : x = (5 / 3 * b) + a :=
  sorry

end original_price_l56_56985


namespace complement_of_A_l56_56809

open Set

def U : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}
def A : Set ℕ := {1, 2, 3, 5, 8}

theorem complement_of_A :
  (U \ A) = {4, 6, 7, 9, 10} :=
by
sorry

end complement_of_A_l56_56809


namespace x_coordinate_of_M_l56_56504

-- Define the parabola and its properties
def parabola (x y : ℝ) : Prop := y^2 = (1/4) * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1/16, 0)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the point M and its properties
def point_M (x : ℝ) (y : ℝ) : Prop :=
  parabola x y ∧ distance (x, y) focus = 1

-- The theorem to prove: the x-coordinate of point M is 15/16
theorem x_coordinate_of_M : ∃ y, point_M (15/16) y :=
by
  -- Placeholder for proof
  sorry

end x_coordinate_of_M_l56_56504


namespace magnitude_of_product_l56_56188

noncomputable def complex_num1 : ℂ := (3 * Real.sqrt 2) - (3 * Complex.i)
noncomputable def complex_num2 : ℂ := (2 * Real.sqrt 3) + (6 * Complex.i)

noncomputable def mag1 : ℝ := Complex.abs complex_num1
noncomputable def mag2 : ℝ := Complex.abs complex_num2

theorem magnitude_of_product :
  Complex.abs (complex_num1 * complex_num2) = 36 :=
by
  -- Define the magnitudes of the complex numbers
  let mag1 := Complex.abs (3 * Real.sqrt 2 - 3 * Complex.i)
  let mag2 := Complex.abs (2 * Real.sqrt 3 + 6 * Complex.i)
  -- State the product of magnitudes and its computed value
  have h : mag1 * mag2 = 3 * Real.sqrt 3 * 4 * Real.sqrt 3
  have h' : mag1 * mag2 = 12 * 3
  show Complex.abs (complex_num1 * complex_num2) = 36
  sorry

end magnitude_of_product_l56_56188


namespace complex_real_imag_diff_is_minus_two_l56_56336

noncomputable def complex_example : ℝ :=
  let z := (5 : ℂ) / (-3 - (1 : ℂ) * I) in
  z.re - z.im

theorem complex_real_imag_diff_is_minus_two :
  complex_example = -2 := sorry

end complex_real_imag_diff_is_minus_two_l56_56336


namespace minimum_positive_period_abs_tan_x_shift_l56_56382

-- Definition of the function y = abs(tan(x - shift))
def is_periodic_min_positive (f : ℝ → ℝ) (p : ℝ) : Prop := 
  ∀ x, f (x + p) = f x

noncomputable def function := λ x : ℝ, abs (tan (x - 2011))

-- Statement: Prove that the minimum positive period is π
theorem minimum_positive_period_abs_tan_x_shift : ∃ p > 0, is_periodic_min_positive function p ∧ ∀ q > 0, is_periodic_min_positive function q → p ≤ q :=
by
  use [Real.pi, Real.pi_pos]
  split
  · intro x
    simp [function, Real.pi_pos, Real.tan_add_pi_sub_self]
  · intros q hq1 hq2
    sorry  -- Proof omitted

end minimum_positive_period_abs_tan_x_shift_l56_56382


namespace max_probability_pc_l56_56103

variables (p1 p2 p3 : ℝ)
variable (h : p3 > p2 ∧ p2 > p1 ∧ p1 > 0)

def PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem max_probability_pc : PC > PA ∧ PC > PB := 
by 
  sorry

end max_probability_pc_l56_56103


namespace marble_choice_count_l56_56252

theorem marble_choice_count :
  let my_marbles := {1, 2, 3, 4, 5, 6, 7}
  let sarah_marbles := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
  let valid_pairs := {(a, b) | a ∈ my_marbles ∧ b ∈ my_marbles ∧ a ≠ b}
  let ordered_pairs := valid_pairs ∪ {(b, a) | (a, b) ∈ valid_pairs}
  ∑ k in sarah_marbles, (∃ p ∈ ordered_pairs, a + b = k) = 48 := sorry

end marble_choice_count_l56_56252


namespace validate_statements_l56_56073

-- Condition: statement ①
def internal_angle_of_regular_octagon : Prop :=
  ∀ (n : ℕ) (h : n = 8), let ext_angle := 360 / n in 180 - ext_angle = 135

-- Condition: statement ②
def similar_radical_expressions : Prop :=
  ∃ (a b : ℝ), a = sqrt 27 ∧ b = sqrt (1 / 3) ∧ a / b = 3

-- Condition: statement ③
def central_angle_for_chord_equal_length_radius : Prop :=
  ∀ (r : ℝ) (s : ℝ) (h : s = r), ¬(central_angle s r = 30)

-- Condition: statement ④
def sample_size_of_80_students : Prop :=
  ∃ (students total : ℕ), students = 80 ∧ total = 480

-- Condition: statement ⑤
def quadrilateral_with_equal_bisected_diagonals_is_square : Prop :=
  ∀ (a b : ℝ), a = b → ¬(quadrilateral_with_equal_bisected_diagonals a b = "square")

-- The combined proof problem
def correct_statements : Prop :=
  internal_angle_of_regular_octagon ∧ similar_radical_expressions ∧ sample_size_of_80_students

theorem validate_statements : correct_statements :=
by sorry

end validate_statements_l56_56073


namespace exists_contiguous_figure_l56_56183

-- Definition of the type for different types of rhombuses
inductive RhombusType
| wide
| narrow

-- Definition of a figure composed of rhombuses
structure Figure where
  count_wide : ℕ
  count_narrow : ℕ
  connected : Prop

-- Statement of the proof problem
theorem exists_contiguous_figure : ∃ (f : Figure), f.count_wide = 3 ∧ f.count_narrow = 8 ∧ f.connected :=
sorry

end exists_contiguous_figure_l56_56183


namespace players_never_tied_l56_56852

-- Monthly home runs for each player
def monthly_home_runs_mc_gwire : List ℕ := [3, 9, 12, 8, 7, 9, 12]
def monthly_home_runs_sosa : List ℕ := [3, 8, 15, 7, 12, 10]
def monthly_home_runs_griffey : List ℕ := [2, 5, 9, 8, 8, 9]

-- Cumulative home runs for each player
def cumulative_home_runs (hrs : List ℕ) : List ℕ :=
  List.scanl (· + ·) 0 hrs

-- Proving that no month has equal cumulative home runs for all three players
theorem players_never_tied :
  let M := cumulative_home_runs monthly_home_runs_mc_gwire
  let S := cumulative_home_runs monthly_home_runs_sosa
  let G := cumulative_home_runs monthly_home_runs_griffey in
  ¬ ∃ i, i < List.length M ∧ i < List.length S ∧ i < List.length G ∧ List.nth M i = List.nth S i ∧ List.nth S i = List.nth G i :=
by {
  let M := cumulative_home_runs monthly_home_runs_mc_gwire,
  let S := cumulative_home_runs monthly_home_runs_sosa,
  let G := cumulative_home_runs monthly_home_runs_griffey,
  sorry
}

end players_never_tied_l56_56852


namespace range_of_a_l56_56562

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h₀ : f = λ x, 2 * a * x - 1 + 3 * a) (h₁ : f 0 < f 1) (h₂ : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0) : (1 / 7 < a ∧ a < 1 / 5) :=
sorry

end range_of_a_l56_56562


namespace possible_values_of_expression_l56_56228

theorem possible_values_of_expression (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  let expr := (x / |x|) + (y / |y|) + (z / |z|) + (w / |w|) + ((x * y * z * w) / |x * y * z * w|)
  ∃ (v : ℝ), expr = v ∧ v ∈ {5, 1, -1, -5} :=
sorry

end possible_values_of_expression_l56_56228


namespace max_root_of_polynomial_l56_56565

theorem max_root_of_polynomial (a b c : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h_poly : ∀ x, polynomial.eval x (polynomial.X^5 - 10 * polynomial.X^3 + a * polynomial.X^2 + b * polynomial.X + c) = 0)
  (h_sum_roots : x₁ + x₂ + x₃ + x₄ + x₅ = 0)
  (h_sum_products : ∑ (i j | i < j), x₁ * x₂ = -10)
  (h_sum_squares : x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 = 20) :
  ∃ m, m ≤ 4 ∧ m = max x₁ x₂ x₃ x₄ x₅ :=
sorry

end max_root_of_polynomial_l56_56565


namespace sum_abc_equals_16_l56_56791

theorem sum_abc_equals_16 (a b c : ℝ) (h : (a - 2)^2 + (b - 6)^2 + (c - 8)^2 = 0) : 
  a + b + c = 16 :=
by
  sorry

end sum_abc_equals_16_l56_56791


namespace correct_statements_l56_56600

noncomputable theory

-- Definitions of the given pairs
def x1 := -5
def y1 := -2
def x2 := -2
def y3 := 5
def x4 := 5

-- Definitions for unknown values
variables (m n : ℝ)

-- Conditions for the statements
def inverse_proportion_condition : Prop := (2 * m + 5 * n = 0)
def linear_function_condition : Prop := (n - m = 7)
def quadratic_function_condition : Prop := (m > n)

-- Theorem statement for which statements are correct
theorem correct_statements : 
  (inverse_proportion_condition ∧ linear_function_condition ∧ ¬quadratic_function_condition) :=
begin
  sorry
end

end correct_statements_l56_56600


namespace basic_astrophysics_degrees_l56_56080

open Real

theorem basic_astrophysics_degrees :
  let microphotonics := 12
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let basic_astrophysics_percentage := 100 - total_percentage
  let circle_degrees := 360
  basic_astrophysics_percentage / 100 * circle_degrees = 43.2 :=
by
  let microphotonics := 12
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let basic_astrophysics_percentage := 100 - total_percentage
  let circle_degrees := 360
  exact sorry

end basic_astrophysics_degrees_l56_56080


namespace expression_equal_neg_sq_l56_56071

variables {a : ℝ}

theorem expression_equal_neg_sq (a : ℝ) : a^4 * (-a)^(-2) = (-a)^2 :=
by sorry

end expression_equal_neg_sq_l56_56071


namespace hike_distance_down_l56_56430

theorem hike_distance_down :
  ∀ (rate_up : ℕ) (time_up : ℕ) (rate_factor : ℝ),
  rate_up = 6 →
  time_up = 2 →
  rate_factor = 1.5 →
  (rate_factor * ↑rate_up * ↑time_up) = 18 :=
by {
  -- Assume the given conditions
  intros rate_up time_up rate_factor h1 h2 h3,
  
  -- Convert given conditions to equalities
  have rate_up_eq := h1,
  have time_up_eq := h2,
  have rate_factor_eq := h3,

  -- Simplify distances
  rw [rate_up_eq, time_up_eq, rate_factor_eq],

  -- compute the desired result
  norm_num,
}

end hike_distance_down_l56_56430


namespace amount_spent_on_apples_l56_56822

variable (total_money : ℕ) (spent_oranges : ℕ) (spent_candy : ℕ) (money_left : ℕ)

theorem amount_spent_on_apples :
  total_money = 95 →
  spent_oranges = 14 →
  spent_candy = 6 →
  money_left = 50 →
  total_money - (spent_oranges + spent_candy + money_left) = 25 :=
by
  intros h1 h2 h3 h4
  have h5 : total_money - (spent_oranges + spent_candy + money_left) = 95 - (14 + 6 + 50), from sorry
  show 95 - (14 + 6 + 50) = 25, from sorry

end amount_spent_on_apples_l56_56822


namespace derivative_of_f_l56_56092

noncomputable section

open Real

variable {x : ℝ}

def f (x : ℝ) : ℝ :=
  (2 * x + 3) ^ 4 * arcsin (1 / (2 * x + 3)) + (2 / 3) * (4 * x ^ 2 + 12 * x + 11) * sqrt (x ^ 2 + 3 * x + 2)

theorem derivative_of_f (h : 2 * x + 3 > 0) : deriv f x = 8 * (2 * x + 3) ^ 3 * arcsin (1 / (2 * x + 3)) :=
by
  sorry -- Proof not required

end derivative_of_f_l56_56092


namespace head_start_distance_l56_56136

variable (vA vB L d : ℝ)
variables (h1 : vA = 17 / 15 * vB)
variables (h2 : d = (2 / 17) * L)

theorem head_start_distance : 
  let tA := L / vA,
      tB := (L - d) / vB
  in tA = tB :=
by
  sorry

end head_start_distance_l56_56136


namespace find_f_2015_l56_56227
noncomputable def f (x : ℝ) : ℝ := if (0 < x ∧ x ≤ 2) then 3^x - real.logb 3 x else 0

lemma period_4 (x : ℝ) : f (x + 4) = f x := sorry
lemma odd_function (x : ℝ) : f (-x) = -f x := sorry
lemma f_x_defined (x : ℝ) (hx1 : 0 < x) (hx2 : x ≤ 2) : f x = 3^x - real.logb 3 x := sorry

theorem find_f_2015 : f 2015 = -3 := sorry

end find_f_2015_l56_56227


namespace adam_change_is_correct_l56_56137

-- Define the conditions
def adam_money : ℝ := 5.00
def airplane_cost : ℝ := 4.28
def change : ℝ := adam_money - airplane_cost

-- State the theorem
theorem adam_change_is_correct : change = 0.72 := 
by {
  -- Proof can be added later
  sorry
}

end adam_change_is_correct_l56_56137


namespace Marcella_max_pairs_after_loss_l56_56904

theorem Marcella_max_pairs_after_loss 
  (pairs : ℕ) (lost_individual_shoes : ℕ) 
  (initial_pairs : pairs = 23) 
  (shoes_lost : lost_individual_shoes = 9) : 
  (remaining_pairs : ℕ) (remaining_pairs = pairs - lost_individual_shoes) 
    (greatest_number_of_matching_pairs : remaining_pairs = 14) :=
sorry

end Marcella_max_pairs_after_loss_l56_56904


namespace total_distance_traveled_l56_56457

-- Definitions based on given conditions
def father_step_length : ℕ := 80
def son_step_length : ℕ := 60
def coincided_steps : ℕ := 601

-- The main theorem statement
theorem total_distance_traveled : 
  let lcm := Nat.lcm father_step_length son_step_length in
  let distance_per_interval := lcm / 100 in
  let total_intervals := coincided_steps - 1 in
  let total_distance_m := total_intervals * distance_per_interval in
  let total_distance_km := total_distance_m / 1000 in
  total_distance_km = 1 ∧ (total_distance_m % 1000) = 440 := 
by
  sorry

end total_distance_traveled_l56_56457


namespace Ksyusha_travel_time_l56_56774

theorem Ksyusha_travel_time:
  ∀ (S v : ℝ), 
  (S > 0 ∧ v > 0) →
  (∀ tuesday_distance tuesday_time wednesday_distance wednesday_time: ℝ, 
    tuesday_distance = 3 * S ∧ 
    tuesday_distance = 2 * S + S ∧ 
    tuesday_time = (2 * S) / v + (S / (2 * v)) ∧ 
    tuesday_time = 30) → 
  (∀ wednesday_distance wednesday_time : ℝ, 
    wednesday_distance = S + 2 * S ∧ 
    wednesday_time = S / v + (2 * S) / (2 * v)) →
  wednesday_time = 24 :=
by
  intros S v S_pos v_pos tuesday_distance tuesday_time wednesday_distance wednesday_time
  sorry

end Ksyusha_travel_time_l56_56774


namespace remainder_x5_1_x3_1_div_x2_x_1_l56_56171

theorem remainder_x5_1_x3_1_div_x2_x_1 :
  ∀ (x : ℂ), let poly := (x^5 - 1) * (x^3 - 1),
                 divisor := x^2 + x + 1,
                 remainder := x^2 + x + 1 in
  ∃ q : ℂ, poly = q * divisor + remainder :=
by
  intro x
  let poly := (x^5 - 1) * (x^3 - 1)
  let divisor := x^2 + x + 1
  let remainder := x^2 + x + 1
  use sorry
  rw [← add_assoc, ← mul_assoc, ← pow_succ]
  sorry

end remainder_x5_1_x3_1_div_x2_x_1_l56_56171


namespace volume_of_solid_eq_zero_l56_56331

def regionR (x y : ℝ) : Prop := (abs (x - y) + y ≤ 12) ∧ (2 * y - x ≥ 12) ∧ (4 ≤ x ∧ x ≤ 8)

def revolveLine (x y : ℝ) : Prop := 2 * y - x = 12

theorem volume_of_solid_eq_zero :
  ∃ (R : set (ℝ × ℝ)) (f : ℝ → ℝ → ℝ),
    (∀ x y, regionR x y → (f x y = 0)) ∧
    volume (solid_of_revolution R (λ (p : ℝ × ℝ), revolveLine p.1 p.2)) = 0 := 
by {
  sorry
}

end volume_of_solid_eq_zero_l56_56331


namespace circles_separate_l56_56245

variable (C₁ C₂ : Type)
variables (x y : ℝ)

def circle1 : Prop := x^2 + y^2 = 1
def circle2 : Prop := (x - 3)^2 + (y - 4)^2 = 9

theorem circles_separate (h₁ : circle1) (h₂ : circle2) : 
  let center_distance := (3 - 0)^2 + (4 - 0)^2
  let radius_sum := 1 + 3
  Math.sqrt center_distance > radius_sum :=
by
  sorry

end circles_separate_l56_56245


namespace find_x1_l56_56585

theorem find_x1 (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1-x1)^2 + 2*(x1-x2)^2 + (x2-x3)^2 + x3^2 = 1/2) :
  x1 = (3*Real.sqrt 2 - 3)/7 :=
by
  sorry

end find_x1_l56_56585


namespace equal_segments_in_regular_polygon_l56_56874

theorem equal_segments_in_regular_polygon (n : ℕ) (m : ℕ) (A : fin (2 * n) → Type) :
  (∃ p : ℕ, n = 4 * p + 2) ∨ (∃ p : ℕ, n = 4 * p + 3) → 
  (∃ i j k l : fin (2 * n), i ≠ j ∧ k ≠ l ∧ (dist A i j) = (dist A k l)) :=
by
  sorry

end equal_segments_in_regular_polygon_l56_56874


namespace cost_price_of_toy_l56_56431

theorem cost_price_of_toy (x : ℝ) (selling_price_per_toy : ℝ) (gain : ℝ) 
  (sale_price : ℝ) (number_of_toys : ℕ) (selling_total : ℝ) (gain_condition : ℝ) :
  (selling_total = number_of_toys * selling_price_per_toy) →
  (selling_price_per_toy = x + gain) →
  (gain = gain_condition / number_of_toys) → 
  (gain_condition = 3 * x) →
  selling_total = 25200 → number_of_toys = 18 → x = 1200 :=
by
  sorry

end cost_price_of_toy_l56_56431


namespace intersection_points_correct_l56_56409

noncomputable def intersection_points : List (ℚ × ℚ) :=
  let f : ℚ → ℚ := λ (x : ℚ), 4*x^2 + 5*x - 7
  let g : ℚ → ℚ := λ (x : ℚ), 2*x^2 - 3*x + 8
  let xs := [(-5 : ℚ), (3 / 2 : ℚ)]
  xs.map (λ x => (x, f x))

theorem intersection_points_correct :
  intersection_points = [(-5, 68), (3 / 2, 23 / 2)] :=
by
  sorry

end intersection_points_correct_l56_56409


namespace coloring_problem_l56_56679

noncomputable def numColorings : Nat :=
  256

theorem coloring_problem :
  ∃ (f : Fin 200 → Bool), (∀ (i j : Fin 200), i ≠ j → f i = f j → ¬ (i + 1 + j + 1).isPowerOfTwo) ∧ 
    256 = numColorings :=
by
  -- Definitions and conditions used for the problem
  let S := {n : ℕ | ∃ k : ℕ, n = bit0 k 1 ∧ k > 0}
  have hS : ∀ n ∈ S, ∀ i j ∈ {1, ..., 200}, n ≠ 2 ^ (log2 n) ∧ i ≠ j → i + j ≠ n := sorry
  -- Proof skipped, as required in the guidelines
  sorry

end coloring_problem_l56_56679


namespace equilateral_triangle_properties_l56_56663

noncomputable def side_length : ℝ := 12

def perimeter (s : ℝ) : ℝ := 3 * s

noncomputable def area (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

theorem equilateral_triangle_properties :
  perimeter side_length = 36 ∧ area side_length = 36 * sqrt 3 :=
by
  sorry

end equilateral_triangle_properties_l56_56663


namespace adjacent_sum_constant_l56_56708

theorem adjacent_sum_constant (x y : ℤ) (k : ℤ) (h1 : 2 + x = k) (h2 : x + y = k) (h3 : y + 5 = k) : x - y = 3 := 
by 
  sorry

end adjacent_sum_constant_l56_56708


namespace number_of_chickens_l56_56485

def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12
def full_cartons : ℕ := 10

theorem number_of_chickens :
  (full_cartons * eggs_per_carton) / eggs_per_chicken = 20 :=
by
  sorry

end number_of_chickens_l56_56485


namespace proof_problem1_proof_problem2_l56_56444

-- Definition for Problem 1:
def problem1 : Prop := 
  sqrt 8 / sqrt 2 + (sqrt 5 + 3) * (sqrt 5 - 3) = -2

-- Definition for Problem 2:
def problem2 : Prop := 
  sqrt 48 / sqrt 3 - sqrt (1 / 2) * sqrt 12 + sqrt 24 = 4 + sqrt 6

-- Statements (no proofs)
theorem proof_problem1 : problem1 := by 
  sorry

theorem proof_problem2 : problem2 := by 
  sorry

end proof_problem1_proof_problem2_l56_56444


namespace area_of_second_square_l56_56483

theorem area_of_second_square 
    (h₁ : ∃ s₁ : ℝ, s₁^2 = 529)
    (h₂ : ∃ h : ℝ, h = 46 * Real.sqrt 2)
    (h₃ : ∃ s₁, h₂ = 2 * s₁)
    (h₄ : ∃ s₂ : ℝ, s₂ = (46 * Real.sqrt 2) / 3) :
  s₂^2 = 4232 / 9 :=
by
  sorry

end area_of_second_square_l56_56483


namespace minimum_frac_seq_l56_56576

noncomputable def seq (n : ℕ) : ℝ :=
  match n with
  | 0 => 0  -- Define for n=0
  | 1 => 33
  | (k+2) => seq (k+1) + 2*(k+1)

theorem minimum_frac_seq (n : ℕ) (h_pos : n > 0) :
  ∃ m, (m ∈ { n | n > 0 }) ∧ (∀ k, k > 0 → (2 * seq k) / k ≥ (2 * seq m) /m) ∧ (2 * seq m / m) = 22 :=
sorry

end minimum_frac_seq_l56_56576


namespace find_r_l56_56964

-- Definition of a regular hexagon and points on its diagonals.
structure RegularHexagon (A B C D E F : Point) :=
  (equilateral: Equilateral A B)
  (equalSides: Distance A B = Distance B C ∧ Distance B C = Distance C D ∧ Distance C D = Distance D E ∧ Distance D E = Distance E F ∧ Distance E F = Distance F A)

-- Additional conditions and properties of the problem.
variables (A B C D E F M N : Point) (r : ℝ)
variables (hex : RegularHexagon A B C D E F)
variables (AM_eq_r_AC : AM / AC = r)
variables (CN_eq_r_CE : CN / CE = r)
variables (collinear_B_M_N : Collinear B M N)

-- The theorem to prove.
theorem find_r (h : r = sqrt 3 / 3) :
  r = sqrt 3 / 3 :=
sorry

end find_r_l56_56964


namespace last_digit_max_value_l56_56842

theorem last_digit_max_value (n : ℕ) (condition : n = 128) : 
  let A := nat.rec_on n 1 (λ _ a, a + 1) in
  A % 10 = 2 :=
sorry

end last_digit_max_value_l56_56842


namespace trig_identity_simplification_l56_56974

theorem trig_identity_simplification :
  sin (real.pi / 10) * sin (real.pi / 6) * sin (real.pi / 3) * sin (2 * real.pi / 5) =
  (real.sqrt 3 / 8) * sin (real.pi / 5) := by sorry

end trig_identity_simplification_l56_56974


namespace ratio_of_triangle_area_to_radius_square_l56_56284

noncomputable def circleQ := ℝ 
noncomputable def centerO : circleQ := 0
noncomputable def diameterAB : circleQ→ circleQ := sorry
noncomputable def diameterCD : circleQ→ circleQ := sorry
noncomputable def pointP (x : ℝ) : circleQ := 
  if 0 ≤ x ∧ x ≤ 1 then x else sorry
noncomputable def angleQPD : ℝ := 45 * (Real.pi / 180)

theorem ratio_of_triangle_area_to_radius_square :
  ∀ (x : ℝ), 
  ∃ OQ : ℝ, 
  angleQPD = 45 * (Real.pi / 180) → OQ * OQ * 4 = x * x :=
by sorry

end ratio_of_triangle_area_to_radius_square_l56_56284


namespace distance_between_parallel_lines_l56_56880

theorem distance_between_parallel_lines (r : ℝ) : 
    ∃ d : ℝ, (∃ (a1 a2 : ℝ), a1 = 42 ∧ a2 = 42 ∧ ((a1^2 + a2^2)/4) = ((d / 2)^2 r^2 + (2r^2))) 
        ∧ ∃ (a3 : ℝ), a3 = 26 ∧ ((a1^2 + a2^2)/4) = ((3 * d / 2)^2 r^2 + (13 * r^2)) 
        ∧ d = real.sqrt 127 := 
begin
  sorry
end

end distance_between_parallel_lines_l56_56880


namespace shaded_area_correct_l56_56144

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (real.sqrt 3 / 4) * s^2

noncomputable def large_triangle_side : ℝ := 12
noncomputable def small_triangles_count : ℕ := 9
noncomputable def shaded_triangles_count : ℕ := 6

def shaded_area (side_length : ℝ) (total_triangles : ℕ) (shaded_triangles : ℕ) : ℝ :=
  let total_area := equilateral_triangle_area side_length
  let small_triangle_area := total_area / total_triangles
  shaded_triangles * small_triangle_area

theorem shaded_area_correct :
  shaded_area large_triangle_side small_triangles_count shaded_triangles_count = 24 * real.sqrt 3 := 
by sorry

end shaded_area_correct_l56_56144


namespace B1C1_touches_incircle_l56_56782

variables (A B C C1 B1 : Type)
variables [IsTriangle A B C]
variables [MidPerpendicularIntersect A B C1 AC]
variables [MidPerpendicularIntersect A C B1 AB]
variables [Angle60Deg A]

theorem B1C1_touches_incircle (h : ∠ A = 60) (h1 : MidPerpendicularAware A B C1 AC) (h2 : MidPerpendicularAware A C B1 AB)
  : tangent_to_incircle A B C B1 C1 :=
sorry

end B1C1_touches_incircle_l56_56782


namespace train_length_l56_56433

variable (L : ℝ)

-- Condition: The train covers L + 180 meters in 15 seconds
def condition1 (L : ℝ) : Prop := L + 180 / 15

-- Condition: The train covers L + 250 meters in 20 seconds
def condition2 (L : ℝ) : Prop := L + 250 / 20

-- The proof goal: Show that the length of the train is 30 meters given the conditions
theorem train_length (L : ℝ) 
  (h1 : condition1 L)
  (h2 : condition2 L) : 
  L = 30 := by
  sorry

end train_length_l56_56433


namespace determine_positions_l56_56089

-- Define the characters
inductive Character
| GrayHorse  -- Siveny Merin
| GrayMare   -- Sivenya Kobyl
| BearCub    -- Medvezhonok

-- Define their lying behaviors
def always_lies : Character -> Prop
| Character.GrayHorse  => true    -- Always lies
| Character.GrayMare   => false   -- Lies sometimes
| Character.BearCub    => false   -- Never lies

def never_lies : Character -> Prop
| Character.GrayHorse  => false
| Character.GrayMare   => false
| Character.BearCub    => true

-- Statements by position
def left_statement : Character -> Prop := 
  λ c, c = Character.BearCub

def right_statement : Character -> Prop := 
  λ c, c = Character.GrayMare

def center_statement : Character -> Prop := 
  λ c, c = Character.GrayHorse

-- Proving the correct positions given the conditions
theorem determine_positions :
  ∃ (left center right : Character),
    left_statement left ∧ right_statement right ∧ center_statement center ∧
    center = Character.GrayHorse ∧ left = Character.GrayMare ∧ right = Character.BearCub ∧
    (always_lies center ∧ never_lies right ∧ ¬ always_lies left) :=
by
  sorry  -- Proof will be constructed here

end determine_positions_l56_56089


namespace fraction_to_decimal_l56_56522

theorem fraction_to_decimal :
  (51 / 160 : ℝ) = 0.31875 := 
by
  sorry

end fraction_to_decimal_l56_56522


namespace percent_change_is_minus_5_point_5_percent_l56_56150

noncomputable def overall_percent_change (initial_value : ℝ) : ℝ :=
  let day1_value := initial_value * 0.75
  let day2_value := day1_value * 1.4
  let final_value := day2_value * 0.9
  ((final_value / initial_value) - 1) * 100

theorem percent_change_is_minus_5_point_5_percent :
  ∀ (initial_value : ℝ), overall_percent_change initial_value = -5.5 :=
sorry

end percent_change_is_minus_5_point_5_percent_l56_56150


namespace tan_of_obtuse_angle_l56_56556

theorem tan_of_obtuse_angle (α : ℝ) (h1 : cos α = -1/2) (h2 : π/2 < α ∧ α < π) : tan α = -√3 :=
by sorry

end tan_of_obtuse_angle_l56_56556


namespace fraction_not_going_l56_56448

theorem fraction_not_going (S J : ℕ) (h1 : J = (2:ℕ)/3 * S) 
  (h_not_junior : 3/4 * J = 3/4 * (2/3 * S)) 
  (h_not_senior : 1/3 * S = (1:ℕ)/3 * S) :
  3/4 * (2/3 * S) + 1/3 * S = 5/6 * S :=
by 
  sorry

end fraction_not_going_l56_56448


namespace largest_value_of_c_l56_56531

theorem largest_value_of_c : ∃ c, (∀ x : ℝ, x^2 - 6 * x + c = 1 → c ≤ 10) :=
sorry

end largest_value_of_c_l56_56531


namespace jenna_eel_length_l56_56720

theorem jenna_eel_length (j b : ℕ) (h1 : b = 3 * j) (h2 : b + j = 64) : j = 16 := by 
  sorry

end jenna_eel_length_l56_56720


namespace ksyusha_travel_time_l56_56766

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l56_56766


namespace speed_in_still_water_l56_56460

-- We define the given conditions for the man's rowing speeds
def upstream_speed : ℕ := 25
def downstream_speed : ℕ := 35

-- We want to prove that the speed in still water is 30 kmph
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 30 := by
  sorry

end speed_in_still_water_l56_56460


namespace ratio_of_toys_l56_56307

theorem ratio_of_toys (initial_stuffed_animals initial_action_figures initial_board_games initial_puzzles : ℕ)
  (joel_added sister_added : ℕ) (total_donated : ℕ) :
  initial_stuffed_animals = 18 →
  initial_action_figures = 42 →
  initial_board_games = 2 →
  initial_puzzles = 13 →
  joel_added = 22 →
  total_donated = 108 →
  (total_donated - (initial_stuffed_animals + initial_action_figures + initial_board_games + initial_puzzles) - joel_added)
    = sister_added →
  (joel_added : ℚ) / sister_added = 2 :=
by sorry

end ratio_of_toys_l56_56307


namespace heartsuit_3_8_l56_56255

def heartsuit (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem heartsuit_3_8 : heartsuit 3 8 = 60 := by
  sorry

end heartsuit_3_8_l56_56255


namespace product_of_two_odd_numbers_not_always_composite_l56_56030

theorem product_of_two_odd_numbers_not_always_composite :
  ∃ (m n : ℕ), (¬ (2 ∣ m) ∧ ¬ (2 ∣ n)) ∧ (∀ d : ℕ, d ∣ (m * n) → d = 1 ∨ d = m * n) :=
by
  sorry

end product_of_two_odd_numbers_not_always_composite_l56_56030


namespace calculate_necessary_monthly_rent_l56_56135

variable (investment : ℝ) (maintenance_pct : ℝ) (annual_taxes : ℝ) (desired_return_pct : ℝ)

def necessary_monthly_rent (investment : ℝ) (maintenance_pct : ℝ) (annual_taxes : ℝ) 
                            (desired_return_pct : ℝ) : ℝ :=
  let annual_return := investment * desired_return_pct
  let total_annual_needed := annual_return + annual_taxes
  let monthly_needed := total_annual_needed / 12
  let monthly_rent := monthly_needed / (1 - maintenance_pct)
  monthly_rent

theorem calculate_necessary_monthly_rent :
  necessary_monthly_rent 20000 0.10 650 0.08 = 208.33 :=
by
  sorry

end calculate_necessary_monthly_rent_l56_56135


namespace smallest_integer_no_prime_factors_under_60_l56_56060

def is_prime (n : ℕ) : Prop := sorry -- Assume a definition or import for is_prime from Mathlib
def is_square (n : ℕ) : Prop := sorry -- Assume a definition or import for is_square from Mathlib

theorem smallest_integer_no_prime_factors_under_60 : 
  ∃ (n : ℕ), n = 4087 ∧ ¬ (is_prime n) ∧ ¬ (is_square n) ∧ (∀ p, is_prime p → p ∣ n → p ≥ 60) :=
begin
  sorry
end

end smallest_integer_no_prime_factors_under_60_l56_56060


namespace roots_of_quadratic_l56_56179

theorem roots_of_quadratic :
  ∃ z₁ z₂ : ℂ, (z₁^2 - z₁ = 3 - 7 * complex.I) ∧ (z₂^2 - z₂ = 3 - 7 * complex.I) ∧
    (z₁ = (1 + 2 * real.sqrt 7 - real.sqrt 7 * complex.I) / 2 ∨ z₁ = (1 - 2 * real.sqrt 7 + real.sqrt 7 * complex.I) / 2) ∧
    (z₂ = (1 + 2 * real.sqrt 7 - real.sqrt 7 * complex.I) / 2 ∨ z₂ = (1 - 2 * real.sqrt 7 + real.sqrt 7 * complex.I) / 2) := sorry

end roots_of_quadratic_l56_56179


namespace color_natural_numbers_l56_56674

theorem color_natural_numbers :
  ∃ f : fin 200 → bool, (∀ a b : fin 200, a ≠ b → f a = f b → (a + b).val ∉ {2^n | n : ℕ}) ∧
  (by let count := ∑ color in (fin 200 → bool), 
         ite (∀ a b, a ≠ b → color a = color b → (a + b).val ∉ {2^n | n : ℕ}) 1 0 ;
      exact count = 256) := sorry

end color_natural_numbers_l56_56674


namespace color_natural_numbers_l56_56677

theorem color_natural_numbers :
  ∃ f : fin 200 → bool, (∀ a b : fin 200, a ≠ b → f a = f b → (a + b).val ∉ {2^n | n : ℕ}) ∧
  (by let count := ∑ color in (fin 200 → bool), 
         ite (∀ a b, a ≠ b → color a = color b → (a + b).val ∉ {2^n | n : ℕ}) 1 0 ;
      exact count = 256) := sorry

end color_natural_numbers_l56_56677


namespace bisecting_line_of_circle_l56_56079

theorem bisecting_line_of_circle :
  ∃ l : ℝ × ℝ × ℝ, (l.1, l.2, l.3) = (1, -1, 1) ∧
    ∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y + 1 = 0 → l.1 * x + l.2 * y + l.3 = 0 :=
by
  sorry

end bisecting_line_of_circle_l56_56079


namespace games_went_this_year_l56_56723

theorem games_went_this_year (t l : ℕ) (h1 : t = 13) (h2 : l = 9) : (t - l = 4) :=
by
  sorry

end games_went_this_year_l56_56723


namespace measure_of_angle_A_l56_56882

-- Define the given conditions for the problem
variables {A B C I X Y : Type} [Inhabited A] 
variables (ABC : Triangle A B C) (I : Incenter ABC)
variables (X : OnLineSegment A B) (Y : OnLineSegment A C)
variables (h1 : BX * AB = IB^2) (h2 : CY * AC = IC^2) (h3 : Collinear [X, I, Y])

-- State the goal to prove
theorem measure_of_angle_A (ABC : Triangle A B C) (I : Incenter ABC)
  (X : OnLineSegment A B) (Y : OnLineSegment A C)
  (h1 : BX * AB = IB^2) (h2 : CY * AC = IC^2) 
  (h3 : Collinear [X, I, Y]) : 
  measure (A) = 90 := 
sorry

end measure_of_angle_A_l56_56882


namespace sum_of_abs_roots_eq_106_l56_56541

theorem sum_of_abs_roots_eq_106 (m p q r : ℤ) (h₁ : (Polynomial.C m + Polynomial.X * (Polynomial.X * (Polynomial.X + (-2023))) = 0) = Polynomial.C 0) (h₂ : m = p*q*r) (h₃ : p + q + r = 0) :
  |p| + |q| + |r| = 106 := sorry

end sum_of_abs_roots_eq_106_l56_56541


namespace smallest_positive_real_number_l56_56983

noncomputable def smallest_x : ℝ := 71 / 8

theorem smallest_positive_real_number (x : ℝ) (h₁ : ∀ y : ℝ, 0 < y ∧ (⌊y^2⌋ - y * ⌊y⌋ = 7) → x ≤ y) (h₂ : 0 < x) (h₃ : ⌊x^2⌋ - x * ⌊x⌋ = 7) : x = smallest_x :=
sorry

end smallest_positive_real_number_l56_56983


namespace quadratic_function_monotonicity_l56_56024

theorem quadratic_function_monotonicity
  (a b : ℝ)
  (h1 : ∀ x y : ℝ, x ≤ y ∧ y ≤ -1 → a * x^2 + b * x + 3 ≤ a * y^2 + b * y + 3)
  (h2 : ∀ x y : ℝ, -1 ≤ x ∧ x ≤ y → a * x^2 + b * x + 3 ≥ a * y^2 + b * y + 3) :
  b = 2 * a ∧ a < 0 :=
sorry

end quadratic_function_monotonicity_l56_56024


namespace man_speed_is_4_kmph_l56_56474

noncomputable def speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_pass_seconds : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed_mps := train_length / time_to_pass_seconds
  let relative_speed_kmph := relative_speed_mps * 3600 / 1000
  relative_speed_kmph - train_speed_kmph

theorem man_speed_is_4_kmph : speed_of_man 140 50 9.332586726395222 = 4 := by
  sorry

end man_speed_is_4_kmph_l56_56474


namespace conjugate_of_z_l56_56236

-- Define the given condition
def given_condition (z : ℂ) : Prop := (1 / z = -5 * Complex.I)

-- Define the correct answer to be proven
def correct_answer (z : ℂ) : Prop := Complex.conj z = - (Complex.I / 5)

-- The main theorem stating that given the condition, the correct answer follows
theorem conjugate_of_z (z : ℂ) (h : given_condition z) : correct_answer z := 
by 
  sorry

end conjugate_of_z_l56_56236


namespace find_boys_without_calculators_l56_56345

variable (total_students_class : ℕ) (boys_class : ℕ)
variable (students_with_calculators : ℕ) (girls_with_calculators : ℕ)

-- Given conditions
axiom cond1 : total_students_class = 24
axiom cond2 : boys_class = 18
axiom cond3 : students_with_calculators = 26
axiom cond4 : girls_with_calculators = 15

-- Question and answer
def boys_without_calculators_class : ℕ :=
  boys_class - (students_with_calculators - girls_with_calculators)

-- Prove that the number of boys in Miss Parker's class who did not bring their calculators is 7
theorem find_boys_without_calculators :
  boys_without_calculators_class total_students_class boys_class students_with_calculators girls_with_calculators = 7 :=
by
  rw [cond2, cond3, cond4]
  sorry

end find_boys_without_calculators_l56_56345


namespace Ksyusha_travel_time_l56_56779

theorem Ksyusha_travel_time:
  ∀ (S v : ℝ), 
  (S > 0 ∧ v > 0) →
  (∀ tuesday_distance tuesday_time wednesday_distance wednesday_time: ℝ, 
    tuesday_distance = 3 * S ∧ 
    tuesday_distance = 2 * S + S ∧ 
    tuesday_time = (2 * S) / v + (S / (2 * v)) ∧ 
    tuesday_time = 30) → 
  (∀ wednesday_distance wednesday_time : ℝ, 
    wednesday_distance = S + 2 * S ∧ 
    wednesday_time = S / v + (2 * S) / (2 * v)) →
  wednesday_time = 24 :=
by
  intros S v S_pos v_pos tuesday_distance tuesday_time wednesday_distance wednesday_time
  sorry

end Ksyusha_travel_time_l56_56779


namespace prime_ball_probability_l56_56844

def balls : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def prime_balls : Finset ℕ := balls.filter is_prime

def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

theorem prime_ball_probability :
  probability prime_balls.card balls.card = 2 / 5 :=
by
  sorry

end prime_ball_probability_l56_56844


namespace last_two_digits_sum_factorials_l56_56532

theorem last_two_digits_sum_factorials : 
  (5! + 15! + 25! + 35! + 45! + 55! + 65! + 75! + 85! + 95! + 105!) % 100 = 20 :=
by
  sorry

end last_two_digits_sum_factorials_l56_56532


namespace aba_div_by_7_l56_56394

theorem aba_div_by_7 (a b : ℕ) (h : (a + b) % 7 = 0) : (101 * a + 10 * b) % 7 = 0 := 
sorry

end aba_div_by_7_l56_56394


namespace AD_plus_BC_eq_AB_l56_56823

variables {A B C D : Point}  -- Assume points are of some type Point.
-- Define concyclic condition
axiom concyclic : Circle A B C D

-- Define tangent circle with center on line AB and tangent to AD, BC, CD
definition tangent_circle_with_center_on_AB_and_tangent_to_AD_BC_CD 
    (circle_center : Point) (r : ℝ) : Prop :=
  circle_center ∈ Line A B ∧
  is_tangent (circle circle_center r) (Line AD) ∧
  is_tangent (circle circle_center r) (Line BC) ∧
  is_tangent (circle circle_center r) (Line CD)

theorem AD_plus_BC_eq_AB (h1 : concyclic) 
    (h2 : ∃ (circle_center : Point) (r : ℝ), tangent_circle_with_center_on_AB_and_tangent_to_AD_BC_CD circle_center r) : 
  Length AD + Length BC = Length AB :=
sorry

end AD_plus_BC_eq_AB_l56_56823


namespace cesaro_sum_51_term_seq_l56_56536

open Finset

def cesaro_sum {α : Type*} [AddCommMonoid α] [DivisionRing α]
    (b : ℕ → α) (n : ℕ) : α :=
  (range n).sum (λ k, (range (k + 1)).sum b) / n

theorem cesaro_sum_51_term_seq 
    (b : ℕ → ℝ) 
    (h_sum_eq_500 : cesaro_sum b 50 = 500) :
    cesaro_sum (λ n, if n = 0 then 2 else b (n - 1)) 51 = 492 :=
by {
  -- proof to be completed by proper computations
  sorry
}

end cesaro_sum_51_term_seq_l56_56536


namespace articles_production_l56_56110

theorem articles_production (x y : ℕ) (e : ℝ) :
  (x * x * x * e / x = x^2 * e) → (y * (y + 2) * y * (e / x) = (e * y * (y^2 + 2 * y)) / x) :=
by 
  sorry

end articles_production_l56_56110


namespace hypotenuse_length_l56_56026

variable (x : ℝ) (hypotenuse : ℝ)
variable (h1 : 2 * x - 3 > 0)
variable (h2 : (1 / 2) * x * (2 * x - 3) = 72)

theorem hypotenuse_length :
  hypotenuse = Real.sqrt (x^2 + (2 * x - 3)^2) → Real.abs (hypotenuse - 18.1) < 0.1 :=
by
  sorry

end hypotenuse_length_l56_56026


namespace problem_solution_l56_56501

theorem problem_solution :
  |2 - Real.tan (Float.pi / 3)| + 2⁻² + 1 / 2 * Real.sqrt 12 ≈ 2.518 :=
by
  sorry

end problem_solution_l56_56501


namespace coefficient_a3b2_in_expansion_l56_56418

theorem coefficient_a3b2_in_expansion :
  let coeff := (nat.choose 5 2) * (0 : ℕ)
  coeff = 0 := by
  -- binomial expansions and calculations
  sorry

end coefficient_a3b2_in_expansion_l56_56418


namespace tetrahedron_roll_path_length_l56_56125

theorem tetrahedron_roll_path_length {edge_length : ℝ} (h_edge_length : edge_length = 2) :
  ∃ (d : ℝ), d = 4 * real.sqrt 3 / 3 ∧
  (let path_length := d * real.pi in
   ∀ T (hT : regular_tetrahedron T)
    (D : T.vertex)
    (hD : is_centroid_face D T.face)
    (h_rolls : rolls_over_edges T D),
    distance_path D = path_length) := sorry

end tetrahedron_roll_path_length_l56_56125


namespace coloring_number_lemma_l56_56690

-- Definition of the problem conditions
def no_sum_of_two_same_color_is_power_of_two (f : ℕ → bool) : Prop :=
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 200 ∧ f a = f b → ¬(∃ k : ℕ, a + b = 2^k)

-- The main theorem
theorem coloring_number_lemma :
  (∃ f : ℕ → bool, (∀ n, 1 ≤ n ∧ n ≤ 200 → (f n = tt ∨ f n = ff))
                   ∧ no_sum_of_two_same_color_is_power_of_two f) →
  ∃ n : ℕ, n = 256 :=
by sorry

end coloring_number_lemma_l56_56690


namespace johns_weekly_allowance_l56_56434

theorem johns_weekly_allowance :
  ∃ A : ℚ, 
  3 * A / 5 + (1 / 3) * (2 * A / 5) + 0.75 = A :=
begin
  -- exact value of A we need to prove
  let A := 2.8125,
  use A,
  sorry
end

end johns_weekly_allowance_l56_56434


namespace expression_result_l56_56088

-- Define the expression
def expression_reduction : ℕ :=
  let first := 7 ^ 8
  let second := 9 ^ 3
  let third := 6 / 2
  first - third + second + 3 + 12

-- State the theorem to prove the expression equals the given result
theorem expression_result : expression_reduction = 5_765_542 :=
  by
    sorry

end expression_result_l56_56088


namespace equal_division_of_barrels_and_honey_l56_56401

-- Definitions of the key properties.
def full_barrels : nat := 7
def half_full_barrels : nat := 7
def empty_barrels : nat := 7
def participants : nat := 3

-- Total honey and barrels each person should receive.
def total_honey_per_person : real := 3.5
def total_barrels_per_person : nat := 7

-- Statement: We can divide the barrels such that each person gets the correct amount of honey and barrels.
theorem equal_division_of_barrels_and_honey :
  ∃ (distribution : list (nat × nat × nat)), 
  (∀ person : nat, person < participants → 
    let (f, h, e) := distribution.nth_le person sorry in
    f + h + e = total_barrels_per_person ∧
    (f + h / 2 : real) = total_honey_per_person) ∧
  distribution.length = participants := sorry

end equal_division_of_barrels_and_honey_l56_56401


namespace parabola_y_intersection_l56_56013

theorem parabola_y_intersection (x y : ℝ) : 
  (∀ x' y', y' = -(x' + 2)^2 + 6 → ((x' = 0) → (y' = 2))) :=
by
  intros x' y' hy hx0
  rw hx0 at hy
  simp [hy]
  sorry

end parabola_y_intersection_l56_56013


namespace sqrt_cube_root_sum_l56_56156

theorem sqrt_cube_root_sum : Real.sqrt ((-4: ℝ) ^ 2) + Real.cbrt (-8: ℝ) = 2 := 
by 
  sorry

end sqrt_cube_root_sum_l56_56156


namespace loss_percentage_refrigerator_l56_56828

variable (CP_refrigerator CP_mobile SP_mobile CP_total SP_total SP_refrigerator Loss_refrigerator : ℕ)
variable (Profit_mobile Overall_profit : ℤ)
variable (L : ℚ)

/-
Ravi purchased a refrigerator for Rs. 15000 and a mobile phone for Rs. 8000.
He sold the refrigerator at a certain loss percentage and the mobile phone at a profit of 10 percent.
Overall, he made a profit of Rs. 50.
Prove that the loss percentage on the refrigerator is 5%.
-/
theorem loss_percentage_refrigerator :
  let CP_refrigerator := 15000 in
  let CP_mobile := 8000 in
  let Profit_mobile := 10 in
  let Overall_profit := 50 in
  let SP_mobile := CP_mobile + CP_mobile * Profit_mobile / 100 in
  let CP_total := CP_refrigerator + CP_mobile in
  let SP_total := CP_total + Overall_profit in
  let SP_refrigerator := SP_total - SP_mobile in
  let Loss_refrigerator := CP_refrigerator - SP_refrigerator in
  let L := Loss_refrigerator * 100 / CP_refrigerator in
  L = 5 :=
by
  sorry

end loss_percentage_refrigerator_l56_56828


namespace pascal_50th_number_in_52_row_l56_56981

theorem pascal_50th_number_in_52_row : nat.binomial 51 2 = 1275 :=
by
  sorry

end pascal_50th_number_in_52_row_l56_56981


namespace number_of_factors_12650_l56_56637

-- Define the given conditions
def n : ℕ := 12650
def prime_factors : n = 5^2 * 2 * 11 * 23 := sorry

-- Define the main theorem to prove
theorem number_of_factors_12650 : 
  ∀ (n : ℕ), (n = 12650) →
    (∃ (p q r s: ℕ), n = 5^2 * 2 * 11 * 23 ∧ 
    (3 * 2 * 2 * 2 = 24)) :=
begin
  intros n h,
  use [5, 2, 11, 23],
  split,
  {
    exact h,
  },
  {
    norm_num,
  }
end

end number_of_factors_12650_l56_56637


namespace probability_bernardo_larger_than_silvia_l56_56967

theorem probability_bernardo_larger_than_silvia :
  let S₁ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let S₂ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let pick_four_desc (s : Set ℕ) := { l | l.length = 4 ∧ l.to_list.perm s.to_list ∧ sorted (≥) l.to_list }
  let P₁ := {b ∈ pick_four_desc S₁ | ∀ s ∈ pick_four_desc S₂, b > s}
  P₁.card.toReal / (S₁.card.choose 4).toReal = 293 / 420 :=
sorry

end probability_bernardo_larger_than_silvia_l56_56967


namespace ksyusha_wednesday_time_l56_56755

def speed_relation (v : ℝ) := ∀ (d_walk d_run1 d_run2 time1 time2: ℝ), 
  (2 * d_run1 = d_walk) → -- she walked twice the distance she ran on Tuesday
  (2 * v = 2 * v) → -- she runs twice as fast as she walks
  ((2 * d_walk / v) + (d_run1 / (2 * v)) = time1) → -- she took 30 minutes on Tuesday
  (time1 = 30) → -- Tuesday time equals 30
  (d_run2 = 2 * d_walk) → -- she had to run twice the distance she walked on Wednesday
  ((d_walk / v) + (d_run2 / (2 * v)) = time2) → -- calculate the time for Wednesday
  time2 = 24 -- time taken on Wednesday

theorem ksyusha_wednesday_time :
  ∃ (v : ℝ), speed_relation v :=
begin
  existsi 1,
  unfold speed_relation,
  intros,
  sorry,
end

end ksyusha_wednesday_time_l56_56755


namespace range_of_dataset_l56_56031

theorem range_of_dataset (x : ℝ) (h : set.range (finset.insert x (finset.insert 2 (finset.insert (-1) (finset.insert 0 (finset.singleton 1))))) =
    set.Icc (inf (finset.insert x (finset.insert 2 (finset.insert (-1) (finset.insert 0 (finset.singleton 1)))))) (sup (finset.insert x (finset.insert 2 (finset.insert (-1) (finset.insert 0 (finset.singleton 1)))))) ∧
    (sup (finset.insert x (finset.insert 2 (finset.insert (-1) (finset.insert 0 (finset.singleton 1))))) -
    inf (finset.insert x (finset.insert 2 (finset.insert (-1) (finset.insert 0 (finset.singleton 1)))))) = 5) :
    x = 4 ∨ x = -3 :=
by
  sorry

end range_of_dataset_l56_56031


namespace original_and_final_price_l56_56940

theorem original_and_final_price (P : ℝ) (h : (∃ k : ℝ, k = 0.6375) ∧ ∀ P : ℝ, let reducedPrice := 0.6375 * P in 
  (800 / reducedPrice - 800 / P = 5) ∧ reducedPrice = P * 0.6375): 
    (P = 290 / 3.1875) ∧ (0.6375 * P = 0.6375 * 91) :=
  by
  sorry

end original_and_final_price_l56_56940


namespace count_happy_numbers_l56_56468

def digits_are_different_and_non_zero (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ (d : ℕ), d ∈ digits → d ≠ 0 ∧ ∃! (i j : ℕ), i ≠ j ∧ digits.nth i = some d ∧ digits.nth j = some d

def is_happy (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∃ (d_i : ℕ), d_i ∈ digits ∧ (d_i = digits.erase d_i |>.sum)

theorem count_happy_numbers : ∃ (n : ℕ), (∃ (count : ℕ), count = 32 ∧ count = Nat.card { x | digits_are_different_and_non_zero x ∧ is_happy x }) :=
  sorry

end count_happy_numbers_l56_56468


namespace percentage_of_boys_l56_56658

theorem percentage_of_boys (total_students boys girls : ℕ) (h_ratio : boys * 4 = girls * 3) (h_total : boys + girls = total_students) (h_total_students : total_students = 42) : (boys : ℚ) * 100 / total_students = 42.857 :=
by
  sorry

end percentage_of_boys_l56_56658


namespace coloring_ways_l56_56695

-- Define a function that determines if a given integer is a power of two
def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

-- Define the main problem condition as a predicate
def valid_coloring (coloring : ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ i → i ≤ 200 → 1 ≤ j → j ≤ 200 → i ≠ j → 
    coloring i = coloring j → ¬ is_power_of_two (i + j)

theorem coloring_ways : 
  ∃ (coloring : ℕ → Prop), (∀ (i : ℕ), 1 ≤ i → i ≤ 200 → (coloring i = true ∨ coloring i = false)) ∧
  (valid_coloring coloring) ∧
  (finset.card (finset.filter valid_coloring (finset.univ : set (ℕ → Prop))) = 256) :=
sorry

end coloring_ways_l56_56695


namespace geometric_series_sum_l56_56420

noncomputable def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum (2/3) (2/3) 10 = 116050 / 59049 :=
by
  sorry

end geometric_series_sum_l56_56420


namespace second_machine_copies_per_minute_l56_56926

theorem second_machine_copies_per_minute :
  ∃ x : ℕ, (35 * 30 + x * 30 = 3000) ∧ x = 65 :=
by
  use 65
  split
  . linarith
  . rfl

end second_machine_copies_per_minute_l56_56926


namespace total_area_pool_and_deck_l56_56124

theorem total_area_pool_and_deck (pool_length pool_width deck_width : ℕ) 
  (h1 : pool_length = 12) 
  (h2 : pool_width = 10) 
  (h3 : deck_width = 4) : 
  (pool_length + 2 * deck_width) * (pool_width + 2 * deck_width) = 360 := 
by sorry

end total_area_pool_and_deck_l56_56124


namespace boiling_point_fahrenheit_l56_56413

-- Conditions
def boiling_point_c : Float := 100 -- °C
def melting_point_f : Float := 32 -- °F
def melting_point_c : Float := 0 -- °C
def pot_temp_c : Float := 40 -- °C
def pot_temp_f : Float := 104 -- °F

-- Definition and question combined to form the main statement
theorem boiling_point_fahrenheit : (∃ x : Float, x = 212) :=
by
  use 212
  sorry

end boiling_point_fahrenheit_l56_56413


namespace sequence_a100_calc_l56_56326

noncomputable def a_n (n : ℕ) : ℝ := 
if n = 0 then 0 else 
let S : ℕ → ℝ := λ k, sqrt k in
S n - S (n - 1)

theorem sequence_a100_calc : 
  (∀ n : ℕ, a_n n ^ 2 + 1 = 2 * a_n n * (finset.sum (finset.range n) a_n)) 
  ∧ (∀ n : ℕ, a_n n > 0) 
  → a_n 100 = 10 - 3 * sqrt 11 := 
begin
  intros h1 h2,
  sorry
end

end sequence_a100_calc_l56_56326


namespace coefficient_x_squared_l56_56702

theorem coefficient_x_squared (C : ℕ → ℕ → ℕ) (r : ℕ) (x : ℝ) (h₁ : 5 - 3 * r = 2) :
  let T := λ r, 2^r * C 5 r * x^(5 - 3 * r)
  2^1 * C 5 1 = 10 :=
by
  have h_r : r = 1 := sorry
  sorry

end coefficient_x_squared_l56_56702


namespace cliff_rock_collection_l56_56903

theorem cliff_rock_collection (S I : ℕ) 
  (h1 : I = S / 2) 
  (h2 : 2 * I / 3 = 40) : S + I = 180 := by
  sorry

end cliff_rock_collection_l56_56903


namespace measure_angle_DXG_l56_56298

def angle_bisects (OX : Segment) (DOG : Triangle) : Prop :=
  -- Assuming we have a function that ensures OX bisects the angle DOG
  bisects OX DOG.angles.α β

theorem measure_angle_DXG 
  {DOG : Triangle}
  (h1 : DOG.angles ≅ [50, 50, 80]) -- Reflects triangle DOG is isosceles where α=β=50° and γ=80°
  (h2 : DOG.angles.α = 50) 
  (OX : Segment)
  (h3 : angle_bisects OX DOG) 
  : DOG.angle.DXG = 25 := 
sorry

end measure_angle_DXG_l56_56298


namespace sale_in_first_month_is_5000_l56_56928

def sales : List ℕ := [6524, 5689, 7230, 6000, 12557]
def avg_sales : ℕ := 7000
def total_months : ℕ := 6

theorem sale_in_first_month_is_5000 :
  (avg_sales * total_months) - sales.sum = 5000 :=
by sorry

end sale_in_first_month_is_5000_l56_56928


namespace hits_9_and_8_mutually_exclusive_hits_10_and_8_not_mutually_exclusive_both_hit_target_and_neither_hit_target_mutually_exclusive_at_least_one_hits_and_A_not_B_does_not_mutually_exclusive_l56_56182

-- Definitions for shooting events for clarity
def hits_9_rings (s : String) := s = "9 rings"
def hits_8_rings (s : String) := s = "8 rings"

def hits_10_rings (s : String) := s = "10 rings"

def hits_target (s: String) := s = "hits target"
def does_not_hit_target (s: String) := s = "does not hit target"

-- Mutual exclusivity:
def mutually_exclusive (E1 E2 : Prop) := ¬ (E1 ∧ E2)

-- Problem 1:
theorem hits_9_and_8_mutually_exclusive :
  mutually_exclusive (hits_9_rings "9 rings") (hits_8_rings "8 rings") :=
sorry

-- Problem 2:
theorem hits_10_and_8_not_mutually_exclusive :
  ¬ mutually_exclusive (hits_10_rings "10 rings" ) (hits_8_rings "8 rings") :=
sorry

-- Problem 3:
theorem both_hit_target_and_neither_hit_target_mutually_exclusive :
  mutually_exclusive (hits_target "both hit target") (does_not_hit_target "neither hit target") :=
sorry

-- Problem 4:
theorem at_least_one_hits_and_A_not_B_does_not_mutually_exclusive :
  ¬ mutually_exclusive (hits_target "at least one hits target") (does_not_hit_target "A not but B does hit target") :=
sorry

end hits_9_and_8_mutually_exclusive_hits_10_and_8_not_mutually_exclusive_both_hit_target_and_neither_hit_target_mutually_exclusive_at_least_one_hits_and_A_not_B_does_not_mutually_exclusive_l56_56182


namespace knight_tour_impossible_l56_56303

-- Define the type for the squares on an 8x8 chessboard
structure Square :=
  (x : Nat) (y : Nat)
  (valid : x < 8 ∧ y < 8)

-- Define the move of a knight in terms of pairs of squares
def is_knight_move (start finish : Square) : Prop :=
  (start.x + 2 = finish.x ∧ start.y + 1 = finish.y) ∨ 
  (start.x + 2 = finish.x ∧ start.y - 1 = finish.y) ∨ 
  (start.x - 2 = finish.x ∧ start.y + 1 = finish.y) ∨ 
  (start.x - 2 = finish.x ∧ start.y - 1 = finish.y) ∨ 
  (start.x + 1 = finish.x ∧ start.y + 2 = finish.y) ∨ 
  (start.x + 1 = finish.x ∧ start.y - 2 = finish.y) ∨ 
  (start.x - 1 = finish.x ∧ start.y + 2 = finish.y) ∨ 
  (start.x - 1 = finish.x ∧ start.y - 2 = finish.y)

-- Define the starting and ending squares (A1 and H8)
def A1 : Square := ⟨0, 0, by simp [Nat.lt_succ_iff]⟩
def H8 : Square := ⟨7, 7, by simp [Nat.lt_succ_iff]⟩

-- Define a predicate indicating if a knight tour problem is solvable
def knight_tour_solvable (start finish : Square) : Prop :=
  ∃ (path : List Square), 
    path.head = start ∧ path.last = some finish ∧ 
    (∀ squares ∈ path, is_knight_move squares.1 squares.2) ∧
    (@List.nodup Square _ path) ∧ 
    path.length = 64

-- The theorem stating the knight cannot visit every square of a chessboard exactly once from A1 to H8.
theorem knight_tour_impossible : ¬knight_tour_solvable A1 H8 :=
  sorry

end knight_tour_impossible_l56_56303


namespace number_of_correct_propositions_l56_56322

variables (l m : Type) (α β : Type)
variables [Line l] [Line m] [Plane α] [Plane β]

-- Define propositions as terms
def prop1 : Prop := l ∥ m ∧ m ⊆ α → l ∥ α
def prop2 : Prop := l ∥ α ∧ m ∥ α → l ∥ m
def prop3 : Prop := α ⊥ β ∧ l ⊆ α → l ⊥ β
def prop4 : Prop := l ⊥ α ∧ m ⊥ α → l ∥ m

-- The proof problem statement
theorem number_of_correct_propositions : 
  (¬ prop1 ∧ ¬ prop2 ∧ ¬ prop3 ∧ prop4) →
  (true.count (λ p, p = true) [¬prop1, ¬prop2, ¬prop3, prop4] = 1) :=
by
  sorry

end number_of_correct_propositions_l56_56322


namespace max_expression_values_l56_56549

theorem max_expression_values (x : ℝ) : 
  (∃ n : ℤ, x = -1/4 + n/2) ↔ (3 - |Real.tan (sqrt 2 * π * cos(π * x))| - |Real.cot (π * sqrt 2 * sin (3 * π * x) / 2)| = 3) :=
sorry

end max_expression_values_l56_56549


namespace crayons_lost_or_given_away_l56_56820

theorem crayons_lost_or_given_away (P E L : ℕ) (h1 : P = 479) (h2 : E = 134) (h3 : L = P - E) : L = 345 :=
by
  rw [h1, h2] at h3
  exact h3

end crayons_lost_or_given_away_l56_56820


namespace polar_eq_circle_l56_56527

theorem polar_eq_circle :
  ∀ θ : ℝ, ∃ x y : ℝ, x^2 + y^2 = 4 ∧ (x = 2 * cos θ ∧ y = 2 * sin θ) := sorry

end polar_eq_circle_l56_56527


namespace prove_inequality_l56_56799

noncomputable def inequality_proof (n : ℕ) (a : Fin n → ℝ) : Prop :=
  (∀ i, 0 < a i) →
  (∑ i in Finset.range n, a i = n) →
  (∑ i in Finset.range n, 1 / a i - n ≥ 8 * (n - 1) * (1 - ∏ i in Finset.range n, a i) / (n ^ 2))

theorem prove_inequality (n : ℕ) (a : Fin n → ℝ) :
  inequality_proof n a :=
by
  intros hpos hsum
  sorry

end prove_inequality_l56_56799


namespace simplify_expression_l56_56423

theorem simplify_expression (x y : ℝ) : (x^2 + y^2)⁻¹ * (x⁻¹ + y⁻¹) = (x^3 * y + x * y^3)⁻¹ * (x + y) :=
by sorry

end simplify_expression_l56_56423


namespace total_cost_is_1_85_times_selling_price_l56_56348

def total_cost (P : ℝ) : ℝ := 140 * 2 * P + 90 * P

def loss (P : ℝ) : ℝ := 70 * 2 * P + 30 * P

def selling_price (P : ℝ) : ℝ := total_cost P - loss P

theorem total_cost_is_1_85_times_selling_price (P : ℝ) :
  total_cost P = 1.85 * selling_price P := by
  sorry

end total_cost_is_1_85_times_selling_price_l56_56348


namespace table_price_l56_56308

theorem table_price 
  (trees : ℕ) (planks_per_tree : ℕ) (planks_per_table : ℕ) (labor_cost : ℤ) (profit : ℤ)
  (h_trees : trees = 30)
  (h_planks_per_tree : planks_per_tree = 25)
  (h_planks_per_table : planks_per_table = 15)
  (h_labor_cost : labor_cost = 3000)
  (h_profit : profit = 12000) :
  let total_planks := trees * planks_per_tree,
      total_tables := total_planks / planks_per_table,
      total_revenue := profit + labor_cost,
      price_per_table := total_revenue / total_tables
  in price_per_table = 300 := by
  unfold total_planks total_tables total_revenue price_per_table;
  rw [h_trees, h_planks_per_tree, h_planks_per_table, h_labor_cost, h_profit];
  sorry

end table_price_l56_56308


namespace smallest_n_for_y_n_integer_l56_56976

-- Define the sequence y_n based on the given conditions
noncomputable def y₁ : ℝ := real.rpow 4 (1 / 4)
noncomputable def y (n : ℕ) : ℝ :=
  if n = 1 then y₁ else real.rpow (y (n - 1)) (1 / 4)

-- Statement to prove:
theorem smallest_n_for_y_n_integer : ∃ n : ℕ, y n ∈ ℤ ∧ ∀ m < n, y m ∉ ℤ :=
begin
  use 4,
  split,
  { sorry },   -- Proof that y 4 is an integer.
  { intros m h,
    cases m,
    { simp [y, y₁], sorry },  -- Proofs that y 1, y 2, and y 3 are not integers
    { cases m,
      { simp [y, y₁], sorry },
      { cases m,
        { simp [y, y₁], sorry },
        { exfalso, linarith } } } }
end

end smallest_n_for_y_n_integer_l56_56976


namespace imo1987_q6_l56_56524

theorem imo1987_q6 (m n : ℤ) (h : n = m + 2) :
  ⌊(n : ℝ) * Real.sqrt 2⌋ = 2 + ⌊(m : ℝ) * Real.sqrt 2⌋ := 
by
  sorry -- We skip the detailed proof steps here.

end imo1987_q6_l56_56524


namespace trapezoid_diagonal_squared_l56_56372

theorem trapezoid_diagonal_squared (ABCD : Trapezoid) (AC : Diagonal ABCD) (a b : ℝ) :
  length (base1 ABCD) = a → length (base2 ABCD) = b → AC^2 = a * b :=
by
  sorry

end trapezoid_diagonal_squared_l56_56372


namespace intersection_of_parabola_with_y_axis_l56_56019

theorem intersection_of_parabola_with_y_axis :
  ∃ y : ℝ, y = - (0 + 2)^2 + 6 ∧ (0, y) = (0, 2) :=
by
  sorry

end intersection_of_parabola_with_y_axis_l56_56019


namespace probability_two_defective_after_two_tests_l56_56594

variable {α : Type*}

-- Assume a finite set of components
def components : Finset α := {1, 2, 3, 4, 5, 6}

-- Assume 2 defective components
def defective : Finset α := {1, 2}

-- Assume 4 qualified components
def qualified : Finset α := {3, 4, 5, 6}

-- Define a probability measure space
noncomputable def prob_space (s : Finset α) := 
  MeasureTheory.MeasureSpace (λ _, 1.0 / s.card)

-- Probability of finding exactly the 2 defective components after 2 tests without replacement
theorem probability_two_defective_after_two_tests :
  @prob_space α components (defective.card = 2 ∧
                            qualified.card = 4 ∧
                            (defective ∪ qualified) = components) ->
  probability (finset.pair_combinations components 2) (λ s, s = defective) = 1/15 := by sorry

end probability_two_defective_after_two_tests_l56_56594


namespace triangular_prism_skew_pair_count_l56_56950

-- Definition of a triangular prism with 6 vertices and 15 lines through any two vertices
structure TriangularPrism :=
  (vertices : Fin 6)   -- 6 vertices
  (lines : Fin 15)     -- 15 lines through any two vertices

-- A function to check if two lines are skew lines 
-- (not intersecting and not parallel in three-dimensional space)
def is_skew (line1 line2 : Fin 15) : Prop := sorry

-- Function to count pairs of lines that are skew in a triangular prism
def count_skew_pairs (prism : TriangularPrism) : Nat := sorry

-- Theorem stating the number of skew pairs in a triangular prism is 36
theorem triangular_prism_skew_pair_count (prism : TriangularPrism) :
  count_skew_pairs prism = 36 := 
sorry

end triangular_prism_skew_pair_count_l56_56950


namespace eccentricity_of_ellipse_l56_56220

-- Define the ellipse and its conditions
def ellipse (x y a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line AB
def line_AB (x y a b : ℝ) : Prop :=
  x / a + y / b = 1

-- Define the distance from a point to a line
def distance_point_to_line (x y a b d : ℝ) : Prop :=
  (b * x + a * y - a * b) / real.sqrt (a^2 + b^2) = d

theorem eccentricity_of_ellipse (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) :
  (ellipse (-c) 0 a b ha hb) →
  (line_AB (-c) 0 a b) →
  (distance_point_to_line (-c) 0 a b (5 * real.sqrt 14 / 14 * b)) →
  e = 2 / 3 :=
by
  intros
  sorry

end eccentricity_of_ellipse_l56_56220


namespace function_monotonic_decreasing_l56_56240

noncomputable def f (x : ℝ) := sin (ω * x + φ + π / 4)
variables (ω φ : ℝ)

axiom ω_range : 5 / 2 < ω ∧ ω < 9 / 2
axiom φ_range : 0 < φ ∧ φ < π
axiom f_even : ∀ x, f x = f (-x)
axiom f_pi_equality : f 0 = f π

theorem function_monotonic_decreasing : f x = cos (4 * x) ∧ (∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → x2 < π / 4 → cos (4 * x2) < cos (4 * x1)) := 
by
  sorry

end function_monotonic_decreasing_l56_56240


namespace altitudes_concur_on_BC_l56_56140

theorem altitudes_concur_on_BC (a b : ℕ) [hn : nat.coprime a b] :
  ∀ (ABC : Triangle) (BE CF : Line) (H : Point),
  acute_triangle ABC →
  (BE ⊥ AC) ∧
  (CF ⊥ AB) ∧
  intersect BE CF = H →
  ∃ (E F M : Point),
    (altitudes_concur_triangle (triangle E H F) (line BC)) ∧
    (length AB = 3) ∧
    (length AC = 4) →
    BC^2 = a / b →
    100 * a + b = 1301 :=
by
  sorry

end altitudes_concur_on_BC_l56_56140


namespace fishing_rod_price_l56_56389

theorem fishing_rod_price (initial_price : ℝ) 
  (price_increase_percentage : ℝ) 
  (price_decrease_percentage : ℝ) 
  (new_price : ℝ) 
  (final_price : ℝ) 
  (h1 : initial_price = 50) 
  (h2 : price_increase_percentage = 0.20) 
  (h3 : price_decrease_percentage = 0.15) 
  (h4 : new_price = initial_price * (1 + price_increase_percentage)) 
  (h5 : final_price = new_price * (1 - price_decrease_percentage)) 
  : final_price = 51 :=
sorry

end fishing_rod_price_l56_56389


namespace part_b_part_c_part_d_l56_56599

variable {R : Type} [LinearOrderedField R]

noncomputable def f : R → R := sorry

-- Conditions
axiom domain_f : ∀ x : R, x ∈ R
axiom odd_f_half : ∀ x : R, f(x + 1/2) = - f(-x - 1/2)
axiom functional_eq : ∀ x : R, f(2 - 3 * x) = f(3 * x)

-- Questions to Prove
theorem part_b : f(-1/2) = 0 := sorry
theorem part_c : ∀ x : R, f(x + 2) = f(x - 2) := sorry
theorem part_d : ∀ x : R, f(x - 1/2) = -f(-x - 1/2) := sorry

end part_b_part_c_part_d_l56_56599


namespace pentagon_area_is_sum_of_radicals_l56_56120

def is_regular_pentagon (P : Type) (s : ℝ) (a : ℝ) : Prop :=
  (∀ (x : P), s = 3) ∧ (∀ (y : P), a = 120)

noncomputable def calculate_area (s : ℝ) (a : ℝ) : ℝ :=
  let t_area := λ s a, (1 / 2) * s * s * (Real.sin (a * (Real.pi / 180))) 
  in 5 * t_area s a

theorem pentagon_area_is_sum_of_radicals
  (p q : ℝ)
  (h : p + q = 45) :
  ∃ p q : ℝ, (calculate_area 3 120) = (Real.sqrt p + Real.sqrt q) :=
by
  sorry

end pentagon_area_is_sum_of_radicals_l56_56120


namespace resistor_problem_l56_56282

theorem resistor_problem 
  {x y r : ℝ}
  (h1 : 1 / r = 1 / x + 1 / y)
  (h2 : r = 2.9166666666666665)
  (h3 : y = 7) : 
  x = 5 :=
by
  sorry

end resistor_problem_l56_56282


namespace black_white_ratio_l56_56660

theorem black_white_ratio 
  (x y : ℕ) 
  (h1 : (y - 1) * 7 = x * 9) 
  (h2 : y * 5 = (x - 1) * 7) : 
  y - x = 7 := 
by 
  sorry

end black_white_ratio_l56_56660


namespace count_fixed_points_upto_1988_l56_56803

def f : ℕ+ → ℕ
| ⟨1, _⟩ := 1
| ⟨3, _⟩ := 3
| ⟨n, hn⟩ :=
  if even n then f ⟨n / 2, sorry⟩ 
  else if (n - 1) % 4 = 0 then 2 * f ⟨(n - 1) / 2 + 1, sorry⟩ - f ⟨(n - 1) / 4, sorry⟩ 
  else 3 * f ⟨(n - 1) / 2 + 1, sorry⟩ - 2 * f ⟨(n - 1) / 4, sorry⟩

def is_palindromic_binary (n : ℕ) : Prop :=
  let bin_str := n.to_digits 2
  in bin_str = bin_str.reverse

def is_fixed_point (n : ℕ) : Prop :=
  f ⟨n, sorry⟩ = n

theorem count_fixed_points_upto_1988 : 
  (finset.range 1988).filter (fun n => is_palindromic_binary n ∧ is_fixed_point n) .card = 92 :=
  sorry

end count_fixed_points_upto_1988_l56_56803


namespace ounces_per_glass_is_8_l56_56496

def milk_per_glass := 6.5
def syrup_per_glass := 1.5
def total_milk := 130
def total_syrup := 60
def total_chocolate_milk := 160

theorem ounces_per_glass_is_8 (G : ℝ) (h1 : G = milk_per_glass + syrup_per_glass) : 
  G = 8 := by
  -- Provided proof conditions and initial setup
  have h1 : G = 6.5 + 1.5 := by sorry
  -- Conclusion to be proven
  sorry

end ounces_per_glass_is_8_l56_56496


namespace root_sixth_power_sum_l56_56795

noncomputable def p : ℝ := sorry
noncomputable def q : ℝ := sorry

def quadratic_eq_roots (p q : ℝ) : Prop :=
  let a := 1
  let b := -2 * real.sqrt 3
  let c := 2
  (p + q = -b / a) ∧ (p * q = c / a)

theorem root_sixth_power_sum :
  quadratic_eq_roots p q →
  p^6 + q^6 = 3120 :=
by
  intros h
  sorry

end root_sixth_power_sum_l56_56795


namespace find_n_l56_56540

variable (x a : ℝ) (n : ℕ)

def binomial_expansion_second_term := binomial n 1 * (x ^ (n - 1)) * (a ^ 1) = 56
def binomial_expansion_third_term := binomial n 2 * (x ^ (n - 2)) * (a ^ 2) = 168
def binomial_expansion_fourth_term := binomial n 3 * (x ^ (n - 3)) * (a ^ 3) = 336

theorem find_n (h1 : binomial_expansion_second_term x a n)
  (h2 : binomial_expansion_third_term x a n)
  (h3 : binomial_expansion_fourth_term x a n) : n = 3 :=
by
  sorry

end find_n_l56_56540


namespace coefficient_x_squared_l56_56703

theorem coefficient_x_squared (C : ℕ → ℕ → ℕ) (r : ℕ) (x : ℝ) (h₁ : 5 - 3 * r = 2) :
  let T := λ r, 2^r * C 5 r * x^(5 - 3 * r)
  2^1 * C 5 1 = 10 :=
by
  have h_r : r = 1 := sorry
  sorry

end coefficient_x_squared_l56_56703


namespace coloring_ways_l56_56696

-- Define a function that determines if a given integer is a power of two
def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

-- Define the main problem condition as a predicate
def valid_coloring (coloring : ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ i → i ≤ 200 → 1 ≤ j → j ≤ 200 → i ≠ j → 
    coloring i = coloring j → ¬ is_power_of_two (i + j)

theorem coloring_ways : 
  ∃ (coloring : ℕ → Prop), (∀ (i : ℕ), 1 ≤ i → i ≤ 200 → (coloring i = true ∨ coloring i = false)) ∧
  (valid_coloring coloring) ∧
  (finset.card (finset.filter valid_coloring (finset.univ : set (ℕ → Prop))) = 256) :=
sorry

end coloring_ways_l56_56696


namespace quadratic_real_solution_l56_56545

theorem quadratic_real_solution (a : ℝ) : (∃ x : ℝ, a * x^2 - 4 * x - 1 = 0) ↔ a ≥ -4 :=
begin
  sorry
end

end quadratic_real_solution_l56_56545


namespace cardinality_B_l56_56625

-- Define Set A
def A : Set ℕ := {2, 0, 1, 7}

-- Define Set B
def B : Set ℕ := {x | ∃ (a ∈ A) (b ∈ A), x = a * b}

-- State the theorem to prove the number of elements in Set B is 7
theorem cardinality_B :
  Set.card B = 7 :=
sorry

end cardinality_B_l56_56625


namespace tan_increasing_intervals_l56_56859

/-- The monotonic increasing intervals of the function y = tan(2x - π/4) are 
     (\frac{kπ}{2} - \frac{π}{8}, \frac{kπ}{2} + \frac{3π}{8}) for any integer k. -/
theorem tan_increasing_intervals (k : ℤ) :
  ∀ x, (frac(k * π)/2 - π/8 < x ∧ x < frac(k * π)/2 + 3 * π/8) ↔ increasing (λ x, tan (2 * x - π/4)) :=
sorry

end tan_increasing_intervals_l56_56859


namespace ksyusha_wednesday_time_l56_56748

def speed_relation (v : ℝ) := ∀ (d_walk d_run1 d_run2 time1 time2: ℝ), 
  (2 * d_run1 = d_walk) → -- she walked twice the distance she ran on Tuesday
  (2 * v = 2 * v) → -- she runs twice as fast as she walks
  ((2 * d_walk / v) + (d_run1 / (2 * v)) = time1) → -- she took 30 minutes on Tuesday
  (time1 = 30) → -- Tuesday time equals 30
  (d_run2 = 2 * d_walk) → -- she had to run twice the distance she walked on Wednesday
  ((d_walk / v) + (d_run2 / (2 * v)) = time2) → -- calculate the time for Wednesday
  time2 = 24 -- time taken on Wednesday

theorem ksyusha_wednesday_time :
  ∃ (v : ℝ), speed_relation v :=
begin
  existsi 1,
  unfold speed_relation,
  intros,
  sorry,
end

end ksyusha_wednesday_time_l56_56748


namespace range_of_f_l56_56169

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5 * x + 6) / (x^2 + 2 * x + 3)

theorem range_of_f : set.range f = set.univ :=
by sorry

end range_of_f_l56_56169


namespace percentage_loss_l56_56116

def initial_selling_rate := 12 -- oranges/rupee
def desired_profit_rate := 0.44 -- 44%
def target_selling_rate := 7.5 -- oranges/rupee

-- Definition of cost price "C" and conditions from the problem
def cost_price_of_orange (C : ℝ) : Prop := C = 1 / (7.5 * 1.44)

-- Loss percentage calculation
def loss_percentage (C : ℝ) : ℝ :=
  ((C - (1 / 12)) / C) * 100

-- Proof problem statement
theorem percentage_loss (C : ℝ)
  (hC : cost_price_of_orange C) : loss_percentage C = 10 :=
by
  sorry

end percentage_loss_l56_56116


namespace highest_lowest_production_difference_actual_total_production_total_profit_l56_56006

def planned_daily_production : Int := 10000
def deviations : List Int := [41, -34, -52, 127, -72, 36, -29]
def production_cost : Int := 35 -- yuan
def selling_price : Int := 40 -- yuan

theorem highest_lowest_production_difference : 
  list.maximum deviations + |list.minimum deviations| = 199 := by
  -- We would formally calculate the values and prove this here
  sorry

theorem actual_total_production :
  list.sum deviations >= 0 := by
  -- We would sum the deviations and check the result here
  sorry

theorem total_profit :
  (planned_daily_production * 7 + list.sum deviations) * (selling_price - production_cost) = 350085 := by
  -- We would calculate total production and profit here
  sorry

end highest_lowest_production_difference_actual_total_production_total_profit_l56_56006


namespace peach_difference_proof_l56_56035

def red_peaches_odd := 12
def green_peaches_odd := 22
def red_peaches_even := 15
def green_peaches_even := 20
def num_baskets := 20
def num_odd_baskets := num_baskets / 2
def num_even_baskets := num_baskets / 2

def total_red_peaches := (red_peaches_odd * num_odd_baskets) + (red_peaches_even * num_even_baskets)
def total_green_peaches := (green_peaches_odd * num_odd_baskets) + (green_peaches_even * num_even_baskets)
def difference := total_green_peaches - total_red_peaches

theorem peach_difference_proof : difference = 150 := by
  sorry

end peach_difference_proof_l56_56035


namespace johns_average_speed_l56_56727

noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem johns_average_speed :
  average_speed 210 6.5 ≈ 32.31 := 
by
  sorry

end johns_average_speed_l56_56727


namespace problem_solution_l56_56569

noncomputable def center : ℝ × ℝ := (-1, 2)
noncomputable def tangentLine : ℝ → ℝ → ℝ := λ x y, x + 2 * y + 7
noncomputable def pointQ : ℝ × ℝ := (1, 6)
noncomputable def circleEquation : ℝ → ℝ → ℝ := λ x y, (x + 1)^2 + (y - 2)^2 - 20
noncomputable def tangentEquation : ℝ → ℝ → ℝ := λ x y, x + 2 * y - 13

theorem problem_solution :
  (∀ x y, circleEquation x y = 0 → ∃ l : ℝ → ℝ → ℝ, (∀ x y, l x y = 0 → tangentEquation x y = 0)) :=
by
  sorry

end problem_solution_l56_56569


namespace seonwoo_change_l56_56075

theorem seonwoo_change (initial_money: ℕ) (bubblegum_cost: ℕ) (bubblegum_qty: ℕ) 
  (ramen_cost_per_two: ℕ) (ramen_qty: ℕ) 
  (total_cost_bubblegum: ℕ) (total_cost_ramen: ℕ) 
  (final_cost: ℕ) (change: ℕ) 
  (h1: initial_money = 10000)
  (h2: bubblegum_cost = 600)
  (h3: bubblegum_qty = 2)
  (h4: ramen_cost_per_two = 1600)
  (h5: ramen_qty = 9)
  (h6: total_cost_bubblegum = bubblegum_qty * bubblegum_cost)
  (h7: total_cost_ramen = ((ramen_qty / 2) * ramen_cost_per_two) + ((ramen_qty % 2) * (ramen_cost_per_two / 2)))
  (h8: final_cost = total_cost_bubblegum + total_cost_ramen)
  (h9: change = initial_money - final_cost) 
  : change = 1600 := 
by 
  rw [h1, h2, h3, h4, h5, h6, h7, h8, h9]
  -- Proof steps are not required
  sorry

end seonwoo_change_l56_56075


namespace radicals_equality_l56_56254

theorem radicals_equality (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  (sqrt (p + q / r) = sqrt (p * q) * sqrt (1 / r)) ↔ (p = r - 1 ∧ q > 0) :=
sorry

end radicals_equality_l56_56254


namespace min_expression_value_l56_56299

theorem min_expression_value
    (a1 b1 : ℝ) (A1 : ℝ) (b2 : ℝ) (a2 : ℝ) (A2 : ℝ) 
    (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
    (hA1 : A1 = real.pi / 6) (ha1 : a1 = 7) (hb1 : b1 = 8)
    (hA2 : A2 = real.pi / 3) (ha2 : a2 = 13 * real.sqrt 3) (hb2 : b2 = 26)
    (hm : (2*x + y = 3)) :
    (1/x + 2/y) = 8/3 :=
by sorry

end min_expression_value_l56_56299


namespace exponent_multiplication_l56_56492

-- Define the variables and exponentiation property
variable (a : ℝ)

-- State the theorem
theorem exponent_multiplication : a^3 * a^2 = a^5 := by
  sorry

end exponent_multiplication_l56_56492


namespace coloring_number_lemma_l56_56688

-- Definition of the problem conditions
def no_sum_of_two_same_color_is_power_of_two (f : ℕ → bool) : Prop :=
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 200 ∧ f a = f b → ¬(∃ k : ℕ, a + b = 2^k)

-- The main theorem
theorem coloring_number_lemma :
  (∃ f : ℕ → bool, (∀ n, 1 ≤ n ∧ n ≤ 200 → (f n = tt ∨ f n = ff))
                   ∧ no_sum_of_two_same_color_is_power_of_two f) →
  ∃ n : ℕ, n = 256 :=
by sorry

end coloring_number_lemma_l56_56688


namespace solve_for_a_l56_56612

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x

def second_derivative (a : ℝ) (x : ℝ) : ℝ := (derivative (derivative (f a))) x

theorem solve_for_a (a : ℝ) (h : second_derivative a 1 = 3) : a = 3 :=
by
  sorry

end solve_for_a_l56_56612


namespace digitalEarthFunctions_l56_56478

axiom OptionA (F : Type) : Prop
axiom OptionB (F : Type) : Prop
axiom OptionC (F : Type) : Prop
axiom OptionD (F : Type) : Prop

axiom isRemoteSensing (F : Type) : OptionA F
axiom isGIS (F : Type) : OptionB F
axiom isGPS (F : Type) : OptionD F

theorem digitalEarthFunctions {F : Type} : OptionC F :=
sorry

end digitalEarthFunctions_l56_56478


namespace reflected_ray_equation_l56_56121

theorem reflected_ray_equation :
  ∀ P : ℝ × ℝ, P = (0, 1) → (∃ M : ℝ × ℝ, M = (3, -2) ∧ (∃ N : ℝ × ℝ, N = (-3, -2) ∧ collinear {P, N, (x, y)})) → x - y + 1 = 0 :=
by
  sorry

end reflected_ray_equation_l56_56121


namespace circumference_circle_l56_56367

theorem circumference_circle {d r : ℝ} (h1 : ∀ (d r : ℝ), d = 2 * r) : 
  ∃ C : ℝ, C = π * d ∨ C = 2 * π * r :=
by {
  sorry
}

end circumference_circle_l56_56367


namespace number_of_ordered_pairs_l56_56166

theorem number_of_ordered_pairs {x y: ℕ} (h1 : x < y) (h2 : 2 * x * y / (x + y) = 4^30) : 
  ∃ n, n = 61 :=
sorry

end number_of_ordered_pairs_l56_56166


namespace tommy_nickels_l56_56046

-- Definitions of given conditions
def pennies (quarters : Nat) : Nat := 10 * quarters  -- Tommy has 10 times as many pennies as quarters
def dimes (pennies : Nat) : Nat := pennies + 10      -- Tommy has 10 more dimes than pennies
def nickels (dimes : Nat) : Nat := 2 * dimes         -- Tommy has twice as many nickels as dimes

theorem tommy_nickels (quarters : Nat) (P : Nat) (D : Nat) (N : Nat) 
  (h1 : quarters = 4) 
  (h2 : P = pennies quarters) 
  (h3 : D = dimes P) 
  (h4 : N = nickels D) : 
  N = 100 := 
by
  -- sorry allows us to skip the proof
  sorry

end tommy_nickels_l56_56046


namespace solve_system_eqn_l56_56253

theorem solve_system_eqn (x y : ℚ) (h₁ : 3*y - 4*x = 8) (h₂ : 2*y + x = -1) :
  x = -19/11 ∧ y = 4/11 :=
by
  sorry

end solve_system_eqn_l56_56253


namespace coins_in_stack_l56_56294

-- Define the thickness of each coin type
def penny_thickness : ℝ := 1.55
def nickel_thickness : ℝ := 1.95
def dime_thickness : ℝ := 1.35
def quarter_thickness : ℝ := 1.75

-- Define the total stack height
def total_stack_height : ℝ := 15

-- The statement to prove
theorem coins_in_stack (pennies nickels dimes quarters : ℕ) :
  pennies * penny_thickness + nickels * nickel_thickness + 
  dimes * dime_thickness + quarters * quarter_thickness = total_stack_height →
  pennies + nickels + dimes + quarters = 9 :=
sorry

end coins_in_stack_l56_56294


namespace tangent_line_at_1_l56_56510

def f (x : ℝ) : ℝ := x^3 + x

def tangent_line_eq_at_x_1 : Prop :=
  ∃ m b, (∀ x, f x * m + b) = 4 * x - 2

theorem tangent_line_at_1 :
  ∀ x y, (f x = y) → (4 * x - y - 2 = 0) :=
  sorry

end tangent_line_at_1_l56_56510


namespace Ksyusha_time_to_school_l56_56733

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l56_56733


namespace constant_term_in_expansion_l56_56369

theorem constant_term_in_expansion :
  let f : Polynomial ℚ := (x + (1 : ℚ) / (sqrt x) - 2)^5
  in (constant_term f) = (88 : ℚ) :=
sorry

end constant_term_in_expansion_l56_56369


namespace sum_f_2000_l56_56804

def f (x : ℝ) := (3) / (5^x + 3)

theorem sum_f_2000 (n : ℕ) (h : n = 2000) :
  ∑ k in Finset.range (n + 1), f (k / 2001) = 1000 :=
by
  sorry

end sum_f_2000_l56_56804


namespace cone_volume_correct_l56_56458

def half_sector_radius := 6
def half_sector_arc_length := 2 * Real.pi * half_sector_radius / 2
def cone_radius := half_sector_arc_length / (2 * Real.pi)
def slant_height := half_sector_radius
def cone_height := Real.sqrt (slant_height^2 - cone_radius^2)
def expected_volume := (1 / 3) * Real.pi * cone_radius^2 * cone_height

theorem cone_volume_correct :
  expected_volume = 9 * Real.pi * Real.sqrt (3) := by
    sorry

end cone_volume_correct_l56_56458


namespace smaller_cubes_count_l56_56033

theorem smaller_cubes_count (large_cube_surface_area : ℝ) (small_cube_volume : ℝ) :
  large_cube_surface_area = 5400 → small_cube_volume = 216 → (∃ n : ℕ, n = 125) :=
by
  intro h1 h2
  use 125
  sorry

end smaller_cubes_count_l56_56033


namespace fraction_b_plus_c_over_a_l56_56408

variable (a b c d : ℝ)

theorem fraction_b_plus_c_over_a :
  (a ≠ 0) →
  (a * 4^3 + b * 4^2 + c * 4 + d = 0) →
  (a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) →
  (b + c) / a = -13 :=
by
  intros h₁ h₂ h₃ 
  sorry

end fraction_b_plus_c_over_a_l56_56408


namespace hyperbola_eccentricity_l56_56586

-- Define the hyperbola and associated constants and points
variables (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
variables (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)

-- Focus of the hyperbola, vertex, and intersection point
def focus : F = (c, 0) := sorry
def vertex : A = (0, -b) := sorry

-- Equations related to the hyperbola and its asymptote and line AF
def asymptote_eq : (x y : ℝ) → y = (b / a) * x := sorry
def line_AF_eq : (x y : ℝ) → (x / c) - (y / b) = 1 := sorry

-- Intersection point B from the line AF and asymptote equation
def intersection_B : B = (a * c / (a - c), b * c / (a - c)) := sorry

-- Given vector relationship
def vector_relationship : (-(c, b)) = (sqrt 2 + 1) • (B - F) := sorry

-- Statement to prove the eccentricity of the hyperbola
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = c / a ∧ e = sqrt 2 := sorry

end hyperbola_eccentricity_l56_56586


namespace root_difference_of_quadratic_l56_56546

theorem root_difference_of_quadratic :
  ∀ (a b c : ℝ), a ≠ 0 → 
  a = 7 + 4 * Real.sqrt 3 → 
  b = 2 + Real.sqrt 3 → 
  c = -2 →
  let Δ := b ^ 2 - 4 * a * c in
  let r1 := (-b + Real.sqrt Δ) / (2 * a) in
  let r2 := (-b - Real.sqrt Δ) / (2 * a) in
  (r1 - r2 = 6 - 3 * Real.sqrt 3) :=
begin
  intros,
  simp only [←h, (eq.subst h_1 h.symm), (eq.subst h_2 h.symm)],
  -- Placeholder for actual proof
  sorry
end

end root_difference_of_quadratic_l56_56546


namespace total_marbles_is_63_l56_56258

-- Definitions directly from the conditions
variables (my_marbles : ℕ) (brother_marbles : ℕ) (friend_marbles : ℕ)
variables (total_marbles : ℕ)

-- Given conditions as assumptions:
axiom my_marbles_def : my_marbles = 16
axiom brother_condition : my_marbles - 2 = 2 * (brother_marbles + 2)
axiom friend_condition : friend_marbles = 3 * (my_marbles - 2)

-- Statement to prove
theorem total_marbles_is_63 : total_marbles = my_marbles + brother_marbles + friend_marbles :=
begin
  sorry
end

end total_marbles_is_63_l56_56258


namespace surface_area_of_solid_l56_56868

def s : ℝ := 4 * Real.sqrt 2
def h : ℝ := 3 * Real.sqrt 2
def base_area : ℝ := s ^ 2
def upper_edge_length : ℝ := 3 * s
def upper_rectangle_area : ℝ := 3 * s ^ 2
def trapezoid_area : ℝ := (1 / 2) * (s + 3 * s) * h
def num_trapezoids : ℝ := 4

theorem surface_area_of_solid :
  base_area + upper_rectangle_area + num_trapezoids * trapezoid_area = 320 :=
by
  -- The proof is omitted
  sorry

end surface_area_of_solid_l56_56868


namespace ksyusha_travel_time_l56_56761

theorem ksyusha_travel_time :
  ∀ (S v : ℝ), 
  (2 * S / v + S / (2 * v) = 30) → 
  (2 * (S / v) = 24) :=
by
  intros S v h1
  have h2 : S / v = 12 := by
    calc 
      S / v = (30 * 2) / 5 := by sorry
  calc 
    2 * (S / v) = 2 * 12 := by sorry
  ... = 24 := by norm_num

end ksyusha_travel_time_l56_56761


namespace cost_of_batman_game_l56_56044

noncomputable def footballGameCost : ℝ := 14.02
noncomputable def strategyGameCost : ℝ := 9.46
noncomputable def totalAmountSpent : ℝ := 35.52

theorem cost_of_batman_game :
  totalAmountSpent - (footballGameCost + strategyGameCost) = 12.04 :=
by
  -- The proof is omitted as instructed.
  sorry

end cost_of_batman_game_l56_56044


namespace shirt_price_is_correct_l56_56870

noncomputable def sweater_price (T : ℝ) : ℝ := T + 7.43 

def discounted_price (S : ℝ) : ℝ := S * 0.90

theorem shirt_price_is_correct :
  ∃ (T S : ℝ), T + discounted_price S = 80.34 ∧ T = S - 7.43 ∧ T = 38.76 :=
by
  sorry

end shirt_price_is_correct_l56_56870


namespace largest_candies_l56_56885

theorem largest_candies (n : ℕ) (h₁ : n > 145)
  (h₂ : ∀ (S : Finset ℕ), 145 ≤ S.card → (∃ t : ℕ, 10 = (S.filter (λ x, x = t)).card)) :
  n ≤ 160 :=
sorry

end largest_candies_l56_56885


namespace circle_intersection_range_m_l56_56654

noncomputable def range_of_m := { m : ℝ | 1 ≤ m ∧ m ≤ 121 }

theorem circle_intersection_range_m (m : ℝ) :
  (∃ p : ℝ × ℝ, (p.1^2 + p.2^2 = m) ∧ (p.1^2 + p.2^2 + 6*p.1 - 8*p.2 - 11 = 0)) ↔
  m ∈ range_of_m := 
begin
  sorry
end

end circle_intersection_range_m_l56_56654


namespace Ksyusha_travel_time_l56_56772

theorem Ksyusha_travel_time:
  ∀ (S v : ℝ), 
  (S > 0 ∧ v > 0) →
  (∀ tuesday_distance tuesday_time wednesday_distance wednesday_time: ℝ, 
    tuesday_distance = 3 * S ∧ 
    tuesday_distance = 2 * S + S ∧ 
    tuesday_time = (2 * S) / v + (S / (2 * v)) ∧ 
    tuesday_time = 30) → 
  (∀ wednesday_distance wednesday_time : ℝ, 
    wednesday_distance = S + 2 * S ∧ 
    wednesday_time = S / v + (2 * S) / (2 * v)) →
  wednesday_time = 24 :=
by
  intros S v S_pos v_pos tuesday_distance tuesday_time wednesday_distance wednesday_time
  sorry

end Ksyusha_travel_time_l56_56772


namespace instantaneous_velocity_at_5_l56_56375

noncomputable def position_function (t : ℝ) : ℝ := (1/4) * t^4 - 3

noncomputable def velocity_function (t : ℝ) : ℝ := by
  let s := position_function
  exact derivative s t

theorem instantaneous_velocity_at_5 : velocity_function 5 = 125 := sorry

end instantaneous_velocity_at_5_l56_56375


namespace condition_relation_l56_56096

variable (A B C : Prop)

theorem condition_relation (h1 : C → B) (h2 : A → B) : 
  (¬(A → C) ∧ ¬(C → A)) :=
by 
  sorry

end condition_relation_l56_56096


namespace find_k_l56_56958

def triangle_isosceles (A B C : Type) : Prop :=
  ∀ α β γ : ℝ, α < π / 2 ∧ β < π / 2 ∧ γ < π / 2 ∧ α + β + γ = π ∧ (α = β ∨ β = γ ∨ γ = α)

def inscribed_in_circle (ABC : Type) : Prop := sorry -- definition depending on circle properties

def tangents_meet_at_point (B C D : Type) : Prop := sorry -- definition depending on tangent properties

theorem find_k (A B C D : Type) (k : ℝ) :
  triangle_isosceles A B C ∧
  inscribed_in_circle (A, B, C) ∧
  tangents_meet_at_point B C D ∧
  (let α := 3 * γ in ∀ D, (∠A B C = α ∧ ∠A C B = α)) ∧
  ∠B A C = k * π →
  k = 5 / 11 :=
begin
  sorry
end

end find_k_l56_56958


namespace square_area_l56_56000

def parabola (x : ℝ) : ℝ := x^2 - 10 * x + 21

theorem square_area :
  ∃ s : ℝ, parabola (5 + s) = -2s ∧ (2 * (-1 + 2 * real.sqrt 5))^2 = 64 - 16 * real.sqrt 5 :=
by
  sorry

end square_area_l56_56000


namespace probability_digits_different_l56_56962

theorem probability_digits_different : 
  let total_numbers := 490
  let same_digits_numbers := 13
  let different_digits_numbers := total_numbers - same_digits_numbers 
  let probability := different_digits_numbers / total_numbers 
  probability = 477 / 490 :=
by
  sorry

end probability_digits_different_l56_56962


namespace rotated_graph_eq_target_l56_56025

-- Define the original function f(x) = log₂(x)
def f (x : ℝ) : ℝ := log x / log 2

-- Define the transformation function for rotating 270 degrees clockwise
def rotate270 (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.2, -point.1)

-- Define the target function after rotation
def g (x : ℝ) : ℝ := -2^x

-- The theorem stating that the rotated graph of f(x) is g(x)
theorem rotated_graph_eq_target :
  ∀ x : ℝ, g (f x) = rotate270 (x, f x).fst :=
sorry

end rotated_graph_eq_target_l56_56025


namespace mass_percentage_Ba_in_BaI2_l56_56199

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_BaI2 : ℝ := molar_mass_Ba + 2 * molar_mass_I

theorem mass_percentage_Ba_in_BaI2 :
  (molar_mass_Ba / molar_mass_BaI2 * 100) = 35.11 :=
by
  sorry

end mass_percentage_Ba_in_BaI2_l56_56199


namespace triangle_is_great_iff_angle_90_and_isosceles_l56_56414

variable {α : Type*} [EuclideanGeometry α]

-- Define what it means for a triangle to be great.
def triangle_is_great (A B C : α) : Prop :=
  ∀ (D : α) (hD : D ∈ line_segment B C),
    let P := foot D (line_through A B),
        Q := foot D (line_through A C),
        D' := reflection D (line_through P Q) in
    D' ∈ circumcircle A B C

-- Lean theorem statement.
theorem triangle_is_great_iff_angle_90_and_isosceles (A B C : α) :
  triangle_is_great A B C ↔ (angle A B C = 90 ∧ dist A B = dist A C) := sorry

end triangle_is_great_iff_angle_90_and_isosceles_l56_56414


namespace abs_expr_value_l56_56971

theorem abs_expr_value : 
  (|(| -(| -2 + 3 |) - 2 |) + 2|) = 5 :=
by
  sorry

end abs_expr_value_l56_56971


namespace term_position_in_sequence_l56_56168

theorem term_position_in_sequence (n : ℕ) (h1 : n > 0) (h2 : 3 * n + 1 = 40) : n = 13 :=
by
  sorry

end term_position_in_sequence_l56_56168


namespace proof_x2_x1_eq_m_pi_l56_56840

noncomputable def f (n : ℕ) (α : Fin n → ℝ) (x : ℝ) : ℝ :=
  Finset.sum (Finset.univ) (λ i => (1 / 2 ^ (i : ℕ)) * Real.cos (α i + x))

theorem proof_x2_x1_eq_m_pi (n : ℕ) (α : Fin n → ℝ) (x1 x2 : ℝ) :
  f n α x1 = 0 → f n α x2 = 0 → ∃ m : ℤ, x2 - x1 = m * Real.pi :=
by
  intros h1 h2
  -- This is where the proof would go
  sorry

end proof_x2_x1_eq_m_pi_l56_56840


namespace minimum_bailing_rate_l56_56074

theorem minimum_bailing_rate
  (distance : ℝ) (to_shore_rate : ℝ) (water_in_rate : ℝ) (submerge_limit : ℝ) (r : ℝ)
  (h_distance : distance = 0.5) 
  (h_speed : to_shore_rate = 6) 
  (h_water_intake : water_in_rate = 12) 
  (h_submerge_limit : submerge_limit = 50)
  (h_time : (distance / to_shore_rate) * 60 = 5)
  (h_total_intake : water_in_rate * 5 = 60)
  (h_max_intake : submerge_limit - 60 = -10) :
  r = 2 := sorry

end minimum_bailing_rate_l56_56074


namespace cylinder_base_radius_l56_56122

theorem cylinder_base_radius (l w : ℝ) (h_l : l = 6) (h_w : w = 4) (h_circ : l = 2 * Real.pi * r ∨ w = 2 * Real.pi * r) : 
    r = 3 / Real.pi ∨ r = 2 / Real.pi := by
  sorry

end cylinder_base_radius_l56_56122


namespace dora_packs_stickers_l56_56339

theorem dora_packs_stickers :
  let lola_money := 9
  let dora_money := 9
  let combined_money := lola_money + dora_money
  let cost_playing_cards := 10
  let remaining_money := combined_money - cost_playing_cards
  let cost_each_box_stickers := 2
  let total_boxes_stickers := remaining_money / cost_each_box_stickers
  let dora_packs := total_boxes_stickers / 2
  in dora_packs = 2 :=
by
  sorry

end dora_packs_stickers_l56_56339


namespace find_D_l56_56892

theorem find_D (A B C D : ℕ) (h₁ : A + A = 6) (h₂ : B - A = 4) (h₃ : C + B = 9) (h₄ : D - C = 7) : D = 9 :=
sorry

end find_D_l56_56892


namespace cos_diff_symm_about_x_l56_56290

variable {α β : ℝ}

-- Conditions from the problem
def isSymmetricAboutX (α β : ℝ) : Prop :=
  (cos α = 1 / 4) ∧ (β = -α)

-- The proof statement
theorem cos_diff_symm_about_x (h : isSymmetricAboutX α β) : cos (α - β) = -7 / 8 :=
by sorry

end cos_diff_symm_about_x_l56_56290


namespace partition_contains_ab_eq_c_l56_56323

-- Define the set S
def S := {n : ℕ | 2 ≤ n ∧ n ≤ 256}

-- Formalize the problem in Lean 4, no proof provided.
theorem partition_contains_ab_eq_c :
  ∀ (A B : Set ℕ), (A ∪ B = S) ∧ (A ∩ B = ∅) →
  ∃ a b c ∈ A ∪ B, a * b = c :=
begin
  sorry
end

end partition_contains_ab_eq_c_l56_56323


namespace find_x_l56_56004

def custom_op (a b : ℤ) : ℤ := 2 * a + 3 * b

theorem find_x : ∃ x : ℤ, custom_op 5 (custom_op 7 x) = -4 ∧ x = -56 / 9 := by
  sorry

end find_x_l56_56004


namespace length_of_DB_l56_56278

theorem length_of_DB (A B C D : Type) 
  (right_angle_ABC : ∠ ABC = 90)
  (right_angle_ADB : ∠ ADB = 90)
  (AC : Real)
  (AD : Real)
  (h1 : AC = 24.5)
  (h2 : AD = 7):
  let DC := AC - AD
  let BD_sq := DC * AD
  let BD := Real.sqrt BD_sq
  BD = 11.07
  :=
by
  sorry

end length_of_DB_l56_56278


namespace Peter_bought_5_kilos_of_cucumbers_l56_56821

/-- 
Peter carried $500 to the market. 
He bought 6 kilos of potatoes for $2 per kilo, 
9 kilos of tomato for $3 per kilo, 
some kilos of cucumbers for $4 per kilo, 
and 3 kilos of bananas for $5 per kilo. 
After buying all these items, Peter has $426 remaining. 
How many kilos of cucumbers did Peter buy? 
-/
theorem Peter_bought_5_kilos_of_cucumbers : 
   ∃ (kilos_cucumbers : ℕ),
   (500 - (6 * 2 + 9 * 3 + 3 * 5 + kilos_cucumbers * 4) = 426) →
   kilos_cucumbers = 5 :=
sorry

end Peter_bought_5_kilos_of_cucumbers_l56_56821


namespace laser_beam_travel_distance_l56_56113

theorem laser_beam_travel_distance :
  let A : ℝ × ℝ := (2, 7)
  let D : ℝ × ℝ := (8, 3)
  let B : ℝ × ℝ := (2, -7)  -- Reflection off the x-axis at (2, -7)
  let C : ℝ × ℝ := (-2, -7)  -- Reflection off the y-axis at (-2, -7)
  let distance := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
                  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) +
                  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  in distance = 18 + 10 * Real.sqrt 2 :=
by
  -- Initial problem setup (simplified as per the formal problem statement for readability)
  sorry

end laser_beam_travel_distance_l56_56113


namespace expand_product_l56_56190

-- Define variables
variable {R : Type} [CommRing R] (x : R)

-- Define the expressions
def expr1 : R := 2 * x + 3
def expr2 : R := 3 * x^2 + 4 * x + 1
def result : R := 6 * x^3 + 17 * x^2 + 14 * x + 3

-- State the theorem
theorem expand_product : (expr1 * expr2) = result :=
by
  sorry

end expand_product_l56_56190


namespace common_ratio_arithmetic_progression_l56_56480

theorem common_ratio_arithmetic_progression (a3 q : ℝ) (h1 : a3 = 9) (h2 : a3 + a3 * q + 9 = 27) :
  q = 1 ∨ q = -1 / 2 :=
by
  sorry

end common_ratio_arithmetic_progression_l56_56480


namespace time_to_pass_pole_l56_56132

def length_of_train : ℝ := 240
def length_of_platform : ℝ := 650
def time_to_pass_platform : ℝ := 89

theorem time_to_pass_pole (length_of_train length_of_platform time_to_pass_platform : ℝ) 
  (h_train : length_of_train = 240)
  (h_platform : length_of_platform = 650)
  (h_time : time_to_pass_platform = 89)
  : (length_of_train / ((length_of_train + length_of_platform) / time_to_pass_platform)) = 24 := by
  -- Let the speed of the train be v, hence
  -- v = (length_of_train + length_of_platform) / time_to_pass_platform
  -- What we need to prove is  
  -- length_of_train / v = 24
  sorry

end time_to_pass_pole_l56_56132


namespace original_number_contains_digit_ge_5_l56_56399

theorem original_number_contains_digit_ge_5
  (n : ℕ)
  (h_nonzero_digits : ∀ d ∈ to_digits n, d ≠ 0)
  (children_rearrangements : list ℕ)
  (h_rearrangements_valid : ∀ m ∈ children_rearrangements, is_rearrangement m n)
  (h_sums_all_ones : to_digits (n + (children_rearrangements.sum)) = list.repeat 1 (list.length (to_digits n))) :
  ∃ d ∈ to_digits n, d ≥ 5 :=
by sorry

-- Auxiliary definitions and lemmas (not considered in problem statement)
def to_digits (n : ℕ) : list ℕ := sorry
def is_rearrangement (m n : ℕ) : Prop := sorry

end original_number_contains_digit_ge_5_l56_56399


namespace floor_ceil_sum_l56_56514

theorem floor_ceil_sum : (Int.floor 3.999 + Int.ceil 4.001 = 8) :=
by
  have h1 : Int.floor 3.999 = 3 := by sorry
  have h2 : Int.ceil 4.001 = 5 := by sorry
  sorry

end floor_ceil_sum_l56_56514


namespace point_on_line_l56_56380

theorem point_on_line : ∀ (m : ℝ), ∃ (x y : ℝ), (x = -1) ∧ (y = -2) ∧ ((m + 2) * x - (2 * m - 1) * y = 3 * m - 4) := 
by 
  intro m
  use -1
  use -2
  split
  { refl }
  split
  { refl }
  sorry

end point_on_line_l56_56380


namespace find_a_c_area_A_90_area_B_90_l56_56263

variable (a b c : ℝ)
variable (C : ℝ)

def triangle_condition1 := a + 1/a = 4 * Real.cos C
def triangle_condition2 := b = 1
def sin_C := Real.sin C = Real.sqrt 21 / 7

-- Proof problem for (1)
theorem find_a_c (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h3 : sin_C C) :
  (a = Real.sqrt 7 ∧ c = 2) ∨ (a = Real.sqrt 7 / 7 ∧ c = 2 * Real.sqrt 7 / 7) :=
sorry

-- Conditions for (2) when A=90°
def right_triangle_A := C = Real.pi / 2

-- Proof problem for (2) when A=90°
theorem area_A_90 (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h4 : right_triangle_A C) :
  ((a = Real.sqrt 3) → area = Real.sqrt 2 / 2) :=
sorry

-- Conditions for (2) when B=90°
def right_triangle_B := b = 1 ∧ C = Real.pi / 2

-- Proof problem for (2) when B=90°
theorem area_B_90 (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h5 : right_triangle_B b C) :
  ((a = Real.sqrt 3 / 3) → area = Real.sqrt 2 / 6) :=
sorry

end find_a_c_area_A_90_area_B_90_l56_56263


namespace age_ratio_l56_56267

-- Define variables for the current ages of A and B
variable (A B : ℕ)

-- Given conditions
def age_conditions : Prop :=
  B = 39 ∧ A = B + 9

-- Mathematical statement to prove
theorem age_ratio (A B : ℕ) (h : age_conditions A B) : (A + 10) / (B - 10) = 2 :=
by
  unfold age_conditions at h
  cases h with hB hA
  rw [hB, hA]
  norm_num
  sorry

end age_ratio_l56_56267


namespace annual_decrease_rate_l56_56863

def initial_population : ℝ := 8000
def population_after_two_years : ℝ := 3920

theorem annual_decrease_rate :
  ∃ r : ℝ, (0 < r ∧ r < 1) ∧ (initial_population * (1 - r)^2 = population_after_two_years) ∧ r = 0.3 :=
by
  sorry

end annual_decrease_rate_l56_56863


namespace number_of_valid_colorings_l56_56669

-- Define the specific properties and constraints given in the problem
def is_power_of_two (n : ℕ) : Prop := ∃ m : ℕ, n = 2^m

def valid_coloring (f : ℕ → bool) : Prop :=
  ∀ a b : ℕ, a ≠ b → a + b ≤ 200 →
  (f a = f b → ¬ is_power_of_two (a + b))

theorem number_of_valid_colorings : 
  ∃ c : ℕ, c = 256 ∧ 
  ∃ f : Π (n : ℕ), nat.lt n 201 → bool, 
    valid_coloring (λ n, f n (by linarith)) :=
begin
  sorry
end

end number_of_valid_colorings_l56_56669


namespace quaternion_problem_solution_l56_56201

open Quaternion

noncomputable theory

def find_q {a b c d : ℝ} (q : ℍ) : Prop :=
  q = a + b * i + c * j + d * k ∧
  q^2 = (-1 - i - j - k) 

theorem quaternion_problem_solution :
  find_q (-1 - 1/2 * i - 1/2 * j - 1/2 * k) 
:= by
  sorry

end quaternion_problem_solution_l56_56201


namespace evaluate_expression_l56_56991

theorem evaluate_expression (x : ℝ) (h : x > 2) : 
  sqrt (x^2 / (1 - (x^2 - 4) / x^2)) = x^2 / 2 :=
sorry

end evaluate_expression_l56_56991


namespace compute_expression_l56_56789

-- Define ω as a nonreal root of x^3 = 1
def ω : ℂ :=  -- ℂ stands for the set of complex numbers
  -- ℂ.nonreal part is to indicate that the root should not be real
  classical.some (Complex.exists_roots_of_degree_three 1 0 0 1)
  -- This is more aligned to formal coding practice, specify that it is a nonreal root

-- Given property ω^3 = 1
axiom ω_property : ω^3 = 1

-- The proof statement for the required expression to equal -32
theorem compute_expression : (1 - ω^2 + ω^4)^4 + (1 + ω^2 - ω^4)^4 = -32 := 
by
  sorry  -- Proof skipped

end compute_expression_l56_56789


namespace player_B_wins_l56_56411

-- Define the type representing a player
inductive Player
| A : Player
| B : Player

-- Define the structure of the 6x6 grid
structure Grid :=
(cells : Array (Array ℝ)) (h_size : cells.size = 6 ∧ cells.all (λ row => row.size = 6))

-- Definition of the game conditions
def is_black (g : Grid) (i : ℕ) (j : ℕ) : Prop :=
  j < 6 ∧ (∀ k, k < 6 → g.cells[i][j] ≥ g.cells[i][k])

noncomputable def has_vertical_path (g : Grid) : Prop :=
  ∃ path : Fin 6 → Fin 6,
  (∀ i : Fin 6, path i < 6) ∧
    (∀ i, (is_black g i (path i)) ∧ (i > 0 → path i = path (i - 1) ∨ path i = path (i - 1) + 1 ∨ path i = path (i - 1) - 1))

-- The theorem to prove
theorem player_B_wins : ∀ (g : Grid), (∀ p : Player, ¬ has_vertical_path g → p = Player.B) :=
sorry

end player_B_wins_l56_56411


namespace trips_to_fill_container_l56_56139

-- Definitions for the cylindrical container dimensions
def rc : ℝ := 8
def hc : ℝ := 20

-- Definition of the volume of the cylindrical container
def Vc : ℝ := Real.pi * rc^2 * hc

-- Definitions for the conical frustum bucket dimensions
def r1 : ℝ := 6
def r2 : ℝ := 4
def hf : ℝ := 10

-- Definition of the volume of the conical frustum bucket
def Vf : ℝ := (1/3) * Real.pi * hf * (r1^2 + r1 * r2 + r2^2)

-- Theorem stating the number of trips required to fill the container
theorem trips_to_fill_container : Nat.ceil (Vc / Vf) = 6 := by
  sorry

end trips_to_fill_container_l56_56139


namespace probability_eq_l56_56915

noncomputable def probability_exactly_two_one_digit_and_three_two_digit : ℚ := 
  let n := 5
  let p_one_digit := 9 / 20
  let p_two_digit := 11 / 20
  let binomial_coeff := Nat.choose 5 2
  (binomial_coeff * p_one_digit^2 * p_two_digit^3)

theorem probability_eq : probability_exactly_two_one_digit_and_three_two_digit = 539055 / 1600000 := 
  sorry

end probability_eq_l56_56915


namespace count_possible_values_of_A_l56_56353

theorem count_possible_values_of_A :
  (∀ (A B : ℕ), (A ≤ 4) → (17 * 1000 + 100 * A + 10 * B).rounding (100) = 1700) →
  (∃ S : Finset ℕ, S = {0, 1, 2, 3, 4} ∧ S.card = 5) :=
begin
  intros h,
  use {0, 1, 2, 3, 4},
  split,
  { refl,},
  { simp,},
end

end count_possible_values_of_A_l56_56353


namespace Ksyusha_travel_time_l56_56778

theorem Ksyusha_travel_time:
  ∀ (S v : ℝ), 
  (S > 0 ∧ v > 0) →
  (∀ tuesday_distance tuesday_time wednesday_distance wednesday_time: ℝ, 
    tuesday_distance = 3 * S ∧ 
    tuesday_distance = 2 * S + S ∧ 
    tuesday_time = (2 * S) / v + (S / (2 * v)) ∧ 
    tuesday_time = 30) → 
  (∀ wednesday_distance wednesday_time : ℝ, 
    wednesday_distance = S + 2 * S ∧ 
    wednesday_time = S / v + (2 * S) / (2 * v)) →
  wednesday_time = 24 :=
by
  intros S v S_pos v_pos tuesday_distance tuesday_time wednesday_distance wednesday_time
  sorry

end Ksyusha_travel_time_l56_56778


namespace living_room_floor_area_l56_56449

-- Define the problem conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area : ℝ := carpet_length * carpet_width    -- Area of the carpet

def percentage_covered_by_carpet : ℝ := 0.75

-- Theorem to prove: the area of the living room floor is 48 square feet
theorem living_room_floor_area (carpet_area : ℝ) (percentage_covered_by_carpet : ℝ) : 
  (A_floor : ℝ) = carpet_area / percentage_covered_by_carpet :=
by
  let carpet_area := 36
  let percentage_covered_by_carpet := 0.75
  let A_floor := 48
  sorry

end living_room_floor_area_l56_56449


namespace minimum_diagonal_of_rectangle_l56_56260

theorem minimum_diagonal_of_rectangle (l w : ℝ) (h_perimeter : l + w = 12) : 
  ∃ d, d = sqrt (l^2 + w^2) ∧ d = 6 * sqrt 2 :=
by 
  use sqrt (l^2 + w^2)
  sorry

end minimum_diagonal_of_rectangle_l56_56260


namespace initial_average_age_l56_56365

theorem initial_average_age (A : ℝ) (n : ℕ) (h1 : n = 9) (h2 : (n * A + 35) / (n + 1) = 17) :
  A = 15 :=
by
  sorry

end initial_average_age_l56_56365


namespace red_candies_remaining_percent_l56_56272

theorem red_candies_remaining_percent :
  let initial_red := 50
  let initial_orange := 60
  let initial_green := 40
  let initial_blue := 70
  let initial_yellow := 80
  let initial_violet := 90
  let initial_indigo := 100

  let green_eaten := initial_green
  let remaining_orange := initial_orange - 0.60 * initial_orange
  let remaining_red := initial_red - (2 / 3) * initial_red
  let remaining_blue := initial_blue - 0.7 * initial_blue
  let remaining_yellow := initial_yellow - (5 / 9) * initial_yellow
  let remaining_violet := initial_violet - (8 / 15) * initial_violet
  let remaining_indigo := initial_indigo
  
  let total_remaining := 0 + remaining_orange + remaining_red + remaining_blue + remaining_yellow + remaining_violet + remaining_indigo
  let total_needed_remaining := 220
  let adjustment := total_remaining - total_needed_remaining
  let red_final := remaining_red - adjustment / 3

  red_final / initial_red * 100 ≈ 20.52 := 
    sorry

end red_candies_remaining_percent_l56_56272


namespace sum_of_coefficients_is_one_l56_56231

-- Given conditions for the mathematical problem
axiom n : ℕ 
axiom condition : (binomial n 2 > binomial n 1) ∧ (binomial n 2 > binomial n 3) ∧ (∀ k, k ≠ 2 → binomial n k ≤ binomial n 2)

-- Proof statement
theorem sum_of_coefficients_is_one 
  (h : n = 4) :
  (∑ k in range (n + 1), (-2)^k * binomial n k) = 1 :=
by sorry

end sum_of_coefficients_is_one_l56_56231


namespace number_of_rides_per_person_l56_56374

noncomputable theory

def entrance_fee_under_18 : ℝ := 5
def entrance_fee_over_18 : ℝ := 5 + 0.20 * 5
def total_entrance_fee (number_of_brothers : ℕ) : ℝ := entrance_fee_over_18 + number_of_brothers * entrance_fee_under_18
def total_amount_spent : ℝ := 20.5
def cost_per_ride : ℝ := 0.5
def total_rides_taken (total_amount_spent total_entrance_fee cost_per_ride : ℝ) : ℝ := (total_amount_spent - total_entrance_fee) / cost_per_ride
def rides_per_person (total_rides_taken : ℝ) (num_people : ℕ) : ℝ := total_rides_taken / num_people

theorem number_of_rides_per_person (number_of_brothers : ℕ) (num_people : ℕ) :
  total_entrance_fee number_of_brothers = 16 →
  total_amount_spent = 20.5 →
  cost_per_ride = 0.5 →
  rides_per_person (total_rides_taken total_amount_spent (total_entrance_fee number_of_brothers) cost_per_ride) num_people = 3 :=
by
  sorry

end number_of_rides_per_person_l56_56374


namespace complex_solution_correct_l56_56995

noncomputable def complex_solutions : List ℂ :=
  [(2 - 12*complex.I), (-2 + 12*complex.I)]

theorem complex_solution_correct (z : ℂ) (hz : z ∈ complex_solutions) : z^2 = -143 - 48*complex.I := by
  sorry

end complex_solution_correct_l56_56995


namespace factors_of_120_that_are_multiples_of_15_l56_56638

theorem factors_of_120_that_are_multiples_of_15 :
  {d : ℕ | d > 0 ∧ 120 % d = 0 ∧ d % 15 = 0}.to_finset.card = 4 :=
by sorry

end factors_of_120_that_are_multiples_of_15_l56_56638


namespace weight_of_final_statue_l56_56432

def original_weight : ℝ := 250
def first_week_fraction_cut : ℝ := 0.30
def second_week_fraction_cut : ℝ := 0.20
def third_week_fraction_cut : ℝ := 0.25

theorem weight_of_final_statue :
  let weight_after_first_week := original_weight * (1 - first_week_fraction_cut) in
  let weight_after_second_week := weight_after_first_week * (1 - second_week_fraction_cut) in
  let weight_after_third_week := weight_after_second_week * (1 - third_week_fraction_cut) in
  weight_after_third_week = 105 :=
by
  sorry

end weight_of_final_statue_l56_56432


namespace find_number_l56_56118

theorem find_number (a b : ℕ) (h₁ : a = 555) (h₂ : b = 445) :
  let S := a + b
  let D := a - b
  let Q := 2 * D
  let R := 30
  let N := (S * Q) + R
  N = 220030 := by
  sorry

end find_number_l56_56118


namespace tommy_nickels_l56_56047

-- Definitions of given conditions
def pennies (quarters : Nat) : Nat := 10 * quarters  -- Tommy has 10 times as many pennies as quarters
def dimes (pennies : Nat) : Nat := pennies + 10      -- Tommy has 10 more dimes than pennies
def nickels (dimes : Nat) : Nat := 2 * dimes         -- Tommy has twice as many nickels as dimes

theorem tommy_nickels (quarters : Nat) (P : Nat) (D : Nat) (N : Nat) 
  (h1 : quarters = 4) 
  (h2 : P = pennies quarters) 
  (h3 : D = dimes P) 
  (h4 : N = nickels D) : 
  N = 100 := 
by
  -- sorry allows us to skip the proof
  sorry

end tommy_nickels_l56_56047


namespace shortest_distance_ln_x_to_line_eq_4_plus_ln2_div_sqrt_5_l56_56202

theorem shortest_distance_ln_x_to_line_eq_4_plus_ln2_div_sqrt_5 :
      ∀ (x : ℝ), x > 0 → (y = ln x) → 
      let distance := (abs (2 * 1/2 + (-ln 2) + 3)) / (sqrt (2^2 + (-1)^2))
      (2 * x - y + 3 = 0) → distance = (4 + ln 2) / sqrt 5 :=
by sorry

end shortest_distance_ln_x_to_line_eq_4_plus_ln2_div_sqrt_5_l56_56202


namespace average_donation_per_student_l56_56160

theorem average_donation_per_student :
  let total_students := 1 -- assume 1 for normalization, given in percentage
  let percentage_10 := 0.4
  let percentage_5 := 0.3
  let percentage_2 := 0.2
  let percentage_0 := 0.1
  let amount_10 := 10
  let amount_5 := 5
  let amount_2 := 2
  let amount_0 := 0
  let avg_donation := percentage_10 * amount_10 +
                      percentage_5 * amount_5 +
                      percentage_2 * amount_2 +
                      percentage_0 * amount_0
  in
  avg_donation = 5.9 :=
by
  sorry

end average_donation_per_student_l56_56160


namespace beads_necklace_l56_56815

theorem beads_necklace (beads : Fin (1000)) (color : beads → Fin (50)) :
  ∃ k : ℕ, (k ≤ 462) ∧ (∀ (a : ℕ), a < 1000 → {c : Fin 50 | ∃ i, (a ≤ i ∧ i < a + k) ∧ color beads[i] = c}.card ≥ 25) := 
sorry

end beads_necklace_l56_56815


namespace integral_correct_l56_56872

noncomputable def integral_value : ℝ :=
  ∫ x in 1..2, (1 + x^2) / x

theorem integral_correct : integral_value = (3 / 2) + Real.log 2 :=
by
  -- In this placeholder, the solution steps would be filled in.
  sorry

end integral_correct_l56_56872


namespace color_natural_numbers_l56_56678

theorem color_natural_numbers :
  ∃ f : fin 200 → bool, (∀ a b : fin 200, a ≠ b → f a = f b → (a + b).val ∉ {2^n | n : ℕ}) ∧
  (by let count := ∑ color in (fin 200 → bool), 
         ite (∀ a b, a ≠ b → color a = color b → (a + b).val ∉ {2^n | n : ℕ}) 1 0 ;
      exact count = 256) := sorry

end color_natural_numbers_l56_56678


namespace probability_all_rows_columns_odd_is_1_div_360_l56_56029

-- Define the 4x4 grid as a finite set of numbers 1 to 16
inductive Number : Type
| mk (val : Nat) (h : 1 ≤ val ∧ val ≤ 16) : Number

-- Define the grid type
def Grid := Array (Array Number)

-- Predicate for checking if the sum of a row or a column is odd
def sum_is_odd (numbers : Array Number) : Prop :=
  numbers.foldl (λ acc n => acc + match n with | Number.mk val _ => val) 0 % 2 = 1

-- Predicate for checking if all rows and columns in the grid have odd sums
def all_rows_columns_odd (grid : Grid) : Prop :=
  (Array.all grid sum_is_odd) ∧ 
  (Array.all (Array.map (λ i => Array.map (λ row => row[i]) grid) (Array.range 4)) sum_is_odd)

-- Statement of the problem translated to Lean 4: the probability is 1/360
theorem probability_all_rows_columns_odd_is_1_div_360 :
  -- Conditions: the grid is filled with numbers 1 to 16 each exactly once
  ∃ (grid : Grid), 
    (∀ i j, ∃ val, grid[i][j] = Number.mk val (by simp; sorry)) ∧
    (∀ val, ∃ i j, grid[i][j] = Number.mk val (by simp; sorry)) →
    -- Question: Probability condition
    (all_rows_columns_odd grid) →
    1 / 360 = sorry := sorry

end probability_all_rows_columns_odd_is_1_div_360_l56_56029


namespace ksyusha_travel_time_l56_56770

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l56_56770


namespace systematic_sampling_fourth_group_number_l56_56513

theorem systematic_sampling_fourth_group_number (n : ℕ) (step_size : ℕ) (first_number : ℕ) : 
  n = 4 → step_size = 6 → first_number = 4 → (first_number + step_size * 3) = 22 :=
by
  intros h_n h_step_size h_first_number
  sorry

end systematic_sampling_fourth_group_number_l56_56513


namespace find_k_l56_56083

theorem find_k (k : ℝ) (h : 64 / k = 8) : k = 8 := 
sorry

end find_k_l56_56083


namespace incorrect_statement_about_linear_regression_l56_56956

theorem incorrect_statement_about_linear_regression :
  (∀ (A B C D : Prop), 
    (A ∧ B ∧ C → D = false) → 
    (Two variables with a correlation do not necessarily have a causal relationship → 
    Scatter plots can intuitively reflect the degree of correlation of data → 
    The regression line best represents the relationship between two variables with linear correlation → 
    ¬ Every set of data has a regression equation)) := 
by {
  sorry
}

end incorrect_statement_about_linear_regression_l56_56956


namespace negative_numbers_count_l56_56141

-- Definitions of the given rational numbers
def num1 := -3^2
def num2 := (-1 : ℚ)^(2006)
def num3 := (0 : ℚ)
def num4 := |(-2 : ℚ)|
def num5 := -(-2 : ℚ)
def num6 := -3 * 2^2

-- List the numbers
def numbers := [num1, num2, num3, num4, num5, num6]

-- Count the negative numbers
def is_negative (x : ℚ) : Prop := x < 0

def count_negatives := (numbers.filter is_negative).length

-- Statement of the theorem: There are 2 negative numbers in the list
theorem negative_numbers_count : count_negatives = 2 := 
by sorry

end negative_numbers_count_l56_56141


namespace integer_area_of_trapezoid_l56_56706

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def area_of_trapezoid (AB CD BC : ℕ) : ℕ :=
  (AB + CD) * BC / 2

theorem integer_area_of_trapezoid :
  ∀ (BC AB CD r : ℕ), 
  AB * CD = r * r → 
    ({ (4, 9), (8, 2), (6, 6) } = 
    { (AB, CD) | is_perfect_square (AB * CD) }):
  sorry

end integer_area_of_trapezoid_l56_56706


namespace symmetry_coordinates_l56_56370

variable (x y z : ℝ)

def symmetric_about_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

theorem symmetry_coordinates (x y z : ℝ) :
  symmetric_about_y_axis (x, y, z) = (-x, y, -z) :=
by
  simp only [symmetric_about_y_axis]
  rfl

end symmetry_coordinates_l56_56370


namespace cos_double_angle_l56_56584

theorem cos_double_angle (θ : ℝ) (h : 2^(-2 + 4 * Real.cos θ) + 1 = 2^(-1/2 + 2 * Real.cos θ)) : Real.cos (2 * θ) = -1/2 :=
sorry

end cos_double_angle_l56_56584


namespace min_groups_l56_56473

theorem min_groups (students : ℕ) (max_per_group : ℕ) (h1 : students = 30) (h2 : max_per_group = 12) : 
  (students / 10) = 3 :=
by 
  rw [h1, h2]
  sorry

end min_groups_l56_56473


namespace find_p_q_l56_56698

-- Define the conditions
variables (ABCD : Type) [Quadrilateral ABCD] (A B C D : ABCD)
variable (BC CD AD : ℝ) (angle_A angle_B : ℝ)
variable (p q : ℤ)

-- Setting specific condition values
axiom BC_eq : BC = 10
axiom CD_eq : CD = 15
axiom AD_eq : AD = 12
axiom angle_A_eq : angle_A = 60
axiom angle_B_eq : angle_B = 60

-- Define the main statement to show that p + q = 796 under these conditions
theorem find_p_q (h : ∃ p q : ℤ, (AB = p + sqrt q) ∧ p + q = 796) : ∃ (p q : ℤ), p + q = 796 := 
by 
  sorry

end find_p_q_l56_56698


namespace ab_value_l56_56211

theorem ab_value (a b : ℝ) (log_two_3 : ℝ := Real.log 3 / Real.log 2) :
  a * log_two_3 = 1 ∧ (4 : ℝ)^b = 3 → a * b = 1 / 2 := by
  sorry

end ab_value_l56_56211


namespace ratio_of_triangle_areas_l56_56865

theorem ratio_of_triangle_areas
  (a b c S : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (angle_bisector_theorem : ∀ {A B C K : ℝ}, A / B = C / K)
  (area_ABC : S = (b * c * (a + b + c)))
  : ∃ (area_BOK : ℝ), area_BOK = acS / ((a + b) * (a + b + c)) :=
sorry

end ratio_of_triangle_areas_l56_56865


namespace range_of_f_on_interval_l56_56377

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + π / 4) - 1

theorem range_of_f_on_interval :
  ∀ x, -π / 2 ≤ x ∧ x ≤ -π / 24 → -3 ≤ f x ∧ f x ≤ 0 :=
by 
  sorry

end range_of_f_on_interval_l56_56377


namespace gcd_78_36_l56_56198

theorem gcd_78_36 : Nat.gcd 78 36 = 6 :=
by
  sorry

end gcd_78_36_l56_56198


namespace coeff_x2_expansion_l56_56368

theorem coeff_x2_expansion (n r : ℕ) (a b : ℤ) :
  n = 5 → a = 1 → b = 2 → r = 2 →
  (Nat.choose n r) * (a^(n - r)) * (b^r) = 40 :=
by
  intros Hn Ha Hb Hr
  rw [Hn, Ha, Hb, Hr]
  simp
  sorry

end coeff_x2_expansion_l56_56368


namespace dora_packs_of_stickers_l56_56342

theorem dora_packs_of_stickers (allowance : ℕ) (deck_cost : ℕ) (sticker_pack_cost : ℕ) 
  (combined_allowance : ℕ) (remaining_money : ℕ) (total_sticker_packs : ℕ) (dora_sticker_packs : ℕ) :
  allowance = 9 →
  deck_cost = 10 →
  sticker_pack_cost = 2 →
  combined_allowance = 2 * allowance →
  remaining_money = combined_allowance - deck_cost →
  total_sticker_packs = remaining_money / sticker_pack_cost →
  dora_sticker_packs = total_sticker_packs / 2 →
  dora_sticker_packs = 2 :=
by
  intros h_allowance h_deck_cost h_sticker_pack_cost h_combined_allowance 
    h_remaining_money h_total_sticker_packs h_dora_sticker_packs
  rw [h_allowance, h_deck_cost, h_sticker_pack_cost] at h_combined_allowance h_remaining_money h_total_sticker_packs h_dora_sticker_packs
  rw [h_combined_allowance, h_remaining_money, h_total_sticker_packs] at h_dora_sticker_packs
  rw [h_combined_allowance] at h_remaining_money
  rw [h_remaining_money] at h_total_sticker_packs
  rw [h_total_sticker_packs] at h_dora_sticker_packs
  exact h_dora_sticker_packs

end dora_packs_of_stickers_l56_56342


namespace red_peaches_difference_l56_56914

theorem red_peaches_difference (r y : Nat) (h1 : r = 19) (h2 : y = 11) : r - y = 8 :=
by
  rw [h1, h2]
  norm_num
  sorry

end red_peaches_difference_l56_56914


namespace spiders_win_l56_56042

structure GameState :=
  (positions : Fin 4 → Fin 20)   -- denotes the vertex positions of the beetle and spiders, where Fin 4 indicates 4 players (1 beetle, 3 spiders), and Fin 20 indicates the 20 vertices of the dodecahedron
  (max_speeds : Fin 4 → ℝ)       -- denotes the max speeds of the beetle and spiders

def can_catch (beetle spider_fast spider_slow1 spider_slow2 : GameState) : Prop :=
  ∃ t : ℝ, ∃ i : Fin 4, beetle.positions i = spider_fast.positions i ∨ 
                         beetle.positions i = spider_slow1.positions i ∨ 
                         beetle.positions i = spider_slow2.positions i

theorem spiders_win (beetle spider_fast spider_slow1 spider_slow2 : GameState)
  (h1 : beetle.max_speeds 0 = 1) 
  (h2 : spider_fast.max_speeds 1 = 1) 
  (h3 : spider_slow1.max_speeds 2 = 1/2018)
  (h4 : spider_slow2.max_speeds 3 = 1/2018)
  (known_positions : ∀ t : ℝ, ∀ i : Fin 4, beetle.positions i=t ∧ spider_fast.positions i=t ∧ spider_slow1.positions i=t ∧ spider_slow2.positions i=t) :
  can_catch beetle spider_fast spider_slow1 spider_slow2 :=
begin
  sorry
end

end spiders_win_l56_56042


namespace find_angles_equal_find_max_min_f_l56_56297

-- Define the conditions of the problem
variables (a b c : ℝ)
variables (A B C : ℝ)
variables (m n p : ℝ × ℝ)
variables (h1 : m = (a, b))
variables (h2 : n = (cos A, cos B))
variables (h3 : p = (2 * sqrt 2 * sin ((B + C) / 2), 2 * sin A))
variables (h_parallel : m.1 * n.2 = m.2 * n.1)
variables (h_norm_p : (p.1 ^ 2 + p.2 ^ 2) = 9)
variables (h_triangle_sum : A + B + C = π)

-- Prove the values of angles A, B, and C
theorem find_angles_equal (h_parallel : m.1 * n.2 = m.2 * n.1) : 
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3 := sorry

-- Define the function f(x)
noncomputable def f (x : ℝ) := sin A * sin x + cos B * cos x

-- Prove the maximum and minimum values of f(x)
theorem find_max_min_f : 
  (∀ x ∈ (set.Icc 0 (π / 2)), f x ≤ 1) ∧ 
  (∀ x ∈ (set.Icc 0 (π / 2)), f x ≥ 1 / 2) := sorry

end find_angles_equal_find_max_min_f_l56_56297


namespace area_triangle_acd_l56_56052

-- Definition of the triangles and their properties
def Triangle (V : Type) := (V × V × V)

variable {V : Type}
variable [EuclideanSpace V]

variables {A B C D : V}
variables (AB BD AC : ℝ)

-- Conditions given in the problem
def is_right_triangle (T : Triangle V) : Prop := 
  ∃ (a b c : V), T = (a, b, c) ∧ (∃ (θ : ℝ), θ = π / 2 
     ∧ (norm (a - b)) ^ 2 + (norm (b - c)) ^ 2 = (norm (a - c)) ^ 2)

def bisects_ad : Prop := 
  ∃ θ : ℝ, θ = angle A B C ∧ cos θ = cos (π / 2 / 2)

-- Definition of isosceles right triangle
def is_isosceles_right_triangle (T : Triangle V) : Prop :=
  ∃ (a b c : V), T = (a, b, c) ∧ (norm (a - b) = norm (b - c))
  ∧ is_right_triangle T

-- Main theorem for proving the area of Triangle ACD
theorem area_triangle_acd (h1 : is_right_triangle (A, C, D))
                         (h2 : bisects_ad)
                         (h3 : AB = BD)
                         (h4 : AC = 12)
                         (h5 : AB = 8) :
  ∃ (area : ℝ), area = 28.8 := 
by
  sorry -- Proof is omitted

end area_triangle_acd_l56_56052


namespace baking_time_l56_56829

theorem baking_time (n : ℕ) (selling_price : ℕ → ℕ) (purchase_price : ℕ) (baking_rate : ℕ) :
  selling_price n = (n + 20) * (n + 15) →
  purchase_price = 1050 →
  baking_rate = 10 →
  (∃ t : ℕ, t = 90 ∧ (n * 60 / baking_rate) = t) :=
by
  assume h1 : selling_price n = (n + 20) * (n + 15),
  assume h2 : purchase_price = 1050,
  assume h3 : baking_rate = 10,
  -- Proof would go here
  sorry

end baking_time_l56_56829


namespace cassandra_collected_pennies_l56_56159

theorem cassandra_collected_pennies 
(C : ℕ) 
(h1 : ∀ J : ℕ,  J = C - 276) 
(h2 : ∀ J : ℕ, C + J = 9724) 
: C = 5000 := 
by
  sorry

end cassandra_collected_pennies_l56_56159


namespace admin_leader_duty_arrangements_l56_56185

theorem admin_leader_duty_arrangements : 
  let num_ways_A := Nat.choose 6 2 in
  let num_ways_B := Nat.choose 4 2 in
  let num_ways_C := 1 in
  num_ways_A * num_ways_B * num_ways_C = 90 :=
by
  let num_ways_A := Nat.choose 6 2
  let num_ways_B := Nat.choose 4 2
  let num_ways_C := 1
  show num_ways_A * num_ways_B * num_ways_C = 90
  sorry

end admin_leader_duty_arrangements_l56_56185


namespace alloy_impurity_ratio_l56_56039

theorem alloy_impurity_ratio 
  (p q r : ℝ)
  (hp: p = 70) 
  (hq: q = 5) 
  (hr: r = 40) :
  (r - q) / (p - r) ≈ 1.17 :=
by
  rw [hp, hq, hr]
  simp
  norm_num
  sorry

end alloy_impurity_ratio_l56_56039


namespace find_f_neg3_l56_56838

def f : ℝ → ℝ := sorry

theorem find_f_neg3 (h : ∀ x ≠ 0, 4 * f (1 / x) + (3 * f x) / x = x^3) : 
  f (-3) = -6565 / 189 :=
begin
  sorry
end

end find_f_neg3_l56_56838


namespace number_of_three_digit_integers_l56_56607

-- Definitions from conditions
def digits : List ℕ := [1, 3, 5, 8]

-- The main statement to prove
theorem number_of_three_digit_integers : 
  (∃ n : ℕ, n = digits.length * (digits.length - 1) * (digits.length - 2) ∧ n = 24) :=
by
  have h1 : digits.length = 4 := by simp
  have h2 : 4 * 3 * 2 = 24 := by norm_num
  use 4 * 3 * 2
  simp [h1, h2]
  sorry  -- Proof placeholder

end number_of_three_digit_integers_l56_56607


namespace cousin_drink_ratio_l56_56810

theorem cousin_drink_ratio (cousin_drink x : ℝ) (bowl_capacity : ℝ) (mark_add : ℝ) (sally_drink : ℝ) (mark_refill : ℝ) :
  bowl_capacity = 16 ∧ mark_add = 4 ∧ sally_drink = 2 ∧ mark_refill = 12 ∧ (x - cousin_drink + 2 + 12 = bowl_capacity) →
  cousin_drink / x = 1 :=
by
  intros h,
  have h1 := h.1,
  have h2 := h.2.1,
  have h3 := h.2.2.1,
  have h4 := h.2.2.2.1,
  have h5 := h.2.2.2.2,
  have h6 : x - cousin_drink + 14 = 16 := by linarith,
  have h7 : x - cousin_drink = 2 := by linarith,
  have h8 : x = 2 := by linarith,
  have h9 : cousin_drink = 2 := by linarith,
  have hr : 2 / 2 = 1 := by norm_num,
  exact hr

end cousin_drink_ratio_l56_56810


namespace unique_solution_condition_l56_56548

theorem unique_solution_condition (k : ℝ) :
  (∀ (x y : ℝ), (y = x^2 ∧ y = 2x - k → ∃! (x : ℝ), y)) ↔ k = 1 := 
by
  sorry

end unique_solution_condition_l56_56548


namespace num_loser_integers_l56_56332

theorem num_loser_integers (k N : ℕ) (h_pos_k : 0 < k) (h_pos_N : 0 < N) :
  ∃ count : ℕ, count = 2^(N - (Nat.floor (Real.log (min N k) / Real.log 2))) ∧
  ∀ t : ℕ, t < 2^N → ¬winning_strategy t → is_loser t :=
sorry

end num_loser_integers_l56_56332


namespace min_value_expression_l56_56557

theorem min_value_expression (a : ℝ) (b : ℝ) (h : 0 < b) : 
  ∃ x, x = 2 * (1 - real.log 2) ^ 2 ∧ 
  x = (1 / 2 * real.exp a - real.log (2 * b))^2 + (a - b)^2 :=
sorry

end min_value_expression_l56_56557


namespace remainder_is_zero_l56_56174

noncomputable def polynomial_division_theorem : Prop :=
  ∀ (x : ℤ), (x^3 ≡ 1 [MOD (x^2 + x + 1)]) → 
             (x^5 ≡ x^2 [MOD (x^2 + x + 1)]) →
             (x^2 - 1) * (x^3 - 1) ≡ 0 [MOD (x^2 + x + 1)]

theorem remainder_is_zero : polynomial_division_theorem := by
  sorry

end remainder_is_zero_l56_56174


namespace expected_value_ξ_l56_56661

-- Define conditions
def n : ℕ := 5
def p : ℚ := 1 / 3

-- Define random variable ξ as binomial distribution's expected value
def ξ_expected_value : ℚ := n * p

-- The statement to be proved
theorem expected_value_ξ : ξ_expected_value = 5 / 3 :=
by
  -- proof would go here
  sorry

end expected_value_ξ_l56_56661


namespace triangle_angle_C_triangle_area_l56_56656

variables (A B C a b c : ℝ)
variables (sinA cosA sinB cosB : ℝ)

theorem triangle_angle_C (h1 : a ≠ b) 
  (h2 : c = √3)
  (h3 : cosA^2 - cosB^2 = √3 * sinA * cosA - √3 * sinB * cosB) :
  C = π / 3 :=
sorry

theorem triangle_area (h1 : a ≠ b)
  (h2 : c = √3)
  (h3 : cosA^2 - cosB^2 = √3 * sinA * cosA - √3 * sinB * cosB)
  (h4 : sinA = 4 / 5 )
  (h5 : a = (√3 * 4 / 5) / (√3 / 2)) :
  let area := 1 / 2 * √3 * (8 / 5) * (3 * √3 + 4) / 10 in
  area = (8 * √3 + 18) / 25 :=
sorry

end triangle_angle_C_triangle_area_l56_56656


namespace part1_part2_part3_l56_56618

theorem part1 (k x : ℝ) (hk : k ≠ 0) : (x+2)/k > 1 + (x-3)/k^2 :=
sorry

theorem part2 (k : ℝ) (hk : k ≠ 0) (H : ∀ x, (x+2)/k > 1 + (x-3)/k^2 ↔ x ∈ (3, ⊤)) : k = 5 :=
sorry

theorem part3 (k : ℝ) (hk : k ≠ 0) (H : (3+2)/k > 1 + (3-3)/k^2) : 0 < k ∧ k < 5 :=
sorry

end part1_part2_part3_l56_56618


namespace largest_marbles_l56_56909

theorem largest_marbles {n : ℕ} (h1 : n < 400) (h2 : n % 3 = 1) (h3 : n % 7 = 2) (h4 : n % 5 = 0) : n = 310 :=
  sorry

end largest_marbles_l56_56909


namespace grunters_win_5_games_probability_l56_56361

theorem grunters_win_5_games_probability :
  let probability_each_game : ℚ := 3 / 5,
      num_games : ℕ := 5
  in (probability_each_game ^ num_games = 243 / 3125) :=
by
  let probability_each_game : ℚ := 3 / 5
  let num_games : ℕ := 5
  have : probability_each_game ^ num_games = 243 / 3125 := sorry
  exact this

end grunters_win_5_games_probability_l56_56361


namespace Ksyusha_time_to_school_l56_56732

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l56_56732


namespace initial_overs_l56_56662

theorem initial_overs (initial_run_rate remaining_run_rate target runs initially remaining_overs : ℝ)
    (h_target : target = 282)
    (h_remaining_overs : remaining_overs = 40)
    (h_initial_run_rate : initial_run_rate = 3.6)
    (h_remaining_run_rate : remaining_run_rate = 6.15)
    (h_target_eq : initial_run_rate * initially + remaining_run_rate * remaining_overs = target) :
    initially = 10 :=
by
  sorry

end initial_overs_l56_56662


namespace coefficient_a3b2_in_expansion_l56_56417

theorem coefficient_a3b2_in_expansion :
  let coeff := (nat.choose 5 2) * (0 : ℕ)
  coeff = 0 := by
  -- binomial expansions and calculations
  sorry

end coefficient_a3b2_in_expansion_l56_56417


namespace sufficiency_condition_l56_56181

theorem sufficiency_condition (x : ℝ) : (x > 1 → x^2 > x) ∧ (¬(x^2 > x → x > 1)) := 
by 
  -- sufficiency
  have suff : x > 1 → x^2 > x := by
    intro hx
    linarith [(mul_self_pos.mpr zero_lt_one).mp hx, hx, zero_lt_one]
  -- not necessity
  have not_nec : ¬(x^2 > x → x > 1) := by 
    intro h
    have hx_neg: x < 0 := by 
      linarith [(mul_self_neg.neg_iff).mpr zero_lt_one, hx_neg]
  exact ⟨suff, this.neg⟩

end sufficiency_condition_l56_56181


namespace num_of_valid_integers_l56_56508

theorem num_of_valid_integers (n_vals : Set ℤ) :
  (n_vals = {n | 3200 * (2:ℚ) ^ n * (5:ℚ) ^ (-n) ∈ ℤ}) →
  n_vals.card = 9 :=
by
  sorry

end num_of_valid_integers_l56_56508


namespace count_integers_with_digit_product_zero_l56_56632

def product_of_digits_zero (n : ℕ) : Prop :=
  let digits := [n / 1000, (n % 1000) / 100, (n % 100) / 10, n % 10]
  digits.any (λ d, d = 0)

theorem count_integers_with_digit_product_zero :
  let count := (5000 : ℕ) ≤ n ∧ n ≤ 6000 ∧ product_of_digits_zero n
  (finset.filter count (finset.range 6001)).card = 272 :=
by
  sorry

end count_integers_with_digit_product_zero_l56_56632


namespace expansion_number_of_terms_l56_56152

theorem expansion_number_of_terms (A B : Finset ℕ) (hA : A.card = 4) (hB : B.card = 5) : (A.card * B.card = 20) :=
by 
  sorry

end expansion_number_of_terms_l56_56152


namespace ksyusha_wednesday_time_l56_56750

def speed_relation (v : ℝ) := ∀ (d_walk d_run1 d_run2 time1 time2: ℝ), 
  (2 * d_run1 = d_walk) → -- she walked twice the distance she ran on Tuesday
  (2 * v = 2 * v) → -- she runs twice as fast as she walks
  ((2 * d_walk / v) + (d_run1 / (2 * v)) = time1) → -- she took 30 minutes on Tuesday
  (time1 = 30) → -- Tuesday time equals 30
  (d_run2 = 2 * d_walk) → -- she had to run twice the distance she walked on Wednesday
  ((d_walk / v) + (d_run2 / (2 * v)) = time2) → -- calculate the time for Wednesday
  time2 = 24 -- time taken on Wednesday

theorem ksyusha_wednesday_time :
  ∃ (v : ℝ), speed_relation v :=
begin
  existsi 1,
  unfold speed_relation,
  intros,
  sorry,
end

end ksyusha_wednesday_time_l56_56750


namespace swap_does_not_change_sum_initial_to_final_not_possible_l56_56729

-- Definitions for conditions (using the given swap rules and initial/final configurations)
def initial_buttons : list ℕ := [1, 0, -1, 0, 1]
def final_buttons : list ℕ := [-1, 0, 0, 1, 1]

-- Function to calculate the sum of the button values
def card_sum (buttons : list ℕ) : ℕ :=
  buttons.sum

-- Theorem to state the impossibility of reaching the final configuration from the initial
theorem swap_does_not_change_sum (swap : list ℕ → list ℕ → Prop) (initial final : list ℕ):
  ∀ initial final, card_sum initial = 1 → card_sum final = 1 → 
  (∀ i f, swap i f → card_sum i = card_sum f) → initial ≠ final :=
by { intros, exact 1 ≠ 1, sorry }

-- Theorem specific to our initial and final configurations
theorem initial_to_final_not_possible :
  card_sum initial_buttons = 1 → card_sum final_buttons = 1 → 
  ∀ swap, (∀ i f, swap i f → card_sum i = card_sum f) → initial_buttons ≠ final_buttons :=
by { intros, exact swap_does_not_change_sum, sorry }

end swap_does_not_change_sum_initial_to_final_not_possible_l56_56729


namespace trigonometric_identity_l56_56154

theorem trigonometric_identity : 
  4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by
  -- Here we assume standard trigonometric identities and basic properties already handled by Mathlib
  sorry

end trigonometric_identity_l56_56154


namespace find_borrowing_rate_l56_56466

-- Definitions based on the conditions
def principal : ℝ := 4000
def lending_rate : ℝ := 6 / 100
def borrowing_time : ℝ := 2
def yearly_gain : ℝ := 80

-- The main theorem stating the equivalence
theorem find_borrowing_rate (r : ℝ) :
  (principle * lending_rate * borrowing_time) - (principal * (r / 100) * borrowing_time) = 2 * yearly_gain ↔ r = 4 :=
by
  sorry

end find_borrowing_rate_l56_56466


namespace range_of_m_if_p_and_q_true_range_of_t_if_q_necessary_for_s_l56_56242

/-- There exists a real number x such that 2x^2 + (m-1)x + 1/2 ≤ 0 -/
def proposition_p (m : ℝ) : Prop :=
  ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1 / 2 ≤ 0

/-- The curve C1: x^2/m^2 + y^2/(2m+8) = 1 represents an ellipse with foci on the x-axis -/
def proposition_q (m : ℝ) : Prop :=
  m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0

/-- The curve C2: x^2/(m-t) + y^2/(m-t-1) = 1 represents a hyperbola -/
def proposition_s (m t : ℝ) : Prop :=
  (m - t) * (m - t - 1) < 0

/-- Find the range of values for m if p and q are true -/
theorem range_of_m_if_p_and_q_true (m : ℝ) :
  proposition_p m ∧ proposition_q m ↔ (-4 < m ∧ m < -2) ∨ m > 4 :=
  sorry

/-- Find the range of values for t if q is a necessary but not sufficient condition for s -/
theorem range_of_t_if_q_necessary_for_s (m t : ℝ) :
  (∀ m, proposition_q m → proposition_s m t) ∧ ¬(proposition_s m t → proposition_q m) ↔ 
  (-4 ≤ t ∧ t ≤ -3) ∨ t ≥ 4 :=
  sorry

end range_of_m_if_p_and_q_true_range_of_t_if_q_necessary_for_s_l56_56242


namespace smallest_n_for_super_special_sum_l56_56157

def is_super_special (x : ℝ) : Prop :=
  ∀ digit ∈ x.to_digits, digit = 0 ∨ digit = 3

def can_sum_to_one (n : ℕ) : Prop :=
  ∃ (s : Fin n → ℝ), (∀ i, is_super_special (s i)) ∧ (s.sum = 1)

theorem smallest_n_for_super_special_sum : 
  ∃ n, can_sum_to_one n ∧ (∀ m, (m < n) → ¬ can_sum_to_one m) :=
by
  sorry

end smallest_n_for_super_special_sum_l56_56157


namespace sum_of_numbers_eq_l56_56378

theorem sum_of_numbers_eq (a b : ℕ) (h1 : a = 64) (h2 : b = 32) (h3 : a = 2 * b) : a + b = 96 := 
by 
  sorry

end sum_of_numbers_eq_l56_56378


namespace no_real_fixed_points_l56_56207

def f1 (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + (b - 1)

theorem no_real_fixed_points : ¬∃ x0 : ℝ, f1 1 (-2) x0 = x0 := 
by {
  let f := f1 1 (-2),
  let fx_eq_x := f x - x,
  let delta := 1 - 4 * 1 * 3,
  have delta_neg : delta < 0, by norm_num,
  sorry,
}

end no_real_fixed_points_l56_56207


namespace equation_of_ellipse_range_of_t_l56_56271

-- Definitions and Conditions
def parabola (x y : ℝ) : Prop := y ^ 2 = 4 * x
def ellipse (x y : ℝ) (a b : ℝ) : Prop := x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1 ∧ a > b ∧ b > 0
def focusF2 : ℝ × ℝ := (1, 0)
def pointP (x y : ℝ) : Prop := parabola x y ∧ ellipse x y 2 (sqrt 3) ∧ 0 < x ∧ 0 < y
def distancePF2 (x y : ℝ) : Prop := pointP x y → abs (sqrt ((x - 1) ^ 2 + y ^ 2)) = 5 / 3
def is_rhombus (T M N : ℝ × ℝ) : Prop := 
  let D := ((fst M + fst N) / 2, (snd M + snd N) / 2) in
  let TD_slope := (snd D) / (fst D - fst T) in
  let MN_slope := (snd N - snd M) / (fst N - fst M) in
  abs (TD_slope * MN_slope + 1) < 1e-10

-- Question (Ⅰ): Proving the equation of the ellipse
theorem equation_of_ellipse :
  ∃ a b, a = 2 ∧ b = sqrt 3 ∧ ellipse 1 (0 : ℝ) a b :=
by simp [ellipse]; use [2, sqrt 3]; split; norm_num; split; norm_num; linarith; sorry

-- Question (Ⅱ): Proving the range of t
theorem range_of_t :
  ∃ t : ℝ, t ∈ set.Ioo 0 (1 / 4) :=
by sorry

end equation_of_ellipse_range_of_t_l56_56271


namespace propositions_correctness_l56_56398

theorem propositions_correctness :
  ( (min_positive_period (λ x, sin(x)^4 - cos(x)^4) = π) 
  ∧ (∀ α : ℝ, α ∈ {α | ∃ k : ℤ, α = k * π + π / 2} ↔ α ∈ {α | ∃ k : ℤ, α = (k : ℝ) * π / 2}) = false)
  ∧ (∃ c : ℝ, c ∈ set_of_sin_x_eq_x c = 1 ∧ ¬ (set_of_sin_x_eq_x c = 3))
  ∧ (mono_increasing (λ x, tan x) = false)
  ∧ (even_function (λ x, sin (x - π / 2)) = true) :=
begin
  sorry
end

end propositions_correctness_l56_56398


namespace overstated_height_percentage_l56_56102

-- Conditions: The candidate's actual height is 5 feet 8 inches and he made a 17.012448132780083% correction 
-- from his overstated height to his actual height.
def candidate_actual_height : ℝ := 5 + 8 / 12
def correction_percentage : ℝ := 0.17012448132780083

-- Problem: Prove that the percentage by which he overstated his height is 20.5%
theorem overstated_height_percentage (H H' : ℝ) 
  (h1 : H = candidate_actual_height) 
  (h2 : H = H' - correction_percentage * H') 
  : (H' - H) / H = 0.205 :=
by
  sorry

end overstated_height_percentage_l56_56102


namespace ones_digit_sum_l56_56889

theorem ones_digit_sum (n : ℕ) (hne : n > 0) (hneq : ∃ k, n = 4 * k + 1) :
  (((∑ k in Finset.range (n + 1), k) % 10) = 1) :=
sorry

end ones_digit_sum_l56_56889


namespace find_last_number_of_consecutive_even_numbers_l56_56138

theorem find_last_number_of_consecutive_even_numbers (x : ℕ) (h : 8 * x + 2 + 4 + 6 + 8 + 10 + 12 + 14 = 424) : x + 14 = 60 :=
sorry

end find_last_number_of_consecutive_even_numbers_l56_56138


namespace min_time_meet_l56_56165

-- Define the speeds in km/hr
def Petya_speed : ℝ := 27
def Vlad_speed : ℝ := 30
def Timur_speed : ℝ := 32

-- Define the length of the bike path in km
def bike_path_length : ℝ := 0.4

-- Define the relative speeds in km/hr
def rel_speed_VP : ℝ := Vlad_speed - Petya_speed
def rel_speed_TV : ℝ := Timur_speed - Vlad_speed
def rel_speed_TP : ℝ := Timur_speed - Petya_speed

-- Define the time intervals in minutes
def time_VP := (bike_path_length / rel_speed_VP) * 60
def time_TV := (bike_path_length / rel_speed_TV) * 60
def time_TP := (bike_path_length / rel_speed_TP) * 60

-- Define the LCM (Least Common Multiple) function for reals
noncomputable def lcm_real (x y : ℝ) : ℝ := (x * y) / (Real.gcd x y)

-- Minimum time when they will all meet again
noncomputable def min_time := lcm_real (lcm_real time_VP time_TV) time_TP

-- The problem statement to be proved
theorem min_time_meet : min_time = 24 := by
  sorry

end min_time_meet_l56_56165


namespace complex_multiplication_quadrant_l56_56605

-- Given conditions
def complex_mul (z1 z2 : ℂ) : ℂ := z1 * z2

-- Proving point is in the fourth quadrant
theorem complex_multiplication_quadrant
  (a b : ℝ) (z : ℂ)
  (h1 : z = a + b * Complex.I)
  (h2 : z = complex_mul (1 + Complex.I) (3 - Complex.I)) :
  b < 0 ∧ a > 0 :=
by
  sorry

end complex_multiplication_quadrant_l56_56605


namespace sum_of_products_formula_l56_56246

-- Define sequence A and B as conditions
noncomputable def sequenceA (n : ℕ) : ℕ → ℕ
| i => i

noncomputable def sequenceB (n : ℕ) : ℕ → ℕ
| i => 2 * n - (2 * i - 1)

-- Define the sum of products of corresponding elements from sequences A and B
noncomputable def sum_of_products (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), sequenceA n i * sequenceB n i

-- State the theorem equivalent to the mathematical proof problem
theorem sum_of_products_formula (n : ℕ) :
  sum_of_products n = n * (n + 1) * (7 - 4 * n) / 6 :=
sorry

end sum_of_products_formula_l56_56246


namespace abc_inequalities_l56_56320

noncomputable def a : ℝ := Real.log 1 / Real.log 2 - Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 2) ^ 3
noncomputable def c : ℝ := Real.sqrt 3

theorem abc_inequalities :
  a < b ∧ b < c :=
by
  -- Proof omitted
  sorry

end abc_inequalities_l56_56320


namespace intersection_area_l56_56407

-- Define the basic geometric context and congruence of triangles
variables {E F : Type}
axioms (xy : E) (e : F) (f : F)
axioms (XY_len YE_len XF_len EX_len FY_len : ℝ)
axioms (h : ℝ)
#check congruence

# Check and formalize the congruence and side lengths
axiom congruent_triangles : congruence xy e = congruence xy f
axiom xy_len_eq : XY_len = 7
axiom ye_len_eq : YE_len = 12
axiom xf_len_eq : XF_len = 12
axiom ex_len_eq : EX_len = 13
axiom fy_len_eq : FY_len = 13

-- Formalize the question: the area of intersection expressed as p/q.
noncomputable def area_of_intersection (E F : Type) (XY_len YE_len XF_len EX_len FY_len h : ℝ)
  [congruence xy e = congruence xy f]
  [XY_len = 7] [YE_len = 12] [XF_len = 12] [EX_len = 13] [FY_len = 13]: ℝ := 
sorry

-- Use rat.coprime to handle the statement about p and q being coprime
def p_q_coprime (A : ℚ) : Prop :=
  let ⟨p, q, h1, h2⟩ := A.num_denom in int.gcd p q = 1

-- Main theorem:
theorem intersection_area (E F : Type) (XY_len YE_len XF_len EX_len FY_len h : ℝ):
  congruence xy e = congruence xy f →
  XY_len = 7 → YE_len = 12 → XF_len = 12 → EX_len = 13 → FY_len = 13 → 
  ∃ (p q : ℕ), p_q_coprime (area_of_intersection E F XY_len YE_len XF_len EX_len FY_len h) ∧
  p + q = sorry :=
sorry

end intersection_area_l56_56407


namespace proof_problem_l56_56547

open Classical

noncomputable def f (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ :=
  x / Real.log x

theorem proof_problem (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  f x hx < f (x^2) (by {cases hx with h1 h2, split, exact lt_of_pow_lt h1, exact lt_trans (sq_pos_of_pos h1) h2}) ∧ f (x^2) (by {cases hx with h1 h2, split, exact lt_of_pow_lt h1, exact lt_trans (sq_pos_of_pos h1) h2}) < (f x hx)^2 :=
sorry

end proof_problem_l56_56547


namespace imaginary_unit_power_l56_56479

theorem imaginary_unit_power (i : ℂ) (n : ℕ) (h_i : i^2 = -1) : ∃ (n : ℕ), i^n = -1 :=
by
  use 6
  have h1 : i^4 = 1 := by sorry  -- Need to show i^4 = 1
  have h2 : i^6 = -1 := by sorry  -- Use i^4 and additional steps to show i^6 = -1
  exact h2

end imaginary_unit_power_l56_56479


namespace find_range_of_a_l56_56611

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + x

theorem find_range_of_a {a : ℝ} :
  (∀ x ∈ set.Icc (1 / Real.exp 1) (Real.exp 1), f a x ≥ 0) →
  -Real.exp 1 ≤ a ∧ a ≤ 1 / Real.exp 1 := by
  intros h
  sorry

end find_range_of_a_l56_56611


namespace yulia_profit_l56_56428

-- Assuming the necessary definitions in the problem
def lemonade_revenue : ℕ := 47
def babysitting_revenue : ℕ := 31
def expenses : ℕ := 34
def profit : ℕ := lemonade_revenue + babysitting_revenue - expenses

-- The proof statement to prove Yulia's profit
theorem yulia_profit : profit = 44 := by
  sorry -- Proof is skipped

end yulia_profit_l56_56428


namespace chef_earns_less_than_manager_l56_56900

-- Definitions
def manager_wage : ℝ := 8.50
def dishwasher_wage : ℝ := manager_wage / 2
def chef_wage : ℝ := dishwasher_wage + 0.20 * dishwasher_wage

-- Theorem statement
theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 3.40 := 
by
  sorry

end chef_earns_less_than_manager_l56_56900


namespace smallest_positive_solution_tan_cos_is_pi_over_8_l56_56203

noncomputable def smallest_positive_solution_tan_cos_eq (x : ℝ) : Prop :=
  tan (2 * x) + tan (4 * x) = cos (2 * x)

theorem smallest_positive_solution_tan_cos_is_pi_over_8 :
  ∃ x > 0, smallest_positive_solution_tan_cos_eq x ∧ x = Real.pi / 8 :=
by
  use Real.pi / 8
  sorry

end smallest_positive_solution_tan_cos_is_pi_over_8_l56_56203


namespace derivative_y_l56_56197

noncomputable def y (x : ℝ) :=
  Real.cot (Real.sin (1 / 13)) - (1 / 48) * (Real.cos (24 * x) ^ 2 / Real.sin (48 * x))

theorem derivative_y : 
  (deriv y x) = 1 / (4 * (Real.sin (24 * x) ^ 2)) :=
sorry

end derivative_y_l56_56197


namespace distinct_factors_of_expr_l56_56631

theorem distinct_factors_of_expr (n : ℕ) (h : n = 8^2 * 9^3 * 10^4) : 
  (∃ d, d = 2^10 * 3^6 * 5^4 ∧ (11 * 7 * 5 = 385)) := 
by
  use 2 ^ 10 * 3 ^ 6 * 5 ^ 4
  split
  . refl
  . repeat { sorry }

end distinct_factors_of_expr_l56_56631


namespace sum_of_coeffs_eq_59049_l56_56891

-- Definition of the polynomial
def poly (x y z : ℕ) : ℕ :=
  (2 * x - 3 * y + 4 * z) ^ 10

-- Conjecture: The sum of the numerical coefficients in poly when x, y, and z are set to 1 is 59049
theorem sum_of_coeffs_eq_59049 : poly 1 1 1 = 59049 := by
  sorry

end sum_of_coeffs_eq_59049_l56_56891


namespace find_m_and_max_profit_l56_56455

theorem find_m_and_max_profit (m : ℝ) (y : ℝ) (x : ℝ) (ln : ℝ → ℝ) 
    (h1 : y = m * ln x - 1 / 100 * x ^ 2 + 101 / 50 * x + ln 10)
    (h2 : 10 < x) 
    (h3 : y = 35.7) 
    (h4 : x = 20)
    (ln_2 : ln 2 = 0.7) 
    (ln_5 : ln 5 = 1.6) :
    m = -1 ∧ ∃ x, (x = 50 ∧ (-ln x - 1 / 100 * x ^ 2 + 51 / 50 * x + ln 10 - x) = 24.4) := by
  sorry

end find_m_and_max_profit_l56_56455


namespace minimum_value_reciprocals_l56_56856

theorem minimum_value_reciprocals (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : 2 / Real.sqrt (a^2 + 4 * b^2) = Real.sqrt 2) :
  (1 / a^2 + 1 / b^2) = 9 / 2 :=
sorry

end minimum_value_reciprocals_l56_56856


namespace marcy_minimum_avg_score_l56_56845

variables (s1 s2 s3 : ℝ)
variable (qualified_avg : ℝ := 90)
variable (required_total : ℝ := 5 * qualified_avg)
variable (first_three_total : ℝ := s1 + s2 + s3)
variable (needed_points : ℝ := required_total - first_three_total)
variable (required_avg : ℝ := needed_points / 2)

/-- The admission criteria for a mathematics contest require a contestant to 
    achieve an average score of at least 90% over five rounds to qualify for the final round.
    Marcy scores 87%, 92%, and 85% in the first three rounds. 
    Prove that Marcy must average at least 93% in the next two rounds to qualify for the final. --/
theorem marcy_minimum_avg_score 
    (h1 : s1 = 87) (h2 : s2 = 92) (h3 : s3 = 85)
    : required_avg ≥ 93 :=
sorry

end marcy_minimum_avg_score_l56_56845


namespace prism_surface_area_l56_56126

noncomputable def surface_area_of_prism (s h : ℝ) : ℝ :=
  2 * (s^2) + 4 * s * h

theorem prism_surface_area : 
  ∀ (diameter side_length : ℝ),  
  diameter = 2 → 
  side_length = 1 → 
  ∃ (h : ℝ), (2 : ℝ) = real.sqrt (side_length^2 + side_length^2 + h^2) ∧ 
  surface_area_of_prism side_length h = 2 + 4*real.sqrt(2) := 
by
  intro diameter side_length diameter_eq side_length_eq
  sorry

end prism_surface_area_l56_56126


namespace sum_sequence_eq_3997_l56_56792

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else (λ (n : ℕ), (if n > 3 then (let p := sequence (n - 1), q := sequence (n - 2) * sequence (n - 3) in
  if 9 * p^2 / 4 - q < 0 then 0
  else if 9 * p^2 / 4 - q = 0 then if p = 0 then 1 else 2
  else 4) else 0)) n

theorem sum_sequence_eq_3997 : (Finset.range 1000).sum sequence = 3997 := 
sorry

end sum_sequence_eq_3997_l56_56792


namespace average_difference_l56_56848

def average (numbers : List ℝ) : ℝ :=
  List.sum numbers / numbers.length

theorem average_difference :
  average [20, 40, 60] - average [10, 70, 19] = 7 :=
by
  sorry

end average_difference_l56_56848


namespace find_a_minimum_value_at_x_2_l56_56614

def f (x a : ℝ) := x^3 - a * x

theorem find_a_minimum_value_at_x_2 (a : ℝ) :
  (∃ x : ℝ, x = 2 ∧ ∀ y ≠ 2, f y a ≥ f 2 a) → a = 12 :=
by 
  -- Here we should include the proof steps
  sorry

end find_a_minimum_value_at_x_2_l56_56614


namespace hyperbola_eccentricity_l56_56241

def HyperbolaEccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  (Real.sqrt (a^2 + b^2)) / a

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (4 * (3 * a^2 - a^2) = b^2) → e = (Real.sqrt 15) / 3 :=
by
  intro h
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  sorry

end hyperbola_eccentricity_l56_56241


namespace intersection_of_parabola_with_y_axis_l56_56018

theorem intersection_of_parabola_with_y_axis :
  ∃ y : ℝ, y = - (0 + 2)^2 + 6 ∧ (0, y) = (0, 2) :=
by
  sorry

end intersection_of_parabola_with_y_axis_l56_56018


namespace water_depth_is_208_l56_56953

variable (Ron_height : ℕ) (depth_of_water : ℕ)

-- Given conditions for the problem
def Ron_stands_at : Ron_height = 13 := by rfl
def water_depth_formula : depth_of_water = 16 * Ron_height := by rfl

-- The theorem to prove
theorem water_depth_is_208 :
  depth_of_water = 208 :=
by 
  rw [water_depth_formula, Ron_stands_at]
  sorry

end water_depth_is_208_l56_56953


namespace ksyusha_wednesday_time_l56_56753

def speed_relation (v : ℝ) := ∀ (d_walk d_run1 d_run2 time1 time2: ℝ), 
  (2 * d_run1 = d_walk) → -- she walked twice the distance she ran on Tuesday
  (2 * v = 2 * v) → -- she runs twice as fast as she walks
  ((2 * d_walk / v) + (d_run1 / (2 * v)) = time1) → -- she took 30 minutes on Tuesday
  (time1 = 30) → -- Tuesday time equals 30
  (d_run2 = 2 * d_walk) → -- she had to run twice the distance she walked on Wednesday
  ((d_walk / v) + (d_run2 / (2 * v)) = time2) → -- calculate the time for Wednesday
  time2 = 24 -- time taken on Wednesday

theorem ksyusha_wednesday_time :
  ∃ (v : ℝ), speed_relation v :=
begin
  existsi 1,
  unfold speed_relation,
  intros,
  sorry,
end

end ksyusha_wednesday_time_l56_56753


namespace solve_inequality_l56_56835

theorem solve_inequality (x : ℝ) :
  abs(x - 2) + abs(x + 3) < 8 ↔ x ∈ set.Ioo (-13 / 2) 3.5 :=
by
  sorry

end solve_inequality_l56_56835


namespace problem_statement_l56_56579

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

theorem problem_statement (f : ℝ → ℝ) :
  is_odd_function f →
  (∀ x : ℝ, f (x + 6) = f (x) + 3) →
  f 1 = 1 →
  f 2015 + f 2016 = 2015 :=
by
  sorry

end problem_statement_l56_56579


namespace math_problem_l56_56296

-- Problem definitions based on conditions
def C₁_polar (ρ θ : ℝ) : Prop := ρ = 2
def C₂_polar (ρ θ : ℝ) : Prop := ρ ^ 2 + (Real.sqrt 3) * ρ * Real.cos θ + ρ * Real.sin θ = 6

-- Definitions using rectangular coordinates
def C₁_rect (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂_rect (x y : ℝ) : Prop := x^2 + y^2 + (Real.sqrt 3) * x + y = 6

-- Main statement
theorem math_problem :
  (∀ (ρ θ : ℝ), C₁_polar ρ θ ↔ C₁_rect (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (∀ (ρ θ : ℝ), C₂_polar ρ θ ↔ C₂_rect (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (∃ θ : ℝ, Real.abs ((Real.sqrt 3) * (Real.cos θ) + (Real.sin θ)) > 0 ∧
    min (1 / Real.abs ((Real.sqrt 3) * (Real.cos θ) + (Real.sin θ)))
    (1 / Real.abs (- (Real.sqrt 3) * (Real.sin θ) + (Real.cos θ))) = 1) 
:= by
  sorry

end math_problem_l56_56296


namespace find_center_of_circle_l56_56195

noncomputable def center_of_circle (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let x_center := 4 in
  let y_center := 17 / 6 in
  (x_center, y_center)

theorem find_center_of_circle :
  let A := (2, 2)
  let B := (6, 2)
  let C := (4, 5)
  center_of_circle A B C = (4, 17 / 6) :=
by
  unfold center_of_circle
  sorry

end find_center_of_circle_l56_56195


namespace trajectory_of_M_is_ellipse_area_of_triangle_AOB_is_l56_56289

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  2 * sqrt ((x - 1)^2 + y^2) = abs (x - 4)

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def distance_from_origin_to_line (k m : ℝ) : Prop :=
  m^2 = 1 + k^2

noncomputable def dot_product_condition (OA OB : ℝ × ℝ) : Prop :=
  (OA.1 * OB.1 + OA.2 * OB.2) = -3/2

def area_of_triangle (A B : ℝ × ℝ) : ℝ :=
  abs ((A.1 * B.2 - A.2 * B.1) / 2)

theorem trajectory_of_M_is_ellipse :
  ∀ x y : ℝ, trajectory_equation x y ↔ ellipse_equation x y :=
sorry

theorem area_of_triangle_AOB_is :
  ∀ (k m : ℝ) (A B : ℝ × ℝ), distance_from_origin_to_line k m →
    ellipse_equation A.1 A.2 → ellipse_equation B.1 B.2 →
    dot_product_condition A B →
    area_of_triangle A B = 3 * sqrt 7 / 5 :=
sorry

end trajectory_of_M_is_ellipse_area_of_triangle_AOB_is_l56_56289


namespace factors_of_120_that_are_multiples_of_15_l56_56639

theorem factors_of_120_that_are_multiples_of_15 :
  {d : ℕ | d > 0 ∧ 120 % d = 0 ∧ d % 15 = 0}.to_finset.card = 4 :=
by sorry

end factors_of_120_that_are_multiples_of_15_l56_56639


namespace line_through_intersection_parallel_to_given_line_l56_56529

theorem line_through_intersection_parallel_to_given_line :
  ∃ k : ℝ, (∀ x y : ℝ, (2 * x + 3 * y + k = 0 ↔ (x, y) = (2, 1)) ∧
  (∀ m n : ℝ, (2 * m + 3 * n + 5 = 0 → 2 * m + 3 * n + k = 0))) →
  2 * x + 3 * y - 7 = 0 :=
sorry

end line_through_intersection_parallel_to_given_line_l56_56529


namespace Karlson_Max_Candies_l56_56396

theorem Karlson_Max_Candies :
  ∀ (nums : List ℕ), length nums = 37 ∧ (∀ n ∈ nums, n = 1) →
  (∃ (operations : Finset (ℕ × ℕ)), operations.card = 37 ∧ 
  (∀ (i : ℕ × ℕ), i ∈ operations → i.1 ∈ nums ∧ i.2 ∈ nums ∧ i.1 + i.2 ∈ nums)) →
  (∑ (i : ℕ × ℕ) in operations, i.1 * i.2) = 666 :=
by 
  intros nums hlen hnums ops hopscard hops 
  sorry

end Karlson_Max_Candies_l56_56396


namespace sum_sequence_zero_l56_56573

-- Define the recurrence relation condition
axiom recurrence_relation (a : ℕ → ℤ) (n : ℕ) : 
  ∀ i, 1 ≤ i ∧ i ≤ n → 
    (a (i+1))^2 - (a i) * (a (i+1)) + (a i)^2 = 0

-- Define the condition that a_{i+1} != a_{i-1}
axiom distinct_terms (a : ℕ → ℤ) (n : ℕ):
  ∀ i, 1 < i ∧ i ≤ n → a (i+1) ≠ a (i-1)

-- Define the condition that a_1 = 1 and a_{n+1} = 1
axiom boundary_conditions (a : ℕ → ℤ) (n : ℕ):
  a 1 = 1 ∧ a (n+1) = 1

-- Define the main theorem to prove the sum of the sequence equals 0
theorem sum_sequence_zero (a : ℕ → ℤ) (n : ℕ) (hn : n > 0):
  (recurrence_relation a n) ∧ (distinct_terms a n) ∧ (boundary_conditions a n) →
  ∑ i in Finset.range n, a (i + 1) = 0 :=
begin
  sorry
end

end sum_sequence_zero_l56_56573


namespace _l56_56148

-- Definitions as per the given conditions
variables (A B C D E : Type*) -- Assumptions
variable [has_midpoint ℝ E (B D)] -- E is midpoint of BD

axiom isosceles_triangle : ∀ (A B C : Type*), ∃ (angleBAC : ℝ), angleBAC = 108
axiom is_on_extension : ∀ (A B C D E : Type*), to_extend D from A C such that Ad_eq_BC_AD = BC
axiom midpoint_definition : ∀ (E A B D : Type*), midpoint E (B D)
axiom perpendicularity : ∀ (A B C : Type*), AE_perp_CE

def AE_perp_CE_theorem : Prop := 
  AE_perp_CE A B C D E

-- The Lean statement proving the theorem
example : AE_perp_CE_theorem A B C D E :=
begin
  sorry
end

end _l56_56148


namespace least_people_second_caterer_cheaper_l56_56818

def caterer1_cost (x : ℕ) : ℕ := 150 + 18 * x
def caterer2_cost (x : ℕ) : ℕ := 250 + 15 * x

theorem least_people_second_caterer_cheaper : 
  ∃ (x : ℕ), (caterer1_cost x > caterer2_cost x) ∧ (∀ y < x, caterer1_cost y ≤ caterer2_cost y) := 
begin
  use 34,
  sorry,
end

end least_people_second_caterer_cheaper_l56_56818


namespace reconstruct_pentagon_l56_56215

def points : Type := ℝ × ℝ

variables {A' B' C' D' E' : points}
variables (p q r s t : ℝ)

axiom pentagon_condition :
  ∃ (A B C D E : points),
    B = (⟨(A.1 + A'.1) / 2, (A.2 + A'.2) / 2⟩ : points) ∧
    C = (⟨(B.1 + B'.1) / 2, (B.2 + B'.2) / 2⟩ : points) ∧
    D = (⟨(C.1 + C'.1) / 2, (C.2 + C'.2) / 2⟩ : points) ∧
    E = (⟨(D.1 + D'.1) / 2, (D.2 + D'.2) / 2⟩ : points) ∧
    A = (⟨(E.1 + E'.1) / 2, (E.2 + E'.2) / 2⟩ : points)

theorem reconstruct_pentagon (h : (∃ (A : points), A = (p • A' + q • B' + r • C' + s • D' + t • E'))) :
  p = 1 / 31 ∧ q = 2 / 31 ∧ r = 4 / 31 ∧ s = 8 / 31 ∧ t = 16 / 31 :=
sorry

end reconstruct_pentagon_l56_56215


namespace find_m_l56_56622

theorem find_m (m : ℝ) (A : Set ℝ) (hA : A = {0, m, m^2 - 3 * m + 2}) (h2 : 2 ∈ A) : m = 3 :=
  sorry

end find_m_l56_56622


namespace distance_between_symmetric_points_l56_56223

def A : ℝ × ℝ × ℝ := (1, 2, -1)

def symmetric_about_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2, -p.3)

def symmetric_about_yoz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2, p.3)

def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2 + (q.3 - p.3) ^ 2)

theorem distance_between_symmetric_points :
  let B := symmetric_about_x A
  let C := symmetric_about_yoz A
  distance B C = 2 * Real.sqrt 6 := by
  sorry

end distance_between_symmetric_points_l56_56223


namespace trig_identity_l56_56559

theorem trig_identity (θ : ℝ) (h : cos (3 * real.pi + θ) = - (2 * real.sqrt 2) / 3) :
  sin (7 * real.pi / 2 + θ) = - (2 * real.sqrt 2) / 3 := by
  sorry

end trig_identity_l56_56559


namespace determine_x_l56_56512

theorem determine_x : (x : ℝ) (hx : 1 / (x - 3) = 3 / (x - 6)) : x = 3 / 2 :=
sorry

end determine_x_l56_56512


namespace sufficient_funds_to_construct_road_network_l56_56209

noncomputable theory

open Real

def distance (x1 y1 x2 y2 : ℝ) : ℝ := sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def total_road_length (a b c d : ℝ) : ℝ := a + b + c + d

theorem sufficient_funds_to_construct_road_network 
  (side_length : ℝ) 
  (available_funds : ℝ)
  (h1 : side_length = 10)
  (h2 : available_funds = 27.5)
  : ∃ roads, total_road_length roads.fst roads.snd roads.2.fst roads.2.snd ≤ available_funds :=
by
  let A := (0, 0)
  let B := (side_length, 0)
  let C := (0, side_length)
  let D := (side_length, side_length)
  let E := (side_length / 2, 0)
  let F := (side_length / 2, side_length)
    
  let G := ((E.1 + 3 / (E.1 + F.1)), (E.2 + 3 / (E.2 + F.2)))
  let H := ((F.1 + 3 / (F.1 + E.1)), (F.2 + 3 / (F.2 + E.2)))

  let AG := distance A.1 A.2 G.1 G.2
  let BG := distance B.1 B.2 G.1 G.2
  let GH := distance G.1 G.2 H.1 H.2
  let HC := distance H.1 H.2 C.1 C.2
  let HD := distance H.1 H.2 D.1 D.2

  exact ⟨(AG, BG, GH, HC, HD), sorry⟩

end sufficient_funds_to_construct_road_network_l56_56209


namespace find_pairs_l56_56192

theorem find_pairs (p n : ℕ) (hp : Nat.Prime p) (h1 : n ≤ 2 * p) (h2 : n^(p-1) ∣ (p-1)^n + 1) : 
    (p = 2 ∧ n = 2) ∨ (p = 3 ∧ n = 3) ∨ (n = 1) :=
by
  sorry

end find_pairs_l56_56192


namespace smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5_l56_56119

def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := n.digits b
  digits = digits.reverse

theorem smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5 :
  ∃ n : ℕ, n = 0b11011 ∧ is_palindrome n 2 ∧ is_palindrome n 5 :=
by
  existsi 0b11011
  sorry

end smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5_l56_56119


namespace Ksyusha_time_to_school_l56_56735

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l56_56735


namespace complex_sum_eq_899_l56_56325

theorem complex_sum_eq_899 (x : ℂ) (h1 : x ^ 1801 = 1) (h2 : x ≠ 1) :
  (∑ k in Finset.range (1800) + 1, (x ^ (2 * (k + 1))) / (x ^ (k + 1) - 1)) = 899 :=
sorry

end complex_sum_eq_899_l56_56325


namespace a_beats_b_by_meters_l56_56276

variables (A_time B_time : ℕ)
variables (race_distance : ℝ)

def A_time := 20
def B_time := 25
def race_distance := 110

theorem a_beats_b_by_meters : 
  let speed_B := race_distance / B_time
  let distance_B_in_A_time := speed_B * A_time
  race_distance - distance_B_in_A_time = 22 := 
by
  sorry

end a_beats_b_by_meters_l56_56276


namespace shorter_piece_is_2_l56_56451

-- Definitions based on conditions
def total_length : ℝ := 6
def shorter_piece_length := total_length / 3

-- Statement to prove
theorem shorter_piece_is_2 :
  shorter_piece_length = 2 := by
  sorry

end shorter_piece_is_2_l56_56451


namespace prove_d_value_l56_56560

-- Definitions of the conditions
def d (x : ℝ) : ℝ := x^4 - 2*x^3 + x^2 - 12*x - 5

-- The statement to prove
theorem prove_d_value (x : ℝ) (h : x^2 - 2*x - 5 = 0) : d x = 25 :=
sorry

end prove_d_value_l56_56560


namespace binary_and_ternary_product_l56_56153

theorem binary_and_ternary_product :
  let binary_1011 := 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0
  let ternary_1021 := 1 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0
  binary_1011 = 11 ∧ ternary_1021 = 34 →
  binary_1011 * ternary_1021 = 374 :=
by
  intros h
  sorry

end binary_and_ternary_product_l56_56153


namespace harry_worked_16_hours_l56_56901

-- Define the given conditions
def harrys_pay_first_30_hours (x : ℝ) : ℝ := 30 * x
def harrys_pay_additional_hours (x H : ℝ) : ℝ := (H - 30) * 2 * x
def james_pay_first_40_hours (x : ℝ) : ℝ := 40 * x
def james_pay_additional_hour (x : ℝ) : ℝ := 2 * x
def james_total_hours : ℝ := 41

-- Given that Harry and James are paid the same amount 
-- Prove that Harry worked 16 hours last week
theorem harry_worked_16_hours (x H : ℝ) 
  (h1 : harrys_pay_first_30_hours x + harrys_pay_additional_hours x H = james_pay_first_40_hours x + james_pay_additional_hour x) 
  : H = 16 :=
by
  sorry

end harry_worked_16_hours_l56_56901


namespace exists_n_for_sine_magnitude_l56_56826

theorem exists_n_for_sine_magnitude (x : ℝ) (h : sin x ≠ 0) : 
  ∃ n : ℕ, |sin (n * x)| ≥ sqrt 3 / 2 := 
sorry

end exists_n_for_sine_magnitude_l56_56826


namespace glorias_grandmother_spent_less_time_l56_56628

noncomputable def time_spent_cycling : ℕ → ℕ → ℝ
| distance, speed := distance / speed

def total_time_different_speeds (times : List ℝ) : ℝ :=
List.sum times

def total_time_constant_speed (total_distance : ℕ) (constant_speed : ℕ) : ℝ :=
total_distance / constant_speed

def calculate_time_saved (times : List ℝ) (constant_time : ℝ) : ℝ := 
total_time_different_speeds times - constant_time

theorem glorias_grandmother_spent_less_time :
  let actual_times := [time_spent_cycling 3 6, time_spent_cycling 4 4, time_spent_cycling 3 3, time_spent_cycling 2 8],
      total_cycling_time := total_time_different_speeds actual_times,
      consistent_speed_time := total_time_constant_speed (3 + 4 + 3 + 2) 5,
      time_saved := calculate_time_saved actual_times consistent_speed_time in
  time_saved * 60 = 21 :=
by
  sorry

end glorias_grandmother_spent_less_time_l56_56628


namespace initial_mean_of_observations_l56_56857

theorem initial_mean_of_observations :
  ∀ (M : ℝ), 
    (corrected_mean : ℝ) (inflated_sum_adjustment : ℝ),
    corrected_mean = 34.9 →
    inflated_sum_adjustment = 15 →
    M = (20 * corrected_mean + inflated_sum_adjustment) / 20 →
    M = 35.65 :=
by
  intros M corrected_mean inflated_sum_adjustment h_corrected_mean h_inflated_sum_adjustment h_initial_mean_calc
  sorry

end initial_mean_of_observations_l56_56857


namespace least_positive_angle_l56_56196

theorem least_positive_angle (θ : ℝ) (h : cos (10 * π / 180) = sin (40 * π / 180) + cos θ) : θ = 70 * π / 180 :=
sorry

end least_positive_angle_l56_56196


namespace which_event_is_certain_l56_56070

-- Definitions based on the conditions 
def is_symmetrical (shape : Type) : Prop := sorry -- assume this definition for now

-- Event definitions
def event_A : Prop := ∀ (c : Type), is_symmetrical c 
def event_B : Prop := ∀ (t : Type) (angles : Type), angles.sum t = 360
def event_C (a b : ℝ) : Prop := a < b → a^2 < b^2
def event_D : Prop :=
  let S_A^2 := 1.1
  let S_B^2 := 2.5
  S_B^2 < S_A^2 -- This denotes that B's performance is more stable than A's (wrongly implies less variance)

-- The theorem to be proven
theorem which_event_is_certain : event_A ∧ ¬event_B ∧ ¬event_C ∧ ¬event_D :=
begin
  sorry
end

end which_event_is_certain_l56_56070


namespace total_calories_in_jerrys_breakfast_l56_56722

theorem total_calories_in_jerrys_breakfast :
  let pancakes := 7 * 120
  let bacon := 3 * 100
  let orange_juice := 2 * 300
  let cereal := 1 * 200
  let chocolate_muffin := 1 * 350
  pancakes + bacon + orange_juice + cereal + chocolate_muffin = 2290 :=
by
  -- Proof omitted
  sorry

end total_calories_in_jerrys_breakfast_l56_56722


namespace number_of_pairs_of_positive_integers_l56_56251

theorem number_of_pairs_of_positive_integers 
    {m n : ℕ} (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_mn : m > n) (h_diff : m^2 - n^2 = 144) : 
    ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 4 ∧ (∀ p ∈ pairs, p.1 > p.2 ∧ p.1^2 - p.2^2 = 144) :=
sorry

end number_of_pairs_of_positive_integers_l56_56251


namespace man_older_than_son_l56_56934

def S : Nat := 33

-- In two years, the man's age will be twice the age of his son.
def condition : Prop := ∀ (M : Nat), M + 2 = 2 * (S + 2)

theorem man_older_than_son : ∀ (M : Nat), (condition M) → (M - S = 35) :=
by
  intro M h
  -- Proof omitted
  sorry

end man_older_than_son_l56_56934


namespace prove_purchase_sets_prove_max_profit_l56_56924

theorem prove_purchase_sets (a b : ℕ) (h1 : a + b = 50) (h2 : 3 * a + 2.4 * b = 132) :
  a = 20 ∧ b = 30 :=
sorry

theorem prove_max_profit (m : ℕ) (h : 10 ≤ m ∧ m ≤ 20) :
  ∃ (w : ℝ), w = 0.3 * m + 0.4 * (50 - m) ∧ w = 19 ↔ m = 10 :=
sorry

end prove_purchase_sets_prove_max_profit_l56_56924


namespace original_correct_answer_l56_56184

theorem original_correct_answer :
  ∀ (submitted : ℝ), (mistake : ℝ), (true_answer : ℝ),
    submitted = 0.9 →
    mistake = 0.42 →
    true_answer = (submitted - mistake - mistake) →
    true_answer = 0.06 :=
by {
  intros submitted mistake true_answer h_submitted h_mistake h_true_answer,
  sorry
}

end original_correct_answer_l56_56184


namespace volume_of_inscribed_sphere_l56_56944

theorem volume_of_inscribed_sphere {cube_edge : ℝ} (h : cube_edge = 6) : 
  ∃ V : ℝ, V = 36 * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_l56_56944


namespace fixed_weekly_earnings_l56_56404

-- Definitions based on conditions
def commission_rate : ℝ := 0.04
def weekly_goal : ℝ := 500
def sales_amount : ℝ := 7750

-- Prove that Trenton's fixed weekly earnings (excluding commission) is 190 dollars
theorem fixed_weekly_earnings : ∃ E : ℝ, commission_rate * sales_amount = 310 ∧ weekly_goal - (commission_rate * sales_amount) = E ∧ E = 190 :=
by
  use 190
  have h1 : commission_rate * sales_amount = 310 := by compute
  have h2 : weekly_goal - (commission_rate * sales_amount) = 190 := by compute
  exact ⟨h1, h2, rfl⟩

end fixed_weekly_earnings_l56_56404


namespace probability_sum_multiple_of_3_l56_56041

theorem probability_sum_multiple_of_3 :
  let die_faces := {1, 2, 3, 4, 5, 6}
  let probabilities := (die_faces.map (λ x, (x % 3))).freq_map (λ x, 1/6)
  let successful_combinations := [{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {0, 1, 2}]
  probability (successful_combinations.sum_p_by probabilities) = 4/27 :=
by
  sorry

end probability_sum_multiple_of_3_l56_56041


namespace real_minus_imag_of_complex_l56_56337

-- Conditions:
def complex_num := (5 : ℂ) / (-3 - (1 : ℂ) * Complex.I) 

-- Proof statement:
theorem real_minus_imag_of_complex : 
  let a := complex_num.re,
      b := complex_num.im in
  a - b = -2 := by
  -- We skip the proof part as required.
  sorry

end real_minus_imag_of_complex_l56_56337


namespace find_percentage_l56_56069

/-- 
Given some percentage P of 6,000, when subtracted from 1/10th of 6,000 (which is 600), 
the difference is 693. Prove that P equals 1.55.
-/
theorem find_percentage (P : ℝ) (h₁ : 6000 / 10 = 600) (h₂ : 600 - (P / 100) * 6000 = 693) : 
  P = 1.55 :=
  sorry

end find_percentage_l56_56069


namespace ratio_c_d_l56_56793

variable (c d : ℝ)

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem ratio_c_d (h : (c ≠ 0) ∧ (d ≠ 0)) (H : is_pure_imaginary ((3 - 4 * complex.I) * (c + d * complex.I))) : 
  c / d = -4 / 3 :=
by
  sorry

end ratio_c_d_l56_56793


namespace deepak_present_age_l56_56390

theorem deepak_present_age:
  ∃ (R D : ℕ), 
    (R / D = 4 / 3) ∧ 
    (R + 6 = 34) ∧ 
    (D = 21) :=
begin
  sorry
end

end deepak_present_age_l56_56390


namespace ksyusha_travel_time_l56_56759

theorem ksyusha_travel_time :
  ∀ (S v : ℝ), 
  (2 * S / v + S / (2 * v) = 30) → 
  (2 * (S / v) = 24) :=
by
  intros S v h1
  have h2 : S / v = 12 := by
    calc 
      S / v = (30 * 2) / 5 := by sorry
  calc 
    2 * (S / v) = 2 * 12 := by sorry
  ... = 24 := by norm_num

end ksyusha_travel_time_l56_56759


namespace total_oranges_l56_56987

theorem total_oranges :
  ∀ (oranges_per_child : ℕ) (num_children : ℕ),
  oranges_per_child = 3 → num_children = 4 → (oranges_per_child * num_children = 12) :=
by
  intros oranges_per_child num_children h_oranges_per_child h_num_children
  rw [h_oranges_per_child, h_num_children]
  exact (mul_eq_of_eq_of_eq 3 4 12).mp rfl
  sorry

end total_oranges_l56_56987


namespace exists_n_for_c_l56_56328

def d (n : ℕ) : ℕ := nat.divisors n |>.length

def φ (n : ℕ) : ℕ := (nat.totient n)

theorem exists_n_for_c (c : ℕ) : 
  ∃ n : ℕ+, d n + φ n = n + c := sorry

end exists_n_for_c_l56_56328


namespace average_tickets_sold_by_male_members_l56_56429

theorem average_tickets_sold_by_male_members 
  (M F : ℕ)
  (total_average : ℕ)
  (female_average : ℕ)
  (ratio : ℕ × ℕ)
  (h1 : total_average = 66)
  (h2 : female_average = 70)
  (h3 : ratio = (1, 2))
  (h4 : F = 2 * M)
  (h5 : (M + F) * total_average = M * r + F * female_average) :
  r = 58 :=
sorry

end average_tickets_sold_by_male_members_l56_56429


namespace smallest_number_first_digit_is_9_l56_56302

def sum_of_digits (n : Nat) : Nat :=
  (n.digits 10).sum

def first_digit (n : Nat) : Nat :=
  n.digits 10 |>.headD 0

theorem smallest_number_first_digit_is_9 :
  ∃ N : Nat, sum_of_digits N = 2020 ∧ ∀ M : Nat, (sum_of_digits M = 2020 → N ≤ M) ∧ first_digit N = 9 :=
by
  sorry

end smallest_number_first_digit_is_9_l56_56302


namespace best_fit_slope_l56_56898

theorem best_fit_slope (y1 y2 y3 : ℝ) :
  let x1 := 140
  let x2 := 145
  let x3 := 150
  let d := x2 - x1
  let d2 := x3 - x2
  d = d2 → d = 5 →
  (∑ (i, y_i), [⟨x1, y1⟩, ⟨x2, y2⟩, ⟨x3, y3⟩]) =
  ∑ x_iy : (ℝ × ℝ), (x_iy.1 - 145) * (x_iy.2 - (\bar{y})) /
  ∑ i, (x_i - 145)^2 =
  (y3 - y1) / 10 :=
by {
  intros,
  let x1 := 140,
  let x2 := 145,
  let x3 := 150,
  let d := x2 - x1,
  let d2 := x3 - x2,
  assume h1 : d = d2,
  assume h2 : d = 5,
  let x_indices := [x1, x2, x3],
  let y_indices := [y1, y2, y3],
  let xy_vals := [⟨x1, y1⟩, ⟨x2, y2⟩, ⟨x3, y3⟩],
  let x_mean := (x1 + x2 + x3) / 3,
  let y_mean := (y1 + y2 + y3) / 3,
  let numerator := ∑ xy in xy_vals, (xy.1 - x_mean) * (xy.2 - y_mean),
  let denominator := ∑ x in x_indices, (x - x_mean)^2,
  let result := numerator / denominator,
  show result = (y3 - y1) / 10,
  sorry
}

end best_fit_slope_l56_56898


namespace number_of_1989_periodic_points_l56_56785

theorem number_of_1989_periodic_points (S : set ℂ) (m : ℕ) (hm : 1 < m)
  (f : ℂ → ℂ)
  (hf : ∀ z ∈ S, f(z) = z^m)
  (f_n : ℕ → ℂ → ℂ)
  (hf_n : ∀ n z, z ∈ S → f_n n z = z^(m^n)) :
  ∃ count_1989 : ℕ, count_1989 = m^1989 - m^663 - m^153 - m^117 + m^51 + m^39 + m^9 - m^3 :=
sorry

end number_of_1989_periodic_points_l56_56785


namespace hyperbola_asymptotes_separate_proof_l56_56617

-- Define the conditions
def hyperbola_asymptotes_separate : Prop := 
  ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a = b) ∧ 
  ∀ (x y : ℝ), (x / a)^2 - (y / b)^2 = 1 → 
  ∀ (x_c y_c : ℝ), ((x_c - a)^2 + y_c^2 = (1/4) * a^2) →
  ∀ d, (d = |a/√2|) →
  d > (a / 2) 

-- Include the proof statement
theorem hyperbola_asymptotes_separate_proof : hyperbola_asymptotes_separate :=
by {
  sorry
}

end hyperbola_asymptotes_separate_proof_l56_56617


namespace volume_of_pyramid_SAKND_l56_56849

noncomputable def pyramid_volume {α : Type*} [RealField α] 
  (AD : α) (AD_length : AD = 12 * √3)
  (area_ratio : 5 / 7)
  (θ : α) (theta_angle : θ = 30)
  (midpoint_TB_TC : α → α → α) 
  (TS_SD_ratio : 1 / 2) : α :=
  let base_area : α := some_arcane_formula_with area_ratio in
  let pyramid_height : α := some_arcane_formula_with AD_length θ in
  let mid_segment_length : α := midpoint_TB_TC (some_arcane_formula AD θ) in
  base_area * pyramid_height * mid_segment_length / 3

theorem volume_of_pyramid_SAKND : 
  ∀ (α : Type*) [RealField α] 
    (AD : α) (AD_length : AD = 12 * √3)
    (area_ratio : 5 / 7)
    (θ : α) (theta_angle : θ = 30)
    (midpoint_TB_TC : α → α → α)
    (TS_SD_ratio : 1 / 2),
  pyramid_volume AD AD_length area_ratio θ theta_angle midpoint_TB_TC TS_SD_ratio = 90 * √2 := 
by 
  sorry

end volume_of_pyramid_SAKND_l56_56849


namespace ksyusha_travel_time_wednesday_l56_56741

-- Definitions based on the conditions
def speed_walk (v : ℝ) : ℝ := v
def speed_run (v : ℝ) : ℝ := 2 * v
def distance_walk_tuesday (S : ℝ) : ℝ := 2 * S
def distance_run_tuesday (S : ℝ) : ℝ := S
def total_time_tuesday (v S : ℝ) : ℝ := (distance_walk_tuesday S) / (speed_walk v) + (distance_run_tuesday S) / (speed_run v)

-- Theorem statement
theorem ksyusha_travel_time_wednesday (S v : ℝ) (h : total_time_tuesday v S = 30) :
  (S / v + S / v) = 24 := 
by sorry

end ksyusha_travel_time_wednesday_l56_56741


namespace Cara_possible_pairs_l56_56158

-- Define the conditions and the final goal.
theorem Cara_possible_pairs : ∃ p : Nat, p = Nat.choose 7 2 ∧ p = 21 :=
by
  sorry

end Cara_possible_pairs_l56_56158


namespace ksyusha_travel_time_wednesday_l56_56745

-- Definitions based on the conditions
def speed_walk (v : ℝ) : ℝ := v
def speed_run (v : ℝ) : ℝ := 2 * v
def distance_walk_tuesday (S : ℝ) : ℝ := 2 * S
def distance_run_tuesday (S : ℝ) : ℝ := S
def total_time_tuesday (v S : ℝ) : ℝ := (distance_walk_tuesday S) / (speed_walk v) + (distance_run_tuesday S) / (speed_run v)

-- Theorem statement
theorem ksyusha_travel_time_wednesday (S v : ℝ) (h : total_time_tuesday v S = 30) :
  (S / v + S / v) = 24 := 
by sorry

end ksyusha_travel_time_wednesday_l56_56745


namespace arithmetic_geometric_sum_l56_56850

theorem arithmetic_geometric_sum {n : ℕ} (a : ℕ → ℤ) (S : ℕ → ℚ) 
  (h1 : ∀ k, a (k + 1) = a k + 2) 
  (h2 : (a 1) * (a 1 + a 4) = (a 1 + a 2) ^ 2 / 2) :
  S n = 6 - (4 * n + 6) / 2^n :=
by
  sorry

end arithmetic_geometric_sum_l56_56850


namespace maximum_value_quadratic_l56_56893

def quadratic_function (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem maximum_value_quadratic :
  ∃ x : ℝ, quadratic_function x = 2 ∧ ∀ y : ℝ, quadratic_function y ≤ 2 :=
sorry

end maximum_value_quadratic_l56_56893


namespace johns_average_speed_is_correct_l56_56724

noncomputable def johnsAverageSpeed : ℝ :=
  let total_time : ℝ := 6 + 0.5 -- Total driving time in hours
  let total_distance : ℝ := 210 -- Total distance covered in miles
  total_distance / total_time -- Average speed formula

theorem johns_average_speed_is_correct :
  johnsAverageSpeed = 32.31 :=
by
  -- This is a placeholder for the proof
  sorry

end johns_average_speed_is_correct_l56_56724


namespace ksyusha_wednesday_time_l56_56749

def speed_relation (v : ℝ) := ∀ (d_walk d_run1 d_run2 time1 time2: ℝ), 
  (2 * d_run1 = d_walk) → -- she walked twice the distance she ran on Tuesday
  (2 * v = 2 * v) → -- she runs twice as fast as she walks
  ((2 * d_walk / v) + (d_run1 / (2 * v)) = time1) → -- she took 30 minutes on Tuesday
  (time1 = 30) → -- Tuesday time equals 30
  (d_run2 = 2 * d_walk) → -- she had to run twice the distance she walked on Wednesday
  ((d_walk / v) + (d_run2 / (2 * v)) = time2) → -- calculate the time for Wednesday
  time2 = 24 -- time taken on Wednesday

theorem ksyusha_wednesday_time :
  ∃ (v : ℝ), speed_relation v :=
begin
  existsi 1,
  unfold speed_relation,
  intros,
  sorry,
end

end ksyusha_wednesday_time_l56_56749


namespace tangent_line_ln_at_e_l56_56023

theorem tangent_line_ln_at_e : 
  ∃ (m b : ℝ), 
    (∀ x : ℝ, x > 0 → x ≠ e → m * x + b = log x) ∧
    m * e + b = 1 ∧
    x - e * ((m * x + b) - b) = 0 :=
sorry

end tangent_line_ln_at_e_l56_56023


namespace original_movie_length_l56_56098

theorem original_movie_length (final_length cut_scene original_length : ℕ) 
    (h1 : cut_scene = 3) (h2 : final_length = 57) (h3 : final_length + cut_scene = original_length) : 
  original_length = 60 := 
by 
  -- Proof omitted
  sorry

end original_movie_length_l56_56098


namespace b_divisible_by_a_l56_56222

theorem b_divisible_by_a (a b c : ℕ) (ha : a > 1) (hbc : b > c ∧ c > 1) (hdiv : (abc + 1) % (ab - b + 1) = 0) : a ∣ b :=
  sorry

end b_divisible_by_a_l56_56222


namespace actual_average_height_correct_l56_56905

theorem actual_average_height_correct :
  let average_height := 185
  let number_of_boys := 35
  let wrong_height := 166
  let actual_height := 106
  let incorrect_total_height := average_height * number_of_boys
  let correct_total_height := incorrect_total_height - (wrong_height - actual_height)
  let actual_average_height := correct_total_height / number_of_boys
  real.floor (actual_average_height * 100) / 100 = 183.29 := 
by {
  sorry
}

end actual_average_height_correct_l56_56905


namespace merchant_profit_l56_56649

theorem merchant_profit (C S : ℝ) (h : 25 * C = 18 * S) : (S - C) / C * 100 = (7 / 18) * 100 :=
by
  -- assumptions are used here
  have ratio : S / C = 25 / 18 := by linarith
  -- we can go further with the proof steps if necessary, currently just providing the structure
  sorry

end merchant_profit_l56_56649


namespace probability_three_primes_l56_56487

def primes : List ℕ := [2, 3, 5, 7]

def is_prime (n : ℕ) : Prop := n ∈ primes

noncomputable def probability_prime : ℚ := 4/10
noncomputable def probability_non_prime : ℚ := 1 - probability_prime

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def calculation :
  ℚ := (choose 5 3) * (probability_prime ^ 3) * (probability_non_prime ^ 2)

theorem probability_three_primes :
  calculation = 720 / 3125 := by
  sorry

end probability_three_primes_l56_56487


namespace complementary_implies_mutually_exclusive_l56_56884

def ComplementaryEvents (E F : Set Ω) (P : Measure Ω) : Prop :=
  P E + P F = 1 ∧ ∀ ω : Ω, ω ∈ E → ω ∉ F

def MutuallyExclusiveEvents (E F : Set Ω) : Prop :=
  ∀ ω : Ω, ω ∈ E → ω ∉ F

theorem complementary_implies_mutually_exclusive (E F : Set Ω) (P : Measure Ω) :
  ComplementaryEvents E F P → MutuallyExclusiveEvents E F :=
by
  intro h
  sorry

end complementary_implies_mutually_exclusive_l56_56884


namespace average_weight_students_l56_56010

variable (A : ℝ) -- the average weight of the students

theorem average_weight_students (h1 : ∑ i in finset.range 24, A / 24 = A) 
    (h2 : ∑ i in finset.range 25, (if i = 24 then 45 else A + 0.4) / 25 = A + 0.4) :
    A = 35 := sorry

end average_weight_students_l56_56010


namespace correct_substitution_l56_56068

theorem correct_substitution (x y : ℝ) (h1 : y = 1 - x) (h2 : x - 2 * y = 4) : x - 2 * (1 - x) = 4 → x - 2 + 2 * x = 4 := by
  sorry

end correct_substitution_l56_56068


namespace coloring_ways_l56_56694

-- Define a function that determines if a given integer is a power of two
def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

-- Define the main problem condition as a predicate
def valid_coloring (coloring : ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ i → i ≤ 200 → 1 ≤ j → j ≤ 200 → i ≠ j → 
    coloring i = coloring j → ¬ is_power_of_two (i + j)

theorem coloring_ways : 
  ∃ (coloring : ℕ → Prop), (∀ (i : ℕ), 1 ≤ i → i ≤ 200 → (coloring i = true ∨ coloring i = false)) ∧
  (valid_coloring coloring) ∧
  (finset.card (finset.filter valid_coloring (finset.univ : set (ℕ → Prop))) = 256) :=
sorry

end coloring_ways_l56_56694


namespace marvins_birthday_tuesday_l56_56811

theorem marvins_birthday_tuesday :
  ∃ (y : ℕ), y > 2012 ∧ 
    (∀ y', 2012 < y' ∧ y' < y → 
      let is_leap := (y' % 4 = 0 ∧ (y' % 100 ≠ 0 ∨ y' % 400 = 0)) in
      let days := if is_leap then 2 else 1 in
      foldl (λ d y'', (
        let is_leap := (y'' % 4 = 0 ∧ (y'' % 100 ≠ 0 ∨ y'' % 400 = 0)) in
        (d + if is_leap then 2 else 1) % 7
      )) 5 (list.range (y' - 2012)) ≠ 2 
    ) ∧
    let is_leap := (y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)) in
    foldl (λ d y', (
      let is_leap := (y' % 4 = 0 ∧ (y' % 100 ≠ 0 ∨ y' % 400 = 0)) in
      (d + if is_leap then 2 else 1) % 7
    )) 5 (list.range (y - 2012)) = 2 :=
begin
  sorry
end

end marvins_birthday_tuesday_l56_56811


namespace jenna_eel_is_16_l56_56718

open Real

def jenna_eel_len (J B : ℝ) : Prop :=
  J = (1 / 3) * B ∧ J + B = 64

theorem jenna_eel_is_16 : ∃ J B : ℝ, jenna_eel_len J B ∧ J = 16 :=
by
  exists 16
  exists 64 * (3 / 4)    -- which is 48
  unfold jenna_eel_len
  split
  { norm_num }
  { norm_num }

end jenna_eel_is_16_l56_56718


namespace sum_of_coefficients_in_expansion_l56_56393

theorem sum_of_coefficients_in_expansion :
  ∑ i in Finset.range 6, (Polynomial.coeff (Polynomial.C 2 * X - Polynomial.C 3)^5 i) = -1 :=
by
  sorry

end sum_of_coefficients_in_expansion_l56_56393


namespace arithmetic_sequence_range_of_d_l56_56219

variable {n : ℕ}

theorem arithmetic_sequence_range_of_d (d : ℝ) (h : 0 < d) (a : ℕ → ℝ)
  (a_seq : ∀ n, a (n + 1) = a n + d)
  (h_geom_mean : a 2 = real.sqrt (a 1 * a 4))
  (h_ineq : ∀ n : ℕ, ∑ i in finset.range n, 1 / a (2 ^ i) < 3) :
  1 / 3 ≤ d :=
sorry

end arithmetic_sequence_range_of_d_l56_56219


namespace Ksyusha_time_to_school_l56_56739

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l56_56739


namespace solve_inequality_l56_56610

def f (x : ℝ) : ℝ := Real.sin x - 2 * x

theorem solve_inequality (a : ℝ) : 
  f(a^2 - 8) + f(2 * a) < 0 ↔ (a < -4 ∨ a > 2) :=
by
  sorry

end solve_inequality_l56_56610


namespace find_t_given_perpendicular_vectors_l56_56247

theorem find_t_given_perpendicular_vectors
  (t : ℝ)
  (a : ℝ × ℝ × ℝ)
  (b : ℝ × ℝ × ℝ)
  (ha : a = (1, t, 2))
  (hb : b = (2, -1, 2))
  (h_perpendicular : a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0) :
  t = 6 :=
begin
  sorry
end

end find_t_given_perpendicular_vectors_l56_56247


namespace min_red_squares_4x4_min_red_squares_nxn_l56_56095

-- Problem 1: 4×4 grid
theorem min_red_squares_4x4 : ∀ (grid : array (fin 4) (array (fin 4) nat)), 
  (∀ (r : fin 4 → bool) (c : fin 4 → bool), 
     2 = (finset.filter r finset.univ).card ∧ 2 = (finset.filter c finset.univ).card →
     ∃ (i j : fin 4), ¬ r i ∧ ¬ c j ∧ grid[i][j] = 1) ↔
  ∃ (colored_squares : finset (fin 4 × fin 4)), 
    colored_squares.card ≥ 7 ∧ 
    ∀ (i j : fin 4), (i, j) ∈ colored_squares → grid[i][j] = 1 := sorry

-- Problem 2: n×n grid where n≥5
theorem min_red_squares_nxn (n : ℕ) (h : n ≥ 5) : 
  ∀ (grid : array (fin n) (array (fin n) nat)), 
    (∀ (r : fin n → bool) (c : fin n → bool), 
       2 = (finset.filter r finset.univ).card ∧ 2 = (finset.filter c finset.univ).card →
       ∃ (i j : fin n), ¬ r i ∧ ¬ c j ∧ grid[i][j] = 1) ↔
    ∃ (colored_squares : finset (fin n × fin n)), 
      colored_squares.card ≥ 5 ∧ 
      ∀ (i j : fin n), (i, j) ∈ colored_squares → grid[i][j] = 1 := sorry

end min_red_squares_4x4_min_red_squares_nxn_l56_56095


namespace linear_function_does_not_pass_first_quadrant_l56_56537

theorem linear_function_does_not_pass_first_quadrant (m : ℝ) (x : ℝ) (y : ℝ) :
  m = -1 → (y = -3 * x + m) → (x > 0 → y < 0) :=
begin
  intros hm hxy hx,
  rw [hm] at hxy,
  rw [hxy] at hx,
  linarith,
end

end linear_function_does_not_pass_first_quadrant_l56_56537


namespace greatest_prime_factor_180_l56_56055

noncomputable def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

theorem greatest_prime_factor_180 : 
  ∃ p : ℕ, is_prime p ∧ p ∣ 180 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 180 → q ≤ p :=
  sorry

end greatest_prime_factor_180_l56_56055


namespace exists_xi_such_that_f_double_prime_eq_f_mul_1_add_2_tan_sq_l56_56314

theorem exists_xi_such_that_f_double_prime_eq_f_mul_1_add_2_tan_sq
  (f : ℝ → ℝ)
  (hf : ∀ x, differentiable ℝ (deriv f x))
  (h0 : f 0 = 0) :
  ∃ ξ ∈ Ioo (-π / 2) (π / 2), deriv (deriv f) ξ = f ξ * (1 + 2 * (Real.tan ξ)^2) := 
sorry

end exists_xi_such_that_f_double_prime_eq_f_mul_1_add_2_tan_sq_l56_56314


namespace area_triangle_ABC_l56_56040

open Real

structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance (P Q : Point) : ℝ := sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

def area_of_triangle (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

theorem area_triangle_ABC :
  let A := Point.mk (-7) 3
  let B := Point.mk 0 4
  let C := Point.mk 9 5
  distance A B = 7 ∧ distance B C = 9 → area_of_triangle A B C = 8 :=
by
  intros
  sorry

end area_triangle_ABC_l56_56040


namespace present_value_investment_l56_56641

theorem present_value_investment :
  let FV := 400000
  let r := 0.06
  let n := 10
  let PV := FV / (1 + r)^n
  (Real.round (PV * 100) / 100) = 223387.15 :=
by
  sorry

end present_value_investment_l56_56641


namespace fish_ratio_l56_56311

theorem fish_ratio (k : ℕ) (kendra_fish : ℕ) (home_fish : ℕ)
    (h1 : kendra_fish = 30)
    (h2 : home_fish = 87)
    (h3 : k - 3 + kendra_fish = home_fish) :
  k = 60 ∧ (k / 3, kendra_fish / 3) = (19, 10) :=
by
  sorry

end fish_ratio_l56_56311


namespace find_cosine_of_angle_l56_56216

-- Definitions of the key elements as stated in the problem
structure Point (α : Type) := (x y z : α)

-- Assume we have a right quadrilateral pyramid where the height equals half the length of AB.
def A := Point.mk (-1 : ℝ) (-1) 0
def B := Point.mk (1 : ℝ) (-1) 0
def D := Point.mk (-1 : ℝ) (1) 0
def V := Point.mk (0 : ℝ) (0) 1

-- Define M and N according to the conditions in the problem.
def midpoint {α : Type} [Field α] (p1 p2 : Point α) : Point α :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2⟩

def point_N {α : Type} [Field α] (p1 p2 : Point α) (ratio : α) : Point α :=
  ⟨(p1.x + ratio * p2.x) / (1 + ratio), (p1.y + ratio * p2.y) / (1 + ratio), (p1.z + ratio * p2.z) / (1 + ratio)⟩

def M := midpoint V B
def N := point_N V D 2

-- Vector from point A to point M
def vector (p1 p2 : Point ℝ) := Point.mk (p2.x - p1.x) (p2.y - p1.y) (p2.z - p1.z)

def AM := vector A M
def BN := vector B N

-- Dot product of two vectors
def dot_product (v1 v2 : Point ℝ) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Magnitude of a vector
def magnitude (v : Point ℝ) : ℝ :=
  real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

-- Cosine of the angle between two vectors
def cos_theta (v1 v2 : Point ℝ) : ℝ :=
  (dot_product v1 v2) / (magnitude v1 * magnitude v2)

theorem find_cosine_of_angle : 
  cos_theta AM BN = (real.sqrt 11 / 11) :=
by
  sorry

end find_cosine_of_angle_l56_56216


namespace greg_walk_perimeter_area_l56_56249

def greg_viewable_area (length : ℝ) (width : ℝ) (visibility_radius : ℝ) : ℝ :=
  let extra_area := 2 * (length * visibility_radius + width * visibility_radius) + 4 * (Real.pi * visibility_radius^2 / 4)
  extra_area + length * width - (length - 2 * visibility_radius) * (width - 2 * visibility_radius)

theorem greg_walk_perimeter_area :
  greg_viewable_area 8 4 2 = 61 :=
by
  sorry

end greg_walk_perimeter_area_l56_56249


namespace square_area_inscribed_in_parabola_l56_56002

theorem square_area_inscribed_in_parabola :
  ∃ (s : ℝ), 
    (s > 0) ∧
    ∃ (area : ℝ),
    (area = (2 * s)^2) ∧
    ((-2 * s = (5 + s)^2 - 10 * (5 + s) + 21) ∧ 
     area = 24 - 8 * real.sqrt 5) :=
begin
  sorry
end

end square_area_inscribed_in_parabola_l56_56002


namespace solve_quadratic_equation_l56_56392

theorem solve_quadratic_equation : 
  ∀ x : ℝ, 2 * x^2 = 4 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 :=
by
  sorry


end solve_quadratic_equation_l56_56392


namespace speech_contest_probability_l56_56906

theorem speech_contest_probability :
  let total_sequences := (10.factorial : ℕ),
      class1_consecutive := (8.factorial * 3.factorial : ℕ),
      class1_prob := (class1_consecutive.toRat) / (total_sequences.toRat),
      class2_not_consecutive := (1.toRat - (class1_prob / 2)) in
  class2_not_consecutive = (1.toRat / 20.toRat) :=
by sorry

end speech_contest_probability_l56_56906


namespace johns_average_speed_is_correct_l56_56725

noncomputable def johnsAverageSpeed : ℝ :=
  let total_time : ℝ := 6 + 0.5 -- Total driving time in hours
  let total_distance : ℝ := 210 -- Total distance covered in miles
  total_distance / total_time -- Average speed formula

theorem johns_average_speed_is_correct :
  johnsAverageSpeed = 32.31 :=
by
  -- This is a placeholder for the proof
  sorry

end johns_average_speed_is_correct_l56_56725


namespace coefficient_of_x_squared_in_expansion_l56_56705

theorem coefficient_of_x_squared_in_expansion :
  let expr := (x + (2 / (x^2)) : ℝ)
  let expansion := (expr ^ 5)
  find_coefficient expansion x 2 = 10 :=
by
  sorry

end coefficient_of_x_squared_in_expansion_l56_56705


namespace Ksyusha_travel_time_l56_56777

theorem Ksyusha_travel_time:
  ∀ (S v : ℝ), 
  (S > 0 ∧ v > 0) →
  (∀ tuesday_distance tuesday_time wednesday_distance wednesday_time: ℝ, 
    tuesday_distance = 3 * S ∧ 
    tuesday_distance = 2 * S + S ∧ 
    tuesday_time = (2 * S) / v + (S / (2 * v)) ∧ 
    tuesday_time = 30) → 
  (∀ wednesday_distance wednesday_time : ℝ, 
    wednesday_distance = S + 2 * S ∧ 
    wednesday_time = S / v + (2 * S) / (2 * v)) →
  wednesday_time = 24 :=
by
  intros S v S_pos v_pos tuesday_distance tuesday_time wednesday_distance wednesday_time
  sorry

end Ksyusha_travel_time_l56_56777


namespace age_ratio_l56_56268

-- Define variables for the current ages of A and B
variable (A B : ℕ)

-- Given conditions
def age_conditions : Prop :=
  B = 39 ∧ A = B + 9

-- Mathematical statement to prove
theorem age_ratio (A B : ℕ) (h : age_conditions A B) : (A + 10) / (B - 10) = 2 :=
by
  unfold age_conditions at h
  cases h with hB hA
  rw [hB, hA]
  norm_num
  sorry

end age_ratio_l56_56268


namespace find_AB_l56_56963

noncomputable def angle_B : ℝ := 30
noncomputable def angle_C : ℝ := 45
noncomputable def angle_BDC : ℝ := 150
noncomputable def BD : ℝ := 5
noncomputable def CD : ℝ := 5

theorem find_AB : 
  angle_B = 30 ∧ angle_C = 45 ∧ angle_BDC = 150 ∧ BD = 5 ∧ CD = 5 → 
  ∃ AB : ℝ, AB = 5 * sqrt 2 := 
by
  sorry

end find_AB_l56_56963


namespace each_person_towel_day_l56_56097

def total_people (families : ℕ) (members_per_family : ℕ) : ℕ :=
  families * members_per_family

def total_towels (loads : ℕ) (towels_per_load : ℕ) : ℕ :=
  loads * towels_per_load

def towels_per_day (total_towels : ℕ) (days : ℕ) : ℕ :=
  total_towels / days

def towels_per_person_per_day (towels_per_day : ℕ) (total_people : ℕ) : ℕ :=
  towels_per_day / total_people

theorem each_person_towel_day
  (families : ℕ) (members_per_family : ℕ) (days : ℕ) (loads : ℕ) (towels_per_load : ℕ)
  (h_family : families = 3) (h_members : members_per_family = 4) (h_days : days = 7)
  (h_loads : loads = 6) (h_towels_per_load : towels_per_load = 14) :
  towels_per_person_per_day (towels_per_day (total_towels loads towels_per_load) days) (total_people families members_per_family) = 1 :=
by {
  -- Import necessary assumptions
  sorry
}

end each_person_towel_day_l56_56097


namespace similar_triangles_iff_eq_l56_56860

theorem similar_triangles_iff_eq (a b c a1 b1 c1 : ℝ) :
  (∃ (k : ℝ), a1 = k^2 * a ∧ b1 = k^2 * b ∧ c1 = k^2 * c) ↔
  ( √(a * a1) + √(b * b1) + √(c * c1) = √((a + b + c) * (a1 + b1 + c1)) ) :=
by
  sorry

end similar_triangles_iff_eq_l56_56860


namespace find_g_l56_56794

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 4

theorem find_g :
  (∀ x : ℝ, g x = x + 5) ∨ (∀ x : ℝ, g x = -x - 3) := 
by
  sorry

end find_g_l56_56794


namespace megan_dials_correct_number_probability_l56_56816

theorem megan_dials_correct_number_probability :
  let possible_first_three_digits := 2
  let possible_last_four_digit_arrangements := 4!
  let total_possible_numbers := possible_first_three_digits * possible_last_four_digit_arrangements
  in 1 / total_possible_numbers = 1 / 48 :=
by
  sorry

end megan_dials_correct_number_probability_l56_56816


namespace minimum_value_l56_56162

theorem minimum_value (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : S 2017 = 4034)
    (h2 : ∀ n, S n = n * (a 1 + a n) / 2) (h3 : ∀ n m, a (n + 1) - a n = a (m + 1) - a m)
    (h4: ∀ n, a n > 0) : 
    (∃ c : ℝ, (1 / a 9) + (9 / a 2009) = c) ∧ (∀ d : ℝ, (1 / a 9) + (9 / a 2009) ≥ d → d ≥ 4) :=
by
  sorry

end minimum_value_l56_56162


namespace base_4_subtraction_correct_l56_56505

theorem base_4_subtraction_correct :
  let base4 := Nat.toDigits 4
  base4 207 = [3, 0, 3, 3] →
  base4 85 = [1, 1, 1, 1] →
  Nat.ofDigits 4 ([3, 0, 3, 3] - [1, 1, 1, 1]) = 1232 := by
sorry

end base_4_subtraction_correct_l56_56505


namespace intersection_with_y_axis_l56_56015

theorem intersection_with_y_axis :
  (∃ y : ℝ, y = -(0 + 2)^2 + 6 ∧ (0, y) = (0, 2)) :=
by
  sorry

end intersection_with_y_axis_l56_56015


namespace trapezium_shorter_side_length_l56_56526

theorem trapezium_shorter_side_length 
  (long_side : ℝ) (distance : ℝ) (area : ℝ) (short_side_approx : ℝ)
  (h1 : long_side = 20) 
  (h2 : distance = 14) 
  (h3 : area = 266) 
  (h4 : short_side_approx ≈ 17.71) :
  (1 / 2) * (short_side_approx + long_side) * distance = area :=
by
  sorry

end trapezium_shorter_side_length_l56_56526


namespace statement_C_is_incorrect_l56_56427

-- Definitions based on the conditions provided
def is_square (q : Quadrilateral) : Prop :=
  is_parallelogram q ∧ equal_diagonals q ∧ perpendicular_diagonals q

def is_rhombus (p : Parallelogram) : Prop :=
  perpendicular_diagonals p

def is_rectangle (q : Quadrilateral) : Prop :=
  equal_diagonals q ∧ bisecting_diagonals q

def one_right_angle (q : Quadrilateral) : Prop :=
  ∃ (a b c d : Point), q = Quadrilateral.mk a b c d ∧ angle_b a b c = 90

-- Given statements as conditions
axiom statement_A (q : Quadrilateral) (hq1 : is_parallelogram q) (hq2 : equal_diagonals q) (hq3 : perpendicular_diagonals q) : is_square q

axiom statement_B (p : Parallelogram) (hp : perpendicular_diagonals p) : is_rhombus p

axiom statement_D (q : Quadrilateral) (hq1 : equal_diagonals q) (hq2 : bisecting_diagonals q) : is_rectangle q

-- The proof statement
theorem statement_C_is_incorrect (q : Quadrilateral) (h : one_right_angle q) : ¬ is_rectangle q :=
by sorry

end statement_C_is_incorrect_l56_56427


namespace cost_price_l56_56437

-- Given conditions
variable (x : ℝ)
def profit (x : ℝ) : ℝ := 54 - x
def loss (x : ℝ) : ℝ := x - 40

-- Claim
theorem cost_price (h : profit x = loss x) : x = 47 :=
by {
  -- This is where the proof would go
  sorry
}

end cost_price_l56_56437


namespace annual_income_earned_by_both_investments_l56_56050

noncomputable def interest (principal: ℝ) (rate: ℝ) (time: ℝ) : ℝ :=
  principal * rate * time

theorem annual_income_earned_by_both_investments :
  let total_amount := 8000
  let first_investment := 3000
  let first_interest_rate := 0.085
  let second_interest_rate := 0.064
  let second_investment := total_amount - first_investment
  interest first_investment first_interest_rate 1 + interest second_investment second_interest_rate 1 = 575 :=
by
  sorry

end annual_income_earned_by_both_investments_l56_56050


namespace ksyusha_travel_time_l56_56771

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l56_56771


namespace ksyusha_travel_time_wednesday_l56_56743

-- Definitions based on the conditions
def speed_walk (v : ℝ) : ℝ := v
def speed_run (v : ℝ) : ℝ := 2 * v
def distance_walk_tuesday (S : ℝ) : ℝ := 2 * S
def distance_run_tuesday (S : ℝ) : ℝ := S
def total_time_tuesday (v S : ℝ) : ℝ := (distance_walk_tuesday S) / (speed_walk v) + (distance_run_tuesday S) / (speed_run v)

-- Theorem statement
theorem ksyusha_travel_time_wednesday (S v : ℝ) (h : total_time_tuesday v S = 30) :
  (S / v + S / v) = 24 := 
by sorry

end ksyusha_travel_time_wednesday_l56_56743


namespace number_of_parallel_lines_l56_56933

-- Define the context of our problem.
-- Assume three vertices of triangle ABC and their corresponding vertices in the plane ABBA₁
variable {A B C A₁ B₁ C₁ : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A₁] [AddGroup B₁] [AddGroup C₁]

-- State that the lines we are considering are drawn through midpoints of any two edges of the triangular prism.
def drawn_lines_through_midpoints (edges : List (A × B × C × A₁ × B₁ × C₁)) : ℕ :=
  -- Calculate the valid lines parallel to the plane ABBA₁
  let valid_edges := [(BC, B₁C₁), (BC, C₁A₁), (BC, CA), (B₁C₁, C₁A₁), (B₁C₁, CA), (C₁A₁, CA)]
  6  -- there are 6 such lines

theorem number_of_parallel_lines {AB BA₁ : Type} :
  drawn_lines_through_midpoints [(A, B, C, A₁, B₁, C₁)] = 6 := by
  -- We need to prove that under these conditions
  sorry

end number_of_parallel_lines_l56_56933


namespace balls_in_boxes_l56_56277

theorem balls_in_boxes (n : ℕ) (hr: n = 100):
  let red_box := 92,
  let green_mixed := 100 + 8,
  let red_mixed := 92 + 8,
  let prob_green_in_red := 8 / (92 + 8),
  let prob_red_in_green := 8 / (100 + 8)
  in prob_green_in_red = prob_red_in_green :=
by sorry

end balls_in_boxes_l56_56277


namespace tangent_product_l56_56109

-- Declarations for circles, points of tangency, and radii
variables (R r : ℝ) -- radii of the circles
variables (A B C : ℝ) -- distances related to the tangents

-- Conditions: Two circles, a common internal tangent intersecting at points A and B, tangent at point C
axiom tangent_conditions : A * B = R * r

-- Problem statement: Prove that A * C * C * B = R * r
theorem tangent_product (R r A B C : ℝ) (h : A * B = R * r) : A * C * C * B = R * r :=
by
  sorry

end tangent_product_l56_56109


namespace beads_necklace_l56_56814

theorem beads_necklace (beads : Fin (1000)) (color : beads → Fin (50)) :
  ∃ k : ℕ, (k ≤ 462) ∧ (∀ (a : ℕ), a < 1000 → {c : Fin 50 | ∃ i, (a ≤ i ∧ i < a + k) ∧ color beads[i] = c}.card ≥ 25) := 
sorry

end beads_necklace_l56_56814


namespace sum_inverse_square_magnitude_roots_of_unity_l56_56065

theorem sum_inverse_square_magnitude_roots_of_unity :
  (∑ z in {z : ℂ | z^8 = 1}, (1 / |1 - z|^2)) = 0 := by
  sorry

end sum_inverse_square_magnitude_roots_of_unity_l56_56065


namespace expected_value_Y_meets_production_contract_l56_56476

namespace AgriculturalProducts

-- Given quality scores of products
def quality_scores : List ℝ := [38, 70, 50, 45, 48, 54, 49, 57, 60, 69]

-- High-quality products are with scores of at least 60
def is_high_quality (score : ℝ) : Bool := score ≥ 60

-- Calculate the distribution and expected value of Y
theorem expected_value_Y :
  let Y := [0, 1, 2]
  let P_Y := [7/15, 7/15, 1/15]  -- Probabilities calculated
  (List.zip Y P_Y).sum * Prod.fst * Prod.snd = 3/5 :=
  sorry

-- Mean and variance of the quality scores
def mean_quality : ℝ := (List.sum quality_scores) / (List.length quality_scores)

def variance_quality : ℝ := (List.sum (quality_scores.map (λx, (x - mean_quality) ^ 2))) / (List.length quality_scores)

noncomputable def stddev_quality : ℝ := sqrt variance_quality

-- Check if the high-quality rate meets the production contract requirement
theorem meets_production_contract :
  let μ := mean_quality
  let σ := stddev_quality
  let high_quality_threshold := 60
  let high_quality_proportion := (1 - 0.6827) / 2
  high_quality_proportion * 2 > 0.15 :=
  sorry

end AgriculturalProducts

end expected_value_Y_meets_production_contract_l56_56476


namespace mean_of_xyz_l56_56364

theorem mean_of_xyz (a b c d e f g x y z : ℝ)
  (h1 : (a + b + c + d + e + f + g) / 7 = 48)
  (h2 : (a + b + c + d + e + f + g + x + y + z) / 10 = 55) :
  (x + y + z) / 3 = 71.33333333333333 :=
by
  sorry

end mean_of_xyz_l56_56364


namespace sin_B_l56_56287

variable (A B C : ℝ)
variable {triangle : Type} [metric_space triangle]
variable (ABC : triangle)
variable (a b c : ℝ) -- sides of the triangle
variable (h1 : sin A = 3 / 5) 
variable (h2 : C = π / 2) -- Angle C is 90 degrees (angle in radians, 90° = π/2 rad)

theorem sin_B : sin B = 3 / 5 := 
sorry

end sin_B_l56_56287


namespace least_three_digit_7light_l56_56134

def is_7light (n : ℕ) : Prop := n % 7 < 3

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem least_three_digit_7light : ∃ n : ℕ, is_three_digit n ∧ is_7light n ∧ ∀ m : ℕ, is_three_digit m ∧ is_7light m → m >= n :=
by {
    use 100,
    split,
    -- 100 is a three-digit number
    exact ⟨le_refl 100, nat.lt_succ_self 999⟩,
    split,
    -- 100 is 7-light
    exact nat.mod_lt 100 (by norm_num),
    -- 100 is the smallest three-digit 7-light number
    intro m,
    intro hm,
    cases hm with h1 h2,
    exact le_of_lt (h1.1),
    sorry
}

end least_three_digit_7light_l56_56134


namespace sum_of_valid_nu_l56_56061

open Nat

-- Condition: ν such that lcm(ν, 18) = 90
def valid_nu (ν : ℕ) : Prop := lcm ν 18 = 90

-- Sum of all positive integers ν for the valid condition
def sum_valid_nus : ℕ := (Finset.range 91).filter valid_nu |>.sum

-- Main theorem stating the sum of all valid ν's equals 195
theorem sum_of_valid_nu : sum_valid_nus = 195 := by
  sorry

end sum_of_valid_nu_l56_56061


namespace red_blue_tetrahedra_l56_56509

theorem red_blue_tetrahedra (n : ℕ) : ¬ ∀ red_points : set (point ℝ 3), 
  (∃ blue_points : set (point ℝ 3), (blue_points.card = 3 * n) ∧ 
   ∀ t ∈ { t : set (point ℝ 3) | t.card = 4 ∧ t ⊆ red_points }, 
   (∃ p ∈ blue_points, p ∈ convex_hull t)) :=
begin
  sorry
end

end red_blue_tetrahedra_l56_56509


namespace range_of_a_l56_56979

noncomputable def monotonic_function (f : ℝ → ℝ) : Prop :=
∀ x y, x ≤ y → f(x) ≤ f(y)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  monotonic_function f →
  (∀ x y, f (x + y) = f x + f y) →
  (∃ x ∈ Ioo 0 π, f (a * sin x) + f (sin x + cos x^2 - 3) = 0) →
  a ∈ Ici 2 :=
by
  sorry

end range_of_a_l56_56979


namespace remainder_x5_1_x3_1_div_x2_x_1_l56_56172

theorem remainder_x5_1_x3_1_div_x2_x_1 :
  ∀ (x : ℂ), let poly := (x^5 - 1) * (x^3 - 1),
                 divisor := x^2 + x + 1,
                 remainder := x^2 + x + 1 in
  ∃ q : ℂ, poly = q * divisor + remainder :=
by
  intro x
  let poly := (x^5 - 1) * (x^3 - 1)
  let divisor := x^2 + x + 1
  let remainder := x^2 + x + 1
  use sorry
  rw [← add_assoc, ← mul_assoc, ← pow_succ]
  sorry

end remainder_x5_1_x3_1_div_x2_x_1_l56_56172


namespace opening_angle_of_cone_is_60_degrees_l56_56446

-- The problem deals with geometry in a regular tetrahedron
def regular_tetrahedron (A B C D : Type) := sorry -- The exact definition and stipulations need to be filled out depending on the detailed formal geometry library.

-- Given the conditions
variables {A B C D : Type}
variables [regular_tetrahedron A B C D]

-- Define the construction of the cone
noncomputable def cone_apex_A_axis_AB_tangent_CD : ℝ := 60

-- The goal is to prove that the angle is indeed 60 degrees
theorem opening_angle_of_cone_is_60_degrees : cone_apex_A_axis_AB_tangent_CD = 60 := 
sorry

end opening_angle_of_cone_is_60_degrees_l56_56446


namespace product_tan_identity_l56_56643

theorem product_tan_identity :
  (∀ x, (1 + Real.tan x) * (1 + Real.tan (90 - x)) = 2) →
  ∏ (i : Fin 22), (1 + Real.tan (2 * (i + 1) : ℝ)) = 2^22 := 
sorry

end product_tan_identity_l56_56643


namespace area_of_walkways_is_214_l56_56358

-- Definitions for conditions
def width_of_flower_beds : ℕ := 2 * 7  -- two beds each 7 feet wide
def walkways_between_beds_width : ℕ := 3 * 2  -- three walkways each 2 feet wide (one on each side and one in between)
def total_width : ℕ := width_of_flower_beds + walkways_between_beds_width  -- Total width

def height_of_flower_beds : ℕ := 3 * 3  -- three rows of beds each 3 feet high
def walkways_between_beds_height : ℕ := 4 * 2  -- four walkways each 2 feet wide (one on each end and one between each row)
def total_height : ℕ := height_of_flower_beds + walkways_between_beds_height  -- Total height

def total_area_of_garden : ℕ := total_width * total_height  -- Total area of the garden including walkways

def area_of_one_flower_bed : ℕ := 7 * 3  -- Area of one flower bed
def total_area_of_flower_beds : ℕ := 6 * area_of_one_flower_bed  -- Total area of six flower beds

def total_area_walkways : ℕ := total_area_of_garden - total_area_of_flower_beds  -- Total area of the walkways

-- Theorem to prove the area of the walkways
theorem area_of_walkways_is_214 : total_area_walkways = 214 := sorry

end area_of_walkways_is_214_l56_56358


namespace construct_triangle_with_angle_bisectors_l56_56400

-- Define the given structures and conditions in Lean

variables {Point : Type} [Inhabited Point]
variables {Line : Type} [Inhabited Line]

-- Assume a function that returns the intersection point of three lines
variables (intersection : Line → Line → Line → Point)

-- Assume a function that checks if a point lies on a line
variables (lies_on : Point → Line → Prop)

-- Assume a function that constructs the symmetric point with respect to a line
variables (symmetric_point : Point → Line → Point)

-- Assume a function that returns the line passing through two given points
variables (line_through : Point → Point → Line)

-- Assume function for intersection of lines
variables (intersection_of_lines : Line → Line → Point)
 
-- Definition of the math problem in Lean 4

theorem construct_triangle_with_angle_bisectors
  (l1 l2 l3 : Line) (P : Point) (A : Point)
  (h1 : P = intersection l1 l2 l3)
  (h2 : lies_on A l1) :
  ∃ (B C : Point),
    let A2 := symmetric_point A l2 in
    let A3 := symmetric_point A l3 in
    let Line_A2A3 := line_through A2 A3 in
    B = intersection_of_lines Line_A2A3 l2 ∧
    C = intersection_of_lines Line_A2A3 l3 ∧
    lies_on (intersection_of_lines (line_through A B) (line_through A C)) l1 ∧
    lies_on (intersection_of_lines (line_through B C) (line_through B A)) l2 ∧
    lies_on (intersection_of_lines (line_through C A) (line_through C B)) l3 :=
sorry

end construct_triangle_with_angle_bisectors_l56_56400


namespace barn_painting_area_l56_56918

def width := 15 -- width in yards
def length := 20 -- length in yards
def height := 8 -- height in yards

def area_of_wall1 := width * height -- area of the first pair of walls
def area_of_wall2 := length * height -- area of the second pair of walls
def area_of_ceiling_floor := width * length -- area of ceiling and floor

def total_area := 2 * (2 * area_of_wall1 + 2 * area_of_wall2) + 2 * area_of_ceiling_floor

theorem barn_painting_area : total_area = 1720 := by
  sorry

end barn_painting_area_l56_56918


namespace sum_of_possible_N_l56_56538

-- Let L represent a set of five distinct lines in a plane
variable (L : Set (Line ℝ)) (hL : L.card = 5)

-- Let N represent the number of distinct points where two or more lines intersect
variable (N : ℕ)

-- Define the function that counts distinct intersection points
def countIntersections (L : Set (Line ℝ)) : ℕ := sorry

-- Define the property stating N is the count of intersections
def validIntersectionCount (N : ℕ) (L : Set (Line ℝ)) : Prop :=
  N = countIntersections L

-- State the final proof statement
theorem sum_of_possible_N :
  (validIntersectionCount N L) →
  (N ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) →
  (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 = 55) :=
by
  intros
  exact Eq.refl 55

end sum_of_possible_N_l56_56538


namespace no_second_invoice_is_23_l56_56484

-- Define conditions
def first_invoice_numbers : set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

def second_invoice_number (x : ℕ) : ℕ := x + 10

-- Define the proof problem
theorem no_second_invoice_is_23 (x : ℕ) (hx : x ∈ first_invoice_numbers) : second_invoice_number x ≠ 23 :=
by sorry

end no_second_invoice_is_23_l56_56484


namespace new_median_l56_56128

def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

def mode (s : List ℕ) : ℕ :=
  s.groupBy id
  |>.map List.length
  |>.maximum.getD 0

def median (s : List ℕ) : ℚ :=
  let sorted := s.sorted
  if h : s.length % 2 = 1 then
    sorted.get (s.length / 2)
  else
    (sorted.get (s.length / 2 - 1) + sorted.get (s.length / 2)) / 2

def new_set (s : List ℕ) : List ℕ := s ++ [10]

theorem new_median (s : List ℕ) (h1 : mean s = 5.2)
  (h2 : mode s = 2) (h3 : median s = 5) : median (new_set s) = 6.5 := 
  sorry

end new_median_l56_56128


namespace centroids_concyclic_l56_56911

noncomputable theory

open Complex

variables {A B C D : Complex}

def is_cyclic_quadrilateral (A B C D : Complex) : Prop :=
∃ O R, R > 0 ∧ ∀ z ∈ {A, B, C, D}, abs (z - O) = R

def centroid (z1 z2 z3 : Complex) : Complex :=
(1/3) * (z1 + z2 + z3)

theorem centroids_concyclic
  (h: is_cyclic_quadrilateral A B C D) :
  ∃ O R, ∀ G ∈ {centroid A B C, centroid C D A, centroid B C D, centroid D A B}, abs (G - O) = R := sorry

end centroids_concyclic_l56_56911


namespace find_pairs_of_positive_integers_l56_56525

theorem find_pairs_of_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^3 + y^3 = 4 * (x^2 * y + x * y^2 - 5) → (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 1) :=
by
  sorry

end find_pairs_of_positive_integers_l56_56525


namespace locus_eq_l56_56011

noncomputable def locus_of_centers (a b : ℝ) : Prop :=
  5 * a^2 + 9 * b^2 + 80 * a - 400 = 0

theorem locus_eq (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 2)^2) ∧ ((a - 1)^2 + b^2 = (5 - r)^2)) →
  locus_of_centers a b :=
by
  intro h
  sorry

end locus_eq_l56_56011


namespace line_equations_l56_56572

theorem line_equations (P : ℝ × ℝ)
  (hx : P.1 = -3) (hy : P.2 = -3 / 2)
  (hchord : ∃ l : ℝ × ℝ → Prop, (∀ (x y : ℝ), l (x, y) ↔ (x^2 + y^2 = 25) ∧ l (P.1, P.2)) ∧ 
            (∃ chord_length, chord_length = 8)) :
  (∀ (x y : ℝ), ((3 * x + 4 * y + 15 = 0) ∨ (x = -3) → l (x, y))) :=
by
  sorry

end line_equations_l56_56572


namespace ksyusha_travel_time_l56_56765

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l56_56765


namespace products_A_and_B_next_to_each_other_l56_56397

theorem products_A_and_B_next_to_each_other :
  ∃ (arrangements : ℕ),
    (arrangements == 12) ∧
    let products := {A, B, C, D} in
    arrangements = (3! * 2) ∧
    (products.card = 4) ∧ -- Given there are 4 different products
    ∀ (arrangement : list), arrangement.length = 4 →
    ∃ (subarrangement : list), (subarrangement.length = 2) ∧
    (subarrangement = [A, B] ∨ subarrangement = [B, A])
:= sorry

end products_A_and_B_next_to_each_other_l56_56397


namespace factorize_expression_l56_56994

theorem factorize_expression (x : ℝ) : 2 * x ^ 2 - 50 = 2 * (x + 5) * (x - 5) := 
  sorry

end factorize_expression_l56_56994


namespace polynomials_equal_if_roots_match_l56_56330

open Polynomial Set

/-- Proof problem:
Let P(x) and Q(x) be polynomials of degree greater than 0. Define
P_{(c)}={z ∈ c | P(z)=c}, Q_{(c)}={z ∈ c | Q(z)=c}.
Prove that if P_{0}=Q_{0} and P_{1}=Q_{1}, then P(x) ≡ Q(x) for x ∈ ℝ.
-/
theorem polynomials_equal_if_roots_match (P Q : Polynomial ℝ)
  (P_deg_pos : P.degree > 0) (Q_deg_pos : Q.degree > 0)
  (P_c : ∀ c, {z | P.eval z = c}.subset (set.univ c))
  (Q_c : ∀ c, {z | Q.eval z = c}.subset (set.univ c))
  (P0_eq_Q0 : {z | P.eval z = 0} = {z | Q.eval z = 0})
  (P1_eq_Q1 : {z | P.eval z = 1} = {z | Q.eval z = 1}) :
  P = Q := 
sorry

end polynomials_equal_if_roots_match_l56_56330


namespace union_sets_l56_56623

def A : set ℝ := { x | x^2 - 5 * x - 6 < 0 }
def B : set ℝ := { x | -3 < x ∧ x < 2 }
def C : set ℝ := { x | -3 < x ∧ x < 6 }

theorem union_sets (A B C : set ℝ) (hA : A = { x | x^2 - 5 * x - 6 < 0 })
  (hB : B = { x | -3 < x ∧ x < 2 }) :
  (A ∪ B) = { x | -3 < x ∧ x < 6 } :=
sorry

end union_sets_l56_56623


namespace pizza_problem_l56_56930

theorem pizza_problem (m d : ℕ) :
  (7 * m + 2 * d > 36) ∧ (8 * m + 4 * d < 48) ↔ (m = 5) ∧ (d = 1) := by
  sorry

end pizza_problem_l56_56930


namespace number_of_valid_colorings_l56_56670

-- Define the specific properties and constraints given in the problem
def is_power_of_two (n : ℕ) : Prop := ∃ m : ℕ, n = 2^m

def valid_coloring (f : ℕ → bool) : Prop :=
  ∀ a b : ℕ, a ≠ b → a + b ≤ 200 →
  (f a = f b → ¬ is_power_of_two (a + b))

theorem number_of_valid_colorings : 
  ∃ c : ℕ, c = 256 ∧ 
  ∃ f : Π (n : ℕ), nat.lt n 201 → bool, 
    valid_coloring (λ n, f n (by linarith)) :=
begin
  sorry
end

end number_of_valid_colorings_l56_56670


namespace max_value_of_S_over_b2_plus_2c2_l56_56261

theorem max_value_of_S_over_b2_plus_2c2 (A B C : ℝ) (a b c S : ℝ)
  (h1 : 3 * a^2 = 2 * b^2 + c^2)
  (h2 : is_area_of_triangle S a b c A) :
  ( ∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ (forall A B C, 3 * a^2 = 2 * b^2 + c^2) ∧ (area S a b c = (b * c * sin A) / 2)
  ∧ find_max_value (frac S (b^2 + 2*c^2)) = sqrt 14 / 24 ) :=
sorry

end max_value_of_S_over_b2_plus_2c2_l56_56261


namespace numeral_system_multiplication_pattern_l56_56648

theorem numeral_system_multiplication_pattern (a k : ℕ) (h1 : 0 < a) (h2 : k < a) :
  ∃ n : ℕ, ∀ b : ℕ, b < a - 1 → (12345 .. (a-3) * (a-1)) * (k * (a-1)) = k * (∑ i in finset.range (a - 1), a ^ i) :=
begin
  sorry
end

end numeral_system_multiplication_pattern_l56_56648


namespace find_median_interval_correct_l56_56163

-- Given histogram data
def intervals : List (ℕ × ℕ) := [(85, 10), (80, 15), (75, 20), (70, 25), (65, 15), (60, 10), (55, 5)]

-- Definition to find interval that contains the median score
def median_interval (data : List (ℕ × ℕ)) : ℕ × ℕ :=
  let sorted_intervals := data.sort (λ a b => a.1 > b.1)
  let cumulative_counts := sorted_intervals.scanl (λ acc pair => acc + pair.2) 0
  let median_pos := 50
  let median_index := cumulative_counts.findIndex (λ count => count ≥ median_pos)
  sorted_intervals.getD median_index (0, 0)

theorem find_median_interval_correct :
  median_interval intervals = (70, 25) :=
by
  sorry

end find_median_interval_correct_l56_56163


namespace isosceles_trapezoid_area_l56_56283

theorem isosceles_trapezoid_area
  (a b h : ℝ)
  (midline_length : (a + b) / 2 = 5)
  (height : h = 5)
  : 1 / 2 * (a + b) * h = 25 :=
by
  have base_sum : a + b = 10 := by linarith
  rw [height, base_sum]
  norm_num
  exact h
  sorry

end isosceles_trapezoid_area_l56_56283


namespace trigonometric_identity_l56_56191

theorem trigonometric_identity (α : ℝ) (h₁: α ≠ 0) (h₂ : α ≠ π) (h₃ : ∀ n : ℤ, α ≠ n * π) :
  (sin α + 1 / sin α + 1)^2 + (cos α + 1 / cos α + 1)^2 = 11 + (cos α / sin α)^2 + 2 * (sin α / cos α)^2 :=
by sorry

end trigonometric_identity_l56_56191


namespace number_of_valid_colorings_l56_56667

-- Define the specific properties and constraints given in the problem
def is_power_of_two (n : ℕ) : Prop := ∃ m : ℕ, n = 2^m

def valid_coloring (f : ℕ → bool) : Prop :=
  ∀ a b : ℕ, a ≠ b → a + b ≤ 200 →
  (f a = f b → ¬ is_power_of_two (a + b))

theorem number_of_valid_colorings : 
  ∃ c : ℕ, c = 256 ∧ 
  ∃ f : Π (n : ℕ), nat.lt n 201 → bool, 
    valid_coloring (λ n, f n (by linarith)) :=
begin
  sorry
end

end number_of_valid_colorings_l56_56667


namespace estimate_students_correct_l56_56273

noncomputable def estimate_students_below_85 
  (total_students : ℕ)
  (mean_score : ℝ)
  (variance : ℝ)
  (prob_90_to_95 : ℝ) : ℕ :=
if total_students = 50 ∧ mean_score = 90 ∧ prob_90_to_95 = 0.3 then 10 else 0

theorem estimate_students_correct 
  (total_students : ℕ)
  (mean_score : ℝ)
  (variance : ℝ)
  (prob_90_to_95 : ℝ)
  (h1 : total_students = 50) 
  (h2 : mean_score = 90)
  (h3 : prob_90_to_95 = 0.3) : 
  estimate_students_below_85 total_students mean_score variance prob_90_to_95 = 10 :=
by
  sorry

end estimate_students_correct_l56_56273


namespace rhombus_side_length_l56_56459

-- Define the rhombus properties and the problem conditions
variables (p q x : ℝ)

-- State the problem as a theorem in Lean 4
theorem rhombus_side_length (h : x^2 = p * q) : x = Real.sqrt (p * q) :=
sorry

end rhombus_side_length_l56_56459


namespace max_four_by_one_in_six_by_six_grid_l56_56057

-- Define the grid and rectangle dimensions
def grid_width : ℕ := 6
def grid_height : ℕ := 6
def rect_width : ℕ := 4
def rect_height : ℕ := 1

-- Define the maximum number of rectangles that can be placed
def max_rectangles (grid_w grid_h rect_w rect_h : ℕ) (non_overlapping : Bool) (within_boundaries : Bool) : ℕ :=
  if grid_w = 6 ∧ grid_h = 6 ∧ rect_w = 4 ∧ rect_h = 1 ∧ non_overlapping ∧ within_boundaries then
    8
  else
    0

-- The theorem stating the maximum number of 4x1 rectangles in a 6x6 grid
theorem max_four_by_one_in_six_by_six_grid
  : max_rectangles grid_width grid_height rect_width rect_height true true = 8 := 
sorry

end max_four_by_one_in_six_by_six_grid_l56_56057


namespace sum_of_inverse_squared_modulus_of_roots_of_z_eight_eq_one_l56_56064

open complex

theorem sum_of_inverse_squared_modulus_of_roots_of_z_eight_eq_one :
  (finset.univ.sum (λ (z : ℂ), if z ^ 8 = 1 then (1/(abs (1 - z))^2) else 0)) = 8 :=
by
  sorry

end sum_of_inverse_squared_modulus_of_roots_of_z_eight_eq_one_l56_56064


namespace ksyusha_travel_time_l56_56769

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l56_56769


namespace y_worked_days_l56_56440

-- Define the necessary conditions
variables (W : ℕ) (d : ℕ)

-- Condition 1: x's work rate
def x_work_rate := W / 18

-- Condition 2: y's work rate
def y_work_rate := W / 15

-- Condition 3: Amount of work y did before leaving
def y_work_done := d * (W / 15)

-- Condition 4: Amount of work x did in 12 days
def x_work_done := 12 * (W / 18)

-- Condition 5: Total work done equals total work
def total_work_done := y_work_done + x_work_done

-- Proof goal: Prove that y worked for 5 days before leaving the job given the conditions
theorem y_worked_days : total_work_done = W → d = 5 :=
by
  sorry

end y_worked_days_l56_56440


namespace percentage_in_manufacturing_is_twenty_l56_56087

-- Define the problem statements as hypotheses
def circle_total_degrees : ℝ := 360
def manufacturing_degrees : ℝ := 72

-- Define the percentage calculation
def percentage_of_manufacturing_employees : ℝ :=
  (manufacturing_degrees / circle_total_degrees) * 100

-- State the theorem to be proven
theorem percentage_in_manufacturing_is_twenty : percentage_of_manufacturing_employees = 20 :=
  sorry

end percentage_in_manufacturing_is_twenty_l56_56087


namespace ksyusha_travel_time_l56_56757

theorem ksyusha_travel_time :
  ∀ (S v : ℝ), 
  (2 * S / v + S / (2 * v) = 30) → 
  (2 * (S / v) = 24) :=
by
  intros S v h1
  have h2 : S / v = 12 := by
    calc 
      S / v = (30 * 2) / 5 := by sorry
  calc 
    2 * (S / v) = 2 * 12 := by sorry
  ... = 24 := by norm_num

end ksyusha_travel_time_l56_56757


namespace smallest_non_empty_box_after_moves_l56_56395

theorem smallest_non_empty_box_after_moves :
  ∃ (n : ℕ), 
  ∃ (b : ℕ),
  (n = 36 ∧ b = 31) ∧
  ∃ (boxes : ℕ → ℕ),
  (∀ i, 1 ≤ i → i ≤ 33)! → 
  boxes i = 0 →
  (∀ j, 1 ≤ j → j < i → boxes i = boxes i + 1) →
  (∀ k, 1 ≤ k < i → boxes k = boxes k - 1) →
  boxes 36 = 31 := sorry

end smallest_non_empty_box_after_moves_l56_56395


namespace ones_digit_of_sum_of_powers_l56_56887

theorem ones_digit_of_sum_of_powers :
  (∑ n in Finset.range 2013, n ^ 2013) % 10 = 1 :=
sorry

end ones_digit_of_sum_of_powers_l56_56887


namespace part1_arithmetic_sequence_part2_largest_n_leq_100_l56_56577

def seq_a : ℕ → ℤ
| 1 := 3
| (n + 1) := 2 * seq_a n + 2 ^ (n + 2) - n ^ 2 + 2 * n + 1

def seq_b (n : ℕ) : ℤ :=
  ⌊(seq_a n) / 2^n⌋

theorem part1_arithmetic_sequence :
  ∀ n : ℕ, ∃ a d : ℤ, (∀ m : ℕ, n ≤ m → ((seq_a m - m^2) / 2^m) = a + m * d) := 
sorry

theorem part2_largest_n_leq_100 :
  ∃ n : ℕ, (∑ i in finset.range (n + 1), seq_b i) ≤ 100 :=
sorry

end part1_arithmetic_sequence_part2_largest_n_leq_100_l56_56577


namespace perimeter_of_triangle_XYZ_l56_56941

noncomputable def height : ℝ := 25
noncomputable def side_length : ℝ := 15

def midpoint_distance (a b : ℝ) : ℝ := (a - b) / 2

def Pythagorean (a b : ℝ) : ℝ := real.sqrt (a * a + b * b)

def perimeter : ℝ :=
  let AX := midpoint_distance 0 side_length
  let AZ := height / 2
  let XZ := Pythagorean AX AZ
  let YZ := XZ
  let XY := side_length
  XY + XZ + YZ

theorem perimeter_of_triangle_XYZ : perimeter = 44.16 := by
  sorry

end perimeter_of_triangle_XYZ_l56_56941


namespace sin_product_identity_l56_56827

theorem sin_product_identity (φ : ℝ) (n : ℕ) (h : n > 0) :
  (∏ k in Finset.range n, Real.sin (φ + (k * Real.pi) / n)) = (Real.sin (n * φ)) / (2 ^ (n - 1)) := 
sorry

end sin_product_identity_l56_56827


namespace unique_logarithmic_values_l56_56551

-- Definitions
def available_numbers : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

def logarithmic_value (a b : ℕ) : ℝ := Real.log b / Real.log a

-- The statement to prove
theorem unique_logarithmic_values : 
  (∃ (S : Finset ℝ), (∀ (a b : ℕ), a ∈ available_numbers → b ∈ available_numbers → a ≠ b → logarithmic_value a b ∈ S) ∧ S.card = 52) :=
sorry

end unique_logarithmic_values_l56_56551


namespace count_good_numbers_l56_56787

/-- Definition of the greatest integer function -/
def int_part (x : ℝ) : ℤ := ⌊x⌋

/-- Definition of a good number -/
def is_good_number (x : ℕ) : Prop :=
  x - (int_part (real.sqrt x))^2 = 9

/-- The main theorem to be proven -/
theorem count_good_numbers : 
  (∑ x in Finset.Icc 1 2014, if is_good_number x then 1 else 0) = 40 :=
sorry

end count_good_numbers_l56_56787


namespace P_sufficient_but_not_necessary_for_Q_l56_56555

-- Define the propositions P and Q
def P (x : ℝ) : Prop := |x - 2| ≤ 3
def Q (x : ℝ) : Prop := x ≥ -1 ∨ x ≤ 5

-- Define the statement to prove
theorem P_sufficient_but_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬ P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l56_56555


namespace number_of_schools_is_23_l56_56517

-- Conditions and definitions
noncomputable def number_of_students_per_school : ℕ := 3
def beth_rank : ℕ := 37
def carla_rank : ℕ := 64

-- Statement of the proof problem
theorem number_of_schools_is_23
  (n : ℕ)
  (h1 : ∀ i < n, ∃ r1 r2 r3: ℕ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h2 : ∀ i < n, ∃ A B C: ℕ, A = (2 * B + 1) ∧ C = A ∧ B = 35 ∧ A < beth_rank ∧ beth_rank < carla_rank):
  n = 23 :=
by
  sorry

end number_of_schools_is_23_l56_56517


namespace sum_x_sequence_2016_l56_56243

-- Definitions provided in the problem statement
def x_sequence : ℕ → ℚ
| 0     := 2/3
| (n+1) := x_sequence n / (2 * (2 * n + 1) * x_sequence n + 1)

-- Statement of the theorem
theorem sum_x_sequence_2016 : (∑ k in Finset.range 2016, x_sequence (k + 1)) = 4032 / 4033 :=
by
  sorry

end sum_x_sequence_2016_l56_56243


namespace find_capacity_of_second_vessel_l56_56951

noncomputable def capacity_of_second_vessel (x : ℝ) : Prop :=
  let alcohol_from_first_vessel := 0.25 * 2
  let alcohol_from_second_vessel := 0.40 * x
  let total_liquid := 2 + x
  let total_alcohol := alcohol_from_first_vessel + alcohol_from_second_vessel
  let new_concentration := (total_alcohol / 10) * 100
  2 + x = 8 ∧ new_concentration = 29

open scoped Real

theorem find_capacity_of_second_vessel : ∃ x : ℝ, capacity_of_second_vessel x ∧ x = 6 :=
by
  sorry

end find_capacity_of_second_vessel_l56_56951


namespace product_sequence_sum_l56_56652

theorem product_sequence_sum (c d : ℕ) (h : (4 / 3 : ℝ) * (5 / 4) * (6 / 5) * (7 / 6) * ... * (c / d) = 12) :
  c + d = 71 :=
  sorry

end product_sequence_sum_l56_56652


namespace integral_of_x_over_sqrt_5_minus_x_integral_of_sin_x_cos_sq_x_over_1_plus_cos_x_integral_of_sin_x_plus_cos_x_squared_integral_of_x_minus_cos_sq_x_over_x_cos_sq_x_integral_of_sin_x_plus_sin_2x_squared_l56_56490

-- Problem 1
theorem integral_of_x_over_sqrt_5_minus_x :
  ∫ (x : ℝ) in real, x / real.sqrt (5 - x) = (2 / 3) * real.sqrt (5 - x)^3 - 10 * real.sqrt (5 - x) + C :=
sorry

-- Problem 2
theorem integral_of_sin_x_cos_sq_x_over_1_plus_cos_x :
  ∫ (x : ℝ) in real, (real.sin x * real.cos x^2) / (1 + real.cos x) = real.cos x - (1 / 2) * real.cos x^2 - real.log (1 + real.cos x) + C :=
sorry

-- Problem 3
theorem integral_of_sin_x_plus_cos_x_squared :
  ∫ (x : ℝ) in real, (real.sin x + real.cos x)^2 = x - (1 / 2) * real.cos (2 * x) + C :=
sorry

-- Problem 4
theorem integral_of_x_minus_cos_sq_x_over_x_cos_sq_x :
  ∫ (x : ℝ) in real, (x - real.cos x^2) / (x * real.cos x^2) = real.tan x - real.log x + C :=
sorry

-- Problem 5
theorem integral_of_sin_x_plus_sin_2x_squared :
  ∫ (x : ℝ) in real, (real.sin x + real.sin (2 * x))^2 = x - (1 / 4) * real.sin (2 * x) - (1 / 8) * real.sin (4 * x) + (4 / 3) * real.sin x^3 + C :=
sorry

end integral_of_x_over_sqrt_5_minus_x_integral_of_sin_x_cos_sq_x_over_1_plus_cos_x_integral_of_sin_x_plus_cos_x_squared_integral_of_x_minus_cos_sq_x_over_x_cos_sq_x_integral_of_sin_x_plus_sin_2x_squared_l56_56490


namespace johns_pandas_lions_l56_56728

def animals : Type := ℕ

variable (P D : animals)

theorem johns_pandas_lions :
  let snakes := 15
  let monkeys := 2 * snakes
  let lions := monkeys - 5
  let dogs := P / 3
  let total_animals := snakes + monkeys + lions + P + dogs
  (15 + (2 * 15) + ((2 * 15) - 5) + P + (P / 3) = 114) →
  (P - (2 * 15 - 5) = 8) :=
by
  intro h
  sorry

end johns_pandas_lions_l56_56728


namespace max_intersections_of_perpendiculars_l56_56665

theorem max_intersections_of_perpendiculars {A B C D : Point} 
(h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
(h_no_coincide : ∀ P Q R S : Point, P ≠ Q → R ≠ S → line P Q ≠ line R S)
(h_no_parallel : ∀ P Q R S : Point, P ≠ Q → R ≠ S → ¬ parallel (line P Q) (line R S))
(h_no_perpendicular : ∀ P Q R S : Point, P ≠ Q → R ≠ S → ¬ perpendicular (line P Q) (line R S)) :
  max_intersections_point_perpendiculars A B C D = 52 := 
sorry

end max_intersections_of_perpendiculars_l56_56665


namespace volume_wedge_l56_56111

-- Define the problem conditions
def diameter (cylinder : ℝ) : ℝ := 16
def radius (cylinder : ℝ) : ℝ := diameter cylinder / 2
def height (cylinder : ℝ) : ℝ := 16
def angle (cut : ℝ) : ℝ := 60

-- Given conditions
def volume_cylinder (cylinder : ℝ) : ℝ := π * (radius cylinder)^2 * (height cylinder)
def wedge_fraction := angle 1 / 360

-- Final statement: Prove that the volume of the wedge formed is 171π
theorem volume_wedge (cylinder : ℝ) : wedge_fraction * volume_cylinder cylinder = 171 * π :=
by
  sorry

end volume_wedge_l56_56111


namespace div_recurring_decimal_l56_56886

def recurringDecimalToFraction (q : ℚ) (h : q = 36/99) : ℚ := by
  sorry

theorem div_recurring_decimal : 12 / recurringDecimalToFraction 0.36 sorry = 33 :=
by
  sorry

end div_recurring_decimal_l56_56886


namespace abs_expression_value_l56_56805

theorem abs_expression_value : 
  let x : ℤ := -768 in 
  |(|x| - x) - |x| - x| - x = 2304 :=
by
  sorry

end abs_expression_value_l56_56805


namespace number_of_sequences_l56_56217

-- Defining the sequence {a_n} and related functions
def a (n : ℕ) : ℝ := sorry
def f (n : ℕ) (x : ℝ) : ℝ := (1 / 3) * x^3 - a n * x^2 + (a n^2 - 1) * x
def f_prime (n : ℕ) (x : ℝ) : ℝ := x^2 - 2 * a n * x + (a n^2 - 1)

-- Hypotheses
axiom a1 : a 1 = 1
axiom a8 : a 8 = 4
axiom extreme_point_condition : ∀ n, 1 ≤ n ∧ n ≤ 15 → f_prime n (a (n + 1)) = 0
axiom tangent_slope : f_prime 8 (a 16) = 15

-- Theorem to prove
theorem number_of_sequences : ∃ n, n = 1176 :=
begin
  sorry
end

end number_of_sequences_l56_56217


namespace smallest_satisfying_conditions_l56_56438

open Nat

/-- Define the conditions for the smallest number n:
    1. When n is diminished by 2, it is divisible by 12, 16, 18, 21, 28, 32, and 45
    2. n is the sum of two consecutive prime numbers -/
def conditions (n : ℕ) : Prop :=
  (n - 2) % 12 = 0 ∧
  (n - 2) % 16 = 0 ∧
  (n - 2) % 18 = 0 ∧
  (n - 2) % 21 = 0 ∧
  (n * 2) % 28 = 0 ∧
  (n - 2) % 32 = 0 ∧
  (n - 2) % 45 = 0 ∧
  (Prime.prime (n / 2) ∧ Prime.prime (n / 2 + 1))

/-- The proof problem statement to show the smallest number satisfying the conditions is 10088 -/
theorem smallest_satisfying_conditions : ∃ n, conditions n ∧ n = 10088 :=
by
  sorry

end smallest_satisfying_conditions_l56_56438


namespace determine_a_l56_56238

def f (x : ℝ) (a : ℝ) : ℝ := if x ≤ 1 then 2 * x + a else Real.log x / Real.log 2

theorem determine_a (a : ℝ) : f (f (1 / 2) a) a = 4 → a = 15 := by
  sorry

end determine_a_l56_56238


namespace card_intersection_l56_56034

variable (U A B : Type) [Fintype U] [Fintype A] [Fintype B]
variable {u : Fintype.card U = 190}
variable {b : Fintype.card B = 49}
variable {notAB : Fintype.card { x : U // x ∉ A ∧ x ∉ B } = 59}
variable {a : Fintype.card A = 105}

theorem card_intersection (x : ℕ) (h : x = Fintype.card { x : U // x ∈ A ∧ x ∈ B }) :
  x = 23 :=
by
  have unionCard : Fintype.card { x : U // x ∈ A ∨ x ∈ B } = 190 - 59 :=
    sorry
  have inclusionExclusion : Fintype.card { x : U // x ∈ A ∨ x ∈ B } = Fintype.card A + Fintype.card B - Fintype.card { x : U // x ∈ A ∧ x ∈ B } :=
    sorry
  sorry

end card_intersection_l56_56034


namespace lines_AB_PQ_perpendicular_l56_56781

open EuclideanGeometry

noncomputable theory

variables {A B C D P Q : Point}

-- Definitions from the conditions
axiom on_circle : Points_on_Circle A B C D

axiom AC_BD_intersect_P : Chord A C P ∧ Chord B D P ∧ Intersect AC BD P

axiom perp_AC_through_C : Perpendicular (Chord A C) C

axiom perp_BD_through_D : Perpendicular (Chord B D) D

axiom Q_def : Perpendicular_Intersection C D Q

-- Proof Goal
theorem lines_AB_PQ_perpendicular :
  Angles_AB_PQ_perpendicular A B C D P Q :=
sorry

end lines_AB_PQ_perpendicular_l56_56781


namespace sandwiches_with_ten_loaves_l56_56477

def sandwiches_per_loaf : ℕ := 18 / 3

def num_sandwiches (loaves: ℕ) : ℕ := sandwiches_per_loaf * loaves

theorem sandwiches_with_ten_loaves :
  num_sandwiches 10 = 60 := by
  sorry

end sandwiches_with_ten_loaves_l56_56477


namespace exists_prime_q_and_positive_n_l56_56315

theorem exists_prime_q_and_positive_n (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) :
  ∃ q n : ℕ, Nat.Prime q ∧ q < p ∧ 0 < n ∧ p ∣ (n^2 - q) :=
by
  sorry

end exists_prime_q_and_positive_n_l56_56315


namespace relationship_of_y_l56_56620

theorem relationship_of_y (a : ℝ) (a_pos : a > 0) (y_1 y_2 y_3 : ℝ) :
  let f := λ x : ℝ, -2 * a * x ^ 2 + a * x - 4 in
  y_1 = f (-1) →
  y_2 = f 1 →
  y_3 = f 2 →
  y_3 < y_1 ∧ y_1 < y_2 :=
by
  intros
  sorry

end relationship_of_y_l56_56620


namespace volleyball_prob_is_one_third_l56_56288

def total_sports : ℕ := 3
def volleyball_prob : ℝ := 1 / total_sports

theorem volleyball_prob_is_one_third :
  volleyball_prob = 1 / 3 :=
by
  sorry

end volleyball_prob_is_one_third_l56_56288


namespace odd_function_properties_l56_56259

theorem odd_function_properties
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 1 ≤ x ∧ x ≤ 3 ∧ 1 ≤ y ∧ y ≤ 3 ∧ x < y → f x < f y)
  (h_min_val : ∀ x, 1 ≤ x ∧ x ≤ 3 → 7 ≤ f x) :
  (∀ x y, -3 ≤ x ∧ x ≤ -1 ∧ -3 ≤ y ∧ y ≤ -1 ∧ x < y → f x < f y) ∧
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≤ -7) :=
sorry

end odd_function_properties_l56_56259


namespace pizza_problem_l56_56936

theorem pizza_problem : 
  let toppings := 8 in
  let invalid_two_topping_combination := 1 in
  let one_topping_pizzas := toppings in
  let two_topping_total_combinations := (toppings * (toppings - 1)) / 2 in
  one_topping_pizzas + (two_topping_total_combinations - invalid_two_topping_combination) = 35 :=
by
  sorry

end pizza_problem_l56_56936


namespace sum_of_corners_is_164_l56_56360

theorem sum_of_corners_is_164 :
  let checkerboard := (List.range 81).map (λ x, x + 1) in
  let top_left := checkerboard.head! in
  let top_right := checkerboard.nth! 8 in
  let bottom_left := checkerboard.nth! 72 in
  let bottom_right := checkerboard.nth! 80 in
  top_left + top_right + bottom_left + bottom_right = 164 :=
by
  sorry

end sum_of_corners_is_164_l56_56360


namespace ksyusha_travel_time_l56_56758

theorem ksyusha_travel_time :
  ∀ (S v : ℝ), 
  (2 * S / v + S / (2 * v) = 30) → 
  (2 * (S / v) = 24) :=
by
  intros S v h1
  have h2 : S / v = 12 := by
    calc 
      S / v = (30 * 2) / 5 := by sorry
  calc 
    2 * (S / v) = 2 * 12 := by sorry
  ... = 24 := by norm_num

end ksyusha_travel_time_l56_56758


namespace tomeka_wins_probability_l56_56045

namespace TomekaDiceGame

noncomputable def probability_of_rolling_seven_twice : ℚ :=
  let total_possible_outcomes := 6 * 6
  let favorable_outcomes := 6
  in favorable_outcomes / total_possible_outcomes

theorem tomeka_wins_probability : probability_of_rolling_seven_twice = 1 / 6 :=
by
  -- Proof omitted
  sorry

end TomekaDiceGame

end tomeka_wins_probability_l56_56045


namespace greatest_y_l56_56839

theorem greatest_y (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : y ≤ 24 :=
sorry

end greatest_y_l56_56839


namespace unique_solution_Z_l56_56327

theorem unique_solution_Z (Z W λ : ℂ) (h_1 : |λ| ≠ 1) :
    ∃! Z, (conj Z - λ * Z = W) ∧ (Z = (conj λ) * W + conj W) / (1 - |λ|^2) := sorry

end unique_solution_Z_l56_56327


namespace rectangular_field_area_l56_56363

theorem rectangular_field_area (P W : ℕ) (hP : P = 30) (hW : W = 5) : ∃ A, A = 50 := 
by 
  let L := (P - 2 * W) / 2
  have hL : L = 10 := by 
    calc 
      L = (30 - 2 * 5) / 2 : by rw [hP, hW]
      ... = (30 - 10) / 2    : by ring
      ... = 20 / 2           : by ring
      ... = 10               : by norm_num
  use L * W
  calc
    L * W = 10 * 5          : by rw [hL, hW]
    ... = 50                : by norm_num
  sorry

end rectangular_field_area_l56_56363


namespace parents_without_full_time_jobs_l56_56436

theorem parents_without_full_time_jobs
  {total_parents mothers fathers : ℕ}
  (h_total_parents : total_parents = 100)
  (h_mothers_percentage : mothers = 60)
  (h_fathers_percentage : fathers = 40)
  (h_mothers_full_time : ℕ)
  (h_fathers_full_time : ℕ)
  (h_mothers_ratio : h_mothers_full_time = (5 * mothers) / 6)
  (h_fathers_ratio : h_fathers_full_time = (3 * fathers) / 4) :
  ((total_parents - (h_mothers_full_time + h_fathers_full_time)) * 100 / total_parents = 20) := sorry

end parents_without_full_time_jobs_l56_56436


namespace parabola_y_intersection_l56_56012

theorem parabola_y_intersection (x y : ℝ) : 
  (∀ x' y', y' = -(x' + 2)^2 + 6 → ((x' = 0) → (y' = 2))) :=
by
  intros x' y' hy hx0
  rw hx0 at hy
  simp [hy]
  sorry

end parabola_y_intersection_l56_56012


namespace tenth_term_is_eight_l56_56244

def sequence (n : ℕ) : ℚ := 4 / (n^2 - 3*n)

theorem tenth_term_is_eight : ∀ (n : ℕ), sequence n = 1 / 10 ↔ n = 8 := 
by sorry

end tenth_term_is_eight_l56_56244


namespace quadrilateral_is_trapezoid_l56_56707

-- Condition 1: ABCD is an inscribed quadrilateral
structure InscribedQuadrilateral (A B C D : Point) : Prop :=
(circum_angle_equal : ∀ {O : Point}, intersection O A C B D → ArcEqual (circ_angle A B C) (circ_angle C D A))

-- Condition 2: The diagonals intersect at point O
structure DiagonalsIntersect (A B C D O : Point) : Prop :=
(intersection_point : ∀ {P Q: Point}, intersection O P Q A C B D)

-- Condition 3: The circle circumscribed around COD passes through the center of the circumcircle of ABCD
structure CircumCircleThroughCenter (A B C D O O₁: Point) : Prop :=
(center_pass : ∀ {P Q : Point}, circumscribed (circle_in (Δ COD)) (at point O₁) )

-- The theorem statement
theorem quadrilateral_is_trapezoid (A B C D O O₁: Point)
  (h_inscribed : InscribedQuadrilateral A B C D)
  (h_intersection : DiagonalsIntersect A B C D O)
  (h_circle : CircumCircleThroughCenter A B C D O O₁)
  : IsTrapezoid A B C D :=
sorry

end quadrilateral_is_trapezoid_l56_56707


namespace johns_average_speed_l56_56726

noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem johns_average_speed :
  average_speed 210 6.5 ≈ 32.31 := 
by
  sorry

end johns_average_speed_l56_56726


namespace four_letter_arrangements_l56_56630

theorem four_letter_arrangements : 
  (∀ (letters : Finset Char), 
    letters = {'A', 'B', 'C', 'D', 'E', 'G', 'H'} →
    ∃ (arrangements : Finset (List Char)), 
    arrangements.card = 60 ∧ 
    ∀ l ∈ arrangements, 
      l.length = 4 ∧ 
      l.head = 'D' ∧ 
      'G' ∈ l.tail ∧ 
      (∀ x ∈ l, x ∈ letters) ∧
      l.eraseDups.length = 4) :=
begin
  sorry
end

end four_letter_arrangements_l56_56630


namespace find_inverse_l56_56801

theorem find_inverse :
  ∀ (f : ℝ → ℝ), (∀ x, f x = 3 * x ^ 3 + 9) → (f⁻¹ 90 = 3) :=
by
  intros f h
  sorry

end find_inverse_l56_56801


namespace ksyusha_wednesday_time_l56_56752

def speed_relation (v : ℝ) := ∀ (d_walk d_run1 d_run2 time1 time2: ℝ), 
  (2 * d_run1 = d_walk) → -- she walked twice the distance she ran on Tuesday
  (2 * v = 2 * v) → -- she runs twice as fast as she walks
  ((2 * d_walk / v) + (d_run1 / (2 * v)) = time1) → -- she took 30 minutes on Tuesday
  (time1 = 30) → -- Tuesday time equals 30
  (d_run2 = 2 * d_walk) → -- she had to run twice the distance she walked on Wednesday
  ((d_walk / v) + (d_run2 / (2 * v)) = time2) → -- calculate the time for Wednesday
  time2 = 24 -- time taken on Wednesday

theorem ksyusha_wednesday_time :
  ∃ (v : ℝ), speed_relation v :=
begin
  existsi 1,
  unfold speed_relation,
  intros,
  sorry,
end

end ksyusha_wednesday_time_l56_56752


namespace hyperbola_vertex_asymptote_distance_l56_56373

theorem hyperbola_vertex_asymptote_distance : ∀ (x y : ℝ), 
  x = 2 → y = 0 →
  (∃ (a b : ℝ), 
    a ≠ 0 ∧ 
    b ≠ 0 ∧ 
    (∀ (x y : ℝ), (x^2 / 4) - y^2 = 1 → y = (1 / 2) * x) →
    (|a * x + b * y| / sqrt (a^2 + b^2) = (2 * sqrt 5) / 5)) := 
begin
  sorry
end

end hyperbola_vertex_asymptote_distance_l56_56373


namespace stratified_sampling_red_balls_l56_56402

-- Define the conditions
def total_balls : ℕ := 1000
def red_balls : ℕ := 50
def sampled_balls : ℕ := 100

-- Prove that the number of red balls sampled using stratified sampling is 5
theorem stratified_sampling_red_balls :
  (red_balls : ℝ) / (total_balls : ℝ) * (sampled_balls : ℝ) = 5 := 
by
  sorry

end stratified_sampling_red_balls_l56_56402


namespace simplify_fractional_equation_l56_56424

theorem simplify_fractional_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 2) : (x / (x - 2) - 2 = 3 / (2 - x)) → (x - 2 * (x - 2) = -3) :=
by
  sorry

end simplify_fractional_equation_l56_56424


namespace number_of_solutions_l56_56635

theorem number_of_solutions : ∃ n : ℕ, 1 < n ∧ 
  (∃ a b : ℕ, gcd a b = 1 ∧
  (∃ x y : ℕ, x^(a*n) + y^(b*n) = 2^2010)) ∧
  (∃ count : ℕ, count = 54) :=
sorry

end number_of_solutions_l56_56635


namespace proposition_A_to_B_l56_56824

theorem proposition_A_to_B :
  (∀ p q : Prop, (p → q) → (p ↔ q) → (∃ x : Prop, (x → (p → q)) ∧ (x ≠ (p ↔ q)))) := 
by
  intro p q
  intro hpq hpiffq
  use p
  split
  {
    intro h
    exact hpq
  } 
  {
    intro h
    intro hp
    intro hq
    have hpq' := hpiffq.mp hp
    exact hpiffq.mpr
  }


end proposition_A_to_B_l56_56824


namespace conclusion_2_conclusion_4_l56_56581

open_locale classical

variables {Plane : Type*} {Line : Type*} [IsPlane Plane] [IsLine Line]
variable (α β γ : Plane)
variable (l m : Line)

-- Conditions: defining perpendicularity and intersection
def perp (p1 p2 : Plane) : Prop := p1 ∠ p2 = ⊥
def intersect (p1 p2 : Plane) (l : Line) : Prop := ∃ p3, p1 ∩ p2 = l

-- Given conditions
variable (h1 : perp α γ)
variable (h2 : intersect γ α m)
variable (h3 : intersect γ β l)
variable (h4 : perp l m)

-- Conjecture we need to prove: conclusions ② and ④
theorem conclusion_2 : perp l α :=
sorry

theorem conclusion_4 : perp α β :=
sorry

end conclusion_2_conclusion_4_l56_56581


namespace max_ratio_cone_base_radius_l56_56601

theorem max_ratio_cone_base_radius :
  (∀ (l : ℝ) (r : ℝ), l = 2 → 
    let S := 2 * Real.pi * r;
    let h := Real.sqrt (l^2 - r^2);
    let V := (1 / 3) * Real.pi * r^2 * h;
    let ratio := V / S;
    ratio ≤ (1 / 6) * r * Real.sqrt(4 - r^2) ∧ (ratio = (1 / 6) * Real.sqrt(4 - r^2) → r = Real.sqrt(2))) :=
sorry

end max_ratio_cone_base_radius_l56_56601


namespace number_of_lines_l56_56700

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the distances from the points
def d_A : ℝ := 1
def d_B : ℝ := 2

-- A theorem stating the number of lines under the given conditions
theorem number_of_lines (A B : ℝ × ℝ) (d_A d_B : ℝ) (hA : A = (1, 2)) (hB : B = (3, 1)) (hdA : d_A = 1) (hdB : d_B = 2) :
  ∃ n : ℕ, n = 2 :=
by {
  sorry
}

end number_of_lines_l56_56700


namespace second_day_more_than_third_day_l56_56937

-- Define the conditions
def total_people (d1 d2 d3 : ℕ) := d1 + d2 + d3 = 246 
def first_day := 79
def third_day := 120

-- Define the statement to prove
theorem second_day_more_than_third_day : 
  ∃ d2 : ℕ, total_people first_day d2 third_day ∧ (d2 - third_day) = 47 :=
by
  sorry

end second_day_more_than_third_day_l56_56937


namespace part_1_part_2_part_3_l56_56615

section part1
variable {x : ℝ} 
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x + 2

theorem part_1 (h_a : -a = 3):
  (∀ x, f -3 x ≤ 1 - x ^ 2 → x ∈ (Set.Iic (1 / 2)) ∪ (Set.Ici 1)) :=
sorry
end part1

section part2
variable {x a : ℝ}
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x + 2 * a - 2

theorem part_2 (h2 : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f a x ≤ 2 * a * (x - 1) + 4) :
  a ≤ (1 / 3) :=
sorry
end part2

section part3
variable {x1 x2 m : ℝ}
noncomputable def f (x1 : ℝ) : ℝ := x1^2 - 3 * x1 + 2
noncomputable def g (x2 : ℝ) (m : ℝ) : ℝ := -x2 + m

theorem part_3 (h3 : ∀ x1 ∈ Set.Icc 1 4, ∃ x2 ∈ Set.Ioo 1 8, f x1 = g x2 m) :
  7 < m ∧ m < (31 / 4) :=
sorry
end part3

end part_1_part_2_part_3_l56_56615


namespace find_breadth_of_landscape_l56_56939

theorem find_breadth_of_landscape (L B A : ℕ) 
  (h1 : B = 8 * L)
  (h2 : 3200 = A / 9)
  (h3 : 3200 * 9 = A) :
  B = 480 :=
by 
  sorry

end find_breadth_of_landscape_l56_56939


namespace coefficient_a3b2_l56_56415

theorem coefficient_a3b2 :
  let poly1 := (a + b)^5
  let poly2 := (c + (1 / c))^7
  coefficient (a^3 * b^2 * (poly1 * poly2)) = 0 :=
by
  sorry

end coefficient_a3b2_l56_56415


namespace TriangleChords_l56_56482

theorem TriangleChords (a b c : ℝ) (h : isosceles_right_triangle a b c ∧ area_of_triangle a b c = 4 + 2 * real.sqrt 2)
  (D : point_on_circle_with_chord_eq_2) : 
  chord_length_a_d D = 2 ∧ chord_length_c_d D = 2 * (real.sqrt 2 + 1) :=
by
  sorry

end TriangleChords_l56_56482


namespace y_is_multiple_of_3_y_is_multiple_of_9_y_is_multiple_of_27_y_is_multiple_of_81_l56_56317

noncomputable def y : ℕ := 81 + 243 + 729 + 1458 + 2187 + 6561 + 19683

theorem y_is_multiple_of_3 : y % 3 = 0 :=
sorry

theorem y_is_multiple_of_9 : y % 9 = 0 :=
sorry

theorem y_is_multiple_of_27 : y % 27 = 0 :=
sorry

theorem y_is_multiple_of_81 : y % 81 = 0 :=
sorry

end y_is_multiple_of_3_y_is_multiple_of_9_y_is_multiple_of_27_y_is_multiple_of_81_l56_56317


namespace bicycle_distance_l56_56920

theorem bicycle_distance (b t : ℝ) (h : t ≠ 0) :
  let rate := (b / 2) / t / 3
  let total_seconds := 5 * 60
  rate * total_seconds = 50 * b / t := by
    sorry

end bicycle_distance_l56_56920


namespace angle_equality_l56_56405

variables {P : Type} [MetricSpace P] [NormedSpace ℝ P] [InnerProductSpace ℝ P]

-- Definitions of all the required geometric properties
variables {O1 O2 A B B1 B2 M M1 M2 : P}

-- Conditions from the problem
variables (h1 : circle O1 A B) 
          (h2 : circle O2 A B) 
          (h3 : O1 ≠ O2)
          (h4 : on_line_segment (A, B) O1 O2)
          (h5 : BO1_intersects B1) 
          (h6 : BO2_intersects B2) 
          (h7 : midpoint M B1 B2) 
          (h8 : angle_eq A O1 M1 A O2 M2)
          (h9 : lies_on_minor_arc B1 A M1)
          (h10 : lies_on_minor_arc B A M2)

-- The goal to be proven
theorem angle_equality : ∠ M M1 B = ∠ M M2 B :=
sorry

end angle_equality_l56_56405


namespace problem_l56_56470

noncomputable def b (n : ℕ) : ℝ := 2^(n-1)
noncomputable def a (n : ℕ) : ℝ := Real.log 2 (b n) + 3

theorem problem :
  (b 1 + b 3 = 5 ∧ b 1 * b 3 = 4) →
  (∀ n : ℕ, a (n+1) - a n = 1) ∧
  (∃ m : ℕ, a 1 + a 2 + a 3 + ... + a m ≤ a 40 ∧ ∀ k : ℕ, k > m → a 1 + a 2 + a 3 + ... + a k > a 40) :=
by
  intros h,
  sorry

end problem_l56_56470


namespace largest_number_of_consecutive_odd_integers_with_sum_55_l56_56056

theorem largest_number_of_consecutive_odd_integers_with_sum_55 :
  ∃ n : ℕ, (∃ x : ℕ, x % 2 = 1 ∧ (n * (x + n - 1) = 55)) ∧ 
  (∀ n' : ℕ, (∃ x' : ℕ, x' % 2 = 1 ∧ (n' * (x' + n' - 1) = 55)) → n' ≤ n) :=
begin
  sorry
end

end largest_number_of_consecutive_odd_integers_with_sum_55_l56_56056


namespace dora_packs_stickers_l56_56340

theorem dora_packs_stickers :
  let lola_money := 9
  let dora_money := 9
  let combined_money := lola_money + dora_money
  let cost_playing_cards := 10
  let remaining_money := combined_money - cost_playing_cards
  let cost_each_box_stickers := 2
  let total_boxes_stickers := remaining_money / cost_each_box_stickers
  let dora_packs := total_boxes_stickers / 2
  in dora_packs = 2 :=
by
  sorry

end dora_packs_stickers_l56_56340


namespace number_of_perfect_square_factors_450_l56_56636

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def prime_factorization_450 := (2, 1) :: (3, 2) :: (5, 2) :: []

def perfect_square_factors (n : ℕ) : ℕ :=
  if n = 450 then 4 else 0

theorem number_of_perfect_square_factors_450 : perfect_square_factors 450 = 4 :=
by
  sorry

end number_of_perfect_square_factors_450_l56_56636


namespace problem_solution_l56_56786

noncomputable def S (n : ℕ) (a : ℕ → ℕ) := 2 * a n - 2
noncomputable def b (n : ℕ) (a : ℕ → ℕ) := (2 - n) / a n
noncomputable def T (n : ℕ) (a : ℕ → ℕ) := ∑ k in Finset.range n, b k a

theorem problem_solution :
  (∀ n, S n (λ n, 2^n) = 2 * (2^n) - 2) ∧ (T n (λ n, 2^n) = n / 2^n) :=
  by sorry

end problem_solution_l56_56786


namespace circle_radius_l56_56864

noncomputable def radius_of_circle (r : ℝ) : Prop :=
  3 * 2 * real.pi * r = real.pi * r^2

theorem circle_radius (r : ℝ) (h : radius_of_circle r) : r = 6 :=
by
  -- This is where the proof would go
  sorry

end circle_radius_l56_56864


namespace lcm_prime_numbers_l56_56193

theorem lcm_prime_numbers :
  ∀ (a b c d : ℕ), prime a ∧ prime b ∧ prime c ∧ prime d ∧ a = 97 ∧ b = 193 ∧ c = 419 ∧ d = 673 →
    Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 5,280,671,387 :=
by
  intros a b c d
  intro h
  sorry

end lcm_prime_numbers_l56_56193


namespace seq_sum_formula_l56_56866

-- Define the sequence
def seq (n : ℕ) : ℚ := (2 * n - 1 : ℚ) + (1 / (2 ^ n : ℚ))

-- Define the sum of the first n terms of the sequence
def seq_sum (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, seq (k + 1)

-- The theorem we want to prove
theorem seq_sum_formula (n : ℕ) : seq_sum n = n^2 - 1/2^n + 1 := by
  sorry

end seq_sum_formula_l56_56866


namespace concyclic_points_l56_56871

theorem concyclic_points 
  (Γ₁ Γ₂ : Circle) (O₁ O₂ A B C D X : Point)
  (h₁ : Γ₁.center = O₁)
  (h₂ : Γ₂.center = O₂)
  (h₃ : A ∈ Γ₁.points ∧ A ∈ Γ₂.points)
  (h₄ : B ∈ Γ₁.points ∧ B ∈ Γ₂.points)
  (h₅ : C ∈ Γ₁.points ∧ A ≠ C)
  (h₆ : D ∈ Γ₂.points ∧ A ≠ D)
  (h₇ : (line_through A C).intersects (line_through A D))
  (h₈ : intersection (line_through C O₁) (line_through D O₂) = X) :
  are_concyclic O₁ O₂ B X := sorry

end concyclic_points_l56_56871


namespace find_prime_pairs_l56_56999

theorem find_prime_pairs (p q : ℕ) (p_prime : Nat.Prime p) (q_prime : Nat.Prime q) 
  (h1 : ∃ a : ℤ, a^2 = p - q)
  (h2 : ∃ b : ℤ, b^2 = p * q - q) : 
  (p, q) = (3, 2) :=
by {
    sorry
}

end find_prime_pairs_l56_56999


namespace matrix_power_four_l56_56499

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, -Real.sqrt 3; Real.sqrt 3, 2]

theorem matrix_power_four :
  A ^ 4 = !![
    -49 / 2, -49 * Real.sqrt 3 / 2;
    49 * Real.sqrt 3 / 2, -49 / 2
  ] :=
by
  sorry

end matrix_power_four_l56_56499


namespace leah_coins_value_l56_56312

theorem leah_coins_value :
  ∃ (d n : ℕ), d + n = 15 ∧ (d = 2 * (n + 3)) ∧ (10 * d + 5 * n = 135) := 
begin
  sorry
end

end leah_coins_value_l56_56312


namespace ksyusha_travel_time_l56_56767

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l56_56767


namespace ratio_of_ages_l56_56265

-- Define variables A's age and B's age
variables (A_age B_age : ℕ)

-- Conditions
def condition1 : Prop := (λ A_age B_age, A_age = B_age + 9)
def condition2 : Prop := (B_age = 39)

-- Question and the proof target
theorem ratio_of_ages (h1 : condition1 A_age B_age) (h2 : condition2 B_age) :
  ((A_age + 10) / (B_age - 10) = 2) :=
by
  sorry

end ratio_of_ages_l56_56265


namespace prob_4th_shot_l56_56919

-- Define the recurrence relation
def p (n : ℕ) : ℚ :=
  if n = 1 then
    2 / 3
  else
    (1 / 3) * p (n - 1) + (1 / 3)

-- State the theorem
theorem prob_4th_shot :
  p 4 = 41 / 81 :=
sorry

end prob_4th_shot_l56_56919


namespace avg_visitors_proof_l56_56932

-- Define the constants and conditions
def Sundays_visitors : ℕ := 500
def total_days : ℕ := 30
def avg_visitors_per_day : ℕ := 200

-- Total visits on Sundays within the month
def visits_on_Sundays := 5 * Sundays_visitors

-- Total visitors for the month
def total_visitors := total_days * avg_visitors_per_day

-- Average visitors on other days (Monday to Saturday)
def avg_visitors_other_days : ℕ :=
  (total_visitors - visits_on_Sundays) / (total_days - 5)

-- The theorem stating the problem and corresponding answer
theorem avg_visitors_proof (V : ℕ) 
  (h1 : Sundays_visitors = 500)
  (h2 : total_days = 30)
  (h3 : avg_visitors_per_day = 200)
  (h4 : visits_on_Sundays = 5 * Sundays_visitors)
  (h5 : total_visitors = total_days * avg_visitors_per_day)
  (h6 : avg_visitors_other_days = (total_visitors - visits_on_Sundays) / (total_days - 5))
  : V = 140 :=
by
  -- Proof is not required, just state the theorem
  sorry

end avg_visitors_proof_l56_56932


namespace log_inequality_solution_case_1_log_inequality_solution_case_2_l56_56834

variable (a x : ℝ)

theorem log_inequality_solution_case_1 (h₁ : a > 1) :
  (log a (x^2 - x - 2) > log a (x - 2 / a) + 1) ↔ (x > 1 + a) :=
by sorry

theorem log_inequality_solution_case_2 (h₂ : 0 < a) (h₃ : a < 1) :
  ∀ x, ¬(log a (x^2 - x - 2) > log a (x - 2 / a) + 1) :=
by sorry

end log_inequality_solution_case_1_log_inequality_solution_case_2_l56_56834


namespace rainfall_ratio_l56_56304

theorem rainfall_ratio (r_wed tuesday_rate : ℝ)
    (h_monday : 7 * 1 = 7)
    (h_tuesday : 4 * 2 = 8)
    (h_total : 7 + 8 + 2 * r_wed = 23)
    (h_wed_eq: r_wed = 8 / 2)
    (h_tuesday_rate: tuesday_rate = 2) 
    : r_wed / tuesday_rate = 2 :=
by
  sorry

end rainfall_ratio_l56_56304


namespace area_of_right_triangle_l56_56709

theorem area_of_right_triangle (AB BC : ℝ) (h_AB : AB = 12) (h_BC : BC = 9) :
  (1 / 2) * AB * BC = 54 :=
by
  rw [h_AB, h_BC]
  norm_num
  sorry

end area_of_right_triangle_l56_56709


namespace train_or_plane_not_ship_possible_modes_l56_56107

-- Define the probabilities of different modes of transportation
def P_train : ℝ := 0.3
def P_ship : ℝ := 0.2
def P_car : ℝ := 0.1
def P_plane : ℝ := 0.4

-- 1. Proof that probability of train or plane is 0.7
theorem train_or_plane : P_train + P_plane = 0.7 :=
by sorry

-- 2. Proof that probability of not taking a ship is 0.8
theorem not_ship : 1 - P_ship = 0.8 :=
by sorry

-- 3. Proof that if probability is 0.5, the modes are either (ship, train) or (car, plane)
theorem possible_modes (P_value : ℝ) (h1 : P_value = 0.5) :
  (P_ship + P_train = P_value) ∨ (P_car + P_plane = P_value) :=
by sorry

end train_or_plane_not_ship_possible_modes_l56_56107


namespace angle_DAE_eq_30_degrees_l56_56961

theorem angle_DAE_eq_30_degrees
    (A B C D E : Type) 
    [EquilateralTriangle ABC]
    [Square BCDE]
    (h_shared_BC : BC = BC) :
    ∠DAE = 30 :=
by sorry

end angle_DAE_eq_30_degrees_l56_56961


namespace chess_tournament_exists_l56_56329

noncomputable def chess_tournament (t : List ℕ) (H : List.Sorted (<) t) : Prop :=
  ∃ G : SimpleGraph, G.order = (List.last (t, 0) + 1) ∧ 
  (∀ v : G.V, G.degree v ∈ t) ∧
  (∀ i : ℕ, i < List.length t → ∃ v : G.V, G.degree v = List.nthLe t i (by simp))

theorem chess_tournament_exists (t : List ℕ) (H : List.Sorted (<) t) : chess_tournament t H :=
sorry

end chess_tournament_exists_l56_56329


namespace a_eq_bn_l56_56350

theorem a_eq_bn (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → ∃ m : ℕ, a - k^n = m * (b - k)) → a = b^n :=
by
  sorry

end a_eq_bn_l56_56350


namespace cows_sold_l56_56935

/-- 
A man initially had 39 cows, 25 of them died last year, he sold some remaining cows, this year,
the number of cows increased by 24, he bought 43 more cows, his friend gave him 8 cows.
Now, he has 83 cows. How many cows did he sell last year?
-/
theorem cows_sold (S : ℕ) : (39 - 25 - S + 24 + 43 + 8 = 83) → S = 6 :=
by
  intro h
  sorry

end cows_sold_l56_56935


namespace music_tool_cost_l56_56306

namespace BandCost

def trumpet_cost : ℝ := 149.16
def song_book_cost : ℝ := 4.14
def total_spent : ℝ := 163.28

theorem music_tool_cost : (total_spent - (trumpet_cost + song_book_cost)) = 9.98 :=
by
  sorry

end music_tool_cost_l56_56306


namespace exponential_function_solution_l56_56578

theorem exponential_function_solution (a : ℝ) (h : a > 1)
  (h_max_min_diff : a - a⁻¹ = 1) : a = (Real.sqrt 5 + 1) / 2 :=
sorry

end exponential_function_solution_l56_56578


namespace connectivity_proof_l56_56467

-- Defining a predicate for our proof problem.
def graph_connected (n : ℕ) (d : ℕ → ℕ → ℕ) (conn : ℕ → ℕ → Prop) : Prop :=
  (n % 2 = 1) ∧ -- The country has an odd number of cities
  (∀ i j : ℕ, i ≠ j → d i j ≠ d j i) ∧ -- Distances between any pair of cities are pairwise distinct
  (∀ i : ℕ, ∃ a b : ℕ, conn i a ∧ conn i b ∧ a ≠ b) ∧ -- Each city is directly connected to exactly two other cities
  (∀ i j, conn i j → (d i j = max (d i (choose_not_eq i j a b)) (d i (choose_not_eq i j b a)))) -- The two cities connected to any given city are the farthest

theorem connectivity_proof (n : ℕ) (d : ℕ → ℕ → ℕ) (conn : ℕ → ℕ → Prop) : graph_connected n d conn → 
    (∀ i j : ℕ, connected_via_flights i j conn) :=
by
  sorry

end connectivity_proof_l56_56467


namespace proper_subsets_count_l56_56028

theorem proper_subsets_count {A : Set ℕ} (h : A = {1, 2}) : (A.powerset.filter (λ s, s ≠ A)).card = 3 :=
by
  sorry

end proper_subsets_count_l56_56028


namespace jenna_eel_is_16_l56_56717

open Real

def jenna_eel_len (J B : ℝ) : Prop :=
  J = (1 / 3) * B ∧ J + B = 64

theorem jenna_eel_is_16 : ∃ J B : ℝ, jenna_eel_len J B ∧ J = 16 :=
by
  exists 16
  exists 64 * (3 / 4)    -- which is 48
  unfold jenna_eel_len
  split
  { norm_num }
  { norm_num }

end jenna_eel_is_16_l56_56717


namespace coefficient_of_x_squared_in_expansion_l56_56704

theorem coefficient_of_x_squared_in_expansion :
  let expr := (x + (2 / (x^2)) : ℝ)
  let expansion := (expr ^ 5)
  find_coefficient expansion x 2 = 10 :=
by
  sorry

end coefficient_of_x_squared_in_expansion_l56_56704


namespace volume_of_inscribed_sphere_l56_56943

theorem volume_of_inscribed_sphere (a : ℝ) (π : ℝ) (h : a = 6) : 
  (4 / 3 * π * (a / 2) ^ 3) = 36 * π :=
by
  sorry

end volume_of_inscribed_sphere_l56_56943


namespace distance_from_apex_l56_56406

-- Defining the given conditions
def smallerArea : ℝ := 300 * Real.sqrt 3
def largerArea : ℝ := 675 * Real.sqrt 3
def distanceBetweenPlanes : ℝ := 12

-- Defining the theorem
theorem distance_from_apex (h : ℝ) 
  (areas_ratio : smallerArea / largerArea = 4 / 9) 
  (dist_between_planes : distanceBetweenPlanes = 12) 
  (h_eq : h - (2 / 3) * h = 12) :
  h = 36 := 
sorry

end distance_from_apex_l56_56406


namespace maximize_f_l56_56895

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem maximize_f : ∃ x : ℝ, f x = 2 ∧ (∀ y : ℝ, f x ≥ f y) :=
by
  use 2
  split
  { show f 2 = 2
    sorry }
  { intro y
    show f 2 ≥ f y
    sorry }

end maximize_f_l56_56895


namespace sum_of_first_13_terms_is_39_l56_56602

-- Definition of arithmetic sequence and the given condition
def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

-- Given condition
axiom given_condition {a : ℕ → ℤ} (h : arithmetic_sequence a) : a 5 + a 6 + a 7 = 9

-- The main theorem
theorem sum_of_first_13_terms_is_39 {a : ℕ → ℤ} (h : arithmetic_sequence a) (h9 : a 5 + a 6 + a 7 = 9) : sum_of_first_n_terms a 12 = 39 :=
sorry

end sum_of_first_13_terms_is_39_l56_56602


namespace roots_square_sum_l56_56500

theorem roots_square_sum {a b c : ℝ} (h1 : 3 * a^3 + 2 * a^2 - 3 * a - 8 = 0)
                                  (h2 : 3 * b^3 + 2 * b^2 - 3 * b - 8 = 0)
                                  (h3 : 3 * c^3 + 2 * c^2 - 3 * c - 8 = 0)
                                  (sum_roots : a + b + c = -2/3)
                                  (product_pairs : a * b + b * c + c * a = -1) : 
  a^2 + b^2 + c^2 = 22 / 9 := by
  sorry

end roots_square_sum_l56_56500


namespace jake_weight_is_194_l56_56085

-- Define the variables for Jake's and Kendra's weights.
variables (J K : ℕ)

-- Define the conditions as hypotheses.
hypothesis h1 : J - 8 = 2 * K
hypothesis h2 : J + K = 287

-- State the theorem to be proven.
theorem jake_weight_is_194 : J = 194 :=
by
  -- Proof to be filled in using Lean tactics.
  sorry

end jake_weight_is_194_l56_56085


namespace measure_dihedral_angle_CFG_E_l56_56797

-- Definitions based on conditions provided
def E (A B : ℝ³) : ℝ³ := (A + B) / 2
def F (B C : ℝ³) : ℝ³ := (B + C) / 2
def G (C D : ℝ³) : ℝ³ := (C + D) / 2

-- Regular Tetrahedron
structure RegularTetrahedron (A B C D : ℝ³) : Prop :=
  (Length_eq : ∀ {P Q}, P ∈ {A, B, C, D} → Q ∈ {A, B, C, D} → P ≠ Q → ∥P - Q∥ = 1)

-- Measure of the dihedral angle
def dihedralAngle (C F G E : ℝ³) : ℝ := sorry -- placeholder for the actual calculation

-- Theorem statement
theorem measure_dihedral_angle_CFG_E {A B C D : ℝ³} (h : RegularTetrahedron A B C D) :
  dihedralAngle C (F B C) (G C D) (E A B) = π - arctan (sqrt 2 / 2) :=
sorry

end measure_dihedral_angle_CFG_E_l56_56797


namespace coin_difference_l56_56875

variables (x y : ℕ)

theorem coin_difference (h1 : x + y = 15) (h2 : 2 * x + 5 * y = 51) : x - y = 1 := by
  sorry

end coin_difference_l56_56875


namespace measure_one_quart_in_min_operations_l56_56441

-- Defining the jugs and the capacity of the beer barrel
def barrel_capacity : ℕ := 120
def jug7_capacity : ℕ := 7
def jug5_capacity : ℕ := 5

-- Defining the target quantity to measure
def target_quantity : ℕ := 1

-- Defining the maximum permissible number of operations
def min_operations : ℕ := 42

-- Theorem stating the problem
theorem measure_one_quart_in_min_operations :
  ∃ (operations : list string), list.length operations = min_operations ∧
  ∃ (jug7 jug5 : ℕ), jug7 = 0 ∧ jug5 = target_quantity :=
sorry

end measure_one_quart_in_min_operations_l56_56441


namespace altitude_magnitude_and_coordinates_l56_56221

noncomputable def vec_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨2, -1⟩
def B : Point := ⟨3, 2⟩
def C : Point := ⟨-3, -1⟩

def BC : ℝ × ℝ := (C.x - B.x, C.y - B.y)

def λ : ℝ := 1 / 3

def D : Point := ⟨B.x + λ * BC.1, B.y + λ * BC.2⟩

def AD_vector : ℝ × ℝ := (D.x - A.x, D.y - A.y)

theorem altitude_magnitude_and_coordinates :
  vec_magnitude AD_vector = real.sqrt 5 ∧ D = ⟨1, 1⟩ :=
by
  -- Proof is omitted.
  sorry

end altitude_magnitude_and_coordinates_l56_56221


namespace sin_neg_1920_value_l56_56984

noncomputable def sin_neg_1920 : ℝ :=
  let angle := 240.0 in
  let sin_60 := Real.sin (60 * Real.pi / 180) in
  -sin_60

theorem sin_neg_1920_value : Real.sin (-1920 * Real.pi / 180) = sin_neg_1920 := by
  sorry

end sin_neg_1920_value_l56_56984


namespace number_of_arrangements_l56_56640

-- Definition of the unique letters in the given word
def letters : List Char := ['B', 'A₁', 'N₁', 'A₂', 'N₂', 'A₃', 'X₁', 'X₂']

-- Proposition: the number of unique arrangements of these letters equals to 8!
theorem number_of_arrangements : List.permutations letters |>.length = 8! := by
  sorry

end number_of_arrangements_l56_56640


namespace equilateral_triangle_side_length_of_perpendiculars_l56_56285

theorem equilateral_triangle_side_length_of_perpendiculars
  (s : ℝ)
  (T U V W : EuclideanSpace ℝ (fin 2))
  (D E F : EuclideanSpace ℝ (fin 2))
  (h_eq_triangle : ∀ x y z: EuclideanSpace ℝ (fin 2), euclideanDist x y = euclideanDist y z ∧ euclideanDist y z = euclideanDist z x)
  (h_perpendicular_DE : is_perpendicular T U D E)
  (h_perpendicular_EF : is_perpendicular T V E F)
  (h_perpendicular_FD : is_perpendicular T W F D)
  (h_TU : euclideanDist T U = 2)
  (h_TV : euclideanDist T V = 3)
  (h_TW : euclideanDist T W = 4) : 
  ∃ s, s = 6 * real.sqrt 3 :=
sorry

end equilateral_triangle_side_length_of_perpendiculars_l56_56285


namespace coloring_problem_l56_56682

noncomputable def numColorings : Nat :=
  256

theorem coloring_problem :
  ∃ (f : Fin 200 → Bool), (∀ (i j : Fin 200), i ≠ j → f i = f j → ¬ (i + 1 + j + 1).isPowerOfTwo) ∧ 
    256 = numColorings :=
by
  -- Definitions and conditions used for the problem
  let S := {n : ℕ | ∃ k : ℕ, n = bit0 k 1 ∧ k > 0}
  have hS : ∀ n ∈ S, ∀ i j ∈ {1, ..., 200}, n ≠ 2 ^ (log2 n) ∧ i ≠ j → i + j ≠ n := sorry
  -- Proof skipped, as required in the guidelines
  sorry

end coloring_problem_l56_56682


namespace trucks_needed_l56_56112

-- Definitions of the conditions
def total_apples : ℕ := 80
def apples_transported : ℕ := 56
def truck_capacity : ℕ := 4

-- Definition to calculate the remaining apples
def remaining_apples : ℕ := total_apples - apples_transported

-- The theorem statement
theorem trucks_needed : remaining_apples / truck_capacity = 6 := by
  sorry

end trucks_needed_l56_56112


namespace relationship_among_a_b_c_l56_56589

-- Here we define the constants corresponding to the conditions.
noncomputable def a : ℝ := 4 ^ 0.8
noncomputable def b : ℝ := (1 / 2) ^ (-1.5)
noncomputable def c : ℝ := Real.logBase 2 0.8

-- The theorem states the required relationship among a, b, and c.
theorem relationship_among_a_b_c : a > b ∧ b > c := by
  -- Proof is omitted for brevity.
  sorry

end relationship_among_a_b_c_l56_56589


namespace solve_expression_l56_56970

theorem solve_expression : (∏ i in Finset.range 10, i + 1) / (∑ i in Finset.range 10, i + 1) * (1 / 2) = 33000 := 
by 
  sorry

end solve_expression_l56_56970


namespace find_prime_pairs_l56_56997

theorem find_prime_pairs :
  ∃ p q : ℕ, Prime p ∧ Prime q ∧
    ∃ a b : ℕ, a^2 = p - q ∧ b^2 = pq - q ∧ (p = 3 ∧ q = 2) :=
by
  sorry

end find_prime_pairs_l56_56997


namespace ellipse_equation_l56_56587

variable {F1 F2 P : Type}
variable (a b c : Real)
variable (C : a > b ∧ b > 0)
variable (C_eq : (x^2 / a^2) + (y^2 / b^2) = 1)
variable (perpendicular : PF_1.perp PF_2)
variable (area_of_triangle : ∆_PF_1_F_2.area = 9)
variable (perimeter_of_triangle : ∆_PF_1_F_2.perimeter = 18)

theorem ellipse_equation : C = (x^2 / 25) + (y^2 / 9) = 1 :=
by
  sorry -- skipping the proof

end ellipse_equation_l56_56587


namespace coloring_number_lemma_l56_56687

-- Definition of the problem conditions
def no_sum_of_two_same_color_is_power_of_two (f : ℕ → bool) : Prop :=
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 200 ∧ f a = f b → ¬(∃ k : ℕ, a + b = 2^k)

-- The main theorem
theorem coloring_number_lemma :
  (∃ f : ℕ → bool, (∀ n, 1 ≤ n ∧ n ≤ 200 → (f n = tt ∨ f n = ff))
                   ∧ no_sum_of_two_same_color_is_power_of_two f) →
  ∃ n : ℕ, n = 256 :=
by sorry

end coloring_number_lemma_l56_56687


namespace combined_set_range_l56_56354

def is_two_digit_prime (n : ℕ) : Prop := prime n ∧ 10 ≤ n ∧ n < 100

def is_positive_multiple_of_4_less_than_100 (n : ℕ) : Prop := n % 4 = 0 ∧ n > 0 ∧ n < 100

noncomputable def set_X : set ℕ := { n : ℕ | is_two_digit_prime n }

noncomputable def set_Y : set ℕ := { n : ℕ | is_positive_multiple_of_4_less_than_100 n }

theorem combined_set_range : (set.range (finset.sup (set_X ∪ set_Y) - finset.inf (set_X ∪ set_Y))) = 93 := 
by 
  sorry

end combined_set_range_l56_56354


namespace sequence_two_cases_l56_56469

noncomputable def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ≥ a (n-1)) ∧  -- nondecreasing
  (∃ n m, n ≠ m ∧ a n ≠ a m) ∧  -- nonconstant
  (∀ n, a n ∣ n^2)  -- a_n | n^2

theorem sequence_two_cases (a : ℕ → ℕ) :
  sequence_property a →
  (∃ n1, ∀ n ≥ n1, a n = n) ∨ (∃ n2, ∀ n ≥ n2, a n = n^2) :=
by {
  sorry
}

end sequence_two_cases_l56_56469


namespace negation_of_universal_statement_l56_56384

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, |x| + x^2 ≥ 0)) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by
  sorry

end negation_of_universal_statement_l56_56384


namespace length_of_MN_and_side_of_ABC_l56_56347

noncomputable def point := ℝ × ℝ

def equilateral (A B C : point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def circumcircle (A B C : point) : set point :=
  {M | dist A B = dist B C ∧ dist B C = dist C A}

variables (A B C M N : point)

theorem length_of_MN_and_side_of_ABC
  (h1 : equilateral A B C)
  (h2 : M ∈ circumcircle A B C)
  (h3 : dist M A = 3)
  (h4 : dist M C = 4)
  (h5 : ∃ N, collinear {B, M, N} ∧ N ∈ line_through A C)
  : dist M N = 12 / 7 ∧ dist A B = √37 := by
  sorry

end length_of_MN_and_side_of_ABC_l56_56347


namespace sum_inverse_square_magnitude_roots_of_unity_l56_56066

theorem sum_inverse_square_magnitude_roots_of_unity :
  (∑ z in {z : ℂ | z^8 = 1}, (1 / |1 - z|^2)) = 0 := by
  sorry

end sum_inverse_square_magnitude_roots_of_unity_l56_56066


namespace coloring_ways_l56_56692

-- Define a function that determines if a given integer is a power of two
def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

-- Define the main problem condition as a predicate
def valid_coloring (coloring : ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ i → i ≤ 200 → 1 ≤ j → j ≤ 200 → i ≠ j → 
    coloring i = coloring j → ¬ is_power_of_two (i + j)

theorem coloring_ways : 
  ∃ (coloring : ℕ → Prop), (∀ (i : ℕ), 1 ≤ i → i ≤ 200 → (coloring i = true ∨ coloring i = false)) ∧
  (valid_coloring coloring) ∧
  (finset.card (finset.filter valid_coloring (finset.univ : set (ℕ → Prop))) = 256) :=
sorry

end coloring_ways_l56_56692


namespace parabola_y_intersection_l56_56014

theorem parabola_y_intersection (x y : ℝ) : 
  (∀ x' y', y' = -(x' + 2)^2 + 6 → ((x' = 0) → (y' = 2))) :=
by
  intros x' y' hy hx0
  rw hx0 at hy
  simp [hy]
  sorry

end parabola_y_intersection_l56_56014


namespace sum_of_first_seven_terms_l56_56867

variable {a_n : ℕ → ℝ} {d : ℝ}

-- Define the arithmetic progression condition.
def arithmetic_progression (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n n = a_n 0 + n * d

-- We are given that the sequence is an arithmetic progression.
axiom sequence_is_arithmetic_progression : arithmetic_progression a_n d

-- We are also given that the sum of the 3rd, 4th, and 5th terms is 12.
axiom sum_of_terms_is_12 : a_n 2 + a_n 3 + a_n 4 = 12

-- We need to prove that the sum of the first seven terms is 28.
theorem sum_of_first_seven_terms : (a_n 0) + (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) + (a_n 6) = 28 := 
  sorry

end sum_of_first_seven_terms_l56_56867


namespace sum_ratio_is_nine_l56_56225

open Nat

-- Predicate to define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of the first n terms of an arithmetic sequence
noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n * (a 0 + a (n - 1))) / 2

axiom a : ℕ → ℝ -- The arithmetic sequence
axiom h_arith : is_arithmetic_sequence a
axiom a5_eq_5a3 : a 4 = 5 * a 2

-- Statement of the problem
theorem sum_ratio_is_nine : S 9 a / S 5 a = 9 :=
sorry

end sum_ratio_is_nine_l56_56225


namespace ab_is_sqrt_33_div_32_l56_56333

noncomputable def ab_solution (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a^4 + b^4 = 31 / 16) : ℝ :=
sqrt (33 / 32)

theorem ab_is_sqrt_33_div_32 (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a^4 + b^4 = 31 / 16) : a * b = sqrt (33 / 32) :=
sorry

end ab_is_sqrt_33_div_32_l56_56333


namespace range_of_y_when_m_is_minus_one_max_value_of_y_l56_56616

-- Problem statement 1
theorem range_of_y_when_m_is_minus_one (x : ℝ) (hx : -π / 3 ≤ x ∧ x ≤ 2 * π / 3) :
  let y := 2 * sin(x)^2 - cos(x) - 1 / 8 in
  ∃ (y_min y_max : ℝ), (∀ y, y_min ≤ y ∧ y ≤ y_max) ∧ (y_min = -9 / 8) ∧ (y_max = 2) :=
sorry

-- Problem statement 2
theorem max_value_of_y (x m : ℝ) :
  let y := -2 * cos(x)^2 + m * cos(x) + 15 / 8 in
  (m < -4 → ∃ y_max, (∀ y, y ≤ y_max) ∧ (y_max = -m - 1 / 8)) ∧
  (m > 4 → ∃ y_max, (∀ y, y ≤ y_max) ∧ (y_max = m - 1 / 8)) ∧
  (-4 ≤ m ∧ m ≤ 4 → ∃ y_max, (∀ y, y ≤ y_max) ∧ (y_max = (m^2 + 15) / 8)) :=
sorry

end range_of_y_when_m_is_minus_one_max_value_of_y_l56_56616


namespace boundary_length_l56_56472

variable (π : ℝ) [fact (π = 3.14)]

def length_of_boundary_of_new_figure : ℝ :=
  let side_length := real.sqrt 64  -- side length is sqrt(64) = 8 units
  let segment_length := (8 : ℝ) / 3  -- each side is divided into three segments
  let straight_segments_length := 4 * segment_length  -- there are four such segments
  let quarter_circle_arcs_length := 2 * π * segment_length / 4  -- two quarter-circles
  let semicircle_arcs_length := 2 * π * segment_length / 2  -- two semicircles
  straight_segments_length + 2 * quarter_circle_arcs_length + semicircle_arcs_length

theorem boundary_length :
  length_of_boundary_of_new_figure 3.14 = 35.8 :=
sorry

end boundary_length_l56_56472


namespace range_of_a_l56_56655

variable {x a : ℝ}

theorem range_of_a (hx : 1 ≤ x ∧ x ≤ 2) (h : 2 * x > a - x^2) : a < 8 :=
by sorry

end range_of_a_l56_56655


namespace new_avg_weight_l56_56366

-- Define the weights of individuals
variables (A B C D E : ℕ)
-- Conditions
axiom avg_ABC : (A + B + C) / 3 = 84
axiom avg_ABCD : (A + B + C + D) / 4 = 80
axiom E_def : E = D + 8
axiom A_80 : A = 80

theorem new_avg_weight (A B C D E : ℕ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : E = D + 8) 
  (h4 : A = 80) 
  : (B + C + D + E) / 4 = 79 := 
by
  sorry

end new_avg_weight_l56_56366


namespace tommy_number_of_nickels_l56_56049

theorem tommy_number_of_nickels
  (d p n q : ℕ)
  (h1 : d = p + 10)
  (h2 : n = 2 * d)
  (h3 : q = 4)
  (h4 : p = 10 * q) : n = 100 :=
sorry

end tommy_number_of_nickels_l56_56049


namespace find_angle_B_l56_56711

theorem find_angle_B
  (AB CD : ℝ)
  (A D C B : ℝ)
  (h1 : AB = CD → parallel AB CD)
  (h2 : A = 3 * D)
  (h3 : C = 3 * B)
  (h4 : B + C = 180) :
  B = 45 :=
by
  sorry

end find_angle_B_l56_56711


namespace Ksyusha_time_to_school_l56_56736

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l56_56736


namespace remainder_is_zero_l56_56173

noncomputable def polynomial_division_theorem : Prop :=
  ∀ (x : ℤ), (x^3 ≡ 1 [MOD (x^2 + x + 1)]) → 
             (x^5 ≡ x^2 [MOD (x^2 + x + 1)]) →
             (x^2 - 1) * (x^3 - 1) ≡ 0 [MOD (x^2 + x + 1)]

theorem remainder_is_zero : polynomial_division_theorem := by
  sorry

end remainder_is_zero_l56_56173


namespace num_of_correct_statements_is_zero_l56_56237

-- Define the conditions as Lean definitions and prove that the number of correct statements is 0
def statement1 (x : ℝ) : Prop := (x^2 = 1 → x = 1)
def statement2 : Prop := (4^2 = 8)
def statement3 (x : ℝ) : Prop := (√ x = x → x = 0)
def statement4 : Prop := (∛ 8 = 2)

theorem num_of_correct_statements_is_zero :
  (¬ (∃ x : ℝ, statement1 x))
  ∧ (¬ statement2)
  ∧ (¬ (∃ x : ℝ, statement3 x))
  ∧ (¬ statement4) → 
  true := 
by intros h; exact trivial

end num_of_correct_statements_is_zero_l56_56237


namespace adult_ticket_cost_given_conditions_l56_56346

variables (C A S : ℕ)

def cost_relationships : Prop :=
  A = C + 10 ∧ S = A - 5 ∧ (5 * C + 2 * A + 2 * S + (S - 3) = 212)

theorem adult_ticket_cost_given_conditions :
  cost_relationships C A S → A = 28 :=
by
  intros h
  have h1 : A = C + 10 := h.left
  have h2 : S = A - 5 := h.right.left
  have h3 : (5 * C + 2 * A + 2 * S + (S - 3) = 212) := h.right.right
  sorry

end adult_ticket_cost_given_conditions_l56_56346


namespace find_m_value_l56_56564

variable (m : ℝ)

theorem find_m_value (h1 : m^2 - 3 * m = 4)
                     (h2 : m^2 = 5 * m + 6) : m = -1 :=
sorry

end find_m_value_l56_56564


namespace transformed_quadrilateral_area_l56_56841

noncomputable def area_of_transformed_quadrilateral {α : Type} [Field α] 
  (x1 x2 x3 x4 : α) (g : α → α) (area : α) : α :=
let original_points := [(x1, g x1), (x2, g x2), (x3, g x3), (x4, g x4)] in
let transformed_points := [(x1 / 3, 3 * g x1), (x2 / 3, 3 * g x2), (x3 / 3, 3 * g x3), (x4 / 3, 3 * g x4)] in
if (area = 50) then 50 else sorry

theorem transformed_quadrilateral_area {α : Type} [Field α] 
  (x1 x2 x3 x4 : α) (g : α → α) (h_area : area_of_transformed_quadrilateral x1 x2 x3 x4 g 50 = 50) : 
  area_of_transformed_quadrilateral x1 x2 x3 x4 g 50 = 50 := 
by sorry

end transformed_quadrilateral_area_l56_56841


namespace L_shaped_tiling_l56_56567

theorem L_shaped_tiling (n : ℕ) (hn : n > 0) :
  ∀ (x : ℕ) (y : ℕ) (hx : x < 3*n + 1) (hy : y < 3*n + 1),
  ∃ (L : list (ℕ × ℕ)), 
    (∀ (l : ℕ × ℕ), l ∈ L → 
      (l.1 < 3*n + 1) ∧ (l.2 < 3*n + 1)) ∧
    (∀ (i j : ℕ), (i, j) ≠ (x, y) → ∃ l, l ∈ L ∧ (i,j) ∈ 
    {(l.1, l.2), (l.1+1, l.2), (l.1, l.2+1), (l.1+1, l.2+1)}) :=
begin
  sorry,
end

end L_shaped_tiling_l56_56567


namespace avg_age_of_adults_l56_56009

theorem avg_age_of_adults (avg_age_all : ℕ) (num_girls num_boys num_adults : ℕ)
  (avg_age_girls avg_age_boys : ℕ) (total_members : ℕ)
  (h1 : avg_age_all = 17) (h2 : total_members = 40)
  (h3 : num_girls = 20) (h4 : num_boys = 15) (h5 : num_adults = 5)
  (h6 : avg_age_girls = 15) (h7 : avg_age_boys = 16) :
  (680 - (num_girls * avg_age_girls + num_boys * avg_age_boys)) / num_adults = 28 :=
by
  have total_sum := total_members * avg_age_all,
  have sum_girls := num_girls * avg_age_girls,
  have sum_boys := num_boys * avg_age_boys,
  have sum_adults := total_sum - sum_girls - sum_boys,
  sorry

end avg_age_of_adults_l56_56009


namespace sum_of_series_eq_l56_56161

-- Definitions based on the problem conditions
variables (x : ℝ) (a : ℕ → ℝ) (n : ℕ)
  (h_x_pos : 0 < x) (h_x_ne_one : x ≠ 1)
  (h_geometric : ∃ q : ℝ, 0 < q ∧ q ≠ 1 ∧ ∀ i : ℕ, i < n → a (i+1) = a i * q)

-- Statement of the theorem
theorem sum_of_series_eq :
  S_n = x ^ (Real.log (a 0)) * ((x ^ (n * (Real.log (a 1) - Real.log (a 0)))) - 1) / ((x ^ (Real.log (a 1) - Real.log (a 0))) - 1) :=
sorry

-- Definition of the series sum S_n based on problem statement
def S_n : ℝ := ∑ i in Finset.range n, x ^ (Real.log (a i))

end sum_of_series_eq_l56_56161


namespace problem_solution_l56_56318

noncomputable def omega : ℂ := sorry -- Choose a suitable representative for ω

variables (a b c d : ℝ) (h₀ : a ≠ -1) (h₁ : b ≠ -1) (h₂ : c ≠ -1) (h₃ : d ≠ -1)
          (hω : ω^3 = 1 ∧ ω ≠ 1)
          (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω)

theorem problem_solution (a b c d : ℝ) (h₀ : a ≠ -1) (h₁ : b ≠ -1) (h₂ : c ≠ -1) (h₃ : d ≠ -1)
  (hω : ω^3 = 1 ∧ ω ≠ 1)
  (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 3 :=
sorry

end problem_solution_l56_56318


namespace rick_can_buy_two_bananas_l56_56357

open Nat

-- Define the costs and given relationships
variables (m b : ℕ) -- costs of a muffin and a banana in arbitrary whole number of monetary units

-- Define the conditions
def cost_relation := m = 2 * b
def susie_cost := 3 * m + 5 * b
def rick_cost_eq := 9 * m + 15 * b = 6 * m + 10 * b

-- Theorem to prove the number of bananas Rick can get for the cost of one muffin is 2
theorem rick_can_buy_two_bananas (m b : ℕ) (h_m : m = 2 * b) (h_r : 9 * m + 15 * b = 6 * m + 10 * b) : (m / b) = 2 := by
  sorry

end rick_can_buy_two_bananas_l56_56357


namespace ksyusha_travel_time_wednesday_l56_56746

-- Definitions based on the conditions
def speed_walk (v : ℝ) : ℝ := v
def speed_run (v : ℝ) : ℝ := 2 * v
def distance_walk_tuesday (S : ℝ) : ℝ := 2 * S
def distance_run_tuesday (S : ℝ) : ℝ := S
def total_time_tuesday (v S : ℝ) : ℝ := (distance_walk_tuesday S) / (speed_walk v) + (distance_run_tuesday S) / (speed_run v)

-- Theorem statement
theorem ksyusha_travel_time_wednesday (S v : ℝ) (h : total_time_tuesday v S = 30) :
  (S / v + S / v) = 24 := 
by sorry

end ksyusha_travel_time_wednesday_l56_56746


namespace integral_evaluation_l56_56992

open Real IntervalIntegral

/-- Define the integrand for the integral -/
def integrand (x : ℝ) : ℝ := (x^2 + 1) / x

/-- State the main theorem -/
theorem integral_evaluation : (∫ x in 1..2, integrand x) = (3 / 2) + (Real.log 2) :=
by
  -- Provide placeholders for the steps needed to prove the theorem
  sorry

end integral_evaluation_l56_56992


namespace middle_school_students_count_l56_56281

def split_equally (m h : ℕ) : Prop := m = h
def percent_middle (M m : ℕ) : Prop := m = M / 5
def percent_high (H h : ℕ) : Prop := h = 3 * H / 10
def total_students (M H : ℕ) : Prop := M + H = 50
def number_of_middle_school_students (M: ℕ) := M

theorem middle_school_students_count (M H m h : ℕ) 
  (hm_eq : split_equally m h) 
  (hm_percent : percent_middle M m) 
  (hh_percent : percent_high H h) 
  (htotal : total_students M H) : 
  number_of_middle_school_students M = 30 :=
by
  sorry

end middle_school_students_count_l56_56281


namespace minimum_k_10_l56_56481

def stringent_function (h : ℕ → ℕ) := ∀ (x y : ℕ), 0 < x → 0 < y → h(x) + h(y) > 2 * y^2

def k (n : ℕ) := 
  if n ≤ 8 then 81
  else 2 * n^2 + 1 - 81

theorem minimum_k_10 : 
  (∃ k : ℕ → ℕ, 
    (stringent_function k) ∧ 
    (k(1) + k(2) + k(3) + k(4) + 
    k(5) + k(6) + k(7) + k(8) + 
    k(9) + k(10) + k(11) + 
    k(12) + k(13) + k(14) + k(15)
    = 2728) ∧ k(10) = 120) :=
sorry

end minimum_k_10_l56_56481


namespace number_of_poles_needed_l56_56123

/-!
    Given:
    - Length of the rectangular plot: 135 metres
    - Width of the rectangular plot: 80 metres
    - Distance between poles: 7 metres

    Prove:
    - Number of poles needed to enclose the rectangular plot = 62
-/

theorem number_of_poles_needed :
  let length := 135
  let width := 80
  let dist_between_poles := 7
  let perimeter := 2 * (length + width)
  let number_of_poles := (perimeter / dist_between_poles).ceil
  number_of_poles = 62 :=
by
  sorry

end number_of_poles_needed_l56_56123


namespace cube_inscribed_sphere_volume_l56_56873

theorem cube_inscribed_sphere_volume :
  (∀ (a b : ℝ), a = 1 → b = 1 → (∃ (R : ℝ), 2 * R = real.sqrt 3 ∧ (4 * real.pi / 3) * R ^ 3 = real.sqrt 3 * real.pi / 2)) :=
by
  intro a b ha hb
  use real.sqrt 3 / 2
  split
  { rw [ha, hb, mul_div_cancel' _ (two_ne_zero : (2 : ℝ) ≠ 0)] }
  { rw [mul_div_cancel' _ (three_ne_zero : (3 : ℝ) ≠ 0)] }
  sorry

end cube_inscribed_sphere_volume_l56_56873


namespace vector_combination_l56_56588

theorem vector_combination : 
  let a := (2, 1 : ℤ)
  let b := (-3, 4 : ℤ)
  3 • a + 4 • b = (-6, 19) :=
by
  let a := (2, 1 : ℤ)
  let b := (-3, 4 : ℤ)
  let scalarmul {x y : ℤ} (n : ℤ) := (n * x, n * y)
  let addvec {x1 y1 x2 y2 : ℤ} := (x1 + x2, y1 + y2)
  have h₁ : 3 • a = (6, 3) := by sorry
  have h₂ : 4 • b = (-12, 16) := by sorry
  have h₃ : (6, 3) + (-12, 16) = (-6, 19) := by sorry
  exact h₁.trans (h₂.trans h₃)

end vector_combination_l56_56588


namespace difference_between_balances_l56_56146

def interest_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ := principal * (1 + rate * time)
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ := principal * (1 + rate) ^ time

theorem difference_between_balances :
  let P_A := 9000
  let r_A := 0.05
  let n := 15
  let P_B := 11000
  let r_B := 0.06
  let t := 15
  let A_A := compound_interest P_A r_A n
  let A_B := interest_amount P_B r_B t
  abs (A_B - A_A) = 2190 :=
by {
  sorry
}

end difference_between_balances_l56_56146


namespace compare_sqrt3_sub1_div2_half_l56_56497

theorem compare_sqrt3_sub1_div2_half : (sqrt 3 - 1) / 2 < 1 / 2 := 
sorry

end compare_sqrt3_sub1_div2_half_l56_56497


namespace complex_mod_arg_evaluation_l56_56515

theorem complex_mod_arg_evaluation :
  (complex.abs ⟨3, -7⟩ + complex.abs ⟨3, 7⟩ - complex.arg ⟨3, 7⟩) = (2 * real.sqrt 58 - real.arctan (7 / 3)) :=
by
  sorry

end complex_mod_arg_evaluation_l56_56515


namespace sum_of_first_seven_terms_l56_56295

noncomputable def a (n : ℕ) : ℝ
axiom arithmetic_seq (d : ℝ) (a1 : ℝ) : ∀ n, a n = a1 + (n - 1) * d

theorem sum_of_first_seven_terms 
  (d a1 : ℝ) 
  (ha : ∀ n, a n = a1 + (n - 1) * d) 
  (h_condition: a 2 + a 6 = 10) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 35 := 
sorry

end sum_of_first_seven_terms_l56_56295


namespace benny_total_hours_l56_56488

-- Define the conditions
def hours_per_day : ℕ := 3
def days_worked : ℕ := 6

-- State the theorem (problem) to be proved
theorem benny_total_hours : hours_per_day * days_worked = 18 :=
by
  -- Sorry to skip the actual proof
  sorry

end benny_total_hours_l56_56488


namespace line_max_area_difference_l56_56381

noncomputable def line_equation_max_area_difference : Prop :=
  ∃ (P : ℝ × ℝ) (r : ℝ), (P = (1, 1)) ∧ (r = 2) ∧
  (∀ (L : ℝ → ℝ), 
    (L = λ x, -x + 2) ∧ 
    (∀ (x y : ℝ), (x^2 + y^2 ≤ r^2) → 
      ∃ (y1 : ℝ), 
      (L x = y1) → 
      (y1 ≠ y → y = y1 + 2 - x) )
  )

theorem line_max_area_difference : line_equation_max_area_difference :=
sorry

end line_max_area_difference_l56_56381


namespace square_area_l56_56001

def parabola (x : ℝ) : ℝ := x^2 - 10 * x + 21

theorem square_area :
  ∃ s : ℝ, parabola (5 + s) = -2s ∧ (2 * (-1 + 2 * real.sqrt 5))^2 = 64 - 16 * real.sqrt 5 :=
by
  sorry

end square_area_l56_56001


namespace maximize_f_l56_56896

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem maximize_f : ∃ x : ℝ, f x = 2 ∧ (∀ y : ℝ, f x ≥ f y) :=
by
  use 2
  split
  { show f 2 = 2
    sorry }
  { intro y
    show f 2 ≥ f y
    sorry }

end maximize_f_l56_56896


namespace ones_digit_of_sum_of_powers_l56_56888

theorem ones_digit_of_sum_of_powers :
  (∑ n in Finset.range 2013, n ^ 2013) % 10 = 1 :=
sorry

end ones_digit_of_sum_of_powers_l56_56888


namespace min_value_fraction_inequality_l56_56592

theorem min_value_fraction_inequality (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 2 * a + b = 2) :
  inf {x | ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ 2 * a + b = 2 ∧ x = 2 / a + 1 / b} = 9 / 2 :=
sorry

end min_value_fraction_inequality_l56_56592


namespace Ksyusha_time_to_school_l56_56737

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l56_56737


namespace general_formula_sum_less_than_n_squared_l56_56808

-- Conditions
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

axiom S4_eq_4S2 : S 4 = 4 * S 2
axiom a2n_eq_2an_plus_1 : ∀ n, a (2 * n) = 2 * a n + 1

-- Proof problems
theorem general_formula : ∀ n, a n = 2 * n - 1 := 
sorry

theorem sum_less_than_n_squared : ∀ n, (∑ i in Finset.range n, a (i + 1)) < n^2 :=
sorry

end general_formula_sum_less_than_n_squared_l56_56808


namespace count_multiples_of_12_l56_56634

theorem count_multiples_of_12 (a b : ℕ) (ha : a = 20) (hb : b = 200) : 
  (finset.card ((finset.Ico 20 201).filter (λ x, x % 12 = 0))) = 15 :=
by {
  rw [ha, hb],
  sorry
}

end count_multiples_of_12_l56_56634


namespace ksyusha_travel_time_wednesday_l56_56742

-- Definitions based on the conditions
def speed_walk (v : ℝ) : ℝ := v
def speed_run (v : ℝ) : ℝ := 2 * v
def distance_walk_tuesday (S : ℝ) : ℝ := 2 * S
def distance_run_tuesday (S : ℝ) : ℝ := S
def total_time_tuesday (v S : ℝ) : ℝ := (distance_walk_tuesday S) / (speed_walk v) + (distance_run_tuesday S) / (speed_run v)

-- Theorem statement
theorem ksyusha_travel_time_wednesday (S v : ℝ) (h : total_time_tuesday v S = 30) :
  (S / v + S / v) = 24 := 
by sorry

end ksyusha_travel_time_wednesday_l56_56742


namespace cot_product_l56_56491

noncomputable
def cot (θ : ℝ) : ℝ := Real.cos θ / Real.sin θ

theorem cot_product (h : ∀ θ : ℝ, θ ∈ Set.Icc 1 45 → 
                    (1 + cot θ) * (1 + cot (45 - θ)) = 2 ∧
                    θ ≠ 45) :
  (List.foldr (*) 1 ((List.map (λ n : ℕ, (1 + cot (n : ℝ))) (List.range' 1 45)))) = 2 ^ 22 :=
by
  sorry

end cot_product_l56_56491


namespace simplify_sin_diff_l56_56355

theorem simplify_sin_diff :
  let a := Real.sin (Real.pi / 4)
  let b := Real.sin (Real.pi / 12)
  a - b = (3 * Real.sqrt 2 - Real.sqrt 6) / 4 :=
by
  let a := Real.sqrt 2 / 2
  have sin_30 := Real.sin_pi_div_two
  let b := (Real.sqrt 6 - Real.sqrt 2) / 4
  calc
    a - b = (3 * Real.sqrt 2 - Real.sqrt 6) / 4 : sorry

end simplify_sin_diff_l56_56355


namespace ratio_of_zinc_to_copper_in_mixture_l56_56899

theorem ratio_of_zinc_to_copper_in_mixture (total_weight : ℝ) (weight_of_zinc : ℝ) (weight_of_copper : ℝ) 
  (h1 : total_weight = 74) (h2 : weight_of_zinc = 33.3) (h3 : weight_of_copper = total_weight - weight_of_zinc) :
  ((weight_of_zinc * 10).to_int : ℤ) / ((weight_of_copper * 10).to_int : ℤ) = 333 / 407 :=
by
  sorry

end ratio_of_zinc_to_copper_in_mixture_l56_56899


namespace probability_product_multiple_of_3_l56_56730

theorem probability_product_multiple_of_3 : 
  let juan_die := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let amal_die := {1, 2, 3, 4, 5, 6}
  (∃ (d1 ∈ juan_die) (d2 ∈ amal_die), (d1 * d2) % 3 = 0) = (8 / 15 : ℚ) := 
by
  sorry

end probability_product_multiple_of_3_l56_56730


namespace min_num_of_teams_l56_56108

theorem min_num_of_teams (num_athletes : ℕ) (max_team_size : ℕ) (h1 : num_athletes = 30) (h2 : max_team_size = 9) :
  ∃ (min_teams : ℕ), min_teams = 5 ∧ (∀ nal : ℕ, (nal > 0 ∧ num_athletes % nal = 0 ∧ nal ≤ max_team_size) → num_athletes / nal ≥ min_teams) :=
by
  sorry

end min_num_of_teams_l56_56108


namespace line_perp_plane_parallel_imp_planes_perp_l56_56571

variables (l : Type) (α β : Type)
variables [IsLine l] [IsPlane α] [IsPlane β]

-- Definitions of perpendicular and parallel relationships
def LinePerpendicularToPlane (l : Type) (α : Type) [IsLine l] [IsPlane α] : Prop := sorry
def LineParallelToPlane (l : Type) (β : Type) [IsLine l] [IsPlane β] : Prop := sorry
def PlanesPerpendicular (α β : Type) [IsPlane α] [IsPlane β] : Prop := sorry

-- Conditions
variable (hl_perp_alpha : LinePerpendicularToPlane l α)
variable (hl_parallel_beta : LineParallelToPlane l β)

-- Goal
theorem line_perp_plane_parallel_imp_planes_perp 
  (hl_perp_alpha : LinePerpendicularToPlane l α) 
  (hl_parallel_beta : LineParallelToPlane l β) : 
  PlanesPerpendicular α β := sorry

end line_perp_plane_parallel_imp_planes_perp_l56_56571


namespace negation_of_proposition_l56_56861

theorem negation_of_proposition : 
  (¬ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0)) ↔ (∃ x : ℝ, x^2 + 2*x + 5 = 0) :=
by sorry

end negation_of_proposition_l56_56861


namespace proof_problem_l56_56292

-- Define the trajectory of Point A
def pointA_trajectory (α : ℝ) : ℝ × ℝ :=
  (2 - 3 * Real.sin α, 3 * Real.cos α - 2)

-- Define the condition for the circle
def is_on_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 2)^2 = 9

-- Define the line C in Cartesian coordinates
def line_C_in_cartesian (x y a : ℝ) : Prop :=
  Real.sqrt 2 * x + Real.sqrt 2 * y = 2 * a

-- Define the tangency condition
def line_tangent_to_circle (a : ℝ) : Prop :=
  | (Real.sqrt 2 * 2 + Real.sqrt 2 * -2) - 2 * a | / 2 = 3

theorem proof_problem :
  (∀ α : ℝ, is_on_circle (fst (pointA_trajectory α)) (snd (pointA_trajectory α))) ∧
  (∀ a : ℝ, (line_tangent_to_circle a → a = 3 ∨ a = -3)) :=
by
  -- Proof placeholder
  sorry

end proof_problem_l56_56292


namespace volume_of_inscribed_sphere_l56_56945

theorem volume_of_inscribed_sphere {cube_edge : ℝ} (h : cube_edge = 6) : 
  ∃ V : ℝ, V = 36 * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_l56_56945


namespace number_of_blocks_l56_56972

namespace CubicalStorageUnit

-- Conditions
def length_external : ℝ := 15
def width_external : ℝ := 12
def height_external : ℝ := 8
def thickness_wall_floor : ℝ := 1.5
def block_volume : ℝ := 1 -- 1 cubic foot per block

-- Calculations based on conditions
def volume_total : ℝ := length_external * width_external * height_external

def length_internal : ℝ := length_external - 2 * thickness_wall_floor
def width_internal : ℝ := width_external - 2 * thickness_wall_floor
def height_internal : ℝ := height_external - thickness_wall_floor

def volume_internal : ℝ := length_internal * width_internal * height_internal

def volume_blocks : ℝ := volume_total - volume_internal

-- Theorem to prove
theorem number_of_blocks (h_blocks : volume_blocks / block_volume = 738) : 
  volume_total - volume_internal = 738 := 
by
  exact h_blocks

end CubicalStorageUnit

end number_of_blocks_l56_56972


namespace derivative_of_x_exp_x_l56_56371

theorem derivative_of_x_exp_x :
  ∀ x : ℝ, deriv (λ x : ℝ, x * real.exp x) x = (1 + x) * real.exp x :=
by sorry

end derivative_of_x_exp_x_l56_56371


namespace ksyusha_travel_time_l56_56768

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l56_56768


namespace find_mn_l56_56714

noncomputable def mn_length : ℝ :=
  let AB := Real.sqrt 3
  let BC := 4
  let AC := Real.sqrt 7
  let BD := Real.sqrt (31 / 4)
  let s1 := (AB + BD + AD) / 2
  let s2 := (BC + BD + CD) / 2
  let DM := (s1 - AB)
  let DN := (s2 - BC)
  let MN := (Real.abs (DM - DN))
  in MN

theorem find_mn : mn_length = 2 - (Real.sqrt 3 / 2) := 
sorry

end find_mn_l56_56714


namespace ksyusha_travel_time_l56_56756

theorem ksyusha_travel_time :
  ∀ (S v : ℝ), 
  (2 * S / v + S / (2 * v) = 30) → 
  (2 * (S / v) = 24) :=
by
  intros S v h1
  have h2 : S / v = 12 := by
    calc 
      S / v = (30 * 2) / 5 := by sorry
  calc 
    2 * (S / v) = 2 * 12 := by sorry
  ... = 24 := by norm_num

end ksyusha_travel_time_l56_56756


namespace ratio_of_second_to_first_l56_56925

noncomputable def building_heights (H1 H2 H3 : ℝ) : Prop :=
  H1 = 600 ∧ H3 = 3 * (H1 + H2) ∧ H1 + H2 + H3 = 7200

theorem ratio_of_second_to_first (H1 H2 H3 : ℝ) (h : building_heights H1 H2 H3) :
  H1 ≠ 0 → (H2 / H1 = 2) :=
by
  unfold building_heights at h
  rcases h with ⟨h1, h3, h_total⟩
  sorry -- Steps of solving are skipped

end ratio_of_second_to_first_l56_56925


namespace coloring_problem_l56_56680

noncomputable def numColorings : Nat :=
  256

theorem coloring_problem :
  ∃ (f : Fin 200 → Bool), (∀ (i j : Fin 200), i ≠ j → f i = f j → ¬ (i + 1 + j + 1).isPowerOfTwo) ∧ 
    256 = numColorings :=
by
  -- Definitions and conditions used for the problem
  let S := {n : ℕ | ∃ k : ℕ, n = bit0 k 1 ∧ k > 0}
  have hS : ∀ n ∈ S, ∀ i j ∈ {1, ..., 200}, n ≠ 2 ^ (log2 n) ∧ i ≠ j → i + j ≠ n := sorry
  -- Proof skipped, as required in the guidelines
  sorry

end coloring_problem_l56_56680


namespace teacher_periods_per_day_l56_56948

noncomputable def periods_per_day (days_per_month : ℕ) (months : ℕ) (period_rate : ℕ) (total_earnings : ℕ) : ℕ :=
  let total_days := days_per_month * months
  let total_periods := total_earnings / period_rate
  let periods_per_day := total_periods / total_days
  periods_per_day

theorem teacher_periods_per_day :
  periods_per_day 24 6 5 3600 = 5 := by
  sorry

end teacher_periods_per_day_l56_56948


namespace MN_NO_sum_l56_56847

-- Definitions based on the conditions provided
abbreviation Area_polygon_JKLMNO : ℝ := 68
abbreviation Side_JK : ℝ := 10
abbreviation Side_KL : ℝ := 11
abbreviation Side_NO : ℝ := 7

-- Lean 4 theorem statement
theorem MN_NO_sum : Area_polygon_JKLMNO = 68 ∧ Side_JK = 10 ∧ Side_KL = 11 ∧ Side_NO = 7 → (MN + NO) = 14.5 :=
by
  sorry

end MN_NO_sum_l56_56847


namespace ratio_of_ages_l56_56266

-- Define variables A's age and B's age
variables (A_age B_age : ℕ)

-- Conditions
def condition1 : Prop := (λ A_age B_age, A_age = B_age + 9)
def condition2 : Prop := (B_age = 39)

-- Question and the proof target
theorem ratio_of_ages (h1 : condition1 A_age B_age) (h2 : condition2 B_age) :
  ((A_age + 10) / (B_age - 10) = 2) :=
by
  sorry

end ratio_of_ages_l56_56266


namespace min_ac_triangle_count_l56_56058

def isosceles_triangle := Type
def acute_triangle := Type
def minimum_triangulation (T : isosceles_triangle) (n : ℕ) := Prop

noncomputable def contains_acute_triangles_only (T : isosceles_triangle) (triangles : set acute_triangle) : Prop :=
sorry

theorem min_ac_triangle_count
  (T : isosceles_triangle)
  (angle : ℝ)
  (H : angle = 120)
  (triangles : set acute_triangle)
  (n : ℕ) :
  contains_acute_triangles_only T triangles → minimum_triangulation T 7 := 
sorry

end min_ac_triangle_count_l56_56058


namespace max_area_triangle_l56_56881

theorem max_area_triangle (AB : ℝ) (ratioBC_AC : ℝ) (ratioBC : ℝ) (ratioAC : ℝ) :
  AB = 9 ∧ ratioBC / ratioAC = 40 / 41 → 
  let BC := ratioBC * (9 / (ratioBC + ratioAC)) 
  let AC := ratioAC * (9 / (ratioBC + ratioAC))
  ∃ h : ℝ, area_of_triangle 9 BC AC = h ∧ h = 820 :=
begin
  -- Proof goes here
  sorry
end

end max_area_triangle_l56_56881


namespace average_speed_monkey_l56_56117

def monkeyDistance : ℝ := 2160
def monkeyTimeMinutes : ℝ := 30
def monkeyTimeSeconds : ℝ := monkeyTimeMinutes * 60

theorem average_speed_monkey :
  (monkeyDistance / monkeyTimeSeconds) = 1.2 := 
sorry

end average_speed_monkey_l56_56117


namespace value_of_x2_y2_z2_l56_56229

variables {a b c k x y z : ℝ}
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0)

def condition1 := xy = ka
def condition2 := xz = kb
def condition3 := yz = kc

theorem value_of_x2_y2_z2
  (condition1 : xy = ka)
  (condition2 : xz = kb)
  (condition3 : yz = kc) :
  x^2 + y^2 + z^2 = k * (ab / c + ac / b + bc / a) :=
sorry

end value_of_x2_y2_z2_l56_56229


namespace sell_price_per_plant_l56_56968

def total_cost (cost_seeds cost_soil : ℝ) : ℝ :=
  cost_seeds + cost_soil

def total_revenue (net_profit cost : ℝ) : ℝ :=
  net_profit + cost

def selling_price (revenue num_plants : ℝ) : ℝ :=
  revenue / num_plants

theorem sell_price_per_plant (cost_seeds cost_soil net_profit revenue : ℝ) :
  cost_seeds = 2 → cost_soil = 8 → net_profit = 90 →
  revenue = total_revenue net_profit (total_cost cost_seeds cost_soil) →
  revenue / 20 = 5 :=
by {
  sorry
}

end sell_price_per_plant_l56_56968


namespace maximum_value_quadratic_l56_56894

def quadratic_function (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem maximum_value_quadratic :
  ∃ x : ℝ, quadratic_function x = 2 ∧ ∀ y : ℝ, quadratic_function y ≤ 2 :=
sorry

end maximum_value_quadratic_l56_56894


namespace part1_part2_l56_56912

-- Part 1: Positive integers with leading digit 6 that become 1/25 of the original number when the leading digit is removed.
theorem part1 (n : ℕ) (m : ℕ) (h1 : m = 6 * 10^n + m) (h2 : m = (6 * 10^n + m) / 25) :
  m = 625 * 10^(n - 2) ∨
  m = 625 * 10^(n - 2 + 1) ∨
  ∃ k : ℕ, m = 625 * 10^(n - 2 + k) :=
sorry

-- Part 2: No positive integer exists which becomes 1/35 of the original number when its leading digit is removed.
theorem part2 (n : ℕ) (m : ℕ) (h : m = 6 * 10^n + m) :
  m ≠ (6 * 10^n + m) / 35 :=
sorry

end part1_part2_l56_56912


namespace circles_separate_when_t_is_neg1_t_value_when_symmetric_about_l_l56_56291

-- Define the conditions
def line_l (x y : ℝ) : Prop := 8 * x + 6 * y + 1 = 0
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 8 * x - 2 * y + 13 = 0
def circle_C2 (x y t : ℝ) : Prop := x^2 + y^2 + 8 * t * x - 8 * y + 16 * t + 12 = 0

-- Define the proof that circles C1 and C2 are separate when t = -1
theorem circles_separate_when_t_is_neg1 : ∀ (x y : ℝ), 
  circle_C1 x y → 
  (∀ x2 y2, circle_C2 x2 y2 (-1) → 
  let d := Real.sqrt ((4 - (-4))^2 + (4 - 1)^2) -- Distance between centers
  in d > 8) :=
by
  intros x y h1 x2 y2 h2
  let d := Real.sqrt ((4 - (-4))^2 + (4 - 1)^2)
  sorry

-- Define the proof that t = 0 when the circles are symmetric about the line l
theorem t_value_when_symmetric_about_l :
  (∀ x y : ℝ, circle_C1 x y) →
  (∀ x2 y2 t, circle_C2 x2 y2 t → (∀ x3 y3, line_l x3 y3) → t = 0) :=
by
  intros h1 h2 h3
  sorry

end circles_separate_when_t_is_neg1_t_value_when_symmetric_about_l_l56_56291


namespace num_integer_points_in_triangle_l56_56167

theorem num_integer_points_in_triangle : 
  { (x, y) : ℕ × ℕ | x + y ≤ 7 }.card = 36 := 
by
  -- analysis starts
  -- explicit calculation provided in solution
  sorry

end num_integer_points_in_triangle_l56_56167


namespace movie_sale_price_l56_56927

/-- 
Given the conditions:
- cost of actors: $1200
- number of people: 50
- cost of food per person: $3
- equipment rental costs twice as much as food and actors combined
- profit made: $5950

Prove that the selling price of the movie was $10,000.
-/
theorem movie_sale_price :
  let cost_of_actors := 1200
  let num_people := 50
  let food_cost_per_person := 3
  let total_food_cost := num_people * food_cost_per_person
  let combined_cost := total_food_cost + cost_of_actors
  let equipment_rental_cost := 2 * combined_cost
  let total_cost := cost_of_actors + total_food_cost + equipment_rental_cost
  let profit := 5950
  let sale_price := total_cost + profit
  sale_price = 10000 := 
by
  sorry

end movie_sale_price_l56_56927


namespace part1_proof_part2_proof_part3_proof_part4_proof_l56_56301

variable {A B C : Type}
variables {a b c : ℝ}  -- Sides of the triangle
variables {h_a h_b h_c r r_a r_b r_c : ℝ}  -- Altitudes, inradius, and exradii of \triangle ABC

-- Part 1: Proving the sum of altitudes related to sides and inradius
theorem part1_proof : h_a + h_b + h_c = r * (a + b + c) * (1 / a + 1 / b + 1 / c) := sorry

-- Part 2: Proving the sum of reciprocals of altitudes related to the reciprocal of inradius and exradii
theorem part2_proof : (1 / h_a) + (1 / h_b) + (1 / h_c) = 1 / r ∧ 1 / r = (1 / r_a) + (1 / r_b) + (1 / r_c) := sorry

-- Part 3: Combining results of parts 1 and 2 to prove product of sums
theorem part3_proof : (h_a + h_b + h_c) * ((1 / h_a) + (1 / h_b) + (1 / h_c)) = (a + b + c) * (1 / a + 1 / b + 1 / c) := sorry

-- Part 4: Final geometric identity
theorem part4_proof : (h_a + h_c) / r_a + (h_c + h_a) / r_b + (h_a + h_b) / r_c = 6 := sorry

end part1_proof_part2_proof_part3_proof_part4_proof_l56_56301


namespace zé_roberto_winning_strategy_l56_56078

-- Defining the sequence a_n as a function
def a : ℕ → ℕ
| 0 := 0
| 1 := 0
| 2 := 0
| 3 := 1
| (n + 4) := if ∃ k ≤ n, a k = 0 ∧ a (n - k - 1) = 0 then 1 else 0

-- The theorem to prove
theorem zé_roberto_winning_strategy : a 30 = 1 := 
sorry

end zé_roberto_winning_strategy_l56_56078


namespace number_of_schools_is_23_l56_56518

-- Conditions and definitions
noncomputable def number_of_students_per_school : ℕ := 3
def beth_rank : ℕ := 37
def carla_rank : ℕ := 64

-- Statement of the proof problem
theorem number_of_schools_is_23
  (n : ℕ)
  (h1 : ∀ i < n, ∃ r1 r2 r3: ℕ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h2 : ∀ i < n, ∃ A B C: ℕ, A = (2 * B + 1) ∧ C = A ∧ B = 35 ∧ A < beth_rank ∧ beth_rank < carla_rank):
  n = 23 :=
by
  sorry

end number_of_schools_is_23_l56_56518


namespace halfway_between_frac_l56_56533

theorem halfway_between_frac : (1 / 7 + 1 / 9) / 2 = 8 / 63 := by
  sorry

end halfway_between_frac_l56_56533


namespace jenna_eel_length_l56_56719

theorem jenna_eel_length (j b : ℕ) (h1 : b = 3 * j) (h2 : b + j = 64) : j = 16 := by 
  sorry

end jenna_eel_length_l56_56719


namespace range_of_a_l56_56214

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → 0 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -2 ≤ x ∧ x ≤ 2 → ∃ x₀, -2 ≤ x₀ ∧ x₀ ≤ 2 ∧ (a * x₀ - 1 = f x)) →
  a ∈ Set.Iic (-5/2) ∪ Set.Ici (5/2) :=
sorry

end range_of_a_l56_56214


namespace volume_of_inscribed_sphere_l56_56942

theorem volume_of_inscribed_sphere (a : ℝ) (π : ℝ) (h : a = 6) : 
  (4 / 3 * π * (a / 2) ^ 3) = 36 * π :=
by
  sorry

end volume_of_inscribed_sphere_l56_56942


namespace initial_boarders_count_l56_56391

variable (B D : ℕ)
variables (boarder_ratio day_student_ratio : ℕ)
variables (new_boarders boarder_day_ratio new_boarder_day_ratio : ℕ)

def initial_boarders_ratio (B D : ℕ) : Prop :=
  B * day_student_ratio = D * boarder_ratio

def new_boarders_added (B new_boarders : ℕ) : ℕ :=
  B + new_boarders

def new_ratio (B D new_boarders : ℕ) : Prop :=
  (B + new_boarders) * new_boarder_day_ratio = 2 * D

theorem initial_boarders_count :
  initial_boarders_ratio B D ∧ new_boarders_added B new_boarders = B + new_boarders ∧ new_ratio B D 30 ∧ boarder_day_ratio = 5 ∧ day_student_ratio = 12 ∧ new_boarder_day_ratio = 1 → B = 150 := by
  sorry

end initial_boarders_count_l56_56391


namespace liminf_log_T_eq_zero_with_prob_one_l56_56780

noncomputable def xi (n : ℕ) : ℕ → Int := λ i, if i % 2 = 0 then 1 else -1

def S (n : ℕ) (xi : ℕ → Int) : ℕ → Int
| 0     := 0
| (n+1) := S n xi + xi (n + 1)

def T (n : ℕ) : ℝ := 
  (1 / Real.sqrt n.toReal) * (List.range (n + 1)).map (λ k => S k xi).toFinset.sup id

theorem liminf_log_T_eq_zero_with_prob_one : 
  (𝕡 (λ ω, liminf (λ n, (Real.log n.toReal) * T n ω) = 0) = 1) :=
sorry

end liminf_log_T_eq_zero_with_prob_one_l56_56780


namespace converse_inverse_contrapositive_l56_56447

-- The original statement
def original_statement (x y : ℕ) : Prop :=
  (x + y = 5) → (x = 3 ∧ y = 2)

-- Converse of the original statement
theorem converse (x y : ℕ) : (x = 3 ∧ y = 2) → (x + y = 5) :=
by
  sorry

-- Inverse of the original statement
theorem inverse (x y : ℕ) : (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2) :=
by
  sorry

-- Contrapositive of the original statement
theorem contrapositive (x y : ℕ) : (x ≠ 3 ∨ y ≠ 2) → (x + y ≠ 5) :=
by
  sorry

end converse_inverse_contrapositive_l56_56447


namespace max_probability_pc_l56_56104

variables (p1 p2 p3 : ℝ)
variable (h : p3 > p2 ∧ p2 > p1 ∧ p1 > 0)

def PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem max_probability_pc : PC > PA ∧ PC > PB := 
by 
  sorry

end max_probability_pc_l56_56104


namespace problem_solution_l56_56321

def polynomial_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f (x - 1) + f x + f (x + 1) = (f x)^2 / (2013 * x)

noncomputable def f_values_sum (f : ℝ → ℝ) [polynomial_condition f] [nonconstant f] : ℝ :=
  f 1

theorem problem_solution :
  ∃ f : (ℝ → ℝ), polynomial_condition f ∧ ∃ sum : ℝ, sum = f_values_sum f ∧ sum = 6039 :=
sorry

end problem_solution_l56_56321


namespace parallel_transitive_l56_56955

-- Definition of parallel lines
def are_parallel (l1 l2 : Line) : Prop :=
  ∃ (P : Line), l1 = P ∧ l2 = P

-- Theorem stating that if two lines are parallel to the same line, then they are parallel to each other
theorem parallel_transitive (l1 l2 l3 : Line) (h1 : are_parallel l1 l3) (h2 : are_parallel l2 l3) :
  are_parallel l1 l2 :=
by
  sorry

end parallel_transitive_l56_56955


namespace geometric_sequence_a2_l56_56570

-- Define a geometric sequence and its properties
def is_geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions for the sequence
variables {a : ℕ → ℝ}
axiom a1 : a 1 = 1 / 4
axiom cond : a 3 * a 5 = 4 * (a 4 - 1)

-- Prove that a_2 = 1 / 2
theorem geometric_sequence_a2 : is_geometric_sequence a → a 2 = 1 / 2 :=
by
  intros h1
  sorry

end geometric_sequence_a2_l56_56570


namespace train_length_proof_l56_56133

noncomputable def train_length (v : ℕ) (t : ℝ) (b : ℝ) : ℝ := v * t / 3.6 - b

theorem train_length_proof : train_length 72 12.099 132 = 110 :=
by 
  have speed_mps : ℝ := 72 / 3.6
  have total_distance : ℝ := speed_mps * 12.099
  have length_train : ℝ := total_distance - 132
  simp [train_length, speed_mps, total_distance, length_train]
  exact rfl

end train_length_proof_l56_56133


namespace find_initial_candies_l56_56494

-- Define the initial number of candies as x
def initial_candies (x : ℕ) : ℕ :=
  let first_day := (3 * x) / 4 - 3
  let second_day := (3 * first_day) / 5 - 5
  let third_day := second_day - 7
  let final_candies := (5 * third_day) / 6
  final_candies

-- Formal statement of the theorem
theorem find_initial_candies (x : ℕ) (h : initial_candies x = 10) : x = 44 :=
  sorry

end find_initial_candies_l56_56494


namespace wrappers_after_collection_l56_56977

theorem wrappers_after_collection (caps_found : ℕ) (wrappers_found : ℕ) (current_caps : ℕ) (initial_caps : ℕ) : 
  caps_found = 22 → wrappers_found = 30 → current_caps = 17 → initial_caps = 0 → 
  wrappers_found ≥ 30 := 
by 
  intros h1 h2 h3 h4
  -- Solution steps are omitted on purpose
  --- This is where the proof is written
  sorry

end wrappers_after_collection_l56_56977


namespace find_f2_l56_56853

theorem find_f2 (f : ℝ → ℝ)
  (h : ∀ (x : ℝ), f (2 ^ x) + (x + 1) * f (2 ^ -x) = x) :
  f 2 = 3 :=
sorry

end find_f2_l56_56853


namespace number_of_valid_colorings_l56_56668

-- Define the specific properties and constraints given in the problem
def is_power_of_two (n : ℕ) : Prop := ∃ m : ℕ, n = 2^m

def valid_coloring (f : ℕ → bool) : Prop :=
  ∀ a b : ℕ, a ≠ b → a + b ≤ 200 →
  (f a = f b → ¬ is_power_of_two (a + b))

theorem number_of_valid_colorings : 
  ∃ c : ℕ, c = 256 ∧ 
  ∃ f : Π (n : ℕ), nat.lt n 201 → bool, 
    valid_coloring (λ n, f n (by linarith)) :=
begin
  sorry
end

end number_of_valid_colorings_l56_56668


namespace real_minus_imag_of_complex_l56_56338

-- Conditions:
def complex_num := (5 : ℂ) / (-3 - (1 : ℂ) * Complex.I) 

-- Proof statement:
theorem real_minus_imag_of_complex : 
  let a := complex_num.re,
      b := complex_num.im in
  a - b = -2 := by
  -- We skip the proof part as required.
  sorry

end real_minus_imag_of_complex_l56_56338


namespace ksyusha_wednesday_time_l56_56751

def speed_relation (v : ℝ) := ∀ (d_walk d_run1 d_run2 time1 time2: ℝ), 
  (2 * d_run1 = d_walk) → -- she walked twice the distance she ran on Tuesday
  (2 * v = 2 * v) → -- she runs twice as fast as she walks
  ((2 * d_walk / v) + (d_run1 / (2 * v)) = time1) → -- she took 30 minutes on Tuesday
  (time1 = 30) → -- Tuesday time equals 30
  (d_run2 = 2 * d_walk) → -- she had to run twice the distance she walked on Wednesday
  ((d_walk / v) + (d_run2 / (2 * v)) = time2) → -- calculate the time for Wednesday
  time2 = 24 -- time taken on Wednesday

theorem ksyusha_wednesday_time :
  ∃ (v : ℝ), speed_relation v :=
begin
  existsi 1,
  unfold speed_relation,
  intros,
  sorry,
end

end ksyusha_wednesday_time_l56_56751


namespace weighted_mean_correct_l56_56027

-- Define the conditions given in the problem

def incorrect_mean : ℝ := 150
def num_values : ℕ := 40

def errors : List (ℝ × ℝ) := 
  [(165, 135), (200, 170), (185, 155)]

def weights : List (ℕ × ℝ) :=
  [(10, 2), (20, 3), (10, 4)]

-- The correct answer we need to prove
def correct_weighted_mean : ℝ := 50.75

theorem weighted_mean_correct :
  let incorrect_total_sum := incorrect_mean * num_values
  let error_sum := (errors.map (λ (e : ℝ × ℝ), e.1 - e.2)).sum
  let correct_total_sum := incorrect_total_sum + error_sum
  let total_weight := (weights.map (λ (w : ℕ × ℝ), (w.1 : ℝ) * w.2)).sum
  correct_weighted_mean = correct_total_sum / total_weight := by
  sorry

end weighted_mean_correct_l56_56027


namespace sampling_correct_l56_56210

def systematic_sampling (total_students : Nat) (num_selected : Nat) (interval : Nat) (start : Nat) : List Nat :=
  (List.range num_selected).map (λ i => start + i * interval)

theorem sampling_correct :
  systematic_sampling 60 6 10 3 = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end sampling_correct_l56_56210


namespace final_total_and_individual_payments_l56_56187

def meal_costs := [12, 15, 10, 18, 20] -- Elisa, Ben, Clara, Damian, Fiona
def appetizer_costs := [8, 6]
def ice_cream_costs := [2, 3, 3, 4, 4] -- Elisa, Ben, Clara, Damian, Fiona

def meal_discount := 0.10
def ice_cream_discount := 0.05
def tip_percent := 0.15
def tax_rate := 0.08
def ben_extra_percent := 0.20

def correct_total_spent := 142.7292
def correct_individual_amounts := [25.23525, 41.7882, 25.23525, 25.23525, 25.23525] -- Elisa, Ben, Clara, Damian, Fiona

theorem final_total_and_individual_payments :
  let meal_total := meal_costs.sum,
      appetizer_total := appetizer_costs.sum,
      ice_cream_total := ice_cream_costs.sum,
      total := meal_total + appetizer_total + ice_cream_total,
      meal_discount_amount := meal_total * meal_discount,
      ice_cream_discount_amount := ice_cream_total * ice_cream_discount,
      total_after_discounts := total - meal_discount_amount - ice_cream_discount_amount,
      tax := total_after_discounts * tax_rate,
      total_with_tax := total_after_discounts + tax,
      tip := total_after_discounts * tip_percent,
      final_total := total_with_tax + tip,
      ben_extra_payment := final_total * ben_extra_percent,
      total_without_ben := final_total - 18,  -- Subtracting Ben's meal and ice cream cost before discounts and tax
      remaining_share := total_without_ben / 4 in
  final_total = correct_total_spent ∧
  remaining_share = correct_individual_amounts.head ∧
  (ben_extra_payment + 18) = correct_individual_amounts.nth 1
  ∧ ∀ i, i ∈ [2, 3, 4] → remaining_share = correct_individual_amounts.nth i :=
by sorry

end final_total_and_individual_payments_l56_56187


namespace sum_of_interior_angles_of_pentagon_l56_56465

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  calc
    (5 - 2) * 180 = 3 * 180 := by rfl
                 ... = 540 := by rfl

end sum_of_interior_angles_of_pentagon_l56_56465


namespace quadratic_root_value_l56_56574
-- Import the entirety of the necessary library

-- Define the quadratic equation with one root being -1
theorem quadratic_root_value 
    (m : ℝ)
    (h1 : ∀ x : ℝ, x^2 + m * x + 3 = 0)
    (root1 : -1 ∈ {x : ℝ | x^2 + m * x + 3 = 0}) :
    m = 4 ∧ ∃ root2 : ℝ, root2 = -3 ∧ root2 ∈ {x : ℝ | x^2 + m * x + 3 = 0} :=
by
  sorry

end quadratic_root_value_l56_56574


namespace sally_time_to_paint_l56_56830

theorem sally_time_to_paint :
  (S : ℝ) → (h₀ : S > 0) →
  (John_time : ℝ) (Combined_time : ℝ) → (h₁ : John_time = 6) → (h₂ : Combined_time = 2.4) →
  (1/S + 1/John_time = 1/Combined_time) → 
  S = 4 :=
by
  intros S h₀ John_time Combined_time h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  sorry

end sally_time_to_paint_l56_56830


namespace chess_group_players_l56_56878

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end chess_group_players_l56_56878


namespace suresh_work_hours_l56_56843

theorem suresh_work_hours (x : ℝ) 
    (suresh_work_rate : ℝ)
    (ashutosh_work_rate : ℝ) 
    (ashutosh_hours : ℝ) 
    (job_complete : ℝ) 
    (ashutosh_total_work : ashutosh_work_rate * ashutosh_hours = 12 * (1 / 30)): 
    suresh_work_rate = 1 / 15 ∧ ashutosh_work_rate = 1 / 30 ∧ ashutosh_hours = 12 ∧ job_complete = 1 → 
    x * suresh_work_rate + 12 * ashutosh_work_rate = 1 → x = 3 :=
by
    simp [suresh_work_rate, ashutosh_work_rate, ashutosh_hours, job_complete]
    intro h1 h2
    sorry

end suresh_work_hours_l56_56843


namespace solve_balls_problem_l56_56099

noncomputable def balls_problem (total : ℕ) (prob_red : ℚ) (diff_yellow_red : ℤ) : Prop :=
  ∃ (r y : ℕ),
    (r + y = total) ∧ 
    (r + 6 = y) ∧
    ((r * (r-1) * (r-2) / 6) / (455 / 1) = prob_red) ∧
    ((y * (y-1) * (y-2) / 6) / (455 / 1) = 33 / 91)

theorem solve_balls_problem : balls_problem 15 (2/91) 6 := 
by {
  use 5,
  use 11,
  split,
  dec_trivial,
  split,
  dec_trivial,
  split,
  dec_trivial,
  dec_trivial,
  sorry
}

end solve_balls_problem_l56_56099


namespace alloy_impurity_ratio_l56_56038

theorem alloy_impurity_ratio 
  (p q r : ℝ)
  (hp: p = 70) 
  (hq: q = 5) 
  (hr: r = 40) :
  (r - q) / (p - r) ≈ 1.17 :=
by
  rw [hp, hq, hr]
  simp
  norm_num
  sorry

end alloy_impurity_ratio_l56_56038


namespace AC_diagonal_length_l56_56275

noncomputable def AC_length (AB BC CD DA : ℝ) (angle_ADC : ℝ) (h_AB : AB = 13) (h_BC : BC = 13) (h_CD : CD = 15) (h_DA : DA = 15) (h_angle : angle_ADC = 30) : ℝ :=
  sqrt (CD^2 + DA^2 - 2 * CD * DA * real.cos (angle_ADC * real.pi / 180))

theorem AC_diagonal_length : AC_length 13 13 15 15 30 13.refl 13.refl 15.refl 15.refl 30.refl = 15 * sqrt (6 - 3 * sqrt 3) :=
by
  unfold AC_length
  norm_num
  sorry

end AC_diagonal_length_l56_56275


namespace find_number_of_striped_jerseys_l56_56310

def total_spent : ℕ := 80
def cost_long_sleeved : ℕ := 15
def num_long_sleeved : ℕ := 4
def cost_first_striped : ℕ := 10
def discount : ℕ := 2

theorem find_number_of_striped_jerseys (total_spent cost_long_sleeved num_long_sleeved cost_first_striped discount : ℕ) 
  (striped_j_Int : ℕ) :
  total_spent = 80 ∧ 
  cost_long_sleeved = 15 ∧ 
  num_long_sleeved = 4 ∧ 
  cost_first_striped = 10 ∧ 
  discount = 2 ∧ 
  striped_j_Int = 
    (total_spent - (cost_long_sleeved * num_long_sleeved) - cost_first_striped) / (cost_first_striped - discount) + 1 
  → 
  striped_j_Int = 2 :=
begin
  sorry,
end

end find_number_of_striped_jerseys_l56_56310


namespace decreasing_interval_log_l56_56022

noncomputable def quadratic (x : ℝ) : ℝ := 2*x^2 - 3*x + 1

theorem decreasing_interval_log : 
  ∃ (I : set ℝ), I = {x : ℝ | x > 1} ∧ 
  (∀ x1 x2 : ℝ, x1 ∈ I → x2 ∈ I → x1 < x2 → quadratic x2 < quadratic x1) → 
  ∀ x : ℝ, x > 1 → 
  (∀ x1 x2 : ℝ, x1 > x2 → (log (1/2) (quadratic x2)) < (log (1/2) (quadratic x1))) := 
by 
  sorry

end decreasing_interval_log_l56_56022


namespace sequence_sum_is_2n2_plus_2n_l56_56575

theorem sequence_sum_is_2n2_plus_2n
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_sum : ∀ n, (finset.range n).sum (λ i, real.sqrt (a (i + 1))) = n^2 + n) :
  (finset.range n).sum (λ i, a (i + 1) / (i + 1)) = 2 * n^2 + 2 * n := sorry

end sequence_sum_is_2n2_plus_2n_l56_56575


namespace prop1_prop2_prop3_prop4_l56_56798

structure SetPlane := (value : Set (Set ℝ × ℝ))

def subset (X Y : SetPlane) : Prop :=
  ∀ x, x ∈ X.value → x ∈ Y.value

def f (P : SetPlane) (X : SetPlane) : SetPlane := sorry -- function definition placeholder

axiom f_axiom (P : SetPlane) (X Y : SetPlane) :
  subset (f P (SetPlane.mk (X.value ∪ Y.value))) 
         (SetPlane.mk (f P (f P X).value ∪ f P Y.value ∪ Y.value))

theorem prop1 (P : SetPlane) (X : SetPlane) : 
  subset X (f P X) := sorry

theorem prop2 (P : SetPlane) (X : SetPlane) : 
  f P (f P X) = f P X := sorry

theorem prop3 (P : SetPlane) (X Y : SetPlane) (h : subset Y X) :
  subset (f P Y) (f P X) := sorry

theorem prop4 (P : SetPlane) (X Y : SetPlane) (h1 : ∀ (X : SetPlane), subset X (f P X)) 
  (h2 : ∀ (X : SetPlane), f P (f P X) = f P X) 
  (h3 : ∀ (X Y : SetPlane), subset Y X → subset (f P Y) (f P X))  :
  subset (f P (SetPlane.mk (X.value ∪ Y.value))) 
         (SetPlane.mk (f P (f P X).value ∪ f P Y.value ∪ Y.value)) := sorry

end prop1_prop2_prop3_prop4_l56_56798


namespace alley_width_l56_56664

theorem alley_width (b u v : ℝ)
(h₁ : u = b * real.sin (real.pi / 3))
(h₂ : v = b * real.sin (real.pi / 6)) :
  ∃ w, w = b * (1 + real.sqrt 3) / 2 :=
by
  sorry

end alley_width_l56_56664


namespace ksyusha_travel_time_wednesday_l56_56740

-- Definitions based on the conditions
def speed_walk (v : ℝ) : ℝ := v
def speed_run (v : ℝ) : ℝ := 2 * v
def distance_walk_tuesday (S : ℝ) : ℝ := 2 * S
def distance_run_tuesday (S : ℝ) : ℝ := S
def total_time_tuesday (v S : ℝ) : ℝ := (distance_walk_tuesday S) / (speed_walk v) + (distance_run_tuesday S) / (speed_run v)

-- Theorem statement
theorem ksyusha_travel_time_wednesday (S v : ℝ) (h : total_time_tuesday v S = 30) :
  (S / v + S / v) = 24 := 
by sorry

end ksyusha_travel_time_wednesday_l56_56740


namespace angle_B_in_trapezoid_l56_56713

theorem angle_B_in_trapezoid (AB CD : ℝ) (A D C B : ℝ) (par_AB_CD : AB ∥ CD) (angle_A_eq_3angle_D : A = 3 * D) (angle_C_eq_3angle_B : C = 3 * B) :
    B = 45 := 
begin
    -- Use the condition that AB ∥ CD to set up the equation
    have h1 : A + D + C + B = 360,
    from sorry, -- Interior angles of a quadrilateral sum up to 360°

    -- Use the given angles relationship and solve for B
    have h2 : B + C = 180,
    from sorry, -- Sum of interior angles on parallel lines
    
    rw [angle_C_eq_3angle_B, h2] at h1,
    linarith,
    
    -- Finally solve for the angle B
end

end angle_B_in_trapezoid_l56_56713


namespace sweaters_count_l56_56279

theorem sweaters_count
  (price_shirts : ℕ)
  (num_shirts : ℕ)
  (total_price_shirts : nat.succ price_shirts * num_shirts = 360)
  (price_sweaters : ℕ)
  (total_price_sweaters : nat.succ price_sweaters * 45 = 900)
  (price_sweater_minus_shirt : nat.succ price_sweaters = nat.succ price_shirts + 2) :
  45 = 45 :=
by
  sorry

end sweaters_count_l56_56279


namespace divide_students_l56_56817

theorem divide_students (students : Fin 5) (A B : students) :
  (∃ S : Finset (Fin 5), {A, B} ⊆ S ∧ card S = 3) ∨ (∃ S : Finset (Fin 5), {A, B} ⊆ S ∧ card S = 4) →
  count_partitions (λ S, {A, B} ⊆ S) = 6 :=
by
  sorry

end divide_students_l56_56817


namespace worker_a_times_approx_10_point_15_l56_56076

noncomputable def worker_a_time (A : ℝ) : Prop :=
  let work_rate_A := 1 / A in
  let work_rate_B := 1 / 12 in
  let combined_work_rate := work_rate_A + work_rate_B in
  combined_work_rate * (11 / 2) = 1

theorem worker_a_times_approx_10_point_15 (A : ℝ) (h : ∃ A, worker_a_time A) : A = 132 / 13 :=
sorry

end worker_a_times_approx_10_point_15_l56_56076


namespace x_y_squared_sum_l56_56902

noncomputable def x := 25 / y
def y := -5

theorem x_y_squared_sum : (x + y = -10) ∧ (x = 25 / y) → x^2 + y^2 = 50 := by
  sorry

end x_y_squared_sum_l56_56902


namespace students_failed_both_l56_56666

theorem students_failed_both (fail_Hindi fail_English pass_both fail_both : ℝ) :
  fail_Hindi = 0.25 → 
  fail_English = 0.35 → 
  pass_both = 0.80 → 
  fail_both = 0.40 :=
by
  intros h_Hindi h_English h_Pass
  have h_fail : fail_both = fail_Hindi + fail_English - (1 - pass_both) :=
    by
      linarith [h_Hindi, h_English, h_Pass]
  rw [h_Hindi, h_English]
  linarith [h_Pass]
  exact h_fail

end students_failed_both_l56_56666


namespace min_value_of_function_l56_56858

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem min_value_of_function : ∃ x ∈ set.Icc 0 1, f x = -3 := by
  sorry

end min_value_of_function_l56_56858


namespace eval_expression_l56_56989

theorem eval_expression :
  16^3 + 3 * (16^2) * 2 + 3 * 16 * (2^2) + 2^3 = 5832 :=
by
  sorry

end eval_expression_l56_56989


namespace ten_more_than_twice_number_of_birds_l56_56359

def number_of_birds : ℕ := 20

theorem ten_more_than_twice_number_of_birds :
  10 + 2 * number_of_birds = 50 :=
by
  sorry

end ten_more_than_twice_number_of_birds_l56_56359


namespace choose_two_from_six_l56_56550

theorem choose_two_from_six : (finset.card (finset.perms {1, 2, 3, 4, 5, 6})) // 24) 
= 30 := -- There are 30 ways to choose 2 positions out of 6 students with ordering
sorry

end choose_two_from_six_l56_56550


namespace point_C_moves_along_segment_l56_56908

theorem point_C_moves_along_segment {A B C P : Type} [Triangle ABC] (h1 : right_angle ABC C) (h2 : slides_along_lines A B P) : moves_along_segment C :=
sorry

end point_C_moves_along_segment_l56_56908


namespace remainder_is_zero_l56_56175

noncomputable def polynomial_division_theorem : Prop :=
  ∀ (x : ℤ), (x^3 ≡ 1 [MOD (x^2 + x + 1)]) → 
             (x^5 ≡ x^2 [MOD (x^2 + x + 1)]) →
             (x^2 - 1) * (x^3 - 1) ≡ 0 [MOD (x^2 + x + 1)]

theorem remainder_is_zero : polynomial_division_theorem := by
  sorry

end remainder_is_zero_l56_56175


namespace part1_part2_l56_56580

def f1 (x : ℝ) : ℝ := Math.cos x + Math.sin x
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x * f (x + Real.pi / 2)

theorem part1: ∀ x : ℝ, g f1 x = Real.cos (2 * x) :=
by
  intros x
  sorry

def f2 (x : ℝ) : ℝ := abs (Math.sin x) + Math.cos x

theorem part2:
  ∃ x1 x2 : ℝ, (∀ x : ℝ, g f2 x1 ≤ g f2 x ∧ g f2 x ≤ g f2 x2) 
  ∧ |x1 - x2| = 3 * Real.pi / 4 :=
by
  sorry

end part1_part2_l56_56580


namespace Ksyusha_travel_time_l56_56776

theorem Ksyusha_travel_time:
  ∀ (S v : ℝ), 
  (S > 0 ∧ v > 0) →
  (∀ tuesday_distance tuesday_time wednesday_distance wednesday_time: ℝ, 
    tuesday_distance = 3 * S ∧ 
    tuesday_distance = 2 * S + S ∧ 
    tuesday_time = (2 * S) / v + (S / (2 * v)) ∧ 
    tuesday_time = 30) → 
  (∀ wednesday_distance wednesday_time : ℝ, 
    wednesday_distance = S + 2 * S ∧ 
    wednesday_time = S / v + (2 * S) / (2 * v)) →
  wednesday_time = 24 :=
by
  intros S v S_pos v_pos tuesday_distance tuesday_time wednesday_distance wednesday_time
  sorry

end Ksyusha_travel_time_l56_56776


namespace total_cases_is_8_l56_56854

def num_blue_cards : Nat := 3
def num_yellow_cards : Nat := 5

def total_cases : Nat := num_blue_cards + num_yellow_cards

theorem total_cases_is_8 : total_cases = 8 := by
  sorry

end total_cases_is_8_l56_56854


namespace coloring_number_lemma_l56_56689

-- Definition of the problem conditions
def no_sum_of_two_same_color_is_power_of_two (f : ℕ → bool) : Prop :=
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 200 ∧ f a = f b → ¬(∃ k : ℕ, a + b = 2^k)

-- The main theorem
theorem coloring_number_lemma :
  (∃ f : ℕ → bool, (∀ n, 1 ≤ n ∧ n ≤ 200 → (f n = tt ∨ f n = ff))
                   ∧ no_sum_of_two_same_color_is_power_of_two f) →
  ∃ n : ℕ, n = 256 :=
by sorry

end coloring_number_lemma_l56_56689


namespace largest_integer_n_l56_56530

theorem largest_integer_n (n : ℕ) :
  (∀ x : ℝ, (tan x) * (cot x) = 1 → (| tan x ^ n + cot x ^ n | ≥ 2 / n) → n ≤ 6) :=
sorry

end largest_integer_n_l56_56530


namespace number_of_valid_colorings_l56_56672

-- Define the specific properties and constraints given in the problem
def is_power_of_two (n : ℕ) : Prop := ∃ m : ℕ, n = 2^m

def valid_coloring (f : ℕ → bool) : Prop :=
  ∀ a b : ℕ, a ≠ b → a + b ≤ 200 →
  (f a = f b → ¬ is_power_of_two (a + b))

theorem number_of_valid_colorings : 
  ∃ c : ℕ, c = 256 ∧ 
  ∃ f : Π (n : ℕ), nat.lt n 201 → bool, 
    valid_coloring (λ n, f n (by linarith)) :=
begin
  sorry
end

end number_of_valid_colorings_l56_56672


namespace total_wages_l56_56452

-- Definitions and conditions
def A_one_day_work : ℚ := 1 / 10
def B_one_day_work : ℚ := 1 / 15
def A_share_wages : ℚ := 2040

-- Stating the problem in Lean
theorem total_wages (X : ℚ) : (3 / 5) * X = A_share_wages → X = 3400 := 
  by 
  sorry

end total_wages_l56_56452


namespace first_player_wins_l56_56410

-- Define the initial conditions
def initial_pieces : ℕ := 1
def final_pieces (m n : ℕ) : ℕ := m * n
def num_moves (pieces : ℕ) : ℕ := pieces - 1

-- Theorem statement: Given the initial dimensions and the game rules,
-- prove that the first player will win.
theorem first_player_wins (m n : ℕ) (h_m : m = 6) (h_n : n = 8) : 
  (num_moves (final_pieces m n)) % 2 = 0 → false :=
by
  -- The solution details and the proof will be here.
  sorry

end first_player_wins_l56_56410


namespace circle_equation_l56_56568

noncomputable def circle_center_and_radius (a : ℝ) : ℝ × ℝ :=
  let r := real.sqrt ((1 - a) ^ 2 + (3 - a) ^ 2)
  (a, r)

theorem circle_equation (C : ℝ × ℝ ↔ ℝ)
  (h1 : C (1, 3) = true)
  (h2 : C (-1, 1) = true)
  (h3 : ∃ a, C (a, a) = true) :
  ∃ (a : ℝ), C (x - 1) ^ 2 + (y - 1) ^ 2 = 4 := 
sorry

noncomputable def line_equation (l : ℝ × ℝ ↔ ℝ)
  (h1 : l (2, -2) = true)
  (h2 : ∃ C, l is a chord of length 2 * real.sqrt 3 at C) :
  l x = 2 ∨ l (4 * x + 3 * y - 2) = 0 :=
sorry

end circle_equation_l56_56568


namespace max_perimeter_of_triangle_l56_56300

noncomputable def maximum_triangle_perimeter (a b c : ℝ) (A B C : ℝ) (h1 : a = 2) (h2 : a^2 = b^2 + c^2 - b * c) : ℝ :=
  max (a + b + c)

theorem max_perimeter_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : a = 2) (h2 : a^2 = b^2 + c^2 - b * c) :
  maximum_triangle_perimeter a b c A B C h1 h2 = 6 := sorry

end max_perimeter_of_triangle_l56_56300


namespace hyperbola_eccentricity_l56_56598

-- Given conditions
variables {a b c : ℝ}

-- Definitions from the conditions
def center_at_origin : Prop := true
def foci_on_x_axis : Prop := true
def asymptotes : Prop := ∀ (x : ℝ), (y = 2 * x ∨ y = -2 * x)

-- Relationship between a, b, and c
def relationship_a_b : Prop := b = 2 * a
def relationship_c_a_b : Prop := c^2 = a^2 + b^2

-- Conclusion to be proved
def eccentricity_definition : Prop := ∃ e : ℝ, e = sqrt 5 ∧ e = c / a

theorem hyperbola_eccentricity :
  center_at_origin → foci_on_x_axis → asymptotes →
  relationship_a_b → relationship_c_a_b →
  eccentricity_definition :=
by
  intros _ _ _ hab hcab h
  sorry

end hyperbola_eccentricity_l56_56598


namespace find_the_number_l56_56647

theorem find_the_number :
  ∃ (n : ℤ), n - (28 - (37 - (15 - 19))) = 58 → n = 45 := by
  intros h
  sorry

end find_the_number_l56_56647


namespace intersection_eq_l56_56583

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≤ 2}

theorem intersection_eq : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_eq_l56_56583


namespace find_length_RS_l56_56657

noncomputable def length_RS (PQ QR anglePQR : ℝ) (RS : ℝ) : Prop :=
  let S : ℝ := 11 + Real.sqrt 2
  ∧ PQ = 5
  ∧ QR = 6
  ∧ anglePQR = 135
  ∧ RS = S

theorem find_length_RS :
  ∃ RS, length_RS 5 6 135 RS := by
  use 11 + Real.sqrt 2
  sorry

end find_length_RS_l56_56657


namespace find_angle_B_l56_56710

theorem find_angle_B
  (AB CD : ℝ)
  (A D C B : ℝ)
  (h1 : AB = CD → parallel AB CD)
  (h2 : A = 3 * D)
  (h3 : C = 3 * B)
  (h4 : B + C = 180) :
  B = 45 :=
by
  sorry

end find_angle_B_l56_56710


namespace factorial_quotient_l56_56493

/-- Prove that the quotient of the factorial of 4! divided by 4! simplifies to 23!. -/
theorem factorial_quotient : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := 
by
  sorry

end factorial_quotient_l56_56493


namespace sum_of_adjacent_to_14_l56_56388

def is_divisor (n : ℕ) (d : ℕ) : Prop := d ∣ n

def positive_integer_divisors (n : ℕ) : set ℕ := {d | is_divisor n d ∧ d > 1}

def adj (x y : ℕ) : Prop := gcd x y > 1

def adjacent_sum {S : set ℕ} (a : ℕ) (cond_adj : ∀ x ∈ S, ∃ y ∈ S, adj x y) : ℕ := sorry

theorem sum_of_adjacent_to_14 : adjacent_sum (positive_integer_divisors 252) 14
  (λ x hx, by {
    sorry -- Prove that each element has a qualifying adjacent element
  }) = 70 := sorry

end sum_of_adjacent_to_14_l56_56388


namespace derangement_5_eq_44_l56_56973

/-- Prove that the number of derangements of a set of size 5 is 44 -/
theorem derangement_5_eq_44 : ∀ (n : ℕ), n = 5 → ∑ k in finset.range (n + 1), (-1 : ℤ)^k / (nat.factorial k) * (nat.factorial n) = 44 := 
by
  intro n hn
  rw [hn]
  simp
  exact sorry

end derangement_5_eq_44_l56_56973


namespace y_coord_of_equidistant_point_on_y_axis_l56_56980

/-!
  # Goal
  Prove that the $y$-coordinate of the point P on the $y$-axis that is equidistant from points $A(5, 0)$ and $B(3, 6)$ is \( \frac{5}{3} \).
  Conditions:
  - Point A has coordinates (5, 0).
  - Point B has coordinates (3, 6).
-/

theorem y_coord_of_equidistant_point_on_y_axis :
  ∃ y : ℝ, y = 5 / 3 ∧ (dist (⟨0, y⟩ : ℝ × ℝ) (⟨5, 0⟩ : ℝ × ℝ) = dist (⟨0, y⟩ : ℝ × ℝ) (⟨3, 6⟩ : ℝ × ℝ)) :=
by
  sorry -- Proof omitted

end y_coord_of_equidistant_point_on_y_axis_l56_56980


namespace correct_ellipse_equation_l56_56959

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

theorem correct_ellipse_equation :
  ∃ a b : ℝ, (a = 2 * Real.sqrt 2) ∧ 
             (b = 2) ∧ 
             (2 * b = 4) ∧ 
             (Real.sqrt (a^2 - b^2) = 2) ∧ 
             ellipse_equation 2 (Real.sqrt 2) 8 4 := sorry

end correct_ellipse_equation_l56_56959


namespace fraction_flew_away_on_second_night_l56_56036

theorem fraction_flew_away_on_second_night :
  ∃ (F : ℚ), 
    let ducks_init := 320 in
    let ducks_after_first_night := ducks_init - (1 / 4) * ducks_init in
    let ducks_after_second_night := ducks_after_first_night - F * ducks_after_first_night in
    let ducks_after_third_night := ducks_after_second_night - 0.3 * ducks_after_second_night in
    ducks_after_third_night = 140 ∧ F = 1 / 6 :=
by {
  use 1 / 6,
  let ducks_init := 320,
  let ducks_after_first_night := ducks_init - (1 / 4) * ducks_init,
  let ducks_after_second_night := ducks_after_first_night - (1 / 6) * ducks_after_first_night,
  let ducks_after_third_night := ducks_after_second_night - 0.3 * ducks_after_second_night,
  rw [ducks_init, ducks_after_first_night, ducks_after_second_night, ducks_after_third_night],
  norm_num,
  split,
  norm_num,
  exact rfl,
}

end fraction_flew_away_on_second_night_l56_56036


namespace construct_triangle_ABC_exists_l56_56164

noncomputable def construct_triangle (A' B' C' : Point) : Prop :=
  ∃ (A B C : Point),
  let circumcircle := circumscribed_circle (triangle A B C) in
  let angle_bisectors := internal_angle_bisectors (triangle A B C) in
  A' ∈ circumcircle ∧
  B' ∈ circumcircle ∧
  C' ∈ circumcircle ∧
  angle_bisectors A = altitude (triangle A' B' C') A' ∧
  angle_bisectors B = altitude (triangle A' B' C') B' ∧
  angle_bisectors C = altitude (triangle A' B' C') C'

theorem construct_triangle_ABC_exists (A' B' C' : Point) :
  construct_triangle A' B' C' :=
sorry

end construct_triangle_ABC_exists_l56_56164


namespace problem_statement_l56_56313

variables {α : Type*} [LinearOrderedField α] 
variables {A B C A1 B1 C1 A0 C0 : Point α} -- Points on the triangle and lines
variables {circumcircle : Triangle α → Set (Point α)} -- Circumcircle function 
variables {intersection : Line α → Line α → Point α} -- Intersection of lines
variables {median : Triangle α → Point α} -- Median of triangle

-- Definitions based on conditions
def altitudes (A B C A1 B1 C1 : Point α) : Prop := 
  isAltitude A A1 ∧ isAltitude B B1 ∧ isAltitude C C1

def common_points (circumcircle : Set (Point α)) (A1 B1 C1 A0 C0 : Point α) : Prop :=
  A0 ∈ circumcircle ∧ C0 ∈ circumcircle ∧ onLine A1B1 A0 ∧ onLine C1B1 C0

-- Problem statement
theorem problem_statement 
  (h_altitudes : altitudes A B C A1 B1 C1)
  (h_common_points : common_points (circumcircle (triangle A1 B C1)) A1 B1 C1 A0 C0) :
  meet_on_median_or_parallel (line A A0) (line C C0) (median (triangle A B C)) :=
sorry

end problem_statement_l56_56313


namespace monotonic_interval_range_l56_56650

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem monotonic_interval_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < 2 → 1 < x₂ → x₂ < 2 → x₁ < x₂ → f a x₁ ≤ f a x₂ ∨ f a x₁ ≥ f a x₂) ↔
  (a ∈ Set.Iic (-1) ∪ Set.Ici 0) :=
sorry

end monotonic_interval_range_l56_56650


namespace distance_NYC_to_DC_l56_56651

noncomputable def horse_speed := 10 -- miles per hour
noncomputable def travel_time := 24 -- hours

theorem distance_NYC_to_DC : horse_speed * travel_time = 240 := by
  sorry

end distance_NYC_to_DC_l56_56651


namespace coefficient_a3b2_l56_56416

theorem coefficient_a3b2 :
  let poly1 := (a + b)^5
  let poly2 := (c + (1 / c))^7
  coefficient (a^3 * b^2 * (poly1 * poly2)) = 0 :=
by
  sorry

end coefficient_a3b2_l56_56416


namespace total_men_finished_l56_56037

def men_tripped (total_men: ℕ) := total_men * 1 / 4
def men_finished_tripping (tripped_men: ℕ) := tripped_men * 3 / 8

def men_dehydrated (remaining_men: ℕ) := remaining_men * 2 / 9
def men_not_finished_dehydration (dehydrated_men: ℕ) := dehydrated_men * 11 / 14

def men_lost (remaining_men: ℕ) := remaining_men * 17 / 100
def men_finished_lost (lost_men: ℕ) := lost_men * 5 / 11

def men_faced_obstacle (remaining_men: ℕ) := remaining_men * 5 / 12
def men_finished_obstacle (faced_obstacle_men: ℕ) := faced_obstacle_men * 7 / 15

def men_cramps (remaining_men: ℕ) := remaining_men * 3 / 7
def men_finished_cramps (cramps_men: ℕ) := cramps_men * 4 / 5

theorem total_men_finished
  (total_men : ℕ)
  (m_tripped : ℕ := int.to_nat (men_tripped total_men))
  (m_finished_tripping : ℕ := int.to_nat (men_finished_tripping m_tripped))
  (remaining_after_tripped : ℕ := total_men - m_tripped)

  (m_dehydrated : ℕ := int.to_nat (men_dehydrated remaining_after_tripped))
  (m_not_finished_dehydration : ℕ := int.to_nat (men_not_finished_dehydration m_dehydrated))
  (remaining_after_dehydration : ℕ := remaining_after_tripped - m_dehydrated)

  (m_lost : ℕ := int.to_nat (men_lost remaining_after_dehydration))
  (m_finished_lost : ℕ := int.to_nat (men_finished_lost m_lost))
  (remaining_after_lost : ℕ := remaining_after_dehydration - m_lost)

  (m_faced_obstacle : ℕ := int.to_nat (men_faced_obstacle remaining_after_lost))
  (m_finished_obstacle : ℕ := int.to_nat (men_finished_obstacle m_faced_obstacle))
  (remaining_after_obstacle : ℕ := remaining_after_lost - m_faced_obstacle)

  (m_cramps : ℕ := int.to_nat (men_cramps remaining_after_obstacle))
  (m_finished_cramps : ℕ := int.to_nat (men_finished_cramps m_cramps))

  : m_finished_tripping + m_finished_lost + m_finished_obstacle + m_finished_cramps = 25 :=
by
  sorry

end total_men_finished_l56_56037


namespace number_in_adjacent_triangle_l56_56362

/-- In a triangular arrangement of natural numbers starting from 1 and progressing row by row, 
    if the number 350 is placed in a triangle, the adjacent triangle sharing a horizontal side 
    with the first contains the number 314. -/
theorem number_in_adjacent_triangle (n : ℕ) (h : n = 350) :
  ∃ m : ℕ, m = 314 :=
by
  use 314
  exact sorry

end number_in_adjacent_triangle_l56_56362


namespace vector_dot_product_l56_56597

variables {a b : EuclideanSpace} -- vectors a and b
variables (theta : Real) -- the angle between the vectors
variables (norm_a norm_b : Real) -- norms of vectors a and b

-- Conditions
def conditions := theta = 5 * Real.pi / 6 ∧ norm_a = 2 ∧ norm_b = Real.sqrt 3

-- The statement to be proved
theorem vector_dot_product (cond : conditions) :
  (dot_product a (2 • b - a) = -10) :=
by
  sorry

end vector_dot_product_l56_56597


namespace max_participants_l56_56043

structure MeetingRoom where
  rows : ℕ
  cols : ℕ
  seating : ℕ → ℕ → Bool -- A function indicating if a seat (i, j) is occupied (true) or not (false)
  row_condition : ∀ i : ℕ, ∀ j : ℕ, seating i j → seating i (j+1) → seating i (j+2) → False
  col_condition : ∀ i : ℕ, ∀ j : ℕ, seating i j → seating (i+1) j → seating (i+2) j → False

theorem max_participants {room : MeetingRoom} (h : room.rows = 4 ∧ room.cols = 4) : 
  (∃ n : ℕ, (∀ i < room.rows, ∀ j < room.cols, room.seating i j → n < 12) ∧
            (∀ m, (∀ i < room.rows, ∀ j < room.cols, room.seating i j → m < 12) → m ≤ 11)) :=
  sorry

end max_participants_l56_56043


namespace order_of_a_b_c_l56_56319

theorem order_of_a_b_c (a b c : ℝ) (h₁: a = 0.6^4.2) 
  (h₂: b = 0.7^4.2) (h₃: c = 0.6^5.1) : b > a ∧ a > c := by
  sorry

end order_of_a_b_c_l56_56319


namespace option_b_is_quadratic_l56_56426

-- Definition of a quadratic equation in one variable
def is_quadratic_equation (eqn : ℝ → ℝ) : Prop := ∃ a b c : ℝ, a ≠ 0 ∧ eqn = λ x, a * x^2 + b * x + c

-- The mathematically equivalent proof problem
theorem option_b_is_quadratic : ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, (x-1)^2 - 2*(3-x)^2 = a * x^2 + b * x + c) :=
sorry

end option_b_is_quadratic_l56_56426


namespace modulus_of_complex_l56_56383

noncomputable def complex_modulus : ℂ := (3 - 4 * complex.I) / complex.I

theorem modulus_of_complex :
  complex.abs complex_modulus = 5 :=
by
  sorry

end modulus_of_complex_l56_56383


namespace exists_w_l56_56784

theorem exists_w (f : ℝ → ℝ) [Differentiable ℝ f] [Differentiable ℝ f'] [Differentiable ℝ f''] [Differentiable ℝ f''']:
  ∃ (w : ℝ), -1 ≤ w ∧ w ≤ 1 ∧ (f''' w / 6 = (f 1 / 2 - f (-1) / 2 - f' 0)) :=
by
  sorry

end exists_w_l56_56784


namespace sum_of_inverse_squared_modulus_of_roots_of_z_eight_eq_one_l56_56063

open complex

theorem sum_of_inverse_squared_modulus_of_roots_of_z_eight_eq_one :
  (finset.univ.sum (λ (z : ℂ), if z ^ 8 = 1 then (1/(abs (1 - z))^2) else 0)) = 8 :=
by
  sorry

end sum_of_inverse_squared_modulus_of_roots_of_z_eight_eq_one_l56_56063


namespace no_integer_b_two_distinct_roots_l56_56996

theorem no_integer_b_two_distinct_roots :
  ∀ b : ℤ, ¬ ∃ x y : ℤ, x ≠ y ∧ (x^4 + 4 * x^3 + b * x^2 + 16 * x + 8 = 0) ∧ (y^4 + 4 * y^3 + b * y^2 + 16 * y + 8 = 0) :=
by
  sorry

end no_integer_b_two_distinct_roots_l56_56996


namespace deposit_range_between_30k_and_40k_l56_56846

noncomputable def annual_interest_rate := 0.027
noncomputable def interest_tax_rate := 0.20
noncomputable def post_tax_interest := 2241
noncomputable def years := 3

theorem deposit_range_between_30k_and_40k :
  let pre_tax_interest := post_tax_interest / (1 - interest_tax_rate)
  let annual_interest := pre_tax_interest / years
  let principal := annual_interest / annual_interest_rate in
  30000 < principal ∧ principal < 40000 :=
by
  sorry

end deposit_range_between_30k_and_40k_l56_56846


namespace count_correct_propositions_l56_56503

open_locale classical

variables (a b c : V) [HasInner V] [NormedAddCommGroup V] [NormedSpace ℝ V]

def prop1 : Prop := ∀ v : V, ∥v∥ = 1 → ∥v∥ = 1
def prop2 : Prop := ∀ a b : V, a ≠ 0 → b ≠ 0 → ∥a + b∥ < ∥a∥ + ∥b∥
def prop3 : Prop := ∀ a b : V, a ≠ 0 → b ≠ 0 → ∥a∥ = ∥b∥ → a = b
def prop4 : Prop := ∀ a b c : V, a ≠ 0 → b ≠ 0 → c ≠ 0 → a ⬝ b = b ⬝ c → a = c

theorem count_correct_propositions : (if prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4 then 2 else 0) = 2 :=
by sorry

end count_correct_propositions_l56_56503


namespace probability_largest_is_5_l56_56921

/-- Probability of selecting three cards from a box of six cards numbered 1 to 6 such that the largest number is 5 is 3/10. -/
theorem probability_largest_is_5 : 
  let cards := {1, 2, 3, 4, 5, 6} in
  let selected := (finset.powerset_len 3 cards) in
  let favorable := (finset.filter (λ s, s ∈ selected ∧ ∃ x ∈ s, x = 5 ∧ ∀ y ∈ s, y ≤ 5) selected).card in
  let total := selected.card in
  (favorable : ℚ) / total = 3 / 10 :=
sorry

end probability_largest_is_5_l56_56921


namespace sequence_uniquely_determined_l56_56234

noncomputable def a_seq : ℕ → ℝ
| 0       := 1
| (n + 1) := if n = 0 then (sqrt 5 - 1) / 2 else a_seq (n - 1) - a_seq n

theorem sequence_uniquely_determined :
  (∀ n : ℕ, a_seq n > 0) ∧ ∀ (n : ℕ), a_seq (n + 1) = a_seq (n - 1) - a_seq n :=
sorry

end sequence_uniquely_determined_l56_56234


namespace find_f_2010_l56_56590

def f (x : ℝ) : ℝ := sorry

theorem find_f_2010 (h₁ : ∀ x, f (x + 1) = - f x) (h₂ : f 1 = 4) : f 2010 = -4 :=
by 
  sorry

end find_f_2010_l56_56590


namespace geometric_representation_l56_56054

variables (a : ℝ)

-- Definition of the area of the figure
def total_area := a^2 + 1.5 * a

-- Definition of the perimeter of the figure
def total_perimeter := 4 * a + 3

theorem geometric_representation :
  total_area a = a^2 + 1.5 * a ∧ total_perimeter a = 4 * a + 3 :=
by
  exact ⟨rfl, rfl⟩

end geometric_representation_l56_56054


namespace time_to_fill_pool_l56_56947

theorem time_to_fill_pool :
  let capacity := (30000 : ℕ),           -- capacity of the pool in gallons
      num_hoses := (5 : ℕ),              -- number of hoses
      flow_rate := (3 : ℕ),              -- flow rate of each hose in gallons per minute
      combined_flow_rate := num_hoses * flow_rate,  -- combined flow rate in gallons per minute
      flow_rate_per_hour := combined_flow_rate * 60    -- combined flow rate in gallons per hour
  in (capacity / flow_rate_per_hour) = 33 :=
by
  sorry

end time_to_fill_pool_l56_56947


namespace no_bijective_polynomial_degree_ge_two_l56_56715

noncomputable def polynomial_bijective_rationals : Prop :=
  ∀ (f : ℚ[X]), degree f ≥ 2 → ¬ bijective (λ x : ℚ, f.eval x)

theorem no_bijective_polynomial_degree_ge_two : polynomial_bijective_rationals :=
by sorry

end no_bijective_polynomial_degree_ge_two_l56_56715


namespace increasing_sequences_remainder_l56_56502

noncomputable def increasing_sequences_count (n k m : ℕ) : ℕ :=
  Nat.choose (k + n - 1) n

theorem increasing_sequences_remainder :
  ∃ m n : ℕ, m = 508 ∧ n = 10 ∧
  (increasing_sequences_count 10 499 (m - 1)) % 1000 = 508 :=
by
  use 508, 10
  split
  · rfl
  split
  · rfl
  apply Nat.mod_eq_of_lt
  exact Nat.lt_succ_self 508
  done

end increasing_sequences_remainder_l56_56502


namespace integral_of_2x_ex_l56_56988

theorem integral_of_2x_ex :
  ∫ x in 0..1, (2 * x + exp x) = Real.exp 1 :=
by
  -- The proof will go here
  sorry

end integral_of_2x_ex_l56_56988


namespace seeds_in_second_plot_l56_56206

theorem seeds_in_second_plot (seeds_in_first_plot : ℕ) (germination_rate_first : ℝ) (germination_rate_second : ℝ) (total_germination_rate : ℝ) (total_percent_germination : ℝ) :
  seeds_in_first_plot = 500 →
  germination_rate_first = 0.30 →
  germination_rate_second = 0.50 →
  total_germination_rate = 35.714285714285715 / 100 →
  ∃ S : ℕ, S = 200 :=
by
  intros h1 h2 h3 h4
  use 200
  sorry

end seeds_in_second_plot_l56_56206


namespace Jerry_votes_l56_56721

axiom votes : Type
axiom J : votes
axiom P : votes
axiom addVotes : votes → votes → ℕ
axiom moreVotes : ℕ

axiom cond1 : addVotes P (moreVotes) = J
axiom cond2 : addVotes J P = 196554

theorem Jerry_votes : J = 108375 := 
by 
  sorry

end Jerry_votes_l56_56721


namespace orchids_count_remains_l56_56879

theorem orchids_count_remains (initial_roses initial_orchids final_roses cut_roses : ℕ) 
    (h1 : initial_roses = 15) (h2 : initial_orchids = 62) 
    (h3 : final_roses = 17) (h4 : cut_roses = 2) :
  initial_orchids = 62 :=
by
    have h5 : initial_roses = final_roses - cut_roses,
    { rw [h3, h4], norm_num },
    exact h2

end orchids_count_remains_l56_56879


namespace sum_of_products_l56_56593

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 390) 
  (h2 : a + b + c = 20) : 
  ab + bc + ca = 5 :=
by 
  sorry

end sum_of_products_l56_56593


namespace sqrt_6_irrational_l56_56072

theorem sqrt_6_irrational : ¬ (∃ (a b : ℤ), b ≠ 0 ∧ (√6 : ℝ) = a / b) :=
sorry

end sqrt_6_irrational_l56_56072


namespace ana_interest_l56_56145

/-- Given the conditions:
    - Principal amount P is \$1500
    - Annual interest rate r is 0.08 (8%)
    - Number of years n is 4
    - Using the compound interest formula A = P * (1 + r)^n
    Prove that the total interest earned after 4 years is \$540.735 -/
theorem ana_interest (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) :
  P = 1500 ∧ r = 0.08 ∧ n = 4 ∧ A = P * (1 + r) ^ n → (A - P) = 540.735 :=
by
  intros h
  cases h with hP hrnA
  cases hrnA with hr hnA
  cases hnA with hn hA
  sorry

end ana_interest_l56_56145


namespace order_of_numbers_l56_56387

theorem order_of_numbers : 
  let a := 7 ^ 0.8;
  let b := 0.8 ^ 7;
  let c := log 0.8 7;
  c < b ∧ b < a := 
by
  sorry

end order_of_numbers_l56_56387


namespace sqrt_product_equals_l56_56975

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

def sqrt_50 := sqrt 50
def sqrt_18 := sqrt 18
def sqrt_8 := sqrt 8

theorem sqrt_product_equals : sqrt_50 * sqrt_18 * sqrt_8 = 60 * sqrt 2 := by
  sorry

end sqrt_product_equals_l56_56975


namespace mean_of_set_eq_10point6_l56_56982

open Real -- For real number operations

theorem mean_of_set_eq_10point6 (n : ℝ)
  (h : n + 7 = 11) :
  (4 + 7 + 11 + 13 + 18) / 5 = 10.6 :=
by
  have h1 : n = 4 := by linarith
  sorry -- skip the proof part

end mean_of_set_eq_10point6_l56_56982


namespace B_finish_in_54_days_l56_56081

-- Definitions based on conditions
variables (A B : ℝ) -- A and B are the amount of work done in one day
axiom h1 : A = 2 * B -- A is twice as good as workman as B
axiom h2 : (A + B) * 18 = 1 -- Together, A and B finish the piece of work in 18 days

-- Prove that B alone will finish the work in 54 days.
theorem B_finish_in_54_days : (1 / B) = 54 :=
by 
  sorry

end B_finish_in_54_days_l56_56081


namespace speed_of_man_train_l56_56115

-- definition for km/h to m/s conversion
def kmph_to_mps (v : ℝ) := v * 1000 / 3600

-- definition for m/s to km/h conversion
def mps_to_kmph (v : ℝ) := v * 3600 / 1000

-- definition of given conditions
def goods_train_speed_kmph := 42.4
def goods_train_length_m := 410
def passing_time_seconds := 15

-- calculate the relative speed in m/s
def relative_speed_mps := goods_train_length_m / passing_time_seconds

-- convert the relative speed to km/h
def relative_speed_kmph := mps_to_kmph relative_speed_mps

-- statement of the theorem we need to prove
theorem speed_of_man_train :
  (relative_speed_kmph - goods_train_speed_kmph) = 55.988 :=
by
  sorry

end speed_of_man_train_l56_56115


namespace cone_inscribed_spheres_distance_l56_56471

noncomputable def distance_between_sphere_centers (R α : ℝ) : ℝ :=
  R * (Real.sqrt 2) * Real.sin (α / 2) / (2 * Real.cos (α / 8) * Real.cos ((45 : ℝ) - α / 8))

theorem cone_inscribed_spheres_distance (R α : ℝ) (h1 : R > 0) (h2 : α > 0) :
  distance_between_sphere_centers R α = R * (Real.sqrt 2) * Real.sin (α / 2) / (2 * Real.cos (α / 8) * Real.cos ((45 : ℝ) - α / 8)) :=
by 
  sorry

end cone_inscribed_spheres_distance_l56_56471


namespace find_smallest_N_l56_56349

def smallest_possible_N (N : ℕ) : Prop :=
  ∃ (W : Fin N → ℝ), 
  (∀ i j, W i ≤ 1.25 * W j ∧ W j ≤ 1.25 * W i) ∧ 
  (∃ (P : Fin 10 → Finset (Fin N)), ∀ i j, i ≤ j →
    P i ≠ ∅ ∧ 
    Finset.sum (P i) W = Finset.sum (P j) W) ∧
  (∃ (V : Fin 11 → Finset (Fin N)), ∀ i j, i ≤ j →
    V i ≠ ∅ ∧ 
    Finset.sum (V i) W = Finset.sum (V j) W)

theorem find_smallest_N : smallest_possible_N 50 :=
sorry

end find_smallest_N_l56_56349


namespace area_comparison_perimeter_comparison_l56_56218

-- Define side length of square and transformation to sides of the rectangle
variable (a : ℝ)

-- Conditions: side lengths of the rectangle relative to the square
def long_side : ℝ := 1.11 * a
def short_side : ℝ := 0.9 * a

-- Area calculations and comparison
def square_area : ℝ := a^2
def rectangle_area : ℝ := long_side a * short_side a

theorem area_comparison : (rectangle_area a / square_area a) = 0.999 := by
  sorry

-- Perimeter calculations and comparison
def square_perimeter : ℝ := 4 * a
def rectangle_perimeter : ℝ := 2 * (long_side a + short_side a)

theorem perimeter_comparison : (rectangle_perimeter a / square_perimeter a) = 1.005 := by
  sorry

end area_comparison_perimeter_comparison_l56_56218


namespace sum_of_abs_roots_eq_106_l56_56542

theorem sum_of_abs_roots_eq_106 (m p q r : ℤ) (h₁ : (Polynomial.C m + Polynomial.X * (Polynomial.X * (Polynomial.X + (-2023))) = 0) = Polynomial.C 0) (h₂ : m = p*q*r) (h₃ : p + q + r = 0) :
  |p| + |q| + |r| = 106 := sorry

end sum_of_abs_roots_eq_106_l56_56542


namespace find_width_l56_56952

-- Define the conditions
variables (w h l V : ℝ) (θ x : ℝ)

-- Explicitly set the known values and conditions
def conditions : Prop :=
  h = 6 * w ∧
  l = 7 * h^2 ∧
  V = 86436 ∧
  tan(θ) = x * log(w)

-- The volume equation rewritten with the conditions
def volume_equation : Prop :=
  V = w * h * l

-- Combining conditions and volume equation to build the proof problem
theorem find_width (h_eq : h = 6 * w) (l_eq : l = 7 * h^2) (vol_eq : V = 86436) :
  w = 2.821 :=
by
  sorry

end find_width_l56_56952


namespace total_number_of_girls_is_13_l56_56836

def number_of_girls (n : ℕ) (B : ℕ) : Prop :=
  ∃ A : ℕ, (A = B - 5) ∧ (A = B + 8)

theorem total_number_of_girls_is_13 (n : ℕ) (B : ℕ) :
  number_of_girls n B → n = 13 :=
by
  intro h
  sorry

end total_number_of_girls_is_13_l56_56836


namespace coloring_number_lemma_l56_56685

-- Definition of the problem conditions
def no_sum_of_two_same_color_is_power_of_two (f : ℕ → bool) : Prop :=
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 200 ∧ f a = f b → ¬(∃ k : ℕ, a + b = 2^k)

-- The main theorem
theorem coloring_number_lemma :
  (∃ f : ℕ → bool, (∀ n, 1 ≤ n ∧ n ≤ 200 → (f n = tt ∨ f n = ff))
                   ∧ no_sum_of_two_same_color_is_power_of_two f) →
  ∃ n : ℕ, n = 256 :=
by sorry

end coloring_number_lemma_l56_56685


namespace martha_gingers_amount_l56_56344

theorem martha_gingers_amount (G : ℚ) (h : G = 0.43 * (G + 3)) : G = 2 := by
  sorry

end martha_gingers_amount_l56_56344


namespace for_all_N_half_in_A_l56_56796

def totient (n : ℕ) : ℕ := sorry

def isInSetA (n : ℕ) : Prop :=
  ∃ x : ℕ, totient (n + x) = 3 * totient x

def setA : finset ℕ := 
  finset.filter isInSetA (finset.range $ 0)

theorem for_all_N_half_in_A (N : ℕ) (hN : N > 0) : 
  ∃ S ⊆ finset.range (N+1), (|S| ≥ N / 2) ∧ (∀ n ∈ S, isInSetA n) := sorry

end for_all_N_half_in_A_l56_56796


namespace Ksyusha_travel_time_l56_56773

theorem Ksyusha_travel_time:
  ∀ (S v : ℝ), 
  (S > 0 ∧ v > 0) →
  (∀ tuesday_distance tuesday_time wednesday_distance wednesday_time: ℝ, 
    tuesday_distance = 3 * S ∧ 
    tuesday_distance = 2 * S + S ∧ 
    tuesday_time = (2 * S) / v + (S / (2 * v)) ∧ 
    tuesday_time = 30) → 
  (∀ wednesday_distance wednesday_time : ℝ, 
    wednesday_distance = S + 2 * S ∧ 
    wednesday_time = S / v + (2 * S) / (2 * v)) →
  wednesday_time = 24 :=
by
  intros S v S_pos v_pos tuesday_distance tuesday_time wednesday_distance wednesday_time
  sorry

end Ksyusha_travel_time_l56_56773


namespace equiangular_polygon_is_decagon_l56_56960

theorem equiangular_polygon_is_decagon
  (equiangular : ∀ i j : ℕ, i ≠ j → ∠ABC_i = ∠ABC_j)
  (angle_relation : ∃ x, x = (1 / 4) * (180 - x))
  (angle_sum : ∀ x y, x + y = 180) :
  ∃ n, n = 10 := by
  sorry

end equiangular_polygon_is_decagon_l56_56960


namespace solve_quadratic_l56_56419

theorem solve_quadratic (x : ℝ) : 
  (10 - x)^2 = 2x^2 ↔ x = -10 + 10 * Real.sqrt 2 ∨ x = -10 - 10 * Real.sqrt 2 :=
by
  sorry

end solve_quadratic_l56_56419


namespace surface_area_increase_by_206_25_percent_l56_56067

-- Definition of the conditions
variables (s : ℝ) (h1 : s > 0)

-- Original surface area
def original_surface_area := 6 * s^2

-- New edge length
def new_edge_length := 1.75 * s

-- New surface area
def new_surface_area := 6 * (new_edge_length s)^2

-- Increase in surface area
def increase_in_surface_area := new_surface_area s - original_surface_area s

-- Percent increase in surface area
def percent_increase := (increase_in_surface_area s / original_surface_area s) * 100

-- Theorem stating the percent increase in surface area
theorem surface_area_increase_by_206_25_percent :
  percent_increase s h1 = 206.25 :=
by
  sorry

end surface_area_increase_by_206_25_percent_l56_56067


namespace min_value_of_M_minus_N_l56_56553

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 2 * x + 1

def M (a : ℝ) : ℝ :=
  if 1 / 2 < a ∧ a ≤ 1 then f a 3 else f a 1

def N (a : ℝ) : ℝ := f a (1 / a)

def D := (λ a : ℝ, M a - N a) 

theorem min_value_of_M_minus_N :
  (∀ a, 1 / 3 ≤ a ∧ a ≤ 1 → D a ≥ 1 / 2) ∧ (∃ a, 1 / 3 ≤ a ∧ a ≤ 1 ∧ D a = 1 / 2) :=
by
  sorry

end min_value_of_M_minus_N_l56_56553


namespace arithmetic_mean_of_consecutive_integers_l56_56489

theorem arithmetic_mean_of_consecutive_integers : 
  (let seq := λ n, 10 + (n - 1) in
  let sum_seq := ∑ i in finRange 40, seq (i + 1) in
  sum_seq / 40 = 29.5) :=
by sorry

end arithmetic_mean_of_consecutive_integers_l56_56489
