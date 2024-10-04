import Mathlib
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.FunctionalEquations
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Probability.Conditional
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Geometry
import Mathlib.MeasureTheory.Measure.Zero
import Mathlib.Probability.Basic
import Mathlib.Probability.Statistics.Normalize
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.SolveByElim
import Mathlib.Topology.Euclidean.InnerProduct

namespace inverse_function_l509_509041

def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 2
def g (x : ℝ) : ℝ := 2^(x - 1)

theorem inverse_function (x : ℝ) (hx : x > 0) : f (g x) = x ∧ g (f x) = x :=
by 
  sorry

end inverse_function_l509_509041


namespace eq_hyperbola_line_passes_four_points_l509_509982

-- Define the hyperbola and related parameters
variables (a b : ℝ) (hyp1 : a > 0) (hyp2 : b > 0)
variables (P : ℝ × ℝ) (suppose_P : P = (3,Real.sqrt 2))
variables (c : ℝ) (distance_foci : c = 2)
variables (hyperbola : ∀ (x y : ℝ), (x/a)^2 - (y/b)^2 = 1)

-- Equation of C
theorem eq_hyperbola : 
  (∀ x y : ℝ, (x/a)^2 - (y/b)^2 = 1 ↔ (x^2)/3 - y^2 = 1) :=
sorry

-- Prove line l passes through specific points
variables (A M N D : ℝ × ℝ)
variables (A_on_xaxis : A.2 = 0)
variables (l_not_perpendicular : ¬ (∀ x y : ℝ, y = 0 ∨ x = 0))
variables (l_intersects_C : ∀ (x y : ℝ), (x/a)^2 - (y/b)^2 = 1 → ∃ M N, C = M ∧ C = N)
variables (midpoint_D : D.2 = 0)
variables (AM_AN_AD : |A.1 - M.1||A.1 - N.1| = 2|A.1 - D.1|)

theorem line_passes_four_points : 
  l_not_perpendicular → l_intersects_C → AM_AN_AD → 
  A = (-3,0) ∨ A = (-1,0) ∨ A = (1,0) ∨ A = (3,0) :=
sorry

end eq_hyperbola_line_passes_four_points_l509_509982


namespace alice_meets_bob_at_25_km_l509_509542

-- Define variables for times, speeds, and distances
variables (t : ℕ) (d : ℕ)

-- Conditions
def distance_between_homes := 41
def alice_speed := 5
def bob_speed := 4
def alice_start_time := 1

-- Relating the distances covered by Alice and Bob when they meet
def alice_walk_distance := alice_speed * (t + alice_start_time)
def bob_walk_distance := bob_speed * t
def total_walk_distance := alice_walk_distance + bob_walk_distance

-- Alexander walks 25 kilometers before meeting Bob
theorem alice_meets_bob_at_25_km :
  total_walk_distance = distance_between_homes → alice_walk_distance = 25 :=
by
  sorry

end alice_meets_bob_at_25_km_l509_509542


namespace work_done_l509_509384

-- Definitions of given conditions
def is_cyclic_process (gas: Type) (a b c: gas) : Prop := sorry
def isothermal_side (a b: Type) : Prop := sorry
def line_through_origin (b c: Type) : Prop := sorry
def parabolic_arc_through_origin (c a: Type) : Prop := sorry

def temperature_equality (T_a T_c: ℝ) : Prop :=
  T_a = T_c

def half_pressure (P_a P_c: ℝ) : Prop :=
  P_a = 0.5 * P_c

-- Main theorem statement
theorem work_done (T_0 P_a P_c: ℝ) (a b c: Type) 
  (H_cycle: is_cyclic_process gas a b c)
  (H_isothermal: isothermal_side a b)
  (H_line_origin: line_through_origin b c)
  (H_parabolic_arc: parabolic_arc_through_origin c a)
  (H_temp_eq: temperature_equality T_0 320)
  (H_pressure_half: half_pressure P_a P_c) :
  (work_done gas a b c) = 665 := sorry

end work_done_l509_509384


namespace double_bed_heavier_than_single_l509_509059

theorem double_bed_heavier_than_single (
    S D B : ℝ 
    (h1 : 5 * S + 2 * D = 85)
    (h2 : 8 * S + 3 * D + 4 * B = 230)
    (h3 : 4 * B + 5 * D = 175)
) : D - S = 4.8078 :=
by
  sorry

end double_bed_heavier_than_single_l509_509059


namespace tan_alpha_plus_pi_by_4_eq_1_by_7_cos_beta_eq_6_sqrt2_minus_4_by_15_l509_509998

-- Define the conditions for the first proof
def sin_alpha : ℝ := 3 / 5
def alpha_in_second_quadrant : Prop := π / 2 < pi α ∧ α < π
def tan_alpha_plus_pi_by_4 : ℝ := (tan (α + π / 4))

-- Define the first theorem statement
theorem tan_alpha_plus_pi_by_4_eq_1_by_7 (α : ℝ) (h₁ : sin α = sin_alpha) (h₂ : alpha_in_second_quadrant) :
  tan_alpha_plus_pi_by_4 = 1 / 7 :=
sorry

-- Define the conditions for the second proof
def cos_alpha_minus_beta : ℝ := 1 / 3
def beta_in_first_quadrant : Prop := 0 < β ∧ β < π / 2
def cos_beta : ℝ := cos β

-- Define the second theorem statement
theorem cos_beta_eq_6_sqrt2_minus_4_by_15 (α β : ℝ) 
  (h₁ : cos (α - β) = cos_alpha_minus_beta) 
  (h₂ : sin α = sin_alpha) 
  (h₃ : cos α = -(sqrt (1 - sin_alpha^2)))
  (h₄ : alpha_in_second_quadrant)
  (h₅ : beta_in_first_quadrant) : 
  cos_beta = (6 * sqrt 2 - 4) / 15 := 
sorry

end tan_alpha_plus_pi_by_4_eq_1_by_7_cos_beta_eq_6_sqrt2_minus_4_by_15_l509_509998


namespace part1_part2_part3_part4_l509_509980

-- Given that for any a, b ∈ ℝ, the function f satisfies f(a + b) = f(a) + f(b) - 1
-- And when x > 0, f(x) > 1
-- And f(2) = 3

-- Prove that f(0) = 1
theorem part1 (f : ℝ → ℝ) (H : ∀ (a b : ℝ), f(a + b) = f(a) + f(b) - 1)
  (H_pos : ∀ x : ℝ, x > 0 → f(x) > 1) (H_f2 : f(2) = 3) : f(0) = 1 :=
sorry

-- Prove that f(1) = 2
theorem part2 (f : ℝ → ℝ) (H : ∀ (a b : ℝ), f(a + b) = f(a) + f(b) - 1)
  (H_pos : ∀ x : ℝ, x > 0 → f(x) > 1) (H_f2 : f(2) = 3) : f(1) = 2 :=
sorry

-- Prove that f is strictly increasing on ℝ
theorem part3 (f : ℝ → ℝ) (H : ∀ (a b : ℝ), f(a + b) = f(a) + f(b) - 1)
  (H_pos : ∀ x : ℝ, x > 0 → f(x) > 1) (H_f2 : f(2) = 3) : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f(x₁) < f(x₂) :=
sorry

-- Prove the range of values for k given f(-kx²) + f(kx - 2) < 2 for any x ∈ ℝ
theorem part4 (f : ℝ → ℝ) (H : ∀ (a b : ℝ), f(a + b) = f(a) + f(b) - 1)
  (H_pos : ∀ x : ℝ, x > 0 → f(x) > 1) (H_f2 : f(2) = 3)
  (H_ineq : ∀ x : ℝ, f(-kx²) + f(kx - 2) < 2) : 0 ≤ k ∧ k < 8 :=
sorry

end part1_part2_part3_part4_l509_509980


namespace perpendicular_lines_l509_509917

variable {Circle : Type} [MetricSpace Circle]
variable {A B C D P Q : Circle}

-- Conditions
variables (h1 : isChord A C ∧ isChord B D)
variables (h2 : inter P A C = inter P B D)
variables (h3 : perp Q C A ∧ perp Q D B)

-- To prove
theorem perpendicular_lines (h1 : isChord A C ∧ isChord B D)
                          (h2 : inter P A C = inter P B D)
                          (h3 : perp Q C A ∧ perp Q D B) :
  perp (line P Q) (line A B) := 
  sorry

end perpendicular_lines_l509_509917


namespace rectangle_ratio_l509_509838

-- Define the width of the rectangle
def width : ℕ := 7

-- Define the area of the rectangle
def area : ℕ := 196

-- Define that the length is a multiple of the width
def length_is_multiple_of_width (l w : ℕ) : Prop := ∃ k : ℕ, l = k * w

-- Define that the ratio of the length to the width is 4:1
def ratio_is_4_to_1 (l w : ℕ) : Prop := l / w = 4

theorem rectangle_ratio (l w : ℕ) (h1 : w = width) (h2 : area = l * w) (h3 : length_is_multiple_of_width l w) : ratio_is_4_to_1 l w :=
by
  sorry

end rectangle_ratio_l509_509838


namespace modulus_of_complex_l509_509630

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := 2 + (1 / i)

-- The statement of the proof problem
theorem modulus_of_complex : Complex.abs z = Real.sqrt 5 := by
  sorry -- Proof is omitted, the statement alone is provided.

end modulus_of_complex_l509_509630


namespace arrow_velocity_at_impact_l509_509934

def position_arrow (t : ℝ) : ℝ :=
  -0.5 * t^2 + 100 * t

def velocity_arrow (t : ℝ) : ℝ :=
  -t + 100

def position_edward (t : ℝ) : ℝ :=
  0.5 * t^2 + 1875

theorem arrow_velocity_at_impact :
  ∃ t : ℝ, position_arrow t = position_edward t ∧ velocity_arrow t = 75 :=
by
  use 25
  split
  · -- Show position_arrow 25 = position_edward 25
    sorry
  · -- Show velocity_arrow 25 = 75
    sorry

end arrow_velocity_at_impact_l509_509934


namespace number_of_solutions_l509_509953

open Real

theorem number_of_solutions :
  ∃ (n : ℕ), n = 28 ∧
  ∀ θ ∈ Ioo 0 (4 * π),
  tan (7 * π * cos θ) = cot (7 * π * sin θ) :=
by
  sorry

end number_of_solutions_l509_509953


namespace min_value_inverse_sum_l509_509722

theorem min_value_inverse_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) :
  (1 / x + 1 / y + 1 / z) ≥ 9 :=
  sorry

end min_value_inverse_sum_l509_509722


namespace find_t_if_factor_l509_509944

theorem find_t_if_factor (t : ℝ) : (x - t) divides (4 * x^2 + 11 * x - 3) ↔ t = 1/4 ∨ t = -3 := by
  sorry

end find_t_if_factor_l509_509944


namespace simplify_complex_expr_l509_509756

theorem simplify_complex_expr : 
  ∀ (i : ℂ), i^2 = -1 → ( (2 + 4 * i) / (2 - 4 * i) - (2 - 4 * i) / (2 + 4 * i) )
  = -8/5 + (16/5 : ℂ) * i :=
by
  intro i h_i_squared
  sorry

end simplify_complex_expr_l509_509756


namespace find_g_of_3_over_8_l509_509146

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g_of_3_over_8 :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g(x) ≥ 0) ∧
  g(0) = 1 ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g(x) ≥ g(y)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g(1 - x) = 1 - g(x)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g(x / 4) = g(x) / 2) →
  g (3 / 8) = 1 / 2 :=
begin
  intros h,
  sorry
end

end find_g_of_3_over_8_l509_509146


namespace tangent_value_skew_lines_l509_509236

theorem tangent_value_skew_lines
  (A B C P O : Type)
  (hAB_perp_BC : is_perpendicular A B C)
  (hAB_eq_2 : distance A B = 2)
  (hBC_eq_4 : distance B C = 4)
  (hVolume_sphere : volume_sphere O = 8 * real.sqrt 6 * real.pi)
  (hOn_sphere : points_on_sphere [A, B, C, P] O)
  (hPC_diameter : diameter P C O) :
  tangent_value_skew_angle P B A C = 3 :=
sorry

end tangent_value_skew_lines_l509_509236


namespace area_of_region_of_tangent_segments_l509_509198

theorem area_of_region_of_tangent_segments 
  (r : ℝ) (l : ℝ) (a : ℝ)
  (h_r : r = 3)
  (h_l : l = 4)
  (h_a : a = 4 * π) :
  ∀ (radius : ℝ) (length : ℝ), 
  (radius = r → length = l) → 
  ∃ (area : ℝ), area = a :=
by {
  intros radius length h,
  use a,
  rw [h.1, h.2, h_r, h_l, h_a],
  sorry
}

end area_of_region_of_tangent_segments_l509_509198


namespace inequality_proof_l509_509393

theorem inequality_proof (m n : ℕ) 
  (x y : Fin n → ℝ) 
  (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) 
  (hy : ∀ i, 0 ≤ y i ∧ y i ≤ 1)
  (hxy : ∀ i, x i + y i = 1) : 
  (1 - (Finset.prod (Finset.univ) x))^m + 
  (Finset.prod (Finset.univ) (λ i, 1 - (y i)^m)) ≥ 1 := 
sorry

end inequality_proof_l509_509393


namespace stream_speed_l509_509499

theorem stream_speed {v : ℝ} : 
  (man_swim_still : ℝ) 
  (man_upstream_speed : ℝ)
  (man_downstream_speed : ℝ) : 
  man_swim_still = 9 → 
  man_upstream_speed = 9 - v → 
  man_downstream_speed = 9 + v → 
  (man_downstream_speed / man_upstream_speed) = 2 → 
  v = 3 := 
begin
  intros h1 h2 h3 h4,
  sorry
end

end stream_speed_l509_509499


namespace abc_triangle_intersection_l509_509742

variables {α : Type*} [ordered_ring α] [ordered_comm_ring α]
variables (A B C O K L M N P : point α)

/-- Given a triangle ABC with a point O on segment AB, a circle ω centered at O intersects AO at K and OB at L, and is tangent to AC at M and BC at N. Prove that the intersection of KN and LM lie on the altitude AH of triangle ABC. -/
theorem abc_triangle_intersection
  (triangle : is_triangle A B C)
  (O_on_AB : O ∈ segment A B)
  (circle_omega : circle ω)
  (center_omega : center ω = O)
  (intersects_AO_at_K : K ∈ (segment A O))
  (intersects_OB_at_L : L ∈ (segment O B))
  (tangent_AC_at_M : tangent ω AC M)
  (tangent_BC_at_N : tangent ω BC N)
  (intersection_KN_LM_at_P : P ∈ line (KN) ∧ P ∈ line (LM))
  (altitude_AH : H ⊆ (altitude A B C)) :
  P ∈ altitude A H :=
sorry

end abc_triangle_intersection_l509_509742


namespace original_price_of_suit_l509_509442

theorem original_price_of_suit (P : ℝ) (hP : 0.70 * 1.30 * P = 182) : P = 200 :=
by
  sorry

end original_price_of_suit_l509_509442


namespace three_keys_can_turn_on_two_lamps_l509_509334

-- Definition of the problem
noncomputable def switch_three_keys_to_light_lamps : Prop :=
  ∃ (keys : Finset ℕ) (lamps : Finset ℕ)
    (mapping : ℕ → Finset ℕ),
    (keys.card = 5) ∧
    (∀ k ∈ keys, (mapping k) ⊆ lamps ∧ (mapping k).nonempty) ∧
    (∀ k1 k2 ∈ keys, k1 ≠ k2 → (mapping k1) ≠ (mapping k2)) ∧
    (∀ k, k ∈ keys → ∀ l ∈ lamps, false ∨ (mapping k).use l → false) ∧
    (∀ selected_keys, selected_keys.card = 3 →
       (∃ selected_lamps, selected_lamps.card ≥ 2 ∧
       selected_lamps ⊆ ⋃ (k ∈ selected_keys), mapping k))

-- The theorem statement
theorem three_keys_can_turn_on_two_lamps : switch_three_keys_to_light_lamps :=
sorry

end three_keys_can_turn_on_two_lamps_l509_509334


namespace peter_pizza_total_l509_509388

theorem peter_pizza_total (total_slices : ℕ) (whole_slice : ℕ) (shared_slice : ℚ) (shared_parts : ℕ) :
  total_slices = 16 ∧ whole_slice = 1 ∧ shared_parts = 3 ∧ shared_slice = 1 / (total_slices * shared_parts) →
  whole_slice / total_slices + shared_slice = 1 / 12 :=
by
  sorry

end peter_pizza_total_l509_509388


namespace subtraction_example_l509_509814

theorem subtraction_example :
  145.23 - 0.07 = 145.16 :=
sorry

end subtraction_example_l509_509814


namespace eight_digit_palindromes_using_6_7_8_l509_509811

theorem eight_digit_palindromes_using_6_7_8 :
  ∃ n : ℕ, n = 81 ∧ ∀ (a b c d : ℕ), 
    a ∈ {6, 7, 8} ∧ b ∈ {6, 7, 8} ∧ c ∈ {6, 7, 8} ∧ d ∈ {6, 7, 8} →
    let h := a
    let g := b
    let f := c
    let e := d
    in n = 81 :=
sorry

end eight_digit_palindromes_using_6_7_8_l509_509811


namespace distribution_plans_6_volunteers_4_groups_l509_509160

theorem distribution_plans_6_volunteers_4_groups : 
  let number_of_ways := 4^6 - 4 * 3^6 + 6 * 2^6
  in number_of_ways = 1564 := by
  let number_of_ways := 4^6 - 4 * 3^6 + 6 * 2^6
  show number_of_ways = 1564 from sorry

end distribution_plans_6_volunteers_4_groups_l509_509160


namespace evaluate_g_l509_509359

def g (x : ℝ) : ℝ :=
  if x > 5 then x + 1 else x ^ 3

theorem evaluate_g (h_g : g(g(g(2))) = 10) : g(g(g(2))) = 10 :=
  by
  sorry

end evaluate_g_l509_509359


namespace function_range_is_correct_l509_509052

noncomputable def function_range : set ℝ :=
  {y : ℝ | ∃ x : ℝ, x ∈ Icc (- (2 * Real.pi / 3)) (2 * Real.pi / 3) ∧ 
                    y = (1 + Real.cos x) ^ 2023 + (1 - Real.cos x) ^ 2023}

theorem function_range_is_correct : function_range = set.Icc 2 (2 ^ 2023) :=
by
  sorry

end function_range_is_correct_l509_509052


namespace tom_is_telling_the_truth_l509_509743

-- Define the four gangsters
inductive Gangster
| Alex
| Louis
| Tom
| George

open Gangster

-- Define their statements
def statement (g : Gangster) : Prop :=
match g with
| Alex   => Louis stole the suitcase
| Louis  => Tom is the thief
| Tom    => ¬ (Louis stole the suitcase)
| George => ¬ (George stole the suitcase)
end

-- Define that only one statement is true
def exactly_one_true (statements : List Prop) : Prop :=
(statements.count (λ s => s = true) = 1)

-- The main theorem
theorem tom_is_telling_the_truth :
  (∃ unique g, statement g) → statement Tom :=
by
  intros
  sorry


end tom_is_telling_the_truth_l509_509743


namespace transportation_cost_invariant_l509_509398

-- Definitions for the problem conditions
def weight (dist : ℕ) : ℕ := dist -- The weight is equal to the distance.

def trip_cost (weights : List ℕ) (dist : ℕ) : ℕ :=
(weights.sum) * dist -- Cost of the trip is the sum of weights times the distance.

-- Helper to calculate total cost given an order of visiting settlements
def total_cost (distances : List ℕ) : ℕ :=
(distances.inits.tail.map (λ t => trip_cost t (t.lastD 0))).sum

-- Predicate to check if the transportation cost is invariant under permutation
def is_cost_invariant (distances : List ℕ) : Prop :=
∀ perms : List ℕ, perms ∈ distances.permutations → total_cost perms = total_cost distances

-- Main theorem statement
theorem transportation_cost_invariant (distances : List ℕ) :
  is_cost_invariant distances :=
sorry

end transportation_cost_invariant_l509_509398


namespace find_p_of_tangency_l509_509635

-- condition definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def parabola_eq (p x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0
def circle_center : (ℝ × ℝ) := (3, 0)
def circle_radius : ℝ := 4
def directrix (p : ℝ) : ℝ := -p / 2

-- theorem definition
theorem find_p_of_tangency (p : ℝ) :
  (∀ x y : ℝ, circle_eq x y) →
  (∀ x y : ℝ, parabola_eq p x y) →
  dist (circle_center.fst) (directrix p / 0) = circle_radius →
  p = 2 :=
sorry

end find_p_of_tangency_l509_509635


namespace simplify_rationalize_l509_509407

theorem simplify_rationalize : (1 : ℝ) / (1 + (1 / (real.sqrt 5 + 2))) = (real.sqrt 5 + 1) / 4 :=
by sorry

end simplify_rationalize_l509_509407


namespace eval_expression_l509_509581

theorem eval_expression : (⌈(7: ℚ) / 3⌉ + ⌊ -((7: ℚ) / 3)⌋) = 0 :=
begin
  sorry
end

end eval_expression_l509_509581


namespace max_soap_boxes_in_carton_l509_509860

-- Define the measurements of the carton
def L_carton := 25
def W_carton := 42
def H_carton := 60

-- Define the measurements of the soap box
def L_soap_box := 7
def W_soap_box := 12
def H_soap_box := 5

-- Calculate the volume of the carton
def V_carton := L_carton * W_carton * H_carton

-- Calculate the volume of the soap box
def V_soap_box := L_soap_box * W_soap_box * H_soap_box

-- Define the number of soap boxes that can fit in the carton
def number_of_soap_boxes := V_carton / V_soap_box

-- Prove that the number of soap boxes that can fit in the carton is 150
theorem max_soap_boxes_in_carton : number_of_soap_boxes = 150 :=
by
  -- Placeholder for the proof
  sorry

end max_soap_boxes_in_carton_l509_509860


namespace ordering_eight_four_three_l509_509478

noncomputable def eight_pow_ten := 8 ^ 10
noncomputable def four_pow_fifteen := 4 ^ 15
noncomputable def three_pow_twenty := 3 ^ 20

theorem ordering_eight_four_three :
  eight_pow_ten < three_pow_twenty ∧ three_pow_twenty < four_pow_fifteen :=
by
  sorry

end ordering_eight_four_three_l509_509478


namespace marbles_left_after_operations_l509_509800

theorem marbles_left_after_operations :
  ∃ (W R B G Y : ℕ),
    W = 50 ∧ R = 40 ∧ B = ((200 - 50 - 40) / 2) ∧ G = ((200 - 50 - 40) / 2) ∧ Y = 200 - (50 + 40 + ((200 - 50 - 40) / 2) + ((200 - 50 - 40) / 2)) ∧
    let R' := R - (R / 4),
        B' := B + (R / 4),
        G' := G - (G / 5),
        W' := W + (G / 5 / 2),
        R'' := R' + (G / 5 / 2),
        W'' := W' - 5,
        Y' := Y + 5,
        R''' := R'' + (3 * 5 / 2),
        removed := (3 * |W'' - B'| + Y' / 2)
    in 200 - removed = 153
:= sorry

end marbles_left_after_operations_l509_509800


namespace quadratic_roots_product_sum_l509_509987

theorem quadratic_roots_product_sum :
  ∀ (f g : ℝ), 
  (∀ x : ℝ, 3 * x^2 - 4 * x + 2 = 0 → x = f ∨ x = g) → 
  (f + g = 4 / 3) → 
  (f * g = 2 / 3) → 
  (f + 2) * (g + 2) = 22 / 3 :=
by
  intro f g roots_eq sum_eq product_eq
  sorry

end quadratic_roots_product_sum_l509_509987


namespace area_of_square_with_circles_l509_509933

theorem area_of_square_with_circles (r : ℝ) (h_r : r = 8) : 
  let d := 2 * r in 
  let side := 2 * d in 
  (side ^ 2) = 1024 :=
by
  intros
  have h_d : d = 16 := by linarith
  have h_side : side = 32 := by linarith
  rw [h_side]
  norm_num

end area_of_square_with_circles_l509_509933


namespace smallest_positive_b_l509_509022

variable (f : ℝ → ℝ)

theorem smallest_positive_b 
  (h_period : ∀ x, f(x - 15) = f(x))
  : ∃ b > 0, (∀ x, f((x - b) / 3) = f(x / 3)) ∧ b = 45 :=
sorry

end smallest_positive_b_l509_509022


namespace analytical_expression_transformation_description_monotonic_intervals_center_of_symmetry_l509_509258

-- Conditions
variable (A ω φ : ℝ) 
variable (x0 : ℝ)
variable (k : ℤ)

-- Given conditions
axiom A_pos : A > 0
axiom ω_pos : ω > 0
axiom φ_bound : |φ| < Real.pi / 2
axiom intersection_y : A * Real.sin φ = 3 / 2
axiom max_point : A * Real.sin (ω * x0 + φ) = 3
axiom min_point : A * Real.sin (ω * (x0 + 2 * Real.pi) + φ) = -3

-- Prove the analytical expression, translation, intervals of monotonicity, and symmetry center
theorem analytical_expression :
  f(x) = 3 * Real.sin(1/2 * x + Real.pi / 6) :=
sorry

theorem transformation_description :
  ∃ t s, ∀ x, f(x) = A * Real.sin(ω * (x + t) + φ) := 
sorry

theorem monotonic_intervals :
  ∀ k : ℤ, ∀ x : ℝ, 
  (x ∈ set.Icc (4 * k * Real.pi - 4 * Real.pi / 3) (4 * k * Real.pi + 2 * Real.pi / 3)) ↔ 
  monotonic_increasing_intervals f k :=
sorry

theorem center_of_symmetry :
  ∀ k : ℤ, f (- Real.pi / 3 + 2 * k * Real.pi) = f (2 * k * Real.pi - Real.pi / 3) :=
sorry

end analytical_expression_transformation_description_monotonic_intervals_center_of_symmetry_l509_509258


namespace find_radius_l509_509100

def point := ℝ × ℝ

def distance (p q : point) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def radius_of_circle (r a : ℝ) : ℝ :=
  (a^2 + r^2) / (2 * r)

theorem find_radius (r a R : ℝ)
  (M A B : point)
  (circle_touches_line : A.2 = 0 ∧ B.2 = 0 ∧ A.1 = -a ∧ B.1 = a)
  (M_distance : distance M A = a ∧ distance M B = a)
  (given_circle_touches_line : distance M (r, 0) = r)
  : R = radius_of_circle r a := by
  sorry

end find_radius_l509_509100


namespace product_binary1101_ternary202_eq_260_l509_509938

-- Define the binary number 1101 in base 10
def binary1101 := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the ternary number 202 in base 10
def ternary202 := 2 * 3^2 + 0 * 3^1 + 2 * 3^0

-- Prove that their product in base 10 is 260
theorem product_binary1101_ternary202_eq_260 : binary1101 * ternary202 = 260 := by
  -- Proof 
  sorry

end product_binary1101_ternary202_eq_260_l509_509938


namespace geom_sequence_sum_positive_l509_509614

noncomputable def geom_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geom_sequence_sum_positive {a₁ : ℝ} (hyp_a₁ : a₁ > 0) (q : ℝ) :
  (∀ n : ℕ, n > 0 → geom_sequence (a₁) (q) (n) > 0) ↔ q ∈ Ioo (-1) 0 ∪ Ioi 0 :=
by
  sorry

end geom_sequence_sum_positive_l509_509614


namespace required_english_score_l509_509319

theorem required_english_score (C M : ℕ)
  (h_avg_CM : (C + M) / 2 = 88) :
  let E := 270 - (C + M) in
  E = 94 :=
by
  sorry

end required_english_score_l509_509319


namespace abcde_sum_to_628_l509_509091

theorem abcde_sum_to_628 (a b c d e : ℕ) (h_distinct : (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 5) ∧ 
                                                 (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5) ∧ 
                                                 (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5) ∧ 
                                                 (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5) ∧ 
                                                 (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4 ∨ e = 5) ∧
                                                 a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
                                                 b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
                                                 c ≠ d ∧ c ≠ e ∧
                                                 d ≠ e)
  (h1 : b ≤ d)
  (h2 : c ≥ a)
  (h3 : a ≤ e)
  (h4 : b ≥ e)
  (h5 : d ≠ 5) :
  a^b + c^d + e = 628 := sorry

end abcde_sum_to_628_l509_509091


namespace closest_percentage_covered_by_pentagons_is_63_l509_509443

-- Define the conditions for the problem
def large_square_grid := 4 * 4 = 16
def small_squares_contributing_to_pentagons := 10
def total_small_squares := 16

-- State that 10 out of the 16 small squares contribute to the pentagons
def fraction_covered_by_pentagons := 
  (small_squares_contributing_to_pentagons : ℚ) / total_small_squares = 10 / 16

-- Define the statement we need to prove
theorem closest_percentage_covered_by_pentagons_is_63 :
  (fraction_covered_by_pentagons * 100).round = 63 :=
by
  sorry

end closest_percentage_covered_by_pentagons_is_63_l509_509443


namespace sin_alpha_minus_pi_over_3_l509_509249

theorem sin_alpha_minus_pi_over_3
    (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
    (h3 : Real.tan (α / 2) + Real.cot (α / 2) = 5 / 2) :
    Real.sin (α - π / 3) = (4 - 3 * Real.sqrt 3) / 10 := by
    sorry

end sin_alpha_minus_pi_over_3_l509_509249


namespace find_measure_of_BIC_l509_509699

-- Given a triangle PQR with angle bisectors PS, QT, and RU intersecting at the incenter I
-- and ∠PRQ = 60°, we want to prove that ∠BIC = 60°.

def triangle (P Q R : Type) := true

def angle_bisectors_intersect_at_incenter (P Q R I : Type) := true

variables (P Q R S T U I : Type)

theorem find_measure_of_BIC 
  (h1 : triangle P Q R)
  (h2 : angle_bisectors_intersect_at_incenter P Q R I)
  (h3 : ∠PRQ = 60) : 
  ∠BIC = 60 := 
sorry

end find_measure_of_BIC_l509_509699


namespace bounded_intersection_of_four_half_planes_l509_509040

-- Definition of a closed half-plane
structure HalfPlane where
  boundary : Set Point  -- set of points representing the boundary line
  is_upper : Point → Prop  -- predicate for determining points contained in the half-plane

-- The problem statement in Lean
theorem bounded_intersection_of_four_half_planes 
  (hp1 hp2 hp3 hp4 hp5 : HalfPlane) 
  (h_inter : is_bounded (hp1.is_upper ∩ hp2.is_upper ∩ hp3.is_upper ∩ hp4.is_upper ∩ hp5.is_upper)) :
  ∃ (hp : Set HalfPlane), hp.card = 4 ∧ is_bounded (⋂ h ∈ hp, h.is_upper) :=
by 
  sorry

end bounded_intersection_of_four_half_planes_l509_509040


namespace problem2a_l509_509850

theorem problem2a (AB CE : ℝ) (h_AB : AB = 3) 
  (h_CE : CE = 6) (area_ABED : ℝ) 
  (h_area : area_ABED = 48) : 
  let BE := (BC : ℝ) → BC = sqrt (BC^2 + CE^2),
  BC = 8 in 
  BE = 10 :=
by
  have h1: BC = 8, from sorry  -- This will contain the calculation to find BC
  have BE := sqrt (8^2 + 6^2), from sorry -- This will contain the Pythagorean theorem calculation
  show BE = 10, from sorry -- This needs to be filled in with the concluding steps

end problem2a_l509_509850


namespace sin_alpha_minus_pi_over_3_l509_509247

theorem sin_alpha_minus_pi_over_3 
  (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : tan (α / 2) + cot (α / 2) = 5 / 2) :
  sin (α - π / 3) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end sin_alpha_minus_pi_over_3_l509_509247


namespace production_time_l509_509857

variable (a m : ℝ) -- Define a and m as real numbers

-- State the problem as a theorem in Lean
theorem production_time : (a / m) * 200 = 200 * (a / m) := by
  sorry

end production_time_l509_509857


namespace largest_integer_with_sum_17_l509_509479

theorem largest_integer_with_sum_17 : 
  ∃ n : ℕ, (∀ d ∈ (int.to_digits 10 n), d ≠ 0 ∧ d ∈ finset.range 1 10) ∧ (∑ d in (int.to_digits 10 n).to_finset, d = 17) ∧ n = 98 :=
by
  -- ∃ n, it exists an integer n
  -- ∀ d ∈ (int.to_digits 10 n), all digits d in the base 10 representation of n
  -- are non-zero and belong to the range {1,...,9}
  -- and the sum of all the unique digits equals 17
  -- and n equals 98
  sorry

end largest_integer_with_sum_17_l509_509479


namespace julian_notes_problem_l509_509303

theorem julian_notes_problem (x y : ℤ) (h1 : 3 * x + 4 * y = 151) (h2 : x = 19 ∨ y = 19) :
  x = 25 ∨ y = 25 := 
by
  sorry

end julian_notes_problem_l509_509303


namespace find_B_of_1B8_divisibility_l509_509739

theorem find_B_of_1B8_divisibility :
  ∃ (B : ℕ), B < 10 ∧ (1 + B + 8) % 9 = 0 ∧ B = 0 :=
by {
  use 0,
  split,
  { linarith },
  split,
  { norm_num },
  { refl }
}

end find_B_of_1B8_divisibility_l509_509739


namespace tangent_line_angle_at_x1_l509_509447

noncomputable def f (x : ℝ) : ℝ := - (Real.sqrt 3 / 3) * x^3 + 2

def derivative_f (x : ℝ) : ℝ := - Real.sqrt 3 * x^2

theorem tangent_line_angle_at_x1 :
  let α := arctan (derivative_f 1)
  in α = 2 * Real.pi / 3 :=
by
  sorry

end tangent_line_angle_at_x1_l509_509447


namespace center_digit_is_two_l509_509892

theorem center_digit_is_two :
  ∃ (a b : ℕ), (a^2 < 1000 ∧ b^2 < 1000 ∧ (a^2 ≠ b^2) ∧
  (∀ d, d ∈ [a^2 / 100, (a^2 / 10) % 10, a^2 % 10] → d ∈ [2, 3, 4, 5, 6]) ∧
  (∀ d, d ∈ [b^2 / 100, (b^2 / 10) % 10, b^2 % 10] → d ∈ [2, 3, 4, 5, 6])) ∧
  (∀ d, (d ∈ [2, 3, 4, 5, 6]) → (d ∈ [a^2 / 100, (a^2 / 10) % 10, a^2 % 10] ∨ d ∈ [b^2 / 100, (b^2 / 10) % 10, b^2 % 10])) ∧
  2 = (a^2 / 10) % 10 ∨ 2 = (b^2 / 10) % 10 :=
sorry -- no proof needed, just the statement

end center_digit_is_two_l509_509892


namespace pencils_combined_length_l509_509328

theorem pencils_combined_length (length_pencil1 length_pencil2 : Nat) (h1 : length_pencil1 = 12) (h2 : length_pencil2 = 12) :
  length_pencil1 + length_pencil2 = 24 := by
  sorry

end pencils_combined_length_l509_509328


namespace min_C_over_D_l509_509197

-- Define y + 1/y = D and y^2 + 1/y^2 = C.
theorem min_C_over_D (y C D : ℝ) (hy_pos : 0 < y) (hC : y ^ 2 + 1 / (y ^ 2) = C) (hD : y + 1 / y = D) (hC_pos : 0 < C) (hD_pos : 0 < D) :
  C / D = 2 := by
  sorry

end min_C_over_D_l509_509197


namespace simplify_and_evaluate_l509_509416

theorem simplify_and_evaluate (m : ℝ) (h : m = 4 * real.sqrt 3) : 
  (1 - m / (m - 3)) / ((m^2 - 3*m) / (m^2 - 6*m + 9)) = -real.sqrt 3 / 4 :=
by
  sorry

end simplify_and_evaluate_l509_509416


namespace circumcenter_midpoint_of_excircle_intersection_l509_509131

open Classical

variable {α : Type} [MetricSpace α]

-- Given conditions
variables (A B C O O_A O_B O_C P F G : α)
variables (h_triangle : ∃ Δ : Triangle α, Δ.A = A ∧ Δ.B = B ∧ Δ.C = C)
variables (h_circumcenter : IsCircumcenter O (Triangle.mk A B C) )
variables (h_excircle_B : IsExcircle O_B (Triangle.mk A B C) B F AB)
variables (h_excircle_C : IsExcircle O_C (Triangle.mk A B C) C G AC)
variables (h_P_intersection : LineThrough O_B F ∩ LineThrough O_C G = {P})
-- Proving O is the midpoint of PO_A
theorem circumcenter_midpoint_of_excircle_intersection :
  IsMidpoint O (Segment.mk P O_A) :=
  sorry

end circumcenter_midpoint_of_excircle_intersection_l509_509131


namespace operation_commutative_operation_not_associative_operation_properties_l509_509964

variable (x y z k : ℝ)
variable (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_k : 0 < k)

noncomputable def operation (a b : ℝ) : ℝ := (a * b) / (k * (a + b))

theorem operation_commutative : operation x y = operation y x := by
  sorry

theorem operation_not_associative : operation (operation x y) z ≠ operation x (operation y z) := by
  sorry

theorem operation_properties :
  (operation x y = operation y x) ∧ (operation (operation x y) z ≠ operation x (operation y z)) :=
by
  exact ⟨operation_commutative x y z k pos_x pos_y pos_z pos_k, 
         operation_not_associative x y z k pos_x pos_y pos_z pos_k⟩

end operation_commutative_operation_not_associative_operation_properties_l509_509964


namespace greatest_integer_third_side_l509_509300

theorem greatest_integer_third_side (a b : ℕ) (hab : a = 8 ∧ b = 15) : 
  ∃ c : ℕ, 7 < c ∧ c < 23 ∧ c = 22 :=
by
  rcases hab with ⟨ha, hb⟩
  use 22
  split
  · norm_num
  split
  · norm_num
  sorry

end greatest_integer_third_side_l509_509300


namespace toes_on_bus_is_164_l509_509375

def num_toes_hoopit : Nat := 3 * 4
def num_toes_neglart : Nat := 2 * 5

def num_hoopits : Nat := 7
def num_neglarts : Nat := 8

def total_toes_on_bus : Nat :=
  num_hoopits * num_toes_hoopit + num_neglarts * num_toes_neglart

theorem toes_on_bus_is_164 : total_toes_on_bus = 164 := by
  sorry

end toes_on_bus_is_164_l509_509375


namespace find_salary_l509_509502

variable (S : ℝ)
variable (cashInHand : ℝ) : 2380
variable (fixedDepositPercentage : ℝ) : 0.15
variable (groceriesPercentage : ℝ) : 0.30

theorem find_salary (h1 : cashInHand = 2380) 
  (h2 : fixedDepositPercentage = 0.15)
  (h3 : groceriesPercentage = 0.30) : 
  S = 4000 := 
by
  let remaining_after_fixed_deposit := S * (1 - fixedDepositPercentage)
  let groceries_cost := remaining_after_fixed_deposit * groceriesPercentage
  let remaining_after_groceries := remaining_after_fixed_deposit - groceries_cost
  have h : remaining_after_groceries = cashInHand, from sorry
  sorry

end find_salary_l509_509502


namespace seunghyo_daily_dosage_l509_509016

theorem seunghyo_daily_dosage (total_medicine : ℝ) (daily_fraction : ℝ) (correct_dosage : ℝ) :
  total_medicine = 426 → daily_fraction = 0.06 → correct_dosage = 25.56 →
  total_medicine * daily_fraction = correct_dosage :=
by
  intros ht hf hc
  simp [ht, hf, hc]
  sorry

end seunghyo_daily_dosage_l509_509016


namespace BD_equals_regular_10_gon_side_length_l509_509205

-- Define the points and lengths
variable {A B C D E : ℝ}

-- Define the properties and conditions
axiom h1 : Dist A D = Dist D E
axiom h2 : Dist A D = Dist A C 
axiom h3 : Dist B D = Dist A E
axiom h4 : Parallel (Segment D E) (Segment B C)

-- Main theorem statement
theorem BD_equals_regular_10_gon_side_length (A B C D E : ℝ) 
  (h1 : Dist A D = Dist D E)
  (h2 : Dist A D = Dist A C)
  (h3 : Dist B D = Dist A E)
  (h4 : Parallel (Segment D E) (Segment B C)) :
  Dist B D = 2 * Dist A C * sin (π / 10) := sorry

end BD_equals_regular_10_gon_side_length_l509_509205


namespace angle_between_AC_and_BD_is_90_or_alpha_l509_509315

theorem angle_between_AC_and_BD_is_90_or_alpha
  (A B C D : Type*)
  (dist : A → B → ℝ)
  (angle : A → B → C → ℝ)
  (alpha : ℝ)
  (h1 : dist A B = dist B C)
  (h2 : dist B C = dist C D)
  (h3 : dist C D = dist A B)
  (h4 : angle A B C = alpha)
  (h5 : angle B C D = alpha)
  (h6 : angle C D A = alpha) :
  ∃ (theta : ℝ), (theta = 90 ∨ theta = alpha) :=
sorry

end angle_between_AC_and_BD_is_90_or_alpha_l509_509315


namespace tan_double_angle_l509_509241

theorem tan_double_angle (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 2) : tan (2 * α) = 3 / 4 := 
by sorry

end tan_double_angle_l509_509241


namespace fraction_A_B_l509_509153

noncomputable def A_series : ℝ := 
  ∑' n in (Finset.filter (λ n, even n ∧ n % 3 ≠ 0) Finset.range ∞), (-1)^(n / 2 + 1) / (n^2)

noncomputable def B_series : ℝ :=
  ∑' n in (Finset.filter (λ n, even n ∧ n % 3 = 0) Finset.range ∞), (-1)^(n / (2 * 3)) / (n^2)

theorem fraction_A_B : A_series / B_series = 37 := sorry

end fraction_A_B_l509_509153


namespace minimum_games_l509_509683

/--
Given that in a tournament with five teams where each team was supposed to play exactly one match with each of the other teams:
1. Some games were canceled.
2. All teams ended up with different points.
3. None of the teams had zero points.
4. A win awards three points, a draw one point, and a loss zero points.

Prove that the minimum number of games played is 6.
-/
theorem minimum_games (teams : Finset ℕ) (games_played : ℕ) (points : ℕ → ℕ) 
  (h_teams : teams.card = 5)
  (h_all_diff: Function.Injective points)
  (h_nonzero : ∀ t ∈ teams, points t > 0)
  (h_achieve_points : ∀ t ∈ teams, points t ∈ {1, 2, 3, 4, 5})
  (h_max_points : ∀ t₁ t₂, t₁ ∈ teams → t₂ ∈ teams → t₁ ≠ t₂ → 
                  points t₁ + points t₂ ≤ 3) :
  games_played ≥ 6 :=
sorry

end minimum_games_l509_509683


namespace solve_eq_solve_ineq_l509_509261

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then 2^(-x) else log 3 x

-- 1. Prove that f(x) = 2 implies x = -1 or x = 9
theorem solve_eq : ∀ x : ℝ, f(x) = 2 -> (x = -1 ∨ x = 9) := by
  sorry

-- 2. Prove that f(x) > 1 implies x < 0 or x > 3
theorem solve_ineq : ∀ x : ℝ, f(x) > 1 -> (x < 0 ∨ x > 3) := by
  sorry

end solve_eq_solve_ineq_l509_509261


namespace return_trip_time_is_15_or_67_l509_509112

variable (d p w : ℝ)

-- Conditions
axiom h1 : (d / (p - w)) = 100
axiom h2 : ∃ t : ℝ, t = d / p ∧ (d / (p + w)) = t - 15

-- Correct answer to prove: time for the return trip is 15 minutes or 67 minutes
theorem return_trip_time_is_15_or_67 : (d / (p + w)) = 15 ∨ (d / (p + w)) = 67 := 
by 
  sorry

end return_trip_time_is_15_or_67_l509_509112


namespace smallest_prime_factor_of_2551_l509_509488

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, is_prime p ∧ p ∣ n then Nat.find h else n

theorem smallest_prime_factor_of_2551 : smallest_prime_factor 2551 = 13 :=
by
  sorry

end smallest_prime_factor_of_2551_l509_509488


namespace isosceles_triangle_congruent_side_length_l509_509771

theorem isosceles_triangle_congruent_side_length
  (B : ℕ) (A : ℕ) (P : ℕ) (L : ℕ)
  (h₁ : B = 36) (h₂ : A = 108) (h₃ : P = 84) :
  L = 24 :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_congruent_side_length_l509_509771


namespace centroid_proportionality_l509_509618

variables {A B C D D' O : Point}
variables {AB AC AD AD' : ℝ}
variables (triangleABC : Triangle A B C)
variable (is_centroid : Centroid O A B C)
variables (line_through_O : Line O)
variables (D_on_AB : D ∈ Seg A B ∧ D ∈ line_through_O)
variables (D'_on_AC : D' ∈ Seg A C ∧ D' ∈ line_through_O)

theorem centroid_proportionality (H1 : IsTriangle triangleABC) (H2 : IsCentroid O A B C)
    (H3 : D ∈ Seg A B ∧ D ∈ Line O) (H4 : D' ∈ Seg A C ∧ D' ∈ line_through_O) :
    AB / AD + AC / AD' = 3 :=
sorry

end centroid_proportionality_l509_509618


namespace AA2_perp_KC_l509_509679

-- Definitions and assumptions
variables {A B C A1 A2 K : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
[MetricSpace A1] [MetricSpace A2] [MetricSpace K]

-- Triangle ABC
variable (ABC : Triangle A B C)

-- AA1 is the median
variable (AA1_is_median : is_median (A, A1, B, C))

-- AA2 is the angle bisector
variable (AA2_is_angle_bisector : is_angle_bisector (A, A2, B, C))

-- K is a point on AA1
variable (K_on_AA1 : K ∈ line_segment A A1)

-- KA2 parallel to AC
variable (KA2_parallel_AC : parallel (line_segment K A2) (line_segment A C))

--Goal: AA2 is perpendicular to KC
theorem AA2_perp_KC : perpendicular (line_segment A A2) (line_segment K C) :=
sorry

end AA2_perp_KC_l509_509679


namespace total_toes_on_bus_l509_509372

/-- Definition for the number of toes a Hoopit has -/
def toes_per_hoopit : ℕ := 4 * 3

/-- Definition for the number of toes a Neglart has -/
def toes_per_neglart : ℕ := 5 * 2

/-- Definition for the total number of Hoopits on the bus -/
def hoopit_students_on_bus : ℕ := 7

/-- Definition for the total number of Neglarts on the bus -/
def neglart_students_on_bus : ℕ := 8

/-- Proving that the total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus : hoopit_students_on_bus * toes_per_hoopit + neglart_students_on_bus * toes_per_neglart = 164 := by
  sorry

end total_toes_on_bus_l509_509372


namespace grid_coloring_unique_l509_509895

-- Inclusion of noncomputable theory since we're not providing computation here.
noncomputable theory

-- Define the grid coloring problem
def grid_coloring (n : ℕ) : ℕ :=
  -- Coloring function stub, returning the final count of distinct colorings
  if h : n > 1 then 2 else 0 

-- Theorem statement confirming the solution
theorem grid_coloring_unique (n : ℕ) (h : n > 1) :
  grid_coloring n = 2 := by
  sorry

end grid_coloring_unique_l509_509895


namespace increasing_interval_l509_509039

theorem increasing_interval :
  ∃ a, ∀ x ∈ set.Iio a, ∃ δ > 0, ∀ ε > 0, x < a - ε → f'(x) > 0 :=
by
  let f := λ x : ℝ, -(x - 2) * x
  let f' := derivative f
  have : ∀ x, f' x = -2x + 2 := sorry
  use 1
  intros x hx
  use 1
  split
  · linarith
  intros ε hε h
  rw [this]
  linarith

end increasing_interval_l509_509039


namespace g_inv_f_7_l509_509921

theorem g_inv_f_7 (f g : ℝ → ℝ) (h : ∀ x, f⁻¹ (g x) = 5 * x^2 + 3) :
  g⁻¹ (f 7) = √(4 / 5) ∨ g⁻¹ (f 7) = -√(4 / 5) :=
sorry

end g_inv_f_7_l509_509921


namespace subtract_3a_result_l509_509768

theorem subtract_3a_result (a : ℝ) : 
  (9 * a^2 - 3 * a + 8) + 3 * a = 9 * a^2 + 8 := 
sorry

end subtract_3a_result_l509_509768


namespace smallest_area_of_right_triangle_l509_509076

noncomputable def right_triangle_area (a b : ℝ) : ℝ :=
  if a^2 + b^2 = 6^2 then (1/2) * a * b else 12

theorem smallest_area_of_right_triangle :
  min (right_triangle_area 4 (2 * Real.sqrt 5)) 12 = 4 * Real.sqrt 5 :=
by
  -- Proof goes here
  sorry

end smallest_area_of_right_triangle_l509_509076


namespace conjugate_of_z_l509_509611

theorem conjugate_of_z (z : ℂ) (hz : z = 1 - 2 * complex.I) : complex.conj z = 1 + 2 * complex.I :=
by
  rw [hz]
  rw [complex.conj]
  rw [complex.re, complex.im]
  sorry

end conjugate_of_z_l509_509611


namespace parabola_points_l509_509965

noncomputable def parabola_curve (t : ℝ) : ℝ × ℝ :=
  let x := 3^t - 4 in
  let y := 9^t - 7 * 3^t - 6 in
  (x, y)

theorem parabola_points (t : ℝ) :
  let (x, y) := parabola_curve t in
  y = x^2 + x - 3 :=
by
  sorry

end parabola_points_l509_509965


namespace inequality_proof_l509_509714

-- Definitions and conditions

variables {n : ℕ} {a b : ℝ}

-- We only need to prove the given inequality with the appropriate conditions.
theorem inequality_proof (h_int : n > 1) (h_positive : a > 0) (h_order : a > b > 0) :
  (a^n - b^n) * (1 / b^(n-1) - 1 / a^(n-1)) > 4 * n * (n-1) * ((sqrt a - sqrt b)^2) :=
begin
  sorry
end

end inequality_proof_l509_509714


namespace fifth_powers_sum_eq_l509_509219

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l509_509219


namespace cos_RPT_l509_509691

-- Angle RPQ and angle RPT definitions
variables {α : Type*} [LinearOrderedField α]
variables (P Q R T : α)

-- Given conditions
def cos_angle_RPQ (α : Type*) [LinearOrderedField α] (P Q R : α) : α :=
  24 / 25

def supplementary (x : α) : α := 180 - x

-- Proof statement
theorem cos_RPT (P Q R T : α) (h1 : cos (angle RPQ) = 24 / 25) (h2 : T = -Q) :
  cos (180 - angle RPQ) = -24/25 :=
by {
  sorry
}

end cos_RPT_l509_509691


namespace value_of_b_l509_509878

theorem value_of_b (b : ℝ) (f g : ℝ → ℝ) :
  (∀ x, f x = 2 * x^2 - b * x + 3) ∧ 
  (∀ x, g x = 2 * x^2 + b * x + 3) ∧ 
  (∀ x, g x = f (x + 6)) →
  b = 12 :=
by
  sorry

end value_of_b_l509_509878


namespace product_of_x_values_product_of_all_possible_x_values_l509_509292

theorem product_of_x_values (x : ℚ) (h : abs ((18 : ℚ) / x - 4) = 3) :
  x = 18 ∨ x = 18 / 7 :=
sorry

theorem product_of_all_possible_x_values (x1 x2 : ℚ) (h1 : abs ((18 : ℚ) / x1 - 4) = 3) (h2 : abs ((18 : ℚ) / x2 - 4) = 3) :
  x1 * x2 = 324 / 7 :=
sorry

end product_of_x_values_product_of_all_possible_x_values_l509_509292


namespace quadratic_discriminant_l509_509923

theorem quadratic_discriminant (k : ℝ) : 
  (5 * (x : ℝ) ^ 2 - 10 * (real.sqrt 3) * x + k = 0) → 
  (real.discriminant 5 (-10 * (real.sqrt 3)) k = 0) → 
  k = 15 := 
by
  sorry

end quadratic_discriminant_l509_509923


namespace combinations_of_coins_l509_509282

theorem combinations_of_coins (p n d : ℕ) (h₁ : p ≥ 0) (h₂ : n ≥ 0) (h₃ : d ≥ 0) 
  (value_eq : p + 5 * n + 10 * d = 25) : 
  ∃! c : ℕ, c = 12 :=
sorry

end combinations_of_coins_l509_509282


namespace total_amount_received_l509_509908

-- Define the initial prices and the increases
def initial_price_tv : ℝ := 500
def increase_ratio_tv : ℝ := 2/5
def initial_price_phone : ℝ := 400
def increase_ratio_phone : ℝ := 0.4

-- Calculate the total amount received
theorem total_amount_received : 
  initial_price_tv + increase_ratio_tv * initial_price_tv + initial_price_phone + increase_ratio_phone * initial_price_phone = 1260 :=
by {
  sorry
}

end total_amount_received_l509_509908


namespace problem_statement_l509_509338

def euler_totient (n : ℕ) : ℕ :=
  if n = 1 then 1
  else n * (List.foldl (*) 1 (List.map (λ p, 1 - 1/p) (unique_factorization n)))

def gcd_sum (n : ℕ) : ℚ :=
  let phi_n := euler_totient n
  ((List.range (phi_n + 1)).sum (λ i, rat.of_int (gcd i phi_n) / rat.of_int phi_n))

theorem problem_statement : gcd_sum 2023 = 19.4 := sorry

end problem_statement_l509_509338


namespace water_consumption_l509_509736

theorem water_consumption (num_cows num_goats num_pigs num_sheep : ℕ)
  (water_per_cow water_per_goat water_per_pig water_per_sheep daily_total weekly_total : ℕ)
  (h1 : num_cows = 40)
  (h2 : num_goats = 25)
  (h3 : num_pigs = 30)
  (h4 : water_per_cow = 80)
  (h5 : water_per_goat = water_per_cow / 2)
  (h6 : water_per_pig = water_per_cow / 3)
  (h7 : num_sheep = 10 * num_cows)
  (h8 : water_per_sheep = water_per_cow / 4)
  (h9 : daily_total = num_cows * water_per_cow + num_goats * water_per_goat + num_pigs * water_per_pig + num_sheep * water_per_sheep)
  (h10 : weekly_total = daily_total * 7) :
  weekly_total = 91000 := by
  sorry

end water_consumption_l509_509736


namespace max_chord_length_l509_509645

theorem max_chord_length (θ : Real) : 
  let c := 2 * (2 * Real.sin θ - Real.cos θ + 3)
  let k := 8 * Real.sin θ + Real.cos θ + 1
  let line : (x y:Real) → Prop := y = 2 * x
  let conic : (x y:Real) → Prop := c * x^2 - k * y = 0
  ∃ L : ℝ, (conic L (2*L)) ∧ (∀ L' : ℝ, (conic L' (2*L')) → |L'| ≤ 8) ∧ (L = 8 ∘ Real.sqrt 5) :=
sorry

end max_chord_length_l509_509645


namespace trapezoid_area_l509_509027

theorem trapezoid_area 
  (h : ℝ) (BM CM : ℝ) 
  (height_cond : h = 12) 
  (BM_cond : BM = 15) 
  (CM_cond : CM = 13) 
  (angle_bisectors_intersect : ∃ M : ℝ, (BM^2 - h^2) = 9^2 ∧ (CM^2 - h^2) = 5^2) : 
  ∃ (S : ℝ), S = 260.4 :=
by
  -- Skipping the proof part by using sorry
  sorry

end trapezoid_area_l509_509027


namespace remaining_battery_life_l509_509737

theorem remaining_battery_life (h1 : ∀ t, t = 24 → t / t = 1)
  (h2 : ∀ t, t = 3 → t / t = 1)
  (h3 : ∀ t, t = 9)
  (h4 : ∀ t, t = 60) :
  ∃ t, t = 8 :=
by
  -- Discharge rate when not in use
  have rate_not_in_use := 1 / 24
  -- Discharge rate when in use
  have rate_in_use := 1 / 3
  -- Convert 60 minutes to hours
  have usage_time := 60 / 60
  
  -- Using the given battery conditions
  have battery_not_in_use := 8 * rate_not_in_use
  have battery_in_use := 1 * rate_in_use
  have battery_consumed := battery_not_in_use + battery_in_use
  have battery_remaining := 1 - battery_consumed
  have remaining_time := battery_remaining / rate_not_in_use

  -- Return the required result
  exact ⟨remaining_time, rfl⟩

end remaining_battery_life_l509_509737


namespace find_p_from_circle_and_parabola_tangency_l509_509636

theorem find_p_from_circle_and_parabola_tangency :
  (∃ x y : ℝ, (x^2 + y^2 - 6*x - 7 = 0) ∧ (y^2 = 2*p * x) ∧ p > 0) →
  p = 2 :=
by {
  sorry
}

end find_p_from_circle_and_parabola_tangency_l509_509636


namespace derivative_sum_of_distinct_real_roots_l509_509266

noncomputable def f (a x : ℝ) := a * x - Real.log x
noncomputable def f' (a x : ℝ) := a - 1 / x

theorem derivative_sum_of_distinct_real_roots (a x1 x2 : ℝ) (h1 : a ∈ Set.Icc 0 (Real.log x2 + Real.log x1) / x2)
  (h2 : f a x1 = 0) (h3 : f a x2 = 0) (h4 : 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2) :
  f' a x1 + f' a x2 < 0 := sorry

end derivative_sum_of_distinct_real_roots_l509_509266


namespace find_angle_A_find_area_l509_509698

-- Definition for angle A
theorem find_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
  (h_tria : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_A : 0 < A ∧ A < Real.pi) :
  A = Real.pi / 3 :=
by
  sorry

-- Definition for area of triangle ABC
theorem find_area (a b c : ℝ) (A : ℝ)
  (h_a : a = Real.sqrt 7) 
  (h_b : b = 2)
  (h_A : A = Real.pi / 3) 
  (h_c : c = 3) :
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end find_angle_A_find_area_l509_509698


namespace num_ordered_pairs_l509_509567

theorem num_ordered_pairs :
  ∃ (m n : ℤ), (m * n ≥ 0) ∧ (m^3 + n^3 + 99 * m * n = 33^3) ∧ (35 = 35) :=
by
  sorry

end num_ordered_pairs_l509_509567


namespace find_unknown_number_l509_509701

-- Define the problem conditions and required proof
theorem find_unknown_number (a b : ℕ) (h1 : 2 * a = 3 + b) (h2 : (a - 6)^2 = 3 * b) : b = 3 ∨ b = 27 :=
sorry

end find_unknown_number_l509_509701


namespace minor_premise_of_tangent_l509_509690

theorem minor_premise_of_tangent (h1 : ∀ f, TrigFunction f → PeriodicFunction f)
                                 (h2 : TrigFunction (fun x => Real.tan x))
                                 (h3 : PeriodicFunction (fun x => Real.tan x)) :
                                 (∃ P Q, h2 = P ∧ h3 = Q ∧ (∀ P, TrigFunction (fun x => Real.tan x) → P)) :=
  sorry

end minor_premise_of_tangent_l509_509690


namespace hyperbola_slope_of_asymptote_positive_value_l509_509785

noncomputable def hyperbola_slope_of_asymptote (x y : ℝ) : ℝ :=
  if h : (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4)
  then (Real.sqrt 5) / 2
  else 0

-- Statement of the mathematically equivalent proof problem
theorem hyperbola_slope_of_asymptote_positive_value :
  ∃ x y : ℝ, (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) ∧
  hyperbola_slope_of_asymptote x y = (Real.sqrt 5) / 2 :=
by
  sorry

end hyperbola_slope_of_asymptote_positive_value_l509_509785


namespace product_binary1101_ternary202_eq_260_l509_509937

-- Define the binary number 1101 in base 10
def binary1101 := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the ternary number 202 in base 10
def ternary202 := 2 * 3^2 + 0 * 3^1 + 2 * 3^0

-- Prove that their product in base 10 is 260
theorem product_binary1101_ternary202_eq_260 : binary1101 * ternary202 = 260 := by
  -- Proof 
  sorry

end product_binary1101_ternary202_eq_260_l509_509937


namespace tens_digit_of_8_pow_1234_l509_509815

theorem tens_digit_of_8_pow_1234 :
  (8^1234 / 10) % 10 = 0 :=
sorry

end tens_digit_of_8_pow_1234_l509_509815


namespace not_perfect_square_4n_squared_plus_4n_plus_4_l509_509835

theorem not_perfect_square_4n_squared_plus_4n_plus_4 :
  ¬ ∃ m n : ℕ, m^2 = 4 * n^2 + 4 * n + 4 := 
by
  sorry

end not_perfect_square_4n_squared_plus_4n_plus_4_l509_509835


namespace total_toes_on_bus_l509_509380

-- Definitions based on conditions
def toes_per_hand_hoopit : Nat := 3
def hands_per_hoopit : Nat := 4
def number_of_hoopits_on_bus : Nat := 7

def toes_per_hand_neglart : Nat := 2
def hands_per_neglart : Nat := 5
def number_of_neglarts_on_bus : Nat := 8

-- We need to prove that the total number of toes on the bus is 164
theorem total_toes_on_bus :
  (toes_per_hand_hoopit * hands_per_hoopit * number_of_hoopits_on_bus) +
  (toes_per_hand_neglart * hands_per_neglart * number_of_neglarts_on_bus) = 164 :=
by sorry

end total_toes_on_bus_l509_509380


namespace problem_statement_l509_509762

theorem problem_statement (p : ℕ) (hp : Nat.Prime p) :
  ∀ n : ℕ, (∃ φn : ℕ, φn = Nat.totient n ∧ p ∣ φn ∧ (∀ a : ℕ, Nat.gcd a n = 1 → n ∣ a ^ (φn / p) - 1)) ↔ 
  (∃ q1 q2 : ℕ, q1 ≠ q2 ∧ Nat.Prime q1 ∧ Nat.Prime q2 ∧ q1 ≡ 1 [MOD p] ∧ q2 ≡ 1 [MOD p] ∧ q1 ∣ n ∧ q2 ∣ n ∨ 
  (∃ q : ℕ, Nat.Prime q ∧ q ≡ 1 [MOD p] ∧ q ∣ n ∧ p ^ 2 ∣ n)) :=
by {
  sorry
}

end problem_statement_l509_509762


namespace fixed_point_l509_509602

-- Definitions of the point P inside a right angle O, and constructs PA and PB as orthogonals
variables {O P A B : Type} [MetricSpace O A B P]

-- Define OA and OB distances as 'a' and 'b' respectively
variables (a b : ℝ) (P : ℝ)
hypothesis h1 : P = 2 * (a + b)

-- Statement to prove the fixed point of perpendicular to diagonal AB 
theorem fixed_point (h1 : P = 2 * (a + b)) :
  ∃ F, ∀ O A P B : Type, (PA ⊥ O) ∧ (PB ⊥ O) → P % diagonal_P_in_AB → passes_fixed_point F :=
by
  sorry

end fixed_point_l509_509602


namespace number_of_students_l509_509536

theorem number_of_students (pencils: ℕ) (pencils_per_student: ℕ) (total_students: ℕ) 
  (h1: pencils = 195) (h2: pencils_per_student = 3) (h3: total_students = pencils / pencils_per_student) :
  total_students = 65 := by
  -- proof would go here, but we skip it with sorry for now
  sorry

end number_of_students_l509_509536


namespace max_distance_difference_l509_509983

-- Define the coordinates of point P
def point_P (t : ℝ) : ℝ × ℝ := (t, t - 1)

-- Circle E equation: x^2 + y^2 = 1/4
def on_circle_E (x y : ℝ) : Prop := x^2 + y^2 = 1 / 4

-- Circle F equation: (x - 3)^2 + (y + 1)^2 = 9 / 4
def on_circle_F (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 9 / 4

-- Distance between two points (x1, y1) and (x2, y2)
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the function |PF| - |PE|
def distance_difference (t : ℝ) (xE yE xF yF : ℝ) : ℝ :=
  distance t (t - 1) xF yF - distance t (t - 1) xE yE

-- Declare the Lean statement that proves the maximum value is 4
theorem max_distance_difference : 
  ∃ (t : ℝ) (xE yE xF yF : ℝ), 
    on_circle_E xE yE ∧ on_circle_F xF yF ∧ distance_difference t xE yE xF yF = 4 := 
sorry

end max_distance_difference_l509_509983


namespace sequence_equality_l509_509989

theorem sequence_equality (n : ℕ) (hpos : ∀ n : ℕ, n > 0 → a_n > 0) 
  (hcondition : ∀ n : ℕ, n > 0 → (∑ i in range n.succ, a i ^ 3) = (∑ i in range n.succ, a i) ^ 2) : 
  ∀ n : ℕ, n > 0 → a n = n :=
by
  sorry

end sequence_equality_l509_509989


namespace walnut_trees_remaining_l509_509060

theorem walnut_trees_remaining (initial_walnut_trees cut_walnut_trees : ℕ) (h1 : initial_walnut_trees = 42) (h2 : cut_walnut_trees = 13) : 
  initial_walnut_trees - cut_walnut_trees = 29 := by
  rw [h1, h2]
  norm_num
  sorry

end walnut_trees_remaining_l509_509060


namespace tan_exists_n_l509_509172

theorem tan_exists_n (n : ℤ) (h : -90 < n ∧ n < 90) : 
  tan (n * Real.pi / 180) = tan (850 * Real.pi / 180) :=
by
  use -50
  sorry

end tan_exists_n_l509_509172


namespace greatest_possible_value_expression_l509_509660

theorem greatest_possible_value_expression (x : ℝ) (h : x > 10) :
  ((log x) ^ (log (log (log x)))) - ((log (log x)) ^ (log (log x))) = 0 :=
by
  sorry

end greatest_possible_value_expression_l509_509660


namespace opposite_of_neg_2023_is_2023_l509_509440

theorem opposite_of_neg_2023_is_2023 :
  opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_is_2023_l509_509440


namespace value_of_x_in_equation_l509_509078

theorem value_of_x_in_equation : 
  ∃ x, 8^3 + 8^3 + 8^3 = 2^x ∧ x = 9 + Real.log2 3 :=
begin
  sorry
end

end value_of_x_in_equation_l509_509078


namespace total_students_l509_509308

def number_of_girls : ℕ := 190

def ratio_boys_girls : ℕ × ℕ := (8, 5)

theorem total_students (h: number_of_girls = 190) (r: ratio_boys_girls = (8, 5)) : 
  let boys := (8 * (190 / 5)) in
  let girls := 190 in
  boys + girls = 494 :=
by
  sorry

end total_students_l509_509308


namespace k_value_l509_509656

open Set

theorem k_value {k : ℕ} :
  let A := {1, 2, k}
  let B := {1, 2, 3, 5}
  let union_AB := {1, 2, 3, 5} in
  A ∪ B = union_AB → (k = 3 ∨ k = 5) :=
by
  intro A B union_AB h
  have hA : A = {1, 2, k} := rfl
  have hB : B = {1, 2, 3, 5} := rfl
  have hunion_AB : union_AB = {1, 2, 3, 5} := rfl
  sorry

end k_value_l509_509656


namespace distinct_real_numbers_f_satisfy_l509_509352

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem distinct_real_numbers_f_satisfy :
  {c : ℝ | f (f (f (f c))) = 3}.to_finset.card = 9 :=
by
  sorry

end distinct_real_numbers_f_satisfy_l509_509352


namespace all_of_the_above_were_used_as_money_l509_509826

-- Defining the conditions that each type was used as money
def gold_used_as_money : Prop := True
def stones_used_as_money : Prop := True
def horses_used_as_money : Prop := True
def dried_fish_used_as_money : Prop := True
def mollusk_shells_used_as_money : Prop := True

-- The statement to prove
theorem all_of_the_above_were_used_as_money :
  gold_used_as_money ∧
  stones_used_as_money ∧
  horses_used_as_money ∧
  dried_fish_used_as_money ∧
  mollusk_shells_used_as_money ↔
  (∀ x, (x = "gold" ∨ x = "stones" ∨ x = "horses" ∨ x = "dried fish" ∨ x = "mollusk shells") → 
  (x = "gold" ∧ gold_used_as_money) ∨ 
  (x = "stones" ∧ stones_used_as_money) ∨ 
  (x = "horses" ∧ horses_used_as_money) ∨ 
  (x = "dried fish" ∧ dried_fish_used_as_money) ∨ 
  (x = "mollusk shells" ∧ mollusk_shells_used_as_money)) :=
by
  sorry

end all_of_the_above_were_used_as_money_l509_509826


namespace coloring_ways_l509_509560

-- Define the 8x8 board concept and properties
structure Board :=
(width : Nat)
(height : Nat)
(board_cells : Fin width → Fin height → Bool) -- Bool represents colored (True = black, False = white)

def no_adjacent_black_cells (b : Board) : Prop :=
  ∀ (x : Fin b.width) (y : Fin b.height),
    b.board_cells x y = tt → (
      (x.val > 0 → b.board_cells ⟨x.val - 1, x.isLt⟩ y = ff) ∧
      (x.val < b.width - 1 → b.board_cells ⟨x.val + 1, x.isLt⟩ y = ff) ∧
      (y.val > 0 → b.board_cells x ⟨y.val - 1, y.isLt⟩ = ff) ∧
      (y.val < b.height - 1 → b.board_cells x ⟨y.val + 1, y.isLt⟩ = ff)
    )

def count_black_cells (b : Board) : Nat :=
  Finset.univ.sum (λ x, Finset.univ.sum (λ y, if b.board_cells x y then 1 else 0))

def is_valid_coloring (b : Board) : Prop :=
  count_black_cells b = 31 ∧ no_adjacent_black_cells b

-- The proof problem statement
theorem coloring_ways : ∃ (count : Nat), count = 68 :=
by
  exists 68
  -- The actual proof is omitted
  sorry

end coloring_ways_l509_509560


namespace cassidy_current_posters_l509_509143

theorem cassidy_current_posters : 
  ∃ P : ℕ, 
    (P + 6 = 2 * 14) → 
    P = 22 :=
begin
  sorry
end

end cassidy_current_posters_l509_509143


namespace martha_flower_cost_l509_509368

theorem martha_flower_cost :
  let roses_per_centerpiece := 8
  let orchids_per_centerpiece := 2 * roses_per_centerpiece
  let lilies_per_centerpiece := 6
  let centerpieces := 6
  let cost_per_flower := 15
  let total_roses := roses_per_centerpiece * centerpieces
  let total_orchids := orchids_per_centerpiece * centerpieces
  let total_lilies := lilies_per_centerpiece * centerpieces
  let total_flowers := total_roses + total_orchids + total_lilies
  let total_cost := total_flowers * cost_per_flower
  total_cost = 2700 :=
by
  let roses_per_centerpiece := 8
  let orchids_per_centerpiece := 2 * roses_per_centerpiece
  let lilies_per_centerpiece := 6
  let centerpieces := 6
  let cost_per_flower := 15
  let total_roses := roses_per_centerpiece * centerpieces
  let total_orchids := orchids_per_centerpiece * centerpieces
  let total_lilies := lilies_per_centerpiece * centerpieces
  let total_flowers := total_roses + total_orchids + total_lilies
  let total_cost := total_flowers * cost_per_flower
  -- Proof to be added here
  sorry

end martha_flower_cost_l509_509368


namespace angle_B_measure_l509_509626

def is_axisymmetric (triangle : Type) : Prop :=
  ∃ (A B C : triangle), isosceles A B C

noncomputable def measure_angle_A (A : ℕ) : ℕ := 70

noncomputable def measure_angle_B (triangle : Type) (A : ℕ) :=
  if A = 70 then 70 else 55

theorem angle_B_measure (triangle : Type) (h : is_axisymmetric triangle) (A : ℕ) (hA : A = measure_angle_A A) :
  measure_angle_B triangle A = 70 ∨ measure_angle_B triangle A = 55 :=
by
  apply or.inl
  sorry

end angle_B_measure_l509_509626


namespace length_of_AB_l509_509615

-- Define the parabola and the line passing through (0, 1) with angle of inclination π/6
def parabola (x y : ℝ) : Prop := x^2 = 4 * y
def line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x + 1

-- Define the intersection points A and B on the parabola and line
def intersection_points : set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line p.1 p.2}
-- The length |AB| is the Euclidean distance between the intersection points A and B
noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem length_of_AB :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ dist A B = 16 / 3 :=
sorry

end length_of_AB_l509_509615


namespace count_multiples_of_6_l509_509072

def is_divisible_by_6 (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n % 3 = 0

def valid_permutations := {
  136, 164, 314, 316, 346, 364,
  461, 463, 613, 614, 631, 643
}

theorem count_multiples_of_6 : 
  finset.card (finset.filter is_divisible_by_6 valid_permutations) = 2 :=
by
  sorry

end count_multiples_of_6_l509_509072


namespace tetrahedron_symmetries_cube_symmetries_octahedron_symmetries_dodecahedron_symmetries_icosahedron_symmetries_all_polyhedra_symmetries_correct_l509_509286

-- Definitions based on the problem conditions
def self_symmetries (n γ : ℕ) : ℕ := 2 * n * γ

-- Given face counts for polyhedra
def faces_tetrahedron := 4
def faces_cube := 6
def faces_octahedron := 8
def faces_dodecahedron := 12
def faces_icosahedron := 20

-- The number of sides of each polygon (face) and expected symmetries
theorem tetrahedron_symmetries : self_symmetries 3 faces_tetrahedron = 24 := 
by simp [self_symmetries, faces_tetrahedron]; rfl

theorem cube_symmetries : self_symmetries 4 faces_cube = 48 := 
by simp [self_symmetries, faces_cube]; rfl

theorem octahedron_symmetries : self_symmetries 3 faces_octahedron = 48 := 
by simp [self_symmetries, faces_octahedron]; rfl

theorem dodecahedron_symmetries : self_symmetries 5 faces_dodecahedron = 120 := 
by simp [self_symmetries, faces_dodecahedron]; rfl

theorem icosahedron_symmetries : self_symmetries 3 faces_icosahedron = 120 := 
by simp [self_symmetries, faces_icosahedron]; rfl

-- Grouping them into a single theorem can be useful to validate all together
theorem all_polyhedra_symmetries_correct :
  self_symmetries 3 faces_tetrahedron = 24 ∧
  self_symmetries 4 faces_cube = 48 ∧
  self_symmetries 3 faces_octahedron = 48 ∧
  self_symmetries 5 faces_dodecahedron = 120 ∧
  self_symmetries 3 faces_icosahedron = 120 := by
  simp [self_symmetries, faces_tetrahedron, faces_cube, faces_octahedron,
        faces_dodecahedron, faces_icosahedron]; rfl

end tetrahedron_symmetries_cube_symmetries_octahedron_symmetries_dodecahedron_symmetries_icosahedron_symmetries_all_polyhedra_symmetries_correct_l509_509286


namespace f_of_neg2_f_of_3_l509_509729

def f : ℝ → ℝ :=
λ x, if x < 1 then 4 * x + 7 else 10 - 3 * x

theorem f_of_neg2 : f (-2) = -1 := by
  sorry

theorem f_of_3 : f 3 = 1 := by
  sorry

end f_of_neg2_f_of_3_l509_509729


namespace find_angle_ACB_l509_509317

-- Definitions
variable {A B C D E : Type} [AffineSpace ℝ ℝ A] [AffineSpace ℝ ℝ B] [AffineSpace ℝ ℝ C]

def parallel_DC_AB (DC AB : ℝ) : Prop := DC ∥ AB
def perpendicular_CE_AB (CE AB : ℝ) : Prop := CE ⟂ AB
def angle_DCA : ℝ := 50
def angle_ABC : ℝ := 80

-- Theorem to prove
theorem find_angle_ACB (h1 : parallel_DC_AB DC AB) (h2 : perpendicular_CE_AB CE AB) (h3 : angle_DCA = 50) (h4 : angle_ABC = 80) :
  ∠ ACB = 40 :=
sorry

end find_angle_ACB_l509_509317


namespace estimate_students_in_range_l509_509329

noncomputable def n_students := 3000
noncomputable def score_range_low := 70
noncomputable def score_range_high := 80
noncomputable def est_students_in_range := 408

theorem estimate_students_in_range : ∀ (n : ℕ) (k : ℕ), n = n_students →
  k = est_students_in_range →
  normal_distribution :=
sorry

end estimate_students_in_range_l509_509329


namespace remainder_valid_arrangements_mod_1000_l509_509147

def is_valid_arrangement (arr : list (list ℕ)) : Prop :=
  arr.length = 3 ∧ (∀ row ∈ arr, row.length = 3 ∧ 
  let a₁ := median row[0] row[1] row[2] in
  let a₂ := median row[0] row[1] row[2] in
  let a₃ := median row[0] row[1] row[2] in
  median [a₁, a₂, a₃] = 5)

def number_of_valid_arrangements : ℕ :=
  list.length (filter is_valid_arrangement (all_3x3_permutations [1, 2, 3, 4, 5, 6, 7, 8, 9]))

theorem remainder_valid_arrangements_mod_1000 : number_of_valid_arrangements % 1000 = 360 :=
by sorry

end remainder_valid_arrangements_mod_1000_l509_509147


namespace steven_more_peaches_than_apples_l509_509330

theorem steven_more_peaches_than_apples : 
  (∀ (apples peaches : ℕ), apples = 11 ∧ peaches = 18 → peaches - apples = 7) :=
by
  assume (apples peaches : ℕ),
  assume h : apples = 11 ∧ peaches = 18,
  cases h with happles hpeaches,
  rw [happles, hpeaches],
  compute,
  exact rfl

end steven_more_peaches_than_apples_l509_509330


namespace radius_of_curvature_correct_l509_509184

open Real

noncomputable def radius_of_curvature_squared (a b t_0 : ℝ) : ℝ :=
  (a^2 * sin t_0^2 + b^2 * cos t_0^2)^3 / (a^2 * b^2)

theorem radius_of_curvature_correct (a b t_0 : ℝ) (h : a > 0) (h₁ : b > 0) :
  radius_of_curvature_squared a b t_0 = (a^2 * sin t_0^2 + b^2 * cos t_0^2)^3 / (a^2 * b^2) :=
sorry

end radius_of_curvature_correct_l509_509184


namespace lcm_14_21_35_l509_509075

-- Define the numbers
def a : ℕ := 14
def b : ℕ := 21
def c : ℕ := 35

-- Define the prime factorizations
def prime_factors_14 : List (ℕ × ℕ) := [(2, 1), (7, 1)]
def prime_factors_21 : List (ℕ × ℕ) := [(3, 1), (7, 1)]
def prime_factors_35 : List (ℕ × ℕ) := [(5, 1), (7, 1)]

-- Prove the least common multiple
theorem lcm_14_21_35 : Nat.lcm (Nat.lcm a b) c = 210 := by
  sorry

end lcm_14_21_35_l509_509075


namespace probability_of_A_selected_l509_509187

def people := {A, B, C, D}
def selected_event : set (people × people) := {(A, B), (A, C), (A, D), (B, C), (B, D), (C, D)}
def total_events := selected_event.to_finset.card
def events_including_A : set (people × people) := {(A, B), (A, C), (A, D)}
def count_including_A := events_including_A.to_finset.card
def probability_A_selected := count_including_A / total_events

theorem probability_of_A_selected : probability_A_selected = 1 / 2 :=
by
sorry

end probability_of_A_selected_l509_509187


namespace mean_of_combined_sets_l509_509045

theorem mean_of_combined_sets (A : Finset ℝ) (B : Finset ℝ)
  (hA_len : A.card = 7) (hB_len : B.card = 8)
  (hA_mean : (A.sum id) / 7 = 15) (hB_mean : (B.sum id) / 8 = 22) :
  (A.sum id + B.sum id) / 15 = 18.73 :=
by sorry

end mean_of_combined_sets_l509_509045


namespace find_conjugate_of_z_l509_509426

def complex_conjugate_of_z_given_condition (z : ℂ) : Prop :=
  let sqrt3_minus_i_abs := complex.abs (⟨√3, -1⟩ : ℂ)
  (1 + complex.I) * z = sqrt3_minus_i_abs → complex.conj z = 1 - complex.I

theorem find_conjugate_of_z
  (z : ℂ)
  (h : complex_conjugate_of_z_given_condition z) :
  complex.conj z = 1 - complex.I :=
by sorry

end find_conjugate_of_z_l509_509426


namespace proof_ellipse_proof_lambda_l509_509641

noncomputable def ellipse_equation : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ 
  let c := sqrt 3 in
  let C := (λ (x y : ℝ), ((x^2) / (a^2) + (y^2) / (b^2) = 1)) in
  C (sqrt 3) 0 ∧ C (-sqrt 3, 1 / 2) ∧
  (∀ x y : ℝ, C x y ↔ ((x^2) / 4 + y^2 = 1)) ∧
  λ : ℝ,
  ∀ l : ℝ → ℝ, let A B : ℝ × ℝ := (0, l 0), (sqrt 3, l(sqrt 3)) in 
  let P : ℝ × ℝ := (0, 0) in
  let OP := sqrt (sum (λ (i : ℝ × ℝ), i.1 ^ 2 + i.2 ^ 2) [(0:ℝ, 0:ℝ), (sqrt 3, 0)]) in
  let AB := sqrt (sum (λ (i : ℝ × ℝ), (i.1 - sqrt 3) ^ 2 + i.2 ^ 2) [(0:ℝ, l 0), (sqrt 3, l (sqrt 3))]) in
  (SABC = (λab op λ:ℝ, (λ ∆, 1/2 *ab * op = (λ |ab| λ:ℝ, |ab| = 4))/ (2 * op)) ∧
  (SABC = (|OP|^2 - (4/|AB|))

theorem proof_ellipse :
  ellipse_equation
:= sorry 

theorem proof_lambda :
  λ = -1
:= sorry 

end proof_ellipse_proof_lambda_l509_509641


namespace ants_meet_again_l509_509066

/-- Define the radius of the larger circle A --/
def radius_A : ℝ := 7

/-- Define the radius of the smaller circle B --/
def radius_B : ℝ := 3

/-- Define the speed of ant 1 on circle A --/
def speed_ant1 : ℝ := 4 * Real.pi

/-- Define the speed of ant 2 on circle B --/
def speed_ant2 : ℝ := 3 * Real.pi

/-- Define the circumference of circle A --/
def circumference_A : ℝ := 2 * radius_A * Real.pi

/-- Define the circumference of circle B --/
def circumference_B : ℝ := 2 * radius_B * Real.pi

/-- Define the time for ant 1 to make one full revolution --/
def time_one_revolution_ant1 : ℝ := circumference_A / speed_ant1

/-- Define the time for ant 2 to make one full revolution --/
def time_one_revolution_ant2 : ℝ := circumference_B / speed_ant2

/-- Define the least common multiple function for two real numbers --/
noncomputable def lcm (a b : ℝ) : ℝ :=
  let gcd := Nat.gcd in
  a * b / gcd a b

/-- Prove the ants will meet again at point P after 7 minutes --/
theorem ants_meet_again : lcm time_one_revolution_ant1 time_one_revolution_ant2 = 7 := 
sorry

end ants_meet_again_l509_509066


namespace sum_of_fifth_powers_l509_509221

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l509_509221


namespace range_of_a_l509_509271

def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

theorem range_of_a (a : ℝ) : (-2 / 3 : ℝ) ≤ a ∧ a < 0 := sorry

end range_of_a_l509_509271


namespace angle_bw_vectors_correct_l509_509625

open Real

variables {a b : ℝ^3} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : ∥ a + b ∥ = ∥ a - 2 • b ∥)

noncomputable def angle_between_vectors : ℝ :=
  Real.arccos ((∥ b ∥) / (2 * ∥ a ∥))

theorem angle_bw_vectors_correct : 
  angle_between_vectors h1 h2 h3 = Real.arccos ((∥ b ∥) / (2 * ∥ a ∥)) :=
sorry

end angle_bw_vectors_correct_l509_509625


namespace length_of_intervals_l509_509166

theorem length_of_intervals : 
  (2 - Real.pi / 4) = (finset.Icc 0 (Real.pi / 4)).measure -- the length of interval where max(sqrt(x/2), tan x) <= 1
:= sorry

end length_of_intervals_l509_509166


namespace each_person_owes_correct_amount_l509_509523

noncomputable def total_bill : ℝ := 257.36
noncomputable def tip_percent : ℝ := 0.12
noncomputable def num_people : ℕ := 15
noncomputable def tip : ℝ := tip_percent * total_bill
noncomputable def total_payment : ℝ := total_bill + tip
noncomputable def amount_per_person : ℝ := total_payment / num_people

theorem each_person_owes_correct_amount :
  Real.ceil (amount_per_person * 100) / 100 = 19.22 :=
by
  sorry

end each_person_owes_correct_amount_l509_509523


namespace log_sum_zero_l509_509508

theorem log_sum_zero {a b : ℝ} (h1 : a = 2 * real.log 10 / real.log 5) (h2 : b = real.log 0.25 / real.log 5) :
  a + b = 0 := by {
  sorry
}

end log_sum_zero_l509_509508


namespace traced_figure_is_asterisk_l509_509354

theorem traced_figure_is_asterisk (n : ℕ) (h : n ≥ 3) (r : ℝ) 
  (polygon : ℕ → ℂ) 
  (is_regular : ∀ i j, abs (polygon i - polygon j) = r * abs (complex.exp (2 * π * (i - j) / n * complex.I)))
  (A B : ℂ)
  (adj : ∃ i, polygon i = A ∧ polygon (i + 1 % n) = B)
  (O : ℂ)
  (centre : O = 0) : 
  figure_traced_by_O_is_asterisk n :=
sorry

end traced_figure_is_asterisk_l509_509354


namespace trains_arrived_in_30_minutes_l509_509705

theorem trains_arrived_in_30_minutes :
  ∃ d > (20 / 7) ∧ d ≤ 3,
  10 = floor (30 / d) + 1 :=
by
  let d₁ : ℝ := 20 / 7
  let d₂ : ℝ := 3
  have h₁ : d₁ < d₂ := by linarith
  use d₁
  split
  { norm_num }
  split
  { exact le_refl _ }
  sorry

end trains_arrived_in_30_minutes_l509_509705


namespace figure100_squares_l509_509034

/-- 
Problem Statement: 
Given the sequence of figures characterized by the number of non-overlapping unit squares 
for $n = 0, 1, 2, 3$ are $3, 9, 19, 33$, prove that the number of non-overlapping unit squares in figure 100 is 20403.
-/
def squares_in_figure : ℕ → ℕ
| 0     := 3
| 1     := 9
| 2     := 19
| 3     := 33
-- Define the proposed quadratic function
| (n+4) := 2 * (n + 4) * (n + 4) + 4 * (n + 4) + 3

theorem figure100_squares : squares_in_figure 100 = 20403 :=
by {
  sorry -- Proof is omitted as per instructions
}

end figure100_squares_l509_509034


namespace required_run_rate_is_correct_l509_509083

-- Define the initial conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_10 : ℝ := 10
def target_runs : ℝ := 282
def remaining_overs : ℝ := 40

-- Given total runs in the first 10 overs
def total_runs_first_10_overs : ℝ := run_rate_first_10_overs * overs_first_10
-- Given runs needed in the remaining 40 overs
def runs_needed_remaining_overs : ℝ := target_runs - total_runs_first_10_overs

-- Lean statement to prove the required run rate in the remaining 40 overs
theorem required_run_rate_is_correct (h1 : run_rate_first_10_overs = 3.2)
                                     (h2 : overs_first_10 = 10)
                                     (h3 : target_runs = 282)
                                     (h4 : remaining_overs = 40) :
  (runs_needed_remaining_overs / remaining_overs) = 6.25 :=
by sorry


end required_run_rate_is_correct_l509_509083


namespace total_amount_received_l509_509913

def initial_price_tv : ℕ := 500
def tv_increase_rate : ℚ := 2 / 5
def initial_price_phone : ℕ := 400
def phone_increase_rate : ℚ := 0.40

theorem total_amount_received :
  initial_price_tv + initial_price_tv * tv_increase_rate + initial_price_phone + initial_price_phone * phone_increase_rate = 1260 :=
by
  sorry

end total_amount_received_l509_509913


namespace sequence_non_integer_element_l509_509732

theorem sequence_non_integer_element (x : ℕ → ℕ → ℤ) (n : ℕ) (h1 : n > 2) (h2 : n % 2 = 1)
  (h3 : ¬ ∀ i j, i ≠ j → x i 1 = x j 1)
  (h4 : ∀ i k, i < n → x i (k + 1) = (x i k + x (i + 1) k) / 2)
  (h5 : ∀ k, x n (k + 1) = (x n k + x 1 k) / 2) :
  ∃ j k, ¬ isInt (x j k) :=
sorry

end sequence_non_integer_element_l509_509732


namespace max_page_number_with_25_threes_l509_509386

def digits_available : set ℕ := {0, 1, 2, 4, 5, 6, 7, 8, 9}

def count_digit_occurrences (digit : ℕ) (n : ℕ) : ℕ :=
  (n.to_string.to_list.filter (λ c, c = digit.digit_to_char)).length

theorem max_page_number_with_25_threes :
  ∀ {n : ℕ}, (∀ k ≤ n, count_digit_occurrences 3 k ≤ 25) → n ≤ 139 :=
sorry

end max_page_number_with_25_threes_l509_509386


namespace hyperbola_equation_l509_509267

theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c = 6) (h4 : b / a = sqrt 3) (h5 : a^2 + b^2 = c^2) :
  ∀ x y : ℝ, (x^2 / 9 - y^2 / 27 = 1) :=
by
  sorry

end hyperbola_equation_l509_509267


namespace number_of_integers_a_satisfies_conditions_l509_509450

theorem number_of_integers_a_satisfies_conditions : 
  let cond1 := ∀ x : ℝ, (3 * x - a)/(x - 3) + (x + 1)/(3 - x) = 1 → x > 0 → a = x + 2 ∧ x ≠ 3
      cond2 := ∀ y : ℝ, (y + 9 ≤ 2 * (y + 2)) ∧ ((2 * y - a) / 3 ≥ 1) → (y ≥ 5)
  in #[(a : ℤ) | a > 2 ∧ a ≠ 5 ∧ a ≤ 7].card = 4 := sorry

end number_of_integers_a_satisfies_conditions_l509_509450


namespace smallest_interval_for_probability_of_both_events_l509_509956

theorem smallest_interval_for_probability_of_both_events {C D : Prop} (hC : prob_C = 5 / 6) (hD : prob_D = 7 / 9) :
  ∃ I : set ℝ, I = set.Icc (11 / 18) (7 / 9) ∧ (∃ p : ℝ, p ∈ I ∧ p = prob_C_and_D) :=
begin
  sorry
end

end smallest_interval_for_probability_of_both_events_l509_509956


namespace sum_of_fifth_powers_l509_509222

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l509_509222


namespace weight_of_b_l509_509770

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45) 
  (h2 : (a + b) / 2 = 41) 
  (h3 : (b + c) / 2 = 43) 
  : b = 33 :=
by
  sorry

end weight_of_b_l509_509770


namespace snails_remaining_l509_509460

theorem snails_remaining 
  (original_snails : ℕ) (removed_snails : ℕ) 
  (h_original : original_snails = 11760) (h_removed : removed_snails = 3482) : 
  original_snails - removed_snails = 8278 :=
by
  rw [h_original, h_removed]
  norm_num

end snails_remaining_l509_509460


namespace domain_log_function_min_value_h_l509_509512

open Real

-- Problem (1)
theorem domain_log_function (m : ℝ) :
  (∀ x : ℝ, mx^2 + 2x + m > 0) ↔ m > 1 :=
sorry

-- Problem (2)
def h (a : ℝ) : ℝ :=
  if a < 1 / 3 then (28 - 6 * a) / 9
  else if 1 / 3 ≤ a ∧ a ≤ 3 then 3 - a^2
  else 12 - 6 * a

theorem min_value_h (a : ℝ) : ∀ x : ℝ, x ∈ Icc (-1 : ℝ) 1 → 
  let y = (1 / 3) ^ x in 
  (y^2 - 2 * a * y + 3) ≥ (h a) :=
sorry

end domain_log_function_min_value_h_l509_509512


namespace nell_cards_l509_509370

theorem nell_cards :
  ∀ (initial_cards given_cards : ℕ), initial_cards = 455 ∧ given_cards = 301 → 
  initial_cards - given_cards = 154 :=
by
  intros initial_cards given_cards
  intro h
  cases h with h1 h2
  rw [h1, h2]
  exact rfl

end nell_cards_l509_509370


namespace decimal_nearest_hundredth_l509_509420

noncomputable def calc_subtract_and_multiply (a b c : ℝ) : ℝ :=
  (a - b) * c

theorem decimal_nearest_hundredth (a b c : ℝ) (res : ℝ) :
  a = 456.78 → b = 234.56 → c = 1.5 → res = 333.33 →
  Float.roundTo (calc_subtract_and_multiply a b c) 2 = res :=
by
  intros ha hb hc hr
  rw [ha, hb, hc]
  -- The actual shooting would involve calculation, which we skip
  sorry

end decimal_nearest_hundredth_l509_509420


namespace darrel_quarters_l509_509152

-- Definitions for conditions given in the problem
def dimes := 85
def nickels := 20
def pennies := 150
def after_fee := 27
def fee_percentage := 0.10
def value_dime := 0.1
def value_nickel := 0.05
def value_penny := 0.01
def value_quarter := 0.25

-- Define total_amount before fee
noncomputable def total_amount := after_fee / (1 - fee_percentage)

-- Define the total value of dimes, nickels, and pennies
noncomputable def value_dimes := dimes * value_dime
noncomputable def value_nickels := nickels * value_nickel
noncomputable def value_pennies := pennies * value_penny

-- Define the total value of dimes, nickels, and pennies
noncomputable def total_value_other_coins := value_dimes + value_nickels + value_pennies

-- Define the value of quarters
noncomputable def value_quarters := total_amount - total_value_other_coins

-- The main theorem to prove
theorem darrel_quarters : value_quarters / value_quarter = 76 := by
  sorry

end darrel_quarters_l509_509152


namespace amount_invested_by_q_l509_509085

variable (P_investment : ℝ) (profit_ratio_p q_ratio : ℝ)

theorem amount_invested_by_q
  (h1 : profit_ratio_p = 3)
  (h2 : q_ratio = 4)
  (h3 : P_investment = 50000) :
  let Q_investment := (q_ratio / profit_ratio_p) * P_investment in
  Q_investment = 66666.67 := 
by
  let Q_investment := (q_ratio / profit_ratio_p) * P_investment
  sorry

end amount_invested_by_q_l509_509085


namespace no_such_rectangle_exists_l509_509161

theorem no_such_rectangle_exists :
  ¬(∃ (x y : ℝ), (∃ a b c d : ℕ, x = a + b * Real.sqrt 3 ∧ y = c + d * Real.sqrt 3) ∧ 
                (x * y = (3 * Real.sqrt 3) / 2 + n * (Real.sqrt 3 / 2))) :=
sorry

end no_such_rectangle_exists_l509_509161


namespace ellipse_centered_at_origin_with_focus_has_given_equation_l509_509208

noncomputable 
def ellipse_equation {C : Type*} (center : C) (right_focus : C) (intersection_x_coordinate : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
    ∀ (x y : ℝ), (x = 2 ∧ y = 2) → (x, y) ∈ set_of (λ (p : ℝ × ℝ), (p.1)^2 / a^2 + (p.2)^2 / b^2 = 1) ∧ 
    (center = (0, 0) ∧ right_focus = (sqrt 15, 0)) ∧ a^2 = 20 ∧ b^2 = 5

theorem ellipse_centered_at_origin_with_focus_has_given_equation : 
  ellipse_equation (0, 0) (sqrt 15, 0) 2 := 
begin
  sorry
end

end ellipse_centered_at_origin_with_focus_has_given_equation_l509_509208


namespace simplify_rationalize_l509_509412

theorem simplify_rationalize
  : (1 / (1 + (1 / (Real.sqrt 5 + 2)))) = ((Real.sqrt 5 + 1) / 4) := 
sorry

end simplify_rationalize_l509_509412


namespace length_of_intervals_l509_509169

noncomputable def totalLength : ℝ :=
  let interval1 := Icc 0 (Real.pi / 4)
  let interval2 := Icc (Real.pi / 4) 2
  interval2.length - interval1.length

theorem length_of_intervals : 
  totalLength = 1.21 :=
by
  -- lengths of individual intervals
  let length1 := (Real.pi / 4 : ℝ)
  let length2 := (2 - Real.pi / 4 : ℝ)
  have : totalLength = length2 - length1, by sorry
  rw this
  -- compute length in decimal form and rounding to nearest hundredth
  have : length2 = 1.21460183660255 := by sorry
  have : length1 = 0.785398163397448 := by sorry
  have computation := length2 - length1
  -- rounding to the nearest hundredth
  have : (Real.toEuclidean length2) - (Real.toEuclidean length1) = 1.21460183660255 - 0.785398163397448 := by sorry
  have rounding := (Real.round (1.21460183660255 - 0.785398163397448)) = 1.21 :=
by sorry
  have exact_len := 1.21
  exact exact_len

end length_of_intervals_l509_509169


namespace systematic_sampling_l509_509196

theorem systematic_sampling :
  ∀ (bottles : Finset ℕ), bottles = Finset.range (60 + 1) → 
  ∃ (selected : Finset ℕ), selected = {3, 13, 23, 33, 43, 53} ∧ 
  ∀ i : ℕ, i < 5 → selected.nth(i + 1) - selected.nth(i) = 10 :=
by
  -- Bottles are numbered from 1 to 60
  assume bottles,
  assume bottles_range,
  -- Define the set of bottles selected using systematic sampling with interval 10
  let selected := {3, 13, 23, 33, 43, 53},
  exists.intro selected,
  split,
  -- Show that the selected bottles set is as defined
  refl,
  -- Show that the differences between consecutive bottles are 10
  intros i hi,
  sorry

end systematic_sampling_l509_509196


namespace emmalyn_earnings_l509_509574

theorem emmalyn_earnings
  (rate_per_meter : ℚ := 0.20)
  (number_of_fences : ℚ := 50)
  (length_per_fence : ℚ := 500) :
  rate_per_meter * (number_of_fences * length_per_fence) = 5000 := by
  sorry

end emmalyn_earnings_l509_509574


namespace minimum_value_is_3_sqrt_2_over_2_l509_509995

noncomputable def minimum_value (a b c : ℝ) (h : 2 * a^2 - 5 * Real.log a - b = 0) :=
  Real.sqrt ((a - c)^2 + (b + c)^2)

theorem minimum_value_is_3_sqrt_2_over_2 (a b c : ℝ) (h : 2 * a^2 - 5 * Real.log a - b = 0)  :
  ∃ d : ℝ, d = Real.sqrt ((a - c)^2 + (b + c)^2) ∧ d = 3 * Real.sqrt 2 / 2 :=
begin
  use Real.sqrt ((a - c)^2 + (b + c)^2),
  split,
  { refl },
  { sorry }
end

end minimum_value_is_3_sqrt_2_over_2_l509_509995


namespace min_value_of_squares_l509_509349

-- Assumes a, b, c are distinct positive integers
variables {a b c n : ℕ}

-- Minimum value of a^2 + b^2 + c^2 given the defined conditions
theorem min_value_of_squares (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
    (h₄ : 0 < a) (h₅ : 0 < b) (h₆ : 0 < c)
    (h₇ : {a + b, b + c, c + a} = {n^2, (n + 1)^2, (n + 2)^2}) :
  a^2 + b^2 + c^2 = 1297 :=
sorry

end min_value_of_squares_l509_509349


namespace product_of_solutions_eq_neg_one_third_l509_509486

theorem product_of_solutions_eq_neg_one_third :
  (∏ x in {x : ℝ | x + 1 / x = 4 * x}, x) = -1 / 3 :=
by
  sorry

end product_of_solutions_eq_neg_one_third_l509_509486


namespace fresh_grapes_water_percentage_l509_509968

/--
Given:
- Fresh grapes contain a certain percentage (P%) of water by weight.
- Dried grapes contain 25% water by weight.
- The weight of dry grapes obtained from 200 kg of fresh grapes is 66.67 kg.

Prove:
- The percentage of water (P) in fresh grapes is 75%.
-/
theorem fresh_grapes_water_percentage
  (P : ℝ) (H1 : ∃ P, P / 100 * 200 = 0.75 * 66.67) :
  P = 75 :=
sorry

end fresh_grapes_water_percentage_l509_509968


namespace number_of_digits_2_15_times_5_10_l509_509158

theorem number_of_digits_2_15_times_5_10 : (2^15 * 5^10).natAbs.digits = 12 := by
  sorry

end number_of_digits_2_15_times_5_10_l509_509158


namespace construct_triangle_l509_509925

noncomputable theory
open_locale classical

variables {A B C : Type} [metric_space A] [has_dist A]
variables (a : ℝ) (α : ℝ) (s_b : ℝ)
variables (triangle_exists : A × A × A)

structure Triangle (A B C : Type*) :=
(v₁ : A)
(v₂ : A)
(v₃ : A)
(side_ac : dist v₁ v₃ = a)
(∠_bac : dist v₁ v₂ = α)
(median_sb : dist v₂ (midpoint ℝ v₁ v₃) = s_b)

theorem construct_triangle :
  ∃ (T : Triangle A B C), true :=
begin
  sorry
end

end construct_triangle_l509_509925


namespace number_of_integer_values_l509_509046

def star (a b : ℕ) := a^3 / b

theorem number_of_integer_values (x : ℕ) :
  ∃ n, n = (Set.filter (λ (d : ℕ), 512 % d = 0) (Finset.range 513)).card ∧ n = 10 := by
  sorry

end number_of_integer_values_l509_509046


namespace range_of_m_l509_509975

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 8 * x / y > m^2 + 2 * m)) → -4 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l509_509975


namespace anand_income_l509_509503

theorem anand_income (x y : ℕ)
  (income_A : ℕ := 5 * x)
  (income_B : ℕ := 4 * x)
  (expenditure_A : ℕ := 3 * y)
  (expenditure_B : ℕ := 2 * y)
  (savings_A : ℕ := 800)
  (savings_B : ℕ := 800)
  (hA : income_A - expenditure_A = savings_A)
  (hB : income_B - expenditure_B = savings_B) :
  income_A = 2000 := by
  sorry

end anand_income_l509_509503


namespace eval_expression_l509_509579

theorem eval_expression : (Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ)) = 0 := by
  sorry

end eval_expression_l509_509579


namespace compare_abc_l509_509970

theorem compare_abc (a b c : ℝ) (ha : a = 2^1.2) (hb : b = (1/2)^(-0.2)) (hc : c = 2 * log 2 / log 5) :
  c < b ∧ b < a := 
by 
  sorry

end compare_abc_l509_509970


namespace find_first_term_of_geometric_series_l509_509899

theorem find_first_term_of_geometric_series 
  (r : ℚ) (S : ℚ) (a : ℚ) 
  (hr : r = -1/3) (hS : S = 9)
  (h_sum_formula : S = a / (1 - r)) : 
  a = 12 := 
by
  sorry

end find_first_term_of_geometric_series_l509_509899


namespace everything_used_as_money_l509_509829

theorem everything_used_as_money :
  (used_as_money gold) ∧
  (used_as_money stones) ∧
  (used_as_money horses) ∧
  (used_as_money dried_fish) ∧
  (used_as_money mollusk_shells) →
  (∀ x ∈ {gold, stones, horses, dried_fish, mollusk_shells}, used_as_money x) :=
by
  intro h
  cases h with
  | intro h_gold h_stones =>
    cases h_stones with
    | intro h_stones h_horses =>
      cases h_horses with
      | intro h_horses h_dried_fish =>
        cases h_dried_fish with
        | intro h_dried_fish h_mollusk_shells =>
          intro x h_x
          cases Set.mem_def.mpr h_x with
          | or.inl h => exact h_gold
          | or.inr h_x1 => cases Set.mem_def.mpr h_x1 with
            | or.inl h => exact h_stones
            | or.inr h_x2 => cases Set.mem_def.mpr h_x2 with
              | or.inl h => exact h_horses
              | or.inr h_x3 => cases Set.mem_def.mpr h_x3 with
                | or.inl h => exact h_dried_fish
                | or.inr h_x4 => exact h_mollusk_shells

end everything_used_as_money_l509_509829


namespace largest_k_dividing_floor_l509_509183

theorem largest_k_dividing_floor (n : ℕ) : 
  ∃ k : ℕ+, (k = n ∧ 2 ^ k ∣ (⌊(3 + Real.sqrt 11)^(2 * n - 1)⌋ : ℤ)) :=
sorry

end largest_k_dividing_floor_l509_509183


namespace find_five_digit_number_l509_509940

theorem find_five_digit_number :
  ∃ (A B C D E : ℕ), A ∈ {1, 2, 3, 4, 5, 6} ∧ 
                     B ∈ {1, 2, 3, 4, 5, 6} ∧ 
                     C ∈ {1, 2, 3, 4, 5, 6} ∧ 
                     D ∈ {1, 2, 3, 4, 5, 6} ∧ 
                     E ∈ {1, 2, 3, 4, 5, 6} ∧ 
                     A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
                     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
                     C ≠ D ∧ C ≠ E ∧
                     D ≠ E ∧
                     A + B = 6 ∧ 
                     C + D = 6 ∧ 
                     1 = 4 ∧ 2 = 1 ∧ 4 = 2 ∧ 4 = 4 ∧
                     ∃ (overline_ABCDE : ℕ), overline_ABCDE = 41244 :=
by
  -- The proof is omitted
  sorry

end find_five_digit_number_l509_509940


namespace cyclic_and_constant_perimeter_l509_509871

variables (A B C D P Q R S : Type) [IncidenceGeometry A B C D] [HasReflectiveLightRay A B C D P Q R S]

def is_cyclic_quadrilateral (Q : Quadrilateral A B C D) := ∀ (P : Point) (L : LightRay P) (Q : Point) (R : Point) (S : Point),
  IsReflectiveLightRay P Q R S ∧ IsClosedLightPath P Q R S → IsCyclicQuad ABCD

def const_perimeter_pqrs (Q : Quadrilateral A B C D) := ∀ (P : Point) (L : LightRay P) (Q : Point) (R : Point) (S : Point),
  IsReflectiveLightRay P Q R S ∧ IsClosedLightPath P Q R S → 
  Perimeter PQRS = Constant ABDC

-- Theorem to prove the cyclic nature of quadrilateral and constant perimeter
theorem cyclic_and_constant_perimeter (Q : Quadrilateral A B C D) 
  (h1 : ∀ P : Point, Exists (Q : Point) (R : Point) (S : Point), 
  IsReflectiveLightRay P Q R S ∧ IsClosedLightPath P Q R S) :
  is_cyclic_quadrilateral Q A B C D ∧ const_perimeter_pqrs Q A B C D :=
begin
  sorry -- This is where the proof would go
end

end cyclic_and_constant_perimeter_l509_509871


namespace dan_gets_fewest_cookies_l509_509510

theorem dan_gets_fewest_cookies (D : ℝ) 
  (h_alice : 15 * (10 : ℝ) = D)
  (area_ben : 12)
  (area_cindy : 14)
  (area_dan : 16) : 
  ∀ (n_alice : ℝ) (n_ben : ℝ) (n_cindy : ℝ) (n_dan : ℝ),
  n_alice = D / 10 →
  n_ben = D / 12 →
  n_cindy = D / 14 →
  n_dan = D / 16 →
  n_dan < n_alice ∧ n_dan < n_ben ∧ n_dan < n_cindy :=
by intros; sorry

end dan_gets_fewest_cookies_l509_509510


namespace percent_of_geese_among_non_swans_l509_509708

theorem percent_of_geese_among_non_swans 
    (p_geese p_swans p_herons p_ducks : ℚ)  -- Percentages of bird types
    (h_sum : p_geese + p_swans + p_herons + p_ducks = 1) :  -- Sum of all percentages is 1 (100%)
    p_geese = 0.35 ∧ p_swans = 0.20 ∧ p_herons = 0.15 ∧ p_ducks = 0.30 →
    (p_geese / (1 - p_swans)) * 100 = 43.75 :=
by
  intros h
  cases h with h_geese h_swans_herons_ducks
  cases h_swans_herons_ducks with h_swans h_herons_ducks
  cases h_herons_ducks with h_herons h_ducks_sum
  sorry

end percent_of_geese_among_non_swans_l509_509708


namespace simplify_and_rationalize_denominator_l509_509400

noncomputable def problem (x : ℚ) := 
  x = 1 / (1 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_denominator (x : ℚ) (h: problem x) :
  x = (Real.sqrt 5 - 1) / 4 :=
by
  sorry

end simplify_and_rationalize_denominator_l509_509400


namespace total_toes_on_bus_l509_509378

-- Definitions based on conditions
def toes_per_hand_hoopit : Nat := 3
def hands_per_hoopit : Nat := 4
def number_of_hoopits_on_bus : Nat := 7

def toes_per_hand_neglart : Nat := 2
def hands_per_neglart : Nat := 5
def number_of_neglarts_on_bus : Nat := 8

-- We need to prove that the total number of toes on the bus is 164
theorem total_toes_on_bus :
  (toes_per_hand_hoopit * hands_per_hoopit * number_of_hoopits_on_bus) +
  (toes_per_hand_neglart * hands_per_neglart * number_of_neglarts_on_bus) = 164 :=
by sorry

end total_toes_on_bus_l509_509378


namespace arithmetic_sequence_a10_l509_509984

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (d : ℕ) 
  (h_seq : ∀ n, a (n + 1) = a n + d) 
  (h_positive : ∀ n, a n > 0) 
  (h_sum : a 1 + a 2 + a 3 = 15) 
  (h_geo : (a 1 + 2) * (a 3 + 13) = (a 2 + 5) * (a 2 + 5))  
  : a 10 = 21 := sorry

end arithmetic_sequence_a10_l509_509984


namespace overlapping_area_l509_509030

def area_of_overlap (g1 g2 : Grid) : ℝ :=
  -- Dummy implementation to ensure code compiles
  6.0

structure Grid :=
  (size : ℝ) (arrow_direction : Direction)

inductive Direction
| North
| West

theorem overlapping_area (g1 g2 : Grid) 
  (h1 : g1.size = 4) 
  (h2 : g2.size = 4) 
  (d1 : g1.arrow_direction = Direction.North) 
  (d2 : g2.arrow_direction = Direction.West) 
  : area_of_overlap g1 g2 = 6 :=
by
  sorry

end overlapping_area_l509_509030


namespace min_omega_value_l509_509646

theorem min_omega_value (ω : ℝ) (hω : ω > 0)
  (f : ℝ → ℝ)
  (hf_def : ∀ x, f x = Real.cos (ω * x - (Real.pi / 6))) :
  (∀ x, f x ≤ f (Real.pi / 4)) → ω = 2 / 3 :=
by
  sorry

end min_omega_value_l509_509646


namespace two_star_neg_five_eq_neg_one_neg_two_star_two_star_neg_three_eq_one_l509_509477

def star (a b : ℤ) : ℤ := a ^ 2 - b + a * b

theorem two_star_neg_five_eq_neg_one : star 2 (-5) = -1 := by
  sorry

theorem neg_two_star_two_star_neg_three_eq_one : star (-2) (star 2 (-3)) = 1 := by
  sorry

end two_star_neg_five_eq_neg_one_neg_two_star_two_star_neg_three_eq_one_l509_509477


namespace distance_from_pole_solution_set_of_inequality_l509_509849

-- Science Problem
theorem distance_from_pole (ρ θ : ℝ) (h_ρ : ρ = 3) (h_θ : θ = -4) : |ρ| = 3 := 
by
  rw [abs_of_nonneg]
  exact h_ρ
  sorry

-- Arts Problem
theorem solution_set_of_inequality : {x : ℝ | abs (2*x - 6) > x} = {x : ℝ | x < 2} ∪ {x : ℝ | 6 < x} :=
by
  ext x
  simp only [set.mem_set_of_eq, set.mem_union]
  sorry

end distance_from_pole_solution_set_of_inequality_l509_509849


namespace number_of_excellent_sequences_l509_509985

theorem number_of_excellent_sequences (n : ℕ) (hn : 2 ≤ n) :
  let f := {s : Fin (n + 1) → ℤ // 
    (∀ i, 0 ≤ i → i ≤ n → abs (s i) ≤ n) ∧
    (∀ i j, 0 ≤ i → i < j → j ≤ n → s i ≠ s j) ∧
    (∀ i j k, 0 ≤ i → i < j → j < k → k ≤ n → 
      max (int.abs ((s k) - (s i))) (int.abs ((s k) - (s j))) =
      (1/2 : ℚ) * (int.abs ((s i) - (s j)) + int.abs ((s j) - (s k)) + int.abs ((s k) - (s i))))} in
  fintype.card f = nat.choose (2 * n + 1) (n + 1) * 2 ^ n :=
by
  sorry

end number_of_excellent_sequences_l509_509985


namespace find_first_term_l509_509896

noncomputable def first_term_of_arithmetic_sequence : ℝ := -19.2

theorem find_first_term
  (a d : ℝ)
  (h1 : 50 * (2 * a + 99 * d) = 1050)
  (h2 : 50 * (2 * a + 199 * d) = 4050) :
  a = first_term_of_arithmetic_sequence :=
by
  -- Given conditions
  have h1' : 2 * a + 99 * d = 21 := by sorry
  have h2' : 2 * a + 199 * d = 81 := by sorry
  -- Solve for d
  have hd : d = 0.6 := by sorry
  -- Substitute d into h1'
  have h_subst : 2 * a + 99 * 0.6 = 21 := by sorry
  -- Solve for a
  have ha : a = -19.2 := by sorry
  exact ha

end find_first_term_l509_509896


namespace lines_intersect_l509_509862

-- Define the parameterizations of the two lines
def line1 (t : ℚ) : ℚ × ℚ := ⟨2 + 3 * t, 3 - 4 * t⟩
def line2 (u : ℚ) : ℚ × ℚ := ⟨4 + 5 * u, 1 + 3 * u⟩

theorem lines_intersect :
  ∃ t u : ℚ, line1 t = line2 u ∧ line1 t = ⟨26 / 11, 19 / 11⟩ :=
by
  sorry

end lines_intersect_l509_509862


namespace units_digit_same_units_and_tens_digit_same_l509_509089

theorem units_digit_same (n : ℕ) : 
  (∃ a : ℕ, a ∈ [0, 1, 5, 6] ∧ n % 10 = a ∧ n^2 % 10 = a) := 
sorry

theorem units_and_tens_digit_same (n : ℕ) : 
  n ∈ [0, 1, 25, 76] ↔ (n % 100 = n^2 % 100) := 
sorry

end units_digit_same_units_and_tens_digit_same_l509_509089


namespace remainder_of_N_mod_37_l509_509108

theorem remainder_of_N_mod_37 (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_of_N_mod_37_l509_509108


namespace selling_prices_correct_l509_509121

theorem selling_prices_correct :
  ∀ (cp1 cp2 cp3 sp1 sp2 sp3 : ℝ),
    (sp1 = 500 ∧ ((cp1 + 0.25 * cp1) = sp1)) →
    (cp2 = 800 ∧ sp2 = (cp2 - 0.25 * cp2)) →
    (cp3 = 1800 ∧ sp3 = (cp3 + 0.5 * cp3)) →
    (sp1 = 500 ∧ sp2 = 600 ∧ sp3 = 2700) :=
by
  intros cp1 cp2 cp3 sp1 sp2 sp3 h1 h2 h3
  cases h1 with h1_sp1 h1_cp1
  cases h2 with h2_cp2 h2_sp2
  cases h3 with h3_cp3 h3_sp3
  rw [←h1_sp1, h1_cp1] at h1_sp1
  rw [h2_cp2] at h2_sp2
  rw [h3_cp3, h3_sp3]
  split
  assumption
  split
  assumption
  have h2_sp200 : sp2 = 600 := by linarith
  have h3_sp2700 : sp3 = 2700 := by linarith
  split
  exact h2_sp200
  exact h3_sp2700

#check selling_prices_correct

end selling_prices_correct_l509_509121


namespace work_done_by_gas_l509_509381

theorem work_done_by_gas (n : ℕ) (R T0 Pa : ℝ) (V0 : ℝ) (W : ℝ) :
  -- Conditions
  n = 1 ∧
  R = 8.314 ∧
  T0 = 320 ∧
  Pa * V0 = n * R * T0 ∧
  -- Question Statement and Correct Answer
  W = Pa * V0 / 2 →
  W = 665 :=
by sorry

end work_done_by_gas_l509_509381


namespace number_of_valid_ns_l509_509952

def is_linear_factors (n : ℕ) : Prop :=
  ∃ a b : ℤ, (1 ≤ n ∧ n ≤ 5000) ∧ (x^2 - 3 * x - n = (x - a) * (x - b))

noncomputable def count_valid_ns : ℕ :=
  {n // is_linear_factors n}.card

theorem number_of_valid_ns : count_valid_ns = 67 := by
  sorry

end number_of_valid_ns_l509_509952


namespace num_valid_sequences_length_22_l509_509928

def valid_sequence (n : Nat) : Prop :=
  ∀ (s : List Bool), 
    s.length = n → 
    (s.head = false ∧ s.last = false) ∧
    ∀ i : Nat, i < n - 1 → ¬ (s.nth i = some false ∧ s.nth (i + 1) = some false) ∧
    ∀ i : Nat, i < n - 3 → ¬ (s.nth i = some true ∧ s.nth (i + 1) = some true ∧ 
                              s.nth (i + 2) = some true ∧ s.nth (i + 3) = some true) ∧
    1 < s.countp (λ x, x = true),
    
theorem num_valid_sequences_length_22 : 
  (f : Nat → Nat) → 
  (∀ n : Nat, f n = f (n - 4) + 2 * f (n - 5) + f (n - 6)) → 
  f 22 = 105 :=
by sorry

end num_valid_sequences_length_22_l509_509928


namespace time_switch_l509_509903

theorem time_switch (x y : ℝ) (T1 T2 : Time) 
    (h1 : 120 < x ∧ x < 180) 
    (h2 : 240 < y ∧ y < 300)
    (h3 : T1 = Time.mk 2 (y / 6))
    (h4 : T2 = Time.mk 4 (x / 6))
    (h5 : 12 * (x - 60) = y)
    (h6 : 2 * (y - 120) = x / 6) :
  T1 = Time.mk 2 20.986 ∧ T2 = Time.mk 4 11.750 :=
by 
    sorry

end time_switch_l509_509903


namespace square_side_length_l509_509487

theorem square_side_length (A : ℝ) (h : A = 625) : ∃ l : ℝ, l^2 = A ∧ l = 25 :=
by {
  sorry
}

end square_side_length_l509_509487


namespace min_ping_pong_balls_needed_l509_509302

theorem min_ping_pong_balls_needed :
  ∃ (balls : Fin 10 → ℤ), 
    (∀ i, balls i ≥ 11) ∧ 
    (∀ i, balls i ≠ 17) ∧ 
    (∀ i, balls i % 6 ≠ 0) ∧ 
    (∀ i j, i ≠ j → balls i ≠ balls j) ∧ 
    (∑ i, balls i = 174) :=
sorry

end min_ping_pong_balls_needed_l509_509302


namespace distance_from_A4_to_A1A2A3_l509_509845

noncomputable def distance_from_vertex_to_face
  (A1 A2 A3 A4 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_from_A4_to_A1A2A3 :
  let A1 := (2, 1, 4)
  let A2 := (-1, 5, -2)
  let A3 := (-7, -3, 2)
  let A4 := (-6, -3, 6)
  distance_from_vertex_to_face A1 A2 A3 A4 = 5 * real.sqrt (2 / 11) :=
by
  sorry

end distance_from_A4_to_A1A2A3_l509_509845


namespace percent_increase_l509_509517

variable (E : ℝ)

-- Given conditions
def enrollment_1992 := 1.20 * E
def enrollment_1993 := 1.26 * E

-- Theorem to prove
theorem percent_increase :
  ((enrollment_1993 E - enrollment_1992 E) / enrollment_1992 E) * 100 = 5 := by
  sorry

end percent_increase_l509_509517


namespace parabola_problem_statement_l509_509867

noncomputable def parabola_focus := (1, 0)
noncomputable def line_angle := Real.pi / 3
noncomputable def slope := Real.sqrt 3

-- Definitions of the distances involved
noncomputable def AF := 2
noncomputable def BF := 2 / 3
noncomputable def BC := 4

-- Definitions of λ1 and λ2
noncomputable def λ1 := AF / BF
noncomputable def λ2 := BC / BF

-- The statement to prove
theorem parabola_problem_statement :
  λ1 + λ2 = 9 :=
by
  sorry

end parabola_problem_statement_l509_509867


namespace total_arrangements_l509_509853

theorem total_arrangements :
  let students := 6
  let venueA := 1
  let venueB := 2
  let venueC := 3
  (students.choose venueA) * ((students - venueA).choose venueB) = 60 :=
by
  -- placeholder for the proof
  sorry

end total_arrangements_l509_509853


namespace sum_of_six_terms_l509_509608

variable {a : ℕ → ℝ} {q : ℝ}

/-- Given conditions:
* a is a decreasing geometric sequence with ratio q
-/
def is_decreasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem sum_of_six_terms
  (h_geo : is_decreasing_geometric_sequence a q)
  (h_decreasing : 0 < q ∧ q < 1)
  (h_a1 : 0 < a 1)
  (h_a1a3 : a 1 * a 3 = 1)
  (h_a2a4 : a 2 + a 4 = 5 / 4) :
  (a 1 * (1 - q^6) / (1 - q)) = 63 / 16 := by
  sorry

end sum_of_six_terms_l509_509608


namespace principal_amount_borrowed_l509_509500

theorem principal_amount_borrowed (R T A : ℝ) (hR : R = 6) (hT : T = 9) (hA : A = 8410) : ∃ P : ℝ, P = 5461 :=
by
  let P := 8410 / (1 + (6 * 9 / 100))
  use P
  have hP : P = 8410 / 1.54 := by
    sorry
  have hP_value : P = 5461 := by
    linarith
  exact hP_value

end principal_amount_borrowed_l509_509500


namespace regression_equation_is_correct_l509_509256

-- Defining the conditions
variable (x y : ℝ) (bar_x : ℝ := 3) (bar_y : ℝ := 3.5)
variable (pos_corr : x ≥ 0 → y ≥ 0)

-- Definition of the potential linear regression models
def option_A (x : ℝ) : ℝ := -2 * x + 9.5
def option_B (x : ℝ) : ℝ := -0.3 * x + 4.2
def option_C (x : ℝ) : ℝ := 0.4 * x + 2.3
def option_D (x : ℝ) : ℝ := 2 * x - 2.4

-- The target regression function that is to be proved as correct
def correct_option (x : ℝ) : ℝ := 0.4 * x + 2.3

-- The statement we need to prove
theorem regression_equation_is_correct :
  (pos_corr x y) →
  (option_C bar_x = bar_y) →
  correct_option x = option_C x :=
by 
  sorry

end regression_equation_is_correct_l509_509256


namespace lcm_problem_l509_509598

/-- 
Given the integer representations:
9^9 = 3^18,
8^8 = 2^24,
18^18 = 2^18 * 3^36,
k = 2^a * 3^b where (a, b) are integers,

Prove that the least common multiple (LCM) of 9^9, 8^8, and k 
is equal to 18^18 under the conditions:
Given that LCM(9^9, 8^8) = 2^24 * 3^18, 
determine the number of values for k.
-/
theorem lcm_problem (k : ℕ) (a b : ℕ) :
  k = 2^a * 3^b ∧ (0 ≤ a ∧ a ≤ 24) ∧ b = 36 →
  (nat.lcm (nat.lcm (9^9) (8^8)) k = 18^18) ∧ 
  (∃ n, 0 ≤ n ∧ n = 25) := sorry

end lcm_problem_l509_509598


namespace range_f_l509_509262

def f : ℝ → ℝ :=
  λ x, if x > 0 then x^2 + 1 else real.cos x

theorem range_f : set.range f = set.Icc (-1) (⊤) :=
  by sorry

end range_f_l509_509262


namespace range_of_a_l509_509272

variable (a : ℝ)
variable p : Prop
variable q : Prop

def prop_p := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x^2 + 1 ≥ a)
def prop_q := ∃ x : ℝ, x^2 + 2 * a * x + 1 = 0

theorem range_of_a : (¬ (¬ prop_p ∨ ¬ prop_q)) → (a ≤ -1 ∨ (1 ≤ a ∧ a ≤ 2)) :=
by
  sorry

end range_of_a_l509_509272


namespace floor_ceil_diff_l509_509242

theorem floor_ceil_diff (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) : ⌊x⌋ + x - ⌈x⌉ = x - 1 :=
sorry

end floor_ceil_diff_l509_509242


namespace isosceles_triangle_median_lines_l509_509991

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt (((p2.1 - p1.1) ^ 2) + ((p2.2 - p1.2) ^ 2))

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

noncomputable def median_line_equation (A B C : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let D := midpoint A C in
  let (x1, y1) := B in 
  let (x2, y2) := D in
  (y2 - y1, x1 - x2, x1 * (y2 - y1) - y1 * (x1 - x2))

theorem isosceles_triangle_median_lines :
  let A := (-4, 0)
  let B := (0, 3)
  ∃ C : ℝ × ℝ, let distA := distance A B in let distB := distance B C in
  (distA = distB ∧ slope A B * slope B C = -1) ∧
  (C = (3, -1) ∨ C = (-3, 7)) ∧
  ∃ eq1 eq2 : ℝ × ℝ × ℝ, eq1 = (7, -1, 3) ∧ eq2 = (1, 7, -21) ∧ (median_line_equation A B (3, -1) = eq1 ∨ median_line_equation A B (-3, 7) = eq2) := by
  sorry

end isosceles_triangle_median_lines_l509_509991


namespace log_inequality_solution_set_l509_509054

theorem log_inequality_solution_set :
  { x : ℝ | log10 (x - 1) < 1 ∧ x > 1 } = { x : ℝ | 1 < x ∧ x < 11 } :=
sorry

end log_inequality_solution_set_l509_509054


namespace count_three_digit_integers_l509_509658

theorem count_three_digit_integers : 
  let available_digits := {2, 3, 5, 5, 5, 7, 7, 7} in
  let count := 
    (∑ _ in (finset.powerset available_digits).filter (λ s, s.card = 3 ∧ ∀ d ∈ s, s.count d ≤ available_digits.count d), s.factorial) + 
    2 * 2 * 3 + 
    2 
  in
  count = 38 :=
by 
  sorry

end count_three_digit_integers_l509_509658


namespace arithmetic_proof_l509_509178

theorem arithmetic_proof : (28 + 48 / 69) * 69 = 1980 :=
by
  sorry

end arithmetic_proof_l509_509178


namespace angle_is_pi_over_2_l509_509990

noncomputable def angle_between_vectors {V : Type*} [inner_product_space ℝ V] (a b : V) :=
  real.arccos ((inner_product a b) / (∥a∥ * ∥b∥))

-- Given conditions as hypotheses
variables {V : Type*} [inner_product_space ℝ V]
variables {a b : V}
variable [H1 : (2 • a) ⋅ (2 • a - b) = b ⋅ (b - 2 • a)]
variable [H2 : ∥a - sqrt 2 • b∥ = 3 * ∥a∥]

-- The statement to be proven
theorem angle_is_pi_over_2 : angle_between_vectors a b = real.pi / 2 :=
sorry

end angle_is_pi_over_2_l509_509990


namespace imaginary_unit_real_part_eq_l509_509974

theorem imaginary_unit_real_part_eq (a : ℝ) (i : ℂ) (h : i * i = -1) :
  (∃ r : ℝ, ((3 + i) * (a + 2 * i) / (1 + i) = r)) → a = 4 :=
by
  sorry

end imaginary_unit_real_part_eq_l509_509974


namespace finitely_many_negative_terms_l509_509927

theorem finitely_many_negative_terms (A : ℝ) :
  (∀ (x : ℕ → ℝ), (∀ n, x n ≠ 0) ∧ (∀ n, x (n+1) = A - 1 / x n) →
  (∃ N, ∀ n ≥ N, x n ≥ 0)) ↔ A ≥ 2 :=
sorry

end finitely_many_negative_terms_l509_509927


namespace distinct_real_roots_l509_509716

def floor (x : ℝ) : ℤ := int.floor x

theorem distinct_real_roots : 
∃! R : set ℝ, has_size R 2 ∧ ∀ x ∈ R, ∃ n : ℤ, n = floor x ∧ x^2 - (n:ℝ) - 2 = 0 := by
  sorry

end distinct_real_roots_l509_509716


namespace power_function_value_at_2_l509_509253

def power_function : Type :=
∀ (x : ℝ), ∃ (α : ℝ), f(x) = x^α

theorem power_function_value_at_2 (f : ℝ → ℝ) (h : ∀ x, ∃ α, f(x) = x^α)
    (hf : f(3) = 1/9) : f(2) = 1/4 :=
by
  sorry

end power_function_value_at_2_l509_509253


namespace ratio_of_volumes_l509_509122

theorem ratio_of_volumes (s : ℝ) (h : s > 0) :
  let r := s * Real.sqrt 6 / 12,
      V_sphere := (4 / 3) * Real.pi * r^3,
      V_tetrahedron := (s^3 * Real.sqrt 2) / 12 
  in V_sphere / V_tetrahedron = Real.pi * Real.sqrt 3 / 18 := by
  sorry

end ratio_of_volumes_l509_509122


namespace largest_circle_area_rounded_to_nearest_int_l509_509531

theorem largest_circle_area_rounded_to_nearest_int
  (x : Real)
  (hx : 3 * x^2 = 180) :
  let r := (16 * Real.sqrt 15) / (2 * Real.pi)
  let area_of_circle := Real.pi * r^2
  round (area_of_circle) = 306 :=
by
  sorry

end largest_circle_area_rounded_to_nearest_int_l509_509531


namespace number_of_digits_in_expr_l509_509138

-- Number of digits in a number in decimal form
noncomputable def num_digits (n : ℕ) : ℕ :=
if n = 0 then 1 else ⌊log 10 n⌋.to_nat + 1

-- Definition of the expression in the problem
def expr : ℕ := 4 ^ 25 * 5 ^ 22

-- Main statement to prove
theorem number_of_digits_in_expr : num_digits expr = 31 := 
sorry

end number_of_digits_in_expr_l509_509138


namespace probability_of_sum_divisible_by_3_game_is_fair_l509_509837

-- Define the conditions of the problem.
noncomputable def is_divisible_by_3 (x y : ℕ) : Prop := (x + y) % 3 = 0
noncomputable def wang_wins (x y : ℕ) : Prop := (x + y) ≥ 10
noncomputable def li_wins (x y : ℕ) : Prop := (x + y) ≤ 4
noncomputable def fair_game (x y : ℕ) : Prop := 
  (probability_wang_wins = probability_li_wins)

-- Define the probability of an event.
noncomputable def probability_event (event : ℕ → ℕ → Prop) : ℝ :=
  (finset.card (finset.filter (λ pair, event pair.1 pair.2) 
                 (finset.product (finset.range 6) (finset.range 6)))) / 36

-- Define the probabilities of winning.
noncomputable def probability_wang_wins : ℝ := probability_event wang_wins
noncomputable def probability_li_wins : ℝ := probability_event li_wins
noncomputable def probability_divisible_by_3 : ℝ := probability_event is_divisible_by_3

-- Statements to be proven.
theorem probability_of_sum_divisible_by_3 : probability_divisible_by_3 = 1 / 3 := sorry
theorem game_is_fair : fair_game x y := sorry

end probability_of_sum_divisible_by_3_game_is_fair_l509_509837


namespace distance_ratio_l509_509111

variables (dw dr : ℝ)

theorem distance_ratio (h1 : 4 * (dw / 4) + 8 * (dr / 8) = 8)
  (h2 : dw + dr = 8)
  (h3 : (dw / 4) + (dr / 8) = 1.5) :
  dw / dr = 1 :=
by
  sorry

end distance_ratio_l509_509111


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l509_509229

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l509_509229


namespace prime_condition_l509_509702

noncomputable def isPrime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → coprime m n → prime m

theorem prime_condition {n : ℕ} : 
  (∀ n > 1, ∃ p : ℕ, p > n ∧ p < 2 * n ∧ prime p) → 
  (isPrime n ↔ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 30) :=
by 
  sorry

end prime_condition_l509_509702


namespace xyz_sqrt_eq_10_sqrt_70_l509_509357

theorem xyz_sqrt_eq_10_sqrt_70
  (x y z : ℝ)
  (h1 : y + z = 15)
  (h2 : z + x = 18)
  (h3 : x + y = 17) :
  real.sqrt (x * y * z * (x + y + z)) = 10 * real.sqrt 70 :=
begin
  sorry
end

end xyz_sqrt_eq_10_sqrt_70_l509_509357


namespace fifth_powers_sum_eq_l509_509215

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l509_509215


namespace number_of_true_statements_l509_509193

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

def no_real_roots (a b c : ℝ) : Prop := (b - 1) ^ 2 - 4 * a * c < 0

def statement_1 (a b c : ℝ) : Prop := ∀ x : ℝ, f a b c (f a b c x) ≠ x

def statement_2 (a b c : ℝ) : Prop := a > 0 → ∀ x : ℝ, f a b c (f a b c x) ≥ 0

def statement_3 (a b c : ℝ) : Prop := a < 0 → ∃ x : ℝ, f a b c (f a b c x) > x

def statement_4 (a b c : ℝ) : Prop := a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x

theorem number_of_true_statements (a b c : ℝ) (h : no_real_roots a b c) :
  (if statement_1 a b c then 1 else 0) +
  (if statement_2 a b c then 1 else 0) +
  (if statement_3 a b c then 1 else 0) +
  (if statement_4 a b c then 1 else 0) = 3 :=
by sorry

end number_of_true_statements_l509_509193


namespace side_length_of_rhombus_l509_509058

theorem side_length_of_rhombus (L S m : ℝ) :
  (∃ (a b : ℝ), a + b = L / 2 ∧ 2 * a * b = S) →
  m = (sqrt (L^2 - 4 * S)) / 2 :=
by
  sorry

end side_length_of_rhombus_l509_509058


namespace not_a_term_of_sequence_l509_509780

-- Define the sequence as described in conditions
def sequence : ℕ → ℕ
| 0       := 1
| 1       := 2
| (n + 2) := (Finset.range (n + 2)).sum sequence

-- Prove that 72 is not in the image of the sequence function
theorem not_a_term_of_sequence : ∀ n, sequence n ≠ 72 :=
sorry

end not_a_term_of_sequence_l509_509780


namespace fraction_evaluation_l509_509592

theorem fraction_evaluation (x z : ℚ) (hx : x = 4/7) (hz : z = 8/11) :
  (7 * x + 10 * z) / (56 * x * z) = 31 / 176 := by
  sorry

end fraction_evaluation_l509_509592


namespace LA_perpendicular_LH_l509_509712

variables {A B C H P E F G T M L : Type} [PlaneGeometry A B C H P E F G T M L]

-- Define the orthocenter H of triangle ABC
axiom orthocenter_H (A B C : Type) : Point H

-- Let P be any point in the plane of the triangle
axiom any_point_P (A B C : Type) : Point P

-- Define the circle Ω with diameter AP
axiom circle_Omega (A P : Type) : Circle Ω (diameter A P)

-- Define the points E and F
axiom intersects_CA_AB (Ω : Circle) (CA AB : Type) : Point E F

-- Define the second intersection of PH with Ω as G
axiom second_intersection (P H : Type) (Ω : Circle) : Point G

-- Define the tangents at E and F intersecting at T
axiom tangents_intersect (Ω : Circle) (E F : Type) : Point T

-- Define M as the midpoint of BC
axiom midpoint_M (B C : Type) : Point M

-- Define L on MG such that AL is parallel to MT
axiom L_on_MG (M G A T : Type) (parallel : Line A L = Line M T) : Point L

-- Prove that LA is perpendicular to LH
theorem LA_perpendicular_LH (A L H : Type) [Orthogonal Line L A Line L H] : Prop :=
begin
  sorry
end

end LA_perpendicular_LH_l509_509712


namespace problem1_problem2_l509_509848

-- Given conditions
variables {a m n : ℝ}
axiom log_a_2_eq_m : log a 2 = m
axiom log_a_3_eq_n : log a 3 = n

-- First problem to solve
theorem problem1 : a^(2 * m + n) = 12 :=
by
  sorry

-- Given conditions for second problem
axiom log4_9 : log 4 9 = log 2 3
axiom log2_12 : log 2 12 = log 2 (4 * 3)
axiom lg_5_2 : log 10 (5/2) = -log 10 2.5

-- Second problem to solve
theorem problem2 : log 4 9 - log 2 12 + 10^(-log 10 (5/2)) = -(8 / 5) :=
by
  sorry

end problem1_problem2_l509_509848


namespace triangle_congruence_SAS_l509_509894

-- Define the condition that two triangles have the appropriate sides and included angle equal.
def congruent_by_SAS (AB BC AC DE EF DF : ℝ) (A D C F : ℝ)
  (h1 : BC = EF)
  (h2 : AC = DF)
  (h3 : C = F) : Prop :=
  △ (mk_triangle AB BC AC) ≃ △ (mk_triangle DE EF DF)

theorem triangle_congruence_SAS :
  ∀ (AB BC AC DE EF DF : ℝ) (A D C F : ℝ),
    BC = EF →
    AC = DF →
    C = F →
    congruent_by_SAS AB BC AC DE EF DF A D C F := 
by
  intros
  unfold congruent_by_SAS
  sorry

end triangle_congruence_SAS_l509_509894


namespace min_value_of_expression_l509_509343

theorem min_value_of_expression (a b c : ℤ) (h1 : ¬ (c = b + 1 ∨ c = b - 1))
  (ω : ℂ) (h2 : ω^4 = 1) (h3 : ω ≠ 1) :
  ∃ u : ℝ, (u = |a + b * ω + c * (ω^3)|) ∧ u = 2 :=
by
  have ω_sqr : ω^2 = -1, from 
    calc 
    ω^4 = 1 : h2
    ... - 1 + ω^2 = 0 : by sorry,
    
  have ω_cub : ω^3 = -ω, from 
    calc 
    ω^3 = ω * ω^2 : by ring
    ... = ω * (-1) : by rw [ω_sqr]
    ... = -ω : by ring,

  sorry

end min_value_of_expression_l509_509343


namespace tan_sum_eq_one_l509_509094

theorem tan_sum_eq_one (tan22 tan23 : ℝ) (h22 : tan 22 = tan22) (h23 : tan 23 = tan23) :
  tan22 + tan23 + tan22 * tan23 = 1 := 
by
  sorry

end tan_sum_eq_one_l509_509094


namespace crates_in_load_l509_509876

variable (c : ℕ)                                   -- Number of crates
variable (weight_crate : ℕ) (weight_carton : ℕ)     -- Weights of crates and cartons
variable (num_cartons : ℕ) (total_weight : ℕ)       -- Number of cartons and total weight of the load

-- Conditions
def crate_weighs_4 : Prop := weight_crate = 4
def carton_weighs_3 : Prop := weight_carton = 3
def load_weights_96 : Prop := total_weight = 96
def cartons_count_16 : Prop := num_cartons = 16
def total_weight_condition : Prop := total_weight = c * weight_crate + num_cartons * weight_carton

-- Proof statement
theorem crates_in_load :
  crate_weighs_4 → 
  carton_weighs_3 → 
  load_weights_96 → 
  cartons_count_16 → 
  total_weight_condition → 
  c = 12 :=
by
  intros
  sorry

end crates_in_load_l509_509876


namespace emmalyn_earnings_l509_509576

theorem emmalyn_earnings :
  let rate_per_meter := 0.20
  let number_of_fences := 50
  let length_of_each_fence := 500
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * rate_per_meter
  total_income = 5000 :=
by
  sorry

end emmalyn_earnings_l509_509576


namespace tan_product_l509_509047

theorem tan_product (P Q R D : Type) 
(h1 : HD = 8) 
(h2 : HQ = 20) 
(h3 : ∃ q : ℝ, PD = q * cos P) 
(h4 : ∃ R : ℝ, q = 2 * R * sin Q)
(h5 : ∠PHD = Q) 
: tan P * tan Q = 7 / 2 :=
sorry

end tan_product_l509_509047


namespace no_interval_satisfies_inequality_l509_509730

theorem no_interval_satisfies_inequality (a : ℝ) (h : 0 < a) :
  ¬ ∃ (b c : ℝ), b < c ∧ ∀ (x y : ℝ), b < x ∧ x < c ∧ b < y ∧ y < c ∧ x ≠ y → abs ((x + y) / (x - y)) ≤ a := 
begin
  sorry
end

end no_interval_satisfies_inequality_l509_509730


namespace triangle_ABC_area_l509_509587

-- Define the triangle ABC with given conditions
structure Triangle :=
  (A B C : Point)
  (AB BC AC : ℝ)
  (angleBAC : ℝ)
  (angleBCA : ℝ)

noncomputable def is_isosceles_right_triangle (T : Triangle) : Prop :=
  T.angleBAC = π / 4 ∧ T.angleBCA = π / 4 ∧ T.AC = 1

noncomputable def area_of_triangle (T : Triangle) : ℝ :=
  1 / 2 * T.AB * T.BC

theorem triangle_ABC_area {T : Triangle}
  (hT : is_isosceles_right_triangle T)
  (h1 : T.AB = T.BC)
  (hx : T.AB = (1 / sqrt 2)) :
  area_of_triangle T = 1 / 4 :=
sorry

end triangle_ABC_area_l509_509587


namespace find_m_l509_509861

theorem find_m (m : ℝ) :
    (∃ m, (2, 9), (10, m), and (25, 4) are collinear) →
    m = 167 / 23 :=
by
  -- Define the condition for collinearity: equality of slopes
  intro h
  have slope12 := (m - 9) / 8
  have slope23 := (4 - m) / 15
  have h_eq := slope12 = slope23
  sorry

end find_m_l509_509861


namespace abc_sum_l509_509514

def f (x : Int) (a b c : Nat) : Int :=
  if x > 0 then a * x + 4
  else if x = 0 then a * b
  else b * x + c

theorem abc_sum :
  ∃ a b c : Nat, 
  f 3 a b c = 7 ∧ 
  f 0 a b c = 6 ∧ 
  f (-3) a b c = -15 ∧ 
  a + b + c = 10 :=
by
  sorry

end abc_sum_l509_509514


namespace remainder_when_divided_l509_509105

theorem remainder_when_divided (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_when_divided_l509_509105


namespace unique_solution_for_lines_intersection_l509_509079

theorem unique_solution_for_lines_intersection (n : ℕ) (h : n * (n - 1) / 2 = 2) : n = 2 :=
by
  sorry

end unique_solution_for_lines_intersection_l509_509079


namespace circle_diameter_other_endpoint_l509_509919

theorem circle_diameter_other_endpoint
  (C : ℝ × ℝ) (A : ℝ × ℝ)
  (hC : C = (3, 4))
  (hA : A = (1, -2)) :
  ∃ B : ℝ × ℝ, B = (5, 10) := 
by 
  let x_disp := (3 - 1 : ℝ)
  let y_disp := (4 - (-2) : ℝ)
  use (3 + x_disp, 4 + y_disp)
  have hx_disp : x_disp = 2 := by norm_num
  have hy_disp : y_disp = 6 := by norm_num
  rw [hx_disp, hy_disp]
  norm_num
  sorry

end circle_diameter_other_endpoint_l509_509919


namespace greatest_increase_bobbleheads_l509_509044

theorem greatest_increase_bobbleheads :
  let sales := [20, 35, 40, 38, 60, 75],
      increases := [sales[1] - sales[0], sales[2] - sales[1], sales[3] - sales[2], sales[4] - sales[3], sales[5] - sales[4]]
  in increases.lookup 3 = 22 ->
  ∀ i : Nat, i ≠ 3 → increases.i ≤ increases.lookup 3 := 
sorry

end greatest_increase_bobbleheads_l509_509044


namespace radius_of_larger_circle_l509_509601

noncomputable theory

-- Define the problem conditions
def fourSmallCircles (radius : ℝ) := radius = 1

def externallyTangent (r : ℝ) := r = 1

def internallyTangent (bigR smallR : ℝ) := bigR = smallR + (smallR * Real.sqrt 2)

-- The theorem to be proven
theorem radius_of_larger_circle (R : ℝ) (r : ℝ) (four_small : fourSmallCircles r) 
  (ext_tangent : externallyTangent r) (int_tangent : internallyTangent R r) :
  R = 1 + Real.sqrt 2 := by
  sorry

end radius_of_larger_circle_l509_509601


namespace perimeter_triangle_ABF2_l509_509519

section
-- Define the hyperbola and its properties
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1

-- Define the length of the chord AB
def chord_length_AB : ℝ := 6

-- Define the left and right foci
def F1 : (ℝ × ℝ) := (-sqrt(25), 0)  -- Since c^2 = a^2 + b^2 => c = 5, F1 at (-5, 0)
def F2 : (ℝ × ℝ) := (sqrt(25), 0)   -- F2 at (5, 0)

-- Define the key property of the triangle ABF2
theorem perimeter_triangle_ABF2 
  (A B : (ℝ × ℝ)) 
  (hA : hyperbola A.1 A.2)
  (hB : hyperbola B.1 B.2)
  (hChord : (A.1 - B.1)^2 + (A.2 - B.2)^2 = chord_length_AB^2)
  (hA_F1 : A = F1 ∨ B = F1) : 
  A ≠ B ∧ (: (ℝ × ℝ) := 28 := 
  sorry
end

end perimeter_triangle_ABF2_l509_509519


namespace inradius_inequality_l509_509884

-- Definitions for the problem
variables (A B C O : Type) [PointInCircle A B C O]
variables (R r1 r2 r3 : ℝ)

-- Definition of the inradius
class Inradius (Δ : Type) :=
(inradius : Type)

-- Assume triangle ABC
instance triangle_ABC (A B C O : Type) [PointInCircle A B C O] : Inradius (Triangle O B C) := ⟨r1⟩
instance triangle_OBC (A B C O : Type) [PointInCircle A B C O] : Inradius (Triangle O C A) := ⟨r2⟩
instance triangle_OCA (A B C O : Type) [PointInCircle A B C O] : Inradius (Triangle O A B) := ⟨r3⟩

-- Statement of the theorem
theorem inradius_inequality {A B C O : Type} [PointInCircle A B C O] (R : ℝ) (r1 r2 r3 : ℝ)
  [Inradius (Triangle O B C)] [Inradius (Triangle O C A)] [Inradius (Triangle O A B)] :
  (1 / r1) + (1 / r2) + (1 / r3) ≥ (4 * Real.sqrt 3 + 6) / R :=
sorry

end inradius_inequality_l509_509884


namespace count_squares_below_graph_l509_509038

theorem count_squares_below_graph (x y: ℕ) (h_eq : 12 * x + 180 * y = 2160) (h_first_quadrant : x ≥ 0 ∧ y ≥ 0) :
  let total_squares := 180 * 12
  let diagonal_squares := 191
  let below_squares := total_squares - diagonal_squares
  below_squares = 1969 :=
by
  sorry

end count_squares_below_graph_l509_509038


namespace angle_E_is_120_l509_509543

-- Define the conditions in the problem
variables (A B C D E : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq E]
variables (AB BC CD DE EA : ℝ)
variables (angle_A angle_B : ℝ)
variables (is_convex : Prop)

-- Given known values for angles and equal side lengths
def conditions : Prop :=
  is_convex ∧ (AB = BC) ∧ (BC = CD) ∧ (CD = DE) ∧ (DE = EA) ∧ (EA = AB) ∧ (angle_A = 100) ∧ (angle_B = 80)

-- The proof goal based on these conditions
theorem angle_E_is_120 (h : conditions A B C D E AB BC CD DE EA angle_A angle_B is_convex) : 
  let sum_interior_angles := 540 in
  let remaining_angles_sum := sum_interior_angles - (angle_A + angle_B) in
  let angle_CD_sum := 240 in
  let angle_C_eq_D := (remaining_angles_sum - angle_CD_sum) in
  (angle_C_eq_D / 2) = 120 :=
by sorry

end angle_E_is_120_l509_509543


namespace calculator_cost_l509_509099

theorem calculator_cost (price_per_unit : ℝ) (x : ℝ) (h_x : x > 20) :
  let discount := 0.7
  let y := discount * price_per_unit * (x - 20) + price_per_unit * 20
  y = 0.7 * 80 * (x - 20) + 80 * 20 :=
by
  let discount := 0.7
  let price_per_unit := 80
  have h_price_per_unit : price_per_unit = 80 := rfl
  have h_discount : discount = 0.7 := rfl
  have h20 : (20:ℝ) = 20 := rfl
  let y := discount * price_per_unit * (x - 20) + price_per_unit * 20
  have h := calc
    y = discount * price_per_unit * (x - 20) + price_per_unit * 20 : by rw [h20]
    ... = 0.7 * 80 * (x - 20) + 80 * 20 : by rw [h_discount, h_price_per_unit]
  exact h

end calculator_cost_l509_509099


namespace pyramid_coloring_l509_509986

theorem pyramid_coloring (n m : ℕ) (hn : n ≥ 3) (hm : m ≥ 4) :
  ∃ N : ℕ, N = m * (m - 1) * (m - 2) * ((m - 2)^(n - 1) + (-1)^n) :=
  sorry

end pyramid_coloring_l509_509986


namespace fifth_powers_sum_eq_l509_509217

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l509_509217


namespace exact_value_range_l509_509792

theorem exact_value_range (a : ℝ) (h : |170 - a| < 0.5) : 169.5 ≤ a ∧ a < 170.5 :=
by
  sorry

end exact_value_range_l509_509792


namespace num_of_valid_subsets_l509_509950

def is_multiple_of_7 (n : ℕ) : Prop := n % 7 = 0

def num_valid_subsets : ℕ :=
  let A := {1, 2, 3, ..., 100} in
  let subsets := {(a, b) | a ∈ A ∧ b ∈ A ∧ a < b ∧ is_multiple_of_7 (a * b + a + b)} in
  subsets.card

theorem num_of_valid_subsets : num_valid_subsets = 602 := by
  sorry

end num_of_valid_subsets_l509_509950


namespace train_passing_bridge_time_l509_509538

theorem train_passing_bridge_time :
  ∀ (length_train : ℝ) (speed_train_kmh : ℝ) (length_bridge : ℝ),
     length_train = 320 ∧ speed_train_kmh = 45 ∧ length_bridge = 140 →
     let total_distance := length_train + length_bridge in
     let speed_train_ms := speed_train_kmh * (1000 / 3600) in
     let time_to_pass_bridge := total_distance / speed_train_ms in
     time_to_pass_bridge = 36.8 :=
by
  intros length_train speed_train_kmh length_bridge h,
  cases h with h_train h1,
  cases h1 with h_speed h_bridge,
  simp [h_train, h_speed, h_bridge],
  let total_distance := 320 + 140,
  let speed_train_ms := 45 * (1000 / 3600),
  let time_to_pass_bridge := total_distance / speed_train_ms,
  have : total_distance = 460 := by norm_num,
  have : speed_train_ms = 12.5 := by norm_num,
  have : time_to_pass_bridge = 460 / 12.5 := by simp [this, this_1],
  norm_num at this,
  exact this

end train_passing_bridge_time_l509_509538


namespace brick_height_l509_509591

def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem brick_height 
  (l : ℝ) (w : ℝ) (SA : ℝ) (h : ℝ) 
  (surface_area_eq : surface_area l w h = SA)
  (length_eq : l = 10)
  (width_eq : w = 4)
  (surface_area_given : SA = 164) :
  h = 3 :=
by
  sorry

end brick_height_l509_509591


namespace intersection_divides_AC_l509_509694

open Real

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨2, 1⟩
def C : Point := ⟨3, -3⟩
def D : Point := ⟨0, 0⟩

def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

def intersection_of_diagonals (A C B D : Point) : Point := sorry

theorem intersection_divides_AC (A B C D : Point)
  (h_convex : is_convex_quadrilateral A B C D)
  (M := intersection_of_diagonals A C B D) :
  ∃ (k l : ℝ), k / l = 1 / 3 ∧ (k + l = dist A C) :=
sorry

end intersection_divides_AC_l509_509694


namespace fraction_spent_on_travel_l509_509869

theorem fraction_spent_on_travel 
  (initial_amount : ℝ)
  (after_spending : ℝ)
  (spent_on_clothes : ℝ := 1/3 * initial_amount)
  (remaining_after_clothes : ℝ := initial_amount - spent_on_clothes)
  (spent_on_food : ℝ := 1/5 * remaining_after_clothes)
  (remaining_after_food : ℝ := remaining_after_clothes - spent_on_food)
  (spent_on_travel : ℝ := remaining_after_food - after_spending) :
  initial_amount = 1249.9999999999998 →
  after_spending = 500 →
  spent_on_travel / remaining_after_food ≈ 1/4 :=
by
  intros
  sorry

end fraction_spent_on_travel_l509_509869


namespace concyclic_feet_midpoints_l509_509394

theorem concyclic_feet_midpoints {A B C H A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ : Point} 
    (H_orthocenter : is_orthocenter H A B C)
    (H_feet_altitude_A : is_foot_of_altitude A₁ A B C)
    (H_feet_altitude_B : is_foot_of_altitude B₁ B A C)
    (H_feet_altitude_C : is_foot_of_altitude C₁ C A B)
    (H_midpoint_A2 : is_midpoint A₂ B C)
    (H_midpoint_B2 : is_midpoint B₂ C A)
    (H_midpoint_C2 : is_midpoint C₂ A B)
    (H_midpoint_A3 : is_midpoint A₃ A H)
    (H_midpoint_B3 : is_midpoint B₃ B H)
    (H_midpoint_C3 : is_midpoint C₃ C H):
  are_concyclic {A₁, B₁, C₁, A₂, B₂, C₂, A₃, B₃, C₃} :=
sorry

end concyclic_feet_midpoints_l509_509394


namespace find_number_l509_509464

theorem find_number :
  ∃ x : ℤ, 3 * (x + 2) = 24 + x ∧ x = 9 :=
by
  use 9
  split
  · calc
      3 * (9 + 2) = 3 * 11 : by rw [Int.add_comm, Int.add_left_comm]
      _ = 33 : by norm_num
      24 + 9 = 33 : by norm_num
  · rfl

end find_number_l509_509464


namespace three_distinct_real_roots_l509_509600

theorem three_distinct_real_roots (c : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * |x + 1 / 4| + c = 0) → x ∈ ℝ) →
  (count_roots (λ x => x^2 - 2 * |x + 1 / 4| + c) = 3) ↔ (c = 1 / 2 ∨ c = -1 / 16) :=
by
-- We'll add sorry just to skip the proof.
sorry

end three_distinct_real_roots_l509_509600


namespace ratio_of_sums_of_arithmetic_sequences_l509_509163

theorem ratio_of_sums_of_arithmetic_sequences :
  let a₁ := 4, d₁ := 4, l₁ := 68,
      a₂ := 5, d₂ := 5, l₂ := 85,
      n₁ := (l₁ - a₁) / d₁ + 1,
      n₂ := (l₂ - a₂) / d₂ + 1,
      S₁ := (n₁ / 2) * (a₁ + l₁),
      S₂ := (n₂ / 2) * (a₂ + l₂) in
  S₁ / S₂ = 4 / 5 :=
by
  sorry

end ratio_of_sums_of_arithmetic_sequences_l509_509163


namespace cosine_angle_is_16_over_65_l509_509280

noncomputable def cosine_angle_between_vectors : ℝ :=
  let a : ℝ × ℝ := (4, 3),
      b : ℝ × ℝ := (-5, 12),
      dot_product := a.1 * b.1 + a.2 * b.2,
      norm_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2),
      norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  in dot_product / (norm_a * norm_b)

theorem cosine_angle_is_16_over_65 :
  let a : ℝ × ℝ := (4, 3),
      b : ℝ × ℝ := (-5, 12)
  in (2 * a.1 + b.1 = 3) ∧ (2 * a.2 + b.2 = 18) →
     cosine_angle_between_vectors = 16 / 65 :=
by
  -- proof would be written here
  sorry

end cosine_angle_is_16_over_65_l509_509280


namespace sales_at_new_prices_l509_509887

variable (p_b c_b k_b p_m c_m k_m : ℝ)

theorem sales_at_new_prices 
  (inv_prop_blender : p_b * c_b = 4500) 
  (inv_prop_microwave : p_m * c_m = 10000) 
  (new_cb : c_b = 450) 
  (new_cm : c_m = 500) :
  (p_b = 4500 / 450) ∧ (p_m = 10000 / 500) :=
by
  have pb_new := 4500 / 450
  have pm_new := 10000 / 500
  simp [pb_new, pm_new]
  split;
  sorry

end sales_at_new_prices_l509_509887


namespace probability_odd_sum_given_even_product_l509_509966

def roll_dice_and_check (dices : Fin 6 → ℕ) : Prop :=
(dices 0 + dices 1 + dices 2 + dices 3) % 2 = 1

def even_product (dices : Fin 6 → ℕ) :=
∃ i : Fin 6, dices i % 2 = 0

theorem probability_odd_sum_given_even_product :
  ∃ (dices : Fin 6 → ℕ), even_product dices →
  (8 / 15 = if roll_dice_and_check dices then 1 else 0).to_real :=
begin
  sorry
end

end probability_odd_sum_given_even_product_l509_509966


namespace mark_more_than_kate_l509_509501

section

variable (K P M : ℕ)

-- Definitions based on problem conditions
def total_hours : Prop := P + K + M = 144
def pat_to_kate : Prop := P = 2 * K
def pat_to_mark : Prop := P = M / 3

-- The final statement to prove
theorem mark_more_than_kate 
  (total_hours : total_hours)
  (pat_to_kate : pat_to_kate)
  (pat_to_mark : pat_to_mark) : M - K = 80 :=
sorry

end

end mark_more_than_kate_l509_509501


namespace sum_of_elements_zero_l509_509348

theorem sum_of_elements_zero (M : Set ℝ) (m : ℕ) (hm : 2 < m)
  (hM : ∃ a : ℕ → ℝ, (∀ i < m, a i ∈ M) ∧ 
                      (∀ i < m, |a i| ≥ |(∑ j in finset.range m, a j) - a i|)) :
  (∑ i in finset.range m, Classical.some hM i) = 0 := 
sorry

end sum_of_elements_zero_l509_509348


namespace emmalyn_earnings_l509_509573

theorem emmalyn_earnings
  (rate_per_meter : ℚ := 0.20)
  (number_of_fences : ℚ := 50)
  (length_per_fence : ℚ := 500) :
  rate_per_meter * (number_of_fences * length_per_fence) = 5000 := by
  sorry

end emmalyn_earnings_l509_509573


namespace scientific_notation_l509_509750

theorem scientific_notation (x : ℝ) (h : x = 70819) : x = 7.0819 * 10^4 :=
by 
  -- Proof goes here
  sorry

end scientific_notation_l509_509750


namespace no_triangle_exists_l509_509788

theorem no_triangle_exists (a b l : ℕ) (h1 : a = 12) (h2 : b = 20) (h3 : l = 15) :
  ∀ c : ℕ, (c + a > b) ∧ (c + b > a) ∧ (a + b > c) → False :=
begin
  intros c h,
  have ha := h.left,
  have hb := h.right.left,
  have hc := h.right.right,
  sorry
end

end no_triangle_exists_l509_509788


namespace count_no_real_solutions_pairs_l509_509596

-- Define the condition for the quadratic equation not having real solutions
def no_real_solutions_quadratic (b c : ℕ) : Prop :=
  c ^ 2 < 4 * (b + 1)

-- Prove the main statement
theorem count_no_real_solutions_pairs : ∃ n, n = 8 ∧
  (∃ (pairs : finset (ℕ × ℕ)), pairs.card = n ∧
    ∀ (pair : ℕ × ℕ), pair ∈ pairs ↔
      (pair.1 > 0 ∧ pair.2 > 0 ∧ no_real_solutions_quadratic pair.1 pair.2)) :=
by
  sorry

end count_no_real_solutions_pairs_l509_509596


namespace ratio_S4_a4_l509_509029

variable (a1 : ℝ)
def q : ℝ := 1 / 2

def S4 : ℝ := a1 * (1 - q^4) / (1 - q)
def a4 : ℝ := a1 * q^3

theorem ratio_S4_a4 : S4 a1 / a4 a1 = 240 := by
  sorry

end ratio_S4_a4_l509_509029


namespace car_travel_distance_per_hour_l509_509032

theorem car_travel_distance_per_hour :
  ∀ (distance_between_poles : ℕ) (poles_seen : ℕ) (time_minutes : ℕ),
  (distance_between_poles = 50) →
  (poles_seen = 41) →
  (time_minutes = 2) →
  let intervals := poles_seen - 1 in
  let total_distance := distance_between_poles * intervals in
  let speed_per_minute := total_distance / time_minutes in
  let speed_per_hour := speed_per_minute * 60 in
  speed_per_hour = 60000 :=
by
  intros distance_between_poles poles_seen time_minutes
  sorry

end car_travel_distance_per_hour_l509_509032


namespace binary_10101_to_decimal_l509_509151

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.zipWith (λ digit idx => digit * 2^idx) (List.range b.length) |>.sum

theorem binary_10101_to_decimal : binary_to_decimal [1, 0, 1, 0, 1] = 21 := by
  sorry

end binary_10101_to_decimal_l509_509151


namespace area_triangle_OA₂C_area_triangle_A₁A₂C_l509_509322

open Real

-- Mathematical conditions
variables (A B C O : Point) (A₁ A₂ : Point)
variables AB AC ∠BAC : ℝ
variables (circumcircle : Circle)
variables [IsCenter O circumcircle] [OnCircle circumcircle A] [OnCircle circumcircle B]
variables [OnCircle circumcircle C] [OnCircle circumcircle A₂]
variables [IsAngleBisector A A₁] [Extends A A₁ A₂]

-- Definitions for conditions
def AB_length : AB = 3 := sorry
def AC_length : AC = 4 := sorry
def angle_BAC : ∠ BAC = 60 := sorry 
def bisector_intersect : ∃ A₁ A₂, AA₁ ∩ circumcircle = {A₂} := sorry

-- Foster the ensuing expected results based on the conditions
noncomputable def area_OA₂C : ℝ := (13 * sqrt(3)) / 12
noncomputable def area_A₁A₂C : ℝ := (13 * sqrt(3)) / 21

-- Lean theorem statements
theorem area_triangle_OA₂C :
  area (triangle O A₂ C) = (13 * sqrt(3)) / 12 := sorry

theorem area_triangle_A₁A₂C :
  area (triangle A₁ A₂ C) = (13 * sqrt(3)) / 21 := sorry

end area_triangle_OA₂C_area_triangle_A₁A₂C_l509_509322


namespace right_triangle_median_angles_l509_509946

theorem right_triangle_median_angles {A B C M : Type*} 
  (h_right : ∠C = 90°)
  (h_med : is_median C M)
  (h_ratio : ∠ACM = 1 / 3 * ∠MCB) :
  (∠BAC = 60° ∧ ∠ABC = 30°) :=
sorry

end right_triangle_median_angles_l509_509946


namespace simplify_rationalize_l509_509404

theorem simplify_rationalize : (1 : ℝ) / (1 + (1 / (real.sqrt 5 + 2))) = (real.sqrt 5 + 1) / 4 :=
by sorry

end simplify_rationalize_l509_509404


namespace distinct_prime_factors_count_l509_509566

theorem distinct_prime_factors_count :
  let n := 77 * 79 * 81 * 83 in
  ∃ (p : Finset ℕ), (∀ x ∈ p, Prime x) ∧ (n = p.prod id) ∧ (p.card = 5) :=
by
  let n := 77 * 79 * 81 * 83
  use {3, 7, 11, 79, 83}
  split
  · intros x hx
    finset.mem_insert.mp hx
    repeat { cases hx; try { exact Prime_def.mp hx }}
  · split
    · -- Calculate the product of primes {3, 7, 11, 79, 83} and check equality with n.
      sorry
    · -- Show the cardinality of the set {3, 7, 11, 79, 83} is 5
      sorry

end distinct_prime_factors_count_l509_509566


namespace series_approximation_l509_509558

noncomputable def infinite_series_value : ℤ :=
  ∑' n in (Set.Ici 3 : Set ℤ), (n^4 + 4 * n^2 + 11 * n + 15) / (2^n * (n^4 + 9))

theorem series_approximation :
  infinite_series_value ≈ 1 / 4 :=
sorry

end series_approximation_l509_509558


namespace smaller_root_of_equation_l509_509448

theorem smaller_root_of_equation :
  let eq := (fun x => (x - (5 / 6)) * (x - (5 / 6)) + (x - (5 / 6)) * (x - (2 / 3))) in
  ∃ x : ℝ, eq x = 0 ∧ x = (5 / 6) ∧
  ∀ y : ℝ, eq y = 0 → (y = (5 / 6) ∨ y = (13 / 12)) → y ≥ (5 / 6) := by
  sorry

end smaller_root_of_equation_l509_509448


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l509_509227

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l509_509227


namespace calculation1_calculation2_calculation3_calculation4_l509_509136

theorem calculation1 : 10 - (-6) + 8 - (+2) = 22 := by
  sorry

theorem calculation2 : (1 / 6) * (-6) / (- (1 / 7)) * 7 = 49 := by
  sorry

theorem calculation3 : -1^2021 * (4 - (-3)^2) + 3 / (-(3 / 4)) = 1 := by
  sorry

theorem calculation4 : ((5 / 12) - (7 / 9) + (2 / 3)) / (1 / 36) = 11 := by
  sorry

end calculation1_calculation2_calculation3_calculation4_l509_509136


namespace circles_product_condition_l509_509321

theorem circles_product_condition :
  (∃ a b c : ℕ, 2 * a = 3 * c ∧ (6 * c^2 * b = 56) ∧ 
    (∃ k ∈ {1, 2, 4, 7, 14, 28}, b = 28 / k ∧ a = 3 * k ∧ c = 2 * k)) ∧ 
  ((∃! k ∈ {1, 2, 4, 7, 14, 28}, true) = 6) := 
by {
  sorry
}

end circles_product_condition_l509_509321


namespace problem_part1_problem_part2_l509_509301

-- Define the problem conditions
variables (A B C a b c : ℝ)
variables (h1 : c = sqrt 7)
variables (h2 : c * sin A = sqrt 3 * a * cos C)
variables (h3 : sin C + sin (B - A) = 3 * sin (2 * A))

-- Define the problem statement
theorem problem_part1 (h2 : c * sin A = sqrt 3 * a * cos C) : C = π / 3 :=
by sorry

theorem problem_part2 (h1 : c = sqrt 7) (h3 : sin C + sin (B - A) = 3 * sin (2 * A)) :
  ∃ (area : ℝ), area = 7 * sqrt 3 / 6 ∨ area = 3 * sqrt 3 / 4 :=
by sorry

end problem_part1_problem_part2_l509_509301


namespace find_line_passing_P_area_l509_509252

noncomputable def eq_set_area (x y : ℝ) : Prop := (y + x - 3 = 0) ∨ (y + 4 * x - 6 = 0) ∨
  (2 * y - (13 + 3 * Real.sqrt 17) * x + 9 + 3 * Real.sqrt 17 = 0) ∨
  (2 * y - (13 - 3 * Real.sqrt 17) * x + 9 - 3 * Real.sqrt 17 = 0)

theorem find_line_passing_P_area :
  let l1 := (λ (x y : ℝ), 2 * x - 3 * y + 4 = 0) 
  let l2 := (λ (x y : ℝ), x + 2 * y - 5 = 0) 
  let P := (1, 2) in  -- intersection point of l1 and l2
  ∃ k : ℝ, k ≠ 0 ∧ eq_set_area (λ x y, y - 2 = k * (x - 1)) :=
sorry

end find_line_passing_P_area_l509_509252


namespace eval_expression_l509_509580

theorem eval_expression : (⌈(7: ℚ) / 3⌉ + ⌊ -((7: ℚ) / 3)⌋) = 0 :=
begin
  sorry
end

end eval_expression_l509_509580


namespace num_ordered_pairs_l509_509958

theorem num_ordered_pairs (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x * y = 4410) : 
  ∃ (n : ℕ), n = 36 :=
sorry

end num_ordered_pairs_l509_509958


namespace arithmetic_sequence_conclusions_l509_509244

variable {a : ℕ → ℝ}
variable (d : ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
n * (a 1 + a n) / 2

def minimum_term_of_sum (a : ℕ → ℝ) (d : ℝ) (n : ℕ) : Prop :=
S_4 = (4 * (a 1 + a 4) / 2) ∧ ∀ k, S_4 ≤ sum_first_n_terms a k

theorem arithmetic_sequence_conclusions (a : ℕ → ℝ) (d : ℝ) (S_4 S_7 : ℝ) :
  is_arithmetic_sequence a d →
  sum_first_n_terms a 4 = S_4 →
  S_4 < 0 →
  minimum_term_of_sum a d 4 →
  (d > 0) ∧ (a 4 < 0) ∧ (a 5 > 0) ∧ (S_7 < 0) :=
sorry

end arithmetic_sequence_conclusions_l509_509244


namespace years_between_2000_and_3000_with_property_l509_509062

theorem years_between_2000_and_3000_with_property :
  ∃ n : ℕ, n = 143 ∧
  ∀ Y, 2000 ≤ Y ∧ Y ≤ 3000 → ∃ p q : ℕ, p + q = Y ∧ 2 * p = 5 * q →
  (2 * Y) % 7 = 0 :=
sorry

end years_between_2000_and_3000_with_property_l509_509062


namespace NaCl_moles_formed_l509_509951

-- Definitions for the conditions
def NaOH_moles : ℕ := 2
def Cl2_moles : ℕ := 1

-- Chemical reaction of NaOH and Cl2 resulting in NaCl and H2O
def reaction (n_NaOH n_Cl2 : ℕ) : ℕ :=
  if n_NaOH = 2 ∧ n_Cl2 = 1 then 2 else 0

-- Statement to be proved
theorem NaCl_moles_formed : reaction NaOH_moles Cl2_moles = 2 :=
by
  sorry

end NaCl_moles_formed_l509_509951


namespace problem_solution_l509_509035

def sequence_graphical_representation_isolated (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ x : ℝ, x = a n

def sequence_terms_infinite (a : ℕ → ℝ) : Prop :=
  ∃ l : List ℝ, ∃ n : ℕ, l.length = n

def sequence_general_term_formula_unique (a : ℕ → ℝ) : Prop :=
  ∀ f g : ℕ → ℝ, (∀ n, f n = g n) → f = g

theorem problem_solution
  (h1 : ∀ a : ℕ → ℝ, sequence_graphical_representation_isolated a)
  (h2 : ¬ ∀ a : ℕ → ℝ, sequence_terms_infinite a)
  (h3 : ¬ ∀ a : ℕ → ℝ, sequence_general_term_formula_unique a) :
  ∀ a : ℕ → ℝ, sequence_graphical_representation_isolated a ∧ 
                ¬ (sequence_terms_infinite a) ∧
                ¬ (sequence_general_term_formula_unique a) := by
  sorry

end problem_solution_l509_509035


namespace conjugate_in_fourth_quadrant_l509_509775

def complex_modulus (z : ℂ) : ℝ := complex.abs z

def z_condition : Prop :=
  ∃ z : ℂ, (1 + complex.i) * z = complex_modulus (complex.mk (real.sqrt 3) 1)

def quadrant (z : ℂ) : string :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
  else "On axis"

theorem conjugate_in_fourth_quadrant :
  (∃ z : ℂ, z_condition ∧ quadrant (complex.conj z) = "Fourth quadrant") :=
sorry

end conjugate_in_fourth_quadrant_l509_509775


namespace count_circle_points_l509_509678

-- Define the integer points on the circle x^2 + y^2 = 25
def circle_points : Set (ℤ × ℤ) :=
  { p | (p.1 ^ 2 + p.2 ^ 2 = 25) }

-- assert statement
theorem count_circle_points : circle_points.toFinset.card = 12 :=
  sorry

end count_circle_points_l509_509678


namespace solve_equation_l509_509783

theorem solve_equation :
  ∀ (f : ℤ → ℤ) (x : ℤ), f x = x + 4 → (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1) → x = 2 / 5 :=
by
  -- Definitions of f(0) and f(2x+1)
  intros f x Hf H_eq
  have f0 : f 0 = 4 := by rw [Hf, add_zero]; exact rfl
  have f_2x1 : f (2 * x + 1) = 2 * x + 5 := by rw [Hf]; exact rfl
  
  -- Definitions of f(x-2)
  have f_x2 : f (x - 2) = x + 2 := by rw [Hf, sub_add_eq_add_sub]; exact rfl
  
  -- The equation
  sorry

end solve_equation_l509_509783


namespace tens_place_of_8_pow_1234_l509_509817

theorem tens_place_of_8_pow_1234 : (8^1234 / 10) % 10 = 0 := by
  sorry

end tens_place_of_8_pow_1234_l509_509817


namespace compute_x_over_w_l509_509557

theorem compute_x_over_w (w x y z : ℚ) (hw : w ≠ 0)
  (h1 : (x + 6 * y - 3 * z) / (-3 * x + 4 * w) = 2 / 3)
  (h2 : (-2 * y + z) / (x - w) = 2 / 3) :
  x / w = 2 / 3 :=
sorry

end compute_x_over_w_l509_509557


namespace boosters_club_average_sales_l509_509766

theorem boosters_club_average_sales:
  let january_sales := 120
  let february_sales := 80
  let march_sales := -20
  let april_sales := 100
  let may_sales := 140
  let total_sales := january_sales + february_sales + march_sales + april_sales + may_sales
  let number_of_months := 5
  average_sales_per_month = total_sales / number_of_months 
  → average_sales_per_month = 84 :=
by
  let january_sales := 120
  let february_sales := 80
  let march_sales := -20
  let april_sales := 100
  let may_sales := 140
  let total_sales := january_sales + february_sales + march_sales + april_sales + may_sales
  let number_of_months := 5
  let average_sales_per_month := total_sales / number_of_months 
  have h1: average_sales_per_month = 84
  · sorry
  exact h1

end boosters_club_average_sales_l509_509766


namespace sum_of_roots_eq_sixteen_l509_509077

theorem sum_of_roots_eq_sixteen :
  (∑ x in { x : ℝ | x^2 = 16 * x - 9 }) = 16 :=
sorry

end sum_of_roots_eq_sixteen_l509_509077


namespace minimum_value_l509_509672

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 / x + 1 / y = 1) : 3 * x + 4 * y ≥ 25 :=
sorry

end minimum_value_l509_509672


namespace one_point_repeating_03_as_fraction_l509_509583

/-- Prove that 1.03 repeated (1.\overline{03}) is equal to 34/33 given that 0.01 repeated (0.\overline{01}) is equal to 1/99 -/
theorem one_point_repeating_03_as_fraction (h : (0.\overline{01} : ℝ) = 1/99) : (1.\overline{03} : ℝ) = 34/33 :=
begin
  sorry,
end

end one_point_repeating_03_as_fraction_l509_509583


namespace parabola_focus_directrix_l509_509654

theorem parabola_focus_directrix :
  ∀ (p : ℝ), (∃ y : ℝ, (p / 2 + 2 * y - 4 = 0) ∧ (y^2 = 2 * p * (p / 2))) →
    p = 8 ∧ ∀ x : ℝ, x = - p / 2 →
    x = -4 :=
by
  intros p h
  cases h with y hp
  sorry

end parabola_focus_directrix_l509_509654


namespace Amara_clothes_remaining_l509_509129

-- Defining the initial conditions
def initial_clothes : ℕ := 500
def fraction_first_orphanage : ℚ := 1 / 10
def fraction_second_orphanage : ℚ := 3 / 10
def fraction_shelter : ℚ := 1 / 5
def increase_percentage : ℚ := 0.20
def fraction_discard : ℚ := 1 / 8

-- Define the proof problem
theorem Amara_clothes_remaining :
  let donated_first := fraction_first_orphanage * initial_clothes,
      donated_second := fraction_second_orphanage * initial_clothes,
      remaining_after_orphanages := initial_clothes - (donated_first + donated_second),
      donated_shelter := fraction_shelter * remaining_after_orphanages,
      remaining_after_shelter := remaining_after_orphanages - donated_shelter,
      increased_clothes := increase_percentage * remaining_after_shelter,
      new_total := remaining_after_shelter + increased_clothes,
      discarded := fraction_discard * new_total in
  new_total - discarded = 252 := 
by {
  -- final sorry to skip the proof steps
  sorry
}

end Amara_clothes_remaining_l509_509129


namespace evaluate_expression_l509_509336

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

def a := 2 * sqrt 5 + sqrt 7 + sqrt 35
def b := -2 * sqrt 5 + sqrt 7 + sqrt 35
def c := 2 * sqrt 5 - sqrt 7 + sqrt 35
def d := -2 * sqrt 5 - sqrt 7 + sqrt 35

theorem evaluate_expression :
  ( (1 / a) + (1 / b) + (1 / c) + (1 / d) ) ^ 2 = 560 / 155432121 := by
  sorry

end evaluate_expression_l509_509336


namespace sam_quarters_mowing_lawns_l509_509012

-- Definitions based on the given conditions
def pennies : ℕ := 9
def total_amount_dollars : ℝ := 1.84
def penny_value_dollars : ℝ := 0.01
def quarter_value_dollars : ℝ := 0.25

-- Theorem statement that Sam got 7 quarters given the conditions
theorem sam_quarters_mowing_lawns : 
  (total_amount_dollars - pennies * penny_value_dollars) / quarter_value_dollars = 7 := by
  sorry

end sam_quarters_mowing_lawns_l509_509012


namespace cos_120_eq_neg_half_l509_509092

theorem cos_120_eq_neg_half : cos (120 : ℝ) = - (1 / 2) :=
by
  -- Given conditions and known values
  -- We need to show cos 120 = -1/2 given cos (180 - 60) = - cos (60)
  -- And cos 60 = 1/2
  sorry

end cos_120_eq_neg_half_l509_509092


namespace closest_factors_of_2016_l509_509290

theorem closest_factors_of_2016 :
  ∃ a b : ℕ, a * b = 2016 ∧ a > 0 ∧ b > 0 ∧ ∀ c d : ℕ, c * d = 2016 → c > 0 → d > 0 → abs (a - b) ≤ abs (c - d) → abs (a - b) = 6 :=
by
  sorry

end closest_factors_of_2016_l509_509290


namespace solve_for_nonzero_x_l509_509492

open Real

theorem solve_for_nonzero_x (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 → x = 2 / 9 :=
by
  sorry

end solve_for_nonzero_x_l509_509492


namespace twelfth_term_sequence_l509_509877

def sequence : ℕ → ℕ
| 0       => 1
| 1       => 2
| (n + 2) => (List.range (n + 2)).sum $ λ i => sequence i

theorem twelfth_term_sequence : sequence 11 = 1536 := by
  sorry

end twelfth_term_sequence_l509_509877


namespace circle_equation_l509_509778

noncomputable def equation_of_circle (h k r : ℝ) : ℝ → ℝ → ℝ := 
  λ x y, (x - h)^2 + (y - k)^2 - r^2

theorem circle_equation (x y : ℝ) :
  (3, 4) ∈ set_of (λ c : ℝ × ℝ, equation_of_circle c.1 c.2 5 0 0 = 25) :=
sorry

end circle_equation_l509_509778


namespace eval_expression_l509_509578

theorem eval_expression : (Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ)) = 0 := by
  sorry

end eval_expression_l509_509578


namespace smallest_integer_in_set_l509_509313

theorem smallest_integer_in_set : 
  ∀ (n : ℤ), (n + 6 < 2 * (n + 3)) → n ≥ 1 :=
by 
  sorry

end smallest_integer_in_set_l509_509313


namespace second_sphere_surface_area_l509_509103

theorem second_sphere_surface_area (C : ℝ) (hC : C = 6 * Real.pi) : 
  ∃ A : ℝ, A = 36 * Real.pi :=
by
  -- definitions extracted from conditions
  let r := C / (2 * Real.pi)
  let D := 2 * r
  let s := D
  let r2 := s / 2
  let surface_area := 4 * Real.pi * r2^2
  -- prove the statement
  use surface_area
  have h1 : r = 3 := by sorry
  have h2 : D = 6 := by sorry
  have h3 : r2 = 3 := by sorry
  rw [h3]
  have h4 : surface_area = 36 * Real.pi := by sorry
  rw [h4]
  exact h4

end second_sphere_surface_area_l509_509103


namespace competition_end_time_l509_509101

def time_in_minutes := 24 * 60  -- Total minutes in 24 hours

def competition_start_time := 14 * 60 + 30  -- 2:30 p.m. in minutes from midnight

theorem competition_end_time :
  competition_start_time + 1440 = competition_start_time :=
by 
  sorry

end competition_end_time_l509_509101


namespace checkerboard_problem_l509_509777

def is_rel_prime (a b : ℕ) : Prop := ∀ d : ℕ, d ∣ a → d ∣ b → d = 1

theorem checkerboard_problem :
  let r := 784,
      s := 140,
      m := 5,
      n := 28
  in is_rel_prime m n ∧ (s / r = m / n) ∧ (m + n = 33) := by
      sorry

end checkerboard_problem_l509_509777


namespace max_sqrt_expression_l509_509358

noncomputable def max_value_sqrt (x y z : ℝ) : ℝ := 
  √(2 * x) + √(2 * y + 3) + √(2 * z + 2)

theorem max_sqrt_expression : 
  ∀ x y z : ℝ, 
  x + y + z = 3 ∧ x ≥ 0 ∧ y ≥ -3/2 ∧ z ≥ -1 → 
  max_value_sqrt x y z ≤ 3 * √(2) :=
sorry

end max_sqrt_expression_l509_509358


namespace quadratic_has_distinct_real_roots_expression_value_l509_509203

variable (x m : ℝ)

-- Condition: Quadratic equation
def quadratic_eq := (x^2 - 2 * (m - 1) * x - m * (m + 2) = 0)

-- Prove that the quadratic equation always has two distinct real roots
theorem quadratic_has_distinct_real_roots (m : ℝ) : 
  ∃ a b : ℝ, a ≠ b ∧ quadratic_eq a m ∧ quadratic_eq b m :=
by
  sorry

-- Given that x = -2 is a root, prove that 2018 - 3(m-1)^2 = 2015
theorem expression_value (m : ℝ) (h : quadratic_eq (-2) m) : 
  2018 - 3 * (m - 1)^2 = 2015 :=
by
  sorry

end quadratic_has_distinct_real_roots_expression_value_l509_509203


namespace product_roots_h_equals_676_l509_509363

noncomputable def f (x : ℝ) := x^6 + 2*x^3 + 1
noncomputable def h (x : ℝ) := x^3 - 3*x

theorem product_roots_h_equals_676
  (y : Fin 6 → ℝ)
  (h_roots : ∀ i, f (y i) = 0) :
  ∏ i, h (y i) = 676 := sorry

end product_roots_h_equals_676_l509_509363


namespace binary_10101_to_decimal_l509_509150

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.zipWith (λ digit idx => digit * 2^idx) (List.range b.length) |>.sum

theorem binary_10101_to_decimal : binary_to_decimal [1, 0, 1, 0, 1] = 21 := by
  sorry

end binary_10101_to_decimal_l509_509150


namespace intersection_S_T_l509_509275

def S : Set ℤ := {-4, -3, 6, 7}
def T : Set ℤ := {x | x^2 > 4 * x}

theorem intersection_S_T : S ∩ T = {-4, -3, 6, 7} :=
by
  sorry

end intersection_S_T_l509_509275


namespace functional_equation_continuous_function_l509_509522

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_continuous_function (f : ℝ → ℝ) (x₀ : ℝ) (h1 : Continuous f) (h2 : f x₀ ≠ 0) 
  (h3 : ∀ x y : ℝ, f (x + y) = f x * f y) : 
  ∃ a > 0, ∀ x : ℝ, f x = a ^ x := 
by
  sorry

end functional_equation_continuous_function_l509_509522


namespace sam_quarters_l509_509010

theorem sam_quarters (pennies : ℕ) (total : ℝ) (value_penny : ℝ) (value_quarter : ℝ) (quarters : ℕ) :
  pennies = 9 →
  total = 1.84 →
  value_penny = 0.01 →
  value_quarter = 0.25 →
  quarters = (total - pennies * value_penny) / value_quarter →
  quarters = 7 :=
by
  intros
  sorry

end sam_quarters_l509_509010


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l509_509212

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l509_509212


namespace largest_cos_x_l509_509347

theorem largest_cos_x (x y z : ℝ) (hx : sin x = cot y) (hy : sin y = cot z) (hz : sin z = cot x) :
  cos x = sqrt ((1 - sqrt 5) / 2) / sqrt 2 :=
sorry

end largest_cos_x_l509_509347


namespace card_rearrangement_count_l509_509593

-- Define the problem conditions and the correct answer
theorem card_rearrangement_count : 
  let positions := ['A', 'B', 'C', 'D', 'E']
  (∑ σ in ((multiset.map (λ σ, list.perm σ ['A', 'B', 'C', 'D', 'E']).to_finset).filter 
    (λ σ, ∀ x ∈ σ.to_list :: ∀ i < 5, x ^ (nat.pred i)) = 8 := 
sorry -- proof is omitted

end card_rearrangement_count_l509_509593


namespace problem_statement_l509_509159

theorem problem_statement :
  (3003 + (1 / 3) * (3002 + (1 / 3) * (3001 + (1 / 3) * (3000 + (1 / 3) * (3001 - 2998) + ... + (1 / 3) * (4 + (1 / 3) * 3))))) = 9006.5 := 
sorry

end problem_statement_l509_509159


namespace ball_draw_probability_l509_509186

/-- 
Four balls labeled with numbers 1, 2, 3, 4 are placed in an urn. 
A ball is drawn, its number is recorded, and then the ball is returned to the urn. 
This process is repeated three times. Each ball is equally likely to be drawn on each occasion. 
Given that the sum of the numbers recorded is 7, the probability that the ball numbered 2 was drawn twice is 1/4. 
-/
theorem ball_draw_probability :
  let draws := [(1, 1, 5),(1, 2, 4),(1, 3, 3),(2, 2, 3)]
  (3 / 12 = 1 / 4) :=
by
  sorry

end ball_draw_probability_l509_509186


namespace distinct_integers_product_sum_l509_509051

theorem distinct_integers_product_sum (
  a b c d e f : ℤ 
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f)
  (h_prod : a * b * c * d * e * f = 36)) :
  a + b + c + d + e + f = 0 :=
by
  sorry

end distinct_integers_product_sum_l509_509051


namespace problem1_l509_509939

variable (m : ℤ)

theorem problem1 : m * (m - 3) + 3 * (3 - m) = (m - 3) ^ 2 := by
  sorry

end problem1_l509_509939


namespace math_problem_l509_509195

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [MulAction ℤ α] [SMulWithZero ℕ α]

def problem (a : Fin 100 → α) : Prop :=
  (a 0 - 3 * a 1 + 2 * a 2) ≥ 0 ∧
  (a 1 - 3 * a 2 + 2 * a 3) ≥ 0 ∧
  (a 2 - 3 * a 3 + 2 * a 4) ≥ 0 ∧
  -- ...
  (a 98 - 3 * a 99 + 2 * a 0) ≥ 0 ∧
  (a 99 - 3 * a 0 + 2 * a 1) ≥ 0

theorem math_problem (a : Fin 100 → α) (h : problem a) : ∀ i j, a i = a j := 
by
  sorry

end math_problem_l509_509195


namespace work_done_by_resultant_force_l509_509098

-- Define the vectors and points as given in the problem
def F1 : ℝ × ℝ := (3, -4)
def F2 : ℝ × ℝ := (2, -5)
def F3 : ℝ × ℝ := (3, 1)
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (0, 5)

-- Work done by the resultant force
theorem work_done_by_resultant_force : 
  let F := (F1.1 + F2.1 + F3.1, F1.2 + F2.2 + F3.2) in
  let AB := (B.1 - A.1, B.2 - A.2) in
  F.1 * AB.1 + F.2 * AB.2 = -40 :=
by
  -- The proof will be provided here
  sorry

end work_done_by_resultant_force_l509_509098


namespace count_num_positive_integers_satisfy_condition_l509_509156

def num_positive_integers_satisfy_condition := 
  {n : ℕ // 300 < n^2 ∧ n^2 < 1200}

theorem count_num_positive_integers_satisfy_condition : 
  (fintype.card num_positive_integers_satisfy_condition) = 17 :=
by
  sorry

end count_num_positive_integers_satisfy_condition_l509_509156


namespace range_of_x_l509_509243

theorem range_of_x (a x : ℝ) (h1 : log a (1/2) > 0) (h2 : a^(x^2 + 2*x - 4) ≤ 1/a) : 
  x ≤ -3 ∨ x ≥ 1 :=
by
  sorry

end range_of_x_l509_509243


namespace fifth_powers_sum_eq_l509_509216

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l509_509216


namespace incorrect_option_c_l509_509545

theorem incorrect_option_c (a b c d : ℝ)
  (h1 : a + b + c ≥ d)
  (h2 : a + b + d ≥ c)
  (h3 : a + c + d ≥ b)
  (h4 : b + c + d ≥ a) :
  ¬(a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0) :=
by sorry

end incorrect_option_c_l509_509545


namespace equal_lengths_of_perpendicular_dropped_points_l509_509026

/--
Given an acute-angled triangle ABC with altitudes intersecting at O,
and points B1 on segment OB, and C1 on segment OC
such that ∠A B_1 C = ∠A C_1 B = 90 degrees,
prove that AB_1 = AC_1.
-/
theorem equal_lengths_of_perpendicular_dropped_points 
  (A B C O B1 C1 : Point) 
  (h_acute : acute_angle_triangle A B C)
  (h_O : O = intersection_of_altitudes A B C)
  (h_B1_on_OB : B1 ∈ segment O B)
  (h_C1_on_OC : C1 ∈ segment O C)
  (h_angle_ABC1 : ∠ A B1 C = 90°)
  (h_angle_AC1B : ∠ A C1 B = 90°)
  : distance A B1 = distance A C1 := 
sorry

end equal_lengths_of_perpendicular_dropped_points_l509_509026


namespace number_of_boys_l509_509802

-- Definitions based on conditions
def students_in_class : ℕ := 30
def cups_brought_total : ℕ := 90
def cups_per_boy : ℕ := 5

-- Definition of boys and girls, with a constraint from the conditions
variable (B : ℕ)
def girls_in_class (B : ℕ) : ℕ := 2 * B

-- Properties from the conditions
axiom h1 : B + girls_in_class B = students_in_class
axiom h2 : B * cups_per_boy = cups_brought_total - (students_in_class - B) * 0 -- Assume no girl brought any cup

-- We state the question as a theorem to be proved
theorem number_of_boys (B : ℕ) : B = 10 :=
by
  sorry

end number_of_boys_l509_509802


namespace range_values_y1_y2_calculate_x1_given_x2_l509_509316

-- Prove the range of \( y_1 + y_2 \)
theorem range_values_y1_y2 :
  ∀ (α : ℝ), (π / 2 < α ∧ α < π) →
  let x1 := 2 * Real.cos α
  let y1 := 2 * Real.sin α
  let x2 := 2 * Real.cos (α - π / 3)
  let y2 := 2 * Real.sin (α - π / 3)
  y1 + y2 = 2 * sqrt 3 * Real.sin (α - (π / 6)) →
  sqrt 3 < y1 + y2 ∧ y1 + y2 ≤ 2 * sqrt 3 :=
by
  intros
  simp [x1, y1, x2, y2]
  sorry

-- Prove \( x_1 = \frac{1 - 3\sqrt{5}}{4} \) given \( x_2 = \frac{1}{2} \)
theorem calculate_x1_given_x2 :
  ∀ (α : ℝ), (π / 2 < α ∧ α < π) →
  let x2 := 2 * Real.cos (α - π / 3)
  x2 = 1 / 2 →
  let x1 := 2 * Real.cos α
  x1 = (1 - 3 * sqrt 5) / 4 :=
by
  intros
  simp [x1, x2]
  sorry

end range_values_y1_y2_calculate_x1_given_x2_l509_509316


namespace part1_max_value_part2_max_value_l509_509511

noncomputable def f (θ : ℝ) : ℝ := (1 + Real.cos θ) * (1 + Real.sin θ)

theorem part1_max_value :
  ∃ θ ∈ (0 : ℝ, Real.pi / 2), f θ = (3 + 2 * Real.sqrt 2) / 2 :=
sorry

noncomputable def g (θ : ℝ) : ℝ := (1 / 2 + Real.cos θ) * (Real.sqrt 3 / 2 + Real.sin θ)

theorem part2_max_value :
  ∃ θ ∈ (0 : ℝ, Real.pi / 2), g θ = Real.sqrt 3 / 4 + 3 / 2 * Real.sin (5 * Real.pi / 9) :=
sorry

end part1_max_value_part2_max_value_l509_509511


namespace minimum_value_of_expression_l509_509356

theorem minimum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (sum_eq : x + y + z = 5) :
  (9 / x + 4 / y + 25 / z) ≥ 20 :=
sorry

end minimum_value_of_expression_l509_509356


namespace largest_prime_divisor_of_17sq_plus_30sq_l509_509173

theorem largest_prime_divisor_of_17sq_plus_30sq :
  ∃ (p : ℕ), (prime p) ∧ (p ∣ (17^2 + 30^2)) ∧ ∀ q, (prime q) ∧ (q ∣ (17^2 + 30^2)) → q ≤ p := 
sorry

end largest_prime_divisor_of_17sq_plus_30sq_l509_509173


namespace find_sin_alpha_l509_509718

noncomputable def P (m : ℝ) : ℝ × ℝ := (m, Real.sqrt 5)
noncomputable def cos_alpha (m : ℝ) : ℝ := (Real.sqrt 2 / 4) * m
noncomputable def sin_alpha (m : ℝ) : ℝ := -Real.sqrt ((5 - m^2) / (m^2 + 5))

theorem find_sin_alpha {α : ℝ}
  (m : ℝ)
  (h1 : α ∈ (π/2, π))
  (h2 : P m = (m, Real.sqrt 5))
  (h3 : cos_alpha m = Real.sqrt 2 / 4 * m)
  (h4 : m = -Real.sqrt 3 ) : 
  sin_alpha m = Real.sqrt 10 / 4 :=
by
  sorry

end find_sin_alpha_l509_509718


namespace company_employees_speak_french_l509_509515

noncomputable def percentage_of_employees_who_speak_french
  (total_employees : ℕ)
  (perc_men : ℝ)
  (perc_men_speak_french : ℝ)
  (perc_women_not_speak_french : ℝ) : ℝ :=
let total_men := total_employees * perc_men in
let total_women := total_employees * (1 - perc_men) in
let men_speaking_french := total_men * perc_men_speak_french in
let women_speaking_french := total_women * (1 - perc_women_not_speak_french) in
(men_speaking_french + women_speaking_french) / total_employees * 100

theorem company_employees_speak_french :
  percentage_of_employees_who_speak_french 100 0.6 0.6 0.65 = 50 := 
by sorry

end company_employees_speak_french_l509_509515


namespace locus_of_centers_of_tangent_circles_l509_509028

theorem locus_of_centers_of_tangent_circles :
  ∀ (a b : ℝ), (∃ r : ℝ, a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (3 - r)^2) →
    (16 * a^2 + 25 * b^2 - 48 * a - 64 = 0) :=
begin
  intros a b h,
  -- since we are only required to set up the statement, the proof is omitted
  sorry
end

end locus_of_centers_of_tangent_circles_l509_509028


namespace num_positive_solutions_l509_509930

-- Definitions based on given conditions

def theta (x : ℝ) := Real.arcsin x
def cos_theta (x : ℝ) := Real.sqrt (1 - x^2)
def cot_theta (x : ℝ) := cos_theta x / x
def psi (x : ℝ) := Real.arccos (cot_theta x)

-- Main theorem
theorem num_positive_solutions :
  ∃! x : ℝ, (0 < x ∧ x ≤ 1) ∧ (Real.sin (psi x) = x) := by
  sorry

end num_positive_solutions_l509_509930


namespace imaginary_part_of_z_l509_509978

def complex_i : ℂ := complex.I

def z : ℂ := (complex_i / (1 + complex_i)) - (1 / (2 * complex_i))

theorem imaginary_part_of_z :
  complex.im z = 1 :=
sorry

end imaginary_part_of_z_l509_509978


namespace point_position_l509_509870

theorem point_position (initial_pos : ℤ) (move_right : ℤ) (move_left : ℤ) (final_pos : ℤ) :
  initial_pos = -2 → move_right = 7 → move_left = 4 → final_pos = initial_pos + move_right - move_left → final_pos = 1 :=
by
  intros h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  rw h4
  exact Eq.refl 1

end point_position_l509_509870


namespace winning_candidate_percentage_l509_509061

theorem winning_candidate_percentage (votes1 votes2 votes3 : ℕ) 
  (h1 : votes1 = 1036) (h2 : votes2 = 4636) (h3 : votes3 = 11628) :
  let total_votes := votes1 + votes2 + votes3 in
  let winning_votes := max votes1 (max votes2 votes3) in
  (winning_votes : ℚ) / total_votes * 100 = 67.2 :=
by
  rw [h1, h2, h3]
  let total_votes := 1036 + 4636 + 11628
  have : total_votes = 17300 := rfl
  rw this
  let winning_votes := 11628
  have : winning_votes = 11628 := rfl
  rw this
  norm_num
  sorry

end winning_candidate_percentage_l509_509061


namespace auction_sale_l509_509905

theorem auction_sale (TV_initial_price : ℝ) (TV_increase_fraction : ℝ) (Phone_initial_price : ℝ) (Phone_increase_percent : ℝ) :
  TV_initial_price = 500 → 
  TV_increase_fraction = 2 / 5 → 
  Phone_initial_price = 400 →
  Phone_increase_percent = 40 →
  let TV_final_price := TV_initial_price + TV_increase_fraction * TV_initial_price in
  let Phone_final_price := Phone_initial_price + (Phone_increase_percent / 100) * Phone_initial_price in
  TV_final_price + Phone_final_price = 1260 := by
  sorry

end auction_sale_l509_509905


namespace smallest_three_digit_using_1_0_7_l509_509823

theorem smallest_three_digit_using_1_0_7 : ∃ n : ℕ, n = 107 ∧ (∀ m, m < 1000 → m > 99 → uses_digits m [1, 0, 7] → m ≥ n) :=
by
  sorry

end smallest_three_digit_using_1_0_7_l509_509823


namespace area_N1N2N3_is_third_ABC_l509_509840

def area_triangle (A B C : Point) : ℝ := -- some definition for area here
sorry -- Placeholder for the actual implementation

variables (A B C D E F N1 N2 N3 : Point)

-- Conditions
axiom CD_fourth : distance C D = (1/4) * distance A C
axiom AE_fourth : distance A E = (1/4) * distance A B
axiom BF_fourth : distance B F = (1/4) * distance B C

-- Given: Triangle ABC
def triangle_ABC := (A, B, C)

-- Define areas
def area_ABC := area_triangle A B C
def area_ADC := (1/4) * area_ABC
def area_N1DC := (1/36) * area_ADC
def area_N2EA := (1/36) * area_ADC
def area_N3FB := (1/36) * area_ADC

-- Area of the quadrilaterals
def area_N2N1CE := area_ADC - area_N1DC - area_N2EA
def area_N1N2N3 := area_ABC - 3 * area_N1DC - 3 * area_N2N1CE

-- Statement to be proven
theorem area_N1N2N3_is_third_ABC :
  area_N1N2N3 = (1/3) * area_ABC :=
sorry -- Proof to be provided

end area_N1N2N3_is_third_ABC_l509_509840


namespace number_multiply_increase_l509_509109

theorem number_multiply_increase (x : ℕ) (h : 25 * x = 25 + 375) : x = 16 := by
  sorry

end number_multiply_increase_l509_509109


namespace intersection_A_B_l509_509239

-- Define the sets A and B
def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 < 3}

-- Prove that A ∩ B = {0, 1}
theorem intersection_A_B :
  A ∩ B = {0, 1} :=
by
  -- Proof goes here
  sorry

end intersection_A_B_l509_509239


namespace exists_fibonacci_factors_l509_509713

def fibonacci_sequence (n : ℕ) : ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci_sequence n + fibonacci_sequence (n+1)

noncomputable def sum_of_squares_of_divisors (N : ℕ) : ℕ :=
  (List.range (N+1)).filter (λ d, d > 0 ∧ N % d = 0).map (λ d, d^2).sum

theorem exists_fibonacci_factors (N : ℕ) :
  (0 < N) →
  (sum_of_squares_of_divisors N = N * (N + 3)) →
  ∃ i j : ℕ, N = fibonacci_sequence i * fibonacci_sequence j :=
begin
  intros,
  sorry,
end

end exists_fibonacci_factors_l509_509713


namespace log_inequality_solution_l509_509057

theorem log_inequality_solution (x : ℝ) (h1 : x > 1) (h2 : log 10 (x - 1) < 1) : 1 < x ∧ x < 11 := 
by
  sorry

end log_inequality_solution_l509_509057


namespace probability_of_selecting_boy_given_girl_A_selected_l509_509014

-- Define the total number of girls and boys
def total_girls : ℕ := 5
def total_boys : ℕ := 2

-- Define the group size to be selected
def group_size : ℕ := 3

-- Define the probability of selecting at least one boy given girl A is selected
def probability_at_least_one_boy_given_girl_A : ℚ := 3 / 5

-- Math problem reformulated as a Lean theorem
theorem probability_of_selecting_boy_given_girl_A_selected : 
  (total_girls = 5) → (total_boys = 2) → (group_size = 3) → 
  (probability_at_least_one_boy_given_girl_A = 3 / 5) :=
by sorry

end probability_of_selecting_boy_given_girl_A_selected_l509_509014


namespace total_amount_received_l509_509911

def initial_price_tv : ℕ := 500
def tv_increase_rate : ℚ := 2 / 5
def initial_price_phone : ℕ := 400
def phone_increase_rate : ℚ := 0.40

theorem total_amount_received :
  initial_price_tv + initial_price_tv * tv_increase_rate + initial_price_phone + initial_price_phone * phone_increase_rate = 1260 :=
by
  sorry

end total_amount_received_l509_509911


namespace double_sheet_sum_is_constant_l509_509863

-- Define the total number of pages based on conditions
constant total_pages : ℕ := 64

-- Define a double sheet in terms of page pairs (n, total_pages + 1 - n)
def page_pairs (n : ℕ) : ℕ × ℕ := (n, total_pages + 1 - n)

-- The sum of the two pages on a double sheet
def double_sheet_sum (n : ℕ) : ℕ := 
  let (a, b) := page_pairs n in a + b

-- The problem statement: for any n, 1 ≤ n ≤ total_pages / 2,
-- the sum of pages on any double sheet is always 130.
theorem double_sheet_sum_is_constant : ∀ n : ℕ, 1 ≤ n ∧ n ≤ total_pages / 2 → double_sheet_sum n = 130 := 
by
  sorry

end double_sheet_sum_is_constant_l509_509863


namespace nearest_integer_to_sum_l509_509137

noncomputable def telescoping_sum : ℚ := 
  500 * ∑ n in (finset.range (10005 - 4 + 1)).map (λ x, x + 4), 1 / (n^2 - 9)

theorem nearest_integer_to_sum : (⌊ telescoping_sum + 0.5 ⌋ : ℤ) = 174 := 
  sorry

end nearest_integer_to_sum_l509_509137


namespace ratio_of_newspapers_l509_509918

theorem ratio_of_newspapers (C L : ℕ) (h1 : C = 42) (h2 : L = C + 23) : C / (C + 23) = 42 / 65 := by
  sorry

end ratio_of_newspapers_l509_509918


namespace total_amount_received_l509_509912

def initial_price_tv : ℕ := 500
def tv_increase_rate : ℚ := 2 / 5
def initial_price_phone : ℕ := 400
def phone_increase_rate : ℚ := 0.40

theorem total_amount_received :
  initial_price_tv + initial_price_tv * tv_increase_rate + initial_price_phone + initial_price_phone * phone_increase_rate = 1260 :=
by
  sorry

end total_amount_received_l509_509912


namespace find_c_l509_509362

variable (X : ℝ → ℝ)

noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ) ^ 2) / (2 * σ ^ 2))

axiom normal_distribution (μ σ : ℝ) : ∃ X, ∀ x, X x = normal_pdf μ σ x

theorem find_c (c : ℝ) (μ : ℝ) (σ : ℝ) (h : μ = 2) (hσ : σ = 3) 
              (hp : ∀ c, P (λ X, X ≤ c) = P (λ X, X > c)) : c = 2 := by
  sorry

end find_c_l509_509362


namespace inclination_angle_of_line_l509_509644

theorem inclination_angle_of_line (x y : ℝ) : 
  (l : ℝ → ℝ) := x - y - 1 = 0 → 
  let k := 1 in ∠l = 45 := sorry

end inclination_angle_of_line_l509_509644


namespace equal_projection_distances_l509_509685

noncomputable def point3D := (ℝ × ℝ × ℝ)
noncomputable def line3D (θ : ℝ) := {l : set point3D | ∃ v : point3D, ∃ d : ℝ, l = {p | (v.2 / v.1) = tan θ ∧ p = (v.1 * d, v.2 * d, v.3 * d)}}

theorem equal_projection_distances (A B M H K : point3D) (θ : ℝ)
    (hx_AB_in_xy_plane : A.3 = 0 ∧ B.3 = 0)
    (hy_M_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2, 0))
    (hz_HK_projection : ∃ v : point3D, ∃ d1 d2 : ℝ, H = (v.1 * d1, v.2 * d1, v.3 * d1) ∧ K = (v.1 * d2, v.2 * d2, v.3 * d2) ∧ 
                            line3D 45 = {p | (v.2 / v.1) = tan (45) ∧ p = (v.1 * d1, v.2 * d1, v.3 * d1)}) :
    dist M H = dist M K :=
    by 
        sorry

end equal_projection_distances_l509_509685


namespace log_inequality_solution_set_l509_509055

theorem log_inequality_solution_set :
  { x : ℝ | log10 (x - 1) < 1 ∧ x > 1 } = { x : ℝ | 1 < x ∧ x < 11 } :=
sorry

end log_inequality_solution_set_l509_509055


namespace no_valid_prime_base_p_l509_509423

theorem no_valid_prime_base_p :
  ∀ p : ℕ, Prime p → 
  (1 * p^3 + 0 * p^2 + 1 * p + 4 + 3 * p^2 + 0 * p + 7 +
  1 * p^2 + 1 * p + 4 + 1 * p^2 + 2 * p + 6 + 7 =
  1 * p^2 + 4 * p + 3 + 2 * p^2 + 7 * p + 2 + 3 * p^2 + 6 * p + 1) →
  False :=
begin
  intros p hp heq,
  sorry
end

end no_valid_prime_base_p_l509_509423


namespace area_comparison_ratio_given_perimeters_l509_509810

-- Definition of the problem conditions
variables {a b : ℝ}

def Area1 := a * b
def Area2 := 1.11 * a * 0.9 * b
def Perimeter1 := 2 * (a + b)
def Perimeter2 := 2 * (1.11 * a + 0.9 * b)

-- The statement proving the areas condition
theorem area_comparison (a b : ℝ) : 
  Area2 = 0.999 * Area1 :=
by 
  sorry

-- The statement proving the ratio of sides given the perimeter condition
theorem ratio_given_perimeters (a b : ℝ) 
  (h : Perimeter1 = 0.95 * Perimeter2) : 
  a / b = 2.66 :=
by 
  sorry

end area_comparison_ratio_given_perimeters_l509_509810


namespace rate_of_interest_l509_509518

-- Assumptions given in the problem
def principal : ℝ := 671.2018140589569
def amount : ℝ := 740
def years : ℕ := 2

-- Statement to prove the rate of interest
theorem rate_of_interest :
  let r := ((amount / principal)^(1 / (years : ℝ)) - 1) * 100
  r = 4.95 :=
by
  sorry -- Proof of the theorem

end rate_of_interest_l509_509518


namespace num_apartments_per_floor_l509_509102

-- Definitions used in the proof
def num_buildings : ℕ := 2
def floors_per_building : ℕ := 12
def doors_per_apartment : ℕ := 7
def total_doors_needed : ℕ := 1008

-- Lean statement to proof the number of apartments per floor
theorem num_apartments_per_floor : 
  (total_doors_needed / (doors_per_apartment * num_buildings * floors_per_building)) = 6 :=
by
  sorry

end num_apartments_per_floor_l509_509102


namespace min_xy_min_x_add_y_l509_509610

open Real

theorem min_xy (x y : ℝ) (h1 : log x + log y = log (x + y + 3)) (hx : x > 0) (hy : y > 0) : xy ≥ 9 := sorry

theorem min_x_add_y (x y : ℝ) (h1 : log x + log y = log (x + y + 3)) (hx : x > 0) (hy : y > 0) : x + y ≥ 6 := sorry

end min_xy_min_x_add_y_l509_509610


namespace maximize_profit_l509_509126

theorem maximize_profit : 
  ∃ (a b : ℕ), 
  a ≤ 8 ∧ 
  b ≤ 7 ∧ 
  2 * a + b ≤ 19 ∧ 
  a + b ≤ 12 ∧ 
  10 * a + 6 * b ≥ 72 ∧ 
  (a * 450 + b * 350) = 4900 :=
by
  sorry

end maximize_profit_l509_509126


namespace product_of_roots_l509_509483

theorem product_of_roots (x : ℝ) (h : x + 1/x = 4 * x) :
  x * (- (1 / sqrt 3)) = 1 / 3 := 
sorry

end product_of_roots_l509_509483


namespace part1_sin_tan_of_cos_part2_fraction_of_tan_l509_509847

theorem part1_sin_tan_of_cos (α : ℝ) (h_cos : cos α = -4/5) (h_quadrant : π/2 < α ∧ α < π) : 
  sin α = 3/5 ∧ tan α = -3/4 :=
by 
  sorry

theorem part2_fraction_of_tan (α : ℝ) (h_tan : tan α = -2) : 
  (sin α + cos α) / (sin α - 3 * cos α) = 1 / 5 :=
by 
  sorry

end part1_sin_tan_of_cos_part2_fraction_of_tan_l509_509847


namespace domain_of_f_l509_509170

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_of_f :
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ -3) ↔ ((x < -3) ∨ (-3 < x ∧ x < 3) ∨ (x > 3)) :=
by
  sorry

end domain_of_f_l509_509170


namespace wheel_turns_l509_509885

theorem wheel_turns (rA rB rC : ℝ) (turnsA_per_30s : ℝ) (ratio_AB ratio_AC : ℝ) 
(tw_hours seconds_period: ℝ) :
    rA = 4 →
    rB = 2 →
    rC = 3 →
    turnsA_per_30s = 6 →
    ratio_AB = 2 →
    ratio_AC = 3 →
    tw_hours = 2 →
    seconds_period = 30 →
    (7200 / seconds_period * turnsA_per_30s = 1440) ∧ 
    (7200 / seconds_period * turnsA_per_30s * ratio_AB = 2880) ∧ 
    (7200 / seconds_period * turnsA_per_30s * ratio_AC = 4320) :=
    
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8,
  split,
  {
    -- Proof for Wheel A
    calc 7200 / 30 * 6 = 1440 : by norm_num,
  },
  split,
  {
    -- Proof for Wheel B
    calc 7200 / 30 * 6 * 2 = 2880 : by norm_num,
  },
  {
    -- Proof for Wheel C
    calc 7200 / 30 * 6 * 3 = 4320 : by norm_num,
  },
  sorry            -- Placeholder for complete proof
end

end wheel_turns_l509_509885


namespace bailing_rate_l509_509419

theorem bailing_rate 
  (distance_from_shore : ℝ) 
  (initial_intake_rate : ℝ) 
  (boat_capacity : ℝ) 
  (rowing_speed : ℝ) : 
  distance_from_shore = 2 →
  initial_intake_rate = 6 →
  boat_capacity = 60 →
  rowing_speed = 3 →
  ∃ (min_bail_rate : ℝ), min_bail_rate = 4.5 := 
by {
  intros,
  sorry
}

end bailing_rate_l509_509419


namespace distance_between_parallel_lines_is_2sqrt30_l509_509063

theorem distance_between_parallel_lines_is_2sqrt30 
  (r : ℝ) -- radius of the circle
  (d : ℝ) -- distance between the adjacent parallel lines
  (h1 : ∃ P Q : ℝ, P = 15 ∧ Q = 20)
  (h2 : 450 + (15/4) * d^2 = 30 * r^2) 
  (h3 : 1600 + (90/4) * d^2 = 80 * r^2) :
  d = 2 * real.sqrt 30 :=
sorry

end distance_between_parallel_lines_is_2sqrt30_l509_509063


namespace parallel_lines_m_eq_l509_509176

theorem parallel_lines_m_eq (m : ℝ) : 
  (∃ k : ℝ, (x y : ℝ) → 2 * x + (m + 1) * y + 4 = k * (m * x + 3 * y - 2)) → 
  (m = 2 ∨ m = -3) :=
by
  intro h
  sorry

end parallel_lines_m_eq_l509_509176


namespace emmalyn_earnings_l509_509577

theorem emmalyn_earnings :
  let rate_per_meter := 0.20
  let number_of_fences := 50
  let length_of_each_fence := 500
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * rate_per_meter
  total_income = 5000 :=
by
  sorry

end emmalyn_earnings_l509_509577


namespace scaling_transformation_l509_509797

theorem scaling_transformation :
  ∀ (x y x' y' : ℝ), 
    (x' = x) ∧ (y' = 4 * y) ∧ (x - 2 * y = 2) → (2 * x' - y' = 4) :=
by{
  intros x y x' y',
  assume h,
  sorry
}

end scaling_transformation_l509_509797


namespace evaluate_propositions_l509_509237

variable (x y : ℝ)

def p : Prop := (x > y) → (-x < -y)
def q : Prop := (x < y) → (x^2 > y^2)

theorem evaluate_propositions : (p x y ∨ q x y) ∧ (p x y ∧ ¬q x y) := by
  -- Correct answer: \( \boxed{\text{C}} \)
  sorry

end evaluate_propositions_l509_509237


namespace total_children_l509_509803

-- Definitions based on the problem conditions
def happy_children : Nat := 30
def sad_children : Nat := 10
def neither_children : Nat := 20
def boys : Nat := 17
def girls : Nat := 43
def happy_boys : Nat := 6
def sad_girls : Nat := 4
def neither_boys : Nat := 5

-- Theorem stating the total number of children
theorem total_children : boys + girls = 60 := by
  have num_boys : Nat := boys
  have num_girls : Nat := girls
  calc
    num_boys + num_girls = 17 + 43 := by rfl
    ...               = 60         := by rfl

end total_children_l509_509803


namespace last_digit_in_mod_fib_sequence_l509_509528

def mod_fib_sequence (n : ℕ) : ℕ :=
  if n = 1 ∨ n = 2 then 2
  else (mod_fib_sequence (n - 1) + mod_fib_sequence (n - 2)) % 10

theorem last_digit_in_mod_fib_sequence
  : ∃ N, ∀ n, n > N → mod_fib_sequence n ≠ 1 ∧
  -- Further, it captures the first appearance of the digit 1 in the sequence
    ∃ m, m > 0 ∧ m ≤ N ∧ mod_fib_sequence m = 1 := sorry

end last_digit_in_mod_fib_sequence_l509_509528


namespace rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l509_509430

variables (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0)

def S1 := (x + 5) * (y + 5)
def S2 := (x - 2) * (y - 2)
def perimeter := 2 * (x + y)

theorem rectangle_perimeter (h : S1 - S2 = 196) :
  perimeter = 50 :=
sorry

theorem difference_multiple_of_7 (h : S1 - S2 = 196) :
  ∃ k : ℕ, S1 - S2 = 7 * k :=
sorry

theorem area_seamless_combination (h : S1 - S2 = 196) :
  S1 - x * y = (x + 5) * (y + 5) - x * y ∧ x = y + 5 :=
sorry

end rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l509_509430


namespace four_digit_factors_l509_509496

def is4DigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def nonZeroTensDigit (n : ℕ) : Prop := (n / 10) % 10 ≠ 0
def digits (n : ℕ) : ℕ × ℕ := ((n / 100) % 100, n % 100)

theorem four_digit_factors (n : ℕ) :
  is4DigitNumber n ∧ nonZeroTensDigit n ∧
  let (A, B) := digits n in A * B ∣ n
  ↔ n = 1734 ∨ n = 1352 :=
sorry

end four_digit_factors_l509_509496


namespace product_of_solutions_eq_neg_one_third_l509_509485

theorem product_of_solutions_eq_neg_one_third :
  (∏ x in {x : ℝ | x + 1 / x = 4 * x}, x) = -1 / 3 :=
by
  sorry

end product_of_solutions_eq_neg_one_third_l509_509485


namespace radius_of_C1_l509_509684

/-- In a geometric configuration where circle C1 has its center O on circle C2, they intersect at points X and Y, and a point Z outside circle C1 lies on circle C2 with XZ = 15, OZ = 13, and YZ = 8, prove that the radius of circle C1 is sqrt(394). -/
theorem radius_of_C1 (C1 C2 : Circle) (O X Y Z : Point) (r : ℝ)
  (h_center_O : is_center O C1)
  (h_center_O_on_C2 : is_on_circle O C2)
  (h_intersect_XY : C1.intersects_at C2 X Y)
  (h_Z_on_C2 : is_on_circle Z C2)
  (h_Z_outside_C1 : ¬is_on_circle Z C1)
  (h_XZ_eq_15 : dist X Z = 15)
  (h_OZ_eq_13 : dist O Z = 13)
  (h_YZ_eq_8 : dist Y Z = 8) :
  r = real.sqrt 394 := sorry

end radius_of_C1_l509_509684


namespace height_a_leq_half_a_cot_half_α_l509_509391

-- Definitions for the angles and sides
variable (α β γ : ℝ)
variable (a R ha : ℝ)

-- Auxiliary definitions
def height_a := 2 * R * (Real.sin β) * (Real.sin γ)
def side_a := 2 * R * (Real.sin α)

-- Theorem statement
theorem height_a_leq_half_a_cot_half_α (hα : α ∈ Icc 0 π) (hβ : β ∈ Icc 0 π) (hγ : γ ∈ Icc 0 π)
  (hha : ha = height_a α β γ) (ha : a = side_a α) :
  ha ≤ (a / 2) * Real.cot (α / 2) := 
by
  sorry

end height_a_leq_half_a_cot_half_α_l509_509391


namespace trig_identity_l509_509607

theorem trig_identity
  (α β : ℝ)
  (h1 : sin (2 * α) = (sqrt 5) / 5)
  (h2 : sin (β - α) = (sqrt 10) / 10)
  (hα : α ∈ set.Icc (π / 4) π)
  (hβ : β ∈ set.Icc π (3 * π / 2))
  : α + β = (7 * π) / 4 :=
sorry

end trig_identity_l509_509607


namespace usamo_2010_p1_solution_l509_509585

def functional_equation_holds (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, x ≠ 0 → x * f(2 * f(y) - x) + y^2 * f(2 * x - f(y)) = (f(x)^2) / x + f(y * f(y))

theorem usamo_2010_p1_solution (f : ℤ → ℤ) :
  (∀ x : ℤ, f x = x^2) ∨ (∀ x : ℤ, f x = 0) ↔ functional_equation_holds f :=
by
  sorry

end usamo_2010_p1_solution_l509_509585


namespace nucleic_acid_testing_function_total_time_for_6000_residents_l509_509162

def y_of_x (x : ℝ) : ℝ :=
  if x ≤ 10 then 0 else 30 * x - 300

theorem nucleic_acid_testing_function :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ 10 → y_of_x x = 0) ∧ 
           (x > 10 → y_of_x x = 30 * x - 300) := sorry

theorem total_time_for_6000_residents :
  (∃ x : ℝ, 30 * x - 300 = 6000) ∧ (let x := 210)  
  → x = 210 := sorry

end nucleic_acid_testing_function_total_time_for_6000_residents_l509_509162


namespace evaluate_expression_l509_509164

theorem evaluate_expression : (2 * log 3 (1 / 2) + log 3 12 - 0.7^0 + 0.25^(-1) = 4) :=
by
  sorry

end evaluate_expression_l509_509164


namespace transformed_stats_l509_509257

-- Define the mean of the original data
def original_mean : ℝ := 6

-- Define the standard deviation of the original data
def original_std_dev : ℝ := 2

-- Define the transformed data
def transformed_data (x : ℝ → ℝ) : ℝ := 2 * x - 6

-- Theorem stating the mean and variance of the transformed data
theorem transformed_stats :
  let trans_mean := transformed_data original_mean * 1
  let trans_variance := (2 * original_std_dev)^2
  trans_mean = 6 ∧ trans_variance = 16 :=
by
  sorry

end transformed_stats_l509_509257


namespace simplify_product_fractions_l509_509417

theorem simplify_product_fractions : 
  (∏ n in Finset.range 206, (5 * (n + 2) + 5) / (5 * (n + 2))) = 206 := by
sorry

end simplify_product_fractions_l509_509417


namespace single_reduction_equivalent_l509_509444

/-- If a price is first reduced by 25%, and the new price is further reduced by 30%, 
the single percentage reduction equivalent to these two reductions together is 47.5%. -/
theorem single_reduction_equivalent :
  ∀ P : ℝ, (1 - 0.25) * (1 - 0.30) * P = P * (1 - 0.475) :=
by
  intros
  sorry

end single_reduction_equivalent_l509_509444


namespace all_of_the_above_used_as_money_l509_509832

-- Definition to state that each item was used as money
def gold_used_as_money : Prop := true
def stones_used_as_money : Prop := true
def horses_used_as_money : Prop := true
def dried_fish_used_as_money : Prop := true
def mollusk_shells_used_as_money : Prop := true

-- Statement that all of the above items were used as money
theorem all_of_the_above_used_as_money : gold_used_as_money ∧ stones_used_as_money ∧ horses_used_as_money ∧ dried_fish_used_as_money ∧ mollusk_shells_used_as_money :=
by {
  split; -- Split conjunctions
  all_goals { exact true.intro }; -- Each assumption is true
}

end all_of_the_above_used_as_money_l509_509832


namespace orchestra_member_count_l509_509791

theorem orchestra_member_count :
  ∃ x : ℕ, 150 ≤ x ∧ x ≤ 250 ∧ 
           x % 4 = 2 ∧
           x % 5 = 3 ∧
           x % 8 = 4 ∧
           x % 9 = 5 :=
sorry

end orchestra_member_count_l509_509791


namespace problem_l509_509452

theorem problem (a : ℤ) :
  (∃ x : ℝ, x > 0 ∧ (3 * x - a) / (x - 3) + (x + 1) / (3 - x) = 1) ∧
  (∀ y : ℝ, (y + 9 ≤ 2 * (y + 2)) ∧ ((2 * y - a) / 3 ≥ 1) ↔ (y ≥ 5)) →
  (a ∈ {3, 4, 6, 7} → a ∈ {3, 4, 6, 7}) :=
by
  sorry

end problem_l509_509452


namespace infinitely_many_n_divide_b_pow_n_plus_1_l509_509599

theorem infinitely_many_n_divide_b_pow_n_plus_1 (b : ℕ) (h1 : b > 2) :
  (∃ᶠ n in at_top, n^2 ∣ b^n + 1) ↔ ¬ ∃ k : ℕ, b + 1 = 2^k :=
sorry

end infinitely_many_n_divide_b_pow_n_plus_1_l509_509599


namespace hyperbola_asymptotes_l509_509268

noncomputable def equation_of_parabola (x := ℝ) (y := ℝ) : Prop :=
  y ^ 2 = 8 * x

noncomputable def equation_of_hyperbola (x := ℝ) (y := ℝ) (m := ℝ) : Prop :=
  (x ^ 2) / m - (y ^ 2) = 1

noncomputable def focus_of_parabola (x := ℝ) (y := ℝ) : Prop :=
  (x, y) = (2, 0)

theorem hyperbola_asymptotes (m : ℝ) (x := ℝ) (y := ℝ) :
  equation_of_parabola 2 0 → equation_of_hyperbola 2 0 m →
  m = 4 →
  ∀ x y,
    equation_of_hyperbola x y m →
    y = (1/2) * x ∨ y = -(1/2) * x :=
by
  intro parabola_focus hyperbola_through_focus m_value
  exact sorry

end hyperbola_asymptotes_l509_509268


namespace calculate_expression_l509_509135

theorem calculate_expression :
  (1/4 * 6.16^2) - (4 * 1.04^2) = 5.16 :=
by
  sorry

end calculate_expression_l509_509135


namespace smallest_result_obtained_l509_509188

def set_numbers : Set ℕ := {2, 3, 5, 7, 11, 13}

-- Define a function that performs the operation described in the problem.
def operation (a b c : ℕ) : ℕ := (a + a + b) * c

-- Define a function to find the minimum result based on the operation for each combination of three numbers
noncomputable def min_result : ℕ :=
  let results := {operation 2 3 5, operation 2 5 3, operation 3 5 2}
  results.min

theorem smallest_result_obtained : min_result = 22 := 
  sorry

end smallest_result_obtained_l509_509188


namespace travel_time_total_l509_509883

theorem travel_time_total (dist1 dist2 dist3 speed1 speed2 speed3 : ℝ)
  (h_dist1 : dist1 = 50) (h_dist2 : dist2 = 100) (h_dist3 : dist3 = 150)
  (h_speed1 : speed1 = 50) (h_speed2 : speed2 = 80) (h_speed3 : speed3 = 120) :
  dist1 / speed1 + dist2 / speed2 + dist3 / speed3 = 3.5 :=
by
  sorry

end travel_time_total_l509_509883


namespace student_arrangements_l509_509855

theorem student_arrangements  :
  let num_students : ℕ := 6 in
  let venues : Finset ℕ := {1, 2, 3} in
  let venueA_students : ℕ := 1 in
  let venueB_students : ℕ := 2 in
  let venueC_students : ℕ := 3 in
  num_students = venueA_students + venueB_students + venueC_students →
  nat.choose num_students venueA_students *
  nat.choose (num_students - venueA_students) venueB_students *
  nat.choose (num_students - venueA_students - venueB_students) venueC_students = 60 :=
by
  intros num_students venues venueA_students venueB_students venueC_students h_sum_eq
  sorry

end student_arrangements_l509_509855


namespace ferrisWheelPeopleCount_l509_509033

/-!
# Problem Description

We are given the following conditions:
- The ferris wheel has 6.0 seats.
- It has to run 2.333333333 times for everyone to get a turn.

We need to prove that the total number of people who want to ride the ferris wheel is 14.
-/

def ferrisWheelSeats : ℕ := 6
def ferrisWheelRuns : ℚ := 2333333333 / 1000000000

theorem ferrisWheelPeopleCount :
  (ferrisWheelSeats : ℚ) * ferrisWheelRuns = 14 :=
by
  sorry

end ferrisWheelPeopleCount_l509_509033


namespace identify_translation_l509_509081

def phenomenon (x : String) : Prop :=
  x = "translational"

def option_A : Prop := phenomenon "rotational"
def option_B : Prop := phenomenon "rotational"
def option_C : Prop := phenomenon "translational"
def option_D : Prop := phenomenon "rotational"

theorem identify_translation :
  (¬ option_A) ∧ (¬ option_B) ∧ option_C ∧ (¬ option_D) :=
  by {
    sorry
  }

end identify_translation_l509_509081


namespace gcd_sum_inequality_l509_509339

open Nat

theorem gcd_sum_inequality (m n : ℕ) (hmn : m ≠ n) (hm_pos : m > 0) (hn_pos : n > 0) :
  gcd m n + gcd (m + 1) (n + 1) + gcd (m + 2) (n + 2) ≤ 2 * |m - n| + 1 := 
sorry

end gcd_sum_inequality_l509_509339


namespace locus_of_intersection_l509_509073

open Real

theorem locus_of_intersection 
  (A B C : Point)
  (d e : ℝ)
  (hA : A = ⟨d, e⟩)
  (hB : B = ⟨-d, -e⟩)
  (hCA_CB : dist A C ≠ dist B C)
  (t : ℝ)
  (φ_a φ_b : ℝ)
  (hφ_a : φ_a = (π - angle A C B) / 2 - t)
  (hφ_b : φ_b = (π + angle A C B) / 2 + t) :
  locus (λ t, intersection (rotate_line (line_through A C) A t) (rotate_line (line_through B C) B (-t))) = {p | p.x * p.y = d * e} := by
  sorry

end locus_of_intersection_l509_509073


namespace domain_and_range_of_p_l509_509613

variable {ι : Type} (h : ι → ℝ)

-- Defining the conditions
def h_domain : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def h_range : Set ℝ := {y | 2 ≤ y ∧ y ≤ 5}

-- Defining the function p
noncomputable def p (x : ℝ) := 2 - h(x - 2)

-- Facts derived from given conditions
axiom h_def (x : ℝ) : x ∈ h_domain → h(x) ∈ h_range

-- The proof problem statement
theorem domain_and_range_of_p :
  (∀ x, x ∈ (Set.Icc 3 6) ↔ (x - 2) ∈ h_domain) ∧
  (∀ y, y ∈ (Set.Icc (-3) 0) ↔ y = 2 - h((x - 2)) for some x ∈ (Set.Icc 3 6)) :=
sorry

end domain_and_range_of_p_l509_509613


namespace opposite_of_negative_2023_l509_509439

-- Define the opposite condition
def is_opposite (y x : Int) : Prop := y + x = 0

theorem opposite_of_negative_2023 : ∃ x : Int, is_opposite (-2023) x ∧ x = 2023 :=
by 
  use 2023
  sorry

end opposite_of_negative_2023_l509_509439


namespace problem_sequence_formulas_l509_509204

theorem problem_sequence_formulas :
  (∀ (n : ℕ), S_n = (n * (1 + a_1)) / 2 ∧ 1 + (n - 1) = 2 * (S_n))  ∧
  (b_n = log 2 (a_n) ∧ c_n = a_n * b_n)  →
  (∀ (n : ℕ), a_n = 2^(n-1)) ∧ (∀ (T_n : ℕ), T_n = (n-1) * 2^(n-1)) :=
sorry

end problem_sequence_formulas_l509_509204


namespace simplify_expression_l509_509019

theorem simplify_expression : 
  (1 / ((1 / ((1 / 2)^1)) + (1 / ((1 / 2)^3)) + (1 / ((1 / 2)^4)))) = (1 / 26) := 
by 
  sorry

end simplify_expression_l509_509019


namespace find_g_50_l509_509782

theorem find_g_50 (g : ℝ → ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - y * g x = g (x / y) + x - y) :
  g 50 = -24.5 :=
sorry

end find_g_50_l509_509782


namespace probability_infinite_to_2_l509_509307

noncomputable def normal_distribution := sorry -- Normally, we would define the normal distribution. We are skipping it here.

variable (xi : ℝ → ℝ) -- xi represents a random variable that follows a normal distribution.

axiom normal_xi : ∀ σ > 0, xi σ ∼ Normal(1, σ^2)

axiom probability_interval : ∀ σ > 0, P (0 < xi σ ∧ xi σ < 2) = 0.8

theorem probability_infinite_to_2 (σ : ℝ) (hσ : σ > 0) : P (xi σ ≤ 2) = 0.9 :=
sorry

end probability_infinite_to_2_l509_509307


namespace cassidy_number_of_posters_l509_509142

/-- Cassidy's current number of posters -/
def current_posters (C : ℕ) : Prop := 
  C + 6 = 2 * 14

theorem cassidy_number_of_posters : ∃ C : ℕ, current_posters C := 
  Exists.intro 22 sorry

end cassidy_number_of_posters_l509_509142


namespace cos2alpha_plus_sin2alpha_l509_509254

def point_angle_condition (x y : ℝ) (r : ℝ) (α : ℝ) : Prop :=
  x = -3 ∧ y = 4 ∧ r = 5 ∧ x^2 + y^2 = r^2

theorem cos2alpha_plus_sin2alpha (α : ℝ) (x y r : ℝ)
  (h : point_angle_condition x y r α) : 
  (Real.cos (2 * α) + Real.sin (2 * α)) = -31/25 :=
by
  sorry

end cos2alpha_plus_sin2alpha_l509_509254


namespace simplify_and_rationalize_denominator_l509_509401

noncomputable def problem (x : ℚ) := 
  x = 1 / (1 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_denominator (x : ℚ) (h: problem x) :
  x = (Real.sqrt 5 - 1) / 4 :=
by
  sorry

end simplify_and_rationalize_denominator_l509_509401


namespace max_consecutive_interesting_integers_l509_509364

open Nat

-- Definition of interesting number
def is_interesting (n : ℕ) : Prop :=
  ∃ p q : ℕ, prime p ∧ prime q ∧ p * q = n

-- Final problem statement
theorem max_consecutive_interesting_integers : 
  ∀ (seq : ℕ → ℕ) (n : ℕ), (∀ i < n, is_interesting (seq i)) → n ≤ 3 := 
sorry

end max_consecutive_interesting_integers_l509_509364


namespace minimum_distance_sum_l509_509653

noncomputable def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem minimum_distance_sum :
  let l1 := (4, -3, 16) in    -- coefficients (A, B, C) for line l1
  let l2 := (-1, 0, 0) in     -- special case for vertical line x = -1, represented as standard form (A, B, C) = (-1, 0, 0)
  let F := (1, 0) in          -- focus of the parabola y^2 = 4x
  let d1 := distance_point_to_line F.1 F.2 l1.1 l1.2 l1.3 in
  let d2 := 0 in              -- distance to vertical line x = -1 from focus F(1,0)
  d1 + d2 = 4 := by
begin
  let l1 := (4, -3, 16),
  let l2 := (-1, 0, 0),
  let F := (1, 0),
  let d1 := distance_point_to_line F.1 F.2 l1.1 l1.2 l1.3,
  have hd1 : d1 = 4 := sorry,
  have hd2 : d2 = 0 := by simp,
  calc
    d1 + d2 = 4 + 0 : by rw [hd1, hd2]
          ... = 4    : by simp
end

end minimum_distance_sum_l509_509653


namespace cos_beta_value_l509_509623

-- Definitions and assumptions
variable (α β : ℝ)
variable (h_alpha_acute : 0 < α ∧ α < π / 2)
variable (h_beta_acute : 0 < β ∧ β < π / 2)
variable (h_cos_sum : Real.cos (α + β) = 3 / 5)
variable (h_sin_alpha : Real.sin α = 5 / 13)

-- Statement of the theorem
theorem cos_beta_value :
  Real.cos β = 56 / 65 :=
  sorry

end cos_beta_value_l509_509623


namespace base_h_identity_l509_509175

theorem base_h_identity (h : ℕ) (h_gt_9 : h > 9) :
  h = 9 :=
  -- Define base_h addition results for each digit:
  let first_col := 4 + 5 = 9 in
  let second_col := (6 + 5 = 2 * h + h) -> 11 = h + 2 in
  let third_col := (7 + 8 = 6 * h + h) -> 15 = h + 6 in
  let fourth_col := (8 + 9 = 9 * h + h) -> 17 = h + 9 in
  let fifth_col := (9 + 6 = 7 * h + h) -> 15 = h + 7 in
sorry

end base_h_identity_l509_509175


namespace total_snake_owners_l509_509455

theorem total_snake_owners (only_snakes both_cats_snakes both_dogs_snakes all_categories : ℕ) :
  only_snakes = 10 →
  both_cats_snakes = 7 →
  both_dogs_snakes = 6 →
  all_categories = 2 →
  only_snakes + both_cats_snakes + both_dogs_snakes + all_categories = 25 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  rfl

end total_snake_owners_l509_509455


namespace cassidy_current_posters_l509_509144

theorem cassidy_current_posters : 
  ∃ P : ℕ, 
    (P + 6 = 2 * 14) → 
    P = 22 :=
begin
  sorry
end

end cassidy_current_posters_l509_509144


namespace categorize_numbers_l509_509941

theorem categorize_numbers :
  (∀ x ∈ ({-5/6, 0, -3.5, 1.2, 6} : set ℝ), x < 0 ↔ x ∈ ({-5/6, -3.5} : set ℝ)) ∧
  (∀ x ∈ ({-5/6, 0, -3.5, 1.2, 6} : set ℝ), x ≥ 0 ↔ x ∈ ({0, 1.2, 6} : set ℝ)) :=
by
  sorry

end categorize_numbers_l509_509941


namespace find_g_9_l509_509345

-- Define the function g
def g (a b c x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 7

-- Given conditions
variables (a b c : ℝ)

-- g(-9) = 9
axiom h : g a b c (-9) = 9

-- Prove g(9) = -23
theorem find_g_9 : g a b c 9 = -23 :=
by
  sorry

end find_g_9_l509_509345


namespace incenter_centroid_parallel_to_side_c_l509_509747

noncomputable def distance_from_incenter_to_base {a b c : ℝ} (h₁ : a + b = 2c) : ℝ := sorry
noncomputable def distance_from_centroid_to_base {a b c : ℝ} (h₁ : a + b = 2c) : ℝ := sorry

theorem incenter_centroid_parallel_to_side_c {a b c : ℝ} (h₁ : a + b = 2c) (h₂ : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    distance_from_incenter_to_base h₁ = distance_from_centroid_to_base h₁ :=
sorry

end incenter_centroid_parallel_to_side_c_l509_509747


namespace distinct_arrangements_on_3x3_grid_l509_509305

def is_valid_position (pos : ℤ × ℤ) : Prop :=
  0 ≤ pos.1 ∧ pos.1 < 3 ∧ 0 ≤ pos.2 ∧ pos.2 < 3

def rotations_equiv (pos1 pos2 : ℤ × ℤ) : Prop :=
  pos1 = pos2 ∨ pos1 = (2 - pos2.2, pos2.1) ∨ pos1 = (2 - pos2.1, 2 - pos2.2) ∨ pos1 = (pos2.2, 2 - pos2.1)

def distinct_positions_count (grid_size : ℕ) : ℕ :=
  10  -- given from the problem solution

theorem distinct_arrangements_on_3x3_grid : distinct_positions_count 3 = 10 := sorry

end distinct_arrangements_on_3x3_grid_l509_509305


namespace exists_face_with_projection_on_itself_l509_509532

variables {P : Type} [convex_polyhedron P] (point_in_polyhedron : P → convex_polyhedron P → Prop)

theorem exists_face_with_projection_on_itself (P : P) (poly : convex_polyhedron P)
  (hP : point_in_polyhedron P poly) :
  ∃ F ∈ faces poly, (orthogonal_projection F P) ∈ F :=
sorry

end exists_face_with_projection_on_itself_l509_509532


namespace trapezoidAreaCorrect_l509_509087

-- Definition of the isosceles trapezoid
structure IsoscelesTrapezoid where
  side : ℝ
  base1 : ℝ
  base2 : ℝ
  height : ℝ

-- Specific instance of the isosceles trapezoid
def trapezoid : IsoscelesTrapezoid :=
  { side := 5, base1 := 9, base2 := 15, height := 4 }

-- Area calculation function
def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  (1 / 2) * (t.base1 + t.base2) * t.height

-- Prove the area is 48
theorem trapezoidAreaCorrect : trapezoidArea trapezoid = 48 :=
  by
    -- Skipping the proof for now
    sorry

end trapezoidAreaCorrect_l509_509087


namespace minimize_total_cost_l509_509036

noncomputable def total_cost (v : ℝ) : ℝ :=
  16 * v + 6400 / v

theorem minimize_total_cost (a : ℝ) (h : 0 < a) :
  (a ≤ 20 → ∀ v : ℝ, 0 < v ∧ v ≤ a → total_cost v = total_cost a) ∧
  (a > 20 → total_cost 20 = infi (λ v, total_cost v)) :=
by
  sorry

end minimize_total_cost_l509_509036


namespace sin_alpha_minus_pi_over_3_l509_509248

theorem sin_alpha_minus_pi_over_3 
  (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : tan (α / 2) + cot (α / 2) = 5 / 2) :
  sin (α - π / 3) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end sin_alpha_minus_pi_over_3_l509_509248


namespace min_value_of_sum_l509_509655

-- Define the parabola
def parabola (x y : ℝ) := y^2 = -4 * x

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define point A
def A : ℝ × ℝ := (-2, 1)

-- Define what it means for a point to be on the parabola
def on_parabola (P : ℝ × ℝ) := parabola P.1 P.2

-- Define the distance function
def dist (P Q : ℝ × ℝ) := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

-- Define the directrix of the parabola
def directrix (x : ℝ) := x = 1

-- Prove that the minimum value of |PF| + |PA| is 3
theorem min_value_of_sum (P : ℝ × ℝ) (hP : on_parabola P) : 
  ∃ D : ℝ × ℝ, directrix D.1 ∧ dist P F + dist P A = 3 :=
sorry

end min_value_of_sum_l509_509655


namespace my_expression_is_minus_three_l509_509915

noncomputable def my_expression : ℝ :=
  6 * Real.sin (Real.pi / 4) - |1 - Real.sqrt 2| - Real.sqrt 8 * ((Real.pi - 2023) ^ 0) - (1 / 2) ^ (-2)

theorem my_expression_is_minus_three :
  my_expression = -3 :=
by
  sorry

end my_expression_is_minus_three_l509_509915


namespace sqrt_div_five_l509_509489

theorem sqrt_div_five : real.sqrt 625 / 5 = 5 := 
by sorry

end sqrt_div_five_l509_509489


namespace slower_train_speed_l509_509473

noncomputable def speed_of_slower_train (V_f : ℕ) (L_1 L_2 T : ℕ) : ℚ :=
let distance := (L_1 + L_2) / 1000 -- converting meters to kilometers
let time := T / 3600 -- converting seconds to hours
in V_f - distance / time

theorem slower_train_speed
  (L₁ L₂ T : ℕ)
  (V_f := 46) :
  speed_of_slower_train V_f L₁ L₂ T = 40.05 := by
  -- Given the lengths of the trains in meters and the time in seconds
  have h₁ : L₁ = 200 := by rfl
  have h₂ : L₂ = 150 := by rfl
  have h₃ : T = 210 := by rfl
  -- Use the speed of the faster train and the conditions
  rw [h₁, h₂, h₃]
  -- Apply the conditions to the defined speed_of_slower_train function
  have h₄ : speed_of_slower_train V_f 200 150 210 = 40.05 := by
    dsimp [speed_of_slower_train]
    norm_num
  exact h₄

end slower_train_speed_l509_509473


namespace emmalyn_earnings_l509_509575

theorem emmalyn_earnings :
  let rate_per_meter := 0.20
  let number_of_fences := 50
  let length_of_each_fence := 500
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * rate_per_meter
  total_income = 5000 :=
by
  sorry

end emmalyn_earnings_l509_509575


namespace square_area_l509_509875

theorem square_area (x : ℝ) (PR : ℝ) (H_PR : PR = 90) (H_eq : PR^2 = (2 * x)^2 + x^2) :
  x^2 = 1620 :=
by
  have H_sqrt : 90^2 = 8100 := by norm_num,
  rw [H_PR, H_sqrt] at H_eq,
  calc
    8100 = 5 * x^2 : by ring_nf
      ... = 1620  : by linarith

end square_area_l509_509875


namespace simplify_rationalize_l509_509413

theorem simplify_rationalize
  : (1 / (1 + (1 / (Real.sqrt 5 + 2)))) = ((Real.sqrt 5 + 1) / 4) := 
sorry

end simplify_rationalize_l509_509413


namespace product_eq_5_l509_509711

noncomputable def product_of_positive_integers_less_10 :=
  ∏ k in {k ∈ Finset.range 10 | ∀ (v : ℕ → ℝ × ℝ),
    (∀ i < 5, ∥v i∥ = 1) ∧ (v 0 = (1, 0)) ∧
    (v 1 = (x2 k, y2 k)) ∧ (v 2 = (x3 k, y3 k)) ∧
    (v 3 = (x4 k, y4 k)) ∧ (v 4 = (x5 k, y5 k))
      ⟹ (x1^k + x2^k + x3^k + x4^k + x5^k = y1^k + y2^k + y3^k + y4^k + y5^k)}, k

theorem product_eq_5 : product_of_positive_integers_less_10 = 5 :=
sorry

end product_eq_5_l509_509711


namespace find_a_l509_509294

open Real

def line1_perpendicular_to_line2 (a : ℝ) : Prop := 
  let l1 := 2*x - a*y - 1 = 0
  let l2 := x + 2*y = 0
  ∃x y, l1 ∧ l2 ∧ (2 * 1 + (-a) * 2 = 0)

theorem find_a :
  ∃ a : ℝ, line1_perpendicular_to_line2 a ∧ a = 1 :=
by
  use 1
  unfold line1_perpendicular_to_line2
  sorry

end find_a_l509_509294


namespace muffin_banana_cost_ratio_l509_509024

variables (m b c : ℕ) -- costs of muffin, banana, and cookie respectively
variables (susie_cost calvin_cost : ℕ)

-- Conditions
def susie_cost_eq : Prop := susie_cost = 5 * m + 4 * b + 2 * c
def calvin_cost_eq : Prop := calvin_cost = 3 * (5 * m + 4 * b + 2 * c)
def calvin_cost_eq_reduced : Prop := calvin_cost = 3 * m + 20 * b + 6 * c
def cookie_cost_eq : Prop := c = 2 * b

-- Question and Answer
theorem muffin_banana_cost_ratio
  (h1 : susie_cost_eq m b c susie_cost)
  (h2 : calvin_cost_eq m b c calvin_cost)
  (h3 : calvin_cost_eq_reduced m b c calvin_cost)
  (h4 : cookie_cost_eq b c)
  : m = 4 * b / 3 :=
sorry

end muffin_banana_cost_ratio_l509_509024


namespace equilateral_triangle_unique_lines_l509_509686

theorem equilateral_triangle_unique_lines :
  ∀ (Δ : Type) [triangle Δ], (∀ (a b c : Δ), equilateral a b c ∧ distinct_lines_from_each_vertex_to_opposite_side a b c) →
  total_distinct_lines Δ = 3 :=
  sorry

end equilateral_triangle_unique_lines_l509_509686


namespace square_decomposition_possible_l509_509726

theorem square_decomposition_possible (K K' : square) (h1 : K.is_in_plane) (h2 : K'.is_in_plane)
  (h3 : K.side_length = K'.side_length) :
  ∃ (T : list triangle) (t : list (triangle → triangle)),   
    (T.pairwise_disjoint ∧ (K' = K.translate_using_triangles T t)) :=
sorry

end square_decomposition_possible_l509_509726


namespace molecular_weight_C4H10_l509_509481

theorem molecular_weight_C4H10
  (atomic_weight_C : ℝ)
  (atomic_weight_H : ℝ)
  (C4H10_C_atoms : ℕ)
  (C4H10_H_atoms : ℕ)
  (moles : ℝ) : 
  atomic_weight_C = 12.01 →
  atomic_weight_H = 1.008 →
  C4H10_C_atoms = 4 →
  C4H10_H_atoms = 10 →
  moles = 6 →
  (C4H10_C_atoms * atomic_weight_C + C4H10_H_atoms * atomic_weight_H) * moles = 348.72 :=
by
  sorry

end molecular_weight_C4H10_l509_509481


namespace fraction_of_budget_is_31_percent_l509_509707

def coffee_pastry_cost (B : ℝ) (c : ℝ) (p : ℝ) :=
  c = 0.25 * (B - p) ∧ p = 0.10 * (B - c)

theorem fraction_of_budget_is_31_percent (B c p : ℝ) (h : coffee_pastry_cost B c p) :
  c + p = 0.31 * B :=
sorry

end fraction_of_budget_is_31_percent_l509_509707


namespace opposite_of_neg_2023_is_2023_l509_509441

theorem opposite_of_neg_2023_is_2023 :
  opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_is_2023_l509_509441


namespace first_term_geometric_series_l509_509897

theorem first_term_geometric_series (a r S : ℝ) (h1 : r = -1/3) (h2 : S = 9)
  (h3 : S = a / (1 - r)) : a = 12 :=
sorry

end first_term_geometric_series_l509_509897


namespace expected_value_of_smallest_seven_selected_from_sixty_three_l509_509749

noncomputable def expected_value_smallest_selected (n r : ℕ) : ℕ :=
  (n + 1) / (r + 1)

theorem expected_value_of_smallest_seven_selected_from_sixty_three :
  expected_value_smallest_selected 63 7 = 8 :=
by
  sorry -- Proof is omitted as per instructions

end expected_value_of_smallest_seven_selected_from_sixty_three_l509_509749


namespace balcony_more_than_orchestra_l509_509125

-- Conditions
def total_tickets (O B : ℕ) : Prop := O + B = 340
def total_cost (O B : ℕ) : Prop := 12 * O + 8 * B = 3320

-- The statement we need to prove based on the conditions
theorem balcony_more_than_orchestra (O B : ℕ) (h1 : total_tickets O B) (h2 : total_cost O B) :
  B - O = 40 :=
sorry

end balcony_more_than_orchestra_l509_509125


namespace combinatorial_sum_l509_509189

theorem combinatorial_sum (n : ℕ) 
  (h : ∑ i in Finset.range (n+1), (2^i * (Nat.choose n i)) = 81) : 
  ∑ i in Finset.range (n+1), (Nat.choose n i) = 16 :=
sorry

end combinatorial_sum_l509_509189


namespace find_f_expression_l509_509192

theorem find_f_expression (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 - real.cos x) = real.sin x * real.sin x) : 
  ∀ x : ℝ, f x = 2 * x - x * x :=
by
  sorry

end find_f_expression_l509_509192


namespace intersecting_line_circle_cos_l509_509320

noncomputable def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (-1/2 + (Real.sqrt 2) / 2 * t, 1 + (Real.sqrt 2) / 2 * t)

def circle_C (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + (y - 1) ^ 2 = 5

def polar_line_l (ρ θ : ℝ) : Prop :=
  ρ * Real.sin θ = ρ * Real.cos θ + 2

def polar_circle_C (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.cos θ + 2 * Real.sin θ

def cos_AOB (θ_A θ_B : ℝ) : ℝ :=
  Real.sin θ_B / Real.sqrt (1 + (Real.tan θ_B) ^ 2)

theorem intersecting_line_circle_cos
  (θ_A θ_B : ℝ) :
  parametric_line_l t → 
  circle_C x y →
  polar_line_l ρ θ →
  polar_circle_C ρ θ → 
  θ_A = Real.pi / 2 ∧ Real.tan θ_B = 3 →
  cos_AOB θ_A θ_B = (3 * Real.sqrt 10) / 10 :=
by
  sorry

end intersecting_line_circle_cos_l509_509320


namespace ratio_x_w_l509_509190

variable {x y z w : ℕ}

theorem ratio_x_w (h1 : x / y = 24) (h2 : z / y = 8) (h3 : z / w = 1 / 12) : x / w = 1 / 4 := by
  sorry

end ratio_x_w_l509_509190


namespace fraction_of_full_tank_used_l509_509904

-- Define the initial conditions as per the problem statement
def speed : ℝ := 50 -- miles per hour
def time : ℝ := 5   -- hours
def miles_per_gallon : ℝ := 30
def full_tank_capacity : ℝ := 15 -- gallons

-- We need to prove that the fraction of gasoline used is 5/9
theorem fraction_of_full_tank_used : 
  ((speed * time) / miles_per_gallon) / full_tank_capacity = 5 / 9 := by
sorry

end fraction_of_full_tank_used_l509_509904


namespace largest_even_multiple_of_15_lt_500_l509_509819

theorem largest_even_multiple_of_15_lt_500 : 
  ∃ (n : ℕ), (15 * n < 500) ∧ (15 * n % 2 = 0) ∧ ∀ (m : ℕ), (15 * m < 500) ∧ (15 * m % 2 = 0) → 15 * m ≤ 15 * n :=
begin
  sorry
end

end largest_even_multiple_of_15_lt_500_l509_509819


namespace different_genre_book_pairs_l509_509288

theorem different_genre_book_pairs :
  ∃ (mystery fantasy biography : ℕ), 
    mystery = 4 ∧ 
    fantasy = 4 ∧ 
    biography = 3 ∧ 
    (mystery * fantasy + mystery * biography + fantasy * biography = 40) :=
by {
  use [4, 4, 3],
  split; 
  { refl }, 
  split; 
  { refl },
  split; 
  { refl },
  sorry
}

end different_genre_book_pairs_l509_509288


namespace find_the_number_l509_509530

theorem find_the_number (x : ℝ) (h : 8 * x + 64 = 336) : x = 34 :=
by
  sorry

end find_the_number_l509_509530


namespace smallest_number_of_oranges_l509_509839

theorem smallest_number_of_oranges (n : ℕ) (total_oranges : ℕ) :
  (total_oranges > 200) ∧ total_oranges = 15 * n - 6 ∧ n ≥ 14 → total_oranges = 204 :=
by
  sorry

end smallest_number_of_oranges_l509_509839


namespace parallelogram_symmetry_center_l509_509110

theorem parallelogram_symmetry_center (P : Type) [parallelogram P] :
  (∀ p ∈ P, rotate_180 deg_intersection(p) = p) → symmetrical_center P :=
sorry

end parallelogram_symmetry_center_l509_509110


namespace cory_cleans_more_minutes_l509_509752

theorem cory_cleans_more_minutes (x C : ℕ) (Richard_time : ℕ := 22) (total_time : ℕ := 136) :
  C = Richard_time + x →
  let Blake_time := C - 4 in
  Richard_time + C + Blake_time = total_time →
  C - Richard_time = 37 :=
by
  intros hC hTotal
  have h1 : 2 * C + 18 = 136 := by
    calc
      Richard_time + C + (C - 4) = 136 : by exact hTotal
      22 + C + C - 4 = 136 : by rw [Richard_time]
      2 * C + 18 = 136 : by linarith
  have h2 : 2 * C = 118 := by linarith
  have h3 : C = 59 := by linarith
  calc
    C - Richard_time = 37 : by
      rw [h3, Richard_time]
      linarith

end cory_cleans_more_minutes_l509_509752


namespace centroid_line_distances_l509_509064

section centroid_line_problem

variables {A B C O A1 B1 C1 : Point}
variables [triangle ABC]
variables (centroid_O : centroid O ABC)
variables (line_l : ∀ {P : Point}, P ∈ l → ∃ {Q R : Point}, Q ∈ AB ∧ R ∈ BC ∧ P = O)
variables (perpendicular_AA1 : ∀ {P : Point}, P ∈ l → ⟂ AA1 P)
variables (perpendicular_BB1 : ∀ {P : Point}, P ∈ l → ⟂ BB1 P)
variables (perpendicular_CC1 : ∀ {P : Point}, P ∈ l → ⟂ CC1 P)

theorem centroid_line_distances :
  |distance A A1| + |distance C C1| = |distance B B1| :=
  sorry

end centroid_line_problem

end centroid_line_distances_l509_509064


namespace option_c_not_always_true_l509_509627

theorem option_c_not_always_true (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  ¬ (∀ a b, a > 0 → b > 0 → a + b = 2 → sqrt a + sqrt b ≤ sqrt 2) :=
sorry

end option_c_not_always_true_l509_509627


namespace segment_intersects_squares_and_circles_l509_509556

-- Define the conditions
def is_circle_at_point (x y : ℝ) : Prop := ∃ r : ℝ, r = 1 / 8 ∧ (∃ center : ℝ × ℝ, center = (x, y))
def is_square_at_point (x y : ℝ) : Prop := ∃ s : ℝ, s = 1 / 4 ∧ (∃ corners : list (ℝ × ℝ),
                                                              corners.length = 4 ∧ 
                                                              (∀ corner ∈ corners, 
                                                               abs (corner.1 - x) ≤ s / 2 
                                                               ∧ 
                                                               abs (corner.2 - y) ≤ s / 2))

-- Define the line segment
def line_segment_intersects (p1 p2 : ℝ × ℝ) (intersect : ℝ × ℝ → Prop) : Prop :=
  ∃ x, x ∈ segment p1 p2 ∧ intersect x

-- Main theorem
theorem segment_intersects_squares_and_circles :
  ∀ (lattice_points : set (ℝ × ℝ)), 
  (∀ p ∈ lattice_points, ∃ i j : ℤ, (i : ℝ, j : ℝ) = p) → 
  (∀ p ∈ lattice_points, 
    line_segment_intersects (0, 0) (803, 345) (λ p, is_circle_at_point p.1 p.2) 
    ∨ 
    line_segment_intersects (0, 0) (803, 345) (λ p, is_square_at_point p.1 p.2)) → 
  lattice_points.card = 24 → 
  24 + 24 = 48 :=
by 
  intros lattice_points lattice_points_condition intersect_condition lattice_points_card
  have m := 24 
  have n := 24 
  exact (m + n) = 48
sory

end segment_intersects_squares_and_circles_l509_509556


namespace inequality_sqrt_ab_l509_509191

theorem inequality_sqrt_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  2 / (1 / a + 1 / b) ≤ Real.sqrt (a * b) :=
sorry

end inequality_sqrt_ab_l509_509191


namespace exact_value_range_l509_509793

theorem exact_value_range (a : ℝ) (h : |170 - a| < 0.5) : 169.5 ≤ a ∧ a < 170.5 :=
by
  sorry

end exact_value_range_l509_509793


namespace range_of_a_l509_509675

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (ae ^ x₁ - x₁ - 2a = 0) ∧ (ae ^ x₂ - x₂ - 2a = 0)) ↔ 0 < a := 
sorry

end range_of_a_l509_509675


namespace larger_number_is_34_l509_509015

theorem larger_number_is_34 (a b : ℕ) (h1 : a > b) (h2 : (a + b) + (a - b) = 68) : a = 34 := 
by
  sorry

end larger_number_is_34_l509_509015


namespace angle_EOA_of_convex_quadrilateral_l509_509612

theorem angle_EOA_of_convex_quadrilateral
  {A B C D E O : Type}
  (h_convex : convex_quadrilateral A B C D)
  (h_lengths : dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A)
  (h_angle_ACD : angle A C D = 10)
  (omega : circle circumscribing_triangle B C D centered_at O)
  (h_intersection : line_through D A intersects_circle omega again_at E) :
  angle E O A = 65 := sorry

end angle_EOA_of_convex_quadrilateral_l509_509612


namespace summation_problem_l509_509961

noncomputable def poly := (z : ℂ) → (x - z)^3

theorem summation_problem (z : ℕ → ℂ) (distinct : ∀ i j, i ≠ j → z i ≠ z j)
  (h : (∏ i in finset.range 673, poly (z i) = λ x, x^2019 + 20*x^2018 + 19*x^2017 + g x))
  (g : polynomial ℂ) :
  let S := ∑ (1 ≤ j < k ≤ 673), z j * z k in
  |S| = 1067 / 9 ∧ nat.gcd 1067 9 = 1 ∧ 1067 + 9 = 1076 := sorry

end summation_problem_l509_509961


namespace decreasing_function_range_l509_509296

theorem decreasing_function_range (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → (2 * a - 1) ^ x1 > (2 * a - 1) ^ x2) :
  1 / 2 < a ∧ a < 1 :=
by
  sorry

end decreasing_function_range_l509_509296


namespace angle_at_third_vertex_sixty_degrees_l509_509540

theorem angle_at_third_vertex_sixty_degrees
  (A B C O M : Type)
  (h_triangle : IsTriangle A B C)
  (h_incircle_center : IsIncenter A B C O)
  (h_orthocenter : IsOrthocenter A B C M)
  (h_cyclic : Cyclic [A, B, O, M]) :
  angle_at_vertex C = 60 :=
sorry

end angle_at_third_vertex_sixty_degrees_l509_509540


namespace all_of_the_above_were_used_as_money_l509_509824

-- Defining the conditions that each type was used as money
def gold_used_as_money : Prop := True
def stones_used_as_money : Prop := True
def horses_used_as_money : Prop := True
def dried_fish_used_as_money : Prop := True
def mollusk_shells_used_as_money : Prop := True

-- The statement to prove
theorem all_of_the_above_were_used_as_money :
  gold_used_as_money ∧
  stones_used_as_money ∧
  horses_used_as_money ∧
  dried_fish_used_as_money ∧
  mollusk_shells_used_as_money ↔
  (∀ x, (x = "gold" ∨ x = "stones" ∨ x = "horses" ∨ x = "dried fish" ∨ x = "mollusk shells") → 
  (x = "gold" ∧ gold_used_as_money) ∨ 
  (x = "stones" ∧ stones_used_as_money) ∨ 
  (x = "horses" ∧ horses_used_as_money) ∨ 
  (x = "dried fish" ∧ dried_fish_used_as_money) ∨ 
  (x = "mollusk shells" ∧ mollusk_shells_used_as_money)) :=
by
  sorry

end all_of_the_above_were_used_as_money_l509_509824


namespace find_closest_point_on_plane_l509_509954

open Real

def plane (x y z : ℝ) : Prop := 5 * x - 2 * y + 6 * z = 40

def closest_point (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ → ℝ :=
  λ P, dist A P

noncomputable def closest_point_on_plane : ℝ × ℝ × ℝ :=
  (138 / 65, -73 / 65, 274 / 65)

theorem find_closest_point_on_plane :
  ∀ (x y z : ℝ), plane x y z → 
  (closest_point (2, -1, 4) (x, y, z) ≥ closest_point (2, -1, 4) closest_point_on_plane) :=
by
  sorry

end find_closest_point_on_plane_l509_509954


namespace germination_rate_in_second_plot_l509_509180

theorem germination_rate_in_second_plot (num_seeds_plot1 : ℕ) (num_seeds_plot2 : ℕ)
  (germination_rate_plot1 : ℝ) (total_germination_rate : ℝ) :
  num_seeds_plot1 = 300 →
  num_seeds_plot2 = 200 →
  germination_rate_plot1 = 0.25 →
  total_germination_rate = 28.999999999999996 →
  let total_seeds := num_seeds_plot1 + num_seeds_plot2 in
  let germinated_plot1 := germination_rate_plot1 * num_seeds_plot1 in
  let total_germinated := (total_germination_rate / 100) * total_seeds in
  let germinated_plot2 := total_germinated - germinated_plot1 in
  let germination_rate_plot2 := (germinated_plot2 / num_seeds_plot2) * 100 in
  germination_rate_plot2 = 35 :=
by
  intros
  let total_seeds := num_seeds_plot1 + num_seeds_plot2
  let germinated_plot1 := germination_rate_plot1 * num_seeds_plot1
  let total_germinated := (total_germination_rate / 100) * total_seeds
  let germinated_plot2 := total_germinated - germinated_plot1
  let germination_rate_plot2 := (germinated_plot2 / num_seeds_plot2) * 100
  exact sorry

end germination_rate_in_second_plot_l509_509180


namespace minimum_value_of_f_l509_509949

open Real

noncomputable def f (x : ℝ) : ℝ := 3^x - 9^x

theorem minimum_value_of_f : ∀ x : ℝ, f(x) ≥ (1 / 4) :=
by
  sorry

end minimum_value_of_f_l509_509949


namespace intersection_line_through_circles_l509_509278

def circle1_equation (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2_equation (x y : ℝ) : Prop := x^2 + y^2 + 2 * x + 2 * y - 14 = 0

theorem intersection_line_through_circles : 
  (∀ x y : ℝ, circle1_equation x y → circle2_equation x y → x + y - 2 = 0) :=
by
  intros x y h1 h2
  sorry

end intersection_line_through_circles_l509_509278


namespace num_unique_seven_digit_integers_l509_509283

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

noncomputable def unique_seven_digit_integers : ℕ :=
  factorial 7 / (factorial 2 * factorial 2 * factorial 2)

theorem num_unique_seven_digit_integers : unique_seven_digit_integers = 630 := by
  sorry

end num_unique_seven_digit_integers_l509_509283


namespace vanessa_numbers_unique_l509_509474

theorem vanessa_numbers_unique :
  ∃ (S : Finset ℕ), S.card = 50 ∧ (∀ x ∈ S, x < 100) ∧ 
  (∀ a b ∈ S, a ≠ b → a + b ≠ 99 ∧ a + b ≠ 100) ∧ 
  (S = Finset.range' 50 50) :=
by {
  sorry
}

end vanessa_numbers_unique_l509_509474


namespace polynomial_solution_exists_l509_509174

theorem polynomial_solution_exists (p : ℝ → ℝ) (h₀ : p 3 = 10)
    (h₁ : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) :
    p = λ x, x^2 + 1 :=
by
  sorry

end polynomial_solution_exists_l509_509174


namespace particle_final_position_l509_509868

theorem particle_final_position :
  let initial_position := (5 : ℝ, 0 : ℝ)
  let move (pos : ℝ × ℝ) :=
    let rotated_pos : ℝ × ℝ := (cos (π/4) * pos.1 - sin (π/4) * pos.2, sin (π/4) * pos.1 + cos (π/4) * pos.2)
    (rotated_pos.1 + 10, rotated_pos.2)
  let final_position := (Nat.iterate move 150 initial_position)
  |final_position.1| + |final_position.2| ≤ 19 := by
  sorry

end particle_final_position_l509_509868


namespace eighth_grade_students_l509_509470

def avg_books (total_books : ℕ) (num_students : ℕ) : ℚ :=
  total_books / num_students

theorem eighth_grade_students (x : ℕ) (y : ℕ)
  (h1 : x + y = 1800)
  (h2 : y = x - 150)
  (h3 : avg_books x 1800 = 1.5 * avg_books (x - 150) 1800) :
  y = 450 :=
by {
  sorry
}

end eighth_grade_students_l509_509470


namespace find_b_l509_509881

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 2 * x^2 - b * x + 3
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := 2 * x^2 + b * x + 3

theorem find_b : (b : ℝ) (h : ∀ x : ℝ, g x b = f (x + 6) b) → b = 12 :=
by
  intros b h
  sorry

end find_b_l509_509881


namespace combined_5_eq_10_l509_509504

-- Define the sequences a_n and b_n
def a (n : ℕ) : ℕ := 2 * n
def b (n : ℕ) : ℕ := 16 - 2 * n

-- Define the combined sequence where the even-indexed terms come from a and the odd-indexed terms come from b
def combined (n : ℕ) : ℕ :=
  if even n then a (n / 2) else b (n / 2)

-- Prove that the 6th term (index 5) of the combined sequence is 10
theorem combined_5_eq_10 : combined 5 = 10 := by
  sorry

end combined_5_eq_10_l509_509504


namespace find_coordinates_and_area_l509_509997

variables {x y : ℝ}

def ellipse_eq (x y : ℝ) : Prop := 
  (x^2 / 100) + (y^2 / 36) = 1

def perpendicular_distance (F1 F2 P : ℝ × ℝ) : Prop :=
  let d1 := (P.1 - F1.1)^2 + (P.2 - F1.2)^2
  let d2 := (P.1 - F2.1)^2 + (P.2 - F2.2)^2
  PF_1 ⊥ PF_2 → F1.1 ≠ F2.1 → F1.2 ≠ F2.2 →
  d1 + d2 = 256

theorem find_coordinates_and_area (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) 
  (h : ellipse_eq P.1 P.2) (h_perpendicular : perpendicular_distance F1 F2 P) :
  (P = (5 / 2 * real.sqrt 7, 9 / 2 * real.sqrt 7)) ∧ 
  (∀ y, y = (9 * real.sqrt 7) / 2 → 
   let area := 1/2 * 16 * y in area = 36 * real.sqrt 7) :=
by sorry

end find_coordinates_and_area_l509_509997


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l509_509231

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l509_509231


namespace parallel_planes_condition_l509_509717

theorem parallel_planes_condition (α β : Plane) (L : Line) (hα : α ≠ β)
  (hβ_perp : perpendicular α L) (hα_perp : perpendicular β L) : parallel α β :=
sorry

end parallel_planes_condition_l509_509717


namespace find_p_from_circle_and_parabola_tangency_l509_509637

theorem find_p_from_circle_and_parabola_tangency :
  (∃ x y : ℝ, (x^2 + y^2 - 6*x - 7 = 0) ∧ (y^2 = 2*p * x) ∧ p > 0) →
  p = 2 :=
by {
  sorry
}

end find_p_from_circle_and_parabola_tangency_l509_509637


namespace player_a_winning_strategy_l509_509113

-- Define the context of the problem: convex 100-gon, point X inside it, etc.
structure Polygon :=
(vertices : Finset (Point))
(orthogonality : ∀ p1 p2 ∈ vertices, p1 ≠ p2)

structure Game :=
(polygon : Polygon)
(point_X : Point)
(not_on_diagonal : ∀ d ∈ polygon.diagonals, point_X ∉ d)

def Player := A | B

inductive Turn
| PlayerA : Player
| PlayerB : Player

-- Define the initial game state and turns
structure GameState :=
(colored_vertices : Finset (Point))
(current_turn : Turn)

-- game state update rules
def update_game_state (state : GameState) (new_vertex : Point) : GameState :=
{ colored_vertices := state.colored_vertices.insert new_vertex,
  current_turn := match state.current_turn with
                 | Turn.PlayerA => Turn.PlayerB
                 | Turn.PlayerB => Turn.PlayerA
                 end }

-- Define the winning condition
def winning_strategy (game : Game) (state : GameState) : Prop :=
∀ new_vertex ∈ game.polygon.vertices \ state.colored_vertices,
update_game_state state new_vertex ≠ ∅

-- Prove that Player A has a winning strategy
theorem player_a_winning_strategy (game : Game) (state : GameState) :
  ∃ strat : Point ∈ game.polygon.vertices, winning_strategy game state :=
sorry

end player_a_winning_strategy_l509_509113


namespace minimum_value_f_l509_509435

def f (x : ℝ) : ℝ := (x + 1) * (x + 2) * (x + 3) * (x + 4) + 35

theorem minimum_value_f : ∃ x : ℝ, ∀ y : ℝ, f(y) ≥ f(x) ∧ f(x) = 34 :=
sorry

end minimum_value_f_l509_509435


namespace simplify_and_evaluate_expr_l509_509017

theorem simplify_and_evaluate_expr (x : ℝ) (h : x = Real.sqrt 2) :
    (x - 1) / x / (x - 1 / x) = Real.sqrt 2 - 1 :=
by
  rw h
  sorry

end simplify_and_evaluate_expr_l509_509017


namespace trapezoid_acute_angle_l509_509434

theorem trapezoid_acute_angle (ABCD : Trapezoid) (AD BC : ℝ) (AD_gt_BC : AD > BC)
  (AB_perp_AD : Perpendicular AB AD) (M : Point) (M_midpoint_CD : is_midpoint M CD)
  (isosceles_triangles : DividesIntoIsoscelesTriangles M ABCD) :
  acute_angle ABCD = 72 :=
sorry

end trapezoid_acute_angle_l509_509434


namespace raw_and_central_moments_l509_509533

noncomputable def pdf (x : ℝ) : ℝ :=
if 0 < x ∧ x < 2 then 0.5 * x else 0

def v_k (k : ℕ) : ℝ :=
∫ x in 0..2, x^k * pdf x

def μ2 := v_k 2 - (v_k 1)^2
def μ3 := v_k 3 - 3 * (v_k 1) * (v_k 2) + 2 * (v_k 1)^3
def μ4 := v_k 4 - 4 * (v_k 1) * (v_k 3) + 6 * (v_k 1)^2 * (v_k 2) - 3 * (v_k 1)^4

theorem raw_and_central_moments :
  v_k 1 = 4 / 3 ∧
  v_k 2 = 2 ∧
  v_k 3 = 16 / 5 ∧
  v_k 4 = 16 / 3 ∧
  μ2 = 2 / 9 ∧
  μ3 = -8 / 135 ∧
  μ4 = 16 / 135 := by
  sorry

end raw_and_central_moments_l509_509533


namespace hyperbola_slope_of_asymptote_positive_value_l509_509784

noncomputable def hyperbola_slope_of_asymptote (x y : ℝ) : ℝ :=
  if h : (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4)
  then (Real.sqrt 5) / 2
  else 0

-- Statement of the mathematically equivalent proof problem
theorem hyperbola_slope_of_asymptote_positive_value :
  ∃ x y : ℝ, (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) ∧
  hyperbola_slope_of_asymptote x y = (Real.sqrt 5) / 2 :=
by
  sorry

end hyperbola_slope_of_asymptote_positive_value_l509_509784


namespace sam_quarters_mowing_lawns_l509_509011

-- Definitions based on the given conditions
def pennies : ℕ := 9
def total_amount_dollars : ℝ := 1.84
def penny_value_dollars : ℝ := 0.01
def quarter_value_dollars : ℝ := 0.25

-- Theorem statement that Sam got 7 quarters given the conditions
theorem sam_quarters_mowing_lawns : 
  (total_amount_dollars - pennies * penny_value_dollars) / quarter_value_dollars = 7 := by
  sorry

end sam_quarters_mowing_lawns_l509_509011


namespace first_term_geometric_series_l509_509898

theorem first_term_geometric_series (a r S : ℝ) (h1 : r = -1/3) (h2 : S = 9)
  (h3 : S = a / (1 - r)) : a = 12 :=
sorry

end first_term_geometric_series_l509_509898


namespace Iesha_num_books_about_school_l509_509289

theorem Iesha_num_books_about_school (total_books sports_books : ℕ) (h1 : total_books = 58) (h2 : sports_books = 39) : total_books - sports_books = 19 :=
by
  sorry

end Iesha_num_books_about_school_l509_509289


namespace length_AE_l509_509776

open Real

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem length_AE :
  let A := (0, 3) in
  let B := (4, 0) in
  let C := (3, 3) in
  let D := (1, 0) in
  let E_x := 3 / 2 in
  let E_y := 15 / 8 in
  let E := (E_x, E_y) in
  distance A E = 15 * sqrt 13 / 8 :=
by
  sorry

end length_AE_l509_509776


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l509_509230

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l509_509230


namespace total_income_l509_509570

-- Definitions of conditions
def charge_per_meter : ℝ := 0.2
def number_of_fences : ℝ := 50
def length_of_each_fence : ℝ := 500

-- Theorem statement
theorem total_income :
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * charge_per_meter
  total_income = 5000 := 
by
  sorry

end total_income_l509_509570


namespace all_of_the_above_used_as_money_l509_509831

-- Definition to state that each item was used as money
def gold_used_as_money : Prop := true
def stones_used_as_money : Prop := true
def horses_used_as_money : Prop := true
def dried_fish_used_as_money : Prop := true
def mollusk_shells_used_as_money : Prop := true

-- Statement that all of the above items were used as money
theorem all_of_the_above_used_as_money : gold_used_as_money ∧ stones_used_as_money ∧ horses_used_as_money ∧ dried_fish_used_as_money ∧ mollusk_shells_used_as_money :=
by {
  split; -- Split conjunctions
  all_goals { exact true.intro }; -- Each assumption is true
}

end all_of_the_above_used_as_money_l509_509831


namespace rectangle_area_l509_509874

namespace RectangleAreaProof

theorem rectangle_area (SqrArea : ℝ) (SqrSide : ℝ) (RectWidth : ℝ) (RectLength : ℝ) (RectArea : ℝ) :
  SqrArea = 36 →
  SqrSide = Real.sqrt SqrArea →
  RectWidth = SqrSide →
  RectLength = 3 * RectWidth →
  RectArea = RectWidth * RectLength →
  RectArea = 108 := by
  sorry

end RectangleAreaProof

end rectangle_area_l509_509874


namespace algebraic_expression_value_l509_509631

theorem algebraic_expression_value (m : ℝ) (h : m^2 + 2*m - 1 = 0) : 2*m^2 + 4*m + 2021 = 2023 := 
sorry

end algebraic_expression_value_l509_509631


namespace decreasing_interval_l509_509344

variable (a : ℝ) (x : ℝ)
def f (x : ℝ) := log a (x + 1)

theorem decreasing_interval (h₀ : ∀ x ∈ Ioo (-1 : ℝ) 0, f a x > 0) :
  ∀ x y, x < y → f a y ≤ f a x :=
by
  sorry

end decreasing_interval_l509_509344


namespace sum_of_corners_of_9x9_grid_l509_509856

theorem sum_of_corners_of_9x9_grid : 
    let topLeft := 1
    let topRight := 9
    let bottomLeft := 73
    let bottomRight := 81
    topLeft + topRight + bottomLeft + bottomRight = 164 :=
by {
  sorry
}

end sum_of_corners_of_9x9_grid_l509_509856


namespace sqrt_div_simplification_correct_l509_509093

noncomputable def sqrt_div_simplification : ℝ :=
  real.sqrt 18 / real.sqrt 8

theorem sqrt_div_simplification_correct :
  sqrt_div_simplification = 3/2 := by
  sorry

end sqrt_div_simplification_correct_l509_509093


namespace binomial_expansion_value_l509_509491

theorem binomial_expansion_value : 
  105^3 - 3 * 105^2 + 3 * 105 - 1 = 1124864 := by
  sorry

end binomial_expansion_value_l509_509491


namespace projections_on_circle_l509_509049

/-- Given a tetrahedron ABCD -/
structure Tetrahedron (V : Type*) :=
(A B C D : V)

namespace Tetrahedron

variables {V : Type*} [EuclideanSpace V]
variables {A B C D K L M N : V}

/-- We have a plane \α that intersects the edges AB, BC, CD and DA at points K, L, M, N respectively -/
structure PlaneInteraction (t : Tetrahedron V) (α : Set V) :=
(inter_K : K ∈ α ∧ Segment A B = K)
(inter_L : L ∈ α ∧ Segment B C = L)
(inter_M : M ∈ α ∧ Segment C D = M)
(inter_N : N ∈ α ∧ Segment D A = N)

variables {α : Set V} [PlaneInteraction (Tetrahedron.mk A B C D) α]

/-- Dihedral angles are equal -/
axiom equal_dihedral_angles : 
  dihedral_angle (K, L, A, K, L, M) = dihedral_angle (L, M, B, L, M, N) ∧
  dihedral_angle (M, N, C, M, N, K) = dihedral_angle (N, K, D, N, K, L)

/-- The projections of vertices A, B, C and D onto the plane α lie on a single circle -/
theorem projections_on_circle :
  ∃ (A' B' C' D' : V), 
    is_projection α A A' ∧
    is_projection α B B' ∧
    is_projection α C C' ∧
    is_projection α D D' ∧
    cyclic {A', B', C', D'} :=
sorry

end Tetrahedron

end projections_on_circle_l509_509049


namespace matrix_transform_l509_509586

theorem matrix_transform (M : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ N : Matrix (Fin 3) (Fin 3) ℝ, M.mul N =
    (λ (ABC : Matrix (Fin 3) (Fin 3) ℝ), λ i j, if i = 0 then 3 * ABC 2 j else if i = 1 then ABC 1 j else ABC 0 j) N) →
  M = ![![0, 0, 3], ![0, 1, 0], ![1, 0, 0]] :=
begin
  intro h,
  sorry
end

end matrix_transform_l509_509586


namespace maximum_k_l509_509676

theorem maximum_k (k : ℤ) (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∀ x y, 0 < x ∧ 0 < y → 4 * x^2 + 9 * y^2 ≥ 2^k * x * y) → k ≤ 3 :=
by
  sorry

end maximum_k_l509_509676


namespace range_of_m_l509_509994

variables (m : ℝ) (p q : Prop)

-- Definition of proposition p: The solution set of the inequality |x - 1| > m - 1 is ℝ
def prop_p (m : ℝ) : Prop := ∀ x : ℝ, |x - 1| > m - 1

-- Definition of proposition q: f(x) = -(5-2m)^x is a decreasing function
def prop_q (m : ℝ) : Prop := ∀ x1 x2 : ℝ, (x1 < x2) → (-(5-2m)^x1 > -(5-2m)^x2)

-- Main theorem
theorem range_of_m (h : (prop_p m ∨ prop_q m) ∧ ¬ (prop_p m ∧ prop_q m)) : 1 ≤ m ∧ m < 2 :=
by
  sorry

end range_of_m_l509_509994


namespace renu_suma_work_together_l509_509751

-- Define the time it takes for Renu to do the work by herself
def renu_days : ℕ := 6

-- Define the time it takes for Suma to do the work by herself
def suma_days : ℕ := 12

-- Define the work rate for Renu
def renu_work_rate : ℚ := 1 / renu_days

-- Define the work rate for Suma
def suma_work_rate : ℚ := 1 / suma_days

-- Define the combined work rate
def combined_work_rate : ℚ := renu_work_rate + suma_work_rate

-- Define the days it takes for both Renu and Suma to complete the work together
def days_to_complete_together : ℚ := 1 / combined_work_rate

-- The theorem stating that Renu and Suma can complete the work together in 4 days
theorem renu_suma_work_together : days_to_complete_together = 4 :=
by
  have h1 : renu_days = 6 := rfl
  have h2 : suma_days = 12 := rfl
  have h3 : renu_work_rate = 1 / 6 := by simp [renu_work_rate, h1]
  have h4 : suma_work_rate = 1 / 12 := by simp [suma_work_rate, h2]
  have h5 : combined_work_rate = 1 / 6 + 1 / 12 := by simp [combined_work_rate, h3, h4]
  have h6 : combined_work_rate = 1 / 4 := by norm_num [h5]
  have h7 : days_to_complete_together = 1 / (1 / 4) := by simp [days_to_complete_together, h6]
  have h8 : days_to_complete_together = 4 := by norm_num [h7]
  exact h8

end renu_suma_work_together_l509_509751


namespace product_of_1101_2_and_202_3_is_260_l509_509935

   /-- Convert a binary string to its decimal value -/
   def binary_to_decimal (b : String) : ℕ :=
     b.foldl (λ acc bit, acc * 2 + bit.toNat - '0'.toNat) 0

   /-- Convert a ternary string to its decimal value -/
   def ternary_to_decimal (t : String) : ℕ :=
     t.foldl (λ acc bit, acc * 3 + bit.toNat - '0'.toNat) 0

   theorem product_of_1101_2_and_202_3_is_260 :
     binary_to_decimal "1101" * ternary_to_decimal "202" = 260 :=
   by
     calc
       binary_to_decimal "1101" = 13 : by rfl
       ternary_to_decimal "202" = 20 : by rfl
       13 * 20 = 260 : by rfl
   
end product_of_1101_2_and_202_3_is_260_l509_509935


namespace increasing_implies_max_eq_not_max_implies_increasing_l509_509246

noncomputable def strictly_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem increasing_implies_max_eq (f : ℝ → ℝ) :
  (strictly_increasing f (set.Icc 0 1)) → (∀ x ∈ (set.Icc 0 1), f x ≤ f 1) :=
  sorry

theorem not_max_implies_increasing (f : ℝ → ℝ) : 
  ¬ (∀ x ∈ set.Icc 0 1, f x ≤ f 1) ∨ (strictly_increasing f (set.Icc 0 1)) :=
  sorry

end increasing_implies_max_eq_not_max_implies_increasing_l509_509246


namespace Joey_run_time_l509_509706

theorem Joey_run_time
  (route_distance : ℝ) (round_trip_avg_speed : ℝ) 
  (return_speed : ℝ) (one_way_distance : ℝ) 
  (round_trip_distance : ℝ)
  (average_speed_formula : round_trip_avg_speed = round_trip_distance / ((one_way_distance / return_speed) + one_way_distance / t))
  : t = 1 :=
by
  -- given conditions
  have route_distance_eq : route_distance = 5 := sorry
  have round_trip_avg_speed_eq : round_trip_avg_speed = 8 := sorry
  have return_speed_eq : return_speed = 20 := sorry
  have round_trip_distance_eq : round_trip_distance = 10 := sorry
  have one_way_distance_eq : one_way_distance = 5 := sorry
  -- additional conditions derivable from the conditions
  have average_speed_eq : average_speed_formula := sorry

  -- mathematical proof skipped
  sorry

end Joey_run_time_l509_509706


namespace min_people_for_no_empty_triplet_60_l509_509456

noncomputable def min_people_for_no_empty_triplet (total_chairs : ℕ) : ℕ :=
  if h : total_chairs % 3 = 0 then total_chairs / 3 else sorry

theorem min_people_for_no_empty_triplet_60 :
  min_people_for_no_empty_triplet 60 = 20 :=
by
  sorry

end min_people_for_no_empty_triplet_60_l509_509456


namespace everything_used_as_money_l509_509827

theorem everything_used_as_money :
  (used_as_money gold) ∧
  (used_as_money stones) ∧
  (used_as_money horses) ∧
  (used_as_money dried_fish) ∧
  (used_as_money mollusk_shells) →
  (∀ x ∈ {gold, stones, horses, dried_fish, mollusk_shells}, used_as_money x) :=
by
  intro h
  cases h with
  | intro h_gold h_stones =>
    cases h_stones with
    | intro h_stones h_horses =>
      cases h_horses with
      | intro h_horses h_dried_fish =>
        cases h_dried_fish with
        | intro h_dried_fish h_mollusk_shells =>
          intro x h_x
          cases Set.mem_def.mpr h_x with
          | or.inl h => exact h_gold
          | or.inr h_x1 => cases Set.mem_def.mpr h_x1 with
            | or.inl h => exact h_stones
            | or.inr h_x2 => cases Set.mem_def.mpr h_x2 with
              | or.inl h => exact h_horses
              | or.inr h_x3 => cases Set.mem_def.mpr h_x3 with
                | or.inl h => exact h_dried_fish
                | or.inr h_x4 => exact h_mollusk_shells

end everything_used_as_money_l509_509827


namespace emmalyn_earnings_l509_509572

theorem emmalyn_earnings
  (rate_per_meter : ℚ := 0.20)
  (number_of_fences : ℚ := 50)
  (length_per_fence : ℚ := 500) :
  rate_per_meter * (number_of_fences * length_per_fence) = 5000 := by
  sorry

end emmalyn_earnings_l509_509572


namespace determine_m_n_l509_509640

theorem determine_m_n (m n : ℕ) (h1 : m = 2n - 3) (h2 : 2 + 3n = 8) : m = 1 ∧ n = 2 := by
  sorry

end determine_m_n_l509_509640


namespace ratio_of_areas_l509_509341

variables {A B C P E : Type*} [point A] [point B] [point C] [point P] [point E]
variables (h1 : ∠ PBC = ∠ PCA < ∠ PAB)
variables (BP_circumcircle_intersect : BP ∩ circumcircle A B C = {B, E})
variables (CE_circumcircle_intersect : CE ∩ circumcircle A P E ∃ Q, Q ≠ P)

noncomputable def area (A B C : point) : ℝ := sorry

theorem ratio_of_areas (h2 : BP_circumcircle_intersect) (h3 : CE_circumcircle_intersect) :
  area A P E / area A B P = (dist A C / dist A B) ^ 2 :=
begin
  sorry
end

end ratio_of_areas_l509_509341


namespace problem_PH_length_l509_509697

noncomputable
def triangle_ABC (A B C D E F P H : Type*) [euclidean_space A] :=
  ∃ (AB AC BC : ℝ),
    AB = AC ∧
    ∃ (D is_midpoint D), 
      D = midpoint B C ∧
    ∃ (E is_midpoint E), 
      E = midpoint A C ∧
    ∃ (on_circle P), 
      ∃ (F on_segment AB), 
        BE intersects CF at P ∧
        (B, D, P, F) are_cyclic ∧
    ∃ (H on_intersection AD CP), 
      AD intersects CP at H ∧
      AP = sqrt(5) + 2

theorem problem_PH_length (A B C D E F P H : Type*) [euclidean_space A]
  (conditions : triangle_ABC A B C D E F P H) : 
  PH = 1 :=
sorry

end problem_PH_length_l509_509697


namespace walter_fraction_fewer_bananas_l509_509331

theorem walter_fraction_fewer_bananas (f : ℚ) (h1 : 56 + (56 - 56 * f) = 98) : f = 1 / 4 :=
sorry

end walter_fraction_fewer_bananas_l509_509331


namespace count_integers_in_interval_l509_509659

theorem count_integers_in_interval : 
  let lower_bound := -5 * Real.pi
  let upper_bound := 12 * Real.pi
  (Int.floor upper_bound - Int.ceil lower_bound + 1) = 53 := 
by
  let lower_bound := -5 * Real.pi
  let upper_bound := 12 * Real.pi
  have hpi_lower: Real.pi > 3.14 := Real.pi_pos
  have hpi_upper: Real.pi < 3.15 := sorry
  let approx_lower := -15
  let approx_upper := 37
  have hl: Int.ceil lower_bound = approx_lower := by 
    sorry
  have hu: Int.floor upper_bound = approx_upper := by 
    sorry
  calc 
    Int.floor upper_bound - Int.ceil lower_bound + 1
        = approx_upper - approx_lower + 1 := by rw [hl, hu]
    ... = 53 := by decide

end count_integers_in_interval_l509_509659


namespace find_x2013_l509_509154

def sequence (x : ℕ → ℚ) : Prop :=
  x 0 = 1 ∧ x 1 = 1 ∧ x 2 = 1 ∧ ∀ k, k > 2 → x k = (x (k-1) + x (k-2) + 1) / x (k-3)

theorem find_x2013 (x : ℕ → ℚ) (h : sequence x) : x 2013 = 9 := sorry

end find_x2013_l509_509154


namespace polynomial_positive_root_l509_509284

-- Define the polynomial with the given conditions
noncomputable def P (x : ℝ) : ℝ := 
  x^2002 + 2002 * x^2001 + ∑ k in finset.range 2001, (-k - 1) * x^k

-- State the theorem that the polynomial P has exactly one positive root
theorem polynomial_positive_root : 
  ∃! r : ℝ, r > 0 ∧ is_root P r :=
sorry

end polynomial_positive_root_l509_509284


namespace p_necessary_not_sufficient_q_l509_509620

variables {x : ℝ}

def p (x : ℝ) : Prop := exp x > 1
def q (x : ℝ) : Prop := log x < 0

theorem p_necessary_not_sufficient_q : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) := by
  sorry

end p_necessary_not_sufficient_q_l509_509620


namespace line_equation_exists_chord_length_l509_509043

-- Define the ellipse equation and midpoint condition.
def on_ellipse (A B : ℝ × ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ, 
  A = (x1, y1) ∧ B = (x2, y2) ∧
  (x1^2 / 4 + y1^2 = 1) ∧
  (x2^2 / 4 + y2^2 = 1) ∧
  (x1 + x2 = 2) ∧
  (y1 + y2 = 1 / 2)

-- Prove the equation of the line.
theorem line_equation_exists (A B : ℝ × ℝ) 
  (h: on_ellipse A B) :
  ∃ (a b c : ℝ), 
    a = 1 ∧ b = 2 ∧ c = -2 ∧
    ∀ (x y : ℝ), (x, y) = A ∨ (x, y) = B → a * x + b * y + c = 0 :=
sorry

-- Prove the length of the chord.
theorem chord_length (A B : ℝ × ℝ) 
  (h: on_ellipse A B)
  (h_line : ∃ (x y : ℝ), x + 2*y - 2 = 0) :
  (Real.sqrt ((fst A - fst B)^2 + (snd A - snd B)^2)) = 2 * Real.sqrt 5 :=
sorry

end line_equation_exists_chord_length_l509_509043


namespace part1_part2_l509_509680

-- Definitions and conditions
variable {a b c A B C : ℝ}
variable (ABC : Triangle) -- Assuming there's a Triangle type
variable (h_sides : ABC.sides = (a, b, c))
variable (h_angles : ABC.angles = (A, B, C))
variable (h_condition : tan C / tan B = - c / (2 * a + c))

-- Part I: Prove B = 2π / 3
theorem part1 : B = 2 * π / 3 := by
  sorry

-- Part II: Given additional conditions prove area = √3
variable (h_b : b = 2 * Real.sqrt 3)
variable (h_sum : a + c = 4)

theorem part2 : area ABC = Real.sqrt 3 := by
  sorry

end part1_part2_l509_509680


namespace certain_number_l509_509668

theorem certain_number (p q x : ℝ) (h1 : 3 / p = x) (h2 : 3 / q = 15) (h3 : p - q = 0.3) : x = 6 :=
sorry

end certain_number_l509_509668


namespace problem_statement_l509_509971

theorem problem_statement (a b : ℝ) (h : a^2 > b^2) : a > b → a > 0 :=
sorry

end problem_statement_l509_509971


namespace parallel_line_intercept_l509_509207

-- Conditions
variables {A B C M K P Q : Type*}

-- Given an angle ABC
variables (ABC : ∠ (A B C))

-- Given a line l
variables (l : set.Point)

-- Given a segment length a
variables (a : ℝ) (h_a_pos : 0 < a)

-- Proof Statement
theorem parallel_line_intercept {ABC l a} :
  ∃ PQ : set.Point, 
  is_parallel PQ l ∧
  segment_intercept_angle (A B C) PQ = a :=
sorry

end parallel_line_intercept_l509_509207


namespace window_side_length_is_five_l509_509023

def pane_width (x : ℝ) : ℝ := x
def pane_height (x : ℝ) : ℝ := 3 * x
def border_width : ℝ := 1
def pane_rows : ℕ := 2
def pane_columns : ℕ := 3

theorem window_side_length_is_five (x : ℝ) (h : pane_height x = 3 * pane_width x) : 
  (3 * x + 4 = 6 * x + 3) -> (3 * x + 4 = 5) :=
by
  intros h1
  sorry

end window_side_length_is_five_l509_509023


namespace cos_theta_is_correct_l509_509527

def vector_1 : ℝ × ℝ := (4, 5)
def vector_2 : ℝ × ℝ := (2, 7)

noncomputable def cos_theta (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1 * v1.1 + v1.2 * v1.2) * Real.sqrt (v2.1 * v2.1 + v2.2 * v2.2))

theorem cos_theta_is_correct :
  cos_theta vector_1 vector_2 = 43 / (Real.sqrt 41 * Real.sqrt 53) :=
by
  -- proof goes here
  sorry

end cos_theta_is_correct_l509_509527


namespace problem_1_problem_2_l509_509140

-- Problem 1: Prove that (-1)^3 + sqrt(4) - (2 - sqrt(2))^0 = 0
theorem problem_1 : (-1)^3 + Real.sqrt(4) - (2 - Real.sqrt(2))^0 = 0 :=
by
  sorry

-- Problem 2: Prove that (a + 3) * (a - 3) - a * (a - 2) = 2 * a - 9
theorem problem_2 (a : ℝ) : (a + 3) * (a - 3) - a * (a - 2) = 2 * a - 9 :=
by
  sorry

end problem_1_problem_2_l509_509140


namespace stating_area_trapezoid_AMBQ_is_18_l509_509561

/-- Definition of the 20-sided polygon configuration with 2 unit sides and right-angle turns. -/
structure Polygon20 where
  sides : ℕ → ℝ
  units : ∀ i, sides i = 2
  right_angles : ∀ i, (i + 1) % 20 ≠ i -- Right angles between consecutive sides

/-- Intersection point of AJ and DP, named M, under the given polygon configuration. -/
def intersection_point (p : Polygon20) : ℝ × ℝ :=
  (5 * p.sides 0, 5 * p.sides 1)  -- Assuming relevant distances for simplicity

/-- Area of the trapezoid AMBQ formed given the defined Polygon20. -/
noncomputable def area_trapezoid_AMBQ (p : Polygon20) : ℝ :=
  let base1 := 10 * p.sides 0
  let base2 := 8 * p.sides 0
  let height := p.sides 0
  (base1 + base2) * height / 2

/-- 
  Theorem stating the area of the trapezoid AMBQ in the given configuration.
  We prove that the area is 18 units.
-/
theorem area_trapezoid_AMBQ_is_18 (p : Polygon20) :
  area_trapezoid_AMBQ p = 18 :=
sorry -- Proof to be done

end stating_area_trapezoid_AMBQ_is_18_l509_509561


namespace correct_number_of_conclusions_l509_509458

-- Define the initial polynomials and the operations function
def initial_polynomials (a : ℝ) : list ℝ := [a, a + 2]

def operation (polys : list ℝ) : list ℝ :=
  list.zip_with (λ x y, y - x) polys (polys.tail ++ [polys.head])

noncomputable def second_operation_polynomials (a : ℝ) : list ℝ :=
  operation (operation (initial_polynomials a))

noncomputable def third_operation_polynomials (a : ℝ) : list ℝ :=
  operation (second_operation_polynomials a)

noncomputable def sum_polynomials_after_n_operations (a : ℝ) (n : ℕ) : ℝ :=
  2 * a + 2 * (n + 1)

-- Define the theorem and proof structure
theorem correct_number_of_conclusions (a : ℝ) (h1 : second_operation_polynomials a ≠ [a, 2 - a, a, a + 2])
  (h2 : |a| ≥ 2 → (2 * a ^ 2 * (4 - a ^ 2)) ≤ 0)
  (h3 : third_operation_polynomials a ≠ 9)
  (h4 : sum_polynomials_after_n_operations a 2023 = 2 * a + 4048) : 
  true :=
by sorry

end correct_number_of_conclusions_l509_509458


namespace initial_girls_l509_509806

theorem initial_girls (G : ℕ) 
  (h1 : G + 7 + (15 - 4) = 36) : G = 18 :=
by
  sorry

end initial_girls_l509_509806


namespace seventh_triangular_number_eq_28_l509_509437

noncomputable def triangular_number (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem seventh_triangular_number_eq_28 :
  triangular_number 7 = 28 :=
by
  sorry

end seventh_triangular_number_eq_28_l509_509437


namespace seeds_per_flowerbed_l509_509744

theorem seeds_per_flowerbed (total_seeds : ℕ) (flowerbeds : ℕ) (seeds_per_bed : ℕ) 
  (h1 : total_seeds = 45) (h2 : flowerbeds = 9) 
  (h3 : total_seeds = flowerbeds * seeds_per_bed) : seeds_per_bed = 5 :=
by sorry

end seeds_per_flowerbed_l509_509744


namespace circle_reflection_l509_509772

/-- The reflection of a point over the line y = -x results in swapping the x and y coordinates 
and changing their signs. Given a circle with center (3, -7), the reflected center should be (7, -3). -/
theorem circle_reflection (x y : ℝ) (h : (x, y) = (3, -7)) : (y, -x) = (7, -3) :=
by
  -- since the problem is stated to skip the proof, we use sorry
  sorry

end circle_reflection_l509_509772


namespace stored_sugar_amount_l509_509471

theorem stored_sugar_amount (total_sugar_needed : ℝ) (additional_sugar_needed : ℝ) : 
  total_sugar_needed = 450 ∧ additional_sugar_needed = 163 → 
  total_sugar_needed - additional_sugar_needed = 287 :=
by
  intros h
  cases h with ht ha
  rw [ht, ha]
  norm_num
  sorry

end stored_sugar_amount_l509_509471


namespace sum_proper_divisors_of_720_l509_509931

def sum_of_proper_divisors (n : ℕ) : ℕ :=
  (List.range n).filter (λ d, d ≠ 0 ∧ n % d = 0).sum

theorem sum_proper_divisors_of_720 :
  sum_of_proper_divisors 720 = 1698 :=
by
  -- Proof details can be filled here
  sorry

end sum_proper_divisors_of_720_l509_509931


namespace sequence_sum_l509_509274

open Nat

noncomputable def a_n (n : ℕ) : ℤ := (-1) ^ n * (3 * (n + 1) - 2)

theorem sequence_sum : (∑ n in range 20, a_n n) = 30 := by
  sorry

end sequence_sum_l509_509274


namespace production_ratio_l509_509130

-- Define the number of apples produced in each season as variables
variables (first_season second_season third_season : ℕ)

-- Define the conditions
def conditions (first_season second_season third_season : ℕ) : Prop :=
  first_season = 200 ∧
  second_season = first_season - (0.20 * first_season).toNat ∧
  first_season + second_season + third_season = 680

-- Define the desired ratio
def ratio (second_season third_season : ℕ) : ℚ :=
  third_season / second_season

-- The theorem to prove
theorem production_ratio (first_season second_season third_season : ℕ) 
  (h : conditions first_season second_season third_season) : 
  ratio second_season third_season = 2 :=
by
  -- This is just the statement, actual proof is omitted
  sorry

end production_ratio_l509_509130


namespace chocolate_milk_container_size_l509_509657

/-- Holly's chocolate milk consumption conditions and container size -/
theorem chocolate_milk_container_size
  (morning_initial: ℝ)  -- Initial amount in the morning
  (morning_drink: ℝ)    -- Amount drank in the morning with breakfast
  (lunch_drink: ℝ)      -- Amount drank at lunch
  (dinner_drink: ℝ)     -- Amount drank with dinner
  (end_of_day: ℝ)       -- Amount she ends the day with
  (lunch_container_size: ℝ) -- Size of the container bought at lunch
  (C: ℝ)                -- Container size she bought at lunch
  (h_initial: morning_initial = 16)
  (h_morning_drink: morning_drink = 8)
  (h_lunch_drink: lunch_drink = 8)
  (h_dinner_drink: dinner_drink = 8)
  (h_end_of_day: end_of_day = 56) :
  (morning_initial - morning_drink) + C - lunch_drink - dinner_drink = end_of_day → 
  lunch_container_size = 64 :=
by
  sorry

end chocolate_milk_container_size_l509_509657


namespace count_two_digit_divisors_l509_509445

-- Define the given conditions for the problem
def two_digit_positive_integers := {d : ℕ | 10 ≤ d ∧ d < 100}

-- Define the count of elements in a finite set
noncomputable def count_valid_divisors (x : ℕ) (valid_div : ℕ → Prop) : ℕ :=
  (finset.filter valid_div (finset.Icc 1 x)).card

-- Problem statement: Prove that c = 4
theorem count_two_digit_divisors : 
  let c := count_valid_divisors 392 (λ d, d ∈ two_digit_positive_integers ∧ 392 % d = b) in
  c = 4 :=
sorry

end count_two_digit_divisors_l509_509445


namespace sum_f_l509_509979

noncomputable def f (x : ℝ) : ℝ := Real.cos (π * x / 6)

theorem sum_f (n : ℕ) (hn : n = 2017) : (∑ i in Finset.range n.succ, f i) = √3 / 2 :=
by
  have h_period : ∀ x, f (x + 12) = f x := by
    intro x
    simp [f, Real.cos_periodic (π / 6) 12]
  sorry

end sum_f_l509_509979


namespace greatest_x_lcm_l509_509042

theorem greatest_x_lcm (x : ℕ) (h1 : Nat.lcm x 15 = Nat.lcm 90 15) (h2 : Nat.lcm x 18 = Nat.lcm 90 18) : x = 90 := 
sorry

end greatest_x_lcm_l509_509042


namespace exists_four_integers_multiple_1984_l509_509595

theorem exists_four_integers_multiple_1984 (a : Fin 97 → ℕ) (h_distinct : Function.Injective a) :
  ∃ i j k l : Fin 97, i ≠ j ∧ k ≠ l ∧ 1984 ∣ (a i - a j) * (a k - a l) :=
sorry

end exists_four_integers_multiple_1984_l509_509595


namespace minimum_ticket_cost_l509_509365

theorem minimum_ticket_cost :
  let num_people := 12
  let num_adults := 8
  let num_children := 4
  let adult_ticket_cost := 100
  let child_ticket_cost := 50
  let group_ticket_cost := 70
  num_people = num_adults + num_children →
  (num_people >= 10) →
  ∃ (cost : ℕ), cost = min (num_adults * adult_ticket_cost + num_children * child_ticket_cost) (group_ticket_cost * num_people) ∧
  cost = min (group_ticket_cost * 10 + child_ticket_cost * (num_people - 10)) (group_ticket_cost * num_people) →
  cost = 800 :=
by
  intro h1 h2
  sorry

end minimum_ticket_cost_l509_509365


namespace rectangle_length_to_width_ratio_l509_509418

-- Define the side length of the square
def s : ℝ := 1 -- Since we only need the ratio, the actual length does not matter

-- Define the length and width of the large rectangle
def length_of_large_rectangle : ℝ := 3 * s
def width_of_large_rectangle : ℝ := 3 * s

-- Define the dimensions of the small rectangle
def length_of_rectangle : ℝ := 3 * s
def width_of_rectangle : ℝ := s

-- Proving that the length of the rectangle is 3 times its width
theorem rectangle_length_to_width_ratio : length_of_rectangle = 3 * width_of_rectangle := 
by
  -- The proof is omitted
  sorry

end rectangle_length_to_width_ratio_l509_509418


namespace radius_of_third_circle_l509_509562

theorem radius_of_third_circle (r_inner r_outer : ℝ) (r_inner_pos : r_inner = 21) (r_outer_pos : r_outer = 31) : 
  ∃ r_new : ℝ, r_new = 2 * Real.sqrt 130 ∧ r_outer^2 * Real.pi - r_inner^2 * Real.pi = r_new^2 * Real.pi := 
by 
  sorry

end radius_of_third_circle_l509_509562


namespace soccer_camp_afternoon_kids_l509_509454

theorem soccer_camp_afternoon_kids (total_kids : ℕ) 
  (half_kids_to_soccer : total_kids / 2 = 1000)
  (ten_percent_to_different : 0.1 * 1000 = 100)
  (remaining_kids_soccer : 1000 - 100 = 900)
  (one_quarter_in_morning : 0.25 * 900 = 225)
  (switch_to_morning : 30)
  : (900 - (225 + switch_to_morning) = 645) :=
by
  sorry

end soccer_camp_afternoon_kids_l509_509454


namespace categorize_numbers_l509_509942

theorem categorize_numbers :
  (∀ x ∈ ({-5/6, 0, -3.5, 1.2, 6} : set ℝ), x < 0 ↔ x ∈ ({-5/6, -3.5} : set ℝ)) ∧
  (∀ x ∈ ({-5/6, 0, -3.5, 1.2, 6} : set ℝ), x ≥ 0 ↔ x ∈ ({0, 1.2, 6} : set ℝ)) :=
by
  sorry

end categorize_numbers_l509_509942


namespace exercise_l509_509609

-- Define the piecewise function
def f (x: ℝ) : ℝ :=
  if x ≤ 0 then 
    Real.sin (π * x / 6)
  else 
    1 - 2 * x

-- State the theorem to be proved
theorem exercise : f (f 3) = -1 / 2 := by
  sorry

end exercise_l509_509609


namespace simplify_rationalize_l509_509415

theorem simplify_rationalize
  : (1 / (1 + (1 / (Real.sqrt 5 + 2)))) = ((Real.sqrt 5 + 1) / 4) := 
sorry

end simplify_rationalize_l509_509415


namespace f_series_sum_l509_509972

-- Define the function f
def f (x : ℝ) : ℝ := 4 * Real.log2 x + 233

-- Define the given condition
axiom f_condition : ∀ x : ℝ, f (3^x) = 4 * x * Real.log2 3 + 233

-- State the theorem
theorem f_series_sum : f 2 + f 4 + f 8 + f 16 + f 32 + f 64 + f 128 + f 256 = 2008 :=
sorry

end f_series_sum_l509_509972


namespace product_of_roots_l509_509484

theorem product_of_roots (x : ℝ) (h : x + 1/x = 4 * x) :
  x * (- (1 / sqrt 3)) = 1 / 3 := 
sorry

end product_of_roots_l509_509484


namespace tens_digit_17_pow_1993_l509_509932

theorem tens_digit_17_pow_1993 :
  (17 ^ 1993) % 100 / 10 = 3 := by
  sorry

end tens_digit_17_pow_1993_l509_509932


namespace sum_first_13_terms_l509_509619

variable (a_n : ℕ → ℝ) (a₁ d : ℝ)
variable (n : ℕ)
variable (S_n : ℕ → ℝ)

-- Definition for arithmetic sequence
def is_arithmetic_seq (a_n : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
  ∀ n : ℕ, a_n n = a₁ + (n - 1) * d

-- Given conditions
def sum_condition (a_n : ℕ → ℝ) : Prop :=
  a_n 2 + a_n 8 + a_n 11 = 30

-- Sum of the first n terms
def sum_first_n_terms (a_n : ℕ → ℝ) : ℕ → ℝ
| 0 := 0
| (n+1) := sum_first_n_terms n + a_n (n+1)

-- The statement to be proved
theorem sum_first_13_terms (h₁ : is_arithmetic_seq a_n a₁ d) (h₂ : sum_condition a_n) :
  sum_first_n_terms a_n 13 = 130 :=
sorry

end sum_first_13_terms_l509_509619


namespace rectangle_area_theorem_l509_509116

def rectangle_area (d : ℝ) (area : ℝ) : Prop :=
  ∃ w : ℝ, 0 < w ∧ 9 * w^2 + w^2 = d^2 ∧ area = 3 * w^2

theorem rectangle_area_theorem (d : ℝ) : rectangle_area d (3 * d^2 / 10) :=
sorry

end rectangle_area_theorem_l509_509116


namespace arithmetic_sequence_term_l509_509922

theorem arithmetic_sequence_term (p q : ℤ)
  (h1 : p + 6 - p = 4p - q - (p + 6))
  (h2 : 4p + q - (4p - q) = 2q) :
  p + 2022 * (2*q) = 12137 := 
by
  sorry

end arithmetic_sequence_term_l509_509922


namespace expected_heads_three_coins_is_three_halves_l509_509463

noncomputable def expected_heads_three_coins : ℚ :=
  ((0 * (1 / 8)) + (1 * (3 / 8)) + (2 * (3 / 8)) + (3 * (1 / 8)))

theorem expected_heads_three_coins_is_three_halves :
  expected_heads_three_coins = 3 / 2 :=
by
  sorry

end expected_heads_three_coins_is_three_halves_l509_509463


namespace sec_pi_div_3_eq_2_l509_509584

-- Definition of secant in terms of cosine
def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- The specific angle θ = π / 3 radians
def angle : ℝ := Real.pi / 3

-- The theorem stating that sec π / 3 = 2
theorem sec_pi_div_3_eq_2 : sec angle = 2 := by
  sorry

end sec_pi_div_3_eq_2_l509_509584


namespace sum_of_fifth_powers_l509_509224

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l509_509224


namespace range_of_a_l509_509643

theorem range_of_a (a : ℝ) :
  (∃! x : ℝ, sqrt (x^2 - 1) = a * x - 2) ↔ a ∈ set.Ico (-real.sqrt 5) (-1) ∪ set.Ioo 1 (real.sqrt 5) :=
by sorry

end range_of_a_l509_509643


namespace gina_balance_fractions_half_l509_509604

def gina_fractions_sum_half (betty_balance gina_combined_balance : ℤ) (f1 f2 : ℚ) : Prop :=
  gina_combined_balance = (456 * 8) / 2 ∧
  betty_balance = 456 * 8 ∧
  gina_combined_balance = (betty_balance : ℚ) * (f1 + f2) ∧
  (f1 + f2 = 1 / 2)

-- We can state the theorem
theorem gina_balance_fractions_half :
  ∃ (f1 f2 : ℚ), gina_fractions_sum_half 3456 1728 f1 f2 :=
begin
  sorry
end

end gina_balance_fractions_half_l509_509604


namespace sum_gt_two_l509_509265

noncomputable def f (x : ℝ) : ℝ := ((x - 1) * Real.log x) / x

theorem sum_gt_two (x₁ x₂ : ℝ) (h₁ : f x₁ = f x₂) (h₂ : x₁ ≠ x₂) : x₁ + x₂ > 2 := 
sorry

end sum_gt_two_l509_509265


namespace tank_empty_time_l509_509124

-- Define the given conditions
def volume_in_cubic_feet : ℝ := 60
def volume_conversion_factor : ℝ := 1728
def inlet_rate : ℝ := 3
def outlet_rates : List ℝ := [12, 6, 18, 9]
def cubic_feet_to_cubic_inches (v : ℝ) := v * volume_conversion_factor
def total_outflow_rate (rates : List ℝ) := rates.foldr (+) 0

-- Define the question statement and the final answer
def time_to_empty_tank (volume : ℝ) (net_outflow_rate : ℝ) := volume / net_outflow_rate
def expected_time : ℝ := 2468.57

theorem tank_empty_time :
  let volume := cubic_feet_to_cubic_inches volume_in_cubic_feet
  let total_outflow := total_outflow_rate outlet_rates
  let net_outflow := total_outflow - inlet_rate
  time_to_empty_tank volume net_outflow = expected_time :=
by
  sorry

end tank_empty_time_l509_509124


namespace algebraic_expression_value_l509_509666

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 2 + 1) : x^2 - 2 * x + 2 = 3 :=
by
  sorry

end algebraic_expression_value_l509_509666


namespace smallest_constant_inequality_l509_509590

theorem smallest_constant_inequality :
  ∀ (x y : ℝ), 1 + (x + y)^2 ≤ (4 / 3) * (1 + x^2) * (1 + y^2) :=
by
  intro x y
  sorry

end smallest_constant_inequality_l509_509590


namespace function_constant_l509_509351

variable {f : ℝ → ℝ}

theorem function_constant 
  (h : ∀ x y, 0 < x → 0 < y → f (sqrt (x * y)) = f ((x + y) / 2)) : 
  ∀ z w, f z = f w :=
by
  sorry

end function_constant_l509_509351


namespace No_response_percentage_l509_509754

theorem No_response_percentage (total_guests : ℕ) (yes_percentage : ℕ) (non_respondents : ℕ) (yes_guests := total_guests * yes_percentage / 100) (no_guests := total_guests - yes_guests - non_respondents) (no_percentage := no_guests * 100 / total_guests) :
  total_guests = 200 → yes_percentage = 83 → non_respondents = 16 → no_percentage = 9 :=
by
  sorry

end No_response_percentage_l509_509754


namespace sum_of_first_8_terms_l509_509617

/-- Given a sequence {a_n} such that 3a_{n+1} + a_n = 0 and a_3 = 4/9,
    prove that the sum of the first 8 terms of the sequence is 3(1 - 3^(-8)). -/

theorem sum_of_first_8_terms (a : ℕ → ℝ) 
  (h1 : ∀ n, 3 * a (n + 1) + a n = 0)
  (h2 : a 3 = 4 / 9) : 
  ∑ k in finset.range 8, a k = 3 * (1 - (3 : ℝ)⁻⁸) :=
sorry

end sum_of_first_8_terms_l509_509617


namespace ice_cream_total_difference_l509_509740

def scoops := ℕ

structure BananaSplit :=
  (vanilla : scoops)
  (chocolate : scoops)
  (strawberry : scoops)

def Oli := BananaSplit.mk 2 1 1
def Victoria := BananaSplit.mk 4 2 2
def Brian := BananaSplit.mk 3 3 1

def diff (a b : scoops) : scoops := if a > b then a - b else b - a

noncomputable def total_difference (x y z: BananaSplit) : scoops :=
  let vanilla_diff := (diff x.vanilla y.vanilla) + (diff x.vanilla z.vanilla) + (diff y.vanilla z.vanilla)
  let chocolate_diff := (diff x.chocolate y.chocolate) + (diff x.chocolate z.chocolate) + (diff y.chocolate z.chocolate)
  let strawberry_diff := (diff x.strawberry y.strawberry) + (diff x.strawberry z.strawberry) + (diff y.strawberry z.strawberry)
  vanilla_diff + chocolate_diff + strawberry_diff

theorem ice_cream_total_difference :
  total_difference Oli Victoria Brian = 10 := 
sorry

end ice_cream_total_difference_l509_509740


namespace hyperbola_asymptote_slope_proof_l509_509786

noncomputable def hyperbola_asymptote_slope : ℝ :=
  let foci_distance := Real.sqrt ((8 - 2)^2 + (3 - 3)^2)
  let c := foci_distance / 2
  let a := 2  -- Given that 2a = 4
  let b := Real.sqrt (c^2 - a^2)
  b / a

theorem hyperbola_asymptote_slope_proof :
  ∀ x y : ℝ, 
  (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) →
  hyperbola_asymptote_slope = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_asymptote_slope_proof_l509_509786


namespace blocks_probability_l509_509550

-- Definitions for problem conditions
def ang_blocks : list (String × ℕ) := [("red", 1), ("blue", 1), ("yellow", 1), ("white", 1), ("green", 1), ("purple", 1)]
def ben_blocks : list (String × ℕ) := [("red", 1), ("blue", 1), ("yellow", 1), ("white", 1), ("green", 1), ("purple", 1)]
def jasmin_blocks : list (String × ℕ) := [("red", 1), ("blue", 1), ("yellow", 1), ("white", 1), ("green", 1), ("purple", 1)]
def boxes : ℕ := 6

-- The probability calculation is part of the following problem representation
theorem blocks_probability :
  ∃ (m n : ℕ), Nat.rel_prime m n ∧ (m + n = 14471) ∧
  (( ∏ x : fin 6, if ( ∏ x_1 : fin 6, ang_blocks.nth x_1) = ( ∏ x_2 : fin 6, ben_blocks.nth x_2) = ( ∏ x_3 : fin 6, jasmin_blocks.nth x_3) then 1 else 0) /
  ( 6 * 6 * 6 ) = (m / n)) :=
sorry

-- Definitions and calculations that are required should be here

end blocks_probability_l509_509550


namespace volume_of_56_ounces_is_24_cubic_inches_l509_509088

-- Given information as premises
def directlyProportional (V W : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ V = k * W

-- The specific conditions in the problem
def initial_volume := 48   -- in cubic inches
def initial_weight := 112  -- in ounces
def target_weight := 56    -- in ounces
def target_volume := 24    -- in cubic inches (the value we need to prove)

-- The theorem statement 
theorem volume_of_56_ounces_is_24_cubic_inches
  (h1 : directlyProportional initial_volume initial_weight)
  (h2 : directlyProportional target_volume target_weight)
  (h3 : target_weight = 56)
  (h4 : initial_volume = 48)
  (h5 : initial_weight = 112) :
  target_volume = 24 :=
sorry -- Proof not required as per instructions

end volume_of_56_ounces_is_24_cubic_inches_l509_509088


namespace toes_on_bus_is_164_l509_509376

def num_toes_hoopit : Nat := 3 * 4
def num_toes_neglart : Nat := 2 * 5

def num_hoopits : Nat := 7
def num_neglarts : Nat := 8

def total_toes_on_bus : Nat :=
  num_hoopits * num_toes_hoopit + num_neglarts * num_toes_neglart

theorem toes_on_bus_is_164 : total_toes_on_bus = 164 := by
  sorry

end toes_on_bus_is_164_l509_509376


namespace find_star_1993_1932_l509_509616

-- Define the operation and conditions as per the problem statement
def star_op (x y : ℤ) : ℤ := x - y

axiom star_eq_zero : ∀ x : ℤ, star_op x x = 0
axiom star_distributive : ∀ x y z : ℤ, star_op x (star_op y z) = star_op (star_op x y) z

-- This will be our statement to prove
theorem find_star_1993_1932 : star_op 1993 1932 = 61 :=
by {
  -- Proof would be placed here
  sorry
}

end find_star_1993_1932_l509_509616


namespace compare_abc_l509_509663

theorem compare_abc
  (a : ℤ) (b : ℤ) (c : ℤ)
  (h_a : a = (-1 : ℤ) * 3)
  (h_b : b = 2 - 4)
  (h_c : c = 2 ∕ (-1 ∕ 2)) : c < a ∧ a < b :=
by
  sorry

end compare_abc_l509_509663


namespace trisect_diagonal_l509_509314

theorem trisect_diagonal 
  (A B C D E F : Point)
  (G H : Point) 
  (h1 : Rectangle ABCD) 
  (h2 : Midpoint E B C) 
  (h3 : Midpoint F C D) 
  (h4 : IntersectPoint G A E B D)
  (h5 : IntersectPoint H A F B D) : 
  SegmentEqual B G G H ∧ SegmentEqual G H H D :=
sorry

end trisect_diagonal_l509_509314


namespace innings_played_l509_509425

noncomputable def cricket_player_innings : Nat :=
  let average_runs := 32
  let increase_in_average := 6
  let next_innings_runs := 158
  let new_average := average_runs + increase_in_average
  let runs_before_next_innings (n : Nat) := average_runs * n
  let total_runs_after_next_innings (n : Nat) := runs_before_next_innings n + next_innings_runs
  let total_runs_with_new_average (n : Nat) := new_average * (n + 1)

  let n := (total_runs_after_next_innings 20) - (total_runs_with_new_average 20)
  
  n
     
theorem innings_played : cricket_player_innings = 20 := by
  sorry

end innings_played_l509_509425


namespace similar_triangles_of_right_triangle_l509_509395

noncomputable def right_triangle (A B C D : Type) [MetricSpace A] :=
  ∃ (a b c ab ac bc : A), is_right_angle ∠C ∧ is_altitude (CD) ∧
    sim_triangle (ACD) (ABC) ∧ sim_triangle (CBD) (ABC)

theorem similar_triangles_of_right_triangle (A B C : Type) [MetricSpace A] 
  (h : ∀ A B C : Type, right_triangle A B C) :
  ∃ (D : Type), sim_triangle (ACD) (ABC) ∧ sim_triangle (CBD) (ABC) :=
by sorry

end similar_triangles_of_right_triangle_l509_509395


namespace work_done_l509_509383

-- Definitions of given conditions
def is_cyclic_process (gas: Type) (a b c: gas) : Prop := sorry
def isothermal_side (a b: Type) : Prop := sorry
def line_through_origin (b c: Type) : Prop := sorry
def parabolic_arc_through_origin (c a: Type) : Prop := sorry

def temperature_equality (T_a T_c: ℝ) : Prop :=
  T_a = T_c

def half_pressure (P_a P_c: ℝ) : Prop :=
  P_a = 0.5 * P_c

-- Main theorem statement
theorem work_done (T_0 P_a P_c: ℝ) (a b c: Type) 
  (H_cycle: is_cyclic_process gas a b c)
  (H_isothermal: isothermal_side a b)
  (H_line_origin: line_through_origin b c)
  (H_parabolic_arc: parabolic_arc_through_origin c a)
  (H_temp_eq: temperature_equality T_0 320)
  (H_pressure_half: half_pressure P_a P_c) :
  (work_done gas a b c) = 665 := sorry

end work_done_l509_509383


namespace all_of_the_above_were_used_as_money_l509_509825

-- Defining the conditions that each type was used as money
def gold_used_as_money : Prop := True
def stones_used_as_money : Prop := True
def horses_used_as_money : Prop := True
def dried_fish_used_as_money : Prop := True
def mollusk_shells_used_as_money : Prop := True

-- The statement to prove
theorem all_of_the_above_were_used_as_money :
  gold_used_as_money ∧
  stones_used_as_money ∧
  horses_used_as_money ∧
  dried_fish_used_as_money ∧
  mollusk_shells_used_as_money ↔
  (∀ x, (x = "gold" ∨ x = "stones" ∨ x = "horses" ∨ x = "dried fish" ∨ x = "mollusk shells") → 
  (x = "gold" ∧ gold_used_as_money) ∨ 
  (x = "stones" ∧ stones_used_as_money) ∨ 
  (x = "horses" ∧ horses_used_as_money) ∨ 
  (x = "dried fish" ∧ dried_fish_used_as_money) ∨ 
  (x = "mollusk shells" ∧ mollusk_shells_used_as_money)) :=
by
  sorry

end all_of_the_above_were_used_as_money_l509_509825


namespace sum_of_acute_angles_l509_509624

theorem sum_of_acute_angles (α β γ : ℝ) (h1 : α > 0 ∧ α < π / 2) (h2 : β > 0 ∧ β < π / 2) (h3: γ > 0 ∧ γ < π / 2) (h4 : (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 = 1) :
  (3 * π / 4) < α + β + γ ∧ α + β + γ < π :=
by
  sorry

end sum_of_acute_angles_l509_509624


namespace players_taking_exactly_two_subjects_l509_509902

theorem players_taking_exactly_two_subjects
    (total_players : ℕ)
    (players_physics : ℕ)
    (players_chemistry : ℕ)
    (players_biology : ℕ)
    (players_all_three : ℕ)
    (h_total : total_players = 30)
    (h_physics : players_physics = 12)
    (h_chemistry : players_chemistry = 10)
    (h_biology : players_biology = 8)
    (h_all_three : players_all_three = 3)
    (h_union : total_players = players_physics + players_chemistry + players_biology - (players_physics ∩ players_chemistry) - (players_physics ∩ players_biology) - (players_chemistry ∩ players_biology) + players_all_three) :
  (players_physics ∩ players_chemistry) + (players_physics ∩ players_biology) + (players_chemistry ∩ players_biology) - 3 * players_all_three = 0 := by
  sorry

end players_taking_exactly_two_subjects_l509_509902


namespace regular_polygon_sides_l509_509671

theorem regular_polygon_sides (n : ℕ) (h₁ : n > 2) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n → True) (h₃ : (360 / n : ℝ) = 30) : n = 12 := by
  sorry

end regular_polygon_sides_l509_509671


namespace positive_difference_between_payments_is_6542_l509_509366

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (n : ℕ) (time : ℕ) : ℝ :=
  principal * (1 + rate / n)^(n * time)

noncomputable def plan_1_payment (principal : ℝ) (rate : ℝ) (half_time full_time : ℕ) : ℝ :=
  let amount_after_half_time := compound_interest principal rate 2 half_time
  let half_payment := amount_after_half_time / 2
  let remaining_balance := compound_interest half_payment rate 2 half_time
  half_payment + remaining_balance

noncomputable def plan_2_payment (principal : ℝ) (rate : ℝ) (full_time : ℕ) : ℝ :=
  compound_interest principal rate 1 full_time

theorem positive_difference_between_payments_is_6542 (principal : ℝ) (rate : ℝ)
  (half_time full_time : ℕ) (difference : ℝ) :
  principal = 15000 → rate = 0.08 → half_time = 6 → full_time = 12 →
  difference = (plan_2_payment principal rate full_time -
               plan_1_payment principal rate half_time full_time).round →
  difference = 6542 :=
by
  intros
  sorry

end positive_difference_between_payments_is_6542_l509_509366


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l509_509211

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l509_509211


namespace roots_inequality_l509_509650

noncomputable theory

open Real

def f (x : ℝ) (m : ℝ) : ℝ := log x - m * x

theorem roots_inequality (m : ℝ) (x₁ x₂ : ℝ) (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hroots : f x₁ m = 0 ∧ f x₂ m = 0) (hne : x₁ ≠ x₂) :
  m * (x₁ + x₂) > 2 :=
sorry

end roots_inequality_l509_509650


namespace first_job_men_count_l509_509020

variable (M : ℕ)

def work1 (M : ℕ) : ℕ := M * 16
def work2 : ℕ := 600 * 20

-- Condition: the second job is 3 times the first job
def job_condition (M : ℕ) : Prop := work2 = 3 * work1 M

theorem first_job_men_count : job_condition M → M = 250 :=
by
  intro h
  have h_work2 : work2 = 12000 := rfl
  unfold work1 at h
  rw [h_work2] at h
  sorry

end first_job_men_count_l509_509020


namespace exists_positive_integer_expressible_in_two_ways_l509_509748

theorem exists_positive_integer_expressible_in_two_ways (n : ℕ) (hn : n = 2015) : 
  ∃ (S : ℕ), 
  ∃ (x1 x2 ... x2015 y1 y2 ... y2015 : ℕ), 
  (x1 < x2 < ... < x2015) ∧ 
  (y1 < y2 < ... < y2015) ∧ 
  (x1 ≠ y1 ∨ x2 ≠ y2 ∨ ... ∨ x2015 ≠ y2015) ∧ 
  (S = x1 ^ 2014 + ... + x2015 ^ 2014) ∧ 
  (S = y1 ^ 2014 + ... + y2015 ^ 2014) := sorry

end exists_positive_integer_expressible_in_two_ways_l509_509748


namespace range_of_r_l509_509763

theorem range_of_r (a b c r : ℝ) (h1 : b + c ≤ 4 * a) (h2 : c - b ≥ 0) (h3 : b ≥ a) (h4 : a > 0) : 
  ((a + b) ^ 2 + (a + c) ^ 2 ≠ (a * r) ^ 2) ↔ (r ∈ set.Iio (2 * real.sqrt 2) ∨ r ∈ set.Ioi (3 * real.sqrt 2)) :=
by sorry

end range_of_r_l509_509763


namespace employees_in_january_l509_509844

variable (CompanyP : Type)
variable (employeesInDec : Int := 450)
variable (percentIncrease : Float := 0.15)

-- Define the number of employees in January based on the conditions
def employeesInJan (d : Int) (p : Float) : Int :=
  Int.ofFloat ((d : Float) / (1 + p))

-- The main theorem stating that the number of employees in January is 391
theorem employees_in_january (h : employeesInDec = 450) (h_percent : percentIncrease = 0.15) : 
  employeesInJan employeesInDec percentIncrease = 391 :=
by
  sorry

end employees_in_january_l509_509844


namespace proof_problem_number_of_true_propositions_is_1_l509_509238

theorem proof_problem (p q : Prop) (hp : p) (hq : ¬q) : 
  (¬(p ∧ q) ∧ (p ∨ q) ∧ ¬¬p) :=
by {
  split,
  { intro h,
    exact hq h.right },
  split,
  { left,
    exact hp },
  { exact hp }
}

def number_of_true_propositions (p q : Prop) (hp : p) (hq : ¬q) : ℕ :=
  if ((¬(p ∧ q)) ∧ (p ∨ q) ∧ ¬¬p) then 1 else 0

theorem number_of_true_propositions_is_1 (p q : Prop) (hp : p) (hq : ¬q) : 
  number_of_true_propositions p q hp hq = 1 := sorry

end proof_problem_number_of_true_propositions_is_1_l509_509238


namespace max_value_of_f_when_m_is_neg4_range_of_m_l509_509361

def f (x m : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

-- Part (Ⅰ)
theorem max_value_of_f_when_m_is_neg4 : 
  ∃ x : ℝ, f x (-4) = 2 :=
sorry

-- Part (Ⅱ)
theorem range_of_m (m : ℝ) :
  (∃ x₀ : ℝ, f x₀ m ≥ 1 / m - 4) → m ∈ set.union (set.Iio 0) ({1} : set ℝ) :=
sorry

end max_value_of_f_when_m_is_neg4_range_of_m_l509_509361


namespace abs_eq_condition_l509_509667

theorem abs_eq_condition (x : ℝ) (h : |x - 1| + x = 1) : x ≤ 1 :=
by sorry

end abs_eq_condition_l509_509667


namespace A_is_5_years_older_than_B_l509_509833

-- Given conditions
variables (A B : ℕ) -- A and B are the current ages
variables (x y : ℕ) -- x is the current age of A, y is the current age of B
variables 
  (A_was_B_age : A = y)
  (B_was_10_when_A_was_B_age : B = 10)
  (B_will_be_A_age : B = x)
  (A_will_be_25_when_B_will_be_A_age : A = 25)

-- Define the theorem to prove that A is 5 years older than B: A = B + 5
theorem A_is_5_years_older_than_B (x y : ℕ) (A B : ℕ) 
  (A_was_B_age : x = y) 
  (B_was_10_when_A_was_B_age : y = 10) 
  (B_will_be_A_age : y = x) 
  (A_will_be_25_when_B_will_be_A_age : x = 25): 
  x - y = 5 := 
by sorry

end A_is_5_years_older_than_B_l509_509833


namespace distance_between_stations_l509_509529

theorem distance_between_stations (x v1 v2 d : ℝ)
  (h1 : v1 = 1 / 2)
  (h2: v2 = 1)
  (h3 : d = 244 / 9)
  (h4: (1 / v1) + (1 / (v1/2)) = (1 / v2 - 2 / 3 * x + 1 / 3 * x - d) : Prop)
  : x = 528 := 
by
  sorry

end distance_between_stations_l509_509529


namespace area_ratio_circles_l509_509065

theorem area_ratio_circles (O P X' : Type) (r : ℝ) (h1 : X' = O + (1/3) * (P - O))
: (π * (1/3 * r)^2) / (π * r^2) = 1 / 9 :=
by
  sorry

end area_ratio_circles_l509_509065


namespace original_laborers_l509_509497

theorem original_laborers (x : ℕ) : (x * 8 = (x - 3) * 14) → x = 7 :=
by
  intro h
  sorry

end original_laborers_l509_509497


namespace monotonic_increasing_interval_l509_509436

noncomputable def function_y (x : ℝ) : ℝ :=
  cos^2 (x + π / 4) + sin^2 (x - π / 4)

def is_monotonically_increasing_interval (interval : Set ℝ) (k : ℤ) : Prop :=
  ∀ x₁ x₂ ∈ interval, x₁ < x₂ → function_y x₁ ≤ function_y x₂

theorem monotonic_increasing_interval :
  ∀ (k : ℤ), is_monotonically_increasing_interval (Set.Icc (k * π + π / 4) (k * π + 3 * π / 4)) k :=
sorry

end monotonic_increasing_interval_l509_509436


namespace simplify_rationalize_l509_509414

theorem simplify_rationalize
  : (1 / (1 + (1 / (Real.sqrt 5 + 2)))) = ((Real.sqrt 5 + 1) / 4) := 
sorry

end simplify_rationalize_l509_509414


namespace gcd_possible_values_count_l509_509067

theorem gcd_possible_values_count : 
  ∃ (a b : ℤ), a * b = 180 ∧ 
  let gcd_lcm := Int.gcd a b * Int.lcm a b 
  in ab = gcd_lcm ∧ 
  (∃ S : set ℤ, S = {Int.gcd a b | ∀ (a b : ℤ), a * b = 180} ∧ S.card = 9) :=
begin
  sorry
end

end gcd_possible_values_count_l509_509067


namespace complex_number_of_vertex_C_l509_509202

noncomputable def vertex_A : ℂ := 1 - 3 * I
noncomputable def vertex_B : ℂ := 4 + 2 * I

theorem complex_number_of_vertex_C : vertex_B - vertex_A = 3 + 5 * I := by
  sorry

end complex_number_of_vertex_C_l509_509202


namespace num_sheets_in_stack_l509_509534

variable (sheets_per_ream : ℕ) (height_per_ream : ℝ) (stack_height : ℝ)

theorem num_sheets_in_stack : 
  sheets_per_ream = 400 → 
  height_per_ream = 4 → 
  stack_height = 9 → 
  (stack_height / (height_per_ream / sheets_per_ream) = 900) := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end num_sheets_in_stack_l509_509534


namespace least_integer_of_sum_in_ratio_l509_509960

theorem least_integer_of_sum_in_ratio (a b c : ℕ) (h1 : a + b + c = 90) (h2 : a * 3 = b * 2) (h3 : a * 5 = c * 2) : a = 18 :=
by
  sorry

end least_integer_of_sum_in_ratio_l509_509960


namespace log_sum_l509_509916

theorem log_sum :
  log 5 + log 2 + 2 = 3 :=
by
  --
  sorry

end log_sum_l509_509916


namespace difference_in_gems_l509_509133

theorem difference_in_gems (r d : ℕ) (h : d = 3 * r) : d - r = 2 * r := 
by 
  sorry

end difference_in_gems_l509_509133


namespace motorcycle_uphill_speed_l509_509864

/-- 
A man rides on a motorcycle with a certain speed uphill and 100 kmph downhill.
He takes 12 hours to get from point A to point B and back to A, covering a total distance of 800 km.
Prove that his speed going uphill is 50 kmph.
-/
theorem motorcycle_uphill_speed : 
  ∃ V_up : ℝ, V_up = 50 ∧ (∀ D : ℝ, D = 400 → (D / V_up) + (D / 100) = 12) ∧ (2 * 400 = 800) :=
by {
  use 50, 
  split,
  { refl, },
  split,
  { intros D hD,
    rw hD,
    norm_num, },
  { norm_num, }
}

end motorcycle_uphill_speed_l509_509864


namespace inequality_proving_l509_509353

theorem inequality_proving (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x^2 + y^2 + z^2 = 1) :
  (1 / x + 1 / y + 1 / z) - (x + y + z) ≥ 2 * Real.sqrt 3 :=
by
  sorry

end inequality_proving_l509_509353


namespace parabola_hyperbola_tangent_l509_509048

theorem parabola_hyperbola_tangent :  
  ∃ m : ℝ, (∀ x y : ℝ, y = x^2 + 2 ∧ y^2 - m * x^2 = 1 → m = 4 + 2*real.sqrt 3) :=
by
  sorry

end parabola_hyperbola_tangent_l509_509048


namespace min_shirts_to_save_money_l509_509888

theorem min_shirts_to_save_money :
  ∃ (x : ℕ), 75 + 8 * x < 12 * x ∧ x = 19 :=
sorry

end min_shirts_to_save_money_l509_509888


namespace f_is_odd_f_is_decreasing_f_range_on_interval_l509_509199

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x + y) = f x + f y
axiom negative_for_positive : ∀ x : ℝ, x > 0 → f x < 0
axiom f_neg_1 : f (-1) = 2

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := sorry

theorem f_is_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := sorry

theorem f_range_on_interval : set.range (λ x : ℝ, x ∈ Icc (-2 : ℝ) 4 → f x) = Icc (-8 : ℝ) 4 := sorry

end f_is_odd_f_is_decreasing_f_range_on_interval_l509_509199


namespace path_lengths_l509_509387

variable (x y z : ℝ)

-- Conditions
variable (path1_straight : ∀ t, Path 1 consists of straight-line segments)
variable (path2_semi_and_straight : ∀ t, Path 2 consists of straight-line segments and a semi-circle)
variable (path3_straight : ∀ t, Path 3 consists of straight-line segments)

-- Lean statement based on problem conditions and solution
theorem path_lengths (x y z : ℝ)
  (path1_straight : Path 1 consists of straight-line segments)
  (path2_semi_and_straight : Path 2 consists of straight-line segments and a semi-circle)
  (path3_straight : Path 3 consists of straight-line segments)
  (length_path1 : real.norm x = length of Path 1)
  (length_path2 : real.norm y = length of Path 2)
  (length_path3 : real.norm z = length of Path 3) :
  x = z ∧ z < y := 
sorry

end path_lengths_l509_509387


namespace integer_count_log_inequality_l509_509597

theorem integer_count_log_inequality :
  {x : ℕ | 50 < x ∧ x < 70 ∧ logBase 10 (x - 50) + logBase 10 (70 - x) < 2.3}.finite.card = 9 := 
sorry

end integer_count_log_inequality_l509_509597


namespace interest_difference_l509_509031

theorem interest_difference :
  let P := 625
  let R := 4
  let T := 2
  let SI := P * R * T / 100
  let CI := P * ((1 + R / 100) ^ T - 1)
  SI - CI = 26 :=
by
  let P := 625
  let R := 4
  let T := 2
  let SI := P * R * T / 100
  let CI := P * ((1 + R / 100) ^ T - 1)
  calc
    SI - CI = sorry -- Filled in to ensure it builds

end interest_difference_l509_509031


namespace find_x_l509_509959

theorem find_x (x : ℝ) (h_pos : x > 0) (h_eq : x * (⌊x⌋) = 132) : x = 12 := sorry

end find_x_l509_509959


namespace cylinder_base_radii_l509_509670

theorem cylinder_base_radii {l w : ℝ} (hl : l = 3 * Real.pi) (hw : w = Real.pi) :
  (∃ r : ℝ, l = 2 * Real.pi * r ∧ r = 3 / 2) ∨ (∃ r : ℝ, w = 2 * Real.pi * r ∧ r = 1 / 2) :=
sorry

end cylinder_base_radii_l509_509670


namespace sum_of_possible_x_l509_509493

def mean (lst : List ℝ) := (lst.sum / lst.length)

def mode (lst : List ℝ) := lst.mode

def median (lst : List ℝ) :=
  let sorted := lst.qsort (· ≤ ·)
  if sorted.length % 2 == 1 then sorted.get! (sorted.length / 2)
  else (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2

def is_geometric_sequence (a b c : ℝ) := (a * c = b * b)

def valid_x (x : ℝ) :=
  let lst := [12, 3, 6, 3, 9, 3, x]
  mean lst, median lst, mode lst 
  match (mean lst, median lst, mode lst) with
  | m, med, mod =>
    is_geometric_sequence (min (min med mod) m) (median lst) (max (max med mod) m) 

theorem sum_of_possible_x : ([-15, 48].sum = 33) :=
by
  sorry

end sum_of_possible_x_l509_509493


namespace find_k_eq_two_thirds_l509_509179

def f (x : ℝ) := Real.cot (x / 3) - Real.cot x

theorem find_k_eq_two_thirds (x : ℝ) (hx : x ≠ 0) : 
  f(x) = Real.sin (2 * x / 3) / (Real.sin (x / 3) * Real.sin x) :=
begin
  sorry
end

end find_k_eq_two_thirds_l509_509179


namespace approximation_range_l509_509795

theorem approximation_range (a : ℝ) (h : ∃ a_approx : ℝ, 170 = a_approx ∧ a = real_floor (a_approx) + 0.5) :
  169.5 ≤ a ∧ a < 170.5 :=
sorry

end approximation_range_l509_509795


namespace log_inequality_solution_l509_509056

theorem log_inequality_solution (x : ℝ) (h1 : x > 1) (h2 : log 10 (x - 1) < 1) : 1 < x ∧ x < 11 := 
by
  sorry

end log_inequality_solution_l509_509056


namespace probability_of_getting_4_heads_in_5_tosses_l509_509005

theorem probability_of_getting_4_heads_in_5_tosses :
  let n := 5
  let k := 4
  let p := 0.5
  let binom (n k : ℕ) : ℕ := n.choose k
  let binom_prob (n k : ℕ) (p : ℝ) : ℝ := (binom n k) * p^k * (1 - p)^(n - k)
  in binom_prob n k p = 0.15625 := 
by
  sorry

end probability_of_getting_4_heads_in_5_tosses_l509_509005


namespace find_p_l509_509309

noncomputable def complex_p (f w : ℂ) : ℂ := (15000 + w) / f

theorem find_p :
  let f : ℂ := 10
  let w : ℂ := 10 + 250 * complex.I
  ∃ p : ℂ, f * p - w = 15000 ∧ p = 1501 + 25 * complex.I :=
by
  let f : ℂ := 10
  let w : ℂ := 10 + 250 * complex.I
  existsi (complex_p f w)
  split
  -- First part of the proof
  sorry
  -- Second part of the proof, showing that p = 1501 + 25*i
  sorry

end find_p_l509_509309


namespace partition_exists_l509_509340

def is_even (n : ℕ) := ∃ k : ℕ, n = 2 * k

def is_non_zero_binary_seq (a : list ℕ) (n : ℕ) : Prop :=
  a.length = n ∧ (∀ x ∈ a, x = 0 ∨ x = 1) ∧ a ≠ list.repeat 0 n

def set_A (n : ℕ) : set (list ℕ) :=
  {a | is_non_zero_binary_seq a n}

def can_be_partitioned (s : set (list ℕ)) : Prop :=
  ∃ (T : set (set (list ℕ))), (∀ t ∈ T, t.card = 3) ∧ ∀ t ∈ T, ∀ i, ∃ a b c ∈ t, (list.nth_le a i sorry = 1 ∧ list.nth_le b i sorry = 1) ∨ (list.nth_le a i sorry = 0 ∧ list.nth_le b i sorry = 0) ∧ (list.nth_le c i sorry = 1 ∨ list.nth_le c i sorry = 0)

theorem partition_exists (n : ℕ) (h : is_even n) : can_be_partitioned (set_A n) :=
  sorry

end partition_exists_l509_509340


namespace number_of_games_X_l509_509765

variable (x : ℕ) -- Total number of games played by team X
variable (y : ℕ) -- Wins by team Y
variable (ly : ℕ) -- Losses by team Y
variable (dy : ℕ) -- Draws by team Y
variable (wx : ℕ) -- Wins by team X
variable (lx : ℕ) -- Losses by team X
variable (dx : ℕ) -- Draws by team X

axiom wins_ratio_X : wx = 3 * x / 4
axiom wins_ratio_Y : y = 2 * (x + 12) / 3
axiom wins_difference : y = wx + 4
axiom losses_difference : ly = lx + 5
axiom draws_difference : dy = dx + 3
axiom eq_losses_draws : lx + dx = (x - wx)

theorem number_of_games_X : x = 48 :=
by
  sorry

end number_of_games_X_l509_509765


namespace multiplier_of_difference_l509_509549

variable (x y : ℕ)
variable (h : x + y = 49) (h1 : x > y)

theorem multiplier_of_difference (h2 : x^2 - y^2 = k * (x - y)) : k = 49 :=
by sorry

end multiplier_of_difference_l509_509549


namespace painting_area_l509_509734

def wall_height : ℝ := 10
def wall_length : ℝ := 15
def door_height : ℝ := 3
def door_length : ℝ := 5

noncomputable def area_of_wall : ℝ :=
  wall_height * wall_length

noncomputable def area_of_door : ℝ :=
  door_height * door_length

noncomputable def area_to_paint : ℝ :=
  area_of_wall - area_of_door

theorem painting_area :
  area_to_paint = 135 := by
  sorry

end painting_area_l509_509734


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l509_509228

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l509_509228


namespace find_sin_A_and_c_l509_509325

variables {A B C : ℝ} {a b c: ℝ}

noncomputable def triangle_conditions : Prop :=
  a = 1 ∧ b = 2 * Real.sqrt 3 ∧ B - A = Real.pi / 6

theorem find_sin_A_and_c 
  (h : triangle_conditions)
  : sin A = Real.sqrt 7 / 14 ∧ c = (11 / 7) * Real.sqrt 7 :=
by
  cases h with ha hb
  sorry

end find_sin_A_and_c_l509_509325


namespace awards_distribution_l509_509758

theorem awards_distribution : 
  let awards := 6
  let students := 3
  (∃ (f : Fin awards → Fin students),
    (∀ (s : Fin students), ∃ (a₁ a₂ a₃: Fin awards), f a₁ = s ∧ f a₂ = s ∧ f a₃ = s) ∧ 
    (Finset.card (Finset.image f (Finset.univ : Finset (Fin awards))) = 465)
  :=
sorry

end awards_distribution_l509_509758


namespace sqrt_plus_inv_sqrt_eq_l509_509720

noncomputable def sqrt_plus_inv_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x + 1 / Real.sqrt x

theorem sqrt_plus_inv_sqrt_eq (x : ℝ) (h₁ : 0 < x) (h₂ : x + 1 / x = 50) :
  sqrt_plus_inv_sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_plus_inv_sqrt_eq_l509_509720


namespace balancing_process_leads_to_balanced_triple_l509_509548

-- Define the structure of an ordered triple with conditions
structure Triple :=
  (x₁ x₂ x₃ : ℝ)
  (pos_irrational_x₁ : ∃ (a : ℚ), a > 0 ∧ a ∉ ℝ)
  (pos_irrational_x₂ : ∃ (a : ℚ), a > 0 ∧ a ∉ ℝ)
  (pos_irrational_x₃ : ∃ (a : ℚ), a > 0 ∧ a ∉ ℝ)
  (sum_eq_one : x₁ + x₂ + x₃ = 1)

-- Define the property of being balanced
def balanced (t : Triple) : Prop :=
  t.x₁ < 0.5 ∧ t.x₂ < 0.5 ∧ t.x₃ < 0.5

-- Define the balancing act
def B (t : Triple) : Triple :=
  if h : t.x₁ > 0.5 then
    ⟨2 * t.x₁ - 1, 2 * t.x₂, 2 * t.x₃,
     sorry, sorry, sorry,
     by field_simp [t.sum_eq_one]; ring⟩
  else if h : t.x₂ > 0.5 then
    ⟨2 * t.x₁, 2 * t.x₂ - 1, 2 * t.x₃,
     sorry, sorry, sorry,
     by field_simp [t.sum_eq_one]; ring⟩
  else
    ⟨2 * t.x₁, 2 * t.x₂, 2 * t.x₃ - 1,
     sorry, sorry, sorry,
     by field_simp [t.sum_eq_one]; ring⟩

-- Define the main theorem
theorem balancing_process_leads_to_balanced_triple :
  ∀ (t : Triple), ∃ (n : ℕ), balanced (nat.iterate B n t) :=
by
  sorry

end balancing_process_leads_to_balanced_triple_l509_509548


namespace simplify_rationalize_l509_509405

theorem simplify_rationalize : (1 : ℝ) / (1 + (1 / (real.sqrt 5 + 2))) = (real.sqrt 5 + 1) / 4 :=
by sorry

end simplify_rationalize_l509_509405


namespace angle_in_first_quadrant_l509_509999

theorem angle_in_first_quadrant (x : ℝ) (h1 : Real.tan x > 0) (h2 : Real.sin x + Real.cos x > 0) : 
  0 < Real.sin x ∧ 0 < Real.cos x := 
by 
  sorry

end angle_in_first_quadrant_l509_509999


namespace profit_percentage_each_portion_l509_509120

theorem profit_percentage_each_portion (P : ℝ) (total_apples : ℝ) 
  (portion1_percentage : ℝ) (portion2_percentage : ℝ) (total_profit_percentage : ℝ) :
  total_apples = 280 →
  portion1_percentage = 0.4 →
  portion2_percentage = 0.6 →
  total_profit_percentage = 0.3 →
  portion1_percentage * P + portion2_percentage * P = total_profit_percentage →
  P = 0.3 :=
by
  intros
  sorry

end profit_percentage_each_portion_l509_509120


namespace sum_a3_a4_eq_14_l509_509988

open Nat

-- Define variables
def S (n : ℕ) : ℕ := n^2 + n
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem sum_a3_a4_eq_14 : a 3 + a 4 = 14 := by
  sorry

end sum_a3_a4_eq_14_l509_509988


namespace original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l509_509432

-- Condition declarations
variables (x y : ℕ)
variables (hx : x > 0) (hy : y > 0)
variables (h_sum : x + y = 25)
variables (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196)

-- Lean 4 statements for the proof problem
theorem original_rectangle_perimeter (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 25) : 2 * (x + y) = 50 := by
  sorry

theorem difference_is_multiple_of_7 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196) : 7 ∣ ((x + 5) * (y + 5) - (x - 2) * (y - 2)) := by
  sorry

theorem relationship_for_seamless_combination (x y : ℕ) (h_sum : x + y = 25) (h_seamless : (x+5)*(y+5) = (x*(y+5))) : x = y + 5 := by
  sorry

end original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l509_509432


namespace tim_zoo_cost_l509_509468

theorem tim_zoo_cost :
  let number_of_goats := 3 in
  let cost_per_goat := 400 in
  let number_of_llamas := 2 * number_of_goats in
  let cost_per_llama := 1.5 * cost_per_goat in
  let total_cost := (number_of_goats * cost_per_goat) + (number_of_llamas * cost_per_llama) in
  total_cost = 4800 :=
by
  -- definition simplifications
  let number_of_goats := 3
  let cost_per_goat := 400
  let number_of_llamas := 2 * number_of_goats
  let cost_per_llama := 1.5 * cost_per_goat
  let total_cost := (number_of_goats * cost_per_goat) + (number_of_llamas * cost_per_llama)
  -- the target proof
  have : number_of_goats = 3 := rfl
  have : cost_per_goat = 400 := rfl
  have : number_of_llamas = 6 := by simp [number_of_goats]
  have : cost_per_llama = 600 := by simp [cost_per_goat, mul_assoc, mul_comm]
  have : total_cost = 4800 := by simp [total_cost, number_of_goats, cost_per_goat, number_of_llamas, cost_per_llama, mul_add, mul_comm, mul_assoc, add_comm]
  exact this

end tim_zoo_cost_l509_509468


namespace expected_profit_may_is_3456_l509_509735

-- Given conditions as definitions
def february_profit : ℝ := 2000
def april_profit : ℝ := 2880
def growth_rate (x : ℝ) : Prop := (2000 * (1 + x)^2 = 2880)

-- The expected profit in May
def expected_may_profit (x : ℝ) : ℝ := april_profit * (1 + x)

-- The theorem to be proved based on the given conditions
theorem expected_profit_may_is_3456 (x : ℝ) (h : growth_rate x) (h_pos : x = (1:ℝ)/5) : 
    expected_may_profit x = 3456 :=
by sorry

end expected_profit_may_is_3456_l509_509735


namespace triangle_sides_sum_l509_509472

noncomputable def sum_of_remaining_sides : ℝ :=
  let angleA := 60.0 in
  let angleC := 30.0 in
  let sideBC := 6.0 in
  let BD := sideBC / 2 in
  let AD := BD * Real.sqrt 3 in
  let AC := AD + BD * Real.sqrt 3 in
  let sideAB := 2 * BD in
  let sum := AC + sideAB in
  Real.round (sum * 10) / 10

theorem triangle_sides_sum :
  let angleA := 60 in
  let angleC := 30 in
  let sideBC := 6 in
  sum_of_remaining_sides = 16.4 :=
by
  let angleA := 60.0
  let angleC := 30.0
  let sideBC := 6.0
  let BD := sideBC / 2
  let AD := BD * Real.sqrt 3
  let AC := AD + BD * Real.sqrt 3
  let sideAB := 2 * BD
  let sum := AC + sideAB
  have h : Real.round (sum * 10) / 10 = 16.4 := sorry
  exact h

end triangle_sides_sum_l509_509472


namespace simplify_expression_l509_509757

open Complex

theorem simplify_expression :
  ((4 + 6 * I) / (4 - 6 * I) * (4 - 6 * I) / (4 + 6 * I) + (4 - 6 * I) / (4 + 6 * I) * (4 + 6 * I) / (4 - 6 * I)) = 2 :=
by
  sorry

end simplify_expression_l509_509757


namespace find_first_term_of_geometric_series_l509_509900

theorem find_first_term_of_geometric_series 
  (r : ℚ) (S : ℚ) (a : ℚ) 
  (hr : r = -1/3) (hS : S = 9)
  (h_sum_formula : S = a / (1 - r)) : 
  a = 12 := 
by
  sorry

end find_first_term_of_geometric_series_l509_509900


namespace volume_and_radius_of_pyramid_l509_509428

noncomputable def cone_and_pyramid :=
  let h := 4
  let slant_height := 5
  let OB := 3
  let r := real.sqrt (slant_height^2 - h^2)
  let base_area := real.sqrt(3) * slant_height^2 / 4
  let V := (1 / 3 : ℝ) * base_area * h 
  let R := 10
  ⟨h, slant_height, OB, r, base_area, V, R⟩

theorem volume_and_radius_of_pyramid :
  let ⟨h, slant_height, OB, r, base_area, V, R⟩ := cone_and_pyramid
  V = 50 ∧ R = 10 :=
by {
  sorry
}

end volume_and_radius_of_pyramid_l509_509428


namespace angles_of_triangle_BMN_l509_509779

noncomputable theory

-- Given condition definitions
def exterior_angle_A : ℝ := 115
def exterior_angle_C : ℝ := 140

-- theorem statement
theorem angles_of_triangle_BMN :
  ∀ (A B C M N : Type)
  (exterior_angle_A : ℝ)
  (exterior_angle_C : ℝ)
  (is_parallel : ∀ (l1 l2 : Type), l1 ∥ l2)
  (line_AC_parallel : ∀ P Q, is_parallel (AC P Q) (AC M N))
  (triangle_angles : ∀ (α β γ : ℝ),
    (exterior_angle p q) -- widely adapt for variables function oversight by all generality matching curryingCalculi
       =(α + β = 115 ∧ α + γ = 140) in abc converge angles 

  ∃ ∠BMN,
  ∠BMN(B M N) in ∆] --henchi-span true affix result orthogonal sum + cum direct law derivation

   ∧ deconstruct_eqs (α+ϼ) =  180 -
     ((α == 37.5) + (β == 40) + (γ ==∥65.0))

  sorry

end angles_of_triangle_BMN_l509_509779


namespace median_of_list_3030_l509_509929

def median_of_list (n : ℕ) : ℝ :=
  let natural_numbers := (List.range (n + 1)).tail
  let squared_numbers := natural_numbers.map (λ x : ℕ => x * x)
  let cubed_numbers := natural_numbers.map (λ x : ℕ => x * x * x)
  let full_list := natural_numbers ++ squared_numbers ++ cubed_numbers
  let sorted_list := full_list.toArray.qsort (· < ·)
  if sorted_list.size % 2 = 1 then
    sorted_list[sorted_list.size / 2].toReal
  else
    (sorted_list[sorted_list.size / 2 - 1].toReal + sorted_list[sorted_list.size / 2].toReal) / 2

theorem median_of_list_3030 : median_of_list 3030 = 2297340.5 := by
  sorry

end median_of_list_3030_l509_509929


namespace pies_with_no_ingredients_l509_509008

theorem pies_with_no_ingredients (total_pies : ℕ)
  (pies_with_chocolate : ℕ)
  (pies_with_blueberries : ℕ)
  (pies_with_vanilla : ℕ)
  (pies_with_almonds : ℕ)
  (H_total : total_pies = 60)
  (H_chocolate : pies_with_chocolate = 1 / 3 * total_pies)
  (H_blueberries : pies_with_blueberries = 3 / 4 * total_pies)
  (H_vanilla : pies_with_vanilla = 2 / 5 * total_pies)
  (H_almonds : pies_with_almonds = 1 / 10 * total_pies) :
  ∃ (pies_without_ingredients : ℕ), pies_without_ingredients = 15 :=
by
  sorry

end pies_with_no_ingredients_l509_509008


namespace find_m_value_l509_509298

open Real

def tangent_perpendicular_to_line (m : ℝ) : Prop :=
  let k := (exp 1) in          -- slope of the tangent line at x = 1
  let perpendicular_slope := -2 / m in
  k * perpendicular_slope = -1

theorem find_m_value (m : ℝ) : tangent_perpendicular_to_line m → m = 2 * exp 1 :=
sorry

end find_m_value_l509_509298


namespace min_value_four_points_l509_509963

theorem min_value_four_points 
  (P₁ P₂ P₃ P₄ : ℝ²) :
  (∃ (d₁ d₂ d₃ d₄ d₅ d₆ : ℝ), 
    (d₁ = dist P₁ P₂) ∧ 
    (d₂ = dist P₁ P₃) ∧ 
    (d₃ = dist P₁ P₄) ∧ 
    (d₄ = dist P₂ P₃) ∧ 
    (d₅ = dist P₂ P₄) ∧ 
    (d₆ = dist P₃ P₄) ∧ 
    (d₁ > 0) ∧ (d₂ > 0) ∧ (d₃ > 0) ∧ (d₄ > 0) ∧ (d₅ > 0) ∧ (d₆ > 0) ∧ 
    let min_d := min (min (min d₁ d₂) (min d₃ d₄)) (min d₅ d₆) in
    let sum_d := d₁ + d₂ + d₃ + d₄ + d₅ + d₆ in
    min_d = 1 → sum_d = 4 + 2 * real.sqrt 2) :=
sorry

end min_value_four_points_l509_509963


namespace survey_pop_and_coke_l509_509304

theorem survey_pop_and_coke (total_people : ℕ) (angle_pop angle_coke : ℕ) 
  (h_total : total_people = 500) (h_angle_pop : angle_pop = 240) (h_angle_coke : angle_coke = 90) :
  ∃ (pop_people coke_people : ℕ), pop_people = 333 ∧ coke_people = 125 :=
by 
  sorry

end survey_pop_and_coke_l509_509304


namespace fraction_zero_l509_509255

theorem fraction_zero (x : ℝ) (h : (x - 1) * (x + 2) = 0) (hne : x^2 - 1 ≠ 0) : x = -2 :=
by
  sorry

end fraction_zero_l509_509255


namespace problem_proof_l509_509669

def M : ℚ := 28
def N : ℚ := 147

theorem problem_proof : (M - N) = -119 := by
  -- Given conditions
  have hM : (4 : ℚ) / 7 = M / 49 := by
    rw [M]
    norm_num
  have hN : (4 : ℚ) / 7 = 84 / N := by
    rw [N]
    norm_num
  -- Prove the required result
  calc
    M - N = 28 - 147 := by rw [M, N]
    ... = -119 := by norm_num

end problem_proof_l509_509669


namespace speed_of_current_l509_509865

theorem speed_of_current (m c : ℝ) (h1 : m + c = 18) (h2 : m - c = 11.2) : c = 3.4 :=
by
  sorry

end speed_of_current_l509_509865


namespace orange_ring_weight_l509_509332

theorem orange_ring_weight :
  ∀ (p w t o : ℝ), 
  p = 0.33 → w = 0.42 → t = 0.83 → t - (p + w) = o → 
  o = 0.08 :=
by
  intro p w t o hp hw ht h
  rw [hp, hw, ht] at h
  -- Additional steps would go here, but
  sorry -- Skipping the proof as instructed

end orange_ring_weight_l509_509332


namespace value_of_a_l509_509976

theorem value_of_a (x : ℝ) (n : ℕ) (h : x > 0) (h_n : n > 0) :
  (∀ k : ℕ, 1 ≤ k → k ≤ n → x + k ≥ k + 1) → a = n^n :=
by
  sorry

end value_of_a_l509_509976


namespace cookies_divided_among_tins_l509_509516

noncomputable def fraction_of_cookies_in_blue_tin (total_cookies : ℝ) (cookies_in_blue_green : ℝ)
  (cookies_in_red : ℝ) (fraction_green : ℝ) (fraction_blue : ℝ) : ℝ :=
  cookies_in_blue_green * fraction_blue / total_cookies

theorem cookies_divided_among_tins :
  ∀ (C : ℝ), 2 / 3 * C = C * 2 / 3 →
  1 / 3 * C = C * 1 / 3 →
  5 / 9 * (2 / 3 * C) = 2 / 3 * C * 5 / 9 →
  4 / 9 * (2 / 3 * C) = 2 / 3 * C * 4 / 9 →
  fraction_of_cookies_in_blue_tin C (2 / 3 * C) (C * 1 / 3) (5 / 9) (4 / 9) = 8 / 27 :=
by
  intro C
  intro h1
  intro h2
  intro h3
  intro h4
  rw [fraction_of_cookies_in_blue_tin, h1, h2, h3, h4]
  sorry

end cookies_divided_among_tins_l509_509516


namespace percentage_of_small_bottles_sold_l509_509873

variables (x : ℕ) -- Representing percentage as a natural number for simplicity

-- Define initial conditions
def initial_small_bottles : ℕ := 5000
def initial_big_bottles : ℕ := 12000
def percentage_big_sold : ℕ := 18
def remaining_bottles : ℕ := 14090

-- Calculation of bottles sold
def small_bottles_sold : ℕ := x * initial_small_bottles / 100
def big_bottles_sold : ℕ := percentage_big_sold * initial_big_bottles / 100

-- Final condition for remaining bottles
def total_remaining_bottles : ℕ :=
  initial_small_bottles - small_bottles_sold +
  initial_big_bottles - big_bottles_sold

theorem percentage_of_small_bottles_sold :
  total_remaining_bottles x = remaining_bottles → x = 15 :=
by
  -- Proof to be provided by the user
  sorry

end percentage_of_small_bottles_sold_l509_509873


namespace sequence_formula_and_sum_l509_509201

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) (d : α) :=
  ∀ m n, a(m) = a(1) + (m - n) * d

def is_geometric_mean (a : ℕ → α) :=
  a 5 = Real.sqrt (a 2 * a 14)

theorem sequence_formula_and_sum 
  (a : ℕ → α)
  (d : α)
  (h_arith : is_arithmetic_sequence a d)
  (h_cond1 : a 3 + a 8 = 20)
  (h_cond2 : is_geometric_mean a)
  (b : ℕ → α := λ n, 1 / (a n * a (n + 1)))
  :
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, ∑ i in Finset.range n, b i = n / (2 * n + 1)) 
:= 
  by sorry

end sequence_formula_and_sum_l509_509201


namespace distance_between_points_l509_509588

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem distance_between_points :
  distance 3.5 (-1.2) (-0.5) 2.3 = 5.3125 :=
by
  sorry

end distance_between_points_l509_509588


namespace mappings_f_y_eq_2_l509_509996

open Function

-- Define the sets P and Q
def P : Finset (fin 3) := {0, 1, 2}
def Q : Finset (fin 3) := {0, 1, 2}

-- Define the condition that f(y) = 2
def condition (f : P → Q) : Prop := f 1 = 2

-- Statement: the number of such functions is 9
theorem mappings_f_y_eq_2 : (P → Q).filter condition).card = 9 := sorry

end mappings_f_y_eq_2_l509_509996


namespace job_interview_probability_l509_509808

open_locale big_operators

theorem job_interview_probability (n : ℕ) (h : (nat.choose 2 2) * (nat.choose (n-2) 1) / (nat.choose n 3) = 1 / 70) : n = 21 :=
sorry

end job_interview_probability_l509_509808


namespace total_toes_on_bus_l509_509373

/-- Definition for the number of toes a Hoopit has -/
def toes_per_hoopit : ℕ := 4 * 3

/-- Definition for the number of toes a Neglart has -/
def toes_per_neglart : ℕ := 5 * 2

/-- Definition for the total number of Hoopits on the bus -/
def hoopit_students_on_bus : ℕ := 7

/-- Definition for the total number of Neglarts on the bus -/
def neglart_students_on_bus : ℕ := 8

/-- Proving that the total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus : hoopit_students_on_bus * toes_per_hoopit + neglart_students_on_bus * toes_per_neglart = 164 := by
  sorry

end total_toes_on_bus_l509_509373


namespace part_a_l509_509842

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def between_1_and_100 (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 100

def condition_a (n : ℕ) : Prop :=
  ¬ (n ∣ (Nat.factorial (n - 1)))

def solution_a (n : ℕ) : Prop :=
  (is_prime n ∨ n = 4)

theorem part_a :
  ∀ n : ℕ, between_1_and_100 n → (condition_a n ↔ solution_a n) :=
begin
  sorry
end

end part_a_l509_509842


namespace odd_function_f_x_l509_509723

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x * (1 - x) else x * (1 + x)

theorem odd_function_f_x (x : ℝ) (h_odd : ∀ x, f(-x) = -f(x)) (h_pos : x > 0 → f(x) = x * (1 - x)) : 
  (x < 0 → f(x) = x * (1 + x)) :=
by
  sorry

end odd_function_f_x_l509_509723


namespace simplify_expr_l509_509410

noncomputable def expr := 1 / (1 + 1 / (Real.sqrt 5 + 2))
noncomputable def simplified := (Real.sqrt 5 + 1) / 4

theorem simplify_expr : expr = simplified :=
by
  sorry

end simplify_expr_l509_509410


namespace percent_increase_area_exposed_l509_509773

theorem percent_increase_area_exposed : 
  let r₁ := 10
  let r₂ := 15
  let pi := Real.pi
  let initial_inner_area := pi * r₁ ^ 2
  let initial_outer_area := pi * r₂ ^ 2
  let initial_total_area := initial_outer_area - initial_inner_area
  let new_r₁ := r₁ * (1 + 0.35)
  let new_r₂ := r₂ * (1 - 0.45)
  let new_inner_area := pi * new_r₁ ^ 2
  let new_outer_area := pi * new_r₂ ^ 2
  let new_total_area := new_inner_area + new_outer_area

  new_total_area / initial_total_area * 100 - 100 = 100.25
:= by
  let percent_increase := (new_total_area - initial_total_area) / initial_total_area * 100
  show percent_increase = 100.25
  sorry

end percent_increase_area_exposed_l509_509773


namespace total_seats_taken_l509_509696

theorem total_seats_taken (
  tables_in_group_A : ℕ := 10
  tables_in_group_B : ℕ := 7
  tables_in_group_C : ℕ := 8
  capacity_group_A : ℕ := 8
  capacity_group_B : ℕ := 12
  capacity_group_C : ℕ := 10
  unseated_ratio_A : ℚ := 1/4
  unseated_ratio_B : ℚ := 1/3
  unseated_ratio_C : ℚ := 1/5
) : 
  let seats_taken_group_A := tables_in_group_A * (capacity_group_A - (unseated_ratio_A * capacity_group_A).toNat)
  let seats_taken_group_B := tables_in_group_B * (capacity_group_B - (unseated_ratio_B * capacity_group_B).toNat)
  let seats_taken_group_C := tables_in_group_C * (capacity_group_C - (unseated_ratio_C * capacity_group_C).toNat)
  seats_taken_group_A + seats_taken_group_B + seats_taken_group_C = 180 :=
by
  sorry

end total_seats_taken_l509_509696


namespace collinear_A_D_T_l509_509206

-- Define the inputs and construct the problem in Lean.
variables {A B C D E F T : Type} 
variables [triangle ABC : Type] [excircle_tangent BC D : Type]
          [extension_meets_excircle AB AC E F : Type]
          [line_intersection BF CE T : Type]

-- We state that points A, D, and T are collinear
theorem collinear_A_D_T
  (h_triangle : triangle ABC)
  (h_excircle_tangent : excircle_tangent BC D)
  (h_extension_meets_excircle : extension_meets_excircle AB AC E F)
  (h_line_intersection : line_intersection BF CE T) :
  collinear A D T :=
sorry

end collinear_A_D_T_l509_509206


namespace proportion_equation_correct_l509_509605

theorem proportion_equation_correct (x y : ℝ) (h1 : 2 * x = 3 * y) (h2 : x ≠ 0) (h3 : y ≠ 0) : 
  x / 3 = y / 2 := 
  sorry

end proportion_equation_correct_l509_509605


namespace find_t_l509_509661

variables (V V₀ g a S t : ℝ)

-- Conditions
axiom eq1 : V = 3 * g * t + V₀
axiom eq2 : S = (3 / 2) * g * t^2 + V₀ * t + (1 / 2) * a * t^2

-- Theorem to prove
theorem find_t : t = (9 * g * S) / (2 * (V - V₀)^2 + 3 * V₀ * (V - V₀)) :=
by
  sorry

end find_t_l509_509661


namespace find_possible_values_of_n_l509_509127

theorem find_possible_values_of_n (n : ℕ) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ 
    (2*n*(2*n + 1))/2 - (n*k + (n*(n-1))/2) = 1615) ↔ (n = 34 ∨ n = 38) :=
by
  sorry

end find_possible_values_of_n_l509_509127


namespace product_of_solutions_l509_509759

theorem product_of_solutions : 
  let x1 := 10
  let x2 := 4
  |2*x1 - 14| - 5 = 1 ∧ |2*x2 - 14| - 5 = 1 → x1 * x2 = 40 :=
by
  sorry

end product_of_solutions_l509_509759


namespace lower_bound_of_quadratic_function_l509_509649

theorem lower_bound_of_quadratic_function (a : ℝ) (h : a ≠ 0) : 
  ∃ M, (∀ a, (a^2 - 4 * a + 6) ≥ M) ∧ (∀ M' > M, ∃ a, (a^2 - 4 * a + 6) < M') :=
begin
  use 2,
  split,
  { intro a,
    calc
      a^2 - 4 * a + 6 = (a - 2)^2 + 2 : by ring
                    ... ≥ 2           : by linarith [(pow_two_nonneg (a - 2))] },
  { intros M' hM',
    have h := calc
      M' > 2 : hM',
    -- Note: the proof step to derive a contradiction would normally appear here
    sorry }
end

end lower_bound_of_quadratic_function_l509_509649


namespace part_a_part_b_l509_509741

open Matrix

-- Chessboard is represented as a 6x6 matrix
def chessboard := matrix (fin 6) (fin 6) ℕ

-- Function to determine if all unoccupied squares are red
def all_red (rooks : list (fin 6 × fin 6)) (board : chessboard) : Prop :=
  ∀ (i j : fin 6), i ≠ j →
    board i j > 0 → ∃ (r1 r2 : fin 6 × fin 6), r1 ∈ rooks ∧ r2 ∈ rooks ∧
      r1.fst = r2.fst ∧ r1.snd = r2.snd ∧
      abs (r1.fst.val - i.val) = abs (r2.fst.val - i.val)

-- Function to determine if all unoccupied squares are blue
def all_blue (rooks : list (fin 6 × fin 6)) (board : chessboard) : Prop :=
  ∀ (i j : fin 6), i ≠ j →
    board i j > 0 →
    ∃ (r1 r2 : fin 6 × fin 6), r1 ∈ rooks ∧ r2 ∈ rooks ∧
      r1.fst ≠ r2.fst ∨ r1.snd ≠ r2.snd ∨
      abs (r1.fst.val - i.val) ≠ abs (r2.fst.val - i.val)

-- Prove that there is a configuration of rooks on a 6x6 chessboard such that all unoccupied squares are red.
theorem part_a : ∃ (rooks : list (fin 6 × fin 6)), all_red rooks (λ i j, if (i, j) ∈ rooks then 0 else 1) :=
by
  sorry

-- Prove that there is no configuration of rooks on a 6x6 chessboard such that all unoccupied squares are blue.
theorem part_b : ¬ (∃ (rooks : list (fin 6 × fin 6)), all_blue rooks (λ i j, if (i, j) ∈ rooks then 0 else 1)) :=
by
  sorry

end part_a_part_b_l509_509741


namespace smallest_yummy_integer_l509_509753

def is_yummy (B : ℤ) : Prop :=
  ∃ (n : ℤ), n ≥ 0 ∧ ∑ i in finset.range (n.to_nat + 1), B + i = 2023

theorem smallest_yummy_integer : ∀ B : ℤ, (is_yummy B) → B ≥ -2022 := 
begin
  intro B,
  intro h,
  sorry
end

end smallest_yummy_integer_l509_509753


namespace length_of_BC_l509_509544

theorem length_of_BC :
  ∃ a : ℝ, (∀ (A B C : ℝ × ℝ), 
  A = (0, 0) ∧ 
  B = (-a, 4 * a^2) ∧ 
  C = (a, 4 * a^2) ∧ 
  (4 * a^3 = 128) ∧
  BC = (C.1 - B.1) → BC = 4 * real.cbrt 4) := sorry

end length_of_BC_l509_509544


namespace tim_zoo_cost_l509_509467

theorem tim_zoo_cost :
  let number_of_goats := 3 in
  let cost_per_goat := 400 in
  let number_of_llamas := 2 * number_of_goats in
  let cost_per_llama := 1.5 * cost_per_goat in
  let total_cost := (number_of_goats * cost_per_goat) + (number_of_llamas * cost_per_llama) in
  total_cost = 4800 :=
by
  -- definition simplifications
  let number_of_goats := 3
  let cost_per_goat := 400
  let number_of_llamas := 2 * number_of_goats
  let cost_per_llama := 1.5 * cost_per_goat
  let total_cost := (number_of_goats * cost_per_goat) + (number_of_llamas * cost_per_llama)
  -- the target proof
  have : number_of_goats = 3 := rfl
  have : cost_per_goat = 400 := rfl
  have : number_of_llamas = 6 := by simp [number_of_goats]
  have : cost_per_llama = 600 := by simp [cost_per_goat, mul_assoc, mul_comm]
  have : total_cost = 4800 := by simp [total_cost, number_of_goats, cost_per_goat, number_of_llamas, cost_per_llama, mul_add, mul_comm, mul_assoc, add_comm]
  exact this

end tim_zoo_cost_l509_509467


namespace sum_modulo_7_remainder_l509_509480

open Nat

/- 
Theorem: The modulo 7 remainder of the sum 1 + 2 + 3 + ... + 1000 is 0.
-/
theorem sum_modulo_7_remainder : (∑ i in Finset.range 1001, i) % 7 = 0 :=
sorry

end sum_modulo_7_remainder_l509_509480


namespace integral_of_segment_l509_509552

noncomputable def integral_segment : ℝ :=
∫ x in Segment (0, -2) (4, 0), (1 / sqrt (|x.1| + |x.2|))

theorem integral_of_segment : 
  integral_segment = 2 * (sqrt 20 - sqrt 10) :=
sorry

end integral_of_segment_l509_509552


namespace tangent_circumcircles_l509_509695

-- Define your geometrical objects and conditions
variable {A B C D P Q R : Point}
variable (ℓ : Line) 

-- Assuming the conditions in the problem
axiom quadrilateral_ABCD (quadrilateral : Quadrilateral)
  (h₁ : quadrilateral.has_angle_ABC = 90)
  (h₂ : quadrilateral.has_angle_CDA = 90) : 
    quadrilateral.A = A ∧
    quadrilateral.B = B ∧
    quadrilateral.C = C ∧
    quadrilateral.D = D

axiom pts_PQR 
  (h₃ : P = AC ∩ BD)
  (h₄ : Q = AB ∩ CD)
  (h₅ : R = AD ∩ BC)

axiom line_l (h₆ : ℓ.is_midline_of_triangle P Q R)
  (h₇ : ℓ.parallel QR)

-- The statement of the proof
theorem tangent_circumcircles :
  (∃ O1 O2 : Circle, 
    (Triangle_circumcircle ⟨AB, AD, ℓ⟩ O1) ∧ 
    (Triangle_circumcircle ⟨CD, CB, ℓ⟩ O2) ∧
    (O1.is_tangent_to O2)) :=
by sorry

end tangent_circumcircles_l509_509695


namespace work_done_by_gas_l509_509382

theorem work_done_by_gas (n : ℕ) (R T0 Pa : ℝ) (V0 : ℝ) (W : ℝ) :
  -- Conditions
  n = 1 ∧
  R = 8.314 ∧
  T0 = 320 ∧
  Pa * V0 = n * R * T0 ∧
  -- Question Statement and Correct Answer
  W = Pa * V0 / 2 →
  W = 665 :=
by sorry

end work_done_by_gas_l509_509382


namespace hotel_max_revenue_l509_509521

theorem hotel_max_revenue (rooms : ℕ) (days : ℕ) (room_1_empty : ℕ) (room_20_empty : ℕ) (price_per_room_per_day : ℕ) :
  rooms = 20 → 
  days = 100 → 
  room_1_empty = 1 → 
  room_20_empty = 100 → 
  price_per_room_per_day = 1 →
  ∃ max_revenue, max_revenue ≤ 1996 :=
by {
  intros h1 h2 h3 h4 h5,
  use 1996,
  apply le_refl,
}

end hotel_max_revenue_l509_509521


namespace cost_of_one_dozen_pens_is_780_l509_509427

-- Defining the cost of pens and pencils
def cost_of_pens (n : ℕ) := n * 65

def cost_of_pencils (m : ℕ) := m * 13

-- Given conditions
def total_cost (x y : ℕ) := cost_of_pens x + cost_of_pencils y

theorem cost_of_one_dozen_pens_is_780
  (h1 : total_cost 3 5 = 260)
  (h2 : 65 = 5 * 13)
  (h3 : 65 = 65) :
  12 * 65 = 780 := by
    sorry

end cost_of_one_dozen_pens_is_780_l509_509427


namespace total_tickets_proof_l509_509594

-- Definitions of the variables based on the given problem
def total_amount_collected (student_tickets nonstudent_tickets : ℕ) :=
  2 * student_tickets + 3 * nonstudent_tickets

def total_tickets_sold (student_tickets nonstudent_tickets : ℕ) :=
  student_tickets + nonstudent_tickets

-- Conditions given in the problem
def student_tickets_sold : ℕ := 530
def amount_collected : ℕ := 1933

-- Statement to prove: total tickets sold equals 821
theorem total_tickets_proof : ∃ nonstudent_tickets, total_amount_collected student_tickets_sold nonstudent_tickets = amount_collected ∧ total_tickets_sold student_tickets_sold nonstudent_tickets = 821 :=
begin
  -- Here we need to demonstrate the correct numbers that satisfy the conditions
  sorry
end

end total_tickets_proof_l509_509594


namespace certain_event_A_l509_509893

def isCertainEvent {α : Type} (event : α -> Prop) : Prop :=
  ∀ x, event x

def eventA : Prop :=
  ∀ balls : list (fin 3), balls = [red, white, red] → ∀ drawn : list (fin 3), list.length drawn = 2 → ∃ r, r ∈ drawn ∧ r = red

def eventB : Prop :=
  ∃ d, d ∈ {0, 2, 4, 6, 8} ∨ d ∈ {1, 3, 5, 7, 9}

def eventC : Prop :=
  Toss coin = Heads ∨ Toss coin = Tails

def eventD : Prop :=
  ∃ weather, weather = Rain ∨ weather = NoRain

theorem certain_event_A : isCertainEvent (fun _ => eventA) :=
sorry

end certain_event_A_l509_509893


namespace expected_profit_correct_l509_509805

-- Define the conditions
def ticket_cost : ℝ := 2
def winning_probability : ℝ := 0.01
def prize : ℝ := 50

-- Define the expected profit calculation
def expected_profit : ℝ := (winning_probability * prize) - ticket_cost

-- The theorem we want to prove
theorem expected_profit_correct : expected_profit = -1.5 := by
  sorry

end expected_profit_correct_l509_509805


namespace sample_mean_unbiased_unbiased_variance_unbiased_t_distribution_confidence_interval_μ_confidence_interval_σ_l509_509396

variables {α : Type*} [Fintype α] 
variables (a σ : ℝ) (n : ℕ) (xi : α → ℝ)

-- Define sample mean
def sample_mean (xi : α → ℝ) : ℝ := 
  (1 / n) * ∑ i, xi i 

-- Define unbiased estimator for variance
def unbiased_variance (xi : α → ℝ) (x̄ : ℝ) : ℝ := 
  (1 / (n - 1)) * ∑ i, (xi i - x̄) ^ 2

-- Define unbiased estimator for standard deviation
def unbiased_sd (xi : α → ℝ) (x̄ : ℝ) : ℝ := 
  sqrt (unbiased_variance xi x̄)

-- Sample mean as unbiased estimator of a 
theorem sample_mean_unbiased (h: n > 1) : 
  (sample_mean xi) = a := 
begin
  sorry
end

-- Unbiased variance as unbiased estimator of σ^2 
theorem unbiased_variance_unbiased (h: n > 1) : 
  (unbiased_variance xi (sample_mean xi)) = σ^2 := 
begin
  sorry
end

-- t_{n-1}(x) follows Student's t-distribution with n-1 degrees of freedom
theorem t_distribution (h: n > 1) : 
  ∃ t : α, (t (xi)) = (sample_mean xi - a) / (unbiased_sd xi (sample_mean xi) / sqrt n) := 
begin
  sorry
end

-- Construct 1-α confidence intervals for a
theorem confidence_interval_μ (h: n > 1) (α : ℝ) : 
  ∃ b, (b = sample_mean xi ± (Quantile.t_inv (α/2) (n - 1)) * (unbiased_sd xi (sample_mean xi) / sqrt n)) := 
begin
  sorry
end

-- Construct 1-α confidence intervals for σ
theorem confidence_interval_σ (h: n > 1) (α : ℝ) : 
  ∃ l u, (l, u = sqrt ((n - 1) * (unbiased_variance xi (sample_mean xi)) / Quantile.chi2 (1 - α / 2) (n - 1)), 
                      sqrt ((n - 1) * (unbiased_variance xi (sample_mean xi)) / Quantile.chi2 (α / 2) (n - 1))) := 
begin
  sorry
end

end sample_mean_unbiased_unbiased_variance_unbiased_t_distribution_confidence_interval_μ_confidence_interval_σ_l509_509396


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l509_509214

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l509_509214


namespace find_ellipse_equation_l509_509957

def long_axis_three_times_short_axis : Prop :=
  ∃ (a b : ℝ), a > b ∧ (a / b = 3)

def passes_through_point_A : Prop :=
  ∃ (A : ℝ × ℝ), A = (3, 0)

def equilateral_triangle_condition : Prop :=
  ∃ (a b c : ℝ), a = 2 * c ∧ a - c = sqrt 3

def ellipse_equation_x_axis : Prop :=
  ∀ (x y : ℝ), x^2 / 9 + y^2 / 12 = 1

def ellipse_equation_y_axis : Prop :=
  ∀ (x y : ℝ), y^2 / 12 + x^2 / 9 = 1

theorem find_ellipse_equation 
  (h1 : long_axis_three_times_short_axis) 
  (h2 : passes_through_point_A) 
  (h3 : equilateral_triangle_condition) :
  ellipse_equation_x_axis ∨ ellipse_equation_y_axis :=
sorry

end find_ellipse_equation_l509_509957


namespace hyperbola_area_eccentricity_l509_509981

theorem hyperbola_area_eccentricity (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b)
    (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
    (e : ℝ) (he : e = Real.sqrt 5 / 2)
    (O : ℝ × ℝ) (F : ℝ × ℝ)
    (h1 : F.1 = a ∧ O = (0, 0) ∧ ∃A, A ∈ line_through O F ∧ 
         ∃O, ∃ A, circle_diameter (euclidean_distance O F) ∩ asymptote_intersect = {O, A})
    (h_area : ∃ A, ∃ F, 1/2 * a * b = 1) :
    a = 2 :=
begin
  sorry
end

end hyperbola_area_eccentricity_l509_509981


namespace find_number_l509_509095

theorem find_number (x : ℝ) (h : 0.5 * x = 0.1667 * x + 10) : x = 30 :=
by {
  sorry
}

end find_number_l509_509095


namespace find_b_l509_509880

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 2 * x^2 - b * x + 3
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := 2 * x^2 + b * x + 3

theorem find_b : (b : ℝ) (h : ∀ x : ℝ, g x b = f (x + 6) b) → b = 12 :=
by
  intros b h
  sorry

end find_b_l509_509880


namespace find_a_plus_b_l509_509342

open Complex Polynomial

noncomputable def a : ℝ := -5
noncomputable def b : ℝ := 44

theorem find_a_plus_b 
  (ha : a = -5)
  (hb : b = 44)
  (h₁ : a + b = 39)
  (hroots : (2 + I * Real.sqrt 7) ∈ (RootSet (X^3 + C(a) * X + C(b)) ℂ))
  (hconjugate : (2 - I * Real.sqrt 7) ∈ (RootSet (X^3 + C(a) * X + C(b)) ℂ))
  (hthird : -4 ∈ (RootSet (X^3 + C(a) * X + C(b)) ℂ)) : a + b = 39 :=
by
  have ha : a = -5 := ha
  have hb : b = 44 := hb
  have h : a + b = 39 := h
  rw [ha, hb]
  exact h

end find_a_plus_b_l509_509342


namespace instantaneous_angular_velocity_at_1_6_l509_509273

noncomputable def angular_velocity (t : ℝ) : ℝ := (2 * Real.pi / 0.64) * t^2

theorem instantaneous_angular_velocity_at_1_6 : 
  deriv angular_velocity 1.6 = 10 * Real.pi := 
by
  sorry

end instantaneous_angular_velocity_at_1_6_l509_509273


namespace union_of_sets_l509_509276

open Set

theorem union_of_sets (A B : Set ℝ) (hA : A = {x | -2 < x ∧ x < 1}) (hB : B = {x | 0 < x ∧ x < 2}) :
  A ∪ B = {x | -2 < x ∧ x < 2} :=
sorry

end union_of_sets_l509_509276


namespace incorrect_statement_d_l509_509547

theorem incorrect_statement_d (A : Prop) (B : Prop) (C : Prop) (D : Prop):
  (A = ("Degree" and "radian" are two different units of measure for angles)) →
  (B = ($0^{\circ} = 0 \text{ rad}$)) →
  (C = (Whether an angle is measured in degrees or radians, it is independent of the length of the circle's radius)) →
  (D = (For the same angle, the larger the radius of the circle it belongs to, the larger the radian measure of the angle)) →
  ¬D :=
by
  intros _ _ _ _
  -- proof goes here
  sorry

end incorrect_statement_d_l509_509547


namespace differential_equation_solution_l509_509725

variable (f : ℝ → ℝ) (n : ℕ)
variable (hf : continuous f)

def differential_operator (n : ℕ) (D : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ :=
  λ x y, (x * D x (D x ... (D x y)...))

theorem differential_equation_solution (hD : ∀ x y, D x y = (deriv y)) :
  ∃ y, (∫ t in 1..x, (x - t)^n * (f t) / (n! * t^(n + 1))) =
       ((differential_operator n (λ x y, D x y)) x y) ∧
       (y 1 = 0 ∧ (deriv y 1 = 0) ∧ ... ∧ (iterated_deriv n y 1 = 0)) :=
sorry

end differential_equation_solution_l509_509725


namespace tangent_identity_ONE_l509_509096

theorem tangent_identity_ONE (A E B D C O: Point)
  (AD_tangent: IsTangent AD \( \odot O \) at A) 
  (DC_tangent: IsTangent DC \( \odot O \) at E) 
  (CB_tangent: IsTangent CB \( \odot O \) at B)
  (AB_passing_center: AB \passes through the center O): 
  AD * BC = OE^2 :=
sorry

end tangent_identity_ONE_l509_509096


namespace range_of_a_l509_509259

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * a * x^3 + x^2 + a * x + 1

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = f a y ∧ 
  ∀ z : ℝ, f a z ≤ (max (f a x) (f a y)) ∧ f a z ≥ (min (f a x) (f a y))) →
  a ∈ set.Ioo (-1 : ℝ) 0 ∪ set.Ioo 0 1 :=
sorry

end range_of_a_l509_509259


namespace determine_b_for_inverse_function_l509_509424

theorem determine_b_for_inverse_function (b : ℝ) :
  (∀ x, (2 - 3 * (1 / (2 * x + b))) / (3 * (1 / (2 * x + b))) = x) ↔ b = 3 / 2 := by
  sorry

end determine_b_for_inverse_function_l509_509424


namespace find_a_l509_509973

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 * Real.exp x

theorem find_a (a : ℝ) : (∀ x : ℝ, -1 < x ∧ x < 1 → (x - a) * (x - a + 2) ≤ 0) → a = 1 :=
by
  intro h
  sorry 

end find_a_l509_509973


namespace f1_not_Γ_function_f2_is_Γ_function_f3_Γ_function_conditions_range_of_unknown_function_l509_509200

def is_Γ_function (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x : ℝ, f (a + x) * f (a - x) = b

/-- Function f1(x) = x -/
def f1 (x : ℝ) : ℝ := x

/-- Function f2(x) = 3^x -/
def f2 (x : ℝ) : ℝ := 3 ^ x

/-- Function f3(x) = tan x -/
def f3 (x : ℝ) : ℝ := Mathlib.tan x

theorem f1_not_Γ_function : ¬ ∃ a b : ℝ, is_Γ_function f1 a b := sorry

theorem f2_is_Γ_function : ∃ a : ℝ, ∃ b : ℝ, is_Γ_function f2 a b := by
  use 0
  use 1
  sorry

theorem f3_Γ_function_conditions :
  (∀ x : ℝ, ∃ k : ℤ, ∀ x ≠ k * π + π / 2, ∃ a b : ℝ, is_Γ_function f3 a b ↔
  (a = k * π + π / 4 ∨ a = k * π - π / 4) ∧ b = 1) := sorry

-- Third problem conditions
def unknown_function (x : ℝ) : ℝ := if x = 0 then 1 else if x = 1 then 2 else 4 / x

theorem range_of_unknown_function :
  (∀ x ∈ [-2016, 2016], unknown_function x ∈ set.Icc (2^(-2016)) (2^2016)) :=
sorry

end f1_not_Γ_function_f2_is_Γ_function_f3_Γ_function_conditions_range_of_unknown_function_l509_509200


namespace razorback_tshirt_revenue_l509_509767

theorem razorback_tshirt_revenue :
  ∀ (total_tshirts arkansas_tshirts: ℕ) (texas_tech_revenue: ℝ) (x: ℝ),
    total_tshirts = 186 →
    arkansas_tshirts = 172 →
    texas_tech_revenue = 1092 →
    total_tshirts = arkansas_tshirts + (1092 / x) →
    x = 78 :=
by {
  intros total_tshirts arkansas_tshirts texas_tech_revenue x,
  assume h1 : total_tshirts = 186,
  assume h2 : arkansas_tshirts = 172,
  assume h3 : texas_tech_revenue = 1092,
  assume h4 : total_tshirts = arkansas_tshirts + (1092 / x),
  sorry
}

end razorback_tshirt_revenue_l509_509767


namespace student_arrangements_l509_509854

theorem student_arrangements  :
  let num_students : ℕ := 6 in
  let venues : Finset ℕ := {1, 2, 3} in
  let venueA_students : ℕ := 1 in
  let venueB_students : ℕ := 2 in
  let venueC_students : ℕ := 3 in
  num_students = venueA_students + venueB_students + venueC_students →
  nat.choose num_students venueA_students *
  nat.choose (num_students - venueA_students) venueB_students *
  nat.choose (num_students - venueA_students - venueB_students) venueC_students = 60 :=
by
  intros num_students venues venueA_students venueB_students venueC_students h_sum_eq
  sorry

end student_arrangements_l509_509854


namespace length_of_ED_l509_509326

-- Definitions for points and segments
variables {A B C E D L : Type}
variables {ED AC : Segment}
variables {AE AC_length : ℝ} (AE_value : AE = 15) (AC_length_value : AC_length = 12)

-- Definitions for triangle, angle bisector, and parallel conditions
variables (triangle_ABC : Triangle A B C)
variables (angle_bisector_AL : AngleBisector A L)
variables (DL_eq_LC : SegmentLength D L = SegmentLength L C)
variables (ED_parallel_AC : Parallel ED AC)

-- Main theorem statement
theorem length_of_ED (ED : Segment) (ac_12 : SegmentLength AC = 12) (ae_15 : SegmentLength AE = 15) :
  SegmentLength ED = 3 :=
sorry

end length_of_ED_l509_509326


namespace allocate_students_l509_509118

theorem allocate_students (classes students : ℕ) (h_class : classes = 6) (h_students : students = 8) 
  (h_at_least_one : ∀ (c : ℕ), 1 ≤ c → c ≤ classes → students ≥ classes): 
  (∃ allocation : ℕ → ℕ, 
    (∀ c, 1 ≤ c ∧ c ≤ classes → allocation c ≥ 1) ∧ 
    (∑ c in finset.range classes, allocation c = students) ∧ 
    (nat.choose classes 2 + classes = 21)) :=
sorry

end allocate_students_l509_509118


namespace find_a_b_c_sum_l509_509761

theorem find_a_b_c_sum (a b c : ℤ)
  (h_gcd : gcd (x ^ 2 + a * x + b) (x ^ 2 + b * x + c) = x + 1)
  (h_lcm : lcm (x ^ 2 + a * x + b) (x ^ 2 + b * x + c) = x ^ 3 - 4 * x ^ 2 + x + 6) :
  a + b + c = -6 := 
sorry

end find_a_b_c_sum_l509_509761


namespace sum_of_valid_n_l509_509114

def is_splitty (f : Polynomial ℤ) : Prop :=
  ∀ (p : ℕ) [Fact (Nat.Prime p)], ∃ (g_p h_p : Polynomial ℤ),
    g_p.degree < f.degree ∧ h_p.degree < f.degree ∧ (∀ n, ((f - g_p * h_p).coeff n) % p = 0)

def polynomial_example (n : ℕ) : Polynomial ℤ :=
  Polynomial.C n + Polynomial.X ^ 4 + Polynomial.C 16 * Polynomial.X ^ 2

def valid_n (n : ℕ) : Prop :=
  is_splitty (polynomial_example n)

theorem sum_of_valid_n :
  (Finset.range 101).filter valid_n |> Finset.sum id = 693 :=
by
  sorry

end sum_of_valid_n_l509_509114


namespace range_of_a_l509_509859

noncomputable def f : ℝ → ℝ := sorry

axiom f_pos_of_pos (x : ℝ) (h : 0 < x) : 0 < f x
axiom f_1_eq_2 : f 1 = 2
axiom f_add (m n : ℝ) : f (m + n) = f m + f n

def A := { p : ℝ × ℝ | f (3 * (p.1)^2) + f (4 * (p.2)^2) ≤ 24 }
def B (a : ℝ) := { p : ℝ × ℝ | f (p.1) - f (a * p.2) + f 3 = 0 }
def C (a : ℝ) := { p : ℝ × ℝ | f (p.1) = (1 / 2) * f ((p.2)^2) + f a }

theorem range_of_a (a : ℝ) (hAB : (A ∩ B a).nonempty) (hAC : (A ∩ C a).nonempty) :
  a ≤ 2 :=
sorry

end range_of_a_l509_509859


namespace peaches_eaten_l509_509462

theorem peaches_eaten (P B Baskets P_each R Boxes P_box : ℕ) 
  (h1 : B = 5) 
  (h2 : P_each = 25)
  (h3 : Baskets = B * P_each)
  (h4 : R = 8) 
  (h5 : P_box = 15)
  (h6 : Boxes = R * P_box)
  (h7 : P = Baskets - Boxes) : P = 5 :=
by sorry

end peaches_eaten_l509_509462


namespace spider_socks_shoes_orders_l509_509123

theorem spider_socks_shoes_orders :
  let actions := 16
  let socks := 8
  let shoes := 8
  ∑ sock_shoe_comb : (Σ (a : Fin socks), Fin shoes) in
  ({u | ∀ i, (u i).fst < (u i).snd}.finite.toFinset).card = 
  81729648000 :=
sorry

end spider_socks_shoes_orders_l509_509123


namespace sum_of_roots_is_neg_nine_l509_509568

-- Define the quadratic equation
def quadratic_eq (x : ℤ) := 81 - 27 * x - 3 * x^2

-- Theorem: Sum of the solutions to the quadratic equation 81 - 27x - 3x^2 = 0 is -9
theorem sum_of_roots_is_neg_nine (r s : ℝ) (h : quadratic_eq r = 0 ∧ quadratic_eq s = 0) :
  r + s = -9 :=
by
  sorry

end sum_of_roots_is_neg_nine_l509_509568


namespace negation_of_p_l509_509993

-- Define the proposition p
def p : Prop := ∀ x : ℝ, sin x ≤ 1

-- State the goal: the negation of p
theorem negation_of_p : ¬ p ↔ ∃ x : ℝ, sin x > 1 := by
  sorry

end negation_of_p_l509_509993


namespace total_money_l509_509886

-- Define constants for amounts
variables (A B C : ℕ)

-- Define conditions
def condition1 : Prop := A + C = 200
def condition2 : Prop := B + C = 340
def condition3 : Prop := C = 40

-- Prove the total amount of A, B, and C.
theorem total_money (h1 : condition1) (h2 : condition2) (h3 : condition3) : A + B + C = 500 :=
by sorry

end total_money_l509_509886


namespace set_intersection_l509_509299

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def complement (S : Set ℝ) := {x | x ∉ S}

theorem set_intersection :
  (U = univ) →
  (A = {x : ℝ | x + 1 < 0}) →
  (B = {x : ℝ | x - 3 < 0}) →
  (inter (complement A) B = {x : ℝ | -1 ≤ x ∧ x < 3}) :=
by
  intros hU hA hB
  sorry

end set_intersection_l509_509299


namespace fifth_powers_sum_eq_l509_509218

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l509_509218


namespace tangent_line_tangency_l509_509264

-- Definition of the derivatives of the functions
def derivative_f := fun x => 2 * x
def derivative_g := fun x => 1 / x

-- Tangent line conditions
def tangent_line_f (x0 : ℝ) := derivative_f x0 * (x - x0) + x0^2
def tangent_line_g (m : ℝ) := derivative_g m * (x - m) + Real.log m

-- Prove the conditions for x₀
theorem tangent_line_tangency (x0 : ℝ) : 
  (0 < x0) ∧ (x0 < 1) →
  (exists m, 0 < m ∧ m < 1 ∧ 2 * x0 = 1 / m ∧ Real.log m - 1 = -x0^2) →
  x0 ∈ (sqrt 2, sqrt 3) :=
begin
  -- Proof goes here
  sorry
end

end tangent_line_tangency_l509_509264


namespace auction_sale_l509_509906

theorem auction_sale (TV_initial_price : ℝ) (TV_increase_fraction : ℝ) (Phone_initial_price : ℝ) (Phone_increase_percent : ℝ) :
  TV_initial_price = 500 → 
  TV_increase_fraction = 2 / 5 → 
  Phone_initial_price = 400 →
  Phone_increase_percent = 40 →
  let TV_final_price := TV_initial_price + TV_increase_fraction * TV_initial_price in
  let Phone_final_price := Phone_initial_price + (Phone_increase_percent / 100) * Phone_initial_price in
  TV_final_price + Phone_final_price = 1260 := by
  sorry

end auction_sale_l509_509906


namespace sum_of_digits_of_m_l509_509461

-- Define the logarithms and intermediate expressions
noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem sum_of_digits_of_m :
  ∃ m : ℕ, log_b 3 (log_b 81 m) = log_b 9 (log_b 9 m) ∧ sum_of_digits m = 10 := 
by
  sorry

end sum_of_digits_of_m_l509_509461


namespace arithmetic_sequence_sum_l509_509446

theorem arithmetic_sequence_sum (c d e : ℕ) (h1 : 10 - 3 = 7) (h2 : 17 - 10 = 7) (h3 : c - 17 = 7) (h4 : d - c = 7) (h5 : e - d = 7) : 
  c + d + e = 93 :=
sorry

end arithmetic_sequence_sum_l509_509446


namespace cassidy_number_of_posters_l509_509141

/-- Cassidy's current number of posters -/
def current_posters (C : ℕ) : Prop := 
  C + 6 = 2 * 14

theorem cassidy_number_of_posters : ∃ C : ℕ, current_posters C := 
  Exists.intro 22 sorry

end cassidy_number_of_posters_l509_509141


namespace triangle_ABC_area_l509_509323

-- Definitions of the points and the relationship between them
variables (A B C D E : Type*)
variables [HasZero D] [HasZero E] -- To represent points on a coordinate plane possibly
variables (area : A → B → C → E) -- A function representing area of triangle with vertices

-- Conditions
def condition_BD_DC := true -- Placeholder, as exact definitions of the ratio are complex
def condition_AE := true -- Placeholder, as exact intersection definitions are complex
def condition_area_ABE := area A B E = 30

-- Main theorem to prove
theorem triangle_ABC_area :
  condition_BD_DC → condition_AE → condition_area_ABE → area A B C = 90 :=
by
  intros h1 h2 h3
  sorry

end triangle_ABC_area_l509_509323


namespace find_fx_at_3_l509_509194

theorem find_fx_at_3 (a b c : ℝ) (f g : ℝ → ℝ) (h1 : ∀ x, f x = x^7 - a * x^5 + b * x^3 + c * x + 2)
  (h2 : f (-3) = -3) : f 3 = 7 :=
by
  let g (x : ℝ) := x^7 - a * x^5 + b * x^3 + c * x
  have h3 : ∀ x, f x = g x + 2 := by
    intro x
    rw [h1]
    exact rfl    
  have h4 : g (-3) = -5 := by
    rw [← h3 (-3), h2]
    exact rfl    
  have h5 : g (3) = 5 := by
    exact h4 ▸ odd.neg_eq_iff_eq_neg.mpr rfl
  calc
    f 3 = g 3 + 2 := h3 3
       ... = 7     := by rw [h5]; ring

end find_fx_at_3_l509_509194


namespace tens_digit_of_8_pow_1234_l509_509816

theorem tens_digit_of_8_pow_1234 :
  (8^1234 / 10) % 10 = 0 :=
sorry

end tens_digit_of_8_pow_1234_l509_509816


namespace train_speed_l509_509882

theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : length = 55) 
    (h2 : time = 5.5) 
    (h3 : speed = (length / time) * (3600 / 1000)) : 
    speed = 36 :=
sorry

end train_speed_l509_509882


namespace sin_2_angle_CPD_correct_l509_509390

noncomputable def sin_2_angle_CPD (cos_alpha cos_beta : ℝ) (h_cos_alpha : cos_alpha = 3/5) (h_cos_beta : cos_beta = 2/5) : ℝ :=
  let alpha := real.arccos cos_alpha
  let beta := real.arccos cos_beta
  let gamma := alpha + beta
  2 * real.sin gamma * real.cos gamma

theorem sin_2_angle_CPD_correct : 
  (sin_2_angle_CPD (3/5) (2/5) (by norm_num) (by norm_num)) = (64 + 24 * real.sqrt 21) / 125 :=
by
  sorry

end sin_2_angle_CPD_correct_l509_509390


namespace segment_length_between_l509_509603

variable (A B : Type) [MetricSpace A] [MetricSpace B]
variable (a : A → A → Prop) (b : A → A → Prop) -- representing lines a and b
variable (A1 A2 A3 B1 B2 B3 : A)
variable (between : A → A → A → Prop) -- representing the "between" relationship on line a
variable (perpendicular : A → A → B) -- representing perpendicular drops from points A_i to line b

axiom points_on_line_a : a A1 A2 ∧ a A2 A3
axiom A2_between_A1_A3 : between A1 A2 A3
axiom perp_drops : (perpendicular A1 B1) ∧ (perpendicular A2 B2) ∧ (perpendicular A3 B3)

theorem segment_length_between :
  ∀ (dist : A → A → ℝ), dist A2 B2 ∈ Set.Icc (min (dist A1 B1) (dist A3 B3)) (max (dist A1 B1) (dist A3 B3)) :=
by
  sorry

end segment_length_between_l509_509603


namespace Jake_width_proof_l509_509013

-- Define the dimensions of Sara's birdhouse in feet
def Sara_width_feet := 1
def Sara_height_feet := 2
def Sara_depth_feet := 2

-- Convert the dimensions to inches
def Sara_width_inch := Sara_width_feet * 12
def Sara_height_inch := Sara_height_feet * 12
def Sara_depth_inch := Sara_depth_feet * 12

-- Calculate Sara's birdhouse volume
def Sara_volume := Sara_width_inch * Sara_height_inch * Sara_depth_inch

-- Define the dimensions of Jake's birdhouse in inches
def Jake_height_inch := 20
def Jake_depth_inch := 18
def Jake_volume (Jake_width_inch : ℝ) := Jake_width_inch * Jake_height_inch * Jake_depth_inch

-- Difference in volume
def volume_difference := 1152

-- Prove the width of Jake's birdhouse
theorem Jake_width_proof : ∃ (W : ℝ), Jake_volume W - Sara_volume = volume_difference ∧ W = 22.4 := by
  sorry

end Jake_width_proof_l509_509013


namespace min_value_f_prime_at_2_l509_509263

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2 * a * x^2 + (1/a) * x

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 4*a*x + (1/a)

theorem min_value_f_prime_at_2 (a : ℝ) (h : a > 0) : 
  f_prime a 2 >= 12 + 4 * Real.sqrt 2 := 
by
  -- proof will be written here
  sorry

end min_value_f_prime_at_2_l509_509263


namespace students_in_both_clubs_l509_509132

theorem students_in_both_clubs (total_students : ℕ) (drama_club : ℕ) (science_club : ℕ) (either_or_both_clubs : ℕ) : 
  total_students = 400 → 
  drama_club = 180 → 
  science_club = 230 → 
  either_or_both_clubs = 350 → 
  (drama_club + science_club - either_or_both_clubs) = 60 :=
by
  intros h_total h_drama h_science h_either_both
  rw [h_drama, h_science, h_either_both]
  sorry -- Proof omitted

end students_in_both_clubs_l509_509132


namespace problem_l509_509291

variable {a b c d e : ℝ}

def condition1 : Prop := (a / b = 5)
def condition2 : Prop := (b / c = 1 / 4)
def condition3 : Prop := (c / d = 7)
def condition4 : Prop := (d / e = 1 / 2)

theorem problem : condition1 → condition2 → condition3 → condition4 → (e / a = 8 / 35) :=
by
  intros
  sorry

end problem_l509_509291


namespace find_interest_rate_l509_509681

-- Definitions for principal amounts, time periods, and given interest rates
def P1 : ℝ := 600
def R1 : ℝ := 10
def T1 : ℕ := 4

def P2 : ℝ := 100
def T2 : ℕ := 48

-- Definition for the interest calculation formula
def interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- Given the interest amounts are equal
theorem find_interest_rate (R2 : ℝ) (H : interest P1 R1 T1 = interest P2 R2 T2) : R2 = 5 :=
by
  sorry

end find_interest_rate_l509_509681


namespace auction_sale_l509_509907

theorem auction_sale (TV_initial_price : ℝ) (TV_increase_fraction : ℝ) (Phone_initial_price : ℝ) (Phone_increase_percent : ℝ) :
  TV_initial_price = 500 → 
  TV_increase_fraction = 2 / 5 → 
  Phone_initial_price = 400 →
  Phone_increase_percent = 40 →
  let TV_final_price := TV_initial_price + TV_increase_fraction * TV_initial_price in
  let Phone_final_price := Phone_initial_price + (Phone_increase_percent / 100) * Phone_initial_price in
  TV_final_price + Phone_final_price = 1260 := by
  sorry

end auction_sale_l509_509907


namespace partition_theorem_l509_509165

noncomputable def existence_of_partitions (a : ℝ) (ha : 0 < a ∧ a < 2) : Prop :=
  ∃ (n : ℕ) (A : ℕ → Set ℕ), (∀ i, Set.Infinite (A i)) ∧
  (∀ i j, i ≠ j → Disjoint (A i) (A j)) ∧
  (⋃ i, A i) = Set.Univ ∧
  (∀ i b c, b > c → b ∈ A i → c ∈ A i → b - c ≥ a^i)

theorem partition_theorem (a : ℝ) : (0 < a ∧ a < 2) ↔ existence_of_partitions a (and.intro ((by apply sorry) : 0 < a) ((by apply sorry) : a < 2)) := 
sorry

end partition_theorem_l509_509165


namespace geometric_sequence_preserving_functions_l509_509858

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n * a (n + 2) = (a (n + 1))^2

def preserves_geometric_sequence (f : ℝ → ℝ) :=
∀ (a : ℕ → ℝ), geometric_sequence a → geometric_sequence (λ n, f (a n))

def f1 (x : ℝ) : ℝ := x^2

def f2 (x : ℝ) : ℝ := 2^x

def f3 (x : ℝ) : ℝ := sqrt (abs x)

def f4 (x : ℝ) : ℝ := log (abs x)

theorem geometric_sequence_preserving_functions :
  preserves_geometric_sequence f1 ∧ preserves_geometric_sequence f3 ∧ ¬preserves_geometric_sequence f2 ∧ ¬preserves_geometric_sequence f4 := 
by
  sorry

end geometric_sequence_preserving_functions_l509_509858


namespace prop1_prop2_prop3_l509_509157

-- Proposition p: ∃ x ∈ ℝ, tan x = 1
def p : Prop := ∃ x : ℝ, Real.tan x = 1

-- Proposition q: ∀ x ∈ ℝ, x² + 1 > 0
def q : Prop := ∀ x : ℝ, x^2 + 1 > 0

-- Proposition that p ∧ ¬q is false
theorem prop1 : ¬ (p ∧ ¬q) :=
by {
    sorry
}

-- Lines l1 and l2 and the condition for perpendicularity
variable {a b : ℝ}
def l1 (x y : ℝ) : Prop := a * x + 3 * y - 1 = 0
def l2 (x y : ℝ) : Prop := x + b * y + 1 = 0

-- Necessary and sufficient condition for l1 ⊥ l2 is not always a / b = -3
theorem prop2 : ¬ (l1 x y ∧ l2 x y → (a / b = -3)) :=
by {
    sorry
}

-- Original and converse propositions
def original : Prop := ∀ x : ℝ, (x^2 - 3*x + 2 = 0) → (x = 1)
def converse : Prop := ∀ x : ℝ, (x ≠ 1) → (x^2 - 3*x + 2 ≠ 0)

-- The converse proposition is correct
theorem prop3 : converse :=
by {
    sorry
}

end prop1_prop2_prop3_l509_509157


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l509_509226

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l509_509226


namespace hyperbola_asymptote_slope_proof_l509_509787

noncomputable def hyperbola_asymptote_slope : ℝ :=
  let foci_distance := Real.sqrt ((8 - 2)^2 + (3 - 3)^2)
  let c := foci_distance / 2
  let a := 2  -- Given that 2a = 4
  let b := Real.sqrt (c^2 - a^2)
  b / a

theorem hyperbola_asymptote_slope_proof :
  ∀ x y : ℝ, 
  (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) →
  hyperbola_asymptote_slope = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_asymptote_slope_proof_l509_509787


namespace sum_inequality_l509_509727

theorem sum_inequality (n : ℕ) (x : Fin n → ℝ) 
  (h0 : x 0 = 0)
  (h_pos : ∀ i : Fin n, 0 < x i)
  (h_sum : ∑ i : Fin n, x i = 1) :
  1 ≤ ∑ i : Fin n, x i / 
  (Real.sqrt (1 + ∑ j in Finset.range i, x j) * Real.sqrt (∑ j in Finset.Ico (i + 1) n, x j)) ∧
  ∑ i : Fin n, x i / 
  (Real.sqrt (1 + ∑ j in Finset.range i, x j) * Real.sqrt (∑ j in Finset.Ico (i + 1) n, x j)) < Real.pi / 2 :=
  sorry

end sum_inequality_l509_509727


namespace sum_rel_prime_l509_509760

noncomputable def probability_reaching (n : ℕ) (start finish : ℕ × ℕ) : ℚ :=
  let possible_steps := 4^n in
  let valid_steps := (Nat.choose 8 3) * (Nat.choose 5 1) * (Nat.choose 4 2) in
  (valid_steps : ℚ) / (possible_steps : ℚ)

theorem sum_rel_prime (m n : ℕ) (hmn : Nat.rel_prime m n)
  (h : probability_reaching 8 (0, 0) (3, 1) = m / n) : m + n = 4201 :=
by
  sorry

end sum_rel_prime_l509_509760


namespace power_function_passing_point_l509_509297

theorem power_function_passing_point (f : ℝ → ℝ) (n : ℝ) :
  (∀ x : ℝ, f x = x ^ n) →
  f 2 = real.sqrt 2 →
  f 4 = 2 :=
by
  intros h1 h2
  sorry

end power_function_passing_point_l509_509297


namespace q_work_alone_in_10_days_l509_509086

theorem q_work_alone_in_10_days (p_rate : ℝ) (q_rate : ℝ) (d : ℕ) (h1 : p_rate = 1 / 20)
                                    (h2 : q_rate = 1 / d) (h3 : 2 * (p_rate + q_rate) = 0.3) :
                                    d = 10 :=
by sorry

end q_work_alone_in_10_days_l509_509086


namespace zoo_spending_l509_509466

def total_cost (goat_price llama_multiplier : ℕ) (num_goats num_llamas : ℕ) : ℕ :=
  let goat_cost := num_goats * goat_price
  let llama_price := llama_multiplier * goat_price
  let llama_cost := num_llamas * llama_price
  goat_cost + llama_cost

theorem zoo_spending :
  total_cost 400 3 2 == 4800 :=
by 
  let goat_price := 400
  let num_goats := 3
  let num_llamas := 2 * num_goats
  let llama_multiplier := 3
  let total := total_cost goat_price llama_multiplier num_goats num_llamas
  have h_goat_cost : total_cost goat_price llama_multiplier num_goats num_llamas = 4800 := sorry
  exact h_goat_cost

end zoo_spending_l509_509466


namespace cubic_identity_l509_509665

theorem cubic_identity (x : ℝ) (h : x + 1/x = -6) : x^3 + 1/x^3 = -198 := 
by
  sorry

end cubic_identity_l509_509665


namespace angle_greater_than_90_for_n_pairs_l509_509090

open real

-- Definition of points in n-dimensional Euclidean space
def Point (n : ℕ) := EuclideanSpace ℝ (fin n)

-- Given conditions
variables (n : ℕ) [fact (2 ≤ n)]
variables (A : fin (n + 1) → Point n)
variable  (B : Point n)
variables (hA_not_on_same_hyperplane : ¬ affineDep (A 0, A 1, A 2, …, A i, A j))
variables (hB_inside_convex_hull :
  ∃ (w : fin (n + 1) → ℝ), (∀ i, 0 < w i ∧ w i < 1) ∧
                            (∑ i, w i = 1) ∧
                            (∑ i, w i • A i = B))

-- The theorem to be proven
theorem angle_greater_than_90_for_n_pairs :
  ∃ (pairs : finset (fin (n + 1) × fin (n + 1))),
  pairs.card ≥ n ∧
  ∀ (i j : fin (n + 1)), (i, j) ∈ pairs → i < j ∧ inner (A i) (A j) < 0 := 
sorry

end angle_greater_than_90_for_n_pairs_l509_509090


namespace relationship_among_three_numbers_l509_509796

theorem relationship_among_three_numbers :
  let a := -0.3
  let b := 0.3^2
  let c := 2^0.3
  b < a ∧ a < c :=
by
  let a := -0.3
  let b := 0.3^2
  let c := 2^0.3
  sorry

end relationship_among_three_numbers_l509_509796


namespace max_separation_l509_509807

theorem max_separation (R r : ℝ) : 
(radius_large : R = 24) → 
(radius_small : r = 15) → 
(far_laps : R * 2 * π * (5.0 / 2.0) = r * 2 * π * 4) :=

    let C_large := 2 * π * R
    let C_small := 2 * π * r
    let max_dist := 48
    ∃ n₁ n₂, lcm (R * π) (r * 2 * π) = n₁ * C_large ∧ 
                             = n₂ * C_small ∧ 
                             n₂ = 4 :=
begin
    intro hR,
    intro hr,
    sorry
end

end max_separation_l509_509807


namespace product_of_1101_2_and_202_3_is_260_l509_509936

   /-- Convert a binary string to its decimal value -/
   def binary_to_decimal (b : String) : ℕ :=
     b.foldl (λ acc bit, acc * 2 + bit.toNat - '0'.toNat) 0

   /-- Convert a ternary string to its decimal value -/
   def ternary_to_decimal (t : String) : ℕ :=
     t.foldl (λ acc bit, acc * 3 + bit.toNat - '0'.toNat) 0

   theorem product_of_1101_2_and_202_3_is_260 :
     binary_to_decimal "1101" * ternary_to_decimal "202" = 260 :=
   by
     calc
       binary_to_decimal "1101" = 13 : by rfl
       ternary_to_decimal "202" = 20 : by rfl
       13 * 20 = 260 : by rfl
   
end product_of_1101_2_and_202_3_is_260_l509_509936


namespace other_investment_interest_rate_l509_509335

open Real

-- Definitions of the given conditions
def total_investment : ℝ := 22000
def investment_at_8_percent : ℝ := 17000
def total_interest : ℝ := 1710
def interest_rate_8_percent : ℝ := 0.08

-- Derived definitions from the conditions
def other_investment_amount : ℝ := total_investment - investment_at_8_percent
def interest_from_8_percent : ℝ := investment_at_8_percent * interest_rate_8_percent
def interest_from_other : ℝ := total_interest - interest_from_8_percent

-- Proof problem: Prove that the percentage of the other investment is 0.07 (or 7%).
theorem other_investment_interest_rate :
  interest_from_other / other_investment_amount = 0.07 := by
  sorry

end other_investment_interest_rate_l509_509335


namespace horner_evaluation_l509_509071

-- Define the polynomial function
def f (x : ℝ) : ℝ := 4 * x^4 + 3 * x^3 - 6 * x^2 + x - 1

-- The theorem that we need to prove
theorem horner_evaluation : f (-1) = -5 :=
  by
  -- This is the statement without the proof steps
  sorry

end horner_evaluation_l509_509071


namespace simplify_rationalize_l509_509406

theorem simplify_rationalize : (1 : ℝ) / (1 + (1 / (real.sqrt 5 + 2))) = (real.sqrt 5 + 1) / 4 :=
by sorry

end simplify_rationalize_l509_509406


namespace area_of_rectangular_field_l509_509385

noncomputable def calculate_area_of_rectangle (w d : ℕ) : ℕ :=
if h : d^2 > w^2 then
  let l := (d^2 - w^2).sqrt in
  l * w
else
  0

theorem area_of_rectangular_field :
  calculate_area_of_rectangle 15 17 = 120 := by
  sorry

end area_of_rectangular_field_l509_509385


namespace product_of_ninth_and_tenth_l509_509682

def scores_first_8 := [7, 4, 3, 6, 8, 3, 1, 5]
def total_points_first_8 := scores_first_8.sum

def condition1 (ninth_game_points tenth_game_points : ℕ) : Prop :=
  ninth_game_points < 10 ∧ tenth_game_points < 10

def condition2 (ninth_game_points : ℕ) : Prop :=
  (total_points_first_8 + ninth_game_points) % 9 = 0

def condition3 (ninth_game_points tenth_game_points : ℕ) : Prop :=
  (total_points_first_8 + ninth_game_points + tenth_game_points) % 10 = 0

theorem product_of_ninth_and_tenth (ninth_game_points : ℕ) (tenth_game_points : ℕ) 
  (h1 : condition1 ninth_game_points tenth_game_points)
  (h2 : condition2 ninth_game_points)
  (h3 : condition3 ninth_game_points tenth_game_points) : 
  ninth_game_points * tenth_game_points = 40 :=
sorry

end product_of_ninth_and_tenth_l509_509682


namespace geometric_sequence_properties_l509_509774

noncomputable theory

variables (a : ℕ → ℝ) (q : ℝ) (T : ℕ → ℝ)

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def is_product_of_terms (T : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n : ℕ, T n = ∏ i in finset.range n, a i

variables (h0 : is_geometric_sequence a q)
          (h1 : is_product_of_terms T a)
          (h2 : a 1 > 1)
          (h3 : a 99 * a 100 - 1 > 0)
          (h4 : (a 99 - 1) / (a 100 - 1) < 0)

theorem geometric_sequence_properties :
  (0 < q ∧ q < 1) ∧
  (a 99 * a 101 - 1 < 0) ∧
  ¬(∀ n, T 100 ≤ T n) ∧
  (∀ n, T n > 1 → n ≤ 198) :=
sorry

end geometric_sequence_properties_l509_509774


namespace simplify_expr_l509_509411

noncomputable def expr := 1 / (1 + 1 / (Real.sqrt 5 + 2))
noncomputable def simplified := (Real.sqrt 5 + 1) / 4

theorem simplify_expr : expr = simplified :=
by
  sorry

end simplify_expr_l509_509411


namespace min_dot_product_max_area_triangle_l509_509642

-- Definitions and conditions for problem (I)
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1
def F1 : ℝ × ℝ := (-Real.sqrt 6, 0)
def F2 : ℝ × ℝ := (Real.sqrt 6, 0)
def vec_PF1 (x0 y0 : ℝ) : ℝ × ℝ := (F1.fst - x0, F1.snd - y0)
def vec_PF2 (x0 y0 : ℝ) : ℝ × ℝ := (F2.fst - x0, F2.snd - y0)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.fst * v2.fst + v1.snd * v2.snd

-- Statement for problem (I)
theorem min_dot_product (x0 y0 : ℝ) (hP : ellipse x0 y0) :
  dot_product (vec_PF1 x0 y0) (vec_PF2 x0 y0) = -4 :=
sorry

-- Definitions and conditions for problem (II)
def line (m b x : ℝ) : ℝ := m * x + b
def slope_line : ℝ := 1/2
def P_condition (x0 y0 : ℝ) : Prop := 0 < x0 ∧ 0 < y0
def dot_product_condition (x0 y0 : ℝ) : Prop := dot_product (vec_PF1 x0 y0) (vec_PF2 x0 y0) = -1
def area_triangle (x y : ℝ) (b : ℝ) : ℝ := Real.sqrt (b^2 * (4 - b^2)) -- simplified condition for max area
def in_interval (b : ℝ) : Prop := -2 < b ∧ b < 2

-- Statement for problem (II)
theorem max_area_triangle (x0 y0 b : ℝ) 
    (h1 : ellipse x0 y0) (h2 : P_condition x0 y0) (h3 : dot_product_condition x0 y0) (h4 : in_interval b) :
  area_triangle x0 y0 b = 2 :=
sorry

end min_dot_product_max_area_triangle_l509_509642


namespace number_of_triangles_l509_509469

-- Definition of given conditions
def original_wire_length : ℝ := 84
def remaining_wire_length : ℝ := 12
def wire_per_triangle : ℝ := 3

-- The goal is to prove that the number of triangles that can be made is 24
theorem number_of_triangles : (original_wire_length - remaining_wire_length) / wire_per_triangle = 24 := by
  sorry

end number_of_triangles_l509_509469


namespace number_of_right_triangles_with_hypotenuse_is_12_l509_509285

theorem number_of_right_triangles_with_hypotenuse_is_12 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b : ℕ), 
     (b < 150) →
     (a^2 + b^2 = (b + 2)^2) →
     ∃ (k : ℕ), a = 2 * k ∧ k^2 = b + 1) := 
  sorry

end number_of_right_triangles_with_hypotenuse_is_12_l509_509285


namespace scrap_cookie_radius_is_correct_l509_509738

noncomputable def radius_of_scrap_cookie (large_radius small_radius : ℝ) (number_of_cookies : ℕ) : ℝ :=
  have large_area : ℝ := Real.pi * large_radius^2
  have small_area : ℝ := Real.pi * small_radius^2
  have total_small_area : ℝ := small_area * number_of_cookies
  have scrap_area : ℝ := large_area - total_small_area
  Real.sqrt (scrap_area / Real.pi)

theorem scrap_cookie_radius_is_correct :
  radius_of_scrap_cookie 8 2 9 = 2 * Real.sqrt 7 :=
sorry

end scrap_cookie_radius_is_correct_l509_509738


namespace road_length_kopatych_to_losyash_l509_509333

variable (T Krosh_dist Yozhik_dist : ℕ)
variable (d_k d_y r_k r_y : ℕ)

theorem road_length_kopatych_to_losyash : 
    (d_k = 20) → (d_y = 16) → (r_k = 30) → (r_y = 60) → 
    (Krosh_dist = 5 * T / 9) → (Yozhik_dist = 4 * T / 9) → 
    (T = Krosh_dist + r_k) →
    (T = Yozhik_dist + r_y) → 
    (T = 180) :=
by
  intros
  sorry

end road_length_kopatych_to_losyash_l509_509333


namespace order_of_abc_l509_509628

section
variables {a b c : ℝ}

def a_def : a = (1/2) * Real.log 2 := by sorry
def b_def : b = (1/4) * Real.log 16 := by sorry
def c_def : c = (1/6) * Real.log 27 := by sorry

theorem order_of_abc : a < c ∧ c < b :=
by
  have ha : a = (1/2) * Real.log 2 := by sorry
  have hb : b = (1/2) * Real.log 4 := by sorry
  have hc : c = (1/2) * Real.log 3 := by sorry
  sorry
end

end order_of_abc_l509_509628


namespace harmonic_inequality_l509_509350

theorem harmonic_inequality (a : ℕ → ℕ) (n : ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j) :
  (1 + (∑ i in finset.range (n + 1), 1 / (i + 1 : ℝ))) ≤ (a 1 + (∑ i in finset.range (n + 1), a (i + 1) / ((i + 1) ^ 2 : ℝ))) :=
sorry

end harmonic_inequality_l509_509350


namespace total_toes_on_bus_l509_509374

/-- Definition for the number of toes a Hoopit has -/
def toes_per_hoopit : ℕ := 4 * 3

/-- Definition for the number of toes a Neglart has -/
def toes_per_neglart : ℕ := 5 * 2

/-- Definition for the total number of Hoopits on the bus -/
def hoopit_students_on_bus : ℕ := 7

/-- Definition for the total number of Neglarts on the bus -/
def neglart_students_on_bus : ℕ := 8

/-- Proving that the total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus : hoopit_students_on_bus * toes_per_hoopit + neglart_students_on_bus * toes_per_neglart = 164 := by
  sorry

end total_toes_on_bus_l509_509374


namespace batsman_average_increase_l509_509097

noncomputable def increase_in_average (A : ℚ) (total_runs_after_11_innings : ℕ := 11) (runs_scored_in_12th : ℕ := 80) (new_average : ℚ := 58) : ℚ :=
  new_average - (total_runs_after_11_innings * A / 11)

theorem batsman_average_increase :
  (∃ A : ℚ, 11 * A + 80 = 12 * 58) →
  increase_in_average (some (classical.some_spec (exists.intro A sorry))) = 2 := 
by
  sorry

end batsman_average_increase_l509_509097


namespace smallest_four_consecutive_numbers_l509_509050

theorem smallest_four_consecutive_numbers (n : ℕ) 
  (h : n * (n + 1) * (n + 2) * (n + 3) = 4574880) : n = 43 :=
sorry

end smallest_four_consecutive_numbers_l509_509050


namespace length_of_intervals_l509_509167

theorem length_of_intervals : 
  (2 - Real.pi / 4) = (finset.Icc 0 (Real.pi / 4)).measure -- the length of interval where max(sqrt(x/2), tan x) <= 1
:= sorry

end length_of_intervals_l509_509167


namespace hyperbola_eccentricity_range_l509_509992

variable (b a c : ℝ) (P F1 F2 O : Point) (e : ℝ)
hypothesis hb : b > 0
hypothesis O_origin : O = Point.origin
hypothesis P_on_hyperbola : on_hyperbola P (Hyperbola c b)
hypothesis Foci : foci F1 F2 (Hyperbola c b)
hypothesis Distance_F1F2_OP : |distance F1 F2| = 2 * |distance O P|
hypothesis Angle_PF2F1 : tan (∠ P F2 F1) ≥ 4

theorem hyperbola_eccentricity_range :
  1 < e ∧ e ≤ (sqrt 17)/3 :=
sorry

end hyperbola_eccentricity_range_l509_509992


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l509_509233

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l509_509233


namespace problem_l509_509651

def f (x : ℝ) : ℝ := x^2

def a : ℝ := f (Real.sin (2 * Real.pi / 7))
def b : ℝ := f (Real.cos (5 * Real.pi / 7))
def c : ℝ := f (Real.tan (5 * Real.pi / 7))

theorem problem : b < a ∧ a < c := by
  sorry

end problem_l509_509651


namespace necessary_but_not_sufficient_for_ellipse_l509_509688

theorem necessary_but_not_sufficient_for_ellipse (a b : ℝ) : 
  (ax^2 + by^2 = 1 ↔ ab > 0) → False ∧ (ab > 0 → ∃ x y, ax^2 + by^2 = 1) :=
by
  intros h
  sorry

end necessary_but_not_sufficient_for_ellipse_l509_509688


namespace smallest_missing_unit_digit_l509_509822

theorem smallest_missing_unit_digit :
  (∀ n, n ∈ [0, 1, 4, 5, 6, 9]) → ∃ smallest_digit, smallest_digit = 2 :=
by
  sorry

end smallest_missing_unit_digit_l509_509822


namespace triangle_arith_tan_half_angles_l509_509000

theorem triangle_arith_tan_half_angles 
  (a b c : ℝ)
  (α γ: ℝ)
  (h : a + c = 2 * b)
  (h1 : α < (π / 2))
  (h2 : γ > (π / 2) )
  : 3 * tan (α / 2) * tan (γ / 2) = 1 := 
sorry

end triangle_arith_tan_half_angles_l509_509000


namespace group_is_abelian_l509_509709

variable {G : Type} [Group G]
variable (e : G)
variable (h : ∀ x : G, x * x = e)

theorem group_is_abelian (a b : G) : a * b = b * a :=
sorry

end group_is_abelian_l509_509709


namespace simplify_and_rationalize_denominator_l509_509403

noncomputable def problem (x : ℚ) := 
  x = 1 / (1 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_denominator (x : ℚ) (h: problem x) :
  x = (Real.sqrt 5 - 1) / 4 :=
by
  sorry

end simplify_and_rationalize_denominator_l509_509403


namespace remainder_when_divided_l509_509106

theorem remainder_when_divided (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_when_divided_l509_509106


namespace james_writes_to_how_many_people_l509_509704

variable (pages_per_hour : ℕ) (pages_per_day : ℕ) (hours_per_week : ℕ)

-- Given conditions
def james_writes_pages_per_hour := pages_per_hour = 10
def james_writes_pages_per_day := pages_per_day = 5
def james_writes_hours_per_week := hours_per_week = 7

-- Question: To how many different people does James write daily?
theorem james_writes_to_how_many_people (pages_per_hour pages_per_day hours_per_week : ℕ) 
  (h1 : james_writes_pages_per_hour pages_per_hour) 
  (h2 : james_writes_pages_per_day pages_per_day) 
  (h3 : james_writes_hours_per_week hours_per_week) : 
  (hours_per_week * pages_per_hour) / pages_per_day = 14 := 
by 
  simp [james_writes_pages_per_hour, james_writes_pages_per_day, james_writes_hours_per_week] 
  sorry

end james_writes_to_how_many_people_l509_509704


namespace water_temp_conversion_l509_509813

-- Define the temperature conversion formula
def celsiusToFahrenheit (c : ℝ) : ℝ :=
  (c * (9 / 5)) + 32

-- The proof statement
theorem water_temp_conversion (c : ℝ) (h : c = 60) : celsiusToFahrenheit c = 140 :=
by
  rw [h]
  dsimp [celsiusToFahrenheit]
  norm_num
  sorry

end water_temp_conversion_l509_509813


namespace suitable_for_comprehensive_survey_l509_509494

/- Definitions for each condition -/
def optionA : Prop := "Investigating the service life of a certain brand of ballpoint pen refills"
def optionB : Prop := "Investigating the quality of a batch of food products"
def optionC : Prop := "Investigating the crash resistance of a batch of cars"
def optionD : Prop := "Investigating the nucleic acid test results of returning students in Pidu District"

/- The theorem stating that option D is the suitable survey for a comprehensive survey -/
theorem suitable_for_comprehensive_survey :
  optionD :=
sorry

end suitable_for_comprehensive_survey_l509_509494


namespace perpendicular_MP_AD_l509_509115

variable (A B C D K M P : Type)
variable [DecidableNonempty A] [DecidableNonempty B] [DecidableNonempty C] 
         [DecidableNonempty D] [DecidableNonempty K] [DecidableNonempty M] 
         [DecidableNonempty P]
variable (circle : Type → Prop)
variable (midpoint : Type → Type → Type → Prop)
variable (perpendicular : Type → Type → Prop)
variable (intersection : Type → Type → Type)
variable (diameter : Type → Type → Prop)
variable (cyclic : Type → Type → Type → Type → Prop)
variable (distinct : Type → Type → Prop)

-- Given a cyclic quadrilateral ABCD
axiom h1 : cyclic A B C D

-- M is the midpoint of side BC
axiom h2 : midpoint B C M

-- A perpendicular to BC through M intersects AB at K
axiom h3 : perpendicular (intersection (perpendicular BC M) AB) K

-- The circle with diameter KC intersects segment CD at P (P ≠ C)
axiom h4 : diameter K C (circle (intersection (circle diameter KC) CD))
axiom h5 : distinct P C

-- Prove that MP and AD are perpendicular
theorem perpendicular_MP_AD : perpendicular (intersection M P) AD :=
sorry

end perpendicular_MP_AD_l509_509115


namespace pb_eq_diameter_of_incircle_l509_509318

theorem pb_eq_diameter_of_incircle {ABC : Triangle} 
  (h_iso : is_isosceles_right_triangle ABC)
  (A B C P : Point)
  (h_AC_eq_BC : dist A C = dist B C)
  (h_bisector : is_angle_bisector A P (B, C)) : 
  dist P B = 2 * inradius ABC :=
sorry

end pb_eq_diameter_of_incircle_l509_509318


namespace magnitude_of_z_is_5_l509_509977
-- Import the necessary library

-- Define the complex number z
def z : ℂ := 4 - 3 * complex.i

-- Define the magnitude |z|
def magnitude (z : ℂ) : ℝ := complex.norm z

-- The theorem that we need to prove
theorem magnitude_of_z_is_5 : magnitude z = 5 := by
  sorry

end magnitude_of_z_is_5_l509_509977


namespace angle_B_in_right_triangle_in_degrees_l509_509687

def angleSum (A B C: ℝ) : Prop := A + B + C = 180

theorem angle_B_in_right_triangle_in_degrees (A B C : ℝ) (h1 : C = 90) (h2 : A = 35.5) (h3 : angleSum A B C) : B = 54.5 := 
by
  sorry

end angle_B_in_right_triangle_in_degrees_l509_509687


namespace irises_after_addition_l509_509053

theorem irises_after_addition
  (initial_roses : ℕ)
  (added_roses : ℕ)
  (ratio_irises_roses : ℕ)
  (ratio_roses_total : ℕ)
  : initial_roses = 30 → added_roses = 15 → ratio_irises_roses = 3 → ratio_roses_total = 7 →
  let final_roses := initial_roses + added_roses
  in (initial_roses * ratio_irises_roses + added_roses * ratio_irises_roses + final_roses * ratio_irises_roses)
  = 32 :=
by
  intros
  -- Conditions
  have initial_roses := 30
  have added_roses := 15
  have ratio_irises_roses := 3
  have ratio_roses_total := 7
  -- Final calculation
  let final_roses := initial_roses + added_roses
  let initial_irises := initial_roses * ratio_irises_roses / ratio_roses_total
  let additional_irises := (initial_roses + added_roses) * ratio_irises_roses / ratio_roses_total - initial_irises
  let total_irises := initial_irises + additional_irises
  -- Expected answer
  show total_irises = 32, from sorry

end irises_after_addition_l509_509053


namespace subtraction_decimal_nearest_hundredth_l509_509421

theorem subtraction_decimal_nearest_hundredth : 
  (845.59 - 249.27 : ℝ) = 596.32 :=
by
  sorry

end subtraction_decimal_nearest_hundredth_l509_509421


namespace solve_for_x_l509_509606

theorem solve_for_x (x : ℝ) : 
  let a := (x, 2) in
  let b := (x - 1, 1) in
  dot_product (a.fst + b.fst, a.snd + b.snd) (a.fst - b.fst, a.snd - b.snd) = 0 →
  x = -1 := sorry

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

end solve_for_x_l509_509606


namespace rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l509_509431

variables (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0)

def S1 := (x + 5) * (y + 5)
def S2 := (x - 2) * (y - 2)
def perimeter := 2 * (x + y)

theorem rectangle_perimeter (h : S1 - S2 = 196) :
  perimeter = 50 :=
sorry

theorem difference_multiple_of_7 (h : S1 - S2 = 196) :
  ∃ k : ℕ, S1 - S2 = 7 * k :=
sorry

theorem area_seamless_combination (h : S1 - S2 = 196) :
  S1 - x * y = (x + 5) * (y + 5) - x * y ∧ x = y + 5 :=
sorry

end rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l509_509431


namespace rohan_entertainment_percentage_l509_509007

def salary : ℝ := 12500
def savings : ℝ := 2500
def food_percentage : ℝ := 40 / 100
def rent_percentage : ℝ := 20 / 100
def conveyance_percentage : ℝ := 10 / 100
def entertainment_percentage (E : ℝ) : ℝ := E / 100

theorem rohan_entertainment_percentage (E : ℝ) :
  savings / salary = 1 / 5 →
  food_percentage + rent_percentage + conveyance_percentage + entertainment_percentage E + (savings / salary) = 1 →
  E = 10 :=
by
  intros h_savings h_expenses
  sorry

end rohan_entertainment_percentage_l509_509007


namespace no_integer_triple_exists_for_10_l509_509185

theorem no_integer_triple_exists_for_10 :
  ¬ ∃ (x y z : ℤ), 3 * x^2 + 4 * y^2 - 5 * z^2 = 10 :=
sorry

end no_integer_triple_exists_for_10_l509_509185


namespace all_of_the_above_used_as_money_l509_509830

-- Definition to state that each item was used as money
def gold_used_as_money : Prop := true
def stones_used_as_money : Prop := true
def horses_used_as_money : Prop := true
def dried_fish_used_as_money : Prop := true
def mollusk_shells_used_as_money : Prop := true

-- Statement that all of the above items were used as money
theorem all_of_the_above_used_as_money : gold_used_as_money ∧ stones_used_as_money ∧ horses_used_as_money ∧ dried_fish_used_as_money ∧ mollusk_shells_used_as_money :=
by {
  split; -- Split conjunctions
  all_goals { exact true.intro }; -- Each assumption is true
}

end all_of_the_above_used_as_money_l509_509830


namespace correct_choice_C_l509_509546

theorem correct_choice_C (x : ℝ) : x^2 ≥ x - 1 := 
sorry

end correct_choice_C_l509_509546


namespace sin_alpha_minus_pi_over_3_l509_509250

theorem sin_alpha_minus_pi_over_3
    (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
    (h3 : Real.tan (α / 2) + Real.cot (α / 2) = 5 / 2) :
    Real.sin (α - π / 3) = (4 - 3 * Real.sqrt 3) / 10 := by
    sorry

end sin_alpha_minus_pi_over_3_l509_509250


namespace find_s_for_closest_vector_l509_509177

def vector_u (s : ℝ) : ℝ × ℝ × ℝ :=
  (5 + 4 * s, -1 + 6 * s, -2 - 2 * s)

def vector_b : ℝ × ℝ × ℝ :=
  (3, 5, 7)

def direction_vector : ℝ × ℝ × ℝ :=
  (4, 6, -2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_s_for_closest_vector :
  ∃ s : ℝ, dot_product ((vector_u s).1 - vector_b.1, (vector_u s).2 - vector_b.2, (vector_u s).3 - vector_b.3) direction_vector = 0
  ∧ s = 5 / 19 :=
begin
  sorry
end

end find_s_for_closest_vector_l509_509177


namespace original_cost_l509_509843

theorem original_cost (P : ℝ) (h : 0.76 * P = 608) : P = 800 :=
by
  sorry

end original_cost_l509_509843


namespace lateral_edges_of_prism_l509_509820

/-- 
  A prism is a polyhedron with two congruent and parallel bases.
  The lateral faces of the prism are parallelograms.
  Prove that the lateral edges of a prism are parallel and equal in length.
-/
theorem lateral_edges_of_prism (bases_congruent : ∀ (P Q : Polyhedron), P.base ≃ Q.base) 
  (bases_parallel : ∀ (P : Polyhedron), P.base.parallel) 
  (lateral_faces_parallelogram : ∀ (P : Polyhedron), P.lateral.faces = Parallelogram) : 
  ∀ (P : Polyhedron), P.lateral.edges = Parallel ∧ P.lateral.edges = Equal_length :=
by
  sorry

end lateral_edges_of_prism_l509_509820


namespace mutually_exclusive_not_opposite_l509_509764

noncomputable def student_events_exclusive (students : Finset String) (balls : Finset String) 
  (ball_assignment : (String -> String)) : Prop :=
  (students = {"A", "B", "C", "D", "E"}) ∧
  (balls = {"red", "blue", "green", "yellow", "black"}) ∧
  (∀ s ∈ students, ball_assignment s ∈ balls) ∧
  (Function.Injective ball_assignment) ∧
  (ball_assignment "A" = "red" ∧ ball_assignment "B" = "red" → False)

theorem mutually_exclusive_not_opposite :
  ∃ ball_assignment : (String -> String), student_events_exclusive {"A", "B", "C", "D", "E"} 
  {"red", "blue", "green", "yellow", "black"} ball_assignment :=
begin
  sorry
end

end mutually_exclusive_not_opposite_l509_509764


namespace cos_C_value_l509_509277

theorem cos_C_value (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = 3 * c * Real.cos C)
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  : Real.cos C = (Real.sqrt 10) / 10 :=
sorry

end cos_C_value_l509_509277


namespace triangle_DEF_is_acute_l509_509539

noncomputable def triangle (A B C : Type) : Type :=
{ contains_inscribed_circle : A = B ∧ B = C ∧ C = A }

def has_inscribed_circle {A B C D E F : Type} (t : triangle A B C) : Prop :=
∀ (D E F : Type), touches_circle D ∧ touches_circle E ∧ touches_circle F

def touches_circle (P : Type) : Prop :=
∀ (circle : Type), P = point_of_tangent_pos 

def is_acute (A B C : Type) : Prop :=
∀ (angle_A angle_B angle_C : ℝ), angle_A < 90 ∧ angle_B < 90 ∧ angle_C < 90

theorem triangle_DEF_is_acute
  {A B C D E F : Type}
  (hABC : triangle A B C)
  (hInscribed : has_inscribed_circle hABC)
  : is_acute D E F :=
begin
  sorry
end

end triangle_DEF_is_acute_l509_509539


namespace angle_A_is_70_degrees_l509_509327

-- Defining the scalene triangle ABC with point O satisfying given conditions
noncomputable def scalene_triangle (A B C O : Type) : Prop :=
  ∃ (A B C O : ℝ),
    ∠OBC = 20 ∧
    ∠OCB = 20 ∧
    ∠BAO + ∠OCA = 70 ∧
    triangle A B C

-- Theorem statement
theorem angle_A_is_70_degrees (A B C O : Type) (h : scalene_triangle A B C O) : 
  ∃ (A B C : ℝ),
  ∠BAC = 70 :=
by
  sorry

end angle_A_is_70_degrees_l509_509327


namespace range_of_a_l509_509648

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x - a/(2^x)
noncomputable def C₁ (a : ℝ) (x : ℝ) : ℝ := 2^(x-2) - a/(2^(x-2))
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a/(2^(x-2)) - 2^(x-2) + 2
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := (1/a - 1/4) * 2^x + (4*a - 1)/(2^x) + 2

theorem range_of_a (a : ℝ) (m : ℝ) (h₁ : m > 2 + real.sqrt 7)
    (h₂ : ∀ x, F a x = (1/a - 1/4) * 2^x + (4*a - 1)/(2^x) + 2)
    (h₃ : ∃ x, F a x = m) :
    (1/2 < a ∧ a < 2) :=
sorry

end range_of_a_l509_509648


namespace sally_shots_l509_509890

/--
Sally took 30 shots initially and made 58% of her shots.
Then she took 8 more shots and increased her percentage to 60%.
Prove that Sally made 6 of the last 8 shots.
-/
theorem sally_shots (shots_taken_initially shots_additional: ℕ)
    (percent_initially percent_final: ℝ)
    (shots_made_initially shots_made_final: ℕ)
    (percent_initially_eq: percent_initially = 0.58)
    (shots_taken_initially_eq: shots_taken_initially = 30)
    (shots_additional_eq: shots_additional = 8)
    (percent_final_eq: percent_final = 0.60)
    (shots_made_initially_eq: shots_made_initially = 17)
    (shots_made_final_eq: shots_made_final = 23) :
    shots_made_final - shots_made_initially = 6 :=
begin
    sorry
end

end sally_shots_l509_509890


namespace angela_more_marbles_l509_509541

/--
Albert has three times as many marbles as Angela.
Allison has 28 marbles.
Albert and Allison have 136 marbles together.
Prove that Angela has 8 more marbles than Allison.
-/
theorem angela_more_marbles 
  (albert_angela : ℕ) 
  (angela: ℕ) 
  (albert: ℕ) 
  (allison: ℕ) 
  (h_albert_is_three_times_angela : albert = 3 * angela) 
  (h_allison_is_28 : allison = 28) 
  (h_albert_allison_is_136 : albert + allison = 136) 
  : angela - allison = 8 := 
by
  sorry

end angela_more_marbles_l509_509541


namespace david_marks_in_mathematics_l509_509926

-- Define marks in individual subjects and the average
def marks_in_english : ℝ := 70
def marks_in_physics : ℝ := 78
def marks_in_chemistry : ℝ := 60
def marks_in_biology : ℝ := 65
def average_marks : ℝ := 66.6
def number_of_subjects : ℕ := 5

-- Define a statement to be proven
theorem david_marks_in_mathematics : 
    average_marks * number_of_subjects 
    - (marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology) = 60 := 
by simp [average_marks, number_of_subjects, marks_in_english, marks_in_physics, marks_in_chemistry, marks_in_biology]; sorry

end david_marks_in_mathematics_l509_509926


namespace birds_find_more_than_half_millet_on_thursday_l509_509369

def millet_on_day (n : ℕ) : ℝ :=
  2 - 2 * (0.7 ^ n)

def more_than_half_millet (day : ℕ) : Prop :=
  millet_on_day day > 1

theorem birds_find_more_than_half_millet_on_thursday : more_than_half_millet 4 :=
by
  sorry

end birds_find_more_than_half_millet_on_thursday_l509_509369


namespace tan_of_sum_pi_over_4_and_alpha_l509_509632

theorem tan_of_sum_pi_over_4_and_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = sqrt 5 / 5) :
    tan (π / 4 + α) = -3 := sorry

end tan_of_sum_pi_over_4_and_alpha_l509_509632


namespace sam_quarters_l509_509009

theorem sam_quarters (pennies : ℕ) (total : ℝ) (value_penny : ℝ) (value_quarter : ℝ) (quarters : ℕ) :
  pennies = 9 →
  total = 1.84 →
  value_penny = 0.01 →
  value_quarter = 0.25 →
  quarters = (total - pennies * value_penny) / value_quarter →
  quarters = 7 :=
by
  intros
  sorry

end sam_quarters_l509_509009


namespace simplify_and_rationalize_denominator_l509_509402

noncomputable def problem (x : ℚ) := 
  x = 1 / (1 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_denominator (x : ℚ) (h: problem x) :
  x = (Real.sqrt 5 - 1) / 4 :=
by
  sorry

end simplify_and_rationalize_denominator_l509_509402


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l509_509232

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l509_509232


namespace quadrilateral_parallelogram_l509_509082

theorem quadrilateral_parallelogram (A B C D : Type) [Quadrilateral A B C D]
  (h1 : opposite_sides_parallel_and_equal A B C D) : is_parallelogram A B C D :=
sorry

end quadrilateral_parallelogram_l509_509082


namespace overall_percentage_change_is_correct_l509_509006

variable (S : ℝ)

-- Define the percentage changes as sequences of multipliers
def dec20 (x : ℝ) := x * 0.8
def inc15 (x : ℝ) := x * 1.15
def dec10 (x : ℝ) := x * 0.9
def inc25 (x : ℝ) := x * 1.25

-- Define the final salary after applying all changes
def finalSalary (S : ℝ) :=
  let afterDec20 := dec20 S
  let afterInc15 := inc15 afterDec20
  let afterDec10 := dec10 afterInc15
  inc25 afterDec10

-- Overall percentage change
def overallPercentageChange (initial final : ℝ) :=
  ((final - initial) / initial) * 100

-- The final theorem statement
theorem overall_percentage_change_is_correct :
  overallPercentageChange S (finalSalary S) = 3.5 := by
  sorry

end overall_percentage_change_is_correct_l509_509006


namespace perpendicular_lines_sufficient_l509_509240

noncomputable def line1_slope (a : ℝ) : ℝ :=
-((a + 2) / (3 * a))

noncomputable def line2_slope (a : ℝ) : ℝ :=
-((a - 2) / (a + 2))

theorem perpendicular_lines_sufficient (a : ℝ) (h : a = -2) :
  line1_slope a * line2_slope a = -1 :=
by
  sorry

end perpendicular_lines_sufficient_l509_509240


namespace sum_of_coordinates_D_l509_509389

theorem sum_of_coordinates_D :
  (∃ x, point_C = (0, 0) ∧ point_D = (x, 6) ∧ (6 - 0) / (x - 0) = 3 / 4) →
  (6 + 8 = 14) :=
by
  intro h
  rcases h with ⟨x, point_C_def, point_D_def, slope_def⟩
  -- Define coordinates
  let point_C := (0, 0)
  let point_D := (x, 6)
  
  -- Verifying point_C coordinates
  have point_C_coord : point_C = (0,0) := point_C_def
  
  -- Verifying point_D coordinates
  have point_D_coord : point_D = (x,6) := point_D_def

  -- Verifying slope
  have slope_eq : (6 - 0) / (x - 0) = 3 / 4 := slope_def
  -- Solve for x from the slope
  have x_value : x = 8 := by
    have h1 : 6 / x = 3 / 4 := slope_eq
    have h2 : 6 * 4 = 3 * x := by linarith
    have h3 : 24 = 3 * x := h2
    exact eq.symm (eq_div_of_mul_eq (eq.symm rfl : 3 * x = 24) (eq.refl 8))
  
  -- Coordinates is now (8, 6)
  -- Therefore, sum of coordinates is 8 + 6
  have coord_sum : 8 + 6 = 14 := by norm_num
  exact coord_sum

end sum_of_coordinates_D_l509_509389


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l509_509210

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l509_509210


namespace decrease_radius_to_balance_arcs_l509_509021

noncomputable def circle_on_chessboard 
  (unit_length : ℝ) (radius : ℝ) (centered_at_interior_square : Prop) 
  (intersected_squares_count_constant : Prop) : Prop :=
  radius = 1.9 →
  centered_at_interior_square →
  intersected_squares_count_constant →
  decrease_radius : Prop

axiom decrease_radius (radius : ℝ) (unit_length : ℝ) : Prop

theorem decrease_radius_to_balance_arcs 
  (unit_length : ℝ) (radius : ℝ) 
  (centered_at_interior_square : Prop) 
  (intersected_squares_count_constant: Prop)
  (h_radius: radius = 1.9) 
  (h_centered: centered_at_interior_square) 
  (h_intersected: intersected_squares_count_constant) 
  : decrease_radius radius unit_length :=
sorry

end decrease_radius_to_balance_arcs_l509_509021


namespace probability_both_quitters_same_tribe_l509_509311

theorem probability_both_quitters_same_tribe :
  (∃ p : ℚ, p = 24 / 51) :=
begin
  -- Define the number of contestants
  let n := 18,
  let tribe_size := 9,
  let total_ways := Nat.choose n 2,
  let ways_first_tribe := Nat.choose tribe_size 2,
  let ways_second_tribe := Nat.choose tribe_size 2,
  let ways_same_tribe := ways_first_tribe + ways_second_tribe,
  
  -- Calculate the probability
  let p : ℚ := ways_same_tribe / total_ways,

  -- State the proof goal to show that this probability is equal to 24 / 51
  use p,
  linarith,
end

end probability_both_quitters_same_tribe_l509_509311


namespace pipeC_drain_rate_l509_509809

-- Definitions of the conditions as provided:
def rateA := 1 / 12         -- Pipe A fills the tank in 12 minutes
def rateB := 1 / 20         -- Pipe B fills the tank in 20 minutes
def combinedRateABC := 1 / 15 -- All pipes together fill the tank in 15 minutes
def tankCapacity := 675     -- Tank capacity in liters

-- Theorem to prove the rate at which pipe C drains water
theorem pipeC_drain_rate : 
  let rateC := rateA + rateB - combinedRateABC in
  rateC * tankCapacity = 45 := 
by
  sorry

end pipeC_drain_rate_l509_509809


namespace least_odd_prime_factor_2048_pow_10_plus_1_l509_509948

noncomputable def least_odd_prime_factor (n : ℕ) : ℕ :=
  if h : n > 0 then Nat.find_greatest_prime_divisor h else 1

theorem least_odd_prime_factor_2048_pow_10_plus_1 :
  least_odd_prime_factor (2048^10 + 1) = 61 := by
  sorry

end least_odd_prime_factor_2048_pow_10_plus_1_l509_509948


namespace no_nat_fourfold_digit_move_l509_509003

theorem no_nat_fourfold_digit_move :
  ¬ ∃ (N : ℕ), ∃ (a : ℕ), ∃ (n : ℕ), ∃ (x : ℕ),
    (1 ≤ a ∧ a ≤ 9) ∧ 
    (N = a * 10^n + x) ∧ 
    (4 * N = 10 * x + a) :=
by
  sorry

end no_nat_fourfold_digit_move_l509_509003


namespace problem_solution_l509_509209

-- Define the given circle equation C
def circle_C_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 3 = 0

-- Define the line of symmetry
def line_symmetry_eq (x y : ℝ) : Prop := y = -x - 4

-- Define the symmetric circle equation
def sym_circle_eq (x y : ℝ) : Prop := (x + 4)^2 + (y + 6)^2 = 1

theorem problem_solution (x y : ℝ)
  (H1 : circle_C_eq x y)
  (H2 : line_symmetry_eq x y) :
  sym_circle_eq x y :=
sorry

end problem_solution_l509_509209


namespace fourth_square_area_l509_509069

theorem fourth_square_area (AB BC CD AC x : ℝ) 
  (h_AB : AB^2 = 49) 
  (h_BC : BC^2 = 25) 
  (h_CD : CD^2 = 64) 
  (h_AC1 : AC^2 = AB^2 + BC^2) 
  (h_AC2 : AC^2 = CD^2 + x^2) :
  x^2 = 10 :=
by
  sorry

end fourth_square_area_l509_509069


namespace distinct_values_count_l509_509692

def distinct_values : ℕ :=
  { s : ℕ |
    ∃ (a b c d : ℕ), 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      (a ∈ {2, 3, 5, 6}) ∧ 
      (b ∈ {2, 3, 5, 6}) ∧ 
      (c ∈ {2, 3, 5, 6}) ∧ 
      (d ∈ {2, 3, 5, 6}) ∧ 
      s = (a * b) + (c * d)
  }.to_finset.card

theorem distinct_values_count : distinct_values = 3 :=
  by
    sorry

end distinct_values_count_l509_509692


namespace Q_correct_l509_509182

noncomputable def probability (b : ℝ) (h_b : (1 / 2) ≤ b ∧ b ≤ 1) : ℝ :=
  -- Definition of the probability function goes here but is skipped with sorry
  sorry

noncomputable def Q (b : ℝ) (h_b : (1 / 2) ≤ b ∧ b ≤ 1) : ℝ :=
  probability b h_b

theorem Q_correct : ∀ (b : ℝ) (h_b : (1 / 2) ≤ b ∧ b ≤ 1), Q b h_b = 2 - sqrt 2 := 
  by
    sorry

end Q_correct_l509_509182


namespace change_positions_of_three_out_of_eight_l509_509513

theorem change_positions_of_three_out_of_eight :
  (Nat.choose 8 3) * (Nat.factorial 3) = (Nat.choose 8 3) * 6 :=
by
  sorry

end change_positions_of_three_out_of_eight_l509_509513


namespace simplify_expression_l509_509399

variable (x : ℝ)

theorem simplify_expression : 2 * (1 - (2 * (1 - (1 + (2 - (3 * x)))))) = -10 + 12 * x := 
  sorry

end simplify_expression_l509_509399


namespace bridget_apples_l509_509914

theorem bridget_apples (x : ℕ) (hx_half : x / 2 = y) (hx_cassie : y - 5 = z) (hx_self : z = 2): x = 14 :=
by
  have hx_half: x / 2 = 7 := sorry
  have hx_cassie: 7 - 5 = 2 := by 
  have hx_self: (7 - 5) = 2 := sorry
  sorry

end bridget_apples_l509_509914


namespace point_C_correct_l509_509119

-- Definitions of point A and B
def A : ℝ × ℝ := (4, -4)
def B : ℝ × ℝ := (18, 6)

-- Coordinate of C obtained from the conditions of the problem
def C : ℝ × ℝ := (25, 11)

-- Proof statement
theorem point_C_correct :
  ∃ C : ℝ × ℝ, (∃ (BC : ℝ × ℝ), BC = (1/2) • (B.1 - A.1, B.2 - A.2) ∧ C = (B.1 + BC.1, B.2 + BC.2)) ∧ C = (25, 11) :=
by
  sorry

end point_C_correct_l509_509119


namespace total_dice_in_James_bag_l509_509367

theorem total_dice_in_James_bag :
  ∀ (J : ℕ), 
    (∀ (M : ℕ), (M = 10) → (60% * M = 6) → 
     (∀ N, (14 - N = 12) → (∃ N, 75% * J = N)) →
     J = 8) :=
by 
  intros,
  sorry

end total_dice_in_James_bag_l509_509367


namespace abs_diff_of_prod_and_sum_l509_509346

theorem abs_diff_of_prod_and_sum (m n : ℝ) (h1 : m * n = 8) (h2 : m + n = 6) : |m - n| = 2 :=
by
  -- The proof is not required as per the instructions.
  sorry

end abs_diff_of_prod_and_sum_l509_509346


namespace fred_has_9_dimes_l509_509967

-- Fred has 90 cents in his bank.
def freds_cents : ℕ := 90

-- A dime is worth 10 cents.
def value_of_dime : ℕ := 10

-- Prove that the number of dimes Fred has is 9.
theorem fred_has_9_dimes : (freds_cents / value_of_dime) = 9 := by
  sorry

end fred_has_9_dimes_l509_509967


namespace person_A_takes_12_more_minutes_l509_509068

-- Define distances, speeds, times
variables (S : ℝ) (v_A v_B : ℝ) (t : ℝ)

-- Define conditions as hypotheses
def conditions (h1 : t = 2/5) (h2 : v_A = (2/3) * S / (t + 4/5)) (h3 : v_B = (2/3) * S / t) : Prop :=
  (v_A * (t + 4/5) = 2/3 * S) ∧ (v_B * t = 2/3 * S) ∧ (v_A * (t + 4/5 + 1/2 * t + 1/10) + 1/10 * v_B = S)

-- The proof problem statement
theorem person_A_takes_12_more_minutes
  (S : ℝ) (v_A v_B : ℝ) (t : ℝ)
  (h1 : t = 2/5) (h2 : v_A = (2/3) * S / (t + 4/5)) (h3 : v_B = (2/3) * S / t)
  (h4 : conditions S v_A v_B t h1 h2 h3) : (t + 4/5) + 6/5 = 96 / 60 + 12 / 60 :=
sorry

end person_A_takes_12_more_minutes_l509_509068


namespace blake_change_l509_509551

noncomputable def lollipop_cost : ℚ := 2
noncomputable def chocolate_cost : ℚ := 4 * lollipop_cost
noncomputable def gummy_bear_cost : ℚ := 3
noncomputable def candy_bar_cost : ℚ := 1.5

noncomputable def num_lollipops : ℕ := 4
noncomputable def num_chocolates : ℕ := 6
noncomputable def num_gummy_bears : ℕ := 3
noncomputable def num_candy_bars : ℕ := 5

noncomputable def total_given : ℚ := (4 * 20) + (2 * 5) + 5

noncomputable def lollipop_total_cost : ℚ := (3 * lollipop_cost) + (lollipop_cost / 2)
noncomputable def chocolate_total_cost : ℚ := 
  (2 * chocolate_cost) + (chocolate_cost * 0.75) +
  (2 * chocolate_cost) + (chocolate_cost * 0.75)
noncomputable def gummy_bear_total_cost : ℚ := num_gummy_bears * gummy_bear_cost
noncomputable def candy_bar_total_cost : ℚ := num_candy_bars * candy_bar_cost
noncomputable def total_cost : ℚ := lollipop_total_cost + chocolate_total_cost + gummy_bear_total_cost + candy_bar_total_cost

noncomputable def change : ℚ := total_given - total_cost

theorem blake_change : change = 27.5 := 
by simp [change, total_given, total_cost, lollipop_total_cost, chocolate_total_cost, gummy_bear_total_cost, candy_bar_total_cost, lollipop_cost, chocolate_cost, gummy_bear_cost, candy_bar_cost, num_lollipops, num_chocolates, num_gummy_bears, num_candy_bars]; norm_num; sorry

end blake_change_l509_509551


namespace train_passing_time_l509_509841

def train_length : ℝ := 440    -- Length of the train in meters
def train_speed_kmh : ℝ := 60  -- Speed of the train in km/h
def man_speed_kmh : ℝ := 6     -- Speed of the man in km/h
  
theorem train_passing_time : 
  let relative_speed_kmh := train_speed_kmh + man_speed_kmh in
  let relative_speed_ms := (relative_speed_kmh * 5) / 18 in
  let time_to_pass := train_length / relative_speed_ms in
  abs (time_to_pass - 24) < 1 :=
by 
  sorry

end train_passing_time_l509_509841


namespace sum_g_equals_12_l509_509181

def g (n : ℕ) : ℝ :=
  if ∃ (k : ℕ), n = 3^k then real.logb 27 n else 0

theorem sum_g_equals_12 : (∑ n in finset.range 7290.succ, g n) = 12 := by
  sorry

end sum_g_equals_12_l509_509181


namespace triangle_det_zero_l509_509724

-- Variables representing angles of a triangle
variables (A B C : ℝ)

-- Given that A, B, and C are angles of a triangle
-- i.e., they satisfy A + B + C = π, and are all positive

-- Define the matrix in terms of trigonometric functions of A, B, and C
def triangle_matrix (A B C : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
!![sin A * cos A, cot A, 1;
  sin B * cos B, cot B, 1;
  sin C * cos C, cot C, 1]

-- The determinant of the matrix, which should equal 0
theorem triangle_det_zero (h : A + B + C = π) : 
  (triangle_matrix A B C).det = 0 :=
by
  sorry

end triangle_det_zero_l509_509724


namespace approximation_range_l509_509794

theorem approximation_range (a : ℝ) (h : ∃ a_approx : ℝ, 170 = a_approx ∧ a = real_floor (a_approx) + 0.5) :
  169.5 ≤ a ∧ a < 170.5 :=
sorry

end approximation_range_l509_509794


namespace maximal_sequence_length_l509_509312

def sequence_property (s : ℕ → ℕ) : Prop :=
  ∀ n, s (n + 2) = abs (s n - s (n + 1))

theorem maximal_sequence_length :
  ∀ s : ℕ → ℕ, sequence_property s → (∀ n, s n ≤ 1967) → (∀ n, s n > 0) → ∃ N, N = 2952 :=
by
  sorry

end maximal_sequence_length_l509_509312


namespace total_amount_received_l509_509909

-- Define the initial prices and the increases
def initial_price_tv : ℝ := 500
def increase_ratio_tv : ℝ := 2/5
def initial_price_phone : ℝ := 400
def increase_ratio_phone : ℝ := 0.4

-- Calculate the total amount received
theorem total_amount_received : 
  initial_price_tv + increase_ratio_tv * initial_price_tv + initial_price_phone + increase_ratio_phone * initial_price_phone = 1260 :=
by {
  sorry
}

end total_amount_received_l509_509909


namespace PBD_angle_l509_509337

/-- Let ABCD be a convex quadrilateral with m(∠ADB) = 15°, m(∠BCD) = 90°. The diagonals intersect 
perpendicularly at E. Let P be a point on |AE| such that |EC| = 4, |EA| = 8, and |EP| = 2.
Prove that m(∠PBD) = 75°. -/
theorem PBD_angle (ABCD : ConvexQuadrilateral)
  (m_ADB : MeasureAngle ABCD.A ABCD.D ABCD.B = 15)
  (m_BCD : MeasureAngle ABCD.B ABCD.C ABCD.D = 90)
  (perpendicular_diagonals : Perpendicular ABCD.ABCD_diagnoals)
  (E : Point)
  (on_AC_diagonal : E ∈ Line (Diagonal ABCD))
  (EC_length : |Segment E ABCD.C| = 4)
  (EA_length : |Segment E ABCD.A| = 8)
  (EP_length : |Segment E P| = 2)
  : MeasureAngle P ABCD.B ABCD.D = 75 := 
sorry

end PBD_angle_l509_509337


namespace tamara_has_30_crackers_l509_509025

theorem tamara_has_30_crackers :
  ∀ (Tamara Nicholas Marcus Mona : ℕ),
    Tamara = 2 * Nicholas →
    Marcus = 3 * Mona →
    Nicholas = Mona + 6 →
    Marcus = 27 →
    Tamara = 30 :=
by
  intros Tamara Nicholas Marcus Mona h1 h2 h3 h4
  sorry

end tamara_has_30_crackers_l509_509025


namespace construct_larger_equilateral_triangle_l509_509565

theorem construct_larger_equilateral_triangle (T1 T2 : Triangle) 
  (h1 : T1.is_equilateral) (h2 : T1.side_length = 2) 
  (h3 : T2.is_equilateral) (h4 : T2.side_length = 3) : 
  ∃ T : Triangle, T.is_equilateral ∧ 
                  (T1.area + T2.area = T.area) := 
sorry

end construct_larger_equilateral_triangle_l509_509565


namespace smallest_int_solution_l509_509955

theorem smallest_int_solution (n : ℤ) (h : n^2 - 13 * n + 22 ≤ 0) : n ≥ 2 :=
begin
  sorry
end

end smallest_int_solution_l509_509955


namespace triangle_similarity_and_side_lengths_l509_509715

-- Definitions of semiperimeter and inradius are assumed to be part of Mathlib

variable {A B C : ℝ} -- Angles of the triangle
variable (s r: ℝ) -- Semiperimeter and inradius

-- Assume the given cotangent squared equation
axiom cot_half_angle_eq : 
  (Real.cot (A / 2))^2 + (2 * Real.cot (B / 2))^2 + (3 * Real.cot (C / 2))^2 = (6 * s / (7 * r))^2

-- Prove the similarity and side lengths
theorem triangle_similarity_and_side_lengths 
  (h₀: real.to_real (A + B + C) = π)
  (h₁: 0 < s)
  (h₂: 0 < r) :
  (exists (a b c : ℕ), nat.coprime a b ∧ nat.coprime b c ∧ nat.coprime c a ∧ 
   set.prod.mk a (set.mk b ∩ set.mk c) ⊆ {13, 40, 45}) :=
sorry

end triangle_similarity_and_side_lengths_l509_509715


namespace remainder_of_N_mod_37_l509_509107

theorem remainder_of_N_mod_37 (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_of_N_mod_37_l509_509107


namespace marble_probability_l509_509526

noncomputable def total_marbles : ℕ := 3 + 8 + 9
noncomputable def red_marbles : ℕ := 3
noncomputable def blue_marbles : ℕ := 8
noncomputable def yellow_marbles : ℕ := 9
noncomputable def prob_specific_sequence : ℚ := (3 / total_marbles) * (8 / (total_marbles - 1)) * (1 / 2)
noncomputable def num_ways : ℕ := 3!

theorem marble_probability :
  let total_prob : ℚ := num_ways * prob_specific_sequence in
  total_prob = 18 / 95 :=
by
  sorry

end marble_probability_l509_509526


namespace A_equals_half_C_equals_half_l509_509834

noncomputable def A := 2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180)
noncomputable def C := Real.sin (45 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - Real.cos (45 * Real.pi / 180) * Real.sin (15 * Real.pi / 180)

theorem A_equals_half : A = 1 / 2 := 
by
  sorry

theorem C_equals_half : C = 1 / 2 := 
by
  sorry

end A_equals_half_C_equals_half_l509_509834


namespace toes_on_bus_is_164_l509_509377

def num_toes_hoopit : Nat := 3 * 4
def num_toes_neglart : Nat := 2 * 5

def num_hoopits : Nat := 7
def num_neglarts : Nat := 8

def total_toes_on_bus : Nat :=
  num_hoopits * num_toes_hoopit + num_neglarts * num_toes_neglart

theorem toes_on_bus_is_164 : total_toes_on_bus = 164 := by
  sorry

end toes_on_bus_is_164_l509_509377


namespace coke_to_sprite_ratio_l509_509703

theorem coke_to_sprite_ratio
  (x : ℕ) 
  (Coke Sprite MountainDew : ℕ)
  (total_volume : ℕ) 
  (coke_volume : ℕ)
  (ratio_condition : Coke : Sprite : MountainDew = x : 1 : 3)
  (coke_volume_condition : coke_volume = 6)
  (total_volume_condition : total_volume = 18) :
  (Coke = 2) :=
by
  sorry

end coke_to_sprite_ratio_l509_509703


namespace corrected_mean_of_observations_l509_509789

theorem corrected_mean_of_observations (mean : ℝ) (n : ℕ) (incorrect_observation : ℝ) (correct_observation : ℝ) 
  (h_mean : mean = 41) (h_n : n = 50) (h_incorrect_observation : incorrect_observation = 23) (h_correct_observation : correct_observation = 48) 
  (h_sum_incorrect : mean * n = 2050) : 
  (mean * n - incorrect_observation + correct_observation) / n = 41.5 :=
by
  sorry

end corrected_mean_of_observations_l509_509789


namespace largest_is_C_l509_509836

def A : ℝ := 0.978
def B : ℝ := 0.9719
def C : ℝ := 0.9781
def D : ℝ := 0.917
def E : ℝ := 0.9189

theorem largest_is_C : 
  (C > A) ∧ 
  (C > B) ∧ 
  (C > D) ∧ 
  (C > E) := by
  sorry

end largest_is_C_l509_509836


namespace sphere_surface_area_of_right_square_prism_all_vertices_on_sphere_l509_509251

theorem sphere_surface_area_of_right_square_prism_all_vertices_on_sphere
  (height : ℝ) (volume : ℝ) (A : volume = 16) (B : height = 4)
  (h : ∀ {a b c d e f g h : ℝ}, 
    (∃ (s : set ℝ), s = {a, b, c, d, e, f, g, h} ∧ 
      ∀ (x : ℝ) (hx : x ∈ s), ∃ (S : sphere ℝ), S.contains x)) :
  ∃ (surface_area : ℝ), surface_area = 24 * π :=
by
  sorry

end sphere_surface_area_of_right_square_prism_all_vertices_on_sphere_l509_509251


namespace pigs_remaining_l509_509310

def initial_pigs : ℕ := 364
def pigs_joined : ℕ := 145
def pigs_moved : ℕ := 78

theorem pigs_remaining : initial_pigs + pigs_joined - pigs_moved = 431 := by
  sorry

end pigs_remaining_l509_509310


namespace total_amount_received_l509_509910

-- Define the initial prices and the increases
def initial_price_tv : ℝ := 500
def increase_ratio_tv : ℝ := 2/5
def initial_price_phone : ℝ := 400
def increase_ratio_phone : ℝ := 0.4

-- Calculate the total amount received
theorem total_amount_received : 
  initial_price_tv + increase_ratio_tv * initial_price_tv + initial_price_phone + increase_ratio_phone * initial_price_phone = 1260 :=
by {
  sorry
}

end total_amount_received_l509_509910


namespace incenter_on_circumcircle_l509_509355

-- Definitions based on conditions
structure Triangle (α : Type*) :=
(A B C : α)

variable {α : Type*} [linear_ordered_field α]

noncomputable def foot_of_angle_bisector (A B C : α) : α := sorry
noncomputable def perpendicular_bisector_intersection (A D : α) : α := sorry
noncomputable def angle_bisector_intersection (A B C : α) : α := sorry

def incenter (A B C : α) : α := sorry
def circumcircle (A B C : α) : set α := sorry

theorem incenter_on_circumcircle (A B C : α) : 
    let D := foot_of_angle_bisector A B C,
        X := perpendicular_bisector_intersection A D,
        Y := angle_bisector_intersection A B C in
    incenter A B C ∈ circumcircle A X Y := sorry

end incenter_on_circumcircle_l509_509355


namespace solve_a_for_pure_imaginary_l509_509295

theorem solve_a_for_pure_imaginary (a : ℝ) : (1 - a^2 = 0) ∧ (2 * a ≠ 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end solve_a_for_pure_imaginary_l509_509295


namespace sufficient_but_not_necessary_not_necessary_l509_509733

-- Define the sets M and P
def M := {x : ℝ | x > 1}
def P := {x : ℝ | x < 4}

-- Define the statement of the problem in Lean
theorem sufficient_but_not_necessary : 
  ∀ x : ℝ, (x ∈ {x : ℝ | 1 < x ∧ x < 4}) → (x ∈ {x : ℝ | x > 1 ∨ x < 4}) :=
begin
  sorry
end

-- Define the statement that it is not necessary
theorem not_necessary : 
  ∃ x : ℝ, (x ∈ {x : ℝ | x > 1 ∨ x < 4}) ∧ ¬(x ∈ {x : ℝ | 1 < x ∧ x < 4}) :=
begin
  sorry
end

end sufficient_but_not_necessary_not_necessary_l509_509733


namespace horse_merry_go_round_l509_509525

theorem horse_merry_go_round (r1 r2 rev1 rev2 : ℝ) (h1 : r1 = 24) (h2 : r2 = 8) (h3 : rev1 = 32) :
  rev2 = 96 :=
by
  -- Definitions and conditions directly translated to the form of Lean definitions
  have C1 := 2 * real.pi * r1,
  have C2 := 2 * real.pi * r2,
  have d1 := C1 * rev1,
  have d2 := C2 * rev2,
  -- The main result follows from the conditions and the proportionality
  have h := d1 = d2,
  have hc := r2 / r1 = 1 / 3,
  calc
    rev2 = (3 : ℝ) * rev1 : sorry
  done

end horse_merry_go_round_l509_509525


namespace find_value_of_expression_l509_509943

theorem find_value_of_expression : 
  (real.sqrt (5 ^ 5)) ^ 8 = 3125 := 
sorry

end find_value_of_expression_l509_509943


namespace task_selection_ways_l509_509457

theorem task_selection_ways :
  let group := 10 in
  let select := 4 in
  let taskA_people := 2 in
  let taskB_people := 1 in
  let taskC_people := 1 in
  let total_tasks := taskA_people + taskB_people + taskC_people in
  total_tasks = select →
  (Nat.choose group taskA_people) *
  (Nat.choose (group - taskA_people) (taskB_people + taskC_people)) *
  (taskB_people + taskC_people)! = 2520 :=
by
  intros
  sorry

end task_selection_ways_l509_509457


namespace cone_height_ratio_l509_509117

theorem cone_height_ratio
    (r : ℝ) (V_new : ℝ) (H : ℝ) (C : ℝ) (h_new : ℝ) :
    C = 20 * Real.pi →
    H = 40 →
    r = 10 →
    V_new = 320 * Real.pi →
    h_new = (3 * V_new) / (Real.pi * r ^ 2) →
    (h_new / H) = (6 / 25) :=
by {
  intros h1 h2 h3 h4 h5,
  sorry
}

end cone_height_ratio_l509_509117


namespace total_toes_on_bus_l509_509379

-- Definitions based on conditions
def toes_per_hand_hoopit : Nat := 3
def hands_per_hoopit : Nat := 4
def number_of_hoopits_on_bus : Nat := 7

def toes_per_hand_neglart : Nat := 2
def hands_per_neglart : Nat := 5
def number_of_neglarts_on_bus : Nat := 8

-- We need to prove that the total number of toes on the bus is 164
theorem total_toes_on_bus :
  (toes_per_hand_hoopit * hands_per_hoopit * number_of_hoopits_on_bus) +
  (toes_per_hand_neglart * hands_per_neglart * number_of_neglarts_on_bus) = 164 :=
by sorry

end total_toes_on_bus_l509_509379


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l509_509213

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l509_509213


namespace larger_of_two_numbers_l509_509799

open Real

theorem larger_of_two_numbers : ∃ x y : ℝ, x + y = 60 ∧ x * y = 882 ∧ x > y ∧ x = 30 + 3 * sqrt 2 :=
begin
  sorry
end

end larger_of_two_numbers_l509_509799


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l509_509234

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l509_509234


namespace biology_physics_ratio_l509_509804

theorem biology_physics_ratio (boys_bio : ℕ) (girls_bio : ℕ) (total_bio : ℕ) (total_phys : ℕ) 
  (h1 : boys_bio = 25) 
  (h2 : girls_bio = 3 * boys_bio) 
  (h3 : total_bio = boys_bio + girls_bio) 
  (h4 : total_phys = 200) : 
  total_bio / total_phys = 1 / 2 :=
by
  sorry

end biology_physics_ratio_l509_509804


namespace intersection_A_B_l509_509621

def A : Set ℕ := {x ∈ Set.univ | (x - 2) / x ≤ 0}
def B : Set ℤ := {x ∈ Set.univ | Real.log x / Real.log 2 ^ (1/2) < 1}

theorem intersection_A_B : A ∩ (B : Set ℕ) = {1, 2} :=
  sorry

end intersection_A_B_l509_509621


namespace necklace_probability_l509_509866

noncomputable def total_bead_arrangements : ℕ := 9.factorial / (4.factorial * 3.factorial * 2.factorial)

noncomputable def valid_red_bead_arrangements : ℕ := sorry -- Placeholder for the exact count of valid arrangements

/--
A necklace is made by arranging in random order four red beads, 
three white beads, and two blue beads. Prove the probability that 
no more than two red beads are next to each other is 1/8.
-/
theorem necklace_probability (total_arrangements valid_arrangements : ℕ) (h1 : total_arrangements = 1260)
  (h2 : valid_arrangements = 1260 / 8) :
  (valid_arrangements : ℝ) / (total_arrangements : ℝ) = 1 / 8 := by
  sorry

end necklace_probability_l509_509866


namespace find_vectors_and_cosine_l509_509281

noncomputable def vector_a (x : ℝ) : ℝ × ℝ × ℝ := (x, 4, 1)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ × ℝ := (-2, y, -1)
noncomputable def vector_c (z : ℝ) : ℝ × ℝ × ℝ := (3, -2, z)

def parallel (a b : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2, k * a.3)
def orthogonal (b c : ℝ × ℝ × ℝ) : Prop := b.1 * c.1 + b.2 * c.2 + b.3 * c.3 = 0
def cosine (u v : ℝ × ℝ × ℝ) : ℝ := (u.1 * v.1 + u.2 * v.2 + u.3 * v.3) / (Real.sqrt (u.1^2 + u.2^2 + u.3 * 2) * Real.sqrt (v.1^2 + v.2^2 + v.3^2))

theorem find_vectors_and_cosine:
  ∃ x y z, 
  parallel (vector_a x) (vector_b y) ∧ orthogonal (vector_b y) (vector_c z) ∧
  vector_a x = (2, 4, 1) ∧ vector_b y = (-2, -4, -1) ∧ vector_c z = (3, -2, 2) ∧
  cosine ((2 + 3, 4 -2, 1 + 2)) ((-2 + 3, -4 - 2, -1 + 2)) = -2 / 19 :=
by {
  sorry
}

end find_vectors_and_cosine_l509_509281


namespace rationalize_denominator_and_sum_l509_509004

noncomputable def A := 25
noncomputable def B := 15
noncomputable def C := 9
noncomputable def D := 2

theorem rationalize_denominator_and_sum:
  (1 / (real.cbrt 5 - real.cbrt 3) = 
  (real.cbrt A + real.cbrt B + real.cbrt C) / D) ∧ 
  (A + B + C + D = 51) :=
by {
  sorry
}

end rationalize_denominator_and_sum_l509_509004


namespace remainder_when_pow_304_div_5_l509_509821

theorem remainder_when_pow_304_div_5 (n : ℕ) : n = 304 -> (3^n) % 5 = 1 := 
by intro h; rw h; sorry

end remainder_when_pow_304_div_5_l509_509821


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l509_509225

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l509_509225


namespace parallelogram_of_conditions_l509_509689

theorem parallelogram_of_conditions
  {A B C D E F : Point}
  (hConvex : ConvexQuadrilateral A B C D)
  (hE : OnSegment E A B)
  (hF : OnSegment F B C)
  (hTrisection : Trisects DE AC ∧ Trisects DF AC)
  (hAreaADE : Area (Triangle A D E) = (1/4) * Area (Quadrilateral A B C D))
  (hAreaCDF : Area (Triangle C D F) = (1/4) * Area (Quadrilateral A B C D)) :
  Parallelogram A B C D :=
sorry

end parallelogram_of_conditions_l509_509689


namespace total_paid_is_201_l509_509889

def adult_ticket_price : ℕ := 8
def child_ticket_price : ℕ := 5
def total_tickets : ℕ := 33
def child_tickets : ℕ := 21
def adult_tickets : ℕ := total_tickets - child_tickets
def total_paid : ℕ := (child_tickets * child_ticket_price) + (adult_tickets * adult_ticket_price)

theorem total_paid_is_201 : total_paid = 201 :=
by
  sorry

end total_paid_is_201_l509_509889


namespace translation_right_2_units_l509_509891

theorem translation_right_2_units (A : ℝ × ℝ) (H : A = (-2, 3)) : 
  let A' := (A.1 + 2, A.2)
  A' = (0, 3) :=
by
  simp [H]
  rfl

end translation_right_2_units_l509_509891


namespace value_of_b_l509_509879

theorem value_of_b (b : ℝ) (f g : ℝ → ℝ) :
  (∀ x, f x = 2 * x^2 - b * x + 3) ∧ 
  (∀ x, g x = 2 * x^2 + b * x + 3) ∧ 
  (∀ x, g x = f (x + 6)) →
  b = 12 :=
by
  sorry

end value_of_b_l509_509879


namespace find_valid_pairs_l509_509633

theorem find_valid_pairs :
  ∃ (a b c : ℕ), 
    (a = 33 ∧ b = 22 ∧ c = 1111) ∨
    (a = 66 ∧ b = 88 ∧ c = 4444) ∨
    (a = 88 ∧ b = 33 ∧ c = 7777) ∧
    (11 ≤ a ∧ a ≤ 99) ∧ (11 ≤ b ∧ b ≤ 99) ∧ (1111 ≤ c ∧ c ≤ 9999) ∧
    (a % 11 = 0) ∧ (b % 11 = 0) ∧ (c % 1111 = 0) ∧
    (a * a + b = c) := sorry

end find_valid_pairs_l509_509633


namespace inequality_proof_l509_509755

theorem inequality_proof (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h : 2 * m ≤ n) :
  ( (n - m)! / m! ) ≤ ( (n / 2 + 1 / 2) ^ (n - 2 * m) ) :=
by
  sorry

end inequality_proof_l509_509755


namespace median_of_set_l509_509662

theorem median_of_set
  (a : ℤ) (b : ℝ)
  (h1 : a ≠ 0)
  (h2 : b > 0)
  (h3 : a * b^3 = Real.logb 2 b) :
  let s := ({0, 1, a, b, 1/b, b^2} : Set ℝ).toList.sort (≤) in
  median s = 3 / 8 :=
by
  sorry

end median_of_set_l509_509662


namespace compute_expression_value_l509_509920

-- Define the expression
def expression : ℤ := 1013^2 - 1009^2 - 1011^2 + 997^2

-- State the theorem with the required conditions and conclusions
theorem compute_expression_value : expression = -19924 := 
by 
  -- The proof steps would go here.
  sorry

end compute_expression_value_l509_509920


namespace largest_n_mod7_l509_509074

theorem largest_n_mod7 (n : ℕ) (h1 : n < 150_000) (h2 : (9 * (n - 3)^7 - 2 * n^3 + 15 * n - 33) % 7 = 0) : n = 149998 :=
sorry

end largest_n_mod7_l509_509074


namespace zoo_spending_l509_509465

def total_cost (goat_price llama_multiplier : ℕ) (num_goats num_llamas : ℕ) : ℕ :=
  let goat_cost := num_goats * goat_price
  let llama_price := llama_multiplier * goat_price
  let llama_cost := num_llamas * llama_price
  goat_cost + llama_cost

theorem zoo_spending :
  total_cost 400 3 2 == 4800 :=
by 
  let goat_price := 400
  let num_goats := 3
  let num_llamas := 2 * num_goats
  let llama_multiplier := 3
  let total := total_cost goat_price llama_multiplier num_goats num_llamas
  have h_goat_cost : total_cost goat_price llama_multiplier num_goats num_llamas = 4800 := sorry
  exact h_goat_cost

end zoo_spending_l509_509465


namespace order_of_magnitude_l509_509245

-- Define the constants for the problem
noncomputable def a := Real.log 0.8 / Real.log 1.1
noncomputable def b := Real.log 0.9 / Real.log 1.1
noncomputable def c := 1.1 ^ 0.9

-- State the theorem
theorem order_of_magnitude : a < b ∧ b < c :=
by
  sorry

end order_of_magnitude_l509_509245


namespace ellipse_distance_calc_l509_509148

noncomputable def distance (P Q : ℝ×ℝ) : ℝ :=
    real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem ellipse_distance_calc :
  let C := (-2, 4)
  let D := (-4, 0)
  distance C D = 2 * real.sqrt 5 :=
by sorry

end ellipse_distance_calc_l509_509148


namespace fraction_of_repeating_decimal_l509_509589

theorem fraction_of_repeating_decimal : ∀ (x : ℝ), 
  (x = 0.85858585...) → 
  (∃ (a : ℝ) (r : ℝ), x = a / (1 - r) ∧ a = 85 / 100 ∧ r = 1 / 100) → 
  x = 85 / 99 :=
by
  intro x
  intro hx
  intro h_series
  sorry

end fraction_of_repeating_decimal_l509_509589


namespace ratio_of_cone_altitude_to_radius_l509_509535

-- Given conditions
variables (r h : ℝ)
axiom shared_radius : ∀ (r h : ℝ), volume (cone_with_radius_base r h) = 1 / 3 * (volume_sphere r)

-- Volume calculations
def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

-- Lean 4 statement
theorem ratio_of_cone_altitude_to_radius (r h : ℝ) (hp : volume_cone r h = 1 / 3 * volume_sphere r) :
  h / r = 4 / 3 :=
by
  have h1 : (1 / 3) * Real.pi * r^2 * h = (1 / 3) * (4 / 3) * Real.pi * r^3 := by
    rw [volume_cone, volume_sphere] at hp
  have h2 : h = (4 / 3) * r := by
    simp at h1
    sorry
  sorry

end ratio_of_cone_altitude_to_radius_l509_509535


namespace all_marked_points_rational_l509_509397

theorem all_marked_points_rational
  (n : ℕ)
  (x : fin (n+2) → ℚ)
  (h0 : x 0 = 0)
  (h1 : x (n+1) = 1)
  (h_distinct : ∀ i j : fin (n+2), i ≠ j → x i ≠ x j)
  (h_middle : ∀ i : fin (n), ∃ (a b : fin (n+2)), (a ≠ i ∧ b ≠ i) ∧ x i = (x a + x b) / 2) :
  ∀ i : fin (n+2), is_rational (x i) :=
by
  sorry

end all_marked_points_rational_l509_509397


namespace find_de_l509_509693

def magic_square (f : ℕ × ℕ → ℕ) : Prop :=
  (f (0, 0) = 30) ∧ (f (0, 1) = 20) ∧ (f (0, 2) = f (0, 2)) ∧
  (f (1, 0) = f (1, 0)) ∧ (f (1, 1) = f (1, 1)) ∧ (f (1, 2) = f (1, 2)) ∧
  (f (2, 0) = 24) ∧ (f (2, 1) = 32) ∧ (f (2, 2) = f (2, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (1, 0) + f (1, 1) + f (1, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (2, 0) + f (2, 1) + f (2, 2)) ∧
  (f (0, 0) + f (0, 1) + f (0, 2) = f (0, 0) + f (1, 0) + f (2, 0)) ∧
  (f (0, 0) + f (1, 1) + f (2, 2) = f (0, 2) + f (1, 1) + f (2, 0)) 

theorem find_de (f : ℕ × ℕ → ℕ) (h : magic_square f) : 
  (f (1, 0) + f (1, 1) = 54) :=
sorry

end find_de_l509_509693


namespace exists_nonagon_with_given_midpoints_l509_509563

-- Definition of the points on the plane
variables {P : Type*} [MetricSpace P]

-- noncomputable instance to allow for the construction with classical geometry tools.
noncomputable instance : MetricSpace P := by sorry

-- Given nine points B1, B2, ..., B9 on the plane.
variables (B1 B2 B3 B4 B5 B6 B7 B8 B9 : P)

-- Lean statement: Prove that there exists a nonagon with these as midpoints.
theorem exists_nonagon_with_given_midpoints :
  ∃ A1 A2 A3 A4 A5 A6 A7 A8 A9 : P,
  midpoint A1 A2 = B1 ∧
  midpoint A2 A3 = B2 ∧
  midpoint A3 A4 = B3 ∧
  midpoint A4 A5 = B4 ∧
  midpoint A5 A6 = B5 ∧
  midpoint A6 A7 = B6 ∧
  midpoint A7 A8 = B7 ∧
  midpoint A8 A9 = B8 ∧
  midpoint A9 A1 = B9 :=
sorry

end exists_nonagon_with_given_midpoints_l509_509563


namespace sum_of_fifth_powers_l509_509223

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l509_509223


namespace value_at_2_l509_509629

-- Define f such that f(x^5) = log_b x
def f (x : ℝ) : ℝ := sorry

theorem value_at_2 (b : ℝ) (hb : b > 1) : f 2 = (1 / 5) * log b 2 :=
by
  sorry

end value_at_2_l509_509629


namespace inequality_solution_l509_509945

theorem inequality_solution (y : ℝ) :
  (1 / (y * (y + 2)) - 1 / ((y + 2) * (y + 4)) < 1 / 4) ↔
  (y ∈ set.Iio (-4) ∪ set.Ioo (-2) 0 ∪ set.Ioi 2) :=
by
  sorry

end inequality_solution_l509_509945


namespace louisa_average_speed_l509_509084

def average_speed (v : ℝ) : Prop :=
  (350 / v) - (200 / v) = 3

theorem louisa_average_speed :
  ∃ v : ℝ, average_speed v ∧ v = 50 := 
by
  use 50
  unfold average_speed
  sorry

end louisa_average_speed_l509_509084


namespace unique_arrangements_of_BANANA_l509_509553

-- Define the conditions as separate definitions in Lean 4
def word := "BANANA"
def total_letters := 6
def count_A := 3
def count_N := 2
def count_B := 1

-- State the theorem to be proven
theorem unique_arrangements_of_BANANA : 
  (total_letters.factorial) / (count_A.factorial * count_N.factorial * count_B.factorial) = 60 := 
by
  sorry

end unique_arrangements_of_BANANA_l509_509553


namespace length_of_AE_l509_509149

-- The given conditions
variables (A B C D E : Type)
variables (AB AC AD AE : ℝ)
variables (angle_BAC : ℝ)
variables (area_ABC area_ADE : ℝ)

-- Initial assumptions based on the problem conditions
variables (h_AB : AB = 4)
variables (h_AC : AC = 5)
variables (h_AD : AD = 2)
variables (h_angle_DAE : ∃ angle_DAE, angle_DAE = 2 * angle_BAC)
variables (h_area_eq : area_ABC = area_ADE)
variables (h_area_ABC : area_ABC = 0.5 * AB * AC * real.sin angle_BAC)
variables (h_area_ADE : area_ADE = 0.5 * AD * AE * real.sin(2 * angle_BAC))

-- The proof goal
theorem length_of_AE : AE = 5 / real.cos angle_BAC :=
sorry

end length_of_AE_l509_509149


namespace exists_positive_int_solutions_l509_509001

theorem exists_positive_int_solutions (a : ℕ) (ha : a > 2) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = a^2 :=
by
  sorry

end exists_positive_int_solutions_l509_509001


namespace de_moivres_theorem_l509_509392

open Complex

theorem de_moivres_theorem (φ : ℝ) (n : ℕ) : (cos φ + sin φ * Complex.i)^n = cos (n * φ) + sin (n * φ) * Complex.i :=
by
  sorry

end de_moivres_theorem_l509_509392


namespace xy_relationship_l509_509582

theorem xy_relationship (x y : ℝ) (h : y = 2 * x - 1 - Real.sqrt (y^2 - 2 * x * y + 3 * x - 2)) :
  (x ≠ 1 → y = 2 * x - 1.5) ∧ (x = 1 → y ≤ 1) :=
by
  sorry

end xy_relationship_l509_509582


namespace simplify_expr_l509_509408

noncomputable def expr := 1 / (1 + 1 / (Real.sqrt 5 + 2))
noncomputable def simplified := (Real.sqrt 5 + 1) / 4

theorem simplify_expr : expr = simplified :=
by
  sorry

end simplify_expr_l509_509408


namespace correct_equation_among_options_l509_509080

theorem correct_equation_among_options
  (a : ℝ) (x : ℝ) :
  (-- Option A
  ¬ ((-1)^3 = -3)) ∧
  (-- Option B
  ¬ (((-2)^2 * (-2)^3) = (-2)^6)) ∧
  (-- Option C
  ¬ ((2 * a - a) = 2)) ∧
  (-- Option D
  ((x - 2)^2 = x^2 - 4*x + 4)) :=
by
  sorry

end correct_equation_among_options_l509_509080


namespace intersection_sets_l509_509622

theorem intersection_sets :
  let M := {x : ℝ | 0 < x} 
  let N := {y : ℝ | 1 ≤ y}
  M ∩ N = {z : ℝ | 1 ≤ z} :=
by
  -- Proof goes here
  sorry

end intersection_sets_l509_509622


namespace quadratic_curve_passes_through_points_necessary_sufficient_condition_circle_collinearity_of_D_E_F_l509_509270

noncomputable def BC (x y θ₁ p₁ : ℝ) := x * cos θ₁ + y * sin θ₁ - p₁ = 0
noncomputable def CA (x y θ₂ p₂ : ℝ) := x * cos θ₂ + y * sin θ₂ - p₂ = 0
noncomputable def AB (x y θ₃ p₃ : ℝ) := x * cos θ₃ + y * sin θ₃ - p₃ = 0

noncomputable def quadratic_curve (x y θ₁ θ₂ θ₃ p₁ p₂ p₃ a b c : ℝ) :=
  a * (x * cos θ₂ + y * sin θ₂ - p₂) * (x * cos θ₃ + y * sin θ₃ - p₃) +
  b * (x * cos θ₃ + y * sin θ₃ - p₃) * (x * cos θ₁ + y * sin θ₁ - p₁) +
  c * (x * cos θ₁ + y * sin θ₁ - p₁) * (x * cos θ₂ + y * sin θ₂ - p₂) = 0

theorem quadratic_curve_passes_through_points
  (x y θ₁ θ₂ θ₃ p₁ p₂ p₃ a b c : ℝ)
  (h₁ : BC x y θ₁ p₁)
  (h₂ : CA x y θ₂ p₂)
  (h₃ : AB x y θ₃ p₃) :
  quadratic_curve x y θ₁ θ₂ θ₃ p₁ p₂ p₃ a b c :=
sorry

theorem necessary_sufficient_condition_circle
  (θ₁ θ₂ θ₃ a b c : ℝ) :
  a * cos (θ₂ + θ₃) + b * cos (θ₃ + θ₁) + c * cos (θ₁ + θ₂) = 0 ∧
  a * sin (θ₂ + θ₃) + b * sin (θ₃ + θ₁) + c * sin (θ₁ + θ₂) = 0 ↔
  (a : b : c) = (sin (θ₂ - θ₃) : sin (θ₃ - θ₁) : sin (θ₁ - θ₂)) :=
sorry

theorem collinearity_of_D_E_F
  (x y θ₁ θ₂ θ₃ p₁ p₂ p₃ : ℝ)
  (h_circle : quadratic_curve x y θ₁ θ₂ θ₃ p₁ p₂ p₃ (sin (θ₂ - θ₃)) (sin (θ₃ - θ₁)) (sin (θ₁ - θ₂)))
  (D E F : ℝ) :
  -- To-do: Define D, E, F in terms of x, y, θ₁, θ₂, θ₃, p₁, p₂, p₃
  collinear D E F :=
sorry

end quadratic_curve_passes_through_points_necessary_sufficient_condition_circle_collinearity_of_D_E_F_l509_509270


namespace max_value_2xy_sqrt6_8yz2_l509_509721

theorem max_value_2xy_sqrt6_8yz2 (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 6 + 8 * y * z^2 ≤ Real.sqrt 6 :=
sorry

end max_value_2xy_sqrt6_8yz2_l509_509721


namespace everything_used_as_money_l509_509828

theorem everything_used_as_money :
  (used_as_money gold) ∧
  (used_as_money stones) ∧
  (used_as_money horses) ∧
  (used_as_money dried_fish) ∧
  (used_as_money mollusk_shells) →
  (∀ x ∈ {gold, stones, horses, dried_fish, mollusk_shells}, used_as_money x) :=
by
  intro h
  cases h with
  | intro h_gold h_stones =>
    cases h_stones with
    | intro h_stones h_horses =>
      cases h_horses with
      | intro h_horses h_dried_fish =>
        cases h_dried_fish with
        | intro h_dried_fish h_mollusk_shells =>
          intro x h_x
          cases Set.mem_def.mpr h_x with
          | or.inl h => exact h_gold
          | or.inr h_x1 => cases Set.mem_def.mpr h_x1 with
            | or.inl h => exact h_stones
            | or.inr h_x2 => cases Set.mem_def.mpr h_x2 with
              | or.inl h => exact h_horses
              | or.inr h_x3 => cases Set.mem_def.mpr h_x3 with
                | or.inl h => exact h_dried_fish
                | or.inr h_x4 => exact h_mollusk_shells

end everything_used_as_money_l509_509828


namespace perpendicular_lines_condition_l509_509507

theorem perpendicular_lines_condition (m : ℝ) : (m = -1) ↔ ∀ (x y : ℝ), (x + y = 0) ∧ (x + m * y = 0) → 
  ((m ≠ 0) ∧ (-1) * (-1 / m) = 1) :=
by 
  sorry

end perpendicular_lines_condition_l509_509507


namespace opposite_of_negative_2023_l509_509438

-- Define the opposite condition
def is_opposite (y x : Int) : Prop := y + x = 0

theorem opposite_of_negative_2023 : ∃ x : Int, is_opposite (-2023) x ∧ x = 2023 :=
by 
  use 2023
  sorry

end opposite_of_negative_2023_l509_509438


namespace period_of_sin3x_plus_cos3x_l509_509482

noncomputable def period_of_trig_sum (x : ℝ) : ℝ := 
  let y := (fun x => Real.sin (3 * x) + Real.cos (3 * x))
  (2 * Real.pi) / 3

theorem period_of_sin3x_plus_cos3x : (fun x => Real.sin (3 * x) + Real.cos (3 * x)) =
  (fun x => Real.sin (3 * (x + period_of_trig_sum x)) + Real.cos (3 * (x + period_of_trig_sum x))) :=
by
  sorry

end period_of_sin3x_plus_cos3x_l509_509482


namespace problem_statement_l509_509235

variable (f g : ℝ → ℝ)
variable (f' g' : ℝ → ℝ)
variable (a b : ℝ)

theorem problem_statement (h1 : ∀ x, HasDerivAt f (f' x) x)
                         (h2 : ∀ x, HasDerivAt g (g' x) x)
                         (h3 : ∀ x, f' x < g' x)
                         (h4 : a = Real.log 2 / Real.log 5)
                         (h5 : b = Real.log 3 / Real.log 8) :
                         f a + g b > g a + f b := 
     sorry

end problem_statement_l509_509235


namespace classroom_ratio_l509_509520

theorem classroom_ratio (length width : ℕ) (h_length : length = 23) (h_width : width = 13) :
  let area := length * width in
  let ratio := width / area in
  ratio = 13 / 299 :=
by
  have area_def : area = length * width := rfl
  have ratio_def : ratio = width / area := rfl
  rw [area_def, h_length, h_width]
  suffices h1 : 23 * 13 = 299, by
  suffices h2 : width / 299 = 13 / 299, by
  exact h2
  simp [h_width]
  exact rfl
  simp [h1]
  exact sorry

end classroom_ratio_l509_509520


namespace convex_pentagon_cosine_distinct_l509_509505

theorem convex_pentagon_cosine_distinct {α β γ δ ε : ℝ}
  (h_convex : ∀ x ∈ {α, β, γ, δ, ε}, 0 < x ∧ x < π)
  (h_sum : α + β + γ + δ + ε = 3 * π)
  (h_sine_not_four_distinct : ¬ (number_of_distinct (sine ⁻¹' {α, β, γ, δ, ε}) ≥ 4)) :
  ¬ all_cosine_distinct {α, β, γ, δ, ε} :=
by
  sorry

end convex_pentagon_cosine_distinct_l509_509505


namespace area_of_PQYW_l509_509700

theorem area_of_PQYW 
(X Y Z W V P Q: Type*) 
[metric_space X] [metric_space Y] [metric_space Z] [metric_space W] [metric_space V] [metric_space P] [metric_space Q]
(XY XZ YZ : X → Y → ℝ) 
(a b : ℝ)
(hXYZ : XY (X) (Y) = 60)
(hXZ : XZ (X) (Z) = 15)
(area_XYZ : ℝ)
(h_area_XYZ : area_XYZ = 225)
(W_point : midpoint X Y W)
(V_point : midpoint X Z V)
(angle_bisector : bisector Y X Z P Q)
: area_quadrilateral P Q Y W = 123.75 := sorry

end area_of_PQYW_l509_509700


namespace sum_of_geometric_numbers_l509_509555

def is_geometric (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ∃ r : ℕ, r > 0 ∧ 
  (d2 = d1 * r) ∧ 
  (d3 = d2 * r) ∧ 
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

theorem sum_of_geometric_numbers : 
  (∃ smallest largest : ℕ,
    (smallest = 124) ∧ 
    (largest = 972) ∧ 
    is_geometric (smallest) ∧ 
    is_geometric (largest)
  ) →
  124 + 972 = 1096 :=
by
  sorry

end sum_of_geometric_numbers_l509_509555


namespace part1_part2_l509_509018

-- Part 1
theorem part1 (x y : ℤ) (hx : x = -2) (hy : y = -3) :
  x^2 - 2 * (x^2 - 3 * y) - 3 * (2 * x^2 + 5 * y) = -1 :=
by
  -- Proof to be provided
  sorry

-- Part 2
theorem part2 (a b : ℤ) (hab : a - b = 2 * b^2) :
  2 * (a^3 - 2 * b^2) - (2 * b - a) + a - 2 * a^3 = 0 :=
by
  -- Proof to be provided
  sorry

end part1_part2_l509_509018


namespace original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l509_509433

-- Condition declarations
variables (x y : ℕ)
variables (hx : x > 0) (hy : y > 0)
variables (h_sum : x + y = 25)
variables (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196)

-- Lean 4 statements for the proof problem
theorem original_rectangle_perimeter (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 25) : 2 * (x + y) = 50 := by
  sorry

theorem difference_is_multiple_of_7 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196) : 7 ∣ ((x + 5) * (y + 5) - (x - 2) * (y - 2)) := by
  sorry

theorem relationship_for_seamless_combination (x y : ℕ) (h_sum : x + y = 25) (h_seamless : (x+5)*(y+5) = (x*(y+5))) : x = y + 5 := by
  sorry

end original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l509_509433


namespace smallest_value_for_0_lt_x_lt_1_l509_509293

theorem smallest_value_for_0_lt_x_lt_1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : 
  x^2 < x ∧ x^2 < 2 * x ∧ x^2 < sqrt x ∧ x^2 < 1 / x :=
sorry

end smallest_value_for_0_lt_x_lt_1_l509_509293


namespace total_income_l509_509569

-- Definitions of conditions
def charge_per_meter : ℝ := 0.2
def number_of_fences : ℝ := 50
def length_of_each_fence : ℝ := 500

-- Theorem statement
theorem total_income :
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * charge_per_meter
  total_income = 5000 := 
by
  sorry

end total_income_l509_509569


namespace subtraction_of_decimals_l509_509422

theorem subtraction_of_decimals :
  888.8888 - 444.4444 = 444.4444 := 
sorry

end subtraction_of_decimals_l509_509422


namespace tens_place_of_8_pow_1234_l509_509818

theorem tens_place_of_8_pow_1234 : (8^1234 / 10) % 10 = 0 := by
  sorry

end tens_place_of_8_pow_1234_l509_509818


namespace sum_of_fifth_powers_l509_509220

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l509_509220


namespace range_of_f_l509_509260

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := 1 / (x ^ k)

theorem range_of_f (k : ℝ) (h : 0 < k) :
  set.range (fun x : ℝ => f x k) (set.Ici 1) = set.Ioc 0 1 :=
sorry

end range_of_f_l509_509260


namespace horizontal_asymptote_f_l509_509037

noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 4) / (4 * x^2 + 6 * x + 3)

theorem horizontal_asymptote_f : filter.tendsto (λ x : ℝ, f x) filter.at_top (𝓝 (3 / 2)) :=
sorry

end horizontal_asymptote_f_l509_509037


namespace least_sum_p_q_r_l509_509360

theorem least_sum_p_q_r (p q r : ℕ) (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (h : 17 * (p + 1) = 28 * (q + 1) ∧ 28 * (q + 1) = 35 * (r + 1)) : p + q + r = 290 :=
  sorry

end least_sum_p_q_r_l509_509360


namespace element_squared_is_identity_l509_509710

universe u

variables {G : Type u} [group G]
variables {H K : set G} [is_subgroup H] [is_subgroup K]
variables (e : G)
variables (G_closed : ∀ x y ∈ (G \setminus (H ∪ K)) ∪ {e}, x * y ∈ (G \setminus (H ∪ K)) ∪ {e})

open set

theorem element_squared_is_identity (G_neutral : ∀ x : G, x * e = x ∧ e * x = x)
  (H_proper : is_proper_subgroup H)
  (K_proper : is_proper_subgroup K)
  (H_inter_K : H ∩ K = {e}) :
  ∀ x : G, x * x = e := 
by
  sorry

end element_squared_is_identity_l509_509710


namespace total_number_of_students_is_3000_l509_509524

-- Definitions based on the given conditions
def total_students_selected : ℕ := 20
def senior_class_students : ℕ := 900
def selected_non_senior_students : ℕ := 14

-- The problem statement in Lean 4 to be proved
theorem total_number_of_students_is_3000
  (total_selected : ℕ) (senior_students : ℕ) (selected_non_seniors : ℕ)
  (h_total_selected : total_students_selected = total_selected)
  (h_senior_students : senior_class_students = senior_students)
  (h_selected_non_seniors : selected_non_senior_students = selected_non_seniors) :
  ∃ n : ℕ, (6 * n = 900 * 20) ∧ n = 3000 :=
by
  existsi 3000
  split
  case 1 =>
    calc
      6 * 3000 = 18000 := by norm_num
      900 * 20 = 18000 := by norm_num
      6 * 3000 = 900 * 20 := by refl
  case 2 =>
    rfl

end total_number_of_students_is_3000_l509_509524


namespace minimum_berries_left_l509_509846

def geometric_sum (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

theorem minimum_berries_left {a r n S : ℕ} 
  (h_a : a = 1) 
  (h_r : r = 2) 
  (h_n : n = 100) 
  (h_S : S = geometric_sum a r n) 
  : S = 2^100 - 1 -> ∃ k, k = 100 :=
by
  sorry

end minimum_berries_left_l509_509846


namespace share_of_B_l509_509498

theorem share_of_B (x : ℕ) (A B C : ℕ) (h1 : A = 3 * B) (h2 : B = C + 25)
  (h3 : A + B + C = 645) : B = 134 :=
by
  sorry

end share_of_B_l509_509498


namespace problem_l509_509451

theorem problem (a : ℤ) :
  (∃ x : ℝ, x > 0 ∧ (3 * x - a) / (x - 3) + (x + 1) / (3 - x) = 1) ∧
  (∀ y : ℝ, (y + 9 ≤ 2 * (y + 2)) ∧ ((2 * y - a) / 3 ≥ 1) ↔ (y ≥ 5)) →
  (a ∈ {3, 4, 6, 7} → a ∈ {3, 4, 6, 7}) :=
by
  sorry

end problem_l509_509451


namespace valerie_needs_21_stamps_l509_509812

def thank_you_cards : ℕ := 3
def bills : ℕ := 2
def mail_in_rebates : ℕ := bills + 3
def job_applications : ℕ := 2 * mail_in_rebates
def water_bill_stamps : ℕ := 1
def electric_bill_stamps : ℕ := 2

def stamps_for_thank_you_cards : ℕ := thank_you_cards * 1
def stamps_for_bills : ℕ := 1 * water_bill_stamps + 1 * electric_bill_stamps
def stamps_for_rebates : ℕ := mail_in_rebates * 1
def stamps_for_job_applications : ℕ := job_applications * 1

def total_stamps : ℕ :=
  stamps_for_thank_you_cards +
  stamps_for_bills +
  stamps_for_rebates +
  stamps_for_job_applications

theorem valerie_needs_21_stamps : total_stamps = 21 := by
  sorry

end valerie_needs_21_stamps_l509_509812


namespace tangent_circle_exists_l509_509924

open EuclideanGeometry

noncomputable def constructTangentCircle
  (circle : Circle)
  (line : Line)
  (M : Point)
  (onLineM : M ∈ line) : Circle :=
  let A := (some method to find A: Point) in
  let B := (some method to find B: Point) in
  let AB := Line.mk A B in
  let E := (some method to find E: Point where E is intersection of Line.mk A M with circle) in
  let MN := (some method to find MN: Segment with M and N where N is intersection of Perpendicular through M to line with Line.mk B E) in
  Circle.mk M N

theorem tangent_circle_exists
  (circle : Circle)
  (line : Line)
  (M : Point)
  (onLineM : M ∈ line ) :
  ∃ (circ' : Circle), (tangent_to circ' circle) ∧ (tangent_to circ' line) :=
by
  let circ' := constructTangentCircle circle line M onLineM
  exists_val
  split
  · sorry -- Prove that circ' is tangent to circle
  · sorry -- Prove that circ' is tangent to line


end tangent_circle_exists_l509_509924


namespace arrange_chairs_and_stools_l509_509459

-- We need to define the conditions and prove the corresponding combination formula
theorem arrange_chairs_and_stools :
  let total_slots := 10
  let stools := 3
  let chairs := 7
  7 + 3 = total_slots →
  Nat.choose total_slots stools = 120 :=
by
  intros h
  rw [Nat.choose_eq_factorial_div_factorial (le_of_lt (Nat.lt_succ_self stools))]
  simp only [Nat.factorial_succ]
  rw [Nat.factorial, Nat.factorial, Nat.factorial, Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_succ]
  norm_num
  exact rfl

end arrange_chairs_and_stools_l509_509459


namespace distance_circumcenter_incenter_l509_509324

variable (A B C O I : Type)
variable (R r : ℝ)
variable [fact (triangle A B C)] -- Assuming triangle configuration as a fact
variable [circumcenter A B C O]  -- O is the circumcenter of triangle ABC
variable [incenter A B C I]      -- I is the incenter of triangle ABC
variable [circumradius A B C O R] -- R is the circumradius
variable [inradius A B C I r]     -- r is the inradius

theorem distance_circumcenter_incenter 
  (h₁ : O = circumcenter A B C) 
  (h₂ : I = incenter A B C) 
  (h₃ : circumradius A B C = R) 
  (h₄ : inradius A B C = r) : 
  dist O I^2 = R^2 - 2 * R * r := 
sorry

end distance_circumcenter_incenter_l509_509324


namespace func1_bijective_func2_bijective_func3_neither_injective_nor_surjective_func4_neither_injective_nor_surjective_l509_509962

noncomputable def func1 (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f(f(x) - 1) = x + 1

def injective (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f(x) = f(y) → x = y

def surjective (f : ℝ → ℝ) : Prop :=
∀ y : ℝ, ∃ x : ℝ, f(x) = y

def bijective (f : ℝ → ℝ) : Prop :=
injective f ∧ surjective f

theorem func1_bijective {f : ℝ → ℝ} (h : func1 f) : bijective f := 
sorry

noncomputable def func2 (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f(x + f(y)) = f(x) + y^5

theorem func2_bijective {f : ℝ → ℝ} (h : func2 f) : bijective f := 
sorry

noncomputable def func3 (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f(f(x)) = sin x

theorem func3_neither_injective_nor_surjective {f : ℝ → ℝ} (h : func3 f) : ¬(injective f) ∧ ¬(surjective f) := 
sorry

noncomputable def func4 (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f(x + y^2) = f(x) * f(y) + x * f(y) - y^3 * f(x)

theorem func4_neither_injective_nor_surjective {f : ℝ → ℝ} (h : func4 f) : ¬(injective f) ∧ ¬(surjective f) := 
sorry

end func1_bijective_func2_bijective_func3_neither_injective_nor_surjective_func4_neither_injective_nor_surjective_l509_509962


namespace garden_perimeter_l509_509872

theorem garden_perimeter
  (a b : ℝ)
  (h1 : a^2 + b^2 = 1156)
  (h2 : a * b = 240) :
  2 * (a + b) = 80 :=
sorry

end garden_perimeter_l509_509872


namespace exists_increasing_sequence_l509_509746

theorem exists_increasing_sequence (a_1 : ℕ) (h : a_1 > 1) :
  ∃ a : ℕ → ℕ, (∀ k, a k = 2 * 3 ^ k) ∧ ∀ k ≥ 1, (∑ i in finset.range (k + 1), (a i) ^ 2) % (∑ i in finset.range (k + 1), a i) = 0 :=
by {
  -- proof not required.
  sorry
}

end exists_increasing_sequence_l509_509746


namespace probability_of_bijection_l509_509673

-- Given sets A and B
def A : set ℕ := {1, 2}
def B : set ℕ := {1, 2, 3}

-- Definition of a bijection between subsets of A and B
def is_bijection (f : ℕ → ℕ) : Prop :=
  (∀ x ∈ A, f x ∈ B) ∧ function.bijective f

-- The probability of forming a bijection from subsets of A to B is 2/3.
theorem probability_of_bijection :
  ∃ (f : ℕ → ℕ), is_bijection f →
  (6 / 9 : ℚ) = 2 / 3 :=
by
  sorry

end probability_of_bijection_l509_509673


namespace commute_times_absolute_difference_l509_509453

theorem commute_times_absolute_difference
  (x y : ℝ)
  (H_avg : (x + y + 10 + 11 + 9) / 5 = 10)
  (H_var : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2) :
  abs (x - y) = 4 :=
by
  -- proof steps are omitted
  sorry

end commute_times_absolute_difference_l509_509453


namespace total_income_l509_509571

-- Definitions of conditions
def charge_per_meter : ℝ := 0.2
def number_of_fences : ℝ := 50
def length_of_each_fence : ℝ := 500

-- Theorem statement
theorem total_income :
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * charge_per_meter
  total_income = 5000 := 
by
  sorry

end total_income_l509_509571


namespace litter_collection_total_weight_l509_509969

/-- Gina collected 8 bags of litter: 5 bags of glass bottles weighing 7 pounds each and 3 bags of plastic waste weighing 4 pounds each. The 25 neighbors together collected 120 times as much glass as Gina and 80 times as much plastic as Gina. Prove that the total weight of all the collected litter is 5207 pounds. -/
theorem litter_collection_total_weight
  (glass_bags_gina : ℕ)
  (glass_weight_per_bag : ℕ)
  (plastic_bags_gina : ℕ)
  (plastic_weight_per_bag : ℕ)
  (neighbors_glass_multiplier : ℕ)
  (neighbors_plastic_multiplier : ℕ)
  (total_weight : ℕ)
  (h1 : glass_bags_gina = 5)
  (h2 : glass_weight_per_bag = 7)
  (h3 : plastic_bags_gina = 3)
  (h4 : plastic_weight_per_bag = 4)
  (h5 : neighbors_glass_multiplier = 120)
  (h6 : neighbors_plastic_multiplier = 80)
  (h_total_weight : total_weight = 5207) : total_weight = 
  glass_bags_gina * glass_weight_per_bag + 
  plastic_bags_gina * plastic_weight_per_bag + 
  neighbors_glass_multiplier * (glass_bags_gina * glass_weight_per_bag) + 
  neighbors_plastic_multiplier * (plastic_bags_gina * plastic_weight_per_bag) := 
by {
  /- Proof omitted -/
  sorry
}

end litter_collection_total_weight_l509_509969


namespace factorial_divide_l509_509559

theorem factorial_divide {n m k : ℕ} (h : n = 11) (h1 : m = 8) (h2 : k = 3) : (nat.factorial n / (nat.factorial m * nat.factorial k)) = 165 := 
by {
  rw [h, h1, h2],
  sorry
}

end factorial_divide_l509_509559


namespace total_arrangements_l509_509852

theorem total_arrangements :
  let students := 6
  let venueA := 1
  let venueB := 2
  let venueC := 3
  (students.choose venueA) * ((students - venueA).choose venueB) = 60 :=
by
  -- placeholder for the proof
  sorry

end total_arrangements_l509_509852


namespace find_integer_modulo_l509_509947

theorem find_integer_modulo :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -2023 [MOD 9] ∧ n = 2 :=
by
  sorry

end find_integer_modulo_l509_509947


namespace proof_problem_l509_509728

noncomputable def problem_statement (s t : ℕ → ℚ) : Prop :=
  (∀ i j, (s i - s j) * (t i - t j) ∈ ℤ) ∧
  ∃ r : ℚ, ∀ i j, (s i - s j) * r ∈ ℤ ∧ (t i - t j) / r ∈ ℤ

theorem proof_problem (s t : ℕ → ℚ) (h1 : ∀ n, s n ≠ s (n + 1)) (h2 : ∀ n, t n ≠ t (n + 1))
  (h3 : ∀ i j, (s i - s j) * (t i - t j) ∈ ℤ) :
  ∃ r : ℚ, ∀ i j, (s i - s j) * r ∈ ℤ ∧ (t i - t j) / r ∈ ℤ :=
sorry

end proof_problem_l509_509728


namespace usual_time_of_reach_school_l509_509475

-- The conditions from the problem
variables (R T : ℝ)
-- The boy walks at 3/2 of his usual rate and arrives 4 minutes early.
-- We need these facts:
variable h_faster : T - 4 = (2 / 3) * T

-- The proof goal stated:
theorem usual_time_of_reach_school : T = 12 := by
  sorry

end usual_time_of_reach_school_l509_509475


namespace expected_points_l509_509851

/-- 
Define the expected values given the game conditions.
Let E_0 be the expected number of points starting with no players on base.
Define E_1, E_2, and E_3 as the expected number of points starting with a player on
first, second, and third base, respectively.
-/
def E₀ : ℝ := (E₁ + E₂ + E₃ + 1)
def E₁ : ℝ := (1/5 * E₂) + (1/5 * E₃) + (2/5)
def E₂ : ℝ := (1/5 * E₃) + (3/5)
def E₃ : ℝ := (4/5)

/--
Expected number of points that a given team will score in RNG baseball is 409/125.
-/
theorem expected_points : E₀ = 409 / 125 := by
  sorry -- Proof not required, place-holder.

end expected_points_l509_509851


namespace line_slope_m_l509_509798

theorem line_slope_m :
  ∀ (m : ℝ), (∀ x : ℝ, mx + sin (135 * π / 180) ≤ 0) → m = 1 :=
by
  intro m
  intro H
  sorry

end line_slope_m_l509_509798


namespace tangent_length_l509_509145

structure Point where
  x : ℝ
  y : ℝ

def O : Point := { x := 0, y := 0 }
def A : Point := { x := 2, y := 3 }
def B : Point := { x := 4, y := 6 }
def C : Point := { x := 3, y := 9 }

-- Definition of a tangent length from a point to a circle passing through A, B, and C
def length_of_tangent (O A B C : Point) : ℝ := 
  sqrt 15

theorem tangent_length :
  length_of_tangent O A B C = 3 * sqrt 5 := 
by
  -- Placeholder for the proof
  sorry

end tangent_length_l509_509145


namespace geometric_sequence_sum_l509_509639

theorem geometric_sequence_sum :
  ∀ (a : ℕ → ℕ) (n : ℕ),
  (a 1 = 1) → (a 3 = 4) → (∀ n, a (n+1) = a 1 * (2^n)) → Σ (i in fin 5, a (i + 1)) = 31 :=
by
  sorry

end geometric_sequence_sum_l509_509639


namespace min_distance_from_circle_to_line_l509_509652

-- Define the circle and line conditions
def is_on_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def line (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0

-- The theorem to prove
theorem min_distance_from_circle_to_line (x y : ℝ) (h : is_on_circle x y) : 
  ∃ m_dist : ℝ, m_dist = 2 :=
by
  -- Place holder proof
  sorry

end min_distance_from_circle_to_line_l509_509652


namespace sum_rows_7_8_pascal_triangle_l509_509490

theorem sum_rows_7_8_pascal_triangle : (2^7 + 2^8 = 384) :=
by
  sorry

end sum_rows_7_8_pascal_triangle_l509_509490


namespace midpoint_quadrilateral_of_equal_diagonals_is_rhombus_midpoint_quadrilateral_of_perpendicular_diagonals_is_not_rhombus_midpoint_quadrilateral_of_rectangle_is_not_square_l509_509476

structure Quadrilateral :=
(a b c d : ℝ × ℝ)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def midpoint_quadrilateral (quad : Quadrilateral) : Quadrilateral :=
{ a := midpoint quad.a quad.b,
  b := midpoint quad.b quad.c,
  c := midpoint quad.c quad.d,
  d := midpoint quad.d quad.a }

def diagonals_equal (quad : Quadrilateral) : Prop :=
let (a, b, c, d) := (quad.a, quad.b, quad.c, quad.d) in
((a.1 - c.1)^2 + (a.2 - c.2)^2 = (b.1 - d.1)^2 + (b.2 - d.2)^2)

def diagonals_perpendicular (quad : Quadrilateral) : Prop :=
let (a, b, c, d) := (quad.a, quad.b, quad.c, quad.d) in
((a.1 - c.1) * (b.1 - d.1) + (a.2 - c.2) * (b.2 - d.2) = 0)

def is_rectangle (quad : Quadrilateral) : Prop :=
let (a, b, c, d) := (quad.a, quad.b, quad.c, quad.d) in
(a.1 = d.1 ∧ b.1 = c.1 ∧ a.2 = b.2 ∧ c.2 = d.2)

def is_rhombus (quad : Quadrilateral) : Prop :=
let (a, b, c, d) := (quad.a, quad.b, quad.c, quad.d) in
(a.1 - b.1)^2 + (a.2 - b.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∧
(c.1 - d.1)^2 + (c.2 - d.2)^2 = (d.1 - a.1)^2 + (d.2 - a.2)^2

def is_square (quad : Quadrilateral) : Prop :=
is_rhombus quad ∧ is_rectangle quad

theorem midpoint_quadrilateral_of_equal_diagonals_is_rhombus (quad : Quadrilateral) (h : diagonals_equal quad) :
  is_rhombus (midpoint_quadrilateral quad) :=
sorry

theorem midpoint_quadrilateral_of_perpendicular_diagonals_is_not_rhombus (quad : Quadrilateral) (h : diagonals_perpendicular quad) :
  ¬ is_rhombus (midpoint_quadrilateral quad) :=
sorry

theorem midpoint_quadrilateral_of_rectangle_is_not_square (quad : Quadrilateral) (h : is_rectangle quad) :
  ¬ is_square (midpoint_quadrilateral quad) :=
sorry

end midpoint_quadrilateral_of_equal_diagonals_is_rhombus_midpoint_quadrilateral_of_perpendicular_diagonals_is_not_rhombus_midpoint_quadrilateral_of_rectangle_is_not_square_l509_509476


namespace initial_pigs_count_l509_509371

theorem initial_pigs_count (P : ℕ) (h1 : 2 + P + 6 + 3 + 5 + 2 = 21) : P = 3 :=
by
  sorry

end initial_pigs_count_l509_509371


namespace find_p_of_tangency_l509_509634

-- condition definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def parabola_eq (p x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0
def circle_center : (ℝ × ℝ) := (3, 0)
def circle_radius : ℝ := 4
def directrix (p : ℝ) : ℝ := -p / 2

-- theorem definition
theorem find_p_of_tangency (p : ℝ) :
  (∀ x y : ℝ, circle_eq x y) →
  (∀ x y : ℝ, parabola_eq p x y) →
  dist (circle_center.fst) (directrix p / 0) = circle_radius →
  p = 2 :=
sorry

end find_p_of_tangency_l509_509634


namespace hyperbola_eccentricity_sqrt2_l509_509509

-- Definitions based on the problem conditions
structure Hyperbola := 
  (a b : ℝ) 
  (a_pos : a > 0)
  (b_pos : b > 0)

def eccentricity (a b : ℝ) : ℝ :=
  (b * b + a * a).sqrt / a

-- Problem statement based on the conditions and correct answer
theorem hyperbola_eccentricity_sqrt2 (a b : ℝ) (h : Hyperbola a b) 
  (area_ratio : ∀ A B O F Q, (area_of_triangle A B O / area_of_triangle F Q O) = 1 / 2) :
  eccentricity a b = sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_sqrt2_l509_509509


namespace ring_groups_in_first_tree_l509_509134

variable (n : ℕ) (y1 y2 : ℕ) (t : ℕ) (groupsPerYear : ℕ := 6)

-- each tree's rings are in groups of 2 fat rings and 4 thin rings, representing 6 years
def group_represents_years : ℕ := groupsPerYear

-- second tree has 40 ring groups, so it is 40 * 6 = 240 years old
def second_tree_groups : ℕ := 40

-- first tree is 180 years older, so its age in years
def first_tree_age : ℕ := (second_tree_groups * groupsPerYear) + 180

-- number of ring groups in the first tree
def number_of_ring_groups_in_first_tree := first_tree_age / groupsPerYear

theorem ring_groups_in_first_tree :
  number_of_ring_groups_in_first_tree = 70 :=
by
  sorry

end ring_groups_in_first_tree_l509_509134


namespace identify_true_propositions_l509_509279

-- Definitions based on the conditions provided
variables (m n : Line) (α β : Plane)

-- Conditions
def parallel_lines (x y : Line) : Prop := x ∥ y
def perpendicular_line_plane (x : Line) (p : Plane) : Prop := x ⟂ p
def parallel_planes (p q : Plane) : Prop := p ∥ q
def line_in_plane (x : Line) (p : Plane) : Prop := x ⊆ p

-- Propositions
def prop_1 : Prop := parallel_lines m n ∧ perpendicular_line_plane m α → perpendicular_line_plane n α
def prop_2 : Prop := parallel_planes α β ∧ line_in_plane m α ∧ line_in_plane n β → parallel_lines m n
def prop_3 : Prop := parallel_lines m n ∧ parallel_planes m α → parallel_planes n α
def prop_4 : Prop := parallel_planes α β ∧ parallel_lines m n ∧ perpendicular_line_plane m α → perpendicular_line_plane n β

-- The theorem stating which propositions are true
theorem identify_true_propositions :
  prop_1 ∧ ¬prop_2 ∧ ¬prop_3 ∧ prop_4 :=
by
  sorry

end identify_true_propositions_l509_509279


namespace number_of_rectangles_l509_509901

-- Definition of the problem: We have 12 equally spaced points on a circle.
def points_on_circle : ℕ := 12

-- The number of diameters is half the number of points, as each diameter involves two points.
def diameters (n : ℕ) : ℕ := n / 2

-- The number of ways to choose 2 diameters out of n/2 is given by the binomial coefficient.
noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Prove the number of rectangles that can be formed is 15.
theorem number_of_rectangles :
  binomial_coefficient (diameters points_on_circle) 2 = 15 := by
  sorry

end number_of_rectangles_l509_509901


namespace area_of_triangle_COD_eq_5_abs_p_l509_509745

noncomputable def area_COD (p : ℝ) : ℝ :=
  let C := (0, p) in
  let D := (10, 10) in
  let O := (0, 0) in
  5 * |p|

theorem area_of_triangle_COD_eq_5_abs_p (p : ℝ) :
  area_COD p = 5 * |p| :=
by 
  obtain ⟨x_C, y_C⟩ := (0, p),
  obtain ⟨x_D, y_D⟩ := (10, 10),
  obtain ⟨x_O, y_O⟩ := (0, 0),
  -- Additional steps would normally follow
  sorry

end area_of_triangle_COD_eq_5_abs_p_l509_509745


namespace sufficient_condition_lines_perpendicular_l509_509506

def line1 (m : ℝ) : ℝ → ℝ → Prop := λ x y, (m + 1) * x + y - 2 = 0
def line2 (m : ℝ) : ℝ → ℝ → Prop := λ x y, m * x + (2 * m + 2) * y + 1 = 0

theorem sufficient_condition_lines_perpendicular (m : ℝ) : 
  m = -2 → 
  ∀ x1 y1 x2 y2, 
    line1 m x1 y1 → line2 m x2 y2 → 
    (x1 - x2) * (y1 - y2) = - (1 + m) * (-1 / (2 * m + 2)) := 
by
  sorry

end sufficient_condition_lines_perpendicular_l509_509506


namespace minimize_expected_value_expected_value_min_t_l509_509638

theorem minimize_expected_value (t : ℝ) (h : -1 ≤ t ∧ t ≤ 2) :
  E(X) = 0.2 * (t + 0.25) ^ 2 + 2.1875 :=
  sorry

def E(X) := 0.2 * (t + 0.25) ^ 2 + 2.1875

theorem expected_value_min_t (t : ℝ) (h : -1 ≤ t ∧ t ≤ 2) :
  t = -0.25 ↔ E(X) = 0 := 
  sorry

end minimize_expected_value_expected_value_min_t_l509_509638


namespace f_odd_and_invertible_l509_509781

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then log (1 - x)
  else if -1 < x ∧ x < 0 then log (1 / (1 + x))
  else 0 -- the definition to be adjusted based on the domain of f(x)

-- Prove that f is an odd function and has an inverse function
theorem f_odd_and_invertible :
  (∀ x, f (-x) = -f x) ∧ (∃ g, ∀ y, g (f y) = y) :=
by
  sorry

end f_odd_and_invertible_l509_509781


namespace mark_second_time_l509_509801

theorem mark_second_time (n : ℕ) (hn : n = 2021):
  ∃ b : ℕ, b = 67 ∧ (∀ i: ℕ, i < b → 
   (∑ j in range(i + 1), (j + 1) % n) % n ≠ 0) :=
by
  sorry

end mark_second_time_l509_509801


namespace Apollonius_circle_l509_509769

noncomputable def locus_of_M : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | let MA := real.sqrt ((p.1 + 3)^2 + p.2^2),
                     MB := real.sqrt ((p.1 - 3)^2 + p.2^2)
                 in MA / MB = 2 }

theorem Apollonius_circle :
  locus_of_M = { p : ℝ × ℝ | (p.1 - 5)^2 + p.2^2 = 16 } :=
by sorry

end Apollonius_circle_l509_509769


namespace simplify_expression_l509_509139

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 2) (h₂ : a ≠ -2) : 
  (2 * a / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2)) :=
by
  -- proof to be added
  sorry

end simplify_expression_l509_509139


namespace domain_of_y_is_minus2_to_2_l509_509674

variable {α : Type} [LinearOrder α]

def domain_f_x : Set α := {x | -2 ≤ x ∧ x ≤ 4}

def domain_y : Set α := {x | -2 ≤ x ∧ x ≤ 2}

theorem domain_of_y_is_minus2_to_2
  (f : α → α) (hf : ∀ x, x ∈ domain_f_x → f x = f x) :
  ∀ x, x ∈ domain_y ↔ (x ∈ domain_f_x ∧ -x ∈ domain_f_x) :=
sorry

end domain_of_y_is_minus2_to_2_l509_509674


namespace length_of_intervals_l509_509168

noncomputable def totalLength : ℝ :=
  let interval1 := Icc 0 (Real.pi / 4)
  let interval2 := Icc (Real.pi / 4) 2
  interval2.length - interval1.length

theorem length_of_intervals : 
  totalLength = 1.21 :=
by
  -- lengths of individual intervals
  let length1 := (Real.pi / 4 : ℝ)
  let length2 := (2 - Real.pi / 4 : ℝ)
  have : totalLength = length2 - length1, by sorry
  rw this
  -- compute length in decimal form and rounding to nearest hundredth
  have : length2 = 1.21460183660255 := by sorry
  have : length1 = 0.785398163397448 := by sorry
  have computation := length2 - length1
  -- rounding to the nearest hundredth
  have : (Real.toEuclidean length2) - (Real.toEuclidean length1) = 1.21460183660255 - 0.785398163397448 := by sorry
  have rounding := (Real.round (1.21460183660255 - 0.785398163397448)) = 1.21 :=
by sorry
  have exact_len := 1.21
  exact exact_len

end length_of_intervals_l509_509168


namespace reversed_digits_sum_l509_509719

theorem reversed_digits_sum (a b n : ℕ) (x y : ℕ) (ha : a < 10) (hb : b < 10) 
(hx : x = 10 * a + b) (hy : y = 10 * b + a) (hsq : x^2 + y^2 = n^2) : 
  x + y + n = 264 :=
sorry

end reversed_digits_sum_l509_509719


namespace shortest_chord_m_l509_509269

theorem shortest_chord_m (m : ℝ) :
  (∀ l C, l = (λ x y, mx + y - 2m - 1 = 0) ∧ C = (λ x y, x^2 + y^2 - 2x - 4y = 0) →
    for (P : ℝ × ℝ), P = (2,1) →
    for (center : ℝ × ℝ), center = (1,2) →
    (l ⊥ (line_through center P)) →
    shortest_chord_length l C center P →
    m = -1) :=
by
  sorry

end shortest_chord_m_l509_509269


namespace gcd_2015_15_l509_509070

theorem gcd_2015_15 : Nat.gcd 2015 15 = 5 :=
by
  have h1 : 2015 = 15 * 134 + 5 := by rfl
  have h2 : 15 = 5 * 3 := by rfl
  sorry

end gcd_2015_15_l509_509070


namespace problem1_problem2_problem3_l509_509554

theorem problem1 : (-3) - (-5) - 6 + (-4) = -8 := by sorry

theorem problem2 : ((1 / 9) + (1 / 6) - (1 / 2)) / (-1 / 18) = 4 := by sorry

theorem problem3 : -1^4 + abs (3 - 6) - 2 * (-2) ^ 2 = -6 := by sorry

end problem1_problem2_problem3_l509_509554


namespace number_of_windows_and_doors_l509_509429

structure Palace where
  rooms : ℕ
  gridSide : ℕ
  outerWalls : ℕ
  internalPartitions : ℕ

axiom palace_conditions : Palace :=
{
  rooms := 100,
  gridSide := 10,
  outerWalls := 4,
  internalPartitions := 18
}

theorem number_of_windows_and_doors (p : Palace) :
  (p.rooms = 100) → (p.gridSide = 10) → (p.outerWalls = 4) → (p.internalPartitions = 18) →
  let perimeter := p.gridSide * 4 in
  let windows := perimeter in
  let doors := p.internalPartitions * p.gridSide in
  windows = 40 ∧ doors = 180 :=
by 
  intro h_rooms h_gridSide h_outerWalls h_internalPartitions
  let perimeter := p.gridSide * 4
  let windows := perimeter
  let doors := p.internalPartitions * p.gridSide
  exact ⟨by sorry, by sorry⟩

end number_of_windows_and_doors_l509_509429


namespace simplify_expr_l509_509409

noncomputable def expr := 1 / (1 + 1 / (Real.sqrt 5 + 2))
noncomputable def simplified := (Real.sqrt 5 + 1) / 4

theorem simplify_expr : expr = simplified :=
by
  sorry

end simplify_expr_l509_509409


namespace service_years_range_l509_509790

theorem service_years_range (years : List ℕ) (h : years = [15, 10, 9, 17, 6, 3, 14, 16]) :
  List.maximum years - List.minimum years = 14 := by
  sorry

end service_years_range_l509_509790


namespace intersection_of_sets_l509_509731

theorem intersection_of_sets :
  let A := {x : ℝ | 2^x ≤ 4}
  let B := {x : ℝ | 1 < x}
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by {
  let A := {x : ℝ | 2^x ≤ 4},
  let B := {x : ℝ | 1 < x},
  sorry
}

end intersection_of_sets_l509_509731


namespace range_of_fx_on_interval_l509_509647

def f (x a b : ℝ) : ℝ := (1 + 2 * x) * (x ^ 2 + a * x + b)

theorem range_of_fx_on_interval (a b : ℝ) (h_symmetry : f 1 a b = 0 ∧ f (-1 / 2) a b = f (5 / 2) a b)
  : set.range (λ x, f x a b) ∩ set.Icc (-1 : ℝ) 1 =
      ([-7, (3 * real.sqrt 3 / 2)] : set ℝ) :=
sorry

end range_of_fx_on_interval_l509_509647


namespace relationship_between_m_and_n_l509_509664

variable (x : ℝ)

def m := x^2 + 2*x + 3
def n := 2

theorem relationship_between_m_and_n :
  m x ≥ n := by
  sorry

end relationship_between_m_and_n_l509_509664


namespace Alexandra_magazines_total_l509_509128

noncomputable def magazinesOnFriday := 18
noncomputable def magazinesOnSaturday := 25
noncomputable def magazinesOnSunday := 5 * magazinesOnFriday
noncomputable def magazinesOnMonday := 3 * magazinesOnSaturday
noncomputable def magazinesChewedUp := 10

theorem Alexandra_magazines_total :
  let totalPurchased := magazinesOnFriday + magazinesOnSaturday + magazinesOnSunday + magazinesOnMonday 
  in (totalPurchased - magazinesChewedUp) = 198 :=
by
  sorry

end Alexandra_magazines_total_l509_509128


namespace calc_square_difference_and_square_l509_509537

theorem calc_square_difference_and_square (a b : ℤ) (h1 : a = 7) (h2 : b = 3)
  (h3 : a^2 = 49) (h4 : b^2 = 9) : (a^2 - b^2)^2 = 1600 := by
  sorry

end calc_square_difference_and_square_l509_509537


namespace all_real_K_have_real_roots_l509_509155

noncomputable def quadratic_discriminant (K : ℝ) : ℝ :=
  let a := K ^ 3
  let b := -(4 * K ^ 3 + 1)
  let c := 3 * K ^ 3
  b ^ 2 - 4 * a * c

theorem all_real_K_have_real_roots : ∀ K : ℝ, quadratic_discriminant K ≥ 0 :=
by
  sorry

end all_real_K_have_real_roots_l509_509155


namespace number_of_integers_a_satisfies_conditions_l509_509449

theorem number_of_integers_a_satisfies_conditions : 
  let cond1 := ∀ x : ℝ, (3 * x - a)/(x - 3) + (x + 1)/(3 - x) = 1 → x > 0 → a = x + 2 ∧ x ≠ 3
      cond2 := ∀ y : ℝ, (y + 9 ≤ 2 * (y + 2)) ∧ ((2 * y - a) / 3 ≥ 1) → (y ≥ 5)
  in #[(a : ℤ) | a > 2 ∧ a ≠ 5 ∧ a ≤ 7].card = 4 := sorry

end number_of_integers_a_satisfies_conditions_l509_509449


namespace max_value_of_a_plus_b_l509_509495

theorem max_value_of_a_plus_b (a b : ℕ) (h1 : 7 * a + 19 * b = 213) (h2 : a > 0) (h3 : b > 0) : a + b = 27 :=
sorry

end max_value_of_a_plus_b_l509_509495


namespace probability_five_blue_marbles_is_correct_l509_509306

noncomputable def probability_of_five_blue_marbles : ℝ :=
let p_blue := (9 : ℝ) / 15
let p_red := (6 : ℝ) / 15
let specific_sequence_prob := p_blue ^ 5 * p_red ^ 3
let number_of_ways := (Nat.choose 8 5 : ℝ)
(number_of_ways * specific_sequence_prob)

theorem probability_five_blue_marbles_is_correct :
  probability_of_five_blue_marbles = 0.279 := by
sorry

end probability_five_blue_marbles_is_correct_l509_509306


namespace star_decagon_area_l509_509564

-- Define the regular star decagon problem
def star_decagon_uncovered_area (r : ℝ) : ℝ :=
  r^2 * (Real.pi - 5 * (Real.sqrt 5 + 1) / 2)

theorem star_decagon_area (r : ℝ) :
  let circle_area := Real.pi * r^2
  let star_decagon_area := 5 * (Real.sqrt 5 + 1) / 2 * r^2
  circle_area - star_decagon_area = star_decagon_uncovered_area r :=
by
  sorry

end star_decagon_area_l509_509564


namespace quadrilateral_incenter_distance_ratio_l509_509002

variables (A B C D O E F : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited O] [inhabited E] [inhabited F]

-- Define the relevant distances and properties
variables (d : ℝ)
variables (AO OC AB AD BC CD : ℝ)
variables (angle1 angle2 : ℝ)

-- Given conditions
axiom angle_condition1 : angle1 + angle2 = 2 * d
axiom angle_condition2 : ∃ (E F : Type), true
axiom angle_equality1 : angle1 = angle2

-- Triangle similarity and proportionality
axiom ratio_AF_FC : ∀ AE EC AD DC AB BC : ℝ, (AE / EC) = (AD / DC) ∧ (AF / FC) = (AB / BC)

-- Define the proof problem
theorem quadrilateral_incenter_distance_ratio : (AO^2 / OC^2) = (AB * AD / (BC * CD)) :=
by sorry

end quadrilateral_incenter_distance_ratio_l509_509002


namespace f_30_eq_1_l509_509171

noncomputable def primes_ge_5 := {p : ℕ | p ≥ 5 ∧ nat.prime p}

def f (y : ℕ) : ℕ :=
  (finset.powerset (finset.filter (λ p : ℕ, p ∈ primes_ge_5) (finset.range y + 1))).filter
    (λ s, s.sum id = y ∧ (∀ p ∈ s, p ∈ primes_ge_5) ∧ (∃ t, t.sort (≤) = s.val)).card

theorem f_30_eq_1 : f 30 = 1 :=
begin
  sorry
end

end f_30_eq_1_l509_509171


namespace money_spent_on_games_is_correct_l509_509287

-- Given conditions
def total_allowance : ℝ := 50
def fraction_on_books : ℝ := 1 / 4
def fraction_on_apps : ℝ := 3 / 10
def fraction_on_snacks : ℝ := 1 / 5
def fraction_on_games : ℝ := 1 - fraction_on_books - fraction_on_apps - fraction_on_snacks

-- Formal statement to prove
theorem money_spent_on_games_is_correct :
  fraction_on_games * total_allowance = 12.5 :=
by
  suffices h: 1 - fraction_on_books - fraction_on_apps - fraction_on_snacks = 1 - (1 / 4 + 3 / 10 + 1 / 5),
  {
    rw [fraction_on_games, h],
    norm_num,
  }
  sorry

end money_spent_on_games_is_correct_l509_509287


namespace shortest_path_on_surface_l509_509677

/-- Given a rectangular parallelepiped with edge lengths 3, 4, and 5, the shortest path
    along the surface from one endpoint of a space diagonal to the other is √125. -/
theorem shortest_path_on_surface (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  exists (shortest_path : ℝ), shortest_path = real.sqrt 125 :=
by
  use real.sqrt 125
  sorry

end shortest_path_on_surface_l509_509677


namespace tank_capacity_l509_509104

theorem tank_capacity 
  (trucks : ℕ) (tanks_per_truck : ℕ) (total_capacity : ℕ) 
  (h_trucks : trucks = 3) 
  (h_tanks_per_truck : tanks_per_truck = 3) 
  (h_total_capacity : total_capacity = 1350) 
  : ∃ (x : ℕ), trucks * tanks_per_truck * x = total_capacity ∧ x = 150 :=
by {
  use 150,
  rw [h_trucks, h_tanks_per_truck, h_total_capacity],
  exact ⟨rfl, rfl⟩,
}

end tank_capacity_l509_509104
